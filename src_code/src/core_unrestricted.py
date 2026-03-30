"""
python==3.11.15
CUDA Toolkit==11.8

numpy==2.1.3
scipy==1.14.1
opt_einsum==3.4.0
matplotlib==3.9.4
tqdm==4.67.1
h5py==3.12.1
pytest==8.4.2
pytorch==2.5.1
"""

import numpy as np
import scipy
import opt_einsum as oe
import torch
from scipy.sparse.linalg import aslinearoperator, svds


#################################
# TODO: GPU USE gesvd! -- Fisherman: https://journals.aps.org/prb/pdf/10.1103/PhysRevB.98.235148
#################################




# ── Precision control ─────────────────────────────────────────────────────────
# Default: float32 / complex64 — fastest on both CPU (MKL) and CUDA.
# Call  set_dtype(True)  BEFORE allocating any tensor to switch to
# float64 / complex128 (double precision).
#   CPU (Intel MKL): float64 is native → same throughput, 2× more memory.
#   CUDA: float64 is 2–4× slower (no fp64 tensor cores on consumer GPUs).
CDTYPE: torch.dtype = torch.complex64   # complex dtype for all tensors
RDTYPE: torch.dtype = torch.float32     # real dtype (SVD singular values, norms)

# Set to True to enable a runtime check inside trunc_rhoCCC that the three
# SV sums (sum(sv32[:chi]), sum(sv13[:chi]), sum(sv21[:chi])) are nearly equal.
# This is a physical consistency check; disable for production runs.
DEBUG_CHECK_TRUNC_RHOCCC_SV_SUMS: bool = False

# Set to True to enable truncation quality diagnostics: anti-Hermitian measures
# of the three rho matrices and their full SV spectra are buffered on every
# trunc_rhoCCC call and can be retrieved with get_trunc_diag_buffer().
# Disable for production runs (adds minor overhead per truncation call).
DEBUG_CHECK_TRUNCATION: bool = False

# ── Truncation diagnostics buffer ────────────────────────────────────────────
_trunc_diag_buffer: list[dict] = []




def mem_mb() -> float:
    """Return current process Resident Set Size (RSS) in MB.

    Uses psutil when available; falls back to /proc/self/status (Linux).
    Returns NaN when neither is available.
    """
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e6
    except ImportError:
        pass
    try:
        with open("/proc/self/status") as _fh:
            for _line in _fh:
                if _line.startswith("VmRSS:"):
                    return int(_line.split()[1]) / 1e3
    except OSError:
        pass
    return float('nan')





def safe_inverse(x, epsilon=1E-12):
    return x/(x**2 + epsilon**2)
    
def safe_inverse_2(x, epsilon):
    x[abs(x)<epsilon]=float('inf')
    return x.pow(-1)





def clear_trunc_diag_buffer() -> None:
    """Clear the module-level truncation diagnostics buffer."""
    global _trunc_diag_buffer
    _trunc_diag_buffer.clear()


def get_trunc_diag_buffer() -> list[dict]:
    """Return a shallow copy of the truncation diagnostics buffer.

    Each entry (one per trunc_rhoCCC call while DEBUG_CHECK_TRUNCATION is True)
    is a dict with keys:
        'anti_herm_rho32', 'anti_herm_rho13', 'anti_herm_rho21'  (float)
            Anti-Hermitian measure  ||M - M†|| / 2 / ||M||.
        'sv32', 'sv13', 'sv21'  (list[float])
            Full singular-value spectrum for each rho (may be long for large chi).
        'chi'  (int)  —  truncation cutoff index.
    """
    return list(_trunc_diag_buffer)


def set_check_truncation(enabled: bool) -> None:
    """Enable or disable truncation diagnostics buffering at runtime."""
    global DEBUG_CHECK_TRUNCATION
    DEBUG_CHECK_TRUNCATION = enabled


def set_dtype(use_double: bool) -> None:
    """Switch all core computations between float32/complex64 and float64/complex128.

    Must be called BEFORE any tensor is allocated — ideally right after import.

    Args:
        use_double: ``True``  → complex128 / float64 (double precision).
                    ``False`` → complex64  / float32 (single precision, default).
    """
    global CDTYPE, RDTYPE
    if use_double:
        CDTYPE = torch.complex128
        RDTYPE = torch.float64
    else:
        CDTYPE = torch.complex64
        RDTYPE = torch.float32


# for now only the case of ABCDEF, nearest neighbor, exact SVD, D^4 >= chi > D^2 .

def normalize_tensor(tensor, *, rtol: float | None = None, atol: float | None = None):
    
    """
    Normalise a tensor in-place-style by its Frobenius (L2) norm.

    Divides every element by ``||tensor||_F``.  If the norm is exactly zero
    (e.g. an all-zero tensor) the input is returned unchanged to avoid a
    division-by-zero NaN.

    Args:
        tensor (torch.Tensor): Any complex or real tensor of arbitrary shape.

    Returns:
        torch.Tensor: A new tensor with the same shape and dtype as the input
        whose Frobenius norm is 1, or the original tensor if its norm is 0.
    """
    # Frobenius norm (for arbitrary-rank tensors). Returns a real scalar.
    norm = torch.linalg.norm(tensor)

    # Choose dtype-appropriate default tolerances.
    # For complex64 (norm is float32), norms around 1 typically fluctuate at ~1e-7,
    # so a strict 1e-12 check would *keep renormalizing* and introduce drift.
    if rtol is None or atol is None:
        if norm.dtype == torch.float32:
            default_rtol = 4e-7
            default_atol = 4e-7
        else:
            default_rtol = 1e-13
            default_atol = 1e-13
        rtol = default_rtol if rtol is None else rtol
        atol = default_atol if atol is None else atol

    # Guard against NaNs/Infs.
    if not torch.isfinite(norm):
        return tensor

    # Avoid division-by-zero for (near) all-zero tensors.
    if norm <= atol:
        return tensor

    # Make repeated calls effectively idempotent:
    # if already normalized within tolerance, return unchanged.
    one = torch.ones((), dtype=norm.dtype, device=norm.device)
    if torch.isclose(norm, one, rtol=rtol, atol=atol):
        return tensor

    return tensor / norm


def normalize_single_layer_tensor_for_double_layer(
    tensor: torch.Tensor,
    *,
    rtol: float | None = None,
    atol: float | None = None,
) -> torch.Tensor:
    """Rescale a single-layer site tensor so its *unnormalized* double-layer has unit Frobenius norm.

    In this codebase, CTMRG consumes double-layer site tensors (A..F) produced by
    ``abcdef_to_ABCDEF`` which *then* normalizes each A..F to unit Frobenius norm.
    If energy/contraction routines use the original single-layer tensors directly,
    a convention mismatch can make denominators like <iPEPS|iPEPS> far from 1 even
    when the CTMRG environment itself is well-normalized.

    This helper chooses a single-layer rescaling such that the corresponding
    unnormalized double-layer tensor already has ||DL||_F ≈ 1, making single-layer
    contractions consistent with the CTMRG convention.

    Scaling rule:
      DL(t) = contract(t, conj(t)) so ||DL(α t)||_F = |α|^2 ||DL(t)||_F.
      Choose α = 1/sqrt(||DL(t)||_F) ⇒ ||DL(α t)||_F = 1.
    """
    # Compute the unnormalized double-layer tensor DL(t) (rank-6), then fuse
    # the three (D,D) virtual index pairs into (D^2,D^2,D^2) like abcdef_to_ABCDEF.
    dl = oe.contract(
        "uvwp,xyzp->uxvywz",
        tensor,
        tensor.conj(),
        optimize=[(0, 1)],
        backend="torch",
    )
    D_bond = int(tensor.shape[0])
    D_squared = D_bond * D_bond
    dl = dl.reshape(D_squared, D_squared, D_squared)

    dl_norm = torch.linalg.norm(dl)

    # Choose dtype-appropriate default tolerances.
    if rtol is None or atol is None:
        if dl_norm.dtype == torch.float32:
            default_rtol = 4e-7
            default_atol = 4e-7
        else:
            default_rtol = 1e-13
            default_atol = 1e-13
        rtol = default_rtol if rtol is None else rtol
        atol = default_atol if atol is None else atol

    if not torch.isfinite(dl_norm):
        return tensor
    if dl_norm <= atol:
        return tensor

    one = torch.ones((), dtype=dl_norm.dtype, device=dl_norm.device)
    if torch.isclose(dl_norm, one, rtol=rtol, atol=atol):
        return tensor

    return tensor / torch.sqrt(dl_norm)








def _keep_multiplets(U,S,V,chi,eps_multiplet,abs_tol):
    # estimate the chi_new 
    chi_new= chi
    # regularize by discarding small values
    gaps=S[:chi+1].clone().detach()
    # S[S < abs_tol]= 0.
    gaps[gaps < abs_tol]= 0.
    # compute gaps and normalize by larger sing. value. Introduce cutoff
    # for handling vanishing values set to exact zero
    gaps=(gaps[:chi]-S[1:chi+1])/(gaps[:chi]+1.0e-16)
    gaps[gaps > 1.0]= 0.

    if gaps[chi-1] < eps_multiplet:
        # the chi is within the multiplet - find the largest chi_new < chi
        # such that the complete multiplets are preserved
        for i in range(chi-1,-1,-1):
            if gaps[i] > eps_multiplet:
                chi_new= i
                break

    St = S[:chi].clone()
    St[chi_new+1:]=0.

    Ut = U[:, :St.shape[0]].clone()
    Ut[:, chi_new+1:]=0.
    Vt = V[:, :St.shape[0]].clone()
    Vt[:, chi_new+1:]=0.
    return Ut, St, Vt
    


class SVD_PROPACK(torch.autograd.Function):
    @staticmethod
    def forward(self, M, k, k_extra, rel_cutoff, v0):
        r"""
        :param M: square matrix :math:`N \times N`
        :param k: desired rank (must be smaller than :math:`N`)
        :type M: torch.Tensor
        :type k: int
        :return: leading k left eigenvectors U, singular values S, and right 
                 eigenvectors V
        :rtype: torch.Tensor, torch.Tensor, torch.Tensor

        **Note:** `depends on scipy`

        Return leading k-singular triples of a matrix M, by computing 
        the partial SVD decomposition using SciPy's PROPACK wrapper up to rank k.
        """
        # Dense LAPACK SVD via numpy — identical numerical backend to the
        # original torch.linalg.svd calls this replaced.
        # ARPACK and PROPACK are NOT used: ARPACK silently falls back to
        # scipy.linalg.eig (wrong algorithm) when k >= N-1, and PROPACK's
        # Fortran ZLASCL routine crashes on ill-conditioned matrices.
        # The CTMRG matrices are at most ~100×100; dense SVD is fast.
        _np_dtype = np.complex128 if M.is_complex() else np.float64
        _rdtype = torch.float64 if M.dtype in (torch.complex128, torch.float64) else torch.float32

        m_dim, n_dim = M.shape
        k_total = min(k + k_extra, min(m_dim, n_dim))
        k_total = max(k_total, 1)

        # print(f"Performing SVD with k={k}, k_extra={k_extra}, rel_cutoff={rel_cutoff.item()}, m_dim={m_dim}, n_dim={n_dim}, k_total={k_total}")


        if True:
            # M.detach() so .numpy() works even when M.requires_grad=True
            M_np = M.detach().cpu().numpy().astype(_np_dtype, copy=False)

            # Full thin SVD via LAPACK (numpy uses gesdd by default).
            # np.linalg.svd always returns S in descending order.
            # We compute ALL min(m,n) singular triples here even though we
            # only RETURN k_total of them.  The backward uses ALL singular
            # triples for the F,G cross-coupling matrices; using only k_total
            # would drop the interaction terms between kept and discarded
            # singular values, causing ~38% gradient error for typical CTMRG
            # projector losses (equivalent to the "5th term" correction of
            # arXiv:2311.11894v3 — verified by test_svd_trunc_term5.py).
            _Uf, _Sf, _Vhf = np.linalg.svd(M_np, full_matrices=False)

            k_full = min(m_dim, n_dim)  # = _Sf.shape[0]

            # Convert FULL SVD to tensors for backward
            S_full = torch.as_tensor(_Sf.copy(),  dtype=_rdtype, device=M.device)
            U_full = torch.as_tensor(_Uf.copy(),  dtype=M.dtype,  device=M.device)
            V_full = torch.as_tensor(_Vhf.copy(), dtype=M.dtype,  device=M.device).transpose(-2,-1).conj()

            # Save full-rank tensors so backward has all cross-coupling information
            self.save_for_backward(U_full, S_full, V_full, rel_cutoff)
            self.k_return = k_total   # only k_total triples are returned; backward pads gu/gv/gs

            # Return only the TOP k_total singular triples to the caller.
            # The backward already handles the zero-padding of upstream grads.
            return U_full[:, :k_total], S_full[:k_total], V_full[:, :k_total]
            # return U_full, S_full, V_full
        
        else:
            _uPartial, _sPartial, _vhPartial = svds(aslinearoperator(M.detach().cpu().numpy()), k=k_total, tol=0, which='LM', v0=v0.cpu().numpy() if v0 is not None else None, solver='arpack')
            S_partial = torch.as_tensor(_sPartial.copy(), dtype=_rdtype, device=M.device)
            U_partial = torch.as_tensor(_uPartial.copy(), dtype=M.dtype, device=M.device)
            V_partial = torch.as_tensor(_vhPartial.copy(), dtype=M.dtype, device=M.device).transpose(-2,-1).conj()

            self.save_for_backward(U_partial, S_partial, V_partial, rel_cutoff)
            self.k_return = k_total   # how many singular triples were returned

            return U_partial, S_partial, V_partial

    @staticmethod
    def backward(self, gu, gsigma, gv):
        r"""
        :param gu: gradient on U
        :type gu: torch.Tensor
        :param gsigma: gradient on S
        :type gsigma: torch.Tensor
        :param gv: gradient on V
        :type gv: torch.Tensor
        :return: gradient
        :rtype: torch.Tensor

        Computes backward gradient for SVD, adopted from 
        https://github.com/pytorch/pytorch/blob/v1.10.2/torch/csrc/autograd/FunctionsManual.cpp
        
        For complex-valued input there is an additional term, see

            * https://giggleliu.github.io/2019/04/02/einsumbp.html
            * https://arxiv.org/abs/1909.02659

        The backward is regularized following
        
            * https://github.com/wangleiphy/tensorgrad/blob/master/tensornets/adlib/svd.py
            * https://arxiv.org/abs/1903.09650

        using 

        .. math:: 
            S_i/(S^2_i-S^2_j) = (F_{ij}+G_{ij})/2\ \ \textrm{and}\ \ S_j/(S^2_i-S^2_j) = (F_{ij}-G_{ij})/2
        
        where 
        
        .. math:: 
            F_{ij}=1/(S_i-S_j),\ G_{ij}=1/(S_i+S_j)
        """
        # 
        # TORCH_CHECK(compute_uv,
        #    "svd_backward: Setting compute_uv to false in torch.svd doesn't compute singular matrices, ",
        #    "and hence we cannot compute backward. Please use torch.svd(compute_uv=True)");
        u, sigma, v, eps = self.saved_tensors
        m= u.size(0) # first dim of original tensor A = u sigma v^\dag 
        n= v.size(0) # second dim of A
        k= sigma.size(0)   # = min(m, n) — FULL rank (saved in forward)
        sigma_scale= sigma[0]

        # Forward returned only k_return singular triples; gu and gv have width
        # k_return ≤ k.  Pad them with zeros so the F,G cross-coupling between
        # kept and discarded singular values is captured correctly.
        k_return = getattr(self, 'k_return', k)  # fallback to k for safety
        if k_return < k:
            _z = lambda g, cols: torch.cat(
                [g, g.new_zeros(*g.shape[:-1], cols)], dim=-1)
            if gu is not None:
                gu = _z(gu, k - k_return)
            if gv is not None:
                gv = _z(gv, k - k_return)
            if gsigma is not None:
                gsigma = torch.cat(
                    [gsigma, gsigma.new_zeros(k - k_return)], dim=-1)

        # u, sigma, v are now full-rank (min(m,n) columns) so the
        # "free subspace" narrowing below will normally be a no-op.
        if (u.size(-2)!=u.size(-1)) or (v.size(-2)!=v.size(-1)):
            # We ignore the free subspace here because possible base vectors cancel
            # each other, e.g., both -v and +v are valid base for a dimension.
            # Don't assume behavior of any particular implementation of svd.
            u = u.narrow(-1, 0, k)
            v = v.narrow(-1, 0, k)
            if not (gu is None): gu = gu.narrow(-1, 0, k)
            if not (gv is None): gv = gv.narrow(-1, 0, k)
        vh= v.conj().transpose(-2,-1)

        if not (gsigma is None):
            # computes u @ diag(gsigma) @ vh
            sigma_term = u * gsigma.unsqueeze(-2) @ vh
        else:
            sigma_term = torch.zeros(m,n,dtype=u.dtype,device=u.device)
        # in case that there are no gu and gv, we can avoid the series of kernel
        # calls below
        if (gu is None) and (gv is None):
            return sigma_term, None, None

        # eps was stored with real dtype (fixed in truncated_svd_propack), but
        # defend against any legacy path: extract .real so comparisons in
        # safe_inverse_2 are never attempted on a complex scalar.
        _eps_real = eps.real if eps.is_complex() else eps
        sigma_inv= safe_inverse_2(sigma.clone(), sigma_scale*_eps_real)

        F = sigma.unsqueeze(-2) - sigma.unsqueeze(-1)
        F = safe_inverse(F, sigma_scale*_eps_real)
        F.diagonal(0,-2,-1).fill_(0)

        G = sigma.unsqueeze(-2) + sigma.unsqueeze(-1)
        G = safe_inverse(G, sigma_scale*_eps_real)
        G.diagonal(0,-2,-1).fill_(0)

        uh= u.conj().transpose(-2,-1)
        if not (gu is None):
            guh = gu.conj().transpose(-2, -1);
            u_term = u @ ( (F+G).mul( uh @ gu - guh @ u) ) * 0.5
            if m > k:
                # projection operator onto subspace orthogonal to span(U) defined as I - UU^H
                proj_on_ortho_u = -u @ uh
                proj_on_ortho_u.diagonal(0, -2, -1).add_(1);
                u_term = u_term + proj_on_ortho_u @ (gu * sigma_inv.unsqueeze(-2)) 
            u_term = u_term @ vh
        else:
            u_term = torch.zeros(m,n,dtype=u.dtype,device=u.device)
        
        if not (gv is None):
            gvh = gv.conj().transpose(-2, -1);
            v_term = ( (F-G).mul(vh @ gv - gvh @ v) ) @ vh * 0.5
            if n > k:
                # projection operator onto subspace orthogonal to span(V) defined as I - VV^H
                proj_on_v_ortho =  -v @ vh
                proj_on_v_ortho.diagonal(0, -2, -1).add_(1);
                v_term = v_term + sigma_inv.unsqueeze(-1) * (gvh @ proj_on_v_ortho)
            v_term = u @ v_term
        else:
            v_term = torch.zeros(m,n,dtype=u.dtype,device=u.device)
        

        # // for complex-valued input there is an additional term
        # // https://giggleliu.github.io/2019/04/02/einsumbp.html
        # // https://arxiv.org/abs/1909.02659
        dA= u_term + sigma_term + v_term
        if u.is_complex() or v.is_complex():
            if gu is not None:
                L= (uh @ gu).diagonal(0,-2,-1).clone()
                L.real.zero_()
                L.imag.mul_(sigma_inv)
                imag_term= (u * L.unsqueeze(-2)) @ vh
                dA= dA + imag_term

        return dA, None, None, None, None




def truncated_svd_propack(M, chi, chi_extra, rel_cutoff, v0=None,
    abs_tol=1.0e-14,  keep_multiplets=False, \
    eps_multiplet=1.0e-12, ):
    r"""
    :param M: square matrix of dimensions :math:`N \times N`
    :param chi: desired maximal rank :math:`\chi`
    :param abs_tol: absolute tolerance on minimal singular value 
    :param rel_tol: relative tolerance on minimal singular value
    :param keep_multiplets: truncate spectrum down to last complete multiplet
    :param eps_multiplet: allowed splitting within multiplet
    :param verbosity: logging verbosity
    :type M: torch.tensor
    :type chi: int
    :type abs_tol: float
    :type rel_tol: float
    :type keep_multiplets: bool
    :type eps_multiplet: float
    :type verbosity: int
    :return: leading :math:`\chi` left singular vectors U, right singular vectors V, and
             singular values S
    :rtype: torch.tensor, torch.tensor, torch.tensor

    **Note:** `depends on scipy`

    Returns leading :math:`\chi`-singular triples of a matrix M,
    by computing the partial symmetric decomposition of :math:`H=M^TM` as :math:`H= UDU^T` 
    up to rank :math:`\chi`. Returned tensors have dimensions 

    .. math:: dim(U)=(N,\chi),\ dim(S)=(\chi,\chi),\ \textrm{and}\ dim(V)=(N,\chi)

    .. note::
        This function does not support autograd.
    """
    # rel_cutoff must always be real-typed: singular values are real, and
    # safe_inverse / safe_inverse_2 use < comparisons that are undefined for
    # complex tensors (raises "lt_cpu not implemented for 'ComplexDouble'").
    _rdtype = torch.float64 if M.dtype in (torch.complex128, torch.float64) else torch.float32
    U, S, V = SVD_PROPACK.apply(M, chi, max(chi_extra,1),torch.as_tensor([rel_cutoff],dtype=_rdtype, device=M.device), v0)

    # estimate the chi_new 
    if keep_multiplets and chi<S.shape[0]:
        return _keep_multiplets(U,S,V,chi,eps_multiplet,abs_tol)

    St = S[:min(chi,S.shape[0])]
    Ut = U[:, :St.shape[0]]
    Vt = V[:, :St.shape[0]]

    return Ut, St, Vt

"""
use of truncate_svd_propack:

truncated_svd_propack(M, chi,
                    chi_extra=1,
                    rel_cutoff=1e-12,
                    v0=None,
                    keep_multiplets=False,
                    abs_tol=1e-14,
                    eps_multiplet=1e-12)  # returns U, S, V of M= USV^\dag

"""










def initialize_abcdef(initialize_way:str, D_bond:int, d_PHYS:int, noise_scale:float):
    """
    Initialise the six single-layer iPEPS site tensors a, b, c, d, e, f.

    Each tensor has shape ``(D_bond, D_bond, D_bond, d_PHYS)`` where the first
    three indices are virtual bond indices (one per incoming edge of the
    honeycomb-like unit cell) and the last is the physical index.
    All tensors are normalised to unit Frobenius norm after creation.

    Args:
        initialize_way (str): Initialisation strategy.
            ``'random'``  — independent complex-Gaussian random tensors.
            ``'product'`` — product-state initialisation (not yet implemented).
            ``'singlet'`` — Mz=0 singlet initialisation (not yet implemented).
        D_bond (int): Virtual bond dimension D.
        d_PHYS (int): Physical Hilbert-space dimension.
        noise_scale (float): Amplitude of added noise (used by non-random
            initialisations; currently unused for ``'random'``).

    Returns:
        Tuple[torch.Tensor, ...]: ``(a, b, c, d, e, f)``, each of shape
        ``(D_bond, D_bond, D_bond, d_PHYS)``, dtype ``torch.complex64``.

    Raises:
        ValueError: If ``initialize_way`` is not a recognised strategy.
    """

    if initialize_way == 'random' :

        a = torch.randn(D_bond, D_bond, D_bond, d_PHYS, dtype=CDTYPE)
        b = torch.randn(D_bond, D_bond, D_bond, d_PHYS, dtype=CDTYPE)
        c = torch.randn(D_bond, D_bond, D_bond, d_PHYS, dtype=CDTYPE)
        d = torch.randn(D_bond, D_bond, D_bond, d_PHYS, dtype=CDTYPE)
        e = torch.randn(D_bond, D_bond, D_bond, d_PHYS, dtype=CDTYPE)
        f = torch.randn(D_bond, D_bond, D_bond, d_PHYS, dtype=CDTYPE)

    elif initialize_way == 'product' : # product state but always with small noise

        raise NotImplementedError("'product' initialisation is not yet implemented")

    elif initialize_way == 'singlet' : # Mz=0 sector's singlet state representable as PEPS

        raise NotImplementedError("'singlet' initialisation is not yet implemented")

    else :

        raise ValueError(f"Invalid initialize_way: {initialize_way}")
    
    return a,b,c,d,e,f




def abcdef_to_ABCDEF(a,b,c,d,e,f, D_squared:int):
    """
    Build the double-layer ("bra⊗ket") site tensors A, B, C, D, E, F.

    Each double-layer tensor is obtained by contracting a single-layer tensor
    with its complex conjugate over the physical index::

        A_{(u,x),(v,y),(w,z)} = sum_φ  a_{u,v,w,φ} * conj(a_{x,y,z,φ})

    then reshaping the fused virtual indices ``(u,x)``, ``(v,y)``, ``(w,z)``
    into a ``(D_squared, D_squared, D_squared)`` tensor and finally into a
    ``(D_squared, D_squared)`` matrix (two legs fused into one on each side).
    Each output tensor is normalised to unit Frobenius norm.

    Args:
        a, b, c, d, e, f (torch.Tensor): Single-layer site tensors, each of
            shape ``(D_bond, D_bond, D_bond, d_PHYS)``.
        D_squared (int): ``D_bond ** 2``, the bond dimension of the
            double-layer tensors.

    Returns:
        Tuple[torch.Tensor, ...]: ``(A, B, C, D, E, F)``, each of shape
        ``(D_squared, D_squared, D_squared)`` (rank-3, one fused virtual-index
        pair per honeycomb leg), dtype ``torch.complex64``.
    """

    A = oe.contract("uvwp,xyzp->uxvywz", a, a.conj(), optimize=[(0, 1)], backend='torch')
    A = A.reshape(D_squared, D_squared, D_squared)
    # A = normalize_tensor(A)

    B = oe.contract("uvwp,xyzp->uxvywz", b, b.conj(), optimize=[(0, 1)], backend='torch')
    B = B.reshape(D_squared, D_squared, D_squared)
    # B = normalize_tensor(B)

    C = oe.contract("uvwp,xyzp->uxvywz", c, c.conj(), optimize=[(0, 1)], backend='torch')
    C = C.reshape(D_squared, D_squared, D_squared)
    # C = normalize_tensor(C)

    D = oe.contract("uvwp,xyzp->uxvywz", d, d.conj(), optimize=[(0, 1)], backend='torch')
    D = D.reshape(D_squared, D_squared, D_squared)
    # D = normalize_tensor(D)

    E = oe.contract("uvwp,xyzp->uxvywz", e, e.conj(), optimize=[(0, 1)], backend='torch')
    E = E.reshape(D_squared, D_squared, D_squared)
    # E = normalize_tensor(E)

    F = oe.contract("uvwp,xyzp->uxvywz", f, f.conj(), optimize=[(0, 1)], backend='torch')
    F = F.reshape(D_squared, D_squared, D_squared)
    # F = normalize_tensor(F)

    return A,B,C,D,E,F




# All the deprecated function are not used at all!!!!!!!! 
# No svd_fixed! No trunc_rhoCCC_deprecated!!!!! 
# Stops trying to call these tested-to-be-even-more-unstable
# functions than the bare torch.linalg.svd





def trunc_rhoCCC(matC21, matC32, matC13, chi, D_squared):

    #Q1, R1 = torch.linalg.qr(matC21.T)
    #Q2, R2 = torch.linalg.qr(torch.mm(matC13,matC32))
    R1 = matC21.T
    R2 = torch.mm(matC13,matC32)
    
    #U, S, Vh = torch.linalg.svd(torch.mm(R1,R2.T))
    #truncUh = U[:,:chi].conj().T
    #truncV = Vh[:chi,:].conj().T



    U,S,V = truncated_svd_propack(torch.mm(R1,R2.T), chi,
                        chi_extra=1,
                        rel_cutoff=1e-12,
                        v0=None,
                        keep_multiplets=False,
                        abs_tol=1e-14,
                        eps_multiplet=1e-12)
    
    truncV = V
    truncUh = U.conj().T

    

    #approx_product =  U[:,:chi] @ torch.diag(S[:chi]).to(CDTYPE) @ Vh[:chi,:]
    #exact_product = R1 @ R2.T
    #product_error = torch.linalg.norm(approx_product - exact_product) / torch.linalg.norm(exact_product)
    #VERIFIED print(f"R1 @ R2.T vs truncated U @ diag(S) @ Vh check: relative error = {product_error:.2e}")

    sqrtInvTruncS = torch.diag(1.0 / torch.sqrt(S[:chi]).to(CDTYPE))
    #VERIFIED print((S[chi]/S[0]).item())
    #VERIFIED print(torch.diag(sqrtInvTruncS @ sqrtInvTruncS @ torch.diag(S[:chi]).to(CDTYPE)).detach().cpu().numpy())

    #product_inverse = Vh.conj().T @ torch.diag(1.0 / torch.sqrt(S).to(CDTYPE)) @ torch.diag(1.0 / torch.sqrt(S).to(CDTYPE)) @ U.conj().T
    #exact_inverse = torch.linalg.inv(R1 @ R2.T)
    #inverse_error = torch.linalg.norm(product_inverse - exact_inverse) / torch.linalg.norm(exact_inverse)
    #VERIFIED print(f"(R1 @ R2.T)^-1 vs Vh @ diag(1/sqrt(S)) @ diag(1/sqrt(S)) @ U check: relative error = {inverse_error:.2e}")

    #U_inv = torch.linalg.inv(U)
    #U_dag = U.conj().T
    #Vh_inv = torch.linalg.inv(Vh)
    #Vh_dag = Vh.conj().T
    #print_error_U = torch.linalg.norm(U_inv - U_dag) / torch.linalg.norm(U_dag)
    #print_error_Vh = torch.linalg.norm(Vh_inv - Vh_dag) / torch.linalg.norm(Vh_dag)
    #VERIFIED print(f"U-1 vs U† check: relative error = {print_error_U:.4e}")
    #VERIFIED print(f"V†-1 vs V check: relative error = {print_error_Vh:.4e}")

    # print(torch.norm(approx_product @ truncV @ sqrtInvTruncS @ sqrtInvTruncS @ truncUh * 2 - torch.eye(chi*D_squared, device=approx_product.device, dtype=approx_product.dtype)).item())
    # print((approx_product @ truncV @ sqrtInvTruncS @ sqrtInvTruncS @ truncUh).detach().cpu().numpy())

    #approx_produc_inverse = truncV @ sqrtInvTruncS @ sqrtInvTruncS @ truncUh 
    #produc_inverse = torch.linalg.inv(R1 @ R2.T)

########################################
    # TODO: BECAUSE THE C is normalized s.t. the determinant can be very small!
########################################

    #print("detR1=", torch.linalg.det(R1).item(), "   detR2=", torch.linalg.det(R2).item())
    #inverse_error = torch.linalg.norm(approx_produc_inverse - produc_inverse) #/ torch.linalg.norm(produc_inverse)
    #print(f"approx_produc_inverse vs exact inverse check: relative error = {inverse_error:.4e}")
################################
    # TODO: But still, the error for R2 V 1/S Uh R1 is much larger than V 1/S Uh R1 R2
    # TODO: QR (SVD-1)R1R2 >~= no-QR (SVD-1)R1R2 > QR R2(SVD-1)R1 ~= no-QR R2(SVD-1)R1
###############################


    P21My = torch.mm( R2.T ,torch.mm( truncV , sqrtInvTruncS ))
    P32yM = torch.mm(torch.mm( sqrtInvTruncS , truncUh ), R1 )

    #almost_identity = torch.mm(P21My,P32yM)
    #average_diag = torch.diag(almost_identity).mean().item()
    #standard_error_diag = torch.sqrt(torch.var(torch.diag(almost_identity))).item()
    #print(f"P21My @ P32yM ≈ I check: average diag = {average_diag:.4e}, standard error diag = {standard_error_diag:.4e}")
    
    #corresponding_identity = torch.eye(chi*D_squared, device=almost_identity.device, dtype=almost_identity.dtype)
    #projected_difference =  truncUh @ (almost_identity - corresponding_identity) @ truncV
    #identity_error = torch.linalg.norm(projected_difference)/torch.linalg.norm(corresponding_identity[:chi,:chi])
    # print(f"P21My @ P32yM ≈ I check: relative error = {identity_error:.4e}")

    #almost_identity_2 = truncV @ sqrtInvTruncS @ sqrtInvTruncS @ truncUh @ R1 @ R2.T
    #only_display_1_digit_almost_identity_2 = torch.round(torch.real(almost_identity_2) * 10) / 10 + 1j * torch.round(torch.imag(almost_identity_2) * 10) / 10
    #print(only_display_1_digit_almost_identity_2.detach().cpu().numpy())



    #Q1, R1 = torch.linalg.qr(matC32.T)
    #Q2, R2 = torch.linalg.qr(torch.mm(matC21,matC13))
    R1 = matC32.T
    R2 = torch.mm(matC21,matC13)
    #U, S, Vh = torch.linalg.svd(torch.mm(R1,R2.T))
    U, S, V = truncated_svd_propack(torch.mm(R1,R2.T), chi,
                    chi_extra=1,
                    rel_cutoff=1e-12,
                    v0=None,
                    keep_multiplets=False,
                    abs_tol=1e-14,
                    eps_multiplet=1e-12)
    
    truncUh = U.conj().T
    truncV = V

    #truncUh = U[:,:chi].conj().T
    #truncV = Vh[:chi,:].conj().T
    sqrtInvTruncS = torch.diag(1.0 / torch.sqrt(S[:chi]).to(CDTYPE))

    P32Nz = torch.mm( R2.T ,torch.mm( truncV , sqrtInvTruncS ))
    P13zN = torch.mm(torch.mm( sqrtInvTruncS , truncUh ), R1 )




    #Q1, R1 = torch.linalg.qr(matC13.T)
    #Q2, R2 = torch.linalg.qr(torch.mm(matC32,matC21))
    R1 = matC13.T
    R2 = torch.mm(matC32,matC21)
    #U, S, Vh = torch.linalg.svd(torch.mm(R1,R2.T))
    U, S, V = truncated_svd_propack(torch.mm(R1,R2.T), chi,
                    chi_extra=1,
                    rel_cutoff=1e-12,
                    v0=None,
                    keep_multiplets=False,
                    abs_tol=1e-14,
                    eps_multiplet=1e-12)
    #print("1st 'QR'-cut done")

    truncUh = U.conj().T
    truncV = V

    #truncUh = U[:,:chi].conj().T
    #truncV = Vh[:chi,:].conj().T
    sqrtInvTruncS = torch.diag(1.0 / torch.sqrt(S[:chi]).to(CDTYPE))

    P13Lx = torch.mm( R2.T ,torch.mm( truncV , sqrtInvTruncS ))
    P21xL = torch.mm(torch.mm( sqrtInvTruncS , truncUh ), R1 )



    C21 = torch.mm(P21My.T, torch.mm(matC21, P21xL.T))
    C32 = torch.mm(P32Nz.T, torch.mm(matC32, P32yM.T))
    C13 = torch.mm(P13Lx.T, torch.mm(matC13, P13zN.T))
    
    
    C21 = C21/torch.linalg.norm(C21)
    C32 = C32/torch.linalg.norm(C32)
    C13 = C13/torch.linalg.norm(C13)
    

    
    rho_trunc = C13 @ C32 @ C21
    tr_rho = torch.trace(rho_trunc)
    #print(f"Tr(rho_trunc) before normalization: {tr_rho.item():.6e}")
    toBeDivided = torch.pow(tr_rho, 1.0/3.0)
    C21 = C21 / toBeDivided
    C32 = C32 / toBeDivided
    C13 = C13 / toBeDivided


    """
    tr_abs = tr_rho.abs()

    if torch.isfinite(tr_abs).item() and (tr_abs > torch.finfo(tr_abs.dtype).tiny).item():
        phase = tr_rho / tr_abs
        C21 = C21 / phase

        scaleC = tr_abs.pow(1.0 / 3.0)
        if torch.isfinite(scaleC).item() and (scaleC > torch.finfo(scaleC.dtype).tiny).item():
            C21 = C21 / scaleC
            C32 = C32 / scaleC
            C13 = C13 / scaleC

    # ── Individual equalization: preserve |Tr(rho)|=1 while balancing magnitudes ──
    # Scale factors multiply to 1  ⟹  |Tr(rho)| = |Tr(C13·C32·C21)| unchanged.
    n21 = torch.linalg.norm(C21).real
    n32 = torch.linalg.norm(C32).real
    n13 = torch.linalg.norm(C13).real
    nC_prod = n21 * n32 * n13
    if torch.isfinite(nC_prod).item() and (nC_prod > torch.finfo(n21.dtype).tiny).item():
        nC_geo = nC_prod.pow(1.0 / 3.0)
        C21 = C21 * (nC_geo / n21)
        C32 = C32 * (nC_geo / n32)
        C13 = C13 * (nC_geo / n13)
    """


    U1dag = P21My.reshape(chi,D_squared, chi)
    U2dag = P32Nz.reshape(chi,D_squared, chi)
    U3dag = P13Lx.reshape(chi,D_squared, chi)

    V1 = P13zN.reshape(chi, chi,D_squared)
    V2 = P21xL.reshape(chi, chi,D_squared)
    V3 = P32yM.reshape(chi, chi,D_squared)

    return V2, C21, U1dag, V3, C32, U2dag, V1, C13, U3dag


def initialize_envCTs_1(A,B,C,D,E,F, chi, D_squared, identity_init=False):
    """
    Initialise the 9 CTMRG corner matrices for the first CTMRG cycle.

    The first cycle is a special case because we have no previous environment to
    truncate from.  We therefore build each corner by contracting together the
    three double-layer site tensors that meet at that corner, then optionally
    add a small identity component to ensure full rank before truncation.

    Args:
        A, B, C, D, E, F (torch.Tensor): Double-layer site tensors of shape
            ``(D_squared, D_squared, D_squared)``.
        chi (int): Target bond dimension for the initial corners (before truncation).
        D_squared (int): ``D_bond ** 2``.
        identity_init (bool): If True, add a small identity component to each
            corner to ensure full rank before truncation.  This can help avoid
            numerical issues in the first truncation step when the raw corners are
            very low-rank.

    Returns:
        Tuple[torch.Tensor, ...]: The 9 initial corner matrices, each of shape
        ``(chi, chi)`` for C21CD, C32EF, C13AB, and shape ``(chi, chi, D_squared)``
        for the transfer tensors T1F, T2A, T2B, T3C, T3D, T1E.
    """
    D_bond = round(D_squared ** 0.5)

    if identity_init:
        # Add a small identity component to each corner to ensure full rank.
        # The scale is arbitrary but should be small enough not to dominate the
        # physical corner contribution; 1e-3 is a reasonable starting point.
        id_noise_scale = 1e-2

        if False: 

            identityC = torch.eye(chi, dtype=A.dtype, device=A.device)
            C21CD = identityC + id_noise_scale * torch.randn_like(identityC)
            C32EF = identityC + id_noise_scale * torch.randn_like(identityC)
            C13AB = identityC + id_noise_scale * torch.randn_like(identityC)

            identityT = torch.eye(chi*D_bond, dtype=A.dtype, device=A.device)
            identityT = identityT.reshape(chi, D_bond, chi, D_bond).permute(0, 2, 1, 3).reshape(chi, chi, D_squared)
            T1F = identityT + id_noise_scale * torch.randn_like(identityT)
            T2A = identityT + id_noise_scale * torch.randn_like(identityT)
            T2B = identityT + id_noise_scale * torch.randn_like(identityT)
            T3C = identityT + id_noise_scale * torch.randn_like(identityT)
            T3D = identityT + id_noise_scale * torch.randn_like(identityT)
            T1E = identityT + id_noise_scale * torch.randn_like(identityT)

        else:

            allOnesC = torch.ones((chi, chi), dtype=A.dtype, device=A.device)
            C21CD = allOnesC + id_noise_scale * torch.randn_like(allOnesC)
            C32EF = allOnesC + id_noise_scale * torch.randn_like(allOnesC)
            C13AB = allOnesC + id_noise_scale * torch.randn_like(allOnesC)

            allOnesT = torch.ones((chi, chi, D_squared), dtype=A.dtype, device=A.device)
            T1F = allOnesT + id_noise_scale * torch.randn_like(allOnesT)
            T2A = allOnesT + id_noise_scale * torch.randn_like(allOnesT)
            T2B = allOnesT + id_noise_scale * torch.randn_like(allOnesT)
            T3C = allOnesT + id_noise_scale * torch.randn_like(allOnesT)
            T3D = allOnesT + id_noise_scale * torch.randn_like(allOnesT)
            T1E = allOnesT + id_noise_scale * torch.randn_like(allOnesT)


        C21CD = normalize_tensor(C21CD)
        C32EF = normalize_tensor(C32EF)
        C13AB = normalize_tensor(C13AB)
        T1F = normalize_tensor(T1F)
        T2A = normalize_tensor(T2A)
        T2B = normalize_tensor(T2B)
        T3C = normalize_tensor(T3C)
        T3D = normalize_tensor(T3D)
        T1E = normalize_tensor(T1E)



    else: 
        # brutal contract env from 8*3 + 5*4*3 + 6 = 90 local tensors

        # enlarged C*3
        BC2323 = oe.contract("iBG,ibg->BGbg",B,C, optimize=[(0,1)], backend='torch')
        DE3131 = oe.contract("AiG,aig->GAga",D,E, optimize=[(0,1)], backend='torch')
        FA1212 = oe.contract("ABi,abi->ABab",F,A, optimize=[(0,1)], backend='torch')

        A23 = A.reshape(D_bond,D_bond,D_squared,D_squared).diagonal(dim1=0, dim2=1).sum(-1)
        F31 = F.permute(1,2,0).reshape(D_bond,D_bond,D_squared,D_squared).diagonal(dim1=0, dim2=1).sum(-1)
        C31 = C.permute(1,2,0).reshape(D_bond,D_bond,D_squared,D_squared).diagonal(dim1=0, dim2=1).sum(-1)
        B12 = B.permute(2,0,1).reshape(D_bond,D_bond,D_squared,D_squared).diagonal(dim1=0, dim2=1).sum(-1)
        E12 = E.permute(2,0,1).reshape(D_bond,D_bond,D_squared,D_squared).diagonal(dim1=0, dim2=1).sum(-1)
        D23 = D.reshape(D_bond,D_bond,D_squared,D_squared).diagonal(dim1=0, dim2=1).sum(-1)

        E23 = E.reshape(D_bond,D_bond,D_squared,D_squared).diagonal(dim1=0, dim2=1).sum(-1)
        B31 = B.permute(1,2,0).reshape(D_bond,D_bond,D_squared,D_squared).diagonal(dim1=0, dim2=1).sum(-1)
        A31 = A.permute(1,2,0).reshape(D_bond,D_bond,D_squared,D_squared).diagonal(dim1=0, dim2=1).sum(-1)
        D12 = D.permute(2,0,1).reshape(D_bond,D_bond,D_squared,D_squared).diagonal(dim1=0, dim2=1).sum(-1)
        C12 = C.permute(2,0,1).reshape(D_bond,D_bond,D_squared,D_squared).diagonal(dim1=0, dim2=1).sum(-1)
        F23 = F.reshape(D_bond,D_bond,D_squared,D_squared).diagonal(dim1=0, dim2=1).sum(-1)
        
        C21CD = oe.contract("yj,ijYk,in,nm,kXlm,lx->YyXx",
                            E23,BC2323,A23,F31,DE3131,B31,
                            optimize=[(0,1),(0,1),(0,1),(0,1),(0,1)],
                            backend='torch').reshape(D_squared*D_squared,D_squared*D_squared)
        C32EF = oe.contract("zj,ijZk,in,nm,kYlm,ly->ZzYy",
                            A31,DE3131,C31,B12,FA1212,D12,
                            optimize=[(0,1),(0,1),(0,1),(0,1),(0,1)],
                            backend='torch').reshape(D_squared*D_squared,D_squared*D_squared)
        C13AB = oe.contract("xj,ijXk,in,nm,kZlm,lz->XxZz",
                            C12,FA1212,E12,D23,BC2323,F23,
                            optimize=[(0,1),(0,1),(0,1),(0,1),(0,1)],
                            backend='torch').reshape(D_squared*D_squared,D_squared*D_squared)
        

        # enlarged T*6
        A12 = A.permute(2,0,1).reshape(D_bond,D_bond,D_squared,D_squared).diagonal(dim1=0, dim2=1).sum(-1)
        B23 = B.reshape(D_bond,D_bond,D_squared,D_squared).diagonal(dim1=0, dim2=1).sum(-1)
        C23 = C.reshape(D_bond,D_bond,D_squared,D_squared).diagonal(dim1=0, dim2=1).sum(-1)
        D31 = D.permute(1,2,0).reshape(D_bond,D_bond,D_squared,D_squared).diagonal(dim1=0, dim2=1).sum(-1)
        E31 = E.permute(1,2,0).reshape(D_bond,D_bond,D_squared,D_squared).diagonal(dim1=0, dim2=1).sum(-1)
        F12 = F.permute(2,0,1).reshape(D_bond,D_bond,D_squared,D_squared).diagonal(dim1=0, dim2=1).sum(-1)

        T1F = oe.contract("mi,jyi,aYjM->MmYya",
                          C23,D,FA1212,
                          optimize=[(0,1),(0,1)],
                          backend='torch').reshape(D_squared*D_squared,D_squared*D_squared,D_squared)
        T2A = oe.contract("il,xji,LjXb->LlXxb",
                          D31,C,FA1212,
                          optimize=[(0,1),(0,1)],
                          backend='torch').reshape(D_squared*D_squared,D_squared*D_squared,D_squared)   
        T2B = oe.contract("ni,ijz,bZjN->NnZzb",
                          E31,F,BC2323,
                          optimize=[(0,1),(0,1)],
                          backend='torch').reshape(D_squared*D_squared,D_squared*D_squared,D_squared)   
        T3C = oe.contract("im,iyj,MjYg->MmYyg",
                          F12,E,BC2323,
                          optimize=[(0,1),(0,1)],
                          backend='torch').reshape(D_squared*D_squared,D_squared*D_squared,D_squared)   
        T3D = oe.contract("li,xij,gXjL->LlXxg",
                          A12,B,DE3131,
                          optimize=[(0,1),(0,1)],
                          backend='torch').reshape(D_squared*D_squared,D_squared*D_squared,D_squared)   
        T1E = oe.contract("in,jiz,NjZa->NnZza",
                          B23,A,DE3131,
                          optimize=[(0,1),(0,1)],
                          backend='torch').reshape(D_squared*D_squared,D_squared*D_squared,D_squared)   

        R1 = C21CD.T
        R2 = torch.mm(C13AB,C32EF)

        #Q1, R1 = torch.linalg.qr(matC21.T)
        #Q2, R2 = torch.linalg.qr(torch.mm(matC13,matC32))
        U, S, V = truncated_svd_propack(torch.mm(R1,R2.T), chi,
                    chi_extra=1,
                    rel_cutoff=1e-12,
                    v0=None,
                    keep_multiplets=False,
                    abs_tol=1e-14,
                    eps_multiplet=1e-12)

        truncUh = U.conj().T
        truncV = V
        sqrtInvTruncS = torch.diag(1.0 / torch.sqrt(S[:chi]).to(CDTYPE))

        P21My = torch.mm( R2.T ,torch.mm( truncV , sqrtInvTruncS ))
        P32yM = torch.mm(torch.mm( sqrtInvTruncS , truncUh ), R1 )


        R1 = C32EF.T
        R2 = torch.mm(C21CD,C13AB)

        #Q1, R1 = torch.linalg.qr(matC32.T)
        #Q2, R2 = torch.linalg.qr(torch.mm(matC21,matC13))
        U, S, V = truncated_svd_propack(torch.mm(R1,R2.T), chi,
                    chi_extra=1,
                    rel_cutoff=1e-12,
                    v0=None,
                    keep_multiplets=False,
                    abs_tol=1e-14,
                    eps_multiplet=1e-12)

        truncUh = U.conj().T
        truncV = V
        sqrtInvTruncS = torch.diag(1.0 / torch.sqrt(S[:chi]).to(CDTYPE))

        P32Nz = torch.mm( R2.T ,torch.mm( truncV , sqrtInvTruncS ))
        P13zN = torch.mm(torch.mm( sqrtInvTruncS , truncUh ), R1 )


        R1 = C13AB.T
        R2 = torch.mm(C32EF,C21CD)

        #Q1, R1 = torch.linalg.qr(matC13.T)
        #Q2, R2 = torch.linalg.qr(torch.mm(matC32,matC21))
        U, S, V = truncated_svd_propack(torch.mm(R1,R2.T), chi,
                    chi_extra=1,
                    rel_cutoff=1e-12,
                    v0=None,
                    keep_multiplets=False,
                    abs_tol=1e-14,
                    eps_multiplet=1e-12)

        truncUh = U.conj().T
        truncV = V
        sqrtInvTruncS = torch.diag(1.0 / torch.sqrt(S[:chi]).to(CDTYPE))

        P13Lx = torch.mm( R2.T ,torch.mm( truncV , sqrtInvTruncS ))
        P21xL = torch.mm(torch.mm( sqrtInvTruncS , truncUh ), R1 )



        C21 = torch.mm(P21My.T, torch.mm(C21CD, P21xL.T))
        C32 = torch.mm(P32Nz.T, torch.mm(C32EF, P32yM.T))
        C13 = torch.mm(P13Lx.T, torch.mm(C13AB, P13zN.T))
        
        C21 = C21/torch.linalg.norm(C21)
        C32 = C32/torch.linalg.norm(C32)
        C13 = C13/torch.linalg.norm(C13)
        
        rho_trunc = C13 @ C32 @ C21
        tr_rho = torch.trace(rho_trunc)
        #print(f"Tr(rho_trunc) before normalization: {tr_rho.item():.6e}")
        toBeDivided = torch.pow(tr_rho, 1.0/3.0)
        C21CD = C21 / toBeDivided
        C32EF = C32 / toBeDivided
        C13AB = C13 / toBeDivided

        """
        tr_abs = tr_rho.abs()
        if torch.isfinite(tr_abs).item() and (tr_abs > torch.finfo(tr_abs.dtype).tiny).item():
            phase = tr_rho / tr_abs
            C21 = C21 / phase

            scaleC = tr_abs.pow(1.0 / 3.0)
            if torch.isfinite(scaleC).item() and (scaleC > torch.finfo(scaleC.dtype).tiny).item():
                C21 = C21 / scaleC
                C32 = C32 / scaleC
                C13 = C13 / scaleC

        # ── Individual equalization: preserve Tr(rho)=1 while balancing magnitudes ──
        # Scale factors multiply to 1  ⟹  Tr(rho) = Tr(C13·C32·C21) unchanged.
        n21 = torch.linalg.norm(C21).real
        n32 = torch.linalg.norm(C32).real
        n13 = torch.linalg.norm(C13).real
        nC_prod = n21 * n32 * n13
        if torch.isfinite(nC_prod).item() and (nC_prod > torch.finfo(n21.dtype).tiny).item():
            nC_geo = nC_prod.pow(1.0 / 3.0)
            C21CD = C21 * (nC_geo / n21)
            C32EF = C32 * (nC_geo / n32)
            C13AB = C13 * (nC_geo / n13)
        """

        U1B = P21My.reshape(D_squared*D_squared, chi)
        U2D = P32Nz.reshape(D_squared*D_squared, chi)
        U3F = P13Lx.reshape(D_squared*D_squared, chi)

        V1C = P13zN.reshape(chi, D_squared*D_squared)
        V2E = P21xL.reshape(chi, D_squared*D_squared)
        V3A = P32yM.reshape(chi, D_squared*D_squared)


        # Uh,V project on one side(XYZ) of Ts

        T1F = oe.contract("MYa,yY->Mya",T1F,V3A,optimize=[(0,1)],backend='torch').reshape(D_squared*D_squared,chi*D_squared)
        T2A = oe.contract("LXb,Xx->Lxb",T2A,U3F,optimize=[(0,1)],backend='torch').reshape(D_squared*D_squared,chi*D_squared)
        T2B = oe.contract("NZb,zZ->Nzb",T2B,V1C,optimize=[(0,1)],backend='torch').reshape(D_squared*D_squared,chi*D_squared)
        T3C = oe.contract("MYg,Yy->Myg",T3C,U1B,optimize=[(0,1)],backend='torch').reshape(D_squared*D_squared,chi*D_squared)
        T3D = oe.contract("LXg,xX->Lxg",T3D,V2E,optimize=[(0,1)],backend='torch').reshape(D_squared*D_squared,chi*D_squared)
        T1E = oe.contract("NZa,Zz->Nza",T1E,U2D,optimize=[(0,1)],backend='torch').reshape(D_squared*D_squared,chi*D_squared)  




        
        #the other side(LMN) projection of Ts

        if False:
            out2Ain3D = torch.mm(T2A.T, T3D)
            out3Cin1F = torch.mm(T3C.T, T1F)
            out1Ein2B = torch.mm(T1E.T, T2B)
            T2A, svL, T3D = truncated_svd_propack(out2Ain3D, chi,
                        chi_extra=1,
                        rel_cutoff=1e-12,
                        v0=None,
                        keep_multiplets=False,
                        abs_tol=1e-14,
                        eps_multiplet=1e-12)
            T3C, svM, T1F = truncated_svd_propack(out3Cin1F, chi,
                        chi_extra=1,
                        rel_cutoff=1e-12,
                        v0=None,
                        keep_multiplets=False,
                        abs_tol=1e-14,
                        eps_multiplet=1e-12)
            T1E, svN, T2B = truncated_svd_propack(out1Ein2B, chi,
                        chi_extra=1,
                        rel_cutoff=1e-12,
                        v0=None,
                        keep_multiplets=False,
                        abs_tol=1e-14,
                        eps_multiplet=1e-12)

            sqrt_svL = torch.sqrt(svL)
            sqrt_svM = torch.sqrt(svM)
            sqrt_svN = torch.sqrt(svN)
            T2A = T2A.T * sqrt_svL.unsqueeze(1)
            T3D = T3D.conj().T * sqrt_svL.unsqueeze(1)
            T3C = T3C.T * sqrt_svM.unsqueeze(1)
            T1F = T1F.conj().T * sqrt_svM.unsqueeze(1)
            T1E = T1E.T * sqrt_svN.unsqueeze(1)
            T2B = T2B.conj().T * sqrt_svN.unsqueeze(1)
        
        else: 
            U, S, V = truncated_svd_propack(torch.mm(T1F.T,T3C), chi,
                        chi_extra=1,
                        rel_cutoff=1e-12,
                        v0=None,
                        keep_multiplets=False,
                        abs_tol=1e-14,
                        eps_multiplet=1e-12)
            truncUh = U.conj().T
            truncV = V
            sqrtInvTruncS = torch.diag(1.0 / torch.sqrt(S[:chi]).to(CDTYPE))
            projFor1st = torch.mm(T3C, torch.mm(truncV, sqrtInvTruncS))
            projFor2nd = torch.mm( torch.mm(sqrtInvTruncS, truncUh), T1F.T)
            T1F = torch.mm(projFor1st.T, T1F)
            T3C = torch.mm(projFor2nd, T3C)

            U, S, V = truncated_svd_propack(torch.mm(T3D.T,T2A), chi,
                        chi_extra=1,
                        rel_cutoff=1e-12,
                        v0=None,
                        keep_multiplets=False,
                        abs_tol=1e-14,
                        eps_multiplet=1e-12)
            truncUh = U.conj().T
            truncV = V
            sqrtInvTruncS = torch.diag(1.0 / torch.sqrt(S[:chi]).to(CDTYPE))
            projFor1st = torch.mm(T2A, torch.mm(truncV, sqrtInvTruncS))
            projFor2nd = torch.mm( torch.mm(sqrtInvTruncS, truncUh), T3D.T)
            T3D = torch.mm(projFor1st.T, T3D)
            T2A = torch.mm(projFor2nd, T2A)

            U, S, V = truncated_svd_propack(torch.mm(T2B.T,T1E), chi,
                        chi_extra=1,
                        rel_cutoff=1e-12,
                        v0=None,
                        keep_multiplets=False,
                        abs_tol=1e-14,
                        eps_multiplet=1e-12)
            truncUh = U.conj().T
            truncV = V
            sqrtInvTruncS = torch.diag(1.0 / torch.sqrt(S[:chi]).to(CDTYPE))
            projFor1st = torch.mm(T1E, torch.mm(truncV, sqrtInvTruncS))
            projFor2nd = torch.mm( torch.mm(sqrtInvTruncS, truncUh), T2B.T)
            T2B = torch.mm(projFor1st.T, T2B)
            T1E = torch.mm(projFor2nd, T1E)
            





        T1F = T1F.reshape(chi, chi, D_squared)
        T2A = T2A.reshape(chi, chi, D_squared)
        T2B = T2B.reshape(chi, chi, D_squared)
        T3C = T3C.reshape(chi, chi, D_squared)
        T3D = T3D.reshape(chi, chi, D_squared)
        T1E = T1E.reshape(chi, chi, D_squared)

        T1F = T1F / torch.linalg.norm(T1F)
        T2A = T2A / torch.linalg.norm(T2A)
        T2B = T2B / torch.linalg.norm(T2B)
        T3C = T3C / torch.linalg.norm(T3C)
        T3D = T3D / torch.linalg.norm(T3D)
        T1E = T1E / torch.linalg.norm(T1E)



        # normalization of T*6
        closed_E = oe.contract("YX,MYa,abg->MbXg",
                                C21CD, T1F, E,
                                optimize=[(0,1),(0,1)], backend='torch'
                                ).reshape(chi*D_squared, chi*D_squared)

        closed_D = oe.contract("MYg,abg->YaMb",
                                T3C, D,
                                optimize=[(0,1)], backend='torch'
                                ).reshape(chi*D_squared, chi*D_squared)
        closed_A = oe.contract("ZY,NZb,abg->NgYa",
                                C32EF, T2B, A,
                                optimize=[(0,1),(0,1)], backend='torch'
                                ).reshape(chi*D_squared, chi*D_squared)
        closed_F = oe.contract("NZa,abg->ZbNg",
                                T1E, F,
                                optimize=[(0,1)], backend='torch'
                                ).reshape(chi*D_squared, chi*D_squared)
        closed_C = oe.contract("XZ,LXg,abg->LaZb",
                                C13AB, T3D, C,
                                optimize=[(0,1),(0,1)], backend='torch'
                                ).reshape(chi*D_squared, chi*D_squared)
        closed_B = oe.contract("LXb,abg->XgLa",
                                T2A, B,
                                optimize=[(0,1)], backend='torch'
                                ).reshape(chi*D_squared, chi*D_squared)
        
        AD = torch.mm(closed_A, closed_D)
        CF = torch.mm(closed_C, closed_F)
        EB = torch.mm(closed_E, closed_B)
        TrRho_val  = oe.contract("xy,yz,zx->", AD, EB, CF, backend='torch')
        #print(f"Tr(Rho) for T normalization check: {TrRho_val.item():.6e}")

        toBeDivided = torch.pow(TrRho_val, 1.0/6.0)
        T1F = T1F / toBeDivided
        T2A = T2A / toBeDivided
        T2B = T2B / toBeDivided
        T3C = T3C / toBeDivided
        T3D = T3D / toBeDivided
        T1E = T1E / toBeDivided

        """
        norm_abs = norm_val.abs()
        if torch.isfinite(norm_abs).item() and (norm_abs > torch.finfo(norm_abs.dtype).tiny).item():
            phase = norm_val / norm_abs
            T3C = T3C / phase
            scaleT = norm_abs.pow(1.0 / 6.0)
            if torch.isfinite(scaleT).item() and (scaleT > torch.finfo(scaleT.dtype).tiny).item():
                T3C = T3C / scaleT
                T3D = T3D / scaleT
                T1E = T1E / scaleT
                T1F = T1F / scaleT
                T2A = T2A / scaleT
                T2B = T2B / scaleT

        nT3C = torch.linalg.norm(T3C).real
        nT3D = torch.linalg.norm(T3D).real
        nT1E = torch.linalg.norm(T1E).real
        nT1F = torch.linalg.norm(T1F).real
        nT2A = torch.linalg.norm(T2A).real
        nT2B = torch.linalg.norm(T2B).real
        nT_prod = nT3C * nT3D * nT1E * nT1F * nT2A * nT2B
        if torch.isfinite(nT_prod).item() and (nT_prod > torch.finfo(nT3C.dtype).tiny).item():
            nT_geo = nT_prod.pow(1.0 / 6.0)
            T3C = T3C * (nT_geo / nT3C)
            T3D = T3D * (nT_geo / nT3D)
            T1E = T1E * (nT_geo / nT1E)
            T1F = T1F * (nT_geo / nT1F)
            T2A = T2A * (nT_geo / nT2A)
            T2B = T2B * (nT_geo / nT2B)
    """
        

    return C21CD, C32EF, C13AB, T1F, T2A, T2B, T3C, T3D, T1E






def check_env_convergence_DEPRECATED(lastC21CD, lastC32EF, lastC13AB, lastT1F, lastT2A, lastT2B, lastT3C, lastT3D, lastT1E, 
                          nowC21CD, nowC32EF, nowC13AB, nowT1F, nowT2A, nowT2B, nowT3C, nowT3D, nowT1E, 
                          lastC21EB, lastC32AD, lastC13CF, lastT1D, lastT2C, lastT2F, lastT3E, lastT3B, lastT1A, 
                          nowC21EB, nowC32AD, nowC13CF, nowT1D, nowT2C, nowT2F, nowT3E, nowT3B, nowT1A, 
                          lastC21AF, lastC32CB, lastC13ED, lastT1B, lastT2E, lastT2D, lastT3A, lastT3F, lastT1C, 
                          nowC21AF, nowC32CB, nowC13ED, nowT1B, nowT2E, nowT2D, nowT3A, nowT3F, nowT1C,
                          env_conv_threshold):
    """
    Decide whether all 9 CTMRG corner matrices have converged.

    Convergence is measured by comparing the **singular-value spectra** of the
    nine corner matrices (three per environment type) between consecutive
    CTMRG cycles.  Singular values are gauge-invariant — unlike raw tensor
    elements, they are unaffected by the sign/phase ambiguity that arises from
    the SVD-based projectors inside ``trunc_rhoCCC``.  Using raw Frobenius
    distance on the environment tensors would *never* converge because those
    sign flips produce an O(1) Frobenius difference even when the physical
    environment is at its fixed point.

    The convergence metric is:

    .. math::

        d = \\max_{c \\in \\{9\\,\\text{corners}\\}}
              \\max_k \\left|
                \\sigma_k^{\\text{now}}(c) - \\sigma_k^{\\text{last}}(c)
              \\right|

    where :math:`\\sigma_k(c)` are the singular values of corner matrix ``c``
    normalised so that :math:`\\sigma_1 = 1`.  The function returns
    ``True`` when ``d < env_conv_threshold``.

    The function returns ``False`` immediately when any of the ``last*``
    corner tensors is ``None`` (i.e. during the first full cycle where not all
    three environment types have been computed yet).

    Args:
        lastC21CD … lastT1C: The 27 environment tensors from the previous
            CTMRG cycle (corner tensors may be ``None`` during warm-up;
            transfer tensors are accepted but not used for convergence).
        nowC21CD … nowT1C: The 27 environment tensors from the current cycle.
        env_conv_threshold (float): Convergence threshold for ``d``.

    Returns:
        bool: ``True`` iff ``d < env_conv_threshold``, ``False`` otherwise
        (including when any ``last*`` corner tensor is ``None``).
    """

    # Warm-up guard: all three env types must have been computed at least once.
    if lastC21CD is None or lastC21EB is None or lastC21AF is None:
        return False

    # The 9 corner pairs (last, now) across all three environment types.
    corner_pairs = [
        (lastC21CD, nowC21CD), (lastC32EF, nowC32EF), (lastC13AB, nowC13AB),
        (lastC21EB, nowC21EB), (lastC32AD, nowC32AD), (lastC13CF, nowC13CF),
        (lastC21AF, nowC21AF), (lastC32CB, nowC32CB), (lastC13ED, nowC13ED),
    ]

    max_delta = 0.0
    for last_c, now_c in corner_pairs:
        sv_last = torch.linalg.svdvals(last_c).real
        sv_now  = torch.linalg.svdvals(now_c ).real
        # Normalise so that the largest singular value is 1 (scale-invariant).
        sv_last = sv_last / (sv_last[0] + 1e-30)
        sv_now  = sv_now  / (sv_now [0] + 1e-30)
        delta = (sv_now - sv_last).abs().max().item()
        if delta > max_delta:
            max_delta = delta

    return bool(max_delta < env_conv_threshold)



def check_env_CV_using_3rho(lastC21CD, lastC32EF, lastC13AB, 
                            nowC21CD, nowC32EF, nowC13AB, 
                            lastC21EB, lastC32AD, lastC13CF, 
                            nowC21EB, nowC32AD, nowC13CF, 
                            lastC21AF, lastC32CB, lastC13ED, 
                            nowC21AF, nowC32CB, nowC13ED, 
                            env_conv_threshold):
   
    # Warm-up guard: all three env types must have been computed at least once.
    if lastC21CD is None or lastC21EB is None or lastC21AF is None:
        return False

    max_delta = 0.0
    # The rhos defined by 1-1 cut: 13 @ 32 @ 21
    last_rho1 = oe.contract("UZ,ZY,YV->UV",
                                lastC13AB,lastC32EF,lastC21CD,
                                optimize=[(0,1),(0,1)],
                                backend='torch')
    now_rho1 = oe.contract("UZ,ZY,YV->UV",
                                nowC13AB,nowC32EF,nowC21CD,
                                optimize=[(0,1),(0,1)],
                                backend='torch')
    last_rho2 = oe.contract("UZ,ZY,YV->UV",
                                lastC13CF,lastC32AD,lastC21EB,
                                optimize=[(0,1),(0,1)],
                                backend='torch')
    now_rho2 = oe.contract("UZ,ZY,YV->UV",
                                nowC13CF,nowC32AD,nowC21EB,
                                optimize=[(0,1),(0,1)],
                                backend='torch')
    last_rho3 = oe.contract("UZ,ZY,YV->UV",
                                lastC13ED,lastC32CB,lastC21AF,
                                optimize=[(0,1),(0,1)],
                                backend='torch')
    now_rho3 = oe.contract("UZ,ZY,YV->UV",
                                nowC13ED,nowC32CB,nowC21AF, 
                                optimize=[(0,1),(0,1)],
                                backend='torch')

    rho_pairs = [
        (last_rho1, now_rho1),
        (last_rho2, now_rho2),
        (last_rho3, now_rho3),
    ]

    for last_c, now_c in rho_pairs:
        sv_last = torch.linalg.svdvals(last_c).real
        sv_now  = torch.linalg.svdvals(now_c ).real
        # Normalise so that the largest singular value is 1 (scale-invariant).
        sv_last = sv_last / (sv_last[0] + 1e-30)
        sv_now  = sv_now  / (sv_now [0] + 1e-30)
        delta = (sv_now - sv_last).abs().max().item()
        if delta > max_delta:
            max_delta = delta

    last_rho1 = oe.contract("UY,YX,XV->UV",
                                lastC32EF,lastC21CD,lastC13AB,
                                optimize=[(0,1),(0,1)],
                                backend='torch')
    now_rho1 = oe.contract("UY,YX,XV->UV",
                                nowC32EF,nowC21CD,nowC13AB,
                                optimize=[(0,1),(0,1)],
                                backend='torch')
    last_rho2 = oe.contract("UY,YX,XV->UV",
                                lastC32AD,lastC21EB,lastC13CF,
                                optimize=[(0,1),(0,1)],
                                backend='torch')
    now_rho2 = oe.contract("UY,YX,XV->UV",
                                nowC32AD,nowC21EB,nowC13CF,
                                optimize=[(0,1),(0,1)],
                                backend='torch')
    last_rho3 = oe.contract("UY,YX,XV->UV",
                                lastC32CB,lastC21AF,lastC13ED,
                                optimize=[(0,1),(0,1)],
                                backend='torch')
    now_rho3 = oe.contract("UY,YX,XV->UV",
                                nowC32CB,nowC21AF,nowC13ED, 
                                optimize=[(0,1),(0,1)],
                                backend='torch')

    rho_pairs = [
        (last_rho1, now_rho1),
        (last_rho2, now_rho2),
        (last_rho3, now_rho3),
    ]

    for last_c, now_c in rho_pairs:
        sv_last = torch.linalg.svdvals(last_c).real
        sv_now  = torch.linalg.svdvals(now_c ).real
        # Normalise so that the largest singular value is 1 (scale-invariant).
        sv_last = sv_last / (sv_last[0] + 1e-30)
        sv_now  = sv_now  / (sv_now [0] + 1e-30)
        delta = (sv_now - sv_last).abs().max().item()
        if delta > max_delta:
            max_delta = delta

    last_rho1 = oe.contract("UX,XZ,ZV->UV",
                                lastC21CD,lastC13AB,lastC32EF,
                                optimize=[(0,1),(0,1)],
                                backend='torch')
    now_rho1 = oe.contract("UX,XZ,ZV->UV",
                                nowC21CD,nowC13AB,nowC32EF,
                                optimize=[(0,1),(0,1)],
                                backend='torch')
    last_rho2 = oe.contract("UX,XZ,ZV->UV",
                                lastC21EB,lastC13CF,lastC32AD,
                                optimize=[(0,1),(0,1)],
                                backend='torch')
    now_rho2 = oe.contract("UX,XZ,ZV->UV",
                                nowC21EB,nowC13CF,nowC32AD,
                                optimize=[(0,1),(0,1)],
                                backend='torch')
    last_rho3 = oe.contract("UX,XZ,ZV->UV",
                                lastC21AF,lastC13ED,lastC32CB,
                                optimize=[(0,1),(0,1)],
                                backend='torch')
    now_rho3 = oe.contract("UX,XZ,ZV->UV",
                                nowC21AF,nowC13ED,nowC32CB, 
                                optimize=[(0,1),(0,1)],
                                backend='torch')

    rho_pairs = [
        (last_rho1, now_rho1),
        (last_rho2, now_rho2),
        (last_rho3, now_rho3),
    ]

    for last_c, now_c in rho_pairs:
        sv_last = torch.linalg.svdvals(last_c).real
        sv_now  = torch.linalg.svdvals(now_c ).real
        # Normalise so that the largest singular value is 1 (scale-invariant).
        sv_last = sv_last / (sv_last[0] + 1e-30)
        sv_now  = sv_now  / (sv_now [0] + 1e-30)
        delta = (sv_now - sv_last).abs().max().item()
        if delta > max_delta:
            max_delta = delta

    return bool(max_delta < env_conv_threshold)



def update_environmentCTs_1to2(C21CD, C32EF, C13AB, T1F, T2A, T2B, T3C, T3D, T1E, A,B,C,D,E,F, chi, D_squared):
    """
    Perform one CTMRG renormalisation step: environment type 1 → type 2.

    Absorbs the site tensors ``E``, ``B``, ``A``, ``D``, ``C``, ``F`` into
    the type-1 corner and transfer tensors and produces the type-2 set.  The
    three new (grown) corner matrices are truncated to bond dimension ``chi``
    using ``trunc_rhoCCC``, and all six new transfer tensors are renormalised.

    Tensor naming convention:  ``C21EB`` is the corner connecting legs 2→1
    in the sublattice-EB orientation; ``T1D`` is the type-1 transfer tensor
    carrying the ``D``-site bond; etc.

    Args:
        C21CD, C32EF, C13AB (torch.Tensor): Type-1 corner matrices,
            shape ``(chi, chi)``.
        T1F, T2A, T2B, T3C, T3D, T1E (torch.Tensor): Type-1 transfer tensors,
            shape ``(chi, D_squared, D_squared)``.
        A, B, C, D, E, F (torch.Tensor): Double-layer site tensors,
            shape ``(D_squared, D_squared)``.
        chi (int): Target bond dimension.
        D_squared (int): ``D_bond ** 2``.

    Returns:
        Tuple of 9 torch.Tensor: ``(C21EB, C32AD, C13CF, T1D, T2C, T2F,
        T3E, T3B, T1A)`` — the type-2 corners (shape ``(chi, chi)``) and
        transfer tensors (shape ``(chi, D_squared, D_squared)``).
    """

    matC21EB = oe.contract("YX,MYa,LXb,amg,lbg->MmLl",
                           C21CD,T1F,T2A,E,B,
                           optimize=[(0,2),(0,3),(1,2),(0,1)],
                           backend='torch')
    
    matC21EB = matC21EB.reshape(chi*D_squared,chi*D_squared)
    
    matC32AD = oe.contract("ZY,NZb,MYg,abn,amg->NnMm",
                           C32EF,T2B,T3C,A,D,
                           optimize=[(0,2),(0,3),(1,2),(0,1)],
                           backend='torch')
    
    matC32AD = matC32AD.reshape(chi*D_squared,chi*D_squared)
    
    matC13CF = oe.contract("XZ,LXg,NZa,lbg,abn->LlNn",
                           C13AB,T3D,T1E,C,F,
                           optimize=[(0,2),(0,3),(1,2),(0,1)],
                           backend='torch')
    
    matC13CF = matC13CF.reshape(chi*D_squared,chi*D_squared)

    V2A, C21EB, U1F, V3C, C32AD, U2B, V1E, C13CF, U3D = trunc_rhoCCC(
                        matC21EB, matC32AD, matC13CF, chi, D_squared)


    T3E = oe.contract("OYa,abg,ObM->YMg",
                        T1F,E,U1F,
                        optimize=[(0,1),(0,1)],
                        backend='torch')

    T3B = oe.contract("OXb,abg,LOa->XLg",
                        T2A,B,V2A,
                        optimize=[(0,1),(0,1)],
                        backend='torch')

    T1A = oe.contract("OZb,abg,OgN->ZNa",
                        T2B,A,U2B,
                        optimize=[(0,1),(0,1)],
                        backend='torch')

    T1D = oe.contract("OYg,abg,MOb->YMa",
                        T3C,D,V3C,
                        optimize=[(0,1),(0,1)],
                        backend='torch')

    T2C = oe.contract("OXg,abg,OaL->XLb",
                        T3D,C,U3D,
                        optimize=[(0,1),(0,1)],
                        backend='torch')

    T2F = oe.contract("OZa,abg,NOg->ZNb",
                        T1E,F,V1E,
                        optimize=[(0,1),(0,1)],
                        backend='torch')

    # ── End-of-update transfer normalization (mirrors norm_env_2) ────────────
    # Expand double-layer (D²,D²,D²) → rank-6 (D,D,D,D,D,D) so the contraction
    # strings are structurally identical to norm_env_2; then divide all 6 Ts
    # by <iPEPS|iPEPS>^(1/6).







    T3E = T3E/torch.linalg.norm(T3E)
    T3B = T3B/torch.linalg.norm(T3B)
    T1A = T1A/torch.linalg.norm(T1A)
    T1D = T1D/torch.linalg.norm(T1D)
    T2C = T2C/torch.linalg.norm(T2C)
    T2F = T2F/torch.linalg.norm(T2F)







    D_bond = int(round(D_squared ** 0.5))
    A6 = A.reshape(D_bond, D_bond, D_bond, D_bond, D_bond, D_bond)  # indices: (a,r,b,s,c,t)
    B6 = B.reshape(D_bond, D_bond, D_bond, D_bond, D_bond, D_bond)
    C6 = C.reshape(D_bond, D_bond, D_bond, D_bond, D_bond, D_bond)
    D6 = D.reshape(D_bond, D_bond, D_bond, D_bond, D_bond, D_bond)
    E6 = E.reshape(D_bond, D_bond, D_bond, D_bond, D_bond, D_bond)
    F6 = F.reshape(D_bond, D_bond, D_bond, D_bond, D_bond, D_bond)
    T1D4 = T1D.reshape(chi, chi, D_bond, D_bond)
    T2C4 = T2C.reshape(chi, chi, D_bond, D_bond)
    T2F4 = T2F.reshape(chi, chi, D_bond, D_bond)
    T3E4 = T3E.reshape(chi, chi, D_bond, D_bond)
    T3B4 = T3B.reshape(chi, chi, D_bond, D_bond)
    T1A4 = T1A.reshape(chi, chi, D_bond, D_bond)
    # norm_env_2: open_A= "YX,MYar,abci,rstj->MbsXctij" → double-layer: "YX,MYar,arbsct->MbsXct"
    closed_A = oe.contract("YX,MYar,arbsct->MbsXct",
                            C21EB, T1D4, A6,
                            optimize=[(0,1),(0,1)], backend='torch'
                            ).reshape(chi*D_bond*D_bond, chi*D_bond*D_bond)
    # open_B= "MYct,abci,rstj->YarMbsij" → "MYct,arbsct->YarMbs"
    closed_B = oe.contract("MYct,arbsct->YarMbs",
                            T3E4, B6,
                            optimize=[(0,1)], backend='torch'
                            ).reshape(chi*D_bond*D_bond, chi*D_bond*D_bond)
    # open_C= "ZY,NZbs,abci,rstj->NctYarij" → "ZY,NZbs,arbsct->NctYar"
    closed_C = oe.contract("ZY,NZbs,arbsct->NctYar",
                            C32AD, T2F4, C6,
                            optimize=[(0,1),(0,1)], backend='torch'
                            ).reshape(chi*D_bond*D_bond, chi*D_bond*D_bond)
    # open_D= "NZar,abci,rstj->ZbsNctij" → "NZar,arbsct->ZbsNct"
    closed_D = oe.contract("NZar,arbsct->ZbsNct",
                            T1A4, D6,
                            optimize=[(0,1)], backend='torch'
                            ).reshape(chi*D_bond*D_bond, chi*D_bond*D_bond)
    # open_E= "XZ,LXct,abci,rstj->LarZbsij" → "XZ,LXct,arbsct->LarZbs"
    closed_E = oe.contract("XZ,LXct,arbsct->LarZbs",
                            C13CF, T3B4, E6,
                            optimize=[(0,1),(0,1)], backend='torch'
                            ).reshape(chi*D_bond*D_bond, chi*D_bond*D_bond)
    # open_F= "LXbs,abci,rstj->XctLarij" → "LXbs,arbsct->XctLar"
    closed_F = oe.contract("LXbs,arbsct->XctLar",
                            T2C4, F6,
                            optimize=[(0,1)], backend='torch'
                            ).reshape(chi*D_bond*D_bond, chi*D_bond*D_bond)
    CB = torch.mm(closed_C, closed_B)
    ED = torch.mm(closed_E, closed_D)
    AF = torch.mm(closed_A, closed_F)
    TrRho_val  = oe.contract("xy,yz,zx->", CB, AF, ED, backend='torch')
    #print("TrRhoFor6Ts_val", TrRho_val.item())

    toBeDivided = torch.pow(TrRho_val.abs(), 1.0/6.0)
    T3E = T3E / toBeDivided
    T3B = T3B / toBeDivided
    T1A = T1A / toBeDivided
    T1D = T1D / toBeDivided
    T2C = T2C / toBeDivided
    T2F = T2F / toBeDivided

    """

    # Fix both magnitude and complex phase so <iPEPS|iPEPS> ≈ 1 (real).
    norm_abs = norm_val.abs()
    if torch.isfinite(norm_abs).item() and (norm_abs > torch.finfo(norm_abs.dtype).tiny).item():
        phase = norm_val / norm_abs
        T3E = T3E / phase

        scaleT = norm_abs.pow(1.0 / 6.0)
        if torch.isfinite(scaleT).item() and (scaleT > torch.finfo(scaleT.dtype).tiny).item():
            T3E = T3E / scaleT
            T3B = T3B / scaleT
            T1A = T1A / scaleT
            T1D = T1D / scaleT
            T2C = T2C / scaleT
            T2F = T2F / scaleT

    # ── Individual equalization: preserve norm_env_2=1 while balancing magnitudes ──
    # Each T appears linearly in exactly one closed_* factor; scale factors multiply to 1.
    nT3E = torch.linalg.norm(T3E).real
    nT3B = torch.linalg.norm(T3B).real
    nT1A = torch.linalg.norm(T1A).real
    nT1D = torch.linalg.norm(T1D).real
    nT2C = torch.linalg.norm(T2C).real
    nT2F = torch.linalg.norm(T2F).real
    nT_prod = nT3E * nT3B * nT1A * nT1D * nT2C * nT2F
    if torch.isfinite(nT_prod).item() and (nT_prod > torch.finfo(nT3E.dtype).tiny).item():
        nT_geo = nT_prod.pow(1.0 / 6.0)
        T3E = T3E * (nT_geo / nT3E)
        T3B = T3B * (nT_geo / nT3B)
        T1A = T1A * (nT_geo / nT1A)
        T1D = T1D * (nT_geo / nT1D)
        T2C = T2C * (nT_geo / nT2C)
        T2F = T2F * (nT_geo / nT2F)
    
    """

    return C21EB, C32AD, C13CF, T1D, T2C, T2F, T3E, T3B, T1A




def update_environmentCTs_2to3(C21EB, C32AD, C13CF, T1D, T2C, T2F, T3E, T3B, T1A, A,B,C,D,E,F, chi, D_squared):
    """
    Perform one CTMRG renormalisation step: environment type 2 → type 3.

    Absorbs site tensors ``A``, ``F``, ``C``, ``B``, ``E``, ``D`` into the
    type-2 environment and produces the type-3 set, following the same
    absorb-truncate-renormalise pattern as ``update_environmentCTs_1to2``.

    Args:
        C21EB, C32AD, C13CF (torch.Tensor): Type-2 corner matrices,
            shape ``(chi, chi)``.
        T1D, T2C, T2F, T3E, T3B, T1A (torch.Tensor): Type-2 transfer tensors,
            shape ``(chi, D_squared, D_squared)``.
        A, B, C, D, E, F (torch.Tensor): Double-layer site tensors,
            shape ``(D_squared, D_squared)``.
        chi (int): Target bond dimension.
        D_squared (int): ``D_bond ** 2``.

    Returns:
        Tuple of 9 torch.Tensor: ``(C21AF, C32CB, C13ED, T1B, T2E, T2D,
        T3A, T3F, T1C)`` — the type-3 corners and transfer tensors.
    """

    matC21AF = oe.contract("YX,MYa,LXb,amg,lbg->MmLl",
                           C21EB,T1D,T2C,A,F,
                           optimize=[(0,2),(0,3),(1,2),(0,1)],
                           backend='torch')
    
    matC21AF = matC21AF.reshape(chi*D_squared,chi*D_squared)
    
    matC32CB = oe.contract("ZY,NZb,MYg,abn,amg->NnMm",
                           C32AD,T2F,T3E,C,B,
                           optimize=[(0,2),(0,3),(1,2),(0,1)],
                           backend='torch')
    
    matC32CB = matC32CB.reshape(chi*D_squared,chi*D_squared)
    
    matC13ED = oe.contract("XZ,LXg,NZa,lbg,abn->LlNn",
                           C13CF,T3B,T1A,E,D,
                           optimize=[(0,2),(0,3),(1,2),(0,1)],
                           backend='torch')
    
    matC13ED = matC13ED.reshape(chi*D_squared,chi*D_squared)

    V2C, C21AF, U1D, V3E, C32CB, U2F, V1A, C13ED, U3B = trunc_rhoCCC(
                        matC21AF, matC32CB, matC13ED, chi, D_squared)

    T3A = oe.contract("OYa,abg,ObM->YMg",
                        T1D,A,U1D,
                        optimize=[(0,1),(0,1)],
                        backend='torch')

    T3F = oe.contract("OXb,abg,LOa->XLg",
                        T2C,F,V2C,
                        optimize=[(0,1),(0,1)],
                        backend='torch')

    T1C = oe.contract("OZb,abg,OgN->ZNa",
                        T2F,C,U2F,
                        optimize=[(0,1),(0,1)],
                        backend='torch')

    T1B = oe.contract("OYg,abg,MOb->YMa",
                        T3E,B,V3E,
                        optimize=[(0,1),(0,1)],
                        backend='torch')

    T2E = oe.contract("OXg,abg,OaL->XLb",
                        T3B,E,U3B,
                        optimize=[(0,1),(0,1)],
                        backend='torch')

    T2D = oe.contract("OZa,abg,NOg->ZNb",
                        T1A,D,V1A,
                        optimize=[(0,1),(0,1)],
                        backend='torch')
    









    T3A = T3A/torch.linalg.norm(T3A)
    T3F = T3F/torch.linalg.norm(T3F)
    T1C = T1C/torch.linalg.norm(T1C)
    T1B = T1B/torch.linalg.norm(T1B)
    T2E = T2E/torch.linalg.norm(T2E)
    T2D = T2D/torch.linalg.norm(T2D)









    # ── End-of-update transfer normalization (mirrors norm_env_3) ────────────
    D_bond = int(round(D_squared ** 0.5))
    A6 = A.reshape(D_bond, D_bond, D_bond, D_bond, D_bond, D_bond)
    B6 = B.reshape(D_bond, D_bond, D_bond, D_bond, D_bond, D_bond)
    C6 = C.reshape(D_bond, D_bond, D_bond, D_bond, D_bond, D_bond)
    D6 = D.reshape(D_bond, D_bond, D_bond, D_bond, D_bond, D_bond)
    E6 = E.reshape(D_bond, D_bond, D_bond, D_bond, D_bond, D_bond)
    F6 = F.reshape(D_bond, D_bond, D_bond, D_bond, D_bond, D_bond)
    T1B4 = T1B.reshape(chi, chi, D_bond, D_bond)
    T2E4 = T2E.reshape(chi, chi, D_bond, D_bond)
    T2D4 = T2D.reshape(chi, chi, D_bond, D_bond)
    T3A4 = T3A.reshape(chi, chi, D_bond, D_bond)
    T3F4 = T3F.reshape(chi, chi, D_bond, D_bond)
    T1C4 = T1C.reshape(chi, chi, D_bond, D_bond)
    # norm_env_3: open_C= "YX,MYar,abci,rstj->MbsXctij" → "YX,MYar,arbsct->MbsXct"
    closed_C = oe.contract("YX,MYar,arbsct->MbsXct",
                            C21AF, T1B4, C6,
                            optimize=[(0,1),(0,1)], backend='torch'
                            ).reshape(chi*D_bond*D_bond, chi*D_bond*D_bond)
    # open_F= "MYct,abci,rstj->YarMbsij" → "MYct,arbsct->YarMbs"
    closed_F = oe.contract("MYct,arbsct->YarMbs",
                            T3A4, F6,
                            optimize=[(0,1)], backend='torch'
                            ).reshape(chi*D_bond*D_bond, chi*D_bond*D_bond)
    # open_E= "ZY,NZbs,abci,rstj->NctYarij" → "ZY,NZbs,arbsct->NctYar"
    closed_E = oe.contract("ZY,NZbs,arbsct->NctYar",
                            C32CB, T2D4, E6,
                            optimize=[(0,1),(0,1)], backend='torch'
                            ).reshape(chi*D_bond*D_bond, chi*D_bond*D_bond)
    # open_B= "NZar,abci,rstj->ZbsNctij" → "NZar,arbsct->ZbsNct"
    closed_B = oe.contract("NZar,arbsct->ZbsNct",
                            T1C4, B6,
                            optimize=[(0,1)], backend='torch'
                            ).reshape(chi*D_bond*D_bond, chi*D_bond*D_bond)
    # open_A= "XZ,LXct,abci,rstj->LarZbsij" → "XZ,LXct,arbsct->LarZbs"
    closed_A = oe.contract("XZ,LXct,arbsct->LarZbs",
                            C13ED, T3F4, A6,
                            optimize=[(0,1),(0,1)], backend='torch'
                            ).reshape(chi*D_bond*D_bond, chi*D_bond*D_bond)
    # open_D= "LXbs,abci,rstj->XctLarij" → "LXbs,arbsct->XctLar"
    closed_D = oe.contract("LXbs,arbsct->XctLar",
                            T2E4, D6,
                            optimize=[(0,1)], backend='torch'
                            ).reshape(chi*D_bond*D_bond, chi*D_bond*D_bond)
    EF = torch.mm(closed_E, closed_F)
    AB = torch.mm(closed_A, closed_B)
    CD = torch.mm(closed_C, closed_D)
    TrRho_val  = oe.contract("xy,yz,zx->", EF, CD, AB, backend='torch')

    toBeDivided = torch.pow(TrRho_val.abs(), 1.0/6.0)
    T3A = T3A / toBeDivided
    T3F = T3F / toBeDivided
    T1C = T1C / toBeDivided
    T1B = T1B / toBeDivided
    T2E = T2E / toBeDivided
    T2D = T2D / toBeDivided

    """
    # Fix both magnitude and complex phase so <iPEPS|iPEPS> ≈ 1 (real).
    norm_abs = norm_val.abs()
    if torch.isfinite(norm_abs).item() and (norm_abs > torch.finfo(norm_abs.dtype).tiny).item():
        phase = norm_val / norm_abs
        T3A = T3A / phase

        scaleT = norm_abs.pow(1.0 / 6.0)
        if torch.isfinite(scaleT).item() and (scaleT > torch.finfo(scaleT.dtype).tiny).item():
            T3A = T3A / scaleT
            T3F = T3F / scaleT
            T1C = T1C / scaleT
            T1B = T1B / scaleT
            T2E = T2E / scaleT
            T2D = T2D / scaleT

    # ── Individual equalization: preserve norm_env_3=1 while balancing magnitudes ──
    nT3A = torch.linalg.norm(T3A).real
    nT3F = torch.linalg.norm(T3F).real
    nT1C = torch.linalg.norm(T1C).real
    nT1B = torch.linalg.norm(T1B).real
    nT2E = torch.linalg.norm(T2E).real
    nT2D = torch.linalg.norm(T2D).real
    nT_prod = nT3A * nT3F * nT1C * nT1B * nT2E * nT2D
    if torch.isfinite(nT_prod).item() and (nT_prod > torch.finfo(nT3A.dtype).tiny).item():
        nT_geo = nT_prod.pow(1.0 / 6.0)
        T3A = T3A * (nT_geo / nT3A)
        T3F = T3F * (nT_geo / nT3F)
        T1C = T1C * (nT_geo / nT1C)
        T1B = T1B * (nT_geo / nT1B)
        T2E = T2E * (nT_geo / nT2E)
        T2D = T2D * (nT_geo / nT2D)
    """

    return C21AF, C32CB, C13ED, T1B, T2E, T2D, T3A, T3F, T1C




def update_environmentCTs_3to1(C21AF, C32CB, C13ED, T1B, T2E, T2D, T3A, T3F, T1C, A,B,C,D,E,F, chi, D_squared):
    """
    Perform one CTMRG renormalisation step: environment type 3 → type 1.

    Absorbs site tensors ``C``, ``D``, ``E``, ``F``, ``A``, ``B`` into the
    type-3 environment and produces the type-1 set, completing the three-step
    cycle ``1→2→3→1``.

    Args:
        C21AF, C32CB, C13ED (torch.Tensor): Type-3 corner matrices,
            shape ``(chi, chi)``.
        T1B, T2E, T2D, T3A, T3F, T1C (torch.Tensor): Type-3 transfer tensors,
            shape ``(chi, D_squared, D_squared)``.
        A, B, C, D, E, F (torch.Tensor): Double-layer site tensors,
            shape ``(D_squared, D_squared)``.
        chi (int): Target bond dimension.
        D_squared (int): ``D_bond ** 2``.

    Returns:
        Tuple of 9 torch.Tensor: ``(C21CD, C32EF, C13AB, T1F, T2A, T2B,
        T3C, T3D, T1E)`` — the renewed type-1 corners and transfer tensors.
    """

    matC21CD = oe.contract("YX,MYa,LXb,amg,lbg->MmLl",
                           C21AF,T1B,T2E,C,D,
                           optimize=[(0,2),(0,3),(1,2),(0,1)],
                           backend='torch')
    
    matC21CD = matC21CD.reshape(chi*D_squared,chi*D_squared)
    
    matC32EF = oe.contract("ZY,NZb,MYg,abn,amg->NnMm",
                           C32CB,T2D,T3A,E,F,
                           optimize=[(0,2),(0,3),(1,2),(0,1)],
                           backend='torch')
    
    matC32EF = matC32EF.reshape(chi*D_squared,chi*D_squared)
    
    matC13AB = oe.contract("XZ,LXg,NZa,lbg,abn->LlNn",
                           C13ED,T3F,T1C,A,B,
                           optimize=[(0,2),(0,3),(1,2),(0,1)],
                           backend='torch')
    
    matC13AB = matC13AB.reshape(chi*D_squared,chi*D_squared)

    V2E, C21CD, U1B, V3A, C32EF, U2D, V1C, C13AB, U3F = trunc_rhoCCC(
                        matC21CD, matC32EF, matC13AB, chi, D_squared)




    T3C = oe.contract("OYa,abg,ObM->YMg",
                        T1B,C,U1B,
                        optimize=[(0,1),(0,1)],
                        backend='torch')

    T3D = oe.contract("OXb,abg,LOa->XLg",
                        T2E,D,V2E,
                        optimize=[(0,1),(0,1)],
                        backend='torch')

    T1E = oe.contract("OZb,abg,OgN->ZNa",
                        T2D,E,U2D,
                        optimize=[(0,1),(0,1)],
                        backend='torch')

    T1F = oe.contract("OYg,abg,MOb->YMa",
                        T3A,F,V3A,
                        optimize=[(0,1),(0,1)],
                        backend='torch')

    T2A = oe.contract("OXg,abg,OaL->XLb",
                        T3F,A,U3F,
                        optimize=[(0,1),(0,1)],
                        backend='torch')

    T2B = oe.contract("OZa,abg,NOg->ZNb",
                        T1C,B,V1C,
                        optimize=[(0,1),(0,1)],
                        backend='torch')
    









    T3C = T3C/torch.linalg.norm(T3C)
    T3D = T3D/torch.linalg.norm(T3D)
    T1E = T1E/torch.linalg.norm(T1E)
    T1F = T1F/torch.linalg.norm(T1F)
    T2A = T2A/torch.linalg.norm(T2A)
    T2B = T2B/torch.linalg.norm(T2B)









    # ── End-of-update transfer normalization (mirrors norm_env_1) ────────────
    # norm_env_1: open_E= "YX,MYar,abci,rstj->MbsXctij" → "YX,MYar,arbsct->MbsXct"
    closed_E = oe.contract("YX,MYa,abg->MbXg",
                            C21CD, T1F, E,
                            optimize=[(0,1),(0,1)], backend='torch'
                            ).reshape(chi*D_squared, chi*D_squared)
    # open_D= "MYct,abci,rstj->YarMbsij" → "MYct,arbsct->YarMbs"
    closed_D = oe.contract("MYg,abg->YaMb",
                            T3C, D,
                            optimize=[(0,1)], backend='torch'
                            ).reshape(chi*D_squared, chi*D_squared)
    # open_A= "ZY,NZbs,abci,rstj->NctYarij" → "ZY,NZbs,arbsct->NctYar"
    closed_A = oe.contract("ZY,NZb,abg->NgYa",
                            C32EF, T2B, A,
                            optimize=[(0,1),(0,1)], backend='torch'
                            ).reshape(chi*D_squared, chi*D_squared)
    # open_F= "NZar,abci,rstj->ZbsNctij" → "NZar,arbsct->ZbsNct"
    closed_F = oe.contract("NZa,abg->ZbNg",
                            T1E, F,
                            optimize=[(0,1)], backend='torch'
                            ).reshape(chi*D_squared, chi*D_squared)
    # open_C= "XZ,LXct,abci,rstj->LarZbsij" → "XZ,LXct,arbsct->LarZbs"
    closed_C = oe.contract("XZ,LXg,abg->LaZb",
                            C13AB, T3D, C,
                            optimize=[(0,1),(0,1)], backend='torch'
                            ).reshape(chi*D_squared, chi*D_squared)
    # open_B= "LXbs,abci,rstj->XctLarij" → "LXbs,arbsct->XctLar"
    closed_B = oe.contract("LXb,abg->XgLa",
                            T2A, B,
                            optimize=[(0,1)], backend='torch'
                            ).reshape(chi*D_squared, chi*D_squared)
    AD = torch.mm(closed_A, closed_D)
    CF = torch.mm(closed_C, closed_F)
    EB = torch.mm(closed_E, closed_B)
    TrRho_val  = oe.contract("xy,yz,zx->", AD, EB, CF, backend='torch')

    toBeDivided = torch.pow(TrRho_val.abs(), 1.0/6.0)
    T3C = T3C / toBeDivided
    T3D = T3D / toBeDivided
    T1E = T1E / toBeDivided
    T1F = T1F / toBeDivided
    T2A = T2A / toBeDivided
    T2B = T2B / toBeDivided

    """
    
    # Fix both magnitude and complex phase so <iPEPS|iPEPS> ≈ 1 (real).
    norm_abs = norm_val.abs()
    if torch.isfinite(norm_abs).item() and (norm_abs > torch.finfo(norm_abs.dtype).tiny).item():
        phase = norm_val / norm_abs
        T3C = T3C / phase

        scaleT = norm_abs.pow(1.0 / 6.0)
        if torch.isfinite(scaleT).item() and (scaleT > torch.finfo(scaleT.dtype).tiny).item():
            T3C = T3C / scaleT
            T3D = T3D / scaleT
            T1E = T1E / scaleT
            T1F = T1F / scaleT
            T2A = T2A / scaleT
            T2B = T2B / scaleT

    # ── Individual equalization: preserve norm_env_1=1 while balancing magnitudes ──
    nT3C = torch.linalg.norm(T3C).real
    nT3D = torch.linalg.norm(T3D).real
    nT1E = torch.linalg.norm(T1E).real
    nT1F = torch.linalg.norm(T1F).real
    nT2A = torch.linalg.norm(T2A).real
    nT2B = torch.linalg.norm(T2B).real
    nT_prod = nT3C * nT3D * nT1E * nT1F * nT2A * nT2B
    if torch.isfinite(nT_prod).item() and (nT_prod > torch.finfo(nT3C.dtype).tiny).item():
        nT_geo = nT_prod.pow(1.0 / 6.0)
        T3C = T3C * (nT_geo / nT3C)
        T3D = T3D * (nT_geo / nT3D)
        T1E = T1E * (nT_geo / nT1E)
        T1F = T1F * (nT_geo / nT1F)
        T2A = T2A * (nT_geo / nT2A)
        T2B = T2B * (nT_geo / nT2B)

    """

    return C21CD, C32EF, C13AB, T1F, T2A, T2B, T3C, T3D, T1E




def CTMRG_from_init_to_stop(A,B,C,D,E,F,
                            chi: int,
                            D_squared: int,
                            a_third_max_iterations: int,
                            env_conv_threshold: float,
                            identity_init: bool = False) -> tuple:
    """
    This function performs the CTMRG algorithm from the initial state to the stopping criterion.

    Args:
        A, B, C, D, E, F (torch.Tensor): The 6 local tensors.
        max_iterations (int): The maximum number of iterations to perform.
        D_squared (int): The square of the bond dimension of the local state projector a~f.
        chi (int): The nominal desired bond dimension for the transfer tensors.
        env_conv_threshold (float): The threshold for environment convergence.

    Returns:
        A tuple of 28 elements: the 27 final environment tensors followed by
        ctm_steps (int) — the number of CTMRG iterations actually performed.
    """
    # Initialize the environment corner and edge transfer tensors

    lastC21CD, lastC32EF, lastC13AB, lastT1F, lastT2A, lastT2B, lastT3C, lastT3D, lastT1E = None, None, None, None, None, None, None, None, None
    lastC21EB, lastC32AD, lastC13CF, lastT1D, lastT2C, lastT2F, lastT3E, lastT3B, lastT1A = None, None, None, None, None, None, None, None, None
    lastC21AF, lastC32CB, lastC13ED, lastT1B, lastT2E, lastT2D, lastT3A, lastT3F, lastT1C = None, None, None, None, None, None, None, None, None
    nowC21CD, nowC32EF, nowC13AB, nowT1F, nowT2A, nowT2B, nowT3C, nowT3D, nowT1E = initialize_envCTs_1(A,B,C,D,E,F, chi, D_squared, identity_init=identity_init)
    nowC21EB, nowC32AD, nowC13CF, nowT1D, nowT2C, nowT2F, nowT3E, nowT3B, nowT1A = None, None, None, None, None, None, None, None, None
    nowC21AF, nowC32CB, nowC13ED, nowT1B, nowT2E, nowT2D, nowT3A, nowT3F, nowT1C = None, None, None, None, None, None, None, None, None





    # Perform the CTMRG iterations until convergence
    ctm_steps = a_third_max_iterations  # will be overwritten on early convergence
    for iteration in range(a_third_max_iterations):

        # ── MEM-E: per-iteration env tensor footprint (first 3 iterations only) ──
        # BREAKPOINT-E  ← set breakpoint on the print line below.
        # Shows how many tensors are live and their combined size as the env grows.
        # Only fires for iterations 0–2 to avoid log spam; remove the guard to log all.
        # Debug Console queries when paused here:
        #   iteration                                   → current iteration index
        #   nowC21CD.shape if nowC21CD is not None else None  → corner shape
        #   nowT1F.shape   if nowT1F   is not None else None  → transfer shape
        #   sum(t.element_size()*t.nelement() for t in
        #       [t for t in [nowC21CD,nowC32EF,nowC13AB,nowT1F,nowT2A,nowT2B,
        #                    nowT3C,nowT3D,nowT1E,nowC21EB,nowC32AD,nowC13CF,
        #                    nowT1D,nowT2C,nowT2F,nowT3E,nowT3B,nowT1A,
        #                    nowC21AF,nowC32CB,nowC13ED,nowT1B,nowT2E,nowT2D,
        #                    nowT3A,nowT3F,nowT1C] if t is not None]) / 1e6
        if True:
            _all_now = [t for t in [
                nowC21CD, nowC32EF, nowC13AB, nowT1F,  nowT2A,  nowT2B,
                nowT3C,   nowT3D,   nowT1E,   nowC21EB, nowC32AD, nowC13CF,
                nowT1D,   nowT2C,   nowT2F,   nowT3E,   nowT3B,   nowT1A,
                nowC21AF, nowC32CB, nowC13ED,  nowT1B,  nowT2E,   nowT2D,
                nowT3A,   nowT3F,   nowT1C,
            ] if t is not None]
            _all_last = [t for t in [
                lastC21CD, lastC32EF, lastC13AB, lastT1F,  lastT2A,  lastT2B,
                lastT3C,   lastT3D,   lastT1E,   lastC21EB, lastC32AD, lastC13CF,
                lastT1D,   lastT2C,   lastT2F,   lastT3E,   lastT3B,   lastT1A,
                lastC21AF, lastC32CB, lastC13ED,  lastT1B,  lastT2E,   lastT2D,
                lastT3A,   lastT3F,   lastT1C,
            ] if t is not None]
            _now_mb  = sum(t.element_size() * t.nelement() / 1e6 for t in _all_now)
            _last_mb = sum(t.element_size() * t.nelement() / 1e6 for t in _all_last)
            _c_sh = tuple(nowC21CD.shape) if nowC21CD is not None else None
            _t_sh = tuple(nowT1F.shape)   if nowT1F   is not None else None
            #print(f"[MEM-E] RSS={mem_mb():.1f}MB CTMRG iter={iteration}  "
            #      f"now={len(_all_now)}tensors/{_now_mb:.1f}MB  "
            #      f"last={len(_all_last)}tensors/{_last_mb:.1f}MB  "
            #      f" T-shape={_t_sh}")

        if check_env_CV_using_3rho(lastC21CD, lastC32EF, lastC13AB, #lastT1F, lastT2A, lastT2B, lastT3C, lastT3D, lastT1E, 
                                 nowC21CD, nowC32EF, nowC13AB, #nowT1F, nowT2A, nowT2B, nowT3C, nowT3D, nowT1E, 
                                 lastC21EB, lastC32AD, lastC13CF, #lastT1D, lastT2C, lastT2F, lastT3E, lastT3B, lastT1A, 
                                 nowC21EB, nowC32AD, nowC13CF, #nowT1D, nowT2C, nowT2F, nowT3E, nowT3B, nowT1A, 
                                 lastC21AF, lastC32CB, lastC13ED, #lastT1B, lastT2E, lastT2D, lastT3A, lastT3F, lastT1C, 
                                 nowC21AF, nowC32CB, nowC13ED, #nowT1B, nowT2E, nowT2D, nowT3A, nowT3F, nowT1C, 
                                 env_conv_threshold):
            ctm_steps = iteration + 1
            break

        # Update the environment corner and edge transfer tensors
        # match iteration % 3 :
        #     case 0 : 
        lastC21EB, lastC32AD, lastC13CF, lastT1D, lastT2C, lastT2F, lastT3E, lastT3B, lastT1A = \
        nowC21EB, nowC32AD, nowC13CF, nowT1D, nowT2C, nowT2F, nowT3E, nowT3B, nowT1A
        nowC21EB, nowC32AD, nowC13CF, nowT1D, nowT2C, nowT2F, nowT3E, nowT3B, nowT1A = update_environmentCTs_1to2(
        nowC21CD, nowC32EF, nowC13AB, nowT1F, nowT2A, nowT2B, nowT3C, nowT3D, nowT1E, A,B,C,D,E,F, chi, D_squared)
            
        #     case 1 : 
        lastC21AF, lastC32CB, lastC13ED, lastT1B, lastT2E, lastT2D, lastT3A, lastT3F, lastT1C = \
        nowC21AF, nowC32CB, nowC13ED, nowT1B, nowT2E, nowT2D, nowT3A, nowT3F, nowT1C

        nowC21AF, nowC32CB, nowC13ED, nowT1B, nowT2E, nowT2D, nowT3A, nowT3F, nowT1C = update_environmentCTs_2to3(
        nowC21EB, nowC32AD, nowC13CF, nowT1D, nowT2C, nowT2F, nowT3E, nowT3B, nowT1A, A,B,C,D,E,F, chi, D_squared)
            
        #     case 2 : 
        lastC21CD, lastC32EF, lastC13AB, lastT1F, lastT2A, lastT2B, lastT3C, lastT3D, lastT1E = \
        nowC21CD, nowC32EF, nowC13AB, nowT1F, nowT2A, nowT2B, nowT3C, nowT3D, nowT1E
        
        nowC21CD, nowC32EF, nowC13AB, nowT1F, nowT2A, nowT2B, nowT3C, nowT3D, nowT1E = update_environmentCTs_3to1(
        nowC21AF, nowC32CB, nowC13ED, nowT1B, nowT2E, nowT2D, nowT3A, nowT3F, nowT1C, A,B,C,D,E,F, chi, D_squared)
    
    return  nowC21CD, nowC32EF, nowC13AB, nowT1F, nowT2A, nowT2B, nowT3C, nowT3D, nowT1E, \
            nowC21EB, nowC32AD, nowC13CF, nowT1D, nowT2C, nowT2F, nowT3E, nowT3B, nowT1A, \
            nowC21AF, nowC32CB, nowC13ED, nowT1B, nowT2E, nowT2D, nowT3A, nowT3F, nowT1C, \
            ctm_steps







def energy_expectation_nearest_neighbor_6_bonds(a,b,c,d,e,f, 
                                                Hed,Had,Haf,Hcf,Hcb,Heb, # (d_PHYS * d_PHYS^*, d_PHYS * d_PHYS^*) matrices
                                                chi, D_bond, # d_PHYS,
                                                C21CD,C32EF,C13AB,T1F,T2A,T2B,T3C,T3D,T1E):
    """
    Compute the variational energy expectation value for the 6 bonds in the
    type-1 environment.

    The six nearest-neighbour bonds covered here are E-D, A-D, F-A, C-F,
    B-C, E-B (the bonds associated with the type-1 corner/transfer
    environment).  For each bond, an ``open`` tensor is built by contracting
    the single-layer tensor with its conjugate while leaving the physical
    indices un-contracted (the ``ij`` dangling pair).  A ``closed`` version
    (trace over ``ij``) gives the denominator norm contributions.  The
    numerator for bond ``XY`` is obtained by inserting the Hamiltonian
    ``H_XY[i,j,k,l]`` between the open physical legs.

    The final energy is::

        E_6 = (sum over 6 bonds of E_unnormed) / norm_1st_env

    where ``norm_1st_env = Tr[DE * BC * FA]`` (cyclic product of the three
    closed two-site operators around the unit cell).

    Args:
        a, b, c, d, e, f (torch.Tensor): Single-layer site tensors,
            shape ``(D_bond, D_bond, D_bond, d_PHYS)``, with
            ``requires_grad=True`` for the optimisation.
        Hed, Had, Haf, Hcf, Hcb, Heb (torch.Tensor): Two-site nearest-neighbour
            Hamiltonian matrices, shape ``(d_PHYS, d_PHYS, d_PHYS, d_PHYS)``
            (bra1, bra2, ket1, ket2 ordering).
        chi (int): Environment bond dimension.
        D_bond (int): Virtual bond dimension.
        C21CD, C32EF, C13AB (torch.Tensor): Type-1 corner matrices,
            shape ``(chi, chi)``.
        T1F, T2A, T2B, T3C, T3D, T1E (torch.Tensor): Type-1 transfer tensors,
            shape ``(chi*D_bond*D_bond, chi*D_bond*D_bond)`` before the internal
            reshape.

    Returns:
        torch.Tensor: Scalar energy per bond (complex64; imaginary part should
        vanish for a Hermitian Hamiltonian).
    """
    
    T1F = T1F.reshape(chi,chi,D_bond,D_bond)
    T2A = T2A.reshape(chi,chi,D_bond,D_bond)
    T2B = T2B.reshape(chi,chi,D_bond,D_bond)
    T3C = T3C.reshape(chi,chi,D_bond,D_bond)
    T3D = T3D.reshape(chi,chi,D_bond,D_bond)
    T1E = T1E.reshape(chi,chi,D_bond,D_bond)

    open_E = oe.contract("YX,MYar,abci,rstj->MbsXctij", C21CD, T1F, e, e.conj(), optimize=[(0,1),(0,1),(0,1)], backend="torch")
    open_D = oe.contract("MYct,abci,rstj->YarMbsij", T3C, d, d.conj(), optimize=[(0,1),(0,1)], backend="torch")
    open_A = oe.contract("ZY,NZbs,abci,rstj->NctYarij", C32EF, T2B, a, a.conj(), optimize=[(0,1),(0,1),(0,1)], backend="torch")
    open_F = oe.contract("NZar,abci,rstj->ZbsNctij", T1E, f, f.conj(), optimize=[(0,1),(0,1)], backend="torch")
    open_C = oe.contract("XZ,LXct,abci,rstj->LarZbsij", C13AB, T3D, c, c.conj(), optimize=[(0,1),(0,1),(0,1)], backend="torch")
    open_B = oe.contract("LXbs,abci,rstj->XctLarij", T2A, b, b.conj(), optimize=[(0,1),(0,1)], backend="torch")
                              
    closed_E = oe.contract("MbsXctii->MbsXct", open_E, backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    closed_D = oe.contract("YarMbsii->YarMbs", open_D, backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    closed_A = oe.contract("NctYarii->NctYar", open_A, backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    closed_F = oe.contract("ZbsNctii->ZbsNct", open_F, backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    closed_C = oe.contract("LarZbsii->LarZbs", open_C, backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    closed_B = oe.contract("XctLarii->XctLar", open_B, backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)

    H_DE = oe.contract("MbsXctij,ijkl,YarMbskl->YarXct", open_E, Hed, open_D, optimize=[(0,1),(0,1)], backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    H_AD = oe.contract("NctYarij,ijkl,YarMbskl->NctMbs", open_A, Had, open_D, optimize=[(0,1),(0,1)], backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    H_FA = oe.contract("NctYarij,ijkl,ZbsNctkl->ZbsYar", open_A, Haf, open_F, optimize=[(0,1),(0,1)], backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    H_CF = oe.contract("LarZbsij,ijkl,ZbsNctkl->LarNct", open_C, Hcf, open_F, optimize=[(0,1),(0,1)], backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    H_BC = oe.contract("LarZbsij,ijkl,XctLarkl->XctZbs", open_C, Hcb, open_B, optimize=[(0,1),(0,1)], backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    H_EB = oe.contract("MbsXctij,ijkl,XctLarkl->MbsLar", open_E, Heb, open_B, optimize=[(0,1),(0,1)], backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)

    DE = torch.mm(closed_D, closed_E)
    AD = torch.mm(closed_A, closed_D)
    FA = torch.mm(closed_F, closed_A)
    CF = torch.mm(closed_C, closed_F)
    BC = torch.mm(closed_B, closed_C)
    EB = torch.mm(closed_E, closed_B)

    E_unnormed_DE = oe.contract("xy,yz,zx->", H_DE, BC, FA, backend="torch")
    E_unnormed_AD = oe.contract("xy,yz,zx->", H_AD, EB, CF, backend="torch")
    E_unnormed_FA = oe.contract("xy,yz,zx->", H_FA, DE, BC, backend="torch")
    E_unnormed_CF = oe.contract("xy,yz,zx->", H_CF, AD, EB, backend="torch")
    E_unnormed_BC = oe.contract("xy,yz,zx->", H_BC, FA, DE, backend="torch")
    E_unnormed_EB = oe.contract("xy,yz,zx->", H_EB, CF, AD, backend="torch")

    norm_1st_env = oe.contract("xy,yz,zx->", DE, BC, FA, backend="torch")
    energyNearestNeighbor_6_bonds = (E_unnormed_DE + E_unnormed_AD + E_unnormed_FA + E_unnormed_CF + E_unnormed_BC + E_unnormed_EB) / norm_1st_env
    return energyNearestNeighbor_6_bonds





def energy_expectation_nearest_neighbor_3ebadcf_bonds(
                a,b,c,d,e,f, 
                Heb,Had,Hcf,
                chi, D_bond, d_PHYS, 
                C21CD,C32EF,C13AB,T1F,T2A,T2B,T3C,T3D,T1E):
    
    T1F = T1F.reshape(chi,chi,D_bond,D_bond)
    T2A = T2A.reshape(chi,chi,D_bond,D_bond)
    T2B = T2B.reshape(chi,chi,D_bond,D_bond)
    T3C = T3C.reshape(chi,chi,D_bond,D_bond)
    T3D = T3D.reshape(chi,chi,D_bond,D_bond)
    T1E = T1E.reshape(chi,chi,D_bond,D_bond)

    open_E = oe.contract("YX,MYar,abci,rstj->MbsXctij", C21CD, T1F, e, e.conj(), optimize=[(0,1),(0,1),(0,1)], backend="torch")
    open_D = oe.contract("MYct,abci,rstj->YarMbsij", T3C, d, d.conj(), optimize=[(0,1),(0,1)], backend="torch")
    open_A = oe.contract("ZY,NZbs,abci,rstj->NctYarij", C32EF, T2B, a, a.conj(), optimize=[(0,1),(0,1),(0,1)], backend="torch")
    open_F = oe.contract("NZar,abci,rstj->ZbsNctij", T1E, f, f.conj(), optimize=[(0,1),(0,1)], backend="torch")
    open_C = oe.contract("XZ,LXct,abci,rstj->LarZbsij", C13AB, T3D, c, c.conj(), optimize=[(0,1),(0,1),(0,1)], backend="torch")
    open_B = oe.contract("LXbs,abci,rstj->XctLarij", T2A, b, b.conj(), optimize=[(0,1),(0,1)], backend="torch")
                              
    closed_E = oe.contract("MbsXctii->MbsXct", open_E, backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    closed_D = oe.contract("YarMbsii->YarMbs", open_D, backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    closed_A = oe.contract("NctYarii->NctYar", open_A, backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    closed_F = oe.contract("ZbsNctii->ZbsNct", open_F, backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    closed_C = oe.contract("LarZbsii->LarZbs", open_C, backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    closed_B = oe.contract("XctLarii->XctLar", open_B, backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)

    # H_AD = oe.contract("NctYarij,ijkl,YarMbskl->NctMbs", open_A, Had, open_D, optimize=[(0,1),(0,1)], backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    # H_CF = oe.contract("LarZbsij,ijkl,ZbsNctkl->LarNct", open_C, Hcf, open_F, optimize=[(0,1),(0,1)], backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    # H_EB = oe.contract("MbsXctij,ijkl,XctLarkl->MbsLar", open_E, Heb, open_B, optimize=[(0,1),(0,1)], backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)


    AD = torch.mm(closed_A, closed_D)
    CF = torch.mm(closed_C, closed_F)
    EB = torch.mm(closed_E, closed_B)

    # E_unnormed_AD = oe.contract("xy,yz,zx->", H_AD, EB, CF, backend="torch")
    # E_unnormed_CF = oe.contract("xy,yz,zx->", H_CF, AD, EB, backend="torch")
    # E_unnormed_EB = oe.contract("xy,yz,zx->", H_EB, CF, AD, backend="torch")


    # norm_1st_env = oe.contract("xy,yz,zx->", AD, EB, CF, backend="torch")
    
    # """ manually hermitianize!!!

    rho = oe.contract("NctYarij,YarMbskl->ikjlNctMbs", open_A, open_D, backend="torch").reshape(d_PHYS*d_PHYS,d_PHYS*d_PHYS,chi*D_bond*D_bond,chi*D_bond*D_bond)
    rhoAD = oe.contract('IJxy,yz,zx->IJ', rho, EB, CF, backend="torch")
    #to_print = torch.norm(rhoAD - rhoAD.conj().T).item()/2.0/torch.norm(rhoAD).item()
    #if to_print > 1e-4: print("rhoAD anti-hermiticity: ", to_print)
    theOriTrace = torch.trace(rhoAD)
    rhoAD = (rhoAD + rhoAD.conj().T)/2.0
    # positive-semidefinite projection:
    eigvals, eigvecs = torch.linalg.eigh(rhoAD)
    #print("rhoAD eigvals before clipping: ", eigvals.detach().numpy())
    eigvals_clipped = torch.clamp(eigvals, min=0.0)
    rhoAD = (eigvecs * eigvals_clipped) @ eigvecs.conj().T
    theTrace = eigvals_clipped.sum().clamp(min=1e-30)  # real scalar → safe real division
    rhoAD = (rhoAD / theTrace).reshape(d_PHYS,d_PHYS,d_PHYS,d_PHYS)
    E_AD = oe.contract("ikjl,ijkl->", rhoAD, Had, backend="torch")
    #if to_print > 1e-4: print("E_AD = ", E_AD.item(), "trace rhoAD = ", theTrace.item(), "(ori:", theOriTrace.item(), ")")

    
    rho = oe.contract("NctYarij,YarMbskl->ikjlNctMbs", open_C, open_F, backend="torch").reshape(d_PHYS*d_PHYS,d_PHYS*d_PHYS,chi*D_bond*D_bond,chi*D_bond*D_bond)
    rhoCF = oe.contract('IJxy,yz,zx->IJ', rho, AD, EB, backend="torch")
    #to_print = torch.norm(rhoCF - rhoCF.conj().T).item()/2.0/torch.norm(rhoCF).item()
    #if to_print > 1e-4: print("rhoCF anti-hermiticity: ", to_print)
    theOriTrace = torch.trace(rhoCF)
    rhoCF = (rhoCF + rhoCF.conj().T)/2.0
    # positive-semidefinite projection:
    eigvals, eigvecs = torch.linalg.eigh(rhoCF)
    #print("rhoCF eigvals before clipping: ", eigvals.detach().numpy())
    eigvals_clipped = torch.clamp(eigvals, min=0.0)
    rhoCF = (eigvecs * eigvals_clipped) @ eigvecs.conj().T  # PSD reconstruction (was accidentally commented out)
    theTrace = eigvals_clipped.sum().clamp(min=1e-30)  # real scalar → safe real division
    rhoCF = (rhoCF / theTrace).reshape(d_PHYS,d_PHYS,d_PHYS,d_PHYS)
    E_CF = oe.contract("ikjl,ijkl->", rhoCF, Hcf, backend="torch")
    #if to_print > 1e-4: print("E_CF = ", E_CF.item(), "trace rhoCF = ", theTrace.item(), "(ori:", theOriTrace.item(), ")")


    rho = oe.contract("NctYarij,YarMbskl->ikjlNctMbs", open_E, open_B, backend="torch").reshape(d_PHYS*d_PHYS,d_PHYS*d_PHYS,chi*D_bond*D_bond,chi*D_bond*D_bond)
    rhoEB = oe.contract('IJxy,yz,zx->IJ', rho, CF, AD, backend="torch")
    #to_print = torch.norm(rhoEB - rhoEB.conj().T).item()/2.0/torch.norm(rhoEB).item()
    #if to_print > 1e-4: print("rhoEB anti-hermiticity: ", to_print)
    theOriTrace = torch.trace(rhoEB)
    rhoEB = (rhoEB + rhoEB.conj().T)/2.0
    # positive-semidefinite projection:
    eigvals, eigvecs = torch.linalg.eigh(rhoEB)
    #print("rhoEB eigvals before clipping: ", eigvals.detach().numpy())
    eigvals_clipped = torch.clamp(eigvals, min=0.0)
    rhoEB = (eigvecs * eigvals_clipped) @ eigvecs.conj().T
    theTrace = eigvals_clipped.sum().clamp(min=1e-30)  # real scalar → safe real division
    rhoEB = (rhoEB / theTrace).reshape(d_PHYS,d_PHYS,d_PHYS,d_PHYS)
    E_EB = oe.contract("ikjl,ijkl->", rhoEB, Heb, backend="torch")
    #if to_print > 1e-4: print("E_EB = ", E_EB.item(), "trace rhoEB = ", theTrace.item(), "(ori:", theOriTrace.item(), ")")


    # """
    
    
    # compare_energy = (torch.abs(E_unnormed_AD)+torch.abs(E_unnormed_CF)+torch.abs(E_unnormed_EB)) / torch.abs(norm_1st_env)
    # print("E_unnormed_AD = ", E_unnormed_AD.real.item(),"+i*", E_unnormed_AD.imag.item())
    # print("E_unnormed_CF = ", E_unnormed_CF.real.item(),"+i*", E_unnormed_CF.imag.item())
    # print("E_unnormed_EB = ", E_unnormed_EB.real.item(),"+i*", E_unnormed_EB.imag.item())
    # print("norm_1st_env = ", norm_1st_env.real.item(),"+i*", norm_1st_env.imag.item())
    # print("energyNearestNeighbor_3_bonds = ", energyNearestNeighbor_3_bonds.item())
    # print("compare_energy = ", compare_energy.item())
    
    return torch.real(E_AD + E_CF + E_EB)


def energy_expectation_nearest_neighbor_3afcbed_bonds(a,b,c,d,e,f,Haf,Hcb,Hed, 
                chi, D_bond, d_PHYS, 
                C21EB, C32AD,C13CF,T1D,T2C,T2F,T3E,T3B,T1A):

    T1D = T1D.reshape(chi,chi,D_bond,D_bond)
    T2C = T2C.reshape(chi,chi,D_bond,D_bond)
    T2F = T2F.reshape(chi,chi,D_bond,D_bond)
    T3E = T3E.reshape(chi,chi,D_bond,D_bond)
    T3B = T3B.reshape(chi,chi,D_bond,D_bond)
    T1A = T1A.reshape(chi,chi,D_bond,D_bond)

    open_A = oe.contract("YX,MYar,abci,rstj->MbsXctij", C21EB, T1D, a, a.conj(), optimize=[(0,1),(0,1),(0,1)], backend="torch")
    open_B = oe.contract("MYct,abci,rstj->YarMbsij", T3E, b, b.conj(), optimize=[(0,1),(0,1)], backend="torch")
    open_C = oe.contract("ZY,NZbs,abci,rstj->NctYarij", C32AD, T2F, c, c.conj(), optimize=[(0,1),(0,1),(0,1)], backend="torch")
    open_D = oe.contract("NZar,abci,rstj->ZbsNctij", T1A, d, d.conj(), optimize=[(0,1),(0,1)], backend="torch")
    open_E = oe.contract("XZ,LXct,abci,rstj->LarZbsij", C13CF, T3B, e, e.conj(), optimize=[(0,1),(0,1),(0,1)], backend="torch")
    open_F = oe.contract("LXbs,abci,rstj->XctLarij", T2C, f, f.conj(), optimize=[(0,1),(0,1)], backend="torch")
                              
    closed_A = oe.contract("MbsXctii->MbsXct", open_A, backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    closed_B = oe.contract("YarMbsii->YarMbs", open_B, backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    closed_C = oe.contract("NctYarii->NctYar", open_C, backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    closed_D = oe.contract("ZbsNctii->ZbsNct", open_D, backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    closed_E = oe.contract("LarZbsii->LarZbs", open_E, backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    closed_F = oe.contract("XctLarii->XctLar", open_F, backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)

    #H_CB = oe.contract("NctYarij,ijkl,YarMbskl->NctMbs", open_C, Hcb, open_B, optimize=[(0,1),(0,1)], backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    #H_ED = oe.contract("LarZbsij,ijkl,ZbsNctkl->LarNct", open_E, Hed, open_D, optimize=[(0,1),(0,1)], backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    #H_AF = oe.contract("MbsXctij,ijkl,XctLarkl->MbsLar", open_A, Haf, open_F, optimize=[(0,1),(0,1)], backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)

    CB = torch.mm(closed_C, closed_B)
    ED = torch.mm(closed_E, closed_D)
    AF = torch.mm(closed_A, closed_F)



    rho = oe.contract("NctYarij,YarMbskl->ikjlNctMbs", open_C, open_B, backend="torch").reshape(d_PHYS*d_PHYS,d_PHYS*d_PHYS,chi*D_bond*D_bond,chi*D_bond*D_bond)
    rhoCB = oe.contract('IJxy,yz,zx->IJ', rho, AF, ED, backend="torch")
    #to_print = torch.norm(rhoCB - rhoCB.conj().T).item()/2.0/torch.norm(rhoCB).item()
    #if to_print > 1e-4: print("rhoCB anti-hermiticity: ", to_print)
    theOriTrace = torch.trace(rhoCB)
    rhoCB = (rhoCB + rhoCB.conj().T)/2.0
    # positive-semidefinite projection:
    eigvals, eigvecs = torch.linalg.eigh(rhoCB)
    #print("rhoCB eigvals before clipping: ", eigvals.detach().numpy())
    eigvals_clipped = torch.clamp(eigvals, min=0.0)
    rhoCB = (eigvecs * eigvals_clipped) @ eigvecs.conj().T
    theTrace = eigvals_clipped.sum().clamp(min=1e-30)  # real scalar → safe real division
    rhoCB = (rhoCB / theTrace).reshape(d_PHYS,d_PHYS,d_PHYS,d_PHYS)
    E_CB = oe.contract("ikjl,ijkl->", rhoCB, Hcb, backend="torch")
    #if to_print > 1e-4: print("E_CB = ", E_CB.item(), "trace rhoCB = ", theTrace.item(), "(ori:", theOriTrace.item(), ")")

    
    rho = oe.contract("NctYarij,YarMbskl->ikjlNctMbs", open_A, open_F, backend="torch").reshape(d_PHYS*d_PHYS,d_PHYS*d_PHYS,chi*D_bond*D_bond,chi*D_bond*D_bond)
    rhoAF = oe.contract('IJxy,yz,zx->IJ', rho, ED, CB, backend="torch")
    #to_print = torch.norm(rhoAF - rhoAF.conj().T).item()/2.0/torch.norm(rhoAF).item()
    #if to_print > 1e-4: print("rhoAF anti-hermiticity: ", to_print)
    theOriTrace = torch.trace(rhoAF)
    rhoAF = (rhoAF + rhoAF.conj().T)/2.0
    # positive-semidefinite projection:
    eigvals, eigvecs = torch.linalg.eigh(rhoAF)
    #print("rhoAF eigvals before clipping: ", eigvals.detach().numpy())
    eigvals_clipped = torch.clamp(eigvals, min=0.0)
    rhoAF = (eigvecs * eigvals_clipped) @ eigvecs.conj().T
    theTrace = eigvals_clipped.sum().clamp(min=1e-30)  # real scalar → safe real division
    rhoAF = (rhoAF / theTrace).reshape(d_PHYS,d_PHYS,d_PHYS,d_PHYS)
    E_AF = oe.contract("ikjl,ijkl->", rhoAF, Haf, backend="torch")
    #if to_print > 1e-4: print("E_AF = ", E_AF.item(), "trace rhoAF = ", theTrace.item(), "(ori:", theOriTrace.item(), ")")


    rho = oe.contract("NctYarij,YarMbskl->ikjlNctMbs", open_E, open_D, backend="torch").reshape(d_PHYS*d_PHYS,d_PHYS*d_PHYS,chi*D_bond*D_bond,chi*D_bond*D_bond)
    rhoED = oe.contract('IJxy,yz,zx->IJ', rho, CB, AF, backend="torch")
    #to_print = torch.norm(rhoED - rhoED.conj().T).item()/2.0/torch.norm(rhoED).item()
    #if to_print > 1e-4: print("rhoED anti-hermiticity: ", to_print)
    theOriTrace = torch.trace(rhoED)
    rhoED = (rhoED + rhoED.conj().T)/2.0
    # positive-semidefinite projection:
    eigvals, eigvecs = torch.linalg.eigh(rhoED)
    #print("rhoED eigvals before clipping: ", eigvals.detach().numpy())
    eigvals_clipped = torch.clamp(eigvals, min=0.0)
    rhoED = (eigvecs * eigvals_clipped) @ eigvecs.conj().T
    theTrace = eigvals_clipped.sum().clamp(min=1e-30)  # real scalar → safe real division
    rhoED = (rhoED / theTrace).reshape(d_PHYS,d_PHYS,d_PHYS,d_PHYS)
    E_ED = oe.contract("ikjl,ijkl->", rhoED, Hed, backend="torch")
    #if to_print > 1e-4: print("E_ED = ", E_ED.item(), "trace rhoED = ", theTrace.item(), "(ori:", theOriTrace.item(), ")")


    
    return torch.real(E_AF + E_CB + E_ED)





def energy_expectation_nearest_neighbor_other_3_bonds(a,b,c,d,e,f, 
                                                      Hcd,Hef,Hab, # (d_PHYS, d_PHYS^*, d_PHYS, d_PHYS^*) matrices
                                                      chi, D_bond, d_PHYS,
                                                      C21AF,C32CB,C13ED,T1B,T2E,T2D,T3A,T3F,T1C):
    """
    Compute the variational energy expectation value for the remaining 3 bonds
    in the type-3 environment.

    This function covers bonds C-D, E-F, A-B (the bonds not handled by
    ``energy_expectation_nearest_neighbor_6_bonds``).  The same open/closed
    tensor construction is used, but the environment here is the type-3 set
    ``(C21AF, C32CB, C13ED, T1B, T2E, T2D, T3A, T3F, T1C)``.

    The energy is::

        E_3 = (E_EF + E_AB + E_CD) / norm_3rd_env

    where ``norm_3rd_env = Tr[EF * CD * AB]``.

    Args:
        a, b, c, d, e, f (torch.Tensor): Single-layer site tensors.
        Hcd, Hef, Hab (torch.Tensor): Two-site Hamiltonians for the three
            remaining bonds, shape ``(d_PHYS, d_PHYS, d_PHYS, d_PHYS)``.
        chi (int): Environment bond dimension.
        D_bond (int): Virtual bond dimension.
        C21AF, C32CB, C13ED (torch.Tensor): Type-3 corner matrices,
            shape ``(chi, chi)``.
        T1B, T2E, T2D, T3A, T3F, T1C (torch.Tensor): Type-3 transfer tensors.

    Returns:
        torch.Tensor: Scalar energy per bond (complex64).
    """
    T1B = T1B.reshape(chi,chi,D_bond,D_bond)
    T2E = T2E.reshape(chi,chi,D_bond,D_bond)
    T2D = T2D.reshape(chi,chi,D_bond,D_bond)
    T3A = T3A.reshape(chi,chi,D_bond,D_bond)
    T3F = T3F.reshape(chi,chi,D_bond,D_bond)
    T1C = T1C.reshape(chi,chi,D_bond,D_bond)

    open_C = oe.contract("YX,MYar,abci,rstj->MbsXctij", C21AF, T1B, c, c.conj(), optimize=[(0,1),(0,1),(0,1)], backend="torch")
    open_F = oe.contract("MYct,abci,rstj->YarMbsij", T3A, f, f.conj(), optimize=[(0,1),(0,1)], backend="torch")
    open_E = oe.contract("ZY,NZbs,abci,rstj->NctYarij", C32CB, T2D, e, e.conj(), optimize=[(0,1),(0,1),(0,1)], backend="torch")
    open_B = oe.contract("NZar,abci,rstj->ZbsNctij", T1C, b, b.conj(), optimize=[(0,1),(0,1)], backend="torch")
    open_A = oe.contract("XZ,LXct,abci,rstj->LarZbsij", C13ED, T3F, a, a.conj(), optimize=[(0,1),(0,1),(0,1)], backend="torch")
    open_D = oe.contract("LXbs,abci,rstj->XctLarij", T2E, d, d.conj(), optimize=[(0,1),(0,1)], backend="torch")
                              
    closed_C = oe.contract("MbsXctii->MbsXct", open_C, backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    closed_F = oe.contract("YarMbsii->YarMbs", open_F, backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    closed_E = oe.contract("NctYarii->NctYar", open_E, backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    closed_B = oe.contract("ZbsNctii->ZbsNct", open_B, backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    closed_A = oe.contract("LarZbsii->LarZbs", open_A, backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    closed_D = oe.contract("XctLarii->XctLar", open_D, backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)

    #H_EF = oe.contract("NctYarij,ijkl,YarMbskl->NctMbs", open_E, Hef, open_F, optimize=[(0,1),(0,1)], backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    #H_AB = oe.contract("LarZbsij,ijkl,ZbsNctkl->LarNct", open_A, Hab, open_B, optimize=[(0,1),(0,1)], backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    #H_CD = oe.contract("MbsXctij,ijkl,XctLarkl->MbsLar", open_C, Hcd, open_D, optimize=[(0,1),(0,1)], backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)

    EF = torch.mm(closed_E, closed_F)
    AB = torch.mm(closed_A, closed_B)
    CD = torch.mm(closed_C, closed_D)


    rho = oe.contract("NctYarij,YarMbskl->ikjlNctMbs", open_E, open_F, backend="torch").reshape(d_PHYS*d_PHYS,d_PHYS*d_PHYS,chi*D_bond*D_bond,chi*D_bond*D_bond)
    rhoEF = oe.contract('IJxy,yz,zx->IJ', rho, CD, AB, backend="torch")
    #to_print = torch.norm(rhoEF - rhoEF.conj().T).item()/2.0/torch.norm(rhoEF).item()
    #if to_print > 1e-4: print("rhoEF anti-hermiticity: ", to_print)
    theOriTrace = torch.trace(rhoEF)
    rhoEF = (rhoEF + rhoEF.conj().T)/2.0
    # positive-semidefinite projection:
    eigvals, eigvecs = torch.linalg.eigh(rhoEF)
    #print("rhoEF eigvals before clipping: ", eigvals.detach().numpy())
    eigvals_clipped = torch.clamp(eigvals, min=0.0)
    rhoEF = (eigvecs * eigvals_clipped) @ eigvecs.conj().T
    theTrace = eigvals_clipped.sum().clamp(min=1e-30)  # real scalar → safe real division
    rhoEF = (rhoEF / theTrace).reshape(d_PHYS,d_PHYS,d_PHYS,d_PHYS)
    E_EF = oe.contract("ikjl,ijkl->", rhoEF, Hef, backend="torch")
    #if to_print > 1e-4: print("E_EF = ", E_EF.item(), "trace rhoEF = ", theTrace.item(), "(ori:", theOriTrace.item(), ")")

    
    rho = oe.contract("NctYarij,YarMbskl->ikjlNctMbs", open_A, open_B, backend="torch").reshape(d_PHYS*d_PHYS,d_PHYS*d_PHYS,chi*D_bond*D_bond,chi*D_bond*D_bond)
    rhoAB = oe.contract('IJxy,yz,zx->IJ', rho, EF, CD, backend="torch")
    #to_print = torch.norm(rhoAB - rhoAB.conj().T).item()/2.0/torch.norm(rhoAB).item()
    #if to_print > 1e-4: print("rhoAB anti-hermiticity: ", to_print)
    theOriTrace = torch.trace(rhoAB)
    rhoAB = (rhoAB + rhoAB.conj().T)/2.0
    # positive-semidefinite projection:
    eigvals, eigvecs = torch.linalg.eigh(rhoAB)
    #print("rhoAB eigvals before clipping: ", eigvals.detach().numpy())
    eigvals_clipped = torch.clamp(eigvals, min=0.0)
    rhoAB = (eigvecs * eigvals_clipped) @ eigvecs.conj().T
    theTrace = eigvals_clipped.sum().clamp(min=1e-30)  # real scalar → safe real division
    rhoAB = (rhoAB / theTrace).reshape(d_PHYS,d_PHYS,d_PHYS,d_PHYS)
    E_AB = oe.contract("ikjl,ijkl->", rhoAB, Hab, backend="torch")
    #if to_print > 1e-4: print("E_AB = ", E_AB.item(), "trace rhoAB = ", theTrace.item(), "(ori:", theOriTrace.item(), ")")


    rho = oe.contract("NctYarij,YarMbskl->ikjlNctMbs", open_C, open_D, backend="torch").reshape(d_PHYS*d_PHYS,d_PHYS*d_PHYS,chi*D_bond*D_bond,chi*D_bond*D_bond)
    rhoCD = oe.contract('IJxy,yz,zx->IJ', rho, AB, EF, backend="torch")
    #to_print = torch.norm(rhoCD - rhoCD.conj().T).item()/2.0/torch.norm(rhoCD).item()
    #if to_print > 1e-4: print("rhoCD anti-hermiticity: ", to_print)
    theOriTrace = torch.trace(rhoCD)
    rhoCD = (rhoCD + rhoCD.conj().T)/2.0
    # positive-semidefinite projection:
    eigvals, eigvecs = torch.linalg.eigh(rhoCD)
    #print("rhoCD eigvals before clipping: ", eigvals.detach().numpy())
    eigvals_clipped = torch.clamp(eigvals, min=0.0)
    rhoCD = (eigvecs * eigvals_clipped) @ eigvecs.conj().T
    theTrace = eigvals_clipped.sum().clamp(min=1e-30)  # real scalar → safe real division
    rhoCD = (rhoCD / theTrace).reshape(d_PHYS,d_PHYS,d_PHYS,d_PHYS)
    E_CD = oe.contract("ikjl,ijkl->", rhoCD, Hcd, backend="torch")
    #if to_print > 1e-4: print("E_CD = ", E_CD.item(), "trace rhoCD = ", theTrace.item(), "(ori:", theOriTrace.item(), ")")
    

    return torch.real(E_EF + E_AB + E_CD)




def optmization_iPEPS(Hed,Had,Haf,Hcf,Hcb,Heb,Hcd,Hef,Hab, # (d_PHYS, d_PHYS^*, d_PHYS, d_PHYS^*) matrices
                      opt_conv_threshold: float = 1e-6, # SCALES with Hamiltonian!!!!
                      chi: int=10, D_bond: int=3, d_PHYS: int=2,
                      a_third_max_steps_CTMRG: int = 70, 
                      CTM_env_conv_threshold: float = 1e-7,
                      a2f_initialize_way: str = 'random',
                      a2f_noise_scale: float = 1e-3,
                      max_opt_steps: int = 200,
                      lbfgs_max_iter: int = 20,
                      lbfgs_lr: float = 1.0,
                      lbfgs_history: int = 100,
                      opt_tolerance_grad: float = 1e-7,
                      opt_tolerance_change: float = 1e-8,
                      init_abcdef=None,
                      identity_init=False):
    """


    [UNUSED!!!! 
      MANY OF ITS COMPONENTS ARE NOW OUTDATED AND IN NEED OF REWORKING;
      STILL RETAINED AS A REFERENCE FOR THE OPTIMISATION STRATEGY]


    Optimize the iPEPS tensors a,b,c,d,e,f using L-BFGS.

    Strategy (standard for AD-CTMRG):
      • Outer loop  (max_opt_steps iterations):
          1. Recompute double-layer tensors A..F and run CTMRG to convergence
             inside torch.no_grad() — environment is treated as fixed.
          2. Run L-BFGS (up to lbfgs_max_iter sub-steps with strong-Wolfe line
             search).  The closure recomputes only the cheap energy evaluation
             and calls backward() through a..f; gradients do NOT flow back
             through the CTMRG iterations.
          3. Check outer convergence on the loss change.

    This is the "environment-fixed" / implicit-differentiation variant.
    Gradients through the full CTMRG unroll are available in principle but
    are prohibitively expensive for production runs.

    Args:
        Hab..Hfa                : 2-site nearest-neighbour Hamiltonians
        chi                     : environment bond dimension
        D_bond                  : physical projector bond dimension
        d_PHYS                  : physical Hilbert-space dimension
        a_third_max_steps_CTMRG  : max CTMRG sweeps per environment update
        CTM_env_conv_threshold      : CTMRG convergence criterion
        a2f_initialize_way      : initialisation mode ('random', 'product', ...)
        a2f_noise_scale         : noise amplitude for initialisation
        max_opt_steps           : max outer optimisation iterations
        lbfgs_max_iter          : max L-BFGS sub-iterations per outer step
        lbfgs_lr                : initial step size for L-BFGS
        lbfgs_history           : L-BFGS history size
        opt_conv_threshold      : stop when |Δloss| < this value

    Returns:
        a, b, c, d, e, f  —  optimised site tensors. TODO:(still require_grad=True),
    """
    D_squared = D_bond ** 2

    # ── 1. Initialise site tensors ────────────────────────────────────────────
    if init_abcdef is not None:
        a, b, c, d, e, f = [t.detach().clone().to(CDTYPE) for t in init_abcdef]
    else:
        a, b, c, d, e, f = initialize_abcdef(a2f_initialize_way, D_bond, d_PHYS, a2f_noise_scale)
    a.requires_grad_(True)
    b.requires_grad_(True)
    c.requires_grad_(True)
    d.requires_grad_(True)
    e.requires_grad_(True)
    f.requires_grad_(True)

    prev_loss = None
    loss_item: float = float('nan')

    # ── 2 & 3. Outer optimisation loop ───────────────────────────────────────
    # Note: a–f are scale-redundant (the energy is a ratio, hence invariant to
    # the common norm of each tensor).  We therefore normalise each tensor after
    # every outer step so that their scale never drifts.  Because the environment
    # is re-converged from scratch at the start of every outer step, the L-BFGS
    # curvature history from the previous outer step is stale; a fresh optimizer
    # is created each iteration — this costs nothing and avoids misleading
    # quasi-Newton updates built against a different loss landscape.
    for opt_step in range(max_opt_steps):

        # 3a. Normalise a–f (scale redundancy; no-op on first step if init is
        #     already unit-norm, harmless otherwise).  Done via .data so that
        #     the computation graph and grad accumulators are untouched.
        # with torch.no_grad():
        #     a.data = normalize_tensor(a.data)
        #     b.data = normalize_tensor(b.data)
        #     c.data = normalize_tensor(c.data)
        #     d.data = normalize_tensor(d.data)
        #     e.data = normalize_tensor(e.data)
        #     f.data = normalize_tensor(f.data)

        # 3b. Fresh L-BFGS for this outer step (environment will change, so old
        #     curvature history is irrelevant and would only mislead the solver).
        optimizer = torch.optim.LBFGS(
            [a, b, c, d, e, f],
            lr=lbfgs_lr,
            max_iter=lbfgs_max_iter,
            tolerance_grad=opt_tolerance_grad,
            tolerance_change=opt_tolerance_change,
            history_size=lbfgs_history,
            line_search_fn='strong_wolfe',
        )

        # 3d. L-BFGS closure: energy evaluation + backward through a...f only.
        def closure():
            optimizer.zero_grad()
        # with torch.no_grad():

            A, B, C, D, E, F = abcdef_to_ABCDEF(a, b, c, d, e, f, D_squared)
            
            (C21CD, C32EF, C13AB, T1F,  T2A,  T2B,  T3C,  T3D,  T1E,
                C21EB, C32AD, C13CF, T1D,  T2C,  T2F,  T3E,  T3B,  T1A,
                C21AF, C32CB, C13ED, T1B,  T2E,  T2D,  T3A,  T3F,  T1C,
                _ctm_steps) = \
            CTMRG_from_init_to_stop(A, B, C, D, E, F, chi, D_squared,
                a_third_max_steps_CTMRG, CTM_env_conv_threshold, identity_init=identity_init)

            loss = (
            energy_expectation_nearest_neighbor_3ebadcf_bonds(
                a,b,c,d,e,f, 
                Heb,Had,Hcf,
                chi, D_bond, d_PHYS, 
                C21CD,C32EF,C13AB,T1F,T2A,T2B,T3C,T3D,T1E)
            +
            energy_expectation_nearest_neighbor_3afcbed_bonds(
                a,b,c,d,e,f, 
                Haf,Hcb,Hed, 
                chi, D_bond, d_PHYS, 
                C21EB, C32AD,C13CF,T1D,T2C,T2F,T3E,T3B,T1A)
            +
            energy_expectation_nearest_neighbor_other_3_bonds(
                a,b,c,d,e,f, 
                Hcd,Hef,Hab, 
                chi, D_bond, d_PHYS, 
                C21AF,C32CB,C13ED,T1B,T2E,T2D,T3A,T3F,T1C)
            )

            loss.backward()
            return loss

        # 3e. Run L-BFGS sub-iterations (strong-Wolfe line search built in).
        loss_val = optimizer.step(closure)

        loss_item = loss_val.item()
        delta = (loss_item - prev_loss) if prev_loss is not None else float('inf')
        print(f"  opt {opt_step:4d}  loss = {loss_item:+.10f}  Δloss = {delta:.3e}")

        # 3f. Outer convergence check.
        if prev_loss is not None and abs(delta) < opt_conv_threshold:
            print(f"  Outer convergence achieved at step {opt_step} (Δloss={delta:.3e}).")
            break
        prev_loss = loss_item

    return a, b, c, d, e, f, loss_item


# PG: To avoid complex numbers Hamiltonian can be wriiten with S+ and S-

def build_heisenberg_H(J: float = 1.0, d: int = 2) -> torch.Tensor:
    # use spin s=(d-1)/2 to build spin-s operators sx, sy, sz:
    spin = (d - 1) / 2
    spin = torch.tensor(spin, dtype=CDTYPE)
    Sx = torch.zeros((d, d), dtype=CDTYPE)
    Sy = torch.zeros((d, d), dtype=CDTYPE)
    Sz = torch.zeros((d, d), dtype=CDTYPE)
    for i in range(d):
        m = float(spin - i)                   # numeric m-value (float)
        Sz[i, i] = m                          # diagonal Sz

        if i < d - 1:
            # coefficient for the transition m -> m-1
            coeff = 0.5 * (spin * (spin + 1) - m * (m - 1)) ** 0.5  # real CS-coeff/2

            # Sx: symmetric real off-diagonals
            Sx[i, i + 1] = coeff
            Sx[i + 1, i] = coeff

            # Sy: purely imaginary, Hermitian (use conj for the lower diag)
            Sy[i, i + 1] = -1j * coeff
            Sy[i + 1, i] = -Sy[i, i + 1]

    # print(Sx,Sy,Sz)

    #sx = torch.tensor([[0, 1], [1, 0]], dtype=CDTYPE) / 2
    #sy = torch.tensor([[0, -1j], [1j, 0]], dtype=CDTYPE) / 2
    #sz = torch.tensor([[1, 0], [0, -1]], dtype=CDTYPE) / 2

    SdotS = (oe.contract("ij,kl->ijkl", Sx, Sx)
           + oe.contract("ij,kl->ijkl", Sy, Sy)
           + oe.contract("ij,kl->ijkl", Sz, Sz))
    return J * SdotS


    
