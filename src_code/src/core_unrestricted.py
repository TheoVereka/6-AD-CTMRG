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
import opt_einsum as oe
import torch
from torch.utils.checkpoint import checkpoint as _ckpt






# ── Precision control ─────────────────────────────────────────────────────────
# Three dtype globals control the entire codebase:
#   CDTYPE      – always complex (complex64 or complex128).
#                 Used for inherently complex quantities: Hamiltonian (Sy has -1j),
#                 spin operators, and any future complex-valued objects.
#   RDTYPE      – always real (float32 or float64).
#                 Used for SVD singular values, norms, and other inherently real
#                 quantities.
#   TENSORDTYPE – the dtype of the iPEPS site tensors a..f, environment tensors
#                 C/T, and all intermediate contractions.  Can be real (RDTYPE)
#                 for the S+/S- Hamiltonian formulation, or complex (CDTYPE) for
#                 the full Sx/Sy/Sz formulation.
#
# Call  set_dtype(use_double, use_real)  BEFORE allocating any tensor.
CDTYPE: torch.dtype = torch.complex128   # complex dtype for all tensors
RDTYPE: torch.dtype = torch.float64      # real dtype (SVD singular values, norms)
TENSORDTYPE: torch.dtype = torch.float64 # tensor dtype (real or complex)

# Call  set_device(device)  BEFORE allocating any tensor.
# Defaults to CPU; set to  torch.device('cuda')  from the driver script to
# run all CTMRG + energy computations on GPU.
DEVICE: torch.device = torch.device('cpu')




# ── SVD full/partial control ─────────────────────
# When True, SVD_PROPACK uses full deterministic SVD (np.linalg.svd).
# Set directly from the optimizer driver; cleared after one L-BFGS step.
_USE_FULL_SVD: bool = False

# ── SVD GPU/CPU dispatch threshold ───────────────
# For CUDA inputs, matrices with min(m,n) < this threshold are SVD'd on CPU
# and results moved back to GPU.  cuSOLVER SVD has a fixed per-call overhead
# that makes it slower than CPU LAPACK for small matrices on any GPU, and
# slower at ALL sizes on low-end mobile GPUs (MX250 etc.).
#
# Crossover (approx, depends on GPU model):
#   MX250 / laptop GPU : CPU always faster  → set to 99999
#   A100 / V100 / H100 : GPU faster for n > ~300-500 → set to 0 or 512
#
# Default 512: small CTMRG matrices (low chi/D) go to CPU, large ones to GPU.
# Change to 0 to always use GPU (best for large-chi runs on cluster A100).
# Change to 99999 to always use CPU (best on MX250 / quick local testing).
_SVD_CPU_OFFLOAD_THRESHOLD: int = 0






def safe_inverse(x, epsilon=1e-12):
    return x/(x**2 + epsilon**2)
    
def safe_inverse_2(x, epsilon):
    x[abs(x)<epsilon]=float('inf')
    return x.pow(-1)


def _safe_sqrt_inv_diag(S: torch.Tensor) -> torch.Tensor:
    """Compute diag(1/sqrt(S)) safely, clamping near-zero singular values.

    On GPU with float32, singular values near machine epsilon can underflow
    to zero → 1/sqrt(0) = inf → exploding CTMRG projectors → garbage
    environment → wrong energy.  Fix: clamp S[i] to at least
    S[0] * eps_machine before inverting:
        float32: threshold = S[0] × 1.19e-7
        float64: threshold = S[0] × 2.22e-16
    Only genuine underflows (S[i] represented as 0 in floating point) are
    affected; physically meaningful small singular values are untouched.
    """
    s_max    = S[0].abs().clamp(min=float(torch.finfo(S.dtype).tiny))
    S_safe   = S.clamp(min=(s_max * torch.finfo(S.dtype).eps ))
    return torch.diag(1.0 / torch.sqrt(S_safe).to(TENSORDTYPE))


def _solve_fifth_term_svd(A, U, S, V, gu, gv, eps):
    """
    Compute  Γ_A^{trunc}  from arXiv:2311.11894v3 (5th term of truncated SVD backward).

    Solves the coupled Sylvester system:
        Γ_U  =  γ S  −  A(I−VV†)γ̃      (*)
        Γ_V  =  γ̃ S  −  A†(I−UU†)γ    (**)

    Decoupled column-by-column via EVD of R = B†B  (B = A − USV†):
        (s_j² I_n − R) γ̃_j = s_j Γ_V_j + B† Γ_U_j
        γ_j = (Γ_U_j + B γ̃_j) / s_j

    5th term:  Γ_A^{trunc} = (I−UU†)γ V† + U γ̃†(I−VV†)

    This term is:
      •  zero when k = min(m,n) (full-thin SVD, B = 0, reduces exactly to the
         out-of-subspace projector terms used previously).
      •  the correct truncation correction when k < min(m,n)  (partial SVD).

    A : (m, n)  — original input matrix (saved in forward)
    U : (m, k)  — retained left singular vectors
    S : (k,)    — retained singular values (real, descending)
    V : (n, k)  — retained right singular vectors
    gu, gv : upstream gradients (m×k, n×k; zeros substituted when None)
    eps    : regularisation threshold
    """
    m, n = A.shape
    k = S.shape[0]
    Uh = U.conj().t()   # k×m
    Vh = V.conj().t()   # k×n

    # B = A(I − VV†) = A − U diag(S) V†
    B  = A - U @ torch.diag(S.to(A.dtype)) @ Vh   # m×n
    Bh = B.conj().t()                               # n×m

    # R = B†B  (n×n, positive semi-definite, symmetrized numerically)
    R = Bh @ B
    R = (R + R.conj().t()) * 0.5

    # EVD: R = Q Λ Q†  (Λ ascending)
    Lambda, Q = torch.linalg.eigh(R)   # Lambda: n, Q: n×n
    Qh = Q.conj().t()

    # RHS_j = s_j · gv_j + B† · gu_j  ⟹  RHS matrix (n×k)
    RHS = gv * S.to(gv.dtype).unsqueeze(0) + Bh @ gu   # n×k

    # denom[i,j] = S[j]² − Λ[i]   (n×k)
    S2    = S ** 2                                          # k  (real)
    denom = S2.unsqueeze(0) - Lambda.unsqueeze(1)          # (1,k) − (n,1) = n×k

    # Regularise near-zero denominators (touching a multiplet boundary)
    invDenom_reg = safe_inverse(denom, epsilon=eps)  # n×k

    # γ̃ = Q [(Q†·RHS) / denom]   (n×k)
    gamma_tilde = Q @ (Qh @ RHS * invDenom_reg)

    # γ_j = (gu_j + B·γ̃_j) / s_j   (m×k)
    gamma = (gu + B @ gamma_tilde) / S.to(A.dtype).unsqueeze(0)

    # 5th term:  (I−UU†)γ V†  +  U γ̃†(I−VV†)
    PU_gamma      = gamma - U @ (Uh @ gamma)            # (I−UU†)γ : m×k
    gamma_tilde_h = gamma_tilde.conj().t()              # k×n
    gtilde_h_PV   = gamma_tilde_h - (gamma_tilde_h @ V) @ Vh  # γ̃†(I−VV†) : k×n

    return PU_gamma @ Vh + U @ gtilde_h_PV              # m×n








def set_dtype(use_double: bool, use_real: bool = True) -> None:
    """Switch all core computations between float32/complex64 and float64/complex128.

    Must be called BEFORE any tensor is allocated — ideally right after import.

    Args:
        use_double: ``True``  → complex128 / float64 (double precision).
                    ``False`` → complex64  / float32 (single precision, default).
        use_real:   ``True``  → TENSORDTYPE = RDTYPE  (real iPEPS tensors, S+/S- Hamiltonian).
                    ``False`` → TENSORDTYPE = CDTYPE  (complex iPEPS tensors, Sx/Sy/Sz Hamiltonian).
    """
    global CDTYPE, RDTYPE, TENSORDTYPE

    if use_double:
        CDTYPE = torch.complex128
        RDTYPE = torch.float64
    else:
        CDTYPE = torch.complex64
        RDTYPE = torch.float32

    if use_real:
        TENSORDTYPE = RDTYPE
    else:
        TENSORDTYPE = CDTYPE


def set_device(device) -> None:
    """Switch all core tensor allocations to a different compute device.

    Must be called BEFORE any tensor is allocated — ideally right after
    ``set_dtype()``.

    Args:
        device: Anything accepted by ``torch.device()``:
            ``'cpu'``, ``'cuda'``, ``'cuda:0'``, or a ``torch.device``
            instance.  Passing ``None`` is treated as ``'cpu'``.
    """
    global DEVICE
    DEVICE = torch.device(device) if device is not None else torch.device('cpu')


def normalize_tensor(tensor, *, rtol: float | None = None, atol: float | None = None):
    
    """
    Normalise a tensor in-place-style by its Frobenius (L2) norm.

    Divides every element by ``||tensor||_F``.  If the norm is exactly zero
    (e.g. an all-zero tensor) the input is returned unchanged to avoid a
    division-by-zero NaN.

    GPU-friendly: no GPU→CPU synchronisation (no .item(), no Python ``if``
    on a CUDA scalar).  Uses ``clamp`` to guard against zero/NaN norms
    instead of branching.

    Args:
        tensor (torch.Tensor): Any complex or real tensor of arbitrary shape.

    Returns:
        torch.Tensor: A new tensor with the same shape and dtype as the input
        whose Frobenius norm is 1, or the original tensor if its norm is 0.
    """
    # Frobenius norm (for arbitrary-rank tensors). Returns a real scalar.
    norm = torch.linalg.norm(tensor)

    # Clamp norm to a safe minimum to avoid division-by-zero.
    # For NaN/Inf/zero norms this produces a harmless (possibly large) result
    # rather than NaN — same logical effect as the old early-return branches
    # but without GPU→CPU sync.
    safe_norm = norm.clamp(min=1e-30)
    return tensor / safe_norm


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

    # GPU-friendly: clamp dl_norm to a safe minimum to avoid division-by-zero
    # and sqrt(0).  No GPU→CPU sync — no .item(), no Python ``if`` on a CUDA
    # scalar.  For NaN/Inf/zero norms the clamp produces a safe denominator.
    safe_dl_norm = dl_norm.clamp(min=1e-30)
    return tensor / torch.sqrt(safe_dl_norm)








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
    


from torch import _linalg_utils as _utils, Tensor

class SVD_PROPACK(torch.autograd.Function):
    @staticmethod
    def forward(self, A, k, k_extra, rel_cutoff, v0, use_full_svd):
        r"""
        :param M: square matrix :math:`N \times N`
        :param k: desired rank (must be smaller than :math:`N`)
        :param use_full_svd: if True, use full deterministic SVD (np.linalg.svd)
                             instead of the randomized projection SVD.
        :type M: torch.Tensor
        :type k: int
        :type use_full_svd: bool
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
        _rdtype = torch.float64 if A.dtype in (torch.complex128, torch.float64) else torch.float32

        m_dim, n_dim = A.shape
        k_total = min(k + k_extra, min(m_dim, n_dim))
        q = max(k_total, 1)

        # print(f"Performing SVD with k={k}, k_extra={k_extra}, rel_cutoff={rel_cutoff.item()}, m_dim={m_dim}, n_dim={n_dim}, k_total={k_total}")


        if False and not use_full_svd:  # Partial (randomized) SVD — fast but can blow up
            #print("Trunc", end=' ')
            matmul = _utils.matmul


            niter = 1 

            dtype = _utils.get_floating_dtype(A) if not A.is_complex() else A.dtype
            matmul = _utils.matmul

            R = torch.randn(A.shape[-1], q, dtype=dtype, device=A.device)

            # The following code could be made faster using torch.geqrf + torch.ormqr
            # but geqrf is not differentiable

            X = matmul(A, R)
            Q = torch.linalg.qr(X).Q
            for _ in range(niter):
                X = matmul(A.mH, Q)
                Q = torch.linalg.qr(X).Q
                X = matmul(A, Q)
                Q = torch.linalg.qr(X).Q

            B = matmul(Q.mH, A)
            
            U, S, Vh = torch.linalg.svd(B, full_matrices=False)
            V = Vh.mH
            U = Q.matmul(U)
            
            #S = torch.as_tensor(S.copy(),  dtype=_rdtype,  device=A.device)
            #U = torch.as_tensor(U.copy(),  dtype=A.dtype,  device=A.device)
            #V = torch.as_tensor(V.copy(), dtype=A.dtype,  device=A.device)

            self.save_for_backward(U,S,V, A.detach(), rel_cutoff)


            return U, S, V

        else:
            #print("Full", end='')
            # Full thin SVD.  cuSOLVER has a fixed per-call overhead that makes
            # it slower than CPU LAPACK for matrices below the crossover size
            # (varies by GPU: always on MX250, n<~500 on A100/V100).
            # Use _SVD_CPU_OFFLOAD_THRESHOLD to control dispatch per hardware.
            dev = A.device
            n_min = min(A.shape)
            if dev.type == 'cuda' and n_min < _SVD_CPU_OFFLOAD_THRESHOLD:
                A_cpu = A.detach().cpu()
                U_cpu, S_cpu, Vh_cpu = torch.linalg.svd(A_cpu, full_matrices=False)
                U = U_cpu.to(dev)
                S = S_cpu.to(dev)
                V = Vh_cpu.mH.to(dev)
            else:
                U, S, V = torch.linalg.svd(A.detach(), full_matrices=False)
                V = V.conj().T

            self.save_for_backward(U, S, V, None, rel_cutoff)

            return U[:,:k], S[:k], V[:,:k]

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
        u, sigma, v, A_input, eps = self.saved_tensors
        m = u.size(0)      # first dim of A
        n = v.size(0)      # second dim of A
        k = sigma.size(0)  # number of retained singular triples
        sigma_scale = sigma[0]

        # eps was stored with real dtype; extract .real to guard legacy paths.
        _eps_real   = eps.real if eps.is_complex() else eps
        # Keep eps_val as a scalar tensor to avoid GPU→CPU sync.
        eps_val     = (sigma_scale * _eps_real).clamp(min=1e-30)

        sigma_inv = safe_inverse_2(sigma.clone(), sigma_scale * _eps_real)

        if A_input is None:
                # Forward returned only k_return singular triples; gu and gv have width
            # k_return ≤ k.  Pad them with zeros so the F,G cross-coupling between
            # kept and discarded singular values is captured correctly.
            k_return = gsigma.size(0)  # fallback to k for safety
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

        uh = u.conj().transpose(-2, -1)
        vh = v.conj().transpose(-2, -1)
        

        
        # ── 1st term: σ-gradient ─────────────────────────────────────────────
        if gsigma is not None:
            sigma_term = u * gsigma.unsqueeze(-2) @ vh
        else:
            sigma_term = torch.zeros(m, n, dtype=u.dtype, device=u.device)

        if (gu is None) and (gv is None):
            return sigma_term, None, None, None, None, None
        
        # k×k F, G matrices (within-subspace rotation coupling)
        F = safe_inverse(sigma.unsqueeze(-2) - sigma.unsqueeze(-1),
                         sigma_scale * _eps_real)
        F.diagonal(0, -2, -1).fill_(0)
        G = safe_inverse(sigma.unsqueeze(-2) + sigma.unsqueeze(-1),
                         sigma_scale * _eps_real)
        G.diagonal(0, -2, -1).fill_(0)
        
        # ── 2nd term: within-subspace U rotation ─────────────────────────────
        if gu is not None:
            guh    = gu.conj().transpose(-2, -1)
            u_term = u @ ((F + G).mul(uh @ gu - guh @ u)) * 0.5 @ vh
        else:
            u_term = torch.zeros(m, n, dtype=u.dtype, device=u.device)

        # ── 3rd term: within-subspace V rotation ─────────────────────────────
        if gv is not None:
            gvh    = gv.conj().transpose(-2, -1)
            v_term = u @ ((F - G).mul(vh @ gv - gvh @ v)) @ vh * 0.5
        else:
            v_term = torch.zeros(m, n, dtype=u.dtype, device=u.device)
        
        # ── 4th term: complex phase correction (arXiv:1909.02659) ────────────
        dA = u_term + sigma_term + v_term
        if (u.is_complex() or v.is_complex()) and (gu is not None):
            L = (uh @ gu).diagonal(0, -2, -1).clone()
            L.real.zero_()
            L.imag.mul_(sigma_inv)
            dA = dA + (u * L.unsqueeze(-2)) @ vh

        # ── 5th term: Γ_A^{trunc}  (arXiv:2311.11894v3) ───────────────────────
        # Solves the coupled Sylvester system for γ, γ̃:
        #   Γ_U = γ S − A(I−VV†)γ̃      Γ_V = γ̃ S − A†(I−UU†)γ
        # Then: Γ_A^{trunc} = (I−UU†)γV† + Uγ̃†(I−VV†)
        #
        # When k = min(m,n) (full-thin branch, B = A−USV† = 0) this reduces
        # exactly to the old proj_on_ortho terms and gives zero extra correction.
        # When k < min(m,n) (partial SVD branch or ARPACK) this provides the
        # previously-missing truncation correction.

        if A_input is not None:
            _gu = gu if gu is not None else torch.zeros(m, k, dtype=u.dtype, device=u.device)
            _gv = gv if gv is not None else torch.zeros(n, k, dtype=u.dtype, device=u.device)
            dA += _solve_fifth_term_svd(A_input, u, sigma, v, _gu, _gv, eps_val)
            



        return dA, None, None, None, None, None





def truncated_svd_propack(M, chi, chi_extra, rel_cutoff, v0=None,
    abs_tol=1.0e-14,  keep_multiplets=False, \
    eps_multiplet=1e-12, ):
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

    # Clamp all tolerances to be at least proportional to the dtype's machine
    # epsilon.  Hardcoded 1e-12/1e-14 values are fine for float64 (eps≈2e-16)
    # but are effectively zero for float32 (eps≈1.2e-7), giving no regularisation.
    _eps_dtype = float(torch.finfo(_rdtype).eps)
    rel_cutoff    = max(rel_cutoff,    _eps_dtype)
    abs_tol       = max(abs_tol,       _eps_dtype)
    eps_multiplet = max(eps_multiplet, _eps_dtype)

    U, S, V = SVD_PROPACK.apply(M, chi, 0, torch.as_tensor([rel_cutoff], dtype=_rdtype, device=M.device), v0, _USE_FULL_SVD)





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
        ``(D_bond, D_bond, D_bond, d_PHYS)``, dtype ``torch.float32``.

    Raises:
        ValueError: If ``initialize_way`` is not a recognised strategy.
    """

    if initialize_way == 'random' :

        a = torch.randn(D_bond, D_bond, D_bond, d_PHYS, dtype=TENSORDTYPE, device=DEVICE)
        b = torch.randn(D_bond, D_bond, D_bond, d_PHYS, dtype=TENSORDTYPE, device=DEVICE)
        c = torch.randn(D_bond, D_bond, D_bond, d_PHYS, dtype=TENSORDTYPE, device=DEVICE)
        d = torch.randn(D_bond, D_bond, D_bond, d_PHYS, dtype=TENSORDTYPE, device=DEVICE)
        e = torch.randn(D_bond, D_bond, D_bond, d_PHYS, dtype=TENSORDTYPE, device=DEVICE)
        f = torch.randn(D_bond, D_bond, D_bond, d_PHYS, dtype=TENSORDTYPE, device=DEVICE)
        global _USE_FULL_SVD
        _USE_FULL_SVD = True

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
        pair per honeycomb leg), dtype ``torch.float32``.
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









def trunc_rhoCCC(matC21, matC32, matC13, chi, D_squared):

    #Q1, R1 = torch.linalg.qr(matC21.T)
    #Q2, R2 = torch.linalg.qr(torch.mm(matC13,matC32))
    R1 = matC21.T
    R2 = torch.mm(matC13,matC32)
    
    #U, S, Vh = torch.linalg.svd(torch.mm(R1,R2.T))
    #truncUh = U[:,:chi].conj().T
    #truncV = Vh[:chi,:].conj().T



    U, S, V = truncated_svd_propack(torch.mm(R1,R2.T), chi,
                    chi_extra=round(2*np.sqrt(D_squared)),
                    rel_cutoff=1e-12,
                    v0=None,
                    keep_multiplets=False,
                    abs_tol=1e-14,
                    eps_multiplet=1e-12)
    

    

    #approx_product =  U[:,:chi] @ torch.diag(S[:chi]).to(TENSORDTYPE) @ Vh[:chi,:]
    #exact_product = R1 @ R2.T
    #product_error = torch.linalg.norm(approx_product - exact_product) / torch.linalg.norm(exact_product)
    #VERIFIED print(f"R1 @ R2.T vs truncated U @ diag(S) @ Vh check: relative error = {product_error:.2e}")

    sqrtInvTruncS = _safe_sqrt_inv_diag(S[:chi])
    #VERIFIED print((S[chi]/S[0]).item())
    #VERIFIED print(torch.diag(sqrtInvTruncS @ sqrtInvTruncS @ torch.diag(S[:chi]).to(TENSORDTYPE)).detach().cpu().numpy())

    #product_inverse = Vh.conj().T @ torch.diag(1.0 / torch.sqrt(S).to(TENSORDTYPE)) @ torch.diag(1.0 / torch.sqrt(S).to(TENSORDTYPE)) @ U.conj().T
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
    # NOTE: But still, the error for R2 V 1/S Uh R1 is much larger than V 1/S Uh R1 R2
    # NOTE: QR (SVD-1)R1R2 >~= no-QR (SVD-1)R1R2 > QR R2(SVD-1)R1 ~= no-QR R2(SVD-1)R1
###############################


    P21My = torch.mm( R2.T ,torch.mm( V , sqrtInvTruncS ))
    P32yM = torch.mm(torch.mm( sqrtInvTruncS , U.conj().T ), R1 )

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
                    chi_extra=round(2*np.sqrt(D_squared)),
                    rel_cutoff=1e-12,
                    v0=None,
                    keep_multiplets=False,
                    abs_tol=1e-14,
                    eps_multiplet=1e-12)
    

    #truncUh = U[:,:chi].conj().T
    #truncV = Vh[:chi,:].conj().T
    sqrtInvTruncS = _safe_sqrt_inv_diag(S[:chi])

    P32Nz = torch.mm( R2.T ,torch.mm( V, sqrtInvTruncS ))
    P13zN = torch.mm(torch.mm( sqrtInvTruncS , U.conj().T ), R1 )




    #Q1, R1 = torch.linalg.qr(matC13.T)
    #Q2, R2 = torch.linalg.qr(torch.mm(matC32,matC21))
    R1 = matC13.T
    R2 = torch.mm(matC32,matC21)
    #U, S, Vh = torch.linalg.svd(torch.mm(R1,R2.T))
    U, S, V = truncated_svd_propack(torch.mm(R1,R2.T), chi,
                    chi_extra=round(2*np.sqrt(D_squared)),
                    rel_cutoff=1e-12,
                    v0=None,
                    keep_multiplets=False,
                    abs_tol=1e-14,
                    eps_multiplet=1e-12)




    #truncUh = U[:,:chi].conj().T
    #truncV = Vh[:chi,:].conj().T
    sqrtInvTruncS = _safe_sqrt_inv_diag(S[:chi])

    P13Lx = torch.mm( R2.T ,torch.mm( V, sqrtInvTruncS ))
    P21xL = torch.mm(torch.mm( sqrtInvTruncS , U.conj().T ), R1 )



    matC21 = torch.mm(P21My.T, torch.mm(matC21, P21xL.T))
    matC32 = torch.mm(P32Nz.T, torch.mm(matC32, P32yM.T))
    matC13 = torch.mm(P13Lx.T, torch.mm(matC13, P13zN.T))
    
    
    matC21 = matC21/torch.linalg.norm(matC21)
    matC32 = matC32/torch.linalg.norm(matC32)
    matC13 = matC13/torch.linalg.norm(matC13)
    

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


    return P21xL.reshape(chi, chi,D_squared), matC21, P21My.reshape(chi,D_squared, chi), P32yM.reshape(chi, chi,D_squared), matC32, P32Nz.reshape(chi,D_squared, chi), P13zN.reshape(chi, chi,D_squared), matC13, P13Lx.reshape(chi,D_squared, chi)


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
                        chi_extra=round(2*np.sqrt(D_squared)),
                        rel_cutoff=1e-12,
                        v0=None,
                        keep_multiplets=False,
                        abs_tol=1e-14,
                        eps_multiplet=1e-12)
        

        sqrtInvTruncS = _safe_sqrt_inv_diag(S[:chi])

        P21My = torch.mm( R2.T ,torch.mm( V , sqrtInvTruncS ))
        P32yM = torch.mm(torch.mm( sqrtInvTruncS , U.conj().T ), R1 )


        R1 = C32EF.T
        R2 = torch.mm(C21CD,C13AB)

        #Q1, R1 = torch.linalg.qr(matC32.T)
        #Q2, R2 = torch.linalg.qr(torch.mm(matC21,matC13))
        U, S, V = truncated_svd_propack(torch.mm(R1,R2.T), chi,
                        chi_extra=round(2*np.sqrt(D_squared)),
                        rel_cutoff=1e-12,
                        v0=None,
                        keep_multiplets=False,
                        abs_tol=1e-14,
                        eps_multiplet=1e-12)

        sqrtInvTruncS = _safe_sqrt_inv_diag(S[:chi])

        P32Nz = torch.mm( R2.T ,torch.mm( V , sqrtInvTruncS ))
        P13zN = torch.mm(torch.mm( sqrtInvTruncS , U.conj().T ), R1 )


        R1 = C13AB.T
        R2 = torch.mm(C32EF,C21CD)

        #Q1, R1 = torch.linalg.qr(matC13.T)
        #Q2, R2 = torch.linalg.qr(torch.mm(matC32,matC21))
        U, S, V = truncated_svd_propack(torch.mm(R1,R2.T), chi,
                        chi_extra=round(2*np.sqrt(D_squared)),
                        rel_cutoff=1e-12,
                        v0=None,
                        keep_multiplets=False,
                        abs_tol=1e-14,
                        eps_multiplet=1e-12)

        sqrtInvTruncS = _safe_sqrt_inv_diag(S[:chi])

        P13Lx = torch.mm( R2.T ,torch.mm( V , sqrtInvTruncS ))
        P21xL = torch.mm(torch.mm( sqrtInvTruncS , U.conj().T ), R1 )



        C21CD = torch.mm(P21My.T, torch.mm(C21CD, P21xL.T))
        C32EF = torch.mm(P32Nz.T, torch.mm(C32EF, P32yM.T))
        C13AB = torch.mm(P13Lx.T, torch.mm(C13AB, P13zN.T))
        
        C21CD = C21CD/torch.linalg.norm(C21CD)
        C32EF = C32EF/torch.linalg.norm(C32EF)
        C13AB = C13AB/torch.linalg.norm(C13AB)
   
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



        # Uh,V project on one side(XYZ) of Ts



        T1F = oe.contract("MYa,yY->Mya",T1F,P32yM.reshape(chi, D_squared*D_squared),optimize=[(0,1)],backend='torch').reshape(D_squared*D_squared,chi*D_squared)
        T2A = oe.contract("LXb,Xx->Lxb",T2A,P13Lx.reshape(D_squared*D_squared, chi),optimize=[(0,1)],backend='torch').reshape(D_squared*D_squared,chi*D_squared)
        T2B = oe.contract("NZb,zZ->Nzb",T2B,P13zN.reshape(chi, D_squared*D_squared),optimize=[(0,1)],backend='torch').reshape(D_squared*D_squared,chi*D_squared)
        T3C = oe.contract("MYg,Yy->Myg",T3C,P21My.reshape(D_squared*D_squared, chi),optimize=[(0,1)],backend='torch').reshape(D_squared*D_squared,chi*D_squared)
        T3D = oe.contract("LXg,xX->Lxg",T3D,P21xL.reshape(chi, D_squared*D_squared),optimize=[(0,1)],backend='torch').reshape(D_squared*D_squared,chi*D_squared)
        T1E = oe.contract("NZa,Zz->Nza",T1E,P32Nz.reshape(D_squared*D_squared, chi),optimize=[(0,1)],backend='torch').reshape(D_squared*D_squared,chi*D_squared)  

        U, S, V = truncated_svd_propack(torch.mm(T1F.T,T3C), chi,
                    chi_extra=round(2*np.sqrt(D_squared)),
                    rel_cutoff=1e-12,
                    v0=None,
                    keep_multiplets=False,
                    abs_tol=1e-14,
                    eps_multiplet=1e-12)
        
        

        sqrtInvTruncS = _safe_sqrt_inv_diag(S[:chi])

        old = T1F
        T1F = torch.mm((torch.mm(T3C, torch.mm(V, sqrtInvTruncS))).T, T1F)
        T3C = torch.mm(torch.mm( torch.mm(sqrtInvTruncS, U.conj().T), old.T), T3C)


        U, S, V = truncated_svd_propack(torch.mm(T3D.T,T2A), chi,
                    chi_extra=round(2*np.sqrt(D_squared)),
                    rel_cutoff=1e-12,
                    v0=None,
                    keep_multiplets=False,
                    abs_tol=1e-14,
                    eps_multiplet=1e-12)
        
        sqrtInvTruncS = _safe_sqrt_inv_diag(S[:chi])
        
        old = T3D
        T3D = torch.mm((torch.mm(T2A, torch.mm(V, sqrtInvTruncS))).T, T3D)
        T2A = torch.mm(torch.mm( torch.mm(sqrtInvTruncS, U.conj().T), old.T), T2A)


        U, S, V = truncated_svd_propack(torch.mm(T2B.T,T1E), chi,
                    chi_extra=round(2*np.sqrt(D_squared)),
                    rel_cutoff=1e-12,
                    v0=None,
                    keep_multiplets=False,
                    abs_tol=1e-14,
                    eps_multiplet=1e-12)
        
        sqrtInvTruncS = _safe_sqrt_inv_diag(S[:chi])

        old = T2B
        T2B = torch.mm((torch.mm(T1E, torch.mm(V, sqrtInvTruncS))).T, T2B)
        T1E = torch.mm(torch.mm( torch.mm(sqrtInvTruncS, U.conj().T), old.T), T1E)
            





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








def check_env_CV_using_3rho(lastC21CD, lastC32EF, lastC13AB, 
                            nowC21CD, nowC32EF, nowC13AB, 
                            lastC21EB, lastC32AD, lastC13CF, 
                            nowC21EB, nowC32AD, nowC13CF, 
                            lastC21AF, lastC32CB, lastC13ED, 
                            nowC21AF, nowC32CB, nowC13ED, 
                            env_conv_threshold):
   
    # Warm-up guard: all three env types must have been computed at least once.

    # Keep max_delta as a GPU scalar tensor; extract to Python only once at return.
    max_delta = lastC21CD.new_tensor(0.0).real
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
        delta = (sv_now - sv_last).abs().max()
        max_delta = torch.maximum(max_delta, delta)

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
        delta = (sv_now - sv_last).abs().max()
        max_delta = torch.maximum(max_delta, delta)

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
        delta = (sv_now - sv_last).abs().max()
        max_delta = torch.maximum(max_delta, delta)

    # Single GPU→CPU sync for the entire convergence check.
    return (max_delta < env_conv_threshold).item()



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

    C21EB = oe.contract("YX,MYa,LXb,amg,lbg->MmLl",
                           C21CD,T1F,T2A,E,B,
                           optimize=[(0,1),(1,3),(0,1),(0,1)],
                           backend='torch')
    
    C21EB = C21EB.reshape(chi*D_squared,chi*D_squared)
    
    C32AD = oe.contract("ZY,NZb,MYg,abn,amg->NnMm",
                           C32EF,T2B,T3C,A,D,
                           optimize=[(0,1),(1,3),(0,1),(0,1)],
                           backend='torch')
    
    C32AD = C32AD.reshape(chi*D_squared,chi*D_squared)
    
    C13CF = oe.contract("XZ,LXg,NZa,lbg,abn->LlNn",
                           C13AB,T3D,T1E,C,F,
                           optimize=[(0,1),(1,3),(0,1),(0,1)],
                           backend='torch')
    
    C13CF = C13CF.reshape(chi*D_squared,chi*D_squared)


    V2A, C21EB, U1F, V3C, C32AD, U2B, V1E, C13CF, U3D = trunc_rhoCCC(
                        C21EB, C32AD, C13CF, chi, D_squared)


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

    C21AF = oe.contract("YX,MYa,LXb,amg,lbg->MmLl",
                           C21EB,T1D,T2C,A,F,
                           optimize=[(0,1),(1,3),(0,1),(0,1)],
                           backend='torch')
    
    C21AF = C21AF.reshape(chi*D_squared,chi*D_squared)
    
    C32CB = oe.contract("ZY,NZb,MYg,abn,amg->NnMm",
                           C32AD,T2F,T3E,C,B,
                           optimize=[(0,1),(1,3),(0,1),(0,1)],
                           backend='torch')
    
    C32CB = C32CB.reshape(chi*D_squared,chi*D_squared)
    
    C13ED = oe.contract("XZ,LXg,NZa,lbg,abn->LlNn",
                           C13CF,T3B,T1A,E,D,
                           optimize=[(0,1),(1,3),(0,1),(0,1)],
                           backend='torch')
    
    C13ED = C13ED.reshape(chi*D_squared,chi*D_squared)

    V2C, C21AF, U1D, V3E, C32CB, U2F, V1A, C13ED, U3B = trunc_rhoCCC(
                        C21AF, C32CB, C13ED, chi, D_squared)

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

    C21CD = oe.contract("YX,MYa,LXb,amg,lbg->MmLl",
                           C21AF,T1B,T2E,C,D,
                           optimize=[(0,1),(1,3),(0,1),(0,1)],
                           backend='torch')
    
    C21CD = C21CD.reshape(chi*D_squared,chi*D_squared)
    
    C32EF = oe.contract("ZY,NZb,MYg,abn,amg->NnMm",
                           C32CB,T2D,T3A,E,F,
                           optimize=[(0,1),(1,3),(0,1),(0,1)],
                           backend='torch')
    
    C32EF = C32EF.reshape(chi*D_squared,chi*D_squared)
    
    C13AB = oe.contract("XZ,LXg,NZa,lbg,abn->LlNn",
                           C13ED,T3F,T1C,A,B,
                           optimize=[(0,1),(1,3),(0,1),(0,1)],
                           backend='torch')
    
    C13AB = C13AB.reshape(chi*D_squared,chi*D_squared)

    V2E, C21CD, U1B, V3A, C32EF, U2D, V1C, C13AB, U3F = trunc_rhoCCC(
                        C21CD, C32EF, C13AB, chi, D_squared)




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




# ─────────────────────────────────────────────────────────────────────────────
# Memory-efficient CTMRG: each of the three update steps is wrapped with
# torch.utils.checkpoint.checkpoint (aliased as _ckpt).
#
# Why checkpoint and NOT a custom autograd.Function:
#   The update functions call truncated_svd_propack which, on the partial-SVD
#   path, draws R=torch.randn(...).  A custom Function that re-runs the forward
#   in backward() would draw a *different* R → different U,S,V → wrong gradient.
#   torch.utils.checkpoint saves and restores the full PyTorch RNG state before
#   the recompute, so R is bit-identical to the original forward pass.
#
# What is saved vs recomputed:
#   forward:  only the 15 small input tensors are kept alive for the recompute;
#             all M×M intermediate contractions and SVD factors are freed.
#   backward: the update function is re-run once per step to rebuild the local
#             graph, grads are computed, then the local graph is freed.
#
# Memory: O(N_iter × chi² × D²)  instead of  O(N_iter × (chi·D²)²).
# Cost  : ≈1 extra forward pass per update step during the backward phase.
# ─────────────────────────────────────────────────────────────────────────────

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
    # ── Initialization under torch.no_grad() ─────────────────────────────
    # The init only sets the starting point for the iterative CTMRG loop.
    # Gradients flow through the converged update steps, not the init.
    # Running under no_grad prevents the (D²×D², D²×D²) SVD intermediates
    # from being kept alive in the autograd graph (saves ~75 MB for D=5).
    with torch.no_grad():
        nowC21CD, nowC32EF, nowC13AB, nowT1F, nowT2A, nowT2B, nowT3C, nowT3D, nowT1E = initialize_envCTs_1(A,B,C,D,E,F, chi, D_squared, identity_init=identity_init)
    nowC21EB, nowC32AD, nowC13CF, nowT1D, nowT2C, nowT2F, nowT3E, nowT3B, nowT1A = None, None, None, None, None, None, None, None, None
    nowC21AF, nowC32CB, nowC13ED, nowT1B, nowT2E, nowT2D, nowT3A, nowT3F, nowT1C = None, None, None, None, None, None, None, None, None





    # Perform the CTMRG iterations until convergence
    ctm_steps = a_third_max_iterations  # will be overwritten on early convergence
    for iteration in range(a_third_max_iterations):

        if not (lastC21CD is None or lastC21EB is None or lastC21AF is None):
            if check_env_CV_using_3rho(lastC21CD.detach(), lastC32EF.detach(), lastC13AB.detach(), #lastT1F, lastT2A, lastT2B, lastT3C, lastT3D, lastT1E, 
                                    nowC21CD.detach(), nowC32EF.detach(), nowC13AB.detach(), #nowT1F, nowT2A, nowT2B, nowT3C, nowT3D, nowT1E, 
                                    lastC21EB.detach(), lastC32AD.detach(), lastC13CF.detach(), #lastT1D, lastT2C, lastT2F, lastT3E, lastT3B, lastT1A, 
                                    nowC21EB.detach(), nowC32AD.detach(), nowC13CF.detach(), #nowT1D, nowT2C, nowT2F, nowT3E, nowT3B, nowT1A, 
                                    lastC21AF.detach(), lastC32CB.detach(), lastC13ED.detach(), #lastT1B, lastT2E, lastT2D, lastT3A, lastT3F, lastT1C, 
                                    nowC21AF.detach(), nowC32CB.detach(), nowC13ED.detach(), #nowT1B, nowT2E, nowT2D, nowT3A, nowT3F, nowT1C, 
                                    env_conv_threshold):
                ctm_steps = iteration + 1
                break

        # Update the environment corner and edge transfer tensors
        # match iteration % 3 :
        #     case 0 : 
        lastC21EB, lastC32AD, lastC13CF, lastT1D, lastT2C, lastT2F, lastT3E, lastT3B, lastT1A = \
        nowC21EB, nowC32AD, nowC13CF, nowT1D, nowT2C, nowT2F, nowT3E, nowT3B, nowT1A
        nowC21EB, nowC32AD, nowC13CF, nowT1D, nowT2C, nowT2F, nowT3E, nowT3B, nowT1A = _ckpt(
            update_environmentCTs_1to2,
            nowC21CD, nowC32EF, nowC13AB, nowT1F, nowT2A, nowT2B, nowT3C, nowT3D, nowT1E,
            A, B, C, D, E, F, chi, D_squared,
            use_reentrant=False)
            
        #     case 1 : 
        lastC21AF, lastC32CB, lastC13ED, lastT1B, lastT2E, lastT2D, lastT3A, lastT3F, lastT1C = \
        nowC21AF, nowC32CB, nowC13ED, nowT1B, nowT2E, nowT2D, nowT3A, nowT3F, nowT1C

        nowC21AF, nowC32CB, nowC13ED, nowT1B, nowT2E, nowT2D, nowT3A, nowT3F, nowT1C = _ckpt(
            update_environmentCTs_2to3,
            nowC21EB, nowC32AD, nowC13CF, nowT1D, nowT2C, nowT2F, nowT3E, nowT3B, nowT1A,
            A, B, C, D, E, F, chi, D_squared,
            use_reentrant=False)
            
        #     case 2 : 
        lastC21CD, lastC32EF, lastC13AB, lastT1F, lastT2A, lastT2B, lastT3C, lastT3D, lastT1E = \
        nowC21CD, nowC32EF, nowC13AB, nowT1F, nowT2A, nowT2B, nowT3C, nowT3D, nowT1E

        nowC21CD, nowC32EF, nowC13AB, nowT1F, nowT2A, nowT2B, nowT3C, nowT3D, nowT1E = _ckpt(
            update_environmentCTs_3to1,
            nowC21AF, nowC32CB, nowC13ED, nowT1B, nowT2E, nowT2D, nowT3A, nowT3F, nowT1C,
            A, B, C, D, E, F, chi, D_squared,
            use_reentrant=False)
    
    # Release Python references to stale last-iteration env tensors so the
    # autograd engine can GC them as soon as the backward pass processes them.
    del lastC21CD, lastC32EF, lastC13AB, lastT1F, lastT2A, lastT2B, lastT3C, lastT3D, lastT1E
    del lastC21EB, lastC32AD, lastC13CF, lastT1D, lastT2C, lastT2F, lastT3E, lastT3B, lastT1A
    del lastC21AF, lastC32CB, lastC13ED, lastT1B, lastT2E, lastT2D, lastT3A, lastT3F, lastT1C

    return  nowC21CD, nowC32EF, nowC13AB, nowT1F, nowT2A, nowT2B, nowT3C, nowT3D, nowT1E, \
            nowC21EB, nowC32AD, nowC13CF, nowT1D, nowT2C, nowT2F, nowT3E, nowT3B, nowT1A, \
            nowC21AF, nowC32CB, nowC13ED, nowT1B, nowT2E, nowT2D, nowT3A, nowT3F, nowT1C, \
            ctm_steps










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
        torch.Tensor: Scalar energy per bond (float32).
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




# PG: To avoid complex numbers Hamiltonian can be wriiten with S+ and S-

def build_heisenberg_H(J: float = 1.0, d: int = 2) -> torch.Tensor:
    # use spin s=(d-1)/2 to build spin-s operators sx, sy, sz:
    spin = (d - 1) / 2
    spin = torch.tensor(spin, dtype=RDTYPE)
    Splus = torch.zeros((d, d), dtype=RDTYPE)
    Sminus = torch.zeros((d, d), dtype=RDTYPE)
    Sz = torch.zeros((d, d), dtype=RDTYPE)
    for i in range(d):
        m = float(spin - i)                   # numeric m-value (float)
        Sz[i, i] = m                          # diagonal Sz

        if i < d - 1:
            # coefficient for the transition m -> m-1
            CScoeff = torch.sqrt(spin * (spin + 1) - m * (m - 1))  # real CS-coeff/2
            Splus[i, i + 1] = CScoeff   # S+ raises m by 1
            Sminus[i + 1, i] = CScoeff  # S- lowers m by 1

    print(Splus,Sminus,Sz)

    SdotS = torch.tensor(oe.contract("ij,kl->ijkl", Splus, Sminus) * 0.5
                        +oe.contract("ij,kl->ijkl", Sminus, Splus) * 0.5
                        +oe.contract("ij,kl->ijkl", Sz, Sz), dtype=TENSORDTYPE,
                        device=DEVICE)
    return J * SdotS


    
