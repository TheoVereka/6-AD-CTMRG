"""
test_fifth_term_partial_svd.py — Rigorous tests for arXiv:2311.11894v3 5th term
=================================================================================

PURPOSE
-------
Prove that the paper's 5th term (Γ^{trunc}) makes a genuine PARTIAL SVD (storing
only U_k, S_k, V_k plus the input matrix A) give EXACT gradients for truncated
SVD, WITHOUT needing the full thin SVD.

This was the user's original intuition, and it is CORRECT.  The full-thin zero-
padding approach from the previous session is merely one *equivalent* shortcut
but is NOT the point of the paper and is wasteful (forces full O(mn·min(m,n)) SVD
because sparse SVD cannot be used).

MATHEMATICAL DERIVATION
-----------------------
A = U S V† + U_⊥ S_⊥ V_⊥†    (rank-k truncated SVD; B ≡ A - USV†  is the residual)

The paper (Appendix on SVD) gives:

  5th term:  Γ_A^{trunc} = (I - UU†) γ V† + U γ̃†(I - VV†)

where γ (m×k) and γ̃ (n×k) satisfy the coupled system:

  Γ_U = γ S  -  A(I - VV†) γ̃         ...(*)
  Γ_V = γ̃ S  -  A†(I - UU†) γ        ...(**)

with B ≡ A(I-VV†) = A - USV† and B† = A†(I-UU†) = A† - VSU†.

DECOUPLED SOLUTION (column-by-column)
--------------------------------------
From (*):  γ_j = (Γ_U_j + B γ̃_j) / s_j
Into (**): (s_j² I_n - B†B) γ̃_j = s_j Γ_V_j + B† Γ_U_j

Since s_j are the RETAINED (largest) singular values and eigenvalues of B†B are
the DISCARDED squared singular values, s_j² > all eigenvalues of B†B, so the
system is positive definite and well-conditioned.

BATCH SOLUTION VIA EVD
-----------------------
Let R = B†B = Q Λ Q† (eigendecomposition, Λ sorted ascending).

For all columns simultaneously:
  γ̃ = Q  [(S² ⊗ 1_n - 1_k ⊗ Λ)^{-1}  ⊙  (Q† RHS)]      (n×k)
  γ  = (Γ_U + B γ̃) diag(S)^{-1}                           (m×k)

where RHS = Γ_V · diag(S) + B† Γ_U  (n×k),
and   denom[i,j] = S[j]² - Λ[i]  (n×k, element-wise).

WHAT IS TESTED
--------------
1.  SVD_With5thTerm — partial SVD + standard 4 terms (k×k F-matrix) + 5th term
    via Sylvester solve.  Stores A + U_k + S_k + V_k (NO full thin SVD).
    → must pass all gradient checks.

2.  SVD_FullThin — full thin SVD + zero-padding of upstream grads.
    → also must pass (for comparison / regression check).

3.  SVD_NaiveKOnly — partial SVD + standard 4 terms (k×k) + approximate
    out-of-subspace term WITHOUT the Sylvester correction.
    → must FAIL on projector-type losses (demonstrating why the 5th term matters).

4.  Equivalence: SVD_With5thTerm ≡ SVD_FullThin gradient-wise (they are two
    mathematically equivalent ways to compute the exact gradient).

NOTATIONS
---------
m, n   : matrix dimensions
k      : truncation rank (k < min(m,n) for genuine truncation)
k_full : min(m, n)
gu     : upstream gradient dL/dU (m×k, complex allowed)
gv     : upstream gradient dL/dV (n×k, complex allowed)
gs     : upstream gradient dL/dS (k, real)
"""

import sys
import numpy as np
import torch
import pytest

# ---------------------------------------------------------------------------
# Numerical helpers (regularised inversions identical to core_unrestricted.py)
# ---------------------------------------------------------------------------

def _safe_inv(x, eps):
    """x / (x² + ε²)  — regularised reciprocal."""
    return x / (x ** 2 + eps ** 2)


def _safe_inv2(x, eps):
    """Hard-threshold reciprocal: returns 0 when |x| < eps."""
    out = x.clone()
    out[out.abs() < eps] = float('inf')
    return out.pow(-1)


SVD_REG_EPS = 1e-12


# ===========================================================================
#  5th-term Sylvester solver
# ===========================================================================

def _solve_fifth_term(A, U, S, V, gu, gv, eps):
    """
    Compute  Γ_A^{trunc}  via the coupled Sylvester system from arXiv:2311.11894v3.

    Parameters
    ----------
    A   : (m, n) complex tensor — the ORIGINAL input matrix (saved in forward)
    U   : (m, k) — retained left singular vectors
    S   : (k,)   — retained singular values (real, descending)
    V   : (n, k) — retained right singular vectors
    gu  : (m, k) upstream gradient dL/dU  (or zeros if None)
    gv  : (n, k) upstream gradient dL/dV  (or zeros if None)
    eps : float  — regularisation threshold

    Returns
    -------
    trunc_term : (m, n) complex tensor
    """
    m, n = A.shape
    k = S.shape[0]
    Uh = U.conj().t()   # k×m
    Vh = V.conj().t()   # k×n

    # ---- residual B = A(I - VV†) = A - USV†  (m×n) -------------------------
    B  = A - U @ torch.diag(S.to(A.dtype)) @ Vh   # m×n
    Bh = B.conj().t()                               # n×m

    # ---- R = B†B  (n×n, positive semi-definite) ----------------------------
    R = Bh @ B
    R = (R + R.conj().t()) * 0.5   # symmetrize numerically

    # ---- EVD of R = Q Λ Q†  (Λ ascending) ----------------------------------
    Lambda, Q = torch.linalg.eigh(R)   # Lambda: n, Q: n×n
    Qh = Q.conj().t()

    # ---- RHS_j = s_j · gv_j + B† · gu_j  →  RHS matrix (n×k) --------------
    RHS = gv * S.to(gv.dtype).unsqueeze(0) + Bh @ gu   # n×k

    # ---- denom[i,j] = S[j]² - Λ[i]  (n×k) ---------------------------------
    S2     = S ** 2                                        # k (real)
    denom  = S2.unsqueeze(0) - Lambda.unsqueeze(1)         # (1,k) - (n,1) = n×k

    # Regularise near-zero denominators (split-multiplet edge case).
    # In practice, denom > 0 for all i,j because retained singular values
    # are strictly larger than all discarded ones (proper truncation).
    denom_reg = torch.where(
        denom.abs() >= eps,
        denom,
        torch.full_like(denom, eps))

    # ---- solve: γ̃ = Q [(Q†RHS) / denom]  (n×k) ----------------------------
    gamma_tilde = Q @ (Qh @ RHS / denom_reg)     # n×k

    # ---- back-substitute: γ_j = (gu_j + B γ̃_j) / s_j  (m×k) --------------
    gamma = (gu + B @ gamma_tilde) / S.to(A.dtype).unsqueeze(0)   # m×k

    # ---- assemble 5th term: (I-UU†)γV† + Uγ̃†(I-VV†) ----------------------
    PU_gamma          = gamma - U @ (Uh @ gamma)          # (I-UU†)γ : m×k
    gamma_tilde_h     = gamma_tilde.conj().t()            # k×n
    gtilde_h_PV       = gamma_tilde_h - (gamma_tilde_h @ V) @ Vh  # k×n

    trunc_term = PU_gamma @ Vh + U @ gtilde_h_PV           # m×n
    return trunc_term


# ===========================================================================
#  SVD autograd Functions
# ===========================================================================

class SVD_With5thTerm(torch.autograd.Function):
    """
    Correct truncated SVD backward via the paper's 5th-term Sylvester approach.

    forward:  computes partial SVD, stores U_k, S_k, V_k, AND the original A.
              (In production this could use ARPACK for O(mn·k) cost; here we use
               full numpy SVD but only RETURN k triples, to validate the math.)
    backward: standard 4-term backward (k×k F-matrix only) plus 5th term.

    This is the CORRECT and EFFICIENT approach described in arXiv:2311.11894v3.
    """

    @staticmethod
    def forward(ctx, A, k_trunc, eps_reg):
        m, n   = A.shape
        k_full = min(m, n)
        k      = min(k_trunc, k_full)
        is_cpx = A.is_complex()
        rdtype = torch.float64 if A.dtype in (torch.complex128, torch.float64) else torch.float32
        np_dt  = np.complex128 if is_cpx else np.float64

        A_np = A.detach().cpu().numpy().astype(np_dt, copy=False)
        # In production: replace with partial SVD solver (ARPACK/LOBPCG) for k << k_full.
        # Here we use full SVD but only return the top-k triples to validate the math.
        Uf, Sf, Vhf = np.linalg.svd(A_np, full_matrices=False)

        U_k = torch.as_tensor(Uf[:, :k].copy(),  dtype=A.dtype,  device=A.device)
        S_k = torch.as_tensor(Sf[  :k].copy(),   dtype=rdtype,   device=A.device)
        V_k = torch.as_tensor(Vhf[:k, :].copy(), dtype=A.dtype,  device=A.device).conj().t()

        # ── save U_k, S_k, V_k, AND A (needed for B = A - USV†) ─────────────
        ctx.save_for_backward(U_k, S_k, V_k, A.detach(), eps_reg)
        ctx.k_trunc = k

        return U_k, S_k, V_k   # only k triples returned — no full thin SVD stored

    @staticmethod
    def backward(ctx, gu, gsigma, gv):
        U, S, V, A, eps_reg = ctx.saved_tensors
        m, n = A.shape
        k    = ctx.k_trunc

        _eps    = max(eps_reg.item() * S[0].item(), 1e-30)
        sig_inv = _safe_inv2(S.clone(), _eps)

        Uh = U.conj().t()   # k×m
        Vh = V.conj().t()   # k×n

        # ── 1st term: σ-gradient ─────────────────────────────────────────────
        if gsigma is not None:
            sigma_term = U * gsigma.to(U.dtype).unsqueeze(-2) @ Vh
        else:
            sigma_term = torch.zeros(m, n, dtype=A.dtype, device=A.device)

        if gu is None and gv is None:
            return sigma_term, None, None

        # F, G matrices: k×k
        F = _safe_inv(S.unsqueeze(-2) - S.unsqueeze(-1), _eps)
        F.diagonal(0, -2, -1).fill_(0)
        G = _safe_inv(S.unsqueeze(-2) + S.unsqueeze(-1), _eps)
        G.diagonal(0, -2, -1).fill_(0)

        # ── 2nd term: U_1-rotation (within kept subspace) ────────────────────
        if gu is not None:
            guh    = gu.conj().t()
            u_term = U @ ((F + G) * (Uh @ gu - guh @ U)) * 0.5 @ Vh
        else:
            u_term = torch.zeros(m, n, dtype=A.dtype, device=A.device)

        # ── 3rd term: V_1-rotation (within kept subspace) ────────────────────
        if gv is not None:
            gvh    = gv.conj().t()
            v_term = U @ ((F - G) * (Vh @ gv - gvh @ V)) * 0.5 @ Vh
        else:
            v_term = torch.zeros(m, n, dtype=A.dtype, device=A.device)

        # ── 4th term: complex phase correction (arXiv:1909.02659) ─────────────
        # NOTE: arXiv:2311.11894v3 uses an equivalent "diag_term" from Γ_V side
        # instead.  Using BOTH would double-count.  We keep the standard 1909
        # convention (phase from Γ_U diagonal) to match core_unrestricted.py.
        phase_term = torch.zeros(m, n, dtype=A.dtype, device=A.device)
        if A.is_complex() and gu is not None:
            L = (Uh @ gu).diagonal(0, -2, -1).clone()
            L.real.zero_()
            L.imag.mul_(sig_inv)
            phase_term = (U * L.unsqueeze(-2)) @ Vh

        # ── 5th term: Sylvester-system solution (THE NEW TERM) ───────────────
        _gu = gu if gu is not None else torch.zeros(m, k, dtype=A.dtype, device=A.device)
        _gv = gv if gv is not None else torch.zeros(n, k, dtype=A.dtype, device=A.device)
        trunc_term = _solve_fifth_term(A, U, S, V, _gu, _gv, _eps)

        dA = sigma_term + u_term + v_term + phase_term + trunc_term
        return dA, None, None


class SVD_FullThin(torch.autograd.Function):
    """
    Full-thin SVD with zero-padded upstream grads: mathematically equivalent to
    SVD_With5thTerm, but stores all k_full singular triples instead of A.
    Used here as a reference / regression check.
    """

    @staticmethod
    def forward(ctx, A, k_trunc, eps_reg):
        m, n   = A.shape
        k_full = min(m, n)
        k      = min(k_trunc, k_full)
        is_cpx = A.is_complex()
        rdtype = torch.float64 if A.dtype in (torch.complex128, torch.float64) else torch.float32
        np_dt  = np.complex128 if is_cpx else np.float64

        A_np = A.detach().cpu().numpy().astype(np_dt, copy=False)
        Uf, Sf, Vhf = np.linalg.svd(A_np, full_matrices=False)

        U_full = torch.as_tensor(Uf.copy(),  dtype=A.dtype,  device=A.device)
        S_full = torch.as_tensor(Sf.copy(),  dtype=rdtype,   device=A.device)
        V_full = torch.as_tensor(Vhf.copy(), dtype=A.dtype,  device=A.device).conj().t()

        ctx.save_for_backward(U_full, S_full, V_full, eps_reg)
        ctx.k_trunc = k
        ctx.k_full  = k_full

        return U_full[:, :k], S_full[:k], V_full[:, :k]

    @staticmethod
    def backward(ctx, gu, gsigma, gv):
        U_f, S_f, V_f, eps_reg = ctx.saved_tensors
        k      = ctx.k_trunc
        k_full = ctx.k_full
        m = U_f.shape[0]; n = V_f.shape[0]

        _eps    = max(eps_reg.item() * S_f[0].item(), 1e-30)
        sig_inv = _safe_inv2(S_f.clone(), _eps)

        # zero-pad upstream grads from k to k_full
        def _pad(g, extra):
            if g is None or extra == 0:
                return g
            return torch.cat([g, g.new_zeros(*g.shape[:-1], extra)], dim=-1)

        extra = k_full - k
        gu_p  = _pad(gu,     extra)
        gv_p  = _pad(gv,     extra)
        gs_p  = (torch.cat([gsigma, gsigma.new_zeros(extra)]) if gsigma is not None else None)

        Uh = U_f.conj().t(); Vh = V_f.conj().t()

        if gs_p is not None:
            sigma_term = U_f * gs_p.to(U_f.dtype).unsqueeze(-2) @ Vh
        else:
            sigma_term = torch.zeros(m, n, dtype=U_f.dtype, device=U_f.device)

        if gu_p is None and gv_p is None:
            return sigma_term, None, None

        F = _safe_inv(S_f.unsqueeze(-2) - S_f.unsqueeze(-1), _eps)
        F.diagonal(0, -2, -1).fill_(0)
        G = _safe_inv(S_f.unsqueeze(-2) + S_f.unsqueeze(-1), _eps)
        G.diagonal(0, -2, -1).fill_(0)

        if gu_p is not None:
            guh    = gu_p.conj().t()
            u_term = U_f @ ((F + G) * (Uh @ gu_p - guh @ U_f)) * 0.5
            if m > k_full:
                P = torch.eye(m, dtype=U_f.dtype, device=U_f.device) - U_f @ Uh
                u_term = u_term + P @ (gu_p * sig_inv.unsqueeze(-2))
            u_term = u_term @ Vh
        else:
            u_term = torch.zeros(m, n, dtype=U_f.dtype, device=U_f.device)

        if gv_p is not None:
            gvh    = gv_p.conj().t()
            # Apply @ Vh FIRST to convert (k_full,k_full) → (k_full,n),
            # then add out-of-subspace which is also (k_full,n).
            v_term = ((F - G) * (Vh @ gv_p - gvh @ V_f)) @ Vh * 0.5
            if n > k_full:
                P = -V_f @ Vh
                P.diagonal(0, -2, -1).add_(1)   # n×n projector I-VV†
                v_term = v_term + sig_inv.unsqueeze(-1) * (gvh @ P)
            v_term = U_f @ v_term
        else:
            v_term = torch.zeros(m, n, dtype=U_f.dtype, device=U_f.device)

        phase_term = torch.zeros(m, n, dtype=U_f.dtype, device=U_f.device)
        if U_f.is_complex() and gu_p is not None:
            L = (Uh @ gu_p).diagonal(0, -2, -1).clone()
            L.real.zero_()
            L.imag.mul_(sig_inv)
            phase_term = (U_f * L.unsqueeze(-2)) @ Vh

        return sigma_term + u_term + v_term + phase_term, None, None


class SVD_NaiveKOnly(torch.autograd.Function):
    """
    INTENTIONALLY WRONG: partial SVD + standard 4 terms, no 5th term or zero-padding.
    The out-of-subspace term uses the APPROXIMATE formula (ignores residual A - USV†).
    Documents WHY the 5th term is necessary.
    """

    @staticmethod
    def forward(ctx, A, k_trunc, eps_reg):
        m, n   = A.shape
        k_full = min(m, n)
        k      = min(k_trunc, k_full)
        is_cpx = A.is_complex()
        rdtype = torch.float64 if A.dtype in (torch.complex128, torch.float64) else torch.float32
        np_dt  = np.complex128 if is_cpx else np.float64

        A_np = A.detach().cpu().numpy().astype(np_dt, copy=False)
        Uf, Sf, Vhf = np.linalg.svd(A_np, full_matrices=False)

        U_k = torch.as_tensor(Uf[:, :k].copy(),  dtype=A.dtype,  device=A.device)
        S_k = torch.as_tensor(Sf[  :k].copy(),   dtype=rdtype,   device=A.device)
        V_k = torch.as_tensor(Vhf[:k, :].copy(), dtype=A.dtype,  device=A.device).conj().t()

        ctx.save_for_backward(U_k, S_k, V_k, eps_reg)
        ctx.k_trunc = k
        return U_k, S_k, V_k

    @staticmethod
    def backward(ctx, gu, gsigma, gv):
        U, S, V, eps_reg = ctx.saved_tensors
        m = U.shape[0]; n = V.shape[0]; k = ctx.k_trunc
        _eps    = max(eps_reg.item() * S[0].item(), 1e-30)
        sig_inv = _safe_inv2(S.clone(), _eps)

        Uh = U.conj().t(); Vh = V.conj().t()

        if gsigma is not None:
            sigma_term = U * gsigma.to(U.dtype).unsqueeze(-2) @ Vh
        else:
            sigma_term = torch.zeros(m, n, dtype=U.dtype, device=U.device)

        if gu is None and gv is None:
            return sigma_term, None, None

        F = _safe_inv(S.unsqueeze(-2) - S.unsqueeze(-1), _eps)
        F.diagonal(0, -2, -1).fill_(0)
        G = _safe_inv(S.unsqueeze(-2) + S.unsqueeze(-1), _eps)
        G.diagonal(0, -2, -1).fill_(0)

        if gu is not None:
            guh    = gu.conj().t()
            u_term = U @ ((F + G) * (Uh @ gu - guh @ U)) * 0.5
            if m > k:   # approximate out-of-subspace: ignores the A residual
                P = torch.eye(m, dtype=U.dtype, device=U.device) - U @ Uh
                u_term = u_term + P @ (gu * sig_inv.unsqueeze(-2))
            u_term = u_term @ Vh
        else:
            u_term = torch.zeros(m, n, dtype=U.dtype, device=U.device)

        if gv is not None:
            gvh    = gv.conj().t()
            # Apply @ Vh FIRST, then add out-of-subspace. Approximate: ignores A residual.
            v_term = ((F - G) * (Vh @ gv - gvh @ V)) @ Vh * 0.5
            if n > k:   # approximate out-of-subspace: ignores the A residual
                P = -V @ Vh
                P.diagonal(0, -2, -1).add_(1)  # n×n
                v_term = v_term + sig_inv.unsqueeze(-1) * (gvh @ P)
            v_term = U @ v_term
        else:
            v_term = torch.zeros(m, n, dtype=U.dtype, device=U.device)

        phase_term = torch.zeros(m, n, dtype=U.dtype, device=U.device)
        if U.is_complex() and gu is not None:
            L = (Uh @ gu).diagonal(0, -2, -1).clone()
            L.real.zero_()
            L.imag.mul_(sig_inv)
            phase_term = (U * L.unsqueeze(-2)) @ Vh

        dA = sigma_term + u_term + v_term + phase_term
        return dA, None, None


# ===========================================================================
#  Gauge-invariant loss functions  (depend only on U diag(S) V†)
# ===========================================================================

def _make_A(m, n, seed, dtype=torch.complex128):
    g = torch.Generator(); g.manual_seed(seed)
    return torch.randn(m, n, dtype=dtype, generator=g) / np.sqrt(max(m, n))


def _make_B(m, n, seed, dtype=torch.complex128):
    """Fixed random matrix independent of A — used as 'data' in loss functions."""
    g = torch.Generator(); g.manual_seed(seed + 10000)
    return (torch.randn(m, n, dtype=dtype, generator=g) * 0.5).detach()


def _recon_loss(U, S, V, B):
    """||U diag(S) V† - B||_F²  — gauge-invariant, depends on USV†."""
    diff = U * S.to(U.dtype).unsqueeze(-2) @ V.conj().t() - B
    return (diff.real ** 2 + diff.imag ** 2).sum()


def _ssq_loss(U, S, V):
    """Σ_i S_i²  — depends only on S, still gauge-invariant."""
    return (S ** 2).sum()


def _recon_loss_real(U, S, V, B):
    diff = U * S.unsqueeze(-2) @ V.t() - B
    return (diff ** 2).sum()


# ===========================================================================
#  Numerical Jacobian (PyTorch convention, no 1/2 factor)
# ===========================================================================

def _num_grad(loss_fn, A, eps_fd=1e-6):
    """
    Finite-difference Jacobian: ng[i,j] = dL/dRe(A_ij) + i·dL/dIm(A_ij).
    Matches the convention of A.grad after loss.backward().
    """
    m, n = A.shape
    grad  = torch.zeros_like(A)
    A0    = A.detach().clone()
    for i in range(m):
        for j in range(n):
            if A.is_complex():
                # real part
                Ap = A0.clone(); Ap[i, j] = Ap[i, j] + eps_fd
                Am = A0.clone(); Am[i, j] = Am[i, j] - eps_fd
                grad[i, j]  = (loss_fn(Ap) - loss_fn(Am)) / (2 * eps_fd)
                # imaginary part
                Ap = A0.clone(); Ap[i, j] = Ap[i, j] + 1j * eps_fd
                Am = A0.clone(); Am[i, j] = Am[i, j] - 1j * eps_fd
                grad[i, j] += 1j * (loss_fn(Ap) - loss_fn(Am)) / (2 * eps_fd)
            else:
                Ap = A0.clone(); Ap[i, j] = Ap[i, j] + eps_fd
                Am = A0.clone(); Am[i, j] = Am[i, j] - eps_fd
                grad[i, j]  = (loss_fn(Ap) - loss_fn(Am)) / (2 * eps_fd)
    return grad


def _check(loss_fn, A, k, eps_fd=1e-6, rtol=5e-3, atol=1e-5, fn_cls=SVD_With5thTerm):
    """
    Compare analytic gradient (via fn_cls.apply) to numerical Jacobian.

    Returns (ok, max_abs_err, rel_err).
    """
    eps_t = torch.tensor(SVD_REG_EPS, dtype=torch.float64)
    A_leaf = A.detach().clone().requires_grad_(True)
    U, S, V = fn_cls.apply(A_leaf, k, eps_t)
    loss_fn(U, S, V).backward()
    ag = A_leaf.grad.clone()

    def _ls(Ap):
        with torch.no_grad():
            U2, S2, V2 = fn_cls.apply(Ap, k, eps_t)
        return loss_fn(U2, S2, V2).item()

    ng        = _num_grad(_ls, A, eps_fd)
    max_err   = (ag - ng).abs().max().item()
    rel_err   = max_err / (ng.abs().max().item() + 1e-30)
    ok        = (max_err < atol) or (rel_err < rtol)
    return ok, max_err, rel_err


# ===========================================================================
#  Tests
# ===========================================================================

class TestSylvesterSolverDirectly:
    """
    Unit-test _solve_fifth_term by verifying it satisfies the defining equations:
      Γ_U = γ S - A(I-VV†)γ̃   and   Γ_V = γ̃ S - A†(I-UU†)γ
    """

    @pytest.mark.parametrize("m,n,k,seed", [
        (6, 6, 2, 1), (8, 5, 3, 2), (5, 8, 3, 3), (10, 8, 4, 4), (8, 10, 5, 5),
    ])
    def test_sylvester_residual_complex(self, m, n, k, seed):
        """γ, γ̃ must satisfy the defining Sylvester equations exactly."""
        torch.manual_seed(seed)
        A = (_make_A(m, n, seed) * 0.5 + _make_A(m, n, seed + 1) * 0.5j).to(torch.complex128)
        Uf, Sf, Vhf = np.linalg.svd(A.numpy(), full_matrices=False)
        U = torch.as_tensor(Uf[:, :k].copy(), dtype=torch.complex128)
        S = torch.as_tensor(Sf[:k].copy(), dtype=torch.float64)
        V = torch.as_tensor(Vhf[:k, :].copy(), dtype=torch.complex128).conj().t()

        # random upstream grads
        torch.manual_seed(seed + 999)
        gu = torch.randn(m, k, dtype=torch.complex128) + 1j * torch.randn(m, k, dtype=torch.complex128)
        gv = torch.randn(n, k, dtype=torch.complex128) + 1j * torch.randn(n, k, dtype=torch.complex128)

        eps = 1e-14
        B  = A - U @ torch.diag(S.to(A.dtype)) @ V.conj().t()
        Bh = B.conj().t()
        Uh = U.conj().t(); Vh = V.conj().t()

        # Solve for γ, γ̃ by running _solve_fifth_term partially
        # (reconstruct γ, γ̃ from inside the solver)
        R = (Bh @ B)
        R = (R + R.conj().t()) * 0.5
        Lambda, Q = torch.linalg.eigh(R)
        Qh = Q.conj().t()
        RHS = gv * S.to(gv.dtype).unsqueeze(0) + Bh @ gu
        S2    = S ** 2
        denom = S2.unsqueeze(0) - Lambda.unsqueeze(1)
        denom_reg = torch.where(denom.abs() >= eps, denom, torch.full_like(denom, eps))
        gamma_tilde = Q @ (Qh @ RHS / denom_reg)
        gamma = (gu + B @ gamma_tilde) / S.to(A.dtype).unsqueeze(0)

        # Verify equation (*)  Γ_U  =  γ S  -  A(I-VV†)γ̃
        lhs_star = gamma * S.to(A.dtype).unsqueeze(0) - B @ gamma_tilde
        res_star  = (lhs_star - gu).abs().max().item()

        # Verify equation (**)  Γ_V  =  γ̃ S  -  A†(I-UU†)γ
        lhs_dstar = gamma_tilde * S.to(A.dtype).unsqueeze(0) - Bh @ gamma
        res_dstar  = (lhs_dstar - gv).abs().max().item()

        assert res_star  < 1e-8, f"(*) residual={res_star:.3e}"
        assert res_dstar < 1e-8, f"(**) residual={res_dstar:.3e}"

    @pytest.mark.parametrize("m,n,k,seed", [
        (6, 6, 3, 10), (8, 5, 2, 11), (5, 8, 4, 12),
    ])
    def test_sylvester_residual_real(self, m, n, k, seed):
        """Same verification for real dtype."""
        Af = _make_A(m, n, seed, dtype=torch.float64)
        Uf, Sf, Vhf = np.linalg.svd(Af.numpy(), full_matrices=False)
        U = torch.as_tensor(Uf[:, :k].copy(), dtype=torch.float64)
        S = torch.as_tensor(Sf[:k].copy(), dtype=torch.float64)
        V = torch.as_tensor(Vhf[:k, :].copy(), dtype=torch.float64).t()

        torch.manual_seed(seed + 999)
        gu = torch.randn(m, k, dtype=torch.float64)
        gv = torch.randn(n, k, dtype=torch.float64)

        eps = 1e-14
        B  = Af - U @ torch.diag(S) @ V.t()
        Bh = B.t()

        R = (Bh @ B + (Bh @ B).t()) * 0.5
        Lambda, Q = torch.linalg.eigh(R)
        Qh = Q.t()
        RHS = gv * S.unsqueeze(0) + Bh @ gu
        S2    = S ** 2
        denom = S2.unsqueeze(0) - Lambda.unsqueeze(1)
        denom_reg = torch.where(denom.abs() >= eps, denom, torch.full_like(denom, eps))
        gamma_tilde = Q @ (Qh @ RHS / denom_reg)
        gamma = (gu + B @ gamma_tilde) / S.unsqueeze(0)

        lhs  = gamma * S.unsqueeze(0) - B @ gamma_tilde
        lhs2 = gamma_tilde * S.unsqueeze(0) - Bh @ gamma
        assert (lhs  - gu).abs().max().item() < 1e-8
        assert (lhs2 - gv).abs().max().item() < 1e-8


class TestWith5thTermVsNumerical:
    """
    SVD_With5thTerm (partial SVD + Sylvester 5th term) vs numerical Jacobian.
    This is the main validation: proves the implementation is exact.
    """

    @pytest.mark.parametrize("m,n,k,seed", [
        # square
        (5, 5, 1, 1), (5, 5, 2, 2), (5, 5, 3, 3), (5, 5, 4, 4),
        # tall
        (8, 5, 1, 10), (8, 5, 2, 11), (8, 5, 3, 12), (8, 5, 4, 13),
        # wide
        (5, 8, 1, 20), (5, 8, 2, 21), (5, 8, 3, 22), (5, 8, 4, 23),
        # larger
        (10, 8, 3, 30), (8, 10, 4, 31), (10, 10, 5, 32), (12, 8, 5, 33),
    ])
    def test_recon_complex128(self, m, n, k, seed):
        A = _make_A(m, n, seed)
        B = _make_B(m, n, seed)
        ok, err, rel = _check(lambda U, S, V: _recon_loss(U, S, V, B), A, k)
        assert ok, f"({m},{n},k={k}) max={err:.3e} rel={rel:.3e}"

    @pytest.mark.parametrize("m,n,k,seed", [
        (6, 6, 2, 40), (8, 5, 3, 41), (5, 8, 3, 42), (10, 8, 4, 43),
    ])
    def test_ssq_complex128(self, m, n, k, seed):
        A = _make_A(m, n, seed)
        ok, err, rel = _check(_ssq_loss, A, k)
        assert ok, f"({m},{n},k={k}) max={err:.3e} rel={rel:.3e}"

    @pytest.mark.parametrize("m,n,k,seed", [
        (6, 6, 2, 50), (8, 5, 3, 51), (5, 8, 3, 52), (10, 8, 4, 53), (8, 10, 4, 54),
    ])
    def test_recon_float64(self, m, n, k, seed):
        A = _make_A(m, n, seed, dtype=torch.float64)
        B = _make_B(m, n, seed, dtype=torch.float64)
        ok, err, rel = _check(lambda U, S, V: _recon_loss_real(U, S, V, B), A, k)
        assert ok, f"({m},{n},k={k}) max={err:.3e} rel={rel:.3e}"

    # k = 1 edge case
    @pytest.mark.parametrize("m,n,seed", [(8, 6, 60), (6, 8, 61), (10, 10, 62)])
    def test_k1(self, m, n, seed):
        A = _make_A(m, n, seed); B = _make_B(m, n, seed)
        ok, err, rel = _check(lambda U, S, V: _recon_loss(U, S, V, B), A, 1,
                               rtol=2e-2, atol=1e-4)
        assert ok, f"({m},{n},k=1) max={err:.3e}"

    # k = k_full (no truncation): 5th term should vanish
    @pytest.mark.parametrize("m,n,seed", [(5, 5, 70), (8, 5, 71), (5, 8, 72)])
    def test_k_full(self, m, n, seed):
        k = min(m, n)
        A = _make_A(m, n, seed); B = _make_B(m, n, seed)
        ok, err, rel = _check(lambda U, S, V: _recon_loss(U, S, V, B), A, k)
        assert ok, f"k=k_full({m},{n}) max={err:.3e}"


class TestEquivalence5thTermVsFullThin:
    """
    Prove SVD_With5thTerm ≡ SVD_FullThin: same gradient, down to floating-point
    precision.  Both are mathematically equivalent ways to compute the exact
    truncated SVD backward.  (SVD_With5thTerm can use partial SVD in forward;
    SVD_FullThin requires full thin SVD.)
    """

    @pytest.mark.parametrize("m,n,k,seed", [
        (6, 6, 2, 1), (6, 6, 3, 2), (8, 5, 2, 3), (5, 8, 3, 4),
        (8, 8, 1, 5), (10, 8, 3, 6), (8, 10, 4, 7), (10, 10, 5, 8),
        (12, 8, 4, 9), (8, 12, 5, 10),
    ])
    def test_identical_gradients(self, m, n, k, seed):
        eps_t = torch.tensor(SVD_REG_EPS, dtype=torch.float64)
        A = _make_A(m, n, seed)
        B = _make_B(m, n, seed)
        loss_fn = lambda U, S, V: _recon_loss(U, S, V, B)

        A1 = A.detach().clone().requires_grad_(True)
        U1, S1, V1 = SVD_With5thTerm.apply(A1, k, eps_t)
        loss_fn(U1, S1, V1).backward()
        g1 = A1.grad.clone()

        A2 = A.detach().clone().requires_grad_(True)
        U2, S2, V2 = SVD_FullThin.apply(A2, k, eps_t)
        loss_fn(U2, S2, V2).backward()
        g2 = A2.grad.clone()

        diff = (g1 - g2).abs().max().item()
        relmax = g1.abs().max().item()
        rel  = diff / (relmax + 1e-30)

        assert rel < 1e-5, (
            f"({m},{n},k={k}): 5th-term vs full-thin grad diff={diff:.3e} rel={rel:.3e}"
        )

    @pytest.mark.parametrize("m,n,k,seed", [
        (6, 5, 2, 20), (5, 6, 3, 21), (8, 8, 4, 22),
    ])
    def test_identical_gradients_real(self, m, n, k, seed):
        eps_t = torch.tensor(SVD_REG_EPS, dtype=torch.float64)
        A = _make_A(m, n, seed, dtype=torch.float64)
        B = _make_B(m, n, seed, dtype=torch.float64)
        loss_fn = lambda U, S, V: _recon_loss_real(U, S, V, B)

        A1 = A.detach().clone().requires_grad_(True)
        U1, S1, V1 = SVD_With5thTerm.apply(A1, k, eps_t)
        loss_fn(U1, S1, V1).backward()
        g1 = A1.grad.clone()

        A2 = A.detach().clone().requires_grad_(True)
        U2, S2, V2 = SVD_FullThin.apply(A2, k, eps_t)
        loss_fn(U2, S2, V2).backward()
        g2 = A2.grad.clone()

        diff = (g1 - g2).abs().max().item()
        rel  = diff / (g1.abs().max().item() + 1e-30)
        assert rel < 1e-5, f"({m},{n},k={k}) diff={diff:.3e} rel={rel:.3e}"


class TestNaiveFailsWithout5thTerm:
    """
    Document that using only the k-truncated vectors WITHOUT the 5th term gives
    WRONG gradients for gauge-invariant losses when k < k_full.
    The error is typically 10-100% of the true gradient magnitude.

    We compare naive gradient against SVD_With5thTerm (verified correct against
    numerical finite differences) instead of against its own numerical Jacobian,
    because the latter can suffer from phase-flip inconsistencies for complex SVD.
    """

    @pytest.mark.parametrize("m,n,k,seed", [
        (8, 6, 3, 1), (8, 6, 2, 2), (6, 8, 3, 3), (10, 8, 4, 4), (8, 10, 3, 5),
    ])
    def test_naive_fails(self, m, n, k, seed):
        eps_t = torch.tensor(SVD_REG_EPS, dtype=torch.float64)
        A = _make_A(m, n, seed)
        B = _make_B(m, n, seed)
        loss_fn = lambda U, S, V: _recon_loss(U, S, V, B)

        # correct gradient (5th term, verified by TestWith5thTermVsNumerical)
        A_5th = A.detach().clone().requires_grad_(True)
        U5, S5, V5 = SVD_With5thTerm.apply(A_5th, k, eps_t)
        loss_fn(U5, S5, V5).backward()
        g_5th = A_5th.grad.clone()

        # naive gradient
        A_naive = A.detach().clone().requires_grad_(True)
        Un, Sn, Vn = SVD_NaiveKOnly.apply(A_naive, k, eps_t)
        loss_fn(Un, Sn, Vn).backward()
        g_naive = A_naive.grad.clone()

        diff = (g_5th - g_naive).abs().max().item()
        rel  = diff / (g_5th.abs().max().item() + 1e-30)

        # 5th term must be correct vs own numerical:
        ok_5th, err_5th, _ = _check(loss_fn, A, k, fn_cls=SVD_With5thTerm)

        # Naive must deviate from the correct gradient significantly:
        assert rel > 1e-2, (
            f"Expected naive to disagree with correct grad by >1%, "
            f"but rel diff={rel:.3e}  (diff={diff:.3e})")
        assert ok_5th, f"5th term should pass but got err={err_5th:.3e}"


class TestMassiveRandom5thTerm:
    """Randomised sweep over matrix shapes and truncation ranks."""

    @pytest.mark.parametrize("seed", list(range(50)))
    def test_complex128_random(self, seed):
        rng = np.random.default_rng(seed)
        m   = int(rng.integers(4, 14))
        n   = int(rng.integers(4, 14))
        k   = int(rng.integers(1, min(m, n)))
        A   = _make_A(m, n, seed)
        B   = _make_B(m, n, seed)
        ok, err, rel = _check(lambda U, S, V: _recon_loss(U, S, V, B), A, k)
        assert ok, f"seed={seed} ({m},{n},k={k}) max={err:.3e} rel={rel:.3e}"

    @pytest.mark.parametrize("seed", list(range(20)))
    def test_float64_random(self, seed):
        rng = np.random.default_rng(seed + 200)
        m   = int(rng.integers(4, 14))
        n   = int(rng.integers(4, 14))
        k   = int(rng.integers(1, min(m, n)))
        A   = _make_A(m, n, seed, dtype=torch.float64)
        B   = _make_B(m, n, seed, dtype=torch.float64)
        ok, err, rel = _check(lambda U, S, V: _recon_loss_real(U, S, V, B), A, k,
                               fn_cls=SVD_With5thTerm)
        assert ok, f"seed={seed} ({m},{n},k={k}) max={err:.3e} rel={rel:.3e}"


class TestCTMRGLike5thTerm:
    """
    CTMRG-style matrices: taller/wider shapes with larger chi-like bond dimensions.
    Mimics typical shapes encountered in 6-site CTMRG: (chi·D², chi) style.
    """

    @pytest.mark.parametrize("chi,D,k,seed", [
        (4,  2, 4,  1),   # chi=4, D=2: (16,4) matrix, k=4
        (8,  2, 8,  2),   # chi=8, D=2: (32,8) matrix, k=8
        (9,  2, 6,  3),   # chi=9, D=2: (36,9) matrix, k=6
        (16, 2, 10, 4),   # chi=16, D=2: (64,16) matrix, k=10
        (4,  3, 4,  5),   # chi=4, D=3: (36,4) matrix, k=4
        (8,  3, 6,  6),   # chi=8, D=3: (72,8) matrix, k=6
    ])
    def test_ctmrg_shape(self, chi, D, k, seed):
        m = chi * D * D   # tall dimension (chi·D²)
        n = chi           # short dimension
        k = min(k, min(m, n))
        A = _make_A(m, n, seed)
        B = _make_B(m, n, seed)
        ok, err, rel = _check(lambda U, S, V: _recon_loss(U, S, V, B), A, k,
                               rtol=1e-2, atol=1e-4)
        assert ok, f"chi={chi},D={D},k={k}: max={err:.3e} rel={rel:.3e}"

    @pytest.mark.parametrize("chi,D,k,seed", [
        (4, 2, 4, 10), (8, 2, 8, 11), (4, 3, 4, 12),
    ])
    def test_ctmrg_equivalence(self, chi, D, k, seed):
        """5th-term approach must match full-thin approach for CTMRG shapes."""
        m = chi * D * D; n = chi
        k = min(k, min(m, n))
        A = _make_A(m, n, seed); B = _make_B(m, n, seed)
        eps_t  = torch.tensor(SVD_REG_EPS, dtype=torch.float64)
        loss_fn = lambda U, S, V: _recon_loss(U, S, V, B)

        A1 = A.detach().clone().requires_grad_(True)
        SVD_With5thTerm.apply(A1, k, eps_t); loss_fn(*SVD_With5thTerm.apply(A1, k, eps_t)).backward()
        # redo cleanly
        A1 = A.detach().clone().requires_grad_(True)
        U1, S1, V1 = SVD_With5thTerm.apply(A1, k, eps_t)
        loss_fn(U1, S1, V1).backward()
        g1 = A1.grad.clone()

        A2 = A.detach().clone().requires_grad_(True)
        U2, S2, V2 = SVD_FullThin.apply(A2, k, eps_t)
        loss_fn(U2, S2, V2).backward()
        g2 = A2.grad.clone()

        rel = (g1 - g2).abs().max().item() / (g1.abs().max().item() + 1e-30)
        assert rel < 1e-4, f"chi={chi},D={D},k={k}: rel={rel:.3e}"


# ===========================================================================
if __name__ == "__main__":
    print("Running test_fifth_term_partial_svd.py ...")
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
