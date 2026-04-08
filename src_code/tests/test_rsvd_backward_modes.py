"""
test_rsvd_backward_modes.py — Tests for rSVD backward dispatch modes
=======================================================================

PURPOSE
-------
Verify correctness of all three rSVD backward modes added for fast gradient
computation:

  'full_svd'  — reference: full thin SVD + zero-padding (exact)
  'neumann'   — Neumann series 5th term: O(2·L·mnk) vs O(N³) exact eigh
  'augmented' — save k+extra singular triples, zero-pad gu/gv to k_aug,
                F,G cross-coupling implicitly captures the 5th term
  'none'      — k-only backward, 5th term dropped (known to be inaccurate)

DESIGN
------
Because 'neumann', 'augmented', and 'none' modes use a randomised SVD forward
(stochastic Q in the range-finder), comparing them directly against numerical
finite-differences or against each other would be noisy — each call draws a
different Q, producing a different approximate SVD.

We therefore split the tests into levels:

  Level 1 – Direct function tests (no randomness):
    _solve_fifth_term_neumann vs _solve_fifth_term_svd, given EXACT
    singular vectors from torch.linalg.svd.  Proves Neumann converges.

  Level 2 – Augmented backward kernel (no randomness):
    Build a mini Function that saves k_aug exact triples; verify that
    zero-padding upstream grads from k to k_aug gives better gradients than
    saving only k triples ('none' analog).

  Level 3 – full_svd end-to-end vs numerical Jacobian (deterministic).

  Level 4 – Smoke tests for all modes through SVD_PROPACK.apply.
"""

import sys
import os
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import core_unrestricted as cu
from core_unrestricted import (SVD_PROPACK, set_rsvd_mode,
                               _solve_fifth_term_svd,
                               _solve_fifth_term_neumann)

DTYPE  = torch.float64
CDTYPE = torch.complex128


# ─── helpers ─────────────────────────────────────────────────────────────────

def _make_mat(m, n, k_ignored, spectrum_decay=0.5, dtype=DTYPE, seed=42):
    """m×n matrix with geometrically decaying singular values σ_i = decay^i."""
    rng = torch.Generator(); rng.manual_seed(seed)
    k_f = min(m, n)
    U_r, _ = torch.linalg.qr(torch.randn(m, k_f, dtype=dtype, generator=rng))
    V_r, _ = torch.linalg.qr(torch.randn(n, k_f, dtype=dtype, generator=rng))
    s = torch.tensor([spectrum_decay ** i for i in range(k_f)], dtype=dtype)
    return (U_r @ torch.diag(s) @ V_r.T).detach()


def _make_complex_mat(m, n, dtype=CDTYPE, seed=42):
    rng = torch.Generator(); rng.manual_seed(seed)
    k_f = min(m, n)
    U_r, _ = torch.linalg.qr(
        torch.randn(m, k_f, dtype=dtype, generator=rng)
        + 1j * torch.randn(m, k_f, dtype=dtype, generator=rng))
    V_r, _ = torch.linalg.qr(
        torch.randn(n, k_f, dtype=dtype, generator=rng)
        + 1j * torch.randn(n, k_f, dtype=dtype, generator=rng))
    s = torch.arange(1, k_f + 1, dtype=torch.float64).flip(0).to(dtype)
    return (U_r @ torch.diag(s) @ V_r.mH).detach()


def _exact_svd_k(A, k):
    """Exact top-k singular triples (deterministic)."""
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    return U[:, :k].contiguous(), S[:k].contiguous(), Vh.mH[:, :k].contiguous()


def _rgu_gv(m, n, k, dtype=DTYPE, seed=0):
    rng = torch.Generator(); rng.manual_seed(seed)
    return (torch.randn(m, k, dtype=dtype, generator=rng),
            torch.randn(n, k, dtype=dtype, generator=rng))


def cosine(a, b):
    af, bf = a.flatten().double(), b.flatten().double()
    return (af @ bf / (af.norm() * bf.norm() + 1e-30)).item()


def rel_err(a, b):
    return ((a - b).norm() / (b.norm() + 1e-30)).item()


@pytest.fixture(autouse=True)
def restore_mode():
    yield
    set_rsvd_mode('full_svd')


# ══════════════════════════════════════════════════════════════════════════════
# 0. Global-state tests
# ══════════════════════════════════════════════════════════════════════════════

def test_default_mode_is_full_svd():
    assert cu._RSVD_BACKWARD_MODE == 'full_svd'


def test_set_rsvd_mode_updates_globals():
    set_rsvd_mode('neumann', neumann_terms=3, augment_extra=7)
    assert cu._RSVD_BACKWARD_MODE == 'neumann'
    assert cu._RSVD_NEUMANN_TERMS == 3
    assert cu._RSVD_AUGMENT_EXTRA == 7


def test_set_rsvd_mode_invalid_raises():
    with pytest.raises(ValueError, match="Unknown rSVD mode"):
        set_rsvd_mode('bad_mode')


# ══════════════════════════════════════════════════════════════════════════════
# 1. Direct _solve_fifth_term_neumann  vs  _solve_fifth_term_svd tests
#    (no rSVD randomness — uses exact U,S,V from torch.linalg.svd)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("m,n,k,decay", [
    (20, 20, 5,  0.05),   # ρ = 0.0025
    (20, 20, 5,  0.2),    # ρ = 0.04
    (20, 20, 5,  0.5),    # ρ = 0.25
    (25, 20, 6,  0.3),    # rectangular
])
def test_neumann_L8_converges_to_exact(m, n, k, decay):
    """
    _solve_fifth_term_neumann L=8 must closely match exact eigh.
    Error bound: ρ^8 / (1-ρ)·||RHS||.
      decay=0.05: ρ^8 < 4e-11  → virtually exact
      decay=0.5:  ρ^8 = 3.9e-3 → ~0.5% relative error
    """
    A = _make_mat(m, n, k, spectrum_decay=decay)
    U, S, V = _exact_svd_k(A, k)
    gu, gv  = _rgu_gv(m, n, k)
    eps_t   = torch.tensor(1e-12 * S[0].item())

    dA_ex = _solve_fifth_term_svd(A, U, S, V, gu, gv, eps_t)
    dA_n8 = _solve_fifth_term_neumann(A, U, S, V, gu, gv, n_terms=8)

    c = cosine(dA_n8, dA_ex)
    r = rel_err(dA_n8, dA_ex)
    assert c >= 0.999, (
        f"Neumann L=8 (m={m},n={n},k={k},decay={decay}): cos={c:.5f}, rel={r:.3e}"
    )


@pytest.mark.parametrize("n_terms,min_cos", [
    (1, 0.80),
    (2, 0.95),
    (4, 0.990),
    (8, 0.999),
])
def test_neumann_convergence_with_terms(n_terms, min_cos):
    """cos(Neumann_L, exact) is monotonically increasing in L."""
    A = _make_mat(20, 20, 5, spectrum_decay=0.4)
    U, S, V = _exact_svd_k(A, 5)
    gu, gv  = _rgu_gv(20, 20, 5)
    eps_t   = torch.tensor(1e-12 * S[0].item())

    dA_ex = _solve_fifth_term_svd(A, U, S, V, gu, gv, eps_t)
    dA_n  = _solve_fifth_term_neumann(A, U, S, V, gu, gv, n_terms)
    c = cosine(dA_n, dA_ex)
    assert c >= min_cos, f"Neumann L={n_terms}: cos={c:.4f} < {min_cos}"


def test_neumann_full_rank_gives_zero():
    """
    When k = min(m,n) (full thin SVD), B = A-USV† = 0.
    Both exact and Neumann must return the zero matrix.
    """
    m = 10
    A = _make_mat(m, m, m, spectrum_decay=0.6)
    U, S, V = _exact_svd_k(A, m)     # full rank: k == m
    gu, gv  = _rgu_gv(m, m, m)
    eps_t   = torch.tensor(1e-12 * S[0].item())

    dA_ex = _solve_fifth_term_svd(A, U, S, V, gu, gv, eps_t)
    dA_n1 = _solve_fifth_term_neumann(A, U, S, V, gu, gv, n_terms=1)

    assert dA_n1.norm() < 1e-8, f"Neumann L=1 full-rank ||dA||={dA_n1.norm():.2e}"
    assert dA_ex.norm()  < 1e-8, f"exact full-rank ||dA||={dA_ex.norm():.2e}"


@pytest.mark.parametrize("decay", [0.05, 0.2, 0.5])
def test_neumann_4terms_default_accuracy(decay):
    """L=4 (default) meets ≥0.99 cosine for CTMRG-typical spectra (ρ≤0.25)."""
    A = _make_mat(25, 25, 5, spectrum_decay=decay)
    U, S, V = _exact_svd_k(A, 5)
    gu, gv  = _rgu_gv(25, 25, 5)
    eps_t   = torch.tensor(1e-12 * S[0].item())

    dA_ex = _solve_fifth_term_svd(A, U, S, V, gu, gv, eps_t)
    dA_n4 = _solve_fifth_term_neumann(A, U, S, V, gu, gv, n_terms=4)
    c = cosine(dA_n4, dA_ex)
    assert c >= 0.99, f"Neumann L=4, decay={decay}: cos={c:.4f} (ρ={decay**2:.3f})"


def test_neumann_complex_input():
    """_solve_fifth_term_neumann handles complex matrices.
    Uses a fast-decaying (ρ≈0.04) spectrum so L=4 is sufficient.
    """
    m, n, k = 16, 16, 4
    # Build a complex matrix with geometric decay 0.2 → ρ = 0.04
    rng_m = torch.Generator(); rng_m.manual_seed(42)
    k_f   = min(m, n)
    Ur, _ = torch.linalg.qr(torch.randn(m, k_f, dtype=CDTYPE, generator=rng_m)
                             + 1j*torch.randn(m, k_f, dtype=CDTYPE, generator=rng_m))
    Vr, _ = torch.linalg.qr(torch.randn(n, k_f, dtype=CDTYPE, generator=rng_m)
                             + 1j*torch.randn(n, k_f, dtype=CDTYPE, generator=rng_m))
    sv = torch.tensor([0.2 ** i for i in range(k_f)], dtype=torch.float64).to(CDTYPE)
    A  = (Ur @ torch.diag(sv) @ Vr.mH).detach()
    U, S, V = _exact_svd_k(A, k)

    rng = torch.Generator(); rng.manual_seed(7)
    dk  = CDTYPE
    gu  = torch.randn(m, k, dtype=dk, generator=rng) + 1j*torch.randn(m, k, dtype=dk, generator=rng)
    gv  = torch.randn(n, k, dtype=dk, generator=rng) + 1j*torch.randn(n, k, dtype=dk, generator=rng)
    eps_t = torch.tensor(1e-12 * S[0].item())

    dA_ex = _solve_fifth_term_svd(A, U, S, V, gu, gv, eps_t)
    dA_n4 = _solve_fifth_term_neumann(A, U, S, V, gu, gv, n_terms=4)

    assert torch.isfinite(dA_n4).all(), "Neumann complex: non-finite"
    c = cosine(dA_n4.real, dA_ex.real)
    assert c >= 0.99, f"Neumann complex (real part): cos={c:.4f}"


def test_neumann_slow_spectrum_finite():
    """
    Slow spectrum (ρ=0.81, decay=0.9): L=4 may be inaccurate, but MUST stay
    finite and in the same half-space as the exact result.
    """
    A = _make_mat(20, 20, 4, spectrum_decay=0.9)
    U, S, V = _exact_svd_k(A, 4)
    gu, gv  = _rgu_gv(20, 20, 4)
    eps_t   = torch.tensor(1e-12 * S[0].item())

    dA_ex = _solve_fifth_term_svd(A, U, S, V, gu, gv, eps_t)
    dA_n4 = _solve_fifth_term_neumann(A, U, S, V, gu, gv, n_terms=4)

    assert torch.isfinite(dA_n4).all(), "slow-spectrum: non-finite"
    # Only require same direction (positive cosine)
    c = cosine(dA_n4, dA_ex)
    assert c >= 0.0, f"slow-spectrum gradient direction reversed: cos={c:.3f}"


# ══════════════════════════════════════════════════════════════════════════════
# 2. Augmented backward kernel (exact singular vectors, no rSVD noise)
# ══════════════════════════════════════════════════════════════════════════════

class _AugSVD(torch.autograd.Function):
    """
    Mini SVD Function: exact thin SVD in forward; saves k_aug triples.
    Zero-padding of upstream grads in backward tests the augmented logic.
    """
    @staticmethod
    def forward(ctx, A, k_ret, k_aug):
        U, S, Vh = torch.linalg.svd(A, full_matrices=False)
        V = Vh.mH
        k_aug = min(k_aug, U.shape[1])
        eps   = torch.tensor(1e-14, dtype=torch.float64, device=A.device)
        ctx.save_for_backward(U[:, :k_aug], S[:k_aug], V[:, :k_aug], eps)
        ctx.k_ret = k_ret
        return U[:, :k_ret], S[:k_ret], V[:, :k_ret]

    @staticmethod
    def backward(ctx, gu, gs, gv):
        U, S, V, eps = ctx.saved_tensors
        k_aug = S.size(0);  k_ret = ctx.k_ret
        m = U.size(0);      n = V.size(0)

        if k_ret < k_aug:
            extra = k_aug - k_ret
            def _p(g, c): return torch.cat([g, g.new_zeros(*g.shape[:-1], c)], dim=-1)
            gu = _p(gu, extra) if gu is not None else None
            gv = _p(gv, extra) if gv is not None else None
            gs = torch.cat([gs, gs.new_zeros(extra)]) if gs is not None else None

        from core_unrestricted import safe_inverse, safe_inverse_2
        eps_v = (S[0] * eps.item()).clamp(min=1e-30)
        Uh = U.conj().T; Vh_ = V.conj().T

        st = (U * gs.to(U.dtype).unsqueeze(-2) @ Vh_
              if gs is not None else torch.zeros(m, n, dtype=U.dtype, device=U.device))
        if gu is None and gv is None:
            return st, None, None

        F = safe_inverse(S.unsqueeze(-2) - S.unsqueeze(-1), eps_v)
        F.diagonal(0,-2,-1).fill_(0)
        G = safe_inverse(S.unsqueeze(-2) + S.unsqueeze(-1), eps_v)
        G.diagonal(0,-2,-1).fill_(0)

        guh = gu.conj().T
        ut  = U @ ((F + G) * (Uh @ gu - guh @ U)) * 0.5 @ Vh_
        gvh = gv.conj().T
        vt  = U @ ((F - G) * (Vh_ @ gv - gvh @ V)) * 0.5 @ Vh_
        return st + ut + vt, None, None


def _aug_grad(A, k_ret, k_aug, loss_fn):
    A2 = A.detach().clone().requires_grad_(True)
    U, S, V = _AugSVD.apply(A2, k_ret, k_aug)
    loss_fn(U, S, V, A).backward()
    return A2.grad.detach()


def _gauge_loss(C):
    """||USV†-C||² — depends on U and V jointly (gauge-invariant)."""
    C = C.detach()
    def fn(U, S, V, _A):
        rc = U @ torch.diag(S.to(U.dtype)) @ V.mH
        d  = rc - C.to(U.dtype)
        return (d * d.conj()).real.sum()
    return fn


@pytest.mark.parametrize("k_aug_mult,min_cos", [
    (1, None),     # aug=k=none → no extra correction, no min_cos constraint
    (2, 0.40),     # 5 extra modes out of 20 missing → modest improvement
    (4, 0.85),     # 15 extra modes out of 20 missing → substantial improvement
])
@pytest.mark.parametrize("decay", [0.3, 0.6])
def test_augmented_kernel_improves_with_extra(k_aug_mult, min_cos, decay):
    """
    Saving k*mult triples must match full-thin gradient better than saving k.

    Expected cosines (k=5, k_f=25):
      k_aug_mult=2 → captures 5 of 20 missing modes → cos ≈ 0.4–0.7
      k_aug_mult=4 → captures 15 of 20 missing modes → cos ≈ 0.85+
    """
    m, n, k = 25, 25, 5
    A = _make_mat(m, n, k, spectrum_decay=decay)
    rng = torch.Generator(); rng.manual_seed(1)
    C = torch.randn(*A.shape, dtype=A.dtype, device=A.device, generator=rng)
    fn = _gauge_loss(C)

    k_f   = min(m, n)
    k_aug = min(k * k_aug_mult, k_f)

    g_full = _aug_grad(A, k, k_f,   fn)
    g_none = _aug_grad(A, k, k,     fn)
    g_aug  = _aug_grad(A, k, k_aug, fn)

    assert torch.isfinite(g_aug).all(), f"k_aug_mult={k_aug_mult}: non-finite"

    c_none = cosine(g_none, g_full)
    c_aug  = cosine(g_aug,  g_full)

    if k_aug_mult > 1:
        assert c_aug >= c_none, (
            f"aug k*{k_aug_mult} should be ≥ none: "
            f"c_aug={c_aug:.4f}, c_none={c_none:.4f}, decay={decay}"
        )
    if min_cos is not None:
        assert c_aug >= min_cos, (
            f"aug k*{k_aug_mult} vs full_thin: cos={c_aug:.4f} < {min_cos}, decay={decay}"
        )


@pytest.mark.parametrize("extra,min_cos", [
    (0,  None),
    (2,  0.40),     # 2 extra modes out of 16 missing (k=4, k_f=20)
    (5,  0.60),     # 5 extra modes (~31% of missing)
    (14, 0.90),     # 14 extra modes → k_aug=18/20, only 2 missing
])
def test_augmented_kernel_parametric(extra, min_cos):
    m, n, k = 20, 20, 4
    A = _make_mat(m, n, k, spectrum_decay=0.3)
    rng = torch.Generator(); rng.manual_seed(3)
    C = torch.randn(*A.shape, dtype=A.dtype, device=A.device, generator=rng)
    fn = _gauge_loss(C)

    k_f   = min(m, n)
    k_aug = min(k + extra, k_f)

    g_full = _aug_grad(A, k, k_f,   fn)
    g_aug  = _aug_grad(A, k, k_aug, fn)

    assert torch.isfinite(g_aug).all()
    if min_cos is not None:
        c = cosine(g_aug, g_full)
        assert c >= min_cos, f"k+{extra}: cos={c:.4f} < {min_cos}"


# ══════════════════════════════════════════════════════════════════════════════
# 3. full_svd mode end-to-end vs numerical Jacobian (deterministic reference)
# ══════════════════════════════════════════════════════════════════════════════

def _apply_gauge_loss(A, k, C):
    rc_  = torch.tensor([1e-14], dtype=torch.float64, device=A.device)
    U, S, V = SVD_PROPACK.apply(A, k, 0, rc_, None, False)
    rc = U @ torch.diag(S.to(U.dtype)) @ V.mH
    d  = rc - C.to(U.dtype)
    return (d * d.conj()).real.sum()


def _num_jac(A, k, C, eps=1e-7):
    grad = torch.zeros_like(A)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            Ap = A.clone(); Ap[i, j] += eps
            Am = A.clone(); Am[i, j] -= eps
            grad[i, j] = (_apply_gauge_loss(Ap, k, C).item()
                         - _apply_gauge_loss(Am, k, C).item()) / (2 * eps)
    return grad


@pytest.mark.parametrize("m,n,k,decay", [
    (15, 15, 4, 0.4),
    (15, 15, 5, 0.3),
])
def test_full_svd_matches_numerical(m, n, k, decay):
    """full_svd mode: analytical Jacobian matches numerical to 1e-3 relative error."""
    set_rsvd_mode('full_svd')
    A = _make_mat(m, n, k, spectrum_decay=decay)
    rng = torch.Generator(); rng.manual_seed(5)
    C = torch.randn(m, n, dtype=DTYPE, generator=rng)

    g_num = _num_jac(A, k, C)
    A2 = A.detach().clone().requires_grad_(True)
    _apply_gauge_loss(A2, k, C).backward()
    g_an = A2.grad.detach()

    c = cosine(g_an, g_num)
    r = rel_err(g_an, g_num)
    assert c >= 1 - 1e-4, f"full_svd cos={c:.6f}, rel={r:.4e}"
    assert r <= 1e-3,      f"full_svd rel_err={r:.4e}"


# ══════════════════════════════════════════════════════════════════════════════
# 4. Smoke tests — all modes through SVD_PROPACK.apply
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("mode,nt,ae", [
    ('full_svd',   4,  0),
    ('none',       4,  0),
    ('neumann',    1,  0),
    ('neumann',    4,  0),
    ('neumann',    8,  0),
    ('neumann',   -1,  0),   # exact eigh
    ('augmented',  4,  0),
    ('augmented',  4,  3),
    ('augmented',  4,  6),
])
@pytest.mark.parametrize("m,n,k", [(20, 20, 5), (25, 20, 6)])
def test_smoke_finite(mode, nt, ae, m, n, k):
    """All modes → finite, non-NaN gradients."""
    set_rsvd_mode(mode, neumann_terms=nt, augment_extra=ae)
    A = _make_mat(m, n, k).requires_grad_(True)
    rng = torch.Generator(); rng.manual_seed(9)
    C   = torch.randn(m, n, dtype=DTYPE, generator=rng)
    rc_ = torch.tensor([1e-14], dtype=torch.float64)
    U, S, V = SVD_PROPACK.apply(A, k, 0, rc_, None, False)
    rc  = U @ torch.diag(S.to(U.dtype)) @ V.mH
    ((rc - C) ** 2).sum().backward()
    g = A.grad
    assert g is not None                  and \
           not torch.isnan(g).any()       and \
           torch.isfinite(g).all(),           \
           f"{mode}({nt},{ae}): non-finite grad"


@pytest.mark.parametrize("mode,nt,ae", [
    ('full_svd',  4, 0),
    ('neumann',   4, 0),
    ('augmented', 4, 4),
])
def test_smoke_complex(mode, nt, ae):
    """Complex float64 inputs: all modes remain finite."""
    set_rsvd_mode(mode, neumann_terms=nt, augment_extra=ae)
    m, n, k = 16, 16, 4
    A  = _make_complex_mat(m, n).requires_grad_(True)
    rng = torch.Generator(); rng.manual_seed(11)
    C  = (torch.randn(m, n, dtype=CDTYPE, generator=rng)
          + 1j*torch.randn(m, n, dtype=CDTYPE, generator=rng))
    rc_ = torch.tensor([1e-14], dtype=torch.float64)
    U, S, V = SVD_PROPACK.apply(A, k, 0, rc_, None, False)
    rc  = U @ torch.diag(S.to(U.dtype)) @ V.mH
    d   = rc - C
    (d * d.conj()).real.sum().backward()
    g = A.grad
    assert g is not None and torch.isfinite(g).all(), f"{mode}: complex non-finite"


def test_near_degenerate_all_modes():
    """
    Near-degenerate spectrum at the k/k+1 boundary: all modes must stay finite.
    """
    m, n, k = 25, 25, 6
    rng  = torch.Generator(); rng.manual_seed(99)
    k_f  = min(m, n)
    Ur, _ = torch.linalg.qr(torch.randn(m, k_f, dtype=DTYPE, generator=rng))
    Vr, _ = torch.linalg.qr(torch.randn(n, k_f, dtype=DTYPE, generator=rng))
    sv = torch.ones(k_f, dtype=DTYPE)
    sv[:k]    = torch.linspace(5.0, 1.01, k, dtype=DTYPE)
    sv[k]     = 1.00
    sv[k+1:]  = torch.linspace(0.9, 0.1, k_f - k - 1, dtype=DTYPE)
    A   = Ur @ torch.diag(sv) @ Vr.T
    C   = torch.randn(*A.shape, dtype=A.dtype, device=A.device, generator=rng)
    rc_ = torch.tensor([1e-14], dtype=torch.float64)

    for mode, nt, ae in [
        ('full_svd',   4,  0),
        ('neumann',    4,  0),
        ('neumann',   -1,  0),
        ('augmented',  4,  3),
        ('none',       4,  0),
    ]:
        set_rsvd_mode(mode, neumann_terms=nt, augment_extra=ae)
        A2  = A.detach().requires_grad_(True)
        U, S, V = SVD_PROPACK.apply(A2, k, 0, rc_, None, False)
        rc  = U @ torch.diag(S.to(U.dtype)) @ V.mH
        ((rc - C)**2).sum().backward()
        g = A2.grad
        assert g is not None and torch.isfinite(g).all(), \
            f"{mode}: near-degenerate non-finite"
