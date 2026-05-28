"""
solver.py
---------
Variational solver for  min ||H - X||^2  s.t.  L(X) = 0
where  L(X)_k = i * sum_m [H_m, X_{k-m}]  +  i*(k.omega) * X_k  (2-torus).

Normal equation:  A phi = b,   A = -L^2,   b = -L(H),   X = H - L(phi).

MATHEMATICAL STRUCTURE
----------------------
Physical inner product on matrix-valued fields on the torus:
    <X, Y> = sum_k  tr(X_k† Y_k)

L is skew-Hermitian under this inner product (L† = -L):
    <X, L(Y)> = -<L(X), Y>
This follows from H Hermitian (H_{-k} = H_k†) and trace cyclicity.

Therefore A = -L^2 = L†L is Hermitian PSD.  STANDARD complex CG applies:
    alpha_k = (r_k† z_k) / (p_k† A p_k)

Flat-vector convention: FULL grid (all k with -K<=k<=K), NO scaling.
v[fg_idx[k]*d² : (fg_idx[k]+1)*d²] = X_k.ravel().

A maps Hermitian fields to Hermitian fields (L maps Herm->antiHerm and vice
versa).  Starting from phi_0=0 and Hermitian b=-L(H), all CG iterates stay
Hermitian.

PRECONDITIONER (Stage 3)
Per-mode block:  M_k = (k.omega)² I_{d²} + (-ad_{H0}²)
where -ad_{H0}² = ad_{H0}† ad_{H0} is PSD and captures intra-mode coupling
from the zero-mode commutator.  Block-inverting M_k reduces iterations when
the spectrum of A is heterogeneous across modes.

BUGS FIXED (relative to original half-space pseudocode):
  Bug 1: sqrt(2) scaling made A non-Hermitian in K_plus0 representation.
         Fixed: work on full_grid, no scaling.
  Bug 2: H_pos = IFFT(H_grid) missing * N1*N2 normalization.
         Fixed: H_pos = ifft2(H_grid) * N1 * N2.
  Bug 3: pad size 2*(2K+1) insufficient; must be power-of-two >= 4K+1.
         Fixed: N1 = next_pow2(4*K1+1).

THREE STAGES
  Stage 1: direct O(|K|²d³) convolution, no precond.  Validates L.
  Stage 2: FFT O(|K|log|K|d³) convolution, no precond.  Validates FFT L.
  Stage 3 (=solve): FFT + block preconditioner.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Callable, Optional, List

import numpy as np


# ============================================================
# Mode-set helpers
# ============================================================

def _build_modes(K1: int, K2: int):
    """
    Returns:
        full_grid : list of all (i,j) with -K1<=i<=K1, -K2<=j<=K2
        K_plus    : positive modes (i>0 or i==0,j>0)
        K_plus0   : [(0,0)] + K_plus
        fg_idx    : dict k -> index in full_grid
    """
    full_grid = [(i, j) for i in range(-K1, K1 + 1) for j in range(-K2, K2 + 1)]
    K_plus    = [(i, j) for (i, j) in full_grid if i > 0 or (i == 0 and j > 0)]
    K_plus0   = [(0, 0)] + K_plus
    fg_idx    = {k: idx for idx, k in enumerate(full_grid)}
    return full_grid, K_plus, K_plus0, fg_idx


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


# ============================================================
# Flat-vector helpers (full_grid, no scaling)
# ============================================================

def _build_h_flat(H: dict, full_grid: list, fg_idx: dict, d: int) -> np.ndarray:
    """Pack H into full_grid flat vector with no scaling."""
    d2     = d * d
    h_flat = np.zeros(len(full_grid) * d2, dtype=complex)
    for k in full_grid:
        if k in H:
            h_flat[fg_idx[k] * d2 : (fg_idx[k] + 1) * d2] = H[k].ravel()
    return h_flat


# ============================================================
# Stage 1: direct convolution  O(|K|^2 d^3)
# ============================================================

def _apply_L_direct(v_flat: np.ndarray,
                    H: dict, full_grid: list, fg_idx: dict,
                    d: int, omega: np.ndarray) -> np.ndarray:
    """L(X)_k = i*sum_m [H_m, X_{k-m}] + i*(k.omega)*X_k  on full_grid."""
    d2 = d * d
    X  = {k: v_flat[fg_idx[k] * d2 : (fg_idx[k] + 1) * d2].reshape(d, d)
           for k in full_grid}
    out = np.empty(v_flat.size, dtype=complex)
    for k in full_grid:
        comm = np.zeros((d, d), dtype=complex)
        for m in full_grid:
            km = (k[0] - m[0], k[1] - m[1])
            if km in fg_idx:
                Xkm   = X[km]
                comm += H[m] @ Xkm - Xkm @ H[m]
        Lk = 1j * comm + 1j * float(np.dot(k, omega)) * X[k]
        out[fg_idx[k] * d2 : (fg_idx[k] + 1) * d2] = Lk.ravel()
    return out


# ============================================================
# Stage 2: FFT convolution  O(|K| log|K| d^3)
# ============================================================

def _precompute_H_fft(H: dict, K1: int, K2: int, d: int):
    """Prepare H in position space for FFT convolution."""
    N1 = _next_pow2(4 * K1 + 1)
    N2 = _next_pow2(4 * K2 + 1)
    H_grid = np.zeros((N1, N2, d, d), dtype=complex)
    for (i, j), mat in H.items():
        H_grid[i % N1, j % N2] += mat
    H_pos = np.fft.ifft2(H_grid, axes=(0, 1)) * (N1 * N2)
    return H_pos, N1, N2


def _apply_L_fft(v_flat: np.ndarray,
                 H_pos: np.ndarray, N1: int, N2: int,
                 full_grid: list, fg_idx: dict,
                 d: int, omega: np.ndarray) -> np.ndarray:
    """Apply L via FFT convolution on full_grid flat vector."""
    d2 = d * d
    X_grid = np.zeros((N1, N2, d, d), dtype=complex)
    for k in full_grid:
        i, j = k
        X_grid[i % N1, j % N2] = v_flat[fg_idx[k] * d2 : (fg_idx[k] + 1) * d2].reshape(d, d)

    X_pos = np.fft.ifft2(X_grid, axes=(0, 1)) * (N1 * N2)
    C_pos = 1j * (
        np.einsum('...ik,...kj->...ij', H_pos, X_pos)
        - np.einsum('...ik,...kj->...ij', X_pos, H_pos)
    )
    C_fft = np.fft.fft2(C_pos, axes=(0, 1)) / (N1 * N2)

    out = np.empty(v_flat.size, dtype=complex)
    for k in full_grid:
        i, j = k
        X_k = v_flat[fg_idx[k] * d2 : (fg_idx[k] + 1) * d2].reshape(d, d)
        Lk  = C_fft[i % N1, j % N2] + 1j * float(np.dot(k, omega)) * X_k
        out[fg_idx[k] * d2 : (fg_idx[k] + 1) * d2] = Lk.ravel()
    return out


# ============================================================
# Standard complex CG
# A = -L^2 is Hermitian PSD => standard CG with u†v inner product is correct.
# ============================================================

def _cg(
    A: Callable,
    b: np.ndarray,
    x0: Optional[np.ndarray] = None,
    M: Optional[Callable] = None,
    tol: float = 1e-10,
    maxiter: int = 10_000,
    callback: Optional[Callable] = None,
) -> tuple:
    """
    Standard CG for Hermitian PSD A.
    Returns (x, info, n_iter).  info=0: converged; info=-1: maxiter reached.
    """
    x  = np.zeros_like(b) if x0 is None else x0.copy()
    r  = b - A(x)
    z  = r if M is None else M(r)
    p  = z.copy()
    rz = np.vdot(r, z)
    b_norm = np.linalg.norm(b)
    if b_norm == 0:
        return x, 0, 0

    for k in range(maxiter):
        Ap    = A(p)
        pAp   = np.vdot(p, Ap)
        alpha = rz / pAp
        x     = x + alpha * p
        r     = r - alpha * Ap
        if callback is not None:
            callback(x, k + 1, np.linalg.norm(r) / b_norm)
        if np.linalg.norm(r) / b_norm < tol:
            return x, 0, k + 1
        z_new  = r if M is None else M(r)
        rz_new = np.vdot(r, z_new)
        beta   = rz_new / rz
        p      = z_new + beta * p
        rz     = rz_new

    return x, -1, maxiter


# ============================================================
# Block preconditioner  M_k = (k.omega)^2 I + (-ad_{H0}^2)
# ============================================================

def _build_precond_blocks(
    H: dict, full_grid: list, d: int, omega: np.ndarray,
    reg: float = 1e-8,
) -> List[np.ndarray]:
    """
    Per-mode block preconditioner inverses.

    M_k = (k.omega)^2 * I_{d^2} + (-ad_{H0}^2)
    -ad_{H0}^2 = ad_{H0}† ad_{H0} is PSD, captures intra-mode coupling.

    Returns list of d^2 x d^2 inverse matrices (one per mode in full_grid).
    """
    d2 = d * d
    H0 = H.get((0, 0), np.zeros((d, d), dtype=complex))

    neg_ad2 = np.zeros((d2, d2), dtype=complex)
    for j in range(d2):
        ej       = np.zeros(d2, dtype=complex); ej[j] = 1.0
        Xj       = ej.reshape(d, d)
        ad1      = H0 @ Xj - Xj @ H0
        ad2_col  = H0 @ ad1 - ad1 @ H0
        neg_ad2[:, j] = -ad2_col.ravel()

    neg_ad2 = (neg_ad2 + neg_ad2.conj().T) / 2   # Hermitize

    blocks_inv = []
    for k in full_grid:
        kw2 = float(np.dot(k, omega)) ** 2
        Mk  = (kw2 + reg) * np.eye(d2, dtype=complex) + neg_ad2
        Mk  = (Mk + Mk.conj().T) / 2
        blocks_inv.append(np.linalg.inv(Mk))
    return blocks_inv


def _apply_precond(v: np.ndarray, blocks_inv: List[np.ndarray], d2: int) -> np.ndarray:
    out = np.empty_like(v)
    for i, Binv in enumerate(blocks_inv):
        out[i * d2 : (i + 1) * d2] = Binv @ v[i * d2 : (i + 1) * d2]
    return out


# ============================================================
# Result dataclass
# ============================================================

@dataclass
class SolverResult:
    X:                    Dict    # {k: d×d matrix} for all k in full_grid
    phi_flat:             np.ndarray
    residual:             float   # ||A phi - b|| / ||b||
    n_iter:               int
    converged:            bool
    constraint_violation: float   # ||L(X)|| / ||H||


# ============================================================
# Core solver
# ============================================================

def _solve_core(
    H: dict,
    K1: int, K2: int, d: int, omega: np.ndarray,
    apply_L_fn: Callable,
    use_precond: bool,
    tol: float, maxiter: int, verbose: bool,
) -> SolverResult:
    full_grid, K_plus, K_plus0, fg_idx = _build_modes(K1, K2)
    d2 = d * d
    N  = len(full_grid) * d2

    h_flat = _build_h_flat(H, full_grid, fg_idx, d)
    b_flat = -apply_L_fn(h_flat)
    b_norm = float(np.linalg.norm(b_flat))

    if b_norm < 1e-30:
        if verbose:
            print("b ~ 0: L(H) ~ 0 already -> X = H.")
        phi = np.zeros(N, dtype=complex)
        X   = {k: h_flat[fg_idx[k]*d2:(fg_idx[k]+1)*d2].reshape(d,d) for k in full_grid}
        return SolverResult(X, phi, 0.0, 0, True, 0.0)

    def matvec_A(v):
        return -apply_L_fn(apply_L_fn(v))

    M_fn = None
    if use_precond:
        blocks_inv = _build_precond_blocks(H, full_grid, d, omega)
        M_fn = lambda v: _apply_precond(v, blocks_inv, d2)

    report = [0]
    def callback(xk, k, rel_res):
        report[0] = k
        if verbose and k % 50 == 0:
            print(f"  iter {k:5d}  rel_res={rel_res:.3e}")

    phi, info, n_iter = _cg(
        matvec_A, b_flat, M=M_fn,
        tol=tol, maxiter=maxiter, callback=callback
    )
    n_iter    = max(report[0], n_iter)
    converged = (info == 0)
    residual  = float(np.linalg.norm(matvec_A(phi) - b_flat)) / b_norm

    if verbose:
        status = "CONVERGED" if converged else "NOT CONVERGED"
        print(f"CG {status} in {n_iter} iters, rel_res={residual:.3e}")

    x_flat = h_flat - apply_L_fn(phi)
    X      = {k: x_flat[fg_idx[k]*d2:(fg_idx[k]+1)*d2].reshape(d,d)
               for k in full_grid}

    h_norm = float(np.linalg.norm(h_flat))
    cv     = float(np.linalg.norm(apply_L_fn(x_flat))) / max(h_norm, 1e-30)

    if verbose:
        print(f"Constraint violation ||L(X)|| / ||H|| = {cv:.3e}")

    return SolverResult(X, phi, residual, n_iter, converged, cv)


# ============================================================
# Public API
# ============================================================

def solve_stage1(
    H: dict,
    K1: int, K2: int, d: int, omega,
    tol: float = 1e-10,
    maxiter: int = 10_000,
    verbose: bool = True,
) -> SolverResult:
    """Stage 1: direct O(|K|^2 d^3), no preconditioner."""
    omega = np.asarray(omega, dtype=float)
    full_grid, _, _, fg_idx = _build_modes(K1, K2)

    def apply_L(v):
        return _apply_L_direct(v, H, full_grid, fg_idx, d, omega)

    return _solve_core(H, K1, K2, d, omega,
                       apply_L, use_precond=False,
                       tol=tol, maxiter=maxiter, verbose=verbose)


def solve_stage2(
    H: dict,
    K1: int, K2: int, d: int, omega,
    tol: float = 1e-10,
    maxiter: int = 10_000,
    verbose: bool = True,
) -> SolverResult:
    """Stage 2: FFT O(|K| log|K| d^3), no preconditioner."""
    omega = np.asarray(omega, dtype=float)
    H_pos, N1, N2 = _precompute_H_fft(H, K1, K2, d)
    full_grid, _, _, fg_idx = _build_modes(K1, K2)

    def apply_L(v):
        return _apply_L_fft(v, H_pos, N1, N2, full_grid, fg_idx, d, omega)

    return _solve_core(H, K1, K2, d, omega,
                       apply_L, use_precond=False,
                       tol=tol, maxiter=maxiter, verbose=verbose)


def solve(
    H: dict,
    K1: int, K2: int, d: int, omega,
    tol: float = 1e-10,
    maxiter: int = 10_000,
    use_precond: bool = False,
    fft_threshold: int = 25,
    force_direct: bool = False,
    verbose: bool = False,
) -> SolverResult:
    """
    Production solver (Stage 3): FFT convolution, standard complex CG.

    FFT used when |full_grid| > fft_threshold (unless force_direct=True).
    use_precond: per-mode block M_k = (k.omega)^2*I + (-ad_{H0}^2).
    Only helps when |k.omega| >> ||[H, .]||.  Default False (no precond).
    """
    omega = np.asarray(omega, dtype=float)
    full_grid, _, _, fg_idx = _build_modes(K1, K2)
    use_fft = (len(full_grid) > fft_threshold) and not force_direct

    if use_fft:
        H_pos, N1, N2 = _precompute_H_fft(H, K1, K2, d)
        def apply_L(v):
            return _apply_L_fft(v, H_pos, N1, N2, full_grid, fg_idx, d, omega)
    else:
        def apply_L(v):
            return _apply_L_direct(v, H, full_grid, fg_idx, d, omega)

    return _solve_core(H, K1, K2, d, omega,
                       apply_L, use_precond=use_precond,
                       tol=tol, maxiter=maxiter, verbose=verbose)
