"""
solver.py
---------
Variational solver for  min_{X : L(X)=0} ||H - X||²

on the 2-torus T², where H is a d×d Hermitian matrix field with truncated
Fourier expansion  H_k  for k in K = {(i,j) : |i|≤K1, |j|≤K2}.

The linear operator L acts on matrix fields as
    L(X)_k = i * sum_{m in K} [H_m, X_{k-m}]  +  i*(k·ω) * X_k

Normal equation:  A phi = b,   A = -L²,   b = -L(H),   X* = H - L(phi*)

MATHEMATICAL STRUCTURE
======================
Physical inner product on matrix-valued fields:
    <X, Y> = sum_{k in K} tr(X_k† Y_k)

L is skew-Hermitian under this inner product:
    <X, L(Y)> = -<L(X), Y>
This follows from H_{-k} = H_k† and the trace cyclicity identity
    tr([H_m, X_k]† Y_p) = tr(X_k† [H_m†, Y_p]) = -tr(X_k† [H_{-m}, Y_p])

Therefore A = -L² = L†L is Hermitian PSD in the natural complex Euclidean
sense on the flat vector of all-K-mode coefficients.  Standard complex CG
(Hermitian CG) applies with inner product u†v = np.vdot(u, v):
    alpha_k = (r_k†r_k) / (p_k† A p_k)   (real scalar, no preconditioner)

Representation
--------------
Flat vector:  FULL grid K, NO scaling.
    v[fg_idx[k]*d² : (fg_idx[k]+1)*d²] = X_k.ravel()

The CG iterates stay in the Hermitian subspace {X : X_{-k} = X_k†} since:
  - x0 = 0 is Hermitian
  - b = -L(H) is a Hermitian field  (L maps Hermitian ↔ anti-Hermitian)
  - A = -L² maps the Hermitian subspace to itself

Therefore X* is Hermitian and the Pythagorean identity holds exactly:
    ||H||² = ||X*||² + ||H - X*||²

FFT convolution
---------------
[H,X]_k = sum_m H_m X_{k-m} - X_{k-m} H_m is a convolution bounded by 2K.
The DFT grid must satisfy N >= 4K+1 to avoid aliasing; N is rounded up to
the nearest power of two.  Definition of position-space H:
    H_pos[n] = H(θ_n) = sum_k H_k exp(i k·θ_n) = N1*N2 * IFFT2(H_grid)[n]
Commutator Fourier coefficient:
    [H,X]_k = FFT2(i[H_pos, X_pos])_k / N1*N2

Public API
----------
    solve(H, K1, K2, d, omega, ...)         → SolverResult   (production)
    solve_stage1(H, K1, K2, d, omega, ...)  → SolverResult   (direct L, validate)
    solve_stage2(H, K1, K2, d, omega, ...)  → SolverResult   (FFT L, validate)

SolverResult.X  : dict {k: d×d matrix} for every k in full_grid.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Callable, Optional

import numpy as np


# ==============================================================================
# Mode-set helpers
# ==============================================================================

def _build_modes(K1: int, K2: int):
    """
    Returns (full_grid, K_plus, K_plus0, fg_idx).

    full_grid : all (i,j) with -K1<=i<=K1, -K2<=j<=K2
    K_plus    : positive half  { k : k1>0,  or k1=0 and k2>0 }
    K_plus0   : [(0,0)] + K_plus
    fg_idx    : dict k → index in full_grid
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


# ==============================================================================
# Flat-vector packing  (full_grid, no scaling)
# ==============================================================================

def _build_h_flat(H: dict, full_grid: list, fg_idx: dict, d: int) -> np.ndarray:
    """Pack H Fourier coefficients into a flat complex vector on full_grid."""
    d2     = d * d
    h_flat = np.zeros(len(full_grid) * d2, dtype=complex)
    for k in full_grid:
        if k in H:
            h_flat[fg_idx[k] * d2 : (fg_idx[k] + 1) * d2] = H[k].ravel()
    return h_flat


# ==============================================================================
# L: direct convolution  O(|K|² d³)
# ==============================================================================

def _apply_L_direct(v_flat: np.ndarray,
                    H: dict, full_grid: list, fg_idx: dict,
                    d: int, omega: np.ndarray) -> np.ndarray:
    """
    Apply L(X) on the full_grid flat vector by explicit mode-sum convolution.

        L(X)_k = i * sum_m [H_m, X_{k-m}] + i*(k·omega)*X_k

    Modes absent from H are treated as zero.  O(|K|² d³) per call.
    """
    d2   = d * d
    X    = {k: v_flat[fg_idx[k] * d2 : (fg_idx[k] + 1) * d2].reshape(d, d)
            for k in full_grid}
    zero = np.zeros((d, d), dtype=complex)
    out  = np.empty(v_flat.size, dtype=complex)
    for k in full_grid:
        comm = zero.copy()
        for m in full_grid:
            Hm = H.get(m)
            if Hm is None:
                continue
            km = (k[0] - m[0], k[1] - m[1])
            if km in fg_idx:
                Xkm   = X[km]
                comm += Hm @ Xkm - Xkm @ Hm
        Lk = 1j * comm + 1j * float(np.dot(k, omega)) * X[k]
        out[fg_idx[k] * d2 : (fg_idx[k] + 1) * d2] = Lk.ravel()
    return out


# ==============================================================================
# L: FFT convolution  O(|K| log|K| d³)
# ==============================================================================

def _precompute_H_fft(H: dict, K1: int, K2: int, d: int):
    """
    Precompute H in position space for FFT-based L application.

    Pad to N >= 4K+1 (power of two) for alias-free convolution.
    Returns (H_pos, N1, N2) where H_pos[n1,n2,:,:] = H(θ_{n1,n2}).
    """
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
    """
    Apply L(X) via FFT convolution.  O(|K| log|K| d³) per call.

    Commutator identity used:
        [H, X]_k = FFT2(i[H_pos, X_pos])_k / N1*N2
    """
    d2     = d * d
    X_grid = np.zeros((N1, N2, d, d), dtype=complex)
    for k in full_grid:
        i, j = k
        X_grid[i % N1, j % N2] = (
            v_flat[fg_idx[k] * d2 : (fg_idx[k] + 1) * d2].reshape(d, d)
        )
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


# ==============================================================================
# Standard complex CG for Hermitian PSD operator
# ==============================================================================

def _cg(
    A: Callable,
    b: np.ndarray,
    tol: float = 1e-10,
    maxiter: int = 10_000,
    callback: Optional[Callable] = None,
) -> tuple:
    """
    Standard conjugate gradient for  A x = b  with A Hermitian PSD.

    Uses inner product u†v = np.vdot(u, v).  Since A is Hermitian PSD:
      - p†Ap is real and >= 0
      - r†r is real and >= 0
      - alpha and beta are real scalars (though stored as complex)

    Parameters
    ----------
    A        : callable, v -> A(v)
    b        : right-hand side
    tol      : convergence: ||r|| / ||b|| < tol
    maxiter  : maximum iterations
    callback : called as callback(k, rel_res) after each step

    Returns
    -------
    (x, info, n_iter)
      info = 0   converged
      info = -1  maxiter reached without convergence
    """
    b_norm = np.linalg.norm(b)
    if b_norm == 0.0:
        return np.zeros_like(b), 0, 0

    x  = np.zeros_like(b)
    r  = b.copy()           # r_0 = b  (since x_0 = 0)
    p  = r.copy()
    rr = np.vdot(r, r).real  # ||r||²  — always real for exact arithmetic

    for k in range(maxiter):
        Ap  = A(p)
        pAp = np.vdot(p, Ap).real          # p†Ap real >= 0
        if pAp < 1e-14 * rr:               # p in null(A): exact solution found
            break
        alpha  = rr / pAp                  # real
        x      = x + alpha * p
        r      = r - alpha * Ap
        rr_new = np.vdot(r, r).real

        if callback is not None:
            callback(k + 1, np.sqrt(rr_new) / b_norm)

        if np.sqrt(rr_new) / b_norm < tol:
            return x, 0, k + 1

        beta = rr_new / rr                 # real
        p    = r + beta * p
        rr   = rr_new

    return x, -1, maxiter


# ==============================================================================
# Result dataclass
# ==============================================================================

@dataclass
class SolverResult:
    X:                    Dict    # {k: d×d matrix} for every k in full_grid
    phi_flat:             np.ndarray
    residual:             float   # ||A phi - b|| / ||b||   (CG residual)
    n_iter:               int
    converged:            bool
    constraint_violation: float   # ||L(X*)|| / ||H||


# ==============================================================================
# Core solver (shared by all public entry points)
# ==============================================================================

def _solve_core(
    H: dict,
    K1: int, K2: int, d: int, omega: np.ndarray,
    apply_L: Callable,
    tol: float, maxiter: int, verbose: bool,
) -> SolverResult:
    full_grid, _, _, fg_idx = _build_modes(K1, K2)
    d2 = d * d
    N  = len(full_grid) * d2

    h_flat = _build_h_flat(H, full_grid, fg_idx, d)
    b_flat = -apply_L(h_flat)
    b_norm = float(np.linalg.norm(b_flat))

    if b_norm < 1e-30:                          # H already in ker(L)
        if verbose:
            print("||L(H)|| ≈ 0  →  H ∈ ker(L), returning X* = H directly.")
        X = {k: h_flat[fg_idx[k]*d2:(fg_idx[k]+1)*d2].reshape(d, d)
             for k in full_grid}
        return SolverResult(X, np.zeros(N, dtype=complex), 0.0, 0, True, 0.0)

    def matvec_A(v):
        return -apply_L(apply_L(v))

    def callback(k, rel_res):
        if verbose and k % 50 == 0:
            print(f"  iter {k:5d}  rel_res = {rel_res:.3e}")

    phi, info, n_iter = _cg(matvec_A, b_flat, tol=tol, maxiter=maxiter,
                            callback=callback)
    converged = (info == 0)
    residual  = float(np.linalg.norm(matvec_A(phi) - b_flat)) / b_norm

    if verbose:
        status = "CONVERGED" if converged else "NOT CONVERGED"
        print(f"CG {status}  {n_iter} iters  rel_res = {residual:.3e}")

    x_flat = h_flat - apply_L(phi)
    X      = {k: x_flat[fg_idx[k]*d2:(fg_idx[k]+1)*d2].reshape(d, d)
              for k in full_grid}
    h_norm = float(np.linalg.norm(h_flat))
    cv     = float(np.linalg.norm(apply_L(x_flat))) / max(h_norm, 1e-30)

    if verbose:
        print(f"Constraint violation  ||L(X*)|| / ||H|| = {cv:.3e}")

    return SolverResult(X, phi, residual, n_iter, converged, cv)


# ==============================================================================
# Public API
# ==============================================================================

def solve_stage1(
    H: dict, K1: int, K2: int, d: int, omega,
    tol: float = 1e-10, maxiter: int = 10_000, verbose: bool = True,
) -> SolverResult:
    """
    Stage 1: direct O(|K|² d³) convolution for L.

    Exact (no truncation artefacts).  Use for small problems and validation.
    """
    omega = np.asarray(omega, dtype=float)
    full_grid, _, _, fg_idx = _build_modes(K1, K2)
    return _solve_core(H, K1, K2, d, omega,
                       lambda v: _apply_L_direct(v, H, full_grid, fg_idx, d, omega),
                       tol, maxiter, verbose)


def solve_stage2(
    H: dict, K1: int, K2: int, d: int, omega,
    tol: float = 1e-10, maxiter: int = 10_000, verbose: bool = True,
) -> SolverResult:
    """
    Stage 2: FFT O(|K| log|K| d³) convolution for L.

    Numerically equivalent to stage1 (aliasing error ~1e-13).
    Use to validate FFT convolution before production runs.
    """
    omega = np.asarray(omega, dtype=float)
    H_pos, N1, N2 = _precompute_H_fft(H, K1, K2, d)
    full_grid, _, _, fg_idx = _build_modes(K1, K2)
    return _solve_core(H, K1, K2, d, omega,
                       lambda v: _apply_L_fft(v, H_pos, N1, N2,
                                              full_grid, fg_idx, d, omega),
                       tol, maxiter, verbose)


def solve(
    H: dict, K1: int, K2: int, d: int, omega,
    tol: float = 1e-10,
    maxiter: int = 10_000,
    fft_threshold: int = 25,
    force_direct: bool = False,
    verbose: bool = False,
) -> SolverResult:
    """
    Production solver: standard complex CG on A = -L² (Hermitian PSD).

    Automatically selects FFT convolution when |full_grid| > fft_threshold.
    Default threshold 25 means FFT for K1=K2>=3 (|full_grid|=49), direct for
    K1=K2<=2 (|full_grid|<=25).

    Parameters
    ----------
    H             : Hermitian field dict {k: d×d matrix}, all full_grid keys
    K1, K2        : truncation order (|k_i| <= Ki)
    d             : local Hilbert space dimension
    omega         : frequency vector, shape (2,)
    tol           : CG convergence: ||r|| / ||b|| < tol
    maxiter       : maximum CG iterations
    fft_threshold : use FFT when |full_grid| > this value
    force_direct  : override — always use direct convolution
    verbose       : print CG progress and final diagnostics
    """
    omega = np.asarray(omega, dtype=float)
    full_grid, _, _, fg_idx = _build_modes(K1, K2)
    use_fft = (len(full_grid) > fft_threshold) and not force_direct

    if use_fft:
        H_pos, N1, N2 = _precompute_H_fft(H, K1, K2, d)
        apply_L = lambda v: _apply_L_fft(v, H_pos, N1, N2,
                                         full_grid, fg_idx, d, omega)
    else:
        apply_L = lambda v: _apply_L_direct(v, H, full_grid, fg_idx, d, omega)

    return _solve_core(H, K1, K2, d, omega, apply_L, tol, maxiter, verbose)
