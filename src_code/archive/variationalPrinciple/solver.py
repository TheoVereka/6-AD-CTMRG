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
from datetime import datetime
from pathlib import Path
from scipy.sparse.linalg import LinearOperator, minres, lsmr
from typing import Dict, Callable, Optional
import sys
import numpy as np
import mpmath as mp

DEFAULT_RESULT_DIR = Path(r"D:\HyraiOn\ENS_Lyon\Internship\2025-MPIPKS\data\raw\AE_variational")


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


def save_solver_result(
    result: SolverResult,
    K1: int,
    K2: int,
    d: int,
    omega: np.ndarray,
    save_dir: Optional[str | Path] = None,
    timestamp: Optional[str] = None,
) -> Path:
    """
    Save SolverResult to a compressed .npz file.

    File name format: K1_<K1>_K2_<K2>_<timestamp>.npz
    """
    out_dir = Path(save_dir) if save_dir is not None else DEFAULT_RESULT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    stamp = timestamp if timestamp is not None else datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = out_dir / f"K1_{K1}_K2_{K2}_{stamp}.npz"

    full_grid, _, _, _ = _build_modes(K1, K2)
    full_grid_arr = np.asarray(full_grid, dtype=np.int32)
    X_stack = np.stack([np.asarray(result.X[k], dtype=complex) for k in full_grid], axis=0)

    np.savez_compressed(
        file_path,
        K1=np.int32(K1),
        K2=np.int32(K2),
        d=np.int32(d),
        omega=np.asarray(omega, dtype=float),
        full_grid=full_grid_arr,
        X_stack=X_stack,
        phi_flat=np.asarray(result.phi_flat, dtype=complex),
        residual=np.float64(result.residual),
        n_iter=np.int64(result.n_iter),
        converged=np.bool_(result.converged),
        constraint_violation=np.float64(result.constraint_violation),
    )
    return file_path


def load_solver_result(file_path: str | Path) -> tuple[SolverResult, dict]:
    """
    Load SolverResult from a file created by save_solver_result.

    Returns
    -------
    (result, metadata)
      result   : SolverResult instance
      metadata : dict with K1, K2, d, omega, full_grid, file_path
    """
    path = Path(file_path)
    with np.load(path, allow_pickle=False) as data:
        K1 = int(data["K1"])
        K2 = int(data["K2"])
        d = int(data["d"])
        omega = np.asarray(data["omega"], dtype=float)
        full_grid_arr = np.asarray(data["full_grid"], dtype=np.int64)
        full_grid = [tuple(map(int, row)) for row in full_grid_arr]

        X_stack = np.asarray(data["X_stack"], dtype=complex)
        X = {k: X_stack[idx] for idx, k in enumerate(full_grid)}

        result = SolverResult(
            X=X,
            phi_flat=np.asarray(data["phi_flat"], dtype=complex),
            residual=float(data["residual"]),
            n_iter=int(data["n_iter"]),
            converged=bool(data["converged"]),
            constraint_violation=float(data["constraint_violation"]),
        )

    metadata = {
        "K1": K1,
        "K2": K2,
        "d": d,
        "omega": omega,
        "full_grid": full_grid,
        "file_path": str(path),
    }
    return result, metadata


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

    progress_marks = (0, 1, 2, 5, 10, 20, 30, 50, 70, 90)
    next_mark_idx = 0
    report_every = max(1, min(500, maxiter // 20))

    def callback(k, rel_res):
        nonlocal next_mark_idx
        if not verbose:
            return

        pct = int((100 * k) / maxiter)
        while next_mark_idx < len(progress_marks) and pct >= progress_marks[next_mark_idx]:
            print(f"  CG progress {progress_marks[next_mark_idx]:2d}%  (iter={k:6d}, rel_res={rel_res:.3e})")
            next_mark_idx += 1

        if k % report_every == 0:
            print(f"  CG iter {k:6d}/{maxiter}  rel_res={rel_res:.3e}")

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



def solve(
    H: dict, K1: int, K2: int, d: int, omega,
    tol: float = 1e-10,
    maxiter: int = 100_000,
    fft_threshold: int = 0,
    force_direct: bool = False,
    verbose: bool = True,
    save_result: bool = False,
    save_dir: Optional[str | Path] = None,
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
    save_result   : if True, save SolverResult to disk
    save_dir      : output folder (default: DEFAULT_RESULT_DIR)
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

    result = _solve_core(H, K1, K2, d, omega, apply_L, tol, maxiter, verbose)

    if save_result:
        out_path = save_solver_result(result, K1, K2, d, omega, save_dir=save_dir)
        if verbose:
            print(f"Saved SolverResult: {out_path}")

    return result












def _apply_L_matvec(v_flat, H, full_grid, fg_idx, d, omega,):
    """Return the matrix representation of L as a dense array (debug only)."""
    N = len(v_flat)
    L_mat = np.zeros((N, N), dtype=complex)
    e = np.zeros(N, dtype=complex)
    for i in range(N):
        e[i] = 1.0
        L_mat[:, i] = _apply_L_direct(e, H, full_grid, fg_idx, d, omega)
        e[i] = 0.0
    return L_mat

def solve_direct(H, K1, K2, d, omega,
    save_result: bool = False,
    save_dir: Optional[str | Path] = None,):
    """
    Solve  -L² φ = -L(H)  via dense matrix (pseudoinverse).
    Returns the same SolverResult as solve().
    """
    full_grid, _, _, fg_idx = _build_modes(K1, K2)
    h_flat = _build_h_flat(H, full_grid, fg_idx, d)
    b_flat = -_apply_L_direct(h_flat, H, full_grid, fg_idx, d, omega)
    L_mat = _apply_L_matvec(h_flat, H, full_grid, fg_idx, d, omega)   # reuse h_flat as dummy vector
    A_mat = -L_mat @ L_mat
    # A is Hermitian PSD; use lstsq to obtain minimal-norm solution
    phi, res, rank, s = np.linalg.lstsq(A_mat, b_flat, rcond=None)
    x_flat = h_flat - _apply_L_direct(phi, H, full_grid, fg_idx, d, omega)
    X = {k: x_flat[fg_idx[k]*d*d:(fg_idx[k]+1)*d*d].reshape(d, d) for k in full_grid}
    h_norm = np.linalg.norm(h_flat)
    cv = np.linalg.norm(_apply_L_direct(x_flat, H, full_grid, fg_idx, d, omega)) / max(h_norm, 1e-30)
    result = SolverResult(X, phi, np.linalg.norm(A_mat @ phi - b_flat)/np.linalg.norm(b_flat),
                          rank, True, cv)
    if save_result:
        out_path = save_solver_result(result, K1, K2, d, omega, save_dir=save_dir)
        print(f"Saved SolverResult: {out_path}")
    return result



def solve_saddle(
    H: dict,
    K1: int, K2: int,
    d: int,
    omega,
    tol: float = 1e-10,
    maxiter: int = 100_000,          # ignored, kept for signature
    fft_threshold: int = 0,          # ignored
    force_direct: bool = False,      # ignored
    verbose: bool = True,
    save_result: bool = False,
    save_dir: Optional[str | Path] = None,
    reg: float = 1e-12,              # tiny regularisation for min‑norm ϕ
) -> SolverResult:
    """
    Direct solver for the saddle‑point system (no squaring of L).

    Solves
        [  I     L    ] [ X ]   [ H ]
        [  L   -reg·I ] [ ϕ ] = [ 0 ]

    where L is the skew‑adjoint operator applied to the full‑grid flat vector.
    reg is a tiny positive number that selects the minimum‑norm ϕ.
    The solution is then X* = H − L(ϕ*), which automatically satisfies L(X*) ≈ 0.

    Parameters
    ----------
    H, K1, K2, d, omega : same as in solve()
    reg : regularisation for the (2,2) block (default 1e-12)
    Other arguments are kept for signature compatibility only.

    Returns
    -------
    SolverResult  (same format as solve())
    """
    omega = np.asarray(omega, dtype=float)
    full_grid, _, _, fg_idx = _build_modes(K1, K2)
    d2 = d * d
    N = len(full_grid) * d2

    # Pack H into flat vector
    h_flat = _build_h_flat(H, full_grid, fg_idx, d)

    # Choose apply_L: use FFT if grid is large, else direct (consistent with solve)
    use_fft = (len(full_grid) > fft_threshold) and not force_direct
    if use_fft:
        H_pos, N1_fft, N2_fft = _precompute_H_fft(H, K1, K2, d)
        def apply_L(v):
            return _apply_L_fft(v, H_pos, N1_fft, N2_fft, full_grid, fg_idx, d, omega)
    else:
        def apply_L(v):
            return _apply_L_direct(v, H, full_grid, fg_idx, d, omega)

    # Build dense matrix of L (N×N complex)
    if verbose:
        print(f"Building dense L matrix (N={N}) …")
    L_mat = np.zeros((N, N), dtype=complex)
    e = np.zeros(N, dtype=complex)
    for j in range(N):
        e[j] = 1.0
        L_mat[:, j] = apply_L(e)
        e[j] = 0.0

    # Assemble saddle‑point matrix
    # M = [[ I,    L     ],
    #      [ L,  -reg*I ]]
    I_N = np.eye(N, dtype=complex)
    O_N = np.zeros((N, N), dtype=complex)

    M_top = np.hstack([I_N, L_mat])
    M_bot = np.hstack([L_mat, -reg * I_N])
    M = np.vstack([M_top, M_bot])

    rhs = np.zeros(2 * N, dtype=complex)
    rhs[:N] = h_flat

    if verbose:
        print(f"Solving dense saddle‑point system (size {2*N}) …")
    sol = np.linalg.solve(M, rhs)

    # Extract ϕ and compute X* = H − L(ϕ)
    phi_flat = sol[N:]   # second half
    X_flat = h_flat - apply_L(phi_flat)

    # Construct X dictionary
    X = {k: X_flat[fg_idx[k]*d2:(fg_idx[k]+1)*d2].reshape(d, d) for k in full_grid}

    # Diagnostics
    h_norm = float(np.linalg.norm(h_flat))
    L_H_flat = apply_L(h_flat)
    b_norm = float(np.linalg.norm(-L_H_flat))
    if b_norm < 1e-30:
        residual = 0.0
    else:
        residual = float(np.linalg.norm(-apply_L(apply_L(phi_flat)) - (-L_H_flat))) / b_norm
    cv = float(np.linalg.norm(apply_L(X_flat))) / max(h_norm, 1e-30)
    converged = cv < tol   # consider converged if constraint violation is small

    if verbose:
        print(f"Direct saddle solve done.  ||L(X*)|| / ||H|| = {cv:.3e}")
        print(f"Normal eq. residual = {residual:.3e}")

    result = SolverResult(X, phi_flat, residual, 0, converged, cv)

    if save_result:
        out_path = save_solver_result(result, K1, K2, d, omega, save_dir=save_dir)
        if verbose:
            print(f"Saved SolverResult: {out_path}")

    return result



from scipy.sparse.linalg import LinearOperator, minres
import numpy as np

def solve_minres(
    H: Dict,
    K1: int, K2: int,
    d: int,
    omega,
    tol: float = 1e-10,
    maxiter: int = 100_000,
    fft_threshold: int = 0,
    force_direct: bool = False,
    verbose: bool = True,
    save_result: bool = False,
    save_dir: Optional[str | Path] = None,
) -> SolverResult:
    """
    MINRES on the real‑symmetric equivalent of the Hermitian saddle‑point system

        [ I   L ] [ X ]   [ H ]
        [ L^T 0 ] [ ϕ ] = [ 0 ]    (L real‑skew‑symmetric after doubling)

    The complex equation is converted to a real system by splitting
    real and imaginary parts. This avoids any complex‑matrix symmetry
    issues in SciPy's minres and scales to large K without memory overhead.

    Parameters identical to solve_saddle.
    """
    omega = np.asarray(omega, dtype=float)
    full_grid, _, _, fg_idx = _build_modes(K1, K2)
    d2 = d * d
    N = len(full_grid) * d2

    # -------- pack H and define L (complex) --------
    h_flat = _build_h_flat(H, full_grid, fg_idx, d)

    use_fft = (len(full_grid) > fft_threshold) and not force_direct
    if use_fft:
        H_pos, N1_fft, N2_fft = _precompute_H_fft(H, K1, K2, d)
        def apply_L(v):
            return _apply_L_fft(v, H_pos, N1_fft, N2_fft, full_grid, fg_idx, d, omega)
    else:
        def apply_L(v):
            return _apply_L_direct(v, H, full_grid, fg_idx, d, omega)

    # -------- Real symmetric saddle‑point operator (dimension 4N) --------
    def matvec_M_real(w_real):
        """
        w_real = [real(v1), imag(v1), real(v2), imag(v2)]  (each length N)
        returns the same layout for M * [v1; v2]
        """
        re_v1 = w_real[0:N]
        im_v1 = w_real[N:2*N]
        re_v2 = w_real[2*N:3*N]
        im_v2 = w_real[3*N:4*N]

        v1 = re_v1 + 1j * im_v1
        v2 = re_v2 + 1j * im_v2

        # M * [v1; v2] = [ v1 + L(v2); -L(v1) ]
        w1 = v1 + apply_L(v2)
        w2 = -apply_L(v1)

        return np.concatenate([w1.real, w1.imag, w2.real, w2.imag])

    total_real = 4 * N
    M_op = LinearOperator(
        shape=(total_real, total_real),
        matvec=matvec_M_real,
        rmatvec=matvec_M_real,   # real symmetric
        dtype=float,
    )

    # -------- right‑hand side --------
    rhs_real = np.zeros(total_real, dtype=float)
    rhs_real[0:N] = h_flat.real
    rhs_real[N:2*N] = h_flat.imag

    if verbose:
        print(f"Starting real MINRES on {total_real} × {total_real} system "
              f"({N} complex unknowns in ϕ), tol={tol:.0e}, maxiter={maxiter}")
        #sys.stdout.flush()

    # -------- solve --------
    x0 = np.zeros(total_real, dtype=float)          # minimum‑norm ϕ
    sol, info = minres(M_op, rhs_real, x0=x0, rtol=tol, maxiter=maxiter, show=False)

    if verbose:
        if info == 0:
            print("MINRES converged (residual below tolerance).")
        elif info == 1:
            print("MINRES reached maximum number of iterations.")
        else:
            print(f"MINRES stopped with info={info}.")
    #sys.stdout.flush()

    # -------- extract complex ϕ and compute X --------
    phi_real = sol[2*N:3*N]
    phi_imag = sol[3*N:4*N]
    phi_flat = phi_real + 1j * phi_imag

    X_flat = h_flat - apply_L(phi_flat)

    # -------- diagnostics --------
    h_norm = float(np.linalg.norm(h_flat))
    L_H_flat = apply_L(h_flat)
    b_norm = float(np.linalg.norm(-L_H_flat))
    if b_norm < 1e-30:
        normal_res = 0.0
    else:
        normal_res = float(np.linalg.norm(-apply_L(apply_L(phi_flat)) + L_H_flat)) / b_norm
    cv = float(np.linalg.norm(apply_L(X_flat))) / max(h_norm, 1e-30)
    converged = (info == 0)

    X = {k: X_flat[fg_idx[k]*d2:(fg_idx[k]+1)*d2].reshape(d, d) for k in full_grid}
    result = SolverResult(X, phi_flat, normal_res, 0, converged, cv)

    if verbose:
        print(f"Constraint violation  ||L(X*)|| / ||H|| = {cv:.3e}")
        print(f"Normal‑eq residual      = {normal_res:.3e}")

    if save_result:
        out_path = save_solver_result(result, K1, K2, d, omega, save_dir=save_dir)
        if verbose:
            print(f"Saved SolverResult: {out_path}")
    #sys.stdout.flush()

    return result


def solve_minres_reg(
    H: Dict,
    K1: int, K2: int,
    d: int,
    omega,
    tol: float = 1e-10,
    maxiter: int = 100_000_000,
    fft_threshold: int = 0,
    force_direct: bool = False,
    verbose: bool = True,
    save_result: bool = False,
    save_dir: Optional[str | Path] = None,
    reg: float = 1e-12,          # <-- same shift as in the dense solver
) -> SolverResult:
    """
    MINRES on the regularised Hermitian saddle‑point system

        [ I     L   ] [ X ]   [ H ]
        [ -L  -reg·I ] [ ϕ ] = [ 0 ]

    (L is skew‑adjoint, so L^† = -L).  The shift -reg·I removes the
    null‑space of the singular matrix, guarantees a unique minimum‑norm
    solution and lets MINRES converge in few iterations.

    All arguments match solve / solve_saddle.  The parameter `reg`
    defaults to 1e-12, exactly as in the proven dense solver.
    """
    omega = np.asarray(omega, dtype=float)
    full_grid, _, _, fg_idx = _build_modes(K1, K2)
    d2 = d * d
    N = len(full_grid) * d2

    # -------- pack H and define L (complex) --------
    h_flat = _build_h_flat(H, full_grid, fg_idx, d)

    use_fft = (len(full_grid) > fft_threshold) and not force_direct
    if use_fft:
        H_pos, N1_fft, N2_fft = _precompute_H_fft(H, K1, K2, d)
        def apply_L(v):
            return _apply_L_fft(v, H_pos, N1_fft, N2_fft, full_grid, fg_idx, d, omega)
    else:
        def apply_L(v):
            return _apply_L_direct(v, H, full_grid, fg_idx, d, omega)

    # -------- Regularised Hermitian operator (real symmetric lift) --------
    def matvec_M_real(w_real):
        # Layout: [re(X), im(X), re(ϕ), im(ϕ)]
        re_v1 = w_real[0:N]
        im_v1 = w_real[N:2*N]
        re_v2 = w_real[2*N:3*N]
        im_v2 = w_real[3*N:4*N]

        v1 = re_v1 + 1j * im_v1
        v2 = re_v2 + 1j * im_v2

        # w1 = X + L ϕ
        w1 = v1 + apply_L(v2)
        # w2 = -L X - reg * ϕ
        w2 = -apply_L(v1) - reg * v2

        return np.concatenate([w1.real, w1.imag, w2.real, w2.imag])

    total_real = 4 * N
    M_op = LinearOperator(
        shape=(total_real, total_real),
        matvec=matvec_M_real,
        rmatvec=matvec_M_real,          # real symmetric
        dtype=float,
    )

    # -------- right‑hand side --------
    rhs_real = np.zeros(total_real, dtype=float)
    rhs_real[0:N]      = h_flat.real
    rhs_real[N:2*N]    = h_flat.imag

    if verbose:
        print(f"Starting regularised MINRES (reg={reg:.0e}) on "
              f"{total_real} × {total_real} system, tol={tol:.0e}, maxiter={maxiter}")
    sys.stdout.flush()
    # -------- solve --------
    x0 = np.zeros(total_real, dtype=float)      # minimum‑norm solution
    sol, info = minres(M_op, rhs_real, x0=x0, rtol=tol, maxiter=maxiter,
                       show=verbose)

    if verbose:
        if info == 0:
            print("MINRES converged (residual below tolerance).")
        elif info == 1:
            print("MINRES reached maximum number of iterations.")
        else:
            print(f"MINRES stopped with info={info}.")
    sys.stdout.flush()
    # -------- extract complex ϕ and compute X --------
    phi_flat = sol[2*N:3*N] + 1j * sol[3*N:4*N]
    X_flat   = h_flat - apply_L(phi_flat)

    # -------- diagnostics --------
    h_norm = float(np.linalg.norm(h_flat))
    L_H_flat = apply_L(h_flat)
    b_norm = float(np.linalg.norm(-L_H_flat))
    if b_norm < 1e-30:
        normal_res = 0.0
    else:
        normal_res = float(np.linalg.norm(-apply_L(apply_L(phi_flat)) + L_H_flat)) / b_norm
    cv = float(np.linalg.norm(apply_L(X_flat))) / max(h_norm, 1e-30)
    converged = (info == 0)

    X = {k: X_flat[fg_idx[k]*d2:(fg_idx[k]+1)*d2].reshape(d, d) for k in full_grid}
    result = SolverResult(X, phi_flat, normal_res, 0, converged, cv)

    if verbose:
        print(f"Constraint violation  ||L(X*)|| / ||H|| = {cv:.3e}")
        print(f"Normal‑eq residual      = {normal_res:.3e}")

    if save_result:
        out_path = save_solver_result(result, K1, K2, d, omega, save_dir=save_dir)
        if verbose:
            print(f"Saved SolverResult: {out_path}")
    sys.stdout.flush()
    return result









# ==============================================================================
# 1. Standard Double-Precision Damped LSMR
# ==============================================================================

def solve_d_lsmr(
    H: Dict,
    K1: int, K2: int,
    d: int,
    omega,
    lam: float = 1e-11,        # Tikhonov damping
    tol: float = 1e-13,
    maxiter: int = 500_000,
    fft_threshold: int = 0,
    force_direct: bool = False,
    verbose: bool = True,
    save_result: bool = False,
    save_dir: Optional[str | Path] = None,
):
    """
    Damped LSMR in standard float64.
    Solves min_phi || L phi - H ||^2 + lam^2 || phi ||^2.
    """
    omega = np.asarray(omega, dtype=float)
    full_grid, _, _, fg_idx = _build_modes(K1, K2)
    d2 = d * d
    N = len(full_grid) * d2

    h_flat = _build_h_flat(H, full_grid, fg_idx, d)

    use_fft = (len(full_grid) > fft_threshold) and not force_direct
    if use_fft:
        H_pos, N1f, N2f = _precompute_H_fft(H, K1, K2, d)
        def apply_L(v):
            return _apply_L_fft(v, H_pos, N1f, N2f, full_grid, fg_idx, d, omega)
    else:
        def apply_L(v):
            return _apply_L_direct(v, H, full_grid, fg_idx, d, omega)

    # L is skew-adjoint: L^† = -L
    L_op = LinearOperator(
        shape=(N, N),
        matvec=apply_L,
        rmatvec=lambda v: -apply_L(v),
        dtype=complex
    )

    if verbose:
        print(f"--- Double-Precision LSMR ---")
        print(f"damp={lam:.0e}, tol={tol:.0e}, N={N}, maxiter={maxiter}")
        sys.stdout.flush()

    res = lsmr(L_op, h_flat, damp=lam, atol=tol, btol=tol, maxiter=maxiter, show=verbose)
    phi = res[0]
    istop, itn = res[1], res[2]

    X_flat = h_flat - apply_L(phi)

    # Diagnostics
    h_norm = float(np.linalg.norm(h_flat))
    L_H_flat = apply_L(h_flat)
    b_norm = float(np.linalg.norm(-L_H_flat))
    normal_res = float(np.linalg.norm(-apply_L(apply_L(phi)) + L_H_flat)) / b_norm if b_norm > 1e-30 else 0.0
    cv = float(np.linalg.norm(apply_L(X_flat))) / max(h_norm, 1e-30)

    X = {k: X_flat[fg_idx[k]*d2:(fg_idx[k]+1)*d2].reshape(d,d) for k in full_grid}
    
    # Assuming SolverResult is a namedtuple or dataclass in your code
    # result = SolverResult(X=X, phi=phi, residual=normal_res, n_iter=itn, converged=(istop==1), constraint_violation=cv)

    if verbose:
        print(f"LSMR finished after {itn} iterations (istop={istop})")
        print(f"Constraint violation ||L(X)||/||H|| = {cv:.3e}")
        print(f"Normal eq residual              = {normal_res:.3e}\n")

    return X, phi, normal_res, itn, cv

# ==============================================================================
# 2. Arbitrary-Precision Damped LSMR (Native mpmath implementation)
# ==============================================================================

def _lsmr_mp(A_matvec, A_rmatvec, b, damp, tol, maxiter, verbose):
    """Native LSMR algorithm written purely in mpmath arithmetic."""
    n = len(b)
    if all(bi == 0 for bi in b):
        return [mp.mpc(0) for _ in range(n)], 1, 0

    beta = mp.sqrt(mp.fsum([mp.conj(bi)*bi for bi in b]))
    u = [bi / beta for bi in b]
    v_raw = A_rmatvec(u)
    alpha = mp.sqrt(mp.fsum([mp.conj(vi)*vi for vi in v_raw]))
    
    if alpha == 0:
        return [mp.mpc(0) for _ in range(n)], 1, 0
    v = [vi / alpha for vi in v_raw]

    w = v[:]
    x = [mp.mpc(0) for _ in range(n)]
    phi_bar = beta
    rho_bar = alpha

    for itn in range(1, maxiter+1):
        # Bidiagonalization
        Av = A_matvec(v)
        u_raw = [av - alpha * ui for av, ui in zip(Av, u)]
        beta = mp.sqrt(mp.fsum([mp.conj(ur)*ur for ur in u_raw]))
        if beta == 0: break
        u = [ur / beta for ur in u_raw]

        Adu = A_rmatvec(u)
        v_raw = [adu - beta * vi for adu, vi in zip(Adu, v)]
        alpha = mp.sqrt(mp.fsum([mp.conj(vr)*vr for vr in v_raw]))
        if alpha == 0: break
        v = [vr / alpha for vr in v_raw]

        # Givens rotation
        rho = mp.sqrt(rho_bar**2 + damp**2 + beta**2)
        c = rho_bar / rho
        s = beta / rho
        theta = s * alpha
        rho_bar = -c * alpha
        phi = c * phi_bar
        phi_bar = s * phi_bar

        # Update x and w
        x = [xi + (phi/rho) * wi for xi, wi in zip(x, w)]
        w = [vi - (theta/rho) * wi for vi, wi in zip(v, w)]

        # Convergence test
        if itn % 50 == 0 or itn == maxiter:
            r = [a_mv - bi for a_mv, bi in zip(A_matvec(x), b)]
            atr = A_rmatvec(r)
            atr_damp = [atri + damp**2 * xi for atri, xi in zip(atr, x)]
            norm_atr = mp.sqrt(mp.fsum([mp.conj(t)*t for t in atr_damp]))
            
            atb = A_rmatvec(b)
            norm_atb = mp.sqrt(mp.fsum([mp.conj(t)*t for t in atb]))
            rel_res = norm_atr / norm_atb if norm_atb > 0 else norm_atr
            
            if verbose and itn % 100 == 0:
                print(f"  mpLSMR iter {itn}: rel_res = {mp.nstr(rel_res, 4)}")

            if rel_res < tol:
                return x, 1, itn

    return x, 0, maxiter

def solve_d_lsmr_arbitraryPrecision(
    H: Dict,
    K1: int, K2: int,
    d: int,
    omega,
    lam: float = 1e-12,
    prec: int = 50,
    tol: float = 1e-25,
    maxiter: int = 1000,
    verbose: bool = True
):
    """
    Arbitrary-precision Damped LSMR. 
    Constructs operators and vectors entirely in mpmath.
    """
    mp.mp.dps = prec
    omega_mp = [mp.mpf(omega[0]), mp.mpf(omega[1])]
    full_grid, _, _, fg_idx = _build_modes(K1, K2)
    d2 = d * d
    N = len(full_grid) * d2

    # Map H to mpmath
    H_mp = {k: [[mp.mpc(c) for c in row] for row in mat.tolist()] for k, mat in H.items()}
    h_flat_mp = [mp.mpc('0') for _ in range(N)]
    
    for k in full_grid:
        if k in H_mp:
            start = fg_idx[k] * d2
            flat = [H_mp[k][i][j] for i in range(d) for j in range(d)]
            for idx, val in enumerate(flat):
                h_flat_mp[start + idx] = val

    mat_map = {k: mp.matrix(H_mp[k]) if k in H_mp else mp.matrix(d, d) for k in full_grid}
    zero_mat = mp.matrix(d, d)

    def apply_L_mp(v_flat_mp):
        X = {}
        for k in full_grid:
            blk = v_flat_mp[fg_idx[k]*d2 : (fg_idx[k]+1)*d2]
            X[k] = mp.matrix([[blk[i*d + j] for j in range(d)] for i in range(d)])

        out_flat = [mp.mpc('0') for _ in range(N)]
        for k in full_grid:
            comm = zero_mat.copy()
            kx, ky = k
            for m in full_grid:
                Hm = mat_map[m]
                # if Hm.is_zero: continue 
                km = (kx - m[0], ky - m[1])
                if km in fg_idx:
                    comm += Hm * X[km] - X[km] * Hm
            Lk = mp.j * comm + mp.j * (kx * omega_mp[0] + ky * omega_mp[1]) * X[k]
            
            start = fg_idx[k] * d2
            for i in range(d):
                for j in range(d):
                    out_flat[start + i*d + j] = Lk[i, j]
        return out_flat

    def apply_L_adj_mp(v):
        return [-x for x in apply_L_mp(v)]

    if verbose:
        print(f"--- Arbitrary-Precision LSMR ({prec} digits) ---")
        print(f"damp={lam:.0e}, tol={tol:.0e}, N={N}, maxiter={maxiter}")
        sys.stdout.flush()

    damp_mp = mp.mpf(str(lam))
    tol_mp = mp.mpf(str(tol))
    
    phi_mp, istop, itn = _lsmr_mp(
        apply_L_mp, apply_L_adj_mp, h_flat_mp, 
        damp=damp_mp, tol=tol_mp, maxiter=maxiter, verbose=verbose
    )

    ## Cast back to standard float64/complex128 for compatibility
    #phi_flat = np.array([complex(x) for x in phi_mp], dtype=complex)
    #
    # Standard Apply_L for fast diagnostics reconstruction
    #def apply_L_np(v):
    #    return _apply_L_direct(v, H, full_grid, fg_idx, d, omega)
    #
    #h_flat_np = _build_h_flat(H, full_grid, fg_idx, d)
    #X_flat = h_flat_np - apply_L_np(phi_flat)
    #
    #h_norm = float(np.linalg.norm(h_flat_np))
    #L_H_flat = apply_L_np(h_flat_np)
    #b_norm = float(np.linalg.norm(-L_H_flat))
    #normal_res = float(np.linalg.norm(-apply_L_np(apply_L_np(phi_flat)) + L_H_flat)) / b_norm if b_norm > 1e-30 else 0.0
    #cv = float(np.linalg.norm(apply_L_np(X_flat))) / max(h_norm, 1e-30)
    #
    #X = {k: X_flat[fg_idx[k]*d2:(fg_idx[k]+1)*d2].reshape(d, d) for k in full_grid}
    #
    #if verbose:
    #    print(f"mpLSMR finished: {itn} iters (istop={istop})")
    #    print(f"Constraint violation ||L(X)||/||H|| = {cv:.3e}")
    #    print(f"Normal eq residual              = {normal_res:.3e}\n")



    # ------------------------------------------------------------------------
    # Reconstruct X and diagnostics fully in mpmath (maximal accuracy)
    # ------------------------------------------------------------------------
    # X_mp = H - L(phi)
    L_phi_mp = apply_L_mp(phi_mp)
    X_mp = [h - l for h, l in zip(h_flat_mp, L_phi_mp)]

    # Constraint violation: ||L(X)|| / ||H||
    L_X_mp = apply_L_mp(X_mp)
    h_norm_mp = mp.sqrt(mp.fsum([mp.conj(v) * v for v in h_flat_mp]))
    cv_mp = mp.sqrt(mp.fsum([mp.conj(v) * v for v in L_X_mp])) / max(h_norm_mp, mp.mpf('1e-30'))

    # Normal‑equation residual: ||-L^2 φ - (-L(H))|| / ||-L(H)||
    L_H_mp = apply_L_mp(h_flat_mp)           # = L(H)
    L2_phi_mp = apply_L_mp(L_phi_mp)         # = L(L(φ))
    norm_b_mp = mp.sqrt(mp.fsum([mp.conj(v) * v for v in L_H_mp]))
    if norm_b_mp > mp.mpf('1e-30'):
        normal_res_mp = mp.sqrt(mp.fsum([mp.conj(a - b) * (a - b)
                                         for a, b in zip(L_H_mp, L2_phi_mp)])) / norm_b_mp
    else:
        normal_res_mp = mp.mpf('0.0')

    # ------------------------------------------------------------------------
    # Convert to standard float64/complex128 for the return API
    # ------------------------------------------------------------------------
    phi_flat = np.array([complex(x) for x in phi_mp], dtype=complex)
    X_flat   = np.array([complex(x) for x in X_mp],   dtype=complex)

    X = {k: X_flat[fg_idx[k] * d2 : (fg_idx[k] + 1) * d2].reshape(d, d)
         for k in full_grid}

    normal_res = float(normal_res_mp)
    cv         = float(cv_mp)

    if verbose:
        print(f"mpLSMR finished: {itn} iters (istop={istop})")
        print(f"Constraint violation ||L(X)||/||H|| = {mp.nstr(cv_mp, 3)}")
        print(f"Normal‑eq residual              = {mp.nstr(normal_res_mp, 3)}\n")

    return X, phi_flat, normal_res, itn, cv

def solve_eigreg(
    H: dict,
    K1: int, K2: int,
    d: int,
    omega,
    reg: float = 1e-4,
    fft_threshold: int = 0,
    force_direct: bool = False,
    verbose: bool = True,
    save_result: bool = False,
    save_dir: Optional[str | Path] = None,
) -> SolverResult:
    """
    Dense exact eigendecomposition solver with a smooth bump filter.
    
    Calculates AE via: X* = sum_i f(lambda_i) <v_i, H> v_i
    where f(λ) = exp(1 - 1/(1 - (|λ|/reg)^2)) for |λ| < reg, and 0 otherwise.
    
    WARNING: Scales as O(N^3). Only use for small K validation.
    """
    omega = np.asarray(omega, dtype=float)
    full_grid, _, _, fg_idx = _build_modes(K1, K2)
    d2 = d * d
    N = len(full_grid) * d2

    h_flat = _build_h_flat(H, full_grid, fg_idx, d)

    # 1. Build the dense skew-Hermitian matrix L
    if verbose:
        print(f"Building dense L matrix (N={N}) ...")
    sys.stdout.flush()
    L_mat = _apply_L_matvec(h_flat, H, full_grid, fg_idx, d, omega)

    # 2. Dense Eigendecomposition
    # Trick: i*L is Hermitian, so we can use eigh for guaranteed orthogonality and speed
    if verbose:
        print("Computing dense eigendecomposition (eigh on i*L) ...")
    sys.stdout.flush()
    evals_hermitian, V = np.linalg.eigh(1j * L_mat)
    
    # Recover original eigenvalues of L (which are purely imaginary)
    # L v = (-i * evals_hermitian) v
    eigenvalues = -1j * evals_hermitian

    # 3. Apply the smooth bump filter f(λ)
    if verbose:
        print(f"Applying smooth bump filter (reg={reg:.0e}) ...")
    sys.stdout.flush()
    x2 = (np.abs(eigenvalues) / reg)**2
    mask = x2 < 1.0  # Only keep modes within the regularization window
    
    f_val = np.zeros(N, dtype=float)
    f_val[mask] = np.exp(1.0 - 1.0 / (1.0 - x2[mask]))

    # 4. Construct the exact AE
    # Project H onto the eigenbasis: c_i = <v_i, H> = v_i^† H
    c = V.conj().T @ h_flat
    
    # Scale by the filter and reconstruct: X = sum (f(λ_i) * c_i) * v_i
    c_filtered = f_val * c
    X_flat = V @ c_filtered

    # Pack back into dictionary format
    X = {k: X_flat[fg_idx[k]*d2:(fg_idx[k]+1)*d2].reshape(d, d) for k in full_grid}

    # 5. Diagnostics
    h_norm = float(np.linalg.norm(h_flat))
    use_fft = (len(full_grid) > fft_threshold) and not force_direct
    if use_fft:
        H_pos, N1f, N2f = _precompute_H_fft(H, K1, K2, d)
        apply_L = lambda v: _apply_L_fft(v, H_pos, N1f, N2f, full_grid, fg_idx, d, omega)
    else:
        apply_L = lambda v: _apply_L_direct(v, H, full_grid, fg_idx, d, omega)

    cv = float(np.linalg.norm(apply_L(X_flat))) / max(h_norm, 1e-30)

    # Note: phi_flat is fundamentally not computed in this formulation, 
    # so we return a zero array for compatibility.
    phi_flat = np.zeros_like(X_flat)
    
    result = SolverResult(X, phi_flat, 0.0, 0, True, cv)

    if verbose:
        print(f"Dense eigendecomposition done.  ||L(X*)|| / ||H|| = {cv:.3e}")
        print(f"Modes retained strictly inside bump: {np.sum(mask)} / {N}")

    if save_result:
        out_path = save_solver_result(result, K1, K2, d, omega, save_dir=save_dir)
        if verbose:
            print(f"Saved SolverResult: {out_path}")
    sys.stdout.flush()
    return result

