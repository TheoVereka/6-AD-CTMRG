from __future__ import annotations
import os
import sys

# ==============================================================================
# 1. CLUSTER THREAD PINNING (Robust Fallback Layout)
# ==============================================================================
try:
    NUM_CORES = len(os.sched_getaffinity(0))
except (AttributeError, NotImplementedError):
    NUM_CORES = os.cpu_count() or 72

os.environ["MKL_NUM_THREADS"] = str(NUM_CORES)
os.environ["OPENBLAS_NUM_THREADS"] = str(NUM_CORES)
os.environ["OMP_NUM_THREADS"] = str(NUM_CORES)
os.environ["MKL_DYNAMIC"] = "FALSE"

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Callable, Union, Literal
import numpy as np
import scipy.linalg
import warnings
from scipy.sparse.linalg import LinearOperator, lsmr, lsqr

DEFAULT_RESULT_DIR = Path(r"/scratch/chye/varPri")

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

def _build_h_flat(H: dict, full_grid: list, fg_idx: dict, d: int) -> np.ndarray:
    d2     = d * d
    h_flat = np.zeros(len(full_grid) * d2, dtype=complex)
    for k in full_grid:
        if k in H:
            h_flat[fg_idx[k] * d2 : (fg_idx[k] + 1) * d2] = H[k].ravel()
    return h_flat

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



if True:
   



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



# Assuming your helper functions and SolverResult dataclass are imported here:
# from .utils import _build_modes, _build_h_flat, _precompute_H_fft, _apply_L_fft, _apply_L_direct, save_solver_result
# from .types import SolverResult

def solve_krylov(
    H: Dict,
    K1: int, 
    K2: int,
    d: int,
    omega,
    method: Literal['lsmr', 'lsqr'] = 'lsmr',
    lam: float = 1e-11,        # Tikhonov damping
    tol: float = 1e-13,
    maxiter: int = 500_000,
    fft_threshold: int = 0,
    force_direct: bool = False,
    verbose: bool = True,
    save_result: bool = False,
    save_dir: Optional[str | Path] = None,
) -> SolverResult:
    """
    Unified Damped Least Squares solver using LSMR or LSQR in float64.
    Solves min_phi || L phi - H ||^2 + lam^2 || phi ||^2.
    """
    omega = np.asarray(omega, dtype=float)
    full_grid, _, _, fg_idx = _build_modes(K1, K2)
    d2 = d * d
    N = len(full_grid) * d2

    h_flat = _build_h_flat(H, full_grid, fg_idx, d)
    h_norm = float(np.linalg.norm(h_flat)) or 1e-30

    # Route the operator application
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

    method = method.lower()
    if verbose:
        print(f"--- Double-Precision {method.upper()} ---")
        print(f"damp={lam:.0e}, tol={tol:.0e}, N={N}, maxiter={maxiter}, use_fft={use_fft}")
        sys.stdout.flush()

    # Route to chosen SciPy solver
    if method == 'lsmr':
        # lsmr returns: (x, istop, itn, normr, normar, normA, condA, normx)
        res = lsmr(L_op, h_flat, damp=lam, atol=tol, btol=tol, maxiter=maxiter, show=verbose)
        phi, istop, itn, normar = res[0], res[1], res[2], res[4]
    elif method == 'lsqr':
        # lsqr returns: (x, istop, itn, r1norm, r2norm, anorm, acond, arnorm, xnorm, var)
        # Note: lsqr uses `iter_lim` instead of `maxiter`
        res = lsqr(L_op, h_flat, damp=lam, atol=tol, btol=tol, iter_lim=maxiter, show=verbose)
        phi, istop, itn, normar = res[0], res[1], res[2], res[7]
    else:
        raise ValueError(f"Unknown method '{method}'. Please choose 'lsmr' or 'lsqr'.")

    # 1: exact solution, 2: least-squares residual < tol, 3: normal eq residual < tol
    converged = istop in (1, 2, 3)

    # Calculate final Action Error: X* = H - L phi
    L_phi = apply_L(phi)
    X_flat = h_flat - L_phi
    X = {k: X_flat[fg_idx[k]*d2:(fg_idx[k]+1)*d2].reshape(d, d) for k in full_grid}

    # Diagnostics
    L_X_flat = apply_L(X_flat)
    cv = float(np.linalg.norm(L_X_flat)) / h_norm
    
    # Calculate b_norm ( || L H || ) for normalizing the normal residual
    # We use apply_L(h_flat) because ||-L H|| == ||L H||
    L_H_flat = apply_L(h_flat)
    b_norm = float(np.linalg.norm(L_H_flat)) or 1e-30
    
    abs_normal_res = float(normar)
    rel_normal_res = abs_normal_res / b_norm

    result = SolverResult(
        X=X,
        phi_flat=phi,
        residual=rel_normal_res,  # Keeping the standard API, but we print both below
        n_iter=itn,
        converged=converged,
        constraint_violation=cv,
    )

    if verbose:
        if istop == 4:
            print(f"  --> WARNING: {method.upper()} reached the iteration limit ({maxiter}) without convergence.")
        elif istop == 7:
            print(f"  --> WARNING: {method.upper()} halted due to extreme ill-conditioning!")
            print(f"  --> Consider increasing Tikhonov damping (lam) or checking grid scaling.")
        print(f"Constraint violation ||L(X)||/||H|| = {cv:.3e}")
        print(f"Absolute normal residual            = {abs_normal_res:.3e}")
        print(f"Relative normal residual            = {rel_normal_res:.3e}\n")

    if save_result:
        out_path = save_solver_result(result, K1, K2, d, omega, save_dir=save_dir)
        if verbose:
            print(f"Saved {method.upper()} result: {out_path}")

    return result