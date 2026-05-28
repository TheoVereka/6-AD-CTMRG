"""
solver.py
---------
Solves the variational problem:

    min_{X} ||H - X||^2_{L2}     s.t.   L(X) = 0
    L(X) = i[H, X] + omega^mu d_mu X

via the equivalent normal equation

    A phi = b,    A = -L^2 = L^dag L  (Hermitian PSD)
    b_k = -i(k.omega) tilde_h_k

and recovers  X = H - L(phi).

The CG method is used because A is Hermitian PSD.
A diagonal preconditioner M_k = (k.omega)^2 + eps  is applied per mode
to account for the dominant (derivative) contribution to A.

Result dataclass
----------------
SolverResult.X_unscaled   : complete unscaled Fourier dict  {k: X_k}
SolverResult.phi_flat     : the Lagrange multiplier as flat vector
SolverResult.residual     : ||A phi - b|| / ||b|| at convergence
SolverResult.n_iter       : CG iteration count
SolverResult.converged    : bool
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict

import numpy as np

from scipy.sparse.linalg import cg, LinearOperator

from mode_sets import ModeVec, build_mode_sets
from operator_L import OperatorL
from field_utils import dict_to_flat, flat_to_dict


def _c2r(v: np.ndarray) -> np.ndarray:
    """Stack [Re(v); Im(v)] → real vector of length 2N."""
    return np.concatenate([v.real, v.imag])


def _r2c(v: np.ndarray) -> np.ndarray:
    """Recover complex vector from [Re; Im] stacking."""
    n = len(v) // 2
    return v[:n] + 1j * v[n:]


def _make_real_matvec(matvec_complex):
    """
    Wrap a complex linear operator A : C^N → C^N to act on real R^{2N}.

    For Hermitian field vectors v ∈ C^N, A maps Hermitian → Hermitian.
    The real operator A_real : R^{2N} → R^{2N} is then symmetric PSD
    (under the standard Euclidean real inner product), allowing real CG.

    A_real([Re(v); Im(v)]) = [Re(A v); Im(A v)]
    """
    def matvec_real(vr: np.ndarray) -> np.ndarray:
        vc = _r2c(vr)
        Avc = matvec_complex(vc)
        return _c2r(Avc)
    return matvec_real


# ---------------------------------------------------------------------------
# Reduced real parametrization  R^{N_real} ↔ Hermitian flat C^N
# ---------------------------------------------------------------------------
# For the zero mode (d×d Hermitian matrix A_0):
#   Independent DOF = diagonal (d real) + upper triangle (d(d-1)/2 complex
#   = d(d-1) real).  Total: d² real DOF.
#   Stored as: [A_0_{00}, A_0_{11}, ..., Re(A_0_{01}), Im(A_0_{01}), ...]
#
# For each positive mode k∈K+ (d×d arbitrary complex A_k):
#   2d² real DOF.  Stored as [Re(A_k).ravel(), Im(A_k).ravel()].
#
# N_real = d² + |K+| × 2d²   (vs   N = d² × (1 + |K+|) for complex flat)
# ---------------------------------------------------------------------------

def _n_real(d: int, n_K_plus: int) -> int:
    return d * d + n_K_plus * 2 * d * d


def _herm_flat_to_real(v_flat: np.ndarray, d: int, K_plus0: list) -> np.ndarray:
    """Map Hermitian flat vector v_flat ∈ C^N → reduced real vector ∈ R^{N_real}.

    The real Euclidean inner product on R^{N_real} equals
    Re[v_flat† w_flat] (the true L² inner product).

    Zero mode (Hermitian d×d matrix A_0):
      - Diagonal d entries: r_i = A_{0,ii}    (no scaling)
      - Off-diag d(d-1) entries: r_{ij,re} = √2 Re(A_{0,ij}),
                                 r_{ij,im} = √2 Im(A_{0,ij})
    Positive modes (arbitrary complex d×d): r_k = [Re(ã_k).ravel(), Im(ã_k).ravel()]
    (ã_k = √2 A_k is already stored in v_flat, no additional scaling.)
    """
    d2 = d * d
    SQ2 = np.sqrt(2.0)
    parts: list[np.ndarray] = []

    # --- Zero mode ---
    A0 = v_flat[0:d2].reshape(d, d)
    r0 = np.empty(d2)
    ptr = 0
    for i in range(d):
        r0[ptr] = A0[i, i].real
        ptr += 1
    for i in range(d):
        for j in range(i + 1, d):
            r0[ptr]     = SQ2 * A0[i, j].real
            r0[ptr + 1] = SQ2 * A0[i, j].imag
            ptr += 2
    parts.append(r0)

    # --- Positive modes ---
    for idx, k in enumerate(K_plus0):
        if k == (0, 0):
            continue
        Ak = v_flat[idx * d2 : (idx + 1) * d2]
        parts.append(Ak.real)
        parts.append(Ak.imag)

    return np.concatenate(parts)


def _real_to_herm_flat(r: np.ndarray, d: int, K_plus0: list) -> np.ndarray:
    """Inverse of _herm_flat_to_real."""
    d2 = d * d
    SQ2 = np.sqrt(2.0)
    n_modes = len(K_plus0)
    v_flat = np.zeros(n_modes * d2, dtype=complex)

    # --- Zero mode ---
    A0 = np.zeros((d, d), dtype=complex)
    ptr = 0
    for i in range(d):
        A0[i, i] = r[ptr]
        ptr += 1
    for i in range(d):
        for j in range(i + 1, d):
            A0[i, j] = (r[ptr] + 1j * r[ptr + 1]) / SQ2
            A0[j, i] = (r[ptr] - 1j * r[ptr + 1]) / SQ2
            ptr += 2
    v_flat[0:d2] = A0.ravel()

    # --- Positive modes ---
    rptr = d2
    for idx, k in enumerate(K_plus0):
        if k == (0, 0):
            continue
        v_flat[idx * d2 : (idx + 1) * d2] = r[rptr : rptr + d2] + 1j * r[rptr + d2 : rptr + 2 * d2]
        rptr += 2 * d2

    return v_flat


@dataclass
class SolverResult:
    X_unscaled: Dict[ModeVec, np.ndarray]
    phi_flat:   np.ndarray
    residual:   float
    n_iter:     int
    converged:  bool
    constraint_violation: float   # ||L(X)||_2 / ||H||_2  (should be ~tol)


def solve_variational(
    H_unscaled: Dict[ModeVec, np.ndarray],
    K1: int,
    K2: int,
    d: int,
    omega: np.ndarray,
    *,
    tol: float = 1e-12,
    maxiter: int = 10_000,
    reg: float = 1e-10,
    fft_threshold: int = 2000,
    force_direct: bool = False,
    force_fft: bool = False,
    verbose: bool = False,
) -> SolverResult:
    """
    Solve  min ||H-X||^2  s.t.  L(X)=0  on the truncated 2-torus.

    Parameters
    ----------
    H_unscaled    : complete unscaled Fourier dict {k: H_k} for k in full_grid
    K1, K2        : truncation parameters
    d             : matrix dimension
    omega         : incommensurate frequency (length-2 array)
    tol           : relative CG tolerance  ||r||/||b|| < tol
    maxiter       : max CG iterations
    reg           : Tikhonov regularization: solves (A + reg*I)phi = b instead
                    of A*phi = b. Removes singularity at k·ω=0 (zero mode).
                    The constraint violation scales as ~sqrt(reg)*||H||, so
                    reg = 1e-10 is negligible for typical inputs.
    fft_threshold : switch to FFT when |K| > this
    force_direct  : always use direct convolution (for testing/debugging)
    force_fft     : always use FFT
    verbose       : print CG progress

    Returns
    -------
    SolverResult
    """
    omega = np.asarray(omega, dtype=float)

    # --- Build operator ---
    L_op = OperatorL(
        H_unscaled, K1, K2, d, omega,
        fft_threshold=fft_threshold,
        force_direct=force_direct,
        force_fft=force_fft,
    )

    full_grid, K_plus, K_plus0, k2idx = build_mode_sets(K1, K2)
    N       = L_op.N
    n_modes = L_op.n_modes
    d2      = d * d

    # --- Scaled input vector  h_flat ---
    h_flat = dict_to_flat(H_unscaled, K_plus0)

    # --- Right-hand side  b = -L(h)  (computed analytically) ---
    b_flat = L_op.build_rhs(h_flat)

    # Fast exit: if b=0 (e.g. H is already L-invariant), X=H is the solution
    b_norm = float(np.linalg.norm(b_flat))
    if b_norm < 1e-30:
        if verbose:
            print("b=0: H already satisfies L(H)=0, returning X=H.")
        phi_flat = np.zeros(N, dtype=complex)
        X_unscaled = flat_to_dict(h_flat, d, K_plus0, full_grid)
        return SolverResult(
            X_unscaled=X_unscaled,
            phi_flat=phi_flat,
            residual=0.0,
            n_iter=0,
            converged=True,
            constraint_violation=0.0,
        )

    # --- Regularized operator  A_reg = A + reg*I ---
    # A = -L^2 is singular at k=0 (k·ω = 0) and near-singular for small k·ω.
    # Tikhonov regularization shifts all eigenvalues up by reg, making CG
    # convergent. Error in L(X) scales as ~sqrt(reg)*||H||.
    def matvec_A_reg(v: np.ndarray) -> np.ndarray:
        return L_op.apply_A(v) + reg * v

    # --- Reduced real parametrization ---
    # A_reg is symmetric only on the Hermitian subspace.  Lift to the
    # REDUCED real DOF space R^{N_real} which naturally enforces the
    # Hermitian constraint. This yields a truly symmetric PSD real operator.
    n_K_plus = len(K_plus)
    Nr = _n_real(d, n_K_plus)

    def A_reg_real(rr: np.ndarray) -> np.ndarray:
        vc = _real_to_herm_flat(rr, d, K_plus0)
        return _herm_flat_to_real(matvec_A_reg(vc), d, K_plus0)

    # --- Diagonal preconditioner in reduced-real space ---
    # zero mode block: reg (all d² entries)
    # positive-mode block: (k·ω)² + reg (all 2d² entries each)
    diag_r = np.zeros(Nr, dtype=float)
    diag_r[0 : d2] = reg
    rptr = d2
    for k in K_plus:
        kdotw2 = float(np.dot(k, omega)) ** 2
        diag_r[rptr : rptr + 2 * d2] = kdotw2 + reg
        rptr += 2 * d2

    def precond_real(rr: np.ndarray) -> np.ndarray:
        return rr / diag_r

    # --- Convert RHS to reduced real space ---
    b_real      = _herm_flat_to_real(b_flat, d, K_plus0)
    b_real_norm = float(np.linalg.norm(b_real))

    A_real_op = LinearOperator(shape=(Nr, Nr), matvec=A_reg_real, dtype=float)
    M_real_op = LinearOperator(shape=(Nr, Nr), matvec=precond_real, dtype=float)

    iteration_count = [0]

    def callback(xk):
        iteration_count[0] += 1
        if verbose and iteration_count[0] % 100 == 0:
            r = b_real - A_reg_real(xk)
            print(f"  CG iter {iteration_count[0]:5d}  "
                  f"rel_res = {np.linalg.norm(r)/b_real_norm:.3e}")

    phi_real, info = cg(
        A_real_op, b_real,
        x0=np.zeros(Nr),
        M=M_real_op,
        rtol=tol,
        maxiter=maxiter,
        callback=callback,
    )
    converged = (info == 0)
    n_iter    = iteration_count[0]

    phi_flat = _real_to_herm_flat(phi_real, d, K_plus0)

    # Compute residual wrt the regularized system
    res_vec  = b_flat - matvec_A_reg(phi_flat)
    residual = float(np.linalg.norm(res_vec)) / b_norm

    if verbose:
        status = "CONVERGED" if converged else "NOT CONVERGED"
        print(f"CG {status} in {n_iter} iterations, rel_res = {residual:.3e}")

    # --- Recover X = H - L(phi) ---
    Lphi_flat  = L_op.apply(phi_flat)
    x_flat     = h_flat - Lphi_flat
    X_unscaled = flat_to_dict(x_flat, d, K_plus0, full_grid)

    # --- Constraint violation check  ||L(X)||_2 / ||H||_2 ---
    LX_flat = L_op.apply(x_flat)
    h_norm  = float(np.linalg.norm(h_flat))
    cv      = float(np.linalg.norm(LX_flat)) / max(h_norm, 1e-30)

    if verbose:
        print(f"Constraint violation ||L(X)|| / ||H|| = {cv:.3e}")

    return SolverResult(
        X_unscaled=X_unscaled,
        phi_flat=phi_flat,
        residual=residual,
        n_iter=n_iter,
        converged=converged,
        constraint_violation=cv,
    )