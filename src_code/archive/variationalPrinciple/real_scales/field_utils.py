
# old scaled-𝐾+,0 version, not the current full_grid version.

"""
field_utils.py
--------------
Utilities for converting between:
  - real-space Hermitian matrix fields  A(theta)
  - unscaled Fourier coefficient dicts  {k: A_k}
  - scaled flat complex vectors         v_flat  used by the solver

Scaling convention  (see mode_sets.py)
    tilde_A_0 = A_0
    tilde_A_k = sqrt(2) * A_k   for k in K+

The scaled inner product satisfies
    <A,A>_{L2} = sum_{k in K+0} ||tilde_A_k||_F^2 = ||v_flat||_2^2
"""

from typing import Callable, Dict, List, Tuple

import numpy as np

from mode_sets import ModeVec, build_mode_sets, scaling_weights


# --------------------------------------------------------------------------- #
#  Unscaled Fourier dict  <->  flat scaled vector                              #
# --------------------------------------------------------------------------- #

def dict_to_flat(
    A_unscaled: Dict[ModeVec, np.ndarray],
    K_plus0: List[ModeVec],
) -> np.ndarray:
    """
    Pack the independent (K+0) unscaled Fourier coefficients into a flat
    scaled complex vector.

    v_flat[idx * d^2 : (idx+1) * d^2]  =  tilde_A_{K_plus0[idx]} . ravel()
    """
    d = next(iter(A_unscaled.values())).shape[0]
    d2 = d * d
    n_modes = len(K_plus0)
    v = np.empty(n_modes * d2, dtype=complex)

    for idx, k in enumerate(K_plus0):
        w_k = 1.0 if k == (0, 0) else np.sqrt(2.0)
        v[idx * d2 : (idx + 1) * d2] = (w_k * A_unscaled[k]).ravel()

    return v


def flat_to_dict(
    v_flat: np.ndarray,
    d: int,
    K_plus0: List[ModeVec],
    full_grid: List[ModeVec],
) -> Dict[ModeVec, np.ndarray]:
    """
    Unpack a flat scaled vector to a complete unscaled Fourier dict over
    all modes in full_grid (positive and negative halves).

    A_k for k in K+0:  A_k = v_flat[idx] / w_k
    A_{-k}            = A_k^dagger  (Hermitian symmetry)
    """
    d2 = d * d
    A: Dict[ModeVec, np.ndarray] = {}

    for idx, k in enumerate(K_plus0):
        w_k = 1.0 if k == (0, 0) else np.sqrt(2.0)
        mat = v_flat[idx * d2 : (idx + 1) * d2].reshape(d, d) / w_k
        A[k] = mat
        if k != (0, 0):
            A[(-k[0], -k[1])] = mat.conj().T

    return A


# --------------------------------------------------------------------------- #
#  Real-space reconstruction  A(theta)                                         #
# --------------------------------------------------------------------------- #

def make_field_function(
    A_unscaled: Dict[ModeVec, np.ndarray],
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Return a function  theta -> A(theta)  that evaluates the Hermitian field
        A(theta) = sum_{k in K} A_k exp(i k.theta)
    at a given point theta in R^2.

    Parameters
    ----------
    A_unscaled : complete unscaled Fourier dict (all modes, inc. negative half)

    Returns
    -------
    Callable  f(theta) -> d×d complex ndarray  (will be Hermitian if A is)
    """
    items = list(A_unscaled.items())

    def field(theta: np.ndarray) -> np.ndarray:
        d = next(iter(A_unscaled.values())).shape[0]
        result = np.zeros((d, d), dtype=complex)
        for k, mat in items:
            result += mat * np.exp(1j * float(np.dot(k, theta)))
        return result

    return field


def sample_field_on_grid(
    A_unscaled: Dict[ModeVec, np.ndarray],
    n_theta1: int,
    n_theta2: int,
) -> np.ndarray:
    """
    Sample  A(theta)  on a uniform n_theta1 x n_theta2 grid of the 2-torus.

    Returns array of shape (n_theta1, n_theta2, d, d).
    Uses IFFT for efficiency when the grid is dense.
    """
    d = next(iter(A_unscaled.values())).shape[0]

    # Place Fourier coefficients on the output grid
    A_grid = np.zeros((n_theta1, n_theta2, d, d), dtype=complex)
    for (k1, k2), mat in A_unscaled.items():
        A_grid[k1 % n_theta1, k2 % n_theta2] += mat

    # IFFT: A(theta_n) = N1*N2 * IFFT2(A_grid)[n]
    A_pos = np.fft.ifft2(A_grid, axes=(0, 1)) * (n_theta1 * n_theta2)
    return A_pos


# --------------------------------------------------------------------------- #
#  Hermiticity checks and random field generation                              #
# --------------------------------------------------------------------------- #

def check_hermitian_symmetry(
    A: Dict[ModeVec, np.ndarray],
    tol: float = 1e-12,
) -> float:
    """Return max relative error of A_{-k} - A_k^dagger over all positive k."""
    errs = []
    for k, mat in A.items():
        nk = (-k[0], -k[1])
        if nk in A:
            diff = A[nk] - mat.conj().T
            norm = max(np.linalg.norm(mat), 1e-30)
            errs.append(np.linalg.norm(diff) / norm)
    return float(max(errs)) if errs else 0.0


def random_hermitian_field(
    K1: int,
    K2: int,
    d: int,
    rng: np.random.Generator | None = None,
    scale: float = 1.0,
) -> Dict[ModeVec, np.ndarray]:
    """
    Generate a random Hermitian field with truncated Fourier expansion.

    Returns unscaled Fourier dict  {k: A_k}  for k in full_grid.
    Ensures A_{-k} = A_k^dagger  and A_0 = A_0^dagger.
    """
    if rng is None:
        rng = np.random.default_rng()

    full_grid, K_plus, K_plus0, _ = build_mode_sets(K1, K2)

    A: Dict[ModeVec, np.ndarray] = {}

    # Zero mode: Hermitian
    m = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    A[(0, 0)] = scale * (m + m.conj().T) / 2.0

    # Positive modes: arbitrary complex, negative = conjugate transpose
    for k in K_plus:
        m = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
        m *= scale / np.sqrt(2.0)
        A[k] = m
        A[(-k[0], -k[1])] = m.conj().T

    return A
