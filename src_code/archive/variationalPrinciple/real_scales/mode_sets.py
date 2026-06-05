
# old scaled-𝐾+,0 version, not the current full_grid version.

"""
mode_sets.py
------------
Defines the truncated Fourier mode sets for the 2-torus T^2.

Convention
----------
Full mode grid  K  = { (k1,k2) : |k1|<=K1, |k2|<=K2 }
Positive half  K+  = { k in K : k1>0 } union { (0,k2) in K : k2>0 }
K^{+,0}        K+0 = K+ union {(0,0)}

These obey  K = -K+ union {0} union K+  (disjoint).

The independent scaled Fourier coefficients of a Hermitian field A are:
  tilde_A_0  = A_0                 (k=0,   Hermitian d×d matrix)
  tilde_A_k  = sqrt(2) * A_k      (k in K+,  unconstrained complex d×d)
with  A_{-k} = A_k^dagger  determining the negative-half modes.
"""

from typing import List, Tuple, Dict
import numpy as np

ModeVec = Tuple[int, int]


def build_mode_sets(K1: int, K2: int):
    """
    Returns
    -------
    full_grid : list of (k1,k2) covering all K
    K_plus    : list of positive-half modes (lexicographic order)
    K_plus0   : [( 0,0)] + K_plus
    k2idx     : dict mapping each mode in K_plus0 to its position index
    """
    full_grid: List[ModeVec] = [
        (i, j)
        for i in range(-K1, K1 + 1)
        for j in range(-K2, K2 + 1)
    ]

    K_plus: List[ModeVec] = [
        (i, j)
        for (i, j) in full_grid
        if i > 0 or (i == 0 and j > 0)
    ]

    K_plus0: List[ModeVec] = [(0, 0)] + K_plus

    k2idx: Dict[ModeVec, int] = {k: idx for idx, k in enumerate(K_plus0)}

    return full_grid, K_plus, K_plus0, k2idx


def scaling_weights(K_plus0: List[ModeVec]) -> np.ndarray:
    """
    Returns the real weight array w of length len(K_plus0):
      w[0] = 1          for the zero mode
      w[i] = sqrt(2)    for all positive modes

    With these weights,  ||A||^2 = sum_{k in K+0} ||tilde_A_k||_F^2
    where  tilde_A_k = w[k] * A_k.
    """
    w = np.array([1.0 if k == (0, 0) else np.sqrt(2.0) for k in K_plus0])
    return w


def next_power_of_two(n: int) -> int:
    """Smallest power of two >= n."""
    p = 1
    while p < n:
        p <<= 1
    return p


def fft_pad_sizes(K1: int, K2: int) -> Tuple[int, int]:
    """
    Compute padded FFT grid sizes for alias-free circular convolution.

    [H,X] with both H,X having modes in [-K,K] produces modes in [-2K,2K].
    We need the grid to have at least 4K+1 points in each direction to avoid
    aliasing when we truncate back to [-K,K] after the convolution.

    Rounds up to the next power of two for FFT efficiency.
    """
    N1 = next_power_of_two(4 * K1 + 1)
    N2 = next_power_of_two(4 * K2 + 1)
    return N1, N2
