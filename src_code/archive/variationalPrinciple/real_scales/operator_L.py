
# This has direct and FFT matvecs, but it is the older scaled-𝐾+,0 version, not the current full_grid version.

"""
operator_L.py
-------------
Implements the discrete operator  L  corresponding to
    L(X) = i[H, X] + omega^mu * d_mu X
acting on the truncated scaled-Fourier coefficient vector.

The *unscaled* Fourier convention is
    A(theta) = sum_{k in K} A_k exp(i k.theta),   A_{-k} = A_k^dagger.

The *scaled* coefficients stored in the flat complex vector v are
    v[idx(k)] = tilde_A_k  where
        tilde_A_0 = A_0,
        tilde_A_k = sqrt(2) * A_k   for k in K+.

Skew-Hermiticity (proven in the derivation)
-------------------------------------------
Under  <A,B> = int Tr(A^dag B),
    L^dag = -L,    hence  A := -L^2 = L^dag L  is Hermitian PSD.

Two implementations are provided:
  - OperatorL_direct  :  O(|K|^2 d^3) per matvec via explicit convolution.
  - OperatorL_fft     :  O(|K| log|K| d^3) via FFT-based convolution.

Use OperatorL.build() to select automatically based on problem size.
"""

from __future__ import annotations
from typing import Dict, List, Tuple

import numpy as np

from mode_sets import ModeVec, build_mode_sets, scaling_weights, fft_pad_sizes


# --------------------------------------------------------------------------- #
#  Unscaled-to-full-grid helpers                                               #
# --------------------------------------------------------------------------- #

def _unscale_to_full(
    v_flat: np.ndarray,
    d: int,
    K_plus0: List[ModeVec],
) -> Dict[ModeVec, np.ndarray]:
    """
    Convert a flat scaled coefficient vector to a dict mapping every mode in
    K_plus0 U (-K+) to its unscaled d×d matrix.

    v_flat[idx*d^2 : (idx+1)*d^2]  =  tilde_A_{K_plus0[idx]}  (complex, d×d)

    Returns X_full with keys covering *all* modes in K (full grid):
        k=0          : X_0      = tilde_x_0  (w_0=1)
        k in K+      : X_k      = tilde_x_k / sqrt(2)
        k in -K+     : X_{-k}   = X_k^dagger  (enforced)
    """
    d2 = d * d
    X_full: Dict[ModeVec, np.ndarray] = {}

    for idx, k in enumerate(K_plus0):
        mat = v_flat[idx * d2 : (idx + 1) * d2].reshape(d, d)
        if k == (0, 0):
            X_full[k] = mat                          # w_0 = 1
        else:
            unscaled = mat / np.sqrt(2.0)
            X_full[k] = unscaled
            X_full[(-k[0], -k[1])] = unscaled.conj().T  # Hermitian symmetry

    return X_full


def _scale_output_to_flat(
    L_unscaled: Dict[ModeVec, np.ndarray],
    d: int,
    K_plus0: List[ModeVec],
) -> np.ndarray:
    """
    Convert the dict of unscaled output modes to a flat scaled vector.
    Scaled output: tilde_{LX}_k = w_k * (LX)_k.
    """
    d2 = d * d
    out = np.empty(len(K_plus0) * d2, dtype=complex)
    for idx, k in enumerate(K_plus0):
        w_k = 1.0 if k == (0, 0) else np.sqrt(2.0)
        out[idx * d2 : (idx + 1) * d2] = (w_k * L_unscaled[k]).ravel()
    return out


# --------------------------------------------------------------------------- #
#  Direct (exact, slow for large grids)                                        #
# --------------------------------------------------------------------------- #

class OperatorL_direct:
    """
    Applies the scaled operator  tilde_L  via explicit mode-sum convolution.

    Complexity : O(|K|^2 * d^3)  per matvec.
    Exact for any truncation since no circular-aliasing assumption is made.
    """

    def __init__(
        self,
        H_unscaled: Dict[ModeVec, np.ndarray],
        full_grid: List[ModeVec],
        K_plus0: List[ModeVec],
        omega: np.ndarray,
        d: int,
    ):
        self.H_unscaled = H_unscaled
        self.full_grid  = full_grid
        self.K_plus0    = K_plus0
        self.omega      = omega
        self.d          = d
        self.n_modes    = len(K_plus0)
        self.N          = self.n_modes * d * d

    def apply(self, v_flat: np.ndarray) -> np.ndarray:
        """Apply  tilde_L  to a scaled coefficient vector."""
        d = self.d
        X_full = _unscale_to_full(v_flat, d, self.K_plus0)

        L_unscaled: Dict[ModeVec, np.ndarray] = {}

        for k in self.K_plus0:
            # Commutator sum: sum_{m in K} [H_m, X_{k-m}]
            comm = np.zeros((d, d), dtype=complex)
            for m in self.full_grid:
                km = (k[0] - m[0], k[1] - m[1])
                if km in X_full:
                    Hm  = self.H_unscaled[m]
                    Xkm = X_full[km]
                    comm += Hm @ Xkm - Xkm @ Hm

            # Derivative term: i(omega.k) X_k
            kdotw = float(np.dot(k, self.omega))
            Xk    = X_full[k]
            L_unscaled[k] = 1j * comm + 1j * kdotw * Xk

        return _scale_output_to_flat(L_unscaled, d, self.K_plus0)


# --------------------------------------------------------------------------- #
#  FFT-accelerated                                                             #
# --------------------------------------------------------------------------- #

class OperatorL_fft:
    """
    Applies the scaled operator  tilde_L  via FFT-based circular convolution.

    Complexity : O(|K| log|K| * d^3)  per matvec.

    Normalization derivation
    ------------------------
    Place unscaled coefficients A_k at grid index (k1 % N1, k2 % N2).
    Let  A_grid[k1%N1, k2%N2] = A_k   for k in K (zero elsewhere).

    Position-space values at gridpoints theta_n = (2*pi*n1/N1, 2*pi*n2/N2):
        A(theta_n) = sum_k A_k exp(i k.theta_n)
                   = N1*N2 * IFFT2(A_grid)[n1,n2]
    because  IFFT2(x)[n] = (1/N) sum_k x[k] exp(2*pi*i*n*k/N).

    Fourier coefficients of the commutator  C = i[H,X]:
        C_k = (1/N1N2) * sum_n  C(theta_n) exp(-2*pi*i*(k1*n1/N1+k2*n2/N2))
            = FFT2(C_pos)[k1%N1, k2%N2]  /  N1N2
    where  C_pos[n] = i[H(theta_n), X(theta_n)].

    Since H is fixed, we precompute
        H_pos = N1*N2 * IFFT2(H_grid)
    once.  Per matvec we compute
        X_pos = N1*N2 * IFFT2(X_grid)
        C_pos = i*(H_pos @ X_pos - X_pos @ H_pos)   (pointwise matrix mul)
        comm_k = FFT2(C_pos)[k1%N1, k2%N2]  /  N1N2
    """

    def __init__(
        self,
        H_unscaled: Dict[ModeVec, np.ndarray],
        full_grid: List[ModeVec],
        K_plus0: List[ModeVec],
        omega: np.ndarray,
        d: int,
        K1: int,
        K2: int,
    ):
        self.full_grid = full_grid
        self.K_plus0   = K_plus0
        self.omega     = omega
        self.d         = d
        self.n_modes   = len(K_plus0)
        self.N         = self.n_modes * d * d

        N1, N2 = fft_pad_sizes(K1, K2)
        self.N1 = N1
        self.N2 = N2

        # Build H on padded grid and precompute position-space H
        H_grid = np.zeros((N1, N2, d, d), dtype=complex)
        for (i, j), mat in H_unscaled.items():
            H_grid[i % N1, j % N2] += mat   # '+=' in case of aliased indices

        # H_pos[n1, n2, :, :] = H(theta_{n1,n2})
        self.H_pos = np.fft.ifft2(H_grid, axes=(0, 1)) * (N1 * N2)

    def apply(self, v_flat: np.ndarray) -> np.ndarray:
        """Apply  tilde_L  to a scaled coefficient vector."""
        d  = self.d
        N1 = self.N1
        N2 = self.N2

        # --- Build X_grid ---
        X_grid = np.zeros((N1, N2, d, d), dtype=complex)
        d2 = d * d
        for idx, k in enumerate(self.K_plus0):
            mat = v_flat[idx * d2 : (idx + 1) * d2].reshape(d, d)
            if k == (0, 0):
                unscaled = mat
            else:
                unscaled = mat / np.sqrt(2.0)
            i, j = k
            X_grid[i % N1, j % N2] += unscaled
            if k != (0, 0):
                X_grid[(-i) % N1, (-j) % N2] += unscaled.conj().T

        # --- Position-space X ---
        # X_pos[n1,n2,:,:] = X(theta_{n1,n2})
        X_pos = np.fft.ifft2(X_grid, axes=(0, 1)) * (N1 * N2)

        # --- Pointwise matrix commutator in position space ---
        # C_pos[n1,n2,:,:] = i[H(theta_n), X(theta_n)]
        # Uses einsum for batch matrix multiplication over the spatial grid
        C_pos = 1j * (
            np.einsum("...ik,...kj->...ij", self.H_pos, X_pos)
            - np.einsum("...ik,...kj->...ij", X_pos, self.H_pos)
        )

        # --- Commutator Fourier coefficients (C_k = FFT(C_pos) / N1N2) ---
        C_fft = np.fft.fft2(C_pos, axes=(0, 1)) / (N1 * N2)

        # --- Assemble output ---
        L_unscaled: Dict[ModeVec, np.ndarray] = {}
        for idx, k in enumerate(self.K_plus0):
            i, j   = k
            comm_k = C_fft[i % N1, j % N2]            # (i[H,X])_k

            # Unscaled X_k
            mat = v_flat[idx * d2 : (idx + 1) * d2].reshape(d, d)
            Xk  = mat if k == (0, 0) else mat / np.sqrt(2.0)

            kdotw = float(np.dot(k, self.omega))
            L_unscaled[k] = comm_k + 1j * kdotw * Xk

        return _scale_output_to_flat(L_unscaled, d, self.K_plus0)


# --------------------------------------------------------------------------- #
#  Unified interface                                                           #
# --------------------------------------------------------------------------- #

class OperatorL:
    """
    Wraps either OperatorL_direct or OperatorL_fft.

    Provides:
        .apply(v_flat)                  -> tilde_L @ v_flat
        .apply_A(v_flat)                -> (-tilde_L^2) @ v_flat   (= L^dag L)
        .N                              -> total length of flat vector
        .build_rhs(h_flat)              -> -tilde_L @ h_flat  (= tilde_b)

    Parameters
    ----------
    H_unscaled : dict  {(k1,k2): d×d complex ndarray}  for k in full_grid
    K1, K2     : truncation parameters (integers >= 1)
    d          : local Hilbert space dimension
    omega      : frequency vector, shape (2,)
    fft_threshold : switch to FFT path when |K| > this  (default 2000)
    force_direct  : if True, always use direct (for testing)
    force_fft     : if True, always use FFT
    """

    def __init__(
        self,
        H_unscaled: Dict[ModeVec, np.ndarray],
        K1: int,
        K2: int,
        d: int,
        omega: np.ndarray,
        fft_threshold: int = 2000,
        force_direct: bool = False,
        force_fft: bool = False,
    ):
        full_grid, K_plus, K_plus0, k2idx = build_mode_sets(K1, K2)

        self.K1       = K1
        self.K2       = K2
        self.d        = d
        self.omega    = omega
        self.K_plus0  = K_plus0
        self.K_plus   = K_plus
        self.full_grid= full_grid
        self.k2idx    = k2idx
        self.n_modes  = len(K_plus0)
        self.N        = self.n_modes * d * d

        use_fft = (len(full_grid) > fft_threshold) and not force_direct
        if force_fft:
            use_fft = True

        if use_fft:
            self._impl = OperatorL_fft(
                H_unscaled, full_grid, K_plus0, omega, d, K1, K2
            )
        else:
            self._impl = OperatorL_direct(
                H_unscaled, full_grid, K_plus0, omega, d
            )

    def apply(self, v_flat: np.ndarray) -> np.ndarray:
        """Compute  tilde_L(v)."""
        return self._impl.apply(v_flat)

    def apply_A(self, v_flat: np.ndarray) -> np.ndarray:
        """
        Compute  A(v) = -tilde_L^2(v) = tilde_L^dag(tilde_L(v)).

        A is Hermitian PSD; this is used as the matrix in the CG solve.
        """
        return -self.apply(self.apply(v_flat))

    def build_rhs(self, h_flat: np.ndarray) -> np.ndarray:
        """
        Compute  tilde_b = -tilde_L(h)  analytically.

        Uses L(H) = omega^mu d_mu H  (i.e. [H,H]=0).
        In scaled Fourier:
            (tilde_L h)_0 = 0
            (tilde_L h)_k = i(k.omega) tilde_h_k    for k in K+

        So  tilde_b_k = -i(k.omega) tilde_h_k,   tilde_b_0 = 0.

        This is exact for the truncated system and avoids one operator apply.
        """
        d2 = self.d * self.d
        b = np.zeros(self.N, dtype=complex)
        for idx, k in enumerate(self.K_plus0):
            if k == (0, 0):
                continue  # b_0 = 0
            kdotw = float(np.dot(k, self.omega))
            b[idx * d2 : (idx + 1) * d2] = (
                -1j * kdotw * h_flat[idx * d2 : (idx + 1) * d2]
            )
        return b
