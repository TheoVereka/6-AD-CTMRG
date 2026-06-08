import numpy as np
from solver import solve, solve_direct, solve_saddle, solve_minres, solve_minres_reg
import sys
GOLDEN_RATIO = (1 + np.sqrt(5)) / 2
N_HILBERT = 4
OMEGA_GOLDENRATIO = np.array([1.0, GOLDEN_RATIO])

SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)
SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI = np.array([SIGMA_X, SIGMA_Y, SIGMA_Z], dtype=complex)
IDENTITY_2 = np.eye(2, dtype=complex)
PAULI_A = np.array([np.kron(sigma, IDENTITY_2) for sigma in PAULI], dtype=complex)
PAULI_B = np.array([np.kron(IDENTITY_2, sigma) for sigma in PAULI], dtype=complex)


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def _build_full_grid(K1: int, K2: int) -> list:
    return [(i, j) for i in range(-K1, K1 + 1) for j in range(-K2, K2 + 1)]


def _build_K_plus0(full_grid: list) -> list:
    return [(0, 0)] + [(i, j) for (i, j) in full_grid if i > 0 or (i == 0 and j > 0)]


def _twoqubit_operator(v_A: np.ndarray, v_B: np.ndarray) -> np.ndarray:
    return np.tensordot(v_A, PAULI_A, axes=(0, 0)) + np.tensordot(v_B, PAULI_B, axes=(0, 0))


def _b_field(t1, t2, m: float):
    return np.sqrt(np.sin(t1)**2 + np.sin(t2)**2 + (m + np.cos(t1) + np.cos(t2))**2)


def _z_field(t1, t2, Delta: float, epsilon: float, omega: np.ndarray):
    return 0.0


def _compute_ae_prefactors(
    Delta: float,
    m: float,
    epsilon: float,
    omega: np.ndarray,
    N1: int,
    N2: int,
) -> tuple:
    theta_1_vals = 2.0 * np.pi * np.arange(N1) / N1
    theta_2_vals = 2.0 * np.pi * np.arange(N2) / N2
    t1_grid, t2_grid = np.meshgrid(theta_1_vals, theta_2_vals, indexing="ij")

    b_grid = _b_field(t1_grid, t2_grid, m)

    ae_b = float(np.mean(b_grid))
    return 0.0, ae_b


def _project_theta_field_to_fourier(field_theta, K1: int, K2: int, d: int) -> dict:
    """
    Project a Hermitian field A(θ1, θ2) onto the solver's Fourier convention

        A(θ) = sum_k A_k exp(i k·θ),

    by sampling on a uniform grid and applying fft2 / (N1*N2).

    The solver stores the unscaled Fourier coefficients A_k directly in the
    mode dictionary.  The power-of-two choice here is only for FFT speed; it
    does not change the convention or the target coefficients.
    """
    N1 = _next_pow2(4 * K1 + 1)
    N2 = _next_pow2(4 * K2 + 1)

    theta_1_vals = 2 * np.pi * np.arange(N1) / N1
    theta_2_vals = 2 * np.pi * np.arange(N2) / N2

    field_grid = np.empty((N1, N2, d, d), dtype=complex)
    for i, theta_1 in enumerate(theta_1_vals):
        for j, theta_2 in enumerate(theta_2_vals):
            mat = np.asarray(field_theta(float(theta_1), float(theta_2)), dtype=complex)
            field_grid[i, j] = 0.5 * (mat + mat.conj().T)

    coeff_grid = np.fft.fft2(field_grid, axes=(0, 1)) / (N1 * N2)
    full_grid = _build_full_grid(K1, K2)
    field = {k: coeff_grid[k[0] % N1, k[1] % N2].copy() for k in full_grid}

    K_plus0 = _build_K_plus0(full_grid)
    for k in K_plus0:
        if k == (0, 0):
            field[k] = 0.5 * (field[k] + field[k].conj().T)
            continue
        k_neg = (-k[0], -k[1])
        sym = 0.5 * (field[k] + field[k_neg].conj().T)
        field[k] = sym
        field[k_neg] = sym.conj().T
    return field


def _make_H(K1: int, K2: int, d: int, Delta: float, m: float, epsilon: float, omega: np.ndarray) -> tuple:
    """
    Returns (H, full_grid) :
        H_{-k} = H_k†,   H_0 = H_0†
    H has keys for every mode in full_grid = {(i,j): |i|≤K1, |j|≤K2}.
    """
    if d != 4:
        raise ValueError("This script uses a two-qubit construction, so d must be 4.")

    full_grid = _build_full_grid(K1, K2)
    H = _project_theta_field_to_fourier(
        field_theta=lambda theta_1, theta_2: _H_theta(theta_1, theta_2, Delta, m, epsilon, omega),
        K1=K1,
        K2=K2,
        d=d,
    )
    return H, full_grid


def _make_AE(K1: int, K2: int, d: int, Delta: float, m: float, epsilon: float, omega: np.ndarray) -> dict:
    """
    Returns AE :
        AE_{-k} = AE_k†,   AE_0 = AE_0†
    AE has keys for every mode in full_grid = {(i,j): |i|≤K1, |j|≤K2}.
    """
    if d != 4:
        raise ValueError("This script uses a two-qubit construction, so d must be 4.")

    N1 = _next_pow2(4 * K1 + 1)
    N2 = _next_pow2(4 * K2 + 1)
    ae_a, ae_b = _compute_ae_prefactors(Delta, m, epsilon, omega, N1, N2)

    if abs(Delta)>2+epsilon:
        AE = _project_theta_field_to_fourier(
            field_theta=lambda theta_1, theta_2: _AE_theta_Abel(
                theta_1, theta_2, Delta, m, epsilon, omega, ae_a, ae_b
            ),
            K1=K1,
            K2=K2,
            d=d,
        )
    elif abs(Delta)<2-epsilon:
        AE = _project_theta_field_to_fourier(
            field_theta=lambda theta_1, theta_2: _AE_theta_nonAbel(
                theta_1, theta_2, m, ae_b
            ),
            K1=K1,
            K2=K2,
            d=d,
        )
    else:
        raise ValueError("|Delta|-2∈O(ε) is where the construction fails; this script does not handle that case.")

    return AE


def _norm2(A: dict, keys: list) -> float:
    """||A||² = sum_k ||A[k]||_F²"""
    return sum(np.linalg.norm(A[k], 'fro')**2 for k in keys)


def _diff_norm2(A: dict, B: dict, keys: list) -> float:
    """||A - B||²"""
    return sum(np.linalg.norm(A[k] - B[k], 'fro')**2 for k in keys)


def _rel_diff(A: dict, B: dict, keys: list) -> float:
    """Relative Frobenius error: ||A-B|| / ||A|| over all mode matrices."""
    num = _diff_norm2(A, B, keys)**0.5
    den = _norm2(A, keys)**0.5
    return float(num / max(den, 1e-30))













def _H_theta(t1: float, t2: float, Delta: float, m: float, epsilon: float, omega: np.ndarray) -> np.ndarray:
    o1, o2 = omega[0], omega[1]
    v_A = np.array([
        epsilon,
        0,
        Delta+np.cos(t1)+np.cos(t2),
    ], dtype=float)
    v_B = np.array([
        np.sin(t1)+2/(np.sin(t1)**2+np.sin(t2)**2+(m+np.cos(t1)+np.cos(t2))**2)*(-o1*np.sin(t1)*np.sin(t2)-o2*(1+(m+np.cos(t1))*np.cos(t2))),
        np.sin(t2)+2/(np.sin(t1)**2+np.sin(t2)**2+(m+np.cos(t1)+np.cos(t2))**2)*(o1*(1+(m+np.cos(t2))*np.cos(t1))+o2*np.sin(t1)*np.sin(t2)),
        m+np.cos(t1)+np.cos(t2)+2/(np.sin(t1)**2+np.sin(t2)**2+(m+np.cos(t1)+np.cos(t2))**2)*(-o1*np.cos(t1)*np.sin(t2)+o2*np.sin(t1)*np.cos(t2)),
    ], dtype=float)
    return _twoqubit_operator(v_A, v_B)


def _AE_theta_Abel(
    t1: float,
    t2: float,
    Delta: float,
    m: float,
    epsilon: float,
    omega: np.ndarray,
    ae_a: float,
    ae_b: float,
) -> np.ndarray:
    z_val = _z_field(t1, t2, Delta, epsilon, omega)
    z_abs2 = float(np.real(z_val.conj() * z_val))
    b_val = float(_b_field(t1, t2, m))
    b_safe = max(b_val, 1e-15)

    v_A = ae_a * np.array([
        2.0 * np.real(z_val),
        2.0 * np.imag(z_val),
        1.0 - 2.0 * z_abs2,
    ], dtype=float)
    v_B = (ae_b / b_safe) * np.array([
        np.sin(t1),
        np.sin(t2),
        m + np.cos(t1) + np.cos(t2),
    ], dtype=float)
    return _twoqubit_operator(v_A, v_B)


def _AE_theta_nonAbel(
    t1: float,
    t2: float,
    m: float,
    ae_b: float,
) -> np.ndarray:
    b_val = float(_b_field(t1, t2, m))
    b_safe = max(b_val, 1e-15)
    v_B = (ae_b / b_safe) * np.array([
        np.sin(t1),
        np.sin(t2),
        m + np.cos(t1) + np.cos(t2),
    ], dtype=float)
    return np.tensordot(v_B, PAULI_B, axes=(0, 0))


FFT_THRESHOLD = 0
TOL = 1e-10
MAXITER = round(1e8)

K1, K2 = 25, 25

Delta = 0.0
m = 3.0

epsilon = 0.0431
omega_norm = 0.032145710251

omega = omega_norm * OMEGA_GOLDENRATIO / np.linalg.norm(OMEGA_GOLDENRATIO)
H, full_grid = _make_H(K1, K2, N_HILBERT, Delta, m, epsilon, omega)
AE = _make_AE(K1, K2, N_HILBERT, Delta, m, epsilon, omega)

AE_varitional = solve_minres_reg(H, K1, K2, N_HILBERT, omega,
                      tol=TOL, 
                      maxiter=MAXITER, 
                      fft_threshold=FFT_THRESHOLD,
                      save_result=True,
                      save_dir=r"/scratch/chye/varPri/reg25",
                      reg=1e-12)

AE_rel_error = _rel_diff(AE, AE_varitional.X, full_grid)
print(f"K=({K1},{K2}), N={N_HILBERT}")
print(f"    complex: conv={AE_varitional.converged}  it={AE_varitional.n_iter:4d}  "
        f"cv={AE_varitional.constraint_violation:.2e}  res={AE_varitional.residual:.2e}")
print(f"    rel ||AE - AE_varitional|| / ||AE|| = {AE_rel_error:.3e}")
sys.stdout.flush()