from __future__ import annotations

import argparse
from pathlib import Path

from solver import load_solver_result, _build_modes
import numpy as np
import matplotlib.pyplot as plt

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Read and summarize a saved SolverResult (.npz)")
    parser.add_argument("file", type=str, help="Path to saved result file")
    args = parser.parse_args()

    result, meta = load_solver_result(Path(args.file))

    print(f"file: {meta['file_path']}")
    print(f"K=({meta['K1']},{meta['K2']}), d={meta['d']}, modes={len(meta['full_grid'])}")
    print(
        f"converged={result.converged}, iters={result.n_iter}, "
        f"residual={result.residual:.3e}, cv={result.constraint_violation:.3e}"
    )








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
    return 0


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
    ae_a = 0.0
    b_grid = _b_field(t1_grid, t2_grid, m)
    ae_b = float(np.mean(b_grid))
    return ae_a, ae_b


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

def _mode_dict_to_lattice_array(mode_dict: dict, K1: int, K2: int) -> np.ndarray:
    arr = np.zeros((2 * K1 + 1, 2 * K2 + 1), dtype=float)
    for (n1, n2), mat in mode_dict.items():
        arr[n1 + K1, n2 + K2] = np.linalg.norm(mat, ord='fro')
    return arr


def plot_array_circles(arr, min_size=-0.1, max_size=25, cmap='viridis_r', mode='log'):
    """
    Plot a 2D array as circles on a grid, with circle sizes proportional to array values.

    Parameters:
    - arr: 2D numpy array of values.
    - min_size: minimum marker size (area in points^2) for the smallest circle.
    - max_size: maximum marker size (area in points^2) for the largest circle.
    - cmap: colormap for coloring circles based on values.
    """
    arr = np.abs(arr)
    ncols, nrows = arr.shape

    xs, ys = np.meshgrid(np.arange(ncols), np.arange(nrows))
    xs = xs.flatten() - ncols // 2
    ys = ys.flatten() - nrows // 2
    vals = arr.T.flatten()

    total_norm = np.sqrt(np.sum(vals**2))
    total_norm = max(float(total_norm), 1e-30)
    norm_vals = vals / total_norm
    sizes = norm_vals * (max_size - min_size) + min_size
    if mode == 'log':
        norm_vals = np.log10(norm_vals + 1e-15)
        sizes = (norm_vals + 7) / 3 * (max_size - min_size) + min_size

    sizes = np.clip(sizes, a_min=0, a_max=None)

    plt.figure(figsize=(12, 12 * nrows / ncols))
    scatter = plt.scatter(
        xs,
        ys,
        s=sizes,
        c=norm_vals,
        cmap=cmap,
        clim=[-7, 0],
        alpha=np.clip((norm_vals + 8) / 11, a_min=0, a_max=1),
    )

    plt.gca().set_aspect('equal')
    plt.xlim(-ncols // 2, ncols // 2 + 1)
    plt.ylim(-nrows // 2, nrows // 2 + 1)
    plt.grid(True, linestyle='--', alpha=0.5)

    cbar = plt.colorbar(scatter, orientation='vertical')
    cbar.set_label(r'$\mathrm{log}_{10}\frac{|\mathcal{M}_{\vec{n}}|}{\sqrt{\sum_{\vec{n}}|\mathcal{M}_{\vec{n}}|^2}}$')

    plt.title('Frequency lattice contribution of this model')
    plt.xlabel(r'$n_1$')
    plt.ylabel(r'$n_2$')
    plt.tick_params(direction='out')


def _fourier_modes_to_torus_field(mode_dict: dict, K1: int, K2: int, N1: int | None = None, N2: int | None = None) -> tuple:
    if N1 is None:
        N1 = _next_pow2(4 * K1 + 1)
    if N2 is None:
        N2 = _next_pow2(4 * K2 + 1)

    theta_1_vals = 2.0 * np.pi * np.arange(N1) / N1
    theta_2_vals = 2.0 * np.pi * np.arange(N2) / N2

    coeff = np.zeros((2 * K1 + 1, 2 * K2 + 1, N_HILBERT, N_HILBERT), dtype=complex)
    for (k1, k2), mat in mode_dict.items():
        coeff[k1 + K1, k2 + K2] = mat

    k1_vals = np.arange(-K1, K1 + 1)
    k2_vals = np.arange(-K2, K2 + 1)
    phase_1 = np.exp(1j * np.outer(theta_1_vals, k1_vals))
    phase_2 = np.exp(1j * np.outer(theta_2_vals, k2_vals))
    field_theta = np.einsum('ia,jb,abmn->ijmn', phase_1, phase_2, coeff, optimize=True)
    return theta_1_vals, theta_2_vals, field_theta


def _partial_trace_qubit_B(field_ab: np.ndarray) -> np.ndarray:
    reshaped = field_ab.reshape(field_ab.shape[0], field_ab.shape[1], 2, 2, 2, 2)
    return np.trace(reshaped, axis1=3, axis2=5)


def _project_to_sigma_A_vector(field_a: np.ndarray) -> np.ndarray:
    vec = np.einsum('ijab,kba->ijk', field_a, PAULI, optimize=True)
    return np.real_if_close(vec).real


def bloch_to_color_continuous(n_batch: np.ndarray) -> np.ndarray:
    """Continuous injective mapping: S^2 -> RGB using HSL colorspace."""
    n = np.atleast_2d(n_batch).copy()
    norms = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.clip(norms, 1e-10, None)

    nx, ny, nz = n[:, 0], n[:, 1], n[:, 2]

    phi = np.arctan2(ny, nx)
    H = (phi + np.pi) / (2 * np.pi)

    L = np.clip(0.5 + np.arcsin(nz) / np.pi, 0, 1)

    S = 1.0

    m2 = np.where(L <= 0.5, L * (1.0 + S), L + S - L * S)
    m1 = 2.0 * L - m2

    def v_component(hue):
        hue = hue % 1.0
        return np.where(hue < 1.0 / 6.0, m1 + (m2 - m1) * hue * 6.0,
               np.where(hue < 0.5, m2,
               np.where(hue < 2.0 / 3.0, m1 + (m2 - m1) * (2.0 / 3.0 - hue) * 6.0, m1)))

    R = np.where(S == 0, L, v_component(H + 1.0 / 3.0))
    G = np.where(S == 0, L, v_component(H))
    B = np.where(S == 0, L, v_component(H - 1.0 / 3.0))

    return np.stack([R, G, B], axis=1)


result, meta = load_solver_result(Path(r"/home/chye/Documents/K1_50.npz"))
print(f"file: {meta['file_path']}")
print(f"K=({meta['K1']},{meta['K2']}), d={meta['d']}, modes={len(meta['full_grid'])}")
print(
    f"converged={result.converged}, iters={result.n_iter}, "
    f"residual={result.residual:.3e}, cv={result.constraint_violation:.3e}"
)
Delta = 0.0
m = 3.0
epsilon = 0.0431
omega = 0.032145710251 * OMEGA_GOLDENRATIO / np.linalg.norm(OMEGA_GOLDENRATIO)
AEanal = _make_AE(meta['K1'], meta['K2'], meta['d'], Delta, m, epsilon, omega)

abs_diff = _diff_norm2(AEanal, result.X, meta['full_grid'])**0.5
print(f"Absolute Frobenius error ||AE_analytic - AE|| = {abs_diff:.3e}")
rel_diff = _rel_diff(AEanal, result.X, meta['full_grid'])
print(f"Relative Frobenius error ||AE_analytic - AE|| / ||AE_analytic|| = {rel_diff:.3e}")




print("=" * 60)
print("Pythagorean identity  ||H||² = ||X*||² + ||H-X*||²")
H = _make_H(meta['K1'], meta['K2'], meta['d'], Delta, m, epsilon, omega)[0]
nH2   = _norm2(H, meta['full_grid'])
nX2   = _norm2(result.X, meta['full_grid'])
nHX2  = _diff_norm2(H, result.X, meta['full_grid'])
rel   = abs(nH2 - nX2 - nHX2) / nH2

print(f"  ||H||²     = {nH2:.8f}")
print(f"  ||X*||²    = {nX2:.8f}")
print(f"  ||H-X*||²  = {nHX2:.8f}")
print(f"  rel error  = {rel:.3e}   (must be ~0)")



AEanal_lattice = _mode_dict_to_lattice_array(AEanal, meta['K1'], meta['K2'])
resultX_lattice = _mode_dict_to_lattice_array(result.X, meta['K1'], meta['K2'])
full_grid, _, _, fg_idx = _build_modes(meta['K1'], meta['K2'])
my_phi = {k: result.phi_flat[fg_idx[k]*N_HILBERT**2:(fg_idx[k]+1)*N_HILBERT**2].reshape(N_HILBERT, N_HILBERT) for k in full_grid}
resultPhi_lattice = _mode_dict_to_lattice_array(my_phi, meta['K1'], meta['K2'])
plot_array_circles(AEanal_lattice, mode='log')
plt.title('Frequency lattice contribution of AEanal')
# save under name "AEanal.png"
plt.savefig(f"AEanal_{meta['K1']}.png")
plot_array_circles(resultX_lattice, mode='log')
plt.title('Frequency lattice contribution of result.AE')
plt.savefig(f"variationalAE_{meta['K1']}.png")
plot_array_circles(resultPhi_lattice, mode='log')
plt.title('Frequency lattice contribution of result.phi (constraint field)')
plt.savefig(f"constraintPhi_{meta['K1']}.png")

theta_1_vals, theta_2_vals, resultX_theta = _fourier_modes_to_torus_field(result.X, meta['K1'], meta['K2'])
resultX_A_theta = _partial_trace_qubit_B(resultX_theta)
resultX_A_vec = _project_to_sigma_A_vector(resultX_A_theta)

resultX_A_norm = np.linalg.norm(resultX_A_vec, axis=-1)
resultX_A_vec_unit = resultX_A_vec / np.clip(resultX_A_norm[..., None], 1e-10, None)

plt.figure(figsize=(8, 6))
im_norm = plt.imshow(
    resultX_A_norm.T,
    origin='lower',
    extent=[theta_1_vals[0], theta_1_vals[-1], theta_2_vals[0], theta_2_vals[-1]],
    cmap='magma',
    aspect='auto',
)
plt.colorbar(im_norm, orientation='vertical', label=r'$\|\vec{v}_A(\theta_1,\theta_2)\|$')
plt.xlabel(r'$\theta_1$')
plt.ylabel(r'$\theta_2$')
plt.title(r'Norm of projected vector from result.X on torus')
plt.savefig(f"ae_{meta['K1']}.png")

resultX_A_rgb = bloch_to_color_continuous(resultX_A_vec_unit.reshape(-1, 3)).reshape(resultX_A_vec_unit.shape)
plt.figure(figsize=(8, 6))
plt.imshow(
    np.transpose(resultX_A_rgb, (1, 0, 2)),
    origin='lower',
    extent=[theta_1_vals[0], theta_1_vals[-1], theta_2_vals[0], theta_2_vals[-1]],
    aspect='auto',
)
plt.xlabel(r'$\theta_1$')
plt.ylabel(r'$\theta_2$')
plt.title(r'Normalized projected Bloch vector mapped to RGB on torus')
plt.savefig(f"Projector_{meta['K1']}.png")
plt.show()



if __name__ == "__main__":
    # main()
    print("Done")
    