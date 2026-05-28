"""
example.py
----------
Demonstrates and validates the variational solver.

Tests
-----
1. Consistency:  ||L(X)|| / ||H|| should be ~tol (CG tolerance)
2. Minimality:   ||H-X|| <= ||H-Y|| for random Hermitian Y with L(Y)=0
3. Direct vs FFT: results must agree when both are used
4. Trivial case: H already in kernel of L → X = H exactly

Run:
    python example.py
"""

import numpy as np
from field_utils import random_hermitian_field, dict_to_flat, flat_to_dict
from mode_sets import build_mode_sets
from operator_L import OperatorL
from solver import solve_variational


def make_random_H(K1: int, K2: int, d: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return random_hermitian_field(K1, K2, d, rng)


# ---------------------------------------------------------------------------
def test_constraint_violation() -> None:
    print("=" * 60)
    print("Test 1: constraint violation ||L(X)|| / ||H||")
    K1, K2, d = 3, 3, 2
    omega = np.array([1.0, np.sqrt(2.0)])
    H = make_random_H(K1, K2, d, seed=42)

    tol = 1e-10
    result = solve_variational(H, K1, K2, d, omega, tol=tol, verbose=True)

    print(f"  converged          : {result.converged}")
    print(f"  CG iterations      : {result.n_iter}")
    print(f"  CG residual        : {result.residual:.3e}")
    print(f"  constraint violation: {result.constraint_violation:.3e}")
    print(f"  (tol was {tol:.0e})")
    assert result.converged, "CG did not converge!"
    assert result.constraint_violation < 1e-6, (
        f"Constraint violated: {result.constraint_violation:.3e}")
    print("  PASSED\n")


# ---------------------------------------------------------------------------
def test_minimality() -> None:
    """
    Check that ||H - X||^2 <= ||H - Y||^2 for several random Y that satisfy
    L(Y) = 0 (constructed by starting from H and subtracting L(phi) for
    random phi).
    """
    print("=" * 60)
    print("Test 2: minimality  ||H-X|| <= ||H-Y|| for L(Y)=0")
    K1, K2, d = 3, 3, 2
    omega = np.array([1.0, np.sqrt(2.0)])
    H = make_random_H(K1, K2, d, seed=7)

    result = solve_variational(H, K1, K2, d, omega, tol=1e-12, verbose=False)
    full_grid, _, K_plus0, _ = build_mode_sets(K1, K2)
    L_op = OperatorL(H, K1, K2, d, omega)

    h_flat    = dict_to_flat(H, K_plus0)
    x_flat    = dict_to_flat(result.X_unscaled, K_plus0)
    dist_X    = float(np.linalg.norm(h_flat - x_flat))

    rng = np.random.default_rng(99)
    all_ok = True
    for trial in range(5):
        # random Hermitian phi
        phi_rand = random_hermitian_field(K1, K2, d, rng)
        phi_flat = dict_to_flat(phi_rand, K_plus0)
        # Y = H - L(phi_rand)  satisfies L(Y)=0 (approximately, up to numerics)
        Y_flat = h_flat - L_op.apply(phi_flat)
        dist_Y = float(np.linalg.norm(h_flat - Y_flat))
        ok = dist_X <= dist_Y + 1e-10
        all_ok = all_ok and ok
        print(f"  trial {trial}: ||H-X||={dist_X:.6f}  ||H-Y||={dist_Y:.6f}  "
              f"{'OK' if ok else 'FAIL'}")

    assert all_ok, "Minimality violated!"
    print("  PASSED\n")


# ---------------------------------------------------------------------------
def test_direct_vs_fft() -> None:
    print("=" * 60)
    print("Test 3: direct vs FFT consistency")
    K1, K2, d = 2, 2, 2   # small enough for direct to be fast
    omega = np.array([1.0, np.sqrt(3.0)])
    H = make_random_H(K1, K2, d, seed=123)

    r_dir = solve_variational(H, K1, K2, d, omega, tol=1e-11,
                              force_direct=True, verbose=False)
    r_fft = solve_variational(H, K1, K2, d, omega, tol=1e-11,
                              force_fft=True, verbose=False)

    full_grid, _, K_plus0, _ = build_mode_sets(K1, K2)
    x_dir = dict_to_flat(r_dir.X_unscaled, K_plus0)
    x_fft = dict_to_flat(r_fft.X_unscaled, K_plus0)
    diff  = float(np.linalg.norm(x_dir - x_fft)) / float(np.linalg.norm(x_dir))

    print(f"  direct converged : {r_dir.converged}  res={r_dir.residual:.2e}")
    print(f"  fft    converged : {r_fft.converged}  res={r_fft.residual:.2e}")
    print(f"  rel ||X_dir - X_fft|| = {diff:.3e}")
    assert diff < 1e-6, f"Direct/FFT disagree: {diff:.3e}"
    print("  PASSED\n")


# ---------------------------------------------------------------------------
def test_trivial() -> None:
    """H already in kernel of L (e.g., H=const diagonal)."""
    print("=" * 60)
    print("Test 4: trivial case  H in ker(L) → X = H")
    K1, K2, d = 2, 2, 3
    omega = np.array([1.0, np.sqrt(5.0)])

    # Constant Hermitian field: only k=(0,0) mode, rest zero
    full_grid, _, K_plus0, _ = build_mode_sets(K1, K2)
    H = {}
    for k in full_grid:
        if k == (0, 0):
            A = np.diag([1.0, 2.0, 3.0]).astype(complex)
        else:
            A = np.zeros((d, d), dtype=complex)
        H[k] = A

    result = solve_variational(H, K1, K2, d, omega, verbose=False)
    h_flat = dict_to_flat(H, K_plus0)
    x_flat = dict_to_flat(result.X_unscaled, K_plus0)
    err    = float(np.linalg.norm(x_flat - h_flat)) / float(np.linalg.norm(h_flat))
    print(f"  ||X - H|| / ||H|| = {err:.3e}  (should be ~0)")
    assert err < 1e-10, f"Trivial case failed: X ≠ H, err={err:.3e}"
    print("  PASSED\n")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_constraint_violation()
    test_minimality()
    test_direct_vs_fft()
    test_trivial()
    print("All tests passed.")
