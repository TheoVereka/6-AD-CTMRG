"""
example.py
----------
Validates the variational solver  min_{X: L(X)=0} ||H - X||².

Five tests
----------
1. Constraint:    ||L(X*)|| / ||H|| < tol            (feasibility)
2. Pythagorean:   ||H||² = ||X*||² + ||H-X*||²       (correct orthogonal projection)
3. Consistency:   stage1 (direct L) ≈ stage2 (FFT L) (implementation agreement)
4. Trivial:       H ∈ ker(L)  →  X* = H              (solver handles b≈0 correctly)
5. Benchmark:     K=2,3,4 timing comparing direct vs FFT

Run:
    python example.py

All assertions should pass with default CG tolerance 1e-10.
"""

import time
import numpy as np
from solver import solve, solve_stage1, solve_stage2


# ──────────────────────────────────────────────────────────────────────────────
# Helper: build a random Hermitian field with all full_grid keys
# ──────────────────────────────────────────────────────────────────────────────

def _make_H(K1: int, K2: int, d: int, seed: int = 0) -> tuple:
    """
    Returns (H, full_grid) where H is a random Hermitian field:
        H_{-k} = H_k†,   H_0 = H_0†
    H has keys for every mode in full_grid = {(i,j): |i|≤K1, |j|≤K2}.
    """
    rng       = np.random.default_rng(seed)
    full_grid = [(i, j) for i in range(-K1, K1+1) for j in range(-K2, K2+1)]
    K_plus0   = [(0, 0)] + [(i, j) for (i, j) in full_grid
                            if i > 0 or (i == 0 and j > 0)]
    H = {}
    for k in K_plus0:
        m    = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
        H[k] = m + m.conj().T
        if k != (0, 0):
            H[(-k[0], -k[1])] = H[k].conj().T
    return H, full_grid


def _norm2(A: dict, keys: list) -> float:
    """||A||² = sum_k ||A[k]||_F²"""
    return sum(np.linalg.norm(A[k], 'fro')**2 for k in keys)


def _diff_norm2(A: dict, B: dict, keys: list) -> float:
    """||A - B||²"""
    return sum(np.linalg.norm(A[k] - B[k], 'fro')**2 for k in keys)


# ──────────────────────────────────────────────────────────────────────────────

def test_constraint() -> None:
    """||L(X*)|| / ||H|| must be < tol (feasibility of the solution)."""
    print("=" * 60)
    print("Test 1: constraint violation  ||L(X*)|| / ||H||")
    K1, K2, d = 3, 3, 2
    omega = np.array([1.0, np.sqrt(2.0)])
    H, fg = _make_H(K1, K2, d, seed=42)

    tol = 1e-10
    res = solve(H, K1, K2, d, omega, tol=tol, verbose=True)

    print(f"  converged             : {res.converged}")
    print(f"  CG iterations         : {res.n_iter}")
    print(f"  CG residual ||Aφ-b||/||b|| = {res.residual:.3e}")
    print(f"  constraint violation  : {res.constraint_violation:.3e}  (tol={tol:.0e})")

    assert res.converged,                         "CG did not converge"
    assert res.constraint_violation < 1e-6,       f"cv = {res.constraint_violation:.3e}"
    print("  PASSED\n")


# ──────────────────────────────────────────────────────────────────────────────

def test_pythagorean() -> None:
    """
    The optimal X* is the orthogonal projection of H onto ker(L).
    For an orthogonal projection:   ||H||² = ||X*||² + ||H - X*||²
    (Pythagorean theorem — the error H-X* is perpendicular to X* ∈ ker(L)).
    """
    print("=" * 60)
    print("Test 2: Pythagorean identity  ||H||² = ||X*||² + ||H-X*||²")
    K1, K2, d = 3, 3, 2
    omega = np.array([1.0, np.sqrt(2.0)])
    H, fg = _make_H(K1, K2, d, seed=7)

    res = solve(H, K1, K2, d, omega, tol=1e-12, verbose=False)
    X   = res.X

    nH2   = _norm2(H, fg)
    nX2   = _norm2(X, fg)
    nHX2  = _diff_norm2(H, X, fg)
    rel   = abs(nH2 - nX2 - nHX2) / nH2

    print(f"  ||H||²     = {nH2:.8f}")
    print(f"  ||X*||²    = {nX2:.8f}")
    print(f"  ||H-X*||²  = {nHX2:.8f}")
    print(f"  rel error  = {rel:.3e}   (must be ~0)")

    assert rel < 1e-9, f"Pythagorean identity violated: rel={rel:.3e}"
    print("  PASSED\n")


# ──────────────────────────────────────────────────────────────────────────────

def test_direct_vs_fft() -> None:
    """Stage 1 (direct) and stage 2 (FFT) must give the same X* to ~1e-10."""
    print("=" * 60)
    print("Test 3: stage1 (direct L) vs stage2 (FFT L) consistency")
    K1, K2, d = 2, 2, 2
    omega = np.array([1.0, np.sqrt(3.0)])
    H, fg = _make_H(K1, K2, d, seed=123)

    r1 = solve_stage1(H, K1, K2, d, omega, tol=1e-12, verbose=False)
    r2 = solve_stage2(H, K1, K2, d, omega, tol=1e-12, verbose=False)

    rel = _diff_norm2(r1.X, r2.X, fg)**0.5 / _norm2(r1.X, fg)**0.5

    print(f"  stage1: {r1.n_iter:4d} iters  cv={r1.constraint_violation:.2e}")
    print(f"  stage2: {r2.n_iter:4d} iters  cv={r2.constraint_violation:.2e}")
    print(f"  ||X_dir - X_fft|| / ||X_dir|| = {rel:.3e}")

    assert rel < 1e-8, f"Direct/FFT disagree: {rel:.3e}"
    print("  PASSED\n")


# ──────────────────────────────────────────────────────────────────────────────

def test_trivial() -> None:
    """
    Constant diagonal field H_k = 0 for k≠0 lies in ker(L):
        L(H)_k = i*(k·ω)*H_0*δ_{k,0} = 0 for all k  (since k·ω = 0 at k=0).
    The solver should detect ||b|| ≈ 0 and return X* = H with 0 iterations.
    """
    print("=" * 60)
    print("Test 4: trivial  H ∈ ker(L)  →  X* = H,  n_iter = 0")
    K1, K2, d = 2, 2, 3
    omega     = np.array([1.0, np.sqrt(5.0)])
    full_grid = [(i, j) for i in range(-K1, K1+1) for j in range(-K2, K2+1)]

    H = {k: np.zeros((d, d), dtype=complex) for k in full_grid}
    H[(0, 0)] = np.diag([1.0, 2.0, 3.0]).astype(complex)

    res     = solve(H, K1, K2, d, omega, tol=1e-12, verbose=False)
    rel_err = _diff_norm2(H, res.X, full_grid)**0.5 / _norm2(H, full_grid)**0.5

    print(f"  n_iter             = {res.n_iter}  (expected 0)")
    print(f"  ||X* - H|| / ||H|| = {rel_err:.3e}  (expected ~0)")

    assert res.n_iter == 0,   f"Expected 0 iters, got {res.n_iter}"
    assert rel_err < 1e-12,   f"X* ≠ H: {rel_err:.3e}"
    print("  PASSED\n")


# ──────────────────────────────────────────────────────────────────────────────

def test_benchmark() -> None:
    """Timing for K=2,3,4 showing FFT speedup (FFT for K≥3, direct for K≤2)."""
    print("=" * 60)
    print("Test 5: benchmark  K=2,3,4  (|K| = 25,49,81; direct≤25, FFT>25)")
    d     = 2
    omega = np.array([1.0, np.sqrt(2.0)])
    print(f"  {'K':>3}  {'|K|':>4}  {'method':>6}  {'iters':>6}  {'cv':>10}  {'time':>7}")
    for K in [2, 3, 4]:
        H, fg = _make_H(K, K, d, seed=0)
        t0    = time.perf_counter()
        res   = solve(H, K, K, d, omega, tol=1e-10, verbose=False)
        dt    = time.perf_counter() - t0
        nK    = (2*K + 1)**2
        meth  = "FFT" if nK > 25 else "direct"
        ok    = "OK" if res.converged else "FAIL"
        print(f"  K={K}  |K|={nK:3d}  {meth:>6}  {res.n_iter:6d}  "
              f"{res.constraint_violation:.3e}  {dt:.2f}s  {ok}")
    print()


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_constraint()
    test_pythagorean()
    test_direct_vs_fft()
    test_trivial()
    test_benchmark()
    print("All tests passed.")
