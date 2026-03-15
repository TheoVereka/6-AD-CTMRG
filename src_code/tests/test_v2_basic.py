"""
Tests for core_unres_v2.py — basic correctness checks.
Run: python -m pytest tests/test_v2_basic.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import core_unres_v2 as cv2

torch.set_default_dtype(torch.float64)
cv2.set_dtype(True)

# ── helpers ──────────────────────────────────────────────────────────────────

def _make_neel_tensors(D=1, d=2):
    """Néel state: sublattice A = spin-up, sublattice B = spin-down.
    
    Honeycomb has 2 sublattices. For the 6-site cell (a,b,c,d,e,f):
    - Sites a, c, e are sublattice A (spin up)
    - Sites b, d, f are sublattice B (spin down)
    
    Each tensor is (D, D, D, d) with D=1 and physical = spin up or down.
    """
    up = torch.zeros(D, D, D, d, dtype=torch.float64)
    up[0, 0, 0, 0] = 1.0
    down = torch.zeros(D, D, D, d, dtype=torch.float64)
    down[0, 0, 0, 1] = 1.0
    # a=up, b=down, c=up, d=down, e=up, f=down
    return up.clone(), down.clone(), up.clone(), down.clone(), up.clone(), down.clone()


def _make_fm_tensors(D=1, d=2):
    """Ferromagnetic state: all spins up."""
    up = torch.zeros(D, D, D, d, dtype=torch.float64)
    up[0, 0, 0, 0] = 1.0
    return tuple(up.clone() for _ in range(6))


# ══════════════════════════════════════════════════════════════════════════════
# Test 1: Hamiltonian eigenvalues
# ══════════════════════════════════════════════════════════════════════════════

def test_hamiltonian_eigenvalues():
    H = cv2.build_heisenberg_H(J=1.0)
    assert H.dtype == torch.float64, f"H dtype = {H.dtype}"
    assert not H.is_complex(), "H must be real"
    
    # Reshape to (4, 4) and check eigenvalues
    H_mat = H.reshape(4, 4)
    eigvals = torch.linalg.eigvalsh(H_mat)
    expected = torch.tensor([-3/4, 1/4, 1/4, 1/4], dtype=torch.float64)
    err = (eigvals - expected).abs().max().item()
    print(f"  H eigenvalues: {eigvals.tolist()}")
    print(f"  Expected:      {expected.tolist()}")
    print(f"  Max error:     {err:.2e}")
    assert err < 1e-14, f"H eigenvalue error {err}"


# ══════════════════════════════════════════════════════════════════════════════
# Test 2: Néel state energy = -0.25 per bond
# ══════════════════════════════════════════════════════════════════════════════

def test_neel_energy():
    """For the Néel state on honeycomb, each NN bond has <S·S> = -1/4.
    So E/bond = -0.25, total for 9 bonds = -2.25."""
    D = 1
    d = 2
    D_sq = D ** 2
    chi = 1
    a, b, c, dt, e, f = _make_neel_tensors(D, d)
    H = cv2.build_heisenberg_H(J=1.0)
    
    # Build double-layer
    A, B, C, DD, E, F = cv2.abcdef_to_ABCDEF(a, b, c, dt, e, f, D_sq)
    
    # Run CTMRG
    env1, env2, env3, ctm_steps = cv2.ctmrg(A, B, C, DD, E, F, chi, D_sq,
                                              max_steps=10, conv_thr=1e-10)
    print(f"  CTMRG converged in {ctm_steps} steps")
    
    # Compute energy
    e_per_bond = cv2.compute_total_energy(a, b, c, dt, e, f, H, chi, D, env1, env2, env3)
    e_val = e_per_bond.item()
    print(f"  E/bond = {e_val:.10f}")
    print(f"  Expected: -0.25")
    err = abs(e_val - (-0.25))
    print(f"  Error: {err:.2e}")
    assert err < 1e-10, f"Néel E/bond = {e_val}, expected -0.25, error {err}"


# ══════════════════════════════════════════════════════════════════════════════
# Test 3: FM state energy = +0.25 per bond
# ══════════════════════════════════════════════════════════════════════════════

def test_fm_energy():
    """For the FM state, each NN bond has <S·S> = +1/4.
    So E/bond = +0.25, total for 9 bonds = +2.25."""
    D = 1
    d = 2
    D_sq = D ** 2
    chi = 1
    a, b, c, dt, e, f = _make_fm_tensors(D, d)
    H = cv2.build_heisenberg_H(J=1.0)
    
    A, B, C, DD, E, F = cv2.abcdef_to_ABCDEF(a, b, c, dt, e, f, D_sq)
    env1, env2, env3, ctm_steps = cv2.ctmrg(A, B, C, DD, E, F, chi, D_sq,
                                              max_steps=10, conv_thr=1e-10)
    print(f"  CTMRG converged in {ctm_steps} steps")
    
    e_per_bond = cv2.compute_total_energy(a, b, c, dt, e, f, H, chi, D, env1, env2, env3)
    e_val = e_per_bond.item()
    print(f"  E/bond = {e_val:.10f}")
    print(f"  Expected: +0.25")
    err = abs(e_val - 0.25)
    print(f"  Error: {err:.2e}")
    assert err < 1e-10, f"FM E/bond = {e_val}, expected +0.25, error {err}"


# ══════════════════════════════════════════════════════════════════════════════
# Test 4: All tensors are real
# ══════════════════════════════════════════════════════════════════════════════

def test_all_real():
    """Every tensor produced by the CTMRG pipeline should be real float64."""
    D = 2
    d = 2
    D_sq = D ** 2
    chi = 4
    torch.manual_seed(42)
    a, b, c, dt, e, f = cv2.initialize_abcdef(D, d)
    
    for name, t in [('a', a), ('b', b), ('c', c), ('d', dt), ('e', e), ('f', f)]:
        assert t.dtype == torch.float64, f"{name} dtype={t.dtype}"
        assert not t.is_complex(), f"{name} is complex"
    
    A, B, C, DD, E, F = cv2.abcdef_to_ABCDEF(a, b, c, dt, e, f, D_sq)
    for name, t in [('A', A), ('B', B), ('C', C), ('D', DD), ('E', E), ('F', F)]:
        assert t.dtype == torch.float64, f"{name} dtype={t.dtype}"
        assert not t.is_complex(), f"{name} is complex"
    
    env1, env2, env3, ctm_steps = cv2.ctmrg(A, B, C, DD, E, F, chi, D_sq,
                                              max_steps=30, conv_thr=1e-7)
    
    for env_name, env in [('env1', env1), ('env2', env2), ('env3', env3)]:
        for i, t in enumerate(env):
            assert t.dtype == torch.float64, f"{env_name}[{i}] dtype={t.dtype}"
            assert not t.is_complex(), f"{env_name}[{i}] is complex"
            assert torch.isfinite(t).all(), f"{env_name}[{i}] has non-finite values"
    
    print(f"  All tensors are real float64. CTM converged in {ctm_steps} steps.")


# ══════════════════════════════════════════════════════════════════════════════
# Test 5: CTMRG convergence for D=2
# ══════════════════════════════════════════════════════════════════════════════

def test_ctmrg_convergence():
    """CTMRG should converge (not hit max_steps) for random D=2 tensors."""
    D = 2
    d = 2
    D_sq = D ** 2
    chi = 8
    max_steps = 80
    torch.manual_seed(123)
    a, b, c, dt, e, f = cv2.initialize_abcdef(D, d)
    A, B, C, DD, E, F = cv2.abcdef_to_ABCDEF(a, b, c, dt, e, f, D_sq)
    
    env1, env2, env3, ctm_steps = cv2.ctmrg(A, B, C, DD, E, F, chi, D_sq,
                                              max_steps=max_steps, conv_thr=1e-7)
    print(f"  CTMRG steps: {ctm_steps} / {max_steps}")
    assert ctm_steps < max_steps, f"CTMRG did not converge in {max_steps} steps"
    
    # Also compute energy to check it's finite
    H = cv2.build_heisenberg_H()
    e_per_bond = cv2.compute_total_energy(a, b, c, dt, e, f, H, chi, D, env1, env2, env3)
    e_val = e_per_bond.item()
    print(f"  E/bond = {e_val:.10f}")
    assert abs(e_val) < 1.0, f"E/bond = {e_val}, unreasonably large"


# ══════════════════════════════════════════════════════════════════════════════
# Run all
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    tests = [
        ("Hamiltonian eigenvalues", test_hamiltonian_eigenvalues),
        ("Néel energy", test_neel_energy),
        ("FM energy", test_fm_energy),
        ("All tensors real", test_all_real),
        ("CTMRG convergence", test_ctmrg_convergence),
    ]
    
    passed = 0
    failed = 0
    for name, fn in tests:
        print(f"\n{'='*60}")
        print(f"TEST: {name}")
        print(f"{'='*60}")
        try:
            fn()
            print(f"  ✓ PASSED")
            passed += 1
        except Exception as ex:
            print(f"  ✗ FAILED: {ex}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(tests)}")
    print(f"{'='*60}")
