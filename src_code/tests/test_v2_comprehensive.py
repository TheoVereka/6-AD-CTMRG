"""
Comprehensive test suite for core_unres_v2.py
Run: python tests/test_v2_comprehensive.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import core_unres_v2 as cv2

torch.set_default_dtype(torch.float64)
cv2.set_dtype(True)


def _neel_tensors(D=1, d=2):
    up = torch.zeros(D, D, D, d); up[0, 0, 0, 0] = 1.0
    dn = torch.zeros(D, D, D, d); dn[0, 0, 0, 1] = 1.0
    return up.clone(), dn.clone(), up.clone(), dn.clone(), up.clone(), dn.clone()


def _fm_tensors(D=1, d=2):
    up = torch.zeros(D, D, D, d); up[0, 0, 0, 0] = 1.0
    return tuple(up.clone() for _ in range(6))


# ═══════════════════════════════════════════════════════════════════════
# 1. Hamiltonian
# ═══════════════════════════════════════════════════════════════════════

def test_hamiltonian():
    H = cv2.build_heisenberg_H(J=1.0)
    assert H.dtype == torch.float64
    assert not H.is_complex()
    eigvals = torch.linalg.eigvalsh(H.reshape(4, 4))
    exp = torch.tensor([-0.75, 0.25, 0.25, 0.25])
    assert (eigvals - exp).abs().max() < 1e-14
    # Symmetry: H[i,k,j,l] = H[j,l,i,k] (site exchange)
    assert (H - H.permute(2, 3, 0, 1)).abs().max() < 1e-15
    print("  ✓ eigenvalues correct, symmetric under site exchange")


# ═══════════════════════════════════════════════════════════════════════
# 2. Néel exact energy at D=1 and D=2
# ═══════════════════════════════════════════════════════════════════════

def test_neel_D1():
    a, b, c, d, e, f = _neel_tensors(D=1)
    H = cv2.build_heisenberg_H()
    A, B, C, D2, E, F = cv2.abcdef_to_ABCDEF(a, b, c, d, e, f, 1)
    env1, env2, env3, s = cv2.ctmrg(A, B, C, D2, E, F, 1, 1, max_steps=5)
    E_val = cv2.compute_total_energy(a, b, c, d, e, f, H, 1, 1, env1, env2, env3).item()
    assert abs(E_val - (-0.25)) < 1e-12, f"Néel@D=1: {E_val}"
    print(f"  ✓ Néel@D=1 chi=1: E/bond={E_val:.10f}")


def test_neel_D2():
    a, b, c, d, e, f = _neel_tensors(D=2)
    H = cv2.build_heisenberg_H()
    A, B, C, D2, E, F = cv2.abcdef_to_ABCDEF(a, b, c, d, e, f, 4)
    env1, env2, env3, s = cv2.ctmrg(A, B, C, D2, E, F, 8, 4, max_steps=30)
    E_val = cv2.compute_total_energy(a, b, c, d, e, f, H, 8, 2, env1, env2, env3).item()
    assert abs(E_val - (-0.25)) < 1e-10, f"Néel@D=2: {E_val}"
    print(f"  ✓ Néel@D=2 chi=8: E/bond={E_val:.10f}")


# ═══════════════════════════════════════════════════════════════════════
# 3. FM exact energy at D=1 and D=2
# ═══════════════════════════════════════════════════════════════════════

def test_fm_D1():
    a, b, c, d, e, f = _fm_tensors(D=1)
    H = cv2.build_heisenberg_H()
    A, B, C, D2, E, F = cv2.abcdef_to_ABCDEF(a, b, c, d, e, f, 1)
    env1, env2, env3, s = cv2.ctmrg(A, B, C, D2, E, F, 1, 1, max_steps=5)
    E_val = cv2.compute_total_energy(a, b, c, d, e, f, H, 1, 1, env1, env2, env3).item()
    assert abs(E_val - 0.25) < 1e-12, f"FM@D=1: {E_val}"
    print(f"  ✓ FM@D=1 chi=1: E/bond={E_val:.10f}")


# ═══════════════════════════════════════════════════════════════════════
# 4. All tensors real (float64) throughout pipeline
# ═══════════════════════════════════════════════════════════════════════

def test_dtype():
    torch.manual_seed(99)
    D, D_sq, chi = 2, 4, 4
    a, b, c, d, e, f = cv2.initialize_abcdef(D)
    for t in (a, b, c, d, e, f):
        assert t.dtype == torch.float64 and t.is_floating_point()
    A, B, C, D2, E, F = cv2.abcdef_to_ABCDEF(a, b, c, d, e, f, D_sq)
    for t in (A, B, C, D2, E, F):
        assert t.dtype == torch.float64
    env1, env2, env3, _ = cv2.ctmrg(A, B, C, D2, E, F, chi, D_sq, max_steps=20)
    for env in (env1, env2, env3):
        for t in env:
            assert t.dtype == torch.float64 and torch.isfinite(t).all()
    print("  ✓ all tensors real float64, finite")


# ═══════════════════════════════════════════════════════════════════════
# 5. CTMRG convergence for random D=2 tensors
# ═══════════════════════════════════════════════════════════════════════

def test_ctmrg_convergence():
    torch.manual_seed(77)
    D, D_sq, chi = 2, 4, 8
    a, b, c, d, e, f = cv2.initialize_abcdef(D)
    A, B, C, D2, E, F = cv2.abcdef_to_ABCDEF(a, b, c, d, e, f, D_sq)
    env1, env2, env3, steps = cv2.ctmrg(A, B, C, D2, E, F, chi, D_sq, max_steps=80)
    print(f"  CTM converged in {steps} steps")
    assert steps < 80, f"CTMRG not converged in 80 steps"
    H = cv2.build_heisenberg_H()
    E_val = cv2.compute_total_energy(a, b, c, d, e, f, H, chi, D, env1, env2, env3).item()
    assert abs(E_val) < 1.0, f"E/bond={E_val} unreasonable"
    print(f"  ✓ converged in {steps} steps, E/bond={E_val:.6f}")


# ═══════════════════════════════════════════════════════════════════════
# 6. Cross-validation against original code
# ═══════════════════════════════════════════════════════════════════════

def test_cross_validate():
    try:
        import core_unrestricted as orig
    except ImportError:
        print("  ⊘ SKIPPED (original module not available)")
        return

    D, D_sq, chi = 2, 4, 8
    torch.manual_seed(42)
    a, b, c, d, e, f = cv2.initialize_abcdef(D)
    H_mine = cv2.build_heisenberg_H()
    H_orig = H_mine.permute(2, 0, 3, 1).to(torch.complex128)

    A, B, C, D2, E, F = cv2.abcdef_to_ABCDEF(
        a.detach(), b.detach(), c.detach(), d.detach(), e.detach(), f.detach(), D_sq)
    env1, env2, env3, _ = cv2.ctmrg(A, B, C, D2, E, F, chi, D_sq, max_steps=50)

    ac, bc, cc, dc, ec, fc = [t.to(torch.complex128) for t in [a, b, c, d, e, f]]

    for name, mine_fn, orig_fn, env in [
        ("env1", cv2.energy_env1, orig.energy_expectation_nearest_neighbor_3ebadcf_bonds, env1),
        ("env2", cv2.energy_env2, orig.energy_expectation_nearest_neighbor_3afcbed_bonds, env2),
        ("env3", cv2.energy_env3, orig.energy_expectation_nearest_neighbor_other_3_bonds, env3),
    ]:
        e_mine = mine_fn(a, b, c, d, e, f, H_mine, chi, D, env).item()
        env_c = [t.to(torch.complex128) for t in env]
        e_orig = orig_fn(ac, bc, cc, dc, ec, fc, H_orig, H_orig, H_orig, chi, D, *env_c).real.item()
        diff = abs(e_mine - e_orig)
        assert diff < 1e-12, f"{name}: diff={diff}"
        print(f"  {name}: mine={e_mine:.12f}  orig={e_orig:.12f}  diff={diff:.2e}")
    print("  ✓ all 3 energy functions match original to machine precision")


# ═══════════════════════════════════════════════════════════════════════
# 7. Gradient flow (torch.autograd works)
# ═══════════════════════════════════════════════════════════════════════

def test_gradient_flow():
    torch.manual_seed(55)
    D, D_sq, chi = 2, 4, 4
    a, b, c, d, e, f = cv2.initialize_abcdef(D)
    for t in (a, b, c, d, e, f):
        t.requires_grad_(True)
    H = cv2.build_heisenberg_H()

    A, B, C, D2, E, F = cv2.abcdef_to_ABCDEF(
        a.detach(), b.detach(), c.detach(), d.detach(), e.detach(), f.detach(), D_sq)
    with torch.no_grad():
        env1, env2, env3, _ = cv2.ctmrg(A, B, C, D2, E, F, chi, D_sq, max_steps=30)

    loss = (
        cv2.energy_env1(a, b, c, d, e, f, H, chi, D, env1)
        + cv2.energy_env2(a, b, c, d, e, f, H, chi, D, env2)
        + cv2.energy_env3(a, b, c, d, e, f, H, chi, D, env3)
    )
    loss.backward()

    for name, t in [('a', a), ('b', b), ('c', c), ('d', d), ('e', e), ('f', f)]:
        assert t.grad is not None, f"{name}.grad is None"
        assert torch.isfinite(t.grad).all(), f"{name}.grad has non-finite values"
        assert t.grad.abs().max() > 0, f"{name}.grad is all zero"
    print(f"  ✓ all 6 tensors have finite, nonzero gradients. loss={loss.item():.6f}")


# ═══════════════════════════════════════════════════════════════════════
# 8. Multi-seed energy consistency
# ═══════════════════════════════════════════════════════════════════════

def test_multiseed_consistency():
    D, D_sq, chi = 2, 4, 8
    H = cv2.build_heisenberg_H()
    energies = []
    for seed in [10, 20, 30, 40, 50]:
        torch.manual_seed(seed)
        a, b, c, d, e, f = cv2.initialize_abcdef(D)
        A, B, C, D2, E, F = cv2.abcdef_to_ABCDEF(a, b, c, d, e, f, D_sq)
        env1, env2, env3, _ = cv2.ctmrg(A, B, C, D2, E, F, chi, D_sq, max_steps=50)
        E_val = cv2.compute_total_energy(a, b, c, d, e, f, H, chi, D, env1, env2, env3).item()
        energies.append(E_val)
    spread = max(energies) - min(energies)
    mean_e = sum(energies) / len(energies)
    print(f"  energies: {[f'{e:.6f}' for e in energies]}")
    print(f"  mean={mean_e:.6f}, spread={spread:.6f}")
    # Random state energies should all be in a reasonable range
    assert all(abs(e) < 0.5 for e in energies), f"unreasonable energy in {energies}"
    print(f"  ✓ all 5 seeds give reasonable energies")


# ═══════════════════════════════════════════════════════════════════════
# Run all
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    tests = [
        ("1. Hamiltonian", test_hamiltonian),
        ("2a. Néel D=1", test_neel_D1),
        ("2b. Néel D=2", test_neel_D2),
        ("3. FM D=1", test_fm_D1),
        ("4. All real dtype", test_dtype),
        ("5. CTMRG convergence", test_ctmrg_convergence),
        ("6. Cross-validate vs original", test_cross_validate),
        ("7. Gradient flow", test_gradient_flow),
        ("8. Multi-seed consistency", test_multiseed_consistency),
    ]

    passed = failed = 0
    for name, fn in tests:
        print(f"\n{'─'*60}\nTEST: {name}\n{'─'*60}")
        try:
            fn()
            passed += 1
        except Exception as ex:
            print(f"  ✗ FAILED: {ex}")
            import traceback; traceback.print_exc()
            failed += 1

    print(f"\n{'═'*60}")
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(tests)}")
    print(f"{'═'*60}")
