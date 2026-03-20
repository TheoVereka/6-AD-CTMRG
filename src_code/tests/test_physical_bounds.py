#!/usr/bin/env python3
"""
Test physical bounds on energy expectation values.
The Heisenberg AFM model has known bounds:
  - Minimum (singlet): -0.75 per bond
  - Maximum (triplet): +0.25 per bond

Any energy outside [-0.75, +0.25] per bond is unphysical.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
import opt_einsum as oe
from core_unrestricted import (
    initialize_abcdef, normalize_tensor, abcdef_to_ABCDEF,
    CTMRG_from_init_to_stop,
    energy_expectation_nearest_neighbor_3ebadcf_bonds,
    energy_expectation_nearest_neighbor_3afcbed_bonds,
    energy_expectation_nearest_neighbor_other_3_bonds,
    set_dtype,
)

set_dtype(True)  # float64
CDTYPE = torch.complex128

def build_heisenberg_H(J=1.0):
    """Build S_i · S_j Hamiltonian."""
    sx = torch.tensor([[0, 1], [1, 0]], dtype=CDTYPE) / 2
    sy = torch.tensor([[0, -1j], [1j, 0]], dtype=CDTYPE) / 2
    sz = torch.tensor([[1, 0], [0, -1]], dtype=CDTYPE) / 2
    SdotS = (oe.contract("ij,kl->ikjl", sx, sx)
           + oe.contract("ij,kl->ikjl", sy, sy)
           + oe.contract("ij,kl->ikjl", sz, sz))
    return J * SdotS

def test_random_tensors():
    """Test that random iPEPS tensors give physically reasonable energies."""
    print("=" * 70)
    print("TEST 1: Random tensor initialization")
    print("=" * 70)
    
    D_bond, chi, d_PHYS = 2, 5, 2
    D_sq = D_bond ** 2
    
    # Build Hamiltonian
    H = build_heisenberg_H(J=1.0)
    Hs = [H] * 9
    
    num_trials = 10
    energies = []
    
    for trial in range(num_trials):
        # Random initialization
        a, b, c, d, e, f = initialize_abcdef('random', D_bond, d_PHYS, 1e-3)
        
        # Build environment
        A, B, C, D, E, F = abcdef_to_ABCDEF(a, b, c, d, e, f, D_sq)
        all27 = CTMRG_from_init_to_stop(A, B, C, D, E, F, chi, D_sq, 50, 1e-7)
        
        # Compute energy
        with torch.no_grad():
            E3_1 = energy_expectation_nearest_neighbor_3ebadcf_bonds(
                a, b, c, d, e, f, Hs[0], Hs[1], Hs[2],
                chi, D_bond, *all27[:9])
            E3_2 = energy_expectation_nearest_neighbor_3afcbed_bonds(
                a, b, c, d, e, f, Hs[3], Hs[4], Hs[5],
                chi, D_bond, *all27[9:18])
            E3_3 = energy_expectation_nearest_neighbor_other_3_bonds(
                a, b, c, d, e, f, Hs[6], Hs[7], Hs[8],
                chi, D_bond, *all27[18:27])
            
            E_total = (E3_1 + E3_2 + E3_3).real.item()
        
        E_per_bond = E_total / 9
        energies.append(E_per_bond)
        
        # Check bounds
        lower_bound = -0.75  # singlet ground state of single bond
        upper_bound = +0.25  # triplet state of single bond
        
        status = "OK" if (lower_bound <= E_per_bond <= upper_bound) else "FAIL"
        print(f"  Trial {trial+1:2d}: E/bond = {E_per_bond:+.6f}  [{status}]")
        
        if E_per_bond < lower_bound or E_per_bond > upper_bound:
            print(f"    ⚠️  UNPHYSICAL! Outside [{lower_bound}, {upper_bound}]")
            print(f"    E3_1 = {E3_1.real.item():.6f}, E3_2 = {E3_2.real.item():.6f}, E3_3 = {E3_3.real.item():.6f}")
    
    print(f"\nSummary: E/bond range = [{min(energies):.6f}, {max(energies):.6f}]")
    print(f"Expected range: [-0.75, +0.25]")
    
    all_ok = all((-0.75 <= e <= 0.25) for e in energies)
    return all_ok

def test_product_state():
    """Test a ferromagnetic product state |↑↑↑↑↑↑⟩."""
    print("\n" + "=" * 70)
    print("TEST 2: Ferromagnetic product state")
    print("=" * 70)
    
    D_bond, chi, d_PHYS = 2, 5, 2
    D_sq = D_bond ** 2
    
    # Build Hamiltonian
    H = build_heisenberg_H(J=1.0)
    Hs = [H] * 9
    
    # Create ferromagnetic state: all sites in |↑⟩ (spin up)
    # For a product state, iPEPS tensor is just |↑⟩ on physical index
    a = torch.zeros(D_bond, D_bond, D_bond, d_PHYS, dtype=CDTYPE)
    a[:, :, :, 0] = torch.randn(D_bond, D_bond, D_bond, dtype=CDTYPE)
    a = normalize_tensor(a)
    
    b, c, d, e, f = [a.clone() for _ in range(5)]
    
    # Build environment
    A, B, C, D, E, F = abcdef_to_ABCDEF(a, b, c, d, e, f, D_sq)
    all27 = CTMRG_from_init_to_stop(A, B, C, D, E, F, chi, D_sq, 50, 1e-7)
    
    # Compute energy
    with torch.no_grad():
        E3_1 = energy_expectation_nearest_neighbor_3ebadcf_bonds(
            a, b, c, d, e, f, Hs[0], Hs[1], Hs[2],
            chi, D_bond, *all27[:9])
        E3_2 = energy_expectation_nearest_neighbor_3afcbed_bonds(
            a, b, c, d, e, f, Hs[3], Hs[4], Hs[5],
            chi, D_bond, *all27[9:18])
        E3_3 = energy_expectation_nearest_neighbor_other_3_bonds(
            a, b, c, d, e, f, Hs[6], Hs[7], Hs[8],
            chi, D_bond, *all27[18:27])
        
        E_total = (E3_1 + E3_2 + E3_3).real.item()
    
    E_per_bond = E_total / 9
    
    # For |↑↑⟩ state: S_i · S_j = (1/2)(1/2) = +1/4
    expected = +0.25
    
    print(f"  E/bond = {E_per_bond:+.6f}")
    print(f"  Expected (ferromagnetic |↑↑⟩): {expected:+.6f}")
    print(f"  Difference: {abs(E_per_bond - expected):.6f}")
    
    close = abs(E_per_bond - expected) < 0.05
    status = "OK" if close else "FAIL"
    print(f"  Status: [{status}]")
    
    if not close:
        print(f"    E3_1 = {E3_1.real.item():.6f}, E3_2 = {E3_2.real.item():.6f}, E3_3 = {E3_3.real.item():.6f}")
    
    return close

def test_energy_components():
    """Test that individual 3-bond energies are within bounds."""
    print("\n" + "=" * 70)
    print("TEST 3: Individual 3-bond energy components")
    print("=" * 70)
    
    D_bond, chi, d_PHYS = 2, 5, 2
    D_sq = D_bond ** 2
    
    # Build Hamiltonian
    H = build_heisenberg_H(J=1.0)
    Hs = [H] * 9
    
    # Random initialization
    a, b, c, d, e, f = initialize_abcdef('random', D_bond, d_PHYS, 1e-3)
    
    # Build environment
    A, B, C, D, E, F = abcdef_to_ABCDEF(a, b, c, d, e, f, D_sq)
    all27 = CTMRG_from_init_to_stop(A, B, C, D, E, F, chi, D_sq, 50, 1e-7)
    
    # Compute each component
    with torch.no_grad():
        E3_1 = energy_expectation_nearest_neighbor_3ebadcf_bonds(
            a, b, c, d, e, f, Hs[0], Hs[1], Hs[2],
            chi, D_bond, *all27[:9]).real.item()
        E3_2 = energy_expectation_nearest_neighbor_3afcbed_bonds(
            a, b, c, d, e, f, Hs[3], Hs[4], Hs[5],
            chi, D_bond, *all27[9:18]).real.item()
        E3_3 = energy_expectation_nearest_neighbor_other_3_bonds(
            a, b, c, d, e, f, Hs[6], Hs[7], Hs[8],
            chi, D_bond, *all27[18:27]).real.item()
    
    E_total = E3_1 + E3_2 + E3_3
    
    print(f"  E3_ebadcf (bonds EB,AD,CF) = {E3_1:+.6f}  (per bond: {E3_1/3:+.6f})")
    print(f"  E3_afcbed (bonds AF,CB,ED) = {E3_2:+.6f}  (per bond: {E3_2/3:+.6f})")
    print(f"  E3_other  (bonds CD,EF,AB) = {E3_3:+.6f}  (per bond: {E3_3/3:+.6f})")
    print(f"  E_total (9 bonds)          = {E_total:+.6f}  (per bond: {E_total/9:+.6f})")
    
    # Each 3-bond group should be within [-2.25, +0.75] (3 bonds × bounds)
    lower = -2.25
    upper = +0.75
    
    ok1 = lower <= E3_1 <= upper
    ok2 = lower <= E3_2 <= upper
    ok3 = lower <= E3_3 <= upper
    ok_total = (-0.75 * 9) <= E_total <= (+0.25 * 9)
    
    print(f"\n  E3_1 in [{lower}, {upper}]? {ok1}")
    print(f"  E3_2 in [{lower}, {upper}]? {ok2}")
    print(f"  E3_3 in [{lower}, {upper}]? {ok3}")
    print(f"  E_total in [{-6.75}, {+2.25}]? {ok_total}")
    
    all_ok = ok1 and ok2 and ok3 and ok_total
    status = "OK" if all_ok else "FAIL"
    print(f"\n  Status: [{status}]")
    
    return all_ok

if __name__ == "__main__":
    print("Testing physical bounds of energy calculations\n")
    
    test1_ok = test_random_tensors()
    test2_ok = test_product_state()
    test3_ok = test_energy_components()
    
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"  Test 1 (random tensors):       {'PASS' if test1_ok else 'FAIL'}")
    print(f"  Test 2 (product state):        {'PASS' if test2_ok else 'FAIL'}")
    print(f"  Test 3 (energy components):    {'PASS' if test3_ok else 'FAIL'}")
    print("=" * 70)
    
    if all([test1_ok, test2_ok, test3_ok]):
        print("✅ All physical bounds tests PASSED")
        sys.exit(0)
    else:
        print("❌ Some tests FAILED - energy calculation may be unphysical")
        sys.exit(1)
