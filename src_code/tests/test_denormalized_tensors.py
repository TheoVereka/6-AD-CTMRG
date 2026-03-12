"""
Test if using denormalized tensors with a fixed environment causes unphysical energies.
This mimics what happens during L-BFGS line search:
1. Environment is built with normalized tensors
2. L-BFGS modifies tensors without renormalization
3. Energy is evaluated with denormalized tensors + old environment
"""

import sys
sys.path.append('/home/chye/6ADctmrg/6-AD-CTMRG/src_code/src')
sys.path.append('/home/chye/6ADctmrg/6-AD-CTMRG/src_code/scripts')

import torch
import numpy as np
from core_unrestricted import (
    normalize_tensor,
    initialize_abcdef,
    abcdef_to_ABCDEF,
    CTMRG_from_init_to_stop,
    energy_expectation_nearest_neighbor_3ebadcf_bonds,
    energy_expectation_nearest_neighbor_3afcbed_bonds,
    energy_expectation_nearest_neighbor_other_3_bonds,
    set_dtype, CDTYPE
)

# Use complex128 for precision
set_dtype(True)

torch.manual_seed(42)
np.random.seed(42)

# Build Heisenberg Hamiltonian
d_phys = 2
sx = torch.tensor([[0., 1.], [1., 0.]], dtype=torch.complex128)
sy = torch.tensor([[0., -1.j], [1.j, 0.]], dtype=torch.complex128)
sz = torch.tensor([[1., 0.], [0., -1.]], dtype=torch.complex128)

import opt_einsum as oe
H = (
    oe.contract("ij,kl->ikjl", sx, sx, backend="torch") +
    oe.contract("ij,kl->ikjl", sy, sy, backend="torch") +
    oe.contract("ij,kl->ikjl", sz, sz, backend="torch")
)

Hs = [H] * 9

# Test parameters
D_bond = 2
chi = 5
d_phys = 2
D_sq = D_bond * D_bond

print("="*70)
print("Testing energy with NORMALIZED tensors (baseline)")
print("="*70)

# Initialize normalized tensors
a, b, c, d, e, f = initialize_abcdef('random', D_bond, d_phys, 0.0)

for t in [a, b, c, d, e, f]:
    t.data = normalize_tensor(t.data)

# Build environment with normalized tensors
with torch.no_grad():
    A, B, C, D, E, F = abcdef_to_ABCDEF(a, b, c, d, e, f, D_sq)
    (C21CD, C32EF, C13AB, T1F,  T2A,  T2B,  T3C,  T3D,  T1E,
     C21EB, C32AD, C13CF, T1D,  T2C,  T2F,  T3E,  T3B,  T1A,
     C21AF, C32CB, C13ED, T1B,  T2E,  T2D,  T3A,  T3F,  T1C,
     ctm_steps) = \
        CTMRG_from_init_to_stop(A, B, C, D, E, F, chi, D_sq, 100, 1e-10)

print(f"CTMRG converged in {ctm_steps} steps")

# Energy with normalized tensors
E1 = energy_expectation_nearest_neighbor_3ebadcf_bonds(
    a,b,c,d,e,f, Hs[0],Hs[1],Hs[2], chi, D_bond,
    C21CD,C32EF,C13AB,T1F,T2A,T2B,T3C,T3D,T1E
).real.item()

E2 = energy_expectation_nearest_neighbor_3afcbed_bonds(
    a,b,c,d,e,f, Hs[3],Hs[4],Hs[5], chi, D_bond,
    C21AF,C32CB,C13ED,T1B,T2E,T2D,T3A,T3F,T1C
).real.item()

E3 = energy_expectation_nearest_neighbor_other_3_bonds(
    a,b,c,d,e,f, Hs[6],Hs[7],Hs[8], chi, D_bond,
    C21EB,C32AD,C13CF,T1D,T2C,T2F,T3E,T3B,T1A
).real.item()

E_total_normalized = E1 + E2 + E3
E_per_bond_normalized = E_total_normalized / 9

print(f"E_total (normalized tensors): {E_total_normalized:.6f}")
print(f"E_per_bond (normalized): {E_per_bond_normalized:.6f}")
print(f"Physical bounds: [-0.75, +0.25]")
if -0.75 <= E_per_bond_normalized <= 0.25:
    print("✅ Within physical bounds")
else:
    print("❌ OUTSIDE physical bounds!")

# Now test with DENORMALIZED tensors but SAME environment
print("\n" + "="*70)
print("Testing energy with DENORMALIZED tensors (environment unchanged)")
print("="*70)

denorm_factors = [0.01, 0.1, 2.0, 10.0, 100.0, 1000.0]

for scale in denorm_factors:
    # Scale all tensors by the same factor
    a_denorm = a * scale
    b_denorm = b * scale
    c_denorm = c * scale
    d_denorm = d * scale
    e_denorm = e * scale
    f_denorm = f * scale
    
    # Compute energy with denormalized tensors but OLD environment
    E1_denorm = energy_expectation_nearest_neighbor_3ebadcf_bonds(
        a_denorm,b_denorm,c_denorm,d_denorm,e_denorm,f_denorm,
        Hs[0],Hs[1],Hs[2], chi, D_bond,
        C21CD,C32EF,C13AB,T1F,T2A,T2B,T3C,T3D,T1E
    ).real.item()
    
    E2_denorm = energy_expectation_nearest_neighbor_3afcbed_bonds(
        a_denorm,b_denorm,c_denorm,d_denorm,e_denorm,f_denorm,
        Hs[3],Hs[4],Hs[5], chi, D_bond,
        C21AF,C32CB,C13ED,T1B,T2E,T2D,T3A,T3F,T1C
    ).real.item()
    
    E3_denorm = energy_expectation_nearest_neighbor_other_3_bonds(
        a_denorm,b_denorm,c_denorm,d_denorm,e_denorm,f_denorm,
        Hs[6],Hs[7],Hs[8], chi, D_bond,
        C21EB,C32AD,C13CF,T1D,T2C,T2F,T3E,T3B,T1A
    ).real.item()
    
    E_total_denorm = E1_denorm + E2_denorm + E3_denorm
    E_per_bond_denorm = E_total_denorm / 9
    
    print(f"\nScale factor: {scale:8.2f}  ||tensor|| ≈ {scale:.2e}")
    print(f"  E_total: {E_total_denorm:+.6e}")
    print(f"  E_per_bond: {E_per_bond_denorm:+.6f}")
    
    if -0.75 <= E_per_bond_denorm <= 0.25:
        print("  ✅ Within physical bounds")
    else:
        print(f"  ❌ OUTSIDE physical bounds by {abs(E_per_bond_denorm - (-0.75 if E_per_bond_denorm < -0.75 else 0.25)):.3f}")

print("\n" + "="*70)
print("Testing with EXTREMELY large denormalization (reproducing +80 bug)")
print("="*70)

# Try to get E_total ≈ +80 (which is +8.89 per bond)
# If energy scales wrong, we might hit this
for scale in [1e4, 1e5, 1e6, 1e7]:
    a_huge = a * scale
    b_huge = b * scale
    c_huge = c * scale
    d_huge = d * scale
    e_huge = e * scale
    f_huge = f * scale
    
    try:
        E1_huge = energy_expectation_nearest_neighbor_3ebadcf_bonds(
            a_huge,b_huge,c_huge,d_huge,e_huge,f_huge,
            Hs[0],Hs[1],Hs[2], chi, D_bond,
            C21CD,C32EF,C13AB,T1F,T2A,T2B,T3C,T3D,T1E
        ).real.item()
        
        E2_huge = energy_expectation_nearest_neighbor_3afcbed_bonds(
            a_huge,b_huge,c_huge,d_huge,e_huge,f_huge,
            Hs[3],Hs[4],Hs[5], chi, D_bond,
            C21AF,C32CB,C13ED,T1B,T2E,T2D,T3A,T3F,T1C
        ).real.item()
        
        E3_huge = energy_expectation_nearest_neighbor_other_3_bonds(
            a_huge,b_huge,c_huge,d_huge,e_huge,f_huge,
            Hs[6],Hs[7],Hs[8], chi, D_bond,
            C21EB,C32AD,C13CF,T1D,T2C,T2F,T3E,T3B,T1A
        ).real.item()
        
        E_total_huge = E1_huge + E2_huge + E3_huge
        E_per_bond_huge = E_total_huge / 9
        
        print(f"\nScale={scale:.0e}: E_total={E_total_huge:+.6e}, E/bond={E_per_bond_huge:+.6f}")
        
        if abs(E_total_huge) > 10 or E_per_bond_huge > 0.25 or E_per_bond_huge < -0.75:
            print(f"  🔴 BUG REPRODUCED! Unphysical energy: {E_total_huge:+.2f}")
            
    except Exception as ex:
        print(f"\nScale={scale:.0e}: ❌ Numerical error: {ex}")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)
print("If denormalized tensors cause unphysical energies, the bug is in")
print("the optimizer not renormalizing tensors frequently enough during")
print("L-BFGS line search.")
