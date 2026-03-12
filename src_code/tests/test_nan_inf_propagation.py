"""
Test for NaN/Inf propagation in energy calculation.
Check if intermediate contractions can produce NaN or Inf values.
"""

import sys
sys.path.append('/home/chye/6ADctmrg/6-AD-CTMRG/src_code/src')
sys.path.append('/home/chye/6ADctmrg/6-AD-CTMRG/src_code/scripts')

import torch
import numpy as np
from core_unrestricted import (
    normalize_tensor,
    initialize_abcdef, abcdef_to_ABCDEF,
    CTMRG_from_init_to_stop,
    energy_expectation_nearest_neighbor_3ebadcf_bonds,
    set_dtype
)
import opt_einsum as oe

set_dtype(True)
torch.manual_seed(42)

# Build Hamiltonian
sx = torch.tensor([[0., 1.], [1., 0.]], dtype=torch.complex128)
sy = torch.tensor([[0., -1.j], [1.j, 0.]], dtype=torch.complex128)
sz = torch.tensor([[1., 0.], [0., -1.]], dtype=torch.complex128)

H = (oe.contract("ij,kl->ikjl", sx, sx, backend="torch") +
     oe.contract("ij,kl->ikjl", sy, sy, backend="torch") +
     oe.contract("ij,kl->ikjl", sz, sz, backend="torch"))

# Test with user's actual parameters
D_bond = 3
chi = 29
d_phys = 2
D_sq = D_bond * D_bond

print("="*70)
print(f"Testing with D={D_bond}, chi={chi} (user's actual parameters)")
print("="*70)

# Initialize and normalize
a, b, c, d, e, f = initialize_abcdef('random', D_bond, d_phys, 0.0)
for t in [a, b, c, d, e, f]:
    t.data = normalize_tensor(t.data)
    
print(f"Initial tensor norms: {[torch.norm(t).item() for t in [a,b,c,d,e,f]]}")
print(f"Any NaN in tensors? {any(torch.isnan(t).any() for t in [a,b,c,d,e,f])}")
print(f"Any Inf in tensors? {any(torch.isinf(t).any() for t in [a,b,c,d,e,f])}")

# Build environment
print("\nBuilding environment with CTMRG...")
with torch.no_grad():
    A, B, C, D, E, F = abcdef_to_ABCDEF(a, b, c, d, e, f, D_sq)
    (C21CD, C32EF, C13AB, T1F,  T2A,  T2B,  T3C,  T3D,  T1E,
     C21EB, C32AD, C13CF, T1D,  T2C,  T2F,  T3E,  T3B,  T1A,
     C21AF, C32CB, C13ED, T1B,  T2E,  T2D,  T3A,  T3F,  T1C,
     ctm_steps) = \
        CTMRG_from_init_to_stop(A, B, C, D, E, F, chi, D_sq, 100, 1e-10)

print(f"CTMRG converged in {ctm_steps} steps")

# Check environment tensors
env_tensors = [C21CD, C32EF, C13AB, T1F, T2A, T2B, T3C, T3D, T1E]
env_norms = [torch.norm(t).item() for t in env_tensors]
print(f"\nEnvironment tensor norms: min={min(env_norms):.3e}, max={max(env_norms):.3e}")
print(f"Any NaN in environment? {any(torch.isnan(t).any() for t in env_tensors)}")
print(f"Any Inf in environment? {any(torch.isinf(t).any() for t in env_tensors)}")

# Compute energy
print("\nComputing energy...")
E1 = energy_expectation_nearest_neighbor_3ebadcf_bonds(
    a,b,c,d,e,f, H,H,H, chi, D_bond,
    C21CD,C32EF,C13AB,T1F,T2A,T2B,T3C,T3D,T1E
)

print(f"Energy (3 bonds): {E1.real.item():.6f}")
print(f"Energy per bond: {E1.real.item()/3:.6f}")
print(f"Is NaN? {torch.isnan(E1).item()}")
print(f"Is Inf? {torch.isinf(E1).item()}")

print("\n" + "="*70)
print("Testing with pathological tensor configurations")
print("="*70)

# Test 1: All zero tensors
print("\n1. All-zero tensors:")
a_zero = torch.zeros_like(a)
try:
    E_zero = energy_expectation_nearest_neighbor_3ebadcf_bonds(
        a_zero,a_zero,a_zero,a_zero,a_zero,a_zero, H,H,H, chi, D_bond,
        C21CD,C32EF,C13AB,T1F,T2A,T2B,T3C,T3D,T1E
    )
    print(f"   Energy: {E_zero.real.item():.6e} (NaN={torch.isnan(E_zero).item()}, Inf={torch.isinf(E_zero).item()})")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 2: Extremely large tensors (simulating bad L-BFGS step)
print("\n2. Extremely large tensors (norm=1e10):")
a_huge = a * 1e10 / torch.norm(a)
try:
    E_huge = energy_expectation_nearest_neighbor_3ebadcf_bonds(
        a_huge,a_huge,a_huge,a_huge,a_huge,a_huge, H,H,H, chi, D_bond,
        C21CD,C32EF,C13AB,T1F,T2A,T2B,T3C,T3D,T1E
    )
    print(f"   Energy: {E_huge.real.item():.6e} (NaN={torch.isnan(E_huge).item()}, Inf={torch.isinf(E_huge).item()})")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 3: Tensors with deliberate NaN
print("\n3. Tensors with NaN:")
a_nan = a.clone()
a_nan[0, 0, 0, 0] = float('nan')
try:
    E_nan = energy_expectation_nearest_neighbor_3ebadcf_bonds(
        a_nan,a,a,a,a,a, H,H,H, chi, D_bond,
        C21CD,C32EF,C13AB,T1F,T2A,T2B,T3C,T3D,T1E
    )
    print(f"   Energy: {E_nan.real.item()} (NaN={torch.isnan(E_nan).item()}, Inf={torch.isinf(E_nan).item()})")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 4: Mixed very large and very small
print("\n4. Mixed large/small tensors:")
a_big = a * 1e6
a_small = a * 1e-6
try:
    E_mixed = energy_expectation_nearest_neighbor_3ebadcf_bonds(
        a_big,a_small,a_big,a_small,a_big,a_small, H,H,H, chi, D_bond,
        C21CD,C32EF,C13AB,T1F,T2A,T2B,T3C,T3D,T1E
    )
    print(f"   Energy: {E_mixed.real.item():.6e} (NaN={torch.isnan(E_mixed).item()}, Inf={torch.isinf(E_mixed).item()})")
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)
print("If NaN/Inf appears in normal operation or pathological cases,")
print("that's likely the source of unphysical +80, +52, -19 values.")
