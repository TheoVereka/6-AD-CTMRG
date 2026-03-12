"""
Check if the energy has a significant imaginary component.
For a Hermitian Hamiltonian, <ψ|H|ψ> should be purely real.
If .imag is large, that indicates a bug in the contraction logic.
"""

import sys
sys.path.append('/home/chye/6ADctmrg/6-AD-CTMRG/src_code/src')
sys.path.append('/home/chye/6ADctmrg/6-AD-CTMRG/src_code/scripts')

import torch
import numpy as np
from core_unrestricted import (
    normalize_tensor, initialize_abcdef, abcdef_to_ABCDEF,
    CTMRG_from_init_to_stop,
    energy_expectation_nearest_neighbor_3ebadcf_bonds,
    energy_expectation_nearest_neighbor_3afcbed_bonds,
    energy_expectation_nearest_neighbor_other_3_bonds,
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

D_bond = 3
chi = 29
d_phys = 2
D_sq = D_bond * D_bond

print("="*70)
print("Checking for imaginary components in energy expectation values")
print("="*70)

# Test 1: Normal random state
print("\n1. Random initialized state (normalized):")
a, b, c, d, e, f = initialize_abcdef('random', D_bond, d_phys, 0.0)
for t in [a, b, c, d, e, f]:
    t.data = normalize_tensor(t.data)

with torch.no_grad():
    A, B, C, D, E, F = abcdef_to_ABCDEF(a, b, c, d, e, f, D_sq)
    (C21CD, C32EF, C13AB, T1F,  T2A,  T2B,  T3C,  T3D,  T1E,
     C21EB, C32AD, C13CF, T1D,  T2C,  T2F,  T3E,  T3B,  T1A,
     C21AF, C32CB, C13ED, T1B,  T2E,  T2D,  T3A,  T3F,  T1C,
     ctm_steps) = \
        CTMRG_from_init_to_stop(A, B, C, D, E, F, chi, D_sq, 100, 1e-10)

E1 = energy_expectation_nearest_neighbor_3ebadcf_bonds(
    a,b,c,d,e,f, H,H,H, chi, D_bond,
    C21CD,C32EF,C13AB,T1F,T2A,T2B,T3C,T3D,T1E
)
E2 = energy_expectation_nearest_neighbor_3afcbed_bonds(
    a,b,c,d,e,f, H,H,H, chi, D_bond,
    C21EB,C32AD,C13CF,T1D,T2C,T2F,T3E,T3B,T1A
)
E3 = energy_expectation_nearest_neighbor_other_3_bonds(
    a,b,c,d,e,f, H,H,H, chi, D_bond,
    C21AF,C32CB,C13ED,T1B,T2E,T2D,T3A,T3F,T1C
)

E_total = E1 + E2 + E3

print(f"   E1: real={E1.real.item():+.6f}, imag={E1.imag.item():+.3e}")
print(f"   E2: real={E2.real.item():+.6f}, imag={E2.imag.item():+.3e}")
print(f"   E3: real={E3.real.item():+.6f}, imag={E3.imag.item():+.3e}")
print(f"   E_total: real={E_total.real.item():+.6f}, imag={E_total.imag.item():+.3e}")
print(f"   |Im(E)|/|Re(E)| = {abs(E_total.imag.item())/abs(E_total.real.item()):.3e}")

if abs(E_total.imag.item()) < 1e-6 * abs(E_total.real.item()):
    print("   ✅ Imaginary part is negligible (numerical noise)")
else:
    print(f"   ⚠️  Imaginary part is {abs(E_total.imag.item())/abs(E_total.real.item()):.1e} of real part!")

# Test 2: Denormalized tensors (simulating bad L-BFGS step)
print("\n2. Denormalized tensors (1000× scale):")
a_big = a * 1000
e_big = e * 1000

E1_big = energy_expectation_nearest_neighbor_3ebadcf_bonds(
    a_big,b,c,d,e_big,f, H,H,H, chi, D_bond,
    C21CD,C32EF,C13AB,T1F,T2A,T2B,T3C,T3D,T1E
)
E2_big = energy_expectation_nearest_neighbor_3afcbed_bonds(
    a_big,b,c,d,e_big,f, H,H,H, chi, D_bond,
    C21EB,C32AD,C13CF,T1D,T2C,T2F,T3E,T3B,T1A
)
E3_big = energy_expectation_nearest_neighbor_other_3_bonds(
    a_big,b,c,d,e_big,f, H,H,H, chi, D_bond,
    C21AF,C32CB,C13ED,T1B,T2E,T2D,T3A,T3F,T1C
)

E_total_big = E1_big + E2_big + E3_big

print(f"   E_total: real={E_total_big.real.item():+.6f}, imag={E_total_big.imag.item():+.3e}")
print(f"   |Im(E)|/|Re(E)| = {abs(E_total_big.imag.item())/abs(E_total_big.real.item()):.3e}")

# Test 3: Random requires_grad tensors (simulating gradient computation)
print("\n3. With requires_grad=True (simulating optimization):")
a_grad = a.detach().clone().requires_grad_(True)
e_grad = e.detach().clone().requires_grad_(True)

E1_grad = energy_expectation_nearest_neighbor_3ebadcf_bonds(
    a_grad,b,c,d,e_grad,f, H,H,H, chi, D_bond,
    C21CD,C32EF,C13AB,T1F,T2A,T2B,T3C,T3D,T1E
)
E2_grad = energy_expectation_nearest_neighbor_3afcbed_bonds(
    a_grad,b,c,d,e_grad,f, H,H,H, chi, D_bond,
    C21EB,C32AD,C13CF,T1D,T2C,T2F,T3E,T3B,T1A
)
E3_grad = energy_expectation_nearest_neighbor_other_3_bonds(
    a_grad,b,c,d,e_grad,f, H,H,H, chi, D_bond,
    C21AF,C32CB,C13ED,T1B,T2E,T2D,T3A,T3F,T1C
)

E_total_grad = E1_grad + E2_grad + E3_grad

print(f"   E_total: real={E_total_grad.real.item():+.6f}, imag={E_total_grad.imag.item():+.3e}")
print(f"   |Im(E)|/|Re(E)| = {abs(E_total_grad.imag.item())/abs(E_total_grad.real.item()):.3e}")

# Test 4: After backward()
print("\n4. After calling .backward() (checking if gradients corrupt state):")
E_total_grad.real.backward()  # Only backprop through real part
print(f"   Gradients computed: a_grad.grad norm = {torch.norm(a_grad.grad).item():.3e}")

# Recompute energy after backward
E1_after = energy_expectation_nearest_neighbor_3ebadcf_bonds(
    a_grad,b,c,d,e_grad,f, H,H,H, chi, D_bond,
    C21CD,C32EF,C13AB,T1F,T2A,T2B,T3C,T3D,T1E
)
E2_after = energy_expectation_nearest_neighbor_3afcbed_bonds(
    a_grad,b,c,d,e_grad,f, H,H,H, chi, D_bond,
    C21EB,C32AD,C13CF,T1D,T2C,T2F,T3E,T3B,T1A
)
E3_after = energy_expectation_nearest_neighbor_other_3_bonds(
    a_grad,b,c,d,e_grad,f, H,H,H, chi, D_bond,
    C21AF,C32CB,C13ED,T1B,T2E,T2D,T3A,T3F,T1C
)

E_total_after = E1_after + E2_after + E3_after

print(f"   E_total (recomputed): real={E_total_after.real.item():+.6f}, imag={E_total_after.imag.item():+.3e}")
print(f"   Energy changed after backward? {abs(E_total_grad.real.item() - E_total_after.real.item()) > 1e-10}")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("If imaginary component is >>1e-6 times real part, there's a bug.")
print("If energy changes after .backward(), there's state corruption.")
print("If neither, the energy calculation is structurally correct.")
