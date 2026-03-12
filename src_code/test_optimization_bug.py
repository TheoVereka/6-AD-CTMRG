#!/usr/bin/env python3
"""
Test for unphysical energies during optimization.
Mimics the actual optimization loop to trigger the bug.
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

set_dtype(True)
CDTYPE = torch.complex128

def build_heisenberg_H(J=1.0):
    sx = torch.tensor([[0, 1], [1, 0]], dtype=CDTYPE) / 2
    sy = torch.tensor([[0, -1j], [1j, 0]], dtype=CDTYPE) / 2
    sz = torch.tensor([[1, 0], [0, -1]], dtype=CDTYPE) / 2
    SdotS = (oe.contract("ij,kl->ikjl", sx, sx)
           + oe.contract("ij,kl->ikjl", sy, sy)
           + oe.contract("ij,kl->ikjl", sz, sz))
    return J * SdotS

print("=" * 70)
print("Testing optimization loop for unphysical energies")
print("=" * 70)

D_bond, chi, d_PHYS = 2, 5, 2
D_sq = D_bond ** 2

# Build Hamiltonian
H = build_heisenberg_H(J=1.0)
Hs = [H] * 9

# Initialize tensors WITH gradients
a, b, c, d, e, f = initialize_abcdef('random', D_bond, d_PHYS, 1e-3)
for t in (a, b, c, d, e, f):
    t.requires_grad_(True)

print(f"Initial tensors have requires_grad={a.requires_grad}")

# Run several optimization steps
for step in range(10):
    # Normalize (like the real code)
    with torch.no_grad():
        a.data = normalize_tensor(a.data)
        b.data = normalize_tensor(b.data)
        c.data = normalize_tensor(c.data)
        d.data = normalize_tensor(d.data)
        e.data = normalize_tensor(e.data)
        f.data = normalize_tensor(f.data)
    
    # Build environment (no grad)
    with torch.no_grad():
        A, B, C, D, E, F = abcdef_to_ABCDEF(a, b, c, d, e, f, D_sq)
        all27 = CTMRG_from_init_to_stop(A, B, C, D, E, F, chi, D_sq, 50, 1e-7)
    
    # Compute energy WITH gradients (like in closure)
    E3_1 = energy_expectation_nearest_neighbor_3ebadcf_bonds(
        a, b, c, d, e, f, Hs[0], Hs[1], Hs[2],
        chi, D_bond, *all27[:9])
    E3_2 = energy_expectation_nearest_neighbor_3afcbed_bonds(
        a, b, c, d, e, f, Hs[3], Hs[4], Hs[5],
        chi, D_bond, *all27[9:18])
    E3_3 = energy_expectation_nearest_neighbor_other_3_bonds(
        a, b, c, d, e, f, Hs[6], Hs[7], Hs[8],
        chi, D_bond, *all27[18:27])
    
    loss = (E3_1 + E3_2 + E3_3).real
    
    E_per_bond = loss.item() / 9
    
    # Check bounds
    within_bounds = (-0.75 <= E_per_bond <= 0.25)
    status = "OK" if within_bounds else "FAIL⚠️"
    
    print(f"Step {step:2d}: loss={loss.item():+12.6f}  E/bond={E_per_bond:+.6f}  [{status}]")
    
    if not within_bounds:
        print(f"  ⚠️  UNPHYSICAL! E3_1={E3_1.real.item():.4f}, E3_2={E3_2.real.item():.4f}, E3_3={E3_3.real.item():.4f}")
        print(f"  Tensor norms: a={torch.norm(a).item():.4f}, b={torch.norm(b).item():.4f}")
    
    # Take gradient step (simplified L-BFGS)
    loss.backward()
    
    with torch.no_grad():
        # Simple gradient descent step
        lr = 0.01
        a.data -= lr * a.grad
        b.data -= lr * b.grad
        c.data -= lr * c.grad
        d.data -= lr * d.grad
        e.data -= lr * e.grad
        f.data -= lr * f.grad
        
        # Zero gradients
        a.grad.zero_()
        b.grad.zero_()
        c.grad.zero_()
        d.grad.zero_()
        e.grad.zero_()
        f.grad.zero_()

print("\n" + "=" * 70)
print("Now testing with actual L-BFGS optimizer...")
print("=" * 70)

# Reset
a, b, c, d, e, f = initialize_abcdef('random', D_bond, d_PHYS, 1e-3)
for t in (a, b, c, d, e, f):
    t.requires_grad_(True)

optimizer = torch.optim.LBFGS(
    [a, b, c, d, e, f],
    lr=0.1,
    max_iter=3,
    tolerance_grad=1e-7,
    tolerance_change=1e-9,
    history_size=3,
    line_search_fn='strong_wolfe',
)

for outer_step in range(3):
    # Normalize
    with torch.no_grad():
        for t in (a, b, c, d, e, f):
            t.data = normalize_tensor(t.data)
    
    # Build environment
    with torch.no_grad():
        A, B, C, D, E, F = abcdef_to_ABCDEF(a, b, c, d, e, f, D_sq)
        all27 = CTMRG_from_init_to_stop(A, B, C, D, E, F, chi, D_sq, 50, 1e-7)
    
    # L-BFGS step
    def closure():
        optimizer.zero_grad()
        E3_1 = energy_expectation_nearest_neighbor_3ebadcf_bonds(
            a, b, c, d, e, f, Hs[0], Hs[1], Hs[2],
            chi, D_bond, *all27[:9])
        E3_2 = energy_expectation_nearest_neighbor_3afcbed_bonds(
            a, b, c, d, e, f, Hs[3], Hs[4], Hs[5],
            chi, D_bond, *all27[9:18])
        E3_3 = energy_expectation_nearest_neighbor_other_3_bonds(
            a, b, c, d, e, f, Hs[6], Hs[7], Hs[8],
            chi, D_bond, *all27[18:27])
        loss = (E3_1 + E3_2 + E3_3).real
        loss.backward()
        return loss
    
    loss_val = optimizer.step(closure)
    loss_item = loss_val.item()
    E_per_bond = loss_item / 9
    
    within_bounds = (-0.75 <= E_per_bond <= 0.25)
    status = "OK" if within_bounds else "FAIL⚠️"
    
    print(f"L-BFGS step {outer_step}: loss={loss_item:+12.6f}  E/bond={E_per_bond:+.6f}  [{status}]")
    
    if not within_bounds:
        print(f"  ⚠️  UN PHYSICAL!")

print("\n✅ Test complete")
