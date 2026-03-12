"""
Test to verify all three energy functions produce POSITIVE normalizations.
This is critical - negative norm indicates a bug in contraction logic.

This test EXACTLY reproduces the code from the three energy functions in
core_unrestricted.py to extract the normalization factors.
"""

import sys
sys.path.append('/home/chye/6ADctmrg/6-AD-CTMRG/src_code/src')

import torch
import numpy as np
from core_unrestricted import (
    normalize_tensor, initialize_abcdef, abcdef_to_ABCDEF,
    CTMRG_from_init_to_stop, set_dtype
)
import opt_einsum as oe

set_dtype(True)

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
print("Testing normalization sign across multiple random initializations")
print("="*70)

# Function to extract normalization from each energy function
def get_norms(a, b, c, d, e, f, chi, D_bond,
              C21CD, C32EF, C13AB, T1F, T2A, T2B, T3C, T3D, T1E,
              C21EB, C32AD, C13CF, T1D, T2C, T2F, T3E, T3B, T1A,
              C21AF, C32CB, C13ED, T1B, T2E, T2D, T3A, T3F, T1C):
    """Extract normalization factors using EXACT code from core_unrestricted.py"""
    
    # ========================================================================
    # Function 1: energy_expectation_nearest_neighbor_3ebadcf_bonds
    # ========================================================================
    T1F_1 = T1F.reshape(chi,chi,D_bond,D_bond)
    T2A_1 = T2A.reshape(chi,chi,D_bond,D_bond)
    T2B_1 = T2B.reshape(chi,chi,D_bond,D_bond)
    T3C_1 = T3C.reshape(chi,chi,D_bond,D_bond)
    T3D_1 = T3D.reshape(chi,chi,D_bond,D_bond)
    T1E_1 = T1E.reshape(chi,chi,D_bond,D_bond)

    open_E_1 = oe.contract("YX,MYar,abci,rstj->MbsXctij", C21CD, T1F_1, e, e.conj(), optimize=[(0,1),(0,1),(0,1)], backend="torch")
    open_D_1 = oe.contract("MYct,abci,rstj->YarMbsij", T3C_1, d, d.conj(), optimize=[(0,1),(0,1)], backend="torch")
    open_A_1 = oe.contract("ZY,NZbs,abci,rstj->NctYarij", C32EF, T2B_1, a, a.conj(), optimize=[(0,1),(0,1),(0,1)], backend="torch")
    open_F_1 = oe.contract("NZar,abci,rstj->ZbsNctij", T1E_1, f, f.conj(), optimize=[(0,1),(0,1)], backend="torch")
    open_C_1 = oe.contract("XZ,LXct,abci,rstj->LarZbsij", C13AB, T3D_1, c, c.conj(), optimize=[(0,1),(0,1),(0,1)], backend="torch")
    open_B_1 = oe.contract("LXbs,abci,rstj->XctLarij", T2A_1, b, b.conj(), optimize=[(0,1),(0,1)], backend="torch")
                          
    closed_E_1 = oe.contract("MbsXctii->MbsXct", open_E_1, backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    closed_D_1 = oe.contract("YarMbsii->YarMbs", open_D_1, backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    closed_A_1 = oe.contract("NctYarii->NctYar", open_A_1, backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    closed_F_1 = oe.contract("ZbsNctii->ZbsNct", open_F_1, backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    closed_C_1 = oe.contract("LarZbsii->LarZbs", open_C_1, backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    closed_B_1 = oe.contract("XctLarii->XctLar", open_B_1, backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)

    AD = torch.mm(closed_A_1, closed_D_1)
    CF = torch.mm(closed_C_1, closed_F_1)
    EB = torch.mm(closed_E_1, closed_B_1)

    norm_1st_env = oe.contract("xy,yz,zx->", AD, EB, CF, backend="torch")
    norm1 = norm_1st_env.real.item()
    
    # ========================================================================
    # Function 2: energy_expectation_nearest_neighbor_3afcbed_bonds
    # ========================================================================
    T1D_2 = T1D.reshape(chi,chi,D_bond,D_bond)
    T2C_2 = T2C.reshape(chi,chi,D_bond,D_bond)
    T2F_2 = T2F.reshape(chi,chi,D_bond,D_bond)
    T3E_2 = T3E.reshape(chi,chi,D_bond,D_bond)
    T3B_2 = T3B.reshape(chi,chi,D_bond,D_bond)
    T1A_2 = T1A.reshape(chi,chi,D_bond,D_bond)

    open_A_2 = oe.contract("YX,MYar,abci,rstj->MbsXctij", C21EB, T1D_2, a, a.conj(), optimize=[(0,1),(0,1),(0,1)], backend="torch")
    open_B_2 = oe.contract("MYct,abci,rstj->YarMbsij", T3E_2, b, b.conj(), optimize=[(0,1),(0,1)], backend="torch")
    open_C_2 = oe.contract("ZY,NZbs,abci,rstj->NctYarij", C32AD, T2F_2, c, c.conj(), optimize=[(0,1),(0,1),(0,1)], backend="torch")
    open_D_2 = oe.contract("NZar,abci,rstj->ZbsNctij", T1A_2, d, d.conj(), optimize=[(0,1),(0,1)], backend="torch")
    open_E_2 = oe.contract("XZ,LXct,abci,rstj->LarZbsij", C13CF, T3B_2, e, e.conj(), optimize=[(0,1),(0,1),(0,1)], backend="torch")
    open_F_2 = oe.contract("LXbs,abci,rstj->XctLarij", T2C_2, f, f.conj(), optimize=[(0,1),(0,1)], backend="torch")
                          
    closed_A_2 = oe.contract("MbsXctii->MbsXct", open_A_2, backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    closed_B_2 = oe.contract("YarMbsii->YarMbs", open_B_2, backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    closed_C_2 = oe.contract("NctYarii->NctYar", open_C_2, backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    closed_D_2 = oe.contract("ZbsNctii->ZbsNct", open_D_2, backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    closed_E_2 = oe.contract("LarZbsii->LarZbs", open_E_2, backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    closed_F_2 = oe.contract("XctLarii->XctLar", open_F_2, backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)

    CB = torch.mm(closed_C_2, closed_B_2)
    ED = torch.mm(closed_E_2, closed_D_2)
    AF = torch.mm(closed_A_2, closed_F_2)

    norm_2nd_env = oe.contract("xy,yz,zx->", CB, AF, ED, backend="torch")
    norm2 = norm_2nd_env.real.item()
    
    # ========================================================================
    # Function 3: energy_expectation_nearest_neighbor_other_3_bonds
    # ========================================================================
    T1B_3 = T1B.reshape(chi,chi,D_bond,D_bond)
    T2E_3 = T2E.reshape(chi,chi,D_bond,D_bond)
    T2D_3 = T2D.reshape(chi,chi,D_bond,D_bond)
    T3A_3 = T3A.reshape(chi,chi,D_bond,D_bond)
    T3F_3 = T3F.reshape(chi,chi,D_bond,D_bond)
    T1C_3 = T1C.reshape(chi,chi,D_bond,D_bond)

    open_C_3 = oe.contract("YX,MYar,abci,rstj->MbsXctij", C21AF, T1B_3, c, c.conj(), optimize=[(0,1),(0,1),(0,1)], backend="torch")
    open_F_3 = oe.contract("MYct,abci,rstj->YarMbsij", T3A_3, f, f.conj(), optimize=[(0,1),(0,1)], backend="torch")
    open_E_3 = oe.contract("ZY,NZbs,abci,rstj->NctYarij", C32CB, T2D_3, e, e.conj(), optimize=[(0,1),(0,1),(0,1)], backend="torch")
    open_B_3 = oe.contract("NZar,abci,rstj->ZbsNctij", T1C_3, b, b.conj(), optimize=[(0,1),(0,1)], backend="torch")
    open_A_3 = oe.contract("XZ,LXct,abci,rstj->LarZbsij", C13ED, T3F_3, a, a.conj(), optimize=[(0,1),(0,1),(0,1)], backend="torch")
    open_D_3 = oe.contract("LXbs,abci,rstj->XctLarij", T2E_3, d, d.conj(), optimize=[(0,1),(0,1)], backend="torch")
                          
    closed_C_3 = oe.contract("MbsXctii->MbsXct", open_C_3, backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    closed_F_3 = oe.contract("YarMbsii->YarMbs", open_F_3, backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    closed_E_3 = oe.contract("NctYarii->NctYar", open_E_3, backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    closed_B_3 = oe.contract("ZbsNctii->ZbsNct", open_B_3, backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    closed_A_3 = oe.contract("LarZbsii->LarZbs", open_A_3, backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    closed_D_3 = oe.contract("XctLarii->XctLar", open_D_3, backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)

    EF = torch.mm(closed_E_3, closed_F_3)
    AB = torch.mm(closed_A_3, closed_B_3)
    CD = torch.mm(closed_C_3, closed_D_3)

    norm_3rd_env = oe.contract("xy,yz,zx->", EF, CD, AB, backend="torch")
    norm3 = norm_3rd_env.real.item()
    
    return norm1, norm2, norm3

# Test with 5 random initializations
print("\nTesting 5 random initializations:\n")
all_positive = True

for trial in range(5):
    torch.manual_seed(42 + trial)
    
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
    
    norm1, norm2, norm3 = get_norms(
        a, b, c, d, e, f, chi, D_bond,
        C21CD, C32EF, C13AB, T1F, T2A, T2B, T3C, T3D, T1E,
        C21EB, C32AD, C13CF, T1D, T2C, T2F, T3E, T3B, T1A,
        C21AF, C32CB, C13ED, T1B, T2E, T2D, T3A, T3F, T1C
    )
    
    print(f"Trial {trial+1}:")
    print(f"  norm1 = {norm1:+.6e}  {'✅' if norm1 > 0 else '❌ NEGATIVE'}")
    print(f"  norm2 = {norm2:+.6e}  {'✅' if norm2 > 0 else '❌ NEGATIVE'}")
    print(f"  norm3 = {norm3:+.6e}  {'✅' if norm3 > 0 else '❌ NEGATIVE'}")
    
    if norm1 <= 0 or norm2 <= 0 or norm3 <= 0:
        all_positive = False
        print(f"  ⚠️  DETECTED NEGATIVE NORMALIZATION!")
    print()

print("="*70)
if all_positive:
    print("✅ ALL normalizations are positive across all trials")
else:
    print("❌ BUG CONFIRMED: Some normalizations are negative")
    print("   This indicates an error in the contraction logic")
    print("   Likely cause: wrong environment tensor passed")
    print("   or incorrect index ordering in torch.mm()")
print("="*70)
