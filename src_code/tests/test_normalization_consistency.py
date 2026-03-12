"""
Check if the three energy functions have consistent normalization factors.
If norm1 ≠ norm2 ≠ norm3, that indicates state inconsistency.
"""

import sys
sys.path.append('/home/chye/6ADctmrg/6-AD-CTMRG/src_code/src')
sys.path.append('/home/chye/6ADctmrg/6-AD-CTMRG/src_code/scripts')

import torch
import numpy as np
from core_unrestricted import (
    normalize_tensor, initialize_abcdef, abcdef_to_ABCDEF,
    CTMRG_from_init_to_stop, set_dtype
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
print("Testing normalization consistency across three energy functions")
print("="*70)

# Initialize normalized tensors
a, b, c, d, e, f = initialize_abcdef('random', D_bond, d_phys, 0.0)
for t in [a, b, c, d, e, f]:
    t.data = normalize_tensor(t.data)

# Build environment
with torch.no_grad():
    A, B, C, D, E, F = abcdef_to_ABCDEF(a, b, c, d, e, f, D_sq)
    (C21CD, C32EF, C13AB, T1F,  T2A,  T2B,  T3C,  T3D,  T1E,
     C21EB, C32AD, C13CF, T1D,  T2C,  T2F,  T3E,  T3B,  T1A,
     C21AF, C32CB, C13ED, T1B,  T2E,  T2D,  T3A,  T3F,  T1C,
     ctm_steps) = \
        CTMRG_from_init_to_stop(A, B, C, D, E, F, chi, D_sq, 100, 1e-10)

print(f"CTMRG converged in {ctm_steps} steps\n")

# Manually compute what each function does, extracting the normalization
def compute_energy_and_norm_ebadcf(a,b,c,d,e,f, Heb,Had,Hcf, chi, D_bond,
                                    C21CD,C32EF,C13AB,T1F,T2A,T2B,T3C,T3D,T1E):
    T1F = T1F.reshape(chi,chi,D_bond,D_bond)
    T2A = T2A.reshape(chi,chi,D_bond,D_bond)
    T2B = T2B.reshape(chi,chi,D_bond,D_bond)
    T3C = T3C.reshape(chi,chi,D_bond,D_bond)
    T3D = T3D.reshape(chi,chi,D_bond,D_bond)
    T1E = T1E.reshape(chi,chi,D_bond,D_bond)

    open_E = oe.contract("YX,MYar,abci,rstj->MbsXctij", C21CD, T1F, e, e.conj(), optimize=[(0,1),(0,1),(0,1)], backend="torch")
    open_D = oe.contract("MYct,abci,rstj->YarMbsij", T3C, d, d.conj(), optimize=[(0,1),(0,1)], backend="torch")
    open_A = oe.contract("ZY,NZbs,abci,rstj->NctYarij", C32EF, T2B, a, a.conj(), optimize=[(0,1),(0,1),(0,1)], backend="torch")
    open_F = oe.contract("NZar,abci,rstj->ZbsNctij", T1E, f, f.conj(), optimize=[(0,1),(0,1)], backend="torch")
    open_C = oe.contract("XZ,LXct,abci,rstj->LarZbsij", C13AB, T3D, c, c.conj(), optimize=[(0,1),(0,1),(0,1)], backend="torch")
    open_B = oe.contract("LXbs,abci,rstj->XctLarij", T2A, b, b.conj(), optimize=[(0,1),(0,1)], backend="torch")
                              
    closed_E = oe.contract("MbsXctii->MbsXct", open_E, backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    closed_D = oe.contract("YarMbsii->YarMbs", open_D, backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    closed_A = oe.contract("NctYarii->NctYar", open_A, backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    closed_F = oe.contract("ZbsNctii->ZbsNct", open_F, backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    closed_C = oe.contract("LarZbsii->LarZbs", open_C, backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    closed_B = oe.contract("XctLarii->XctLar", open_B, backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)

    H_AD = oe.contract("NctYarij,ijkl,YarMbskl->NctMbs", open_A, Had, open_D, optimize=[(0,1),(0,1)], backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    H_CF = oe.contract("LarZbsij,ijkl,ZbsNctkl->LarNct", open_C, Hcf, open_F, optimize=[(0,1),(0,1)], backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    H_EB = oe.contract("MbsXctij,ijkl,XctLarkl->MbsLar", open_E, Heb, open_B, optimize=[(0,1),(0,1)], backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)

    AD = torch.mm(closed_A, closed_D)
    CF = torch.mm(closed_C, closed_F)
    EB = torch.mm(closed_E, closed_B)

    E_unnormed_AD = oe.contract("xy,yz,zx->", H_AD, EB, CF, backend="torch")
    E_unnormed_CF = oe.contract("xy,yz,zx->", H_CF, AD, EB, backend="torch")
    E_unnormed_EB = oe.contract("xy,yz,zx->", H_EB, CF, AD, backend="torch")

    norm = oe.contract("xy,yz,zx->", AD, EB, CF, backend="torch")
    energy = (E_unnormed_AD + E_unnormed_CF + E_unnormed_EB) / norm
    return energy, norm

# Similarly for the other two functions (abbreviated copies)
def compute_energy_and_norm_afcbed(a,b,c,d,e,f, Haf,Hcb,Hed, chi, D_bond,
                                    C21AF,C32CB,C13ED,T1B,T2E,T2D,T3A,T3F,T1C):
    T1B = T1B.reshape(chi,chi,D_bond,D_bond)
    T2E = T2E.reshape(chi,chi,D_bond,D_bond)
    T2D = T2D.reshape(chi,chi,D_bond,D_bond)
    T3A = T3A.reshape(chi,chi,D_bond,D_bond)
    T3F = T3F.reshape(chi,chi,D_bond,D_bond)
    T1C = T1C.reshape(chi,chi,D_bond,D_bond)
    
    open_A = oe.contract("YX,MYct,abci,rstj->MbsXarij", C21AF, T1B, a, a.conj(),optimize=[(0,1),(0,1),(0,1)],backend="torch")
    open_F = oe.contract("MYbs,abci,rstj->YctMarij", T3A, f, f.conj(),optimize=[(0,1),(0,1)],backend="torch")
    open_C = oe.contract("ZY,NZar,abci,rstj->NbsYctij", C32CB, T2D, c, c.conj(),optimize=[(0,1),(0,1),(0,1)],backend="torch")
    open_B = oe.contract("NZct,abci,rstj->ZarNbsij", T1C, b, b.conj(),optimize=[(0,1),(0,1)],backend="torch")
    open_E = oe.contract("XZ,LXbs,abci,rstj->LctZarij", C13ED, T3F, e, e.conj(),optimize=[(0,1),(0,1),(0,1)],backend="torch")
    open_D = oe.contract("LXar,abci,rstj->XbsLctij", T2E, d, d.conj(),optimize=[(0,1),(0,1)],backend="torch")
    
    closed_A = oe.contract("MbsXarii->MbsXar", open_A,backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    closed_F = oe.contract("YctMarij->YctMar", open_F,backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    closed_C = oe.contract("NbsYctii->NbsYct", open_C,backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    closed_B = oe.contract("ZarNbsii->ZarNbs", open_B,backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    closed_E = oe.contract("LctZarii->LctZar", open_E,backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    closed_D = oe.contract("XbsLctii->XbsLct", open_D,backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    
    H_AF = oe.contract("MbsXarij,ijkl,YctMarkl->MbsYct", open_A, Haf, open_F,optimize=[(0,1),(0,1)],backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    H_CB = oe.contract("NbsYctij,ijkl,ZarNbskl->ZarYct", open_C, Hcb, open_B,optimize=[(0,1),(0,1)],backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    H_ED = oe.contract("LctZarij,ijkl,XbsLctkl->XbsZar", open_E, Hed, open_D,optimize=[(0,1),(0,1)],backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    
    AF = torch.mm(closed_A, closed_F)
    CB = torch.mm(closed_C, closed_B)
    ED = torch.mm(closed_E, closed_D)
    
    E_unnormed_AF = oe.contract("xy,yz,zx->", H_AF, CB, ED,backend="torch")
    E_unnormed_CB = oe.contract("xy,yz,zx->", H_CB, ED, AF,backend="torch")
    E_unnormed_ED = oe.contract("xy,yz,zx->", H_ED, AF, CB,backend="torch")
    
    norm = oe.contract("xy,yz,zx->", AF, CB, ED,backend="torch")
    energy = (E_unnormed_AF + E_unnormed_CB + E_unnormed_ED) / norm
    return energy, norm

def compute_energy_and_norm_other(a,b,c,d,e,f, Hcd,Hef,Hab, chi, D_bond,
                                    C21EB,C32AD,C13CF,T1D,T2C,T2F,T3E,T3B,T1A):
    T1D = T1D.reshape(chi,chi,D_bond,D_bond)
    T2C = T2C.reshape(chi,chi,D_bond,D_bond)
    T2F = T2F.reshape(chi,chi,D_bond,D_bond)
    T3E = T3E.reshape(chi,chi,D_bond,D_bond)
    T3B = T3B.reshape(chi,chi,D_bond,D_bond)
    T1A = T1A.reshape(chi,chi,D_bond,D_bond)
    
    open_C = oe.contract("YX,MYbs,abci,rstj->MarXctij", C21EB, T1D, c, c.conj(),optimize=[(0,1),(0,1),(0,1)],backend="torch")
    open_D = oe.contract("MYar,abci,rstj->YbsMctij", T3E, d, d.conj(),optimize=[(0,1),(0,1)],backend="torch")
    open_E = oe.contract("ZY,NZct,abci,rstj->NarYbsij", C32AD, T2F, e, e.conj(),optimize=[(0,1),(0,1),(0,1)],backend="torch")
    open_F = oe.contract("NZbs,abci,rstj->ZctNarij", T1A, f, f.conj(),optimize=[(0,1),(0,1)],backend="torch")
    open_A = oe.contract("XZ,LXar,abci,rstj->LbsZctij", C13CF, T3B, a, a.conj(),optimize=[(0,1),(0,1),(0,1)],backend="torch")
    open_B = oe.contract("LXct,abci,rstj->XarLbsij", T2C, b, b.conj(),optimize=[(0,1),(0,1)],backend="torch")
    
    closed_C = oe.contract("MarXctii->MarXct", open_C,backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    closed_D = oe.contract("YbsMctii->YbsMct", open_D,backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    closed_E = oe.contract("NarYbsii->NarYbs", open_E,backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    closed_F = oe.contract("ZctNarij->ZctNar", open_F,backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    closed_A = oe.contract("LbsZctii->LbsZct", open_A,backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    closed_B = oe.contract("XarLbsii->XarLbs", open_B,backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    
    H_CD = oe.contract("MarXctij,ijkl,YbsMctkl->MarYbs", open_C, Hcd, open_D,optimize=[(0,1),(0,1)],backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    H_EF = oe.contract("NarYbsij,ijkl,ZctNarkl->ZctYbs", open_E, Hef, open_F,optimize=[(0,1),(0,1)],backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    H_AB = oe.contract("LbsZctij,ijkl,XarLbskl->XarZct", open_A, Hab, open_B,optimize=[(0,1),(0,1)],backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    
    CD = torch.mm(closed_C, closed_D)
    EF = torch.mm(closed_E, closed_F)
    AB = torch.mm(closed_A, closed_B)
    
    E_unnormed_CD = oe.contract("xy,yz,zx->", H_CD, EF, AB,backend="torch")
    E_unnormed_EF = oe.contract("xy,yz,zx->", H_EF, AB, CD,backend="torch")
    E_unnormed_AB = oe.contract("xy,yz,zx->", H_AB, CD, EF,backend="torch")
    
    norm = oe.contract("xy,yz,zx->", CD, EF, AB,backend="torch")
    energy = (E_unnormed_CD + E_unnormed_EF + E_unnormed_AB) / norm
    return energy, norm

# Compute all three with their normalizations
E1, norm1 = compute_energy_and_norm_ebadcf(
    a,b,c,d,e,f, H,H,H, chi, D_bond,
    C21CD,C32EF,C13AB,T1F,T2A,T2B,T3C,T3D,T1E
)

E2, norm2 = compute_energy_and_norm_afcbed(
    a,b,c,d,e,f, H,H,H, chi, D_bond,
    C21AF,C32CB,C13ED,T1B,T2E,T2D,T3A,T3F,T1C
)

E3, norm3 = compute_energy_and_norm_other(
    a,b,c,d,e,f, H,H,H, chi, D_bond,
    C21EB,C32AD,C13CF,T1D,T2C,T2F,T3E,T3B,T1A
)

print("Function 1 (ebadcf bonds EB,AD,CF):")
print(f"  Normalization: {norm1.real.item():.10e}")
print(f"  Energy (3 bonds): {E1.real.item():.6f}")
print(f"  Energy per bond: {E1.real.item()/3:.6f}")

print("\nFunction 2 (afcbed bonds AF,CB,ED):")
print(f"  Normalization: {norm2.real.item():.10e}")
print(f"  Energy (3 bonds): {E2.real.item():.6f}")
print(f"  Energy per bond: {E2.real.item()/3:.6f}")

print("\nFunction 3 (other bonds CD,EF,AB):")
print(f"  Normalization: {norm3.real.item():.10e}")
print(f"  Energy (3 bonds): {E3.real.item():.6f}")
print(f"  Energy per bond: {E3.real.item()/3:.6f}")

print("\nTotal energy:", (E1 + E2 + E3).real.item())
print("Total energy per bond:", (E1 + E2 + E3).real.item() / 9)

print("\n" + "="*70)
print("NORMALIZATION CONSISTENCY CHECK")
print("="*70)

norm1_val = norm1.real.item()
norm2_val = norm2.real.item()
norm3_val = norm3.real.item()

rel_diff_12 = abs(norm1_val - norm2_val) / max(abs(norm1_val), abs(norm2_val))
rel_diff_23 = abs(norm2_val - norm3_val) / max(abs(norm2_val), abs(norm3_val))
rel_diff_31 = abs(norm3_val - norm1_val) / max(abs(norm3_val), abs(norm1_val))

print(f"Relative difference norm1 vs norm2: {rel_diff_12:.3e}")
print(f"Relative difference norm2 vs norm3: {rel_diff_23:.3e}")
print(f"Relative difference norm3 vs norm1: {rel_diff_31:.3e}")

if max(rel_diff_12, rel_diff_23, rel_diff_31) < 1e-6:
    print("\n✅ Normalizations are CONSISTENT (diff < 1e-6)")
    print("   This is expected for a properly converged CTMRG state.")
else:
    print(f"\n⚠️  Normalizations are INCONSISTENT (max diff = {max(rel_diff_12, rel_diff_23, rel_diff_31):.3e})")
    print("   This suggests the CTMRG environment doesn't match the current tensors.")
    print("   Could be the source of unphysical energy values during optimization!")
