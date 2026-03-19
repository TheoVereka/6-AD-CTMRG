"""
diag_nan_grad.py
================
Trace NaN gradients at the D=2, chi=5→chi=8 transition.

Run:  python src_code/diag_nan_grad.py

What this script checks:
  1. Confirms NaN appears in a..f grads after .backward() at chi=8.
  2. Identifies sqrt(sv) backward NaN in initialize_envCTs_1.
  3. Identifies SVD F-matrix NaN in trunc_rhoCCC (all update steps).
  4. Checks which tensors carry NaN in the autograd graph during CTMRG.
  5. Prints a ranked summary of NaN/Inf sources.
"""

import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
import opt_einsum as oe

# ---------- patch svd_fixed to log every backward --------------------------
import core_unrestricted as core

# Save originals
_orig_svd_fixed = core.svd_fixed
_orig_trunc     = core.trunc_rhoCCC

# Use double precision for cleaner numerics in diagnosis
core.set_dtype(True)
CDTYPE = core.CDTYPE
RDTYPE = core.RDTYPE

# ─────────────────────────────────────────────────────────────────────────────
# Instrumented svd_fixed: records exact call sites where SVD backward blows up
# ─────────────────────────────────────────────────────────────────────────────
_svd_call_log = []   # list of dicts filled in during backward

def _instrumented_svd_fixed(A):
    """Wrap svd_fixed to log SV gaps and detect degenerate/zero SVs."""
    U, S, Vh = _orig_svd_fixed(A)
    if not A.requires_grad:
        return U, S, Vh

    call_idx = len(_svd_call_log)
    _svd_call_log.append({
        'call_idx': call_idx,
        'shape': tuple(A.shape),
        'sv_min': float(S.min().detach()),
        'sv_max': float(S.max().detach()),
        'n_zeros': int((S.detach() < 1e-12).sum()),
        'min_gap': float(torch.diff(S.detach().sort(descending=True).values[:min(len(S), 40)]).abs().min())
                    if len(S) > 1 else float('inf'),
        'backward_nan': False,
        'backward_inf': False,
        'backward_max_abs': float('nan'),
    })
    log_entry = _svd_call_log[-1]

    def _hook_A(grad_A):
        with torch.no_grad():
            has_nan = bool(torch.isnan(grad_A).any())
            has_inf = bool(torch.isinf(grad_A).any())
            finite  = grad_A[torch.isfinite(grad_A)]
            max_abs = float(finite.abs().max()) if finite.numel() > 0 else 0.0
            log_entry['backward_nan'] = has_nan
            log_entry['backward_inf'] = has_inf
            log_entry['backward_max_abs'] = max_abs
        return grad_A   # don't modify; let nans through so we see them

    A.register_hook(_hook_A)   # hook on input A, triggered when grad_A computed
    return U, S, Vh

core.svd_fixed = _instrumented_svd_fixed

# ─────────────────────────────────────────────────────────────────────────────
# Instrumented torch.sqrt: records whether sqrt backward blows up
# ─────────────────────────────────────────────────────────────────────────────
_sqrt_log = []

_orig_sqrt = torch.sqrt
def _instrumented_sqrt(x, **kw):
    out = _orig_sqrt(x, **kw)
    if not isinstance(x, torch.Tensor) or not x.requires_grad:
        return out

    call_idx = len(_sqrt_log)
    _sqrt_log.append({
        'call_idx': call_idx,
        'n_zeros': int((x.detach() < 1e-20).sum()),
        'n_near_zero': int((x.detach() < 1e-8).sum()),
        'x_min': float(x.detach().min()),
        'backward_nan': False,
        'backward_inf': False,
    })
    log_entry = _sqrt_log[-1]

    def _hook_x(grad_x):
        with torch.no_grad():
            log_entry['backward_nan'] = bool(torch.isnan(grad_x).any())
            log_entry['backward_inf'] = bool(torch.isinf(grad_x).any())
        return grad_x

    x.register_hook(_hook_x)
    return out

torch.sqrt = _instrumented_sqrt

# ─────────────────────────────────────────────────────────────────────────────
# Setup: D=2, chi=5 (warm-start), then chi=8
# ─────────────────────────────────────────────────────────────────────────────
D_BOND  = 2
D_PHYS  = 2
D_SQ    = D_BOND ** 2

print("=" * 65)
print("NaN gradient diagnostic: D=2, chi=5→chi=8")
print("=" * 65)

# Random initial tensors (simulating warm-started chi=5 result)
torch.manual_seed(42)
a, b, c, d, e, f = core.initialize_abcdef('random', D_BOND, D_PHYS, 1e-3)
for t in (a, b, c, d, e, f):
    t.requires_grad_(True)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Check grad at chi=5 (should be clean)
# ─────────────────────────────────────────────────────────────────────────────
CHI_SMALL = 5
print(f"\n[Step 1] chi={CHI_SMALL} forward + backward")

Hs = [core.build_heisenberg_H(1.0, D_PHYS)] * 9

aN = core.normalize_single_layer_tensor_for_double_layer(a)
bN = core.normalize_single_layer_tensor_for_double_layer(b)
cN = core.normalize_single_layer_tensor_for_double_layer(c)
dN = core.normalize_single_layer_tensor_for_double_layer(d)
eN = core.normalize_single_layer_tensor_for_double_layer(e)
fN = core.normalize_single_layer_tensor_for_double_layer(f)

A, B, C, Dt, E, F = core.abcdef_to_ABCDEF(aN, bN, cN, dN, eN, fN, D_SQ)
all28 = core.CTMRG_from_init_to_stop(A,B,C,Dt,E,F, CHI_SMALL, D_SQ, 50, 4e-7, False)
(C21CD, C32EF, C13AB, T1F, T2A, T2B, T3C, T3D, T1E,
 C21EB, C32AD, C13CF, T1D, T2C, T2F, T3E, T3B, T1A,
 C21AF, C32CB, C13ED, T1B, T2E, T2D, T3A, T3F, T1C, ctm_steps) = all28

loss5 = (
    core.energy_expectation_nearest_neighbor_3ebadcf_bonds(
        aN,bN,cN,dN,eN,fN, Hs[0],Hs[1],Hs[2], CHI_SMALL, D_BOND, D_PHYS,
        C21CD,C32EF,C13AB,T1F,T2A,T2B,T3C,T3D,T1E)
  + core.energy_expectation_nearest_neighbor_3afcbed_bonds(
        aN,bN,cN,dN,eN,fN, Hs[3],Hs[4],Hs[5], CHI_SMALL, D_BOND, D_PHYS,
        C21EB,C32AD,C13CF,T1D,T2C,T2F,T3E,T3B,T1A)
  + core.energy_expectation_nearest_neighbor_other_3_bonds(
        aN,bN,cN,dN,eN,fN, Hs[6],Hs[7],Hs[8], CHI_SMALL, D_BOND, D_PHYS,
        C21AF,C32CB,C13ED,T1B,T2E,T2D,T3A,T3F,T1C)
)
print(f"  loss5 = {loss5.item():.8f}  (ctm_steps={ctm_steps})")

_svd_call_log.clear(); _sqrt_log.clear()
loss5.backward()

grad_nan_5 = {name: bool(t.grad is not None and torch.isnan(t.grad).any())
              for name, t in zip('abcdef', [a,b,c,d,e,f])}
print(f"  Grad NaN at chi=5: {grad_nan_5}")

svd_nan_5 = [e for e in _svd_call_log if e['backward_nan'] or e['backward_inf']]
sqrt_nan_5 = [e for e in _sqrt_log  if e['backward_nan'] or e['backward_inf']]
print(f"  SVD backward anomalies: {len(svd_nan_5)}/{len(_svd_call_log)}")
print(f"  sqrt backward anomalies: {len(sqrt_nan_5)}/{len(_sqrt_log)}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Re-init a..f with same values, check grad at chi=8
# ─────────────────────────────────────────────────────────────────────────────
CHI_LARGE = 8
print(f"\n[Step 2] chi={CHI_LARGE} forward + backward (warm-started tensors)")

# Detach / re-leaf the tensors
a = a.detach().clone().requires_grad_(True)
b = b.detach().clone().requires_grad_(True)
c = c.detach().clone().requires_grad_(True)
d = d.detach().clone().requires_grad_(True)
e = e.detach().clone().requires_grad_(True)
f = f.detach().clone().requires_grad_(True)

_svd_call_log.clear(); _sqrt_log.clear()

aN = core.normalize_single_layer_tensor_for_double_layer(a)
bN = core.normalize_single_layer_tensor_for_double_layer(b)
cN = core.normalize_single_layer_tensor_for_double_layer(c)
dN = core.normalize_single_layer_tensor_for_double_layer(d)
eN = core.normalize_single_layer_tensor_for_double_layer(e)
fN = core.normalize_single_layer_tensor_for_double_layer(f)

A, B, C, Dt, E, F = core.abcdef_to_ABCDEF(aN, bN, cN, dN, eN, fN, D_SQ)
all28 = core.CTMRG_from_init_to_stop(A,B,C,Dt,E,F, CHI_LARGE, D_SQ, 50, 4e-7, False)
(C21CD, C32EF, C13AB, T1F, T2A, T2B, T3C, T3D, T1E,
 C21EB, C32AD, C13CF, T1D, T2C, T2F, T3E, T3B, T1A,
 C21AF, C32CB, C13ED, T1B, T2E, T2D, T3A, T3F, T1C, ctm_steps) = all28

loss8 = (
    core.energy_expectation_nearest_neighbor_3ebadcf_bonds(
        aN,bN,cN,dN,eN,fN, Hs[0],Hs[1],Hs[2], CHI_LARGE, D_BOND, D_PHYS,
        C21CD,C32EF,C13AB,T1F,T2A,T2B,T3C,T3D,T1E)
  + core.energy_expectation_nearest_neighbor_3afcbed_bonds(
        aN,bN,cN,dN,eN,fN, Hs[3],Hs[4],Hs[5], CHI_LARGE, D_BOND, D_PHYS,
        C21EB,C32AD,C13CF,T1D,T2C,T2F,T3E,T3B,T1A)
  + core.energy_expectation_nearest_neighbor_other_3_bonds(
        aN,bN,cN,dN,eN,fN, Hs[6],Hs[7],Hs[8], CHI_LARGE, D_BOND, D_PHYS,
        C21AF,C32CB,C13ED,T1B,T2E,T2D,T3A,T3F,T1C)
)
print(f"  loss8 = {loss8.item():.8f}  (ctm_steps={ctm_steps})")

n_svd_forward = len(_svd_call_log)
n_sqrt_forward = len(_sqrt_log)

# ── Run backward with anomaly detection ──────────────────────────────────────
torch.autograd.set_detect_anomaly(True)
try:
    loss8.backward()
    anomaly_triggered = False
    anomaly_msg = ""
except RuntimeError as exc:
    anomaly_triggered = True
    anomaly_msg = str(exc)
    print(f"\n  [anomaly] RuntimeError during backward:\n  {anomaly_msg[:500]}")
torch.autograd.set_detect_anomaly(False)

grad_nan_8 = {name: bool(t.grad is not None and torch.isnan(t.grad).any())
              for name, t in zip('abcdef', [a,b,c,d,e,f])}
grad_inf_8 = {name: bool(t.grad is not None and torch.isinf(t.grad).any())
              for name, t in zip('abcdef', [a,b,c,d,e,f])}
print(f"  Grad NaN at chi=8: {grad_nan_8}")
print(f"  Grad Inf at chi=8: {grad_inf_8}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: SVD backward audit
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[Step 3] SVD backward audit ({n_svd_forward} SVDs in forward graph)")

svd_all = _svd_call_log
svd_bad  = [e for e in svd_all if e['backward_nan'] or e['backward_inf']]
svd_zero = [e for e in svd_all if e['n_zeros'] > 0]
svd_small_gap = [e for e in svd_all if e['min_gap'] < 1e-8]

print(f"  SVDs with degenerate/zero SVs:   {len(svd_zero)}")
print(f"  SVDs with small SV gap (<1e-8):  {len(svd_small_gap)}")
print(f"  SVDs with NaN/Inf backward:      {len(svd_bad)}")

if svd_bad:
    print("\n  --- PROBLEMATIC SVDs ---")
    for e in svd_bad:
        print(f"    call #{e['call_idx']:3d}: shape={e['shape']}, "
              f"sv_min={e['sv_min']:.3e}, n_zeros={e['n_zeros']}, "
              f"min_gap={e['min_gap']:.3e}, "
              f"max|grad_A|={e['backward_max_abs']:.3e}, "
              f"NaN={e['backward_nan']}, Inf={e['backward_inf']}")

if svd_zero:
    print("\n  --- SVDs with zero/near-zero singular values ---")
    for e in svd_zero[:10]:
        print(f"    call #{e['call_idx']:3d}: shape={e['shape']}, "
              f"n_zeros={e['n_zeros']}, sv_min={e['sv_min']:.3e}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: sqrt backward audit
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[Step 4] sqrt backward audit ({n_sqrt_forward} sqrt calls in forward graph)")

sqrt_bad  = [e for e in _sqrt_log if e['backward_nan'] or e['backward_inf']]
sqrt_zero = [e for e in _sqrt_log if e['n_zeros'] > 0]

print(f"  sqrt calls with zero inputs:     {len(sqrt_zero)}")
print(f"  sqrt calls with NaN/Inf backward:{len(sqrt_bad)}")

if sqrt_zero:
    print("\n  --- sqrt calls with zero/near-zero inputs ---")
    for e in sqrt_zero:
        print(f"    call #{e['call_idx']:3d}: n_zeros={e['n_zeros']}, "
              f"n_near_zero={e['n_near_zero']}, x_min={e['x_min']:.3e}, "
              f"NaN={e['backward_nan']}, Inf={e['backward_inf']}")

if sqrt_bad:
    print("\n  --- PROBLEMATIC sqrt calls ---")
    for e in sqrt_bad:
        print(f"    call #{e['call_idx']:3d}: n_zeros={e['n_zeros']}, "
              f"x_min={e['x_min']:.3e}, "
              f"NaN={e['backward_nan']}, Inf={e['backward_inf']}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Directly probe the rank of the three outer-product matrices
#         used in initialize_envCTs_1 (out2Ain3D, out3Cin1F, out1Ein2B)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[Step 5] Probe rank of outer-product matrices in initialize_envCTs_1 (chi={CHI_LARGE})")

with torch.no_grad():
    a_d = a.detach(); b_d = b.detach(); c_d = c.detach()
    d_d = d.detach(); e_d = e.detach(); f_d = f.detach()

    aN_d = core.normalize_single_layer_tensor_for_double_layer(a_d)
    bN_d = core.normalize_single_layer_tensor_for_double_layer(b_d)
    cN_d = core.normalize_single_layer_tensor_for_double_layer(c_d)
    dN_d = core.normalize_single_layer_tensor_for_double_layer(d_d)
    eN_d = core.normalize_single_layer_tensor_for_double_layer(e_d)
    fN_d = core.normalize_single_layer_tensor_for_double_layer(f_d)
    A_d, B_d, C_d, Dt_d, E_d, F_d = core.abcdef_to_ABCDEF(aN_d, bN_d, cN_d, dN_d, eN_d, fN_d, D_SQ)

    D_bond_loc = round(D_SQ ** 0.5)

    # Reproduce the T projections from initialize_envCTs_1
    BC2323 = oe.contract("iBG,ibg->BGbg", B_d, C_d, optimize=[(0,1)], backend='torch')
    DE3131 = oe.contract("AiG,aig->GAga", Dt_d, E_d, optimize=[(0,1)], backend='torch')
    FA1212 = oe.contract("ABi,abi->ABab", F_d, A_d, optimize=[(0,1)], backend='torch')

    A23=A_d.reshape(D_bond_loc,D_bond_loc,D_SQ,D_SQ).diagonal(dim1=0,dim2=1).sum(-1)
    F31=F_d.permute(1,2,0).reshape(D_bond_loc,D_bond_loc,D_SQ,D_SQ).diagonal(dim1=0,dim2=1).sum(-1)
    C31=C_d.permute(1,2,0).reshape(D_bond_loc,D_bond_loc,D_SQ,D_SQ).diagonal(dim1=0,dim2=1).sum(-1)
    B12=B_d.permute(2,0,1).reshape(D_bond_loc,D_bond_loc,D_SQ,D_SQ).diagonal(dim1=0,dim2=1).sum(-1)
    E12=E_d.permute(2,0,1).reshape(D_bond_loc,D_bond_loc,D_SQ,D_SQ).diagonal(dim1=0,dim2=1).sum(-1)
    D23=Dt_d.reshape(D_bond_loc,D_bond_loc,D_SQ,D_SQ).diagonal(dim1=0,dim2=1).sum(-1)
    E23=E_d.reshape(D_bond_loc,D_bond_loc,D_SQ,D_SQ).diagonal(dim1=0,dim2=1).sum(-1)
    B31=B_d.permute(1,2,0).reshape(D_bond_loc,D_bond_loc,D_SQ,D_SQ).diagonal(dim1=0,dim2=1).sum(-1)
    A31=A_d.permute(1,2,0).reshape(D_bond_loc,D_bond_loc,D_SQ,D_SQ).diagonal(dim1=0,dim2=1).sum(-1)
    D12=Dt_d.permute(2,0,1).reshape(D_bond_loc,D_bond_loc,D_SQ,D_SQ).diagonal(dim1=0,dim2=1).sum(-1)
    C12=C_d.permute(2,0,1).reshape(D_bond_loc,D_bond_loc,D_SQ,D_SQ).diagonal(dim1=0,dim2=1).sum(-1)
    F23=F_d.reshape(D_bond_loc,D_bond_loc,D_SQ,D_SQ).diagonal(dim1=0,dim2=1).sum(-1)
    C21CD_raw = oe.contract("yj,ijYk,in,nm,kXlm,lx->YyXx",
                        E23,BC2323,A23,F31,DE3131,B31,
                        optimize=[(0,1),(0,1),(0,1),(0,1),(0,1)],backend='torch'
                        ).reshape(D_SQ*D_SQ,D_SQ*D_SQ)
    C32EF_raw = oe.contract("zj,ijZk,in,nm,kYlm,ly->ZzYy",
                        A31,DE3131,C31,B12,FA1212,D12,
                        optimize=[(0,1),(0,1),(0,1),(0,1),(0,1)],backend='torch'
                        ).reshape(D_SQ*D_SQ,D_SQ*D_SQ)
    C13AB_raw = oe.contract("xj,ijXk,in,nm,kZlm,lz->XxZz",
                        C12,FA1212,E12,D23,BC2323,F23,
                        optimize=[(0,1),(0,1),(0,1),(0,1),(0,1)],backend='torch'
                        ).reshape(D_SQ*D_SQ,D_SQ*D_SQ)

    A12=A_d.permute(2,0,1).reshape(D_bond_loc,D_bond_loc,D_SQ,D_SQ).diagonal(dim1=0,dim2=1).sum(-1)
    B23=B_d.reshape(D_bond_loc,D_bond_loc,D_SQ,D_SQ).diagonal(dim1=0,dim2=1).sum(-1)
    C23=C_d.reshape(D_bond_loc,D_bond_loc,D_SQ,D_SQ).diagonal(dim1=0,dim2=1).sum(-1)
    D31=Dt_d.permute(1,2,0).reshape(D_bond_loc,D_bond_loc,D_SQ,D_SQ).diagonal(dim1=0,dim2=1).sum(-1)
    E31=E_d.permute(1,2,0).reshape(D_bond_loc,D_bond_loc,D_SQ,D_SQ).diagonal(dim1=0,dim2=1).sum(-1)
    F12=F_d.permute(2,0,1).reshape(D_bond_loc,D_bond_loc,D_SQ,D_SQ).diagonal(dim1=0,dim2=1).sum(-1)

    # SVD of rho matrices BEFORE truncation
    rho32 = oe.contract("UZ,ZY,YV->UV",C13AB_raw,C32EF_raw,C21CD_raw,
                        optimize=[(0,1),(0,1)],backend='torch')
    _U32, sv32, _V32 = torch.linalg.svd(rho32)
    rho13 = oe.contract("UX,XZ,ZV->UV",C21CD_raw,C13AB_raw,C32EF_raw,
                        optimize=[(0,1),(0,1)],backend='torch')
    _U13, sv13, _V13 = torch.linalg.svd(rho13)
    rho21 = oe.contract("UY,YX,XV->UV",C32EF_raw,C21CD_raw,C13AB_raw,
                        optimize=[(0,1),(0,1)],backend='torch')
    _U21, sv21, _V21 = torch.linalg.svd(rho21)

    chi_l = CHI_LARGE
    print(f"  rho32 ({rho32.shape}): sv[:chi+4]={sv32[:chi_l+4].tolist()}")
    print(f"    min_gap in sv[:chi]={torch.diff(sv32[:chi_l].sort(descending=True).values).abs().min():.3e}")
    print(f"  rho13 ({rho13.shape}): sv[:chi+4]={sv13[:chi_l+4].tolist()}")
    print(f"    min_gap in sv[:chi]={torch.diff(sv13[:chi_l].sort(descending=True).values).abs().min():.3e}")
    print(f"  rho21 ({rho21.shape}): sv[:chi+4]={sv21[:chi_l+4].tolist()}")
    print(f"    min_gap in sv[:chi]={torch.diff(sv21[:chi_l].sort(descending=True).values).abs().min():.3e}")

    # Now compute the T matrices and their outer products
    # Reproduce the first projection step
    U3F, svU3F, V2E_raw = torch.linalg.svd(rho32)
    U3F = U3F[:,:chi_l].conj()
    V2E = V2E_raw[:chi_l,:].conj()
    U1B, svU1B, V3A_raw = torch.linalg.svd(rho13)
    U1B = U1B[:,:chi_l].conj()
    V3A = V3A_raw[:chi_l,:].conj()
    U2D, svU2D, V1C_raw = torch.linalg.svd(rho21)
    U2D = U2D[:,:chi_l].conj()
    V1C = V1C_raw[:chi_l,:].conj()

    T1F_raw = oe.contract("mi,jyi,aYjM->MmYya",C23,Dt_d,FA1212,optimize=[(0,1),(0,1)],backend='torch'
                ).reshape(D_SQ*D_SQ,D_SQ*D_SQ,D_SQ)
    T2A_raw = oe.contract("il,xji,LjXb->LlXxb",D31,C_d,FA1212,optimize=[(0,1),(0,1)],backend='torch'
                ).reshape(D_SQ*D_SQ,D_SQ*D_SQ,D_SQ)
    T2B_raw = oe.contract("ni,ijz,bZjN->NnZzb",E31,F_d,BC2323,optimize=[(0,1),(0,1)],backend='torch'
                ).reshape(D_SQ*D_SQ,D_SQ*D_SQ,D_SQ)
    T3C_raw = oe.contract("im,iyj,MjYg->MmYyg",F12,E_d,BC2323,optimize=[(0,1),(0,1)],backend='torch'
                ).reshape(D_SQ*D_SQ,D_SQ*D_SQ,D_SQ)
    T3D_raw = oe.contract("li,xij,gXjL->LlXxg",A12,B_d,DE3131,optimize=[(0,1),(0,1)],backend='torch'
                ).reshape(D_SQ*D_SQ,D_SQ*D_SQ,D_SQ)
    T1E_raw = oe.contract("in,jiz,NjZa->NnZza",B23,A_d,DE3131,optimize=[(0,1),(0,1)],backend='torch'
                ).reshape(D_SQ*D_SQ,D_SQ*D_SQ,D_SQ)

    T1F_p = oe.contract("MYa,yY->Mya",T1F_raw,V3A,optimize=[(0,1)],backend='torch'
                ).reshape(D_SQ*D_SQ,chi_l*D_SQ)
    T2A_p = oe.contract("LXb,Xx->Lxb",T2A_raw,U3F,optimize=[(0,1)],backend='torch'
                ).reshape(D_SQ*D_SQ,chi_l*D_SQ)
    T2B_p = oe.contract("NZa,zZ->Nza",T2B_raw,V1C,optimize=[(0,1)],backend='torch'
                ).reshape(D_SQ*D_SQ,chi_l*D_SQ)
    T3C_p = oe.contract("MYg,Yy->Myg",T3C_raw,U1B,optimize=[(0,1)],backend='torch'
                ).reshape(D_SQ*D_SQ,chi_l*D_SQ)
    T3D_p = oe.contract("LXg,xX->Lxg",T3D_raw,V2E,optimize=[(0,1)],backend='torch'
                ).reshape(D_SQ*D_SQ,chi_l*D_SQ)
    T1E_p = oe.contract("NZa,Zz->Nza",T1E_raw,U2D,optimize=[(0,1)],backend='torch'
                ).reshape(D_SQ*D_SQ,chi_l*D_SQ)

    out2Ain3D = torch.mm(T2A_p.T, T3D_p)   # (chi*D_sq) x (chi*D_sq)
    out3Cin1F = torch.mm(T3C_p.T, T1F_p)
    out1Ein2B = torch.mm(T1E_p.T, T2B_p)

    _, svL, _ = torch.linalg.svd(out2Ain3D)
    _, svM, _ = torch.linalg.svd(out3Cin1F)
    _, svN, _ = torch.linalg.svd(out1Ein2B)

    print(f"\n  out2Ain3D ({out2Ain3D.shape}): rank estimate, sv[:chi+4]={svL[:chi_l+4].tolist()}")
    print(f"  out3Cin1F ({out3Cin1F.shape}): sv[:chi+4]={svM[:chi_l+4].tolist()}")
    print(f"  out1Ein2B ({out1Ein2B.shape}): sv[:chi+4]={svN[:chi_l+4].tolist()}")

    zero_svL = (svL[:chi_l] < 1e-12).sum().item()
    zero_svM = (svM[:chi_l] < 1e-12).sum().item()
    zero_svN = (svN[:chi_l] < 1e-12).sum().item()
    print(f"\n  Zero SVs in svL[:chi]={zero_svL}, svM[:chi]={zero_svM}, svN[:chi]={zero_svN}")
    print(f"  sqrt(svL[:chi]) min = {svL[:chi_l].sqrt().min():.3e}")
    print(f"  sqrt backward sensitivity ~ 1/(2*sqrt(sv_min)) = {1/(2*max(svL[:chi_l].min().sqrt().item(), 1e-30)):.3e}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: Final summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)
any_nan_8 = any(grad_nan_8.values())
any_inf_8 = any(grad_inf_8.values())
print(f"chi=5 backward: {'CLEAN' if not any(grad_nan_5.values()) else 'NaN PRESENT'}")
print(f"chi=8 backward: {'NaN in grads' if any_nan_8 else ''} {'Inf in grads' if any_inf_8 else ''}")
if not any_nan_8 and not any_inf_8:
    print("chi=8 backward: CLEAN")

print(f"\nNaN/Inf SVD backward calls at chi=8: {len(svd_bad)}")
print(f"NaN/Inf sqrt backward calls at chi=8: {len(sqrt_bad)}")

print("\nTop NaN sources:")
if sqrt_bad:
    for e in sqrt_bad[:3]:
        print(f"  [sqrt]  call #{e['call_idx']}, n_zeros={e['n_zeros']}, x_min={e['x_min']:.3e}")
if svd_bad:
    for e in svd_bad[:5]:
        print(f"  [svd]   call #{e['call_idx']}, shape={e['shape']}, "
              f"n_zeros={e['n_zeros']}, min_gap={e['min_gap']:.3e}, "
              f"max|grad|={e['backward_max_abs']:.3e}")

print("\nDone. See full output above for details.")
