#!/usr/bin/env python3
"""
test_memory.py — GPU memory diagnostic for AD-CTMRG

Runs D=5, chi=45 (neel_legacy ansatz) for a configurable number of Adam steps
and prints detailed memory stats at every sub-step.

Usage (from src_code/scripts/):
    python test_memory.py             # D=5 chi=45 (default), 5 Adam steps
    python test_memory.py --D 6 --chi 60 --steps 3

Scale reference (rule of thumb: mem ~ D^4 * chi^2):
    D=5,chi=45  → scale 1.0×
    D=6,chi=60  → scale 3.8×
    D=7,chi=77  → scale 10.9×
    D=8,chi=88  → scale 25.2×

So if D=5 uses X MB, D=8,chi=88 would use ~25X MB.
"""

import os, sys, gc, argparse, time

# ── path so we can import core.py from the same directory ────────────────────
sys.path.insert(0, os.path.dirname(__file__))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F

# ── Parse args ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--D',     type=int, default=5)
parser.add_argument('--chi',   type=int, default=45)
parser.add_argument('--steps', type=int, default=5,  help='Adam steps')
parser.add_argument('--J2',    type=float, default=0.26)
parser.add_argument('--cpu',   action='store_true',   help='force CPU')
args = parser.parse_args()

D_BOND  = args.D
CHI     = args.chi
N_STEPS = args.steps
J2      = args.J2

# ── Device / dtype ────────────────────────────────────────────────────────────
USE_GPU = torch.cuda.is_available() and not args.cpu
DEVICE  = torch.device('cuda' if USE_GPU else 'cpu')
torch.set_num_threads(1)

print(f"Device : {DEVICE}")
print(f"D={D_BOND}  chi={CHI}  J2={J2}  steps={N_STEPS}")
print()

# ── Memory helpers ────────────────────────────────────────────────────────────
def mem(tag: str, reset_peak: bool = False):
    if not USE_GPU:
        print(f"  [MEM] {tag}  (CPU — no GPU stats)")
        return
    alloc   = torch.cuda.memory_allocated() / 1024**2
    reserv  = torch.cuda.memory_reserved()  / 1024**2
    peak    = torch.cuda.max_memory_allocated() / 1024**2
    print(f"  [MEM] {tag:50s}  alloc={alloc:7.1f}MB  reserv={reserv:7.1f}MB  peak={peak:7.1f}MB")
    if reset_peak:
        torch.cuda.reset_peak_memory_stats()

def mem_delta(tag: str, before_alloc: float):
    if not USE_GPU:
        return
    alloc  = torch.cuda.memory_allocated() / 1024**2
    delta  = alloc - before_alloc
    reserv = torch.cuda.memory_reserved() / 1024**2
    peak   = torch.cuda.max_memory_allocated() / 1024**2
    sign   = '+' if delta >= 0 else ''
    print(f"  [MEM] {tag:50s}  alloc={alloc:7.1f}MB ({sign}{delta:+.1f}MB)  reserv={reserv:7.1f}MB  peak={peak:7.1f}MB")

def cur_alloc() -> float:
    return torch.cuda.memory_allocated() / 1024**2 if USE_GPU else 0.0


# ── Import core ───────────────────────────────────────────────────────────────
import core as _core
import core
from core import (
    set_dtype, set_device,
    normalize_single_layer_tensor_for_double_layer,
    abcdef_to_ABCDEF,
    CTMRG_from_init_to_stop,
    build_heisenberg_H,
    energy_expectation_nearest_neighbor_3ebadcf_bonds,
    energy_expectation_nearest_neighbor_3afcbed_bonds,
    energy_expectation_nearest_neighbor_other_3_bonds,
    symmetrize_virtual_legs,
    neel_abcdef_from_a,
    initialize_neel,
)
from torch.utils.checkpoint import checkpoint as _ckpt

set_dtype(use_double=True, use_real=True)
set_device(DEVICE)

# ── Build Hamiltonian ─────────────────────────────────────────────────────────
# Core energy functions take:
#   energy_expectation_nearest_neighbor_3ebadcf_bonds(
#       a,b,c,d,e,f,
#       Jeb,Jad,Jcf, Jfa,Jde,Jbc, Jae,Jec,Jca,Jdb,Jbf,Jfd,   ← 12 scalar J-values
#       SdotS,    ← (d,d,d,d) spin-operator tensor  (d²×d² real)
#       chi, D_bond, d_PHYS, *env9)
# Js is a flat list of 36 scalar coupling constants:
#   Js = ([J1]*6 + [J2]*6) * 3   (matches main.py line 2348)
# The 12 scalars per env group map to the 9 nn bonds (J1) + 3 nnn bonds (J2).
J1    = 1.0
d_PHYS = 2

Js_list = ([J1] * 6 + [J2] * 6) * 3   # length 36 — matches main.py

# SdotS operator (real, S+/S- formulation, same dtype as tensors)
# build_heisenberg_H returns a (d,d,d,d) complex tensor; core energy functions
# want a real one when TENSORDTYPE is float64.  Build directly:
_sp   = torch.tensor([[0.,1.],[0.,0.]], dtype=_core.TENSORDTYPE, device=DEVICE)
_sm   = torch.tensor([[0.,0.],[1.,0.]], dtype=_core.TENSORDTYPE, device=DEVICE)
_sz   = torch.tensor([[.5,0.],[0.,-.5]], dtype=_core.TENSORDTYPE, device=DEVICE)
SdotS = (torch.einsum('ij,kl->ikjl', _sp, _sm)
       + torch.einsum('ij,kl->ikjl', _sm, _sp)) * 0.5 \
       + torch.einsum('ij,kl->ikjl', _sz, _sz)
SdotS = SdotS.to(DEVICE)  # shape (d,d,d,d)

D_sq   = D_BOND * D_BOND
CTM_MAX_STEPS   = 130
CTM_CONV_THR    = 1e-7
# True = production behaviour (identity + noise init, O(chi²D²) init peak).
# False = contraction-based init, O(D^10) tensors → 13.5 GB for D=7, OOM for D=8.
ENV_IDENTITY_INIT = True

_core.set_rsvd_mode('neumann', neumann_terms=2)
_core.set_ctm_conv_mode('both', e_threshold=2e-8)

# ── Initialize neel_legacy tensors ────────────────────────────────────────────
a_raw = initialize_neel(D_BOND, d_PHYS, noise_scale=1e-2)
a_raw = a_raw.to(DEVICE).requires_grad_(True)

_adam = torch.optim.Adam([a_raw], lr=5e-3, betas=(0.85, 0.96))

if USE_GPU:
    torch.cuda.reset_peak_memory_stats()

mem("INITIAL (after setup)", reset_peak=True)
print()

# ── Monkey-patch for fine-grained per-substep memory tracing ─────────────────
# Patches CTMRG + all 3 update functions + trunc_rhoCCC to print memory at:
#   • entry of each update_environmentCTs call (inside _ckpt forward)
#   • just before trunc_rhoCCC (grown corners alive = 3 × chi²D⁴)
#   • just after trunc_rhoCCC (corners compressed, grown corners freed)
#   • after all T projections (output T's alive)
# Also patches CTMRG_from_init_to_stop to print per-iteration peak.
#
# NOTE: these wrappers run inside _ckpt, so during the FORWARD pass only
# (no autograd graph is built yet).  The backward rerun will trigger a
# second pass of all prints; look for the SECOND occurrence to diagnose
# backward peak.

_original_CTMRG      = core.CTMRG_from_init_to_stop
_original_trunc      = core.trunc_rhoCCC
_original_upd1       = core.update_environmentCTs_1to2
_original_upd2       = core.update_environmentCTs_2to3
_original_upd3       = core.update_environmentCTs_3to1

_ctm_iter_counter = [0]   # mutable list so inner functions can increment

# ── trunc_rhoCCC wrapper ──────────────────────────────────────────────────────
def _instrumented_trunc(matC21, matC32, matC13, chi, D_sq):
    """Print memory before/after the corner SVD truncation."""
    if USE_GPU:
        _grown_MB = (matC21.nelement() * matC21.element_size() / 1024**2)
        _pk = torch.cuda.max_memory_allocated() / 1024**2
        _al = torch.cuda.memory_allocated()    / 1024**2
        print(f"      [trunc_rho IN ] grown-C size={_grown_MB:.1f}MB each  "
              f"alloc={_al:.1f}MB  peak={_pk:.1f}MB")
    result = _original_trunc(matC21, matC32, matC13, chi, D_sq)
    if USE_GPU:
        _pk2 = torch.cuda.max_memory_allocated() / 1024**2
        _al2 = torch.cuda.memory_allocated()     / 1024**2
        print(f"      [trunc_rho OUT] alloc={_al2:.1f}MB  peak={_pk2:.1f}MB")
    return result

# ── update wrapper factory ────────────────────────────────────────────────────
def _make_upd_wrapper(fn, label):
    def _wrapper(*args, **kwargs):
        if USE_GPU:
            torch.cuda.reset_peak_memory_stats()
            _al = torch.cuda.memory_allocated() / 1024**2
            print(f"    [{label} start] alloc={_al:.1f}MB  peak_reset")
        result = fn(*args, **kwargs)
        if USE_GPU:
            _al2 = torch.cuda.memory_allocated() / 1024**2
            _pk2 = torch.cuda.max_memory_allocated() / 1024**2
            print(f"    [{label} end  ] alloc={_al2:.1f}MB  peak={_pk2:.1f}MB")
        return result
    return _wrapper

# ── CTMRG wrapper ─────────────────────────────────────────────────────────────
def _instrumented_CTMRG(A, B, C, D_, E, F, chi, D_squared,
                         a_third_max_iterations, env_conv_threshold,
                         identity_init=False, energy_proxy_fn=None, **kw):
    if USE_GPU:
        torch.cuda.reset_peak_memory_stats()
    mem("  CTMRG enter", reset_peak=True)
    mem_before = cur_alloc()

    # Install fine-grained sub-patches for this run
    core.trunc_rhoCCC                    = _instrumented_trunc
    core.update_environmentCTs_1to2      = _make_upd_wrapper(_original_upd1, "upd1→2")
    core.update_environmentCTs_2to3      = _make_upd_wrapper(_original_upd2, "upd2→3")
    core.update_environmentCTs_3to1      = _make_upd_wrapper(_original_upd3, "upd3→1")
    # trunc_rhoCCC is called inside the update functions which reference
    # core.trunc_rhoCCC at import time; patch the module attribute so calls
    # from update_* (which do  from core import trunc_rhoCCC  at definition
    # time) still hit the wrapper.  Since update_* is already patched to call
    # the original via _original_upd*, we need to re-patch _original_upd* to
    # use instrumented trunc.  Simplest: rely on module-level lookup in core.py.
    # The update functions reference trunc_rhoCCC as a module global, so patching
    # core.trunc_rhoCCC is sufficient.

    result = _original_CTMRG(A, B, C, D_, E, F, chi, D_squared,
                              a_third_max_iterations, env_conv_threshold,
                              identity_init, energy_proxy_fn, **kw)

    # Restore
    core.trunc_rhoCCC               = _original_trunc
    core.update_environmentCTs_1to2 = _original_upd1
    core.update_environmentCTs_2to3 = _original_upd2
    core.update_environmentCTs_3to1 = _original_upd3

    ctm_steps = result[-1]
    mem_delta(f"  CTMRG exit  (ctm_steps={ctm_steps})", mem_before)
    return result

# Install top-level patches
core.CTMRG_from_init_to_stop = _instrumented_CTMRG
CTMRG_from_init_to_stop      = _instrumented_CTMRG


# ── Energy helper (mirrors _three_env_energy_loss_parallel) ──────────────────
def compute_loss_instrumented(a_raw):
    """Full forward + backward with memory instrumentation."""
    # ── derive tensors ────────────────────────────────────────────────────────
    a_sym = symmetrize_virtual_legs(a_raw)
    a, b, c, d, e, f = neel_abcdef_from_a(a_sym)

    aN = normalize_single_layer_tensor_for_double_layer(a)
    bN = normalize_single_layer_tensor_for_double_layer(b)
    cN = normalize_single_layer_tensor_for_double_layer(c)
    dN = normalize_single_layer_tensor_for_double_layer(d)
    eN = normalize_single_layer_tensor_for_double_layer(e)
    fN = normalize_single_layer_tensor_for_double_layer(f)

    mem("  after derive a..f + normalize")

    A, B, C_, D_, E, F = abcdef_to_ABCDEF(aN, bN, cN, dN, eN, fN, D_sq)
    mem("  after abcdef_to_ABCDEF (double-layer)")

    # ── CTMRG (with grad, looser threshold for Adam) ─────────────────────────
    t0 = time.perf_counter()
    all28 = CTMRG_from_init_to_stop(
        A, B, C_, D_, E, F, chi=CHI, D_squared=D_sq,
        a_third_max_iterations=CTM_MAX_STEPS,
        env_conv_threshold=5*CTM_CONV_THR,
        identity_init=ENV_IDENTITY_INIT,
        energy_proxy_fn=None)
    t_ctm = time.perf_counter() - t0

    (C21CD, C32EF, C13AB, T1F,  T2A,  T2B,  T3C,  T3D,  T1E,
     C21EB, C32AD, C13CF, T1D,  T2C,  T2F,  T3E,  T3B,  T1A,
     C21AF, C32CB, C13ED, T1B,  T2E,  T2D,  T3A,  T3F,  T1C,
     ctm_steps) = all28

    mem(f"  after CTMRG unpack (ctm_steps={ctm_steps}, {t_ctm:.1f}s)")

    # ── three energy terms ────────────────────────────────────────────────────
    def _e1_fn():
        return energy_expectation_nearest_neighbor_3ebadcf_bonds(
            aN, bN, cN, dN, eN, fN, *Js_list[0:12], SdotS,
            CHI, D_BOND, d_PHYS,
            C21CD, C32EF, C13AB, T1F, T2A, T2B, T3C, T3D, T1E)

    def _e2_fn():
        return energy_expectation_nearest_neighbor_3afcbed_bonds(
            aN, bN, cN, dN, eN, fN, *Js_list[12:24], SdotS,
            CHI, D_BOND, d_PHYS,
            C21EB, C32AD, C13CF, T1D, T2C, T2F, T3E, T3B, T1A)

    def _e3_fn():
        return energy_expectation_nearest_neighbor_other_3_bonds(
            aN, bN, cN, dN, eN, fN, *Js_list[24:36], SdotS,
            CHI, D_BOND, d_PHYS,
            C21AF, C32CB, C13ED, T1B, T2E, T2D, T3A, T3F, T1C)

    m0 = cur_alloc()
    e1 = _ckpt(_e1_fn, use_reentrant=False)
    mem_delta("  after _ckpt(e1)", m0)

    m1 = cur_alloc()
    e2 = _ckpt(_e2_fn, use_reentrant=False)
    mem_delta("  after _ckpt(e2)", m1)

    m2 = cur_alloc()
    e3 = _ckpt(_e3_fn, use_reentrant=False)
    mem_delta("  after _ckpt(e3)", m2)

    loss = e1 + e2 + e3
    mem("  after sum(e1+e2+e3)")
    return loss, int(ctm_steps)


# ══════════════════════════════════════════════════════════════════════════════
# Main loop — N_STEPS Adam steps
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 75)
print(f"Running {N_STEPS} Adam steps with memory tracing")
print("=" * 75)

for step in range(N_STEPS):
    if USE_GPU:
        torch.cuda.reset_peak_memory_stats()
    print(f"\n{'─'*75}")
    print(f"STEP {step}")
    print(f"{'─'*75}")

    mem("step start", reset_peak=True)

    # ── Forward ───────────────────────────────────────────────────────────────
    _adam.zero_grad()
    m_fwd_start = cur_alloc()
    _loss, ctm_steps = compute_loss_instrumented(a_raw)
    mem_delta("  after full forward", m_fwd_start)

    # ── Backward ─────────────────────────────────────────────────────────────
    m_bwd_start = cur_alloc()
    mem("  before backward")
    _loss.backward()
    mem_delta("  after backward", m_bwd_start)

    # ── Release graph ─────────────────────────────────────────────────────────
    del _loss
    gc.collect()
    if USE_GPU:
        torch.cuda.empty_cache()
    mem("  after del+gc+empty_cache")

    # ── Optimizer step ────────────────────────────────────────────────────────
    if USE_GPU:
        torch.nn.utils.clip_grad_norm_([a_raw], max_norm=1.0)
    _adam.step()
    gc.collect()
    if USE_GPU:
        torch.cuda.empty_cache()
    mem("  after adam.step + gc + empty_cache")

    # ── Riemannian retraction (neel_legacy) ───────────────────────────────────
    with torch.no_grad():
        a_raw.data.copy_(symmetrize_virtual_legs(a_raw.data))
        _an = a_raw.data.norm()
        if _an > 1e-30:
            a_raw.data.div_(_an)

    loss_val = None  # we already did del _loss; just use a proxy
    print(f"  step={step}  ctm_steps={ctm_steps}")

print()
print("=" * 75)
print("DONE")
mem("final")
print("=" * 75)
