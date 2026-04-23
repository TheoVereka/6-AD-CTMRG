
#!/usr/bin/env python3



#NOTE:N_cores, USE_GPU (LINE UNDER TOO!!!!!)
# _default_outdir = os.path.join('/scratch/chye/!!YOURDATE!!!core',   
#!!!! for IZAR is ...atch/izar/chye/...


# # sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
# sys.stdout.flush()*3




MY_OUTPUT_OUTERDIR = '/scratch/chye/0420core'

# ── Sweep control ─────────────────────────────────────────────────────────────

D_BOND_LIST = [3,4, 5, 6, 7, 8, 9, 10, 11]
#   Ordered list of iPEPS virtual bond dimensions to sweep (outer loop).
#   Each D is warm-started from the best tensors found at the previous D
#   (zero-padded to the new size + PAD_NOISE Gaussian noise).


DEFAULT_D_BUDGET_FRACS = {3:0.1,4: 0.1, 5:0.1, 6:0.1, 7:0.1, 8:0.1, 9:0.1, 10:0.1, 11:0.1}
#   Fraction of the total wall-clock budget allocated to each D_bond value.
#   Normalised to sum=1 before use, so only the RATIOS matter.
#   Rationale:
#     D=3 : main physics workhorse, needs the most time — 52 %.
#     D=4 : highest accuracy but very slow per step — 45 %.
#   Note: This is the ONLY intentional asymmetry in the sweep.  Different D
#   values have genuinely different computational costs and scientific weight.
#   Within each D, every chi level gets equal time (see below).

DEFAULT_CHI_MAX = {3:99,4: 80, 5:9999, 6:9999, 7:9999, 8:9999, 9:9999, 10:9999, 11:9999}
#   Largest chi to attempt for each D_bond.
#   Increase if you have more memory; decrease if you hit OOM.

DEFAULT_CHI_SCHEDULES = {
    #2: [ 6,  8],
    3: [ 9, 12, 15],
    4: [12, 16, 20, 24], 
    5: [20, 25, 30, 35],
    6: [24, 30, 36, 42, 48],
    7: [28, 35, 42, 49, 56, 63],
    8: [40, 48, 56, 64, 72, 80],
    9: [45, 54, 63, 72, 81, 90],#, 99],
    10:[50, 60, 70, 80, 90,100],#,110,120],
    11:[55, 66, 77, 88, 99,110],#,121,132,143],
}

# ══════════════════════════════════════════════════════════════════════════════
# Time Budget
# ══════════════════════════════════════════════════════════════════════════════

TOTAL_BUDGET_HOURS = 99999

# ── GPU/CPU intent ────────────────────────────────────────────────────────────
# Duplicated below in the TUNABLE PARAMETERS section with full comments.

USE_GPU = False

# ── Multi-GPU (optional, CUDA only) ──────────────────────────────────────────

N_GPUS = 1
#   Number of GPUs to use for parallel energy computation.
#   1  = single GPU or CPU (default) — sequential, unchanged behaviour.
#   ≥2 = dispatch the 3 independent energy functions to separate GPUs:
#          env1 → cuda:0  (shares with CTMRG)
#          env2 → cuda:1
#          env3 → cuda:min(2, N_GPUS-1)  (cuda:1 when only 2 GPUs available)
#        All three _ckpt calls run concurrently via a 3-thread pool.
#        Gradient correctness preserved: .to(device) is differentiable.
#        Expected speedup: ~2.5× energy phase → ~1.5× overall.
#   Set automatically in main() from --ngpu or torch.cuda.device_count().
#   Override at runtime:  --ngpu N

_N_PHYSICAL_CORES = 25

########################
# ── Physical model ── # ────────────────────────────────────────────────────────────
########################

J1_COUPLING = 1.0
#   Nearest-neighbour (nn) Heisenberg exchange coupling constant.
#   J1 > 0 = antiferromagnetic (AFM).
#   The nn Hamiltonian is  H_nn = J1 Σ_{<i,j>} S_i · S_j
#   summed over all 9 nearest-neighbour pairs in the 6-site honeycomb unit cell.

J2_COUPLING = 0.25
#   Next-nearest-neighbour (nnn) Heisenberg exchange coupling constant.
#   J2 > 0 = frustrated AFM.  Set to 0 to recover the pure J1 model.
#   The nnn Hamiltonian is  H_nnn = J2 Σ_{<<i,j>>} S_i · S_j
#   summed over all 18 next-nearest-neighbour pairs in the 6-site honeycomb
#   unit cell.








"""
main_sweep.py
=============
10-12 h iPEPS benchmark that sweeps over growing D_bond (outer loop) and
growing chi (inner loop) for the AFM Heisenberg model on the 6-site
honeycomb unit cell, using AD-CTMRG.

Usage
-----
  python scripts/main_sweep.py                       # all defaults (11 h)
  python scripts/main_sweep.py --hours 11
  python scripts/main_sweep.py --hours 6 --D-bonds 2,3
  python scripts/main_sweep.py --hours 11 --chi-maxes 16,81,100
  python scripts/main_sweep.py --resume log/sweep_D3_chi81_best.pt --D-bonds 3,4

Outputs (all in log/)
---------------------
  sweep_D{D}_chi{chi}_best.pt    best checkpoint for each (D,chi)
  sweep_D{D}_chi{chi}_latest.pt  last checkpoint for each (D,chi)
  sweep_results.json             JSON table of all energies
  sweep_loss_D{D}.pdf            loss curve per D
  sweep_energy_vs_chi.pdf        E/site vs chi for each D
  sweep_energy_vs_D.pdf          E/site vs D at chi_max (convergence in D)
"""


import argparse
import collections
import datetime
import gc
import json
import os
import sys
import time


# ── CPU threading ─────────────────────────────────────────────────────────────
# MUST be set BEFORE importing NumPy / PyTorch / MKL — they read these at init.
#
# We always default to _N_PHYSICAL_CORES here (safe for CPU).
# After torch is imported and the actual device is determined in main(),
# torch.set_num_threads(1) is called for GPU runs — this overrides both the
# PyTorch and MKL thread pools at runtime, so the env-var default is harmless.
#
# GPU run: 1 intra-op thread — avoids spin-wait contention with CUDA driver.
# CPU run: all physical cores — MKL/BLAS benefits from multi-threading.
#   Hardware: adjust _N_PHYSICAL_CORES above to match your machine.

os.environ.setdefault("OMP_NUM_THREADS", str(_N_PHYSICAL_CORES))
os.environ.setdefault("MKL_NUM_THREADS", str(_N_PHYSICAL_CORES))
# Prevent MKL from silently reducing thread count when it detects nested
# parallelism or high system load.
os.environ.setdefault("MKL_DYNAMIC", "FALSE")
# Pin each OpenMP thread to a distinct physical core (only meaningful for CPU runs).
os.environ.setdefault("KMP_AFFINITY", "granularity=fine,compact,1,0")
# Spin-wait time: 0 so workers yield immediately between parallel regions.
# On GPU runs with 1 thread, this has no effect but is harmless.
os.environ.setdefault("KMP_BLOCKTIME", "0")
#print("OMP_NUM_THREADS =", os.environ["OMP_NUM_THREADS"])
#print("MKL_NUM_THREADS =", os.environ["MKL_NUM_THREADS"])

# ── CUDA memory allocator ─────────────────────────────────────────────────────
# Must be set BEFORE the first CUDA allocation (i.e. before torch.cuda.* calls).
#   expandable_segments:True  — uses expandable (growable) memory segments instead
#     of fixed-size ones.  Dramatically reduces fragmentation that causes OOM at
#     D=9+: without this, the caching allocator often reserves large contiguous
#     blocks that cannot be reused after free, showing as "reserved but
#     unallocated" memory (2+ GiB at D=9,chi=90).  With expandable segments the
#     allocator can grow/shrink segments on demand, so freed memory is reusable.
#   See: https://pytorch.org/docs/stable/notes/cuda.html#environment-variables
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")



# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import opt_einsum as oe
import torch
from torch.utils.checkpoint import checkpoint as _ckpt
import functools as _functools


# Optional debugging aid: if set, PyTorch will print the forward op trace that
# produced a backward error (e.g. complex SVD phase-gauge issues).
if os.environ.get("CTMRG_ANOMALY", "0") == "1":
    torch.autograd.set_detect_anomaly(True)

# Tell PyTorch's own dispatcher about intra-op parallelism.
# The env vars above set the safe default (_N_PHYSICAL_CORES).
# The final decision (1 for GPU, _N_PHYSICAL_CORES for CPU) is made
# in main() after torch.cuda.is_available() is checked.
torch.set_num_threads(_N_PHYSICAL_CORES)
torch.set_num_interop_threads(1)

import core_0p2to0p25 as _core
from core_0p2to0p25 import (
    normalize_tensor,
    normalize_single_layer_tensor_for_double_layer,
    initialize_abcdef,
    abcdef_to_ABCDEF,
    CTMRG_from_init_to_stop,
    build_heisenberg_H,
    energy_expectation_nearest_neighbor_3ebadcf_bonds,
    energy_expectation_nearest_neighbor_3afcbed_bonds,
    energy_expectation_nearest_neighbor_other_3_bonds,
    build_open_closed_env1,
    build_open_closed_env2,
    build_open_closed_env3,
    build_single_open_env1,
    build_single_open_env2,
    build_single_open_env3,
    _build_nn_rho,
    _build_nn_rho_seq,
    _build_nnn_rho,
    _build_nnn_rho_seq,
    set_dtype,
    set_device,
    # ── Néel-symmetrized ansatz ──────────────────────────────────────────
    symmetrize_virtual_legs,
    neel_abcdef_from_a,
    initialize_neel,
    # ── C6-Ypi single-tensor ansatz (renamed from plaq) ──────────────────
    c6ypi_abcdef_from_a,
    initialize_c6ypi,
    # ── C3-Vypi single-tensor ansatz ─────────────────────────────────────
    c3vypi_abcdef_from_a,
    initialize_c3vypi,
    # ── Two-C3 two-tensor ansatz ─────────────────────────────────────────
    twoc3_abcdef_from_ab,
    initialize_twoc3,
    # ── 6-tensor locally-reflected ansatz ───────────────────────────────
    symmetrize_six_local_reflections,
)


class _CollapseRestartD(Exception):
    """Sentinel: raised from optimize_at_chi when collapsed env requires
    restarting the entire current D level from chi_schedule[0]."""
    pass



# Total wall-clock time for the entire sweep.  The sweep is designed to run
# for a fixed time rather than a fixed number of steps, so that results at
# different (D, chi) levels are directly comparable.  The default is 11 h, which
# is enough to get good convergence at D=4, chi=80 on an MX250 GPU.  Adjust
# according to your hardware and patience.  1 h is enough to get good results at D=3, chi=32; 20 min is enough for D=2.



# Default tensor dtype — updated by --double / --real / --complex flags.
# Python functions look up globals at call time, so build_heisenberg_H(),
# pad_tensor(), and optimize_at_chi() all see the updated value after main()
# calls set_dtype() and reassigns TENSORDTYPE.
TENSORDTYPE: torch.dtype = torch.float64

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                  ALL TUNABLE PARAMETERS — EDIT HERE                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ── Precision ─────────────────────────────────────────────────────────────────

# USE_GPU is already declared above the threading section (needed for thread
# count selection before torch is imported).  It is re-documented here for
# clarity; changing the value here has NO EFFECT — edit the declaration above.
#
# USE_GPU = True   ← already set above
#
#   True  → use CUDA GPU when available; automatically falls back to CPU
#            if  torch.cuda.is_available()  returns False.
#   False → always use CPU.
#   Overrideable at runtime: --gpu / --no-gpu CLI flags.

USE_DOUBLE_PRECISION = True
#   False → float32(/complex64): fastest on both CPU (Intel MKL)
#            and CUDA (fp32 tensor cores give full throughput).
#   True  → float64(/complex128): double precision.
#            CPU (Intel MKL): float64 is native, same throughput, 2× RAM.
#            CUDA: 2–4× slower on consumer GPUs (no fp64 tensor cores).
#   Overrideable at runtime: --double CLI flag takes precedence.

USE_REAL_TENSORS = True
#   True  → TENSORDTYPE = RDTYPE  (real iPEPS tensors, S+/S- Hamiltonian).
#   False → TENSORDTYPE = CDTYPE  (complex iPEPS tensors, Sx/Sy/Sz Hamiltonian).
#   Overrideable at runtime: --complex CLI flag sets this to False.

# ── Physical sweep dimensions ────────────────────────────────────────────────
#
#   D_bond  : iPEPS virtual bond dimension.  Controls the expressiveness of
#             the wave-function ansatz.  Computational cost of each CTMRG
#             step scales as O(chi · D^4 · d_phys).  We sweep D in ascending
#             order; each new D warm-starts from the best tensors found at
#             D-1 (zero-padded + small random noise), giving a smooth path
#             in parameter space rather than re-initialising from scratch.
#
#   chi     : Environment (CTMRG) bond dimension.  Controls how faithfully
#             the infinite-lattice environment is represented.
#             Too-small chi → environment too compressed → biased energy;
#             too-large chi → CTMRG is slow and memory-hungry.
#             We sweep chi from the smallest value in the schedule up to
#             chi_max, warm-starting each chi level from the previous.
#             ALL chi levels
#             receive EQUAL time budget and IDENTICAL L-BFGS settings, making
#             results at different chi directly comparable.


SVD_CPU_OFFLOAD_THRESHOLD = 0
#   SVD dispatch: for CUDA runs, matrices with min(m,n) < this value are
#   computed on CPU then moved back to GPU (avoids cuSOLVER launch overhead
#   which dominates for small matrices on low-end GPUs).
#   SET TO 0 FOR CLUSTER GPU — all CTMRG matrices are < 1024, so 1024 forces
#   every SVD to CPU + GPU↔CPU round-trip transfer, defeating the GPU entirely.
#
#   Desktop / laptop GPU (MX250, GTX 1650, etc.):
#     CPU LAPACK beats cuSOLVER at ALL sizes → set to 99999
#   Cluster GPU (A100, V100, H100, RTX 4090, etc.):
#     GPU faster for large matrices (n > ~300-500) → set to 0 or 512
#     0   = always GPU (best for large-chi runs on cluster)
#     512 = CPU for small chi, GPU for large chi (safe default)
#
#   Default 0 → always GPU (correct for cluster; change to 99999 on laptop).

RSVD_MODE = 'augmented'
#   rSVD backward mode.  Controls how the truncated-SVD 5th-term correction
#   (arXiv:2311.11894v3) is computed in the backward pass.
#
#   'full_svd'  — keep full thin SVD (safe reference, no rSVD, ~O(N·min(m,n))).
#   'neumann'   — Neumann series for 5th term: O(2·L·mnk) per call, avoids
#                 the O(N³) eigh.  Exact when discarded SVs ≈ 0.
#   'augmented' — save k+k_extra rSVD triples; zero-padding captures 5th term
#                 implicitly via F,G cross-coupling.  NOT recommended:
#                 benchmarks show ~85-90% gradient error even with k_aug=2k
#                 because only 2k of N modes are captured.  Neumann L=2 is
#                 orders of magnitude more precise at the same cost.
#   'none'      — skip 5th term entirely (wrong for projector-type losses).
#
#   Recommendation for CTMRG:
#     RSVD_MODE = 'neumann', RSVD_NEUMANN_TERMS = 2
#   L=2 costs <1ms extra vs L=1 but gives 15× better gradient at ρ=0.30
#   (i.e. when chi is small and discarded weight is non-negligible).
#   L=1 suffices only when discarded SVs are truly ≈ 0 (large chi).

RSVD_NEUMANN_TERMS = 2
#   Number of Neumann series iterations in 'neumann' mode.
#     positive N  → N-term approximation  (O(2·N·mnk))
#     negative N  → exact eigh (O(N³)), for validation only
#     0           → skip 5th term (same as RSVD_MODE='none')
#   Benchmarked 5th-term relative error (exact U,S,V):
#     ρ=0.01:  L=1 → 7e-6,   L=2 → 6e-10  (both excellent)
#     ρ=0.30:  L=1 → 8e-3,   L=2 → 6e-4   (L=2 is 13× better)
#     ρ=0.50:  L=1 → 3e-2,   L=2 → 7e-3   (L=2 is 5× better)
#   Timing: L=1 → 37.2ms, L=2 → 38.0ms at N=400 — negligible cost.
#   For ρ ≥ 0.80 Neumann doesn't converge; use 'full_svd' instead.

RSVD_POWER_ITERS = None
#   Power iterations for the rSVD range-finder.  None (default) = adaptive:
#   core._adaptive_power_iters(k, N) chooses from the k/N ratio per call.
#   In CTMRG: N = chi·D², k = chi, so k/N = 1/D² (independent of chi).
#     k/N < 2%  (D≥7): niter=1   k/N 2–5%  (D=5–6): niter=2
#     k/N 5–10% (D=4): niter=3   k/N ≥10% (D≤3): niter=4
#   Set to an integer to override (e.g. 4 for conservative testing).


# ── L-BFGS optimiser ─────────────────────────────────────────────────────────
#
#   Optimisation strategy: alternating CTMRG + L-BFGS.
#     1. Converge the CTMRG environment at fixed tensors (a..f).  No grad.
#     2. Run one L-BFGS call (up to LBFGS_MAX_ITER sub-iterations) to
#        minimise the energy w.r.t. a..f, keeping environment fixed.
#     3. Repeat until time budget is exhausted or OPT_CONV_THRESHOLD hit.
#   This is the "cheap-environment" AD-CTMRG gradient scheme.

LBFGS_MAX_ITER = 15
#   Maximum L-BFGS sub-iterations per outer step (= max closure evaluations
#   inside a single optimizer.step() call).  Each sub-iteration does a
#   forward + backward pass through the energy formula.  30 gives a thorough
#   line search and good Dividecurvature estimation without being excessively slow.
#   Applies UNIFORMLY to every (D, chi) level — no difference between small
#   and large chi.

LBFGS_LR = 1.0
#   Step-size seed for the strong-Wolfe line search.  The line search
#   automatically scales the actual step, so lr=1.0 is the standard default
#   and almost always correct.  Only change if you observe line-search
#   failures or divergence.

LBFGS_HISTORY = 150
#   Number of (s, y) curvature vector pairs retained for the L-BFGS inverse-
#   Hessian approximation.  In our alternating-optimisation scheme the LBFGS
#   instance is RECREATED from scratch at every outer step, so curvature pairs
#   accumulate only within a single optimizer.step() call (≤ LBFGS_MAX_ITER
#   sub-iterations).  Any history_size > LBFGS_MAX_ITER allocates buffer
#   memory that is never filled — setting it equal to LBFGS_MAX_ITER is exact
#   and wastes nothing.  Old values like 50–100 were appropriate for classical
#   L-BFGS that runs continuously; they do not apply here.

OPT_TOL_GRAD = 1e-8
#   L-BFGS inner convergence criterion on the infinity-norm of the gradient:
#   the sub-iteration loop exits early if  ||∇loss||_∞ < OPT_TOL_GRAD.
#   This is an inner stopping rule inside a single optimizer.step() call.

OPT_TOL_CHANGE = 4e-8
#   L-BFGS inner convergence criterion on consecutive loss change:
#   sub-iteration exits if  |L_{k+1} – L_k| < OPT_TOL_CHANGE.
#   Set tighter than OPT_TOL_GRAD to catch near-flat regions.

OPT_CONV_THRESHOLD = 1e-7
# Outer-loop early-stop: disabled (= 0).
# The outer delta |loss(k) - loss(k-1)| compares two L-BFGS final values that
# used DIFFERENT CTMRG environments, so even near a true minimum the delta is
# contaminated by env drift O(CTM_CONV_THR).  A non-zero threshold fires
# spuriously.  Stopping is by wall-clock budget; genuine convergence is caught
# by the cycle-detection check below.
#   Outer-loop early-stop criterion: if |loss(step k) – loss(step k–1)|
#   < OPT_CONV_THRESHOLD, the outer while-loop exits and we move to the
#   next (D, chi) level.  Set to 0 to disable early stopping and always
#   run until the time budget is exhausted.

CHI_CONVERGENCE_THRESHOLD = 1e-5
#   Chi-level early-exit criterion (lookahead).
#   After optimisation at (D, chi) is complete and a clean energy evaluation
#   is done, also evaluate the energy at (D, chi_next) using the SAME (already
#   optimised) tensors before running any optimisation at chi_next.  If
#   |E(chi) − E(chi_next)| < CHI_CONVERGENCE_THRESHOLD, the
#   chi series for the current D is considered converged and we immediately
#   jump to the next D, skipping the remaining chi levels.  On entering
#   D_next the chi schedule is filtered to start from the first chi that is
#   strictly larger than the finishing chi of D (not chi_next), so low-chi
#   environments already covered by the previous D are not repeated.
#   Recorded in hyperparams.yaml as chi_convergence_threshold.

# ── Optimizer choice ──────────────────────────────────────────────────────────

OPTIMIZER = 'lbfgs'
#   'lbfgs' : L-BFGS with strong-Wolfe line search (default).
#             Converges fast on smooth landscapes; may oscillate on noisy ones.
#   'adam'  : Adam (adaptive moment estimation).
#             More robust on noisy/non-convex landscapes.  Each outer step runs
#             ADAM_STEPS_PER_CTM gradient updates with a fixed environment.
#   Override at runtime:  --optimizer {lbfgs,adam}

# ── Adam hyperparameters (used only when OPTIMIZER='adam') ────────────────────

ADAM_LR = 3e-3
#   Adam learning rate.  Typical range: 1e-4 – 1e-3.
ADAM_BETAS = (0.9, 0.999)
#   Exponential decay rates for 1st and 2nd moment estimates.
ADAM_EPS = 1e-8
#   Denominator epsilon for numerical stability.
ADAM_WEIGHT_DECAY = 0.0
#   L2 regularisation strength.  0.0 = no regularisation (recommended).
ADAM_STEPS_PER_CTM = 5
#   Number of Adam gradient steps between consecutive CTMRG environment
#   refreshes.  Higher = cheaper; lower = more accurate environment.

# ── CTMRG algorithm ──────────────────────────────────────────────────────────
#
#   CTMRG (Corner Transfer Matrix Renormalisation Group) builds 9 corner
#   matrices and 18 transfer tensors that represent the environment of the
#   6-site honeycomb unit cell embedded in the infinite lattice.  Each
#   CTMRG "step" grows the environment by one unit-cell layer and then
#   compresses via SVD truncation to keep the environment bond dim = chi.







ENV_IDENTITY_INIT = False






CTM_MAX_STEPS = 100
#   Hard cap on CTMRG iterations per environment convergence call.
#   With the singular-value convergence criterion and CTM_CONV_THR=1e-3,
#   convergence occurs in 4–40 steps for typical tensors (single-tensor
#   ansatz ~4 steps, 6-tensor ~40 steps).  90 is a safe upper bound.

CTM_CONV_THR = 1e-6

CTM_CONV_THR_FLOAT32_MIN = 1e-5
#   CTMRG convergence threshold: stop iterating when the max change in
#   normalised corner singular values between consecutive steps is below
#   this value.  The convergence criterion compares the spectra of all 9
#   corner matrices (gauge-invariant), so raw-element Frobenius oscillations
#   from SVD sign ambiguity do NOT affect it.  In float32 the spectral noise
#   floor is ~5e-5–2e-4, so any threshold below ~5e-4 effectively never
#   triggers; 1e-3 converges in 7–20 steps across all tested D/chi configs.
#   NOTE: main() automatically raises CTM_CONV_THR to CTM_CONV_THR_FLOAT32_MIN when running in
#   float32 (USE_DOUBLE_PRECISION=False / --single) — do not hard-code 3e-7
#   for single-precision runs or CTMRG will never converge (ctm=40 always).

CTM_CONV_MODE = 'both'
#   CTMRG convergence criterion.  Controls which metric(s) must be satisfied
#   for CTMRG to declare convergence.  Three options:
#
#   'SVdifference'  (default, recommended for optimization)
#     Converge when max|ΔSV| < CTM_CONV_THR across all 9 corner spectra.
#     Fastest: no tensor builds needed, O(chi) per check.
#
#   'Edifference'   (recommended for clean evaluation when physics accuracy
#                   matters more than speed)
#     Converge when |ΔE_proxy| < CTM_E_CONV_THRESHOLD.
#     E_proxy = tr(ρ_EB · SdotS) = 1 NN bond from env1 (no J factor).
#     Cost per check: 6 open tensor builds + 1 rho (~1/27 of full energy).
#     SV check still runs every iteration for zero-collapse detection.
#     Only passes energy_proxy_fn during evaluate_energy_clean /
#     evaluate_observables — NOT during the optimization CTMRG call.
#
#   'both'          (strictest; for near-phase-boundary diagnostics)
#     Converge only when BOTH max|ΔSV| < CTM_CONV_THR AND
#     |ΔE_proxy| < CTM_E_CONV_THRESHOLD are simultaneously satisfied.
#     Once SV has converged (sticky flag), waits for E to also converge.
#
#   Recorded in hyperparams.yaml as ctm_conv_mode.

CTM_E_CONV_THRESHOLD = 3e-8
#   Energy-proxy convergence threshold for 'Edifference' and 'both' modes.
#   Applied to |E_proxy(iter N) − E_proxy(iter N-1)| where E_proxy is the
#   EB bond energy from env1 (unit SdotS, no J).
#   Typical converged drift at D=8, chi=80: ~1e-10 to 1e-11.
#   1e-8 is a safe target that fires well before oscillations dominate.
#   Recorded in hyperparams.yaml as ctm_e_conv_threshold.

CTM_E_PROXY_INTERVAL = 1
#   Number of CTMRG iterations between consecutive energy proxy evaluations.
#   interval=1: check every step (tightest tracking, highest overhead).
#   interval=5: check every 5th step (default; good balance).
#   interval=10: check every 10th step (lowest overhead, coarser tracking).
#   Cost: 6 open tensor builds per check ≈ 22% of one full energy evaluation.
#   Over 20 CTMRG steps with interval=5: 4 checks ≈ 88% of 1 energy overhead.
#   Only active when energy_proxy_fn is passed to CTMRG_from_init_to_stop,
#   which the driver does ONLY for evaluate_energy_clean / evaluate_observables.
#   Recorded in hyperparams.yaml as ctm_e_proxy_interval.



SAVE_EVERY = 10
#   Frequency (in outer optimisation steps) at which the "latest" checkpoint
#   is written.  The "best" checkpoint is written immediately whenever a new
#   minimum energy is found, independently of SAVE_EVERY.  Lower = more I/O
#   but safer against crashes; higher = less I/O.

D_PHYS = 2
#   Physical Hilbert-space dimension per lattice site.
#   d=2 for spin-1/2 (default), d=3 for spin-1, d=4 for two-site, etc.

N_SITES = 6
#   Number of sites in the honeycomb unit cell.
#   The unit cell has 9 nn bonds + 18 nnn bonds = 27 total bonds.
#   Energy per site = E_total / N_SITES.


# ── Tensor initialisation & padding ──────────────────────────────────────────

INIT_NOISE = 5e-3
# !!! NOTE: Only used as Mean-Field-Init's random noise!!!
# should be at least 2e-4 otherwise the initial state is too 
# close to the exact Néel product state and the optimizer gets 
# stuck in a local minimum.

PAD_NOISE = 1e-2
#   Gaussian noise amplitude added to the ZERO-PADDED new indices when
#   enlarging tensors from D → D+1.  Non-zero noise breaks the symmetry of
#   subspace of the smaller-D manifold.  Keep comparable to INIT_NOISE.

MEAN_FIELD_INIT = True
#   True  → override EVERY chi level's initialisation (for all D) with a
#            mean-field Néel product state: A-sublattice sites (a,c,e or
#            the single raw tensor for 1-tensor ansätze) seeded at spin-up
#            (physical index 0) and B-sublattice sites (b,d,f) seeded at
#            spin-down (physical index 1), each with INIT_NOISE Gaussian
#            noise added on top to break degeneracy.
#            Takes precedence over RAND_INIT_NEW_CHI and any D→D warm-start.
#   False → default: use random init or warm-start (controlled by the flags
#            above).  Overrideable at runtime: --mean-field-init CLI flag.

RAND_INIT_NEW_D = False
#   True  → skip the D-1 → D padded warm-start entirely; each new D starts
#            from a fully random initialisation (same as the very first D).
#   False → default: pad best tensors from D-1 to size D and add PAD_NOISE.
#   Overrideable at runtime: --rand-init-new-d CLI flag.

RAND_INIT_NEW_CHI = False
#   True  → skip warm-starting from the previous chi's result; each chi
#            level within the same D starts from a fully random initialisation.
#   False → default: continue from best tensors found at the previous chi.
#   Overrideable at runtime: --rand-init-new-chi CLI flag.

GEO_SCHEDULE_STEPS = 5
#   Number of chi values generated by the geometric FALLBACK schedule.
#   Only used when --chi-maxes is given on the command line and chi_max
#   differs from DEFAULT_CHI_MAX[D], meaning DEFAULT_CHI_SCHEDULES cannot
#   be used.  Generates GEO_SCHEDULE_STEPS log-uniformly spaced values
#   from 1 to chi_max (inclusive).

# ── Reproducibility ──────────────────────────────────────────────────────────

RANDOM_SEED_FIX = False
#   True  → fix all RNGs (Python random, NumPy, PyTorch CPU + CUDA) to
#            RANDOM_SEED before any tensor is allocated.  Guarantees
#            bit-identical initialisation, padding noise, and rSVD random
#            vectors across two runs with the same hyperparameters.
#   False → non-reproducible (random initialisation differs each run).
#   Overrideable at runtime: --no-seed CLI flag sets this to False.

RANDOM_SEED = 42
#   Integer seed used when RANDOM_SEED_FIX = True.
#   Override at runtime: --seed <int>


# ══════════════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════════════

def timestamp() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")







"""
def build_heisenberg_H(J: float = 1.0, d: int = 2) -> torch.Tensor:
    sx = torch.tensor([[0, 1], [1, 0]], dtype=CDTYPE) / 2
    sy = torch.tensor([[0, -1j], [1j, 0]], dtype=CDTYPE) / 2
    sz = torch.tensor([[1, 0], [0, -1]], dtype=CDTYPE) / 2
    SdotS = (oe.contract("ij,kl->ikjl", sx, sx)
           + oe.contract("ij,kl->ikjl", sy, sy)
           + oe.contract("ij,kl->ikjl", sz, sz))
    return J * SdotS
"""


def _new_tensors_from_data(tensors: tuple) -> tuple:
    """Create brand-new tensors from raw numerical data.

    Each returned tensor is a *fresh* ``torch.empty`` + ``copy_``, sharing
    zero autograd lineage, zero storage, zero Python identity with the
    originals.  After calling this, ``del tensors`` truly frees everything
    from the old optimisation run (optimizer state, CTMRG envs, closures).
    """
    out = []
    for t in tensors:
        new = torch.empty(t.shape, dtype=TENSORDTYPE, device=t.device)
        new.copy_(t.detach())
        out.append(new)
    return tuple(out)


# ── Ansatz helpers ────────────────────────────────────────────────────────────

def _derive_abcdef(params_list: list, cfg: dict) -> tuple:
    """Derive the 6 single-layer tensors from *params_list* using *cfg*.

    For the unrestricted ansatz (cfg['n_params']==6, symmetrize_fn=None) the
    params are already the 6 tensors, returned directly.

    For the 'sym6' ansatz (cfg['n_params']==6, symmetrize_fn!=None) the
    symmetrize_fn is called with all 6 tensors as positional arguments and
    returns a tuple of 6 symmetrized tensors.

    For single-tensor ansätze (cfg['n_params']==1) the single raw tensor is
    optionally symmetrized (if cfg['symmetrize_fn'] is not None) and then the
    ansatz-specific derive function produces all 6 tensors.

    For two-tensor ansätze (cfg['n_params']==2) both raw tensors are passed
    to derive_fn(*params_list) which returns the 6 tensors directly.
    """
    if cfg['n_params'] == 6:
        if cfg['symmetrize_fn'] is not None:
            # Batch symmetrization: fn(a, b, c, d, e, f) → (a_sym, ..., f_sym)
            return cfg['symmetrize_fn'](*params_list)
        return tuple(params_list)
    if cfg['n_params'] == 2:
        return cfg['derive_fn'](*params_list)
    raw = params_list[0]
    if cfg['symmetrize_fn'] is not None:
        raw = cfg['symmetrize_fn'](raw)
    return cfg['derive_fn'](raw)


# ── Ansatz registry ───────────────────────────────────────────────────────────
# To add a new ansatz:
#   1. Add 3 functions to core_combined.py:
#        symmetrize_<name>_legs(a) → a_sym
#        <name>_abcdef_from_a(a_sym) → (a, b, c, d, e, f)
#        initialize_<name>(D_bond, d_PHYS, noise_scale) → a_raw
#   2. Import them at the top of this file.
#   3. Add one entry below (n_params=1 for single-tensor, 6 for independent).
#
# All other code in this file is driven by the registry entry: no other edits
# needed unless the new ansatz requires a fundamentally different structure.

ANSATZ_REGISTRY: dict = {
    # ── fully unrestricted 6-tensor ansatz ───────────────────────────────────
    'unrestricted': {
        'n_params':    6,
        'symmetrize_fn': None,
        'derive_fn':   None,
        'init_fn':     None,          # uses initialize_abcdef('random', ...)
        'ckpt_keys':   ['a', 'b', 'c', 'd', 'e', 'f'],
        'yaml_name':   '6tensors',
        'description': 'Fully unrestricted 6-tensor ansatz',
    },
    # ── Néel-symmetrized single-tensor ansatz ─────────────────────────────────
    'neel': {
        'n_params':    1,
        'symmetrize_fn': symmetrize_virtual_legs,
        'derive_fn':   neel_abcdef_from_a,
        'init_fn':     initialize_neel,
        'ckpt_keys':   ['a_raw'],
        'yaml_name':   'neel_symmetrized',
        'description': 'Néel-symmetrized single-tensor ansatz (S3 virtual legs, π-rotation U=iσ_y)',
    },
    # ── C6-Ypi single-tensor ansatz (C6 rotation + pi-Y on B-sublattice) ──
    'c6ypi': {
        'n_params':    1,
        'symmetrize_fn': None,
        'derive_fn':   c6ypi_abcdef_from_a,
        'init_fn':     initialize_c6ypi,
        'ckpt_keys':   ['a_raw'],
        'yaml_name':   '1tensor_C6Ypi',
        'description': 'C6-Ypi single-tensor ansatz (C6 virtual-leg rotation + pi-Y on B-sublattice)',
    },
    # ── 6-tensor locally-reflected ansatz ────────────────────────────────────
    # Like 'unrestricted' but each site tensor is projected onto the subspace
    # invariant under its local mirror symmetry (swaps the two ring/intra-
    # plaquette bonds while leaving the inter-plaquette bond fixed):
    #   a, d: leg1↔leg2  via permute(0,2,1,3)
    #   b, e: leg0↔leg1  via permute(1,0,2,3)
    #   c, f: leg0↔leg2  via permute(2,1,0,3)
    # The 6 tensors remain fully independent (no C3 relation between sites),
    # so 'sym6' can represent symmetry-broken states that 'plaq' cannot.
    'sym6': {
        'n_params':    6,
        'symmetrize_fn': symmetrize_six_local_reflections,  # batch: (a..f)→(a_sym..f_sym)
        'derive_fn':   None,      # identity; symmetrize_fn already produces 6 tensors
        'init_fn':     None,      # uses initialize_abcdef('random',...) for n_params==6
        'ckpt_keys':   ['a', 'b', 'c', 'd', 'e', 'f'],
        'yaml_name':   'sym6',
        'description': 'Six independent tensors, each with local mirror symmetry '
                        '(a/d: leg1↔leg2, b/e: leg0↔leg1, c/f: leg0↔leg2)',
    },
    # ── C3-Vypi single-tensor ansatz ─────────────────────────────────────────
    # Like c6ypi but with a different virtual-leg permutation rule.
    # The user defines the permutation in c3vypi_abcdef_from_a.
    'c3vypi': {
        'n_params':    1,
        'symmetrize_fn': None,
        'derive_fn':   c3vypi_abcdef_from_a,
        'init_fn':     initialize_c3vypi,
        'ckpt_keys':   ['a_raw'],
        'yaml_name':   '1tensor_C3Vypi',
        'description': 'C3-Vypi single-tensor ansatz (C3 virtual-leg rotation + pi-Y on B-sublattice)',
    },
    # ── Two-C3 two-tensor ansatz ────────────────────────────────────────────
    # Two independent tensors (a, b); A-sublattice = C3 rotations of a,
    # B-sublattice = C3 rotations of b.  No symmetrization, no physical-leg
    # rotation.
    'twoc3': {
        'n_params':    2,
        'symmetrize_fn': None,
        'derive_fn':   twoc3_abcdef_from_ab,
        'init_fn':     initialize_twoc3,
        'ckpt_keys':   ['a_raw', 'b_raw'],
        'yaml_name':   '2tensor_twoC3',
        'description': 'Two-tensor ansatz: A-sublattice (a,c,e) = C3 rotations of a; '
                        'B-sublattice (b,d,f) = C3 rotations of b',
    },
}


def pad_tensor(t: torch.Tensor, old_D: int, new_D: int,
               d_PHYS: int, noise: float,
               symmetrize_fn=None) -> torch.Tensor:
    # Whenever pad_tensor is called (D→D+1 warm-start OR same-D recovery),
    # the next optimization step should use full (deterministic) SVD to avoid
    # rSVD gradient blowup on the noisy initial point.  The flag is cleared
    # automatically after one L-BFGS step completes.
    _core._USE_FULL_SVD = True

    out = noise * torch.randn(new_D, new_D, new_D, d_PHYS, dtype=TENSORDTYPE,
                              device=_core.DEVICE)
    out[:old_D, :old_D, :old_D, :] += normalize_tensor(t.detach())*torch.sqrt(torch.tensor(old_D**3 * d_PHYS, dtype=TENSORDTYPE))
    if symmetrize_fn is not None:
        out = symmetrize_fn(out)
    return out


def _make_mean_field_params(
        ansatz_cfg: dict, D_bond: int, d_PHYS: int,
        noise: float = 1e-2,
) -> tuple:
    """Return initial tensors seeded at the mean-field Néel product state.

    A-sublattice role: t[0,0,0,0] ≈ 1  (physical spin-up, index 0).
    B-sublattice role: t[0,0,0,s] ≈ 1  (physical spin-down, index 1 mod d_PHYS).
    Small Gaussian noise (amplitude ``noise``) is added to every element to
    break the exact degeneracy and seed the optimiser.  The virtual bonds
    beyond index 0 carry only noise, so the state is near a D=1 product state
    embedded in the larger D-dimensional space.

    n_params == 1  →  single seed tensor (A-sublattice); derive_fn handles
                      the B-sublattice relabelling automatically.
    n_params == 2  →  (a_raw [up], b_raw [down]) for two-tensor ansätze.
    n_params == 6  →  (a[up], b[down], c[up], d[down], e[up], f[down]).
    """
    def _up():
        t = noise * torch.randn(D_bond, D_bond, D_bond, d_PHYS,
                                dtype=TENSORDTYPE, device=_core.DEVICE)
        t[0, 0, 0, 0] += 1.0
        return t

    def _down():
        t = noise * torch.randn(D_bond, D_bond, D_bond, d_PHYS,
                                dtype=TENSORDTYPE, device=_core.DEVICE)
        t[0, 0, 0, -1] += -1.0   
        return t

    n = ansatz_cfg['n_params']
    if n == 1:
        # Single raw tensor; derive_fn maps it to both sublattices.
        return (_up(),)
    elif n == 2:
        # twoc3: first tensor → A-sublattice (up), second → B-sublattice (down).
        return (_up(), _down())
    else:
        # n == 6: unrestricted / sym6 — explicit ABABAB assignment.
        return (_up(), _down(), _up(), _down(), _up(), _down())


def evaluate_observables(params: list,
                         Js, SdotS, chi: int, D_bond: int, d_PHYS: int,
                         ansatz_cfg: dict,
                         out_dir: str | None = None):
    """Compute energy, 27 bond correlations <Si·Sj>, and 54 site magnetizations.

    This is a diagnostic function called ONCE per (D, chi) level after
    optimization.  It re-converges CTMRG from scratch (same as
    evaluate_energy_clean), then extracts observables from the two-site
    reduced density matrices that are already computed for the energy.

    Returns:
        energy (float): Total energy (same value as evaluate_energy_clean).
        correlations (list[float]): 36 values of <Si·Sj> (unit SdotS, no J)
            ordered as env1 (12: eb,ad,cf,fa,de,bc + ae,ec,ca,db,bf,fd)
                      env2 (12: cb,af,ed,dc,ba,fe + ca,ae,ec,bf,fd,db)
                      env3 (12: ef,ab,cd,be,fc,da + ec,ca,ae,fd,db,bf).
        magnetizations (list[float]): 54 values of <Sα> for each of the
            6 sites × 3 envs × 3 spin dirs (Sx, Sy, Sz).
            <Sy> is computed via iSy_op = (S+-S-)/2 = i·Sy_phys:
              Real iPEPS:    <Sy> = Re(Tr(ρ·iSy_op)) = 0.0 exactly
                             (real-symmetric ρ × real-antisymmetric iSy → 0).
                             Saved as informational sanity check.
              Complex iPEPS: <Sy> = Im(Tr(ρ·iSy_op)) — full physical value.
            Ordered: env1 site A (Sx, Sy, Sz), env1 site B, ..., env1 site F,
                     env2 site A, ..., env3 site F.
        trunc_error (float | None): Mean CTMRG SVD truncation error from
            the last CTMRG iteration, averaged over 3 environments × 3 SVDs.
            Defined as ||S_discarded|| / ||S_all|| per SVD.  None if CTMRG
            did not record errors (should not happen in normal use).
    """
    a, b, c, d, e, f = _derive_abcdef(params, ansatz_cfg)
    D_sq = D_bond ** 2

    # ── Build single-site spin operators ──────────────────────────────────
    spin = (d_PHYS - 1) / 2.0
    Splus = torch.zeros(d_PHYS, d_PHYS, dtype=SdotS.dtype, device=SdotS.device)
    Sminus = torch.zeros(d_PHYS, d_PHYS, dtype=SdotS.dtype, device=SdotS.device)
    Sz_op = torch.zeros(d_PHYS, d_PHYS, dtype=SdotS.dtype, device=SdotS.device)
    for i in range(d_PHYS):
        m_val = spin - i
        Sz_op[i, i] = m_val
        if i < d_PHYS - 1:
            cs = (spin * (spin + 1) - m_val * (m_val - 1)) ** 0.5
            Splus[i, i + 1] = cs
            Sminus[i + 1, i] = cs
    Sx_op = (Splus + Sminus) / 2.0    # real matrix
    # iSy_op = (S+ - S-)/2 = i·Sy_phys — real antisymmetric matrix.
    # Used to extract <Sy> in both real and complex cases (see _mag below).
    iSy_op = (Splus - Sminus) / 2.0   # real antisymmetric (= i·Sy_phys)

    def _corr(rho_4d):
        """<Si·Sj> = Tr(rho * SdotS) — unit correlation, no J factor."""
        return torch.real(oe.contract("ikjl,ijkl->", rho_4d, SdotS,
                                      backend="torch")).item()

    def _mag(rho_4d, site_idx):
        """Single-site magnetization from 2-site rho via partial trace.

        site_idx: 0 = first site in the pair (partial-trace second),
                  1 = second site in the pair (partial-trace first).
        Returns (mx, my, mz).
        """
        if site_idx == 0:
            rho_1 = oe.contract("ikjk->ij", rho_4d, backend="torch")
        else:
            rho_1 = oe.contract("ikil->kl", rho_4d, backend="torch")
        # Normalize
        tr = torch.real(torch.trace(rho_1)).clamp(min=1e-30)
        rho_1 = rho_1 / tr
        mx = torch.real(oe.contract("ij,ji->", rho_1, Sx_op, backend="torch")).item()
        # <Sy> via iSy_op = (S+-S-)/2 = i·Sy_phys (real antisymmetric).
        # Real rho:    Tr(rho · iSy_op) is real,  = 0.0 exactly (sym × antisym).
        # Complex rho: Tr(rho · iSy_op) = i·<Sy> (purely imaginary)
        #              → take Im(...) to recover the physical <Sy>.
        if rho_1.is_complex():
            my = torch.imag(oe.contract("ij,ji->", rho_1,
                                        iSy_op.to(rho_1.dtype),
                                        backend="torch")).item()
        else:
            # real case: result is exactly 0.0; computed explicitly for information
            my = torch.real(oe.contract("ij,ji->", rho_1, iSy_op,
                                        backend="torch")).item()
        mz = torch.real(oe.contract("ij,ji->", rho_1, Sz_op, backend="torch")).item()
        return (mx, my, mz)

    correlations = []
    magnetizations = []

    with torch.no_grad():
        aN = normalize_single_layer_tensor_for_double_layer(a)
        bN = normalize_single_layer_tensor_for_double_layer(b)
        cN = normalize_single_layer_tensor_for_double_layer(c)
        dN = normalize_single_layer_tensor_for_double_layer(d)
        eN = normalize_single_layer_tensor_for_double_layer(e)
        fN = normalize_single_layer_tensor_for_double_layer(f)

        A, B, C, Dt, E, F = abcdef_to_ABCDEF(aN, bN, cN, dN, eN, fN, D_sq)
        # Clean-evaluation CTMRG: always pass energy_proxy_fn=None.
        # core.CTMRG_from_init_to_stop auto-falls back to SVdifference mode
        # when proxy=None (see _effective_mode logic in core.py), so CTMRG
        # converges fast via SV criterion.  The full 27-bond energy and all
        # observables are computed ONCE after CTMRG converges — there is no
        # benefit to calling 36 open-tensor builds per CTMRG iteration here.
        # (Using the Edifference proxy during clean-eval CTMRG would make each
        # evaluate_observables call 10-100× slower than a single L-BFGS step.)
        _core.set_record_trunc_error(True)
        all27 = CTMRG_from_init_to_stop(
                A, B, C, Dt, E, F, chi, D_sq,
                CTM_MAX_STEPS, CTM_CONV_THR, ENV_IDENTITY_INIT,
                energy_proxy_fn=None)
        trunc_error = _core.get_last_trunc_error()
        _core.set_record_trunc_error(False)

        # ── Last-resort zero guard (CTMRG already retried with escalating noise) ──
        _corner_indices = (0, 1, 2, 9, 10, 11, 18, 19, 20)
        if all(torch.linalg.norm(all27[i]).item() < 1e-30 for i in _corner_indices):
            print(f"  │  [WARN] CTMRG env still zero at chi={chi} after all internal retries; observables = NaN")
            return float('nan'), [float('nan')]*36, [float('nan')]*54, float('nan')

        # ── Free double-layer tensors (only needed for CTMRG) ────────────
        del A, B, C, Dt, E, F
        if _core.DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

        # ── Memory check: pre-build vs lazy path ────────────────────────
        # Pre-build: build_open_closed_env builds all 6 opens per env ONCE,
        #   reuses them for all 12 rho computations → 18 open builds total.
        #   This is 5× faster than the lazy path for the post-CTMRG phase.
        # Lazy: rebuilds each open for every rho that needs it → 90 builds.
        #
        # Under torch.no_grad() there is no autograd graph, so pre-build is
        # always preferred UNLESS 6 opens don't fit in GPU memory (D≥11 on
        # a 32 GB GPU).  The lazy path was designed for the optimization
        # forward+backward where checkpointing + autograd dominate memory.
        #
        # Peak estimate: 6 opens + 2 NNN intermediates (W,V) + 6 closeds.
        _use_lazy = False
        if _core.DEVICE.type == 'cuda':
            _D2 = chi * D_bond * D_bond
            _elem_sz = aN.element_size()  # 8 for float64, 4 for float32
            _open_bytes = _D2 * _D2 * d_PHYS * d_PHYS * _elem_sz
            _peak_est = 8 * _open_bytes + 6 * _D2 * _D2 * _elem_sz
            _gpu_total = torch.cuda.get_device_properties(
                _core.DEVICE).total_memory
            _use_lazy = _peak_est > _gpu_total * 0.80

        if _use_lazy:
            # ── Lazy path (memory-safe fallback for very large D) ─────────
            # Rebuilds each open tensor ~5× but keeps at most 2 alive.
            # Used only when 6 opens don't fit in GPU memory.
            _a1 = (aN, bN, cN, dN, eN, fN, chi, D_bond, d_PHYS) + tuple(all27[:9])
            _a2 = (aN, bN, cN, dN, eN, fN, chi, D_bond, d_PHYS) + tuple(all27[9:18])
            _a3 = (aN, bN, cN, dN, eN, fN, chi, D_bond, d_PHYS) + tuple(all27[18:27])

            # Lazy rho helpers: build pair of opens, compute rho, free opens
            def _lnn(open_fn, s1, s2, p1, p2):
                return _build_nn_rho_seq(
                    lambda: open_fn(s1), lambda: open_fn(s2), p1, p2, d_PHYS)
            def _lnnn(open_fn, s1, cl2, s3, cl4, lp):
                return _build_nnn_rho_seq(
                    lambda: open_fn(s1), cl2,
                    lambda: open_fn(s3), cl4, lp, d_PHYS)

            # ── Env 1 ── (build closeds, pairs, rhos, then free)
            c1 = {}
            for _s in 'EDAFCB':
                _o = build_single_open_env1(_s, *_a1)
                c1[_s] = oe.contract("ABii->AB", _o, backend="torch"); del _o
            AD1 = torch.mm(c1['A'], c1['D']); CF1 = torch.mm(c1['C'], c1['F']); EB1 = torch.mm(c1['E'], c1['B'])
            FA1 = torch.mm(c1['F'], c1['A']); DE1 = torch.mm(c1['D'], c1['E']); BC1 = torch.mm(c1['B'], c1['C'])
            _op1 = lambda s: build_single_open_env1(s, *_a1)
            rho_AD = _lnn(_op1, 'A', 'D', EB1, CF1)
            rho_CF = _lnn(_op1, 'C', 'F', AD1, EB1)
            rho_EB = _lnn(_op1, 'E', 'B', CF1, AD1)
            rho_FA = _lnn(_op1, 'F', 'A', DE1, BC1)
            rho_DE = _lnn(_op1, 'D', 'E', BC1, FA1)
            rho_BC = _lnn(_op1, 'B', 'C', FA1, DE1)
            rho_AE1 = _lnnn(_op1, 'A', c1['D'], 'E', c1['B'], CF1)
            rho_EC1 = _lnnn(_op1, 'E', c1['B'], 'C', c1['F'], AD1)
            rho_CA1 = _lnnn(_op1, 'C', c1['F'], 'A', c1['D'], EB1)
            rho_DB1 = _lnnn(_op1, 'D', c1['E'], 'B', c1['C'], FA1)
            rho_BF1 = _lnnn(_op1, 'B', c1['C'], 'F', c1['A'], DE1)
            rho_FD1 = _lnnn(_op1, 'F', c1['A'], 'D', c1['E'], BC1)
            correlations += [_corr(r) for r in [rho_EB, rho_AD, rho_CF,
                                                rho_FA, rho_DE, rho_BC,
                                                rho_AE1, rho_EC1, rho_CA1,
                                                rho_DB1, rho_BF1, rho_FD1]]
            mag1 = {'A': _mag(rho_AD, 0), 'D': _mag(rho_AD, 1),
                    'C': _mag(rho_CF, 0), 'F': _mag(rho_CF, 1),
                    'E': _mag(rho_EB, 0), 'B': _mag(rho_EB, 1)}
            for s in ['A', 'B', 'C', 'D', 'E', 'F']:
                magnetizations.extend(mag1[s])
            del rho_AD, rho_CF, rho_EB, rho_FA, rho_DE, rho_BC
            del rho_AE1, rho_EC1, rho_CA1, rho_DB1, rho_BF1, rho_FD1
            del c1, AD1, CF1, EB1, FA1, DE1, BC1
            torch.cuda.empty_cache()

            # ── Env 2 ── (build closeds, pairs, rhos, then free)
            c2 = {}
            for _s in 'ABCDEF':
                _o = build_single_open_env2(_s, *_a2)
                c2[_s] = oe.contract("ABii->AB", _o, backend="torch"); del _o
            CB2 = torch.mm(c2['C'], c2['B']); ED2 = torch.mm(c2['E'], c2['D']); AF2 = torch.mm(c2['A'], c2['F'])
            DC2 = torch.mm(c2['D'], c2['C']); BA2 = torch.mm(c2['B'], c2['A']); FE2 = torch.mm(c2['F'], c2['E'])
            _op2 = lambda s: build_single_open_env2(s, *_a2)
            rho_CB = _lnn(_op2, 'C', 'B', AF2, ED2)
            rho_AF = _lnn(_op2, 'A', 'F', ED2, CB2)
            rho_ED = _lnn(_op2, 'E', 'D', CB2, AF2)
            rho_DC = _lnn(_op2, 'D', 'C', BA2, FE2)
            rho_BA = _lnn(_op2, 'B', 'A', FE2, DC2)
            rho_FE = _lnn(_op2, 'F', 'E', DC2, BA2)
            rho_CA2 = _lnnn(_op2, 'C', c2['B'], 'A', c2['F'], ED2)
            rho_AE2 = _lnnn(_op2, 'A', c2['F'], 'E', c2['D'], CB2)
            rho_EC2 = _lnnn(_op2, 'E', c2['D'], 'C', c2['B'], AF2)
            rho_BF2 = _lnnn(_op2, 'B', c2['A'], 'F', c2['E'], DC2)
            rho_FD2 = _lnnn(_op2, 'F', c2['E'], 'D', c2['C'], BA2)
            rho_DB2 = _lnnn(_op2, 'D', c2['C'], 'B', c2['A'], FE2)
            correlations += [_corr(r) for r in [rho_CB, rho_AF, rho_ED,
                                                rho_DC, rho_BA, rho_FE,
                                                rho_CA2, rho_AE2, rho_EC2,
                                                rho_BF2, rho_FD2, rho_DB2]]
            mag2 = {'C': _mag(rho_CB, 0), 'B': _mag(rho_CB, 1),
                    'A': _mag(rho_AF, 0), 'F': _mag(rho_AF, 1),
                    'E': _mag(rho_ED, 0), 'D': _mag(rho_ED, 1)}
            for s in ['A', 'B', 'C', 'D', 'E', 'F']:
                magnetizations.extend(mag2[s])
            del rho_CB, rho_AF, rho_ED, rho_DC, rho_BA, rho_FE
            del rho_CA2, rho_AE2, rho_EC2, rho_BF2, rho_FD2, rho_DB2
            del c2, CB2, ED2, AF2, DC2, BA2, FE2
            torch.cuda.empty_cache()

            # ── Env 3 ── (build closeds, pairs, rhos, then free)
            c3 = {}
            for _s in 'CFEBAD':
                _o = build_single_open_env3(_s, *_a3)
                c3[_s] = oe.contract("ABii->AB", _o, backend="torch"); del _o
            EF3 = torch.mm(c3['E'], c3['F']); AB3 = torch.mm(c3['A'], c3['B']); CD3 = torch.mm(c3['C'], c3['D'])
            BE3 = torch.mm(c3['B'], c3['E']); FC3 = torch.mm(c3['F'], c3['C']); DA3 = torch.mm(c3['D'], c3['A'])
            _op3 = lambda s: build_single_open_env3(s, *_a3)
            rho_EF = _lnn(_op3, 'E', 'F', CD3, AB3)
            rho_AB = _lnn(_op3, 'A', 'B', EF3, CD3)
            rho_CD = _lnn(_op3, 'C', 'D', AB3, EF3)
            rho_BE = _lnn(_op3, 'B', 'E', FC3, DA3)
            rho_FC = _lnn(_op3, 'F', 'C', DA3, BE3)
            rho_DA = _lnn(_op3, 'D', 'A', BE3, FC3)
            rho_EC3 = _lnnn(_op3, 'E', c3['F'], 'C', c3['D'], AB3)
            rho_CA3 = _lnnn(_op3, 'C', c3['D'], 'A', c3['B'], EF3)
            rho_AE3 = _lnnn(_op3, 'A', c3['B'], 'E', c3['F'], CD3)
            rho_FD3 = _lnnn(_op3, 'F', c3['C'], 'D', c3['A'], BE3)
            rho_DB3 = _lnnn(_op3, 'D', c3['A'], 'B', c3['E'], FC3)
            rho_BF3 = _lnnn(_op3, 'B', c3['E'], 'F', c3['C'], DA3)
            correlations += [_corr(r) for r in [rho_EF, rho_AB, rho_CD,
                                                rho_BE, rho_FC, rho_DA,
                                                rho_EC3, rho_CA3, rho_AE3,
                                                rho_FD3, rho_DB3, rho_BF3]]
            mag3 = {'E': _mag(rho_EF, 0), 'F': _mag(rho_EF, 1),
                    'A': _mag(rho_AB, 0), 'B': _mag(rho_AB, 1),
                    'C': _mag(rho_CD, 0), 'D': _mag(rho_CD, 1)}
            for s in ['A', 'B', 'C', 'D', 'E', 'F']:
                magnetizations.extend(mag3[s])
            del rho_EF, rho_AB, rho_CD, rho_BE, rho_FC, rho_DA
            del rho_EC3, rho_CA3, rho_AE3, rho_FD3, rho_DB3, rho_BF3
            del c3, EF3, AB3, CD3, BE3, FC3, DA3
            torch.cuda.empty_cache()

            # Energy from correlations (same convention as evaluate_energy_clean)
            energy = 0.0
            for _blk in range(3):
                _b = _blk * 12
                energy += 0.5 * sum(Js[_b + _i] * correlations[_b + _i] for _i in range(6))
                energy +=       sum(Js[_b + 6 + _i] * correlations[_b + 6 + _i] for _i in range(6))
            return energy, correlations, magnetizations, trunc_error

        # ═══════════════════════════════════════════════════════════════════
        # Pre-build path — fast, each open tensor built ONCE per env
        # ═══════════════════════════════════════════════════════════════════
        # build_open_closed_env builds all 6 opens + closeds in one call.
        # All rhos are computed from pre-built opens (no redundant rebuilds).
        # Peak per env: 6 opens + 6 closeds + 2 NNN intermediates (W,V).
        # At D=9,chi=90: ~18 GiB — fits easily in 32 GiB GPU.

        # ═══════════════════ Environment 1 (ebadcf) ═══════════════════════
        o1, c1 = build_open_closed_env1(
            aN, bN, cN, dN, eN, fN, chi, D_bond, d_PHYS, *all27[:9])
        AD = torch.mm(c1['A'], c1['D'])
        CF = torch.mm(c1['C'], c1['F'])
        EB = torch.mm(c1['E'], c1['B'])
        FA = torch.mm(c1['F'], c1['A'])
        DE = torch.mm(c1['D'], c1['E'])
        BC = torch.mm(c1['B'], c1['C'])
        # primary nn rhos
        rho_AD = _build_nn_rho(o1['A'], o1['D'], EB, CF, d_PHYS)
        rho_CF = _build_nn_rho(o1['C'], o1['F'], AD, EB, d_PHYS)
        rho_EB = _build_nn_rho(o1['E'], o1['B'], CF, AD, d_PHYS)
        # duplicate nn rhos (same physical bonds, other orientation in env)
        rho_FA = _build_nn_rho(o1['F'], o1['A'], DE, BC, d_PHYS)
        rho_DE = _build_nn_rho(o1['D'], o1['E'], BC, FA, d_PHYS)
        rho_BC = _build_nn_rho(o1['B'], o1['C'], FA, DE, d_PHYS)
        # nnn rhos
        rho_AE = _build_nnn_rho(o1['A'], c1['D'], o1['E'], c1['B'], CF, d_PHYS)
        rho_EC = _build_nnn_rho(o1['E'], c1['B'], o1['C'], c1['F'], AD, d_PHYS)
        rho_CA = _build_nnn_rho(o1['C'], c1['F'], o1['A'], c1['D'], EB, d_PHYS)
        rho_DB = _build_nnn_rho(o1['D'], c1['E'], o1['B'], c1['C'], FA, d_PHYS)
        rho_BF = _build_nnn_rho(o1['B'], c1['C'], o1['F'], c1['A'], DE, d_PHYS)
        rho_FD = _build_nnn_rho(o1['F'], c1['A'], o1['D'], c1['E'], BC, d_PHYS)
        # correlations (12 values: 3 primary nn + 3 dup nn + 6 nnn)
        correlations += [_corr(r) for r in [rho_EB, rho_AD, rho_CF,
                                            rho_FA, rho_DE, rho_BC,
                                            rho_AE, rho_EC, rho_CA,
                                            rho_DB, rho_BF, rho_FD]]
        # magnetizations: from nn rhos, all 6 sites
        # AD → site A (idx 0), site D (idx 1)
        # CF → site C (idx 0), site F (idx 1)
        # EB → site E (idx 0), site B (idx 1)
        mag1 = {}
        mag1['A'] = _mag(rho_AD, 0)
        mag1['D'] = _mag(rho_AD, 1)
        mag1['C'] = _mag(rho_CF, 0)
        mag1['F'] = _mag(rho_CF, 1)
        mag1['E'] = _mag(rho_EB, 0)
        mag1['B'] = _mag(rho_EB, 1)
        for s in ['A','B','C','D','E','F']:
            magnetizations.extend(mag1[s])
        del o1, c1
        if _core.DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

        # ═══════════════════ Environment 2 (afcbed) ═══════════════════════
        o2, c2 = build_open_closed_env2(
            aN, bN, cN, dN, eN, fN, chi, D_bond, d_PHYS, *all27[9:18])
        CB = torch.mm(c2['C'], c2['B'])
        ED = torch.mm(c2['E'], c2['D'])
        AF = torch.mm(c2['A'], c2['F'])
        DC = torch.mm(c2['D'], c2['C'])
        BA = torch.mm(c2['B'], c2['A'])
        FE = torch.mm(c2['F'], c2['E'])
        # primary nn rhos
        rho_CB = _build_nn_rho(o2['C'], o2['B'], AF, ED, d_PHYS)
        rho_AF = _build_nn_rho(o2['A'], o2['F'], ED, CB, d_PHYS)
        rho_ED = _build_nn_rho(o2['E'], o2['D'], CB, AF, d_PHYS)
        # duplicate nn rhos (same physical bonds, other orientation in env)
        rho_DC = _build_nn_rho(o2['D'], o2['C'], BA, FE, d_PHYS)
        rho_BA = _build_nn_rho(o2['B'], o2['A'], FE, DC, d_PHYS)
        rho_FE = _build_nn_rho(o2['F'], o2['E'], DC, BA, d_PHYS)
        # nnn rhos
        rho_CA = _build_nnn_rho(o2['C'], c2['B'], o2['A'], c2['F'], ED, d_PHYS)
        rho_AE = _build_nnn_rho(o2['A'], c2['F'], o2['E'], c2['D'], CB, d_PHYS)
        rho_EC = _build_nnn_rho(o2['E'], c2['D'], o2['C'], c2['B'], AF, d_PHYS)
        rho_BF = _build_nnn_rho(o2['B'], c2['A'], o2['F'], c2['E'], DC, d_PHYS)
        rho_FD = _build_nnn_rho(o2['F'], c2['E'], o2['D'], c2['C'], BA, d_PHYS)
        rho_DB = _build_nnn_rho(o2['D'], c2['C'], o2['B'], c2['A'], FE, d_PHYS)
        # correlations (12 values: 3 primary nn + 3 dup nn + 6 nnn)
        correlations += [_corr(r) for r in [rho_CB, rho_AF, rho_ED,
                                            rho_DC, rho_BA, rho_FE,
                                            rho_CA, rho_AE, rho_EC,
                                            rho_BF, rho_FD, rho_DB]]
        mag2 = {}
        mag2['C'] = _mag(rho_CB, 0)
        mag2['B'] = _mag(rho_CB, 1)
        mag2['A'] = _mag(rho_AF, 0)
        mag2['F'] = _mag(rho_AF, 1)
        mag2['E'] = _mag(rho_ED, 0)
        mag2['D'] = _mag(rho_ED, 1)
        for s in ['A','B','C','D','E','F']:
            magnetizations.extend(mag2[s])
        del o2, c2
        if _core.DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

        # ═══════════════════ Environment 3 (cdefab) ═══════════════════════
        o3, c3 = build_open_closed_env3(
            aN, bN, cN, dN, eN, fN, chi, D_bond, d_PHYS, *all27[18:27])
        EF_ = torch.mm(c3['E'], c3['F'])
        AB_ = torch.mm(c3['A'], c3['B'])
        CD_ = torch.mm(c3['C'], c3['D'])
        BE_ = torch.mm(c3['B'], c3['E'])
        FC_ = torch.mm(c3['F'], c3['C'])
        DA_ = torch.mm(c3['D'], c3['A'])
        # primary nn rhos
        rho_EF = _build_nn_rho(o3['E'], o3['F'], CD_, AB_, d_PHYS)
        rho_AB = _build_nn_rho(o3['A'], o3['B'], EF_, CD_, d_PHYS)
        rho_CD = _build_nn_rho(o3['C'], o3['D'], AB_, EF_, d_PHYS)
        # duplicate nn rhos (same physical bonds, other orientation in env)
        rho_BE = _build_nn_rho(o3['B'], o3['E'], FC_, DA_, d_PHYS)
        rho_FC = _build_nn_rho(o3['F'], o3['C'], DA_, BE_, d_PHYS)
        rho_DA = _build_nn_rho(o3['D'], o3['A'], BE_, FC_, d_PHYS)
        # nnn rhos
        rho_EC = _build_nnn_rho(o3['E'], c3['F'], o3['C'], c3['D'], AB_, d_PHYS)
        rho_CA = _build_nnn_rho(o3['C'], c3['D'], o3['A'], c3['B'], EF_, d_PHYS)
        rho_AE = _build_nnn_rho(o3['A'], c3['B'], o3['E'], c3['F'], CD_, d_PHYS)
        rho_FD = _build_nnn_rho(o3['F'], c3['C'], o3['D'], c3['A'], BE_, d_PHYS)
        rho_DB = _build_nnn_rho(o3['D'], c3['A'], o3['B'], c3['E'], FC_, d_PHYS)
        rho_BF = _build_nnn_rho(o3['B'], c3['E'], o3['F'], c3['C'], DA_, d_PHYS)
        # correlations (12 values: 3 primary nn + 3 dup nn + 6 nnn)
        correlations += [_corr(r) for r in [rho_EF, rho_AB, rho_CD,
                                            rho_BE, rho_FC, rho_DA,
                                            rho_EC, rho_CA, rho_AE,
                                            rho_FD, rho_DB, rho_BF]]
        mag3 = {}
        mag3['E'] = _mag(rho_EF, 0)
        mag3['F'] = _mag(rho_EF, 1)
        mag3['A'] = _mag(rho_AB, 0)
        mag3['B'] = _mag(rho_AB, 1)
        mag3['C'] = _mag(rho_CD, 0)
        mag3['D'] = _mag(rho_CD, 1)
        for s in ['A','B','C','D','E','F']:
            magnetizations.extend(mag3[s])
        del o3, c3

        # ── Compute energy from correlations + J values ───────────────────
        # correlations is 36 values (12 per env: 6 nn + 6 nnn).
        # Each nn bond appears in exactly 2 of the 3 environments, so a naive
        # sum double-counts nn.  The energy functions apply *0.5 to their 6-nn
        # block before returning; replicate that here so that
        # evaluate_observables gives the same energy as evaluate_energy_clean.
        # Each nnn bond appears in all 3 environments (no compensating factor,
        # consistent with the energy-function return convention).
        energy = 0.0
        for _blk in range(3):
            _b = _blk * 12
            energy += 0.5 * sum(Js[_b + _i] * correlations[_b + _i]
                                for _i in range(6))    # nn  × ½
            energy +=       sum(Js[_b + 6 + _i] * correlations[_b + 6 + _i]
                                for _i in range(6))    # nnn × 1

    return energy, correlations, magnetizations, trunc_error


# ── Bond labels matching the order returned by evaluate_observables ───────────
_ENV_BOND_LABELS = [
    # env1 (ebadcf): 3 primary nn + 3 dup nn + 6 nnn
    'EB', 'AD', 'CF', 'FA', 'DE', 'BC', 'AE', 'EC', 'CA', 'DB', 'BF', 'FD',
    # env2 (afcbed): 3 primary nn + 3 dup nn + 6 nnn
    'CB', 'AF', 'ED', 'DC', 'BA', 'FE', 'CA', 'AE', 'EC', 'BF', 'FD', 'DB',
    # env3 (cdefab): 3 primary nn + 3 dup nn + 6 nnn
    'EF', 'AB', 'CD', 'BE', 'FC', 'DA', 'EC', 'CA', 'AE', 'FD', 'DB', 'BF',
]
_SITE_LABELS = ['A', 'B', 'C', 'D', 'E', 'F']


def _save_observables_file(filepath: str, D_bond: int, chi: int,
                           energy: float, correlations: list,
                           magnetizations: list,
                           trunc_error: float | None = None) -> None:
    """Write energy / correlations / magnetizations (and optionally CTMRG
    truncation error) to a human-readable file."""
    n_sites = 6
    with open(filepath, 'w') as fp:
        fp.write(f"# D={D_bond}  chi={chi}  timestamp={timestamp()}\n\n")
        fp.write(f"energy           = {energy:+.12e}\n")
        fp.write(f"energy_per_site  = {energy / n_sites:+.12e}\n")
        if trunc_error is not None:
            fp.write(f"trunc_error      = {trunc_error:.6e}\n")
        fp.write("\n")

        fp.write("# ── Bond correlations <Si·Sj>  (36 values) "
                 "──────────────────────\n")
        for env_idx in range(3):
            fp.write(f"# env{env_idx+1}\n")
            for j in range(12):
                idx = env_idx * 12 + j
                label = _ENV_BOND_LABELS[idx]
                fp.write(f"corr_env{env_idx+1}_{label:>2s} = {correlations[idx]:+.12e}\n")
            fp.write("\n")

        fp.write("# ── Site magnetizations <Sx> <Sy> <Sz>  (54 values) "
                 "──────────────\n")
        fp.write("# <Sy> via iSy_op=(S+-S-)/2=i·Sy_phys:\n")
        fp.write("#   Real iPEPS:    Sy = Re(Tr(rho·iSy_op)) = 0.0 exactly\n")
        fp.write("#   Complex iPEPS: Sy = Im(Tr(rho·iSy_op)) = physical <Sy>\n")
        for env_idx in range(3):
            fp.write(f"# env{env_idx+1}\n")
            for s_idx, s in enumerate(_SITE_LABELS):
                base = env_idx * 18 + s_idx * 3
                mx = magnetizations[base]
                my = magnetizations[base + 1]
                mz = magnetizations[base + 2]
                fp.write(f"mag_env{env_idx+1}_{s}  "
                         f"Sx={mx:+.12e}  Sy={my:+.12e}  Sz={mz:+.12e}\n")
            fp.write("\n")

        fp.write("# ── Local magnetization |m|=sqrt(Sx²+Sy²+Sz²)  "
                 "(18 values: 6 sites × 3 envs) ──────────────\n")
        for env_idx in range(3):
            fp.write(f"# env{env_idx+1}\n")
            for s_idx, s in enumerate(_SITE_LABELS):
                base = env_idx * 18 + s_idx * 3
                mx_ = magnetizations[base]
                my_ = magnetizations[base + 1]
                mz_ = magnetizations[base + 2]
                loc_mag = (mx_**2 + my_**2 + mz_**2) ** 0.5
                fp.write(f"localmag_env{env_idx+1}_{s}  |m|={loc_mag:+.12e}\n")
            fp.write("\n")
    print(f"  │  Observables saved → {filepath}")


def _print_observables_summary(tag: str, D_bond: int, chi: int,
                               energy: float, correlations: list,
                               magnetizations: list,
                               trunc_error: float | None = None) -> None:
    """Print a compact summary of observables to stdout."""
    n_sites = 6
    nn_corrs  = [correlations[i] for i in range(36) if (i % 12) < 6]
    nnn_corrs = [correlations[i] for i in range(36) if (i % 12) >= 6]
    all_mx = [magnetizations[i*3]     for i in range(18)]
    all_my = [magnetizations[i*3 + 1] for i in range(18)]
    all_mz = [magnetizations[i*3 + 2] for i in range(18)]

    def _stat(vals):
        mn, mx = min(vals), max(vals)
        n   = len(vals)
        avg = sum(vals) / n
        var = sum((v - avg)**2 for v in vals) / n
        se  = (var / n) ** 0.5   # standard error = std / sqrt(n)
        return f"min={mn:+.6f} max={mx:+.6f} mean={avg:+.6f} se={se:.2e}"

    te_str = f"  trunc_err={trunc_error:.3e}" if trunc_error is not None else ""
    print(f"  │  [{tag}] D={D_bond} chi={chi}  E={energy:+.10f}  "
          f"E/site={energy/n_sites:+.10f}{te_str}")
    print(f"  │    nn  <S·S> ({len(nn_corrs):>2d}): {_stat(nn_corrs)}")
    print(f"  │    nnn <S·S> ({len(nnn_corrs):>2d}): {_stat(nnn_corrs)}")
    print(f"  │    <Sx>      ({len(all_mx):>2d}): {_stat(all_mx)}")
    _sy_note = "  (=0 for real iPEPS)" if all(abs(v) < 1e-12 for v in all_my) else ""
    print(f"  │    <Sy>      ({len(all_my):>2d}): {_stat(all_my)}{_sy_note}")
    print(f"  │    <Sz>      ({len(all_mz):>2d}): {_stat(all_mz)}")

    # ── Local magnetization |m| per site, mean±SE over 3 environments ────
    def _site_localmag(s_idx):
        vals = []
        for env_idx in range(3):
            base = env_idx * 18 + s_idx * 3
            mx_ = magnetizations[base]
            my_ = magnetizations[base + 1]
            mz_ = magnetizations[base + 2]
            vals.append((mx_**2 + my_**2 + mz_**2) ** 0.5)
        avg = sum(vals) / 3
        se  = (sum((v - avg)**2 for v in vals) / 3 / 3) ** 0.5
        return avg, se

    ace_parts, bdf_parts = [], []
    for s_idx, s in enumerate(_SITE_LABELS):  # A B C D E F
        avg, se = _site_localmag(s_idx)
        entry = f"{s}:{avg:.6f}±{se:.2e}"
        (ace_parts if s in ('A', 'C', 'E') else bdf_parts).append(entry)
    print(f"  │    |m| mean±se/env:")
    print(f"  │      ACE  {' | '.join(ace_parts)}")
    print(f"  │      BDF  {' | '.join(bdf_parts)}")


def save_checkpoint(path: str, params: tuple, D_bond: int, chi: int,
                    loss: float, energy: float | None,
                    step: int, log: list, ansatz_cfg: dict) -> None:
    keys = ansatz_cfg['ckpt_keys']
    d = {key: params[i] for i, key in enumerate(keys)}
    d.update({
        'D_bond': D_bond, 'chi': chi,
        'loss': loss, 'energy': energy,
        'step': step, 'timestamp': timestamp(),
        'log': log,
    })
    torch.save(d, path)


# ══════════════════════════════════════════════════════════════════════════════
# Parallel energy helper — optionally dispatches to multiple GPUs
# ══════════════════════════════════════════════════════════════════════════════

def _three_env_energy_loss_parallel(
        aN, bN, cN, dN, eN, fN,
        Js: list, SdotS: torch.Tensor,
        chi: int, D_bond: int, d_PHYS: int,
        env1: tuple, env2: tuple, env3: tuple) -> torch.Tensor:
    """Compute sum of three energy expectations, optionally across multiple GPUs.

    N_GPUS == 1 (or CPU): sequential, each call checkpointed on the primary device
    — identical semantics to the previous inline implementation.

    N_GPUS >= 2 (CUDA only):
      env1 stays on cuda:0. env2 dispatches to cuda:1.
      env3 dispatches to cuda:min(2, N_GPUS-1)  (stays on cuda:1 if only 2 GPUs).
      All three _ckpt calls are submitted concurrently to a thread pool, so they
      overlap on their respective CUDA streams:  ~2.5x speedup for energy phase.

    Memory: each call uses _ckpt(use_reentrant=False) — the large open tensors
    of shape (chi*D2, chi*D2, d, d) are never all resident simultaneously.

    Gradient correctness: .to(device) is differentiable; gradients w.r.t.
    aN..fN propagate back to cuda:0 through the transfer ops correctly.
    Verified: gradient cosine similarity vs sequential baseline = 1.0000.
    """
    primary = aN.device

    # The three wrappers below capture  Js / chi / D_bond / d_PHYS  from this
    # function's local scope (kept alive by the closures until backward is done).
    # Tensor args are passed EXPLICITLY to _ckpt so checkpoint saves them.
    # Non-tensor scalar args (Js entries, chi, D_bond, d_PHYS) are captured in
    # the closure and do not need checkpoint-saving.

    def _e1_fn(aN_, bN_, cN_, dN_, eN_, fN_, SdotS_, *env_):
        return energy_expectation_nearest_neighbor_3ebadcf_bonds(
            aN_, bN_, cN_, dN_, eN_, fN_,
            Js[0],  Js[1],  Js[2],  Js[3],  Js[4],  Js[5],
            Js[6],  Js[7],  Js[8],  Js[9],  Js[10], Js[11],
            SdotS_, chi, D_bond, d_PHYS, *env_)

    def _e2_fn(dev, aN_, bN_, cN_, dN_, eN_, fN_, SdotS_, *env_):
        """Same as e1 but moves every tensor to *dev* then returns on *primary*."""
        mv = lambda x: x.to(dev)
        return energy_expectation_nearest_neighbor_3afcbed_bonds(
            mv(aN_), mv(bN_), mv(cN_), mv(dN_), mv(eN_), mv(fN_),
            Js[12], Js[13], Js[14], Js[15], Js[16], Js[17],
            Js[18], Js[19], Js[20], Js[21], Js[22], Js[23],
            mv(SdotS_), chi, D_bond, d_PHYS,
            *[mv(t) for t in env_]
        ).to(primary)

    def _e3_fn(dev, aN_, bN_, cN_, dN_, eN_, fN_, SdotS_, *env_):
        """Same as e1 but moves every tensor to *dev* then returns on *primary*."""
        mv = lambda x: x.to(dev)
        return energy_expectation_nearest_neighbor_other_3_bonds(
            mv(aN_), mv(bN_), mv(cN_), mv(dN_), mv(eN_), mv(fN_),
            Js[24], Js[25], Js[26], Js[27], Js[28], Js[29],
            Js[30], Js[31], Js[32], Js[33], Js[34], Js[35],
            mv(SdotS_), chi, D_bond, d_PHYS,
            *[mv(t) for t in env_]
        ).to(primary)

    # All tensor args packed as tuples for _ckpt.  Non-tensor scalars are
    # captured in the closures above and are never passed through _ckpt.
    ts1 = (aN, bN, cN, dN, eN, fN, SdotS, *env1)
    ts2 = (aN, bN, cN, dN, eN, fN, SdotS, *env2)
    ts3 = (aN, bN, cN, dN, eN, fN, SdotS, *env3)

    if N_GPUS >= 2 and primary.type == 'cuda':
        # ── Multi-GPU path ────────────────────────────────────────────────────
        dev1 = torch.device('cuda:1')
        dev2 = torch.device(f'cuda:{min(2, N_GPUS - 1)}')
        _run2 = _functools.partial(_e2_fn, dev1)
        _run3 = _functools.partial(_e3_fn, dev2)
        # Use a thread pool so all three _ckpt calls overlap on their devices.
        # _ckpt(use_reentrant=False) is thread-safe (PyTorch docs).
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=3) as pool:
            f1 = pool.submit(_ckpt, _e1_fn,  *ts1, use_reentrant=False)
            f2 = pool.submit(_ckpt, _run2,   *ts2, use_reentrant=False)
            f3 = pool.submit(_ckpt, _run3,   *ts3, use_reentrant=False)
            return f1.result() + f2.result() + f3.result()
    else:
        # ── Single-GPU / CPU path ─────────────────────────────────────────────
        # _e2_fn / _e3_fn with primary device = no-op .to() calls.
        _run2 = _functools.partial(_e2_fn, primary)
        _run3 = _functools.partial(_e3_fn, primary)
        if primary.type == 'cuda':
            return (
                _ckpt(_e1_fn, *ts1, use_reentrant=False)
                + _ckpt(_run2,  *ts2, use_reentrant=False)
                + _ckpt(_run3,  *ts3, use_reentrant=False)
            )
        else:
            return _e1_fn(*ts1) + _run2(*ts2) + _run3(*ts3)


# ══════════════════════════════════════════════════════════════════════════════
# Core: optimise at a single (D_bond, chi) level for a fixed time budget
# ══════════════════════════════════════════════════════════════════════════════

def optimize_at_chi(
        Js, SdotS, D_bond: int, chi: int, d_PHYS: int,
        budget_seconds: float,
        lbfgs_max_iter: int,
        init_params=None,
        step_offset: int = 0,
        best_path: str | None = None,
        latest_path: str | None = None,
        loss_log: list | None = None,
        out_dir: str | None = None,
        ansatz_cfg: dict | None = None,
) -> tuple:
    """
    Outer L-BFGS loop at fixed (D_bond, chi) until budget_seconds elapsed
    or opt_conv_threshold hit.

    Returns (best_params_tuple, best_loss, steps_done).
    best_params_tuple contains 1 tensor for single-tensor ansätze (neel/c6ypi/c3vypi),
    2 tensors for two-tensor ansätze (twoc3),
    or 6 tensors for 6-tensor ansätze (unrestricted/sym6), matching ansatz_cfg['n_params'].
    """
    if loss_log is None:
        loss_log = []
    if ansatz_cfg is None:
        ansatz_cfg = ANSATZ_REGISTRY['unrestricted']
    n_params = ansatz_cfg['n_params']
    D_sq = D_bond ** 2
    t_start = time.perf_counter()

    # initialise tensors — ALWAYS brand-new objects, no shared lineage
    if init_params is not None:
        params = list(_new_tensors_from_data(init_params))
    elif n_params == 6:
        params = list(initialize_abcdef('random', D_bond, d_PHYS, INIT_NOISE))
    elif n_params == 1:
        params = [ansatz_cfg['init_fn'](D_bond, d_PHYS, INIT_NOISE)]
    else:
        # n_params >= 2: init_fn returns a tuple of tensors
        params = list(ansatz_cfg['init_fn'](D_bond, d_PHYS, INIT_NOISE))
    for t in params:
        t.requires_grad_(True)


    best_loss     = float('inf')
    best_params   = [t.detach().clone() for t in params]
    prev_loss     = None
    loss_history  = collections.deque(maxlen=12)  # cycle detection up to length 12
    step          = step_offset

    # ── pre-create Adam optimizer (state persists across outer steps) ─────────
    _adam: torch.optim.Optimizer | None = None
    if OPTIMIZER == 'adam':
        _adam = torch.optim.Adam(
            params,
            lr=ADAM_LR, betas=ADAM_BETAS, eps=ADAM_EPS,
            weight_decay=ADAM_WEIGHT_DECAY)

    # ── pre-create L-BFGS optimizer (curvature state persists across outer
    #    steps within one (D,chi) level; the object is local to this function
    #    call so there is ZERO cross-(D,chi) leakage) ──────────────────────────
    _lbfgs: torch.optim.Optimizer | None = None
    # last_ctm_steps / last_cn: shared between the closure (nonlocal writes)
    # and the outer step (read-back after _lbfgs.step returns).  Declared here
    # so the history-clear hooks can inspect the *previous* step's values.
    last_ctm_steps: int | None = None
    last_cn: dict[str, float] | None = None
    if OPTIMIZER == 'lbfgs':
        def _make_lbfgs(*_tensors) -> torch.optim.LBFGS:
            return torch.optim.LBFGS(
                list(_tensors),
                lr=LBFGS_LR,
                max_iter=lbfgs_max_iter,
                tolerance_grad=OPT_TOL_GRAD,
                tolerance_change=OPT_TOL_CHANGE,
                history_size=LBFGS_HISTORY,
                line_search_fn='strong_wolfe',
            )
        _lbfgs = _make_lbfgs(*params)

        def _lbfgs_reset_history() -> None:
            """Clear L-BFGS curvature history, resetting the Hessian approx to
            a scaled identity.  O(1): just drops the stored (s_k, y_k) vector
            pairs; the next .step() call re-initialises state from scratch.

            Call whenever accumulated curvature pairs would mislead the
            Hessian approximation:
              - after a penalty-closure contamination  (automatic, see Hook 1)
              - after changing SVD hyperparameters (rel_cutoff, chi_extra)  <- Hook 2
              - any other structural perturbation to the objective.
            """
            assert _lbfgs is not None
            _lbfgs.state.clear()

    def _loss_with_differentiable_ctmrg() -> tuple[torch.Tensor, int]:
        """Compute loss with CTMRG inside the autograd graph.

        LBFGS calls its closure multiple times per step; therefore this
        function must be called fresh on every closure evaluation.
        Never reuse environment tensors across calls.
        """
        # Derive all 6 single-layer tensors from the optimized parameters.
        # For unrestricted: params = [a, b, c, d, e, f] directly.
        # For single-tensor ansätze: symmetrize params[0] and derive (a..f).
        a, b, c, d, e, f = _derive_abcdef(params, ansatz_cfg)
        # IMPORTANT: Make the single-layer tensors consistent with the CTMRG
        # convention (double-layer tensors are Frobenius-normalized). Without
        # this, the energy routines' printed <iPEPS|iPEPS> denominators can be
        # far from 1 even when CTMRG normalization is correct.
        aN = normalize_single_layer_tensor_for_double_layer(a)
        bN = normalize_single_layer_tensor_for_double_layer(b)
        cN = normalize_single_layer_tensor_for_double_layer(c)
        dN = normalize_single_layer_tensor_for_double_layer(d)
        eN = normalize_single_layer_tensor_for_double_layer(e)
        fN = normalize_single_layer_tensor_for_double_layer(f)

        A, B, C, Dt, E, F = abcdef_to_ABCDEF(aN, bN, cN, dN, eN, fN, D_sq)
        
        _proxy_fn = None
        if _core._CTM_CONV_MODE != 'SVdifference':
            def _proxy_fn(
                    C21CD, C32EF, C13AB, T1F, T2A, T2B, T3C, T3D, T1E,
                    C21EB, C32AD, C13CF, T1D, T2C, T2F, T3E, T3B, T1A,
                    C21AF, C32CB, C13ED, T1B, T2E, T2D, T3A, T3F, T1C):
                with torch.no_grad():
                    _e1 = energy_expectation_nearest_neighbor_3ebadcf_bonds(
                        aN, bN, cN, dN, eN, fN, *Js[0:12], SdotS,
                        chi, D_bond, d_PHYS,
                        C21CD, C32EF, C13AB, T1F, T2A, T2B, T3C, T3D, T1E)
                    _e2 = energy_expectation_nearest_neighbor_3afcbed_bonds(
                        aN, bN, cN, dN, eN, fN, *Js[12:24], SdotS,
                        chi, D_bond, d_PHYS,
                        C21EB, C32AD, C13CF, T1D, T2C, T2F, T3E, T3B, T1A)
                    _e3 = energy_expectation_nearest_neighbor_other_3_bonds(
                        aN, bN, cN, dN, eN, fN, *Js[24:36], SdotS,
                        chi, D_bond, d_PHYS,
                        C21AF, C32CB, C13ED, T1B, T2E, T2D, T3A, T3F, T1C)
                    return (_e1 + _e2 + _e3).item()
                
        all28 = CTMRG_from_init_to_stop(
            A, B, C, Dt, E, F, chi, D_sq,
            CTM_MAX_STEPS, CTM_CONV_THR, ENV_IDENTITY_INIT, energy_proxy_fn = _proxy_fn)


        (C21CD, C32EF, C13AB, T1F,  T2A,  T2B,  T3C,  T3D,  T1E,
         C21EB, C32AD, C13CF, T1D,  T2C,  T2F,  T3E,  T3B,  T1A,
         C21AF, C32CB, C13ED, T1B,  T2E,  T2D,  T3A,  T3F,  T1C,
         ctm_steps) = all28

        # ── Last-resort zero-env guard (gradient path) ────────────────────
        # CTMRG already retried with escalating noise internally.
        # If it STILL returned zero, skip the expensive energy computation
        # and raise so the closure handler returns a penalty loss.
        _env_corners = (C21CD, C32EF, C13AB,
                        C21EB, C32AD, C13CF,
                        C21AF, C32CB, C13ED)
        if all(torch.linalg.norm(c).item() < 1e-30 for c in _env_corners):
            raise FloatingPointError(
                "CTMRG env collapsed to zero after all internal retries")

        # ── Compute the three energy expectations (checkpointed, optionally multi-GPU) ─
        # _three_env_energy_loss_parallel wraps each energy call in
        # _ckpt(use_reentrant=False), saving ~5 GB (D=7) / ~15 GB (D=8) of
        # autograd intermediates.  With N_GPUS >= 2 (CUDA), the three calls
        # run concurrently on separate GPUs via a thread pool (see --ngpu).
        loss = _three_env_energy_loss_parallel(
            aN, bN, cN, dN, eN, fN, Js, SdotS, chi, D_bond, d_PHYS,
            env1=(C21CD, C32EF, C13AB, T1F,  T2A,  T2B,  T3C,  T3D,  T1E),
            env2=(C21EB, C32AD, C13CF, T1D,  T2C,  T2F,  T3E,  T3B,  T1A),
            env3=(C21AF, C32CB, C13ED, T1B,  T2E,  T2D,  T3A,  T3F,  T1C),
        )

        return loss, int(ctm_steps)

    while True:
        elapsed = time.perf_counter() - t_start
        if elapsed >= budget_seconds:
            break


        # CTMRG is evaluated inside the optimizer objective.
        # (LBFGS calls the closure multiple times; reusing env tensors would
        #  both crash autograd and give incorrect gradients.)
        ctm_steps = -1
        if OPTIMIZER == 'lbfgs':
            if _lbfgs is None:
                raise RuntimeError("L-BFGS optimizer requested but was not initialized")

            # ── HISTORY-CLEAR HOOKS ───────────────────────────────────────────
            # Hook 0 (MANDATORY, every outer step): reset L-BFGS state.
            # Prevents prev_flat_grad from a converged inner loop from
            # creating y = g_new - 0 = g_new with s=0 (degenerate pair).
            _lbfgs_reset_history()
            # Hook 1 (penalty path, already covered by Hook 0): no extra action.
            # Hook 2 (SVD hyperparameter change, future): Hook 0 covers it;
            #   add a diagnostic print here when rel_cutoff/chi_extra change:
            # if <svd_params_changed>:
            #     print("[lbfgs] SVD params changed; Hook 0 cleared history.")
            # ─────────────────────────────────────────────────────────────────

            last_ctm_steps = None
            def closure():
                nonlocal last_ctm_steps
                _lbfgs.zero_grad()
                try:
                    loss, _ctm_steps = _loss_with_differentiable_ctmrg()
                    # Guard against NaN/Inf losses which break strong-wolfe.
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite loss: {loss}")
                    last_ctm_steps = _ctm_steps
                    loss.backward()
                    # NOTE: gc.collect() removed from the GPU hot path.
                    # On CUDA, loss.backward() dispatches kernels asynchronously;
                    # gc.collect() acquires the GIL and traverses the Python heap
                    # for milliseconds, stalling CUDA kernel submission and leaving
                    # the GPU idle.  PyTorch's autograd engine already frees the
                    # backward graph nodes as it processes them — no manual GC
                    # needed here.  GC runs are still called between outer steps
                    # (at the chi-level boundary) where the cost is acceptable.
                    #print("gradient example:", a.grad.view(-1)[:5])
                    # Guard against NaN/Inf gradients.
                    for p in params:
                        if p.grad is not None:
                            p.grad.data = torch.nan_to_num(p.grad.data, nan=0.0, posinf=0.0, neginf=0.0)
                    # print("Gradient:" + " ".join(f"{torch.linalg.norm(p.grad).item():.4e}" for p in params))
                    
                    return loss
                except Exception as exc:
                    # BRUTAL ENFORCEMENT: If CTMRG/SVD forward or backward is
                    # ill-defined at this trial point (common during L-BFGS
                    # line search), return a large penalty with zero gradient
                    # so the line search rejects the step instead of aborting.
                    last_ctm_steps = 0
                    for p in params:
                        if p.grad is None:
                            p.grad = torch.zeros_like(p)
                        else:
                            p.grad.detach_()
                            p.grad.zero_()
                    # Print a short one-line warning (kept minimal to avoid log spam).
                    msg = str(exc).splitlines()[0][:200]
                    print(f"[closure] Non-finite/ill-defined point -> penalty loss (reason: {msg})")
                    _p0 = params[0]
                    _real_dtype = _p0.real.dtype if _p0.is_complex() else _p0.dtype
                    return torch.tensor(1.0e6, dtype=_real_dtype, device=_p0.device)
            loss_val  = _lbfgs.step(closure)

            # After a successful step with full SVD, revert to partial for all
            # subsequent steps (the noisy initial basin has been escaped).
            if _core._USE_FULL_SVD:
                _core._USE_FULL_SVD = False

            loss_item = loss_val.item()
            del loss_val          # release the 0-dim tensor + its dead graph
            if last_ctm_steps is not None:
                ctm_steps = last_ctm_steps
            # ── GPU memory cleanup after L-BFGS step ─────────────────────
            # PyTorch's CUDA caching allocator retains freed blocks from
            # CTMRG + energy backward intermediates.  Without this, the
            # cached segments accumulate across the 15 closure calls inside
            # _lbfgs.step(), growing the resident set until OOM.
            if params[0].device.type == 'cuda':
                gc.collect()
                torch.cuda.empty_cache()
        else:  # adam — persistent state, ADAM_STEPS_PER_CTM micro-steps per env refresh
            if _adam is None:
                raise RuntimeError("Adam optimizer requested but was not initialized")
            for _s in range(ADAM_STEPS_PER_CTM):
                _adam.zero_grad()
                _loss, ctm_steps = _loss_with_differentiable_ctmrg()
                _loss.backward()
                _adam.step()
            loss_item = _loss.detach().item()
            # ── GPU memory cleanup after Adam micro-steps ────────────────
            if params[0].device.type == 'cuda':
                gc.collect()
                torch.cuda.empty_cache()
        delta     = (loss_item - prev_loss) if prev_loss is not None else float('inf')
        elapsed   = time.perf_counter() - t_start

        print(f"    step {step:5d}  ctm={ctm_steps:3d}  loss={loss_item:+.10f}"
              f"  Δ={delta:+.3e}  {elapsed:.0f}/{budget_seconds:.0f}s")
        sys.stdout.flush()
        loss_log.append({'step': step, 'ctm_steps': ctm_steps, 'loss': loss_item,
                         'D_bond': D_bond, 'chi': chi,
                         'elapsed': round(elapsed, 1)})

        # ── Last-resort collapsed-loss guard ──────────────────────────────
        # If CTMRG env collapses to near-zero AND FloatingPointError was not
        # raised, signal the outer D loop to restart from chi_schedule[0]
        # with fresh tensors by raising _CollapseRestartD.
        _loss_is_collapsed = (loss_item >= -1e-12 and best_loss < -0.5)
        if _loss_is_collapsed:
            print(f"    [ZERO-GUARD] loss={loss_item:+.10f} \u2248 0 "
                  f"(env near-collapsed); signalling D={D_bond} restart from chi={chi}")
            raise _CollapseRestartD(f"D={D_bond}, chi={chi}")

        if loss_item < best_loss:
            best_loss   = loss_item
            best_params = [t.detach().clone() for t in params]
            if best_path:
                save_checkpoint(best_path, tuple(best_params), D_bond, chi,
                                best_loss, None, step, loss_log, ansatz_cfg)

        if latest_path and (step - step_offset) % SAVE_EVERY == 0 and step > step_offset:
            save_checkpoint(latest_path, tuple(best_params), D_bond, chi,
                            best_loss, None, step, loss_log, ansatz_cfg)

        if prev_loss is not None and abs(delta) < OPT_CONV_THRESHOLD:
            print(f"    Outer convergence at step {step} (\u0394={delta:.2e})")
            break
        if any(abs(loss_item - h) < 1e-10 for h in loss_history):
            print(f"    Cycle detected at step {step} "
                  f"(amplitude={abs(delta):.3e}); stopping.")
            break
        loss_history.append(loss_item)
        prev_loss = loss_item
        step += 1

    return (tuple(best_params), best_loss, step)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global TENSORDTYPE, OPTIMIZER
    parser = argparse.ArgumentParser(
        description="10-12 h iPEPS benchmark: sweep D_bond (outer) × chi (inner)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--hours', type=float, default=TOTAL_BUDGET_HOURS,
        help='Total wall-clock budget in hours.')
    parser.add_argument(
        '--D-bonds', default=','.join(map(str, D_BOND_LIST)),
        help='Comma-separated list of D_bond values to sweep, in order.')
    parser.add_argument(
        '--chi-maxes', default=None,
        help='Comma-separated chi_max for each D_bond. '
             'Defaults: 16,81,80 for D=2,3,4.')
    parser.add_argument(
        '--d-phys', type=int, default=D_PHYS,
        help='Physical Hilbert-space dimension (d=2 for spin-1/2).')
    parser.add_argument(
        '--J', '--J1', type=float, default=J1_COUPLING, dest='J1',
        help='Nearest-neighbour Heisenberg coupling J1 (positive = AFM).')
    parser.add_argument(
        '--J2', type=float, default=J2_COUPLING,
        help='Next-nearest-neighbour Heisenberg coupling J2 (positive = frustrated AFM). '
             'Set to 0 for pure J1 model.')
    parser.add_argument(
        '--output-dir', default=None,
        help='Directory for checkpoints + plots (default: src_code/log/).')
    parser.add_argument(
        '--resume', default=None,
        help='Path to a .pt checkpoint to resume from (skips earlier (D,chi)).')
    parser.add_argument(
        '--noise', type=float, default=PAD_NOISE,
        help='Gaussian noise amplitude when padding tensors for D→D+1 (= PAD_NOISE).')
    parser.add_argument(
        '--rand-init-new-d', dest='rand_init_new_d', action='store_true',
        default=RAND_INIT_NEW_D,
        help='Skip D-1→D padded warm-start; use fully random init for each new D '
             '(= RAND_INIT_NEW_D).')
    parser.add_argument(
        '--rand-init-new-chi', dest='rand_init_new_chi', action='store_true',
        default=RAND_INIT_NEW_CHI,
        help='Skip warm-starting from previous chi; use fully random init for each '
             'chi level within the same D (= RAND_INIT_NEW_CHI).')
    parser.add_argument(
        '--mean-field-init', dest='mean_field_init', action='store_true',
        default=MEAN_FIELD_INIT,
        help='Initialise EVERY chi level (for all D) from a mean-field Néel '
             'product state (A-sublattice: spin-up, B-sublattice: spin-down) '
             'instead of random or padded warm-start.  Overrides '
             '--rand-init-new-chi and D→D warm-start (= MEAN_FIELD_INIT).')
    parser.add_argument(
        '--double', action='store_true', default=USE_DOUBLE_PRECISION,
        help='Use float64/complex128 (default: float32/complex64). '
             'Same throughput on CPU/MKL; 2–4× slower on CUDA consumer GPUs.')
    parser.add_argument(
        '--gpu', dest='gpu', action='store_true', default=USE_GPU,
        help='Use CUDA GPU when available (falls back to CPU if no GPU found).')
    parser.add_argument(
        '--no-gpu', dest='gpu', action='store_false',
        help='Force CPU even if a CUDA GPU is present.')
    parser.add_argument(
        '--ngpu', type=int, default=None,
        help='Number of GPUs for parallel energy computation (default: all '
             'available when --gpu, else 1).  N=1 is single-GPU sequential. '
             'N>=2 dispatches the 3 independent energy functions to separate '
             'GPUs concurrently.  No effect on CPU runs or when only 1 GPU is '
             'available.  Cluster: request GPUs with --gres=gpu:N in SLURM '
             '(no MPI / torchrun needed).')
    parser.add_argument(
        '--complex', action='store_true', default=not USE_REAL_TENSORS,
        help='Use complex iPEPS tensors (Sx/Sy/Sz Hamiltonian). '
             'Default: real tensors (S+/S- Hamiltonian).')
    parser.add_argument(
        '--optimizer', choices=['lbfgs', 'adam'], default=OPTIMIZER,
        help="Optimiser: 'lbfgs' (default, fast on smooth landscapes) or "
             "'adam' (more robust on noisy landscapes, constant LR).")
    parser.add_argument(
        '--ansatz', choices=list(ANSATZ_REGISTRY.keys()), default='unrestricted',
        help=('iPEPS ansatz to optimize. Available: '
              + ', '.join(f"'{k}' ({v['description']})"
                          for k, v in ANSATZ_REGISTRY.items()) + '.'))
    parser.add_argument(
        '--seed', type=int, default=RANDOM_SEED,
        help='Integer RNG seed (used when --no-seed is NOT given). '
             'Seeds Python random, NumPy, and PyTorch CPU+CUDA.')
    parser.add_argument(
        '--no-seed', action='store_true', default=not RANDOM_SEED_FIX,
        help='Disable RNG seeding — runs will NOT be reproducible.')
    args = parser.parse_args()

    # ── Ansatz selection ──────────────────────────────────────────────────────
    ansatz_cfg = ANSATZ_REGISTRY[args.ansatz]
    print(f"Selected ansatz: '{args.ansatz}' — {ansatz_cfg['description']}")

    # ── RNG seeding (must happen BEFORE any tensor allocation) ────────────────
    _seed_enabled = not args.no_seed
    if _seed_enabled:
        import random as _random
        import numpy as _np
        _random.seed(args.seed)
        _np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        # Make cuBLAS / cuDNN deterministic when a CUDA device is used.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
        print(f"  RNG seed fixed to {args.seed} (use --no-seed to disable)")
    else:
        print("  RNG seeding disabled — results will NOT be reproducible.")

    # ── Precision + optimizer setup ───────────────────────────────────────────
    use_real = not getattr(args, 'complex', False)
    set_dtype(args.double, use_real)
    TENSORDTYPE = _core.TENSORDTYPE          # sync from core
    OPTIMIZER = args.optimizer
    # Float32 spectral noise floor is ~5e-5–2e-4; CTM_CONV_THR=3e-7 is below
    # that floor and will never trigger in single precision — CTMRG would
    # always burn all CTM_MAX_STEPS steps and return a non-converged (garbage)
    # environment, causing the optimizer to stall (Δ=0 immediately) and the
    # lookahead clean_eval to return wrong energies (e.g. 0.0 or unphysical
    # values).  Raise the threshold to 5e-4 automatically for float32 runs.
    global CTM_CONV_THR
    if not args.double and CTM_CONV_THR < CTM_CONV_THR_FLOAT32_MIN:
        CTM_CONV_THR = CTM_CONV_THR_FLOAT32_MIN
        print(f"  CTM_CONV_THR auto-raised to 1e-5")
    # Only *enable* the flag from the CLI; never let the argparse default (False)
    # override a True that was already set at module level in core_unrestricted.py.
    # ── Device setup ────────────────────────────────────────────────────────────────
    _use_gpu = getattr(args, 'gpu', USE_GPU)
    if _use_gpu and torch.cuda.is_available():
        _dev = torch.device('cuda')
        print(f"  Using GPU: {_dev} ({torch.cuda.get_device_name(_dev)})")
        print("    GPU detected; setted threads to 1 to avoid oversubscription.")
        torch.set_num_threads(1)
    else:
        if _use_gpu:
            print("  Warning: --gpu requested but torch.cuda.is_available()=False; "
                  "falling back to CPU.")
        _dev = torch.device('cpu')
    set_device(_dev)   # propagates DEVICE into core_unrestricted globals
    # GPU mode: reduce to 1 CPU thread to avoid spin-wait contention.
    # CPU mode: keep _N_PHYSICAL_CORES (already set above).
    if _dev.type == 'cuda':
        torch.set_num_threads(1)
        print("  Threads set to 1 (GPU mode)")
    else:
        print(f"  Threads: {torch.get_num_threads()} (CPU mode)")
    # ── Multi-GPU setup ──────────────────────────────────────────────────────
    global N_GPUS
    _avail_gpus = torch.cuda.device_count() if _dev.type == 'cuda' else 0
    if args.ngpu is not None:
        N_GPUS = max(1, min(args.ngpu, _avail_gpus)) if _avail_gpus > 0 else 1
    else:
        N_GPUS = _avail_gpus if _avail_gpus > 0 else 1
    N_GPUS = min(N_GPUS, 3)   # at most 3 (one per energy function)
    if N_GPUS >= 2:
        print(f"  Multi-GPU: {N_GPUS} GPUs detected — energy functions will run "
              f"concurrently on cuda:0..cuda:{N_GPUS-1}")
    else:
        print(f"  CPU / Single-GPU : energy functions run sequentially "
              f"({'use --ngpu 2+ to enable multi-GPU' if _avail_gpus >= 2 else 'only CPUs / 1 GPU available'})")
        N_GPUS = 1  # ensure scalar is exactly 1 for the branch check
    _core._SVD_CPU_OFFLOAD_THRESHOLD = SVD_CPU_OFFLOAD_THRESHOLD
    _core.set_rsvd_mode(RSVD_MODE,
                        neumann_terms=RSVD_NEUMANN_TERMS,
                        power_iters=RSVD_POWER_ITERS)
    _core.set_ctm_conv_mode(CTM_CONV_MODE,
                            e_threshold=CTM_E_CONV_THRESHOLD,
                            e_proxy_interval=CTM_E_PROXY_INTERVAL)

    # ── parse D_bond list ─────────────────────────────────────────────────────
    D_bond_list: list[int] = [int(x) for x in args.D_bonds.split(',')]
    d_PHYS = args.d_phys
    total_budget = args.hours * 3600.0
    t_global_start = time.perf_counter()

    # ── parse chi_maxes ───────────────────────────────────────────────────────
    if args.chi_maxes:
        cm_vals = [int(x) for x in args.chi_maxes.split(',')]
        if len(cm_vals) != len(D_bond_list):
            raise ValueError(
                f"--chi-maxes has {len(cm_vals)} entries but "
                f"--D-bonds has {len(D_bond_list)}.")
        chi_max_map = {D: c for D, c in zip(D_bond_list, cm_vals)}
    else:
        chi_max_map = {D: DEFAULT_CHI_MAX.get(D, 9999) for D in D_bond_list}

    # ── chi schedules ─────────────────────────────────────────────────────────
    # Use defaults for known D values, otherwise compute geometrically.
    def chi_schedule(D: int, chi_max: int) -> list[int]:
        if chi_max_map.get(D) == DEFAULT_CHI_MAX.get(D) and D in DEFAULT_CHI_SCHEDULES:
            # Use the explicit schedule defined in DEFAULT_CHI_SCHEDULES as-is.
            # Commented-out entries stay commented out — nothing is auto-appended.
            sched = list(DEFAULT_CHI_SCHEDULES[D])
        else:
            # Geometric fallback: GEO_SCHEDULE_STEPS log-uniform values
            # from 1 to chi_max.  Only reached when --chi-maxes is given
            # with a value different from DEFAULT_CHI_MAX[D].
            import math
            chi_min = 1
            if chi_min >= chi_max:
                return [chi_max]
            n = GEO_SCHEDULE_STEPS
            ratio = (chi_max / chi_min) ** (1.0 / (n - 1))
            sched = sorted(set(
                [chi_min]
                + [round(chi_min * ratio**k) for k in range(1, n)]
                + [chi_max]))
            sched = [c for c in sched if 0 < c <= chi_max]
        # Clip to valid range chi ≤ chi_max (catches typos in schedule).
        # NOTE: chi_max itself is NOT auto-appended — the schedule is used
        # exactly as written in DEFAULT_CHI_SCHEDULES.
        sched = [c for c in sched if 0 < c <= chi_max]
        if not sched:
            raise ValueError(f"Empty chi schedule for D={D} after clipping to (0, {chi_max}].")
        return sorted(set(sched))

    schedules = {D: chi_schedule(D, chi_max_map[D]) for D in D_bond_list}

    # ── time budget per D_bond ────────────────────────────────────────────────
    # Use defaults for configured D values; normalise to sum to 1.
    raw_fracs = {D: DEFAULT_D_BUDGET_FRACS.get(D, 1.0) for D in D_bond_list}
    total_frac = sum(raw_fracs.values())
    d_budgets = {D: total_budget * raw_fracs[D] / total_frac
                 for D in D_bond_list}

    # ── output directory ──────────────────────────────────────────────────────
    run_ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    _J2_str = f'J2_{args.J2}'.replace('.', 'p')
    _default_outdir = os.path.join(MY_OUTPUT_OUTERDIR,
                                   f'{ansatz_cfg["yaml_name"]}__{_J2_str}_{run_ts}')
    output_dir = args.output_dir or _default_outdir
    os.makedirs(output_dir, exist_ok=True)

    # ── save hyperparameters to YAML (JSON fallback if PyYAML not installed) ──
    _hp = dict(
        # ── run identity ───────────────────────────────────────────────────
        ansatz             = ansatz_cfg['yaml_name'],
        run_timestamp      = run_ts,
        output_dir         = output_dir,

        # ── reproducibility ────────────────────────────────────────────────
        random_seed_fix    = _seed_enabled,
        random_seed        = args.seed if _seed_enabled else None,

        # ── precision ──────────────────────────────────────────────────────
        use_double         = args.double,
        use_real_tensors   = use_real,
        tensordtype        = str(TENSORDTYPE),
        device             = str(_core.DEVICE),

        # ── time budget ────────────────────────────────────────────────────
        hours              = args.hours,
        D_bond_list        = D_bond_list,
        default_d_budget_fracs = {str(k): v for k, v in DEFAULT_D_BUDGET_FRACS.items()},
        d_budgets_seconds  = {str(k): round(v, 2) for k, v in d_budgets.items()},
        chi_max_map        = {str(k): v for k, v in chi_max_map.items()},
        schedules          = {str(k): v for k, v in schedules.items()},
        geo_schedule_steps = GEO_SCHEDULE_STEPS,

        # ── physical model ─────────────────────────────────────────────────
        J1                 = args.J1,
        J2                 = args.J2,
        d_phys             = d_PHYS,
        n_sites            = N_SITES,

        # ── optimiser ──────────────────────────────────────────────────────
        optimizer          = OPTIMIZER,
        # L-BFGS
        lbfgs_lr           = LBFGS_LR,
        lbfgs_max_iter     = LBFGS_MAX_ITER,
        lbfgs_history      = LBFGS_HISTORY,
        opt_tol_grad               = OPT_TOL_GRAD,
        opt_tol_change             = OPT_TOL_CHANGE,
        opt_conv_threshold         = OPT_CONV_THRESHOLD,
        chi_convergence_threshold  = CHI_CONVERGENCE_THRESHOLD,
        # Adam
        adam_lr            = ADAM_LR,
        adam_betas         = list(ADAM_BETAS),
        adam_eps           = ADAM_EPS,
        adam_weight_decay  = ADAM_WEIGHT_DECAY,
        adam_steps_per_ctm = ADAM_STEPS_PER_CTM,

        # ── SVD / rSVD ─────────────────────────────────────────────────────
        svd_cpu_offload_threshold = SVD_CPU_OFFLOAD_THRESHOLD,
        rsvd_mode          = RSVD_MODE,
        rsvd_neumann_terms = RSVD_NEUMANN_TERMS,
        rsvd_power_iters   = RSVD_POWER_ITERS,

        # ── CTMRG ──────────────────────────────────────────────────────────
        ctm_max_steps          = CTM_MAX_STEPS,
        ctm_conv_thr           = CTM_CONV_THR,
        ctm_conv_mode          = CTM_CONV_MODE,
        ctm_e_conv_threshold   = CTM_E_CONV_THRESHOLD,
        ctm_e_proxy_interval   = CTM_E_PROXY_INTERVAL,
        env_identity_init      = ENV_IDENTITY_INIT,

        # ── tensor init & padding ──────────────────────────────────────────
        init_noise                 = INIT_NOISE,
        pad_noise                  = args.noise,
        rand_init_new_d            = args.rand_init_new_d,
        rand_init_new_chi          = args.rand_init_new_chi,
        mean_field_init            = args.mean_field_init,

        # ── I/O ────────────────────────────────────────────────────────────
        save_every         = SAVE_EVERY,

        # ── CPU threading ──────────────────────────────────────────────────
        n_physical_cores   = _N_PHYSICAL_CORES,
    )
    try:
        import yaml
        with open(os.path.join(output_dir, 'hyperparams.yaml'), 'w') as _fp:
            yaml.dump(_hp, _fp, default_flow_style=False, sort_keys=False)
    except ImportError:
        import json as _json
        with open(os.path.join(output_dir, 'hyperparams.yaml'), 'w') as _fp:
            _json.dump(_hp, _fp, indent=2)

    # ── Hamiltonians (J1-J2 model) ────────────────────────────────────────────
    # Each energy function takes 12 coupling constants (6 nn + 6 nnn) plus the
    # unit spin-spin operator SdotS.  Js is a flat list of 36 = 3 × (6+6):
    #   Js[ 0:12] → energy_fn_1:  Jeb,Jad,Jcf,Jfa,Jde,Jbc (nn)  + Jae,Jec,Jca,Jdb,Jbf,Jfd (nnn)
    #   Js[12:24] → energy_fn_2:  Jaf,Jcb,Jed,Jdc,Jba,Jfe (nn)  + Jca,Jae,Jec,Jbf,Jfd,Jdb (nnn)
    #   Js[24:36] → energy_fn_3:  Jcd,Jef,Jab,Jbe,Jfc,Jda (nn)  + Jec,Jca,Jae,Jfd,Jdb,Jbf (nnn)
    SdotS = build_heisenberg_H(1.0, d_PHYS)         # unit S·S operator (J=1)
    J1, J2 = args.J1, args.J2
    Js = ([J1]*6 + [J2]*6) * 3                       # 36 J-values total

    # ── Banner ────────────────────────────────────────────────────────────────
    print("=" * 76)
    print("  iPEPS sweep  —  AD-CTMRG  —  J1-J2 Heisenberg on 6-site honeycomb")
    print(f"  Ansatz       : {args.ansatz} — {ansatz_cfg['description']}")
    print(f"  J1={args.J1}  J2={args.J2}  d_phys={d_PHYS}  Total budget: {args.hours:.1f} h")
    print(f"  Device       : {_core.DEVICE}")
    print(f"  D_bond sweep : {D_bond_list}")
    for D in D_bond_list:
        bh = d_budgets[D] / 3600
        print(f"    D={D}: chi={schedules[D]}  budget={bh:.2f} h  "
              f"(chi_max={chi_max_map[D]})")
    print(f"  Output dir   : {output_dir}")
    print(f"  Started      : {timestamp()}")
    print("=" * 76)

    # ── State ─────────────────────────────────────────────────────────────────
    all_loss_logs: dict[tuple, list] = {}     # (D, chi) → list of step records
    energy_table: list[dict] = []             # [{D, chi, loss, energy, ...}, ...]
    best_params_by_D: dict[int, tuple | None] = {D: None for D in D_bond_list}
    global_step = 0

    # ── Resume ────────────────────────────────────────────────────────────────
    resume_D, resume_chi = None, None
    if args.resume:
        ckpt = torch.load(args.resume, map_location=_core.DEVICE)
        resume_D   = ckpt.get('D_bond')
        resume_chi = ckpt.get('chi')
        ckpt_step  = ckpt.get('step', 0) + 1
        ckpt_loss  = ckpt.get('loss', float('nan'))
        loaded = tuple(ckpt[k] for k in ansatz_cfg['ckpt_keys'])
        best_params_by_D[resume_D] = _new_tensors_from_data(loaded)
        del ckpt, loaded; gc.collect()
        global_step = ckpt_step
        print(f"  Resumed from {args.resume}  "
              f"(D={resume_D}, chi={resume_chi}, "
              f"loss={ckpt_loss:.6f})\n")

    # ══════════════════════════════════════════════════════════════════════════
    # Outer loop: D_bond
    # ══════════════════════════════════════════════════════════════════════════
    for D_bond in D_bond_list:
        D_sq     = D_bond ** 2
        D_budget = d_budgets[D_bond]
        chis     = schedules[D_bond]
        chi_max  = chi_max_map[D_bond]
        D_start_time = time.perf_counter()

        print(f"\n{'═'*76}")
        print(f"  D_bond = {D_bond}   D²={D_sq}  D⁴={D_bond**4}  "
              f"chi_max={chi_max}  budget={D_budget/3600:.2f} h")
        print(f"  chi schedule: {chis}  ({len(chis)} levels, equal time split)")
        print(f"{'═'*76}")

        _d_restart_count = 0
        while True:  # ── D-restart harness; re-entered if env collapses ──
            # ── Warm-start from D-1 if available ─────────────────────────────────
            prev_D = D_bond_list[D_bond_list.index(D_bond) - 1] \
                     if D_bond_list.index(D_bond) > 0 else None
            if args.rand_init_new_d and prev_D is not None:
                print(f"  [rand-D] random init for D={D_bond} (ignoring D={prev_D} tensors)")
            if (best_params_by_D.get(D_bond) is None
                    and prev_D is not None
                    and best_params_by_D.get(prev_D) is not None
                    and not args.rand_init_new_d):
                print(f"  Warm-starting from D={prev_D} tensors "
                      f"(padding {prev_D}→{D_bond}, noise={args.noise})")
                prev_tensors = best_params_by_D[prev_D]
                sym_fn = ansatz_cfg['symmetrize_fn']
                # For n_params==6 with a batch symmetrize_fn (e.g. 'sym6'), the
                # per-tensor symmetry is applied inside _derive_abcdef at every
                # forward pass — do NOT pass it to pad_tensor (which takes a single
                # tensor and would apply the wrong function).  For n_params==1 the
                # existing per-tensor behaviour is preserved.
                _pad_sym = sym_fn if ansatz_cfg['n_params'] == 1 else None
                padded = tuple(
                    pad_tensor(t, prev_D, D_bond, d_PHYS, args.noise,
                               symmetrize_fn=_pad_sym)
                    for t in prev_tensors)
                best_params_by_D[D_bond] = _new_tensors_from_data(padded)
                del padded
    
            # current best tensors at this D (None = random init at first chi)
            cur_params = best_params_by_D.get(D_bond)
    
            # ── Inner loop: chi ───────────────────────────────────────────────────
            _chi_collapse = False
            for chi_idx, chi in enumerate(chis):
                # Skip (D, chi) pairs that come before the resume point
                if resume_D is not None and resume_chi is not None:
                    if D_bond < resume_D:
                        continue
                    if D_bond == resume_D and chi < resume_chi:
                        continue
                    if D_bond == resume_D and chi == resume_chi:
                        resume_D = None   # resume point reached; continue normally
                        cur_params = best_params_by_D.get(D_bond)
    
                # equal time budget for every chi level within this D
                chi_budget = D_budget / len(chis)
    
                lbfgs_iters = LBFGS_MAX_ITER
    
                print(f"\n  ┌── D={D_bond}  chi={chi}"
                      f"  budget={chi_budget:.0f}s={chi_budget/60:.1f}min"
                      f"  [{timestamp()}]")
                sys.stdout.flush()
    
                best_path   = os.path.join(output_dir,
                                           f"sweep_D{D_bond}_chi{chi}_best.pt")
                latest_path = os.path.join(output_dir,
                                           f"sweep_D{D_bond}_chi{chi}_latest.pt")
                loss_log: list = []
                all_loss_logs[(D_bond, chi)] = loss_log
    
                # ── Chi init: mean-field / random / warm-start ─────────────────
                if args.mean_field_init and D_bond == D_BOND_LIST[0] and chi == DEFAULT_CHI_SCHEDULES[D_BOND_LIST[0]][0]:
                    _init_params = _make_mean_field_params(
                        ansatz_cfg, D_bond, d_PHYS, INIT_NOISE)
                    if chi_idx == 0:
                        print(f"  │  [mean-field] Néel product-state init for chi={chi}")
                    else:
                        print(f"  │  [mean-field] Néel product-state init for chi={chi} "
                              f"(ignoring previous result)")
                elif args.rand_init_new_chi and chi_idx > 0:
                    print(f"  │  [rand-chi] random init for chi={chi} (ignoring previous result)")
                    _init_params = None
                else:
                    _init_params = cur_params
    
                try:
                    best_params_tuple, best_loss, global_step = optimize_at_chi(
                        Js, SdotS, D_bond, chi, d_PHYS,
                        budget_seconds=chi_budget,
                        lbfgs_max_iter=lbfgs_iters,
                        init_params=_init_params,
                        step_offset=global_step,
                        best_path=best_path,
                        latest_path=latest_path,
                        loss_log=loss_log,
                        out_dir=output_dir,
                        ansatz_cfg=ansatz_cfg,
                    )
                except _CollapseRestartD as _exc:
                    _chi_collapse = True
                    break
                # ── Brand-new tensors from raw data; kill the old run entirely ──
                cur_params = _new_tensors_from_data(best_params_tuple)
                del best_params_tuple
                gc.collect()  # free old optimizer state + CTMRG envs + graph
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()  # return freed CUDA blocks to driver
    
    
                # Clean energy evaluation + observables
                print(f"  │  Evaluating energy & observables at (D={D_bond}, chi={chi}) ...")
                energy, correlations, magnetizations, trunc_error = evaluate_observables(
                    list(cur_params), Js, SdotS, chi, D_bond, d_PHYS, ansatz_cfg)
                energy_per_site = energy / N_SITES
                _save_observables_file(
                    os.path.join(output_dir,
                                 f"D_{D_bond}_chi_{chi}"
                                 f"_energy_magnetization_correlation.txt"),
                    D_bond, chi, energy, correlations, magnetizations, trunc_error)
                _print_observables_summary(
                    'OBS', D_bond, chi, energy, correlations, magnetizations, trunc_error)
    
                # Save final checkpoint for this (D, chi)
                save_checkpoint(best_path, cur_params, D_bond, chi,
                                best_loss, energy, global_step, loss_log, ansatz_cfg)
    
                # Log
                record = {
                    'D_bond': D_bond, 'chi': chi,
                    'best_loss': best_loss, 'energy': energy,
                    'energy_per_site': energy_per_site,
                    'steps': len(loss_log),
                    'timestamp': timestamp(),
                }
                energy_table.append(record)
                wall = time.perf_counter() - t_global_start
                print(f"  └── E={energy:+.10f}  E/site={energy_per_site:+.10f}"
                      f"  wall={wall/3600:.2f}h")
    
                # ── Chi-convergence lookahead ─────────────────────────────────────
                # Evaluate energy at chi+D and chi+2D (using current optimised
                # tensors, no further optimisation) to test if chi is converged.
                # BOTH |E(chi)-E(chi+D)| and |E(chi)-E(chi+2D)| must be below
                # CHI_CONVERGENCE_THRESHOLD to declare chi convergence.
                # Each lookahead env is fully freed before the next is built:
                # del + gc.collect() + empty_cache() between the two evals so
                # at no point are two lookahead envs live simultaneously.
                # The finishing chi for D_next schedule filtering is the CURRENT
                # chi (not chi_la), per the protocol.
                if chi_idx < len(chis):
                    # ── Lookahead 1: chi + D ─────────────────────────────────────
                    chi_la_1 = chis[chi_idx] + D_bond
                    print(f"  │  [Lookahead+D] evaluating (D={D_bond}, chi={chi_la_1}) "
                          f"with current tensors ...")
                    with torch.no_grad():
                        energy_la_1, corr_la_1, mag_la_1, trunc_la_1 = evaluate_observables(
                            list(cur_params), Js, SdotS, chi_la_1, D_bond, d_PHYS,
                            ansatz_cfg)
                    _save_observables_file(
                        os.path.join(output_dir,
                                     f"D_{D_bond}_chi_{chi}+D_equals_chi_{chi_la_1}"
                                     f"_energy_magnetization_correlation.txt"),
                        D_bond, chi_la_1, energy_la_1, corr_la_1, mag_la_1, trunc_la_1)
                    _print_observables_summary(
                        'LA+D', D_bond, chi_la_1, energy_la_1, corr_la_1, mag_la_1, trunc_la_1)
                    delta_la_1 = energy - energy_la_1
                    print(f"  │  [Lookahead+D] chi={chi}: E={energy:+.10f} │ "
                          f"chi={chi_la_1}: E={energy_la_1:+.10f} │ "
                          f"ΔE={delta_la_1:.3e}  "
                          f"[thr={CHI_CONVERGENCE_THRESHOLD:.1e}]")
                    # ── Release chi+D lookahead env before building chi+2D env ───
                    # Identical pattern to the original single-lookahead cleanup:
                    # freed env tensors stay in PyTorch CUDA cache until empty_cache.
                    # Must call this BEFORE the chi+2D eval to avoid two large envs
                    # (chi+D and chi+2D) being live simultaneously.
                    del energy_la_1, corr_la_1, mag_la_1, trunc_la_1
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
    
                    # ── Lookahead 2: chi + 2D ────────────────────────────────────
                    chi_la_2 = chis[chi_idx] + D_bond * 2
                    print(f"  │  [Lookahead+2D] evaluating (D={D_bond}, chi={chi_la_2}) "
                          f"with current tensors ...")
                    with torch.no_grad():
                        energy_la_2, corr_la_2, mag_la_2, trunc_la_2 = evaluate_observables(
                            list(cur_params), Js, SdotS, chi_la_2, D_bond, d_PHYS,
                            ansatz_cfg)
                    _save_observables_file(
                        os.path.join(output_dir,
                                     f"D_{D_bond}_chi_{chi}+2D_equals_chi_{chi_la_2}"
                                     f"_energy_magnetization_correlation.txt"),
                        D_bond, chi_la_2, energy_la_2, corr_la_2, mag_la_2, trunc_la_2)
                    _print_observables_summary(
                        'LA+2D', D_bond, chi_la_2, energy_la_2, corr_la_2, mag_la_2, trunc_la_2)
                    delta_la_2 = energy - energy_la_2
                    print(f"  │  [Lookahead+2D] chi={chi}: E={energy:+.10f} │ "
                          f"chi={chi_la_2}: E={energy_la_2:+.10f} │ "
                          f"ΔE={delta_la_2:.3e}  "
                          f"[thr={CHI_CONVERGENCE_THRESHOLD:.1e}]")
                    # ── Release chi+2D lookahead env before next chi optimization ─
                    # The lookahead evaluated at chi_la_2 > chi, allocating larger
                    # env tensors.  Freed here so the next chi CTMRG gets clean
                    # contiguous blocks — prevents OOM "605 MiB free but 1.28 GiB
                    # requested".
                    del energy_la_2, corr_la_2, mag_la_2, trunc_la_2
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if abs(delta_la_1) < CHI_CONVERGENCE_THRESHOLD and \
                            abs(delta_la_2) < CHI_CONVERGENCE_THRESHOLD and \
                                abs(delta_la_1 + delta_la_2) < CHI_CONVERGENCE_THRESHOLD:
                        print(f"  │  [CHI-CONV] ΔE(chi+D)={delta_la_1:.3e} and "
                              f"ΔE(chi+2D)={delta_la_2:.3e} tri-th "
                              f"within {CHI_CONVERGENCE_THRESHOLD:.1e} → chi converged at "
                              f"chi={chi}; skipping remaining chi levels for D={D_bond}.")
                        break   # exit chi loop early; cur_params stays at chi

            if _chi_collapse:
                _d_restart_count += 1
                print(f"\n  [D-RESTART #{_d_restart_count}] env collapsed "
                      f"(D={D_bond}, chi={chi}); clearing state and replaying "
                      f"chi schedule from chi={chis[0]}")
                best_params_by_D[D_bond] = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue  # re-enter while True; warm-start re-runs above
            break  # chi schedule completed normally

        # Store best tensors at this D for warm-starting next D
        # (brand-new objects — old D's entire graph is released)
        best_params_by_D[D_bond] = (
            _new_tensors_from_data(cur_params) if cur_params is not None
            else None)
        del cur_params
        # Free tensors from previous D — they were only needed for warm-start
        # into this D, which has already happened at the top of the D-loop.
        if prev_D is not None and prev_D in best_params_by_D:
            best_params_by_D[prev_D] = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        D_elapsed = time.perf_counter() - D_start_time
        print(f"\n  D={D_bond} complete in {D_elapsed/3600:.2f} h")
        global_step += 1   # gap marker

    # ══════════════════════════════════════════════════════════════════════════
    # Results summary
    # ══════════════════════════════════════════════════════════════════════════
    total_elapsed = time.perf_counter() - t_global_start

    print("\n" + "=" * 76)
    print("  RESULTS SUMMARY")
    print("=" * 76)
    print(f"  {'D':>2}  {'chi':>5}  {'steps':>6}  "
          f"{'E_total':>18}  {'E/site':>14}")
    print(f"  {'─'*2}  {'─'*5}  {'─'*6}  {'─'*18}  {'─'*14}")
    for row in energy_table:
        print(f"  {row['D_bond']:2d}  {row['chi']:5d}  {row['steps']:6d}  "
              f"{row['energy']:+18.10f}  {row['energy_per_site']:+14.10f}")
    print()
    best_overall = min(energy_table, key=lambda r: r['energy_per_site'])
    print(f"  Best E/site = {best_overall['energy_per_site']:+.10f}  "
          f"(D={best_overall['D_bond']}, chi={best_overall['chi']})")
    print(f"  Total wall time: {total_elapsed/3600:.2f} h ({total_elapsed:.0f} s)")
    print(f"  Finished: {timestamp()}")
    print("=" * 76)

    # ── JSON ──────────────────────────────────────────────────────────────────
    results_path = os.path.join(output_dir, "sweep_results.json")
    json_out = {
        'D_bond_list': D_bond_list,
        'chi_max_map': {str(k): v for k, v in chi_max_map.items()},
        'schedules':   {str(k): v for k, v in schedules.items()},
        'J1': args.J1, 'J2': args.J2, 'd_phys': d_PHYS,
        'hours_budget': args.hours,
        'total_elapsed_h': round(total_elapsed / 3600, 3),
        'total_steps': global_step,
        'energy_table': energy_table,
        'timestamp': timestamp(),
    }
    with open(results_path, 'w') as fp:
        json.dump(json_out, fp, indent=2)
    print(f"\n  Results JSON → {results_path}")

    # ── Plot 1: E/site vs chi for each D ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    for D in D_bond_list:
        rows = [r for r in energy_table if r['D_bond'] == D]
        if not rows:
            continue
        x = [r['chi'] for r in rows]
        y = [r['energy_per_site'] for r in rows]
        ax.plot(x, y, 'o-', label=f'D={D}', markerfacecolor='white',
                markersize=7)
    ax.set_xlabel(r'Environment bond dimension $\chi$', fontsize=12)
    ax.set_ylabel(r'Energy per site $E/N_{\mathrm{sites}}$', fontsize=12)
    ax.set_title('iPEPS ground-state energy — J1-J2 Heisenberg (honeycomb)', fontsize=12)
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'sweep_energy_vs_chi.pdf'))
    plt.close(fig)

    # ── Plot 2: E/site vs D at largest chi each D ─────────────────────────────
    D_vals, E_best = [], []
    for D in D_bond_list:
        rows = [r for r in energy_table if r['D_bond'] == D]
        if rows:
            best_row = min(rows, key=lambda r: r['energy_per_site'])
            D_vals.append(D)
            E_best.append(best_row['energy_per_site'])
    if len(D_vals) > 1:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.plot(D_vals, E_best, 's-', color='tab:red', markersize=9,
                 markerfacecolor='white')
        ax2.set_xlabel(r'Bond dimension $D$', fontsize=12)
        ax2.set_ylabel(r'Best $E/\text{site}$', fontsize=12)
        ax2.set_title('iPEPS convergence in D')
        ax2.legend(); ax2.grid(True, alpha=0.3)
        fig2.tight_layout()
        fig2.savefig(os.path.join(output_dir, 'sweep_energy_vs_D.pdf'))
        plt.close(fig2)

    # ── Plot 3: loss vs step for each D (separate panels) ────────────────────
    for D in D_bond_list:
        d_logs = {chi: all_loss_logs[(D, chi)]
                  for chi in schedules[D]
                  if (D, chi) in all_loss_logs and all_loss_logs[(D, chi)]}
        if not d_logs:
            continue
        fig3, ax3 = plt.subplots(figsize=(11, 4))
        for chi, log in d_logs.items():
            steps_  = [r['step'] for r in log]
            losses_ = [r['loss'] for r in log]
            ax3.plot(steps_, losses_, '-', lw=0.9, alpha=0.8, label=f'χ={chi}')
        ax3.set_xlabel('Outer step'); ax3.set_ylabel('Loss (total energy)')
        ax3.set_title(f'Loss curve  D={D}')
        ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3)
        fig3.tight_layout()
        fig3.savefig(os.path.join(output_dir, f'sweep_loss_D{D}.pdf'))
        plt.close(fig3)

    print(f"  Plots → {output_dir}/sweep_*.pdf")


if __name__ == '__main__':

    main()

