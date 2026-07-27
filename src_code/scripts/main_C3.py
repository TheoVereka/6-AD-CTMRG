
#!/usr/bin/env python3





MY_OUTPUT_OUTERDIR = '/scratch/izar/chye/0727core'
USE_GPU = True
_N_PHYSICAL_CORES = 25

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


# ── Sweep control ─────────────────────────────────────────────────────────────

D_BOND_LIST   = [  2,  3,  4,  5,  6,  7,  8,  9, 10]
CHI_MIN_LIST  = [ 16, 24, 36, 50, 72, 91,104, 108,140]
CHI_MAX_LIST  = [ 20, 30, 44, 60, 84, 105,112,108,160]
CHI_STEP_LIST = [  4,  6,  8, 10, 12, 14, 16, 18, 20]
#   Default chi schedule parameters (one per D in D_BOND_LIST).
#   For each D_bond, the chi schedule is generated as:
#     range(chi_min, chi_max+1, chi_step)  (Python range semantics)
#   Must satisfy: len(CHI_MIN_LIST) == len(CHI_MAX_LIST) == len(CHI_STEP_LIST) == len(D_BOND_LIST)
#   Override at runtime: --chi-min 40 55 72  --chi-max 80 100 120  --chi-step 10 10 10

# ══════════════════════════════════════════════════════════════════════════════
# Time Budget
# ══════════════════════════════════════════════════════════════════════════════

TOTAL_BUDGET_HOURS = 99999

# ── GPU/CPU intent ────────────────────────────────────────────────────────────
# Duplicated below in the TUNABLE PARAMETERS section with full comments.


########################
# ── Physical model ── # ────────────────────────────────────────────────────────────
########################

J1_COUPLING = 1.0
#   Nearest-neighbour (nn) Heisenberg exchange coupling constant.
#   J1 > 0 = antiferromagnetic (AFM).
#   The nn Hamiltonian is  H_nn = J1 Σ_{<i,j>} S_i · S_j
#   summed over all 9 nearest-neighbour pairs in the 6-site honeycomb unit cell.

J2_COUPLING = 0.26
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
  python scripts/main_sweep.py                       # all defaults
  python scripts/main_sweep.py --hours 11
  python scripts/main_sweep.py --Ds 4 5 6 --chi-min 40 50 60 --chi-max 80 100 120 --chi-step 10 10 10
  python scripts/main_sweep.py --resume log/sweep_D3_chi81_best.pt --Ds 3 4

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

# if MY_OUTPUT_OUTERDIR contain izar then automatically set USE_GPU = True
if "izar" in MY_OUTPUT_OUTERDIR:
    USE_GPU = True

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


# ── Zero-overhead dual-stream logger ──────────────────────────────────────────
# _TeeStream wraps stdout and mirrors every write to a file object.
# The file is opened (line-buffered) once output_dir is known, via tee_stdout().
# All print() calls — including those from core_C3.py and PyTorch — are captured
# because they ultimately call sys.stdout.write().  No threads, no queues, no
# extra formatting overhead.

class _TeeStream:
    """Proxy for sys.stdout that mirrors every write to *logfile*."""
    def __init__(self, original, logfile):
        self._orig   = original
        self._logfile = logfile
    def write(self, s):
        self._orig.write(s)
        self._logfile.write(s)
    def flush(self):
        self._orig.flush()
        self._logfile.flush()
    def fileno(self):         # needed by some libraries that call os.write
        return self._orig.fileno()
    def isatty(self):
        return self._orig.isatty()
    # Forward every other attribute transparently
    def __getattr__(self, name):
        return getattr(self._orig, name)

# Installed in main() once output_dir is known (before banner is printed).
_tee_installed: bool = False
def _install_tee(output_dir: str) -> None:
    global _tee_installed
    if _tee_installed:
        return
    import sys as _sys
    _logfile = open(os.path.join(output_dir, 'run.log'), 'a', buffering=1,
                    encoding='utf-8', errors='replace')
    _sys.stdout = _TeeStream(_sys.stdout, _logfile)
    _tee_installed = True


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

import core_C3 as _core
from core_C3 import (
    normalize_tensor,
    normalize_single_layer_tensor_for_double_layer,
    initialize_abcdef,
    abcdef_to_ABCDEF,
    CTMRG_from_init_to_stop,
    build_heisenberg_H,
    energy_expectation_nearest_neighbor_3ebadcf_bonds,
    energy_expectation_nearest_neighbor_3afcbed_bonds,
    energy_expectation_nearest_neighbor_other_3_bonds,
    set_dtype,
    set_device,
    # ── Néel-symmetrized ansatz ──────────────────────────────────────────
    symmetrize_virtual_legs,
    neel_abcdef_from_a,
    initialize_neel,
    # ── Néel free-param morphism ─────────────────────────────────────────
    neel_param_to_a,
    neel_a_to_free_param,
    initialize_neel_free_param,
    pad_neel_free_param,
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
    # ── sym6 free-param morphism ─────────────────────────────────────────
    sym6_params_to_abcdef,
    sym6_abcdef_to_free_params,
    initialize_sym6_free_params,
    pad_sym6_free_params,
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


# SVD_CPU_OFFLOAD_THRESHOLD is always 0 (GPU always handles SVD; never
# offload to CPU for cluster GPUs).  Hard-coded in core.py; not a tunable.

RSVD_MODE = 'augmented'
#   rSVD backward mode.  Controls how the truncated-SVD 5th-term correction
#   (arXiv:2311.11894v3) is computed in the backward pass.
#
#   'full_svd'  — keep full thin SVD (safe reference, no rSVD, ~O(N·min(m,n))).
#   'neumann'   — Neumann series for 5th term: O(2·L·mnk) per call, avoids
#                 the O(N³) eigh.  Exact when discarded SVs ≈ 0.
#   'augmented' — save k+k_extra rSVD triples; zero-padding captures 5th term
#                 implicitly via F,G cross-coupling.
#   'none'      — skip 5th term entirely (wrong for projector-type losses).
#
#   D=3, chi=12, three-step differentiable CTMRG regression (2026-07-27):
#     augmented gradient error vs full SVD = 4.56e-4
#     Neumann L=2 gradient error vs full SVD = 6.995e-2
#   Therefore augmented is the validated production mode for this code path.
#   Neumann remains available for diagnostics but is not the default.

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

LBFGS_LR = 1e0
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

OPT_TOL_GRAD = 0.0 #1e-8
#   L-BFGS inner convergence criterion on the infinity-norm of the gradient:
#   the sub-iteration loop exits early if  ||∇loss||_∞ < OPT_TOL_GRAD.
#   This is an inner stopping rule inside a single optimizer.step() call.

OPT_TOL_CHANGE = 2e-8
#   L-BFGS inner convergence criterion on consecutive loss change:
#   sub-iteration exits if  |L_{k+1} – L_k| < OPT_TOL_CHANGE.
#   Set tighter than OPT_TOL_GRAD to catch near-flat regions.

OPT_CONV_THRESHOLD = 3e-8
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

CHI_CONVERGENCE_THRESHOLD = 3e-5
#   Chi-level early-exit criterion (lookahead comparison).
#   After optimisation at (D, chi) completes, a lookahead evaluation is run
#   at chi_la = chi + D_bond. If |E(chi) − E(chi_la)| < CHI_CONVERGENCE_THRESHOLD,
#   the chi series for this D is considered converged and we skip remaining
#   chi levels, immediately proceeding to the next D.
#   Set to 0 to disable (always run full chi schedule).

# ── Optimizer choice ──────────────────────────────────────────────────────────

OPTIMIZER = 'lbfgs'
#   'lbfgs' : L-BFGS with strong-Wolfe line search (default).
#             Converges fast on smooth landscapes; may oscillate on noisy ones.
#   'adam'  : Adam (adaptive moment estimation).
#             More robust on noisy/non-convex landscapes.  Each outer step runs
#             ADAM_STEPS_PER_CTM gradient updates with a fixed environment.
#   Override at runtime:  --optimizer {lbfgs,adam}

# ── Adam hyperparameters (used only when OPTIMIZER='adam') ────────────────────

ADAM_LR = 7e-3
#   Base Adam learning rate.  Per-ansatz effective LR is derived below
#   from the coupling factor: how many site tensors are controlled by one
#   parameter block.  See optimize_at_chi for the per-ansatz divisors.
ADAM_BETAS = (0.85, 0.96)
#   (β₁, β₂): exponential decay rates for 1st/2nd moment estimates.
#   Derived from first principles for the Adam→L-BFGS warmup scheme:
#
#   β₁ = 0.85:  look-back window τ₁ = 1/(1−β₁) = 6.67 steps.
#     Momentum decays within ~6.67 steps once Adam decelerates near the basin,
#     so |Δloss| reported to the patience counter reflects the CURRENT
#     gradient, not stale momentum from the rugged mid-landscape phase.
#     Higher β₁ (e.g. 0.99, τ₁=100) delays the switch because inflated
#     momentum keeps |Δloss| > threshold even when the true gradient is small.
#     β₁=0.9 still provides enough inertia for barrier crossing (gradient
#     is consistently downhill over many steps during directed descent).
#
#   β₂ = 0.99:  RMS look-back window τ₂ = 1/(1−β₂) = 100 steps.
#     With β₂=0.999 (τ₂=1000), v̂_t carries the large-gradient memory from
#     the initial rugged phase for the entire Adam run.  This means the
#     effective step α/√v̂ never shrinks even when the true gradient is small
#     near the basin → |Δloss| stays large → patience counter never fires →
#     the Adam→L-BFGS switch is indefinitely delayed.
#     With β₂=0.95 (τ₂=20): once Adam decelerates, v̂ adapts in 20 steps,
#     the effective step genuinely shrinks, |Δloss| falls below threshold,
#     and the patience counter fires correctly.  Constraint: τ₂ < T_adam
#     (estimated 300–1000 steps from logs), so β₂ < 0.997 — 0.99 with margin.
ADAM_EPS = 1e-8
#   Denominator epsilon for numerical stability (increased from 1e-9 to dampen
#   adaptive scaling in small-parameter regimes).
ADAM_WEIGHT_DECAY = 0.0
#   L2 regularisation strength.  0.0 = no regularisation (recommended).
ADAM_GRAD_CLIP = None
#   Max L2 norm of the gradient before Adam's step; None to disable.
#   Prevents overshooting steps when all site tensors move together (e.g.
#   neel ansatz: one h-step changes all 6 sites simultaneously, making
#   the effective learning rate ~6× larger than for 6-independent-tensor
#   ansatze like sym6).  1.0 is a standard conservative clip; raise to
#   5.0 or disable for faster descent when sym6/c6ypi are used.
# Adam always makes exactly 1 gradient step per CTMRG environment refresh;
# ADAM_STEPS_PER_CTM is removed — it was always 1 and configuring otherwise
# would bias the alternating-optimisation scheme.

# ── Adam → L-BFGS warmup protocol ────────────────────────────────────────────

USE_ADAM_WARMUP_THEN_LBFGS = True
#   True  → start each chi-level with Adam (robust exploration of the
#            non-convex, frustrated landscape), then switch to a fresh
#            L-BFGS instance once the flat-plateau window fires.
#            Rationale: Adam handles the rugged early-phase landscape (where
#            L-BFGS's quadratic model is inaccurate and line-search failures
#            are common) and converges to the basin quickly.  Once the loss
#            is nearly flat, L-BFGS's superlinear convergence polishes the
#            solution far more efficiently than Adam's linear rate.
#            Recommended for frustrated systems (J2 > 0, near critical
#            points).  Uses Adam hyperparams for the warmup phase and
#            LBFGS_* hyperparams for the finishing phase.
#   False → use OPTIMIZER ('lbfgs' or 'adam') for the full optimization.
#   Overrideable at runtime: --adam-warmup-lbfgs CLI flag.

ADAM_FLAT_PATIENCE = 20
#   Flat-landscape escape: if the Adam loss window of this many steps has
#   max(window) - min(window) < 2e-3 (hardcoded spread threshold), switch to
#   L-BFGS.  Switch is suppressed if EITHER:
#     (A) same-sign:  the last 17 first-diffs d_i are all strictly the same
#         sign (all <0 or all >0) — Adam still descending or ascending.
#     (B) monotone:   the second-diffs d_i - d_{i+1} are all strictly the
#         same sign — first-diffs themselves accelerating/decelerating
#         monotonically (even if sign alternates).
#   If the first L-BFGS run from a false-early fire converges quickly, the
#   code cycles back to Adam automatically, so early fires are self-healing.

# ── CTMRG algorithm ──────────────────────────────────────────────────────────
#
#   C3-CTMRG builds 3 corner matrices and 6 transfer tensors that represent
#   the three inequivalent environments of the
#   6-site honeycomb unit cell embedded in the infinite lattice.  Each
#   CTMRG "step" grows the environment by one unit-cell layer and then
#   compresses via SVD truncation to keep the environment bond dim = chi.







ENV_IDENTITY_INIT = True






CTM_MAX_STEPS = 50
#   Hard cap on CTMRG iterations per environment convergence call.
#   With the singular-value convergence criterion and CTM_CONV_THR=1e-7,
#   convergence occurs in 4–40 steps for typical tensors (single-tensor
#   ansatz ~4 steps, 6-tensor ~40 steps).  90 is a safe upper bound.

CTM_CONV_THR = 1e-7

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
#     Converge when max|ΔSV| < CTM_CONV_THR across the 3 C3 corner-product spectra.
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

CTM_E_CONV_THRESHOLD = 2e-8
#   Energy-proxy convergence threshold for 'Edifference' and 'both' modes.
#   Applied to |E_proxy(iter N) − E_proxy(iter N-1)| where E_proxy is the
#   EB bond energy from env1 (unit SdotS, no J).
#   Typical converged drift at D=8, chi=80: ~1e-10 to 1e-11.
#   1e-8 is a safe target that fires well before oscillations dominate.
#   Recorded in hyperparams.yaml as ctm_e_conv_threshold.

# CTM_E_PROXY_INTERVAL is always 1 — the energy proxy is checked every CTMRG
# step for the tightest convergence tracking.  Hard-coded; not a tunable.


SAVE_EVERY = 1
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

INIT_NOISE = 1e-2
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
        new = torch.empty(t.shape, dtype=t.dtype, device=t.device)
        new.copy_(t.detach())
        out.append(new)
    return tuple(out)


# ── Ansatz helpers ────────────────────────────────────────────────────────────

def _derive_abcdef(params_list: list, cfg: dict, D_bond: int = None) -> tuple:
    """Derive the 6 single-layer tensors from *params_list* using *cfg*.

    For ansatze with a free-param morphism (cfg['free_morphism_fn'] is not None):
      n_params==6 (sym6): free_morphism_fn(*params_list) → (a,b,c,d,e,f) directly.
      n_params==1 (neel): free_morphism_fn(h, D_bond) → a_sym, then derive_fn(a_sym)
        gives all 6 tensors.  D_bond is required for the neel morphism.

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
    free_morph = cfg.get('free_morphism_fn')
    if free_morph is not None:
        if cfg['n_params'] == 6:
            # sym6: (h_a..h_f) → (a,b,c,d,e,f) via gather morphism
            return free_morph(*params_list)
        else:
            # neel: h → a_sym (needs D) → derive_fn → (a..f)
            a_sym = free_morph(params_list[0], D_bond)
            return cfg['derive_fn'](a_sym)
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
    # Free-param reparametrization: optimize h of shape (D*(D+1)*(D+2)//6, d)
    # instead of a full (D,D,D,d) tensor subject to S₃ constraints.  The
    # morphism  a[i,j,k,s] = h[tri3[i,j,k], s]  is a pure gather (exact,
    # differentiable, no gradient waste).  Old 'neel' entry kept as 'neel_legacy'.
    'neel': {
        'n_params':        1,
        'free_morphism_fn': neel_param_to_a,   # h → a_sym (needs D_bond arg)
        'symmetrize_fn':   None,               # not used: morphism is exact
        'derive_fn':       neel_abcdef_from_a,
        'init_fn':         initialize_neel_free_param,
        'pad_fn':          lambda old, od, nd, d, n: (pad_neel_free_param(old[0], od, nd, d, n),),
        'ckpt_keys':       ['h_neel'],
        'yaml_name':       'neel_free_param',
        'description':     'Néel ansatz with unconstrained S₃ free-param reparametrization',
    },
    # Legacy neel: optimize the full (D,D,D,d) tensor with S₃ projection.
    # Use this only if you need backward compatibility with old checkpoints.
    'neel_legacy': {
        'n_params':    1,
        'free_morphism_fn': None,
        'symmetrize_fn': symmetrize_virtual_legs,
        'derive_fn':   neel_abcdef_from_a,
        'init_fn':     initialize_neel,
        'pad_fn':      None,
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
    # Free-param reparametrization: optimize 6 reduced tensors of shape
    # D²(D+1)/2·d (true dimension of each site's symmetric subspace) instead
    # of 6 full D³·d tensors subject to per-site mirror constraints.  The
    # morphism is a pure gather (no projection gradient waste).
    'sym6': {
        'n_params':        6,
        'free_morphism_fn': sym6_params_to_abcdef,  # (h_a..h_f) → (a..f)
        'symmetrize_fn':   None,                    # not used: morphism is exact
        'derive_fn':       None,
        'init_fn':         initialize_sym6_free_params,  # returns 6-tuple
        'pad_fn':          pad_sym6_free_params,
        'ckpt_keys':       ['h_a', 'h_b', 'h_c', 'h_d', 'h_e', 'h_f'],
        'yaml_name':       'sym6_free_param',
        'description':     'Six-tensor ansatz with local mirror symmetry, '
                           'unconstrained free-param reparametrization '
                           '(a/d: leg1↔leg2, b/e: leg0↔leg1, c/f: leg0↔leg2)',
    },
    # Legacy sym6: optimize full D³d tensors with per-step symmetrize projection.
    'sym6_legacy': {
        'n_params':    6,
        'free_morphism_fn': None,
        'symmetrize_fn': symmetrize_six_local_reflections,
        'derive_fn':   None,
        'init_fn':     None,
        'pad_fn':      None,
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

C3_COMPATIBLE_ANSATZES = ('neel', 'neel_legacy', 'c6ypi', 'c3vypi', 'twoc3')


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

    n          = ansatz_cfg['n_params']
    free_morph = ansatz_cfg.get('free_morphism_fn')

    if free_morph is not None and n == 1:
        # neel free-param: build full mean-field symmetric tensor, extract h
        a_full = _up()
        a_sym  = symmetrize_virtual_legs(a_full)
        return (neel_a_to_free_param(a_sym),)

    if free_morph is not None and n == 6:
        # sym6 free-param: build full ABABAB mean-field tensors, project, extract
        a6_full = (_up(), _down(), _up(), _down(), _up(), _down())
        a6_sym  = symmetrize_six_local_reflections(*a6_full)
        return sym6_abcdef_to_free_params(*a6_sym)

    if n == 1:
        # Single raw tensor; derive_fn maps it to both sublattices.
        t = _up()
        # For ansatze with a symmetrize constraint (e.g. neel_legacy),
        # project the initial tensor onto the symmetric subspace and normalise
        # to the unit sphere.  Without this, the parameter starts outside the
        # constraint manifold and the gradient hook projects relative to the
        # wrong base-point for the first step.
        sym_fn = ansatz_cfg.get('symmetrize_fn')
        if sym_fn is not None:
            t = sym_fn(t)
            _tn = t.norm()
            if _tn > 1e-30:
                t = t / _tn
        return (t,)
    elif n == 2:
        # twoc3: first tensor → A-sublattice (up), second → B-sublattice (down).
        return (_up(), _down())
    else:
        # n == 6: unrestricted / sym6_legacy — explicit ABABAB assignment.
        return (_up(), _down(), _up(), _down(), _up(), _down())


def _observables_from_rhos(rho_data, Js, SdotS, d_PHYS):
    """Compute energy, correlations, and magnetizations from cached density matrices.

    rho_data: tuple of ((_r1, _m1), (_r2, _m2), (_r3, _m3)) as stored by
              _last_rhos_cache inside _loss_with_differentiable_ctmrg.
      _rN: list of 12 rho tensors [NN×6, NNN×6] for env N.
      _mN: dict mapping site label → (rho, site_idx) for magnetization.

    Returns (energy, correlations, magnetizations) — all floats/lists.
    No trunc_error (not tracked on the optimization path).
    """
    if rho_data is None or any(x is None for x in rho_data):
        nan36 = [float('nan')] * 36
        nan54 = [float('nan')] * 54
        return float('nan'), nan36, nan54

    # ── Spin operators (same logic as evaluate_observables) ──────────────
    spin = (d_PHYS - 1) / 2.0
    Splus  = torch.zeros(d_PHYS, d_PHYS, dtype=SdotS.dtype, device=SdotS.device)
    Sminus = torch.zeros(d_PHYS, d_PHYS, dtype=SdotS.dtype, device=SdotS.device)
    Sz_op  = torch.zeros(d_PHYS, d_PHYS, dtype=SdotS.dtype, device=SdotS.device)
    for _i in range(d_PHYS):
        _m = spin - _i
        Sz_op[_i, _i] = _m
        if _i < d_PHYS - 1:
            _cs = (spin * (spin + 1) - _m * (_m - 1)) ** 0.5
            Splus[_i, _i + 1] = _cs
            Sminus[_i + 1, _i] = _cs
    Sx_op  = (Splus + Sminus) / 2.0
    iSy_op = (Splus - Sminus) / 2.0  # real antisymmetric = i·Sy_phys

    def _corr(rho_4d):
        return torch.real(oe.contract("ikjl,ijkl->", rho_4d, SdotS,
                                      backend="torch")).item()

    def _mag(rho_4d, site_idx):
        if site_idx == 0:
            rho_1 = oe.contract("ikjk->ij", rho_4d, backend="torch")
        else:
            rho_1 = oe.contract("ikil->kl", rho_4d, backend="torch")
        tr = torch.real(torch.trace(rho_1)).clamp(min=1e-30)
        rho_1 = rho_1 / tr
        mx = torch.real(oe.contract("ij,ji->", rho_1, Sx_op, backend="torch")).item()
        if rho_1.is_complex():
            my = torch.imag(oe.contract("ij,ji->", rho_1,
                                        iSy_op.to(rho_1.dtype),
                                        backend="torch")).item()
        else:
            my = torch.real(oe.contract("ij,ji->", rho_1, iSy_op,
                                        backend="torch")).item()
        mz = torch.real(oe.contract("ij,ji->", rho_1, Sz_op, backend="torch")).item()
        return (mx, my, mz)

    correlations  = []
    magnetizations = []
    for (rhos, mag_info) in rho_data:
        correlations += [_corr(r) for r in rhos]
        for s in ['A', 'B', 'C', 'D', 'E', 'F']:
            rho, idx = mag_info[s]
            magnetizations.extend(_mag(rho, idx))

    # Energy from correlations (same convention as evaluate_observables)
    energy = 0.0
    for _blk in range(3):
        _b = _blk * 12
        energy += 0.5 * sum(Js[_b + _i] * correlations[_b + _i] for _i in range(6))
        energy +=       sum(Js[_b + 6 + _i] * correlations[_b + 6 + _i] for _i in range(6))

    return energy, correlations, magnetizations


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
        skip_adam_warmup: bool = False,
        first_chi_of_D: bool = False,
) -> tuple:
    """
    Outer L-BFGS loop at fixed (D_bond, chi) until budget_seconds elapsed
    or opt_conv_threshold hit.

    Returns (best_params_tuple, best_loss, steps_done, cached_rhos, switched_to_lbfgs).
    best_params_tuple contains 1 tensor for single-tensor ansätze (neel/c6ypi/c3vypi),
    2 tensors for two-tensor ansätze (twoc3),
    or 6 tensors for 6-tensor ansätze (unrestricted/sym6), matching ansatz_cfg['n_params'].
    switched_to_lbfgs is True if the optimizer switched from Adam to LBFGS during this run.
    """
    if loss_log is None:
        loss_log = []
    if ansatz_cfg is None:
        ansatz_cfg = ANSATZ_REGISTRY['c3vypi']
    n_params = ansatz_cfg['n_params']
    D_sq = D_bond ** 2
    t_start = time.perf_counter()

    # initialise tensors — ALWAYS brand-new objects, no shared lineage
    if init_params is not None:
        params = list(_new_tensors_from_data(init_params))
    elif n_params == 6:
        if ansatz_cfg['init_fn'] is not None:
            # sym6 free-param: init_fn returns a 6-tuple of reduced tensors
            params = list(ansatz_cfg['init_fn'](D_bond, d_PHYS, INIT_NOISE))
        else:
            params = list(initialize_abcdef('random', D_bond, d_PHYS, INIT_NOISE))
    elif n_params == 1:
        params = [ansatz_cfg['init_fn'](D_bond, d_PHYS, INIT_NOISE)]
    else:
        # n_params >= 2: init_fn returns a tuple of tensors
        params = list(ansatz_cfg['init_fn'](D_bond, d_PHYS, INIT_NOISE))
    for t in params:
        t.requires_grad_(True)

    if ansatz_cfg == ANSATZ_REGISTRY['neel'] or ansatz_cfg == ANSATZ_REGISTRY['neel_legacy']:
        effective_ADAM_LR = ADAM_LR/6
    elif ansatz_cfg == ANSATZ_REGISTRY['twoc3']:
        effective_ADAM_LR = ADAM_LR/3
    elif ansatz_cfg == ANSATZ_REGISTRY['c3vypi'] or ansatz_cfg == ANSATZ_REGISTRY['c6ypi']:
        effective_ADAM_LR = ADAM_LR/5
    elif ansatz_cfg == ANSATZ_REGISTRY['sym6']:
        effective_ADAM_LR = ADAM_LR/2
    else:
        effective_ADAM_LR = ADAM_LR


    best_loss     = float('inf')
    best_params   = [t.detach().clone() for t in params]
    prev_loss     = None
    loss_history  = collections.deque(maxlen=12)  # cycle detection up to length 12
    step          = step_offset

    # ── effective optimizer — starts as 'adam' in warmup mode ─────────────────
    # USE_ADAM_WARMUP_THEN_LBFGS overrides OPTIMIZER for the exploration phase;
    # the variable is updated to 'lbfgs' when the switch fires.
    # skip_adam_warmup=True (inherited warm-start from previous chi of same D):
    # tensors are already near the local minimum, so bypass Adam entirely and
    # start directly with L-BFGS (convergence guard still active for first 3 steps).
    _effective_optimizer: str = ('adam' if (USE_ADAM_WARMUP_THEN_LBFGS
                                            and not skip_adam_warmup)
                                  else OPTIMIZER)
    _adam_steps_taken: int = 0  # counts Adam outer steps (for early-switch)
    _switched_to_lbfgs: bool   = False  # True after Adam→L-BFGS transition
    _final_lbfgs_run:  bool   = False  # True on the terminal LBFGS (early-switch)
    # Sliding window of the last ADAM_FLAT_PATIENCE Adam loss values.
    # Used to detect a flat plateau independently of the step-by-step
    # deceleration signal.  Reset on every Adam→LBFGS or LBFGS→Adam switch.
    _adam_loss_window: collections.deque = collections.deque(maxlen=ADAM_FLAT_PATIENCE)
    # _final_lbfgs_run is set when the EARLY NEAR-OPTIMAL SWITCH fires (Adam
    # step ≤3 with Δ > -1e-6).  When that LBFGS run converges the loop exits.
    # All other LBFGS runs cycle back to Adam on convergence.

    # ── pre-create Adam optimizer (state persists across outer steps) ─────────
    _adam: torch.optim.Optimizer | None = None
    if _effective_optimizer == 'adam':
        _adam = torch.optim.Adam(
            params,
            lr=effective_ADAM_LR, betas=ADAM_BETAS, eps=ADAM_EPS,
            weight_decay=ADAM_WEIGHT_DECAY)
        # ── Riemannian Adam for neel ansatz ─────────────────────────────────
        # The neel free param h lives on the unit sphere S^(N-1) in h-space.
        # Standard (Euclidean) Adam ignores this constraint: its accumulated
        # first moment m_t = β₁·m_{t-1} + (1-β₁)·g_t averages tangential
        # gradients from DIFFERENT sphere basepoints, so m_t drifts radially.
        # When m_t has a radial component, Adam overshoots the sphere,
        # causing oscillations in the loss after ~O(1/lr) steps.
        #
        # Fix: register a gradient hook that projects ∂E/∂h to the tangent
        # plane at the CURRENT h before Adam sees it.  Combined with:
        #   (a) post-step sphere retraction  (h ← h/||h||)
        #   (b) first-moment projection      (m ← m - (m·ĥ)ĥ)
        # this implements Riemannian Adam on S^(N-1) at negligible extra cost.
        #
        # Algebraic identity used: ||gather_scale(h)||_F = ||h||_F exactly
        # (the sqrt_mult weights are chosen to make the map isometric), so
        # keeping h on S^(N-1) automatically gives a unit-norm physical tensor
        # without any explicit normalisation in neel_param_to_a.
        if ansatz_cfg == ANSATZ_REGISTRY['neel'] and len(params) == 1:
            _neel_h_param = params[0]  # closed over in the hook (current ref)
            def _neel_tangent_hook(grad: torch.Tensor) -> torch.Tensor:
                """Project gradient to tangent plane at current h."""
                h_now = _neel_h_param.detach()
                h_dir = h_now / (h_now.norm() + 1e-30)
                return grad - (grad.reshape(-1) @ h_dir.reshape(-1)) * h_dir
            _neel_h_param.register_hook(_neel_tangent_hook)

        # ── Riemannian gradient hook for neel_legacy (sphere tangent projection) ──
        # The parameter a_raw is a S3-symmetric unit-norm tensor.  Gradients that
        # flow back through _derive_abcdef → symmetrize_virtual_legs are already
        # S3-symmetric (symmetric projection is self-adjoint).  We only need to
        # subtract the radial component (sphere tangent constraint).
        if ansatz_cfg == ANSATZ_REGISTRY['neel_legacy'] and len(params) == 1:
            _neel_leg_param = params[0]
            def _neel_legacy_tangent_hook(grad: torch.Tensor) -> torch.Tensor:
                """Project gradient to tangent plane of S^(N-1) at current a."""
                a_now = _neel_leg_param.detach()
                a_dir = a_now / (a_now.norm() + 1e-30)
                return grad - (grad.reshape(-1) @ a_dir.reshape(-1)) * a_dir
            _neel_leg_param.register_hook(_neel_legacy_tangent_hook)

    # ── L-BFGS factory + optional initial instance ────────────────────────────
    # _make_lbfgs is always defined when lbfgs might be used (both in pure-lbfgs
    # mode and in warmup mode where lbfgs is the finishing optimizer).
    # last_ctm_steps / last_cn: shared between the closure (nonlocal writes)
    # and the outer step (read-back after _lbfgs.step returns).  Declared here
    # so the history-clear hooks can inspect the *previous* step's values.
    _lbfgs: torch.optim.Optimizer | None = None
    last_ctm_steps: int | None = None
    last_cn: dict[str, float] | None = None
    if OPTIMIZER == 'lbfgs' or USE_ADAM_WARMUP_THEN_LBFGS or skip_adam_warmup:
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


        if _effective_optimizer == 'lbfgs' or skip_adam_warmup:
            # Pure L-BFGS mode (or skipping Adam warmup for warm-started chi):
            # create initial LBFGS instance now.
            _lbfgs = _make_lbfgs(*params)
            _lbfgs_outer_steps: int = 0  # reset on fresh LBFGS creation
            if skip_adam_warmup and USE_ADAM_WARMUP_THEN_LBFGS:
                _effective_optimizer = 'lbfgs'  # ensure correct branch in loop
                _switched_to_lbfgs   = True
                _final_lbfgs_run     = True  # warm-start: skip-Adam path is always the final run
                # _do_switch_to_adam is never defined in this code path (no Adam phase
                # was entered), but it's also never called: _final_lbfgs_run=True means
                # the convergence check hits `break` every time.

    # ── Density-matrix cache (fed by every closure call; read after loop) ──────
    # The three energy expectation functions in core.py write all 36 density
    # matrices (6 NN + 6 NNN per env) into _core._RHO_CACHE whenever
    # _core._CACHE_RHOS is True.  This reuses the open/closed tensors already
    # built for the energy computation — zero extra tensor builds.
    # The last closure call inside optimizer.step() is at the accepted parameter
    # point (strong-Wolfe line search finalises there), so _core._RHO_CACHE
    # after the step holds density matrices at the true optimum of this step.
    _core._CACHE_RHOS = True
    _core._RHO_CACHE.clear()

    def _loss_with_differentiable_ctmrg() -> tuple[torch.Tensor, int]:
        """Compute loss with CTMRG inside the autograd graph.

        LBFGS calls its closure multiple times per step; therefore this
        function must be called fresh on every closure evaluation.
        Never reuse environment tensors across calls.
        """
        # Derive all 6 single-layer tensors from the optimized parameters.
        # For unrestricted: params = [a, b, c, d, e, f] directly.
        # For single-tensor ansätze: symmetrize params[0] and derive (a..f).
        a, b, c, d, e, f = _derive_abcdef(params, ansatz_cfg, D_bond)
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
                    C21CD, T1F, T2A,
                    C21EB, T1D, T2C,
                    C21AF, T1B, T2E):
                with torch.no_grad():
                    _e1 = energy_expectation_nearest_neighbor_3ebadcf_bonds(
                        aN, bN, cN, dN, eN, fN, *Js[0:12], SdotS,
                        chi, D_bond, d_PHYS,
                        C21CD, T1F, T2A)
                    _e2 = energy_expectation_nearest_neighbor_3afcbed_bonds(
                        aN, bN, cN, dN, eN, fN, *Js[12:24], SdotS,
                        chi, D_bond, d_PHYS,
                        C21EB, T1D, T2C)
                    _e3 = energy_expectation_nearest_neighbor_other_3_bonds(
                        aN, bN, cN, dN, eN, fN, *Js[24:36], SdotS,
                        chi, D_bond, d_PHYS,
                        C21AF, T1B, T2E)
                    return (_e1 + _e2 + _e3).item()
                
        all10 = CTMRG_from_init_to_stop(
            A, B, C, Dt, E, F, chi, D_sq,
            CTM_MAX_STEPS, CTM_CONV_THR, ENV_IDENTITY_INIT,
            energy_proxy_fn=_proxy_fn)


        (C21CD, T1F, T2A,
         C21EB, T1D, T2C,
         C21AF, T1B, T2E,
         ctm_steps) = all10

        # ── Last-resort zero-env guard (gradient path) ────────────────────
        # CTMRG already retried with escalating noise internally.
        # If it STILL returned zero, skip the expensive energy computation
        # and raise so the closure handler returns a penalty loss.
        _env_corners = (C21CD, C21EB, C21AF)
        if all(torch.linalg.norm(c).item() < 1e-30 for c in _env_corners):
            raise FloatingPointError(
                "CTMRG env collapsed to zero after all internal retries")

        # ── Compute the three energy expectations (checkpointed, optionally multi-GPU) ─
        # _three_env_energy_loss_parallel wraps each energy call in
        # _ckpt(use_reentrant=False), saving ~5 GB (D=7) / ~15 GB (D=8) of
        # autograd intermediates.  With N_GPUS >= 2 (CUDA), the three calls
        # run concurrently on separate GPUs via a thread pool (see --ngpu).
        # When _core._CACHE_RHOS is True, each energy function also writes its
        # 12 density matrices into _core._RHO_CACHE — zero extra tensor builds.
        loss = _three_env_energy_loss_parallel(
            aN, bN, cN, dN, eN, fN, Js, SdotS, chi, D_bond, d_PHYS,
            env1=(C21CD, T1F, T2A),
            env2=(C21EB, T1D, T2C),
            env3=(C21AF, T1B, T2E),
        )

        return loss, int(ctm_steps)


    def _loss_with_differentiable_ctmrg_ADAMlooser() -> tuple[torch.Tensor, int]:
        """Compute loss with CTMRG inside the autograd graph.

        LBFGS calls its closure multiple times per step; therefore this
        function must be called fresh on every closure evaluation.
        Never reuse environment tensors across calls.
        """
        # Derive all 6 single-layer tensors from the optimized parameters.
        # For unrestricted: params = [a, b, c, d, e, f] directly.
        # For single-tensor ansätze: symmetrize params[0] and derive (a..f).
        a, b, c, d, e, f = _derive_abcdef(params, ansatz_cfg, D_bond)
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
                    C21CD, T1F, T2A,
                    C21EB, T1D, T2C,
                    C21AF, T1B, T2E):
                with torch.no_grad():
                    _e1 = energy_expectation_nearest_neighbor_3ebadcf_bonds(
                        aN, bN, cN, dN, eN, fN, *Js[0:12], SdotS,
                        chi, D_bond, d_PHYS,
                        C21CD, T1F, T2A)
                    _e2 = energy_expectation_nearest_neighbor_3afcbed_bonds(
                        aN, bN, cN, dN, eN, fN, *Js[12:24], SdotS,
                        chi, D_bond, d_PHYS,
                        C21EB, T1D, T2C)
                    _e3 = energy_expectation_nearest_neighbor_other_3_bonds(
                        aN, bN, cN, dN, eN, fN, *Js[24:36], SdotS,
                        chi, D_bond, d_PHYS,
                        C21AF, T1B, T2E)
                    return (_e1 + _e2 + _e3).item()
                
        all10 = CTMRG_from_init_to_stop(
            A, B, C, Dt, E, F, chi, D_sq,
            CTM_MAX_STEPS, 5*CTM_CONV_THR, ENV_IDENTITY_INIT,
            energy_proxy_fn=_proxy_fn)


        (C21CD, T1F, T2A,
         C21EB, T1D, T2C,
         C21AF, T1B, T2E,
         ctm_steps) = all10

        # ── Last-resort zero-env guard (gradient path) ────────────────────
        # CTMRG already retried with escalating noise internally.
        # If it STILL returned zero, skip the expensive energy computation
        # and raise so the closure handler returns a penalty loss.
        _env_corners = (C21CD, C21EB, C21AF)
        if all(torch.linalg.norm(c).item() < 1e-30 for c in _env_corners):
            raise FloatingPointError(
                "CTMRG env collapsed to zero after all internal retries")

        # ── Compute the three energy expectations (checkpointed, optionally multi-GPU) ─
        # _three_env_energy_loss_parallel wraps each energy call in
        # _ckpt(use_reentrant=False), saving ~5 GB (D=7) / ~15 GB (D=8) of
        # autograd intermediates.  With N_GPUS >= 2 (CUDA), the three calls
        # run concurrently on separate GPUs via a thread pool (see --ngpu).
        # When _core._CACHE_RHOS is True, each energy function also writes its
        # 12 density matrices into _core._RHO_CACHE — zero extra tensor builds.
        loss = _three_env_energy_loss_parallel(
            aN, bN, cN, dN, eN, fN, Js, SdotS, chi, D_bond, d_PHYS,
            env1=(C21CD, T1F, T2A),
            env2=(C21EB, T1D, T2C),
            env3=(C21AF, T1B, T2E),
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
        if _effective_optimizer == 'lbfgs':
            _core.set_ctm_conv_mode(CTM_CONV_MODE,
                        e_threshold=CTM_E_CONV_THRESHOLD)
            if _lbfgs is None:
                raise RuntimeError("L-BFGS optimizer requested but was not initialized")

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

            # ── Sphere retraction for neel ansatz after L-BFGS step ──────
            # L-BFGS line search may move h off S^(N-1).  neel_param_to_a
            # normalises h internally so energy values are always correct,
            # but the stored parameter can drift.  Retract it now so the
            # next step's linearisation is at the correct basepoint.
            if ansatz_cfg == ANSATZ_REGISTRY['neel'] and len(params) == 1:
                with torch.no_grad():
                    _nh = params[0]
                    _hn = _nh.data.norm()
                    if _hn > 1e-30:
                        _nh.data.div_(_hn)

            # ── Sphere retraction for neel_legacy after L-BFGS step ───────
            # L-BFGS may move a off S^(N-1).  Resymmetrize for safety and
            # renormalize so the next linearisation is at a valid base-point.
            if ansatz_cfg == ANSATZ_REGISTRY['neel_legacy'] and len(params) == 1:
                with torch.no_grad():
                    _na = params[0]
                    _na.data.copy_(symmetrize_virtual_legs(_na.data))
                    _an = _na.data.norm()
                    if _an > 1e-30:
                        _na.data.div_(_an)

            # After a successful step with full SVD, revert to partial for all
            # subsequent steps (the noisy initial basin has been escaped).
            if _core._USE_FULL_SVD:
                _core._USE_FULL_SVD = False

            loss_item = loss_val.item()
            del loss_val          # release the 0-dim tensor + its dead graph
            _lbfgs_outer_steps += 1
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
        else:  # adam (initial or full run) — 1 gradient step per env refresh
            _core.set_ctm_conv_mode(CTM_CONV_MODE,
                        e_threshold=5*CTM_E_CONV_THRESHOLD)
            if _adam is None:
                raise RuntimeError("Adam optimizer requested but was not initialized")
            _adam.zero_grad()
            _loss, ctm_steps = _loss_with_differentiable_ctmrg_ADAMlooser()
            loss_item = _loss.detach().item()  # save scalar before releasing graph
            _loss.backward()
            del _loss   # release graph immediately before Adam updates params
            if ADAM_GRAD_CLIP is not None:
                torch.nn.utils.clip_grad_norm_(params, max_norm=ADAM_GRAD_CLIP)
            _adam.step()
            # ── GPU memory cleanup after Adam step ───────────────────────
            if params[0].device.type == 'cuda':
                gc.collect()
                torch.cuda.empty_cache()
            # ── Riemannian sphere retraction for neel ansatz ─────────────
            # After every Adam step, h has moved off the unit sphere by
            # O(lr²) (tangential step + discretisation error).  Retract it
            # back to S^(N-1) and project Adam's first moment to the new
            # tangent plane.  Together with the gradient hook registered at
            # Adam creation, this completes the Riemannian Adam update.
            if ansatz_cfg == ANSATZ_REGISTRY['neel'] and len(params) == 1:
                with torch.no_grad():
                    _nh = params[0]
                    _hn = _nh.data.norm()
                    if _hn > 1e-30:
                        _nh.data.div_(_hn)   # retract to S^(N-1)
                    _state = _adam.state.get(_nh)
                    if _state and 'exp_avg' in _state:
                        _h_dir = _nh.data / (_nh.data.norm() + 1e-30)
                        _m = _state['exp_avg']
                        _m.sub_((_m.reshape(-1) @ _h_dir.reshape(-1)) * _h_dir)
            # ── Riemannian sphere retraction for neel_legacy ──────────────
            # Resymmetrize + renormalize a, then project Adam's first moment
            # to the new tangent plane.  Resymmetrizing each step is a no-op
            # when gradients flow through symmetrize_virtual_legs (self-adjoint
            # proj preserves symmetry), but guards against floating-point drift.
            if ansatz_cfg == ANSATZ_REGISTRY['neel_legacy'] and len(params) == 1:
                with torch.no_grad():
                    _na = params[0]
                    _na.data.copy_(symmetrize_virtual_legs(_na.data))
                    _an = _na.data.norm()
                    if _an > 1e-30:
                        _na.data.div_(_an)   # retract to S^(N-1)
                    _state_leg = _adam.state.get(_na)
                    if _state_leg and 'exp_avg' in _state_leg:
                        _a_dir = _na.data / (_na.data.norm() + 1e-30)
                        _m_leg = _state_leg['exp_avg']
                        _m_leg.sub_((_m_leg.reshape(-1) @ _a_dir.reshape(-1)) * _a_dir)
            # loss_item already saved above before del _loss
            _adam_steps_taken += 1
            _adam_loss_window.append(loss_item)  # feed flat-landscape tracker
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
            print(f"    [ZERO-GUARD] loss>={loss_item:+.10f} \u2248 0 "
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

        # ── Adam → L-BFGS warmup switch ──────────────────────────────────────
        if (USE_ADAM_WARMUP_THEN_LBFGS
                and _effective_optimizer == 'adam'
                and prev_loss is not None):

            def _do_switch_to_lbfgs(reason: str, final: bool = False) -> None:
                nonlocal _adam, _lbfgs, _lbfgs_outer_steps, _effective_optimizer
                nonlocal _switched_to_lbfgs
                nonlocal _final_lbfgs_run
                _tag = '[warmup-final]' if final else '[warmup]'
                print(f"    {_tag} Adam→L-BFGS switch at step {step} ({reason})")
                del _adam
                _adam = None
                _adam_loss_window.clear()  # reset for any future Adam phase
                if params[0].device.type == 'cuda':
                    gc.collect()
                    torch.cuda.empty_cache()
                _lbfgs = _make_lbfgs(*params)  # type: ignore[name-defined]
                _lbfgs_outer_steps = 0
                _effective_optimizer = 'lbfgs'
                _switched_to_lbfgs   = True
                _final_lbfgs_run     = final
                loss_history.clear()  # don't let stale Adam losses trigger cycle check

            def _do_switch_to_adam(reason: str) -> None:
                """Switch from L-BFGS back to Adam for another exploration round.

                Called whenever a non-final LBFGS run converges or detects a
                cycle.  Deletes the LBFGS instance, creates a fresh Adam (starting
                from zero moments — this is intentional: Adam re-warms momentum
                from the current LBFGS minimum rather than from stale history),
                and resets all Adam tracking counters.  Gradient hooks registered
                at the very first Adam creation persist on the param tensors and
                are NOT re-registered (they close over the params list which
                never changes).  _switched_to_lbfgs is kept True so that the
                next LBFGS run's _lbfgs_warmed_up guard still fires correctly.
                """
                nonlocal _adam, _lbfgs, _effective_optimizer, _lbfgs_outer_steps
                nonlocal _adam_steps_taken
                print(f"    [cycle] L-BFGS→Adam switch at step {step} ({reason})")
                _lbfgs = None  # clear ref; existing LBFGS object GC'd naturally
                if params[0].device.type == 'cuda':
                    gc.collect()
                    torch.cuda.empty_cache()
                # Fresh Adam with zero state: re-warms momentum from the
                # current point (the LBFGS minimum), not from stale Adam history.
                _adam = torch.optim.Adam(
                    params,
                    lr=effective_ADAM_LR, betas=ADAM_BETAS, eps=ADAM_EPS,
                    weight_decay=ADAM_WEIGHT_DECAY)
                # Gradient hooks are already registered on the param tensors
                # from the first Adam creation and persist across optimizer swaps.
                # Do NOT re-register them.
                _adam_steps_taken = 0
                _effective_optimizer = 'adam'
                # CRITICAL: reset _lbfgs_outer_steps to 0.  During the new Adam
                # phase _lbfgs_warmed_up = _switched_to_lbfgs and _lbfgs_outer_steps >= 3.
                # Without this reset, _lbfgs_warmed_up stays True with the old LBFGS
                # step count and the convergence check fires spuriously on Adam deltas.
                _lbfgs_outer_steps = 0
                loss_history.clear()  # fresh history for new Adam phase
                _adam_loss_window.clear()  # reset flat-landscape window
                # Keep _switched_to_lbfgs=True and _final_lbfgs_run unchanged.

            # ── Early near-optimal switch (first 3 Adam steps) ───────────
            # If Δloss > -1e-6 on any of the first 3 steps, the starting
            # point is already near the minimum — Adam's fixed-lr steps
            # will only kick it away.  Switch to the FINAL L-BFGS run.
            # This can happen on any Adam phase (initial or after cycling back),
            # not just the first one.  We do NOT gate on first_chi_of_D here:
            # the "first 3 steps" condition is enough to detect near-optimality
            # regardless of how this chi was initialised.
            if _adam_steps_taken <= 3 and delta > -1e-7:
                _do_switch_to_lbfgs(
                    f"Δ={delta:+.2e} > -1e-7 on Adam step {_adam_steps_taken} "
                    f"(near-optimal start → final LBFGS)",
                    final=True)
            else:
                # ── Flat-landscape escape (window-based) ─────────────────
                # Only switch mechanism for Adam (besides the near-optimal
                # start check above): fire when the total spread over
                # ADAM_FLAT_PATIENCE steps is negligible.  Patience-based
                # switching on |Δloss| is intentionally removed — Adam at a
                # local summit decelerates and can make 17 consecutive tiny
                # steps before bouncing back, causing premature switching.
                # The flat-plateau window is safer: it requires sustained
                # near-zero movement over 70 steps, not just 17.
                # Fires when the total spread over ADAM_FLAT_PATIENCE steps
                # is negligible (< threshold).  Suppressed if EITHER:
                #   (A) same-sign:  all first-diffs d_i = loss[i+1]-loss[i]
                #       are strictly same sign (Adam making directional steps)
                #   (B) monotone:   all second-diffs d_i-d_{i+1} are strictly
                #       same sign (first-diffs themselves accelerating/decelerating
                #       consistently), even if sign alternates
                # Both checked over the last 17 recent losses.
                if (_effective_optimizer == 'adam'
                        and len(_adam_loss_window) == ADAM_FLAT_PATIENCE):
                    _wmin = min(_adam_loss_window)
                    _spread = max(_adam_loss_window) - _wmin
                    if _spread < 2e-3:
                        # First-diffs: d[0]=window[-1]-window[-2], ...
                        # Use deque negative indexing — no list copy.
                        _N  = 17
                        _d  = [_adam_loss_window[-j] - _adam_loss_window[-(j+1)]
                               for j in range(1, _N)]      # _N-1 values
                        # (A) same-sign check (strict < or strict >)
                        _d0 = _d[0]
                        _same_sign = ((_d0 < 0 and all(v < 0 for v in _d[1:])) or
                                      (_d0 > 0 and all(v > 0 for v in _d[1:])))
                        if not _same_sign:
                            # (B) monotone check on second-diffs
                            _dd  = [_d[j] - _d[j+1] for j in range(len(_d) - 1)]
                            _dd0 = _dd[0]
                            _monotone = ((_dd0 < 0 and all(v < 0 for v in _dd[1:])) or
                                         (_dd0 > 0 and all(v > 0 for v in _dd[1:])))
                            if not _monotone:
                                _do_switch_to_lbfgs(
                                    f"flat plateau over {ADAM_FLAT_PATIENCE} Adam steps "
                                    f"(spread={_spread:.2e} < "
                                    f"2e-3)")
        # ─────────────────────────────────────────────────────────────────────

        # Convergence / cycle-detection checks are ONLY active after the switch
        # to L-BFGS.  Adam must run freely until _do_switch_to_lbfgs fires —
        # stopping it early on a small Δ or a loss-history repeat defeats the
        # entire purpose of the Adam warmup.
        # Additionally skip the first 3 L-BFGS outer steps: the Hessian
        # approximation is still bootstrapping from identity, so the energy
        # drop there may be artificially small.
        _lbfgs_warmed_up = _switched_to_lbfgs and _lbfgs_outer_steps >= 3
        if _lbfgs_warmed_up and prev_loss is not None and abs(delta) < OPT_CONV_THRESHOLD:
            if _final_lbfgs_run:
                print(f"    Outer convergence at step {step} (Δ={delta:.2e}) — final LBFGS done.")
                break
            else:
                # Normal (non-final) LBFGS converged: cycle back to Adam for
                # further exploration, then LBFGS again until early switch fires.
                _do_switch_to_adam(
                    f"LBFGS converged (Δ={delta:.2e}), cycling back to Adam")
        elif _lbfgs_warmed_up and any(abs(loss_item - h) < 1e-9 for h in loss_history):
            if _final_lbfgs_run:
                print(f"    Cycle detected at step {step} "
                      f"(amplitude={abs(delta):.3e}); stopping (final LBFGS).")
                break
            else:
                _do_switch_to_adam(
                    f"LBFGS cycle detected (amplitude={abs(delta):.3e}), cycling back to Adam")
        # ─────────────────────────────────────────────────────────────────────
        loss_history.append(loss_item)
        prev_loss = loss_item
        step += 1

    return (tuple(best_params), best_loss, step,
            (_core._RHO_CACHE.get('env1'),
             _core._RHO_CACHE.get('env2'),
             _core._RHO_CACHE.get('env3')),
            _switched_to_lbfgs)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global TENSORDTYPE, OPTIMIZER, USE_ADAM_WARMUP_THEN_LBFGS, ADAM_FLAT_PATIENCE
    parser = argparse.ArgumentParser(
        description="10-12 h iPEPS benchmark: sweep D_bond (outer) × chi (inner)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--hours', type=float, default=TOTAL_BUDGET_HOURS,
        help='Total wall-clock budget in hours.')
    parser.add_argument(
        '--config', type=str, default=None,
        help='Path to YAML config file with hyperparameters. Values from file '
             'override defaults but are overridden by explicit CLI arguments. '
             'Example: --config sweep_hyperparams.yaml')
    parser.add_argument(
        '--Ds', nargs='+', type=int, default=None, dest='D_bonds',
        help='Space-separated list of D_bond values to sweep. '
             'Example: --Ds 3 7 4  (sweeps D=3,7,4 in that order). '
             'If provided, chi-min/max/step must either match length or be omitted '
             '(defaults will be inferred).')
    parser.add_argument(
        '--chi-min', nargs='+', type=int, default=None, dest='chi_min',
        help='Minimum chi for each D (space-separated). If length differs from '
             '--Ds, missing values are taken from CHI_MIN_LIST defaults. '
             'Example: --chi-min 40 55 72')
    parser.add_argument(
        '--chi-max', nargs='+', type=int, default=None, dest='chi_max',
        help='Maximum chi for each D (space-separated). If length differs from '
             '--Ds, missing values are taken from CHI_MAX_LIST defaults. '
             'Example: --chi-max 80 100 120')
    parser.add_argument(
        '--chi-step', nargs='+', type=int, default=None, dest='chi_step',
        help='Chi step size for each D (space-separated). If length differs from '
             '--Ds, missing values are taken from CHI_STEP_LIST defaults. '
             'Example: --chi-step 10 10 10  → schedules will be range(min, max+1, step)')
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
        '--resume-folder', default=None, dest='resume_folder',
        help='Directory to scan for per-D warm-start checkpoints.  '
             'For each D in D_bond_list, finds the highest-chi '
             'sweep_D{D}_chi*_best.pt in the directory and uses it as '
             'the starting point for that D\'s first chi level.')
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
             "'adam' (more robust on noisy landscapes, constant LR). "
             "Ignored when --adam-warmup-lbfgs is set (warmup always uses adam "
             "then switches to lbfgs regardless of this flag).")
    parser.add_argument(
        '--adam-warmup-lbfgs', dest='adam_warmup_lbfgs', action='store_true',
        default=USE_ADAM_WARMUP_THEN_LBFGS,
        help='Use Adam for initial non-convex exploration, then switch to L-BFGS '
             'once the flat-plateau window fires. Recommended for frustrated '
             'systems (J2 > 0). Overrides --optimizer for the warmup phase; '
             'L-BFGS phase uses LBFGS_* hyperparameters as usual.')
    parser.add_argument(
        '--adam-flat-patience', type=int, default=ADAM_FLAT_PATIENCE,
        dest='adam_flat_patience',
        help='Number of Adam steps in the sliding window for flat-plateau detection '
             '(= ADAM_FLAT_PATIENCE). If max-min of window < switch threshold, '
             'triggers a non-final switch to L-BFGS. '
             'Only active when --adam-warmup-lbfgs is set.')
    parser.add_argument(
        '--ansatz', choices=list(ANSATZ_REGISTRY.keys()), default='c3vypi',
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
    if args.ansatz not in C3_COMPATIBLE_ANSATZES:
        print(f"Ansatz '{args.ansatz}' is not C3-compatible; skipping this run.")
        return
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
    if args.adam_warmup_lbfgs:
        USE_ADAM_WARMUP_THEN_LBFGS = True
    ADAM_FLAT_PATIENCE = args.adam_flat_patience
    if True: #USE_ADAM_WARMUP_THEN_LBFGS:
        print(f"  Optimizer: Adam warmup \u2192 L-BFGS "
              f"(flat-plateau switch over {ADAM_FLAT_PATIENCE} steps, spread < 2e-3)")
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
    _core._SVD_CPU_OFFLOAD_THRESHOLD = 0  # always GPU (SVD_CPU_OFFLOAD_THRESHOLD removed)
    _core.set_rsvd_mode(RSVD_MODE,
                        neumann_terms=RSVD_NEUMANN_TERMS,
                        power_iters=RSVD_POWER_ITERS)
    _core.set_ctm_conv_mode(CTM_CONV_MODE,
                            e_threshold=CTM_E_CONV_THRESHOLD)

    # ── parse D_bond list ─────────────────────────────────────────────────────
    D_bond_list: list[int] = list(args.D_bonds)
    d_PHYS = args.d_phys
    total_budget = args.hours * 3600.0
    t_global_start = time.perf_counter()

    # ── chi schedules (generated from chi_min/max/step) ───────────────────────
    # Smart inference: if chi param not provided, look up defaults by D index
    _n_D = len(D_bond_list)
    
    def _infer_chi_param(user_vals, default_vals, default_D_list, param_name):
        """Infer chi parameter list from user input + defaults.
        
        Logic:
        - If None (not provided): for each D in D_bond_list, find its index
          in default_D_list and use the corresponding value from default_vals
        - If provided: length MUST equal len(D_bond_list), otherwise ERROR
        """
        if user_vals is None:
            # No user input: look up defaults by D index
            result = []
            for D in D_bond_list:
                if D in default_D_list:
                    idx = default_D_list.index(D)
                    if idx < len(default_vals):
                        result.append(default_vals[idx])
                    else:
                        raise ValueError(
                            f"{param_name}: D={D} found at index {idx} in D_BOND_LIST, "
                            f"but {param_name[2:].upper()}_LIST only has {len(default_vals)} values.")
                else:
                    raise ValueError(
                        f"{param_name}: D={D} not found in default D_BOND_LIST={default_D_list}. "
                        f"When {param_name} is not provided, all D values must exist in D_BOND_LIST "
                        f"so defaults can be looked up.")
            return result
        
        user_list = list(user_vals)
        if len(user_list) != _n_D:
            raise ValueError(
                f"{param_name}: length mismatch. --Ds has {_n_D} values, "
                f"but {param_name} has {len(user_list)} values. "
                f"They must be equal (or omit {param_name} to use defaults).")
        return user_list
    
    chi_min_list  = _infer_chi_param(args.chi_min,  CHI_MIN_LIST,  D_BOND_LIST, '--chi-min')
    chi_max_list  = _infer_chi_param(args.chi_max,  CHI_MAX_LIST,  D_BOND_LIST, '--chi-max')
    chi_step_list = _infer_chi_param(args.chi_step, CHI_STEP_LIST, D_BOND_LIST, '--chi-step')
    
    schedules = {}
    for i, D in enumerate(D_bond_list):
        chi_min  = chi_min_list[i]
        chi_max  = chi_max_list[i]
        chi_step = chi_step_list[i]
        schedules[D] = list(range(chi_min, chi_max + 1, chi_step))
        if not schedules[D]:
            raise ValueError(
                f"Empty chi schedule for D={D}: chi_min={chi_min}, chi_max={chi_max}, "
                f"chi_step={chi_step} produces no values.")

    # ── time budget: equal split across D values ──────────────────────────────
    d_budgets = {D: total_budget / len(D_bond_list) for D in D_bond_list}

    # ── output directory ──────────────────────────────────────────────────────
    run_ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    _J2_str = f'J2_{args.J2}'.replace('.', 'p')
    _default_outdir = os.path.join(MY_OUTPUT_OUTERDIR,
                                   f'{ansatz_cfg["yaml_name"]}__{_J2_str}_{run_ts}')
    output_dir = args.output_dir or _default_outdir
    os.makedirs(output_dir, exist_ok=True)
    _install_tee(output_dir)  # mirror all stdout to run.log

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
        d_budgets_seconds  = {str(k): round(v, 2) for k, v in d_budgets.items()},
        schedules          = {str(k): v for k, v in schedules.items()},

        # ── physical model ─────────────────────────────────────────────────
        J1                 = args.J1,
        J2                 = args.J2,
        d_phys             = d_PHYS,
        n_sites            = N_SITES,

        # ── optimiser ──────────────────────────────────────────────────────
        optimizer          = OPTIMIZER,
        use_adam_warmup_then_lbfgs    = USE_ADAM_WARMUP_THEN_LBFGS,
        adam_flat_patience            = ADAM_FLAT_PATIENCE,
        # L-BFGS
        lbfgs_lr           = LBFGS_LR,
        lbfgs_max_iter     = LBFGS_MAX_ITER,
        lbfgs_history      = LBFGS_HISTORY,
        opt_tol_grad               = OPT_TOL_GRAD,
        opt_tol_change             = OPT_TOL_CHANGE,
        opt_conv_threshold         = OPT_CONV_THRESHOLD,
        # Adam
        adam_lr            = ADAM_LR,
        adam_betas         = list(ADAM_BETAS),
        adam_eps           = ADAM_EPS,
        adam_weight_decay  = ADAM_WEIGHT_DECAY,
        adam_grad_clip     = ADAM_GRAD_CLIP,
        # (adam_steps_per_ctm removed — always 1)

        # ── SVD / rSVD ─────────────────────────────────────────────────────
        # (svd_cpu_offload_threshold removed — always 0 / GPU)
        rsvd_mode          = RSVD_MODE,
        rsvd_neumann_terms = RSVD_NEUMANN_TERMS,
        rsvd_power_iters   = RSVD_POWER_ITERS,

        # ── CTMRG ──────────────────────────────────────────────────────────
        ctm_max_steps          = CTM_MAX_STEPS,
        ctm_conv_thr           = CTM_CONV_THR,
        ctm_conv_mode          = CTM_CONV_MODE,
        ctm_e_conv_threshold   = CTM_E_CONV_THRESHOLD,
        # (ctm_e_proxy_interval removed — always 1)
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
    print(f"  J1={args.J1}  J2={args.J2}  d_phys={d_PHYS}  Total budget: {args.hours:.2f} h")
    print(f"  Device       : {_core.DEVICE}")
    print(f"  D_bond sweep : {D_bond_list}")
    for D in D_bond_list:
        bh = d_budgets[D] / 3600
        print(f"    D={D}: chi={schedules[D]}  budget={bh:.2f} h")
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
        ckpt = torch.load(args.resume, map_location=_core.DEVICE, weights_only=False)
        resume_D   = ckpt.get('D_bond')
        resume_chi = ckpt.get('chi')
        ckpt_step  = ckpt.get('step', 0) + 1
        ckpt_loss  = ckpt.get('loss', float('nan'))
        loaded = tuple(ckpt[k] for k in ansatz_cfg['ckpt_keys'])
        best_params_by_D[resume_D] = _new_tensors_from_data(loaded)
        # Normalise neel free param to unit sphere — old checkpoints may have
        # stored h with arbitrary norm; the first gradient step would otherwise
        # have degraded scaling until the Adam retraction fixes it.
        if ansatz_cfg == ANSATZ_REGISTRY['neel']:
            with torch.no_grad():
                _p = best_params_by_D[resume_D][0]
                _n = _p.data.norm()
                if _n > 1e-30:
                    _p.data.div_(_n)
        if ansatz_cfg == ANSATZ_REGISTRY['neel_legacy']:
            with torch.no_grad():
                _p = best_params_by_D[resume_D][0]
                _p.data.copy_(symmetrize_virtual_legs(_p.data))
                _n = _p.data.norm()
                if _n > 1e-30:
                    _p.data.div_(_n)
        del ckpt, loaded; gc.collect()
        global_step = ckpt_step
        print(f"  Resumed from {args.resume}  "
              f"(D={resume_D}, chi={resume_chi}, "
              f"loss={ckpt_loss:.6f})\n")

    # ── Resume-folder: per-D warm-start from highest-chi checkpoint ───────────
    if args.resume_folder:
        import glob as _glob, re as _re
        def _chi_from_path(p: str) -> int:
            m = _re.search(r'_chi(\d+)_best', p)
            return int(m.group(1)) if m else -1
        for _D in D_bond_list:
            _pattern = os.path.join(args.resume_folder, f'sweep_D{_D}_chi*_best.pt')
            _files = _glob.glob(_pattern)
            if not _files:
                print(f"  [resume-folder] D={_D}: no checkpoint found in "
                      f"{args.resume_folder} — using default init")
                continue
            _best_f = max(_files, key=_chi_from_path)
            _ckpt = torch.load(_best_f, map_location=_core.DEVICE, weights_only=False)
            _loaded = tuple(_ckpt[k] for k in ansatz_cfg['ckpt_keys'])
            best_params_by_D[_D] = _new_tensors_from_data(_loaded)
            if ansatz_cfg == ANSATZ_REGISTRY['neel']:
                with torch.no_grad():
                    _rp = best_params_by_D[_D][0]
                    _rn = _rp.data.norm()
                    if _rn > 1e-30:
                        _rp.data.div_(_rn)
            if ansatz_cfg == ANSATZ_REGISTRY['neel_legacy']:
                with torch.no_grad():
                    _rp = best_params_by_D[_D][0]
                    _rp.data.copy_(symmetrize_virtual_legs(_rp.data))
                    _rn = _rp.data.norm()
                    if _rn > 1e-30:
                        _rp.data.div_(_rn)
            print(f"  [resume-folder] D={_D}: loaded {os.path.basename(_best_f)} "
                  f"(chi={_chi_from_path(_best_f)})")
            del _ckpt, _loaded; gc.collect()
        print()

    # ══════════════════════════════════════════════════════════════════════════
    # Outer loop: D_bond
    # ══════════════════════════════════════════════════════════════════════════
    for D_bond in D_bond_list:
        D_sq     = D_bond ** 2
        D_budget = d_budgets[D_bond]
        chis     = schedules[D_bond]
        D_start_time = time.perf_counter()

        print(f"\n{'═'*76}")
        print(f"  D_bond = {D_bond}   D²={D_sq}  D⁴={D_bond**4}  "
              f"budget={D_budget/3600:.2f} h")
        print(f"  chi schedule: {chis}  ({len(chis)} levels, equal time split)")
        print(f"{'═'*76}")

        _d_restart_count = 0
        while True:  # ── D-restart harness; re-entered if env collapses ──
            # ── Warm-start from closest available smaller D ───────────────────
            # Primary: previous D in this run's schedule (D_bond_list order)
            _d_idx = D_bond_list.index(D_bond)
            prev_D = D_bond_list[_d_idx - 1] if _d_idx > 0 else None
            # Secondary: tensors loaded by --resume that may be outside D_bond_list
            # (e.g. --resume D6_chi54 while D_bond_list starts at D=7).
            # Always prefer the closest D < D_bond regardless of source.
            if best_params_by_D.get(D_bond) is None and not args.rand_init_new_d:
                _resume_src = [
                    d for d in best_params_by_D
                    if d < D_bond
                    and best_params_by_D.get(d) is not None
                    and d not in D_bond_list
                ]
                if _resume_src:
                    _best_resume_src = max(_resume_src)
                    # Use it if closer to D_bond than the in-schedule prev_D,
                    # or if prev_D has no tensors yet.
                    if (prev_D is None
                            or best_params_by_D.get(prev_D) is None
                            or _best_resume_src > prev_D):
                        prev_D = _best_resume_src
            if args.rand_init_new_d and prev_D is not None:
                print(f"  [rand-D] random init for D={D_bond} (ignoring D={prev_D} tensors)")
            if (best_params_by_D.get(D_bond) is None
                    and prev_D is not None
                    and best_params_by_D.get(prev_D) is not None
                    and not args.rand_init_new_d):
                print(f"  Warm-starting from D={prev_D} tensors "
                      f"(padding {prev_D}→{D_bond}, noise={args.noise})")
                prev_tensors = best_params_by_D[prev_D]
                _pad_fn = ansatz_cfg.get('pad_fn')
                if _pad_fn is not None:
                    # Free-param ansatz: special pad that works on the reduced tensors
                    padded = _pad_fn(prev_tensors, prev_D, D_bond, d_PHYS, args.noise)
                else:
                    sym_fn = ansatz_cfg['symmetrize_fn']
                    # For n_params==6 with a batch symmetrize_fn (e.g. 'sym6_legacy'),
                    # the per-tensor symmetry is applied inside _derive_abcdef at every
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
                # Normalise sphere-constrained single-tensor ansatze after padding.
                # pad_tensor (symmetrize_fn path) symmetrizes but does not normalise;
                # the result has ||a|| ≈ √(D_new³·d) ≫ 1.  Without this, multiple
                # Adam steps elapse with the parameter far from the unit sphere.
                if ansatz_cfg == ANSATZ_REGISTRY['neel_legacy']:
                    with torch.no_grad():
                        _p = best_params_by_D[D_bond][0]
                        _p.data.copy_(symmetrize_virtual_legs(_p.data))
                        _pn = _p.data.norm()
                        if _pn > 1e-30:
                            _p.data.div_(_pn)
    
            # current best tensors at this D (None = random init at first chi)
            cur_params = best_params_by_D.get(D_bond)
    
            # ── Inner loop: chi ───────────────────────────────────────────────────
            _chi_collapse = False
            for chi_idx, chi in enumerate(chis):
                resumeEqulCurrent = False
                # Skip (D, chi) pairs that come before the resume point
                if resume_D is not None and resume_chi is not None:
                    if D_bond < resume_D:
                        continue
                    if D_bond == resume_D and chi < resume_chi:
                        continue
                    if D_bond == resume_D and chi == resume_chi:
                        resume_D = None   # resume point reached; continue normally
                        resumeEqulCurrent = True
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
                #print(cur_params)
                if chi_idx==0 and args.mean_field_init and not args.resume_folder and not resumeEqulCurrent:
                    _init_params = _make_mean_field_params(
                        ansatz_cfg, D_bond, d_PHYS, INIT_NOISE)
                    if chi_idx == 0:
                        print(f"  │  [mean-field] Néel product-state init for chi={chi}")
                    else:
                        print(f"  │  [mean-field] Néel product-state init for chi={chi} "
                              f"(overrides warm-start)")
                elif args.rand_init_new_chi and chi_idx > 0:
                    print(f"  │  [rand-chi] random init for chi={chi} (ignoring previous result)")
                    _init_params = None
                else:
                    _init_params = cur_params
    
                # Warm-started from the previous chi of the same D: the
                # tensors are already near a local minimum, so skip Adam
                # exploration and go straight to L-BFGS.
                _skip_adam = (
                    chi_idx > 0
                    and _init_params is not None
                    and _init_params is cur_params
                )

                _skip_adam = False
                
                if _skip_adam:
                    print(f"  │  [warm-start] chi_idx={chi_idx}>0: "
                          f"skipping Adam, starting directly with L-BFGS")

                try:
                    best_params_tuple, best_loss, global_step, _cached_rhos, _switched_to_lbfgs = optimize_at_chi(
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
                        skip_adam_warmup=_skip_adam,
                        first_chi_of_D=(chi_idx == 0),
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

                # ── Compute observables from cached density matrices ──────────
                # No extra CTMRG run: _cached_rhos holds the 36 density matrices
                # from the last accepted optimization step (see the side-channel
                # in _loss_with_differentiable_ctmrg).
                with torch.no_grad():
                    energy, correlations, magnetizations = _observables_from_rhos(
                        _cached_rhos, Js, SdotS, d_PHYS)
                energy_per_site = energy / N_SITES
                _save_observables_file(
                    os.path.join(output_dir,
                                 f"D_{D_bond}_chi_{chi}"
                                 f"_energy_magnetization_correlation.txt"),
                    D_bond, chi, energy, correlations, magnetizations, None)
                _print_observables_summary(
                    'OBS', D_bond, chi, energy, correlations, magnetizations, None)

                # ── Lookahead: chi_la = chi + D_bond (one LBFGS inner step) ──
                # Create temporary LBFGS with max_iter=1, run one outer step
                # (= one closure call = CTMRG + energy with _CACHE_RHOS), then
                # extract observables from cached rhos. Reuses all existing
                # machinery from optimize_at_chi instead of manually coding CTMRG.
                chi_la = chi + D_bond
                print(f"  │  [lookahead] one fwd step at chi_la={chi_la} (D={D_bond})")
                sys.stdout.flush()
                try:
                    _la_params = [p.detach().clone().requires_grad_(True) for p in cur_params]
                    _core._CACHE_RHOS = True
                    _core._RHO_CACHE.clear()
                    
                    D_sq_la = D_bond ** 2
                    # Closure for lookahead (identical to optimize_at_chi logic)
                    def _la_closure():
                        a_la, b_la, c_la, d_la, e_la, f_la = _derive_abcdef(
                            _la_params, ansatz_cfg, D_bond)
                        aN_la = normalize_single_layer_tensor_for_double_layer(a_la)
                        bN_la = normalize_single_layer_tensor_for_double_layer(b_la)
                        cN_la = normalize_single_layer_tensor_for_double_layer(c_la)
                        dN_la = normalize_single_layer_tensor_for_double_layer(d_la)
                        eN_la = normalize_single_layer_tensor_for_double_layer(e_la)
                        fN_la = normalize_single_layer_tensor_for_double_layer(f_la)
                        A_la, B_la, C_la, Dt_la, E_la, F_la = abcdef_to_ABCDEF(
                            aN_la, bN_la, cN_la, dN_la, eN_la, fN_la, D_sq_la)
                        
                        _proxy_fn = None
                        if _core._CTM_CONV_MODE != 'SVdifference':
                            def _proxy_fn(
                                    C21CD, T1F, T2A,
                                    C21EB, T1D, T2C,
                                    C21AF, T1B, T2E):
                                with torch.no_grad():
                                    _e1 = energy_expectation_nearest_neighbor_3ebadcf_bonds(
                                        aN_la, bN_la, cN_la, dN_la, eN_la, fN_la, *Js[0:12], SdotS,
                                        chi_la, D_bond, d_PHYS,
                                        C21CD, T1F, T2A)
                                    _e2 = energy_expectation_nearest_neighbor_3afcbed_bonds(
                                        aN_la, bN_la, cN_la, dN_la, eN_la, fN_la, *Js[12:24], SdotS,
                                        chi_la, D_bond, d_PHYS,
                                        C21EB, T1D, T2C)
                                    _e3 = energy_expectation_nearest_neighbor_other_3_bonds(
                                        aN_la, bN_la, cN_la, dN_la, eN_la, fN_la, *Js[24:36], SdotS,
                                        chi_la, D_bond, d_PHYS,
                                        C21AF, T1B, T2E)
                                    return (_e1 + _e2 + _e3).item()
 
                        all10_la = CTMRG_from_init_to_stop(
                            A_la, B_la, C_la, Dt_la, E_la, F_la, chi_la, D_sq_la,
                            CTM_MAX_STEPS, CTM_CONV_THR, ENV_IDENTITY_INIT,
                            energy_proxy_fn=_proxy_fn)
                        (C21CD_la, T1F_la, T2A_la,
                         C21EB_la, T1D_la, T2C_la,
                         C21AF_la, T1B_la, T2E_la, _) = all10_la
                        loss_la = _three_env_energy_loss_parallel(
                            aN_la, bN_la, cN_la, dN_la, eN_la, fN_la, Js, SdotS,
                            chi_la, D_bond, d_PHYS,
                            env1=(C21CD_la, T1F_la, T2A_la),
                            env2=(C21EB_la, T1D_la, T2C_la),
                            env3=(C21AF_la, T1B_la, T2E_la))
                        return loss_la
                    
                    # LBFGS with max_iter=1 → one inner step only
                    _la_lbfgs = torch.optim.LBFGS(
                        _la_params, lr=LBFGS_LR, max_iter=1,
                        tolerance_grad=0.0, tolerance_change=0.0,
                        history_size=1, line_search_fn='strong_wolfe')
                    _la_lbfgs.step(_la_closure)  # one outer step = one closure call
                    
                    # Extract observables from cached rhos
                    _la_cached_rhos = (
                        _core._RHO_CACHE.get('env1'),
                        _core._RHO_CACHE.get('env2'),
                        _core._RHO_CACHE.get('env3'))
                    with torch.no_grad():
                        energy_la, correlations_la, magnetizations_la = \
                            _observables_from_rhos(_la_cached_rhos, Js, SdotS, d_PHYS)
                    _save_observables_file(
                        os.path.join(output_dir,
                                     f"D_{D_bond}_chi_{chi}_lookahead_{chi_la}"
                                     f"_energy_magnetization_correlation.txt"),
                        D_bond, chi_la, energy_la, correlations_la,
                        magnetizations_la, None)
                    _print_observables_summary(
                        'LA ', D_bond, chi_la, energy_la, correlations_la,
                        magnetizations_la, None)
                    
                    # ── Chi convergence check ────────────────────────────────
                    # Only check convergence if LBFGS was reached (not during Adam phase).
                    # Adam can never converge — it only triggers the switch to LBFGS.
                    # LBFGS is the one that determines optimization convergence.
                    if CHI_CONVERGENCE_THRESHOLD > 0 and _switched_to_lbfgs:
                        _delta_E = energy - energy_la
                        if abs(_delta_E) <= CHI_CONVERGENCE_THRESHOLD:
                            print(f"  │  [chi-converged] ΔE={_delta_E:.2e} <= "
                                  f"{CHI_CONVERGENCE_THRESHOLD:.2e} → skipping "
                                  f"remaining chi levels, moving to next D")
                            _chi_converged = True
                        else:
                            _chi_converged = False
                            print(f"  │  [chi-not-converged] ΔE={_delta_E:.2e} > "
                                  f"{CHI_CONVERGENCE_THRESHOLD:.2e} → continuing "
                                  f"to next chi level")
                    else:
                        _chi_converged = False
                        if CHI_CONVERGENCE_THRESHOLD > 0 and not _switched_to_lbfgs:
                            print(f"  │  [chi-convergence] skipped: still in Adam phase "
                                  f"(LBFGS not reached yet)")
                    # ─────────────────────────────────────────────────────────
                    
                    del _la_params, _la_lbfgs
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception as _la_exc:
                    print(f"  │  [lookahead] failed: {_la_exc}")
                    _chi_converged = False
                finally:
                    _core._CACHE_RHOS = False
                # ─────────────────────────────────────────────────────────────

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
                
                # Early exit from chi loop if converged
                if _chi_converged:
                    print(f"  [chi-series-complete] D={D_bond} converged at chi={chi}")
                    break

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
