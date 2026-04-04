
#!/usr/bin/env python3



#NOTE:N_cores, validate_chi, TOTAL_BUDGET_HOURS = 9999
# # sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
# _default_outdir = os.path.join('/scratch/chye/1stTrialRun',
#  f'6tensors_{run_ts}')
# sys.stdout.flush()



# ── Sweep control ─────────────────────────────────────────────────────────────

D_BOND_LIST = [2,3, 4, 5, 6, 7, 8, 9, 10, 11]
#   Ordered list of iPEPS virtual bond dimensions to sweep (outer loop).
#   Each D is warm-started from the best tensors found at the previous D
#   (zero-padded to the new size + PAD_NOISE Gaussian noise).


DEFAULT_D_BUDGET_FRACS = {2: 0.1, 3: 0.1, 4: 0.1, 5:0.1, 6:0.1, 7:0.1, 8:0.1, 9:0.1, 10:0.1, 11:0.1}
#   Fraction of the total wall-clock budget allocated to each D_bond value.
#   Normalised to sum=1 before use, so only the RATIOS matter.
#   Rationale:
#     D=2 : small tensors, converges quickly — 3 % is enough.
#     D=3 : main physics workhorse, needs the most time — 52 %.
#     D=4 : highest accuracy but very slow per step — 45 %.
#   Note: This is the ONLY intentional asymmetry in the sweep.  Different D
#   values have genuinely different computational costs and scientific weight.
#   Within each D, every chi level gets equal time (see below).

DEFAULT_CHI_MAX = {2: 16, 3: 81, 4: 80, 5:9999, 6:9999, 7:9999, 8:9999, 9:9999, 10:9999, 11:9999}
#   Largest chi to attempt for each D_bond.  Hard upper bound is D⁴
#   (gives 16, 81, 256 for D=2,3,4).  We cap D=4 at 80 because chi >> 80
#   requires too much RAM on a typical workstation and adds negligible accuracy.
#   Increase if you have more memory; decrease if you hit OOM.

DEFAULT_CHI_SCHEDULES = {
    2: [5, 7, 9],
    3: [10, 13, 16],       # , 57, 81],  ← append to extend the schedule
    4: [17, 21, 25],           # , 40, 62, 80] ← append to extend the schedule
    5: [26, 31, 36],
    6: [37, 43, 49],
    7: [50, 57, 64],
    8: [65, 73, 81],
    9: [82, 91, 100],
    10:[101,111,121],
    11:[122,133,144],
}







"""
main_sweep.py
=============
10-12 h iPEPS benchmark that sweeps over growing D_bond (outer loop) and
growing chi (inner loop) for the AFM Heisenberg model on the 6-site
honeycomb unit cell, using AD-CTMRG.

Structure
---------
  for D_bond in [2, 3, 4]:          ← outer loop
      for chi in chi_schedule(D):    ← inner loop, D²<chi≤D⁴
          optimise until chi-budget exhausted

  Each (D, chi) level is warm-started from the previous one.
  D → D+1 warm-start: best tensors padded to new bond dimension.

Default chi schedules (geometric spacing):
  D=2: [5, 8, 12, 16]          (D⁴ = 16)
  D=3: [10, 18, 32, 57, 81]   (D⁴ = 81)
  D=4: [17, 26, 40, 62, 80]   (D⁴ = 256; capped at 80 ≈ 250 s/step)

Default time budget fractions (of wall-clock total):
  D=2:  3 %    (~20 min for 11 h run)
  D=3: 52 %    (~ 5.7 h — main workhorse)
  D=4: 45 %    (~ 5.0 h — best physical result)

Within each D budget:
  all chi levels receive equal time  (D_budget / num_chi_levels each).
  Every chi level uses the same L-BFGS hyper-parameters, so results are
  directly comparable across chi values.

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
  sweep_energy_vs_chi.pdf        E/bond vs chi for each D
  sweep_energy_vs_D.pdf          E/bond vs D at chi_max (convergence in D)
"""


# ── glibc malloc tuning (MUST be before any heavy imports) ────────────────────
# Without this, freed intermediate tensors stay in glibc arenas and RSS
# grows 10-100× larger than the actual live tensors.  These mallopt calls
# force allocations ≥ 64 KB to use mmap (returned to OS on free) and limit
# arenas to 2 (less fragmentation).
import ctypes as _ctypes
_libc = _ctypes.CDLL(None)
_libc.mallopt(_ctypes.c_int(-3), _ctypes.c_int(65536))   # M_MMAP_THRESHOLD  = 64 KB
_libc.mallopt(_ctypes.c_int(-1), _ctypes.c_int(0))       # M_TRIM_THRESHOLD  = 0 (trim eagerly)
_libc.mallopt(_ctypes.c_int(-8), _ctypes.c_int(2))       # M_ARENA_MAX       = 2
del _libc


import argparse
import collections
import datetime
import gc
import json
import os
import sys
import time

memory_diagn = False

# ── CPU threading ─────────────────────────────────────────────────────────────
# MUST be set BEFORE importing NumPy / PyTorch / MKL — they read these at init.
#
# Hardware: Intel Core i7-8665U — 4 physical cores × 2 HT = 8 logical CPUs.
# PyTorch is built with Intel MKL (BLAS/LAPACK) + OpenMP.
#
# Use 4 = physical core count.  Hyperthreading does NOT help SVD / dense
# matmul: both HTs on the same core share the same FPU and L1/L2 cache, so
# using 8 threads starves each other rather than doubling throughput.
#
# Use os.environ.setdefault so a user can still override from the shell:
#   OMP_NUM_THREADS=2 python scripts/main_sweep_CPU.py   ← respected
_N_PHYSICAL_CORES = 16
os.environ.setdefault("OMP_NUM_THREADS", str(_N_PHYSICAL_CORES))
os.environ.setdefault("MKL_NUM_THREADS", str(_N_PHYSICAL_CORES))
# Prevent MKL from silently reducing thread count when it detects nested
# parallelism or high system load — we always want the full 4 threads.
os.environ.setdefault("MKL_DYNAMIC", "FALSE")
# Pin each OpenMP thread to a distinct physical core (not a hyperthread sibling).
# granularity=fine  → thread-level pinning (finest possible)
# compact           → pack onto as few sockets as possible (single-socket here)
# 1                 → permute: fill one HT per core before doubling up (with
#                     4 threads on 4 cores this has no effect, but future-proofs)
# 0                 → offset: start from logical CPU 0
# Result: threads 0-3 map to cores 0-3, no migration between iterations.
os.environ.setdefault("KMP_AFFINITY", "granularity=fine,compact,1,0")
# After a parallel region the OpenMP workers normally spin-wait for ~200 ms
# before going to sleep, burning a whole core.  Set to 0 so they yield
# immediately — the outer L-BFGS loop is sequential so this saves real time.
os.environ.setdefault("KMP_BLOCKTIME", "0")

# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import opt_einsum as oe
import torch
from torch.utils.checkpoint import checkpoint as _ckpt

# Optional debugging aid: if set, PyTorch will print the forward op trace that
# produced a backward error (e.g. complex SVD phase-gauge issues).
if os.environ.get("CTMRG_ANOMALY", "0") == "1":
    torch.autograd.set_detect_anomaly(True)

# Tell PyTorch's own dispatcher to use all physical cores for intra-op
# parallelism (matrix multiply, SVD, etc.) and only 1 thread for the
# inter-op pool — our outer loop is serial, extra inter-op threads just
# compete with the intra-op pool for the same physical cores.
torch.set_num_threads(_N_PHYSICAL_CORES)
torch.set_num_interop_threads(1)

import core_unrestricted as _core
from core_unrestricted import (
    normalize_tensor,
    normalize_single_layer_tensor_for_double_layer,
    initialize_abcdef,
    abcdef_to_ABCDEF,
    CTMRG_from_init_to_stop,
    build_heisenberg_H,
    #energy_expectation_nearest_neighbor_6_bonds,
    energy_expectation_nearest_neighbor_3ebadcf_bonds,
    energy_expectation_nearest_neighbor_3afcbed_bonds,
    energy_expectation_nearest_neighbor_other_3_bonds,
    set_dtype,
    set_check_truncation,
    clear_trunc_diag_buffer,
    get_trunc_diag_buffer,
)

# ══════════════════════════════════════════════════════════════════════════════
# Time Budget
# ══════════════════════════════════════════════════════════════════════════════

TOTAL_BUDGET_HOURS = 9999

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
#             the infinite-lattice environment is represented.  The physical
#             constraint for our honeycomb ansatz is  D² < chi ≤ D⁴.
#             Too-small chi → environment too compressed → biased energy;
#             too-large chi → CTMRG is slow and memory-hungry.
#             We sweep chi from the smallest valid value up to chi_max,
#             warm-starting each chi level from the previous.  ALL chi levels
#             receive EQUAL time budget and IDENTICAL L-BFGS settings, making
#             results at different chi directly comparable.

# ── L-BFGS optimiser ─────────────────────────────────────────────────────────
#
#   Optimisation strategy: alternating CTMRG + L-BFGS.
#     1. Converge the CTMRG environment at fixed tensors (a..f).  No grad.
#     2. Run one L-BFGS call (up to LBFGS_MAX_ITER sub-iterations) to
#        minimise the energy w.r.t. a..f, keeping environment fixed.
#     3. Repeat until time budget is exhausted or OPT_CONV_THRESHOLD hit.
#   This is the "cheap-environment" AD-CTMRG gradient scheme.

LBFGS_MAX_ITER = 20
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

LBFGS_HISTORY = 160
#   Number of (s, y) curvature vector pairs retained for the L-BFGS inverse-
#   Hessian approximation.  In our alternating-optimisation scheme the LBFGS
#   instance is RECREATED from scratch at every outer step, so curvature pairs
#   accumulate only within a single optimizer.step() call (≤ LBFGS_MAX_ITER
#   sub-iterations).  Any history_size > LBFGS_MAX_ITER allocates buffer
#   memory that is never filled — setting it equal to LBFGS_MAX_ITER is exact
#   and wastes nothing.  Old values like 50–100 were appropriate for classical
#   L-BFGS that runs continuously; they do not apply here.

OPT_TOL_GRAD = 1e-9
#   L-BFGS inner convergence criterion on the infinity-norm of the gradient:
#   the sub-iteration loop exits early if  ||∇loss||_∞ < OPT_TOL_GRAD.
#   This is an inner stopping rule inside a single optimizer.step() call.

OPT_TOL_CHANGE = 3e-9
#   L-BFGS inner convergence criterion on consecutive loss change:
#   sub-iteration exits if  |L_{k+1} – L_k| < OPT_TOL_CHANGE.
#   Set tighter than OPT_TOL_GRAD to catch near-flat regions.

OPT_CONV_THRESHOLD = 1e-8
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






CTM_MAX_STEPS = 50
#   Hard cap on CTMRG iterations per environment convergence call.
#   With the singular-value convergence criterion and CTM_CONV_THR=1e-3,
#   convergence occurs in 4–40 steps for typical tensors (single-tensor
#   ansatz ~4 steps, 6-tensor ~40 steps).  90 is a safe upper bound.

CTM_CONV_THR = 2e-7
#   CTMRG convergence threshold: stop iterating when the max change in
#   normalised corner singular values between consecutive steps is below
#   this value.  The convergence criterion compares the spectra of all 9
#   corner matrices (gauge-invariant), so raw-element Frobenius oscillations
#   from SVD sign ambiguity do NOT affect it.  In float32 the spectral noise
#   floor is ~5e-5–2e-4, so any threshold below ~5e-4 effectively never
#   triggers; 1e-3 converges in 7–20 steps across all tested D/chi configs.

# ── Checkpointing & memory guard ─────────────────────────────────────────────

SAVE_EVERY = 10
#   Frequency (in outer optimisation steps) at which the "latest" checkpoint
#   is written.  The "best" checkpoint is written immediately whenever a new
#   minimum energy is found, independently of SAVE_EVERY.  Lower = more I/O
#   but safer against crashes; higher = less I/O.

RAM_SAFETY_GB = 0.4
#   Minimum free RAM that must remain available before attempting a (D, chi)
#   level.  If free_RAM < peak_estimate + RAM_SAFETY_GB, the level is skipped
#   with a warning.  Peak memory estimate per CTMRG step: 3 corners × 4×
#   SVD workspace × (chi·D²)² × 8 bytes (float32).  Increase if OOM.

# ── Physical model ────────────────────────────────────────────────────────────

J_COUPLING = 1.0
#   Isotropic Heisenberg exchange coupling constant.  J > 0 = antiferromagnetic
#   (AFM), ground state is a singlet.  The Hamiltonian is
#   H = J Σ_{<i,j>} S_i · S_j  summed over all nearest-neighbour pairs on the
#   honeycomb.  For the AFM sign convention the optimal iPEPS energy is
#   E/bond ≈ −0.3630 J in the D→∞ limit (QMC reference).

D_PHYS = 2
#   Physical Hilbert-space dimension per lattice site.
#   d=2 for spin-1/2 (default), d=3 for spin-1, d=4 for two-site, etc.

N_BONDS = 9
#   Number of nearest-neighbour bonds in the 6-site honeycomb unit cell.
#   Breakdown: 6 bonds of the primary type (connecting sub-lattices A–D–C–F–B–E
#   cyclically) + 3 bonds of the secondary type = 9 total.
#   Used only to compute the reported E/bond = E_total / N_BONDS.


# ── Tensor initialisation & padding ──────────────────────────────────────────

INIT_NOISE = 3e-3
# abcdef init noise, NOTE: NOT USED.

PAD_NOISE = 2e-1
# NOTE: NOT USED, noise is divided by sqrt(new_D**3 * d_PHYS) to keep the total noise power per tensor constant as D grows.
#   Gaussian noise amplitude added to the ZERO-PADDED new indices when
#   enlarging tensors from D → D+1.  Non-zero noise breaks the symmetry of
#   the padded zeros and prevents the optimiser from getting stuck in the
#   subspace of the smaller-D manifold.  Keep comparable to INIT_NOISE.

USE_BLOWUP_RECOVERY = False
#   When True: if rSVD gradient blowup is detected AND the outer loss stalls
#   (|Δ| ≤ OPT_CONV_THRESHOLD), recover by re-noising the current tensors and
#   using full deterministic SVD for the next step instead of treating it as
#   genuine convergence.
#   When False: always use partial (randomized) SVD and ignore blowup events.

GEO_SCHEDULE_STEPS = 5
#   Number of chi values generated by the geometric FALLBACK schedule.
#   Only used when --chi-maxes is given on the command line and chi_max
#   differs from DEFAULT_CHI_MAX[D], meaning DEFAULT_CHI_SCHEDULES cannot
#   be used.  Generates GEO_SCHEDULE_STEPS log-uniformly spaced values
#   from D²+1 to chi_max (inclusive).

# ── Reproducibility ──────────────────────────────────────────────────────────

RANDOM_SEED_FIX = True
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


def free_ram_gb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().available / 1e9
    except ImportError:
        pass
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1e6
    except OSError:
        pass
    return float('inf')


def mem_mb() -> float:
    """Return current process Resident Set Size (RSS) in MB.

    Uses psutil when available; falls back to /proc/self/status (Linux).
    Returns NaN when neither is available.
    """
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e6
    except ImportError:
        pass
    try:
        with open("/proc/self/status") as _fh:
            for _line in _fh:
                if _line.startswith("VmRSS:"):
                    return int(_line.split()[1]) / 1e3
    except OSError:
        pass
    return float('nan')


def tensor_mb(t: torch.Tensor) -> float:
    """Return the allocated memory of a single tensor in MB."""
    return t.element_size() * t.nelement() / 1e6


def env_mem_report(env28: tuple, chi: int, D_bond: int) -> str:
    """One-line summary of the 27 CTMRG environment tensor sizes.

    env28 is the 28-tuple from CTMRG_from_init_to_stop (27 tensors + ctm_steps).
    """
    _names = [
        'C21CD', 'C32EF', 'C13AB', 'T1F',  'T2A',  'T2B',  'T3C',  'T3D',  'T1E',
        'C21EB', 'C32AD', 'C13CF', 'T1D',  'T2C',  'T2F',  'T3E',  'T3B',  'T1A',
        'C21AF', 'C32CB', 'C13ED', 'T1B',  'T2E',  'T2D',  'T3A',  'T3F',  'T1C',
    ]
    _tns = env28[:27]
    _total   = sum(tensor_mb(t) for t in _tns)
    _corners = sum(tensor_mb(t) for n, t in zip(_names, _tns) if n.startswith('C'))
    _trans   = sum(tensor_mb(t) for n, t in zip(_names, _tns) if n.startswith('T'))
    _c_shape = tuple(_tns[0].shape)   # C21CD
    _t_shape = tuple(_tns[3].shape)   # T1F
    return (
        f"env chi={chi} D={D_bond}: total={_total:.1f}MB "
        f"(corners={_corners:.1f}MB transfer={_trans:.1f}MB) "
        f" T-shape={_t_shape}"
    )


def abcdef_mem_report(tensors: tuple, D_bond: int) -> str:
    """One-line summary of the 6 site tensors (a..f)."""
    _total = sum(tensor_mb(t) for t in tensors)
    return (
        f"site D={D_bond}: total={_total:.4f}MB "
        f"shape={tuple(tensors[0].shape)} dtype={tensors[0].dtype}"
    )


def peak_ram_gb(chi: int, D_sq: int) -> float:
    """Estimate peak RAM for one CTMRG step in GB."""
    n = chi * D_sq
    return 3 * 4 * n * n * 8 / 1e9   # 3 corners × 4× SVD workspace × element size


def validate_chi(chi: int, D_bond: int, label: str = '') -> None:
    D_sq, D4 = D_bond ** 2, D_bond ** 4
    if False and not (D_sq < chi <= D4):
        raise ValueError(
            f"{label}chi={chi} violates D²={D_sq} < chi ≤ D⁴={D4} "
            f"(D_bond={D_bond})."
        )

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


def pad_tensor(t: torch.Tensor, old_D: int, new_D: int,
               d_PHYS: int, noise: float) -> torch.Tensor:
    # Whenever pad_tensor is called (D→D+1 warm-start OR same-D recovery),
    # the next optimization step should use full (deterministic) SVD to avoid
    # rSVD gradient blowup on the noisy initial point.  The flag is cleared
    # automatically after one L-BFGS step completes.
    _core._USE_FULL_SVD = True

    out = torch.randn(new_D, new_D, new_D, d_PHYS, dtype=TENSORDTYPE)/torch.sqrt(torch.tensor(new_D**3 * d_PHYS, dtype=TENSORDTYPE))
    out[:old_D, :old_D, :old_D, :] += normalize_tensor(t.detach())*torch.sqrt(torch.tensor(old_D**3 * d_PHYS, dtype=TENSORDTYPE))
    
    return out


def evaluate_energy_clean(a, b, c, d, e, f,
                          Hs, chi: int, D_bond: int, d_PHYS: int) -> float:
    """Re-converge environment from scratch and return total energy (float).

    IMPORTANT: CTMRG consumes *double-layer* site tensors which are normalised
    inside ``abcdef_to_ABCDEF``. For consistency, we rescale the single-layer
    tensors here in the same convention used inside the optimiser objective.
    This keeps printed denominators like <iPEPS|iPEPS> close to real 1.
    """
    D_sq = D_bond ** 2
    with torch.no_grad():
        aN = normalize_single_layer_tensor_for_double_layer(a)
        bN = normalize_single_layer_tensor_for_double_layer(b)
        cN = normalize_single_layer_tensor_for_double_layer(c)
        dN = normalize_single_layer_tensor_for_double_layer(d)
        eN = normalize_single_layer_tensor_for_double_layer(e)
        fN = normalize_single_layer_tensor_for_double_layer(f)

        A, B, C, Dt, E, F = abcdef_to_ABCDEF(aN, bN, cN, dN, eN, fN, D_sq)
        all27 = CTMRG_from_init_to_stop(
                A, B, C, Dt, E, F, chi, D_sq, CTM_MAX_STEPS, CTM_CONV_THR, ENV_IDENTITY_INIT)
        #E6 = energy_expectation_nearest_neighbor_6_bonds(
        #    a, b, c, d, e, f,
        #    Hs[0], Hs[1], Hs[2], Hs[3], Hs[4], Hs[5],
        #    chi, D_bond, *all27[:9])
        E3ebadcf = energy_expectation_nearest_neighbor_3ebadcf_bonds(
            aN, bN, cN, dN, eN, fN,
                Hs[0],Hs[1],Hs[2],
                chi, D_bond, d_PHYS, 
                *all27[:9])
        E3afcbed = energy_expectation_nearest_neighbor_3afcbed_bonds(
            aN, bN, cN, dN, eN, fN,
                Hs[3],Hs[4],Hs[5], 
                chi, D_bond, d_PHYS, 
                *all27[9:18])
            
        E3 = energy_expectation_nearest_neighbor_other_3_bonds(
            aN, bN, cN, dN, eN, fN,
            Hs[6], Hs[7], Hs[8],
            chi, D_bond, d_PHYS, 
            *all27[18:27])
        return (E3ebadcf + E3afcbed + E3).item()


def _plot_truncation_diagnostics(
        diag_buffer: list,
        D_bond: int,
        chi: int,
        out_dir: str,
) -> None:
    """Generate and save truncation-diagnostics figure.

    Two tracks:
      1. Anti-Hermitian measure  ||ρ − ρ†||/2/||ρ||  for each of the three ρ
         matrices, plotted vs. ``trunc_rhoCCC`` call index (semilogy).
      2. Normalised SV spectra of the three ρ matrices from the *last* recorded
         call, with a dashed vertical line at the χ truncation cutoff.
    """
    if not diag_buffer:
        return
    import numpy as np

    calls = list(range(len(diag_buffer)))
    ah32 = [d['anti_herm_rho32'] for d in diag_buffer]
    ah13 = [d['anti_herm_rho13'] for d in diag_buffer]
    ah21 = [d['anti_herm_rho21'] for d in diag_buffer]

    last  = diag_buffer[-1]
    chi_cut = last['chi']
    sv32 = np.array(last['sv32'], dtype=float)
    sv13 = np.array(last['sv13'], dtype=float)
    sv21 = np.array(last['sv21'], dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(10, 9), constrained_layout=True)

    # ── Track 1: anti-Hermitian measures ─────────────────────────────────────
    ax1 = axes[0]
    ax1.semilogy(calls, ah32, lw=1.2, label=r'$\rho_{32}$ = C13·C32·C21')
    ax1.semilogy(calls, ah13, lw=1.2, label=r'$\rho_{13}$ = C21·C13·C32')
    ax1.semilogy(calls, ah21, lw=1.2, label=r'$\rho_{21}$ = C32·C21·C13')
    ax1.set_xlabel('trunc_rhoCCC call index')
    ax1.set_ylabel(r'$\|\rho - \rho^\dagger\|_F \;/\; 2 \|\rho\|_F$')
    ax1.set_title(f'Anti-Hermitian measure of ρ  (D={D_bond}, χ={chi})')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, which='both')

    # ── Track 2: SV spectra at last call ─────────────────────────────────────
    ax2 = axes[1]
    def _norm(sv: np.ndarray) -> np.ndarray:
        return sv / sv[0] if sv[0] != 0 else sv

    idx = lambda sv: np.arange(1, len(sv) + 1)
    ax2.semilogy(idx(sv32), _norm(sv32), lw=1.2, label=r'$\sigma(\rho_{32})$')
    ax2.semilogy(idx(sv13), _norm(sv13), lw=1.2, label=r'$\sigma(\rho_{13})$')
    ax2.semilogy(idx(sv21), _norm(sv21), lw=1.2, label=r'$\sigma(\rho_{21})$')
    ax2.axvline(
        chi_cut + 0.5, color='red', linestyle='--', linewidth=1.5,
        label=f'χ cutoff  (χ={chi_cut})',
    )
    ax2.set_xlabel('singular-value index  k')
    ax2.set_ylabel(r'$\sigma_k \;/\; \sigma_1$  (normalised)')
    ax2.set_title(f'SV spectrum at last trunc call  (D={D_bond}, χ={chi})')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, which='both')

    out_path = os.path.join(out_dir, f'trunc_diagnostics_D{D_bond}_chi{chi}.pdf')
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Truncation diagnostics → {out_path}")


def save_checkpoint(path: str, abcdef: tuple, D_bond: int, chi: int,
                    loss: float, energy: float | None,
                    step: int, log: list) -> None:
    torch.save({
        'a': abcdef[0], 'b': abcdef[1], 'c': abcdef[2],
        'd': abcdef[3], 'e': abcdef[4], 'f': abcdef[5],
        'D_bond': D_bond, 'chi': chi,
        'loss': loss, 'energy': energy,
        'step': step, 'timestamp': timestamp(),
        'log': log,
    }, path)


# ══════════════════════════════════════════════════════════════════════════════
# Core: optimise at a single (D_bond, chi) level for a fixed time budget
# ══════════════════════════════════════════════════════════════════════════════

def optimize_at_chi(
        Hs, D_bond: int, chi: int, d_PHYS: int,
        budget_seconds: float,
        lbfgs_max_iter: int,
        init_abcdef=None,
        step_offset: int = 0,
        best_path: str | None = None,
        latest_path: str | None = None,
        loss_log: list | None = None,
        out_dir: str | None = None,
) -> tuple:
    """
    Outer L-BFGS loop at fixed (D_bond, chi) until budget_seconds elapsed
    or opt_conv_threshold hit.

    Returns (a, b, c, d, e, f, best_loss, steps_done).
    """
    if loss_log is None:
        loss_log = []
    D_sq = D_bond ** 2
    t_start = time.perf_counter()

    # initialise tensors — ALWAYS brand-new objects, no shared lineage
    if init_abcdef is not None:
        a, b, c, d, e, f = _new_tensors_from_data(init_abcdef)
    else:
        a, b, c, d, e, f = initialize_abcdef('random', D_bond, d_PHYS, INIT_NOISE)
    for t in (a, b, c, d, e, f):
        t.requires_grad_(True)

    # ── MEM-A: site-tensor footprint before any CTMRG allocation ─────────────
    # BREAKPOINT-A  ← set breakpoint on the print line below.
    # Debug Console queries when paused here:
    #   tensor_mb(a)                           → MB for one site tensor
    #   abcdef_mem_report((a,b,c,d,e,f), D_bond)  → one-liner summary
    #   mem_mb()                               → total process RSS
    if memory_diagn: print(f"[MEM-A] RSS={mem_mb():.1f}MB, {abcdef_mem_report((a, b, c, d, e, f), D_bond)}")

    best_loss     = float('inf')
    best_abcdef   = tuple(t.detach().clone() for t in (a, b, c, d, e, f))
    prev_loss     = None
    loss_history  = collections.deque(maxlen=12)  # cycle detection up to length 12
    step          = step_offset

    # ── pre-create Adam optimizer (state persists across outer steps) ─────────
    _adam: torch.optim.Optimizer | None = None
    if OPTIMIZER == 'adam':
        _adam = torch.optim.Adam(
            [a, b, c, d, e, f],
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
        def _make_lbfgs(*params) -> torch.optim.LBFGS:
            return torch.optim.LBFGS(
                list(params),
                lr=LBFGS_LR,
                max_iter=lbfgs_max_iter,
                tolerance_grad=OPT_TOL_GRAD,
                tolerance_change=OPT_TOL_CHANGE,
                history_size=LBFGS_HISTORY,
                line_search_fn='strong_wolfe',
            )
        _lbfgs = _make_lbfgs(a, b, c, d, e, f)

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

    def _loss_with_differentiable_ctmrg() -> tuple[torch.Tensor, int, dict[str, float]]:
        """Compute loss with CTMRG inside the autograd graph.

        LBFGS calls its closure multiple times per step; therefore this
        function must be called fresh on every closure evaluation.
        Never reuse environment tensors across calls.
        """
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
        all28 = CTMRG_from_init_to_stop(
            A, B, C, Dt, E, F, chi, D_sq,
            CTM_MAX_STEPS, CTM_CONV_THR, ENV_IDENTITY_INIT)

        # ── MEM-B: env tensors allocated by CTMRG forward pass ───────────────
        # BREAKPOINT-B  ← set breakpoint on the print line below.
        # This is the PEAK env allocation point (27 tensors live + autograd graph).
        # Debug Console queries when paused here:
        #   env_mem_report(all28, chi, D_bond)     → detailed per-category sizes
        #   mem_mb()                               → total RSS (includes graph)
        #   tensor_mb(all28[3])                    → size of T1F (transfer tensor)
        #   all28[0].shape                         → C21CD corner shape
        #   sum(t.element_size()*t.nelement() for t in all28[:27]) / 1e6
        if memory_diagn: print(f"[MEM-B] RSS={mem_mb():.1f}MB, {env_mem_report(all28, chi, D_bond)}")

        (C21CD, C32EF, C13AB, T1F,  T2A,  T2B,  T3C,  T3D,  T1E,
         C21EB, C32AD, C13CF, T1D,  T2C,  T2F,  T3E,  T3B,  T1A,
         C21AF, C32CB, C13ED, T1B,  T2E,  T2D,  T3A,  T3F,  T1C,
         ctm_steps) = all28

        # ── Checkpoint the three energy-function calls. ────────────────────
        # Why: each energy_expectation call creates large intermediate tensors
        # (6 open_* tensors ≈ 4.4 MB each, 3 rho tensors ≈ 17 MB each at
        # chi=29, D=3) that would all be retained in the autograd graph for
        # the backward pass.  At D=3, chi=29 that is ~88 MB per function, or
        # ~265 MB total — the dominant source of backward peak memory.
        # These functions are purely deterministic (no randomness), so the
        # checkpoint recompute is bit-identical to the original forward.
        loss = (
            _ckpt(energy_expectation_nearest_neighbor_3ebadcf_bonds,
                  aN, bN, cN, dN, eN, fN,
                  Hs[0], Hs[1], Hs[2],
                  chi, D_bond, d_PHYS,
                  C21CD, C32EF, C13AB, T1F, T2A, T2B, T3C, T3D, T1E,
                  use_reentrant=False)
            +
            _ckpt(energy_expectation_nearest_neighbor_3afcbed_bonds,
                  aN, bN, cN, dN, eN, fN,
                  Hs[3], Hs[4], Hs[5],
                  chi, D_bond, d_PHYS,
                  C21EB, C32AD, C13CF, T1D, T2C, T2F, T3E, T3B, T1A,
                  use_reentrant=False)
            +
            _ckpt(energy_expectation_nearest_neighbor_other_3_bonds,
                  aN, bN, cN, dN, eN, fN,
                  Hs[6], Hs[7], Hs[8],
                  chi, D_bond, d_PHYS,
                  C21AF, C32CB, C13ED, T1B, T2E, T2D, T3A, T3F, T1C,
                  use_reentrant=False)
        )

        # Physical corner normalisation diagnostic: Tr(rho)=Tr(C13·C32·C21)
        tr_rho_1 = torch.trace(C13AB @ C32EF @ C21CD)
        tr_rho_2 = torch.trace(C13CF @ C32AD @ C21EB)
        tr_rho_3 = torch.trace(C13ED @ C32CB @ C21AF)

        corner_norms = {
            'C21CD': float(torch.linalg.norm(C21CD.detach()).item()),
            'C32EF': float(torch.linalg.norm(C32EF.detach()).item()),
            'C13AB': float(torch.linalg.norm(C13AB.detach()).item()),
            'C21EB': float(torch.linalg.norm(C21EB.detach()).item()),
            'C32AD': float(torch.linalg.norm(C32AD.detach()).item()),
            'C13CF': float(torch.linalg.norm(C13CF.detach()).item()),
            'C21AF': float(torch.linalg.norm(C21AF.detach()).item()),
            'C32CB': float(torch.linalg.norm(C32CB.detach()).item()),
            'C13ED': float(torch.linalg.norm(C13ED.detach()).item()),
            'Tr_rho_1_re': float(tr_rho_1.real.detach().item()),
            #'Tr_rho_1_im': float(tr_rho_1.imag.detach().item()),
            'Tr_rho_2_re': float(tr_rho_2.real.detach().item()),
            #'Tr_rho_2_im': float(tr_rho_2.imag.detach().item()),
            'Tr_rho_3_re': float(tr_rho_3.real.detach().item()),
            #'Tr_rho_3_im': float(tr_rho_3.imag.detach().item()),
        }
        return loss, int(ctm_steps), corner_norms

    while True:
        elapsed = time.perf_counter() - t_start
        if elapsed >= budget_seconds:
            break

        # ── MEM-C: RSS at the start of each outer optimization step ──────────
        # BREAKPOINT-C  ← set breakpoint on the print line below.
        # Watch for monotonic RSS growth → indicates graph/tensor leak.
        # Debug Console queries when paused here:
        #   mem_mb()           → current RSS
        #   free_ram_gb()      → free system RAM (cluster OOM risk if < 1 GB)
        #   step               → current step number
        if True and memory_diagn: 
            print(f"[MEM-C] RSS={mem_mb():.1f}MB, step={step} free={free_ram_gb()*1e3:.0f}MB")
            sys.stdout.flush()  # ensure the print appears before a potential OOM crash

        # # normalise (scale-redundancy fix, preserves requires_grad)
        # with torch.no_grad():
        #     a.data = normalize_tensor(a.data)
        #     b.data = normalize_tensor(b.data)
        #     c.data = normalize_tensor(c.data)
        #     d.data = normalize_tensor(d.data)
        #     e.data = normalize_tensor(e.data)
        #     f.data = normalize_tensor(f.data)

        # CTMRG is evaluated inside the optimizer objective.
        # (LBFGS calls the closure multiple times; reusing env tensors would
        #  both crash autograd and give incorrect gradients.)
        ctm_steps = -1
        cn: dict[str, float] = {}
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
            last_cn = None
            def closure():
                nonlocal last_ctm_steps, last_cn
                _lbfgs.zero_grad()
                try:
                    loss, _ctm_steps, _cn = _loss_with_differentiable_ctmrg()
                    # Guard against NaN/Inf losses which break strong-wolfe.
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite loss: {loss}")
                    last_ctm_steps = _ctm_steps
                    last_cn = _cn
                    loss.backward()
                    gc.collect()  # free cyclic refs in backward graph immediately
                    #print("gradient example:", a.grad.view(-1)[:5])
                    # ── MEM-D: RSS after backward (peak: forward graph + grad graph) ──
                    # BREAKPOINT-D  ← set breakpoint on the print line below.
                    # After this point the autograd graph is released when `loss`
                    # goes out of scope.  The delta MEM-D minus MEM-B tells you
                    # how large the backward computation graph was.
                    # Debug Console queries when paused here:
                    #   mem_mb()                       → RSS at backward peak
                    #   a.grad.shape, tensor_mb(a.grad) → gradient tensor sizes
                    if memory_diagn: print(f"[MEM-D] RSS={mem_mb():.1f}MB after backward")
                    # Guard against NaN/Inf gradients.
                    for p in (a, b, c, d, e, f):
                        if p.grad is not None:
                            p.grad.data = torch.nan_to_num(p.grad.data, nan=0.0, posinf=0.0, neginf=0.0)
                    return loss
                except Exception as exc:
                    # BRUTAL ENFORCEMENT: If CTMRG/SVD forward or backward is
                    # ill-defined at this trial point (common during L-BFGS
                    # line search), return a large penalty with zero gradient
                    # so the line search rejects the step instead of aborting.
                    last_ctm_steps = 0
                    last_cn = {"closure_error": float('nan')}
                    for p in (a, b, c, d, e, f):
                        if p.grad is None:
                            p.grad = torch.zeros_like(p)
                        else:
                            p.grad.detach_()
                            p.grad.zero_()
                    # Print a short one-line warning (kept minimal to avoid log spam).
                    msg = str(exc).splitlines()[0][:200]
                    print(f"[closure] Non-finite/ill-defined point -> penalty loss (reason: {msg})")
                    return torch.tensor(1.0e6, dtype=a.real.dtype, device=a.device)
            loss_val  = _lbfgs.step(closure)

            # After a successful step with full SVD, revert to partial for all
            # subsequent steps (the noisy initial basin has been escaped).
            if _core._USE_FULL_SVD:
                _core._USE_FULL_SVD = False

            loss_item = loss_val.item()
            if last_ctm_steps is not None:
                ctm_steps = last_ctm_steps
            if last_cn is not None:
                cn = last_cn
        else:  # adam — persistent state, ADAM_STEPS_PER_CTM micro-steps per env refresh
            if _adam is None:
                raise RuntimeError("Adam optimizer requested but was not initialized")
            for _s in range(ADAM_STEPS_PER_CTM):
                _adam.zero_grad()
                _loss, ctm_steps, cn = _loss_with_differentiable_ctmrg()
                # print(_loss.item(), ctm_steps, cn)
                _loss.backward()
                _adam.step()
            loss_item = _loss.detach().item()
        delta     = (loss_item - prev_loss) if prev_loss is not None else float('inf')
        elapsed   = time.perf_counter() - t_start

        # ──────────────────────────────────────────────────────────────────────
        # DIAGNOSTIC CODE: Enhanced detection and logging of loss values
        # NOTE: Per-component E1/E2/E3 re-computation is omitted here because
        # the environment (C21*/T*) is stale after L-BFGS updates a..f, giving
        # wildly wrong values.  loss_item IS the correct energy over all 9 bonds.
        # ──────────────────────────────────────────────────────────────────────
        
        # Physical bounds and diagnostic levels (based on loss_item only)
        N_BONDS = 9
        PHYSICAL_MIN_PER_BOND = -0.75
        PHYSICAL_MAX_PER_BOND = +0.25
        
        loss_per_bond = loss_item / N_BONDS
        
        # Multiple diagnostic levels
        is_extreme = (loss_per_bond < PHYSICAL_MIN_PER_BOND - 0.05 or 
                      loss_per_bond > PHYSICAL_MAX_PER_BOND + 0.05)
        is_suspicious = (loss_per_bond < PHYSICAL_MIN_PER_BOND + 0.1 or 
                        loss_per_bond > PHYSICAL_MAX_PER_BOND - 0.1)
        
        # Track diagnostic statistics
        if not hasattr(optimize_at_chi, '_diag_stats'):
            optimize_at_chi._diag_stats = {
                'total_steps': 0, 'extreme_count': 0, 'suspicious_count': 0,
                'min_loss_per_bond': float('inf'), 'max_loss_per_bond': float('-inf'),
                'extreme_examples': []
            }
        
        stats = optimize_at_chi._diag_stats
        stats['total_steps'] += 1
        stats['min_loss_per_bond'] = min(stats['min_loss_per_bond'], loss_per_bond)
        stats['max_loss_per_bond'] = max(stats['max_loss_per_bond'], loss_per_bond)
        
        if is_extreme:
            stats['extreme_count'] += 1
            stats['extreme_examples'].append((step, loss_per_bond))
        if is_suspicious:
            stats['suspicious_count'] += 1
        
        # Report extreme cases
        if is_extreme:
            print("\n" + "="*70)
            print(f"⚠️  EXTREME LOSS DETECTED AT STEP {step}")
            print("="*70)
            print(f"Loss: {loss_item:+.6f} (per bond: {loss_per_bond:+.6f})")
            print(f"Physical bounds: [{PHYSICAL_MIN_PER_BOND:.2f}, {PHYSICAL_MAX_PER_BOND:.2f}] per bond")
            
            print(f"\nTensor statistics:")
            tensor_norms = {
                'a': torch.norm(a).item(),
                'b': torch.norm(b).item(),
                'c': torch.norm(c).item(),
                'd': torch.norm(d).item(),
                'e': torch.norm(e).item(),
                'f': torch.norm(f).item()
            }
            for name, norm in tensor_norms.items():
                print(f"  ||{name}|| = {norm:.6f}")
            
            has_nan = any(torch.isnan(t).any().item() for t in [a,b,c,d,e,f])
            has_inf = any(torch.isinf(t).any().item() for t in [a,b,c,d,e,f])
            print(f"  Contains NaN: {has_nan}")
            print(f"  Contains Inf: {has_inf}")
            
            print(f"\nOptimization state:")
            print(f"  D_bond={D_bond}, chi={chi}")
            print(f"  CTMRG steps: {ctm_steps} (max={CTM_MAX_STEPS})")
            if ctm_steps >= CTM_MAX_STEPS:
                print(f"  ⚠️  CTMRG DID NOT CONVERGE")
            print(f"  Optimizer: {OPTIMIZER}")
            print("="*70 + "\n")
            
            # Save diagnostic data
            diag_dir = os.path.join(os.path.dirname(__file__), "..", "diagnostics")
            os.makedirs(diag_dir, exist_ok=True)
            diag_file = os.path.join(diag_dir, f"extreme_loss_step_{step}.json")
            
            diag_data = {
                'step': step,
                'D_bond': D_bond,
                'chi': chi,
                'loss': loss_item,
                'loss_per_bond': loss_per_bond,
                'ctm_steps': ctm_steps,
                'ctm_converged': ctm_steps < CTM_MAX_STEPS,
                'tensor_norms': tensor_norms,
                'has_nan': has_nan,
                'has_inf': has_inf,
                'optimizer': OPTIMIZER,
                'is_extreme': is_extreme
            }
            
            with open(diag_file, 'w') as f_diag:
                json.dump(diag_data, f_diag, indent=2)
            
            print(f"Diagnostic data saved to: {diag_file}")
            torch.save({
                'a': a.detach().cpu(),
                'b': b.detach().cpu(),
                'c': c.detach().cpu(),
                'd': d.detach().cpu(), 
                'e': e.detach().cpu(),
                'f': f.detach().cpu(),
            }, os.path.join(diag_dir, f"tensors_step_{step}.pt"))
        
        # Print periodic progress summary with more detail  
        if is_suspicious: # or step % 10 == 0:
            print(f"    step {step:5d}  ctm={ctm_steps:3d}  loss={loss_item:+.10f}"
                  f"  (per bond: {loss_per_bond:+.6f})"
                  f"  Δ={delta:+.3e}  {elapsed:.0f}/{budget_seconds:.0f}s")
            # ── Corner norm diagnostic (from the last objective evaluation) ──
            if cn:
                print(f"           corner norms │"
                      f" C21CD={cn['C21CD']:.3e} C32EF={cn['C32EF']:.3e} C13AB={cn['C13AB']:.3e}"
                      f" │ C21EB={cn['C21EB']:.3e} C32AD={cn['C32AD']:.3e} C13CF={cn['C13CF']:.3e}"
                      f" │ C21AF={cn['C21AF']:.3e} C32CB={cn['C32CB']:.3e} C13ED={cn['C13ED']:.3e}")
                if 'Tr_rho_1_re' in cn:
                    print(f"           Tr(rho)=Tr(C13·C32·C21)"
                          f" │ type1={cn['Tr_rho_1_re']:.6f}+i*{cn['Tr_rho_1_im']:.2e}"
                          f" │ type2={cn['Tr_rho_2_re']:.6f}+i*{cn['Tr_rho_2_im']:.2e}"
                          f" │ type3={cn['Tr_rho_3_re']:.6f}+i*{cn['Tr_rho_3_im']:.2e}")
        else:
            print(f"    step {step:5d}  ctm={ctm_steps:3d}  loss={loss_item:+.10f}"
                  f"  Δ={delta:+.3e}  {elapsed:.0f}/{budget_seconds:.0f}s")
            sys.stdout.flush()
        
        # ────────────────────────────────────────────────────────────────────────
        # END DIAGNOSTIC CODE
        # ────────────────────────────────────────────────────────────────────────

        # Note: Progress printing is now handled in diagnostic section above
        loss_log.append({'step': step, 'ctm_steps': ctm_steps, 'loss': loss_item,
                         'D_bond': D_bond, 'chi': chi,
                         'elapsed': round(elapsed, 1)})

        if loss_item < best_loss:
            best_loss   = loss_item
            best_abcdef = tuple(t.detach().clone() for t in (a, b, c, d, e, f))
            if best_path:
                save_checkpoint(best_path, best_abcdef, D_bond, chi,
                                best_loss, None, step, loss_log)

        if latest_path and (step - step_offset) % SAVE_EVERY == 0 and step > step_offset:
            save_checkpoint(latest_path, best_abcdef, D_bond, chi,
                            best_loss, None, step, loss_log)

        _blowup, _blowup_max = _core._SVD_GRAD_BLOWUP_DETECTED, _core._SVD_GRAD_BLOWUP_MAX_VALUE
        _core._SVD_GRAD_BLOWUP_DETECTED, _core._SVD_GRAD_BLOWUP_MAX_VALUE = False, 0.0
        if prev_loss is not None and abs(delta) < OPT_CONV_THRESHOLD:
            if USE_BLOWUP_RECOVERY and _blowup:
                print(f"[SVD] rSVD grad blowup (max|dA|={_blowup_max:.3e}) + "
                      f"\u0394={delta:.2e} \u2264 OPT_CONV_THRESHOLD "
                      f"\u2192 recovering from latest tensors + noise (D={D_bond})")
                padded = tuple(
                    pad_tensor(t.detach(), D_bond, D_bond, d_PHYS, PAD_NOISE)
                    for t in (a, b, c, d, e, f))
                a, b, c, d, e, f = _new_tensors_from_data(padded)
                del padded
                for t in (a, b, c, d, e, f):
                    t.requires_grad_(True)
                _lbfgs = _make_lbfgs(a, b, c, d, e, f)
                prev_loss = None
                step += 1
                continue
            print(f"    Outer convergence at step {step} (\u0394={delta:.2e})")
            break
        if any(abs(loss_item - h) < 1e-10 for h in loss_history):
            print(f"    Cycle detected at step {step} "
                  f"(amplitude={abs(delta):.3e}); stopping.")
            break
        loss_history.append(loss_item)
        prev_loss = loss_item
        step += 1

    # Print diagnostic summary
    if hasattr(optimize_at_chi, '_diag_stats'):
        stats = optimize_at_chi._diag_stats
        print(f"\n🔍 DIAGNOSTIC SUMMARY (D={D_bond}, chi={chi}):")
        print(f"   Total steps: {stats['total_steps']}")
        print(f"   Loss range per bond: [{stats['min_loss_per_bond']:+.6f}, {stats['max_loss_per_bond']:+.6f}]")
        print(f"   Extreme cases: {stats['extreme_count']} ({100*stats['extreme_count']/stats['total_steps']:.1f}%)")
        print(f"   Suspicious cases: {stats['suspicious_count']} ({100*stats['suspicious_count']/stats['total_steps']:.1f}%)")
        if stats['extreme_examples']:
            print(f"   Examples of extreme losses (step, total_per_bond):")
            for example in stats['extreme_examples'][:3]:  # Show first 3 examples
                step_ex, total_ex = example
                print(f"     Step {step_ex}: {total_ex:+.6f}")
        print()

    # ── Truncation diagnostics figure ────────────────────────────────────────
    _trunc_buf = get_trunc_diag_buffer()
    if _trunc_buf and out_dir:
        _plot_truncation_diagnostics(_trunc_buf, D_bond, chi, out_dir)
    clear_trunc_diag_buffer()

    return (*best_abcdef, best_loss, step)


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
        help='Comma-separated chi_max for each D_bond (must satisfy D²<chi≤D⁴). '
             'Defaults: 16,81,80 for D=2,3,4.')
    parser.add_argument(
        '--d-phys', type=int, default=D_PHYS,
        help='Physical Hilbert-space dimension (d=2 for spin-1/2).')
    parser.add_argument(
        '--J', type=float, default=J_COUPLING,
        help='Isotropic Heisenberg coupling J (positive = AFM).')
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
        '--double', action='store_true', default=USE_DOUBLE_PRECISION,
        help='Use float64/complex128 (default: float32/complex64). '
             'Same throughput on CPU/MKL; 2–4× slower on CUDA consumer GPUs.')
    parser.add_argument(
        '--complex', action='store_true', default=not USE_REAL_TENSORS,
        help='Use complex iPEPS tensors (Sx/Sy/Sz Hamiltonian). '
             'Default: real tensors (S+/S- Hamiltonian).')
    parser.add_argument(
        '--optimizer', choices=['lbfgs', 'adam'], default=OPTIMIZER,
        help="Optimiser: 'lbfgs' (default, fast on smooth landscapes) or "
             "'adam' (more robust on noisy landscapes, constant LR).")
    parser.add_argument(
        '--check-truncation', action='store_true', default=False,
        help='Enable truncation diagnostics: buffer anti-Hermitian measures and '
             'SV spectra of the ρ matrices on every trunc_rhoCCC call, and save '
             'a two-panel figure (anti-Hermitian track + SV spectrum with χ cutoff) '
             'at the end of each (D, χ) optimisation level.')
    parser.add_argument(
        '--seed', type=int, default=RANDOM_SEED,
        help='Integer RNG seed (used when --no-seed is NOT given). '
             'Seeds Python random, NumPy, and PyTorch CPU+CUDA.')
    parser.add_argument(
        '--no-seed', action='store_true', default=not RANDOM_SEED_FIX,
        help='Disable RNG seeding — runs will NOT be reproducible.')
    args = parser.parse_args()

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
    # Only *enable* the flag from the CLI; never let the argparse default (False)
    # override a True that was already set at module level in core_unrestricted.py.
    if args.check_truncation:
        set_check_truncation(True)

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
        chi_max_map = {D: DEFAULT_CHI_MAX.get(D, D**4) for D in D_bond_list}
    for D in D_bond_list:
        validate_chi(chi_max_map[D], D, label=f'D={D} ')

    # ── chi schedules ─────────────────────────────────────────────────────────
    # Use defaults for known D values, otherwise compute geometrically.
    def chi_schedule(D: int, chi_max: int) -> list[int]:
        if chi_max_map.get(D) == DEFAULT_CHI_MAX.get(D) and D in DEFAULT_CHI_SCHEDULES:
            # Use the explicit schedule defined in DEFAULT_CHI_SCHEDULES as-is.
            # Commented-out entries stay commented out — nothing is auto-appended.
            sched = list(DEFAULT_CHI_SCHEDULES[D])
        else:
            # Geometric fallback: GEO_SCHEDULE_STEPS log-uniform values
            # from D²+1 to chi_max.  Only reached when --chi-maxes is given
            # with a value different from DEFAULT_CHI_MAX[D].
            import math
            D_sq = D ** 2
            chi_min = D_sq + 1
            if chi_min >= chi_max:
                return [chi_max]
            n = GEO_SCHEDULE_STEPS
            ratio = (chi_max / chi_min) ** (1.0 / (n - 1))
            sched = sorted(set(
                [chi_min]
                + [round(chi_min * ratio**k) for k in range(1, n)]
                + [chi_max]))
            sched = [c for c in sched if D_sq < c <= chi_max]
        # Clip to valid range D² < chi ≤ chi_max (catches typos in schedule).
        # NOTE: chi_max itself is NOT auto-appended — the schedule is used
        # exactly as written in DEFAULT_CHI_SCHEDULES.
        sched = [c for c in sched if D**2 < c <= chi_max]
        if not sched:
            raise ValueError(f"Empty chi schedule for D={D} after clipping to ({D**2}, {chi_max}].")
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
    _default_outdir = os.path.join('/scratch/chye/1stTrialRun',
                                   f'6tensors_{run_ts}')
    output_dir = args.output_dir or _default_outdir
    os.makedirs(output_dir, exist_ok=True)

    # ── save hyperparameters to YAML (JSON fallback if PyYAML not installed) ──
    _hp = dict(
        # ── run identity ───────────────────────────────────────────────────
        ansatz             = '6tensors',
        run_timestamp      = run_ts,
        output_dir         = output_dir,

        # ── reproducibility ────────────────────────────────────────────────
        random_seed_fix    = _seed_enabled,
        random_seed        = args.seed if _seed_enabled else None,

        # ── precision ──────────────────────────────────────────────────────
        use_double         = args.double,
        use_real_tensors   = use_real,
        tensordtype        = str(TENSORDTYPE),

        # ── time budget ────────────────────────────────────────────────────
        hours              = args.hours,
        D_bond_list        = D_bond_list,
        default_d_budget_fracs = {str(k): v for k, v in DEFAULT_D_BUDGET_FRACS.items()},
        d_budgets_seconds  = {str(k): round(v, 2) for k, v in d_budgets.items()},
        chi_max_map        = {str(k): v for k, v in chi_max_map.items()},
        schedules          = {str(k): v for k, v in schedules.items()},
        geo_schedule_steps = GEO_SCHEDULE_STEPS,

        # ── physical model ─────────────────────────────────────────────────
        J                  = args.J,
        d_phys             = d_PHYS,
        n_bonds            = N_BONDS,

        # ── optimiser ──────────────────────────────────────────────────────
        optimizer          = OPTIMIZER,
        # L-BFGS
        lbfgs_lr           = LBFGS_LR,
        lbfgs_max_iter     = LBFGS_MAX_ITER,
        lbfgs_history      = LBFGS_HISTORY,
        opt_tol_grad       = OPT_TOL_GRAD,
        opt_tol_change     = OPT_TOL_CHANGE,
        opt_conv_threshold = OPT_CONV_THRESHOLD,
        # Adam
        adam_lr            = ADAM_LR,
        adam_betas         = list(ADAM_BETAS),
        adam_eps           = ADAM_EPS,
        adam_weight_decay  = ADAM_WEIGHT_DECAY,
        adam_steps_per_ctm = ADAM_STEPS_PER_CTM,

        # ── CTMRG ──────────────────────────────────────────────────────────
        ctm_max_steps      = CTM_MAX_STEPS,
        ctm_conv_thr       = CTM_CONV_THR,
        env_identity_init  = ENV_IDENTITY_INIT,

        # ── tensor init & padding ──────────────────────────────────────────
        init_noise         = INIT_NOISE,
        pad_noise          = args.noise,
        use_blowup_recovery= USE_BLOWUP_RECOVERY,

        # ── I/O & memory ───────────────────────────────────────────────────
        save_every         = SAVE_EVERY,
        ram_safety_gb      = RAM_SAFETY_GB,
        check_truncation   = args.check_truncation,

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

    # ── Hamiltonians (isotropic; all 9 bonds equal) ───────────────────────────
    H  = build_heisenberg_H(args.J, d_PHYS)
    Hs = [H] * 9  # the order defined as 0~8 as Heb,Had,Hcf, Haf,Hcb,Hed, Hcd,Hef,Hab

    # ── Banner ────────────────────────────────────────────────────────────────
    print("=" * 76)
    print("  iPEPS sweep  —  AD-CTMRG  —  AFM Heisenberg on 6-site honeycomb")
    print(f"  J={args.J}  d_phys={d_PHYS}  Total budget: {args.hours:.1f} h")
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
    best_abcdef_by_D: dict[int, tuple | None] = {D: None for D in D_bond_list}
    global_step = 0

    # ── Resume ────────────────────────────────────────────────────────────────
    resume_D, resume_chi = None, None
    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu')
        resume_D   = ckpt.get('D_bond')
        resume_chi = ckpt.get('chi')
        ckpt_step  = ckpt.get('step', 0) + 1
        ckpt_loss  = ckpt.get('loss', float('nan'))
        best_abcdef_by_D[resume_D] = _new_tensors_from_data((
            ckpt['a'], ckpt['b'], ckpt['c'],
            ckpt['d'], ckpt['e'], ckpt['f']))
        del ckpt; gc.collect()
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

        # ── Warm-start from D-1 if available ─────────────────────────────────
        prev_D = D_bond_list[D_bond_list.index(D_bond) - 1] \
                 if D_bond_list.index(D_bond) > 0 else None
        if (best_abcdef_by_D.get(D_bond) is None
                and prev_D is not None
                and best_abcdef_by_D.get(prev_D) is not None):
            print(f"  Warm-starting from D={prev_D} tensors "
                  f"(padding {prev_D}→{D_bond}, noise={args.noise})")
            prev_tensors = best_abcdef_by_D[prev_D]
            padded = tuple(
                pad_tensor(t, prev_D, D_bond, d_PHYS, args.noise)
                for t in prev_tensors)
            best_abcdef_by_D[D_bond] = _new_tensors_from_data(padded)
            del padded

        # current best tensors at this D (None = random init at first chi)
        cur_abcdef = best_abcdef_by_D.get(D_bond)

        # ── Inner loop: chi ───────────────────────────────────────────────────
        for chi in chis:
            # Skip (D, chi) pairs that come before the resume point
            if resume_D is not None and resume_chi is not None:
                if D_bond < resume_D:
                    continue
                if D_bond == resume_D and chi < resume_chi:
                    continue
                if D_bond == resume_D and chi == resume_chi:
                    resume_D = None   # resume point reached; continue normally
                    cur_abcdef = best_abcdef_by_D.get(D_bond)

            # equal time budget for every chi level within this D
            chi_budget = D_budget / len(chis)

            # RAM guard
            need_gb  = peak_ram_gb(chi, D_sq)
            avail_gb = free_ram_gb()
            if avail_gb < need_gb + RAM_SAFETY_GB:
                print(f"\n  ⚠  Skipping (D={D_bond}, chi={chi}): "
                      f"need {need_gb:.2f} GB + {RAM_SAFETY_GB} GB safety, "
                      f"only {avail_gb:.1f} GB free.")
                continue

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

            *out, best_loss, global_step = optimize_at_chi(
                Hs, D_bond, chi, d_PHYS,
                budget_seconds=chi_budget,
                lbfgs_max_iter=lbfgs_iters,
                init_abcdef=cur_abcdef,
                step_offset=global_step,
                best_path=best_path,
                latest_path=latest_path,
                loss_log=loss_log,
                out_dir=output_dir,
            )
            # ── Brand-new tensors from raw data; kill the old run entirely ──
            cur_abcdef = _new_tensors_from_data(tuple(out[:6]))
            del out
            gc.collect()  # free old optimizer state + CTMRG envs + graph
            # ── Return fragmented heap pages to the OS. ──────────────────────
            # glibc's ptmalloc2 does not automatically shrink the heap after
            # freeing large tensors; RSS keeps accumulating across chi levels
            # even after gc.collect().  malloc_trim(0) asks glibc to release
            # all releasable pages immediately, resetting RSS close to the
            # actual live-data footprint.  This is the primary cause of the
            # 600+ MB baseline seen at the start of large (D, chi) levels.
            try:
                import ctypes
                ctypes.CDLL(None).malloc_trim(0)
            except Exception:
                pass  # non-Linux systems: silently skip

            # Clean energy evaluation
            print(f"  │  Evaluating energy at (D={D_bond}, chi={chi}) ...")
            energy = evaluate_energy_clean(
                *cur_abcdef, Hs, chi, D_bond, d_PHYS)
            energy_per_bond = energy / N_BONDS

            # Save final checkpoint for this (D, chi)
            save_checkpoint(best_path, cur_abcdef, D_bond, chi,
                            best_loss, energy, global_step, loss_log)

            # Log
            record = {
                'D_bond': D_bond, 'chi': chi,
                'best_loss': best_loss, 'energy': energy,
                'energy_per_bond': energy_per_bond,
                'steps': len(loss_log),
                'timestamp': timestamp(),
            }
            energy_table.append(record)
            wall = time.perf_counter() - t_global_start
            print(f"  └── E={energy:+.10f}  E/bond={energy_per_bond:+.10f}"
                  f"  wall={wall/3600:.2f}h")

        # Store best tensors at this D for warm-starting next D
        # (brand-new objects — old D's entire graph is released)
        best_abcdef_by_D[D_bond] = (
            _new_tensors_from_data(cur_abcdef) if cur_abcdef is not None
            else None)
        del cur_abcdef
        gc.collect()
        try:
            import ctypes
            ctypes.CDLL(None).malloc_trim(0)
        except Exception:
            pass

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
          f"{'E_total':>18}  {'E/bond':>14}")
    print(f"  {'─'*2}  {'─'*5}  {'─'*6}  {'─'*18}  {'─'*14}")
    for row in energy_table:
        print(f"  {row['D_bond']:2d}  {row['chi']:5d}  {row['steps']:6d}  "
              f"{row['energy']:+18.10f}  {row['energy_per_bond']:+14.10f}")
    print()
    best_overall = min(energy_table, key=lambda r: r['energy_per_bond'])
    print(f"  Best E/bond = {best_overall['energy_per_bond']:+.10f}  "
          f"(D={best_overall['D_bond']}, chi={best_overall['chi']})")
    print(f"  QMC reference (D→∞): E/bond ≈ −0.3630")
    print(f"  Total wall time: {total_elapsed/3600:.2f} h ({total_elapsed:.0f} s)")
    print(f"  Finished: {timestamp()}")
    print("=" * 76)

    # ── JSON ──────────────────────────────────────────────────────────────────
    results_path = os.path.join(output_dir, "sweep_results.json")
    json_out = {
        'D_bond_list': D_bond_list,
        'chi_max_map': {str(k): v for k, v in chi_max_map.items()},
        'schedules':   {str(k): v for k, v in schedules.items()},
        'J': args.J, 'd_phys': d_PHYS,
        'hours_budget': args.hours,
        'total_elapsed_h': round(total_elapsed / 3600, 3),
        'total_steps': global_step,
        'energy_table': energy_table,
        'timestamp': timestamp(),
    }
    with open(results_path, 'w') as fp:
        json.dump(json_out, fp, indent=2)
    print(f"\n  Results JSON → {results_path}")

    # ── Plot 1: E/bond vs chi for each D ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    for D in D_bond_list:
        rows = [r for r in energy_table if r['D_bond'] == D]
        if not rows:
            continue
        x = [r['chi'] for r in rows]
        y = [r['energy_per_bond'] for r in rows]
        ax.plot(x, y, 'o-', label=f'D={D}', markerfacecolor='white',
                markersize=7)
    ax.axhline(-0.3630, color='grey', ls='--', lw=1, label='QMC (D→∞) ≈ −0.3630')
    ax.set_xlabel(r'Environment bond dimension $\chi$', fontsize=12)
    ax.set_ylabel(r'Energy per bond $E/J$', fontsize=12)
    ax.set_title('iPEPS ground-state energy — AFM Heisenberg (honeycomb)', fontsize=12)
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'sweep_energy_vs_chi.pdf'))
    plt.close(fig)

    # ── Plot 2: E/bond vs D at largest chi each D ─────────────────────────────
    D_vals, E_best = [], []
    for D in D_bond_list:
        rows = [r for r in energy_table if r['D_bond'] == D]
        if rows:
            best_row = min(rows, key=lambda r: r['energy_per_bond'])
            D_vals.append(D)
            E_best.append(best_row['energy_per_bond'])
    if len(D_vals) > 1:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.plot(D_vals, E_best, 's-', color='tab:red', markersize=9,
                 markerfacecolor='white')
        ax2.axhline(-0.3630, color='grey', ls='--', lw=1, label='QMC')
        ax2.set_xlabel(r'Bond dimension $D$', fontsize=12)
        ax2.set_ylabel(r'Best $E/\text{bond}$', fontsize=12)
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

