#!/usr/bin/env python3
"""
main_sweep_CPU_single_tensor.py
===============================
CPU edition of main_sweep_GPU_single_tensor.py — single-tensor ansatz
  b = d = f = –a,  c = e = a   (3 constraints, 1 free tensor).

This ansatz is much cheaper than the 6-tensor version:
  • CTMRG converges in ~4 steps (vs ~40 for 6-tensor).
  • Only one gradient tensor → 6× gradient amplification via chain rule;
    use LBFGS_LR = 0.01 (not 1.0) to compensate.
  • On 4–16 core CPU hardware this can keep pace with or exceed a typical
    laptop GPU (MX250/MX450) for D ≤ 3 because the problem is small enough
    that CPU MKL outperforms GPU kernel-launch overhead.

Structure
---------
  for D_bond in [2, 3, 4]:          <- outer loop
      for chi in chi_schedule(D):    <- inner loop, D²<chi<=D⁴
          optimise until chi-budget exhausted

Usage
-----
  python scripts/main_sweep_CPU_single_tensor.py                # all defaults
  python scripts/main_sweep_CPU_single_tensor.py --hours 4
  python scripts/main_sweep_CPU_single_tensor.py --double       # float64

Outputs (all in log/)
---------------------
  sweep_D{D}_chi{chi}_best.pt    best checkpoint for each (D,chi)
  sweep_D{D}_chi{chi}_latest.pt  last checkpoint for each (D,chi)
  sweep_results.json             JSON table of all energies
  sweep_loss_D{D}.pdf            loss curve per D
  sweep_energy_vs_chi.pdf        E/bond vs chi for each D
  sweep_energy_vs_D.pdf          E/bond vs D at chi_max
"""

import argparse
import collections
import datetime
import json
import os
import sys
import time

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
#   OMP_NUM_THREADS=2 python scripts/main_sweep_CPU_single_tensor.py
_N_PHYSICAL_CORES = 4
os.environ.setdefault("OMP_NUM_THREADS", str(_N_PHYSICAL_CORES))
os.environ.setdefault("MKL_NUM_THREADS", str(_N_PHYSICAL_CORES))
# Prevent MKL from silently reducing thread count when it detects nested
# parallelism or high system load — we always want the full 4 threads.
os.environ.setdefault("MKL_DYNAMIC", "FALSE")
# Pin each OpenMP thread to a distinct physical core (not a hyperthread sibling).
# granularity=fine  -> thread-level pinning (finest possible)
# compact           -> pack onto as few sockets as possible (single-socket here)
# 1                 -> permute: fill one HT per core before doubling up (with
#                     4 threads on 4 cores this has no effect, but future-proofs)
# 0                 -> offset: start from logical CPU 0
os.environ.setdefault("KMP_AFFINITY", "granularity=fine,compact,1,0")
# After a parallel region the OpenMP workers normally spin-wait for ~200 ms
# before going to sleep, burning a whole core.  Set to 0 so they yield
# immediately — the outer L-BFGS loop is sequential so this saves real time.
os.environ.setdefault("KMP_BLOCKTIME", "0")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import opt_einsum as oe
import torch

# Tell PyTorch's own dispatcher to use all physical cores for intra-op
# parallelism (matrix multiply, SVD, etc.) and only 1 thread for the
# inter-op pool — our outer loop is serial, extra inter-op threads just
# compete with the intra-op pool for the same physical cores.
torch.set_num_threads(_N_PHYSICAL_CORES)
torch.set_num_interop_threads(1)

from core_unrestricted_single_tensor import (
    normalize_tensor,
    initialize_abcdef,
    abcdef_to_ABCDEF,
    CTMRG_from_init_to_stop,
    #energy_expectation_nearest_neighbor_6_bonds,
    energy_expectation_nearest_neighbor_3ebadcf_bonds,
    energy_expectation_nearest_neighbor_3afcbed_bonds,
    energy_expectation_nearest_neighbor_other_3_bonds,
    set_dtype,
)

# ══════════════════════════════════════════════════════════════════════════════
# Time Budget
# ══════════════════════════════════════════════════════════════════════════════

TOTAL_BUDGET_HOURS = 1.0

# Total wall-clock time for the entire sweep.  The sweep is designed to run
# for a fixed time rather than a fixed number of steps, so that results at
# different (D, chi) levels are directly comparable.  The default is 11 h, which
# is enough to get good convergence at D=4, chi=80 on an MX250 GPU.  Adjust
# according to your hardware and patience.  1 h is enough to get good results at D=3, chi=32; 20 min is enough for D=2.




# Default complex dtype — updated to complex128 by --double / USE_DOUBLE_PRECISION.
# Python functions look up globals at call time, so build_heisenberg_H(),
# pad_tensor(), and optimize_at_chi() all see the updated value after main()
# calls set_dtype() and reassigns CDTYPE.
CDTYPE: torch.dtype = torch.complex64

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                  ALL TUNABLE PARAMETERS — EDIT HERE                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ── Precision ─────────────────────────────────────────────────────────────────

USE_DOUBLE_PRECISION = True
#   False -> complex64  / float32 (default): fastest on both CPU (Intel MKL)
#            and CUDA (fp32 tensor cores give full throughput).
#   True  -> complex128 / float64: double precision.
#            CPU (Intel MKL): float64 is native, same throughput, 2x RAM.
#            CUDA: 2-4x slower on consumer GPUs (no fp64 tensor cores).
#   Overrideable at runtime: --double CLI flag takes precedence.

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
#             constraint for our honeycomb ansatz is  D² < chi <= D⁴.

DEFAULT_D_BUDGET_FRACS = {2: 0.15, 3: 0.35, 4: 0.5}
#   Fraction of the total wall-clock budget allocated to each D_bond value.
#   Normalised to sum=1 before use, so only the RATIOS matter.

DEFAULT_CHI_MAX = {2: 16, 3: 81, 4: 80}
#   Largest chi to attempt for each D_bond.  Hard upper bound is D⁴.

DEFAULT_CHI_SCHEDULES = {
    2: [5, 8, 14],
    3: [10, 17, 29],       # , 57, 81],  <- append to extend the schedule
    4: [17, 29],           # , 40, 62, 80] <- append to extend the schedule
}

# ── L-BFGS optimiser ─────────────────────────────────────────────────────────

LBFGS_MAX_ITER = 30
#   Maximum L-BFGS sub-iterations per outer step.

LBFGS_LR = 0.01
#   Step-size seed for the strong-Wolfe line search.
#   NOTE: the single-tensor constraint (b=d=f=-a, c=e=a) produces a ~6x
#   gradient amplification via chain rule compared to the unrestricted ansatz.
#   lr=0.01 compensates; do NOT use lr=1.0 here.

LBFGS_HISTORY = LBFGS_MAX_ITER
#   Number of (s, y) curvature vector pairs retained for the L-BFGS inverse-
#   Hessian approximation.  In our alternating-optimisation scheme the LBFGS
#   instance is RECREATED from scratch at every outer step, so curvature pairs
#   accumulate only within a single optimizer.step() call (≤ LBFGS_MAX_ITER
#   sub-iterations).  Any history_size > LBFGS_MAX_ITER allocates buffer
#   memory that is never filled — setting it equal to LBFGS_MAX_ITER is exact
#   and wastes nothing.

OPT_TOL_GRAD = 1e-7
#   L-BFGS inner convergence: infinity-norm of gradient.

OPT_TOL_CHANGE = 1e-9
#   L-BFGS inner convergence: consecutive loss change.

OPT_CONV_THRESHOLD = 1e-8
#   Outer-loop early-stop: |loss(k) - loss(k-1)| < this -> move on.

# ── Optimizer choice ─────────────────────────────────────────────────────────

OPTIMIZER = 'lbfgs'
#   'lbfgs' : L-BFGS with strong-Wolfe line search (default).
#             Converges fast on smooth landscapes; may oscillate on noisy ones.
#   'adam'  : Adam (adaptive moment estimation).
#             More robust on noisy/non-convex landscapes.  Each outer step runs
#             ADAM_STEPS_PER_CTM gradient updates with a fixed environment.
#   Override at runtime:  --optimizer {lbfgs,adam}

# ── Adam hyperparameters (used only when OPTIMIZER='adam') ───────────────────

ADAM_LR = 3e-4
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8
ADAM_WEIGHT_DECAY = 0.0
ADAM_STEPS_PER_CTM = 5
#   Number of Adam gradient steps between CTMRG environment refreshes.

# ── CTMRG algorithm ──────────────────────────────────────────────────────────

CTM_MAX_STEPS = 70
#   Hard cap on CTMRG iterations.  Single-tensor converges in ~4 steps.

CTM_CONV_THR = 1e-9
#   CTMRG convergence threshold (corner SVD spectrum, gauge-invariant).

# ── Checkpointing & memory guard ─────────────────────────────────────────────

SAVE_EVERY = 20
#   Frequency (outer steps) for writing the "latest" checkpoint.

RAM_SAFETY_GB = 0.4
#   Minimum free RAM before attempting a (D, chi) level.

# ── Physical model ────────────────────────────────────────────────────────────

J_COUPLING = 1.0
#   Isotropic Heisenberg exchange coupling.  J > 0 = AFM.

D_PHYS = 2
#   Physical Hilbert-space dimension (d=2 for spin-1/2).

N_BONDS = 9
#   Number of nearest-neighbour bonds in the 6-site honeycomb unit cell.

# ── Sweep control ─────────────────────────────────────────────────────────────

D_BOND_LIST = [2, 3, 4]
#   Ordered D_bond values for the outer sweep.

# ── Tensor initialisation & padding ──────────────────────────────────────────

INIT_NOISE = 1e-3
PAD_NOISE  = 1e-3

GEO_SCHEDULE_STEPS = 5


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


def peak_ram_gb(chi: int, D_sq: int) -> float:
    """Estimate peak RAM for one CTMRG step in GB."""
    n = chi * D_sq
    return 3 * 4 * n * n * 8 / 1e9


def validate_chi(chi: int, D_bond: int, label: str = '') -> None:
    D_sq, D4 = D_bond ** 2, D_bond ** 4
    if not (D_sq < chi <= D4):
        raise ValueError(
            f"{label}chi={chi} violates D²={D_sq} < chi <= D⁴={D4} "
            f"(D_bond={D_bond}).")


def build_heisenberg_H(J: float = 1.0, d: int = 2) -> torch.Tensor:
    sx = torch.tensor([[0, 1], [1, 0]], dtype=CDTYPE) / 2
    sy = torch.tensor([[0, -1j], [1j, 0]], dtype=CDTYPE) / 2
    sz = torch.tensor([[1, 0], [0, -1]], dtype=CDTYPE) / 2
    SdotS = (oe.contract("ij,kl->ikjl", sx, sx)
           + oe.contract("ij,kl->ikjl", sy, sy)
           + oe.contract("ij,kl->ikjl", sz, sz))
    return J * SdotS


def pad_tensor(t: torch.Tensor, old_D: int, new_D: int,
               d_PHYS: int, noise: float) -> torch.Tensor:
    out = noise * torch.randn(new_D, new_D, new_D, d_PHYS, dtype=CDTYPE)
    s = old_D
    out[:s, :s, :s, :] = t[:s, :s, :s, :]
    return normalize_tensor(out)


def evaluate_energy_clean(a, Hs, chi: int, D_bond: int) -> float:
    """Re-converge environment from scratch and return total energy (float).

    Single-tensor ansatz: b=d=f=-a, c=e=a.
    """
    b, c, d, e, f = -a, a, -a, a, -a
    D_sq = D_bond ** 2
    with torch.no_grad():
        A, B, C, Dt, E, F = abcdef_to_ABCDEF(a, b, c, d, e, f, D_sq)
        all27 = CTMRG_from_init_to_stop(
                A, B, C, Dt, E, F, chi, D_sq, CTM_MAX_STEPS, CTM_CONV_THR)
        E3ebadcf = energy_expectation_nearest_neighbor_3ebadcf_bonds(
                a,b,c,d,e,f, 
                Hs[0],Hs[1],Hs[2],
                chi, D_bond, # d_PHYS, 
                *all27[:9])
        E3afcbed = energy_expectation_nearest_neighbor_3afcbed_bonds(
                a,b,c,d,e,f, 
                Hs[3],Hs[4],Hs[5], 
                chi, D_bond, # d_PHYS, 
                *all27[9:18])
            
        E3 = energy_expectation_nearest_neighbor_other_3_bonds(
            a, b, c, d, e, f,
            Hs[6], Hs[7], Hs[8],
            chi, D_bond, *all27[18:27])
        return (E3ebadcf + E3afcbed + E3).real.item()


def save_checkpoint(path: str, abcdef: tuple, D_bond: int, chi: int,
                    loss: float, energy: float | None,
                    step: int, log: list) -> None:
    # Only `a` is the independent tensor; b..f are derived.
    torch.save({
        'a': abcdef[0],
        'D_bond': D_bond, 'chi': chi,
        'loss': loss, 'energy': energy,
        'step': step, 'timestamp': timestamp(),
        'log': log,
        'ansatz': 'single_tensor',
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
) -> tuple:
    """
    Outer L-BFGS loop at fixed (D_bond, chi) until budget_seconds elapsed.

    Single-tensor ansatz: only `a` is optimised; b..f derived.
    Returns (a, best_loss, steps_done).
    """
    if loss_log is None:
        loss_log = []
    D_sq = D_bond ** 2
    t_start = time.perf_counter()

    # initialise tensor `a`
    if init_abcdef is not None:
        a = init_abcdef[0].detach().clone().to(CDTYPE)
    else:
        a = initialize_abcdef('random', D_bond, d_PHYS, INIT_NOISE)[0]
    a.requires_grad_(True)

    best_loss   = float('inf')
    best_abcdef = (a.detach().clone(),)
    prev_loss   = None
    loss_history = collections.deque(maxlen=12)
    step = step_offset

    # ── pre-create Adam optimizer (state persists across outer steps) ─────────
    _adam: torch.optim.Optimizer | None = None
    if OPTIMIZER == 'adam':
        _adam = torch.optim.Adam(
            [a],
            lr=ADAM_LR, betas=ADAM_BETAS, eps=ADAM_EPS,
            weight_decay=ADAM_WEIGHT_DECAY)

    while True:
        elapsed = time.perf_counter() - t_start
        if elapsed >= budget_seconds:
            break

        # normalise (scale-redundancy fix, preserves requires_grad)
        with torch.no_grad():
            a.data = normalize_tensor(a.data)

        # converge environment (no grad)
        with torch.no_grad():
            b, c, d, e, f = -a, a, -a, a, -a
            A, B, C, D, E, F = abcdef_to_ABCDEF(a, b, c, d, e, f, D_sq)
            (C21CD, C32EF, C13AB, T1F,  T2A,  T2B,  T3C,  T3D,  T1E,
             C21EB, C32AD, C13CF, T1D,  T2C,  T2F,  T3E,  T3B,  T1A,
             C21AF, C32CB, C13ED, T1B,  T2E,  T2D,  T3A,  T3F,  T1C,
             ctm_steps) = \
                CTMRG_from_init_to_stop(
                    A, B, C, D, E, F, chi, D_sq,
                    CTM_MAX_STEPS, CTM_CONV_THR)

        def _energy():
            b, c, d, e, f = -a, a, -a, a, -a
            return (
            energy_expectation_nearest_neighbor_3ebadcf_bonds(
                a,b,c,d,e,f, 
                Hs[0],Hs[1],Hs[2],
                chi, D_bond, # d_PHYS, 
                C21CD,C32EF,C13AB,T1F,T2A,T2B,T3C,T3D,T1E)
            +
            energy_expectation_nearest_neighbor_3afcbed_bonds(
                a,b,c,d,e,f, 
                Hs[3],Hs[4],Hs[5], 
                chi, D_bond, # d_PHYS, 
                C21EB, C32AD,C13CF,T1D,T2C,T2F,T3E,T3B,T1A)
            +
            energy_expectation_nearest_neighbor_other_3_bonds(
                    a, b, c, d, e, f,
                    Hs[6], Hs[7], Hs[8],
                    chi, D_bond,
                    C21AF, C32CB, C13ED, T1B, T2E, T2D, T3A, T3F, T1C)
            ).real

        if OPTIMIZER == 'lbfgs':
            _opt = torch.optim.LBFGS(
                [a],
                lr=LBFGS_LR,
                max_iter=lbfgs_max_iter,
                tolerance_grad=OPT_TOL_GRAD,
                tolerance_change=OPT_TOL_CHANGE,
                history_size=LBFGS_HISTORY,
                line_search_fn='strong_wolfe',
            )
            def closure():
                _opt.zero_grad()
                loss = _energy()
                loss.backward()
                return loss
            loss_val  = _opt.step(closure)
            loss_item = loss_val.item()
        else:  # adam
            for _s in range(ADAM_STEPS_PER_CTM):
                _adam.zero_grad()
                _loss = _energy()
                _loss.backward()
                _adam.step()
            loss_item = _loss.detach().item()
        delta     = (loss_item - prev_loss) if prev_loss is not None else float('inf')
        elapsed   = time.perf_counter() - t_start

        print(f"    step {step:5d}  ctm={ctm_steps:3d}  loss={loss_item:+.10f}"
              f"  Δ={delta:+.3e}  {elapsed:.0f}/{budget_seconds:.0f}s")
        loss_log.append({'step': step, 'ctm_steps': ctm_steps, 'loss': loss_item,
                         'D_bond': D_bond, 'chi': chi,
                         'elapsed': round(elapsed, 1)})

        if loss_item < best_loss:
            best_loss   = loss_item
            best_abcdef = (a.detach().clone(),)
            if best_path:
                save_checkpoint(best_path, best_abcdef, D_bond, chi,
                                best_loss, None, step, loss_log)

        if latest_path and (step - step_offset) % SAVE_EVERY == 0 and step > step_offset:
            save_checkpoint(latest_path, best_abcdef, D_bond, chi,
                            best_loss, None, step, loss_log)

        if prev_loss is not None and abs(delta) < OPT_CONV_THRESHOLD:
            print(f"    Outer convergence at step {step} (Δ={delta:.2e})")
            break
        if any(abs(loss_item - h) < 1e-10 for h in loss_history):
            print(f"    Cycle detected at step {step} "
                  f"(amplitude={abs(delta):.3e}); stopping.")
            break
        loss_history.append(loss_item)
        prev_loss = loss_item
        step += 1

    return (*best_abcdef, best_loss, step)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global CDTYPE, OPTIMIZER
    parser = argparse.ArgumentParser(
        description="iPEPS sweep (single-tensor ansatz) — AD-CTMRG — CPU",
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
        help='Comma-separated chi_max for each D_bond (must satisfy D²<chi<=D⁴).')
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
        help='Gaussian noise amplitude when padding tensors for D->D+1.')
    parser.add_argument(
        '--double', action='store_true', default=USE_DOUBLE_PRECISION,
        help='Use float64/complex128 everywhere (default: float32/complex64). '
             'Same throughput on CPU/MKL; 2-4x slower on CUDA consumer GPUs.')
    parser.add_argument(
        '--optimizer', choices=['lbfgs', 'adam'], default=OPTIMIZER,
        help="Optimiser: 'lbfgs' (default) or 'adam' (robust on noisy landscapes).")
    args = parser.parse_args()

    # ── Precision + optimizer setup ───────────────────────────────────────────
    CDTYPE = torch.complex128 if args.double else torch.complex64
    set_dtype(args.double)
    OPTIMIZER = args.optimizer

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
    def chi_schedule(D: int, chi_max: int) -> list[int]:
        if chi_max_map.get(D) == DEFAULT_CHI_MAX.get(D) and D in DEFAULT_CHI_SCHEDULES:
            sched = list(DEFAULT_CHI_SCHEDULES[D])
        else:
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
        sched = [c for c in sched if D**2 < c <= chi_max]
        if not sched:
            raise ValueError(
                f"Empty chi schedule for D={D} after clipping to ({D**2}, {chi_max}].")
        return sorted(set(sched))

    schedules = {D: chi_schedule(D, chi_max_map[D]) for D in D_bond_list}

    # ── time budget per D_bond ────────────────────────────────────────────────
    raw_fracs = {D: DEFAULT_D_BUDGET_FRACS.get(D, 1.0) for D in D_bond_list}
    total_frac = sum(raw_fracs.values())
    d_budgets = {D: total_budget * raw_fracs[D] / total_frac
                 for D in D_bond_list}

    # ── output directory ──────────────────────────────────────────────────────
    run_ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    _default_outdir = os.path.join('/home/chye/6ADctmrg/data/raw',
                                   f'1tensor_{run_ts}')
    output_dir = args.output_dir or _default_outdir
    os.makedirs(output_dir, exist_ok=True)

    # ── save hyperparameters to YAML (JSON fallback if PyYAML not installed) ──
    _hp = dict(
        ansatz='1tensor',
        optimizer=OPTIMIZER,
        lbfgs_lr=LBFGS_LR, lbfgs_max_iter=LBFGS_MAX_ITER,
        lbfgs_history=LBFGS_HISTORY, opt_tol_grad=OPT_TOL_GRAD,
        opt_tol_change=OPT_TOL_CHANGE, opt_conv_threshold=OPT_CONV_THRESHOLD,
        adam_lr=ADAM_LR, adam_betas=list(ADAM_BETAS), adam_eps=ADAM_EPS,
        adam_weight_decay=ADAM_WEIGHT_DECAY, adam_steps_per_ctm=ADAM_STEPS_PER_CTM,
        ctm_max_steps=CTM_MAX_STEPS, ctm_conv_thr=CTM_CONV_THR,
        J=args.J, d_phys=d_PHYS, double=args.double,
        hours=args.hours, D_bond_list=D_bond_list,
        chi_max_map={str(k): v for k, v in chi_max_map.items()},
        schedules={str(k): v for k, v in schedules.items()},
        run_timestamp=run_ts,
    )
    try:
        import yaml
        with open(os.path.join(output_dir, 'hyperparams.yaml'), 'w') as _fp:
            yaml.dump(_hp, _fp, default_flow_style=False, sort_keys=False)
    except ImportError:
        import json as _json
        with open(os.path.join(output_dir, 'hyperparams.yaml'), 'w') as _fp:
            _json.dump(_hp, _fp, indent=2)

    # ── Hamiltonians ──────────────────────────────────────────────────────────
    H  = build_heisenberg_H(args.J, d_PHYS)
    Hs = [H] * 9

    # ── Banner ────────────────────────────────────────────────────────────────
    dtype_str = 'float64/complex128' if args.double else 'float32/complex64'
    print("=" * 76)
    print("  iPEPS sweep  —  AD-CTMRG  —  AFM Heisenberg on 6-site honeycomb")
    print("  Ansatz       : single-tensor  (b=d=f=-a,  c=e=a)")
    print(f"  J={args.J}  d_phys={d_PHYS}  dtype={dtype_str}")
    print(f"  CPU threads  : {_N_PHYSICAL_CORES} (intra-op), 1 (inter-op)")
    print(f"  Total budget : {args.hours:.1f} h")
    print(f"  D_bond sweep : {D_bond_list}")
    for D in D_bond_list:
        bh = d_budgets[D] / 3600
        print(f"    D={D}: chi={schedules[D]}  budget={bh:.2f} h  "
              f"(chi_max={chi_max_map[D]})")
    print(f"  Output dir   : {output_dir}")
    print(f"  Started      : {timestamp()}")
    print("=" * 76)

    # ── State ─────────────────────────────────────────────────────────────────
    all_loss_logs: dict[tuple, list] = {}
    energy_table: list[dict] = []
    best_abcdef_by_D: dict[int, tuple | None] = {D: None for D in D_bond_list}
    global_step = 0

    # ── Resume ────────────────────────────────────────────────────────────────
    resume_D, resume_chi = None, None
    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu')
        resume_D   = ckpt.get('D_bond')
        resume_chi = ckpt.get('chi')
        best_abcdef_by_D[resume_D] = (ckpt['a'],)
        global_step = ckpt.get('step', 0) + 1
        print(f"  Resumed from {args.resume}  "
              f"(D={resume_D}, chi={resume_chi}, "
              f"loss={ckpt.get('loss', float('nan')):.6f})\n")

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
                  f"(padding {prev_D}->{D_bond}, noise={args.noise})")
            prev_tensors = best_abcdef_by_D[prev_D]
            best_abcdef_by_D[D_bond] = tuple(
                pad_tensor(t, prev_D, D_bond, d_PHYS, args.noise)
                for t in prev_tensors)

        cur_abcdef = best_abcdef_by_D.get(D_bond)

        # ── Inner loop: chi ───────────────────────────────────────────────────
        for chi in chis:
            if resume_D is not None and resume_chi is not None:
                if D_bond < resume_D:
                    continue
                if D_bond == resume_D and chi < resume_chi:
                    continue
                if D_bond == resume_D and chi == resume_chi:
                    resume_D = None
                    cur_abcdef = best_abcdef_by_D.get(D_bond)

            chi_budget = D_budget / len(chis)

            # RAM guard
            need_gb  = peak_ram_gb(chi, D_sq)
            avail_gb = free_ram_gb()
            if avail_gb < need_gb + RAM_SAFETY_GB:
                print(f"\n  ⚠  Skipping (D={D_bond}, chi={chi}): "
                      f"need {need_gb:.2f} GB + {RAM_SAFETY_GB} GB safety, "
                      f"only {avail_gb:.1f} GB free.")
                continue

            print(f"\n  ┌── D={D_bond}  chi={chi}"
                  f"  budget={chi_budget:.0f}s={chi_budget/60:.1f}min"
                  f"  [{timestamp()}]")

            best_path   = os.path.join(output_dir,
                                       f"sweep_D{D_bond}_chi{chi}_best.pt")
            latest_path = os.path.join(output_dir,
                                       f"sweep_D{D_bond}_chi{chi}_latest.pt")
            loss_log: list = []
            all_loss_logs[(D_bond, chi)] = loss_log

            *out, best_loss, global_step = optimize_at_chi(
                Hs, D_bond, chi, d_PHYS,
                budget_seconds=chi_budget,
                lbfgs_max_iter=LBFGS_MAX_ITER,
                init_abcdef=cur_abcdef,
                step_offset=global_step,
                best_path=best_path,
                latest_path=latest_path,
                loss_log=loss_log,
            )
            cur_abcdef = tuple(out[:1])  # only `a`

            # Clean energy evaluation
            print(f"  │  Evaluating energy at (D={D_bond}, chi={chi}) ...")
            energy = evaluate_energy_clean(cur_abcdef[0], Hs, chi, D_bond)
            energy_per_bond = energy / N_BONDS

            save_checkpoint(best_path, cur_abcdef, D_bond, chi,
                            best_loss, energy, global_step, loss_log)

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

        best_abcdef_by_D[D_bond] = cur_abcdef

        D_elapsed = time.perf_counter() - D_start_time
        print(f"\n  D={D_bond} complete in {D_elapsed/3600:.2f} h")
        global_step += 1

    # ══════════════════════════════════════════════════════════════════════════
    # Results summary
    # ══════════════════════════════════════════════════════════════════════════
    total_elapsed = time.perf_counter() - t_global_start

    print("\n" + "=" * 76)
    print("  RESULTS SUMMARY  (single-tensor ansatz)")
    print("=" * 76)
    print(f"  {'D':>2}  {'chi':>5}  {'steps':>6}  "
          f"{'E_total':>18}  {'E/bond':>14}")
    print(f"  {'─'*2}  {'─'*5}  {'─'*6}  {'─'*18}  {'─'*14}")
    for row in energy_table:
        print(f"  {row['D_bond']:2d}  {row['chi']:5d}  {row['steps']:6d}  "
              f"{row['energy']:+18.10f}  {row['energy_per_bond']:+14.10f}")
    print()
    if energy_table:
        best_overall = min(energy_table, key=lambda r: r['energy_per_bond'])
        print(f"  Best E/bond = {best_overall['energy_per_bond']:+.10f}  "
              f"(D={best_overall['D_bond']}, chi={best_overall['chi']})")
    print(f"  QMC reference (D->inf): E/bond ≈ −0.3646")
    print(f"  Total wall time: {total_elapsed/3600:.2f} h ({total_elapsed:.0f} s)")
    print(f"  Finished: {timestamp()}")
    print("=" * 76)

    # ── JSON ──────────────────────────────────────────────────────────────────
    results_path = os.path.join(output_dir, "sweep_results_single_tensor.json")
    json_out = {
        'ansatz': 'single_tensor',
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
    print(f"\n  Results JSON -> {results_path}")

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
    ax.axhline(-0.3646, color='grey', ls='--', lw=1, label='QMC (D->inf) ≈ −0.3646')
    ax.set_xlabel(r'Environment bond dimension $\chi$', fontsize=12)
    ax.set_ylabel(r'Energy per bond $E/J$', fontsize=12)
    ax.set_title('iPEPS — single-tensor ansatz — AFM Heisenberg (honeycomb)',
                 fontsize=12)
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'sweep_energy_vs_chi_single_tensor.pdf'))
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
        ax2.axhline(-0.3646, color='grey', ls='--', lw=1, label='QMC')
        ax2.set_xlabel(r'Bond dimension $D$', fontsize=12)
        ax2.set_ylabel(r'Best $E/\text{bond}$', fontsize=12)
        ax2.set_title('iPEPS convergence in D (single-tensor)')
        ax2.legend(); ax2.grid(True, alpha=0.3)
        fig2.tight_layout()
        fig2.savefig(
            os.path.join(output_dir, 'sweep_energy_vs_D_single_tensor.pdf'))
        plt.close(fig2)

    # ── Plot 3: loss vs step for each D ──────────────────────────────────────
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
        ax3.set_title(f'Loss curve  D={D}  (single-tensor)')
        ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3)
        fig3.tight_layout()
        fig3.savefig(
            os.path.join(output_dir, f'sweep_loss_D{D}_single_tensor.pdf'))
        plt.close(fig3)

    print(f"  Plots -> {output_dir}/sweep_*_single_tensor.pdf")


if __name__ == '__main__':
    main()
