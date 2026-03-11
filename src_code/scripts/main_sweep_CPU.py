#!/usr/bin/env python3
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
  lower-chi ramp :  25 %  (spread evenly over all chi < chi_max)
  production chi  :  75 %  (poured into the largest chi)

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

import argparse
import datetime
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import opt_einsum as oe
import torch

from core_unrestricted import (
    normalize_tensor,
    initialize_abcdef,
    abcdef_to_ABCDEF,
    CTMRG_from_init_to_stop,
    energy_expectation_nearest_neighbor_6_bonds,
    energy_expectation_nearest_neighbor_other_3_bonds,
)

# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

# Fraction of each D_bond's time budget reserved for the ramp-up chi levels
# (all chi < chi_max).  The remaining (1 - RAMP_FRAC) goes to chi_max.
RAMP_FRAC = 0.25

# Default fraction of total budget assigned to each D_bond
DEFAULT_D_BUDGET_FRACS = {2: 0.03, 3: 0.52, 4: 0.45}

# Default chi_max for each D_bond (must satisfy D²<chi≤D⁴)
DEFAULT_CHI_MAX = {2: 16, 3: 81, 4: 80}

# Default chi schedules (geometric, all values in D²<chi≤D⁴)
DEFAULT_CHI_SCHEDULES = {
    2: [5, 8, 12, 16],
    3: [10, 18, 32],#, 57, 81],
    4: [17, 26]#, 40, 62, 80],
}

# L-BFGS hyper-parameters
LBFGS_MAX_ITER_RAMP = 15    # sub-iters during ramp-up chi (cheaper)
LBFGS_MAX_ITER_PROD = 30    # sub-iters at production chi (richer)
LBFGS_LR            = 1.0
LBFGS_HISTORY       = 100
OPT_TOL_GRAD        = 1e-7
OPT_TOL_CHANGE      = 1e-9
OPT_CONV_THRESHOLD  = 1e-8  # stop outer loop when |Δloss| < this

# CTMRG parameters
CTM_MAX_STEPS = 90
CTM_CONV_THR  = 1e-7

# Checkpointing
SAVE_EVERY    = 20   # save latest checkpoint every N outer steps
RAM_SAFETY_GB = 2.0  # GB of free RAM to keep in reserve


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
    return 3 * 4 * n * n * 8 / 1e9   # 3 corners × 4× SVD workspace × element size


def validate_chi(chi: int, D_bond: int, label: str = '') -> None:
    D_sq, D4 = D_bond ** 2, D_bond ** 4
    if not (D_sq < chi <= D4):
        raise ValueError(
            f"{label}chi={chi} violates D²={D_sq} < chi ≤ D⁴={D4} "
            f"(D_bond={D_bond})."
        )


def build_heisenberg_H(J: float = 1.0, d: int = 2) -> torch.Tensor:
    sx = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64) / 2
    sy = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64) / 2
    sz = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64) / 2
    SdotS = (oe.contract("ij,kl->ikjl", sx, sx)
           + oe.contract("ij,kl->ikjl", sy, sy)
           + oe.contract("ij,kl->ikjl", sz, sz))
    return J * SdotS


def pad_tensor(t: torch.Tensor, old_D: int, new_D: int,
               d_PHYS: int, noise: float) -> torch.Tensor:
    out = noise * torch.randn(new_D, new_D, new_D, d_PHYS,
                              dtype=torch.complex64)
    s = old_D
    out[:s, :s, :s, :] = t[:s, :s, :s, :]
    return normalize_tensor(out)


def evaluate_energy_clean(a, b, c, d, e, f,
                          Hs, chi: int, D_bond: int) -> float:
    """Re-converge environment from scratch and return total energy (float)."""
    D_sq = D_bond ** 2
    with torch.no_grad():
        A, B, C, Dt, E, F = abcdef_to_ABCDEF(a, b, c, d, e, f, D_sq)
        all27 = CTMRG_from_init_to_stop(
            A, B, C, Dt, E, F, chi, D_sq, CTM_MAX_STEPS, CTM_CONV_THR)
        E6 = energy_expectation_nearest_neighbor_6_bonds(
            a, b, c, d, e, f,
            Hs[0], Hs[1], Hs[2], Hs[3], Hs[4], Hs[5],
            chi, D_bond, *all27[:9])
        E3 = energy_expectation_nearest_neighbor_other_3_bonds(
            a, b, c, d, e, f,
            Hs[6], Hs[7], Hs[8],
            chi, D_bond, *all27[18:27])
        return (E6 + E3).real.item()


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

    # initialise tensors
    if init_abcdef is not None:
        a, b, c, d, e, f = [t.detach().clone().to(torch.complex64)
                             for t in init_abcdef]
    else:
        a, b, c, d, e, f = initialize_abcdef('random', D_bond, d_PHYS, 1e-3)
    for t in (a, b, c, d, e, f):
        t.requires_grad_(True)

    best_loss   = float('inf')
    best_abcdef = tuple(t.detach().clone() for t in (a, b, c, d, e, f))
    prev_loss   = None
    step        = step_offset

    while True:
        elapsed = time.perf_counter() - t_start
        if elapsed >= budget_seconds:
            break

        # normalise (scale-redundancy fix, preserves requires_grad)
        with torch.no_grad():
            a.data = normalize_tensor(a.data)
            b.data = normalize_tensor(b.data)
            c.data = normalize_tensor(c.data)
            d.data = normalize_tensor(d.data)
            e.data = normalize_tensor(e.data)
            f.data = normalize_tensor(f.data)

        # fresh L-BFGS for this outer step
        optimizer = torch.optim.LBFGS(
            [a, b, c, d, e, f],
            lr=LBFGS_LR,
            max_iter=lbfgs_max_iter,
            tolerance_grad=OPT_TOL_GRAD,
            tolerance_change=OPT_TOL_CHANGE,
            history_size=LBFGS_HISTORY,
            line_search_fn='strong_wolfe',
        )

        # converge environment (no grad)
        with torch.no_grad():
            A, B, C, D, E, F = abcdef_to_ABCDEF(a, b, c, d, e, f, D_sq)
            (C21CD, C32EF, C13AB, T1F,  T2A,  T2B,  T3C,  T3D,  T1E,
             C21EB, C32AD, C13CF, T1D,  T2C,  T2F,  T3E,  T3B,  T1A,
             C21AF, C32CB, C13ED, T1B,  T2E,  T2D,  T3A,  T3F,  T1C) = \
                CTMRG_from_init_to_stop(
                    A, B, C, D, E, F, chi, D_sq,
                    CTM_MAX_STEPS, CTM_CONV_THR)

        # closure: cheap energy + backward through a..f only
        def closure():
            optimizer.zero_grad()
            loss = (
                energy_expectation_nearest_neighbor_6_bonds(
                    a, b, c, d, e, f,
                    Hs[0], Hs[1], Hs[2], Hs[3], Hs[4], Hs[5],
                    chi, D_bond,
                    C21CD, C32EF, C13AB, T1F, T2A, T2B, T3C, T3D, T1E)
              + energy_expectation_nearest_neighbor_other_3_bonds(
                    a, b, c, d, e, f,
                    Hs[6], Hs[7], Hs[8],
                    chi, D_bond,
                    C21AF, C32CB, C13ED, T1B, T2E, T2D, T3A, T3F, T1C)
            ).real
            loss.backward()
            return loss

        loss_val  = optimizer.step(closure)
        loss_item = loss_val.item()
        delta     = (loss_item - prev_loss) if prev_loss is not None else float('inf')
        elapsed   = time.perf_counter() - t_start

        print(f"    step {step:5d}  loss={loss_item:+.10f}  Δ={delta:+.3e}"
              f"  {elapsed:.0f}/{budget_seconds:.0f}s")
        loss_log.append({'step': step, 'loss': loss_item,
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

        if prev_loss is not None and abs(delta) < OPT_CONV_THRESHOLD:
            print(f"    Outer convergence at step {step} (Δ={delta:.2e})")
            break
        prev_loss = loss_item
        step += 1

    return (*best_abcdef, best_loss, step)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="10-12 h iPEPS benchmark: sweep D_bond (outer) × chi (inner)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--hours', type=float, default=11.0,
        help='Total wall-clock budget in hours.')
    parser.add_argument(
        '--D-bonds', default='2,3,4',
        help='Comma-separated list of D_bond values to sweep, in order.')
    parser.add_argument(
        '--chi-maxes', default=None,
        help='Comma-separated chi_max for each D_bond (must satisfy D²<chi≤D⁴). '
             'Defaults: 16,81,80 for D=2,3,4.')
    parser.add_argument(
        '--d-phys', type=int, default=2,
        help='Physical Hilbert-space dimension.')
    parser.add_argument(
        '--J', type=float, default=1.0,
        help='Isotropic Heisenberg coupling J (positive = AFM).')
    parser.add_argument(
        '--output-dir', default=None,
        help='Directory for checkpoints + plots (default: src_code/log/).')
    parser.add_argument(
        '--resume', default=None,
        help='Path to a .pt checkpoint to resume from (skips earlier (D,chi)).')
    parser.add_argument(
        '--noise', type=float, default=1e-3,
        help='Gaussian noise amplitude when padding tensors for D→D+1.')
    args = parser.parse_args()

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
            sched = DEFAULT_CHI_SCHEDULES[D]
        else:
            # geometric from D²+1 to chi_max in ~5 steps
            D_sq = D ** 2
            chi_min = D_sq + 1
            if chi_min >= chi_max:
                return [chi_max]
            import math
            n = 5
            ratio = (chi_max / chi_min) ** (1.0 / (n - 1))
            sched = sorted(set(
                [chi_min]
                + [round(chi_min * ratio**k) for k in range(1, n)]
                + [chi_max]))
            sched = [c for c in sched if D_sq < c <= chi_max]
        # ensure all entries satisfy the constraint and chi_max is last
        sched = [c for c in sched if D**2 < c <= chi_max]
        if not sched or sched[-1] != chi_max:
            sched.append(chi_max)
        return sorted(set(sched))

    schedules = {D: chi_schedule(D, chi_max_map[D]) for D in D_bond_list}

    # ── time budget per D_bond ────────────────────────────────────────────────
    # Use defaults for configured D values; normalise to sum to 1.
    raw_fracs = {D: DEFAULT_D_BUDGET_FRACS.get(D, 1.0) for D in D_bond_list}
    total_frac = sum(raw_fracs.values())
    d_budgets = {D: total_budget * raw_fracs[D] / total_frac
                 for D in D_bond_list}

    # ── output directory ──────────────────────────────────────────────────────
    output_dir = args.output_dir or os.path.join(
        os.path.dirname(__file__), '..', 'log')
    os.makedirs(output_dir, exist_ok=True)

    # ── Hamiltonians (isotropic; all 9 bonds equal) ───────────────────────────
    H  = build_heisenberg_H(args.J, d_PHYS)
    Hs = [H] * 9

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
        best_abcdef_by_D[resume_D] = (
            ckpt['a'], ckpt['b'], ckpt['c'],
            ckpt['d'], ckpt['e'], ckpt['f'])
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
        ramp_chis = [c for c in chis if c < chi_max]
        D_start_time = time.perf_counter()

        print(f"\n{'═'*76}")
        print(f"  D_bond = {D_bond}   D²={D_sq}  D⁴={D_bond**4}  "
              f"chi_max={chi_max}  budget={D_budget/3600:.2f} h")
        print(f"  chi schedule: {chis}")
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
            best_abcdef_by_D[D_bond] = tuple(
                pad_tensor(t, prev_D, D_bond, d_PHYS, args.noise)
                for t in prev_tensors)

        # current best tensors at this D (None = random init at first chi)
        cur_abcdef = best_abcdef_by_D.get(D_bond)

        # ── Ramp-up chi budget ────────────────────────────────────────────────
        n_ramp = len(ramp_chis)
        ramp_total = D_budget * RAMP_FRAC if n_ramp > 0 else 0.0
        prod_total = D_budget - ramp_total  # goes to chi_max

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

            # compute budget for this chi level
            is_production = (chi == chi_max)
            if is_production:
                # remaining D budget after ramp-up (may be more if ramp ended early)
                D_elapsed  = time.perf_counter() - D_start_time
                chi_budget = max(0.0, D_budget - D_elapsed)
            else:
                chi_budget = ramp_total / n_ramp if n_ramp > 0 else 0.0
                chi_budget = max(0.0, chi_budget)

            # RAM guard
            need_gb  = peak_ram_gb(chi, D_sq)
            avail_gb = free_ram_gb()
            if avail_gb < need_gb + RAM_SAFETY_GB:
                print(f"\n  ⚠  Skipping (D={D_bond}, chi={chi}): "
                      f"need {need_gb:.2f} GB + {RAM_SAFETY_GB} GB safety, "
                      f"only {avail_gb:.1f} GB free.")
                continue

            lbfgs_iters = LBFGS_MAX_ITER_PROD if is_production else LBFGS_MAX_ITER_RAMP

            print(f"\n  ┌── D={D_bond}  chi={chi}  "
                  f"{'[PRODUCTION]' if is_production else '[ramp-up]'}"
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
                lbfgs_max_iter=lbfgs_iters,
                init_abcdef=cur_abcdef,
                step_offset=global_step,
                best_path=best_path,
                latest_path=latest_path,
                loss_log=loss_log,
            )
            cur_abcdef = tuple(out[:6])  # warm-start for next chi

            # Clean energy evaluation
            print(f"  │  Evaluating energy at (D={D_bond}, chi={chi}) ...")
            energy = evaluate_energy_clean(
                *cur_abcdef, Hs, chi, D_bond)
            energy_per_bond = energy / 9

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
        best_abcdef_by_D[D_bond] = cur_abcdef

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
    print(f"  QMC reference (D→∞): E/bond ≈ −0.3646")
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
    ax.axhline(-0.3646, color='grey', ls='--', lw=1, label='QMC (D→∞) ≈ −0.3646')
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
        ax2.axhline(-0.3646, color='grey', ls='--', lw=1, label='QMC')
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
