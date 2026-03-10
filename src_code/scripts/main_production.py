#!/usr/bin/env python3
"""
main_production.py
==================
Production 10-12 h iPEPS optimisation for the AFM Heisenberg model on the
6-site honeycomb unit cell via AD-CTMRG.

Usage
-----
    python scripts/main_production.py            # all defaults (D=3, chi sweep)
    python scripts/main_production.py --D-bond 2 --chi 16 --hours 11
    python scripts/main_production.py --D-bond 3 --hours 11
    python scripts/main_production.py --D-bond 3 --chi 81 --hours 11

Strategy
--------
The script runs in two phases:

  Phase 1 – warm-up  (≤ WARMUP_FRAC of the time budget):
      Sweep chi geometrically from D²+1 up to chi_max / 2, spending
      WARMUP_STEPS outer L-BFGS steps per chi value.  Each chi level is
      warm-started from the previous one.  This provides a reasonably
      converged initial state for Phase 2.

  Phase 2 – production  (remaining time budget):
      Anchor at chi_max and run the outer L-BFGS loop until the wall-clock
      budget is exhausted or opt_conv_threshold is hit.  The best state found
      (lowest loss) is checkpointed to disk every SAVE_EVERY outer steps and
      at the end.

Physical constraint enforced on every chi:  D² < chi ≤ D⁴.

RAM guard
---------
The cost-limiting object is the SVD of the (chi·D²) × (chi·D²) corner
matrix (complex64, 8 bytes/element).  Peak memory for 3 corners + SVD
workspace ≈ 3 × 4 × (chi·D²)² × 8 bytes.  The script estimates this and
refuses to proceed if free RAM is insufficient (with a 1 GB safety margin).

Checkpoints
-----------
  log/production_D{D}_chi{chi}_best.pt    – best tensors (lowest loss)
  log/production_D{D}_chi{chi}_latest.pt  – most recent tensors
  log/production_results.json             – JSON summary of all energies

Author: AD-CTMRG project
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
    optmization_iPEPS,
)

# ── tuneable knobs (not exposed via CLI to keep the interface simple) ──────
WARMUP_FRAC  = 0.12   # fraction of total budget spent on warm-up chi sweep
WARMUP_STEPS = 60     # outer L-BFGS steps per warm-up chi value
SAVE_EVERY   = 25     # save checkpoint every N outer steps in Phase 2
RAM_SAFETY_GB = 2.0   # GB of free RAM to keep in reserve

# ── optimiser hyper-parameters ─────────────────────────────────────────────
LBFGS_MAX_ITER_WARMUP = 15   # L-BFGS sub-iters during warm-up (cheaper)
LBFGS_MAX_ITER_PROD   = 30   # L-BFGS sub-iters during production (richer)
LBFGS_LR              = 1.0
LBFGS_HISTORY         = 100
OPT_TOL_GRAD          = 1e-7
OPT_TOL_CHANGE        = 1e-9
OPT_CONV_THRESHOLD    = 1e-8

CTM_MAX_STEPS   = 90    # max CTMRG (1→2→3→1) cycles per environment update
CTM_CONV_THR    = 1e-7  # CTMRG convergence threshold


# ══════════════════════════════════════════════════════════════════════════════
# Helper utilities
# ══════════════════════════════════════════════════════════════════════════════

def timestamp() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def build_heisenberg_H(J: float = 1.0, d: int = 2) -> torch.Tensor:
    """J·(S·S) as a (d,d,d,d) tensor with index convention bra1,bra2,ket1,ket2."""
    sx = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64) / 2
    sy = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64) / 2
    sz = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64) / 2
    SdotS = (oe.contract("ij,kl->ikjl", sx, sx)
           + oe.contract("ij,kl->ikjl", sy, sy)
           + oe.contract("ij,kl->ikjl", sz, sz))
    return J * SdotS


def free_ram_gb() -> float:
    """Return available system RAM in GB via /proc/meminfo (Linux)."""
    try:
        import psutil
        return psutil.virtual_memory().available / 1e9
    except ImportError:
        pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    kb = int(line.split()[1])
                    return kb / 1e6
    except OSError:
        pass
    return float('inf')   # unknown – assume safe


def peak_ram_estimate_gb(chi: int, D_sq: int) -> float:
    """Rough peak RAM estimate for one CTMRG update at (chi, D_sq)."""
    n = chi * D_sq
    # 3 corner SVDs; each SVD of (n×n) complex64 needs ~4× matrix workspace
    return 3 * 4 * n * n * 8 / 1e9


def validate_chi(chi: int, D_bond: int) -> None:
    D_sq, D4 = D_bond ** 2, D_bond ** 4
    if not (D_sq < chi <= D4):
        raise ValueError(
            f"chi={chi} violates D²={D_sq} < chi ≤ D⁴={D4} "
            f"for D_bond={D_bond}."
        )


def geometric_chi_schedule(D_bond: int, chi_max: int, n_steps: int = 5) -> list[int]:
    """
    Return a list of chi values from D²+1 up to chi_max spaced geometrically.
    The last entry is always chi_max.
    """
    D_sq = D_bond ** 2
    chi_min = D_sq + 1
    if chi_min >= chi_max:
        return [chi_max]
    import math
    ratio = (chi_max / chi_min) ** (1.0 / max(n_steps - 1, 1))
    chis = sorted(set(
        [chi_min]
        + [min(chi_max, max(chi_min + 1, round(chi_min * ratio ** k)))
           for k in range(1, n_steps)]
        + [chi_max]
    ))
    return chis


def evaluate_energy_clean(a, b, c, d, e, f, Hs, chi, D_bond,
                           ctm_steps, ctm_thr) -> float:
    """Re-converge environment and return total energy as a Python float."""
    D_sq = D_bond ** 2
    with torch.no_grad():
        A, B, C, Dt, E, F = abcdef_to_ABCDEF(a, b, c, d, e, f, D_sq)
        all27 = CTMRG_from_init_to_stop(
            A, B, C, Dt, E, F, chi, D_sq, ctm_steps, ctm_thr)
        E6 = energy_expectation_nearest_neighbor_6_bonds(
            a, b, c, d, e, f,
            Hs[0], Hs[1], Hs[2], Hs[3], Hs[4], Hs[5],
            chi, D_bond, *all27[:9])
        E3 = energy_expectation_nearest_neighbor_other_3_bonds(
            a, b, c, d, e, f,
            Hs[6], Hs[7], Hs[8],
            chi, D_bond, *all27[18:27])
        return (E6 + E3).real.item()


def pad_tensor(t: torch.Tensor, old_D: int, new_D: int,
               d_PHYS: int, noise: float) -> torch.Tensor:
    out = noise * torch.randn(new_D, new_D, new_D, d_PHYS, dtype=torch.complex64)
    s = old_D
    out[:s, :s, :s, :] = t[:s, :s, :s, :]
    return normalize_tensor(out)


def save_checkpoint(path: str, abcdef, D_bond: int, chi: int,
                    loss: float, energy: float, step: int, log: list) -> None:
    torch.save({
        'a': abcdef[0], 'b': abcdef[1], 'c': abcdef[2],
        'd': abcdef[3], 'e': abcdef[4], 'f': abcdef[5],
        'D_bond': D_bond, 'chi': chi,
        'loss': loss, 'energy': energy,
        'step': step, 'timestamp': timestamp(),
        'log': log,
    }, path)


# ══════════════════════════════════════════════════════════════════════════════
# Core optimisation loop (single chi, using wallclock budget)
# ══════════════════════════════════════════════════════════════════════════════

def optimize_at_chi(
        Hs, D_bond: int, chi: int, d_PHYS: int,
        budget_seconds: float,
        lbfgs_max_iter: int,
        init_abcdef=None,
        step_offset: int = 0,
        best_path: str | None = None,
        latest_path: str | None = None,
        log_list: list | None = None,
) -> tuple:
    """
    Run the outer L-BFGS loop at fixed (D_bond, chi) until either
    `budget_seconds` wall-clock time has elapsed or opt convergence.

    Returns (a,b,c,d,e,f, best_loss, best_energy, n_steps_done).
    """
    if log_list is None:
        log_list = []

    D_sq = D_bond ** 2
    t_start = time.perf_counter()

    # ---------- initialise tensors -----------------------------------------
    if init_abcdef is not None:
        a, b, c, d, e, f = [
            t.detach().clone().to(torch.complex64) for t in init_abcdef]
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

        # ── normalise (scale redundancy) ────────────────────────────────────
        with torch.no_grad():
            a.data = normalize_tensor(a.data)
            b.data = normalize_tensor(b.data)
            c.data = normalize_tensor(c.data)
            d.data = normalize_tensor(d.data)
            e.data = normalize_tensor(e.data)
            f.data = normalize_tensor(f.data)

        # ── fresh L-BFGS (env changes every outer step → stale curvature) ──
        optimizer = torch.optim.LBFGS(
            [a, b, c, d, e, f],
            lr=LBFGS_LR,
            max_iter=lbfgs_max_iter,
            tolerance_grad=OPT_TOL_GRAD,
            tolerance_change=OPT_TOL_CHANGE,
            history_size=LBFGS_HISTORY,
            line_search_fn='strong_wolfe',
        )

        # ── converge environment ─────────────────────────────────────────────
        with torch.no_grad():
            A, B, C, D, E, F = abcdef_to_ABCDEF(a, b, c, d, e, f, D_sq)
            (C21CD, C32EF, C13AB, T1F,  T2A,  T2B,  T3C,  T3D,  T1E,
             C21EB, C32AD, C13CF, T1D,  T2C,  T2F,  T3E,  T3B,  T1A,
             C21AF, C32CB, C13ED, T1B,  T2E,  T2D,  T3A,  T3F,  T1C) = \
                CTMRG_from_init_to_stop(
                    A, B, C, D, E, F, chi, D_sq,
                    CTM_MAX_STEPS, CTM_CONV_THR)

        # ── L-BFGS closure ───────────────────────────────────────────────────
        def closure():
            optimizer.zero_grad()
            loss = (energy_expectation_nearest_neighbor_6_bonds(
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

        delta = (loss_item - prev_loss) if prev_loss is not None else float('inf')
        elapsed = time.perf_counter() - t_start
        print(f"    step {step:4d}  loss={loss_item:+.10f}  Δ={delta:+.3e}"
              f"  t={elapsed:.0f}s/{budget_seconds:.0f}s")

        record = {'step': step, 'loss': loss_item, 'elapsed': round(elapsed, 1)}
        log_list.append(record)

        # track best
        if loss_item < best_loss:
            best_loss   = loss_item
            best_abcdef = tuple(t.detach().clone() for t in (a, b, c, d, e, f))
            if best_path:
                save_checkpoint(
                    best_path, best_abcdef, D_bond, chi,
                    best_loss, float('nan'), step, log_list)

        # periodic save
        if latest_path and (step - step_offset) % SAVE_EVERY == 0 and step > step_offset:
            save_checkpoint(
                latest_path, best_abcdef, D_bond, chi,
                best_loss, float('nan'), step, log_list)

        # outer convergence
        if prev_loss is not None and abs(delta) < OPT_CONV_THRESHOLD:
            print(f"    Outer convergence at step {step} (Δloss={delta:.2e})")
            break
        prev_loss = loss_item
        step += 1

    return (*best_abcdef, best_loss, step)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Production iPEPS optimisation — 10-12 h run",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--D-bond', type=int, default=3,
                        help='Virtual bond dimension D of the iPEPS tensors.')
    parser.add_argument('--chi', type=int, default=None,
                        help='Environment bond dimension.  If omitted, uses D⁴ '
                             '(maximum allowed).  Validated: D²<chi≤D⁴.')
    parser.add_argument('--hours', type=float, default=11.0,
                        help='Total wall-clock budget in hours.')
    parser.add_argument('--d-phys', type=int, default=2,
                        help='Physical Hilbert-space dimension.')
    parser.add_argument('--J', type=float, default=1.0,
                        help='Isotropic Heisenberg coupling (positive=AFM).')
    parser.add_argument('--output-dir', default=None,
                        help='Directory for checkpoints and logs '
                             '(default: src_code/log/).')
    parser.add_argument('--resume', default=None,
                        help='Path to a .pt checkpoint file to resume from.')
    args = parser.parse_args()

    D_bond  = args.D_bond
    d_PHYS  = args.d_phys
    D_sq    = D_bond ** 2
    D4      = D_bond ** 4
    chi_max = args.chi if args.chi is not None else D4
    validate_chi(chi_max, D_bond)

    total_budget = args.hours * 3600.0   # seconds
    t_global_start = time.perf_counter()

    output_dir = args.output_dir or os.path.join(
        os.path.dirname(__file__), '..', 'log')
    os.makedirs(output_dir, exist_ok=True)

    # ── RAM guard ─────────────────────────────────────────────────────────────
    peak_gb = peak_ram_estimate_gb(chi_max, D_sq)
    free_gb = free_ram_gb()
    print(f"RAM check: need ~{peak_gb:.2f} GB peak, {free_gb:.1f} GB available.")
    if free_gb < peak_gb + RAM_SAFETY_GB:
        raise RuntimeError(
            f"Insufficient RAM: need ~{peak_gb+RAM_SAFETY_GB:.1f} GB but only "
            f"{free_gb:.1f} GB free.  Reduce chi or close other applications."
        )

    # ── Build Hamiltonians (isotropic; all 9 bonds identical) ────────────────
    H = build_heisenberg_H(args.J, d_PHYS)
    Hs = [H] * 9   # all bonds same coupling

    # ── Warm-up chi schedule ──────────────────────────────────────────────────
    # Geometrically spaced values *below* chi_max/2 used for phase 1.
    warmup_chis_raw = geometric_chi_schedule(D_bond, chi_max, n_steps=6)
    # Keep only chi values strictly less than chi_max for warm-up.
    warmup_chis = [c for c in warmup_chis_raw if c < chi_max]

    print("=" * 72)
    print(f"  Production iPEPS optimisation — AD-CTMRG")
    print(f"  D_bond={D_bond}  d_phys={d_PHYS}  chi_max={chi_max}"
          f"  (D²={D_sq}, D⁴={D4})")
    print(f"  J={args.J}  (positive=AFM Heisenberg)")
    print(f"  Time budget: {args.hours:.1f} h  ({total_budget:.0f} s)")
    print(f"  Warm-up chi schedule: {warmup_chis}  (Phase 1 ≤ {WARMUP_FRAC*100:.0f}% budget)")
    print(f"  Production chi:       {chi_max}  (Phase 2 = rest of budget)")
    print(f"  Output dir: {output_dir}")
    print(f"  Started: {timestamp()}")
    print("=" * 72)

    best_path   = os.path.join(output_dir,
                               f"production_D{D_bond}_chi{chi_max}_best.pt")
    latest_path = os.path.join(output_dir,
                               f"production_D{D_bond}_chi{chi_max}_latest.pt")
    results_path = os.path.join(output_dir, "production_results.json")

    all_log   = []
    init_abcdef = None
    global_step = 0

    # ── Resume from checkpoint ────────────────────────────────────────────────
    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu')
        init_abcdef = (ckpt['a'], ckpt['b'], ckpt['c'],
                       ckpt['d'], ckpt['e'], ckpt['f'])
        global_step = ckpt.get('step', 0) + 1
        print(f"  Resumed from {args.resume}  (step={ckpt.get('step',0)}, "
              f"loss={ckpt.get('loss',float('nan')):.6f})")

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 1 — warm-up sweep over lower chi values
    # ══════════════════════════════════════════════════════════════════════════
    warmup_budget = total_budget * WARMUP_FRAC
    phase1_start  = time.perf_counter()
    energy_log: list[dict] = []

    if warmup_chis and init_abcdef is None:
        print(f"\n{'─'*72}")
        print(f"  PHASE 1 — warm-up  (budget ≤ {warmup_budget:.0f} s)")
        print(f"{'─'*72}")

        for wchi in warmup_chis:
            chi_elapsed = time.perf_counter() - phase1_start
            remaining   = warmup_budget - chi_elapsed
            if remaining <= 0:
                print(f"  Phase 1 budget exhausted, skipping chi={wchi}.")
                break

            # RAM check per chi
            need_gb = peak_ram_estimate_gb(wchi, D_sq)
            avail   = free_ram_gb()
            if avail < need_gb + RAM_SAFETY_GB:
                print(f"  Skipping chi={wchi}: "
                      f"need {need_gb:.2f} GB, only {avail:.1f} GB free.")
                continue

            per_chi_budget = min(remaining, warmup_budget / max(len(warmup_chis), 1))
            print(f"\n  chi={wchi}  budget={per_chi_budget:.0f}s  [{timestamp()}]")

            *warmup_out, wloss, global_step = optimize_at_chi(
                Hs, D_bond, wchi, d_PHYS,
                budget_seconds=per_chi_budget,
                lbfgs_max_iter=LBFGS_MAX_ITER_WARMUP,
                init_abcdef=init_abcdef,
                step_offset=global_step,
                log_list=all_log,
            )
            init_abcdef = tuple(warmup_out[:6])   # warm-start for next chi

            # evaluate energy cleanly at this chi
            wenergy = evaluate_energy_clean(
                *init_abcdef, Hs, wchi, D_bond, CTM_MAX_STEPS, CTM_CONV_THR)
            energy_log.append({'phase': 1, 'chi': wchi,
                                'loss': wloss, 'energy': wenergy,
                                'energy_per_bond': wenergy / 9})
            print(f"  ✓ chi={wchi}  E={wenergy:+.8f}  E/bond={wenergy/9:+.6f}")

    elif init_abcdef is None:
        print("  No warm-up chi values < chi_max → starting Phase 2 from random init.")

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 2 — production run at chi_max
    # ══════════════════════════════════════════════════════════════════════════
    phase2_elapsed_so_far = time.perf_counter() - t_global_start
    phase2_budget = total_budget - phase2_elapsed_so_far

    print(f"\n{'─'*72}")
    print(f"  PHASE 2 — production at chi={chi_max}  (budget={phase2_budget:.0f}s "
          f"= {phase2_budget/3600:.2f}h)  [{timestamp()}]")
    print(f"{'─'*72}\n")

    # RAM check for chi_max
    need_gb = peak_ram_estimate_gb(chi_max, D_sq)
    avail   = free_ram_gb()
    if avail < need_gb + RAM_SAFETY_GB:
        raise RuntimeError(
            f"Insufficient RAM for chi_max={chi_max}: "
            f"need {need_gb+RAM_SAFETY_GB:.2f} GB, "
            f"only {avail:.1f} GB free."
        )

    *prod_out, prod_loss, global_step = optimize_at_chi(
        Hs, D_bond, chi_max, d_PHYS,
        budget_seconds=phase2_budget,
        lbfgs_max_iter=LBFGS_MAX_ITER_PROD,
        init_abcdef=init_abcdef,
        step_offset=global_step,
        best_path=best_path,
        latest_path=latest_path,
        log_list=all_log,
    )
    best_abcdef_prod = tuple(prod_out[:6])

    # final clean energy evaluation
    print(f"\n  Final clean energy evaluation at chi={chi_max} ...")
    final_energy = evaluate_energy_clean(
        *best_abcdef_prod, Hs, chi_max, D_bond, CTM_MAX_STEPS, CTM_CONV_THR)
    energy_log.append({'phase': 2, 'chi': chi_max,
                       'loss': prod_loss, 'energy': final_energy,
                       'energy_per_bond': final_energy / 9})

    # final checkpoint
    save_checkpoint(best_path, best_abcdef_prod, D_bond, chi_max,
                    prod_loss, final_energy, global_step, all_log)
    save_checkpoint(latest_path, best_abcdef_prod, D_bond, chi_max,
                    prod_loss, final_energy, global_step, all_log)

    # ══════════════════════════════════════════════════════════════════════════
    # Results and plots
    # ══════════════════════════════════════════════════════════════════════════
    total_elapsed = time.perf_counter() - t_global_start

    print("\n" + "=" * 72)
    print("  RESULTS SUMMARY")
    print("=" * 72)
    print(f"  {'chi':>6}  {'E_total':>18}  {'E/bond':>14}  phase")
    print(f"  {'---':>6}  {'-------':>18}  {'------':>14}  -----")
    for entry in energy_log:
        print(f"  {entry['chi']:6d}  {entry['energy']:+18.10f}  "
              f"{entry['energy_per_bond']:+14.10f}  {entry['phase']}")
    print(f"\n  Best E/bond = {final_energy/9:+.10f}  (D={D_bond}, chi={chi_max})")
    print(f"  QMC reference (honeycomb AFM Heisenberg, D→∞): E/bond ≈ −0.3646")
    print(f"  Total wall time: {total_elapsed/3600:.2f} h ({total_elapsed:.0f} s)")
    print(f"  Best checkpoint: {best_path}")
    print(f"  Finished: {timestamp()}")
    print("=" * 72)

    # JSON results
    json_out = {
        'D_bond': D_bond, 'chi_max': chi_max, 'd_phys': d_PHYS,
        'J': args.J, 'hours_budget': args.hours,
        'total_elapsed_h': round(total_elapsed / 3600, 3),
        'total_steps': global_step,
        'best_loss': prod_loss,
        'final_energy': final_energy,
        'final_energy_per_bond': final_energy / 9,
        'energy_log': energy_log,
        'full_loss_log': all_log,
        'timestamp': timestamp(),
    }
    with open(results_path, 'w') as fp:
        json.dump(json_out, fp, indent=2)
    print(f"  Results saved → {results_path}")

    # Loss curve plot
    if all_log:
        steps   = [r['step'] for r in all_log]
        losses  = [r['loss'] for r in all_log]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(steps, losses, 'b-', lw=0.8, alpha=0.7)
        ax.set_xlabel('Outer step')
        ax.set_ylabel('Loss (total energy)')
        ax.set_title(f'iPEPS optimisation  D={D_bond}, χ={chi_max}')
        ax.grid(True, alpha=0.3)
        fig_path = os.path.join(output_dir,
                                f"production_D{D_bond}_chi{chi_max}_loss.pdf")
        fig.savefig(fig_path, bbox_inches='tight')
        plt.close(fig)
        print(f"  Loss curve → {fig_path}")


if __name__ == '__main__':
    main()
