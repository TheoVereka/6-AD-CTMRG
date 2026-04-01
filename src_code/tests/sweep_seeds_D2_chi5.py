#!/usr/bin/env python3
"""
sweep_seeds_D2_chi5.py
======================
Run D=2 chi=5 optimisation from N independent random seeds.

Internally this script imports main_sweep_CPU as a module and calls
optimize_at_chi() N times — one call per seed — with exactly the same
globals (LBFGS settings, CTM parameters, blowup recovery flag, etc.) that
main_sweep_CPU.main() would use when invoked as:

    python scripts/main_sweep_CPU.py --D-bonds 2  (chi schedule = [5])

The only difference is that each call gets a freshly seeded random init
(torch.manual_seed + np.random.seed) and runs for --budget seconds.

Usage:
    python tests/sweep_seeds_D2_chi5.py
    python tests/sweep_seeds_D2_chi5.py --seeds 100 --budget 120
    python tests/sweep_seeds_D2_chi5.py --no-recovery
    python tests/sweep_seeds_D2_chi5.py --no-double
"""

import argparse
import datetime
import gc
import json
import os
import sys
import time
import traceback

_N_PHYSICAL_CORES = 4
os.environ.setdefault("OMP_NUM_THREADS",  str(_N_PHYSICAL_CORES))
os.environ.setdefault("MKL_NUM_THREADS",  str(_N_PHYSICAL_CORES))
os.environ.setdefault("MKL_DYNAMIC",      "FALSE")
os.environ.setdefault("KMP_AFFINITY",     "granularity=fine,compact,1,0")
os.environ.setdefault("KMP_BLOCKTIME",    "0")

# ── add scripts/ to path so we can import main_sweep_CPU as a module ─────────
_THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS_DIR = os.path.join(_THIS_DIR, '..', 'scripts')
sys.path.insert(0, _SCRIPTS_DIR)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

# Do NOT call torch.set_num_threads / set_num_interop_threads here:
# main_sweep_CPU sets them at module level on import, and calling them
# a second time after parallel work has started raises a RuntimeError.

import main_sweep_CPU as _m   # the real thing — all globals live here

# Fixed parameters for this sweep (same as the first chi level of D=2 in main)
D_BOND  = 2
CHI     = 5
D_PHYS  = _m.D_PHYS
J       = _m.J_COUPLING
N_BONDS = _m.N_BONDS


def timestamp() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Robustness test: D=2 chi=5, N random seeds "
                    "(each run is identical to one main_sweep_CPU.py run for D=2, chi=5)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seeds',       type=int,   default=100,
                        help='Number of independent random seeds.')
    parser.add_argument('--budget',      type=float, default=120.0,
                        help='Wall-clock budget per seed in seconds.')
    parser.add_argument('--double',      action='store_true',
                        default=_m.USE_DOUBLE_PRECISION,
                        help='Use float64/complex128 (mirrors USE_DOUBLE_PRECISION in main).')
    parser.add_argument('--no-double',   dest='double', action='store_false')
    parser.add_argument('--no-recovery', action='store_true', default=False,
                        help='Set USE_BLOWUP_RECOVERY=False in main_sweep_CPU '
                             '(always use partial SVD, never recover).')
    parser.add_argument('--output-dir',  default=None)
    args = parser.parse_args()

    # ── configure main_sweep_CPU globals exactly as main() would ─────────────
    _m.CDTYPE = torch.complex128 if args.double else torch.complex64
    _m.set_dtype(args.double)
    if args.no_recovery:
        _m.USE_BLOWUP_RECOVERY = False

    # ── output dir ────────────────────────────────────────────────────────────
    run_ts  = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = args.output_dir or os.path.join(
        '/home/chye/6ADctmrg/data/raw',
        f'seed_sweep_D{D_BOND}_chi{CHI}_{run_ts}')
    os.makedirs(out_dir, exist_ok=True)

    # ── Hamiltonians (identical to main) ─────────────────────────────────────
    H  = _m.build_heisenberg_H(J, D_PHYS)
    Hs = [H] * N_BONDS

    # ── Banner ────────────────────────────────────────────────────────────────
    print("=" * 76)
    print(f"  Seed sweep using main_sweep_CPU.optimize_at_chi()")
    print(f"  D={D_BOND}, chi={CHI}, d_phys={D_PHYS}, J={J}")
    print(f"  Seeds: {args.seeds}   Budget/seed: {args.budget:.0f}s")
    print(f"  Precision: {'complex128' if args.double else 'complex64'}  "
          f"(USE_DOUBLE_PRECISION={_m.USE_DOUBLE_PRECISION})")
    print(f"  USE_BLOWUP_RECOVERY={_m.USE_BLOWUP_RECOVERY}   "
          f"OPTIMIZER={_m.OPTIMIZER}")
    print(f"  LBFGS: max_iter={_m.LBFGS_MAX_ITER}  lr={_m.LBFGS_LR}  "
          f"history={_m.LBFGS_HISTORY}")
    print(f"  CTM: max_steps={_m.CTM_MAX_STEPS}  conv_thr={_m.CTM_CONV_THR}")
    print(f"  OPT_CONV_THRESHOLD={_m.OPT_CONV_THRESHOLD}")
    print(f"  Output: {out_dir}")
    print(f"  Started: {timestamp()}")
    print("=" * 76)

    results   = []
    loss_logs = {}        # seed → list of step records from optimize_at_chi
    t_global  = time.perf_counter()

    for seed in range(args.seeds):
        print(f"\n{'─'*76}")
        print(f"  Seed {seed:3d}/{args.seeds}  [{timestamp()}]")
        print(f"{'─'*76}")

        torch.manual_seed(seed)
        np.random.seed(seed)

        # Reset full-SVD flag: each seed is a fresh launch of main, so the
        # first optimize_at_chi step must use full deterministic SVD (same as
        # _USE_FULL_SVD = True at module-load time when main is relaunched).
        _m._core._USE_FULL_SVD = True

        loss_log: list = []
        error_msg      = None
        best_loss      = float('nan')
        final_abcdef   = None

        try:
            # ── This is EXACTLY the call that main_sweep_CPU.main() makes ────
            *abcdef_out, best_loss, _steps_done = _m.optimize_at_chi(
                Hs,
                D_bond         = D_BOND,
                chi            = CHI,
                d_PHYS         = D_PHYS,
                budget_seconds = args.budget,
                lbfgs_max_iter = _m.LBFGS_MAX_ITER,
                init_abcdef    = None,    # random init, same as first chi of D=2 in main
                step_offset    = 0,
                best_path      = None,    # no per-seed checkpoints (use out_dir summary)
                latest_path    = None,
                loss_log       = loss_log,
                out_dir        = out_dir,
            )
            final_abcdef = tuple(abcdef_out)
            del abcdef_out
        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()

        loss_logs[seed] = loss_log

        # ── Clean energy (same as evaluate_energy_clean in main) ─────────────
        clean_energy = float('nan')
        if final_abcdef is not None:
            try:
                clean_energy = _m.evaluate_energy_clean(
                    *final_abcdef, Hs, CHI, D_BOND, D_PHYS)
            except Exception:
                pass

        clean_per_bond = clean_energy / N_BONDS if np.isfinite(clean_energy) else float('nan')
        loss_items     = [r['loss'] for r in loss_log]
        gnorms: list   = []   # optimize_at_chi doesn't export gnorms; omit

        result = {
            'seed':                  seed,
            'steps':                 len(loss_log),
            'best_loss':             best_loss,
            'final_loss':            loss_items[-1] if loss_items else float('nan'),
            'clean_energy':          clean_energy,
            'clean_energy_per_bond': clean_per_bond,
            'error':                 error_msg,
        }
        results.append(result)

        status = "OK" if error_msg is None else f"ERR:{error_msg[:40]}"
        nan_in_log = sum(1 for r in loss_log if not np.isfinite(r['loss']))
        print(f"  → E/bond={clean_per_bond:+.8f}  steps={len(loss_log)}  "
              f"best_loss={best_loss:+.8f}  {status}"
              + (f"  non-finite={nan_in_log}" if nan_in_log else ""))

        del final_abcdef
        gc.collect()

    total_time = time.perf_counter() - t_global

    # ══════════════════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════════════════
    ok      = [r for r in results if r['error'] is None and np.isfinite(r['clean_energy'])]
    failed  = [r for r in results if r['error'] is not None or not np.isfinite(r['clean_energy'])]
    energies = [r['clean_energy_per_bond'] for r in ok]

    print("\n" + "=" * 76)
    print("  SUMMARY")
    print("=" * 76)
    print(f"  Total / OK / Failed : {args.seeds} / {len(ok)} / {len(failed)}")
    if failed:
        print(f"  Failed seeds: {[r['seed'] for r in failed]}")

    if energies:
        e = np.array(energies)
        print(f"\n  E/bond  mean={e.mean():+.10f}  std={e.std():.2e}  "
              f"min={e.min():+.10f}  max={e.max():+.10f}")
        print(f"  Below −0.30 E/bond : {sum(1 for x in energies if x < -0.30)}/{len(energies)}")
        print(f"  Within 0.01 of best: "
              f"{sum(1 for x in energies if abs(x - e.min()) < 0.01)}/{len(energies)}")

    print(f"\n  Wall time: {total_time/60:.1f} min  |  QMC ref: E/bond ≈ −0.3630")
    print("=" * 76)

    # ── JSON ──────────────────────────────────────────────────────────────────
    json_path = os.path.join(out_dir, 'seed_sweep_results.json')
    with open(json_path, 'w') as fp:
        json.dump({
            'D_bond': D_BOND, 'chi': CHI, 'd_phys': D_PHYS, 'J': J,
            'n_seeds': args.seeds, 'budget_per_seed_s': args.budget,
            'double': args.double,
            'USE_BLOWUP_RECOVERY': _m.USE_BLOWUP_RECOVERY,
            'OPTIMIZER': _m.OPTIMIZER,
            'OPT_CONV_THRESHOLD': _m.OPT_CONV_THRESHOLD,
            'LBFGS_MAX_ITER': _m.LBFGS_MAX_ITER,
            'CTM_MAX_STEPS': _m.CTM_MAX_STEPS,
            'CTM_CONV_THR': _m.CTM_CONV_THR,
            'total_time_s': round(total_time, 1),
            'n_ok': len(ok), 'n_failed': len(failed),
            'results': results,
            'timestamp': timestamp(),
        }, fp, indent=2)
    print(f"\n  JSON → {json_path}")

    traj_path = os.path.join(out_dir, 'seed_sweep_trajectories.json')
    with open(traj_path, 'w') as fp:
        json.dump({seed: log for seed, log in loss_logs.items()}, fp)
    print(f"  Trajectories → {traj_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    if not energies:
        print("  No successful runs — skipping plots.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.hist(energies, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(-0.3630, color='red', ls='--', lw=1.5, label='QMC ref')
    ax.set_xlabel('E/bond'); ax.set_ylabel('Count')
    ax.set_title(f'E/bond distribution (N={len(energies)})'); ax.legend()

    ax = axes[0, 1]
    seeds_ok = [r['seed'] for r in ok]
    ax.scatter(seeds_ok, energies, s=12, alpha=0.7, c='tab:blue')
    ax.axhline(-0.3630, color='red', ls='--', lw=1, label='QMC ref')
    for r in failed:
        ax.axvline(r['seed'], color='red', alpha=0.3, lw=0.5)
    ax.set_xlabel('Seed'); ax.set_ylabel('E/bond')
    ax.set_title('E/bond vs seed'); ax.legend()

    ax = axes[1, 0]
    steps_ok = [r['steps'] for r in ok]
    ax.scatter(seeds_ok, steps_ok, s=12, alpha=0.7, c='tab:orange')
    ax.set_xlabel('Seed'); ax.set_ylabel('Outer steps')
    ax.set_title('Steps to convergence / budget exhaustion')

    ax = axes[1, 1]
    for seed, log in loss_logs.items():
        traj = [r['loss'] / N_BONDS for r in log if np.isfinite(r['loss'])]
        if traj:
            ax.plot(range(len(traj)), traj, lw=0.4, alpha=0.4)
    ax.axhline(-0.3630, color='red', ls='--', lw=1, label='QMC ref')
    ax.set_xlabel('Outer step'); ax.set_ylabel('Loss/N_BONDS')
    ax.set_title(f'All {args.seeds} loss trajectories'); ax.legend()

    fig.suptitle(
        f'Seed sweep D={D_BOND} χ={CHI}  '
        f'{"complex128" if args.double else "complex64"}  '
        f'recovery={_m.USE_BLOWUP_RECOVERY}',
        fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig_path = os.path.join(out_dir, 'seed_sweep_D2_chi5.pdf')
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"  Plot  → {fig_path}")


if __name__ == '__main__':
    main()
