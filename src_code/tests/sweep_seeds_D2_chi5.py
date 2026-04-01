#!/usr/bin/env python3
"""
sweep_seeds_D2_chi5.py
======================
Robustness test: run D=2, chi=5 optimization from 100 independent random
seeds to check whether rel_cutoff=1e-8 in SVD backward (core_unrestricted.py)
gives reliable convergence across all random initialisations.

For each seed we:
  1. Init fresh random abcdef (no warm-start).
  2. Run optimize_at_chi for a fixed time budget.
  3. Record: seed, final loss, best loss, number of steps, whether NaN/Inf
     appeared, and gradient norm statistics.

Results are saved as a JSON + a histogram PDF.

Usage:
  python scripts/sweep_seeds_D2_chi5.py                   # defaults
  python scripts/sweep_seeds_D2_chi5.py --seeds 100 --budget 120
  python scripts/sweep_seeds_D2_chi5.py --double           # complex128
"""

import argparse
import collections
import datetime
import gc
import json
import os
import sys
import time
import traceback

# ── CPU threading (must precede numpy/torch imports) ──────────────────────────
_N_PHYSICAL_CORES = 4
os.environ.setdefault("OMP_NUM_THREADS", str(_N_PHYSICAL_CORES))
os.environ.setdefault("MKL_NUM_THREADS", str(_N_PHYSICAL_CORES))
os.environ.setdefault("MKL_DYNAMIC", "FALSE")
os.environ.setdefault("KMP_AFFINITY", "granularity=fine,compact,1,0")
os.environ.setdefault("KMP_BLOCKTIME", "0")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

torch.set_num_threads(_N_PHYSICAL_CORES)
torch.set_num_interop_threads(1)

from core_unrestricted import (
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
)

# ══════════════════════════════════════════════════════════════════════════════
# Fixed parameters for this test
# ══════════════════════════════════════════════════════════════════════════════

D_BOND = 2
D_SQ   = D_BOND ** 2
CHI    = 5
D_PHYS = 2
J      = 1.0
N_BONDS = 9

# Optimizer
LBFGS_MAX_ITER     = 15
LBFGS_LR           = 1.0
LBFGS_HISTORY      = LBFGS_MAX_ITER
OPT_TOL_GRAD       = 1e-9
OPT_TOL_CHANGE     = 3e-9
OPT_CONV_THRESHOLD = 1e-8

# CTMRG
CTM_MAX_STEPS     = 30
CTM_CONV_THR      = 2e-7
ENV_IDENTITY_INIT = False

# Init
INIT_NOISE = 3e-3

CDTYPE = torch.complex64  # overridden by --double


def timestamp() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ══════════════════════════════════════════════════════════════════════════════
# Single-seed optimisation (stripped-down version of optimize_at_chi)
# ══════════════════════════════════════════════════════════════════════════════

def run_single_seed(
    seed: int,
    Hs: list,
    budget_seconds: float,
) -> dict:
    """Run one optimisation from random init and return a results dict."""
    global CDTYPE

    torch.manual_seed(seed)
    np.random.seed(seed)

    a, b, c, d, e, f = initialize_abcdef('random', D_BOND, D_PHYS, INIT_NOISE)
    for t in (a, b, c, d, e, f):
        t.requires_grad_(True)

    best_loss    = float('inf')
    best_abcdef  = tuple(t.detach().clone() for t in (a, b, c, d, e, f))
    prev_loss    = None
    loss_history = collections.deque(maxlen=12)
    step         = 0
    t_start      = time.perf_counter()

    nan_count    = 0
    inf_count    = 0
    gnorm_list   = []
    loss_list    = []
    converged    = False
    error_msg    = None

    try:
        while True:
            elapsed = time.perf_counter() - t_start
            if elapsed >= budget_seconds:
                break

            # ── One L-BFGS outer step ─────────────────────────────────────────
            optimizer = torch.optim.LBFGS(
                [a, b, c, d, e, f],
                lr=LBFGS_LR,
                max_iter=LBFGS_MAX_ITER,
                history_size=LBFGS_HISTORY,
                tolerance_grad=OPT_TOL_GRAD,
                tolerance_change=OPT_TOL_CHANGE,
                line_search_fn='strong_wolfe',
            )

            _loss = None
            _ctm_steps = 0

            def closure():
                nonlocal _loss, _ctm_steps
                optimizer.zero_grad()

                aN = normalize_single_layer_tensor_for_double_layer(a)
                bN = normalize_single_layer_tensor_for_double_layer(b)
                cN = normalize_single_layer_tensor_for_double_layer(c)
                dN = normalize_single_layer_tensor_for_double_layer(d)
                eN = normalize_single_layer_tensor_for_double_layer(e)
                fN = normalize_single_layer_tensor_for_double_layer(f)

                A, B, C, Dt, E, F = abcdef_to_ABCDEF(aN, bN, cN, dN, eN, fN, D_SQ)
                all28 = CTMRG_from_init_to_stop(
                    A, B, C, Dt, E, F, CHI, D_SQ,
                    CTM_MAX_STEPS, CTM_CONV_THR, ENV_IDENTITY_INIT)

                (C21CD, C32EF, C13AB, T1F, T2A, T2B, T3C, T3D, T1E,
                 C21EB, C32AD, C13CF, T1D, T2C, T2F, T3E, T3B, T1A,
                 C21AF, C32CB, C13ED, T1B, T2E, T2D, T3A, T3F, T1C,
                 ctm_steps_inner) = all28
                _ctm_steps = ctm_steps_inner

                loss = (
                    energy_expectation_nearest_neighbor_3ebadcf_bonds(
                        aN, bN, cN, dN, eN, fN,
                        Hs[0], Hs[1], Hs[2],
                        CHI, D_BOND, D_PHYS,
                        C21CD, C32EF, C13AB, T1F, T2A, T2B, T3C, T3D, T1E)
                    + energy_expectation_nearest_neighbor_3afcbed_bonds(
                        aN, bN, cN, dN, eN, fN,
                        Hs[3], Hs[4], Hs[5],
                        CHI, D_BOND, D_PHYS,
                        C21EB, C32AD, C13CF, T1D, T2C, T2F, T3E, T3B, T1A)
                    + energy_expectation_nearest_neighbor_other_3_bonds(
                        aN, bN, cN, dN, eN, fN,
                        Hs[6], Hs[7], Hs[8],
                        CHI, D_BOND, D_PHYS,
                        C21AF, C32CB, C13ED, T1B, T2E, T2D, T3A, T3F, T1C)
                )

                loss.backward()
                _loss = loss
                return loss

            optimizer.step(closure)

            if _loss is None:
                break

            loss_item = _loss.detach().item()

            # Track NaN / Inf
            if not np.isfinite(loss_item):
                if np.isnan(loss_item):
                    nan_count += 1
                else:
                    inf_count += 1
                break  # abort this seed

            # Gradient norm
            gnorm = 0.0
            for t in (a, b, c, d, e, f):
                if t.grad is not None:
                    gnorm += t.grad.detach().norm().item() ** 2
            gnorm = gnorm ** 0.5
            gnorm_list.append(gnorm)
            loss_list.append(loss_item)

            if loss_item < best_loss:
                best_loss = loss_item
                best_abcdef = tuple(t.detach().clone() for t in (a, b, c, d, e, f))

            # Convergence check
            if prev_loss is not None:
                delta = abs(loss_item - prev_loss)
                if delta < OPT_CONV_THRESHOLD:
                    converged = True
                    break
            # Cycle detection
            if any(abs(loss_item - h) < 1e-10 for h in loss_history):
                converged = True  # cycle = effectively converged
                break

            loss_history.append(loss_item)
            prev_loss = loss_item
            step += 1

    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()

    elapsed = time.perf_counter() - t_start

    # ── Clean energy evaluation at best tensors ───────────────────────────────
    clean_energy = float('nan')
    try:
        with torch.no_grad():
            aN = normalize_single_layer_tensor_for_double_layer(best_abcdef[0])
            bN = normalize_single_layer_tensor_for_double_layer(best_abcdef[1])
            cN = normalize_single_layer_tensor_for_double_layer(best_abcdef[2])
            dN = normalize_single_layer_tensor_for_double_layer(best_abcdef[3])
            eN = normalize_single_layer_tensor_for_double_layer(best_abcdef[4])
            fN = normalize_single_layer_tensor_for_double_layer(best_abcdef[5])
            A, B, C, Dt, E, F = abcdef_to_ABCDEF(aN, bN, cN, dN, eN, fN, D_SQ)
            all27 = CTMRG_from_init_to_stop(
                A, B, C, Dt, E, F, CHI, D_SQ,
                CTM_MAX_STEPS, CTM_CONV_THR, ENV_IDENTITY_INIT)
            E3ebadcf = energy_expectation_nearest_neighbor_3ebadcf_bonds(
                aN, bN, cN, dN, eN, fN,
                Hs[0], Hs[1], Hs[2],
                CHI, D_BOND, D_PHYS, *all27[:9])
            E3afcbed = energy_expectation_nearest_neighbor_3afcbed_bonds(
                aN, bN, cN, dN, eN, fN,
                Hs[3], Hs[4], Hs[5],
                CHI, D_BOND, D_PHYS, *all27[9:18])
            E3 = energy_expectation_nearest_neighbor_other_3_bonds(
                aN, bN, cN, dN, eN, fN,
                Hs[6], Hs[7], Hs[8],
                CHI, D_BOND, D_PHYS, *all27[18:27])
            clean_energy = (E3ebadcf + E3afcbed + E3).item()
    except Exception:
        pass

    return {
        'seed': seed,
        'steps': step + 1,
        'best_loss': best_loss,
        'final_loss': loss_list[-1] if loss_list else float('nan'),
        'clean_energy': clean_energy,
        'clean_energy_per_bond': clean_energy / N_BONDS,
        'converged': converged,
        'nan_count': nan_count,
        'inf_count': inf_count,
        'gnorm_mean': float(np.mean(gnorm_list)) if gnorm_list else float('nan'),
        'gnorm_max': float(np.max(gnorm_list)) if gnorm_list else float('nan'),
        'gnorm_min': float(np.min(gnorm_list)) if gnorm_list else float('nan'),
        'elapsed_s': round(elapsed, 2),
        'error': error_msg,
        'loss_trajectory': loss_list,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global CDTYPE

    parser = argparse.ArgumentParser(
        description="Robustness test: D=2 chi=5, 100 random seeds",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seeds', type=int, default=100,
                        help='Number of independent random seeds to test.')
    parser.add_argument('--budget', type=float, default=120.0,
                        help='Time budget per seed in seconds.')
    parser.add_argument('--double', action='store_true', default=True,
                        help='Use float64/complex128.')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory (default: auto-timestamped).')
    args = parser.parse_args()

    # ── Precision ─────────────────────────────────────────────────────────────
    if args.double:
        CDTYPE = torch.complex128
        set_dtype(True)
    else:
        CDTYPE = torch.complex64
        set_dtype(False)

    # ── Output dir ────────────────────────────────────────────────────────────
    run_ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    default_dir = os.path.join('/home/chye/6ADctmrg/data/raw',
                               f'seed_sweep_D{D_BOND}_chi{CHI}_{run_ts}')
    out_dir = args.output_dir or default_dir
    os.makedirs(out_dir, exist_ok=True)

    # ── Hamiltonian ───────────────────────────────────────────────────────────
    H  = build_heisenberg_H(J, D_PHYS)
    Hs = [H] * 9

    # ── Banner ────────────────────────────────────────────────────────────────
    print("=" * 76)
    print(f"  Seed sweep: D={D_BOND}, chi={CHI}, d_phys={D_PHYS}, J={J}")
    print(f"  Seeds: {args.seeds}   Budget/seed: {args.budget:.0f}s")
    print(f"  Precision: {'complex128' if args.double else 'complex64'}")
    print(f"  rel_cutoff in core: 1e-8 (hardcoded)")
    print(f"  Output: {out_dir}")
    print(f"  Started: {timestamp()}")
    print("=" * 76)

    # ── Run seeds ─────────────────────────────────────────────────────────────
    results = []
    t_global = time.perf_counter()

    for i in range(args.seeds):
        seed = i
        print(f"\n── Seed {seed:3d}/{args.seeds} ", end='', flush=True)

        result = run_single_seed(seed, Hs, args.budget)
        results.append(result)

        status = "OK" if result['error'] is None else f"ERR: {result['error'][:50]}"
        e_bond = result['clean_energy_per_bond']
        nan_s = f" NaN={result['nan_count']}" if result['nan_count'] else ""
        inf_s = f" Inf={result['inf_count']}" if result['inf_count'] else ""
        print(f"  E/bond={e_bond:+.8f}  steps={result['steps']:3d}  "
              f"gnorm_max={result['gnorm_max']:.2e}  "
              f"conv={'Y' if result['converged'] else 'N'}  "
              f"{status}{nan_s}{inf_s}  ({result['elapsed_s']:.1f}s)")

        # Free memory between seeds
        gc.collect()

    total_time = time.perf_counter() - t_global

    # ══════════════════════════════════════════════════════════════════════════
    # Analysis
    # ══════════════════════════════════════════════════════════════════════════
    ok_results = [r for r in results if r['error'] is None and np.isfinite(r['clean_energy'])]
    failed     = [r for r in results if r['error'] is not None or not np.isfinite(r['clean_energy'])]

    energies    = [r['clean_energy_per_bond'] for r in ok_results]
    gnorm_maxes = [r['gnorm_max'] for r in ok_results]
    steps_list  = [r['steps'] for r in ok_results]

    print("\n" + "=" * 76)
    print("  SUMMARY")
    print("=" * 76)
    print(f"  Total seeds   : {args.seeds}")
    print(f"  Succeeded     : {len(ok_results)}")
    print(f"  Failed (NaN/Inf/Error): {len(failed)}")
    if failed:
        print(f"    Failed seeds: {[r['seed'] for r in failed]}")
    print()

    if energies:
        e_arr = np.array(energies)
        print(f"  E/bond statistics (N={len(e_arr)}):")
        print(f"    mean   = {e_arr.mean():+.10f}")
        print(f"    std    = {e_arr.std():.2e}")
        print(f"    min    = {e_arr.min():+.10f}")
        print(f"    max    = {e_arr.max():+.10f}")
        print(f"    median = {np.median(e_arr):+.10f}")
        print()

        # Convergence quality: how many reach near -0.3630?
        good_threshold = -0.30   # loose bar for D=2 chi=5
        n_good = sum(1 for e in energies if e < good_threshold)
        print(f"  Convergence (<{good_threshold} E/bond): {n_good}/{len(energies)}"
              f" ({100*n_good/len(energies):.0f}%)")

        # Spread
        near_best = sum(1 for e in energies if abs(e - e_arr.min()) < 0.01)
        print(f"  Within 0.01 of best: {near_best}/{len(energies)}"
              f" ({100*near_best/len(energies):.0f}%)")

    if gnorm_maxes:
        gn = np.array(gnorm_maxes)
        print(f"\n  Gradient norm max statistics:")
        print(f"    mean = {gn.mean():.2e}  max = {gn.max():.2e}  "
              f"min = {gn.min():.2e}")
        n_exploded = sum(1 for g in gnorm_maxes if g > 1e6)
        print(f"    Exploded (>1e6): {n_exploded}/{len(gnorm_maxes)}")

    print(f"\n  Total wall time: {total_time/60:.1f} min")
    print(f"  QMC reference: E/bond ≈ −0.3630 (D→∞)")
    print("=" * 76)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    # Strip loss_trajectory to keep JSON small
    results_slim = []
    for r in results:
        r2 = dict(r)
        r2['loss_trajectory_len'] = len(r2.pop('loss_trajectory', []))
        results_slim.append(r2)

    json_path = os.path.join(out_dir, 'seed_sweep_results.json')
    with open(json_path, 'w') as fp:
        json.dump({
            'D_bond': D_BOND, 'chi': CHI, 'd_phys': D_PHYS, 'J': J,
            'n_seeds': args.seeds, 'budget_per_seed_s': args.budget,
            'double': args.double, 'rel_cutoff_in_core': '1e-8',
            'total_time_s': round(total_time, 1),
            'n_ok': len(ok_results), 'n_failed': len(failed),
            'results': results_slim,
            'timestamp': timestamp(),
        }, fp, indent=2)
    print(f"\n  JSON → {json_path}")

    # ── Save full trajectories in a separate file ─────────────────────────────
    traj_path = os.path.join(out_dir, 'seed_sweep_trajectories.json')
    with open(traj_path, 'w') as fp:
        json.dump({r['seed']: r['loss_trajectory'] for r in results}, fp)
    print(f"  Trajectories → {traj_path}")

    # ══════════════════════════════════════════════════════════════════════════
    # Plots
    # ══════════════════════════════════════════════════════════════════════════

    if not energies:
        print("  No successful runs — skipping plots.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ── Panel 1: Histogram of E/bond ──────────────────────────────────────────
    ax1 = axes[0, 0]
    ax1.hist(energies, bins=30, edgecolor='black', alpha=0.7)
    ax1.axvline(-0.3630, color='red', ls='--', lw=1.5, label='QMC ref (D→∞)')
    ax1.set_xlabel('E/bond')
    ax1.set_ylabel('Count')
    ax1.set_title(f'Clean E/bond distribution (N={len(energies)})')
    ax1.legend()

    # ── Panel 2: E/bond vs seed ───────────────────────────────────────────────
    ax2 = axes[0, 1]
    seeds_ok = [r['seed'] for r in ok_results]
    ax2.scatter(seeds_ok, energies, s=12, alpha=0.7, c='tab:blue')
    ax2.axhline(-0.3630, color='red', ls='--', lw=1, label='QMC ref')
    ax2.set_xlabel('Seed')
    ax2.set_ylabel('E/bond')
    ax2.set_title('E/bond vs seed')
    ax2.legend()
    # Mark failed seeds
    if failed:
        for r in failed:
            ax2.axvline(r['seed'], color='red', alpha=0.3, lw=0.5)

    # ── Panel 3: Max gradient norm vs seed ────────────────────────────────────
    ax3 = axes[1, 0]
    ax3.scatter(seeds_ok, gnorm_maxes, s=12, alpha=0.7, c='tab:orange')
    ax3.set_yscale('log')
    ax3.set_xlabel('Seed')
    ax3.set_ylabel('Max ||grad||')
    ax3.set_title('Max gradient norm vs seed')

    # ── Panel 4: Loss trajectories (overlay all seeds) ────────────────────────
    ax4 = axes[1, 1]
    for r in results:
        traj = r.get('loss_trajectory', [])
        if traj:
            ax4.plot(range(len(traj)), [l / N_BONDS for l in traj],
                     lw=0.4, alpha=0.4)
    ax4.axhline(-0.3630, color='red', ls='--', lw=1, label='QMC ref')
    ax4.set_xlabel('Outer step')
    ax4.set_ylabel('Loss / N_BONDS')
    ax4.set_title(f'All {args.seeds} loss trajectories')
    ax4.legend()

    fig.suptitle(f'Seed sweep: D={D_BOND}, χ={CHI}, rel_cutoff=1e-8, '
                 f'{"complex128" if args.double else "complex64"}',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig_path = os.path.join(out_dir, 'seed_sweep_D2_chi5.pdf')
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"  Plot  → {fig_path}")


if __name__ == '__main__':
    main()
