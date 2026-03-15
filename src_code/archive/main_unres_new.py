#!/usr/bin/env python3
"""
main_unres_new.py
=================
iPEPS optimization for the AFM Heisenberg model on the 6-site honeycomb
unit cell using real-tensor AD-CTMRG.

Sweep structure:
    for D_bond in [2, 3, ...]:
        for chi in chi_schedule(D):
            optimize until time budget or convergence

Each (D, chi) level warm-starts from the previous.
D → D+1 warm-start: best tensors padded + small noise.

Usage:
    python scripts/main_unres_new.py                    # defaults
    python scripts/main_unres_new.py --hours 2
    python scripts/main_unres_new.py --D-bonds 2 --chi-schedule 5,8,16
    python scripts/main_unres_new.py --resume log/new_D2_chi16_best.pt
"""

import argparse
import datetime
import json
import os
import sys
import time

# ── CPU threading (set before torch import) ──────────────────────────────────
_N_CORES = min(os.cpu_count() or 4, 8)
os.environ.setdefault("OMP_NUM_THREADS", str(_N_CORES))
os.environ.setdefault("MKL_NUM_THREADS", str(_N_CORES))
os.environ.setdefault("MKL_DYNAMIC", "FALSE")
os.environ.setdefault("KMP_BLOCKTIME", "0")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import opt_einsum as oe  # noqa: F401

torch.set_num_threads(_N_CORES)
torch.set_num_interop_threads(1)

from src_code.archive.core_unres_new import (
    set_dtype,
    normalize_tensor,
    build_heisenberg_H,
    initialize_abcdef,
    abcdef_to_ABCDEF,
    ctmrg,
    energy_3ebadcf,
    energy_3afcbed,
    energy_other_3,
    optimize_ipeps,
)

# ══════════════════════════════════════════════════════════════════════════════
# Default hyperparameters
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_CHI_SCHEDULES = {
    2: [5, 8, 12, 16],
    3: [10, 18, 32, 57],
    4: [17, 30, 50, 80],
}

DEFAULT_D_BUDGET_FRACS = {2: 0.10, 3: 0.45, 4: 0.45}

# L-BFGS
LBFGS_MAX_ITER = 30
LBFGS_LR = 1.0
LBFGS_HISTORY = 30
OPT_TOL_GRAD = 1e-7
OPT_TOL_CHANGE = 1e-9
OPT_CONV_THRESHOLD = 1e-8

# CTMRG
CTM_MAX_STEPS = 100
CTM_CONV_THR = 1e-7

# Physical model
J_COUPLING = 1.0
D_PHYS = 2
N_BONDS = 9

# Warm-start
PAD_NOISE = 1e-3
SAVE_EVERY = 10


# ══════════════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════════════

def timestamp() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def pad_tensor(t: torch.Tensor, new_D: int, noise: float = PAD_NOISE) -> torch.Tensor:
    """Pad iPEPS tensor from D_old to new_D with small Gaussian noise."""
    D_old = t.shape[0]
    d_phys = t.shape[-1]
    new_t = noise * torch.randn(new_D, new_D, new_D, d_phys, dtype=t.dtype)
    new_t[:D_old, :D_old, :D_old, :] = t
    return new_t


def compute_energy(a, b, c, d_t, e, f, H, chi, D_bond, env1, env2, env3):
    """Compute total energy from all 3 environment types.  Returns (total, e/bond)."""
    e1 = energy_3ebadcf(a, b, c, d_t, e, f, H, H, H, chi, D_bond, *env1)
    e2 = energy_3afcbed(a, b, c, d_t, e, f, H, H, H, chi, D_bond, *env2)
    e3 = energy_other_3(a, b, c, d_t, e, f, H, H, H, chi, D_bond, *env3)
    total = (e1 + e2 + e3)
    return total, total / N_BONDS


def save_checkpoint(path, tensors, D_bond, chi, step, loss, best_loss, loss_hist):
    """Save a checkpoint dict."""
    a, b, c, d_t, e, f = tensors
    torch.save({
        'a': a.detach().cpu(), 'b': b.detach().cpu(),
        'c': c.detach().cpu(), 'd': d_t.detach().cpu(),
        'e': e.detach().cpu(), 'f': f.detach().cpu(),
        'D_bond': D_bond, 'chi': chi, 'step': step,
        'loss': loss, 'best_loss': best_loss,
        'loss_history': loss_hist,
    }, path)


def load_checkpoint(path):
    """Load tensors from a checkpoint."""
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    tensors = tuple(ckpt[k] for k in 'abcdef')
    return tensors, ckpt


# ══════════════════════════════════════════════════════════════════════════════
# Core optimization at a single (D, chi) level
# ══════════════════════════════════════════════════════════════════════════════

def optimize_at_chi(
    tensors, H, chi, D_bond, time_budget_sec, log_dir,
    label='', verbose=True,
):
    """Run alternating CTMRG + L-BFGS until time budget exhausted.

    Args:
        tensors: (a, b, c, d_t, e, f) initial site tensors.
        H: Hamiltonian tensor (bra1,bra2,ket1,ket2 convention).
        chi: environment bond dimension.
        D_bond: virtual bond dimension.
        time_budget_sec: wall-clock time budget in seconds.
        log_dir: directory for checkpoints.
        label: prefix for checkpoint files and print.
        verbose: print progress.

    Returns: (best_tensors, best_loss, loss_history)
    """
    D_sq = D_bond ** 2
    a, b, c, d_t, e, f = [t.detach().clone().to(torch.float64) for t in tensors]
    for t in [a, b, c, d_t, e, f]:
        t.requires_grad_(True)

    best_loss = float('inf')
    best_tensors = tuple(t.detach().clone() for t in [a, b, c, d_t, e, f])
    prev_loss = None
    loss_history = []

    t_start = time.time()
    step = 0

    while True:
        elapsed = time.time() - t_start
        if elapsed >= time_budget_sec:
            break

        # Normalize
        with torch.no_grad():
            a.data = normalize_tensor(a.data)
            b.data = normalize_tensor(b.data)
            c.data = normalize_tensor(c.data)
            d_t.data = normalize_tensor(d_t.data)
            e.data = normalize_tensor(e.data)
            f.data = normalize_tensor(f.data)

        # Fresh L-BFGS
        optimizer = torch.optim.LBFGS(
            [a, b, c, d_t, e, f],
            lr=LBFGS_LR, max_iter=LBFGS_MAX_ITER,
            tolerance_grad=OPT_TOL_GRAD, tolerance_change=OPT_TOL_CHANGE,
            history_size=LBFGS_HISTORY, line_search_fn='strong_wolfe',
        )

        # Converge CTMRG
        with torch.no_grad():
            DL = abcdef_to_ABCDEF(a, b, c, d_t, e, f, D_sq)
            env1, env2, env3, ctm_steps = ctmrg(*DL, chi, D_sq, CTM_MAX_STEPS, CTM_CONV_THR)

        # Unpack envs for energy closure
        env1_t = env1
        env2_t = env2
        env3_t = env3

        def closure():
            optimizer.zero_grad()
            e1 = energy_3ebadcf(a, b, c, d_t, e, f, H, H, H, chi, D_bond, *env1_t)
            e2 = energy_3afcbed(a, b, c, d_t, e, f, H, H, H, chi, D_bond, *env2_t)
            e3 = energy_other_3(a, b, c, d_t, e, f, H, H, H, chi, D_bond, *env3_t)
            loss = e1 + e2 + e3
            loss.backward()
            return loss

        loss_val = optimizer.step(closure)
        loss_item = loss_val.item()
        loss_history.append(loss_item)
        e_bond = loss_item / N_BONDS

        delta = (loss_item - prev_loss) if prev_loss is not None else float('inf')

        if verbose:
            eta = time_budget_sec - (time.time() - t_start)
            print(f"  {label} step {step:4d}  E/bond={e_bond:+.10f}  "
                  f"Δ={delta:.3e}  CTM={ctm_steps}  "
                  f"eta={eta:.0f}s")

        if loss_item < best_loss:
            best_loss = loss_item
            best_tensors = tuple(t.detach().clone() for t in [a, b, c, d_t, e, f])
            save_checkpoint(
                os.path.join(log_dir, f'{label}best.pt'),
                [a, b, c, d_t, e, f], D_bond, chi, step, loss_item, best_loss, loss_history)

        if step > 0 and step % SAVE_EVERY == 0:
            save_checkpoint(
                os.path.join(log_dir, f'{label}latest.pt'),
                [a, b, c, d_t, e, f], D_bond, chi, step, loss_item, best_loss, loss_history)

        if prev_loss is not None and abs(delta) < OPT_CONV_THRESHOLD:
            if verbose:
                print(f"  {label} converged at step {step} (|Δ|={abs(delta):.3e})")
            break

        prev_loss = loss_item
        step += 1

    if verbose:
        print(f"  {label} done: {step} steps, best E/bond={best_loss/N_BONDS:.10f}")

    return best_tensors, best_loss, loss_history


# ══════════════════════════════════════════════════════════════════════════════
# Main sweep
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='iPEPS Heisenberg honeycomb sweep')
    parser.add_argument('--hours', type=float, default=1.0,
                        help='Total wall-clock time budget in hours (default: 1)')
    parser.add_argument('--D-bonds', type=str, default='2',
                        help='Comma-separated D_bond values (default: "2")')
    parser.add_argument('--chi-schedule', type=str, default=None,
                        help='Comma-separated chi values (overrides default schedule)')
    parser.add_argument('--double', action='store_true', default=True,
                        help='Use float64 precision (default: True)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint for warm-start')
    parser.add_argument('--log-dir', type=str, default=None,
                        help='Output directory (default: log/sweep_YYYYMMDD_HHMMSS)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    args = parser.parse_args()

    # Setup
    torch.manual_seed(args.seed)
    set_dtype(args.double)

    D_bonds = [int(x) for x in args.D_bonds.split(',')]
    total_budget = args.hours * 3600

    if args.log_dir:
        log_dir = args.log_dir
    else:
        log_dir = os.path.join(
            os.path.dirname(__file__), '..', 'log',
            f'sweep_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(log_dir, exist_ok=True)

    H = build_heisenberg_H(J=J_COUPLING)

    # Build chi schedules
    chi_schedules = {}
    for D in D_bonds:
        if args.chi_schedule:
            chi_schedules[D] = [int(x) for x in args.chi_schedule.split(',')]
        elif D in DEFAULT_CHI_SCHEDULES:
            chi_schedules[D] = DEFAULT_CHI_SCHEDULES[D]
        else:
            D_sq = D * D
            D4 = D_sq * D_sq
            chi_schedules[D] = [D_sq + 1, min(D4, max(D_sq + 1, D4 // 4)),
                                min(D4, max(D_sq + 1, D4 // 2)), D4]

    # Compute time budgets
    frac_sum = sum(DEFAULT_D_BUDGET_FRACS.get(D, 0.3) for D in D_bonds)
    d_budgets = {}
    for D in D_bonds:
        frac = DEFAULT_D_BUDGET_FRACS.get(D, 0.3) / frac_sum
        d_budgets[D] = total_budget * frac

    print('=' * 72)
    print(f'iPEPS Heisenberg Honeycomb Sweep (Real Tensors)')
    print(f'  start:    {timestamp()}')
    print(f'  budget:   {args.hours:.2f} h ({total_budget:.0f} s)')
    print(f'  D_bonds:  {D_bonds}')
    print(f'  dtype:    float64' if args.double else '  dtype:    float32')
    print(f'  seed:     {args.seed}')
    print(f'  log_dir:  {log_dir}')
    for D in D_bonds:
        print(f'  D={D}: chi={chi_schedules[D]}, budget={d_budgets[D]:.0f}s')
    print('=' * 72)

    # Save config
    config = {
        'hours': args.hours, 'D_bonds': D_bonds,
        'chi_schedules': chi_schedules, 'seed': args.seed,
        'LBFGS_MAX_ITER': LBFGS_MAX_ITER, 'LBFGS_LR': LBFGS_LR,
        'CTM_MAX_STEPS': CTM_MAX_STEPS, 'CTM_CONV_THR': CTM_CONV_THR,
    }
    with open(os.path.join(log_dir, 'config.json'), 'w') as fh:
        json.dump(config, fh, indent=2)

    # Load warm-start if provided
    warm_tensors = None
    warm_D = None
    if args.resume:
        warm_tensors, ckpt = load_checkpoint(args.resume)
        warm_D = ckpt.get('D_bond', None)
        print(f'[resume] Loaded {args.resume} (D={warm_D})')

    results = []
    t_sweep_start = time.time()

    for D in D_bonds:
        chis = chi_schedules[D]
        D_budget = d_budgets[D]
        chi_budget = D_budget / len(chis)

        # Initialize or warm-start tensors
        if warm_tensors is not None:
            src_D = warm_tensors[0].shape[0]
            if src_D == D:
                tensors = warm_tensors
            elif src_D < D:
                tensors = tuple(pad_tensor(t, D) for t in warm_tensors)
                print(f'[pad] D={src_D} → {D}')
            else:
                tensors = initialize_abcdef(D, D_PHYS)
                print(f'[init] Fresh D={D} (warm D={src_D} > target)')
        else:
            tensors = initialize_abcdef(D, D_PHYS)
            print(f'[init] Fresh D={D}')

        for chi in chis:
            # Check time
            elapsed = time.time() - t_sweep_start
            remaining = total_budget - elapsed
            if remaining < 10:
                print(f'[skip] D={D} chi={chi} — time budget exhausted')
                break

            budget = min(chi_budget, remaining)
            label = f'D{D}_chi{chi}_'

            print(f'\n{"─" * 60}')
            print(f'[{timestamp()}] D={D}, chi={chi}, budget={budget:.0f}s')
            print(f'{"─" * 60}')

            best_t, best_loss, hist = optimize_at_chi(
                tensors, H, chi, D, budget, log_dir, label=label, verbose=True)

            e_bond = best_loss / N_BONDS
            results.append({
                'D_bond': D, 'chi': chi,
                'E_bond': float(e_bond),
                'E_total': float(best_loss),
                'n_steps': len(hist),
                'timestamp': timestamp(),
            })

            # Update warm-start
            tensors = best_t
            warm_tensors = best_t

        warm_tensors = tensors
        warm_D = D

    # Save results
    results_path = os.path.join(log_dir, 'results.json')
    with open(results_path, 'w') as fh:
        json.dump(results, fh, indent=2)

    # Print summary
    print(f'\n{"=" * 72}')
    print(f'SWEEP COMPLETE  {timestamp()}')
    print(f'Total time: {(time.time() - t_sweep_start)/3600:.2f} h')
    print(f'{"=" * 72}')
    print(f'{"D":>3s} {"chi":>5s} {"E/bond":>14s} {"steps":>6s}')
    print(f'{"-"*3:>3s} {"-"*5:>5s} {"-"*14:>14s} {"-"*6:>6s}')
    for r in results:
        print(f'{r["D_bond"]:3d} {r["chi"]:5d} {r["E_bond"]:+14.10f} {r["n_steps"]:6d}')
    print(f'\nQMC reference: E/bond ≈ −0.3646 (AFM Heisenberg on honeycomb)')
    print(f'Results saved to {results_path}')


if __name__ == '__main__':
    main()
