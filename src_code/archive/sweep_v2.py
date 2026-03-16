"""
sweep_v2.py — Production sweep using core_unres_v2.optimize_ipeps
=================================================================
Runs iPEPS energy optimization for the Heisenberg honeycomb over a
grid of (D_bond, chi) values with multiple random seeds per point.

Usage (examples):
    python scripts/sweep_v2.py
    python scripts/sweep_v2.py --D 2 3 --chi 8 16 32 --seeds 5 --steps 100
    python scripts/sweep_v2.py --D 2 --chi 8 --seeds 3 --steps 50 --log /tmp/test.log

Per-step output comes from optimize_ipeps (verbose=True); a summary table
is printed at the end and optionally appended to --log.

QMC reference: E0/bond = -0.3646  (Sandvik 1997, Heisenberg honeycomb J=1)
"""

import sys, os, time, argparse, datetime
from itertools import groupby
from operator import itemgetter
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import core_unres_v2 as cv2

torch.set_default_dtype(torch.float64)
cv2.set_dtype(True)

QMC_REF  = -0.3646   # QMC E0/bond
NEEL_REF = -0.25     # exact Neel state (D=1 upper bound)


# ─────────────────────────────────────────────────────────────────────────────
def run_seed(D: int, chi: int, seed: int,
             max_opt_steps: int, max_ctm: int, verbose: bool = True,
             init_mode: str = 'neel', neel_noise: float = 0.1,
             sym_reg: float = 0.5):
    """Run one (D, chi, seed) using optimize_ipeps.  Returns summary dict."""
    H = cv2.build_heisenberg_H()
    torch.manual_seed(seed)

    t0 = time.time()
    _a, _b, _c, _d, _e, _f, history = cv2.optimize_ipeps(
        H=H,
        chi=chi,
        D_bond=D,
        max_ctm_steps=max_ctm,
        ctm_conv_thr=1e-7,
        max_opt_steps=max_opt_steps,
        lbfgs_max_iter=1,    # one gradient step per CTM convergence
        lbfgs_lr=1.0,
        lbfgs_history=20,
        opt_tol_grad=1e-7,
        opt_tol_change=1e-9,
        opt_conv_thr=1e-8,
        grad_clip_norm=1.0,
        sym_reg=sym_reg,
        init_abcdef=None,
        init_mode=init_mode,
        neel_noise=neel_noise,
        verbose=verbose,
    )
    elapsed = time.time() - t0

    min_E     = min(history) if history else float('nan')
    final_E   = history[-1]  if history else float('nan')
    below_qmc = sum(1 for ek in history if ek < QMC_REF)
    n_steps   = len(history)

    return dict(D=D, chi=chi, seed=seed,
                min_E=min_E, final_E=final_E,
                n_steps=n_steps, below_qmc=below_qmc,
                elapsed=elapsed)


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--D",    nargs="+", type=int, default=[2, 3],
                        help="Bond dimensions to sweep (default: 2 3)")
    parser.add_argument("--chi",  nargs="+", type=int, default=[8, 16, 32],
                        help="CTM bond dimensions to sweep (default: 8 16 32)")
    parser.add_argument("--seeds", type=int, default=5,
                        help="Number of random seeds per (D, chi) point (default: 5)")
    parser.add_argument("--steps", type=int, default=80,
                        help="Max optimization steps per seed (default: 80)")
    parser.add_argument("--max_ctm", type=int, default=200,
                        help="Max CTMRG iterations per call (default: 200)")
    parser.add_argument("--init", type=str, default='random',
                        choices=['neel', 'random'],
                        help="Initialization mode: 'neel' (Neel + noise) or 'random' (default: neel)")
    parser.add_argument("--noise", type=float, default=0.1,
                        help="Noise level for Neel init (default: 0.1)")
    parser.add_argument("--sym_reg", type=float, default=0.0,
                        help="Symmetry regularizer coefficient (default: 0.5)")
    parser.add_argument("--log", type=str, default=None,
                        help="Append summary table to this file")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-step verbose output from optimize_ipeps")
    args = parser.parse_args()

    seeds = list(range(args.seeds))
    results = []

    # ── Header ───────────────────────────────────────────────────────────────
    ts  = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    sep = "=" * 78
    hdr = (f"\n{sep}\n"
           f"  sweep_v2  |  {ts}\n"
           f"  D={args.D}  chi={args.chi}  seeds={args.seeds}  "
           f"max_steps={args.steps}  max_ctm={args.max_ctm}\n"
           f"  init={args.init}  noise={args.noise}  sym_reg={args.sym_reg}\n"
           f"  QMC ref = {QMC_REF}/bond  |  Neel ref = {NEEL_REF}/bond\n"
           f"{sep}")
    print(hdr, flush=True)

    # ── Main loop ─────────────────────────────────────────────────────────────
    for D in args.D:
        for chi in args.chi:
            if chi < D * D:
                print(f"\n  [skip] D={D} chi={chi}: chi < D**2={D*D}", flush=True)
                continue
            for seed in seeds:
                print(f"\n{'─'*60}", flush=True)
                print(f"  D={D}  chi={chi}  seed={seed}", flush=True)
                print(f"{'─'*60}", flush=True)

                r = run_seed(D, chi, seed,
                             max_opt_steps=args.steps,
                             max_ctm=args.max_ctm,
                             verbose=(not args.quiet),
                             init_mode=args.init,
                             neel_noise=args.noise,
                             sym_reg=args.sym_reg)
                results.append(r)

                flag = "  *** BELOW QMC ***" if r['below_qmc'] > 0 else ""
                print(f"\n  Summary  min={r['min_E']:+.8f}  "
                      f"final={r['final_E']:+.8f}  "
                      f"steps={r['n_steps']}  "
                      f"below_qmc={r['below_qmc']}  "
                      f"elapsed={r['elapsed']:.1f}s{flag}",
                      flush=True)

    # ── Summary table ─────────────────────────────────────────────────────────
    sep2 = "─" * 78
    col_hdr = (f"\n{sep2}\n"
               f"{'D':>3} {'chi':>5} {'seed':>5} | "
               f"{'min E/bond':>12} {'final E/bond':>13} | "
               f"{'steps':>6} {'<QMC':>5} | elapsed\n"
               + sep2)
    print(col_hdr, flush=True)

    table_lines = [col_hdr]
    for r in results:
        flag = " ***" if r['below_qmc'] > 0 else ""
        row = (f"{r['D']:>3} {r['chi']:>5} {r['seed']:>5} | "
               f"{r['min_E']:>+12.6f} {r['final_E']:>+13.6f} | "
               f"{r['n_steps']:>6} {r['below_qmc']:>5} | "
               f"{r['elapsed']:.0f}s{flag}")
        print(row, flush=True)
        table_lines.append(row)

    # ── Aggregate per (D, chi) ────────────────────────────────────────────────
    print(f"\n{sep2}", flush=True)
    print("  Aggregate  (best seed per config)", flush=True)
    print(sep2, flush=True)
    agg_lines = []
    for (D, chi), grp in groupby(sorted(results, key=itemgetter('D', 'chi')),
                                  key=itemgetter('D', 'chi')):
        grp = list(grp)
        best = min(grp, key=itemgetter('min_E'))
        total_below = sum(r['below_qmc'] for r in grp)
        aline = (f"  D={D} chi={chi:>3}  best_min={best['min_E']:+.8f}  "
                 f"best_seed={best['seed']}  total_below_qmc={total_below}")
        print(aline, flush=True)
        agg_lines.append(aline)

    footer = f"\nSweep complete  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    print(footer, flush=True)

    # ── Write log ─────────────────────────────────────────────────────────────
    if args.log:
        with open(args.log, "a") as fh:
            fh.write(hdr + "\n")
            for line in table_lines:
                fh.write(line + "\n")
            for line in agg_lines:
                fh.write(line + "\n")
            fh.write(footer + "\n")
        print(f"Results appended to {args.log}", flush=True)


if __name__ == "__main__":
    main()
