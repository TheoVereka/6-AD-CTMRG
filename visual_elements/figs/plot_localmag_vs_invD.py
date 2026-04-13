"""
Plot mean local magnetization |m| vs 1/D for the 2D Heisenberg model.

For each D value:
  - Parse localmag_envK_X lines from the notebook result files.
  - Average over env1/env2/env3 for each site to obtain one |m| per site.
  - Compute the mean and standard error (std / sqrt(6)) across the 6 sites A–F.
  - Plot mean ± SE vs 1/D.

Output: saves both PDF and PNG next to this script (or to --outdir if given).
"""

import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt

# ── regex ─────────────────────────────────────────────────────────────────────
LOCALMAG_RE = re.compile(
    r'localmag_env(\d+)_([A-F])\s+\|m\|=([+\-]?\d+\.\d+e[+\-]\d+)'
)
FNAME_RE = re.compile(r'^D_(\d+)_chi_')

SITES = list('ABCDEF')
ENVS  = [1, 2, 3]

# ── helpers ───────────────────────────────────────────────────────────────────

def parse_file(fpath: str) -> dict[tuple[str, int], float]:
    """Return {(site, env): |m|} for every localmag line in fpath."""
    data: dict[tuple[str, int], float] = {}
    with open(fpath) as fh:
        for line in fh:
            m = LOCALMAG_RE.search(line)
            if m:
                env  = int(m.group(1))
                site = m.group(2)
                val  = float(m.group(3))
                data[(site, env)] = val
    return data


def load_all(data_dir: str) -> tuple[dict[int, list[float]], dict[int, int]]:
    """
    Scan data_dir for D_*_chi_* txt files.
    Returns ({D: [per_site_mean_A, ..., per_site_mean_F]}, {D: chi}).
    Silently skips files with insufficient data.
    """
    results: dict[int, list[float]] = {}
    chi_map: dict[int, int] = {}

    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith('.txt'):
            continue
        fm = FNAME_RE.match(fname)
        if not fm:
            continue
        D = int(fm.group(1))

        # Extract chi from filename: D_5_chi_20+2D=30 -> chi_eff=30
        chi_m = re.search(r'\+2D=(\d+)', fname)
        if chi_m:
            chi_eff = int(chi_m.group(1))
            chi_map[D] = chi_eff

        raw = parse_file(os.path.join(data_dir, fname))
        if not raw:
            print(f'  [skip] D={D}: no localmag data found in {fname}')
            continue

        per_site: list[float] = []
        for site in SITES:
            vals = [raw[(site, env)] for env in ENVS if (site, env) in raw]
            if len(vals) == 0:
                break
            per_site.append(float(np.mean(vals)))

        if len(per_site) == 6:
            results[D] = per_site
        else:
            print(f'  [skip] D={D}: only {len(per_site)}/6 sites found')

    return results, chi_map


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    here     = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.normpath(os.path.join(here, '..', 'notebooks'))

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', default=data_dir,
                        help='directory containing the result txt files')
    parser.add_argument('--outdir',   default=here,
                        help='directory to write figure files')
    parser.add_argument('--no-show',  action='store_true',
                        help='suppress interactive display')
    args = parser.parse_args()

    print(f'Reading data from: {args.data_dir}')
    results, chi_map = load_all(args.data_dir)

    if not results:
        raise RuntimeError('No valid data files found.')

    Ds   = sorted(results)
    x    = np.array([1.0 / D for D in Ds])
    mus  = np.array([np.mean(results[D])                         for D in Ds])
    sems = np.array([np.std(results[D], ddof=1) / np.sqrt(6)    for D in Ds])

    # ── print table ───────────────────────────────────────────────────────────
    print(f'\n{"D":>4}  {"χ":>6}  {"1/D":>10}  {"mean |m|":>16}  {"SE":>14}')
    print('-' * 60)
    for D, xi, mu, se in zip(Ds, x, mus, sems):
        chi_str = str(chi_map.get(D, '?'))
        print(f'{D:>4}  {chi_str:>6}  {xi:>10.6f}  {mu:>16.10f}  {se:>14.6e}')

    # ── plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.errorbar(
        x, mus, yerr=sems,
        fmt='o-',
        capsize=5, capthick=1.5,
        linewidth=1.5, markersize=7,
        color='steelblue',
        label=r'$\langle|m|\rangle$ ± SE (6 sites A–F)',
    )

    # Create text box with D and χ values (not overlapping data)
    info_text = 'Parameters:\n' + '\n'.join([f'D={D}:  χ={chi}' for D, chi in zip(Ds, [chi_map[D] for D in Ds])])
    ax.text(
        0.98, 0.37, info_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='gray'),
        family='monospace',
    )

    ax.set_xlabel(r'$1/D$', fontsize=13)
    ax.set_ylabel(r'Local magnetization $|m|$', fontsize=13)
    ax.set_title(
        r'Mean local magnetization vs $1/D$' '\n'
        r'(error bars: SE over sites A–F, averaged over env1/2/3)',
        fontsize=11,
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(args.outdir, exist_ok=True)
    for ext in ('pdf', 'png'):
        out = os.path.join(args.outdir, f'localmag_vs_invD.{ext}')
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f'Saved: {out}')

    if not args.no_show:
        plt.show()


if __name__ == '__main__':
    main()
