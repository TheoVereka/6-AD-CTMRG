#!/usr/bin/env python3
"""
plot_chi_convergence_0427.py
For every (J2, D) pair, produce one PDF showing energy, m_neel,
corr_rank1, corr_rank3 vs 1/chi on a shared x-axis.
All panels use globally-shared y-limits across all (J2, D) for each observable.

Output: /home/chye/6ADctmrg/data/extract_obs_0427_plot/chi_conv_J2{j}_D{D}.png
"""

import os
import re
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ──────────────────────────────────────────────────────────────────────────────
DATA_DIR  = '/home/chye/6ADctmrg/data/extract_obs_0427'
OUT_DIR   = '/home/chye/6ADctmrg/data/extract_obs_0427_plot'
os.makedirs(OUT_DIR, exist_ok=True)

RE_FNAME = re.compile(r'^J2([\dp]+)_D(\d+)_obs\.csv$')

# ──────────────────────────────────────────────────────────────────────────────
# Load all per-(J2,D) CSV files
# ──────────────────────────────────────────────────────────────────────────────
def load_all():
    """Returns list of dicts: {j2_str, j2, D, chi, energy, trunc, m, c1, c3}"""
    datasets = []
    for fpath in sorted(glob.glob(os.path.join(DATA_DIR, 'J2*_D*_obs.csv'))):
        fname = os.path.basename(fpath)
        m = RE_FNAME.match(fname)
        if not m:
            continue
        j2_str = m.group(1)
        D      = int(m.group(2))
        j2     = float(j2_str.replace('p', '.'))

        try:
            data = np.genfromtxt(fpath, delimiter=',', skip_header=1,
                                  filling_values=np.nan)
        except Exception as e:
            print(f"  Warning: failed to read {fname}: {e}")
            continue
        if data.ndim == 1:
            data = data[np.newaxis, :]
        if data.shape[0] == 0:
            continue

        datasets.append({
            'j2_str': j2_str,
            'j2':     j2,
            'D':      D,
            'chi':    data[:, 0].astype(int),
            'energy': data[:, 1],
            'trunc':  data[:, 2],
            'm_neel': data[:, 3],
            'c1':     data[:, 4],
            'c3':     data[:, 5],
        })
    return datasets


# ──────────────────────────────────────────────────────────────────────────────
# Compute global y-limits for each observable (with padding)
# ──────────────────────────────────────────────────────────────────────────────
def global_ylims(datasets):
    def _safe(arr):
        v = arr[np.isfinite(arr)]
        return v if len(v) else np.array([float('nan')])

    all_e  = np.concatenate([_safe(d['energy']) for d in datasets])
    all_m  = np.concatenate([_safe(d['m_neel']) for d in datasets])
    all_c1 = np.concatenate([_safe(d['c1'])     for d in datasets])
    all_c3 = np.concatenate([_safe(d['c3'])     for d in datasets])

    def _pad(arr, frac=0.08):
        lo, hi = arr.min(), arr.max()
        margin = max((hi - lo) * frac, 1e-8)
        return (lo - margin, hi + margin)

    return {
        'energy': _pad(all_e),
        'm_neel': (0.0, _pad(all_m)[1]),
        'c1':     _pad(all_c1),
        'c3':     _pad(all_c3),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Plot one (J2, D) figure
# ──────────────────────────────────────────────────────────────────────────────
PANEL_CFG = [
    ('energy', 'Energy per site',              '#1f77b4'),
    ('m_neel', r'm$_\mathrm{N\acute{e}el}$',   '#9467bd'),
    ('c1',     r'$\langle S_iS_j\rangle$ rank 1 (most neg)', 'tab:red'),
    ('c3',     r'$\langle S_iS_j\rangle$ rank 3 (least neg)', 'tab:blue'),
]

# common x tick positions (1/chi values)
def _xticks(x_max):
    step = 0.005
    ticks = [0.0]
    k = 1
    while step * k <= x_max + 1e-9:
        ticks.append(round(step * k, 6))
        k += 1
    return ticks


def plot_one(ds, ylims, out_dir):
    chi    = ds['chi']
    inv    = 1.0 / chi.astype(float)
    x_max  = max(inv) * 1.15

    fig, axes = plt.subplots(4, 1, figsize=(7, 14),
                              gridspec_kw={'hspace': 0.08})

    x_ticks = _xticks(x_max)
    for ax in axes:
        ax.set_xticks(x_ticks)
        ax.set_xlim(-0.0005, x_max)
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.3g'))
        ax.axvline(0, color='gray', lw=0.6, ls='--', alpha=0.4)

    for ax in axes[:-1]:
        ax.set_xticklabels([])
    axes[-1].set_xlabel(r'1/$\chi$', fontsize=11)

    for ax, (key, ylabel, color) in zip(axes, PANEL_CFG):
        y = ds[key]
        valid = np.isfinite(y)
        ax.plot(inv[valid], y[valid], 'o-', color=color, ms=6, lw=1.5)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_ylim(ylims[key])
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.6g'))
        # mark last chi (chimax)
        if valid.any():
            ax.axhline(y[valid][-1], color=color, lw=0.8, ls=':', alpha=0.5)

    axes[0].set_title(f'C6Yπ  J2={ds["j2"]:.2f}  D={ds["D"]}', fontsize=12, pad=5)

    fname = f'chi_conv_J2{ds["j2_str"]}_D{ds["D"]}.png'
    fpath = os.path.join(out_dir, fname)
    fig.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fname


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print("Loading CSV data ...")
    datasets = load_all()
    print(f"  Found {len(datasets)} (J2, D) datasets")

    print("Computing global y-limits ...")
    ylims = global_ylims(datasets)

    print("Generating figures ...")
    for ds in datasets:
        fname = plot_one(ds, ylims, OUT_DIR)
        print(f"  Saved: {fname}")

    print(f"\nDone.  {len(datasets)} PDFs in: {OUT_DIR}")


if __name__ == '__main__':
    main()
