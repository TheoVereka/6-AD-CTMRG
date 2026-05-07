#!/usr/bin/env python3
"""
plot_per_folder.py
==================
Scan every sub-folder under DATA_DIR and produce one 3-panel PDF per folder:
  Panel 1 – Energy per site vs 1/D
  Panel 2 – m_Néel vs 1/D
  Panel 3 – NN bond correlations (3 rank groups) vs 1/D

Output: one PDF per folder saved inside OUT_DIR (created next to this script).

Usage:
    python plot_per_folder.py
"""

import os, re
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.optimize import curve_fit

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG — edit here
# ──────────────────────────────────────────────────────────────────────────────
DATA_DIR  = '/home/chye/6ADctmrg/data/0504core'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR   = os.path.join(SCRIPT_DIR, 'per_folder_plots_0504')

# Bond group raw definitions (env-index, pair-label) — same as 0427 analysis
NN_GROUPS_RAW = [
    [(1,'EB'),(1,'AD'),(1,'CF'),(3,'BE'),(3,'FC'),(3,'DA')],
    [(2,'CB'),(2,'AF'),(2,'ED'),(1,'FA'),(1,'DE'),(1,'BC')],
    [(3,'EF'),(3,'AB'),(3,'CD'),(2,'DC'),(2,'BA'),(2,'FE')],
]
RANK_COLORS = {1: 'tab:red', 2: 'tab:green', 3: 'tab:blue'}
RANK_LABELS = {1: 'rank 1 (most neg)', 2: 'rank 2 (mid)', 3: 'rank 3 (least neg)'}

# ──────────────────────────────────────────────────────────────────────────────
# REGEXES
# ──────────────────────────────────────────────────────────────────────────────
RE_ENERGY    = re.compile(r'^energy_per_site\s*=\s*([+-]?[\d.e+-]+)', re.MULTILINE)
RE_CORR      = re.compile(r'^corr_env(\d+)_([A-F]{2})\s*=\s*([+-]?[\d.e+-]+)', re.MULTILINE)
RE_MAG       = re.compile(
    r'^mag_env(\d+)_([A-F])\s+Sx=([+-]?[\d.e+-]+)\s+Sy=([+-]?[\d.e+-]+)\s+Sz=([+-]?[\d.e+-]+)',
    re.MULTILINE,
)
RE_PLAIN_CHI = re.compile(r'^D_(\d+)_chi_(\d+)_energy_magnetization_correlation\.txt$')
RE_PLUS2D    = re.compile(r'^D_(\d+)_chi_(\d+)\+2D_equals_chi_(\d+)_energy_magnetization_correlation\.txt$')

# ──────────────────────────────────────────────────────────────────────────────
# PARSING
# ──────────────────────────────────────────────────────────────────────────────
def parse_plain_file(fpath):
    txt = open(fpath).read()
    energy = float(RE_ENERGY.search(txt).group(1))
    corr = {}
    for m in RE_CORR.finditer(txt):
        corr[(int(m.group(1)), m.group(2))] = float(m.group(3))
    mag = {}
    for m in RE_MAG.finditer(txt):
        mag[(int(m.group(1)), m.group(2))] = {
            'Sx': float(m.group(3)), 'Sy': float(m.group(4)), 'Sz': float(m.group(5)),
        }
    return {'energy_per_site': energy, 'corr': corr, 'mag': mag}


def load_folder_data(folder_path):
    """For each D: use highest plain chi; prefer +2D file if available."""
    plain_map  = {}
    plus2d_map = {}
    for fname in os.listdir(folder_path):
        m = RE_PLAIN_CHI.match(fname)
        if m:
            D, chi = int(m.group(1)), int(m.group(2))
            plain_map.setdefault(D, []).append((chi, os.path.join(folder_path, fname)))
            continue
        m = RE_PLUS2D.match(fname)
        if m:
            D, chi, eff = int(m.group(1)), int(m.group(2)), int(m.group(3))
            plus2d_map[(D, chi)] = (eff, os.path.join(folder_path, fname))
    result = {}
    for D in sorted(plain_map.keys()):
        best_chi = max(c for c, _ in plain_map[D])
        fpath = (plus2d_map[(D, best_chi)][1] if (D, best_chi) in plus2d_map
                 else next(p for c, p in plain_map[D] if c == best_chi))
        try:
            result[D] = parse_plain_file(fpath)
            result[D]['chi'] = best_chi
        except Exception as e:
            print(f"  Warning: failed to parse {fpath}: {e}")
    return result

# ──────────────────────────────────────────────────────────────────────────────
# ANALYSIS HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def _exp_model(D, E0, c, Dchar):
    return E0 + c * np.exp(-np.asarray(D / Dchar, dtype=float))


def compute_energy_extrap(Ds, eps):
    Ds_f  = np.array(Ds, dtype=float)
    eps_f = np.array(eps, dtype=float)
    inv   = 1.0 / Ds_f
    E_horiz = float(eps_f[-1])
    n3 = min(3, len(Ds_f))
    c3 = np.polyfit(inv[-n3:], eps_f[-n3:], 1)
    E_lin3 = float(c3[1])
    c_all  = np.polyfit(inv, eps_f, 1) if len(Ds_f) >= 2 else c3
    E_lin_all = float(c_all[1])
    E_exp = None; exp_popt = None
    if len(Ds_f) >= 3:
        try:
            p0 = [eps_f[-1] - 0.01, 0.01, float(Ds_f[-1])]
            popt, _ = curve_fit(_exp_model, Ds_f, eps_f, p0=p0, maxfev=8000)
            if popt[2] > 0:
                E_exp = float(popt[0]); exp_popt = popt
        except Exception:
            pass
    if E_exp is not None:
        lo = min(E_horiz, E_lin3) - 2 * abs(E_horiz - E_lin3) - 0.3
        hi = max(E_horiz, E_lin3) + 2 * abs(E_horiz - E_lin3) + 0.3
        E_best = E_exp if lo <= E_exp <= hi else E_lin3
    else:
        E_best = E_lin3
    E_err = (0.5 * abs(E_lin3 - E_exp)
             if (E_exp is not None and E_best == E_exp) else 0.0)
    return {'E_horiz': E_horiz, 'E_lin3': E_lin3, 'E_lin_all': E_lin_all,
            'E_exp': E_exp, 'E_best': E_best, 'E_err': E_err, 'exp_popt': exp_popt}


def compute_bond_groups(D_data, groups_raw):
    result = {}
    for D in sorted(D_data.keys()):
        corr = D_data[D]['corr']
        means, stds = [], []
        for group in groups_raw:
            vals = [corr.get((env, pair), float('nan')) for env, pair in group]
            vals = [v for v in vals if not np.isnan(v)]
            means.append(float(np.mean(vals)) if vals else float('nan'))
            stds.append(float(np.std(vals))  if vals else float('nan'))
        order = np.argsort(means)
        ranks = [0] * len(means)
        for rank_idx, g_idx in enumerate(order):
            ranks[g_idx] = rank_idx + 1
        result[D] = {'means': means, 'stds': stds, 'ranks': ranks}
    return result


def compute_order_param(D_data):
    result = {}
    for D in sorted(D_data.keys()):
        mag = D_data[D].get('mag', {})
        sz_ace, sz_bdf, sx_ace, sx_bdf = [], [], [], []
        for env in [1, 2, 3]:
            for site in ['A', 'C', 'E']:
                if (env, site) in mag:
                    sz_ace.append(mag[(env, site)]['Sz'])
                    sx_ace.append(mag[(env, site)]['Sx'])
            for site in ['B', 'D', 'F']:
                if (env, site) in mag:
                    sz_bdf.append(mag[(env, site)]['Sz'])
                    sx_bdf.append(mag[(env, site)]['Sx'])
        sz_a = float(np.mean(sz_ace)) if sz_ace else float('nan')
        sz_b = float(np.mean(sz_bdf)) if sz_bdf else float('nan')
        m_neel = (abs(sz_a - sz_b) / 2.0
                  if not any(np.isnan(x) for x in [sz_a, sz_b]) else float('nan'))
        result[D] = {'m_ace': sz_a, 'm_bdf': sz_b, 'm_neel': m_neel}
    return result


def compute_mag_extrap(Ds, mneel_list):
    Ds_f = np.array(Ds, dtype=float); m_f = np.array(mneel_list, dtype=float)
    inv  = 1.0 / Ds_f
    valid = ~np.isnan(m_f)
    if valid.sum() < 2:
        return float('nan'), float('nan'), float('nan'), float('nan')
    inv_v = inv[valid]; m_v = m_f[valid]
    c2 = np.polyfit(inv_v[-min(2, len(inv_v)):], m_v[-min(2, len(inv_v)):], 1)
    c3 = np.polyfit(inv_v[-min(3, len(inv_v)):], m_v[-min(3, len(inv_v)):], 1)
    m2, m3 = float(c2[1]), float(c3[1])
    return m2, m3, 0.5 * (m2 + m3), 0.5 * abs(m2 - m3)

# ──────────────────────────────────────────────────────────────────────────────
# PLOT
# ──────────────────────────────────────────────────────────────────────────────
def plot_one_folder(folder_name, D_data, out_dir):
    Ds    = sorted(D_data.keys())
    if not Ds:
        print(f"  Skipping {folder_name}: no data")
        return
    eps   = [D_data[D]['energy_per_site'] for D in Ds]
    Ds_f  = np.array(Ds, dtype=float)
    inv   = 1.0 / Ds_f

    ex        = compute_energy_extrap(Ds, eps)
    nn_grp    = compute_bond_groups(D_data, NN_GROUPS_RAW)
    order     = compute_order_param(D_data)
    mneel_list = [order[D]['m_neel'] for D in Ds]
    m_lin2, m_lin3, m_star, m_err = compute_mag_extrap(Ds, mneel_list)

    x_max_plot = max(inv) * 1.25
    inv_line   = np.linspace(0, x_max_plot, 300)

    fig, axes  = plt.subplots(3, 1, figsize=(7, 13),
                               gridspec_kw={'hspace': 0.08})
    ax_e, ax_m, ax_nn = axes

    tick_step = 0.025
    x_ticks = [0.0] + [round(tick_step * k, 6)
                        for k in range(1, int(np.ceil(x_max_plot / tick_step)) + 1)
                        if tick_step * k <= x_max_plot + 1e-9]
    for ax in axes:
        ax.set_xticks(x_ticks)
        ax.set_xlim(left=-0.002, right=x_max_plot)
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.3g'))
    ax_e.set_xticklabels([])
    ax_m.set_xticklabels([])
    ax_nn.set_xlabel('1/D', fontsize=11)

    # ── Panel 1: Energy ───────────────────────────────────────────────────────
    col_e = '#1f77b4'
    # annotate each point with chi
    for i, D in enumerate(Ds):
        chi_val = D_data[D].get('chi', '')
        ax_e.annotate(f'χ={chi_val}', (inv[i], eps[i]),
                      textcoords='offset points', xytext=(4, 3), fontsize=7, color=col_e)
    ax_e.plot(inv, eps, 'o-', color=col_e, ms=6, lw=1.6, zorder=5, label='E(D)')
    ax_e.axhline(ex['E_horiz'], color=col_e, lw=0.9, ls=':', alpha=0.55,
                  label=f'E(D={Ds[-1]})')
    n3 = min(3, len(Ds_f))
    c3 = np.polyfit(inv[-n3:], np.array(eps[-n3:]), 1)
    ax_e.plot(inv_line, np.polyval(c3, inv_line),
              color='tab:orange', lw=1.1, ls='--', alpha=0.85,
              label=f'lin fit (last {n3} D): {ex["E_lin3"]:.6f}')
    if ex['exp_popt'] is not None:
        Ds_line = np.where(inv_line > 1e-12, 1.0 / np.maximum(inv_line, 1e-12), 1e12)
        ax_e.plot(inv_line, _exp_model(Ds_line, *ex['exp_popt']),
                  color='tab:green', lw=1.1, ls='-.', alpha=0.85,
                  label=f'exp fit: {ex["E_exp"]:.6f}')
    ax_e.plot(0, ex['E_best'], '*', color=col_e, ms=14, zorder=7,
              markeredgewidth=0.5, markeredgecolor='k',
              label=f'extrap = {ex["E_best"]:.6f}')
    ax_e.set_ylabel('Energy per site', fontsize=11)
    ax_e.set_title(folder_name, fontsize=10, pad=6)
    ax_e.legend(fontsize=8, loc='upper right')
    ax_e.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.6g'))

    # ── Panel 2: m_Néel ───────────────────────────────────────────────────────
    col_m = '#9467bd'
    mneel_arr = np.array(mneel_list, dtype=float)
    ax_m.plot(inv, mneel_arr, 'o-', color=col_m, ms=6, lw=1.6, zorder=5,
              label=r'm$_\mathrm{N\acute{e}el}$(D)')
    valid = ~np.isnan(mneel_arr)
    inv_v = inv[valid]; m_v = mneel_arr[valid]
    if len(m_v) >= 2:
        ax_m.axhline(float(m_v[-1]), color=col_m, lw=0.9, ls=':', alpha=0.55)
        n2m = min(2, len(inv_v))
        ax_m.plot(inv_line, np.polyval(np.polyfit(inv_v[-n2m:], m_v[-n2m:], 1), inv_line),
                  color='tab:orange', lw=1.1, ls='--', alpha=0.85,
                  label=f'lin2: {m_lin2:.4f}')
        n3m = min(3, len(inv_v))
        ax_m.plot(inv_line, np.polyval(np.polyfit(inv_v[-n3m:], m_v[-n3m:], 1), inv_line),
                  color='tab:red', lw=1.1, ls='-.', alpha=0.85,
                  label=f'lin3: {m_lin3:.4f}')
    m_lastD = float(mneel_arr[-1]) if not np.isnan(mneel_arr[-1]) else 0.0
    three_v = sorted([m_lastD, m_lin2 if not np.isnan(m_lin2) else 0.0,
                       m_lin3 if not np.isnan(m_lin3) else 0.0])
    m_ctr = max(0.0, three_v[1])
    ax_m.errorbar([0], [m_ctr],
                  yerr=[[m_ctr - max(0.0, three_v[0])], [max(0.0, three_v[2]) - m_ctr]],
                  fmt='*', color=col_m, ms=14, capsize=5, elinewidth=1.3,
                  markeredgewidth=0.5, markeredgecolor='k', zorder=7,
                  label=f'extrap = {m_ctr:.4f}')
    ax_m.set_ylim(bottom=0.0)
    ax_m.set_ylabel(r'm$_\mathrm{N\acute{e}el}$', fontsize=11)
    ax_m.legend(fontsize=8, loc='upper right')
    ax_m.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.5g'))

    # ── Panel 3: NN bond correlations ─────────────────────────────────────────
    D_list = sorted(nn_grp.keys())
    for target_rank in [1, 2, 3]:
        D_r, m_r = [], []
        for D in D_list:
            entry = nn_grp[D]
            for gi, rank in enumerate(entry['ranks']):
                if rank == target_rank:
                    D_r.append(D); m_r.append(entry['means'][gi]); break
        if D_r:
            ax_nn.plot([1.0 / D for D in D_r], m_r, 'o-',
                       color=RANK_COLORS[target_rank], ms=5, lw=1.2,
                       label=RANK_LABELS[target_rank])
    ax_nn.set_ylabel(r'$\langle S_i \cdot S_j \rangle$ (NN)', fontsize=11)
    ax_nn.legend(fontsize=8, loc='upper right')
    ax_nn.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.5g'))

    safe_name = folder_name.replace('/', '_').replace(' ', '_')
    fpath = os.path.join(out_dir, f'{safe_name}.pdf')
    fig.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {os.path.relpath(fpath)}")

# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    folders = sorted(
        f for f in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, f))
    )
    if not folders:
        print(f"No sub-folders found in {DATA_DIR}")
        return
    print(f"Found {len(folders)} folders in {DATA_DIR}")
    for folder_name in folders:
        folder_path = os.path.join(DATA_DIR, folder_name)
        print(f"\nProcessing: {folder_name}")
        D_data = load_folder_data(folder_path)
        if not D_data:
            print(f"  No observable .txt files found — skipping.")
            continue
        plot_one_folder(folder_name, D_data, OUT_DIR)
    print(f"\nDone. All figures in: {OUT_DIR}")


if __name__ == '__main__':
    main()
