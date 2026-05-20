#!/usr/bin/env python3
"""
plot_analysis_0507D456v2.py
Multi-ansatz analysis: one PDF per J2.

For each J2 value, a single figure is produced with
  • columns  = ansatze found for that J2  (e.g. Néel / C6Yπ / C3vYπ)
  • rows     = 3  (energy / m_Néel / NN correlations)  all vs 1/D

y-limits are shared across all columns within the same J2 (per observable).
m_Néel always has lower limit 0.

Outputs in:  analysis_plots_0507D456v2/   (next to this script)
"""

import os, re, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.optimize import curve_fit

# ──────────────────────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────────────────────
DATA_DIR   = '/home/chye/6ADctmrg/data/0507core/D4567'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(SCRIPT_DIR, 'analysis_plots_0507D4567')
os.makedirs(OUT_DIR, exist_ok=True)

# Only these D values
D_ALLOWED = {4, 5, 6, 7}

# Bond group raw definitions (env-index, pair-label)
NN_GROUPS_RAW = [
    [(1,'EB'),(1,'AD'),(1,'CF'),(3,'BE'),(3,'FC'),(3,'DA')],
    [(2,'CB'),(2,'AF'),(2,'ED'),(1,'FA'),(1,'DE'),(1,'BC')],
    [(3,'EF'),(3,'AB'),(3,'CD'),(2,'DC'),(2,'BA'),(2,'FE')],
]

RANK_COLORS = {1: 'tab:red', 2: 'tab:green', 3: 'tab:blue'}
RANK_LABELS = {1: 'rank 1 (most neg)', 2: 'rank 2 (mid)', 3: 'rank 3 (least neg)'}

# Preferred column order for ansatze
ANSATZ_ORDER = ['neel_symmetrized', '1tensor_C6Ypi', '1tensor_C3Vypi']

# Pretty labels for ansatze
ANSATZ_LABEL = {
    'neel_symmetrized': 'Néel',
    '1tensor_C6Ypi':   'C6Yπ',
    '1tensor_C3Vypi':  'C3vYπ',
}

# ──────────────────────────────────────────────────────────────────────────────
# REGEXES
# ──────────────────────────────────────────────────────────────────────────────
RE_ENERGY    = re.compile(r'^energy_per_site\s*=\s*([+-]?[\d.e+-]+)', re.MULTILINE)
RE_CORR      = re.compile(r'^corr_env(\d+)_([A-F]{2})\s*=\s*([+-]?[\d.e+-]+)', re.MULTILINE)
RE_MAG       = re.compile(
    r'^mag_env(\d+)_([A-F])\s+Sx=([+-]?[\d.e+-]+)\s+Sy=([+-]?[\d.e+-]+)\s+Sz=([+-]?[\d.e+-]+)',
    re.MULTILINE,
)
# plain file: D_4_chi_32_energy_magnetization_correlation.txt
RE_PLAIN_CHI = re.compile(r'^D_(\d+)_chi_(\d+)_energy_magnetization_correlation\.txt$')

# ──────────────────────────────────────────────────────────────────────────────
# FOLDER DISCOVERY
# ──────────────────────────────────────────────────────────────────────────────
def _parse_j2_from_str(s):
    """Convert e.g. '0p32' -> 0.32, '0p265' -> 0.265"""
    return float(s.replace('p', '.'))


def _j2_fname(j2):
    s = f'{j2:.10f}'.rstrip('0')
    dot = s.index('.')
    if len(s) - dot - 1 < 2:
        s += '0' * (2 - (len(s) - dot - 1))
    return s.replace('.', 'p')


# Folder name pattern:  <ansatz>__J2_<j2str>_<dates>
RE_FOLDER = re.compile(r'^(.+?)__J2_([0-9p]+)_\d{8}')


def discover_folders(data_dir):
    """
    Returns dict:   j2 (float) → { ansatz_key (str) → folder_path (str) }
    Only the LAST matching folder for a given (j2, ansatz) is kept
    (lexicographically last = most recent timestamp suffix).
    """
    mapping = {}   # (j2, ansatz) → folder_path
    for name in sorted(os.listdir(data_dir)):
        full = os.path.join(data_dir, name)
        if not os.path.isdir(full):
            continue
        m = RE_FOLDER.match(name)
        if not m:
            continue
        ansatz = m.group(1)
        j2_str = m.group(2)
        try:
            j2 = round(_parse_j2_from_str(j2_str), 6)
        except ValueError:
            continue
        mapping[(j2, ansatz)] = full   # later (sorted) overwrites earlier

    result = {}
    for (j2, ansatz), path in mapping.items():
        result.setdefault(j2, {})[ansatz] = path
    return result

# ──────────────────────────────────────────────────────────────────────────────
# FILE PARSING
# ──────────────────────────────────────────────────────────────────────────────
def parse_plain_file(fpath):
    txt     = open(fpath).read()
    energy  = float(RE_ENERGY.search(txt).group(1))
    corr    = {}
    for m in RE_CORR.finditer(txt):
        corr[(int(m.group(1)), m.group(2))] = float(m.group(3))
    mag     = {}
    for m in RE_MAG.finditer(txt):
        mag[(int(m.group(1)), m.group(2))] = {
            'Sx': float(m.group(3)), 'Sy': float(m.group(4)), 'Sz': float(m.group(5)),
        }
    return {'energy_per_site': energy, 'corr': corr, 'mag': mag}


def load_folder_data(folder_path):
    """
    For each D in D_ALLOWED: pick the plain file with the highest chi.
    Ignores lookahead files entirely.
    """
    plain_map = {}   # D → [(chi, path)]
    for fname in os.listdir(folder_path):
        m = RE_PLAIN_CHI.match(fname)
        if not m:
            continue
        D, chi = int(m.group(1)), int(m.group(2))
        if D not in D_ALLOWED:
            continue
        plain_map.setdefault(D, []).append((chi, os.path.join(folder_path, fname)))

    result = {}
    for D in sorted(plain_map.keys()):
        best_chi = max(c for c, _ in plain_map[D])
        fpath    = next(p for c, p in plain_map[D] if c == best_chi)
        try:
            result[D] = parse_plain_file(fpath)
        except Exception as e:
            print(f"  Warning: failed to parse {fpath}: {e}")
    return result

# ──────────────────────────────────────────────────────────────────────────────
# OBSERVABLES
# ──────────────────────────────────────────────────────────────────────────────
def compute_bond_groups(D_data, groups_raw):
    result = {}
    for D in sorted(D_data.keys()):
        corr   = D_data[D]['corr']
        means, stds = [], []
        for group in groups_raw:
            vals = [corr[(env, pair)] for (env, pair) in group if (env, pair) in corr]
            means.append(float(np.mean(vals)) if vals else float('nan'))
            stds.append(float(np.std(vals, ddof=0)) if vals else float('nan'))
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
        sz_ace, sz_bdf = [], []
        sx_ace, sx_bdf = [], []
        for env in [1, 2, 3]:
            for site in 'ACE':
                if (env, site) in mag:
                    sz_ace.append(mag[(env, site)]['Sz'])
                    sx_ace.append(mag[(env, site)]['Sx'])
            for site in 'BDF':
                if (env, site) in mag:
                    sz_bdf.append(mag[(env, site)]['Sz'])
                    sx_bdf.append(mag[(env, site)]['Sx'])
        def safe_mean(lst):
            return float(np.mean(lst)) if lst else float('nan')
        sz_ace_m = safe_mean(sz_ace); sz_bdf_m = safe_mean(sz_bdf)
        sx_ace_m = safe_mean(sx_ace); sx_bdf_m = safe_mean(sx_bdf)
        if not any(np.isnan(x) for x in [sz_ace_m, sz_bdf_m, sx_ace_m, sx_bdf_m]):
            stag_sz = 0.5 * (sz_ace_m - sz_bdf_m)
            stag_sx = 0.5 * (sx_ace_m - sx_bdf_m)
            m_neel  = float(np.sqrt(stag_sx**2 + stag_sz**2))
        else:
            m_neel  = float('nan')
        result[D] = {'m_neel': m_neel}
    return result

# ──────────────────────────────────────────────────────────────────────────────
# EXTRAPOLATION HELPERS
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

    E_exp    = None
    exp_popt = None
    if len(Ds_f) >= 3:
        try:
            p0 = [eps_f[-1] - 0.1, 0.1, float(Ds_f.mean())]
            popt, _ = curve_fit(_exp_model, Ds_f, eps_f, p0=p0, maxfev=5000)
            lo = min(E_horiz, E_lin3) - 2 * abs(E_horiz - E_lin3) - 0.3
            hi = max(E_horiz, E_lin3) + 2 * abs(E_horiz - E_lin3) + 0.3
            if lo <= popt[0] <= hi:
                E_exp    = float(popt[0])
                exp_popt = popt.tolist()
        except Exception:
            pass

    E_best = E_exp if E_exp is not None else E_lin3

    return {
        'E_horiz':   E_horiz,
        'E_lin3':    E_lin3,
        'E_exp':     E_exp,
        'E_best':    E_best,
        'exp_popt':  exp_popt,
        'poly_lin3': c3,
    }


def compute_mag_extrap(Ds, mneel_list):
    Ds_f = np.array(Ds,        dtype=float)
    m_f  = np.array(mneel_list, dtype=float)
    inv  = 1.0 / Ds_f

    n2 = min(2, len(Ds_f)); c2 = np.polyfit(inv[-n2:], m_f[-n2:], 1)
    n3 = min(3, len(Ds_f)); c3 = np.polyfit(inv[-n3:], m_f[-n3:], 1)
    m_lin2 = float(c2[1]); m_lin3 = float(c3[1])
    return m_lin2, m_lin3, c2, c3

# ──────────────────────────────────────────────────────────────────────────────
# LOAD ALL DATA
# ──────────────────────────────────────────────────────────────────────────────
def load_ansatz_data(folder_path):
    """Return processed data dict for one ansatz folder."""
    D_data = load_folder_data(folder_path)
    if not D_data:
        return None
    Ds  = sorted(D_data.keys())
    eps = [D_data[D]['energy_per_site'] for D in Ds]
    extrap    = compute_energy_extrap(Ds, eps)
    nn_groups = compute_bond_groups(D_data, NN_GROUPS_RAW)
    order     = compute_order_param(D_data)
    mlist     = [order[D]['m_neel'] for D in Ds]
    m_lin2, m_lin3, c2, c3 = compute_mag_extrap(Ds, mlist)
    return {
        'Ds':              Ds,
        'energy_per_site': eps,
        'extrap':          extrap,
        'nn_groups':       nn_groups,
        'mneel_list':      mlist,
        'm_lin2':          m_lin2,
        'm_lin3':          m_lin3,
        'm_c2':            c2,
        'm_c3':            c3,
    }


def load_all():
    """Returns dict:  j2 → { ansatz_key → processed_data }"""
    folder_map = discover_folders(DATA_DIR)
    all_data   = {}
    for j2 in sorted(folder_map.keys()):
        ansatz_data = {}
        for ansatz, path in folder_map[j2].items():
            print(f"  Loading J2={j2:.4g}  {ansatz} ...")
            d = load_ansatz_data(path)
            if d:
                ansatz_data[ansatz] = d
            else:
                print(f"    (no usable files)")
        if ansatz_data:
            all_data[j2] = ansatz_data
    return all_data

# ──────────────────────────────────────────────────────────────────────────────
# PER-J2 FIGURE
# ──────────────────────────────────────────────────────────────────────────────
def _save(fig, fpath):
    fig.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {os.path.relpath(fpath)}")


def _pad(lo, hi, frac=0.10):
    margin = max((hi - lo) * frac, 1e-6)
    return (lo - margin, hi + margin)


def compute_j2_ylims(ansatz_map):
    """Compute per-observable ylims shared across all ansatze for this J2."""
    all_e, all_m, all_nn = [], [], []
    for v in ansatz_map.values():
        all_e.extend(v['energy_per_site'])
        all_e.append(v['extrap']['E_best'])
        ms = [x for x in v['mneel_list'] if not np.isnan(x)]
        all_m.extend(ms)
        all_m.append(max(0.0, v['m_lin2']))
        all_m.append(max(0.0, v['m_lin3']))
        for D_entry in v['nn_groups'].values():
            all_nn.extend(x for x in D_entry['means'] if not np.isnan(x))

    e_ylim  = _pad(min(all_e), max(all_e))
    m_hi    = max(all_m) if all_m else 1.0
    m_ylim  = (0.0, m_hi + max(m_hi * 0.10, 1e-6))
    nn_ylim = _pad(min(all_nn), max(all_nn)) if all_nn else (-1, 0)
    return {'energy': e_ylim, 'mag': m_ylim, 'nn': nn_ylim}


def plot_col_energy(ax, v, show_xlabel):
    Ds  = np.array(v['Ds'], dtype=float)
    inv = 1.0 / Ds
    eps = np.array(v['energy_per_site'])
    ex  = v['extrap']

    x_max = max(inv) * 1.30
    inv_line = np.linspace(0, x_max, 300)

    col = '#1f77b4'
    ax.plot(inv, eps, 'o-', color=col, ms=6, lw=1.5, zorder=5, label='E(D)')
    ax.axhline(ex['E_horiz'], color=col, ls=':', lw=0.8, alpha=0.5,
               label=f'E(D={int(Ds[-1])})')

    n3 = min(3, len(Ds))
    ax.plot(inv_line, np.polyval(ex['poly_lin3'], inv_line),
            color='tab:orange', ls='--', lw=1.0, alpha=0.85,
            label=f'lin({n3}D)')

    if ex['exp_popt'] is not None:
        D_line = np.where(inv_line > 1e-12, 1.0 / np.maximum(inv_line, 1e-12), 1e12)
        ax.plot(inv_line, _exp_model(D_line, *ex['exp_popt']),
                color='tab:green', ls='-.', lw=1.0, alpha=0.85, label='exp fit')

    ax.plot(0, ex['E_best'], '*', color=col, ms=13, zorder=7,
            markeredgewidth=0.5, markeredgecolor='k',
            label=f'{ex["E_best"]:.5f}')

    ax.set_xlim(left=-0.002, right=x_max)
    ax.legend(fontsize=7, loc='upper right')
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.5g'))
    _set_xticks(ax, x_max, show_xlabel)


def plot_col_mag(ax, v, show_xlabel):
    Ds  = np.array(v['Ds'], dtype=float)
    inv = 1.0 / Ds
    ms  = np.array(v['mneel_list'])

    x_max    = max(inv) * 1.30
    inv_line = np.linspace(0, x_max, 300)

    col = '#9467bd'
    ax.plot(inv, ms, 'o-', color=col, ms=6, lw=1.5, zorder=5,
            label=r'm$_\mathrm{N}$(D)')

    valid = ~np.isnan(ms)
    inv_v, m_v = inv[valid], ms[valid]
    if len(m_v) >= 2:
        ax.plot(inv_line, np.polyval(v['m_c2'], inv_line),
                color='tab:orange', ls='--', lw=1.0, alpha=0.85,
                label=f'lin(2): {v["m_lin2"]:.4f}')
    if len(m_v) >= 3:
        ax.plot(inv_line, np.polyval(v['m_c3'], inv_line),
                color='tab:red', ls='-.', lw=1.0, alpha=0.85,
                label=f'lin(3): {v["m_lin3"]:.4f}')

    m_lastD = float(ms[-1]) if not np.isnan(ms[-1]) else 0.0
    three   = sorted([m_lastD, v['m_lin2'], v['m_lin3']])
    m_ctr   = max(0.0, three[1])
    m_lo    = max(0.0, three[0])
    m_hi    = max(0.0, three[2])
    ax.errorbar([0], [m_ctr], yerr=[[m_ctr - m_lo], [m_hi - m_ctr]],
                fmt='*', color=col, ms=13, capsize=4, elinewidth=1.1,
                markeredgewidth=0.5, markeredgecolor='k', zorder=7,
                label=f'{m_ctr:.4f}')

    ax.set_xlim(left=-0.002, right=x_max)
    ax.legend(fontsize=7, loc='upper right')
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4g'))
    _set_xticks(ax, x_max, show_xlabel)


def plot_col_nn(ax, v, show_xlabel):
    nn_grp = v['nn_groups']
    D_list = sorted(nn_grp.keys())
    inv    = [1.0 / D for D in D_list]

    x_max = max(inv) * 1.30
    for target_rank in [1, 2, 3]:
        m_r = []
        for D in D_list:
            entry = nn_grp[D]
            vals  = [entry['means'][g]
                     for g, r in enumerate(entry['ranks']) if r == target_rank]
            m_r.append(float(np.mean(vals)) if vals else float('nan'))
        ax.plot(inv, m_r, 'o-', color=RANK_COLORS[target_rank], ms=5, lw=1.2,
                label=RANK_LABELS[target_rank])

    ax.set_xlim(left=-0.002, right=x_max)
    ax.legend(fontsize=7, loc='upper right')
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4g'))
    _set_xticks(ax, x_max, show_xlabel, force_label=True)
    if show_xlabel:
        ax.set_xlabel('1/D', fontsize=10)


def _set_xticks(ax, x_max, show_xlabel, force_label=False):
    step = 0.05
    ticks = [0.0] + [round(step * k, 6)
                     for k in range(1, int(np.ceil(x_max / step)) + 3)
                     if step * k <= x_max + 1e-9]
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2g'))
    if not (show_xlabel or force_label):
        ax.set_xticklabels([])


def plot_j2_figure(j2, ansatz_map, out_dir):
    # Column order: prefer ANSATZ_ORDER, then alphabetical
    cols = ([a for a in ANSATZ_ORDER if a in ansatz_map] +
            sorted(a for a in ansatz_map if a not in ANSATZ_ORDER))

    n_cols  = len(cols)
    ylims   = compute_j2_ylims(ansatz_map)

    fig, axes = plt.subplots(3, n_cols,
                             figsize=(4.5 * n_cols, 11),
                             gridspec_kw={'hspace': 0.07, 'wspace': 0.30})

    # Ensure 2D even for n_cols==1
    if n_cols == 1:
        axes = np.array(axes).reshape(3, 1)

    ROW_LABELS = ['Energy per site', r'm$_\mathrm{N\acute{e}el}$',
                  r'$\langle S_i \cdot S_j \rangle$ (NN)']

    for col_idx, ansatz in enumerate(cols):
        v   = ansatz_map[ansatz]
        lbl = ANSATZ_LABEL.get(ansatz, ansatz)

        ax_e  = axes[0, col_idx]
        ax_m  = axes[1, col_idx]
        ax_nn = axes[2, col_idx]

        plot_col_energy(ax_e,  v, show_xlabel=False)
        plot_col_mag   (ax_m,  v, show_xlabel=False)
        plot_col_nn    (ax_nn, v, show_xlabel=True)

        # Apply shared ylims
        ax_e.set_ylim(ylims['energy'])
        ax_m.set_ylim(ylims['mag'])
        ax_nn.set_ylim(ylims['nn'])

        ax_e.set_title(lbl, fontsize=13, pad=5)

        # left-column y-labels
        if col_idx == 0:
            ax_e.set_ylabel(ROW_LABELS[0],  fontsize=10)
            ax_m.set_ylabel(ROW_LABELS[1],  fontsize=10)
            ax_nn.set_ylabel(ROW_LABELS[2], fontsize=10)

    fig.suptitle(f'J₂ = {j2:.4g}', fontsize=14, y=1.005)

    jstr  = _j2_fname(j2)
    fpath = os.path.join(out_dir, f'J2_{jstr}.pdf')
    _save(fig, fpath)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print("Discovering and loading data ...")
    all_data = load_all()
    if not all_data:
        print("No data found — check DATA_DIR.")
        return

    print(f"\nFound {len(all_data)} J2 values: {sorted(all_data.keys())}")
    print(f"Generating {len(all_data)} figures ...\n")

    for j2, ansatz_map in sorted(all_data.items()):
        ansatze = sorted(ansatz_map.keys())
        print(f"  J2={j2:.4g}  ansatze: {ansatze}")
        plot_j2_figure(j2, ansatz_map, OUT_DIR)

    print(f"\nDone.  Figures in: {OUT_DIR}")


if __name__ == '__main__':
    main()
