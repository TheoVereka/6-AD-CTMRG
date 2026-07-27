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
DATA_DIR   = r'/home/chye/6ADctmrg/data/0507core/D45678'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(SCRIPT_DIR, 'analysis_plots_0507D45678910')
os.makedirs(OUT_DIR, exist_ok=True)

# Only these D values
D_ALLOWED = {3, 4, 5, 6, 7, 8, 9, 10}
#D_ALLOWED = {9}



# Special-case override: (j2, ansatz_key) → n points for the main extrap linear fit
SPECIAL_MAG_N = {
    (0.25, 'neel_symmetrized'): 4,
    (0.20, 'neel_symmetrized'): 4,
    (0.23, 'neel_symmetrized'): 4,
    (0.22, 'neel_symmetrized'): 4,
}



# Preferred column order for ansatze
ANSATZ_ORDER = ['neel_symmetrized', '1tensor_C6Ypi', '1tensor_C3Vypi', '2tensor_twoC3', '6tensors']

# Pretty labels for ansatze
ANSATZ_LABEL = {
    'neel_symmetrized': 'Néel',
    '1tensor_C6Ypi':   'C6Yπ',
    '1tensor_C3Vypi':  'C3vYπ',
    '2tensor_twoC3':   '2 C3',
    '6tensors':       'unres',
}




# Bond group raw definitions (env-index, pair-label)
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
def parse_plain_file_UBUNTU(fpath):
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

def parse_plain_file(fpath):
    with open(fpath, encoding='utf-8') as f:
        txt = f.read()

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
    n     = len(Ds_f)

    if n == 1:
        E0 = float(eps_f[0])
        return {
            'E_horiz':   E0,
            'E_lin3':    None,          # no linear extrapolation possible
            'E_exp':     None,          # exponential fit disabled
            'E_best':    E0,
            'exp_popt':  None,
            'poly_lin3': None,
        }

    inv   = 1.0 / Ds_f
    E_horiz = float(eps_f[-1])

    # linear fit using last min(3, n) points – requires n>=2
    n3 = min(3, n)
    c3 = np.polyfit(inv[-n3:], eps_f[-n3:], 1)
    E_lin3 = float(c3[1])

    E_exp    = None
    exp_popt = None
    exp_mask = Ds_f >= 5
    Ds_exp = Ds_f[exp_mask]
    eps_exp = eps_f[exp_mask]
    if len(Ds_exp) >= 3:     # exponential fit needs at least 3 D>=5 points
        try:
            p0 = [eps_exp[-1] - 0.1, 0.1, float(Ds_exp.mean())]
            popt, _ = curve_fit(
                _exp_model, Ds_exp, eps_exp, p0=p0, maxfev=5000
            )
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


def compute_mag_extrap(Ds, mneel_list, n_extrap=3):
    Ds_f = np.array(Ds,        dtype=float)
    m_f  = np.array(mneel_list, dtype=float)
    inv  = 1.0 / Ds_f

    n2 = min(2, len(Ds_f)); c2 = np.polyfit(inv[-n2:], m_f[-n2:], 1)
    n3 = min(n_extrap, len(Ds_f)); c3 = np.polyfit(inv[-n3:], m_f[-n3:], 1)
    m_lin2 = float(c2[1]); m_lin3 = float(c3[1])
    return m_lin2, m_lin3, c2, c3

# ──────────────────────────────────────────────────────────────────────────────
# LOAD ALL DATA
# ──────────────────────────────────────────────────────────────────────────────
def load_ansatz_data(folder_path, n_mag=3):
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
    m_lin2, m_lin3, c2, c3 = compute_mag_extrap(Ds, mlist, n_extrap=n_mag)
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
        'm_n_extrap':      n_mag,
    }


def load_all():
    """Returns dict:  j2 → { ansatz_key → processed_data }"""
    folder_map = discover_folders(DATA_DIR)
    all_data   = {}
    for j2 in sorted(folder_map.keys()):
        ansatz_data = {}
        for ansatz, path in folder_map[j2].items():
            print(f"  Loading J2={j2:.4g}  {ansatz} ...")
            n_mag = SPECIAL_MAG_N.get((round(j2, 6), ansatz), 3)
            d = load_ansatz_data(path, n_mag=n_mag)
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
    if ex['poly_lin3'] is not None:                # draw only if lin3 was computed
        ax.plot(inv_line, np.polyval(ex['poly_lin3'], inv_line),
                color='tab:orange', ls='--', lw=1.0, alpha=0.85,
                label=f'lin({n3}D)')

    if ex['exp_popt'] is not None:                # draw only if exp fit succeeded
        D_line = np.where(inv_line > 1e-12, 1.0 / np.maximum(inv_line, 1e-12), 1e12)
        ax.plot(inv_line, _exp_model(D_line, *ex['exp_popt']),
                color='tab:green', ls='-.', lw=1.0, alpha=0.85,
                label=r'exp fit $D\geq5$')

# ... (star marker at E_best remains – always drawn)

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
    n_ex = v.get('m_n_extrap', 3)
    if len(m_v) >= 2:
        ax.plot(inv_line, np.polyval(v['m_c2'], inv_line),
                color='tab:orange', ls='--', lw=1.0, alpha=0.85,
                label=f'lin(2): {v["m_lin2"]:.4f}')
    if len(m_v) >= min(n_ex, 3):
        ax.plot(inv_line, np.polyval(v['m_c3'], inv_line),
                color='tab:red', ls='-.', lw=1.0, alpha=0.85,
                label=f'lin({min(n_ex, len(m_v))}): {v["m_lin3"]:.4f}')

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
# PART 2 & 3: Summary vs J2
# ──────────────────────────────────────────────────────────────────────────────

def compute_m_lin3_extrap_with_error(Ds, mlist, n_extrap=3):
    """
    Fit m = k*(1/D) + b using last min(n_extrap, n) valid Ds via polyfit with cov.
    Returns (b, b_lo_clipped, b_hi) with b_lo >= 0.
    """
    Ds_f = np.array(Ds, dtype=float)
    m_f  = np.array(mlist, dtype=float)
    valid = ~np.isnan(m_f)
    if valid.sum() < 2:
        return float('nan'), float('nan'), float('nan')
    Dv = Ds_f[valid]; mv = m_f[valid]
    inv = 1.0 / Dv
    n = min(n_extrap, len(Dv))
    try:
        if n >= 3:
            coeffs, cov = np.polyfit(inv[-n:], mv[-n:], 1, cov=True)
            sigma_b = float(np.sqrt(abs(cov[1, 1])))
        else:
            coeffs  = np.polyfit(inv[-n:], mv[-n:], 1)
            sigma_b = 0.0
    except Exception:
        return float(mv[-1]), float(mv[-1]), float(mv[-1])
    b     = float(coeffs[1])
    b_lo  = max(0.0, b - sigma_b)
    b_hi  = max(0.0, b + sigma_b)
    return b, b_lo, b_hi


def _all_ansatze(all_data):
    keys = set()
    for v_map in all_data.values():
        keys.update(v_map.keys())
    return ([a for a in ANSATZ_ORDER if a in keys] +
            sorted(a for a in keys if a not in ANSATZ_ORDER))


def _all_Ds_for_ansatz(all_data, ansatz_key):
    Ds = set()
    for v_map in all_data.values():
        if ansatz_key in v_map:
            Ds.update(v_map[ansatz_key]['Ds'])
    return sorted(Ds)


# ── Drawing helpers ────────────────────────────────────────────────────────────

def _draw_E_raw(ax, all_data, ansatz_key):
    all_Ds = _all_Ds_for_ansatz(all_data, ansatz_key)
    n_D    = max(len(all_Ds) - 1, 1)
    for D in all_Ds:
        j2s, Es = [], []
        for j2 in sorted(all_data.keys()):
            if ansatz_key not in all_data[j2]:
                continue
            v = all_data[j2][ansatz_key]
            if D in v['Ds']:
                idx = v['Ds'].index(D)
                j2s.append(j2); Es.append(v['energy_per_site'][idx])
        if not j2s:
            continue
        alpha = 0.25 + 0.75 * all_Ds.index(D) / n_D
        ax.plot(j2s, Es, 'o-', color='#1f77b4', alpha=alpha, ms=5, lw=1.4,
                label=f'D={D}')
    ax.set_xlabel('J₂', fontsize=10)
    ax.set_ylabel('Energy per site', fontsize=10)
    ax.set_title(ANSATZ_LABEL.get(ansatz_key, ansatz_key), fontsize=11)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.5g'))
    ax.legend(fontsize=8)


def _draw_E_extrap(ax, all_data, ansatz_key):
    """Asymmetric errorbars: center=E_exp (or E_lin3), upper=E_horiz, lower=E_lin3.
    Skip points where any of these values is None (i.e. insufficient D data)."""
    j2s, centers, uppers, lowers = [], [], [], []
    for j2 in sorted(all_data.keys()):
        if ansatz_key not in all_data[j2]:
            continue
        ex       = all_data[j2][ansatz_key]['extrap']
        E_center = ex['E_exp'] if ex['E_exp'] is not None else ex['E_lin3']
        E_upper  = ex['E_horiz']
        E_lower  = ex['E_lin3']
        # Skip if any essential quantity is missing (e.g. only one D)
        if E_center is None or E_upper is None or E_lower is None:
            continue
        j2s.append(j2)
        centers.append(E_center)
        uppers.append(max(0.0, E_upper - E_center))
        lowers.append(max(0.0, E_center - E_lower))
    if not j2s:
        return
    ax.errorbar(j2s, centers, yerr=[lowers, uppers],
                fmt='*-', color='#1f77b4', ms=10, lw=1.4,
                capsize=4, elinewidth=1.1,
                markeredgewidth=0.5, markeredgecolor='k',
                label='extrap (exp|lin3 center, last-D upper, lin3 lower)')
    ax.set_xlabel('J₂', fontsize=10)
    ax.set_ylabel('Energy per site', fontsize=10)
    ax.set_title(ANSATZ_LABEL.get(ansatz_key, ansatz_key), fontsize=11)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.5g'))
    ax.legend(fontsize=7)


def _draw_m_raw(ax, all_data, ansatz_key):
    all_Ds = _all_Ds_for_ansatz(all_data, ansatz_key)
    n_D    = max(len(all_Ds) - 1, 1)
    for D in all_Ds:
        j2s, ms = [], []
        for j2 in sorted(all_data.keys()):
            if ansatz_key not in all_data[j2]:
                continue
            v = all_data[j2][ansatz_key]
            if D in v['Ds']:
                idx = v['Ds'].index(D)
                j2s.append(j2); ms.append(v['mneel_list'][idx])
        if not j2s:
            continue
        alpha = 0.25 + 0.75 * all_Ds.index(D) / n_D
        ax.plot(j2s, ms, 'o-', color='#9467bd', alpha=alpha, ms=5, lw=1.4,
                label=f'D={D}')
    ax.set_xlabel('J₂', fontsize=10)
    ax.set_ylabel(r'm$_\mathrm{N\acute{e}el}$', fontsize=10)
    ax.set_ylim(bottom=0.0)
    ax.set_title(ANSATZ_LABEL.get(ansatz_key, ansatz_key), fontsize=11)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4g'))
    ax.legend(fontsize=8)


def _draw_m_extrap(ax, all_data, ansatz_key):
    """Asymmetric errorbars from lin3 polyfit intercept ± sigma_b, clipped to >=0."""
    j2s, centers, lowers, uppers = [], [], [], []
    for j2 in sorted(all_data.keys()):
        if ansatz_key not in all_data[j2]:
            continue
        v = all_data[j2][ansatz_key]
        b, b_lo, b_hi = compute_m_lin3_extrap_with_error(
            v['Ds'], v['mneel_list'], n_extrap=v.get('m_n_extrap', 3))
        if np.isnan(b):
            continue
        b_c = max(0.0, b)
        j2s.append(j2)
        centers.append(b_c)
        lowers.append(b_c - b_lo)          # b_lo already clipped >=0
        uppers.append(b_hi - b_c)
    if not j2s:
        return
    ax.errorbar(j2s, centers, yerr=[lowers, uppers],
                fmt='*-', color='#9467bd', ms=10, lw=1.4,
                capsize=4, elinewidth=1.1,
                markeredgewidth=0.5, markeredgecolor='k',
                label='lin3 extrap ± σ (clipped ≥0)')
    ax.set_xlabel('J₂', fontsize=10)
    ax.set_ylabel(r'm$_\mathrm{N\acute{e}el}$', fontsize=10)
    ax.set_ylim(bottom=0.0)
    ax.set_title(ANSATZ_LABEL.get(ansatz_key, ansatz_key), fontsize=11)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4g'))
    ax.legend(fontsize=7)


# ── Three-figure generators ───────────────────────────────────────────────────

def plot_e_vs_j2_figures(all_data, out_dir):
    ansatze = _all_ansatze(all_data)
    n = len(ansatze)

    fig1, axes1 = plt.subplots(1, n, figsize=(5 * n, 4.5), squeeze=False)
    for i, a in enumerate(ansatze):
        _draw_E_raw(axes1[0, i], all_data, a)
    fig1.suptitle('Energy per site vs J₂  —  raw D curves', fontsize=13)
    fig1.tight_layout()
    _save(fig1, os.path.join(out_dir, 'summary_E_vs_J2_fig1_raw.pdf'))

    fig2, axes2 = plt.subplots(1, n, figsize=(5 * n, 4.5), squeeze=False)
    for i, a in enumerate(ansatze):
        _draw_E_extrap(axes2[0, i], all_data, a)
    fig2.suptitle('Energy per site vs J₂  —  extrapolated', fontsize=13)
    fig2.tight_layout()
    _save(fig2, os.path.join(out_dir, 'summary_E_vs_J2_fig2_extrap.pdf'))

    fig3, axes3 = plt.subplots(1, n, figsize=(5 * n, 4.5), squeeze=False)
    for i, a in enumerate(ansatze):
        ax = axes3[0, i]
        _draw_E_raw(ax, all_data, a)
        _draw_E_extrap(ax, all_data, a)
        ax.legend(fontsize=7)
    fig3.suptitle('Energy per site vs J₂  —  combined', fontsize=13)
    fig3.tight_layout()
    _save(fig3, os.path.join(out_dir, 'summary_E_vs_J2_fig3_combined.pdf'))


def plot_m_vs_j2_figures(all_data, out_dir):
    ansatze = _all_ansatze(all_data)
    n = len(ansatze)

    fig1, axes1 = plt.subplots(1, n, figsize=(5 * n, 4.5), squeeze=False)
    for i, a in enumerate(ansatze):
        _draw_m_raw(axes1[0, i], all_data, a)
    fig1.suptitle(r'm$_\mathrm{N\acute{e}el}$ vs J₂  —  raw D curves', fontsize=13)
    fig1.tight_layout()
    _save(fig1, os.path.join(out_dir, 'summary_m_vs_J2_fig1_raw.pdf'))

    fig2, axes2 = plt.subplots(1, n, figsize=(5 * n, 4.5), squeeze=False)
    for i, a in enumerate(ansatze):
        _draw_m_extrap(axes2[0, i], all_data, a)
    fig2.suptitle(r'm$_\mathrm{N\acute{e}el}$ vs J₂  —  lin3 extrapolated ± σ', fontsize=13)
    fig2.tight_layout()
    _save(fig2, os.path.join(out_dir, 'summary_m_vs_J2_fig2_extrap.pdf'))

    fig3, axes3 = plt.subplots(1, n, figsize=(5 * n, 4.5), squeeze=False)
    for i, a in enumerate(ansatze):
        ax = axes3[0, i]
        _draw_m_raw(ax, all_data, a)
        _draw_m_extrap(ax, all_data, a)
        ax.set_ylim(bottom=0.0)
        ax.legend(fontsize=7)
    fig3.suptitle(r'm$_\mathrm{N\acute{e}el}$ vs J₂  —  combined', fontsize=13)
    fig3.tight_layout()
    _save(fig3, os.path.join(out_dir, 'summary_m_vs_J2_fig3_combined.pdf'))


# ── Part 4: ΔNN (rank3 − rank1) vs J2 ────────────────────────────────────────

D_CORR_SHOW = [3,4,5,6,7,8, 9, 10]   # only these D values; skip if absent
CORR_ANSATZ_SKIP = {'neel_symmetrized'}   # excluded ansatze

D_CORR_COLORS = {3:"#fbfabc",4:"#ffefa9",5:"#fed78d",6:"#f9b664",7:"#db7a24",8: "#ae4829", 9: "#731608", 10: "#3B0202"}   # Oranges


def _rank_mean(nn_grp_D, target_rank):
    """Mean of all group means that were assigned target_rank for a given D entry."""
    vals = [nn_grp_D['means'][g]
            for g, r in enumerate(nn_grp_D['ranks']) if r == target_rank]
    return float(np.mean(vals)) if vals else float('nan')


def _draw_delta_nn(ax, all_data, ansatz_key):
    """Plot (rank3 − rank1) NN bond mean vs J2 for D in D_CORR_SHOW."""
    for D in D_CORR_SHOW:
        j2s, deltas = [], []
        for j2 in sorted(all_data.keys()):
            if ansatz_key not in all_data[j2]:
                continue
            v = all_data[j2][ansatz_key]
            if D not in v['nn_groups']:
                continue
            entry = v['nn_groups'][D]
            r1 = _rank_mean(entry, 1)
            r3 = _rank_mean(entry, 3)
            if np.isnan(r1) or np.isnan(r3):
                continue
            j2s.append(j2)
            deltas.append(r3 - r1)
        if not j2s:
            continue
        ax.plot(j2s, deltas, 'o-', color=D_CORR_COLORS.get(D, 'tab:orange'),
                ms=6, lw=1.4, label=f'D={D}')

    ax.set_xlabel('J₂', fontsize=10)
    ax.set_ylabel(r'$\Delta_\mathrm{NN}$ = rank3 $-$ rank1', fontsize=10)
    ax.set_ylim(bottom=0.0)
    ax.set_title(ANSATZ_LABEL.get(ansatz_key, ansatz_key), fontsize=11)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4g'))
    ax.legend(fontsize=8)


def plot_delta_nn_vs_j2(all_data, out_dir):
    ansatze = [a for a in _all_ansatze(all_data) if a not in CORR_ANSATZ_SKIP]
    if not ansatze:
        print("  (no eligible ansatze for ΔNN plot)")
        return
    n = len(ansatze)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), squeeze=False)
    for i, a in enumerate(ansatze):
        _draw_delta_nn(axes[0, i], all_data, a)
    fig.suptitle(r'NN bond splitting $\Delta_\mathrm{NN}$ vs J₂  (D=8,9,10)', fontsize=13)
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, 'summary_deltaNNN_vs_J2.pdf'))


# ──────────────────────────────────────────────────────────────────
# Extra: m_Néel extrapolated vs J2 for Néel ansatz, J2 ∈ [0.20, 0.28]
# ──────────────────────────────────────────────────────────────────
def plot_m_vs_j2_neel_020_028(all_data, out_dir):
    # Filter J2 values in [0.20, 0.28]
    j2s_subset = sorted([j2 for j2 in all_data if 0.20 <= j2 <= 0.28 and 'neel_symmetrized' in all_data[j2]])
    if not j2s_subset:
        print("No Néel data in the range 0.20 ≤ J2 ≤ 0.28")
        return

    # Temporary dict containing only the relevant J2 + ansatz entries
    subset = {j2: {'neel_symmetrized': all_data[j2]['neel_symmetrized']} for j2 in j2s_subset}

    fig, ax = plt.subplots(figsize=(4, 3))
    _draw_m_extrap(ax, subset, 'neel_symmetrized')   # exactly the same drawing as in the summary figure

    ax.set_xlim(0.195, 0.285)          # tight x‑limits around the requested interval
    ax.set_ylim(bottom=0.0)            # as in the original, bottom at zero
    ax.set_title(r'm$_\mathrm{N\acute{e}el}$ extrapolated (Néel, $0.20 \leq J_2 \leq 0.28$)', fontsize=12)
    fig.tight_layout()

    fpath = os.path.join(out_dir, 'm_neel_extrap_J2_020_028.pdf')
    fig.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {os.path.relpath(fpath)}")



# ─────────────────────────────────────────────────────────────────
# Extra: ΔNN (rank3 − rank1) for 2C3 (D=8) vs C6Yπ (D=9)
# ─────────────────────────────────────────────────────────────────
def plot_delta_2C3_D8_vs_C6Ypi_D9(all_data, out_dir):
    ansatz_D_spec = [
        ('2tensor_twoC3', 8,  '#fdae6b', '^', '2 C3 (D=8)'),      # light orange, triangle
        ('1tensor_C6Ypi', 9,  '#d94801', 'H', 'C6Yπ (D=9)'),      # dark orange, hexagon
    ]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    any_data = False

    for ansatz_key, D_target, color, marker, label in ansatz_D_spec:
        j2s, deltas = [], []
        for j2 in sorted(all_data.keys()):
            if ansatz_key not in all_data[j2]:
                continue
            v = all_data[j2][ansatz_key]
            if D_target not in v['nn_groups']:
                continue
            entry = v['nn_groups'][D_target]
            r1 = _rank_mean(entry, 1)
            r3 = _rank_mean(entry, 3)
            if np.isnan(r1) or np.isnan(r3):
                continue
            j2s.append(j2)
            deltas.append(r3 - r1)
        if not j2s:
            continue
        any_data = True
        ax.plot(j2s, deltas, marker=marker, linestyle='-', color=color,
                ms=8, lw=1.4, label=label)

    if not any_data:
        print("  No data for the requested 2C3/D=8 or C6Yπ/D=9 ΔNN plot.")
        plt.close(fig)
        return

    ax.set_xlabel('J₂', fontsize=12)
    ax.set_ylabel(r'$\Delta_\mathrm{NN}$ = rank3 $-$ rank1', fontsize=12)
    ax.set_title(r'NN bond splitting: 2C3 (D=8) vs C6Y$\pi$ (D=9)', fontsize=12)
    ax.set_ylim(bottom=0.0)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4g'))
    ax.legend(fontsize=9)

    fig.tight_layout()
    fpath = os.path.join(out_dir, 'delta_NN_2C3_D8_vs_C6Ypi_D9.pdf')
    _save(fig, fpath)

def plot_m_extrap_and_delta_combined(all_data, out_dir):
    ansatz_neel = 'neel_symmetrized'
    ansatz_2c3  = '2tensor_twoC3'
    ansatz_c6y  = '1tensor_C6Ypi'

    # --- 1. Néel magnetization extrapolation (left axis) ---
    j2s_subset = sorted([j2 for j2 in all_data
                         if 0.20 <= j2 <= 0.32 and ansatz_neel in all_data[j2]])
    if not j2s_subset:
        print("No Néel data in [0.20,0.32] for combined plot.")
        return

    j2_m, m_c, m_lo, m_hi = [], [], [], []
    for j2 in j2s_subset:
        v = all_data[j2][ansatz_neel]
        b, b_lo, b_hi = compute_m_lin3_extrap_with_error(
            v['Ds'], v['mneel_list'], n_extrap=v.get('m_n_extrap', 3))
        if np.isnan(b):
            continue
        b_c = max(0.0, b)
        j2_m.append(j2)
        m_c.append(b_c)
        m_lo.append(b_c - b_lo)   # b_lo already ≥0
        m_hi.append(b_hi - b_c)

    # --- 2. ΔNN data (right axis) ---
    delta_2c3_x, delta_2c3_y = [], []
    delta_c6y_x, delta_c6y_y = [], []
    for j2 in sorted(all_data.keys()):
        if not (0.20 <= j2 <= 0.32):
            continue
        if ansatz_2c3 in all_data[j2]:
            v = all_data[j2][ansatz_2c3]
            if 8 in v['nn_groups']:
                entry = v['nn_groups'][8]
                r1 = _rank_mean(entry, 1)
                r3 = _rank_mean(entry, 3)
                if not np.isnan(r1) and not np.isnan(r3):
                    delta_2c3_x.append(j2)
                    delta_2c3_y.append(r3 - r1)
        if ansatz_c6y in all_data[j2]:
            v = all_data[j2][ansatz_c6y]
            if 9 in v['nn_groups']:
                entry = v['nn_groups'][9]
                r1 = _rank_mean(entry, 1)
                r3 = _rank_mean(entry, 3)
                if not np.isnan(r1) and not np.isnan(r3):
                    delta_c6y_x.append(j2)
                    delta_c6y_y.append(r3 - r1)

    # --- 3. Create figure with twin axes ---
    fig, ax1 = plt.subplots(figsize=(7, 5))

    # Left axis: magnetization
    ax1.errorbar(j2_m, m_c, yerr=[m_lo, m_hi],
                 fmt='*-', color='#9467bd', ms=10, lw=1.4,
                 capsize=4, elinewidth=1.1,
                 label=r'm$_\mathrm{N\acute{e}el}$ extrap (lin3)')
    ax1.set_xlabel('J₂', fontsize=12)
    ax1.set_ylabel(r'm$_\mathrm{N\acute{e}el}$', fontsize=12, color='#9467bd')
    ax1.tick_params(axis='y', labelcolor='#9467bd')

    # Right axis: ΔNN
    ax2 = ax1.twinx()
    ax2.plot(delta_2c3_x, delta_2c3_y, marker='^', linestyle='-',
             color='#fdae6b', ms=8, lw=1.4, label='2 C3 D=8')
    ax2.plot(delta_c6y_x, delta_c6y_y, marker='H', linestyle='-',
             color='#d94801', ms=8, lw=1.4, label='C6Yπ D=9')
    ax2.set_ylabel(r'$\Delta_\mathrm{NN}$ (rank3 $-$ rank1)', fontsize=12, color='#d94801')
    ax2.tick_params(axis='y', labelcolor='#d94801')

    # --- 4. Adjust both y-limits: bottom = 0, top = max(data) * 1.1 ---
    # Left axis top
    if m_hi:
        left_max = max(c + u for c, u in zip(m_c, m_hi))
    else:
        left_max = max(m_c) if m_c else 0.0
    ax1.set_ylim(0, left_max * 1.1 if left_max > 0 else 0.1)

    # Right axis top
    all_delta = delta_2c3_y + delta_c6y_y
    if all_delta:
        right_max = max(all_delta)
    else:
        right_max = 0.0
    ax2.set_ylim(0, right_max * 1.1 if right_max > 0 else 0.1)

    # --- 5. Combined legend ---
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower left', fontsize=8)

    ax1.set_title(r'Néel m$_\mathrm{N\acute{e}el}$ extrap & NN splitting (0.20 ≤ J₂ ≤ 0.32)', fontsize=12)
    fig.tight_layout()

    fpath = os.path.join(out_dir, 'combined_m_extrap_delta_020_032.pdf')
    _save(fig, fpath)

def plot_energy_Neel_c6_88(all_data, out_dir):
    ansatz_D_spec = [
        ('neel_symmetrized', 8,  "#6b40ee", '^', 'Néel (D=8)'),
        ('1tensor_C6Ypi', 8,  '#d94801', 'H', 'C6Yπ (D=8)'),
    ]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    any_data = False

    for ansatz_key, D_target, color, marker, label in ansatz_D_spec:
        j2s, energies = [], []
        for j2 in sorted(all_data.keys()):
            if ansatz_key not in all_data[j2]:
                continue
            v = all_data[j2][ansatz_key]
            # v['Ds'] is a sorted list of D values; find the index of D_target
            if D_target not in v['Ds']:
                continue
            idx = v['Ds'].index(D_target)
            energy_val = v['energy_per_site'][idx]
            j2s.append(j2)
            energies.append(energy_val)

        if not j2s:
            continue
        any_data = True
        ax.plot(j2s, energies, marker=marker, linestyle='-', color=color,
                ms=8, lw=1.4, label=label)

    if not any_data:
        print("  No data for the requested Neel/D=8 or C6Yπ/D=8 energy plot.")
        plt.close(fig)
        return

    ax.set_xlabel('J₂', fontsize=12)
    ax.set_ylabel('Energy per site', fontsize=12)
    ax.set_title(r'Energy: Néel (D=8) vs C6Y$\pi$ (D=8)', fontsize=12)
    # Remove the fixed bottom=0; energy is negative, let matplotlib auto-scale
    # ax.set_ylim(bottom=0.0)   # <-- not appropriate for energy
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.5g'))
    ax.legend(fontsize=9)

    fig.tight_layout()
    fpath = os.path.join(out_dir, 'energy_Neel_D8_vs_C6Ypi_D8.pdf')
    _save(fig, fpath)


def plot_energy_2c3_c6_88(all_data, out_dir):
    ansatz_D_spec = [
        ('2tensor_twoC3', 8,  "#6b40ee", '^', 'twoC3 (D=8)'),
        ('1tensor_C6Ypi', 8,  '#d94801', 'H', 'C6Yπ (D=8)'),
    ]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    any_data = False

    for ansatz_key, D_target, color, marker, label in ansatz_D_spec:
        j2s, energies = [], []
        for j2 in sorted(all_data.keys()):
            if ansatz_key not in all_data[j2]:
                continue
            v = all_data[j2][ansatz_key]
            # v['Ds'] is a sorted list of D values; find the index of D_target
            if D_target not in v['Ds']:
                continue
            idx = v['Ds'].index(D_target)
            energy_val = v['energy_per_site'][idx]
            j2s.append(j2)
            energies.append(energy_val)

        if not j2s:
            continue
        any_data = True
        ax.plot(j2s, energies, marker=marker, linestyle='-', color=color,
                ms=8, lw=1.4, label=label)

    if not any_data:
        print("  No data for the requested Neel/D=8 or C6Yπ/D=8 energy plot.")
        plt.close(fig)
        return

    ax.set_xlabel('J₂', fontsize=12)
    ax.set_ylabel('Energy per site', fontsize=12)
    ax.set_title(r'Energy: twoC3 (D=8) vs C6Y$\pi$ (D=8)', fontsize=12)
    # Remove the fixed bottom=0; energy is negative, let matplotlib auto-scale
    # ax.set_ylim(bottom=0.0)   # <-- not appropriate for energy
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.5g'))
    ax.legend(fontsize=9)

    fig.tight_layout()
    fpath = os.path.join(out_dir, 'energy_twoC3_D8_vs_C6Ypi_D8.pdf')
    _save(fig, fpath)

def plot_energy_Neel_c6_89(all_data, out_dir):
    ansatz_D_spec = [
        ('neel_symmetrized', 8,  "#6b40ee", '^', 'Néel (D=8)'),
        ('1tensor_C6Ypi', 9,  '#d94801', 'H', 'C6Yπ (D=9)'),
    ]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    any_data = False

    for ansatz_key, D_target, color, marker, label in ansatz_D_spec:
        j2s, energies = [], []
        for j2 in sorted(all_data.keys()):
            if ansatz_key not in all_data[j2]:
                continue
            v = all_data[j2][ansatz_key]
            # v['Ds'] is a sorted list of D values; find the index of D_target
            if D_target not in v['Ds']:
                continue
            idx = v['Ds'].index(D_target)
            energy_val = v['energy_per_site'][idx]
            j2s.append(j2)
            energies.append(energy_val)

        if not j2s:
            continue
        any_data = True
        ax.plot(j2s, energies, marker=marker, linestyle='-', color=color,
                ms=8, lw=1.4, label=label)

    if not any_data:
        print("  No data for the requested Neel/D=8 or C6Yπ/D=9 energy plot.")
        plt.close(fig)
        return

    ax.set_xlabel('J₂', fontsize=12)
    ax.set_ylabel('Energy per site', fontsize=12)
    ax.set_title(r'Energy: Néel (D=8) vs C6Y$\pi$ (D=9)', fontsize=12)
    # Remove the fixed bottom=0; energy is negative, let matplotlib auto-scale
    # ax.set_ylim(bottom=0.0)   # <-- not appropriate for energy
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.5g'))
    ax.legend(fontsize=9)

    fig.tight_layout()
    fpath = os.path.join(out_dir, 'energy_Neel_D8_vs_C6Ypi_D9.pdf')
    _save(fig, fpath)

def plot_energy_extrap_Neel_C6(all_data, out_dir):
    """Extrapolated energy (E_best) with error bars for Néel and C6Yπ."""
    ansatz_keys = [
        ('neel_symmetrized', '#6b40ee', '^', 'Néel'),
        ('1tensor_C6Ypi', '#d94801', 'H', 'C6Yπ'),
    ]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    any_data = False

    for ansatz_key, color, marker, label in ansatz_keys:
        j2s, centers, uppers, lowers = [], [], [], []
        for j2 in sorted(all_data.keys()):
            if ansatz_key not in all_data[j2]:
                continue
            ex = all_data[j2][ansatz_key]['extrap']
            E_center = ex['E_exp'] if ex['E_exp'] is not None else ex['E_lin3']
            E_upper  = ex['E_horiz']
            E_lower  = ex['E_lin3']
            # Skip if any essential quantity is missing
            if E_center is None or E_upper is None or E_lower is None:
                continue
            j2s.append(j2)
            centers.append(E_center)
            uppers.append(max(0.0, E_upper - E_center))
            lowers.append(max(0.0, E_center - E_lower))

        if not j2s:
            continue
        any_data = True
        ax.errorbar(j2s, centers, yerr=[lowers, uppers],
                    fmt='-', marker=marker, color=color,
                    ms=8, lw=1.4, capsize=4, elinewidth=1.1,
                    markeredgewidth=0.5, markeredgecolor='k',
                    label=label)

    if not any_data:
        print("  No extrapolated energy data for Néel or C6Yπ.")
        plt.close(fig)
        return

    ax.set_xlabel('J₂', fontsize=12)
    ax.set_ylabel('Extrapolated energy per site', fontsize=12)
    ax.set_title('Energy extrapolation: Néel vs C6Yπ', fontsize=12)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.5g'))
    ax.legend(fontsize=9)

    fig.tight_layout()
    fpath = os.path.join(out_dir, 'energy_extrap_Neel_vs_C6Ypi.pdf')
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

    #plot_energy_Neel_c6(all_data, OUT_DIR)
    plot_energy_2c3_c6_88(all_data, OUT_DIR)
    plot_energy_Neel_c6_88(all_data, OUT_DIR)
    plot_energy_Neel_c6_89(all_data, OUT_DIR)
    plot_energy_extrap_Neel_C6(all_data, OUT_DIR)
    print(f"\nFound {len(all_data)} J2 values: {sorted(all_data.keys())}")
    print(f"Generating {len(all_data)} per-J2 figures ...\n")

    for j2, ansatz_map in sorted(all_data.items()):
        ansatze = sorted(ansatz_map.keys())
        print(f"  J2={j2:.4g}  ansatze: {ansatze}")
        plot_j2_figure(j2, ansatz_map, OUT_DIR)

    print("\nGenerating E vs J2 summary figures ...")
    plot_e_vs_j2_figures(all_data, OUT_DIR)

    print("Generating m vs J2 summary figures ...")
    plot_m_vs_j2_figures(all_data, OUT_DIR)

    print("Generating ΔNN vs J2 figure ...")
    plot_delta_nn_vs_j2(all_data, OUT_DIR)

    plot_m_vs_j2_neel_020_028(all_data, OUT_DIR)
    plot_delta_2C3_D8_vs_C6Ypi_D9(all_data, OUT_DIR)
    plot_m_extrap_and_delta_combined(all_data, OUT_DIR)
    print(f"\nDone.  Figures in: {OUT_DIR}")



if __name__ == '__main__':
    main()
