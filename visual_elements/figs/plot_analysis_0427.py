#!/usr/bin/env python3
"""
plot_analysis_0427.py
Single-ansatz (C6Ypi) analysis over J2 = 0.20 .. 0.30 (11 points).

Outputs (in analysis_plots_0427/)
──────────────────────────────────
  overview_J2_{jstr}.pdf   — 3-panel per J2 (energy / magnetization / NN bonds),
                              all panels share the same 1/D x-ticks.
  summary_vs_J2.pdf        — 2-panel statistics vs J2 (energy; order + ΔNN).
"""
import os
import re
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.optimize import curve_fit

# ──────────────────────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────────────────────
DATA_DIR   = '/home/chye/6ADctmrg/data/0420core'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(SCRIPT_DIR, 'analysis_plots_0427')
os.makedirs(OUT_DIR, exist_ok=True)

FOLDER_MAP = {
    0.20: '1tensor_C6Ypi__J2_0p2_20260423_185210',
    0.21: '1tensor_C6Ypi__J2_0p21_20260423_185249',
    0.22: '1tensor_C6Ypi__J2_0p22_20260423_185346',
    0.23: '1tensor_C6Ypi__J2_0p23_20260423_185446',
    0.24: '1tensor_C6Ypi__J2_0p24_20260423_185748',
    0.25: '1tensor_C6Ypi__J2_0p25_20260423_192207',
    0.26: '1tensor_C6Ypi__J2_0p26_20260425_025529',
    0.27: '1tensor_C6Ypi__J2_0p27_20260425_025529',
    0.28: '1tensor_C6Ypi__J2_0p28_20260425_030516',
    0.29: '1tensor_C6Ypi__J2_0p29_20260423_192250',
    0.30: '1tensor_C6Ypi__J2_0p3_20260423_193610',
}

J2_LIST = sorted(FOLDER_MAP.keys())

# Bond group raw definitions (env-index, pair-label)
NN_GROUPS_RAW = [
    [(1,'EB'),(1,'AD'),(1,'CF'),(3,'BE'),(3,'FC'),(3,'DA')],
    [(2,'CB'),(2,'AF'),(2,'ED'),(1,'FA'),(1,'DE'),(1,'BC')],
    [(3,'EF'),(3,'AB'),(3,'CD'),(2,'DC'),(2,'BA'),(2,'FE')],
]

RANK_COLORS = {1: 'tab:red', 2: 'tab:green', 3: 'tab:blue'}
RANK_LABELS = {1: 'rank 1 (most neg)', 2: 'rank 2 (mid)', 3: 'rank 3 (least neg)'}

# Summary plot: which D values to show explicitly
D_SHOW    = [6, 7, 8]
D_MARKERS = {6: '^', 7: 's', 8: 'D'}
D_ALPHAS  = {6: 0.35, 7: 0.60, 8: 0.85}

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
# 1. File parsing
# ──────────────────────────────────────────────────────────────────────────────
def parse_plain_file(fpath):
    txt  = open(fpath).read()
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
    plain_map  = {}  # D → [(chi, path)]
    plus2d_map = {}  # (D, chi) → (eff_chi, path)
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
        if (D, best_chi) in plus2d_map:
            _, fpath = plus2d_map[(D, best_chi)]
        else:
            fpath = next(p for c, p in plain_map[D] if c == best_chi)
        try:
            result[D] = parse_plain_file(fpath)
        except Exception as e:
            print(f"  Warning: failed to parse {fpath}: {e}")
    return result

# ──────────────────────────────────────────────────────────────────────────────
# 2. Bond groups + order parameter
# ──────────────────────────────────────────────────────────────────────────────
def compute_bond_groups(D_data, groups_raw):
    result = {}
    for D in sorted(D_data.keys()):
        corr = D_data[D]['corr']
        means, stds = [], []
        for group in groups_raw:
            vals = [corr[(env, pair)] for (env, pair) in group if (env, pair) in corr]
            means.append(float(np.mean(vals)) if vals else float('nan'))
            stds.append(float(np.std(vals, ddof=0)) if vals else float('nan'))
        order  = np.argsort(means)
        ranks  = [0] * len(means)
        for rank_idx, g_idx in enumerate(order):
            ranks[g_idx] = rank_idx + 1
        result[D] = {'means': means, 'stds': stds, 'ranks': ranks}
    return result


def compute_order_param(D_data):
    result = {}
    for D in sorted(D_data.keys()):
        mag = D_data[D].get('mag', {})
        sz_ace, sz_bdf = [], []
        for env in [1, 2, 3]:
            for site in 'ACE':
                k = (env, site)
                if k in mag:
                    sz_ace.append(mag[k]['Sz'])
            for site in 'BDF':
                k = (env, site)
                if k in mag:
                    sz_bdf.append(mag[k]['Sz'])
        m_ace  = float(np.mean(sz_ace)) if sz_ace else float('nan')
        m_bdf  = float(np.mean(sz_bdf)) if sz_bdf else float('nan')
        m_neel = (0.5 * abs(m_ace - m_bdf)
                  if not (np.isnan(m_ace) or np.isnan(m_bdf)) else float('nan'))
        result[D] = {'m_ace': m_ace, 'm_bdf': m_bdf, 'm_neel': m_neel}
    return result

# ──────────────────────────────────────────────────────────────────────────────
# 3. Extrapolation helpers
# ──────────────────────────────────────────────────────────────────────────────
def _exp_model(D, E0, c, Dchar):
    return E0 + c * np.exp(-np.asarray(D, dtype=float) / Dchar)


def compute_energy_extrap(Ds, eps):
    """
    Returns dict with:
      E_horiz   — value at last (largest) D
      E_lin3    — linear fit (1/D) of last 3 Ds, extrapolated to 0
      E_lin_all — linear fit of all Ds
      E_exp     — exp fit of all Ds (None if unavailable)
      E_best    — preferred extrapolation (exp if sane, else lin3)
      E_err     — half-distance between lin3 and E_exp (0 if exp unavailable)
      exp_popt  — [E0, c, Dchar] for exp fit (None if failed)
    """
    Ds_f  = np.array(Ds, dtype=float)
    eps_f = np.array(eps, dtype=float)
    inv   = 1.0 / Ds_f

    E_horiz = float(eps_f[-1])

    # Linear fit: last 3 Ds
    n3 = min(3, len(Ds_f))
    c3 = np.polyfit(inv[-n3:], eps_f[-n3:], 1)
    E_lin3 = float(c3[1])

    # Linear fit: all Ds
    c_all  = np.polyfit(inv, eps_f, 1) if len(Ds_f) >= 2 else c3
    E_lin_all = float(c_all[1])

    # Exponential fit: all Ds
    E_exp    = None
    exp_popt = None
    if len(Ds_f) >= 3:
        try:
            p0 = [eps_f[-1] - 0.1, 0.1, float(Ds_f.mean())]
            popt, _ = curve_fit(_exp_model, Ds_f, eps_f, p0=p0, maxfev=5000)
            E_exp    = float(popt[0])
            exp_popt = popt.tolist()
        except Exception:
            pass

    # Best estimate
    if E_exp is not None:
        lo = min(E_horiz, E_lin3) - 2 * abs(E_horiz - E_lin3) - 0.3
        hi = max(E_horiz, E_lin3) + 2 * abs(E_horiz - E_lin3) + 0.3
        E_best = E_exp if lo <= E_exp <= hi else E_lin3
    else:
        E_best = E_lin3

    E_err = (0.5 * abs(E_lin3 - E_exp)
             if (E_exp is not None and E_best == E_exp)
             else 0.0)

    return {
        'E_horiz':   E_horiz,
        'E_lin3':    E_lin3,
        'E_lin_all': E_lin_all,
        'E_exp':     E_exp,
        'E_best':    E_best,
        'E_err':     E_err,
        'exp_popt':  exp_popt,
    }


def compute_mag_extrap(Ds, mneel_list):
    """
    Linear extrapolation of m_Néel to 1/D → 0.
    Returns (m_lin2, m_lin3, m_star, m_err):
      m_lin2 — intercept of lin fit using last 2 Ds
      m_lin3 — intercept of lin fit using last 3 Ds
      m_star — midpoint → best estimate
      m_err  — half-range → error bar half-width
    """
    Ds_f   = np.array(Ds, dtype=float)
    m_f    = np.array(mneel_list, dtype=float)
    inv    = 1.0 / Ds_f

    n2 = min(2, len(Ds_f))
    c2 = np.polyfit(inv[-n2:], m_f[-n2:], 1)
    m_lin2 = float(c2[1])

    n3 = min(3, len(Ds_f))
    c3 = np.polyfit(inv[-n3:], m_f[-n3:], 1)
    m_lin3 = float(c3[1])

    m_star = 0.5 * (m_lin2 + m_lin3)
    m_err  = 0.5 * abs(m_lin2 - m_lin3)
    return m_lin2, m_lin3, m_star, m_err

# ──────────────────────────────────────────────────────────────────────────────
# 4. Load all data
# ──────────────────────────────────────────────────────────────────────────────
def load_all_data():
    """Returns dict keyed by j2, each value is the full per-j2 data dict."""
    data = {}
    for j2 in J2_LIST:
        folder_name = FOLDER_MAP[j2]
        folder_path = os.path.join(DATA_DIR, folder_name)
        if not os.path.isdir(folder_path):
            print(f"  WARNING: folder not found: {folder_path}")
            continue
        print(f"  Loading J2={j2:.2f} from {folder_name} ...")
        D_data = load_folder_data(folder_path)
        if not D_data:
            print(f"    (no data files found)")
            continue
        Ds  = sorted(D_data.keys())
        eps = [D_data[D]['energy_per_site'] for D in Ds]
        extrap      = compute_energy_extrap(Ds, eps)
        nn_groups   = compute_bond_groups(D_data, NN_GROUPS_RAW)
        order       = compute_order_param(D_data)
        mneel_list  = [order[D]['m_neel'] for D in Ds]
        m_lin2, m_lin3, m_star, m_err = compute_mag_extrap(Ds, mneel_list)
        data[j2] = {
            'Ds':                Ds,
            'energy_per_site':   eps,
            'extrap':            extrap,
            'nn_groups':         nn_groups,
            'order':             order,
            'mneel_list':        mneel_list,
            'm_lin2':            m_lin2,
            'm_lin3':            m_lin3,
            'm_star':            m_star,
            'm_err':             m_err,
        }
    return data

# ──────────────────────────────────────────────────────────────────────────────
# 5. Utilities
# ──────────────────────────────────────────────────────────────────────────────
def _save(fig, fpath):
    fig.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {os.path.relpath(fpath)}")


def _j2_fname(j2):
    s = f'{j2:.10f}'.rstrip('0')
    dot = s.index('.')
    if len(s) - dot - 1 < 2:
        s += '0' * (2 - (len(s) - dot - 1))
    return s.replace('.', 'p')


def _jsonify(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(x) for x in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

# ──────────────────────────────────────────────────────────────────────────────
# 6a. Global y-limit computation (shared across all J2 and both plot types)
# ──────────────────────────────────────────────────────────────────────────────
def compute_global_ylims(all_data):
    """
    Scan every J2 entry and return tight global ylims for the three observables.
    Returns dict: {'energy': (lo, hi), 'mag': (lo, hi), 'nn': (lo, hi)}
    with small padding added.
    """
    all_e, all_m, all_nn = [], [], []
    for v in all_data.values():
        # Energy: raw data + extrapolation
        all_e.extend(v['energy_per_site'])
        all_e.append(v['extrap']['E_best'])
        # Mag: raw m_neel + all three extrapolation values (clipped >=0)
        vals_m = [x for x in v['mneel_list'] if not np.isnan(x)]
        all_m.extend(vals_m)
        for val in [v['m_lin2'], v['m_lin3'], v['m_star']]:
            all_m.append(max(0.0, val))
        # NN bond means
        for D_entry in v['nn_groups'].values():
            all_nn.extend(D_entry['means'])
    def _pad(lo, hi, frac=0.08):
        margin = max((hi - lo) * frac, 1e-6)
        return (lo - margin, hi + margin)
    e_ylim  = _pad(min(all_e), max(all_e))
    m_ylim  = _pad(0.0, max(all_m))   # always start from 0
    m_ylim  = (0.0, m_ylim[1])        # hard floor at 0
    nn_ylim = _pad(min(all_nn), max(all_nn))
    return {'energy': e_ylim, 'mag': m_ylim, 'nn': nn_ylim}


# ──────────────────────────────────────────────────────────────────────────────
# 6b. Per-J2 overview figure  (3 panels, shared 1/D x-axis)
# ──────────────────────────────────────────────────────────────────────────────
def plot_overview_j2(v, j2, out_dir, ylims=None):
    """
    3 vertically stacked panels sharing the 1/D x-axis.
      Panel 1: Energy per site vs 1/D
               • dots+line, horizontal line at last D,
               • dashed: linear fit of last 3 Ds,
               • dash-dot: exponential fit (all Ds),
               • star ★ extrapolated value at x=0
      Panel 2: m_Néel vs 1/D
               • dots+line, horizontal line at last D,
               • orange-dashed: linear fit of last 2 Ds (→ 0 extrapolation),
               • red-dashdot: linear fit of last 3 Ds (→ 0 extrapolation),
               • star ★ at midpoint of the two intercepts ± half-range
      Panel 3: NN correlation (3 rank groups) vs 1/D — raw dots+lines, no fit
    """
    Ds         = np.array(v['Ds'], dtype=float)
    inv        = 1.0 / Ds
    eps        = np.array(v['energy_per_site'])
    ex         = v['extrap']
    order      = v['order']
    nn_grp     = v['nn_groups']
    mneel_list = np.array(v['mneel_list'])
    m_lin2     = v['m_lin2']
    m_lin3     = v['m_lin3']
    m_star     = v['m_star']
    m_err      = v['m_err']

    x_max_plot = max(inv) * 1.25  # right edge of axes (slightly beyond largest 1/D)
    inv_line   = np.linspace(0, x_max_plot, 300)

    # ── Figure setup ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(7, 13),
                              gridspec_kw={'hspace': 0.08})
    ax_e, ax_m, ax_nn = axes

    # shared x-ticks: evenly spaced at 0.025, 0.05, 0.075, 0.10 ...
    tick_step = 0.025
    x_ticks = [0.0] + [round(tick_step * k, 6)
                        for k in range(1, int(np.ceil(x_max_plot / tick_step)) + 1)
                        if tick_step * k <= x_max_plot + 1e-9]
    for ax in axes:
        ax.set_xticks(x_ticks)
        ax.set_xlim(left=-0.002, right=x_max_plot)
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.3g'))

    # only bottom panel gets x label
    ax_e.set_xticklabels([])
    ax_m.set_xticklabels([])
    ax_nn.set_xlabel('1/D', fontsize=11)

    # ── Panel 1: Energy ───────────────────────────────────────────────────────
    col_e = '#1f77b4'  # tab:blue
    ax_e.plot(inv, eps, 'o-', color=col_e, ms=6, lw=1.6, zorder=5, label='E(D)')

    # Horizontal line: last D value
    ax_e.axhline(ex['E_horiz'], color=col_e, lw=0.9, ls=':', alpha=0.55,
                  label=f'E(D={int(Ds[-1])})')

    # Dashed: linear fit of last 3 Ds
    n3 = min(3, len(Ds))
    c3 = np.polyfit(inv[-n3:], eps[-n3:], 1)
    ax_e.plot(inv_line, np.polyval(c3, inv_line),
              color='tab:orange', lw=1.1, ls='--', alpha=0.85,
              label=f'lin fit (last {n3} D)')

    # Dash-dot: exponential fit (all Ds)
    if ex['exp_popt'] is not None:
        Ds_line = np.where(inv_line > 0, 1.0 / inv_line, np.inf)
        y_exp   = _exp_model(Ds_line, *ex['exp_popt'])
        ax_e.plot(inv_line, y_exp,
                  color='tab:green', lw=1.1, ls='-.', alpha=0.85,
                  label='exp fit (all D)')

    # Star at x=0
    ax_e.plot(0, ex['E_best'], '*', color=col_e, ms=14, zorder=7,
              markeredgewidth=0.5, markeredgecolor='k',
              label=f'extrap = {ex["E_best"]:.6f}')

    ax_e.set_ylabel('Energy per site', fontsize=11)
    ax_e.set_title(f'C6Yπ  |  J2 = {j2:.2f}', fontsize=12, pad=6)
    ax_e.legend(fontsize=8, loc='upper right')
    ax_e.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.6g'))
    if ylims is not None:
        ax_e.set_ylim(ylims['energy'])

    # ── Panel 2: Magnetization ────────────────────────────────────────────────
    valid = ~np.isnan(mneel_list)
    inv_v = inv[valid]; m_v = mneel_list[valid]
    col_m = '#9467bd'  # tab:purple

    ax_m.plot(inv, mneel_list, 'o-', color=col_m, ms=6, lw=1.6, zorder=5,
              label=r'm$_\mathrm{N\acute{e}el}$(D)')

    if len(m_v):
        # Horizontal line: last valid D
        ax_m.axhline(float(m_v[-1]), color=col_m, lw=0.9, ls=':', alpha=0.55,
                      label=f'm(D={int(1/inv_v[-1])})')

        # Dashed orange: lin fit last 2 Ds
        n2 = min(2, len(inv_v))
        c2 = np.polyfit(inv_v[-n2:], m_v[-n2:], 1)
        ax_m.plot(inv_line, np.polyval(c2, inv_line),
                  color='tab:orange', lw=1.1, ls='--', alpha=0.85,
                  label=f'lin fit (last {n2} D): intercept={m_lin2:.4f}')

        # Dash-dot red: lin fit last 3 Ds
        n3m = min(3, len(inv_v))
        c3m = np.polyfit(inv_v[-n3m:], m_v[-n3m:], 1)
        ax_m.plot(inv_line, np.polyval(c3m, inv_line),
                  color='tab:red', lw=1.1, ls='-.', alpha=0.85,
                  label=f'lin fit (last {n3m} D): intercept={m_lin3:.4f}')

    # Star with ASYMMETRIC clipped error bar at x=0
    # three values: m_lastD, m_lin2, m_lin3 sorted → median=center, min=lower, max=upper
    m_lastD = mneel_list[-1] if len(mneel_list) > 0 and not np.isnan(mneel_list[-1]) else 0.0
    three_vals = sorted([m_lastD, m_lin2, m_lin3])
    m_ctr = max(0.0, three_vals[1])   # median
    m_lo  = max(0.0, three_vals[0])   # min
    m_hi  = max(0.0, three_vals[2])   # max
    yerr_down = m_ctr - m_lo
    yerr_up   = m_hi - m_ctr
    ax_m.errorbar([0], [m_ctr],
                  yerr=[[yerr_down], [yerr_up]],
                  fmt='*', color=col_m, ms=14, capsize=5, elinewidth=1.3,
                  markeredgewidth=0.5, markeredgecolor='k', zorder=7,
                  label=f'extrap = {m_ctr:.4f} [{m_lo:.4f}, {m_hi:.4f}]')

    ax_m.set_ylabel(r'm$_\mathrm{N\acute{e}el}$', fontsize=11)
    if ylims is not None:
        ax_m.set_ylim(ylims['mag'])
    ax_m.legend(fontsize=8, loc='upper right')
    ax_m.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.5g'))

    # ── Panel 3: NN correlations (3 ranks, no fit) ────────────────────────────
    D_list = sorted(nn_grp.keys())
    for target_rank in [1, 2, 3]:
        D_r, m_r = [], []
        for D in D_list:
            entry = nn_grp[D]
            for g_i, r in enumerate(entry['ranks']):
                if r == target_rank:
                    D_r.append(D); m_r.append(entry['means'][g_i]); break
        if not D_r:
            continue
        ax_nn.plot([1.0/D for D in D_r], m_r, 'o-',
                   color=RANK_COLORS[target_rank], ms=5, lw=1.2,
                   label=RANK_LABELS[target_rank])

    ax_nn.set_ylabel(r'$\langle S_i \cdot S_j \rangle$ (NN)', fontsize=11)
    ax_nn.legend(fontsize=8, loc='upper right')
    ax_nn.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.5g'))
    if ylims is not None:
        ax_nn.set_ylim(ylims['nn'])

    jstr = _j2_fname(j2)
    _save(fig, os.path.join(out_dir, f'overview_J2_{jstr}.pdf'))

# ──────────────────────────────────────────────────────────────────────────────
# 7. Summary figure: statistics vs J2
# ──────────────────────────────────────────────────────────────────────────────
def plot_summary_vs_j2(all_data, out_dir, ylims=None):
    """
    Two panels sharing the J2 x-axis.

    Panel 1: E per site vs J2
      • D=5, D=7, D=8 raw curves (Blues cmap, increasing opacity, ▲/■/◆)
      • Extrapolation with error bars (darkest blue, ★)

    Panel 2: m_Néel (left y-axis, Purples) + ΔNN (right y-axis, Oranges)
      • m_Néel:  D=5, D=7, D=8 + extrapolation  (same markers as panel 1)
      • ΔNN:     D=5, D=7, D=8 only              (same markers, dashed)
      • Axis label colors match their respective colormaps.
    """
    j2_arr = np.array(sorted(all_data.keys()))

    # ── Collect arrays ────────────────────────────────────────────────────────
    # Energy
    e_by_D   = {D: ([], []) for D in D_SHOW}   # (j2_list, e_list)
    e_ext_j2 = []; e_ext_val = []; e_ext_err = []

    # Magnetization (asymmetric clipped error bars)
    m_by_D    = {D: ([], []) for D in D_SHOW}
    m_ext_j2  = []; m_ext_val = []
    m_ext_elo = []; m_ext_ehi = []   # asymmetric yerr [down, up]

    # NN Delta
    d_by_D   = {D: ([], []) for D in D_SHOW}   # Delta = max(means) - min(means)

    for j2 in sorted(all_data.keys()):
        v  = all_data[j2]
        Ds = v['Ds']

        for D in D_SHOW:
            if D in Ds:
                idx = Ds.index(D)
                e_by_D[D][0].append(j2); e_by_D[D][1].append(v['energy_per_site'][idx])
                m_by_D[D][0].append(j2); m_by_D[D][1].append(v['mneel_list'][idx])
                nn = v['nn_groups'][D]['means']
                d_by_D[D][0].append(j2); d_by_D[D][1].append(max(nn) - min(nn))

        e_ext_j2.append(j2)
        e_ext_val.append(v['extrap']['E_best'])
        e_ext_err.append(v['extrap']['E_err'])

        # asymmetric clipped m extrapolation: center=median, bounds=min/max of [m_lastD, m_lin2, m_lin3]
        m_lastD = v['mneel_list'][-1] if len(v['mneel_list']) > 0 and not np.isnan(v['mneel_list'][-1]) else 0.0
        three_vals = sorted([m_lastD, v['m_lin2'], v['m_lin3']])
        m_ctr = max(0.0, three_vals[1])   # median
        m_lo  = max(0.0, three_vals[0])   # min
        m_hi  = max(0.0, three_vals[2])   # max
        m_ext_j2.append(j2)
        m_ext_val.append(m_ctr)
        m_ext_elo.append(m_ctr - m_lo)
        m_ext_ehi.append(m_hi - m_ctr)

    # ── Color palettes ────────────────────────────────────────────────────────
    E_cmap      = plt.get_cmap('Blues')
    m_cmap      = plt.get_cmap('Purples')
    d_cmap      = plt.get_cmap('Oranges')

    n_curves    = len(D_SHOW)
    # base colors at evenly spaced positions; extrapolation gets darkest
    def _palette(cmap, n):
        return [cmap(0.35 + 0.45 * i / max(n - 1, 1)) for i in range(n)]

    E_cols   = _palette(E_cmap, n_curves); E_col_ext = E_cmap(0.95)
    m_cols   = _palette(m_cmap, n_curves); m_col_ext = m_cmap(0.95)
    d_cols   = _palette(d_cmap, n_curves)

    # Axis brand colors (used for y-axis labels / ticks)
    m_axis_color = m_cmap(0.80)
    d_axis_color = d_cmap(0.80)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, (ax_e, ax_m) = plt.subplots(2, 1, figsize=(9, 10), sharex=True,
                                      gridspec_kw={'hspace': 0.10})

    # ── Panel 1: Energy ───────────────────────────────────────────────────────
    for i, D in enumerate(D_SHOW):
        j2s, es = e_by_D[D]
        if j2s:
            ax_e.plot(j2s, es, D_MARKERS[D] + '-',
                      color=E_cols[i], alpha=D_ALPHAS[D],
                      ms=7, lw=1.3, label=f'D={D}')

    ax_e.errorbar(e_ext_j2, e_ext_val, yerr=e_ext_err,
                  fmt='*-', color=E_col_ext, alpha=1.0,
                  ms=12, lw=1.5, capsize=4, elinewidth=1.1,
                  markeredgewidth=0.5, markeredgecolor='k',
                  label='extrap (D→∞)')

    ax_e.set_ylabel('Energy per site', fontsize=11)
    ax_e.set_title('C6Yπ — statistics vs J2', fontsize=12, pad=6)
    ax_e.legend(fontsize=9, loc='best')
    ax_e.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.5g'))
    ax_e.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    if ylims is not None:
        ax_e.set_ylim(ylims['energy'])

    # ── Panel 2: m_Néel (left) + ΔNN (right) ─────────────────────────────────
    ax_d = ax_m.twinx()

    # --- m_Néel data (left axis) ---
    for i, D in enumerate(D_SHOW):
        j2s, ms = m_by_D[D]
        if j2s:
            ax_m.plot(j2s, ms, D_MARKERS[D] + '-',
                      color=m_cols[i], alpha=D_ALPHAS[D],
                      ms=7, lw=1.3, label=f'm D={D}')

    ax_m.errorbar(m_ext_j2, m_ext_val,
                  yerr=[m_ext_elo, m_ext_ehi],
                  fmt='*-', color=m_col_ext, alpha=1.0,
                  ms=12, lw=1.5, capsize=4, elinewidth=1.1,
                  markeredgewidth=0.5, markeredgecolor='k',
                  label='m extrap')

    ax_m.set_ylabel(r'm$_\mathrm{N\acute{e}el}$', fontsize=11, color=m_axis_color)
    ax_m.tick_params(axis='y', labelcolor=m_axis_color)
    if ylims is not None:
        ax_m.set_ylim(ylims['mag'])
    else:
        ax_m.set_ylim(bottom=0.0)

    # --- ΔNN data (right axis) ---
    for i, D in enumerate(D_SHOW):
        j2s, ds = d_by_D[D]
        if j2s:
            ax_d.plot(j2s, ds, D_MARKERS[D] + '--',
                      color=d_cols[i], alpha=D_ALPHAS[D],
                      ms=7, lw=1.1, label=f'ΔNN D={D}')

    ax_d.set_ylabel(r'$\Delta_\mathrm{NN}$ = max$-$min corr', fontsize=11,
                    color=d_axis_color)
    ax_d.tick_params(axis='y', labelcolor=d_axis_color)
    ax_d.spines['right'].set_color(d_axis_color)
    ax_m.spines['left'].set_color(m_axis_color)
    if ylims is not None:
        ax_d.set_ylim(ylims['nn_delta'])

    # Combined legend
    lines_m, labs_m = ax_m.get_legend_handles_labels()
    lines_d, labs_d = ax_d.get_legend_handles_labels()
    ax_m.legend(lines_m + lines_d, labs_m + labs_d,
                fontsize=8, loc='upper left', ncol=2)

    ax_m.set_xlabel('J2', fontsize=11)
    ax_m.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax_m.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4g'))
    ax_d.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4g'))

    _save(fig, os.path.join(out_dir, 'summary_vs_J2.pdf'))

# ──────────────────────────────────────────────────────────────────────────────
# 8. JSON dump
# ──────────────────────────────────────────────────────────────────────────────
def save_json(all_data, out_dir):
    fpath = os.path.join(out_dir, 'analysis_results_0427.json')
    with open(fpath, 'w') as f:
        json.dump(_jsonify(all_data), f, indent=2)
    print(f"Saved JSON: {fpath}")

# ──────────────────────────────────────────────────────────────────────────────
# 9. Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print("Loading data ...")
    all_data = load_all_data()
    if not all_data:
        print("No data found — check DATA_DIR and FOLDER_MAP.")
        return

    print("Saving JSON ...")
    save_json(all_data, OUT_DIR)

    # Compute shared y-limits for consistent axes across all figures
    _ov_ylims = compute_global_ylims(all_data)
    # nn_delta (max-min of bond means) is different from nn (individual means)
    all_deltas = []
    for v in all_data.values():
        for D_entry in v['nn_groups'].values():
            m = D_entry['means']
            all_deltas.append(max(m) - min(m))
    _delta_pad = max(all_deltas) * 0.08
    _sum_ylims = dict(_ov_ylims)
    _sum_ylims['nn_delta'] = (0.0, max(all_deltas) + _delta_pad)

    print(f"Generating {len(all_data)} per-J2 overview figures ...")
    for j2, v in sorted(all_data.items()):
        plot_overview_j2(v, j2, OUT_DIR, ylims=_ov_ylims)

    print("Generating summary figure ...")
    plot_summary_vs_j2(all_data, OUT_DIR, ylims=_sum_ylims)

    print(f"\nDone.  All figures in: {OUT_DIR}")


if __name__ == '__main__':
    main()
