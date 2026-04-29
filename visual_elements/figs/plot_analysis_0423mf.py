#!/usr/bin/env python3
"""
Analysis & plotting for 0420core jobs from node-jobid.log.

13 jobs tracked:
  "mf"   : MF Neel init + two-tensor C3 ansatz  (J2 = 0.20 .. 0.25)
  "two"  : vanilla random init + two-tensor C3   (J2 = 0.20 .. 0.25)
  "mf1t" : MF Neel init + 1-tensor C6Ypi         (J2 = 0.20 only)

Plots (only vs 1/D, no vs J2):
  For each J2 in [0.20, 0.21, 0.22, 0.23, 0.24, 0.25]:
    energy_vsInvD_J2_XXX.pdf     — 2×2: (0,0) two | (0,1) mf | (1,0) both | empty
    NN_vsInvD_J2_XXX.pdf         — 3×2 grid with rank-compare panels
    NNN_vsInvD_J2_XXX.pdf        — 3×2 grid with rank-compare panels
    order_vsInvD_J2_XXX.pdf      — 2×2: (0,0) two | (0,1) mf | (1,0) both | empty

  For J2 = 0.20 only (extra comparison mf vs mf1t):
    energy_vsInvD_J2_0p20_mf_vs_mf1t.pdf
    NN_vsInvD_J2_0p20_mf_vs_mf1t.pdf
    NNN_vsInvD_J2_0p20_mf_vs_mf1t.pdf
    order_vsInvD_J2_0p20_mf_vs_mf1t.pdf
"""
import os
import re
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from scipy.optimize import curve_fit

# ──────────────────────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────────────────────
DATA_DIR   = '/home/chye/6ADctmrg/data/0420core'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(SCRIPT_DIR, 'analysis_plots_0423mf')
os.makedirs(OUT_DIR, exist_ok=True)


FOLDER_MAP = {
    'C6': {
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
    },
    'C3v': {
        0.21: '1tensor_C3Vypi__J2_0p21_20260428_201139',
        0.23: '1tensor_C3Vypi__J2_0p23_20260427_191513',
        0.25: '1tensor_C3Vypi__J2_0p25_20260427_205614',
        0.26: '1tensor_C3Vypi__J2_0p26_20260425_032814',
        0.27: '1tensor_C3Vypi__J2_0p27_20260425_032814',
        0.28: '1tensor_C3Vypi__J2_0p28_20260425_031820',
    },
}

J2_LIST = [0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30]

ANSATZ_STYLE = {
    'C6':  dict(color='tab:blue',   ls='-',  label='C6',   marker='o'),
    'C3v':   dict(color='tab:red',    ls='--', label='C3v',   marker='s'),
}

RANK_COLORS = {1: 'tab:red', 2: 'tab:green', 3: 'tab:blue'}
RANK_LABELS = {1: 'rank 1 (most neg)', 2: 'rank 2 (mid)', 3: 'rank 3 (least neg)'}
RANK_CMAPS  = {1: 'Reds', 2: 'Greens', 3: 'Blues'}

def _D_alpha(D):
    _map = {4: 0.15, 5: 0.25, 6: 0.40, 7: 0.55, 8: 0.75, 9: 0.95, 10: 1.0, 11: 1.0}
    return _map.get(D, min(1.0, max(0.15, D / 10.0)))

def _j2_fname(j2):
    s = f'{j2:.10f}'.rstrip('0')
    dot = s.index('.')
    if len(s) - dot - 1 < 2:
        s += '0' * (2 - (len(s) - dot - 1))
    return s.replace('.', 'p')

def _j2_label(j2):
    return f'J2={j2:.2f}'

# ──────────────────────────────────────────────────────────────────────────────
# BOND GROUP DEFINITIONS (same as 0418)
# ──────────────────────────────────────────────────────────────────────────────
NN_GROUPS_RAW = [
    [(1,'EB'),(1,'AD'),(1,'CF'),(3,'BE'),(3,'FC'),(3,'DA')],
    [(2,'CB'),(2,'AF'),(2,'ED'),(1,'FA'),(1,'DE'),(1,'BC')],
    [(3,'EF'),(3,'AB'),(3,'CD'),(2,'DC'),(2,'BA'),(2,'FE')],
]
NNN_GROUPS_RAW = [
    [(1,'AE'),(1,'EC'),(1,'CA'),(1,'DB'),(1,'BF'),(1,'FD')],
    [(2,'CA'),(2,'AE'),(2,'EC'),(2,'BF'),(2,'FD'),(2,'DB')],
    [(3,'EC'),(3,'CA'),(3,'AE'),(3,'FD'),(3,'DB'),(3,'BF')],
]

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
# 1. parse_plain_file
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
            'Sx': float(m.group(3)), 'Sy': float(m.group(4)), 'Sz': float(m.group(5))
        }
    return {'energy_per_site': energy, 'corr': corr, 'mag': mag}

# ──────────────────────────────────────────────────────────────────────────────
# 2. load_folder_data
# ──────────────────────────────────────────────────────────────────────────────
def load_folder_data(folder_path):
    """For each D: highest plain chi, use +2D file if available."""
    plain_map = {}   # D → list of (chi, path)
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
        best_chi = max(chi for chi, _ in plain_map[D])
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
# 3. compute_energy_extrap
# ──────────────────────────────────────────────────────────────────────────────
def _exp_model(D, E0, c, Dchar):
    return E0 + c * np.exp(-np.asarray(D, dtype=float) / Dchar)

def compute_energy_extrap(Ds, eps):
    Ds_f  = np.array(Ds, dtype=float)
    eps_f = np.array(eps, dtype=float)
    E_horiz = float(eps_f[-1])
    if len(Ds_f) >= 2:
        coeffs = np.polyfit(1.0 / Ds_f, eps_f, 1)
        E_lin2 = float(coeffs[1])
    else:
        E_lin2 = E_horiz
    E_exp_all = None
    exp_popt  = None
    if len(Ds_f) >= 3:
        try:
            p0 = [eps_f[-1] - 0.1, 0.1, float(Ds_f.mean())]
            popt, _ = curve_fit(_exp_model, Ds_f, eps_f, p0=p0, maxfev=5000)
            E_exp_all = float(popt[0])
            exp_popt  = popt.tolist()
        except Exception:
            pass
    lo = min(E_horiz, E_lin2) - 0.5 * abs(E_horiz - E_lin2) - 0.1
    hi = max(E_horiz, E_lin2) + 0.5 * abs(E_horiz - E_lin2) + 0.1
    if E_exp_all is not None and lo <= E_exp_all <= hi:
        E_0 = E_exp_all
    elif E_exp_all is not None:
        E_0 = E_lin2
    else:
        E_0 = E_lin2
    return {
        'E_horiz':   E_horiz,
        'E_lin2':    E_lin2,
        'E_exp_all': E_exp_all,
        'E_0':       E_0,
        'exp_popt':  exp_popt,
    }

# ──────────────────────────────────────────────────────────────────────────────
# 4. compute_bond_groups  — ranking per-D
# ──────────────────────────────────────────────────────────────────────────────
def compute_bond_groups(D_data, groups_raw):
    result = {}
    for D in sorted(D_data.keys()):
        corr = D_data[D]['corr']
        means, stds, sterrs = [], [], []
        for group in groups_raw:
            vals = []
            for (env, pair) in group:
                key = (env, pair)
                if key in corr:
                    vals.append(corr[key])
            if vals:
                means.append(float(np.mean(vals)))
                stds.append(float(np.std(vals, ddof=0)))
                sterrs.append(float(np.std(vals, ddof=0) / np.sqrt(len(vals))))
            else:
                means.append(float('nan'))
                stds.append(float('nan'))
                sterrs.append(float('nan'))
        order = np.argsort(means)
        ranks = [0] * len(means)
        for rank_idx, group_idx in enumerate(order):
            ranks[group_idx] = rank_idx + 1
        result[D] = {'means': means, 'stds': stds, 'sterrs': sterrs, 'ranks': ranks}
    return result

# ──────────────────────────────────────────────────────────────────────────────
# 5. compute_order_param
# ──────────────────────────────────────────────────────────────────────────────
def compute_order_param(D_data):
    result = {}
    for D in sorted(D_data.keys()):
        mag = D_data[D].get('mag', {})
        sz_ace, sz_bdf = [], []
        for env in [1, 2, 3]:
            for site in ['A', 'C', 'E']:
                k = (env, site)
                if k in mag:
                    sz_ace.append(mag[k]['Sz'])
            for site in ['B', 'D', 'F']:
                k = (env, site)
                if k in mag:
                    sz_bdf.append(mag[k]['Sz'])
        m_ace = float(np.mean(sz_ace)) if sz_ace else float('nan')
        m_bdf = float(np.mean(sz_bdf)) if sz_bdf else float('nan')
        m_neel = 0.5 * abs(m_ace - m_bdf) if not (np.isnan(m_ace) or np.isnan(m_bdf)) else float('nan')
        result[D] = {'m_ace': m_ace, 'm_bdf': m_bdf, 'm_neel': m_neel}
    return result

# ──────────────────────────────────────────────────────────────────────────────
# 6. load_all_data: load all 13 folders
# ──────────────────────────────────────────────────────────────────────────────
def load_all_data():
    """
    Returns nested dict: all_data[ansatz][j2] = {
        'Ds': [...], 'energy_per_site': [...], 'extrap': {...},
        'nn_groups': {...}, 'nnn_groups': {...}, 'order': {...}
    }
    """
    all_data = {}
    for ansatz, j2_map in FOLDER_MAP.items():
        all_data[ansatz] = {}
        for j2, folder_name in j2_map.items():
            folder_path = os.path.join(DATA_DIR, folder_name)
            if not os.path.isdir(folder_path):
                print(f"  WARNING: folder not found: {folder_path}")
                continue
            print(f"  Loading {ansatz} J2={j2:.2f} from {folder_name} ...")
            D_data = load_folder_data(folder_path)
            if not D_data:
                print(f"    (no data files found)")
                continue
            Ds  = sorted(D_data.keys())
            eps = [D_data[D]['energy_per_site'] for D in Ds]
            extrap = compute_energy_extrap(Ds, eps)
            nn_groups  = compute_bond_groups(D_data, NN_GROUPS_RAW)
            nnn_groups = compute_bond_groups(D_data, NNN_GROUPS_RAW)
            order      = compute_order_param(D_data)
            all_data[ansatz][j2] = {
                'Ds': Ds,
                'energy_per_site': eps,
                'extrap': extrap,
                'nn_groups': nn_groups,
                'nnn_groups': nnn_groups,
                'order': order,
            }
    return all_data

# ──────────────────────────────────────────────────────────────────────────────
# JSON utility
# ──────────────────────────────────────────────────────────────────────────────
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

def save_json(all_data, out_dir):
    fpath = os.path.join(out_dir, 'analysis_results_0420mf.json')
    with open(fpath, 'w') as f:
        json.dump(_jsonify(all_data), f, indent=2)
    print(f"Saved JSON: {fpath}")

# ──────────────────────────────────────────────────────────────────────────────
# PLOTTING UTILITIES
# ──────────────────────────────────────────────────────────────────────────────
def _save(fig, fpath):
    fig.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {os.path.relpath(fpath)}")

def _share_limits(*axes):
    visible = [ax for ax in axes if ax.get_visible()]
    if len(visible) < 2:
        return
    xl = [ax.get_xlim() for ax in visible]
    yl = [ax.get_ylim() for ax in visible]
    shared_x = (min(x[0] for x in xl), max(x[1] for x in xl))
    shared_y = (min(y[0] for y in yl), max(y[1] for y in yl))
    for ax in visible:
        ax.set_xlim(shared_x)
        ax.set_ylim(shared_y)

def _D_colormap(D_list, ansatz):
    Ds   = sorted(set(D_list))
    cname = {'two': 'Blues', 'mf': 'Reds', 'mf1t': 'Greens'}.get(ansatz, 'Greys')
    cmap  = plt.get_cmap(cname)
    if len(Ds) == 1:
        return {Ds[0]: cmap(0.60)}
    return {D: cmap(0.30 + 0.60 * i / (len(Ds) - 1)) for i, D in enumerate(Ds)}

def _D_colormap_rank(D_list, rank):
    Ds   = sorted(set(D_list))
    cmap = plt.get_cmap(RANK_CMAPS[rank])
    if len(Ds) == 1:
        return {Ds[0]: cmap(0.60)}
    return {D: cmap(0.30 + 0.60 * i / (len(Ds) - 1)) for i, D in enumerate(Ds)}

# ──────────────────────────────────────────────────────────────────────────────
# ENERGY VS 1/D helpers
# ──────────────────────────────────────────────────────────────────────────────
def _plot_energy_ax(ax, v, col, lbl, marker='o'):
    global GLOBAL_MIN_E, GLOBAL_MAX_E
    Ds  = np.array(v['Ds'], dtype=float)
    eps = np.array(v['energy_per_site'])
    ex  = v['extrap']
    inv = 1.0 / Ds
    ax.plot(inv, eps, marker + '-', color=col, ms=5, lw=1.4, label=lbl)
    x_max = max(inv) * 1.15
    ax.plot([0, x_max], [ex['E_horiz'], ex['E_horiz']],
            color=col, lw=0.8, ls=':', alpha=0.7, label='_nolegend_')
    if len(Ds) >= 2:
        inv_ext = np.linspace(0, max(inv) * 1.1, 100)
        coeffs = np.polyfit(inv, eps, 1)
        ax.plot(inv_ext, np.polyval(coeffs, inv_ext),
                color=col, lw=0.8, ls='--', alpha=0.7, label='_nolegend_')
    ax.plot(0, ex['E_0'], marker='*', color=col, ms=11, zorder=6,
            markeredgewidth=0.4, markeredgecolor='k', label='_nolegend_')
    ax.set_xlim(left=0, right=1/2.5)
    # y lim set to global min and max of E across all ansätze
    ax.set_ylim(bottom=GLOBAL_MIN_E - 0.001, top=GLOBAL_MAX_E + 0.001)
    ax.set_xlabel('1/D', fontsize=11)
    ax.set_ylabel('Energy per site', fontsize=11)

# ──────────────────────────────────────────────────────────────────────────────
# BOND CORRELATION helpers
# ──────────────────────────────────────────────────────────────────────────────
def _plot_bonds_invD_ax(ax, v, bond_key, ansatz):
    """Plot all 3 rank-curves on one ax for one ansatz."""
    global GLOBAL_MIN_BOND_NN, GLOBAL_MAX_BOND_NN, GLOBAL_MIN_BOND_NNN, GLOBAL_MAX_BOND_NNN
    grp_data = v[bond_key]
    D_list = sorted(grp_data.keys())
    for target_rank in [1, 2, 3]:
        col = RANK_COLORS[target_rank]
        D_for_rank, means_for_rank = [], []
        for D in D_list:
            entry = grp_data[D]
            for g_i, r in enumerate(entry['ranks']):
                if r == target_rank:
                    D_for_rank.append(D)
                    means_for_rank.append(entry['means'][g_i])
                    break
        if not D_for_rank:
            continue
        inv = [1.0 / D for D in D_for_rank]
        ax.plot(inv, means_for_rank, 'o-', color=col, ms=4, lw=1.2,
                label=RANK_LABELS[target_rank])
    ax.set_xlim(left=0,right=1/2.5)
    ax.set_ylim(bottom=GLOBAL_MIN_BOND_NN-0.001 if bond_key == 'nn_groups' else GLOBAL_MIN_BOND_NNN-0.001, top=GLOBAL_MAX_BOND_NN+0.001 if bond_key == 'nn_groups' else GLOBAL_MAX_BOND_NNN+0.001)
    ax.set_xlabel('1/D', fontsize=11)
    ax.set_ylabel('⟨Sᵢ·Sⱼ⟩', fontsize=11)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.4g'))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.5g'))

def _plot_bonds_invD_rank_compare(ax, v_a, v_b, bond_key, rank, ansatz_a, ansatz_b):
    """One rank, both ansätze on same ax."""
    global GLOBAL_MIN_BOND_NN, GLOBAL_MAX_BOND_NN, GLOBAL_MIN_BOND_NNN, GLOBAL_MAX_BOND_NNN
    for ansatz_key, v, mkr, ls in [(ansatz_a, v_a, 'o', '-'), (ansatz_b, v_b, 's', '--')]:
        if v is None:
            continue
        grp_data = v[bond_key]
        D_list   = sorted(grp_data.keys())
        col      = ANSATZ_STYLE[ansatz_key]['color']
        lbl      = ANSATZ_STYLE[ansatz_key]['label']
        D_for_rank, means_for_rank = [], []
        for D in D_list:
            entry = grp_data[D]
            for g_i, r in enumerate(entry['ranks']):
                if r == rank:
                    D_for_rank.append(D)
                    means_for_rank.append(entry['means'][g_i])
                    break
        if not D_for_rank:
            continue
        inv = [1.0 / D for D in D_for_rank]
        ax.plot(inv, means_for_rank, mkr + ls, color=col, ms=4, lw=1.2, label=lbl)
    ax.set_xlim(left=0, right=1/2.5)
    ax.set_ylim(bottom=GLOBAL_MIN_BOND_NN-0.001 if bond_key == 'nn_groups' else GLOBAL_MIN_BOND_NNN-0.001, top=GLOBAL_MAX_BOND_NN+0.001 if bond_key == 'nn_groups' else GLOBAL_MAX_BOND_NNN+0.001)
    ax.set_xlabel('1/D', fontsize=11)
    ax.set_ylabel('⟨Sᵢ·Sⱼ⟩', fontsize=11)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.4g'))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.5g'))

# ──────────────────────────────────────────────────────────────────────────────
# ORDER PARAM helpers
# ──────────────────────────────────────────────────────────────────────────────
def _plot_order_ax(ax, v, col, lbl, marker='o'):
    global GLOBAL_MAX_ORDER
    Ds = np.array(v['Ds'], dtype=float)
    inv = 1.0 / Ds
    mneel = [v['order'][D]['m_neel'] for D in v['Ds']]
    ax.plot(inv, mneel, marker + '-', color=col, ms=5, lw=1.4, label=lbl)
    ax.set_xlim(left=0, right=1/2.5)
    ax.set_ylim(bottom=-0.001, top=GLOBAL_MAX_ORDER+0.001)
    ax.set_xlabel('1/D', fontsize=11)
    ax.set_ylabel('m_Néel', fontsize=11)

# ──────────────────────────────────────────────────────────────────────────────
# MAIN FIGURE BUILDERS
# ──────────────────────────────────────────────────────────────────────────────

def _fig_2x2_both(all_data, j2, ansatz_a, ansatz_b, plot_fn_single, plot_fn_both,
                  title_suffix, fname):
    """
    Generic 2×2 figure:
      (0,0) ansatz_a  |  (0,1) ansatz_b
      (1,0) both      |  (1,1) empty
    plot_fn_single(ax, v, col, lbl, marker)
    plot_fn_both(ax, v_a, v_b, ansatz_a, ansatz_b)
    """
    v_a = all_data.get(ansatz_a, {}).get(j2)
    v_b = all_data.get(ansatz_b, {}).get(j2)
    if v_a is None and v_b is None:
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax_a, ax_b, ax_both, ax_empty = axes[0,0], axes[0,1], axes[1,0], axes[1,1]
    ax_empty.set_visible(False)
    st_a = ANSATZ_STYLE[ansatz_a]
    st_b = ANSATZ_STYLE[ansatz_b]
    if v_a is not None:
        plot_fn_single(ax_a, v_a, st_a['color'], st_a['label'], st_a['marker'])
        ax_a.set_title(st_a['label'], fontsize=11)
        ax_a.legend(fontsize=8)
    else:
        ax_a.text(0.5, 0.5, 'no data', ha='center', va='center', transform=ax_a.transAxes)
    if v_b is not None:
        plot_fn_single(ax_b, v_b, st_b['color'], st_b['label'], st_b['marker'])
        ax_b.set_title(st_b['label'], fontsize=11)
        ax_b.legend(fontsize=8)
    else:
        ax_b.text(0.5, 0.5, 'no data', ha='center', va='center', transform=ax_b.transAxes)
    if v_a is not None or v_b is not None:
        plot_fn_both(ax_both, v_a, v_b, ansatz_a, ansatz_b)
        ax_both.set_title('Both overlaid', fontsize=11)
        ax_both.legend(fontsize=8)
    #_share_limits(ax_a, ax_b, ax_both)
    fig.suptitle(f'{title_suffix}  |  {_j2_label(j2)}', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, fname)


def _fig_3x2_bonds(all_data, j2, ansatz_a, ansatz_b, bond_key, bond_label, fname):
    """
    3×2 bonds figure:
      (0,0) ansatz_a all ranks  |  (0,1) ansatz_b all ranks
      (1,0) both overlaid       |  (1,1) rank1 compare
      (2,0) rank2 compare       |  (2,1) rank3 compare
    """
    v_a = all_data.get(ansatz_a, {}).get(j2)
    v_b = all_data.get(ansatz_b, {}).get(j2)
    if v_a is None and v_b is None:
        return
    st_a = ANSATZ_STYLE[ansatz_a]
    st_b = ANSATZ_STYLE[ansatz_b]
    fig = plt.figure(figsize=(14, 14))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)
    ax_a   = fig.add_subplot(gs[0, 0])
    ax_b   = fig.add_subplot(gs[0, 1])
    ax_all = fig.add_subplot(gs[1, 0])
    ax_r1  = fig.add_subplot(gs[1, 1])
    ax_r2  = fig.add_subplot(gs[2, 0])
    ax_r3  = fig.add_subplot(gs[2, 1])

    if v_a is not None:
        _plot_bonds_invD_ax(ax_a, v_a, bond_key, ansatz_a)
        ax_a.set_title(f'{st_a["label"]} — all ranks', fontsize=10)
        ax_a.legend(fontsize=8)
    else:
        ax_a.text(0.5, 0.5, 'no data', ha='center', va='center', transform=ax_a.transAxes)

    if v_b is not None:
        _plot_bonds_invD_ax(ax_b, v_b, bond_key, ansatz_b)
        ax_b.set_title(f'{st_b["label"]} — all ranks', fontsize=10)
        ax_b.legend(fontsize=8)
    else:
        ax_b.text(0.5, 0.5, 'no data', ha='center', va='center', transform=ax_b.transAxes)

    # Both overlaid (all 3 ranks × 2 ansätze)
    for v, ansatz_key, mkr, ls in [(v_a, ansatz_a, 'o', '-'), (v_b, ansatz_b, 's', '--')]:
        if v is None:
            continue
        grp_data = v[bond_key]
        D_list   = sorted(grp_data.keys())
        col      = ANSATZ_STYLE[ansatz_key]['color']
        lbl_base = ANSATZ_STYLE[ansatz_key]['label']
        for target_rank in [1, 2, 3]:
            D_for_rank, means_for_rank = [], []
            for D in D_list:
                entry = grp_data[D]
                for g_i, r in enumerate(entry['ranks']):
                    if r == target_rank:
                        D_for_rank.append(D)
                        means_for_rank.append(entry['means'][g_i])
                        break
            if not D_for_rank:
                continue
            inv = [1.0 / D for D in D_for_rank]
            ax_all.plot(inv, means_for_rank, mkr + ls,
                        color=col, alpha=min(1.0, 0.6 + 0.2 * target_rank), ms=4, lw=1.2,
                        label=f'{lbl_base} r{target_rank}')
    #ax_all.set_xlim(left=0)
    ax_all.set_xlabel('1/D', fontsize=10)
    ax_all.set_ylabel('⟨Sᵢ·Sⱼ⟩', fontsize=10)
    ax_all.set_title('Both ansätze — all ranks', fontsize=10)
    ax_all.legend(fontsize=7, ncol=2)
    ax_all.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.4g'))
    ax_all.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.5g'))

    for ax, rank in [(ax_r1, 1), (ax_r2, 2), (ax_r3, 3)]:
        _plot_bonds_invD_rank_compare(ax, v_a, v_b, bond_key, rank, ansatz_a, ansatz_b)
        ax.set_title(RANK_LABELS[rank], fontsize=10)
        ax.legend(fontsize=8)

    #_share_limits(ax_a, ax_b, ax_all, ax_r1, ax_r2, ax_r3)
    fig.suptitle(f'{bond_label}  vs  1/D  |  {_j2_label(j2)}', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, fname)


def _energy_both_ax(ax, v_a, v_b, ansatz_a, ansatz_b):
    """Both ansätze overlaid energy on one ax."""
    for v, key in [(v_a, ansatz_a), (v_b, ansatz_b)]:
        if v is None:
            continue
        st = ANSATZ_STYLE[key]
        _plot_energy_ax(ax, v, st['color'], st['label'], st['marker'])

def _order_both_ax(ax, v_a, v_b, ansatz_a, ansatz_b):
    for v, key in [(v_a, ansatz_a), (v_b, ansatz_b)]:
        if v is None:
            continue
        st = ANSATZ_STYLE[key]
        _plot_order_ax(ax, v, st['color'], st['label'], st['marker'])


# ──────────────────────────────────────────────────────────────────────────────
# MAIN PLOTTING CALLS
# ──────────────────────────────────────────────────────────────────────────────
def plot_all(all_data, out_dir):
    # --- For each J2: compare 'two' vs 'mf' ---
    for j2 in J2_LIST:
        jstr = _j2_fname(j2)

        # Energy vs 1/D
        _fig_2x2_both(
            all_data, j2, 'C6', 'C3v',
            plot_fn_single=_plot_energy_ax,
            plot_fn_both=_energy_both_ax,
            title_suffix='Energy per site vs 1/D',
            fname=os.path.join(out_dir, f'energy_vsInvD_J2_{jstr}.pdf'),
        )

        # NN bonds vs 1/D
        _fig_3x2_bonds(
            all_data, j2, 'C6', 'C3v', 'nn_groups',
            'NN bond correlations',
            fname=os.path.join(out_dir, f'NN_vsInvD_J2_{jstr}.pdf'),
        )

        # NNN bonds vs 1/D
        _fig_3x2_bonds(
            all_data, j2, 'C6', 'C3v', 'nnn_groups',
            'NNN bond correlations',
            fname=os.path.join(out_dir, f'NNN_vsInvD_J2_{jstr}.pdf'),
        )

        # Order param vs 1/D
        _fig_2x2_both(
            all_data, j2, 'C6', 'C3v',
            plot_fn_single=_plot_order_ax,
            plot_fn_both=_order_both_ax,
            title_suffix='Néel order parameter vs 1/D',
            fname=os.path.join(out_dir, f'order_vsInvD_J2_{jstr}.pdf'),
        )

    if False:
        # --- For J2=0.20: extra comparison mf vs mf1t ---
        j2 = 0.20
        jstr = _j2_fname(j2)
        fstr = f'J2_{jstr}_mf_vs_mf1t'

        _fig_2x2_both(
            all_data, j2, 'mf', 'mf1t',
            plot_fn_single=_plot_energy_ax,
            plot_fn_both=_energy_both_ax,
            title_suffix='Energy per site vs 1/D  [mf vs mf1t]',
            fname=os.path.join(out_dir, f'energy_vsInvD_{fstr}.pdf'),
        )
        _fig_3x2_bonds(
            all_data, j2, 'mf', 'mf1t', 'nn_groups',
            'NN bond correlations  [mf vs mf1t]',
            fname=os.path.join(out_dir, f'NN_vsInvD_{fstr}.pdf'),
        )
        _fig_3x2_bonds(
            all_data, j2, 'mf', 'mf1t', 'nnn_groups',
            'NNN bond correlations  [mf vs mf1t]',
            fname=os.path.join(out_dir, f'NNN_vsInvD_{fstr}.pdf'),
        )
        _fig_2x2_both(
            all_data, j2, 'mf', 'mf1t',
            plot_fn_single=_plot_order_ax,
            plot_fn_both=_order_both_ax,
            title_suffix='Néel order parameter vs 1/D  [mf vs mf1t]',
            fname=os.path.join(out_dir, f'order_vsInvD_{fstr}.pdf'),
        )


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    global GLOBAL_MIN_BOND_NN, GLOBAL_MAX_BOND_NN, GLOBAL_MIN_BOND_NNN, GLOBAL_MAX_BOND_NNN, GLOBAL_MIN_E, GLOBAL_MAX_E, GLOBAL_MAX_ORDER
    print("Loading data ...")
    all_data = load_all_data()
    print("Saving JSON ...")
    save_json(all_data, OUT_DIR)

    GLOBAL_MIN_BOND_NN  = min(
        v['nn_groups'][D]['means'][g_i]
        for ansatz in all_data.values() for v in ansatz.values() for D in v['nn_groups'] for g_i in range(len(v['nn_groups'][D]['means']))
    )
    GLOBAL_MAX_BOND_NN  = max(
        v['nn_groups'][D]['means'][g_i]
        for ansatz in all_data.values() for v in ansatz.values() for D in v['nn_groups'] for g_i in range(len(v['nn_groups'][D]['means']))
    )
    GLOBAL_MIN_BOND_NNN = min(
        v['nnn_groups'][D]['means'][g_i]
        for ansatz in all_data.values() for v in ansatz.values() for D in v['nnn_groups'] for g_i in range(len(v['nnn_groups'][D]['means']))
    )
    GLOBAL_MAX_BOND_NNN = max(
        v['nnn_groups'][D]['means'][g_i]
        for ansatz in all_data.values() for v in ansatz.values() for D in v['nnn_groups'] for g_i in range(len(v['nnn_groups'][D]['means']))
    )
    GLOBAL_MIN_E = min(
        v['energy_per_site'][i]
        for ansatz in all_data.values() for v in ansatz.values() for i in range(len(v['energy_per_site']))
    )
    #GLOBAL_MIN_E = -0.452
    GLOBAL_MAX_E = max(
        v['energy_per_site'][i]
        for ansatz in all_data.values() for v in ansatz.values() for i in range(len(v['energy_per_site']))
    )
    #GLOBAL_MAX_E = -0.432
    GLOBAL_MAX_ORDER = max(
        v['order'][D]['m_neel']
        for ansatz in all_data.values() for v in ansatz.values() for D in v['order']
    )

    print("Plotting ...")
    plot_all(all_data, OUT_DIR)

    # Plot the all Ds energy per site raw data across all ansatze and with respect to their J2 value (vs J2!!!)
    # different color per ansatz with lines, and different D has different transparency (lighter for smaller D, darker for larger D)
    fig, ax = plt.subplots(figsize=(8, 6))
    for ansatz_key, ansatz_data in all_data.items():
        st = ANSATZ_STYLE[ansatz_key]
        for j2, v in ansatz_data.items():
            Ds = np.array(v['Ds'], dtype=float)
            eps = np.array(v['energy_per_site'])
            col = st['color']
            lbl = f"{st['label']} J2={j2:.2f}"
            for D, e in zip(Ds, eps):
                alpha = 0.3 + 0.7 * (D - Ds.min()) / (Ds.max() - Ds.min()) if len(Ds) > 1 else 1.0
                ax.plot(j2, e, 'o-', color=col, alpha=alpha, ms=5, label=lbl if D == Ds[0] else None)
    ax.set_xlabel('J2', fontsize=11)
    ax.set_ylabel('Energy per site', fontsize=11)
    ax.legend(fontsize=8)
    _save(fig, os.path.join(OUT_DIR, 'energy_vs_J2_all_Ds.pdf'))



    print("Done.")

if __name__ == '__main__':
    main()
