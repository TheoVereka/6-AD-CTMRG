#!/usr/bin/env python3
"""
Comprehensive 6AD-CTMRG analysis script for 0418core dataset.

Ansätze:
  - '6tensors'       → label "unres"
  - '1tensor_C6Ypi'  → label "C6+π"

Data selection (no job-log required):
  For each D, find the highest plain chi.  If a +2D file exists for that chi,
  use the +2D file; otherwise use the plain file.

vs 1/D figures (one per J2, in subfolders):
  (0,0) unres  |  (0,1) C6+π  |  (1,0) both overlaid
  Energy: horizontal + linear fit + E₀ star (exp curve NOT plotted).
  All others: data only, no fits.

vs J2 figures (single figure each):
  (0,0) unres 3 D-curves  |  (0,1) C6+π 3 D-curves
  For NN/NNN bonds:
    (1,0) both ansätze rank 1  |  (1,1) rank 2  |  (2,0) rank 3
  For energy/order-param:
    (1,0) both ansätze together

Ranking for NN/NNN: per-(D, J2) independently — no global sorting.
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
DATA_DIR   = '/home/chye/6ADctmrg/data/0418core'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(SCRIPT_DIR, 'analysis_plots_0418')
os.makedirs(OUT_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def _j2_str(j2):
    s = f'{j2:.10f}'.rstrip('0')
    dot = s.index('.')
    if len(s) - dot - 1 < 2:
        s += '0' * (2 - (len(s) - dot - 1))
    return s

def _j2_fname(j2):
    return _j2_str(j2).replace('.', 'p')

def _j2_label(j2):
    return f'J2={_j2_str(j2)}'


# ──────────────────────────────────────────────────────────────────────────────
# BOND GROUP DEFINITIONS
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
SITES_ACE = ['A', 'C', 'E']
SITES_BDF = ['B', 'D', 'F']


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
RE_J2        = re.compile(r'J2_0p(\d+)')


# ──────────────────────────────────────────────────────────────────────────────
# STYLE
# ──────────────────────────────────────────────────────────────────────────────
ANSATZ_STYLE = {
    '6tensors': dict(color='tab:blue',   ls='-',  label='unres',  marker='o'),
    'c6ypi':    dict(color='tab:orange',  ls='--', label='C6+π',   marker='s'),
}
RANK_COLORS  = {1: 'tab:red', 2: 'tab:green', 3: 'tab:blue'}
RANK_LABELS  = {1: 'rank 1 (most neg)', 2: 'rank 2 (mid)', 3: 'rank 3 (least neg)'}
RANK_CMAPS   = {1: 'Reds', 2: 'Greens', 3: 'Blues'}


def _D_alpha(D):
    """More contrasty alpha: D=5 very transparent, D=7 somewhat, D=9 opaque."""
    _map = {4: 0.15, 5: 0.25, 6: 0.40, 7: 0.55, 8: 0.75, 9: 0.95}
    return _map.get(D, min(1.0, max(0.15, D / 10.0)))


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
        mag[(int(m.group(1)), m.group(2))] = np.array([
            float(m.group(3)), float(m.group(4)), float(m.group(5)),
        ])
    return {'energy_per_site': energy, 'corr': corr, 'mag': mag}


# ──────────────────────────────────────────────────────────────────────────────
# 2. load_folder_data  (simplified: no job-log needed)
# ──────────────────────────────────────────────────────────────────────────────
def load_folder_data(folder_path):
    """
    For each D present:
      1. Find the highest plain chi value.
      2. If a +2D file exists for that chi → use the +2D file.
      3. Otherwise use the plain file.
    Returns: dict { D: parsed_data }
    """
    # Collect plain chi → path  and (D, chi) → +2D path
    plain_map = {}     # D → list of (chi, path)
    plus2d_map = {}    # (D, chi) → (eff_chi, path)
    for fname in os.listdir(folder_path):
        m = RE_PLAIN_CHI.match(fname)
        if m:
            D, chi = int(m.group(1)), int(m.group(2))
            plain_map.setdefault(D, []).append((chi, os.path.join(folder_path, fname)))
        m2 = RE_PLUS2D.match(fname)
        if m2:
            D, chi_base, eff_chi = int(m2.group(1)), int(m2.group(2)), int(m2.group(3))
            plus2d_map[(D, chi_base)] = (eff_chi, os.path.join(folder_path, fname))

    result = {}
    for D in sorted(plain_map.keys()):
        best_plain_chi, best_plain_path = max(plain_map[D], key=lambda x: x[0])
        # Check if +2D exists for that chi
        if (D, best_plain_chi) in plus2d_map:
            eff_chi, use_path = plus2d_map[(D, best_plain_chi)]
        else:
            eff_chi, use_path = best_plain_chi, best_plain_path
        data = parse_plain_file(use_path)
        data['chi_base'] = eff_chi
        result[D] = data
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
        inv2   = 1.0 / Ds_f[-2:]
        y2     = eps_f[-2:]
        coeffs = np.polyfit(inv2, y2, 1)
        E_lin2 = float(np.polyval(coeffs, 0.0))
    else:
        E_lin2 = E_horiz

    E_exp_all = None
    exp_popt  = None
    if len(Ds_f) >= 2:
        delta   = abs(eps_f[-1] - eps_f[0])
        E0_g    = eps_f[-1] - delta * 0.1
        c_g     = eps_f[0] - E0_g
        Dchar_g = 4.0
        try:
            popt, _ = curve_fit(_exp_model, Ds_f, eps_f, p0=[E0_g, c_g, Dchar_g],
                                maxfev=5000)
            E_exp_all = float(popt[0])
            exp_popt  = popt.tolist()
        except Exception:
            pass

    lo, hi = min(E_horiz, E_lin2), max(E_horiz, E_lin2)
    if E_exp_all is not None and lo <= E_exp_all <= hi:
        E_0 = E_exp_all
    elif E_exp_all is not None:
        E_0 = (E_horiz + E_lin2) / 2.0
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
# 4. compute_bond_groups  — ranking per-D (no global ranking)
# ──────────────────────────────────────────────────────────────────────────────
def compute_bond_groups(D_data, groups_raw):
    """
    Returns: { D: { 'means': [...], 'stds': [...], 'sterrs': [...], 'ranks': [...] } }
    ranks assigned independently at each D: rank 1 = most-negative value.
    """
    result = {}
    for D in sorted(D_data.keys()):
        corr = D_data[D]['corr']
        means, stds, sterrs = [], [], []
        for grp in groups_raw:
            vals = [corr[(env, pair)] for (env, pair) in grp if (env, pair) in corr]
            if vals:
                s = float(np.std(vals, ddof=0))
                means.append(float(np.mean(vals)))
                stds.append(s)
                sterrs.append(s / np.sqrt(len(vals)))
            else:
                means.append(float('nan'))
                stds.append(float('nan'))
                sterrs.append(float('nan'))
        order = np.argsort(means)
        ranks = [0, 0, 0]
        for rank_val, idx in enumerate(order, start=1):
            ranks[idx] = rank_val
        result[D] = {'means': means, 'stds': stds, 'sterrs': sterrs, 'ranks': ranks}
    return result


# ──────────────────────────────────────────────────────────────────────────────
# 5. compute_order_param
# ──────────────────────────────────────────────────────────────────────────────
def compute_order_param(D_data):
    result = {}
    for D in sorted(D_data.keys()):
        mag = D_data[D]['mag']
        vecs = []
        for env in [1, 2, 3]:
            for site in SITES_ACE:
                k = (env, site)
                if k in mag:
                    vecs.append(mag[k].copy())
            for site in SITES_BDF:
                k = (env, site)
                if k in mag:
                    vecs.append(-mag[k].copy())
        if not vecs:
            continue
        vecs     = np.array(vecs)
        mean_vec = vecs.mean(axis=0)
        m        = float(np.linalg.norm(mean_vec))
        diffs    = vecs - mean_vec[None, :]
        err      = float(np.sqrt(np.mean(np.sum(diffs**2, axis=1))))
        result[D] = {'m': m, 'err': err}
    return result


# ──────────────────────────────────────────────────────────────────────────────
# 6. collect_all_data  (no job-log, simple scan)
# ──────────────────────────────────────────────────────────────────────────────
def collect_all_data(data_dir):
    # For duplicate (ansatz, J2) folders, keep the latest (by timestamp in name).
    best_folder = {}   # (ansatz_key, j2_str) → (timestamp_str, fname)

    for fname in sorted(os.listdir(data_dir)):
        folder_path = os.path.join(data_dir, fname)
        if not os.path.isdir(folder_path):
            continue
        if fname.startswith('6tensors'):
            ansatz = '6tensors'
        elif fname.startswith('1tensor_C6Ypi'):
            ansatz = 'c6ypi'
        else:
            continue
        m_j2 = RE_J2.search(fname)
        if not m_j2:
            continue
        j2_key = m_j2.group(1)
        # Extract timestamp from folder name (last part after last _)
        parts = fname.rsplit('_', 2)
        ts = parts[-2] + parts[-1] if len(parts) >= 3 else fname
        key = (ansatz, j2_key)
        if key not in best_folder or ts > best_folder[key][0]:
            best_folder[key] = (ts, fname)

    all_results = {}
    for (ansatz, j2_key), (_, fname) in sorted(best_folder.items()):
        folder_path = os.path.join(data_dir, fname)
        j2 = float('0.' + j2_key)

        D_data = load_folder_data(folder_path)
        if not D_data:
            print(f"  SKIP (no data files): {fname}")
            continue

        Ds_sorted  = sorted(D_data.keys())
        eps_sorted = [D_data[D]['energy_per_site'] for D in Ds_sorted]
        extrap     = compute_energy_extrap(Ds_sorted, eps_sorted)
        nn_groups  = compute_bond_groups(D_data, NN_GROUPS_RAW)
        nnn_groups = compute_bond_groups(D_data, NNN_GROUPS_RAW)
        order_p    = compute_order_param(D_data)

        all_results[fname] = {
            'ansatz':          ansatz,
            'j2':              j2,
            'Ds':              Ds_sorted,
            'energy_per_site': eps_sorted,
            'chi_bases':       [D_data[D]['chi_base'] for D in Ds_sorted],
            'extrap':          extrap,
            'nn_groups':       dict(nn_groups),
            'nnn_groups':      dict(nnn_groups),
            'order_param':     dict(order_p),
        }
        print(f"  Loaded {fname}: J2={j2} ansatz={ansatz} Ds={Ds_sorted}")

    return all_results


# ──────────────────────────────────────────────────────────────────────────────
# 7. save_json
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

def save_json(all_results, out_dir):
    fpath = os.path.join(out_dir, 'analysis_results.json')
    with open(fpath, 'w') as f:
        json.dump(_jsonify(all_results), f, indent=2)
    print(f"Saved JSON: {fpath}")


# ──────────────────────────────────────────────────────────────────────────────
# PLOTTING UTILITIES
# ──────────────────────────────────────────────────────────────────────────────
def _save(fig, fpath):
    fig.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {os.path.relpath(fpath)}")


def _share_limits(*axes):
    """Make all visible axes share the same xlim and ylim."""
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


def _D_colormap_ansatz(D_list, ansatz):
    Ds   = sorted(set(D_list))
    cname = 'Blues' if ansatz == '6tensors' else 'Oranges'
    cmap  = plt.get_cmap(cname)
    if len(Ds) == 1:
        return {Ds[0]: cmap(0.65)}
    return {D: cmap(0.30 + 0.60 * i / (len(Ds) - 1)) for i, D in enumerate(Ds)}


def _D_colormap_rank(D_list, rank):
    Ds   = sorted(set(D_list))
    cmap = plt.get_cmap(RANK_CMAPS[rank])
    if len(Ds) == 1:
        return {Ds[0]: cmap(0.65)}
    return {D: cmap(0.30 + 0.60 * i / (len(Ds) - 1)) for i, D in enumerate(Ds)}


def _get_ansatz_data(all_results, ansatz):
    return {k: v for k, v in all_results.items() if v['ansatz'] == ansatz}


def _get_j2_data(sub, j2):
    matches = [v for v in sub.values() if v['j2'] == j2]
    return matches[0] if matches else None


# ──────────────────────────────────────────────────────────────────────────────
# 8. Energy vs 1/D  – one figure per J2
#    (0,0) unres | (0,1) C6+π | (1,0) both
# ──────────────────────────────────────────────────────────────────────────────
def _plot_energy_ax(ax, v, col, lbl, marker='o'):
    """Plot E vs 1/D with horizontal line, linear fit, and E₀ star on ax."""
    Ds  = np.array(v['Ds'], dtype=float)
    eps = np.array(v['energy_per_site'])
    ex  = v['extrap']
    inv = 1.0 / Ds

    ax.plot(inv, eps, marker + '-', color=col, ms=5, lw=1.4, label=lbl)

    x_max = max(inv) * 1.15
    ax.plot([0, x_max], [ex['E_horiz'], ex['E_horiz']],
            color=col, lw=0.8, ls=':', alpha=0.7, label='_nolegend_')

    if len(Ds) >= 2:
        inv2   = 1.0 / Ds[-2:]
        y2     = eps[-2:]
        coeffs = np.polyfit(inv2, y2, 1)
        x_lin  = np.array([0.0, inv2[0]])
        ax.plot(x_lin, np.polyval(coeffs, x_lin),
                color=col, lw=1.0, ls='--', alpha=0.8, label='_nolegend_')

    ax.plot(0, ex['E_0'], marker='*', color=col, ms=11, zorder=6,
            markeredgewidth=0.4, markeredgecolor='k', label='_nolegend_')

    ax.set_xlim(left=0)
    ax.set_xlabel('1/D', fontsize=11)
    ax.set_ylabel('Energy per site', fontsize=11)


def plot_energy_vs_invD(all_results, out_dir):
    subdir = os.path.join(out_dir, 'energy_vs_invD')
    os.makedirs(subdir, exist_ok=True)

    sub_6t = _get_ansatz_data(all_results, '6tensors')
    sub_c6 = _get_ansatz_data(all_results, 'c6ypi')
    all_j2 = sorted(set(v['j2'] for v in all_results.values()))

    for j2 in all_j2:
        v_6t = _get_j2_data(sub_6t, j2)
        v_c6 = _get_j2_data(sub_c6, j2)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        ax_6t, ax_c6, ax_both, ax_empty = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
        ax_empty.set_visible(False)

        if v_6t:
            sty = ANSATZ_STYLE['6tensors']
            _plot_energy_ax(ax_6t, v_6t, sty['color'], sty['label'], sty['marker'])
            ax_6t.set_title(f'{sty["label"]}  –  {_j2_label(j2)}', fontsize=10)
            _plot_energy_ax(ax_both, v_6t, sty['color'], sty['label'], sty['marker'])
        else:
            ax_6t.set_visible(False)

        if v_c6:
            sty = ANSATZ_STYLE['c6ypi']
            _plot_energy_ax(ax_c6, v_c6, sty['color'], sty['label'], sty['marker'])
            ax_c6.set_title(f'{sty["label"]}  –  {_j2_label(j2)}', fontsize=10)
            _plot_energy_ax(ax_both, v_c6, sty['color'], sty['label'], sty['marker'])
        else:
            ax_c6.set_visible(False)

        _share_limits(ax_6t, ax_c6)

        ax_both.set_title(f'Both ansätze  –  {_j2_label(j2)}', fontsize=10)

        fig.suptitle(
            f'Energy vs 1/D  –  {_j2_label(j2)}\n'
            '(dotted = horizontal, dashed = linear last-2, ★ = E₀)',
            fontsize=11,
        )
        fig.legend(*ax_both.get_legend_handles_labels(),
                   loc='lower right', bbox_to_anchor=(0.95, 0.05), fontsize=9)
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        _save(fig, os.path.join(subdir, f'J2_{_j2_fname(j2)}.pdf'))


# ──────────────────────────────────────────────────────────────────────────────
# 9. Energy E₀ vs J2
#    (0,0) unres | (0,1) C6+π | (1,0) both
# ──────────────────────────────────────────────────────────────────────────────
def plot_energy_vs_J2(all_results, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_6t, ax_c6, ax_both, ax_empty = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    ax_empty.set_visible(False)

    for ansatz, ax_single in [('6tensors', ax_6t), ('c6ypi', ax_c6)]:
        sub = _get_ansatz_data(all_results, ansatz)
        if not sub:
            ax_single.set_visible(False)
            continue
        sty  = ANSATZ_STYLE[ansatz]
        rows = sorted(sub.values(), key=lambda v: v['j2'])
        j2s  = [r['j2'] for r in rows]
        E0s  = [r['extrap']['E_0']     for r in rows]
        Ehor = [r['extrap']['E_horiz'] for r in rows]
        Elin = [r['extrap']['E_lin2']  for r in rows]

        yerr_up   = np.clip(np.array(Ehor) - np.array(E0s),  0, None)
        yerr_down = np.clip(np.array(E0s)  - np.array(Elin), 0, None)

        for ax in (ax_single, ax_both):
            ax.errorbar(j2s, E0s, yerr=[yerr_down, yerr_up],
                        fmt=sty['marker'] + '-', capsize=4, color=sty['color'],
                        label=sty['label'])
        ax_single.set_title(f'{sty["label"]}', fontsize=10)

    _share_limits(ax_6t, ax_c6)

    for ax in (ax_6t, ax_c6, ax_both):
        ax.set_xlabel('J2', fontsize=11)
        ax.set_ylabel('E₀ (energy per site)', fontsize=11)
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.5g'))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.5g'))

    ax_both.set_title('Both ansätze compared', fontsize=10)
    fig.suptitle('Extrapolated energy E₀ vs J2\n(error bar up = E_horiz, down = E_lin2)',
                 fontsize=11)
    fig.legend(*ax_both.get_legend_handles_labels(),
               loc='lower right', bbox_to_anchor=(0.95, 0.05), fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, os.path.join(out_dir, 'energy_E0_vs_J2.pdf'))


# ──────────────────────────────────────────────────────────────────────────────
# 10. Bond correlations vs 1/D  – one figure per J2
#     (0,0) unres all 3 ranks | (0,1) C6+π all 3 ranks | (1,0) both overlaid
#     Ranking per-D to avoid crossings.
# ──────────────────────────────────────────────────────────────────────────────
def _plot_bonds_invD_ax(ax, v, bond_key):
    """Plot all 3 rank-curves on a single ax for one (ansatz, J2)."""
    grp_data = v[bond_key]
    for target_rank in [1, 2, 3]:
        xs, ys, yes = [], [], []
        for D, gd in sorted(grp_data.items()):
            gi = next((i for i, r in enumerate(gd['ranks']) if r == target_rank), None)
            if gi is None:
                continue
            xs.append(1.0 / D)
            ys.append(gd['means'][gi])
            yes.append(gd['sterrs'][gi])
        if xs:
            ax.errorbar(xs, ys, yerr=yes, fmt='o-', color=RANK_COLORS[target_rank],
                        ms=5, lw=1.3, capsize=3, elinewidth=0.8,
                        label=RANK_LABELS[target_rank])
    ax.set_xlim(left=0)
    ax.set_xlabel('1/D', fontsize=11)
    ax.set_ylabel('⟨Sᵢ·Sⱼ⟩', fontsize=11)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.4g'))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.5g'))


def _plot_bonds_invD_rank_compare(ax, v_6t, v_c6, bond_key, rank):
    """Plot one rank for both ansätze on same ax, vs 1/D."""
    for ansatz_key, v, mkr, ls_sty in [('6tensors', v_6t, 'o', '-'),
                                        ('c6ypi', v_c6, 's', '--')]:
        if v is None:
            continue
        grp_data = v[bond_key]
        xs, ys, yes = [], [], []
        for D, gd in sorted(grp_data.items()):
            gi = next((i for i, r in enumerate(gd['ranks']) if r == rank), None)
            if gi is None:
                continue
            xs.append(1.0 / D)
            ys.append(gd['means'][gi])
            yes.append(gd['sterrs'][gi])
        if xs:
            lbl = ANSATZ_STYLE[ansatz_key]['label']
            ax.errorbar(xs, ys, yerr=yes, fmt=mkr + ls_sty,
                        color=ANSATZ_STYLE[ansatz_key]['color'],
                        ms=5, lw=1.3, capsize=3, elinewidth=0.8,
                        label=lbl)
    ax.set_xlim(left=0)
    ax.set_xlabel('1/D', fontsize=11)
    ax.set_ylabel('⟨Sᵢ·Sⱼ⟩', fontsize=11)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.4g'))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.5g'))


def plot_bonds_vs_invD(all_results, out_dir, bond_key, bond_type_label):
    bond_short = bond_key.replace('_groups', '')
    subdir = os.path.join(out_dir, f'{bond_short}_vs_invD')
    os.makedirs(subdir, exist_ok=True)

    sub_6t = _get_ansatz_data(all_results, '6tensors')
    sub_c6 = _get_ansatz_data(all_results, 'c6ypi')
    all_j2 = sorted(set(v['j2'] for v in all_results.values()))

    for j2 in all_j2:
        v_6t = _get_j2_data(sub_6t, j2)
        v_c6 = _get_j2_data(sub_c6, j2)

        fig = plt.figure(figsize=(14, 18))
        gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.30)
        ax_6t  = fig.add_subplot(gs[0, 0])
        ax_c6  = fig.add_subplot(gs[0, 1])
        ax_all = fig.add_subplot(gs[1, 0])
        ax_r1  = fig.add_subplot(gs[1, 1])
        ax_r2  = fig.add_subplot(gs[2, 0])
        ax_r3  = fig.add_subplot(gs[2, 1])

        # (0,0) unres all 3 ranks
        if v_6t:
            _plot_bonds_invD_ax(ax_6t, v_6t, bond_key)
            ax_6t.set_title(f'{ANSATZ_STYLE["6tensors"]["label"]}  –  {_j2_label(j2)}', fontsize=10)
        else:
            ax_6t.set_visible(False)

        # (0,1) C6+π all 3 ranks
        if v_c6:
            _plot_bonds_invD_ax(ax_c6, v_c6, bond_key)
            ax_c6.set_title(f'{ANSATZ_STYLE["c6ypi"]["label"]}  –  {_j2_label(j2)}', fontsize=10)
        else:
            ax_c6.set_visible(False)

        # (1,0) both ansätze all 3 ranks overlaid
        if v_6t:
            _plot_bonds_invD_ax(ax_all, v_6t, bond_key)
        if v_c6:
            grp_data = v_c6[bond_key]
            for target_rank in [1, 2, 3]:
                xs, ys, yes = [], [], []
                for D, gd in sorted(grp_data.items()):
                    gi = next((i for i, r in enumerate(gd['ranks']) if r == target_rank), None)
                    if gi is None:
                        continue
                    xs.append(1.0 / D)
                    ys.append(gd['means'][gi])
                    yes.append(gd['sterrs'][gi])
                if xs:
                    ax_all.errorbar(xs, ys, yerr=yes, fmt='s--',
                                    color=RANK_COLORS[target_rank],
                                    ms=5, lw=1.1, capsize=3, elinewidth=0.8, alpha=0.7,
                                    label=f'{RANK_LABELS[target_rank]} (C6+π)')
        ax_all.set_xlim(left=0)
        ax_all.set_xlabel('1/D', fontsize=11)
        ax_all.set_ylabel('⟨Sᵢ·Sⱼ⟩', fontsize=11)
        ax_all.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.4g'))
        ax_all.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.5g'))
        ax_all.set_title(f'Both ansätze  –  {_j2_label(j2)}', fontsize=10)

        # (1,1) rank1, (2,0) rank2, (2,1) rank3 comparison
        for ax, rank in [(ax_r1, 1), (ax_r2, 2), (ax_r3, 3)]:
            _plot_bonds_invD_rank_compare(ax, v_6t, v_c6, bond_key, rank)
            ax.set_title(f'{RANK_LABELS[rank]}  –  both ansätze', fontsize=10)

        _share_limits(ax_6t, ax_c6, ax_all, ax_r1, ax_r2, ax_r3)

        fig.suptitle(f'{bond_type_label}  vs  1/D  –  {_j2_label(j2)}', fontsize=11)
        # Collect unique legend handles
        all_handles, all_labels = [], []
        for ax in [ax_6t, ax_c6, ax_all, ax_r1, ax_r2, ax_r3]:
            h, l = ax.get_legend_handles_labels()
            for hi, li in zip(h, l):
                if li not in all_labels:
                    all_handles.append(hi)
                    all_labels.append(li)
        if all_handles:
            fig.legend(all_handles, all_labels, loc='lower right',
                       bbox_to_anchor=(0.98, 0.02), fontsize=6, ncol=3, frameon=True)
        plt.tight_layout(rect=[0, 0.04, 1, 0.95])
        _save(fig, os.path.join(subdir, f'J2_{_j2_fname(j2)}.pdf'))


# ──────────────────────────────────────────────────────────────────────────────
# 11. Bond correlations vs J2
#     (0,0) unres 3D-curves | (0,1) C6+π 3D-curves
#     (1,0) rank1 both | (1,1) rank2 both | (2,0) rank3 both
# ──────────────────────────────────────────────────────────────────────────────
def _select_three_Ds(all_Ds):
    """Select [5, 7, hi] where hi is 9 if present else 8."""
    chosen = [d for d in [5, 7] if d in all_Ds]
    for d in [9, 8]:
        if d in all_Ds:
            chosen.append(d)
            break
    return chosen


def _gather_bonds_vsJ2(sub, bond_key):
    """
    Returns { rank → { D → [(j2, val, sterr), ...] } }
    Ranking done per-(D, J2) — each folder's grp_data already has per-D ranks.
    """
    data_by_rank_D = {r: {} for r in [1, 2, 3]}
    for v in sub.values():
        j2       = v['j2']
        grp_data = v[bond_key]
        for D, gd in grp_data.items():
            for rank in [1, 2, 3]:
                gi = next((i for i, r in enumerate(gd['ranks']) if r == rank), None)
                if gi is None:
                    continue
                data_by_rank_D[rank].setdefault(D, []).append(
                    (j2, gd['means'][gi], gd['sterrs'][gi]))
    return data_by_rank_D


def _plot_bonds_vsJ2_allranks(ax, sub, bond_key, ansatz):
    """Plot all 3 ranks with 3 D-curves each on one ax."""
    data = _gather_bonds_vsJ2(sub, bond_key)
    all_Ds = sorted(set(D for rd in data.values() for D in rd))
    chosen = _select_three_Ds(all_Ds)
    if not chosen:
        return
    for rank in [1, 2, 3]:
        D_col = _D_colormap_rank(chosen, rank)
        for D in chosen:
            pts = sorted(data[rank].get(D, []))
            if not pts:
                continue
            j2s  = [p[0] for p in pts]
            vals = [p[1] for p in pts]
            errs = [p[2] for p in pts]
            alpha = _D_alpha(D)
            ax.errorbar(j2s, vals, yerr=errs, fmt='o-', color=D_col[D], ms=5,
                        alpha=alpha, capsize=3, elinewidth=0.8,
                        label=f'{RANK_LABELS[rank]} D={D}')
    ax.set_xlabel('J2', fontsize=10)
    ax.set_ylabel('⟨Sᵢ·Sⱼ⟩', fontsize=10)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.5g'))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.5g'))


def _plot_bonds_vsJ2_rank(ax, sub_6t, sub_c6, bond_key, rank):
    """Plot one rank with 3 D-curves, both ansätze on same ax."""
    for ansatz_key, sub, mkr, ls in [('6tensors', sub_6t, 'o', '-'),
                                      ('c6ypi',   sub_c6, 's', '--')]:
        if not sub:
            continue
        data = _gather_bonds_vsJ2(sub, bond_key)
        all_Ds = sorted(set(D for rd in data.values() for D in rd))
        chosen = _select_three_Ds(all_Ds)
        D_col  = _D_colormap_ansatz(chosen, ansatz_key)
        lbl    = ANSATZ_STYLE[ansatz_key]['label']
        for D in chosen:
            pts = sorted(data[rank].get(D, []))
            if not pts:
                continue
            j2s  = [p[0] for p in pts]
            vals = [p[1] for p in pts]
            errs = [p[2] for p in pts]
            alpha = _D_alpha(D)
            ax.errorbar(j2s, vals, yerr=errs, fmt=mkr+ls, color=D_col[D], ms=5,
                        alpha=alpha, capsize=3, elinewidth=0.8,
                        label=f'{lbl} D={D}')
    ax.set_xlabel('J2', fontsize=10)
    ax.set_ylabel('⟨Sᵢ·Sⱼ⟩', fontsize=10)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.5g'))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.5g'))


def _plot_bonds_vsJ2_all_both(ax, sub_6t, sub_c6, bond_key):
    """Plot all 3 ranks × both ansätze on one ax, vs J2 with D as alpha."""
    for ansatz_key, sub, mkr, ls in [('6tensors', sub_6t, 'o', '-'),
                                      ('c6ypi',   sub_c6, 's', '--')]:
        if not sub:
            continue
        data = _gather_bonds_vsJ2(sub, bond_key)
        all_Ds = sorted(set(D for rd in data.values() for D in rd))
        chosen = _select_three_Ds(all_Ds)
        if not chosen:
            continue
        lbl = ANSATZ_STYLE[ansatz_key]['label']
        for rank in [1, 2, 3]:
            D_col = _D_colormap_rank(chosen, rank)
            for D in chosen:
                pts = sorted(data[rank].get(D, []))
                if not pts:
                    continue
                j2s  = [p[0] for p in pts]
                vals = [p[1] for p in pts]
                errs = [p[2] for p in pts]
                alpha = _D_alpha(D)
                ax.errorbar(j2s, vals, yerr=errs, fmt=mkr+ls, color=D_col[D],
                            ms=5, alpha=alpha, capsize=3, elinewidth=0.8,
                            label=f'{RANK_LABELS[rank]} {lbl} D={D}')
    ax.set_xlabel('J2', fontsize=10)
    ax.set_ylabel('⟨Sᵢ·Sⱼ⟩', fontsize=10)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.5g'))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.5g'))


def plot_bonds_vs_J2(all_results, out_dir, bond_key, bond_type_label):
    sub_6t = _get_ansatz_data(all_results, '6tensors')
    sub_c6 = _get_ansatz_data(all_results, 'c6ypi')

    fig = plt.figure(figsize=(14, 16))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.30)
    ax_6t  = fig.add_subplot(gs[0, 0])
    ax_c6  = fig.add_subplot(gs[0, 1])
    ax_all = fig.add_subplot(gs[1, 0])
    ax_r1  = fig.add_subplot(gs[1, 1])
    ax_r2  = fig.add_subplot(gs[2, 0])
    ax_r3  = fig.add_subplot(gs[2, 1])

    # (0,0) unres all ranks, 3 D-curves
    if sub_6t:
        _plot_bonds_vsJ2_allranks(ax_6t, sub_6t, bond_key, '6tensors')
        ax_6t.set_title(f'{ANSATZ_STYLE["6tensors"]["label"]}', fontsize=10)
    else:
        ax_6t.set_visible(False)

    # (0,1) C6+π all ranks, 3 D-curves
    if sub_c6:
        _plot_bonds_vsJ2_allranks(ax_c6, sub_c6, bond_key, 'c6ypi')
        ax_c6.set_title(f'{ANSATZ_STYLE["c6ypi"]["label"]}', fontsize=10)
    else:
        ax_c6.set_visible(False)

    # (1,0) all 3 ranks × both ansätze
    _plot_bonds_vsJ2_all_both(ax_all, sub_6t, sub_c6, bond_key)
    ax_all.set_title('All ranks – both ansätze', fontsize=10)

    # (1,1) rank1, (2,0) rank2, (2,1) rank3
    for ax, rank in [(ax_r1, 1), (ax_r2, 2), (ax_r3, 3)]:
        _plot_bonds_vsJ2_rank(ax, sub_6t, sub_c6, bond_key, rank)
        ax.set_title(f'{RANK_LABELS[rank]}  –  both ansätze', fontsize=10)

    _share_limits(ax_6t, ax_c6, ax_all, ax_r1, ax_r2, ax_r3)

    fig.suptitle(f'{bond_type_label}  vs  J2', fontsize=12)

    # Collect all handles for shared legend
    all_handles, all_labels = [], []
    for ax in [ax_6t, ax_c6, ax_all, ax_r1, ax_r2, ax_r3]:
        h, l = ax.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li not in all_labels:
                all_handles.append(hi)
                all_labels.append(li)
    if all_handles:
        fig.legend(all_handles, all_labels,
                   loc='lower right', bbox_to_anchor=(0.98, 0.02),
                   fontsize=6, ncol=3, frameon=True)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    _save(fig, os.path.join(out_dir, f'{bond_key.replace("_groups","")}_vs_J2.pdf'))


# ──────────────────────────────────────────────────────────────────────────────
# 12. Néel order parameter vs 1/D  – one figure per J2
#     (0,0) unres | (0,1) C6+π | (1,0) both
# ──────────────────────────────────────────────────────────────────────────────
def _plot_order_invD_ax(ax, v, col, lbl, marker='o'):
    op = v['order_param']
    Ds_op = sorted(op.keys())
    if not Ds_op:
        return
    inv  = [1.0 / D for D in Ds_op]
    ms_v = [op[D]['m']   for D in Ds_op]
    errs = [op[D]['err'] for D in Ds_op]
    ax.errorbar(inv, ms_v, yerr=errs, fmt=marker + '-', color=col,
                ms=5, lw=1.4, capsize=3, label=lbl)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel('1/D', fontsize=11)
    ax.set_ylabel('Néel order m', fontsize=11)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.4g'))


def plot_order_param_vs_invD(all_results, out_dir):
    subdir = os.path.join(out_dir, 'order_param_vs_invD')
    os.makedirs(subdir, exist_ok=True)

    sub_6t = _get_ansatz_data(all_results, '6tensors')
    sub_c6 = _get_ansatz_data(all_results, 'c6ypi')
    all_j2 = sorted(set(v['j2'] for v in all_results.values()))

    for j2 in all_j2:
        v_6t = _get_j2_data(sub_6t, j2)
        v_c6 = _get_j2_data(sub_c6, j2)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        ax_6t, ax_c6, ax_both, ax_empty = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
        ax_empty.set_visible(False)

        if v_6t:
            sty = ANSATZ_STYLE['6tensors']
            _plot_order_invD_ax(ax_6t, v_6t, sty['color'], sty['label'], sty['marker'])
            ax_6t.set_title(f'{sty["label"]}  –  {_j2_label(j2)}', fontsize=10)
            _plot_order_invD_ax(ax_both, v_6t, sty['color'], sty['label'], sty['marker'])
        else:
            ax_6t.set_visible(False)

        if v_c6:
            sty = ANSATZ_STYLE['c6ypi']
            _plot_order_invD_ax(ax_c6, v_c6, sty['color'], sty['label'], sty['marker'])
            ax_c6.set_title(f'{sty["label"]}  –  {_j2_label(j2)}', fontsize=10)
            _plot_order_invD_ax(ax_both, v_c6, sty['color'], sty['label'], sty['marker'])
        else:
            ax_c6.set_visible(False)

        _share_limits(ax_6t, ax_c6)

        ax_both.set_title(f'Both ansätze  –  {_j2_label(j2)}', fontsize=10)

        fig.suptitle(f'Néel order m  vs  1/D  –  {_j2_label(j2)}', fontsize=11)
        fig.legend(*ax_both.get_legend_handles_labels(),
                   loc='lower right', bbox_to_anchor=(0.95, 0.05), fontsize=9)
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        _save(fig, os.path.join(subdir, f'J2_{_j2_fname(j2)}.pdf'))


# ──────────────────────────────────────────────────────────────────────────────
# 13. Néel order parameter vs J2
#     (0,0) unres 3D | (0,1) C6+π 3D | (1,0) both
# ──────────────────────────────────────────────────────────────────────────────
def _gather_order_vsJ2(sub):
    """Return { D: [(j2, m, err), ...] }."""
    data_by_D = {}
    for v in sub.values():
        j2 = v['j2']
        for D, op in v['order_param'].items():
            data_by_D.setdefault(D, []).append((j2, op['m'], op['err']))
    return data_by_D


def _plot_order_vsJ2_ax(ax, sub, ansatz):
    data_by_D = _gather_order_vsJ2(sub)
    all_Ds = sorted(data_by_D.keys())
    chosen = _select_three_Ds(all_Ds)
    D_col  = _D_colormap_ansatz(chosen, ansatz)
    lbl    = ANSATZ_STYLE[ansatz]['label']
    mkr    = ANSATZ_STYLE[ansatz]['marker']
    for D in chosen:
        pts  = sorted(data_by_D.get(D, []))
        if not pts:
            continue
        j2s  = [p[0] for p in pts]
        ms   = [p[1] for p in pts]
        errs = [p[2] for p in pts]
        alpha = _D_alpha(D)
        ax.errorbar(j2s, ms, yerr=errs, fmt=mkr+'-', color=D_col[D], ms=5,
                    alpha=alpha, capsize=3, elinewidth=0.8,
                    label=f'{lbl} D={D}')
    ax.set_xlabel('J2', fontsize=10)
    ax.set_ylabel('Néel order m', fontsize=10)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.5g'))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.5g'))


def plot_order_param_vs_J2(all_results, out_dir):
    sub_6t = _get_ansatz_data(all_results, '6tensors')
    sub_c6 = _get_ansatz_data(all_results, 'c6ypi')

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_6t, ax_c6, ax_both, ax_empty = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    ax_empty.set_visible(False)

    if sub_6t:
        _plot_order_vsJ2_ax(ax_6t, sub_6t, '6tensors')
        ax_6t.set_title(f'{ANSATZ_STYLE["6tensors"]["label"]}', fontsize=10)
        _plot_order_vsJ2_ax(ax_both, sub_6t, '6tensors')
    else:
        ax_6t.set_visible(False)

    if sub_c6:
        _plot_order_vsJ2_ax(ax_c6, sub_c6, 'c6ypi')
        ax_c6.set_title(f'{ANSATZ_STYLE["c6ypi"]["label"]}', fontsize=10)
        _plot_order_vsJ2_ax(ax_both, sub_c6, 'c6ypi')
    else:
        ax_c6.set_visible(False)

    _share_limits(ax_6t, ax_c6)

    ax_both.set_title('Both ansätze compared', fontsize=10)

    fig.suptitle('Néel order parameter  m  vs  J2', fontsize=12)
    all_handles, all_labels = [], []
    for ax in [ax_6t, ax_c6, ax_both]:
        h, l = ax.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li not in all_labels:
                all_handles.append(hi)
                all_labels.append(li)
    if all_handles:
        fig.legend(all_handles, all_labels,
                   loc='lower right', bbox_to_anchor=(0.98, 0.02),
                   fontsize=7, ncol=3, frameon=True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    _save(fig, os.path.join(out_dir, 'order_param_vs_J2.pdf'))


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print('=== 0418core comprehensive analysis ===')
    print(f'Data dir : {DATA_DIR}')
    print(f'Output   : {OUT_DIR}')
    print()

    all_results = collect_all_data(DATA_DIR)
    print(f'\nLoaded {len(all_results)} folder(s).\n')

    save_json(all_results, OUT_DIR)

    print('\n--- Energy vs 1/D ---')
    plot_energy_vs_invD(all_results, OUT_DIR)

    print('\n--- Energy E₀ vs J2 ---')
    plot_energy_vs_J2(all_results, OUT_DIR)

    print('\n--- NN bond correlations vs 1/D ---')
    plot_bonds_vs_invD(all_results, OUT_DIR, 'nn_groups',  'NN bond ⟨Sᵢ·Sⱼ⟩')

    print('\n--- NN bond correlations vs J2 ---')
    plot_bonds_vs_J2(all_results, OUT_DIR, 'nn_groups',  'NN bond ⟨Sᵢ·Sⱼ⟩')

    print('\n--- NNN bond correlations vs 1/D ---')
    plot_bonds_vs_invD(all_results, OUT_DIR, 'nnn_groups', 'NNN bond ⟨Sᵢ·Sⱼ⟩')

    print('\n--- NNN bond correlations vs J2 ---')
    plot_bonds_vs_J2(all_results, OUT_DIR, 'nnn_groups', 'NNN bond ⟨Sᵢ·Sⱼ⟩')

    print('\n--- Néel order parameter vs 1/D ---')
    plot_order_param_vs_invD(all_results, OUT_DIR)

    print('\n--- Néel order parameter vs J2 ---')
    plot_order_param_vs_J2(all_results, OUT_DIR)

    print('\nDone. All figures written to:', OUT_DIR)


if __name__ == '__main__':
    main()
