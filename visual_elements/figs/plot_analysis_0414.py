#!/usr/bin/env python3
"""
Comprehensive 6AD-CTMRG analysis script.

Generates:
  - Energy vs 1/D  (per ansatz, with 3 extrapolation lines)
  - E_0 vs J2      (asymmetric error bars: E_horiz above, E_lin2 below)
  - NN bond corr   vs 1/D  (6 figures: 3 ranks × 2 ansätze ... combined by rank)
  - NN bond corr   vs J2   (6 figures: 3 ranks × 2 ansätze ... combined by rank)
  - NNN bond corr  vs 1/D  (6 figures: 3 ranks × 2 ansätze)
  - NNN bond corr  vs J2   (6 figures: 3 ranks × 2 ansätze)
  - Néel order m   vs 1/D  (both ansätze)
  - Néel order m   vs J2   (both ansätze)
  - Saves analysis_results.json to output directory

File selection: plain files (no '+2D' in name), highest chi_base per completed D.
Completed-D detection: 'D=X complete' lines in job-*.out.
"""

import os
import re
import glob
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from scipy.optimize import curve_fit


def _j2_str(j2):
    """Full-precision string for J2, used in filenames and labels.
    Strips trailing zeros but always keeps at least 2 decimal places.
    e.g. 0.20 → '0p20', 0.266 → '0p266', 0.26605 → '0p26605'  (filename form)
    """
    s = f'{j2:.10f}'.rstrip('0')   # full float, strip trailing zeros
    # ensure at least 2 decimal places
    dot = s.index('.')
    if len(s) - dot - 1 < 2:
        s = s + '0' * (2 - (len(s) - dot - 1))
    return s


def _j2_fname(j2):
    """Filename-safe J2 string: dots replaced by 'p'."""
    return _j2_str(j2).replace('.', 'p')


def _j2_label(j2):
    """Human-readable J2 label without trailing zeros, e.g. 'J2=0.266'."""
    return f'J2={_j2_str(j2)}'


def _fill_zoom_ax(ax_zoom, ax_main, xlim=(0.26, 0.28)):
    """
    Populate an existing ax_zoom with data re-plotted from ax_main, restricted to xlim.
    Only plots D=5, D=7, and (D=9 if available else D=8 if available).
    Always connects dots with lines in zoom plot.
    The caller must create ax_zoom beforehand.
    """
    # Pass 1: determine which D values are available in ax_main
    available_D = set()
    for cont in ax_main.containers:
        if not hasattr(cont, 'get_children'):
            continue
        lbl = cont.get_label()
        for d in [5, 7, 8, 9]:
            if f'D={d}' in lbl:
                available_D.add(d)
                break
    
    # Build allowed list: always [5, 7] + one of [9, 8] if available
    allowed_D = [5, 7]
    for d in [9, 8]:  # prefer D=9, fallback to D=8
        if d in available_D:
            allowed_D.append(d)
            break
    
    # Pass 2: re-plot every errorbar collection from ax_main that matches allowed D
    for cont in ax_main.containers:
        if not hasattr(cont, 'get_children'):
            continue
        
        lbl = cont.get_label()
        
        # Filter by allowed D values: check if label contains 'D=X' where X in allowed_D
        has_allowed_D = any(f'D={d}' in lbl for d in allowed_D)
        if not has_allowed_D:
            continue
        
        children = cont.get_children()
        # errorbar container: [line_marker, caplines..., errlines...]
        # Extract original data from the data line
        data_line = None
        for ch in children:
            if hasattr(ch, 'get_xdata') and len(getattr(ch, 'get_xdata', lambda: [])()) > 0:
                # first artist with x/y data is the marker line
                if hasattr(ch, 'get_markersize') and not hasattr(ch, 'get_segments'):
                    data_line = ch
                    break
        if data_line is None:
            continue
        xs  = np.asarray(data_line.get_xdata())
        ys  = np.asarray(data_line.get_ydata())
        col = data_line.get_color()
        mstyle = data_line.get_marker()
        ms    = data_line.get_markersize()
        lw    = data_line.get_linewidth()
        alpha = data_line.get_alpha() or 1.0
        # Get yerr from the error line segments if present
        yerr_lo = np.zeros(len(xs))
        yerr_hi = np.zeros(len(xs))
        for ch in children:
            if hasattr(ch, 'get_segments'):
                segs = ch.get_segments()
                for seg in segs:
                    if len(seg) == 2:
                        x0, y0 = seg[0]
                        x1, y1 = seg[1]
                        if abs(x0 - x1) < 1e-12:   # vertical → error bar
                            # find matching point index
                            idx = np.argmin(np.abs(xs - x0))
                            if idx < len(yerr_lo):
                                yerr_lo[idx] = ys[idx] - min(y0, y1)
                                yerr_hi[idx] = max(y0, y1) - ys[idx]
        mask = (xs >= xlim[0]) & (xs <= xlim[1])
        if mask.sum() == 0:
            continue
        # Always use marker + line in zoom plot (default to 'o-' if no marker found)
        if mstyle and mstyle != 'None':
            fmt = mstyle + '-'
        else:
            fmt = 'o-'
        ax_zoom.errorbar(
            xs[mask], ys[mask],
            yerr=[yerr_lo[mask], yerr_hi[mask]],
            fmt=fmt,
            color=col, ms=ms, lw=lw, alpha=alpha,
            capsize=3, elinewidth=0.8, label=lbl
        )
    ax_zoom.set_xlim(xlim)
    ax_zoom.set_xlabel('J2', fontsize=9)
    ax_zoom.set_ylabel(ax_main.get_ylabel(), fontsize=9)
    ax_zoom.set_title(f'Zoom  J2 ∈ [{xlim[0]}, {xlim[1]}]', fontsize=8)
    ax_zoom.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.5g'))
    ax_zoom.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.5g'))
    ax_zoom.tick_params(labelsize=8)

# ──────────────────────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────────────────────
DATA_DIR   = '/home/chye/6ADctmrg/data/0414core'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(SCRIPT_DIR, 'analysis_plots')
os.makedirs(OUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# BOND GROUP DEFINITIONS
# ──────────────────────────────────────────────────────────────────────────────
# Each entry: (env_int, raw_pair_str)
NN_GROUPS_RAW = [
    # rank-pool 0: env1 EB,AD,CF  +  env3 BE,FC,DA
    [(1,'EB'),(1,'AD'),(1,'CF'),(3,'BE'),(3,'FC'),(3,'DA')],
    # rank-pool 1: env2 CB,AF,ED  +  env1 FA,DE,BC
    [(2,'CB'),(2,'AF'),(2,'ED'),(1,'FA'),(1,'DE'),(1,'BC')],
    # rank-pool 2: env3 EF,AB,CD  +  env2 DC,BA,FE
    [(3,'EF'),(3,'AB'),(3,'CD'),(2,'DC'),(2,'BA'),(2,'FE')],
]
NNN_GROUPS_RAW = [
    # rank-pool 0: env1  AE,EC,CA,DB,BF,FD
    [(1,'AE'),(1,'EC'),(1,'CA'),(1,'DB'),(1,'BF'),(1,'FD')],
    # rank-pool 1: env2  CA,AE,EC,BF,FD,DB
    [(2,'CA'),(2,'AE'),(2,'EC'),(2,'BF'),(2,'FD'),(2,'DB')],
    # rank-pool 2: env3  EC,CA,AE,FD,DB,BF
    [(3,'EC'),(3,'CA'),(3,'AE'),(3,'FD'),(3,'DB'),(3,'BF')],
]

# Sublattice sign convention for Néel order parameter
SITES_ACE = ['A', 'C', 'E']   # keep sign
SITES_BDF = ['B', 'D', 'F']   # flip sign

# ──────────────────────────────────────────────────────────────────────────────
# REGEXES
# ──────────────────────────────────────────────────────────────────────────────
RE_ENERGY     = re.compile(r'^energy_per_site\s*=\s*([+-]?[\d.e+-]+)', re.MULTILINE)
RE_CORR       = re.compile(r'^corr_env(\d+)_([A-F]{2})\s*=\s*([+-]?[\d.e+-]+)', re.MULTILINE)
RE_MAG        = re.compile(
    r'^mag_env(\d+)_([A-F])\s+Sx=([+-]?[\d.e+-]+)\s+Sy=([+-]?[\d.e+-]+)\s+Sz=([+-]?[\d.e+-]+)',
    re.MULTILINE
)
RE_PLAIN_CHI  = re.compile(r'^D_(\d+)_chi_(\d+)_energy_magnetization_correlation\.txt$')
RE_COMPLETE_D = re.compile(r'D=(\d+) complete')
RE_OUT_DIR    = re.compile(r'Output dir\s*:\s*(\S+)')
RE_J2         = re.compile(r'J2_0p(\d+)')

# ──────────────────────────────────────────────────────────────────────────────
# 1.  build_complete_D_map
# ──────────────────────────────────────────────────────────────────────────────
def build_complete_D_map(data_dir):
    """
    Scan all job-*.out in data_dir.
    Returns: dict { folder_basename : sorted list of completed D }
    Job Output dir lines may be word-wrapped; we strip non-path characters.
    """
    d_map = {}
    for jf in sorted(glob.glob(os.path.join(data_dir, 'job-*.out'))):
        txt = open(jf).read()
        m_dir = RE_OUT_DIR.search(txt)
        if not m_dir:
            continue
        raw      = m_dir.group(1).strip()
        basename = os.path.basename(raw)
        # Strip any line-wrap artefact (non-path chars that crept in)
        basename = re.sub(r'[^a-zA-Z0-9_\-].*$', '', basename)
        ds = sorted(int(m.group(1)) for m in RE_COMPLETE_D.finditer(txt))
        if ds:
            prev   = d_map.get(basename, [])
            merged = sorted(set(prev) | set(ds))
            d_map[basename] = merged
    return d_map


# ──────────────────────────────────────────────────────────────────────────────
# 2.  parse_plain_file
# ──────────────────────────────────────────────────────────────────────────────
def parse_plain_file(fpath):
    """
    Returns dict:
        energy_per_site : float
        corr : { (env_int, pair_str) : float }
        mag  : { (env_int, site_str) : np.ndarray([Sx, Sy, Sz]) }
    """
    txt = open(fpath).read()
    energy = float(RE_ENERGY.search(txt).group(1))
    corr   = {}
    for m in RE_CORR.finditer(txt):
        corr[(int(m.group(1)), m.group(2))] = float(m.group(3))
    mag = {}
    for m in RE_MAG.finditer(txt):
        mag[(int(m.group(1)), m.group(2))] = np.array([
            float(m.group(3)), float(m.group(4)), float(m.group(5))
        ])
    return {'energy_per_site': energy, 'corr': corr, 'mag': mag}


# ──────────────────────────────────────────────────────────────────────────────
# 3.  load_folder_data
# ──────────────────────────────────────────────────────────────────────────────
def load_folder_data(folder_path, completed_Ds):
    """
    For each D in completed_Ds, find the plain (no '+2D') text file with the
    highest chi_base.  Returns: dict { D : parsed_data_dict }
    """
    chi_map = {}  # D → list of (chi_base, filepath)
    for fname in os.listdir(folder_path):
        m = RE_PLAIN_CHI.match(fname)
        if not m:
            continue
        d, chi = int(m.group(1)), int(m.group(2))
        chi_map.setdefault(d, []).append((chi, os.path.join(folder_path, fname)))

    result = {}
    for D in completed_Ds:
        if D not in chi_map:
            continue
        best_chi, best_path = max(chi_map[D], key=lambda x: x[0])
        data = parse_plain_file(best_path)
        data['chi_base'] = best_chi
        result[D] = data
    return result


# ──────────────────────────────────────────────────────────────────────────────
# 4.  compute_energy_extrap
# ──────────────────────────────────────────────────────────────────────────────
def _exp_model(D, E0, c, Dchar):
    return E0 + c * np.exp(-np.asarray(D, dtype=float) / Dchar)


def compute_energy_extrap(Ds, eps):
    """
    Ds  : list of ints (completed D values, already sorted ascending)
    eps : list of floats (energy_per_site at each D)

    Returns dict:
        E_horiz   : E(D_last)
        E_lin2    : y-intercept of linear fit through last 2 points in 1/D space
        E_exp_all : E_0 from exp fit of all Ds (None if fit failed)
        E_0       : best infinite-D estimate  (E_exp_all if in bracket, else E_lin2)
        exp_popt  : (E0, c, Dchar) array if exp fit succeeded, else None
    """
    Ds_f  = np.array(Ds, dtype=float)
    eps_f = np.array(eps, dtype=float)
    E_horiz = float(eps_f[-1])

    # Linear fit on last 2 points in 1/D space
    if len(Ds_f) >= 2:
        inv2 = 1.0 / Ds_f[-2:]
        y2   = eps_f[-2:]
        coeffs = np.polyfit(inv2, y2, 1)
        E_lin2 = float(np.polyval(coeffs, 0.0))
    else:
        E_lin2 = E_horiz

    # Exponential fit on ALL Ds:  E(D) = E0 + c·exp(-D/Dchar)
    E_exp_all = None
    exp_popt  = None
    if len(Ds_f) >= 2:
        x_all = Ds_f
        y_all = eps_f
        # Initial guesses
        delta    = abs(y_all[-1] - y_all[0])
        E0_g     = y_all[-1] - delta * 0.1  # slightly below last point
        c_g      = y_all[0] - E0_g          # sign depends on monotonicity
        Dchar_g  = 4.0
        try:
            popt, _ = curve_fit(
                _exp_model, x_all, y_all,
                p0=[E0_g, c_g, Dchar_g],
                bounds=([-np.inf, -np.inf, 0.1], [np.inf, np.inf, 100.0]),
                maxfev=20000
            )
            E_exp_all = float(popt[0])
            exp_popt = popt.tolist()
        except Exception:
            pass

    # Use exp fit only when it falls inside the bracket [min(E_lin2,E_horiz), max(E_lin2,E_horiz)].
    # An exp fit that overshoots both brackets is unreliable; fall back to the bracket mean.
    lo, hi = min(E_horiz, E_lin2), max(E_horiz, E_lin2)
    if E_exp_all is not None and lo <= E_exp_all <= hi:
        E_0 = E_exp_all
    elif E_exp_all is not None:          # outside bracket → use mean of the two bounds
        E_0 = (E_horiz + E_lin2) / 2.0
    else:
        E_0 = E_lin2
    return {
        'E_horiz' : E_horiz,
        'E_lin2'  : E_lin2,
        'E_exp_all': E_exp_all,
        'E_0'     : E_0,
        'exp_popt': exp_popt,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 5.  compute_bond_groups
# ──────────────────────────────────────────────────────────────────────────────
def compute_bond_groups(D_data, groups_raw):
    """
    D_data    : dict { D : parsed_data }
    groups_raw: list of 3 group specs (NN or NNN)

    Returns: dict { D : { 'means': [m0,m1,m2], 'stds': [s0,s1,s2], 'ranks': [r0,r1,r2] } }
    ranks[i] = 1 (smallest value) … 3 (largest value) within this D.
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
        # Rank ascending (rank 1 = smallest)
        order = np.argsort(means)   # indices that sort ascending
        ranks = [0, 0, 0]
        for rank_val, idx in enumerate(order, start=1):
            ranks[idx] = rank_val
        result[D] = {'means': means, 'stds': stds, 'sterrs': sterrs, 'ranks': ranks}
    return result


# ──────────────────────────────────────────────────────────────────────────────
# 6.  compute_order_param
# ──────────────────────────────────────────────────────────────────────────────
def compute_order_param(D_data):
    """
    Néel order parameter: 18 site-magnetization vectors.
    ACE sublattice → raw vector; BDF sublattice → sign-flipped.
    m   = || mean(18 vectors) ||
    err = sqrt( 1/18 · Σ_i || v_i − mean ||² )

    Returns: dict { D : {'m': float, 'err': float} }
    """
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
        if len(vecs) == 0:
            continue
        vecs     = np.array(vecs)            # (18, 3)
        mean_vec = vecs.mean(axis=0)
        m        = float(np.linalg.norm(mean_vec))
        diffs    = vecs - mean_vec[None, :]
        err      = float(np.sqrt(np.mean(np.sum(diffs**2, axis=1))))
        result[D] = {'m': m, 'err': err}
    return result


# ──────────────────────────────────────────────────────────────────────────────
# 7.  collect_all_data  (main data aggregation)
# ──────────────────────────────────────────────────────────────────────────────
def collect_all_data(data_dir):
    """
    Returns: dict { folder_basename : result_dict }
    result_dict keys:
        ansatz, j2, Ds, energy_per_site, chi_bases,
        extrap, nn_groups, nnn_groups, order_param
    """
    d_map = build_complete_D_map(data_dir)
    print(f"Completed-D map loaded for {len(d_map)} folders.")

    all_results = {}
    for fname in sorted(os.listdir(data_dir)):
        folder_path = os.path.join(data_dir, fname)
        if not os.path.isdir(folder_path):
            continue

        # Determine ansatz
        if fname.startswith('6tensors'):
            ansatz = '6tensors'
        elif fname.startswith('neel_symmetrized'):
            ansatz = 'neel_sym'
        else:
            continue

        # Extract J2
        m_j2 = RE_J2.search(fname)
        if not m_j2:
            continue
        j2 = float('0.' + m_j2.group(1))

        # Resolve completed Ds
        if fname in d_map:
            completed_Ds = d_map[fname]
        else:
            # Try prefix match (line-wrap may have truncated basename in log)
            matched = next(
                (key for key in d_map if fname.startswith(key) or key.startswith(fname)),
                None
            )
            if matched is None:
                print(f"  SKIP (no job log match): {fname}")
                continue
            completed_Ds = d_map[matched]

        if not completed_Ds:
            print(f"  SKIP (0 completed Ds): {fname}")
            continue

        D_data = load_folder_data(folder_path, completed_Ds)
        if not D_data:
            print(f"  SKIP (no data files found): {fname}")
            continue

        Ds_sorted  = sorted(D_data.keys())
        eps_sorted = [D_data[D]['energy_per_site'] for D in Ds_sorted]
        extrap     = compute_energy_extrap(Ds_sorted, eps_sorted)
        nn_groups  = compute_bond_groups(D_data, NN_GROUPS_RAW)
        nnn_groups = compute_bond_groups(D_data, NNN_GROUPS_RAW)
        order_p    = compute_order_param(D_data)

        all_results[fname] = {
            'ansatz'         : ansatz,
            'j2'             : j2,
            'Ds'             : Ds_sorted,
            'energy_per_site': eps_sorted,
            'chi_bases'      : [D_data[D]['chi_base'] for D in Ds_sorted],
            'extrap'         : extrap,
            'nn_groups'      : {D: v for D, v in nn_groups.items()},
            'nnn_groups'     : {D: v for D, v in nnn_groups.items()},
            'order_param'    : {D: v for D, v in order_p.items()},
        }
        print(f"  Loaded {fname}: J2={j2:.2f} ansatz={ansatz} Ds={Ds_sorted}")

    return all_results


# ──────────────────────────────────────────────────────────────────────────────
# 8.  save_json
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
ANSATZ_STYLE = {
    '6tensors': dict(color='tab:blue',   linestyle='-',  label='6-tensor'),
    'neel_sym' : dict(color='tab:orange', linestyle='--', label='Néel-sym'),
}

def _j2_colormap(j2_list):
    """Return dict j2 → color from 'plasma' colormap."""
    j2s = sorted(set(j2_list))
    if len(j2s) == 1:
        return {j2s[0]: plt.get_cmap('plasma')(0.5)}
    cmap = plt.get_cmap('plasma')
    return {j2: cmap(i / (len(j2s) - 1)) for i, j2 in enumerate(j2s)}


def _D_colormap(D_list):
    """Return dict D → color from 'viridis' colormap."""
    Ds = sorted(set(D_list))
    if len(Ds) == 1:
        return {Ds[0]: plt.get_cmap('viridis')(0.5)}
    cmap = plt.get_cmap('viridis')
    return {D: cmap(i / (len(Ds) - 1)) for i, D in enumerate(Ds)}


def _D_colormap_ansatz(D_list, ansatz):
    """Ansatz-specific D colormap: small D → light, large D → dark.
    6tensors → Blues, neel_sym → Oranges.
    """
    Ds    = sorted(set(D_list))
    cname = 'Blues' if ansatz == '6tensors' else 'Oranges'
    cmap  = plt.get_cmap(cname)
    if len(Ds) == 1:
        return {Ds[0]: cmap(0.65)}
    # remap to [0.30, 0.90] to avoid near-white at the low end
    return {D: cmap(0.30 + 0.60 * i / (len(Ds) - 1)) for i, D in enumerate(Ds)}


# Rank → colormap family: rank 1 (most negative) → Reds, rank 2 → Greens, rank 3 → Blues
RANK_CMAPS = {1: 'Reds', 2: 'Greens', 3: 'Blues'}
RANK_LABELS = {1: 'rank 1 (most neg)', 2: 'rank 2 (mid)', 3: 'rank 3 (least neg)'}


def _D_colormap_rank(D_list, rank):
    """Return dict D → color within the rank's color family.
    Small D → light, large D → dark.
    """
    Ds   = sorted(set(D_list))
    cmap = plt.get_cmap(RANK_CMAPS[rank])
    if len(Ds) == 1:
        return {Ds[0]: cmap(0.65)}
    return {D: cmap(0.30 + 0.60 * i / (len(Ds) - 1)) for i, D in enumerate(Ds)}


def _save(fig, fpath):
    fig.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {os.path.relpath(fpath)}")


# ──────────────────────────────────────────────────────────────────────────────
# 9.  Plot: Energy vs 1/D  (per ansatz subfolder, one figure per J2)
# ──────────────────────────────────────────────────────────────────────────────
def plot_energy_vs_invD(all_results, out_dir):
    all_ansatz = ['6tensors', 'neel_sym']
    
    for ansatz in all_ansatz:
        # Create subfolder for this ansatz
        ansatz_subdir = os.path.join(out_dir, f'energy_vs_invD_{ansatz}')
        os.makedirs(ansatz_subdir, exist_ok=True)
        
        # Collect all (j2, data) for this ansatz
        sub = {k: v for k, v in all_results.items() if v['ansatz'] == ansatz}
        if not sub:
            continue
        
        all_j2 = sorted(set(v['j2'] for v in sub.values()))
        col = ANSATZ_STYLE[ansatz]['color']
        lbl = ANSATZ_STYLE[ansatz]['label']
        
        # Create one figure per J2
        for j2 in all_j2:
            j2_str = _j2_fname(j2)
            fig, ax = plt.subplots(figsize=(8, 5))
            
            # Find data for this (ansatz, J2)
            matches = [v for v in sub.values() if v['j2'] == j2]
            if not matches:
                plt.close(fig)
                continue
            
            v   = matches[0]
            Ds  = np.array(v['Ds'], dtype=float)
            eps = np.array(v['energy_per_site'])
            ex  = v['extrap']
            inv = 1.0 / Ds

            # Data points
            ax.plot(inv, eps, 'o-', color=col, ms=5, lw=1.4, label=lbl)

            # 1. Horizontal line (dotted) at E_horiz
            x_max = max(inv) * 1.15
            ax.plot([0, x_max], [ex['E_horiz'], ex['E_horiz']],
                    color=col, lw=0.8, ls=':', alpha=0.7)

            # 2. Linear fit on last 2 points → y-intercept (dashed)
            if len(Ds) >= 2:
                inv2   = 1.0 / Ds[-2:]
                y2     = eps[-2:]
                coeffs = np.polyfit(inv2, y2, 1)
                x_lin  = np.array([0.0, inv2[0]])
                ax.plot(x_lin, np.polyval(coeffs, x_lin),
                        color=col, lw=1.0, ls='--', alpha=0.8)

            # 3. Exponential fit on all Ds (dash-dot)
            if ex['exp_popt'] is not None:
                E0, c, Dchar = ex['exp_popt']
                # Linspace in 1/D space → smooth curve on the 1/D axis
                inv_max   = 1.0 / Ds[0] * 1.05
                inv_dense = np.linspace(1e-5, inv_max, 800)
                D_dense   = 1.0 / inv_dense
                y_dense   = _exp_model(D_dense, E0, c, Dchar)
                ax.plot(inv_dense, y_dense, color=col, lw=1.3, ls='-.', alpha=0.8)

            # Marker at E_0, 1/D=0
            ax.plot(0, ex['E_0'], marker='*', color=col, ms=11, zorder=6,
                    markeredgewidth=0.4, markeredgecolor='k')

            ax.set_xlim(left=-0.005)
            ax.set_xlabel('1/D', fontsize=12)
            ax.set_ylabel('Energy per site', fontsize=12)
            ax.set_title(
                f'{_j2_label(j2)}  –  Energy vs 1/D  ({lbl})\n'
                '(dotted=horizontal, dashed=linear last-2, dash-dot=exp all-Ds, ★=E₀)',
                fontsize=10
            )
            ax.legend(fontsize=9)
            plt.tight_layout()
            _save(fig, os.path.join(ansatz_subdir, f'energy_vs_invD_J2_{j2_str}.pdf'))


# ──────────────────────────────────────────────────────────────────────────────
# 10. Plot: E_0 vs J2  (both ansätze on one figure, asymmetric error bars)
# ──────────────────────────────────────────────────────────────────────────────
def plot_energy_vs_J2(all_results, out_dir):
    ZOOM = (0.26, 0.28)
    markers = {'6tensors': 'o', 'neel_sym': 's'}
    style   = {'6tensors': dict(color='tab:blue'),
               'neel_sym' : dict(color='tab:orange')}

    # Layout: top = both ansätze compared, bottom = zoom of 6tensors only
    fig = plt.figure(figsize=(8, 8))
    gs  = gridspec.GridSpec(2, 1, figure=fig,
                            height_ratios=[3, 1.8], hspace=0.45)
    ax_main = fig.add_subplot(gs[0])
    ax_zoom = fig.add_subplot(gs[1])

    for ansatz in ['6tensors', 'neel_sym']:
        sub = {k: v for k, v in all_results.items() if v['ansatz'] == ansatz}
        if not sub:
            continue
        rows = sorted(sub.values(), key=lambda v: v['j2'])
        j2s  = [r['j2'] for r in rows]
        E0s  = [r['extrap']['E_0']     for r in rows]
        Ehor = [r['extrap']['E_horiz'] for r in rows]
        Elin = [r['extrap']['E_lin2']  for r in rows]

        yerr_up   = np.clip(np.array(Ehor) - np.array(E0s),  0, None)
        yerr_down = np.clip(np.array(E0s)  - np.array(Elin), 0, None)

        ax_main.errorbar(
            j2s, E0s,
            yerr=[yerr_down, yerr_up],
            fmt=markers[ansatz]+'-',
            capsize=4,
            label=ANSATZ_STYLE[ansatz]['label'],
            **style[ansatz]
        )

    ax_main.set_xlabel('J2', fontsize=11)
    ax_main.set_ylabel('E₀  (energy per site)', fontsize=11)
    ax_main.set_title('Both ansätze  (error bar up=last D, down=linear-2pts)', fontsize=9)
    ax_main.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.5g'))
    ax_main.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.5g'))
    ax_main.legend(fontsize=9)

    # Zoom panel: only 6tensors data in [0.26, 0.28]
    sub_6t = {k: v for k, v in all_results.items() if v['ansatz'] == '6tensors'}
    if sub_6t:
        rows_z = sorted(sub_6t.values(), key=lambda v: v['j2'])
        mask_z = [(ZOOM[0] <= r['j2'] <= ZOOM[1]) for r in rows_z]
        j2s_z  = [r['j2']          for r, m in zip(rows_z, mask_z) if m]
        E0s_z  = [r['extrap']['E_0']     for r, m in zip(rows_z, mask_z) if m]
        Ehor_z = [r['extrap']['E_horiz'] for r, m in zip(rows_z, mask_z) if m]
        Elin_z = [r['extrap']['E_lin2']  for r, m in zip(rows_z, mask_z) if m]
        if j2s_z:
            eu = np.clip(np.array(Ehor_z) - np.array(E0s_z), 0, None)
            ed = np.clip(np.array(E0s_z)  - np.array(Elin_z), 0, None)
            ax_zoom.errorbar(j2s_z, E0s_z, yerr=[ed, eu],
                             fmt='o-', capsize=4, color='tab:blue',
                             label=ANSATZ_STYLE['6tensors']['label'])
    ax_zoom.set_xlim(ZOOM)
    ax_zoom.set_xlabel('J2', fontsize=10)
    ax_zoom.set_ylabel('E₀  (energy per site)', fontsize=10)
    ax_zoom.set_title(f'Zoom: 6-tensor  J2 ∈ [{ZOOM[0]}, {ZOOM[1]}]', fontsize=9)
    ax_zoom.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.5g'))
    ax_zoom.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.5g'))
    ax_zoom.legend(fontsize=9)

    fig.suptitle('Extrapolated energy E₀ vs J2', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, os.path.join(out_dir, 'energy_E0_vs_J2.pdf'))


# ──────────────────────────────────────────────────────────────────────────────
# 11. Plot: Bond correlations vs 1/D  (NN and NNN, per-rank figure)
# ──────────────────────────────────────────────────────────────────────────────
def _rank_label(bond_type, rank):
    return f'{bond_type} group rank {rank}'


def plot_bonds_vs_invD(all_results, out_dir, bond_key, bond_type_label):
    """
    bond_key : 'nn_groups' or 'nnn_groups'
    Creates 3 figures (one per rank) with 2 panels (one per ansatz).
    Each panel shows one curve per J2.
    """
    for target_rank in [1, 2, 3]:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

        for ax, ansatz in zip(axes, ['6tensors', 'neel_sym']):
            sub = {k: v for k, v in all_results.items() if v['ansatz'] == ansatz}
            if not sub:
                ax.set_visible(False)
                continue

            all_j2 = [v['j2'] for v in sub.values()]
            j2_col  = _j2_colormap(all_j2)

            for fname in sorted(sub, key=lambda k: sub[k]['j2']):
                v   = sub[fname]
                j2  = v['j2']
                col = j2_col[j2]
                grp_data = v[bond_key]

                xs, ys, yes = [], [], []
                for D, gd in sorted(grp_data.items()):
                    ranks  = gd['ranks']
                    means  = gd['means']
                    sterrs = gd['sterrs']
                    # Find which group index has target_rank at this D
                    gi = next((i for i, r in enumerate(ranks) if r == target_rank), None)
                    if gi is None:
                        continue
                    xs.append(1.0 / D)
                    ys.append(means[gi])
                    yes.append(sterrs[gi])

                if xs:
                    ax.errorbar(xs, ys, yerr=yes, fmt='o-', color=col, ms=5, lw=1.3,
                                capsize=3, elinewidth=0.8, label=_j2_label(j2))

            ax.set_xlabel('1/D', fontsize=11)
            ax.set_ylabel(f'⟨Sᵢ·Sⱼ⟩', fontsize=11)
            ax.set_title(f'{ANSATZ_STYLE[ansatz]["label"]}', fontsize=10)
            ax.legend(fontsize=7, ncol=2)

        fig.suptitle(f'{bond_type_label}  –  rank {target_rank} group  vs  1/D', fontsize=12)
        plt.tight_layout()
        fname_out = f'{bond_key.replace("_groups","")}_rank{target_rank}_vs_invD.pdf'
        _save(fig, os.path.join(out_dir, fname_out))


# ──────────────────────────────────────────────────────────────────────────────
# 12. Plot: Bond correlations vs J2  (one figure per bond type, 2 panels, RGB per rank)
# ──────────────────────────────────────────────────────────────────────────────
def plot_bonds_vs_J2(all_results, out_dir, bond_key, bond_type_label):
    """
    One figure per bond type, 2 panels (6tensors left, neel_sym right) + zoom below 6tensors + legend row.
    All 3 ranks shown together per panel: Reds=rank1, Greens=rank2, Blues=rank3.
    Markers only. D controls darkness within each rank's color family.
    """
    ZOOM = (0.26, 0.28)

    fig = plt.figure(figsize=(13, 9.5))
    gs  = gridspec.GridSpec(3, 2, figure=fig,
                            height_ratios=[3, 1.8, 0.6], hspace=0.5, wspace=0.25)
    ax_6t   = fig.add_subplot(gs[0, 0])
    ax_neel = fig.add_subplot(gs[0, 1])
    ax_zoom = fig.add_subplot(gs[1, 0])
    ax_neel.sharey(ax_6t)
    ax_dummy = fig.add_subplot(gs[1, 1])
    ax_dummy.set_visible(False)
    ax_legend = fig.add_subplot(gs[2, :])
    ax_legend.axis('off')

    ax_map = {'6tensors': ax_6t, 'neel_sym': ax_neel}

    for ansatz in ['6tensors', 'neel_sym']:
        ax = ax_map[ansatz]
        sub = {k: v for k, v in all_results.items() if v['ansatz'] == ansatz}
        if not sub:
            ax.set_visible(False)
            continue

        # Collect { rank → { D → [(j2, val, sterr), ...] } }
        data_by_rank_D = {rank: {} for rank in [1, 2, 3]}
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

        # All Ds across all ranks (for shared colormap range within each rank)
        all_Ds = sorted(set(D for rk in data_by_rank_D.values() for D in rk))

        for rank in [1, 2, 3]:
            D_col = _D_colormap_rank(all_Ds, rank)
            for D in sorted(data_by_rank_D[rank].keys()):
                pts  = sorted(data_by_rank_D[rank][D])
                j2s  = [p[0] for p in pts]
                vals = [p[1] for p in pts]
                errs = [p[2] for p in pts]
                alpha = min(1.0, max(0.2, D / 10.0))
                ax.errorbar(j2s, vals, yerr=errs, fmt='o', color=D_col[D], ms=6,
                            alpha=alpha, capsize=3, elinewidth=0.8, lw=1.1,
                            label=f'{RANK_LABELS[rank]} D={D}')

        ax.set_xlabel('J2', fontsize=11)
        ax.set_ylabel('⟨Sᵢ·Sⱼ⟩', fontsize=11)
        ax.set_title(
            f'{ANSATZ_STYLE[ansatz]["label"]}\n'
            '(R=rank1, G=rank2, B=rank3;  bigger D = darker & more opaque)',
            fontsize=9
        )
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.5g'))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.5g'))


    # Zoom panel: re-plot from ax_6t into ax_zoom
    _fill_zoom_ax(ax_zoom, ax_6t, xlim=ZOOM)

    # Collect legend from ax_6t and display in dedicated row
    handles, labels = ax_6t.get_legend_handles_labels()
    if handles:
        ax_legend.legend(handles, labels, loc='center', ncol=6, fontsize=6,
                        frameon=True, fancybox=True, shadow=False)

    fig.suptitle(f'{bond_type_label}  vs  J2', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fname_out = f'{bond_key.replace("_groups","")}_vs_J2.pdf'
    _save(fig, os.path.join(out_dir, fname_out))


# ──────────────────────────────────────────────────────────────────────────────
# 13. Plot: Néel order parameter m vs 1/D  (2 panels: 6tensors left, neel_sym right)
# ──────────────────────────────────────────────────────────────────────────────
def plot_order_param_vs_invD(all_results, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for ax, ansatz in zip(axes, ['6tensors', 'neel_sym']):
        sub = {k: v for k, v in all_results.items() if v['ansatz'] == ansatz}
        if not sub:
            ax.set_visible(False)
            continue
        all_j2 = [v['j2'] for v in sub.values()]
        j2_col  = _j2_colormap(all_j2)
        style   = ANSATZ_STYLE[ansatz]

        for fname in sorted(sub, key=lambda k: sub[k]['j2']):
            v  = sub[fname]
            j2 = v['j2']
            op = v['order_param']
            Ds_op = sorted(op.keys())
            if not Ds_op:
                continue
            inv  = [1.0 / D for D in Ds_op]
            ms   = [op[D]['m']   for D in Ds_op]
            errs = [op[D]['err'] for D in Ds_op]
            col  = j2_col[j2]
            ls   = style['linestyle']
            ax.errorbar(inv, ms, yerr=errs,
                        fmt='o'+ls, color=col, ms=4, lw=1.2, capsize=3,
                        label=_j2_label(j2))

        ax.set_xlabel('1/D', fontsize=12)
        ax.set_ylabel('Néel order m', fontsize=12)
        ax.set_title(f'{ANSATZ_STYLE[ansatz]["label"]}', fontsize=10)
        ax.legend(fontsize=7, ncol=2)

    fig.suptitle('Néel order parameter  m  vs  1/D', fontsize=12)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, 'order_param_vs_invD.pdf'))


# ──────────────────────────────────────────────────────────────────────────────
# 14. Plot: Néel order parameter m vs J2  (2 panels: 6tensors left, neel_sym right)
# ──────────────────────────────────────────────────────────────────────────────
def plot_order_param_vs_J2(all_results, out_dir):
    ZOOM = (0.26, 0.28)

    fig = plt.figure(figsize=(13, 9.5))
    gs  = gridspec.GridSpec(3, 2, figure=fig,
                            height_ratios=[3, 1.8, 0.6], hspace=0.5, wspace=0.25)
    ax_6t   = fig.add_subplot(gs[0, 0])
    ax_neel = fig.add_subplot(gs[0, 1])
    ax_zoom = fig.add_subplot(gs[1, 0])
    ax_neel.sharey(ax_6t)
    ax_dummy = fig.add_subplot(gs[1, 1])
    ax_dummy.set_visible(False)
    ax_legend = fig.add_subplot(gs[2, :])
    ax_legend.axis('off')

    ax_map = {'6tensors': ax_6t, 'neel_sym': ax_neel}

    for ansatz in ['6tensors', 'neel_sym']:
        ax = ax_map[ansatz]
        sub = {k: v for k, v in all_results.items() if v['ansatz'] == ansatz}
        if not sub:
            ax.set_visible(False)
            continue

        # Collect D → list of (j2, m, err)
        data_by_D = {}
        for v in sub.values():
            j2 = v['j2']
            for D, op in v['order_param'].items():
                data_by_D.setdefault(D, []).append((j2, op['m'], op['err']))

        all_Ds = sorted(data_by_D.keys())
        D_col  = _D_colormap_ansatz(all_Ds, ansatz)

        for D in all_Ds:
            pts  = sorted(data_by_D[D])
            j2s  = [p[0] for p in pts]
            ms   = [p[1] for p in pts]
            errs = [p[2] for p in pts]
            alpha = min(1.0, max(0.2, D / 10.0))
            ax.errorbar(j2s, ms, yerr=errs,
                        fmt='o', color=D_col[D], ms=6, lw=1.1,
                        elinewidth=1.0, alpha=alpha, capsize=3, label=f'D={D}')

        ax.set_xlabel('J2', fontsize=11)
        ax.set_ylabel('Néel order m', fontsize=11)
        ax.set_title(
            f'{ANSATZ_STYLE[ansatz]["label"]}  (bigger D = more opaque & darker)',
            fontsize=10
        )
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.5g'))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.5g'))

    # Zoom panel: re-plot from ax_6t into ax_zoom
    _fill_zoom_ax(ax_zoom, ax_6t, xlim=ZOOM)

    # Collect legend from ax_6t and display in dedicated row
    handles, labels = ax_6t.get_legend_handles_labels()
    if handles:
        ax_legend.legend(handles, labels, loc='center', ncol=4, fontsize=7,
                        frameon=True, fancybox=True, shadow=False)

    fig.suptitle('Néel order parameter  m  vs  J2', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, os.path.join(out_dir, 'order_param_vs_J2.pdf'))


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print('=== 0414core comprehensive analysis ===')
    print(f'Data dir : {DATA_DIR}')
    print(f'Output   : {OUT_DIR}')
    print()

    all_results = collect_all_data(DATA_DIR)
    print(f'\nLoaded {len(all_results)} folder(s).\n')

    # Save raw analysis values to disk
    save_json(all_results, OUT_DIR)

    # ── Figures ───────────────────────────────────────────────────────────────
    print('\n--- Energy vs 1/D ---')
    plot_energy_vs_invD(all_results, OUT_DIR)

    print('\n--- Energy E₀ vs J2 ---')
    plot_energy_vs_J2(all_results, OUT_DIR)

    print('\n--- NN bond correlations vs 1/D  (3 ranks) ---')
    plot_bonds_vs_invD(all_results, OUT_DIR, 'nn_groups',  'NN bond ⟨Sᵢ·Sⱼ⟩')

    print('\n--- NN bond correlations vs J2   (3 ranks) ---')
    plot_bonds_vs_J2(all_results, OUT_DIR, 'nn_groups',  'NN bond ⟨Sᵢ·Sⱼ⟩')

    print('\n--- NNN bond correlations vs 1/D (3 ranks) ---')
    plot_bonds_vs_invD(all_results, OUT_DIR, 'nnn_groups', 'NNN bond ⟨Sᵢ·Sⱼ⟩')

    print('\n--- NNN bond correlations vs J2  (3 ranks) ---')
    plot_bonds_vs_J2(all_results, OUT_DIR, 'nnn_groups', 'NNN bond ⟨Sᵢ·Sⱼ⟩')

    print('\n--- Néel order parameter vs 1/D ---')
    plot_order_param_vs_invD(all_results, OUT_DIR)

    print('\n--- Néel order parameter vs J2 ---')
    plot_order_param_vs_J2(all_results, OUT_DIR)

    print('\nDone. All figures written to:', OUT_DIR)


if __name__ == '__main__':
    main()
