#!/usr/bin/env python3
"""
plot_summary_j2_manual.py
=========================
Manually specify which folder corresponds to each J2 value, then produce a
2-panel summary PDF:
  Panel 1 – Extrapolated energy per site vs J2
  Panel 2 – Extrapolated m_Néel and ΔNN (rank1−rank3) vs J2

Edit only the CONFIG section below; then run:
    python plot_summary_j2_manual.py
"""

import os, re
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.optimize import curve_fit

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG — edit FOLDER_TO_PLOT_J2 with your actual folder names
# ──────────────────────────────────────────────────────────────────────────────
DATA_DIR  = '/home/chye/6ADctmrg/data/0504core'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR   = os.path.join(SCRIPT_DIR, 'summary_j2_manual_0504')
OUT_FILE  = 'summary_vs_J2.pdf'

# Map J2 (float) → sub-folder name under DATA_DIR.
# Leave the dict empty or with placeholders until you have real results.
FOLDER_TO_PLOT_J2: dict[float, str] = {
    # 0.20: '1tensor_C6Ypi__J2_0p20_20260428_xxxxxx',
    # 0.25: '1tensor_C6Ypi__J2_0p25_20260428_yyyyyy',
    # 0.30: '1tensor_C6Ypi__J2_0p30_20260428_zzzzzz',
}

# ──────────────────────────────────────────────────────────────────────────────
# Bond group definitions — same as 0427 analysis
# ──────────────────────────────────────────────────────────────────────────────
NN_GROUPS_RAW = [
    [(1,'EB'),(1,'AD'),(1,'CF'),(3,'BE'),(3,'FC'),(3,'DA')],
    [(2,'CB'),(2,'AF'),(2,'ED'),(1,'FA'),(1,'DE'),(1,'BC')],
    [(3,'EF'),(3,'AB'),(3,'CD'),(2,'DC'),(2,'BA'),(2,'FE')],
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
    """For each D: use highest plain chi, prefer +2D file if available."""
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
    E_exp = None
    if len(Ds_f) >= 3:
        try:
            p0 = [eps_f[-1] - 0.01, 0.01, float(Ds_f[-1])]
            popt, _ = curve_fit(_exp_model, Ds_f, eps_f, p0=p0, maxfev=8000)
            if popt[2] > 0:
                E_exp = float(popt[0])
        except Exception:
            pass
    if E_exp is not None:
        lo = min(E_horiz, E_lin3) - 2 * abs(E_horiz - E_lin3) - 0.3
        hi = max(E_horiz, E_lin3) + 2 * abs(E_horiz - E_lin3) + 0.3
        E_best = E_exp if lo <= E_exp <= hi else E_lin3
    else:
        E_best = E_lin3
    E_err  = (0.5 * abs(E_lin3 - E_exp)
              if (E_exp is not None and E_best == E_exp) else 0.0)
    return E_best, E_err


def compute_mag_extrap(Ds, mneel_list):
    Ds_f = np.array(Ds, dtype=float); m_f = np.array(mneel_list, dtype=float)
    inv  = 1.0 / Ds_f
    valid = ~np.isnan(m_f)
    if valid.sum() < 2:
        return float('nan'), float('nan')
    inv_v = inv[valid]; m_v = m_f[valid]
    n3m   = min(3, len(inv_v))
    m_lin3 = float(np.polyfit(inv_v[-n3m:], m_v[-n3m:], 1)[1])
    n2m    = min(2, len(inv_v))
    m_lin2 = float(np.polyfit(inv_v[-n2m:], m_v[-n2m:], 1)[1])
    three_v = sorted([float(m_v[-1]), m_lin2, m_lin3])
    m_best = max(0.0, three_v[1])
    m_err  = max(0.0, three_v[2]) - m_best
    return m_best, m_err


def compute_order_param(D_data):
    result = {}
    for D in sorted(D_data.keys()):
        mag = D_data[D].get('mag', {})
        sz_ace, sz_bdf = [], []
        for env in [1, 2, 3]:
            for site in ['A', 'C', 'E']:
                if (env, site) in mag:
                    sz_ace.append(mag[(env, site)]['Sz'])
            for site in ['B', 'D', 'F']:
                if (env, site) in mag:
                    sz_bdf.append(mag[(env, site)]['Sz'])
        sz_a = float(np.mean(sz_ace)) if sz_ace else float('nan')
        sz_b = float(np.mean(sz_bdf)) if sz_bdf else float('nan')
        m_neel = (abs(sz_a - sz_b) / 2.0
                  if not any(np.isnan(x) for x in [sz_a, sz_b]) else float('nan'))
        result[D] = m_neel
    return result


def compute_delta_nn(Ds, D_data):
    """ΔNN = rank1_mean − rank3_mean (most-negative minus least-negative),
    averaged over all D. Here we report the value at the largest D."""
    corr = D_data[Ds[-1]]['corr']
    group_means = []
    for group in NN_GROUPS_RAW:
        vals = [corr.get((env, pair), float('nan')) for env, pair in group]
        vals = [v for v in vals if not np.isnan(v)]
        group_means.append(float(np.mean(vals)) if vals else float('nan'))
    sorted_means = sorted(group_means)
    return sorted_means[0] - sorted_means[-1]   # most neg − least neg (≤ 0)

# ──────────────────────────────────────────────────────────────────────────────
# SUMMARY PLOT
# ──────────────────────────────────────────────────────────────────────────────
def plot_summary(all_results, out_dir, out_file):
    """all_results: {j2: {E_best, E_err, m_best, m_err, delta_nn}}"""
    j2_vals = sorted(all_results.keys())
    E_best  = np.array([all_results[j]['E_best'] for j in j2_vals])
    E_err   = np.array([all_results[j]['E_err']  for j in j2_vals])
    m_best  = np.array([all_results[j]['m_best'] for j in j2_vals])
    m_err   = np.array([all_results[j]['m_err']  for j in j2_vals])
    delta   = np.array([all_results[j]['delta_nn'] for j in j2_vals])
    j2_arr  = np.array(j2_vals)

    fig, (ax_e, ax_m) = plt.subplots(2, 1, figsize=(7, 9),
                                      gridspec_kw={'hspace': 0.12})

    col_e = '#1f77b4'; col_m = '#9467bd'; col_d = '#d62728'

    # Panel 1: Energy
    ax_e.errorbar(j2_arr, E_best, yerr=E_err,
                  color=col_e, marker='o', ms=7, lw=1.6, capsize=5, elinewidth=1.2,
                  label='E/site (D→∞ extrap)')
    ax_e.set_ylabel('Extrapolated energy per site', fontsize=11)
    ax_e.set_title('Summary vs J₂  (1/D extrapolation)', fontsize=12)
    ax_e.legend(fontsize=9)
    ax_e.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.3g'))
    ax_e.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.6g'))
    ax_e.set_xticklabels([])

    # Panel 2: m_Néel + ΔNN on twin axes
    lns1 = ax_m.errorbar(j2_arr, m_best, yerr=m_err,
                          color=col_m, marker='o', ms=7, lw=1.6, capsize=5, elinewidth=1.2,
                          label=r'm$_\mathrm{N\acute{e}el}$ (extrap)')
    ax_m.set_ylabel(r'Extrapolated m$_\mathrm{N\acute{e}el}$', fontsize=11, color=col_m)
    ax_m.tick_params(axis='y', labelcolor=col_m)
    ax_m.set_xlabel('J₂', fontsize=11)
    ax_m.set_ylim(bottom=0.0)
    ax_m.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.3g'))
    ax_m.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4g'))

    ax_d = ax_m.twinx()
    lns2, = ax_d.plot(j2_arr, np.abs(delta), 's--', color=col_d, ms=7, lw=1.4,
                       label=r'|ΔNN| at largest D')
    ax_d.set_ylabel('|ΔNN| (rank1 − rank3 bond corr)', fontsize=11, color=col_d)
    ax_d.tick_params(axis='y', labelcolor=col_d)
    ax_d.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4g'))

    lines  = [lns1, lns2]
    labels = [l.get_label() if hasattr(l, 'get_label') else str(l) for l in lines]
    ax_m.legend(lines, labels, fontsize=9, loc='upper right')

    fig.tight_layout()
    fpath = os.path.join(out_dir, out_file)
    fig.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved summary: {fpath}")

# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    if not FOLDER_TO_PLOT_J2:
        print("FOLDER_TO_PLOT_J2 is empty — please edit the CONFIG section.")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    all_results = {}

    for j2, folder_name in sorted(FOLDER_TO_PLOT_J2.items()):
        folder_path = os.path.join(DATA_DIR, folder_name)
        if not os.path.isdir(folder_path):
            print(f"  Folder not found for J2={j2}: {folder_path} — skipping.")
            continue
        print(f"Loading J2={j2:.4f}: {folder_name}")
        D_data = load_folder_data(folder_path)
        if not D_data:
            print(f"  No data — skipping.")
            continue

        Ds   = sorted(D_data.keys())
        eps  = [D_data[D]['energy_per_site'] for D in Ds]
        E_best, E_err = compute_energy_extrap(Ds, eps)

        order_D = compute_order_param(D_data)
        mneel_list = [order_D[D] for D in Ds]
        m_best, m_err = compute_mag_extrap(Ds, mneel_list)

        delta = compute_delta_nn(Ds, D_data)

        all_results[j2] = {
            'E_best':   E_best,
            'E_err':    E_err,
            'm_best':   max(0.0, m_best),
            'm_err':    m_err,
            'delta_nn': delta,
        }
        print(f"  E/site* = {E_best:.6f} ± {E_err:.2e}   "
              f"m* = {max(0.0, m_best):.4f} ± {m_err:.2e}   "
              f"|ΔNN| = {abs(delta):.4f}")

    if len(all_results) < 2:
        print("\nNeed at least 2 J2 points for a summary plot.")
        return

    plot_summary(all_results, OUT_DIR, OUT_FILE)
    print("Done.")


if __name__ == '__main__':
    main()
