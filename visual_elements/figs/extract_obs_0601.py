#!/usr/bin/env python3
"""
extract_obs_0601.py
Extract raw observables from 0507core data into CSV tables.

Data source  : /home/chye/6ADctmrg/data/0507core/D45678
               Same folder discovery and file selection as plot_analysis_0507D456v2.py:
                 - auto-discover folders named  <ansatz>__J2_<j2str>_<dates>
                 - D_ALLOWED = {4, 5, 6, 7, 8, 9, 10}
                 - per D: highest plain chi (ignores lookahead files)

Output       : extract_obs_0601/  (next to this script)
               One CSV per (j2, ansatz):
                 J2{jstr}_{ansatz}_obs.csv
               Columns: D, chi_used, energy_per_site,
                        m_neel, corr_rank1, corr_rank2, corr_rank3, delta_nn

Magnetization (fixed vs 0427):
  Staggered magnetization is a 2D vector in the (Sx, Sz) plane:
    stag_x = 0.5 * (mean_Sx_ACE - mean_Sx_BDF)
    stag_z = 0.5 * (mean_Sz_ACE - mean_Sz_BDF)
    m_neel = sqrt(stag_x^2 + stag_z^2)
  (averaged over all 3 environments, matching plot_analysis_0507D456v2.py)

Bond groups / ranks:
  Three NN bond groups of 6 bonds each (see NN_GROUPS_RAW).
  rank 1 = most-negative group mean, rank 3 = least-negative.
  delta_nn = corr_rank3 - corr_rank1  (always >= 0)
"""

import os, re
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────────────────────
DATA_DIR   = '/home/chye/6ADctmrg/data/0507core/D45678'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join('/home/chye/6ADctmrg/data/processed', 'extract_obs_0601')
os.makedirs(OUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# SETTINGS  (mirror plot_analysis_0507D456v2.py)
# ──────────────────────────────────────────────────────────────────────────────
D_ALLOWED = {3, 4, 5, 6, 7, 8, 9, 10}

NN_GROUPS_RAW = [
    [(1,'EB'),(1,'AD'),(1,'CF'),(3,'BE'),(3,'FC'),(3,'DA')],
    [(2,'CB'),(2,'AF'),(2,'ED'),(1,'FA'),(1,'DE'),(1,'BC')],
    [(3,'EF'),(3,'AB'),(3,'CD'),(2,'DC'),(2,'BA'),(2,'FE')],
]

# ──────────────────────────────────────────────────────────────────────────────
# REGEXES
# ──────────────────────────────────────────────────────────────────────────────
RE_ENERGY    = re.compile(r'^energy_per_site\s*=\s*([+-]?[\d.e+-]+)', re.MULTILINE)
RE_TRUNC     = re.compile(r'^trunc_error\s*=\s*([+-]?[\d.e+-]+)', re.MULTILINE)
RE_CORR      = re.compile(r'^corr_env(\d+)_([A-F]{2})\s*=\s*([+-]?[\d.e+-]+)', re.MULTILINE)
RE_MAG       = re.compile(
    r'^mag_env(\d+)_([A-F])\s+Sx=([+-]?[\d.e+-]+)\s+Sy=([+-]?[\d.e+-]+)\s+Sz=([+-]?[\d.e+-]+)',
    re.MULTILINE,
)
RE_PLAIN_CHI = re.compile(r'^D_(\d+)_chi_(\d+)_energy_magnetization_correlation\.txt$')
RE_FOLDER    = re.compile(r'^(.+?)__J2_([0-9p]+)_\d{8}')

# ──────────────────────────────────────────────────────────────────────────────
# FOLDER DISCOVERY  (identical to plot_analysis_0507D456v2.py)
# ──────────────────────────────────────────────────────────────────────────────
def _parse_j2(s):
    return float(s.replace('p', '.'))


def discover_folders(data_dir):
    """j2 (float) → { ansatz_key → folder_path }  (latest timestamp wins)."""
    mapping = {}
    for name in sorted(os.listdir(data_dir)):
        full = os.path.join(data_dir, name)
        if not os.path.isdir(full):
            continue
        m = RE_FOLDER.match(name)
        if not m:
            continue
        ansatz = m.group(1)
        try:
            j2 = round(_parse_j2(m.group(2)), 6)
        except ValueError:
            continue
        mapping[(j2, ansatz)] = full
    result = {}
    for (j2, ansatz), path in mapping.items():
        result.setdefault(j2, {})[ansatz] = path
    return result

# ──────────────────────────────────────────────────────────────────────────────
# FILE PARSING
# ──────────────────────────────────────────────────────────────────────────────
def parse_plain_file(fpath):
    """
    Parse one observation file.  Returns dict with raw corr/mag dicts,
    or None on failure.
    """
    try:
        txt = open(fpath).read()
    except OSError as e:
        print(f"  Warning: cannot read {fpath}: {e}")
        return None

    m = RE_ENERGY.search(txt)
    if not m:
        return None
    energy = float(m.group(1))

    m = RE_TRUNC.search(txt)
    trunc = float(m.group(1)) if m else float('nan')

    corr = {}
    for m in RE_CORR.finditer(txt):
        corr[(int(m.group(1)), m.group(2))] = float(m.group(3))

    mag = {}
    for m in RE_MAG.finditer(txt):
        mag[(int(m.group(1)), m.group(2))] = {
            'Sx': float(m.group(3)),
            'Sy': float(m.group(4)),
            'Sz': float(m.group(5)),
        }

    return {'energy': energy, 'trunc': trunc, 'corr': corr, 'mag': mag}


def load_highest_chi(folder_path):
    """
    For each D in D_ALLOWED: pick plain file with highest chi.
    Returns dict: D -> (chi_used, parsed_data)
    """
    plain_map = {}  # D -> [(chi, path)]
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
        data     = parse_plain_file(fpath)
        if data is not None:
            result[D] = (best_chi, data)
        else:
            print(f"  Warning: failed to parse {fpath}")
    return result

# ──────────────────────────────────────────────────────────────────────────────
# OBSERVABLE COMPUTATION  (identical formulas to plot_analysis_0507D456v2.py)
# ──────────────────────────────────────────────────────────────────────────────
def compute_observables(data):
    """
    Given parsed data dict, return a flat observables dict:
      energy_per_site, trunc_error,
      m_neel  (vector staggered magnetization in Sx-Sz plane),
      corr_rank1/2/3  (NN bond group means, sorted most-neg to least-neg),
      delta_nn  (= corr_rank3 - corr_rank1)
    """
    energy = data['energy']
    trunc  = data['trunc']
    corr   = data['corr']
    mag    = data['mag']

    # ── Staggered magnetization: vector in (Sx, Sz) plane ────────────────────
    # Sublattices ACE vs BDF, averaged over environments 1, 2, 3
    # (follows compute_order_param in plot_analysis_0507D456v2.py exactly)
    sz_ace, sz_bdf = [], []
    sx_ace, sx_bdf = [], []
    for env in [1, 2, 3]:
        for site in 'ACE':
            k = (env, site)
            if k in mag:
                sz_ace.append(mag[k]['Sz'])
                sx_ace.append(mag[k]['Sx'])
        for site in 'BDF':
            k = (env, site)
            if k in mag:
                sz_bdf.append(mag[k]['Sz'])
                sx_bdf.append(mag[k]['Sx'])

    def _mean(lst):
        return float(np.mean(lst)) if lst else float('nan')

    sz_ace_m = _mean(sz_ace); sz_bdf_m = _mean(sz_bdf)
    sx_ace_m = _mean(sx_ace); sx_bdf_m = _mean(sx_bdf)

    if not any(np.isnan(x) for x in [sz_ace_m, sz_bdf_m, sx_ace_m, sx_bdf_m]):
        stag_sz = 0.5 * (sz_ace_m - sz_bdf_m)
        stag_sx = 0.5 * (sx_ace_m - sx_bdf_m)
        m_neel  = float(np.sqrt(stag_sx**2 + stag_sz**2))
    else:
        m_neel = float('nan')

    # ── NN bond group means and ranking ──────────────────────────────────────
    # (follows compute_bond_groups in plot_analysis_0507D456v2.py exactly)
    group_means = []
    for group in NN_GROUPS_RAW:
        vals = [corr[(env, pair)] for (env, pair) in group if (env, pair) in corr]
        group_means.append(float(np.mean(vals)) if vals else float('nan'))

    # Assign ranks: rank 1 = most negative (smallest), rank 3 = least negative
    valid_idx = [i for i, v in enumerate(group_means) if not np.isnan(v)]
    rank_to_mean = {1: float('nan'), 2: float('nan'), 3: float('nan')}
    if valid_idx:
        sorted_idx = sorted(valid_idx, key=lambda i: group_means[i])
        for rank_pos, g_idx in enumerate(sorted_idx):
            rank_to_mean[rank_pos + 1] = group_means[g_idx]

    corr_r1 = rank_to_mean[1]
    corr_r2 = rank_to_mean[2]
    corr_r3 = rank_to_mean[3]
    delta_nn = (corr_r3 - corr_r1) if not (np.isnan(corr_r1) or np.isnan(corr_r3)) else float('nan')

    return {
        'energy_per_site': energy,
        'trunc_error':     trunc,
        'm_neel':          m_neel,
        'corr_rank1':      corr_r1,
        'corr_rank2':      corr_r2,
        'corr_rank3':      corr_r3,
        'delta_nn':        delta_nn,
    }

# ──────────────────────────────────────────────────────────────────────────────
# CSV HELPERS
# ──────────────────────────────────────────────────────────────────────────────
HEADER = 'D,chi_used,energy_per_site,trunc_error,m_neel,corr_rank1,corr_rank2,corr_rank3,delta_nn'


def _fmt(x):
    return 'nan' if (isinstance(x, float) and np.isnan(x)) else f'{x:.10f}'


def _csv_row(D, chi, obs):
    return ','.join([
        str(D),
        str(chi),
        _fmt(obs['energy_per_site']),
        _fmt(obs['trunc_error']),
        _fmt(obs['m_neel']),
        _fmt(obs['corr_rank1']),
        _fmt(obs['corr_rank2']),
        _fmt(obs['corr_rank3']),
        _fmt(obs['delta_nn']),
    ])


def _j2_str(j2):
    s = f'{j2:.10f}'.rstrip('0')
    dot = s.index('.')
    if len(s) - dot - 1 < 2:
        s += '0' * (2 - (len(s) - dot - 1))
    return s.replace('.', 'p')

# ──────────────────────────────────────────────────────────────────────────────
# README
# ──────────────────────────────────────────────────────────────────────────────
def write_readme(out_dir, all_j2_ansatz):
    fpath = os.path.join(out_dir, 'README.txt')
    with open(fpath, 'w') as f:
        f.write('extract_obs_0601  —  Raw observable data for 0507core iPEPS runs\n')
        f.write('=' * 72 + '\n\n')
        f.write('Source: /home/chye/6ADctmrg/data/0507core/D45678\n')
        f.write('        Same folder discovery and file selection as\n')
        f.write('        plot_analysis_0507D456v2.py\n\n')
        f.write('All files are CSV (comma-separated values).\n')
        f.write('Numbers in fixed-point decimal notation (10 decimal places).\n')
        f.write('Import in Mathematica: Import["file.csv", "CSV"]\n\n')
        f.write('FILE NAMING\n')
        f.write('-----------\n')
        f.write('  J2{jstr}_{ansatz}_obs.csv\n')
        f.write('  One file per (J2, ansatz). Each row is the highest-chi\n')
        f.write('  plain observation file for that D.\n\n')
        f.write('COLUMNS\n')
        f.write('-------\n')
        f.write('  D              : iPEPS bond dimension\n')
        f.write('  chi_used       : CTM environment bond dimension actually used\n')
        f.write('  energy_per_site: variational energy per site\n')
        f.write('  trunc_error    : CTMRG truncation error (nan if absent)\n')
        f.write('  m_neel         : Neel order parameter\n')
        f.write('                   = sqrt(stag_x^2 + stag_z^2)  where\n')
        f.write('                   stag_x = 0.5*(mean_Sx_ACE - mean_Sx_BDF)\n')
        f.write('                   stag_z = 0.5*(mean_Sz_ACE - mean_Sz_BDF)\n')
        f.write('                   averaged over environments 1, 2, 3\n')
        f.write('  corr_rank1     : mean of the most-negative NN bond group\n')
        f.write('  corr_rank2     : mean of the mid NN bond group\n')
        f.write('  corr_rank3     : mean of the least-negative NN bond group\n')
        f.write('  delta_nn       : corr_rank3 - corr_rank1  (>= 0)\n\n')
        f.write('NN BOND GROUPS (3 groups of 6 bonds each)\n')
        f.write('-----------------------------------------\n')
        f.write('  Group 0: env1(EB,AD,CF)  env3(BE,FC,DA)\n')
        f.write('  Group 1: env2(CB,AF,ED)  env1(FA,DE,BC)\n')
        f.write('  Group 2: env3(EF,AB,CD)  env2(DC,BA,FE)\n')
        f.write('  Rank assignment: sort group means ascending\n')
        f.write('  rank 1 = smallest (most negative), rank 3 = largest\n\n')
        f.write('FILES GENERATED\n')
        f.write('---------------\n')
        for j2, ansatz in sorted(all_j2_ansatz):
            jstr = _j2_str(j2)
            f.write(f'  J2{jstr}_{ansatz}_obs.csv   (J2={j2:.6g})\n')
    print(f'  Wrote README.txt')

# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    folder_map = discover_folders(DATA_DIR)
    if not folder_map:
        print(f'No folders found in {DATA_DIR}')
        return

    all_j2_ansatz = []

    for j2 in sorted(folder_map.keys()):
        jstr = _j2_str(j2)
        for ansatz in sorted(folder_map[j2].keys()):
            folder_path = folder_map[j2][ansatz]
            print(f'J2={j2:.6g}  {ansatz}')
            D_data = load_highest_chi(folder_path)
            if not D_data:
                print('  (no data files found)')
                continue

            rows = []
            for D in sorted(D_data.keys()):
                chi, raw = D_data[D]
                obs = compute_observables(raw)
                rows.append((D, chi, obs))

            fname = f'J2{jstr}_{ansatz}_obs.csv'
            fout  = os.path.join(OUT_DIR, fname)
            with open(fout, 'w') as f:
                f.write(HEADER + '\n')
                for D, chi, obs in rows:
                    f.write(_csv_row(D, chi, obs) + '\n')
            print(f'  Wrote {fname}  ({len(rows)} D values, D={[r[0] for r in rows]})')
            all_j2_ansatz.append((j2, ansatz))

    write_readme(OUT_DIR, all_j2_ansatz)
    print(f'\nDone.  Files in: {OUT_DIR}')


if __name__ == '__main__':
    main()
