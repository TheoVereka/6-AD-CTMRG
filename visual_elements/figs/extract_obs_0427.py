#!/usr/bin/env python3
"""
extract_obs_0427.py
Extract observables from 0420core data into CSV tables.
See extract_obs_0427/README.txt for column definitions.
"""

import os
import re
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────────────────────
DATA_DIR   = '/home/chye/6ADctmrg/data/0420core'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join('/home/chye/6ADctmrg/data', 'extract_obs_0427')
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

# NN bond groups: each group is a list of (env_index, pair_label)
NN_GROUPS_RAW = [
    [(1,'EB'),(1,'AD'),(1,'CF'),(3,'BE'),(3,'FC'),(3,'DA')],
    [(2,'CB'),(2,'AF'),(2,'ED'),(1,'FA'),(1,'DE'),(1,'BC')],
    [(3,'EF'),(3,'AB'),(3,'CD'),(2,'DC'),(2,'BA'),(2,'FE')],
]

# ──────────────────────────────────────────────────────────────────────────────
# REGEXES
# ──────────────────────────────────────────────────────────────────────────────
RE_ENERGY_PS = re.compile(r'^energy_per_site\s*=\s*([+-]?[\d.e+-]+)', re.MULTILINE)
RE_TRUNC     = re.compile(r'^trunc_error\s*=\s*([+-]?[\d.e+-]+)', re.MULTILINE)
RE_CORR      = re.compile(r'^corr_env(\d+)_([A-F]{2})\s*=\s*([+-]?[\d.e+-]+)', re.MULTILINE)
RE_MAG       = re.compile(
    r'^mag_env(\d+)_([A-F])\s+Sx=([+-]?[\d.e+-]+)\s+Sy=([+-]?[\d.e+-]+)\s+Sz=([+-]?[\d.e+-]+)',
    re.MULTILINE,
)
# Plain chi only (no +2D or +D suffix)
RE_PLAIN_CHI = re.compile(r'^D_(\d+)_chi_(\d+)_energy_magnetization_correlation\.txt$')

# ──────────────────────────────────────────────────────────────────────────────
# File parsing
# ──────────────────────────────────────────────────────────────────────────────
def parse_file(fpath):
    """Parse one .txt file; return dict of observables or None on failure."""
    try:
        txt = open(fpath).read()
    except OSError as e:
        print(f"  Warning: cannot read {fpath}: {e}")
        return None

    # Energy per site
    m = RE_ENERGY_PS.search(txt)
    if not m:
        return None
    energy_per_site = float(m.group(1))

    # Truncation error (may be absent in older files)
    m = RE_TRUNC.search(txt)
    trunc_error = float(m.group(1)) if m else float('nan')

    # Correlations
    corr = {}
    for m in RE_CORR.finditer(txt):
        corr[(int(m.group(1)), m.group(2))] = float(m.group(3))

    # Magnetizations
    mag = {}
    for m in RE_MAG.finditer(txt):
        mag[(int(m.group(1)), m.group(2))] = {
            'Sx': float(m.group(3)),
            'Sy': float(m.group(4)),
            'Sz': float(m.group(5)),
        }

    # ── Compute m_neel from Sz sublattice averages ────────────────────────────
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
    if not (np.isnan(m_ace) or np.isnan(m_bdf)):
        m_neel = 0.5 * abs(m_ace - m_bdf)
    else:
        m_neel = float('nan')

    # ── Compute bond group means and assign ranks ─────────────────────────────
    group_means = []
    for group in NN_GROUPS_RAW:
        vals = [corr[(env, pair)] for (env, pair) in group if (env, pair) in corr]
        group_means.append(float(np.mean(vals)) if vals else float('nan'))

    # rank 1 = most negative (smallest value), rank 3 = least negative
    valid_idx = [i for i, v in enumerate(group_means) if not np.isnan(v)]
    ranks = [float('nan')] * len(group_means)
    if valid_idx:
        sorted_idx = sorted(valid_idx, key=lambda i: group_means[i])
        for rank_pos, g_idx in enumerate(sorted_idx):
            ranks[g_idx] = rank_pos + 1  # 1 = most negative

    corr_rank1 = group_means[ranks.index(1)] if 1 in ranks else float('nan')
    corr_rank3 = group_means[ranks.index(3)] if 3 in ranks else float('nan')

    return {
        'energy_per_site': energy_per_site,
        'trunc_error':     trunc_error,
        'm_neel':          m_neel,
        'corr_rank1':      corr_rank1,
        'corr_rank3':      corr_rank3,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Scan folder for plain-chi files
# ──────────────────────────────────────────────────────────────────────────────
def scan_plain_files(folder_path):
    """
    Returns dict: D -> sorted list of (chi, fpath)
    Only includes files matching D_N_chi_M_energy...txt (plain chi, no +2D/+D).
    """
    plain_map = {}
    for fname in os.listdir(folder_path):
        m = RE_PLAIN_CHI.match(fname)
        if m:
            D   = int(m.group(1))
            chi = int(m.group(2))
            plain_map.setdefault(D, []).append((chi, os.path.join(folder_path, fname)))
    for D in plain_map:
        plain_map[D].sort(key=lambda x: x[0])
    return plain_map


# ──────────────────────────────────────────────────────────────────────────────
# CSV helpers
# ──────────────────────────────────────────────────────────────────────────────

def _fmt(x):
    """Fixed-point, 10 decimal places; 'nan' if not finite."""
    if np.isnan(x):
        return 'nan'
    return f'{x:.10f}'


def _csv_row(index_val, obs):
    """Return one comma-separated data row."""
    return ','.join([
        str(index_val),
        _fmt(obs['energy_per_site']),
        _fmt(obs['trunc_error']),
        _fmt(obs['m_neel']),
        _fmt(obs['corr_rank1']),
        _fmt(obs['corr_rank3']),
    ])


HEADER_CHI = 'chi,energy_per_site,trunc_error,m_neel,corr_rank1,corr_rank3'
HEADER_D   = 'D,energy_per_site,trunc_error,m_neel,corr_rank1,corr_rank3'
HEADER_D_CHI = 'D,energy_per_site,trunc_error,m_neel,corr_rank1,corr_rank3,chi_used'


def _j2_str(j2):
    """e.g. 0.20 -> '0p20'"""
    s = f'{j2:.10f}'.rstrip('0')
    dot = s.index('.')
    if len(s) - dot - 1 < 2:
        s += '0' * (2 - (len(s) - dot - 1))
    return s.replace('.', 'p')


# ──────────────────────────────────────────────────────────────────────────────
# README writer
# ──────────────────────────────────────────────────────────────────────────────
def write_readme(out_dir):
    readme = os.path.join(out_dir, 'README.txt')
    with open(readme, 'w') as f:
        f.write('extract_obs_0427  —  Observable data tables for C6Ypi iPEPS (0420core dataset)\n')
        f.write('=' * 78 + '\n\n')
        f.write('All files are CSV (comma-separated values).\n')
        f.write('Numbers are in fixed-point decimal notation (10 decimal places).\n')
        f.write('Import in Mathematica: Import["file.csv", "CSV"]\n\n')
        f.write('FILE TYPES\n')
        f.write('----------\n')
        f.write('  J2{jstr}_D{D}_obs.csv\n')
        f.write('      Observables vs chi (bond dimension of the environment)\n')
        f.write('      for a fixed iPEPS bond dimension D and J2.\n')
        f.write('      Only plain-chi runs are included (files named\n')
        f.write('      D_N_chi_M_energy_magnetization_correlation.txt).\n')
        f.write('      Rows are sorted by chi in ascending order.\n\n')
        f.write('  chimax_J2{jstr}_obs.csv\n')
        f.write('      Observables vs D at the highest available plain chi for each D.\n')
        f.write('      Rows are sorted by D in ascending order.\n')
        f.write('      The last column (chi_used) records which chi was selected.\n\n')
        f.write('COLUMNS (same for both file types, first row is the header)\n')
        f.write('-----------------------------------------------------------\n')
        f.write('  chi / D        : environment / iPEPS bond dimension\n')
        f.write('  energy_per_site: variational energy per site\n')
        f.write('  trunc_error    : CTMRG truncation error (nan if absent)\n')
        f.write('  m_neel         : Neel order parameter = 0.5*|<Sz>_ACE - <Sz>_BDF|\n')
        f.write('                   averaged over all 3 environments\n')
        f.write('  corr_rank1     : mean of the most-negative NN bond group\n')
        f.write('  corr_rank3     : mean of the least-negative NN bond group\n\n')
        f.write('NN BOND GROUPS (3 groups of 6 bonds each)\n')
        f.write('-----------------------------------------\n')
        f.write('  Group 0: env1(EB,AD,CF)  env3(BE,FC,DA)\n')
        f.write('  Group 1: env2(CB,AF,ED)  env1(FA,DE,BC)\n')
        f.write('  Group 2: env3(EF,AB,CD)  env2(DC,BA,FE)\n')
        f.write('  Rank 1 = smallest (most negative) group mean\n')
        f.write('  Rank 3 = largest  (least negative) group mean\n\n')
        f.write('J2 VALUES COVERED\n')
        f.write('-----------------\n')
        for j2 in J2_LIST:
            f.write(f'  J2 = {j2:.2f}  ->  {FOLDER_MAP[j2]}\n')
    print(f'  Wrote README.txt')


# ──────────────────────────────────────────────────────────────────────────────
# Main extraction
# ──────────────────────────────────────────────────────────────────────────────
def main():
    write_readme(OUT_DIR)

    for j2 in J2_LIST:
        jstr        = _j2_str(j2)
        folder_name = FOLDER_MAP[j2]
        folder_path = os.path.join(DATA_DIR, folder_name)

        if not os.path.isdir(folder_path):
            print(f"  WARNING: folder not found: {folder_path}")
            continue

        print(f"J2={j2:.2f}  ({folder_name})")
        plain_map = scan_plain_files(folder_path)

        # ── Per-D vs chi files ────────────────────────────────────────────────
        chimax_rows = []   # (D, chi_used, obs) for the chimax file

        for D in sorted(plain_map.keys()):
            chi_list = plain_map[D]   # already sorted by chi
            rows = []
            for chi, fpath in chi_list:
                obs = parse_file(fpath)
                if obs is None:
                    continue
                rows.append((chi, obs))

            if not rows:
                continue

            # Write J2{jstr}_D{D}_obs.csv
            fname = f'J2{jstr}_D{D}_obs.csv'
            fout  = os.path.join(OUT_DIR, fname)
            with open(fout, 'w') as f:
                f.write(HEADER_CHI + '\n')
                for chi, obs in rows:
                    f.write(_csv_row(chi, obs) + '\n')
            print(f"  Wrote {fname}  ({len(rows)} chi values)")

            # Record chimax row
            chi_max, obs_max = rows[-1]
            chimax_rows.append((D, chi_max, obs_max))

        # ── chimax vs D file ──────────────────────────────────────────────────
        if chimax_rows:
            fname = f'chimax_J2{jstr}_obs.csv'
            fout  = os.path.join(OUT_DIR, fname)
            with open(fout, 'w') as f:
                f.write(HEADER_D_CHI + '\n')
                for D, chi_used, obs in chimax_rows:
                    f.write(_csv_row(D, obs) + f',{chi_used}\n')
            print(f"  Wrote {fname}  ({len(chimax_rows)} D values)")

    print(f'\nDone.  Files in: {OUT_DIR}')


if __name__ == '__main__':
    main()
