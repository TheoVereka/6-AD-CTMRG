"""
Ansatz comparison: 1/D→0 extrapolated observables vs J₂.

For every J₂ in 0414core that has BOTH a 6tensors__* folder and a
neel_symmetrized__* folder:

  1. The job-*.out log is scanned to find the **last fully-completed D** for that
     folder (the highest D for which "D=X complete" appears in the log).
  2. All data files at Ds ≤ last-complete-D are loaded (best chi per D).
  3. Each observable is extrapolated to 1/D=0 by:
       • linear fit through the last 3 completed Ds  → y_3
       • linear fit through the last 2 completed Ds  → y_2
       • weighted estimate: y = (2/3)·y_2 + (1/3)·y_3
       • error bar      : ½|y_2 − y_3|
  4. The extrapolated values are plotted vs J₂ for both ansätze.

Four figures saved to the data directory:
  compare_nn_bonds_extrap.{pdf,png}
  compare_nnn_bonds_extrap.{pdf,png}
  compare_site_mag_extrap.{pdf,png}
  compare_energy_extrap.{pdf,png}

Usage
-----
    python plot_compare_ansatz_lastD.py [--data-dir DIR] [--no-show]
"""

import os
import re
import glob
import argparse
from collections import defaultdict
from typing import Any, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# ── regex patterns ─────────────────────────────────────────────────────────────
CORR_RE    = re.compile(r'corr_env(\d+)_([A-F]{2})\s*=\s*([+\-]?\d+\.\d+e[+\-]\d+)')
MAG_RE     = re.compile(
    r'mag_env(\d+)_([A-F])\s+'
    r'Sx=([+\-]?\d+\.\d+e[+\-]\d+)\s+'
    r'Sy=([+\-]?\d+\.\d+e[+\-]\d+)\s+'
    r'Sz=([+\-]?\d+\.\d+e[+\-]\d+)'
)
EPS_RE     = re.compile(r'^energy_per_site\s*=\s*([+\-]?\d+\.\d+e[+\-]\d+)')
FNAME_RE   = re.compile(r'D_(\d+)_chi_')
CHIEFF_RE  = re.compile(r'\+2D(?:_equals_chi_|=)(\d+)')
CHIBASE_RE = re.compile(r'D_\d+_chi_(\d+)')
J2_RE      = re.compile(r'J2_0p(\d+)')
# "D=7 complete" in .out log
D_COMPLETE_RE = re.compile(r'D=(\d+) complete')
# "Output dir   : /scratch/.../FOLDER_NAME"
OUTDIR_RE  = re.compile(r'Output dir\s*:\s*(\S+)')


def _canon(pair: str) -> str:
    return ''.join(sorted(pair))

NN_CANONICAL  = frozenset({'AB', 'AD', 'AF', 'BC', 'BE', 'CD', 'CF', 'DE', 'EF'})
NNN_CANONICAL = frozenset({'AC', 'AE', 'BD', 'BF', 'CE', 'DF'})
SITES = list('ABCDEF')
_NNN_BONDS_SORTED = sorted(NNN_CANONICAL)

ANSATZ_STYLE = {
    '6tensors':         {'ls': '-',  'label': '6-tensor (unrestricted)'},
    'neel_symmetrized': {'ls': '--', 'label': 'Néel-symmetrized'},
    '1tensor_C6Ypi':    {'ls': '-.', 'label': '1-tensor C6+Yπ'},
}

# ── build folder→last-complete-D map from job-*.out logs ──────────────────────

def build_complete_D_map(data_dir: str) -> dict[str, int]:
    """
    Scan every job-*.out in *data_dir*, extract the Output dir folder basename
    and the highest D that has "D=X complete" in that log.
    Returns {folder_basename: last_complete_D}.
    """
    result: dict[str, int] = {}
    for fpath in glob.glob(os.path.join(data_dir, 'job-*.out')):
        folder_name: Optional[str] = None
        last_D = 0
        with open(fpath, 'r', encoding='utf-8', errors='replace') as fh:
            for line in fh:
                if folder_name is None:
                    m = OUTDIR_RE.search(line)
                    if m:
                        folder_name = os.path.basename(m.group(1).strip())
                m = D_COMPLETE_RE.search(line)
                if m:
                    last_D = max(last_D, int(m.group(1)))
        if folder_name and last_D > 0:
            result[folder_name] = last_D
    return result


# ── file parser ────────────────────────────────────────────────────────────────

def parse_file(fpath: str) -> dict[str, Any]:
    nn:      dict[str, list[float]]       = defaultdict(list)
    nnn_raw: dict[tuple[int, str], float] = {}
    mag:     dict[str, list[float]]       = defaultdict(list)
    eps:     Optional[float]              = None

    with open(fpath) as fh:
        for line in fh:
            m = EPS_RE.match(line.strip())
            if m:
                eps = float(m.group(1)); continue
            m = CORR_RE.search(line)
            if m:
                env_k = int(m.group(1)); raw = m.group(2)
                cb = _canon(raw); val = float(m.group(3))
                if cb in NN_CANONICAL:
                    nn[cb].append(val)
                elif cb in NNN_CANONICAL:
                    nnn_raw[(env_k, raw)] = val
                continue
            m = MAG_RE.search(line)
            if m:
                sx, sy, sz = float(m.group(3)), float(m.group(4)), float(m.group(5))
                mag[m.group(2)].append(float(np.sqrt(sx**2 + sy**2 + sz**2)))

    return {'nn': dict(nn), 'nnn_raw': nnn_raw, 'mag': dict(mag), 'eps': eps}


# ── per-folder full loader (all Ds ≤ D_max_complete) ──────────────────────────

def load_folder_all_D(
    folder: str,
    D_max_complete: int,
) -> dict[int, dict]:
    """
    Return {D: data_dict} for every D ≤ D_max_complete, using best-chi
    (highest chi_eff first, then lowest energy) selection.
    """
    candidates: dict[int, list[tuple[float, int, str]]] = defaultdict(list)

    for fname in os.listdir(folder):
        if not (fname.endswith('.txt') and FNAME_RE.search(fname)):
            continue
        D = int(FNAME_RE.search(fname).group(1))
        if D > D_max_complete:
            continue
        fpath = os.path.join(folder, fname)
        data  = parse_file(fpath)
        if data['eps'] is None:
            continue
        chi_eff = -1
        cm = CHIEFF_RE.search(fname)
        if cm:
            chi_eff = int(cm.group(1))
        else:
            cb = CHIBASE_RE.search(fname)
            if cb:
                chi_eff = int(cb.group(1))
        candidates[D].append((data['eps'], chi_eff, fpath))

    out: dict[int, dict] = {}
    for D, items in sorted(candidates.items()):
        items.sort(key=lambda t: (-t[1], t[0]))
        out[D] = parse_file(items[0][2])
    return out


# ── 1/D→0 extrapolation ────────────────────────────────────────────────────────

def _extrap_intercept(Ds_used: list[int], vals: list[float]) -> float:
    """Linear fit y = a + b*(1/D), return intercept a (value at 1/D=0)."""
    inv_D = np.array([1.0 / D for D in Ds_used])
    v     = np.array(vals, dtype=float)
    if len(Ds_used) == 1:
        return float(v[0])
    coeffs = np.polyfit(inv_D, v, 1)   # degree-1: [b, a]
    return float(coeffs[1])


def extrap_weighted(
    Ds_sorted: list[int],
    vals: list[float],
) -> tuple[float, float]:
    """
    Use last 2 and last 3 completed Ds to extrapolate to 1/D→0.
    Weighted estimate: (2/3)·fit2 + (1/3)·fit3.
    Error: ½|fit2 − fit3|.
    Falls back gracefully if fewer than 3 Ds available.
    Returns (value, error).
    """
    n = len(Ds_sorted)
    if n == 0:
        return np.nan, np.nan
    if n == 1:
        return float(vals[0]), 0.0

    # last-2 fit
    y2 = _extrap_intercept(Ds_sorted[-2:], vals[-2:])

    if n == 2:
        # only 2 points; no 3-point fit possible — use y2 with zero weight fallback
        return y2, 0.0

    # last-3 fit
    y3 = _extrap_intercept(Ds_sorted[-3:], vals[-3:])

    y_mean = (2.0 / 3.0) * y2 + (1.0 / 3.0) * y3
    y_err  = 0.5 * abs(y2 - y3)
    return y_mean, y_err


# ── statistical helpers ────────────────────────────────────────────────────────

def _mean_halfrange(vals: list[float]) -> tuple[float, float]:
    a = np.asarray(vals, dtype=float)
    return float(a.mean()), float((a.max() - a.min()) / 2.0)


def _mean_se(vals: list[float]) -> tuple[float, float]:
    a = np.asarray(vals, dtype=float)
    if len(a) < 2:
        return float(a.mean()), 0.0
    return float(a.mean()), float(a.std(ddof=1) / np.sqrt(len(a)))


# ── build per-folder extrapolated data ────────────────────────────────────────

def compute_extrap(all_D_data: dict[int, dict]) -> dict[str, Any]:
    """
    Given {D: parsed_data} for a folder, compute 1/D→0 extrapolated values
    for every observable. Returns a dict with same keys as parse_file output
    but values replaced by (y_mean, y_err) tuples.
    """
    Ds   = sorted(all_D_data.keys())
    if not Ds:
        return {}

    # ── NN bonds ──────────────────────────────────────────────────
    nn_extrap: dict[str, tuple[float, float]] = {}
    for bond in NN_CANONICAL:
        series = []
        valid_Ds = []
        for D in Ds:
            vals = all_D_data[D]['nn'].get(bond, [])
            if vals:
                series.append(_mean_halfrange(vals)[0])
                valid_Ds.append(D)
        if valid_Ds:
            nn_extrap[bond] = extrap_weighted(valid_Ds, series)

    # ── NNN individual ────────────────────────────────────────────
    all_nnn_keys: set[tuple[int, str]] = set()
    for D in Ds:
        all_nnn_keys |= all_D_data[D]['nnn_raw'].keys()

    nnn_extrap: dict[tuple[int, str], tuple[float, float]] = {}
    for key in all_nnn_keys:
        series, valid_Ds = [], []
        for D in Ds:
            v = all_D_data[D]['nnn_raw'].get(key)
            if v is not None:
                series.append(v)
                valid_Ds.append(D)
        if valid_Ds:
            nnn_extrap[key] = extrap_weighted(valid_Ds, series)

    # ── site magnetizations ───────────────────────────────────────
    mag_extrap: dict[str, tuple[float, float]] = {}
    for site in SITES:
        series, valid_Ds = [], []
        for D in Ds:
            vals = all_D_data[D]['mag'].get(site, [])
            if vals:
                series.append(_mean_se(vals)[0])
                valid_Ds.append(D)
        if valid_Ds:
            mag_extrap[site] = extrap_weighted(valid_Ds, series)

    # ── energy ────────────────────────────────────────────────────
    eps_series = [all_D_data[D]['eps'] for D in Ds if all_D_data[D]['eps'] is not None]
    eps_Ds     = [D for D in Ds if all_D_data[D]['eps'] is not None]
    eps_extrap = extrap_weighted(eps_Ds, eps_series) if eps_Ds else (np.nan, np.nan)

    return {
        'nn':      nn_extrap,
        'nnn_raw': nnn_extrap,
        'mag':     mag_extrap,
        'eps':     eps_extrap,
        'Ds_used': Ds,
        'D_max':   Ds[-1],
    }


# ── j2 helpers ─────────────────────────────────────────────────────────────────

def _j2_float(folder_name: str) -> Optional[float]:
    m = J2_RE.search(folder_name)
    if not m:
        return None
    return float('0.' + m.group(1))


def _j2_str(val: float) -> str:
    s = f'{val:.4f}'.rstrip('0').rstrip('.')
    return s


# ── figure helpers ─────────────────────────────────────────────────────────────

def _save_fig(fig: 'plt.Figure', out_dir: str, stem: str) -> None:
    for ext in ('pdf', 'png'):
        path = os.path.join(out_dir, f'{stem}.{ext}')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f'  Saved → {path}')


def _xticks_j2(
    ax: 'plt.Axes',
    j2_vals: list[float],
    extrap_6t:   dict[float, dict],
    extrap_neel: dict[float, dict],
) -> None:
    """Three-line tick: J₂=X / 6T: Ds / Néel: Ds"""
    labels = []
    for j2 in j2_vals:
        e6   = extrap_6t.get(j2,   {})
        ene  = extrap_neel.get(j2, {})
        ds6  = e6.get('Ds_used',  [])
        dsn  = ene.get('Ds_used', [])
        top  = f'$J_2$={_j2_str(j2)}'
        mid  = f'6T: D={",".join(map(str,ds6))}' if ds6 else '6T: —'
        bot  = f'Néel: D={",".join(map(str,dsn))}' if dsn else 'Néel: —'
        labels.append(f'{top}\n{mid}\n{bot}')
    ax.set_xticks(range(len(j2_vals)))
    ax.set_xticklabels(labels, fontsize=6.5)


def _ansatz_legend_handles() -> list:
    return [
        mlines.Line2D([], [], color='grey', ls='-',  lw=1.8, marker='o', ms=6,
                      label='6-tensor (unrestricted)'),
        mlines.Line2D([], [], color='grey', ls='--', lw=1.8, marker='o', ms=6,
                      mfc='none', label='Néel-symmetrized'),
        mlines.Line2D([], [], color='grey', ls='-.', lw=1.8, marker='o', ms=6,
                      mfc='none', label='1-tensor C6+Yπ'),
    ]


# ── Figure 1 – NN bond correlations ───────────────────────────────────────────

def plot_nn_bonds(j2_vals, x, extrap_6t, extrap_neel, out_dir):
    bonds  = sorted(NN_CANONICAL)
    colors = plt.cm.tab10(np.linspace(0, 0.9, 9))

    fig, ax = plt.subplots(figsize=(11, 6))

    for idx, bond in enumerate(bonds):
        clr = colors[idx]
        for ansatz, extrap_map, ls in [
            ('6tensors',         extrap_6t,   '-' ),
            ('neel_symmetrized', extrap_neel, '--'),
        ]:
            mus, errs = [], []
            for j2 in j2_vals:
                e = extrap_map.get(j2, {})
                pair = e.get('nn', {}).get(bond)
                if pair is None:
                    mus.append(np.nan); errs.append(0.0)
                else:
                    mus.append(pair[0]); errs.append(pair[1])
            mfc = clr if ansatz == '6tensors' else 'none'
            ax.errorbar(x, mus, yerr=errs, fmt='o', ls=ls,
                        capsize=4, capthick=1.2, linewidth=1.5, markersize=6,
                        color=clr, mfc=mfc)

    bond_handles = [
        mlines.Line2D([], [], color=colors[i], ls='-', marker='o', ms=6, lw=1.5,
                      label=bond)
        for i, bond in enumerate(bonds)
    ]
    leg1 = ax.legend(handles=bond_handles, fontsize=8, loc='upper right',
                     title='Bond', ncol=3)
    ax.add_artist(leg1)
    ax.legend(handles=_ansatz_legend_handles(), fontsize=9, loc='upper left',
              title='Ansatz')

    ax.set_xticks(x); _xticks_j2(ax, j2_vals, extrap_6t, extrap_neel)
    ax.set_ylabel(r'$\langle \mathbf{S}_i \cdot \mathbf{S}_j \rangle_\mathrm{NN}$',
                  fontsize=13)
    ax.set_title(
        r'NN bond correlations — $1/D\to 0$ extrapolated, ansatz comparison' '\n'
        r'(error bar = $\frac{1}{2}|y_2 - y_3|$;  '
        r'$y = \frac{2}{3}y_2 + \frac{1}{3}y_3$)',
        fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, out_dir, 'compare_nn_bonds_extrap')
    return fig


# ── Figure 2 – NNN bond correlations (all 18 individual) ──────────────────────

def plot_nnn_bonds(j2_vals, x, extrap_6t, extrap_neel, out_dir):
    _ENV_STYLES = {1: ('-', 'o'), 2: ('--', 's'), 3: (':', '^')}
    nnn_colors  = plt.cm.tab10(np.linspace(0, 0.55, 6))

    all_keys: set[tuple[int, str]] = set()
    for em in list(extrap_6t.values()) + list(extrap_neel.values()):
        all_keys |= em.get('nnn_raw', {}).keys()
    sorted_keys = sorted(all_keys, key=lambda ek: (_canon(ek[1]), ek[0]))

    fig, ax = plt.subplots(figsize=(13, 7))

    for env_k, raw_pair in sorted_keys:
        canon = _canon(raw_pair)
        c_idx = _NNN_BONDS_SORTED.index(canon)
        clr   = nnn_colors[c_idx]
        ls_env, mk_env = _ENV_STYLES[env_k]

        for ansatz, extrap_map in [('6tensors', extrap_6t), ('neel_symmetrized', extrap_neel)]:
            ys, errs = [], []
            for j2 in j2_vals:
                e = extrap_map.get(j2, {})
                pair = e.get('nnn_raw', {}).get((env_k, raw_pair))
                if pair is None:
                    ys.append(np.nan); errs.append(0.0)
                else:
                    ys.append(pair[0]); errs.append(pair[1])
            lw  = 1.6 if ansatz == '6tensors' else 0.85
            mfc = clr if ansatz == '6tensors' else 'none'
            ax.errorbar(x, ys, yerr=errs, fmt=mk_env, ls=ls_env,
                        linewidth=lw, markersize=5 if ansatz == '6tensors' else 4,
                        capsize=3, capthick=0.8,
                        color=clr, mfc=mfc,
                        alpha=1.0 if ansatz == '6tensors' else 0.65)

    bond_handles = [
        mlines.Line2D([], [], color=nnn_colors[i], ls='-', marker='o', ms=5, lw=1.5,
                      label=bond)
        for i, bond in enumerate(_NNN_BONDS_SORTED)
    ]
    env_handles = [
        mlines.Line2D([], [], color='grey', ls=ls, marker=mk, ms=5, lw=1.5,
                      label=f'env {e}')
        for e, (ls, mk) in _ENV_STYLES.items()
    ]
    ansatz_handles = [
        mlines.Line2D([], [], color='grey', ls='-', lw=1.9, label='6-tensor (unrestricted)'),
        mlines.Line2D([], [], color='grey', ls='-', lw=0.9, alpha=0.65, label='Néel-symmetrized'),
    ]
    leg1 = ax.legend(handles=bond_handles, fontsize=8, loc='lower right',
                     title='NNN bond', ncol=3)
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=env_handles, fontsize=8, loc='lower left', title='Env')
    ax.add_artist(leg2)
    ax.legend(handles=ansatz_handles, fontsize=9, loc='upper right', title='Ansatz')

    ax.set_xticks(x); _xticks_j2(ax, j2_vals, extrap_6t, extrap_neel)
    ax.set_ylabel(r'$\langle \mathbf{S}_i \cdot \mathbf{S}_j \rangle_\mathrm{NNN}$',
                  fontsize=13)
    ax.set_title(
        r'All 18 NNN bond correlations — $1/D\to 0$ extrapolated, ansatz comparison' '\n'
        '(colour=bond, env=line style, ansatz=line weight)',
        fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, out_dir, 'compare_nnn_bonds_extrap')
    return fig


# ── Figure 3 – per-site magnetizations ────────────────────────────────────────

def plot_site_mag(j2_vals, x, extrap_6t, extrap_neel, out_dir):
    colors = plt.cm.tab10(np.linspace(0, 0.6, 6))

    fig, ax = plt.subplots(figsize=(10, 6))

    for idx, site in enumerate(SITES):
        clr = colors[idx]
        for ansatz, extrap_map in [('6tensors', extrap_6t), ('neel_symmetrized', extrap_neel)]:
            mus, errs = [], []
            for j2 in j2_vals:
                pair = extrap_map.get(j2, {}).get('mag', {}).get(site)
                if pair is None:
                    mus.append(np.nan); errs.append(0.0)
                else:
                    mus.append(pair[0]); errs.append(pair[1])
            mfc = clr if ansatz == '6tensors' else 'none'
            ls  = '-'  if ansatz == '6tensors' else '--'
            ax.errorbar(x, mus, yerr=errs, fmt='D', ls=ls,
                        capsize=4, capthick=1.2, linewidth=1.5, markersize=6,
                        color=clr, mfc=mfc)

    site_handles = [
        mlines.Line2D([], [], color=colors[i], ls='-', marker='D', ms=6, lw=1.5,
                      label=f'site {s}')
        for i, s in enumerate(SITES)
    ]
    leg1 = ax.legend(handles=site_handles, fontsize=9, loc='upper right',
                     title='Site', ncol=2)
    ax.add_artist(leg1)
    ax.legend(handles=_ansatz_legend_handles(), fontsize=9, loc='upper left',
              title='Ansatz')

    ax.set_xticks(x); _xticks_j2(ax, j2_vals, extrap_6t, extrap_neel)
    ax.set_ylabel(r'$|m| = \sqrt{S_x^2+S_y^2+S_z^2}$', fontsize=12)
    ax.set_title(
        r'Per-site local magnetization — $1/D\to 0$ extrapolated, ansatz comparison' '\n'
        r'(error bar = $\frac{1}{2}|y_2 - y_3|$)',
        fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, out_dir, 'compare_site_mag_extrap')
    return fig


# ── Figure 4 – energy per site ────────────────────────────────────────────────

def plot_energy(j2_vals, x, extrap_6t, extrap_neel, out_dir):
    fig, ax = plt.subplots(figsize=(9, 5))

    for ansatz, extrap_map, clr, ls in [
        ('6tensors',         extrap_6t,   'steelblue', '-' ),
        ('neel_symmetrized', extrap_neel, 'tomato',    '--'),
    ]:
        ys, errs = [], []
        for j2 in j2_vals:
            pair = extrap_map.get(j2, {}).get('eps', (np.nan, np.nan))
            ys.append(pair[0]); errs.append(pair[1])
        mfc_kw = {} if ansatz == '6tensors' else {'mfc': 'none'}
        ax.errorbar(x, ys, yerr=errs, fmt='o', ls=ls,
                    capsize=4, capthick=1.2, linewidth=1.8, markersize=8,
                    color=clr, label=ANSATZ_STYLE[ansatz]['label'], **mfc_kw)

    ax.set_xticks(x); _xticks_j2(ax, j2_vals, extrap_6t, extrap_neel)
    ax.set_ylabel(r'Energy per site $E/N$', fontsize=13)
    ax.set_title(
        r'Energy per site — $1/D\to 0$ extrapolated, ansatz comparison' '\n'
        r'(error bar = $\frac{1}{2}|y_2 - y_3|$;  '
        r'$y = \frac{2}{3}y_2 + \frac{1}{3}y_3$)',
        fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, out_dir, 'compare_energy_extrap')
    return fig


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    here     = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.normpath(os.path.join(here, '..', '..', '..', 'data', '0414core'))
    if not os.path.isdir(data_dir):
        data_dir = here

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', default=data_dir,
                        help=f'0414core parent directory (default: {data_dir})')
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()
    data_dir = args.data_dir

    # ── Step 1: build folder → last-complete-D map ────────────────────────
    print('Scanning job-*.out logs for last completed D …')
    complete_D_map = build_complete_D_map(data_dir)
    for folder, d_last in sorted(complete_D_map.items()):
        print(f'  {folder}  → last complete D={d_last}')

    # ── Step 2: discover run folders ──────────────────────────────────────
    folders_6t:   dict[float, str] = {}
    folders_neel: dict[float, str] = {}

    for entry in sorted(os.scandir(data_dir), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        name = entry.name
        j2 = _j2_float(name)
        if j2 is None:
            continue
        if name.startswith('6tensors__'):
            folders_6t[j2]   = entry.path
        elif name.startswith('neel_symmetrized__'):
            folders_neel[j2] = entry.path

    common_j2 = sorted(set(folders_6t) & set(folders_neel))
    if not common_j2:
        raise RuntimeError('No J₂ values found in both 6tensors and neel_symmetrized folders.')
    print(f'\nCommon J₂ values ({len(common_j2)}): '
          + ', '.join(_j2_str(v) for v in common_j2))

    # ── Step 3: load all D data ≤ last-complete-D and extrapolate ─────────
    extrap_6t:   dict[float, dict] = {}
    extrap_neel: dict[float, dict] = {}

    for label, folders, extrap_out in [
        ('6tensors',         folders_6t,   extrap_6t),
        ('neel_symmetrized', folders_neel, extrap_neel),
    ]:
        print(f'\n── {label} ──')
        for j2 in common_j2:
            folder     = folders[j2]
            basename   = os.path.basename(folder)
            D_last     = complete_D_map.get(basename)
            if D_last is None:
                print(f'  WARNING: no "D=X complete" found in any log for {basename} — skipping')
                continue
            all_D_data = load_folder_all_D(folder, D_last)
            if not all_D_data:
                print(f'  WARNING: no data files found for {basename} — skipping')
                continue
            Ds_loaded = sorted(all_D_data.keys())
            print(f'  J₂={_j2_str(j2):5s}  D_last={D_last}  '
                  f'Ds loaded={Ds_loaded}  '
                  f'n_extrap_Ds={min(3, len(Ds_loaded))}')
            extrap_out[j2] = compute_extrap(all_D_data)

    j2_vals = sorted(set(extrap_6t) & set(extrap_neel))
    if not j2_vals:
        raise RuntimeError('No J₂ values successfully loaded for both ansätze.')

    x = np.arange(len(j2_vals), dtype=float)

    print(f'\n── Generating extrapolated comparison figures → {data_dir} ──')
    figs = [
        plot_nn_bonds (j2_vals, x, extrap_6t, extrap_neel, data_dir),
        plot_nnn_bonds(j2_vals, x, extrap_6t, extrap_neel, data_dir),
        plot_site_mag (j2_vals, x, extrap_6t, extrap_neel, data_dir),
        plot_energy   (j2_vals, x, extrap_6t, extrap_neel, data_dir),
    ]

    if not args.no_show:
        plt.show()
    for f in figs:
        plt.close(f)


if __name__ == '__main__':
    main()
