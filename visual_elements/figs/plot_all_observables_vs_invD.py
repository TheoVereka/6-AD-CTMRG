"""
Plot scaling behaviour of all observables vs 1/D for 6-site Heisenberg iPEPS.

Four separate figures are produced **per run folder** and saved inside that folder.
The script scans every immediate subdirectory of --data-dir that contains at least
one D_*_chi_*.txt file; if --data-dir itself contains such files it is also treated
as a run folder.

Figures (PDF + PNG):
  1. nn_bond_corr_vs_invD
       9 unique NN bonds.  Each bond appears in exactly 2 of the 3 envs.
       Error bar: centre = mean of those 2 measurements,
                  half-width = ½(max − min).

  2. nnn_bond_corr_vs_invD
       All 18 NNN measurements plotted individually (6 canonical bonds × 3 envs).
       No averaging — each (env, bond-label) is its own curve.
       Colour: canonical bond (6 colours).  Line style: env (solid/dashed/dotted).

  3. site_magnetization_vs_invD
       |m| = √(Sx²+Sy²+Sz²) per site A–F.
       Mean ± SE (std/√3) over the 3 envs.

  4. energy_per_site_vs_invD
       Scalar E/N curve.

χ_eff is extracted from the filename pattern  D_<D>_chi_<base>+2D=<chi_eff>_....txt
When a folder contains multiple files for the same D the one with the lowest
energy_per_site (best variational state) is selected.

Usage
-----
    python plot_all_observables_vs_invD.py [--data-dir DIR] [--no-show]
"""

import os
import re
import argparse
from collections import defaultdict
from typing import Any, Optional

import numpy as np
import matplotlib.pyplot as plt

# ── regex patterns ─────────────────────────────────────────────────────────────
CORR_RE   = re.compile(
    r'corr_env(\d+)_([A-F]{2})\s*=\s*([+\-]?\d+\.\d+e[+\-]\d+)'
)
MAG_RE    = re.compile(
    r'mag_env(\d+)_([A-F])\s+'
    r'Sx=([+\-]?\d+\.\d+e[+\-]\d+)\s+'
    r'Sy=([+\-]?\d+\.\d+e[+\-]\d+)\s+'
    r'Sz=([+\-]?\d+\.\d+e[+\-]\d+)'
)
EPS_RE    = re.compile(r'^energy_per_site\s*=\s*([+\-]?\d+\.\d+e[+\-]\d+)')
FNAME_RE  = re.compile(r'D_(\d+)_chi_')
CHIEFF_RE  = re.compile(r'\+2D(?:_equals_chi_|=)(\d+)')
CHIBASE_RE = re.compile(r'D_\d+_chi_(\d+)')

# ── bond classification ────────────────────────────────────────────────────────
def _canon(pair: str) -> str:
    return ''.join(sorted(pair))


# 9 unique NN bonds (each seen in exactly 2 envs):
NN_CANONICAL: frozenset = frozenset({
    'AB', 'AD', 'AF', 'BC', 'BE', 'CD', 'CF', 'DE', 'EF',
})

# 6 unique NNN bonds (each seen in all 3 envs):
NNN_CANONICAL: frozenset = frozenset({
    'AC', 'AE', 'BD', 'BF', 'CE', 'DF',
})

SITES = list('ABCDEF')

# NNN plotting config: colour per canonical bond, linestyle/marker per env
_NNN_BONDS_SORTED = sorted(NNN_CANONICAL)   # ['AC','AE','BD','BF','CE','DF']
_NNN_COLORS       = plt.cm.tab10(np.linspace(0, 0.55, 6))
_ENV_STYLES       = {1: ('-',  'o'), 2: ('--', 's'), 3: (':', '^')}

# ── parsing ────────────────────────────────────────────────────────────────────

def parse_file(fpath: str) -> dict[str, Any]:
    """
    Returns
    -------
    nn      : {canonical_bond: [val_envA, val_envB]}   – 2 values per bond
    nnn_raw : {(env_int, raw_pair_str): float}          – 18 individual values
    mag     : {site: [|m|_env1, |m|_env2, |m|_env3]}
    eps     : float | None
    """
    nn:      dict[str, list[float]]         = defaultdict(list)
    nnn_raw: dict[tuple[int, str], float]   = {}
    mag:     dict[str, list[float]]         = defaultdict(list)
    eps:     Optional[float]                = None

    with open(fpath) as fh:
        for line in fh:
            m = EPS_RE.match(line.strip())
            if m:
                eps = float(m.group(1))
                continue

            m = CORR_RE.search(line)
            if m:
                env_k = int(m.group(1))
                raw   = m.group(2)          # original pair string, e.g. "EC"
                cb    = _canon(raw)         # canonical, e.g. "CE"
                val   = float(m.group(3))
                if cb in NN_CANONICAL:
                    nn[cb].append(val)
                elif cb in NNN_CANONICAL:
                    nnn_raw[(env_k, raw)] = val
                continue

            m = MAG_RE.search(line)
            if m:
                sx = float(m.group(3))
                sy = float(m.group(4))
                sz = float(m.group(5))
                mag[m.group(2)].append(float(np.sqrt(sx**2 + sy**2 + sz**2)))

    return {
        'nn':      dict(nn),
        'nnn_raw': nnn_raw,
        'mag':     dict(mag),
        'eps':     eps,
    }


# ── per-folder data loader ─────────────────────────────────────────────────────

def load_folder(
    folder: str,
) -> tuple[dict[int, dict], dict[int, int], dict[int, str]]:
    """
    Scan *folder* (non-recursively) for D_*_chi_*.txt files.
    Returns (results, chi_map, best_files).
    For duplicate D values the file with the lowest energy_per_site is used.
    """
    candidates: dict[int, list[tuple[float, int, str]]] = defaultdict(list)

    for fname in sorted(os.listdir(folder)):
        if not fname.endswith('.txt'):
            continue
        dm = FNAME_RE.search(fname)
        if not dm:
            continue
        D     = int(dm.group(1))
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

    results:    dict[int, dict] = {}
    chi_map:    dict[int, int]  = {}
    best_files: dict[int, str]  = {}

    for D, items in sorted(candidates.items()):
        items.sort(key=lambda t: (-t[1], t[0]))  # highest chi_eff first, then lowest energy
        best_eps, best_chi, best_path = items[0]
        if len(items) > 1:
            print(
                f'    [D={D}] {len(items)} files; '
                f'selected E/N={best_eps:.6e} ← {os.path.basename(best_path)}'
            )
        results[D]    = parse_file(best_path)
        chi_map[D]    = best_chi
        best_files[D] = best_path

    return results, chi_map, best_files


def find_run_folders(data_dir: str) -> list[str]:
    """
    Return every immediate subdirectory of *data_dir* that contains at least
    one D_*_chi_*.txt file.  Also include *data_dir* itself if it does.
    """
    run_dirs: list[str] = []

    def _has_data(d: str) -> bool:
        try:
            return any(
                FNAME_RE.search(f) and f.endswith('.txt')
                for f in os.listdir(d)
                if os.path.isfile(os.path.join(d, f))
            )
        except PermissionError:
            return False

    if _has_data(data_dir):
        run_dirs.append(data_dir)

    for entry in sorted(os.scandir(data_dir), key=lambda e: e.name):
        if entry.is_dir() and _has_data(entry.path):
            run_dirs.append(entry.path)

    return run_dirs


# ── statistical helpers ────────────────────────────────────────────────────────

def _mean_halfrange(vals: list[float]) -> tuple[float, float]:
    a = np.asarray(vals, dtype=float)
    return float(a.mean()), float((a.max() - a.min()) / 2.0)


def _mean_se(vals: list[float]) -> tuple[float, float]:
    a = np.asarray(vals, dtype=float)
    if len(a) < 2:
        return float(a.mean()), 0.0
    return float(a.mean()), float(a.std(ddof=1) / np.sqrt(len(a)))


# ── shared helpers ─────────────────────────────────────────────────────────────

def _set_xticks_with_chi(
    ax: 'plt.Axes', Ds: list[int], x: 'np.ndarray', chi_map: dict[int, int]
) -> None:
    """Set x-ticks to D values; label each tick with 'D=X\nχ=Y'."""
    ax.set_xticks(x)
    labels = [
        f'D={D}\nχ={chi_map.get(D, "?")}'
        for D in Ds
    ]
    ax.set_xticklabels(labels, fontsize=8)


def _save_fig(fig: 'plt.Figure', folder: str, stem: str) -> None:
    for ext in ('pdf', 'png'):
        path = os.path.join(folder, f'{stem}.{ext}')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f'    Saved: {path}')


# ── Figure 1 – NN bond correlations ───────────────────────────────────────────

def plot_nn_bonds(
    Ds: list[int], x: 'np.ndarray',
    results: dict[int, dict], chi_map: dict[int, int],
    folder: str, run_label: str,
) -> 'plt.Figure':
    bonds  = sorted(NN_CANONICAL)
    colors = plt.cm.tab10(np.linspace(0, 0.9, 9))

    fig, ax = plt.subplots(figsize=(9, 6))

    for idx, bond in enumerate(bonds):
        mus, errs = [], []
        for D in Ds:
            vals = results[D]['nn'].get(bond, [])
            if not vals:
                mus.append(np.nan); errs.append(0.0)
            else:
                mu, hr = _mean_halfrange(vals)
                mus.append(mu); errs.append(hr)
        ax.errorbar(
            x, mus, yerr=errs,
            fmt='o-', capsize=4, capthick=1.2,
            linewidth=1.5, markersize=6,
            color=colors[idx], label=bond,
        )

    ax.set_xlabel(r'$D$', fontsize=13)
    ax.set_ylabel(
        r'$\langle \mathbf{S}_i \cdot \mathbf{S}_j \rangle_\mathrm{NN}$',
        fontsize=13,
    )
    ax.set_title(
        f'9 NN bond correlations vs $D$  [{run_label}]\n'
        r'(error bars: centre = mean of 2 env duplicates, '
        r'half-width = $\frac{1}{2}|\max-\min|$)',
    )
    _set_xticks_with_chi(ax, Ds, x, chi_map)
    ax.legend(fontsize=9, ncol=3)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_fig(fig, folder, 'nn_bond_corr_vs_D')
    return fig


# ── Figure 2 – NNN bond correlations (18 individual curves) ───────────────────

def plot_nnn_bonds(
    Ds: list[int], x: 'np.ndarray',
    results: dict[int, dict], chi_map: dict[int, int],
    folder: str, run_label: str,
) -> 'plt.Figure':
    # Collect every (env, raw_pair) key that exists in any D
    all_keys: set[tuple[int, str]] = set()
    for D in Ds:
        all_keys |= results[D]['nnn_raw'].keys()

    # Sort: first by canonical bond (→ colour group), then by env (→ style)
    def _sort_key(ek: tuple[int, str]) -> tuple[str, int]:
        return (_canon(ek[1]), ek[0])

    sorted_keys = sorted(all_keys, key=_sort_key)

    fig, ax = plt.subplots(figsize=(11, 7))

    for env_k, raw_pair in sorted_keys:
        canon = _canon(raw_pair)
        c_idx = _NNN_BONDS_SORTED.index(canon)
        ls, mk = _ENV_STYLES[env_k]

        ys = []
        for D in Ds:
            ys.append(results[D]['nnn_raw'].get((env_k, raw_pair), np.nan))

        label = f'e{env_k}:{raw_pair}'
        ax.plot(
            x, ys,
            linestyle=ls, marker=mk,
            linewidth=1.4, markersize=5,
            color=_NNN_COLORS[c_idx],
            label=label,
        )

    ax.set_xlabel(r'$D$', fontsize=13)
    ax.set_ylabel(
        r'$\langle \mathbf{S}_i \cdot \mathbf{S}_j \rangle_\mathrm{NNN}$',
        fontsize=13,
    )
    ax.set_title(
        f'18 NNN bond correlations vs $D$  [{run_label}]\n'
        '(each of 6 canonical NNN bonds × 3 envs shown individually;\n'
        'colour = bond, line style = env: solid/dashed/dotted for env1/2/3)',
        fontsize=10,
    )
    _set_xticks_with_chi(ax, Ds, x, chi_map)
    ax.legend(fontsize=7.5, ncol=6, loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_fig(fig, folder, 'nnn_bond_corr_vs_D')
    return fig


# ── Figure 3 – per-site magnetizations ────────────────────────────────────────

def plot_site_mag(
    Ds: list[int], x: 'np.ndarray',
    results: dict[int, dict], chi_map: dict[int, int],
    folder: str, run_label: str,
) -> 'plt.Figure':
    colors = plt.cm.tab10(np.linspace(0, 0.6, 6))
    fig, ax = plt.subplots(figsize=(9, 6))

    for idx, site in enumerate(SITES):
        mus, sems = [], []
        for D in Ds:
            vals = results[D]['mag'].get(site, [])
            if not vals:
                mus.append(np.nan); sems.append(0.0)
            else:
                mu, se = _mean_se(vals)
                mus.append(mu); sems.append(se)
        ax.errorbar(
            x, mus, yerr=sems,
            fmt='D-', capsize=4, capthick=1.2,
            linewidth=1.5, markersize=6,
            color=colors[idx], label=f'site {site}',
        )

    ax.set_xlabel(r'$D$', fontsize=13)
    ax.set_ylabel(
        r'Local magnetization $|m| = \sqrt{S_x^2+S_y^2+S_z^2}$',
        fontsize=12,
    )
    ax.set_title(
        f'Per-site local magnetization vs $1/D$  [{run_label}]\n'
        r'(error bars: SE = std/$\sqrt{3}$ over 3 envs)',
        fontsize=11,
    )
    _set_xticks_with_chi(ax, Ds, x, chi_map)
    ax.legend(fontsize=9, ncol=3)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_fig(fig, folder, 'site_magnetization_vs_D')
    return fig


# ── Figure 4 – energy per site ────────────────────────────────────────────────

def plot_energy(
    Ds: list[int], x: 'np.ndarray',
    results: dict[int, dict], chi_map: dict[int, int],
    folder: str, run_label: str,
) -> 'plt.Figure':
    eps_vals = [results[D]['eps'] for D in Ds]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, eps_vals, 'o-', linewidth=1.8, markersize=8,
            color='steelblue', label=r'$E/N$')

    ax.set_xlabel(r'$D$', fontsize=13)
    ax.set_ylabel(r'Energy per site $E/N$', fontsize=13)
    ax.set_title(f'Energy per site vs $D$  [{run_label}]', fontsize=12)
    _set_xticks_with_chi(ax, Ds, x, chi_map)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_fig(fig, folder, 'energy_per_site_vs_D')
    return fig


# ── summary table ──────────────────────────────────────────────────────────────

def print_summary(
    Ds: list[int], x: 'np.ndarray',
    results: dict[int, dict], chi_map: dict[int, int],
    best_files: dict[int, str],
    folder: str,
) -> None:
    print(f'\n  {"D":>4}  {"χ_eff":>6}  {"E/N":>20}  file')
    print('  ' + '-' * 65)
    for D in Ds:
        eps  = results[D]['eps']
        chi  = chi_map.get(D, '?')
        fname = os.path.basename(best_files[D])
        print(f'  {D:>4}  {chi!s:>6}  {eps:>20.12e}  {fname}')

    # NN bond table
    print(f'\n  {"Bond":>6}  ', end='')
    for D in Ds:
        print(f'  D={D}(mean±½Δ)', end='')
    print()
    print('  ' + '-' * (8 + 18 * len(Ds)))
    for bond in sorted(NN_CANONICAL):
        print(f'  {bond:>6}  ', end='')
        for D in Ds:
            vals = results[D]['nn'].get(bond, [])
            if vals:
                mu, hr = _mean_halfrange(vals)
                print(f'  {mu:+.5e}±{hr:.1e}', end='')
            else:
                print(f'  {"—":>16}', end='')
        print()

    # NNN bond table (individual)
    print()
    all_keys: set[tuple[int, str]] = set()
    for D in Ds:
        all_keys |= results[D]['nnn_raw'].keys()
    for env_k, raw in sorted(all_keys, key=lambda ek: (_canon(ek[1]), ek[0])):
        label = f'e{env_k}:{raw}'
        print(f'  {label:>8}  ', end='')
        for D in Ds:
            v = results[D]['nnn_raw'].get((env_k, raw), None)
            if v is not None:
                print(f'  {v:+.8e}', end='')
            else:
                print(f'  {"—":>14}', end='')
        print()


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    here     = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.normpath(os.path.join(here, '..', '..', '..', 'data', '0414core'))
    if not os.path.isdir(data_dir):
        data_dir = here   # fallback to script location

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--data-dir', default=data_dir,
        help='directory whose immediate subdirs are scanned for run data '
             f'(default: {data_dir})',
    )
    parser.add_argument(
        '--no-show', action='store_true',
        help='suppress interactive display',
    )
    args = parser.parse_args()

    run_folders = find_run_folders(args.data_dir)
    if not run_folders:
        raise RuntimeError(
            f'No run folders with D_*_chi_*.txt found under {args.data_dir!r}.\n'
            'Use --data-dir to point at the correct parent directory.'
        )

    print(f'Found {len(run_folders)} run folder(s) under {args.data_dir}')

    all_figs: list['plt.Figure'] = []

    for folder in run_folders:
        run_label = os.path.basename(folder) or os.path.basename(args.data_dir)
        print(f'\n── {run_label} ──')

        results, chi_map, best_files = load_folder(folder)
        if not results:
            print('  [skip] no valid data found.')
            continue

        Ds = sorted(results)
        x  = np.array([float(D) for D in Ds])

        print_summary(Ds, x, results, chi_map, best_files, folder)

        print(f'  Generating figures → {folder}')
        all_figs += [
            plot_nn_bonds (Ds, x, results, chi_map, folder, run_label),
            plot_nnn_bonds(Ds, x, results, chi_map, folder, run_label),
            plot_site_mag (Ds, x, results, chi_map, folder, run_label),
            plot_energy   (Ds, x, results, chi_map, folder, run_label),
        ]

    if not args.no_show:
        plt.show()


if __name__ == '__main__':
    main()
