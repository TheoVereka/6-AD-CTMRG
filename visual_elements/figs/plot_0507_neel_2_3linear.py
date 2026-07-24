#!/usr/bin/env python3
"""Plot legacy Neel extrapolation together with 0713-summary 2C3 splitting.

Only these figures are generated:
  - m_neel_extrap_J2_020_028.pdf
  - combined_m_extrap_delta_020_030.pdf
"""

import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

import plot_analysis_Windows as analysis


NEEL_DATA_DIR = r'D:\HyraiOn\ENS_Lyon\Internship\2026-EPFL\data\D345678910'
TWO_C3_DATA_DIR = r'D:\HyraiOn\ENS_Lyon\Internship\2026-EPFL\data\0713summary'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, 'analysis_plots_0507_neel_2_3linear')

ANSATZ_NEEL = 'neel_symmetrized'
ANSATZ_2C3 = '2tensor_twoC3'
NEEL_D_ALLOWED = set(range(3, 11))
TWO_C3_D_ALLOWED = {7, 8, 9}

# ---------------------------------------------------------------------------
# MANUAL NEEL LINEAR-FIT POINT COUNTS
# ---------------------------------------------------------------------------
# Every J2 uses these three fit-window entries unless overridden below.
# Repeating one entry requests the two-window symmetric-error fallback.
NEEL_DEFAULT_LINEAR_POINTS = (2, 3, 3)

# Per-J2 overrides. At least two entries must differ; every n must be >= 1.
# n=1 means the raw magnetization at the largest available D.
NEEL_LINEAR_POINTS_BY_J2 = {
    # 0.20: (2, 3, 4),  # three intercepts -> asymmetric errors
    # 0.25: (3, 4, 4),  # two intercepts -> symmetric errors
}

# J2 points omitted from both figures.
NEEL_EXCLUDED_J2 = {0.255}

# Medium purple: darker than the previous pale version.
NEEL_COLOR = '#9b72b0'

SPLITTING_STYLE = {
    7: ('#fdd0a2', 'o'),
    8: ('#f16913', 's'),
    9: ('#7f2704', '^'),
}

RE_LEGACY_FOLDER = re.compile(r'^neel_symmetrized__J2_([0-9p]+)_\d{8}')
RE_LEGACY_PLAIN = re.compile(
    r'^D_(\d+)_chi_(\d+)_energy_magnetization_correlation\.txt$'
)
RE_SUMMARY_J2 = re.compile(r'^J2_([0-9p]+)$')
RE_SUMMARY_D = re.compile(r'^D_(\d+)$')


def parse_j2(label):
    return round(float(label.replace('p', '.')), 6)


def fit_points_for_j2(j2):
    overrides = {
        round(float(key), 6): tuple(int(n) for n in value)
        for key, value in NEEL_LINEAR_POINTS_BY_J2.items()
    }
    pair = overrides.get(
        round(float(j2), 6),
        tuple(int(n) for n in NEEL_DEFAULT_LINEAR_POINTS),
    )
    if len(pair) != 3 or min(pair) < 1 or len(set(pair)) < 2:
        raise ValueError(
            'Neel fit setting must contain three n>=1 entries with at least '
            f'two distinct values; J2={j2} has {pair}'
        )
    return pair


def discover_legacy_neel_folders():
    """Use the lexicographically last legacy Neel folder for each J2."""
    selected = {}
    for name in sorted(os.listdir(NEEL_DATA_DIR)):
        path = os.path.join(NEEL_DATA_DIR, name)
        match = RE_LEGACY_FOLDER.match(name)
        if not match or not os.path.isdir(path):
            continue
        try:
            j2 = parse_j2(match.group(1))
        except ValueError:
            continue
        if j2 == 0.0 or 0.20 <= j2 <= 0.30:
            selected[j2] = path
    return selected


def load_legacy_neel_folder(folder_path):
    """For each D, read the non-lookahead observable with highest chi."""
    candidates = {}
    for name in os.listdir(folder_path):
        match = RE_LEGACY_PLAIN.match(name)
        if not match:
            continue
        D, chi = int(match.group(1)), int(match.group(2))
        if D not in NEEL_D_ALLOWED:
            continue
        path = os.path.join(folder_path, name)
        incumbent = candidates.get(D)
        if incumbent is None or chi > incumbent[0]:
            candidates[D] = (chi, path)

    D_data = {}
    for D, (_, path) in sorted(candidates.items()):
        try:
            D_data[D] = analysis.parse_plain_file(path)
        except Exception as exc:
            print(f'  Warning: failed to parse {path}: {exc}')
    if not D_data:
        return None

    Ds = sorted(D_data)
    order = analysis.compute_order_param(D_data)
    return {
        'Ds': Ds,
        'mneel_list': [order[D]['m_neel'] for D in Ds],
    }


def load_legacy_neel():
    result = {}
    for j2, folder in sorted(discover_legacy_neel_folders().items()):
        values = load_legacy_neel_folder(folder)
        if values:
            result[j2] = values
            print(f'  Legacy Neel J2={j2:.3g}: D={values["Ds"]}')
    return result


def load_0713_two_c3():
    """Load only D=7,8,9 2C3 observations from structured 0713summary."""
    result = {}
    for j2_name in sorted(os.listdir(TWO_C3_DATA_DIR)):
        j2_path = os.path.join(TWO_C3_DATA_DIR, j2_name)
        match = RE_SUMMARY_J2.match(j2_name)
        if not match or not os.path.isdir(j2_path):
            continue
        try:
            j2 = parse_j2(match.group(1))
        except ValueError:
            continue
        if not (0.20 <= j2 <= 0.30):
            continue

        ansatz_path = os.path.join(j2_path, ANSATZ_2C3)
        if not os.path.isdir(ansatz_path):
            continue
        D_data = {}
        for d_name in sorted(os.listdir(ansatz_path)):
            d_path = os.path.join(ansatz_path, d_name)
            d_match = RE_SUMMARY_D.match(d_name)
            if not d_match or not os.path.isdir(d_path):
                continue
            D = int(d_match.group(1))
            if D not in TWO_C3_D_ALLOWED:
                continue
            observable = os.path.join(d_path, 'energy_magnetization_correlation.txt')
            if not os.path.isfile(observable):
                continue
            try:
                D_data[D] = analysis.parse_plain_file(observable)
            except Exception as exc:
                print(f'  Warning: failed to parse {observable}: {exc}')
        if D_data:
            result[j2] = {
                'nn_groups': analysis.compute_bond_groups(
                    D_data, analysis.NN_GROUPS_RAW
                )
            }
            print(f'  0713 2C3 J2={j2:.3g}: D={sorted(D_data)}')
    return result


def neel_extrap_series(neel_data, j2_max, include_j2_zero=False):
    """Build center/errors from two or three distinct linear-fit intercepts."""
    j2s, centers, lower_errors, upper_errors, fit_windows = [], [], [], [], []
    for j2 in sorted(neel_data):
        if j2 in NEEL_EXCLUDED_J2:
            continue
        if not (0.20 <= j2 <= j2_max or (include_j2_zero and j2 == 0.0)):
            continue
        values = neel_data[j2]
        requested_windows = fit_points_for_j2(j2)
        valid = [
            (D, m)
            for D, m in zip(values['Ds'], values['mneel_list'])
            if not np.isnan(m)
        ]
        if len(valid) < 2:
            continue
        Ds = np.asarray([item[0] for item in valid], dtype=float)
        mags = np.asarray([item[1] for item in valid], dtype=float)
        actual_windows = tuple(min(n, len(Ds)) for n in requested_windows)
        distinct_windows = sorted(set(actual_windows))
        if len(distinct_windows) < 2:
            raise ValueError(
                f'Fit windows collapse to one value at J2={j2}: '
                f'requested={requested_windows}, actual={actual_windows}'
            )
        # n=1 is intentionally the raw largest-D point. For n>=2, use the
        # y-intercept of the last-n-point linear fit against 1/D.
        intercepts = [
            float(mags[-1])
            if n == 1
            else float(np.polyfit(1.0 / Ds[-n:], mags[-n:], 1)[1])
            for n in distinct_windows
        ]
        if len(distinct_windows) >= 3:
            lower_bound, center, upper_bound = sorted(intercepts)[:3]
            clipped_center = max(0.0, center)
            lower_error = max(0.0, clipped_center - max(0.0, lower_bound))
            upper_error = max(0.0, max(0.0, upper_bound) - clipped_center)
        else:
            lower_bound, upper_bound = sorted(intercepts)
            center = 0.5 * (lower_bound + upper_bound)
            clipped_center = max(0.0, center)
            # With two distinct fits, keep exactly the requested symmetric
            # half-difference even when the physical center is clipped to zero.
            lower_error = upper_error = 0.5 * (upper_bound - lower_bound)
        j2s.append(j2)
        centers.append(clipped_center)
        lower_errors.append(lower_error)
        upper_errors.append(upper_error)
        fit_windows.append(actual_windows)
    return j2s, centers, lower_errors, upper_errors, fit_windows


def rank_mean(entry, target_rank):
    values = [
        entry['means'][group]
        for group, rank in enumerate(entry['ranks'])
        if rank == target_rank
    ]
    return float(np.mean(values)) if values else float('nan')


def splitting_series(two_c3_data, D):
    j2s, deltas = [], []
    for j2 in sorted(two_c3_data):
        if not (0.20 <= j2 <= 0.30):
            continue
        nn_groups = two_c3_data[j2]['nn_groups']
        if D not in nn_groups:
            continue
        rank1 = rank_mean(nn_groups[D], 1)
        rank3 = rank_mean(nn_groups[D], 3)
        if np.isnan(rank1) or np.isnan(rank3):
            continue
        j2s.append(j2)
        deltas.append(rank3 - rank1)
    return j2s, deltas


def style_j2_axis(ax, x_min, x_max):
    ax.set_xlim(x_min, x_max)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.3g'))
    ax.grid(alpha=0.18, linewidth=0.6)


def draw_neel_centers(ax, neel_data, j2_max, label, include_j2_zero=False):
    j2s, centers, lower, upper, fit_windows = neel_extrap_series(
        neel_data, j2_max, include_j2_zero=include_j2_zero
    )
    if not j2s:
        raise RuntimeError(f'No Neel data found up to J2={j2_max}')
    ax.errorbar(
        j2s,
        centers,
        yerr=[lower, upper],
        fmt='*-',
        color=NEEL_COLOR,
        ms=9,
        lw=1.4,
        capsize=4,
        elinewidth=1.1,
        markeredgewidth=0.45,
        markeredgecolor='#6f5a7e',
        label=label,
    )
    return j2s, centers, lower, upper, fit_windows


def plot_neel_extrap(neel_data):
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    _, _, _, _, fit_windows = draw_neel_centers(
        ax,
        neel_data,
        0.28,
        'Néel linear extrapolation',
        include_j2_zero=True,
    )
    ax.set_xlabel(r'$J_2$')
    ax.set_ylabel(r'extrapolated $m_\mathrm{N\acute{e}el}$')
    ax.set_ylim(bottom=0.0)
    style_j2_axis(ax, -0.01, 0.285)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, 'm_neel_extrap_J2_020_028.pdf')


def plot_combined(neel_data, two_c3_data):
    fig, ax_m = plt.subplots(figsize=(7.4, 5.0))
    draw_neel_centers(
        ax_m,
        neel_data,
        0.30,
        'Néel extrapolated center',
    )
    ax_m.set_xlabel(r'$J_2$')
    ax_m.set_ylabel(
        r'extrapolated $m_\mathrm{N\acute{e}el}$', color=NEEL_COLOR
    )
    ax_m.tick_params(axis='y', labelcolor=NEEL_COLOR)
    ax_m.set_ylim(bottom=0.0)
    style_j2_axis(ax_m, 0.195, 0.305)

    ax_delta = ax_m.twinx()
    for D in (7, 8, 9):
        x, y = splitting_series(two_c3_data, D)
        color, marker = SPLITTING_STYLE[D]
        ax_delta.plot(
            x,
            y,
            marker=marker,
            linestyle='-',
            color=color,
            ms=5.5,
            lw=1.5,
            label=f'0713 2C3 splitting, D={D}',
        )
    ax_delta.set_ylabel(
        r'2C3 $\Delta_\mathrm{NN}$ (rank 3 $-$ rank 1)', color='#7f2704'
    )
    ax_delta.tick_params(axis='y', labelcolor='#7f2704')
    ax_delta.set_ylim(bottom=0.0)

    handles1, labels1 = ax_m.get_legend_handles_labels()
    handles2, labels2 = ax_delta.get_legend_handles_labels()
    ax_m.legend(handles1 + handles2, labels1 + labels2, fontsize=8, loc='lower left')
    fig.tight_layout()
    save(fig, 'combined_m_extrap_delta_020_030.pdf')


def save(fig, filename):
    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    managed = {
        'm_neel_extrap_J2_020_028.pdf',
        'combined_m_extrap_delta_020_030.pdf',
    }
    for name in managed:
        path = os.path.join(OUT_DIR, name)
        if os.path.isfile(path):
            os.remove(path)

    print('Loading legacy Neel data ...')
    neel_data = load_legacy_neel()
    print('Loading 0713summary 2C3 data ...')
    two_c3_data = load_0713_two_c3()
    plot_neel_extrap(neel_data)
    plot_combined(neel_data, two_c3_data)
    print(f'Done. Only two figures were generated in: {OUT_DIR}')


if __name__ == '__main__':
    main()
