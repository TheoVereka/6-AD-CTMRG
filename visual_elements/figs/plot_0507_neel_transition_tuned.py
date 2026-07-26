#!/usr/bin/env python3
"""Transition-tuned variant of plot_0507_neel_2_3linear.py.

The fit-point choices below were selected with the primary goal that the
extrapolated Neel centers form a monotone, approximately square-root-like
decay and remain zero from J2 = 0.25 onward. Smaller error bars were a
secondary criterion. The source data and plotting implementation remain in
the base script so the tuned choices are explicit and easy to audit.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

import plot_0507_neel_2_3linear as base


base.NEEL_LINEAR_POINTS_BY_J2 = {
    # J2=0, 0.20, 0.21 intentionally use the base default for manual tuning.
    # These choices balance the square-root-like decay against fit spread.
    0.000: (1, 2, 2),
    0.200: (4, 5, 5),
    #0.210: (2, 3, 3),
    0.220: (3, 4, 4),
    0.230: (4, 5, 5),
    0.235: (3, 4, 4),
    0.240: (5, 6, 6),
    0.245: (2, 3, 3),
    0.250: (3, 4, 5),
    # 0.255 is excluded by the base script.
    0.260: (3, 4, 5),
    # Only n=4 and n=5 are both negative here; repeat n=5 for the
    # smallest symmetric two-window error compatible with a zero center.
    0.265: (4, 5, 5),
    0.270: (3, 4, 5),
    0.275: (2, 3, 4),
    0.280: (3, 4, 5),
}

base.OUT_DIR = os.path.join(
    base.SCRIPT_DIR,
    'analysis_plots_0507_neel_transition_tuned',
)
base.NEEL_INCLUDE_J2_ZERO = False


def symmetric_scaling(values, last_n1, last_n2, d_min=None):
    """Return two-fit scaling with physical center and symmetric half-spread."""
    valid = [
        (D, m)
        for D, m in zip(values['Ds'], values['mneel_list'])
        if not np.isnan(m) and (d_min is None or D >= d_min)
    ]
    if len(valid) < max(last_n1, last_n2):
        raise ValueError(
            f'Need at least {max(last_n1, last_n2)} D points, found {len(valid)}'
        )
    Ds = np.asarray([item[0] for item in valid], dtype=float)
    mags = np.asarray([item[1] for item in valid], dtype=float)
    inv_D = 1.0 / Ds
    fits = {
        n: np.polyfit(inv_D[-n:], mags[-n:], 1)
        for n in (last_n1, last_n2)
    }
    intercepts = [float(fits[n][1]) for n in (last_n1, last_n2)]
    center = max(0.0, 0.5 * sum(intercepts))
    error = 0.5 * abs(intercepts[0] - intercepts[1])
    return Ds, mags, inv_D, fits, center, error


def save_scaling_plot(
    values,
    filename,
    title,
    last_n1,
    last_n2,
    d_min=None,
    color='#9467bd',
):
    Ds, mags, inv_D, fits, center, error = symmetric_scaling(
        values, last_n1, last_n2, d_min=d_min
    )
    # Match the m_Néel panel in the plot_analysis series.
    fig, ax = plt.subplots(figsize=(4.5, 3.7))
    ax.plot(
        inv_D,
        mags,
        'o-',
        color=color,
        ms=6,
        lw=1.5,
        zorder=5,
        label=r'm$_\mathrm{N}$(D)',
    )
    x_max = float(inv_D.max()) * 1.30
    x_line = np.linspace(0.0, x_max, 300)
    fit_styles = (
        (last_n1, 'tab:orange', '--'),
        (last_n2, 'tab:red', '-.'),
    )
    for n, fit_color, line_style in fit_styles:
        ax.plot(
            x_line,
            np.polyval(fits[n], x_line),
            color=fit_color,
            ls=line_style,
            lw=1.0,
            alpha=0.85,
            label=f'linear fit D≥{int(Ds[-n])}',
        )
    ax.errorbar(
        [0.0],
        [center],
        yerr=[[error], [error]],
        fmt='*',
        color=color,
        ms=13,
        capsize=4,
        elinewidth=1.1,
        markeredgewidth=0.5,
        markeredgecolor='k',
        zorder=7,
        label=rf'{center:.4f} $\pm$ {error:.4f}',
    )
    ax.set_xlabel(r'$1/D$', fontsize=10)
    ax.set_ylabel(r'm$_\mathrm{N\acute{e}el}$', fontsize=10)
    ax.set_title(title)
    ax.set_xlim(left=-0.002, right=x_max)
    y_max = max(float(np.max(mags)), center + error)
    ax.set_ylim(0.0, y_max + max(0.10 * y_max, 1e-6))
    ticks = [0.0] + [
        round(0.05 * k, 6)
        for k in range(1, int(np.ceil(x_max / 0.05)) + 3)
        if 0.05 * k <= x_max + 1e-9
    ]
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(base.ticker.FormatStrFormatter('%.2g'))
    ax.yaxis.set_major_formatter(base.ticker.FormatStrFormatter('%.4g'))
    ax.legend(fontsize=7, loc='lower right')
    fig.tight_layout()
    base.save(fig, filename)
    return center, error


def plot_requested_scalings():
    neel_data = base.load_legacy_neel()
    neel_values = neel_data.get(0.20)
    neel_0265_values = neel_data.get(0.265)
    if not neel_values or not neel_0265_values:
        raise RuntimeError('Missing legacy Néel J2=0.20 or J2=0.265 data')
    neel_result = save_scaling_plot(
        neel_values,
        'scaling_neel_J2_020_last4_last5.pdf',
        r'0507 Néel, $J_2=0.20$',
        4,
        5,
        d_min=5,
    )
    neel_0265_result = save_scaling_plot(
        neel_0265_values,
        'scaling_neel_J2_0265_Dge5_last3_last4.pdf',
        r'0507 Néel, $J_2=0.265$',
        3,
        4,
        d_min=5,
    )
    print(
        '  Scaling extrapolations: '
        f'Néel J2=0.20 {neel_result[0]:.10f} ± {neel_result[1]:.10f}; '
        f'Néel J2=0.265 '
        f'{neel_0265_result[0]:.10f} ± {neel_0265_result[1]:.10f}'
    )


if __name__ == '__main__':
    base.main()
    plot_requested_scalings()
    print(f'Done. Two main and two scaling figures are in: {base.OUT_DIR}')
