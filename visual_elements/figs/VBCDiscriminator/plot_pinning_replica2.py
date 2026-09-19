#!/usr/bin/env python3
"""Plot Izar replica-2 raw results without syncing or writing processed data.

Outputs two supervisor-style PDFs per J2 and two aggregate quadratic-Pinning
PDFs.  This entry point is deliberately isolated from the replica-1 archive,
processed CSVs, and figures.
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

from analyze_three_source_runs import COLORS, discover, select_highest_chi
from fit_pinned_correlations import (
    BRANCH_LABELS,
    LINEAR_RESPONSE_REFERENCE_H,
    RANKS,
    fit_group,
)
from plot_pinning_replica1 import RankedStage, rank_stage


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DEFAULT_INPUT = (REPO / "models" / "VBCPinningClusterBundle"
                 / "Results_VBC_branches")
DEFAULT_OUTPUT = HERE / "replica2_supervisor"
DEFAULT_QUADRATIC_OUTPUT = HERE / "replica2_quadratic_pinning_extrapolation"
PIN_ORDER = ("dimer-plaquette", "plaquette", "rank-split")
PHYSICAL_BRANCHES = ("dimer-plaquette", "plaquette")
PIN_TITLES = {
    "dimer-plaquette": "Dimer-plaquette pin",
    "plaquette": "Plaquette pin",
    "rank-split": "Rank-split pin",
}
PIN_MARKERS = {"dimer-plaquette": "s", "plaquette": "o"}
NN_STYLES = (
    ("strongest", "rank1", "-", "s", 11.0, 3),
    ("middle", "rank2", "--", "o", 8.0, 4),
    ("weakest", "rank3", ":", "^", 5.0, 5),
)


def j2_tag(value: float) -> str:
    whole, fraction = f"{value:.3f}".split(".")
    fraction = fraction.rstrip("0").ljust(2, "0")
    return f"J2_{whole}p{fraction}"


def omega_order(row: RankedStage) -> float:
    omega1 = row.rank2 - row.rank1
    omega2 = row.rank3 - row.rank2
    denominator = omega1 + omega2
    if denominator <= 1e-12:
        return math.nan
    return (omega1 - omega2) / denominator


def padded_limits(values: list[float], *, fraction: float = 0.05,
                  minimum_pad: float = 1e-5) -> tuple[float, float]:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        raise ValueError("cannot set an axis range without finite values")
    lo, hi = min(finite), max(finite)
    pad = max(minimum_pad, fraction * (hi - lo))
    return lo - pad, hi + pad


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_nn_vs_h(rows: list[RankedStage], J2: float, path: Path) -> None:
    dimensions = sorted({row.D for row in rows})
    fields = sorted({row.h for row in rows}, reverse=True)
    y_limits = padded_limits(
        [value for row in rows for value in (row.rank1, row.rank2, row.rank3)],
        fraction=0.045,
    )
    figure_height = 2.4 * len(dimensions) + 1.5
    fig, axes = plt.subplots(len(dimensions), 3, squeeze=False,
                             figsize=(14.2, figure_height))
    for i, D in enumerate(dimensions):
        for j, branch in enumerate(PIN_ORDER):
            ax = axes[i, j]
            subset = sorted((row for row in rows
                             if row.D == D and row.branch == branch),
                            key=lambda row: -row.h)
            if subset:
                xs = [row.h for row in subset]
                color = COLORS[branch]
                for label, attribute, line, marker, size, zorder in NN_STYLES:
                    ax.plot(xs, [getattr(row, attribute) for row in subset],
                            color=color, linestyle=line, linewidth=1.65,
                            marker=marker, markersize=size,
                            markerfacecolor="white", markeredgewidth=1.55,
                            zorder=zorder, label=label)
            else:
                ax.text(0.5, 0.5, "No replica-2 data", ha="center", va="center",
                        transform=ax.transAxes, color="0.45", fontsize=10)
            ax.set_xlim(0.085, -0.005)
            ax.set_ylim(*y_limits)
            ax.set_xticks(fields)
            ax.grid(alpha=0.22)
            if i == 0:
                ax.set_title(PIN_TITLES[branch], color=COLORS[branch],
                             fontsize=12, pad=9)
            if i == len(dimensions) - 1:
                ax.set_xlabel("pinning field $h$")
            else:
                ax.tick_params(labelbottom=False)
            if j == 0:
                ax.set_ylabel(f"$D={D}$\nNN correlation")
            else:
                ax.tick_params(labelleft=False)
    handles = [Line2D([], [], color="0.2", linestyle=line, marker=marker,
                      markersize=size, markerfacecolor="white",
                      markeredgewidth=1.55, label=label)
               for label, _, line, marker, size, _ in NN_STYLES]
    fig.legend(handles=handles, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1 - 0.43 / figure_height), frameon=False)
    fig.suptitle(f"Izar replica 2, $J_2/J_1={J2:g}$: sorted NN correlations",
                 fontsize=14, y=1 - 0.08 / figure_height)
    fig.subplots_adjust(left=0.085, right=0.98,
                        bottom=0.48 / figure_height,
                        top=1 - 1.30 / figure_height,
                        wspace=0.18, hspace=0.30)
    save(fig, path)


def plot_observables_vs_inverse_D(rows: list[RankedStage], J2: float,
                                  path: Path) -> None:
    dimensions = sorted({row.D for row in rows}, reverse=True)
    fields = sorted({row.h for row in rows}, reverse=True)
    xs = [1.0 / D for D in dimensions]
    x_limits = padded_limits(xs, fraction=0.07, minimum_pad=0.003)
    y_limits = (
        padded_limits([row.energy_per_site for row in rows], fraction=0.06,
                      minimum_pad=2e-5),
        padded_limits([row.rank3 - row.rank1 for row in rows], fraction=0.06,
                      minimum_pad=0.005),
        (-1.14, 1.14),
    )
    columns = (
        ("Energy / site, $E$", lambda row: row.energy_per_site),
        (r"NN splitting, $\Delta=C_3-C_1$", lambda row: row.rank3 - row.rank1),
        (r"$\eta=(\omega_1-\omega_2)/(\omega_1+\omega_2)$"
         "\n" r"$\omega_1=C_2-C_1,\quad\omega_2=C_3-C_2$", omega_order),
    )
    fig, axes = plt.subplots(len(fields), 3, squeeze=False,
                             figsize=(14.2, max(5.0, 2.3 * len(fields))))
    for i, field in enumerate(fields):
        for j, (title, quantity) in enumerate(columns):
            ax = axes[i, j]
            for branch in PHYSICAL_BRANCHES:
                subset = sorted((row for row in rows if row.branch == branch
                                 and math.isclose(row.h, field, abs_tol=1e-10)),
                                key=lambda row: 1.0 / row.D)
                if not subset:
                    continue
                ax.plot([1.0 / row.D for row in subset],
                        [quantity(row) for row in subset],
                        color=COLORS[branch], marker=PIN_MARKERS[branch],
                        markersize=7.5, markerfacecolor="white",
                        markeredgewidth=1.3, linewidth=1.5,
                        label=PIN_TITLES[branch])
            ax.set_xlim(*x_limits)
            ax.set_ylim(*y_limits[j])
            ax.grid(alpha=0.22)
            if i == 0:
                ax.set_title(title, fontsize=10.5, pad=11)
            if i == len(fields) - 1:
                ax.set_xlabel("$1/D$")
                ax.set_xticks(xs)
                ax.set_xticklabels([f"{x:.3f}" for x in xs], fontsize=8)
            else:
                ax.tick_params(labelbottom=False)
            if j == 0:
                ax.set_ylabel(f"$h={field:g}$")
            if j == 2:
                ax.axhline(1, color="0.6", linestyle="--", linewidth=0.8)
                ax.axhline(-1, color="0.6", linestyle="--", linewidth=0.8)
                ax.text(0.98, 1, "+1  dimer-plaquette", ha="right", va="bottom",
                        fontsize=8, transform=ax.get_yaxis_transform())
                ax.text(0.98, -1, "-1  plaquette", ha="right", va="top",
                        fontsize=8, transform=ax.get_yaxis_transform())
    handles = [Line2D([], [], color=COLORS[branch],
                      marker=PIN_MARKERS[branch], markersize=7.5,
                      markerfacecolor="white", linewidth=1.5,
                      label=PIN_TITLES[branch])
               for branch in PHYSICAL_BRANCHES]
    fig.legend(handles=handles, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 0.964), frameon=False)
    fig.suptitle(f"Izar replica 2, $J_2/J_1={J2:g}$: fixed-field $1/D$ trends",
                 fontsize=14, y=0.995)
    fig.text(0.5, 0.018,
             r"Sorted $C_1\leq C_2\leq C_3$; finite-h energies belong to "
             "different pinned Hamiltonians.", ha="center", fontsize=9)
    fig.subplots_adjust(left=0.085, right=0.98, bottom=0.075,
                        top=0.895, wspace=0.22, hspace=0.30)
    save(fig, path)


def build_fits(rows: list[RankedStage]):
    groups: dict[tuple[float, int, str], list[RankedStage]] = defaultdict(list)
    for row in rows:
        if row.branch in PHYSICAL_BRANCHES:
            groups[(row.J2, row.D, row.branch)].append(row)
    fits = {}
    summaries = {}
    omissions = {}
    for key, subset in sorted(groups.items()):
        fields = sorted({row.h for row in subset if row.h > 0}, reverse=True)
        if len(fields) < 4:
            omissions[key] = fields
            continue
        rank_fits, summary = fit_group(subset)
        fits[key] = {fit.rank: fit for fit in rank_fits}
        summaries[key] = summary
    return groups, fits, summaries, omissions


def plot_quadratic_fit_pages(groups, fits, omissions, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    page_keys = sorted({(key[0], key[1]) for key in groups})
    with PdfPages(path) as pdf:
        for J2, D in page_keys:
            page_groups = {branch: groups.get((J2, D, branch), [])
                           for branch in PHYSICAL_BRANCHES}
            all_values = [getattr(row, attribute)
                          for subset in page_groups.values() for row in subset
                          for _, attribute, _, _ in RANKS]
            lo, hi = min(all_values), max(all_values)
            pad = max(0.015, 0.05 * (hi - lo))
            fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.3),
                                     sharex=True, sharey=True)
            for ax, branch in zip(axes, PHYSICAL_BRANCHES):
                key = (J2, D, branch)
                subset = page_groups[branch]
                positive = sorted((row for row in subset if row.h > 0),
                                  key=lambda row: row.h)
                for rank_name, attribute, marker, _ in RANKS:
                    ax.plot([row.h for row in positive],
                            [getattr(row, attribute) for row in positive],
                            linestyle="none", marker=marker, markersize=7,
                            markerfacecolor="white", markeredgewidth=1.4,
                            color=COLORS[branch], label=rank_name)
                if key in fits:
                    h_curve = np.linspace(0, max(row.h for row in positive), 200)
                    for rank_name, _, _, line in RANKS:
                        fit = fits[key][rank_name]
                        ax.plot(h_curve,
                                fit.C0 + fit.c1 * h_curve + fit.c2 * h_curve ** 2,
                                color=COLORS[branch], linestyle=line,
                                linewidth=1.6)
                        ax.plot(0, fit.C0, marker="*", markersize=10,
                                color=COLORS[branch], linestyle="none")
                    fit_values = list(fits[key].values())
                    max_ratio = max(
                        fit.quadratic_over_linear_at_reference_h
                        for fit in fit_values)
                    ax.text(
                        0.03, 0.04,
                        f"min $R^2$={min(f.r_squared for f in fit_values):.5f}\n"
                        f"max $|c_2h/c_1|$ at $h={LINEAR_RESPONSE_REFERENCE_H:g}$="
                        f"{max_ratio:.3f}",
                        transform=ax.transAxes, fontsize=8.5, va="bottom",
                        bbox={"facecolor": "white", "edgecolor": "0.8",
                              "alpha": 0.85},
                    )
                else:
                    fields = omissions.get(key, [])
                    ax.text(
                        0.03, 0.04,
                        f"Not fitted: {len(fields)} distinct $h>0$ points\n"
                        f"available: {', '.join(f'{field:g}' for field in fields)}\n"
                        "need at least 4",
                        transform=ax.transAxes, fontsize=8.5, va="bottom",
                        bbox={"facecolor": "white", "edgecolor": "0.8",
                              "alpha": 0.85},
                    )
                zero = [row for row in subset
                        if math.isclose(row.h, 0.0, abs_tol=1e-14)]
                if zero:
                    for _, attribute, marker, _ in RANKS:
                        ax.plot(0, getattr(zero[0], attribute), marker="x",
                                markersize=7, color="0.25", linestyle="none")
                ax.set_title(BRANCH_LABELS[branch], color=COLORS[branch])
                ax.set_xlabel("pinning field $h$")
                ax.set_xlim(-0.004, 0.084)
                ax.set_ylim(lo - pad, hi + pad)
                ax.grid(alpha=0.22)
            axes[0].set_ylabel("sorted NN correlation")
            rank_handles = [Line2D([], [], color="0.2", linestyle=line,
                                   marker=marker, markerfacecolor="white",
                                   label=name)
                            for name, _, marker, line in RANKS]
            fig.legend(handles=rank_handles, loc="lower center", ncol=3,
                       frameon=False, fontsize=8.5)
            fig.suptitle(
                f"Izar replica 2: $J_2/J_1={J2:g}$, $D={D}$; "
                r"$C(h)=C_0+c_1h+c_2h^2$ from $h>0$ only",
                fontsize=13,
            )
            fig.tight_layout(rect=(0, 0.10, 1, 0.94))
            pdf.savefig(fig)
            plt.close(fig)


def plot_quadratic_splitting(summaries, path: Path) -> None:
    dimensions = sorted({key[1] for key in summaries})
    fig, axes = plt.subplots(1, len(dimensions),
                             figsize=(5.2 * len(dimensions), 4.8),
                             squeeze=False, sharey=True)
    for ax, D in zip(axes[0], dimensions):
        for branch in PHYSICAL_BRANCHES:
            subset = sorted((summary for (J2, d, b), summary in summaries.items()
                             if d == D and b == branch), key=lambda row: row.J2)
            if not subset:
                continue
            ax.errorbar(
                [row.J2 for row in subset], [row.Delta0 for row in subset],
                yerr=[row.Delta0_fit_stderr for row in subset],
                color=COLORS[branch], marker=PIN_MARKERS[branch],
                linestyle="-", linewidth=1.35, markersize=7,
                markerfacecolor="white", capsize=2.5,
                label=BRANCH_LABELS[branch],
            )
        ax.set_title(f"$D={D}$")
        ax.set_xlabel("$J_2/J_1$")
        ax.grid(alpha=0.22)
        panel_j2 = sorted({J2 for J2, d, _ in summaries if d == D})
        ax.set_xticks(panel_j2)
    axes[0, 0].set_ylabel(
        r"extrapolated splitting $\Delta_0=C_{\rm weakest}(0)-C_{\rm strongest}(0)$"
    )
    handles, labels = [], []
    for ax in axes[0]:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Izar replica 2: quadratic $h>0$ extrapolation", fontsize=14)
    fig.tight_layout(rect=(0, 0.12, 1, 0.93))
    save(fig, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--quadratic-output-dir", type=Path,
                        default=DEFAULT_QUADRATIC_OUTPUT)
    args = parser.parse_args()
    if not args.input.is_dir():
        raise FileNotFoundError(f"replica-2 input directory is missing: {args.input}")

    stages = [stage for stage in select_highest_chi(discover(args.input, strict=True))
              if stage.replica == 2]
    if not stages:
        raise RuntimeError(f"no readable replica-2 observations in {args.input}")
    rows = [rank_stage(stage, "Izar") for stage in stages]

    by_J2: dict[float, list[RankedStage]] = defaultdict(list)
    for row in rows:
        by_J2[row.J2].append(row)
    for J2, subset in sorted(by_J2.items()):
        folder = args.output_dir / "Izar" / j2_tag(J2)
        plot_nn_vs_h(subset, J2, folder / "01_sorted_nn_vs_h.pdf")
        plot_observables_vs_inverse_D(
            subset, J2, folder / "02_energy_delta_omega_vs_inverse_D.pdf")
        print(f"Izar replica 2 J2={J2:g}: {len(subset)} stages; {folder}")

    groups, fits, summaries, omissions = build_fits(rows)
    plot_quadratic_fit_pages(
        groups, fits, omissions,
        args.quadratic_output_dir / "01_correlation_fits.pdf",
    )
    if not summaries:
        raise RuntimeError("no replica-2 group has four distinct positive fields")
    plot_quadratic_splitting(
        summaries,
        args.quadratic_output_dir / "02_extrapolated_splitting_vs_J2.pdf",
    )
    print(f"Replica-2 quadratic fits: {len(summaries)}; "
          f"incomplete groups shown but not fitted: {len(omissions)}")
    print(f"Quadratic figures: {args.quadratic_output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
