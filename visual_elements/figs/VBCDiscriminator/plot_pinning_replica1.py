#!/usr/bin/env python3
"""Separate, rank-sorted pinning plots for Kuma and Izar replica 1.

Each pin source gets its own color and PDF set.  NN correlations are sorted
independently at every measured (J2,D,h): rank1 is the most-negative/strongest
AF bond group, rank3 the least-negative/weakest.  Rank-to-geometry mappings
are retained in CSV.  Finite-h energies from different sources are not
variational comparisons because the Hamiltonians differ.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze_three_source_runs import COLORS, LABELS, Stage, discover, select_highest_chi


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DEFAULT_BUNDLE = REPO / "models" / "VBCPinningClusterBundle"
DEFAULT_OUTPUT = HERE / "replica1_separate_pins"
RANK_STYLES = (
    ("strongest (rank1)", "-", "o", 6.5, 3),
    ("middle (rank2)", "--", "s", 5.8, 4),
    ("weakest (rank3)", ":", "^", 5.0, 5),
)
OBSERVABLES = (
    ("energy_per_site", "E/site"),
    ("delta", "NN splitting Delta"),
    ("middle_fraction", "middle fraction q"),
    ("clock_z6", "clock K6"),
)


@dataclass(frozen=True)
class RankedStage:
    cluster: str
    branch: str
    J2: float
    D: int
    chi: int
    h: float
    energy_per_site: float
    delta: float
    middle_fraction: float
    clock_z6: float
    texture: str
    rank1: float
    rank2: float
    rank3: float
    rank1_group: int
    rank2_group: int
    rank3_group: int
    gap_1_to_2: float
    gap_2_to_3: float
    chi_energy_shift: float
    hours_budget: float
    path: str


def rank_stage(row: Stage, cluster: str) -> RankedStage:
    groups = (row.G0, row.G1, row.G2)
    order = sorted(range(3), key=lambda group: (groups[group], group))
    values = tuple(groups[group] for group in order)
    return RankedStage(
        cluster=cluster, branch=row.branch, J2=row.J2, D=row.D,
        chi=row.chi, h=row.field, energy_per_site=row.energy_per_site,
        delta=row.delta, middle_fraction=row.middle_fraction,
        clock_z6=row.clock_z6, texture=row.texture,
        rank1=values[0], rank2=values[1], rank3=values[2],
        rank1_group=order[0], rank2_group=order[1], rank3_group=order[2],
        gap_1_to_2=values[1] - values[0],
        gap_2_to_3=values[2] - values[1],
        chi_energy_shift=row.chi_energy_shift,
        hours_budget=row.hours_budget, path=row.path,
    )


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _axes_for_facets(n: int, *, width: float = 12.0) -> tuple[plt.Figure, list]:
    ncols = 2
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, 3.4 * nrows),
                             squeeze=False)
    flat = list(axes.flat)
    for ax in flat[n:]:
        ax.set_visible(False)
    return fig, flat[:n]


def _plot_rank_lines(ax, subset: list[RankedStage], xvalues: list[float],
                     color: str) -> None:
    for rank, (label, line, marker, size, zorder) in enumerate(RANK_STYLES, start=1):
        ax.plot(xvalues, [getattr(row, f"rank{rank}") for row in subset],
                color=color, linestyle=line, linewidth=1.7, marker=marker,
                markersize=size, markerfacecolor="white", markeredgewidth=1.4,
                zorder=zorder, label=label)


def _common_h_limits(ax) -> None:
    # A fixed field axis makes missing continuation stages obvious.  In
    # particular, a lone h=0.08 point must not masquerade as a full curve.
    ax.set_xlim(0.085, -0.005)


def _common_inverse_D_limits(ax, rows: list[RankedStage]) -> None:
    xs = [1.0 / row.D for row in rows]
    span = max(xs) - min(xs)
    pad = max(0.003, 0.06 * span)
    ax.set_xlim(min(xs) - pad, max(xs) + pad)


def plot_rank_vs_h(rows: list[RankedStage], path: Path, title: str,
                   color: str) -> None:
    dimensions = sorted({row.D for row in rows})
    fig, axes = _axes_for_facets(len(dimensions))
    for ax, D in zip(axes, dimensions):
        subset = sorted((r for r in rows if r.D == D), key=lambda r: -r.h)
        _plot_rank_lines(ax, subset, [r.h for r in subset], color)
        ax.set_title(f"D={D}, chi={max(r.chi for r in subset)}")
        ax.set_xlabel("pinning field h")
        ax.set_ylabel("NN spin-spin correlation")
        _common_h_limits(ax)
        ax.grid(alpha=0.22)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center",
               ncol=3, fontsize=9)
    fig.suptitle(f"{title}: sorted NN correlations versus h\n"
                 "Nearly coincident ranks are resolved in rank_gaps_vs_h.pdf")
    fig.tight_layout(rect=(0, 0.065, 1, 0.96))
    _save(fig, path)


def plot_rank_gaps_vs_h(rows: list[RankedStage], path: Path, title: str,
                        color: str) -> None:
    dimensions = sorted({row.D for row in rows})
    fig, axes = _axes_for_facets(len(dimensions))
    for ax, D in zip(axes, dimensions):
        subset = sorted((r for r in rows if r.D == D), key=lambda r: -r.h)
        xs = [r.h for r in subset]
        ax.plot(xs, [r.gap_1_to_2 for r in subset], color=color,
                linestyle="-", marker="o", markersize=6,
                markerfacecolor="white", label="middle - strongest")
        ax.plot(xs, [r.gap_2_to_3 for r in subset], color=color,
                linestyle=":", marker="^", markersize=5,
                markerfacecolor="white", label="weakest - middle")
        ax.set_title(f"D={D}")
        ax.set_xlabel("pinning field h")
        ax.set_ylabel("adjacent-rank correlation gap")
        _common_h_limits(ax)
        ax.grid(alpha=0.22)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center",
               ncol=2, fontsize=9)
    fig.suptitle(f"{title}: resolves nearly overlapping rank curves")
    fig.tight_layout(rect=(0, 0.065, 1, 0.96))
    _save(fig, path)


def plot_each_rank_vs_h(rows: list[RankedStage], path: Path, title: str,
                        color: str) -> None:
    """Give nearly degenerate ranks independent panels without shifting data."""
    dimensions = sorted({row.D for row in rows})
    fig, axes = plt.subplots(len(dimensions), 3,
                             figsize=(12.0, max(3.0, 1.95 * len(dimensions))),
                             squeeze=False)
    for row_index, D in enumerate(dimensions):
        subset = sorted((r for r in rows if r.D == D), key=lambda r: -r.h)
        for column, (label, line, marker, size, _) in enumerate(RANK_STYLES):
            ax = axes[row_index, column]
            ax.plot([r.h for r in subset],
                    [getattr(r, f"rank{column + 1}") for r in subset],
                    color=color, linestyle=line, linewidth=1.7,
                    marker=marker, markersize=size,
                    markerfacecolor="white", markeredgewidth=1.4)
            _common_h_limits(ax)
            ax.margins(y=0.18)
            ax.grid(alpha=0.22)
            if row_index == 0:
                ax.set_title(label)
            if row_index == len(dimensions) - 1:
                ax.set_xlabel("pinning field h")
        axes[row_index, 0].set_ylabel(f"D={D}\nNN correlation")
    fig.suptitle(f"{title}: one rank per panel (independent vertical scales)")
    fig.tight_layout()
    _save(fig, path)


def plot_rank_vs_inverse_D(rows: list[RankedStage], path: Path, title: str,
                           color: str) -> None:
    fields = sorted({row.h for row in rows}, reverse=True)
    fig, axes = _axes_for_facets(len(fields))
    for ax, field in zip(axes, fields):
        subset = sorted((r for r in rows if r.h == field), key=lambda r: 1.0 / r.D)
        xs = [1.0 / r.D for r in subset]
        _plot_rank_lines(ax, subset, xs, color)
        ax.set_title(f"h={field:g}")
        ax.set_ylabel("NN spin-spin correlation")
        ax.set_xticks(xs)
        ax.set_xticklabels([str(r.D) for r in subset])
        ax.set_xlabel("D (position = 1/D)")
        _common_inverse_D_limits(ax, rows)
        ax.grid(alpha=0.22)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center",
               ncol=3, fontsize=9)
    fig.suptitle(f"{title}: sorted NN correlations versus 1/D")
    fig.tight_layout(rect=(0, 0.065, 1, 0.96))
    _save(fig, path)


def plot_observables_vs_h(rows: list[RankedStage], path: Path, title: str,
                          color: str) -> None:
    dimensions = sorted({row.D for row in rows})
    fig, axes = plt.subplots(len(dimensions), 4,
                             figsize=(14.0, max(3.0, 2.05 * len(dimensions))),
                             squeeze=False)
    for row_index, D in enumerate(dimensions):
        subset = sorted((r for r in rows if r.D == D), key=lambda r: -r.h)
        for column, (attribute, label) in enumerate(OBSERVABLES):
            ax = axes[row_index, column]
            ax.plot([r.h for r in subset],
                    [getattr(r, attribute) for r in subset],
                    color=color, marker="o", markersize=4.5, linewidth=1.25,
                    markerfacecolor="white")
            _common_h_limits(ax)
            ax.grid(alpha=0.22)
            if row_index == 0:
                ax.set_title(label)
            if row_index == len(dimensions) - 1:
                ax.set_xlabel("pinning field h")
        axes[row_index, 0].set_ylabel(f"D={D}")
    fig.suptitle(f"{title}: available optimization endpoints versus h")
    fig.tight_layout()
    _save(fig, path)


def plot_observables_vs_inverse_D(rows: list[RankedStage], path: Path,
                                  title: str, color: str) -> None:
    fields = sorted({row.h for row in rows}, reverse=True)
    fig, axes = plt.subplots(len(fields), 4,
                             figsize=(14.0, max(3.0, 2.15 * len(fields))),
                             squeeze=False)
    for row_index, field in enumerate(fields):
        subset = sorted((r for r in rows if r.h == field), key=lambda r: 1.0 / r.D)
        for column, (attribute, label) in enumerate(OBSERVABLES):
            ax = axes[row_index, column]
            ax.plot([1.0 / r.D for r in subset],
                    [getattr(r, attribute) for r in subset],
                    color=color, marker="o", markersize=4.5, linewidth=1.25,
                    markerfacecolor="white")
            ax.grid(alpha=0.22)
            _common_inverse_D_limits(ax, rows)
            if row_index == 0:
                ax.set_title(label)
            if row_index == len(fields) - 1:
                ax.set_xlabel("1/D")
        axes[row_index, 0].set_ylabel(f"h={field:g}")
    fig.suptitle(f"{title}: fixed-h trends; only h=0 energy is unbiased")
    fig.tight_layout()
    _save(fig, path)


def write_rows(rows: list[RankedStage], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def plot_one_source(rows: list[RankedStage], folder: Path,
                    cluster: str, branch: str, J2: float) -> None:
    color = COLORS[branch]
    title = f"{cluster} replica 1, J2={J2:.2f}, {LABELS[branch]}"
    write_rows(rows, folder / "sorted_nn_data.csv")
    plot_rank_vs_h(rows, folder / "sorted_nn_vs_h.pdf", title, color)
    plot_each_rank_vs_h(rows, folder / "sorted_nn_each_rank_vs_h.pdf",
                        title, color)
    plot_rank_gaps_vs_h(rows, folder / "rank_gaps_vs_h.pdf", title, color)
    plot_rank_vs_inverse_D(rows, folder / "sorted_nn_vs_inverse_D.pdf", title, color)
    plot_observables_vs_h(rows, folder / "observables_vs_h.pdf", title, color)
    plot_observables_vs_inverse_D(rows, folder / "observables_vs_inverse_D.pdf",
                                  title, color)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kuma", type=Path,
                        default=DEFAULT_BUNDLE / "Results_VBC_three")
    parser.add_argument("--izar", type=Path,
                        default=DEFAULT_BUNDLE / "Results_VBC_branches")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    total = 0
    for cluster, root in (("Kuma", args.kuma), ("Izar", args.izar)):
        if not root.is_dir():
            print(f"{cluster}: missing {root}; skipping")
            continue
        stages = [r for r in select_highest_chi(discover(root)) if r.replica == 1]
        if not stages:
            print(f"{cluster}: no readable replica-1 observations yet")
            continue
        ranked = [rank_stage(r, cluster) for r in stages]
        by_source: dict[tuple[str, float], list[RankedStage]] = defaultdict(list)
        for row in ranked:
            by_source[row.branch, row.J2].append(row)
        for (branch, J2), subset in sorted(by_source.items()):
            j2_tag = f"J2_{J2:.2f}".replace(".", "p")
            folder = args.output_dir / cluster / branch / j2_tag
            plot_one_source(subset, folder, cluster, branch, J2)
            complete = sum(r.h == 0.0 for r in subset)
            print(f"{cluster:4s} {branch:18s} J2={J2:.2f}: "
                  f"{len(subset)} stages, {complete} h=0; wrote {folder}")
            total += len(subset)
    print(f"Total plotted replica-1 stages: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
