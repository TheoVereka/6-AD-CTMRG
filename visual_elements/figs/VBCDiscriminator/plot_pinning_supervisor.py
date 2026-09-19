#!/usr/bin/env python3
"""Two replica-1 summary figures per cluster and J2, including partial runs.

The three NN correlations are sorted afresh at every (J2, D, h).  More
negative means a stronger antiferromagnetic bond.  Finite-field energies
belong to different Hamiltonians; only h=0 energies can be compared across
pinning sources as variational energies.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from analyze_three_source_runs import (
    BRANCHES, COLORS, EXPECTED_FIELDS, discover, select_highest_chi,
)
from plot_pinning_replica1 import (
    DEFAULT_BUNDLE, HERE, REPO, RankedStage, rank_stage,
)


DEFAULT_OUTPUT = HERE / "replica1_supervisor"
DEFAULT_CSV_OUTPUT = REPO.parent / "data" / "processed" / "VBCPinningSupervisorCSV"
PIN_TITLES = {
    "dimer-plaquette": "Dimer-plaquette pin",
    "plaquette": "Plaquette pin",
    "rank-split": "Rank-split pin",
}
PIN_ORDER = ("dimer-plaquette", "plaquette", "rank-split")
PIN_MARKERS = {"dimer-plaquette": ("s", 8.5),
               "plaquette": ("o", 6.5), "rank-split": ("^", 5.0)}
NN_STYLES = (
    ("strongest", "rank1", "-", "s", 11.0, 3),
    ("middle", "rank2", "--", "o", 8.0, 4),
    ("weakest", "rank3", ":", "^", 5.0, 5),
)


def omega_order(row: RankedStage) -> float:
    """eta=(omega1-omega2)/(omega1+omega2), where omega1=C2-C1.

    omega2=C3-C2.  Sorted C1<=C2<=C3 gives eta=+1 for an ideal
    dimer-plaquette texture and -1 for an ideal plaquette texture.  The
    ratio has no meaning if all three correlations coincide.
    """
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


def write_source_csvs(rows: list[RankedStage], folder: Path) -> int:
    """Write two compact tables for every available pinning source."""
    written = 0
    for branch in PIN_ORDER:
        subset = sorted((row for row in rows if row.branch == branch),
                        key=lambda row: (row.D, -row.h))
        if not subset:
            continue
        branch_folder = folder / branch
        branch_folder.mkdir(parents=True, exist_ok=True)
        with (branch_folder / "energy.csv").open(
                "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("D", "h", "E"))
            writer.writeheader()
            writer.writerows({"D": row.D, "h": row.h,
                              "E": row.energy_per_site} for row in subset)
        with (branch_folder / "nn_correlations.csv").open(
                "w", encoding="utf-8", newline="") as stream:
            fields = ("D", "h", "NNcorrStrongest", "NNcorrMiddle",
                      "NNcorrWeakest")
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows({
                "D": row.D,
                "h": row.h,
                "NNcorrStrongest": row.rank1,
                "NNcorrMiddle": row.rank2,
                "NNcorrWeakest": row.rank3,
            } for row in subset)
        written += 2
    return written


def plot_nn_vs_h(rows: list[RankedStage], cluster: str, J2: float,
                 path: Path) -> None:
    dimensions = sorted({row.D for row in rows})
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
                message = ("Not run on Izar" if cluster == "Izar"
                           and branch == "rank-split" else "No data yet")
                ax.text(0.5, 0.5, message, ha="center", va="center",
                        transform=ax.transAxes, color="0.45", fontsize=10)
            ax.set_xlim(0.085, -0.005)
            ax.set_ylim(*y_limits)
            ax.set_xticks(EXPECTED_FIELDS)
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
    fig.suptitle(f"{cluster} replica 1, $J_2/J_1={J2:.2f}$: sorted NN correlations",
                 fontsize=14, y=1 - 0.08 / figure_height)
    fig.subplots_adjust(left=0.085, right=0.98,
                        bottom=0.48 / figure_height,
                        top=1 - 1.30 / figure_height,
                        wspace=0.18, hspace=0.30)
    save(fig, path)


def plot_observables_vs_inverse_D(rows: list[RankedStage], cluster: str,
                                  J2: float, path: Path) -> None:
    dimensions = sorted({row.D for row in rows}, reverse=True)
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
        (r"NN splitting, $\Delta=C_3-C_1$",
         lambda row: row.rank3 - row.rank1),
        (r"$\eta=(\omega_1-\omega_2)/(\omega_1+\omega_2)$"
         "\n" r"$\omega_1=C_2-C_1,\quad\omega_2=C_3-C_2$", omega_order),
    )
    fig, axes = plt.subplots(len(EXPECTED_FIELDS), 3, squeeze=False,
                             figsize=(14.2, 13.8))

    for i, field in enumerate(EXPECTED_FIELDS):
        for j, (title, quantity) in enumerate(columns):
            ax = axes[i, j]
            for branch in PIN_ORDER:
                subset = sorted((row for row in rows if row.branch == branch
                                 and math.isclose(row.h, field, abs_tol=1e-10)),
                                key=lambda row: 1.0 / row.D)
                if not subset:
                    continue
                marker, marker_size = PIN_MARKERS[branch]
                ax.plot([1.0 / row.D for row in subset],
                        [quantity(row) for row in subset],
                        color=COLORS[branch], marker=marker,
                        markersize=marker_size,
                        markerfacecolor="white", markeredgewidth=1.3,
                        linewidth=1.5, label=PIN_TITLES[branch])
            ax.set_xlim(*x_limits)
            ax.set_ylim(*y_limits[j])
            ax.grid(alpha=0.22)
            if i == 0:
                ax.set_title(title, fontsize=10.5, pad=11)
            if i == len(EXPECTED_FIELDS) - 1:
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
                        fontsize=8, transform=ax.get_yaxis_transform(),
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8})
                ax.text(0.98, -1, "−1  plaquette", ha="right", va="top",
                        fontsize=8, transform=ax.get_yaxis_transform(),
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8})

    present = [branch for branch in PIN_ORDER
               if any(row.branch == branch for row in rows)]
    handles = [Line2D([], [], color=COLORS[branch],
                      marker=PIN_MARKERS[branch][0],
                      markersize=PIN_MARKERS[branch][1],
                      markerfacecolor="white", linewidth=1.5,
                      label=PIN_TITLES[branch]) for branch in present]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles),
               bbox_to_anchor=(0.5, 0.964), frameon=False)
    fig.suptitle(f"{cluster} replica 1, $J_2/J_1={J2:.2f}$: fixed-field $1/D$ trends",
                 fontsize=14, y=0.995)
    fig.text(0.5, 0.018,
             r"Sorted $C_1\leq C_2\leq C_3$ (more negative = stronger AF bond). "
             "Only $h=0$ energies compare the same Hamiltonian.",
             ha="center", fontsize=9)
    fig.subplots_adjust(left=0.085, right=0.98, bottom=0.075,
                        top=0.895, wspace=0.22, hspace=0.30)
    save(fig, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kuma", type=Path,
                        default=DEFAULT_BUNDLE / "Results_VBC_three")
    parser.add_argument("--izar", type=Path,
                        default=DEFAULT_BUNDLE / "Results_VBC_branches")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--csv-output-dir", type=Path, default=DEFAULT_CSV_OUTPUT,
        help="processed-data root; writes cluster/J2/pinning-source/*.csv",
    )
    args = parser.parse_args()

    # Validate both input trees before writing anything.  In particular, a
    # half-copied observation is a fatal error, never a point to skip.
    validated: list[tuple[str, list]] = []
    for cluster, root in (("Kuma", args.kuma), ("Izar", args.izar)):
        if not root.is_dir():
            raise FileNotFoundError(f"{cluster} input directory is missing: {root}")
        stages = [stage for stage in select_highest_chi(discover(root, strict=True))
                  if stage.replica == 1]
        if not stages:
            raise RuntimeError(f"{cluster}: no readable replica-1 observations in {root}")
        validated.append((cluster, stages))

    total = 0
    for cluster, stages in validated:
        by_J2: dict[float, list[RankedStage]] = defaultdict(list)
        for stage in stages:
            by_J2[stage.J2].append(rank_stage(stage, cluster))
        for J2, rows in sorted(by_J2.items()):
            folder = args.output_dir / cluster / f"J2_{J2:.2f}".replace(".", "p")
            plot_nn_vs_h(rows, cluster, J2, folder / "01_sorted_nn_vs_h.pdf")
            plot_observables_vs_inverse_D(
                rows, cluster, J2, folder / "02_energy_delta_omega_vs_inverse_D.pdf")
            csv_folder = (args.csv_output_dir / cluster
                          / f"J2_{J2:.2f}".replace(".", "p"))
            csv_count = write_source_csvs(rows, csv_folder)
            print(f"{cluster:4s} J2={J2:.2f}: {len(rows)} replica-1 stages; "
                  f"two figures in {folder}; {csv_count} CSV files in {csv_folder}")
            total += len(rows)
    print(f"Total plotted replica-1 stages: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
