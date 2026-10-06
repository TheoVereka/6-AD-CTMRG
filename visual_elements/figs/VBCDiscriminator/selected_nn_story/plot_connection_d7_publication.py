#!/usr/bin/env python3
"""Render the D=7 A-I connection paths as one publication-style four-panel figure."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


HERE = Path(__file__).resolve().parent
FIGS = HERE.parents[1]
REPO = HERE.parents[3]
STYLE = FIGS / "PublicationPlots" / "plottingStyle" / "everyday_stylesheet.mplstyle"
DEFAULT_DATA = REPO.parent / "data" / "connectionPlaqDimer_trees_20261003"
DEFAULT_OUTPUT = HERE / "pubPlots" / "D7_connection_adiabatic_four_panel"
CHAIN = "ABCDEFGHI"
COLORS = ("#b2182b", "#238b45", "#2166ac")  # strong, middle, weak
RANK_MARKERS = ("o", "s", "^")
RANK_LINESTYLES = ("-", "--", ":")

sys.path.insert(0, str(FIGS / "connectionPlaqDimer_trees_20261003"))
from plot_connection_energy import load_baselines, load_nn_groups, load_points  # noqa: E402


def collect(data_root: Path):
    points, warnings = load_points(data_root)
    selected = [point for point in points if point.D == 7 and point.node in CHAIN]
    for connection in ("dimer", "plaq"):
        group = [point for point in selected if point.connection == connection]
        if len(group) != 9 or {point.node for point in group} != set(CHAIN):
            raise ValueError(f"Expected all nine D=7 {connection} A-I stages; found {len(group)}")
        if any(not point.completed_stage for point in group):
            raise ValueError(f"D=7 {connection} A-I has an unfinished stage")
    nn_groups, nn_warnings = load_nn_groups(selected, data_root)
    if len(nn_groups) != 18:
        raise ValueError(f"Expected 18 D=7 NN triplets; got {len(nn_groups)}. "
                         + "; ".join(nn_warnings))
    baseline = load_baselines(data_root.parent / "0713summary")[7]
    for warning in warnings:
        print(f"Warning: {warning}")
    return selected, nn_groups, baseline


def render(points, nn_groups, baseline: float, output_stem: Path) -> None:
    if not STYLE.is_file():
        raise FileNotFoundError(STYLE)
    plt.style.use(str(STYLE))
    matplotlib.rcParams.update({
        "legend.fontsize": 16,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "axes.titlepad": 0.0,
    })

    fig, axes = plt.subplots(2, 2, figsize=(15.0, 9.4), sharex=True)
    fig.subplots_adjust(left=0.105, right=0.985, top=0.977, bottom=0.245,
                        wspace=0.12, hspace=0.16)
    percent_ticks = [12.5 * index for index in range(9)]
    percent_labels = [f"{value:g}\\%" for value in percent_ticks]
    corr_values = [value for triple in nn_groups.values() for value in triple]
    corr_span = max(corr_values) - min(corr_values)
    corr_pad = max(0.035 * corr_span, 0.015)
    corr_ylim = (min(corr_values) - corr_pad, max(corr_values) + corr_pad)

    for col, (connection, sector_label) in enumerate((
        ("dimer", "dimer"), ("plaq", "plaquette"),
    )):
        by_node = {point.node: point for point in points if point.connection == connection}
        ordered = [by_node[node] for node in CHAIN]
        percent = [point.x_percent for point in ordered]
        local_energies = [point.energy_per_site for point in ordered] + [baseline]
        energy_pad = max(0.05 * (max(local_energies) - min(local_energies)), 0.0015)

        energy_ax = axes[0, col]
        energy_ax.plot(percent, [point.energy_per_site for point in ordered],
                       color="#222222", marker="o", markersize=6.2,
                       linewidth=2.25, zorder=3)
        energy_ax.axhline(baseline, color="0.43", linestyle=(0, (6, 3)),
                          linewidth=2.0, zorder=1)
        energy_ax.set_ylim(min(local_energies) - energy_pad,
                           max(local_energies) + energy_pad)
        if col == 0:
            energy_ax.set_ylabel(r"$E$ per site")
        label_y = 0.13 if col == 0 else 0.90
        energy_ax.text(0.04, label_y, sector_label + r"  $D=7$",
                       transform=energy_ax.transAxes,
                       va="center", ha="left", fontsize=19)
        if col == 0:
            energy_ax.legend(
                [Line2D([], [], color="0.43", linestyle=(0, (6, 3)), linewidth=2)],
                ["original 2C3"], loc="upper right", fontsize=16,
                handlelength=2.3, borderaxespad=0.5,
            )

        corr_ax = axes[1, col]
        for rank, color in enumerate(COLORS):
            values = [sorted(nn_groups[point.job_id])[rank] for point in ordered]
            corr_ax.plot(percent, values, color=color,
                         marker=RANK_MARKERS[rank], markersize=6.2,
                         linestyle=RANK_LINESTYLES[rank],
                         linewidth=2.25, zorder=3 + rank)
        corr_ax.set_ylim(*corr_ylim)
        if col == 0:
            corr_ax.set_ylabel(r"nearest-neighbour correlation $\langle\mathbf{S}_i\cdot\mathbf{S}_j\rangle$")
        if col == 1:
            corr_ax.legend(
                [Line2D([], [], color=color, marker=RANK_MARKERS[rank],
                        linestyle=RANK_LINESTYLES[rank], linewidth=2.25,
                        markersize=6.2) for rank, color in enumerate(COLORS)],
                ["strong", "middle", "weak"], loc="center",
                bbox_to_anchor=(0.50, 0.65),
                fontsize=16, handlelength=1.8, labelspacing=0.2,
                borderaxespad=0.4,
            )

        for ax in (energy_ax, corr_ax):
            ax.set_xlim(-3, 103)
            ax.set_xticks(percent_ticks, percent_labels, rotation=45, ha="right")
            ax.tick_params(which="both", top=True, right=True, labelsize=19)
            ax.tick_params(axis="x", labelbottom=(ax is corr_ax))
            ax.grid(False)
        if col == 1:
            corr_ax.tick_params(axis="y", labelleft=False)

    fig.text(0.55, 0.115, r"tuning parameter, $t$", ha="center",
             va="center", fontsize=22)
    fig.text(
        0.55, 0.035,
        r"$J_1=1$ on dimer/plaquette singlet NN bonds, "
        r"$J_1=t$ on non-singlet NN bonds, $J_2=0.32t$ on NNN bonds",
        ha="center", va="center", fontsize=18,
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
    fig.savefig(output_stem.with_suffix(".png"), dpi=220,
                bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    points, nn_groups, baseline = collect(args.data_root)
    render(points, nn_groups, baseline, args.output_stem)
    print(f"D=7: {len(points)} A-I stages, {len(nn_groups)} NN triplets; "
          f"original 2C3 E/site = {baseline:.12f}")
    print(args.output_stem.with_suffix(".pdf"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
