#!/usr/bin/env python3
"""Render the final selected-VBC figures with the PublicationPlots style.

This module performs no data selection and no physical fitting.  It consumes
the CSV products written by ``select_and_plot.py`` and
``pinning_energy_phase_boundary.py`` so the publication layer cannot alter
the scientific pipeline.  Its default output is the sibling ``pubPlots``
directory and therefore never overwrites the analysis figures in ``plots``.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.lines import Line2D
import numpy as np
from scipy.interpolate import griddata


HERE = Path(__file__).resolve().parent
VBC_DIR = HERE.parent
FIGS_DIR = VBC_DIR.parent
REPO = HERE.parents[3]
DATA = REPO.parent / "data"
ANALYSIS = HERE / "plots"
PHASE = ANALYSIS / "pinning_energy_phase_boundary"
FIGURE17_DATA = DATA / "processed" / "publicationPlots" / "figure_17_data.csv"
DEFAULT_OUTPUT = HERE / "pubPlots"
STYLE = FIGS_DIR / "PublicationPlots" / "plottingStyle" / "everyday_stylesheet.mplstyle"

sys.path.insert(0, str(HERE))
import pinning_energy_phase_boundary as phase  # noqa: E402
import select_and_plot as selected_story  # noqa: E402


DOUBLE_FIGSIZE = (13.3, 5.2)
NN_DOUBLE_FIGSIZE = (13.3, 7.8)
HC_COMPARISON_FIGSIZE = (15.2, 5.4)
SINGLE_FIGSIZE = (6.65, 5.2)
COLORBAR_FIGSIZE = (7.85, 5.2)
PHASE_FIGSIZE = (10.6, 6.0)
VIDEO_FIGSIZE = (8.8, 6.6)

YELLOW = np.asarray((190, 190, 0), dtype=float) / 255.0
CYAN = np.asarray((0, 190, 190), dtype=float) / 255.0
PURPLE = "#6b2f8a"
GREEN = "#00d000"
RANK_FIELDS = ("strongest", "middle", "weakest")
TEXTURES = ("dimer-plaquette", "plaquette")
TEXTURE_TEXT = {
    "dimer-plaquette": "dimer-plaquette",
    "plaquette": "plaquette",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required analysis product is missing: {path}\n"
            "Run select_and_plot.py and pinning_energy_phase_boundary.py first."
        )
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def finite(value: str | float | int | None) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=1.0e-10)


def apply_publication_style() -> None:
    if not STYLE.is_file():
        raise FileNotFoundError(f"Publication stylesheet is missing: {STYLE}")
    plt.style.use(str(STYLE))
    # The canonical stylesheet fixes the science-panel typography at 22 pt.
    # Dense legends and colorbars are the only deliberately smaller text.
    matplotlib.rcParams.update({
        "legend.fontsize": 14,
        "legend.frameon": False,
        "axes.titlepad": 0.0,
        "pdf.fonttype": 42,
    })


def save_pdf(figure: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight", pad_inches=0.04)
    plt.close(figure)


def selected_rows() -> list[dict[str, str]]:
    rows = read_csv(ANALYSIS / "selected_nn_data.csv")
    return [row for row in rows if row.get("selected", "").lower() == "true"]


def selected_objects(rows: list[dict[str, str]]) -> dict[tuple[str, int, float], SimpleNamespace]:
    objects = {}
    for row in rows:
        texture = row["texture"]
        D = int(row["D"])
        J2 = float(row["J2"])
        ranks = tuple(
            (float(row[name]), float(row[f"{name}_error"]))
            for name in RANK_FIELDS
        )
        objects[(texture, D, J2)] = SimpleNamespace(
            texture=texture, D=D, J2=J2, ranks=ranks,
            delta=float(row["delta"]),
            delta_error=float(row["delta_error"]),
            energy=float(row["energy"]),
        )
    return objects


def dimension_encoding(D: int) -> tuple[float, float]:
    """Compressed absolute-D marker scale and original analysis linewidth."""
    fraction = np.clip((D - 5) / (11 - 5), 0.0, 1.0)
    # D=5 keeps its current size; D=11 is exactly 70% of its old 9.2 size.
    marker_size = 3.2 + fraction * (0.70 * 9.2 - 3.2)
    return float(marker_size), 1.05


NN_RANK_GRADIENTS = (
    # Brown-yellow / green / purple rank families.  Low/high endpoints shift
    # hue as well as brightness, and the central colors anchor middle-high D.
    mcolors.LinearSegmentedColormap.from_list(
        "nn_brown_yellow_D",
        [(0.0, "#FFD447"), (0.62, "#D89000"), (1.0, "#8A4F00")],
    ),
    mcolors.LinearSegmentedColormap.from_list(
        "nn_green_D", [(0.0, "#67D66F"), (0.62, "#00A65A"), (1.0, "#006B3C")],
    ),
    mcolors.LinearSegmentedColormap.from_list(
        "nn_purple_D", [(0.0, "#C067E8"), (0.62, "#7B2CBF"), (1.0, "#3C178F")],
    ),
)


def rank_D_color(rank: int, D: int, dimensions: list[int]) -> np.ndarray:
    fraction = ((D - dimensions[0]) / max(1, dimensions[-1] - dimensions[0]))
    return NN_RANK_GRADIENTS[rank](fraction)


def add_sector_text(axis: plt.Axes, texture: str) -> None:
    axis.text(
        0.04, 0.95, TEXTURE_TEXT[texture], transform=axis.transAxes,
        ha="left", va="top", fontsize=19,
    )


def plot_nn_vs_j2(rows: list[dict[str, str]], output: Path) -> None:
    figure, axes = plt.subplots(
        1, 2, figsize=NN_DOUBLE_FIGSIZE, sharex=True, sharey=True,
        constrained_layout=True,
    )
    all_dimensions = sorted({int(row["D"]) for row in rows})
    for axis, texture in zip(axes, TEXTURES):
        local = [row for row in rows if row["texture"] == texture]
        dimensions = sorted({int(row["D"]) for row in local})
        for D in dimensions:
            subset = sorted(
                [row for row in local if int(row["D"]) == D],
                key=lambda row: float(row["J2"]),
            )
            marker_size, linewidth = dimension_encoding(D)
            alpha = 0.08 + 0.92 * (
                (D - all_dimensions[0])
                / max(1, all_dimensions[-1] - all_dimensions[0])
            )
            for rank, field in enumerate(RANK_FIELDS):
                color = rank_D_color(rank, D, all_dimensions)
                axis.errorbar(
                    [float(row["J2"]) for row in subset],
                    [float(row[field]) for row in subset],
                    yerr=[float(row[f"{field}_error"]) for row in subset],
                    color=color, alpha=alpha,
                    marker=selected_story.RANK_MARKERS[rank],
                    markerfacecolor=color,
                    markeredgecolor=color,
                    markeredgewidth=0.8,
                    markersize=marker_size,
                    linestyle=selected_story.RANK_LINESTYLES[rank],
                    linewidth=linewidth, elinewidth=0.75,
                    capsize=2.0, zorder=2 + D,
                )
        add_sector_text(axis, texture)
        axis.set_xlabel(r"$J_2$")
        axis.tick_params(which="both", top=True, right=True)
    axes[0].set_ylabel(r"nearest-neighbour correlation")

    # Interleave ranks within each D because Matplotlib packs a multi-column
    # legend column-major.  The visible rows are therefore weak, middle,
    # strong, each containing the complete D sequence.
    handles, labels = [], []
    rank_order = (2, 1, 0)
    rank_labels = {2: "weak", 1: "middle", 0: "strong"}
    for D in all_dimensions:
        marker_size, linewidth = dimension_encoding(D)
        alpha = 0.08 + 0.92 * (
            (D - all_dimensions[0])
            / max(1, all_dimensions[-1] - all_dimensions[0])
        )
        for rank in rank_order:
            color = rank_D_color(rank, D, all_dimensions)
            handle = Line2D(
                [], [], color=color, alpha=alpha,
                marker=selected_story.RANK_MARKERS[rank],
                linestyle=selected_story.RANK_LINESTYLES[rank],
                markerfacecolor=color, markeredgecolor=color,
                markeredgewidth=0.8, markersize=marker_size,
                linewidth=linewidth,
            )
            handles.append(handle)
            prefix = rf"{rank_labels[rank]}, " if D == all_dimensions[0] else ""
            labels.append(prefix + rf"$D={D}$")
    figure.legend(
        handles, labels, loc="outside upper center", ncol=len(all_dimensions),
        fontsize=12.5, handlelength=1.45, columnspacing=0.72,
        handletextpad=0.45, labelspacing=0.38,
    )
    save_pdf(figure, output)


def plot_delta_vs_j2(
    rows: list[dict[str, str]], selection: dict, output: Path,
) -> None:
    fixed_a = selected_story.load_fixed_a()
    selected_values = list(selection.values())
    extrapolated: dict[str, list[dict[str, float]]] = {
        texture: [] for texture in TEXTURES
    }
    for texture in TEXTURES:
        for J2 in selected_story.J2_GRID:
            if J2 not in fixed_a:
                continue
            subset = [
                row for row in selected_values
                if row.texture == texture and close(row.J2, J2)
            ]
            fit = selected_story.stable_high_D_rank_fit(subset, fixed_a[J2])
            if fit is None:
                continue
            correlations = fit["correlations"]
            extrapolated[texture].append({
                "J2": J2,
                "Delta": float(correlations[2] - correlations[0]),
                "Delta_error": float(fit["Delta_error"]),
            })
    figure, axes = plt.subplots(
        1, 2, figsize=DOUBLE_FIGSIZE, sharex=True, sharey=True,
        constrained_layout=True,
    )
    palettes = {
        "dimer-plaquette": plt.get_cmap("YlOrRd"),
        "plaquette": plt.get_cmap("PuBu"),
    }
    for axis, texture in zip(axes, TEXTURES):
        local = [row for row in rows if row["texture"] == texture]
        dimensions = sorted({int(row["D"]) for row in local})
        for index, D in enumerate(dimensions):
            subset = sorted(
                [row for row in local if int(row["D"]) == D],
                key=lambda row: float(row["J2"]),
            )
            fraction = 0.30 + 0.68 * index / max(1, len(dimensions) - 1)
            alpha = 0.25 + 0.75 * index / max(1, len(dimensions) - 1)
            axis.errorbar(
                [float(row["J2"]) for row in subset],
                [float(row["delta"]) for row in subset],
                yerr=[float(row["delta_error"]) for row in subset],
                color=palettes[texture](fraction), alpha=alpha,
                marker="o", markersize=5.8, linestyle="-",
                linewidth=1.65, elinewidth=1.1, capsize=2.5,
                label=rf"$D={D}$",
            )
        axis.errorbar(
            [row["J2"] for row in extrapolated[texture]],
            [row["Delta"] for row in extrapolated[texture]],
            yerr=[row["Delta_error"] for row in extrapolated[texture]],
            color="black", marker="D", markerfacecolor="white",
            markeredgewidth=1.5, markersize=6.2, linewidth=2.4,
            elinewidth=1.35, capsize=3.0,
            label=(r"extrapolated dimer-plaquette $\Delta$"
                   if texture == "dimer-plaquette"
                   else r"extrapolated plaquette $\Delta$"),
            zorder=10,
        )
        add_sector_text(axis, texture)
        axis.set_xlabel(r"$J_2$")
        axis.set_ylim(bottom=0.0)
        axis.tick_params(which="both", top=True, right=True)
        axis.legend(
            loc="lower right", fontsize=12, ncol=2,
            handlelength=1.55, columnspacing=0.8,
        )
    axes[0].set_ylabel(r"$\Delta=C_{\rm weak}-C_{\rm strong}$")
    save_pdf(figure, output)


def crossing_rows() -> list[dict[str, str]]:
    rows = read_csv(PHASE / "hc_D_infinity.csv")
    return [row for row in rows if math.isfinite(finite(row.get("h_c")))]


def plot_hc_phase_boundary(
    rows: list[dict[str, str]], texture_candidates: list[dict[str, str]],
    output: Path,
) -> None:
    ordered = sorted(rows, key=lambda row: float(row["J2"]))
    h = np.asarray([float(row["h_c"]) for row in ordered])
    J2 = np.asarray([float(row["J2"]) for row in ordered])
    lower = np.asarray([float(row["h_c_error_low"]) for row in ordered])
    upper = np.asarray([float(row["h_c_error_high"]) for row in ordered])

    figure = plt.figure(figsize=HC_COMPARISON_FIGSIZE)
    grid = figure.add_gridspec(
        1, 2, width_ratios=(0.92, 1.28),
        left=0.065, right=0.945, bottom=0.17, top=0.96, wspace=0.14,
    )
    axis = figure.add_subplot(grid[0])
    crossing_color = "black"
    axis.plot(h, J2, color=crossing_color, linewidth=2.2, zorder=3)
    axis.errorbar(
        h, J2, xerr=np.vstack((lower, upper)), color=crossing_color,
        linestyle="none", marker="o", markersize=6.5,
        elinewidth=1.8, capsize=3.0, zorder=4,
        label=r"energy crossing $h_c$",
    )
    axis.plot(
        [0.0, 0.0], [0.24, 0.275], color=GREEN, linewidth=7.0,
        solid_capstyle="butt", zorder=2,
        label=r"QSL at $h=0$",
    )
    limit = max(1.0e-3, 1.18 * max(
        abs(value) + error
        for value, error in zip(h, np.maximum(lower, upper))
    ))
    axis.set_xlim(-limit, limit)
    axis.set_ylim(0.235, 0.345)
    axis.set_xlabel(r"crossing field $h_c$")
    axis.set_ylabel(r"$J_2$")
    axis.tick_params(which="both", top=True, right=True)
    axis.legend(loc="best", fontsize=14)

    # Right panel: reproduce publication figure 17 from its immutable exported
    # data, then add the two independently extrapolated h=0 VBC sectors.
    figure17 = read_csv(FIGURE17_DATA)
    m_rows = sorted([
        row for row in figure17
        if (row["series"] == "m_extrap"
            and 0.20 <= float(row["J2"]) <= 0.32)
    ], key=lambda row: float(row["J2"]))
    old_delta_rows = sorted([
        row for row in figure17
        if (row["series"] == "delta_extrap"
            and 0.20 <= float(row["J2"]) <= 0.32)
    ], key=lambda row: float(row["J2"]))
    h0_sector_rows = {
        branch: sorted([
            row for row in texture_candidates
            if row["branch"] == branch
            and close(float(row["signed_h"]), 0.0)
            and float(row["J2"]) <= 0.32 + 1.0e-12
        ], key=lambda row: float(row["J2"]))
        for branch in TEXTURES
    }

    comparison_left = figure.add_subplot(grid[1])
    comparison_right = comparison_left.twinx()
    fig17_purple, fig17_orange = "#6f2dbd", "#d95f02"
    comparison_left.errorbar(
        [float(row["J2"]) for row in m_rows],
        [float(row["y"]) for row in m_rows],
        yerr=[float(row["y_error"]) for row in m_rows],
        color=fig17_purple, marker="s", linestyle="-",
        linewidth=2.2, markersize=5.8, elinewidth=1.4, capsize=2.7,
        label=r"$m_{\rm extrap}$", zorder=4,
    )
    # Keep J2=.27 as the fully opaque endpoint of the solid segment and the
    # starting point of the translucent dashed segment.  Draw the latter
    # first so the shared marker remains fully opaque.
    for segment, linestyle, alpha, label in (
        ([row for row in old_delta_rows if float(row["J2"]) >= 0.27],
         "--", 0.40, "_nolegend_"),
        ([row for row in old_delta_rows if float(row["J2"]) <= 0.27],
         "-", 1.00, r"original 2C3 $\Delta_{\rm extrap}$"),
    ):
        comparison_right.errorbar(
            [float(row["J2"]) for row in segment],
            [float(row["y"]) for row in segment],
            yerr=[float(row["y_error"]) for row in segment],
            color=fig17_orange, alpha=alpha, marker="s",
            linestyle=linestyle, linewidth=2.2, markersize=5.8,
            elinewidth=1.4, capsize=2.7, label=label, zorder=4,
        )
    for branch, color, marker, label in (
        ("dimer-plaquette", "#C62828", "v",
         r"dimer-plaquette $\Delta_{\rm extrap}(h=0)$"),
        ("plaquette", "#1565C0", "^",
         r"plaquette $\Delta_{\rm extrap}(h=0)$"),
    ):
        sector_rows = h0_sector_rows[branch]
        comparison_right.errorbar(
            [float(row["J2"]) for row in sector_rows],
            [float(row["extrapolated_Delta"]) for row in sector_rows],
            yerr=[float(row["Delta_error"]) for row in sector_rows],
            color=color, marker=marker, markerfacecolor=color,
            linestyle="-", linewidth=1.70, markersize=6.4,
            elinewidth=1.20, capsize=2.8,
            label=label, zorder=6,
        )
    for transition, label in ((0.24, r"$J_2=0.24$"),
                              (0.27, r"$J_2=0.27$")):
        comparison_left.axvline(
            transition, color="0.35", linestyle="--", linewidth=1.4,
            zorder=1,
        )
        comparison_left.text(
            transition - 0.0022, 0.965, label,
            transform=comparison_left.get_xaxis_transform(), rotation=90,
            ha="right", va="top", color="0.28", fontsize=15,
        )
    comparison_left.set_xlim(0.195, 0.325)
    comparison_left.set_ylim(bottom=0.0)
    comparison_right.set_ylim(bottom=0.0)
    comparison_left.set_xlabel(r"$J_2$")
    comparison_left.set_ylabel(r"$m$", color=fig17_purple)
    comparison_right.set_ylabel(r"$\Delta$", color=fig17_orange)
    comparison_left.tick_params(axis="y", colors=fig17_purple)
    comparison_right.tick_params(axis="y", colors=fig17_orange)
    comparison_left.tick_params(which="both", top=True)
    handles_left, labels_left = comparison_left.get_legend_handles_labels()
    handles_right, labels_right = comparison_right.get_legend_handles_labels()
    comparison_left.legend(
        handles_left + handles_right, labels_left + labels_right,
        loc="lower right", fontsize=11.5,
    )
    save_pdf(figure, output)


def plot_phase_diagram(
    crossings: list[dict[str, str]], texture_rows: list[dict[str, str]],
    output: Path,
) -> None:
    h_values = np.asarray([float(row["signed_h"]) for row in texture_rows])
    j_values = np.asarray([float(row["J2"]) for row in texture_rows])
    q_values = np.asarray([float(row["q"]) for row in texture_rows])
    delta_values = np.asarray([
        float(row["extrapolated_Delta"]) for row in texture_rows
    ])
    h_centers = np.asarray(sorted(set(h_values)), dtype=float)
    x_centers = np.asarray(phase.field_plot_coordinate(h_centers), dtype=float)
    j_centers = np.asarray(sorted(set(j_values)), dtype=float)
    H, J = np.meshgrid(x_centers, j_centers)
    points = np.column_stack((phase.field_plot_coordinate(h_values), j_values))
    q_grid = griddata(points, q_values, (H, J), method="nearest")
    delta_grid = griddata(points, delta_values, (H, J), method="nearest")
    delta_max = math.ceil(float(np.nanmax(delta_values)) * 20.0) / 20.0
    rgb = phase.texture_rgb(q_grid, delta_grid, delta_max)
    h_edges = phase.centered_edges(
        x_centers, phase.field_plot_coordinate(-1.0e-1),
        phase.field_plot_coordinate(1.0e-1),
    )
    j_edges = phase.centered_edges(j_centers)
    truncated_at_032 = float(np.max(j_centers)) <= 0.32 + 1.0e-12
    texture_label_y = 0.315 if truncated_at_032 else 0.334

    figure = plt.figure(figsize=PHASE_FIGSIZE)
    grid = figure.add_gridspec(
        1, 3, width_ratios=(12.0, 0.55, 0.55),
        left=0.09, right=0.95, bottom=0.14, top=0.96, wspace=0.27,
    )
    axis = figure.add_subplot(grid[0])
    hue_axis = figure.add_subplot(grid[1])
    delta_axis = figure.add_subplot(grid[2])
    cell_count = rgb.shape[0] * rgb.shape[1]
    cell_cmap = mcolors.ListedColormap(rgb.reshape(cell_count, 3))
    cell_norm = mcolors.BoundaryNorm(
        np.arange(cell_count + 1) - 0.5, cell_count,
    )
    axis.pcolormesh(
        h_edges, j_edges, np.arange(cell_count).reshape(rgb.shape[:2]),
        cmap=cell_cmap, norm=cell_norm, shading="flat",
        antialiased=False, rasterized=True, zorder=0,
    )

    ordered = sorted(crossings, key=lambda row: float(row["J2"]))
    hc = np.asarray([float(row["h_c"]) for row in ordered])
    hc_x = np.asarray(phase.field_plot_coordinate(hc))
    crossing_J2 = np.asarray([float(row["J2"]) for row in ordered])
    boundary, = axis.plot(
        hc_x, crossing_J2, color="#fff176", linewidth=2.4,
        marker="o", markersize=5.5, zorder=5,
        label=r"energy crossing $h_c$",
    )
    boundary.set_path_effects([
        path_effects.Stroke(linewidth=4.2, foreground="black"),
        path_effects.Normal(),
    ])
    for row, value, ordinate in zip(ordered, hc, crossing_J2):
        left = phase.field_plot_coordinate(
            value - float(row["h_c_error_low"])
        )
        right = phase.field_plot_coordinate(
            value + float(row["h_c_error_high"])
        )
        axis.plot([left, right], [ordinate, ordinate], color="#fff176",
                  linewidth=1.6, zorder=4)
        axis.plot([left, left], [ordinate - 0.0007, ordinate + 0.0007],
                  color="#fff176", linewidth=1.4, zorder=4)
        axis.plot([right, right], [ordinate - 0.0007, ordinate + 0.0007],
                  color="#fff176", linewidth=1.4, zorder=4)
    axis.plot(
        [0.0, 0.0], [0.24, 0.275], color=GREEN, linewidth=7.0,
        solid_capstyle="butt", zorder=2, label=r"QSL at $h=0$",
    )
    axis.text(
        phase.field_plot_coordinate(-2.5e-2), texture_label_y,
        "dimer-plaquette", color="white", fontsize=16,
        fontweight="semibold", ha="center", va="center", zorder=3,
    )
    axis.text(
        phase.field_plot_coordinate(2.5e-2), texture_label_y,
        "plaquette", color="white", fontsize=16,
        fontweight="semibold", ha="center", va="center", zorder=3,
    )
    ticks_h = np.asarray([
        -1.0e-1, -1.0e-2, -1.0e-3, 0.0, 1.0e-3, 1.0e-2, 1.0e-1,
    ])
    axis.set_xticks(phase.field_plot_coordinate(ticks_h))
    axis.set_xticklabels([
        r"$-10^{-1}$", r"$-10^{-2}$", r"$-10^{-3}$", r"$0$",
        r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$",
    ], fontsize=15)
    # Point the two labels bordering zero away from the deliberately narrow
    # quarter-decade h=0 gap instead of letting centered text overlap there.
    axis.get_xticklabels()[2].set_ha("right")
    axis.get_xticklabels()[4].set_ha("left")
    axis.set_xlim(phase.field_plot_coordinate(-1.0e-1),
                  phase.field_plot_coordinate(1.0e-1))
    if truncated_at_032:
        # Keep the complete J2=.32 colour cell (whose upper edge is .325),
        # while ensuring the visible axis and tick labels stop at J2=.32.
        axis.set_ylim(0.238, 0.325)
        axis.set_yticks([0.24, 0.26, 0.28, 0.30, 0.32])
    else:
        axis.set_ylim(0.238, 0.343)
    axis.set_xlabel(r"signed pinning field $h$")
    axis.set_ylabel(r"$J_2$")
    axis.tick_params(axis="y", labelsize=18)
    axis.legend(loc="lower right", fontsize=12, framealpha=0.88)

    hue_cmap = mcolors.LinearSegmentedColormap.from_list(
        "dimer_purple_plaquette",
        [(0.92, 0.10, 0.10), (0.60, 0.16, 0.72), (0.10, 0.30, 1.00)],
    )
    hue_bar = matplotlib.colorbar.ColorbarBase(
        hue_axis, cmap=hue_cmap, norm=mcolors.Normalize(-1.0, 1.0),
        orientation="vertical", ticks=[-1.0, 0.0, 1.0],
    )
    hue_bar.ax.tick_params(labelsize=15)
    hue_bar.set_label(
        r"extrapolated $q=(C_{\rm weak}+C_{\rm strong}-2C_{\rm mid})/"
        r"(C_{\rm weak}-C_{\rm strong})$", fontsize=14,
    )
    brightness_cmap = mcolors.LinearSegmentedColormap.from_list(
        "delta_brightness", ["black", "white"],
    )
    delta_bar = matplotlib.colorbar.ColorbarBase(
        delta_axis, cmap=brightness_cmap,
        norm=mcolors.Normalize(vmin=0.0, vmax=delta_max),
        orientation="vertical", ticks=np.linspace(0.0, delta_max, 4),
    )
    delta_bar.ax.tick_params(labelsize=15)
    delta_bar.set_label(
        r"extrapolated $\Delta=C_{\rm weak}-C_{\rm strong}$", fontsize=16,
    )
    save_pdf(figure, output)


def plot_energy_vs_inverse_D(
    rows: list[dict[str, str]], texture: str,
    fits: dict[float, dict[str, float]], output: Path,
) -> None:
    local = [row for row in rows if row["texture"] == texture]
    J2_values = sorted({float(row["J2"]) for row in local})
    if not local:
        raise RuntimeError(f"No selected rows for {texture}")
    missing = [J2 for J2 in J2_values if J2 not in fits]
    if missing:
        raise RuntimeError(f"Missing original-2C3 gapped fits at J2={missing}")

    figure = plt.figure(figsize=COLORBAR_FIGSIZE)
    width, height = COLORBAR_FIGSIZE
    axis = figure.add_axes([1.15 / width, 0.85 / height, 5.0 / width, 4.0 / height])
    color_axis = figure.add_axes([6.40 / width, 0.85 / height, 0.22 / width, 4.0 / height])
    base = plt.get_cmap("Reds" if texture == "dimer-plaquette" else "Blues")
    palette = mcolors.LinearSegmentedColormap.from_list(
        f"{texture}_J2", [base(0.95), base(0.38)], N=256,
    )
    norm = mcolors.Normalize(min(J2_values), max(J2_values))
    xmax = max(1.0 / int(row["D"]) for row in local)
    xline = np.linspace(0.0, xmax, 400)
    all_y = [float(row["energy"]) for row in local]
    for J2 in J2_values:
        subset = sorted(
            [row for row in local if close(float(row["J2"]), J2)],
            key=lambda row: int(row["D"]),
        )
        color = palette(norm(J2))
        axis.plot(
            [1.0 / int(row["D"]) for row in subset],
            [float(row["energy"]) for row in subset],
            linestyle="none", marker="o", markersize=6.0,
            color=color, zorder=4,
        )
        fit = fits[J2]
        curve = fit["E0"] + fit["k"] * np.exp(
            -fit["a"] / np.maximum(xline, 1.0e-15)
        )
        alpha = 0.78 - 0.53 * norm(J2)
        axis.plot(xline, curve, color="black", alpha=alpha,
                  linewidth=1.65, zorder=2)
        all_y.extend(curve.tolist())
    values = np.asarray(all_y)
    span = max(float(np.ptp(values)), 1.0e-5)
    axis.set_xlim(0.0, 1.035 * xmax)
    axis.set_ylim(float(np.min(values)) - 0.07 * span,
                  float(np.max(values)) + 0.07 * span)
    axis.set_xlabel(r"$1/D$")
    axis.set_ylabel(r"$E$")
    axis.tick_params(which="both", top=True, right=True)
    axis.legend(handles=[
        Line2D([], [], linestyle="none", marker="o", markersize=6,
               color=base(0.72), label=r"selected $h=0$ tensors"),
        Line2D([], [], color="black", alpha=0.48, linewidth=2,
               label=r"original 2C3 gapped fit"),
    ], loc="best", fontsize=13)
    scalar = plt.cm.ScalarMappable(norm=norm, cmap=palette)
    scalar.set_array([])
    colorbar = figure.colorbar(scalar, cax=color_axis)
    colorbar.set_label(r"$J_2$")
    colorbar.set_ticks(J2_values)
    colorbar.set_ticklabels([f"{J2:g}" for J2 in J2_values])
    colorbar.ax.tick_params(labelsize=15)
    save_pdf(figure, output)


def plot_all_good_video(rows: list[dict[str, str]], output: Path) -> None:
    if not FFMpegWriter.isAvailable():
        raise RuntimeError(
            "Matplotlib cannot find ffmpeg on PATH; publication MP4 was not written."
        )
    dimensions = sorted({int(row["D"]) for row in rows})
    norm = mcolors.Normalize(min(dimensions), max(dimensions))
    cmap = plt.get_cmap("viridis")
    figure = plt.figure(figsize=VIDEO_FIGSIZE)
    axis = figure.add_subplot(111, projection="3d")
    axis.computed_zorder = False

    reference = sorted(
        (J2, energy) for (J2, D), energy in phase.load_original_energies().items()
        if D == 10 and 0.24 - 1.0e-12 <= J2 <= 0.34 + 1.0e-12
    )
    solid = [(J2, energy) for J2, energy in reference
             if J2 <= 0.275 + 1.0e-12]
    dashed = [(J2, energy) for J2, energy in reference
              if J2 >= 0.275 - 1.0e-12]
    for segment, linestyle, color in (
        (solid, "-", "#c62828"), (dashed, "--", "0.10"),
    ):
        axis.plot(
            [item[0] for item in segment], [0.0] * len(segment),
            [item[1] for item in segment], color=color,
            linestyle=linestyle, linewidth=3.0, alpha=0.92, zorder=0,
        )
    for D in dimensions:
        alpha = 0.24 + 0.72 * norm(D)
        color = cmap(norm(D))
        subset = [row for row in rows if int(row["D"]) == D]
        axis.scatter(
            [float(row["J2"]) for row in subset],
            [float(row["signed_h"]) for row in subset],
            [float(row["energy"]) for row in subset],
            color=[color], alpha=alpha, s=10, marker="o",
            linewidths=0.0, depthshade=False, zorder=2,
        )
        keys = sorted({(float(row["J2"]), row["branch"]) for row in subset})
        for J2, branch in keys:
            samples: dict[float, list[float]] = defaultdict(list)
            for row in subset:
                if close(float(row["J2"]), J2) and row["branch"] == branch:
                    samples[round(float(row["signed_h"]), 12)].append(
                        float(row["energy"])
                    )
            if len(samples) < 2:
                continue
            fields = sorted(samples)
            energies = [float(np.median(samples[field])) for field in fields]
            axis.plot(
                [J2] * len(fields), fields, energies, color=color,
                alpha=alpha, linewidth=1.0, zorder=1,
            )
    axis.set_xlabel(r"$J_2$", labelpad=13)
    axis.set_ylabel(r"signed pinning field $h$", labelpad=15)
    axis.set_zlabel(r"$E$", labelpad=12)
    axis.tick_params(axis="both", which="major", labelsize=16)
    axis.view_init(elev=24, azim=-58)
    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    colorbar = figure.colorbar(scalar, ax=axis, pad=0.10, shrink=0.70)
    colorbar.set_label(r"$D$")
    colorbar.set_ticks(dimensions)
    colorbar.ax.tick_params(labelsize=16)
    axis.legend(handles=[
        Line2D([], [], color="#c62828", linewidth=3.0,
               label=r"original 2C3 $D=10$, $J_2\leq0.275$"),
        Line2D([], [], color="0.10", linewidth=3.0, linestyle="--",
               label=r"original 2C3 $D=10$, $J_2\geq0.275$"),
    ], loc="upper left", fontsize=13)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.tmp{output.suffix}")
    azimuths = np.linspace(-58.0, -238.0, 61)

    def rotate(frame: int) -> tuple:
        axis.view_init(elev=24, azim=float(azimuths[frame]))
        return (axis,)

    animation = FuncAnimation(
        figure, rotate, frames=len(azimuths), interval=100, blit=False,
    )
    animation.save(
        temporary,
        writer=FFMpegWriter(
            fps=10, bitrate=3000,
            metadata={"title": "all-good pinning-energy surface"},
        ),
        dpi=150,
    )
    temporary.replace(output)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--skip-video", action="store_true",
        help="render the six PDFs only (useful for fast style checks)",
    )
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    apply_publication_style()

    raw_selected = selected_rows()
    selection = selected_objects(raw_selected)
    crossings = crossing_rows()
    texture_rows = read_csv(PHASE / "texture_D_infinity_nodes.csv")
    texture_candidates = read_csv(
        PHASE / "texture_D_infinity_candidates.csv"
    )
    all_good = read_csv(PHASE / "all_good_points.csv")

    plot_delta_vs_j2(
        raw_selected, selection, output / "Delta_vs_J2_selected.pdf",
    )
    plot_nn_vs_j2(raw_selected, output / "NN_corr_vs_J2_selected.pdf")
    plot_hc_phase_boundary(
        crossings, texture_candidates,
        output / "02_hc_vs_J2_phase_boundary.pdf",
    )
    plot_phase_diagram(crossings, texture_rows, output / "04_phasediagram.pdf")
    plot_phase_diagram(
        [row for row in crossings if float(row["J2"]) <= 0.32 + 1.0e-12],
        [row for row in texture_rows
         if float(row["J2"]) <= 0.32 + 1.0e-12],
        output / "04_phasediagram_J2_le_0p32.pdf",
    )
    energy_fits = selected_story.load_gapped_energy_fits()
    plot_energy_vs_inverse_D(
        raw_selected, "dimer-plaquette", energy_fits,
        output / "energy_vs_inverse_D_all_J2_dimer.pdf",
    )
    plot_energy_vs_inverse_D(
        raw_selected, "plaquette", energy_fits,
        output / "energy_vs_inverse_D_all_J2_plaquette.pdf",
    )
    if not args.skip_video:
        plot_all_good_video(
            all_good, output / "03_all_good_points_z_rotation.mp4",
        )

    expected = [
        "Delta_vs_J2_selected.pdf", "NN_corr_vs_J2_selected.pdf",
        "02_hc_vs_J2_phase_boundary.pdf", "04_phasediagram.pdf",
        "04_phasediagram_J2_le_0p32.pdf",
        "energy_vs_inverse_D_all_J2_dimer.pdf",
        "energy_vs_inverse_D_all_J2_plaquette.pdf",
    ]
    if not args.skip_video:
        expected.append("03_all_good_points_z_rotation.mp4")
    missing = [name for name in expected if not (output / name).is_file()]
    if missing:
        raise RuntimeError(f"Publication outputs were not created: {missing}")
    print(f"Publication outputs: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
