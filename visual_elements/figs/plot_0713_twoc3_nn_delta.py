#!/usr/bin/env python3
"""Plot the 0713summary 2C3 NN ranks and Delta against 1/D and 1/xi."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "PublicationPlots"))
from publication_common import NN_GROUPS, inverse_xi, parse_observable, rms


J2_VALUES = (0.27, 0.275, 0.28, 0.29, 0.30, 0.31, 0.32, 0.33, 0.34)
RANK_COLORS = ("#b2182b", "#238b45", "#2166ac")
DELTA_COLOR = "#d95f02"
RANK_LABELS = (
    "group of strongest 3 NN corr",
    "group of 3 middle NN corr",
    "group of 3 weakest NN corr",
)
DELTA_LABEL = r"$\Delta:=|\mathrm{strongest}-\mathrm{weakest}|$"


def read_rows(data_root: Path) -> dict[float, list[dict]]:
    result = {}
    for j2 in J2_VALUES:
        label = str(j2).replace(".", "p").rstrip("0")
        folder = data_root / f"J2_{label}" / "2tensor_twoC3"
        rows = []
        for d_folder in folder.glob("D_*"):
            try:
                D = int(d_folder.name[2:])
            except ValueError:
                continue
            if D < 5:
                continue
            obs_path = d_folder / "energy_magnetization_correlation.txt"
            if not obs_path.is_file():
                continue
            obs = parse_observable(obs_path)
            groups = []
            for group in NN_GROUPS:
                values = [obs["corr"].get(key) for key in group]
                if any(value is None for value in values):
                    break
                groups.append((float(np.mean(values)), rms(values)))
            if len(groups) != 3:
                continue
            groups.sort(key=lambda pair: pair[0])
            dlt = abs(groups[-1][0] - groups[0][0])
            dlt_error = math.hypot(groups[-1][1], groups[0][1])
            xi_path = d_folder / "correlation_length.json"
            inv_xi = math.nan
            if xi_path.is_file():
                try:
                    inv_xi, _ = inverse_xi(xi_path, expected_j2=j2, expected_D=D)
                except (KeyError, TypeError, ValueError, OSError, ZeroDivisionError):
                    pass
            rows.append({"D": D, "ranks": groups, "delta": dlt,
                         "delta_error": dlt_error, "inverse_xi": inv_xi})
        result[j2] = sorted(rows, key=lambda row: row["D"])
    return result


def make_figure(data: dict[float, list[dict]], x_mode: str, output: Path) -> None:
    eligible = [row for rows in data.values() for row in rows
                if x_mode == "inverse_D" or math.isfinite(row["inverse_xi"])]
    if not eligible:
        raise ValueError(f"No 2C3 observations with {x_mode} found")

    nn_low = min(mean - error for row in eligible for mean, error in row["ranks"])
    nn_high = max(mean + error for row in eligible for mean, error in row["ranks"])
    nn_pad = max(0.005, 0.07 * (nn_high - nn_low))
    nn_limits = (nn_low - nn_pad, nn_high + nn_pad)
    delta_high = max(row["delta"] + row["delta_error"] for row in eligible)
    delta_limit = 1.12 * delta_high
    x_values = [1 / row["D"] if x_mode == "inverse_D" else row["inverse_xi"]
                for row in eligible]
    x_pad = max(0.002, 0.06 * (max(x_values) - min(x_values)))
    x_limits = (min(0.0, min(x_values) - x_pad), max(x_values) + x_pad)

    fig, axes = plt.subplots(3, 3, figsize=(15, 11.5), sharex=True, sharey=True,
                             constrained_layout=True)
    legend_handles = [None, None, None, None]
    for ax, j2 in zip(axes.flat, J2_VALUES):
        twin = ax.twinx()
        twin.set_zorder(ax.get_zorder() - 1)
        ax.patch.set_visible(False)
        rows = [row for row in data[j2]
                if x_mode == "inverse_D" or math.isfinite(row["inverse_xi"])]
        rows.sort(key=lambda row: 1 / row["D"] if x_mode == "inverse_D"
                  else row["inverse_xi"])
        if rows:
            x = np.array([1 / row["D"] if x_mode == "inverse_D"
                          else row["inverse_xi"] for row in rows])
            delta_handle = twin.errorbar(
                x, [row["delta"] for row in rows],
                yerr=[row["delta_error"] for row in rows],
                fmt="^--", color=DELTA_COLOR, markersize=5.0,
                linewidth=1.1, elinewidth=0.8, capsize=2,
                zorder=1, label=DELTA_LABEL,
            )
            if legend_handles[3] is None:
                legend_handles[3] = delta_handle
            for rank, color in enumerate(RANK_COLORS):
                rank_handle = ax.errorbar(
                    x, [row["ranks"][rank][0] for row in rows],
                    yerr=[row["ranks"][rank][1] for row in rows],
                    fmt="o-", color=color, markersize=3.8, linewidth=1.1,
                    elinewidth=0.8, capsize=2, zorder=3,
                    label=RANK_LABELS[rank],
                )
                if legend_handles[rank] is None:
                    legend_handles[rank] = rank_handle
        ax.set_title(rf"$J_2={j2:g}$", fontsize=12)
        ax.set_xlim(*x_limits)
        ax.set_ylim(*nn_limits)
        twin.set_ylim(0, delta_limit)
        ax.grid(alpha=0.2)
        ax.tick_params(axis="both", labelsize=9)
        twin.tick_params(axis="y", labelsize=9, colors=DELTA_COLOR)
        twin.spines["right"].set_color(DELTA_COLOR)
        if ax.get_subplotspec().is_first_col():
            ax.set_ylabel("NN correlation", fontsize=10)
        twin.set_ylabel(r"$\Delta$", fontsize=10, color=DELTA_COLOR)
        if ax.get_subplotspec().is_last_row():
            ax.set_xlabel(r"$1/D$" if x_mode == "inverse_D"
                          else r"$1/\xi=\ln|\lambda_{\max}/\lambda_2|$", fontsize=11)

    fig.legend(handles=legend_handles, labels=[*RANK_LABELS, DELTA_LABEL],
               loc="outside upper center", ncol=2,
               frameon=False, fontsize=10)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    print(f"Saved {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path,
                        default=HERE.parents[2] / "data" / "0713summary")
    parser.add_argument("--output-dir", type=Path,
                        default=HERE / "analysis_plots_0713summary")
    args = parser.parse_args()
    data = read_rows(args.data_root)
    make_figure(data, "inverse_D", args.output_dir / "2C3_NN_ranks_Delta_vs_inverse_D.pdf")
    make_figure(data, "inverse_xi", args.output_dir / "2C3_NN_ranks_Delta_vs_inverse_xi.pdf")


if __name__ == "__main__":
    main()
