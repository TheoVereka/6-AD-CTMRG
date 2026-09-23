#!/usr/bin/env python3
"""Plot the three ranked 2C3 NN correlations versus J2 for fixed D."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_0713_twoc3_nn_delta import J2_VALUES, RANK_COLORS, RANK_LABELS, read_rows


HERE = Path(__file__).resolve().parent
D_VALUES = (5, 6, 7, 8, 9, 10, 11)


def plot_fixed_D(data: dict, D: int, output: Path) -> None:
    points = []
    for j2 in J2_VALUES:
        row = next((item for item in data[j2] if item["D"] == D), None)
        points.append((j2, row))
    if not any(row is not None for _, row in points):
        raise ValueError(f"No data found for D={D}")

    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    x = list(J2_VALUES)
    for rank, color in enumerate(RANK_COLORS):
        ax.errorbar(
            x,
            [row["ranks"][rank][0] if row is not None else math.nan
             for _, row in points],
            yerr=[row["ranks"][rank][1] if row is not None else math.nan
                  for _, row in points],
            fmt="o-", color=color, markersize=3.8, linewidth=1.1,
            elinewidth=0.8, capsize=2, zorder=3,
            label=RANK_LABELS[rank],
        )
    ax.set_title(rf"2C3, $D={D}$", fontsize=13)
    ax.set_xlabel(r"$J_2$", fontsize=12)
    ax.set_ylabel("NN correlation", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{value:g}" for value in x])
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=9)
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
    for D in D_VALUES:
        plot_fixed_D(data, D, args.output_dir / f"2C3_NN_ranks_vs_J2_D{D}.pdf")


if __name__ == "__main__":
    main()
