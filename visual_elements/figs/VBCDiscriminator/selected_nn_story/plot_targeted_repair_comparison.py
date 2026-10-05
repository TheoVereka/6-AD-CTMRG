#!/usr/bin/env python3
"""Overlay targeted Kuma repairs without changing the selected data.

This is deliberately a diagnostic-only renderer.  It reads the frozen
``selected_nn_data.csv``, adds the previously rejected D=10/J2=.28 dimer
candidate and the three completed Kuma h=0 observations, and rewrites only
the J2=.28 and .32 inverse-D PDFs.  No CSV or selection is modified.
"""

from __future__ import annotations

import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


HERE = Path(__file__).resolve().parent
PLOTS = HERE / "plots"
REPO = HERE.parents[3]
DATA = REPO.parent / "data"
REPAIR_ROOT = (
    DATA / "distinVBCsKumaTargetedRepairs" / "Results_Kuma_TargetedRepairs"
)

sys.path.insert(0, str(HERE))
import select_and_plot as selected_story  # noqa: E402


TEXTURES = ("dimer-plaquette", "plaquette")
TITLES = {
    "dimer-plaquette": "Dimer-plaquette sector",
    "plaquette": "Plaquette sector",
}
RANKS = ("strongest", "middle", "weakest")
COLORS = ("#c92535", "#2f9855", "#2878b8")
MARKERS = ("o", "s", "^")
LINESTYLES = ("-", "--", ":")
MARKER_SIZES = (8.2, 6.1, 4.2)


@dataclass(frozen=True)
class ExtraPoint:
    label: str
    texture: str
    J2: float
    D: int
    energy: float
    ranks: tuple[tuple[float, float], ...]
    offset: float
    connector: str


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def selected_rows() -> list[dict[str, str]]:
    return [
        row for row in read_csv(PLOTS / "selected_nn_data.csv")
        if row.get("selected", "").lower() == "true"
    ]


def rejected_d10_dimer_028() -> ExtraPoint:
    candidates = [
        row for row in read_csv(PLOTS / "candidate_audit.csv")
        if (row["texture"] == "dimer-plaquette"
            and int(row["D"]) == 10
            and math.isclose(float(row["J2"]), 0.28, abs_tol=1.0e-10)
            and row["source"].startswith("sep27:a02:left"))
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            "Expected exactly one rejected Sep27 D10/J2=.28 dimer candidate; "
            f"found {len(candidates)}"
        )
    row = candidates[0]
    ranks = tuple(
        (float(row[name]), float(row[f"{name}_error"])) for name in RANKS
    )
    return ExtraPoint(
        "old D=10", "dimer-plaquette", 0.28, 10,
        float(row["energy"]), ranks, -0.00145, ":",
    )


def repair_point(
    alias: str, texture: str, J2: float, D: int, chi: int, offset: float,
) -> ExtraPoint:
    stage = REPAIR_ROOT / alias / "h_0"
    observation = stage / f"D_{D}_chi_{chi}_energy_magnetization_correlation.txt"
    if not (stage / "COMPLETED.stage").is_file() or not observation.is_file():
        raise FileNotFoundError(f"Completed repair observation missing: {observation}")
    energy, ranks, _delta, _delta_error, _eta = (
        selected_story.summarize_observation(observation)
    )
    return ExtraPoint(
        f"{alias} D={D}", texture, J2, D, energy, ranks,
        offset, "-.",
    )


def extras() -> list[ExtraPoint]:
    return [
        rejected_d10_dimer_028(),
        repair_point("r01", "dimer-plaquette", 0.28, 10, 120, 0.00145),
        repair_point("r02", "plaquette", 0.32, 10, 120, -0.00145),
        repair_point("r03", "plaquette", 0.32, 11, 140, 0.00145),
    ]


def plot_one(
    selected: list[dict[str, str]], added: list[ExtraPoint],
    J2: float, output: Path,
) -> None:
    figure, axes = plt.subplots(
        1, 2, figsize=(13.2, 5.7), sharex=True, sharey=True,
        constrained_layout=True,
    )
    all_values: list[float] = []
    for axis, texture in zip(axes, TEXTURES):
        subset = sorted(
            [row for row in selected
             if row["texture"] == texture
             and math.isclose(float(row["J2"]), J2, abs_tol=1.0e-10)],
            key=lambda row: 1.0 / int(row["D"]),
        )
        for rank, (name, color) in enumerate(zip(RANKS, COLORS)):
            all_values.extend(float(row[name]) for row in subset)
            axis.errorbar(
                [1.0 / int(row["D"]) for row in subset],
                [float(row[name]) for row in subset],
                yerr=[float(row[f"{name}_error"]) for row in subset],
                color=color, marker=MARKERS[rank],
                markersize=MARKER_SIZES[rank],
                linestyle=LINESTYLES[rank], linewidth=1.05,
                elinewidth=0.75, capsize=2.0, zorder=2,
            )
        local_extras = [
            point for point in added
            if point.texture == texture and math.isclose(point.J2, J2)
        ]
        for point in local_extras:
            x = 1.0 / point.D + point.offset
            values = [rank[0] for rank in point.ranks]
            all_values.extend(values)
            axis.plot(
                [x, x], [min(values), max(values)], color="black",
                linestyle=point.connector, linewidth=1.2, alpha=0.72,
                zorder=3,
            )
            for rank, (value, error) in enumerate(point.ranks):
                axis.errorbar(
                    x, value, yerr=error, color=COLORS[rank],
                    marker=MARKERS[rank], markersize=MARKER_SIZES[rank] + 1.8,
                    markeredgecolor="black", markeredgewidth=1.1,
                    linestyle="none", elinewidth=1.05, capsize=2.7,
                    zorder=5,
                )
            axis.annotate(
                point.label, (x, max(values)), xytext=(4, 7),
                textcoords="offset points", fontsize=8.5,
                rotation=90, ha="left", va="bottom",
            )
        axis.set_title(rf"{TITLES[texture]}, $J_2={J2:g}$", fontsize=12)
        axis.set_xlabel(r"$1/D$", fontsize=12)
        axis.grid(alpha=0.18)
    span = max(max(all_values) - min(all_values), 1.0e-3)
    axes[0].set_ylim(
        min(all_values) - 0.06 * span,
        max(all_values) + 0.24 * span,
    )
    axes[0].set_ylabel("NN correlation", fontsize=12)

    rank_handles = [
        Line2D([], [], color=color, marker=MARKERS[index],
               linestyle=LINESTYLES[index], linewidth=1.05,
               markersize=MARKER_SIZES[index], label=name)
        for index, (color, name) in enumerate(zip(COLORS, RANKS))
    ]
    identity_handles = [
        Line2D([], [], color="0.35", linestyle="-", marker="o",
               label="current selected curve"),
    ]
    if any(math.isclose(point.J2, J2) and point.label.startswith("old")
           for point in added):
        identity_handles.append(Line2D(
            [], [], color="black", linestyle=":", marker="o",
            markeredgecolor="black", label="previously discarded candidate",
        ))
    if any(math.isclose(point.J2, J2) and point.label.startswith("r")
           for point in added):
        identity_handles.append(Line2D(
            [], [], color="black", linestyle="-.", marker="o",
            markeredgecolor="black", label="new Kuma repair",
        ))
    figure.legend(
        handles=rank_handles + identity_handles,
        loc="outside upper center", ncol=len(rank_handles + identity_handles),
        frameon=False, fontsize=8.5,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    plt.close(figure)


def selected_energy(
    rows: list[dict[str, str]], texture: str, J2: float, D: int,
) -> float:
    matches = [
        row for row in rows
        if row["texture"] == texture and int(row["D"]) == D
        and math.isclose(float(row["J2"]), J2, abs_tol=1.0e-10)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one selected point for {(texture, J2, D)}; "
            f"found {len(matches)}"
        )
    return float(matches[0]["energy"])


def main() -> int:
    frozen = selected_rows()
    added = extras()
    plot_one(
        frozen, added, 0.28,
        PLOTS / "NN_corr_vs_inverse_D_J2_0p28.pdf",
    )
    plot_one(
        frozen, added, 0.32,
        PLOTS / "NN_corr_vs_inverse_D_J2_0p32.pdf",
    )
    print("energy_per_site")
    print(f"old_discarded_D10_dimer_J2_0p28={added[0].energy:.13f}")
    print(f"r01_D10_dimer_J2_0p28={added[1].energy:.13f}")
    print("old_selected_D10_plaquette_J2_0p32="
          f"{selected_energy(frozen, 'plaquette', 0.32, 10):.13f}")
    print(f"r02_D10_plaquette_J2_0p32={added[2].energy:.13f}")
    print("old_selected_D11_plaquette_J2_0p32="
          f"{selected_energy(frozen, 'plaquette', 0.32, 11):.13f}")
    print(f"r03_D11_plaquette_J2_0p32={added[3].energy:.13f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
