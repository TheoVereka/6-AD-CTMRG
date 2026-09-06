#!/usr/bin/env python3
"""Production-style stacked phase diagrams for the honeycomb J1-J2 model.

All phase boundaries are intentionally hardcoded below.  Running this file
creates both the full literature comparison and the variational-method subset.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
STYLE = HERE.parent / "PublicationPlots" / "plottingStyle" / "everyday_stylesheet.mplstyle"

J2_MAX = 0.35

COLORS = {
    "Neel": "#C0392B",
    "PVB": "#E67E22",
    "Columnar": "#2980B9",
    "QSL": "#27AE60",
    "Stripe": "#8E44AD",
}

# (left boundary, right boundary, phase key); values beyond J2_MAX are
# retained here as source information but clipped completely by the renderer.
STUDIES = [
    dict(label="Albuquerque 2011", year=2011, method="ED", phases=[
        (0.000, 0.200, "Neel"), (0.200, 0.350, "PVB"),
        (0.350, 0.500, "Columnar"),
    ]),
    dict(label="Gong 2013", year=2013, method="DMRG", phases=[
        (0.000, 0.220, "Neel"), (0.220, 0.250, "QSL"),
        (0.250, 0.350, "PVB"),
    ]),
    dict(label="Bishop 2013", year=2013, method="CCM", phases=[
        (0.000, 0.207, "Neel"), (0.207, 0.385, "PVB"),
        (0.385, 0.500, "Columnar"),
    ]),
    dict(label="Ganesh 2013", year=2013, method="DMRG", phases=[
        (0.000, 0.220, "Neel"), (0.220, 0.350, "PVB"),
        (0.350, 0.500, "Columnar"),
    ]),
    dict(label="Zhu 2013", year=2013, method="DMRG", phases=[
        (0.000, 0.260, "Neel"), (0.260, 0.360, "PVB"),
        (0.360, 0.500, "Columnar"),
    ]),
    dict(label="Ghorbani 2016", year=2016, method="Modified Spin Wave", phases=[
        (0.000, 0.207, "Neel"), (0.207, 0.396, "QSL"),
        (0.396, 0.500, "Stripe"),
    ]),
    dict(label="Ferrari 2017", year=2017, method="VMC (Gutzwiller proj.)", phases=[
        (0.000, 0.230, "Neel"), (0.230, 0.360, "PVB"),
        (0.360, 0.500, "Columnar"),
    ]),
    dict(label="Merino 2018", year=2018, method="Schwinger Boson MF", phases=[
        (0.000, 0.200, "Neel"), (0.200, 0.400, "QSL"),
        (0.400, 0.500, "Stripe"),
    ]),
    dict(label="Liu 2020", year=2020, method="Hubbard-Stratonovich MF", phases=[
        (0.000, 0.200, "Neel"), (0.200, 0.320, "QSL"),
        (0.320, 0.500, "QSL"),
    ]),
    dict(label="Mukherjee 2023", year=2023, method="Schwinger Boson MF", phases=[
        (0.000, 0.220, "Neel"), (0.220, 0.370, "QSL"),
        (0.370, 0.400, "Columnar"), (0.400, 0.500, "Stripe"),
    ]),
]

REMOVED_FROM_VARIATIONAL = {
    "Bishop 2013", "Ghorbani 2016", "Merino 2018", "Liu 2020", "Mukherjee 2023"
}


def _visible_segments(study: dict) -> list[tuple[float, float, str]]:
    segments = []
    for lo, hi, phase in study["phases"]:
        if phase == "Stripe":
            continue
        clipped_lo, clipped_hi = max(0.0, lo), min(J2_MAX, hi)
        if clipped_hi > clipped_lo:
            segments.append((clipped_lo, clipped_hi, phase))
    return segments


def _boundary_label(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")


def draw(studies: list[dict], basename: str) -> None:
    # Fixed-height rows make the reduced plot genuinely flatter.
    width = 11.0
    height = 1.65 + 0.49 * len(studies)
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([3.75 / width, 0.92 / height, 6.65 / width, (height - 1.42) / height])

    bar_height = 0.24
    for y, study in enumerate(studies):  # chronological order: older at bottom
        segments = _visible_segments(study)
        for lo, hi, phase in segments:
            ax.barh(
                y, hi - lo, left=lo, height=bar_height,
                color=COLORS[phase], edgecolor="none", zorder=2,
            )

        boundaries = sorted({hi for _lo, hi, _phase in segments if 0.0 < hi < J2_MAX})
        for boundary in boundaries:
            ax.vlines(
                boundary, y - bar_height / 2, y + bar_height / 2,
                color="black", linewidth=1.15, zorder=3,
            )
            ax.text(
                boundary, y + bar_height / 2 + 0.055, _boundary_label(boundary),
                ha="center", va="bottom", fontsize=15,
            )

    ax.set_xlim(0.0, J2_MAX)
    ax.set_ylim(-0.45, len(studies) - 0.28)
    ax.set_xlabel(r"$J_2/J_1$")
    ax.set_yticks(range(len(studies)))
    ax.set_yticklabels([f"{s['label']}  ({s['method']})" for s in studies], fontsize=13)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=17)
    ax.set_xticks([0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35])

    # Arrow continuation on the y axis and its requested label.
    ax.annotate(
        "", xy=(0.0, 1.045), xytext=(0.0, 0.985), xycoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", color="black", lw=2.0),
        annotation_clip=False,
    )
    ax.text(-0.018, 1.038, "year", transform=ax.transAxes,
            ha="right", va="center", fontsize=15, clip_on=False)

    legend_specs = [
        ("Neel", r"N\'eel AFM"),
        ("PVB", "PVB/Dimer-plaquette VBC"),
        ("QSL", "QSL"),
    ]
    handles = [mpatches.Patch(color=COLORS[key], label=label) for key, label in legend_specs]
    ax.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.23),
        ncols=3, columnspacing=1.45, handlelength=1.7, fontsize=13, frameon=True,
    )

    output = HERE / f"{basename}.pdf"
    fig.savefig(output, bbox_inches="tight")
    print(output)
    plt.close(fig)


def main() -> None:
    plt.style.use(STYLE)
    draw(STUDIES, "J1J2_honeycomb_phase_comparison")
    variational = [dict(s) for s in STUDIES if s["label"] not in REMOVED_FROM_VARIATIONAL]
    for study in variational:
        if study["label"] == "Ferrari 2017":
            study["method"] = "VMC"
    draw(variational, "J1J2_honeycomb_phase_variational")


if __name__ == "__main__":
    main()
