#!/usr/bin/env python3
"""Compare the pinning-branch crossing energy with original-2C3 E_infinity."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
DATA = REPO.parent / "data"
PHASE_DIR = HERE / "plots" / "pinning_energy_phase_boundary"
CROSSINGS = PHASE_DIR / "hc_03_all_good_D_infinity.csv"
ORIGINAL_FITS = DATA / "processed" / "publicationPlots" / "figure_24_fits.csv"
OUTPUT = PHASE_DIR / "05_E_crossing_vs_original_2C3.pdf"
OUTPUT_CSV = PHASE_DIR / "05_E_crossing_vs_original_2C3.csv"


def read_crossings() -> list[dict]:
    output = []
    with CROSSINGS.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            values = {key: float(row[key]) for key in (
                "J2", "h_c", "E_crossing",
            )}
            if not all(math.isfinite(value) for value in values.values()):
                continue
            output.append({
                "J2": values["J2"], "h_c": values["h_c"],
                "E_crossing": values["E_crossing"],
                "E_crossing_error_low": float(
                    row.get("E_crossing_error_low", math.nan)
                ),
                "E_crossing_error_high": float(
                    row.get("E_crossing_error_high", math.nan)
                ),
                "h_c_sigma": float(row.get("error_1sigma", math.nan)),
            })
    return sorted(output, key=lambda row: row["J2"])


def read_original() -> list[dict]:
    output = []
    with ORIGINAL_FITS.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if (row.get("ansatz") != "2C3" or row.get("model") != "gapped"
                    or row.get("parameter") != "E0"):
                continue
            J2 = float(row["J2"])
            if 0.22 <= J2 <= 0.34:
                output.append({
                    "J2": J2, "energy": float(row["central"]),
                    "error": float(row["error"]),
                })
    return sorted(output, key=lambda row: row["J2"])


def main() -> int:
    crossing = read_crossings()
    original = read_original()
    if not crossing or not original:
        raise RuntimeError("crossing or original-2C3 energy data are missing")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(crossing[0]))
        writer.writeheader()
        writer.writerows(crossing)

    x_o = np.asarray([row["J2"] for row in original])
    y_o = np.asarray([row["energy"] for row in original])
    e_o = np.asarray([row["error"] for row in original])
    x_c = np.asarray([row["J2"] for row in crossing])
    y_c = np.asarray([row["E_crossing"] for row in crossing])
    e_c_low = np.asarray([row["E_crossing_error_low"] for row in crossing])
    e_c_high = np.asarray([row["E_crossing_error_high"] for row in crossing])

    figure, axis = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    axis.fill_between(x_o, y_o - e_o, y_o + e_o, color="black", alpha=0.12,
                      linewidth=0)
    axis.plot(x_o, y_o, color="black", linewidth=2.2,
              label=r"original 2C3 $E_\infty(J_2)$")
    axis.plot(x_c, y_c, color="#d89000", linewidth=1.8, zorder=2)
    axis.errorbar(
        x_c, y_c, yerr=np.vstack((e_c_low, e_c_high)),
        color="#d89000", linestyle="none", marker="o", markersize=5.5,
        markeredgecolor="black", markeredgewidth=0.6, capsize=2.2, zorder=3,
        label=r"$E_{\rm crossing}(J_2)$",
    )
    axis.set_xlim(0.22, 0.34)
    axis.set_xlabel(r"$J_2$")
    axis.set_ylabel("energy per site")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False)
    figure.savefig(OUTPUT)
    plt.close(figure)
    print(f"Output: {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
