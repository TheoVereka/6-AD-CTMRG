#!/usr/bin/env python3
"""Plot original and crossed-environment inverse correlation lengths.

Every plotted value is recomputed directly from the two recorded leading
eigenvalues:

    inverse_xi = ln(abs(lambda_max / lambda_second))

The stored ``correlation_length`` field is never inverted.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_INPUT = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "0727core"
    / "temporary_crossed_environment_correlation_lengths.json"
)
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "crossed_environment_inverse_xi_comparison"
)
TARGET_J2 = (0.00, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30)
TARGET_DS = (3, 4, 5, 6)
CURVES = (
    (
        "existing_env2_baseline",
        "original env2",
        "#000000",
        "o",
        "-",
    ),
    (
        "env1_ab_x_env3_ba",
        r"env1$(a,b)\times$env3$(b,a)$",
        "#0072B2",
        "s",
        "--",
    ),
    (
        "env3_ab_x_env1_ba",
        r"env3$(a,b)\times$env1$(b,a)$",
        "#D55E00",
        "^",
        "-.",
    ),
)


def inverse_xi_from_eigenvalues(
    eigenvalues: Any,
    *,
    context: str,
) -> tuple[float, float, float]:
    if not isinstance(eigenvalues, list) or len(eigenvalues) < 2:
        raise ValueError(f"{context}: fewer than two eigenvalues")
    values = [
        complex(float(item["real"]), float(item["imag"]))
        for item in eigenvalues[:2]
    ]
    magnitudes = sorted((abs(value) for value in values), reverse=True)
    largest, second = magnitudes
    if not (
        math.isfinite(largest)
        and math.isfinite(second)
        and largest > 0.0
        and second > 0.0
    ):
        raise ValueError(f"{context}: invalid eigenvalue magnitudes")
    inverse_xi = math.log(largest / second)
    # A one-ulp negative value can arise after independently refined
    # eigenvalues.  The definition by sorted magnitudes is non-negative.
    if inverse_xi < 0.0 and abs(inverse_xi) <= 64.0 * math.ulp(1.0):
        inverse_xi = 0.0
    if not math.isfinite(inverse_xi) or inverse_xi < 0.0:
        raise ValueError(f"{context}: invalid inverse xi {inverse_xi}")
    return inverse_xi, largest, second


def load_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_results = payload.get("results")
    if not isinstance(raw_results, dict):
        raise ValueError(f"{path}: missing results mapping")

    rows: list[dict[str, Any]] = []
    seen: set[tuple[float, int]] = set()
    for result in raw_results.values():
        j2 = float(result["J2"])
        D_bond = int(result["D"])
        if not any(
            math.isclose(j2, target, rel_tol=0.0, abs_tol=1.0e-12)
            for target in TARGET_J2
        ) or D_bond not in TARGET_DS:
            continue
        key = (round(j2, 12), D_bond)
        if key in seen:
            raise ValueError(f"Duplicate result for J2={j2:g}, D={D_bond}")
        seen.add(key)

        row: dict[str, Any] = {
            "J2": j2,
            "D": D_bond,
            "inverse_D": 1.0 / D_bond,
        }
        for field, _, _, _, _ in CURVES:
            spectrum = result.get(field)
            if not isinstance(spectrum, dict):
                raise ValueError(
                    f"J2={j2:g}, D={D_bond}: missing {field}"
                )
            inverse_xi, largest, second = inverse_xi_from_eigenvalues(
                spectrum.get("eigenvalues"),
                context=f"J2={j2:g}, D={D_bond}, {field}",
            )
            row[field] = inverse_xi
            row[f"{field}_lambda_max_abs"] = largest
            row[f"{field}_lambda_second_abs"] = second
        rows.append(row)

    expected = {
        (round(j2, 12), D_bond)
        for j2 in TARGET_J2
        for D_bond in TARGET_DS
    }
    missing = expected - seen
    if missing:
        formatted = ", ".join(
            f"(J2={j2:g},D={D})" for j2, D in sorted(missing)
        )
        raise ValueError(f"Missing comparison points: {formatted}")
    return sorted(rows, key=lambda row: (row["J2"], row["D"]))


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = ["J2", "D", "inverse_D"]
    for field, _, _, _, _ in CURVES:
        fieldnames.extend(
            (
                field,
                f"{field}_lambda_max_abs",
                f"{field}_lambda_second_abs",
            )
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_one(
    ax: plt.Axes,
    rows: list[dict[str, Any]],
    *,
    j2: float,
    title: bool,
) -> None:
    points = [
        row
        for row in rows
        if math.isclose(row["J2"], j2, rel_tol=0.0, abs_tol=1.0e-12)
    ]
    points.sort(key=lambda row: row["inverse_D"])
    inverse_D = [row["inverse_D"] for row in points]
    for field, label, color, marker, linestyle in CURVES:
        ax.plot(
            inverse_D,
            [row[field] for row in points],
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.65,
            markersize=5.2,
            label=label,
        )
    ax.set_xlim(-0.002, 1.3 / 3.0)
    ax.set_xlabel(r"$1/D$")
    ax.set_ylabel(r"$1/\xi=\ln|\lambda_{\max}/\lambda_{\mathrm{second}}|$")
    if title:
        ax.set_title(rf"$J_2={j2:g}$")
    ax.grid(True, alpha=0.25)


def j2_token(j2: float) -> str:
    return f"{j2:.2f}".replace(".", "p")


def make_plots(rows: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for j2 in TARGET_J2:
        fig, ax = plt.subplots(figsize=(6.5, 4.9), constrained_layout=True)
        plot_one(ax, rows, j2=j2, title=True)
        ax.legend(frameon=False, fontsize=9)
        path = output_dir / (
            f"J2_{j2_token(j2)}_crossed_environment_inverse_xi.pdf"
        )
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(15.2, 7.2),
        constrained_layout=True,
        sharex=True,
    )
    for ax, j2 in zip(axes.flat, TARGET_J2, strict=True):
        plot_one(ax, rows, j2=j2, title=True)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=3,
        frameon=False,
    )
    overview = output_dir / "all_J2_crossed_environment_inverse_xi.pdf"
    fig.savefig(overview, bbox_inches="tight")
    plt.close(fig)
    paths.append(overview)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(input_path)
    csv_path = output_dir / "crossed_environment_inverse_xi.csv"
    write_csv(rows, csv_path)
    paths = make_plots(rows, output_dir)
    print(f"Loaded {len(rows)} comparison points from {input_path}.")
    print(f"CSV: {csv_path}")
    print(f"Plots: {len(paths)} PDFs in {output_dir}")


if __name__ == "__main__":
    main()
