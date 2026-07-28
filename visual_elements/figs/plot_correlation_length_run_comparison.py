"""Compare two correlation-length runs and plot inverse xi versus inverse D.

The ordinate is recomputed directly from the two leading eigenvalues:

    inverse_xi = ln(abs(lambda_max / lambda_second))

It is deliberately not computed as 1 / payload["correlation_length"].
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_DATA_ROOT = Path(
    r"D:\HyraiOn\ENS_Lyon\Internship\2026-EPFL\data\0713summary"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / (
    "correlation_length_run_comparison"
)
OLD_FILENAME = "correlation_length_unseeded_run1.json"
NEW_FILENAME = "correlation_length.json"
TARGET_J2 = (0.00, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30)


@dataclass(frozen=True)
class RunValue:
    inverse_xi: float
    correlation_length: float
    lambda_max_abs: float
    lambda_second_abs: float


@dataclass(frozen=True)
class Comparison:
    j2: float
    ansatz: str
    D: int
    old: RunValue
    new: RunValue
    old_path: Path
    new_path: Path


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_value(payload: dict[str, Any], path: Path) -> RunValue:
    eigenvalues = payload.get("eigenvalues")
    if not isinstance(eigenvalues, list) or len(eigenvalues) < 2:
        raise ValueError(f"{path}: fewer than two eigenvalues")

    leading = [
        complex(float(value["real"]), float(value["imag"]))
        for value in eigenvalues[:2]
    ]
    lambda_max, lambda_second = sorted(leading, key=abs, reverse=True)
    if abs(lambda_max) <= 0.0 or abs(lambda_second) <= 0.0:
        raise ValueError(f"{path}: leading eigenvalue magnitude is not positive")

    inverse_xi = math.log(abs(lambda_max / lambda_second))
    if not math.isfinite(inverse_xi):
        raise ValueError(f"{path}: non-finite inverse xi")

    return RunValue(
        inverse_xi=inverse_xi,
        correlation_length=float(payload["correlation_length"]),
        lambda_max_abs=abs(lambda_max),
        lambda_second_abs=abs(lambda_second),
    )


def discover_comparisons(data_root: Path) -> list[Comparison]:
    comparisons: list[Comparison] = []
    target_keys = {round(value, 12) for value in TARGET_J2}

    for old_path in sorted(data_root.rglob(OLD_FILENAME)):
        new_path = old_path.with_name(NEW_FILENAME)
        if not new_path.is_file():
            raise FileNotFoundError(f"Missing second-run file for {old_path}")

        old_payload = load_json(old_path)
        new_payload = load_json(new_path)
        j2_old = float(old_payload["calculation_hyperparameters"]["J2"])
        j2_new = float(new_payload["calculation_hyperparameters"]["J2"])
        if not math.isclose(j2_old, j2_new, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"J2 mismatch: {old_path} and {new_path}")
        if round(j2_old, 12) not in target_keys:
            continue

        D_old = int(old_payload["D_bond"])
        D_new = int(new_payload["D_bond"])
        if D_old != D_new:
            raise ValueError(f"D mismatch: {old_path} and {new_path}")

        comparisons.append(
            Comparison(
                j2=j2_old,
                ansatz=old_path.parent.parent.name,
                D=D_old,
                old=run_value(old_payload, old_path),
                new=run_value(new_payload, new_path),
                old_path=old_path,
                new_path=new_path,
            )
        )

    if not comparisons:
        raise FileNotFoundError(
            f"No {OLD_FILENAME!r} files found below {data_root}"
        )

    discovered_j2 = {round(item.j2, 12) for item in comparisons}
    missing_j2 = target_keys - discovered_j2
    if missing_j2:
        missing = ", ".join(f"{value:g}" for value in sorted(missing_j2))
        raise ValueError(f"Missing paired results for J2: {missing}")

    ansatz_names = {item.ansatz for item in comparisons}
    if len(ansatz_names) != 1:
        raise ValueError(
            "Exactly one ansatz is required for two curves per figure; found "
            + ", ".join(sorted(ansatz_names))
        )

    keys = [(round(item.j2, 12), item.ansatz, item.D) for item in comparisons]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate (J2, ansatz, D) comparison records found")

    return sorted(comparisons, key=lambda item: (item.j2, item.D))


def finite_difference(new: float, old: float) -> float:
    if math.isfinite(new) and math.isfinite(old):
        return new - old
    if math.isinf(new) and math.isinf(old) and (new > 0) == (old > 0):
        return 0.0
    return math.nan


def relative_difference(new: float, old: float) -> float:
    delta = finite_difference(new, old)
    if not math.isfinite(delta) or not math.isfinite(old) or old == 0.0:
        return math.nan
    return delta / old


def write_csv(comparisons: list[Comparison], output_path: Path) -> None:
    fieldnames = [
        "J2",
        "ansatz",
        "D",
        "inverse_D",
        "old_inverse_xi",
        "new_inverse_xi",
        "delta_inverse_xi",
        "abs_delta_inverse_xi",
        "relative_delta_inverse_xi",
        "old_correlation_length",
        "new_correlation_length",
        "delta_correlation_length",
        "relative_delta_correlation_length",
        "old_lambda_max_abs",
        "old_lambda_second_abs",
        "new_lambda_max_abs",
        "new_lambda_second_abs",
        "old_path",
        "new_path",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in comparisons:
            delta_inverse_xi = item.new.inverse_xi - item.old.inverse_xi
            writer.writerow(
                {
                    "J2": f"{item.j2:.12g}",
                    "ansatz": item.ansatz,
                    "D": item.D,
                    "inverse_D": 1.0 / item.D,
                    "old_inverse_xi": item.old.inverse_xi,
                    "new_inverse_xi": item.new.inverse_xi,
                    "delta_inverse_xi": delta_inverse_xi,
                    "abs_delta_inverse_xi": abs(delta_inverse_xi),
                    "relative_delta_inverse_xi": relative_difference(
                        item.new.inverse_xi, item.old.inverse_xi
                    ),
                    "old_correlation_length": item.old.correlation_length,
                    "new_correlation_length": item.new.correlation_length,
                    "delta_correlation_length": finite_difference(
                        item.new.correlation_length,
                        item.old.correlation_length,
                    ),
                    "relative_delta_correlation_length": relative_difference(
                        item.new.correlation_length,
                        item.old.correlation_length,
                    ),
                    "old_lambda_max_abs": item.old.lambda_max_abs,
                    "old_lambda_second_abs": item.old.lambda_second_abs,
                    "new_lambda_max_abs": item.new.lambda_max_abs,
                    "new_lambda_second_abs": item.new.lambda_second_abs,
                    "old_path": item.old_path,
                    "new_path": item.new_path,
                }
            )


def j2_filename(j2: float) -> str:
    return f"{j2:.2f}".replace(".", "p")


def plot_comparisons(
    comparisons: list[Comparison], output_dir: Path
) -> list[Path]:
    output_paths: list[Path] = []
    for j2 in TARGET_J2:
        points = [
            item
            for item in comparisons
            if math.isclose(item.j2, j2, rel_tol=0.0, abs_tol=1e-12)
        ]
        points.sort(key=lambda item: 1.0 / item.D)

        inverse_D = [1.0 / item.D for item in points]
        old_inverse_xi = [item.old.inverse_xi for item in points]
        new_inverse_xi = [item.new.inverse_xi for item in points]

        fig, ax = plt.subplots(figsize=(6.4, 4.8), constrained_layout=True)
        ax.plot(
            inverse_D,
            old_inverse_xi,
            color="#0072B2",
            marker="o",
            linewidth=1.7,
            markersize=5.5,
            label="previous run",
        )
        ax.plot(
            inverse_D,
            new_inverse_xi,
            color="#D55E00",
            marker="s",
            linestyle="--",
            linewidth=1.7,
            markersize=5.2,
            label="current run",
        )
        ax.set_xlabel(r"$1/D$")
        ax.set_ylabel(r"$\ln|\lambda_{\max}/\lambda_{\mathrm{second}}|$")
        ax.set_title(rf"$J_2={j2:g}$")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)

        output_path = output_dir / (
            f"J2_{j2_filename(j2)}_inverse_xi_comparison.pdf"
        )
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        output_paths.append(output_path)

    return output_paths


def print_summary(comparisons: list[Comparison]) -> None:
    print(f"Compared {len(comparisons)} paired results.")
    print("J2       max |delta inverse_xi|    RMS delta inverse_xi")
    print("-------  -----------------------  --------------------")
    for j2 in TARGET_J2:
        values = [
            item.new.inverse_xi - item.old.inverse_xi
            for item in comparisons
            if math.isclose(item.j2, j2, rel_tol=0.0, abs_tol=1e-12)
        ]
        max_abs = max(abs(value) for value in values)
        rms = math.sqrt(sum(value * value for value in values) / len(values))
        print(f"{j2:7.2f}  {max_abs:23.15g}  {rms:20.15g}")

    worst = max(
        comparisons,
        key=lambda item: abs(item.new.inverse_xi - item.old.inverse_xi),
    )
    delta = worst.new.inverse_xi - worst.old.inverse_xi
    print(
        "\nLargest change: "
        f"J2={worst.j2:g}, D={worst.D}, "
        f"old={worst.old.inverse_xi:.15g}, "
        f"new={worst.new.inverse_xi:.15g}, "
        f"delta={delta:.15g}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    if not data_root.is_dir():
        raise NotADirectoryError(data_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    comparisons = discover_comparisons(data_root)
    csv_path = output_dir / "correlation_length_run_comparison.csv"
    write_csv(comparisons, csv_path)
    output_paths = plot_comparisons(comparisons, output_dir)
    print_summary(comparisons)
    print(f"\nCSV: {csv_path}")
    print(f"Plots: {len(output_paths)} PDFs in {output_dir}")


if __name__ == "__main__":
    main()
