#!/usr/bin/env python3
"""Plot generalized and ordinary Neel inverse correlation lengths.

For each ``(J2,D)`` and each eigenproblem type, the three geometric transfer
directions are sorted.  The median is the plotted center and the minimum and
maximum form an asymmetric error bar.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
GENERALIZED_KEYS = (
    "env2_generalized",
    "env1_ab_env3_ba_generalized",
    "env3_ab_env1_ba_generalized",
)
ORDINARY_KEYS = (
    "env2_ordinary",
    "env1_ab_env3_ba_ordinary",
    "env3_ab_env1_ba_ordinary",
)


@dataclass(frozen=True)
class Point:
    D: int
    lower: float
    center: float
    upper: float


def _inverse_xi(spectrum: dict[str, Any]) -> float:
    stored = spectrum.get("inverse_correlation_length")
    if stored is not None:
        return float(stored)
    values = [
        complex(float(item["real"]), float(item["imag"]))
        for item in spectrum["eigenvalues"][:2]
    ]
    return float(math.log(abs(values[0] / values[1])))


def _triplet(payload: dict[str, Any], keys: tuple[str, ...]) -> tuple[float, float, float]:
    values = sorted(_inverse_xi(payload["spectra"][key]) for key in keys)
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise ValueError("A finite three-direction spectrum is required.")
    return values[0], values[1], values[2]


def _is_converged(payload: dict[str, Any]) -> bool:
    ctm = payload.get("ctm", {})
    return bool(
        ctm.get("converged_ab_within_budget", ctm.get("converged_ab_before_limit", False))
        and ctm.get("converged_ba_within_budget", ctm.get("converged_ba_before_limit", False))
    )


def _load(root: Path) -> dict[tuple[float, int], dict[str, Any]]:
    cases: dict[tuple[float, int], tuple[int, Path, dict[str, Any]]] = {}
    for path in root.rglob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("schema") != "neel_six_correlation_lengths":
            continue
        if payload.get("schema_version") != 3:
            continue
        if payload.get("transfer_network_schema") != "three_geometric_straight_rows_v3":
            continue
        if not _is_converged(payload):
            print(f"Skipping non-converged result: {path}", flush=True)
            continue
        try:
            _triplet(payload, GENERALIZED_KEYS)
            _triplet(payload, ORDINARY_KEYS)
            key = (float(payload["J2"]), int(payload["D"]))
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            print(f"Skipping incomplete result: {path}", flush=True)
            continue
        # A hash-bearing result is newer than a legacy v3 file.  Modification
        # time breaks ties if several returned copies are present.
        rank = int(bool(payload.get("checkpoint_sha256")))
        previous = cases.get(key)
        if previous is None or (rank, path.stat().st_mtime_ns) > (
            previous[0], previous[1].stat().st_mtime_ns
        ):
            cases[key] = (rank, path, payload)
    return {key: item[2] for key, item in cases.items()}


def _series(
    cases: dict[tuple[float, int], dict[str, Any]], keys: tuple[str, ...]
) -> dict[float, list[Point]]:
    output: dict[float, list[Point]] = {}
    for (j2, D), payload in cases.items():
        lower, center, upper = _triplet(payload, keys)
        output.setdefault(j2, []).append(Point(D, lower, center, upper))
    for points in output.values():
        points.sort(key=lambda point: 1.0 / point.D)
    return output


def _arrays(points: list[Point]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray([1.0 / point.D for point in points])
    y = np.asarray([point.center for point in points])
    yerr = np.asarray(
        [
            [point.center - point.lower for point in points],
            [point.upper - point.center for point in points],
        ]
    )
    return x, y, yerr


def _limits(*collections: dict[float, list[Point]]) -> tuple[tuple[float, float], tuple[float, float]]:
    points = [point for collection in collections for values in collection.values() for point in values]
    if not points:
        raise ValueError("No points available for axis limits.")
    x_values = [1.0 / point.D for point in points]
    y_values = [value for point in points for value in (point.lower, point.upper)]
    x_span = max(x_values) - min(x_values)
    y_span = max(y_values) - min(y_values)
    x_padding = max(0.03 * max(x_values), 0.08 * x_span)
    y_padding = max(0.03 * max(y_values), 0.08 * y_span, 1.0e-8)
    return (
        (max(0.0, min(x_values) - x_padding), max(x_values) + x_padding),
        (max(0.0, min(y_values) - y_padding), max(y_values) + y_padding),
    )


def _save(figure: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(figure)


def _overview(
    series: dict[float, list[Point]],
    colors: dict[float, Any],
    *,
    title: str,
    output: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(9.2, 6.2), constrained_layout=True)
    for j2 in sorted(series):
        x, y, yerr = _arrays(series[j2])
        axis.errorbar(
            x,
            y,
            yerr=yerr,
            marker="o",
            markersize=4.2,
            linewidth=1.35,
            elinewidth=1.0,
            capsize=2.5,
            color=colors[j2],
            label=rf"$J_2={j2:g}$",
        )
    axis.set_xlabel(r"$1/D$")
    axis.set_ylabel(r"$1/\xi$")
    axis.set_title(title + " (center = median; bars = min/max direction)")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2, fontsize=8.5, frameon=False, bbox_to_anchor=(1.02, 1.0), loc="upper left")
    _save(figure, output)


def _per_j2(
    generalized: dict[float, list[Point]],
    ordinary: dict[float, list[Point]],
    *,
    output_dir: Path,
) -> int:
    xlim, ylim = _limits(generalized, ordinary)
    output_dir.mkdir(parents=True, exist_ok=True)
    expected: set[Path] = set()
    count = 0
    for j2 in sorted(set(generalized) | set(ordinary)):
        figure, axis = plt.subplots(figsize=(7.4, 5.4), constrained_layout=True)
        for label, collection, color, marker in (
            ("generalized", generalized, "#1f77b4", "o"),
            ("ordinary", ordinary, "#d62728", "s"),
        ):
            points = collection.get(j2, [])
            if not points:
                continue
            x, y, yerr = _arrays(points)
            axis.errorbar(
                x,
                y,
                yerr=yerr,
                marker=marker,
                markersize=5.0,
                linewidth=1.5,
                elinewidth=1.1,
                capsize=3.0,
                color=color,
                label=label,
            )
        axis.set_xlim(*xlim)
        axis.set_ylim(*ylim)
        axis.set_xlabel(r"$1/D$")
        axis.set_ylabel(r"$1/\xi$")
        axis.set_title(rf"$J_2={j2:g}$ (shared limits; median and min/max)")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
        tag = f"J2_{j2:g}".replace(".", "p")
        output = output_dir / f"{tag}_inverse_xi.pdf"
        expected.update((output, output.with_suffix(".png")))
        _save(figure, output)
        count += 1
    for path in output_dir.glob("J2_*_inverse_xi.*"):
        if path not in expected and path.suffix.lower() in {".pdf", ".png"}:
            path.unlink()
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=HERE / "results")
    parser.add_argument("--output-dir", type=Path, default=HERE / "figures")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cases = _load(args.root.resolve())
    if not cases:
        raise FileNotFoundError(f"No converged schema-v3 results below {args.root.resolve()}.")
    generalized = _series(cases, GENERALIZED_KEYS)
    ordinary = _series(cases, ORDINARY_KEYS)
    j2_values = sorted(set(generalized) | set(ordinary))
    color_values = plt.colormaps["turbo"](np.linspace(0.04, 0.96, len(j2_values)))
    colors = dict(zip(j2_values, color_values, strict=True))
    output_dir = args.output_dir.resolve()
    _overview(
        generalized,
        colors,
        title="Generalized Neel inverse correlation length",
        output=output_dir / "all_J2_generalized_inverse_xi.pdf",
    )
    _overview(
        ordinary,
        colors,
        title="Ordinary Neel inverse correlation length",
        output=output_dir / "all_J2_ordinary_inverse_xi.pdf",
    )
    per_j2_count = _per_j2(
        generalized, ordinary, output_dir=output_dir / "per_J2"
    )
    print(
        f"Loaded {len(cases)} cases across {len(j2_values)} J2 values; "
        f"wrote 2 overview figures and {per_j2_count} per-J2 figures "
        f"(PDF + PNG) below {output_dir}.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
