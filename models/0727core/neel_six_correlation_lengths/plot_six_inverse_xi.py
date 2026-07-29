#!/usr/bin/env python3
"""Merge all returned result JSON files and plot the six inverse lengths."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
SPECTRA = {
    "env2_generalized": ("env2 generalized", "#1f77b4", "-"),
    "env1_ab_env3_ba_generalized": (
        "env1(ab)-env3(ba) generalized",
        "#ff7f0e",
        "-",
    ),
    "env3_ab_env1_ba_generalized": (
        "env3(ab)-env1(ba) generalized",
        "#2ca02c",
        "-",
    ),
    "env2_ordinary": ("env2 ordinary", "#1f77b4", "--"),
    "env1_ab_env3_ba_ordinary": (
        "env1(ab)-env3(ba) ordinary",
        "#ff7f0e",
        "--",
    ),
    "env3_ab_env1_ba_ordinary": (
        "env3(ab)-env1(ba) ordinary",
        "#2ca02c",
        "--",
    ),
}


def _inverse_xi_from_lambdas(spectrum: dict[str, Any]) -> float:
    eigenvalues = spectrum["eigenvalues"][:2]
    values = [
        complex(float(item["real"]), float(item["imag"]))
        for item in eigenvalues
    ]
    return float(math.log(abs(values[0] / values[1])))


def _load(root: Path) -> dict[tuple[float, int], tuple[Path, dict[str, Any]]]:
    cases: dict[tuple[float, int], tuple[Path, dict[str, Any]]] = {}
    for path in root.rglob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("schema") != "neel_six_correlation_lengths":
            continue
        if payload.get("schema_version") != 3:
            continue
        if (
            payload.get("transfer_network_schema")
            != "three_geometric_straight_rows_v3"
        ):
            continue
        ctm = payload.get("ctm", {})
        if not (
            ctm.get("converged_ab_before_limit", False)
            and ctm.get("converged_ba_before_limit", False)
        ):
            print(f"Skipping non-converged CTM result {path}", flush=True)
            continue
        key = (float(payload["J2"]), int(payload["D"]))
        previous = cases.get(key)
        if previous is None or path.stat().st_mtime > previous[0].stat().st_mtime:
            cases[key] = (path, payload)
    return cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=HERE)
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "neel_six_inverse_xi_comparison.pdf",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    cases = _load(root)
    if not cases:
        raise FileNotFoundError(f"No six-spectrum result JSON below {root}.")
    j2_values = sorted({key[0] for key in cases})
    if j2_values != [0.0, 0.265]:
        print(f"Warning: available J2 values are {j2_values}.", flush=True)
    figure, axes = plt.subplots(
        1,
        len(j2_values),
        figsize=(6.4 * len(j2_values), 5.0),
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )
    for axis, j2 in zip(axes[0], j2_values, strict=True):
        selected = sorted(
            (
                (D, payload)
                for (case_j2, D), (_, payload) in cases.items()
                if math.isclose(case_j2, j2, rel_tol=0.0, abs_tol=1.0e-12)
            ),
            key=lambda item: 1.0 / item[0],
        )
        for key, (label, color, linestyle) in SPECTRA.items():
            x: list[float] = []
            y: list[float] = []
            for D, payload in selected:
                spectrum = payload.get("spectra", {}).get(key)
                if spectrum is None:
                    continue
                value = _inverse_xi_from_lambdas(spectrum)
                if math.isfinite(value):
                    x.append(1.0 / D)
                    y.append(value)
            if x:
                axis.plot(
                    x,
                    y,
                    marker="o",
                    linewidth=1.6,
                    markersize=4.5,
                    color=color,
                    linestyle=linestyle,
                    label=label,
                )
        axis.set_title(rf"$J_2={j2:g}$")
        axis.set_xlabel(r"$1/D$")
        axis.grid(alpha=0.25)
        axis.set_xticks(
            sorted({1.0 / D for D, _ in selected})
        )
        derived_D = [
            D
            for D, payload in selected
            if payload.get("checkpoint_derived_validation_only", False)
        ]
        if derived_D:
            joined = ",".join(str(D) for D in derived_D)
            axis.text(
                0.02,
                0.98,
                f"D={joined}: validation-only projection"
                f"{'s' if len(derived_D) > 1 else ''}",
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=8.5,
                color="0.35",
            )
    axes[0, 0].set_ylabel(
        r"$1/\xi=\ln\left|\lambda_1/\lambda_2\right|$"
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=3,
        frameon=False,
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(f"Loaded {len(cases)} cases; wrote {output} and PNG.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
