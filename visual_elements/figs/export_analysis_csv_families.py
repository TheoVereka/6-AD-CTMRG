#!/usr/bin/env python3
"""Export both requested per-J2 CSV families in one command."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import plot_analysis_Windows as summary_plot
import plot_analysis_neel_legacy as legacy_plot


DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "csv_data_points_complete"


def inverse_xi_value_and_errors(entry: dict) -> tuple[float, float, float]:
    return (
        float(entry["inverse_xi"]),
        float(entry["inverse_xi_upper_error"]),
        float(entry["inverse_xi_lower_error"]),
    )


def j2_tag(j2: float) -> str:
    return f"{j2:g}".replace(".", "p")


def write_csv(directory: Path, j2: float, header: tuple[str, ...], rows: list[tuple]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"J2_{j2_tag(j2)}.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def export_legacy(output: Path) -> int:
    count = 0
    for j2, ansatz_map in sorted(legacy_plot.load_all().items()):
        values = ansatz_map["neel_symmetrized"]
        energies = dict(zip(values["Ds"], values["energy_per_site"]))
        mags = dict(zip(values["Ds"], values["mneel_list"]))
        rows = []
        for D, correlation in sorted(values["inverse_correlation_lengths"].items()):
            if D not in energies or D not in mags:
                continue
            inverse_xi, inverse_xi_upper, inverse_xi_lower = (
                inverse_xi_value_and_errors(correlation)
            )
            rows.append(
                (
                    D,
                    f"{energies[D]:.17g}",
                    f"{inverse_xi:.17g}",
                    f"{inverse_xi_upper:.17g}",
                    f"{inverse_xi_lower:.17g}",
                    f"{mags[D]:.17g}",
                    f"{values['mneel_error_by_D'].get(D, 0.0):.17g}",
                )
            )
        if rows:
            write_csv(
                output / "0507coreD45678910_neel",
                j2,
                ("D", "E", "inverse_xi_central_value", "inverse_xi_upper_error", "inverse_xi_lower_error", "m_Neel", "m_Neel_error"),
                rows,
            )
            count += 1
    return count


def delta_and_error(entry: dict) -> tuple[float, float]:
    rank1 = next(index for index, rank in enumerate(entry["ranks"]) if rank == 1)
    rank3 = next(index for index, rank in enumerate(entry["ranks"]) if rank == 3)
    delta = abs(float(entry["means"][rank3]) - float(entry["means"][rank1]))
    error = math.sqrt(
        float(entry["stds"][rank1]) ** 2 + float(entry["stds"][rank3]) ** 2
    ) / math.sqrt(2.0)
    return delta, error


def export_twoc3(output: Path) -> int:
    count = 0
    for j2, ansatz_map in sorted(summary_plot.load_all().items()):
        values = ansatz_map.get("2tensor_twoC3")
        if values is None:
            continue
        energies = dict(zip(values["Ds"], values["energy_per_site"]))
        rows = []
        for D, correlation in sorted(values["inverse_correlation_lengths"].items()):
            nn = values["nn_groups"].get(D)
            if D not in energies or nn is None:
                continue
            inverse_xi, inverse_xi_upper, inverse_xi_lower = (
                inverse_xi_value_and_errors(correlation)
            )
            delta, delta_error = delta_and_error(nn)
            rows.append(
                (
                    D,
                    f"{energies[D]:.17g}",
                    f"{inverse_xi:.17g}",
                    f"{inverse_xi_upper:.17g}",
                    f"{inverse_xi_lower:.17g}",
                    f"{delta:.17g}",
                    f"{delta_error:.17g}",
                )
            )
        if rows:
            write_csv(
                output / "0713summary_twoc3",
                j2,
                ("D", "E", "inverse_xi_central_value", "inverse_xi_upper_error", "inverse_xi_lower_error", "Delta", "Delta_error"),
                rows,
            )
            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    legacy_count = export_legacy(args.output.resolve())
    twoc3_count = export_twoc3(args.output.resolve())
    print(f"Exported legacy-Neel J2 files={legacy_count}; 0713summary twoC3 J2 files={twoc3_count}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
