#!/usr/bin/env python3
"""Plot energy per site from locally downloaded Izar connection trees."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


CHAIN = "ABCDEFGHI"
DIRECT = "OPQRSTU"
OBS_NAME = re.compile(r"D_(\d+)_chi_(\d+)_energy_magnetization_correlation\.txt")
ENERGY_LINE = re.compile(r"^energy_per_site\s*=\s*(\S+)", re.MULTILINE)
COLORS = {"chain": "#2358a6", "direct": "#d27726"}


class MetadataMismatchError(ValueError):
    """A result belongs to a different Hamiltonian or D than its job row."""


@dataclass(frozen=True)
class Point:
    job_id: str
    D: int
    connection: str
    node: str
    t: int
    chi: int
    energy_per_site: float
    source: str
    completed_stage: bool

    @property
    def series(self) -> str:
        return "chain" if self.node in CHAIN else "direct"

    @property
    def x_percent(self) -> float:
        return 12.5 * self.t


def energy_from_json(path: Path, D: int, connection: str, t: int) -> tuple[int, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (payload.get("connection") != connection or payload.get("t") != t
            or payload.get("D_bond_list") != [D]):
        raise MetadataMismatchError(f"Hamiltonian or D metadata mismatch: {path}")
    candidates = []
    for row in payload.get("energy_table", []):
        if int(row["D_bond"]) == D:
            energy = float(row["energy_per_site"])
            if math.isfinite(energy):
                candidates.append((int(row["chi"]), energy))
    if not candidates:
        raise ValueError(f"No finite E/site in {path}")
    return max(candidates, key=lambda item: item[0])


def energy_from_observables(directory: Path, D: int) -> tuple[int, float] | None:
    candidates = []
    for path in directory.glob(f"D_{D}_chi_*_energy_magnetization_correlation.txt"):
        match = OBS_NAME.fullmatch(path.name)
        if not match or int(match.group(1)) != D:
            continue
        energy_match = ENERGY_LINE.search(path.read_text(encoding="utf-8"))
        if energy_match:
            energy = float(energy_match.group(1))
            if math.isfinite(energy):
                candidates.append((int(match.group(2)), energy))
    return max(candidates, key=lambda item: item[0]) if candidates else None


def load_points(data_root: Path) -> tuple[list[Point], list[str]]:
    status_path = data_root / "sync_status.csv"
    with status_path.open(encoding="utf-8-sig", newline="") as stream:
        jobs = list(csv.DictReader(stream))
    points = []
    warnings = []
    for job in jobs:
        if job["status"] != "downloaded":
            continue
        D = int(job["D"])
        connection = job["connection"]
        node = job["node"]
        t = int(job["t"])
        directory = data_root / "results" / f"D{D}_{connection}" / node
        if not directory.is_dir():
            warnings.append(f"Missing local directory for job {job['job_id']}: {directory}")
            continue

        energy = None
        source = ""
        json_path = directory / "sweep_results.json"
        if json_path.is_file():
            try:
                energy = energy_from_json(json_path, D, connection, t)
                source = "sweep_results.json"
            except MetadataMismatchError as exc:
                warnings.append(str(exc))
                continue
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                warnings.append(str(exc))
        if energy is None:
            energy = energy_from_observables(directory, D)
            source = "observables" if energy is not None else ""
        if energy is None:
            warnings.append(f"No finite E/site for finished job {job['job_id']} ({directory})")
            continue
        chi, energy_per_site = energy
        points.append(Point(
            job_id=job["job_id"], D=D, connection=connection, node=node,
            t=t, chi=chi, energy_per_site=energy_per_site, source=source,
            completed_stage=(directory / "COMPLETED.stage").is_file(),
        ))
    return points, warnings


def write_points_csv(points: list[Point], path: Path) -> None:
    fields = ("job_id", "D", "connection", "node", "series", "t",
              "x_percent", "chi", "energy_per_site", "source", "completed_stage")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for point in sorted(points, key=lambda p: (p.D, p.connection, p.t, p.node)):
            writer.writerow({field: getattr(point, field) for field in fields})


def plot(points: list[Point], output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.0), sharex=True, sharey=True)
    energies = [point.energy_per_site for point in points]
    if energies:
        low, high = min(energies), max(energies)
        padding = max(0.06 * (high - low), 0.003)
        ylim = (low - padding, high + padding)
    else:
        ylim = (-1.0, 0.0)

    ticks = [12.5 * i for i in range(9)]
    tick_labels = [f"{x:g}%" for x in ticks]
    has_partial = any(not point.completed_stage for point in points)
    for row, D in enumerate((6, 7)):
        for col, connection in enumerate(("dimer", "plaq")):
            ax = axes[row, col]
            group = [point for point in points
                     if point.D == D and point.connection == connection]
            by_node = {point.node: point for point in group}

            chain_y = [by_node[node].energy_per_site
                       if node in by_node and by_node[node].completed_stage
                       else math.nan for node in CHAIN]
            ax.plot(ticks, chain_y, color=COLORS["chain"], marker="o",
                    markersize=5, linewidth=1.8)
            direct = [by_node[node] for node in DIRECT
                      if node in by_node and by_node[node].completed_stage]
            if direct:
                ax.scatter([point.x_percent for point in direct],
                           [point.energy_per_site for point in direct],
                           color=COLORS["direct"], marker="s", s=45, zorder=3)
            for point in group:
                if not point.completed_stage:
                    ax.scatter([point.x_percent], [point.energy_per_site],
                               facecolors="none", edgecolors=COLORS[point.series],
                               marker="o" if point.series == "chain" else "s",
                               s=55, linewidths=1.5, zorder=4)

            ax.set_title(f"D = {D}  |  {'Dimer' if connection == 'dimer' else 'Plaquette'}")
            ax.set_xlim(-3, 103)
            ax.set_ylim(*ylim)
            ax.set_xticks(ticks, tick_labels, rotation=45, ha="right")
            ax.tick_params(axis="x", labelbottom=True, labelsize=8)
            ax.tick_params(axis="y", labelsize=9)
            ax.grid(True, color="#e1e4e8", linewidth=0.7)
            ax.set_axisbelow(True)
            if not group:
                ax.text(0.5, 0.5, "No energy result yet", transform=ax.transAxes,
                        ha="center", va="center", color="#777777")

    handles = [
        Line2D([0], [0], color=COLORS["chain"], marker="o", linewidth=1.8,
               label="A–I: sequential continuation"),
        Line2D([0], [0], color=COLORS["direct"], marker="s", linestyle="none",
               label="O–U: each resumed from A"),
    ]
    if has_partial:
        handles.append(Line2D([0], [0], color="#555555", marker="o",
                              markerfacecolor="none", linestyle="none",
                              label="Open marker: unfinished stage with energy"))
    fig.suptitle("Connection paths: energy per site", fontsize=15, y=0.985)
    fig.supxlabel("t / 8 × 100%", fontsize=12, y=0.025)
    fig.supylabel("E per site", fontsize=12, x=0.025)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.955),
               ncol=len(handles), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0.04, 0.065, 0.99, 0.91), h_pad=2.0, w_pad=1.8)
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"connection_energy_per_site.{suffix}",
                    dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    points, warnings = load_points(args.data_root)
    write_points_csv(points, args.output_dir / "plotted_energies.csv")
    plot(points, args.output_dir)
    for message in warnings:
        print(f"Warning: {message}")
    print(f"Plotted {len(points)} energy points; y range shared by all four panels.")
    print(f"Figure: {args.output_dir / 'connection_energy_per_site.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
