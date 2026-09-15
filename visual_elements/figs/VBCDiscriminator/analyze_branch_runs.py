#!/usr/bin/env python3
"""Analyze the h->0 twoC3 plaquette/dimer-plaquette branch competition.

Only h=0 energies enter the variational comparison.  Finite-field data are
used solely to verify adiabatic branch following.  A point is called resolved
only when both target textures survive to h=0 in every replica and the energy
gap exceeds three times a conservative numerical resolution made from the
chi-lookahead shift, replica spread, and an absolute floor.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_existing_twoc3 import Row, parse_observation


HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE.parents[2] / "models" / "0907core" / "Results_VBC_branches"
DEFAULT_OUTPUT = HERE / "branch_comparison"
BRANCHES = ("plaquette", "dimer-plaquette")
REPLICA_RE = re.compile(r"replica_(\d+)")


def read_scalar_hyperparams(path: Path) -> dict[str, str]:
    """Read the scalar subset needed here without requiring PyYAML."""
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.match(r"^([A-Za-z0-9_]+)\s*:\s*(.*?)\s*$", raw_line)
        if match:
            values[match.group(1)] = match.group(2).strip("'\"")
    return values


@dataclass(frozen=True)
class StageRow:
    path: str
    branch: str
    orientation: int
    replica: int
    field: float
    J2: float
    D: int
    chi: int
    energy_per_site: float
    chi_energy_shift: float
    G0: float
    G1: float
    G2: float
    delta: float
    middle_fraction: float
    clock_z6: float
    texture: str


@dataclass(frozen=True)
class Comparison:
    J2: float
    D: int
    orientation: int
    n_plaquette: int
    n_dimer: int
    E_plaquette: float
    E_dimer: float
    gap_Ed_minus_Ep: float
    numerical_resolution: float
    gap_over_resolution: float
    plaquette_middle_mean: float
    dimer_middle_mean: float
    plaquette_clock_mean: float
    dimer_clock_mean: float
    branch_textures_survive: bool
    decision: str


def find_lookahead(path: Path) -> Path | None:
    prefix = path.name.replace("_energy_magnetization_correlation.txt", "")
    matches = sorted(path.parent.glob(
        prefix + "_lookahead_*_energy_magnetization_correlation.txt"))
    return matches[-1] if matches else None


def discover(root: Path) -> list[StageRow]:
    rows: list[StageRow] = []
    failures: list[str] = []
    for path in root.rglob("D_*_chi_*_energy_magnetization_correlation.txt"):
        if "lookahead" in path.name:
            continue
        hp_path = path.parent / "hyperparams.yaml"
        if not hp_path.is_file():
            continue
        try:
            hp = read_scalar_hyperparams(hp_path)
            branch = hp.get("vbc_branch", "none")
            if branch not in BRANCHES:
                continue
            field = float(hp["vbc_field"])
            orientation = int(hp["vbc_orientation"])
            replica_match = REPLICA_RE.search(str(path))
            if not replica_match:
                raise ValueError("replica_N is absent from path")
            base: Row = parse_observation(path)
            lookahead_path = find_lookahead(path)
            chi_shift = math.nan
            if lookahead_path is not None:
                lookahead = parse_observation(lookahead_path)
                chi_shift = abs(lookahead.energy_per_site - base.energy_per_site)
            rows.append(StageRow(
                path=str(path), branch=branch, orientation=orientation,
                replica=int(replica_match.group(1)), field=field,
                J2=base.J2, D=base.D, chi=base.chi,
                energy_per_site=base.energy_per_site,
                chi_energy_shift=chi_shift,
                G0=base.G0, G1=base.G1, G2=base.G2,
                delta=base.delta, middle_fraction=base.middle_fraction,
                clock_z6=base.clock_z6, texture=base.texture,
            ))
        except (KeyError, OSError, TypeError, ValueError) as exc:
            failures.append(f"{path}: {exc}")
    if failures:
        print(f"WARNING: skipped {len(failures)} malformed files")
        for failure in failures[:10]:
            print(f"  {failure}")
    return sorted(rows, key=lambda r: (
        r.J2, r.D, r.orientation, r.branch, r.replica, -r.field, r.chi))


def select_highest_chi(rows: list[StageRow]) -> list[StageRow]:
    selected: dict[tuple, StageRow] = {}
    for row in rows:
        key = (row.J2, row.D, row.orientation, row.branch, row.replica, row.field)
        if key not in selected or row.chi > selected[key].chi:
            selected[key] = row
    return sorted(selected.values(), key=lambda r: (
        r.J2, r.D, r.orientation, r.branch, r.replica, -r.field))


def compare_zero_field(rows: list[StageRow], absolute_floor: float) -> list[Comparison]:
    zero = [row for row in rows if abs(row.field) <= 1.0e-14]
    keys = sorted({(r.J2, r.D, r.orientation) for r in zero})
    comparisons: list[Comparison] = []
    for j2, D, orientation in keys:
        by_branch = {
            branch: [r for r in zero if (
                r.J2 == j2 and r.D == D and r.orientation == orientation
                and r.branch == branch)]
            for branch in BRANCHES
        }
        p_rows, d_rows = by_branch["plaquette"], by_branch["dimer-plaquette"]
        if not p_rows or not d_rows:
            continue

        p_energy = min(r.energy_per_site for r in p_rows)
        d_energy = min(r.energy_per_site for r in d_rows)
        gap = d_energy - p_energy
        all_rows = p_rows + d_rows
        chi_errors = [r.chi_energy_shift for r in all_rows
                      if math.isfinite(r.chi_energy_shift)]
        spreads = []
        for branch_rows in (p_rows, d_rows):
            energies = [r.energy_per_site for r in branch_rows]
            if len(energies) >= 2:
                spreads.append(max(energies) - min(energies))
        resolution = max([absolute_floor] + chi_errors + spreads)
        ratio = abs(gap) / resolution

        p_survives = all(r.texture == "plaquette" for r in p_rows)
        d_survives = all(r.texture == "dimer-plaquette" for r in d_rows)
        enough_replicas = len(p_rows) >= 2 and len(d_rows) >= 2
        textures_survive = p_survives and d_survives
        all_textures = {r.texture for r in all_rows}
        if not enough_replicas:
            decision = "UNRESOLVED: fewer than two replicas"
        elif all_textures == {"plaquette"}:
            decision = "PLAQUETTE selected: both continuations converge PVB"
        elif all_textures == {"dimer-plaquette"}:
            decision = "DIMER-PLAQUETTE selected: both continuations converge dimer"
        elif not textures_survive:
            decision = "UNRESOLVED: branch collapsed/mixed inconsistently at h=0"
        elif ratio < 3.0:
            decision = "UNRESOLVED: gap below 3x numerical resolution"
        elif gap > 0.0:
            decision = "PLAQUETTE lower"
        else:
            decision = "DIMER-PLAQUETTE lower"

        comparisons.append(Comparison(
            J2=j2, D=D, orientation=orientation,
            n_plaquette=len(p_rows), n_dimer=len(d_rows),
            E_plaquette=p_energy, E_dimer=d_energy,
            gap_Ed_minus_Ep=gap, numerical_resolution=resolution,
            gap_over_resolution=ratio,
            plaquette_middle_mean=fmean(r.middle_fraction for r in p_rows),
            dimer_middle_mean=fmean(r.middle_fraction for r in d_rows),
            plaquette_clock_mean=fmean(r.clock_z6 for r in p_rows),
            dimer_clock_mean=fmean(r.clock_z6 for r in d_rows),
            branch_textures_survive=textures_survive,
            decision=decision,
        ))
    return comparisons


def write_dataclasses(rows: list, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def plot(rows: list[StageRow], comparisons: list[Comparison], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    colors = {"plaquette": "tab:blue", "dimer-plaquette": "tab:red"}
    markers = {"plaquette": "o", "dimer-plaquette": "s"}

    for key in sorted({(r.J2, r.D, r.orientation, r.branch, r.replica) for r in rows}):
        j2, D, orientation, branch, replica = key
        subset = [r for r in rows if (
            r.J2, r.D, r.orientation, r.branch, r.replica) == key]
        subset.sort(key=lambda r: r.field, reverse=True)
        axes[0].plot(
            [r.field for r in subset], [r.middle_fraction for r in subset],
            marker=markers[branch], color=colors[branch], alpha=0.55,
            ms=3.5, lw=0.8,
        )
    axes[0].axhspan(0.0, 0.25, color="tab:blue", alpha=0.07)
    axes[0].axhspan(0.75, 1.0, color="tab:red", alpha=0.07)
    axes[0].set_xlabel("pinning field h")
    axes[0].set_ylabel("middle fraction (0=PVB, 1=dimer)")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].invert_xaxis()
    axes[0].set_title("Adiabatic branch survival")

    if comparisons:
        x = np.arange(len(comparisons))
        gaps = np.asarray([c.gap_Ed_minus_Ep for c in comparisons])
        errors = 3.0 * np.asarray([c.numerical_resolution for c in comparisons])
        axes[1].errorbar(x, gaps, yerr=errors, fmt="o", color="black", capsize=3)
        axes[1].axhline(0.0, color="black", lw=0.8)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(
            [f"J2={c.J2:g}\nD={c.D}, o={c.orientation}" for c in comparisons],
            rotation=30, ha="right", fontsize=8,
        )
    axes[1].set_ylabel(r"$E_{dimer}-E_{PVB}$ per site (error = $3\epsilon$)")
    axes[1].set_title("Unbiased h=0 variational comparison")

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--absolute-floor", type=float, default=1.0e-6,
                        help="Minimum per-site energy resolution.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input.is_dir():
        raise FileNotFoundError(f"branch result directory does not exist: {args.input}")
    rows = select_highest_chi(discover(args.input))
    if not rows:
        raise RuntimeError("no VBC branch observations found")
    comparisons = compare_zero_field(rows, args.absolute_floor)
    if not comparisons:
        raise RuntimeError("both branches have not reached h=0 for any common point")

    write_dataclasses(rows, args.output_dir / "vbc_branch_hysteresis.csv")
    write_dataclasses(comparisons, args.output_dir / "vbc_branch_comparison.csv")
    plot(rows, comparisons, args.output_dir / "vbc_branch_comparison.pdf")

    print("J2      D  o       Ed-Ep/site   resolution   ratio  decision")
    for comp in comparisons:
        print(f"{comp.J2:0.3f}  {comp.D:2d}  {comp.orientation}  "
              f"{comp.gap_Ed_minus_Ep:+.9e}  {comp.numerical_resolution:.3e}  "
              f"{comp.gap_over_resolution:6.2f}  {comp.decision}")
    print(f"\nWrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
