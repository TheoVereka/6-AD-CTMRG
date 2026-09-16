#!/usr/bin/env python3
"""Plot complete and partial Kuma three-source pinning continuations.

Finite-h energies belong to different Hamiltonians and are shown only as
continuation diagnostics.  Only h=0 points enter source-energy comparisons.
One seeded replica cannot by itself establish a phase decision.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_branch_runs import find_lookahead, read_scalar_hyperparams
from analyze_existing_twoc3 import parse_observation


HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE.parents[2] / "models" / "VBCPinningClusterBundle" / "Results_VBC_three"
DEFAULT_OUTPUT = HERE / "three_source_comparison"
BRANCHES = ("plaquette", "dimer-plaquette", "rank-split")
COLORS = {
    "plaquette": "tab:blue",
    "dimer-plaquette": "tab:red",
    "rank-split": "tab:green",
}
MARKERS = {"plaquette": "o", "dimer-plaquette": "s", "rank-split": "^"}
LABELS = {
    "plaquette": "plaquette source",
    "dimer-plaquette": "dimer source",
    "rank-split": "rank-split source",
}
ATTRS = {
    "plaquette": "plaquette",
    "dimer-plaquette": "dimer",
    "rank-split": "rank_split",
}
REPLICA_RE = re.compile(r"replica_(\d+)")
EXPECTED_DS = tuple(range(5, 12))
EXPECTED_FIELDS = (0.08, 0.04, 0.02, 0.01, 0.0)


@dataclass(frozen=True)
class Stage:
    path: str
    branch: str
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
    hours_budget: float = math.nan


@dataclass(frozen=True)
class ZeroFieldComparison:
    J2: float
    D: int
    E_plaquette: float
    E_dimer: float
    E_rank_split: float
    excess_plaquette: float
    excess_dimer: float
    excess_rank_split: float
    lowest_source: str
    lowest_endpoint_texture: str
    gap_second_minus_best: float
    numerical_resolution: float
    gap_over_resolution: float
    plaquette_middle_fraction: float
    dimer_middle_fraction: float
    rank_split_middle_fraction: float
    plaquette_clock_z6: float
    dimer_clock_z6: float
    rank_split_clock_z6: float
    plaquette_texture: str
    dimer_texture: str
    rank_split_texture: str
    assessment: str


def discover(root: Path) -> list[Stage]:
    rows: list[Stage] = []
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
            replica_match = REPLICA_RE.search(str(path))
            if replica_match is None:
                raise ValueError("replica_N is absent from path")
            obs = parse_observation(path)
            lookahead_path = find_lookahead(path)
            chi_shift = math.nan
            if lookahead_path is not None:
                lookahead = parse_observation(lookahead_path)
                chi_shift = abs(lookahead.energy_per_site - obs.energy_per_site)
            rows.append(Stage(
                path=str(path), branch=branch,
                replica=int(replica_match.group(1)), field=float(hp["vbc_field"]),
                J2=obs.J2, D=obs.D, chi=obs.chi,
                energy_per_site=obs.energy_per_site,
                chi_energy_shift=chi_shift,
                G0=obs.G0, G1=obs.G1, G2=obs.G2, delta=obs.delta,
                middle_fraction=obs.middle_fraction, clock_z6=obs.clock_z6,
                texture=obs.texture,
                hours_budget=float(hp.get("hours", "nan")),
            ))
        except (KeyError, OSError, TypeError, ValueError) as exc:
            failures.append(f"{path}: {exc}")
    if failures:
        print(f"WARNING: skipped {len(failures)} malformed files")
        for failure in failures[:10]:
            print(f"  {failure}")
    return sorted(rows, key=lambda r: (r.J2, r.D, r.branch, r.replica, -r.field, r.chi))


def select_highest_chi(rows: list[Stage]) -> list[Stage]:
    selected: dict[tuple, Stage] = {}
    for row in rows:
        key = (row.J2, row.D, row.branch, row.replica, row.field)
        if key not in selected or row.chi > selected[key].chi:
            selected[key] = row
    return sorted(selected.values(), key=lambda r: (r.J2, r.D, r.branch, r.replica, -r.field))


def compare_zero_field(rows: list[Stage], absolute_floor: float) -> list[ZeroFieldComparison]:
    zero = [row for row in rows if abs(row.field) <= 1.0e-14]
    output: list[ZeroFieldComparison] = []
    for j2, D in sorted({(r.J2, r.D) for r in zero}):
        best: dict[str, Stage] = {}
        for branch in BRANCHES:
            candidates = [r for r in zero if r.J2 == j2 and r.D == D and r.branch == branch]
            if candidates:
                best[branch] = min(candidates, key=lambda r: r.energy_per_site)
        if set(best) != set(BRANCHES):
            continue

        energies = {branch: best[branch].energy_per_site for branch in BRANCHES}
        ordered = sorted(energies, key=energies.get)
        lowest, second = ordered[:2]
        emin = energies[lowest]
        gap = energies[second] - emin
        chi_shifts = [r.chi_energy_shift for r in best.values()
                      if math.isfinite(r.chi_energy_shift)]
        resolution = max([absolute_floor] + chi_shifts)
        ratio = gap / resolution
        endpoint = best[lowest].texture
        textures = {point.texture for point in best.values()}
        if len(textures) == 1:
            assessment = (f"SAME ENDPOINT TEXTURE ({endpoint}); source-energy gaps "
                          "do not compare distinct phases")
        elif ratio < 3.0:
            assessment = "UNRESOLVED: lowest two sources within 3x numerical resolution"
        else:
            assessment = "SOURCE MINIMUM ONLY: distinct textures, one seeded replica"

        p = best["plaquette"]
        d = best["dimer-plaquette"]
        r = best["rank-split"]
        output.append(ZeroFieldComparison(
            J2=j2, D=D,
            E_plaquette=p.energy_per_site,
            E_dimer=d.energy_per_site,
            E_rank_split=r.energy_per_site,
            excess_plaquette=p.energy_per_site - emin,
            excess_dimer=d.energy_per_site - emin,
            excess_rank_split=r.energy_per_site - emin,
            lowest_source=lowest, lowest_endpoint_texture=endpoint,
            gap_second_minus_best=gap, numerical_resolution=resolution,
            gap_over_resolution=ratio,
            plaquette_middle_fraction=p.middle_fraction,
            dimer_middle_fraction=d.middle_fraction,
            rank_split_middle_fraction=r.middle_fraction,
            plaquette_clock_z6=p.clock_z6,
            dimer_clock_z6=d.clock_z6,
            rank_split_clock_z6=r.clock_z6,
            plaquette_texture=p.texture,
            dimer_texture=d.texture,
            rank_split_texture=r.texture,
            assessment=assessment,
        ))
    return output


def write_csv(rows: list, output: Path) -> None:
    if not rows:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def plot_continuation(rows: list[Stage], output: Path) -> None:
    dimensions = sorted({r.D for r in rows})
    quantities = (
        ("energy_per_site", "E/site"),
        ("delta", "NN splitting Delta"),
        ("middle_fraction", "middle fraction q"),
        ("clock_z6", "clock K6"),
    )
    fig, axes = plt.subplots(len(dimensions), 4,
                             figsize=(14.0, max(3.0, 2.0 * len(dimensions))),
                             squeeze=False)
    for row_index, D in enumerate(dimensions):
        for branch in BRANCHES:
            subset = sorted((r for r in rows if r.D == D and r.branch == branch),
                            key=lambda r: r.field, reverse=True)
            if not subset:
                continue
            for column, (attribute, _) in enumerate(quantities):
                axes[row_index, column].plot(
                    [r.field for r in subset],
                    [getattr(r, attribute) for r in subset],
                    color=COLORS[branch], marker=MARKERS[branch], ms=3.3,
                    lw=0.9, label=LABELS[branch])
        axes[row_index, 0].set_ylabel(f"D={D}")
        for ax in axes[row_index]:
            ax.invert_xaxis()
            ax.grid(alpha=0.2)
            if row_index == len(dimensions) - 1:
                ax.set_xlabel("pinning field h")
    for ax, (_, title) in zip(axes[0], quantities):
        ax.set_title(title)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Available continuation points; finite-h energies are not phase comparisons")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_nn_groups(rows: list[Stage], output: Path) -> None:
    dimensions = sorted({r.D for r in rows})
    fig, axes = plt.subplots(2, 4, figsize=(13.0, 6.5), squeeze=False)
    line_styles = ("-", "--", ":")
    for ax, D in zip(axes.flat, dimensions):
        for branch in BRANCHES:
            subset = sorted((r for r in rows if r.D == D and r.branch == branch),
                            key=lambda r: r.field, reverse=True)
            for group, style in enumerate(line_styles):
                if subset:
                    ax.plot([r.field for r in subset],
                            [getattr(r, f"G{group}") for r in subset],
                            color=COLORS[branch], linestyle=style,
                            marker=MARKERS[branch], ms=2.4, lw=0.9,
                            label=f"{branch} G{group}" if D == dimensions[0] else None)
        ax.set_title(f"D={D}")
        ax.set_xlabel("pinning field h")
        ax.set_ylabel("NN correlation")
        ax.invert_xaxis()
        ax.grid(alpha=0.2)
    for ax in axes.flat[len(dimensions):]:
        ax.set_visible(False)
    fig.legend(*axes.flat[0].get_legend_handles_labels(),
               loc="lower center", ncol=3, fontsize=7)
    fig.suptitle("Geometrically labelled NN groups; lower is stronger AF")
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_fixed_h_vs_inverse_D(rows: list[Stage], output: Path) -> None:
    fields = sorted({r.field for r in rows}, reverse=True)
    quantities = (
        ("energy_per_site", "E/site"),
        ("delta", "NN splitting Delta"),
        ("middle_fraction", "middle fraction q"),
        ("clock_z6", "clock K6"),
    )
    fig, axes = plt.subplots(len(fields), 4,
                             figsize=(14.0, max(3.0, 2.15 * len(fields))),
                             squeeze=False)
    for row_index, field in enumerate(fields):
        for branch in BRANCHES:
            subset = sorted((r for r in rows if r.field == field and r.branch == branch),
                            key=lambda r: 1.0 / r.D)
            if not subset:
                continue
            for column, (attribute, _) in enumerate(quantities):
                axes[row_index, column].plot(
                    [1.0 / r.D for r in subset],
                    [getattr(r, attribute) for r in subset],
                    color=COLORS[branch], marker=MARKERS[branch], ms=3.3,
                    lw=0.9, label=LABELS[branch])
        axes[row_index, 0].set_ylabel(f"h={field:g}")
        for ax in axes[row_index]:
            ax.grid(alpha=0.2)
            if row_index == len(fields) - 1:
                ax.set_xlabel("1/D")
    for ax, (_, title) in zip(axes[0], quantities):
        ax.set_title(title)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Fixed-h trends; only h=0 energy curves share the same Hamiltonian")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def write_stage_status(root: Path, rows: list[Stage], output: Path) -> dict[str, int]:
    observed = {(r.J2, r.D, r.branch, r.replica, r.field): r for r in rows}
    status_rows: list[dict[str, object]] = []
    counts: dict[str, int] = {}
    for D in EXPECTED_DS:
        for branch in BRANCHES:
            for field in EXPECTED_FIELDS:
                key = (0.3, D, branch, 1, field)
                point = observed.get(key)
                stage_dir = (root / "J2_0.30" / f"D_{D}" / "orientation_unknown" /
                             branch / "replica_1" / f"h_{str(field).replace('.', 'p')}")
                # Discover the actual orientation from the copied output tree.
                paths = list((root / "J2_0.30" / f"D_{D}").glob(
                    f"orientation_*/{branch}/replica_1/h_{str(field).replace('.', 'p')}"))
                if paths:
                    stage_dir = paths[0]
                checkpoints = list(stage_dir.glob("sweep_D*_chi*_best.pt"))
                if point is not None and checkpoints:
                    status = "observation+checkpoint"
                elif point is not None:
                    status = "observation_only"
                elif checkpoints:
                    status = "checkpoint_only"
                else:
                    status = "not_yet_observed"
                counts[status] = counts.get(status, 0) + 1
                status_rows.append(dict(J2=0.3, D=D, branch=branch, replica=1,
                                        field=field, status=status,
                                        chi=point.chi if point else "",
                                        hours_budget=point.hours_budget if point else "",
                                        stage_directory=str(stage_dir)))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(status_rows[0]))
        writer.writeheader()
        writer.writerows(status_rows)
    return counts


def plot_zero_field(comparisons: list[ZeroFieldComparison], output: Path) -> None:
    ordered = sorted(comparisons, key=lambda c: 1.0 / c.D)
    x = np.asarray([1.0 / c.D for c in ordered])
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.1))
    for branch in BRANCHES:
        attr = ATTRS[branch]
        axes[0].plot(x, 1.0e6 * np.asarray([getattr(c, f"excess_{attr}") for c in ordered]),
                     color=COLORS[branch], marker=MARKERS[branch], label=LABELS[branch])
        axes[1].plot(x, [getattr(c, f"{attr}_middle_fraction") for c in ordered],
                     color=COLORS[branch], marker=MARKERS[branch])
        axes[2].plot(x, [getattr(c, f"{attr}_clock_z6") for c in ordered],
                     color=COLORS[branch], marker=MARKERS[branch])
    axes[0].set_ylabel(r"$10^6(E-E_{\min})$ per site")
    axes[1].set_ylabel("middle fraction q")
    axes[2].set_ylabel(r"$K_6$")
    axes[1].axhline(0.0, color="tab:blue", alpha=0.25, lw=0.8)
    axes[1].axhline(1.0, color="tab:red", alpha=0.25, lw=0.8)
    axes[2].axhline(0.0, color="black", alpha=0.35, lw=0.8)
    for ax in axes:
        ax.set_xlabel("1/D")
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Unbiased h=0 endpoints (one seeded replica)")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--absolute-floor", type=float, default=1.0e-6)
    args = parser.parse_args()
    if not args.input.is_dir():
        raise FileNotFoundError(args.input)
    rows = select_highest_chi(discover(args.input))
    counts = write_stage_status(args.input, rows, args.output_dir / "stage_status.csv")
    if not rows:
        print(f"No readable stage observations yet; status: {counts}")
        print(f"Wrote {args.output_dir / 'stage_status.csv'}")
        return 0
    comparisons = compare_zero_field(rows, args.absolute_floor)

    write_csv(rows, args.output_dir / "three_source_hysteresis.csv")
    write_csv([r for r in rows if abs(r.field) <= 1.0e-14],
              args.output_dir / "available_h0_endpoints.csv")
    plot_continuation(rows, args.output_dir / "continuation_vs_h.pdf")
    plot_nn_groups(rows, args.output_dir / "nn_groups_vs_h.pdf")
    plot_fixed_h_vs_inverse_D(rows, args.output_dir / "fixed_h_vs_inverse_D.pdf")
    if comparisons:
        write_csv(comparisons, args.output_dir / "three_source_zero_field.csv")
        plot_zero_field(comparisons, args.output_dir / "three_source_zero_field.pdf")

    print(f"Readable stage observations: {len(rows)} / 105 expected; status: {counts}")
    print("D  lowest source       endpoint texture       gap/resolution  assessment")
    for comp in comparisons:
        print(f"{comp.D:2d} {comp.lowest_source:19s} {comp.lowest_endpoint_texture:22s} "
              f"{comp.gap_over_resolution:8.2f}  {comp.assessment}")
    if not comparisons:
        print("No D has all three h=0 endpoints yet; finite-h plots and CSV are available.")
    print(f"\nWrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
