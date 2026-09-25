#!/usr/bin/env python3
"""Plot completed h=0 J2 continuations, split by the seed VBC texture."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


HERE = Path(__file__).resolve().parent
FIGS = HERE.parent
REPO = FIGS.parents[1]
DATA = REPO.parent / "data"
sys.path.insert(0, str(FIGS / "PublicationPlots"))
sys.path.insert(0, str(FIGS))

from publication_common import NN_GROUPS, parse_observable, rms  # noqa: E402
from plot_0713_twoc3_nn_delta import (  # noqa: E402
    RANK_COLORS, RANK_LABELS,
)


DEFAULT_INPUT = DATA / "distinVBCsJ2Continuation" / "Results_Izar_J2_sequences"
DEFAULT_MANIFEST = (REPO / "models" / "VBCJ2SeedContinuationIzar"
                    / "selected_seed_manifest.csv")
DEFAULT_OUTPUT = HERE / "j2_seed_continuations"
RUN_RE = re.compile(
    r"^(s\d+)_J2_([0-9]+p[0-9]+)_D_(\d+)_(plaquette|dimer-plaquette)$"
)
STAGE_RE = re.compile(r"^J2_([0-9]+p[0-9]+)$")
INSURANCE_RE = re.compile(r"^insurance_([12])$")
OBS_RE = re.compile(r"^D_(\d+)_chi_(\d+)_energy_magnetization_correlation\.txt$")
J2_GRID = (0.26, 0.265, 0.27, 0.275, 0.28, 0.29, 0.30, 0.31, 0.32)
TEXTURE_ORDER = ("dimer-plaquette", "plaquette")
TEXTURE_TITLES = {
    "dimer-plaquette": "Dimer-plaquette seed",
    "plaquette": "Plaquette seed",
}
INSURANCE_STYLES = {1: ("-", 1.0), 2: ("--", 0.72)}
SEED_MARKERS = ("o", "s", "^", "D", "P", "X")


@dataclass(frozen=True)
class Seed:
    seed_id: str
    J2: float
    D: int
    chi: int
    texture: str
    observation: Path


@dataclass(frozen=True)
class Point:
    seed_id: str
    seed_texture: str
    D: int
    chi: int
    J2: float
    direction: str
    insurance: int
    ranks: tuple[tuple[float, float], ...]
    observation: str
    is_seed: bool = False


def parse_j2_tag(text: str) -> float:
    return float(text.replace("p", "."))


def ranked_nn(path: Path) -> tuple[tuple[float, float], ...]:
    observation = parse_observable(path)
    groups: list[tuple[float, float]] = []
    for group in NN_GROUPS:
        values = [observation["corr"].get(key) for key in group]
        if any(value is None for value in values):
            missing = [str(key) for key, value in zip(group, values)
                       if value is None]
            raise ValueError(f"{path}: missing NN correlations {missing}")
        samples = [float(value) for value in values]
        groups.append((float(np.mean(samples)), rms(samples)))
    return tuple(sorted(groups, key=lambda item: item[0]))


def read_seeds(manifest: Path) -> dict[str, Seed]:
    if not manifest.is_file():
        raise FileNotFoundError(f"seed manifest is missing: {manifest}")
    seeds: dict[str, Seed] = {}
    manifests = [manifest]
    supplemental = manifest.with_name("d9_supplemental_seed_manifest.csv")
    if manifest.name != supplemental.name and supplemental.is_file():
        manifests.append(supplemental)
    for source_manifest in manifests:
        with source_manifest.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                seed_id = row["seed_id"]
                if seed_id in seeds:
                    raise ValueError(f"duplicate seed id across manifests: {seed_id}")
                relative_tensor = row.get("seed_relative_path", "").strip()
                if not relative_tensor:
                    raise ValueError(
                        f"{source_manifest}: {seed_id} lacks seed_relative_path"
                    )
                observation = (source_manifest.parent / relative_tensor).parent / "observation.txt"
                if not observation.is_file():
                    raise FileNotFoundError(
                        f"seed observation is missing: {observation}"
                    )
                seeds[seed_id] = Seed(
                    seed_id=seed_id, J2=float(row["seed_J2"]), D=int(row["D"]),
                    chi=int(row["run_chi"]), texture=row["selected_texture"],
                    observation=observation,
                )
    if not seeds:
        raise RuntimeError(f"empty seed manifest: {manifest}")
    return seeds


def _read_field(path: Path) -> float:
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        document = json.loads(text)
        if isinstance(document, dict):
            return float(document["vbc_field"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        pass
    match = re.search(r"^vbc_field\s*:\s*([^\s#]+)", text, re.MULTILINE)
    if not match:
        raise ValueError(f"{path}: vbc_field is absent")
    return float(match.group(1))


def discover(root: Path, seeds: dict[str, Seed]) -> tuple[list[Point], int]:
    if not root.is_dir():
        raise FileNotFoundError(f"downloaded result root is missing: {root}")
    points: list[Point] = []
    complete_dirs = {path.parent for path in root.rglob("sweep_results.json")}
    all_stage_dirs = {path for path in root.rglob("J2_*") if path.is_dir()
                      and STAGE_RE.fullmatch(path.name)}
    partial_count = len(all_stage_dirs - complete_dirs)

    for stage in sorted(complete_dirs):
        stage_match = STAGE_RE.fullmatch(stage.name)
        insurance_match = INSURANCE_RE.fullmatch(stage.parent.name)
        direction = stage.parent.parent.name
        run_match = RUN_RE.fullmatch(stage.parent.parent.parent.name)
        if not stage_match or not insurance_match or not run_match:
            raise ValueError(f"unrecognized completed-stage path: {stage}")
        if direction not in {"left", "right"}:
            raise ValueError(f"invalid direction in {stage}")
        seed_id, seed_j2_tag, D_text, texture = run_match.groups()
        if seed_id not in seeds:
            raise ValueError(f"{stage}: seed {seed_id} absent from manifest")
        seed = seeds[seed_id]
        stage_j2 = parse_j2_tag(stage_match.group(1))
        if (seed.D != int(D_text) or seed.texture != texture
                or not math.isclose(seed.J2, parse_j2_tag(seed_j2_tag), abs_tol=1e-12)):
            raise ValueError(f"{stage}: path disagrees with seed manifest")
        if not any(math.isclose(stage_j2, value, abs_tol=1e-12)
                   for value in J2_GRID):
            raise ValueError(f"{stage}: J2 is outside the continuation grid")
        if ((direction == "left" and not stage_j2 < seed.J2)
                or (direction == "right" and not stage_j2 > seed.J2)):
            raise ValueError(f"{stage}: J2 is on the wrong side of its seed")

        observations = [path for path in stage.glob("D_*_chi_*_energy_magnetization_correlation.txt")
                        if OBS_RE.fullmatch(path.name)]
        if len(observations) != 1:
            raise ValueError(
                f"{stage}: expected exactly one base observation, found {len(observations)}"
            )
        observation = observations[0]
        D_obs, chi_obs = map(int, OBS_RE.fullmatch(observation.name).groups())
        if D_obs != seed.D or chi_obs != seed.chi:
            raise ValueError(f"{observation}: D/chi disagrees with manifest")
        hyperparams = stage / "hyperparams.yaml"
        if not hyperparams.is_file() or not math.isclose(
                _read_field(hyperparams), 0.0, abs_tol=1e-14):
            raise ValueError(f"{stage}: not a verified h=0 stage")
        points.append(Point(
            seed_id=seed_id, seed_texture=texture, D=seed.D, chi=chi_obs,
            J2=stage_j2, direction=direction,
            insurance=int(insurance_match.group(1)), ranks=ranked_nn(observation),
            observation=str(observation.resolve()),
        ))
    return sorted(points, key=lambda row: (
        row.D, row.seed_texture, row.seed_id, row.direction,
        row.insurance, row.J2,
    )), partial_count


def seed_point(seed: Seed) -> Point:
    return Point(
        seed_id=seed.seed_id, seed_texture=seed.texture,
        D=seed.D, chi=seed.chi, J2=seed.J2,
        direction="seed", insurance=0, ranks=ranked_nn(seed.observation),
        observation=str(seed.observation.resolve()), is_seed=True,
    )


def limits(rows: list[Point]) -> tuple[float, float]:
    lows = [mean - error for row in rows for mean, error in row.ranks]
    highs = [mean + error for row in rows for mean, error in row.ranks]
    low, high = min(lows), max(highs)
    pad = max(0.005, 0.07 * (high - low))
    return low - pad, high + pad


def plot_D(D: int, points: list[Point], seeds: dict[str, Seed], output: Path) -> int:
    seeds_D = [seed for seed in seeds.values() if seed.D == D]
    by_texture = {texture: [seed for seed in seeds_D if seed.texture == texture]
                  for texture in TEXTURE_ORDER}
    for texture, selected in by_texture.items():
        if not selected:
            raise ValueError(
                f"D={D}: expected at least one {texture} seed"
            )
    plotted_rows = list(points)
    plotted_rows.extend(seed_point(seed) for selected in by_texture.values()
                        for seed in selected)
    y_limits = limits(plotted_rows)

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2), sharex=True, sharey=True,
                             constrained_layout=True)
    for ax, texture in zip(axes, TEXTURE_ORDER):
        selected_seeds = sorted(by_texture[texture], key=lambda item: (item.J2, item.seed_id))
        for seed_index, seed in enumerate(selected_seeds):
            marker = SEED_MARKERS[seed_index % len(SEED_MARKERS)]
            seed_row = seed_point(seed)
            subset = [row for row in points if row.seed_id == seed.seed_id]
            for direction in ("left", "right"):
                for insurance in (1, 2):
                    branch = [row for row in subset
                              if row.direction == direction and row.insurance == insurance]
                    if not branch:
                        continue
                    branch = sorted(branch, key=lambda row: row.J2)
                    series = sorted([seed_row, *branch], key=lambda row: row.J2)
                    linestyle, alpha = INSURANCE_STYLES[insurance]
                    for rank, color in enumerate(RANK_COLORS):
                        ax.errorbar(
                            [row.J2 for row in series],
                            [row.ranks[rank][0] for row in series],
                            yerr=[row.ranks[rank][1] for row in series],
                            fmt=marker, linestyle=linestyle, color=color,
                            markersize=3.8, linewidth=1.1, elinewidth=0.8,
                            capsize=2, alpha=alpha, zorder=3,
                        )
            # Mark every distinct seed once on top of its continuation copies.
            for rank, color in enumerate(RANK_COLORS):
                ax.errorbar(
                    [seed_row.J2], [seed_row.ranks[rank][0]],
                    yerr=[seed_row.ranks[rank][1]], fmt=marker,
                    color=color, markeredgecolor="black", markeredgewidth=0.8,
                    markersize=6.2, elinewidth=0.9, capsize=2, zorder=6,
                )
            ax.axvline(seed.J2, color="0.45", linestyle=":", linewidth=0.8,
                       alpha=0.75, zorder=0)
            ax.text(seed.J2, 0.02, rf"{seed.seed_id} seed $J_2={seed.J2:g}$",
                    rotation=90, ha="right", va="bottom", fontsize=7.2,
                    transform=ax.get_xaxis_transform(), color="0.35")
        ax.set_title(rf"{TEXTURE_TITLES[texture]}, $D={D}$", fontsize=12)
        ax.set_xlabel(r"$J_2$", fontsize=12)
        ax.set_ylim(*y_limits)
        ax.grid(alpha=0.2)
        ax.tick_params(axis="both", labelsize=9)
    axes[0].set_ylabel("NN correlation", fontsize=12)

    all_x = sorted({row.J2 for row in plotted_rows})
    for ax in axes:
        ax.set_xticks(all_x)
        ax.set_xticklabels([f"{value:g}" for value in all_x], rotation=35,
                           ha="right")
    rank_handles = [Line2D([], [], color=color, marker="o", linewidth=1.1,
                           markersize=4.2, label=label)
                    for color, label in zip(RANK_COLORS, RANK_LABELS)]
    insurance_handles = [
        Line2D([], [], color="0.25", marker="o", linestyle=style,
               alpha=alpha, markersize=3.8, label=f"insurance {insurance}")
        for insurance, (style, alpha) in INSURANCE_STYLES.items()
    ]
    fig.legend(handles=[*rank_handles, *insurance_handles],
               loc="outside upper center", ncol=3, frameon=False, fontsize=9)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)

    csv_path = output.with_suffix(".csv")
    fields = ("seed_id", "seed_texture", "D", "chi", "J2", "direction",
              "insurance", "is_seed", "rank1", "rank1_error", "rank2",
              "rank2_error", "rank3", "rank3_error", "observation")
    csv_rows = []
    for row in sorted(plotted_rows, key=lambda item: (
            item.seed_texture, item.direction, item.insurance, item.J2)):
        csv_rows.append({
            "seed_id": row.seed_id, "seed_texture": row.seed_texture,
            "D": row.D, "chi": row.chi, "J2": row.J2,
            "direction": row.direction, "insurance": row.insurance,
            "is_seed": row.is_seed,
            "rank1": row.ranks[0][0], "rank1_error": row.ranks[0][1],
            "rank2": row.ranks[1][0], "rank2_error": row.ranks[1][1],
            "rank3": row.ranks[2][0], "rank3_error": row.ranks[2][1],
            "observation": row.observation,
        })
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"D={D}: plotted {len(points)} completed stages -> {output}")
    return len(points)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    seeds = read_seeds(args.manifest)
    points, partial_count = discover(args.input, seeds)
    if not points:
        raise RuntimeError(f"no individually completed J2 stages found in {args.input}")
    dimensions = sorted({point.D for point in points})
    total = 0
    for D in dimensions:
        rows = [row for row in points if row.D == D]
        output = args.output_dir / f"2C3_NN_ranks_vs_J2_D{D}.pdf"
        total += plot_D(D, rows, seeds, output)
    print(f"Total completed stages plotted: {total}")
    print(f"Partial stage directories ignored (no sweep_results.json): {partial_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
