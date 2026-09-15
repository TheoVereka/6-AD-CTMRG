#!/usr/bin/env python3
"""Classify the VBC texture of existing hexagonal-CTMRG observations.

The old ``Delta = rank3-rank1`` diagnostic measures only the magnitude of
three-sublattice NN-bond order.  It discards the sign of the Z6 clock
anisotropy and therefore cannot distinguish

    plaquette:        two equally strong groups, one weak group
    dimer-plaquette:  one strong group, two equally weak groups.

This script retains the three *geometrically labelled* NN groups and writes
two permutation-invariant diagnostics:

    middle_fraction = (rank2-rank1)/(rank3-rank1)
        ideal plaquette = 0, ideal dimer-plaquette = 1

    clock_z6 = (27/2) prod_i(G_i-mean(G))/Delta**3
        ideal plaquette = +1, ideal dimer-plaquette = -1.

Here "strong" means a more-negative antiferromagnetic correlation.  The
three groups are the nine distinct NN bonds of the six-site cell:

    G0 = {AD, CF, EB}                 (reported in env1 and env3)
    G1 = {AF, BC, DE}                 (reported in env1 and env2)
    G2 = {AB, CD, EF}                 (reported in env2 and env3).

The sign convention of ``clock_z6`` is fixed by these physical definitions
and is independent of which Gi is singled out.  Lookahead observations are
excluded unless explicitly requested.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean, pstdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_INPUT = Path(r"D:\HyraiOn\ENS_Lyon\Internship\2026-EPFL\data\0713summary")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "existing_data"

HEADER_RE = re.compile(r"#\s*D=(\d+)\s+chi=(\d+)")
ENERGY_RE = re.compile(
    r"^energy_per_site\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)",
    re.MULTILINE,
)
CORR_RE = re.compile(
    r"^corr_(env[123]_[A-F]{2})\s*=\s*"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)",
    re.MULTILINE,
)
J2_RE = re.compile(r"J2_(\d+p\d+)")

GROUP_KEYS = (
    ("env1_EB", "env1_AD", "env1_CF", "env3_BE", "env3_FC", "env3_DA"),
    ("env1_FA", "env1_DE", "env1_BC", "env2_AF", "env2_CB", "env2_ED"),
    ("env2_DC", "env2_BA", "env2_FE", "env3_CD", "env3_EF", "env3_AB"),
)


@dataclass(frozen=True)
class Row:
    path: str
    J2: float
    D: int
    chi: int
    energy_per_site: float
    G0: float
    G1: float
    G2: float
    group_internal_std_max: float
    rank1: float
    rank2: float
    rank3: float
    delta: float
    gap_strong_middle: float
    gap_middle_weak: float
    middle_fraction: float
    clock_z6: float
    texture: str


def _metadata(path: Path, text: str) -> tuple[float, int, int]:
    header = HEADER_RE.search(text)
    if not header:
        raise ValueError("missing '# D=... chi=...' header")
    D, chi = map(int, header.groups())

    j2_match = J2_RE.search(str(path))
    if j2_match:
        j2 = float(j2_match.group(1).replace("p", "."))
        return j2, D, chi

    hyperparams = path.parent / "hyperparams.yaml"
    try:
        params_text = hyperparams.read_text(encoding="utf-8")
        try:
            params = json.loads(params_text)
            j2 = float(params["J2"])
        except json.JSONDecodeError:
            yaml_match = re.search(r"^J2\s*:\s*([^\s#]+)", params_text, re.MULTILINE)
            if not yaml_match:
                raise ValueError("J2 absent from hyperparams")
            j2 = float(yaml_match.group(1))
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ValueError("cannot determine J2 from path or hyperparams") from exc
    return j2, D, chi


def classify(delta: float, middle_fraction: float, numerical_floor: float) -> str:
    """Return a deliberately conservative texture label.

    The continuous diagnostics, not this thresholded label, should be used in
    fits.  A point is called uniform when its total splitting is no larger than
    twenty times the disagreement between duplicate CTMRG environments (with
    an absolute floor for old files rounded to finite precision).
    """
    if delta <= max(1.0e-5, 20.0 * numerical_floor):
        return "uniform/unresolved"
    if middle_fraction <= 0.25:
        return "plaquette"
    if middle_fraction >= 0.75:
        return "dimer-plaquette"
    return "three-distinct/mixed"


def parse_observation(path: Path) -> Row:
    text = path.read_text(encoding="utf-8", errors="replace")
    j2, D, chi = _metadata(path, text)
    energy_match = ENERGY_RE.search(text)
    if not energy_match:
        raise ValueError("energy_per_site is absent")
    energy = float(energy_match.group(1))
    correlations = {key: float(value) for key, value in CORR_RE.findall(text)}

    missing = sorted({key for group in GROUP_KEYS for key in group} - correlations.keys())
    if missing:
        raise ValueError(f"missing NN correlations: {', '.join(missing)}")

    group_samples = [[correlations[key] for key in keys] for keys in GROUP_KEYS]
    groups = [fmean(samples) for samples in group_samples]
    internal_std = max(pstdev(samples) for samples in group_samples)
    rank1, rank2, rank3 = sorted(groups)
    delta = rank3 - rank1
    gap_sm = rank2 - rank1
    gap_mw = rank3 - rank2

    if delta > 0.0:
        middle_fraction = gap_sm / delta
        centered = [value - fmean(groups) for value in groups]
        clock_z6 = 13.5 * math.prod(centered) / delta**3
    else:
        middle_fraction = math.nan
        clock_z6 = math.nan

    texture = classify(delta, middle_fraction, internal_std)
    return Row(
        path=str(path), J2=j2, D=D, chi=chi, energy_per_site=energy,
        G0=groups[0], G1=groups[1], G2=groups[2],
        group_internal_std_max=internal_std,
        rank1=rank1, rank2=rank2, rank3=rank3, delta=delta,
        gap_strong_middle=gap_sm, gap_middle_weak=gap_mw,
        middle_fraction=middle_fraction, clock_z6=clock_z6, texture=texture,
    )


def discover(input_dir: Path, ansatz: str, include_lookahead: bool) -> list[Row]:
    rows: list[Row] = []
    failures: list[str] = []
    for path in input_dir.rglob("*energy_magnetization_correlation.txt"):
        if not include_lookahead and "lookahead" in path.name:
            continue
        lowered = str(path).lower()
        if ansatz.lower() not in lowered:
            continue
        try:
            rows.append(parse_observation(path))
        except (OSError, UnicodeError, ValueError) as exc:
            failures.append(f"{path}: {exc}")
    if failures:
        print(f"WARNING: skipped {len(failures)} malformed observations")
        for failure in failures[:10]:
            print(f"  {failure}")
    return sorted(rows, key=lambda row: (row.J2, row.D, row.chi, row.path))


def write_csv(rows: list[Row], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def plot(rows: list[Row], output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.5))

    scatter = axes[0].scatter(
        [row.J2 for row in rows], [row.D for row in rows],
        c=[row.middle_fraction for row in rows], cmap="coolwarm", vmin=0.0, vmax=1.0,
        s=48, edgecolor="black", linewidth=0.35,
    )
    colorbar = fig.colorbar(scatter, ax=axes[0], pad=0.02)
    colorbar.set_label(r"$(r_2-r_1)/(r_3-r_1)$")
    axes[0].set_xlabel(r"$J_2/J_1$")
    axes[0].set_ylabel(r"iPEPS $D$")
    axes[0].set_title("VBC texture (0=PVB, 1=dimer)")

    j2_values = sorted({row.J2 for row in rows})
    colors = plt.cm.viridis(np.linspace(0.0, 1.0, len(j2_values)))
    for j2, color in zip(j2_values, colors):
        subset = [row for row in rows if math.isclose(row.J2, j2)]
        axes[1].plot(
            [1.0 / row.D for row in subset], [row.middle_fraction for row in subset],
            "o-", ms=3.5, lw=0.9, color=color, label=f"{j2:g}",
        )
        axes[2].plot(
            [1.0 / row.D for row in subset], [row.clock_z6 for row in subset],
            "o-", ms=3.5, lw=0.9, color=color, label=f"{j2:g}",
        )

    axes[1].axhspan(0.0, 0.25, color="tab:blue", alpha=0.08)
    axes[1].axhspan(0.75, 1.0, color="tab:red", alpha=0.08)
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_xlabel(r"$1/D$")
    axes[1].set_ylabel("middle fraction")
    axes[1].set_title("Branch continuity in D")

    axes[2].axhline(0.0, color="black", lw=0.8)
    axes[2].axhline(+1.0, color="tab:blue", lw=0.7, ls="--")
    axes[2].axhline(-1.0, color="tab:red", lw=0.7, ls="--")
    axes[2].set_ylim(-1.1, 1.1)
    axes[2].set_xlabel(r"$1/D$")
    axes[2].set_ylabel(r"normalized $\mathrm{Re}\,\Psi^3$")
    axes[2].set_title(r"$Z_6$ clock anisotropy")
    axes[2].legend(title=r"$J_2$", fontsize=7, title_fontsize=8, ncols=2)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def print_high_d_summary(rows: list[Row], min_D: int = 8) -> None:
    print("J2      D   chi       E/site       Delta    middle       Z6  texture")
    for row in rows:
        if row.D < min_D:
            continue
        print(
            f"{row.J2:0.3f}  {row.D:2d}  {row.chi:4d}  {row.energy_per_site:+.10f}  "
            f"{row.delta:8.5f}  {row.middle_fraction:8.3f}  "
            f"{row.clock_z6:+7.3f}  {row.texture}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ansatz", default="2tensor_twoC3")
    parser.add_argument("--j2-min", type=float, default=0.27)
    parser.add_argument("--j2-max", type=float, default=0.34)
    parser.add_argument("--include-lookahead", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input.is_dir():
        raise FileNotFoundError(f"input directory does not exist: {args.input}")
    rows = [
        row for row in discover(args.input, args.ansatz, args.include_lookahead)
        if args.j2_min <= row.J2 <= args.j2_max
    ]
    if not rows:
        raise RuntimeError("no matching observations found")

    csv_path = args.output_dir / "existing_twoc3_vbc_texture.csv"
    pdf_path = args.output_dir / "existing_twoc3_vbc_texture.pdf"
    write_csv(rows, csv_path)
    plot(rows, pdf_path)
    print_high_d_summary(rows)
    print(f"\nWrote {csv_path}")
    print(f"Wrote {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
