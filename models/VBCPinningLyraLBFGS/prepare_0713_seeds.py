#!/usr/bin/env python3
"""Build the self-contained Lyra seed tree from local 0713summary data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import re
import shutil
from pathlib import Path


J2_VALUES = (
    "0.275",
    "0.27",
    "0.28",
    "0.265",
    "0.29",
    "0.26",
    "0.30",
    "0.31",
    "0.32",
    "0.33",
    "0.34",
)
D_VALUES = tuple(range(6, 12))
GROUP_KEYS = (
    ("env1_EB", "env1_AD", "env1_CF", "env3_BE", "env3_FC", "env3_DA"),
    ("env1_FA", "env1_DE", "env1_BC", "env2_AF", "env2_CB", "env2_ED"),
    ("env2_DC", "env2_BA", "env2_FE", "env3_CD", "env3_EF", "env3_AB"),
)
CORR_RE = re.compile(
    r"^corr_(env[123]_[A-F]{2})\s*=\s*"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)",
    re.MULTILINE,
)


def source_tag(J2: str) -> str:
    # 0713summary uses J2_0p3, while this bundle consistently uses J2_0p30.
    return J2.rstrip("0").rstrip(".").replace(".", "p")


def target_tag(J2: str) -> str:
    return J2.replace(".", "p")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_groups(path: Path) -> tuple[float, float, float]:
    text = path.read_text(encoding="utf-8", errors="replace")
    values = {key: float(value) for key, value in CORR_RE.findall(text)}
    missing = sorted({key for group in GROUP_KEYS for key in group} - values.keys())
    if missing:
        raise ValueError(f"{path}: missing correlations: {', '.join(missing)}")
    return tuple(sum(values[key] for key in group) / len(group)
                 for group in GROUP_KEYS)


def texture_and_orientation(groups: tuple[float, float, float]) -> tuple:
    order = sorted(range(3), key=lambda group: (groups[group], group))
    rank1, rank2, rank3 = (groups[group] for group in order)
    delta = rank3 - rank1
    if delta <= 0:
        raise ValueError(f"cannot orient an unsplit seed: {groups}")
    middle_fraction = (rank2 - rank1) / delta
    centered = [value - sum(groups) / 3.0 for value in groups]
    clock = 13.5 * math.prod(centered) / delta ** 3
    if middle_fraction <= 0.25:
        texture = "plaquette"
    elif middle_fraction >= 0.75:
        texture = "dimer-plaquette"
    else:
        texture = "three-distinct/mixed"
    # Continue the seed's existing Z6 orientation: a plaquette-like seed has
    # one weak group, while a dimer-like seed has one strong group.  The 0.5
    # boundary continuously selects the nearer ideal clock sector.
    orientation = order[2] if middle_fraction < 0.5 else order[0]
    return middle_fraction, clock, texture, orientation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(r"D:\HyraiOn\ENS_Lyon\Internship\2026-EPFL\data\0713summary"),
    )
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "seeds")
    parser.add_argument("--manifest", type=Path,
                        default=Path(__file__).resolve().parent / "seed_manifest.csv")
    args = parser.parse_args()

    records = []
    for J2 in J2_VALUES:
        for target_D in D_VALUES:
            seed_D = target_D
            source_dir = (args.source / f"J2_{source_tag(J2)}" / "2tensor_twoC3"
                          / f"D_{seed_D}")
            if not (source_dir / "tensor_best.pt").is_file():
                if target_D == 11:
                    print(f"Skipping J2={J2}, D=11: no native 0713summary seed")
                    continue
                raise FileNotFoundError(
                    f"no 0713summary seed for J2={J2}, D={target_D}"
                )
            source_tensor = source_dir / "tensor_best.pt"
            source_obs = source_dir / "energy_magnetization_correlation.txt"
            if not source_tensor.is_file() or not source_obs.is_file():
                raise FileNotFoundError(f"incomplete seed source: {source_dir}")

            destination = (args.output / f"J2_{target_tag(J2)}"
                           / f"D_{target_D}" / "tensor_best.pt")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_tensor, destination)
            groups = parse_groups(source_obs)
            fraction, clock, texture, orientation = texture_and_orientation(groups)
            records.append({
                "J2": J2,
                "target_D": target_D,
                "seed_D": seed_D,
                "seed_relative_path": destination.relative_to(args.manifest.parent).as_posix(),
                "sha256": sha256(destination),
                "G0": f"{groups[0]:.15g}",
                "G1": f"{groups[1]:.15g}",
                "G2": f"{groups[2]:.15g}",
                "middle_fraction": f"{fraction:.15g}",
                "clock_z6": f"{clock:.15g}",
                "seed_texture": texture,
                "orientation": orientation,
                "source_path": str(source_tensor),
            })

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"Prepared {len(records)} seeds in {args.output}")
    print(f"Manifest: {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
