#!/usr/bin/env python3
"""Prepare the manually selected D<=9 tensors and directional J2 plans."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
from decimal import Decimal
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_SELECTION = (
    REPO.parent / "data" / "distinVBCsH0TensorCandidates"
    / "selection_20260923_170530" / "manually_selected.csv"
)
GRID = ("0.26", "0.265", "0.27", "0.275", "0.28", "0.29",
        "0.30", "0.31", "0.32")
CHI_BY_D = {6: 72, 7: 91, 8: 104, 9: 108}
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_groups(path: Path) -> tuple[float, float, float]:
    text = path.read_text(encoding="utf-8", errors="replace")
    correlations = {key: float(value) for key, value in CORR_RE.findall(text)}
    missing = sorted({key for group in GROUP_KEYS for key in group}
                     - correlations.keys())
    if missing:
        raise ValueError(f"{path}: missing NN correlations: {', '.join(missing)}")
    return tuple(sum(correlations[key] for key in group) / len(group)
                 for group in GROUP_KEYS)


def texture_orientation(groups: tuple[float, float, float]) -> tuple[str, int, float]:
    order = sorted(range(3), key=lambda index: (groups[index], index))
    rank1, rank2, rank3 = (groups[index] for index in order)
    delta = rank3 - rank1
    if delta <= 0.0:
        raise ValueError(f"unsplit seed cannot define a VBC orientation: {groups}")
    middle_fraction = (rank2 - rank1) / delta
    eta = 2.0 * middle_fraction - 1.0
    if eta < 0.0:
        return "plaquette", order[2], eta
    return "dimer-plaquette", order[0], eta


def decimal_text(value: str) -> str:
    number = Decimal(value)
    text = format(number, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def sequences(seed_j2: str) -> list[tuple[str, tuple[str, ...]]]:
    seed = Decimal(seed_j2)
    left = tuple(value for value in reversed(GRID) if Decimal(value) < seed)
    right = tuple(value for value in GRID if Decimal(value) > seed)
    result = []
    if left:
        result.append(("left", left))
    if right:
        result.append(("right", right))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--output", type=Path, default=HERE / "seeds")
    parser.add_argument("--manifest", type=Path,
                        default=HERE / "selected_seed_manifest.csv")
    parser.add_argument("--plan", type=Path,
                        default=HERE / "submission_plan.tsv")
    args = parser.parse_args()

    if not args.selection.is_file():
        raise FileNotFoundError(f"manual selection is missing: {args.selection}")

    with args.selection.open(encoding="utf-8-sig", newline="") as stream:
        selected = list(csv.DictReader(stream))
    selected = [row for row in selected if int(row["D"]) <= 9]
    if not selected:
        raise RuntimeError("manual selection contains no D<=9 tensors")

    records: list[dict[str, str | int | float]] = []
    plans: list[dict[str, str | int | float]] = []
    seen_identity: set[tuple[str, int, str]] = set()
    for ordinal, row in enumerate(selected, start=1):
        D = int(row["D"])
        if D not in CHI_BY_D:
            raise ValueError(f"unsupported Izar D={D}")
        seed_j2 = decimal_text(row["J2"])
        texture = row["selected_texture"].strip()
        if texture not in {"plaquette", "dimer-plaquette"}:
            raise ValueError(f"invalid selected texture: {texture}")
        if row.get("accepted", "").strip().lower() not in {"true", "1", "yes"}:
            raise ValueError(f"manually selected row is not accepted: {row}")

        observation = Path(row["observation_path"])
        hyperparams = Path(row["hyperparams_path"]) if row["hyperparams_path"] else None
        tensor_text = row.get("copied_tensor", "").strip() or row["tensor_path"]
        tensor = Path(tensor_text)
        if not observation.is_file():
            raise FileNotFoundError(f"missing seed observation: {observation}")
        if not tensor.is_file():
            raise FileNotFoundError(f"missing seed tensor: {tensor}")
        groups = parse_groups(observation)
        derived_texture, orientation, eta = texture_orientation(groups)
        if derived_texture != texture:
            raise ValueError(
                f"CSV/observation texture disagreement for {tensor}: "
                f"{texture} versus {derived_texture}"
            )
        csv_orientation = row.get("orientation", "").strip()
        if csv_orientation and int(csv_orientation) != orientation:
            raise ValueError(
                f"CSV/observation orientation disagreement for {tensor}: "
                f"{csv_orientation} versus {orientation}"
            )

        identity = (seed_j2, D, texture)
        if identity in seen_identity:
            raise ValueError(f"duplicate manually selected seed: {identity}")
        seen_identity.add(identity)
        seed_id = f"s{ordinal:03d}"
        destination = args.output / seed_id / "tensor_best.pt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(tensor, destination)
        shutil.copy2(observation, destination.parent / "observation.txt")
        if hyperparams is not None and hyperparams.is_file():
            shutil.copy2(hyperparams, destination.parent / "hyperparams.yaml")

        record: dict[str, str | int | float] = {
            "seed_id": seed_id,
            "source": row["source"],
            "source_seed_branch": row["seed_branch"],
            "seed_J2": seed_j2,
            "D": D,
            "run_chi": CHI_BY_D[D],
            "selected_texture": texture,
            "orientation": orientation,
            "eta": f"{eta:.15g}",
            "seed_relative_path": destination.relative_to(HERE).as_posix(),
            "sha256": sha256(destination),
            "source_tensor": str(tensor),
            "source_observation": str(observation),
        }
        records.append(record)

        stage_hours = 24 * D / 5  # (D/5) days * 24 hours/day
        for direction, values in sequences(seed_j2):
            if D == 7:
                launcher = "izar_3days_sequence.run"
            elif D in {8, 9}:
                launcher = "izar_7days_sequence.run"
            else:
                raise ValueError(
                    f"the requested external walltime/QoS is undefined for D={D}"
                )
            plans.append({
                "seed_id": seed_id,
                "D": D,
                "chi": CHI_BY_D[D],
                "seed_J2": seed_j2,
                "texture": texture,
                "orientation": orientation,
                "direction": direction,
                "j2_sequence": ":".join(values),
                "stage_hours": f"{stage_hours:g}",
                "launcher": launcher,
            })

        provenance = dict(record)
        provenance["G0"], provenance["G1"], provenance["G2"] = groups
        provenance["directions"] = [
            {"direction": direction, "J2": list(values)}
            for direction, values in sequences(seed_j2)
        ]
        (destination.parent / "provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    plan_fields = ("seed_id", "D", "chi", "seed_J2", "texture",
                   "orientation", "direction", "j2_sequence",
                   "stage_hours", "launcher")
    with args.plan.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=plan_fields,
                                delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(plans)

    three_day = sum(int(plan["D"]) == 7 for plan in plans) * 2
    seven_day = sum(int(plan["D"]) in {8, 9} for plan in plans) * 2
    print(f"Prepared {len(records)} manually selected D<=9 seeds")
    print(f"Directional sequences: {len(plans)}")
    total_stage_jobs = sum(len(str(plan["j2_sequence"]).split(":"))
                           for plan in plans) * 2
    print(f"Insurance chain heads: {three_day} three-day + "
          f"{seven_day} seven-day = {three_day + seven_day}")
    print(f"Expanded Slurm stage jobs: {total_stage_jobs}")
    print(f"Manifest: {args.manifest}")
    print(f"Plan:     {args.plan}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
