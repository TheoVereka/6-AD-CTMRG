#!/usr/bin/env python3
"""Bundle three additional D=9 h=0 seeds and their two-direction plans."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from decimal import Decimal
from pathlib import Path

from prepare_selected_seeds import parse_groups, texture_orientation


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_CANDIDATES = (
    REPO.parent / "data" / "distinVBCsH0TensorCandidates"
    / "selection_20260923_170530" / "all_candidates.csv"
)
GRID = ("0.26", "0.265", "0.27", "0.275", "0.28", "0.29",
        "0.30", "0.31", "0.32")
RUN_CHI = 108
STAGE_HOURS = 24 * 9 / 5  # D/5 days = 43.2 hours.

# Exact candidate identities.  s101/s102 are the two best usable alternatives
# to the existing J2=0.33 dimer seed.  s103 is the best plaquette-like seed
# other than J2=0.30 that still has both directions inside the fixed grid.
SPECS = (
    {
        "seed_id": "s101", "J2": "0.28",
        "source": "izar_replica1_small_h",
        "seed_branch": "dimer-plaquette", "replica": "1",
        "source_orientation": "2", "orientation": "2",
        "texture": "dimer-plaquette",
    },
    {
        "seed_id": "s102", "J2": "0.3",
        "source": "kuma_replica1",
        "seed_branch": "dimer-plaquette", "replica": "1",
        "source_orientation": "0", "orientation": "0",
        "texture": "dimer-plaquette",
    },
    {
        "seed_id": "s103", "J2": "0.31",
        "source": "izar_replica1",
        "seed_branch": "dimer-plaquette", "replica": "1",
        # The source pin was orientation 2, but the final h=0 tensor is a
        # plaquette state whose actual weak-group orientation is 0.
        "source_orientation": "2", "orientation": "0",
        "texture": "plaquette",
    },
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized(value: str) -> str:
    number = Decimal(value)
    text = format(number, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def sequences(seed_j2: str) -> tuple[tuple[str, tuple[str, ...]], ...]:
    seed = Decimal(seed_j2)
    left = tuple(value for value in reversed(GRID) if Decimal(value) < seed)
    right = tuple(value for value in GRID if Decimal(value) > seed)
    if not left or not right:
        raise ValueError(f"seed J2={seed_j2} does not have both directions")
    return (("left", left), ("right", right))


def match(row: dict[str, str], spec: dict[str, str]) -> bool:
    return (
        int(row["D"]) == 9
        and normalized(row["J2"]) == normalized(spec["J2"])
        and row["source"] == spec["source"]
        and row["seed_branch"] == spec["seed_branch"]
        and row["replica"] == spec["replica"]
        and row["orientation"] == spec["source_orientation"]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output", type=Path,
                        default=HERE / "d9_supplemental_seeds")
    parser.add_argument("--manifest", type=Path,
                        default=HERE / "d9_supplemental_seed_manifest.csv")
    parser.add_argument("--plan", type=Path,
                        default=HERE / "d9_supplemental_submission_plan.tsv")
    args = parser.parse_args()

    with args.candidates.open(encoding="utf-8-sig", newline="") as stream:
        candidates = list(csv.DictReader(stream))

    records = []
    plans = []
    for spec in SPECS:
        matches = [row for row in candidates if match(row, spec)]
        if len(matches) != 1:
            raise RuntimeError(
                f"{spec['seed_id']}: expected one exact candidate, found {len(matches)}"
            )
        row = matches[0]
        energy_difference = abs(float(row["energy_difference"]))
        delta_difference = float(row["relative_delta_difference"])
        eta = float(row["eta"])
        expected_sign = 1 if spec["texture"] == "dimer-plaquette" else -1
        if energy_difference > 2e-4 or eta * expected_sign <= 0:
            raise ValueError(f"{spec['seed_id']}: candidate quality/sign changed")
        if spec["seed_id"] != "s101" and delta_difference > 0.15:
            raise ValueError(f"{spec['seed_id']}: Delta mismatch now exceeds 15%")

        tensor = Path(row["tensor_path"])
        observation = Path(row["observation_path"])
        hyperparams = Path(row["hyperparams_path"])
        for source in (tensor, observation, hyperparams):
            if not source.is_file():
                raise FileNotFoundError(source)
        derived_texture, derived_orientation, derived_eta = texture_orientation(
            parse_groups(observation)
        )
        if (derived_texture != spec["texture"]
                or derived_orientation != int(spec["orientation"])
                or not math.isclose(derived_eta, eta, abs_tol=1e-12)):
            raise ValueError(
                f"{spec['seed_id']}: observation texture/orientation/eta "
                "disagrees with the candidate table"
            )

        destination = args.output / spec["seed_id"]
        destination.mkdir(parents=True, exist_ok=True)
        tensor_copy = destination / "tensor_best.pt"
        shutil.copy2(tensor, tensor_copy)
        shutil.copy2(observation, destination / "observation.txt")
        shutil.copy2(hyperparams, destination / "hyperparams.yaml")

        seed_j2 = normalized(row["J2"])
        record = {
            "seed_id": spec["seed_id"],
            "source": row["source"],
            "source_seed_branch": row["seed_branch"],
            "seed_J2": seed_j2,
            "D": 9,
            "run_chi": RUN_CHI,
            "selected_texture": spec["texture"],
            "orientation": int(spec["orientation"]),
            "source_orientation": int(spec["source_orientation"]),
            "eta": f"{eta:.15g}",
            "energy_difference": f"{energy_difference:.15g}",
            "relative_delta_difference": f"{delta_difference:.15g}",
            "seed_relative_path": tensor_copy.relative_to(HERE).as_posix(),
            "sha256": sha256(tensor_copy),
            "source_tensor": str(tensor),
            "source_observation": str(observation),
        }
        records.append(record)
        provenance = dict(record)
        provenance["source_hyperparams"] = str(hyperparams)
        provenance["selection_exception"] = (
            "relative Delta difference is 18.49%, intentionally accepted as "
            "the purest non-J2=0.33 D9 dimer seed"
            if spec["seed_id"] == "s101" else None
        )
        (destination / "provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        for direction, values in sequences(seed_j2):
            plans.append({
                "seed_id": spec["seed_id"], "D": 9, "chi": RUN_CHI,
                "seed_J2": seed_j2, "texture": spec["texture"],
                "orientation": int(spec["orientation"]),
                "direction": direction, "j2_sequence": ":".join(values),
                "stage_hours": f"{STAGE_HOURS:g}",
                "launcher": "izar_3days_sequence.run",
            })

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

    stage_jobs = sum(len(plan["j2_sequence"].split(":")) for plan in plans) * 2
    print(f"Prepared {len(records)} supplemental D9 seeds")
    print(f"Directional sequences: {len(plans)}; insurance chain heads: {len(plans) * 2}")
    print(f"Expanded three-day Slurm jobs: {stage_jobs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
