#!/usr/bin/env python3
"""Materialise the editable candidate selection and its 12 direction chains."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
from decimal import Decimal
from pathlib import Path


HERE = Path(__file__).resolve().parent
GRID = ("0.26", "0.265", "0.27", "0.275", "0.28", "0.29", "0.30", "0.31", "0.32")
CHI = {10: 120, 11: 140}
EXPECTED = {(10, "plaquette"): 1, (10, "dimer-plaquette"): 1,
            (11, "plaquette"): 1, (11, "dimer-plaquette"): 1}


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def j2_sequences(seed_j2: str) -> list[tuple[str, list[str]]]:
    values = [Decimal(value) for value in GRID]
    seed = Decimal(seed_j2)
    if seed not in values:
        raise ValueError(f"seed J2={seed_j2} is outside the continuation grid")
    index = values.index(seed)
    left = list(reversed(GRID[:index]))
    right = list(GRID[index + 1:])
    if not left or not right:
        raise ValueError(f"seed J2={seed_j2} does not have both directions")
    return [("left", left), ("right", right)]


def main() -> int:
    catalog_rows = read_tsv(HERE / "candidate_catalog.tsv")
    catalog = {row["candidate_id"]: row for row in catalog_rows}
    selections = read_tsv(HERE / "selection.tsv")
    if len(selections) != 4:
        raise ValueError(f"selection.tsv must contain exactly 4 seeds, got {len(selections)}")
    aliases = [row["alias"] for row in selections]
    if aliases != [f"a{i:02d}" for i in range(1, 5)]:
        raise ValueError("aliases must remain a01..a04 so cluster-visible paths hide D")
    if len({row["candidate_id"] for row in selections}) != 4:
        raise ValueError("the four selected candidates must be distinct")

    counts: dict[tuple[int, str], int] = {}
    manifests: list[dict[str, object]] = []
    plans: list[dict[str, object]] = []
    for selected in selections:
        alias = selected["alias"]
        candidate_id = selected["candidate_id"]
        if candidate_id not in catalog:
            raise ValueError(f"unknown candidate_id {candidate_id}")
        row = catalog[candidate_id]
        D = int(row["D"])
        texture = selected["run_texture"]
        if texture not in {"plaquette", "dimer-plaquette"}:
            raise ValueError(f"invalid run_texture {texture}")
        if row["family"] != texture:
            raise ValueError(f"{candidate_id} belongs to {row['family']}, not {texture}")
        if abs(float(row["energy_difference"])) >= 0.001:
            raise ValueError(f"{candidate_id} violates |E-E_original| < 0.001")
        if float(row["relative_delta_difference"]) > 0.20:
            raise ValueError(f"{candidate_id} violates relative Delta difference <= 20%")
        counts[(D, texture)] = counts.get((D, texture), 0) + 1

        pool = HERE / "candidate_pool" / candidate_id
        seed_dir = HERE / "seeds" / alias
        sources = {
            "tensor_best.pt": pool / "tensor_best.pt",
            "observation.txt": pool / "observation.txt",
            "hyperparams.yaml": pool / "hyperparams.yaml",
        }
        for name, source in sources.items():
            if not source.is_file():
                raise FileNotFoundError(f"missing bundled candidate file: {source}")
            seed_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, seed_dir / name)

        manifest = {
            "alias": alias,
            "candidate_id": candidate_id,
            "D": D,
            "chi": CHI[D],
            "seed_J2": row["J2"],
            "run_texture": texture,
            "actual_texture": row["actual_texture"],
            "orientation": int(row["orientation"]),
            "energy": row["energy"],
            "energy_difference": row["energy_difference"],
            "delta": row["delta"],
            "relative_delta_difference": row["relative_delta_difference"],
            "eta": row["eta"],
            "seed_sha256": sha256(seed_dir / "tensor_best.pt"),
            "note": selected["note"],
        }
        manifests.append(manifest)
        (seed_dir / "provenance.json").write_text(
            json.dumps({**row, **manifest}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        for insurance in (1, 2):
            for direction, sequence in j2_sequences(row["J2"]):
                plans.append({
                    "alias": alias,
                    "candidate_id": candidate_id,
                    "D": D,
                    "chi": CHI[D],
                    "seed_J2": row["J2"],
                    "texture": texture,
                    "orientation": row["orientation"],
                    "insurance": insurance,
                    "direction": direction,
                    "j2_sequence": ":".join(sequence),
                    "stage_hours": f"{24 * D / 5:g}",
                })

    if counts != EXPECTED:
        raise ValueError(f"selection family counts are {counts}; expected {EXPECTED}")
    if len(plans) != 16:
        raise AssertionError(f"expected 16 insurance-direction chains, got {len(plans)}")

    with (HERE / "private_manifest.tsv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifests[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(manifests)
    with (HERE / "submission_plan.tsv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(plans[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(plans)

    jobs = sum(len(str(plan["j2_sequence"]).split(":")) for plan in plans)
    print(f"Prepared 4 seeds, 16 chain heads, {jobs - 16} dependencies, {jobs} stage jobs total")
    print("D/chi mapping is recorded only in private_manifest.tsv and submission_plan.tsv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
