#!/usr/bin/env python3
"""Collect and rename every 0713summary two-C3 tensor_best.pt checkpoint."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from bundle_utils import (
    ANSATZ_DIRECTORY,
    CHECKPOINT_DIRECTORY,
    MANIFEST_JSON,
    MANIFEST_TSV,
    parse_j2_directory,
    staged_checkpoint_name,
)


DEFAULT_SUMMARY_ROOT = Path(
    r"D:\HyraiOn\ENS_Lyon\Internship\2026-EPFL\data\0713summary"
)
DEFAULT_BUNDLE_ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover(summary_root: Path) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for source in sorted(
        summary_root.glob(
            f"J2_*/{ANSATZ_DIRECTORY}/D_*/tensor_best.pt"
        )
    ):
        j2_directory = source.parents[2].name
        j2 = parse_j2_directory(j2_directory)
        D_directory = source.parent.name
        try:
            D_bond = int(D_directory.removeprefix("D_"))
        except ValueError as error:
            raise ValueError(f"Invalid D directory: {source.parent}") from error
        if D_directory != f"D_{D_bond}" or D_bond < 1:
            raise ValueError(f"Invalid D directory: {source.parent}")

        relative = source.relative_to(summary_root)
        items.append(
            {
                "j2_directory": j2_directory,
                "j2": j2,
                "D": D_bond,
                "source": source,
                "original_relative_path": relative.as_posix(),
                "staged_filename": staged_checkpoint_name(
                    j2_directory, D_bond
                ),
                "size_bytes": source.stat().st_size,
            }
        )
    return items


def atomic_copy(source: Path, destination: Path) -> None:
    temporary = destination.with_name(destination.name + ".copying")
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_manifests(
    bundle_root: Path,
    summary_root: Path,
    items: list[dict[str, object]],
) -> None:
    serializable_items = [
        {key: value for key, value in item.items() if key != "source"}
        for item in items
    ]
    solver_files: dict[str, str] = {}
    for filename in (
        "correlation_length.py",
        "core_C3.py",
        "compute_three_ordinary_correlation_lengths.py",
    ):
        solver_path = bundle_root / filename
        if not solver_path.is_file():
            raise FileNotFoundError(
                f"Required solver file is absent from bundle: {solver_path}"
            )
        solver_files[filename] = sha256(solver_path)

    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_summary_root": str(summary_root),
        "ansatz_directory": ANSATZ_DIRECTORY,
        "solver_files_sha256": solver_files,
        "items": serializable_items,
    }

    json_path = bundle_root / MANIFEST_JSON
    json_temporary = json_path.with_name(json_path.name + ".tmp")
    with json_temporary.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    os.replace(json_temporary, json_path)

    tsv_path = bundle_root / MANIFEST_TSV
    tsv_temporary = tsv_path.with_name(tsv_path.name + ".tmp")
    with tsv_temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            (
                "j2_directory",
                "j2",
                "D",
                "staged_filename",
                "original_relative_path",
                "sha256",
            )
        )
        for item in serializable_items:
            writer.writerow(
                (
                    item["j2_directory"],
                    format(float(item["j2"]), ".12g"),
                    item["D"],
                    item["staged_filename"],
                    item["original_relative_path"],
                    item["sha256"],
                )
            )
    os.replace(tsv_temporary, tsv_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-root", type=Path, default=DEFAULT_SUMMARY_ROOT
    )
    parser.add_argument(
        "--bundle-root", type=Path, default=DEFAULT_BUNDLE_ROOT
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace a staged checkpoint if its content differs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover and report without copying or writing manifests.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary_root = args.summary_root.resolve()
    bundle_root = args.bundle_root.resolve()
    if not summary_root.is_dir():
        raise NotADirectoryError(summary_root)

    discovered = discover(summary_root)
    if not discovered:
        raise FileNotFoundError(
            f"No two-C3 tensor_best.pt files below {summary_root}"
        )
    keys = [
        (str(item["j2_directory"]), int(item["D"]))
        for item in discovered
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate (J2,D) checkpoints were discovered")

    print(
        f"Discovered {len(discovered)} two-C3 checkpoint(s) under "
        f"{summary_root}."
    )
    if args.dry_run:
        for item in discovered:
            print(
                f"WOULD COPY {item['original_relative_path']} -> "
                f"{CHECKPOINT_DIRECTORY}/{item['staged_filename']}"
            )
        return 0

    checkpoint_directory = bundle_root / CHECKPOINT_DIRECTORY
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    copied = skipped = 0
    for item in discovered:
        source = Path(item["source"])
        destination = checkpoint_directory / str(item["staged_filename"])
        source_hash = sha256(source)
        item["sha256"] = source_hash

        if destination.exists():
            destination_hash = sha256(destination)
            if destination_hash == source_hash:
                skipped += 1
                print(f"UNCHANGED {destination.name}")
                continue
            if not args.overwrite:
                raise FileExistsError(
                    f"{destination} differs from {source}; pass --overwrite "
                    "to replace it."
                )

        atomic_copy(source, destination)
        if sha256(destination) != source_hash:
            raise OSError(f"Hash verification failed after copying {source}")
        copied += 1
        print(f"COPIED {source} -> {destination.name}")

    write_manifests(bundle_root, summary_root, discovered)
    referenced = {str(item["staged_filename"]) for item in discovered}
    stale = sorted(
        path.name
        for path in checkpoint_directory.glob("tensor_best__J2_*__D_*.pt")
        if path.name not in referenced
    )
    print(
        f"Complete: copied={copied}, unchanged={skipped}, "
        f"manifest_items={len(discovered)}."
    )
    if stale:
        print(
            "WARNING: unreferenced staged checkpoint(s) were retained: "
            + ", ".join(stale)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
