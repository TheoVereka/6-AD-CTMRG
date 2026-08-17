#!/usr/bin/env python3
"""Collect every C3-CTM-compatible 0713summary tensor_best.pt checkpoint.

Checkpoints below every other ansatz directory are deliberately ignored.
"""

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
    C3_COMPATIBLE_ANSATZ_DIRECTORIES,
    CHECKPOINT_DIRECTORY,
    MANIFEST_JSON,
    MANIFEST_TSV,
    is_completed_ordinary_result,
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
    sources = sorted(
        source
        for ansatz in C3_COMPATIBLE_ANSATZ_DIRECTORIES
        for source in summary_root.glob(
            f"J2_*/{ansatz}/D_*/tensor_best.pt"
        )
    )
    for source in sources:
        j2_directory = source.parents[2].name
        ansatz_directory = source.parents[1].name
        j2 = parse_j2_directory(j2_directory)
        D_directory = source.parent.name
        try:
            D_bond = int(D_directory.removeprefix("D_"))
        except ValueError as error:
            raise ValueError(f"Invalid D directory: {source.parent}") from error
        if D_directory != f"D_{D_bond}" or D_bond < 1:
            raise ValueError(f"Invalid D directory: {source.parent}")
        if D_bond < 3:
            continue

        relative = source.relative_to(summary_root)
        source_hash = sha256(source)
        correlation_path = source.with_name("correlation_length.json")
        if not correlation_path.is_file():
            selection_reason = "missing_correlation"
        elif is_completed_ordinary_result(
            correlation_path,
            j2=j2,
            D_bond=D_bond,
            ansatz_directory=ansatz_directory,
            checkpoint_sha256=source_hash,
        ):
            selection_reason = "current_result_matches_tensor"
        elif is_completed_ordinary_result(
            correlation_path,
            j2=j2,
            D_bond=D_bond,
            ansatz_directory=ansatz_directory,
        ):
            selection_reason = "result_checkpoint_hash_missing_or_mismatched"
        else:
            selection_reason = "missing_or_incomplete_ordinary_result"
        items.append(
            {
                "j2_directory": j2_directory,
                "ansatz_directory": ansatz_directory,
                "j2": j2,
                "D": D_bond,
                "source": source,
                "original_relative_path": relative.as_posix(),
                "staged_filename": staged_checkpoint_name(
                    ansatz_directory, j2_directory, D_bond
                ),
                "size_bytes": source.stat().st_size,
                "sha256": source_hash,
                "selected_for_rerun": selection_reason
                != "current_result_matches_tensor",
                "selection_reason": selection_reason,
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
    all_items: list[dict[str, object]],
    selected_items: list[dict[str, object]],
) -> None:
    serializable_items = [
        {key: value for key, value in item.items() if key != "source"}
        for item in all_items
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
        "schema_version": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_summary_root": str(summary_root),
        "c3_compatible_ansatz_directories": list(
            C3_COMPATIBLE_ANSATZ_DIRECTORIES
        ),
        "solver_files_sha256": solver_files,
        "selection_policy": (
            "D>=3 and no complete ordinary-v5/v6 result whose "
            "cluster_bundle_provenance.checkpoint_sha256 equals the current "
            "tensor_best.pt SHA256"
        ),
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
                "ansatz_directory",
                "j2",
                "D",
                "staged_filename",
                "original_relative_path",
                "sha256",
            )
        )
        selected_keys = {
            (
                str(item["ansatz_directory"]),
                str(item["j2_directory"]),
                int(item["D"]),
            )
            for item in selected_items
        }
        for item in serializable_items:
            key = (
                str(item["ansatz_directory"]),
                str(item["j2_directory"]),
                int(item["D"]),
            )
            if key not in selected_keys:
                continue
            writer.writerow(
                (
                    item["j2_directory"],
                    item["ansatz_directory"],
                    format(float(item["j2"]), ".12g"),
                    item["D"],
                    item["staged_filename"],
                    item["original_relative_path"],
                    item["sha256"],
                )
            )
    os.replace(tsv_temporary, tsv_path)


def clear_checkpoint_directory(
    checkpoint_directory: Path, bundle_root: Path
) -> int:
    expected = (bundle_root / CHECKPOINT_DIRECTORY).resolve()
    actual = checkpoint_directory.resolve()
    if actual != expected:
        raise ValueError(f"Refusing to clear unexpected directory: {actual}")
    removed = 0
    for path in checkpoint_directory.iterdir():
        if path.is_dir():
            raise IsADirectoryError(
                f"Refusing to recursively remove unexpected directory: {path}"
            )
        path.unlink()
        removed += 1
    return removed


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
        help=argparse.SUPPRESS,
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
            f"No C3-compatible tensor_best.pt files below {summary_root}"
        )
    keys = [
        (
            str(item["ansatz_directory"]),
            str(item["j2_directory"]),
            int(item["D"]),
        )
        for item in discovered
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate (ansatz,J2,D) checkpoints were discovered")

    selected = [
        item for item in discovered if bool(item["selected_for_rerun"])
    ]
    current = len(discovered) - len(selected)
    print(
        f"Discovered {len(discovered)} C3-compatible D>=3 checkpoint(s): "
        f"current={current}, selected_for_rerun={len(selected)}."
    )
    print(
        "Ansatz filter: "
        + ", ".join(C3_COMPATIBLE_ANSATZ_DIRECTORIES)
        + "; all other ansatz checkpoints are ignored."
    )
    if args.dry_run:
        for item in selected:
            print(
                f"WOULD COPY [{item['selection_reason']}] "
                f"{item['original_relative_path']} -> "
                f"{CHECKPOINT_DIRECTORY}/{item['staged_filename']}"
            )
        return 0

    checkpoint_directory = bundle_root / CHECKPOINT_DIRECTORY
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    removed = clear_checkpoint_directory(checkpoint_directory, bundle_root)
    copied = 0
    for item in selected:
        source = Path(item["source"])
        destination = checkpoint_directory / str(item["staged_filename"])
        source_hash = str(item["sha256"])

        atomic_copy(source, destination)
        if sha256(destination) != source_hash:
            raise OSError(f"Hash verification failed after copying {source}")
        copied += 1
        print(
            f"COPIED [{item['selection_reason']}] {source} -> "
            f"{destination.name}"
        )

    write_manifests(bundle_root, summary_root, discovered, selected)
    print(
        f"Complete: cleared={removed}, copied={copied}, "
        f"manifest_all_items={len(discovered)}, "
        f"submit_manifest_items={len(selected)}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
