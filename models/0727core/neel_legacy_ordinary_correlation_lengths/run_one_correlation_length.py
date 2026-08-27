#!/usr/bin/env python3
"""Run or validate one manifest-selected correlation-length calculation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from bundle_utils import (
    CHECKPOINT_DIRECTORY,
    LEGACY_TWOC3_ANSATZ_DIRECTORY,
    ORDINARY_DIRECTIONS,
    RESULT_DIRECTORY,
    is_completed_partial_result,
    is_completed_ordinary_result,
    load_manifest,
    manifest_index,
    merge_partial_payloads,
    parse_j2_directory,
    partial_result_name,
    result_name,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_solver_snapshot(manifest: dict[str, object]) -> None:
    hashes = manifest.get("solver_files_sha256")
    if not isinstance(hashes, dict):
        raise ValueError("Manifest does not record solver source hashes")
    source_root = Path(__file__).resolve().parent
    for filename in (
        "bundle_utils.py",
        "correlation_length.py",
        "core_C3.py",
        "compute_three_ordinary_correlation_lengths.py",
        "run_one_correlation_length.py",
    ):
        expected = hashes.get(filename)
        source = source_root / filename
        if not source.is_file():
            raise FileNotFoundError(source)
        if not isinstance(expected, str) or sha256(source) != expected:
            raise ValueError(
                f"{source} differs from the solver snapshot recorded in "
                "the scratch manifest"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "case",
        nargs="+",
        help=(
            "New form: ansatz_directory J2_directory D. The legacy "
            "two-C3 form J2_directory D remains accepted for already-queued "
            "Slurm scripts."
        ),
    )
    parser.add_argument(
        "--bundle-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help=(
            "Self-contained bundle holding checkpoints, the ordinary result "
            "directory, and manifests."
        ),
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Return success only when a valid completed result exists.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute even if a valid result already exists.",
    )
    parser.add_argument(
        "--direction",
        choices=ORDINARY_DIRECTIONS,
        help="Run one split direction (used automatically for D>=10).",
    )
    return parser.parse_args()


def add_provenance(
    output: Path,
    *,
    item: dict[str, object],
    bundle_root: Path,
) -> None:
    with output.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["cluster_bundle_provenance"] = {
        "j2_directory": item["j2_directory"],
        "ansatz_directory": item["ansatz_directory"],
        "j2": item["j2"],
        "D": item["D"],
        "staged_checkpoint": item["staged_filename"],
        "original_relative_path": item["original_relative_path"],
        "checkpoint_sha256": item["sha256"],
        "bundle_root": str(bundle_root),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=True)
        handle.write("\n")
    os.replace(temporary, output)


def merge_completed_directions(
    *,
    bundle_root: Path,
    ansatz_directory: str,
    j2_directory: str,
    j2: float,
    D_bond: int,
    checkpoint_sha256: str,
) -> bool:
    result_root = bundle_root / RESULT_DIRECTORY
    destination = result_root / result_name(
        ansatz_directory, j2_directory, D_bond
    )
    if is_completed_ordinary_result(
        destination,
        j2=j2,
        D_bond=D_bond,
        ansatz_directory=ansatz_directory,
        checkpoint_sha256=checkpoint_sha256,
    ):
        return True
    partial_paths = {
        direction: result_root
        / partial_result_name(
            ansatz_directory, j2_directory, D_bond, direction
        )
        for direction in ORDINARY_DIRECTIONS
    }
    if not all(
        is_completed_partial_result(
            path,
            j2=j2,
            D_bond=D_bond,
            ansatz_directory=ansatz_directory,
            direction=direction,
            checkpoint_sha256=checkpoint_sha256,
        )
        for direction, path in partial_paths.items()
    ):
        return False
    payloads = {
        direction: json.loads(path.read_text(encoding="utf-8"))
        for direction, path in partial_paths.items()
    }
    merged = merge_partial_payloads(payloads)
    provenance = merged.setdefault("cluster_bundle_provenance", {})
    provenance["split_directions"] = {
        direction: str(path) for direction, path in partial_paths.items()
    }
    temporary = destination.with_name(
        destination.name + f".merging.{os.getpid()}"
    )
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(merged, handle, indent=2, allow_nan=True)
        handle.write("\n")
    os.replace(temporary, destination)
    return is_completed_ordinary_result(
        destination,
        j2=j2,
        D_bond=D_bond,
        ansatz_directory=ansatz_directory,
        checkpoint_sha256=checkpoint_sha256,
    )


def main() -> int:
    args = parse_args()
    if len(args.case) == 3:
        ansatz_directory, j2_directory, D_text = args.case
    elif len(args.case) == 2:
        ansatz_directory = LEGACY_TWOC3_ANSATZ_DIRECTORY
        j2_directory, D_text = args.case
    else:
        raise ValueError(
            "Expected ansatz_directory J2_directory D, or legacy "
            "J2_directory D"
        )
    D_bond = int(D_text)
    bundle_root = args.bundle_root.resolve()
    j2 = parse_j2_directory(j2_directory)
    if D_bond < 1:
        raise ValueError("D must be positive")

    manifest = load_manifest(bundle_root)
    item = manifest_index(manifest).get(
        (ansatz_directory, j2_directory, D_bond)
    )
    if item is None:
        print(
            f"No checkpoint in manifest for "
            f"({ansatz_directory}, {j2_directory}, D={D_bond}).",
            file=sys.stderr,
        )
        return 2

    checkpoint = (
        bundle_root / CHECKPOINT_DIRECTORY / item["staged_filename"]
    )
    output = (
        bundle_root
        / RESULT_DIRECTORY
        / (
            partial_result_name(
                ansatz_directory, j2_directory, D_bond, args.direction
            )
            if args.direction is not None
            else result_name(ansatz_directory, j2_directory, D_bond)
        )
    )
    expected_checkpoint_sha256 = str(item["sha256"])
    print(
        f"CHECKPOINT_SHA256={expected_checkpoint_sha256}",
        flush=True,
    )

    canonical_output = (
        bundle_root
        / RESULT_DIRECTORY
        / result_name(ansatz_directory, j2_directory, D_bond)
    )
    completed_existing = (
        is_completed_ordinary_result(
            canonical_output,
            j2=j2,
            D_bond=D_bond,
            ansatz_directory=ansatz_directory,
            checkpoint_sha256=expected_checkpoint_sha256,
        )
        or is_completed_partial_result(
            output,
            j2=j2,
            D_bond=D_bond,
            ansatz_directory=ansatz_directory,
            direction=args.direction,
            checkpoint_sha256=expected_checkpoint_sha256,
        )
        if args.direction is not None
        else is_completed_ordinary_result(
            output,
            j2=j2,
            D_bond=D_bond,
            ansatz_directory=ansatz_directory,
            checkpoint_sha256=expected_checkpoint_sha256,
        )
    )
    if args.check_only:
        return 0 if completed_existing else 1
    if completed_existing and not args.overwrite:
        print(f"SKIP completed ordinary result: {output}", flush=True)
        return 0
    validate_solver_snapshot(manifest)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    output.parent.mkdir(parents=True, exist_ok=True)
    solver = Path(__file__).resolve().with_name(
        "compute_three_ordinary_correlation_lengths.py"
    )
    if not solver.is_file():
        raise FileNotFoundError(solver)
    command = [
        sys.executable,
        "-u",
        str(solver),
        str(checkpoint),
        "--ansatz-directory",
        ansatz_directory,
        "--J2",
        format(j2, ".12g"),
        "--progress-every",
        "10",
        "--output",
        str(output),
    ]
    if args.direction is not None:
        # All three split jobs must reconstruct the same CTM environments.
        # A checkpoint-derived seed keeps CTMRG/rSVD identical across jobs;
        # the solver still applies its fixed per-direction eigensolver offset.
        common_seed = int(expected_checkpoint_sha256[:8], 16)
        command.extend(
            ("--direction", args.direction, "--seed", str(common_seed))
        )
    print("RUN " + subprocess.list2cmdline(command), flush=True)
    process = subprocess.run(command, check=False)
    if process.returncode != 0:
        print(
            f"three-environment ordinary solver failed with exit code "
            f"{process.returncode}.",
            file=sys.stderr,
            flush=True,
        )
        return process.returncode

    valid_without_provenance = (
        is_completed_partial_result(
            output,
            j2=j2,
            D_bond=D_bond,
            ansatz_directory=ansatz_directory,
            direction=args.direction,
        )
        if args.direction is not None
        else is_completed_ordinary_result(
            output,
            j2=j2,
            D_bond=D_bond,
            ansatz_directory=ansatz_directory,
        )
    )
    if not valid_without_provenance:
        print(f"Solver did not produce a valid result: {output}", file=sys.stderr)
        return 1
    add_provenance(output, item=item, bundle_root=bundle_root)
    valid_with_provenance = (
        is_completed_partial_result(
            output,
            j2=j2,
            D_bond=D_bond,
            ansatz_directory=ansatz_directory,
            direction=args.direction,
            checkpoint_sha256=expected_checkpoint_sha256,
        )
        if args.direction is not None
        else is_completed_ordinary_result(
            output,
            j2=j2,
            D_bond=D_bond,
            ansatz_directory=ansatz_directory,
            checkpoint_sha256=expected_checkpoint_sha256,
        )
    )
    if not valid_with_provenance:
        print(f"Result failed validation after provenance: {output}", file=sys.stderr)
        return 1
    if args.direction is not None:
        # The last finishing direction creates the canonical three-direction
        # JSON. Brief retries cover simultaneous Slurm-job completion.
        for _ in range(10):
            if merge_completed_directions(
                bundle_root=bundle_root,
                ansatz_directory=ansatz_directory,
                j2_directory=j2_directory,
                j2=j2,
                D_bond=D_bond,
                checkpoint_sha256=expected_checkpoint_sha256,
            ):
                print("MERGED all three split directions", flush=True)
                break
            time.sleep(1.0)
    print(f"DONE {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
