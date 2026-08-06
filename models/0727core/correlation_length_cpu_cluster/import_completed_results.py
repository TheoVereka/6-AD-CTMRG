#!/usr/bin/env python3
"""Import a fully downloaded CPU-cluster bundle into 0713summary.

With no arguments, this script expects the cluster bundle at
``D:/HyraiOn/ENS_Lyon/Internship/2026-EPFL/data/correlation_length_cpu_cluster``.

It reads completed JSON files from that bundle's
results_three_env_ordinary_v5/ directory and moves them into their directly
plottable locations below 0713summary.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

from bundle_utils import (
    MANIFEST_JSON,
    RESULT_DIRECTORY,
    RESULT_NAME_PATTERN,
    is_completed_ordinary_result,
    load_manifest,
    manifest_index,
    parse_result_name,
)


DEFAULT_SUMMARY_ROOT = Path(
    r"D:\HyraiOn\ENS_Lyon\Internship\2026-EPFL\data\0713summary"
)
DEFAULT_DOWNLOADED_BUNDLE_ROOT = Path(
    r"D:\HyraiOn\ENS_Lyon\Internship\2026-EPFL\data"
) / "correlation_length_cpu_cluster"
# Search the downloaded bundle recursively.  This also tolerates scp placing
# a second correlation_length_cpu_cluster directory below an existing one.
DEFAULT_INCOMING = DEFAULT_DOWNLOADED_BUNDLE_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--incoming",
        type=Path,
        default=DEFAULT_INCOMING,
        help=(
            "Downloaded correlation_length_cpu_cluster tree. Only JSON files "
            "below a results_three_env_ordinary_v5 directory are considered."
        ),
    )
    parser.add_argument(
        "--summary-root", type=Path, default=DEFAULT_SUMMARY_ROOT
    )
    parser.add_argument(
        "--bundle-root",
        type=Path,
        default=None,
        help=(
            "Downloaded bundle holding checkpoint_manifest.json. By default "
            "the newest compatible ordinary manifest below --incoming is selected."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing destination correlation_length.json.",
    )
    parser.add_argument(
        "--keep-source",
        action="store_true",
        help="Copy instead of the default move-after-validation behavior.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and report destinations without changing files.",
    )
    parser.add_argument(
        "--verbose-skips",
        action="store_true",
        help="Print one line for every identical already-imported result.",
    )
    return parser.parse_args()


def within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def ordinary_calculation_signature(payload: dict[str, object]) -> str:
    """Identify numerical output independently of local import metadata."""

    scientific_payload = {
        "schema": payload.get("schema"),
        "schema_version": payload.get("schema_version"),
        "transfer_network_schema": payload.get("transfer_network_schema"),
        "ansatz_directory": payload.get("ansatz_directory"),
        "D_bond": payload.get("D_bond"),
        "chi": payload.get("chi"),
        "dtype": payload.get("dtype"),
        "seed": payload.get("seed"),
        "completed_at_utc": payload.get("completed_at_utc"),
        "ctm": payload.get("ctm"),
        "calculation_hyperparameters": payload.get(
            "calculation_hyperparameters"
        ),
        "spectra": payload.get("spectra"),
    }
    return json.dumps(
        scientific_payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=True,
    )


def completion_time(payload: dict[str, object]) -> datetime | None:
    value = payload.get("completed_at_utc")
    if not isinstance(value, str):
        provenance = payload.get("cluster_bundle_provenance")
        if isinstance(provenance, dict):
            value = provenance.get("completed_utc")
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def main() -> int:
    args = parse_args()
    incoming = args.incoming.resolve()
    summary_root = args.summary_root.resolve()
    if not incoming.is_dir():
        raise NotADirectoryError(incoming)
    if not summary_root.is_dir():
        raise NotADirectoryError(summary_root)

    if args.bundle_root is None:
        manifest_paths = sorted(set(incoming.rglob(MANIFEST_JSON)))
        compatible: list[Path] = []
        for manifest_path in manifest_paths:
            try:
                candidate = load_manifest(manifest_path.parent)
                hashes = candidate.get("solver_files_sha256", {})
                if "compute_three_ordinary_correlation_lengths.py" in hashes:
                    compatible.append(manifest_path)
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
        if not compatible:
            raise FileNotFoundError(
                f"No compatible ordinary {MANIFEST_JSON} found below {incoming}"
            )
        manifest_path = max(
            compatible, key=lambda path: path.stat().st_mtime_ns
        )
        bundle_root = manifest_path.parent.resolve()
        print(f"Using bundle manifest: {manifest_path}")
    else:
        bundle_root = args.bundle_root.resolve()

    manifest = load_manifest(bundle_root)
    index = manifest_index(manifest)
    discovered_candidates = sorted(
        path
        for path in incoming.rglob("*.json")
        if RESULT_NAME_PATTERN.fullmatch(path.name)
        and RESULT_DIRECTORY in path.parts
    )
    if not discovered_candidates:
        print(f"No completed result filenames found below {incoming}.")
        return 0

    # Repeated ``scp -r`` can leave identical nested bundle copies. Select the
    # newest file for each (J2,D) rather than failing the entire import.
    newest_by_key: dict[tuple[str, str, int], Path] = {}
    for path in discovered_candidates:
        key = parse_result_name(path.name)
        previous = newest_by_key.get(key)
        if previous is None or path.stat().st_mtime_ns > previous.stat().st_mtime_ns:
            newest_by_key[key] = path
    candidates = sorted(newest_by_key.values())
    duplicates = len(discovered_candidates) - len(candidates)
    if duplicates:
        print(
            f"Ignoring {duplicates} older duplicate result file(s) from "
            "nested downloaded bundles."
        )

    imported = skipped = kept = failed = 0
    for source in candidates:
        ansatz_directory, j2_directory, D_bond = parse_result_name(
            source.name
        )
        key = (ansatz_directory, j2_directory, D_bond)
        item = index.get(key)
        if item is None:
            print(f"REJECT not present in manifest: {source}")
            failed += 1
            continue
        try:
            with source.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if not is_completed_ordinary_result(
                source,
                j2=float(item["j2"]),
                D_bond=D_bond,
                ansatz_directory=ansatz_directory,
            ):
                raise ValueError("not a complete matching ordinary-v5/v6 result")
            provenance = payload.get("cluster_bundle_provenance")
            if isinstance(provenance, dict) and provenance.get(
                "ansatz_directory", ansatz_directory
            ) != ansatz_directory:
                raise ValueError("provenance ansatz differs from result filename")
        except (
            OSError,
            ValueError,
            TypeError,
            KeyError,
            json.JSONDecodeError,
        ) as error:
            print(f"REJECT invalid or incomplete {source}: {error}")
            failed += 1
            continue

        checkpoint = (
            summary_root / str(item["original_relative_path"])
        ).resolve()
        destination = checkpoint.with_name("correlation_length.json")
        if not within(checkpoint, summary_root):
            print(f"REJECT manifest path escapes summary root: {checkpoint}")
            failed += 1
            continue
        if not checkpoint.is_file():
            print(f"REJECT local checkpoint is missing: {checkpoint}")
            failed += 1
            continue
        if destination.exists() and not args.overwrite:
            if is_completed_ordinary_result(
                destination,
                j2=float(item["j2"]),
                D_bond=D_bond,
                ansatz_directory=ansatz_directory,
            ):
                try:
                    with destination.open("r", encoding="utf-8") as handle:
                        destination_payload = json.load(handle)
                except (OSError, TypeError, json.JSONDecodeError) as error:
                    print(f"REPLACE unreadable destination {destination}: {error}")
                else:
                    source_signature = ordinary_calculation_signature(payload)
                    destination_signature = ordinary_calculation_signature(
                        destination_payload
                    )
                    if source_signature == destination_signature:
                        if args.verbose_skips:
                            print(
                                "SKIP identical already-imported calculation: "
                                f"{destination}"
                            )
                        skipped += 1
                        continue
                    source_completed = completion_time(payload)
                    destination_completed = completion_time(destination_payload)
                    if (
                        source_completed is not None
                        and destination_completed is not None
                        and source_completed > destination_completed
                    ):
                        print(
                            "UPDATE destination from newer cluster calculation: "
                            f"{destination}"
                        )
                    else:
                        print(
                            "SKIP different source because it is not newer than "
                            f"the destination (use --overwrite to force): {source}"
                        )
                        skipped += 1
                        continue
            else:
                print(
                    f"REPLACE obsolete non-ordinary destination: {destination}"
                )

        print(f"{'WOULD IMPORT' if args.dry_run else 'IMPORT'} {source}")
        print(f"  -> {destination}")
        if args.dry_run:
            continue

        cluster_checkpoint = payload.get("checkpoint")
        payload["checkpoint"] = str(checkpoint)
        provenance = payload.get("cluster_bundle_provenance")
        if not isinstance(provenance, dict):
            provenance = {}
            payload["cluster_bundle_provenance"] = provenance
        recorded_hash = provenance.get("checkpoint_sha256")
        if recorded_hash is None:
            provenance_status = "missing_in_legacy_result"
        elif recorded_hash == item["sha256"]:
            provenance_status = "matches_current_manifest"
        else:
            provenance_status = "differs_from_current_manifest"
        provenance.update(
            {
                "ansatz_directory": ansatz_directory,
                "j2_directory": j2_directory,
                "j2": item["j2"],
                "D": D_bond,
                "original_relative_path": item["original_relative_path"],
                "current_manifest_checkpoint_sha256": item["sha256"],
                "checkpoint_provenance_status": provenance_status,
                "cluster_checkpoint": cluster_checkpoint,
                "imported_from": str(source),
                "imported_utc": datetime.now(timezone.utc).isoformat(),
            }
        )

        temporary = destination.with_name(destination.name + ".importing")
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, allow_nan=True)
                handle.write("\n")
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        imported += 1
        if args.keep_source:
            kept += 1
        else:
            source.unlink()

    print(
        f"Import summary: imported={imported}, "
        f"already_present={skipped}, "
        f"sources_kept={kept}, rejected={failed}, "
        f"dry_run={args.dry_run}."
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
