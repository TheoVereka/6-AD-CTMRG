#!/usr/bin/env python3
"""Import downloaded ordinary-only legacy-Neel results into D345678910."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


#HERE = Path(__file__).resolve().parent
DEFAULT_INCOMING = Path(
    r"D:\HyraiOn\ENS_Lyon\Internship\2026-EPFL\data"
) / "neel_legacy_ordinary_correlation_lengths"
DEFAULT_LEGACY_ROOT = Path(r"D:\HyraiOn\ENS_Lyon\Internship\2026-EPFL\data\D345678910")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--incoming", type=Path, default=DEFAULT_INCOMING)
    parser.add_argument("--legacy-root", type=Path, default=DEFAULT_LEGACY_ROOT)
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--keep-source",
        dest="keep_source",
        action="store_true",
        help="retain downloaded result files (default)",
    )
    source_group.add_argument(
        "--delete-source",
        dest="keep_source",
        action="store_false",
        help="delete a downloaded result only after a successful import",
    )
    parser.set_defaults(keep_source=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    incoming = args.incoming.resolve()
    legacy_root = args.legacy_root.resolve()
    manifests = sorted(incoming.rglob("checkpoint_manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"No checkpoint_manifest.json below {incoming}")
    manifest_path = max(manifests, key=lambda path: path.stat().st_mtime_ns)
    bundle_root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("bundle_kind") != "D345678910_neel_legacy_ordinary_only":
        raise ValueError(f"Wrong bundle kind: {manifest_path}")
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from bundle_utils import (
        ORDINARY_DIRECTIONS,
        is_completed_ordinary_result,
        is_completed_partial_result,
        merge_partial_payloads,
        recover_split_payloads,
        recover_complete_payload,
        partial_result_name,
        result_name,
    )

    imported = skipped = incomplete = 0
    result_root = bundle_root / "results_three_env_ordinary_v5"
    for item in manifest["items"]:
        if not item.get("selected_for_rerun"):
            continue
        ansatz = str(item["ansatz_directory"])
        token, D, j2 = str(item["j2_directory"]), int(item["D"]), float(item["j2"])
        expected_hash = str(item["sha256"])
        current_checkpoint = legacy_root.parent / str(item["original_relative_path"])
        expected_source_hash = str(item["source_checkpoint_sha256"])
        if not current_checkpoint.is_file():
            print(f"MISSING CHECKPOINT J2={j2:g} D={D}: {current_checkpoint}")
            incomplete += 1
            continue
        actual_source_hash = sha256(current_checkpoint)
        stale_manifest = actual_source_hash != expected_source_hash
        if stale_manifest:
            print(f"STALE MANIFEST J2={j2:g} D={D}: checking numerical recovery")
        destination = legacy_root / str(item["legacy_run_relative_path"]) / str(item["legacy_correlation_filename"])
        source = result_root / result_name(ansatz, token, D)
        payload = None
        partials = {
            direction: result_root / partial_result_name(ansatz, token, D, direction)
            for direction in ORDINARY_DIRECTIONS
        }
        try:
            available = {d: json.loads(p.read_text(encoding="utf-8")) for d, p in partials.items() if p.is_file()}
            if len(available) >= 2:
                payload = recover_split_payloads(available, expected_hash)
            elif is_completed_ordinary_result(source, j2=j2, D_bond=D, ansatz_directory=ansatz):
                payload = json.loads(source.read_text(encoding="utf-8"))
                if payload.get("cluster_bundle_provenance", {}).get("checkpoint_sha256") != expected_hash:
                    payload = recover_complete_payload(payload, expected_hash)
            if payload is None:
                raise ValueError("fewer than two directions and no complete result")
            if stale_manifest:
                payload = recover_complete_payload(payload, expected_hash)
                payload["import_recovery"]["current_source_checkpoint_sha256"] = actual_source_hash
                payload["import_recovery"]["manifest_source_checkpoint_sha256"] = expected_source_hash
        except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError) as error:
            print(f"INCOMPLETE J2={j2:g} D={D}: {error}")
            incomplete += 1
            continue
        if payload.get("import_recovery"):
            print(f"RECOVER J2={j2:g} D={D}: {payload['import_recovery']}")
        if destination.is_file() and not args.overwrite:
            try:
                existing = json.loads(destination.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                existing = {}
            signature_keys = ("spectra", "completed_at_utc", "accepted_directions", "import_recovery")
            identical = all(existing.get(k) == payload.get(k) for k in signature_keys)
            older = str(existing.get("completed_at_utc", "")) > str(payload.get("completed_at_utc", ""))
            if identical or older:
                print(f"ALREADY IMPORTED J2={j2:g} D={D}: {destination}")
                skipped += 1
                continue
        print(f"{'WOULD IMPORT' if args.dry_run else 'IMPORT'} {source} -> {destination}")
        if args.dry_run:
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload["checkpoint"] = str(legacy_root.parent / str(item["original_relative_path"]))
        temporary = destination.with_name(destination.name + ".importing")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, destination)
        imported += 1
        if not args.keep_source and source.exists():
            source.unlink()
    print(f"Import summary: imported={imported}, skipped={skipped}, incomplete={incomplete}, dry_run={args.dry_run}.")
    return 1 if incomplete else 0


if __name__ == "__main__":
    raise SystemExit(main())
