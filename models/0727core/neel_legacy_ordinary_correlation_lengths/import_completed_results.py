#!/usr/bin/env python3
"""Import downloaded ordinary-only legacy-Neel results into D345678910."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_INCOMING = Path(r"D:\HyraiOn\ENS_Lyon\Internship\2026-EPFL\data\neel_legacy_ordinary_correlation_lengths")
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
    parser.add_argument("--keep-source", action="store_true")
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
    sys.path.insert(0, str(bundle_root))
    from bundle_utils import (
        ORDINARY_DIRECTIONS,
        is_completed_ordinary_result,
        is_completed_partial_result,
        merge_partial_payloads,
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
        source = result_root / result_name(ansatz, token, D)
        if not is_completed_ordinary_result(
            source,
            j2=j2,
            D_bond=D,
            ansatz_directory=ansatz,
            checkpoint_sha256=expected_hash,
        ):
            partials = {
                direction: result_root / partial_result_name(ansatz, token, D, direction)
                for direction in ORDINARY_DIRECTIONS
            }
            if all(
                is_completed_partial_result(
                    path,
                    j2=j2,
                    D_bond=D,
                    ansatz_directory=ansatz,
                    direction=direction,
                    checkpoint_sha256=expected_hash,
                )
                for direction, path in partials.items()
            ):
                payloads = {
                    direction: json.loads(path.read_text(encoding="utf-8"))
                    for direction, path in partials.items()
                }
                merged = merge_partial_payloads(payloads)
                if not args.dry_run:
                    temporary = source.with_name(source.name + ".assembling")
                    temporary.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
                    os.replace(temporary, source)
            if not source.is_file() and args.dry_run:
                print(f"WOULD ASSEMBLE AND IMPORT J2={j2:g} D={D}")
                continue
        if not is_completed_ordinary_result(
            source,
            j2=j2,
            D_bond=D,
            ansatz_directory=ansatz,
            checkpoint_sha256=expected_hash,
        ):
            print(f"INCOMPLETE J2={j2:g} D={D}")
            incomplete += 1
            continue
        destination = legacy_root / str(item["legacy_run_relative_path"]) / str(item["legacy_correlation_filename"])
        if destination.is_file() and not args.overwrite:
            if sha256(destination) == sha256(source):
                skipped += 1
                continue
        print(f"{'WOULD IMPORT' if args.dry_run else 'IMPORT'} {source} -> {destination}")
        if args.dry_run:
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = json.loads(source.read_text(encoding="utf-8"))
        payload["checkpoint"] = str(legacy_root.parent / str(item["original_relative_path"]))
        temporary = destination.with_name(destination.name + ".importing")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, destination)
        imported += 1
        if not args.keep_source:
            source.unlink()
    print(f"Import summary: imported={imported}, skipped={skipped}, incomplete={incomplete}, dry_run={args.dry_run}.")
    return 1 if incomplete else 0


if __name__ == "__main__":
    raise SystemExit(main())
