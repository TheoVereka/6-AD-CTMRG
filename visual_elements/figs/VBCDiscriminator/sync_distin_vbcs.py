#!/usr/bin/env python3
"""Non-destructively collect replica-1 VBC raw results in data/distinVBCs.

The archive is authoritative: an older legacy-bundle file never overwrites a
newer file copied directly from a cluster.  No destination file is deleted.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DEFAULT_ARCHIVE = REPO.parent / "data" / "distinVBCs"
LEGACY_BUNDLE = REPO / "models" / "VBCPinningClusterBundle"
IZAR_BUNDLE = REPO / "models" / "VBCPinningLyraLBFGS"


@dataclass
class SyncStats:
    copied: int = 0
    updated: int = 0
    unchanged: int = 0
    kept_newer_archive: int = 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sync_replica1_tree(source: Path, destination: Path) -> SyncStats:
    """Merge every file below a replica_1 directory into destination."""
    stats = SyncStats()
    if not source.is_dir():
        return stats
    for source_file in sorted(path for path in source.rglob("*")
                              if path.is_file()
                              and "replica_1" in path.relative_to(source).parts):
        relative = source_file.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source_file, target)
            stats.copied += 1
            continue
        if (source_file.stat().st_size == target.stat().st_size
                and _sha256(source_file) == _sha256(target)):
            stats.unchanged += 1
            continue
        if source_file.stat().st_mtime_ns > target.stat().st_mtime_ns:
            shutil.copy2(source_file, target)
            stats.updated += 1
        else:
            # A direct SCP into the archive may be newer than the old bundle.
            stats.kept_newer_archive += 1
    return stats


def sync_default_sources(archive_root: Path = DEFAULT_ARCHIVE) -> dict[str, SyncStats]:
    archive_root.mkdir(parents=True, exist_ok=True)
    sources = (
        ("legacy Izar", LEGACY_BUNDLE / "Results_VBC_branches",
         archive_root / "Results_Izar_replica1"),
        ("legacy Kuma", LEGACY_BUNDLE / "Results_VBC_three",
         archive_root / "Results_Kuma_replica1"),
        ("new Izar", IZAR_BUNDLE / "Results_Izar_replica1",
         archive_root / "Results_Izar_replica1"),
    )
    return {label: sync_replica1_tree(source, destination)
            for label, source, destination in sources}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE)
    args = parser.parse_args()
    results = sync_default_sources(args.archive_root)
    for label, stats in results.items():
        print(f"{label:12s}: copied={stats.copied}, updated={stats.updated}, "
              f"unchanged={stats.unchanged}, "
              f"kept-newer-archive={stats.kept_newer_archive}")
    print(f"Replica-1 archive: {args.archive_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
