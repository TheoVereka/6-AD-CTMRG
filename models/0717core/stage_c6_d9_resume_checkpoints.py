#!/usr/bin/env python3
"""Stage the C6 D=9 resume checkpoints for the 0717core submission set."""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path


DEFAULT_SOURCE_ROOT = Path(
    r"D:\HyraiOn\ENS_Lyon\Internship\2026-EPFL\data\0713core"
)
DEFAULT_DEST_ROOT = Path(
    r"D:\HyraiOn\ENS_Lyon\Internship\2026-EPFL\6-AD-CTMRG\models\0717core"
)

CHECKPOINTS = (
    {
        "job_id": "3087040",
        "j2": "0.24",
        "source_dir": "1tensor_C6Ypi__J2_0p24_20260713_205413",
        "source_name": "sweep_D9_chi108_best.pt",
        "dest_name": "c6ypi_J2_0p24_D9_chi108_best.pt",
    },
    {
        "job_id": "3087041",
        "j2": "0.25",
        "source_dir": "1tensor_C6Ypi__J2_0p25_20260713_205421",
        "source_name": "sweep_D9_chi108_best.pt",
        "dest_name": "c6ypi_J2_0p25_D9_chi108_best.pt",
    },
    {
        "job_id": "3087042",
        "j2": "0.26",
        "source_dir": "1tensor_C6Ypi__J2_0p26_20260713_205421",
        "source_name": "sweep_D9_chi108_best.pt",
        "dest_name": "c6ypi_J2_0p26_D9_chi108_best.pt",
    },
    {
        "job_id": "3087043",
        "j2": "0.27",
        "source_dir": "1tensor_C6Ypi__J2_0p27_20260713_205445",
        "source_name": "sweep_D9_chi108_best.pt",
        "dest_name": "c6ypi_J2_0p27_D9_chi108_best.pt",
    },
    {
        "job_id": "3087044",
        "j2": "0.28",
        "source_dir": "1tensor_C6Ypi__J2_0p28_20260713_205444",
        "source_name": "sweep_D9_chi108_best.pt",
        "dest_name": "c6ypi_J2_0p28_D9_chi108_best.pt",
    },
    {
        "job_id": "3087045",
        "j2": "0.29",
        "source_dir": "1tensor_C6Ypi__J2_0p29_20260713_205445",
        "source_name": "sweep_D9_chi108_best.pt",
        "dest_name": "c6ypi_J2_0p29_D9_chi108_best.pt",
    },
    {
        "job_id": "3087181",
        "j2": "0.30",
        "source_dir": "1tensor_C6Ypi__J2_0p3_20260715_023652",
        "source_name": "sweep_D9_chi108_best.pt",
        "dest_name": "c6ypi_J2_0p30_D9_chi108_best.pt",
    },
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage the exact C6 D=9 best.pt files used by 0717core resumes."
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--dest-root", type=Path, default=DEFAULT_DEST_ROOT)
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move instead of copying. The default keeps the 0713core originals intact.",
    )
    parser.add_argument(
        "--clean-stale-d8",
        action="store_true",
        help="Remove stale c6ypi_J2_*_D8_chi104_best.pt files from dest-root.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_root = args.source_root.resolve()
    dest_root = args.dest_root.resolve()

    if not source_root.is_dir():
        raise SystemExit(f"Missing source root: {source_root}")
    if not dest_root.is_dir():
        raise SystemExit(f"Missing destination root: {dest_root}")

    for item in CHECKPOINTS:
        src = source_root / item["source_dir"] / item["source_name"]
        dst = dest_root / item["dest_name"]
        if not src.is_file():
            raise SystemExit(f"Missing source for J2={item['j2']} job={item['job_id']}: {src}")

        if args.move:
            shutil.move(str(src), str(dst))
            action = "moved"
        else:
            shutil.copy2(src, dst)
            action = "copied"

        print(
            f"{action}: job={item['job_id']} J2={item['j2']} "
            f"{src.name} -> {dst.name} sha256={sha256(dst)}"
        )

    if args.clean_stale_d8:
        for stale in dest_root.glob("c6ypi_J2_*_D8_chi104_best.pt"):
            stale.unlink()
            print(f"removed stale D8 checkpoint: {stale.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
