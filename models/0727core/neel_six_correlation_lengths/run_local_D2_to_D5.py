#!/usr/bin/env python3
"""Run every available J2 x D=2,3,4,5 case in a fresh subprocess."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    jobs: list[tuple[float, int, Path]] = []
    for j2_dir in sorted((HERE / "checkpoints").glob("J2_*")):
        j2 = float(j2_dir.name[3:].replace("p", "."))
        for D in (2, 3, 4, 5):
            checkpoint = j2_dir / f"D_{D}" / "tensor_best.pt"
            if checkpoint.is_file():
                jobs.append((j2, D, checkpoint))
    if not jobs:
        raise FileNotFoundError("No local D=2,3,4,5 checkpoints were found.")
    for index, (j2, D, checkpoint) in enumerate(jobs, start=1):
        output = (
            HERE
            / "results"
            / f"J2_{j2:g}".replace(".", "p")
            / f"D_{D}.json"
        )
        command = [
            sys.executable,
            str(HERE / "compute_six_correlation_lengths.py"),
            "--checkpoint",
            str(checkpoint),
            "--J2",
            f"{j2:.12g}",
            "--output",
            str(output),
            "--threads",
            str(args.threads),
        ]
        if args.overwrite:
            command.append("--overwrite")
        if args.seed is not None:
            command.extend(["--seed", str((args.seed + index - 1) % 2**32)])
        print(f"[{index}/{len(jobs)}] J2={j2:g} D={D}", flush=True)
        subprocess.run(command, check=True, cwd=HERE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
