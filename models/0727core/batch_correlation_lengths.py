#!/usr/bin/env python3
"""Batch two-C3 correlation lengths for the 0713 summary tree.

Each result is written next to its ``tensor_best.pt`` as
``correlation_length.json``.  Existing valid outputs are skipped by default,
so an interrupted sweep can be resumed with the same command.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


DEFAULT_ROOT = Path(
    r"D:\HyraiOn\ENS_Lyon\Internship\2026-EPFL\data\0713summary"
)
DEFAULT_DS = (3, 4, 5, 6)
ANSATZ_DIRECTORY = "2tensor_twoC3"
OUTPUT_NAME = "correlation_length.json"
LOG_NAME = "correlation_length.log"
J2_PATTERN = re.compile(r"^J2_([0-9]+(?:p[0-9]+)?)$")


def _parse_j2(directory_name: str) -> float:
    match = J2_PATTERN.fullmatch(directory_name)
    if match is None:
        raise ValueError(
            f"Cannot infer J2 from directory name {directory_name!r}."
        )
    return float(match.group(1).replace("p", "."))


def _valid_existing_result(
    path: Path,
    checkpoint: Path,
    D_bond: int,
    *,
    j2: float,
    seed: int | None,
) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        hyperparameters = payload["calculation_hyperparameters"]
        recorded_seed = hyperparameters["ctm_random_seed"]
        if seed is None:
            seed_matches = (
                isinstance(recorded_seed, int)
                and not hyperparameters.get("seed_was_user_specified", False)
            )
        else:
            seed_matches = recorded_seed == seed
        return (
            payload.get("transfer_network_schema")
            == "straight_row_env2_v3"
            and
            int(payload["D_bond"]) == D_bond
            and "correlation_length" in payload
            and Path(payload["checkpoint"]).resolve() == checkpoint.resolve()
            and float(hyperparameters["J2"]) == j2
            and seed_matches
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute selected-D two-C3 correlation lengths throughout a "
            "J2 summary tree and save each result beside tensor_best.pt."
        )
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--Ds",
        type=int,
        nargs="+",
        default=list(DEFAULT_DS),
        help="Bond dimensions to process (default: 3 4 5 6; D=2 is disabled).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, or a specific CUDA device (default: auto).",
    )
    parser.add_argument(
        "--chi",
        type=int,
        default=None,
        help="Override checkpoint chi for every selected job.",
    )
    parser.add_argument("--single", action="store_true")
    parser.add_argument("--matrix-free", action="store_true")
    parser.add_argument("--ctm-max-steps", type=int, default=None)
    parser.add_argument("--ctm-conv-tol", type=float, default=None)
    parser.add_argument(
        "--ctm-conv-mode",
        choices=("SVdifference", "Edifference", "both"),
        default=None,
    )
    parser.add_argument("--ctm-e-conv-threshold", type=float, default=None)
    parser.add_argument("--max-intermediate-mib", type=float, default=None)
    parser.add_argument("--eig-tol", type=float, default=None)
    parser.add_argument("--arpack-ncv", type=int, default=None)
    parser.add_argument("--arpack-maxiter", type=int, default=None)
    parser.add_argument("--dense-threshold", type=int, default=None)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional randomized-SVD/CTMRG seed (default: unset).",
    )
    parser.add_argument("--progress-every", type=int, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute even when a valid correlation_length.json exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List selected jobs without computing them.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop at the first failed job instead of continuing.",
    )
    return parser


def _optional_argument(
    command: list[str], name: str, value: object | None
) -> None:
    if value is not None:
        command.extend((name, str(value)))


def main() -> int:
    args = _build_parser().parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Summary root does not exist: {root}")
    if not args.Ds or any(D < 1 for D in args.Ds):
        raise ValueError("--Ds must contain positive integers.")

    script = Path(__file__).with_name("correlation_length.py").resolve()
    jobs: list[tuple[Path, Path, float, int]] = []
    missing: list[Path] = []
    for j2_directory in sorted(root.glob("J2_*")):
        if not j2_directory.is_dir():
            continue
        j2 = _parse_j2(j2_directory.name)
        for D_bond in dict.fromkeys(args.Ds):
            checkpoint = (
                j2_directory
                / ANSATZ_DIRECTORY
                / f"D_{D_bond}"
                / "tensor_best.pt"
            )
            if checkpoint.is_file():
                jobs.append(
                    (
                        checkpoint,
                        checkpoint.with_name(OUTPUT_NAME),
                        j2,
                        D_bond,
                    )
                )
            else:
                missing.append(checkpoint)

    print(
        f"Selected {len(jobs)} checkpoint(s) under {root}; "
        f"{len(missing)} requested checkpoint(s) are missing.",
        flush=True,
    )
    for path in missing:
        print(f"MISSING {path}", flush=True)
    if not jobs:
        return 1

    completed = skipped = failed = 0
    for index, (checkpoint, output, j2, D_bond) in enumerate(jobs, start=1):
        if (
            not args.overwrite
            and output.is_file()
            and _valid_existing_result(
                output,
                checkpoint,
                D_bond,
                j2=j2,
                seed=args.seed,
            )
        ):
            skipped += 1
            print(
                f"[{index}/{len(jobs)}] SKIP D={D_bond}, J2={j2:g}: {output}",
                flush=True,
            )
            continue

        command = [
            sys.executable,
            str(script),
            str(checkpoint),
            "--device",
            args.device,
            "--J2",
            str(j2),
            "--output",
            str(output),
        ]
        _optional_argument(command, "--chi", args.chi)
        _optional_argument(command, "--ctm-max-steps", args.ctm_max_steps)
        _optional_argument(command, "--ctm-conv-tol", args.ctm_conv_tol)
        _optional_argument(command, "--ctm-conv-mode", args.ctm_conv_mode)
        _optional_argument(
            command,
            "--ctm-e-conv-threshold",
            args.ctm_e_conv_threshold,
        )
        _optional_argument(
            command, "--max-intermediate-mib", args.max_intermediate_mib
        )
        _optional_argument(command, "--eig-tol", args.eig_tol)
        _optional_argument(command, "--arpack-ncv", args.arpack_ncv)
        _optional_argument(command, "--arpack-maxiter", args.arpack_maxiter)
        _optional_argument(command, "--dense-threshold", args.dense_threshold)
        _optional_argument(command, "--seed", args.seed)
        _optional_argument(command, "--progress-every", args.progress_every)
        if args.single:
            command.append("--single")
        if args.matrix_free:
            command.append("--matrix-free")

        print(
            f"[{index}/{len(jobs)}] RUN D={D_bond}, J2={j2:g}: {checkpoint}",
            flush=True,
        )
        if args.dry_run:
            print(subprocess.list2cmdline(command), flush=True)
            continue

        log_path = checkpoint.with_name(LOG_NAME)
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if process.returncode == 0 and _valid_existing_result(
            output,
            checkpoint,
            D_bond,
            j2=j2,
            seed=args.seed,
        ):
            completed += 1
            print(
                f"[{index}/{len(jobs)}] DONE D={D_bond}, J2={j2:g}: {output}",
                flush=True,
            )
        else:
            failed += 1
            print(
                f"[{index}/{len(jobs)}] FAILED (exit {process.returncode}): "
                f"see {log_path}",
                flush=True,
            )
            if args.stop_on_error:
                break

    print(
        f"Batch summary: completed={completed}, skipped={skipped}, "
        f"failed={failed}, missing={len(missing)}.",
        flush=True,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
