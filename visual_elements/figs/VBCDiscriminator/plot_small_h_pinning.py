#!/usr/bin/env python3
"""Fit and plot only the isolated Izar replica-1 data with h <= 0.005."""

from __future__ import annotations

import argparse
from pathlib import Path

from analyze_existing_twoc3 import DEFAULT_INPUT as DEFAULT_ORIGINAL_INPUT
from fit_pinned_correlations import HERE, REPO, run_analysis


SMALL_H_MAX = 0.005
DEFAULT_INPUT = (REPO.parent / "data" / "distinVBCsSmallH"
                 / "Results_Izar_replica1")
DEFAULT_OUTPUT = HERE / "small_h_pinning_extrapolation"
DEFAULT_CSV_OUTPUT = (REPO.parent / "data" / "processed"
                      / "VBCPinningSmallHQuadraticExtrapolation")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="filtered Results_Izar_replica1 tree")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--csv-output-dir", type=Path,
                        default=DEFAULT_CSV_OUTPUT)
    parser.add_argument("--original-input", type=Path,
                        default=DEFAULT_ORIGINAL_INPUT)
    parser.add_argument("--dimensions", type=int, nargs="+",
                        default=(6, 7, 8, 9))
    args = parser.parse_args()

    dimensions = tuple(sorted(set(args.dimensions)))
    summaries, omissions = run_analysis(
        (("Izar", args.input),),
        args.output_dir,
        args.csv_output_dir,
        dimensions,
        args.original_input,
        h_max=SMALL_H_MAX,
        write_png=True,
    )

    fields = sorted({field for row in summaries
                     for field in map(float, row.positive_fields.split(";"))})
    if fields and max(fields) > SMALL_H_MAX + 1.0e-12:
        raise AssertionError(f"small-h filter leaked fields: {fields}")
    print(f"Small-h fits: {len(summaries)} cluster/J2/D/pin groups")
    print(f"Positive fields actually fitted: {fields}")
    print(f"Incomplete groups reported: {len(omissions)}")
    print(f"PDF/PNG: {args.output_dir}")
    print(f"CSV:     {args.csv_output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
