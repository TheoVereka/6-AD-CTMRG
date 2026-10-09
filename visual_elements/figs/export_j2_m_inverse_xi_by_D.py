#!/usr/bin/env python3
"""Export processed Neel/2C3 data per (ansatz, D), defaulting to 0.2 <= J2 <= 0.24."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


PROCESSED = Path(__file__).resolve().parents[3] / "data" / "processed"
SOURCES = (
    ("neel_legacy_plots", "neel_symmetrized", "0507D45678910_Neel"),
    ("windows_plots", "2tensor_twoC3", "0713summary_2C3"),
)
HEADER = ("J2", "m", "inverse_correlation_length")


def collect(
    processed: Path, j2_min: float = 0.2, j2_max: float = 0.24,
) -> dict[tuple[str, int], list[tuple[str, str, str]]]:
    groups = defaultdict(list)
    seen = set()
    for folder, ansatz, label in SOURCES:
        paths = sorted((processed / folder).glob("J2_*.csv"))
        if not paths:
            raise ValueError(f"No processed CSV files in {processed / folder}")
        for path in paths:
            j2 = float(path.stem[3:].replace("p", "."))
            if not math.isfinite(j2):
                raise ValueError(f"Invalid J2 in {path}")
            if not j2_min <= j2 <= j2_max:
                continue
            with path.open(encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if not {"D", "ansatz", "m", "1/xi"}.issubset(reader.fieldnames or []):
                    raise ValueError(f"Missing source columns in {path}")
                for row in reader:
                    if row["ansatz"] != ansatz:
                        continue
                    d = int(row["D"])
                    key = (label, d, j2)
                    if key in seen:
                        raise ValueError(f"Duplicate (ansatz, D, J2): {key}")
                    seen.add(key)
                    m, inv_xi = row["m"].strip(), row["1/xi"].strip()
                    if not m or not math.isfinite(float(m)):
                        raise ValueError(f"Invalid magnetization in {path}, D={d}")
                    if inv_xi and (not math.isfinite(float(inv_xi)) or float(inv_xi) < 0):
                        raise ValueError(f"Invalid inverse correlation length in {path}, D={d}")
                    groups[(label, d)].append((format(j2, ".12g"), m, inv_xi))
    for rows in groups.values():
        rows.sort(key=lambda row: float(row[0]))
    return dict(groups)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=Path, default=PROCESSED)
    parser.add_argument("--output", type=Path, default=PROCESSED / "J2_m_inverse_xi_by_ansatz_D")
    parser.add_argument("--extension", choices=("csv", "cdv"), default="csv")
    parser.add_argument("--j2-min", type=float, default=0.2)
    parser.add_argument("--j2-max", type=float, default=0.24)
    args = parser.parse_args()
    if not (math.isfinite(args.j2_min) and math.isfinite(args.j2_max) and args.j2_min <= args.j2_max):
        parser.error("J2 bounds must be finite and j2-min <= j2-max")
    groups = collect(args.processed, args.j2_min, args.j2_max)
    output = args.output.resolve()
    destinations = {
        key: output / f"{key[0]}_D{key[1]}.{args.extension}" for key in groups
    }
    existing = [path for path in destinations.values() if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing exports: {existing}")
    output.mkdir(parents=True, exist_ok=True)
    for key, rows in sorted(groups.items()):
        path = destinations[key]
        with path.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(HEADER)
            writer.writerows(rows)
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            assert tuple(next(reader)) == HEADER
            assert list(reader) == [list(row) for row in rows], path
        missing = sum(not row[2] for row in rows)
        print(f"{path.name}: {len(rows)} rows, {missing} missing inverse correlation lengths")
    print(f"Verified {len(groups)} files / {sum(map(len, groups.values()))} rows in {output}")


if __name__ == "__main__":
    main()
