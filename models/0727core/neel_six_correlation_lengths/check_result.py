#!/usr/bin/env python3
"""Exit successfully only for a complete result from the current checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


REQUIRED_SPECTRA = {
    "env2_generalized",
    "env1_ab_env3_ba_generalized",
    "env3_ab_env1_ba_generalized",
    "env2_ordinary",
    "env1_ab_env3_ba_ordinary",
    "env3_ab_env1_ba_ordinary",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_current(
    result: Path, checkpoint: Path, *, j2: float, D: int
) -> bool:
    try:
        document: dict[str, Any] = json.loads(result.read_text(encoding="utf-8"))
        if document.get("schema") != "neel_six_correlation_lengths":
            return False
        if document.get("schema_version") != 3:
            return False
        if document.get("transfer_network_schema") != "three_geometric_straight_rows_v3":
            return False
        if int(document["D"]) != D:
            return False
        if not math.isclose(float(document["J2"]), j2, rel_tol=0.0, abs_tol=1.0e-12):
            return False
        if document.get("checkpoint_sha256") != _sha256(checkpoint):
            return False
        ctm = document["ctm"]
        if not (
            ctm["converged_ab_within_budget"]
            and ctm["converged_ba_within_budget"]
        ):
            return False
        spectra = document["spectra"]
        return all(
            len(spectra[key]["eigenvalues"]) >= 2 for key in REQUIRED_SPECTRA
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--J2", type=float, required=True)
    parser.add_argument("--D", type=int, required=True)
    args = parser.parse_args()
    return 0 if is_current(args.result, args.checkpoint, j2=args.J2, D=args.D) else 1


if __name__ == "__main__":
    raise SystemExit(main())
