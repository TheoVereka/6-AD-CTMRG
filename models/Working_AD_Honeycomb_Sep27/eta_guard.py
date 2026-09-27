#!/usr/bin/env python3
"""Measure the three NN groups and reject a mixed |eta| state."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


GROUP_KEYS = (
    ("env1_EB", "env1_AD", "env1_CF", "env3_BE", "env3_FC", "env3_DA"),
    ("env1_FA", "env1_DE", "env1_BC", "env2_AF", "env2_CB", "env2_ED"),
    ("env2_DC", "env2_BA", "env2_FE", "env3_CD", "env3_EF", "env3_AB"),
)
CORR_RE = re.compile(
    r"^corr_(env[123]_[A-F]{2})\s*=\s*"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)",
    re.MULTILINE,
)


def measure(path: Path, threshold: float) -> dict[str, object]:
    text = path.read_text(encoding="utf-8", errors="replace")
    correlations = {key: float(value) for key, value in CORR_RE.findall(text)}
    missing = sorted({key for group in GROUP_KEYS for key in group} - correlations.keys())
    if missing:
        raise ValueError(f"missing NN correlations: {', '.join(missing)}")
    groups = tuple(sum(correlations[key] for key in group) / len(group)
                   for group in GROUP_KEYS)
    ranked = sorted(groups)
    delta = ranked[2] - ranked[0]
    if delta <= 0.0:
        raise ValueError(f"non-positive NN splitting: groups={groups}")
    omega = (ranked[1] - ranked[0]) / delta
    eta = 2.0 * omega - 1.0
    return {
        "observation": str(path),
        "groups": groups,
        "ranked_strongest_middle_weakest": ranked,
        "delta": delta,
        "omega": omega,
        "eta": eta,
        "abs_eta": abs(eta),
        "threshold": threshold,
        "accepted_for_continuation": abs(eta) >= threshold,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("observation", type=Path)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = measure(args.observation, args.threshold)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(f"eta={result['eta']:+.12f}; |eta|={result['abs_eta']:.12f}; "
          f"threshold={args.threshold:.6f}; "
          f"accepted={result['accepted_for_continuation']}")
    return 0 if result["accepted_for_continuation"] else 10


if __name__ == "__main__":
    raise SystemExit(main())
