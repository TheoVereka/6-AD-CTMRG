#!/usr/bin/env python3
"""Collect the lowest-energy completed 0713core results into 0713summary.

A result is complete when its non-lookahead observable file exists and contains
``energy_per_site``. For every (J2, ansatz, D), only the most negative energy
is retained. Re-running is safe: a summary is replaced only by lower energy.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CORE = Path(r"D:\HyraiOn\ENS_Lyon\Internship\2026-EPFL\data\0713core")
DEFAULT_SUMMARY = Path(r"D:\HyraiOn\ENS_Lyon\Internship\2026-EPFL\data\0713summary")
JOB_RE = re.compile(r"^(?P<ansatz>.+?)__J2_(?P<j2>[0-9]+p[0-9]+)_(?P<timestamp>\d{8}_\d{6})$")
OBS_RE = re.compile(r"^D_(?P<D>\d+)_chi_(?P<chi>\d+)_energy_magnetization_correlation\.txt$")
ENERGY_RE = re.compile(
    r"^energy_per_site\s*=\s*(?P<energy>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)",
    re.MULTILINE,
)


@dataclass(frozen=True)
class Candidate:
    ansatz: str
    j2: str
    timestamp: str
    D: int
    chi: int
    energy: float
    job_dir: Path
    observation: Path

    @property
    def key(self) -> tuple[str, str, int]:
        return self.j2, self.ansatz, self.D


def read_energy(path: Path) -> float | None:
    try:
        match = ENERGY_RE.search(path.read_text(encoding="utf-8"))
        energy = float(match.group("energy")) if match else None
    except (OSError, UnicodeError, ValueError):
        return None
    return energy if energy is not None and math.isfinite(energy) else None


def discover_best(core_dir: Path) -> tuple[dict[tuple[str, str, int], Candidate], int]:
    best: dict[tuple[str, str, int], Candidate] = {}
    completed = 0
    for job_dir in sorted(core_dir.iterdir()):
        if not job_dir.is_dir():
            continue
        job_match = JOB_RE.match(job_dir.name)
        if not job_match:
            continue
        for observation in sorted(job_dir.iterdir()):
            obs_match = OBS_RE.match(observation.name) if observation.is_file() else None
            if not obs_match:
                continue
            energy = read_energy(observation)
            if energy is None:
                print(f"WARNING: cannot read energy; skipping {observation}")
                continue
            completed += 1
            candidate = Candidate(
                ansatz=job_match.group("ansatz"), j2=job_match.group("j2"),
                timestamp=job_match.group("timestamp"), D=int(obs_match.group("D")),
                chi=int(obs_match.group("chi")), energy=energy,
                job_dir=job_dir, observation=observation,
            )
            incumbent = best.get(candidate.key)
            rank = (candidate.energy, candidate.timestamp, str(candidate.observation))
            if incumbent is None or rank < (incumbent.energy, incumbent.timestamp, str(incumbent.observation)):
                best[candidate.key] = candidate
    return best, completed


def existing_energy(target: Path) -> float | None:
    try:
        value = float(json.loads((target / "manifest.json").read_text(encoding="utf-8"))["energy_per_site"])
    except (OSError, UnicodeError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None
    return value if math.isfinite(value) else None


def copy_candidate(candidate: Candidate, target: Path) -> list[str]:
    """Build one D directory in staging, then replace its old version."""
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}_", dir=target.parent))
    copied: list[str] = []
    try:
        def copy(source: Path, destination_name: str) -> None:
            if source.is_file():
                shutil.copy2(source, staging / destination_name)
                copied.append(destination_name)

        copy(candidate.observation, "energy_magnetization_correlation.txt")
        copy(candidate.job_dir / "hyperparams.yaml", "hyperparams.yaml")
        copy(candidate.job_dir / "run.log", "run.log")
        pattern = f"D_{candidate.D}_chi_{candidate.chi}_lookahead_*_energy_magnetization_correlation.txt"
        for source in sorted(candidate.job_dir.glob(pattern)):
            match = re.search(r"_lookahead_(\d+)_", source.name)
            suffix = match.group(1) if match else "unknown"
            copy(source, f"lookahead_chi_{suffix}_energy_magnetization_correlation.txt")
        copy(candidate.job_dir / f"sweep_D{candidate.D}_chi{candidate.chi}_best.pt", "tensor_best.pt")

        manifest_data = {
            "J2": float(candidate.j2.replace("p", ".")), "J2_label": candidate.j2,
            "ansatz": candidate.ansatz, "D": candidate.D, "chi": candidate.chi,
            "energy_per_site": candidate.energy, "source_job": candidate.job_dir.name,
            "source_observation": candidate.observation.name, "copied_files": sorted(copied),
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest_data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if target.exists():
            shutil.rmtree(target)
        staging.replace(target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return copied


def summarize(core_dir: Path, summary_dir: Path, dry_run: bool = False) -> int:
    if not core_dir.is_dir():
        raise FileNotFoundError(f"core directory does not exist: {core_dir}")
    best, completed = discover_best(core_dir)
    created = updated = kept = missing_artifacts = 0
    sort_key = lambda c: (float(c.j2.replace("p", ".")), c.ansatz, c.D)
    for candidate in sorted(best.values(), key=sort_key):
        target = summary_dir / f"J2_{candidate.j2}" / candidate.ansatz / f"D_{candidate.D}"
        old_energy = existing_energy(target)
        if old_energy is not None and old_energy <= candidate.energy:
            kept += 1
            continue
        action = "CREATE" if old_energy is None else "UPDATE"
        print(f"{action}: J2={candidate.j2.replace('p', '.')} {candidate.ansatz} "
              f"D={candidate.D} chi={candidate.chi} E={candidate.energy:.15g}")
        if not dry_run:
            copied = copy_candidate(candidate, target)
            required = {"energy_magnetization_correlation.txt", "hyperparams.yaml", "run.log", "tensor_best.pt"}
            absent = required.difference(copied)
            if absent:
                missing_artifacts += 1
                print(f"WARNING: {target} missing source artifacts: {', '.join(sorted(absent))}")
        created += old_energy is None
        updated += old_energy is not None
    print(f"Completed observations: {completed}; selected groups: {len(best)}; created: {created}; "
          f"updated: {updated}; kept: {kept}; groups with missing artifacts: {missing_artifacts}")
    if dry_run:
        print("Dry run: no files were written.")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-dir", type=Path, default=DEFAULT_CORE)
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(summarize(args.core_dir, args.summary_dir, args.dry_run))
