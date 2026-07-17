#!/usr/bin/env python3
"""Collect selected completed 0713core results into 0713summary.

A result is complete when its non-lookahead observable file exists and contains
``energy_per_site``. For every (J2, ansatz, D), the lowest-energy job at the
highest chi is preferred. A lower chi may override it only when its energy is
lower and ``abs(E(chi) - E(lookahead)) < 2e-5``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

DEFAULT_CORE = Path(r"D:\HyraiOn\ENS_Lyon\Internship\2026-EPFL\data\0713core")
DEFAULT_SUMMARY = Path(r"D:\HyraiOn\ENS_Lyon\Internship\2026-EPFL\data\0713summary")
JOB_RE = re.compile(r"^(?P<ansatz>.+?)__J2_(?P<j2>[0-9]+p[0-9]+)_(?P<timestamp>\d{8}_\d{6})$")
OBS_RE = re.compile(r"^D_(?P<D>\d+)_chi_(?P<chi>\d+)_energy_magnetization_correlation\.txt$")
ENERGY_RE = re.compile(
    r"^energy_per_site\s*=\s*(?P<energy>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)",
    re.MULTILINE,
)
VARIATIONAL_CHI_THRESHOLD = 2e-5


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
    source_job: str
    lookahead_chi: int | None
    lookahead_energy: float | None
    selection_reason: str = ""

    @property
    def key(self) -> tuple[str, str, int]:
        return self.j2, self.ansatz, self.D

    @property
    def lookahead_delta(self) -> float | None:
        if self.lookahead_energy is None:
            return None
        return abs(self.energy - self.lookahead_energy)

    @property
    def is_variational_chi(self) -> bool:
        delta = self.lookahead_delta
        return delta is not None and delta < VARIATIONAL_CHI_THRESHOLD


def read_energy(path: Path) -> float | None:
    try:
        match = ENERGY_RE.search(path.read_text(encoding="utf-8"))
        energy = float(match.group("energy")) if match else None
    except (OSError, UnicodeError, ValueError):
        return None
    return energy if energy is not None and math.isfinite(energy) else None


def normalize_j2(value: str | float) -> str:
    """Convert 0.30 or 0p30 to the canonical folder label 0p3."""
    number = float(str(value).replace("p", "."))
    decimal = f"{number:.12f}".rstrip("0").rstrip(".")
    if "." not in decimal:
        decimal += ".0"
    return decimal.replace(".", "p")


def read_job_metadata(core_dir: Path, observation: Path) -> tuple[str, str, str, str] | None:
    """Read (ansatz, J2 label, timestamp, source label) for either input layout."""
    job_dir = observation.parent
    standard = JOB_RE.match(job_dir.name)
    if standard:
        return (
            standard.group("ansatz"),
            normalize_j2(standard.group("j2")),
            standard.group("timestamp"),
            job_dir.relative_to(core_dir).as_posix(),
        )

    hyperparams = job_dir / "hyperparams.yaml"
    try:
        text = hyperparams.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None
    try:
        params = json.loads(text)
    except json.JSONDecodeError:
        # A small fallback for genuinely YAML-formatted future external data.
        params = {}
        for key in ("ansatz", "J2", "run_timestamp"):
            match = re.search(rf"^{key}\s*:\s*['\"]?([^'\"#\s]+)", text, re.MULTILINE)
            if match:
                params[key] = match.group(1)
    try:
        ansatz = str(params["ansatz"])
        j2 = normalize_j2(params["J2"])
    except (KeyError, TypeError, ValueError):
        return None
    timestamp = str(params.get("run_timestamp", ""))
    return ansatz, j2, timestamp, job_dir.relative_to(core_dir).as_posix()


def read_lookahead(job_dir: Path, D: int, chi: int) -> tuple[int | None, float | None]:
    """Return the highest matching lookahead chi and its energy."""
    prefix = f"D_{D}_chi_{chi}_lookahead_"
    matches: list[tuple[int, float]] = []
    for path in job_dir.glob(f"{prefix}*_energy_magnetization_correlation.txt"):
        match = re.match(
            rf"^D_{D}_chi_{chi}_lookahead_(\d+)_energy_magnetization_correlation\.txt$",
            path.name,
        )
        energy = read_energy(path)
        if match and energy is not None:
            matches.append((int(match.group(1)), energy))
    return max(matches, key=lambda item: item[0]) if matches else (None, None)


def choose_candidate(candidates: list[Candidate]) -> Candidate:
    """Apply the highest-chi-first selection rule to one (J2, ansatz, D)."""
    best_at_chi: dict[int, Candidate] = {}
    for candidate in candidates:
        incumbent = best_at_chi.get(candidate.chi)
        rank = (candidate.energy, candidate.timestamp, str(candidate.observation))
        if incumbent is None or rank < (incumbent.energy, incumbent.timestamp, str(incumbent.observation)):
            best_at_chi[candidate.chi] = candidate

    highest_chi = max(best_at_chi)
    preferred = best_at_chi[highest_chi]
    lower_overrides = [
        candidate for chi, candidate in best_at_chi.items()
        if chi < highest_chi
        and candidate.energy < preferred.energy
        and candidate.is_variational_chi
    ]
    if not lower_overrides:
        return replace(preferred, selection_reason="highest_chi_lowest_energy")

    chosen = min(
        lower_overrides,
        key=lambda candidate: (candidate.energy, -candidate.chi, candidate.timestamp, str(candidate.observation)),
    )
    return replace(chosen, selection_reason="lower_chi_lower_energy_and_variational")


def discover_best(core_dir: Path) -> tuple[dict[tuple[str, str, int], Candidate], int]:
    grouped: dict[tuple[str, str, int], list[Candidate]] = {}
    completed = 0
    for observation in sorted(core_dir.rglob("*")):
        obs_match = OBS_RE.match(observation.name) if observation.is_file() else None
        if not obs_match:
            continue
        metadata = read_job_metadata(core_dir, observation)
        if metadata is None:
            print(f"WARNING: cannot identify job metadata; skipping {observation}")
            continue
        energy = read_energy(observation)
        if energy is None:
            print(f"WARNING: cannot read energy; skipping {observation}")
            continue
        ansatz, j2, timestamp, source_job = metadata
        D = int(obs_match.group("D"))
        chi = int(obs_match.group("chi"))
        lookahead_chi, lookahead_energy = read_lookahead(observation.parent, D, chi)
        completed += 1
        candidate = Candidate(
            ansatz=ansatz, j2=j2, timestamp=timestamp,
            D=D, chi=chi,
            energy=energy, job_dir=observation.parent, observation=observation,
            source_job=source_job, lookahead_chi=lookahead_chi,
            lookahead_energy=lookahead_energy,
        )
        grouped.setdefault(candidate.key, []).append(candidate)
    best = {key: choose_candidate(candidates) for key, candidates in grouped.items()}
    return best, completed


def existing_manifest(target: Path) -> dict | None:
    try:
        value = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def manifest_matches(manifest: dict | None, candidate: Candidate) -> bool:
    if manifest is None:
        return False
    try:
        return (
            manifest["source_job"] == candidate.source_job
            and manifest["source_observation"] == candidate.observation.name
            and int(manifest["chi"]) == candidate.chi
            and math.isclose(float(manifest["energy_per_site"]), candidate.energy, rel_tol=0.0, abs_tol=1e-15)
        )
    except (KeyError, TypeError, ValueError):
        return False


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
            "energy_per_site": candidate.energy, "source_job": candidate.source_job,
            "source_observation": candidate.observation.name, "copied_files": sorted(copied),
            "lookahead_chi": candidate.lookahead_chi,
            "lookahead_energy_per_site": candidate.lookahead_energy,
            "lookahead_delta": candidate.lookahead_delta,
            "variational_chi_threshold": VARIATIONAL_CHI_THRESHOLD,
            "is_variational_chi": candidate.is_variational_chi,
            "selection_reason": candidate.selection_reason,
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
        old_manifest = existing_manifest(target)
        if manifest_matches(old_manifest, candidate):
            kept += 1
            continue
        action = "CREATE" if old_manifest is None else "UPDATE"
        old_description = ""
        if old_manifest is not None:
            old_description = f" old_chi={old_manifest.get('chi')} old_E={old_manifest.get('energy_per_site')} ->"
        delta = candidate.lookahead_delta
        delta_text = "none" if delta is None else f"{delta:.3g}"
        print(f"{action}: J2={candidate.j2.replace('p', '.')} {candidate.ansatz} D={candidate.D}"
              f"{old_description} chi={candidate.chi} E={candidate.energy:.15g} "
              f"lookahead_delta={delta_text} reason={candidate.selection_reason}")
        if not dry_run:
            copied = copy_candidate(candidate, target)
            required = {"energy_magnetization_correlation.txt", "hyperparams.yaml", "run.log", "tensor_best.pt"}
            absent = required.difference(copied)
            if absent:
                missing_artifacts += 1
                print(f"WARNING: {target} missing source artifacts: {', '.join(sorted(absent))}")
        created += old_manifest is None
        updated += old_manifest is not None
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
