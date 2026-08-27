#!/usr/bin/env python3
"""Integrate new data, routing legacy Neel to D345678910.

Normal usage (after copying every source folder below 0730newdata)::

    python integrate_newdata_all_ansatze.py

``0730olddata`` is the explicit human-reviewed ledger.  A completed result is
ignored when its same-relative-path checkpoint is byte-identical in olddata
and olddata already had a readable matching observable.  The workflow is
deliberately ordered:

1. Scan every ansatz recorded by hyperparams or a standard run-folder name.
2. Route every legacy-Neel result to D345678910 and every other ansatz to
   0713summary. Directly import a non-duplicate completed result.
3. Copy every unresolved, unfinished run's ``*_best.pt`` and ``*.log`` files
   into ``0730newdata/_unfinished_all_ansatze_for_rerun`` with unique names.
4. For completed duplicates, show one (ansatz, J2) at a time as one plot
   column.  Radio buttons on the left enforce exactly one source per
   conflicting D.  Every change redraws energy, m_Neel, and NN correlations.
   Confirm replaces those D entries and advances to the next (ansatz, J2).

A non-lookahead ``D_*_chi_*_energy_magnetization_correlation.txt`` containing a
finite ``energy_per_site`` is the definition of a completed result, matching
``summarize_0713core.py``.  Different chi files in one output directory are one
run, not duplicate runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    from summarize_0713core import (
        Candidate,
        OBS_RE,
        choose_candidate,
        copy_candidate,
        normalize_j2,
        read_energy,
        read_lookahead,
    )
except ImportError:  # Also allow import as a namespace package in tests.
    from .summarize_0713core import (
        Candidate,
        OBS_RE,
        choose_candidate,
        copy_candidate,
        normalize_j2,
        read_energy,
        read_lookahead,
    )


DEFAULT_NEW_ROOT = Path(
    r"D:\HyraiOn\ENS_Lyon\Internship\2026-EPFL\data\0713core\0730newdata"
)
DEFAULT_OLD_ROOT = Path(
    r"D:\HyraiOn\ENS_Lyon\Internship\2026-EPFL\data\0713core\0730olddata"
)
DEFAULT_SUMMARY_ROOT = Path(
    r"D:\HyraiOn\ENS_Lyon\Internship\2026-EPFL\data\0713summary"
)
DEFAULT_LEGACY_NEEL_ROOT = Path(
    r"D:\HyraiOn\ENS_Lyon\Internship\2026-EPFL\data\D345678910"
)
LEGACY_NEEL_ROOT = DEFAULT_LEGACY_NEEL_ROOT
DEFAULT_RERUN_FOLDER = "_unfinished_all_ansatze_for_rerun"
LEGACY_RERUN_FOLDER = "_unfinished_2C3_for_rerun"
DEFAULT_REPORT = "newdata_all_ansatze_integration_report.json"

# yaml_name values from both models/0801core main registries.  Hyperparams and
# standard run folders remain authoritative, so a future unknown yaml_name is
# accepted without editing this list.  The table is used only for CLI/log/path
# fallbacks when downloaded external data lacks hyperparams.
KNOWN_YAML_ANSATZE = (
    "6tensors",
    "neel_free_param",
    "neel_symmetrized",
    "1tensor_C6Ypi",
    "1tensor_C6_swave",
    "sym6_free_param",
    "sym2_free_param",
    "sym6",
    "1tensor_C3Vypi",
    "2tensor_twoC3",
    "2tensor_columnar",
)
CLI_ANSATZ_ALIASES = {
    "unrestricted": "6tensors",
    "neel": "neel_free_param",
    "neel_legacy": "neel_symmetrized",
    "c6ypi": "1tensor_C6Ypi",
    "swave": "1tensor_C6_swave",
    "sym6": "sym6_free_param",
    "sym2": "sym2_free_param",
    "sym6_legacy": "sym6",
    "c3vypi": "1tensor_C3Vypi",
    "twoc3": "2tensor_twoC3",
    "columnar": "2tensor_columnar",
}

STANDARD_RUN_RE = re.compile(
    r"^(?P<ansatz>.+?)__J2_(?P<j2>[0-9]+p[0-9]+)"
    r"_(?P<timestamp>\d{8}_\d{6})$",
    re.IGNORECASE,
)
BEST_RE = re.compile(
    r"^(?:sweep_)?D_?(?P<D>\d+)_chi_?(?P<chi>\d+)_best\.pt$",
    re.IGNORECASE,
)
SUMMARY_J2_RE = re.compile(r"^J2_(?P<j2>[0-9]+p[0-9]+)$")
SUMMARY_D_RE = re.compile(r"^D_(?P<D>\d+)$")
LEGACY_NEEL_RUN_RE = re.compile(
    r"^(?:neel_symmetrized|neel_legacy)__J2_(?P<j2>[0-9]+p[0-9]+)(?:_|$)",
    re.IGNORECASE,
)
PATH_J2_PATTERNS = (
    re.compile(r"J2[^0-9+-]*([+-]?(?:\d+(?:[p.]\d*)?|[p.]\d+))", re.IGNORECASE),
    re.compile(r"J2[_-]([+-]?(?:\d+(?:[p.]\d*)?|[p.]\d+))", re.IGNORECASE),
)
LOG_J2_RE = re.compile(
    r"\bJ1\s*=\s*[+-]?[\d.eE+-]+\s+J2\s*=\s*"
    r"(?P<j2>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
)
STARTED_RE = re.compile(
    r"Started\s*:\s*(?P<date>\d{4}-\d{2}-\d{2})\s+"
    r"(?P<time>\d{2}:\d{2}:\d{2})",
    re.IGNORECASE,
)
TOTAL_WALL_RE = re.compile(
    r"Total wall time:\s*(?P<hours>[0-9]+(?:\.[0-9]+)?)\s*h",
    re.IGNORECASE,
)
COMPLETE_WALL_RE_TEMPLATE = (
    r"\bD\s*=\s*{D}\s+complete in\s*"
    r"(?P<hours>[0-9]+(?:\.[0-9]+)?)\s*h"
)
WALL_RE = re.compile(
    r"\bwall\s*=\s*(?P<hours>[0-9]+(?:\.[0-9]+)?)\s*h",
    re.IGNORECASE,
)
SIMPLE_YAML_VALUE_RE = re.compile(
    r"^(?P<key>ansatz|J2|run_timestamp)\s*:\s*"
    r"(?P<value>[^#\r\n]+?)\s*$",
    re.MULTILINE,
)


@dataclass(frozen=True)
class RunMetadata:
    ansatz: str
    j2: str
    timestamp: str


@dataclass(frozen=True)
class Choice:
    """One selectable completed run for one (ansatz, J2, D)."""

    candidate: Candidate
    origin: str  # "new" or "summary"
    source_label: str
    run_id: str
    elapsed_hours: float | None

    @property
    def key(self) -> tuple[str, str, int]:
        return self.candidate.ansatz, self.candidate.j2, self.candidate.D

    @property
    def display_label(self) -> str:
        source = _shorten(self.source_label, 34)
        timestamp = self.candidate.timestamp or "time unknown"
        wall = (
            f"{self.elapsed_hours:.2f} h"
            if self.elapsed_hours is not None
            else "wall unknown"
        )
        return (
            f"{source} | {timestamp} | {wall} | "
            f"E={self.candidate.energy:.12g}"
        )


@dataclass
class ScanResult:
    completed: dict[tuple[str, str, int], list[Choice]]
    incomplete: dict[tuple[str, str, int], list["IncompleteRun"]]
    reviewed_skipped: list[dict]
    warnings: list[str]


@dataclass(frozen=True)
class IncompleteRun:
    ansatz: str
    j2: str
    D: int
    timestamp: str
    source_label: str
    run_id: str
    job_dir: Path
    checkpoints: tuple[Path, ...]
    logs: tuple[Path, ...]

    @property
    def key(self) -> tuple[str, str, int]:
        return self.ansatz, self.j2, self.D


@dataclass
class Preparation:
    initial_summary: dict[tuple[str, str, int], Choice]
    new_scan: ScanResult
    unique: dict[tuple[str, str, int], Choice]
    conflicts: dict[tuple[str, str, int], list[Choice]]
    archived: list[dict]


def _shorten(text: str, width: int) -> str:
    if len(text) <= width:
        return text
    keep = max(4, (width - 1) // 2)
    return f"{text[:keep]}…{text[-keep:]}"


def _key_sort(key: tuple[str, str, int]) -> tuple[str, float, int]:
    ansatz, j2, D = key
    return ansatz.casefold(), float(j2.replace("p", ".")), D


def _canonical_ansatz(value: str, *, cli_context: bool = False) -> str:
    """Canonicalize a registry CLI key or yaml_name without merging ansatze."""
    cleaned = value.strip().strip("'\"")
    if not cleaned:
        return ""
    yaml_by_case = {name.casefold(): name for name in KNOWN_YAML_ANSATZE}
    if not cli_context and cleaned.casefold() in yaml_by_case:
        return yaml_by_case[cleaned.casefold()]
    alias = CLI_ANSATZ_ALIASES.get(cleaned.casefold())
    if alias is not None:
        return alias
    if cleaned.casefold() in yaml_by_case:
        return yaml_by_case[cleaned.casefold()]
    return cleaned


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _load_hyperparams(path: Path) -> dict:
    text = _read_text(path)
    if not text:
        return {}
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        result: dict[str, str] = {}
        for match in SIMPLE_YAML_VALUE_RE.finditer(text):
            result[match.group("key")] = match.group("value").strip(" '\"")
        return result


def _direct_logs(job_dir: Path) -> tuple[Path, ...]:
    return tuple(
        sorted(
            path
            for path in job_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".log"
        )
    )


def _metadata_from_job(job_dir: Path, root: Path) -> RunMetadata | None:
    params = _load_hyperparams(job_dir / "hyperparams.yaml")
    ansatz = _canonical_ansatz(str(params.get("ansatz", "")))
    j2_value = params.get("J2")
    timestamp = str(params.get("run_timestamp", "")).strip()

    for ancestor in (job_dir, *job_dir.parents):
        if ancestor == root.parent:
            break
        match = STANDARD_RUN_RE.match(ancestor.name)
        if match:
            ansatz = ansatz or _canonical_ansatz(match.group("ansatz"))
            j2_value = j2_value if j2_value is not None else match.group("j2")
            timestamp = timestamp or match.group("timestamp")
            break
        if ancestor == root:
            break

    path_text = job_dir.relative_to(root).as_posix()
    if j2_value is None:
        for pattern in PATH_J2_PATTERNS:
            match = pattern.search(path_text)
            if match:
                j2_value = match.group(1)
                break
    if not ansatz:
        for yaml_name in sorted(KNOWN_YAML_ANSATZE, key=len, reverse=True):
            if yaml_name.casefold() in path_text.casefold():
                ansatz = yaml_name
                break
    if not ansatz:
        for cli_name, yaml_name in CLI_ANSATZ_ALIASES.items():
            if re.search(rf"(?:^|[^a-z0-9]){re.escape(cli_name)}(?:[^a-z0-9]|$)", path_text, re.IGNORECASE):
                ansatz = yaml_name
                break

    log_texts = [_read_text(path) for path in _direct_logs(job_dir)]
    joined_logs = "\n".join(log_texts)
    if j2_value is None:
        match = LOG_J2_RE.search(joined_logs)
        if match:
            j2_value = match.group("j2")
    if not ansatz:
        match = re.search(r"Ansatz\s*:\s*['\"]?([^'\"\s]+)", joined_logs, re.IGNORECASE)
        if match:
            ansatz = _canonical_ansatz(match.group(1), cli_context=True)
    if not timestamp:
        match = STARTED_RE.search(joined_logs)
        if match:
            timestamp = (
                match.group("date").replace("-", "")
                + "_"
                + match.group("time").replace(":", "")
            )

    if not ansatz or j2_value is None:
        return None
    try:
        j2 = normalize_j2(str(j2_value))
    except ValueError:
        return None
    return RunMetadata(ansatz=ansatz, j2=j2, timestamp=timestamp)


def _job_mentions_ansatz(job_dir: Path, root: Path) -> bool:
    """Return whether an unidentifiable artifact directory appears to be a run."""
    params = _load_hyperparams(job_dir / "hyperparams.yaml")
    if params.get("ansatz") is not None:
        return bool(str(params["ansatz"]).strip())
    relative = job_dir.relative_to(root).as_posix()
    if any(name.casefold() in relative.casefold() for name in KNOWN_YAML_ANSATZE):
        return True
    return any(
        re.search(r"Ansatz\s*:", _read_text(path), re.IGNORECASE)
        for path in _direct_logs(job_dir)
    )


def _source_label(root: Path, job_dir: Path) -> str:
    relative = job_dir.relative_to(root)
    return relative.parts[0] if relative.parts else root.name


def _elapsed_hours(logs: Iterable[Path], D: int) -> float | None:
    texts = [_read_text(path) for path in logs]
    d_complete = re.compile(
        COMPLETE_WALL_RE_TEMPLATE.format(D=D), re.IGNORECASE
    )
    values = [
        float(match.group("hours"))
        for text in texts
        for match in d_complete.finditer(text)
    ]
    if values:
        return max(values)
    values = [
        float(match.group("hours"))
        for text in texts
        for match in TOTAL_WALL_RE.finditer(text)
    ]
    if values:
        return max(values)
    values = [
        float(match.group("hours"))
        for text in texts
        for match in WALL_RE.finditer(text)
    ]
    return max(values) if values else None


def _started_timestamp(logs: Iterable[Path]) -> str:
    for path in logs:
        match = STARTED_RE.search(_read_text(path))
        if match:
            return (
                match.group("date").replace("-", "")
                + "_"
                + match.group("time").replace(":", "")
            )
    return ""


def _run_source_id(root: Path, job_dir: Path) -> str:
    relative = job_dir.relative_to(root).as_posix()
    return f"{root.name}/{relative}" if relative != "." else root.name


def _sha256(path: Path, cache: dict[Path, str]) -> str:
    cached = cache.get(path)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    value = digest.hexdigest()
    cache[path] = value
    return value


def _files_identical(
    new_path: Path,
    old_path: Path,
    hash_cache: dict[Path, str],
) -> bool:
    """Exact comparison, with size as a cheap rejection before SHA-256."""
    try:
        if not old_path.is_file() or new_path.stat().st_size != old_path.stat().st_size:
            return False
        return _sha256(new_path, hash_cache) == _sha256(old_path, hash_cache)
    except OSError:
        return False


def _old_counterpart(new_path: Path, new_root: Path, old_root: Path) -> Path:
    return old_root / new_path.relative_to(new_root)


def _completed_was_reviewed(
    checkpoint: Path,
    observation: Path,
    new_root: Path,
    old_root: Path | None,
    hash_cache: dict[Path, str],
) -> bool:
    if old_root is None:
        return False
    old_observation = _old_counterpart(observation, new_root, old_root)
    return (
        _files_identical(
            checkpoint,
            _old_counterpart(checkpoint, new_root, old_root),
            hash_cache,
        )
        and read_energy(old_observation) is not None
    )


def _checkpoint_was_seen(
    checkpoint: Path,
    new_root: Path,
    old_root: Path | None,
    hash_cache: dict[Path, str],
) -> bool:
    if old_root is None:
        return False
    return _files_identical(
        checkpoint,
        _old_counterpart(checkpoint, new_root, old_root),
        hash_cache,
    )


def discover_new_runs(
    root: Path,
    old_root: Path | None = None,
    excluded_names: set[str] | None = None,
) -> ScanResult:
    """Discover completed and incomplete runs for every ansatz.

    Each directory directly containing an observable or checkpoint is one run.
    Within that directory, multiple completed chi values for the same D are
    reduced with the exact rule used by summarize_0713core.
    """
    if not root.is_dir():
        raise FileNotFoundError(f"new-data directory does not exist: {root}")
    if old_root is not None and not old_root.is_dir():
        raise FileNotFoundError(f"reviewed old-data directory does not exist: {old_root}")
    excluded_names = excluded_names or {DEFAULT_RERUN_FOLDER, LEGACY_RERUN_FOLDER}
    job_dirs: set[Path] = set()
    for current, dirs, files in os.walk(root):
        dirs[:] = [name for name in dirs if name not in excluded_names]
        if any(OBS_RE.match(name) or BEST_RE.match(name) for name in files):
            job_dirs.add(Path(current))

    completed: dict[tuple[str, str, int], list[Choice]] = {}
    incomplete: dict[tuple[str, str, int], list[IncompleteRun]] = {}
    reviewed_skipped: list[dict] = []
    warnings: list[str] = []
    hash_cache: dict[Path, str] = {}
    for job_dir in sorted(job_dirs):
        metadata = _metadata_from_job(job_dir, root)
        if metadata is None:
            if _job_mentions_ansatz(job_dir, root):
                warnings.append(f"cannot identify ansatz/J2 metadata: {job_dir}")
            continue
        logs = _direct_logs(job_dir)
        by_D_observations: dict[int, list[Candidate]] = {}
        checkpoints_by_D: dict[int, list[Path]] = {}
        checkpoints_by_key: dict[tuple[int, int], list[Path]] = {}

        for path in sorted(job_dir.iterdir()):
            if not path.is_file():
                continue
            best_match = BEST_RE.match(path.name)
            if best_match:
                D = int(best_match.group("D"))
                chi = int(best_match.group("chi"))
                checkpoints_by_D.setdefault(D, []).append(path)
                checkpoints_by_key.setdefault((D, chi), []).append(path)
                continue
            obs_match = OBS_RE.match(path.name)
            if not obs_match:
                continue
            energy = read_energy(path)
            if energy is None:
                warnings.append(f"observable has no finite energy: {path}")
                continue
            D = int(obs_match.group("D"))
            chi = int(obs_match.group("chi"))
            lookahead_chi, lookahead_energy = read_lookahead(job_dir, D, chi)
            candidate = Candidate(
                ansatz=metadata.ansatz,
                j2=metadata.j2,
                timestamp=metadata.timestamp,
                D=D,
                chi=chi,
                energy=energy,
                job_dir=job_dir,
                observation=path,
                source_job=_run_source_id(root, job_dir),
                lookahead_chi=lookahead_chi,
                lookahead_energy=lookahead_energy,
            )
            by_D_observations.setdefault(D, []).append(candidate)

        source_label = _source_label(root, job_dir)
        run_id = job_dir.relative_to(root).as_posix()
        for D, observations in by_D_observations.items():
            new_observations: list[Candidate] = []
            for candidate in observations:
                matching = checkpoints_by_key.get((D, candidate.chi), [])
                if not matching:
                    warnings.append(
                        "completed observable has no matching *_best.pt; skipping: "
                        f"{candidate.observation}"
                    )
                    continue
                canonical_name = f"sweep_D{D}_chi{candidate.chi}_best.pt"
                checkpoint = next(
                    (path for path in matching if path.name == canonical_name),
                    matching[0],
                )
                if _completed_was_reviewed(
                    checkpoint,
                    candidate.observation,
                    root,
                    old_root,
                    hash_cache,
                ):
                    reviewed_skipped.append(
                        {
                            "kind": "completed",
                            "ansatz": metadata.ansatz,
                            "J2_label": metadata.j2,
                            "D": D,
                            "chi": candidate.chi,
                            "checkpoint": checkpoint.relative_to(root).as_posix(),
                            "observation": candidate.observation.relative_to(root).as_posix(),
                        }
                    )
                    continue
                new_observations.append(candidate)
            if not new_observations:
                continue
            chosen = choose_candidate(new_observations)
            choice = Choice(
                candidate=chosen,
                origin="new",
                source_label=source_label,
                run_id=run_id,
                elapsed_hours=_elapsed_hours(logs, D),
            )
            completed.setdefault(choice.key, []).append(choice)

        # A run with any valid non-lookahead result for D is completed for that
        # pair.  A later checkpoint without its own observable does not turn the
        # whole run back into "no result".
        for D, checkpoints in checkpoints_by_D.items():
            if D in by_D_observations:
                continue
            new_checkpoints: list[Path] = []
            for checkpoint in checkpoints:
                if _checkpoint_was_seen(checkpoint, root, old_root, hash_cache):
                    match = BEST_RE.match(checkpoint.name)
                    reviewed_skipped.append(
                        {
                            "kind": "incomplete_checkpoint",
                            "ansatz": metadata.ansatz,
                            "J2_label": metadata.j2,
                            "D": D,
                            "chi": int(match.group("chi")) if match else None,
                            "checkpoint": checkpoint.relative_to(root).as_posix(),
                        }
                    )
                else:
                    new_checkpoints.append(checkpoint)
            if not new_checkpoints:
                continue
            item = IncompleteRun(
                ansatz=metadata.ansatz,
                j2=metadata.j2,
                D=D,
                timestamp=metadata.timestamp,
                source_label=source_label,
                run_id=run_id,
                job_dir=job_dir,
                checkpoints=tuple(new_checkpoints),
                logs=logs,
            )
            incomplete.setdefault(item.key, []).append(item)

    for choices in completed.values():
        choices.sort(
            key=lambda choice: (
                choice.source_label,
                choice.candidate.timestamp,
                choice.run_id,
            )
        )
    for runs in incomplete.values():
        runs.sort(key=lambda run: (run.source_label, run.timestamp, run.run_id))
    return ScanResult(
        completed=completed,
        incomplete=incomplete,
        reviewed_skipped=reviewed_skipped,
        warnings=warnings,
    )


def _load_manifest(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}


def discover_summary(summary_root: Path) -> tuple[dict[tuple[str, str, int], Choice], list[str]]:
    result: dict[tuple[str, str, int], Choice] = {}
    warnings: list[str] = []
    if not summary_root.exists():
        return result, warnings
    for j2_dir in sorted(summary_root.glob("J2_*")):
        j2_match = SUMMARY_J2_RE.match(j2_dir.name)
        if not j2_match:
            continue
        try:
            j2 = normalize_j2(j2_match.group("j2"))
        except ValueError:
            continue
        for ansatz_dir in sorted(path for path in j2_dir.iterdir() if path.is_dir()):
            ansatz = ansatz_dir.name
            for d_dir in sorted(ansatz_dir.glob("D_*")):
                d_match = SUMMARY_D_RE.match(d_dir.name)
                if not d_match or not d_dir.is_dir():
                    continue
                D = int(d_match.group("D"))
                observation = d_dir / "energy_magnetization_correlation.txt"
                energy = read_energy(observation)
                if energy is None:
                    warnings.append(f"summary observable has no finite energy: {observation}")
                    continue
                manifest = _load_manifest(d_dir / "manifest.json")
                try:
                    chi = int(manifest.get("chi", 0))
                except (TypeError, ValueError):
                    chi = 0
                source_job = str(
                    manifest.get(
                        "source_job",
                        d_dir.relative_to(summary_root).as_posix(),
                    )
                )
                timestamp = ""
                for component in Path(source_job).parts:
                    match = STANDARD_RUN_RE.match(component)
                    if match:
                        timestamp = match.group("timestamp")
                logs = tuple(d_dir.glob("*.log"))
                if not timestamp:
                    timestamp = (
                        str(manifest.get("run_timestamp", ""))
                        or _started_timestamp(logs)
                    )
                candidate = Candidate(
                    ansatz=ansatz,
                    j2=j2,
                    timestamp=timestamp,
                    D=D,
                    chi=chi,
                    energy=energy,
                    job_dir=d_dir,
                    observation=observation,
                    source_job=source_job,
                    lookahead_chi=manifest.get("lookahead_chi"),
                    lookahead_energy=manifest.get("lookahead_energy_per_site"),
                    selection_reason="existing_0713summary",
                )
                result[(ansatz, j2, D)] = Choice(
                    candidate=candidate,
                    origin="summary",
                    source_label="0713summary",
                    run_id=source_job,
                    elapsed_hours=_elapsed_hours(logs, D),
                )
    return result, warnings


def discover_legacy_neel(
    legacy_root: Path,
) -> tuple[dict[tuple[str, str, int], Choice], list[str]]:
    """Discover the current per-(J2,D) legacy-Neel choices in D345678910."""
    grouped: dict[tuple[str, str, int], list[Candidate]] = {}
    warnings: list[str] = []
    if not legacy_root.exists():
        return {}, warnings
    for run_dir in sorted(path for path in legacy_root.iterdir() if path.is_dir()):
        run_match = LEGACY_NEEL_RUN_RE.match(run_dir.name)
        if run_match is None:
            continue
        try:
            j2 = normalize_j2(run_match.group("j2"))
        except ValueError:
            continue
        logs = _direct_logs(run_dir)
        timestamp = _started_timestamp(logs)
        for observation in sorted(run_dir.iterdir()):
            obs_match = OBS_RE.match(observation.name)
            if obs_match is None:
                continue
            energy = read_energy(observation)
            if energy is None:
                warnings.append(
                    f"legacy observable has no finite energy: {observation}"
                )
                continue
            D, chi = int(obs_match.group("D")), int(obs_match.group("chi"))
            lookahead_chi, lookahead_energy = read_lookahead(run_dir, D, chi)
            candidate = Candidate(
                ansatz="neel_symmetrized",
                j2=j2,
                timestamp=timestamp,
                D=D,
                chi=chi,
                energy=energy,
                job_dir=run_dir,
                observation=observation,
                source_job=run_dir.relative_to(legacy_root).as_posix(),
                lookahead_chi=lookahead_chi,
                lookahead_energy=lookahead_energy,
                selection_reason="existing_D345678910",
            )
            grouped.setdefault((candidate.ansatz, j2, D), []).append(candidate)

    result = {}
    for key, candidates in grouped.items():
        candidate = choose_candidate(candidates)
        result[key] = Choice(
            candidate=candidate,
            origin="summary",
            source_label="D345678910",
            run_id=candidate.source_job,
            elapsed_hours=_elapsed_hours(_direct_logs(candidate.job_dir), candidate.D),
        )
    return result, warnings


def classify(
    new_scan: ScanResult,
    summary: dict[tuple[str, str, int], Choice],
) -> tuple[
    dict[tuple[str, str, int], Choice],
    dict[tuple[str, str, int], list[Choice]],
]:
    unique: dict[tuple[str, str, int], Choice] = {}
    conflicts: dict[tuple[str, str, int], list[Choice]] = {}
    for key, new_choices in new_scan.completed.items():
        old = summary.get(key)
        merged = ([old] if old is not None else []) + list(new_choices)
        if old is None and len(new_choices) == 1:
            unique[key] = new_choices[0]
        elif len(merged) >= 2:
            conflicts[key] = merged
    return unique, conflicts


def _required_copy_warning(choice: Choice, copied: list[str]) -> str | None:
    required = {
        "energy_magnetization_correlation.txt",
        "hyperparams.yaml",
        "run.log",
        "tensor_best.pt",
    }
    missing = sorted(required.difference(copied))
    if not missing:
        return None
    return (
        f"ansatz={choice.candidate.ansatz} "
        f"J2={choice.candidate.j2.replace('p', '.')} D={choice.candidate.D} "
        f"source={choice.run_id} missing copied artifacts: {', '.join(missing)}"
    )


def _is_legacy_neel(candidate: Candidate) -> bool:
    return candidate.ansatz == "neel_symmetrized"


def _legacy_neel_run_directories(root: Path, j2: str) -> list[Path]:
    if not root.is_dir():
        return []
    matches = []
    for path in root.iterdir():
        match = LEGACY_NEEL_RUN_RE.match(path.name)
        if not path.is_dir() or match is None:
            continue
        try:
            if normalize_j2(match.group("j2")) == j2:
                matches.append(path)
        except ValueError:
            continue
    return sorted(matches)


def _legacy_neel_target(root: Path, j2: str, D: int) -> Path:
    if not root.is_dir():
        return root / f"neel_symmetrized__J2_{j2}_integrated_newdata"
    current, _warnings = discover_legacy_neel(root)
    existing = current.get(("neel_symmetrized", j2, D))
    if existing is not None:
        return existing.candidate.job_dir
    candidates = _legacy_neel_run_directories(root, j2)
    if candidates:
        return candidates[-1]
    return root / f"neel_symmetrized__J2_{j2}_integrated_newdata"


def _checkpoint_hashes(paths: Iterable[Path]) -> set[str]:
    cache: dict[Path, str] = {}
    return {_sha256(path, cache) for path in paths if path.is_file()}


def _correlation_matches_checkpoints(path: Path, checkpoint_hashes: set[str]) -> bool:
    if not path.is_file() or not checkpoint_hashes:
        return False
    payload = _load_manifest(path)
    recorded = payload.get("cluster_bundle_provenance", {}).get("checkpoint_sha256")
    return isinstance(recorded, str) and recorded in checkpoint_hashes


def import_legacy_neel_choice(
    choice: Choice,
    legacy_root: Path,
    *,
    dry_run: bool = False,
) -> list[str]:
    candidate = choice.candidate
    target = _legacy_neel_target(legacy_root, candidate.j2, candidate.D)
    D, chi = candidate.D, candidate.chi
    if dry_run:
        print(
            f"WOULD IMPORT LEGACY NEEL: J2={candidate.j2.replace('p', '.')} "
            f"D={D} chi={chi} "
            f"from {choice.run_id} -> {target}"
        )
        return []
    target.mkdir(parents=True, exist_ok=True)
    checkpoint_patterns = (
        f"sweep_D{D}_chi{chi}_best*.pt",
        f"sweep_D_{D}_chi_{chi}_best*.pt",
    )
    incoming_checkpoints = [
        path
        for pattern in checkpoint_patterns
        for path in sorted(candidate.job_dir.glob(pattern))
    ]
    incoming_hashes = _checkpoint_hashes(incoming_checkpoints)
    correlation = target / f"correlation_length_D_{D}.json"
    keep_correlation = _correlation_matches_checkpoints(
        correlation, incoming_hashes
    )
    replace_patterns = (
        f"D_{D}_chi_*_energy_magnetization_correlation.txt",
        f"sweep_D{D}_chi*_best*.pt",
        f"sweep_D_{D}_chi*_best*.pt",
    )
    cleanup_directories = set(
        _legacy_neel_run_directories(legacy_root, candidate.j2)
    ) | {target}
    for directory in cleanup_directories:
        for pattern in replace_patterns:
            for old in directory.glob(pattern):
                if old.is_file():
                    old.unlink()
        old_correlation = directory / f"correlation_length_D_{D}.json"
        preserve = directory == target and keep_correlation
        if old_correlation.is_file() and not preserve:
            old_correlation.unlink()
            print(f"REMOVED STALE CORRELATION LENGTH: {old_correlation}")

    sources: list[tuple[Path, str]] = [(candidate.observation, candidate.observation.name)]
    sources.extend(
        (path, path.name)
        for path in sorted(candidate.job_dir.glob(
            f"D_{D}_chi_{chi}_lookahead_*_energy_magnetization_correlation.txt"
        ))
    )
    sources.extend((path, path.name) for path in incoming_checkpoints)

    run_digest = hashlib.sha1(choice.run_id.encode("utf-8")).hexdigest()[:8]
    audit_prefix = (
        f"rerun_J2_{candidate.j2}_D{D}_"
        f"{candidate.timestamp or 'time_unknown'}_{run_digest}"
    )
    for log in _direct_logs(candidate.job_dir):
        sources.append((log, f"{audit_prefix}_{log.name}"))
    for name in ("hyperparams.yaml", "sweep_results.json"):
        source = candidate.job_dir / name
        if source.is_file():
            sources.append((source, f"{audit_prefix}_{name}"))

    copied: list[str] = []
    for source, destination_name in sources:
        destination = target / destination_name
        temporary = destination.with_name(destination.name + ".copying")
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
        copied.append(destination.name)
    print(
        f"IMPORTED LEGACY NEEL: J2={candidate.j2.replace('p', '.')} "
        f"D={D} chi={chi} "
        f"E={candidate.energy:.15g} -> {target}"
    )
    return copied


def import_choice(choice: Choice, summary_root: Path, dry_run: bool = False) -> list[str]:
    candidate = choice.candidate
    if _is_legacy_neel(candidate):
        return import_legacy_neel_choice(
            choice, LEGACY_NEEL_ROOT, dry_run=dry_run
        )
    target = (
        summary_root
        / f"J2_{candidate.j2}"
        / candidate.ansatz
        / f"D_{candidate.D}"
    )
    if dry_run:
        print(
            f"WOULD IMPORT: ansatz={candidate.ansatz} "
            f"J2={candidate.j2.replace('p', '.')} D={candidate.D} "
            f"chi={candidate.chi} E={candidate.energy:.15g} from {choice.run_id}"
        )
        return []
    copied = copy_candidate(candidate, target)
    print(
        f"IMPORTED: ansatz={candidate.ansatz} "
        f"J2={candidate.j2.replace('p', '.')} D={candidate.D} "
        f"chi={candidate.chi} E={candidate.energy:.15g} from {choice.run_id}"
    )
    warning = _required_copy_warning(choice, copied)
    if warning:
        print(f"WARNING: {warning}")
    return copied


def import_choices_atomically(
    choices: Iterable[Choice],
    summary_root: Path,
) -> None:
    """Commit all newly selected D entries for one (ansatz, J2), or none."""
    selected = [choice for choice in choices if choice.origin == "new"]
    if not selected:
        return
    if all(_is_legacy_neel(choice.candidate) for choice in selected):
        for choice in selected:
            import_legacy_neel_choice(choice, LEGACY_NEEL_ROOT)
        return
    if any(_is_legacy_neel(choice.candidate) for choice in selected):
        raise ValueError("Legacy-Neel choices cannot be mixed with summary imports")
    summary_root.parent.mkdir(parents=True, exist_ok=True)
    stage_root = Path(tempfile.mkdtemp(prefix=".newdata_stage_", dir=summary_root.parent))
    backup_root = Path(tempfile.mkdtemp(prefix=".newdata_backup_", dir=summary_root.parent))
    staged: list[tuple[Choice, Path, Path, list[str]]] = []
    committed: list[tuple[Path, Path | None]] = []
    try:
        for choice in selected:
            candidate = choice.candidate
            relative = (
                Path(f"J2_{candidate.j2}")
                / candidate.ansatz
                / f"D_{candidate.D}"
            )
            stage_target = stage_root / relative
            copied = copy_candidate(candidate, stage_target)
            target = summary_root / relative
            staged.append((choice, stage_target, target, copied))

        for choice, stage_target, target, _copied in staged:
            target.parent.mkdir(parents=True, exist_ok=True)
            backup = backup_root / target.relative_to(summary_root)
            old_backup: Path | None = None
            if target.exists():
                backup.parent.mkdir(parents=True, exist_ok=True)
                target.replace(backup)
                old_backup = backup
            try:
                stage_target.replace(target)
            except Exception:
                if old_backup is not None and old_backup.exists():
                    old_backup.replace(target)
                raise
            committed.append((target, old_backup))

        for choice, _stage_target, _target, copied in staged:
            candidate = choice.candidate
            print(
                f"IMPORTED: ansatz={candidate.ansatz} "
                f"J2={candidate.j2.replace('p', '.')} D={candidate.D} "
                f"chi={candidate.chi} E={candidate.energy:.15g} "
                f"from {choice.run_id}"
            )
            warning = _required_copy_warning(choice, copied)
            if warning:
                print(f"WARNING: {warning}")
    except Exception:
        for target, old_backup in reversed(committed):
            if target.exists():
                shutil.rmtree(target)
            if old_backup is not None and old_backup.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                old_backup.replace(target)
        raise
    finally:
        shutil.rmtree(stage_root, ignore_errors=True)
        shutil.rmtree(backup_root, ignore_errors=True)


def _slug(value: str, max_length: int = 44) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")
    return (cleaned or "source")[:max_length]


def archive_incomplete_runs(
    incomplete: dict[tuple[str, str, int], list[IncompleteRun]],
    initial_summary: dict[tuple[str, str, int], Choice],
    rerun_dir: Path,
    dry_run: bool = False,
) -> list[dict]:
    """Copy every unresolved incomplete run, including duplicate attempts."""
    records: list[dict] = []
    for key in sorted(incomplete, key=_key_sort):
        if key in initial_summary:
            continue
        for run in incomplete[key]:
            digest = hashlib.sha1(run.run_id.encode("utf-8")).hexdigest()[:8]
            prefix = (
                f"{_slug(run.ansatz)}__J2_{run.j2}_D{run.D}__"
                f"{_slug(run.source_label)}__"
                f"{_slug(run.timestamp or 'time-unknown')}__{digest}"
            )
            sources = list(run.checkpoints) + list(run.logs)
            if not sources:
                continue
            copied: list[dict[str, str]] = []
            for index, source in enumerate(sources, start=1):
                original = _slug(source.name, 70)
                destination_name = f"{prefix}__{index:02d}__{original}"
                destination = rerun_dir / destination_name
                print(
                    f"{'WOULD ARCHIVE' if dry_run else 'ARCHIVED'}: "
                    f"{run.run_id}/{source.name} -> {destination_name}"
                )
                if not dry_run:
                    rerun_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
                copied.append(
                    {
                        "source": str(source),
                        "destination": str(destination),
                    }
                )
            records.append(
                {
                    "ansatz": run.ansatz,
                    "J2_label": run.j2,
                    "D": run.D,
                    "source_label": run.source_label,
                    "run_id": run.run_id,
                    "timestamp": run.timestamp,
                    "files": copied,
                }
            )
    return records


def prepare(
    new_root: Path,
    summary_root: Path,
    rerun_dir: Path,
    old_root: Path | None = None,
    dry_run: bool = False,
) -> Preparation:
    initial_summary, summary_warnings = discover_summary(summary_root)
    initial_summary = {
        key: choice
        for key, choice in initial_summary.items()
        if not _is_legacy_neel(choice.candidate)
    }
    legacy_choices, legacy_warnings = discover_legacy_neel(LEGACY_NEEL_ROOT)
    initial_summary.update(legacy_choices)
    new_scan = discover_new_runs(
        new_root,
        old_root=old_root,
        excluded_names={rerun_dir.name, DEFAULT_RERUN_FOLDER, LEGACY_RERUN_FOLDER},
    )
    unique, conflicts = classify(new_scan, initial_summary)

    print(
        f"Found {sum(len(v) for v in new_scan.completed.values())} completed "
        f"new runs across {len(new_scan.completed)} (ansatz,J2,D) groups; "
        f"{sum(len(v) for v in new_scan.incomplete.values())} unfinished runs; "
        f"{len(new_scan.reviewed_skipped)} olddata artifacts skipped."
    )
    print(
        f"Unique imports: {len(unique)}; conflicting pairs: {len(conflicts)}; "
        f"existing summary pairs: {len(initial_summary)}."
    )
    for warning in summary_warnings + legacy_warnings + new_scan.warnings:
        print(f"WARNING: {warning}")
    for record in new_scan.reviewed_skipped:
        print(
            "SKIP REVIEWED: "
            f"kind={record['kind']} ansatz={record['ansatz']} "
            f"J2={record['J2_label'].replace('p', '.')} D={record['D']} "
            f"chi={record.get('chi')} checkpoint={record['checkpoint']}"
        )

    for key in sorted(unique, key=_key_sort):
        import_choice(unique[key], summary_root, dry_run=dry_run)
    archived = archive_incomplete_runs(
        new_scan.incomplete,
        initial_summary,
        rerun_dir,
        dry_run=dry_run,
    )
    return Preparation(
        initial_summary=initial_summary,
        new_scan=new_scan,
        unique=unique,
        conflicts=conflicts,
        archived=archived,
    )


def _parsed_observation(choice: Choice, analysis_module):
    return analysis_module.parse_plain_file(str(choice.candidate.observation))


def _current_summary_D_data(
    summary_root: Path,
    ansatz: str,
    j2: str,
    analysis_module,
) -> dict[int, dict]:
    if ansatz == "neel_symmetrized":
        choices, warnings = discover_legacy_neel(LEGACY_NEEL_ROOT)
        for warning in warnings:
            print(f"WARNING: {warning}")
        result = {}
        for (choice_ansatz, choice_j2, D), choice in choices.items():
            if choice_ansatz != ansatz or choice_j2 != j2:
                continue
            if D < analysis_module.MIN_PLOT_D:
                continue
            try:
                result[D] = _parsed_observation(choice, analysis_module)
            except Exception as exc:
                print(
                    f"WARNING: cannot plot legacy observable "
                    f"{choice.candidate.observation}: {exc}"
                )
        return result
    folder = summary_root / f"J2_{j2}" / ansatz
    if not folder.is_dir():
        return {}
    result: dict[int, dict] = {}
    for d_dir in sorted(folder.glob("D_*")):
        match = SUMMARY_D_RE.match(d_dir.name)
        if not match or not d_dir.is_dir():
            continue
        D = int(match.group("D"))
        if D < analysis_module.MIN_PLOT_D:
            continue
        observation = d_dir / "energy_magnetization_correlation.txt"
        if not observation.is_file():
            continue
        try:
            result[D] = analysis_module.parse_plain_file(str(observation))
        except Exception as exc:
            print(f"WARNING: cannot plot summary observable {observation}: {exc}")
    return result


def _control_columns(option_lists: list[list[Choice]]) -> list[list[tuple[int, list[Choice]]]]:
    """Pack D radio groups into readable columns on the left of the plots."""
    groups = [(choices[0].candidate.D, choices) for choices in option_lists]
    units = [len(choices) + 1.7 for _, choices in groups]
    n_columns = min(3, max(1, math.ceil(sum(units) / 22.0)))
    columns: list[list[tuple[int, list[Choice]]]] = [[] for _ in range(n_columns)]
    loads = [0.0] * n_columns
    for group, weight in zip(groups, units):
        index = min(range(n_columns), key=loads.__getitem__)
        columns[index].append(group)
        loads[index] += weight
    return columns


def _unique_display_labels(choices: list[Choice]) -> list[str]:
    """Return stable, human-readable labels even for byte-for-byte duplicate runs."""
    raw = [choice.display_label for choice in choices]
    result: list[str] = []
    for label, choice in zip(raw, choices):
        if raw.count(label) == 1:
            result.append(label)
            continue
        run_hint = hashlib.sha1(choice.run_id.encode("utf-8")).hexdigest()[:6]
        result.append(f"{label} | run {run_hint}")
    return result


def resolve_one_ansatz_j2(
    ansatz: str,
    j2: str,
    conflicts: dict[int, list[Choice]],
    summary_root: Path,
    analysis,
    plt,
    widgets,
) -> dict[int, Choice] | None:
    """Show one blocking GUI and return confirmed choices, or None on close."""
    selections: dict[int, Choice] = {}
    for D, choices in conflicts.items():
        summary_index = next(
            (index for index, choice in enumerate(choices) if choice.origin == "summary"),
            None,
        )
        active = summary_index if summary_index is not None else min(
            range(len(choices)),
            key=lambda index: choices[index].candidate.energy,
        )
        selections[D] = choices[active]

    packed = _control_columns(list(conflicts.values()))
    n_control_columns = len(packed)
    fig_width = 9.0 + 4.8 * n_control_columns
    fig = plt.figure(figsize=(fig_width, 9.2))
    control_left = 0.015
    controls_width = 0.30 + 0.08 * (n_control_columns - 1)
    plot_left = controls_width + 0.055
    plot_width = 0.98 - plot_left
    axes = [
        fig.add_axes([plot_left, 0.69, plot_width, 0.25]),
        fig.add_axes([plot_left, 0.38, plot_width, 0.25]),
        fig.add_axes([plot_left, 0.07, plot_width, 0.25]),
    ]
    status_ax = fig.add_axes([control_left, 0.015, controls_width - 0.02, 0.035])
    status_ax.axis("off")
    status_text = status_ax.text(
        0.0,
        0.5,
        "Choose exactly one source for every D, then Confirm.",
        va="center",
        fontsize=9,
    )

    radio_objects = []
    radio_axes = []
    control_gap = 0.012
    column_width = (controls_width - control_left - 0.02) / n_control_columns
    for column_index, groups in enumerate(packed):
        total_units = sum(len(choices) + 1.7 for _, choices in groups)
        y_top = 0.965
        usable_height = 0.84
        for D, choices in groups:
            units = len(choices) + 1.7
            height = usable_height * units / total_units - control_gap
            x = control_left + column_index * column_width
            y = y_top - height
            radio_ax = fig.add_axes(
                [x, y, column_width - 0.018, height],
                facecolor="#f5f5f5",
            )
            radio_ax.set_title(f"D={D}", loc="left", fontsize=10, fontweight="bold")
            labels = _unique_display_labels(choices)
            active_choice = selections[D]
            active = choices.index(active_choice)
            radio = widgets.RadioButtons(
                radio_ax,
                labels,
                active=active,
                activecolor="tab:blue",
            )
            for label in radio.labels:
                label.set_fontsize(7.2)
            radio_objects.append((D, choices, labels, radio))
            radio_axes.append(radio_ax)
            y_top = y - control_gap

    confirmed = {"value": False}
    plot_valid = {"value": False}

    def redraw() -> None:
        try:
            D_data = _current_summary_D_data(summary_root, ansatz, j2, analysis)
            for D, choice in selections.items():
                D_data[D] = _parsed_observation(choice, analysis)
            processed = analysis.process_parsed_D_data(D_data)
            if processed is None:
                raise ValueError("no plottable observations for this (ansatz, J2)")
            # plot_analysis_Windows normally omits D=2.  A conflict at D=2
            # must nevertheless be visible immediately when its radio choice
            # changes, so widen only this interactive figure when necessary.
            smallest_D = min(D_data)
            analysis.INVERSE_D_X_MAX = max(
                (1.0 / analysis.MIN_PLOT_D) * 1.30,
                (1.0 / smallest_D) * 1.10,
            )
            for axis in axes:
                axis.clear()
            analysis.plot_col_energy(axes[0], processed, show_xlabel=False)
            analysis.plot_col_mag(axes[1], processed, show_xlabel=False)
            analysis.plot_col_nn(axes[2], processed, show_xlabel=True)
            ylims = analysis.compute_j2_ylims({ansatz: processed})
            axes[0].set_ylim(ylims["energy"])
            axes[1].set_ylim(ylims["mag"])
            axes[2].set_ylim(ylims["nn"])
            axes[0].set_ylabel("Energy per site")
            axes[1].set_ylabel(r"$m_{\mathrm{N\acute{e}el}}$")
            axes[2].set_ylabel(r"$\langle S_i \cdot S_j \rangle$ (NN)")
            axes[0].set_title(
                f"{ansatz} — J2={j2.replace('p', '.')} — selected sources",
                fontsize=13,
            )
            status_text.set_text(
                "Current: "
                + "; ".join(
                    f"D={D} → {_shorten(choice.source_label, 18)}"
                    for D, choice in sorted(selections.items())
                )
            )
            status_text.set_color("black")
            plot_valid["value"] = True
        except Exception as exc:
            status_text.set_text(f"Cannot redraw: {exc}")
            status_text.set_color("crimson")
            plot_valid["value"] = False
        fig.canvas.draw_idle()

    def make_radio_callback(
        D: int,
        choices: list[Choice],
        labels: list[str],
    ):
        label_map = dict(zip(labels, choices))

        def callback(label: str) -> None:
            selections[D] = label_map[label]
            redraw()

        return callback

    for D, choices, labels, radio in radio_objects:
        radio.on_clicked(make_radio_callback(D, choices, labels))

    confirm_ax = fig.add_axes(
        [control_left, 0.055, min(0.13, controls_width * 0.35), 0.055]
    )
    confirm_button = widgets.Button(
        confirm_ax,
        "Confirm",
        color="#c9efc5",
        hovercolor="#98dc91",
    )

    def on_confirm(_event) -> None:
        if not plot_valid["value"]:
            status_text.set_text(
                "Confirm blocked: the current selection is not plottable."
            )
            status_text.set_color("crimson")
            fig.canvas.draw_idle()
            return
        try:
            import_choices_atomically(
                (choice for _D, choice in sorted(selections.items())),
                summary_root,
            )
            confirmed["value"] = True
            plt.close(fig)
        except Exception as exc:
            status_text.set_text(f"Confirm failed; window kept open: {exc}")
            status_text.set_color("crimson")
            fig.canvas.draw_idle()

    confirm_button.on_clicked(on_confirm)
    redraw()
    plt.show(block=True)
    # Keep widget references live until show() returns.
    _ = (radio_objects, radio_axes, confirm_button)
    return dict(selections) if confirmed["value"] else None


def group_conflicts_by_ansatz_j2(
    conflicts: dict[tuple[str, str, int], list[Choice]],
) -> dict[tuple[str, str], dict[int, list[Choice]]]:
    """Build exactly one future figure group per (ansatz, J2)."""
    grouped: dict[tuple[str, str], dict[int, list[Choice]]] = {}
    for (ansatz, j2, D), choices in conflicts.items():
        grouped.setdefault((ansatz, j2), {})[D] = choices
    return grouped


def resolve_conflicts_interactively(
    conflicts: dict[tuple[str, str, int], list[Choice]],
    summary_root: Path,
    backend: str,
) -> tuple[dict[tuple[str, str], dict[int, Choice]], bool]:
    if not conflicts:
        return {}, True

    os.environ["PLOT_ANALYSIS_INTERACTIVE"] = "1"
    import matplotlib

    try:
        matplotlib.use(backend, force=True)
        import matplotlib.pyplot as plt
        import matplotlib.widgets as widgets
    except Exception as exc:
        raise RuntimeError(
            f"cannot start matplotlib GUI backend {backend!r}: {exc}"
        ) from exc

    try:
        import plot_analysis_Windows as analysis
    except ImportError:
        from . import plot_analysis_Windows as analysis

    by_ansatz_j2 = group_conflicts_by_ansatz_j2(conflicts)

    decisions: dict[tuple[str, str], dict[int, Choice]] = {}
    group_sort = lambda key: (key[0].casefold(), float(key[1].replace("p", ".")))
    for ansatz, j2 in sorted(by_ansatz_j2, key=group_sort):
        print(
            f"Opening conflict window for ansatz={ansatz} "
            f"J2={j2.replace('p', '.')} "
            f"({len(by_ansatz_j2[(ansatz, j2)])} conflicting D values) ..."
        )
        selected = resolve_one_ansatz_j2(
            ansatz,
            j2,
            by_ansatz_j2[(ansatz, j2)],
            summary_root,
            analysis,
            plt,
            widgets,
        )
        if selected is None:
            print(
                f"STOPPED: ansatz={ansatz} J2={j2.replace('p', '.')} "
                "was closed without Confirm. Later groups were not opened."
            )
            return decisions, False
        decisions[(ansatz, j2)] = selected
    return decisions, True


def _choice_record(choice: Choice) -> dict:
    candidate = choice.candidate
    return {
        "origin": choice.origin,
        "ansatz": candidate.ansatz,
        "source_label": choice.source_label,
        "run_id": choice.run_id,
        "timestamp": candidate.timestamp,
        "J2_label": candidate.j2,
        "D": candidate.D,
        "chi": candidate.chi,
        "energy_per_site": candidate.energy,
        "elapsed_hours": choice.elapsed_hours,
        "observation": str(candidate.observation),
    }


def write_report(
    path: Path,
    preparation: Preparation,
    decisions: dict[tuple[str, str], dict[int, Choice]],
    all_confirmed: bool,
) -> None:
    payload = {
        "schema": "newdata_all_ansatze_integration",
        "schema_version": 3,
        "all_conflicts_confirmed": all_confirmed,
        "unique_imports": [
            _choice_record(choice)
            for _, choice in sorted(
                preparation.unique.items(),
                key=lambda item: _key_sort(item[0]),
            )
        ],
        "unfinished_archives": preparation.archived,
        "reviewed_olddata_skipped": preparation.new_scan.reviewed_skipped,
        "conflicts": {
            f"{ansatz}__J2_{j2}_D{D}": [_choice_record(choice) for choice in choices]
            for (ansatz, j2, D), choices in sorted(
                preparation.conflicts.items(),
                key=lambda item: _key_sort(item[0]),
            )
        },
        "confirmed_choices": {
            f"{ansatz}__J2_{j2}_D{D}": _choice_record(choice)
            for (ansatz, j2), selected in decisions.items()
            for D, choice in selected.items()
        },
        "warnings": preparation.new_scan.warnings,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def print_conflicts(conflicts: dict[tuple[str, str, int], list[Choice]]) -> None:
    if not conflicts:
        print("No completed duplicate conflicts.")
        return
    print("Completed duplicate conflicts:")
    for (ansatz, j2, D), choices in sorted(
        conflicts.items(),
        key=lambda item: _key_sort(item[0]),
    ):
        print(f"  ansatz={ansatz} J2={j2.replace('p', '.')} D={D}")
        for choice in choices:
            print(f"    - {choice.display_label} [{choice.origin}] {choice.run_id}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-root", type=Path, default=DEFAULT_NEW_ROOT)
    parser.add_argument(
        "--old-root",
        type=Path,
        default=DEFAULT_OLD_ROOT,
        help=(
            "reviewed baseline tree; byte-identical completed checkpoint/observable "
            "pairs are skipped (default: sibling 0730olddata)"
        ),
    )
    parser.add_argument("--summary-root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument(
        "--legacy-neel-root",
        type=Path,
        default=DEFAULT_LEGACY_NEEL_ROOT,
        help="destination for every neel_legacy result",
    )
    parser.add_argument(
        "--rerun-dir",
        type=Path,
        help=(
            "unfinished output folder (default: "
            "<new-root>/_unfinished_all_ansatze_for_rerun)"
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="audit JSON path (default: <new-root>/newdata_all_ansatze_integration_report.json)",
    )
    parser.add_argument(
        "--backend",
        default="TkAgg",
        help="matplotlib GUI backend for conflict windows (default: TkAgg)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="scan and print every action; do not copy or open GUI windows",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help=(
            "perform unique imports and unfinished archives, print conflicts, "
            "but do not open GUI windows"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    global LEGACY_NEEL_ROOT
    args = parse_args(argv)
    new_root = args.new_root.resolve()
    old_root = args.old_root.resolve()
    summary_root = args.summary_root.resolve()
    LEGACY_NEEL_ROOT = args.legacy_neel_root.resolve()
    rerun_dir = (
        args.rerun_dir.resolve()
        if args.rerun_dir
        else new_root / DEFAULT_RERUN_FOLDER
    )
    report = (
        args.report.resolve()
        if args.report
        else new_root / DEFAULT_REPORT
    )
    if rerun_dir == new_root or new_root not in rerun_dir.parents:
        raise ValueError("--rerun-dir must be a child directory of --new-root")

    preparation = prepare(
        new_root,
        summary_root,
        rerun_dir,
        old_root=old_root,
        dry_run=args.dry_run,
    )
    print_conflicts(preparation.conflicts)
    if args.dry_run:
        print("Dry run: no files were written and no GUI was opened.")
        return 0
    if args.prepare_only:
        write_report(report, preparation, {}, not preparation.conflicts)
        print(f"Preparation report: {report}")
        return 0

    decisions, all_confirmed = resolve_conflicts_interactively(
        preparation.conflicts,
        summary_root,
        args.backend,
    )
    write_report(report, preparation, decisions, all_confirmed)
    print(f"Integration report: {report}")
    if all_confirmed:
        print("Done: every completed duplicate conflict was confirmed.")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
