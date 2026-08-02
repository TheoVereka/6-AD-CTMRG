#!/usr/bin/env python3
"""Build the complete, portable Neel correlation-length checkpoint bundle.

Two local data collections are scanned:

* every ``neel_symmetrized__J2_*`` run in ``D345678910``;
* every ansatz directory whose name starts with ``neel`` in ``0713summary``.

There is exactly one staged checkpoint per ``(J2, D)``.  A summary checkpoint
always replaces a D345678910 checkpoint with the same key.  If the summary
itself contains several Neel ansatze for one key, its lowest recorded energy
is selected.  The generated manifest records every choice and override.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

import core_C3 as core


DEFAULT_DATA_ROOT = Path(r"D:\HyraiOn\ENS_Lyon\Internship\2026-EPFL\data")
HERE = Path(__file__).resolve().parent
DEFAULT_D_SOURCE = DEFAULT_DATA_ROOT / "D345678910"
DEFAULT_SUMMARY = DEFAULT_DATA_ROOT / "0713summary"
DEFAULT_OUTPUT = HERE / "checkpoints"
DEFAULT_MANIFEST = HERE / "checkpoint_manifest.json"
RUN_RE = re.compile(
    r"^neel_symmetrized__J2_(?P<j2>[0-9]+p[0-9]+)(?:_|$)", re.IGNORECASE
)
OBSERVABLE_RE = re.compile(
    r"^D_(?P<D>\d+)_chi_(?P<chi>\d+)_"
    r"energy_magnetization_correlation\.txt$"
)
ENERGY_RE = re.compile(
    r"^energy\s*=\s*(?P<energy>[+-]?[0-9.]+(?:[eE][+-]?\d+)?)\s*$",
    re.MULTILINE,
)
ENERGY_PER_SITE_RE = re.compile(
    r"^energy_per_site\s*=\s*(?P<energy>[+-]?[0-9.]+(?:[eE][+-]?\d+)?)\s*$",
    re.MULTILINE,
)


@dataclass(frozen=True)
class Candidate:
    j2: float
    D: int
    chi: int
    energy: float
    source: str
    ansatz: str
    checkpoint: Path
    observable: Path | None
    manifest: Path | None


def _j2_from_tag(tag: str) -> float:
    return float(tag.replace("p", "."))


def _j2_tag(j2: float) -> str:
    return f"J2_{j2:g}".replace(".", "p")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _symmetry_error(a: torch.Tensor) -> float:
    denominator = max(float(torch.linalg.norm(a)), 1.0e-300)
    return max(
        float(torch.linalg.norm(a - a.permute(*permutation, 3))) / denominator
        for permutation in itertools.permutations(range(3))
    )


def _read_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint is not a dictionary: {path}")
    return payload


def _canonical_payload(
    candidate: Candidate,
) -> tuple[dict[str, Any], float]:
    payload = dict(_read_payload(candidate.checkpoint))
    a = payload.get("a_raw")
    if not isinstance(a, torch.Tensor):
        h = payload.get("h_neel")
        if not isinstance(h, torch.Tensor):
            raise ValueError(
                f"Neel checkpoint has neither a_raw nor h_neel: "
                f"{candidate.checkpoint}"
            )
        a = core.neel_param_to_a(h.detach().cpu(), candidate.D).detach()
        payload["converted_from_h_neel"] = True
    a = a.detach().cpu().contiguous()
    if tuple(a.shape[:3]) != (candidate.D,) * 3 or a.ndim != 4:
        raise ValueError(
            f"Tensor shape {tuple(a.shape)} disagrees with D={candidate.D}: "
            f"{candidate.checkpoint}"
        )
    error = _symmetry_error(a.double())
    if not math.isfinite(error) or error > 1.0e-9:
        raise ValueError(
            f"Tensor is not three-leg symmetric (error={error:.3e}): "
            f"{candidate.checkpoint}"
        )
    payload["a_raw"] = a
    payload["D_bond"] = candidate.D
    payload["chi"] = candidate.chi
    payload["J2"] = candidate.j2
    payload["bundle_source"] = candidate.source
    payload["bundle_ansatz"] = candidate.ansatz
    payload["bundle_source_checkpoint"] = str(candidate.checkpoint.resolve())
    payload["derived_validation_only"] = False
    return payload, error


def _checkpoint_energy_error(
    payload: dict[str, Any], observable_energy: float
) -> float:
    values = [
        float(payload[key])
        for key in ("energy", "loss")
        if payload.get(key) is not None and math.isfinite(float(payload[key]))
    ]
    return min(
        (abs(value - observable_energy) for value in values), default=math.inf
    )


def _select_d345_checkpoint(
    run_dir: Path,
    *,
    data_root: Path,
    j2: float,
    D: int,
    chi: int,
    observable_energy: float,
) -> tuple[Path, int]:
    filename_pattern = f"sweep_D{D}_chi{chi}_best*.pt"

    def rank(
        paths: list[Path], *, require_chi: bool
    ) -> list[tuple[float, int, int, str, Path, int]]:
        ranked: list[tuple[float, int, int, str, Path, int]] = []
        for path in paths:
            try:
                payload = _read_payload(path)
                a = payload.get("a_raw")
                if not isinstance(a, torch.Tensor):
                    continue
                if int(payload.get("D_bond", a.shape[0])) != D:
                    continue
                checkpoint_chi = int(payload.get("chi", -1))
                if checkpoint_chi < 1 or (require_chi and checkpoint_chi != chi):
                    continue
                if tuple(a.shape[:3]) != (D, D, D):
                    continue
                if _symmetry_error(a.detach().double()) > 1.0e-9:
                    continue
                energy_error = _checkpoint_energy_error(payload, observable_energy)
                duplicate_suffix = int(path.stem != f"sweep_D{D}_chi{chi}_best")
                backup_penalty = int(
                    "ALLscratchBackup" in path.parts or "scratchBackup" in path.parts
                )
                ranked.append(
                    (
                        energy_error,
                        backup_penalty,
                        duplicate_suffix,
                        str(path).lower(),
                        path,
                        checkpoint_chi,
                    )
                )
            except (OSError, RuntimeError, TypeError, ValueError):
                continue
        ranked.sort(key=lambda item: item[:4])
        return ranked

    ranked = rank(sorted(run_dir.glob(filename_pattern)), require_chi=True)
    if ranked and ranked[0][0] <= 5.0e-4:
        return ranked[0][4], ranked[0][5]

    # Some files in D345678910 are stale copies.  Recover the matching
    # checkpoint from the complete data archive, but require the J2 tag and
    # rank by agreement with the authoritative D345678910 observable.
    j2_text = "0p0" if j2 == 0.0 else f"{j2:g}".replace(".", "p")
    fallback_paths = [
        path
        for path in data_root.rglob(filename_pattern)
        if any(
            part.casefold().startswith(f"neel_symmetrized__j2_{j2_text}_".casefold())
            for part in path.parts
        )
    ]
    ranked.extend(rank(fallback_paths, require_chi=True))
    ranked.sort(key=lambda item: item[:4])
    if ranked and ranked[0][0] <= 5.0e-4:
        return ranked[0][4], ranked[0][5]

    # The observable chi is occasionally a later CTMRG evaluation chi rather
    # than the optimizer chi stored in the tensor.  In that case use the
    # same-D checkpoint whose energy/loss best matches the observable.
    any_chi_pattern = f"sweep_D{D}_chi*_best*.pt"
    any_chi_paths = sorted(run_dir.glob(any_chi_pattern))
    any_chi_paths.extend(
        path
        for path in data_root.rglob(any_chi_pattern)
        if any(
            part.casefold().startswith(
                f"neel_symmetrized__j2_{j2_text}_".casefold()
            )
            for part in path.parts
        )
    )
    ranked = rank(any_chi_paths, require_chi=False)
    if ranked and ranked[0][0] <= 5.0e-4:
        return ranked[0][4], ranked[0][5]

    # A few manually rescued checkpoints use names such as ``28legD10.pt``:
    # the leading digits encode J2=0.28 while the apparent D in the filename
    # is stale.  This final recovery is deliberately strict: exact payload D
    # and chi, exact S3 symmetry (enforced by rank), and energy agreement at
    # 5e-5.  It therefore recovers the mislabeled tensor without accepting a
    # C6Ypi/twoC3 checkpoint with a merely similar energy.
    fractional_tag = f"{j2:g}".partition(".")[2]
    hinted_paths = [
        path
        for path in data_root.rglob("*.pt")
        if fractional_tag
        and path.stem.casefold().startswith(f"{fractional_tag}leg")
    ]
    ranked = rank(hinted_paths, require_chi=True)
    if ranked and ranked[0][0] <= 5.0e-5:
        return ranked[0][4], ranked[0][5]

    raise FileNotFoundError(
        f"No valid checkpoint matches J2={j2:g}, D={D}, chi={chi} "
        f"in {run_dir} or {data_root}."
    )


def _scan_d345(
    root: Path, *, data_root: Path, min_D: int, max_D: int
) -> tuple[dict[tuple[float, int], Candidate], list[str]]:
    observations: dict[tuple[float, int], list[tuple[float, int, Path, Path]]] = {}
    warnings: list[str] = []
    for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        match = RUN_RE.match(run_dir.name)
        if match is None:
            continue
        j2 = _j2_from_tag(match.group("j2"))
        for observable in sorted(run_dir.iterdir()):
            obs_match = OBSERVABLE_RE.match(observable.name)
            if obs_match is None:
                continue
            D = int(obs_match.group("D"))
            if not min_D <= D <= max_D:
                continue
            chi = int(obs_match.group("chi"))
            text = observable.read_text(encoding="utf-8", errors="replace")
            energy_match = ENERGY_RE.search(text)
            if energy_match is None:
                raise ValueError(f"No total energy in {observable}.")
            energy = float(energy_match.group("energy"))
            try:
                checkpoint, checkpoint_chi = _select_d345_checkpoint(
                    run_dir,
                    data_root=data_root,
                    j2=j2,
                    D=D,
                    chi=chi,
                    observable_energy=energy,
                )
            except FileNotFoundError as error:
                warning = f"Skipped unmatched observable {observable}: {error}"
                warnings.append(warning)
                print(f"WARNING: {warning}", flush=True)
                continue
            observations.setdefault((j2, D), []).append(
                (energy, checkpoint_chi, checkpoint, observable)
            )
    selected: dict[tuple[float, int], Candidate] = {}
    for (j2, D), choices in observations.items():
        energy, chi, checkpoint, observable = min(
            choices, key=lambda item: (item[0], item[1], str(item[2]).lower())
        )
        selected[(j2, D)] = Candidate(
            j2=j2,
            D=D,
            chi=chi,
            energy=energy,
            source="D345678910",
            ansatz="neel_symmetrized",
            checkpoint=checkpoint,
            observable=observable,
            manifest=None,
        )
    return selected, warnings


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _summary_energy(d_dir: Path, manifest: dict[str, Any]) -> tuple[float, Path | None]:
    value = manifest.get("energy_per_site")
    if value is not None and math.isfinite(float(value)):
        observable_name = manifest.get("source_observation")
        observable = d_dir / str(observable_name) if observable_name else None
        if observable is None or not observable.is_file():
            observable = d_dir / "energy_magnetization_correlation.txt"
        return float(value), observable if observable.is_file() else None
    observable = d_dir / "energy_magnetization_correlation.txt"
    if observable.is_file():
        text = observable.read_text(encoding="utf-8", errors="replace")
        match = ENERGY_PER_SITE_RE.search(text)
        if match is not None:
            return float(match.group("energy")), observable
        match = ENERGY_RE.search(text)
        if match is not None:
            return float(match.group("energy")) / 6.0, observable
    return math.inf, observable if observable.is_file() else None


def _scan_summary(
    root: Path, *, min_D: int, max_D: int
) -> tuple[dict[tuple[float, int], Candidate], int]:
    candidates: dict[tuple[float, int], list[Candidate]] = {}
    neel_directories = 0
    for j2_dir in sorted(path for path in root.glob("J2_*") if path.is_dir()):
        try:
            j2 = _j2_from_tag(j2_dir.name.removeprefix("J2_"))
        except ValueError:
            continue
        for ansatz_dir in sorted(path for path in j2_dir.iterdir() if path.is_dir()):
            if not ansatz_dir.name.casefold().startswith("neel"):
                continue
            neel_directories += 1
            for d_dir in sorted(path for path in ansatz_dir.glob("D_*") if path.is_dir()):
                try:
                    D = int(d_dir.name.removeprefix("D_"))
                except ValueError:
                    continue
                if not min_D <= D <= max_D:
                    continue
                checkpoint = d_dir / "tensor_best.pt"
                if not checkpoint.is_file():
                    continue
                manifest_path = d_dir / "manifest.json"
                manifest = _read_json(manifest_path)
                payload = _read_payload(checkpoint)
                tensor = payload.get("a_raw", payload.get("h_neel"))
                inferred_D = int(payload.get("D_bond", D))
                if inferred_D != D:
                    raise ValueError(f"Summary checkpoint D mismatch: {checkpoint}")
                chi = int(manifest.get("chi", payload.get("chi", -1)))
                if chi < 1:
                    raise ValueError(f"No valid chi for summary checkpoint: {checkpoint}")
                energy, observable = _summary_energy(d_dir, manifest)
                candidates.setdefault((j2, D), []).append(
                    Candidate(
                        j2=j2,
                        D=D,
                        chi=chi,
                        energy=energy,
                        source="0713summary",
                        ansatz=str(manifest.get("ansatz", ansatz_dir.name)),
                        checkpoint=checkpoint,
                        observable=observable,
                        manifest=manifest_path if manifest_path.is_file() else None,
                    )
                )
    selected = {
        key: min(
            choices,
            key=lambda item: (
                item.energy,
                item.ansatz.casefold(),
                str(item.checkpoint).lower(),
            ),
        )
        for key, choices in candidates.items()
    }
    return selected, neel_directories


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--d345-root", type=Path, default=DEFAULT_D_SOURCE)
    parser.add_argument("--summary-root", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--min-D", type=int, default=3)
    parser.add_argument("--max-D", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    d345_root = args.d345_root.resolve()
    summary_root = args.summary_root.resolve()
    output = args.output.resolve()
    if not d345_root.is_dir():
        raise FileNotFoundError(d345_root)
    if not summary_root.is_dir():
        raise FileNotFoundError(summary_root)
    if args.min_D < 1 or args.max_D < args.min_D:
        raise ValueError("Invalid D range.")

    original, discovery_warnings = _scan_d345(
        d345_root,
        data_root=args.data_root.resolve(),
        min_D=args.min_D,
        max_D=args.max_D,
    )
    summary, summary_neel_directories = _scan_summary(
        summary_root, min_D=args.min_D, max_D=args.max_D
    )
    selected = dict(original)
    overrides = sorted(set(original) & set(summary))
    selected.update(summary)
    if not selected:
        raise RuntimeError("No Neel checkpoints were discovered.")

    # This directory is entirely generated by this script; rebuilding it
    # prevents removed summary entries from surviving into future uploads.
    if output.exists():
        if output == HERE or HERE not in output.parents:
            raise ValueError(f"Refusing to replace unsafe output path: {output}")
        shutil.rmtree(output)

    cases: list[dict[str, Any]] = []
    for (j2, D), candidate in sorted(selected.items()):
        destination = output / _j2_tag(j2) / f"D_{D}" / "tensor_best.pt"
        payload, symmetry_error = _canonical_payload(candidate)
        _atomic_torch_save(destination, payload)
        entry = {
            "J2": j2,
            "D": D,
            "chi": candidate.chi,
            "ansatz": candidate.ansatz,
            "source_collection": candidate.source,
            "source_checkpoint": str(candidate.checkpoint.resolve()),
            "source_checkpoint_sha256": _sha256(candidate.checkpoint),
            "source_observable": (
                None if candidate.observable is None else str(candidate.observable.resolve())
            ),
            "source_manifest": (
                None if candidate.manifest is None else str(candidate.manifest.resolve())
            ),
            "selection_energy": candidate.energy,
            "overrides_D345678910": (j2, D) in overrides,
            "virtual_symmetry_relative_error": symmetry_error,
            "destination": str(destination.relative_to(HERE)),
            "sha256": _sha256(destination),
        }
        cases.append(entry)
        override_note = " (overrides D345678910)" if (j2, D) in overrides else ""
        print(
            f"J2={j2:g} D={D} chi={candidate.chi}: "
            f"{candidate.source}/{candidate.ansatz}{override_note}",
            flush=True,
        )

    manifest = {
        "schema": "neel_correlation_checkpoint_bundle",
        "schema_version": 2,
        "selection_rule": (
            "0713summary overrides D345678910 for identical (J2,D); "
            "lowest energy resolves multiple summary Neel ansatze"
        ),
        "d345_root": str(d345_root),
        "summary_root": str(summary_root),
        "summary_neel_directories_found": summary_neel_directories,
        "D_range": [args.min_D, args.max_D],
        "counts": {
            "D345678910_candidates": len(original),
            "0713summary_candidates": len(summary),
            "overrides": len(overrides),
            "staged": len(cases),
        },
        "warnings": discovery_warnings,
        "cases": cases,
    }
    _atomic_json(args.manifest.resolve(), manifest)
    print(
        f"Wrote {len(cases)} checkpoints ({len(summary)} from 0713summary, "
        f"{len(overrides)} overrides) and {args.manifest.resolve()}.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
