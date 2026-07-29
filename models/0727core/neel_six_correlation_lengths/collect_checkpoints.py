#!/usr/bin/env python3
"""Collect the audited Neel checkpoints used by the six-spectrum test.

The observable files in ``data/D345678910`` are treated as the authoritative
choice of ``(D, chi, energy)``.  A matching ``sweep_*_best.pt`` is searched
recursively below the complete data root and selected by its checkpoint
metadata.  This recovers checkpoints which are missing or stale in the
summary directory itself.

The historical D345678910 runs start at D=4.  When an older optimized
``h_neel`` D=3 checkpoint exists it is converted exactly back to ``a_raw``.
Any still-missing D=2 or D=3 *validation-only* tensor is made by taking the
leading virtual-index block of that J2's selected D=4 tensor and projecting
it back onto exact S3 virtual symmetry.  Such tensors are explicitly marked
as derived in both the checkpoint and the manifest; they are not claimed to
be independently optimized states.
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


DEFAULT_DATA_ROOT = Path(
    r"D:\HyraiOn\ENS_Lyon\Internship\2026-EPFL\data"
)
DEFAULT_SOURCE_ROOT = DEFAULT_DATA_ROOT / "D345678910"
HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "checkpoints"
DEFAULT_MANIFEST = HERE / "checkpoint_manifest.json"
TARGETS = {
    0.0: "neel_symmetrized__J2_0p0_",
    0.265: "neel_symmetrized__J2_0p265_",
}
OBSERVABLE_RE = re.compile(
    r"^D_(?P<D>\d+)_chi_(?P<chi>\d+)_"
    r"energy_magnetization_correlation\.txt$"
)
ENERGY_RE = re.compile(
    r"^energy\s*=\s*(?P<energy>[+-]?[0-9.]+(?:[eE][+-]?\d+)?)\s*$",
    re.MULTILINE,
)


@dataclass(frozen=True)
class Observable:
    j2: float
    D: int
    chi: int
    energy: float
    path: Path


def _j2_tag(j2: float) -> str:
    return f"J2_{j2:g}".replace(".", "p")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_symmetry_error(a: torch.Tensor) -> float:
    denominator = max(float(torch.linalg.norm(a)), 1.0e-300)
    return max(
        float(torch.linalg.norm(a - a.permute(*permutation, 3)))
        / denominator
        for permutation in itertools.permutations(range(3))
    )


def _load_observables(source_root: Path) -> dict[float, list[Observable]]:
    result: dict[float, list[Observable]] = {}
    for j2, prefix in TARGETS.items():
        directories = sorted(
            path
            for path in source_root.glob(f"{prefix}*")
            if path.is_dir()
        )
        if len(directories) != 1:
            raise RuntimeError(
                f"Expected exactly one {prefix!r} directory below "
                f"{source_root}; found {directories}."
            )
        observations: list[Observable] = []
        for path in directories[0].iterdir():
            match = OBSERVABLE_RE.match(path.name)
            if match is None:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            energy_match = ENERGY_RE.search(text)
            if energy_match is None:
                raise ValueError(f"No total energy in {path}.")
            observations.append(
                Observable(
                    j2=j2,
                    D=int(match.group("D")),
                    chi=int(match.group("chi")),
                    energy=float(energy_match.group("energy")),
                    path=path,
                )
            )
        if not observations:
            raise RuntimeError(f"No authoritative observables in {directories[0]}.")
        result[j2] = observations
    return result


def _read_checkpoint(path: Path) -> dict[str, Any] | None:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    a = payload.get("a_raw")
    if not isinstance(a, torch.Tensor) or a.ndim != 4:
        return None
    return payload


def _candidate_score(
    payload: dict[str, Any],
    observable: Observable,
) -> tuple[float, float] | None:
    a = payload["a_raw"]
    D = int(payload.get("D_bond", a.shape[0]))
    chi = int(payload.get("chi", -1))
    if D != observable.D or chi != observable.chi:
        return None
    if tuple(a.shape[:3]) != (D, D, D):
        return None
    symmetry_error = _tensor_symmetry_error(a.detach().double())
    if not math.isfinite(symmetry_error) or symmetry_error > 1.0e-9:
        return None
    checkpoint_energy = payload.get("energy")
    if checkpoint_energy is not None and math.isfinite(float(checkpoint_energy)):
        energy_error = abs(float(checkpoint_energy) - observable.energy)
    else:
        loss = payload.get("loss")
        if loss is None or not math.isfinite(float(loss)):
            return None
        # Loss is normally within the CTMRG re-evaluation error of the
        # observable energy, but prefer explicit checkpoint energy whenever
        # it exists.
        energy_error = abs(float(loss) - observable.energy) + 1.0e-7
    return energy_error, symmetry_error


def _select_observation_per_D(
    observations: list[Observable],
) -> list[Observable]:
    """Use the lowest authoritative total energy when several chi exist."""

    selected: dict[int, Observable] = {}
    for observation in observations:
        previous = selected.get(observation.D)
        if previous is None or observation.energy < previous.energy:
            selected[observation.D] = observation
    return [selected[D] for D in sorted(selected)]


def _find_best_checkpoint(
    data_root: Path,
    observation: Observable,
) -> tuple[Path, dict[str, Any], float, float]:
    pattern = f"sweep_D{observation.D}_chi{observation.chi}_best*.pt"
    ranked: list[tuple[float, int, str, Path, dict[str, Any], float]] = []
    for path in data_root.rglob(pattern):
        payload = _read_checkpoint(path)
        if payload is None:
            continue
        score = _candidate_score(payload, observation)
        if score is None:
            continue
        energy_error, symmetry_error = score
        # Prefer a non-backup, shallower path when exact duplicates exist.
        backup_penalty = int(
            "ALLscratchBackup" in path.parts or "scratchBackup" in path.parts
        )
        ranked.append(
            (
                energy_error,
                backup_penalty,
                str(path).lower(),
                path,
                payload,
                symmetry_error,
            )
        )
    if not ranked:
        raise FileNotFoundError(
            f"No symmetric checkpoint matches D={observation.D}, "
            f"chi={observation.chi}, J2={observation.j2:g}."
        )
    ranked.sort(key=lambda item: item[:3])
    energy_error, _, _, path, payload, symmetry_error = ranked[0]
    if energy_error > 5.0e-4:
        raise RuntimeError(
            f"Best candidate {path} misses observable energy "
            f"{observation.energy:.16g} by {energy_error:.3e}."
        )
    return path, payload, energy_error, symmetry_error


def _derive_low_D(
    source_payload: dict[str, Any],
    *,
    D: int,
    j2: float,
    source_path: Path,
) -> dict[str, Any]:
    source = source_payload["a_raw"].detach().cpu()
    if source.shape[0] < D:
        raise ValueError("Cannot derive a larger virtual bond dimension.")
    block = source[:D, :D, :D, :].clone()
    block = sum(
        block.permute(*permutation, 3)
        for permutation in itertools.permutations(range(3))
    ) / 6.0
    norm = torch.linalg.norm(block)
    if not torch.isfinite(norm) or float(norm) == 0.0:
        raise RuntimeError("Derived validation tensor has zero/non-finite norm.")
    block = block / norm
    return {
        "a_raw": block,
        "D_bond": D,
        "chi": max(8, 2 * D * D),
        "J2": j2,
        "derived_validation_only": True,
        "derived_from": str(source_path),
        "derivation": (
            f"leading {D}x{D}x{D} virtual block, exact S3 projection, "
            "unit Frobenius normalization"
        ),
    }


def _recover_free_parameter_D3(
    data_root: Path,
    *,
    j2: float,
) -> tuple[Path, dict[str, Any]] | None:
    """Recover a historical optimized h_neel checkpoint when one exists."""

    j2_text = "0p0" if j2 == 0.0 else f"{j2:g}".replace(".", "p")
    candidates: list[tuple[float, str, Path, dict[str, Any]]] = []
    pattern = (
        f"neel_free_param__J2_{j2_text}_*/"
        "sweep_D3_chi*_best*.pt"
    )
    for path in (data_root / "raw").glob(pattern):
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        h = payload.get("h_neel")
        loss = payload.get("loss")
        if not isinstance(h, torch.Tensor) or loss is None:
            continue
        if int(payload.get("D_bond", -1)) != 3:
            continue
        candidates.append((float(loss), str(path).lower(), path, payload))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    _, _, source, payload = candidates[0]
    converted = dict(payload)
    converted["a_raw"] = core.neel_param_to_a(
        payload["h_neel"].detach().cpu(), 3
    ).detach()
    converted["recovered_from_h_neel"] = True
    converted["source_checkpoint"] = str(source)
    converted["derived_validation_only"] = False
    return source, converted


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--no-derived-low-D",
        action="store_true",
        help="Do not make validation-only D=2,3 tensors.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = args.data_root.resolve()
    source_root = args.source_root.resolve()
    output = args.output.resolve()
    observations = _load_observables(source_root)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "data_root": str(data_root),
        "source_root": str(source_root),
        "cases": [],
    }

    for j2, available in sorted(observations.items()):
        selected_payloads: dict[int, tuple[Path, dict[str, Any]]] = {}
        for observation in _select_observation_per_D(available):
            source, payload, energy_error, symmetry_error = (
                _find_best_checkpoint(data_root, observation)
            )
            destination = (
                output / _j2_tag(j2) / f"D_{observation.D}" / "tensor_best.pt"
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            selected_payloads[observation.D] = (source, payload)
            entry = {
                "J2": j2,
                "D": observation.D,
                "chi": observation.chi,
                "derived_validation_only": False,
                "observable": str(observation.path),
                "observable_energy": observation.energy,
                "source_checkpoint": str(source),
                "energy_match_error": energy_error,
                "virtual_symmetry_relative_error": symmetry_error,
                "destination": str(destination),
                "sha256": _sha256(destination),
            }
            manifest["cases"].append(entry)
            print(
                f"J2={j2:g} D={observation.D} chi={observation.chi}: "
                f"{source} -> {destination} "
                f"(energy error {energy_error:.3e})",
                flush=True,
            )

        if not args.no_derived_low_D:
            source_D = min(selected_payloads)
            source_path, source_payload = selected_payloads[source_D]
            for D in (2, 3):
                destination = (
                    output / _j2_tag(j2) / f"D_{D}" / "tensor_best.pt"
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                recovered = (
                    _recover_free_parameter_D3(data_root, j2=j2)
                    if D == 3
                    else None
                )
                if recovered is None:
                    payload = _derive_low_D(
                        source_payload,
                        D=D,
                        j2=j2,
                        source_path=source_path,
                    )
                    case_source = source_path
                    derived = True
                else:
                    case_source, payload = recovered
                    derived = False
                torch.save(payload, destination)
                symmetry_error = _tensor_symmetry_error(
                    payload["a_raw"].detach().double()
                )
                manifest["cases"].append(
                    {
                        "J2": j2,
                        "D": D,
                        "chi": int(payload["chi"]),
                        "derived_validation_only": derived,
                        "recovered_from_h_neel": recovered is not None,
                        "source_checkpoint": str(case_source),
                        "virtual_symmetry_relative_error": symmetry_error,
                        "destination": str(destination),
                        "sha256": _sha256(destination),
                    }
                )
                if recovered is None:
                    print(
                        f"J2={j2:g} D={D}: derived validation tensor from "
                        f"{source_path} -> {destination}",
                        flush=True,
                    )
                else:
                    print(
                        f"J2={j2:g} D={D}: recovered optimized h_neel from "
                        f"{case_source} -> {destination}",
                        flush=True,
                    )

    manifest["cases"].sort(key=lambda item: (item["J2"], item["D"]))
    _atomic_json(args.manifest.resolve(), manifest)
    print(f"Wrote {len(manifest['cases'])} cases to {args.manifest.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
