#!/usr/bin/env python3
"""Collect physically clean h=0 two-C3 tensors from all VBC campaigns.

Every accepted tensor must be optimized with the unpinned Hamiltonian, remain
close to the original 0713 two-C3 state in energy and total NN splitting, and
be at least as texture-pure (within an explicit small tolerance) as the Kuma
J2=0.30, D=10 h=0 dimer-plaquette or plaquette reference tensor.

The script never silently discards an observed h=0 stage.  ``all_candidates``
records every parsed candidate and its rejection reasons; malformed stages are
listed separately in ``discovery_errors.csv``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from analyze_existing_twoc3 import Row, parse_observation


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DATA = REPO.parent / "data"

DEFAULT_ORIGINAL = DATA / "0713summary"
DEFAULT_ARCHIVE = DATA / "distinVBCs"
DEFAULT_SMALL_H = DATA / "distinVBCsSmallH" / "Results_Izar_replica1"
DEFAULT_IZAR_REPLICA2 = (
    REPO / "models" / "VBCPinningClusterBundle" / "Results_VBC_branches"
)
DEFAULT_OUTPUT = DATA / "distinVBCsH0TensorCandidates"

BASE_OBSERVATION_RE = re.compile(
    r"^D_(\d+)_chi_(\d+)_energy_magnetization_correlation\.txt$"
)
REPLICA_RE = re.compile(r"replica_(\d+)")
ORIENTATION_RE = re.compile(r"orientation_(\d+)")
H_ZERO_NAMES = {"h_0", "h_0p0", "h_0p00", "h_0p000"}


@dataclass(frozen=True)
class SourceSpec:
    name: str
    root: Path
    replica_filter: int | None = None


@dataclass
class Candidate:
    source: str
    seed_branch: str
    replica: int | None
    orientation: int | None
    observation_path: str
    tensor_path: str
    hyperparams_path: str
    J2: float
    D: int
    chi: int
    energy: float
    delta: float
    middle_fraction: float
    eta: float
    original_energy: float = math.nan
    energy_difference: float = math.nan
    original_delta: float = math.nan
    relative_delta_difference: float = math.nan
    selected_texture: str = ""
    accepted: bool = False
    rejection_reasons: str = ""
    copied_tensor: str = ""


@dataclass(frozen=True)
class DiscoveryError:
    source: str
    path: str
    error: str


def _load_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass

    # PyYAML is optional.  The regex fallback covers the scalar/list fields
    # required to prove that a stage used the original Hamiltonian.
    try:
        import yaml  # type: ignore

        value = yaml.safe_load(text)
        if isinstance(value, dict):
            return value
    except (ImportError, ValueError, TypeError):
        pass

    result: dict[str, Any] = {}
    for line in text.splitlines():
        match = re.match(r"^([A-Za-z0-9_]+)\s*:\s*([^#]+?)\s*$", line)
        if match:
            result[match.group(1)] = match.group(2).strip(" '\"")
    list_match = re.search(
        r"^nn_group_couplings\s*:\s*\[(.*?)\]", text, re.MULTILINE | re.DOTALL
    )
    if list_match:
        result["nn_group_couplings"] = [
            float(part.strip()) for part in list_match.group(1).split(",")
        ]
    return result


def _float_list(value: Any) -> list[float]:
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    if isinstance(value, str):
        stripped = value.strip().strip("[]")
        return [float(part.strip()) for part in stripped.split(",") if part.strip()]
    raise ValueError(f"not a numeric list: {value!r}")


def _eta(row: Row) -> float:
    return 2.0 * row.middle_fraction - 1.0


def _same_float(a: float, b: float, tolerance: float = 1.0e-10) -> bool:
    return math.isclose(a, b, rel_tol=0.0, abs_tol=tolerance)


def _in_scope(row: Row, d_min: int, d_max: int,
              j2_min: float, j2_max: float) -> bool:
    return (d_min <= row.D <= d_max
            and j2_min - 1.0e-10 <= row.J2 <= j2_max + 1.0e-10)


def discover_original(root: Path, d_min: int, d_max: int,
                      j2_min: float, j2_max: float
                      ) -> tuple[list[Candidate], list[DiscoveryError]]:
    candidates: list[Candidate] = []
    errors: list[DiscoveryError] = []
    if not root.is_dir():
        return candidates, [DiscoveryError("original_2c3", str(root),
                                           "source root is missing")]
    paths = sorted(root.glob("J2_*/2tensor_twoC3/D_*/energy_magnetization_correlation.txt"))
    for observation in paths:
        try:
            row = parse_observation(observation)
            if not _in_scope(row, d_min, d_max, j2_min, j2_max):
                continue
            tensor = observation.parent / "tensor_best.pt"
            hyperparams = observation.parent / "hyperparams.yaml"
            candidates.append(Candidate(
                source="original_2c3", seed_branch="original",
                replica=None, orientation=None,
                observation_path=str(observation.resolve()),
                tensor_path=str(tensor.resolve()) if tensor.is_file() else "",
                hyperparams_path=(str(hyperparams.resolve())
                                  if hyperparams.is_file() else ""),
                J2=row.J2, D=row.D, chi=row.chi,
                energy=row.energy_per_site, delta=row.delta,
                middle_fraction=row.middle_fraction, eta=_eta(row),
            ))
        except (OSError, TypeError, ValueError) as exc:
            errors.append(DiscoveryError("original_2c3", str(observation), str(exc)))
    return candidates, errors


def _base_observations(stage: Path) -> list[Path]:
    observations = []
    for path in stage.glob("D_*_chi_*_energy_magnetization_correlation.txt"):
        if BASE_OBSERVATION_RE.fullmatch(path.name):
            observations.append(path)
    return sorted(observations)


def discover_campaign(spec: SourceSpec, d_min: int, d_max: int,
                      j2_min: float, j2_max: float
                      ) -> tuple[list[Candidate], list[DiscoveryError]]:
    candidates: list[Candidate] = []
    errors: list[DiscoveryError] = []
    if not spec.root.is_dir():
        return candidates, [DiscoveryError(spec.name, str(spec.root),
                                           "source root is missing")]

    stages = sorted(path for path in spec.root.rglob("h_*")
                    if path.is_dir() and path.name in H_ZERO_NAMES)
    for stage in stages:
        replica_match = REPLICA_RE.search(str(stage))
        replica = int(replica_match.group(1)) if replica_match else None
        if spec.replica_filter is not None and replica != spec.replica_filter:
            continue
        orientation_match = ORIENTATION_RE.search(str(stage))
        orientation = int(orientation_match.group(1)) if orientation_match else None
        hyperparams = stage / "hyperparams.yaml"
        observations = _base_observations(stage)
        if not observations:
            errors.append(DiscoveryError(spec.name, str(stage),
                                         "h=0 directory has no base observation"))
            continue
        if not hyperparams.is_file():
            errors.append(DiscoveryError(spec.name, str(stage),
                                         "h=0 directory has no hyperparams.yaml"))
            continue
        try:
            params = _load_mapping(hyperparams)
            field = float(params["vbc_field"])
            couplings = _float_list(params["nn_group_couplings"])
            if not _same_float(field, 0.0):
                raise ValueError(f"folder looks like h=0 but vbc_field={field}")
            if len(couplings) != 3 or any(not _same_float(x, 1.0) for x in couplings):
                raise ValueError(
                    f"not the original Hamiltonian: nn_group_couplings={couplings}"
                )
            seed_branch = str(params.get("vbc_branch", "unknown"))
        except (KeyError, OSError, TypeError, ValueError) as exc:
            errors.append(DiscoveryError(spec.name, str(stage), str(exc)))
            continue

        # If a stage contains multiple completed chi evaluations, keep the
        # highest chi for which the matching best tensor exists; if none has a
        # tensor, retain the highest-chi observation so the omission is audited.
        parsed: list[tuple[Row, Path, Path]] = []
        for observation in observations:
            try:
                row = parse_observation(observation)
                if not _in_scope(row, d_min, d_max, j2_min, j2_max):
                    continue
                tensor = stage / f"sweep_D{row.D}_chi{row.chi}_best.pt"
                parsed.append((row, observation, tensor))
            except (OSError, TypeError, ValueError) as exc:
                errors.append(DiscoveryError(spec.name, str(observation), str(exc)))
        if not parsed:
            continue
        with_tensor = [item for item in parsed if item[2].is_file()]
        row, observation, tensor = max(with_tensor or parsed,
                                       key=lambda item: item[0].chi)
        candidates.append(Candidate(
            source=spec.name, seed_branch=seed_branch,
            replica=replica, orientation=orientation,
            observation_path=str(observation.resolve()),
            tensor_path=str(tensor.resolve()) if tensor.is_file() else "",
            hyperparams_path=str(hyperparams.resolve()),
            J2=row.J2, D=row.D, chi=row.chi,
            energy=row.energy_per_site, delta=row.delta,
            middle_fraction=row.middle_fraction, eta=_eta(row),
        ))
    return candidates, errors


def original_baselines(candidates: Iterable[Candidate]) -> dict[tuple[float, int], Candidate]:
    baselines: dict[tuple[float, int], Candidate] = {}
    for candidate in candidates:
        if candidate.source != "original_2c3":
            continue
        key = (round(candidate.J2, 10), candidate.D)
        previous = baselines.get(key)
        if previous is None or candidate.chi > previous.chi:
            baselines[key] = candidate
    return baselines


def select_benchmarks(candidates: Iterable[Candidate], benchmark_j2: float,
                      benchmark_D: int) -> tuple[Candidate, Candidate]:
    subset = [candidate for candidate in candidates
              if candidate.source == "kuma_replica1"
              and _same_float(candidate.J2, benchmark_j2)
              and candidate.D == benchmark_D]
    dimer = [candidate for candidate in subset
             if candidate.seed_branch == "dimer-plaquette"]
    plaquette = [candidate for candidate in subset
                 if candidate.seed_branch == "plaquette"]
    if len(dimer) != 1 or len(plaquette) != 1:
        raise RuntimeError(
            "purity calibration requires exactly one Kuma replica-1 h=0 "
            f"dimer-plaquette and plaquette tensor at J2={benchmark_j2:g}, "
            f"D={benchmark_D}; found {len(dimer)} and {len(plaquette)}"
        )
    if not dimer[0].tensor_path or not plaquette[0].tensor_path:
        raise RuntimeError("a Kuma purity-benchmark tensor is missing")
    if dimer[0].eta <= plaquette[0].eta:
        raise RuntimeError("Kuma purity benchmarks have inconsistent eta ordering")
    return dimer[0], plaquette[0]


def evaluate(candidates: list[Candidate], baselines: dict[tuple[float, int], Candidate],
             dimer_benchmark: Candidate, plaquette_benchmark: Candidate,
             energy_tolerance: float, delta_relative_tolerance: float,
             purity_tolerance: float) -> None:
    dimer_cut = dimer_benchmark.eta - purity_tolerance
    plaquette_cut = plaquette_benchmark.eta + purity_tolerance
    if plaquette_cut >= dimer_cut:
        raise ValueError("purity thresholds overlap; lower --purity-tolerance")

    for candidate in candidates:
        reasons: list[str] = []
        if not candidate.tensor_path or not Path(candidate.tensor_path).is_file():
            reasons.append("missing optimized best tensor")
        baseline = baselines.get((round(candidate.J2, 10), candidate.D))
        if baseline is None:
            reasons.append("missing original 2c3 baseline at same (J2,D)")
        else:
            candidate.original_energy = baseline.energy
            candidate.energy_difference = candidate.energy - baseline.energy
            candidate.original_delta = baseline.delta
            if abs(candidate.energy_difference) > energy_tolerance:
                reasons.append(
                    f"|dE|={abs(candidate.energy_difference):.6g}>{energy_tolerance:.6g}"
                )
            if baseline.delta <= 0.0:
                reasons.append("original Delta is nonpositive")
            else:
                candidate.relative_delta_difference = (
                    abs(candidate.delta - baseline.delta) / baseline.delta
                )
                if candidate.relative_delta_difference > delta_relative_tolerance:
                    reasons.append(
                        "relative Delta difference="
                        f"{candidate.relative_delta_difference:.3%}>"
                        f"{delta_relative_tolerance:.3%}"
                    )

        if not math.isfinite(candidate.eta):
            reasons.append("texture eta is nonfinite")
        elif candidate.eta >= dimer_cut:
            candidate.selected_texture = "dimer-plaquette"
        elif candidate.eta <= plaquette_cut:
            candidate.selected_texture = "plaquette"
        else:
            reasons.append(
                f"texture not pure enough: eta={candidate.eta:.6g}, "
                f"need eta<={plaquette_cut:.6g} or eta>={dimer_cut:.6g}"
            )

        candidate.accepted = not reasons
        candidate.rejection_reasons = "; ".join(reasons)


def _j2_tag(value: float) -> str:
    return f"J2_{value:.10g}".replace(".", "p")


def _safe_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "unknown"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_accepted(candidates: list[Candidate], output: Path) -> None:
    accepted = [candidate for candidate in candidates if candidate.accepted]
    for index, candidate in enumerate(accepted, start=1):
        identity = [candidate.source, candidate.seed_branch]
        if candidate.replica is not None:
            identity.append(f"replica_{candidate.replica}")
        if candidate.orientation is not None:
            identity.append(f"orientation_{candidate.orientation}")
        identity.append(f"chi_{candidate.chi}")
        folder_name = f"{index:03d}__" + "__".join(
            _safe_component(part) for part in identity
        )
        destination = (output / "tensors" / _j2_tag(candidate.J2)
                       / f"D_{candidate.D}" / candidate.selected_texture
                       / folder_name)
        destination.mkdir(parents=True, exist_ok=False)
        source_tensor = Path(candidate.tensor_path)
        copied_tensor = destination / "tensor_best.pt"
        shutil.copy2(source_tensor, copied_tensor)
        shutil.copy2(candidate.observation_path, destination / "observation.txt")
        if candidate.hyperparams_path:
            shutil.copy2(candidate.hyperparams_path, destination / "hyperparams.yaml")
        metadata = asdict(candidate)
        metadata["source_tensor_sha256"] = _sha256(source_tensor)
        metadata["copied_tensor_sha256"] = _sha256(copied_tensor)
        (destination / "selection_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        candidate.copied_tensor = str(copied_tensor.resolve())


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def unique_output(parent: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = parent / f"selection_{stamp}"
    suffix = 1
    while candidate.exists():
        candidate = parent / f"selection_{stamp}_{suffix}"
        suffix += 1
    return candidate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-root", type=Path, default=DEFAULT_ORIGINAL)
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--small-h-root", type=Path, default=DEFAULT_SMALL_H)
    parser.add_argument("--izar-replica2-root", type=Path,
                        default=DEFAULT_IZAR_REPLICA2)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--D-min", type=int, default=6)
    parser.add_argument("--D-max", type=int, default=10)
    parser.add_argument("--J2-min", type=float, default=0.28)
    parser.add_argument("--J2-max", type=float, default=0.33)
    parser.add_argument("--energy-tolerance", type=float, default=2.0e-4)
    parser.add_argument("--delta-relative-tolerance", type=float, default=0.15)
    parser.add_argument(
        "--purity-tolerance", type=float, default=0.01,
        help="absolute eta allowance for 'comparable' to the two D10 benchmarks",
    )
    parser.add_argument("--benchmark-J2", type=float, default=0.30)
    parser.add_argument("--benchmark-D", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true",
                        help="evaluate and print counts without copying/writing")
    args = parser.parse_args()

    if args.D_min > args.D_max or args.J2_min > args.J2_max:
        parser.error("minimum scope must not exceed maximum scope")
    if min(args.energy_tolerance, args.delta_relative_tolerance,
           args.purity_tolerance) < 0.0:
        parser.error("selection tolerances must be nonnegative")

    all_candidates: list[Candidate] = []
    errors: list[DiscoveryError] = []
    originals, original_errors = discover_original(
        args.original_root, args.D_min, args.D_max, args.J2_min, args.J2_max
    )
    all_candidates.extend(originals)
    errors.extend(original_errors)

    sources = (
        SourceSpec("izar_replica1",
                   args.archive_root / "Results_Izar_replica1", 1),
        SourceSpec("kuma_replica1",
                   args.archive_root / "Results_Kuma_replica1", 1),
        SourceSpec("izar_replica2", args.izar_replica2_root, 2),
        SourceSpec("izar_replica1_small_h", args.small_h_root, 1),
    )
    for spec in sources:
        found, found_errors = discover_campaign(
            spec, args.D_min, args.D_max, args.J2_min, args.J2_max
        )
        all_candidates.extend(found)
        errors.extend(found_errors)

    baselines = original_baselines(all_candidates)
    expected_baselines = {
        (round(candidate.J2, 10), candidate.D)
        for candidate in all_candidates
        if args.D_min <= candidate.D <= args.D_max
    }
    missing_baselines = sorted(expected_baselines - set(baselines))
    if missing_baselines:
        print("WARNING: missing original baselines for: " + ", ".join(
            f"(J2={j2:g},D={D})" for j2, D in missing_baselines
        ))

    dimer_benchmark, plaquette_benchmark = select_benchmarks(
        all_candidates, args.benchmark_J2, args.benchmark_D
    )
    evaluate(
        all_candidates, baselines, dimer_benchmark, plaquette_benchmark,
        args.energy_tolerance, args.delta_relative_tolerance,
        args.purity_tolerance,
    )
    all_candidates.sort(key=lambda item: (
        item.J2, item.D, item.selected_texture, item.source,
        item.seed_branch, item.replica or 0, item.orientation or 0,
    ))
    accepted = [candidate for candidate in all_candidates if candidate.accepted]

    print(
        "Purity calibration: "
        f"dimer eta={dimer_benchmark.eta:.9f}, "
        f"plaquette eta={plaquette_benchmark.eta:.9f}, "
        f"comparable allowance={args.purity_tolerance:g}"
    )
    print(
        f"Discovered {len(all_candidates)} in-scope h=0 candidates; "
        f"accepted {len(accepted)}; rejected {len(all_candidates)-len(accepted)}; "
        f"discovery issues {len(errors)}"
    )
    for texture in ("dimer-plaquette", "plaquette"):
        count = sum(candidate.selected_texture == texture and candidate.accepted
                    for candidate in all_candidates)
        print(f"  accepted {texture}: {count}")

    if args.dry_run:
        print("Dry run: no tensors or reports written")
        return 0

    output = unique_output(args.output_root)
    output.mkdir(parents=True, exist_ok=False)
    copy_accepted(all_candidates, output)
    candidate_fields = list(Candidate.__dataclass_fields__)
    _write_csv(
        output / "all_candidates.csv",
        [asdict(candidate) for candidate in all_candidates],
        candidate_fields,
    )
    _write_csv(
        output / "accepted_tensors.csv",
        [asdict(candidate) for candidate in all_candidates if candidate.accepted],
        candidate_fields,
    )
    _write_csv(
        output / "discovery_errors.csv",
        [asdict(error) for error in errors],
        list(DiscoveryError.__dataclass_fields__),
    )
    config = {
        "scope": {"D_min": args.D_min, "D_max": args.D_max,
                  "J2_min": args.J2_min, "J2_max": args.J2_max},
        "criteria": {
            "absolute_energy_difference_max": args.energy_tolerance,
            "relative_delta_difference_max": args.delta_relative_tolerance,
            "eta_definition": "2*(C2-C1)/(C3-C1)-1, with C1<=C2<=C3",
            "purity_tolerance": args.purity_tolerance,
            "dimer_eta_min": dimer_benchmark.eta - args.purity_tolerance,
            "plaquette_eta_max": plaquette_benchmark.eta + args.purity_tolerance,
        },
        "benchmarks": {
            "J2": args.benchmark_J2,
            "D": args.benchmark_D,
            "dimer": asdict(dimer_benchmark),
            "plaquette": asdict(plaquette_benchmark),
        },
        "sources": [{"name": spec.name, "root": str(spec.root.resolve()),
                     "replica_filter": spec.replica_filter} for spec in sources],
        "original_root": str(args.original_root.resolve()),
    }
    (output / "selection_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Tensor collection: {output}")
    print(f"Accepted manifest: {output / 'accepted_tensors.csv'}")
    print(f"Full audit:        {output / 'all_candidates.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
