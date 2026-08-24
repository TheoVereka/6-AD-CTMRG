#!/usr/bin/env python3
"""Collect missing ordinary-only correlation-length jobs for D345678910 Neel."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
CPU_BUNDLE = HERE.parent / "correlation_length_cpu_cluster"
OLD_NEEL_BUNDLE = HERE.parent / "neel_six_correlation_lengths"
DEFAULT_DATA_ROOT = Path(r"D:\HyraiOn\ENS_Lyon\Internship\2026-EPFL\data")
DEFAULT_LEGACY_ROOT = DEFAULT_DATA_ROOT / "D345678910"
DEFAULT_OLD_RESULTS = OLD_NEEL_BUNDLE / "results"
ANSATZ = "neel_symmetrized"
SOLVER_FILES = (
    "bundle_utils.py",
    "correlation_length.py",
    "core_C3.py",
    "compute_three_ordinary_correlation_lengths.py",
    "run_one_correlation_length.py",
    "correlation_length_job.run",
    "submit_correlation_lengths.sh",
)
ORDINARY_OLD_KEYS = (
    "env2_ordinary",
    "env1_ab_env3_ba_ordinary",
    "env3_ab_env1_ba_ordinary",
)
ORDINARY_KEYS = ("env2", "env1_ab_env3_ba", "env3_ab_env1_ba")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_old_collector():
    sys.path.insert(0, str(OLD_NEEL_BUNDLE))
    spec = importlib.util.spec_from_file_location(
        "old_neel_checkpoint_collector", OLD_NEEL_BUNDLE / "collect_checkpoints.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load the existing Neel checkpoint collector")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def copy_solver_snapshot() -> None:
    for filename in SOLVER_FILES:
        source = CPU_BUNDLE / filename
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = HERE / filename
        if destination.resolve() == source.resolve():
            continue
        temporary = destination.with_name(destination.name + ".copying")
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)


def inverse_from_spectrum(spectrum: dict) -> float:
    values = spectrum["eigenvalues"][:2]
    magnitudes = sorted(
        (math.hypot(float(value["real"]), float(value["imag"])) for value in values),
        reverse=True,
    )
    return math.log(magnitudes[0] / magnitudes[1])


def convert_old_ordinary(
    source: dict,
    *,
    checkpoint: Path,
    checkpoint_hash: str,
) -> dict:
    spectra = {
        new: source["spectra"][old]
        for new, old in zip(ORDINARY_KEYS, ORDINARY_OLD_KEYS, strict=True)
    }
    values = {key: inverse_from_spectrum(value) for key, value in spectra.items()}
    for key, value in values.items():
        spectra[key]["inverse_correlation_length"] = value
        spectra[key]["correlation_length"] = None if value <= 0.0 else 1.0 / value
    lower, center, upper = sorted(values.values())
    ctm = source["ctm"]
    j2 = float(source["J2"])
    return {
        "schema": "c3ctm_three_ordinary_correlation_lengths",
        "schema_version": 6,
        "transfer_network_schema": "three_geometric_straight_rows_ordinary_v6",
        "completed_at_utc": source.get("completed_at_utc", datetime.now(timezone.utc).isoformat()),
        "checkpoint": str(checkpoint),
        "ansatz_directory": ANSATZ,
        "D_bond": int(source["D"]),
        "chi": int(source["chi"]),
        "dtype": source.get("dtype", "torch.float64"),
        "device": source.get("device", "cpu"),
        "seed": source.get("seed"),
        "seed_was_randomized": source.get("seed_was_randomized", True),
        "threads": source.get("threads", 16),
        "ctm": {
            "max_steps": int(ctm["max_steps"]),
            "sv_convergence_tolerance": float(ctm.get("convergence_tolerance", ctm.get("sv_convergence_tolerance", 1e-7))),
            "convergence_mode": ctm.get("convergence_mode", "both"),
            "energy_convergence_threshold": float(ctm.get("energy_convergence_threshold", 2e-8)),
            "identity_init": bool(ctm.get("identity_init", True)),
            "svd_mode": ctm.get("svd_mode", "augmented"),
            "steps_ab": int(ctm["steps_ab"]),
            "steps_ba": int(ctm["steps_ba"]),
            "converged_ab_within_budget": bool(ctm.get("converged_ab_within_budget", True)),
            "converged_ba_within_budget": bool(ctm.get("converged_ba_within_budget", True)),
            "internal_sentinel_max_steps": int(ctm.get("internal_sentinel_max_steps", int(ctm["max_steps"]) + 1)),
        },
        "calculation_hyperparameters": {
            "J1": 1.0,
            "J2": j2,
            "eigenproblem": "ordinary_raw_row_transfer",
            "converted_from_neel_six_schema_v3": True,
        },
        "spectra": spectra,
        "inverse_correlation_length": {
            "definition": "ln(abs(lambda_1/lambda_2))",
            "aggregation": "lower=min, center=median, upper=max",
            "lower": lower,
            "center": center,
            "upper": upper,
            "lower_error": center - lower,
            "upper_error": upper - center,
            "direction_values": values,
        },
        "correlation_length": None if center <= 0.0 else 1.0 / center,
        "elapsed_seconds": float(source.get("elapsed_seconds", 0.0)),
        "cluster_bundle_provenance": {
            "ansatz_directory": ANSATZ,
            "j2": j2,
            "D": int(source["D"]),
            "checkpoint_sha256": checkpoint_hash,
            "converted_from": "neel_six_correlation_lengths",
        },
    }


def old_results_index(root: Path) -> dict[tuple[float, int], tuple[Path, dict]]:
    output = {}
    for path in root.rglob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("schema") != "neel_six_correlation_lengths" or payload.get("schema_version") != 3:
                continue
            if not all(key in payload["spectra"] for key in ORDINARY_OLD_KEYS):
                continue
            key = (round(float(payload["J2"]), 12), int(payload["D"]))
            previous = output.get(key)
            if previous is None or path.stat().st_mtime_ns > previous[0].stat().st_mtime_ns:
                output[key] = (path, payload)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--legacy-root", type=Path, default=DEFAULT_LEGACY_ROOT)
    parser.add_argument("--old-results", type=Path, default=DEFAULT_OLD_RESULTS)
    parser.add_argument("--min-D", type=int, default=4)
    parser.add_argument("--max-D", type=int, default=11)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    collector = load_old_collector()
    selected, warnings = collector._scan_d345(
        args.legacy_root.resolve(),
        data_root=args.data_root.resolve(),
        min_D=args.min_D,
        max_D=args.max_D,
    )
    old_results = old_results_index(args.old_results.resolve())
    if args.dry_run:
        print(f"Discovered {len(selected)} completed legacy Neel checkpoints.")
        return 0
    copy_solver_snapshot()
    sys.path.insert(0, str(HERE))
    from bundle_utils import is_completed_ordinary_result, staged_checkpoint_name

    checkpoint_dir = HERE / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for old in checkpoint_dir.iterdir():
        if old.is_file():
            old.unlink()
    items = []
    selected_items = []
    for (j2, D), candidate in sorted(selected.items()):
        staged_name = staged_checkpoint_name(ANSATZ, f"J2_{j2:g}".replace(".", "p"), D)
        staged = checkpoint_dir / staged_name
        _canonical, symmetry_error = collector._canonical_payload(candidate)
        temporary = staged.with_name(staged.name + ".copying")
        shutil.copy2(candidate.checkpoint, temporary)
        os.replace(temporary, staged)
        staged_hash = sha256(staged)
        destination = candidate.observable.parent / f"correlation_length_D_{D}.json"
        current = is_completed_ordinary_result(
            destination,
            j2=j2,
            D_bond=D,
            ansatz_directory=ANSATZ,
            checkpoint_sha256=staged_hash,
        )
        reason = "current_result_matches_tensor" if current else "missing_correlation"
        old = old_results.get((round(j2, 12), D))
        if not current and old is not None:
            converted = convert_old_ordinary(
                old[1], checkpoint=staged, checkpoint_hash=staged_hash
            )
            atomic_json(destination, converted)
            current = True
            reason = f"converted_existing_ordinary:{old[0]}"
        item = {
            "j2_directory": f"J2_{j2:g}".replace(".", "p"),
            "ansatz_directory": ANSATZ,
            "j2": j2,
            "D": D,
            "staged_filename": staged_name,
            "original_relative_path": candidate.checkpoint.relative_to(args.data_root.resolve()).as_posix(),
            "legacy_run_relative_path": candidate.observable.parent.relative_to(args.legacy_root.resolve()).as_posix(),
            "legacy_observable": candidate.observable.name,
            "legacy_correlation_filename": destination.name,
            "sha256": staged_hash,
            "source_checkpoint_sha256": sha256(candidate.checkpoint),
            "virtual_symmetry_relative_error": symmetry_error,
            "selected_for_rerun": not current,
            "selection_reason": reason,
        }
        items.append(item)
        if not current:
            selected_items.append(item)
        else:
            staged.unlink()
        print(f"J2={j2:g} D={D}: {reason}")

    solver_hashes = {
        filename: sha256(HERE / filename)
        for filename in SOLVER_FILES
        if (HERE / filename).is_file()
    }
    manifest = {
        "schema_version": 2,
        "bundle_kind": "D345678910_neel_legacy_ordinary_only",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_legacy_root": str(args.legacy_root.resolve()),
        "solver_files_sha256": solver_hashes,
        "items": items,
        "warnings": warnings,
    }
    atomic_json(HERE / "checkpoint_manifest.json", manifest)
    with (HERE / "checkpoint_manifest.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("j2_directory", "ansatz_directory", "j2", "D", "staged_filename", "original_relative_path", "sha256"))
        for item in selected_items:
            writer.writerow(tuple(item[key] for key in ("j2_directory", "ansatz_directory", "j2", "D", "staged_filename", "original_relative_path", "sha256")))
    print(f"Staged {len(selected_items)} missing ordinary result(s); materialized {len(items) - len(selected_items)} current result(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
