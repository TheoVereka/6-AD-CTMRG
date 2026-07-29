"""Shared naming and manifest helpers for the CPU-cluster bundle."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any


ANSATZ_DIRECTORY = "2tensor_twoC3"
MANIFEST_JSON = "checkpoint_manifest.json"
MANIFEST_TSV = "checkpoint_manifest.tsv"
CHECKPOINT_DIRECTORY = "checkpoints"
RESULT_DIRECTORY = "results"
J2_DIRECTORY_PATTERN = re.compile(r"^J2_(\d+(?:p\d+)?)$")
CHECKPOINT_NAME_PATTERN = re.compile(
    r"^tensor_best__(J2_\d+(?:p\d+)?)__D_(\d+)\.pt$"
)
RESULT_NAME_PATTERN = re.compile(
    r"^correlation_length__(J2_\d+(?:p\d+)?)__D_(\d+)\.json$"
)


def parse_j2_directory(name: str) -> float:
    match = J2_DIRECTORY_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(f"Invalid J2 directory name: {name!r}")
    return float(match.group(1).replace("p", "."))


def staged_checkpoint_name(j2_directory: str, D_bond: int) -> str:
    parse_j2_directory(j2_directory)
    if D_bond < 1:
        raise ValueError("D must be positive")
    return f"tensor_best__{j2_directory}__D_{D_bond}.pt"


def result_name(j2_directory: str, D_bond: int) -> str:
    parse_j2_directory(j2_directory)
    if D_bond < 1:
        raise ValueError("D must be positive")
    return f"correlation_length__{j2_directory}__D_{D_bond}.json"


def load_manifest(bundle_root: Path) -> dict[str, Any]:
    path = bundle_root / MANIFEST_JSON
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != 1:
        raise ValueError(f"Unsupported manifest schema in {path}")
    if not isinstance(manifest.get("items"), list):
        raise ValueError(f"Manifest has no item list: {path}")
    return manifest


def manifest_index(
    manifest: dict[str, Any],
) -> dict[tuple[str, int], dict[str, Any]]:
    index: dict[tuple[str, int], dict[str, Any]] = {}
    for item in manifest["items"]:
        token = str(item["j2_directory"])
        D_bond = int(item["D"])
        expected_name = staged_checkpoint_name(token, D_bond)
        if item["staged_filename"] != expected_name:
            raise ValueError(
                f"Manifest filename mismatch for ({token}, D={D_bond})"
            )
        key = (token, D_bond)
        if key in index:
            raise ValueError(f"Duplicate manifest item: {key}")
        index[key] = item
    return index


def validate_result_payload(
    payload: dict[str, Any],
    *,
    j2: float,
    D_bond: int,
) -> None:
    if int(payload["D_bond"]) != D_bond:
        raise ValueError("D_bond does not match the requested job")
    recorded_j2 = float(payload["calculation_hyperparameters"]["J2"])
    if not math.isclose(recorded_j2, j2, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("J2 does not match the requested job")
    eigenvalues = payload["eigenvalues"]
    if not isinstance(eigenvalues, list) or len(eigenvalues) < 2:
        raise ValueError("Result does not contain two eigenvalues")
    for value in eigenvalues[:2]:
        real = float(value["real"])
        imag = float(value["imag"])
        if not math.isfinite(real) or not math.isfinite(imag):
            raise ValueError("Result contains a non-finite eigenvalue")
    if "correlation_length" not in payload:
        raise ValueError("Result has no correlation_length field")


def is_valid_result(path: Path, *, j2: float, D_bond: int) -> bool:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        validate_result_payload(payload, j2=j2, D_bond=D_bond)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False
    return True
