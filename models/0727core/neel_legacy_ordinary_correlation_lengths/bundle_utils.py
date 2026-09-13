"""Shared naming and manifest helpers for the CPU-cluster bundle."""

from __future__ import annotations

import json
import math
import re
from statistics import median
from pathlib import Path
from typing import Any


C3_COMPATIBLE_ANSATZ_DIRECTORIES = (
    "neel_free_param",
    "neel_symmetrized",
    "1tensor_C6Ypi",
    "1tensor_C3Vypi",
    "2tensor_twoC3",
)
LEGACY_TWOC3_ANSATZ_DIRECTORY = "2tensor_twoC3"
MANIFEST_JSON = "checkpoint_manifest.json"
MANIFEST_TSV = "checkpoint_manifest.tsv"
CHECKPOINT_DIRECTORY = "checkpoints"
RESULT_DIRECTORY = "results_three_env_ordinary_v5"
ORDINARY_DIRECTIONS = ("env2", "env1_ab_env3_ba", "env3_ab_env1_ba")
J2_DIRECTORY_PATTERN = re.compile(r"^J2_(\d+(?:p\d+)?)$")
CHECKPOINT_NAME_PATTERN = re.compile(
    r"^tensor_best__(?:(.+)__)?(J2_\d+(?:p\d+)?)__D_(\d+)\.pt$"
)
RESULT_NAME_PATTERN = re.compile(
    r"^correlation_length__(?:(.+)__)?(J2_\d+(?:p\d+)?)__D_(\d+)\.json$"
)
PARTIAL_RESULT_NAME_PATTERN = re.compile(
    r"^correlation_length__(?:(.+)__)?(J2_\d+(?:p\d+)?)__D_(\d+)"
    r"__(env2|env1_ab_env3_ba|env3_ab_env1_ba)\.json$"
)


def validate_ansatz_directory(name: str) -> str:
    if name not in C3_COMPATIBLE_ANSATZ_DIRECTORIES:
        raise ValueError(f"Not a C3-CTM-compatible ansatz directory: {name!r}")
    return name


def parse_j2_directory(name: str) -> float:
    match = J2_DIRECTORY_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(f"Invalid J2 directory name: {name!r}")
    return float(match.group(1).replace("p", "."))


def staged_checkpoint_name(
    ansatz_directory: str, j2_directory: str, D_bond: int
) -> str:
    validate_ansatz_directory(ansatz_directory)
    parse_j2_directory(j2_directory)
    if D_bond < 1:
        raise ValueError("D must be positive")
    if ansatz_directory == LEGACY_TWOC3_ANSATZ_DIRECTORY:
        return f"tensor_best__{j2_directory}__D_{D_bond}.pt"
    return (
        f"tensor_best__{ansatz_directory}__{j2_directory}__D_{D_bond}.pt"
    )


def result_name(
    ansatz_directory: str, j2_directory: str, D_bond: int
) -> str:
    validate_ansatz_directory(ansatz_directory)
    parse_j2_directory(j2_directory)
    if D_bond < 1:
        raise ValueError("D must be positive")
    if ansatz_directory == LEGACY_TWOC3_ANSATZ_DIRECTORY:
        return f"correlation_length__{j2_directory}__D_{D_bond}.json"
    return (
        f"correlation_length__{ansatz_directory}__{j2_directory}"
        f"__D_{D_bond}.json"
    )


def partial_result_name(
    ansatz_directory: str,
    j2_directory: str,
    D_bond: int,
    direction: str,
) -> str:
    if direction not in ORDINARY_DIRECTIONS:
        raise ValueError(f"Invalid ordinary direction: {direction!r}")
    return result_name(ansatz_directory, j2_directory, D_bond).removesuffix(
        ".json"
    ) + f"__{direction}.json"


def parse_partial_result_name(name: str) -> tuple[str, str, int, str]:
    match = PARTIAL_RESULT_NAME_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(f"Invalid partial result filename: {name!r}")
    ansatz, j2_directory, D_text, direction = match.groups()
    if ansatz is None:
        ansatz = LEGACY_TWOC3_ANSATZ_DIRECTORY
    validate_ansatz_directory(ansatz)
    parse_j2_directory(j2_directory)
    return ansatz, j2_directory, int(D_text), direction


def parse_result_name(name: str) -> tuple[str, str, int]:
    match = RESULT_NAME_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(f"Invalid result filename: {name!r}")
    ansatz, j2_directory, D_text = match.groups()
    if ansatz is None:
        ansatz = LEGACY_TWOC3_ANSATZ_DIRECTORY
    validate_ansatz_directory(ansatz)
    parse_j2_directory(j2_directory)
    return ansatz, j2_directory, int(D_text)


def load_manifest(bundle_root: Path) -> dict[str, Any]:
    path = bundle_root / MANIFEST_JSON
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") not in (1, 2):
        raise ValueError(f"Unsupported manifest schema in {path}")
    if not isinstance(manifest.get("items"), list):
        raise ValueError(f"Manifest has no item list: {path}")
    if manifest.get("schema_version") == 1:
        if manifest.get("ansatz_directory") != LEGACY_TWOC3_ANSATZ_DIRECTORY:
            raise ValueError(f"Unsupported legacy manifest ansatz: {path}")
        for item in manifest["items"]:
            item.setdefault(
                "ansatz_directory", LEGACY_TWOC3_ANSATZ_DIRECTORY
            )
    return manifest


def manifest_index(
    manifest: dict[str, Any],
) -> dict[tuple[str, str, int], dict[str, Any]]:
    index: dict[tuple[str, str, int], dict[str, Any]] = {}
    for item in manifest["items"]:
        token = str(item["j2_directory"])
        ansatz = validate_ansatz_directory(str(item["ansatz_directory"]))
        D_bond = int(item["D"])
        expected_name = staged_checkpoint_name(ansatz, token, D_bond)
        if item["staged_filename"] != expected_name:
            raise ValueError(
                f"Manifest filename mismatch for ({token}, D={D_bond})"
            )
        key = (ansatz, token, D_bond)
        if key in index:
            raise ValueError(f"Duplicate manifest item: {key}")
        index[key] = item
    return index


def accepted_directions(payload: dict[str, Any]) -> tuple[str, ...]:
    directions = tuple(payload.get("accepted_directions", ORDINARY_DIRECTIONS))
    if len(directions) not in (2, 3) or len(set(directions)) != len(directions):
        raise ValueError("Expected two or three distinct accepted directions")
    if not set(directions) <= set(ORDINARY_DIRECTIONS):
        raise ValueError("Unknown accepted direction")
    return directions


def recover_complete_payload(payload: dict[str, Any], expected_hash: str) -> dict[str, Any]:
    directions = accepted_directions(payload)
    values = []
    for direction in directions:
        spectrum = payload["spectra"][direction]
        magnitudes = sorted((math.hypot(float(v["real"]), float(v["imag"])) for v in spectrum["eigenvalues"][:2]), reverse=True)
        value = math.log(magnitudes[0] / magnitudes[1])
        if not math.isfinite(value):
            raise ValueError("Non-finite inverse xi")
        values.append(value)
    scale = max(map(abs, values))
    spread = (max(values) - min(values)) / scale if scale else 0.0
    if len(values) != 3 or spread >= 1e-4:
        raise ValueError(f"Stale complete result relative spread={spread:.6g} is not below 1e-4")
    result = dict(payload)
    result["accepted_directions"] = list(directions)
    result["direction_count"] = len(directions)
    result["import_recovery"] = {
        "reason": "relative_spread_below_1e-4", "relative_spread": spread,
        "relative_tolerance": 1e-4, "expected_checkpoint_sha256": expected_hash,
        "excluded_directions": [],
    }
    return result


def recover_split_payloads(payloads: dict[str, dict[str, Any]], expected_hash: str) -> dict[str, Any]:
    """Retain a coherent pair, or numerically equivalent mixed directions.

    Hashes are never rewritten. Recovery is an explicit import exception and
    does not satisfy the strict checkpoint check used by cluster submission.
    Relative spread is (max-min)/max(abs(values)), strictly below 1e-4.
    """
    for direction, payload in payloads.items():
        validate_partial_result_payload(
            payload, j2=float(payload["calculation_hyperparameters"]["J2"]),
            D_bond=int(payload["D_bond"]), ansatz_directory=payload["ansatz_directory"],
            direction=direction,
        )
    groups: dict[tuple, dict] = {}
    for direction, payload in payloads.items():
        provenance = payload.get("cluster_bundle_provenance", {})
        checkpoint_hash = provenance.get("checkpoint_sha256")
        if not checkpoint_hash:
            raise ValueError(f"Missing checkpoint hash for {direction}")
        key = (checkpoint_hash, payload["chi"], payload.get("seed"))
        groups.setdefault(key, {})[direction] = payload
    values = [float(p["spectra"][d]["inverse_correlation_length"]) for d, p in payloads.items()]
    scale = max(map(abs, values), default=0.0)
    spread = (max(values) - min(values)) / scale if scale else 0.0
    hashes = {key[0] for key in groups}
    if len(groups) == 1 and hashes == {expected_hash}:
        selected = payloads
        reason = None
    elif len(payloads) == 3 and spread < 1e-4:
        selected = payloads
        reason = "relative_spread_below_1e-4"
    elif len(groups) > 1 and any(len(group) >= 2 for group in groups.values()):
        selected = max(groups.values(), key=len)
        reason = "coherent_pair_from_mixed_results"
    else:
        raise ValueError(f"Checkpoint mismatch without a recoverable pair (relative spread={spread:.6g})")
    merged = merge_partial_payloads(selected, allow_mixed=reason == "relative_spread_below_1e-4")
    if reason:
        merged["import_recovery"] = {
            "reason": reason, "expected_checkpoint_sha256": expected_hash,
            "relative_spread": spread, "relative_tolerance": 1e-4,
            "excluded_directions": [d for d in payloads if d not in selected],
            "observed_direction_values": {d: p["spectra"][d]["inverse_correlation_length"] for d, p in payloads.items()},
            "observed_provenance": {d: p.get("cluster_bundle_provenance") for d, p in payloads.items()},
        }
    return merged


def validate_result_payload(
    payload: dict[str, Any],
    *,
    j2: float,
    D_bond: int,
    ansatz_directory: str = LEGACY_TWOC3_ANSATZ_DIRECTORY,
) -> None:
    validate_ansatz_directory(ansatz_directory)
    legacy_twoc3 = (
        ansatz_directory == LEGACY_TWOC3_ANSATZ_DIRECTORY
        and payload.get("schema")
        == "twoc3_three_ordinary_correlation_lengths"
        and payload.get("schema_version") == 5
        and payload.get("transfer_network_schema")
        == "three_geometric_straight_rows_ordinary_v5"
    )
    current = (
        payload.get("schema") == "c3ctm_three_ordinary_correlation_lengths"
        and payload.get("schema_version") == 6
        and payload.get("transfer_network_schema")
        == "three_geometric_straight_rows_ordinary_v6"
        and payload.get("ansatz_directory") == ansatz_directory
    )
    if not (legacy_twoc3 or current):
        raise ValueError(
            "Result is neither compatible legacy two-C3 ordinary-v5 nor "
            "current C3-CTM ordinary-v6"
        )
    if int(payload["D_bond"]) != D_bond:
        raise ValueError("D_bond does not match the requested job")
    recorded_j2 = float(payload["calculation_hyperparameters"]["J2"])
    if not math.isclose(recorded_j2, j2, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("J2 does not match the requested job")
    spectra = payload["spectra"]
    required = accepted_directions(payload)
    inverse_values: list[float] = []
    for key in required:
        spectrum = spectra[key]
        eigenvalues = spectrum["eigenvalues"]
        if not isinstance(eigenvalues, list) or len(eigenvalues) < 2:
            raise ValueError(f"{key} does not contain two eigenvalues")
        magnitudes: list[float] = []
        for value in eigenvalues[:2]:
            real = float(value["real"])
            imag = float(value["imag"])
            if not math.isfinite(real) or not math.isfinite(imag):
                raise ValueError(f"{key} contains a non-finite eigenvalue")
            magnitudes.append(math.hypot(real, imag))
        magnitudes.sort(reverse=True)
        if magnitudes[1] <= 0.0:
            raise ValueError(f"{key} has a zero subleading eigenvalue")
        inverse_xi = math.log(magnitudes[0] / magnitudes[1])
        recorded = float(spectrum["inverse_correlation_length"])
        if not math.isclose(
            recorded, inverse_xi, rel_tol=1.0e-11, abs_tol=1.0e-13
        ):
            raise ValueError(f"{key} inverse xi was not computed from lambdas")
        inverse_values.append(inverse_xi)
    summary = payload["inverse_correlation_length"]
    ordered = sorted(inverse_values)
    for field, expected in zip(
        ("lower", "center", "upper"), (min(ordered), median(ordered), max(ordered)), strict=True
    ):
        if not math.isclose(
            float(summary[field]), expected, rel_tol=1.0e-11, abs_tol=1.0e-13
        ):
            raise ValueError(f"Invalid inverse-xi summary field: {field}")
    if "correlation_length" not in payload:
        raise ValueError("Result has no correlation_length field")
    ctm = payload["ctm"]
    max_steps = int(ctm["max_steps"])
    if int(ctm["steps_ab"]) > max_steps:
        raise ValueError("(a,b) CTMRG did not converge within its step budget")
    if int(ctm["steps_ba"]) > max_steps:
        raise ValueError("(b,a) CTMRG did not converge within its step budget")


def is_valid_result(
    path: Path,
    *,
    j2: float,
    D_bond: int,
    ansatz_directory: str = LEGACY_TWOC3_ANSATZ_DIRECTORY,
) -> bool:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        validate_result_payload(
            payload,
            j2=j2,
            D_bond=D_bond,
            ansatz_directory=ansatz_directory,
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False
    return True


def is_completed_ordinary_result(
    path: Path,
    *,
    j2: float,
    D_bond: int,
    ansatz_directory: str = LEGACY_TWOC3_ANSATZ_DIRECTORY,
    checkpoint_sha256: str | None = None,
) -> bool:
    """Return whether an atomic ordinary output exists for submission dedup.

    CTMRG convergence diagnostics are deliberately not enforced here.  When
    ``checkpoint_sha256`` is supplied, however, the result is complete only if
    its provenance proves that it was calculated from exactly that tensor.
    """

    try:
        validate_ansatz_directory(ansatz_directory)
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        legacy_twoc3 = (
            ansatz_directory == LEGACY_TWOC3_ANSATZ_DIRECTORY
            and payload.get("schema")
            == "twoc3_three_ordinary_correlation_lengths"
            and payload.get("schema_version") == 5
            and payload.get("transfer_network_schema")
            == "three_geometric_straight_rows_ordinary_v5"
        )
        current = (
            payload.get("schema")
            == "c3ctm_three_ordinary_correlation_lengths"
            and payload.get("schema_version") == 6
            and payload.get("transfer_network_schema")
            == "three_geometric_straight_rows_ordinary_v6"
            and payload.get("ansatz_directory") == ansatz_directory
        )
        if not (legacy_twoc3 or current):
            return False
        if int(payload["D_bond"]) != D_bond:
            return False
        recorded_j2 = float(payload["calculation_hyperparameters"]["J2"])
        if not math.isclose(recorded_j2, j2, rel_tol=0.0, abs_tol=1.0e-12):
            return False
        if checkpoint_sha256 is not None:
            provenance = payload.get("cluster_bundle_provenance")
            if not isinstance(provenance, dict):
                return False
            if provenance.get("checkpoint_sha256") != checkpoint_sha256:
                return False
        spectra = payload["spectra"]
        for key in accepted_directions(payload):
            eigenvalues = spectra[key]["eigenvalues"]
            if not isinstance(eigenvalues, list) or len(eigenvalues) < 2:
                return False
            for value in eigenvalues[:2]:
                if not math.isfinite(float(value["real"])):
                    return False
                if not math.isfinite(float(value["imag"])):
                    return False
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False
    return True


def validate_partial_result_payload(
    payload: dict[str, Any],
    *,
    j2: float,
    D_bond: int,
    ansatz_directory: str,
    direction: str,
    checkpoint_sha256: str | None = None,
) -> None:
    validate_ansatz_directory(ansatz_directory)
    if direction not in ORDINARY_DIRECTIONS:
        raise ValueError(f"Invalid ordinary direction: {direction!r}")
    if payload.get("schema") != "c3ctm_single_ordinary_correlation_length_direction":
        raise ValueError("Not a split ordinary-direction result")
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported split ordinary-direction schema")
    if payload.get("transfer_network_schema") != "three_geometric_straight_rows_ordinary_v6":
        raise ValueError("Unexpected transfer-network schema")
    if payload.get("ansatz_directory") != ansatz_directory:
        raise ValueError("Ansatz does not match the requested split job")
    if payload.get("direction") != direction:
        raise ValueError("Direction does not match the requested split job")
    if int(payload["D_bond"]) != D_bond:
        raise ValueError("D does not match the requested split job")
    recorded_j2 = float(payload["calculation_hyperparameters"]["J2"])
    if not math.isclose(recorded_j2, j2, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("J2 does not match the requested split job")
    spectrum = payload["spectra"][direction]
    eigenvalues = spectrum["eigenvalues"]
    if not isinstance(eigenvalues, list) or len(eigenvalues) < 2:
        raise ValueError("Split result does not contain two eigenvalues")
    magnitudes = []
    for value in eigenvalues[:2]:
        real, imag = float(value["real"]), float(value["imag"])
        if not math.isfinite(real) or not math.isfinite(imag):
            raise ValueError("Split result contains a non-finite eigenvalue")
        magnitudes.append(math.hypot(real, imag))
    magnitudes.sort(reverse=True)
    expected = math.log(magnitudes[0] / magnitudes[1])
    if not math.isclose(
        float(spectrum["inverse_correlation_length"]),
        expected,
        rel_tol=1.0e-11,
        abs_tol=1.0e-13,
    ):
        raise ValueError("Split inverse xi was not computed from eigenvalues")
    if checkpoint_sha256 is not None:
        provenance = payload.get("cluster_bundle_provenance")
        if not isinstance(provenance, dict) or provenance.get(
            "checkpoint_sha256"
        ) != checkpoint_sha256:
            raise ValueError("Split result checkpoint hash does not match")


def is_completed_partial_result(
    path: Path,
    *,
    j2: float,
    D_bond: int,
    ansatz_directory: str,
    direction: str,
    checkpoint_sha256: str | None = None,
) -> bool:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        validate_partial_result_payload(
            payload,
            j2=j2,
            D_bond=D_bond,
            ansatz_directory=ansatz_directory,
            direction=direction,
            checkpoint_sha256=checkpoint_sha256,
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False
    return True


def merge_partial_payloads(
    payloads: dict[str, dict[str, Any]], *, allow_mixed: bool = False,
) -> dict[str, Any]:
    directions = tuple(d for d in ORDINARY_DIRECTIONS if d in payloads)
    if len(directions) not in (2, 3) or set(directions) != set(payloads):
        raise ValueError("Two or three ordinary directions are required")
    reference = payloads[directions[0]]
    for direction, payload in payloads.items():
        validate_partial_result_payload(
            payload,
            j2=float(reference["calculation_hyperparameters"]["J2"]),
            D_bond=int(reference["D_bond"]),
            ansatz_directory=str(reference["ansatz_directory"]),
            direction=direction,
        )
        if not allow_mixed and int(payload["chi"]) != int(reference["chi"]):
            raise ValueError("Split directions disagree on chi")
    spectra = {
        direction: payloads[direction]["spectra"][direction]
        for direction in directions
    }
    values = {
        direction: float(spectra[direction]["inverse_correlation_length"])
        for direction in directions
    }
    lower, center, upper = min(values.values()), median(values.values()), max(values.values())
    ctm_runs = {direction: payloads[direction]["ctm"] for direction in directions}
    provenance_runs = {
        direction: payloads[direction].get("cluster_bundle_provenance")
        for direction in directions
    }
    merged = dict(reference)
    merged.update(
        {
            "schema": "c3ctm_three_ordinary_correlation_lengths",
            "schema_version": 6,
            "accepted_directions": list(directions),
            "direction_count": len(directions),
            "completed_at_utc": max(
                str(payloads[direction]["completed_at_utc"])
                for direction in directions
            ),
            "seed": {direction: payloads[direction]["seed"] for direction in directions},
            "seed_was_randomized": any(
                bool(payloads[direction].get("seed_was_randomized"))
                for direction in directions
            ),
            "ctm": {
                **reference["ctm"],
                "steps_ab": max(int(value["steps_ab"]) for value in ctm_runs.values()),
                "steps_ba": max(int(value["steps_ba"]) for value in ctm_runs.values()),
                "direction_runs": ctm_runs,
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
            "elapsed_seconds": sum(
                float(payloads[direction].get("elapsed_seconds", 0.0))
                for direction in directions
            ),
            "split_direction_results": provenance_runs,
        }
    )
    merged.pop("direction", None)
    return merged
