#!/usr/bin/env python3
"""Compute six Neel row-transfer spectra for one ``(J2,D)`` checkpoint.

The three raw row transfers are:

* env2(a,b) x env2(b,a);
* env1(a,b) x env3(b,a);
* env3(a,b) x env1(b,a).

Each raw transfer is diagonalized twice: once as the corner-metric
generalized pencil and once directly with the right-hand corner map removed.
The output therefore contains exactly six spectra.  ``1/xi`` is always
computed directly as ``log(abs(lambda_1/lambda_2))``.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import itertools
import json
import math
import os
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

import correlation_length as corr
import core_C3 as core
from temporary_crossed_environment_correlation_lengths import (
    CrossedEnvironmentTransferOperator,
    Environment,
    KroneckerCornerWhitenedTransferOperator,
)


HERE = Path(__file__).resolve().parent
DEFAULT_THREADS = int(os.environ.get("SLURM_CPUS_PER_TASK", "16"))
RESULT_DIRECTORY_NAME = "results"
SCHEMA_VERSION = 3
TRANSFER_NETWORK_SCHEMA = "three_geometric_straight_rows_v3"


@dataclasses.dataclass(frozen=True)
class ThreeEnvironments:
    env1: Environment
    env2: Environment
    env3: Environment
    double_layers: tuple[torch.Tensor, ...] | None
    ctm_steps: int


def _as_environment(values: tuple[torch.Tensor, ...]) -> Environment:
    return Environment(
        corner=values[0].detach().contiguous(),
        t1=values[1].detach().contiguous(),
        t2=values[2].detach().contiguous(),
    )


def _physical_y_partner(a: torch.Tensor) -> torch.Tensor:
    d_phys = int(a.shape[-1])
    rotation = torch.zeros(
        d_phys, d_phys, dtype=a.dtype, device=a.device
    )
    for s in range(d_phys):
        rotation[s, d_phys - 1 - s] = (-1.0) ** s
    return torch.einsum("ij,...j->...i", rotation, a).contiguous()


def _symmetry_error(a: torch.Tensor) -> float:
    denominator = max(float(torch.linalg.norm(a)), 1.0e-300)
    return max(
        float(torch.linalg.norm(a - a.permute(*permutation, 3)))
        / denominator
        for permutation in itertools.permutations(range(3))
    )


@torch.no_grad()
def _run_three_environments(
    raw_a: torch.Tensor,
    raw_b: torch.Tensor,
    *,
    chi: int,
    D: int,
    ctm_max_steps: int,
    ctm_conv_tol: float,
    j1: float,
    j2: float,
    identity_init: bool,
    keep_double_layers: bool,
) -> ThreeEnvironments:
    sites, double_layers = corr._build_ctm_layers(raw_a, raw_b)
    energy_proxy, _ = corr._build_lbfgs_energy_proxy(
        sites,
        chi=chi,
        D_bond=D,
        j1=j1,
        j2=j2,
    )
    result = core.CTMRG_from_init_to_stop(
        *double_layers,
        chi,
        D**2,
        # One sentinel iteration disambiguates convergence on the final
        # allowed check from actual loop exhaustion in core_C3.
        ctm_max_steps + 1,
        ctm_conv_tol,
        identity_init,
        energy_proxy_fn=energy_proxy,
    )
    # One matched refresh per representative, always from its predecessor.
    env1_values = core.update_environmentCTs_3to1_C3(
        *result[6:9], *double_layers, chi, D**2
    )
    env2_values = core.update_environmentCTs_1to2_C3(
        *result[0:3], *double_layers, chi, D**2
    )
    env3_values = core.update_environmentCTs_2to3_C3(
        *result[3:6], *double_layers, chi, D**2
    )
    kept = (
        tuple(tensor.detach().contiguous() for tensor in double_layers)
        if keep_double_layers
        else None
    )
    output = ThreeEnvironments(
        env1=_as_environment(env1_values),
        env2=_as_environment(env2_values),
        env3=_as_environment(env3_values),
        double_layers=kept,
        ctm_steps=int(result[-1]),
    )
    del (
        result,
        env1_values,
        env2_values,
        env3_values,
        double_layers,
        sites,
        energy_proxy,
    )
    return output


def _spectrum_payload(
    operator: corr.RowToRowTransferOperator,
    *,
    eig_tol: float,
    arpack_ncv: int,
    arpack_maxiter: int,
    dense_dimension_threshold: int,
    seed: int,
) -> dict[str, Any]:
    result = corr.diagonalize_first_two_largest_eigval(
        operator,
        tol=eig_tol,
        ncv=min(arpack_ncv, operator.shape[0]),
        maxiter=arpack_maxiter,
        dense_dimension_threshold=dense_dimension_threshold,
        seed=seed,
        return_result=True,
    )
    if not isinstance(result, corr.EigensolverResult):
        raise RuntimeError("Detailed eigensolver diagnostics were not returned.")
    first, second = result.eigenvalues
    inverse_xi = float(math.log(abs(first / second)))
    resolution = 64.0 * np.finfo(
        np.float64 if operator.dtype == torch.float64 else np.float32
    ).eps
    payload = {
        "eigenvalues": [
            {
                "real": float(value.real),
                "imag": float(value.imag),
                "abs": float(abs(value)),
                "relative_residual": float(residual),
            }
            for value, residual in zip(
                result.eigenvalues,
                result.relative_residuals,
                strict=True,
            )
        ],
        "inverse_correlation_length": inverse_xi,
        "correlation_length": (
            None if inverse_xi <= resolution else float(1.0 / inverse_xi)
        ),
        "correlation_length_unresolved": bool(inverse_xi <= resolution),
        "used_dense_solver": result.used_dense_solver,
        "eigensolver_matvec_count": result.matvec_count,
        "eigensolver_seconds": result.elapsed_seconds,
    }
    return payload


def _generalized_payload(
    raw: corr.RowToRowTransferOperator,
    factory: Callable[[], corr.RowToRowTransferOperator],
    **eigensolver_options: Any,
) -> dict[str, Any]:
    generalized = factory()
    payload = _spectrum_payload(generalized, **eigensolver_options)
    ranks = getattr(generalized, "corner_effective_ranks", None)
    condition = getattr(generalized, "overlap_condition_number", None)
    payload["corner_effective_ranks"] = (
        None if ranks is None else [int(value) for value in ranks]
    )
    payload["overlap_condition_number"] = (
        None if condition is None else float(condition)
    )
    del generalized
    return payload


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _default_output(checkpoint: Path, j2: float, D: int) -> Path:
    tag = f"J2_{j2:g}".replace(".", "p")
    return HERE / RESULT_DIRECTORY_NAME / tag / f"D_{D}.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_valid_completed_output(
    path: Path, *, j2: float, D: int, checkpoint_sha256: str
) -> bool:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("schema") != "neel_six_correlation_lengths":
            return False
        if document.get("schema_version") != SCHEMA_VERSION:
            return False
        if document.get("transfer_network_schema") != TRANSFER_NETWORK_SCHEMA:
            return False
        if int(document["D"]) != D:
            return False
        if document.get("checkpoint_sha256") != checkpoint_sha256:
            return False
        if not math.isclose(
            float(document["J2"]), j2, rel_tol=0.0, abs_tol=1.0e-12
        ):
            return False
        ctm = document["ctm"]
        if not (
            ctm["converged_ab_within_budget"]
            and ctm["converged_ba_within_budget"]
        ):
            return False
        spectra = document["spectra"]
        required = {
            "env2_generalized",
            "env1_ab_env3_ba_generalized",
            "env3_ab_env1_ba_generalized",
            "env2_ordinary",
            "env1_ab_env3_ba_ordinary",
            "env3_ab_env1_ba_ordinary",
        }
        return all(len(spectra[key]["eigenvalues"]) >= 2 for key in required)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--J2", type=float, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--ctm-max-steps", type=int, default=corr.DEFAULT_CTM_MAX_STEPS
    )
    parser.add_argument(
        "--ctm-conv-tol", type=float, default=corr.DEFAULT_CTM_CONV_TOL
    )
    parser.add_argument(
        "--ctm-conv-mode",
        choices=("SVdifference", "Edifference", "both"),
        default=corr.DEFAULT_CTM_CONV_MODE,
    )
    parser.add_argument(
        "--ctm-e-conv-threshold",
        type=float,
        default=corr.DEFAULT_CTM_E_CONV_THRESHOLD,
    )
    parser.add_argument("--random-init", action="store_true")
    parser.add_argument("--eig-tol", type=float, default=0.0)
    parser.add_argument("--arpack-ncv", type=int, default=64)
    parser.add_argument("--arpack-maxiter", type=int, default=4000)
    parser.add_argument("--dense-dimension-threshold", type=int, default=512)
    parser.add_argument(
        "--corner-relative-cutoff", type=float, default=1.0e-14
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Return success only when --output is a valid completed v3 result.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.threads < 1:
        raise ValueError("--threads must be positive.")
    torch.set_num_threads(args.threads)
    checkpoint = args.checkpoint.resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(
        payload.get("a_raw"), torch.Tensor
    ):
        raise ValueError(f"{checkpoint} is not a Neel a_raw checkpoint.")
    a = payload["a_raw"].detach().to(dtype=torch.float64, device="cpu")
    D = int(payload.get("D_bond", a.shape[0]))
    chi = int(payload["chi"])
    if D < 3:
        raise ValueError("Néel correlation lengths are disabled for D<3.")
    if tuple(a.shape[:3]) != (D, D, D):
        raise ValueError("Checkpoint D_bond disagrees with a_raw.")
    symmetry_error = _symmetry_error(a)
    if symmetry_error > 1.0e-9:
        raise ValueError(
            f"a_raw is not three-leg symmetric: relative error "
            f"{symmetry_error:.3e}."
        )
    b = _physical_y_partner(a)
    output = (
        args.output.resolve()
        if args.output is not None
        else _default_output(checkpoint, args.J2, D)
    )
    checkpoint_sha256 = _sha256(checkpoint)
    if args.check_only:
        return (
            0
            if _is_valid_completed_output(
                output,
                j2=float(args.J2),
                D=D,
                checkpoint_sha256=checkpoint_sha256,
            )
            else 1
        )
    if (
        output.is_file()
        and not args.overwrite
        and _is_valid_completed_output(
            output,
            j2=float(args.J2),
            D=D,
            checkpoint_sha256=checkpoint_sha256,
        )
    ):
        print(f"SKIP existing {output}", flush=True)
        return 0
    if output.is_file() and not args.overwrite:
        print(f"RECOMPUTE invalid or obsolete output {output}", flush=True)

    seed = secrets.randbits(32) if args.seed is None else int(args.seed)
    if not 0 <= seed < 2**32:
        raise ValueError("--seed must lie in [0, 2**32).")
    np.random.seed(seed)
    torch.manual_seed(seed)
    core.set_dtype(True, use_real=True)
    core.set_device(torch.device("cpu"))
    core._SVD_CPU_OFFLOAD_THRESHOLD = 0
    force_full = D <= 4
    core._USE_FULL_SVD = force_full
    core.set_rsvd_mode(
        "full_svd" if force_full else corr.DEFAULT_RSVD_MODE,
        neumann_terms=corr.DEFAULT_RSVD_NEUMANN_TERMS,
        power_iters=corr.DEFAULT_RSVD_POWER_ITERS,
    )
    core.set_ctm_conv_mode(
        args.ctm_conv_mode,
        e_threshold=args.ctm_e_conv_threshold,
    )

    started = time.perf_counter()
    print(
        f"RUN J2={args.J2:g} D={D} chi={chi} seed={seed} "
        f"threads={args.threads}",
        flush=True,
    )
    ab = _run_three_environments(
        a,
        b,
        chi=chi,
        D=D,
        ctm_max_steps=args.ctm_max_steps,
        ctm_conv_tol=args.ctm_conv_tol,
        j1=corr.DEFAULT_J1,
        j2=args.J2,
        identity_init=not args.random_init,
        keep_double_layers=True,
    )
    ba = _run_three_environments(
        b,
        a,
        chi=chi,
        D=D,
        ctm_max_steps=args.ctm_max_steps,
        ctm_conv_tol=args.ctm_conv_tol,
        j1=corr.DEFAULT_J1,
        j2=args.J2,
        identity_init=not args.random_init,
        keep_double_layers=False,
    )
    if ab.double_layers is None:
        raise RuntimeError("The (a,b) double layers were not retained.")
    A, B, C_layer, D_layer, E, F = ab.double_layers

    # The third row direction has upper edges (T2C,T1D), lower edges
    # (T1D,T2C), and a horizontal D--C central bond.  D and C must first be
    # put in canonical [top,bond,bottom] coordinates.  The former env2
    # implementation used a vertical A--B bow-tie and therefore represented
    # a different tensor network.
    raw_env2 = CrossedEnvironmentTransferOperator(
        ab.env2.t2,
        ab.env2.t1,
        D_layer.permute(0, 2, 1).contiguous(),
        C_layer.permute(1, 2, 0).contiguous(),
        ba.env2.t1,
        ba.env2.t2,
    )
    if not isinstance(raw_env2, corr.RowToRowTransferOperator):
        raise RuntimeError("env2 did not produce a transfer operator.")
    raw_13 = CrossedEnvironmentTransferOperator(
        ab.env1.t2,
        ab.env1.t1,
        B.permute(1, 0, 2).contiguous(),
        E,
        ba.env3.t1,
        ba.env3.t2,
    )
    raw_31 = CrossedEnvironmentTransferOperator(
        ab.env3.t2,
        ab.env3.t1,
        F.permute(2, 1, 0).contiguous(),
        A.permute(2, 0, 1).contiguous(),
        ba.env1.t1,
        ba.env1.t2,
    )

    base_options = {
        "eig_tol": args.eig_tol,
        "arpack_ncv": args.arpack_ncv,
        "arpack_maxiter": args.arpack_maxiter,
        "dense_dimension_threshold": args.dense_dimension_threshold,
    }
    spectra: dict[str, Any] = {}
    spectra["env2_generalized"] = _generalized_payload(
        raw_env2,
        lambda: KroneckerCornerWhitenedTransferOperator(
            raw_env2,
            ab.env2.corner.T.contiguous(),
            ba.env2.corner,
            relative_cutoff=args.corner_relative_cutoff,
        ),
        seed=seed,
        **base_options,
    )
    spectra["env1_ab_env3_ba_generalized"] = _generalized_payload(
        raw_13,
        lambda: KroneckerCornerWhitenedTransferOperator(
            raw_13,
            ab.env1.corner.T.contiguous(),
            ba.env3.corner,
            relative_cutoff=args.corner_relative_cutoff,
        ),
        seed=(seed + 1) % 2**32,
        **base_options,
    )
    spectra["env3_ab_env1_ba_generalized"] = _generalized_payload(
        raw_31,
        lambda: KroneckerCornerWhitenedTransferOperator(
            raw_31,
            ab.env3.corner.T.contiguous(),
            ba.env1.corner,
            relative_cutoff=args.corner_relative_cutoff,
        ),
        seed=(seed + 2) % 2**32,
        **base_options,
    )
    spectra["env2_ordinary"] = _spectrum_payload(
        raw_env2, seed=(seed + 3) % 2**32, **base_options
    )
    spectra["env1_ab_env3_ba_ordinary"] = _spectrum_payload(
        raw_13, seed=(seed + 4) % 2**32, **base_options
    )
    spectra["env3_ab_env1_ba_ordinary"] = _spectrum_payload(
        raw_31, seed=(seed + 5) % 2**32, **base_options
    )

    document = {
        "schema": "neel_six_correlation_lengths",
        "schema_version": SCHEMA_VERSION,
        "transfer_network_schema": TRANSFER_NETWORK_SCHEMA,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "J2": float(args.J2),
        "D": D,
        "chi": chi,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_derived_validation_only": bool(
            payload.get("derived_validation_only", False)
        ),
        "virtual_symmetry_relative_error": symmetry_error,
        "seed": seed,
        "seed_was_randomized": args.seed is None,
        "threads": args.threads,
        "dtype": "torch.float64",
        "device": "cpu",
        "ctm": {
            "max_steps": args.ctm_max_steps,
            "convergence_tolerance": args.ctm_conv_tol,
            "convergence_mode": args.ctm_conv_mode,
            "energy_convergence_threshold": args.ctm_e_conv_threshold,
            "identity_init": not args.random_init,
            "svd_mode": "full_svd" if force_full else corr.DEFAULT_RSVD_MODE,
            "steps_ab": ab.ctm_steps,
            "steps_ba": ba.ctm_steps,
            "converged_ab_within_budget": ab.ctm_steps <= args.ctm_max_steps,
            "converged_ba_within_budget": ba.ctm_steps <= args.ctm_max_steps,
            "internal_sentinel_max_steps": args.ctm_max_steps + 1,
        },
        "eigensolver": {
            "tolerance": args.eig_tol,
            "ncv": args.arpack_ncv,
            "maxiter": args.arpack_maxiter,
            "dense_dimension_threshold": args.dense_dimension_threshold,
            "corner_relative_cutoff": args.corner_relative_cutoff,
        },
        "spectra": spectra,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_write(output, document)
    print(
        f"DONE {output} ({document['elapsed_seconds']:.1f} s)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
