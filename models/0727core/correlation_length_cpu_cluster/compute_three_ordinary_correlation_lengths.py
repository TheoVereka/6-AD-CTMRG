#!/usr/bin/env python3
"""Compute three ordinary row-transfer spectra for a C3-CTM ansatz.

For one C3-compatible checkpoint this evaluates the three straight rows

* env2(a,b) x env2(b,a);
* env1(a,b) x env3(b,a);
* env3(a,b) x env1(b,a).

The raw row-to-row operators are diagonalized without a corner metric.  These
ordinary spectra are gauge-fixed diagnostics, not claimed gauge-independent
physical spectra.  Every
inverse correlation length is defined directly from its eigenvalues as
``log(abs(lambda_1 / lambda_2))``.  The plot value is the median of the
three directions and the lower/upper error-bar endpoints are their minimum
and maximum.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

import correlation_length as corr
import core_C3 as core
from bundle_utils import validate_ansatz_directory


SCHEMA = "c3ctm_three_ordinary_correlation_lengths"
SCHEMA_VERSION = 6
TRANSFER_NETWORK_SCHEMA = "three_geometric_straight_rows_ordinary_v6"
DEFAULT_THREADS = int(os.environ.get("SLURM_CPUS_PER_TASK", "16"))


@dataclasses.dataclass(frozen=True)
class Environment:
    corner: torch.Tensor
    t1: torch.Tensor
    t2: torch.Tensor


@dataclasses.dataclass(frozen=True)
class ThreeEnvironments:
    env1: Environment
    env2: Environment
    env3: Environment
    double_layers: tuple[torch.Tensor, ...] | None
    ctm_steps: int
    final_energy_proxy: float


def _as_environment(values: Sequence[torch.Tensor]) -> Environment:
    return Environment(
        corner=values[0].detach().contiguous(),
        t1=values[1].detach().contiguous(),
        t2=values[2].detach().contiguous(),
    )


@torch.no_grad()
def _run_three_environments_from_sites(
    raw_sites: Sequence[torch.Tensor],
    *,
    chi: int,
    ctm_max_steps: int,
    ctm_conv_tol: float,
    identity_init: bool,
    j1: float,
    j2: float,
    keep_double_layers: bool,
) -> ThreeEnvironments:
    """Run CTMRG once and refresh all representatives from predecessors."""

    if len(raw_sites) != 6:
        raise ValueError("A C3-CTM environment requires six derived sites")
    D = int(raw_sites[0].shape[0])
    sites = tuple(
        core.normalize_single_layer_tensor_for_double_layer(site)
        for site in raw_sites
    )
    double_layers = tuple(
        tensor.detach().contiguous()
        for tensor in core.abcdef_to_ABCDEF(*sites, D**2)
    )
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
        # core_C3 returns max_steps both when convergence is first detected
        # at the final allowed check and when the loop simply exhausts.  One
        # extra sentinel iteration disambiguates those cases without changing
        # the stated convergence budget.
        ctm_max_steps + 1,
        ctm_conv_tol,
        identity_init,
        energy_proxy_fn=energy_proxy,
    )
    final_energy_proxy = float(energy_proxy(*result[:9]))

    # Each representative is refreshed exactly once from its own predecessor.
    # Taking result[:3] for all directions would mix three coordinate frames.
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
        final_energy_proxy=final_energy_proxy,
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


@torch.no_grad()
def _run_three_environments(
    raw_a: torch.Tensor,
    raw_b: torch.Tensor,
    **kwargs: Any,
) -> ThreeEnvironments:
    """Backward-compatible two-C3 wrapper used by old diagnostics."""

    return _run_three_environments_from_sites(
        core.twoc3_abcdef_from_ab(raw_a, raw_b), **kwargs
    )


def _load_c3_checkpoint(
    path: Path,
    ansatz_directory: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[tuple[torch.Tensor, ...], dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError("The checkpoint must contain a dictionary")
    D_value = checkpoint.get("D_bond")

    if ansatz_directory == "2tensor_twoC3":
        params = (checkpoint["a_raw"], checkpoint["b_raw"])
        derived = core.twoc3_abcdef_from_ab(*params)
    elif ansatz_directory == "1tensor_C6Ypi":
        derived = core.c6ypi_abcdef_from_a(checkpoint["a_raw"])
    elif ansatz_directory == "1tensor_C3Vypi":
        derived = core.c3vypi_abcdef_from_a(checkpoint["a_raw"])
    elif ansatz_directory == "neel_symmetrized":
        a_sym = core.symmetrize_virtual_legs(checkpoint["a_raw"])
        derived = core.neel_abcdef_from_a(a_sym)
    elif ansatz_directory == "neel_free_param":
        if D_value is None:
            raise KeyError("neel_free_param checkpoint has no D_bond")
        a_sym = core.neel_param_to_a(checkpoint["h_neel"], int(D_value))
        derived = core.neel_abcdef_from_a(a_sym)
    else:
        raise ValueError(
            f"Unsupported C3-compatible ansatz directory: {ansatz_directory}"
        )

    sites = tuple(
        corr._prepare_raw_tensor(
            tensor, dtype=dtype, device=device, name=f"site_{index}"
        )
        for index, tensor in enumerate(derived)
    )
    D = int(sites[0].shape[0])
    if D_value is not None and int(D_value) != D:
        raise ValueError("Checkpoint D_bond disagrees with tensor dimensions")
    metadata = {
        key: checkpoint.get(key)
        for key in ("D_bond", "chi", "loss", "energy", "step", "timestamp")
    }
    del checkpoint, derived
    return sites, metadata


def _possibly_dense(
    operator: corr.SideBySideRowToRowTransferOperator,
    *,
    matrix_free: bool,
    progress_every: int,
) -> corr.RowToRowTransferOperator:
    if matrix_free:
        return operator
    item_size = torch.empty((), dtype=operator.dtype).element_size()
    dense = operator.to_dense(
        max_dense_bytes=max(
            512 * 1024**2,
            operator.shape[0] * operator.shape[1] * item_size,
        )
    )
    wrapped = corr.DenseRowToRowTransferOperator(
        dense,
        chi=operator.chi,
        progress_every=progress_every,
    )
    del dense, operator
    return wrapped


def _spectrum(
    raw: corr.RowToRowTransferOperator,
    *,
    seed: int,
    eig_tol: float,
    arpack_ncv: int,
    arpack_maxiter: int,
    dense_threshold: int,
) -> dict[str, Any]:
    result = corr.diagonalize_first_two_largest_eigval(
        raw,
        tol=eig_tol,
        ncv=min(arpack_ncv, raw.shape[0]),
        maxiter=arpack_maxiter,
        dense_dimension_threshold=dense_threshold,
        seed=seed,
        return_result=True,
    )
    if not isinstance(result, corr.EigensolverResult):
        raise RuntimeError("Detailed eigensolver diagnostics were not returned.")
    largest, second = sorted(
        (float(abs(value)) for value in result.eigenvalues[:2]),
        reverse=True,
    )
    if largest <= 0.0 or second <= 0.0:
        raise ValueError("Both leading eigenvalue magnitudes must be positive")
    inverse_xi = float(math.log(largest / second))
    if not math.isfinite(inverse_xi) or inverse_xi < 0.0:
        raise ValueError(f"Invalid inverse correlation length: {inverse_xi}")
    resolution = 64.0 * np.finfo(
        np.float64 if raw.dtype == torch.float64 else np.float32
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
        "eigenproblem": "ordinary_raw_row_transfer",
        "used_dense_solver": bool(result.used_dense_solver),
        "eigensolver_matvec_count": int(result.matvec_count),
        "eigensolver_seconds": float(result.elapsed_seconds),
    }
    del raw
    return payload


def _summary(spectra: dict[str, dict[str, Any]]) -> dict[str, Any]:
    values = {
        key: float(value["inverse_correlation_length"])
        for key, value in spectra.items()
    }
    ordered = sorted(values.values())
    lower, center, upper = ordered
    return {
        "definition": "ln(abs(lambda_1/lambda_2))",
        "aggregation": "lower=min, center=median, upper=max",
        "lower": lower,
        "center": center,
        "upper": upper,
        "lower_error": center - lower,
        "upper_error": upper - center,
        "direction_values": values,
    }


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--ansatz-directory", required=True)
    parser.add_argument(
        "--direction",
        choices=("env2", "env1_ab_env3_ba", "env3_ab_env1_ba"),
        help="Compute only one direction for a split D>=10 job.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chi", type=int)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--J1", type=float, default=corr.DEFAULT_J1)
    parser.add_argument("--J2", type=float, required=True)
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
    parser.add_argument("--matrix-free", action="store_true")
    parser.add_argument("--max-intermediate-mib", type=float, default=1024.0)
    parser.add_argument("--eig-tol", type=float, default=0.0)
    parser.add_argument(
        "--arpack-ncv", type=int, default=corr.DEFAULT_ARPACK_NCV
    )
    parser.add_argument(
        "--arpack-maxiter", type=int, default=corr.DEFAULT_ARPACK_MAXITER
    )
    parser.add_argument("--dense-threshold", type=int, default=256)
    parser.add_argument("--progress-every", type=int, default=10)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    validate_ansatz_directory(args.ansatz_directory)
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    if args.ctm_max_steps < 1:
        raise ValueError("--ctm-max-steps must be positive")
    if args.ctm_conv_tol <= 0.0:
        raise ValueError("--ctm-conv-tol must be positive")
    torch.set_num_threads(args.threads)
    device = torch.device("cpu")
    dtype = torch.float64
    sites, metadata = _load_c3_checkpoint(
        args.checkpoint.resolve(),
        args.ansatz_directory,
        device=device,
        dtype=dtype,
    )
    D = int(sites[0].shape[0])
    if D < 3:
        raise ValueError("C3-CTM correlation lengths are disabled for D<3")
    chi_value = args.chi if args.chi is not None else metadata.get("chi")
    if chi_value is None:
        raise ValueError("--chi is required when absent from the checkpoint")
    chi = int(chi_value)

    seed = secrets.randbits(32) if args.seed is None else int(args.seed)
    if not 0 <= seed < 2**32:
        raise ValueError("--seed must lie in [0, 2**32)")
    np.random.seed(seed)
    torch.manual_seed(seed)

    force_full = D in corr.FULL_SVD_CORRELATION_LENGTH_DS
    core.set_dtype(True, use_real=True)
    core.set_device(device)
    core._SVD_CPU_OFFLOAD_THRESHOLD = 0
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
        f"CTM(max={args.ctm_max_steps}, sv={args.ctm_conv_tol:g}, "
        f"mode={args.ctm_conv_mode}, energy={args.ctm_e_conv_threshold:g})",
        flush=True,
    )
    ab = _run_three_environments_from_sites(
        sites,
        chi=chi,
        ctm_max_steps=args.ctm_max_steps,
        ctm_conv_tol=args.ctm_conv_tol,
        identity_init=not args.random_init,
        j1=args.J1,
        j2=args.J2,
        keep_double_layers=True,
    )
    reflected_sites = (
        sites[1], sites[0], sites[3], sites[2], sites[5], sites[4]
    )
    ba = _run_three_environments_from_sites(
        reflected_sites,
        chi=chi,
        ctm_max_steps=args.ctm_max_steps,
        ctm_conv_tol=args.ctm_conv_tol,
        identity_init=not args.random_init,
        j1=args.J1,
        j2=args.J2,
        keep_double_layers=False,
    )
    if ab.double_layers is None:
        raise RuntimeError("(a,b) double layers were not retained")
    A, B, C, D_layer, E, F = ab.double_layers
    max_bytes = int(args.max_intermediate_mib * 1024**2)
    operator_options = {
        "max_intermediate_bytes": max_bytes,
        "progress_every": args.progress_every,
    }
    transfer_specs = {
        "env2": (
            ab.env2.t2,
            ab.env2.t1,
            D_layer.permute(0, 2, 1).contiguous(),
            C.permute(1, 2, 0).contiguous(),
            ba.env2.t1,
            ba.env2.t2,
        ),
        "env1_ab_env3_ba": (
            ab.env1.t2,
            ab.env1.t1,
            B.permute(1, 0, 2).contiguous(),
            E,
            ba.env3.t1,
            ba.env3.t2,
        ),
        "env3_ab_env1_ba": (
            ab.env3.t2,
            ab.env3.t1,
            F.permute(2, 1, 0).contiguous(),
            A.permute(2, 0, 1).contiguous(),
            ba.env1.t1,
            ba.env1.t2,
        ),
    }
    spectra: dict[str, dict[str, Any]] = {}
    direction_order = ("env2", "env1_ab_env3_ba", "env3_ab_env1_ba")
    selected_directions = (
        (args.direction,) if args.direction is not None else direction_order
    )
    for key in selected_directions:
        offset = direction_order.index(key)
        print(f"DIAGONALIZE ordinary {key}", flush=True)
        network_tensors = transfer_specs.pop(key)
        side_by_side = corr.SideBySideRowToRowTransferOperator(
            *network_tensors,
            **operator_options,
        )
        raw = _possibly_dense(
            side_by_side,
            matrix_free=args.matrix_free,
            progress_every=args.progress_every,
        )
        spectra[key] = _spectrum(
            raw,
            seed=(seed + offset) % 2**32,
            eig_tol=args.eig_tol,
            arpack_ncv=args.arpack_ncv,
            arpack_maxiter=args.arpack_maxiter,
            dense_threshold=args.dense_threshold,
        )

    inverse_summary = _summary(spectra) if args.direction is None else None
    center_inverse_xi = (
        float(inverse_summary["center"])
        if inverse_summary is not None
        else float(spectra[args.direction]["inverse_correlation_length"])
    )
    document = {
        "schema": (
            SCHEMA
            if args.direction is None
            else "c3ctm_single_ordinary_correlation_length_direction"
        ),
        "schema_version": SCHEMA_VERSION if args.direction is None else 1,
        "transfer_network_schema": TRANSFER_NETWORK_SCHEMA,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint.resolve()),
        "ansatz_directory": args.ansatz_directory,
        "D_bond": D,
        "chi": chi,
        "dtype": str(dtype),
        "device": "cpu",
        "seed": seed,
        "seed_was_randomized": args.seed is None,
        "direction": args.direction,
        "threads": args.threads,
        "ctm": {
            "max_steps": args.ctm_max_steps,
            "sv_convergence_tolerance": args.ctm_conv_tol,
            "convergence_mode": args.ctm_conv_mode,
            "energy_convergence_threshold": args.ctm_e_conv_threshold,
            "identity_init": not args.random_init,
            "svd_mode": (
                "full_svd" if force_full else corr.DEFAULT_RSVD_MODE
            ),
            "steps_ab": ab.ctm_steps,
            "steps_ba": ba.ctm_steps,
            "converged_ab_within_budget": ab.ctm_steps <= args.ctm_max_steps,
            "converged_ba_within_budget": ba.ctm_steps <= args.ctm_max_steps,
            "internal_sentinel_max_steps": args.ctm_max_steps + 1,
            "final_energy_proxy_ab": ab.final_energy_proxy,
            "final_energy_proxy_ba": ba.final_energy_proxy,
        },
        "calculation_hyperparameters": {
            "J1": float(args.J1),
            "J2": float(args.J2),
            "matrix_free": bool(args.matrix_free),
            "max_intermediate_mib": float(args.max_intermediate_mib),
            "eig_tol": float(args.eig_tol),
            "arpack_ncv": int(args.arpack_ncv),
            "arpack_maxiter": int(args.arpack_maxiter),
            "dense_threshold": int(args.dense_threshold),
            "eigenproblem": "ordinary_raw_row_transfer",
        },
        "spectra": spectra,
        "inverse_correlation_length": inverse_summary,
        "correlation_length": (
            None
            if center_inverse_xi <= np.finfo(float).eps
            else 1.0 / center_inverse_xi
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_write(args.output.resolve(), document)
    print(
        f"DONE {args.output.resolve()} center 1/xi={center_inverse_xi:.12g} "
        + (
            f"range=[{inverse_summary['lower']:.12g}, {inverse_summary['upper']:.12g}] "
            if inverse_summary is not None
            else f"direction={args.direction} "
        )
        + f"({document['elapsed_seconds']:.1f} s)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
