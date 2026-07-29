#!/usr/bin/env python3
"""Compare the two unused C3-environment row-to-row transfer directions.

This is a temporary, non-production diagnostic for the two-C3 ansatz.  For
each checkpoint it runs CTMRG for the placements ``(a,b)`` and ``(b,a)`` once,
then constructs the two crossed-environment transfer pencils

1. ``env1(a,b) x env3(b,a)`` with central sites ``B -- C3(A)``;
2. ``env3(a,b) x env1(b,a)`` with central sites ``C3(B) -- A``.

Unlike ``correlation_length.py``, the two central double-layer sites lie
side-by-side: each site has one upper boundary leg, one lower boundary leg,
and the horizontal inter-site bond.  The generalized empty-row map uses the
corner of the upper environment tensor-producted with the corner of the lower
environment.

Default sweep:
    J2 = 0.0, 0.24, 0.25, ..., 0.30
    D  = 3, 4, 5

Results are diagnostic only and are saved beside this script, not below
0713summary.  The JSON is updated atomically after every completed checkpoint
so an interrupted run can be resumed.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import secrets
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

import correlation_length as corr
import core_C3 as core


DEFAULT_ROOT = Path(
    r"D:\HyraiOn\ENS_Lyon\Internship\2026-EPFL\data\0713summary"
)
DEFAULT_J2S = (0.0, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30)
DEFAULT_DS = (3, 4, 5)
DEFAULT_OUTPUT = Path(__file__).with_name(
    "temporary_crossed_environment_correlation_lengths.json"
)
ANSATZ_DIRECTORY = "2tensor_twoC3"


@dataclasses.dataclass(frozen=True)
class Environment:
    """One natural ``(corner, T1, T2)`` C3 environment representative."""

    corner: torch.Tensor
    t1: torch.Tensor
    t2: torch.Tensor


@dataclasses.dataclass(frozen=True)
class OrderingResult:
    """The aligned env1/env3 representatives for one single-layer ordering."""

    env1: Environment
    env3: Environment
    double_layers: tuple[torch.Tensor, ...] | None
    ctm_steps: int
    energy_proxy: float


class CrossedEnvironmentTransferOperator(corr.RowToRowTransferOperator):
    """Matrix-free transfer with two central double-layer sites side-by-side.

    In canonical local coordinates the central sites are
    ``left[top, bond, bottom]`` and ``right[top, bond, bottom]``.  The
    represented left-to-right transfer matrix is

    ``V[Y,y,V,v] = sum``
    ``upper_left[M,Y,i] upper_right[M,V,j]``
    ``left[i,h,k] right[j,h,l]``
    ``lower_left[m,y,k] lower_right[m,v,l]``.

    The horizontal central contraction is cached as
    ``central_pair[i,k,j,l]``.  A matvec then uses intermediates no larger
    than ``O(chi**2 q**2)`` rather than materializing an ``O(chi**4 q)``
    boundary half.
    """

    def __init__(
        self,
        upper_left: torch.Tensor,
        upper_right: torch.Tensor,
        left_site: torch.Tensor,
        right_site: torch.Tensor,
        lower_left: torch.Tensor,
        lower_right: torch.Tensor,
        *,
        progress_every: int = 0,
    ) -> None:
        tensors = (
            upper_left,
            upper_right,
            left_site,
            right_site,
            lower_left,
            lower_right,
        )
        if any(not isinstance(tensor, torch.Tensor) for tensor in tensors):
            raise TypeError("All transfer-network objects must be tensors.")
        if any(tensor.is_complex() for tensor in tensors):
            raise TypeError("This diagnostic requires real tensors.")
        dtype = upper_left.dtype
        device = upper_left.device
        if dtype not in (torch.float32, torch.float64):
            raise TypeError(f"Unsupported transfer dtype: {dtype}.")
        if any(tensor.dtype != dtype for tensor in tensors):
            raise TypeError("All transfer tensors must have the same dtype.")
        if any(tensor.device != device for tensor in tensors):
            raise ValueError("All transfer tensors must use the same device.")

        if upper_left.ndim != 3:
            raise ValueError("Every boundary edge must have rank three.")
        chi = int(upper_left.shape[0])
        q = int(upper_left.shape[2])
        edge_shape = (chi, chi, q)
        for name, tensor in (
            ("upper_left", upper_left),
            ("upper_right", upper_right),
            ("lower_left", lower_left),
            ("lower_right", lower_right),
        ):
            if tuple(tensor.shape) != edge_shape:
                raise ValueError(
                    f"{name} must have shape {edge_shape}; "
                    f"got {tuple(tensor.shape)}."
                )
        site_shape = (q, q, q)
        if tuple(left_site.shape) != site_shape:
            raise ValueError(
                f"left_site must have shape {site_shape}; "
                f"got {tuple(left_site.shape)}."
            )
        if tuple(right_site.shape) != site_shape:
            raise ValueError(
                f"right_site must have shape {site_shape}; "
                f"got {tuple(right_site.shape)}."
            )
        if progress_every < 0:
            raise ValueError("progress_every cannot be negative.")

        self.upper_left = upper_left.detach().contiguous()
        self.upper_right = upper_right.detach().contiguous()
        self.lower_left = lower_left.detach().contiguous()
        self.lower_right = lower_right.detach().contiguous()
        # left[i,h,k] right[j,h,l] -> central_pair[i,k,j,l]
        self.central_pair = torch.tensordot(
            left_site.detach(),
            right_site.detach(),
            dims=([1], [1]),
        ).contiguous()

        self.chi = chi
        self.q = q
        self.shape = (chi**2, chi**2)
        self.dtype = dtype
        self.device = device
        self.numpy_dtype = np.float64 if dtype == torch.float64 else np.float32
        self.progress_every = int(progress_every)
        self.matvec_count = 0
        self._first_matvec_time = time.perf_counter()
        self.chunk_size = q
        self.max_intermediate_bytes = 0

    @torch.no_grad()
    def _matvec_real_tensor(self, vector: torch.Tensor) -> torch.Tensor:
        x = vector.reshape(self.chi, self.chi)
        # upper_right[M,V,j] x[V,v] -> a[M,j,v]
        a = torch.tensordot(self.upper_right, x, dims=([1], [0]))
        # a[M,j,v] lower_right[m,v,l] -> b[M,j,m,l]
        b = torch.tensordot(a, self.lower_right, dims=([2], [1]))
        del a
        # b[M,j,m,l] central_pair[i,k,j,l] -> c[M,m,i,k]
        c = torch.tensordot(
            b,
            self.central_pair,
            dims=([1, 3], [2, 3]),
        )
        del b
        # c[M,m,i,k] lower_left[m,y,k] -> d[M,i,y]
        d = torch.tensordot(
            c,
            self.lower_left,
            dims=([1, 3], [0, 2]),
        )
        del c
        # d[M,i,y] upper_left[M,Y,i] -> output[y,Y] -> output[Y,y]
        output = torch.tensordot(
            d,
            self.upper_left,
            dims=([0, 1], [0, 2]),
        ).transpose(0, 1)
        del d, x
        self.matvec_count += 1
        self._report_progress()
        return output.reshape(-1)

    @torch.no_grad()
    def to_dense(self, *, max_dense_bytes: int = 512 * 1024**2) -> torch.Tensor:
        required = (
            self.shape[0]
            * self.shape[1]
            * torch.empty((), dtype=self.dtype).element_size()
        )
        if required > max_dense_bytes:
            raise MemoryError(
                f"Dense transfer would require {required / 1024**2:.1f} MiB."
            )
        identity = torch.eye(
            self.shape[1], dtype=self.dtype, device=self.device
        )
        columns = [
            self._matvec_real_tensor(identity[:, index])
            for index in range(self.shape[1])
        ]
        return torch.stack(columns, dim=1)


class KroneckerCornerWhitenedTransferOperator(
    corr.RowToRowTransferOperator
):
    """Standard form of ``T x = lambda (C_upper kron C_lower) x``.

    Raw row and column indices are respectively ``(Y,y)`` and ``(V,v)``:

    ``N[Y,y,V,v] = C_upper[Y,V] C_lower[y,v]``.

    Two-sided SVD whitening is applied independently to the two corners
    without constructing the ``chi**2`` Kronecker matrix.
    """

    def __init__(
        self,
        raw_transfer: CrossedEnvironmentTransferOperator,
        upper_corner: torch.Tensor,
        lower_corner: torch.Tensor,
        *,
        relative_cutoff: float,
    ) -> None:
        if relative_cutoff < 0.0:
            raise ValueError("relative_cutoff cannot be negative.")
        chi = raw_transfer.chi
        expected = (chi, chi)
        if (
            tuple(upper_corner.shape) != expected
            or tuple(lower_corner.shape) != expected
        ):
            raise ValueError(f"Both corners must have shape {expected}.")
        if (
            upper_corner.dtype != raw_transfer.dtype
            or lower_corner.dtype != raw_transfer.dtype
        ):
            raise TypeError("Corners and transfer must share one dtype.")
        if (
            upper_corner.device != raw_transfer.device
            or lower_corner.device != raw_transfer.device
        ):
            raise ValueError("Corners and transfer must share one device.")

        upper_u, upper_s, upper_vh = torch.linalg.svd(
            upper_corner.detach(), full_matrices=False
        )
        lower_u, lower_s, lower_vh = torch.linalg.svd(
            lower_corner.detach(), full_matrices=False
        )
        upper_kept = upper_s > relative_cutoff * float(upper_s[0])
        lower_kept = lower_s > relative_cutoff * float(lower_s[0])
        upper_inverse_sqrt = torch.zeros_like(upper_s)
        lower_inverse_sqrt = torch.zeros_like(lower_s)
        upper_inverse_sqrt[upper_kept] = torch.rsqrt(upper_s[upper_kept])
        lower_inverse_sqrt[lower_kept] = torch.rsqrt(lower_s[lower_kept])

        retained_upper = upper_s[upper_kept]
        retained_lower = lower_s[lower_kept]
        if retained_upper.numel() == 0 or retained_lower.numel() == 0:
            raise RuntimeError("The Kronecker corner map has zero rank.")

        self.raw_transfer = raw_transfer
        self.upper_u = upper_u
        self.upper_v = upper_vh.T.contiguous()
        self.lower_u = lower_u
        self.lower_v = lower_vh.T.contiguous()
        self.inverse_sqrt_weights = (
            upper_inverse_sqrt[:, None] * lower_inverse_sqrt[None, :]
        )
        self.upper_corner_numpy = upper_corner.detach().cpu().numpy()
        self.lower_corner_numpy = lower_corner.detach().cpu().numpy()
        self.corner_effective_ranks = (
            int(upper_kept.sum().item()),
            int(lower_kept.sum().item()),
        )
        self.overlap_condition_number = float(
            (retained_upper[0] / retained_upper[-1])
            * (retained_lower[0] / retained_lower[-1])
        )

        self.chi = chi
        self.q = raw_transfer.q
        self.shape = raw_transfer.shape
        self.dtype = raw_transfer.dtype
        self.device = raw_transfer.device
        self.numpy_dtype = raw_transfer.numpy_dtype
        self.progress_every = raw_transfer.progress_every
        self.matvec_count = 0
        self._first_matvec_time = time.perf_counter()
        self.chunk_size = raw_transfer.chunk_size
        self.max_intermediate_bytes = 0

    @torch.no_grad()
    def _right_whiten_real_tensor(self, vector: torch.Tensor) -> torch.Tensor:
        canonical = vector.reshape(self.chi, self.chi)
        weighted = self.inverse_sqrt_weights * canonical
        # (V,v) = V_upper @ weighted @ V_lower.T
        return self.upper_v @ weighted @ self.lower_v.T

    @torch.no_grad()
    def _matvec_real_tensor(self, vector: torch.Tensor) -> torch.Tensor:
        raw_input = self._right_whiten_real_tensor(vector)
        raw_output = self.raw_transfer._matvec_real_tensor(
            raw_input.reshape(-1)
        ).reshape(self.chi, self.chi)
        # (a,b) = U_upper.T @ raw[Y,y] @ U_lower
        output = (
            self.upper_u.T @ raw_output @ self.lower_u
        ) * self.inverse_sqrt_weights
        self.matvec_count += 1
        return output.reshape(-1)

    def generalized_refine_numpy(
        self,
        eigenvalue: complex,
        eigenvector: np.ndarray,
    ) -> tuple[complex, float]:
        vector = np.asarray(eigenvector)
        real_vector = torch.as_tensor(
            np.ascontiguousarray(vector.real),
            dtype=self.dtype,
            device=self.device,
        )
        raw_real = self._right_whiten_real_tensor(real_vector)
        raw_input = raw_real.detach().cpu().numpy().astype(
            self.numpy_dtype, copy=False
        )
        if np.iscomplexobj(vector):
            imag_vector = torch.as_tensor(
                np.ascontiguousarray(vector.imag),
                dtype=self.dtype,
                device=self.device,
            )
            raw_imag = self._right_whiten_real_tensor(imag_vector)
            raw_input = raw_input + 1j * raw_imag.detach().cpu().numpy().astype(
                self.numpy_dtype, copy=False
            )

        applied = self.raw_transfer.matvec_numpy(
            np.ascontiguousarray(raw_input.reshape(-1))
        ).reshape(self.chi, self.chi)
        empty_row = (
            self.upper_corner_numpy
            @ raw_input
            @ self.lower_corner_numpy.T
        )
        denominator = np.vdot(empty_row, empty_row)
        if abs(denominator) == 0.0:
            return complex(eigenvalue), math.inf
        refined = complex(np.vdot(empty_row, applied) / denominator)
        scale = float(np.linalg.norm(applied)) + abs(refined) * float(
            np.linalg.norm(empty_row)
        )
        residual = float(np.linalg.norm(applied - refined * empty_row))
        relative = residual / max(scale, np.finfo(self.numpy_dtype).tiny)
        return refined, relative

    @torch.no_grad()
    def to_dense(self, *, max_dense_bytes: int = 512 * 1024**2) -> torch.Tensor:
        required = (
            self.shape[0]
            * self.shape[1]
            * torch.empty((), dtype=self.dtype).element_size()
        )
        if required > max_dense_bytes:
            raise MemoryError(
                f"Dense whitened transfer needs {required / 1024**2:.1f} MiB."
            )
        identity = torch.eye(
            self.shape[1], dtype=self.dtype, device=self.device
        )
        columns = [
            self._matvec_real_tensor(identity[:, index])
            for index in range(self.shape[1])
        ]
        return torch.stack(columns, dim=1)


def _parse_j2_directory(name: str) -> float:
    if not name.startswith("J2_"):
        raise ValueError(name)
    return float(name[3:].replace("p", "."))


def _find_j2_directory(root: Path, requested: float) -> Path:
    matches = [
        path
        for path in root.glob("J2_*")
        if path.is_dir()
        and math.isclose(
            _parse_j2_directory(path.name),
            requested,
            rel_tol=0.0,
            abs_tol=5.0e-13,
        )
    ]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one directory for J2={requested:g}; found {matches}."
        )
    return matches[0]


def _environment(values: Sequence[torch.Tensor]) -> Environment:
    if len(values) != 3:
        raise ValueError("An environment requires corner, T1, and T2.")
    return Environment(
        corner=values[0].detach().contiguous(),
        t1=values[1].detach().contiguous(),
        t2=values[2].detach().contiguous(),
    )


@torch.no_grad()
def _run_ordering(
    raw_a: torch.Tensor,
    raw_b: torch.Tensor,
    *,
    chi: int,
    D_bond: int,
    j1: float,
    j2: float,
    ctm_max_steps: int,
    ctm_conv_tol: float,
    identity_init: bool,
    keep_double_layers: bool,
) -> OrderingResult:
    """Run CTMRG and refresh env1/env3 from their stored predecessors."""

    sites, double_layers = corr._build_ctm_layers(raw_a, raw_b)
    energy_proxy, _ = corr._build_lbfgs_energy_proxy(
        sites,
        chi=chi,
        D_bond=D_bond,
        j1=j1,
        j2=j2,
    )
    result = core.CTMRG_from_init_to_stop(
        *double_layers,
        chi,
        D_bond**2,
        ctm_max_steps,
        ctm_conv_tol,
        identity_init,
        energy_proxy_fn=energy_proxy,
    )
    ctm_steps = int(result[-1])
    energy = float(energy_proxy(*result[:9]))

    # Mirror the production extraction rule: refresh each requested
    # environment exactly once from the predecessor returned by CTMRG.
    # Chaining several extra updates would propagate finite-convergence error
    # by different numbers of steps for env1 and env3.
    env3 = core.update_environmentCTs_2to3_C3(
        *result[3:6],
        *double_layers,
        chi,
        D_bond**2,
    )
    env1 = core.update_environmentCTs_3to1_C3(
        *result[6:9],
        *double_layers,
        chi,
        D_bond**2,
    )

    kept_layers = (
        tuple(tensor.detach().contiguous() for tensor in double_layers)
        if keep_double_layers
        else None
    )
    output = OrderingResult(
        env1=_environment(env1),
        env3=_environment(env3),
        double_layers=kept_layers,
        ctm_steps=ctm_steps,
        energy_proxy=energy,
    )
    del result, env1, env3, energy_proxy, sites, double_layers
    corr._release_unused_memory(raw_a.device)
    return output


def _configure_ctm(
    *,
    D_bond: int,
    device: torch.device,
    ctm_conv_mode: str,
    ctm_e_conv_threshold: float,
) -> str:
    core.set_dtype(True, use_real=True)
    core.set_device(device)
    core._SVD_CPU_OFFLOAD_THRESHOLD = 0
    force_full = D_bond in corr.FULL_SVD_CORRELATION_LENGTH_DS
    core._USE_FULL_SVD = force_full
    mode = "full_svd" if force_full else corr.DEFAULT_RSVD_MODE
    core.set_rsvd_mode(
        mode,
        neumann_terms=corr.DEFAULT_RSVD_NEUMANN_TERMS,
        power_iters=corr.DEFAULT_RSVD_POWER_ITERS,
    )
    core.set_ctm_conv_mode(
        ctm_conv_mode,
        e_threshold=ctm_e_conv_threshold,
    )
    return mode


def _spectrum(
    raw_transfer: CrossedEnvironmentTransferOperator,
    upper_corner: torch.Tensor,
    lower_corner: torch.Tensor,
    *,
    corner_relative_cutoff: float,
    eig_tol: float,
    arpack_ncv: int,
    arpack_maxiter: int,
    seed: int,
) -> dict[str, Any]:
    transfer = KroneckerCornerWhitenedTransferOperator(
        raw_transfer,
        upper_corner,
        lower_corner,
        relative_cutoff=corner_relative_cutoff,
    )
    eigensolver = corr.diagonalize_first_two_largest_eigval(
        transfer,
        tol=eig_tol,
        ncv=min(arpack_ncv, transfer.shape[0]),
        maxiter=arpack_maxiter,
        dense_dimension_threshold=256,
        seed=seed,
        return_result=True,
    )
    if not isinstance(eigensolver, corr.EigensolverResult):
        raise RuntimeError("Expected detailed eigensolver output.")
    values = eigensolver.eigenvalues
    inverse_xi = float(math.log(abs(values[0] / values[1])))
    inverse_xi_resolution = 64.0 * np.finfo(
        np.float64 if transfer.dtype == torch.float64 else np.float32
    ).eps
    unresolved = bool(inverse_xi <= inverse_xi_resolution)
    xi = None if unresolved else float(1.0 / inverse_xi)
    result = {
        "eigenvalues": [
            {
                "real": float(value.real),
                "imag": float(value.imag),
                "abs": float(abs(value)),
                "relative_residual": float(residual),
            }
            for value, residual in zip(
                values,
                eigensolver.relative_residuals,
                strict=True,
            )
        ],
        "inverse_correlation_length": inverse_xi,
        "correlation_length": xi,
        "correlation_length_unresolved": unresolved,
        "corner_effective_ranks": list(transfer.corner_effective_ranks),
        "overlap_condition_number": transfer.overlap_condition_number,
        "eigensolver_matvec_count": eigensolver.matvec_count,
        "eigensolver_seconds": eigensolver.elapsed_seconds,
    }
    del transfer, eigensolver, raw_transfer
    return result


def _baseline(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        values = [
            complex(float(item["real"]), float(item["imag"]))
            for item in payload["eigenvalues"][:2]
        ]
        inverse_xi = float(math.log(abs(values[0] / values[1])))
        unresolved = bool(
            inverse_xi <= 64.0 * np.finfo(np.float64).eps
        )
        return {
            "source": str(path),
            "eigenvalues": payload["eigenvalues"][:2],
            "inverse_correlation_length_from_eigenvalues": inverse_xi,
            "correlation_length_from_eigenvalues": (
                None if unresolved else float(1.0 / inverse_xi)
            ),
            "correlation_length_unresolved": unresolved,
        }
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        json.JSONDecodeError,
    ):
        return None


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--J2", type=float, nargs="+", default=list(DEFAULT_J2S))
    parser.add_argument("--Ds", type=int, nargs="+", default=list(DEFAULT_DS))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--J1", type=float, default=corr.DEFAULT_J1)
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
    parser.add_argument(
        "--arpack-ncv", type=int, default=corr.DEFAULT_ARPACK_NCV
    )
    parser.add_argument(
        "--arpack-maxiter", type=int, default=corr.DEFAULT_ARPACK_MAXITER
    )
    parser.add_argument(
        "--corner-relative-cutoff",
        type=float,
        default=corr.DEFAULT_CORNER_RELATIVE_CUTOFF,
    )
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional fixed seed; omitted means a fresh random seed per case.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    root = args.root.resolve()
    output = args.output.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if args.threads < 1:
        raise ValueError("--threads must be positive.")
    if args.corner_relative_cutoff < 0.0:
        raise ValueError("--corner-relative-cutoff cannot be negative.")
    if any(D < 3 for D in args.Ds):
        raise ValueError("This diagnostic is restricted to D >= 3.")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    torch.set_num_threads(args.threads)

    if output.is_file() and not args.overwrite:
        document = json.loads(output.read_text(encoding="utf-8"))
    else:
        document = {
            "schema_version": 1,
            "description": (
                "Crossed env1/env3 row-to-row transfer spectra for two-C3."
            ),
            "root": str(root),
            "results": {},
        }
    results: dict[str, Any] = document.setdefault("results", {})

    jobs: list[tuple[float, int, Path]] = []
    for j2 in dict.fromkeys(args.J2):
        j2_directory = _find_j2_directory(root, j2)
        for D_bond in dict.fromkeys(args.Ds):
            checkpoint = (
                j2_directory
                / ANSATZ_DIRECTORY
                / f"D_{D_bond}"
                / "tensor_best.pt"
            )
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            jobs.append((j2, D_bond, checkpoint))

    for index, (j2, D_bond, checkpoint) in enumerate(jobs, start=1):
        key = f"J2={j2:.12g},D={D_bond}"
        if key in results and not args.overwrite:
            print(f"[{index}/{len(jobs)}] SKIP {key}", flush=True)
            continue
        case_seed = (
            secrets.randbits(32)
            if args.seed is None
            else (int(args.seed) + index - 1) % 2**32
        )
        np.random.seed(case_seed)
        torch.manual_seed(case_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(case_seed)

        print(
            f"[{index}/{len(jobs)}] RUN J2={j2:g}, D={D_bond}, "
            f"seed={case_seed}",
            flush=True,
        )
        started = time.perf_counter()
        a, b, metadata = corr._load_twoc3_checkpoint(
            str(checkpoint),
            device=device,
            dtype=torch.float64,
        )
        chi = int(metadata["chi"])
        rsvd_mode = _configure_ctm(
            D_bond=D_bond,
            device=device,
            ctm_conv_mode=args.ctm_conv_mode,
            ctm_e_conv_threshold=args.ctm_e_conv_threshold,
        )
        ordering_ab = _run_ordering(
            a,
            b,
            chi=chi,
            D_bond=D_bond,
            j1=args.J1,
            j2=j2,
            ctm_max_steps=args.ctm_max_steps,
            ctm_conv_tol=args.ctm_conv_tol,
            identity_init=not args.random_init,
            keep_double_layers=True,
        )
        ordering_ba = _run_ordering(
            b,
            a,
            chi=chi,
            D_bond=D_bond,
            j1=args.J1,
            j2=j2,
            ctm_max_steps=args.ctm_max_steps,
            ctm_conv_tol=args.ctm_conv_tol,
            identity_init=not args.random_init,
            keep_double_layers=False,
        )
        if ordering_ab.double_layers is None:
            raise RuntimeError("Missing retained (a,b) double layers.")
        A, B, _, _, E, F = ordering_ab.double_layers

        # env1(ab) x env3(ba):
        # B[a,b,g] is locally [top=b, bond=a, bottom=g].
        # E=C3(A) has axes [A_b,A_g,A_a] and is already
        # [top=A_b, bond=A_g, bottom=A_a].
        raw_13 = CrossedEnvironmentTransferOperator(
            ordering_ab.env1.t2,  # T2A: top-left, attached to B_b
            ordering_ab.env1.t1,  # T1F: top-right, attached to A_b
            B.permute(1, 0, 2).contiguous(),
            E,
            ordering_ba.env3.t1,  # T1B: bottom-left, attached to B_g
            ordering_ba.env3.t2,  # T2E: bottom-right, attached to A_a
            progress_every=args.progress_every,
        )
        # Natural C3 corner order is (T1,T2), whereas the top row above is
        # ordered (T2,T1); transpose the upper corner to obtain C[Y,V].
        spectrum_13 = _spectrum(
            raw_13,
            ordering_ab.env1.corner.T.contiguous(),
            ordering_ba.env3.corner,
            corner_relative_cutoff=args.corner_relative_cutoff,
            eig_tol=args.eig_tol,
            arpack_ncv=args.arpack_ncv,
            arpack_maxiter=args.arpack_maxiter,
            seed=case_seed,
        )

        # env3(ab) x env1(ba):
        # F=C3(B) has axes [B_b,B_g,B_a] and is permuted to
        # [top=B_a, bond=B_g, bottom=B_b].
        # A[a,b,g] is permuted to [top=A_g, bond=A_a, bottom=A_b].
        raw_31 = CrossedEnvironmentTransferOperator(
            ordering_ab.env3.t2,  # T2E: top-left, attached to B_a
            ordering_ab.env3.t1,  # T1B: top-right, attached to A_g
            F.permute(2, 1, 0).contiguous(),
            A.permute(2, 0, 1).contiguous(),
            ordering_ba.env1.t1,  # T1F: bottom-left, attached to B_b
            ordering_ba.env1.t2,  # T2A: bottom-right, attached to A_b
            progress_every=args.progress_every,
        )
        spectrum_31 = _spectrum(
            raw_31,
            ordering_ab.env3.corner.T.contiguous(),
            ordering_ba.env1.corner,
            corner_relative_cutoff=args.corner_relative_cutoff,
            eig_tol=args.eig_tol,
            arpack_ncv=args.arpack_ncv,
            arpack_maxiter=args.arpack_maxiter,
            seed=(case_seed + 1) % 2**32,
        )

        xi_13_value = spectrum_13["correlation_length"]
        xi_31_value = spectrum_31["correlation_length"]
        inv_13 = float(spectrum_13["inverse_correlation_length"])
        inv_31 = float(spectrum_31["inverse_correlation_length"])
        if xi_13_value is None or xi_31_value is None:
            absolute_xi_difference = None
            relative_xi_difference = None
        else:
            xi_13 = float(xi_13_value)
            xi_31 = float(xi_31_value)
            absolute_xi_difference = abs(xi_13 - xi_31)
            relative_xi_difference = (
                2.0 * absolute_xi_difference / (abs(xi_13) + abs(xi_31))
            )
        inverse_denominator = abs(inv_13) + abs(inv_31)
        relative_inverse_xi_difference = (
            None
            if inverse_denominator == 0.0
            else 2.0 * abs(inv_13 - inv_31) / inverse_denominator
        )
        case = {
            "J2": j2,
            "D": D_bond,
            "chi": chi,
            "checkpoint": str(checkpoint),
            "seed": case_seed,
            "dtype": "torch.float64",
            "device": str(device),
            "rsvd_mode": rsvd_mode,
            "ctm_identity_init": not args.random_init,
            "ctm_steps_ab": ordering_ab.ctm_steps,
            "ctm_steps_ba": ordering_ba.ctm_steps,
            "ctm_energy_proxy_ab": ordering_ab.energy_proxy,
            "ctm_energy_proxy_ba": ordering_ba.energy_proxy,
            "env1_ab_x_env3_ba": spectrum_13,
            "env3_ab_x_env1_ba": spectrum_31,
            "comparison": {
                "absolute_xi_difference": absolute_xi_difference,
                "relative_xi_difference": relative_xi_difference,
                "absolute_inverse_xi_difference": abs(inv_13 - inv_31),
                "relative_inverse_xi_difference": (
                    relative_inverse_xi_difference
                ),
            },
            "existing_env2_baseline": _baseline(
                checkpoint.with_name("correlation_length.json")
            ),
            "elapsed_seconds": time.perf_counter() - started,
            "index_conventions": {
                "env1_x_env3": (
                    "B[top=b,bond=a,bottom=g] -- "
                    "C3(A)[top=A_b,bond=A_g,bottom=A_a]"
                ),
                "env3_x_env1": (
                    "C3(B)[top=B_a,bond=B_g,bottom=B_b] -- "
                    "A[top=g,bond=a,bottom=b]"
                ),
                "generalized_overlap": (
                    "N[Y,y,V,v] = C_upper[Y,V] C_lower[y,v] "
                    "= C_upper kron C_lower; "
                    "upper natural (T1,T2) corners are transposed because "
                    "both crossed top rows are ordered (T2,T1)."
                ),
            },
        }
        results[key] = case
        _atomic_write(output, document)
        print(
            f"[{index}/{len(jobs)}] DONE {key}: "
            f"xi13={xi_13_value}, xi31={xi_31_value}, "
            f"relative difference={relative_xi_difference}",
            flush=True,
        )

        del a, b, ordering_ab, ordering_ba, A, B, E, F
        corr._release_unused_memory(device)

    print(f"Saved {len(results)} result(s) to {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
