#!/usr/bin/env python3
"""Correlation length of a two-C3 iPEPS from a double-layer CTMRG boundary.

The row-to-row transfer matrix has shape ``(chi**2, chi**2)``.  It is never
materialized in production.  Instead, :class:`RowToRowTransferOperator`
contracts one vector at a time and bounds its largest intermediate by
chunking one boundary index.

The four edge tensors are obtained from two independent CTMRG runs:

* ``two-C3(a, b)`` supplies the two upper edges.
* ``two-C3(b, a)`` supplies the two lower edges.

Each run is followed by one explicit ``env1 -> env2`` update, as required to
put the two edge representatives in the orientation used by the transfer
operator.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import math
import os
import time
from collections.abc import Sequence
from typing import Any

import numpy as np
import opt_einsum as oe
import torch
from scipy.sparse.linalg import (
    ArpackNoConvergence,
    LinearOperator,
    eigs,
)

try:
    from . import core_C3 as _core
except ImportError:
    import core_C3 as _core


DEFAULT_CTM_MAX_STEPS = 70
DEFAULT_CTM_CONV_TOL = 1.0e-7
DEFAULT_IDENTITY_INIT = True
DEFAULT_RSVD_MODE = "augmented"
DEFAULT_RSVD_NEUMANN_TERMS = 2
DEFAULT_RSVD_POWER_ITERS: int | None = None
DEFAULT_MAX_INTERMEDIATE_MIB = 256.0
DEFAULT_ARPACK_NCV = 32
DEFAULT_ARPACK_MAXITER = 4000

_MIB = 1024**2


@dataclasses.dataclass(frozen=True)
class TransferComponents:
    """Tensors required by the row-to-row transfer operator.

    The names encode the sublattice carried by the physical edge and the
    boundary on which it is used.  All tensors are detached, real, and share
    one dtype and device.
    """

    upper_b: torch.Tensor
    upper_a: torch.Tensor
    lower_a: torch.Tensor
    lower_b: torch.Tensor
    double_layer_a: torch.Tensor
    double_layer_b: torch.Tensor
    ctm_steps_ab: int
    ctm_steps_ba: int

    @property
    def edges(self) -> tuple[torch.Tensor, ...]:
        """Return the four edges in transfer-contraction order."""

        return self.upper_b, self.upper_a, self.lower_a, self.lower_b


@dataclasses.dataclass(frozen=True)
class EigensolverResult:
    """Leading transfer eigenvalues and numerical diagnostics."""

    eigenvalues: tuple[complex, complex]
    relative_residuals: tuple[float, float]
    matvec_count: int
    used_dense_solver: bool
    elapsed_seconds: float


@dataclasses.dataclass(frozen=True)
class CorrelationLengthResult:
    """Complete result for one ``(D, chi)`` calculation."""

    correlation_length: float
    eigensolver: EigensolverResult
    ctm_steps_ab: int
    ctm_steps_ba: int
    D_bond: int
    chi: int


def _release_unused_memory(device: torch.device) -> None:
    """Release dead Python objects and unused CUDA allocator blocks."""

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _real_dtype_from_tensor(tensor: torch.Tensor) -> torch.dtype:
    if tensor.dtype == torch.float64:
        return torch.float64
    if tensor.dtype == torch.float32:
        return torch.float32
    raise TypeError(
        "Correlation-length tensors must use torch.float32 or torch.float64; "
        f"received {tensor.dtype}."
    )


def _prepare_raw_tensor(
    tensor: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if tensor.is_complex():
        raise TypeError(f"{name} must be real.")
    if tensor.ndim != 4:
        raise ValueError(
            f"{name} must have shape (D, D, D, d_phys); got {tuple(tensor.shape)}."
        )
    if not (tensor.shape[0] == tensor.shape[1] == tensor.shape[2]):
        raise ValueError(
            f"The three virtual dimensions of {name} must be equal; "
            f"got {tuple(tensor.shape[:3])}."
        )
    return tensor.detach().to(device=device, dtype=dtype).contiguous()


def _build_double_layers(
    raw_a: torch.Tensor,
    raw_b: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Build normalized double layers for one ordering of the two-C3 ansatz."""

    a = _core.normalize_single_layer_tensor_for_double_layer(raw_a)
    b = _core.normalize_single_layer_tensor_for_double_layer(raw_b)
    sites = _core.twoc3_abcdef_from_ab(a, b)
    D_squared = raw_a.shape[0] ** 2
    double_layers = _core.abcdef_to_ABCDEF(*sites, D_squared)
    del sites, a, b
    return tuple(tensor.detach().contiguous() for tensor in double_layers)


@torch.no_grad()
def _run_ctm_and_extract_edges(
    raw_a: torch.Tensor,
    raw_b: torch.Tensor,
    *,
    chi: int,
    ctm_max_steps: int,
    ctm_conv_tol: float,
    identity_init: bool,
    keep_ab: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, int]:
    """Run one CTMRG ordering and extract the post-update edge pair."""

    D_squared = raw_a.shape[0] ** 2
    double_layers = _build_double_layers(raw_a, raw_b)

    result = _core.CTMRG_from_init_to_stop(
        *double_layers,
        chi,
        D_squared,
        ctm_max_steps,
        ctm_conv_tol,
        identity_init,
        energy_proxy_fn=None,
    )
    corner, edge_1, edge_2 = result[:3]
    ctm_steps = int(result[-1])
    del result

    updated = _core.update_environmentCTs_1to2_C3(
        corner,
        edge_1,
        edge_2,
        *double_layers,
        chi,
        D_squared,
    )

    first = updated[1].detach().contiguous()
    second = updated[2].detach().contiguous()
    kept_a = double_layers[0].detach().contiguous() if keep_ab else None
    kept_b = double_layers[1].detach().contiguous() if keep_ab else None

    del corner, edge_1, edge_2, updated, double_layers
    _release_unused_memory(raw_a.device)
    return first, second, kept_a, kept_b, ctm_steps


def obtain_transfer_components(
    a: torch.Tensor,
    b: torch.Tensor,
    chi: int,
    *,
    ctm_max_steps: int = DEFAULT_CTM_MAX_STEPS,
    ctm_conv_tol: float = DEFAULT_CTM_CONV_TOL,
    identity_init: bool = DEFAULT_IDENTITY_INIT,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> TransferComponents:
    """Obtain four edge tensors and the original ``A, B`` double layers.

    The first CTMRG run uses ``(a, b)`` and returns the upper ``B, A`` edges.
    The second uses ``(b, a)`` and returns the lower ``A, B`` edges.  No tensor
    produced here has an autograd history.
    """

    if chi < 2:
        raise ValueError("chi must be at least 2 to define two eigenvalues.")
    if ctm_max_steps < 1:
        raise ValueError("ctm_max_steps must be positive.")
    if ctm_conv_tol <= 0.0:
        raise ValueError("ctm_conv_tol must be positive.")

    target_device = torch.device(device) if device is not None else a.device
    target_dtype = dtype if dtype is not None else _real_dtype_from_tensor(a)
    if target_dtype not in (torch.float32, torch.float64):
        raise TypeError("dtype must be torch.float32 or torch.float64.")

    raw_a = _prepare_raw_tensor(
        a, dtype=target_dtype, device=target_device, name="a"
    )
    raw_b = _prepare_raw_tensor(
        b, dtype=target_dtype, device=target_device, name="b"
    )
    if raw_a.shape != raw_b.shape:
        raise ValueError(
            "a and b must have identical shapes; "
            f"got {tuple(raw_a.shape)} and {tuple(raw_b.shape)}."
        )

    _core.set_dtype(target_dtype == torch.float64, use_real=True)
    _core.set_device(target_device)
    if target_device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
    _core._SVD_CPU_OFFLOAD_THRESHOLD = 0
    _core.set_rsvd_mode(
        DEFAULT_RSVD_MODE,
        neumann_terms=DEFAULT_RSVD_NEUMANN_TERMS,
        power_iters=DEFAULT_RSVD_POWER_ITERS,
    )

    if target_dtype == torch.float32:
        ctm_conv_tol = max(ctm_conv_tol, 1.0e-5)

    upper_b, upper_a, double_a, double_b, steps_ab = (
        _run_ctm_and_extract_edges(
            raw_a,
            raw_b,
            chi=chi,
            ctm_max_steps=ctm_max_steps,
            ctm_conv_tol=ctm_conv_tol,
            identity_init=identity_init,
            keep_ab=True,
        )
    )
    lower_a, lower_b, _, _, steps_ba = _run_ctm_and_extract_edges(
        raw_b,
        raw_a,
        chi=chi,
        ctm_max_steps=ctm_max_steps,
        ctm_conv_tol=ctm_conv_tol,
        identity_init=identity_init,
        keep_ab=False,
    )

    if double_a is None or double_b is None:
        raise RuntimeError("Internal error while retaining the A and B double layers.")

    del raw_a, raw_b
    _release_unused_memory(target_device)
    return TransferComponents(
        upper_b=upper_b,
        upper_a=upper_a,
        lower_a=lower_a,
        lower_b=lower_b,
        double_layer_a=double_a,
        double_layer_b=double_b,
        ctm_steps_ab=steps_ab,
        ctm_steps_ba=steps_ba,
    )


def obtain_4Ts(
    a: torch.Tensor,
    b: torch.Tensor,
    chi: int,
    *,
    ctm_max_steps: int = DEFAULT_CTM_MAX_STEPS,
    ctm_conv_tol: float = DEFAULT_CTM_CONV_TOL,
    identity_init: bool = DEFAULT_IDENTITY_INIT,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(upper_B, upper_A, lower_A, lower_B)`` edge tensors."""

    components = obtain_transfer_components(
        a,
        b,
        chi,
        ctm_max_steps=ctm_max_steps,
        ctm_conv_tol=ctm_conv_tol,
        identity_init=identity_init,
        dtype=dtype,
        device=device,
    )
    edges = components.edges
    del components
    _release_unused_memory(edges[0].device)
    return edges


class RowToRowTransferOperator:
    """Matrix-free row-to-row transfer matrix.

    The represented matrix is

    ``V[Y, v, y, V] =``
    ``sum upper_b[M,Y,A] upper_a[M,V,G] A[A,b,G] B[a,b,g]``
    ``    lower_a[m,y,a] lower_b[m,v,g]``

    and is reshaped as ``V[(Y,v), (y,V)]``.
    """

    def __init__(
        self,
        upper_b: torch.Tensor,
        upper_a: torch.Tensor,
        double_layer_a: torch.Tensor,
        double_layer_b: torch.Tensor,
        lower_a: torch.Tensor,
        lower_b: torch.Tensor,
        *,
        max_intermediate_bytes: int = int(
            DEFAULT_MAX_INTERMEDIATE_MIB * _MIB
        ),
        progress_every: int = 0,
    ) -> None:
        tensors = (
            upper_b,
            upper_a,
            double_layer_a,
            double_layer_b,
            lower_a,
            lower_b,
        )
        names = (
            "upper_b",
            "upper_a",
            "double_layer_a",
            "double_layer_b",
            "lower_a",
            "lower_b",
        )
        for name, tensor in zip(names, tensors, strict=True):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor.")
            if tensor.is_complex():
                raise TypeError(f"{name} must be real.")
            if tensor.requires_grad:
                tensor = tensor.detach()

        dtype = _real_dtype_from_tensor(upper_b)
        device = upper_b.device
        if any(tensor.dtype != dtype for tensor in tensors):
            raise TypeError("All transfer tensors must have the same dtype.")
        if any(tensor.device != device for tensor in tensors):
            raise ValueError("All transfer tensors must be on the same device.")

        if upper_b.ndim != 3:
            raise ValueError("Each edge tensor must have rank 3.")
        chi = upper_b.shape[0]
        q = upper_b.shape[2]
        expected_edge_shape = (chi, chi, q)
        for name, tensor in (
            ("upper_b", upper_b),
            ("upper_a", upper_a),
            ("lower_a", lower_a),
            ("lower_b", lower_b),
        ):
            if tuple(tensor.shape) != expected_edge_shape:
                raise ValueError(
                    f"{name} must have shape {expected_edge_shape}; "
                    f"got {tuple(tensor.shape)}."
                )
        expected_site_shape = (q, q, q)
        for name, tensor in (
            ("double_layer_a", double_layer_a),
            ("double_layer_b", double_layer_b),
        ):
            if tuple(tensor.shape) != expected_site_shape:
                raise ValueError(
                    f"{name} must have shape {expected_site_shape}; "
                    f"got {tuple(tensor.shape)}."
                )
        if max_intermediate_bytes <= 0:
            raise ValueError("max_intermediate_bytes must be positive.")
        if progress_every < 0:
            raise ValueError("progress_every cannot be negative.")

        self.upper_b = upper_b.detach().contiguous()
        self.upper_a = upper_a.detach().contiguous()
        self.double_layer_a = double_layer_a.detach().contiguous()
        self.double_layer_b = double_layer_b.detach().contiguous()
        self.lower_a = lower_a.detach().contiguous()
        self.lower_b = lower_b.detach().contiguous()
        self.chi = int(chi)
        self.q = int(q)
        self.shape = (self.chi**2, self.chi**2)
        self.dtype = dtype
        self.device = device
        self.numpy_dtype = np.float64 if dtype == torch.float64 else np.float32
        self.progress_every = int(progress_every)
        self.matvec_count = 0
        self._first_matvec_time = time.perf_counter()

        # A non-contiguous contraction can require one contiguous copy of the
        # logical intermediate, so reserve a factor of two in the chunk cap.
        bytes_per_slice = (
            2 * self.chi * self.q * self.q * upper_b.element_size()
        )
        self.chunk_size = max(
            1,
            min(self.chi, int(max_intermediate_bytes // bytes_per_slice)),
        )
        self.max_intermediate_bytes = int(max_intermediate_bytes)

    def _report_progress(self) -> None:
        if (
            self.progress_every > 0
            and self.matvec_count % self.progress_every == 0
        ):
            elapsed = time.perf_counter() - self._first_matvec_time
            print(
                f"  transfer matvec {self.matvec_count} "
                f"({elapsed:.1f} s elapsed)",
                flush=True,
            )

    @torch.no_grad()
    def _matvec_real_tensor(self, vector: torch.Tensor) -> torch.Tensor:
        """Apply the real transfer operator to one real vector."""

        x = vector.reshape(self.chi, self.chi)

        # x[y,V] lower_a[m,y,a] -> work[V,m,a]
        work0 = torch.tensordot(x, self.lower_a, dims=([0], [1]))
        output = torch.empty(
            self.chi,
            self.chi,
            dtype=self.dtype,
            device=self.device,
        )

        for start in range(0, self.chi, self.chunk_size):
            stop = min(start + self.chunk_size, self.chi)
            lower_b_chunk = self.lower_b[:, start:stop, :]

            # The only O(chi * chunk * q**2) intermediate.
            work1 = torch.tensordot(
                work0, lower_b_chunk, dims=([1], [0])
            )
            work2 = torch.tensordot(
                work1, self.double_layer_b, dims=([1, 3], [0, 2])
            )
            del work1

            work3 = torch.tensordot(
                work2, self.double_layer_a, dims=([2], [1])
            )
            del work2

            work4 = torch.tensordot(
                work3, self.upper_a, dims=([0, 3], [1, 2])
            )
            del work3

            work5 = torch.tensordot(
                work4, self.upper_b, dims=([1, 2], [2, 0])
            )
            del work4
            output[:, start:stop] = work5.transpose(0, 1)
            del work5

        del work0, x
        self.matvec_count += 1
        self._report_progress()
        return output.reshape(-1)

    def matvec_numpy(self, vector: np.ndarray) -> np.ndarray:
        """Apply the operator to a NumPy vector for SciPy ARPACK."""

        array = np.asarray(vector)
        if array.size != self.shape[1]:
            raise ValueError(
                f"Expected a vector of length {self.shape[1]}; got {array.size}."
            )

        if np.iscomplexobj(array):
            real_tensor = torch.as_tensor(
                np.ascontiguousarray(array.real),
                dtype=self.dtype,
                device=self.device,
            )
            imag_tensor = torch.as_tensor(
                np.ascontiguousarray(array.imag),
                dtype=self.dtype,
                device=self.device,
            )
            real_output = self._matvec_real_tensor(real_tensor)
            imag_output = self._matvec_real_tensor(imag_tensor)
            return (
                real_output.detach().cpu().numpy().astype(
                    self.numpy_dtype, copy=False
                )
                + 1j
                * imag_output.detach().cpu().numpy().astype(
                    self.numpy_dtype, copy=False
                )
            )

        real_tensor = torch.as_tensor(
            np.ascontiguousarray(array),
            dtype=self.dtype,
            device=self.device,
        )
        output = self._matvec_real_tensor(real_tensor)
        return output.detach().cpu().numpy().copy()

    @torch.no_grad()
    def to_dense(self, *, max_dense_bytes: int = 512 * _MIB) -> torch.Tensor:
        """Materialize the matrix for small validation problems only."""

        required = self.shape[0] * self.shape[1] * torch.empty(
            (), dtype=self.dtype
        ).element_size()
        if required > max_dense_bytes:
            raise MemoryError(
                "Refusing to materialize a "
                f"{self.shape[0]} x {self.shape[1]} matrix requiring "
                f"{required / _MIB:.1f} MiB."
            )

        dense = oe.contract(
            "MYA,MVG,AbG,abg,mya,mvg->YvyV",
            self.upper_b,
            self.upper_a,
            self.double_layer_a,
            self.double_layer_b,
            self.lower_a,
            self.lower_b,
            optimize="auto-hq",
            backend="torch",
        )
        return dense.reshape(self.shape)


def compute_r2rTransferMatrix(
    upper_b: torch.Tensor,
    upper_a: torch.Tensor,
    double_layer_a: torch.Tensor,
    double_layer_b: torch.Tensor,
    lower_a: torch.Tensor,
    lower_b: torch.Tensor,
    *,
    max_intermediate_bytes: int = int(
        DEFAULT_MAX_INTERMEDIATE_MIB * _MIB
    ),
    progress_every: int = 0,
    materialize: bool = False,
    max_dense_bytes: int = 512 * _MIB,
) -> RowToRowTransferOperator | torch.Tensor:
    """Build the matrix-free transfer operator.

    Set ``materialize=True`` only for small tests.  Production calculations
    should keep the default matrix-free return value.
    """

    operator = RowToRowTransferOperator(
        upper_b,
        upper_a,
        double_layer_a,
        double_layer_b,
        lower_a,
        lower_b,
        max_intermediate_bytes=max_intermediate_bytes,
        progress_every=progress_every,
    )
    if materialize:
        return operator.to_dense(max_dense_bytes=max_dense_bytes)
    return operator


def _relative_eigen_residual(
    operator: RowToRowTransferOperator,
    eigenvalue: complex,
    eigenvector: np.ndarray,
) -> tuple[complex, float]:
    """Refine one Ritz value and evaluate its scale-independent residual."""

    applied = operator.matvec_numpy(eigenvector)
    denominator = np.vdot(eigenvector, eigenvector)
    if abs(denominator) == 0.0:
        return complex(eigenvalue), math.inf

    refined = complex(np.vdot(eigenvector, applied) / denominator)
    vector_norm = float(np.linalg.norm(eigenvector))
    scale = float(np.linalg.norm(applied)) + abs(refined) * vector_norm
    residual = float(np.linalg.norm(applied - refined * eigenvector))
    relative = residual / max(scale, np.finfo(operator.numpy_dtype).tiny)
    return refined, relative


def _dense_leading_eigenpairs(
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    order = np.argsort(-np.abs(eigenvalues), kind="stable")
    return eigenvalues[order], eigenvectors[:, order]


def _diagonalize_operator(
    transfer: RowToRowTransferOperator,
    *,
    tol: float,
    ncv: int,
    maxiter: int,
    dense_dimension_threshold: int,
    seed: int,
) -> EigensolverResult:
    if tol < 0.0:
        raise ValueError("tol cannot be negative.")
    if maxiter < 1:
        raise ValueError("maxiter must be positive.")
    if dense_dimension_threshold < 0:
        raise ValueError("dense_dimension_threshold cannot be negative.")

    start = time.perf_counter()
    n = transfer.shape[0]
    initial_matvec_count = transfer.matvec_count
    used_dense = n <= dense_dimension_threshold

    if used_dense:
        dense = transfer.to_dense().detach().cpu().numpy()
        eigenvalues, eigenvectors = _dense_leading_eigenpairs(dense)
        del dense
    else:
        probe_count = min(4, n - 2)
        if probe_count < 2:
            raise ValueError(
                "The transfer matrix is too small for two sparse eigenvalues."
            )
        effective_ncv = min(n, max(ncv, 2 * probe_count + 1))
        if effective_ncv <= probe_count:
            raise ValueError("ncv must be larger than the requested eigenvalue count.")

        scipy_operator = LinearOperator(
            transfer.shape,
            matvec=transfer.matvec_numpy,
            dtype=transfer.numpy_dtype,
        )
        rng = np.random.default_rng(seed)
        initial = rng.standard_normal(n).astype(transfer.numpy_dtype, copy=False)
        initial /= np.linalg.norm(initial)

        try:
            eigenvalues, eigenvectors = eigs(
                scipy_operator,
                k=probe_count,
                which="LM",
                v0=initial,
                ncv=effective_ncv,
                maxiter=maxiter,
                tol=tol,
                return_eigenvectors=True,
            )
        except ArpackNoConvergence as error:
            converged = 0 if error.eigenvalues is None else len(error.eigenvalues)
            raise RuntimeError(
                "ARPACK did not converge to the requested leading spectrum "
                f"within {maxiter} iterations; {converged} Ritz values converged. "
                "Increase --arpack-ncv or --arpack-maxiter."
            ) from error

        order = np.argsort(-np.abs(eigenvalues), kind="stable")
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

    selected_values: list[complex] = []
    residuals: list[float] = []
    for index in range(2):
        refined, residual = _relative_eigen_residual(
            transfer,
            complex(eigenvalues[index]),
            eigenvectors[:, index],
        )
        selected_values.append(refined)
        residuals.append(residual)

    final_order = sorted(
        range(2), key=lambda index: abs(selected_values[index]), reverse=True
    )
    selected_values = [selected_values[index] for index in final_order]
    residuals = [residuals[index] for index in final_order]

    elapsed = time.perf_counter() - start
    return EigensolverResult(
        eigenvalues=(selected_values[0], selected_values[1]),
        relative_residuals=(residuals[0], residuals[1]),
        matvec_count=transfer.matvec_count - initial_matvec_count,
        used_dense_solver=used_dense,
        elapsed_seconds=elapsed,
    )


def diagonalize_first_two_largest_eigval(
    transfer_matrix: RowToRowTransferOperator | torch.Tensor | np.ndarray,
    *,
    tol: float = 0.0,
    ncv: int = DEFAULT_ARPACK_NCV,
    maxiter: int = DEFAULT_ARPACK_MAXITER,
    dense_dimension_threshold: int = 256,
    seed: int = 0,
    return_result: bool = False,
) -> torch.Tensor | EigensolverResult:
    """Return the two eigenvalues with largest absolute value.

    ARPACK is used on the real matrix-free operator.  ``tol=0`` asks ARPACK
    for machine precision.  Four Ritz values are requested internally so a
    complex-conjugate pair near the second-largest magnitude is not split at
    the selection boundary.
    """

    if isinstance(transfer_matrix, RowToRowTransferOperator):
        result = _diagonalize_operator(
            transfer_matrix,
            tol=tol,
            ncv=ncv,
            maxiter=maxiter,
            dense_dimension_threshold=dense_dimension_threshold,
            seed=seed,
        )
    else:
        if isinstance(transfer_matrix, torch.Tensor):
            dense = transfer_matrix.detach().cpu().numpy()
        else:
            dense = np.asarray(transfer_matrix)
        if dense.ndim == 4:
            side = dense.shape[0] * dense.shape[1]
            dense = dense.reshape(side, side)
        if dense.ndim != 2 or dense.shape[0] != dense.shape[1]:
            raise ValueError("A dense transfer matrix must be square.")
        if dense.shape[0] < 2:
            raise ValueError("At least a 2 x 2 transfer matrix is required.")

        start = time.perf_counter()
        eigenvalues, eigenvectors = _dense_leading_eigenpairs(dense)
        selected = tuple(complex(value) for value in eigenvalues[:2])
        residuals = []
        for index in range(2):
            vector = eigenvectors[:, index]
            applied = dense @ vector
            scale = (
                float(np.linalg.norm(applied))
                + abs(selected[index]) * float(np.linalg.norm(vector))
            )
            residuals.append(
                float(np.linalg.norm(applied - selected[index] * vector))
                / max(scale, np.finfo(dense.real.dtype).tiny)
            )
        result = EigensolverResult(
            eigenvalues=(selected[0], selected[1]),
            relative_residuals=(residuals[0], residuals[1]),
            matvec_count=0,
            used_dense_solver=True,
            elapsed_seconds=time.perf_counter() - start,
        )

    if return_result:
        return result

    if isinstance(transfer_matrix, RowToRowTransferOperator):
        is_double = transfer_matrix.dtype == torch.float64
    elif isinstance(transfer_matrix, torch.Tensor):
        is_double = transfer_matrix.dtype in (torch.float64, torch.complex128)
    else:
        is_double = np.asarray(transfer_matrix).dtype in (
            np.dtype(np.float64),
            np.dtype(np.complex128),
        )
    complex_dtype = torch.complex128 if is_double else torch.complex64
    return torch.tensor(result.eigenvalues, dtype=complex_dtype)


def correlation_length_from_eigenvalues(
    eigenvalues: Sequence[complex] | torch.Tensor | np.ndarray,
) -> float:
    """Evaluate ``1 / log(abs(lambda_1 / lambda_2))``."""

    if isinstance(eigenvalues, torch.Tensor):
        values = eigenvalues.detach().cpu().numpy().reshape(-1)
    else:
        values = np.asarray(eigenvalues).reshape(-1)
    if values.size < 2:
        raise ValueError("Two eigenvalues are required.")

    magnitudes = sorted(
        (float(abs(values[0])), float(abs(values[1]))), reverse=True
    )
    largest, second = magnitudes
    if not (math.isfinite(largest) and math.isfinite(second)):
        raise ValueError("Eigenvalue magnitudes must be finite.")
    if largest == 0.0:
        raise ValueError("The leading transfer eigenvalue is zero.")
    if second == 0.0:
        return 0.0

    logarithm = math.log(largest / second)
    if logarithm <= np.finfo(float).eps:
        return math.inf
    return 1.0 / logarithm


def obtain_per_D_correlation_length(
    a: torch.Tensor,
    b: torch.Tensor,
    chi: int,
    *,
    ctm_max_steps: int = DEFAULT_CTM_MAX_STEPS,
    ctm_conv_tol: float = DEFAULT_CTM_CONV_TOL,
    identity_init: bool = DEFAULT_IDENTITY_INIT,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
    max_intermediate_bytes: int = int(
        DEFAULT_MAX_INTERMEDIATE_MIB * _MIB
    ),
    eig_tol: float = 0.0,
    arpack_ncv: int = DEFAULT_ARPACK_NCV,
    arpack_maxiter: int = DEFAULT_ARPACK_MAXITER,
    dense_dimension_threshold: int = 256,
    seed: int = 0,
    progress_every: int = 0,
    return_result: bool = False,
) -> float | CorrelationLengthResult:
    """Run both CTMRG boundaries and compute one correlation length."""

    components = obtain_transfer_components(
        a,
        b,
        chi,
        ctm_max_steps=ctm_max_steps,
        ctm_conv_tol=ctm_conv_tol,
        identity_init=identity_init,
        dtype=dtype,
        device=device,
    )
    transfer = compute_r2rTransferMatrix(
        components.upper_b,
        components.upper_a,
        components.double_layer_a,
        components.double_layer_b,
        components.lower_a,
        components.lower_b,
        max_intermediate_bytes=max_intermediate_bytes,
        progress_every=progress_every,
    )
    if not isinstance(transfer, RowToRowTransferOperator):
        raise RuntimeError("Expected a matrix-free transfer operator.")

    eigensolver = diagonalize_first_two_largest_eigval(
        transfer,
        tol=eig_tol,
        ncv=arpack_ncv,
        maxiter=arpack_maxiter,
        dense_dimension_threshold=dense_dimension_threshold,
        seed=seed,
        return_result=True,
    )
    if not isinstance(eigensolver, EigensolverResult):
        raise RuntimeError("Expected detailed eigensolver output.")
    correlation_length = correlation_length_from_eigenvalues(
        eigensolver.eigenvalues
    )

    result = CorrelationLengthResult(
        correlation_length=correlation_length,
        eigensolver=eigensolver,
        ctm_steps_ab=components.ctm_steps_ab,
        ctm_steps_ba=components.ctm_steps_ba,
        D_bond=int(a.shape[0]),
        chi=int(chi),
    )
    del transfer, components
    _release_unused_memory(
        torch.device(device) if device is not None else a.device
    )
    return result if return_result else result.correlation_length


def _load_twoc3_checkpoint(
    path: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError("The checkpoint must contain a dictionary.")

    if "a_raw" in checkpoint and "b_raw" in checkpoint:
        a = checkpoint["a_raw"]
        b = checkpoint["b_raw"]
    elif "a" in checkpoint and "b" in checkpoint:
        a = checkpoint["a"]
        b = checkpoint["b"]
    elif "params" in checkpoint and len(checkpoint["params"]) == 2:
        a, b = checkpoint["params"]
    else:
        raise KeyError(
            "Could not find a two-C3 tensor pair. Expected keys "
            "('a_raw', 'b_raw'), ('a', 'b'), or a two-entry 'params' sequence."
        )

    a = _prepare_raw_tensor(a, dtype=dtype, device=device, name="a")
    b = _prepare_raw_tensor(b, dtype=dtype, device=device, name="b")
    metadata = {
        key: checkpoint.get(key)
        for key in ("D_bond", "chi", "loss", "energy", "step", "timestamp")
    }
    del checkpoint
    return a, b, metadata


def _complex_to_dict(value: complex, residual: float) -> dict[str, float]:
    return {
        "real": float(value.real),
        "imag": float(value.imag),
        "abs": float(abs(value)),
        "relative_residual": float(residual),
    }


def _result_to_dict(
    result: CorrelationLengthResult,
    *,
    checkpoint: str,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    return {
        "checkpoint": os.path.abspath(checkpoint),
        "D_bond": result.D_bond,
        "chi": result.chi,
        "device": str(device),
        "dtype": str(dtype),
        "ctm_steps_ab": result.ctm_steps_ab,
        "ctm_steps_ba": result.ctm_steps_ba,
        "eigenvalues": [
            _complex_to_dict(value, residual)
            for value, residual in zip(
                result.eigensolver.eigenvalues,
                result.eigensolver.relative_residuals,
                strict=True,
            )
        ],
        "correlation_length": result.correlation_length,
        "eigensolver_matvec_count": result.eigensolver.matvec_count,
        "eigensolver_used_dense": result.eigensolver.used_dense_solver,
        "eigensolver_seconds": result.eigensolver.elapsed_seconds,
    }


def _parse_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute a two-C3 iPEPS correlation length with a matrix-free "
            "row-to-row transfer operator."
        )
    )
    parser.add_argument("checkpoint", help="Path to a two-C3 PyTorch checkpoint.")
    parser.add_argument(
        "--chi",
        type=int,
        default=None,
        help="Environment dimension. Defaults to the checkpoint value.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device, for example cpu, cuda, or cuda:0. Default: auto.",
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="Use float32 instead of the default float64.",
    )
    parser.add_argument(
        "--ctm-max-steps",
        type=int,
        default=DEFAULT_CTM_MAX_STEPS,
    )
    parser.add_argument(
        "--ctm-conv-tol",
        type=float,
        default=DEFAULT_CTM_CONV_TOL,
    )
    parser.add_argument(
        "--random-init",
        action="store_true",
        help="Use the non-identity CTMRG initialization.",
    )
    parser.add_argument(
        "--max-intermediate-mib",
        type=float,
        default=DEFAULT_MAX_INTERMEDIATE_MIB,
        help="Target cap for the largest transfer contraction intermediate.",
    )
    parser.add_argument(
        "--eig-tol",
        type=float,
        default=0.0,
        help="ARPACK tolerance. Zero requests machine precision.",
    )
    parser.add_argument(
        "--arpack-ncv",
        type=int,
        default=DEFAULT_ARPACK_NCV,
    )
    parser.add_argument(
        "--arpack-maxiter",
        type=int,
        default=DEFAULT_ARPACK_MAXITER,
    )
    parser.add_argument(
        "--dense-threshold",
        type=int,
        default=256,
        help="Use dense diagonalization only at or below this matrix dimension.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print progress every N transfer matvecs; zero disables it.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    device = _parse_device(args.device)
    dtype = torch.float32 if args.single else torch.float64
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.set_num_threads(1)

    a, b, metadata = _load_twoc3_checkpoint(
        args.checkpoint,
        device=device,
        dtype=dtype,
    )
    chi = args.chi if args.chi is not None else metadata.get("chi")
    if chi is None:
        raise ValueError("--chi is required when the checkpoint has no chi value.")
    chi = int(chi)

    D_bond = int(a.shape[0])
    checkpoint_D = metadata.get("D_bond")
    if checkpoint_D is not None and int(checkpoint_D) != D_bond:
        raise ValueError(
            f"Checkpoint D_bond={checkpoint_D} disagrees with tensor D={D_bond}."
        )
    if args.max_intermediate_mib <= 0.0:
        raise ValueError("--max-intermediate-mib must be positive.")

    print(
        f"Computing correlation length for D={D_bond}, chi={chi}, "
        f"dtype={dtype}, device={device}.",
        flush=True,
    )
    print("Running CTMRG for the (a, b) and (b, a) boundaries.", flush=True)

    result = obtain_per_D_correlation_length(
        a,
        b,
        chi,
        ctm_max_steps=args.ctm_max_steps,
        ctm_conv_tol=args.ctm_conv_tol,
        identity_init=not args.random_init,
        dtype=dtype,
        device=device,
        max_intermediate_bytes=int(args.max_intermediate_mib * _MIB),
        eig_tol=args.eig_tol,
        arpack_ncv=args.arpack_ncv,
        arpack_maxiter=args.arpack_maxiter,
        dense_dimension_threshold=args.dense_threshold,
        seed=args.seed,
        progress_every=args.progress_every,
        return_result=True,
    )
    if not isinstance(result, CorrelationLengthResult):
        raise RuntimeError("Expected a detailed correlation-length result.")

    payload = _result_to_dict(
        result,
        checkpoint=args.checkpoint,
        device=device,
        dtype=dtype,
    )
    print(json.dumps(payload, indent=2, allow_nan=True), flush=True)

    if args.output is not None:
        output_path = os.path.abspath(args.output)
        output_directory = os.path.dirname(output_path)
        if output_directory:
            os.makedirs(output_directory, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, allow_nan=True)
            handle.write("\n")
        print(f"Saved {output_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
