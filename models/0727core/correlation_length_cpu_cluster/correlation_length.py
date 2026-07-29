#!/usr/bin/env python3
"""Correlation length of a two-C3 iPEPS from a double-layer CTMRG boundary.

The row-to-row transfer pencil has shape ``(chi**2, chi**2)``.  Its two
central double-layer sites lie side-by-side, so the matrix maps the right
boundary pair ``(V,v)`` to the left boundary pair ``(Y,y)``.  The two CTMRG
corners form the empty-row metric ``C_upper kron C_lower``.  Two-sided SVD
whitening applies the generalized pencil without explicitly forming or
inverting that Kronecker matrix.

The four edge tensors are obtained from two independent CTMRG runs:

* ``two-C3(a, b)`` supplies the two upper edges.
* ``two-C3(b, a)`` supplies the two lower edges.

Each run is followed by one explicit ``env1 -> env2`` update, as required to
put the two edge representatives in the orientation used by the transfer
operator.

By default each CTMRG run uses the same convergence path as the main_C3 LBFGS
phase: ``both`` corner-spectrum and full three-environment energy convergence,
with the same thresholds, identity initialization, couplings, and partial-SVD
configuration.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import math
import os
import secrets
import sys
import time
from collections.abc import Callable, Sequence
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
DEFAULT_CTM_CONV_MODE = "both"
DEFAULT_CTM_E_CONV_THRESHOLD = 2.0e-8
DEFAULT_IDENTITY_INIT = True
DEFAULT_RSVD_MODE = "augmented"
DEFAULT_RSVD_NEUMANN_TERMS = 2
DEFAULT_RSVD_POWER_ITERS: int | None = None
DEFAULT_J1 = 1.0
DEFAULT_J2 = 0.26
DEFAULT_MAX_INTERMEDIATE_MIB = 256.0
DEFAULT_ARPACK_NCV = 64
DEFAULT_ARPACK_MAXITER = 4000
DEFAULT_CORNER_RELATIVE_CUTOFF = 1.0e-14
DISABLED_CORRELATION_LENGTH_DS = frozenset({2})
FULL_SVD_CORRELATION_LENGTH_DS = frozenset({3, 4})

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
    ctm_energy_proxy_ab: float
    ctm_energy_proxy_ba: float
    ctm_corner_ab: torch.Tensor
    ctm_corner_ba: torch.Tensor
    ctm_corner_spectra_ab: tuple[tuple[float, ...], ...]
    ctm_corner_spectra_ba: tuple[tuple[float, ...], ...]

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
    ctm_energy_proxy_ab: float
    ctm_energy_proxy_ba: float
    ctm_corner_spectra_ab: tuple[tuple[float, ...], ...]
    ctm_corner_spectra_ba: tuple[tuple[float, ...], ...]
    D_bond: int
    chi: int
    transfer_mode: str
    transfer_matrix_gib: float
    corner_effective_ranks: tuple[int, int]
    overlap_condition_number: float


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


def _build_ctm_layers(
    raw_a: torch.Tensor,
    raw_b: torch.Tensor,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """Build normalized single and double layers for one two-C3 ordering."""

    a = _core.normalize_single_layer_tensor_for_double_layer(raw_a)
    b = _core.normalize_single_layer_tensor_for_double_layer(raw_b)
    sites = tuple(
        tensor.detach().contiguous()
        for tensor in _core.twoc3_abcdef_from_ab(a, b)
    )
    D_squared = raw_a.shape[0] ** 2
    double_layers = tuple(
        tensor.detach().contiguous()
        for tensor in _core.abcdef_to_ABCDEF(*sites, D_squared)
    )
    del a, b
    return sites, double_layers


def _build_lbfgs_energy_proxy(
    sites: tuple[torch.Tensor, ...],
    *,
    chi: int,
    D_bond: int,
    j1: float,
    j2: float,
) -> tuple[Callable[..., float], list[float | None]]:
    """Build the exact three-environment energy proxy used by main_C3 LBFGS."""

    if len(sites) != 6:
        raise ValueError("The energy proxy requires six normalized site tensors.")
    a, b, c, d, e, f = sites
    d_phys = int(a.shape[-1])
    spin_dot_spin = _core.build_heisenberg_H(1.0, d_phys)
    couplings = ([float(j1)] * 6 + [float(j2)] * 6) * 3
    last_value: list[float | None] = [None]

    @torch.no_grad()
    def energy_proxy(
        C21CD: torch.Tensor,
        T1F: torch.Tensor,
        T2A: torch.Tensor,
        C21EB: torch.Tensor,
        T1D: torch.Tensor,
        T2C: torch.Tensor,
        C21AF: torch.Tensor,
        T1B: torch.Tensor,
        T2E: torch.Tensor,
    ) -> float:
        energy_1 = _core.energy_expectation_nearest_neighbor_3ebadcf_bonds(
            a,
            b,
            c,
            d,
            e,
            f,
            *couplings[0:12],
            spin_dot_spin,
            chi,
            D_bond,
            d_phys,
            C21CD,
            T1F,
            T2A,
        )
        energy_2 = _core.energy_expectation_nearest_neighbor_3afcbed_bonds(
            a,
            b,
            c,
            d,
            e,
            f,
            *couplings[12:24],
            spin_dot_spin,
            chi,
            D_bond,
            d_phys,
            C21EB,
            T1D,
            T2C,
        )
        energy_3 = _core.energy_expectation_nearest_neighbor_other_3_bonds(
            a,
            b,
            c,
            d,
            e,
            f,
            *couplings[24:36],
            spin_dot_spin,
            chi,
            D_bond,
            d_phys,
            C21AF,
            T1B,
            T2E,
        )
        value = float((energy_1 + energy_2 + energy_3).item())
        last_value[0] = value
        return value

    return energy_proxy, last_value


@torch.no_grad()
def _run_ctm_and_extract_edges(
    raw_a: torch.Tensor,
    raw_b: torch.Tensor,
    *,
    chi: int,
    ctm_max_steps: int,
    ctm_conv_tol: float,
    identity_init: bool,
    j1: float,
    j2: float,
    keep_ab: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    int,
    float,
    torch.Tensor,
    tuple[tuple[float, ...], ...],
]:
    """Run one CTMRG ordering and extract the post-update edge pair."""

    D_bond = int(raw_a.shape[0])
    D_squared = raw_a.shape[0] ** 2
    sites, double_layers = _build_ctm_layers(raw_a, raw_b)
    energy_proxy, last_energy_proxy = _build_lbfgs_energy_proxy(
        sites,
        chi=chi,
        D_bond=D_bond,
        j1=j1,
        j2=j2,
    )

    result = _core.CTMRG_from_init_to_stop(
        *double_layers,
        chi,
        D_squared,
        ctm_max_steps,
        ctm_conv_tol,
        identity_init,
        energy_proxy_fn=energy_proxy,
    )
    final_energy_proxy = float(energy_proxy(*result[:9]))
    corner_spectra: list[tuple[float, ...]] = []
    for corner_index in (0, 3, 6):
        corner_tensor = result[corner_index]
        corner_rho = corner_tensor @ corner_tensor @ corner_tensor
        singular_values = torch.linalg.svdvals(corner_rho).real
        singular_values = singular_values / (singular_values[0] + 1.0e-30)
        corner_spectra.append(
            tuple(float(value) for value in singular_values.detach().cpu())
        )
    frozen_corner_spectra = tuple(corner_spectra)

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
    updated_corner = updated[0].detach().contiguous()
    kept_a = double_layers[0].detach().contiguous() if keep_ab else None
    kept_b = double_layers[1].detach().contiguous() if keep_ab else None

    del (
        corner,
        edge_1,
        edge_2,
        updated,
        double_layers,
        energy_proxy,
        last_energy_proxy,
        sites,
    )
    _release_unused_memory(raw_a.device)
    return (
        first,
        second,
        kept_a,
        kept_b,
        ctm_steps,
        final_energy_proxy,
        updated_corner,
        frozen_corner_spectra,
    )


def obtain_transfer_components(
    a: torch.Tensor,
    b: torch.Tensor,
    chi: int,
    *,
    ctm_max_steps: int = DEFAULT_CTM_MAX_STEPS,
    ctm_conv_tol: float = DEFAULT_CTM_CONV_TOL,
    ctm_conv_mode: str = DEFAULT_CTM_CONV_MODE,
    ctm_e_conv_threshold: float = DEFAULT_CTM_E_CONV_THRESHOLD,
    identity_init: bool = DEFAULT_IDENTITY_INIT,
    rsvd_mode: str = DEFAULT_RSVD_MODE,
    rsvd_neumann_terms: int = DEFAULT_RSVD_NEUMANN_TERMS,
    rsvd_power_iters: int | None = DEFAULT_RSVD_POWER_ITERS,
    force_full_svd: bool = False,
    j1: float = DEFAULT_J1,
    j2: float = DEFAULT_J2,
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
    if ctm_conv_mode not in ("SVdifference", "Edifference", "both"):
        raise ValueError(f"Unsupported CTMRG convergence mode: {ctm_conv_mode}.")
    if ctm_e_conv_threshold <= 0.0:
        raise ValueError("ctm_e_conv_threshold must be positive.")

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
    _core._USE_FULL_SVD = bool(force_full_svd)
    _core.set_rsvd_mode(
        rsvd_mode,
        neumann_terms=rsvd_neumann_terms,
        power_iters=rsvd_power_iters,
    )
    _core.set_ctm_conv_mode(
        ctm_conv_mode,
        e_threshold=ctm_e_conv_threshold,
    )

    if target_dtype == torch.float32:
        ctm_conv_tol = max(ctm_conv_tol, 1.0e-5)

    # In update_environmentCTs_1to2_C3, output 1 is T1D and is built from
    # double-layer site D, while output 2 is T2C and is built from site C.
    # For two-C3(a,b), D belongs to the b orbit and C belongs to the a orbit.
    # Therefore the returned pair is exactly (upper_b, upper_a).
    #
    # The T1D einsum spells its output as YMa in the old frame.  This is a C3
    # frame relabel, not a pending transpose: update_environmentCTs_2to3_C3
    # consumes that returned tensor unchanged as its canonical MYa edge.
    (
        upper_b,
        upper_a,
        double_a,
        double_b,
        steps_ab,
        energy_proxy_ab,
        corner_ab,
        corner_spectra_ab,
    ) = (
        _run_ctm_and_extract_edges(
            raw_a,
            raw_b,
            chi=chi,
            ctm_max_steps=ctm_max_steps,
            ctm_conv_tol=ctm_conv_tol,
            identity_init=identity_init,
            j1=j1,
            j2=j2,
            keep_ab=True,
        )
    )
    # For two-C3(b,a), D belongs to the original a orbit and C belongs to the
    # original b orbit.  The same T1D,T2C output positions therefore supply
    # the A,B edges needed below the row.  Their first axes are the shared
    # boundary index and their second axes are the two open corner indices,
    # exactly as required by T_A[m,y,a] and T_B[m,v,g].  Transposing these
    # tensors would instead contract the two inequivalent corner axes.
    (
        raw_lower_a,
        raw_lower_b,
        _,
        _,
        steps_ba,
        energy_proxy_ba,
        corner_ba,
        corner_spectra_ba,
    ) = (
        _run_ctm_and_extract_edges(
            raw_b,
            raw_a,
            chi=chi,
            ctm_max_steps=ctm_max_steps,
            ctm_conv_tol=ctm_conv_tol,
            identity_init=identity_init,
            j1=j1,
            j2=j2,
            keep_ab=False,
        )
    )
    lower_a = raw_lower_a
    lower_b = raw_lower_b

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
        ctm_energy_proxy_ab=energy_proxy_ab,
        ctm_energy_proxy_ba=energy_proxy_ba,
        ctm_corner_ab=corner_ab,
        ctm_corner_ba=corner_ba,
        ctm_corner_spectra_ab=corner_spectra_ab,
        ctm_corner_spectra_ba=corner_spectra_ba,
    )


def obtain_4Ts(
    a: torch.Tensor,
    b: torch.Tensor,
    chi: int,
    *,
    ctm_max_steps: int = DEFAULT_CTM_MAX_STEPS,
    ctm_conv_tol: float = DEFAULT_CTM_CONV_TOL,
    ctm_conv_mode: str = DEFAULT_CTM_CONV_MODE,
    ctm_e_conv_threshold: float = DEFAULT_CTM_E_CONV_THRESHOLD,
    identity_init: bool = DEFAULT_IDENTITY_INIT,
    rsvd_mode: str = DEFAULT_RSVD_MODE,
    rsvd_neumann_terms: int = DEFAULT_RSVD_NEUMANN_TERMS,
    rsvd_power_iters: int | None = DEFAULT_RSVD_POWER_ITERS,
    force_full_svd: bool = False,
    j1: float = DEFAULT_J1,
    j2: float = DEFAULT_J2,
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
        ctm_conv_mode=ctm_conv_mode,
        ctm_e_conv_threshold=ctm_e_conv_threshold,
        identity_init=identity_init,
        rsvd_mode=rsvd_mode,
        rsvd_neumann_terms=rsvd_neumann_terms,
        rsvd_power_iters=rsvd_power_iters,
        force_full_svd=force_full_svd,
        j1=j1,
        j2=j2,
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

        self.chi = int(chi)
        self.q = int(q)
        self.shape = (self.chi**2, self.chi**2)
        self.dtype = dtype
        self.device = device
        self.numpy_dtype = np.float64 if dtype == torch.float64 else np.float32
        self.progress_every = int(progress_every)
        self.matvec_count = 0
        self._first_matvec_time = time.perf_counter()
        self.max_intermediate_bytes = int(max_intermediate_bytes)

        # These two exact triangle contractions are independent of the Krylov
        # vector and are therefore computed only once.
        self.upper_half, self.lower_half, self.chunk_size = (
            _build_boundary_halves(
                upper_b,
                upper_a,
                double_layer_a,
                double_layer_b,
                lower_a,
                lower_b,
                max_intermediate_bytes=max_intermediate_bytes,
            )
        )

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
        # lower_half[b,y,v] x[y,V] -> work[b,v,V]
        work = torch.tensordot(
            self.lower_half,
            x,
            dims=([1], [0]),
        )
        # work[b,v,V] upper_half[b,V,Y] -> output[v,Y]
        output = torch.tensordot(
            work,
            self.upper_half,
            dims=([0, 2], [0, 1]),
        ).transpose(0, 1)
        del work, x
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

        # upper_half[b,V,Y] lower_half[b,y,v] -> work[V,Y,y,v]
        work = torch.tensordot(
            self.upper_half,
            self.lower_half,
            dims=([0], [0]),
        )
        dense = work.permute(1, 3, 2, 0).contiguous()
        del work
        return dense.reshape(self.shape)


class DenseRowToRowTransferOperator(RowToRowTransferOperator):
    """Dense real transfer matrix with GPU-backed ARPACK matvecs."""

    def __init__(
        self,
        matrix: torch.Tensor,
        *,
        chi: int,
        progress_every: int = 0,
    ) -> None:
        if not isinstance(matrix, torch.Tensor):
            raise TypeError("matrix must be a torch.Tensor.")
        if matrix.is_complex():
            raise TypeError("The dense transfer matrix must be real.")
        dtype = _real_dtype_from_tensor(matrix)
        expected_shape = (chi**2, chi**2)
        if tuple(matrix.shape) != expected_shape:
            raise ValueError(
                f"matrix must have shape {expected_shape}; got {tuple(matrix.shape)}."
            )
        if progress_every < 0:
            raise ValueError("progress_every cannot be negative.")

        self.matrix = matrix.detach().contiguous()
        self.chi = int(chi)
        self.q = 0
        self.shape = expected_shape
        self.dtype = dtype
        self.device = matrix.device
        self.numpy_dtype = np.float64 if dtype == torch.float64 else np.float32
        self.progress_every = int(progress_every)
        self.matvec_count = 0
        self._first_matvec_time = time.perf_counter()
        self.chunk_size = self.chi
        self.max_intermediate_bytes = 0

    @torch.no_grad()
    def _matvec_real_tensor(self, vector: torch.Tensor) -> torch.Tensor:
        output = self.matrix @ vector.reshape(-1)
        self.matvec_count += 1
        self._report_progress()
        return output

    @torch.no_grad()
    def to_dense(self, *, max_dense_bytes: int | None = None) -> torch.Tensor:
        if max_dense_bytes is not None:
            required = self.matrix.numel() * self.matrix.element_size()
            if required > max_dense_bytes:
                raise MemoryError(
                    f"The resident matrix requires {required / _MIB:.1f} MiB."
                )
        return self.matrix


class SideBySideRowToRowTransferOperator(RowToRowTransferOperator):
    """Correct six-tensor row transfer with horizontal central-site bond.

    In canonical local coordinates the central sites are
    ``left[top, bond, bottom]`` and ``right[top, bond, bottom]``.  This
    operator represents

    ``V[Y,y,V,v] = sum``
    ``upper_left[M,Y,i] upper_right[M,V,j]``
    ``left[i,h,k] right[j,h,l]``
    ``lower_left[m,y,k] lower_right[m,v,l]``.

    Thus rows are the left boundary ``(Y,y)`` and columns are the right
    boundary ``(V,v)``.  The former production contraction joined the two
    central sites vertically and matricized opposite diagonals of the four
    open boundary legs; that is a different tensor network, not a gauge or
    local-leg convention for this row transfer.
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
        max_intermediate_bytes: int = int(
            DEFAULT_MAX_INTERMEDIATE_MIB * _MIB
        ),
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
        names = (
            "upper_left",
            "upper_right",
            "left_site",
            "right_site",
            "lower_left",
            "lower_right",
        )
        for name, tensor in zip(names, tensors, strict=True):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor.")
            if tensor.is_complex():
                raise TypeError(f"{name} must be real.")

        dtype = _real_dtype_from_tensor(upper_left)
        device = upper_left.device
        if any(tensor.dtype != dtype for tensor in tensors):
            raise TypeError("All transfer tensors must have the same dtype.")
        if any(tensor.device != device for tensor in tensors):
            raise ValueError("All transfer tensors must be on the same device.")
        if upper_left.ndim != 3:
            raise ValueError("Each edge tensor must have rank 3.")

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
        for name, tensor in (
            ("left_site", left_site),
            ("right_site", right_site),
        ):
            if tuple(tensor.shape) != site_shape:
                raise ValueError(
                    f"{name} must have shape {site_shape}; "
                    f"got {tuple(tensor.shape)}."
                )
        if max_intermediate_bytes <= 0:
            raise ValueError("max_intermediate_bytes must be positive.")
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
        self.max_intermediate_bytes = int(max_intermediate_bytes)
        self.chunk_size = q

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
        # d[M,i,y] upper_left[M,Y,i] -> output[Y,y]
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
    def to_dense(self, *, max_dense_bytes: int = 512 * _MIB) -> torch.Tensor:
        """Materialize the side-by-side transfer in bounded column batches."""

        required = (
            self.shape[0]
            * self.shape[1]
            * torch.empty((), dtype=self.dtype).element_size()
        )
        if required > max_dense_bytes:
            raise MemoryError(
                f"Dense transfer would require {required / _MIB:.1f} MiB."
            )

        dense = torch.empty(
            self.shape, dtype=self.dtype, device=self.device
        )
        # The two largest batched intermediates have approximately
        # chi**2 * q**2 * batch elements.  Keep both within the requested
        # intermediate-memory budget.
        bytes_per_column = max(
            1,
            2
            * self.chi**2
            * self.q**2
            * torch.empty((), dtype=self.dtype).element_size(),
        )
        batch_size = max(
            1,
            min(self.shape[1], self.max_intermediate_bytes // bytes_per_column),
        )
        for start in range(0, self.shape[1], batch_size):
            stop = min(start + batch_size, self.shape[1])
            basis = torch.zeros(
                self.shape[1],
                stop - start,
                dtype=self.dtype,
                device=self.device,
            )
            indices = torch.arange(
                start, stop, dtype=torch.long, device=self.device
            )
            basis[indices, torch.arange(stop - start, device=self.device)] = 1
            x = basis.reshape(self.chi, self.chi, stop - start)
            # The following is the matvec contraction with an extra batch
            # index K carried through every intermediate.
            a = torch.einsum("MVj,VvK->MjvK", self.upper_right, x)
            b = torch.einsum("MjvK,mvl->MjmlK", a, self.lower_right)
            del a
            c = torch.einsum("MjmlK,ikjl->MmikK", b, self.central_pair)
            del b
            d = torch.einsum("MmikK,myk->MiyK", c, self.lower_left)
            del c
            output = torch.einsum("MiyK,MYi->YyK", d, self.upper_left)
            del d
            dense[:, start:stop] = output.reshape(
                self.shape[0], stop - start
            )
            del basis, indices, x, output
        return dense


class KroneckerCornerWhitenedTransferOperator(RowToRowTransferOperator):
    """Standard form of ``T x = lambda (C_upper kron C_lower) x``.

    Raw row and column indices are respectively ``(Y,y)`` and ``(V,v)``:

    ``N[Y,y,V,v] = C_upper[Y,V] C_lower[y,v]``.
    """

    def __init__(
        self,
        raw_transfer: RowToRowTransferOperator,
        upper_corner: torch.Tensor,
        lower_corner: torch.Tensor,
        *,
        relative_cutoff: float = DEFAULT_CORNER_RELATIVE_CUTOFF,
    ) -> None:
        if relative_cutoff < 0.0:
            raise ValueError("relative_cutoff cannot be negative.")
        chi = int(raw_transfer.chi)
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
        self.max_intermediate_bytes = raw_transfer.max_intermediate_bytes

    @torch.no_grad()
    def _right_whiten_real_tensor(self, vector: torch.Tensor) -> torch.Tensor:
        canonical = vector.reshape(self.chi, self.chi)
        weighted = self.inverse_sqrt_weights * canonical
        return self.upper_v @ weighted @ self.lower_v.T

    @torch.no_grad()
    def _matvec_real_tensor(self, vector: torch.Tensor) -> torch.Tensor:
        raw_input = self._right_whiten_real_tensor(vector)
        raw_output = self.raw_transfer._matvec_real_tensor(
            raw_input.reshape(-1)
        ).reshape(self.chi, self.chi)
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
            if abs(eigenvalue) <= 100.0 * np.finfo(self.numpy_dtype).eps:
                return 0.0j, 0.0
            return complex(eigenvalue), math.inf
        refined = complex(np.vdot(empty_row, applied) / denominator)
        scale = float(np.linalg.norm(applied)) + abs(refined) * float(
            np.linalg.norm(empty_row)
        )
        residual = float(np.linalg.norm(applied - refined * empty_row))
        relative = residual / max(scale, np.finfo(self.numpy_dtype).tiny)
        return refined, relative

    @torch.no_grad()
    def to_dense(self, *, max_dense_bytes: int = 512 * _MIB) -> torch.Tensor:
        required = (
            self.shape[0]
            * self.shape[1]
            * torch.empty((), dtype=self.dtype).element_size()
        )
        if required > max_dense_bytes:
            raise MemoryError(
                f"The whitened dense matrix needs {required / _MIB:.1f} MiB."
            )
        identity = torch.eye(
            self.shape[1], dtype=self.dtype, device=self.device
        )
        columns = [
            self._matvec_real_tensor(identity[:, index])
            for index in range(self.shape[1])
        ]
        return torch.stack(columns, dim=1)


class CornerWhitenedTransferOperator(RowToRowTransferOperator):
    """Gauge-invariant standard form of the generalized transfer pencil.

    The raw six-tensor network maps the numerical basis ``(y,V)`` to
    ``(Y,v)``.  These bases belong to two independent CTMRG runs and must not
    be identified by a Kronecker delta.  Their empty-row map is

    ``N[Y,v,y,V] = C_ab[Y,V] C_ba[y,v]``.

    SVD whitening on both sides converts ``T x = lambda N x`` to an ordinary
    eigenproblem without constructing or inverting the ``chi**2`` matrix N.
    Numerically null corner modes are mapped to zero with a Moore-Penrose
    cutoff.
    """

    def __init__(
        self,
        raw_transfer: RowToRowTransferOperator,
        corner_ab: torch.Tensor,
        corner_ba: torch.Tensor,
        *,
        relative_cutoff: float = DEFAULT_CORNER_RELATIVE_CUTOFF,
    ) -> None:
        if relative_cutoff < 0.0:
            raise ValueError("relative_cutoff cannot be negative.")
        chi = int(raw_transfer.chi)
        expected = (chi, chi)
        if tuple(corner_ab.shape) != expected or tuple(corner_ba.shape) != expected:
            raise ValueError(f"Both corners must have shape {expected}.")
        if corner_ab.dtype != raw_transfer.dtype or corner_ba.dtype != raw_transfer.dtype:
            raise TypeError("Corners and transfer operator must share one dtype.")
        if (
            corner_ab.device != raw_transfer.device
            or corner_ba.device != raw_transfer.device
        ):
            raise ValueError("Corners and transfer operator must share one device.")

        upper_left, upper_s, upper_vh = torch.linalg.svd(
            corner_ab.detach(), full_matrices=False
        )
        lower_left, lower_s, lower_vh = torch.linalg.svd(
            corner_ba.detach(), full_matrices=False
        )
        upper_threshold = relative_cutoff * float(upper_s[0])
        lower_threshold = relative_cutoff * float(lower_s[0])
        upper_kept = upper_s > upper_threshold
        lower_kept = lower_s > lower_threshold
        upper_inverse_sqrt = torch.zeros_like(upper_s)
        lower_inverse_sqrt = torch.zeros_like(lower_s)
        upper_inverse_sqrt[upper_kept] = torch.rsqrt(upper_s[upper_kept])
        lower_inverse_sqrt[lower_kept] = torch.rsqrt(lower_s[lower_kept])

        self.raw_transfer = raw_transfer
        self.upper_left = upper_left
        self.upper_right = upper_vh.transpose(0, 1).contiguous()
        self.lower_left = lower_left
        self.lower_right = lower_vh.transpose(0, 1).contiguous()
        self.inverse_sqrt_weights = (
            upper_inverse_sqrt[:, None] * lower_inverse_sqrt[None, :]
        )
        self.corner_ab_numpy = corner_ab.detach().cpu().numpy()
        self.corner_ba_numpy = corner_ba.detach().cpu().numpy()
        self.corner_effective_ranks = (
            int(upper_kept.sum().item()),
            int(lower_kept.sum().item()),
        )
        retained_upper = upper_s[upper_kept]
        retained_lower = lower_s[lower_kept]
        if retained_upper.numel() == 0 or retained_lower.numel() == 0:
            raise RuntimeError("The CTMRG empty-row map has zero numerical rank.")
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
        self.max_intermediate_bytes = raw_transfer.max_intermediate_bytes

    @torch.no_grad()
    def _right_whiten_real_tensor(self, vector: torch.Tensor) -> torch.Tensor:
        canonical = vector.reshape(self.chi, self.chi)
        weighted = self.inverse_sqrt_weights * canonical
        return (
            self.lower_left @ weighted.transpose(0, 1) @ self.upper_right.T
        )

    @torch.no_grad()
    def _matvec_real_tensor(self, vector: torch.Tensor) -> torch.Tensor:
        # Right whitening: canonical (a,b) -> raw input (y,V).
        raw_input = self._right_whiten_real_tensor(vector)
        raw_output = self.raw_transfer._matvec_real_tensor(
            raw_input.reshape(-1)
        ).reshape(self.chi, self.chi)

        # Left whitening: raw output (Y,v) -> canonical (a,b).
        output = (
            self.upper_left.T @ raw_output @ self.lower_right
        ) * self.inverse_sqrt_weights
        self.matvec_count += 1
        return output.reshape(-1)

    def generalized_refine_numpy(
        self,
        eigenvalue: complex,
        eigenvector: np.ndarray,
    ) -> tuple[complex, float]:
        """Refine an eigenvalue and evaluate the original pencil residual."""

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

        raw_flat = np.ascontiguousarray(raw_input.reshape(-1))
        applied = self.raw_transfer.matvec_numpy(raw_flat).reshape(
            self.chi, self.chi
        )
        empty_row = (
            self.corner_ab_numpy
            @ raw_input.T
            @ self.corner_ba_numpy
        )
        denominator = np.vdot(empty_row, empty_row)
        if abs(denominator) == 0.0:
            if abs(eigenvalue) <= 100.0 * np.finfo(self.numpy_dtype).eps:
                return 0.0j, 0.0
            return complex(eigenvalue), math.inf
        refined = complex(np.vdot(empty_row, applied) / denominator)
        scale = float(np.linalg.norm(applied)) + abs(refined) * float(
            np.linalg.norm(empty_row)
        )
        residual = float(np.linalg.norm(applied - refined * empty_row))
        relative = residual / max(scale, np.finfo(self.numpy_dtype).tiny)
        return refined, relative

    @torch.no_grad()
    def to_dense(self, *, max_dense_bytes: int = 512 * _MIB) -> torch.Tensor:
        required = self.shape[0] * self.shape[1] * torch.empty(
            (), dtype=self.dtype
        ).element_size()
        if required > max_dense_bytes:
            raise MemoryError(
                f"The whitened dense matrix requires {required / _MIB:.1f} MiB."
            )
        identity = torch.eye(
            self.shape[1], dtype=self.dtype, device=self.device
        )
        columns = [
            self._matvec_real_tensor(identity[:, index])
            for index in range(self.shape[1])
        ]
        return torch.stack(columns, dim=1)


@torch.no_grad()
def _build_boundary_halves(
    upper_b: torch.Tensor,
    upper_a: torch.Tensor,
    double_layer_a: torch.Tensor,
    double_layer_b: torch.Tensor,
    lower_a: torch.Tensor,
    lower_b: torch.Tensor,
    *,
    max_intermediate_bytes: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Contract the upper and lower three-tensor triangles.

    The retained middle index ``b`` is axis 1 of both double-layer sites:

    ``upper_half[b,V,Y] = sum_{A,G,M} A[A,b,G] T_A[M,V,G] T_B[M,Y,A]``

    ``lower_half[b,y,v] = sum_{a,g,m} B[a,b,g] T_A[m,y,a] T_B[m,v,g]``
    """

    chi = int(upper_b.shape[0])
    q = int(upper_b.shape[2])
    element_size = upper_b.element_size()

    # The first contraction has shape (chi, chi, q, b_chunk).  Reserve a
    # factor of two for a possible contiguous workspace.
    bytes_per_b = 2 * chi * chi * q * element_size
    b_chunk = max(
        1,
        min(q, int(max_intermediate_bytes // bytes_per_b)),
    )

    upper_half = torch.empty(
        q, chi, chi, dtype=upper_b.dtype, device=upper_b.device
    )
    lower_half = torch.empty_like(upper_half)

    for start in range(0, q, b_chunk):
        stop = min(start + b_chunk, q)

        # upper_a[M,V,G] A[A,b,G] -> work[M,V,A,b]
        work = torch.tensordot(
            upper_a,
            double_layer_a[:, start:stop, :],
            dims=([2], [2]),
        )
        # work[M,V,A,b] upper_b[M,Y,A] -> result[V,b,Y]
        result = torch.tensordot(
            work,
            upper_b,
            dims=([0, 2], [0, 2]),
        )
        del work
        upper_half[start:stop] = result.permute(1, 0, 2)
        del result

        # lower_a[m,y,a] B[a,b,g] -> work[m,y,b,g]
        work = torch.tensordot(
            lower_a,
            double_layer_b[:, start:stop, :],
            dims=([2], [0]),
        )
        # work[m,y,b,g] lower_b[m,v,g] -> result[y,b,v]
        result = torch.tensordot(
            work,
            lower_b,
            dims=([0, 3], [0, 2]),
        )
        del work
        lower_half[start:stop] = result.permute(1, 0, 2)
        del result

    return upper_half, lower_half, b_chunk


@torch.no_grad()
def _build_dense_transfer_matrix(
    upper_b: torch.Tensor,
    upper_a: torch.Tensor,
    double_layer_a: torch.Tensor,
    double_layer_b: torch.Tensor,
    lower_a: torch.Tensor,
    lower_b: torch.Tensor,
    *,
    max_intermediate_bytes: int,
) -> tuple[torch.Tensor, int, int]:
    """Build ``V[(Y,v),(y,V)]`` in blocks without a full-size permutation."""

    chi = int(upper_b.shape[0])
    upper_half, lower_half, b_chunk = _build_boundary_halves(
        upper_b,
        upper_a,
        double_layer_a,
        double_layer_b,
        lower_a,
        lower_b,
        max_intermediate_bytes=max_intermediate_bytes,
    )

    dense_4d = torch.empty(
        chi,
        chi,
        chi,
        chi,
        dtype=upper_b.dtype,
        device=upper_b.device,
    )

    # One Y slice contains chi**3 values.  Reserve a factor of two for the
    # contraction result and its assignment/permutation workspace.
    bytes_per_Y = 2 * chi**3 * upper_b.element_size()
    Y_chunk = max(
        1,
        min(chi, int(max_intermediate_bytes // bytes_per_Y)),
    )

    for start in range(0, chi, Y_chunk):
        stop = min(start + Y_chunk, chi)
        # upper_half[b,V,Y] lower_half[b,y,v] -> work[V,Y,y,v]
        work = torch.tensordot(
            upper_half[:, :, start:stop],
            lower_half,
            dims=([0], [0]),
        )
        # Store directly in the required V[Y,v,y,V] layout.
        dense_4d[start:stop] = work.permute(1, 3, 2, 0)
        del work

    del upper_half, lower_half
    return dense_4d.reshape(chi**2, chi**2), b_chunk, Y_chunk


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
    matrix_free: bool = False,
    materialize: bool = False,
    max_dense_bytes: int = 16 * 1024**3,
) -> RowToRowTransferOperator | torch.Tensor:
    """Build the corrected env2 side-by-side row-transfer operator.

    The historical argument names are retained for API compatibility.  They
    denote env2 edges ``upper_b=T1D``, ``upper_a=T2C``,
    ``lower_a=T1D``, and ``lower_b=T2C``.  The actual row order is therefore
    top ``(T2C,T1D)``, bottom ``(T1D,T2C)``.  From the retained A/B orbit
    representatives, the canonical horizontal D--C central pair is
    ``(B.permute(2,1,0), A)``.
    """

    side_by_side = SideBySideRowToRowTransferOperator(
        upper_a,
        upper_b,
        double_layer_b.permute(2, 1, 0).contiguous(),
        double_layer_a,
        lower_a,
        lower_b,
        max_intermediate_bytes=max_intermediate_bytes,
        progress_every=progress_every,
    )
    if matrix_free:
        operator: RowToRowTransferOperator = side_by_side
    else:
        matrix = side_by_side.to_dense(
            max_dense_bytes=max_dense_bytes,
        )
        operator = DenseRowToRowTransferOperator(
            matrix,
            chi=int(upper_b.shape[0]),
            progress_every=progress_every,
        )
        del matrix, side_by_side
    if materialize:
        return operator.to_dense(max_dense_bytes=max_dense_bytes)
    return operator


def _relative_eigen_residual(
    operator: RowToRowTransferOperator,
    eigenvalue: complex,
    eigenvector: np.ndarray,
) -> tuple[complex, float]:
    """Refine one Ritz value and evaluate its scale-independent residual."""

    generalized_refiner = getattr(
        operator, "generalized_refine_numpy", None
    )
    if callable(generalized_refiner):
        return generalized_refiner(eigenvalue, eigenvector)

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
    seed: int | None,
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
    seed: int | None = None,
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
    ctm_conv_mode: str = DEFAULT_CTM_CONV_MODE,
    ctm_e_conv_threshold: float = DEFAULT_CTM_E_CONV_THRESHOLD,
    identity_init: bool = DEFAULT_IDENTITY_INIT,
    rsvd_mode: str = DEFAULT_RSVD_MODE,
    rsvd_neumann_terms: int = DEFAULT_RSVD_NEUMANN_TERMS,
    rsvd_power_iters: int | None = DEFAULT_RSVD_POWER_ITERS,
    force_full_svd: bool = False,
    j1: float = DEFAULT_J1,
    j2: float = DEFAULT_J2,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
    max_intermediate_bytes: int = int(
        DEFAULT_MAX_INTERMEDIATE_MIB * _MIB
    ),
    eig_tol: float = 0.0,
    arpack_ncv: int = DEFAULT_ARPACK_NCV,
    arpack_maxiter: int = DEFAULT_ARPACK_MAXITER,
    dense_dimension_threshold: int = 256,
    seed: int | None = None,
    progress_every: int = 0,
    matrix_free: bool = False,
    corner_relative_cutoff: float = DEFAULT_CORNER_RELATIVE_CUTOFF,
    return_result: bool = False,
) -> float | CorrelationLengthResult:
    """Run both CTMRG boundaries and compute one correlation length."""

    D_bond = int(a.shape[0])
    if D_bond in DISABLED_CORRELATION_LENGTH_DS:
        raise ValueError(
            "Correlation-length calculations are disabled for D=2 because "
            "the CTM empty-row metric is numerically ill-conditioned."
        )
    effective_rsvd_mode = (
        "full_svd"
        if D_bond in FULL_SVD_CORRELATION_LENGTH_DS
        else rsvd_mode
    )
    effective_force_full_svd = (
        force_full_svd or D_bond in FULL_SVD_CORRELATION_LENGTH_DS
    )

    resolved_seed = (
        secrets.randbits(32) if seed is None else int(seed)
    )
    if not 0 <= resolved_seed < 2**32:
        raise ValueError("seed must be in the interval [0, 2**32).")
    np.random.seed(resolved_seed)
    torch.manual_seed(resolved_seed)
    target_device = torch.device(device) if device is not None else a.device
    if target_device.type == "cuda":
        torch.cuda.manual_seed_all(resolved_seed)

    components = obtain_transfer_components(
        a,
        b,
        chi,
        ctm_max_steps=ctm_max_steps,
        ctm_conv_tol=ctm_conv_tol,
        ctm_conv_mode=ctm_conv_mode,
        ctm_e_conv_threshold=ctm_e_conv_threshold,
        identity_init=identity_init,
        rsvd_mode=effective_rsvd_mode,
        rsvd_neumann_terms=rsvd_neumann_terms,
        rsvd_power_iters=rsvd_power_iters,
        force_full_svd=effective_force_full_svd,
        j1=j1,
        j2=j2,
        dtype=dtype,
        device=device,
    )
    # env2 has top edges (T2C,T1D), bottom edges (T1D,T2C), and the
    # horizontal D--C central bond.  In canonical [top,bond,bottom]
    # coordinates D is B.permute(2,1,0), while C is exactly A.  The old
    # production network instead stacked A and B vertically and contracted
    # their middle legs; that bow-tie is not this row transfer.
    side_by_side = SideBySideRowToRowTransferOperator(
        components.upper_a,
        components.upper_b,
        components.double_layer_b.permute(2, 1, 0).contiguous(),
        components.double_layer_a,
        components.lower_a,
        components.lower_b,
        max_intermediate_bytes=max_intermediate_bytes,
        progress_every=progress_every,
    )
    if matrix_free:
        raw_transfer: RowToRowTransferOperator = side_by_side
    else:
        dense_transfer = side_by_side.to_dense(
            max_dense_bytes=max(
                512 * _MIB,
                side_by_side.shape[0]
                * side_by_side.shape[1]
                * torch.empty(
                    (), dtype=side_by_side.dtype
                ).element_size(),
            )
        )
        raw_transfer = DenseRowToRowTransferOperator(
            dense_transfer,
            chi=side_by_side.chi,
            progress_every=progress_every,
        )
        del dense_transfer, side_by_side
    if not isinstance(raw_transfer, RowToRowTransferOperator):
        raise RuntimeError("Expected a transfer operator.")
    transfer = KroneckerCornerWhitenedTransferOperator(
        raw_transfer,
        components.ctm_corner_ab.T.contiguous(),
        components.ctm_corner_ba,
        relative_cutoff=corner_relative_cutoff,
    )
    del raw_transfer

    steps_ab = components.ctm_steps_ab
    steps_ba = components.ctm_steps_ba
    energy_proxy_ab = components.ctm_energy_proxy_ab
    energy_proxy_ba = components.ctm_energy_proxy_ba
    corner_spectra_ab = components.ctm_corner_spectra_ab
    corner_spectra_ba = components.ctm_corner_spectra_ba
    corner_effective_ranks = transfer.corner_effective_ranks
    overlap_condition_number = transfer.overlap_condition_number
    del components
    _release_unused_memory(transfer.device)

    eigensolver = diagonalize_first_two_largest_eigval(
        transfer,
        tol=eig_tol,
        ncv=arpack_ncv,
        maxiter=arpack_maxiter,
        dense_dimension_threshold=dense_dimension_threshold,
        seed=resolved_seed,
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
        ctm_steps_ab=steps_ab,
        ctm_steps_ba=steps_ba,
        ctm_energy_proxy_ab=energy_proxy_ab,
        ctm_energy_proxy_ba=energy_proxy_ba,
        ctm_corner_spectra_ab=corner_spectra_ab,
        ctm_corner_spectra_ba=corner_spectra_ba,
        D_bond=int(a.shape[0]),
        chi=int(chi),
        transfer_mode=(
            "matrix_free_straight_row_corner_whitened_v3"
            if matrix_free
            else (
                f"dense_{transfer.device.type}_"
                "straight_row_corner_whitened_v3"
            )
        ),
        transfer_matrix_gib=(
            0.0
            if matrix_free
            else transfer.shape[0]
            * transfer.shape[1]
            * torch.empty((), dtype=transfer.dtype).element_size()
            / 1024**3
        ),
        corner_effective_ranks=corner_effective_ranks,
        overlap_condition_number=overlap_condition_number,
    )
    del transfer
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
    checkpoint_metadata: dict[str, Any],
    calculation_hyperparameters: dict[str, Any],
    tensor_metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "transfer_network_schema": "straight_row_env2_v3",
        "checkpoint": os.path.abspath(checkpoint),
        "D_bond": result.D_bond,
        "chi": result.chi,
        "device": str(device),
        "dtype": str(dtype),
        "transfer_mode": result.transfer_mode,
        "transfer_matrix_gib": result.transfer_matrix_gib,
        "corner_effective_ranks": list(result.corner_effective_ranks),
        "overlap_condition_number": result.overlap_condition_number,
        "ctm_steps_ab": result.ctm_steps_ab,
        "ctm_steps_ba": result.ctm_steps_ba,
        "ctm_energy_proxy_ab": result.ctm_energy_proxy_ab,
        "ctm_energy_proxy_ba": result.ctm_energy_proxy_ba,
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
        "eigensolver_used_full_dense_diagonalization": (
            result.eigensolver.used_dense_solver
        ),
        "eigensolver_seconds": result.eigensolver.elapsed_seconds,
        "checkpoint_metadata": checkpoint_metadata,
        "optimized_ipeps_tensors": tensor_metadata,
        "calculation_hyperparameters": calculation_hyperparameters,
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
        "--ctm-conv-mode",
        choices=("SVdifference", "Edifference", "both"),
        default=DEFAULT_CTM_CONV_MODE,
    )
    parser.add_argument(
        "--ctm-e-conv-threshold",
        type=float,
        default=DEFAULT_CTM_E_CONV_THRESHOLD,
    )
    parser.add_argument(
        "--random-init",
        action="store_true",
        help="Use the non-identity CTMRG initialization.",
    )
    parser.add_argument(
        "--rsvd-mode",
        choices=("full_svd", "neumann", "augmented", "none"),
        default=DEFAULT_RSVD_MODE,
    )
    parser.add_argument(
        "--rsvd-neumann-terms",
        type=int,
        default=DEFAULT_RSVD_NEUMANN_TERMS,
    )
    parser.add_argument(
        "--rsvd-power-iters",
        type=int,
        default=DEFAULT_RSVD_POWER_ITERS,
    )
    parser.add_argument("--J1", type=float, default=DEFAULT_J1)
    parser.add_argument("--J2", type=float, default=DEFAULT_J2)
    parser.add_argument(
        "--max-intermediate-mib",
        type=float,
        default=DEFAULT_MAX_INTERMEDIATE_MIB,
        help="Target cap for the largest transfer contraction intermediate.",
    )
    parser.add_argument(
        "--matrix-free",
        action="store_true",
        help=(
            "Do not store the dense transfer matrix. This saves VRAM but is "
            "usually slower because every ARPACK matvec applies the two "
            "factorized boundary halves."
        ),
    )
    parser.add_argument(
        "--eig-tol",
        type=float,
        default=0.0,
        help="ARPACK tolerance. Zero requests machine precision.",
    )
    parser.add_argument(
        "--corner-relative-cutoff",
        type=float,
        default=DEFAULT_CORNER_RELATIVE_CUTOFF,
        help="Relative SVD cutoff for numerically null empty-row corner modes.",
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
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Optional CTMRG/eigensolver seed. When omitted, draw a fresh "
            "random 32-bit seed and record it in the output."
        ),
    )
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
    seed_was_user_specified = args.seed is not None
    resolved_seed = secrets.randbits(32) if args.seed is None else int(args.seed)
    if not 0 <= resolved_seed < 2**32:
        raise ValueError("--seed must be in the interval [0, 2**32).")
    if device.type == "cuda":
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
    if D_bond in DISABLED_CORRELATION_LENGTH_DS:
        raise ValueError(
            "Correlation-length calculations are disabled for D=2 because "
            "the CTM empty-row metric is numerically ill-conditioned."
        )
    effective_rsvd_mode = (
        "full_svd"
        if D_bond in FULL_SVD_CORRELATION_LENGTH_DS
        else args.rsvd_mode
    )
    checkpoint_D = metadata.get("D_bond")
    if checkpoint_D is not None and int(checkpoint_D) != D_bond:
        raise ValueError(
            f"Checkpoint D_bond={checkpoint_D} disagrees with tensor D={D_bond}."
        )
    if args.max_intermediate_mib <= 0.0:
        raise ValueError("--max-intermediate-mib must be positive.")
    if args.corner_relative_cutoff < 0.0:
        raise ValueError("--corner-relative-cutoff cannot be negative.")

    tensor_metadata = {
        "a_shape": list(a.shape),
        "b_shape": list(b.shape),
        "a_raw_frobenius_norm": float(torch.linalg.norm(a).item()),
        "b_raw_frobenius_norm": float(torch.linalg.norm(b).item()),
        "d_phys": int(a.shape[-1]),
    }
    expected_transfer_gib = (
        chi**4 * torch.empty((), dtype=dtype).element_size() / 1024**3
    )
    calculation_hyperparameters = {
        "ctm_max_steps": args.ctm_max_steps,
        "ctm_conv_tol_requested": args.ctm_conv_tol,
        "ctm_conv_tol_effective": (
            max(args.ctm_conv_tol, 1.0e-5)
            if dtype == torch.float32
            else args.ctm_conv_tol
        ),
        "ctm_conv_mode": args.ctm_conv_mode,
        "ctm_e_conv_threshold": args.ctm_e_conv_threshold,
        "ctm_identity_init": not args.random_init,
        "energy_proxy": "main_C3_LBFGS_three_environment_energy",
        "J1": args.J1,
        "J2": args.J2,
        "rsvd_mode_requested": args.rsvd_mode,
        "rsvd_mode": effective_rsvd_mode,
        "rsvd_neumann_terms": args.rsvd_neumann_terms,
        "rsvd_power_iters": args.rsvd_power_iters,
        "force_full_svd_forward": effective_rsvd_mode == "full_svd",
        "max_intermediate_mib": args.max_intermediate_mib,
        "transfer_mode": (
            "matrix_free_straight_row_corner_whitened_v3"
            if args.matrix_free
            else f"dense_{device.type}_straight_row_corner_whitened_v3"
        ),
        "corner_relative_cutoff": args.corner_relative_cutoff,
        "expected_dense_transfer_gib": expected_transfer_gib,
        "eig_tol": args.eig_tol,
        "arpack_ncv": args.arpack_ncv,
        "arpack_maxiter": args.arpack_maxiter,
        "dense_diagonalization_dimension_threshold": args.dense_threshold,
        "ctm_random_seed": resolved_seed,
        "eigensolver_seed": resolved_seed,
        "seed_was_user_specified": seed_was_user_specified,
    }

    print(
        f"Computing correlation length for D={D_bond}, chi={chi}, "
        f"dtype={dtype}, device={device}.",
        flush=True,
    )
    if D_bond in FULL_SVD_CORRELATION_LENGTH_DS:
        print(
            f"D={D_bond} policy: forcing deterministic full-SVD CTMRG "
            "truncations.",
            flush=True,
        )
    print(
        "Running CTMRG for the (a, b) and (b, a) boundaries, then building "
        f"the {calculation_hyperparameters['transfer_mode']} transfer operator.",
        flush=True,
    )
    if not args.matrix_free:
        print(
            f"The resident real dense transfer matrix will require "
            f"{expected_transfer_gib:.3f} GiB.",
            flush=True,
        )
    sys.stdout.flush()

    result = obtain_per_D_correlation_length(
        a,
        b,
        chi,
        ctm_max_steps=args.ctm_max_steps,
        ctm_conv_tol=args.ctm_conv_tol,
        ctm_conv_mode=args.ctm_conv_mode,
        ctm_e_conv_threshold=args.ctm_e_conv_threshold,
        identity_init=not args.random_init,
        rsvd_mode=args.rsvd_mode,
        rsvd_neumann_terms=args.rsvd_neumann_terms,
        rsvd_power_iters=args.rsvd_power_iters,
        j1=args.J1,
        j2=args.J2,
        dtype=dtype,
        device=device,
        max_intermediate_bytes=int(args.max_intermediate_mib * _MIB),
        eig_tol=args.eig_tol,
        arpack_ncv=args.arpack_ncv,
        arpack_maxiter=args.arpack_maxiter,
        dense_dimension_threshold=args.dense_threshold,
        seed=resolved_seed,
        progress_every=args.progress_every,
        matrix_free=args.matrix_free,
        corner_relative_cutoff=args.corner_relative_cutoff,
        return_result=True,
    )
    if not isinstance(result, CorrelationLengthResult):
        raise RuntimeError("Expected a detailed correlation-length result.")

    payload = _result_to_dict(
        result,
        checkpoint=args.checkpoint,
        device=device,
        dtype=dtype,
        checkpoint_metadata=metadata,
        calculation_hyperparameters=calculation_hyperparameters,
        tensor_metadata=tensor_metadata,
    )
    print(json.dumps(payload, indent=2, allow_nan=True), flush=True)
    sys.stdout.flush()

    if args.output is not None:
        output_path = os.path.abspath(args.output)
        output_directory = os.path.dirname(output_path)
        if output_directory:
            os.makedirs(output_directory, exist_ok=True)
        temporary_output_path = output_path + ".tmp"
        with open(temporary_output_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, allow_nan=True)
            handle.write("\n")
        os.replace(temporary_output_path, output_path)
        print(f"Saved {output_path}", flush=True)
        sys.stdout.flush()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
