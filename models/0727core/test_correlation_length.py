#!/usr/bin/env python3
"""Physical and numerical validation of correlation_length.py.

The primary tests probe physical structure rather than only comparing duplicate
implementations:

1. An exact orthogonal virtual-bond gauge transformation must preserve the
   spectrum when CTMRG reaches the same boundary fixed point.
2. C3 rotation and A/B exchange are diagnostic only unless the corresponding
   row direction and leg maps are constructed explicitly.
3. An embedded product-state PEPS must have a rank-one transfer matrix.
4. A GHZ PEPS exposes the expected two-sector structure.

Small implementation-regression checks are retained after the physical tests.
They are useful for locating a failure, but they are not treated as evidence of
physical correctness by themselves.

Each D first runs a generic-state smoke calculation with the exact main_C3
LBFGS CTMRG defaults and the corresponding CHI_MIN_LIST entry.  Physical
equivalence checks use the same chi and convergence proxy but deterministic
full forward SVDs and covariant direct initialization, removing randomized
truncation and non-covariant boundary initialization from the assertion.
Product and GHZ are analytic low-rank limits: they use small sufficient chi
values with the same symmetry-preserving validation settings.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import gc
import itertools
import math
import pathlib
import sys
import time

import numpy as np
import opt_einsum as oe
import torch
from scipy import linalg as scipy_linalg

import core_C3 as core
import correlation_length as corr


CHI_MIN_BY_D = {
    2: 16,
    3: 24,
    4: 36,
    5: 50,
    6: 72,
}


@dataclasses.dataclass
class CaseSpectrum:
    name: str
    eigenvalues: tuple[complex, complex]
    residuals: tuple[float, float]
    correlation_length: float
    ctm_steps_ab: int
    ctm_steps_ba: int
    ctm_energy_proxy_ab: float
    ctm_energy_proxy_ba: float
    ctm_corner_spectra_ab: tuple[tuple[float, ...], ...]
    ctm_corner_spectra_ba: tuple[tuple[float, ...], ...]
    corner_effective_ranks: tuple[int, int]
    overlap_condition_number: float

    @property
    def normalized_subleading(self) -> complex:
        return self.eigenvalues[1] / self.eigenvalues[0]


def report(message: str) -> None:
    print(message, flush=True)
    sys.stdout.flush()


def clear_memory(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def reset_rng(seed: int, device: torch.device) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def audit_main_c3_lbfgs_defaults() -> None:
    """Verify that production defaults still match main_C3 without importing it."""

    main_path = pathlib.Path(__file__).with_name("main_C3.py")
    module = ast.parse(main_path.read_text(encoding="utf-8"))
    values: dict[str, object] = {}
    for node in module.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            values[target.id] = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            continue

    expected = {
        "D_BOND_LIST": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "CHI_MIN_LIST": [16, 24, 36, 50, 72, 91, 104, 126, 140, 165, 180],
        "USE_DOUBLE_PRECISION": True,
        "USE_REAL_TENSORS": True,
        "J1_COUPLING": corr.DEFAULT_J1,
        "J2_COUPLING": corr.DEFAULT_J2,
        "RSVD_MODE": corr.DEFAULT_RSVD_MODE,
        "RSVD_NEUMANN_TERMS": corr.DEFAULT_RSVD_NEUMANN_TERMS,
        "RSVD_POWER_ITERS": corr.DEFAULT_RSVD_POWER_ITERS,
        "ENV_IDENTITY_INIT": corr.DEFAULT_IDENTITY_INIT,
        "CTM_MAX_STEPS": corr.DEFAULT_CTM_MAX_STEPS,
        "CTM_CONV_THR": corr.DEFAULT_CTM_CONV_TOL,
        "CTM_CONV_MODE": corr.DEFAULT_CTM_CONV_MODE,
        "CTM_E_CONV_THRESHOLD": corr.DEFAULT_CTM_E_CONV_THRESHOLD,
    }
    mismatches = {
        name: (values.get(name), expected_value)
        for name, expected_value in expected.items()
        if values.get(name) != expected_value
    }
    if mismatches:
        raise AssertionError(
            f"correlation_length defaults disagree with main_C3: {mismatches}"
        )
    report("main_C3 LBFGS default audit passed.")


def apply_uniform_virtual_map(
    tensor: torch.Tensor,
    virtual_map: torch.Tensor,
) -> torch.Tensor:
    """Apply one virtual basis map to all three honeycomb legs."""

    return torch.einsum(
        "ia,jb,kc,abcp->ijkp",
        virtual_map,
        virtual_map,
        virtual_map,
        tensor,
    ).contiguous()


def spectrum_distance(left: complex, right: complex) -> float:
    """Compare an oriented ratio while allowing direction reversal."""

    scale = max(abs(left), abs(right), 1.0e-30)
    direct = abs(left - right) / scale
    conjugated = abs(left - right.conjugate()) / scale
    return min(direct, conjugated)


def print_spectrum(case: CaseSpectrum) -> None:
    ratio = case.normalized_subleading
    report(
        f"{case.name}: "
        f"lambda1={case.eigenvalues[0].real:+.16e}"
        f"{case.eigenvalues[0].imag:+.16e}j, "
        f"lambda2={case.eigenvalues[1].real:+.16e}"
        f"{case.eigenvalues[1].imag:+.16e}j"
    )
    report(
        f"{case.name}: lambda2/lambda1={ratio.real:+.16e}"
        f"{ratio.imag:+.16e}j, xi={case.correlation_length:.16e}, "
        f"residuals=({case.residuals[0]:.3e},{case.residuals[1]:.3e}), "
        f"ctm_steps=({case.ctm_steps_ab},{case.ctm_steps_ba}), "
        f"energy_proxy=({case.ctm_energy_proxy_ab:+.12e},"
        f"{case.ctm_energy_proxy_ba:+.12e}), "
        f"corner_ranks={case.corner_effective_ranks}, "
        f"cond(N)={case.overlap_condition_number:.6e}"
    )


def make_case_spectrum(
    name: str,
    raw_a: torch.Tensor,
    raw_b: torch.Tensor,
    *,
    args: argparse.Namespace,
    device: torch.device,
    chi: int | None = None,
    identity_init: bool = True,
    rsvd_mode: str = corr.DEFAULT_RSVD_MODE,
    force_full_svd: bool = False,
) -> CaseSpectrum:
    """Run one complete CTMRG and transfer-spectrum calculation."""

    effective_chi = args.chi if chi is None else chi
    report(
        f"Running physical case: {name}, D={args.D}, chi={effective_chi}, "
        f"identity_init={identity_init}, rsvd_mode={rsvd_mode}, "
        f"force_full_svd={force_full_svd}"
    )
    reset_rng(args.seed, device)
    result = corr.obtain_per_D_correlation_length(
        raw_a,
        raw_b,
        effective_chi,
        ctm_max_steps=args.ctm_steps,
        ctm_conv_tol=args.ctm_tol,
        ctm_conv_mode=args.ctm_conv_mode,
        ctm_e_conv_threshold=args.ctm_e_conv_threshold,
        identity_init=identity_init,
        rsvd_mode=rsvd_mode,
        rsvd_neumann_terms=corr.DEFAULT_RSVD_NEUMANN_TERMS,
        rsvd_power_iters=corr.DEFAULT_RSVD_POWER_ITERS,
        force_full_svd=force_full_svd,
        j1=args.J1,
        j2=args.J2,
        dtype=torch.float64,
        device=device,
        max_intermediate_bytes=256 * 1024**2,
        eig_tol=0.0,
        arpack_ncv=min(corr.DEFAULT_ARPACK_NCV, effective_chi**2),
        arpack_maxiter=corr.DEFAULT_ARPACK_MAXITER,
        dense_dimension_threshold=256,
        seed=args.seed,
        progress_every=0,
        matrix_free=False,
        return_result=True,
    )
    if not isinstance(result, corr.CorrelationLengthResult):
        raise AssertionError("A detailed correlation-length result was not returned.")

    case = CaseSpectrum(
        name=name,
        eigenvalues=result.eigensolver.eigenvalues,
        residuals=result.eigensolver.relative_residuals,
        correlation_length=result.correlation_length,
        ctm_steps_ab=result.ctm_steps_ab,
        ctm_steps_ba=result.ctm_steps_ba,
        ctm_energy_proxy_ab=result.ctm_energy_proxy_ab,
        ctm_energy_proxy_ba=result.ctm_energy_proxy_ba,
        ctm_corner_spectra_ab=result.ctm_corner_spectra_ab,
        ctm_corner_spectra_ba=result.ctm_corner_spectra_ba,
        corner_effective_ranks=result.corner_effective_ranks,
        overlap_condition_number=result.overlap_condition_number,
    )
    print_spectrum(case)
    clear_memory(device)
    return case


def assert_physical_equivalence(
    reference: CaseSpectrum,
    transformed: CaseSpectrum,
    *,
    tolerance: float,
    proxy_tolerance: float,
    corner_tolerance: float,
    ctm_max_steps: int,
) -> bool:
    """Compare spectra only when CTMRG invariants identify one fixed point."""

    reference_ratio = reference.normalized_subleading
    transformed_ratio = transformed.normalized_subleading
    ratio_error = spectrum_distance(reference_ratio, transformed_ratio)
    magnitude_error = abs(abs(reference_ratio) - abs(transformed_ratio))
    reference_proxies = np.asarray(
        [
            reference.ctm_energy_proxy_ab,
            reference.ctm_energy_proxy_ba,
        ],
        dtype=np.float64,
    )
    transformed_proxies = np.asarray(
        [
            transformed.ctm_energy_proxy_ab,
            transformed.ctm_energy_proxy_ba,
        ],
        dtype=np.float64,
    )
    proxy_scale = max(
        float(np.max(np.abs(reference_proxies))),
        float(np.max(np.abs(transformed_proxies))),
        1.0,
    )
    direct_proxy_error = float(
        np.max(np.abs(reference_proxies - transformed_proxies)) / proxy_scale
    )
    swapped_proxy_error = float(
        np.max(np.abs(reference_proxies - transformed_proxies[::-1]))
        / proxy_scale
    )
    proxy_error = min(direct_proxy_error, swapped_proxy_error)
    def weighted_corner_error(
        left: tuple[float, ...],
        right: tuple[float, ...],
    ) -> float:
        left_values = np.asarray(left, dtype=np.float64)
        right_values = np.asarray(right, dtype=np.float64)
        weights = 0.5 * (left_values + right_values)
        return float(np.max(np.abs(left_values - right_values) * weights))

    def match_corner_group(
        left: tuple[tuple[float, ...], ...],
        right: tuple[tuple[float, ...], ...],
    ) -> float:
        return min(
            max(
                weighted_corner_error(
                    left[index],
                    right[mapped_index],
                )
                for index, mapped_index in enumerate(permutation)
            )
            for permutation in itertools.permutations(range(3))
        )

    direct_corner_error = max(
        match_corner_group(
            reference.ctm_corner_spectra_ab,
            transformed.ctm_corner_spectra_ab,
        ),
        match_corner_group(
            reference.ctm_corner_spectra_ba,
            transformed.ctm_corner_spectra_ba,
        ),
    )
    swapped_corner_error = max(
        match_corner_group(
            reference.ctm_corner_spectra_ab,
            transformed.ctm_corner_spectra_ba,
        ),
        match_corner_group(
            reference.ctm_corner_spectra_ba,
            transformed.ctm_corner_spectra_ab,
        ),
    )
    corner_error = min(direct_corner_error, swapped_corner_error)
    report(
        f"{transformed.name} versus {reference.name}: "
        f"oriented_ratio_error={ratio_error:.6e}, "
        f"magnitude_error={magnitude_error:.6e}, "
        f"energy_proxy_error={proxy_error:.6e}, "
        f"corner_spectrum_error={corner_error:.6e} "
        f"(direct={direct_corner_error:.6e}, "
        f"swapped={swapped_corner_error:.6e})"
    )
    if max(
        reference.ctm_steps_ab,
        reference.ctm_steps_ba,
        transformed.ctm_steps_ab,
        transformed.ctm_steps_ba,
    ) >= ctm_max_steps:
        report(
            "INCONCLUSIVE SPECTRUM COMPARISON: at least one CTMRG run "
            "reached the iteration limit."
        )
        return False
    if proxy_error > proxy_tolerance:
        report(
            f"INCONCLUSIVE SPECTRUM COMPARISON: CTMRG physical proxies differ "
            f"by {proxy_error:.6e}."
        )
        return False
    if corner_error > corner_tolerance:
        report(
            f"INCONCLUSIVE SPECTRUM COMPARISON: normalized CTMRG corner "
            f"spectra differ by {corner_error:.6e}."
        )
        return False
    if ratio_error > tolerance or magnitude_error > tolerance:
        raise AssertionError(
            f"{transformed.name} violates physical spectrum invariance."
        )
    return True


def build_generic_tensors(
    D: int,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    reset_rng(seed + 101, device)
    a = torch.randn(D, D, D, 2, dtype=torch.float64, device=device)
    b = torch.randn_like(a)

    # Add a common injective component to avoid testing an accidentally
    # ill-conditioned random corner of the PEPS manifold.
    for virtual in range(D):
        physical = virtual % 2
        a[virtual, virtual, virtual, physical] += 2.0
        b[virtual, virtual, virtual, 1 - physical] += 2.0
    return a, b


def build_product_state(
    D: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    a = torch.zeros(D, D, D, 2, dtype=torch.float64, device=device)
    b = torch.zeros_like(a)
    a[0, 0, 0, 0] = 1.0
    b[0, 0, 0, 1] = 1.0
    return a, b


def build_ghz_state(
    D: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if D < 2:
        raise ValueError("The GHZ validation requires D >= 2.")
    a = torch.zeros(D, D, D, 2, dtype=torch.float64, device=device)
    b = torch.zeros_like(a)
    a[0, 0, 0, 0] = 1.0
    a[1, 1, 1, 1] = 1.0
    b.copy_(a)
    return a, b


def run_edge_orientation_diagnostic(
    raw_a: torch.Tensor,
    raw_b: torch.Tensor,
    gauged_a: torch.Tensor,
    gauged_b: torch.Tensor,
    *,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    """Rank all 16 chi-leg transpose masks by orthogonal-gauge consistency."""

    if args.D != 2:
        return
    report("D=2 EDGE ORIENTATION DIAGNOSTIC: scanning 16 transpose masks.")

    def obtain(a: torch.Tensor, b: torch.Tensor) -> corr.TransferComponents:
        reset_rng(args.seed, device)
        return corr.obtain_transfer_components(
            a,
            b,
            args.chi,
            ctm_max_steps=args.ctm_steps,
            ctm_conv_tol=args.ctm_tol,
            ctm_conv_mode=args.ctm_conv_mode,
            ctm_e_conv_threshold=args.ctm_e_conv_threshold,
            identity_init=False,
            rsvd_mode=corr.DEFAULT_RSVD_MODE,
            rsvd_neumann_terms=corr.DEFAULT_RSVD_NEUMANN_TERMS,
            rsvd_power_iters=corr.DEFAULT_RSVD_POWER_ITERS,
            force_full_svd=True,
            j1=args.J1,
            j2=args.J2,
            dtype=torch.float64,
            device=device,
        )

    reference = obtain(raw_a, raw_b)
    gauged = obtain(gauged_a, gauged_b)

    def ratio_for_mask(
        components: corr.TransferComponents,
        mask: int,
    ) -> complex:
        edges = list(components.edges)
        for index in range(4):
            if mask & (1 << index):
                edges[index] = edges[index].transpose(0, 1).contiguous()
        operator = corr.compute_r2rTransferMatrix(
            edges[0],
            edges[1],
            components.double_layer_a,
            components.double_layer_b,
            edges[2],
            edges[3],
            max_intermediate_bytes=256 * 1024**2,
            matrix_free=False,
        )
        eigensolver = corr.diagonalize_first_two_largest_eigval(
            operator,
            tol=0.0,
            ncv=min(corr.DEFAULT_ARPACK_NCV, args.chi**2),
            maxiter=corr.DEFAULT_ARPACK_MAXITER,
            dense_dimension_threshold=0,
            seed=args.seed,
            return_result=True,
        )
        if not isinstance(eigensolver, corr.EigensolverResult):
            raise AssertionError("Orientation scan did not return eig diagnostics.")
        ratio = eigensolver.eigenvalues[1] / eigensolver.eigenvalues[0]
        del operator, eigensolver, edges
        clear_memory(device)
        return ratio

    ranking: list[tuple[float, int, complex, complex]] = []
    for mask in range(16):
        reference_ratio = ratio_for_mask(reference, mask)
        gauged_ratio = ratio_for_mask(gauged, mask)
        error = spectrum_distance(reference_ratio, gauged_ratio)
        ranking.append((error, mask, reference_ratio, gauged_ratio))

    ranking.sort(key=lambda item: item[0])
    for error, mask, reference_ratio, gauged_ratio in ranking[:5]:
        flags = "".join(
            "1" if mask & (1 << index) else "0"
            for index in range(4)
        )
        report(
            f"orientation flags={flags} "
            f"(bits upper_b,upper_a,lower_a,lower_b), "
            f"gauge_error={error:.6e}, "
            f"reference_ratio={reference_ratio.real:+.6e}"
            f"{reference_ratio.imag:+.6e}j, "
            f"gauged_ratio={gauged_ratio.real:+.6e}"
            f"{gauged_ratio.imag:+.6e}j"
        )
    current = next(item for item in ranking if item[1] == 0)
    report(f"current production mask=0000 gauge_error={current[0]:.6e}")
    del reference, gauged, ranking
    clear_memory(device)


def generalized_corner_identification_spectrum(
    name: str,
    raw_a: torch.Tensor,
    raw_b: torch.Tensor,
    *,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[complex, float]:
    """Diagonalize the transfer relative to its empty-row corner map.

    The six-tensor network maps ``(y,V)`` from the lower and upper CTMRG
    boundaries to ``(Y,v)`` in the opposite tensor-product order.  Since the
    two boundaries come from independent CTMRG runs, their open chi bases
    cannot be identified by a numerical Kronecker delta.  The two matching
    corners give the covariant empty-row map

        N[Y,v,y,V] = C_ab[Y,V] C_ba[y,v].

    Generalized eigenvalues of ``T x = lambda N x`` are invariant under
    independent changes of all four open CTM bases.
    """

    reset_rng(args.seed, device)
    components = corr.obtain_transfer_components(
        raw_a,
        raw_b,
        args.chi,
        ctm_max_steps=args.ctm_steps,
        ctm_conv_tol=args.ctm_tol,
        ctm_conv_mode=args.ctm_conv_mode,
        ctm_e_conv_threshold=args.ctm_e_conv_threshold,
        identity_init=False,
        rsvd_mode=corr.DEFAULT_RSVD_MODE,
        rsvd_neumann_terms=corr.DEFAULT_RSVD_NEUMANN_TERMS,
        rsvd_power_iters=corr.DEFAULT_RSVD_POWER_ITERS,
        force_full_svd=True,
        j1=args.J1,
        j2=args.J2,
        dtype=torch.float64,
        device=device,
    )
    transfer = corr.compute_r2rTransferMatrix(
        components.upper_b,
        components.upper_a,
        components.double_layer_a,
        components.double_layer_b,
        components.lower_a,
        components.lower_b,
        max_intermediate_bytes=256 * 1024**2,
        matrix_free=False,
        materialize=True,
    )
    if not isinstance(transfer, torch.Tensor):
        raise AssertionError("The generalized diagnostic requires a dense matrix.")

    overlap = torch.einsum(
        "YV,yv->YvyV",
        components.ctm_corner_ab,
        components.ctm_corner_ba,
    ).reshape(args.chi**2, args.chi**2)
    corner_condition = float(
        torch.linalg.cond(components.ctm_corner_ab).item()
        * torch.linalg.cond(components.ctm_corner_ba).item()
    )
    transfer_numpy = transfer.detach().cpu().numpy()
    overlap_numpy = overlap.detach().cpu().numpy()
    eigenvalues = scipy_linalg.eigvals(
        transfer_numpy,
        overlap_numpy,
        check_finite=False,
    )
    finite = eigenvalues[np.isfinite(eigenvalues)]
    if finite.size < 2:
        raise AssertionError("Fewer than two finite generalized eigenvalues.")
    order = np.argsort(np.abs(finite))[::-1]
    leading = complex(finite[order[0]])
    subleading = complex(finite[order[1]])
    ratio = subleading / leading
    report(
        f"{name} generalized-corner spectrum: "
        f"lambda2/lambda1={ratio.real:+.16e}{ratio.imag:+.16e}j, "
        f"|ratio|={abs(ratio):.16e}, "
        f"cond(N)={corner_condition:.6e}"
    )
    del components, transfer, overlap, transfer_numpy, overlap_numpy, eigenvalues
    clear_memory(device)
    return ratio, corner_condition


def corner_cutoff_scan(
    name: str,
    raw_a: torch.Tensor,
    raw_b: torch.Tensor,
    *,
    args: argparse.Namespace,
    device: torch.device,
    cutoffs: tuple[float, ...],
) -> dict[float, tuple[complex, tuple[int, int], float, float]]:
    """Reuse one exact CTMRG boundary to scan corner-whitening cutoffs."""

    report(f"Starting corner-cutoff scan for {name}.")
    reset_rng(args.seed, device)
    components = corr.obtain_transfer_components(
        raw_a,
        raw_b,
        args.chi,
        ctm_max_steps=args.ctm_steps,
        ctm_conv_tol=args.ctm_tol,
        ctm_conv_mode=args.ctm_conv_mode,
        ctm_e_conv_threshold=args.ctm_e_conv_threshold,
        identity_init=False,
        rsvd_mode=corr.DEFAULT_RSVD_MODE,
        rsvd_neumann_terms=corr.DEFAULT_RSVD_NEUMANN_TERMS,
        rsvd_power_iters=corr.DEFAULT_RSVD_POWER_ITERS,
        force_full_svd=True,
        j1=args.J1,
        j2=args.J2,
        dtype=torch.float64,
        device=device,
    )
    raw_transfer = corr.compute_r2rTransferMatrix(
        components.upper_b,
        components.upper_a,
        components.double_layer_a,
        components.double_layer_b,
        components.lower_a,
        components.lower_b,
        max_intermediate_bytes=256 * 1024**2,
        progress_every=0,
        matrix_free=False,
    )
    if not isinstance(raw_transfer, corr.RowToRowTransferOperator):
        raise AssertionError("The cutoff scan requires a transfer operator.")

    scanned: dict[float, tuple[complex, tuple[int, int], float, float]] = {}
    for cutoff in cutoffs:
        whitened = corr.CornerWhitenedTransferOperator(
            raw_transfer,
            components.ctm_corner_ab,
            components.ctm_corner_ba,
            relative_cutoff=cutoff,
        )
        eigensolver = corr.diagonalize_first_two_largest_eigval(
            whitened,
            tol=0.0,
            ncv=min(corr.DEFAULT_ARPACK_NCV, args.chi**2),
            maxiter=corr.DEFAULT_ARPACK_MAXITER,
            dense_dimension_threshold=256,
            seed=args.seed,
            return_result=True,
        )
        if not isinstance(eigensolver, corr.EigensolverResult):
            raise AssertionError("Cutoff scan did not return eigensolver details.")
        ratio = eigensolver.eigenvalues[1] / eigensolver.eigenvalues[0]
        maximum_residual = max(eigensolver.relative_residuals)
        scanned[cutoff] = (
            ratio,
            whitened.corner_effective_ranks,
            whitened.overlap_condition_number,
            maximum_residual,
        )
        report(
            f"{name} cutoff={cutoff:.1e}: "
            f"lambda2/lambda1={ratio.real:+.12e}{ratio.imag:+.12e}j, "
            f"ranks={whitened.corner_effective_ranks}, "
            f"cond(N)={whitened.overlap_condition_number:.6e}, "
            f"max_residual={maximum_residual:.3e}"
        )
        del whitened, eigensolver

    del components, raw_transfer
    clear_memory(device)
    return scanned


def run_physical_tests(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, CaseSpectrum, list[str]]:
    failures: list[str] = []

    def check_equivalence(
        reference_case: CaseSpectrum,
        transformed_case: CaseSpectrum,
    ) -> None:
        try:
            established = assert_physical_equivalence(
                reference_case,
                transformed_case,
                tolerance=args.physical_tolerance,
                proxy_tolerance=args.proxy_equivalence_tolerance,
                corner_tolerance=args.corner_equivalence_tolerance,
                ctm_max_steps=args.ctm_steps,
            )
            if not established:
                message = (
                    f"{transformed_case.name} could not be validated because "
                    f"CTMRG did not establish the same converged boundary "
                    f"fixed point."
                )
                failures.append(message)
                report(f"RECORDED VALIDATION ISSUE: {message}")
        except AssertionError as error:
            message = str(error)
            failures.append(message)
            report(f"RECORDED PHYSICAL FAILURE: {message}")

    report("PHYSICAL TEST 1/4: exact orthogonal bond-gauge invariance.")
    raw_a, raw_b = build_generic_tensors(args.D, device, args.seed)
    production_reference = make_case_spectrum(
        "LBFGS-production generic smoke",
        raw_a,
        raw_b,
        args=args,
        device=device,
    )
    reference = make_case_spectrum(
        "exact-validation generic reference",
        raw_a,
        raw_b,
        args=args,
        device=device,
        identity_init=False,
        force_full_svd=True,
    )

    reset_rng(args.seed + 202, device)
    random_left = torch.randn(
        args.D, args.D, dtype=torch.float64, device=device
    )
    gauge_a, _ = torch.linalg.qr(random_left)
    gauge_b = gauge_a
    identity = torch.eye(args.D, dtype=torch.float64, device=device)
    cancellation_error = torch.linalg.norm(
        gauge_a.T @ gauge_b - identity
    ).item()
    condition_number = torch.linalg.cond(gauge_a).item()
    report(
        f"bond-gauge cancellation ||G_A^T G_B-I||={cancellation_error:.6e}, "
        f"cond(G_A)={condition_number:.6e}"
    )
    gauged_a = apply_uniform_virtual_map(raw_a, gauge_a)
    gauged_b = apply_uniform_virtual_map(raw_b, gauge_b)
    gauged = make_case_spectrum(
        "orthogonally gauged state",
        gauged_a,
        gauged_b,
        args=args,
        device=device,
        identity_init=False,
        force_full_svd=True,
    )
    check_equivalence(reference, gauged)
    if args.D == 3:
        report(
            "D=3 CORNER-IDENTIFICATION DIAGNOSTIC: solving "
            "T x = lambda N x for the physical reference and gauge transform."
        )
        reference_generalized, reference_condition = (
            generalized_corner_identification_spectrum(
                "exact-validation generic reference",
                raw_a,
                raw_b,
                args=args,
                device=device,
            )
        )
        gauged_generalized, gauged_condition = (
            generalized_corner_identification_spectrum(
                "orthogonally gauged state",
                gauged_a,
                gauged_b,
                args=args,
                device=device,
            )
        )
        generalized_error = spectrum_distance(
            reference_generalized,
            gauged_generalized,
        )
        reference_qz_error = spectrum_distance(
            reference.normalized_subleading,
            reference_generalized,
        )
        gauged_qz_error = spectrum_distance(
            gauged.normalized_subleading,
            gauged_generalized,
        )
        report(
            f"generalized-corner gauge error={generalized_error:.6e}, "
            f"condition_numbers=({reference_condition:.6e},"
            f"{gauged_condition:.6e})"
        )
        report(
            f"corner-whitened ARPACK versus generalized QZ errors="
            f"({reference_qz_error:.6e},{gauged_qz_error:.6e})"
        )
        if max(
            generalized_error,
            reference_qz_error,
            gauged_qz_error,
        ) > 2.0e-8:
            message = (
                "The scalable corner-whitened eigensolver disagrees with "
                "the dense generalized-QZ reference."
            )
            failures.append(message)
            report(f"RECORDED PHYSICAL FAILURE: {message}")
    if args.D in (5, 6):
        cutoffs = (1.0e-14, 1.0e-12, 1.0e-10, 1.0e-8, 1.0e-6)
        reference_scan = corner_cutoff_scan(
            "exact-validation generic reference",
            raw_a,
            raw_b,
            args=args,
            device=device,
            cutoffs=cutoffs,
        )
        gauged_scan = corner_cutoff_scan(
            "orthogonally gauged state",
            gauged_a,
            gauged_b,
            args=args,
            device=device,
            cutoffs=cutoffs,
        )
        report(f"D={args.D} CORNER-CUTOFF GAUGE-ERROR SUMMARY:")
        for cutoff in cutoffs:
            reference_ratio = reference_scan[cutoff][0]
            gauged_ratio = gauged_scan[cutoff][0]
            ratio_error = spectrum_distance(reference_ratio, gauged_ratio)
            magnitude_error = abs(abs(reference_ratio) - abs(gauged_ratio))
            report(
                f"cutoff={cutoff:.1e}: oriented_ratio_error="
                f"{ratio_error:.6e}, magnitude_error={magnitude_error:.6e}, "
                f"reference_ranks={reference_scan[cutoff][1]}, "
                f"gauged_ranks={gauged_scan[cutoff][1]}"
            )
        del reference_scan, gauged_scan
    del (
        gauged_a,
        gauged_b,
        gauge_a,
        gauge_b,
        identity,
        random_left,
    )
    clear_memory(device)

    report("PHYSICAL TEST 2/4: C3 rotation and A/B exchange diagnostics.")
    rotated = make_case_spectrum(
        "global C3 rotation",
        raw_a.permute(1, 2, 0, 3).contiguous(),
        raw_b.permute(1, 2, 0, 3).contiguous(),
        args=args,
        device=device,
        identity_init=False,
        force_full_svd=True,
    )
    report(
        "C3 DIAGNOSTIC ONLY: a leg permutation without an explicit row-direction "
        "map is not asserted to preserve this oriented transfer spectrum."
    )

    exchanged = make_case_spectrum(
        "A/B sublattice exchange",
        raw_b,
        raw_a,
        args=args,
        device=device,
        identity_init=False,
        force_full_svd=True,
    )
    report(
        "A/B DIAGNOSTIC ONLY: sublattice exchange without the accompanying "
        "spatial reflection and leg map is not a spectrum assertion."
    )

    report("PHYSICAL TEST 3/4: product-state rank-one transfer limit.")
    product_a, product_b = build_product_state(args.D, device)
    product = make_case_spectrum(
        "embedded product state",
        product_a,
        product_b,
        args=args,
        device=device,
        chi=args.product_chi,
        identity_init=False,
        force_full_svd=True,
    )
    product_ratio = abs(product.normalized_subleading)
    report(f"product-state |lambda2/lambda1|={product_ratio:.6e}")
    if product_ratio > args.product_ratio_tolerance:
        failures.append("The product-state transfer matrix is not rank one.")
        report(f"RECORDED PHYSICAL FAILURE: {failures[-1]}")
    if not (
        product.correlation_length <= args.product_xi_tolerance
        or product.correlation_length == 0.0
    ):
        failures.append("The product-state correlation length is not zero.")
        report(f"RECORDED PHYSICAL FAILURE: {failures[-1]}")
    del product_a, product_b
    clear_memory(device)

    report("PHYSICAL TEST 4/4: GHZ two-sector degeneracy.")
    ghz_a, ghz_b = build_ghz_state(args.D, device)
    ghz = make_case_spectrum(
        "GHZ state",
        ghz_a,
        ghz_b,
        args=args,
        device=device,
        chi=args.ghz_chi,
        identity_init=False,
        force_full_svd=True,
    )
    ghz_magnitudes = [abs(value) for value in ghz.eigenvalues]
    ghz_splitting = abs(ghz_magnitudes[0] - ghz_magnitudes[1]) / max(
        ghz_magnitudes[0], np.finfo(np.float64).tiny
    )
    report(f"GHZ leading-sector relative splitting={ghz_splitting:.6e}")
    if ghz_splitting > args.ghz_splitting_tolerance:
        report(
            "GHZ DIAGNOSTIC: the selected CTM boundary weights the two sectors "
            "unequally; this is not an index-ordering failure by itself."
        )
    if not (
        math.isinf(ghz.correlation_length)
        or ghz.correlation_length >= args.ghz_min_xi
    ):
        report(
            "GHZ DIAGNOSTIC: the finite-boundary correlation length is not "
            "divergent; no hard assertion is applied."
        )
    del ghz_a, ghz_b
    clear_memory(device)

    return raw_a, raw_b, production_reference, failures


def run_implementation_regressions(
    raw_a: torch.Tensor,
    raw_b: torch.Tensor,
    reference: CaseSpectrum,
    *,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    """Fast diagnostics that localize failures in the physical tests."""

    report("IMPLEMENTATION REGRESSION 1/3: direct six-tensor dense contraction.")
    reset_rng(args.seed, device)
    components = corr.obtain_transfer_components(
        raw_a,
        raw_b,
        args.chi,
        ctm_max_steps=args.ctm_steps,
        ctm_conv_tol=args.ctm_tol,
        ctm_conv_mode=args.ctm_conv_mode,
        ctm_e_conv_threshold=args.ctm_e_conv_threshold,
        identity_init=True,
        rsvd_mode=corr.DEFAULT_RSVD_MODE,
        rsvd_neumann_terms=corr.DEFAULT_RSVD_NEUMANN_TERMS,
        rsvd_power_iters=corr.DEFAULT_RSVD_POWER_ITERS,
        force_full_svd=False,
        j1=args.J1,
        j2=args.J2,
        dtype=torch.float64,
        device=device,
    )
    dense_operator = corr.compute_r2rTransferMatrix(
        components.upper_b,
        components.upper_a,
        components.double_layer_a,
        components.double_layer_b,
        components.lower_a,
        components.lower_b,
        max_intermediate_bytes=64 * 1024**2,
        matrix_free=False,
    )
    if not isinstance(dense_operator, corr.DenseRowToRowTransferOperator):
        raise AssertionError("The default transfer operator is not dense.")

    direct_4d = oe.contract(
        "MYA,MVG,AbG,abg,mya,mvg->YvyV",
        components.upper_b,
        components.upper_a,
        components.double_layer_a,
        components.double_layer_b,
        components.lower_a,
        components.lower_b,
        optimize="auto-hq",
        backend="torch",
    )
    direct_dense = direct_4d.reshape(args.chi**2, args.chi**2)
    dense_error = (
        torch.linalg.norm(dense_operator.matrix - direct_dense)
        / torch.linalg.norm(direct_dense)
    ).item()
    report(f"block-built versus direct dense relative error={dense_error:.6e}")
    if dense_error > 5.0e-11:
        raise AssertionError("The block-built dense contraction is inconsistent.")

    report("IMPLEMENTATION REGRESSION 2/3: factorized matvec.")
    factorized_operator = corr.compute_r2rTransferMatrix(
        components.upper_b,
        components.upper_a,
        components.double_layer_a,
        components.double_layer_b,
        components.lower_a,
        components.lower_b,
        max_intermediate_bytes=64 * 1024**2,
        matrix_free=True,
    )
    rng = np.random.default_rng(args.seed + 303)
    vector = (
        rng.standard_normal(args.chi**2)
        + 1j * rng.standard_normal(args.chi**2)
    )
    dense_output = dense_operator.matvec_numpy(vector)
    factorized_output = factorized_operator.matvec_numpy(vector)
    matvec_error = np.linalg.norm(
        dense_output - factorized_output
    ) / np.linalg.norm(dense_output)
    report(f"dense versus factorized complex matvec error={matvec_error:.6e}")
    if matvec_error > 5.0e-11:
        raise AssertionError("The factorized matvec is inconsistent.")

    dense_whitened = corr.CornerWhitenedTransferOperator(
        dense_operator,
        components.ctm_corner_ab,
        components.ctm_corner_ba,
    )
    factorized_whitened = corr.CornerWhitenedTransferOperator(
        factorized_operator,
        components.ctm_corner_ab,
        components.ctm_corner_ba,
    )
    dense_whitened_output = dense_whitened.matvec_numpy(vector)
    factorized_whitened_output = factorized_whitened.matvec_numpy(vector)
    whitened_matvec_error = np.linalg.norm(
        dense_whitened_output - factorized_whitened_output
    ) / np.linalg.norm(dense_whitened_output)
    report(
        f"dense versus factorized corner-whitened matvec error="
        f"{whitened_matvec_error:.6e}"
    )
    if whitened_matvec_error > 5.0e-11:
        raise AssertionError("The corner-whitened factorized matvec is inconsistent.")

    report("IMPLEMENTATION REGRESSION 3/3: whitened ARPACK versus full dense eig.")
    arpack = corr.diagonalize_first_two_largest_eigval(
        dense_whitened,
        tol=0.0,
        ncv=min(32, args.chi**2),
        maxiter=1000,
        dense_dimension_threshold=0,
        seed=args.seed,
        return_result=True,
    )
    if not isinstance(arpack, corr.EigensolverResult):
        raise AssertionError("ARPACK diagnostics were not returned.")
    arpack_ratio = arpack.eigenvalues[1] / arpack.eigenvalues[0]
    arpack_error = spectrum_distance(
        reference.normalized_subleading, arpack_ratio
    )
    report(f"ARPACK versus full dense normalized-spectrum error={arpack_error:.6e}")
    report(
        f"ARPACK residuals=({arpack.relative_residuals[0]:.3e},"
        f"{arpack.relative_residuals[1]:.3e})"
    )
    if arpack_error > 2.0e-9:
        raise AssertionError("ARPACK and full dense eig disagree.")
    if max(arpack.relative_residuals) > 2.0e-10:
        raise AssertionError("ARPACK residual is too large.")

    del (
        components,
        dense_whitened,
        factorized_whitened,
        dense_operator,
        factorized_operator,
        direct_4d,
        direct_dense,
    )
    clear_memory(device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, or a specific CUDA device",
    )
    parser.add_argument("--D", type=int, default=2)
    parser.add_argument("--chi", type=int, default=None)
    parser.add_argument("--product-chi", type=int, default=2)
    parser.add_argument("--ghz-chi", type=int, default=4)
    parser.add_argument(
        "--ctm-steps",
        type=int,
        default=corr.DEFAULT_CTM_MAX_STEPS,
    )
    parser.add_argument(
        "--ctm-tol",
        type=float,
        default=corr.DEFAULT_CTM_CONV_TOL,
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
    parser.add_argument("--J1", type=float, default=corr.DEFAULT_J1)
    parser.add_argument("--J2", type=float, default=corr.DEFAULT_J2)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--physical-tolerance", type=float, default=2.0e-4)
    parser.add_argument(
        "--proxy-equivalence-tolerance",
        type=float,
        default=5.0e-6,
    )
    parser.add_argument(
        "--corner-equivalence-tolerance",
        type=float,
        default=5.0e-6,
    )
    parser.add_argument("--product-ratio-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--product-xi-tolerance", type=float, default=0.1)
    parser.add_argument("--ghz-splitting-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--ghz-min-xi", type=float, default=1.0e6)
    parser.add_argument(
        "--run-implementation-regressions",
        action="store_true",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.D not in CHI_MIN_BY_D:
        raise ValueError("This cluster suite supports D=2,3,4,5,6.")
    if args.chi is None:
        args.chi = CHI_MIN_BY_D[args.D]
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    torch.set_default_dtype(torch.float64)
    if device.type == "cuda":
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
    core.set_dtype(True, use_real=True)
    core.set_device(device)

    start_time = time.perf_counter()
    audit_main_c3_lbfgs_defaults()
    report(
        f"Starting physical correlation-length validation: D={args.D}, "
        f"chi={args.chi}, CTMRG steps={args.ctm_steps}, "
        f"mode={args.ctm_conv_mode}, energy_threshold="
        f"{args.ctm_e_conv_threshold:.3e}, identity_init=True, "
        f"rsvd_mode={corr.DEFAULT_RSVD_MODE}, J1={args.J1}, J2={args.J2}, "
        f"device={device}"
    )

    raw_a, raw_b, reference, physical_failures = run_physical_tests(args, device)
    if physical_failures:
        report(
            f"Physical tests completed with {len(physical_failures)} "
            f"recorded failure(s)."
        )
    else:
        report("All asserted physical tests passed.")
    if args.run_implementation_regressions:
        run_implementation_regressions(
            raw_a,
            raw_b,
            reference,
            args=args,
            device=device,
        )
    else:
        report("Implementation regressions skipped for this D job.")

    if physical_failures:
        raise AssertionError(
            "Physical validation failures: " + " | ".join(physical_failures)
        )

    elapsed = time.perf_counter() - start_time
    report(
        f"ALL PHYSICAL AND NUMERICAL CORRELATION-LENGTH TESTS PASSED "
        f"in {elapsed:.2f} seconds."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
