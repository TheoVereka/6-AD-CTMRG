#!/usr/bin/env python3
"""Physical and numerical validation of correlation_length.py.

The primary tests are physical equivalences, not duplicate implementations:

1. An exact non-orthogonal virtual-bond gauge transformation must preserve the
   spectrum.
2. A global C3 rotation must preserve the spectrum of a two-C3 state.
3. Exchanging the A/B sublattices must preserve the spectrum up to conjugation.
4. An embedded product-state PEPS must have a rank-one transfer matrix.
5. A GHZ PEPS must have two degenerate leading transfer sectors.

Small implementation-regression checks are retained after the physical tests.
They are useful for locating a failure, but they are not treated as evidence of
physical correctness by themselves.

The generic-state tests use the exact main_C3 LBFGS CTMRG defaults and the
corresponding CHI_MIN_LIST entry.  Product and GHZ are analytic low-rank limits:
they use small sufficient chi values, symmetry-preserving direct initialization,
and deterministic full forward SVDs so boundary symmetry breaking cannot hide
their exact transfer-sector structure.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import gc
import math
import pathlib
import sys
import time

import numpy as np
import opt_einsum as oe
import torch

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
        f"residuals=({case.residuals[0]:.3e},{case.residuals[1]:.3e})"
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
    )
    print_spectrum(case)
    clear_memory(device)
    return case


def assert_physical_equivalence(
    reference: CaseSpectrum,
    transformed: CaseSpectrum,
    *,
    tolerance: float,
) -> None:
    """Compare correlation magnitude and oscillation wavevector."""

    reference_ratio = reference.normalized_subleading
    transformed_ratio = transformed.normalized_subleading
    ratio_error = spectrum_distance(reference_ratio, transformed_ratio)
    magnitude_error = abs(abs(reference_ratio) - abs(transformed_ratio))
    report(
        f"{transformed.name} versus {reference.name}: "
        f"oriented_ratio_error={ratio_error:.6e}, "
        f"magnitude_error={magnitude_error:.6e}"
    )
    if ratio_error > tolerance or magnitude_error > tolerance:
        raise AssertionError(
            f"{transformed.name} violates physical spectrum invariance."
        )


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


def run_physical_tests(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, CaseSpectrum]:
    report("PHYSICAL TEST 1/4: exact non-orthogonal bond-gauge invariance.")
    raw_a, raw_b = build_generic_tensors(args.D, device, args.seed)
    reference = make_case_spectrum(
        "generic reference", raw_a, raw_b, args=args, device=device
    )

    reset_rng(args.seed + 202, device)
    random_left = torch.randn(
        args.D, args.D, dtype=torch.float64, device=device
    )
    random_right = torch.randn_like(random_left)
    orthogonal_left, _ = torch.linalg.qr(random_left)
    orthogonal_right, _ = torch.linalg.qr(random_right)
    scales = torch.linspace(
        0.7, 1.4, args.D, dtype=torch.float64, device=device
    )
    gauge_a = orthogonal_left @ torch.diag(scales) @ orthogonal_right.T
    gauge_b = torch.linalg.inv(gauge_a).T
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
        "non-orthogonally gauged state",
        gauged_a,
        gauged_b,
        args=args,
        device=device,
    )
    assert_physical_equivalence(
        reference, gauged, tolerance=args.physical_tolerance
    )
    del (
        gauged_a,
        gauged_b,
        gauge_a,
        gauge_b,
        identity,
        orthogonal_left,
        orthogonal_right,
        random_left,
        random_right,
        scales,
    )
    clear_memory(device)

    report("PHYSICAL TEST 2/4: C3 rotation and A/B exchange invariance.")
    rotated = make_case_spectrum(
        "global C3 rotation",
        raw_a.permute(1, 2, 0, 3).contiguous(),
        raw_b.permute(1, 2, 0, 3).contiguous(),
        args=args,
        device=device,
    )
    assert_physical_equivalence(
        reference, rotated, tolerance=args.physical_tolerance
    )

    exchanged = make_case_spectrum(
        "A/B sublattice exchange",
        raw_b,
        raw_a,
        args=args,
        device=device,
    )
    assert_physical_equivalence(
        reference, exchanged, tolerance=args.physical_tolerance
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
        raise AssertionError("The product-state transfer matrix is not rank one.")
    if not (
        product.correlation_length <= args.product_xi_tolerance
        or product.correlation_length == 0.0
    ):
        raise AssertionError("The product-state correlation length is not zero.")
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
        raise AssertionError("The GHZ leading transfer sectors are not degenerate.")
    if not (
        math.isinf(ghz.correlation_length)
        or ghz.correlation_length >= args.ghz_min_xi
    ):
        raise AssertionError("The GHZ correlation length is not divergent.")
    del ghz_a, ghz_b
    clear_memory(device)

    return raw_a, raw_b, reference


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

    report("IMPLEMENTATION REGRESSION 3/3: ARPACK versus full dense eig.")
    arpack = corr.diagonalize_first_two_largest_eigval(
        dense_operator,
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

    del components, dense_operator, factorized_operator, direct_4d, direct_dense
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

    raw_a, raw_b, reference = run_physical_tests(args, device)
    report("All physical tests passed.")
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

    elapsed = time.perf_counter() - start_time
    report(
        f"ALL PHYSICAL AND NUMERICAL CORRELATION-LENGTH TESTS PASSED "
        f"in {elapsed:.2f} seconds."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
