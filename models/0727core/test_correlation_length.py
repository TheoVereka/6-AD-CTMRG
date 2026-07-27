#!/usr/bin/env python3
"""Cluster validation for correlation_length.py.

This test independently checks:

1. The four edge tensors come from the intended CTMRG ordering and update.
2. The upper and lower triangle indices match their direct einsum formulas.
3. The block-built dense matrix has the exact ``V[Y,v,y,V]`` layout.
4. Dense and matrix-free matvecs agree for real and complex vectors.
5. ARPACK returns the same two leading eigenvalues as dense diagonalization.

The default D=2, chi=5 problem is intended to finish well within one hour on a
debug node.
"""

from __future__ import annotations

import argparse
import gc
import math
import sys
import time

import numpy as np
import opt_einsum as oe
import torch

import core_C3 as core
import correlation_length as corr


def report(message: str) -> None:
    print(message, flush=True)
    sys.stdout.flush()


def assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    name: str,
    rtol: float = 2.0e-11,
    atol: float = 2.0e-12,
) -> None:
    error = float(torch.max(torch.abs(actual - expected)).item())
    scale = float(torch.max(torch.abs(expected)).item())
    report(f"{name}: max_abs_error={error:.6e}, reference_scale={scale:.6e}")
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


@torch.no_grad()
def manual_ctm_ordering(
    raw_a: torch.Tensor,
    raw_b: torch.Tensor,
    *,
    chi: int,
    ctm_steps: int,
    ctm_tol: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Independently execute one CTMRG ordering and its extra 1->2 update."""

    D_squared = raw_a.shape[0] ** 2
    normalized_a = core.normalize_single_layer_tensor_for_double_layer(raw_a)
    normalized_b = core.normalize_single_layer_tensor_for_double_layer(raw_b)
    sites = core.twoc3_abcdef_from_ab(normalized_a, normalized_b)
    double_layers = core.abcdef_to_ABCDEF(*sites, D_squared)

    environment = core.CTMRG_from_init_to_stop(
        *double_layers,
        chi,
        D_squared,
        ctm_steps,
        ctm_tol,
        True,
        energy_proxy_fn=None,
    )
    updated = core.update_environmentCTs_1to2_C3(
        *environment[:3],
        *double_layers,
        chi,
        D_squared,
    )
    return (
        updated[1].detach().contiguous(),
        updated[2].detach().contiguous(),
        double_layers[0].detach().contiguous(),
        double_layers[1].detach().contiguous(),
        int(environment[-1]),
    )


def nearest_relative_error(value: complex, references: np.ndarray) -> float:
    denominator = np.maximum(np.abs(references), np.finfo(np.float64).tiny)
    return float(np.min(np.abs(references - value) / denominator))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, or a specific CUDA device",
    )
    parser.add_argument("--D", type=int, default=2)
    parser.add_argument("--chi", type=int, default=5)
    parser.add_argument("--ctm-steps", type=int, default=10)
    parser.add_argument("--ctm-tol", type=float, default=1.0e-8)
    parser.add_argument("--seed", type=int, default=20260727)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False

    core.set_dtype(True, use_real=True)
    core.set_device(device)
    core.set_rsvd_mode("augmented", neumann_terms=2, power_iters=None)

    start_time = time.perf_counter()
    report(
        f"Starting correlation-length validation: D={args.D}, chi={args.chi}, "
        f"device={device}, dtype=torch.float64"
    )

    raw_a = torch.randn(
        args.D, args.D, args.D, 2, dtype=torch.float64, device=device
    )
    raw_b = torch.randn_like(raw_a)

    report("Phase 1/5: obtaining production transfer components.")
    ctm_rng_state = torch.random.get_rng_state()
    cuda_rng_state = (
        torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    )
    components = corr.obtain_transfer_components(
        raw_a,
        raw_b,
        args.chi,
        ctm_max_steps=args.ctm_steps,
        ctm_conv_tol=args.ctm_tol,
        identity_init=True,
        dtype=torch.float64,
        device=device,
    )

    report("Phase 2/5: independently reconstructing both CTMRG edge pairs.")
    torch.random.set_rng_state(ctm_rng_state)
    if cuda_rng_state is not None:
        torch.cuda.set_rng_state(cuda_rng_state, device)
    manual_ab = manual_ctm_ordering(
        raw_a,
        raw_b,
        chi=args.chi,
        ctm_steps=args.ctm_steps,
        ctm_tol=args.ctm_tol,
    )
    manual_ba = manual_ctm_ordering(
        raw_b,
        raw_a,
        chi=args.chi,
        ctm_steps=args.ctm_steps,
        ctm_tol=args.ctm_tol,
    )

    # For (a,b), update output T1D carries the B sublattice and T2C carries A.
    assert_close(components.upper_b, manual_ab[0], name="upper_b mapping")
    assert_close(components.upper_a, manual_ab[1], name="upper_a mapping")
    # For (b,a), the same output positions carry original A and original B.
    assert_close(components.lower_a, manual_ba[0], name="lower_a mapping")
    assert_close(components.lower_b, manual_ba[1], name="lower_b mapping")
    assert_close(
        components.double_layer_a,
        manual_ab[2],
        name="double_layer_a mapping",
    )
    assert_close(
        components.double_layer_b,
        manual_ab[3],
        name="double_layer_b mapping",
    )
    report(
        f"CTMRG steps: production=({components.ctm_steps_ab},"
        f"{components.ctm_steps_ba}), manual=({manual_ab[4]},{manual_ba[4]})"
    )
    del manual_ab, manual_ba
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    report("Phase 3/5: checking triangle contractions and dense index layout.")
    upper_half, lower_half, b_chunk = corr._build_boundary_halves(
        components.upper_b,
        components.upper_a,
        components.double_layer_a,
        components.double_layer_b,
        components.lower_a,
        components.lower_b,
        max_intermediate_bytes=32 * 1024**2,
    )
    direct_upper = oe.contract(
        "AbG,MVG,MYA->bVY",
        components.double_layer_a,
        components.upper_a,
        components.upper_b,
        optimize="auto-hq",
        backend="torch",
    )
    direct_lower = oe.contract(
        "abg,mya,mvg->byv",
        components.double_layer_b,
        components.lower_a,
        components.lower_b,
        optimize="auto-hq",
        backend="torch",
    )
    assert_close(upper_half, direct_upper, name="upper_half[b,V,Y]")
    assert_close(lower_half, direct_lower, name="lower_half[b,y,v]")
    del upper_half, lower_half, direct_upper, direct_lower

    dense_operator = corr.compute_r2rTransferMatrix(
        components.upper_b,
        components.upper_a,
        components.double_layer_a,
        components.double_layer_b,
        components.lower_a,
        components.lower_b,
        max_intermediate_bytes=32 * 1024**2,
        matrix_free=False,
    )
    if not isinstance(dense_operator, corr.DenseRowToRowTransferOperator):
        raise AssertionError("The default transfer path is not dense.")

    exact_4d = oe.contract(
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
    exact_dense = exact_4d.reshape(args.chi**2, args.chi**2)
    assert_close(
        dense_operator.matrix,
        exact_dense,
        name="dense V[(Y,v),(y,V)]",
        rtol=5.0e-11,
        atol=5.0e-12,
    )
    report(f"Boundary internal-b chunk used by test: {b_chunk}")

    report("Phase 4/5: comparing dense and matrix-free real/complex matvecs.")
    matrix_free_operator = corr.compute_r2rTransferMatrix(
        components.upper_b,
        components.upper_a,
        components.double_layer_a,
        components.double_layer_b,
        components.lower_a,
        components.lower_b,
        max_intermediate_bytes=32 * 1024**2,
        matrix_free=True,
    )
    rng = np.random.default_rng(args.seed + 1)
    real_vector = rng.standard_normal(args.chi**2)
    complex_vector = real_vector + 1j * rng.standard_normal(args.chi**2)
    dense_real = dense_operator.matvec_numpy(real_vector)
    free_real = matrix_free_operator.matvec_numpy(real_vector)
    dense_complex = dense_operator.matvec_numpy(complex_vector)
    free_complex = matrix_free_operator.matvec_numpy(complex_vector)
    real_error = np.linalg.norm(dense_real - free_real) / np.linalg.norm(dense_real)
    complex_error = (
        np.linalg.norm(dense_complex - free_complex)
        / np.linalg.norm(dense_complex)
    )
    report(f"real matvec relative error: {real_error:.6e}")
    report(f"complex matvec relative error: {complex_error:.6e}")
    if real_error > 2.0e-11 or complex_error > 2.0e-11:
        raise AssertionError("Dense and matrix-free matvecs disagree.")

    report("Phase 5/5: comparing ARPACK and exact dense eigenvalues.")
    arpack = corr.diagonalize_first_two_largest_eigval(
        dense_operator,
        tol=0.0,
        ncv=min(20, args.chi**2),
        maxiter=1000,
        dense_dimension_threshold=0,
        seed=args.seed,
        return_result=True,
    )
    if not isinstance(arpack, corr.EigensolverResult):
        raise AssertionError("Detailed eigensolver result was not returned.")

    exact_values = torch.linalg.eigvals(exact_dense).detach().cpu().numpy()
    exact_values = exact_values[np.argsort(-np.abs(exact_values))]
    reference_pool = exact_values[: min(8, len(exact_values))]
    for index, value in enumerate(arpack.eigenvalues):
        error = nearest_relative_error(value, reference_pool)
        report(
            f"lambda_{index + 1}={value.real:+.16e}"
            f"{value.imag:+.16e}j, nearest exact relative error={error:.6e}, "
            f"residual={arpack.relative_residuals[index]:.6e}"
        )
        if error > 2.0e-9:
            raise AssertionError("ARPACK eigenvalue does not match dense eigvals.")
        if arpack.relative_residuals[index] > 2.0e-10:
            raise AssertionError("ARPACK residual is too large.")

    exact_xi = corr.correlation_length_from_eigenvalues(exact_values[:2])
    arpack_xi = corr.correlation_length_from_eigenvalues(arpack.eigenvalues)
    xi_error = abs(exact_xi - arpack_xi) / max(abs(exact_xi), 1.0)
    report(f"exact xi={exact_xi:.16e}")
    report(f"ARPACK xi={arpack_xi:.16e}")
    report(f"xi scaled error={xi_error:.6e}")
    if not (math.isinf(exact_xi) and math.isinf(arpack_xi)) and xi_error > 2.0e-8:
        raise AssertionError("Correlation lengths disagree.")

    elapsed = time.perf_counter() - start_time
    report(f"ALL CORRELATION-LENGTH TESTS PASSED in {elapsed:.2f} seconds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
