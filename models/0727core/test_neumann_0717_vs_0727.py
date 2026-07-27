#!/usr/bin/env python3
"""Targeted 0717-vs-0727 regression for the production Neumann path.

This test intentionally starts at D=3.  It checks only code paths changed in
0727 and writes a JSON report suitable for inspecting a cluster run:

1. C3 open-tensor orbit premise used by the representative energy path.
2. Neumann trunc_rhoC3 forward and backward.
3. All three energy environments: value, raw-parameter/environment gradients,
   cached density matrices, and CUDA peak allocated memory.
4. Non-identity contraction initialization after D^8/D^10 fusion.
5. Exact truncation-error recording against a full-SVD reference.
6. A short differentiable CTMRG+energy integration run and its gradient.

Exit status is nonzero if an accuracy threshold fails.  Memory numbers are
reported but are not hard pass/fail criteria because CUDA allocator behaviour
depends on the cluster GPU and PyTorch build.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import pathlib
import re
import sys
import time
from typing import Any, Callable

import torch


MIB = 1024.0 ** 2


def load_core(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def rel_tensor(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().to(device="cpu", dtype=torch.float64)
    b = b.detach().to(device="cpu", dtype=torch.float64)
    scale = max(float(torch.linalg.norm(a)), float(torch.linalg.norm(b)), 1e-30)
    return float(torch.linalg.norm(a - b)) / scale


def rel_scalar(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1.0)


def rel_tensor_lists(xs: list[torch.Tensor], ys: list[torch.Tensor]) -> float:
    if len(xs) != len(ys):
        return float("inf")
    diff_sq = 0.0
    scale_sq = 0.0
    for x, y in zip(xs, ys):
        xd = x.detach().to(device="cpu", dtype=torch.float64)
        yd = y.detach().to(device="cpu", dtype=torch.float64)
        diff_sq += float(torch.linalg.norm(xd - yd)) ** 2
        scale_sq += max(float(torch.linalg.norm(xd)),
                        float(torch.linalg.norm(yd))) ** 2
    return math.sqrt(diff_sq / max(scale_sq, 1e-60))


def cleanup_cuda(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()


def measured(
        fn: Callable[[], Any], device: torch.device) -> tuple[Any, float, float]:
    cleanup_cuda(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        baseline = torch.cuda.memory_allocated(device)
    else:
        baseline = 0
    start = time.perf_counter()
    result = fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device)
    else:
        peak = 0
    elapsed = time.perf_counter() - start
    return result, max(0.0, (peak - baseline) / MIB), elapsed


def make_spin_dot_spin(device: torch.device) -> torch.Tensor:
    splus = torch.tensor(
        [[0.0, 1.0], [0.0, 0.0]], dtype=torch.float64, device=device)
    sminus = splus.T
    sz = torch.diag(torch.tensor(
        [0.5, -0.5], dtype=torch.float64, device=device))
    return (
        0.5 * torch.einsum("ij,kl->ijkl", splus, sminus)
        + 0.5 * torch.einsum("ij,kl->ijkl", sminus, splus)
        + torch.einsum("ij,kl->ijkl", sz, sz)
    )


def configure_core(core, device: torch.device, power_iters: int) -> None:
    core.set_dtype(use_double=True, use_real=True)
    core.set_device(device)
    core.set_rsvd_mode(
        "neumann", neumann_terms=2, power_iters=power_iters)
    core._USE_FULL_SVD = False
    core._CACHE_RHOS = False
    core._RHO_CACHE.clear()
    core.set_record_trunc_error(False)


def normalized_twoc3_sites(core, raw_a, raw_b):
    sites = core.twoc3_abcdef_from_ab(raw_a, raw_b)
    return tuple(
        core.normalize_single_layer_tensor_for_double_layer(x) for x in sites)


def cpu_clones(xs) -> list[torch.Tensor]:
    return [x.detach().cpu().clone() for x in xs]


def main() -> int:
    here = pathlib.Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="D=3 Neumann accuracy regression: 0717 core vs 0727 core")
    parser.add_argument(
        "--old-core", type=pathlib.Path,
        default=here.parent / "0717core" / "core_C3.py")
    parser.add_argument(
        "--new-core", type=pathlib.Path, default=here / "core_C3.py")
    parser.add_argument(
        "--new-main", type=pathlib.Path, default=here / "main_C3.py")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--D", type=int, default=3)
    parser.add_argument("--chi", type=int, default=12)
    parser.add_argument("--ctm-steps", type=int, default=3)
    parser.add_argument("--power-iters", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument(
        "--json", type=pathlib.Path,
        default=here / "neumann_0717_vs_0727_results.json")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    device = torch.device(args.device)
    if args.D < 3:
        raise ValueError("This comparison intentionally starts at D>=3")
    if args.chi > args.D ** 4:
        raise ValueError(
            "chi must not exceed D^4 because non-identity initialization "
            "is part of this test")

    old_path = args.old_core.resolve()
    new_path = args.new_core.resolve()
    if not old_path.is_file():
        raise FileNotFoundError(f"0717 core not found: {old_path}")
    if not new_path.is_file():
        raise FileNotFoundError(f"0727 core not found: {new_path}")

    old = load_core("core_C3_0717_test", old_path)
    new = load_core("core_C3_0727_test", new_path)
    configure_core(old, device, args.power_iters)
    configure_core(new, device, args.power_iters)

    main_text = args.new_main.read_text(encoding="utf-8")
    production_neumann = bool(re.search(
        r"(?m)^RSVD_MODE\s*=\s*['\"]neumann['\"]", main_text))

    print("=" * 78)
    print("0717 vs 0727 production-Neumann regression")
    print(f"torch={torch.__version__} device={device}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(device)}")
    print(f"D={args.D} chi={args.chi} ctm_steps={args.ctm_steps} "
          f"power_iters={args.power_iters} seed={args.seed}")
    print(f"old={old_path}")
    print(f"new={new_path}")
    print("=" * 78)

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    D = args.D
    chi = args.chi
    d_phys = 2
    d_sq = D * D
    n = chi * d_sq
    spin_op = make_spin_dot_spin(device)

    raw_a_base = torch.randn(
        D, D, D, d_phys, dtype=torch.float64, device=device)
    raw_b_base = torch.randn_like(raw_a_base)
    env_bases = []
    for _ in range(3):
        corner = torch.randn(chi, chi, dtype=torch.float64, device=device)
        edge1 = torch.randn(
            chi, chi, d_sq, dtype=torch.float64, device=device)
        edge2 = torch.randn_like(edge1)
        env_bases.append(tuple(
            x / torch.linalg.norm(x) for x in (corner, edge1, edge2)))
    # A decaying, well-resolved corner spectrum is more representative of a
    # converged CTM than a flat-spectrum Ginibre matrix (for which a two-term
    # Neumann approximation is not expected to converge).
    q_corner, _ = torch.linalg.qr(torch.randn(
        n, n, dtype=torch.float64, device=device))
    corner_spectrum = torch.exp(
        -0.25 * torch.arange(n, dtype=torch.float64, device=device))
    mat_base = (q_corner * corner_spectrum) @ q_corner.T
    js = [
        ((-1.0) ** i) * (0.17 + 0.031 * i)
        for i in range(36)
    ]

    checks: list[dict[str, Any]] = []
    memory: dict[str, Any] = {}
    timings: dict[str, Any] = {}

    def add_check(name: str, value: float, limit: float) -> None:
        passed = bool(math.isfinite(value) and value <= limit)
        checks.append({
            "name": name, "value": value, "limit": limit, "passed": passed})
        mark = "PASS" if passed else "FAIL"
        print(f"[{mark}] {name:<46} {value:.6e} <= {limit:.6e}")

    add_check("0727 main production mode is Neumann",
              0.0 if production_neumann else 1.0, 0.0)

    # ------------------------------------------------------------------
    # 1. Verify the exact C3 orbit premise used by the new energy path.
    # ------------------------------------------------------------------
    with torch.no_grad():
        open_builders = (
            old.build_open_closed_env1,
            old.build_open_closed_env2,
            old.build_open_closed_env3,
        )
        ansatz_sites = {
            "twoc3": old.twoc3_abcdef_from_ab(raw_a_base, raw_b_base),
            "c6ypi": old.c6ypi_abcdef_from_a(raw_a_base),
            "c3vypi": old.c3vypi_abcdef_from_a(raw_a_base),
            "neel": old.neel_abcdef_from_a(
                old.symmetrize_virtual_legs(raw_a_base)),
        }
        for ansatz_name, raw_sites in ansatz_sites.items():
            sites = tuple(
                old.normalize_single_layer_tensor_for_double_layer(x)
                for x in raw_sites)
            orbit_errors = []
            for builder, env in zip(open_builders, env_bases):
                opens, _ = builder(
                    *sites, chi, D, d_phys, *env)
                orbit_errors.extend([
                    rel_tensor(opens["A"], opens["C"]),
                    rel_tensor(opens["A"], opens["E"]),
                    rel_tensor(opens["B"], opens["D"]),
                    rel_tensor(opens["B"], opens["F"]),
                ])
                del opens
            add_check(
                f"C3 open premise ({ansatz_name})",
                max(orbit_errors), 2e-13)

    # ------------------------------------------------------------------
    # 2. Neumann truncation forward/backward.
    # ------------------------------------------------------------------
    def run_trunc(core):
        m = mat_base.detach().clone().requires_grad_(True)
        torch.manual_seed(args.seed + 101)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed + 101)
        p_in, corner, p_out = core.trunc_rhoC3(m, chi, d_sq)
        loss = (
            0.37 * p_in.square().sum()
            + 0.19 * corner.square().sum()
            + 0.23 * p_out.square().sum()
        )
        grad, = torch.autograd.grad(loss, (m,))
        result = {
            "outputs": cpu_clones((p_in, corner, p_out)),
            "gradients": cpu_clones((grad,)),
            "loss": float(loss.detach()),
        }
        del m, p_in, corner, p_out, loss, grad
        return result

    trunc_old, mem_old, time_old = measured(lambda: run_trunc(old), device)
    trunc_new, mem_new, time_new = measured(lambda: run_trunc(new), device)
    memory["trunc_rhoC3_mib"] = {"0717": mem_old, "0727": mem_new}
    timings["trunc_rhoC3_seconds"] = {"0717": time_old, "0727": time_new}
    add_check(
        "Neumann trunc forward outputs",
        rel_tensor_lists(trunc_old["outputs"], trunc_new["outputs"]), 2e-9)
    add_check(
        "Neumann trunc scalar loss",
        rel_scalar(trunc_old["loss"], trunc_new["loss"]), 2e-10)
    add_check(
        "Neumann trunc backward dL/dmatC",
        rel_tensor_lists(
            trunc_old["gradients"], trunc_new["gradients"]), 2e-7)

    # Dense full-SVD reference: this additionally quantifies Neumann's own
    # approximation error, instead of merely proving that 0717 and 0727 agree.
    old.set_rsvd_mode(
        "full_svd", neumann_terms=2, power_iters=args.power_iters)
    trunc_full, mem_full, time_full = measured(lambda: run_trunc(old), device)
    old.set_rsvd_mode(
        "neumann", neumann_terms=2, power_iters=args.power_iters)
    memory["trunc_full_svd_reference_mib"] = mem_full
    timings["trunc_full_svd_reference_seconds"] = time_full
    old_neumann_vs_full = rel_tensor_lists(
        trunc_old["gradients"], trunc_full["gradients"])
    new_neumann_vs_full = rel_tensor_lists(
        trunc_new["gradients"], trunc_full["gradients"])
    add_check(
        "0717 Neumann gradient vs full-SVD reference",
        old_neumann_vs_full, 5e-2)
    add_check(
        "0727 Neumann gradient vs full-SVD reference",
        new_neumann_vs_full, 5e-2)
    allowed_new_error = old_neumann_vs_full * 1.05 + 1e-7
    add_check(
        "0727 Neumann is not less accurate than 0717",
        new_neumann_vs_full, allowed_new_error)

    # ------------------------------------------------------------------
    # 3. Three energy environments, including constrained raw gradients,
    #    environment gradients, cached rhos, and peak memory.
    # ------------------------------------------------------------------
    old_energy_fns = (
        old.energy_expectation_nearest_neighbor_3ebadcf_bonds,
        old.energy_expectation_nearest_neighbor_3afcbed_bonds,
        old.energy_expectation_nearest_neighbor_other_3_bonds,
    )
    new_energy_fns = (
        new.energy_expectation_nearest_neighbor_3ebadcf_bonds,
        new.energy_expectation_nearest_neighbor_3afcbed_bonds,
        new.energy_expectation_nearest_neighbor_other_3_bonds,
    )

    def run_energy(core, fn, env_index: int):
        raw_a = raw_a_base.detach().clone().requires_grad_(True)
        raw_b = raw_b_base.detach().clone().requires_grad_(True)
        env = tuple(
            x.detach().clone().requires_grad_(True)
            for x in env_bases[env_index])
        sites = normalized_twoc3_sites(core, raw_a, raw_b)
        core._CACHE_RHOS = True
        core._RHO_CACHE.clear()
        energy = fn(
            *sites, *js[12 * env_index:12 * (env_index + 1)],
            spin_op, chi, D, d_phys, *env)
        grads = torch.autograd.grad(energy, (raw_a, raw_b, *env))
        cached_rhos = cpu_clones(core._RHO_CACHE[f"env{env_index + 1}"][0])
        result = {
            "energy": float(energy.detach()),
            "gradients": cpu_clones(grads),
            "rhos": cached_rhos,
        }
        core._CACHE_RHOS = False
        core._RHO_CACHE.clear()
        del raw_a, raw_b, env, sites, energy, grads
        return result

    for env_index, (old_fn, new_fn) in enumerate(
            zip(old_energy_fns, new_energy_fns), start=0):
        old_result, old_peak, old_time = measured(
            lambda i=env_index, f=old_fn: run_energy(old, f, i), device)
        new_result, new_peak, new_time = measured(
            lambda i=env_index, f=new_fn: run_energy(new, f, i), device)
        key = f"env{env_index + 1}"
        memory[f"energy_{key}_mib"] = {
            "0717": old_peak, "0727": new_peak,
            "ratio_0727_over_0717": new_peak / max(old_peak, 1e-30),
        }
        timings[f"energy_{key}_seconds"] = {
            "0717": old_time, "0727": new_time,
            "ratio_0727_over_0717": new_time / max(old_time, 1e-30),
        }
        add_check(
            f"{key} energy value",
            rel_scalar(old_result["energy"], new_result["energy"]), 2e-9)
        add_check(
            f"{key} energy gradients",
            rel_tensor_lists(
                old_result["gradients"], new_result["gradients"]), 2e-6)
        add_check(
            f"{key} cached 12-rho layout",
            rel_tensor_lists(old_result["rhos"], new_result["rhos"]), 2e-8)

    # ------------------------------------------------------------------
    # 4. Fused non-identity initialization.
    # ------------------------------------------------------------------
    def run_nonidentity_init(core):
        with torch.no_grad():
            sites = normalized_twoc3_sites(core, raw_a_base, raw_b_base)
            double_sites = core.abcdef_to_ABCDEF(*sites, d_sq)
            torch.manual_seed(args.seed + 202)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(args.seed + 202)
            outputs = core.initialize_envCTs_C3(
                *double_sites, chi, d_sq, identity_init=False)
            result = cpu_clones(outputs)
            del sites, double_sites, outputs
            return result

    init_old, init_mem_old, init_time_old = measured(
        lambda: run_nonidentity_init(old), device)
    init_new, init_mem_new, init_time_new = measured(
        lambda: run_nonidentity_init(new), device)
    memory["nonidentity_initialization_mib"] = {
        "0717": init_mem_old, "0727": init_mem_new,
        "ratio_0727_over_0717": init_mem_new / max(init_mem_old, 1e-30),
    }
    timings["nonidentity_initialization_seconds"] = {
        "0717": init_time_old, "0727": init_time_new,
    }
    add_check(
        "non-identity initialization outputs",
        rel_tensor_lists(init_old, init_new), 2e-7)

    # ------------------------------------------------------------------
    # 5. New low-memory truncation-error formula vs exact full SVD.
    # ------------------------------------------------------------------
    with torch.no_grad():
        ct = mat_base.T
        cubic = ct @ (ct @ ct)
        exact_s = torch.linalg.svdvals(cubic)
        exact_error = float(
            torch.linalg.norm(exact_s[chi:])
            / torch.linalg.norm(exact_s).clamp(min=1e-30))
        new.set_record_trunc_error(True)
        torch.manual_seed(args.seed + 303)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed + 303)
        new.trunc_rhoC3(mat_base, chi, d_sq)
        recorded_error = new.get_last_trunc_error()
        new.set_record_trunc_error(False)
        if recorded_error is None:
            trunc_error_rel = float("inf")
        else:
            trunc_error_rel = (
                abs(recorded_error - exact_error) / max(abs(exact_error), 1e-30))
    add_check(
        "recorded truncation error vs exact full SVD",
        trunc_error_rel, 2e-3)

    # ------------------------------------------------------------------
    # 6. Short differentiable CTMRG+energy integration.
    # ------------------------------------------------------------------
    def run_full_pipeline(core, energy_fns):
        raw_a = raw_a_base.detach().clone().requires_grad_(True)
        raw_b = raw_b_base.detach().clone().requires_grad_(True)
        sites = normalized_twoc3_sites(core, raw_a, raw_b)
        double_sites = core.abcdef_to_ABCDEF(*sites, d_sq)
        core._CACHE_RHOS = False
        core._RHO_CACHE.clear()
        torch.manual_seed(args.seed + 404)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed + 404)
        all_env = core.CTMRG_from_init_to_stop(
            *double_sites, chi, d_sq, args.ctm_steps, -1.0,
            True, energy_proxy_fn=None)
        env1 = all_env[0:3]
        env2 = all_env[3:6]
        env3 = all_env[6:9]
        energies = [
            energy_fns[0](
                *sites, *js[0:12], spin_op, chi, D, d_phys, *env1),
            energy_fns[1](
                *sites, *js[12:24], spin_op, chi, D, d_phys, *env2),
            energy_fns[2](
                *sites, *js[24:36], spin_op, chi, D, d_phys, *env3),
        ]
        loss = sum(energies)
        grads = torch.autograd.grad(loss, (raw_a, raw_b))
        result = {
            "loss": float(loss.detach()),
            "gradients": cpu_clones(grads),
            "ctm_steps": int(all_env[9]),
        }
        del raw_a, raw_b, sites, double_sites, all_env
        del env1, env2, env3, energies, loss, grads
        return result

    full_old, full_mem_old, full_time_old = measured(
        lambda: run_full_pipeline(old, old_energy_fns), device)
    full_new, full_mem_new, full_time_new = measured(
        lambda: run_full_pipeline(new, new_energy_fns), device)
    memory["full_pipeline_mib"] = {
        "0717": full_mem_old, "0727": full_mem_new,
        "ratio_0727_over_0717": full_mem_new / max(full_mem_old, 1e-30),
    }
    timings["full_pipeline_seconds"] = {
        "0717": full_time_old, "0727": full_time_new,
        "ratio_0727_over_0717": full_time_new / max(full_time_old, 1e-30),
    }
    add_check(
        "short CTMRG total energy",
        rel_scalar(full_old["loss"], full_new["loss"]), 2e-7)
    add_check(
        "short CTMRG raw-parameter gradient",
        rel_tensor_lists(
            full_old["gradients"], full_new["gradients"]), 5e-5)
    add_check(
        "short CTMRG iteration count",
        float(abs(full_old["ctm_steps"] - full_new["ctm_steps"])), 0.0)

    passed = all(item["passed"] for item in checks)
    report = {
        "passed": passed,
        "torch_version": torch.__version__,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device)
        if device.type == "cuda" else None,
        "D": D,
        "chi": chi,
        "ctm_steps": args.ctm_steps,
        "power_iters": args.power_iters,
        "seed": args.seed,
        "old_core": str(old_path),
        "new_core": str(new_path),
        "checks": checks,
        "memory": memory,
        "timings": timings,
        "truncation_error": {
            "exact": exact_error,
            "recorded_0727": recorded_error,
            "relative_error": trunc_error_rel,
        },
        "full_pipeline": {
            "0717": {
                "loss": full_old["loss"],
                "ctm_steps": full_old["ctm_steps"],
                "gradient_norms": [
                    float(torch.linalg.norm(g)) for g in full_old["gradients"]],
            },
            "0727": {
                "loss": full_new["loss"],
                "ctm_steps": full_new["ctm_steps"],
                "gradient_norms": [
                    float(torch.linalg.norm(g)) for g in full_new["gradients"]],
            },
        },
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("-" * 78)
    print("CUDA peak allocated memory above clean baseline (MiB):")
    print(json.dumps(memory, indent=2))
    print("-" * 78)
    print("Wall times (seconds):")
    print(json.dumps(timings, indent=2))
    print("-" * 78)
    print(f"JSON report: {args.json.resolve()}")
    print("OVERALL:", "PASS" if passed else "FAIL")
    cleanup_cuda(device)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
