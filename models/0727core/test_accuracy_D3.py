#!/usr/bin/env python3
"""D=3 accuracy-only comparison for choosing the production SVD mode.

No timing or memory numbers are collected here.  This script compares:

* 0717 full SVD (accuracy reference)
* 0717 Neumann and augmented
* 0727 full SVD, Neumann and augmented

It checks truncation forward/backward, the three C3-reduced energy functions,
non-identity initialization, truncation-error recording, and a short complete
differentiable CTMRG+energy calculation.  Every phase is flushed immediately
and the JSON report is rewritten after every completed case/check.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import pathlib
import re
import sys
import traceback
from typing import Any, Callable

import torch


def log(message: str = "") -> None:
    print(message, flush=True)


def load_core(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def rel_scalar(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1.0)


def rel_tensors(xs: list[torch.Tensor], ys: list[torch.Tensor]) -> float:
    if len(xs) != len(ys):
        return float("inf")
    diff2 = 0.0
    scale2 = 0.0
    for x, y in zip(xs, ys):
        xd = x.detach().to(device="cpu", dtype=torch.float64)
        yd = y.detach().to(device="cpu", dtype=torch.float64)
        diff2 += float(torch.linalg.norm(xd - yd)) ** 2
        scale2 += max(float(torch.linalg.norm(xd)),
                      float(torch.linalg.norm(yd))) ** 2
    return math.sqrt(diff2 / max(scale2, 1e-60))


def cpu_tensors(xs) -> list[torch.Tensor]:
    return [x.detach().cpu().clone() for x in xs]


def make_spin(device: torch.device) -> torch.Tensor:
    sp = torch.tensor(
        [[0.0, 1.0], [0.0, 0.0]], dtype=torch.float64, device=device)
    sm = sp.T
    sz = torch.diag(torch.tensor(
        [0.5, -0.5], dtype=torch.float64, device=device))
    return (
        0.5 * torch.einsum("ij,kl->ijkl", sp, sm)
        + 0.5 * torch.einsum("ij,kl->ijkl", sm, sp)
        + torch.einsum("ij,kl->ijkl", sz, sz)
    )


def configure(core, device: torch.device, mode: str, power_iters: int) -> None:
    core.set_dtype(use_double=True, use_real=True)
    core.set_device(device)
    core.set_rsvd_mode(
        mode, neumann_terms=2, power_iters=power_iters)
    core._USE_FULL_SVD = False
    core._CACHE_RHOS = False
    core._RHO_CACHE.clear()
    core.set_record_trunc_error(False)


def normalized_twoc3(core, raw_a, raw_b):
    return tuple(
        core.normalize_single_layer_tensor_for_double_layer(x)
        for x in core.twoc3_abcdef_from_ab(raw_a, raw_b))


class Reporter:
    def __init__(self, path: pathlib.Path, metadata: dict[str, Any]):
        self.path = path
        self.data: dict[str, Any] = {
            "metadata": metadata,
            "cases": {},
            "checks": [],
            "mode_accuracy": {},
            "errors": [],
            "passed": None,
        }
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.data, indent=2), encoding="utf-8")

    def case(self, name: str, summary: dict[str, Any]) -> None:
        self.data["cases"][name] = summary
        self.save()
        log(f"[CASE DONE] {name}: {json.dumps(summary, sort_keys=True)}")

    def check(self, name: str, value: float, limit: float) -> None:
        passed = bool(math.isfinite(value) and value <= limit)
        item = {
            "name": name, "value": value, "limit": limit, "passed": passed}
        self.data["checks"].append(item)
        self.save()
        log(f"[{'PASS' if passed else 'FAIL'}] {name}: "
            f"{value:.8e} <= {limit:.8e}")

    def error(self, name: str, exc: BaseException) -> None:
        item = {"case": name, "type": type(exc).__name__, "message": str(exc)}
        self.data["errors"].append(item)
        self.save()
        log(f"[CASE ERROR] {name}: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()


def main() -> int:
    here = pathlib.Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
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
        default=here / "accuracy_D3_results.json")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.D < 3:
        raise ValueError("Accuracy test intentionally requires D>=3")
    if args.chi > args.D ** 4:
        raise ValueError("chi must be <= D^4 for initialization comparison")

    device = torch.device(args.device)
    old = load_core("core_0717_accuracy", args.old_core.resolve())
    new = load_core("core_0727_accuracy", args.new_core.resolve())
    metadata = {
        "torch": torch.__version__,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device)
        if device.type == "cuda" else None,
        "D": args.D,
        "chi": args.chi,
        "ctm_steps": args.ctm_steps,
        "power_iters": args.power_iters,
        "seed": args.seed,
        "old_core": str(args.old_core.resolve()),
        "new_core": str(args.new_core.resolve()),
    }
    report = Reporter(args.json.resolve(), metadata)
    main_text = args.new_main.read_text(encoding="utf-8")
    configured_mode_match = re.search(
        r"(?m)^RSVD_MODE\s*=\s*['\"]([^'\"]+)['\"]", main_text)
    configured_mode = (
        configured_mode_match.group(1)
        if configured_mode_match is not None else "not-found")
    report.data["metadata"]["0727_main_configured_mode"] = configured_mode
    report.save()

    log("=" * 80)
    log("D=3 ACCURACY-ONLY TEST (NO TIMING, NO MEMORY MEASUREMENT)")
    log(json.dumps(metadata, indent=2))
    log(f"0727 main configured mode: {configured_mode}")
    log("=" * 80)

    D, chi, d = args.D, args.chi, 2
    d2 = D * D
    n = chi * d2
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    spin = make_spin(device)
    raw_a_base = torch.randn(D, D, D, d, dtype=torch.float64, device=device)
    raw_b_base = torch.randn_like(raw_a_base)
    envs = []
    for _ in range(3):
        C = torch.randn(chi, chi, dtype=torch.float64, device=device)
        T1 = torch.randn(chi, chi, d2, dtype=torch.float64, device=device)
        T2 = torch.randn_like(T1)
        envs.append(tuple(x / x.norm() for x in (C, T1, T2)))
    js = [((-1.0) ** i) * (0.17 + 0.031 * i) for i in range(36)]

    # Decaying spectrum: Neumann is only intended when the retained/discarded
    # boundary is separated.  A flat random spectrum would test a regime where
    # a two-term Neumann series is mathematically not expected to converge.
    q, _ = torch.linalg.qr(torch.randn(
        n, n, dtype=torch.float64, device=device))
    spectrum = torch.exp(
        -0.25 * torch.arange(n, dtype=torch.float64, device=device))
    mat_base = (q * spectrum) @ q.T

    # ------------------------------------------------------------------
    # C3 premise for every production-compatible ansatz.
    # ------------------------------------------------------------------
    log("[PHASE START] C3 representative premise")
    configure(old, device, "neumann", args.power_iters)
    with torch.no_grad():
        ansatz_sites = {
            "twoc3": old.twoc3_abcdef_from_ab(raw_a_base, raw_b_base),
            "c6ypi": old.c6ypi_abcdef_from_a(raw_a_base),
            "c3vypi": old.c3vypi_abcdef_from_a(raw_a_base),
            "neel": old.neel_abcdef_from_a(
                old.symmetrize_virtual_legs(raw_a_base)),
        }
        builders = (
            old.build_open_closed_env1,
            old.build_open_closed_env2,
            old.build_open_closed_env3,
        )
        for ansatz, raw_sites in ansatz_sites.items():
            sites = tuple(
                old.normalize_single_layer_tensor_for_double_layer(x)
                for x in raw_sites)
            errors = []
            for builder, env in zip(builders, envs):
                opens, _ = builder(*sites, chi, D, d, *env)
                errors += [
                    rel_tensors([opens["A"]], [opens["C"]]),
                    rel_tensors([opens["A"]], [opens["E"]]),
                    rel_tensors([opens["B"]], [opens["D"]]),
                    rel_tensors([opens["B"]], [opens["F"]]),
                ]
            report.check(f"C3 open premise {ansatz}", max(errors), 2e-13)
    log("[PHASE END] C3 representative premise")

    # ------------------------------------------------------------------
    # trunc_rhoC3: all module/mode combinations.
    # ------------------------------------------------------------------
    trunc_internal: dict[str, dict[str, Any]] = {}

    def run_trunc(core, mode: str):
        configure(core, device, mode, args.power_iters)
        m = mat_base.detach().clone().requires_grad_(True)
        torch.manual_seed(args.seed + 100)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed + 100)
        p_in, corner, p_out = core.trunc_rhoC3(m, chi, d2)
        loss = (
            0.37 * p_in.square().sum()
            + 0.19 * corner.square().sum()
            + 0.23 * p_out.square().sum())
        grad, = torch.autograd.grad(loss, (m,))
        return {
            "loss": float(loss.detach()),
            "outputs": cpu_tensors((p_in, corner, p_out)),
            "gradients": cpu_tensors((grad,)),
        }

    log("[PHASE START] truncation modes")
    for version, core in (("0717", old), ("0727", new)):
        for mode in ("full_svd", "neumann", "augmented"):
            name = f"trunc_{version}_{mode}"
            log(f"[CASE START] {name}")
            try:
                result = run_trunc(core, mode)
                trunc_internal[name] = result
                report.case(name, {"loss": result["loss"]})
            except Exception as exc:
                report.error(name, exc)
    reference = trunc_internal.get("trunc_0717_full_svd")
    if reference is not None:
        for version in ("0717", "0727"):
            for mode in ("full_svd", "neumann", "augmented"):
                name = f"trunc_{version}_{mode}"
                result = trunc_internal.get(name)
                if result is None:
                    continue
                grad_error = rel_tensors(
                    result["gradients"], reference["gradients"])
                loss_error = rel_scalar(result["loss"], reference["loss"])
                report.data["mode_accuracy"][name] = {
                    "loss_vs_full": loss_error,
                    "gradient_vs_full": grad_error,
                }
                report.save()
                log(f"[MODE ACCURACY] {name}: loss_vs_full={loss_error:.8e}, "
                    f"gradient_vs_full={grad_error:.8e}")
        for mode, limit in (("full_svd", 2e-8),
                            ("neumann", 2e-7),
                            ("augmented", 2e-5)):
            a = trunc_internal.get(f"trunc_0717_{mode}")
            b = trunc_internal.get(f"trunc_0727_{mode}")
            if a is not None and b is not None:
                report.check(
                    f"trunc 0717/0727 gradient agreement ({mode})",
                    rel_tensors(a["gradients"], b["gradients"]), limit)
                report.check(
                    f"trunc 0717/0727 loss agreement ({mode})",
                    rel_scalar(a["loss"], b["loss"]), limit)
    log("[PHASE END] truncation modes")

    # ------------------------------------------------------------------
    # Energy reduction itself contains no SVD: compare values, gradients,
    # and public rho layout for all three environments.
    # ------------------------------------------------------------------
    log("[PHASE START] three energy environments")
    old_energy = (
        old.energy_expectation_nearest_neighbor_3ebadcf_bonds,
        old.energy_expectation_nearest_neighbor_3afcbed_bonds,
        old.energy_expectation_nearest_neighbor_other_3_bonds)
    new_energy = (
        new.energy_expectation_nearest_neighbor_3ebadcf_bonds,
        new.energy_expectation_nearest_neighbor_3afcbed_bonds,
        new.energy_expectation_nearest_neighbor_other_3_bonds)

    def run_energy(core, fn, index: int):
        configure(core, device, "neumann", args.power_iters)
        ra = raw_a_base.detach().clone().requires_grad_(True)
        rb = raw_b_base.detach().clone().requires_grad_(True)
        env = tuple(x.detach().clone().requires_grad_(True) for x in envs[index])
        sites = normalized_twoc3(core, ra, rb)
        core._CACHE_RHOS = True
        core._RHO_CACHE.clear()
        energy = fn(
            *sites, *js[12 * index:12 * (index + 1)],
            spin, chi, D, d, *env)
        grads = torch.autograd.grad(energy, (ra, rb, *env))
        rhos = cpu_tensors(core._RHO_CACHE[f"env{index + 1}"][0])
        core._CACHE_RHOS = False
        core._RHO_CACHE.clear()
        return {
            "energy": float(energy.detach()),
            "gradients": cpu_tensors(grads),
            "rhos": rhos,
        }

    for index, (fo, fn) in enumerate(zip(old_energy, new_energy)):
        name_old, name_new = f"energy_0717_env{index+1}", f"energy_0727_env{index+1}"
        log(f"[CASE START] {name_old}")
        try:
            ro = run_energy(old, fo, index)
            report.case(name_old, {"energy": ro["energy"]})
            log(f"[CASE START] {name_new}")
            rn = run_energy(new, fn, index)
            report.case(name_new, {"energy": rn["energy"]})
            report.check(
                f"env{index+1} energy", rel_scalar(ro["energy"], rn["energy"]), 2e-9)
            report.check(
                f"env{index+1} gradients",
                rel_tensors(ro["gradients"], rn["gradients"]), 2e-6)
            report.check(
                f"env{index+1} cached rhos",
                rel_tensors(ro["rhos"], rn["rhos"]), 2e-8)
        except Exception as exc:
            report.error(f"energy_env{index+1}", exc)
    log("[PHASE END] three energy environments")

    # ------------------------------------------------------------------
    # Non-identity initialization under both approximate modes.
    # ------------------------------------------------------------------
    log("[PHASE START] non-identity initialization")

    def run_init(core, mode: str):
        configure(core, device, mode, args.power_iters)
        with torch.no_grad():
            sites = normalized_twoc3(core, raw_a_base, raw_b_base)
            doubles = core.abcdef_to_ABCDEF(*sites, d2)
            torch.manual_seed(args.seed + 200)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(args.seed + 200)
            return cpu_tensors(core.initialize_envCTs_C3(
                *doubles, chi, d2, identity_init=False))

    for mode, limit in (("neumann", 2e-7), ("augmented", 2e-5)):
        try:
            log(f"[CASE START] init_0717_{mode}")
            a = run_init(old, mode)
            report.case(f"init_0717_{mode}", {"completed": True})
            log(f"[CASE START] init_0727_{mode}")
            b = run_init(new, mode)
            report.case(f"init_0727_{mode}", {"completed": True})
            report.check(
                f"initialization 0717/0727 ({mode})",
                rel_tensors(a, b), limit)
        except Exception as exc:
            report.error(f"initialization_{mode}", exc)
    log("[PHASE END] non-identity initialization")

    # ------------------------------------------------------------------
    # Truncation error: compare new formula with exact full spectrum.
    # ------------------------------------------------------------------
    log("[PHASE START] truncation error")
    try:
        configure(new, device, "neumann", args.power_iters)
        with torch.no_grad():
            ct = mat_base.T
            singular = torch.linalg.svdvals(ct @ (ct @ ct))
            exact = float(
                singular[chi:].norm() / singular.norm().clamp(min=1e-30))
            new.set_record_trunc_error(True)
            torch.manual_seed(args.seed + 300)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(args.seed + 300)
            new.trunc_rhoC3(mat_base, chi, d2)
            recorded = new.get_last_trunc_error()
            new.set_record_trunc_error(False)
        error = (float("inf") if recorded is None else
                 abs(recorded - exact) / max(abs(exact), 1e-30))
        report.case("truncation_error", {
            "exact": exact, "recorded_0727": recorded, "relative_error": error})
        report.check("truncation error vs exact spectrum", error, 2e-3)
    except Exception as exc:
        report.error("truncation_error", exc)
    log("[PHASE END] truncation error")

    # ------------------------------------------------------------------
    # Short full pipeline for mode selection and 0717/0727 integration.
    # ------------------------------------------------------------------
    log("[PHASE START] short differentiable CTMRG")
    pipeline_internal: dict[str, dict[str, Any]] = {}

    def run_pipeline(core, energy_fns, mode: str):
        configure(core, device, mode, args.power_iters)
        ra = raw_a_base.detach().clone().requires_grad_(True)
        rb = raw_b_base.detach().clone().requires_grad_(True)
        sites = normalized_twoc3(core, ra, rb)
        doubles = core.abcdef_to_ABCDEF(*sites, d2)
        torch.manual_seed(args.seed + 400)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed + 400)
        all_env = core.CTMRG_from_init_to_stop(
            *doubles, chi, d2, args.ctm_steps, -1.0,
            True, energy_proxy_fn=None)
        energies = (
            energy_fns[0](*sites, *js[0:12], spin, chi, D, d, *all_env[0:3]),
            energy_fns[1](*sites, *js[12:24], spin, chi, D, d, *all_env[3:6]),
            energy_fns[2](*sites, *js[24:36], spin, chi, D, d, *all_env[6:9]),
        )
        loss = sum(energies)
        grads = torch.autograd.grad(loss, (ra, rb))
        return {
            "loss": float(loss.detach()),
            "gradients": cpu_tensors(grads),
            "ctm_steps": int(all_env[9]),
        }

    for version, core, energy_fns in (
            ("0717", old, old_energy), ("0727", new, new_energy)):
        for mode in ("full_svd", "neumann", "augmented"):
            name = f"pipeline_{version}_{mode}"
            log(f"[CASE START] {name}")
            try:
                result = run_pipeline(core, energy_fns, mode)
                pipeline_internal[name] = result
                report.case(name, {
                    "loss": result["loss"],
                    "ctm_steps": result["ctm_steps"],
                    "gradient_norms": [
                        float(torch.linalg.norm(g)) for g in result["gradients"]],
                })
            except Exception as exc:
                report.error(name, exc)

    pipe_ref = pipeline_internal.get("pipeline_0717_full_svd")
    if pipe_ref is not None:
        for version in ("0717", "0727"):
            for mode in ("full_svd", "neumann", "augmented"):
                name = f"pipeline_{version}_{mode}"
                result = pipeline_internal.get(name)
                if result is None:
                    continue
                loss_err = rel_scalar(result["loss"], pipe_ref["loss"])
                grad_err = rel_tensors(
                    result["gradients"], pipe_ref["gradients"])
                report.data["mode_accuracy"][name] = {
                    "loss_vs_full": loss_err,
                    "gradient_vs_full": grad_err,
                }
                report.save()
                log(f"[MODE ACCURACY] {name}: loss_vs_full={loss_err:.8e}, "
                    f"gradient_vs_full={grad_err:.8e}")
        for mode, limit in (("full_svd", 2e-7),
                            ("neumann", 5e-5),
                            ("augmented", 2e-4)):
            a = pipeline_internal.get(f"pipeline_0717_{mode}")
            b = pipeline_internal.get(f"pipeline_0727_{mode}")
            if a is not None and b is not None:
                report.check(
                    f"pipeline 0717/0727 loss ({mode})",
                    rel_scalar(a["loss"], b["loss"]), limit)
                report.check(
                    f"pipeline 0717/0727 gradient ({mode})",
                    rel_tensors(a["gradients"], b["gradients"]), limit)
    log("[PHASE END] short differentiable CTMRG")

    # Rank modes using the worst gradient error across truncation and pipeline.
    ranking = {}
    for mode in ("neumann", "augmented"):
        values = []
        for prefix in ("trunc_0727_", "pipeline_0727_"):
            item = report.data["mode_accuracy"].get(prefix + mode)
            if item is not None:
                values.append(item["gradient_vs_full"])
        if values:
            ranking[mode] = max(values)
    report.data["mode_ranking_worst_gradient_error"] = ranking
    if ranking:
        recommendation = min(ranking, key=ranking.get)
        report.data["recommended_mode_by_D3_accuracy"] = recommendation
        log(f"[D3 ACCURACY RECOMMENDATION] {recommendation}; "
            f"scores={json.dumps(ranking, sort_keys=True)}")

    passed = (
        not report.data["errors"]
        and all(item["passed"] for item in report.data["checks"]))
    report.data["passed"] = passed
    report.save()
    log("=" * 80)
    log(f"JSON REPORT: {report.path}")
    log(f"OVERALL: {'PASS' if passed else 'FAIL'}")
    log("=" * 80)
    return 0 if passed else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
