#!/usr/bin/env python3
"""Focused D=3 full-pipeline gradient regression after projector changes.

This deliberately measures no runtime or memory. Every case is printed with
flush=True and the JSON report is rewritten immediately after every result.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import sys
import traceback

import torch

import test_accuracy_D3 as common


def log(message: str = "") -> None:
    print(message, flush=True)


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
        default=here / "pipeline_D3_after_fix_results.json")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.D < 3:
        raise ValueError("Accuracy regression intentionally requires D>=3")

    device = torch.device(args.device)
    old = common.load_core("core_0717_pipeline_fix", args.old_core.resolve())
    new = common.load_core("core_0727_pipeline_fix", args.new_core.resolve())
    main_text = args.new_main.read_text(encoding="utf-8")
    mode_match = re.search(
        r"(?m)^RSVD_MODE\s*=\s*['\"]([^'\"]+)['\"]", main_text)
    configured_mode = mode_match.group(1) if mode_match else "not-found"

    report = {
        "metadata": {
            "torch": torch.__version__,
            "device": str(device),
            "gpu": (torch.cuda.get_device_name(device)
                    if device.type == "cuda" else None),
            "D": args.D,
            "chi": args.chi,
            "ctm_steps": args.ctm_steps,
            "power_iters": args.power_iters,
            "seed": args.seed,
            "old_core": str(args.old_core.resolve()),
            "new_core": str(args.new_core.resolve()),
            "0727_main_configured_mode": configured_mode,
        },
        "cases": {},
        "comparisons": {},
        "errors": [],
        "passed": None,
    }
    json_path = args.json.resolve()

    def save() -> None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    def record_case(name: str, result: dict) -> None:
        report["cases"][name] = {
            "loss": result["loss"],
            "ctm_steps": result["ctm_steps"],
            "gradient_norms": [
                float(torch.linalg.norm(g)) for g in result["gradients"]],
            "gradients_finite": all(
                bool(torch.isfinite(g).all()) for g in result["gradients"]),
        }
        save()
        log(f"[CASE DONE] {name}: "
            f"{json.dumps(report['cases'][name], sort_keys=True)}")

    save()
    log("=" * 80)
    log("D=3 PROJECTOR FIX: FULL PIPELINE GRADIENT REGRESSION")
    log("NO TIMING OR MEMORY MEASUREMENT")
    log(json.dumps(report["metadata"], indent=2))
    log("=" * 80)

    D, chi, d = args.D, args.chi, 2
    d2 = D * D
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    spin = common.make_spin(device)
    raw_a_base = torch.randn(
        D, D, D, d, dtype=torch.float64, device=device)
    raw_b_base = torch.randn_like(raw_a_base)
    js = [((-1.0) ** i) * (0.17 + 0.031 * i) for i in range(36)]

    old_energy = (
        old.energy_expectation_nearest_neighbor_3ebadcf_bonds,
        old.energy_expectation_nearest_neighbor_3afcbed_bonds,
        old.energy_expectation_nearest_neighbor_other_3_bonds)
    new_energy = (
        new.energy_expectation_nearest_neighbor_3ebadcf_bonds,
        new.energy_expectation_nearest_neighbor_3afcbed_bonds,
        new.energy_expectation_nearest_neighbor_other_3_bonds)

    def run_pipeline(core, energy_fns, mode: str) -> dict:
        common.configure(core, device, mode, args.power_iters)
        ra = raw_a_base.detach().clone().requires_grad_(True)
        rb = raw_b_base.detach().clone().requires_grad_(True)
        sites = common.normalized_twoc3(core, ra, rb)
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
        gradients = torch.autograd.grad(loss, (ra, rb))
        return {
            "loss": float(loss.detach()),
            "gradients": common.cpu_tensors(gradients),
            "ctm_steps": int(all_env[9]),
        }

    internal = {}
    for version, core, energy_fns in (
            ("0717", old, old_energy), ("0727", new, new_energy)):
        for mode in ("full_svd", "neumann", "augmented"):
            name = f"pipeline_{version}_{mode}"
            log(f"[CASE START] {name}")
            try:
                result = run_pipeline(core, energy_fns, mode)
                internal[name] = result
                record_case(name, result)
            except Exception as exc:
                report["errors"].append({
                    "case": name,
                    "type": type(exc).__name__,
                    "message": str(exc),
                })
                save()
                log(f"[CASE ERROR] {name}: {type(exc).__name__}: {exc}")
                traceback.print_exc()
                sys.stdout.flush()
                sys.stderr.flush()

    reference = internal.get("pipeline_0717_full_svd")
    if reference is not None:
        for name, result in internal.items():
            comparison = {
                "loss_vs_0717_full": common.rel_scalar(
                    result["loss"], reference["loss"]),
                "gradient_vs_0717_full": common.rel_tensors(
                    result["gradients"], reference["gradients"]),
            }
            report["comparisons"][name] = comparison
            save()
            log(f"[VS 0717 FULL] {name}: {json.dumps(comparison)}")

    limits = {"full_svd": 2e-7, "neumann": 5e-5, "augmented": 2e-4}
    old_new_pass = True
    for mode, limit in limits.items():
        old_result = internal.get(f"pipeline_0717_{mode}")
        new_result = internal.get(f"pipeline_0727_{mode}")
        if old_result is None or new_result is None:
            old_new_pass = False
            continue
        loss_error = common.rel_scalar(
            old_result["loss"], new_result["loss"])
        gradient_error = common.rel_tensors(
            old_result["gradients"], new_result["gradients"])
        comparison = {
            "loss_relative_error": loss_error,
            "gradient_relative_error": gradient_error,
            "limit": limit,
            "passed": bool(
                math.isfinite(loss_error)
                and math.isfinite(gradient_error)
                and loss_error <= limit
                and gradient_error <= limit),
        }
        report["comparisons"][f"0717_vs_0727_{mode}"] = comparison
        old_new_pass = old_new_pass and comparison["passed"]
        save()
        log(f"[{'PASS' if comparison['passed'] else 'FAIL'}] "
            f"0717/0727 {mode}: {json.dumps(comparison)}")

    finite = all(
        case["gradients_finite"] for case in report["cases"].values())
    required_modes = {"full_svd", configured_mode}
    required_comparisons = {
        mode: report["comparisons"].get(f"0717_vs_0727_{mode}")
        for mode in required_modes
    }
    report["required_modes"] = sorted(required_modes)
    report["passed"] = bool(
        not report["errors"]
        and finite
        and configured_mode == "augmented"
        and all(
            item is not None and item["passed"]
            for item in required_comparisons.values()))
    save()
    log("=" * 80)
    log(f"JSON REPORT: {json_path}")
    log(f"OVERALL: {'PASS' if report['passed'] else 'FAIL'}")
    log("=" * 80)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
