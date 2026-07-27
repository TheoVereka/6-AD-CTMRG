#!/usr/bin/env python3
"""Production-scale memory/time benchmark for D=8,9,10 and chi>=100.

Unlike test_accuracy_D3.py, this file performs no accuracy judgement.  Each
case is isolated, stdout is flushed at every boundary, and JSON is updated
after every case so partial data survives a later OOM or failure.

Measured stages:
  * trunc_rhoC3 forward and checkpointed forward+backward
  * Neumann and augmented modes
  * 0717 and 0727 implementations
  * one representative energy environment, forward and production-style
    outer-checkpointed forward+backward
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import pathlib
import sys
import time
import traceback
from typing import Any, Callable

import torch
from torch.utils.checkpoint import checkpoint as ckpt


MIB = 1024.0 ** 2


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


def configure(core, device: torch.device, mode: str, power_iters: int) -> None:
    core.set_dtype(use_double=True, use_real=True)
    core.set_device(device)
    core.set_rsvd_mode(
        mode, neumann_terms=2, power_iters=power_iters)
    core._USE_FULL_SVD = False
    core._CACHE_RHOS = False
    core._RHO_CACHE.clear()
    core.set_record_trunc_error(False)


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


def clean(device: torch.device) -> None:
    gc.collect()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()


class Reporter:
    def __init__(self, path: pathlib.Path, metadata: dict[str, Any]):
        self.path = path
        self.data: dict[str, Any] = {
            "metadata": metadata, "cases": {}, "case_order": []}
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.data, indent=2), encoding="utf-8")

    def record(self, name: str, result: dict[str, Any]) -> None:
        self.data["cases"][name] = result
        self.data["case_order"].append(name)
        self.save()
        log(f"[CASE RESULT] {name}: {json.dumps(result, sort_keys=True)}")


def measure_cuda(
        name: str,
        setup: Callable[[], tuple[Callable[[], Any], Callable[[], None]]],
        device: torch.device,
        report: Reporter) -> None:
    log(f"[CASE START] {name}")
    clean(device)
    teardown: Callable[[], None] = lambda: None
    try:
        operation, teardown = setup()
        torch.cuda.synchronize(device)
        baseline_alloc = torch.cuda.memory_allocated(device)
        baseline_reserved = torch.cuda.memory_reserved(device)
        torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        output = operation()
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        peak_alloc = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
        result = {
            "status": "ok",
            "seconds": elapsed,
            "baseline_allocated_mib": baseline_alloc / MIB,
            "peak_allocated_mib": peak_alloc / MIB,
            "incremental_peak_allocated_mib":
                max(0, peak_alloc - baseline_alloc) / MIB,
            "baseline_reserved_mib": baseline_reserved / MIB,
            "peak_reserved_mib": peak_reserved / MIB,
            "output": output,
        }
        report.record(name, result)
    except torch.cuda.OutOfMemoryError as exc:
        result = {
            "status": "oom",
            "error": str(exc),
            "allocated_mib": torch.cuda.memory_allocated(device) / MIB,
            "reserved_mib": torch.cuda.memory_reserved(device) / MIB,
        }
        report.record(name, result)
        log(f"[OOM] {name}: {exc}")
    except Exception as exc:
        result = {
            "status": "error",
            "type": type(exc).__name__,
            "error": str(exc),
        }
        report.record(name, result)
        log(f"[ERROR] {name}: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        try:
            teardown()
        finally:
            clean(device)
            log(f"[CASE END] {name}")


def main() -> int:
    here = pathlib.Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--old-core", type=pathlib.Path,
        default=here.parent / "0717core" / "core_C3.py")
    parser.add_argument(
        "--new-core", type=pathlib.Path, default=here / "core_C3.py")
    parser.add_argument("--D", type=int, required=True, choices=(8, 9, 10))
    parser.add_argument("--chi", type=int, required=True)
    parser.add_argument("--power-iters", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument(
        "--quick-accurate", action="store_true",
        help=("Run only the three accuracy-compatible D=9 cases needed after "
              "the projector rollback: 0727 augmented truncation backward "
              "and 0717/0727 checkpointed energy backward."))
    parser.add_argument(
        "--json", type=pathlib.Path, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Production benchmark requires CUDA")
    if args.chi < 100:
        raise ValueError("Production benchmark requires chi>=100")
    device = torch.device("cuda")
    D, chi, d = args.D, args.chi, 2
    d2 = D * D
    n = chi * d2
    json_path = (
        args.json if args.json is not None
        else here / f"production_D{D}_chi{chi}_results.json")
    old = load_core("core_0717_production", args.old_core.resolve())
    new = load_core("core_0727_production", args.new_core.resolve())
    metadata = {
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(device),
        "gpu_total_memory_mib":
            torch.cuda.get_device_properties(device).total_memory / MIB,
        "D": D,
        "chi": chi,
        "N_chi_D2": n,
        "dtype": "float64",
        "power_iters": args.power_iters,
        "seed": args.seed,
        "old_core": str(args.old_core.resolve()),
        "new_core": str(args.new_core.resolve()),
    }
    report = Reporter(json_path.resolve(), metadata)
    log("=" * 80)
    log("PRODUCTION-SCALE MEMORY/TIME BENCHMARK")
    log(json.dumps(metadata, indent=2))
    log("Each result is flushed and saved immediately.")
    log("=" * 80)

    # --------------------------------------------------------------
    # Truncation cases.  The input is allocated before counters/timer
    # reset, so baseline contains the unavoidable enlarged corner.
    # --------------------------------------------------------------
    def trunc_setup(core, mode: str, backward: bool, case_seed: int):
        configure(core, device, mode, args.power_iters)
        torch.manual_seed(case_seed)
        torch.cuda.manual_seed_all(case_seed)
        mat = (
            torch.randn(n, n, dtype=torch.float64, device=device)
            / math.sqrt(n))
        mat.requires_grad_(backward)

        def trunc_fn(x):
            return core.trunc_rhoC3(x, chi, d2)

        if backward:
            def operation():
                p_in, corner, p_out = ckpt(
                    trunc_fn, mat, use_reentrant=False)
                loss = (
                    0.37 * p_in.square().sum()
                    + 0.19 * corner.square().sum()
                    + 0.23 * p_out.square().sum())
                loss.backward()
                return {"loss": float(loss.detach())}
        else:
            def operation():
                with torch.no_grad():
                    p_in, corner, p_out = trunc_fn(mat)
                    checksum = (
                        p_in.square().sum()
                        + corner.square().sum()
                        + p_out.square().sum())
                return {"checksum": float(checksum)}

        def teardown():
            nonlocal mat
            mat = None

        return operation, teardown

    if args.quick_accurate:
        # The corrected 0727 projector path is the 0717 arithmetic/autograd
        # graph, so duplicating both versions would spend half the short job on
        # an implementation-identical case.  Backward includes checkpoint
        # recomputation and is the relevant peak.
        measure_cuda(
            "trunc_0727_augmented_forward_backward",
            lambda: trunc_setup(
                new, "augmented", True, args.seed + 1011),
            device, report)
    else:
        for mode_index, mode in enumerate(("neumann", "augmented")):
            for version, core in (("0717", old), ("0727", new)):
                for direction_index, direction in enumerate(
                        ("forward", "forward_backward")):
                    backward = direction == "forward_backward"
                    name = f"trunc_{version}_{mode}_{direction}"
                    # Same input and rSVD random stream for 0717 and 0727.
                    seed = args.seed + 1000 + 10 * mode_index + direction_index
                    measure_cuda(
                        name,
                        lambda c=core, m=mode, b=backward, s=seed:
                            trunc_setup(c, m, b, s),
                        device, report)

    # --------------------------------------------------------------
    # Energy cases.  The backward case uses the same outer checkpoint
    # structure as main_C3._three_env_energy_loss_parallel.
    # --------------------------------------------------------------
    spin = make_spin(device)
    couplings = tuple(0.5 + 0.01 * i for i in range(12))

    def energy_setup(core, backward: bool, case_seed: int):
        configure(core, device, "neumann", args.power_iters)
        torch.manual_seed(case_seed)
        torch.cuda.manual_seed_all(case_seed)
        raw_a = torch.randn(
            D, D, D, d, dtype=torch.float64, device=device)
        raw_b = torch.randn_like(raw_a)
        sites = tuple(
            core.normalize_single_layer_tensor_for_double_layer(x)
            for x in core.twoc3_abcdef_from_ab(raw_a, raw_b))
        sites = tuple(x.detach().requires_grad_(backward) for x in sites)
        C = torch.randn(
            chi, chi, dtype=torch.float64, device=device)
        T1 = torch.randn(
            chi, chi, d2, dtype=torch.float64, device=device)
        T2 = torch.randn_like(T1)
        C = (C / C.norm()).detach().requires_grad_(backward)
        T1 = (T1 / T1.norm()).detach().requires_grad_(backward)
        T2 = (T2 / T2.norm()).detach().requires_grad_(backward)
        spin_arg = spin.detach()
        fn = core.energy_expectation_nearest_neighbor_3ebadcf_bonds

        def energy_fn(*tensor_args):
            a, b, c, dd, e, f, spin_local, corner, edge1, edge2 = tensor_args
            return fn(
                a, b, c, dd, e, f, *couplings,
                spin_local, chi, D, d, corner, edge1, edge2)

        tensor_args = (*sites, spin_arg, C, T1, T2)
        if backward:
            def operation():
                energy = ckpt(
                    energy_fn, *tensor_args, use_reentrant=False)
                energy.backward()
                return {"energy": float(energy.detach())}
        else:
            def operation():
                with torch.no_grad():
                    energy = energy_fn(*tensor_args)
                return {"energy": float(energy)}

        def teardown():
            nonlocal raw_a, raw_b, sites, C, T1, T2, tensor_args
            raw_a = raw_b = sites = C = T1 = T2 = tensor_args = None

        return operation, teardown

    energy_directions = (
        ("forward_backward",) if args.quick_accurate
        else ("forward", "forward_backward"))
    for version, core in (("0717", old), ("0727", new)):
        for direction in energy_directions:
            backward = direction == "forward_backward"
            name = f"energy_env1_{version}_{direction}"
            # Same tensors for 0717 and 0727.
            seed = args.seed + 2000 + (1 if backward else 0)
            measure_cuda(
                name,
                lambda c=core, b=backward, s=seed:
                    energy_setup(c, b, s),
                device, report)

    statuses = [case["status"] for case in report.data["cases"].values()]
    expected_cases = 3 if args.quick_accurate else 12
    report.data["expected_cases"] = expected_cases
    report.data["completed_all_cases"] = len(statuses) == expected_cases
    report.data["ok_cases"] = statuses.count("ok")
    report.data["oom_cases"] = statuses.count("oom")
    report.data["error_cases"] = statuses.count("error")
    report.save()
    log("=" * 80)
    log(f"JSON REPORT: {report.path}")
    log(f"SUMMARY: ok={report.data['ok_cases']} "
        f"oom={report.data['oom_cases']} error={report.data['error_cases']}")
    log("=" * 80)
    # OOM is a benchmark result, not a harness failure.  Non-OOM exceptions
    # make the job fail so they are visible in Slurm status.
    return 1 if report.data["error_cases"] else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
