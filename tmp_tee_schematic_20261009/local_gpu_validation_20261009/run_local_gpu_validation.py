"""GPU validation on actual selected 0713summary two-C3 tensors, never mocks.

Only files under this dedicated verification directory are written. A failed
or incomplete case is recorded and does not abort unrelated cases. This does
not create cluster jobs or authorize cleanup. The final gate requires actual
successful computations, including dense D2 references and independent seeds.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src_code" / "scripts"))
sys.path.insert(0, str(ROOT / "tmp_tee_schematic_20261009" / "algorithm_checks"))
import renyi2_twoc3 as routine
from renyi2_spectral import solve_block
from verify_renyi2_core import explicit_transfers

SUMMARY = Path(r"D:\HyraiOn\ENS_Lyon\Internship\2026-EPFL\data\0713summary")
LENGTHS = np.arange(100, 1001, 2, dtype=np.int64)


def plain(value):
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, report):
    path.write_text(json.dumps(plain(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def clear_gpu(device):
    gc.collect()
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()


def maximum_error(left, right):
    return float(np.max(np.abs(np.asarray(left) - np.asarray(right))))


def relative_tensor_error(left, right):
    return float((torch.linalg.vector_norm(left - right) /
                  torch.linalg.vector_norm(right).clamp_min(torch.finfo(torch.float64).tiny)).item())


def checkpoint_path(d):
    return SUMMARY / "J2_0p26" / "2tensor_twoC3" / f"D_{d}" / "tensor_best.pt"


def source_identity(d):
    path = checkpoint_path(d)
    manifest = json.loads((path.parent / "manifest.json").read_text(encoding="utf-8"))
    return {"checkpoint": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "manifest": manifest, "actual_selected_0713summary_tensor": True}


def load_edges(path, device):
    with np.load(path, allow_pickle=False) as archive:
        return routine.Edges(*(torch.as_tensor(archive[name], dtype=torch.float64, device=device)
                               for name in "ABCD"))


def obtain_three_pairs(d, chi, case_dir, args, source):
    """The two expensive CTMRG calculations are shared by all three pairs."""
    metadata_path = case_dir / "ctm_metadata.json"
    edge_paths = {pair: case_dir / f"pair{pair}_edges.npz" for pair in (1, 2, 3)}
    if args.reuse_edges and metadata_path.is_file() and all(path.is_file() for path in edge_paths.values()):
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (previous.get("chi") == chi and previous.get("checkpoint_sha256") == source["sha256"]
                and previous.get("converged", False)):
            previous["reused_saved_edges"] = True
            return edge_paths, previous
        print(f"D={d}: saved edges do not match source/chi or CTMRG was incomplete; recomputing CTMRG.", flush=True)
    import correlation_length as cl
    a, b, loaded_metadata = cl._load_twoc3_checkpoint(str(checkpoint_path(d)),
                                                    device=torch.device(args.device), dtype=torch.float64)
    core = cl._core
    core.set_dtype(True, use_real=True)
    core.set_device(args.device)
    core._SVD_CPU_OFFLOAD_THRESHOLD = 0
    core.set_ctm_conv_mode(args.ctm_conv_mode, e_threshold=args.ctm_e_conv_threshold)
    core._USE_FULL_SVD = args.rsvd_mode == "full_svd"
    if hasattr(core, "set_rsvd_mode"):
        core.set_rsvd_mode(args.rsvd_mode)
    ctm = {"D": d, "chi": chi, "checkpoint_sha256": source["sha256"],
           "loaded_checkpoint_metadata": loaded_metadata, "environments": [],
           "reused_saved_edges": False, "converged": False,
           "configuration": {"initial_max_steps": args.ctm_max_steps, "max_retries": args.ctm_retries,
                             "retry_step_multiplier": 2, "conv_mode": args.ctm_conv_mode,
                             "conv_tol": args.ctm_conv_tol,
                             "energy_threshold": args.ctm_e_conv_threshold,
                             "rsvd_mode": args.rsvd_mode}}
    environments = []
    for label, first, second in (("normal_a_b", a, b), ("swapped_b_a", b, a)):
        started = time.perf_counter()
        print(f"D={d}, chi={chi}: CTMRG {label}.", flush=True)
        sites, layers = cl._build_ctm_layers(first, second)
        proxy, _ = cl._build_lbfgs_energy_proxy(sites, chi=chi, D_bond=d, j1=1., j2=.26)
        env_metadata = {"label": label, "attempts": [], "step_limit_reached": True}
        ctm["environments"].append(env_metadata)
        for attempt in range(args.ctm_retries + 1):
            step_limit = args.ctm_max_steps * 2**attempt
            attempt_started = time.perf_counter()
            with torch.no_grad():
                environment = core.CTMRG_from_init_to_stop(*layers, chi, d**2,
                        step_limit, args.ctm_conv_tol, True, energy_proxy_fn=proxy)
            synchronize(args.device)
            steps = int(environment[-1])
            reached_limit = steps >= step_limit
            env_metadata["attempts"].append({"attempt": attempt + 1, "steps": steps,
                                              "step_limit": step_limit,
                                              "step_limit_reached": reached_limit,
                                              "seconds": time.perf_counter() - attempt_started})
            env_metadata.update({"steps": steps, "step_limit": step_limit,
                                 "step_limit_reached": reached_limit, "retry_count": attempt,
                                 "seconds": time.perf_counter() - started})
            write_json(metadata_path, ctm)
            if not reached_limit or attempt == args.ctm_retries:
                break
            print(f"D={d}: {label} reached {step_limit} CTM steps; retrying from the same "
                  f"identity initialization with limit {step_limit * 2}.", flush=True)
            del environment
            clear_gpu(args.device)
        environments.append(environment)
        write_json(metadata_path, ctm)
        del sites, layers, proxy
    for pair, path in edge_paths.items():
        edges = routine.extract_pair(*environments, pair)
        np.savez(path, _pair=np.asarray(pair), _D=np.asarray(d), _chi=np.asarray(chi),
                 **{name: getattr(edges, name).detach().cpu().numpy() for name in "ABCD"})
        del edges
    ctm["converged"] = all(not item["step_limit_reached"] for item in ctm["environments"])
    write_json(metadata_path, ctm)
    del environments, a, b
    clear_gpu(args.device)
    return edge_paths, ctm


def spectrum(operator, n, k, args, *, seed):
    started = time.perf_counter()
    result = solve_block(operator, n, min(k, n), block_size=min(args.block_size, n),
                         subspace=min(args.subspace, n), tol=args.eig_tol,
                         max_matvec=args.max_matvec, device=args.device, seed=seed,
                         progress_every=args.progress_every)
    synchronize(args.device)
    answer = asdict(result)
    answer["seconds"] = time.perf_counter() - started
    return answer


def solve_sectors(edges, k, args, *, seed):
    results = []
    for parity in (1, -1):
        operator = routine.ReplicaSector(edges, batch=args.batch, parity=parity)
        if not operator.n:
            continue
        print(f"  T2 parity={parity:+}, n={operator.n}, requested={k}, seed={seed}.", flush=True)
        answer = spectrum(operator, operator.n, k, args, seed=seed + (parity < 0))
        answer["parity"] = parity
        answer["packed_dimension"] = operator.n
        answer["transfer_matvec_count"] = operator.matvec_count
        results.append(answer)
        del operator
        clear_gpu(args.device)
    return results


def joined_spectrum(results):
    return np.concatenate([np.asarray(item["eigenvalues"], dtype=np.complex128) for item in results])


def packed_checks(edges, args, dense_t2=None):
    """Random vectors probe the operator only; physical tensors remain actual."""
    rows = []
    generator = torch.Generator(device=torch.empty(0, device=args.device).device)
    generator.manual_seed(args.seed + 739)
    for parity in (1, -1):
        operator = routine.ReplicaSector(edges, batch=args.batch, parity=parity)
        if not operator.n:
            continue
        vector = torch.randn(operator.n, generator=generator, dtype=torch.float64, device=args.device)
        vector /= torch.linalg.vector_norm(vector)
        full = operator.unpack(vector)
        inverse_error = relative_tensor_error(operator.pack(full), vector)
        norm_error = abs(float(torch.linalg.vector_norm(full).item()) - 1.)
        got = operator.unpack(operator(vector))
        if dense_t2 is None:
            reference_operator = routine.ReplicaTransfer(edges, batch=args.batch)
            reference = reference_operator(full)
        else:
            reference = torch.as_tensor(dense_t2, dtype=torch.float64, device=args.device) @ full
        matvec_error = relative_tensor_error(got, reference)
        swap_error = relative_tensor_error(full.reshape((edges.chi,) * 4).permute(2, 3, 0, 1).reshape(-1), parity * full)
        row = {"parity": parity, "packed_n": operator.n,
               "pack_unpack_relative_error": inverse_error, "isometry_absolute_error": norm_error,
               "sector_parity_relative_error": swap_error, "matvec_relative_error": matvec_error,
               "reference": "independent_explicit_numpy_T2" if dense_t2 is not None else "unpacked_full_matvec",
               "passed": max(inverse_error, norm_error, swap_error, matvec_error) < 1e-10}
        rows.append(row)
        del operator, vector, full, got, reference
        clear_gpu(args.device)
    return rows


def test_pair(d, pair, path, case_dir, args):
    started = time.perf_counter()
    output = case_dir / f"pair{pair}_validation.json"
    report = {"D": d, "pair": pair, "status": "running", "passed": False,
              "lengths": LENGTHS, "absolute_S2_target": args.entropy_tol, "stages": []}
    write_json(output, report)
    edges = load_edges(path, args.device).normalized()
    t1 = routine.build_t1(edges)
    eigen1_full = np.linalg.eigvals(t1.detach().cpu().numpy())
    report["T1_dense_full_spectrum"] = eigen1_full
    dense_t2 = None
    dense_entropy = None
    if d == 2:
        values = [getattr(edges, name).detach().cpu().numpy() for name in "ABCD"]
        reference_t1, dense_t2 = explicit_transfers(values)
        dense_eigen1 = np.linalg.eigvals(reference_t1)
        dense_eigen2 = np.linalg.eigvals(dense_t2)
        dense_entropy, _ = routine.entropy_from_spectra(dense_eigen1, dense_eigen2, LENGTHS)
        t1_error = float(np.linalg.norm(t1.detach().cpu().numpy() - reference_t1) /
                         max(np.linalg.norm(reference_t1), np.finfo(float).tiny))
        report["dense_reference"] = {"dimension_T2": dense_t2.shape[0], "T1_relative_error": t1_error,
                                      "T2_full_eigenvalues": dense_eigen2, "S2": dense_entropy,
                                      "independent_numpy_contractions": True}
    report["packed_operator_checks"] = packed_checks(edges, args, dense_t2=dense_t2)
    write_json(output, report)
    previous = None
    selected = None
    for count in args.modes:
        stage_started = time.perf_counter()
        stage = {"requested_modes_per_sector": count, "seed": args.seed}
        stage["T1_iterative"] = spectrum(lambda vector: t1 @ vector, edges.chi**2, count,
                                          args, seed=args.seed)
        stage["T2_sectors"] = solve_sectors(edges, count, args, seed=args.seed)
        stage["eigensolvers_converged"] = bool(stage["T1_iterative"]["converged"] and
                  all(value["converged"] for value in stage["T2_sectors"]))
        stage["seconds"] = time.perf_counter() - stage_started
        # Failed pairs still retain eigenvalues/residuals for diagnosis. Do not
        # interpret a nonconverged spectrum as an accepted entropy estimate.
        if stage["eigensolvers_converged"]:
            eigen2 = joined_spectrum(stage["T2_sectors"])
            entropy, imaginary = routine.entropy_from_spectra(eigen1_full, eigen2, LENGTHS)
            iterative_entropy, _ = routine.entropy_from_spectra(stage["T1_iterative"]["eigenvalues"], eigen2, LENGTHS)
            stage["S2_with_full_T1"] = entropy
            stage["S2_with_iterative_T1"] = iterative_entropy
            stage["T1_iterative_entropy_absolute_error"] = maximum_error(entropy, iterative_entropy)
            stage["trace_imaginary_relative"] = imaginary
            stage["finite_and_Schmidt_bound"] = bool(np.isfinite(entropy).all() and
                  np.all(entropy >= -args.entropy_tol) and
                  np.all(entropy <= LENGTHS * math.log(d) + args.entropy_tol))
            if previous is not None:
                stage["mode_increase_S2_max_absolute_change"] = maximum_error(entropy, previous)
            if dense_entropy is not None:
                stage["dense_reference_S2_max_absolute_error"] = maximum_error(entropy, dense_entropy)
            selected = stage
            previous = entropy
        report["stages"].append(stage)
        write_json(output, report)
        if (selected is stage and count >= 16 and
            stage.get("mode_increase_S2_max_absolute_change", math.inf) < args.entropy_tol and
            stage.get("dense_reference_S2_max_absolute_error", 0.) < args.entropy_tol and
            stage.get("T1_iterative_entropy_absolute_error", math.inf) < args.entropy_tol):
            break
        clear_gpu(args.device)
    if selected is not None and selected["requested_modes_per_sector"] >= 16:
        # At least 16 independent-seed modes are used even when the earlier
        # modes were cheaper; this checks a genuinely new starting subspace.
        repeat_count = selected["requested_modes_per_sector"]
        repeat = solve_sectors(edges, repeat_count, args, seed=args.seed + 100003)
        report["independent_seed"] = {"seed": args.seed + 100003,
                                      "requested_modes_per_sector": repeat_count, "T2_sectors": repeat}
        if all(value["converged"] for value in repeat):
            entropy_repeat, _ = routine.entropy_from_spectra(eigen1_full, joined_spectrum(repeat), LENGTHS)
            error = maximum_error(selected["S2_with_full_T1"], entropy_repeat)
            report["independent_seed"]["S2"] = entropy_repeat
            report["independent_seed"]["max_absolute_S2_change"] = error
            report["independent_seed"]["within_target"] = error < args.entropy_tol
    criteria = {
        "operator_pack_isometry_matvec": all(value["passed"] for value in report["packed_operator_checks"]),
        "a_converged_stage_with_at_least_16_modes": bool(selected is not None and selected["requested_modes_per_sector"] >= 16),
        "mode_increase_stability": bool(selected and selected.get("mode_increase_S2_max_absolute_change", math.inf) < args.entropy_tol),
        "independent_seed_stability": report.get("independent_seed", {}).get("within_target", False),
        "T1_iterative_vs_full_spectrum": bool(selected and selected.get("T1_iterative_entropy_absolute_error", math.inf) < args.entropy_tol),
        "finite_and_physical_bound": bool(selected and selected.get("finite_and_Schmidt_bound", False)),
        "D2_full_dense_reference": bool(d != 2 or selected and selected.get("dense_reference_S2_max_absolute_error", math.inf) < args.entropy_tol),
    }
    report["criteria"] = criteria
    report["passed"] = all(criteria.values())
    report["status"] = "passed" if report["passed"] else "not_passed"
    report["seconds"] = time.perf_counter() - started
    report["accuracy_scope"] = "Actual tensor checks: dense D2 reference; larger D mode/seed convergence is measured evidence, not a rigorous unknown-spectrum-tail certificate."
    if selected is not None:
        np.savetxt(case_dir / f"pair{pair}_S2.csv", np.column_stack((LENGTHS, selected["S2_with_full_T1"])),
                   delimiter=",", header="L_cut_bonds,S2", comments="")
    write_json(output, report)
    del edges, t1
    clear_gpu(args.device)
    return {key: report[key] for key in ("D", "pair", "status", "passed", "criteria", "seconds")}


def run_case(d, args):
    case_dir = args.output / f"D{d}_chi{math.ceil(1.25*d*d)}"
    case_dir.mkdir(parents=True, exist_ok=True)
    output = case_dir / "case.json"
    started = time.perf_counter()
    chi = math.ceil(1.25 * d**2)
    report = {"D": d, "chi": chi, "chi_convention": "ceil(1.25*D**2)", "status": "running",
              "passed": False, "configuration": vars(args).copy(), "pairs": [],
              "device": args.device, "subspace": args.subspace, "block_size": args.block_size}
    if torch.device(args.device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(args.device)
    write_json(output, report)
    try:
        report["source"] = source_identity(d)
        paths, metadata = obtain_three_pairs(d, chi, case_dir, args, report["source"])
        report["ctm"] = metadata
        write_json(output, report)
        if args.edges_only:
            report["status"] = "edges_ready_unvalidated" if metadata["converged"] else "ctm_step_limit"
        elif not metadata["converged"]:
            report["status"] = "ctm_step_limit"
        else:
            pairs = (1, 2, 3) if d == 2 else (1,)
            for pair in pairs:
                try:
                    report["pairs"].append(test_pair(d, pair, paths[pair], case_dir, args))
                except Exception as error:
                    report["pairs"].append({"D": d, "pair": pair, "status": "exception", "passed": False,
                                            "error_type": type(error).__name__, "error": str(error),
                                            "traceback": traceback.format_exc()})
                    clear_gpu(args.device)
                write_json(output, report)
            report["passed"] = bool(report["pairs"] and all(item["passed"] for item in report["pairs"]))
            report["status"] = "passed" if report["passed"] else "not_passed"
    except Exception as error:
        report["status"] = "gpu_out_of_memory" if isinstance(error, torch.cuda.OutOfMemoryError) else "exception"
        report["error_type"] = type(error).__name__
        report["error"] = str(error)
        report["traceback"] = traceback.format_exc()
    finally:
        synchronize(args.device)
        report["seconds"] = time.perf_counter() - started
        if torch.device(args.device).type == "cuda":
            report["GPU_peak_allocated_GiB"] = torch.cuda.max_memory_allocated(args.device) / 2**30
            report["GPU_peak_reserved_GiB"] = torch.cuda.max_memory_reserved(args.device) / 2**30
        write_json(output, report)
        clear_gpu(args.device)
    print(f"D={d}, chi={chi}: {report['status']} ({report['seconds']:.1f}s).", flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--D", type=int, nargs="+", choices=(2, 3, 4, 5, 6), default=[2, 3, 4, 5, 6])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=HERE)
    parser.add_argument("--reuse-edges", action="store_true")
    parser.add_argument("--edges-only", action="store_true")
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--subspace", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--modes", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--eig-tol", type=float, default=1e-9)
    parser.add_argument("--entropy-tol", type=float, default=1e-4)
    parser.add_argument("--max-matvec", type=int, default=1600)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--ctm-max-steps", type=int, default=200)
    parser.add_argument("--ctm-retries", type=int, default=2,
                        help="Retry an exhausted CTMRG step budget with doubled limits, at most this many times.")
    parser.add_argument("--ctm-conv-tol", type=float, default=1e-7)
    parser.add_argument("--ctm-conv-mode", choices=("SVdifference", "Edifference", "both"), default="both")
    parser.add_argument("--ctm-e-conv-threshold", type=float, default=2e-8)
    parser.add_argument("--rsvd-mode", choices=("full_svd", "augmented", "neumann", "none"), default="full_svd")
    args = parser.parse_args(argv)
    if args.ctm_max_steps < 1 or args.ctm_retries < 0:
        parser.error("CTM step limit must be positive and retry count nonnegative.")
    torch.set_num_threads(args.threads)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    runtime = {"python": sys.executable, "torch_version": torch.__version__, "torch_cuda_version": torch.version.cuda,
               "cuda_available": torch.cuda.is_available(), "device": args.device,
               "actual_checkpoints": [str(checkpoint_path(d)) for d in args.D], "chi_values": {d: math.ceil(1.25*d*d) for d in args.D}}
    runtime["source_files_sha256"] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                       for path in (Path(__file__).resolve(),
                                                    ROOT / "src_code" / "scripts" / "renyi2_twoc3.py",
                                                    ROOT / "src_code" / "scripts" / "renyi2_spectral.py")}
    if runtime["cuda_available"]:
        runtime["GPU_name"] = torch.cuda.get_device_name(0)
        runtime["GPU_total_GiB"] = torch.cuda.get_device_properties(0).total_memory / 2**30
    if args.inspect:
        print(json.dumps(plain(runtime), indent=2))
        return 0
    if torch.device(args.device).type == "cuda" and not runtime["cuda_available"]:
        write_json(args.output / "runtime_blocked.json", {**runtime, "status": "CUDA_not_available", "gate_passed": False})
        print("CUDA unavailable: recorded blocked status; no physical test was run.", flush=True)
        return 2
    if not hasattr(routine, "ReplicaSector"):
        print("Packed ReplicaSector implementation is not yet present; no test was run.", flush=True)
        return 2
    try:
        from threadpoolctl import threadpool_limits
        pool = threadpool_limits(limits=args.threads)
    except ImportError:
        pool = None
    reports = []
    started = time.perf_counter()
    for d in args.D:
        reports.append(run_case(d, args))
        aggregate = {"runtime": runtime, "requested_D": args.D, "cases": reports,
                     "all_requested_cases_passed": all(item["passed"] for item in reports),
                     "all_requested_cases_completed": len(reports) == len(args.D),
                     "gate_passed": len(reports) == len(args.D) and all(item["passed"] for item in reports) and not args.edges_only,
                     "full_D2_to_D6_gate_passed": set(args.D) == {2,3,4,5,6} and len(reports) == 5 and all(item["passed"] for item in reports),
                     "total_seconds": time.perf_counter() - started,
                     "cluster_files_written": False, "cleanup_performed": False}
        write_json(args.output / "validation_summary.json", aggregate)
    if pool is not None:
        pool.restore_original_limits()
    return 0 if aggregate["gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
