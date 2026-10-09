#!/usr/bin/env python3
"""Large-circumference Renyi-2 entropy from two-C3 CTMRG edge spectra.

The block solver uses rank-revealing orthogonalization and the two replica
sectors use exactly isometric packed coordinates. Local validation results
are recorded in tmp_tee_schematic_20261009/local_gpu_validation_20261009.
Measured spectral stability is not a rigorous unknown-tail certificate.

L counts cut D-bonds and must be even: the two-row transfer is raised to L/2.
The three normal/swapped environment pairs are (1,1), (2,3), (3,2).
Raw edge axes are (LMN, XYZ, ket*D+bra).  No corner whitening is inserted.
T1 is explicit; T2 is the supplied exact eight contractions, with output
chi batching.  All large tensors and Krylov bases use real float64.
Spectral stability is recorded as a numerical diagnostic, not a rigorous
certificate for an unknown omitted spectrum or for the CTM approximation.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

try:
    from .renyi2_spectral import solve_block
except ImportError:
    from renyi2_spectral import solve_block


PAIR_ENVIRONMENTS = {1: (1, 1), 2: (2, 3), 3: (3, 2)}
EDGE_NAMES = {1: ("T1F", "T2A"), 2: ("T1D", "T2C"), 3: ("T1B", "T2E")}


@dataclass
class Edges:
    A: torch.Tensor
    B: torch.Tensor
    C: torch.Tensor
    D: torch.Tensor
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        shape = tuple(self.A.shape)
        if len(shape) != 4 or shape[0] != shape[1] or shape[2] != shape[3]:
            raise ValueError("Canonical edges must have shape (bond_D,bond_D,chi,chi).")
        for name in "ABCD":
            value = getattr(self, name)
            if tuple(value.shape) != shape or value.dtype != torch.float64:
                raise TypeError("All four canonical edges must be real float64 with identical shapes.")
            if value.device != self.A.device or not torch.isfinite(value).all():
                raise ValueError("Edges must be finite and on one device.")

    @property
    def chi(self):
        return int(self.A.shape[2])

    @property
    def bond_D(self):
        return int(self.A.shape[0])

    def normalized(self):
        # Every edge occurs once in T1 and twice in T2, so these four scalar
        # rescalings cancel identically from -log Z2 + 2 log Z1.
        values = []
        for name in "ABCD":
            value = getattr(self, name)
            norm = torch.linalg.vector_norm(value)
            if float(norm) == 0:
                raise ValueError(f"Zero {name} edge.")
            values.append((value / norm).contiguous())
        return Edges(*values, metadata=self.metadata.copy())


def extract_pair(normal_env, swapped_env, pair: int) -> Edges:
    """Map raw C3 CTM tensors to the background document's exact ABCD axes."""
    normal_number, swapped_number = PAIR_ENVIRONMENTS[pair]
    ni, si = 3 * (normal_number - 1), 3 * (swapped_number - 1)
    n1, n2 = normal_env[ni + 1], normal_env[ni + 2]
    s1, s2 = swapped_env[si + 1], swapped_env[si + 2]
    chi = int(n1.shape[0])
    bond_D = math.isqrt(int(n1.shape[2]))
    shape = (chi, chi, bond_D, bond_D)
    if bond_D**2 != n1.shape[2]:
        raise ValueError("Fused ket/bra dimension is not a square.")
    # Left first-first joins the long LMN link; right first-first is the
    # opposite propagation direction.  Right ket/bra exchange supplies the
    # physical transpose in Tr(PQ), not an additional conjugation.
    return Edges(
        n1.reshape(shape).permute(2, 3, 1, 0).contiguous(),
        s1.reshape(shape).permute(3, 2, 0, 1).contiguous(),
        s2.reshape(shape).permute(3, 2, 1, 0).contiguous(),
        n2.reshape(shape).permute(2, 3, 0, 1).contiguous(),
    )


@torch.no_grad()
def build_t1(edges: Edges) -> torch.Tensor:
    ab = torch.einsum("abip,bajq->ijpq", edges.A, edges.B)
    dc = torch.einsum("abpi,baqj->pqij", edges.D, edges.C)
    # Rows (i,j), columns (k,l), exactly as in the supplied document.
    return torch.einsum("pqij,klpq->ijkl", dc, ab).reshape(edges.chi**2, edges.chi**2).contiguous()


class ReplicaTransfer:
    """T2=F_DC F_AB, never materialized; no truncation in a matvec."""

    def __init__(self, edges: Edges, batch: int = 8, parity: int | None = None):
        if batch < 1 or parity not in (None, 1, -1):
            raise ValueError("Invalid chi batch or replica parity.")
        self.edges = edges
        self.chi = edges.chi
        self.n = self.chi**4
        self.batch = min(int(batch), self.chi)
        self.parity = parity
        self.matvec_count = 0

    def project(self, vector):
        if self.parity is None:
            return vector
        value = vector.reshape((self.chi,) * 4)
        return ((value + self.parity * value.permute(2, 3, 0, 1)) * .5).reshape(-1)

    @torch.no_grad()
    def _ring(self, vector, first, second):
        x = vector.reshape((self.chi,) * 4)
        output = torch.empty_like(x)
        for start in range(0, self.chi, self.batch):
            end = min(start + self.batch, self.chi)
            work = torch.einsum("abip,ijkl->abpjkl", first[:, :, :, start:end], x)
            next_work = torch.einsum("bcjq,abpjkl->acpqkl", second, work)
            del work
            work = torch.einsum("cdkr,acpqkl->adpqrl", first, next_work)
            del next_work
            output[start:end] = torch.einsum("dals,adpqrl->pqrs", second, work)
            del work
        return output.reshape(-1)

    @torch.no_grad()
    def __call__(self, vector):
        if vector.dtype != torch.float64 or vector.numel() != self.n:
            raise ValueError("T2 matvec expects a real float64 chi^4 vector.")
        upper = self._ring(vector, self.edges.A, self.edges.B)
        result = self._ring(upper, self.edges.D, self.edges.C)
        del upper
        self.matvec_count += 1
        return self.project(result)


class ReplicaSector:
    """One replica sector in isometric real upper-triangle coordinates.

    Only the currently applied vector is expanded to chi**4. Q and T2Q
    retain half-sized coordinates; both sectors are still solved separately.
    The sqrt(2) factors make ordinary packed dot products the full-space
    Frobenius inner product. This changes coordinates, with no truncation.
    """

    def __init__(self, edges: Edges, batch: int = 4, parity: int = 1):
        if parity not in (1, -1):
            raise ValueError("Replica sector must be +1 or -1.")
        self.parity = int(parity)
        self.chi = edges.chi
        self.matrix_size = edges.chi**2
        self.operator = ReplicaTransfer(edges, batch=batch)
        self.rows, self.columns = torch.triu_indices(
            self.matrix_size, self.matrix_size, offset=0 if parity == 1 else 1,
            device=edges.A.device)
        self.diagonal = self.rows == self.columns if parity == 1 else None
        self.n = int(self.rows.numel())

    @property
    def matvec_count(self):
        return self.operator.matvec_count

    def unpack(self, vector):
        values = vector * (1 / math.sqrt(2))
        if self.diagonal is not None:
            values[self.diagonal] *= math.sqrt(2)
        matrix = torch.zeros((self.matrix_size, self.matrix_size),
                             dtype=vector.dtype, device=vector.device)
        matrix[self.rows, self.columns] = values
        matrix[self.columns, self.rows] = self.parity * values
        return matrix.reshape(-1)

    def pack(self, vector):
        matrix = vector.reshape(self.matrix_size, self.matrix_size)
        values = (matrix[self.rows, self.columns]
                  + self.parity * matrix[self.columns, self.rows]) * (1 / math.sqrt(2))
        if self.diagonal is not None:
            values[self.diagonal] *= 1 / math.sqrt(2)
        return values

    @torch.no_grad()
    def __call__(self, vector):
        full = self.unpack(vector)
        answer = self.operator(full)
        del full
        return self.pack(answer)


def log_trace_powers(values, lengths):
    """Integer powers retain signs and complex phases, with safe scaling."""
    values = np.asarray(values, dtype=np.complex128)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Empty or non-finite spectrum.")
    scale = float(np.max(np.abs(values)))
    if scale == 0:
        raise ValueError("Zero spectral radius.")
    powers = np.asarray(lengths, dtype=np.int64) // 2
    traces = np.sum((values[:, None] / scale) ** powers[None, :], axis=0)
    imag_relative = np.abs(traces.imag) / np.maximum(np.abs(traces.real), 1e-300)
    if np.any(imag_relative > 1e-8) or np.any(traces.real <= 0):
        raise RuntimeError("Spectral trace is complex or nonpositive; a conjugate cluster/sector may be missing.")
    return powers * math.log(scale) + np.log(traces.real), float(imag_relative.max())


def entropy_from_spectra(eig1, eig2, lengths):
    lengths = np.asarray(lengths, dtype=np.int64)
    if np.any(lengths < 2) or np.any(lengths % 2):
        raise ValueError("L counts cut bonds; positive even L is required.")
    one, im1 = log_trace_powers(eig1, lengths)
    two, im2 = log_trace_powers(eig2, lengths)
    return 2 * one - two, max(im1, im2)


def memory_budget(chi, bond_D, subspace, batch, block_size):
    vector = 8 * chi**4
    packed = 8 * (chi**2 * (chi**2 + 1) // 2)
    # Q+AQ use fixed allocations.  Orthogonalization/restart tiles and GEMM
    # reshapes need implementation-dependent additional space.
    return {
        "dtype": "float64", "one_T2_vector_GB": vector / 1e9,
        "Q_plus_AQ_GB": 2 * subspace * packed / 1e9,
        "packed_sector_vector_GB": packed / 1e9,
        "sector_index_GB": 17 * (packed // 8) / 1e9,
        "two_contraction_work_GB": 16 * bond_D**2 * min(batch, chi) * chi**3 / 1e9,
        "restart_and_block_allowance_GB": 4 * block_size * packed / 1e9 + 8 * vector / 1e9,
        "note": "Main arrays only; CUDA/BLAS workspace and restart keep-size must be included in peak RAM/VRAM.",
    }


def _serial_spectrum(result):
    values = np.asarray(result["eigenvalues"], dtype=np.complex128)
    return {"eigenvalues": [{"real": float(x.real), "imag": float(x.imag)} for x in values],
            **{key: value.tolist() if isinstance(value, np.ndarray) else value
               for key, value in result.items() if key != "eigenvalues"}}


def _solver(matvec, n, count, args, projector=None):
    # Capacity and active block width are distinct. A 64-column basis does
    # not require a 16/32-column QR temporary when requesting more roots.
    block = min(args.block_size, count, n)
    return asdict(solve_block(matvec, n, count, block_size=block, subspace=args.subspace,
                       tol=args.eig_tol, max_matvec=args.max_matvec,
                       device=args.device, seed=args.seed,
                       projector=projector, progress_every=args.progress_every))


def calculate(edges, args, output):
    edges = edges.normalized()
    lengths = np.arange(args.L_min, args.L_max + 1, args.L_step, dtype=int)
    if not len(lengths) or np.any(lengths % 2):
        raise ValueError("All requested L must be even.")
    print(f"Constructing explicit T1 ({edges.chi**2} square), real float64.", flush=True)
    t1 = build_t1(edges)
    report = {"schema": "twoc3_renyi2_spectra_v1", "length_convention": "L=cut D-bonds; exponent=L/2; geometric circumference=3L/2 honeycomb sides",
              "bond_D": edges.bond_D, "chi": edges.chi, "pair": args.pair,
              "normal_swap_envs": PAIR_ENVIRONMENTS[args.pair], "target_absolute_S2_error": args.entropy_tol,
              "memory_budget": memory_budget(edges.chi, edges.bond_D, args.subspace, args.batch,
                  args.block_size),
              "edge_metadata": edges.metadata,
              "configuration": vars(args).copy(),
              "stages": [], "status": "running", "precision_certified": False}
    report["source_sha256"] = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ("renyi2_twoc3.py", "renyi2_spectral.py")}
    previous = None
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for count in args.modes:
        stage_start = time.perf_counter()
        print(f"Spectrum stage: {count} modes per replica sector.", flush=True)
        one = _solver(lambda vector: t1 @ vector, edges.chi**2, min(count, edges.chi**2), args)
        two_results = []
        for parity in (1, -1):
            sector_dim = edges.chi**2 * (edges.chi**2 + parity) // 2
            if sector_dim == 0:
                continue
            operator = ReplicaSector(edges, args.batch, parity)
            print(f"T2 sector {parity:+d}, packed dimension {sector_dim}; starting block {args.block_size}.", flush=True)
            result = _solver(operator, operator.n, min(count, sector_dim), args)
            result["replica_parity"] = parity
            result["transfer_matvec_count"] = operator.matvec_count
            two_results.append(result)
            del operator
            gc.collect()
            if edges.A.device.type == "cuda":
                torch.cuda.empty_cache()
        stage = {"modes_requested_per_sector": count, "T1": _serial_spectrum(one),
                 "T2": [_serial_spectrum(value) for value in two_results],
                 "seconds": time.perf_counter() - stage_start}
        converged = one["converged"] and all(value["converged"] for value in two_results)
        stage["eigensolvers_converged"] = bool(converged)
        if converged:
            eig2 = np.concatenate([value["eigenvalues"] for value in two_results])
            entropies, imag = entropy_from_spectra(one["eigenvalues"], eig2, lengths)
            if np.any(entropies < -args.entropy_tol) or np.any(entropies > lengths * math.log(edges.bond_D) + args.entropy_tol):
                raise RuntimeError("S2 violates the boundary Schmidt-rank bound; stop and inspect edge mapping/spectra.")
            stage["maximum_imaginary_trace_relative"] = imag
            stage["S2"] = entropies.tolist()
            if previous is not None:
                stage["max_absolute_S2_change"] = float(np.max(np.abs(entropies - previous)))
                stage["spectral_stability_within_target"] = stage["max_absolute_S2_change"] <= args.entropy_tol
            previous = entropies
        report["stages"].append(stage)
        report["L"] = lengths.tolist()
        report["status"] = "running" if converged else "eigensolver_not_converged"
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        if not converged:
            return report
        if count >= 16 and stage.get("spectral_stability_within_target", False):
            report["stopped_after_stable_mode_increase"] = True
            print(f"S2 stable to {args.entropy_tol:g} after {count} modes; stopping mode expansion.", flush=True)
            break
    stable = bool(report["stages"][-1].get("spectral_stability_within_target", False))
    report["status"] = "spectrally_stable_estimate" if stable else "spectral_stability_not_reached"
    if edges.metadata.get("ctm_converged") is False:
        report["status"] = "ctm_not_converged"
    report["precision_note"] = "Measured mode/block stability is not a rigorous omitted-spectrum bound. CTM/D errors are separate."
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if previous is not None:
        np.savetxt(output.with_suffix(".csv"), np.column_stack((lengths, previous)), delimiter=",", header="L_cut_bonds,S2", comments="")
    return report


def _obtain_edges(args):
    if args.edge_file:
        with np.load(args.edge_file, allow_pickle=False) as archive:
            if "_pair" in archive and int(archive["_pair"]) != args.pair:
                raise ValueError("The saved canonical edges belong to another environment pair.")
            metadata = (json.loads(str(archive["_metadata_json"].item()))
                        if "_metadata_json" in archive else {})
            return Edges(*(torch.as_tensor(archive[name], dtype=torch.float64, device=args.device)
                           for name in "ABCD"), metadata=metadata)
    try:
        from . import correlation_length as cl
    except ImportError:
        import correlation_length as cl
    a, b, metadata = cl._load_twoc3_checkpoint(args.checkpoint, device=torch.device(args.device), dtype=torch.float64)
    core = cl._core
    core.set_dtype(True, use_real=True)
    core.set_device(args.device)
    core._SVD_CPU_OFFLOAD_THRESHOLD = 0
    core.set_ctm_conv_mode(args.ctm_conv_mode, e_threshold=args.ctm_e_conv_threshold)
    core._USE_FULL_SVD = args.rsvd_mode == "full_svd"
    if hasattr(core, "set_rsvd_mode"):
        core.set_rsvd_mode(args.rsvd_mode)
    environments = []
    ctm_records = []
    for first, second in ((a, b), (b, a)):
        sites, layers = cl._build_ctm_layers(first, second)
        proxy, _ = cl._build_lbfgs_energy_proxy(sites, chi=args.chi, D_bond=int(a.shape[0]), j1=args.J1, j2=args.J2)
        print("Running CTMRG(a,b)." if not environments else "Running CTMRG(b,a).", flush=True)
        step_limit = args.ctm_max_steps
        for attempt in range(args.ctm_retries + 1):
            with torch.no_grad():
                env = core.CTMRG_from_init_to_stop(*layers, args.chi, int(a.shape[0])**2,
                        step_limit, args.ctm_conv_tol, True, energy_proxy_fn=proxy)
            if int(env[-1]) < step_limit:
                break
            if attempt < args.ctm_retries:
                step_limit *= 2
                print(f"CTMRG step limit reached; retrying automatically with {step_limit} steps.", flush=True)
        ctm_records.append({"ordering": "ab" if not environments else "ba",
                            "steps": int(env[-1]), "step_limit": step_limit,
                            "converged": int(env[-1]) < step_limit})
        if int(env[-1]) >= step_limit:
            print("Warning: CTMRG reached its extended step limit; retaining its edges and recording this status.", flush=True)
        environments.append(env)
    edges = extract_pair(*environments, args.pair)
    edges.metadata.update(checkpoint=str(Path(args.checkpoint).resolve()),
                          J2=args.J2, ctm=ctm_records,
                          ctm_converged=all(item["converged"] for item in ctm_records))
    del environments
    if args.save_edges:
        np.savez(args.save_edges, _pair=np.asarray(args.pair),
                 _metadata_json=np.asarray(json.dumps(edges.metadata)),
                 **{name: getattr(edges, name).cpu().numpy() for name in "ABCD"})
    return edges


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint")
    source.add_argument("--edge-file", help="NPZ containing canonical real A,B,C,D arrays; skips CTMRG.")
    parser.add_argument("--pair", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--chi", type=int, default=80)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--J1", type=float, default=1.)
    parser.add_argument("--J2", type=float, default=None,
                        help="Required with --checkpoint so CTM energy convergence uses the correct Hamiltonian.")
    parser.add_argument("--ctm-max-steps", type=int, default=70)
    parser.add_argument("--ctm-retries", type=int, default=2,
                        help="Automatically double a reached CTM step limit before retaining its last edges.")
    parser.add_argument("--ctm-conv-tol", type=float, default=1e-7)
    parser.add_argument("--ctm-conv-mode", choices=("SVdifference", "Edifference", "both"), default="both")
    parser.add_argument("--ctm-e-conv-threshold", type=float, default=2e-8)
    parser.add_argument("--rsvd-mode", choices=("full_svd", "augmented", "neumann", "none"), default="augmented")
    parser.add_argument("--save-edges")
    parser.add_argument("--L-min", type=int, default=100)
    parser.add_argument("--L-max", type=int, default=1000)
    parser.add_argument("--L-step", type=int, default=2)
    parser.add_argument("--modes", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--subspace", type=int, default=64)
    parser.add_argument("--eig-tol", type=float, default=1e-9)
    parser.add_argument("--entropy-tol", type=float, default=1e-4)
    parser.add_argument("--max-matvec", type=int, default=2000)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.checkpoint and args.J2 is None:
        parser.error("--checkpoint requires explicit --J2 for its CTMRG energy convergence monitor.")
    if args.L_min < 2 or args.L_max < args.L_min or args.L_step < 1:
        parser.error("Invalid circumference range.")
    if any(value < 1 for value in args.modes) or sorted(set(args.modes)) != args.modes:
        parser.error("--modes must be positive and strictly increasing.")
    if args.chi < 1 or args.batch < 1 or args.block_size < 1 or args.subspace < 1 or args.max_matvec < 1:
        parser.error("Dimensions, batch, block, subspace and iteration limit must be positive.")
    if args.eig_tol <= 0 or args.entropy_tol <= 0 or args.progress_every < 0:
        parser.error("Tolerances must be positive and progress interval nonnegative.")
    if args.ctm_retries < 0 or args.ctm_max_steps < 1:
        parser.error("CTM step limit must be positive and retries nonnegative.")
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    started = time.perf_counter()
    if torch.device(args.device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(args.device)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    edges = _obtain_edges(args)
    report = calculate(edges, args, output)
    report["runtime"] = {"seconds": time.perf_counter() - started,
                         "torch_version": torch.__version__,
                         "torch_cuda_version": torch.version.cuda,
                         "device": str(edges.A.device)}
    if edges.A.device.type == "cuda":
        report["runtime"].update(
            GPU_name=torch.cuda.get_device_name(edges.A.device),
            peak_allocated_GiB=torch.cuda.max_memory_allocated(edges.A.device) / 2**30,
            peak_reserved_GiB=torch.cuda.max_memory_reserved(edges.A.device) / 2**30)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Result: {report['status']}; saved {output}", flush=True)
    return 0 if report["status"] == "spectrally_stable_estimate" else 2


if __name__ == "__main__":
    raise SystemExit(main())
