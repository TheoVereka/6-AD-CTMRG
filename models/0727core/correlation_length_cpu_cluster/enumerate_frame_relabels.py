#!/usr/bin/env python3
"""Enumerate phase-consistent CTM edge-frame relabels for three 2C3 rows.

This is a diagnostic, not a production correlation-length driver.  One CTM
pair is computed and reused while the two chi axes of each stored T1/T2 orbit
representative are optionally exchanged.  A six-bit rule

    (env1_t1, env1_t2, env2_t1, env2_t2, env3_t1, env3_t2)

is applied identically to the (a,b) and (b,a) CTMs and to every occurrence of
that phase in the three straight-row networks.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import secrets
import time
from pathlib import Path

import numpy as np
import torch

import compute_three_generalized_correlation_lengths as batch
import correlation_length as corr
import core_C3 as core


PHASES = ("env1", "env2", "env3")


def _transpose_edge(edge: torch.Tensor, enabled: bool) -> torch.Tensor:
    if not enabled:
        return edge
    return edge.transpose(0, 1).contiguous()


def _edge(environment: object, phase: str, which: str, rule: dict[str, bool]):
    return _transpose_edge(getattr(environment, phase).__getattribute__(which), rule[f"{phase}_{which}"])


def _raw_rows(
    ab, ba, layers, rule, *, max_bytes: int,
    left_site_perm=(0, 1, 2), right_site_perm=(0, 1, 2),
    physical_grow_sites: bool = False,
    lower_swaps: dict[str, bool] | None = None,
):
    A, B, C, D, E, F = layers
    options = {"max_intermediate_bytes": max_bytes, "progress_every": 0}
    def sites(left, right):
        return (
            left.permute(*left_site_perm).contiguous(),
            right.permute(*right_site_perm).contiguous(),
        )
    if physical_grow_sites:
        # Exact next-grow rows read from the stable general-CTM einsums:
        # env1 -> env2 uses B--E, env2 -> env3 uses F--A, and
        # env3 -> env1 uses D--C.
        env2_sites = sites(F.permute(1, 2, 0), A.permute(0, 2, 1))
        env13_sites = sites(B.permute(1, 2, 0), E.permute(0, 2, 1))
        env31_sites = sites(D.permute(1, 2, 0), C.permute(0, 2, 1))
    else:
        env2_sites = sites(
            D.permute(0, 2, 1).contiguous(),
            C.permute(1, 2, 0).contiguous(),
        )
        env13_sites = sites(B.permute(1, 0, 2).contiguous(), E)
        env31_sites = sites(
            F.permute(2, 1, 0).contiguous(),
            A.permute(2, 0, 1).contiguous(),
        )
    lower_swaps = lower_swaps or {phase: False for phase in PHASES}
    def lower_pair(environment, phase):
        pair = (
            _edge(environment, phase, "t1", rule),
            _edge(environment, phase, "t2", rule),
        )
        return pair[::-1] if lower_swaps[phase] else pair
    return {
        "env2": (
            corr.SideBySideRowToRowTransferOperator(
                _edge(ab, "env2", "t2", rule),
                _edge(ab, "env2", "t1", rule),
                *env2_sites,
                *lower_pair(ba, "env2"),
                **options,
            ),
            ab.env2.corner,
            ba.env2.corner,
        ),
        "env1_ab_env3_ba": (
            corr.SideBySideRowToRowTransferOperator(
                _edge(ab, "env1", "t2", rule),
                _edge(ab, "env1", "t1", rule),
                *env13_sites,
                *lower_pair(ba, "env3"),
                **options,
            ),
            ab.env1.corner,
            ba.env3.corner,
        ),
        "env3_ab_env1_ba": (
            corr.SideBySideRowToRowTransferOperator(
                _edge(ab, "env3", "t2", rule),
                _edge(ab, "env3", "t1", rule),
                *env31_sites,
                *lower_pair(ba, "env1"),
                **options,
            ),
            ab.env3.corner,
            ba.env1.corner,
        ),
    }


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--J2", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--ctm-max-steps", type=int, default=100)
    parser.add_argument("--ctm-conv-tol", type=float, default=1.0e-8)
    parser.add_argument("--ctm-e-conv-threshold", type=float, default=2.0e-8)
    parser.add_argument("--eig-tol", type=float, default=1.0e-10)
    parser.add_argument("--arpack-ncv", type=int, default=64)
    parser.add_argument("--arpack-maxiter", type=int, default=4000)
    parser.add_argument("--corner-relative-cutoff", type=float, default=1.0e-14)
    parser.add_argument("--max-intermediate-mib", type=float, default=512.0)
    parser.add_argument("--force-full-svd", action="store_true")
    parser.add_argument(
        "--edge-rule-index", type=int, action="append",
        help="Only test these six-bit edge rules (repeatable; default: all 64).",
    )
    parser.add_argument(
        "--enumerate-corner-frames", action="store_true",
        help="Also enumerate one stored-corner transpose bit per CTM phase.",
    )
    parser.add_argument(
        "--enumerate-local-frames", action="store_true",
        help="Enumerate one common S3 permutation for every left/right site.",
    )
    parser.add_argument(
        "--independent-row-local-frames", action="store_true",
        help="List all 36 local-frame spectra separately for each row.",
    )
    parser.add_argument("--physical-grow-sites", action="store_true")
    parser.add_argument("--enumerate-lower-order", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    torch.set_num_threads(args.threads)
    seed = secrets.randbits(32) if args.seed is None else args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cpu")
    a, b, metadata = corr._load_twoc3_checkpoint(
        str(args.checkpoint.resolve()), device=device, dtype=torch.float64
    )
    D_bond = int(a.shape[0])
    chi = int(metadata["chi"])
    core.set_dtype(True, use_real=True)
    core.set_device(device)
    core._SVD_CPU_OFFLOAD_THRESHOLD = 0
    core._USE_FULL_SVD = (
        args.force_full_svd
        or D_bond in corr.FULL_SVD_CORRELATION_LENGTH_DS
    )
    core.set_rsvd_mode(
        "full_svd" if core._USE_FULL_SVD else corr.DEFAULT_RSVD_MODE,
        neumann_terms=corr.DEFAULT_RSVD_NEUMANN_TERMS,
        power_iters=corr.DEFAULT_RSVD_POWER_ITERS,
    )
    core.set_ctm_conv_mode("both", e_threshold=args.ctm_e_conv_threshold)

    started = time.perf_counter()
    print(f"CTM D={D_bond} chi={chi} J2={args.J2:g} seed={seed}", flush=True)
    ab = batch._run_three_environments(
        a, b, chi=chi, ctm_max_steps=args.ctm_max_steps,
        ctm_conv_tol=args.ctm_conv_tol, identity_init=True,
        j1=1.0, j2=args.J2, keep_double_layers=True,
    )
    ba = batch._run_three_environments(
        b, a, chi=chi, ctm_max_steps=args.ctm_max_steps,
        ctm_conv_tol=args.ctm_conv_tol, identity_init=True,
        j1=1.0, j2=args.J2, keep_double_layers=False,
    )
    if ab.double_layers is None:
        raise RuntimeError("double layers were not retained")

    bit_names = tuple(f"{phase}_{which}" for phase in PHASES for which in ("t1", "t2"))
    rows = []
    max_bytes = int(args.max_intermediate_mib * 1024**2)
    if args.independent_row_local_frames:
        if args.edge_rule_index is None or len(set(args.edge_rule_index)) != 1:
            raise ValueError(
                "--independent-row-local-frames requires exactly one "
                "--edge-rule-index"
            )
        if args.enumerate_corner_frames:
            raise ValueError(
                "Independent-row mode currently requires fixed corner frames"
            )
        rule_index = next(iter(set(args.edge_rule_index)))
        bits = tuple(itertools.product((False, True), repeat=6))[rule_index]
        rule = dict(zip(bit_names, bits, strict=True))
        row_candidates = {key: [] for key in (
            "env2", "env1_ab_env3_ba", "env3_ab_env1_ba"
        )}
        for left_site_perm in itertools.permutations((0, 1, 2)):
            for right_site_perm in itertools.permutations((0, 1, 2)):
                raw_rows = _raw_rows(
                    ab, ba, ab.double_layers, rule, max_bytes=max_bytes,
                    left_site_perm=left_site_perm,
                    right_site_perm=right_site_perm,
                )
                for offset, key in enumerate(row_candidates):
                    raw, upper_corner, lower_corner = raw_rows.pop(key)
                    spectrum = batch._spectrum(
                        raw, upper_corner, lower_corner,
                        seed=(seed + 1000 * rule_index + 100 * offset) % 2**32,
                        eig_tol=args.eig_tol,
                        arpack_ncv=args.arpack_ncv,
                        arpack_maxiter=args.arpack_maxiter,
                        dense_threshold=0,
                        corner_relative_cutoff=args.corner_relative_cutoff,
                    )
                    candidate = {
                        "left_site_perm": list(left_site_perm),
                        "right_site_perm": list(right_site_perm),
                        "inverse_xi": float(spectrum["inverse_correlation_length"]),
                        "eigenvalues": spectrum["eigenvalues"],
                    }
                    row_candidates[key].append(candidate)
                    print(
                        f"{key} sites={left_site_perm}/{right_site_perm} "
                        f"inverse_xi={candidate['inverse_xi']:.12g}", flush=True,
                    )
        for candidates in row_candidates.values():
            candidates.sort(key=lambda item: item["inverse_xi"])
        nonzero = {
            key: [item for item in candidates if item["inverse_xi"] > 1.0e-8]
            for key, candidates in row_candidates.items()
        }
        closest = []
        for first in nonzero["env2"]:
            for second in nonzero["env1_ab_env3_ba"]:
                for third in nonzero["env3_ab_env1_ba"]:
                    values = (first["inverse_xi"], second["inverse_xi"], third["inverse_xi"])
                    spread = max(values) - min(values)
                    closest.append({
                        "spread": spread,
                        "relative_spread": spread / (sum(values) / 3.0),
                        "env2": first,
                        "env1_ab_env3_ba": second,
                        "env3_ab_env1_ba": third,
                    })
        closest.sort(key=lambda item: item["spread"])
        document = {
            "checkpoint": str(args.checkpoint.resolve()),
            "J2": args.J2, "D": D_bond, "chi": chi, "seed": seed,
            "edge_rule_index": rule_index,
            "edge_transposed": [name for name in bit_names if rule[name]],
            "row_candidates": row_candidates,
            "closest_nonzero_triples": closest[:100],
            "ctm": {
                "steps_ab": ab.ctm_steps, "steps_ba": ba.ctm_steps,
                "energy_ab": ab.final_energy_proxy,
                "energy_ba": ba.final_energy_proxy,
            },
            "elapsed_seconds": time.perf_counter() - started,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        print(f"CLOSEST {closest[0]}", flush=True)
        print(f"WROTE {args.output}", flush=True)
        return 0
    selected_edge_rules = (
        set(args.edge_rule_index) if args.edge_rule_index is not None else None
    )
    if selected_edge_rules is not None and any(
        index < 0 or index >= 64 for index in selected_edge_rules
    ):
        raise ValueError("--edge-rule-index must lie in [0, 63]")
    corner_bit_sets = (
        tuple(itertools.product((False, True), repeat=3))
        if args.enumerate_corner_frames else ((False, False, False),)
    )
    local_permutations = (
        tuple(itertools.permutations((0, 1, 2)))
        if args.enumerate_local_frames else ((0, 1, 2),)
    )
    lower_bit_sets = (
        tuple(itertools.product((False, True), repeat=3))
        if args.enumerate_lower_order else ((False, False, False),)
    )
    for rule_index, bits in enumerate(itertools.product((False, True), repeat=6)):
        if selected_edge_rules is not None and rule_index not in selected_edge_rules:
            continue
        rule = dict(zip(bit_names, bits, strict=True))
        for corner_bits in corner_bit_sets:
         for lower_bits in lower_bit_sets:
          for left_site_perm in local_permutations:
           for right_site_perm in local_permutations:
            corner_rule = dict(zip(PHASES, corner_bits, strict=True))
            lower_swaps = dict(zip(PHASES, lower_bits, strict=True))
            spectra = {}
            raw_rows = _raw_rows(
                ab, ba, ab.double_layers, rule, max_bytes=max_bytes,
                left_site_perm=left_site_perm,
                right_site_perm=right_site_perm,
                physical_grow_sites=args.physical_grow_sites,
                lower_swaps=lower_swaps,
            )
            phase_pairs = {
                "env2": ("env2", "env2"),
                "env1_ab_env3_ba": ("env1", "env3"),
                "env3_ab_env1_ba": ("env3", "env1"),
            }
            failed = None
            try:
                for offset, key in enumerate(("env2", "env1_ab_env3_ba", "env3_ab_env1_ba")):
                    raw, upper_corner, lower_corner = raw_rows.pop(key)
                    upper_phase, lower_phase = phase_pairs[key]
                    if corner_rule[upper_phase]:
                        upper_corner = upper_corner.T.contiguous()
                    if corner_rule[lower_phase]:
                        lower_corner = lower_corner.T.contiguous()
                    spectrum = batch._spectrum(
                        raw,
                        upper_corner,
                        lower_corner,
                        seed=(seed + 1000 * rule_index + 10 * sum(1 << i for i, bit in enumerate(corner_bits) if bit) + offset) % 2**32,
                        eig_tol=args.eig_tol,
                        arpack_ncv=args.arpack_ncv,
                        arpack_maxiter=args.arpack_maxiter,
                        dense_threshold=0,
                        corner_relative_cutoff=args.corner_relative_cutoff,
                    )
                    spectra[key] = float(spectrum["inverse_correlation_length"])
            except Exception as exc:
                failed = f"{type(exc).__name__}: {exc}"
            if failed is None:
                values = list(spectra.values())
                spread = max(values) - min(values)
                relative_spread = spread / max(abs(sum(values) / len(values)), 1.0e-300)
            else:
                spread = math.inf
                relative_spread = math.inf
            row = {
                "rule_index": rule_index,
                "transposed": [name for name in bit_names if rule[name]],
                "corner_transposed_phases": [
                    phase for phase in PHASES if corner_rule[phase]
                ],
                "left_site_perm": list(left_site_perm),
                "right_site_perm": list(right_site_perm),
                "lower_swapped_phases": [
                    phase for phase in PHASES if lower_swaps[phase]
                ],
                "inverse_xi": spectra,
                "spread": spread,
                "relative_spread": relative_spread,
                "error": failed,
            }
            rows.append(row)
            print(
                f"edge={rule_index:02d} corners={row['corner_transposed_phases']} "
                f"sites={left_site_perm}/{right_site_perm} "
                f"lower_swaps={row['lower_swapped_phases']} "
                f"spread={spread:.12g} rel={relative_spread:.6g} values={spectra}",
                flush=True,
            )

    rows.sort(key=lambda item: item["spread"])
    document = {
        "checkpoint": str(args.checkpoint.resolve()),
        "J2": args.J2,
        "D": D_bond,
        "chi": chi,
        "seed": seed,
        "ctm": {
            "steps_ab": ab.ctm_steps,
            "steps_ba": ba.ctm_steps,
            "energy_ab": ab.final_energy_proxy,
            "energy_ba": ba.final_energy_proxy,
        },
        "corner_orientation": "upper_transpose_lower_untransposed",
        "rules_sorted_by_absolute_spread": rows,
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"BEST {rows[0]}", flush=True)
    print(f"WROTE {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
