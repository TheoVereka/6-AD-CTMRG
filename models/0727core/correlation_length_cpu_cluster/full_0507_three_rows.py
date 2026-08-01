#!/usr/bin/env python3
"""Diagnostic three-row pencils using the stable 0507 full general CTM."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

import compute_three_generalized_correlation_lengths as batch
import correlation_length as corr


HERE = Path(__file__).resolve().parent
CORE_PATH = HERE.parents[1] / "0507core" / "core.py"


def _load_full_core():
    spec = importlib.util.spec_from_file_location("stable_0507_core", CORE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {CORE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _run(core, raw_a, raw_b, *, chi, max_steps, tolerance):
    _, layers = corr._build_ctm_layers(raw_a, raw_b)
    result = core.CTMRG_from_init_to_stop(
        *layers, chi, int(raw_a.shape[0]) ** 2,
        max_steps, tolerance, True, energy_proxy_fn=None,
    )
    return result, layers


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--J2", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--seed", type=int, default=173921)
    parser.add_argument("--ctm-max-steps", type=int, default=100)
    parser.add_argument("--ctm-conv-tol", type=float, default=1.0e-8)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    a, b, metadata = corr._load_twoc3_checkpoint(
        str(args.checkpoint.resolve()), device=torch.device("cpu"),
        dtype=torch.float64,
    )
    chi = int(metadata["chi"])
    core = _load_full_core()
    core.set_dtype(True, use_real=True)
    core.set_device(torch.device("cpu"))
    core._SVD_CPU_OFFLOAD_THRESHOLD = 0
    core._USE_FULL_SVD = False
    core.set_rsvd_mode(
        corr.DEFAULT_RSVD_MODE,
        neumann_terms=corr.DEFAULT_RSVD_NEUMANN_TERMS,
        power_iters=corr.DEFAULT_RSVD_POWER_ITERS,
    )
    core.set_ctm_conv_mode("SVdifference")

    print(f"0507 full CTM (a,b), D={a.shape[0]}, chi={chi}", flush=True)
    ab, layers = _run(
        core, a, b, chi=chi, max_steps=args.ctm_max_steps,
        tolerance=args.ctm_conv_tol,
    )
    print(f"0507 full CTM (b,a), steps_ab={ab[-1]}", flush=True)
    ba, _ = _run(
        core, b, a, chi=chi, max_steps=args.ctm_max_steps,
        tolerance=args.ctm_conv_tol,
    )
    print(f"0507 full CTMs complete, steps_ba={ba[-1]}", flush=True)

    A, B, C, D, E, F = layers
    specs = {
        "env2": (
            ab[13], ab[12], D.permute(0, 2, 1).contiguous(),
            C.permute(1, 2, 0).contiguous(), ba[12], ba[13],
            ab[9], ba[9],
        ),
        "env1_ab_env3_ba": (
            ab[4], ab[3], B.permute(1, 0, 2).contiguous(), E,
            ba[21], ba[22], ab[0], ba[18],
        ),
        "env3_ab_env1_ba": (
            ab[22], ab[21], F.permute(2, 1, 0).contiguous(),
            A.permute(2, 0, 1).contiguous(), ba[3], ba[4],
            ab[18], ba[0],
        ),
    }
    spectra = {}
    ordinary = {}
    for offset, key in enumerate(specs):
        *network, upper_corner, lower_corner = specs[key]
        raw = corr.SideBySideRowToRowTransferOperator(
            *network, max_intermediate_bytes=1024**3, progress_every=0,
        )
        spectra[key] = batch._spectrum(
            raw, upper_corner, lower_corner,
            seed=(args.seed + offset) % 2**32, eig_tol=0.0,
            arpack_ncv=64, arpack_maxiter=4000, dense_threshold=0,
            corner_relative_cutoff=1.0e-14,
        )
        ordinary_result = corr._diagonalize_operator(
            raw, tol=0.0, ncv=64, maxiter=4000,
            dense_dimension_threshold=0,
            seed=(args.seed + 10 + offset) % 2**32,
        )
        magnitudes = sorted((abs(v) for v in ordinary_result.eigenvalues), reverse=True)
        ordinary[key] = {
            "inverse_xi": float(np.log(magnitudes[0] / magnitudes[1])),
            "eigenvalues": [
                {"real": float(v.real), "imag": float(v.imag), "abs": float(abs(v))}
                for v in ordinary_result.eigenvalues
            ],
        }
        print(
            f"{key}: generalized={spectra[key]['inverse_correlation_length']:.12g} "
            f"ordinary={ordinary[key]['inverse_xi']:.12g}", flush=True,
        )

    payload = {
        "core": str(CORE_PATH), "checkpoint": str(args.checkpoint.resolve()),
        "J2": args.J2, "D": int(a.shape[0]), "chi": chi,
        "ctm_steps_ab": int(ab[-1]), "ctm_steps_ba": int(ba[-1]),
        "generalized": spectra, "ordinary": ordinary,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"WROTE {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
