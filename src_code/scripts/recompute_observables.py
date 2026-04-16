#!/usr/bin/env python3
"""recompute_observables.py
===========================
Post-process one or more .pt iPEPS checkpoint files and write the full
observables .txt output (energy, correlations, magnetizations, CTMRG
truncation error) using the current version of core.py and the evaluate_obs
pipeline.

This is useful for re-evaluating checkpoints produced before today's
additions (truncation error, c6ypi renaming) or before any future observable
additions, without re-running the full optimisation.

Usage examples
--------------
  # Single file, auto-detect ansatz, use chi stored in checkpoint
  python recompute_observables.py /path/to/sweep_D5_chi35_best.pt

  # Directory: process all *_best.pt files
  python recompute_observables.py /path/to/run_dir --glob '*_best.pt'

  # Override chi, J2, output directory
  python recompute_observables.py data/run/*.pt \\
      --chi 45 --J2 0.265 --output-dir results/reeval/

  # Specify ansatz explicitly (e.g. if the old file used 'plaq' name)
  python recompute_observables.py old.pt --ansatz c6ypi

  # Use GPU, double precision
  python recompute_observables.py run/*.pt --gpu --double

Output
------
For each input file ``<stem>.pt``, writes ``<stem>_observables.txt`` to the
output directory (default: same directory as the input .pt file).
The file format is identical to the one produced by the sweep during training.

Ansatz auto-detection
---------------------
  a_raw          in checkpoint keys  →  c6ypi  (single-tensor C6-Ypi)
  a (but not a_raw) in keys          →  6tensors (unrestricted / sym6)
  Presence of 'ansatz' key           →  use that value directly

Override with --ansatz {neel,c6ypi,6tensors,sym6,plaq}.
"""

import os
import sys
import argparse
import datetime
import glob as glob_module

# ── Path setup ────────────────────────────────────────────────────────────────
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR     = os.path.join(_SCRIPTS_DIR, '..', 'src')
sys.path.insert(0, _SRC_DIR)
sys.path.insert(0, _SCRIPTS_DIR)   # allows 'import main'

# ── Heavy imports (after path setup) ─────────────────────────────────────────
import torch
import core as _core

# Import the exact functions from main.py  (safe: main.py has
# `if __name__ == '__main__': main()` guard, so no sweep is started).
from main import (           # noqa: E402
    evaluate_observables,
    _save_observables_file,
    _print_observables_summary,
    _ANSATZ_CONFIGS,
    _ENV_BOND_LABELS,
    _SITE_LABELS,
    timestamp,
    CTM_MAX_STEPS,
    CTM_CONV_THR,
    ENV_IDENTITY_INIT,
    RSVD_MODE,
    RSVD_NEUMANN_TERMS,
    RSVD_POWER_ITERS,
    SVD_CPU_OFFLOAD_THRESHOLD,
)
from core import build_heisenberg_H, set_dtype, set_device, set_rsvd_mode

# ── Ansatz key detection ───────────────────────────────────────────────────────

def _detect_ansatz(ckpt: dict) -> str:
    """Guess the ansatz name from checkpoint keys.

    Returns one of the _ANSATZ_CONFIGS keys, or raises ValueError.
    """
    if 'ansatz' in ckpt:
        return ckpt['ansatz']
    keys = set(ckpt.keys())
    if 'a_raw' in keys:
        # Could be neel or c6ypi — both use a_raw.  Disambiguate by yaml_name
        # stored in checkpoint if available, otherwise default to c6ypi.
        yn = ckpt.get('yaml_name', '')
        if 'neel' in yn.lower():
            return 'neel'
        return 'c6ypi'   # plaq/c6ypi is more common; fallback
    if all(k in keys for k in ('a', 'b', 'c', 'd', 'e', 'f')):
        # 6-tensor: could be 6tensors or sym6 — both have same keys.
        yn = ckpt.get('yaml_name', '')
        if 'sym6' in yn.lower():
            return 'sym6'
        return '6tensors'
    raise ValueError(
        f"Cannot detect ansatz from checkpoint keys {sorted(keys)}.  "
        "Use --ansatz to specify explicitly.")


def _load_params(ckpt: dict, ansatz_cfg: dict,
                 device: torch.device) -> list:
    """Extract the parameter tensors from a checkpoint dict.

    Returns a list of tensors matching ansatz_cfg['ckpt_keys'] order,
    moved to device.
    """
    params = []
    for key in ansatz_cfg['ckpt_keys']:
        if key not in ckpt:
            raise KeyError(
                f"Checkpoint missing expected key '{key}'.  "
                f"Available keys: {sorted(ckpt.keys())}")
        t = ckpt[key]
        if not isinstance(t, torch.Tensor):
            raise TypeError(
                f"Checkpoint key '{key}' is {type(t)}, expected torch.Tensor.")
        params.append(t.to(device))
    return params


# ── Main logic ────────────────────────────────────────────────────────────────

def process_one(pt_path: str, args: argparse.Namespace) -> bool:
    """Process a single .pt checkpoint file.  Returns True on success."""
    print(f"\n{'─'*70}")
    print(f"  Input  : {pt_path}")

    # ── Load checkpoint ───────────────────────────────────────────────────
    ckpt = torch.load(pt_path, map_location='cpu', weights_only=False)

    D_bond = int(ckpt.get('D_bond', args.D_bond))
    chi    = args.chi if args.chi is not None else int(ckpt.get('chi', -1))
    if chi <= 0:
        print(f"  [ERROR] chi not found in checkpoint and not specified via --chi.  "
              f"Checkpoint keys: {sorted(ckpt.keys())}")
        return False

    # ── Ansatz ────────────────────────────────────────────────────────────
    ansatz_name = args.ansatz or _detect_ansatz(ckpt)
    # Keep backward compat: old checkpoints may use 'plaq'
    if ansatz_name == 'plaq':
        ansatz_name = 'c6ypi'
    if ansatz_name not in _ANSATZ_CONFIGS:
        print(f"  [ERROR] Unknown ansatz '{ansatz_name}'.  "
              f"Valid: {list(_ANSATZ_CONFIGS.keys())}")
        return False
    ansatz_cfg = _ANSATZ_CONFIGS[ansatz_name]
    print(f"  Ansatz : {ansatz_name}  ({ansatz_cfg['description']})")
    print(f"  D={D_bond}  chi={chi}  J1={args.J1}  J2={args.J2}  d_phys={args.d_phys}")

    # ── Dtype / device ────────────────────────────────────────────────────
    use_double = not args.float32
    use_real   = not args.complex
    set_dtype(use_double, use_real)
    device = torch.device('cuda' if (args.gpu and torch.cuda.is_available()) else 'cpu')
    set_device(device)
    set_rsvd_mode(args.rsvd_mode,
                  neumann_terms=RSVD_NEUMANN_TERMS,
                  power_iters=RSVD_POWER_ITERS)
    _core._SVD_CPU_OFFLOAD_THRESHOLD = SVD_CPU_OFFLOAD_THRESHOLD
    print(f"  Device : {device}   dtype : {'float64' if use_double else 'float32'}"
          f"{'(complex)' if not use_real else ''}")

    # ── Build operators ───────────────────────────────────────────────────
    d_PHYS = args.d_phys
    SdotS  = build_heisenberg_H(1.0, d_PHYS)
    J1, J2 = args.J1, args.J2
    Js = ([J1]*6 + [J2]*6) * 3   # 36 J-values: env1/env2/env3 × (6 nn + 6 nnn)

    # ── Load tensors ──────────────────────────────────────────────────────
    try:
        params = _load_params(ckpt, ansatz_cfg, device)
    except (KeyError, TypeError) as exc:
        print(f"  [ERROR] {exc}")
        return False

    # Cast tensors to current dtype
    params = [p.to(_core.TENSORDTYPE) for p in params]

    # ── Evaluate observables ──────────────────────────────────────────────
    print(f"  Running evaluate_observables (CTM_MAX_STEPS={CTM_MAX_STEPS}, "
          f"CTM_CONV_THR={CTM_CONV_THR:.1e}) ...")
    with torch.no_grad():
        energy, correlations, magnetizations, trunc_error = evaluate_observables(
            params, Js, SdotS, chi, D_bond, d_PHYS, ansatz_cfg)

    if energy != energy:   # NaN check
        print(f"  [WARN] energy = NaN — CTMRG did not converge.  "
              f"Try increasing --chi or --ctm-max-steps.")

    # ── Output path ───────────────────────────────────────────────────────
    stem      = os.path.splitext(os.path.basename(pt_path))[0]
    out_dir   = args.output_dir or os.path.dirname(os.path.abspath(pt_path))
    os.makedirs(out_dir, exist_ok=True)
    out_path  = os.path.join(out_dir, f"{stem}_observables.txt")

    _save_observables_file(out_path, D_bond, chi, energy,
                           correlations, magnetizations, trunc_error)
    _print_observables_summary('REC', D_bond, chi, energy,
                               correlations, magnetizations, trunc_error)
    print(f"  Output : {out_path}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # ── Positional / input ────────────────────────────────────────────────
    parser.add_argument(
        'inputs', nargs='+',
        help='One or more .pt checkpoint files or directories.')
    parser.add_argument(
        '--glob', default='*_best.pt',
        help='Glob pattern used when a directory is given as input '
             '(default: *_best.pt).')

    # ── Ansatz / physics ──────────────────────────────────────────────────
    parser.add_argument(
        '--ansatz', default=None,
        choices=list(_ANSATZ_CONFIGS.keys()) + ['plaq'],
        help='Ansatz name (auto-detected from checkpoint keys if omitted).')
    parser.add_argument('--J1', type=float, default=1.0,
                        help='Nearest-neighbour coupling (default 1.0).')
    parser.add_argument('--J2', type=float, default=0.0,
                        help='Next-nearest-neighbour coupling (default 0.0).')
    parser.add_argument('--d-phys', type=int, default=2, dest='d_phys',
                        help='Physical Hilbert space dimension (default 2 for spin-1/2).')
    parser.add_argument('--D-bond', type=int, default=None, dest='D_bond',
                        help='Bond dimension override (normally read from checkpoint).')
    parser.add_argument('--chi', type=int, default=None,
                        help='Environment bond dimension override '
                             '(default: use chi stored in checkpoint).')

    # ── Precision / device ────────────────────────────────────────────────
    parser.add_argument('--float32', action='store_true',
                        help='Use float32 (default: float64).')
    parser.add_argument('--complex', action='store_true',
                        help='Use complex tensors (default: real).')
    parser.add_argument('--gpu', action='store_true',
                        help='Use CUDA GPU if available (default: CPU).')
    parser.add_argument('--rsvd-mode', default=RSVD_MODE, dest='rsvd_mode',
                        choices=['full_svd', 'neumann', 'augmented', 'none'],
                        help=f'rSVD backward mode (default: {RSVD_MODE}).')

    # ── Output ────────────────────────────────────────────────────────────
    parser.add_argument(
        '--output-dir', default=None, dest='output_dir',
        help='Directory for output .txt files (default: same dir as each .pt file).')

    args = parser.parse_args()

    # ── Expand inputs to list of .pt files ───────────────────────────────
    pt_files = []
    for inp in args.inputs:
        if os.path.isdir(inp):
            matched = sorted(glob_module.glob(os.path.join(inp, args.glob)))
            if not matched:
                print(f"  [WARN] No files matching '{args.glob}' in {inp}")
            pt_files.extend(matched)
        elif '*' in inp or '?' in inp:
            matched = sorted(glob_module.glob(inp))
            pt_files.extend(matched)
        else:
            pt_files.append(inp)

    if not pt_files:
        print("[ERROR] No .pt files found.  Check your paths or --glob pattern.")
        sys.exit(1)

    print(f"recompute_observables.py — {len(pt_files)} file(s) to process")
    print(f"  J1={args.J1}  J2={args.J2}  d_phys={args.d_phys}  "
          f"ansatz={'auto' if not args.ansatz else args.ansatz}")

    n_ok = 0
    for pt_path in pt_files:
        ok = process_one(pt_path, args)
        n_ok += ok

    print(f"\n{'='*70}")
    print(f"Done: {n_ok}/{len(pt_files)} files processed successfully.")


if __name__ == '__main__':
    main()
