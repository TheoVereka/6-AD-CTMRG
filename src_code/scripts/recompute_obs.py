#!/usr/bin/env python3
"""
recompute_obs.py
================
Re-run CTMRG to convergence on a saved checkpoint and produce exactly the
same observables file and stdout printout as main.py's OBS + lookahead (LA)
blocks.

Usage (minimal):
    python recompute_obs.py --J2 0.26 --file path/to/D_3_chi_54_best.pt

Usage (full):
    python recompute_obs.py --J2 0.26 --file sweep_D3_chi54_best.pt \
        --J1 1.0 --ansatz sym6 --out-dir /path/to/output

The script auto-detects D_bond and chi from the checkpoint, derives the
ansatz from the filename if --ansatz is omitted, and writes:
    <out-dir>/D_{D}_chi_{chi}_energy_magnetization_correlation.txt
    <out-dir>/D_{D}_chi_{chi}_lookahead_{chi_la}_energy_magnetization_correlation.txt
matching main.py exactly.
"""

# ── threading env (must be before numpy/torch) ────────────────────────────────
import os
_N_CORES = 4
os.environ.setdefault("OMP_NUM_THREADS", str(_N_CORES))
os.environ.setdefault("MKL_NUM_THREADS", str(_N_CORES))
os.environ.setdefault("MKL_DYNAMIC", "FALSE")
os.environ.setdefault("KMP_AFFINITY", "granularity=fine,compact,1,0")
os.environ.setdefault("KMP_BLOCKTIME", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys, gc, time, argparse, datetime
import torch

# ── locate core.py next to this file ─────────────────────────────────────────
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

import core as _core
from core import (
    normalize_tensor,
    normalize_single_layer_tensor_for_double_layer,
    abcdef_to_ABCDEF,
    CTMRG_from_init_to_stop,
    build_heisenberg_H,
    energy_expectation_nearest_neighbor_3ebadcf_bonds,
    energy_expectation_nearest_neighbor_3afcbed_bonds,
    energy_expectation_nearest_neighbor_other_3_bonds,
    set_dtype,
    set_device,
    symmetrize_virtual_legs,
    neel_param_to_a,
    neel_a_to_free_param,
    initialize_neel_free_param,
    pad_neel_free_param,
    c6ypi_abcdef_from_a,
    initialize_c6ypi,
    c3vypi_abcdef_from_a,
    initialize_c3vypi,
    twoc3_abcdef_from_ab,
    initialize_twoc3,
    symmetrize_six_local_reflections,
    sym6_params_to_abcdef,
    sym6_abcdef_to_free_params,
    initialize_sym6_free_params,
    pad_sym6_free_params,
    neel_abcdef_from_a,
    initialize_neel,
)
try:
    import opt_einsum as oe
except ImportError:
    import torch as oe   # fallback (not ideal but won't be needed for obs)

# ── constants matching main.py defaults ───────────────────────────────────────
CTM_MAX_STEPS   = 130
CTM_CONV_THR    = 1e-7
ENV_IDENTITY_INIT = True
N_SITES         = 6
D_PHYS_DEFAULT  = 2

# ── ansatz registry (copied verbatim from main.py) ───────────────────────────
ANSATZ_REGISTRY: dict = {
    'unrestricted': {
        'n_params':    6,
        'free_morphism_fn': None,
        'symmetrize_fn': None,
        'derive_fn':   None,
        'init_fn':     None,
        'pad_fn':      None,
        'ckpt_keys':   ['a', 'b', 'c', 'd', 'e', 'f'],
        'yaml_name':   'unrestricted',
    },
    'neel': {
        'n_params':    1,
        'free_morphism_fn': neel_param_to_a,
        'symmetrize_fn': symmetrize_virtual_legs,
        'derive_fn':   neel_abcdef_from_a,
        'init_fn':     initialize_neel_free_param,
        'pad_fn':      pad_neel_free_param,
        'ckpt_keys':   ['h'],
        'yaml_name':   'neel_free_param',
    },
    'neel_legacy': {
        'n_params':    1,
        'free_morphism_fn': None,
        'symmetrize_fn': symmetrize_virtual_legs,
        'derive_fn':   neel_abcdef_from_a,
        'init_fn':     initialize_neel,
        'pad_fn':      None,
        'ckpt_keys':   ['a'],
        'yaml_name':   'neel_legacy',
    },
    'c6ypi': {
        'n_params':    1,
        'symmetrize_fn': None,
        'free_morphism_fn': None,
        'derive_fn':   c6ypi_abcdef_from_a,
        'init_fn':     initialize_c6ypi,
        'ckpt_keys':   ['a_raw'],
        'yaml_name':   '1tensor_C6Ypi',
    },
    'c3vypi': {
        'n_params':    1,
        'symmetrize_fn': None,
        'free_morphism_fn': None,
        'derive_fn':   c3vypi_abcdef_from_a,
        'init_fn':     initialize_c3vypi,
        'ckpt_keys':   ['a_raw'],
        'yaml_name':   '1tensor_C3Vypi',
    },
    'twoc3': {
        'n_params':    2,
        'symmetrize_fn': None,
        'free_morphism_fn': None,
        'derive_fn':   twoc3_abcdef_from_ab,
        'init_fn':     initialize_twoc3,
        'ckpt_keys':   ['a_raw', 'b_raw'],
        'yaml_name':   '2tensor_twoC3',
    },
    'sym6': {
        'n_params':        6,
        'free_morphism_fn': sym6_params_to_abcdef,
        'symmetrize_fn':   None,
        'derive_fn':       None,
        'init_fn':         initialize_sym6_free_params,
        'pad_fn':          pad_sym6_free_params,
        'ckpt_keys':       ['h_a', 'h_b', 'h_c', 'h_d', 'h_e', 'h_f'],
        'yaml_name':       'sym6_free_param',
    },
    'sym6_legacy': {
        'n_params':    6,
        'free_morphism_fn': None,
        'symmetrize_fn': symmetrize_six_local_reflections,
        'derive_fn':   None,
        'init_fn':     None,
        'pad_fn':      None,
        'ckpt_keys':   ['a', 'b', 'c', 'd', 'e', 'f'],
        'yaml_name':   'sym6',
    },
}


# ── bond / site labels (must match main.py exactly) ───────────────────────────
_ENV_BOND_LABELS = [
    'EB', 'AD', 'CF', 'FA', 'DE', 'BC', 'AE', 'EC', 'CA', 'DB', 'BF', 'FD',
    'CB', 'AF', 'ED', 'DC', 'BA', 'FE', 'CA', 'AE', 'EC', 'BF', 'FD', 'DB',
    'EF', 'AB', 'CD', 'BE', 'FC', 'DA', 'EC', 'CA', 'AE', 'FD', 'DB', 'BF',
]
_SITE_LABELS = ['A', 'B', 'C', 'D', 'E', 'F']


def timestamp() -> str:
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


# ── derive (a,b,c,d,e,f) from params (copy of main.py _derive_abcdef) ────────
def _derive_abcdef(params_list, cfg, D_bond=None):
    n = cfg['n_params']
    free_morph = cfg.get('free_morphism_fn')
    derive_fn  = cfg.get('derive_fn')
    sym_fn     = cfg.get('symmetrize_fn')

    if free_morph is not None and n == 1:
        a_sym = free_morph(params_list[0])
        return derive_fn(a_sym) if derive_fn else (a_sym,) * 6

    if free_morph is not None and n == 6:
        return free_morph(*params_list)

    if n == 1:
        a = params_list[0]
        if sym_fn is not None:
            a = sym_fn(a)
        return derive_fn(a) if derive_fn else (a,) * 6

    if n == 2:
        a_raw, b_raw = params_list
        if derive_fn is not None:
            return derive_fn(a_raw, b_raw)
        return (a_raw,) * 3 + (b_raw,) * 3

    # n == 6
    if sym_fn is not None:
        params_list = list(sym_fn(*params_list))
    return tuple(params_list)


# ── spin operators for observables ────────────────────────────────────────────
def _make_spin_ops(d_PHYS, dtype, device):
    spin = (d_PHYS - 1) / 2.0
    Splus  = torch.zeros(d_PHYS, d_PHYS, dtype=dtype, device=device)
    Sminus = torch.zeros(d_PHYS, d_PHYS, dtype=dtype, device=device)
    Sz_op  = torch.zeros(d_PHYS, d_PHYS, dtype=dtype, device=device)
    for _i in range(d_PHYS):
        _m = spin - _i
        Sz_op[_i, _i] = _m
        if _i < d_PHYS - 1:
            _cs = (spin * (spin + 1) - _m * (_m - 1)) ** 0.5
            Splus[_i, _i + 1]  = _cs
            Sminus[_i + 1, _i] = _cs
    Sx_op  = (Splus + Sminus) / 2.0
    iSy_op = (Splus - Sminus) / 2.0
    return Sx_op, iSy_op, Sz_op


# ── observables from rho cache (identical to main.py _observables_from_rhos) ─
def _observables_from_rhos(rho_data, Js, SdotS, d_PHYS):
    if rho_data is None or any(x is None for x in rho_data):
        nan36 = [float('nan')] * 36
        nan54 = [float('nan')] * 54
        return float('nan'), nan36, nan54

    Sx_op, iSy_op, Sz_op = _make_spin_ops(d_PHYS, SdotS.dtype, SdotS.device)

    def _corr(rho_4d):
        return torch.real(
            torch.einsum("ikjl,ijkl->", rho_4d, SdotS)).item()

    def _mag(rho_4d, site_idx):
        if site_idx == 0:
            rho_1 = torch.einsum("ikjk->ij", rho_4d)
        else:
            rho_1 = torch.einsum("ikil->kl", rho_4d)
        tr = torch.real(torch.trace(rho_1)).clamp(min=1e-30)
        rho_1 = rho_1 / tr
        mx = torch.real(torch.einsum("ij,ji->", rho_1, Sx_op)).item()
        if rho_1.is_complex():
            my = torch.imag(torch.einsum("ij,ji->", rho_1,
                                         iSy_op.to(rho_1.dtype))).item()
        else:
            my = torch.real(torch.einsum("ij,ji->", rho_1, iSy_op)).item()
        mz = torch.real(torch.einsum("ij,ji->", rho_1, Sz_op)).item()
        return (mx, my, mz)

    correlations   = []
    magnetizations = []
    for (rhos, mag_info) in rho_data:
        correlations  += [_corr(r) for r in rhos]
        for s in _SITE_LABELS:
            rho, idx = mag_info[s]
            magnetizations.extend(_mag(rho, idx))

    energy = 0.0
    for _blk in range(3):
        _b = _blk * 12
        energy += 0.5 * sum(Js[_b + _i] * correlations[_b + _i] for _i in range(6))
        energy +=       sum(Js[_b + 6 + _i] * correlations[_b + 6 + _i] for _i in range(6))

    return energy, correlations, magnetizations


# ── save file (identical format to main.py) ───────────────────────────────────
def _save_observables_file(filepath, D_bond, chi,
                           energy, correlations, magnetizations,
                           trunc_error=None):
    with open(filepath, 'w') as fp:
        fp.write(f"# D={D_bond}  chi={chi}  timestamp={timestamp()}\n\n")
        fp.write(f"energy           = {energy:+.12e}\n")
        fp.write(f"energy_per_site  = {energy / N_SITES:+.12e}\n")
        if trunc_error is not None:
            fp.write(f"trunc_error      = {trunc_error:.6e}\n")
        fp.write("\n")

        fp.write("# ── Bond correlations <Si·Sj>  (36 values) "
                 "──────────────────────\n")
        for env_idx in range(3):
            fp.write(f"# env{env_idx+1}\n")
            for j in range(12):
                idx   = env_idx * 12 + j
                label = _ENV_BOND_LABELS[idx]
                fp.write(f"corr_env{env_idx+1}_{label:>2s} = {correlations[idx]:+.12e}\n")
            fp.write("\n")

        fp.write("# ── Site magnetizations <Sx> <Sy> <Sz>  (54 values) "
                 "──────────────\n")
        fp.write("# <Sy> via iSy_op=(S+-S-)/2=i·Sy_phys:\n")
        fp.write("#   Real iPEPS:    Sy = Re(Tr(rho·iSy_op)) = 0.0 exactly\n")
        fp.write("#   Complex iPEPS: Sy = Im(Tr(rho·iSy_op)) = physical <Sy>\n")
        for env_idx in range(3):
            fp.write(f"# env{env_idx+1}\n")
            for s_idx, s in enumerate(_SITE_LABELS):
                base = env_idx * 18 + s_idx * 3
                fp.write(f"mag_env{env_idx+1}_{s}  "
                         f"Sx={magnetizations[base]:+.12e}  "
                         f"Sy={magnetizations[base+1]:+.12e}  "
                         f"Sz={magnetizations[base+2]:+.12e}\n")
            fp.write("\n")

        fp.write("# ── Local magnetization |m|=sqrt(Sx²+Sy²+Sz²)  "
                 "(18 values: 6 sites × 3 envs) ──────────────\n")
        for env_idx in range(3):
            fp.write(f"# env{env_idx+1}\n")
            for s_idx, s in enumerate(_SITE_LABELS):
                base = env_idx * 18 + s_idx * 3
                mx_ = magnetizations[base]
                my_ = magnetizations[base + 1]
                mz_ = magnetizations[base + 2]
                loc_mag = (mx_**2 + my_**2 + mz_**2) ** 0.5
                fp.write(f"localmag_env{env_idx+1}_{s}  |m|={loc_mag:+.12e}\n")
            fp.write("\n")
    print(f"  │  Observables saved → {filepath}")


# ── print summary (identical format to main.py) ───────────────────────────────
def _print_observables_summary(tag, D_bond, chi,
                                energy, correlations, magnetizations,
                                trunc_error=None):
    nn_corrs  = [correlations[i] for i in range(36) if (i % 12) < 6]
    nnn_corrs = [correlations[i] for i in range(36) if (i % 12) >= 6]
    all_mx = [magnetizations[i*3]     for i in range(18)]
    all_my = [magnetizations[i*3 + 1] for i in range(18)]
    all_mz = [magnetizations[i*3 + 2] for i in range(18)]

    def _stat(vals):
        mn, mx = min(vals), max(vals)
        n   = len(vals)
        avg = sum(vals) / n
        var = sum((v - avg)**2 for v in vals) / n
        se  = (var / n) ** 0.5
        return f"min={mn:+.6f} max={mx:+.6f} mean={avg:+.6f} se={se:.2e}"

    te_str = f"  trunc_err={trunc_error:.3e}" if trunc_error is not None else ""
    print(f"  │  [{tag}] D={D_bond} chi={chi}  E={energy:+.10f}  "
          f"E/site={energy/N_SITES:+.10f}{te_str}")
    print(f"  │    nn  <S·S> ({len(nn_corrs):>2d}): {_stat(nn_corrs)}")
    print(f"  │    nnn <S·S> ({len(nnn_corrs):>2d}): {_stat(nnn_corrs)}")
    print(f"  │    <Sx>      ({len(all_mx):>2d}): {_stat(all_mx)}")
    _sy_note = "  (=0 for real iPEPS)" if all(abs(v) < 1e-12 for v in all_my) else ""
    print(f"  │    <Sy>      ({len(all_my):>2d}): {_stat(all_my)}{_sy_note}")
    print(f"  │    <Sz>      ({len(all_mz):>2d}): {_stat(all_mz)}")

    def _site_localmag(s_idx):
        vals = []
        for env_idx in range(3):
            base = env_idx * 18 + s_idx * 3
            mx_ = magnetizations[base]
            my_ = magnetizations[base + 1]
            mz_ = magnetizations[base + 2]
            vals.append((mx_**2 + my_**2 + mz_**2) ** 0.5)
        avg = sum(vals) / 3
        se  = (sum((v - avg)**2 for v in vals) / 3 / 3) ** 0.5
        return avg, se

    ace_parts, bdf_parts = [], []
    for s_idx, s in enumerate(_SITE_LABELS):
        avg, se = _site_localmag(s_idx)
        entry = f"{s}:{avg:.6f}±{se:.2e}"
        (ace_parts if s in ('A', 'C', 'E') else bdf_parts).append(entry)
    print(f"  │    |m| mean±se/env:")
    print(f"  │      ACE  {' | '.join(ace_parts)}")
    print(f"  │      BDF  {' | '.join(bdf_parts)}")


# ── single CTMRG + rho-cache pass ─────────────────────────────────────────────
def _run_ctmrg_and_cache_rhos(params, cfg, D_bond, chi, Js, SdotS, d_PHYS):
    """Run CTMRG to convergence and return (energy, correlations, magnetizations)."""
    D_sq = D_bond ** 2
    with torch.no_grad():
        a, b, c, d, e, f = _derive_abcdef(list(params), cfg, D_bond)
        aN = normalize_single_layer_tensor_for_double_layer(a)
        bN = normalize_single_layer_tensor_for_double_layer(b)
        cN = normalize_single_layer_tensor_for_double_layer(c)
        dN = normalize_single_layer_tensor_for_double_layer(d)
        eN = normalize_single_layer_tensor_for_double_layer(e)
        fN = normalize_single_layer_tensor_for_double_layer(f)
        A, B, C, Dt, E, F = abcdef_to_ABCDEF(aN, bN, cN, dN, eN, fN, D_sq)

        all28 = CTMRG_from_init_to_stop(
            A, B, C, Dt, E, F, chi, D_sq,
            CTM_MAX_STEPS, CTM_CONV_THR, ENV_IDENTITY_INIT,
            energy_proxy_fn=None)
        (C21CD, C32EF, C13AB, T1F, T2A, T2B, T3C, T3D, T1E,
         C21EB, C32AD, C13CF, T1D, T2C, T2F, T3E, T3B, T1A,
         C21AF, C32CB, C13ED, T1B, T2E, T2D, T3A, T3F, T1C, _ctm_steps) = all28

        env1 = (C21CD, C32EF, C13AB, T1F, T2A, T2B, T3C, T3D, T1E)
        env2 = (C21EB, C32AD, C13CF, T1D, T2C, T2F, T3E, T3B, T1A)
        env3 = (C21AF, C32CB, C13ED, T1B, T2E, T2D, T3A, T3F, T1C)

    # Enable rho caching and compute energy (which fills _RHO_CACHE)
    _core._CACHE_RHOS = True
    _core._RHO_CACHE.clear()
    with torch.no_grad():
        _e1 = energy_expectation_nearest_neighbor_3ebadcf_bonds(
            aN, bN, cN, dN, eN, fN,
            *Js[0:12], SdotS, chi, D_bond, d_PHYS, *env1)
        _e2 = energy_expectation_nearest_neighbor_3afcbed_bonds(
            aN, bN, cN, dN, eN, fN,
            *Js[12:24], SdotS, chi, D_bond, d_PHYS, *env2)
        _e3 = energy_expectation_nearest_neighbor_other_3_bonds(
            aN, bN, cN, dN, eN, fN,
            *Js[24:36], SdotS, chi, D_bond, d_PHYS, *env3)
    _core._CACHE_RHOS = False

    rho_data = (
        _core._RHO_CACHE.get('env1'),
        _core._RHO_CACHE.get('env2'),
        _core._RHO_CACHE.get('env3'),
    )
    return _observables_from_rhos(rho_data, Js, SdotS, d_PHYS)


# ── auto-detect ansatz from checkpoint keys ───────────────────────────────────
def _detect_ansatz(ckpt: dict) -> str:
    keys = set(ckpt.keys()) - {'D_bond', 'chi', 'loss', 'energy', 'step',
                                'timestamp', 'log'}
    for name, cfg in ANSATZ_REGISTRY.items():
        if set(cfg['ckpt_keys']) == keys:
            return name
    # fallback
    if 'h' in keys:
        return 'neel'
    if {'h_a', 'h_b'}.issubset(keys):
        return 'sym6'
    if {'a', 'b', 'c', 'd', 'e', 'f'}.issubset(keys):
        return 'unrestricted'
    raise ValueError(f"Cannot auto-detect ansatz from checkpoint keys: {keys}")


# ══════════════════════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser(
        description="Re-compute CTMRG observables from a saved checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--file', '-f', required=True,
                   help='Path to .pt checkpoint (e.g. D_3_chi_54_best.pt)')
    p.add_argument('--J2', type=float, required=True,
                   help='Next-nearest-neighbour coupling J2')
    p.add_argument('--J1', type=float, default=1.0,
                   help='Nearest-neighbour coupling J1')
    p.add_argument('--ansatz', default=None,
                   choices=list(ANSATZ_REGISTRY.keys()),
                   help='Ansatz name (auto-detected from checkpoint keys if omitted)')
    p.add_argument('--d-phys', type=int, default=D_PHYS_DEFAULT,
                   help='Physical bond dimension (2 for spin-1/2)')
    p.add_argument('--chi', type=int, default=None,
                   help='Override chi (default: read from checkpoint)')
    p.add_argument('--no-lookahead', action='store_true',
                   help='Skip the lookahead at chi_la = chi + D_bond')
    p.add_argument('--gpu', action='store_true', default=False,
                   help='Use GPU if available')
    p.add_argument('--double', action='store_true', default=True,
                   help='Use float64 (default)')
    p.add_argument('--single', action='store_true', default=False,
                   help='Use float32 instead of float64')
    p.add_argument('--out-dir', default=None,
                   help='Output directory (default: same dir as --file)')
    args = p.parse_args()

    # ── device + dtype ────────────────────────────────────────────────────────
    use_double = not args.single
    set_dtype(use_double, use_real=True)
    if args.gpu and torch.cuda.is_available():
        dev = torch.device('cuda')
        torch.set_num_threads(1)
        print(f"  Using GPU: {dev} ({torch.cuda.get_device_name(dev)})")
    else:
        dev = torch.device('cpu')
        torch.set_num_threads(_N_CORES)
        print(f"  Using CPU (threads={_N_CORES})")
    set_device(dev)
    _core.set_rsvd_mode('neumann', neumann_terms=2, power_iters=None)
    _core.set_ctm_conv_mode('SVdifference', e_threshold=2e-8)

    # ── load checkpoint ───────────────────────────────────────────────────────
    ckpt_path = os.path.abspath(args.file)
    if not os.path.isfile(ckpt_path):
        sys.exit(f"ERROR: checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=dev)
    D_bond = int(ckpt.get('D_bond', 0))
    chi    = args.chi if args.chi is not None else int(ckpt.get('chi', 0))
    if D_bond == 0 or chi == 0:
        sys.exit("ERROR: checkpoint does not contain D_bond / chi — "
                 "use --chi to override.")

    # ── ansatz ────────────────────────────────────────────────────────────────
    ansatz_name = args.ansatz or _detect_ansatz(ckpt)
    cfg = ANSATZ_REGISTRY[ansatz_name]
    print(f"  Ansatz: {ansatz_name}  D={D_bond}  chi={chi}  "
          f"J1={args.J1}  J2={args.J2}")

    # load tensors
    params = tuple(ckpt[k].to(dev) for k in cfg['ckpt_keys'])
    # neel normalisation
    if ansatz_name == 'neel':
        with torch.no_grad():
            _n = params[0].norm()
            if _n > 1e-30:
                params = (params[0] / _n,)
    elif ansatz_name == 'neel_legacy':
        with torch.no_grad():
            _t = symmetrize_virtual_legs(params[0])
            _n = _t.norm()
            params = (_t / _n if _n > 1e-30 else _t,)
    del ckpt; gc.collect()

    # ── Hamiltonian ───────────────────────────────────────────────────────────
    d_PHYS = args.d_phys
    SdotS  = build_heisenberg_H(1.0, d_PHYS)
    J1, J2 = args.J1, args.J2
    Js = ([J1]*6 + [J2]*6) * 3

    # ── output directory ──────────────────────────────────────────────────────
    out_dir = args.out_dir or os.path.dirname(ckpt_path)
    os.makedirs(out_dir, exist_ok=True)

    # ══════════════════════════════════════════════════════════════════════════
    # OBS pass: fresh CTMRG at chi
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n  Running CTMRG at chi={chi} …")
    t0 = time.perf_counter()
    energy, correlations, magnetizations = _run_ctmrg_and_cache_rhos(
        params, cfg, D_bond, chi, Js, SdotS, d_PHYS)
    print(f"  CTMRG done in {time.perf_counter()-t0:.1f}s")

    obs_path = os.path.join(out_dir,
        f"D_{D_bond}_chi_{chi}_energy_magnetization_correlation.txt")
    _save_observables_file(obs_path, D_bond, chi,
                           energy, correlations, magnetizations)
    _print_observables_summary('OBS', D_bond, chi,
                               energy, correlations, magnetizations)

    # ══════════════════════════════════════════════════════════════════════════
    # LA pass: lookahead at chi_la = chi + D_bond
    # ══════════════════════════════════════════════════════════════════════════
    if not args.no_lookahead:
        chi_la = chi + D_bond
        print(f"\n  [lookahead] running CTMRG at chi_la={chi_la} (D={D_bond}) …")
        t1 = time.perf_counter()
        try:
            energy_la, corr_la, mag_la = _run_ctmrg_and_cache_rhos(
                params, cfg, D_bond, chi_la, Js, SdotS, d_PHYS)
            print(f"  CTMRG done in {time.perf_counter()-t1:.1f}s")

            la_path = os.path.join(out_dir,
                f"D_{D_bond}_chi_{chi}_lookahead_{chi_la}"
                f"_energy_magnetization_correlation.txt")
            _save_observables_file(la_path, D_bond, chi_la,
                                   energy_la, corr_la, mag_la)
            _print_observables_summary('LA ', D_bond, chi_la,
                                       energy_la, corr_la, mag_la)

            _delta_E = energy - energy_la
            print(f"\n  ΔE(chi→chi_la) = {_delta_E:.3e}  "
                  f"({'converged' if abs(_delta_E) <= 3e-5 else 'not converged'})")
        except Exception as exc:
            print(f"  [lookahead] failed: {exc}")

    print(f"\n  Done.  Output in {out_dir}")


if __name__ == '__main__':
    main()
