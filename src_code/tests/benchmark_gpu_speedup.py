#!/usr/bin/env python3
"""
benchmark_gpu_speedup.py
========================
Measures wall-clock time for the critical CTMRG hot paths and reports:
  1. Microbenchmark: check_env_CV_using_3rho  (was 18 svdvals, now 2 batched)
  2. Microbenchmark: trunc_rhoCCC             (was 3 SVDs, now 1 batched)
  3. Full CTMRG step (update_1to2 + 2to3 + 3to1 + convergence check)
  4. Full CTMRG run (reports actual convergence iteration count)
  5. Peak CUDA memory (if GPU available)

Run on CPU (local):
    python tests/benchmark_gpu_speedup.py

Run on GPU cluster:
    python tests/benchmark_gpu_speedup.py --gpu

Output helps diagnose whether batching eliminates the 100x slowdown.
"""

import sys, os, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np

import core_unrestricted as core
from core_unrestricted import (
    set_dtype, set_device,
    initialize_abcdef, abcdef_to_ABCDEF,
    trunc_rhoCCC, check_env_CV_using_3rho,
    update_environmentCTs_1to2, update_environmentCTs_2to3, update_environmentCTs_3to1,
    CTMRG_from_init_to_stop,
    initialize_envCTs_1,
)
import opt_einsum as oe

# ── Slow sequential baselines (reimplemented here for direct comparison) ──────

def _check_env_CV_sequential_baseline(
        lastC21CD, lastC32EF, lastC13AB,
        nowC21CD,  nowC32EF,  nowC13AB,
        lastC21EB, lastC32AD, lastC13CF,
        nowC21EB,  nowC32AD,  nowC13CF,
        lastC21AF, lastC32CB, lastC13ED,
        nowC21AF,  nowC32CB,  nowC13ED,
        env_conv_threshold):
    """Original 18-call sequential implementation (baseline for timing comparison)."""
    dev = lastC21CD.device
    max_delta = torch.tensor(0.0, dtype=torch.float64, device=dev)

    def _sv_norm(m): s = torch.linalg.svdvals(m).real; return s / (s[0:1] + 1e-30)

    rho_pairs = [
        (oe.contract("UZ,ZY,YV->UV", lastC13AB, lastC32EF, lastC21CD, optimize=[(0,1),(0,1)], backend='torch'),
         oe.contract("UZ,ZY,YV->UV",  nowC13AB,  nowC32EF,  nowC21CD, optimize=[(0,1),(0,1)], backend='torch')),
        (oe.contract("UZ,ZY,YV->UV", lastC13CF, lastC32AD, lastC21EB, optimize=[(0,1),(0,1)], backend='torch'),
         oe.contract("UZ,ZY,YV->UV",  nowC13CF,  nowC32AD,  nowC21EB, optimize=[(0,1),(0,1)], backend='torch')),
        (oe.contract("UZ,ZY,YV->UV", lastC13ED, lastC32CB, lastC21AF, optimize=[(0,1),(0,1)], backend='torch'),
         oe.contract("UZ,ZY,YV->UV",  nowC13ED,  nowC32CB,  nowC21AF, optimize=[(0,1),(0,1)], backend='torch')),
        (oe.contract("UY,YX,XV->UV", lastC32EF, lastC21CD, lastC13AB, optimize=[(0,1),(0,1)], backend='torch'),
         oe.contract("UY,YX,XV->UV",  nowC32EF,  nowC21CD,  nowC13AB, optimize=[(0,1),(0,1)], backend='torch')),
        (oe.contract("UY,YX,XV->UV", lastC32AD, lastC21EB, lastC13CF, optimize=[(0,1),(0,1)], backend='torch'),
         oe.contract("UY,YX,XV->UV",  nowC32AD,  nowC21EB,  nowC13CF, optimize=[(0,1),(0,1)], backend='torch')),
        (oe.contract("UY,YX,XV->UV", lastC32CB, lastC21AF, lastC13ED, optimize=[(0,1),(0,1)], backend='torch'),
         oe.contract("UY,YX,XV->UV",  nowC32CB,  nowC21AF,  nowC13ED, optimize=[(0,1),(0,1)], backend='torch')),
        (oe.contract("UX,XZ,ZV->UV", lastC21CD, lastC13AB, lastC32EF, optimize=[(0,1),(0,1)], backend='torch'),
         oe.contract("UX,XZ,ZV->UV",  nowC21CD,  nowC13AB,  nowC32EF, optimize=[(0,1),(0,1)], backend='torch')),
        (oe.contract("UX,XZ,ZV->UV", lastC21EB, lastC13CF, lastC32AD, optimize=[(0,1),(0,1)], backend='torch'),
         oe.contract("UX,XZ,ZV->UV",  nowC21EB,  nowC13CF,  nowC32AD, optimize=[(0,1),(0,1)], backend='torch')),
        (oe.contract("UX,XZ,ZV->UV", lastC21AF, lastC13ED, lastC32CB, optimize=[(0,1),(0,1)], backend='torch'),
         oe.contract("UX,XZ,ZV->UV",  nowC21AF,  nowC13ED,  nowC32CB, optimize=[(0,1),(0,1)], backend='torch')),
    ]
    for (rho_l, rho_n) in rho_pairs:
        sv_l = _sv_norm(rho_l)
        sv_n = _sv_norm(rho_n)
        delta = (sv_n - sv_l).abs().max()
        max_delta = torch.maximum(max_delta, delta)

    if device_is_gpu:
        torch.cuda.synchronize()
    return (max_delta < env_conv_threshold).item()


def _trunc_rhoCCC_sequential_baseline(matC21, matC32, matC13, chi, D_squared):
    """Original 3-call sequential SVD implementation (baseline for timing comparison)."""
    from core_unrestricted import truncated_svd_propack, _safe_sqrt_inv_diag

    def _proj(R1, R2):
        U, S, V = truncated_svd_propack(torch.mm(R1, R2.T), chi,
                    chi_extra=round(2*np.sqrt(D_squared)), rel_cutoff=1e-12,
                    v0=None, keep_multiplets=False, abs_tol=1e-14, eps_multiplet=1e-12)
        sqI = _safe_sqrt_inv_diag(S[:chi])
        P_right = torch.mm(R2.T, torch.mm(V, sqI))
        P_left  = torch.mm(torch.mm(sqI, U.conj().T), R1)
        return P_right, P_left

    P21My, P32yM = _proj(matC21.T, torch.mm(matC13, matC32))
    P32Nz, P13zN = _proj(matC32.T, torch.mm(matC21, matC13))
    P13Lx, P21xL = _proj(matC13.T, torch.mm(matC32, matC21))

    C21 = normalize(torch.mm(P21My.T, torch.mm(matC21, P21xL.T)))
    C32 = normalize(torch.mm(P32Nz.T, torch.mm(matC32, P32yM.T)))
    C13 = normalize(torch.mm(P13Lx.T, torch.mm(matC13, P13zN.T)))
    return P21xL.reshape(chi,chi,D_squared), C21, P21My.reshape(chi,D_squared,chi), \
           P32yM.reshape(chi,chi,D_squared), C32, P32Nz.reshape(chi,D_squared,chi), \
           P13zN.reshape(chi,chi,D_squared), C13, P13Lx.reshape(chi,D_squared,chi)

def normalize(t):
    n = torch.linalg.norm(t); return t / n.clamp(min=1e-30)


# ── Timing utilities ──────────────────────────────────────────────────────────

def sync():
    if device_is_gpu:
        torch.cuda.synchronize()

def timeit(fn, n_warmup=3, n_repeat=10, label=""):
    for _ in range(n_warmup):
        fn()
        sync()
    times = []
    for _ in range(n_repeat):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        times.append(time.perf_counter() - t0)
    t_ms = np.array(times) * 1000
    print(f"  {label:50s}  {np.mean(t_ms):7.2f} ± {np.std(t_ms):5.2f} ms  (min {np.min(t_ms):.2f})")
    return np.mean(t_ms)


# ── Main benchmark ────────────────────────────────────────────────────────────

def benchmark(D_bond, chi, device):
    global device_is_gpu
    device_is_gpu = device.type == 'cuda'

    print(f"\n{'='*70}")
    print(f"  D={D_bond}  chi={chi}  device={device}  dtype=float64")
    print(f"{'='*70}")

    set_dtype(use_double=True, use_real=True)
    set_device(device)

    RDTYPE = torch.float64
    D_sq = D_bond * D_bond

    # ── Build random iPEPS tensors ─────────────────────────────────────────
    torch.manual_seed(42)
    d_PHYS = 2
    a,b,c,d,e,f = [torch.randn(D_bond,D_bond,D_bond,d_PHYS, dtype=RDTYPE, device=device)
                   for _ in range(6)]
    A,B,C,D_t,E,F = abcdef_to_ABCDEF(a,b,c,d,e,f, D_sq)

    # ── Initialize environment ─────────────────────────────────────────────
    print("\n[init envCTs_1]")
    with torch.no_grad():
        t_init = time.perf_counter()
        C21CD, C32EF, C13AB, T1F, T2A, T2B, T3C, T3D, T1E = \
            initialize_envCTs_1(A,B,C,D_t,E,F, chi, D_sq)
        sync()
        print(f"  initialize_envCTs_1                                     "
              f"  {(time.perf_counter()-t_init)*1000:7.2f} ms  (one-shot)")

    # Run one CTMRG step to get a plausible 'last' and 'now' environment
    with torch.no_grad():
        res1 = update_environmentCTs_1to2(C21CD, C32EF, C13AB, T1F, T2A, T2B, T3C, T3D, T1E,
                                           A,B,C,D_t,E,F, chi, D_sq)
        C21EB, C32AD, C13CF, T1D, T2C, T2F, T3E, T3B, T1A = res1
        res2 = update_environmentCTs_2to3(C21EB, C32AD, C13CF, T1D, T2C, T2F, T3E, T3B, T1A,
                                           A,B,C,D_t,E,F, chi, D_sq)
        C21AF, C32CB, C13ED, T1B, T2E, T2D, T3A, T3F_, T1C = res2
        res3 = update_environmentCTs_3to1(C21AF, C32CB, C13ED, T1B, T2E, T2D, T3A, T3F_, T1C,
                                           A,B,C,D_t,E,F, chi, D_sq)
        C21CD2, C32EF2, C13AB2, T1F2, T2A2, T2B2, T3C2, T3D2, T1E2 = res3
        sync()

    # ── Microbenchmark: check_env_CV ───────────────────────────────────────
    print("\n[check_env_CV_using_3rho  —  convergence check]")
    # Dummy 'last' environment = initial, 'now' = after 1 step
    cv_args = (
        C21CD, C32EF, C13AB, C21CD2, C32EF2, C13AB2,
        C21EB, C32AD, C13CF, C21EB,  C32AD,  C13CF,   # use same for EB/AD/CF as dummy
        C21AF, C32CB, C13ED, C21AF,  C32CB,  C13ED,
        1e-7,
    )

    t_new = timeit(lambda: check_env_CV_using_3rho(*cv_args),
                   label="NEW batched (2 svdvals calls on (9,chi,chi))")
    t_old = timeit(lambda: _check_env_CV_sequential_baseline(*cv_args),
                   label="OLD sequential (18 individual svdvals calls)")
    if t_old > 0:
        print(f"  --> Speedup check_env_CV: {t_old/t_new:.1f}x  "
              f"({'FASTER' if t_new < t_old else 'SLOWER'})")

    # ── Microbenchmark: trunc_rhoCCC ───────────────────────────────────────
    print("\n[trunc_rhoCCC  —  projector SVD]")
    # Use grown (chi*D_sq, chi*D_sq) corner matrices
    Cbig = C21CD.reshape(chi, chi)   # already chi×chi after init
    # Build the (chi*D_sq, chi*D_sq) corners that trunc_rhoCCC actually receives
    # by doing a corner absorption first
    with torch.no_grad():
        C21big = oe.contract("YX,MYa,LXb,amg,lbg->MmLl", C21CD, T1F, T2A, E, B,
                            optimize=[(0,1),(1,3),(0,1),(0,1)], backend='torch').reshape(chi*D_sq, chi*D_sq)
        C32big = oe.contract("ZY,NZb,MYg,abn,amg->NnMm", C32EF, T2B, T3C, A, F,
                            optimize=[(0,1),(1,3),(0,1),(0,1)], backend='torch').reshape(chi*D_sq, chi*D_sq)
        C13big = oe.contract("XZ,LXg,NZa,lbg,abn->LlNn", C13AB, T3D, T1E, C, D_t,
                            optimize=[(0,1),(1,3),(0,1),(0,1)], backend='torch').reshape(chi*D_sq, chi*D_sq)
        sync()

    N = chi * D_sq
    print(f"  Matrix size: ({N} × {N})  full SVD size: ({N} × {N})")

    t_new_svd = timeit(lambda: trunc_rhoCCC(C21big, C32big, C13big, chi, D_sq),
                       label="NEW batched (1 svd call on (3,N,N))")
    with torch.no_grad():
        t_old_svd = timeit(lambda: _trunc_rhoCCC_sequential_baseline(C21big, C32big, C13big, chi, D_sq),
                           label="OLD sequential (3 svd calls on (N,N))")
    if t_old_svd > 0:
        print(f"  --> Speedup trunc_rhoCCC: {t_old_svd/t_new_svd:.1f}x  "
              f"({'FASTER' if t_new_svd < t_old_svd else 'SLOWER'})")

    # ── Full single CTMRG iteration ────────────────────────────────────────
    print("\n[Full single CTMRG iteration (1to2 + 2to3 + 3to1 + check_cv)]")

    def one_ctm_step():
        with torch.no_grad():
            r1 = update_environmentCTs_1to2(C21CD, C32EF, C13AB, T1F, T2A, T2B, T3C, T3D, T1E,
                                             A,B,C,D_t,E,F, chi, D_sq)
            C21eb, C32ad, C13cf, T1d, T2c, T2f, T3e, T3b, T1a = r1
            r2 = update_environmentCTs_2to3(C21eb, C32ad, C13cf, T1d, T2c, T2f, T3e, T3b, T1a,
                                             A,B,C,D_t,E,F, chi, D_sq)
            C21af, C32cb, C13ed, T1b, T2e, T2d, T3a, T3f_, T1c = r2
            r3 = update_environmentCTs_3to1(C21af, C32cb, C13ed, T1b, T2e, T2d, T3a, T3f_, T1c,
                                             A,B,C,D_t,E,F, chi, D_sq)
            C21cd2, C32ef2, C13ab2, *_ = r3
            check_env_CV_using_3rho(
                C21CD, C32EF, C13AB, C21cd2, C32ef2, C13ab2,
                C21eb, C32ad, C13cf, C21eb,  C32ad,  C13cf,
                C21af, C32cb, C13ed, C21af,  C32cb,  C13ed,
                1e-7,
            )

    t_step = timeit(one_ctm_step, n_warmup=2, n_repeat=5, label="Full CTM step (3 updates + CV check)")

    # ── Full CTMRG convergence run ─────────────────────────────────────────
    print("\n[Full CTMRG_from_init_to_stop — convergence count]")
    with torch.no_grad():
        t0 = time.perf_counter()
        result = CTMRG_from_init_to_stop(
            A, B, C, D_t, E, F,
            chi=chi, D_squared=D_sq,
            a_third_max_iterations=40,  # 40 is the cap (3×40 = 120 total update steps)
            env_conv_threshold=9e-7,
            identity_init=False,
        )
        sync()
        t_total_ctmrg = (time.perf_counter() - t0) * 1000

    # Count how many iterations happened: CTMRG_from_init_to_stop
    # returns the 9-tuple of env tensors; the iteration count is logged
    # in the driver, but we can infer it from time / step_time
    inferred_iters = round(t_total_ctmrg / t_step)
    print(f"  Total CTMRG time: {t_total_ctmrg:.1f} ms")
    print(f"  Per-step time:    {t_step:.2f} ms")
    print(f"  Inferred iters:   ~{inferred_iters}  (cap=40 → converged={'LIKELY' if inferred_iters < 35 else 'HIT CAP — NOT CONVERGED'})")

    # ── Peak CUDA memory ───────────────────────────────────────────────────
    if device_is_gpu:
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            CTMRG_from_init_to_stop(A, B, C, D_t, E, F, chi=chi, D_squared=D_sq,
                                    a_third_max_iterations=5, env_conv_threshold=9e-7)
        peak_mb = torch.cuda.max_memory_allocated(device) / 1e6
        current_mb = torch.cuda.memory_allocated(device) / 1e6
        print(f"\n[CUDA Memory]")
        print(f"  Peak allocated:    {peak_mb:.1f} MB")
        print(f"  Current allocated: {current_mb:.1f} MB")
        print(f"  (N={N}, 6*N²×8bytes = {6*N*N*8/1e6:.1f} MB  for batched SVD outputs)")

    return t_step


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true', help='Use CUDA GPU')
    args = parser.parse_args()

    if args.gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        if args.gpu:
            print("WARNING: --gpu requested but CUDA not available, falling back to CPU")

    # Test at D=4 (typical medium case)
    benchmark(D_bond=4, chi=8,  device=device)
    benchmark(D_bond=4, chi=16, device=device)

    # Test at D=5 (larger case that highlights the overhead ratio)
    benchmark(D_bond=5, chi=10, device=device)
    benchmark(D_bond=5, chi=25, device=device)

    print("\n" + "="*70)
    print("INTERPRETATION:")
    print("  - If 'Inferred iters' is near 40: CTMRG is NOT converging.")
    print("    Root cause: either threshold too tight or numerical oscillation.")
    print("  - If speedup for check_env_CV < 5x: per-call overhead is not ~1ms.")
    print("    May be kernel fusion or the overhead is elsewhere.")
    print("  - If speedup for trunc_rhoCCC < 2x: SVD overhead is not the bottleneck.")
    print("    Check if the corner absorptions (einsums) dominate instead.")
    print("="*70)
