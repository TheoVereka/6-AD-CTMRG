#!/usr/bin/env python3
"""
test_memory_verification.py
===========================
Empirical verification of memory-growth Root Causes identified in diagnostic.

Tests:
  TEST-1  Per-iteration graph growth: WITH vs WITHOUT autograd
  TEST-2  Saved SVD tensor memory: count & size of tensors stored by save_for_backward
  TEST-3  Norm computation overhead in update functions
  TEST-4  Scale projection to D=5,6

All run with D=3, chi=10 (fast) and D=4, chi=17 (matches memory_report.txt).
"""

import os, sys, gc
import torch
import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from core_unrestricted import (
    set_dtype,
    initialize_abcdef,
    abcdef_to_ABCDEF,
    CTMRG_from_init_to_stop,
    update_environmentCTs_1to2,
    update_environmentCTs_2to3,
    update_environmentCTs_3to1,
    initialize_envCTs_1,
    trunc_rhoCCC,
    truncated_svd_propack,
    energy_expectation_nearest_neighbor_3ebadcf_bonds,
    energy_expectation_nearest_neighbor_3afcbed_bonds,
    energy_expectation_nearest_neighbor_other_3_bonds,
    build_heisenberg_H,
    SVD_PROPACK,
    mem_mb,
)

set_dtype(True)  # complex128, matches memory_report.txt

def rss() -> float:
    return psutil.Process().memory_info().rss / 1e6

rss = mem_mb   # alias: both return RSS in MB

def force_gc():
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

SEPARATOR = "=" * 72


# ──────────────────────────────────────────────────────────────────────────────
#  Helpers: count live PyTorch tensors and their sizes
# ──────────────────────────────────────────────────────────────────────────────

def live_torch_tensors():
    """Return (count, total_MB) of all live torch.Tensor objects."""
    count = 0
    total_bytes = 0
    for obj in gc.get_objects():
        try:
            if isinstance(obj, torch.Tensor) and obj.numel() > 0:
                count += 1
                total_bytes += obj.element_size() * obj.numel()
        except Exception:
            pass
    return count, total_bytes / 1e6


def measure_saved_tensor_mb_during_forward(func, *args, **kwargs):
    """Run func(*args) while intercepting save_for_backward calls.
    Returns (result, total_saved_MB, count) of all tensors passed to save_for_backward.
    """
    saved_sizes_mb = []
    saved_count = [0]

    def pack_hook(t):
        saved_count[0] += 1
        saved_sizes_mb.append(t.element_size() * t.numel() / 1e6)
        return t  # return the tensor unchanged

    def unpack_hook(t):
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        result = func(*args, **kwargs)

    total_mb = sum(saved_sizes_mb)
    return result, total_mb, saved_count[0], saved_sizes_mb


# ──────────────────────────────────────────────────────────────────────────────
#  TEST 1: Per-iteration memory growth — WITH vs WITHOUT autograd
# ──────────────────────────────────────────────────────────────────────────────

def test1_autograd_graph_accumulation(D_bond=3, chi=10, n_iter=6):
    print(f"\n{SEPARATOR}")
    print(f"TEST-1: Per-iteration graph growth — D={D_bond}, chi={chi}")
    print(f"{SEPARATOR}")

    d_PHYS = 2
    D_sq = D_bond * D_bond

    a, b, c, d_t, e, f = initialize_abcdef('random', D_bond, d_PHYS, 1e-3)
    for t in (a, b, c, d_t, e, f):
        t.requires_grad_(True)

    force_gc()
    baseline = rss()
    print(f"  Baseline RSS (after init, before any CTMRG): {baseline:.1f} MB")

    # ── WITH autograd: run CTMRG iterations, record RSS each iter ────────────
    print("\n  [WITH autograd]")
    from core_unrestricted import normalize_single_layer_tensor_for_double_layer
    aN = normalize_single_layer_tensor_for_double_layer(a)
    bN = normalize_single_layer_tensor_for_double_layer(b)
    cN = normalize_single_layer_tensor_for_double_layer(c)
    dN = normalize_single_layer_tensor_for_double_layer(d_t)
    eN = normalize_single_layer_tensor_for_double_layer(e)
    fN = normalize_single_layer_tensor_for_double_layer(f)
    A, B, C, D, E, F = abcdef_to_ABCDEF(aN, bN, cN, dN, eN, fN, D_sq)

    # init env (1 iteration at chi_seed)
    env9 = initialize_envCTs_1(A, B, C, D, E, F, chi, D_sq)
    C21CD, C32EF, C13AB, T1F, T2A, T2B, T3C, T3D, T1E = env9

    rss_with_autograd = [rss()]
    print(f"    iter -1 (init):  RSS={rss_with_autograd[-1]:.1f} MB")

    # Keep all previous env tensors alive to simulate real CTMRG loop behavior
    env_history = [env9]

    for i in range(n_iter):
        env_1to2 = update_environmentCTs_1to2(
            C21CD, C32EF, C13AB, T1F, T2A, T2B, T3C, T3D, T1E,
            A, B, C, D, E, F, chi, D_sq)
        C21EB, C32AD, C13CF, T1D, T2C, T2F, T3E, T3B, T1A = env_1to2

        env_2to3 = update_environmentCTs_2to3(
            C21EB, C32AD, C13CF, T1D, T2C, T2F, T3E, T3B, T1A,
            A, B, C, D, E, F, chi, D_sq)
        C21AF, C32CB, C13ED, T1B, T2E_t, T2D, T3A, T3F, T1C = env_2to3

        env_3to1 = update_environmentCTs_3to1(
            C21AF, C32CB, C13ED, T1B, T2E_t, T2D, T3A, T3F, T1C,
            A, B, C, D, E, F, chi, D_sq)
        C21CD, C32EF, C13AB, T1F, T2A, T2B, T3C, T3D, T1E = env_3to1

        env_history.append(env_1to2)
        env_history.append(env_2to3)
        env_history.append(env_3to1)

        r = rss()
        rss_with_autograd.append(r)
        print(f"    iter {i}:  RSS={r:.1f} MB  (+{r - rss_with_autograd[-2]:.1f} MB/3updates)")

    per_iter_with = (rss_with_autograd[-1] - rss_with_autograd[0]) / n_iter
    print(f"  → Avg growth/iter WITH autograd: {per_iter_with:.1f} MB")

    # Cleanup
    del env_history, env_1to2, env_2to3, env_3to1
    del C21CD, C32EF, C13AB, T1F, T2A, T2B, T3C, T3D, T1E
    del C21EB, C32AD, C13CF, T1D, T2C, T2F, T3E, T3B, T1A
    del C21AF, C32CB, C13ED, T1B, T2E_t, T2D, T3A, T3F, T1C
    del A, B, C, D, E, F, aN, bN, cN, dN, eN, fN
    force_gc()

    # ── WITHOUT autograd ─────────────────────────────────────────────────────
    print("\n  [WITHOUT autograd (torch.no_grad)]")
    a2, b2, c2, d2, e2, f2 = initialize_abcdef('random', D_bond, d_PHYS, 1e-3)
    # Does NOT require grad

    aN2 = normalize_single_layer_tensor_for_double_layer(a2)
    bN2 = normalize_single_layer_tensor_for_double_layer(b2)
    cN2 = normalize_single_layer_tensor_for_double_layer(c2)
    dN2 = normalize_single_layer_tensor_for_double_layer(d2)
    eN2 = normalize_single_layer_tensor_for_double_layer(e2)
    fN2 = normalize_single_layer_tensor_for_double_layer(f2)

    force_gc()
    baseline_nograd = rss()

    with torch.no_grad():
        A2, B2, C2, D2, E2, F2 = abcdef_to_ABCDEF(aN2, bN2, cN2, dN2, eN2, fN2, D_sq)
        env9_2 = initialize_envCTs_1(A2, B2, C2, D2, E2, F2, chi, D_sq)
        C21CD2, C32EF2, C13AB2, T1F2, T2A2, T2B2, T3C2, T3D2, T1E2 = env9_2

        rss_no_autograd = [rss()]
        print(f"    iter -1 (init):  RSS={rss_no_autograd[-1]:.1f} MB")

        for i in range(n_iter):
            env_1to2_2 = update_environmentCTs_1to2(
                C21CD2, C32EF2, C13AB2, T1F2, T2A2, T2B2, T3C2, T3D2, T1E2,
                A2, B2, C2, D2, E2, F2, chi, D_sq)
            C21EB2, C32AD2, C13CF2, T1D2, T2C2, T2F2, T3E2, T3B2, T1A2 = env_1to2_2

            env_2to3_2 = update_environmentCTs_2to3(
                C21EB2, C32AD2, C13CF2, T1D2, T2C2, T2F2, T3E2, T3B2, T1A2,
                A2, B2, C2, D2, E2, F2, chi, D_sq)
            C21AF2, C32CB2, C13ED2, T1B2, T2E2_t, T2D2, T3A2, T3F2, T1C2 = env_2to3_2

            env_3to1_2 = update_environmentCTs_3to1(
                C21AF2, C32CB2, C13ED2, T1B2, T2E2_t, T2D2, T3A2, T3F2, T1C2,
                A2, B2, C2, D2, E2, F2, chi, D_sq)
            C21CD2, C32EF2, C13AB2, T1F2, T2A2, T2B2, T3C2, T3D2, T1E2 = env_3to1_2

            r2 = rss()
            rss_no_autograd.append(r2)
            prev = rss_no_autograd[-2]
            print(f"    iter {i}:  RSS={r2:.1f} MB  (+{r2 - prev:.1f} MB/3updates)")

    per_iter_without = (rss_no_autograd[-1] - rss_no_autograd[0]) / n_iter
    print(f"  → Avg growth/iter WITHOUT autograd: {per_iter_without:.1f} MB")

    print(f"\n  ★ AUTOGRAD GRAPH OVERHEAD per iteration: "
          f"{per_iter_with - per_iter_without:.1f} MB "
          f"({per_iter_with:.1f} - {per_iter_without:.1f})")
    print(f"  ★ For {n_iter} iterations: "
          f"{(per_iter_with - per_iter_without)*n_iter:.0f} MB total graph")

    return per_iter_with, per_iter_without


# ──────────────────────────────────────────────────────────────────────────────
#  TEST 2: Saved SVD tensor memory measurement
# ──────────────────────────────────────────────────────────────────────────────

def test2_svd_saved_tensor_size(D_bond=3, chi=10):
    print(f"\n{SEPARATOR}")
    print(f"TEST-2: SVD saved_for_backward tensor sizes — D={D_bond}, chi={chi}")
    print(f"{SEPARATOR}")

    D_sq = D_bond * D_bond
    M_dim = chi * D_sq
    print(f"  Matrix dimension M = chi*D² = {chi}*{D_sq} = {M_dim}")

    dtype = torch.complex128

    Mmat = torch.randn(M_dim, M_dim, dtype=dtype) * 0.1
    Mmat.requires_grad_(True)

    force_gc()
    before = rss()

    # Intercept what SVD_PROPACK.forward actually stores in save_for_backward
    def do_svd():
        return truncated_svd_propack(Mmat, chi, chi_extra=1, rel_cutoff=1e-12)

    (U, S, V), saved_mb, saved_count, saved_sizes = measure_saved_tensor_mb_during_forward(do_svd)

    after = rss()
    delta = after - before

    # Returned (truncated) shapes
    U_ret_mb = U.element_size() * U.numel() / 1e6
    S_ret_mb = S.element_size() * S.numel() / 1e6
    V_ret_mb = V.element_size() * V.numel() / 1e6

    print(f"  Returned U shape: {tuple(U.shape)} → {U_ret_mb:.3f} MB  (AFTER truncation to chi={chi})")
    print(f"  Returned S shape: {tuple(S.shape)} → {S_ret_mb:.5f} MB")
    print(f"  Returned V shape: {tuple(V.shape)} → {V_ret_mb:.3f} MB")
    print(f"  ─── BUT save_for_backward stored FULL-rank tensors ───")
    print(f"  Tensors stored by save_for_backward: {saved_count}")
    for i, sz in enumerate(saved_sizes):
        print(f"    tensor[{i}]: {sz:.4f} MB")
    print(f"  TOTAL save_for_backward: {saved_mb:.3f} MB  (measured via saved_tensors_hooks)")
    print(f"  RSS delta: {delta:.1f} MB")

    # Per-iteration cost
    n_svd_per_iter = 9
    print(f"\n  Per CTMRG iteration: {n_svd_per_iter} SVDs × {saved_mb:.3f} MB = {n_svd_per_iter*saved_mb:.2f} MB")
    print(f"  Over 18 iterations (typical D=3,chi=29): {n_svd_per_iter*saved_mb*18:.1f} MB")

    # D=4, chi=17 projection using hook measurement
    M_d4 = 17 * 16
    per_U_d4 = M_d4 * M_d4 * 16 / 1e6   # complex128, full thin SVD → (M_d4, M_d4)
    per_V_d4 = per_U_d4
    per_S_d4 = M_d4 * 8 / 1e6
    per_svd_d4 = per_U_d4 + per_V_d4 + per_S_d4 + 8/1e6  # + rel_cutoff scalar
    print(f"\n  [D=4, chi=17 projection] M={M_d4}, full thin SVD → ({M_d4},{M_d4})")
    print(f"  Per SVD save: {per_svd_d4:.2f} MB")
    print(f"  9 SVDs × {per_svd_d4:.2f} MB × 4 iter = {9*per_svd_d4*4:.0f} MB")

    del Mmat, U, S, V
    force_gc()
    return saved_mb


# ──────────────────────────────────────────────────────────────────────────────
#  TEST 3: Norm computation memory overhead
# ──────────────────────────────────────────────────────────────────────────────

def test3_norm_computation_overhead(D_bond=3, chi=10):
    print(f"\n{SEPARATOR}")
    print(f"TEST-3: Norm computation overhead per update function — D={D_bond}, chi={chi}")
    print(f"{SEPARATOR}")

    d_PHYS = 2
    D_sq = D_bond * D_bond
    import opt_einsum as oe

    dtype = torch.complex128
    a_t = torch.randn(D_bond, D_bond, D_bond, d_PHYS, dtype=dtype)
    a_t.requires_grad_(True)

    A = oe.contract("uvwp,xyzp->uxvywz", a_t, a_t.conj(),
                    optimize=[(0, 1)], backend='torch')
    A = A.reshape(D_sq, D_sq, D_sq)
    A6 = A.reshape(D_bond, D_bond, D_bond, D_bond, D_bond, D_bond)

    # Create plausible corner and transfer tensors (rank-4 transfer after reshape)
    C = torch.randn(chi, chi, dtype=dtype) * 0.1
    T4 = torch.randn(chi, chi, D_bond, D_bond, dtype=dtype) * 0.1
    # T4 = T.reshape(chi, chi, D_bond, D_bond) in actual code

    force_gc()
    before_norm = rss()

    # ── The norm computation (active code in update functions) ─────────────────
    # Replicates update_environmentCTs_1to2 lines 1648-1705:
    # 6 closed_* matrices of size (chi*D*D, chi*D*D) built via oe.contract
    closed_1 = oe.contract("YX,MYar,arbsct->MbsXct", C, T4, A6,
                           optimize=[(0,1),(0,1)], backend='torch'
                           ).reshape(chi*D_bond*D_bond, chi*D_bond*D_bond)
    closed_2 = oe.contract("MYct,arbsct->YarMbs", T4, A6,
                           optimize=[(0,1)], backend='torch'
                           ).reshape(chi*D_bond*D_bond, chi*D_bond*D_bond)
    closed_3 = oe.contract("ZY,NZbs,arbsct->NctYar", C, T4, A6,
                           optimize=[(0,1),(0,1)], backend='torch'
                           ).reshape(chi*D_bond*D_bond, chi*D_bond*D_bond)
    closed_4 = oe.contract("NZar,arbsct->ZbsNct", T4, A6,
                           optimize=[(0,1)], backend='torch'
                           ).reshape(chi*D_bond*D_bond, chi*D_bond*D_bond)
    closed_5 = oe.contract("XZ,LXct,arbsct->LarZbs", C, T4, A6,
                           optimize=[(0,1),(0,1)], backend='torch'
                           ).reshape(chi*D_bond*D_bond, chi*D_bond*D_bond)
    closed_6 = oe.contract("LXbs,arbsct->XctLar", T4, A6,
                           optimize=[(0,1)], backend='torch'
                           ).reshape(chi*D_bond*D_bond, chi*D_bond*D_bond)

    # Three matrix products
    P1 = torch.mm(closed_1, closed_2)
    P2 = torch.mm(closed_3, closed_4)
    P3 = torch.mm(closed_5, closed_6)
    TrRho = oe.contract("xy,yz,zx->", P1, P2, P3, backend='torch')

    after_norm = rss()
    delta_norm = after_norm - before_norm

    M_size = chi * D_bond * D_bond
    each_closed_mb = 2 * M_size * M_size * 8 / 1e6  # complex128
    total_explicit_mb = 9 * each_closed_mb

    print(f"  M_size = chi*D*D = {chi}*{D_bond}*{D_bond} = {M_size}")
    print(f"  Each closed_* matrix: ({M_size},{M_size}) complex128 = {each_closed_mb:.3f} MB")
    print(f"  6 closed + 3 products = 9 matrices × {each_closed_mb:.3f} MB = {total_explicit_mb:.2f} MB explicit")
    print(f"  RSS delta (incl. autograd saved intermediates): {delta_norm:.1f} MB")
    print(f"  3 update functions per iteration: ~{3*delta_norm:.1f} MB/iter on graph")
    print(f"  Over 4 iterations (D=4 typical): ~{3*delta_norm*4:.0f} MB total norm overhead")

    # D=4, chi=17 projection
    M_d4 = 17 * 4 * 4
    each_closed_d4 = 2 * M_d4 * M_d4 * 8 / 1e6
    print(f"\n  [D=4, chi=17 projection] M_size={M_d4}")
    print(f"  6 closed + 3 prod = 9 × {each_closed_d4:.2f} MB = {9*each_closed_d4:.1f} MB explicit per update")
    print(f"  3 updates × 4 iter = {3*9*each_closed_d4*4:.0f} MB norm overhead (explicit tensors alone)")

    del closed_1, closed_2, closed_3, closed_4, closed_5, closed_6
    del P1, P2, P3, TrRho, A6, A, a_t, C, T4
    force_gc()
    return delta_norm


# ──────────────────────────────────────────────────────────────────────────────
#  TEST 4: Full closure memory — WITH autograd graph, count saved nodes
# ──────────────────────────────────────────────────────────────────────────────

def test4_full_closure_graph_size(D_bond=3, chi=10):
    print(f"\n{SEPARATOR}")
    print(f"TEST-4: Full closure memory budget — D={D_bond}, chi={chi}")
    print(f"{SEPARATOR}")

    d_PHYS = 2
    D_sq = D_bond * D_bond

    Hs = []
    for _ in range(9):
        H = build_heisenberg_H(J=1.0, d=d_PHYS)
        Hs.append(H)

    from core_unrestricted import normalize_single_layer_tensor_for_double_layer

    a, b, c, d_t, e, f = initialize_abcdef('random', D_bond, d_PHYS, 1e-3)
    for t in (a, b, c, d_t, e, f):
        t.requires_grad_(True)

    # ── Full closure wrapped in saved_tensors_hooks ────────────────────────────
    def full_closure():
        aN = normalize_single_layer_tensor_for_double_layer(a)
        bN = normalize_single_layer_tensor_for_double_layer(b)
        cN = normalize_single_layer_tensor_for_double_layer(c)
        dN = normalize_single_layer_tensor_for_double_layer(d_t)
        eN = normalize_single_layer_tensor_for_double_layer(e)
        fN = normalize_single_layer_tensor_for_double_layer(f)
        A, B, C, D2, E, F = abcdef_to_ABCDEF(aN, bN, cN, dN, eN, fN, D_sq)
        all28 = CTMRG_from_init_to_stop(A, B, C, D2, E, F, chi, D_sq, 10, 4e-7)
        (C21CD, C32EF, C13AB, T1F, T2A, T2B, T3C, T3D, T1E,
         C21EB, C32AD, C13CF, T1D, T2C, T2F, T3E, T3B, T1A,
         C21AF, C32CB, C13ED, T1B, T2E_t, T2D, T3A, T3F, T1C,
         ctm_steps) = all28
        print(f"    CTMRG converged in {ctm_steps} iterations")
        loss = (
            energy_expectation_nearest_neighbor_3ebadcf_bonds(
                aN, bN, cN, dN, eN, fN, Hs[0], Hs[1], Hs[2],
                chi, D_bond, d_PHYS, C21CD, C32EF, C13AB, T1F, T2A, T2B, T3C, T3D, T1E)
            + energy_expectation_nearest_neighbor_3afcbed_bonds(
                aN, bN, cN, dN, eN, fN, Hs[3], Hs[4], Hs[5],
                chi, D_bond, d_PHYS, C21EB, C32AD, C13CF, T1D, T2C, T2F, T3E, T3B, T1A)
            + energy_expectation_nearest_neighbor_other_3_bonds(
                aN, bN, cN, dN, eN, fN, Hs[6], Hs[7], Hs[8],
                chi, D_bond, d_PHYS, C21AF, C32CB, C13ED, T1B, T2E_t, T2D, T3A, T3F, T1C)
        )
        return loss

    force_gc()
    before_closure = rss()
    n_tensors_before, mb_before = live_torch_tensors()
    print(f"  RSS before closure:     {before_closure:.1f} MB   ({n_tensors_before} tensors = {mb_before:.1f} MB data)")

    # Run forward pass while intercepting all save_for_backward calls
    loss, total_saved_mb, n_saved, saved_sizes = measure_saved_tensor_mb_during_forward(full_closure)

    after_fwd = rss()
    n_tensors_after, mb_after = live_torch_tensors()
    print(f"  RSS after forward:      {after_fwd:.1f} MB  (+{after_fwd-before_closure:.1f} MB)")
    print(f"  Live tensors after:     {n_tensors_after} ({mb_after:.1f} MB data)")
    print(f"  New tensor objects:     {n_tensors_after - n_tensors_before} ({mb_after - mb_before:.1f} MB)")
    print(f"\n  ★ save_for_backward total: {n_saved} tensors = {total_saved_mb:.1f} MB")
    print(f"     (this is the irreducible minimum for backward to work)")
    print(f"     RSS overhead above saved tensors: {(after_fwd-before_closure) - total_saved_mb:.1f} MB (intermediates + metadata)")

    # Top saved tensor sizes
    from collections import Counter
    size_counts = Counter(f"{s:.2f}MB" for s in saved_sizes)
    print(f"  Top saved tensor sizes:")
    for size_str, cnt in size_counts.most_common(8):
        print(f"    {size_str} × {cnt} = {float(size_str[:-2]) * cnt:.1f} MB")

    # ── Backward ──────────────────────────────────────────────────────────────
    before_bwd = rss()
    loss.backward()
    after_bwd = rss()
    print(f"\n  RSS during backward:    {after_bwd:.1f} MB  (+{after_bwd-before_bwd:.1f} MB peak grad alloc)")

    del loss
    force_gc()
    after_del = rss()
    print(f"  RSS after graph freed:  {after_del:.1f} MB  (freed {after_bwd - after_del:.1f} MB)")
    print(f"  Persistent (grads+model): {after_del - before_closure + mb_before:.1f} MB")


# ──────────────────────────────────────────────────────────────────────────────
#  TEST 5: D=4, chi=17 — direct comparison to memory_report.txt
# ──────────────────────────────────────────────────────────────────────────────

def test5_d4_chi17_comparison():
    print(f"\n{SEPARATOR}")
    print(f"TEST-5: D=4, chi=17 — match memory_report.txt figures")
    print(f"{SEPARATOR}")
    print("  (Memory report shows: MEM-B=4355 MB, per-iter growth ~333 MB/iter)")

    D_bond = 4
    chi = 17
    D_sq = D_bond * D_bond
    d_PHYS = 2

    from core_unrestricted import normalize_single_layer_tensor_for_double_layer

    a, b, c, d_t, e, f = initialize_abcdef('random', D_bond, d_PHYS, 1e-3)
    for t in (a, b, c, d_t, e, f):
        t.requires_grad_(True)

    aN = normalize_single_layer_tensor_for_double_layer(a)
    bN = normalize_single_layer_tensor_for_double_layer(b)
    cN = normalize_single_layer_tensor_for_double_layer(c)
    dN = normalize_single_layer_tensor_for_double_layer(d_t)
    eN = normalize_single_layer_tensor_for_double_layer(e)
    fN = normalize_single_layer_tensor_for_double_layer(f)

    A, B, C, Dbl, E, F = abcdef_to_ABCDEF(aN, bN, cN, dN, eN, fN, D_sq)

    env9 = initialize_envCTs_1(A, B, C, Dbl, E, F, chi, D_sq)
    C21CD, C32EF, C13AB, T1F, T2A, T2B, T3C, T3D, T1E = env9

    force_gc()
    baseline = rss()
    print(f"  RSS baseline (after init, before CTMRG loop): {baseline:.1f} MB")

    rss_iter = [baseline]

    for i in range(4):
        env_1to2 = update_environmentCTs_1to2(
            C21CD, C32EF, C13AB, T1F, T2A, T2B, T3C, T3D, T1E,
            A, B, C, Dbl, E, F, chi, D_sq)
        C21EB, C32AD, C13CF, T1D, T2C, T2F, T3E, T3B, T1A = env_1to2

        env_2to3 = update_environmentCTs_2to3(
            C21EB, C32AD, C13CF, T1D, T2C, T2F, T3E, T3B, T1A,
            A, B, C, Dbl, E, F, chi, D_sq)
        C21AF, C32CB, C13ED, T1B, T2E_t, T2D, T3A, T3F, T1C = env_2to3

        env_3to1 = update_environmentCTs_3to1(
            C21AF, C32CB, C13ED, T1B, T2E_t, T2D, T3A, T3F, T1C,
            A, B, C, Dbl, E, F, chi, D_sq)
        C21CD, C32EF, C13AB, T1F, T2A, T2B, T3C, T3D, T1E = env_3to1

        r = rss()
        rss_iter.append(r)
        delta = r - rss_iter[-2]
        print(f"  iter {i}: RSS={r:.1f} MB  (+{delta:.1f} MB/iter)")

    actual_per_iter = (rss_iter[-1] - rss_iter[0]) / 4
    expected_per_iter = 333.0  # from memory_report.txt
    print(f"\n  Measured per-iter growth: {actual_per_iter:.1f} MB")
    print(f"  Expected (from memory_report.txt): ~{expected_per_iter:.0f} MB")
    print(f"  Match: {'✓ YES' if abs(actual_per_iter - expected_per_iter) < 100 else '✗ needs investigation'}")


# ──────────────────────────────────────────────────────────────────────────────
#  TEST 6: Verify norm computation IS on autograd tape (is it needed on tape?)
# ──────────────────────────────────────────────────────────────────────────────

def test6_norm_on_tape(D_bond=3, chi=10):
    print(f"\n{SEPARATOR}")
    print(f"TEST-6: Verify norm computation connects to autograd graph — D={D_bond}, chi={chi}")
    print(f"{SEPARATOR}")

    d_PHYS = 2
    D_sq = D_bond * D_bond

    from core_unrestricted import normalize_single_layer_tensor_for_double_layer

    a, b, c, d_t, e, f = initialize_abcdef('random', D_bond, d_PHYS, 1e-3)
    for t in (a, b, c, d_t, e, f):
        t.requires_grad_(True)

    aN = normalize_single_layer_tensor_for_double_layer(a)
    bN = normalize_single_layer_tensor_for_double_layer(b)
    cN = normalize_single_layer_tensor_for_double_layer(c)
    dN = normalize_single_layer_tensor_for_double_layer(d_t)
    eN = normalize_single_layer_tensor_for_double_layer(e)
    fN = normalize_single_layer_tensor_for_double_layer(f)

    A, B, C, D2, E, F = abcdef_to_ABCDEF(aN, bN, cN, dN, eN, fN, D_sq)

    env9 = initialize_envCTs_1(A, B, C, D2, E, F, chi, D_sq)
    C21CD0, C32EF0, C13AB0, T1F0, T2A0, T2B0, T3C0, T3D0, T1E0 = env9

    force_gc()
    before = rss()

    # Run exactly one update
    (C21EB, C32AD, C13CF, T1D, T2C, T2F,
     T3E, T3B, T1A) = update_environmentCTs_1to2(
        C21CD0, C32EF0, C13AB0, T1F0, T2A0, T2B0, T3C0, T3D0, T1E0,
        A, B, C, D2, E, F, chi, D_sq)

    after = rss()
    print(f"  RSS before one update: {before:.1f} MB")
    print(f"  RSS after  one update: {after:.1f} MB  (+{after-before:.1f} MB)")

    # Check if the returned tensors require_grad and have grad_fn (meaning they're on tape)
    print(f"\n  T3E.requires_grad={T3E.requires_grad}, has grad_fn={T3E.grad_fn is not None}")
    print(f"  T1A.requires_grad={T1A.requires_grad}, has grad_fn={T1A.grad_fn is not None}")
    print(f"  C21EB.requires_grad={C21EB.requires_grad}, has grad_fn={C21EB.grad_fn is not None}")

    # Confirm: does the normalization scalar TrRho follow the gradient path from a?
    # A quick proxy: call the update function twice, once with requires_grad=True
    # and once with no_grad, compare RSS deltas.
    force_gc()
    b2_val = rss()
    with torch.no_grad():
        env9_ng = initialize_envCTs_1(A, B, C, D2, E, F, chi, D_sq)
        C21CD_ng, C32EF_ng, C13AB_ng, T1F_ng, T2A_ng, T2B_ng, T3C_ng, T3D_ng, T1E_ng = env9_ng
        (C21EB_ng, C32AD_ng, C13CF_ng, T1D_ng, T2C_ng, T2F_ng,
         T3E_ng, T3B_ng, T1A_ng) = update_environmentCTs_1to2(
            C21CD_ng, C32EF_ng, C13AB_ng, T1F_ng, T2A_ng, T2B_ng, T3C_ng, T3D_ng, T1E_ng,
            A, B, C, D2, E, F, chi, D_sq)

    b2_after = rss()
    print(f"\n  One update WITH autograd:    +{after-before:.1f} MB")
    print(f"  One update WITHOUT autograd: +{b2_after - b2_val:.1f} MB")
    print(f"  → Autograd overhead for one update: {(after-before) - (b2_after-b2_val):.1f} MB")


# ──────────────────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print(f"Process RSS at startup: {rss():.1f} MB")
    import torch
    print(f"PyTorch {torch.__version__}, dtype=complex128, CPU-only")
    print()

    # Run tests in order
    test1_autograd_graph_accumulation(D_bond=3, chi=10, n_iter=6)
    force_gc()

    test2_svd_saved_tensor_size(D_bond=3, chi=10)
    force_gc()

    test2_svd_saved_tensor_size(D_bond=4, chi=17)   # matches memory_report.txt
    force_gc()

    test3_norm_computation_overhead(D_bond=3, chi=10)
    force_gc()

    test3_norm_computation_overhead(D_bond=4, chi=17)
    force_gc()

    test6_norm_on_tape(D_bond=3, chi=10)
    force_gc()

    test4_full_closure_graph_size(D_bond=3, chi=10)
    force_gc()

    test5_d4_chi17_comparison()
    force_gc()

    print(f"\n{SEPARATOR}")
    print("ALL TESTS COMPLETE")
    print(f"{SEPARATOR}")
