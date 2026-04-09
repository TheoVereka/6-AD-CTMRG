#!/usr/bin/env python3
"""
Benchmark: rSVD precision vs (niter, q_over, neumann_terms)
for CTMRG-realistic matrices at various chi / D / degeneracy levels.

Outputs 3 tables:
  1. niter sweep    (q fixed at 2k):  cosine(U_rsvd, U_exact) vs niter
  2. q_over sweep   (niter fixed at 4): cosine vs q_over
  3. neumann sweep  (L=1..8):  5th-term relative error vs L
  4. augmented k_aug sweep
  5. Timing comparison

All matrices are constructed with controlled spectra to simulate:
  (a) fast-decay  (σ_{k+1}/σ_k = 0.01)  — typical CTMRG
  (b) moderate    (σ_{k+1}/σ_k = 0.3)   — small chi, discarded weight non-trivial
  (c) near-degen  (σ_{k+1}/σ_k = 0.95)  — worst case
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import torch
import numpy as np

torch.set_default_dtype(torch.float64)
torch.manual_seed(42)


# ─── helpers ─────────────────────────────────────────────────────────────────

def make_matrix(N, k, ratio, seed=0):
    """Build N×N matrix with controlled spectrum.
    
    First k singular values = 1.0, 0.9, 0.8, ... (well-separated).
    Remaining N-k values = ratio * S[k-1], ratio^2 * S[k-1], ... (geometric decay from gap).
    
    This simulates the CTMRG spectrum where retained SVs are O(1) and
    discarded SVs decay geometrically.
    """
    rng = np.random.RandomState(seed)
    U_full, _ = np.linalg.qr(rng.randn(N, N))
    V_full, _ = np.linalg.qr(rng.randn(N, N))
    
    S = np.zeros(N)
    # Retained: linearly spaced from 1.0 down to 1.0 - 0.03*(k-1) 
    S[:k] = np.linspace(1.0, max(1.0 - 0.03*(k-1), 0.1), k)
    # Gap at position k: S[k] = ratio * S[k-1]
    s_gap = ratio * S[k-1]
    for i in range(k, N):
        S[i] = s_gap * (ratio ** (i - k))
    
    A = (U_full * S) @ V_full.T
    return torch.tensor(A, dtype=torch.float64)


def subspace_cosine(U_test, U_ref):
    """Cosine similarity between column spans: min singular value of U_ref† U_test."""
    M = U_ref.T @ U_test
    s = torch.linalg.svdvals(M)
    return s.min().item()


def rsvd_forward(A, k, q_over, niter):
    """Standalone rSVD forward (same algorithm as SVD_PROPACK.forward)."""
    m, n = A.shape
    q_over = min(q_over, min(m, n))
    
    Rand = torch.randn(n, q_over, dtype=A.dtype, device=A.device)
    X = A @ Rand
    Q, _ = torch.linalg.qr(X)
    for _ in range(niter):
        Q, _ = torch.linalg.qr(A.T @ Q)
        Q, _ = torch.linalg.qr(A @ Q)
    
    B_proj = Q.T @ A
    U_B, S_all, Vh_all = torch.linalg.svd(B_proj, full_matrices=False)
    U_all = Q @ U_B
    V_all = Vh_all.T
    
    order = torch.argsort(S_all, descending=True)
    S_all = S_all[order]; U_all = U_all[:, order]; V_all = V_all[:, order]
    
    return U_all, S_all, V_all


def exact_svd(A, k):
    """Full thin SVD, return top-k."""
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    return U[:, :k], S[:k], Vh.T[:, :k]


# Import 5th-term solvers
import core_unrestricted as _core
_core.set_dtype(True, True)  # float64, real


# ═════════════════════════════════════════════════════════════════════════════
# BENCHMARK 1: niter sweep (power iterations)
# ═════════════════════════════════════════════════════════════════════════════

def bench_niter():
    print("=" * 80)
    print("BENCHMARK 1: Power iterations (niter) sweep")
    print("  q_over = 2*k (fixed),  vary niter = 0,1,2,3,4,6,8")
    print("  Metric: min cosine(U_rsvd[:,:k], U_exact[:,:k]) over 10 seeds")
    print("=" * 80)
    
    configs = [
        # (N, k, ratio, label)
        (100, 10, 0.01, "fast-decay ρ=0.01"),
        (100, 10, 0.30, "moderate ρ=0.30"),
        (100, 10, 0.95, "near-degen ρ=0.95"),
        (100, 10, 0.99, "extreme ρ=0.99"),
        (200, 20, 0.01, "N=200 fast-decay"),
        (200, 20, 0.95, "N=200 near-degen"),
    ]
    niters = [0, 1, 2, 3, 4, 6, 8]
    n_seeds = 10
    
    print(f"\n{'Config':<25s}", end="")
    for ni in niters:
        print(f"  niter={ni:<2d}", end="")
    print()
    print("-" * (25 + 10 * len(niters)))
    
    for N, k, ratio, label in configs:
        print(f"{label:<25s}", end="")
        for ni in niters:
            cosines = []
            for seed in range(n_seeds):
                A = make_matrix(N, k, ratio, seed=seed)
                U_ex, S_ex, V_ex = exact_svd(A, k)
                U_rs, S_rs, V_rs = rsvd_forward(A, k, 2*k, ni)
                cos = subspace_cosine(U_rs[:, :k], U_ex)
                cosines.append(cos)
            worst = min(cosines)
            print(f"  {worst:.6f}", end="")
        print()
    
    # Timing
    print(f"\n{'Config':<25s}", end="")
    for ni in niters:
        print(f"  niter={ni:<2d}", end="")
    print("  full_svd")
    print("-" * (25 + 10 * len(niters) + 10))
    
    for N, k, ratio, label in configs[:3]:
        A = make_matrix(N, k, ratio, seed=0)
        print(f"{label:<25s}", end="")
        for ni in niters:
            t0 = time.perf_counter()
            for _ in range(100):
                rsvd_forward(A, k, 2*k, ni)
            dt = (time.perf_counter() - t0) / 100 * 1000
            print(f"  {dt:7.3f}ms", end="")
        # full svd timing
        t0 = time.perf_counter()
        for _ in range(100):
            exact_svd(A, k)
        dt = (time.perf_counter() - t0) / 100 * 1000
        print(f"  {dt:7.3f}ms")


# ═════════════════════════════════════════════════════════════════════════════
# BENCHMARK 2: q_over (oversampling) sweep
# ═════════════════════════════════════════════════════════════════════════════

def bench_q_over():
    print("\n" + "=" * 80)
    print("BENCHMARK 2: Oversampling (q_over) sweep")
    print("  niter = 4 (fixed),  vary q_over/k = 1.0, 1.2, 1.5, 2.0, 3.0")
    print("  Metric: min cosine over 10 seeds")
    print("=" * 80)
    
    configs = [
        (100, 10, 0.01, "fast-decay ρ=0.01"),
        (100, 10, 0.30, "moderate ρ=0.30"),
        (100, 10, 0.95, "near-degen ρ=0.95"),
        (100, 10, 0.99, "extreme ρ=0.99"),
        (200, 20, 0.95, "N=200 near-degen"),
    ]
    q_ratios = [1.0, 1.2, 1.5, 2.0, 3.0]
    n_seeds = 10
    
    print(f"\n{'Config':<25s}", end="")
    for qr in q_ratios:
        print(f"  q/k={qr:<4.1f}", end="")
    print()
    print("-" * (25 + 10 * len(q_ratios)))
    
    for N, k, ratio, label in configs:
        print(f"{label:<25s}", end="")
        for qr in q_ratios:
            q_over = int(qr * k)
            cosines = []
            for seed in range(n_seeds):
                A = make_matrix(N, k, ratio, seed=seed)
                U_ex, _, _ = exact_svd(A, k)
                U_rs, _, _ = rsvd_forward(A, k, q_over, 4)
                cos = subspace_cosine(U_rs[:, :k], U_ex)
                cosines.append(cos)
            worst = min(cosines)
            print(f"  {worst:.6f}", end="")
        print()


# ═════════════════════════════════════════════════════════════════════════════
# BENCHMARK 3: Neumann terms sweep (5th-term accuracy)
# ═════════════════════════════════════════════════════════════════════════════

def bench_neumann():
    print("\n" + "=" * 80)
    print("BENCHMARK 3: Neumann terms (L) sweep for 5th-term backward")
    print("  Uses EXACT U,S,V (from full SVD), so this isolates 5th-term quality")
    print("  Metric: ||neumann - exact_eigh|| / ||exact_eigh||")
    print("=" * 80)
    
    configs = [
        (50, 8, 0.01, "fast-decay ρ=0.01"),
        (50, 8, 0.10, "moderate ρ=0.10"),
        (50, 8, 0.30, "moderate ρ=0.30"),
        (50, 8, 0.50, "moderate ρ=0.50"),
        (50, 8, 0.80, "slow-decay ρ=0.80"),
        (50, 8, 0.95, "near-degen ρ=0.95"),
        (100, 15, 0.01, "N=100 fast-decay"),
        (100, 15, 0.30, "N=100 moderate"),
        (100, 15, 0.95, "N=100 near-degen"),
    ]
    L_values = [1, 2, 3, 4, 6, 8]
    
    print(f"\n{'Config':<25s}", end="")
    for L in L_values:
        print(f"  L={L:<2d}     ", end="")
    print()
    print("-" * (25 + 11 * len(L_values)))
    
    for N, k, ratio, label in configs:
        A = make_matrix(N, k, ratio, seed=42)
        U_full, S_full, Vh = torch.linalg.svd(A, full_matrices=False)
        V_full = Vh.T
        U_k, S_k, V_k = U_full[:, :k], S_full[:k], V_full[:, :k]
        
        # Random upstream gradients
        torch.manual_seed(123)
        gu = torch.randn(N, k, dtype=torch.float64)
        gv = torch.randn(N, k, dtype=torch.float64)
        
        # Exact reference
        ref = _core._solve_fifth_term_svd(A, U_k, S_k, V_k, gu, gv, eps=1e-12)
        ref_norm = ref.norm().item()
        
        print(f"{label:<25s}", end="")
        for L in L_values:
            approx = _core._solve_fifth_term_neumann(A, U_k, S_k, V_k, gu, gv, n_terms=L)
            err = (approx - ref).norm().item() / max(ref_norm, 1e-30)
            print(f"  {err:.2e} ", end="")
        print()


# ═════════════════════════════════════════════════════════════════════════════
# BENCHMARK 4: Augmented k_aug sweep
# ═════════════════════════════════════════════════════════════════════════════

def bench_augmented():
    print("\n" + "=" * 80)
    print("BENCHMARK 4: Augmented backward — k_aug sweep")
    print("  Extra modes captured from rSVD forward (q_over=2k)")
    print("  Zero-padding backward: grad(||USVh - C||^2) through SVD_PROPACK")
    print("  Metric: ||grad_augmented - grad_full_svd|| / ||grad_full_svd||")
    print("=" * 80)
    
    configs = [
        (50, 8, 0.01, "fast-decay ρ=0.01"),
        (50, 8, 0.30, "moderate ρ=0.30"),
        (50, 8, 0.95, "near-degen ρ=0.95"),
        (100, 15, 0.01, "N=100 fast-decay"),
        (100, 15, 0.30, "N=100 moderate"),
        (100, 15, 0.95, "N=100 near-degen"),
    ]
    # k_aug_extra values: 0 (=none), k//2, k, 2k (=q_over)
    
    print(f"\n{'Config':<25s}  k_aug=k   k_aug=1.5k k_aug=2k(=q)  full_svd")
    print("-" * 80)
    
    for N, k, ratio, label in configs:
        A_np = make_matrix(N, k, ratio, seed=42)
        
        def compute_grad(mode, neumann_terms, augment_extra):
            """Compute dL/dA for loss = ||U*S*Vh - C||^2 through SVD_PROPACK."""
            _core.set_rsvd_mode(mode, neumann_terms=neumann_terms, augment_extra=augment_extra)
            A = A_np.clone().requires_grad_(True)
            
            # Target matrix C (fixed)
            torch.manual_seed(999)
            C = torch.randn(N, N, dtype=torch.float64)
            
            U, S, V = _core.truncated_svd_propack(A, k, chi_extra=1, rel_cutoff=1e-12)
            recon = U @ torch.diag(S) @ V.T
            loss = (recon - C).pow(2).sum()
            loss.backward()
            return A.grad.clone()
        
        # Reference: full_svd
        grad_ref = compute_grad('full_svd', 4, 0)
        ref_norm = grad_ref.norm().item()
        
        print(f"{label:<25s}", end="")
        
        # k_aug = k (no extra = 'none')
        grad_none = compute_grad('none', 4, 0)
        err_none = (grad_none - grad_ref).norm().item() / ref_norm
        print(f"  {err_none:.2e}", end="")
        
        # k_aug = 1.5k
        extra_half = max(k // 2, 1)
        grad_aug_half = compute_grad('augmented', 4, extra_half)
        err_half = (grad_aug_half - grad_ref).norm().item() / ref_norm
        print(f"   {err_half:.2e} ", end="")
        
        # k_aug = 2k (= q_over, reuse everything from rSVD)
        grad_aug_full = compute_grad('augmented', 4, k)
        err_full = (grad_aug_full - grad_ref).norm().item() / ref_norm
        print(f"   {err_full:.2e}  ", end="")
        
        # full_svd (should be 0)
        grad_fs = compute_grad('full_svd', 4, 0)
        err_fs = (grad_fs - grad_ref).norm().item() / ref_norm
        print(f"    {err_fs:.2e}")
    
    # Reset
    _core.set_rsvd_mode('full_svd')


# ═════════════════════════════════════════════════════════════════════════════
# BENCHMARK 5: End-to-end gradient precision (rSVD forward + backward combined)
# ═════════════════════════════════════════════════════════════════════════════

def bench_end_to_end():
    print("\n" + "=" * 80)
    print("BENCHMARK 5: End-to-end gradient error (rSVD forward + backward)")
    print("  Compares gradient through SVD_PROPACK vs full_svd reference")
    print("  for each (mode, niter, q_over) combination")
    print("  Metric: ||grad - grad_full|| / ||grad_full|| averaged over 5 seeds")
    print("=" * 80)
    
    configs = [
        (80, 10, 0.01, "fast ρ=0.01"),
        (80, 10, 0.30, "mod ρ=0.30"),
        (80, 10, 0.95, "degen ρ=0.95"),
    ]
    
    modes = [
        ("neumann L=1", 'neumann', 1, 0),
        ("neumann L=2", 'neumann', 2, 0),
        ("neumann L=4", 'neumann', 4, 0),
        ("aug k_aug=2k", 'augmented', 4, None),  # None = use k
        ("none",         'none',     4, 0),
    ]
    
    n_seeds = 5
    
    print(f"\n{'Config':<18s}", end="")
    for mlabel, _, _, _ in modes:
        print(f"  {mlabel:<14s}", end="")
    print()
    print("-" * (18 + 16 * len(modes)))
    
    for N, k, ratio, label in configs:
        print(f"{label:<18s}", end="")
        for mlabel, mode, n_terms, aug_extra in modes:
            errs = []
            for seed in range(n_seeds):
                A_np = make_matrix(N, k, ratio, seed=seed)
                torch.manual_seed(999 + seed)
                C = torch.randn(N, N, dtype=torch.float64)
                
                # Full SVD reference
                _core.set_rsvd_mode('full_svd')
                A1 = A_np.clone().requires_grad_(True)
                U1, S1, V1 = _core.truncated_svd_propack(A1, k, chi_extra=1, rel_cutoff=1e-12)
                loss1 = (U1 @ torch.diag(S1) @ V1.T - C).pow(2).sum()
                loss1.backward()
                g_ref = A1.grad.clone()
                
                # Mode under test
                ae = aug_extra if aug_extra is not None else k  # k_aug = 2k
                _core.set_rsvd_mode(mode, neumann_terms=n_terms, augment_extra=ae)
                A2 = A_np.clone().requires_grad_(True)
                U2, S2, V2 = _core.truncated_svd_propack(A2, k, chi_extra=1, rel_cutoff=1e-12)
                loss2 = (U2 @ torch.diag(S2) @ V2.T - C).pow(2).sum()
                loss2.backward()
                g_test = A2.grad.clone()
                
                err = (g_test - g_ref).norm().item() / max(g_ref.norm().item(), 1e-30)
                errs.append(err)
            
            avg_err = np.mean(errs)
            print(f"  {avg_err:.2e}    ", end="")
        print()
    
    _core.set_rsvd_mode('full_svd')


# ═════════════════════════════════════════════════════════════════════════════
# BENCHMARK 6: Timing comparison
# ═════════════════════════════════════════════════════════════════════════════

def bench_timing():
    print("\n" + "=" * 80)
    print("BENCHMARK 6: Timing — forward + backward for each mode")
    print("  N×N matrix, k retained SVs, 100 repeats, median time")
    print("=" * 80)
    
    configs = [
        (100, 10, 0.30, "N=100 k=10"),
        (200, 20, 0.30, "N=200 k=20"),
        (400, 30, 0.30, "N=400 k=30"),
    ]
    
    modes = [
        ("full_svd",      'full_svd', 4, 0),
        ("neum L=1",      'neumann',  1, 0),
        ("neum L=2",      'neumann',  2, 0),
        ("neum L=4",      'neumann',  4, 0),
        ("aug k_aug=2k",  'augmented',4, None),
        ("none",          'none',     4, 0),
    ]
    
    n_rep = 50
    
    print(f"\n{'Config':<16s}", end="")
    for mlabel, _, _, _ in modes:
        print(f"  {mlabel:<12s}", end="")
    print()
    print("-" * (16 + 14 * len(modes)))
    
    for N, k, ratio, label in configs:
        A_np = make_matrix(N, k, ratio, seed=42)
        torch.manual_seed(999)
        C = torch.randn(N, N, dtype=torch.float64)
        
        print(f"{label:<16s}", end="")
        for mlabel, mode, n_terms, aug_extra in modes:
            ae = aug_extra if aug_extra is not None else k
            _core.set_rsvd_mode(mode, neumann_terms=n_terms, augment_extra=ae)
            
            times = []
            for _ in range(n_rep):
                A = A_np.clone().requires_grad_(True)
                t0 = time.perf_counter()
                U, S, V = _core.truncated_svd_propack(A, k, chi_extra=1, rel_cutoff=1e-12)
                loss = (U @ torch.diag(S) @ V.T - C).pow(2).sum()
                loss.backward()
                dt = time.perf_counter() - t0
                times.append(dt)
            
            med = np.median(times) * 1000
            print(f"  {med:8.2f}ms  ", end="")
        print()
    
    _core.set_rsvd_mode('full_svd')


# ═════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    bench_niter()
    bench_q_over()
    bench_neumann()
    bench_augmented()
    bench_end_to_end()
    bench_timing()
    print("\nDone.")
