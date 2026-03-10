#!/usr/bin/env python3
"""
main_unrestricted.py
====================
Ground-state iPEPS optimisation of the **anisotropic nearest-neighbour
Heisenberg spin-1/2 model** on the 6-site honeycomb unit cell (A–F) via
automatic-differentiation CTMRG.

Hamiltonian
-----------
    H = Σ_{<i,j>} J_ij  (S_i · S_j)

with **nine independent** nearest-neighbour couplings::

    J_ed, J_ad, J_af, J_cf, J_cb, J_eb   (6 bonds covered by env-type 1)
    J_cd, J_ef, J_ab                       (3 bonds covered by env-type 3)

Strategy
--------
Outer loop  — grow ``D_bond`` (virtual bond dimension of the iPEPS tensor):
    Inner loop — grow ``chi`` (environment bond dimension of the CTMRG):

        1. Optimise site tensors a–f at (D_bond, chi) via L-BFGS.
        2. Check convergence in chi by re-evaluating with the *next* chi.
        3. Once chi-converged, store the ground-state energy for this D_bond.

    Warm-start the next D_bond from the previous solution (padded with noise).

Constraint enforced everywhere:  D_bond² < chi ≤ D_bond⁴.

Run
---
    cd src_code  &&  python scripts/main_unrestricted.py

All tuneable parameters are collected in the ``== CONFIGURATION ==`` block below.
"""

import sys, os, time, datetime, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import opt_einsum as oe
import numpy as np
import matplotlib
matplotlib.use('Agg')          # headless backend — works without X11
import matplotlib.pyplot as plt

from core_unrestricted import (
    normalize_tensor,
    initialize_abcdef,
    abcdef_to_ABCDEF,
    CTMRG_from_init_to_stop,
    energy_expectation_nearest_neighbor_6_bonds,
    energy_expectation_nearest_neighbor_other_3_bonds,
    optmization_iPEPS,
    check_optimized_iPEPS,
)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                       == CONFIGURATION ==                               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ── Physical model: anisotropic Heisenberg couplings ──────────────────────
#    H_ij = J_ij * (S^x_i S^x_j + S^y_i S^y_j + S^z_i S^z_j)
#    Positive J → antiferromagnetic;  Negative J → ferromagnetic.
J = {
    'ed': 1.0,    # bond E-D
    'ad': 1.0,    # bond A-D
    'af': 1.0,    # bond A-F  (≡ F-A)
    'cf': 1.0,    # bond C-F
    'cb': 1.0,    # bond C-B
    'eb': 1.0,    # bond E-B
    'cd': 1.0,    # bond C-D
    'ef': 1.0,    # bond E-F
    'ab': 1.0,    # bond A-B
}

d_PHYS = 2   # spin-1/2

# ── iPEPS bond dimensions to sweep ───────────────────────────────────────
D_bond_list = [2, 3]
# For each D_bond the chi sweep is built automatically:
#     chi_min = D_bond² + 1   (strictly greater than D²)
#     chi_max = D_bond⁴
# with intermediate steps spaced by chi_step.
chi_step = 4   # spacing between successive chi values in the inner sweep

# ── CTMRG parameters ─────────────────────────────────────────────────────
a_third_max_steps_CTMRG = 80   # max CTMRG (1→2→3→1) cycles per env update
CTM_env_conv_threshold  = 1e-7 # RMS convergence threshold for environments

# ── L-BFGS optimisation parameters ───────────────────────────────────────
max_opt_steps       = 300       # outer optimisation iterations per (D, chi)
lbfgs_max_iter      = 20        # L-BFGS sub-iterations per outer step
lbfgs_lr            = 1.0       # L-BFGS initial step size
lbfgs_history       = 100       # L-BFGS curvature history
opt_conv_threshold  = 1e-7      # stop when |Δloss| < this
opt_tolerance_grad  = 1e-7      # L-BFGS gradient tolerance
opt_tolerance_change = 1e-9     # L-BFGS function-change tolerance

# ── Chi-convergence check ────────────────────────────────────────────────
chi_conv_threshold = 1e-5       # |E(chi_next) − E(chi)| < this → chi-converged

# ── Warm-start noise for D_bond upgrade ──────────────────────────────────
warmstart_noise_scale = 1e-3

# ── I/O ──────────────────────────────────────────────────────────────────
output_dir  = os.path.join(os.path.dirname(__file__), '..', 'log')
save_tensors = True   # save optimised a–f tensors to .pt files


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                         HELPER FUNCTIONS                                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def build_heisenberg_H(J_coupling: float, d: int = 2) -> torch.Tensor:
    """Return J * (S·S) as a (d,d,d,d) tensor for a nearest-neighbour bond.

    Convention: bra1, bra2, ket1, ket2  →  ``H[i,j,k,l]``.
    """
    sx = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64) / 2
    sy = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64) / 2
    sz = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64) / 2
    SdotS = (oe.contract("ij,kl->ikjl", sx, sx)
           + oe.contract("ij,kl->ikjl", sy, sy)
           + oe.contract("ij,kl->ikjl", sz, sz))
    return J_coupling * SdotS


def chi_sweep_for_D(D_bond: int, step: int = 4) -> list[int]:
    """Return the list of chi values to sweep for a given D_bond.

    Constraint: D_bond² < chi ≤ D_bond⁴.
    """
    D_sq  = D_bond ** 2
    D4    = D_bond ** 4
    chi_min = D_sq + 1                          # strictly > D²
    chi_max = D4
    if chi_min > chi_max:
        # Only possible if D_bond == 1 where D²=1, D⁴=1 → empty range.
        return [chi_max]
    chis = list(range(chi_min, chi_max + 1, step))
    if chis[-1] != chi_max:
        chis.append(chi_max)
    return chis


def pad_tensor_for_new_D(tensor: torch.Tensor,
                         old_D: int, new_D: int, d: int,
                         noise: float) -> torch.Tensor:
    """Embed an (old_D, old_D, old_D, d) tensor into (new_D, new_D, new_D, d)
    by placing the old values in the top-left corner and filling the rest with
    small Gaussian noise.  Returned tensor is normalised."""
    out = noise * torch.randn(new_D, new_D, new_D, d, dtype=torch.complex64)
    s = old_D
    out[:s, :s, :s, :] = tensor[:s, :s, :s, :]
    return normalize_tensor(out)


def evaluate_energy(a, b, c, d, e, f,
                    Hed, Had, Haf, Hcf, Hcb, Heb, Hcd, Hef, Hab,
                    chi: int, D_bond: int,
                    a_third_max_steps: int,
                    ctm_threshold: float) -> float:
    """Converge the environment at the given chi and return the total energy
    (6-bond + 3-bond) as a Python float."""
    D_squared = D_bond ** 2
    with torch.no_grad():
        A, B, C, D_t, E, F_t = abcdef_to_ABCDEF(a, b, c, d, e, f, D_squared)
        all27 = CTMRG_from_init_to_stop(
            A, B, C, D_t, E, F_t, chi, D_squared,
            a_third_max_steps, ctm_threshold)
        env1 = all27[:9]
        env3 = all27[18:27]
        E6 = energy_expectation_nearest_neighbor_6_bonds(
            a, b, c, d, e, f,
            Hed, Had, Haf, Hcf, Hcb, Heb,
            chi, D_bond,
            *env1)
        E3 = energy_expectation_nearest_neighbor_other_3_bonds(
            a, b, c, d, e, f,
            Hcd, Hef, Hab,
            chi, D_bond,
            *env3)
        return (E6 + E3).real.item()


def timestamp() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                             MAIN LOOP                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def main():
    os.makedirs(output_dir, exist_ok=True)

    # ── Build the 9 Hamiltonian bond tensors ─────────────────────────────
    Hed = build_heisenberg_H(J['ed'], d_PHYS)
    Had = build_heisenberg_H(J['ad'], d_PHYS)
    Haf = build_heisenberg_H(J['af'], d_PHYS)
    Hcf = build_heisenberg_H(J['cf'], d_PHYS)
    Hcb = build_heisenberg_H(J['cb'], d_PHYS)
    Heb = build_heisenberg_H(J['eb'], d_PHYS)
    Hcd = build_heisenberg_H(J['cd'], d_PHYS)
    Hef = build_heisenberg_H(J['ef'], d_PHYS)
    Hab = build_heisenberg_H(J['ab'], d_PHYS)

    # ── Storage for results ──────────────────────────────────────────────
    # results[D_bond] = list of (chi, energy) tuples
    results: dict[int, list[tuple[int, float]]] = {}
    # Keep the best (a–f) tensors from the previous D_bond for warm-start
    best_abcdef: tuple[torch.Tensor, ...] | None = None
    prev_D: int = 0

    print("=" * 72)
    print(f" Anisotropic Heisenberg iPEPS — AD-CTMRG optimisation")
    print(f" Couplings: {J}")
    print(f" D_bond sweep: {D_bond_list}")
    print(f" Started at {timestamp()}")
    print("=" * 72)

    for D_bond in D_bond_list:
        D_sq = D_bond ** 2
        chis = chi_sweep_for_D(D_bond, chi_step)
        results[D_bond] = []

        print(f"\n{'─' * 72}")
        print(f" D_bond = {D_bond}   (D² = {D_sq},  D⁴ = {D_bond**4})")
        print(f" chi sweep: {chis}")
        print(f"{'─' * 72}")

        # ── Warm-start from previous D_bond ──────────────────────────────
        if best_abcdef is not None and prev_D < D_bond:
            print(f"  Warm-starting from D_bond={prev_D} tensors "
                  f"(padding {prev_D}→{D_bond}, noise={warmstart_noise_scale})")
            warm_abcdef = tuple(
                pad_tensor_for_new_D(t, prev_D, D_bond, d_PHYS,
                                     warmstart_noise_scale)
                for t in best_abcdef
            )
        else:
            warm_abcdef = None          # will use random init

        chi_converged = False
        prev_chi_energy: float | None = None

        for chi_idx, chi in enumerate(chis):
            t0 = time.perf_counter()

            print(f"\n  ┌── chi = {chi}  (D_bond={D_bond}, "
                  f"{chi_idx+1}/{len(chis)})  [{timestamp()}]")

            # ── Optimise ─────────────────────────────────────────────────
            # optmization_iPEPS returns 7 values: (a, b, c, d, e, f, loss_item)
            # Use warm_abcdef for first chi step if available (D_bond warm-start),
            # or best_abcdef from the previous chi within the same D_bond.
            init_tensors = warm_abcdef if (chi_idx == 0 and warm_abcdef is not None) else best_abcdef
            a, b, c, d_t, e, f, loss = optmization_iPEPS(
                Hed, Had, Haf, Hcf, Hcb, Heb, Hcd, Hef, Hab,
                opt_conv_threshold=opt_conv_threshold,
                chi=chi, D_bond=D_bond, d_PHYS=d_PHYS,
                a_third_max_steps_CTMRG=a_third_max_steps_CTMRG,
                CTM_env_conv_threshold=CTM_env_conv_threshold,
                a2f_initialize_way='random',
                a2f_noise_scale=1e-3,
                max_opt_steps=max_opt_steps,
                lbfgs_max_iter=lbfgs_max_iter,
                lbfgs_lr=lbfgs_lr,
                lbfgs_history=lbfgs_history,
                opt_tolerance_grad=opt_tolerance_grad,
                opt_tolerance_change=opt_tolerance_change,
                init_abcdef=init_tensors,
            )

            # ── Re-evaluate energy cleanly at this chi ───────────────────
            energy = evaluate_energy(
                a, b, c, d_t, e, f,
                Hed, Had, Haf, Hcf, Hcb, Heb, Hcd, Hef, Hab,
                chi, D_bond,
                a_third_max_steps_CTMRG, CTM_env_conv_threshold)

            elapsed = time.perf_counter() - t0
            results[D_bond].append((chi, energy))

            print(f"  │  E(D={D_bond}, χ={chi}) = {energy:+.10f}"
                  f"   (loss={loss:+.10f},  {elapsed:.1f}s)")

            # ── Chi convergence check ────────────────────────────────────
            if prev_chi_energy is not None:
                delta_chi = abs(energy - prev_chi_energy)
                print(f"  │  ΔE(χ) = {delta_chi:.3e}"
                      f"   (threshold = {chi_conv_threshold:.1e})")
                if delta_chi < chi_conv_threshold:
                    print(f"  └── χ-converged at χ={chi} for D_bond={D_bond}.")
                    chi_converged = True
            else:
                print(f"  │  (first chi — no ΔE yet)")

            prev_chi_energy = energy

            # Save best tensors for this D_bond
            best_abcdef = tuple(
                t.detach().clone() for t in (a, b, c, d_t, e, f)
            )

            if chi_converged:
                break

        # ── End of chi sweep ────────────────────────────────────────────
        if not chi_converged:
            print(f"\n  ⚠  χ-sweep exhausted for D_bond={D_bond} "
                  f"without reaching ΔE < {chi_conv_threshold:.1e}.")

        prev_D = D_bond

        # ── Save tensors ────────────────────────────────────────────────
        if save_tensors and best_abcdef is not None:
            fname = os.path.join(output_dir, f"abcdef_D{D_bond}.pt")
            torch.save({
                'a': best_abcdef[0], 'b': best_abcdef[1],
                'c': best_abcdef[2], 'd': best_abcdef[3],
                'e': best_abcdef[4], 'f': best_abcdef[5],
                'D_bond': D_bond,
                'chi_sweep': chis,
                'energies': results[D_bond],
                'J': J,
            }, fname)
            print(f"  Saved tensors → {fname}")

    # ╔═══════════════════════════════════════════════════════════════════╗
    # ║                     RESULTS SUMMARY                             ║
    # ╚═══════════════════════════════════════════════════════════════════╝

    print("\n" + "=" * 72)
    print("  RESULTS SUMMARY")
    print("=" * 72)
    print(f"  {'D_bond':>6}  {'chi':>6}  {'Energy':>18}")
    print(f"  {'------':>6}  {'---':>6}  {'------------------':>18}")
    for D_bond in D_bond_list:
        for chi, energy in results.get(D_bond, []):
            print(f"  {D_bond:>6}  {chi:>6}  {energy:>+18.10f}")

    # ── Save results JSON ────────────────────────────────────────────────
    results_path = os.path.join(output_dir, "results.json")
    json_results = {
        'couplings': J,
        'd_PHYS': d_PHYS,
        'D_bond_list': D_bond_list,
        'data': {str(D): entries for D, entries in results.items()},
        'timestamp': timestamp(),
    }
    with open(results_path, 'w') as fp:
        json.dump(json_results, fp, indent=2)
    print(f"\n  Results saved → {results_path}")

    # ── Plot: Energy vs chi for each D_bond ──────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    for D_bond in D_bond_list:
        data = results.get(D_bond, [])
        if not data:
            continue
        chis_plot  = [c for c, _ in data]
        energies   = [e for _, e in data]
        ax.plot(chis_plot, energies, 'o-', label=f'D = {D_bond}')

    ax.set_xlabel(r'Environment bond dimension $\chi$')
    ax.set_ylabel(r'Ground-state energy $E_0$')
    ax.set_title('iPEPS ground-state energy — anisotropic Heisenberg')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig_path = os.path.join(output_dir, "energy_vs_chi.pdf")
    fig.savefig(fig_path, bbox_inches='tight')
    print(f"  Figure saved → {fig_path}")
    plt.close(fig)

    # ── Plot: convergence in D_bond (best chi for each D) ────────────────
    if len(D_bond_list) > 1:
        fig2, ax2 = plt.subplots(figsize=(7, 4.5))
        D_vals = []
        E_best = []
        for D_bond in D_bond_list:
            data = results.get(D_bond, [])
            if data:
                D_vals.append(D_bond)
                E_best.append(data[-1][1])   # last chi = best-converged
        ax2.plot(D_vals, E_best, 's-', color='tab:red', markersize=8)
        ax2.set_xlabel(r'Bond dimension $D$')
        ax2.set_ylabel(r'Best energy $E_0$')
        ax2.set_title('iPEPS energy vs bond dimension')
        ax2.grid(True, alpha=0.3)
        fig2_path = os.path.join(output_dir, "energy_vs_D.pdf")
        fig2.savefig(fig2_path, bbox_inches='tight')
        print(f"  Figure saved → {fig2_path}")
        plt.close(fig2)

    print(f"\n  Finished at {timestamp()}")
    print("=" * 72)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                           GPU MIGRATION NOTES                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
#  To run the SAME workflow on GPU, only a handful of changes are needed.
#  The core_unrestricted.py code is already PyTorch-native and device-agnostic
#  for the tensor operations themselves.  The obstacles encountered during
#  testing (and their GPU-specific resolutions) are listed below.
#
#  ┌─────────────────────────────────────────────────────────────────────────┐
#  │  1.  DEVICE PLACEMENT                                                  │
#  │      • Set a global device:                                            │
#  │          device = torch.device('cuda' if torch.cuda.is_available()     │
#  │                                else 'cpu')                             │
#  │      • All tensors created by initialize_abcdef / build_heisenberg_H   │
#  │        must be placed on `device`.  Either:                            │
#  │          - Call  tensor = tensor.to(device)  after creation, or        │
#  │          - Modify the functions to accept a `device=` kwarg.           │
#  │      • Intermediate tensors from opt_einsum and torch.linalg.svd       │
#  │        automatically inherit the device from their inputs — no extra   │
#  │        work is needed there.                                           │
#  ├─────────────────────────────────────────────────────────────────────────┤
#  │  2.  torch.linalg.svd  driver                                         │
#  │      • On CPU the default LAPACK driver is used (we removed            │
#  │        driver='gesvd' during testing because it's CUDA-only).          │
#  │      • On GPU you CAN pass  driver='gesvd'  for better numerical      │
#  │        stability with complex64.  Re-enable it conditionally:          │
#  │          svd_kw = {'driver': 'gesvd'} if tensor.is_cuda else {}       │
#  │          U, S, Vh = torch.linalg.svd(M, **svd_kw)                     │
#  ├─────────────────────────────────────────────────────────────────────────┤
#  │  3.  opt_einsum backend                                                │
#  │      • backend='torch' works identically on CPU and GPU — no change    │
#  │        needed.  opt_einsum dispatches to torch.einsum under the hood,  │
#  │        which is device-agnostic.                                       │
#  ├─────────────────────────────────────────────────────────────────────────┤
#  │  4.  .conj() vs .conjugate()                                          │
#  │      • .conj() (the one we use) works on both CPU and GPU in all       │
#  │        PyTorch >= 1.9.  No change needed.                              │
#  ├─────────────────────────────────────────────────────────────────────────┤
#  │  5.  .view() vs .reshape()                                             │
#  │      • We already switched all .view() calls to .reshape() to handle   │
#  │        non-contiguous memory from einsum.  .reshape() is safe on GPU   │
#  │        too — no change.                                                │
#  ├─────────────────────────────────────────────────────────────────────────┤
#  │  6.  dtype: complex64 on GPU                                           │
#  │      • CUDA fully supports complex64 since PyTorch 1.9.               │
#  │      • For performance, consider complex128 → complex64 if memory is   │
#  │        tight, or use torch.cuda.amp if your GPU supports TF32.         │
#  ├─────────────────────────────────────────────────────────────────────────┤
#  │  7.  Memory management                                                 │
#  │      • Large chi (e.g. 81 for D=3) creates matrices of size            │
#  │        (chi·D², chi·D²).  For D=3, chi=81: that's 729×729 complex64   │
#  │        ≈ 4 MB per matrix — tiny.  But SVD workspace can be large.      │
#  │      • For D≥4 with chi up to 256, watch GPU VRAM.  Add:              │
#  │          torch.cuda.empty_cache()  between outer optimisation steps.   │
#  ├─────────────────────────────────────────────────────────────────────────┤
#  │  8.  PRACTICAL RECIPE (minimal-change migration)                       │
#  │                                                                        │
#  │      Add this near the top of main_unrestricted.py:                    │
#  │                                                                        │
#  │          DEVICE = torch.device('cuda' if torch.cuda.is_available()     │
#  │                                else 'cpu')                             │
#  │                                                                        │
#  │      Then wrap build_heisenberg_H:                                     │
#  │          def build_heisenberg_H(J, d=2):                               │
#  │              ...                                                       │
#  │              return (J * SdotS).to(DEVICE)                             │
#  │                                                                        │
#  │      And patch initialize_abcdef (or add a wrapper):                   │
#  │          a, b, c, d, e, f = initialize_abcdef(...)                     │
#  │          a, b, c, d, e, f = [t.to(DEVICE) for t in (a,b,c,d,e,f)]     │
#  │                                                                        │
#  │      Everything else (CTMRG, energy, L-BFGS) follows the device of    │
#  │      its input tensors automatically.                                  │
#  │                                                                        │
#  │      Optionally re-enable SVD driver:                                  │
#  │          In core_unrestricted.py, change every torch.linalg.svd(M)     │
#  │          to torch.linalg.svd(M, driver='gesvd') — or add a flag.      │
#  └─────────────────────────────────────────────────────────────────────────┘
#
#  Summary of changes needed for GPU:
#     File                        Change
#     ─────────────────────────── ──────────────────────────────────────────
#     main_unrestricted.py        • Set DEVICE; .to(DEVICE) on H tensors
#                                   and on a–f after initialize_abcdef.
#     core_unrestricted.py        • (Optional) add device= kwarg to
#                                   initialize_abcdef so tensors are born
#                                   on GPU.
#                                 • (Optional) re-add driver='gesvd' to
#                                   torch.linalg.svd when input.is_cuda.
#     Nothing else changes.       All other operations are device-agnostic.
#


if __name__ == '__main__':
    main()


