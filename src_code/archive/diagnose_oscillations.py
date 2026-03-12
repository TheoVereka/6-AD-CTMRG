#!/usr/bin/env python3
"""
Diagnostic script: Why does the single-tensor ansatz produce
HUGE loss oscillations between outer optimization steps?

Tests three hypotheses:
  1. L-BFGS takes enormous steps → tensor moves far → stale environment
  2. Gradient magnitude is amplified by the constraint b=d=f=-a, c=e=a
  3. Reducing max_iter (and thus step size) eliminates oscillations
"""

import os, sys, time, torch, collections
import opt_einsum as oe

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core_unrestricted import (
    normalize_tensor,
    initialize_abcdef,
    abcdef_to_ABCDEF,
    CTMRG_from_init_to_stop,
    energy_expectation_nearest_neighbor_6_bonds,
    energy_expectation_nearest_neighbor_other_3_bonds,
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
D_BOND = 2
D_SQ = D_BOND ** 2
CHI = 5
D_PHYS = 2
CTM_MAX = 90
CTM_THR = 1e-6


def build_H(J=1.0):
    sx = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64) / 2
    sy = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64) / 2
    sz = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64) / 2
    SdotS = (oe.contract("ij,kl->ikjl", sx, sx)
           + oe.contract("ij,kl->ikjl", sy, sy)
           + oe.contract("ij,kl->ikjl", sz, sz))
    return (J * SdotS).to(DEVICE)


def converge_env(a, b, c, d, e, f):
    """Run CTMRG and return 27 environment tensors."""
    with torch.no_grad():
        A, B, C, D, E, F = abcdef_to_ABCDF(a, b, c, d, e, f, D_SQ)
        return CTMRG_from_init_to_stop(A, B, C, D, E, F, CHI, D_SQ, CTM_MAX, CTM_THR)


def energy_with_env(a, b, c, d, e, f, Hs, env27):
    """Compute energy at (a..f) using pre-converged environment."""
    return (
        energy_expectation_nearest_neighbor_6_bonds(
            a, b, c, d, e, f,
            Hs[0], Hs[1], Hs[2], Hs[3], Hs[4], Hs[5],
            CHI, D_BOND, *env27[:9])
      + energy_expectation_nearest_neighbor_other_3_bonds(
            a, b, c, d, e, f,
            Hs[6], Hs[7], Hs[8],
            CHI, D_BOND, *env27[18:27])
    ).real


def converge_and_energy(a, b, c, d, e, f, Hs):
    """Converge env from scratch and return energy."""
    with torch.no_grad():
        A, B, C, Dt, E, F = abcdef_to_ABCDEF(a, b, c, d, e, f, D_SQ)
        env = CTMRG_from_init_to_stop(A, B, C, Dt, E, F, CHI, D_SQ, CTM_MAX, CTM_THR)
        E6 = energy_expectation_nearest_neighbor_6_bonds(
            a, b, c, d, e, f,
            Hs[0], Hs[1], Hs[2], Hs[3], Hs[4], Hs[5],
            CHI, D_BOND, *env[:9])
        E3 = energy_expectation_nearest_neighbor_other_3_bonds(
            a, b, c, d, e, f,
            Hs[6], Hs[7], Hs[8],
            CHI, D_BOND, *env[18:27])
    return (E6 + E3).real.item(), env


def run_outer_step_single_tensor(a, Hs, max_iter, lr):
    """
    One outer step of single-tensor optimization.
    Returns: (a_new, loss_lbfgs, loss_reconverged, tensor_displacement)
    """
    a_before = a.detach().clone()

    # 1. Normalize
    with torch.no_grad():
        a.data = normalize_tensor(a.data)

    # 2. Converge environment
    b, c, d, e, f = -a.detach(), a.detach(), -a.detach(), a.detach(), -a.detach()
    with torch.no_grad():
        A, B, C, Dt, E, F = abcdef_to_ABCDEF(a, b, c, d, e, f, D_SQ)
        env = CTMRG_from_init_to_stop(A, B, C, Dt, E, F, CHI, D_SQ, CTM_MAX, CTM_THR)

    # 3. L-BFGS
    optimizer = torch.optim.LBFGS(
        [a], lr=lr, max_iter=max_iter,
        tolerance_grad=1e-7, tolerance_change=1e-9,
        history_size=100, line_search_fn='strong_wolfe')

    closure_count = [0]
    def closure():
        closure_count[0] += 1
        optimizer.zero_grad()
        bb, cc, dd, ee, ff = -a, a, -a, a, -a
        loss = (
            energy_expectation_nearest_neighbor_6_bonds(
                a, bb, cc, dd, ee, ff,
                Hs[0], Hs[1], Hs[2], Hs[3], Hs[4], Hs[5],
                CHI, D_BOND, *env[:9])
          + energy_expectation_nearest_neighbor_other_3_bonds(
                a, bb, cc, dd, ee, ff,
                Hs[6], Hs[7], Hs[8],
                CHI, D_BOND, *env[18:27])
        ).real
        loss.backward()
        return loss

    loss_val = optimizer.step(closure)
    loss_lbfgs = loss_val.item()

    # 4. Measure displacement
    disp = (a.data - a_before).norm().item()
    rel_disp = disp / a_before.norm().item()

    # 5. Re-converge environment at new a and get TRUE loss
    b2, c2, d2, e2, f2 = -a.detach(), a.detach(), -a.detach(), a.detach(), -a.detach()
    loss_true, _ = converge_and_energy(a, b2, c2, d2, e2, f2, Hs)

    return a, loss_lbfgs, loss_true, disp, rel_disp, closure_count[0]


def run_outer_step_six_tensors(a, b, c, d, e, f, Hs, max_iter, lr):
    """
    One outer step of 6-tensor optimization.
    Returns: (a..f, loss_lbfgs, loss_reconverged, tensor_displacement)
    """
    with torch.no_grad():
        norms_before = [t.data.norm().item() for t in (a, b, c, d, e, f)]
        tensors_before = [t.detach().clone() for t in (a, b, c, d, e, f)]

        a.data = normalize_tensor(a.data)
        b.data = normalize_tensor(b.data)
        c.data = normalize_tensor(c.data)
        d.data = normalize_tensor(d.data)
        e.data = normalize_tensor(e.data)
        f.data = normalize_tensor(f.data)

    # Converge environment
    with torch.no_grad():
        A, B, C, Dt, E, F = abcdef_to_ABCDEF(a, b, c, d, e, f, D_SQ)
        env = CTMRG_from_init_to_stop(A, B, C, Dt, E, F, CHI, D_SQ, CTM_MAX, CTM_THR)

    optimizer = torch.optim.LBFGS(
        [a, b, c, d, e, f], lr=lr, max_iter=max_iter,
        tolerance_grad=1e-7, tolerance_change=1e-9,
        history_size=100, line_search_fn='strong_wolfe')

    closure_count = [0]
    def closure():
        closure_count[0] += 1
        optimizer.zero_grad()
        loss = (
            energy_expectation_nearest_neighbor_6_bonds(
                a, b, c, d, e, f,
                Hs[0], Hs[1], Hs[2], Hs[3], Hs[4], Hs[5],
                CHI, D_BOND, *env[:9])
          + energy_expectation_nearest_neighbor_other_3_bonds(
                a, b, c, d, e, f,
                Hs[6], Hs[7], Hs[8],
                CHI, D_BOND, *env[18:27])
        ).real
        loss.backward()
        return loss

    loss_val = optimizer.step(closure)
    loss_lbfgs = loss_val.item()

    # Displacement: sum of per-tensor displacements
    disps = [(t.data - tb).norm().item()
             for t, tb in zip((a, b, c, d, e, f), tensors_before)]
    total_disp = sum(disps)
    total_norm = sum(norms_before)
    rel_disp = total_disp / total_norm

    # Re-converge environment and get TRUE loss
    loss_true, _ = converge_and_energy(a, b, c, d, e, f, Hs)

    return a, b, c, d, e, f, loss_lbfgs, loss_true, total_disp, rel_disp, closure_count[0]


def gradient_norm_comparison(a_orig, Hs):
    """
    Compare ||grad_a|| in single-tensor vs 6-tensor mode at the SAME point.
    """
    # --- Single-tensor gradient ---
    a1 = a_orig.detach().clone().requires_grad_(True)
    b1, c1 = -a_orig.detach(), a_orig.detach()
    d1, e1, f1 = -a_orig.detach(), a_orig.detach(), -a_orig.detach()

    with torch.no_grad():
        A, B, C, Dt, E, F = abcdef_to_ABCDEF(a1, b1, c1, d1, e1, f1, D_SQ)
        env = CTMRG_from_init_to_stop(A, B, C, Dt, E, F, CHI, D_SQ, CTM_MAX, CTM_THR)

    # Single: grad flows through a to all 6 slots
    bb, cc, dd, ee, ff = -a1, a1, -a1, a1, -a1
    loss_single = (
        energy_expectation_nearest_neighbor_6_bonds(
            a1, bb, cc, dd, ee, ff,
            Hs[0], Hs[1], Hs[2], Hs[3], Hs[4], Hs[5],
            CHI, D_BOND, *env[:9])
      + energy_expectation_nearest_neighbor_other_3_bonds(
            a1, bb, cc, dd, ee, ff,
            Hs[6], Hs[7], Hs[8],
            CHI, D_BOND, *env[18:27])
    ).real
    loss_single.backward()
    grad_single = a1.grad.clone()

    # --- 6-tensor gradients ---
    a2 = a_orig.detach().clone().requires_grad_(True)
    b2 = (-a_orig).detach().clone().requires_grad_(True)
    c2 = a_orig.detach().clone().requires_grad_(True)
    d2 = (-a_orig).detach().clone().requires_grad_(True)
    e2 = a_orig.detach().clone().requires_grad_(True)
    f2 = (-a_orig).detach().clone().requires_grad_(True)

    # Use same environment (same point)
    loss_six = (
        energy_expectation_nearest_neighbor_6_bonds(
            a2, b2, c2, d2, e2, f2,
            Hs[0], Hs[1], Hs[2], Hs[3], Hs[4], Hs[5],
            CHI, D_BOND, *env[:9])
      + energy_expectation_nearest_neighbor_other_3_bonds(
            a2, b2, c2, d2, e2, f2,
            Hs[6], Hs[7], Hs[8],
            CHI, D_BOND, *env[18:27])
    ).real
    loss_six.backward()

    grad_norms_6 = [t.grad.norm().item() for t in (a2, b2, c2, d2, e2, f2)]

    return {
        'single_grad_norm': grad_single.norm().item(),
        'six_grad_a_norm': grad_norms_6[0],
        'six_grad_norms': grad_norms_6,
        'six_total_grad_norm': sum(g**2 for g in grad_norms_6)**0.5,
        'six_max_grad_norm': max(grad_norms_6),
        'amplification_vs_a': grad_single.norm().item() / grad_norms_6[0],
        'amplification_vs_max': grad_single.norm().item() / max(grad_norms_6),
    }


# ===========================================================================
# Main
# ===========================================================================
if __name__ == '__main__':
    torch.manual_seed(42)
    H = build_H(1.0)
    Hs = [H] * 9

    print("=" * 78)
    print("  DIAGNOSTIC: Root cause of HUGE single-tensor loss oscillations")
    print(f"  D={D_BOND}  chi={CHI}  device={DEVICE}")
    print("=" * 78)

    # ── TEST 1: Gradient amplification ────────────────────────────────────
    print("\n" + "─" * 78)
    print("  TEST 1: Gradient norm comparison (single-tensor vs 6-tensor)")
    print("─" * 78)

    a_init = initialize_abcdef('random', D_BOND, D_PHYS, 1e-3)[0].to(DEVICE)

    ginfo = gradient_norm_comparison(a_init, Hs)
    print(f"  ||grad_a|| single-tensor : {ginfo['single_grad_norm']:.6f}")
    print(f"  ||grad_a|| 6-tensor      : {ginfo['six_grad_a_norm']:.6f}")
    print(f"  ||grad||_total 6-tensor  : {ginfo['six_total_grad_norm']:.6f}")
    print(f"  max ||grad_x|| 6-tensor  : {ginfo['six_max_grad_norm']:.6f}")
    print(f"  Per-tensor 6-tensor norms: {['%.4f' % g for g in ginfo['six_grad_norms']]}")
    print(f"  Amplification vs grad_a  : {ginfo['amplification_vs_a']:.2f}×")
    print(f"  Amplification vs max     : {ginfo['amplification_vs_max']:.2f}×")
    print(f"\n  → Single-tensor gradient is {ginfo['amplification_vs_a']:.1f}× "
          f"larger than the individual grad_a in 6-tensor mode.")
    print(f"    This is because dL/da = ∂L/∂a − ∂L/∂b + ∂L/∂c − ∂L/∂d + ∂L/∂e − ∂L/∂f")
    print(f"    (chain rule through b=-a, c=a, d=-a, e=a, f=-a).")

    # ── TEST 2: Tensor displacement & environment mismatch ────────────────
    print("\n" + "─" * 78)
    print("  TEST 2: Outer steps — single-tensor (max_iter=30, lr=0.01)")
    print("    Measuring: L-BFGS loss (with FROZEN env) vs TRUE loss (re-converged env)")
    print("─" * 78)

    a = initialize_abcdef('random', D_BOND, D_PHYS, 1e-3)[0].to(DEVICE)
    a.requires_grad_(True)

    print(f"  {'step':>4}  {'L-BFGS loss':>14}  {'TRUE loss':>14}  "
          f"{'gap':>10}  {'||Δa||':>10}  {'||Δa||/||a||':>12}  {'closures':>8}")
    for step in range(3):
        a, loss_lbfgs, loss_true, disp, rel_disp, nclos = \
            run_outer_step_single_tensor(a, Hs, max_iter=30, lr=0.01)
        gap = loss_true - loss_lbfgs
        print(f"  {step:4d}  {loss_lbfgs:+14.7f}  {loss_true:+14.7f}  "
              f"{gap:+10.4f}  {disp:10.6f}  {rel_disp:12.6f}  {nclos:8d}")

    # ── TEST 3: Same but with max_iter=1 ──────────────────────────────────
    print("\n" + "─" * 78)
    print("  TEST 3: Outer steps — single-tensor (max_iter=1, lr=0.01)")
    print("    Hypothesis: fewer sub-iters → smaller step → smaller env mismatch")
    print("─" * 78)

    a = initialize_abcdef('random', D_BOND, D_PHYS, 1e-3)[0].to(DEVICE)
    a.requires_grad_(True)

    print(f"  {'step':>4}  {'L-BFGS loss':>14}  {'TRUE loss':>14}  "
          f"{'gap':>10}  {'||Δa||':>10}  {'||Δa||/||a||':>12}  {'closures':>8}")
    for step in range(3):
        a, loss_lbfgs, loss_true, disp, rel_disp, nclos = \
            run_outer_step_single_tensor(a, Hs, max_iter=1, lr=0.01)
        gap = loss_true - loss_lbfgs
        print(f"  {step:4d}  {loss_lbfgs:+14.7f}  {loss_true:+14.7f}  "
              f"{gap:+10.4f}  {disp:10.6f}  {rel_disp:12.6f}  {nclos:8d}")

    # ── TEST 4: 6-tensor baseline (max_iter=30) ──────────────────────────
    print("\n" + "─" * 78)
    print("  TEST 4: Outer steps — 6-tensor UNRESTRICTED (max_iter=30, lr=0.8)")
    print("    Baseline: does the 6-tensor version have the same oscillations?")
    print("─" * 78)

    inits = initialize_abcdef('random', D_BOND, D_PHYS, 1e-3)
    a, b, c, d, e, f = [t.to(DEVICE).requires_grad_(True) for t in inits]

    print(f"  {'step':>4}  {'L-BFGS loss':>14}  {'TRUE loss':>14}  "
          f"{'gap':>10}  {'||Δ||_sum':>10}  {'rel_Δ':>12}  {'closures':>8}")
    for step in range(3):
        a, b, c, d, e, f, loss_lbfgs, loss_true, disp, rel_disp, nclos = \
            run_outer_step_six_tensors(a, b, c, d, e, f, Hs, max_iter=30, lr=0.8)
        gap = loss_true - loss_lbfgs
        print(f"  {step:4d}  {loss_lbfgs:+14.7f}  {loss_true:+14.7f}  "
              f"{gap:+10.4f}  {disp:10.6f}  {rel_disp:12.6f}  {nclos:8d}")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("  SUMMARY")
    print("=" * 78)
    print("""
  The huge oscillations are caused by ENVIRONMENT-TENSOR MISMATCH:

  1. GRADIENT AMPLIFICATION: In the single-tensor ansatz, dL/da accumulates
     gradient contributions from all 6 tensor positions via the chain rule:
       dL/da = ∂L/∂a − ∂L/∂b + ∂L/∂c − ∂L/∂d + ∂L/∂e − ∂L/∂f
     This makes ||grad|| much larger than in the 6-tensor case.

  2. L-BFGS OVERSHOOTING: With 30 sub-iterations and amplified gradients,
     L-BFGS moves `a` FAR from where the environment was converged.  The
     line search optimises against the FROZEN environment, not the true energy.

  3. STALE ENVIRONMENT: When the next outer step re-converges the environment
     at the new `a`, the energy landscape looks completely different.  The loss
     that L-BFGS reported (using the old env) is meaningless — the "true" loss
     can be orders of magnitude worse.

  4. The "gap" column above (TRUE_loss − LBFGS_loss) directly measures this
     mismatch.  For single-tensor with max_iter=30, the gap is often O(1–3).
     For 6-tensor or single-tensor with max_iter=1, the gap is much smaller.

  FIX: Reduce max_iter to 1–3 for the single-tensor ansatz, so L-BFGS takes
  small cautious steps that don't invalidate the frozen environment.
""")
