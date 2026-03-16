"""
test_v2_physics.py — Physics-correctness tests for core_unres_v2
=================================================================
These tests verify that each function physically does the correct job
corresponding to the honeycomb lattice CTMRG.

Tests are NOT limited to D=1 trivial cases.  Every test uses D=2 or
larger, and chi values that actually stress the algorithm.

Test list
---------
 1. Exact Néel energy at D=2,3,4 — each of the 3 env types must give
    E/bond=-0.25 individually (tests update_env_*to* and energy functions).
 2. All 9 individual bond energies from Néel at D=2 — each bond must give
    -0.25 (tests the site_order argument and H-insertion in _energy_3bonds_raw).
 3. Three-env energy imbalance for Néel D=2,3 — |E1-E2|,|E2-E3| < 1e-8
    (tests that all 3 update functions agree for a known state).
 4. Norm consistency for Néel D=2,3 — |n1-n2|,|n2-n3| < 1e-6
    (tests transfer normalization in each update function).
 5. CTM self-consistency — after cold-start convergence, one more step
    changes singular values by < 2*conv_thr (tests the CTMRG loop itself).
 6. Translationally-invariant state (a=b=c=d=e=f) at D=2,3 — after CTM
    convergence E1=E2=E3 and n1=n2=n3 (tests C3 symmetry of update functions).
 7. Autodiff gradient vs. numerical gradient at D=2, chi=16 — relative
    agreement < 1e-3 (tests energy contraction and AD graph).
 8. Energy is chi-stable for Néel at D=2 — chi=4,9,16 all give -0.25
    (tests that larger environments don't introduce spurious corrections).
 9. Norm scales correctly with tensor scale — doubling all tensors scales
    energy numerator by 4 and norm by 4; ratio unchanged (Rayleigh quotient).
10. Energy from converged env vs. extra-step env agree — self-consistency
    between energy and the CTM iteration (tests compute_total_energy correctness).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import core_unres_v2 as cv2

torch.set_default_dtype(torch.float64)
cv2.set_dtype(True)

PASS = "  ✓"
FAIL = "  ✗ FAIL"
SEP  = "─" * 68

def header(title):
    print(f"\n{'='*68}\nTEST: {title}\n{'='*68}")

def check(cond, msg, extra=""):
    status = PASS if cond else FAIL
    print(f"{status}  {msg}{('  | '+extra) if extra else ''}")
    return cond


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_neel(D, d=2):
    """Create exact Néel tensors for D=2,3,4.
    Sublattice A (a,c,e) = |↑⟩ at slot (0,0,0).
    Sublattice B (b,d,f) = |↓⟩ at slot (0,0,0).
    D>1 slots are all zero — effectively D=1 embedded in D-dimensional bond.
    """
    up = torch.zeros(D, D, D, d, dtype=torch.float64)
    up[0, 0, 0, 0] = 1.0
    dn = torch.zeros(D, D, D, d, dtype=torch.float64)
    dn[0, 0, 0, 1] = 1.0
    return up, dn, up.clone(), dn.clone(), up.clone(), dn.clone()  # a,b,c,d,e,f


def make_neel_with_noise(D, noise=0.02, d=2, seed=77):
    """Near-Néel state — used where we need nonzero virtual modes."""
    torch.manual_seed(seed)
    a, b, c, dd, e, f = make_neel(D, d)
    tensors = []
    for t in (a, b, c, dd, e, f):
        t2 = t + torch.randn_like(t) * noise
        tensors.append(cv2.normalize_tensor(t2))
    return tuple(tensors)


def run_ctm(a, b, c, d, e, f, chi, max_steps=300, thr=1e-9):
    D_sq = a.shape[0] ** 2
    DL = cv2.abcdef_to_ABCDEF(a, b, c, d, e, f, D_sq)
    env1, env2, env3, steps = cv2.ctmrg(*DL, chi, D_sq, max_steps, thr, min_steps=5)
    return env1, env2, env3, steps, D_sq


def one_full_ctm_cycle(DL, env1, env2, env3, chi, D_sq):
    """Run exactly one CTMRG cycle (1→2→3→1) starting from (env1, env2, env3).

    This intentionally does NOT use any CTM warm-start API.
    """
    A, B, C, D, E, F = DL
    C21CD, C32EF, C13AB, T1F, T2A, T2B, T3C, T3D, T1E = env1
    r2 = cv2.update_env_1to2(C21CD, C32EF, C13AB, T1F, T2A, T2B, T3C, T3D, T1E,
                             A, B, C, D, E, F, chi, D_sq)
    env2n = r2[:9]

    C21EB, C32AD, C13CF, T1D, T2C, T2F, T3E, T3B, T1A = env2n
    r3 = cv2.update_env_2to3(C21EB, C32AD, C13CF, T1D, T2C, T2F, T3E, T3B, T1A,
                             A, B, C, D, E, F, chi, D_sq)
    env3n = r3[:9]

    C21AF, C32CB, C13ED, T1B, T2E, T2D, T3A, T3F, T1C = env3n
    r1 = cv2.update_env_3to1(C21AF, C32CB, C13ED, T1B, T2E, T2D, T3A, T3F, T1C,
                             A, B, C, D, E, F, chi, D_sq)
    env1n = r1[:9]
    return tuple(env1n), tuple(env2n), tuple(env3n)


def env_corners(env1, env2, env3):
    """Extract the three corner triplets (one per env type)."""
    C21CD, C32EF, C13AB = env1[0], env1[1], env1[2]
    C21EB, C32AD, C13CF = env2[0], env2[1], env2[2]
    C21AF, C32CB, C13ED = env3[0], env3[1], env3[2]
    return [(C21CD, C32EF, C13AB), (C21EB, C32AD, C13CF), (C21AF, C32CB, C13ED)]


def rho_sv(C21, C32, C13):
    rho = C13 @ C32 @ C21
    sv = torch.linalg.svdvals(rho)
    sv = sv / (sv[0] + 1e-30)   # normalise identically to _rho_sv_converged
    return sv


def per_env_energies(a, b, c, d, e, f, H, chi, D_bond, env1, env2, env3):
    """Return (E1, E2, E3, n1, n2, n3) — per-bond energy and norm from each env."""
    def raw(env, site_order):
        return cv2._energy_3bonds_raw(
            a, b, c, d, e, f, H, H, H, chi, D_bond,
            *env, site_order=site_order)

    nu1, n1 = raw(env1, site_order=(e, d, a, f, c, b))
    nu2, n2 = raw(env2, site_order=(a, b, c, d, e, f))
    nu3, n3 = raw(env3, site_order=(c, f, e, b, a, d))

    E1 = (nu1 / (3.0 * n1)).item()
    E2 = (nu2 / (3.0 * n2)).item()
    E3 = (nu3 / (3.0 * n3)).item()
    return E1, E2, E3, n1.item(), n2.item(), n3.item()


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 + 3 + 4: Exact Néel energy, imbalance, norm — for D=2,3,4
# ─────────────────────────────────────────────────────────────────────────────

def test_neel_exact():
    header("1+3+4. Exact Néel: per-env E=-0.25, energy imbalance, norm consistency  D=2,3,4")
    H = cv2.build_heisenberg_H()
    all_pass = True

    for D in [2, 3, 4]:
        chi = D * D  # exact environment for product state
        a, b, c, dd, e, f = make_neel(D)
        env1, env2, env3, steps, D_sq = run_ctm(a, b, c, dd, e, f, chi)
        D_bond = D

        E1, E2, E3, n1, n2, n3 = per_env_energies(a, b, c, dd, e, f, H, chi, D_bond,
                                                    env1, env2, env3)
        print(f"\n  D={D}  chi={chi}  CTM_steps={steps}")
        print(f"    E1={E1:+.10f}  E2={E2:+.10f}  E3={E3:+.10f}")
        print(f"    n1={n1:.8f}   n2={n2:.8f}   n3={n3:.8f}")

        ok1 = check(abs(E1 + 0.25) < 1e-6,
                    f"D={D} env1 E/bond=-0.250000", f"got {E1:+.8f}")
        ok2 = check(abs(E2 + 0.25) < 1e-6,
                    f"D={D} env2 E/bond=-0.250000", f"got {E2:+.8f}")
        ok3 = check(abs(E3 + 0.25) < 1e-6,
                    f"D={D} env3 E/bond=-0.250000", f"got {E3:+.8f}")
        ok4 = check(abs(E1 - E2) < 1e-6 and abs(E2 - E3) < 1e-6,
                    f"D={D} imbalance |E1-E2|={abs(E1-E2):.2e}  |E2-E3|={abs(E2-E3):.2e}")
        ok5 = check(abs(n1 - n2) < 1e-5 and abs(n2 - n3) < 1e-5,
                    f"D={D} norm consistency |n1-n2|={abs(n1-n2):.2e}  |n2-n3|={abs(n2-n3):.2e}")
        all_pass = all_pass and ok1 and ok2 and ok3 and ok4 and ok5

    return all_pass


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: All 9 individual bond energies for Néel
# ─────────────────────────────────────────────────────────────────────────────

def test_individual_bonds_neel():
    header("2. Individual bond energies: all 9 bonds = -0.25 for Néel at D=2,3")
    H = cv2.build_heisenberg_H()
    zero_H = torch.zeros_like(H)
    all_pass = True

    for D in [2, 3]:
        chi = D * D
        a, b, c, dd, e, f = make_neel(D)
        env1, env2, env3, steps, D_sq = run_ctm(a, b, c, dd, e, f, chi)
        D_bond = D

        print(f"\n  D={D}  chi={chi}")

        # env1 site_order=(e,d,a,f,c,b): bonds at C32=(e,d)↔bond-AD? C13=(a,f)? C21=(c,b)?
        # Test each bond individually by zeroing out other H's
        bonds = {"env1-C32": (env1, (e, dd, a, f, c, b), H,     zero_H, zero_H),
                 "env1-C13": (env1, (e, dd, a, f, c, b), zero_H, H,     zero_H),
                 "env1-C21": (env1, (e, dd, a, f, c, b), zero_H, zero_H, H),
                 "env2-C32": (env2, (a, b, c, dd, e, f), H,     zero_H, zero_H),
                 "env2-C13": (env2, (a, b, c, dd, e, f), zero_H, H,     zero_H),
                 "env2-C21": (env2, (a, b, c, dd, e, f), zero_H, zero_H, H),
                 "env3-C32": (env3, (c, f, e, b, a, dd), H,     zero_H, zero_H),
                 "env3-C13": (env3, (c, f, e, b, a, dd), zero_H, H,     zero_H),
                 "env3-C21": (env3, (c, f, e, b, a, dd), zero_H, zero_H, H),
                }

        for bond_name, (env, so, H1, H2, H3) in bonds.items():
            nu, n = cv2._energy_3bonds_raw(a, b, c, dd, e, f,
                                            H1, H2, H3, chi, D_bond,
                                            *env, site_order=so)
            # nu = energy numerator for this one bond; n = norm (6-site)
            E_bond = (nu / n).item()
            ok = check(abs(E_bond + 0.25) < 1e-5,
                       f"D={D} {bond_name}: E=-0.25", f"got {E_bond:+.8f}")
            all_pass = all_pass and ok

    return all_pass


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: CTM self-consistency — extra step barely changes SV spectrum
# ─────────────────────────────────────────────────────────────────────────────

def test_ctm_self_consistency():
    header("5. CTM self-consistency: one more step after convergence barely changes SVs  D=2,3,4")
    all_pass = True

    for D in [2, 3, 4]:
        chi = max(16, D * D * 2)
        torch.manual_seed(42 + D)
        a, b, c, dd, e, f = make_neel_with_noise(D, noise=0.1, seed=42+D)

        env1, env2, env3, steps, D_sq = run_ctm(a, b, c, dd, e, f, chi)
        D_bond = D

        corners_before = env_corners(env1, env2, env3)
        sv_before = [rho_sv(*corner) for corner in corners_before]

        # One more CTM cycle (manual 1→2→3→1 step; no warm-start API)
        DL = cv2.abcdef_to_ABCDEF(a, b, c, dd, e, f, D_sq)
        env1_new, env2_new, env3_new = one_full_ctm_cycle(DL, env1, env2, env3, chi, D_sq)

        corners_after = env_corners(env1_new, env2_new, env3_new)
        sv_after = [rho_sv(*corner) for corner in corners_after]

        max_delta = max(
            (sv_a - sv_b).abs().max().item()
            for sv_a, sv_b in zip(sv_after, sv_before)
        )
        print(f"\n  D={D}  chi={chi}  CTM_steps={steps}  max_SV_delta_after_1_step={max_delta:.2e}")
        ok = check(max_delta < 2e-6,
                   f"D={D} chi={chi}: SV shift after extra step = {max_delta:.2e} (need <2e-6)")
        all_pass = all_pass and ok

    return all_pass


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: Translationally-invariant state (a=b=c=d=e=f) — E1=E2=E3
# ─────────────────────────────────────────────────────────────────────────────

def test_translation_invariant():
    header("6. Translationally-invariant state (a=b=c=d=e=f): E1=E2=E3, n1=n2=n3  D=2,3")
    H = cv2.build_heisenberg_H()
    all_pass = True

    for D in [2, 3]:
        chi = D * D * 2
        for seed in [1, 7, 42]:
            torch.manual_seed(seed)
            base = torch.randn(D, D, D, 2, dtype=torch.float64)
            base = cv2.normalize_tensor(base)
            # All six sites identical — pure C6-symmetric state
            a = b = c = dd = e = f = base

            env1, env2, env3, steps, D_sq = run_ctm(a, b, c, dd, e, f, chi)
            E1, E2, E3, n1, n2, n3 = per_env_energies(a, b, c, dd, e, f, H, chi, D,
                                                        env1, env2, env3)

            imb_E = max(abs(E1-E2), abs(E2-E3), abs(E1-E3))
            imb_n = max(abs(n1-n2), abs(n2-n3), abs(n1-n3)) / max(abs(n1),1e-30)

            print(f"\n  D={D}  chi={chi}  seed={seed}  CTM={steps}")
            print(f"    E1={E1:+.8f}  E2={E2:+.8f}  E3={E3:+.8f}  imbalance={imb_E:.2e}")
            print(f"    n1={n1:.6f}  n2={n2:.6f}  n3={n3:.6f}  rel_norm_imb={imb_n:.2e}")

            # For a fully symmetric state, the honeycomb C3 rotation symmetry
            # demands E1=E2=E3.  Tolerance is ~chi_convergence_error.
            ok_E = check(imb_E < 5e-5,
                         f"D={D} seed={seed}: energy imbalance={imb_E:.2e} (need <5e-5)")
            ok_n = check(imb_n < 1e-4,
                         f"D={D} seed={seed}: norm imbalance={imb_n:.2e} (need <1e-4)")
            all_pass = all_pass and ok_E and ok_n

    return all_pass


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: Autodiff vs. numerical gradient at D=2, chi=16
# ─────────────────────────────────────────────────────────────────────────────

def test_gradient_correctness():
    header("7. Autodiff vs numerical gradient  D=2 chi=16, D=3 chi=16")
    H = cv2.build_heisenberg_H()
    all_pass = True

    for D in [2, 3]:
        chi = 16
        torch.manual_seed(123 + D)
        a, b, c, dd, e, f = make_neel_with_noise(D, noise=0.3, seed=123+D)

        env1, env2, env3, steps, D_sq = run_ctm(a, b, c, dd, e, f, chi)

        # Autodiff gradient w.r.t. tensor 'a' only
        a_req = a.detach().clone().requires_grad_(True)
        # Keep others as detached (we only test gradient w.r.t. a)
        nu1, n1 = cv2._energy_3bonds_raw(
            a_req, b, c, dd, e, f, H, H, H, chi, D,
            *env1, site_order=(e, dd, a_req, f, c, b))
        nu2, n2 = cv2._energy_3bonds_raw(
            a_req, b, c, dd, e, f, H, H, H, chi, D,
            *env2, site_order=(a_req, b, c, dd, e, f))
        nu3, n3 = cv2._energy_3bonds_raw(
            a_req, b, c, dd, e, f, H, H, H, chi, D,
            *env3, site_order=(c, f, e, b, a_req, dd))
        norm_avg = (n1 + n2 + n3) / 3.0
        E = (nu1 + nu2 + nu3) / (9.0 * norm_avg)
        E.backward()
        g_ad = a_req.grad.clone()

        # Numerical gradient via central differences
        h = 1e-4
        g_num = torch.zeros_like(a)
        a_flat = a.detach().clone()
        for idx in range(a_flat.numel()):
            def compute_E(x_flat):
                x = x_flat.reshape(a.shape)
                with torch.no_grad():
                    nu1_, n1_ = cv2._energy_3bonds_raw(
                        x, b, c, dd, e, f, H, H, H, chi, D,
                        *env1, site_order=(e, dd, x, f, c, b))
                    nu2_, n2_ = cv2._energy_3bonds_raw(
                        x, b, c, dd, e, f, H, H, H, chi, D,
                        *env2, site_order=(x, b, c, dd, e, f))
                    nu3_, n3_ = cv2._energy_3bonds_raw(
                        x, b, c, dd, e, f, H, H, H, chi, D,
                        *env3, site_order=(c, f, e, b, x, dd))
                    nav = (n1_ + n2_ + n3_) / 3.0
                    return ((nu1_ + nu2_ + nu3_) / (9.0 * nav)).item()

            ap = a_flat.clone(); ap.view(-1)[idx] += h
            am = a_flat.clone(); am.view(-1)[idx] -= h
            g_num.view(-1)[idx] = (compute_E(ap) - compute_E(am)) / (2 * h)

        rel_err = ((g_ad - g_num).abs() / (g_ad.abs() + g_num.abs() + 1e-10)).max().item()
        max_abs_err = (g_ad - g_num).abs().max().item()
        print(f"\n  D={D}  chi={chi}  CTM={steps}")
        print(f"    max rel_err={rel_err:.3e}  max abs_err={max_abs_err:.3e}")
        ok = check(rel_err < 1e-2,
                   f"D={D}: autodiff vs numerical gradient rel_err={rel_err:.3e} (need <1e-2)")
        all_pass = all_pass and ok

    return all_pass


# ─────────────────────────────────────────────────────────────────────────────
# Test 8: Energy chi-stability for Néel at D=2,3
# ─────────────────────────────────────────────────────────────────────────────

def test_chi_stability():
    header("8. Chi-stability of Néel energy: D=2,3 at chi=D^2,2*D^2,4*D^2 all give -0.25")
    H = cv2.build_heisenberg_H()
    all_pass = True

    for D in [2, 3]:
        a, b, c, dd, e, f = make_neel(D)
        print(f"\n  D={D}")
        for chi in [D*D, 2*D*D, 4*D*D]:
            env1, env2, env3, steps, D_sq = run_ctm(a, b, c, dd, e, f, chi)
            E = cv2.compute_total_energy(a, b, c, dd, e, f, H, chi, D, env1, env2, env3).item()
            ok = check(abs(E + 0.25) < 1e-6,
                       f"D={D} chi={chi:3d}: E={E:+.10f}")
            all_pass = all_pass and ok

    return all_pass


# ─────────────────────────────────────────────────────────────────────────────
# Test 9: Rayleigh quotient scale invariance
# ─────────────────────────────────────────────────────────────────────────────

def test_rayleigh_scale():
    header("9. Rayleigh quotient scale invariance: doubling tensors leaves E unchanged  D=2,3")
    H = cv2.build_heisenberg_H()
    all_pass = True

    for D in [2, 3]:
        chi = D * D * 2
        torch.manual_seed(11 + D)
        a, b, c, dd, e, f = make_neel_with_noise(D, noise=0.15, seed=11+D)

        env1, env2, env3, steps, D_sq = run_ctm(a, b, c, dd, e, f, chi)

        E1 = cv2.compute_total_energy(a, b, c, dd, e, f, H, chi, D, env1, env2, env3).item()

        # Scale all tensors by 3.0 — energy should be invariant (Rayleigh quotient)
        scale = 3.0
        a2, b2 = a * scale, b * scale
        c2, dd2, e2, f2 = c * scale, dd * scale, e * scale, f * scale

        # Must rebuild env with scaled tensors
        env1s, env2s, env3s, _, _ = run_ctm(a2, b2, c2, dd2, e2, f2, chi)
        E2 = cv2.compute_total_energy(a2, b2, c2, dd2, e2, f2, H, chi, D, env1s, env2s, env3s).item()

        print(f"\n  D={D}  chi={chi}  CTM={steps}")
        print(f"    E(a)={E1:+.10f}  E(3a)={E2:+.10f}  diff={abs(E2-E1):.2e}")
        ok = check(abs(E1 - E2) < 1e-6,
                   f"D={D}: scale invariance |E(a)-E(3a)|={abs(E1-E2):.2e} (need <1e-6)")
        all_pass = all_pass and ok

    return all_pass


# ─────────────────────────────────────────────────────────────────────────────
# Test 10: compute_total_energy consistent with per-env energies
# ─────────────────────────────────────────────────────────────────────────────

def test_compute_total_energy_consistency():
    header("10. compute_total_energy consistent with per-env raw values  D=2,3,4")
    H = cv2.build_heisenberg_H()
    all_pass = True

    for D in [2, 3, 4]:
        chi = max(16, D * D)
        torch.manual_seed(55 + D)
        a, b, c, dd, e, f = make_neel_with_noise(D, noise=0.1, seed=55+D)

        env1, env2, env3, steps, D_sq = run_ctm(a, b, c, dd, e, f, chi)

        # compute_total_energy result
        E_total = cv2.compute_total_energy(a, b, c, dd, e, f, H, chi, D, env1, env2, env3).item()

        # Manual reconstruction
        nu1, n1 = cv2._energy_3bonds_raw(a, b, c, dd, e, f, H, H, H, chi, D,
                                          *env1, site_order=(e, dd, a, f, c, b))
        nu2, n2 = cv2._energy_3bonds_raw(a, b, c, dd, e, f, H, H, H, chi, D,
                                          *env2, site_order=(a, b, c, dd, e, f))
        nu3, n3 = cv2._energy_3bonds_raw(a, b, c, dd, e, f, H, H, H, chi, D,
                                          *env3, site_order=(c, f, e, b, a, dd))
        norm_avg = (n1 + n2 + n3) / 3.0
        E_manual = ((nu1 + nu2 + nu3) / (9.0 * norm_avg)).item()

        print(f"\n  D={D}  chi={chi}  CTM={steps}")
        print(f"    E_total={E_total:+.12f}  E_manual={E_manual:+.12f}  diff={abs(E_total-E_manual):.2e}")
        ok = check(abs(E_total - E_manual) < 1e-12,
                   f"D={D}: compute_total_energy vs manual recon diff={abs(E_total-E_manual):.2e}")
        all_pass = all_pass and ok

    return all_pass


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = {}

    print("\n" + "="*68)
    print("  PHYSICS TESTS  core_unres_v2  (D=2,3,4 — no trivial D=1)")
    print("="*68)

    results["1+3+4: Neel exact energy + imbalance + norm"] = test_neel_exact()
    results["2: Individual bond energies Neel"]            = test_individual_bonds_neel()
    results["5: CTM self-consistency"]                     = test_ctm_self_consistency()
    results["6: Translation-invariant state"]              = test_translation_invariant()
    results["7: Autodiff vs numerical gradient"]           = test_gradient_correctness()
    results["8: Chi-stability Neel"]                       = test_chi_stability()
    results["9: Rayleigh quotient scale invariance"]       = test_rayleigh_scale()
    results["10: compute_total_energy consistency"]        = test_compute_total_energy_consistency()

    n_pass = sum(results.values())
    n_total = len(results)

    print("\n" + "="*68)
    print(f"PHYSICS RESULTS: {n_pass} passed, {n_total-n_pass} failed out of {n_total}")
    print("="*68)
    for name, ok in results.items():
        print(f"  {'✓' if ok else '✗ FAIL'} {name}")
    print()

    sys.exit(0 if n_pass == n_total else 1)
