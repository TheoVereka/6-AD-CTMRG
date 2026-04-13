"""
Numerical correctness test for the lazy-open energy functions.

Each of the 3 energy expectation functions was refactored so that the six
800 MB open tensors are built INSIDE each bond's _ckpt closure instead of
being pre-allocated outside.  This test verifies that the refactoring
produces *exactly* the same numbers as the canonical reference obtained by
using ``build_open_closed_env{1,2,3}`` + direct calls to
``_compute_nn_bond_energy`` / ``_compute_nnn_bond_energy``.

The test checks:
  - Forward value agreement  (all 12 per-bond scalars + total energy)
  - Relies on the same float64 random tensors for both paths, so any
    index-label swap produces a measurable discrepancy.

Run with:
    python -m pytest src_code/tests/test_lazy_open_correctness.py -v
or:
    python src_code/tests/test_lazy_open_correctness.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import opt_einsum as oe

from core_unrestricted import (
    build_open_closed_env1,
    build_open_closed_env2,
    build_open_closed_env3,
    energy_expectation_nearest_neighbor_3ebadcf_bonds,
    energy_expectation_nearest_neighbor_3afcbed_bonds,
    energy_expectation_nearest_neighbor_other_3_bonds,
    _compute_nn_bond_energy,
    _compute_nnn_bond_energy,
    build_heisenberg_H,
)

# ── test hyperparameters ──────────────────────────────────────────────────────
D_BOND = 3
CHI    = 5
D_PHYS = 2
DTYPE  = torch.float64
CDTYPE = torch.complex128
ATOL   = 1e-8   # absolute tolerance for forward value comparison
SEED   = 1234

torch.manual_seed(SEED)


def _crand(*shape):
    """Random complex128 tensor."""
    return (torch.randn(*shape, dtype=DTYPE)
            + 1j * torch.randn(*shape, dtype=DTYPE)).to(dtype=CDTYPE)


def _make_inputs():
    """Build a full set of random site + environment tensors."""
    D, chi, d = D_BOND, CHI, D_PHYS
    # site tensors: (D, D, D, d)
    sites = {k: _crand(D, D, D, d) for k in 'abcdef'}

    # corners: (chi, chi)
    # transfers: (chi, chi, D*D) — flat last index, reshaped internally
    def _corner(): return _crand(chi, chi)
    def _transfer(): return _crand(chi, chi, D * D)

    env1 = dict(
        C21CD=_corner(), C32EF=_corner(), C13AB=_corner(),
        T1F=_transfer(), T2A=_transfer(), T2B=_transfer(),
        T3C=_transfer(), T3D=_transfer(), T1E=_transfer(),
    )
    env2 = dict(
        C21EB=_corner(), C32AD=_corner(), C13CF=_corner(),
        T1D=_transfer(), T2C=_transfer(), T2F=_transfer(),
        T3E=_transfer(), T3B=_transfer(), T1A=_transfer(),
    )
    env3 = dict(
        C21AF=_corner(), C32CB=_corner(), C13ED=_corner(),
        T1B=_transfer(), T2E=_transfer(), T2D=_transfer(),
        T3A=_transfer(), T3F=_transfer(), T1C=_transfer(),
    )
    return sites, env1, env2, env3


# ─────────────────────────────────────────────────────────────────────────────
# Reference implementations: use build_open_closed_envN + direct calls
# ─────────────────────────────────────────────────────────────────────────────

def _ref_energy_env1(sites, env, J_nn, J_nnn, SdotS):
    """Reference energy for env-1 (3ebadcf bonds).

    J_nn  : dict with keys 'ad','cf','eb','fa','de','bc'
    J_nnn : dict with keys 'ae','ec','ca','db','bf','fd'
    """
    a,b,c,d,e,f = (sites[k] for k in 'abcdef')
    o, cl = build_open_closed_env1(
        a, b, c, d, e, f, CHI, D_BOND, D_PHYS,
        env['C21CD'], env['C32EF'], env['C13AB'],
        env['T1F'],   env['T2A'],   env['T2B'],
        env['T3C'],   env['T3D'],   env['T1E'],
    )
    # pair products — exactly as computed inside the lazy energy function
    AD = torch.mm(cl['A'], cl['D'])
    CF = torch.mm(cl['C'], cl['F'])
    EB = torch.mm(cl['E'], cl['B'])
    FA = torch.mm(cl['F'], cl['A'])
    DE = torch.mm(cl['D'], cl['E'])
    BC = torch.mm(cl['B'], cl['C'])

    # NN bonds — same (open_X, open_Y, pair1, pair2) as the lazy closures
    E_AD = _compute_nn_bond_energy(o['A'], o['D'], EB, CF, SdotS)
    E_CF = _compute_nn_bond_energy(o['C'], o['F'], AD, EB, SdotS)
    E_EB = _compute_nn_bond_energy(o['E'], o['B'], CF, AD, SdotS)
    E_FA = _compute_nn_bond_energy(o['F'], o['A'], DE, BC, SdotS)
    E_DE = _compute_nn_bond_energy(o['D'], o['E'], BC, FA, SdotS)
    E_BC = _compute_nn_bond_energy(o['B'], o['C'], FA, DE, SdotS)

    # NNN bonds — (open_X, closed_Y, open_Z, closed_W, large_pair)
    E_AE = _compute_nnn_bond_energy(o['A'], cl['D'], o['E'], cl['B'], CF,  SdotS)
    E_EC = _compute_nnn_bond_energy(o['E'], cl['B'], o['C'], cl['F'], AD,  SdotS)
    E_CA = _compute_nnn_bond_energy(o['C'], cl['F'], o['A'], cl['D'], EB,  SdotS)
    E_DB = _compute_nnn_bond_energy(o['D'], cl['E'], o['B'], cl['C'], FA,  SdotS)
    E_BF = _compute_nnn_bond_energy(o['B'], cl['C'], o['F'], cl['A'], DE,  SdotS)
    E_FD = _compute_nnn_bond_energy(o['F'], cl['A'], o['D'], cl['E'], BC,  SdotS)

    per_bond = {
        'AD': J_nn['ad']  * E_AD,  'CF': J_nn['cf']  * E_CF,
        'EB': J_nn['eb']  * E_EB,  'FA': J_nn['fa']  * E_FA,
        'DE': J_nn['de']  * E_DE,  'BC': J_nn['bc']  * E_BC,
        'AE': J_nnn['ae'] * E_AE,  'EC': J_nnn['ec'] * E_EC,
        'CA': J_nnn['ca'] * E_CA,  'DB': J_nnn['db'] * E_DB,
        'BF': J_nnn['bf'] * E_BF,  'FD': J_nnn['fd'] * E_FD,
    }
    total = torch.real(
        (per_bond['AD'] + per_bond['CF'] + per_bond['EB']
         + per_bond['FA'] + per_bond['DE'] + per_bond['BC']) * 0.5
        + per_bond['AE'] + per_bond['EC'] + per_bond['CA']
        + per_bond['DB'] + per_bond['BF'] + per_bond['FD']
    )
    return total, per_bond


def _ref_energy_env2(sites, env, J_nn, J_nnn, SdotS):
    """Reference energy for env-2 (3afcbed bonds).

    J_nn  : dict with keys 'af','cb','ed','dc','ba','fe'
    J_nnn : dict with keys 'ca','ae','ec','bf','fd','db'
    """
    a,b,c,d,e,f = (sites[k] for k in 'abcdef')
    o, cl = build_open_closed_env2(
        a, b, c, d, e, f, CHI, D_BOND, D_PHYS,
        env['C21EB'], env['C32AD'], env['C13CF'],
        env['T1D'],   env['T2C'],   env['T2F'],
        env['T3E'],   env['T3B'],   env['T1A'],
    )
    CB = torch.mm(cl['C'], cl['B'])
    ED = torch.mm(cl['E'], cl['D'])
    AF = torch.mm(cl['A'], cl['F'])
    DC = torch.mm(cl['D'], cl['C'])
    BA = torch.mm(cl['B'], cl['A'])
    FE = torch.mm(cl['F'], cl['E'])

    E_CB = _compute_nn_bond_energy(o['C'], o['B'], AF, ED, SdotS)
    E_AF = _compute_nn_bond_energy(o['A'], o['F'], ED, CB, SdotS)
    E_ED = _compute_nn_bond_energy(o['E'], o['D'], CB, AF, SdotS)
    E_DC = _compute_nn_bond_energy(o['D'], o['C'], BA, FE, SdotS)
    E_BA = _compute_nn_bond_energy(o['B'], o['A'], FE, DC, SdotS)
    E_FE = _compute_nn_bond_energy(o['F'], o['E'], DC, BA, SdotS)

    E_CA = _compute_nnn_bond_energy(o['C'], cl['B'], o['A'], cl['F'], ED, SdotS)
    E_AE = _compute_nnn_bond_energy(o['A'], cl['F'], o['E'], cl['D'], CB, SdotS)
    E_EC = _compute_nnn_bond_energy(o['E'], cl['D'], o['C'], cl['B'], AF, SdotS)
    E_BF = _compute_nnn_bond_energy(o['B'], cl['A'], o['F'], cl['E'], DC, SdotS)
    E_FD = _compute_nnn_bond_energy(o['F'], cl['E'], o['D'], cl['C'], BA, SdotS)
    E_DB = _compute_nnn_bond_energy(o['D'], cl['C'], o['B'], cl['A'], FE, SdotS)

    per_bond = {
        'CB': J_nn['cb']  * E_CB,  'AF': J_nn['af']  * E_AF,
        'ED': J_nn['ed']  * E_ED,  'DC': J_nn['dc']  * E_DC,
        'BA': J_nn['ba']  * E_BA,  'FE': J_nn['fe']  * E_FE,
        'CA': J_nnn['ca'] * E_CA,  'AE': J_nnn['ae'] * E_AE,
        'EC': J_nnn['ec'] * E_EC,  'BF': J_nnn['bf'] * E_BF,
        'FD': J_nnn['fd'] * E_FD,  'DB': J_nnn['db'] * E_DB,
    }
    total = torch.real(
        (per_bond['AF'] + per_bond['CB'] + per_bond['ED']
         + per_bond['DC'] + per_bond['BA'] + per_bond['FE']) * 0.5
        + per_bond['CA'] + per_bond['AE'] + per_bond['EC']
        + per_bond['BF'] + per_bond['FD'] + per_bond['DB']
    )
    return total, per_bond


def _ref_energy_env3(sites, env, J_nn, J_nnn, SdotS):
    """Reference energy for env-3 (other_3 bonds).

    J_nn  : dict with keys 'ef','ab','cd','be','fc','da'
    J_nnn : dict with keys 'ec','ca','ae','fd','db','bf'
    """
    a,b,c,d,e,f = (sites[k] for k in 'abcdef')
    o, cl = build_open_closed_env3(
        a, b, c, d, e, f, CHI, D_BOND, D_PHYS,
        env['C21AF'], env['C32CB'], env['C13ED'],
        env['T1B'],   env['T2E'],   env['T2D'],
        env['T3A'],   env['T3F'],   env['T1C'],
    )
    EF = torch.mm(cl['E'], cl['F'])
    AB = torch.mm(cl['A'], cl['B'])
    CD = torch.mm(cl['C'], cl['D'])
    BE = torch.mm(cl['B'], cl['E'])
    FC = torch.mm(cl['F'], cl['C'])
    DA = torch.mm(cl['D'], cl['A'])

    E_EF = _compute_nn_bond_energy(o['E'], o['F'], CD, AB, SdotS)
    E_AB = _compute_nn_bond_energy(o['A'], o['B'], EF, CD, SdotS)
    E_CD = _compute_nn_bond_energy(o['C'], o['D'], AB, EF, SdotS)
    E_BE = _compute_nn_bond_energy(o['B'], o['E'], FC, DA, SdotS)
    E_FC = _compute_nn_bond_energy(o['F'], o['C'], DA, BE, SdotS)
    E_DA = _compute_nn_bond_energy(o['D'], o['A'], BE, FC, SdotS)

    E_EC = _compute_nnn_bond_energy(o['E'], cl['F'], o['C'], cl['D'], AB, SdotS)
    E_CA = _compute_nnn_bond_energy(o['C'], cl['D'], o['A'], cl['B'], EF, SdotS)
    E_AE = _compute_nnn_bond_energy(o['A'], cl['B'], o['E'], cl['F'], CD, SdotS)
    E_FD = _compute_nnn_bond_energy(o['F'], cl['C'], o['D'], cl['A'], BE, SdotS)
    E_DB = _compute_nnn_bond_energy(o['D'], cl['A'], o['B'], cl['E'], FC, SdotS)
    E_BF = _compute_nnn_bond_energy(o['B'], cl['E'], o['F'], cl['C'], DA, SdotS)

    per_bond = {
        'EF': J_nn['ef']  * E_EF,  'AB': J_nn['ab']  * E_AB,
        'CD': J_nn['cd']  * E_CD,  'BE': J_nn['be']  * E_BE,
        'FC': J_nn['fc']  * E_FC,  'DA': J_nn['da']  * E_DA,
        'EC': J_nnn['ec'] * E_EC,  'CA': J_nnn['ca'] * E_CA,
        'AE': J_nnn['ae'] * E_AE,  'FD': J_nnn['fd'] * E_FD,
        'DB': J_nnn['db'] * E_DB,  'BF': J_nnn['bf'] * E_BF,
    }
    total = torch.real(
        (per_bond['EF'] + per_bond['AB'] + per_bond['CD']
         + per_bond['BE'] + per_bond['FC'] + per_bond['DA']) * 0.5
        + per_bond['EC'] + per_bond['CA'] + per_bond['AE']
        + per_bond['FD'] + per_bond['DB'] + per_bond['BF']
    )
    return total, per_bond


# ─────────────────────────────────────────────────────────────────────────────
# Helpers for building the coupling-constant dicts and calling the lazy fns
# ─────────────────────────────────────────────────────────────────────────────

def _random_J(keys, seed_offset=0):
    """Return a dict of random non-trivial coupling constants."""
    torch.manual_seed(SEED + seed_offset)
    return {k: float(torch.rand(1).item() * 2 - 1) for k in keys}  # in (-1, 1)


def _lazy_energy_env1(sites, env, J_nn, J_nnn, SdotS):
    a,b,c,d,e,f = (sites[k] for k in 'abcdef')
    return energy_expectation_nearest_neighbor_3ebadcf_bonds(
        a, b, c, d, e, f,
        # parameter order: Jeb, Jad, Jcf  (EB-bond first in the signature)
        J_nn['eb'],  J_nn['ad'],  J_nn['cf'],
        J_nn['fa'],  J_nn['de'],  J_nn['bc'],
        J_nnn['ae'], J_nnn['ec'], J_nnn['ca'],
        J_nnn['db'], J_nnn['bf'], J_nnn['fd'],
        SdotS,
        CHI, D_BOND, D_PHYS,
        env['C21CD'], env['C32EF'], env['C13AB'],
        env['T1F'],   env['T2A'],   env['T2B'],
        env['T3C'],   env['T3D'],   env['T1E'],
    )


def _lazy_energy_env2(sites, env, J_nn, J_nnn, SdotS):
    a,b,c,d,e,f = (sites[k] for k in 'abcdef')
    return energy_expectation_nearest_neighbor_3afcbed_bonds(
        a, b, c, d, e, f,
        J_nn['af'],  J_nn['cb'],  J_nn['ed'],
        J_nn['dc'],  J_nn['ba'],  J_nn['fe'],
        J_nnn['ca'], J_nnn['ae'], J_nnn['ec'],
        J_nnn['bf'], J_nnn['fd'], J_nnn['db'],
        SdotS,
        CHI, D_BOND, D_PHYS,
        env['C21EB'], env['C32AD'], env['C13CF'],
        env['T1D'],   env['T2C'],   env['T2F'],
        env['T3E'],   env['T3B'],   env['T1A'],
    )


def _lazy_energy_env3(sites, env, J_nn, J_nnn, SdotS):
    a,b,c,d,e,f = (sites[k] for k in 'abcdef')
    return energy_expectation_nearest_neighbor_other_3_bonds(
        a, b, c, d, e, f,
        J_nn['cd'],  J_nn['ef'],  J_nn['ab'],
        J_nn['be'],  J_nn['fc'],  J_nn['da'],
        J_nnn['ec'], J_nnn['ca'], J_nnn['ae'],
        J_nnn['fd'], J_nnn['db'], J_nnn['bf'],
        SdotS,
        CHI, D_BOND, D_PHYS,
        env['C21AF'], env['C32CB'], env['C13ED'],
        env['T1B'],   env['T2E'],   env['T2D'],
        env['T3A'],   env['T3F'],   env['T1C'],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

def _make_SdotS():
    """Build SdotS in complex128 to match test inputs."""
    S = build_heisenberg_H(J=1.0, d=D_PHYS)
    return S.to(dtype=CDTYPE)


def test_env1_forward():
    """Env-1: lazy total energy == reference total energy."""
    SdotS = _make_SdotS()
    sites, env1, _, _ = _make_inputs()
    J_nn  = _random_J(['ad','cf','eb','fa','de','bc'],  seed_offset=0)
    J_nnn = _random_J(['ae','ec','ca','db','bf','fd'],  seed_offset=1)

    ref_total, ref_bonds = _ref_energy_env1(sites, env1, J_nn, J_nnn, SdotS)
    lazy_total           = _lazy_energy_env1(sites, env1, J_nn, J_nnn, SdotS)

    diff = abs(float(lazy_total) - float(ref_total))
    print(f"[env1] ref={float(ref_total):.12f}  lazy={float(lazy_total):.12f}  |diff|={diff:.3e}")
    assert diff < ATOL, (
        f"env1 forward mismatch: |lazy - ref| = {diff:.3e} > {ATOL}"
    )


def test_env2_forward():
    """Env-2: lazy total energy == reference total energy."""
    SdotS = _make_SdotS()
    sites, _, env2, _ = _make_inputs()
    J_nn  = _random_J(['af','cb','ed','dc','ba','fe'],  seed_offset=2)
    J_nnn = _random_J(['ca','ae','ec','bf','fd','db'],  seed_offset=3)

    ref_total, ref_bonds = _ref_energy_env2(sites, env2, J_nn, J_nnn, SdotS)
    lazy_total           = _lazy_energy_env2(sites, env2, J_nn, J_nnn, SdotS)

    diff = abs(float(lazy_total) - float(ref_total))
    print(f"[env2] ref={float(ref_total):.12f}  lazy={float(lazy_total):.12f}  |diff|={diff:.3e}")
    assert diff < ATOL, (
        f"env2 forward mismatch: |lazy - ref| = {diff:.3e} > {ATOL}"
    )


def test_env3_forward():
    """Env-3: lazy total energy == reference total energy."""
    SdotS = _make_SdotS()
    sites, _, _, env3 = _make_inputs()
    J_nn  = _random_J(['ef','ab','cd','be','fc','da'],  seed_offset=4)
    J_nnn = _random_J(['ec','ca','ae','fd','db','bf'],  seed_offset=5)

    ref_total, ref_bonds = _ref_energy_env3(sites, env3, J_nn, J_nnn, SdotS)
    lazy_total           = _lazy_energy_env3(sites, env3, J_nn, J_nnn, SdotS)

    diff = abs(float(lazy_total) - float(ref_total))
    print(f"[env3] ref={float(ref_total):.12f}  lazy={float(lazy_total):.12f}  |diff|={diff:.3e}")
    assert diff < ATOL, (
        f"env3 forward mismatch: |lazy - ref| = {diff:.3e} > {ATOL}"
    )


def test_env1_unit_J():
    """Env-1 with all J=1: ensures sign/symmetry are unchanged."""
    SdotS = _make_SdotS()
    sites, env1, _, _ = _make_inputs()
    J1 = {k: 1.0 for k in ['ad','cf','eb','fa','de','bc']}
    J1n = {k: 1.0 for k in ['ae','ec','ca','db','bf','fd']}
    ref_total, _ = _ref_energy_env1(sites, env1, J1, J1n, SdotS)
    lazy_total   = _lazy_energy_env1(sites, env1, J1, J1n, SdotS)
    diff = abs(float(lazy_total) - float(ref_total))
    assert diff < ATOL, f"env1 unit-J mismatch: {diff:.3e}"


def test_env2_unit_J():
    SdotS = _make_SdotS()
    sites, _, env2, _ = _make_inputs()
    J1  = {k: 1.0 for k in ['af','cb','ed','dc','ba','fe']}
    J1n = {k: 1.0 for k in ['ca','ae','ec','bf','fd','db']}
    ref_total, _ = _ref_energy_env2(sites, env2, J1, J1n, SdotS)
    lazy_total   = _lazy_energy_env2(sites, env2, J1, J1n, SdotS)
    diff = abs(float(lazy_total) - float(ref_total))
    assert diff < ATOL, f"env2 unit-J mismatch: {diff:.3e}"


def test_env3_unit_J():
    SdotS = _make_SdotS()
    sites, _, _, env3 = _make_inputs()
    J1  = {k: 1.0 for k in ['ef','ab','cd','be','fc','da']}
    J1n = {k: 1.0 for k in ['ec','ca','ae','fd','db','bf']}
    ref_total, _ = _ref_energy_env3(sites, env3, J1, J1n, SdotS)
    lazy_total   = _lazy_energy_env3(sites, env3, J1, J1n, SdotS)
    diff = abs(float(lazy_total) - float(ref_total))
    assert diff < ATOL, f"env3 unit-J mismatch: {diff:.3e}"


def test_env1_nn_only():
    """Env-1 with NNN J=0: only NN bonds active — isolates NN path."""
    SdotS = _make_SdotS()
    sites, env1, _, _ = _make_inputs()
    J_nn  = _random_J(['ad','cf','eb','fa','de','bc'],  seed_offset=6)
    J_nnn = {k: 0.0 for k in ['ae','ec','ca','db','bf','fd']}
    ref_total, _ = _ref_energy_env1(sites, env1, J_nn, J_nnn, SdotS)
    lazy_total   = _lazy_energy_env1(sites, env1, J_nn, J_nnn, SdotS)
    diff = abs(float(lazy_total) - float(ref_total))
    assert diff < ATOL, f"env1 NN-only mismatch: {diff:.3e}"


def test_env1_nnn_only():
    """Env-1 with NN J=0: only NNN bonds active — isolates NNN path."""
    SdotS = _make_SdotS()
    sites, env1, _, _ = _make_inputs()
    J_nn  = {k: 0.0 for k in ['ad','cf','eb','fa','de','bc']}
    J_nnn = _random_J(['ae','ec','ca','db','bf','fd'],  seed_offset=7)
    ref_total, _ = _ref_energy_env1(sites, env1, J_nn, J_nnn, SdotS)
    lazy_total   = _lazy_energy_env1(sites, env1, J_nn, J_nnn, SdotS)
    diff = abs(float(lazy_total) - float(ref_total))
    assert diff < ATOL, f"env1 NNN-only mismatch: {diff:.3e}"


# ─────────────────────────────────────────────────────────────────────────────
# Standalone runner
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    tests = [
        test_env1_forward,
        test_env2_forward,
        test_env3_forward,
        test_env1_unit_J,
        test_env2_unit_J,
        test_env3_unit_J,
        test_env1_nn_only,
        test_env1_nnn_only,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            import traceback
            print(f"  ERROR {t.__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {passed+failed}")
    if failed > 0:
        sys.exit(1)
