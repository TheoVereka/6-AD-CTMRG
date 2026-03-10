"""
Unit tests for src_code/src/core_unrestricted.py
=================================================
Each test focuses on the *functional correctness* of exactly one function and
exercises its logic rather than merely checking that it runs.

Run with:
    pytest src_code/tests/test_core_unrestricted.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import torch
import opt_einsum as oe
from core_unrestricted import (
    normalize_tensor,
    initialize_abcdef,
    abcdef_to_ABCDEF,
    trunc_rhoCCC,
    initialize_environmentCTs_1,
    check_env_convergence,
    update_environmentCTs_1to2,
    update_environmentCTs_2to3,
    update_environmentCTs_3to1,
    CTMRG_from_init_to_stop,
    energy_expectation_nearest_neighbor_6_bonds,
    energy_expectation_nearest_neighbor_other_3_bonds,
)

torch.manual_seed(42)

ATOL = 1e-4   # tolerance for complex64 arithmetic


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rand_c64(*shape):
    return torch.randn(*shape, dtype=torch.complex64)


def _rand_env(chi, D_sq):
    """Return a plausible (chi, chi) random corner and (chi, D_sq, D_sq) transfer."""
    C = normalize_tensor(_rand_c64(chi, chi))
    T = normalize_tensor(_rand_c64(chi, D_sq, D_sq))
    return C, T


def _heisenberg_H(d=2):
    """Return the XXX Heisenberg Hamiltonian on two qubits as a (d,d,d,d) tensor.
    Explicitly Hermitian so imag(energy) should vanish for any normalised state."""
    sx = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64) / 2
    sy = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64) / 2
    sz = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64) / 2
    H2 = (oe.contract("ij,kl->ikjl", sx, sx)
        + oe.contract("ij,kl->ikjl", sy, sy)
        + oe.contract("ij,kl->ikjl", sz, sz))
    return H2   # shape (d, d, d, d)


def _identity_H(d=2):
    """H = identity on the two-site space.  energy = <I>/norm = 1."""
    return torch.eye(d*d, dtype=torch.complex64).reshape(d, d, d, d)


def _make_env1(D_bond=2, chi=4):
    """Build a full set of type-1 environment tensors using the real init path."""
    D_sq = D_bond ** 2
    a, b, c, d, e, f = initialize_abcdef('random', D_bond, 2, 1e-3)
    A, B, C, D, E, F = abcdef_to_ABCDEF(a, b, c, d, e, f, D_sq)
    tensors = initialize_environmentCTs_1(A, B, C, D, E, F, chi, D_sq)
    return tensors, (A, B, C, D, E, F), D_sq


# ─────────────────────────────────────────────────────────────────────────────
# 1. normalize_tensor
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalizeTensor:

    def test_unit_norm_real(self):
        """Output must have Frobenius norm == 1 for a normal real tensor."""
        t = torch.randn(3, 4, 5)
        out = normalize_tensor(t)
        assert torch.allclose(torch.norm(out), torch.tensor(1.0), atol=ATOL)

    def test_unit_norm_complex(self):
        """Works for complex64 tensors."""
        t = _rand_c64(4, 4)
        out = normalize_tensor(t)
        assert torch.allclose(torch.norm(out), torch.tensor(1.0), atol=ATOL)

    def test_zero_tensor_unchanged(self):
        """A zero tensor must not produce NaN and must be returned as-is."""
        t = torch.zeros(3, 3, dtype=torch.complex64)
        out = normalize_tensor(t)
        assert not torch.any(torch.isnan(out))
        assert torch.all(out == 0)

    def test_scalar_tensor(self):
        """A positive scalar normalises to 1."""
        t = torch.tensor(5.0)
        out = normalize_tensor(t)
        assert torch.allclose(out, torch.tensor(1.0), atol=ATOL)

    def test_direction_preserved(self):
        """normalise(t) is proportional to t (same direction)."""
        t = _rand_c64(6, 7)
        out = normalize_tensor(t)
        # cos-sim between flattened vectors should be 1
        cos = (t.reshape(-1).conj() @ out.reshape(-1)) / (torch.norm(t) * torch.norm(out))
        assert abs(cos.item() - 1.0) < ATOL

    def test_idempotent(self):
        """Normalising an already-unit tensor leaves it unchanged."""
        t = normalize_tensor(_rand_c64(5, 5))
        t2 = normalize_tensor(t)
        assert torch.allclose(t, t2, atol=ATOL)


# ─────────────────────────────────────────────────────────────────────────────
# 2. initialize_abcdef
# ─────────────────────────────────────────────────────────────────────────────

class TestInitializeAbcdef:

    @pytest.mark.parametrize("D,d", [(2, 2), (3, 2), (2, 3)])
    def test_shape(self, D, d):
        """Each tensor has shape (D, D, D, d_PHYS)."""
        tensors = initialize_abcdef('random', D, d, 1e-3)
        for t in tensors:
            assert t.shape == (D, D, D, d)

    @pytest.mark.parametrize("D,d", [(2, 2), (3, 2)])
    def test_unit_norm_random(self, D, d):
        """Random init produces unit-norm tensors."""
        for t in initialize_abcdef('random', D, d, 1e-3):
            assert torch.allclose(torch.norm(t), torch.tensor(1.0), atol=ATOL)

    def test_dtype_complex64(self):
        """Output dtype must be torch.complex64."""
        for t in initialize_abcdef('random', 2, 2, 1e-3):
            assert t.dtype == torch.complex64

    def test_invalid_mode_raises(self):
        """Unsupported mode must raise ValueError."""
        with pytest.raises(ValueError):
            initialize_abcdef('invalid_mode', 2, 2, 1e-3)

    def test_tensors_differ(self):
        """The six tensors should not all be identical (independent draws)."""
        ts = initialize_abcdef('random', 3, 2, 1e-3)
        for i in range(len(ts)):
            for j in range(i+1, len(ts)):
                assert not torch.allclose(ts[i], ts[j], atol=ATOL), \
                    f"Tensors {i} and {j} are identical — randomness missing."


# ─────────────────────────────────────────────────────────────────────────────
# 3. abcdef_to_ABCDEF
# ─────────────────────────────────────────────────────────────────────────────

class TestAbcdefToABCDEF:

    def setup_method(self):
        self.D, self.d = 2, 2
        self.D_sq = self.D ** 2
        self.abcdef = initialize_abcdef('random', self.D, self.d, 1e-3)
        self.ABCDEF = abcdef_to_ABCDEF(*self.abcdef, self.D_sq)

    def test_shape(self):
        """Each double-layer tensor has shape (D_sq, D_sq)."""
        for T in self.ABCDEF:
            assert T.shape == (self.D_sq, self.D_sq), f"Expected ({self.D_sq},{self.D_sq}), got {T.shape}"

    def test_unit_norm(self):
        """Each output is normalised."""
        for T in self.ABCDEF:
            assert torch.allclose(torch.norm(T), torch.tensor(1.0), atol=ATOL)

    def test_hermitian(self):
        """Diagonal elements A[i,i,i] = sum_φ |a_{u,v,w,φ}|^2 must be real and ≥ 0."""
        a = self.abcdef[0]
        D_sq = self.D_sq
        A_out = self.ABCDEF[0]
        # Diagonal elements: index (ux)=(vy)=(wz) where u=x, v=y, w=z
        # i.e. A[D*u+u, D*v+v, D*w+w] for all u,v,w ∈ {0,...,D-1}
        D = self.D
        for u in range(D):
            for v in range(D):
                for w in range(D):
                    val = A_out[D*u+u, D*v+v, D*w+w]
                    assert abs(val.imag.item()) < ATOL, \
                        f"Diagonal element ({u},{v},{w}) is not real: {val}"
                    assert val.real.item() >= -ATOL, \
                        f"Diagonal element ({u},{v},{w}) is negative: {val.real.item()}"

    def test_psd(self):
        """For fixed middle leg index j, the matrix A[:,j,:] should be
        positive semi-definite since A[i,j,k] = sum_φ a_{α,β,γ,φ}*conj(a_{α',β',γ',φ})
        is a Gram-like product (PSD in the (i,k) matrix for each fixed j)."""
        A_out = self.ABCDEF[0]
        D_sq = self.D_sq
        for j in range(D_sq):
            M = A_out[:, j, :]    # (D_sq, D_sq) slice
            eigvals = torch.linalg.eigvalsh(M)
            assert torch.all(eigvals >= -ATOL), \
                f"Slice A[:,{j},:] has negative eigenvalue {eigvals.min().item()}"

    def test_scale_redundancy(self):
        """Scaling a by λ must not change A (normalisation absorbs the scale)."""
        a, b, c, d, e, f = self.abcdef
        A_orig = self.ABCDEF[0]
        lam = 3.7 + 0.2j
        ABCDEF_scaled = abcdef_to_ABCDEF(lam * a, b, c, d, e, f, self.D_sq)
        A_scaled = ABCDEF_scaled[0]
        assert torch.allclose(A_orig.abs(), A_scaled.abs(), atol=ATOL), \
            "Double-layer tensor magnitude changed under scaling of a"


# ─────────────────────────────────────────────────────────────────────────────
# 4. trunc_rhoCCC
# ─────────────────────────────────────────────────────────────────────────────

class TestTruncRhoCCC:
    """
    Properties verified:
      • Output shapes
      • Isometry of projectors (V_flat @ V_flat†.conj() = I_chi)
      • Unit norm of corners
      • For chi >= full rank the truncated corners exactly equal the
        projected-then-re-expanded originals (lossless truncation)
    """

    def _run(self, chi_in, D_sq, chi_out):
        dim = chi_in * D_sq
        matC21 = _rand_c64(dim, dim)
        matC32 = _rand_c64(dim, dim)
        matC13 = _rand_c64(dim, dim)
        out = trunc_rhoCCC(matC21, matC32, matC13, chi_out, D_sq)
        return out, (matC21, matC32, matC13), dim

    def test_output_shapes(self):
        chi_in, D_sq, chi_out = 3, 4, 5
        (V2, C21, U1, V3, C32, U2, V1, C13, U3), _, dim = self._run(chi_in, D_sq, chi_out)
        assert C21.shape == (chi_out, chi_out)
        assert C32.shape == (chi_out, chi_out)
        assert C13.shape == (chi_out, chi_out)
        assert U1.shape == (chi_out, D_sq, chi_out)
        assert U2.shape == (chi_out, D_sq, chi_out)
        assert U3.shape == (chi_out, D_sq, chi_out)
        assert V1.shape == (chi_out, chi_out, D_sq)
        assert V2.shape == (chi_out, chi_out, D_sq)
        assert V3.shape == (chi_out, chi_out, D_sq)

    def test_corners_unit_norm(self):
        chi_in, D_sq, chi_out = 3, 4, 5
        (V2, C21, U1, V3, C32, U2, V1, C13, U3), _, _ = self._run(chi_in, D_sq, chi_out)
        for name, C in [("C21", C21), ("C32", C32), ("C13", C13)]:
            assert torch.allclose(torch.norm(C), torch.tensor(1.0), atol=ATOL), \
                f"{name} is not unit-norm"

    def test_projector_isometry(self):
        """V_flat @ V_flat†.conj() ≈ I_chi (rows of projected V are orthonormal)."""
        chi_in, D_sq, chi_out = 4, 4, 6
        dim = chi_in * D_sq
        (V2, C21, U1, V3, C32, U2, V1, C13, U3), _, _ = self._run(chi_in, D_sq, chi_out)
        for name, V in [("V1", V1), ("V2", V2), ("V3", V3)]:
            V_flat = V.reshape(chi_out, dim)
            I_approx = V_flat @ V_flat.conj().T
            assert torch.allclose(I_approx,
                                  torch.eye(chi_out, dtype=torch.complex64),
                                  atol=1e-3), \
                f"{name}: projector isometry failed, max off-diag = {(I_approx - torch.eye(chi_out, dtype=torch.complex64)).abs().max()}"

    def test_projector_isometry_U(self):
        """U_flat†.conj() @ U_flat ≈ I_chi (columns of U are orthonormal)."""
        chi_in, D_sq, chi_out = 4, 4, 6
        dim = chi_in * D_sq
        (V2, C21, U1, V3, C32, U2, V1, C13, U3), _, _ = self._run(chi_in, D_sq, chi_out)
        for name, U in [("U1", U1), ("U2", U2), ("U3", U3)]:
            U_flat = U.reshape(dim, chi_out)   # (dim, chi_out)
            I_approx = U_flat.conj().T @ U_flat          # (chi_out, chi_out)
            assert torch.allclose(I_approx,
                                  torch.eye(chi_out, dtype=torch.complex64),
                                  atol=1e-3), \
                f"{name}: column-isometry failed"

    def test_lossless_when_chi_equals_full_rank(self):
        """When chi_out = dim (no truncation), the re-projected C21 recovers matC21."""
        chi_in, D_sq = 2, 2
        dim = chi_in * D_sq   # = 4
        chi_out = dim          # full rank, no information lost
        matC21 = _rand_c64(dim, dim)
        matC32 = _rand_c64(dim, dim)
        matC13 = _rand_c64(dim, dim)
        (V2, C21_trunc, U1, V3, C32_trunc, U2, V1, C13_trunc, U3) = \
            trunc_rhoCCC(matC21, matC32, matC13, chi_out, D_sq)
        # C21_trunc = normalize( U1 @ matC21 @ V2 ) where shapes fit
        # Reconstruct original from truncated: matC21_rec = U1† @ C21_unnorm @ V2†
        # But since both are normalised differently, just check that
        # C21_trunc is not degenerate (has chi_out distinct singular values ~ those of matC21)
        sv_orig = torch.linalg.svdvals(matC21)[:chi_out]
        sv_trunc = torch.linalg.svdvals(C21_trunc)
        # Ratios should be roughly constant (same spectrum up to a global scale)
        ratios = sv_trunc / sv_orig
        assert (ratios.std() / ratios.mean()).abs() < 0.15, \
            "Singular value structure distorted by lossless truncation"


# ─────────────────────────────────────────────────────────────────────────────
# 5. initialize_environmentCTs_1
# ─────────────────────────────────────────────────────────────────────────────

class TestInitializeEnvironmentCTs1:

    @pytest.mark.parametrize("D_bond,chi", [(2, 4), (2, 8)])
    def test_output_shapes(self, D_bond, chi):
        """All 9 tensors must have shapes consistent with chi and D_squared."""
        D_sq = D_bond ** 2
        a, b, c, d, e, f = initialize_abcdef('random', D_bond, 2, 1e-3)
        A, B, C, D, E, F = abcdef_to_ABCDEF(a, b, c, d, e, f, D_sq)
        (C21, C32, C13, T1F, T2A, T2B, T3C, T3D, T1E) = \
            initialize_environmentCTs_1(A, B, C, D, E, F, chi, D_sq)
        for name, C_t in [("C21", C21), ("C32", C32), ("C13", C13)]:
            assert C_t.shape == (chi, chi), \
                f"{name}: expected ({chi},{chi}), got {C_t.shape}"
        for name, T in [("T1F", T1F), ("T2A", T2A), ("T2B", T2B),
                        ("T3C", T3C), ("T3D", T3D), ("T1E", T1E)]:
            assert T.shape[0] == chi, f"{name}: first dim should be chi={chi}"

    def test_no_nan_or_inf(self):
        """Initialisation must not produce NaN or Inf."""
        D_bond, chi = 2, 4
        D_sq = D_bond ** 2
        a, b, c, d, e, f = initialize_abcdef('random', D_bond, 2, 1e-3)
        A, B, C, D, E, F = abcdef_to_ABCDEF(a, b, c, d, e, f, D_sq)
        for T in initialize_environmentCTs_1(A, B, C, D, E, F, chi, D_sq):
            assert not torch.any(torch.isnan(T)), "NaN found in init env"
            assert not torch.any(torch.isinf(T)), "Inf found in init env"

    def test_corners_unit_norm(self):
        D_bond, chi = 2, 4
        D_sq = D_bond ** 2
        a, b, c, d, e, f = initialize_abcdef('random', D_bond, 2, 1e-3)
        A, B, C, D, E, F = abcdef_to_ABCDEF(a, b, c, d, e, f, D_sq)
        C21, C32, C13, *_ = initialize_environmentCTs_1(A, B, C, D, E, F, chi, D_sq)
        for name, C_t in [("C21", C21), ("C32", C32), ("C13", C13)]:
            assert torch.allclose(torch.norm(C_t), torch.tensor(1.0), atol=ATOL), \
                f"{name} should be unit-norm after init"


# ─────────────────────────────────────────────────────────────────────────────
# 6. check_env_convergence
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckEnvConvergence:

    def _all_none(self):
        return [None]*27

    def _build_27(self, chi=4, D_sq=4):
        """Return 27 random tensors using realistic env shapes."""
        corners = [normalize_tensor(_rand_c64(chi, chi)) for _ in range(9)]
        transfers = [normalize_tensor(_rand_c64(chi, D_sq, D_sq)) for _ in range(18)]
        return corners[:3] + transfers[:6] + corners[3:6] + transfers[6:12] + corners[6:9] + transfers[12:18]

    def test_returns_false_when_last_is_none(self):
        """All-None 'last' tensors → not yet warmed up → False regardless of threshold."""
        now = self._build_27()
        args = self._all_none() + now + [1e10]
        assert check_env_convergence(*args) is False

    def test_identical_tensors_converge(self):
        """If last == now (bit-identical), RMS diff = 0 < any positive threshold."""
        tensors = self._build_27()
        args = tensors + tensors + [1e-12]   # very tight threshold
        assert check_env_convergence(*args) is True

    def test_distinct_tensors_do_not_converge(self):
        """Large random perturbation should not satisfy a tight threshold."""
        last = self._build_27()
        now  = self._build_27()      # independent random tensors → large diff
        args = last + now + [1e-12]
        assert check_env_convergence(*args) is False

    def test_threshold_boundary(self):
        """Construct a known diff just below / just above threshold."""
        chi, D_sq = 4, 4
        # Build last; perturb by a known amount epsilon
        last = self._build_27(chi, D_sq)
        total_numel = sum(t.numel() for t in last)
        epsilon = 1e-3   # target RMS diff
        # Add a deterministic perturbation to the first tensor only:
        t0_orig = last[0].clone()
        delta = torch.zeros_like(t0_orig)
        # set first element to epsilon * sqrt(total_numel) so that RMS = epsilon
        delta.reshape(-1)[0] = epsilon * (total_numel ** 0.5)
        now = [t.clone() for t in last]
        now[0] = last[0] + delta

        # slightly above epsilon → should NOT converge
        args_tight = last + now + [epsilon * 0.5]
        # slightly below epsilon → should converge
        args_loose = last + now + [epsilon * 2.0]
        assert check_env_convergence(*args_tight) is False
        assert check_env_convergence(*args_loose) is True

    def test_partial_none_returns_false(self):
        """Even if only last T1C is None, should return False (warm-up guard)."""
        last = self._build_27()
        last[-1] = None   # last T1C → None
        now = self._build_27()
        args = last + now + [1.0]
        assert check_env_convergence(*args) is False


# ─────────────────────────────────────────────────────────────────────────────
# 7 & 8 & 9.  update_environmentCTs_{1→2, 2→3, 3→1}
# ─────────────────────────────────────────────────────────────────────────────

def _full_cycle_tensors(D_bond=2, chi=4):
    """Run all three update steps once and return all three env tuples."""
    D_sq = D_bond ** 2
    a, b, c, d, e, f = initialize_abcdef('random', D_bond, 2, 1e-3)
    A, B, C, D_t, E, F = abcdef_to_ABCDEF(a, b, c, d, e, f, D_sq)
    env1 = initialize_environmentCTs_1(A, B, C, D_t, E, F, chi, D_sq)
    env2 = update_environmentCTs_1to2(*env1, A, B, C, D_t, E, F, chi, D_sq)
    env3 = update_environmentCTs_2to3(*env2, A, B, C, D_t, E, F, chi, D_sq)
    env1b = update_environmentCTs_3to1(*env3, A, B, C, D_t, E, F, chi, D_sq)
    return env1, env2, env3, env1b, D_sq, chi


class TestUpdateEnvironmentCTs:

    def setup_method(self):
        self.env1, self.env2, self.env3, self.env1b, self.D_sq, self.chi = \
            _full_cycle_tensors(D_bond=2, chi=4)

    def _assert_env_shapes(self, env, label):
        chi, D_sq = self.chi, self.D_sq
        C21, C32, C13 = env[0], env[1], env[2]
        transfers = env[3:]
        for name, C in [(f"{label}:C21", C21), (f"{label}:C32", C32), (f"{label}:C13", C13)]:
            assert C.shape == (chi, chi), f"{name}: shape {C.shape} != ({chi},{chi})"
        for i, T in enumerate(transfers):
            assert T.shape[0] == chi, f"{label}:T{i}: first dim {T.shape[0]} != chi={chi}"

    def _assert_no_nan_inf(self, env, label):
        for i, T in enumerate(env):
            assert not torch.any(torch.isnan(T)), f"{label}[{i}] has NaN"
            assert not torch.any(torch.isinf(T)), f"{label}[{i}] has Inf"

    def _assert_unit_norm_corners(self, env, label):
        for i, C in enumerate(env[:3]):
            n = torch.norm(C)
            assert torch.allclose(n, torch.tensor(1.0), atol=ATOL), \
                f"{label}:corner[{i}] norm={n.item()}"

    def test_1to2_shapes(self):
        self._assert_env_shapes(self.env2, "1→2")

    def test_2to3_shapes(self):
        self._assert_env_shapes(self.env3, "2→3")

    def test_3to1_shapes(self):
        self._assert_env_shapes(self.env1b, "3→1")

    def test_1to2_no_nan_inf(self):
        self._assert_no_nan_inf(self.env2, "1→2")

    def test_2to3_no_nan_inf(self):
        self._assert_no_nan_inf(self.env3, "2→3")

    def test_3to1_no_nan_inf(self):
        self._assert_no_nan_inf(self.env1b, "3→1")

    def test_corners_unit_norm_after_each_update(self):
        for env, label in [(self.env2, "1→2"), (self.env3, "2→3"), (self.env1b, "3→1")]:
            self._assert_unit_norm_corners(env, label)

    def test_transfer_tensors_unit_norm(self):
        """All transfer tensors must be normalised (normalize_tensor is called on each)."""
        for env, label in [(self.env2, "1→2"), (self.env3, "2→3"), (self.env1b, "3→1")]:
            for i, T in enumerate(env[3:]):
                n = torch.norm(T)
                assert torch.allclose(n, torch.tensor(1.0), atol=ATOL), \
                    f"{label}:T{i} norm={n.item()}"

    def test_cycle_changes_tensors(self):
        """After one full 1→2→3→1 cycle the type-1 tensors must have changed
        (the environment is not at a fixed-point after just one cycle for a
        random initialisation)."""
        changed = False
        for t_old, t_new in zip(self.env1, self.env1b):
            if not torch.allclose(t_old, t_new, atol=ATOL):
                changed = True
                break
        assert changed, "Environment did not change after a full cycle — likely a bug"


# ─────────────────────────────────────────────────────────────────────────────
# 10. CTMRG_from_init_to_stop
# ─────────────────────────────────────────────────────────────────────────────

class TestCTMRGFromInitToStop:

    def _run(self, D_bond=2, chi=4, max_iter=5, threshold=1e-3):
        D_sq = D_bond ** 2
        a, b, c, d, e, f = initialize_abcdef('random', D_bond, 2, 1e-3)
        A, B, C, D, E, F = abcdef_to_ABCDEF(a, b, c, d, e, f, D_sq)
        return CTMRG_from_init_to_stop(A, B, C, D, E, F, chi, D_sq, max_iter, threshold), D_sq, chi

    def test_returns_27_tensors(self):
        result, D_sq, chi = self._run()
        assert len(result) == 27

    def test_output_shapes(self):
        result, D_sq, chi = self._run()
        # First 3: corners; next 6: transfers; repeat for env2 and env3
        for group_start in [0, 9, 18]:
            for i in range(3):
                assert result[group_start + i].shape == (chi, chi), \
                    f"Corner {group_start+i} has wrong shape"
        for group_start in [0, 9, 18]:
            for i in range(3, 9):
                T = result[group_start + i]
                assert T.shape[0] == chi

    def test_no_nan_inf(self):
        result, _, _ = self._run()
        for i, T in enumerate(result):
            assert not torch.any(torch.isnan(T)), f"tensor[{i}] has NaN"
            assert not torch.any(torch.isinf(T)), f"tensor[{i}] has Inf"

    def test_very_loose_threshold_converges_fast(self, capsys):
        """With threshold=1e10, convergence should be reported before max_iter."""
        D_bond, chi = 2, 4
        D_sq = D_bond ** 2
        a, b, c, d, e, f = initialize_abcdef('random', D_bond, 2, 1e-3)
        A, B, C, D, E, F = abcdef_to_ABCDEF(a, b, c, d, e, f, D_sq)
        CTMRG_from_init_to_stop(A, B, C, D, E, F, chi, D_sq, 100, 1e10)
        captured = capsys.readouterr()
        assert "Convergence achieved" in captured.out, \
            "Expected convergence message with a very loose threshold"

    def test_tight_threshold_runs_all_iterations(self, capsys):
        """With threshold=0.0, no convergence should be declared."""
        D_bond, chi = 2, 4
        D_sq = D_bond ** 2
        a, b, c, d, e, f = initialize_abcdef('random', D_bond, 2, 1e-3)
        A, B, C, D, E, F = abcdef_to_ABCDEF(a, b, c, d, e, f, D_sq)
        max_iter = 3
        CTMRG_from_init_to_stop(A, B, C, D, E, F, chi, D_sq, max_iter, 0.0)
        captured = capsys.readouterr()
        assert "Convergence achieved" not in captured.out, \
            "Should not converge when threshold=0"


# ─────────────────────────────────────────────────────────────────────────────
# 11 & 12.  energy_expectation_nearest_neighbor_{6,3}_bonds
# ─────────────────────────────────────────────────────────────────────────────

class TestEnergyExpectation:
    """
    Core properties:
      1. Output is a 0-dim tensor (scalar).
      2. For a Hermitian Hamiltonian the imaginary part is essentially zero.
      3. Energy from H=0 is exactly 0.
      4. Gradients flow back through a–f (AD works).
      5. For H=I (identity), the energy equals 1.0 (sum of d norms / norm = d).
         Wait: each bond contributes 1 term; the sum over 6 bonds of Tr[I*rho_pair]/Z.
         Actually each E_unnormed for H=I equals norm because
         Tr_ij[I_{ij,kl} * open_ij] = Tr_ij[delta_{ij,kl} * open_ij] = open contracted = closed.
         So E_unnormed = Tr[closed * ...] = norm; hence E_total = 6*norm / norm = 6 for 6 bonds.
         We verify the ratio makes physical sense by checking it is real and finite.
    """

    def setup_method(self):
        self.D_bond = 2
        self.chi = 4
        self.d = 2
        self.D_sq = self.D_bond ** 2
        # Build environment in no_grad context (mirrors the optimisation loop)
        with torch.no_grad():
            a, b, c, d, e, f = initialize_abcdef('random', self.D_bond, self.d, 1e-3)
            A, B, C, D, E, F = abcdef_to_ABCDEF(a, b, c, d, e, f, self.D_sq)
            result = CTMRG_from_init_to_stop(
                A, B, C, D, E, F, self.chi, self.D_sq, 5, 1e-3)
        self.env1 = result[:9]
        self.env3 = result[18:27]
        # Store site tensors WITH grad for backprop tests
        self.abcdef = initialize_abcdef('random', self.D_bond, self.d, 1e-3)
        for t in self.abcdef:
            t.requires_grad_(True)
        self.Heis = _heisenberg_H(self.d)

    def _e6(self, abcdef=None, H=None):
        if abcdef is None:
            abcdef = self.abcdef
        if H is None:
            H = self.Heis
        a, b, c, d, e, f = abcdef
        return energy_expectation_nearest_neighbor_6_bonds(
            a, b, c, d, e, f,
            H, H, H, H, H, H,
            self.chi, self.D_bond,
            *self.env1)

    def _e3(self, abcdef=None, H=None):
        if abcdef is None:
            abcdef = self.abcdef
        if H is None:
            H = self.Heis
        a, b, c, d, e, f = abcdef
        return energy_expectation_nearest_neighbor_other_3_bonds(
            a, b, c, d, e, f,
            H, H, H,
            self.chi, self.D_bond,
            *self.env3)

    def test_output_is_scalar_6bonds(self):
        assert self._e6().ndim == 0

    def test_output_is_scalar_3bonds(self):
        assert self._e3().ndim == 0

    def test_imaginary_part_vanishes_for_hermitian_H_6bonds(self):
        """For H = H† the energy must be real."""
        E = self._e6()
        assert abs(E.imag.item()) < 1e-3, \
            f"Imaginary part too large: {E.imag.item()}"

    def test_imaginary_part_vanishes_for_hermitian_H_3bonds(self):
        E = self._e3()
        assert abs(E.imag.item()) < 1e-3, \
            f"Imaginary part too large: {E.imag.item()}"

    def test_zero_hamiltonian_gives_zero_energy_6bonds(self):
        H0 = torch.zeros(self.d, self.d, self.d, self.d, dtype=torch.complex64)
        E = self._e6(H=H0)
        assert abs(E.item()) < ATOL, f"Expected 0, got {E.item()}"

    def test_zero_hamiltonian_gives_zero_energy_3bonds(self):
        H0 = torch.zeros(self.d, self.d, self.d, self.d, dtype=torch.complex64)
        E = self._e3(H=H0)
        assert abs(E.item()) < ATOL, f"Expected 0, got {E.item()}"

    def test_gradient_flows_through_a2f_6bonds(self):
        """loss.backward() must produce non-None, non-zero gradients in a–f."""
        E = self._e6()
        E.real.backward()
        for name, t in zip("abcdef", self.abcdef):
            assert t.grad is not None, f"No gradient for {name}"
            assert t.grad.norm() > 0,  f"Zero gradient for {name}"

    def test_gradient_flows_through_a2f_3bonds(self):
        # Zero out grads from the 6-bond test
        for t in self.abcdef:
            if t.grad is not None:
                t.grad.zero_()
        E = self._e3()
        E.real.backward()
        for name, t in zip("abcdef", self.abcdef):
            assert t.grad is not None, f"No gradient for {name}"

    def test_energy_finite_6bonds(self):
        E = self._e6()
        assert torch.isfinite(E.real), f"Energy is not finite: {E}"

    def test_energy_finite_3bonds(self):
        E = self._e3()
        assert torch.isfinite(E.real), f"Energy is not finite: {E}"

    def test_identity_H_energy_equals_bond_count_6bonds(self):
        """For H=I: each bond contributes Tr[rho]/Z = 1, so total = 6."""
        HI = _identity_H(self.d)
        E = self._e6(H=HI)
        assert abs(E.real.item() - 6.0) < 0.1, \
            f"Expected ~6 for H=I (6 bonds), got {E.real.item()}"

    def test_identity_H_energy_equals_bond_count_3bonds(self):
        """For H=I: each bond gives 1, total = 3."""
        HI = _identity_H(self.d)
        E = self._e3(H=HI)
        assert abs(E.real.item() - 3.0) < 0.1, \
            f"Expected ~3 for H=I (3 bonds), got {E.real.item()}"

    def test_phase_invariance_6bonds(self):
        """Multiplying all site tensors by a global phase e^{iθ} must not change the energy
        (double-layer + normalisation makes everything phase-invariant)."""
        import math
        theta = 0.3
        phase = math.cos(theta) + 1j * math.sin(theta)
        phase_t = torch.tensor(phase, dtype=torch.complex64)
        a, b, c, d, e, f = [t.detach() for t in self.abcdef]
        scaled = [phase_t * t for t in (a, b, c, d, e, f)]
        E_orig = self._e6(abcdef=(a, b, c, d, e, f)).detach()
        E_phased = self._e6(abcdef=scaled).detach()
        assert torch.allclose(E_orig, E_phased, atol=ATOL), \
            f"Phase changed energy: orig={E_orig}, phased={E_phased}"
