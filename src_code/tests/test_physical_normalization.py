"""Physical normalization tests (as demanded).

Two distinct normalizations are enforced:
1) In `trunc_rhoCCC`: the three truncated corners are scaled together by
   (sum(sv32[:chi]))^(1/3) where sv32 are the singular values of
   rho32 = matC13 @ matC32 @ matC21.
2) At the END of each update step: the six transfer tensors of the OUTPUT
   environment are rescaled by <iPEPS|iPEPS>^(1/6), where <iPEPS|iPEPS>
   is computed in the exact same `oe.contract("xy,yz,zx->", ...)` form as
   the corresponding `norm_env_*` in `tests/test_ctmrg_normalization.py`.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import opt_einsum as oe
import torch

from core_unrestricted import (
    initialize_abcdef,
    abcdef_to_ABCDEF,
    trunc_rhoCCC,
    initialize_environmentCTs_2,
    update_environmentCTs_1to2,
    update_environmentCTs_2to3,
    update_environmentCTs_3to1,
)

# reuse the exact norm contraction code (single-layer) from the diagnostic test
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tests.test_ctmrg_normalization import norm_env_1, norm_env_2, norm_env_3  # type: ignore


ATOL = 2e-2  # single-precision-ish contractions; allow some slack


def test_trunc_rhoCCC_corner_scale_uses_sum_sv32():
    torch.manual_seed(0)
    chi = 4
    D_sq = 4
    dim = chi * D_sq

    matC21 = torch.randn(dim, dim, dtype=torch.complex64)
    matC32 = torch.randn(dim, dim, dtype=torch.complex64)
    matC13 = torch.randn(dim, dim, dtype=torch.complex64)

    (_, C21, _, _, C32, _, _, C13, _) = trunc_rhoCCC(matC21, matC32, matC13, chi, D_sq)

    # With the implemented normalization, scaling all input matC's by s scales:
    # - each truncated corner C* by ~s (due to the explicit matC* factor), and
    # - the common scaleC by ~s (since rho32 scales like s^3 and scaleC is the
    #   cube root of sum(sv32[:chi])).
    # Therefore the OUTPUT corners should be approximately invariant.
    s = torch.tensor(3.7, dtype=torch.float32)
    (_, C21s, _, _, C32s, _, _, C13s, _) = trunc_rhoCCC(s * matC21, s * matC32, s * matC13, chi, D_sq)

    r21 = (torch.linalg.norm(C21s) / torch.linalg.norm(C21)).real
    r32 = (torch.linalg.norm(C32s) / torch.linalg.norm(C32)).real
    r13 = (torch.linalg.norm(C13s) / torch.linalg.norm(C13)).real

    assert abs(r21 - 1.0) < 1e-2
    assert abs(r32 - 1.0) < 1e-2
    assert abs(r13 - 1.0) < 1e-2


def test_update_steps_enforce_norm_env_1_2_3_to_one():
    torch.manual_seed(1)
    D_bond = 2
    D_sq = D_bond**2
    chi = 4

    # single-layer tensors for norm_env_*
    a, b, c, d, e, f = initialize_abcdef("random", D_bond, 2, 1e-3)
    # double-layer tensors for CTMRG updates
    A, B, C, D, E, F = abcdef_to_ABCDEF(a, b, c, d, e, f, D_sq)

    env1 = initialize_environmentCTs_2(A, B, C, D, E, F, chi, D_sq)
    env2 = update_environmentCTs_1to2(*env1, A, B, C, D, E, F, chi, D_sq)
    env3 = update_environmentCTs_2to3(*env2, A, B, C, D, E, F, chi, D_sq)
    env1b = update_environmentCTs_3to1(*env3, A, B, C, D, E, F, chi, D_sq)

    n1 = norm_env_1(a, b, c, d, e, f, chi, D_bond, *env1b)
    n2 = norm_env_2(a, b, c, d, e, f, chi, D_bond, *env2)
    n3 = norm_env_3(a, b, c, d, e, f, chi, D_bond, *env3)

    assert torch.allclose(n1.real, torch.tensor(1.0, dtype=n1.real.dtype), atol=ATOL)
    assert torch.allclose(n2.real, torch.tensor(1.0, dtype=n2.real.dtype), atol=ATOL)
    assert torch.allclose(n3.real, torch.tensor(1.0, dtype=n3.real.dtype), atol=ATOL)
