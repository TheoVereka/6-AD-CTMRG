"""
core_unres_new.py — Real-tensor honeycomb iPEPS with CTMRG
==========================================================
ALL tensors are real (float64 or float32). No complex arithmetic anywhere.
Key changes from core_unrestricted.py:
  - float64 throughout (no complex dtype)
  - a.conj() → a (real tensors are self-conjugate)
  - SVD-based truncation (real SVD, ~2x faster than complex)
  - Simple Frobenius-norm normalization for Cs and Ts
  - Hamiltonian built with real arithmetic (S·S = P_swap/2 - I⊗I/4)
"""

import torch
import opt_einsum as oe

# ── Precision control ─────────────────────────────────────────────────────────
DTYPE: torch.dtype = torch.float64


def set_dtype(use_double: bool) -> None:
    """Switch between float32 and float64. Call BEFORE allocating tensors."""
    global DTYPE
    DTYPE = torch.float64 if use_double else torch.float32


# ── Tensor utilities ──────────────────────────────────────────────────────────

def normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Normalize tensor to unit Frobenius norm. Returns unchanged if norm ≈ 0."""
    norm = torch.linalg.norm(tensor)
    if not torch.isfinite(norm) or norm < torch.finfo(tensor.dtype).tiny:
        return tensor
    return tensor / norm


# ── Hamiltonian ───────────────────────────────────────────────────────────────

def build_heisenberg_H(J: float = 1.0, d: int = 2) -> torch.Tensor:
    """Build real Heisenberg Hamiltonian H = J * S·S as a (d,d,d,d) tensor.

    Uses the identity S·S = (P_swap)/2 - (I⊗I)/4 for spin-1/2.
    Output convention: H[i,k,j,l] = <ik|H|jl> (matches einsum "ij,kl->ikjl").
    All elements are real because Sy⊗Sy is real despite Sy being imaginary.
    """
    Id = torch.eye(d, dtype=DTYPE)
    # P_swap[i,k,j,l] = delta(i,l) * delta(k,j)
    P_swap = torch.einsum("il,kj->ikjl", Id, Id)
    # (I⊗I)[i,k,j,l] = delta(i,j) * delta(k,l)
    IdxId = torch.einsum("ij,kl->ikjl", Id, Id)
    SdotS = 0.5 * P_swap - 0.25 * IdxId
    return J * SdotS


# ── Single-layer tensor initialization ────────────────────────────────────────

def initialize_abcdef(D_bond: int, d_phys: int = 2,
                      noise_scale: float = 1e-2) -> tuple:
    """Initialize 6 real single-layer iPEPS site tensors (random).

    Each tensor has shape (D_bond, D_bond, D_bond, d_phys).
    All normalized to unit Frobenius norm.

    Returns: (a, b, c, d, e, f) tuple of 6 tensors.
    """
    tensors = []
    for _ in range(6):
        t = noise_scale * torch.randn(D_bond, D_bond, D_bond, d_phys, dtype=DTYPE)
        tensors.append(normalize_tensor(t))
    return tuple(tensors)


# ── Double-layer construction ─────────────────────────────────────────────────

def abcdef_to_ABCDEF(a, b, c, d, e, f, D_sq: int) -> tuple:
    """Build double-layer tensors by contracting ket⊗bra over physical index.

    A_{(u,x),(v,y),(w,z)} = sum_p a_{u,v,w,p} * a_{x,y,z,p}
    then reshaped to (D², D², D²). Since a is real, a.conj() = a.

    Returns: (A, B, C, D_dl, E, F) — each shape (D², D², D²).
    Note: D_dl to avoid shadowing Python's built-in 'd'.
    """
    result = []
    for t in (a, b, c, d, e, f):
        T = oe.contract("uvwp,xyzp->uxvywz", t, t,
                         optimize=[(0, 1)], backend='torch')
        T = T.reshape(D_sq, D_sq, D_sq)
        result.append(T)
    return tuple(result)


# ── Truncation via real SVD ───────────────────────────────────────────────────

def trunc_rhoCCC(matC21, matC32, matC13, chi: int, D_sq: int):
    """Compute truncated CTM projectors and renormalized corner matrices.

    Given 3 grown corner matrices (each of shape (full_dim, full_dim)),
    builds rho = C_cyc product, does real SVD, keeps top-chi singular vectors
    as projectors, and projects corners to (chi, chi).

    The projectors are reshaped to 3-index form: U -> (dim1, D_sq, chi),
    V -> (chi, dim1, D_sq), where dim1 = full_dim // D_sq. This handles
    both steady-state (dim1=chi) and initialization (dim1=D_sq) cases.

    Returns: (V2, C21, U1, V3, C32, U2, V1, C13, U3,
              sv32_kept, sv13_kept, sv21_kept)
    """
    full_dim = matC21.shape[0]
    dim1 = full_dim // D_sq  # chi in steady state, D_sq during init step 2

    # rho32 = C13 @ C32 @ C21
    rho32 = oe.contract("UZ,ZY,YV->UV", matC13, matC32, matC21,
                         optimize=[(0, 1), (0, 1)], backend='torch')
    U3_full, sv32, V2_full = torch.linalg.svd(rho32)
    U3 = U3_full[:, :chi]
    V2 = V2_full[:chi, :]
    sv32_kept = sv32[:chi].detach().clone()

    # rho13 = C21 @ C13 @ C32
    rho13 = oe.contract("UX,XZ,ZV->UV", matC21, matC13, matC32,
                         optimize=[(0, 1), (0, 1)], backend='torch')
    U1_full, sv13, V3_full = torch.linalg.svd(rho13)
    U1 = U1_full[:, :chi]
    V3 = V3_full[:chi, :]
    sv13_kept = sv13[:chi].detach().clone()

    # rho21 = C32 @ C21 @ C13
    rho21 = oe.contract("UY,YX,XV->UV", matC32, matC21, matC13,
                         optimize=[(0, 1), (0, 1)], backend='torch')
    U2_full, sv21, V1_full = torch.linalg.svd(rho21)
    U2 = U2_full[:, :chi]
    V1 = V1_full[:chi, :]
    sv21_kept = sv21[:chi].detach().clone()

    # Project corners to (chi, chi)
    C21 = oe.contract("Yy,YX,xX->yx", U1, matC21, V2,
                       optimize=[(0, 1), (0, 1)], backend='torch')
    C32 = oe.contract("Zz,ZY,yY->zy", U2, matC32, V3,
                       optimize=[(0, 1), (0, 1)], backend='torch')
    C13 = oe.contract("Xx,XZ,zZ->xz", U3, matC13, V1,
                       optimize=[(0, 1), (0, 1)], backend='torch')

    # Simple Frobenius normalization of corners
    C21 = normalize_tensor(C21)
    C32 = normalize_tensor(C32)
    C13 = normalize_tensor(C13)

    # Reshape projectors dynamically: U -> (dim1, D_sq, chi), V -> (chi, dim1, D_sq)
    U1 = U1.reshape(dim1, D_sq, chi)
    U2 = U2.reshape(dim1, D_sq, chi)
    U3 = U3.reshape(dim1, D_sq, chi)
    V1 = V1.reshape(chi, dim1, D_sq)
    V2 = V2.reshape(chi, dim1, D_sq)
    V3 = V3.reshape(chi, dim1, D_sq)

    return V2, C21, U1, V3, C32, U2, V1, C13, U3, sv32_kept, sv13_kept, sv21_kept


# ── Environment initialization ────────────────────────────────────────────────

def initialize_env(A, B, C, D, E, F, chi: int, D_sq: int):
    """Bootstrap CTMRG environment from double-layer site tensors.

    Three-step init to reach steady-state T shape (chi, chi, D_sq):
      Step 1: seed type-3 → 3to1 grow+truncate → type-1, Ts (D_sq, chi_s, D_sq)
      Step 2: type-1 → 1to2 grow+truncate → type-2, Ts (chi_s, chi_s, D_sq)
      Step 3: type-2 → 2to3 grow+truncate → type-3, Ts (chi, chi, D_sq)
    chi_s = min(chi, D_sq²) to handle chi > D_sq².

    Returns: type-2 env (C21EB, C32AD, C13CF, T1D, T2C, T2F, T3E, T3B, T1A)
    """
    # Cap seed chi to max available rank from SVD of seed transfers
    chi_seed = min(chi, D_sq * D_sq)

    # Seed corners (D², D²) — type-3 naming
    C21AF = oe.contract("oyg,xog->yx", A, F, optimize=[(0, 1)], backend='torch')
    C32CB = oe.contract("aoz,ayo->zy", C, B, optimize=[(0, 1)], backend='torch')
    C13ED = oe.contract("xbo,obz->xz", E, D, optimize=[(0, 1)], backend='torch')

    # Seed transfer pairs (contract 4 sites → SVD split)
    T2ET3F = oe.contract("ubi,jki,jlk,vlg->ubvg", E, B, C, F,
                          optimize=[(1, 2), (0, 1), (0, 1)], backend='torch')
    T3AT1B = oe.contract("iug,ijk,kjl,avl->ugva", A, D, E, B,
                          optimize=[(1, 2), (0, 1), (0, 1)], backend='torch')
    T1CT2D = oe.contract("aiu,kij,lkj,lbv->uavb", C, F, A, D,
                          optimize=[(1, 2), (0, 1), (0, 1)], backend='torch')

    T2ET3F = T2ET3F.reshape(D_sq * D_sq, D_sq * D_sq)
    T3AT1B = T3AT1B.reshape(D_sq * D_sq, D_sq * D_sq)
    T1CT2D = T1CT2D.reshape(D_sq * D_sq, D_sq * D_sq)

    U2E, sv23, Vh3F = torch.linalg.svd(T2ET3F)
    U3A, sv31, Vh1B = torch.linalg.svd(T3AT1B)
    U1C, sv12, Vh2D = torch.linalg.svd(T1CT2D)

    sqrt_sv23 = torch.sqrt(torch.clamp(sv23[:chi_seed], min=1e-12))
    sqrt_sv31 = torch.sqrt(torch.clamp(sv31[:chi_seed], min=1e-12))
    sqrt_sv12 = torch.sqrt(torch.clamp(sv12[:chi_seed], min=1e-12))

    T2E = U2E[:, :chi_seed] @ torch.diag(sqrt_sv23)
    T3A = U3A[:, :chi_seed] @ torch.diag(sqrt_sv31)
    T1C = U1C[:, :chi_seed] @ torch.diag(sqrt_sv12)
    T3F = torch.diag(sqrt_sv23) @ Vh3F[:chi_seed, :]
    T1B = torch.diag(sqrt_sv31) @ Vh1B[:chi_seed, :]
    T2D = torch.diag(sqrt_sv12) @ Vh2D[:chi_seed, :]

    # Seed Ts: shape (chi_seed, D_sq, D_sq)
    T2E = T2E.reshape(D_sq, D_sq, chi_seed).permute(2, 0, 1)
    T3A = T3A.reshape(D_sq, D_sq, chi_seed).permute(2, 0, 1)
    T1C = T1C.reshape(D_sq, D_sq, chi_seed).permute(2, 0, 1)
    T3F = T3F.reshape(chi_seed, D_sq, D_sq)
    T1B = T1B.reshape(chi_seed, D_sq, D_sq)
    T2D = T2D.reshape(chi_seed, D_sq, D_sq)

    # Step 1: 3→1 with chi_seed (Ts: chi_seed,D_sq,D_sq → D_sq,chi_seed,D_sq)
    result_1 = update_env_3to1(
        C21AF, C32CB, C13ED, T1B, T2E, T2D, T3A, T3F, T1C,
        A, B, C, D, E, F, chi_seed, D_sq)

    # Step 2: 1→2 with chi_seed (Ts: D_sq,chi_seed,D_sq → chi_seed,chi_seed,D_sq)
    result_2 = update_env_1to2(
        result_1[0], result_1[1], result_1[2],
        result_1[3], result_1[4], result_1[5], result_1[6], result_1[7], result_1[8],
        A, B, C, D, E, F, chi_seed, D_sq)

    # Step 3: 2→3 with full chi (grown corner: chi_seed*D_sq × chi_seed*D_sq)
    # This grows the env bond dim from chi_seed to chi if chi > chi_seed.
    result_3 = update_env_2to3(
        result_2[0], result_2[1], result_2[2],
        result_2[3], result_2[4], result_2[5], result_2[6], result_2[7], result_2[8],
        A, B, C, D, E, F, chi, D_sq)

    # Step 4: 3→1 with full chi to get back to proper shape
    result_4 = update_env_3to1(
        result_3[0], result_3[1], result_3[2],
        result_3[3], result_3[4], result_3[5], result_3[6], result_3[7], result_3[8],
        A, B, C, D, E, F, chi, D_sq)

    # Step 5: 1→2 with full chi → type-2 env (chi, chi, D_sq)
    result_5 = update_env_1to2(
        result_4[0], result_4[1], result_4[2],
        result_4[3], result_4[4], result_4[5], result_4[6], result_4[7], result_4[8],
        A, B, C, D, E, F, chi, D_sq)

    return result_5[:9]


# ── Update functions ──────────────────────────────────────────────────────────

def _normalize_transfers(*Ts):
    """Normalize each transfer tensor by its Frobenius norm."""
    return tuple(normalize_tensor(T) for T in Ts)


def update_env_1to2(C21CD, C32EF, C13AB, T1F, T2A, T2B, T3C, T3D, T1E,
                    A, B, C, D, E, F, chi: int, D_sq: int):
    """CTMRG step: env type 1 → env type 2.

    Absorbs sites E,B,A,D,C,F into type-1 corners/transfers.
    Returns: (C21EB, C32AD, C13CF, T1D, T2C, T2F, T3E, T3B, T1A,
              sv32_kept, sv13_kept, sv21_kept)
    """
    # Grow corners (M,L from T's 1st dim; m,l from site dim D_sq)
    matC21EB = oe.contract("YX,MYa,LXb,amg,lbg->MmLl",
                            C21CD, T1F, T2A, E, B,
                            optimize=[(0, 2), (0, 3), (1, 2), (0, 1)],
                            backend='torch')
    grown_dim = T1F.shape[0] * D_sq
    matC21EB = matC21EB.reshape(grown_dim, grown_dim)

    matC32AD = oe.contract("ZY,NZb,MYg,abn,amg->NnMm",
                            C32EF, T2B, T3C, A, D,
                            optimize=[(0, 2), (0, 3), (1, 2), (0, 1)],
                            backend='torch')
    matC32AD = matC32AD.reshape(grown_dim, grown_dim)

    matC13CF = oe.contract("XZ,LXg,NZa,lbg,abn->LlNn",
                            C13AB, T3D, T1E, C, F,
                            optimize=[(0, 2), (0, 3), (1, 2), (0, 1)],
                            backend='torch')
    matC13CF = matC13CF.reshape(grown_dim, grown_dim)

    # Truncate
    V2A, C21EB, U1F, V3C, C32AD, U2B, V1E, C13CF, U3D, sv32, sv13, sv21 = \
        trunc_rhoCCC(matC21EB, matC32AD, matC13CF, chi, D_sq)

    # Project transfers
    T3E = oe.contract("OYa,abg,ObM->YMg", T1F, E, U1F,
                       optimize=[(0, 1), (0, 1)], backend='torch')
    T3B = oe.contract("OXb,abg,LOa->XLg", T2A, B, V2A,
                       optimize=[(0, 1), (0, 1)], backend='torch')
    T1A = oe.contract("OZb,abg,OgN->ZNa", T2B, A, U2B,
                       optimize=[(0, 1), (0, 1)], backend='torch')
    T1D = oe.contract("OYg,abg,MOb->YMa", T3C, D, V3C,
                       optimize=[(0, 1), (0, 1)], backend='torch')
    T2C = oe.contract("OXg,abg,OaL->XLb", T3D, C, U3D,
                       optimize=[(0, 1), (0, 1)], backend='torch')
    T2F = oe.contract("OZa,abg,NOg->ZNa", T1E, F, V1E,
                       optimize=[(0, 1), (0, 1)], backend='torch')

    # Simple normalization
    T3E, T3B, T1A, T1D, T2C, T2F = _normalize_transfers(T3E, T3B, T1A, T1D, T2C, T2F)

    return C21EB, C32AD, C13CF, T1D, T2C, T2F, T3E, T3B, T1A, sv32, sv13, sv21


def update_env_2to3(C21EB, C32AD, C13CF, T1D, T2C, T2F, T3E, T3B, T1A,
                    A, B, C, D, E, F, chi: int, D_sq: int):
    """CTMRG step: env type 2 → env type 3.

    Absorbs sites A,F,C,B,E,D into type-2 corners/transfers.
    Returns: (C21AF, C32CB, C13ED, T1B, T2E, T2D, T3A, T3F, T1C,
              sv32_kept, sv13_kept, sv21_kept)
    """
    matC21AF = oe.contract("YX,MYa,LXb,amg,lbg->MmLl",
                            C21EB, T1D, T2C, A, F,
                            optimize=[(0, 2), (0, 3), (1, 2), (0, 1)],
                            backend='torch')
    grown_dim = T1D.shape[0] * D_sq
    matC21AF = matC21AF.reshape(grown_dim, grown_dim)

    matC32CB = oe.contract("ZY,NZb,MYg,abn,amg->NnMm",
                            C32AD, T2F, T3E, C, B,
                            optimize=[(0, 2), (0, 3), (1, 2), (0, 1)],
                            backend='torch')
    matC32CB = matC32CB.reshape(grown_dim, grown_dim)

    matC13ED = oe.contract("XZ,LXg,NZa,lbg,abn->LlNn",
                            C13CF, T3B, T1A, E, D,
                            optimize=[(0, 2), (0, 3), (1, 2), (0, 1)],
                            backend='torch')
    matC13ED = matC13ED.reshape(grown_dim, grown_dim)

    V2C, C21AF, U1D, V3E, C32CB, U2F, V1A, C13ED, U3B, sv32, sv13, sv21 = \
        trunc_rhoCCC(matC21AF, matC32CB, matC13ED, chi, D_sq)

    T3A = oe.contract("OYa,abg,ObM->YMg", T1D, A, U1D,
                       optimize=[(0, 1), (0, 1)], backend='torch')
    T3F = oe.contract("OXb,abg,LOa->XLg", T2C, F, V2C,
                       optimize=[(0, 1), (0, 1)], backend='torch')
    T1C = oe.contract("OZb,abg,OgN->ZNa", T2F, C, U2F,
                       optimize=[(0, 1), (0, 1)], backend='torch')
    T1B = oe.contract("OYg,abg,MOb->YMa", T3E, B, V3E,
                       optimize=[(0, 1), (0, 1)], backend='torch')
    T2E = oe.contract("OXg,abg,OaL->XLb", T3B, E, U3B,
                       optimize=[(0, 1), (0, 1)], backend='torch')
    T2D = oe.contract("OZa,abg,NOg->ZNa", T1A, D, V1A,
                       optimize=[(0, 1), (0, 1)], backend='torch')

    T3A, T3F, T1C, T1B, T2E, T2D = _normalize_transfers(T3A, T3F, T1C, T1B, T2E, T2D)

    return C21AF, C32CB, C13ED, T1B, T2E, T2D, T3A, T3F, T1C, sv32, sv13, sv21


def update_env_3to1(C21AF, C32CB, C13ED, T1B, T2E, T2D, T3A, T3F, T1C,
                    A, B, C, D, E, F, chi: int, D_sq: int):
    """CTMRG step: env type 3 → env type 1.

    Absorbs sites C,D,E,F,A,B into type-3 corners/transfers.
    Returns: (C21CD, C32EF, C13AB, T1F, T2A, T2B, T3C, T3D, T1E,
              sv32_kept, sv13_kept, sv21_kept)
    """
    matC21CD = oe.contract("YX,MYa,LXb,amg,lbg->MmLl",
                            C21AF, T1B, T2E, C, D,
                            optimize=[(0, 2), (0, 3), (1, 2), (0, 1)],
                            backend='torch')
    grown_dim = T1B.shape[0] * D_sq
    matC21CD = matC21CD.reshape(grown_dim, grown_dim)

    matC32EF = oe.contract("ZY,NZb,MYg,abn,amg->NnMm",
                            C32CB, T2D, T3A, E, F,
                            optimize=[(0, 2), (0, 3), (1, 2), (0, 1)],
                            backend='torch')
    matC32EF = matC32EF.reshape(grown_dim, grown_dim)

    matC13AB = oe.contract("XZ,LXg,NZa,lbg,abn->LlNn",
                            C13ED, T3F, T1C, A, B,
                            optimize=[(0, 2), (0, 3), (1, 2), (0, 1)],
                            backend='torch')
    matC13AB = matC13AB.reshape(grown_dim, grown_dim)

    V2E, C21CD, U1B, V3A, C32EF, U2D, V1C, C13AB, U3F, sv32, sv13, sv21 = \
        trunc_rhoCCC(matC21CD, matC32EF, matC13AB, chi, D_sq)

    T3C = oe.contract("OYa,abg,ObM->YMg", T1B, C, U1B,
                       optimize=[(0, 1), (0, 1)], backend='torch')
    T3D = oe.contract("OXb,abg,LOa->XLg", T2E, D, V2E,
                       optimize=[(0, 1), (0, 1)], backend='torch')
    T1E = oe.contract("OZb,abg,OgN->ZNa", T2D, E, U2D,
                       optimize=[(0, 1), (0, 1)], backend='torch')
    T1F = oe.contract("OYg,abg,MOb->YMa", T3A, F, V3A,
                       optimize=[(0, 1), (0, 1)], backend='torch')
    T2A = oe.contract("OXg,abg,OaL->XLb", T3F, A, U3F,
                       optimize=[(0, 1), (0, 1)], backend='torch')
    T2B = oe.contract("OZa,abg,NOg->ZNa", T1C, B, V1C,
                       optimize=[(0, 1), (0, 1)], backend='torch')

    T3C, T3D, T1E, T1F, T2A, T2B = _normalize_transfers(T3C, T3D, T1E, T1F, T2A, T2B)

    return C21CD, C32EF, C13AB, T1F, T2A, T2B, T3C, T3D, T1E, sv32, sv13, sv21


# ── CTMRG main loop ──────────────────────────────────────────────────────────

def _sv_spectrum_converged(old_svs, new_svs, threshold: float) -> bool:
    """Check if SV spectra have converged between iterations.

    Compares normalized SV spectra from truncation. Returns True if
    the max relative change across all 9 SV vectors is below threshold.
    """
    if old_svs is None:
        return False
    max_diff = 0.0
    for old, new in zip(old_svs, new_svs):
        # Normalize each spectrum by its sum
        old_sum = old.sum()
        new_sum = new.sum()
        if old_sum < 1e-30 or new_sum < 1e-30:
            continue
        old_n = old / old_sum
        new_n = new / new_sum
        diff = (old_n - new_n).abs().max().item()
        max_diff = max(max_diff, diff)
    return max_diff < threshold


def ctmrg(A, B, C, D, E, F, chi: int, D_sq: int,
          max_steps: int = 100, conv_thr: float = 1e-7) -> tuple:
    """Run CTMRG until convergence or max_steps reached.

    Each "step" is a full 1→2→3→1 cycle. Returns all 27 environment tensors
    plus the number of steps taken.

    Returns: (env1_tensors, env2_tensors, env3_tensors, ctm_steps)
        env1 = (C21CD, C32EF, C13AB, T1F, T2A, T2B, T3C, T3D, T1E)
        env2 = (C21EB, C32AD, C13CF, T1D, T2C, T2F, T3E, T3B, T1A)
        env3 = (C21AF, C32CB, C13ED, T1B, T2E, T2D, T3A, T3F, T1C)
    """
    # Initialize — returns type-2 env with Ts in steady-state shape (chi, chi, D_sq)
    env2_init = initialize_env(A, B, C, D, E, F, chi, D_sq)
    C21EB, C32AD, C13CF, T1D, T2C, T2F, T3E, T3B, T1A = env2_init

    # Bootstrap: run 2→3 and 3→1 to get all 3 env types
    result_3 = update_env_2to3(C21EB, C32AD, C13CF, T1D, T2C, T2F, T3E, T3B, T1A,
                                A, B, C, D, E, F, chi, D_sq)
    C21AF, C32CB, C13ED, T1B, T2E, T2D, T3A, T3F, T1C = result_3[:9]

    result_1 = update_env_3to1(C21AF, C32CB, C13ED, T1B, T2E, T2D, T3A, T3F, T1C,
                                A, B, C, D, E, F, chi, D_sq)
    C21CD, C32EF, C13AB, T1F, T2A, T2B, T3C, T3D, T1E = result_1[:9]

    # Main loop
    old_svs = None
    ctm_steps = max_steps
    for step in range(max_steps):
        # 3→1
        result_1 = update_env_3to1(C21AF, C32CB, C13ED, T1B, T2E, T2D, T3A, T3F, T1C,
                                    A, B, C, D, E, F, chi, D_sq)
        C21CD, C32EF, C13AB, T1F, T2A, T2B, T3C, T3D, T1E = result_1[:9]
        svs_1 = result_1[9:12]

        # 1→2
        result_2 = update_env_1to2(C21CD, C32EF, C13AB, T1F, T2A, T2B, T3C, T3D, T1E,
                                    A, B, C, D, E, F, chi, D_sq)
        C21EB, C32AD, C13CF, T1D, T2C, T2F, T3E, T3B, T1A = result_2[:9]
        svs_2 = result_2[9:12]

        # 2→3
        result_3 = update_env_2to3(C21EB, C32AD, C13CF, T1D, T2C, T2F, T3E, T3B, T1A,
                                    A, B, C, D, E, F, chi, D_sq)
        C21AF, C32CB, C13ED, T1B, T2E, T2D, T3A, T3F, T1C = result_3[:9]
        svs_3 = result_3[9:12]

        # Convergence check: compare all 9 SV spectra
        new_svs = svs_1 + svs_2 + svs_3  # tuple of 9 tensors
        if _sv_spectrum_converged(old_svs, new_svs, conv_thr):
            ctm_steps = step + 1
            break
        old_svs = new_svs

    env1 = (C21CD, C32EF, C13AB, T1F, T2A, T2B, T3C, T3D, T1E)
    env2 = (C21EB, C32AD, C13CF, T1D, T2C, T2F, T3E, T3B, T1A)
    env3 = (C21AF, C32CB, C13ED, T1B, T2E, T2D, T3A, T3F, T1C)

    return env1, env2, env3, ctm_steps


# ── Energy functions ──────────────────────────────────────────────────────────

def _build_open_closed(a, b, c, d, e, f, chi, D_bond, Cs_and_Ts):
    """Build open (with phys indices) and closed (traced) tensors for one env type.

    The pattern is the same for all 3 env types — only which sites go where differs.
    This is a helper called by the individual energy functions.

    Args:
        a..f: single-layer tensors (D, D, D, d_phys)
        Cs_and_Ts: (C21, C32, C13, T_clockwise_pairs...) — the 3 corners + 6 transfers
        The mapping from position to site is hardcoded per env type.

    Returns: (open_1..6, closed_1..6) as dicts keyed by site label.
    """
    # This will be implemented per-energy function since the site assignments differ
    pass


def energy_3ebadcf(a, b, c, d, e, f, Heb, Had, Hcf,
                   chi: int, D_bond: int,
                   C21CD, C32EF, C13AB, T1F, T2A, T2B, T3C, T3D, T1E):
    """Energy of 3 bonds (EB, AD, CF) using type-1 environment.

    Returns: scalar energy (sum of 3 bond energies / norm).
    """
    # Permute H from (bra1,bra2,ket1,ket2) to (ket1,bra1,ket2,bra2)
    Heb = Heb.permute(2, 0, 3, 1)
    Had = Had.permute(2, 0, 3, 1)
    Hcf = Hcf.permute(2, 0, 3, 1)
    d2 = D_bond * D_bond  # = D_sq, for reshaping Ts
    T1F4 = T1F.reshape(chi, chi, D_bond, D_bond)
    T2A4 = T2A.reshape(chi, chi, D_bond, D_bond)
    T2B4 = T2B.reshape(chi, chi, D_bond, D_bond)
    T3C4 = T3C.reshape(chi, chi, D_bond, D_bond)
    T3D4 = T3D.reshape(chi, chi, D_bond, D_bond)
    T1E4 = T1E.reshape(chi, chi, D_bond, D_bond)

    # Open tensors: contract single-layer ket and bra with environment
    # For real tensors, bra = ket (no conjugation)
    open_E = oe.contract("YX,MYar,abci,rstj->MbsXctij",
                          C21CD, T1F4, e, e, optimize=[(0, 1), (0, 1), (0, 1)], backend="torch")
    open_D = oe.contract("MYct,abci,rstj->YarMbsij",
                          T3C4, d, d, optimize=[(0, 1), (0, 1)], backend="torch")
    open_A = oe.contract("ZY,NZbs,abci,rstj->NctYarij",
                          C32EF, T2B4, a, a, optimize=[(0, 1), (0, 1), (0, 1)], backend="torch")
    open_F = oe.contract("NZar,abci,rstj->ZbsNctij",
                          T1E4, f, f, optimize=[(0, 1), (0, 1)], backend="torch")
    open_C = oe.contract("XZ,LXct,abci,rstj->LarZbsij",
                          C13AB, T3D4, c, c, optimize=[(0, 1), (0, 1), (0, 1)], backend="torch")
    open_B = oe.contract("LXbs,abci,rstj->XctLarij",
                          T2A4, b, b, optimize=[(0, 1), (0, 1)], backend="torch")

    # Closed tensors: trace over physical indices
    cDD = D_bond * D_bond  # chi*D*D for reshape
    closed_E = oe.contract("MbsXctii->MbsXct", open_E).reshape(chi * cDD, chi * cDD)
    closed_D = oe.contract("YarMbsii->YarMbs", open_D).reshape(chi * cDD, chi * cDD)
    closed_A = oe.contract("NctYarii->NctYar", open_A).reshape(chi * cDD, chi * cDD)
    closed_F = oe.contract("ZbsNctii->ZbsNct", open_F).reshape(chi * cDD, chi * cDD)
    closed_C = oe.contract("LarZbsii->LarZbs", open_C).reshape(chi * cDD, chi * cDD)
    closed_B = oe.contract("XctLarii->XctLar", open_B).reshape(chi * cDD, chi * cDD)

    # Hamiltonian insertions for 3 bonds
    H_AD = oe.contract("NctYarij,ijkl,YarMbskl->NctMbs",
                        open_A, Had, open_D, optimize=[(0, 1), (0, 1)], backend="torch"
                        ).reshape(chi * cDD, chi * cDD)
    H_CF = oe.contract("LarZbsij,ijkl,ZbsNctkl->LarNct",
                        open_C, Hcf, open_F, optimize=[(0, 1), (0, 1)], backend="torch"
                        ).reshape(chi * cDD, chi * cDD)
    H_EB = oe.contract("MbsXctij,ijkl,XctLarkl->MbsLar",
                        open_E, Heb, open_B, optimize=[(0, 1), (0, 1)], backend="torch"
                        ).reshape(chi * cDD, chi * cDD)

    # Pair products for norm
    AD = torch.mm(closed_A, closed_D)
    CF = torch.mm(closed_C, closed_F)
    EB = torch.mm(closed_E, closed_B)

    # Energies: Tr(H_bond · other_two_pairs)
    E_AD = oe.contract("xy,yz,zx->", H_AD, EB, CF, backend="torch")
    E_CF = oe.contract("xy,yz,zx->", H_CF, AD, EB, backend="torch")
    E_EB = oe.contract("xy,yz,zx->", H_EB, CF, AD, backend="torch")

    norm = oe.contract("xy,yz,zx->", AD, EB, CF, backend="torch")

    return (E_AD + E_CF + E_EB) / norm


def energy_3afcbed(a, b, c, d, e, f, Haf, Hcb, Hed,
                   chi: int, D_bond: int,
                   C21EB, C32AD, C13CF, T1D, T2C, T2F, T3E, T3B, T1A):
    """Energy of 3 bonds (AF, CB, ED) using type-2 environment."""
    # Permute H from (bra1,bra2,ket1,ket2) to (ket1,bra1,ket2,bra2)
    Haf = Haf.permute(2, 0, 3, 1)
    Hcb = Hcb.permute(2, 0, 3, 1)
    Hed = Hed.permute(2, 0, 3, 1)
    cDD = D_bond * D_bond
    T1D4 = T1D.reshape(chi, chi, D_bond, D_bond)
    T2C4 = T2C.reshape(chi, chi, D_bond, D_bond)
    T2F4 = T2F.reshape(chi, chi, D_bond, D_bond)
    T3E4 = T3E.reshape(chi, chi, D_bond, D_bond)
    T3B4 = T3B.reshape(chi, chi, D_bond, D_bond)
    T1A4 = T1A.reshape(chi, chi, D_bond, D_bond)

    open_A = oe.contract("YX,MYar,abci,rstj->MbsXctij",
                          C21EB, T1D4, a, a, optimize=[(0, 1), (0, 1), (0, 1)], backend="torch")
    open_B = oe.contract("MYct,abci,rstj->YarMbsij",
                          T3E4, b, b, optimize=[(0, 1), (0, 1)], backend="torch")
    open_C = oe.contract("ZY,NZbs,abci,rstj->NctYarij",
                          C32AD, T2F4, c, c, optimize=[(0, 1), (0, 1), (0, 1)], backend="torch")
    open_D = oe.contract("NZar,abci,rstj->ZbsNctij",
                          T1A4, d, d, optimize=[(0, 1), (0, 1)], backend="torch")
    open_E = oe.contract("XZ,LXct,abci,rstj->LarZbsij",
                          C13CF, T3B4, e, e, optimize=[(0, 1), (0, 1), (0, 1)], backend="torch")
    open_F = oe.contract("LXbs,abci,rstj->XctLarij",
                          T2C4, f, f, optimize=[(0, 1), (0, 1)], backend="torch")

    closed_A = oe.contract("MbsXctii->MbsXct", open_A).reshape(chi * cDD, chi * cDD)
    closed_B = oe.contract("YarMbsii->YarMbs", open_B).reshape(chi * cDD, chi * cDD)
    closed_C = oe.contract("NctYarii->NctYar", open_C).reshape(chi * cDD, chi * cDD)
    closed_D = oe.contract("ZbsNctii->ZbsNct", open_D).reshape(chi * cDD, chi * cDD)
    closed_E = oe.contract("LarZbsii->LarZbs", open_E).reshape(chi * cDD, chi * cDD)
    closed_F = oe.contract("XctLarii->XctLar", open_F).reshape(chi * cDD, chi * cDD)

    H_CB = oe.contract("NctYarij,ijkl,YarMbskl->NctMbs",
                        open_C, Hcb, open_B, optimize=[(0, 1), (0, 1)], backend="torch"
                        ).reshape(chi * cDD, chi * cDD)
    H_ED = oe.contract("LarZbsij,ijkl,ZbsNctkl->LarNct",
                        open_E, Hed, open_D, optimize=[(0, 1), (0, 1)], backend="torch"
                        ).reshape(chi * cDD, chi * cDD)
    H_AF = oe.contract("MbsXctij,ijkl,XctLarkl->MbsLar",
                        open_A, Haf, open_F, optimize=[(0, 1), (0, 1)], backend="torch"
                        ).reshape(chi * cDD, chi * cDD)

    CB = torch.mm(closed_C, closed_B)
    ED = torch.mm(closed_E, closed_D)
    AF = torch.mm(closed_A, closed_F)

    E_CB = oe.contract("xy,yz,zx->", H_CB, AF, ED, backend="torch")
    E_ED = oe.contract("xy,yz,zx->", H_ED, CB, AF, backend="torch")
    E_AF = oe.contract("xy,yz,zx->", H_AF, ED, CB, backend="torch")

    norm = oe.contract("xy,yz,zx->", CB, AF, ED, backend="torch")

    return (E_CB + E_ED + E_AF) / norm


def energy_other_3(a, b, c, d, e, f, Hcd, Hef, Hab,
                   chi: int, D_bond: int,
                   C21AF, C32CB, C13ED, T1B, T2E, T2D, T3A, T3F, T1C):
    """Energy of 3 bonds (CD, EF, AB) using type-3 environment."""
    # Permute H from (bra1,bra2,ket1,ket2) to (ket1,bra1,ket2,bra2)
    Hcd = Hcd.permute(2, 0, 3, 1)
    Hef = Hef.permute(2, 0, 3, 1)
    Hab = Hab.permute(2, 0, 3, 1)
    cDD = D_bond * D_bond
    T1B4 = T1B.reshape(chi, chi, D_bond, D_bond)
    T2E4 = T2E.reshape(chi, chi, D_bond, D_bond)
    T2D4 = T2D.reshape(chi, chi, D_bond, D_bond)
    T3A4 = T3A.reshape(chi, chi, D_bond, D_bond)
    T3F4 = T3F.reshape(chi, chi, D_bond, D_bond)
    T1C4 = T1C.reshape(chi, chi, D_bond, D_bond)

    open_C = oe.contract("YX,MYar,abci,rstj->MbsXctij",
                          C21AF, T1B4, c, c, optimize=[(0, 1), (0, 1), (0, 1)], backend="torch")
    open_F = oe.contract("MYct,abci,rstj->YarMbsij",
                          T3A4, f, f, optimize=[(0, 1), (0, 1)], backend="torch")
    open_E = oe.contract("ZY,NZbs,abci,rstj->NctYarij",
                          C32CB, T2D4, e, e, optimize=[(0, 1), (0, 1), (0, 1)], backend="torch")
    open_B = oe.contract("NZar,abci,rstj->ZbsNctij",
                          T1C4, b, b, optimize=[(0, 1), (0, 1)], backend="torch")
    open_A = oe.contract("XZ,LXct,abci,rstj->LarZbsij",
                          C13ED, T3F4, a, a, optimize=[(0, 1), (0, 1), (0, 1)], backend="torch")
    open_D = oe.contract("LXbs,abci,rstj->XctLarij",
                          T2E4, d, d, optimize=[(0, 1), (0, 1)], backend="torch")

    closed_C = oe.contract("MbsXctii->MbsXct", open_C).reshape(chi * cDD, chi * cDD)
    closed_F = oe.contract("YarMbsii->YarMbs", open_F).reshape(chi * cDD, chi * cDD)
    closed_E = oe.contract("NctYarii->NctYar", open_E).reshape(chi * cDD, chi * cDD)
    closed_B = oe.contract("ZbsNctii->ZbsNct", open_B).reshape(chi * cDD, chi * cDD)
    closed_A = oe.contract("LarZbsii->LarZbs", open_A).reshape(chi * cDD, chi * cDD)
    closed_D = oe.contract("XctLarii->XctLar", open_D).reshape(chi * cDD, chi * cDD)

    H_EF = oe.contract("NctYarij,ijkl,YarMbskl->NctMbs",
                        open_E, Hef, open_F, optimize=[(0, 1), (0, 1)], backend="torch"
                        ).reshape(chi * cDD, chi * cDD)
    H_AB = oe.contract("LarZbsij,ijkl,ZbsNctkl->LarNct",
                        open_A, Hab, open_B, optimize=[(0, 1), (0, 1)], backend="torch"
                        ).reshape(chi * cDD, chi * cDD)
    H_CD = oe.contract("MbsXctij,ijkl,XctLarkl->MbsLar",
                        open_C, Hcd, open_D, optimize=[(0, 1), (0, 1)], backend="torch"
                        ).reshape(chi * cDD, chi * cDD)

    EF = torch.mm(closed_E, closed_F)
    AB = torch.mm(closed_A, closed_B)
    CD = torch.mm(closed_C, closed_D)

    E_EF = oe.contract("xy,yz,zx->", H_EF, CD, AB, backend="torch")
    E_AB = oe.contract("xy,yz,zx->", H_AB, EF, CD, backend="torch")
    E_CD = oe.contract("xy,yz,zx->", H_CD, AB, EF, backend="torch")

    norm = oe.contract("xy,yz,zx->", EF, CD, AB, backend="torch")

    return (E_EF + E_AB + E_CD) / norm


# ── Optimization ──────────────────────────────────────────────────────────────

def optimize_ipeps(H, chi: int, D_bond: int, d_phys: int = 2,
                   max_ctm_steps: int = 100, ctm_conv_thr: float = 1e-7,
                   max_opt_steps: int = 200, lbfgs_max_iter: int = 30,
                   lbfgs_lr: float = 1.0, lbfgs_history: int = 30,
                   opt_tol_grad: float = 1e-7, opt_tol_change: float = 1e-9,
                   opt_conv_thr: float = 1e-8,
                   init_abcdef=None, verbose: bool = True):
    """Optimize iPEPS tensors a..f using alternating CTMRG + L-BFGS.

    Args:
        H: Heisenberg Hamiltonian tensor, shape (d, d, d, d).
           Used as H for all 9 bonds (isotropic model).
        chi: environment bond dimension.
        D_bond: virtual bond dimension.
        d_phys: physical dimension (default 2 for spin-1/2).
        max_ctm_steps: max CTMRG iterations per environment convergence.
        ctm_conv_thr: CTMRG convergence threshold.
        max_opt_steps: max outer optimization steps.
        lbfgs_max_iter: max L-BFGS sub-iterations per outer step.
        lbfgs_lr: L-BFGS learning rate (step size seed).
        lbfgs_history: L-BFGS history size.
        opt_tol_grad: L-BFGS gradient tolerance.
        opt_tol_change: L-BFGS loss change tolerance.
        opt_conv_thr: outer convergence threshold on |delta_loss|.
        init_abcdef: optional initial tensors (tuple of 6).
        verbose: print progress.

    Returns: (a, b, c, d_t, e, f, loss_history)
    """
    D_sq = D_bond ** 2

    # Initialize tensors
    if init_abcdef is not None:
        a, b, c, d_t, e, f = [t.detach().clone().to(DTYPE) for t in init_abcdef]
    else:
        a, b, c, d_t, e, f = initialize_abcdef(D_bond, d_phys)

    a.requires_grad_(True)
    b.requires_grad_(True)
    c.requires_grad_(True)
    d_t.requires_grad_(True)
    e.requires_grad_(True)
    f.requires_grad_(True)

    prev_loss = None
    loss_history = []

    for opt_step in range(max_opt_steps):
        # Normalize tensors (scale redundancy)
        with torch.no_grad():
            a.data = normalize_tensor(a.data)
            b.data = normalize_tensor(b.data)
            c.data = normalize_tensor(c.data)
            d_t.data = normalize_tensor(d_t.data)
            e.data = normalize_tensor(e.data)
            f.data = normalize_tensor(f.data)

        # Fresh L-BFGS (env changes each step → old curvature stale)
        optimizer = torch.optim.LBFGS(
            [a, b, c, d_t, e, f],
            lr=lbfgs_lr,
            max_iter=lbfgs_max_iter,
            tolerance_grad=opt_tol_grad,
            tolerance_change=opt_tol_change,
            history_size=lbfgs_history,
            line_search_fn='strong_wolfe',
        )

        # Converge CTMRG environment (no grad)
        with torch.no_grad():
            A, B, C_dl, D_dl, E, F_dl = abcdef_to_ABCDEF(a, b, c, d_t, e, f, D_sq)
            env1, env2, env3, ctm_steps = ctmrg(
                A, B, C_dl, D_dl, E, F_dl, chi, D_sq, max_ctm_steps, ctm_conv_thr)

        # Unpack environments
        C21CD, C32EF, C13AB, T1F, T2A, T2B, T3C, T3D, T1E = env1
        C21EB, C32AD, C13CF, T1D, T2C, T2F, T3E, T3B, T1A = env2
        C21AF, C32CB, C13ED, T1B, T2E, T2D, T3A, T3F, T1C = env3

        # L-BFGS closure
        def closure():
            optimizer.zero_grad()
            loss = (
                energy_3ebadcf(a, b, c, d_t, e, f, H, H, H,
                               chi, D_bond,
                               C21CD, C32EF, C13AB, T1F, T2A, T2B, T3C, T3D, T1E)
                + energy_3afcbed(a, b, c, d_t, e, f, H, H, H,
                                 chi, D_bond,
                                 C21EB, C32AD, C13CF, T1D, T2C, T2F, T3E, T3B, T1A)
                + energy_other_3(a, b, c, d_t, e, f, H, H, H,
                                 chi, D_bond,
                                 C21AF, C32CB, C13ED, T1B, T2E, T2D, T3A, T3F, T1C)
            )
            loss.backward()
            return loss

        # Run L-BFGS
        loss_val = optimizer.step(closure)
        loss_item = loss_val.item()
        loss_history.append(loss_item)

        delta = (loss_item - prev_loss) if prev_loss is not None else float('inf')
        if verbose:
            print(f"  opt {opt_step:4d}  loss={loss_item:+.10f}  Δ={delta:.3e}  CTM={ctm_steps}")

        if prev_loss is not None and abs(delta) < opt_conv_thr:
            if verbose:
                print(f"  Converged at step {opt_step} (|Δ|={abs(delta):.3e} < {opt_conv_thr:.3e})")
            break
        prev_loss = loss_item

    return a, b, c, d_t, e, f, loss_history
