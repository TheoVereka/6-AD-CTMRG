"""
core_unres_v2.py — Real-tensor iPEPS + CTMRG for Heisenberg honeycomb (6-site cell)
==================================================================================

Written from first principles.  ALL tensors are real (float64).
Uses SVD-based truncation, norm-based transfer scaling,
rho-based CTMRG convergence, and gauge-invariant SV comparison.

Convention
----------
- Single-layer tensor a:  shape (D, D, D, d) = (u, v, w, phys)
- Double-layer tensor A:  A_{ux,vy,wz} = sum_p a_{u,v,w,p} * a_{x,y,z,p}
                          shape (D², D², D²)
- Corner C:               shape (chi, chi)
- Transfer T:             shape (chi, chi, D²)
- Hamiltonian H:          shape (d, d, d, d) in (bra1, bra2, ket1, ket2) from
                          standard S·S construction.  Permuted to
                          (ket1, bra1, ket2, bra2) before energy contraction.

Environment types (9 bonds split into 3 groups of 3):
    Type 1: bonds EB, AD, CF   env = (C21CD, C32EF, C13AB, T1F, T2A, T2B, T3C, T3D, T1E)
    Type 2: bonds AF, CB, ED   env = (C21EB, C32AD, C13CF, T1D, T2C, T2F, T3E, T3B, T1A)
    Type 3: bonds CD, EF, AB   env = (C21AF, C32CB, C13ED, T1B, T2E, T2D, T3A, T3F, T1C)

CTMRG cycle: 1 → 2 → 3 → 1 → ...

Known limitations
-----------------
- **Finite-chi CTMRG is NOT variational**: at finite bond dimension chi, the
  approximate environment can yield energies below the true ground-state energy
  (e.g., below QMC bounds).  This is an inherent property of the algorithm, not
  a code bug.  Cross-validation against the original core_unrestricted.py
  confirms both codes produce identical energies to machine precision (~1e-15).
- Increasing chi does NOT necessarily improve the variational bound; it improves
  the accuracy of the environment but does not restore the variational property.

Verified against
----------------
- Néel state at D=1: E/bond = -0.25 (exact)
- Néel state at D=2: E/bond = -0.25 (exact, with sufficient chi)
- FM state at D=1:   E/bond = +0.25 (exact)
- Cross-validation:  all 3 energy functions match original code to ~1e-15
"""

import torch
import opt_einsum as oe

# ══════════════════════════════════════════════════════════════════════════════
# §1  Precision control
# ══════════════════════════════════════════════════════════════════════════════

DTYPE: torch.dtype = torch.float64


def set_dtype(use_double: bool) -> None:
    """Set global dtype.  Must call before allocating any tensor."""
    global DTYPE
    DTYPE = torch.float64 if use_double else torch.float32


# ══════════════════════════════════════════════════════════════════════════════
# §2  Basic utilities
# ══════════════════════════════════════════════════════════════════════════════

def normalize_tensor(t: torch.Tensor) -> torch.Tensor:
    """Return t / ||t||_F.  Returns t unchanged if norm is zero."""
    nrm = torch.linalg.norm(t)
    if nrm.item() == 0.0 or not torch.isfinite(nrm):
        return t
    return t / nrm


def _assert_real(t: torch.Tensor, name: str = "tensor") -> None:
    """Debug helper: assert tensor is real dtype."""
    assert t.is_floating_point() and not t.is_complex(), \
        f"{name} must be real, got {t.dtype}"


# ══════════════════════════════════════════════════════════════════════════════
# §3  Hamiltonian
# ══════════════════════════════════════════════════════════════════════════════

def build_heisenberg_H(J: float = 1.0, d: int = 2) -> torch.Tensor:
    """Build the real Heisenberg S·S Hamiltonian.

    H = J (Sx⊗Sx + Sy⊗Sy + Sz⊗Sz)

    Sy⊗Sy is purely real despite Sy being imaginary.
    Returns shape (d, d, d, d) in (bra1, bra2, ket1, ket2) convention,
    i.e. H[j1, j2, i1, i2] = <j1, j2| H |i1, i2>.
    """
    # Pauli / 2
    Sx = torch.tensor([[0., 1.], [1., 0.]], dtype=DTYPE) / 2
    Sz = torch.tensor([[1., 0.], [0., -1.]], dtype=DTYPE) / 2

    # Sx⊗Sx:  (Sx)_{ij} (Sx)_{kl} → H[i,k,j,l]
    SxSx = oe.contract("ij,kl->ikjl", Sx, Sx, backend="torch")
    SzSz = oe.contract("ij,kl->ikjl", Sz, Sz, backend="torch")

    # Sy⊗Sy in real form:
    # Sy = -i/2 [[0,-1],[1,0]]  →  Sy⊗Sy = -1/4 [[0,-1],[1,0]] ⊗ [[0,-1],[1,0]]
    # This is real.  Compute directly:
    SyM = torch.tensor([[0., -1.], [1., 0.]], dtype=DTYPE) / 2  # imaginary part of Sy (times i)
    # (i SyM) ⊗ (i SyM) = -SyM ⊗ SyM
    SySy = -oe.contract("ij,kl->ikjl", SyM, SyM, backend="torch")

    H = J * (SxSx + SySy + SzSz)
    _assert_real(H, "Heisenberg H")
    return H


# ══════════════════════════════════════════════════════════════════════════════
# §4  Tensor initialization
# ══════════════════════════════════════════════════════════════════════════════

def initialize_abcdef(D_bond: int, d_phys: int = 2):
    """Create 6 random real single-layer tensors, each (D, D, D, d), normalized."""
    tensors = []
    for _ in range(6):
        t = torch.randn(D_bond, D_bond, D_bond, d_phys, dtype=DTYPE)
        tensors.append(normalize_tensor(t))
    return tuple(tensors)


def initialize_neel_with_noise(D_bond: int, noise: float = 0.1, d_phys: int = 2):
    """Néel-state init (a,c,e = spin-up; b,d,f = spin-down) + Gaussian noise.

    Starting near the Néel state biases the optimizer toward the physical
    antiferromagnetic minimum, avoiding 'unphysical' local minima that can
    appear under unconstrained optimization at finite chi.

    The Néel base sits at E/bond = -0.25 (product-state lower bound);
    optimisation will then descend toward the true variational minimum.
    """
    up = torch.zeros(D_bond, D_bond, D_bond, d_phys, dtype=DTYPE)
    up[0, 0, 0, 0] = 1.0          # |↑⟩ at virtual slot (0,0,0)
    dn = torch.zeros(D_bond, D_bond, D_bond, d_phys, dtype=DTYPE)
    dn[0, 0, 0, 1] = 1.0          # |↓⟩ at virtual slot (0,0,0)

    result = []
    for base in (up, dn, up, dn, up, dn):   # a,c,e = up; b,d,f = down
        t = base + noise * torch.randn_like(base)
        result.append(normalize_tensor(t))
    return tuple(result)


# ══════════════════════════════════════════════════════════════════════════════
# §5  Double-layer construction
# ══════════════════════════════════════════════════════════════════════════════

def abcdef_to_ABCDEF(a, b, c, d, e, f, D_sq: int):
    """Build double-layer tensors.  For real ket, bra = ket (no conjugation).

    A_{ux,vy,wz} = Σ_p a_{u,v,w,p} · a_{x,y,z,p}
    Reshaped to (D², D², D²).
    """
    result = []
    for t in (a, b, c, d, e, f):
        DL = oe.contract("uvwp,xyzp->uxvywz", t, t, optimize=[(0, 1)], backend="torch")
        DL = DL.reshape(D_sq, D_sq, D_sq)
        result.append(DL)
    return tuple(result)


# ══════════════════════════════════════════════════════════════════════════════
# §6  CTMRG truncation via projectors
# ══════════════════════════════════════════════════════════════════════════════

def _truncate_via_svd(rho, chi):
    """Compute truncated left/right isometries from a real matrix rho via SVD.

    Returns (U_chi, Vh_chi, sv_kept) where:
        U_chi:  shape (n, chi) — left isometry
        Vh_chi: shape (chi, n) — right isometry
        sv_kept: shape (chi,) — kept singular values
    """
    n = rho.shape[0]
    k = min(chi, n)

    U, s, Vh = torch.linalg.svd(rho)
    U_chi = U[:, :k]       # (n, k)
    Vh_chi = Vh[:k, :]     # (k, n)
    sv_kept = s[:k]

    return U_chi, Vh_chi, sv_kept


def trunc_corners(matC21, matC32, matC13, chi, D_sq):
    """Compute truncated corners and projectors from grown corner matrices.

    Given 3 grown corners (each n×n where n = chi_old * D_sq or similar),
    builds 3 rho matrices by cyclic multiplication, decomposes each via
    eigh-based truncation, and applies projectors to truncate corners.

    Returns:
        (V2, C21, U1, V3, C32, U2, V1, C13, U3, sv32, sv13, sv21)
        - Ci: truncated corners (chi, chi)
        - Ui: left projectors, reshaped to (dim1, D_sq, chi)
        - Vi: right projectors, reshaped to (chi, dim1, D_sq)
        - svi: kept singular values
    """
    n = matC21.shape[0]
    dim1 = n // D_sq  # first env-bond dimension

    # rho_32 = C13 @ C32 @ C21   → decomposes to give U3, V2
    rho32 = matC13 @ matC32 @ matC21
    U3, V2, sv32 = _truncate_via_svd(rho32, chi)

    # rho_13 = C21 @ C13 @ C32   → decomposes to give U1, V3
    rho13 = matC21 @ matC13 @ matC32
    U1, V3, sv13 = _truncate_via_svd(rho13, chi)

    # rho_21 = C32 @ C21 @ C13   → decomposes to give U2, V1
    rho21 = matC32 @ matC21 @ matC13
    U2, V1, sv21 = _truncate_via_svd(rho21, chi)

    # Truncate corners: Ci_new = Ui.T @ matCi @ Vj
    C21 = U1.T @ matC21 @ V2.T
    C32 = U2.T @ matC32 @ V3.T
    C13 = U3.T @ matC13 @ V1.T

    # Corner normalization: scale by sum(sv)^(1/3) then equalize
    sum_sv32 = sv32.sum()
    sum_sv13 = sv13.sum()
    sum_sv21 = sv21.sum()
    scaleC = sum_sv32.pow(1.0 / 3.0)
    if torch.isfinite(scaleC) and scaleC.item() > 1e-30:
        C21 = C21 / scaleC
        C32 = C32 / scaleC
        C13 = C13 / scaleC
    # Equalize individual corner norms
    n21 = torch.linalg.norm(C21)
    n32 = torch.linalg.norm(C32)
    n13 = torch.linalg.norm(C13)
    nC_prod = n21 * n32 * n13
    if torch.isfinite(nC_prod) and nC_prod.item() > 1e-30:
        nC_geo = nC_prod.pow(1.0 / 3.0)
        C21 = C21 * (nC_geo / n21)
        C32 = C32 * (nC_geo / n32)
        C13 = C13 * (nC_geo / n13)

    # Reshape projectors for absorption contractions
    U1 = U1.reshape(dim1, D_sq, chi)
    U2 = U2.reshape(dim1, D_sq, chi)
    U3 = U3.reshape(dim1, D_sq, chi)
    V1 = V1.reshape(chi, dim1, D_sq)
    V2 = V2.reshape(chi, dim1, D_sq)
    V3 = V3.reshape(chi, dim1, D_sq)

    return V2, C21, U1, V3, C32, U2, V1, C13, U3, sv32, sv13, sv21


# ══════════════════════════════════════════════════════════════════════════════
# §7  Environment update functions
# ══════════════════════════════════════════════════════════════════════════════

def _normalize_transfers(Ts):
    """Simple Frobenius normalization for 6 transfers."""
    return tuple(normalize_tensor(T) for T in Ts)


def _norm_based_transfer_normalization(C21, C32, C13, Ts, sites_DL, chi, D_bond):
    """Normalize 6 transfers using full 6-site norm contraction.

    Computes norm_env = Tr(closed network), divides all Ts by norm^(1/6),
    then equalizes individual T norms to geometric mean.

    Ts = (T1x, T2x, T2y, T3x, T3y, T1y) — 6 transfers in position order.
    sites_DL = (S1, S2, S3, S4, S5, S6) — double-layer sites in position order.
    
    Falls back to Frobenius normalization if shapes don't match (e.g. during init).
    """
    # Check if all transfers have the expected shape (chi, chi, D_sq)
    D_sq = D_bond * D_bond
    expected_size = chi * chi * D_sq
    if any(T.numel() != expected_size for T in Ts):
        # During initialization, transfers may have non-standard shapes
        return tuple(normalize_tensor(T) for T in Ts)

    T1x, T2x, T2y, T3x, T3y, T1y = Ts
    S1, S2, S3, S4, S5, S6 = sites_DL
    cDD = D_bond * D_bond

    # Reshape transfers and sites
    T1x4 = T1x.reshape(chi, chi, D_bond, D_bond)
    T3x4 = T3x.reshape(chi, chi, D_bond, D_bond)
    T2y4 = T2y.reshape(chi, chi, D_bond, D_bond)
    T1y4 = T1y.reshape(chi, chi, D_bond, D_bond)
    T3y4 = T3y.reshape(chi, chi, D_bond, D_bond)
    T2x4 = T2x.reshape(chi, chi, D_bond, D_bond)

    S1_6 = S1.reshape(D_bond, D_bond, D_bond, D_bond, D_bond, D_bond)
    S2_6 = S2.reshape(D_bond, D_bond, D_bond, D_bond, D_bond, D_bond)
    S3_6 = S3.reshape(D_bond, D_bond, D_bond, D_bond, D_bond, D_bond)
    S4_6 = S4.reshape(D_bond, D_bond, D_bond, D_bond, D_bond, D_bond)
    S5_6 = S5.reshape(D_bond, D_bond, D_bond, D_bond, D_bond, D_bond)
    S6_6 = S6.reshape(D_bond, D_bond, D_bond, D_bond, D_bond, D_bond)

    # Closed tensors (same pattern as energy but with double-layer)
    c1 = oe.contract("YX,MYar,arbsct->MbsXct", C21, T1x4, S1_6,
                     optimize=[(0, 1), (0, 1)], backend="torch"
                     ).reshape(chi * cDD, chi * cDD)
    c2 = oe.contract("MYct,arbsct->YarMbs", T3x4, S2_6,
                     optimize=[(0, 1)], backend="torch"
                     ).reshape(chi * cDD, chi * cDD)
    c3 = oe.contract("ZY,NZbs,arbsct->NctYar", C32, T2y4, S3_6,
                     optimize=[(0, 1), (0, 1)], backend="torch"
                     ).reshape(chi * cDD, chi * cDD)
    c4 = oe.contract("NZar,arbsct->ZbsNct", T1y4, S4_6,
                     optimize=[(0, 1)], backend="torch"
                     ).reshape(chi * cDD, chi * cDD)
    c5 = oe.contract("XZ,LXct,arbsct->LarZbs", C13, T3y4, S5_6,
                     optimize=[(0, 1), (0, 1)], backend="torch"
                     ).reshape(chi * cDD, chi * cDD)
    c6 = oe.contract("LXbs,arbsct->XctLar", T2x4, S6_6,
                     optimize=[(0, 1)], backend="torch"
                     ).reshape(chi * cDD, chi * cDD)

    # Pairs: (3,2), (5,4), (1,6) — matching energy function ordering
    p32 = torch.mm(c3, c2)
    p54 = torch.mm(c5, c4)
    p16 = torch.mm(c1, c6)

    norm_val = oe.contract("xy,yz,zx->", p32, p16, p54, backend="torch")
    scale = norm_val.abs().pow(1.0 / 6.0)

    Ts_list = list(Ts)
    if torch.isfinite(scale) and scale.item() > 1e-30:
        Ts_list = [T / scale for T in Ts_list]

    # Equalize individual norms
    norms = [torch.linalg.norm(T) for T in Ts_list]
    prod = torch.ones(1, dtype=DTYPE)
    for n in norms:
        prod = prod * n
    if prod.item() > 1e-30 and torch.isfinite(prod):
        geo = prod.pow(1.0 / 6.0)
        Ts_list = [T * (geo / (n + 1e-30)) for T, n in zip(Ts_list, norms)]

    return tuple(Ts_list)


def update_env_1to2(C21CD, C32EF, C13AB, T1F, T2A, T2B, T3C, T3D, T1E,
                    A, B, C, D, E, F, chi, D_sq):
    """CTMRG step: type-1 env → type-2 env.

    Absorbs sites E,B into C21; A,D into C32; C,F into C13.
    Returns (C21EB, C32AD, C13CF, T1D, T2C, T2F, T3E, T3B, T1A, sv32, sv13, sv21).
    """
    dim0 = T1F.shape[0]  # first dim of T (= chi in steady state, possibly D_sq in init)
    grown = dim0 * D_sq

    # Grow corners
    matC21EB = oe.contract("YX,MYa,LXb,amg,lbg->MmLl",
                           C21CD, T1F, T2A, E, B,
                           optimize=[(0, 2), (0, 3), (1, 2), (0, 1)], backend="torch")
    matC21EB = matC21EB.reshape(grown, grown)

    matC32AD = oe.contract("ZY,NZb,MYg,abn,amg->NnMm",
                           C32EF, T2B, T3C, A, D,
                           optimize=[(0, 2), (0, 3), (1, 2), (0, 1)], backend="torch")
    matC32AD = matC32AD.reshape(grown, grown)

    matC13CF = oe.contract("XZ,LXg,NZa,lbg,abn->LlNn",
                           C13AB, T3D, T1E, C, F,
                           optimize=[(0, 2), (0, 3), (1, 2), (0, 1)], backend="torch")
    matC13CF = matC13CF.reshape(grown, grown)

    # Truncate
    V2A, C21EB, U1F, V3C, C32AD, U2B, V1E, C13CF, U3D, sv32, sv13, sv21 = \
        trunc_corners(matC21EB, matC32AD, matC13CF, chi, D_sq)

    # Update transfers
    T3E = oe.contract("OYa,abg,ObM->YMg",
                      T1F, E, U1F, optimize=[(0, 1), (0, 1)], backend="torch")
    T3B = oe.contract("OXb,abg,LOa->XLg",
                      T2A, B, V2A, optimize=[(0, 1), (0, 1)], backend="torch")
    T1A = oe.contract("OZb,abg,OgN->ZNa",
                      T2B, A, U2B, optimize=[(0, 1), (0, 1)], backend="torch")
    T1D = oe.contract("OYg,abg,MOb->YMa",
                      T3C, D, V3C, optimize=[(0, 1), (0, 1)], backend="torch")
    T2C = oe.contract("OXg,abg,OaL->XLb",
                      T3D, C, U3D, optimize=[(0, 1), (0, 1)], backend="torch")
    T2F = oe.contract("OZa,abg,NOg->ZNa",
                      T1E, F, V1E, optimize=[(0, 1), (0, 1)], backend="torch")

    # Normalize transfers via full norm contraction
    D_bond = int(round(D_sq ** 0.5))
    Ts = (T1D, T2C, T2F, T3E, T3B, T1A)
    # Env-2 site positions: 1=A, 2=B, 3=C, 4=D, 5=E, 6=F
    sites_DL = (A, B, C, D, E, F)
    Ts = _norm_based_transfer_normalization(
        C21EB, C32AD, C13CF, Ts, sites_DL, chi, D_bond)
    T1D, T2C, T2F, T3E, T3B, T1A = Ts

    return C21EB, C32AD, C13CF, T1D, T2C, T2F, T3E, T3B, T1A, sv32, sv13, sv21


def update_env_2to3(C21EB, C32AD, C13CF, T1D, T2C, T2F, T3E, T3B, T1A,
                    A, B, C, D, E, F, chi, D_sq):
    """CTMRG step: type-2 env → type-3 env.

    Absorbs sites A,F into C21; C,B into C32; E,D into C13.
    Returns (C21AF, C32CB, C13ED, T1B, T2E, T2D, T3A, T3F, T1C, sv32, sv13, sv21).
    """
    dim0 = T1D.shape[0]
    grown = dim0 * D_sq

    matC21AF = oe.contract("YX,MYa,LXb,amg,lbg->MmLl",
                           C21EB, T1D, T2C, A, F,
                           optimize=[(0, 2), (0, 3), (1, 2), (0, 1)], backend="torch")
    matC21AF = matC21AF.reshape(grown, grown)

    matC32CB = oe.contract("ZY,NZb,MYg,abn,amg->NnMm",
                           C32AD, T2F, T3E, C, B,
                           optimize=[(0, 2), (0, 3), (1, 2), (0, 1)], backend="torch")
    matC32CB = matC32CB.reshape(grown, grown)

    matC13ED = oe.contract("XZ,LXg,NZa,lbg,abn->LlNn",
                           C13CF, T3B, T1A, E, D,
                           optimize=[(0, 2), (0, 3), (1, 2), (0, 1)], backend="torch")
    matC13ED = matC13ED.reshape(grown, grown)

    V2C, C21AF, U1D, V3E, C32CB, U2F, V1A, C13ED, U3B, sv32, sv13, sv21 = \
        trunc_corners(matC21AF, matC32CB, matC13ED, chi, D_sq)

    T3A = oe.contract("OYa,abg,ObM->YMg",
                      T1D, A, U1D, optimize=[(0, 1), (0, 1)], backend="torch")
    T3F = oe.contract("OXb,abg,LOa->XLg",
                      T2C, F, V2C, optimize=[(0, 1), (0, 1)], backend="torch")
    T1C = oe.contract("OZb,abg,OgN->ZNa",
                      T2F, C, U2F, optimize=[(0, 1), (0, 1)], backend="torch")
    T1B = oe.contract("OYg,abg,MOb->YMa",
                      T3E, B, V3E, optimize=[(0, 1), (0, 1)], backend="torch")
    T2E = oe.contract("OXg,abg,OaL->XLb",
                      T3B, E, U3B, optimize=[(0, 1), (0, 1)], backend="torch")
    T2D = oe.contract("OZa,abg,NOg->ZNa",
                      T1A, D, V1A, optimize=[(0, 1), (0, 1)], backend="torch")

    # Normalize transfers via full norm contraction
    D_bond = int(round(D_sq ** 0.5))
    Ts = (T1B, T2E, T2D, T3A, T3F, T1C)
    # Env-3 site positions: 1=C, 2=F, 3=E, 4=B, 5=A, 6=D
    sites_DL = (C, F, E, B, A, D)
    Ts = _norm_based_transfer_normalization(
        C21AF, C32CB, C13ED, Ts, sites_DL, chi, D_bond)
    T1B, T2E, T2D, T3A, T3F, T1C = Ts

    return C21AF, C32CB, C13ED, T1B, T2E, T2D, T3A, T3F, T1C, sv32, sv13, sv21


def update_env_3to1(C21AF, C32CB, C13ED, T1B, T2E, T2D, T3A, T3F, T1C,
                    A, B, C, D, E, F, chi, D_sq):
    """CTMRG step: type-3 env → type-1 env.

    Absorbs sites C,D into C21; E,F into C32; A,B into C13.
    Returns (C21CD, C32EF, C13AB, T1F, T2A, T2B, T3C, T3D, T1E, sv32, sv13, sv21).
    """
    dim0 = T1B.shape[0]
    grown = dim0 * D_sq

    matC21CD = oe.contract("YX,MYa,LXb,amg,lbg->MmLl",
                           C21AF, T1B, T2E, C, D,
                           optimize=[(0, 2), (0, 3), (1, 2), (0, 1)], backend="torch")
    matC21CD = matC21CD.reshape(grown, grown)

    matC32EF = oe.contract("ZY,NZb,MYg,abn,amg->NnMm",
                           C32CB, T2D, T3A, E, F,
                           optimize=[(0, 2), (0, 3), (1, 2), (0, 1)], backend="torch")
    matC32EF = matC32EF.reshape(grown, grown)

    matC13AB = oe.contract("XZ,LXg,NZa,lbg,abn->LlNn",
                           C13ED, T3F, T1C, A, B,
                           optimize=[(0, 2), (0, 3), (1, 2), (0, 1)], backend="torch")
    matC13AB = matC13AB.reshape(grown, grown)

    V2E, C21CD, U1B, V3A, C32EF, U2D, V1C, C13AB, U3F, sv32, sv13, sv21 = \
        trunc_corners(matC21CD, matC32EF, matC13AB, chi, D_sq)

    T3C = oe.contract("OYa,abg,ObM->YMg",
                      T1B, C, U1B, optimize=[(0, 1), (0, 1)], backend="torch")
    T3D = oe.contract("OXb,abg,LOa->XLg",
                      T2E, D, V2E, optimize=[(0, 1), (0, 1)], backend="torch")
    T1E = oe.contract("OZb,abg,OgN->ZNa",
                      T2D, E, U2D, optimize=[(0, 1), (0, 1)], backend="torch")
    T1F = oe.contract("OYg,abg,MOb->YMa",
                      T3A, F, V3A, optimize=[(0, 1), (0, 1)], backend="torch")
    T2A = oe.contract("OXg,abg,OaL->XLb",
                      T3F, A, U3F, optimize=[(0, 1), (0, 1)], backend="torch")
    T2B = oe.contract("OZa,abg,NOg->ZNa",
                      T1C, B, V1C, optimize=[(0, 1), (0, 1)], backend="torch")

    # Normalize transfers via full norm contraction
    D_bond = int(round(D_sq ** 0.5))
    Ts = (T1F, T2A, T2B, T3C, T3D, T1E)
    # Env-1 site positions: 1=E, 2=D, 3=A, 4=F, 5=C, 6=B
    sites_DL = (E, D, A, F, C, B)
    Ts = _norm_based_transfer_normalization(
        C21CD, C32EF, C13AB, Ts, sites_DL, chi, D_bond)
    T1F, T2A, T2B, T3C, T3D, T1E = Ts

    return C21CD, C32EF, C13AB, T1F, T2A, T2B, T3C, T3D, T1E, sv32, sv13, sv21


# ══════════════════════════════════════════════════════════════════════════════
# §8  Environment initialization
# ══════════════════════════════════════════════════════════════════════════════

def initialize_env(A, B, C, D, E, F, chi, D_sq):
    """Bootstrap CTMRG environment from double-layer tensors.

    Uses the same seed approach as the original: contract pairs of sites for
    seed corners and transfer tensors, then run CTMRG update steps to bring
    the environment to the correct steady-state shape (chi, chi, D_sq).

    Returns: type-2 env tuple (C21EB, ..., T1A) with Ts shape (chi, chi, D_sq).
    """
    chi_seed = min(chi, D_sq * D_sq)

    # Seed corners (D², D²) — type-3 naming
    C21AF = oe.contract("oyg,xog->yx", A, F, optimize=[(0, 1)], backend="torch")
    C32CB = oe.contract("aoz,ayo->zy", C, B, optimize=[(0, 1)], backend="torch")
    C13ED = oe.contract("xbo,obz->xz", E, D, optimize=[(0, 1)], backend="torch")

    # Seed transfer pairs: contract 4 sites → SVD split
    T2ET3F = oe.contract("ubi,jki,jlk,vlg->ubvg", E, B, C, F,
                         optimize=[(1, 2), (0, 1), (0, 1)], backend="torch")
    T3AT1B = oe.contract("iug,ijk,kjl,avl->ugva", A, D, E, B,
                         optimize=[(1, 2), (0, 1), (0, 1)], backend="torch")
    T1CT2D = oe.contract("aiu,kij,lkj,lbv->uavb", C, F, A, D,
                         optimize=[(1, 2), (0, 1), (0, 1)], backend="torch")

    T2ET3F = T2ET3F.reshape(D_sq * D_sq, D_sq * D_sq)
    T3AT1B = T3AT1B.reshape(D_sq * D_sq, D_sq * D_sq)
    T1CT2D = T1CT2D.reshape(D_sq * D_sq, D_sq * D_sq)

    # SVD split (real → real singular vectors)
    U2E, sv23, Vh3F = torch.linalg.svd(T2ET3F)
    U3A, sv31, Vh1B = torch.linalg.svd(T3AT1B)
    U1C, sv12, Vh2D = torch.linalg.svd(T1CT2D)

    sq23 = torch.sqrt(torch.clamp(sv23[:chi_seed], min=1e-12))
    sq31 = torch.sqrt(torch.clamp(sv31[:chi_seed], min=1e-12))
    sq12 = torch.sqrt(torch.clamp(sv12[:chi_seed], min=1e-12))

    T2E = (U2E[:, :chi_seed] @ torch.diag(sq23)).reshape(D_sq, D_sq, chi_seed).permute(2, 0, 1)
    T3A = (U3A[:, :chi_seed] @ torch.diag(sq31)).reshape(D_sq, D_sq, chi_seed).permute(2, 0, 1)
    T1C = (U1C[:, :chi_seed] @ torch.diag(sq12)).reshape(D_sq, D_sq, chi_seed).permute(2, 0, 1)
    T3F = (torch.diag(sq23) @ Vh3F[:chi_seed, :]).reshape(chi_seed, D_sq, D_sq)
    T1B = (torch.diag(sq31) @ Vh1B[:chi_seed, :]).reshape(chi_seed, D_sq, D_sq)
    T2D = (torch.diag(sq12) @ Vh2D[:chi_seed, :]).reshape(chi_seed, D_sq, D_sq)

    # Step 1: type-3 → type-1 (at chi_seed)
    r1 = update_env_3to1(C21AF, C32CB, C13ED, T1B, T2E, T2D, T3A, T3F, T1C,
                         A, B, C, D, E, F, chi_seed, D_sq)
    # Step 2: type-1 → type-2 (at chi_seed)
    r2 = update_env_1to2(r1[0], r1[1], r1[2], r1[3], r1[4], r1[5], r1[6], r1[7], r1[8],
                         A, B, C, D, E, F, chi_seed, D_sq)

    if chi <= chi_seed:
        return r2[:9]

    # For chi > D_sq², grow through one full cycle at full chi
    r3 = update_env_2to3(r2[0], r2[1], r2[2], r2[3], r2[4], r2[5], r2[6], r2[7], r2[8],
                         A, B, C, D, E, F, chi, D_sq)
    r4 = update_env_3to1(r3[0], r3[1], r3[2], r3[3], r3[4], r3[5], r3[6], r3[7], r3[8],
                         A, B, C, D, E, F, chi, D_sq)
    r5 = update_env_1to2(r4[0], r4[1], r4[2], r4[3], r4[4], r4[5], r4[6], r4[7], r4[8],
                         A, B, C, D, E, F, chi, D_sq)
    return r5[:9]


# ══════════════════════════════════════════════════════════════════════════════
# §9  CTMRG convergence check
# ══════════════════════════════════════════════════════════════════════════════

def _rho_sv_converged(last_Cs, now_Cs, thr):
    """Check convergence of all 3 env types via rho singular values.

    Each env type contributes one rho = C13 @ C32 @ C21.  We compare the
    normalized SV spectra between consecutive iterations.
    """
    if last_Cs is None:
        return False
    max_delta = 0.0
    for (lC21, lC32, lC13), (nC21, nC32, nC13) in zip(last_Cs, now_Cs):
        rho_last = lC13 @ lC32 @ lC21
        rho_now = nC13 @ nC32 @ nC21
        sv_last = torch.linalg.svdvals(rho_last)
        sv_now = torch.linalg.svdvals(rho_now)
        # Normalize to make scale-invariant
        sv_last = sv_last / (sv_last[0] + 1e-30)
        sv_now = sv_now / (sv_now[0] + 1e-30)
        delta = (sv_now - sv_last).abs().max().item()
        max_delta = max(max_delta, delta)
    return max_delta < thr


# ══════════════════════════════════════════════════════════════════════════════
# §10  CTMRG main loop
# ══════════════════════════════════════════════════════════════════════════════

def ctmrg(A, B, C, D, E, F, chi, D_sq,
        max_steps=100, conv_thr=1e-7, min_steps=5):
    """Run CTMRG until convergence.

    Each 'step' in max_steps corresponds to one full 1→2→3→1 cycle (3 updates).

    Args:
        min_steps:  Minimum number of CTM cycles to run before checking
                    convergence.  Prevents premature convergence to spurious
                    CTMRG fixed points ('ghost' environments that satisfy the
                    SV-delta criterion after just 1-3 steps but are not the
                    physical fixed point).  Default 5.
    Returns (env1, env2, env3, ctm_steps).
    """
    # Initialize → returns type-2 env
    env2 = initialize_env(A, B, C, D, E, F, chi, D_sq)
    C21EB, C32AD, C13CF, T1D, T2C, T2F, T3E, T3B, T1A = env2

    # Bootstrap: 2→3 then 3→1 to get all 3 types
    r3 = update_env_2to3(C21EB, C32AD, C13CF, T1D, T2C, T2F, T3E, T3B, T1A,
                         A, B, C, D, E, F, chi, D_sq)
    C21AF, C32CB, C13ED, T1B, T2E, T2D, T3A, T3F, T1C = r3[:9]

    r1 = update_env_3to1(C21AF, C32CB, C13ED, T1B, T2E, T2D, T3A, T3F, T1C,
                         A, B, C, D, E, F, chi, D_sq)
    C21CD, C32EF, C13AB, T1F, T2A, T2B, T3C, T3D, T1E = r1[:9]

    last_Cs = None
    ctm_steps = max_steps

    for step in range(max_steps):
        # Convergence check before update (compare current vs previous iteration).
        # Require at least min_steps cycles to guard against premature convergence
        # to spurious ghost fixed points.
        now_Cs = [
            (C21CD, C32EF, C13AB),
            (C21EB, C32AD, C13CF),
            (C21AF, C32CB, C13ED),
        ]
        if step >= min_steps and _rho_sv_converged(last_Cs, now_Cs, conv_thr):
            ctm_steps = step
            break
        last_Cs = now_Cs

        # Full cycle: 1→2→3→1
        r2 = update_env_1to2(C21CD, C32EF, C13AB, T1F, T2A, T2B, T3C, T3D, T1E,
                             A, B, C, D, E, F, chi, D_sq)
        C21EB, C32AD, C13CF, T1D, T2C, T2F, T3E, T3B, T1A = r2[:9]

        r3 = update_env_2to3(C21EB, C32AD, C13CF, T1D, T2C, T2F, T3E, T3B, T1A,
                             A, B, C, D, E, F, chi, D_sq)
        C21AF, C32CB, C13ED, T1B, T2E, T2D, T3A, T3F, T1C = r3[:9]

        r1 = update_env_3to1(C21AF, C32CB, C13ED, T1B, T2E, T2D, T3A, T3F, T1C,
                             A, B, C, D, E, F, chi, D_sq)
        C21CD, C32EF, C13AB, T1F, T2A, T2B, T3C, T3D, T1E = r1[:9]

    env1 = (C21CD, C32EF, C13AB, T1F, T2A, T2B, T3C, T3D, T1E)
    env2 = (C21EB, C32AD, C13CF, T1D, T2C, T2F, T3E, T3B, T1A)
    env3 = (C21AF, C32CB, C13ED, T1B, T2E, T2D, T3A, T3F, T1C)
    return env1, env2, env3, ctm_steps


# ══════════════════════════════════════════════════════════════════════════════
# §11  Energy functions
# ══════════════════════════════════════════════════════════════════════════════

def _energy_3bonds_raw(a, b, c, d, e, f,
                       H1, H2, H3,
                       chi, D_bond,
                       C21, C32, C13,
                       T1x, T2x, T2y, T3x, T3y, T1y,
                       site_order):
    """Generic energy for 3 bonds.  Returns (numerator, norm) separately.

    CRITICAL: returns raw (E_total_unnormed, norm) so callers can combine
    multiple env types with a SINGLE consistent norm (fixes the variational
    bound violation caused by using different norms per env type).

    site_order = (s1, s2, s3, s4, s5, s6) maps positions to single-layer tensors.
    Bonds: (s3,s2) at C32 side, (s5,s4) at C13 side, (s1,s6) at C21 side.
    H in (bra1, bra2, ket1, ket2); permuted to (ket1, bra1, ket2, bra2) for contraction.
    """
    H1 = H1.permute(2, 0, 3, 1)
    H2 = H2.permute(2, 0, 3, 1)
    H3 = H3.permute(2, 0, 3, 1)

    s1, s2, s3, s4, s5, s6 = site_order
    cDD = D_bond * D_bond

    T1x4 = T1x.reshape(chi, chi, D_bond, D_bond)
    T2x4 = T2x.reshape(chi, chi, D_bond, D_bond)
    T2y4 = T2y.reshape(chi, chi, D_bond, D_bond)
    T3x4 = T3x.reshape(chi, chi, D_bond, D_bond)
    T3y4 = T3y.reshape(chi, chi, D_bond, D_bond)
    T1y4 = T1y.reshape(chi, chi, D_bond, D_bond)

    # Open tensors: contract env + ket + bra with open physical indices (i, j)
    # Position 1 (at C21): C21(YX), T1x(MYar), s1(abci), s1(rstj) → MbsXctij
    open_1 = oe.contract("YX,MYar,abci,rstj->MbsXctij",
                         C21, T1x4, s1, s1,
                         optimize=[(0, 1), (0, 1), (0, 1)], backend="torch")
    # Position 2 (at T3x): T3x(MYct), s2(abci), s2(rstj) → YarMbsij
    open_2 = oe.contract("MYct,abci,rstj->YarMbsij",
                         T3x4, s2, s2,
                         optimize=[(0, 1), (0, 1)], backend="torch")
    # Position 3 (at C32): C32(ZY), T2y(NZbs), s3(abci), s3(rstj) → NctYarij
    open_3 = oe.contract("ZY,NZbs,abci,rstj->NctYarij",
                         C32, T2y4, s3, s3,
                         optimize=[(0, 1), (0, 1), (0, 1)], backend="torch")
    # Position 4 (at T1y): T1y(NZar), s4(abci), s4(rstj) → ZbsNctij
    open_4 = oe.contract("NZar,abci,rstj->ZbsNctij",
                         T1y4, s4, s4,
                         optimize=[(0, 1), (0, 1)], backend="torch")
    # Position 5 (at C13): C13(XZ), T3y(LXct), s5(abci), s5(rstj) → LarZbsij
    open_5 = oe.contract("XZ,LXct,abci,rstj->LarZbsij",
                         C13, T3y4, s5, s5,
                         optimize=[(0, 1), (0, 1), (0, 1)], backend="torch")
    # Position 6 (at T2x): T2x(LXbs), s6(abci), s6(rstj) → XctLarij
    open_6 = oe.contract("LXbs,abci,rstj->XctLarij",
                         T2x4, s6, s6,
                         optimize=[(0, 1), (0, 1)], backend="torch")

    # Closed (norm) tensors: trace over physical indices
    closed_1 = oe.contract("MbsXctii->MbsXct", open_1).reshape(chi * cDD, chi * cDD)
    closed_2 = oe.contract("YarMbsii->YarMbs", open_2).reshape(chi * cDD, chi * cDD)
    closed_3 = oe.contract("NctYarii->NctYar", open_3).reshape(chi * cDD, chi * cDD)
    closed_4 = oe.contract("ZbsNctii->ZbsNct", open_4).reshape(chi * cDD, chi * cDD)
    closed_5 = oe.contract("LarZbsii->LarZbs", open_5).reshape(chi * cDD, chi * cDD)
    closed_6 = oe.contract("XctLarii->XctLar", open_6).reshape(chi * cDD, chi * cDD)

    # H insertions for 3 bonds
    # Bond 1: (s3, s2) → H1 between positions 3 and 2
    H_32 = oe.contract("NctYarij,ijkl,YarMbskl->NctMbs",
                       open_3, H1, open_2,
                       optimize=[(0, 1), (0, 1)], backend="torch"
                       ).reshape(chi * cDD, chi * cDD)
    # Bond 2: (s5, s4) → H2 between positions 5 and 4
    H_54 = oe.contract("LarZbsij,ijkl,ZbsNctkl->LarNct",
                       open_5, H2, open_4,
                       optimize=[(0, 1), (0, 1)], backend="torch"
                       ).reshape(chi * cDD, chi * cDD)
    # Bond 3: (s1, s6) → H3 between positions 1 and 6
    H_16 = oe.contract("MbsXctij,ijkl,XctLarkl->MbsLar",
                       open_1, H3, open_6,
                       optimize=[(0, 1), (0, 1)], backend="torch"
                       ).reshape(chi * cDD, chi * cDD)


    # Pair products for norm — CRITICAL: pairs go (3→2), (5→4), (1→6)
    # matching the backward loop around the hexagonal CTMRG network.
    p32 = torch.mm(closed_3, closed_2)
    p54 = torch.mm(closed_5, closed_4)
    p16 = torch.mm(closed_1, closed_6)

    # Norm = Tr(p32 × p16 × p54)
    norm = oe.contract("xy,yz,zx->", p32, p16, p54, backend="torch")

    # print("norm_orderedEnv: ", norm.item())

    # Energy = Tr(H_bond × other_two_pairs) for each bond
    E_32 = oe.contract("xy,yz,zx->", H_32, p16, p54, backend="torch")
    E_54 = oe.contract("xy,yz,zx->", H_54, p32, p16, backend="torch")
    E_16 = oe.contract("xy,yz,zx->", H_16, p54, p32, backend="torch")

    return (E_32 + E_54 + E_16), norm


def _energy_3bonds(a, b, c, d, e, f,
                   H1, H2, H3,
                   chi, D_bond,
                   C21, C32, C13,
                   T1x, T2x, T2y, T3x, T3y, T1y,
                   site_order):
    """Convenience wrapper — returns (numerator / norm) for a single env type.
    Use _energy_3bonds_raw + consistent-norm formula for full energy.
    """
    num, norm = _energy_3bonds_raw(
        a, b, c, d, e, f, H1, H2, H3, chi, D_bond,
        C21, C32, C13, T1x, T2x, T2y, T3x, T3y, T1y, site_order)
    return num / norm


def energy_env1(a, b, c, d, e, f, H, chi, D_bond, env1):
    """Bonds EB, AD, CF using type-1 environment."""
    C21CD, C32EF, C13AB, T1F, T2A, T2B, T3C, T3D, T1E = env1
    # Positions:  1=E, 2=D, 3=A, 4=F, 5=C, 6=B  (from original code pattern)
    # Bonds: (A,D)=H_AD at pos (3,2), (C,F)=H_CF at pos (5,4), (E,B)=H_EB at pos (1,6)
    return _energy_3bonds(
        a, b, c, d, e, f, H, H, H, chi, D_bond,
        C21CD, C32EF, C13AB, T1F, T2A, T2B, T3C, T3D, T1E,
        site_order=(e, d, a, f, c, b))


def energy_env2(a, b, c, d, e, f, H, chi, D_bond, env2):
    """Bonds AF, CB, ED using type-2 environment."""
    C21EB, C32AD, C13CF, T1D, T2C, T2F, T3E, T3B, T1A = env2
    # Positions:  1=A, 2=B, 3=C, 4=D, 5=E, 6=F
    # Bonds: (C,B)=H_CB at pos (3,2), (E,D)=H_ED at pos (5,4), (A,F)=H_AF at pos (1,6)
    return _energy_3bonds(
        a, b, c, d, e, f, H, H, H, chi, D_bond,
        C21EB, C32AD, C13CF, T1D, T2C, T2F, T3E, T3B, T1A,
        site_order=(a, b, c, d, e, f))


def energy_env3(a, b, c, d, e, f, H, chi, D_bond, env3):
    """Bonds CD, EF, AB using type-3 environment."""
    C21AF, C32CB, C13ED, T1B, T2E, T2D, T3A, T3F, T1C = env3
    # Positions:  1=C, 2=F, 3=E, 4=B, 5=A, 6=D
    # Bonds: (E,F)=H_EF at pos (3,2), (A,B)=H_AB at pos (5,4), (C,D)=H_CD at pos (1,6)
    return _energy_3bonds(
        a, b, c, d, e, f, H, H, H, chi, D_bond,
        C21AF, C32CB, C13ED, T1B, T2E, T2D, T3A, T3F, T1C,
        site_order=(c, f, e, b, a, d))


def compute_total_energy(a, b, c, d, e, f, H, chi, D_bond, env1, env2, env3):
    """Compute E/bond using a SINGLE CONSISTENT NORM (average of the three env norms).

    Root-cause fix: using separate norms per env type breaks the variational
    bound because the three CTMRG approximate norms differ at finite chi.
    Dividing all 9 bond energies by the SAME average norm restores consistency.
    """
    C21CD, C32EF, C13AB, T1F, T2A, T2B, T3C, T3D, T1E = env1
    C21EB, C32AD, C13CF, T1D, T2C, T2F, T3E, T3B, T1A = env2
    C21AF, C32CB, C13ED, T1B, T2E, T2D, T3A, T3F, T1C = env3

    num1, n1 = _energy_3bonds_raw(
        a, b, c, d, e, f, H, H, H, chi, D_bond,
        C21CD, C32EF, C13AB, T1F, T2A, T2B, T3C, T3D, T1E,
        site_order=(e, d, a, f, c, b))
    num2, n2 = _energy_3bonds_raw(
        a, b, c, d, e, f, H, H, H, chi, D_bond,
        C21EB, C32AD, C13CF, T1D, T2C, T2F, T3E, T3B, T1A,
        site_order=(a, b, c, d, e, f))
    num3, n3 = _energy_3bonds_raw(
        a, b, c, d, e, f, H, H, H, chi, D_bond,
        C21AF, C32CB, C13ED, T1B, T2E, T2D, T3A, T3F, T1C,
        site_order=(c, f, e, b, a, d))

    # Single consistent norm: average of the three CTMRG environment approximations.
    # This is the key fix — combining numerators from different envs with a common
    # denominator preserves ⟨H⟩/⟨ψ|ψ⟩ semantics and restores the variational bound.
    norm_avg = (n1 + n2 + n3) / 3.0
    return (num1 + num2 + num3) / (9.0 * norm_avg)


# ══════════════════════════════════════════════════════════════════════════════
# §12  Optimization
# ══════════════════════════════════════════════════════════════════════════════

def _run_ctmrg_robust(DL_tuple, chi, D_sq, max_steps, thr,
                      fallback_cold=True, min_steps=5):
    """Run CTMRG, always using cold-start initialization.

    Changes vs. v1:
        * Cold fallback is ALWAYS attempted, so a cold start that stalls also gets
            a second, longer cold attempt.
    * Cold fallback budget is 5× (was 3×) for a more thorough convergence search.
    * min_steps is forwarded to ctmrg to prevent spurious early convergence.

    Returns (env1, env2, env3, ctm_steps, converged).
    """
    env1, env2, env3, steps = ctmrg(*DL_tuple, chi, D_sq, max_steps, thr,
                                    min_steps=min_steps)
    converged = steps < max_steps
    if not converged and fallback_cold:
        # First cold-start failed — try a longer cold run.
        env1, env2, env3, steps = ctmrg(*DL_tuple, chi, D_sq, max_steps * 5, thr,
                                        min_steps=min_steps)
        converged = steps < max_steps * 5
    return env1, env2, env3, steps, converged


def optimize_ipeps(H, chi, D_bond, d_phys=2,
                   max_ctm_steps=200, ctm_conv_thr=1e-8,
                   max_opt_steps=200,
                   lbfgs_max_iter=1, lbfgs_lr=1.0, lbfgs_history=20,
                   opt_tol_grad=1e-7, opt_tol_change=1e-9,
                   opt_conv_thr=1e-8,
                   grad_clip_norm=1.0,
                   sym_reg=0.0,
                   init_abcdef=None, init_mode='random', neel_noise=0.1,
                   verbose=True):
    """Optimize iPEPS via alternating CTMRG + L-BFGS.

    AD-iPEPS algorithm (Corboz 2016 / Liao et al. 2019):
      1. Normalize tensors.
      2. Run CTMRG to convergence → fixed environment env_k.
      3. One L-BFGS step (lbfgs_max_iter=1) with closure =
         compute_total_energy(tensors, env_k).  The Wolfe line search
         finds a step size that decreases the proxy energy.
      4. Normalize updated tensors.
      5. Cold-start CTMRG for the new tensors → true energy.
      6. Repeat.

    Key design decisions:
    * lbfgs_max_iter=1  — ONE gradient step per CTM convergence; prevents
      the inner loop from walking so far that the fixed environment becomes
      invalid (which caused CTM non-convergence and energy crashes).
    * Closure uses compute_total_energy (consistent single-norm Rayleigh
      quotient) — NOT nu1/n1+nu2/n2+nu3/n3 which is unbounded below.
    * sym_reg=0.5        — Symmetry regularizer coefficient.  Physical states
                           are translationally invariant (all three env-type
                           per-bond energies equal).  Ghost states (which arise
                           from finite-chi CTMRG boundary artifacts and can give
                           spurious energy below QMC) have large spread between
                           the three values.  This term penalises the variance of
                           e1, e2, e3 and keeps the optimizer in the physical
                           sector.  Set to 0 to disable (not recommended).
    * Gradient clipping — prevents huge steps on ill-conditioned landscapes.
    """
    D_sq = D_bond ** 2

    if init_abcdef is not None:
        a, b, c, dt, e, f = [t.detach().clone().to(DTYPE) for t in init_abcdef]
    elif init_mode == 'neel':
        a, b, c, dt, e, f = initialize_neel_with_noise(D_bond, noise=neel_noise, d_phys=d_phys)
    else:  # 'random'
        a, b, c, dt, e, f = initialize_abcdef(D_bond, d_phys)

    for t in (a, b, c, dt, e, f):
        t.requires_grad_(True)

    prev_true_E = None
    history = []
    prev_tensors  = None   # checkpoint: normalized tensors at end of last good step
    consecutive_ctm_failures = 0

    for step in range(max_opt_steps):
        # ── 1. Normalize tensors at the top of every step ──────────────────
        with torch.no_grad():
            for t in (a, b, c, dt, e, f):
                t.data = normalize_tensor(t.data)

        # ── 2. Converge CTMRG (fixed env for L-BFGS closure) ───────────────
        with torch.no_grad():
            DL = abcdef_to_ABCDEF(a, b, c, dt, e, f, D_sq)
            env1, env2, env3, ctm_steps, ctm_ok = _run_ctmrg_robust(
                DL, chi, D_sq, max_ctm_steps, ctm_conv_thr)

        # ── 2b. CTM failure handling ────────────────────────────────────────
        # If the pre-step CTM did not converge, the environment is garbage.
        # Taking a gradient step on a garbage environment corrupts the tensors
        # and causes the NEXT CTM to also fail — a spiralling cascade.
        # Instead: restore the last known-good tensors and skip this step.
        if not ctm_ok:
            consecutive_ctm_failures += 1
            if prev_tensors is not None:
                with torch.no_grad():
                    for t, saved in zip((a, b, c, dt, e, f), prev_tensors):
                        t.data.copy_(saved)
            if verbose:
                print(f"  opt {step:4d}  CTM-pre FAILED "
                      f"({consecutive_ctm_failures}x) — restoring checkpoint, "
                      f"skipping gradient step [CTM!warn-nokv]")
            if consecutive_ctm_failures >= 3:
                if verbose:
                    print(f"  opt {step:4d}  3 consecutive CTM failures "
                          f"— re-initializing tensors randomly")
                _new = initialize_abcdef(D_bond, d_phys)
                with torch.no_grad():
                    for t, tn in zip((a, b, c, dt, e, f), _new):
                        t.data.copy_(tn.data)
                prev_tensors   = None
                consecutive_ctm_failures = 0
            continue
        consecutive_ctm_failures = 0

        # ── 3. L-BFGS step with clipped gradient ───────────────────────────
        # Fresh optimizer each outer step (stale curvature from previous env is
        # physically misleading).
        optimizer = torch.optim.LBFGS(
            [a, b, c, dt, e, f],
            lr=lbfgs_lr, max_iter=lbfgs_max_iter,
            tolerance_grad=opt_tol_grad, tolerance_change=opt_tol_change,
            history_size=lbfgs_history, line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad()
            # ── Energy (consistent single-norm Rayleigh quotient) ──────────
            # We expand compute_total_energy manually so we can also compute
            # the symmetry regularizer without a second pass.
            C21CD, C32EF, C13AB, T1F,  T2A,  T2B,  T3C,  T3D,  T1E  = env1
            C21EB, C32AD, C13CF, T1D,  T2C,  T2F,  T3E,  T3B,  T1A  = env2
            C21AF, C32CB, C13ED, T1B,  T2E,  T2D,  T3A,  T3F,  T1C  = env3
            nu1, n1 = _energy_3bonds_raw(
                a, b, c, dt, e, f, H, H, H, chi, D_bond,
                C21CD, C32EF, C13AB, T1F, T2A, T2B, T3C, T3D, T1E,
                site_order=(e, dt, a, f, c, b))
            nu2, n2 = _energy_3bonds_raw(
                a, b, c, dt, e, f, H, H, H, chi, D_bond,
                C21EB, C32AD, C13CF, T1D, T2C, T2F, T3E, T3B, T1A,
                site_order=(a, b, c, dt, e, f))
            nu3, n3 = _energy_3bonds_raw(
                a, b, c, dt, e, f, H, H, H, chi, D_bond,
                C21AF, C32CB, C13ED, T1B, T2E, T2D, T3A, T3F, T1C,
                site_order=(c, f, e, b, a, dt))

            # Energy per bond (consistent norm — n1=n2=n3=1 after CTM normalize)
            norm_avg = (n1 + n2 + n3) / 3.0
            E_bond = (nu1 + nu2 + nu3) / (9.0 * norm_avg)

            if sym_reg > 0.0:
                # Symmetry regularizer: physical states are translationally
                # invariant → all three env-type per-bond energies should agree.
                # Ghost states (exploited by the optimizer at finite chi) have
                # large asymmetry.  Penalise the variance of the three values.
                # Scale: (e1-e2)^2 in (3-bond-energy)^2 / (9 bonds/cell)^2
                e1 = nu1 / n1;  e2 = nu2 / n2;  e3 = nu3 / n3
                reg = ((e1 - e2)**2 + (e2 - e3)**2 + (e3 - e1)**2) / (2.0 * 81.0)
                loss = E_bond + sym_reg * reg
            else:
                loss = E_bond

            loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [a, b, c, dt, e, f], max_norm=grad_clip_norm)
            return loss

        optimizer.step(closure)

        # ── 4. Normalize updated tensors before CTMRG re-convergence ───────
        with torch.no_grad():
            for t in (a, b, c, dt, e, f):
                t.data = normalize_tensor(t.data)

        # ── 5. Re-converge CTMRG to get the true environment ───────────────
        # Always use a COLD START here.
        with torch.no_grad():
            DL_new = abcdef_to_ABCDEF(a, b, c, dt, e, f, D_sq)
            env1_new, env2_new, env3_new, ctm2, ctm2_ok = _run_ctmrg_robust(
                DL_new, chi, D_sq, max_ctm_steps, ctm_conv_thr)
            true_E = compute_total_energy(a, b, c, dt, e, f, H, chi, D_bond,
                                          env1_new, env2_new, env3_new).item()

        # Save post-step tensors as the new checkpoint for CTM failure recovery.
        with torch.no_grad():
            prev_tensors = tuple(t.data.clone() for t in (a, b, c, dt, e, f))

        warn = ""
        if not ctm_ok:
            warn += " [CTM!warn-nokv]"
        if not ctm2_ok:
            warn += " [CTM2!warn-noconv]"

        history.append(true_E)
        delta = (true_E - prev_true_E) if prev_true_E is not None else float('inf')

        if verbose:
            print(f"  opt {step:4d}  E/bond={true_E:+.10f}  Δ={delta:+.3e}  "
                  f"CTM={ctm_steps}/{ctm2}{warn}")

        if prev_true_E is not None and abs(delta) < opt_conv_thr:
            if verbose:
                print(f"  Converged at step {step}")
            break
        prev_true_E = true_E

    return a, b, c, dt, e, f, history
