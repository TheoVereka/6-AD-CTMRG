"""
python==3.11.15
CUDA Toolkit==11.8

numpy==2.1.3
scipy==1.14.1
opt_einsum==3.4.0
matplotlib==3.9.4
tqdm==4.67.1
h5py==3.12.1
pytest==8.4.2
pytorch==2.5.1
"""

import numpy as np
import opt_einsum as oe
import torch
from torch.utils.checkpoint import checkpoint as _ckpt






# ── Precision control ─────────────────────────────────────────────────────────
# Three dtype globals control the entire codebase:
#   CDTYPE      – always complex (complex64 or complex128).
#                 Used for inherently complex quantities: Hamiltonian (Sy has -1j),
#                 spin operators, and any future complex-valued objects.
#   RDTYPE      – always real (float32 or float64).
#                 Used for SVD singular values, norms, and other inherently real
#                 quantities.
#   TENSORDTYPE – the dtype of the iPEPS site tensors a..f, environment tensors
#                 C/T, and all intermediate contractions.  Can be real (RDTYPE)
#                 for the S+/S- Hamiltonian formulation, or complex (CDTYPE) for
#                 the full Sx/Sy/Sz formulation.
#
# Call  set_dtype(use_double, use_real)  BEFORE allocating any tensor.
CDTYPE: torch.dtype = torch.complex128   # complex dtype for all tensors
RDTYPE: torch.dtype = torch.float64      # real dtype (SVD singular values, norms)
TENSORDTYPE: torch.dtype = torch.float64 # tensor dtype (real or complex)

# Call  set_device(device)  BEFORE allocating any tensor.
# Defaults to CPU; set to  torch.device('cuda')  from the driver script to
# run all CTMRG + energy computations on GPU.
DEVICE: torch.device = torch.device('cpu')




# ── SVD full/partial control ─────────────────────
# When True, SVD_PROPACK uses full deterministic SVD (np.linalg.svd).
# Set directly from the optimizer driver; cleared after one L-BFGS step.
_USE_FULL_SVD: bool = False

# ── SVD GPU/CPU dispatch threshold ───────────────
# For CUDA inputs, matrices with min(m,n) < this threshold are SVD'd on CPU
# and results moved back to GPU.  cuSOLVER SVD has a fixed per-call overhead
# that makes it slower than CPU LAPACK for small matrices on any GPU, and
# slower at ALL sizes on low-end mobile GPUs (MX250 etc.).
#
# Crossover (approx, depends on GPU model):
#   MX250 / laptop GPU : CPU always faster  → set to 99999
#   A100 / V100 / H100 : GPU faster for n > ~300-500 → set to 0 or 512
#
# Default 512: small CTMRG matrices (low chi/D) go to CPU, large ones to GPU.
# Change to 0 to always use GPU (best for large-chi runs on cluster A100).
# Change to 99999 to always use CPU (best on MX250 / quick local testing).
_SVD_CPU_OFFLOAD_THRESHOLD: int = 0

# ── Truncation error recording ─────────────────────────────────────────────
# Set _RECORD_TRUNC_ERROR=True (via set_record_trunc_error) before a clean-
# evaluation CTMRG to record the SVD truncation errors.  Zero cost when False
# (the recording branch is never entered during normal optimisation).  The
# accumulated errors come from the last CTMRG iteration's three trunc_rhoCCC
# calls (3 calls × 3 SVDs = 9 values), averaged to produce one scalar.
_RECORD_TRUNC_ERROR: bool = False
_TRUNC_ERRORS_ACC: list = []   # populated by trunc_rhoCCC when flag is True


def set_record_trunc_error(flag: bool) -> None:
    """Enable/disable SVD truncation-error recording in trunc_rhoCCC.

    Call with flag=True immediately before CTMRG_from_init_to_stop in a
    clean (no-grad) evaluation; call with flag=False afterwards.  When True,
    each trunc_rhoCCC call appends 3 per-SVD truncation errors to the internal
    accumulator.  After CTMRG completes, call get_last_trunc_error() to
    retrieve the average over the last 9 recorded values (= last full
    CTMRG iteration).
    """
    global _RECORD_TRUNC_ERROR, _TRUNC_ERRORS_ACC
    _RECORD_TRUNC_ERROR = flag
    if flag:
        _TRUNC_ERRORS_ACC = []


def get_last_trunc_error() -> float | None:
    """Return the mean SVD truncation error from the most recent CTMRG run.

    Truncation error per SVD is defined as:
        trunc_err = ||S_discarded|| / ||S_all||
                  = sqrt(max(0, ||M||_F^2 - ||S_kept||^2)) / ||M||_F
    where S_kept are the chi retained singular values and ||M||_F is the
    Frobenius norm of the (chi·D², chi·D²) corner product matrix.

    Returns the mean over the last 9 recorded values (last full CTMRG
    iteration: 3 environments × 3 SVDs each), or None if no data.
    """
    if not _TRUNC_ERRORS_ACC:
        return None
    recent = _TRUNC_ERRORS_ACC[-9:]
    return sum(recent) / len(recent)


def _record_trunc_err(S_all: torch.Tensor, chi: int) -> None:
    """Append one per-SVD truncation error to _TRUNC_ERRORS_ACC.

    Called only when _RECORD_TRUNC_ERROR is True (clean evaluation).
    S_all must be the FULL singular value vector (all min(m,n) values) from
    a torch.linalg.svd call on the detached corner-product matrix.
    trunc_err = ||S_all[chi:]|| / ||S_all||
    """
    with torch.no_grad():
        S_real = S_all.real
        norm_all  = torch.linalg.norm(S_real).clamp(min=1e-30)
        norm_disc = torch.linalg.norm(S_real[chi:])
        err = (norm_disc / norm_all).item()
    _TRUNC_ERRORS_ACC.append(err)

# ── rSVD backward strategy ────────────────────────────────────────────────────
#
# Controls how SVD_PROPACK computes gradients when using the randomised SVD
# forward path.  Call  set_rsvd_mode()  to change at runtime.
#
#   'full_svd'  (default) — full thin SVD in forward, zero-padding backward.
#               Exact gradient, but forward cost is O(N · min(m,n)).
#
#   'neumann'   — rSVD forward O(N²k), Neumann series for 5th term in
#               backward O(N²k × _RSVD_NEUMANN_TERMS).  Avoids O(N³) eigh.
#               Convergence rate ρ = (σ_{k+1}/σ_k)²:
#                 ρ < 0.1  (typical CTMRG):  4 terms → error < 0.01 %
#                 ρ → 1.0  (near-degenerate): many terms; fall back to 'full_svd'
#
#   'augmented' — rSVD forward computing k + _RSVD_AUGMENT_EXTRA triples O(N²·k_aug),
#               zero-padding backward over k_aug modes (no explicit 5th term).
#               The F,G cross-coupling between modes 1..k and k+1..k_aug
#               approximates the 5th term.  Exact when k_aug = min(m,n).
#               k_aug = min(k + _RSVD_AUGMENT_EXTRA, min(m,n)).
#
#   'none'      — rSVD forward O(N²k), naive k-only backward (drops 5th term).
#               Fast but inaccurate for near-degenerate spectra; gradient
#               contribution from discarded modes is completely ignored.
#
_RSVD_BACKWARD_MODE: str = 'full_svd'

_RSVD_NEUMANN_TERMS: int = 2
# Number of Neumann series terms (L) for 'neumann' mode.
# Positive → Neumann approximation (fast).  Negative → exact eigh (for reference).
# L=2 is the sweet spot: <1ms extra vs L=1, but 15× more precise at ρ=0.30.

_RSVD_POWER_ITERS: int | None = None
# Number of power (subspace) iterations in the rSVD range-finder.
# None (default) → adaptive: chosen per-call by _adaptive_power_iters(k, N).
# Integer         → override: always use exactly this many iterations.
#
# Adaptive formula calibrated from benchmarks at N=100–5120:
#   In CTMRG, the SVD matrices have size N = chi·D², k = chi, so k/N = 1/D².
#   The rSVD difficulty scales with k/N: larger ratio → discarded subspace
#   has more spectral weight → range-finder needs more power iterations.
#   Physics note: at small chi the cut falls at non-trivial SVs (larger ρ);
#   at large chi SVs at the cut edge are near-zero (ρ≈0), so fewer iters suffice.
#   Both effects are captured by the k/N proxy via D:
#
#     k/N < 2%  (D≥7): near-zero cut, trivially separated spectrum → 1 iter
#     k/N 2–5%  (D=5–6): typical production, ρ≤0.30 verified safe  → 2 iters
#     k/N 5–10% (D=4): moderate spectral weight at boundary         → 3 iters
#     k/N ≥ 10% (D≤3): non-trivial truncation, small-chi regime     → 4 iters
#
# Per-call timings at N=5120 (D=8, chi=80):
#   niter=0: 315ms   niter=1: ~500ms   niter=2: 700ms   niter=4: 1136ms
#   With 540 SVD calls/step: adaptive (niter=1) saves vs niter=2: +26 min.


def _adaptive_power_iters(k: int, N: int) -> int:
    """Return the number of rSVD power iterations appropriate for rank k out of N.

    Calibrated from benchmarks at N=100–5120 (see _RSVD_POWER_ITERS comment).
    In CTMRG: N = chi·D², k = chi  ⟹  k/N = 1/D² (independent of chi).

    Physics note: small chi → cut at non-trivial SVs (larger ρ, harder);
    large chi   → cut at near-zero SVs  (ρ≈0, easy).  Both effects enter
    through the same D, so the k/N proxy is sufficient.
    """
    ratio = k / max(N, 1)
    if   ratio < 0.02:  return 1   # D≥7: near-zero cut
    elif ratio < 0.05:  return 2   # D=5–6: typical production
    elif ratio < 0.10:  return 3   # D=4: moderate
    else:               return 4   # D≤3: non-trivial truncation


def set_rsvd_mode(mode: str,
                  neumann_terms: int = 2,
                  power_iters: int | None = None) -> None:
    """Switch the rSVD backward strategy.

    Args:
        mode:          One of 'full_svd', 'neumann', 'augmented', 'none'.
        neumann_terms: Neumann series length (positive=approx, negative=exact eigh).
        power_iters:   Number of power iterations in the rSVD range-finder.
                       None (default) → adaptive per-call via _adaptive_power_iters.
                       Integer        → fixed override for all SVD calls.
    """
    global _RSVD_BACKWARD_MODE, _RSVD_NEUMANN_TERMS, _RSVD_POWER_ITERS
    if mode not in ('full_svd', 'neumann', 'augmented', 'none'):
        raise ValueError(
            f"Unknown rSVD mode {mode!r}; choose from 'full_svd','neumann','augmented','none'"
        )
    _RSVD_BACKWARD_MODE = mode
    _RSVD_NEUMANN_TERMS = neumann_terms
    _RSVD_POWER_ITERS = power_iters






def safe_inverse(x, epsilon=1e-12):
    return x/(x**2 + epsilon**2)
    
def safe_inverse_2(x, epsilon):
    x[abs(x)<epsilon]=float('inf')
    return x.pow(-1)


def _safe_sqrt_inv_diag(S: torch.Tensor) -> torch.Tensor:
    """Compute diag(1/sqrt(S)) safely, clamping near-zero singular values.

    On GPU with float32, singular values near machine epsilon can underflow
    to zero → 1/sqrt(0) = inf → exploding CTMRG projectors → garbage
    environment → wrong energy.  Fix: clamp S[i] to at least
    S[0] * eps_machine before inverting:
        float32: threshold = S[0] × 1.19e-7
        float64: threshold = S[0] × 2.22e-16
    Only genuine underflows (S[i] represented as 0 in floating point) are
    affected; physically meaningful small singular values are untouched.
    """
    s_max    = S[0].abs().clamp(min=float(torch.finfo(S.dtype).tiny))
    S_safe   = S.clamp(min=(s_max * torch.finfo(S.dtype).eps ))
    return torch.diag(1.0 / torch.sqrt(S_safe).to(TENSORDTYPE))


def _solve_fifth_term_svd(A, U, S, V, gu, gv, eps):
    """
    Compute  Γ_A^{trunc}  from arXiv:2311.11894v3 (5th term of truncated SVD backward).

    Solves the coupled Sylvester system:
        Γ_U  =  γ S  −  A(I−VV†)γ̃      (*)
        Γ_V  =  γ̃ S  −  A†(I−UU†)γ    (**)

    Decoupled column-by-column via EVD of R = B†B  (B = A − USV†):
        (s_j² I_n − R) γ̃_j = s_j Γ_V_j + B† Γ_U_j
        γ_j = (Γ_U_j + B γ̃_j) / s_j

    5th term:  Γ_A^{trunc} = (I−UU†)γ V† + U γ̃†(I−VV†)

    This term is:
      •  zero when k = min(m,n) (full-thin SVD, B = 0, reduces exactly to the
         out-of-subspace projector terms used previously).
      •  the correct truncation correction when k < min(m,n)  (partial SVD).

    A : (m, n)  — original input matrix (saved in forward)
    U : (m, k)  — retained left singular vectors
    S : (k,)    — retained singular values (real, descending)
    V : (n, k)  — retained right singular vectors
    gu, gv : upstream gradients (m×k, n×k; zeros substituted when None)
    eps    : regularisation threshold
    """
    m, n = A.shape
    k = S.shape[0]
    Uh = U.conj().t()   # k×m
    Vh = V.conj().t()   # k×n

    # B = A(I − VV†) = A − U diag(S) V†
    B  = A - U @ torch.diag(S.to(A.dtype)) @ Vh   # m×n
    Bh = B.conj().t()                               # n×m

    # R = B†B  (n×n, positive semi-definite, symmetrized numerically)
    R = Bh @ B
    R = (R + R.conj().t()) * 0.5

    # EVD: R = Q Λ Q†  (Λ ascending)
    Lambda, Q = torch.linalg.eigh(R)   # Lambda: n, Q: n×n
    Qh = Q.conj().t()

    # RHS_j = s_j · gv_j + B† · gu_j  ⟹  RHS matrix (n×k)
    RHS = gv * S.to(gv.dtype).unsqueeze(0) + Bh @ gu   # n×k

    # denom[i,j] = S[j]² − Λ[i]   (n×k)
    S2    = S ** 2                                          # k  (real)
    denom = S2.unsqueeze(0) - Lambda.unsqueeze(1)          # (1,k) − (n,1) = n×k

    # Regularise near-zero denominators (touching a multiplet boundary)
    invDenom_reg = safe_inverse(denom, epsilon=eps)  # n×k

    # γ̃ = Q [(Q†·RHS) / denom]   (n×k)
    gamma_tilde = Q @ (Qh @ RHS * invDenom_reg)

    # γ_j = (gu_j + B·γ̃_j) / s_j   (m×k)
    # Guard: clamp S so near-zero retained singular values don't produce inf/NaN
    eps_s  = S[0].abs().clamp(min=float(torch.finfo(S.dtype).tiny)) \
             * torch.finfo(S.dtype).eps
    S_safe = S.clamp(min=eps_s)
    gamma = (gu + B @ gamma_tilde) / S_safe.to(A.dtype).unsqueeze(0)

    # 5th term:  (I−UU†)γ V†  +  U γ̃†(I−VV†)
    PU_gamma      = gamma - U @ (Uh @ gamma)            # (I−UU†)γ : m×k
    gamma_tilde_h = gamma_tilde.conj().t()              # k×n
    gtilde_h_PV   = gamma_tilde_h - (gamma_tilde_h @ V) @ Vh  # γ̃†(I−VV†) : k×n

    return PU_gamma @ Vh + U @ gtilde_h_PV              # m×n


def _solve_fifth_term_neumann(A, U, S, V, gu, gv, n_terms: int):
    """
    Fast approximate 5th term via Neumann series — avoids O(N³) eigh.

    Replaces the exact Sylvester solve in ``_solve_fifth_term_svd`` with a
    truncated Neumann series expansion of ``(s_j² I − B†B)⁻¹``:

        γ̃_j  ≈  s_j⁻²  Σ_{l=0}^{L-1}  (B†B / s_j²)^l  RHS_j

    Cost per Neumann term: two matrix products B @ x (m×n × n×k = m×k) and
    B† @ y (n×m × m×k = n×k), i.e. O(2mnk).  Total: O(2L·mnk) vs O(N³) eigh.

    For D=5, chi=25 (N=625, k=25):
        Exact eigh  :  ~244 M ops  (625³)
        L=4 Neumann :   ~78 M ops  (2·4·625·625·25)   — 3× faster
        L=2 Neumann :   ~39 M ops                       — 6× faster

    Convergence: geometric series with ratio ρ = ‖B†B‖ / s_k² ≤ (σ_{k+1}/σ_k)².
      • CTMRG typical (ρ < 0.1): L=4 gives error < ρ^4 ≈ 0.01%.
      • Near-degenerate (ρ → 1)  : convergence slow; fall back to exact eigh
        (:func:`_solve_fifth_term_svd`) or use 'augmented'/'full_svd' mode.

    n_terms : int — number of Neumann terms L (must be ≥ 1).
    """
    m, n = A.shape
    k = S.shape[0]
    Uh = U.conj().t()   # k×m
    Vh = V.conj().t()   # k×n

    # B = A − U S V†   (residual; captures the discarded subspace)
    B  = A - U @ torch.diag(S.to(A.dtype)) @ Vh   # m×n
    Bh = B.conj().t()                               # n×m

    # RHS matrix (n×k): s_j · gv_j + B† · gu_j
    RHS = gv * S.to(gv.dtype).unsqueeze(0) + Bh @ gu   # n×k

    # ── Neumann series ────────────────────────────────────────────────────────
    # γ̃ = Σ_{l=0}^{L-1}  (B†B)^l RHS / s_j^{2(l+1)}
    # Recurrence: rhs ← (B†B / s_j²) · rhs,  accumulate into γ̃.
    #
    # Guard: clamp S so that near-zero retained singular values (e.g. from
    # pad+noise initialisation or rank-deficient matrices) don't produce
    # inf/NaN in the 1/s² or 1/s divisions.  Threshold = S[0]·eps_machine,
    # matching _safe_sqrt_inv_diag convention.
    eps_s  = S[0].abs().clamp(min=float(torch.finfo(S.dtype).tiny)) \
             * torch.finfo(S.dtype).eps
    S_safe = S.clamp(min=eps_s)
    S2     = (S_safe ** 2).unsqueeze(0)   # (1, k) — broadcast over n rows
    S2_inv = 1.0 / S2                     # precompute reciprocal (used L times)

    # ── Spectral ratio pre-check ─────────────────────────────────────────────
    # Estimate ρ = ||B†B||₂ / s_k² ≈ (σ_{k+1}/σ_k)² via one mat-vec product.
    # If ρ ≥ 0.5 the Neumann series converges too slowly (truncation error
    # ≥ ρ^L ≥ 25% at L=2) — fall back to exact eigh Sylvester solver.
    #BECAUSE IT WON'T WAY SLOWER!
    _probe = Bh @ (B @ RHS[:, -1:])       # n×1, cost O(2mn) — negligible
    rho_est = _probe.norm() / (RHS[:, -1:].norm().clamp(min=1e-30) * S2[0, -1])
    if rho_est > 0.5:
        return _solve_fifth_term_svd(A, U, S, V, gu, gv,
                                     S[0].abs().clamp(min=float(torch.finfo(S.dtype).tiny))
                                     * torch.finfo(S.dtype).eps)

    rhs  = RHS * S2_inv                   # l=0 term: RHS / s_j²
    gamma_tilde = rhs.clone()

    for _ in range(n_terms - 1):
        rhs_prev_norm = rhs.norm()
        rhs = Bh @ (B @ rhs)             # apply B†B  (cost: O(2mnk))
        rhs = rhs * S2_inv               # rhs = (B†B/s²)^{l+1} RHS/s²
        # Divergence guard: if this term is larger than the previous one
        # the spectral ratio ρ ≥ 1 and the series won't converge — stop.
        # Fall back to exact eigh solver instead of using bad partial sum.
        #BECAUSE IT WON'T WAY SLOWER!
        if rhs.norm() > rhs_prev_norm:
            return _solve_fifth_term_svd(A, U, S, V, gu, gv,
                                         S[0].abs().clamp(min=float(torch.finfo(S.dtype).tiny))
                                         * torch.finfo(S.dtype).eps)
        gamma_tilde = gamma_tilde + rhs  # accumulate

    # Back-substitute: γ_j = (gu_j + B γ̃_j) / s_j
    gamma = (gu + B @ gamma_tilde) / S_safe.to(A.dtype).unsqueeze(0)   # m×k

    # Assemble: Γ_A^{trunc} = (I−UU†)γ V†  +  U γ̃†(I−VV†)
    PU_gamma      = gamma - U @ (Uh @ gamma)
    gamma_tilde_h = gamma_tilde.conj().t()
    gtilde_h_PV   = gamma_tilde_h - (gamma_tilde_h @ V) @ Vh

    return PU_gamma @ Vh + U @ gtilde_h_PV              # m×n








def set_dtype(use_double: bool, use_real: bool = True) -> None:
    """Switch all core computations between float32/complex64 and float64/complex128.

    Must be called BEFORE any tensor is allocated — ideally right after import.

    Args:
        use_double: ``True``  → complex128 / float64 (double precision).
                    ``False`` → complex64  / float32 (single precision, default).
        use_real:   ``True``  → TENSORDTYPE = RDTYPE  (real iPEPS tensors, S+/S- Hamiltonian).
                    ``False`` → TENSORDTYPE = CDTYPE  (complex iPEPS tensors, Sx/Sy/Sz Hamiltonian).
    """
    global CDTYPE, RDTYPE, TENSORDTYPE

    if use_double:
        CDTYPE = torch.complex128
        RDTYPE = torch.float64
    else:
        CDTYPE = torch.complex64
        RDTYPE = torch.float32

    if use_real:
        TENSORDTYPE = RDTYPE
    else:
        TENSORDTYPE = CDTYPE


def set_device(device) -> None:
    """Switch all core tensor allocations to a different compute device.

    Must be called BEFORE any tensor is allocated — ideally right after
    ``set_dtype()``.

    Args:
        device: Anything accepted by ``torch.device()``:
            ``'cpu'``, ``'cuda'``, ``'cuda:0'``, or a ``torch.device``
            instance.  Passing ``None`` is treated as ``'cpu'``.
    """
    global DEVICE
    DEVICE = torch.device(device) if device is not None else torch.device('cpu')


def normalize_tensor(tensor, *, rtol: float | None = None, atol: float | None = None):
    
    """
    Normalise a tensor in-place-style by its Frobenius (L2) norm.

    Divides every element by ``||tensor||_F``.  If the norm is exactly zero
    (e.g. an all-zero tensor) the input is returned unchanged to avoid a
    division-by-zero NaN.

    GPU-friendly: no GPU→CPU synchronisation (no .item(), no Python ``if``
    on a CUDA scalar).  Uses ``clamp`` to guard against zero/NaN norms
    instead of branching.

    Args:
        tensor (torch.Tensor): Any complex or real tensor of arbitrary shape.

    Returns:
        torch.Tensor: A new tensor with the same shape and dtype as the input
        whose Frobenius norm is 1, or the original tensor if its norm is 0.
    """
    # Frobenius norm (for arbitrary-rank tensors). Returns a real scalar.
    norm = torch.linalg.norm(tensor)

    # Clamp norm to a safe minimum to avoid division-by-zero.
    # For NaN/Inf/zero norms this produces a harmless (possibly large) result
    # rather than NaN — same logical effect as the old early-return branches
    # but without GPU→CPU sync.
    safe_norm = norm.clamp(min=1e-30)
    return tensor / safe_norm


def normalize_single_layer_tensor_for_double_layer(
    tensor: torch.Tensor,
    *,
    rtol: float | None = None,
    atol: float | None = None,
) -> torch.Tensor:
    """Rescale a single-layer site tensor so its *unnormalized* double-layer has unit Frobenius norm.

    In this codebase, CTMRG consumes double-layer site tensors (A..F) produced by
    ``abcdef_to_ABCDEF`` which *then* normalizes each A..F to unit Frobenius norm.
    If energy/contraction routines use the original single-layer tensors directly,
    a convention mismatch can make denominators like <iPEPS|iPEPS> far from 1 even
    when the CTMRG environment itself is well-normalized.

    This helper chooses a single-layer rescaling such that the corresponding
    unnormalized double-layer tensor already has ||DL||_F ≈ 1, making single-layer
    contractions consistent with the CTMRG convention.

    Scaling rule:
      DL(t) = contract(t, conj(t)) so ||DL(α t)||_F = |α|^2 ||DL(t)||_F.
      Choose α = 1/sqrt(||DL(t)||_F) ⇒ ||DL(α t)||_F = 1.

    Fast formula for ||DL||_F:
      DL[ux,vy,wz] = Σ_p t[u,v,w,p] * t*[x,y,z,p]
      ||DL||_F² = Σ_{ux,vy,wz} |DL[ux,vy,wz]|²
                = Σ_{p,q} (Σ_{uvw} t[u,v,w,p]·t*[u,v,w,q])·(Σ_{xyz} t*[x,y,z,p]·t[x,y,z,q])
                = Σ_{p,q} |M†M[p,q]|²  =  ||M†M||_F²
      where M[i,p] = t[u,v,w,p] (i = flattened virtual index).

      Cost: O(D³·d²) vs O(D⁶·d²) for the explicit contraction.
      For D=5, d=2: 50 vs 62,500 multiplications — 1,250× cheaper.

    GPU-friendly: no GPU→CPU sync.
    """
    # Reshape tensor (D,D,D,d) → M of shape (D³, d)
    M = tensor.reshape(-1, tensor.shape[-1])  # (D^3, d)
    # M†M is (d, d) — small matrix, O(D^3 * d^2) cost
    MtM = M.conj().t() @ M  # (d, d)
    # ||DL||_F = ||M†M||_F.
    # Use torch.linalg.norm: returns a REAL scalar for both real and complex
    # input, with correct Wirtinger gradients for complex.  Avoids the fragile
    # (.real) trick which only passes gradient through the real component of a
    # complex tensor — safe here because (MtM * MtM†) is analytically purely
    # real, but breaks silently under floating-point noise in complex runs.
    dl_norm = torch.linalg.norm(MtM)  # real scalar, ≥ 0, correct for real+complex

    # GPU-friendly: clamp dl_norm to a safe minimum to avoid division-by-zero
    # and sqrt(0).  No GPU→CPU sync.
    safe_dl_norm = dl_norm.clamp(min=1e-30)
    return tensor / torch.sqrt(safe_dl_norm)








def _keep_multiplets(U,S,V,chi,eps_multiplet,abs_tol):
    # estimate the chi_new 
    chi_new= chi
    # regularize by discarding small values
    gaps=S[:chi+1].clone().detach()
    # S[S < abs_tol]= 0.
    gaps[gaps < abs_tol]= 0.
    # compute gaps and normalize by larger sing. value. Introduce cutoff
    # for handling vanishing values set to exact zero
    gaps=(gaps[:chi]-S[1:chi+1])/(gaps[:chi]+1.0e-16)
    gaps[gaps > 1.0]= 0.

    if gaps[chi-1] < eps_multiplet:
        # the chi is within the multiplet - find the largest chi_new < chi
        # such that the complete multiplets are preserved
        for i in range(chi-1,-1,-1):
            if gaps[i] > eps_multiplet:
                chi_new= i
                break

    St = S[:chi].clone()
    St[chi_new+1:]=0.

    Ut = U[:, :St.shape[0]].clone()
    Ut[:, chi_new+1:]=0.
    Vt = V[:, :St.shape[0]].clone()
    Vt[:, chi_new+1:]=0.
    return Ut, St, Vt
    


from torch import _linalg_utils as _utils, Tensor

class SVD_PROPACK(torch.autograd.Function):
    @staticmethod
    def forward(self, A, k, k_extra, rel_cutoff, v0, use_full_svd):
        r"""
        :param M: square matrix :math:`N \times N`
        :param k: desired rank (must be smaller than :math:`N`)
        :param use_full_svd: if True, use full deterministic SVD (np.linalg.svd)
                             instead of the randomized projection SVD.
        :type M: torch.Tensor
        :type k: int
        :type use_full_svd: bool
        :return: leading k left eigenvectors U, singular values S, and right 
                 eigenvectors V
        :rtype: torch.Tensor, torch.Tensor, torch.Tensor

        **Note:** `depends on scipy`

        Return leading k-singular triples of a matrix M, by computing 
        the partial SVD decomposition using SciPy's PROPACK wrapper up to rank k.
        """
        # Dense LAPACK SVD via numpy — identical numerical backend to the
        # original torch.linalg.svd calls this replaced.
        # ARPACK and PROPACK are NOT used: ARPACK silently falls back to
        # scipy.linalg.eig (wrong algorithm) when k >= N-1, and PROPACK's
        # Fortran ZLASCL routine crashes on ill-conditioned matrices.
        # The CTMRG matrices are at most ~100×100; dense SVD is fast.
        _rdtype = torch.float64 if A.dtype in (torch.complex128, torch.float64) else torch.float32

        m_dim, n_dim = A.shape
        #k_total = min(k + k_extra, min(m_dim, n_dim))
        #q = max(k_total, 1)

        # print(f"Performing SVD with k={k}, k_extra={k_extra}, rel_cutoff={rel_cutoff.item()}, m_dim={m_dim}, n_dim={n_dim}, k_total={k_total}")


        if _RSVD_BACKWARD_MODE != 'full_svd' and not use_full_svd:
            # ── Stable randomised SVD forward ──────────────────────────────────
            # Range-finder runs under no_grad (Q is treated as constant).
            # Gradient flows only through  B_proj = Q†A  (q×n, differentiable).
            #
            # Oversampling q = 2k: standard Halko-Martinsson-Tropp (2011)
            # recommendation.  q = k fails catastrophically at ρ ≥ 0.3
            # (cosine 0.065 → 0.9996 at ρ=0.95 when going from q=k to q=2k).
            #
            # Power iterations: configurable via _RSVD_POWER_ITERS.
            # At N≥2000 with k/N<5%, niter=0 suffices for ρ≤0.80.
            # Default niter=2 is safe for all practical CTMRG spectra.
            # At D=8 chi=80 (N=5120): saves 3.9 min/step vs niter=4.
            q_over = min(2 * k, min(m_dim, n_dim))
            # Adaptive power iterations: None → choose from k/N ratio;
            # integer override → use exactly that many.
            _niter = (_adaptive_power_iters(k, min(m_dim, n_dim))
                      if _RSVD_POWER_ITERS is None else _RSVD_POWER_ITERS)
            with torch.no_grad():
                Rand = torch.randn(n_dim, q_over, dtype=A.dtype, device=A.device)
                X    = A @ Rand
                Q, _ = torch.linalg.qr(X)           # m × q_over
                for _ in range(_niter):              # power iterations
                    Q, _ = torch.linalg.qr(A.mH @ Q) # n × q_over
                    Q, _ = torch.linalg.qr(A    @ Q) # m × q_over

            # Gradient flows through B_proj = Q†A  (Q treated as constant)
            B_proj = Q.mH @ A                        # q_over × n_dim
            U_B, S_all, Vh_all = torch.linalg.svd(B_proj, full_matrices=False)
            U_all = Q @ U_B                          # m × q_over
            V_all = Vh_all.mH                        # n × q_over

            # Descending sort (cuSOLVER order not guaranteed)
            order = torch.argsort(S_all, descending=True)
            S_all = S_all[order];  U_all = U_all[:, order];  V_all = V_all[:, order]

            if _RSVD_BACKWARD_MODE == 'neumann':
                # Save top-k triples + A for Neumann 5th-term backward
                self.save_for_backward(U_all[:, :k], S_all[:k], V_all[:, :k],
                                       A.detach(), rel_cutoff)

            elif _RSVD_BACKWARD_MODE == 'augmented':
                self.save_for_backward(U_all[:, :q_over], S_all[:q_over], V_all[:, :q_over],
                                       None, rel_cutoff)

            return U_all[:, :k], S_all[:k], V_all[:, :k]

        else:
            #print("Full", end='')
            # Full thin SVD.  cuSOLVER has a fixed per-call overhead that makes
            # it slower than CPU LAPACK for matrices below the crossover size
            # (varies by GPU: always on MX250, n<~500 on A100/V100).
            # Use _SVD_CPU_OFFLOAD_THRESHOLD to control dispatch per hardware.
            dev = A.device
            n_min = min(A.shape)
            if dev.type == 'cuda' and n_min < _SVD_CPU_OFFLOAD_THRESHOLD:
                A_cpu = A.detach().cpu()
                U_cpu, S_cpu, Vh_cpu = torch.linalg.svd(A_cpu, full_matrices=False)
                U = U_cpu.to(dev)
                S = S_cpu.to(dev)
                V = Vh_cpu.mH.to(dev)
            else:
                U, S, V = torch.linalg.svd(A.detach(), full_matrices=False)
                V = V.conj().T

            self.save_for_backward(U, S, V, None, rel_cutoff)

            return U[:,:k], S[:k], V[:,:k]

    @staticmethod
    def backward(self, gu, gsigma, gv):
        r"""
        :param gu: gradient on U
        :type gu: torch.Tensor
        :param gsigma: gradient on S
        :type gsigma: torch.Tensor
        :param gv: gradient on V
        :type gv: torch.Tensor
        :return: gradient
        :rtype: torch.Tensor

        Computes backward gradient for SVD, adopted from 
        https://github.com/pytorch/pytorch/blob/v1.10.2/torch/csrc/autograd/FunctionsManual.cpp
        
        For complex-valued input there is an additional term, see

            * https://giggleliu.github.io/2019/04/02/einsumbp.html
            * https://arxiv.org/abs/1909.02659

        The backward is regularized following
        
            * https://github.com/wangleiphy/tensorgrad/blob/master/tensornets/adlib/svd.py
            * https://arxiv.org/abs/1903.09650

        using 

        .. math:: 
            S_i/(S^2_i-S^2_j) = (F_{ij}+G_{ij})/2\ \ \textrm{and}\ \ S_j/(S^2_i-S^2_j) = (F_{ij}-G_{ij})/2
        
        where 
        
        .. math:: 
            F_{ij}=1/(S_i-S_j),\ G_{ij}=1/(S_i+S_j)
        """
        # 
        # TORCH_CHECK(compute_uv,
        #    "svd_backward: Setting compute_uv to false in torch.svd doesn't compute singular matrices, ",
        #    "and hence we cannot compute backward. Please use torch.svd(compute_uv=True)");
        u, sigma, v, A_input, eps = self.saved_tensors
        m = u.size(0)      # first dim of A
        n = v.size(0)      # second dim of A
        k = sigma.size(0)  # number of retained singular triples
        sigma_scale = sigma[0]

        # eps was stored with real dtype; extract .real to guard legacy paths.
        _eps_real   = eps.real if eps.is_complex() else eps
        # Keep eps_val as a scalar tensor to avoid GPU→CPU sync.
        eps_val     = (sigma_scale * _eps_real).clamp(min=1e-30)

        sigma_inv = safe_inverse_2(sigma.clone(), sigma_scale * _eps_real)

        if A_input is None:
                # Forward returned only k_return singular triples; gu and gv have width
            # k_return ≤ k.  Pad them with zeros so the F,G cross-coupling between
            # kept and discarded singular values is captured correctly.
            k_return = gsigma.size(0)  # fallback to k for safety
            if k_return < k:
                _z = lambda g, cols: torch.cat(
                    [g, g.new_zeros(*g.shape[:-1], cols)], dim=-1)
                if gu is not None:
                    gu = _z(gu, k - k_return)
                if gv is not None:
                    gv = _z(gv, k - k_return)
                if gsigma is not None:
                    gsigma = torch.cat(
                        [gsigma, gsigma.new_zeros(k - k_return)], dim=-1)

            # u, sigma, v are now full-rank (min(m,n) columns) so the
            # "free subspace" narrowing below will normally be a no-op.
            if (u.size(-2)!=u.size(-1)) or (v.size(-2)!=v.size(-1)):
                # We ignore the free subspace here because possible base vectors cancel
                # each other, e.g., both -v and +v are valid base for a dimension.
                # Don't assume behavior of any particular implementation of svd.
                u = u.narrow(-1, 0, k)
                v = v.narrow(-1, 0, k)
                if not (gu is None): gu = gu.narrow(-1, 0, k)
                if not (gv is None): gv = gv.narrow(-1, 0, k)

        uh = u.conj().transpose(-2, -1)
        vh = v.conj().transpose(-2, -1)
        

        
        # ── 1st term: σ-gradient ─────────────────────────────────────────────
        if gsigma is not None:
            sigma_term = u * gsigma.unsqueeze(-2) @ vh
        else:
            sigma_term = torch.zeros(m, n, dtype=u.dtype, device=u.device)

        if (gu is None) and (gv is None):
            return sigma_term, None, None, None, None, None
        
        # k×k F, G matrices (within-subspace rotation coupling)
        F = safe_inverse(sigma.unsqueeze(-2) - sigma.unsqueeze(-1),
                         sigma_scale * _eps_real)
        F.diagonal(0, -2, -1).fill_(0)
        G = safe_inverse(sigma.unsqueeze(-2) + sigma.unsqueeze(-1),
                         sigma_scale * _eps_real)
        G.diagonal(0, -2, -1).fill_(0)
        
        # ── 2nd term: within-subspace U rotation ─────────────────────────────
        if gu is not None:
            guh    = gu.conj().transpose(-2, -1)
            u_term = u @ ((F + G).mul(uh @ gu - guh @ u)) * 0.5 @ vh
        else:
            u_term = torch.zeros(m, n, dtype=u.dtype, device=u.device)

        # ── 3rd term: within-subspace V rotation ─────────────────────────────
        if gv is not None:
            gvh    = gv.conj().transpose(-2, -1)
            v_term = u @ ((F - G).mul(vh @ gv - gvh @ v)) @ vh * 0.5
        else:
            v_term = torch.zeros(m, n, dtype=u.dtype, device=u.device)
        
        # ── 4th term: complex phase correction (arXiv:1909.02659) ────────────
        dA = u_term + sigma_term + v_term
        if (u.is_complex() or v.is_complex()) and (gu is not None):
            L = (uh @ gu).diagonal(0, -2, -1).clone()
            L.real.zero_()
            L.imag.mul_(sigma_inv)
            dA = dA + (u * L.unsqueeze(-2)) @ vh

        # ── 5th term: Γ_A^{trunc}  (arXiv:2311.11894v3) ───────────────────────
        # Solves the coupled Sylvester system for γ, γ̃:
        #   Γ_U = γ S − A(I−VV†)γ̃      Γ_V = γ̃ S − A†(I−UU†)γ
        # Then: Γ_A^{trunc} = (I−UU†)γV† + Uγ̃†(I−VV†)
        #
        # When k = min(m,n) (full-thin branch, B = A−USV† = 0) this reduces
        # exactly to the old proj_on_ortho terms and gives zero extra correction.
        # When k < min(m,n) (partial SVD branch or ARPACK) this provides the
        # previously-missing truncation correction.

        if A_input is not None:
            _gu = gu if gu is not None else torch.zeros(m, k, dtype=u.dtype, device=u.device)
            _gv = gv if gv is not None else torch.zeros(n, k, dtype=u.dtype, device=u.device)
            if _RSVD_NEUMANN_TERMS > 0:
                # Fast Neumann series: O(2·L·mnk) instead of O(N³) eigh
                dA += _solve_fifth_term_neumann(A_input, u, sigma, v, _gu, _gv,
                                                _RSVD_NEUMANN_TERMS)
            elif _RSVD_NEUMANN_TERMS < 0:
                # Exact eigh (reference / validation only)
                dA += _solve_fifth_term_svd(A_input, u, sigma, v, _gu, _gv, eps_val)
            # == 0 : skip 5th term entirely ('none' mode should not reach here
            #        since it saves A_input=None, but guard anyway)
            



        return dA, None, None, None, None, None





def truncated_svd_propack(M, chi, chi_extra, rel_cutoff, v0=None,
    abs_tol=1.0e-14,  keep_multiplets=False, \
    eps_multiplet=1e-12, ):
    r"""
    :param M: square matrix of dimensions :math:`N \times N`
    :param chi: desired maximal rank :math:`\chi`
    :param abs_tol: absolute tolerance on minimal singular value 
    :param rel_tol: relative tolerance on minimal singular value
    :param keep_multiplets: truncate spectrum down to last complete multiplet
    :param eps_multiplet: allowed splitting within multiplet
    :param verbosity: logging verbosity
    :type M: torch.tensor
    :type chi: int
    :type abs_tol: float
    :type rel_tol: float
    :type keep_multiplets: bool
    :type eps_multiplet: float
    :type verbosity: int
    :return: leading :math:`\chi` left singular vectors U, right singular vectors V, and
             singular values S
    :rtype: torch.tensor, torch.tensor, torch.tensor

    **Note:** `depends on scipy`

    Returns leading :math:`\chi`-singular triples of a matrix M,
    by computing the partial symmetric decomposition of :math:`H=M^TM` as :math:`H= UDU^T` 
    up to rank :math:`\chi`. Returned tensors have dimensions 

    .. math:: dim(U)=(N,\chi),\ dim(S)=(\chi,\chi),\ \textrm{and}\ dim(V)=(N,\chi)

    .. note::
        This function does not support autograd.
    """
    # rel_cutoff must always be real-typed: singular values are real, and
    # safe_inverse / safe_inverse_2 use < comparisons that are undefined for
    # complex tensors (raises "lt_cpu not implemented for 'ComplexDouble'").
    _rdtype = torch.float64 if M.dtype in (torch.complex128, torch.float64) else torch.float32

    # Clamp all tolerances to be at least proportional to the dtype's machine
    # epsilon.  Hardcoded 1e-12/1e-14 values are fine for float64 (eps≈2e-16)
    # but are effectively zero for float32 (eps≈1.2e-7), giving no regularisation.
    _eps_dtype = float(torch.finfo(_rdtype).eps)
    rel_cutoff    = max(rel_cutoff,    _eps_dtype)
    abs_tol       = max(abs_tol,       _eps_dtype)
    eps_multiplet = max(eps_multiplet, _eps_dtype)

    U, S, V = SVD_PROPACK.apply(M, chi, chi_extra, torch.as_tensor([rel_cutoff], dtype=_rdtype, device=M.device), v0, _USE_FULL_SVD)


    # estimate the chi_new 
    if keep_multiplets and chi<S.shape[0]:
        return _keep_multiplets(U,S,V,chi,eps_multiplet,abs_tol)

    St = S[:min(chi,S.shape[0])]
    Ut = U[:, :St.shape[0]]
    Vt = V[:, :St.shape[0]]

    return Ut, St, Vt

"""
use of truncate_svd_propack:

truncated_svd_propack(M, chi,
                    chi_extra=1,
                    rel_cutoff=1e-12,
                    v0=None,
                    keep_multiplets=False,
                    abs_tol=1e-14,
                    eps_multiplet=1e-12)  # returns U, S, V of M= USV^dag

"""










def initialize_abcdef(initialize_way:str, D_bond:int, d_PHYS:int, noise_scale:float):
    """
    Initialise the six single-layer iPEPS site tensors a, b, c, d, e, f.

    Each tensor has shape ``(D_bond, D_bond, D_bond, d_PHYS)`` where the first
    three indices are virtual bond indices (one per incoming edge of the
    honeycomb-like unit cell) and the last is the physical index.
    All tensors are normalised to unit Frobenius norm after creation.

    Args:
        initialize_way (str): Initialisation strategy.
            ``'random'``  — independent complex-Gaussian random tensors.
            ``'product'`` — product-state initialisation (not yet implemented).
            ``'singlet'`` — Mz=0 singlet initialisation (not yet implemented).
        D_bond (int): Virtual bond dimension D.
        d_PHYS (int): Physical Hilbert-space dimension.
        noise_scale (float): Amplitude of added noise (used by non-random
            initialisations; currently unused for ``'random'``).

    Returns:
        Tuple[torch.Tensor, ...]: ``(a, b, c, d, e, f)``, each of shape
        ``(D_bond, D_bond, D_bond, d_PHYS)``, dtype ``torch.float32``.

    Raises:
        ValueError: If ``initialize_way`` is not a recognised strategy.
    """

    if initialize_way == 'random' :

        a = torch.randn(D_bond, D_bond, D_bond, d_PHYS, dtype=TENSORDTYPE, device=DEVICE)
        b = torch.randn(D_bond, D_bond, D_bond, d_PHYS, dtype=TENSORDTYPE, device=DEVICE)
        c = torch.randn(D_bond, D_bond, D_bond, d_PHYS, dtype=TENSORDTYPE, device=DEVICE)
        d = torch.randn(D_bond, D_bond, D_bond, d_PHYS, dtype=TENSORDTYPE, device=DEVICE)
        e = torch.randn(D_bond, D_bond, D_bond, d_PHYS, dtype=TENSORDTYPE, device=DEVICE)
        f = torch.randn(D_bond, D_bond, D_bond, d_PHYS, dtype=TENSORDTYPE, device=DEVICE)
        global _USE_FULL_SVD
        _USE_FULL_SVD = True

    elif initialize_way == 'product' : # product state but always with small noise

        raise NotImplementedError("'product' initialisation is not yet implemented")

    elif initialize_way == 'singlet' : # Mz=0 sector's singlet state representable as PEPS

        raise NotImplementedError("'singlet' initialisation is not yet implemented")

    else :

        raise ValueError(f"Invalid initialize_way: {initialize_way}")
    
    return a,b,c,d,e,f


# ── Néel-symmetrized ansatz ───────────────────────────────────────────────────

def symmetrize_virtual_legs(a: torch.Tensor) -> torch.Tensor:
    """Symmetrize the 3 virtual legs of tensor a of shape (D, D, D, d_PHYS).

    Averages over all 6 permutations of the first 3 indices (S₃ group).
    The result satisfies a[i,j,k,s] = a[π(i,j,k),s] for all π ∈ S₃.
    This operation is differentiable; the gradient is projected to the
    symmetric subspace automatically via the chain rule.
    """
    return (  a
            + a.permute(0, 2, 1, 3)   # swap j ↔ k
            + a.permute(1, 0, 2, 3)   # swap i ↔ j
            + a.permute(1, 2, 0, 3)   # cycle i→j→k→i
            + a.permute(2, 0, 1, 3)   # cycle i→k→j→i
            + a.permute(2, 1, 0, 3)   # swap i ↔ k
           ) / 6.0


def neel_abcdef_from_a(a_sym: torch.Tensor) -> tuple:
    """Derive the 6-site unit cell (a, b, c, d, e, f) from a single symmetrized tensor.

    Néel order on the honeycomb lattice:
        sublattice 1 (A, C, E): a_sym
        sublattice 2 (B, D, F): π-rotated via U = iσ_y

    The π-rotation matrix U acts on the physical leg:
        b[i,j,k,s'] = Σ_s U[s',s] a_sym[i,j,k,s]

    For spin-1/2 (d_PHYS=2):
        U = [[0, 1], [-1, 0]] = iσ_y

    For general spin-S (d_PHYS = 2S+1):
        U[s, d_PHYS-1-s] = (-1)^s   (all other entries zero)

    Key properties:
      • U is real and orthogonal (U^T U = I), so the double-layer tensor
        B_DL = A_DL.  CTMRG sees a uniform state, but the open tensors
        carry the physical rotation, giving correct Néel energy gradients.
      • U S_z U^T = -S_z  ⟹  B-sublattice sites have opposite spin.
      • Unlike .flip(-1) (permutation P with P^2=I, no sign), U has
        essential minus signs that prevent the optimizer from collapsing
        to b=a.  For any a=[α,β]: b=[β,-α] ≠ a unless a=0.

    Args:
        a_sym: Symmetrized tensor of shape (D, D, D, d_PHYS).

    Returns:
        (a, b, c, d, e, f) where a=c=e=a_sym, b=d=f=U·a_sym.
    """
    d_PHYS = a_sym.shape[-1]
    # Build the π-rotation matrix: U[s, d-1-s] = (-1)^s
    U = torch.zeros(d_PHYS, d_PHYS, dtype=a_sym.dtype, device=a_sym.device)
    for s in range(d_PHYS):
        U[s, d_PHYS - 1 - s] = (-1.0) ** s
    # Apply U on the physical (last) leg:  b[...,s'] = Σ_s U[s',s] a[...,s]
    b = torch.einsum('ij,...j->...i', U, a_sym)
    return a_sym, b, a_sym, b, a_sym, b


def initialize_neel(D_bond: int, d_PHYS: int, noise_scale: float = 1.0) -> torch.Tensor:
    """Initialize a single symmetric tensor for the Néel ansatz.

    Creates a random tensor of shape (D_bond, D_bond, D_bond, d_PHYS),
    symmetrizes the 3 virtual legs, and returns it.  The result lives in
    the totally-symmetric subspace with D*(D+1)*(D+2)/6 × d_PHYS effective
    degrees of freedom.

    Args:
        D_bond: Virtual bond dimension.
        d_PHYS: Physical Hilbert-space dimension.
        noise_scale: Amplitude of the random initialization (unused, kept
            for API compatibility).

    Returns:
        Symmetric tensor of shape (D_bond, D_bond, D_bond, d_PHYS).
    """
    global _USE_FULL_SVD
    a = torch.randn(D_bond, D_bond, D_bond, d_PHYS, dtype=TENSORDTYPE, device=DEVICE)
    a = symmetrize_virtual_legs(a)
    _USE_FULL_SVD = True
    return a


# ── C6-Ypi single-tensor ansatz ───────────────────────────────────────────────
# C6: derives all 6 site tensors from a single tensor a by C6 (sixty-degree)
#     rotation combined with the pi-Y rotation (U = iσ_y on the physical leg)
#     on the B-sublattice sites (d, b, f).
# Previously called "plaquette" / "plaq" ansatz.

def symmetrize_plaq_legs(a: torch.Tensor) -> torch.Tensor:
    """DEPRECATED. Leg-1↔leg-2 symmetrization — no longer used by any ansatz.

    Kept for backwards compatibility only.  Do not call in new code.
    Returns (a + a.permute(0,2,1,3)) / 2.
    """
    import warnings
    warnings.warn(
        "symmetrize_plaq_legs is deprecated and will be removed.  "
        "The c6ypi ansatz does not use 2-leg symmetrization.",
        DeprecationWarning, stacklevel=2)
    return (a + a.permute(0, 2, 1, 3)) / 2.0


def c6ypi_abcdef_from_a(a_raw: torch.Tensor) -> tuple:
    """Derive (a, b, c, d, e, f) from a single tensor using C6 + pi-Y symmetry.

    The ansatz constructs the full 6-site unit cell from one tensor a_raw:
      - A-sublattice sites (a, c, e): related by C3 virtual-leg rotation.
      - B-sublattice sites (d, b, f): same C3 rotation PLUS a pi-Y rotation
        U on the physical leg (U[s, d-1-s] = (-1)^s, equivalent to iσ_y
        for spin-1/2), implementing the C6 + π-Y symmetry of the honeycomb.

    Specifically:
        a_raw                    → site A
        a_raw.permute(1,2,0,3)  → e (C3 rotation)
        a_raw.permute(2,0,1,3)  → c (C3² rotation)
        U·a_raw                 → site D  (A sublattice partner)
        U·e                     → site B  (E sublattice partner)
        U·c                     → site F  (C sublattice partner)
    where U acts on the physical index: (U·x)[...,s'] = Σ_s U[s',s] x[...,s].

    Note: symmetrize_plaq_legs (2-leg projection) is NOT applied.  It was
    previously used but created a flat energy plateau at E/site=1/8 that
    stalled L-BFGS for hundreds of steps.
    """
    e = a_raw.permute(1, 2, 0, 3)   # C3  rotation
    c = a_raw.permute(2, 0, 1, 3)   # C3² rotation

    # Build the pi-Y rotation matrix: U[s, d-1-s] = (-1)^s
    d_PHYS = a_raw.shape[-1]
    U = torch.zeros(d_PHYS, d_PHYS, dtype=a_raw.dtype, device=a_raw.device)
    for s in range(d_PHYS):
        U[s, d_PHYS - 1 - s] = (-1.0) ** s
    # Apply U on the physical (last) leg:  b[...,s'] = Σ_s U[s',s] a[...,s]
    return (a_raw, torch.einsum('ij,...j->...i', U, e), c,
            torch.einsum('ij,...j->...i', U, a_raw), e,
            torch.einsum('ij,...j->...i', U, c))


# Backward-compatibility alias (deprecated)
plaq_abcdef_from_a = c6ypi_abcdef_from_a


def initialize_c6ypi(D_bond: int, d_PHYS: int,
                     noise_scale: float = 1.0) -> torch.Tensor:
    """Create a random tensor for the C6-Ypi single-tensor ansatz.

    Returns a_raw of shape (D_bond, D_bond, D_bond, d_PHYS).  Sets
    _USE_FULL_SVD = True so the first L-BFGS step uses full deterministic SVD.
    """
    global _USE_FULL_SVD
    a_raw = noise_scale * torch.randn(
        D_bond, D_bond, D_bond, d_PHYS,
        dtype=TENSORDTYPE, device=DEVICE)
    _USE_FULL_SVD = True
    return a_raw


# Backward-compatibility alias (deprecated)
initialize_plaq = initialize_c6ypi


# ── C3-Vypi single-tensor ansatz ──────────────────────────────────────────────
# C3: derives all 6 site tensors from a single tensor a by C3 rotation
#     combined with a pi-Y rotation (U = iσ_y on the physical leg) on the
#     B-sublattice sites (d, b, f).
# The virtual-leg permutation rule is intentionally left for the user to
# define in c3vypi_abcdef_from_a.

def c3vypi_abcdef_from_a(a_raw: torch.Tensor) -> tuple:
    """Derive (a, b, c, d, e, f) from a single tensor using C3 + pi-Y symmetry.

    The virtual-leg permutation differs from c6ypi.  Fill in the desired rule.

    Placeholder (identical to c6ypi for now — EDIT THIS):
        a_raw                    → site A
        a_raw.permute(1,2,0,3)  → e (C3 rotation)
        a_raw.permute(2,0,1,3)  → c (C3² rotation)
        U·a_raw                 → site D
        U·e                     → site B
        U·c                     → site F
    """
    e = a_raw.permute(1, 2, 0, 3)   # C3  rotation
    c = a_raw.permute(2, 0, 1, 3)   # C3² rotation
    b = a_raw.permute(2, 1, 0, 3)
    d = a_raw.permute(0, 2, 1, 3)
    f = a_raw.permute(1, 0, 2, 3)

    d_PHYS = a_raw.shape[-1]
    U = torch.zeros(d_PHYS, d_PHYS, dtype=a_raw.dtype, device=a_raw.device)
    for s in range(d_PHYS):
        U[s, d_PHYS - 1 - s] = (-1.0) ** s

    return (a_raw, torch.einsum('ij,...j->...i', U, b), c,
            torch.einsum('ij,...j->...i', U, d), e,
            torch.einsum('ij,...j->...i', U, f))


def initialize_c3vypi(D_bond: int, d_PHYS: int,
                      noise_scale: float = 1.0) -> torch.Tensor:
    """Create a random tensor for the C3-Vypi single-tensor ansatz.

    Returns a_raw of shape (D_bond, D_bond, D_bond, d_PHYS).
    """
    global _USE_FULL_SVD
    a_raw = noise_scale * torch.randn(
        D_bond, D_bond, D_bond, d_PHYS,
        dtype=TENSORDTYPE, device=DEVICE)
    _USE_FULL_SVD = True
    return a_raw


# ── Two-C3 two-tensor ansatz ─────────────────────────────────────────────────
# Derives all 6 site tensors from two independent tensors (a, b) by C3
# virtual-leg rotation.  No symmetrization, no physical-leg rotation.

def twoc3_abcdef_from_ab(a_raw: torch.Tensor,
                         b_raw: torch.Tensor) -> tuple:
    """Derive (a, b, c, d, e, f) from two tensors using C3 rotation.

    A-sublattice (a, c, e): C3 rotations of a_raw.
    B-sublattice (b, d, f): C3 rotations of b_raw.

        a = a_raw
        e = a_raw.permute(1,2,0,3)   # C3
        c = a_raw.permute(2,0,1,3)   # C3²
        b = b_raw
        f = b_raw.permute(1,2,0,3)   # C3
        d = b_raw.permute(2,0,1,3)   # C3²
    """
    e = a_raw.permute(1, 2, 0, 3)   # C3  rotation
    c = a_raw.permute(2, 0, 1, 3)   # C3² rotation
    f = b_raw.permute(1, 2, 0, 3)   # C3  rotation
    d = b_raw.permute(2, 0, 1, 3)   # C3² rotation
    return (a_raw, b_raw, c, d, e, f)


def initialize_twoc3(D_bond: int, d_PHYS: int,
                     noise_scale: float = 1.0) -> tuple:
    """Create two random tensors for the two-C3 ansatz.

    Returns (a_raw, b_raw), each of shape (D_bond, D_bond, D_bond, d_PHYS).
    """
    global _USE_FULL_SVD
    a_raw = noise_scale * torch.randn(
        D_bond, D_bond, D_bond, d_PHYS,
        dtype=TENSORDTYPE, device=DEVICE)
    b_raw = noise_scale * torch.randn(
        D_bond, D_bond, D_bond, d_PHYS,
        dtype=TENSORDTYPE, device=DEVICE)
    _USE_FULL_SVD = True
    return (a_raw, b_raw)


# ── 6-tensor locally-reflected ansatz ────────────────────────────────────────

def symmetrize_six_local_reflections(
        a: torch.Tensor, b: torch.Tensor, c: torch.Tensor,
        d: torch.Tensor, e: torch.Tensor, f: torch.Tensor,
) -> tuple:
    """Apply a distinct local mirror symmetrization to each of the 6 site tensors.

    Each tensor has shape (D, D, D, d_PHYS) with virtual legs (0, 1, 2) and
    physical leg 3.  The symmetrization maps each tensor onto the subspace
    that is invariant under the unique local reflection which swaps the two
    *ring bonds* (intra-plaquette legs) on that site while leaving the
    outgoing inter-plaquette leg fixed:

        a, d :  leg1 ↔ leg2  →  a_sym = (a + a.permute(0,2,1,3)) / 2
        b, e :  leg0 ↔ leg1  →  b_sym = (b + b.permute(1,0,2,3)) / 2
        c, f :  leg0 ↔ leg2  →  c_sym = (c + c.permute(2,1,0,3)) / 2

    Each projection is idempotent and linear, so the symmetrization does NOT
    create a flat energy landscape (unlike applying the same mirror to all six
    tensors simultaneously, which would over-constrain the variational space
    and stall L-BFGS).

    This function is called as the ``symmetrize_fn`` of the ``'sym6'`` ansatz
    registry entry.  It receives all six tensors at once (unlike the
    single-tensor symmetrizers used by ``'neel'`` and ``'plaq'``) and returns
    a tuple of six symmetrized tensors.

    Args:
        a, b, c, d, e, f:  Six (D, D, D, d_PHYS) site tensors.

    Returns:
        Tuple ``(a_sym, b_sym, c_sym, d_sym, e_sym, f_sym)``.
    """
    a_sym = (a + a.permute(0, 2, 1, 3)) / 2.0   # leg1 ↔ leg2
    b_sym = (b + b.permute(1, 0, 2, 3)) / 2.0   # leg0 ↔ leg1
    c_sym = (c + c.permute(2, 1, 0, 3)) / 2.0   # leg0 ↔ leg2
    d_sym = (d + d.permute(0, 2, 1, 3)) / 2.0   # leg1 ↔ leg2
    e_sym = (e + e.permute(1, 0, 2, 3)) / 2.0   # leg0 ↔ leg1
    f_sym = (f + f.permute(2, 1, 0, 3)) / 2.0   # leg0 ↔ leg2
    return (a_sym, b_sym, c_sym, d_sym, e_sym, f_sym)


def abcdef_to_ABCDEF(a,b,c,d,e,f, D_squared:int):
    """
    Build the double-layer ("bra⊗ket") site tensors A, B, C, D, E, F.

    Each double-layer tensor is obtained by contracting a single-layer tensor
    with its complex conjugate over the physical index::

        A_{(u,x),(v,y),(w,z)} = sum_φ  a_{u,v,w,φ} * conj(a_{x,y,z,φ})

    then reshaping the fused virtual indices ``(u,x)``, ``(v,y)``, ``(w,z)``
    into a ``(D_squared, D_squared, D_squared)`` tensor and finally into a
    ``(D_squared, D_squared)`` matrix (two legs fused into one on each side).
    Each output tensor is normalised to unit Frobenius norm.

    Args:
        a, b, c, d, e, f (torch.Tensor): Single-layer site tensors, each of
            shape ``(D_bond, D_bond, D_bond, d_PHYS)``.
        D_squared (int): ``D_bond ** 2``, the bond dimension of the
            double-layer tensors.

    Returns:
        Tuple[torch.Tensor, ...]: ``(A, B, C, D, E, F)``, each of shape
        ``(D_squared, D_squared, D_squared)`` (rank-3, one fused virtual-index
        pair per honeycomb leg), dtype ``torch.float32``.
    """

    A = oe.contract("uvwp,xyzp->uxvywz", a, a.conj(), optimize=[(0, 1)], backend='torch')
    A = A.reshape(D_squared, D_squared, D_squared)
    # A = normalize_tensor(A)

    B = oe.contract("uvwp,xyzp->uxvywz", b, b.conj(), optimize=[(0, 1)], backend='torch')
    B = B.reshape(D_squared, D_squared, D_squared)
    # B = normalize_tensor(B)

    C = oe.contract("uvwp,xyzp->uxvywz", c, c.conj(), optimize=[(0, 1)], backend='torch')
    C = C.reshape(D_squared, D_squared, D_squared)
    # C = normalize_tensor(C)

    D = oe.contract("uvwp,xyzp->uxvywz", d, d.conj(), optimize=[(0, 1)], backend='torch')
    D = D.reshape(D_squared, D_squared, D_squared)
    # D = normalize_tensor(D)

    E = oe.contract("uvwp,xyzp->uxvywz", e, e.conj(), optimize=[(0, 1)], backend='torch')
    E = E.reshape(D_squared, D_squared, D_squared)
    # E = normalize_tensor(E)

    F = oe.contract("uvwp,xyzp->uxvywz", f, f.conj(), optimize=[(0, 1)], backend='torch')
    F = F.reshape(D_squared, D_squared, D_squared)
    # F = normalize_tensor(F)

    return A,B,C,D,E,F









def trunc_rhoCCC(matC21, matC32, matC13, chi, D_squared):

    #Q1, R1 = torch.linalg.qr(matC21.T)
    #Q2, R2 = torch.linalg.qr(torch.mm(matC13,matC32))
    R1 = matC21.T
    R2 = torch.mm(matC13,matC32)
    
    #U, S, Vh = torch.linalg.svd(torch.mm(R1,R2.T))
    #truncUh = U[:,:chi].conj().T
    #truncV = Vh[:chi,:].conj().T



    if _RECORD_TRUNC_ERROR:
        _U, _S_all, _Vh = torch.linalg.svd(torch.mm(R1, R2.T), full_matrices=False)
        _record_trunc_err(_S_all, chi)
        U = _U[:, :chi]
        S = _S_all[:chi]
        V = _Vh[:chi, :].conj().T
        del _U, _S_all, _Vh
    else:
        U, S, V = truncated_svd_propack(torch.mm(R1,R2.T), chi,
                        chi_extra=round(2*np.sqrt(D_squared)),
                        rel_cutoff=1e-12,
                        v0=None,
                        keep_multiplets=False,
                        abs_tol=1e-14,
                        eps_multiplet=1e-12)
    #product_error = torch.linalg.norm(approx_product - exact_product) / torch.linalg.norm(exact_product)
    #VERIFIED print(f"R1 @ R2.T vs truncated U @ diag(S) @ Vh check: relative error = {product_error:.2e}")

    sqrtInvTruncS = _safe_sqrt_inv_diag(S[:chi])
    #VERIFIED print((S[chi]/S[0]).item())
    #VERIFIED print(torch.diag(sqrtInvTruncS @ sqrtInvTruncS @ torch.diag(S[:chi]).to(TENSORDTYPE)).detach().cpu().numpy())

    #product_inverse = Vh.conj().T @ torch.diag(1.0 / torch.sqrt(S).to(TENSORDTYPE)) @ torch.diag(1.0 / torch.sqrt(S).to(TENSORDTYPE)) @ U.conj().T
    #exact_inverse = torch.linalg.inv(R1 @ R2.T)
    #inverse_error = torch.linalg.norm(product_inverse - exact_inverse) / torch.linalg.norm(exact_inverse)
    #VERIFIED print(f"(R1 @ R2.T)^-1 vs Vh @ diag(1/sqrt(S)) @ diag(1/sqrt(S)) @ U check: relative error = {inverse_error:.2e}")

    #U_inv = torch.linalg.inv(U)
    #U_dag = U.conj().T
    #Vh_inv = torch.linalg.inv(Vh)
    #Vh_dag = Vh.conj().T
    #print_error_U = torch.linalg.norm(U_inv - U_dag) / torch.linalg.norm(U_dag)
    #print_error_Vh = torch.linalg.norm(Vh_inv - Vh_dag) / torch.linalg.norm(Vh_dag)
    #VERIFIED print(f"U-1 vs U† check: relative error = {print_error_U:.4e}")
    #VERIFIED print(f"V†-1 vs V check: relative error = {print_error_Vh:.4e}")

    # print(torch.norm(approx_product @ truncV @ sqrtInvTruncS @ sqrtInvTruncS @ truncUh * 2 - torch.eye(chi*D_squared, device=approx_product.device, dtype=approx_product.dtype)).item())
    # print((approx_product @ truncV @ sqrtInvTruncS @ sqrtInvTruncS @ truncUh).detach().cpu().numpy())

    #approx_produc_inverse = truncV @ sqrtInvTruncS @ sqrtInvTruncS @ truncUh 
    #produc_inverse = torch.linalg.inv(R1 @ R2.T)

########################################
    # TODO: BECAUSE THE C is normalized s.t. the determinant can be very small!
########################################

    #print("detR1=", torch.linalg.det(R1).item(), "   detR2=", torch.linalg.det(R2).item())
    #inverse_error = torch.linalg.norm(approx_produc_inverse - produc_inverse) #/ torch.linalg.norm(produc_inverse)
    #print(f"approx_produc_inverse vs exact inverse check: relative error = {inverse_error:.4e}")
################################
    # NOTE: But still, the error for R2 V 1/S Uh R1 is much larger than V 1/S Uh R1 R2
    # NOTE: QR (SVD-1)R1R2 >~= no-QR (SVD-1)R1R2 > QR R2(SVD-1)R1 ~= no-QR R2(SVD-1)R1
###############################


    P21My = torch.mm( R2.T ,torch.mm( V , sqrtInvTruncS ))
    P32yM = torch.mm(torch.mm( sqrtInvTruncS , U.conj().T ), R1 )

    #almost_identity = torch.mm(P21My,P32yM)
    #average_diag = torch.diag(almost_identity).mean().item()
    #standard_error_diag = torch.sqrt(torch.var(torch.diag(almost_identity))).item()
    #print(f"P21My @ P32yM ≈ I check: average diag = {average_diag:.4e}, standard error diag = {standard_error_diag:.4e}")
    
    #corresponding_identity = torch.eye(chi*D_squared, device=almost_identity.device, dtype=almost_identity.dtype)
    #projected_difference =  truncUh @ (almost_identity - corresponding_identity) @ truncV
    #identity_error = torch.linalg.norm(projected_difference)/torch.linalg.norm(corresponding_identity[:chi,:chi])
    # print(f"P21My @ P32yM ≈ I check: relative error = {identity_error:.4e}")

    #almost_identity_2 = truncV @ sqrtInvTruncS @ sqrtInvTruncS @ truncUh @ R1 @ R2.T
    #only_display_1_digit_almost_identity_2 = torch.round(torch.real(almost_identity_2) * 10) / 10 + 1j * torch.round(torch.imag(almost_identity_2) * 10) / 10
    #print(only_display_1_digit_almost_identity_2.detach().cpu().numpy())



    #Q1, R1 = torch.linalg.qr(matC32.T)
    #Q2, R2 = torch.linalg.qr(torch.mm(matC21,matC13))
    R1 = matC32.T
    R2 = torch.mm(matC21,matC13)
    if _RECORD_TRUNC_ERROR:
        _U, _S_all, _Vh = torch.linalg.svd(torch.mm(R1, R2.T), full_matrices=False)
        _record_trunc_err(_S_all, chi)
        U = _U[:, :chi]
        S = _S_all[:chi]
        V = _Vh[:chi, :].conj().T
        del _U, _S_all, _Vh
    else:
        U, S, V = truncated_svd_propack(torch.mm(R1,R2.T), chi,
                        chi_extra=round(2*np.sqrt(D_squared)),
                        rel_cutoff=1e-12,
                        v0=None,
                        keep_multiplets=False,
                        abs_tol=1e-14,
                        eps_multiplet=1e-12)
    sqrtInvTruncS = _safe_sqrt_inv_diag(S[:chi])
    P32Nz = torch.mm( R2.T ,torch.mm( V , sqrtInvTruncS ))
    P13zN = torch.mm(torch.mm( sqrtInvTruncS , U.conj().T ), R1 )




    #Q1, R1 = torch.linalg.qr(matC13.T)
    #Q2, R2 = torch.linalg.qr(torch.mm(matC32,matC21))
    R1 = matC13.T
    R2 = torch.mm(matC32,matC21)
    #U, S, Vh = torch.linalg.svd(torch.mm(R1,R2.T))
    if _RECORD_TRUNC_ERROR:
        _U, _S_all, _Vh = torch.linalg.svd(torch.mm(R1, R2.T), full_matrices=False)
        _record_trunc_err(_S_all, chi)
        U = _U[:, :chi]
        S = _S_all[:chi]
        V = _Vh[:chi, :].conj().T
        del _U, _S_all, _Vh
    else:
        U, S, V = truncated_svd_propack(torch.mm(R1,R2.T), chi,
                        chi_extra=round(2*np.sqrt(D_squared)),
                        rel_cutoff=1e-12,
                        v0=None,
                        keep_multiplets=False,
                        abs_tol=1e-14,
                        eps_multiplet=1e-12)




    #truncUh = U[:,:chi].conj().T
    #truncV = Vh[:chi,:].conj().T
    sqrtInvTruncS = _safe_sqrt_inv_diag(S[:chi])

    P13Lx = torch.mm( R2.T ,torch.mm( V, sqrtInvTruncS ))
    P21xL = torch.mm(torch.mm( sqrtInvTruncS , U.conj().T ), R1 )



    matC21 = torch.mm(P21My.T, torch.mm(matC21, P21xL.T))
    matC32 = torch.mm(P32Nz.T, torch.mm(matC32, P32yM.T))
    matC13 = torch.mm(P13Lx.T, torch.mm(matC13, P13zN.T))
    
    
    matC21 = matC21/torch.linalg.norm(matC21)
    matC32 = matC32/torch.linalg.norm(matC32)
    matC13 = matC13/torch.linalg.norm(matC13)
    

    """
    tr_abs = tr_rho.abs()

    if torch.isfinite(tr_abs).item() and (tr_abs > torch.finfo(tr_abs.dtype).tiny).item():
        phase = tr_rho / tr_abs
        C21 = C21 / phase

        scaleC = tr_abs.pow(1.0 / 3.0)
        if torch.isfinite(scaleC).item() and (scaleC > torch.finfo(scaleC.dtype).tiny).item():
            C21 = C21 / scaleC
            C32 = C32 / scaleC
            C13 = C13 / scaleC

    # ── Individual equalization: preserve |Tr(rho)|=1 while balancing magnitudes ──
    # Scale factors multiply to 1  ⟹  |Tr(rho)| = |Tr(C13·C32·C21)| unchanged.
    n21 = torch.linalg.norm(C21).real
    n32 = torch.linalg.norm(C32).real
    n13 = torch.linalg.norm(C13).real
    nC_prod = n21 * n32 * n13
    if torch.isfinite(nC_prod).item() and (nC_prod > torch.finfo(n21.dtype).tiny).item():
        nC_geo = nC_prod.pow(1.0 / 3.0)
        C21 = C21 * (nC_geo / n21)
        C32 = C32 * (nC_geo / n32)
        C13 = C13 * (nC_geo / n13)
    """


    return P21xL.reshape(chi, chi,D_squared), matC21, P21My.reshape(chi,D_squared, chi), P32yM.reshape(chi, chi,D_squared), matC32, P32Nz.reshape(chi,D_squared, chi), P13zN.reshape(chi, chi,D_squared), matC13, P13Lx.reshape(chi,D_squared, chi)


def initialize_envCTs_1(A,B,C,D,E,F, chi, D_squared, identity_init=False):
    """
    Initialise the 9 CTMRG corner matrices for the first CTMRG cycle.

    The first cycle is a special case because we have no previous environment to
    truncate from.  We therefore build each corner by contracting together the
    three double-layer site tensors that meet at that corner, then optionally
    add a small identity component to ensure full rank before truncation.

    Args:
        A, B, C, D, E, F (torch.Tensor): Double-layer site tensors of shape
            ``(D_squared, D_squared, D_squared)``.
        chi (int): Target bond dimension for the initial corners (before truncation).
        D_squared (int): ``D_bond ** 2``.
        identity_init (bool): If True, add a small identity component to each
            corner to ensure full rank before truncation.  This can help avoid
            numerical issues in the first truncation step when the raw corners are
            very low-rank.

    Returns:
        Tuple[torch.Tensor, ...]: The 9 initial corner matrices, each of shape
        ``(chi, chi)`` for C21CD, C32EF, C13AB, and shape ``(chi, chi, D_squared)``
        for the transfer tensors T1F, T2A, T2B, T3C, T3D, T1E.
    """
    D_bond = round(D_squared ** 0.5)

    if identity_init:
        # Add a small identity component to each corner to ensure full rank.
        # The scale is arbitrary but should be small enough not to dominate the
        # physical corner contribution; 1e-3 is a reasonable starting point.
        id_noise_scale = 1e-2

        if False: 

            identityC = torch.eye(chi, dtype=A.dtype, device=A.device)
            C21CD = identityC + id_noise_scale * torch.randn_like(identityC)
            C32EF = identityC + id_noise_scale * torch.randn_like(identityC)
            C13AB = identityC + id_noise_scale * torch.randn_like(identityC)

            identityT = torch.eye(chi*D_bond, dtype=A.dtype, device=A.device)
            identityT = identityT.reshape(chi, D_bond, chi, D_bond).permute(0, 2, 1, 3).reshape(chi, chi, D_squared)
            T1F = identityT + id_noise_scale * torch.randn_like(identityT)
            T2A = identityT + id_noise_scale * torch.randn_like(identityT)
            T2B = identityT + id_noise_scale * torch.randn_like(identityT)
            T3C = identityT + id_noise_scale * torch.randn_like(identityT)
            T3D = identityT + id_noise_scale * torch.randn_like(identityT)
            T1E = identityT + id_noise_scale * torch.randn_like(identityT)

        else:

            allOnesC = torch.ones((chi, chi), dtype=A.dtype, device=A.device)
            C21CD = allOnesC + id_noise_scale * torch.randn_like(allOnesC)
            C32EF = allOnesC + id_noise_scale * torch.randn_like(allOnesC)
            C13AB = allOnesC + id_noise_scale * torch.randn_like(allOnesC)

            allOnesT = torch.ones((chi, chi, D_squared), dtype=A.dtype, device=A.device)
            T1F = allOnesT + id_noise_scale * torch.randn_like(allOnesT)
            T2A = allOnesT + id_noise_scale * torch.randn_like(allOnesT)
            T2B = allOnesT + id_noise_scale * torch.randn_like(allOnesT)
            T3C = allOnesT + id_noise_scale * torch.randn_like(allOnesT)
            T3D = allOnesT + id_noise_scale * torch.randn_like(allOnesT)
            T1E = allOnesT + id_noise_scale * torch.randn_like(allOnesT)


        C21CD = normalize_tensor(C21CD)
        C32EF = normalize_tensor(C32EF)
        C13AB = normalize_tensor(C13AB)
        T1F = normalize_tensor(T1F)
        T2A = normalize_tensor(T2A)
        T2B = normalize_tensor(T2B)
        T3C = normalize_tensor(T3C)
        T3D = normalize_tensor(T3D)
        T1E = normalize_tensor(T1E)



    else: 
        # brutal contract env from 8*3 + 5*4*3 + 6 = 90 local tensors

        # enlarged C*3
        BC2323 = oe.contract("iBG,ibg->BGbg",B,C, optimize=[(0,1)], backend='torch')
        DE3131 = oe.contract("AiG,aig->GAga",D,E, optimize=[(0,1)], backend='torch')
        FA1212 = oe.contract("ABi,abi->ABab",F,A, optimize=[(0,1)], backend='torch')

        A23 = A.reshape(D_bond,D_bond,D_squared,D_squared).diagonal(dim1=0, dim2=1).sum(-1)
        F31 = F.permute(1,2,0).reshape(D_bond,D_bond,D_squared,D_squared).diagonal(dim1=0, dim2=1).sum(-1)
        C31 = C.permute(1,2,0).reshape(D_bond,D_bond,D_squared,D_squared).diagonal(dim1=0, dim2=1).sum(-1)
        B12 = B.permute(2,0,1).reshape(D_bond,D_bond,D_squared,D_squared).diagonal(dim1=0, dim2=1).sum(-1)
        E12 = E.permute(2,0,1).reshape(D_bond,D_bond,D_squared,D_squared).diagonal(dim1=0, dim2=1).sum(-1)
        D23 = D.reshape(D_bond,D_bond,D_squared,D_squared).diagonal(dim1=0, dim2=1).sum(-1)

        E23 = E.reshape(D_bond,D_bond,D_squared,D_squared).diagonal(dim1=0, dim2=1).sum(-1)
        B31 = B.permute(1,2,0).reshape(D_bond,D_bond,D_squared,D_squared).diagonal(dim1=0, dim2=1).sum(-1)
        A31 = A.permute(1,2,0).reshape(D_bond,D_bond,D_squared,D_squared).diagonal(dim1=0, dim2=1).sum(-1)
        D12 = D.permute(2,0,1).reshape(D_bond,D_bond,D_squared,D_squared).diagonal(dim1=0, dim2=1).sum(-1)
        C12 = C.permute(2,0,1).reshape(D_bond,D_bond,D_squared,D_squared).diagonal(dim1=0, dim2=1).sum(-1)
        F23 = F.reshape(D_bond,D_bond,D_squared,D_squared).diagonal(dim1=0, dim2=1).sum(-1)
        
        C21CD = oe.contract("yj,ijYk,in,nm,kXlm,lx->YyXx",
                            E23,BC2323,A23,F31,DE3131,B31,
                            optimize=[(0,1),(0,1),(0,1),(0,1),(0,1)],
                            backend='torch').reshape(D_squared*D_squared,D_squared*D_squared)
        C32EF = oe.contract("zj,ijZk,in,nm,kYlm,ly->ZzYy",
                            A31,DE3131,C31,B12,FA1212,D12,
                            optimize=[(0,1),(0,1),(0,1),(0,1),(0,1)],
                            backend='torch').reshape(D_squared*D_squared,D_squared*D_squared)
        C13AB = oe.contract("xj,ijXk,in,nm,kZlm,lz->XxZz",
                            C12,FA1212,E12,D23,BC2323,F23,
                            optimize=[(0,1),(0,1),(0,1),(0,1),(0,1)],
                            backend='torch').reshape(D_squared*D_squared,D_squared*D_squared)
        

        # enlarged T*6
        A12 = A.permute(2,0,1).reshape(D_bond,D_bond,D_squared,D_squared).diagonal(dim1=0, dim2=1).sum(-1)
        B23 = B.reshape(D_bond,D_bond,D_squared,D_squared).diagonal(dim1=0, dim2=1).sum(-1)
        C23 = C.reshape(D_bond,D_bond,D_squared,D_squared).diagonal(dim1=0, dim2=1).sum(-1)
        D31 = D.permute(1,2,0).reshape(D_bond,D_bond,D_squared,D_squared).diagonal(dim1=0, dim2=1).sum(-1)
        E31 = E.permute(1,2,0).reshape(D_bond,D_bond,D_squared,D_squared).diagonal(dim1=0, dim2=1).sum(-1)
        F12 = F.permute(2,0,1).reshape(D_bond,D_bond,D_squared,D_squared).diagonal(dim1=0, dim2=1).sum(-1)

        T1F = oe.contract("mi,jyi,aYjM->MmYya",
                          C23,D,FA1212,
                          optimize=[(0,1),(0,1)],
                          backend='torch').reshape(D_squared*D_squared,D_squared*D_squared,D_squared)
        T2A = oe.contract("il,xji,LjXb->LlXxb",
                          D31,C,FA1212,
                          optimize=[(0,1),(0,1)],
                          backend='torch').reshape(D_squared*D_squared,D_squared*D_squared,D_squared)   
        T2B = oe.contract("ni,ijz,bZjN->NnZzb",
                          E31,F,BC2323,
                          optimize=[(0,1),(0,1)],
                          backend='torch').reshape(D_squared*D_squared,D_squared*D_squared,D_squared)   
        T3C = oe.contract("im,iyj,MjYg->MmYyg",
                          F12,E,BC2323,
                          optimize=[(0,1),(0,1)],
                          backend='torch').reshape(D_squared*D_squared,D_squared*D_squared,D_squared)   
        T3D = oe.contract("li,xij,gXjL->LlXxg",
                          A12,B,DE3131,
                          optimize=[(0,1),(0,1)],
                          backend='torch').reshape(D_squared*D_squared,D_squared*D_squared,D_squared)   
        T1E = oe.contract("in,jiz,NjZa->NnZza",
                          B23,A,DE3131,
                          optimize=[(0,1),(0,1)],
                          backend='torch').reshape(D_squared*D_squared,D_squared*D_squared,D_squared)   

        R1 = C21CD.T
        R2 = torch.mm(C13AB,C32EF)
        
        #Q1, R1 = torch.linalg.qr(matC21.T)
        #Q2, R2 = torch.linalg.qr(torch.mm(matC13,matC32))
        U, S, V = truncated_svd_propack(torch.mm(R1,R2.T), chi,
                        chi_extra=round(2*np.sqrt(D_squared)),
                        rel_cutoff=1e-12,
                        v0=None,
                        keep_multiplets=False,
                        abs_tol=1e-14,
                        eps_multiplet=1e-12)
        

        sqrtInvTruncS = _safe_sqrt_inv_diag(S[:chi])

        P21My = torch.mm( R2.T ,torch.mm( V , sqrtInvTruncS ))
        P32yM = torch.mm(torch.mm( sqrtInvTruncS , U.conj().T ), R1 )


        R1 = C32EF.T
        R2 = torch.mm(C21CD,C13AB)

        #Q1, R1 = torch.linalg.qr(matC32.T)
        #Q2, R2 = torch.linalg.qr(torch.mm(matC21,matC13))
        U, S, V = truncated_svd_propack(torch.mm(R1,R2.T), chi,
                        chi_extra=round(2*np.sqrt(D_squared)),
                        rel_cutoff=1e-12,
                        v0=None,
                        keep_multiplets=False,
                        abs_tol=1e-14,
                        eps_multiplet=1e-12)

        sqrtInvTruncS = _safe_sqrt_inv_diag(S[:chi])

        P32Nz = torch.mm( R2.T ,torch.mm( V , sqrtInvTruncS ))
        P13zN = torch.mm(torch.mm( sqrtInvTruncS , U.conj().T ), R1 )


        R1 = C13AB.T
        R2 = torch.mm(C32EF,C21CD)

        #Q1, R1 = torch.linalg.qr(matC13.T)
        #Q2, R2 = torch.linalg.qr(torch.mm(matC32,matC21))
        U, S, V = truncated_svd_propack(torch.mm(R1,R2.T), chi,
                        chi_extra=round(2*np.sqrt(D_squared)),
                        rel_cutoff=1e-12,
                        v0=None,
                        keep_multiplets=False,
                        abs_tol=1e-14,
                        eps_multiplet=1e-12)

        sqrtInvTruncS = _safe_sqrt_inv_diag(S[:chi])

        P13Lx = torch.mm( R2.T ,torch.mm( V , sqrtInvTruncS ))
        P21xL = torch.mm(torch.mm( sqrtInvTruncS , U.conj().T ), R1 )



        C21CD = torch.mm(P21My.T, torch.mm(C21CD, P21xL.T))
        C32EF = torch.mm(P32Nz.T, torch.mm(C32EF, P32yM.T))
        C13AB = torch.mm(P13Lx.T, torch.mm(C13AB, P13zN.T))
        
        C21CD = C21CD/torch.linalg.norm(C21CD)
        C32EF = C32EF/torch.linalg.norm(C32EF)
        C13AB = C13AB/torch.linalg.norm(C13AB)
   
        """
        tr_abs = tr_rho.abs()
        if torch.isfinite(tr_abs).item() and (tr_abs > torch.finfo(tr_abs.dtype).tiny).item():
            phase = tr_rho / tr_abs
            C21 = C21 / phase

            scaleC = tr_abs.pow(1.0 / 3.0)
            if torch.isfinite(scaleC).item() and (scaleC > torch.finfo(scaleC.dtype).tiny).item():
                C21 = C21 / scaleC
                C32 = C32 / scaleC
                C13 = C13 / scaleC

        # ── Individual equalization: preserve Tr(rho)=1 while balancing magnitudes ──
        # Scale factors multiply to 1  ⟹  Tr(rho) = Tr(C13·C32·C21) unchanged.
        n21 = torch.linalg.norm(C21).real
        n32 = torch.linalg.norm(C32).real
        n13 = torch.linalg.norm(C13).real
        nC_prod = n21 * n32 * n13
        if torch.isfinite(nC_prod).item() and (nC_prod > torch.finfo(n21.dtype).tiny).item():
            nC_geo = nC_prod.pow(1.0 / 3.0)
            C21CD = C21 * (nC_geo / n21)
            C32EF = C32 * (nC_geo / n32)
            C13AB = C13 * (nC_geo / n13)
        """



        # Uh,V project on one side(XYZ) of Ts



        T1F = oe.contract("MYa,yY->Mya",T1F,P32yM.reshape(chi, D_squared*D_squared),optimize=[(0,1)],backend='torch').reshape(D_squared*D_squared,chi*D_squared)
        T2A = oe.contract("LXb,Xx->Lxb",T2A,P13Lx.reshape(D_squared*D_squared, chi),optimize=[(0,1)],backend='torch').reshape(D_squared*D_squared,chi*D_squared)
        T2B = oe.contract("NZb,zZ->Nzb",T2B,P13zN.reshape(chi, D_squared*D_squared),optimize=[(0,1)],backend='torch').reshape(D_squared*D_squared,chi*D_squared)
        T3C = oe.contract("MYg,Yy->Myg",T3C,P21My.reshape(D_squared*D_squared, chi),optimize=[(0,1)],backend='torch').reshape(D_squared*D_squared,chi*D_squared)
        T3D = oe.contract("LXg,xX->Lxg",T3D,P21xL.reshape(chi, D_squared*D_squared),optimize=[(0,1)],backend='torch').reshape(D_squared*D_squared,chi*D_squared)
        T1E = oe.contract("NZa,Zz->Nza",T1E,P32Nz.reshape(D_squared*D_squared, chi),optimize=[(0,1)],backend='torch').reshape(D_squared*D_squared,chi*D_squared)  

        U, S, V = truncated_svd_propack(torch.mm(T1F.T,T3C), chi,
                    chi_extra=round(2*np.sqrt(D_squared)),
                    rel_cutoff=1e-12,
                    v0=None,
                    keep_multiplets=False,
                    abs_tol=1e-14,
                    eps_multiplet=1e-12)
        
        

        sqrtInvTruncS = _safe_sqrt_inv_diag(S[:chi])

        old = T1F
        T1F = torch.mm((torch.mm(T3C, torch.mm(V, sqrtInvTruncS))).T, T1F)
        T3C = torch.mm(torch.mm( torch.mm(sqrtInvTruncS, U.conj().T), old.T), T3C)


        U, S, V = truncated_svd_propack(torch.mm(T3D.T,T2A), chi,
                    chi_extra=round(2*np.sqrt(D_squared)),
                    rel_cutoff=1e-12,
                    v0=None,
                    keep_multiplets=False,
                    abs_tol=1e-14,
                    eps_multiplet=1e-12)
        
        sqrtInvTruncS = _safe_sqrt_inv_diag(S[:chi])
        
        old = T3D
        T3D = torch.mm((torch.mm(T2A, torch.mm(V, sqrtInvTruncS))).T, T3D)
        T2A = torch.mm(torch.mm( torch.mm(sqrtInvTruncS, U.conj().T), old.T), T2A)


        U, S, V = truncated_svd_propack(torch.mm(T2B.T,T1E), chi,
                    chi_extra=round(2*np.sqrt(D_squared)),
                    rel_cutoff=1e-12,
                    v0=None,
                    keep_multiplets=False,
                    abs_tol=1e-14,
                    eps_multiplet=1e-12)
        
        sqrtInvTruncS = _safe_sqrt_inv_diag(S[:chi])

        old = T2B
        T2B = torch.mm((torch.mm(T1E, torch.mm(V, sqrtInvTruncS))).T, T2B)
        T1E = torch.mm(torch.mm( torch.mm(sqrtInvTruncS, U.conj().T), old.T), T1E)
            





        T1F = T1F.reshape(chi, chi, D_squared)
        T2A = T2A.reshape(chi, chi, D_squared)
        T2B = T2B.reshape(chi, chi, D_squared)
        T3C = T3C.reshape(chi, chi, D_squared)
        T3D = T3D.reshape(chi, chi, D_squared)
        T1E = T1E.reshape(chi, chi, D_squared)

        T1F = T1F / torch.linalg.norm(T1F)
        T2A = T2A / torch.linalg.norm(T2A)
        T2B = T2B / torch.linalg.norm(T2B)
        T3C = T3C / torch.linalg.norm(T3C)
        T3D = T3D / torch.linalg.norm(T3D)
        T1E = T1E / torch.linalg.norm(T1E)



        """
        norm_abs = norm_val.abs()
        if torch.isfinite(norm_abs).item() and (norm_abs > torch.finfo(norm_abs.dtype).tiny).item():
            phase = norm_val / norm_abs
            T3C = T3C / phase
            scaleT = norm_abs.pow(1.0 / 6.0)
            if torch.isfinite(scaleT).item() and (scaleT > torch.finfo(scaleT.dtype).tiny).item():
                T3C = T3C / scaleT
                T3D = T3D / scaleT
                T1E = T1E / scaleT
                T1F = T1F / scaleT
                T2A = T2A / scaleT
                T2B = T2B / scaleT

        nT3C = torch.linalg.norm(T3C).real
        nT3D = torch.linalg.norm(T3D).real
        nT1E = torch.linalg.norm(T1E).real
        nT1F = torch.linalg.norm(T1F).real
        nT2A = torch.linalg.norm(T2A).real
        nT2B = torch.linalg.norm(T2B).real
        nT_prod = nT3C * nT3D * nT1E * nT1F * nT2A * nT2B
        if torch.isfinite(nT_prod).item() and (nT_prod > torch.finfo(nT3C.dtype).tiny).item():
            nT_geo = nT_prod.pow(1.0 / 6.0)
            T3C = T3C * (nT_geo / nT3C)
            T3D = T3D * (nT_geo / nT3D)
            T1E = T1E * (nT_geo / nT1E)
            T1F = T1F * (nT_geo / nT1F)
            T2A = T2A * (nT_geo / nT2A)
            T2B = T2B * (nT_geo / nT2B)
    """
        

    return C21CD, C32EF, C13AB, T1F, T2A, T2B, T3C, T3D, T1E








def check_env_CV_using_3rho(lastC21CD, lastC32EF, lastC13AB, 
                            nowC21CD, nowC32EF, nowC13AB, 
                            lastC21EB, lastC32AD, lastC13CF, 
                            nowC21EB, nowC32AD, nowC13CF, 
                            lastC21AF, lastC32CB, lastC13ED, 
                            nowC21AF, nowC32CB, nowC13ED, 
                            env_conv_threshold):
   
    # Warm-up guard: all three env types must have been computed at least once.

    # Keep max_delta as a GPU scalar tensor; extract to Python only once at return.
    max_delta = lastC21CD.new_tensor(0.0).real
    # The rhos defined by 1-1 cut: 13 @ 32 @ 21
    last_rho1 = oe.contract("UZ,ZY,YV->UV",
                                lastC13AB,lastC32EF,lastC21CD,
                                optimize=[(0,1),(0,1)],
                                backend='torch')
    now_rho1 = oe.contract("UZ,ZY,YV->UV",
                                nowC13AB,nowC32EF,nowC21CD,
                                optimize=[(0,1),(0,1)],
                                backend='torch')
    last_rho2 = oe.contract("UZ,ZY,YV->UV",
                                lastC13CF,lastC32AD,lastC21EB,
                                optimize=[(0,1),(0,1)],
                                backend='torch')
    now_rho2 = oe.contract("UZ,ZY,YV->UV",
                                nowC13CF,nowC32AD,nowC21EB,
                                optimize=[(0,1),(0,1)],
                                backend='torch')
    last_rho3 = oe.contract("UZ,ZY,YV->UV",
                                lastC13ED,lastC32CB,lastC21AF,
                                optimize=[(0,1),(0,1)],
                                backend='torch')
    now_rho3 = oe.contract("UZ,ZY,YV->UV",
                                nowC13ED,nowC32CB,nowC21AF, 
                                optimize=[(0,1),(0,1)],
                                backend='torch')

    rho_pairs = [
        (last_rho1, now_rho1),
        (last_rho2, now_rho2),
        (last_rho3, now_rho3),
    ]

    for last_c, now_c in rho_pairs:
        sv_last = torch.linalg.svdvals(last_c).real
        sv_now  = torch.linalg.svdvals(now_c ).real
        # Normalise so that the largest singular value is 1 (scale-invariant).
        sv_last = sv_last / (sv_last[0] + 1e-30)
        sv_now  = sv_now  / (sv_now [0] + 1e-30)
        # Weight differences by SV magnitude: large SVs must converge tightly,
        # small SVs are allowed proportionally larger absolute fluctuations.
        sv_weight = (sv_now + sv_last) / 2
        delta = ((sv_now - sv_last).abs() * sv_weight).max()
        max_delta = torch.maximum(max_delta, delta)

    last_rho1 = oe.contract("UY,YX,XV->UV",
                                lastC32EF,lastC21CD,lastC13AB,
                                optimize=[(0,1),(0,1)],
                                backend='torch')
    now_rho1 = oe.contract("UY,YX,XV->UV",
                                nowC32EF,nowC21CD,nowC13AB,
                                optimize=[(0,1),(0,1)],
                                backend='torch')
    last_rho2 = oe.contract("UY,YX,XV->UV",
                                lastC32AD,lastC21EB,lastC13CF,
                                optimize=[(0,1),(0,1)],
                                backend='torch')
    now_rho2 = oe.contract("UY,YX,XV->UV",
                                nowC32AD,nowC21EB,nowC13CF,
                                optimize=[(0,1),(0,1)],
                                backend='torch')
    last_rho3 = oe.contract("UY,YX,XV->UV",
                                lastC32CB,lastC21AF,lastC13ED,
                                optimize=[(0,1),(0,1)],
                                backend='torch')
    now_rho3 = oe.contract("UY,YX,XV->UV",
                                nowC32CB,nowC21AF,nowC13ED, 
                                optimize=[(0,1),(0,1)],
                                backend='torch')

    rho_pairs = [
        (last_rho1, now_rho1),
        (last_rho2, now_rho2),
        (last_rho3, now_rho3),
    ]

    for last_c, now_c in rho_pairs:
        sv_last = torch.linalg.svdvals(last_c).real
        sv_now  = torch.linalg.svdvals(now_c ).real
        # Normalise so that the largest singular value is 1 (scale-invariant).
        sv_last = sv_last / (sv_last[0] + 1e-30)
        sv_now  = sv_now  / (sv_now [0] + 1e-30)
        # Weight differences by SV magnitude: large SVs must converge tightly,
        # small SVs are allowed proportionally larger absolute fluctuations.
        sv_weight = (sv_now + sv_last) / 2
        delta = ((sv_now - sv_last).abs() * sv_weight).max()
        max_delta = torch.maximum(max_delta, delta)

    last_rho1 = oe.contract("UX,XZ,ZV->UV",
                                lastC21CD,lastC13AB,lastC32EF,
                                optimize=[(0,1),(0,1)],
                                backend='torch')
    now_rho1 = oe.contract("UX,XZ,ZV->UV",
                                nowC21CD,nowC13AB,nowC32EF,
                                optimize=[(0,1),(0,1)],
                                backend='torch')
    last_rho2 = oe.contract("UX,XZ,ZV->UV",
                                lastC21EB,lastC13CF,lastC32AD,
                                optimize=[(0,1),(0,1)],
                                backend='torch')
    now_rho2 = oe.contract("UX,XZ,ZV->UV",
                                nowC21EB,nowC13CF,nowC32AD,
                                optimize=[(0,1),(0,1)],
                                backend='torch')
    last_rho3 = oe.contract("UX,XZ,ZV->UV",
                                lastC21AF,lastC13ED,lastC32CB,
                                optimize=[(0,1),(0,1)],
                                backend='torch')
    now_rho3 = oe.contract("UX,XZ,ZV->UV",
                                nowC21AF,nowC13ED,nowC32CB, 
                                optimize=[(0,1),(0,1)],
                                backend='torch')

    rho_pairs = [
        (last_rho1, now_rho1),
        (last_rho2, now_rho2),
        (last_rho3, now_rho3),
    ]

    for last_c, now_c in rho_pairs:
        sv_last = torch.linalg.svdvals(last_c).real
        sv_now  = torch.linalg.svdvals(now_c ).real
        # Normalise so that the largest singular value is 1 (scale-invariant).
        sv_last = sv_last / (sv_last[0] + 1e-30)
        sv_now  = sv_now  / (sv_now [0] + 1e-30)
        # Weight differences by SV magnitude: large SVs must converge tightly,
        # small SVs are allowed proportionally larger absolute fluctuations.
        sv_weight = (sv_now + sv_last) / 2
        delta = ((sv_now - sv_last).abs() * sv_weight).max()
        max_delta = torch.maximum(max_delta, delta)

    # ── Fixed-point pathology guard ──────────────────────────────────────────
    # Two degenerate fixed points must be rejected even if max_delta is small:
    #
    #   (a) Zero fixed point: all corner/transfer norms ≈ 0 (hard collapse).
    #       Detected by raw norm threshold.
    #
    #   (b) Paramagnetic (maximally-mixed) fixed point: corners have small but
    #       nonzero norms, yet the corner-product rho matrices have a FLAT
    #       singular-value spectrum (all SVs ≈ 1/chi after normalisation).
    #       In this case every rho is proportional to the identity → all
    #       two-site correlations are zero → energy = 0.  This IS a genuine
    #       fixed point so max_delta converges — but to the wrong state.
    #       Detected by the flatness ratio  sv_last / sv_first  of now_rho1..3.
    #
    # Returning False forces CTMRG to keep iterating (or exhaust max_iterations),
    # after which the outer restart loop tries stronger noise.
    #
    # Flatness threshold: a perfectly flat chi-dimensional spectrum has all
    # normalised SVs = 1.0.  We flag collapse when the TRAILING SV (index -1)
    # after normalisation is > 0.85 — i.e., the spectrum spans less than a
    # 15% range from largest to smallest.  Physical iPEPS environments have
    # strongly decaying SV spectra (trailing SV typically < 0.1).
    _ZERO_NORM_THR   = 1e-15   # raised from 1e-30: catch soft-collapse norms
    _FLAT_SPEC_THR   = 0.85    # trailing normalised SV above this → flat/paramagnetic

    # Collect the three "now" corner-product rhos (reuse the last batch computed
    # just above in the third rho_pairs loop: now_rho1, now_rho2, now_rho3).
    _now_rhos_for_guard = (now_rho1, now_rho2, now_rho3)
    _all_bad = True
    for _nc_rho in _now_rhos_for_guard:
        _sv = torch.linalg.svdvals(_nc_rho).real
        _s0 = _sv[0].item()
        if _s0 < _ZERO_NORM_THR:
            continue   # truly zero rho — bad, keep _all_bad=True
        _sv_norm_tail = (_sv[-1] / (_s0 + 1e-30)).item()
        if _sv_norm_tail < _FLAT_SPEC_THR:
            _all_bad = False   # at least one rho has a non-flat spectrum → OK
            break
    if _all_bad:
        # Zero or paramagnetic fixed point.  Return None (not False) so the
        # caller can abort the iteration loop immediately and escalate noise,
        # rather than running all remaining iterations uselessly.
        return None

    # Single GPU→CPU sync for the entire convergence check.
    return (max_delta < env_conv_threshold).item()



def update_environmentCTs_1to2(C21CD, C32EF, C13AB, T1F, T2A, T2B, T3C, T3D, T1E, A,B,C,D,E,F, chi, D_squared):
    """
    Perform one CTMRG renormalisation step: environment type 1 → type 2.

    Absorbs the site tensors ``E``, ``B``, ``A``, ``D``, ``C``, ``F`` into
    the type-1 corner and transfer tensors and produces the type-2 set.  The
    three new (grown) corner matrices are truncated to bond dimension ``chi``
    using ``trunc_rhoCCC``, and all six new transfer tensors are renormalised.

    Tensor naming convention:  ``C21EB`` is the corner connecting legs 2→1
    in the sublattice-EB orientation; ``T1D`` is the type-1 transfer tensor
    carrying the ``D``-site bond; etc.

    Args:
        C21CD, C32EF, C13AB (torch.Tensor): Type-1 corner matrices,
            shape ``(chi, chi)``.
        T1F, T2A, T2B, T3C, T3D, T1E (torch.Tensor): Type-1 transfer tensors,
            shape ``(chi, D_squared, D_squared)``.
        A, B, C, D, E, F (torch.Tensor): Double-layer site tensors,
            shape ``(D_squared, D_squared)``.
        chi (int): Target bond dimension.
        D_squared (int): ``D_bond ** 2``.

    Returns:
        Tuple of 9 torch.Tensor: ``(C21EB, C32AD, C13CF, T1D, T2C, T2F,
        T3E, T3B, T1A)`` — the type-2 corners (shape ``(chi, chi)``) and
        transfer tensors (shape ``(chi, D_squared, D_squared)``).
    """

    C21EB = oe.contract("YX,MYa,LXb,amg,lbg->MmLl",
                           C21CD,T1F,T2A,E,B,
                           optimize=[(0,1),(1,3),(0,1),(0,1)],
                           backend='torch')
    
    C21EB = C21EB.reshape(chi*D_squared,chi*D_squared)
    
    C32AD = oe.contract("ZY,NZb,MYg,abn,amg->NnMm",
                           C32EF,T2B,T3C,A,D,
                           optimize=[(0,1),(1,3),(0,1),(0,1)],
                           backend='torch')
    
    C32AD = C32AD.reshape(chi*D_squared,chi*D_squared)
    
    C13CF = oe.contract("XZ,LXg,NZa,lbg,abn->LlNn",
                           C13AB,T3D,T1E,C,F,
                           optimize=[(0,1),(1,3),(0,1),(0,1)],
                           backend='torch')
    
    C13CF = C13CF.reshape(chi*D_squared,chi*D_squared)


    V2A, C21EB, U1F, V3C, C32AD, U2B, V1E, C13CF, U3D = trunc_rhoCCC(
                        C21EB, C32AD, C13CF, chi, D_squared)


    T3E = oe.contract("OYa,abg,ObM->YMg",
                        T1F,E,U1F,
                        optimize=[(0,1),(0,1)],
                        backend='torch')

    T3B = oe.contract("OXb,abg,LOa->XLg",
                        T2A,B,V2A,
                        optimize=[(0,1),(0,1)],
                        backend='torch')

    T1A = oe.contract("OZb,abg,OgN->ZNa",
                        T2B,A,U2B,
                        optimize=[(0,1),(0,1)],
                        backend='torch')

    T1D = oe.contract("OYg,abg,MOb->YMa",
                        T3C,D,V3C,
                        optimize=[(0,1),(0,1)],
                        backend='torch')

    T2C = oe.contract("OXg,abg,OaL->XLb",
                        T3D,C,U3D,
                        optimize=[(0,1),(0,1)],
                        backend='torch')

    T2F = oe.contract("OZa,abg,NOg->ZNb",
                        T1E,F,V1E,
                        optimize=[(0,1),(0,1)],
                        backend='torch')

    # ── End-of-update transfer normalization (mirrors norm_env_2) ────────────
    # Expand double-layer (D²,D²,D²) → rank-6 (D,D,D,D,D,D) so the contraction
    # strings are structurally identical to norm_env_2; then divide all 6 Ts
    # by <iPEPS|iPEPS>^(1/6).







    T3E = T3E/torch.linalg.norm(T3E)
    T3B = T3B/torch.linalg.norm(T3B)
    T1A = T1A/torch.linalg.norm(T1A)
    T1D = T1D/torch.linalg.norm(T1D)
    T2C = T2C/torch.linalg.norm(T2C)
    T2F = T2F/torch.linalg.norm(T2F)



    """

    # Fix both magnitude and complex phase so <iPEPS|iPEPS> ≈ 1 (real).
    norm_abs = norm_val.abs()
    if torch.isfinite(norm_abs).item() and (norm_abs > torch.finfo(norm_abs.dtype).tiny).item():
        phase = norm_val / norm_abs
        T3E = T3E / phase

        scaleT = norm_abs.pow(1.0 / 6.0)
        if torch.isfinite(scaleT).item() and (scaleT > torch.finfo(scaleT.dtype).tiny).item():
            T3E = T3E / scaleT
            T3B = T3B / scaleT
            T1A = T1A / scaleT
            T1D = T1D / scaleT
            T2C = T2C / scaleT
            T2F = T2F / scaleT

    # ── Individual equalization: preserve norm_env_2=1 while balancing magnitudes ──
    # Each T appears linearly in exactly one closed_* factor; scale factors multiply to 1.
    nT3E = torch.linalg.norm(T3E).real
    nT3B = torch.linalg.norm(T3B).real
    nT1A = torch.linalg.norm(T1A).real
    nT1D = torch.linalg.norm(T1D).real
    nT2C = torch.linalg.norm(T2C).real
    nT2F = torch.linalg.norm(T2F).real
    nT_prod = nT3E * nT3B * nT1A * nT1D * nT2C * nT2F
    if torch.isfinite(nT_prod).item() and (nT_prod > torch.finfo(nT3E.dtype).tiny).item():
        nT_geo = nT_prod.pow(1.0 / 6.0)
        T3E = T3E * (nT_geo / nT3E)
        T3B = T3B * (nT_geo / nT3B)
        T1A = T1A * (nT_geo / nT1A)
        T1D = T1D * (nT_geo / nT1D)
        T2C = T2C * (nT_geo / nT2C)
        T2F = T2F * (nT_geo / nT2F)
    
    """

    return C21EB, C32AD, C13CF, T1D, T2C, T2F, T3E, T3B, T1A




def update_environmentCTs_2to3(C21EB, C32AD, C13CF, T1D, T2C, T2F, T3E, T3B, T1A, A,B,C,D,E,F, chi, D_squared):
    """
    Perform one CTMRG renormalisation step: environment type 2 → type 3.

    Absorbs site tensors ``A``, ``F``, ``C``, ``B``, ``E``, ``D`` into the
    type-2 environment and produces the type-3 set, following the same
    absorb-truncate-renormalise pattern as ``update_environmentCTs_1to2``.

    Args:
        C21EB, C32AD, C13CF (torch.Tensor): Type-2 corner matrices,
            shape ``(chi, chi)``.
        T1D, T2C, T2F, T3E, T3B, T1A (torch.Tensor): Type-2 transfer tensors,
            shape ``(chi, D_squared, D_squared)``.
        A, B, C, D, E, F (torch.Tensor): Double-layer site tensors,
            shape ``(D_squared, D_squared)``.
        chi (int): Target bond dimension.
        D_squared (int): ``D_bond ** 2``.

    Returns:
        Tuple of 9 torch.Tensor: ``(C21AF, C32CB, C13ED, T1B, T2E, T2D,
        T3A, T3F, T1C)`` — the type-3 corners and transfer tensors.
    """

    C21AF = oe.contract("YX,MYa,LXb,amg,lbg->MmLl",
                           C21EB,T1D,T2C,A,F,
                           optimize=[(0,1),(1,3),(0,1),(0,1)],
                           backend='torch')
    
    C21AF = C21AF.reshape(chi*D_squared,chi*D_squared)
    
    C32CB = oe.contract("ZY,NZb,MYg,abn,amg->NnMm",
                           C32AD,T2F,T3E,C,B,
                           optimize=[(0,1),(1,3),(0,1),(0,1)],
                           backend='torch')
    
    C32CB = C32CB.reshape(chi*D_squared,chi*D_squared)
    
    C13ED = oe.contract("XZ,LXg,NZa,lbg,abn->LlNn",
                           C13CF,T3B,T1A,E,D,
                           optimize=[(0,1),(1,3),(0,1),(0,1)],
                           backend='torch')
    
    C13ED = C13ED.reshape(chi*D_squared,chi*D_squared)

    V2C, C21AF, U1D, V3E, C32CB, U2F, V1A, C13ED, U3B = trunc_rhoCCC(
                        C21AF, C32CB, C13ED, chi, D_squared)

    T3A = oe.contract("OYa,abg,ObM->YMg",
                        T1D,A,U1D,
                        optimize=[(0,1),(0,1)],
                        backend='torch')

    T3F = oe.contract("OXb,abg,LOa->XLg",
                        T2C,F,V2C,
                        optimize=[(0,1),(0,1)],
                        backend='torch')

    T1C = oe.contract("OZb,abg,OgN->ZNa",
                        T2F,C,U2F,
                        optimize=[(0,1),(0,1)],
                        backend='torch')

    T1B = oe.contract("OYg,abg,MOb->YMa",
                        T3E,B,V3E,
                        optimize=[(0,1),(0,1)],
                        backend='torch')

    T2E = oe.contract("OXg,abg,OaL->XLb",
                        T3B,E,U3B,
                        optimize=[(0,1),(0,1)],
                        backend='torch')

    T2D = oe.contract("OZa,abg,NOg->ZNb",
                        T1A,D,V1A,
                        optimize=[(0,1),(0,1)],
                        backend='torch')
    









    T3A = T3A/torch.linalg.norm(T3A)
    T3F = T3F/torch.linalg.norm(T3F)
    T1C = T1C/torch.linalg.norm(T1C)
    T1B = T1B/torch.linalg.norm(T1B)
    T2E = T2E/torch.linalg.norm(T2E)
    T2D = T2D/torch.linalg.norm(T2D)




    """
    # Fix both magnitude and complex phase so <iPEPS|iPEPS> ≈ 1 (real).
    norm_abs = norm_val.abs()
    if torch.isfinite(norm_abs).item() and (norm_abs > torch.finfo(norm_abs.dtype).tiny).item():
        phase = norm_val / norm_abs
        T3A = T3A / phase

        scaleT = norm_abs.pow(1.0 / 6.0)
        if torch.isfinite(scaleT).item() and (scaleT > torch.finfo(scaleT.dtype).tiny).item():
            T3A = T3A / scaleT
            T3F = T3F / scaleT
            T1C = T1C / scaleT
            T1B = T1B / scaleT
            T2E = T2E / scaleT
            T2D = T2D / scaleT

    # ── Individual equalization: preserve norm_env_3=1 while balancing magnitudes ──
    nT3A = torch.linalg.norm(T3A).real
    nT3F = torch.linalg.norm(T3F).real
    nT1C = torch.linalg.norm(T1C).real
    nT1B = torch.linalg.norm(T1B).real
    nT2E = torch.linalg.norm(T2E).real
    nT2D = torch.linalg.norm(T2D).real
    nT_prod = nT3A * nT3F * nT1C * nT1B * nT2E * nT2D
    if torch.isfinite(nT_prod).item() and (nT_prod > torch.finfo(nT3A.dtype).tiny).item():
        nT_geo = nT_prod.pow(1.0 / 6.0)
        T3A = T3A * (nT_geo / nT3A)
        T3F = T3F * (nT_geo / nT3F)
        T1C = T1C * (nT_geo / nT1C)
        T1B = T1B * (nT_geo / nT1B)
        T2E = T2E * (nT_geo / nT2E)
        T2D = T2D * (nT_geo / nT2D)
    """

    return C21AF, C32CB, C13ED, T1B, T2E, T2D, T3A, T3F, T1C




def update_environmentCTs_3to1(C21AF, C32CB, C13ED, T1B, T2E, T2D, T3A, T3F, T1C, A,B,C,D,E,F, chi, D_squared):
    """
    Perform one CTMRG renormalisation step: environment type 3 → type 1.

    Absorbs site tensors ``C``, ``D``, ``E``, ``F``, ``A``, ``B`` into the
    type-3 environment and produces the type-1 set, completing the three-step
    cycle ``1→2→3→1``.

    Args:
        C21AF, C32CB, C13ED (torch.Tensor): Type-3 corner matrices,
            shape ``(chi, chi)``.
        T1B, T2E, T2D, T3A, T3F, T1C (torch.Tensor): Type-3 transfer tensors,
            shape ``(chi, D_squared, D_squared)``.
        A, B, C, D, E, F (torch.Tensor): Double-layer site tensors,
            shape ``(D_squared, D_squared)``.
        chi (int): Target bond dimension.
        D_squared (int): ``D_bond ** 2``.

    Returns:
        Tuple of 9 torch.Tensor: ``(C21CD, C32EF, C13AB, T1F, T2A, T2B,
        T3C, T3D, T1E)`` — the renewed type-1 corners and transfer tensors.
    """

    C21CD = oe.contract("YX,MYa,LXb,amg,lbg->MmLl",
                           C21AF,T1B,T2E,C,D,
                           optimize=[(0,1),(1,3),(0,1),(0,1)],
                           backend='torch')
    
    C21CD = C21CD.reshape(chi*D_squared,chi*D_squared)
    
    C32EF = oe.contract("ZY,NZb,MYg,abn,amg->NnMm",
                           C32CB,T2D,T3A,E,F,
                           optimize=[(0,1),(1,3),(0,1),(0,1)],
                           backend='torch')
    
    C32EF = C32EF.reshape(chi*D_squared,chi*D_squared)
    
    C13AB = oe.contract("XZ,LXg,NZa,lbg,abn->LlNn",
                           C13ED,T3F,T1C,A,B,
                           optimize=[(0,1),(1,3),(0,1),(0,1)],
                           backend='torch')
    
    C13AB = C13AB.reshape(chi*D_squared,chi*D_squared)

    V2E, C21CD, U1B, V3A, C32EF, U2D, V1C, C13AB, U3F = trunc_rhoCCC(
                        C21CD, C32EF, C13AB, chi, D_squared)




    T3C = oe.contract("OYa,abg,ObM->YMg",
                        T1B,C,U1B,
                        optimize=[(0,1),(0,1)],
                        backend='torch')

    T3D = oe.contract("OXb,abg,LOa->XLg",
                        T2E,D,V2E,
                        optimize=[(0,1),(0,1)],
                        backend='torch')

    T1E = oe.contract("OZb,abg,OgN->ZNa",
                        T2D,E,U2D,
                        optimize=[(0,1),(0,1)],
                        backend='torch')

    T1F = oe.contract("OYg,abg,MOb->YMa",
                        T3A,F,V3A,
                        optimize=[(0,1),(0,1)],
                        backend='torch')

    T2A = oe.contract("OXg,abg,OaL->XLb",
                        T3F,A,U3F,
                        optimize=[(0,1),(0,1)],
                        backend='torch')

    T2B = oe.contract("OZa,abg,NOg->ZNb",
                        T1C,B,V1C,
                        optimize=[(0,1),(0,1)],
                        backend='torch')
    









    T3C = T3C/torch.linalg.norm(T3C)
    T3D = T3D/torch.linalg.norm(T3D)
    T1E = T1E/torch.linalg.norm(T1E)
    T1F = T1F/torch.linalg.norm(T1F)
    T2A = T2A/torch.linalg.norm(T2A)
    T2B = T2B/torch.linalg.norm(T2B)





    """
    
    # Fix both magnitude and complex phase so <iPEPS|iPEPS> ≈ 1 (real).
    norm_abs = norm_val.abs()
    if torch.isfinite(norm_abs).item() and (norm_abs > torch.finfo(norm_abs.dtype).tiny).item():
        phase = norm_val / norm_abs
        T3C = T3C / phase

        scaleT = norm_abs.pow(1.0 / 6.0)
        if torch.isfinite(scaleT).item() and (scaleT > torch.finfo(scaleT.dtype).tiny).item():
            T3C = T3C / scaleT
            T3D = T3D / scaleT
            T1E = T1E / scaleT
            T1F = T1F / scaleT
            T2A = T2A / scaleT
            T2B = T2B / scaleT

    # ── Individual equalization: preserve norm_env_1=1 while balancing magnitudes ──
    nT3C = torch.linalg.norm(T3C).real
    nT3D = torch.linalg.norm(T3D).real
    nT1E = torch.linalg.norm(T1E).real
    nT1F = torch.linalg.norm(T1F).real
    nT2A = torch.linalg.norm(T2A).real
    nT2B = torch.linalg.norm(T2B).real
    nT_prod = nT3C * nT3D * nT1E * nT1F * nT2A * nT2B
    if torch.isfinite(nT_prod).item() and (nT_prod > torch.finfo(nT3C.dtype).tiny).item():
        nT_geo = nT_prod.pow(1.0 / 6.0)
        T3C = T3C * (nT_geo / nT3C)
        T3D = T3D * (nT_geo / nT3D)
        T1E = T1E * (nT_geo / nT1E)
        T1F = T1F * (nT_geo / nT1F)
        T2A = T2A * (nT_geo / nT2A)
        T2B = T2B * (nT_geo / nT2B)

    """

    return C21CD, C32EF, C13AB, T1F, T2A, T2B, T3C, T3D, T1E




# ─────────────────────────────────────────────────────────────────────────────
# Memory-efficient CTMRG: each of the three update steps is wrapped with
# torch.utils.checkpoint.checkpoint (aliased as _ckpt).
#
# Why checkpoint and NOT a custom autograd.Function:
#   The update functions call truncated_svd_propack which, on the partial-SVD
#   path, draws R=torch.randn(...).  A custom Function that re-runs the forward
#   in backward() would draw a *different* R → different U,S,V → wrong gradient.
#   torch.utils.checkpoint saves and restores the full PyTorch RNG state before
#   the recompute, so R is bit-identical to the original forward pass.
#
# What is saved vs recomputed:
#   forward:  only the 15 small input tensors are kept alive for the recompute;
#             all M×M intermediate contractions and SVD factors are freed.
#   backward: the update function is re-run once per step to rebuild the local
#             graph, grads are computed, then the local graph is freed.
#
# Memory: O(N_iter × chi² × D²)  instead of  O(N_iter × (chi·D²)²).
# Cost  : ≈1 extra forward pass per update step during the backward phase.
# ─────────────────────────────────────────────────────────────────────────────

def CTMRG_from_init_to_stop(A,B,C,D,E,F,
                            chi: int,
                            D_squared: int,
                            a_third_max_iterations: int,
                            env_conv_threshold: float,
                            identity_init: bool = False) -> tuple:
    """
    This function performs the CTMRG algorithm from the initial state to the stopping criterion.

    Args:
        A, B, C, D, E, F (torch.Tensor): The 6 local tensors.
        max_iterations (int): The maximum number of iterations to perform.
        D_squared (int): The square of the bond dimension of the local state projector a~f.
        chi (int): The nominal desired bond dimension for the transfer tensors.
        env_conv_threshold (float): The threshold for environment convergence.

    Returns:
        A tuple of 28 elements: the 27 final environment tensors followed by
        ctm_steps (int) — the number of CTMRG iterations actually performed.
    """
    # ── Auto-restart with escalating random noise on zero-collapse ──────
    #
    # First-principle analysis of zero-fixed-point collapse:
    #
    #   The CTMRG map T(env; A..F) has TWO fixed points:
    #     (1)  env = 0   — trivial, always stable.
    #     (2)  env = env* — physical, represents the infinite-lattice environment.
    #
    #   The zero fixed point is a basin attractor: once env drifts into its
    #   basin (corners all near zero), the update steps T contract it further
    #   toward zero (zero × anything = zero).  Different initialisations probe
    #   different basins of attraction.
    #
    #   Identity / ones init alone is NOT guaranteed to escape zero — the
    #   double-layer contractions can produce cancellations or near-zero
    #   results from specific tensor configurations, pulling the deterministic
    #   identity start toward the zero basin.
    #
    #   HIGH-INTENSITY RANDOM NOISE is what breaks the symmetry: it perturbs
    #   the environment into a generic direction that is unlikely to lie in the
    #   zero basin.  The stronger the noise, the more aggressively we escape.
    #
    #   Strategy: escalating restart schedule (up to 4 attempts):
    #     restart 0 — caller's choice (contraction-based or identity_init)
    #     restart 1 — identity_init + 10% Gaussian noise
    #     restart 2 — identity_init + 100% noise  (random dominates structure)
    #     restart 3 — identity_init + 1000% noise (essentially pure random)
    #
    #   Noise is injected under torch.no_grad() → does not affect gradients.
    #   Gradients flow only through the converged CTMRG update steps.
    #
    _ZERO_COLLAPSE_THR = 1e-20   # raised: catch soft-collapse (norms tiny but above 1e-30)
    _NOISE_SCALES = (0.0, 0.1, 1.0, 10.0)       # per-restart noise fractions

    for _restart in range(len(_NOISE_SCALES)):
        _use_identity = identity_init if _restart == 0 else True

        lastC21CD, lastC32EF, lastC13AB, lastT1F, lastT2A, lastT2B, lastT3C, lastT3D, lastT1E = None, None, None, None, None, None, None, None, None
        lastC21EB, lastC32AD, lastC13CF, lastT1D, lastT2C, lastT2F, lastT3E, lastT3B, lastT1A = None, None, None, None, None, None, None, None, None
        lastC21AF, lastC32CB, lastC13ED, lastT1B, lastT2E, lastT2D, lastT3A, lastT3F, lastT1C = None, None, None, None, None, None, None, None, None
        # ── Initialization under torch.no_grad() ─────────────────────────
        # The init only sets the starting point for the iterative CTMRG loop.
        # Gradients flow through the converged update steps, not the init.
        # Running under no_grad prevents the (D²×D², D²×D²) SVD intermediates
        # from being kept alive in the autograd graph (saves ~75 MB for D=5).
        with torch.no_grad():
            _D_bond = round(D_squared ** 0.5)
            # D ≥ 8: pre-truncation T tensors are (D^4, D^4, D^2) ≥ 8 GiB — OOM
            # even in no_grad.  Force identity_init so only (chi,chi) and
            # (chi,chi,D²) tensors are allocated.
            if _D_bond >= 8 and not _use_identity:
                _use_identity = True
            nowC21CD, nowC32EF, nowC13AB, nowT1F, nowT2A, nowT2B, nowT3C, nowT3D, nowT1E = initialize_envCTs_1(A,B,C,D,E,F, chi, D_squared, identity_init=_use_identity)
            # ── Inject escalating random noise for restarts ──────────────
            _ns = _NOISE_SCALES[_restart]
            if _ns > 0.0:
                for _t in (nowC21CD, nowC32EF, nowC13AB,
                           nowT1F, nowT2A, nowT2B, nowT3C, nowT3D, nowT1E):
                    _tn = torch.linalg.norm(_t).clamp(min=1.0).item()
                    _t.add_(_ns * _tn * torch.randn_like(_t))
        nowC21EB, nowC32AD, nowC13CF, nowT1D, nowT2C, nowT2F, nowT3E, nowT3B, nowT1A = None, None, None, None, None, None, None, None, None
        nowC21AF, nowC32CB, nowC13ED, nowT1B, nowT2E, nowT2D, nowT3A, nowT3F, nowT1C = None, None, None, None, None, None, None, None, None

        # Perform the CTMRG iterations until convergence
        _collapsed = False
        ctm_steps = a_third_max_iterations  # will be overwritten on early convergence
        for iteration in range(a_third_max_iterations):

            if not (lastC21CD is None or lastC21EB is None or lastC21AF is None):
                _cv = check_env_CV_using_3rho(
                                        lastC21CD.detach(), lastC32EF.detach(), lastC13AB.detach(),
                                        nowC21CD.detach(), nowC32EF.detach(), nowC13AB.detach(),
                                        lastC21EB.detach(), lastC32AD.detach(), lastC13CF.detach(),
                                        nowC21EB.detach(), nowC32AD.detach(), nowC13CF.detach(),
                                        lastC21AF.detach(), lastC32CB.detach(), lastC13ED.detach(),
                                        nowC21AF.detach(), nowC32CB.detach(), nowC13ED.detach(),
                                        env_conv_threshold)
                if _cv is None:
                    # Paramagnetic or near-zero fixed point — abort immediately
                    # and let the restart loop escalate the noise level.
                    _collapsed = True
                    break
                elif _cv:
                    ctm_steps = iteration + 1
                    break

            # Update the environment corner and edge transfer tensors
            # match iteration % 3 :
            #     case 0 : 
            lastC21EB, lastC32AD, lastC13CF, lastT1D, lastT2C, lastT2F, lastT3E, lastT3B, lastT1A = \
            nowC21EB, nowC32AD, nowC13CF, nowT1D, nowT2C, nowT2F, nowT3E, nowT3B, nowT1A
            nowC21EB, nowC32AD, nowC13CF, nowT1D, nowT2C, nowT2F, nowT3E, nowT3B, nowT1A = _ckpt(
                update_environmentCTs_1to2,
                nowC21CD, nowC32EF, nowC13AB, nowT1F, nowT2A, nowT2B, nowT3C, nowT3D, nowT1E,
                A, B, C, D, E, F, chi, D_squared,
                use_reentrant=False)
                
            #     case 1 : 
            lastC21AF, lastC32CB, lastC13ED, lastT1B, lastT2E, lastT2D, lastT3A, lastT3F, lastT1C = \
            nowC21AF, nowC32CB, nowC13ED, nowT1B, nowT2E, nowT2D, nowT3A, nowT3F, nowT1C

            nowC21AF, nowC32CB, nowC13ED, nowT1B, nowT2E, nowT2D, nowT3A, nowT3F, nowT1C = _ckpt(
                update_environmentCTs_2to3,
                nowC21EB, nowC32AD, nowC13CF, nowT1D, nowT2C, nowT2F, nowT3E, nowT3B, nowT1A,
                A, B, C, D, E, F, chi, D_squared,
                use_reentrant=False)
                
            #     case 2 : 
            lastC21CD, lastC32EF, lastC13AB, lastT1F, lastT2A, lastT2B, lastT3C, lastT3D, lastT1E = \
            nowC21CD, nowC32EF, nowC13AB, nowT1F, nowT2A, nowT2B, nowT3C, nowT3D, nowT1E

            nowC21CD, nowC32EF, nowC13AB, nowT1F, nowT2A, nowT2B, nowT3C, nowT3D, nowT1E = _ckpt(
                update_environmentCTs_3to1,
                nowC21AF, nowC32CB, nowC13ED, nowT1B, nowT2E, nowT2D, nowT3A, nowT3F, nowT1C,
                A, B, C, D, E, F, chi, D_squared,
                use_reentrant=False)

            # ── Zero-collapse detection ───────────────────────────────────
            # Check on EVERY restart (not just the first).  If collapsed and
            # more restarts remain, the outer loop retries with stronger noise.
            if (    torch.linalg.norm(nowC21CD).item() < _ZERO_COLLAPSE_THR
                    and torch.linalg.norm(nowC21EB).item() < _ZERO_COLLAPSE_THR
                    and torch.linalg.norm(nowC21AF).item() < _ZERO_COLLAPSE_THR):
                _collapsed = True
                break

        # Release Python references to stale last-iteration env tensors so the
        # autograd engine can GC them as soon as the backward pass processes them.
        del lastC21CD, lastC32EF, lastC13AB, lastT1F, lastT2A, lastT2B, lastT3C, lastT3D, lastT1E
        del lastC21EB, lastC32AD, lastC13CF, lastT1D, lastT2C, lastT2F, lastT3E, lastT3B, lastT1A
        del lastC21AF, lastC32CB, lastC13ED, lastT1B, lastT2E, lastT2D, lastT3A, lastT3F, lastT1C

        if not _collapsed:
            break   # success — exit restart loop
        # else: escalate noise and retry

    return  nowC21CD, nowC32EF, nowC13AB, nowT1F, nowT2A, nowT2B, nowT3C, nowT3D, nowT1E, \
            nowC21EB, nowC32AD, nowC13CF, nowT1D, nowT2C, nowT2F, nowT3E, nowT3B, nowT1A, \
            nowC21AF, nowC32CB, nowC13ED, nowT1B, nowT2E, nowT2D, nowT3A, nowT3F, nowT1C, \
            ctm_steps










def energy_expectation_nearest_neighbor_3ebadcf_bonds(
                a,b,c,d,e,f, 
                Jeb,Jad,Jcf,
                Jfa,Jde,Jbc,
                Jae,Jec,Jca,Jdb,Jbf,Jfd,
                SdotS,
                chi, D_bond, d_PHYS, 
                C21CD,C32EF,C13AB,T1F,T2A,T2B,T3C,T3D,T1E):
    """Energy for 3 ebadcf bonds.

    Memory strategy (GPU only):
      D ≤ 7 : open tensors ≤ 440 MB total → pre-build all 6, _ckpt per bond.
      D ≥ 8 : each bond's _ckpt closure builds its OWN pair of opens internally →
              at most 2 opens alive simultaneously → ~1.6 GB peak open-tensor usage.
    CPU: no checkpoint, no closure — pre-build all 6 once, direct calls.
    """
    _on_gpu = DEVICE.type == 'cuda'

    # ── GPU + large-D: lazy-open closures + _ckpt ────────────────────────
    if _on_gpu and D_bond >= 8:
        T1F_r = T1F.reshape(chi,chi,D_bond,D_bond)
        T2A_r = T2A.reshape(chi,chi,D_bond,D_bond)
        T2B_r = T2B.reshape(chi,chi,D_bond,D_bond)
        T3C_r = T3C.reshape(chi,chi,D_bond,D_bond)
        T3D_r = T3D.reshape(chi,chi,D_bond,D_bond)
        T1E_r = T1E.reshape(chi,chi,D_bond,D_bond)

        D2 = chi * D_bond * D_bond
        _bop = lambda einstr, *args, opt: oe.contract(einstr, *args,
                                                       optimize=opt, backend="torch")
        # Build closed tensors from cheaply-constructed opens (freed immediately)
        _open_E = _bop("YX,MYar,abci,rstj->MbsXctij", C21CD, T1F_r, e, e.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
        closed_E = oe.contract("MXii->MX", _open_E, backend="torch"); del _open_E
        _open_D = _bop("MYct,abci,rstj->YarMbsij",    T3C_r, d, d.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
        closed_D = oe.contract("YMii->YM", _open_D, backend="torch"); del _open_D
        _open_A = _bop("ZY,NZbs,abci,rstj->NctYarij", C32EF, T2B_r, a, a.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
        closed_A = oe.contract("NYii->NY", _open_A, backend="torch"); del _open_A
        _open_F = _bop("NZar,abci,rstj->ZbsNctij",    T1E_r, f, f.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
        closed_F = oe.contract("ZNii->ZN", _open_F, backend="torch"); del _open_F
        _open_C = _bop("XZ,LXct,abci,rstj->LarZbsij", C13AB, T3D_r, c, c.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
        closed_C = oe.contract("LZii->LZ", _open_C, backend="torch"); del _open_C
        _open_B = _bop("LXbs,abci,rstj->XctLarij",    T2A_r, b, b.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
        closed_B = oe.contract("XLii->XL", _open_B, backend="torch"); del _open_B

        # ── NN bond closures (pair products computed ON-THE-FLY inside _ckpt
        #    so they are never pinned in saved_tensors; saves 6×(χD²)²) ──
        def _nn_AD(cE, cB, cC, cF, SdotS):
            _oA = _bop("ZY,NZbs,abci,rstj->NctYarij", C32EF, T2B_r, a, a.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
            _oD = _bop("MYct,abci,rstj->YarMbsij",    T3C_r, d, d.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
            return _compute_nn_bond_energy(_oA, _oD, torch.mm(cE, cB), torch.mm(cC, cF), SdotS)
        def _nn_CF(cA, cD, cE, cB, SdotS):
            _oC = _bop("XZ,LXct,abci,rstj->LarZbsij", C13AB, T3D_r, c, c.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
            _oF = _bop("NZar,abci,rstj->ZbsNctij",    T1E_r, f, f.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
            return _compute_nn_bond_energy(_oC, _oF, torch.mm(cA, cD), torch.mm(cE, cB), SdotS)
        def _nn_EB(cC, cF, cA, cD, SdotS):
            _oE = _bop("YX,MYar,abci,rstj->MbsXctij", C21CD, T1F_r, e, e.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
            _oB = _bop("LXbs,abci,rstj->XctLarij",    T2A_r, b, b.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
            return _compute_nn_bond_energy(_oE, _oB, torch.mm(cC, cF), torch.mm(cA, cD), SdotS)
        def _nn_FA(cD, cE, cB, cC, SdotS):
            _oF = _bop("NZar,abci,rstj->ZbsNctij",    T1E_r, f, f.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
            _oA = _bop("ZY,NZbs,abci,rstj->NctYarij", C32EF, T2B_r, a, a.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
            return _compute_nn_bond_energy(_oF, _oA, torch.mm(cD, cE), torch.mm(cB, cC), SdotS)
        def _nn_DE(cB, cC, cF, cA, SdotS):
            _oD = _bop("MYct,abci,rstj->YarMbsij",    T3C_r, d, d.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
            _oE = _bop("YX,MYar,abci,rstj->MbsXctij", C21CD, T1F_r, e, e.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
            return _compute_nn_bond_energy(_oD, _oE, torch.mm(cB, cC), torch.mm(cF, cA), SdotS)
        def _nn_BC(cF, cA, cD, cE, SdotS):
            _oB = _bop("LXbs,abci,rstj->XctLarij",    T2A_r, b, b.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
            _oC = _bop("XZ,LXct,abci,rstj->LarZbsij", C13AB, T3D_r, c, c.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
            return _compute_nn_bond_energy(_oB, _oC, torch.mm(cF, cA), torch.mm(cD, cE), SdotS)

        # ── NNN bond closures (sequential open builds: peak = 2·open+W, not 4·open) ──
        def _nnn_AE(cD, cB, cC, cF, SdotS):
            return _compute_nnn_bond_energy_seq(
                lambda: _bop("ZY,NZbs,abci,rstj->NctYarij", C32EF, T2B_r, a, a.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS),
                cD,
                lambda: _bop("YX,MYar,abci,rstj->MbsXctij", C21CD, T1F_r, e, e.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS),
                cB, torch.mm(cC, cF), SdotS)
        def _nnn_EC(cB, cF, cA, cD, SdotS):
            return _compute_nnn_bond_energy_seq(
                lambda: _bop("YX,MYar,abci,rstj->MbsXctij", C21CD, T1F_r, e, e.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS),
                cB,
                lambda: _bop("XZ,LXct,abci,rstj->LarZbsij", C13AB, T3D_r, c, c.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS),
                cF, torch.mm(cA, cD), SdotS)
        def _nnn_CA(cF, cD, cE, cB, SdotS):
            return _compute_nnn_bond_energy_seq(
                lambda: _bop("XZ,LXct,abci,rstj->LarZbsij", C13AB, T3D_r, c, c.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS),
                cF,
                lambda: _bop("ZY,NZbs,abci,rstj->NctYarij", C32EF, T2B_r, a, a.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS),
                cD, torch.mm(cE, cB), SdotS)
        def _nnn_DB(cE, cC, cF, cA, SdotS):
            return _compute_nnn_bond_energy_seq(
                lambda: _bop("MYct,abci,rstj->YarMbsij",    T3C_r, d, d.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS),
                cE,
                lambda: _bop("LXbs,abci,rstj->XctLarij",    T2A_r, b, b.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS),
                cC, torch.mm(cF, cA), SdotS)
        def _nnn_BF(cC, cA, cD, cE, SdotS):
            return _compute_nnn_bond_energy_seq(
                lambda: _bop("LXbs,abci,rstj->XctLarij",    T2A_r, b, b.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS),
                cC,
                lambda: _bop("NZar,abci,rstj->ZbsNctij",    T1E_r, f, f.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS),
                cA, torch.mm(cD, cE), SdotS)
        def _nnn_FD(cA, cE, cB, cC, SdotS):
            return _compute_nnn_bond_energy_seq(
                lambda: _bop("NZar,abci,rstj->ZbsNctij",    T1E_r, f, f.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS),
                cA,
                lambda: _bop("MYct,abci,rstj->YarMbsij",    T3C_r, d, d.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS),
                cE, torch.mm(cB, cC), SdotS)

        E_AD = Jad * _ckpt(_nn_AD, closed_E, closed_B, closed_C, closed_F, SdotS, use_reentrant=False)
        E_CF = Jcf * _ckpt(_nn_CF, closed_A, closed_D, closed_E, closed_B, SdotS, use_reentrant=False)
        E_EB = Jeb * _ckpt(_nn_EB, closed_C, closed_F, closed_A, closed_D, SdotS, use_reentrant=False)
        E_FA = Jfa * _ckpt(_nn_FA, closed_D, closed_E, closed_B, closed_C, SdotS, use_reentrant=False)
        E_DE = Jde * _ckpt(_nn_DE, closed_B, closed_C, closed_F, closed_A, SdotS, use_reentrant=False)
        E_BC = Jbc * _ckpt(_nn_BC, closed_F, closed_A, closed_D, closed_E, SdotS, use_reentrant=False)

        E_AE = Jae * _ckpt(_nnn_AE, closed_D, closed_B, closed_C, closed_F, SdotS, use_reentrant=False)
        E_EC = Jec * _ckpt(_nnn_EC, closed_B, closed_F, closed_A, closed_D, SdotS, use_reentrant=False)
        E_CA = Jca * _ckpt(_nnn_CA, closed_F, closed_D, closed_E, closed_B, SdotS, use_reentrant=False)
        E_DB = Jdb * _ckpt(_nnn_DB, closed_E, closed_C, closed_F, closed_A, SdotS, use_reentrant=False)
        E_BF = Jbf * _ckpt(_nnn_BF, closed_C, closed_A, closed_D, closed_E, SdotS, use_reentrant=False)
        E_FD = Jfd * _ckpt(_nnn_FD, closed_A, closed_E, closed_B, closed_C, SdotS, use_reentrant=False)

        return torch.real((E_AD+E_CF+E_EB + E_FA+E_DE+E_BC)*0.5 +E_AE+E_EC+E_CA +E_DB+E_BF+E_FD)

    # ── Pre-build path: CPU (any D) or GPU D ≤ 7 ────────────────────────
    o, cl = build_open_closed_env1(a, b, c, d, e, f, chi, D_bond, d_PHYS,
                                   C21CD, C32EF, C13AB,
                                   T1F, T2A, T2B, T3C, T3D, T1E)
    AD = torch.mm(cl['A'], cl['D']);  CF = torch.mm(cl['C'], cl['F'])
    EB = torch.mm(cl['E'], cl['B']);  FA = torch.mm(cl['F'], cl['A'])
    DE = torch.mm(cl['D'], cl['E']);  BC = torch.mm(cl['B'], cl['C'])

    if _on_gpu:
        # GPU D ≤ 7: sub-checkpoint each bond (saves ~9 GB backward memory)
        E_AD = Jad * _ckpt(_compute_nn_bond_energy,  o['A'], o['D'], EB, CF, SdotS, use_reentrant=False)
        E_CF = Jcf * _ckpt(_compute_nn_bond_energy,  o['C'], o['F'], AD, EB, SdotS, use_reentrant=False)
        E_EB = Jeb * _ckpt(_compute_nn_bond_energy,  o['E'], o['B'], CF, AD, SdotS, use_reentrant=False)
        E_FA = Jfa * _ckpt(_compute_nn_bond_energy,  o['F'], o['A'], DE, BC, SdotS, use_reentrant=False)
        E_DE = Jde * _ckpt(_compute_nn_bond_energy,  o['D'], o['E'], BC, FA, SdotS, use_reentrant=False)
        E_BC = Jbc * _ckpt(_compute_nn_bond_energy,  o['B'], o['C'], FA, DE, SdotS, use_reentrant=False)
        E_AE = Jae * _ckpt(_compute_nnn_bond_energy, o['A'], cl['D'], o['E'], cl['B'], CF, SdotS, use_reentrant=False)
        E_EC = Jec * _ckpt(_compute_nnn_bond_energy, o['E'], cl['B'], o['C'], cl['F'], AD, SdotS, use_reentrant=False)
        E_CA = Jca * _ckpt(_compute_nnn_bond_energy, o['C'], cl['F'], o['A'], cl['D'], EB, SdotS, use_reentrant=False)
        E_DB = Jdb * _ckpt(_compute_nnn_bond_energy, o['D'], cl['E'], o['B'], cl['C'], FA, SdotS, use_reentrant=False)
        E_BF = Jbf * _ckpt(_compute_nnn_bond_energy, o['B'], cl['C'], o['F'], cl['A'], DE, SdotS, use_reentrant=False)
        E_FD = Jfd * _ckpt(_compute_nnn_bond_energy, o['F'], cl['A'], o['D'], cl['E'], BC, SdotS, use_reentrant=False)
    else:
        # CPU: no checkpoint — direct calls
        E_AD = Jad * _compute_nn_bond_energy(o['A'], o['D'], EB, CF, SdotS)
        E_CF = Jcf * _compute_nn_bond_energy(o['C'], o['F'], AD, EB, SdotS)
        E_EB = Jeb * _compute_nn_bond_energy(o['E'], o['B'], CF, AD, SdotS)
        E_FA = Jfa * _compute_nn_bond_energy(o['F'], o['A'], DE, BC, SdotS)
        E_DE = Jde * _compute_nn_bond_energy(o['D'], o['E'], BC, FA, SdotS)
        E_BC = Jbc * _compute_nn_bond_energy(o['B'], o['C'], FA, DE, SdotS)
        E_AE = Jae * _compute_nnn_bond_energy(o['A'], cl['D'], o['E'], cl['B'], CF, SdotS)
        E_EC = Jec * _compute_nnn_bond_energy(o['E'], cl['B'], o['C'], cl['F'], AD, SdotS)
        E_CA = Jca * _compute_nnn_bond_energy(o['C'], cl['F'], o['A'], cl['D'], EB, SdotS)
        E_DB = Jdb * _compute_nnn_bond_energy(o['D'], cl['E'], o['B'], cl['C'], FA, SdotS)
        E_BF = Jbf * _compute_nnn_bond_energy(o['B'], cl['C'], o['F'], cl['A'], DE, SdotS)
        E_FD = Jfd * _compute_nnn_bond_energy(o['F'], cl['A'], o['D'], cl['E'], BC, SdotS)

    return torch.real((E_AD+E_CF+E_EB + E_FA+E_DE+E_BC)*0.5 +E_AE+E_EC+E_CA +E_DB+E_BF+E_FD)


def energy_expectation_nearest_neighbor_3afcbed_bonds(a,b,c,d,e,f,
                Jaf,Jcb,Jed, 
                Jdc,Jba,Jfe,
                Jca,Jae,Jec,Jbf,Jfd,Jdb,
                SdotS,
                chi, D_bond, d_PHYS, 
                C21EB, C32AD,C13CF,T1D,T2C,T2F,T3E,T3B,T1A):
    """Energy for 3 afcbed bonds.  GPU+D≥8: lazy-open+_ckpt.  CPU: no checkpoint."""
    _on_gpu = DEVICE.type == 'cuda'

    # ── GPU + large-D: lazy-open closures + _ckpt ────────────────────────
    if _on_gpu and D_bond >= 8:
        T1D_r = T1D.reshape(chi,chi,D_bond,D_bond)
        T2C_r = T2C.reshape(chi,chi,D_bond,D_bond)
        T2F_r = T2F.reshape(chi,chi,D_bond,D_bond)
        T3E_r = T3E.reshape(chi,chi,D_bond,D_bond)
        T3B_r = T3B.reshape(chi,chi,D_bond,D_bond)
        T1A_r = T1A.reshape(chi,chi,D_bond,D_bond)

        D2 = chi * D_bond * D_bond
        _bop = lambda einstr, *args, opt: oe.contract(einstr, *args,
                                                       optimize=opt, backend="torch")
        _open_A = _bop("YX,MYar,abci,rstj->MbsXctij", C21EB, T1D_r, a, a.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
        closed_A = oe.contract("MXii->MX", _open_A, backend="torch"); del _open_A
        _open_B = _bop("MYct,abci,rstj->YarMbsij",    T3E_r, b, b.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
        closed_B = oe.contract("YMii->YM", _open_B, backend="torch"); del _open_B
        _open_C = _bop("ZY,NZbs,abci,rstj->NctYarij", C32AD, T2F_r, c, c.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
        closed_C = oe.contract("NYii->NY", _open_C, backend="torch"); del _open_C
        _open_D = _bop("NZar,abci,rstj->ZbsNctij",    T1A_r, d, d.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
        closed_D = oe.contract("ZNii->ZN", _open_D, backend="torch"); del _open_D
        _open_E = _bop("XZ,LXct,abci,rstj->LarZbsij", C13CF, T3B_r, e, e.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
        closed_E = oe.contract("LZii->LZ", _open_E, backend="torch"); del _open_E
        _open_F = _bop("LXbs,abci,rstj->XctLarij",    T2C_r, f, f.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
        closed_F = oe.contract("XLii->XL", _open_F, backend="torch"); del _open_F

        # ── NN bond closures (pair products computed on-the-fly inside _ckpt) ──
        def _nn_CB(cA, cF, cE, cD, SdotS):
            _oC = _bop("ZY,NZbs,abci,rstj->NctYarij", C32AD, T2F_r, c, c.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
            _oB = _bop("MYct,abci,rstj->YarMbsij",    T3E_r, b, b.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
            return _compute_nn_bond_energy(_oC, _oB, torch.mm(cA, cF), torch.mm(cE, cD), SdotS)
        def _nn_AF(cE, cD, cC, cB, SdotS):
            _oA = _bop("YX,MYar,abci,rstj->MbsXctij", C21EB, T1D_r, a, a.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
            _oF = _bop("LXbs,abci,rstj->XctLarij",    T2C_r, f, f.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
            return _compute_nn_bond_energy(_oA, _oF, torch.mm(cE, cD), torch.mm(cC, cB), SdotS)
        def _nn_ED(cC, cB, cA, cF, SdotS):
            _oE = _bop("XZ,LXct,abci,rstj->LarZbsij", C13CF, T3B_r, e, e.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
            _oD = _bop("NZar,abci,rstj->ZbsNctij",    T1A_r, d, d.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
            return _compute_nn_bond_energy(_oE, _oD, torch.mm(cC, cB), torch.mm(cA, cF), SdotS)
        def _nn_DC(cB, cA, cF, cE, SdotS):
            _oD = _bop("NZar,abci,rstj->ZbsNctij",    T1A_r, d, d.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
            _oC = _bop("ZY,NZbs,abci,rstj->NctYarij", C32AD, T2F_r, c, c.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
            return _compute_nn_bond_energy(_oD, _oC, torch.mm(cB, cA), torch.mm(cF, cE), SdotS)
        def _nn_BA(cF, cE, cD, cC, SdotS):
            _oB = _bop("MYct,abci,rstj->YarMbsij",    T3E_r, b, b.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
            _oA = _bop("YX,MYar,abci,rstj->MbsXctij", C21EB, T1D_r, a, a.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
            return _compute_nn_bond_energy(_oB, _oA, torch.mm(cF, cE), torch.mm(cD, cC), SdotS)
        def _nn_FE(cD, cC, cB, cA, SdotS):
            _oF = _bop("LXbs,abci,rstj->XctLarij",    T2C_r, f, f.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
            _oE = _bop("XZ,LXct,abci,rstj->LarZbsij", C13CF, T3B_r, e, e.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
            return _compute_nn_bond_energy(_oF, _oE, torch.mm(cD, cC), torch.mm(cB, cA), SdotS)

        # ── NNN bond closures (sequential open builds: peak = 2·open+W, not 4·open) ──
        def _nnn_CA(cB, cF, cE, cD, SdotS):
            return _compute_nnn_bond_energy_seq(
                lambda: _bop("ZY,NZbs,abci,rstj->NctYarij", C32AD, T2F_r, c, c.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS),
                cB,
                lambda: _bop("YX,MYar,abci,rstj->MbsXctij", C21EB, T1D_r, a, a.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS),
                cF, torch.mm(cE, cD), SdotS)
        def _nnn_AE(cF, cD, cC, cB, SdotS):
            return _compute_nnn_bond_energy_seq(
                lambda: _bop("YX,MYar,abci,rstj->MbsXctij", C21EB, T1D_r, a, a.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS),
                cF,
                lambda: _bop("XZ,LXct,abci,rstj->LarZbsij", C13CF, T3B_r, e, e.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS),
                cD, torch.mm(cC, cB), SdotS)
        def _nnn_EC(cD, cB, cA, cF, SdotS):
            return _compute_nnn_bond_energy_seq(
                lambda: _bop("XZ,LXct,abci,rstj->LarZbsij", C13CF, T3B_r, e, e.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS),
                cD,
                lambda: _bop("ZY,NZbs,abci,rstj->NctYarij", C32AD, T2F_r, c, c.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS),
                cB, torch.mm(cA, cF), SdotS)
        def _nnn_BF(cA, cE, cD, cC, SdotS):
            return _compute_nnn_bond_energy_seq(
                lambda: _bop("MYct,abci,rstj->YarMbsij",    T3E_r, b, b.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS),
                cA,
                lambda: _bop("LXbs,abci,rstj->XctLarij",    T2C_r, f, f.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS),
                cE, torch.mm(cD, cC), SdotS)
        def _nnn_FD(cE, cC, cB, cA, SdotS):
            return _compute_nnn_bond_energy_seq(
                lambda: _bop("LXbs,abci,rstj->XctLarij",    T2C_r, f, f.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS),
                cE,
                lambda: _bop("NZar,abci,rstj->ZbsNctij",    T1A_r, d, d.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS),
                cC, torch.mm(cB, cA), SdotS)
        def _nnn_DB(cC, cA, cF, cE, SdotS):
            return _compute_nnn_bond_energy_seq(
                lambda: _bop("NZar,abci,rstj->ZbsNctij",    T1A_r, d, d.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS),
                cC,
                lambda: _bop("MYct,abci,rstj->YarMbsij",    T3E_r, b, b.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS),
                cA, torch.mm(cF, cE), SdotS)

        E_CB = Jcb * _ckpt(_nn_CB, closed_A, closed_F, closed_E, closed_D, SdotS, use_reentrant=False)
        E_AF = Jaf * _ckpt(_nn_AF, closed_E, closed_D, closed_C, closed_B, SdotS, use_reentrant=False)
        E_ED = Jed * _ckpt(_nn_ED, closed_C, closed_B, closed_A, closed_F, SdotS, use_reentrant=False)
        E_DC = Jdc * _ckpt(_nn_DC, closed_B, closed_A, closed_F, closed_E, SdotS, use_reentrant=False)
        E_BA = Jba * _ckpt(_nn_BA, closed_F, closed_E, closed_D, closed_C, SdotS, use_reentrant=False)
        E_FE = Jfe * _ckpt(_nn_FE, closed_D, closed_C, closed_B, closed_A, SdotS, use_reentrant=False)

        E_CA = Jca * _ckpt(_nnn_CA, closed_B, closed_F, closed_E, closed_D, SdotS, use_reentrant=False)
        E_AE = Jae * _ckpt(_nnn_AE, closed_F, closed_D, closed_C, closed_B, SdotS, use_reentrant=False)
        E_EC = Jec * _ckpt(_nnn_EC, closed_D, closed_B, closed_A, closed_F, SdotS, use_reentrant=False)
        E_BF = Jbf * _ckpt(_nnn_BF, closed_A, closed_E, closed_D, closed_C, SdotS, use_reentrant=False)
        E_FD = Jfd * _ckpt(_nnn_FD, closed_E, closed_C, closed_B, closed_A, SdotS, use_reentrant=False)
        E_DB = Jdb * _ckpt(_nnn_DB, closed_C, closed_A, closed_F, closed_E, SdotS, use_reentrant=False)

        return torch.real((E_AF+E_CB+E_ED + E_DC+E_BA+E_FE)*0.5 +E_CA+E_AE+E_EC +E_BF+E_FD+E_DB)

    # ── Pre-build path: CPU (any D) or GPU D ≤ 7 ────────────────────────
    o, cl = build_open_closed_env2(a, b, c, d, e, f, chi, D_bond, d_PHYS,
                                   C21EB, C32AD, C13CF,
                                   T1D, T2C, T2F, T3E, T3B, T1A)
    CB = torch.mm(cl['C'], cl['B']);  ED = torch.mm(cl['E'], cl['D'])
    AF = torch.mm(cl['A'], cl['F']);  DC = torch.mm(cl['D'], cl['C'])
    BA = torch.mm(cl['B'], cl['A']);  FE = torch.mm(cl['F'], cl['E'])

    if _on_gpu:
        E_CB = Jcb * _ckpt(_compute_nn_bond_energy,  o['C'], o['B'], AF, ED, SdotS, use_reentrant=False)
        E_AF = Jaf * _ckpt(_compute_nn_bond_energy,  o['A'], o['F'], ED, CB, SdotS, use_reentrant=False)
        E_ED = Jed * _ckpt(_compute_nn_bond_energy,  o['E'], o['D'], CB, AF, SdotS, use_reentrant=False)
        E_DC = Jdc * _ckpt(_compute_nn_bond_energy,  o['D'], o['C'], BA, FE, SdotS, use_reentrant=False)
        E_BA = Jba * _ckpt(_compute_nn_bond_energy,  o['B'], o['A'], FE, DC, SdotS, use_reentrant=False)
        E_FE = Jfe * _ckpt(_compute_nn_bond_energy,  o['F'], o['E'], DC, BA, SdotS, use_reentrant=False)
        E_CA = Jca * _ckpt(_compute_nnn_bond_energy, o['C'], cl['B'], o['A'], cl['F'], ED, SdotS, use_reentrant=False)
        E_AE = Jae * _ckpt(_compute_nnn_bond_energy, o['A'], cl['F'], o['E'], cl['D'], CB, SdotS, use_reentrant=False)
        E_EC = Jec * _ckpt(_compute_nnn_bond_energy, o['E'], cl['D'], o['C'], cl['B'], AF, SdotS, use_reentrant=False)
        E_BF = Jbf * _ckpt(_compute_nnn_bond_energy, o['B'], cl['A'], o['F'], cl['E'], DC, SdotS, use_reentrant=False)
        E_FD = Jfd * _ckpt(_compute_nnn_bond_energy, o['F'], cl['E'], o['D'], cl['C'], BA, SdotS, use_reentrant=False)
        E_DB = Jdb * _ckpt(_compute_nnn_bond_energy, o['D'], cl['C'], o['B'], cl['A'], FE, SdotS, use_reentrant=False)
    else:
        E_CB = Jcb * _compute_nn_bond_energy(o['C'], o['B'], AF, ED, SdotS)
        E_AF = Jaf * _compute_nn_bond_energy(o['A'], o['F'], ED, CB, SdotS)
        E_ED = Jed * _compute_nn_bond_energy(o['E'], o['D'], CB, AF, SdotS)
        E_DC = Jdc * _compute_nn_bond_energy(o['D'], o['C'], BA, FE, SdotS)
        E_BA = Jba * _compute_nn_bond_energy(o['B'], o['A'], FE, DC, SdotS)
        E_FE = Jfe * _compute_nn_bond_energy(o['F'], o['E'], DC, BA, SdotS)
        E_CA = Jca * _compute_nnn_bond_energy(o['C'], cl['B'], o['A'], cl['F'], ED, SdotS)
        E_AE = Jae * _compute_nnn_bond_energy(o['A'], cl['F'], o['E'], cl['D'], CB, SdotS)
        E_EC = Jec * _compute_nnn_bond_energy(o['E'], cl['D'], o['C'], cl['B'], AF, SdotS)
        E_BF = Jbf * _compute_nnn_bond_energy(o['B'], cl['A'], o['F'], cl['E'], DC, SdotS)
        E_FD = Jfd * _compute_nnn_bond_energy(o['F'], cl['E'], o['D'], cl['C'], BA, SdotS)
        E_DB = Jdb * _compute_nnn_bond_energy(o['D'], cl['C'], o['B'], cl['A'], FE, SdotS)

    return torch.real((E_AF+E_CB+E_ED + E_DC+E_BA+E_FE)*0.5 +E_CA+E_AE+E_EC +E_BF+E_FD+E_DB)




def energy_expectation_nearest_neighbor_other_3_bonds(a,b,c,d,e,f, 
                    Jcd,Jef,Jab,
                    Jbe,Jfc,Jda,
                    Jec,Jca,Jae,Jfd,Jdb,Jbf,
                    SdotS,
                    chi, D_bond, d_PHYS,
                    C21AF,C32CB,C13ED,T1B,T2E,T2D,T3A,T3F,T1C):
    """
    Compute the variational energy expectation value for the remaining 3 bonds
    in the type-3 environment.

    This function covers bonds C-D, E-F, A-B (the bonds not handled by
    ``energy_expectation_nearest_neighbor_6_bonds``).  The same open/closed
    tensor construction is used, but the environment here is the type-3 set
    ``(C21AF, C32CB, C13ED, T1B, T2E, T2D, T3A, T3F, T1C)``.

    The energy is::

        E_3 = (E_EF + E_AB + E_CD) / norm_3rd_env

    where ``norm_3rd_env = Tr[EF * CD * AB]``.

    Args:
        a, b, c, d, e, f (torch.Tensor): Single-layer site tensors.
        Jcd, Jef, Jab (float): Coupling constants for the three nn bonds.
        Jec, Jca, Jae, Jfd, Jdb, Jbf (float): Coupling constants for the
            six nnn bonds.
        SdotS (torch.Tensor): Unit spin-spin operator S_i·S_j,
            shape ``(d_PHYS, d_PHYS, d_PHYS, d_PHYS)``.
        chi (int): Environment bond dimension.
        D_bond (int): Virtual bond dimension.
        C21AF, C32CB, C13ED (torch.Tensor): Type-3 corner matrices,
            shape ``(chi, chi)``.
        T1B, T2E, T2D, T3A, T3F, T1C (torch.Tensor): Type-3 transfer tensors.

    Returns:
        torch.Tensor: Scalar energy per bond (float32).
    """
    _on_gpu = DEVICE.type == 'cuda'

    # ── GPU D≥8: lazy-open closures + _ckpt ──────────────────────────────
    if _on_gpu and D_bond >= 8:
        T1B_r = T1B.reshape(chi,chi,D_bond,D_bond)
        T2E_r = T2E.reshape(chi,chi,D_bond,D_bond)
        T2D_r = T2D.reshape(chi,chi,D_bond,D_bond)
        T3A_r = T3A.reshape(chi,chi,D_bond,D_bond)
        T3F_r = T3F.reshape(chi,chi,D_bond,D_bond)
        T1C_r = T1C.reshape(chi,chi,D_bond,D_bond)

        D2 = chi * D_bond * D_bond
        _bop = lambda einstr, *args, opt: oe.contract(einstr, *args,
                                                       optimize=opt, backend="torch")
        _open_C = _bop("YX,MYar,abci,rstj->MbsXctij", C21AF, T1B_r, c, c.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
        closed_C = oe.contract("MXii->MX", _open_C, backend="torch"); del _open_C
        _open_F = _bop("MYct,abci,rstj->YarMbsij",    T3A_r, f, f.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
        closed_F = oe.contract("YMii->YM", _open_F, backend="torch"); del _open_F
        _open_E = _bop("ZY,NZbs,abci,rstj->NctYarij", C32CB, T2D_r, e, e.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
        closed_E = oe.contract("NYii->NY", _open_E, backend="torch"); del _open_E
        _open_B = _bop("NZar,abci,rstj->ZbsNctij",    T1C_r, b, b.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
        closed_B = oe.contract("ZNii->ZN", _open_B, backend="torch"); del _open_B
        _open_A = _bop("XZ,LXct,abci,rstj->LarZbsij", C13ED, T3F_r, a, a.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
        closed_A = oe.contract("LZii->LZ", _open_A, backend="torch"); del _open_A
        _open_D = _bop("LXbs,abci,rstj->XctLarij",    T2E_r, d, d.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
        closed_D = oe.contract("XLii->XL", _open_D, backend="torch"); del _open_D

        # ── NN closures (pair products computed on-the-fly inside _ckpt) ──
        def _nn_EF(cC, cD, cA, cB, SdotS):
            _oE = _bop("ZY,NZbs,abci,rstj->NctYarij", C32CB, T2D_r, e, e.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
            _oF = _bop("MYct,abci,rstj->YarMbsij",    T3A_r, f, f.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
            return _compute_nn_bond_energy(_oE, _oF, torch.mm(cC, cD), torch.mm(cA, cB), SdotS)
        def _nn_AB(cE, cF, cC, cD, SdotS):
            _oA = _bop("XZ,LXct,abci,rstj->LarZbsij", C13ED, T3F_r, a, a.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
            _oB = _bop("NZar,abci,rstj->ZbsNctij",    T1C_r, b, b.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
            return _compute_nn_bond_energy(_oA, _oB, torch.mm(cE, cF), torch.mm(cC, cD), SdotS)
        def _nn_CD(cA, cB, cE, cF, SdotS):
            _oC = _bop("YX,MYar,abci,rstj->MbsXctij", C21AF, T1B_r, c, c.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
            _oD = _bop("LXbs,abci,rstj->XctLarij",    T2E_r, d, d.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
            return _compute_nn_bond_energy(_oC, _oD, torch.mm(cA, cB), torch.mm(cE, cF), SdotS)
        def _nn_BE(cF, cC, cD, cA, SdotS):
            _oB = _bop("NZar,abci,rstj->ZbsNctij",    T1C_r, b, b.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
            _oE = _bop("ZY,NZbs,abci,rstj->NctYarij", C32CB, T2D_r, e, e.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
            return _compute_nn_bond_energy(_oB, _oE, torch.mm(cF, cC), torch.mm(cD, cA), SdotS)
        def _nn_FC(cD, cA, cB, cE, SdotS):
            _oF = _bop("MYct,abci,rstj->YarMbsij",    T3A_r, f, f.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
            _oC = _bop("YX,MYar,abci,rstj->MbsXctij", C21AF, T1B_r, c, c.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
            return _compute_nn_bond_energy(_oF, _oC, torch.mm(cD, cA), torch.mm(cB, cE), SdotS)
        def _nn_DA(cB, cE, cF, cC, SdotS):
            _oD = _bop("LXbs,abci,rstj->XctLarij",    T2E_r, d, d.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
            _oA = _bop("XZ,LXct,abci,rstj->LarZbsij", C13ED, T3F_r, a, a.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
            return _compute_nn_bond_energy(_oD, _oA, torch.mm(cB, cE), torch.mm(cF, cC), SdotS)

        # ── NNN closures (sequential open builds: peak = 2·open+W, not 4·open) ─────
        def _nnn_EC(cF, cD, cA, cB, SdotS):
            return _compute_nnn_bond_energy_seq(
                lambda: _bop("ZY,NZbs,abci,rstj->NctYarij", C32CB, T2D_r, e, e.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS),
                cF,
                lambda: _bop("YX,MYar,abci,rstj->MbsXctij", C21AF, T1B_r, c, c.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS),
                cD, torch.mm(cA, cB), SdotS)
        def _nnn_CA(cD, cB, cE, cF, SdotS):
            return _compute_nnn_bond_energy_seq(
                lambda: _bop("YX,MYar,abci,rstj->MbsXctij", C21AF, T1B_r, c, c.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS),
                cD,
                lambda: _bop("XZ,LXct,abci,rstj->LarZbsij", C13ED, T3F_r, a, a.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS),
                cB, torch.mm(cE, cF), SdotS)
        def _nnn_AE(cB, cF, cC, cD, SdotS):
            return _compute_nnn_bond_energy_seq(
                lambda: _bop("XZ,LXct,abci,rstj->LarZbsij", C13ED, T3F_r, a, a.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS),
                cB,
                lambda: _bop("ZY,NZbs,abci,rstj->NctYarij", C32CB, T2D_r, e, e.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS),
                cF, torch.mm(cC, cD), SdotS)
        def _nnn_FD(cC, cA, cB, cE, SdotS):
            return _compute_nnn_bond_energy_seq(
                lambda: _bop("MYct,abci,rstj->YarMbsij",    T3A_r, f, f.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS),
                cC,
                lambda: _bop("LXbs,abci,rstj->XctLarij",    T2E_r, d, d.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS),
                cA, torch.mm(cB, cE), SdotS)
        def _nnn_DB(cA, cE, cF, cC, SdotS):
            return _compute_nnn_bond_energy_seq(
                lambda: _bop("LXbs,abci,rstj->XctLarij",    T2E_r, d, d.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS),
                cA,
                lambda: _bop("NZar,abci,rstj->ZbsNctij",    T1C_r, b, b.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS),
                cE, torch.mm(cF, cC), SdotS)
        def _nnn_BF(cE, cC, cD, cA, SdotS):
            return _compute_nnn_bond_energy_seq(
                lambda: _bop("NZar,abci,rstj->ZbsNctij",    T1C_r, b, b.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS),
                cE,
                lambda: _bop("MYct,abci,rstj->YarMbsij",    T3A_r, f, f.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS),
                cC, torch.mm(cD, cA), SdotS)

        E_EF = Jef * _ckpt(_nn_EF, closed_C, closed_D, closed_A, closed_B, SdotS, use_reentrant=False)
        E_AB = Jab * _ckpt(_nn_AB, closed_E, closed_F, closed_C, closed_D, SdotS, use_reentrant=False)
        E_CD = Jcd * _ckpt(_nn_CD, closed_A, closed_B, closed_E, closed_F, SdotS, use_reentrant=False)
        E_BE = Jbe * _ckpt(_nn_BE, closed_F, closed_C, closed_D, closed_A, SdotS, use_reentrant=False)
        E_FC = Jfc * _ckpt(_nn_FC, closed_D, closed_A, closed_B, closed_E, SdotS, use_reentrant=False)
        E_DA = Jda * _ckpt(_nn_DA, closed_B, closed_E, closed_F, closed_C, SdotS, use_reentrant=False)

        E_EC = Jec * _ckpt(_nnn_EC, closed_F, closed_D, closed_A, closed_B, SdotS, use_reentrant=False)
        E_CA = Jca * _ckpt(_nnn_CA, closed_D, closed_B, closed_E, closed_F, SdotS, use_reentrant=False)
        E_AE = Jae * _ckpt(_nnn_AE, closed_B, closed_F, closed_C, closed_D, SdotS, use_reentrant=False)
        E_FD = Jfd * _ckpt(_nnn_FD, closed_C, closed_A, closed_B, closed_E, SdotS, use_reentrant=False)
        E_DB = Jdb * _ckpt(_nnn_DB, closed_A, closed_E, closed_F, closed_C, SdotS, use_reentrant=False)
        E_BF = Jbf * _ckpt(_nnn_BF, closed_E, closed_C, closed_D, closed_A, SdotS, use_reentrant=False)

        return torch.real((E_EF+E_AB+E_CD + E_BE+E_FC+E_DA)*0.5 +E_EC+E_CA+E_AE +E_FD+E_DB+E_BF)

    # ── Pre-build path: CPU (any D) or GPU D ≤ 7 ─────────────────────────
    o, cl = build_open_closed_env3(a, b, c, d, e, f, chi, D_bond, d_PHYS,
                                   C21AF, C32CB, C13ED,
                                   T1B, T2E, T2D, T3A, T3F, T1C)
    EF = torch.mm(cl['E'], cl['F']);  AB = torch.mm(cl['A'], cl['B'])
    CD = torch.mm(cl['C'], cl['D']);  BE = torch.mm(cl['B'], cl['E'])
    FC = torch.mm(cl['F'], cl['C']);  DA = torch.mm(cl['D'], cl['A'])

    if _on_gpu:
        E_EF = Jef * _ckpt(_compute_nn_bond_energy,  o['E'], o['F'], CD, AB, SdotS, use_reentrant=False)
        E_AB = Jab * _ckpt(_compute_nn_bond_energy,  o['A'], o['B'], EF, CD, SdotS, use_reentrant=False)
        E_CD = Jcd * _ckpt(_compute_nn_bond_energy,  o['C'], o['D'], AB, EF, SdotS, use_reentrant=False)
        E_BE = Jbe * _ckpt(_compute_nn_bond_energy,  o['B'], o['E'], FC, DA, SdotS, use_reentrant=False)
        E_FC = Jfc * _ckpt(_compute_nn_bond_energy,  o['F'], o['C'], DA, BE, SdotS, use_reentrant=False)
        E_DA = Jda * _ckpt(_compute_nn_bond_energy,  o['D'], o['A'], BE, FC, SdotS, use_reentrant=False)
        E_EC = Jec * _ckpt(_compute_nnn_bond_energy, o['E'], cl['F'], o['C'], cl['D'], AB, SdotS, use_reentrant=False)
        E_CA = Jca * _ckpt(_compute_nnn_bond_energy, o['C'], cl['D'], o['A'], cl['B'], EF, SdotS, use_reentrant=False)
        E_AE = Jae * _ckpt(_compute_nnn_bond_energy, o['A'], cl['B'], o['E'], cl['F'], CD, SdotS, use_reentrant=False)
        E_FD = Jfd * _ckpt(_compute_nnn_bond_energy, o['F'], cl['C'], o['D'], cl['A'], BE, SdotS, use_reentrant=False)
        E_DB = Jdb * _ckpt(_compute_nnn_bond_energy, o['D'], cl['A'], o['B'], cl['E'], FC, SdotS, use_reentrant=False)
        E_BF = Jbf * _ckpt(_compute_nnn_bond_energy, o['B'], cl['E'], o['F'], cl['C'], DA, SdotS, use_reentrant=False)
    else:
        E_EF = Jef * _compute_nn_bond_energy(o['E'], o['F'], CD, AB, SdotS)
        E_AB = Jab * _compute_nn_bond_energy(o['A'], o['B'], EF, CD, SdotS)
        E_CD = Jcd * _compute_nn_bond_energy(o['C'], o['D'], AB, EF, SdotS)
        E_BE = Jbe * _compute_nn_bond_energy(o['B'], o['E'], FC, DA, SdotS)
        E_FC = Jfc * _compute_nn_bond_energy(o['F'], o['C'], DA, BE, SdotS)
        E_DA = Jda * _compute_nn_bond_energy(o['D'], o['A'], BE, FC, SdotS)
        E_EC = Jec * _compute_nnn_bond_energy(o['E'], cl['F'], o['C'], cl['D'], AB, SdotS)
        E_CA = Jca * _compute_nnn_bond_energy(o['C'], cl['D'], o['A'], cl['B'], EF, SdotS)
        E_AE = Jae * _compute_nnn_bond_energy(o['A'], cl['B'], o['E'], cl['F'], CD, SdotS)
        E_FD = Jfd * _compute_nnn_bond_energy(o['F'], cl['C'], o['D'], cl['A'], BE, SdotS)
        E_DB = Jdb * _compute_nnn_bond_energy(o['D'], cl['A'], o['B'], cl['E'], FC, SdotS)
        E_BF = Jbf * _compute_nnn_bond_energy(o['B'], cl['E'], o['F'], cl['C'], DA, SdotS)

    return torch.real((E_EF+E_AB+E_CD + E_BE+E_FC+E_DA)*0.5 +E_EC+E_CA+E_AE +E_FD+E_DB+E_BF)


# ── Observable helpers (used by evaluate_observables in the driver) ───────────

def _psd_normalize_rho(rho_2d, d_PHYS):
    """Hermitianize + PSD-project + normalize a d²×d² density matrix → (d,d,d,d)."""
    rho_2d = (rho_2d + rho_2d.conj().T) / 2.0
    eigvals, eigvecs = torch.linalg.eigh(rho_2d)
    eigvals_clipped = torch.clamp(eigvals, min=0.0)
    rho_2d = (eigvecs * eigvals_clipped) @ eigvecs.conj().T
    trace = eigvals_clipped.sum().clamp(min=1e-30)
    return (rho_2d / trace).reshape(d_PHYS, d_PHYS, d_PHYS, d_PHYS)


def _build_nn_rho(open_X, open_Y, pair1, pair2, d_PHYS):
    """Build normalized nn 2-site rho from two open tensors + pair products.

    Memory-efficient: avoids the (d,d,d,d,D2,D2) = 16·D2² intermediate.
    Old peak: 2·open + 16·D2².  New peak: 3·open = 3·D2²·d².
    For D=11, chi=121: 41.2 GB → 20.6 GB.

    rho[ij,kl] = Σ_{N,Y,M} open_X[N,Y,i,j] · open_Y[Y,M,k,l] · (pair1@pair2)[M,N]
    Path:
      closure[M,N] = (pair1 @ pair2)[M,N]                      — (D2,D2), negligible
      W[M,Y,i,j]   = Σ_N closure[M,N] · open_X[N,Y,i,j]       — (D2,D2,d,d)
      rho[i,j,k,l] = Σ_{Y,M} W[M,Y,i,j] · open_Y[Y,M,k,l]    — (d,d,d,d) tiny
    """
    closure = torch.mm(pair1, pair2)                                           # (D2, D2)
    W   = oe.contract("MN,NYij->MYij", closure, open_X, backend="torch")      # (D2,D2,d,d)
    rho = oe.contract("MYij,YMkl->ikjl", W, open_Y, backend="torch")          # (d,d,d,d) interleaved: row=(i,k), col=(j,l)
    return _psd_normalize_rho(rho.reshape(d_PHYS*d_PHYS, d_PHYS*d_PHYS), d_PHYS)


def _build_nnn_rho(open_X, closed_Y, open_Z, closed_W, large_pair, d_PHYS):
    """Build normalized nnn 2-site rho from open-closed-open + closure.

    Memory-efficient: avoids the (d,d,d,d,D2,D2) = 16·D2² intermediate.
    Old peak: 2·open + 16·D2².  Current peak: 4·D2²·d² (open_X + W + open_Z + V
    simultaneously, since function parameters cannot be freed mid-call).
    Use _build_nnn_rho_seq with callable builders to achieve a true 2-open peak.

    rho[ij,kl] = Σ_{N,A,E,X} open_X[N,A,i,j]·closed_Y[A,E]·open_Z[E,X,k,l]·cl2[X,N]
    where cl2 = closed_W @ large_pair.
    Path:
      cl2[X,N]     = (closed_W @ large_pair)[X,N]               — (D2,D2), negligible
      W[N,E,i,j]   = Σ_A open_X[N,A,i,j] · closed_Y[A,E]       — (D2,D2,d,d)
      V[E,N,k,l]   = Σ_X open_Z[E,X,k,l] · cl2[X,N]            — (D2,D2,d,d)
      rho[i,j,k,l] = Σ_{N,E} W[N,E,i,j] · V[E,N,k,l]           — (d,d,d,d) tiny
    """
    cl2 = torch.mm(closed_W, large_pair)                                       # (D2, D2)
    W   = oe.contract("NAij,AE->NEij", open_X, closed_Y, backend="torch")     # (D2,D2,d,d)
    V   = oe.contract("EXkl,XN->ENkl", open_Z, cl2, backend="torch")          # (D2,D2,d,d)
    rho = oe.contract("NEij,ENkl->ikjl", W, V, backend="torch")               # (d,d,d,d) interleaved: row=(i,k), col=(j,l)
    return _psd_normalize_rho(rho.reshape(d_PHYS*d_PHYS, d_PHYS*d_PHYS), d_PHYS)


def _build_nnn_rho_seq(build_open_X, closed_Y, build_open_Z, closed_W, large_pair, d_PHYS):
    """Sequential NNN rho: build open_X, compute W, free open_X, then build open_Z.

    Peak memory = 2·open + W  (= 3·D2²·d²) instead of 4·D2²·d² from _build_nnn_rho.
    For D=11, chi=121: peak drops from ~37.6 GB to ~29.4 GB, fitting in 32 GB.

    Args:
        build_open_X: callable () → (D2,D2,d,d) tensor for site X
        closed_Y:      (D2,D2) closed tensor for intervening site Y
        build_open_Z: callable () → (D2,D2,d,d) tensor for site Z
        closed_W:      (D2,D2) closed tensor for the outer-pair product
        large_pair:    (D2,D2) large pair-product tensor
        d_PHYS:        physical dimension d
    """
    cl2   = torch.mm(closed_W, large_pair)                                     # (D2,D2) negligible
    oX    = build_open_X()                                                     # (D2,D2,d,d)
    W     = oe.contract("NAij,AE->NEij", oX, closed_Y, backend="torch")       # (D2,D2,d,d)
    del oX                                                                     # free before building oZ
    oZ    = build_open_Z()                                                     # (D2,D2,d,d)
    V     = oe.contract("EXkl,XN->ENkl", oZ, cl2, backend="torch")            # (D2,D2,d,d)
    del oZ
    rho   = oe.contract("NEij,ENkl->ikjl", W, V, backend="torch")             # (d,d,d,d) tiny
    del W, V
    return _psd_normalize_rho(rho.reshape(d_PHYS*d_PHYS, d_PHYS*d_PHYS), d_PHYS)


def _compute_nnn_bond_energy_seq(build_open_X, closed_Y, build_open_Z, closed_W,
                                  large_pair, SdotS):
    """NNN bond energy using sequential open builds (peak = 2·open+W, not 4·open).

    Designed as a sub-checkpoint target for the GPU D≥8 energy-function path.
    Infers d_PHYS from the first open tensor after it is built.
    """
    cl2   = torch.mm(closed_W, large_pair)
    oX    = build_open_X()
    d_PHYS = oX.shape[2]
    W     = oe.contract("NAij,AE->NEij", oX, closed_Y, backend="torch")
    del oX
    oZ    = build_open_Z()
    V     = oe.contract("EXkl,XN->ENkl", oZ, cl2, backend="torch")
    del oZ
    rho   = oe.contract("NEij,ENkl->ikjl", W, V, backend="torch")
    del W, V
    rho_nrm = _psd_normalize_rho(rho.reshape(d_PHYS*d_PHYS, d_PHYS*d_PHYS), d_PHYS)
    return oe.contract("ikjl,ijkl->", rho_nrm, SdotS, backend="torch")


def _compute_nn_bond_energy(open_X, open_Y, pair1, pair2, SdotS):
    """Compute PSD-normalized nn bond energy  tr(rho · SdotS).

    Designed as a sub-checkpoint target: takes ONLY tensor args so
    ``_ckpt(_compute_nn_bond_energy, ..., use_reentrant=False)`` works.
    The big (d²,d²,chiD²,chiD²) rho intermediate is created, used, and
    freed WITHIN this call — it never leaks into the outer autograd graph.

    Memory: peak = one rho tensor  (≈ 800 MB at D=8, chi=40).
    Without sub-checkpointing, 12 rho tensors accumulate → 9.6 GB.
    """
    d_PHYS = open_X.shape[2]
    rho = _build_nn_rho(open_X, open_Y, pair1, pair2, d_PHYS)
    return oe.contract("ikjl,ijkl->", rho, SdotS, backend="torch")


def _compute_nnn_bond_energy(open_X, closed_Y, open_Z, closed_W,
                             large_pair, SdotS):
    """Compute PSD-normalized nnn bond energy  tr(rho · SdotS).

    Same sub-checkpoint rationale as ``_compute_nn_bond_energy``.
    """
    d_PHYS = open_X.shape[2]
    rho = _build_nnn_rho(open_X, closed_Y, open_Z, closed_W,
                         large_pair, d_PHYS)
    return oe.contract("ikjl,ijkl->", rho, SdotS, backend="torch")


def build_open_closed_env1(a, b, c, d, e, f, chi, D_bond, d_PHYS,
                           C21CD, C32EF, C13AB, T1F, T2A, T2B, T3C, T3D, T1E):
    """Construct open/closed tensors for environment 1 (ebadcf).

    Returns (opens, closeds) — dicts keyed by site label ('A'..'F').
    """
    T1F = T1F.reshape(chi,chi,D_bond,D_bond)
    T2A = T2A.reshape(chi,chi,D_bond,D_bond)
    T2B = T2B.reshape(chi,chi,D_bond,D_bond)
    T3C = T3C.reshape(chi,chi,D_bond,D_bond)
    T3D = T3D.reshape(chi,chi,D_bond,D_bond)
    T1E = T1E.reshape(chi,chi,D_bond,D_bond)
    D2 = chi*D_bond*D_bond
    o = {}
    o['E'] = oe.contract("YX,MYar,abci,rstj->MbsXctij", C21CD, T1F, e, e.conj(), optimize=[(0,1),(0,1),(0,1)], backend="torch").reshape(D2,D2,d_PHYS,d_PHYS)
    o['D'] = oe.contract("MYct,abci,rstj->YarMbsij", T3C, d, d.conj(), optimize=[(0,1),(0,1)], backend="torch").reshape(D2,D2,d_PHYS,d_PHYS)
    o['A'] = oe.contract("ZY,NZbs,abci,rstj->NctYarij", C32EF, T2B, a, a.conj(), optimize=[(0,1),(0,1),(0,1)], backend="torch").reshape(D2,D2,d_PHYS,d_PHYS)
    o['F'] = oe.contract("NZar,abci,rstj->ZbsNctij", T1E, f, f.conj(), optimize=[(0,1),(0,1)], backend="torch").reshape(D2,D2,d_PHYS,d_PHYS)
    o['C'] = oe.contract("XZ,LXct,abci,rstj->LarZbsij", C13AB, T3D, c, c.conj(), optimize=[(0,1),(0,1),(0,1)], backend="torch").reshape(D2,D2,d_PHYS,d_PHYS)
    o['B'] = oe.contract("LXbs,abci,rstj->XctLarij", T2A, b, b.conj(), optimize=[(0,1),(0,1)], backend="torch").reshape(D2,D2,d_PHYS,d_PHYS)
    cl = {k: oe.contract("ABii->AB", o[k], backend="torch") for k in o}
    return o, cl


def build_open_closed_env2(a, b, c, d, e, f, chi, D_bond, d_PHYS,
                           C21EB, C32AD, C13CF, T1D, T2C, T2F, T3E, T3B, T1A):
    """Construct open/closed tensors for environment 2 (afcbed)."""
    T1D = T1D.reshape(chi,chi,D_bond,D_bond)
    T2C = T2C.reshape(chi,chi,D_bond,D_bond)
    T2F = T2F.reshape(chi,chi,D_bond,D_bond)
    T3E = T3E.reshape(chi,chi,D_bond,D_bond)
    T3B = T3B.reshape(chi,chi,D_bond,D_bond)
    T1A = T1A.reshape(chi,chi,D_bond,D_bond)
    D2 = chi*D_bond*D_bond
    o = {}
    o['A'] = oe.contract("YX,MYar,abci,rstj->MbsXctij", C21EB, T1D, a, a.conj(), optimize=[(0,1),(0,1),(0,1)], backend="torch").reshape(D2,D2,d_PHYS,d_PHYS)
    o['B'] = oe.contract("MYct,abci,rstj->YarMbsij", T3E, b, b.conj(), optimize=[(0,1),(0,1)], backend="torch").reshape(D2,D2,d_PHYS,d_PHYS)
    o['C'] = oe.contract("ZY,NZbs,abci,rstj->NctYarij", C32AD, T2F, c, c.conj(), optimize=[(0,1),(0,1),(0,1)], backend="torch").reshape(D2,D2,d_PHYS,d_PHYS)
    o['D'] = oe.contract("NZar,abci,rstj->ZbsNctij", T1A, d, d.conj(), optimize=[(0,1),(0,1)], backend="torch").reshape(D2,D2,d_PHYS,d_PHYS)
    o['E'] = oe.contract("XZ,LXct,abci,rstj->LarZbsij", C13CF, T3B, e, e.conj(), optimize=[(0,1),(0,1),(0,1)], backend="torch").reshape(D2,D2,d_PHYS,d_PHYS)
    o['F'] = oe.contract("LXbs,abci,rstj->XctLarij", T2C, f, f.conj(), optimize=[(0,1),(0,1)], backend="torch").reshape(D2,D2,d_PHYS,d_PHYS)
    cl = {k: oe.contract("ABii->AB", o[k], backend="torch") for k in o}
    return o, cl


def build_open_closed_env3(a, b, c, d, e, f, chi, D_bond, d_PHYS,
                           C21AF, C32CB, C13ED, T1B, T2E, T2D, T3A, T3F, T1C):
    """Construct open/closed tensors for environment 3 (cdefab)."""
    T1B = T1B.reshape(chi,chi,D_bond,D_bond)
    T2E = T2E.reshape(chi,chi,D_bond,D_bond)
    T2D = T2D.reshape(chi,chi,D_bond,D_bond)
    T3A = T3A.reshape(chi,chi,D_bond,D_bond)
    T3F = T3F.reshape(chi,chi,D_bond,D_bond)
    T1C = T1C.reshape(chi,chi,D_bond,D_bond)
    D2 = chi*D_bond*D_bond
    o = {}
    o['C'] = oe.contract("YX,MYar,abci,rstj->MbsXctij", C21AF, T1B, c, c.conj(), optimize=[(0,1),(0,1),(0,1)], backend="torch").reshape(D2,D2,d_PHYS,d_PHYS)
    o['F'] = oe.contract("MYct,abci,rstj->YarMbsij", T3A, f, f.conj(), optimize=[(0,1),(0,1)], backend="torch").reshape(D2,D2,d_PHYS,d_PHYS)
    o['E'] = oe.contract("ZY,NZbs,abci,rstj->NctYarij", C32CB, T2D, e, e.conj(), optimize=[(0,1),(0,1),(0,1)], backend="torch").reshape(D2,D2,d_PHYS,d_PHYS)
    o['B'] = oe.contract("NZar,abci,rstj->ZbsNctij", T1C, b, b.conj(), optimize=[(0,1),(0,1)], backend="torch").reshape(D2,D2,d_PHYS,d_PHYS)
    o['A'] = oe.contract("XZ,LXct,abci,rstj->LarZbsij", C13ED, T3F, a, a.conj(), optimize=[(0,1),(0,1),(0,1)], backend="torch").reshape(D2,D2,d_PHYS,d_PHYS)
    o['D'] = oe.contract("LXbs,abci,rstj->XctLarij", T2E, d, d.conj(), optimize=[(0,1),(0,1)], backend="torch").reshape(D2,D2,d_PHYS,d_PHYS)
    cl = {k: oe.contract("ABii->AB", o[k], backend="torch") for k in o}
    return o, cl


def build_single_open_env1(site: str, a, b, c, d, e, f, chi, D_bond, d_PHYS,
                          C21CD, C32EF, C13AB, T1F, T2A, T2B, T3C, T3D, T1E):
    """Return the open tensor for ONE site in environment 1 (ebadcf).

    Memory-efficient single-site version of build_open_closed_env1.
    Used by evaluate_observables on GPU+D>=8 so that only 2 open tensors
    (~chi^2 D^4 d^2) are ever alive simultaneously instead of all 6.
    """
    T1F = T1F.reshape(chi, chi, D_bond, D_bond)
    T2A = T2A.reshape(chi, chi, D_bond, D_bond)
    T2B = T2B.reshape(chi, chi, D_bond, D_bond)
    T3C = T3C.reshape(chi, chi, D_bond, D_bond)
    T3D = T3D.reshape(chi, chi, D_bond, D_bond)
    T1E = T1E.reshape(chi, chi, D_bond, D_bond)
    D2 = chi * D_bond * D_bond
    if site == 'E':
        return oe.contract("YX,MYar,abci,rstj->MbsXctij", C21CD, T1F, e, e.conj(), optimize=[(0,1),(0,1),(0,1)], backend="torch").reshape(D2, D2, d_PHYS, d_PHYS)
    if site == 'D':
        return oe.contract("MYct,abci,rstj->YarMbsij", T3C, d, d.conj(), optimize=[(0,1),(0,1)], backend="torch").reshape(D2, D2, d_PHYS, d_PHYS)
    if site == 'A':
        return oe.contract("ZY,NZbs,abci,rstj->NctYarij", C32EF, T2B, a, a.conj(), optimize=[(0,1),(0,1),(0,1)], backend="torch").reshape(D2, D2, d_PHYS, d_PHYS)
    if site == 'F':
        return oe.contract("NZar,abci,rstj->ZbsNctij", T1E, f, f.conj(), optimize=[(0,1),(0,1)], backend="torch").reshape(D2, D2, d_PHYS, d_PHYS)
    if site == 'C':
        return oe.contract("XZ,LXct,abci,rstj->LarZbsij", C13AB, T3D, c, c.conj(), optimize=[(0,1),(0,1),(0,1)], backend="torch").reshape(D2, D2, d_PHYS, d_PHYS)
    if site == 'B':
        return oe.contract("LXbs,abci,rstj->XctLarij", T2A, b, b.conj(), optimize=[(0,1),(0,1)], backend="torch").reshape(D2, D2, d_PHYS, d_PHYS)
    raise ValueError(f"Unknown site '{site}' for env1; expected one of A-F")


def build_single_open_env2(site: str, a, b, c, d, e, f, chi, D_bond, d_PHYS,
                          C21EB, C32AD, C13CF, T1D, T2C, T2F, T3E, T3B, T1A):
    """Return the open tensor for ONE site in environment 2 (afcbed)."""
    T1D = T1D.reshape(chi, chi, D_bond, D_bond)
    T2C = T2C.reshape(chi, chi, D_bond, D_bond)
    T2F = T2F.reshape(chi, chi, D_bond, D_bond)
    T3E = T3E.reshape(chi, chi, D_bond, D_bond)
    T3B = T3B.reshape(chi, chi, D_bond, D_bond)
    T1A = T1A.reshape(chi, chi, D_bond, D_bond)
    D2 = chi * D_bond * D_bond
    if site == 'A':
        return oe.contract("YX,MYar,abci,rstj->MbsXctij", C21EB, T1D, a, a.conj(), optimize=[(0,1),(0,1),(0,1)], backend="torch").reshape(D2, D2, d_PHYS, d_PHYS)
    if site == 'B':
        return oe.contract("MYct,abci,rstj->YarMbsij", T3E, b, b.conj(), optimize=[(0,1),(0,1)], backend="torch").reshape(D2, D2, d_PHYS, d_PHYS)
    if site == 'C':
        return oe.contract("ZY,NZbs,abci,rstj->NctYarij", C32AD, T2F, c, c.conj(), optimize=[(0,1),(0,1),(0,1)], backend="torch").reshape(D2, D2, d_PHYS, d_PHYS)
    if site == 'D':
        return oe.contract("NZar,abci,rstj->ZbsNctij", T1A, d, d.conj(), optimize=[(0,1),(0,1)], backend="torch").reshape(D2, D2, d_PHYS, d_PHYS)
    if site == 'E':
        return oe.contract("XZ,LXct,abci,rstj->LarZbsij", C13CF, T3B, e, e.conj(), optimize=[(0,1),(0,1),(0,1)], backend="torch").reshape(D2, D2, d_PHYS, d_PHYS)
    if site == 'F':
        return oe.contract("LXbs,abci,rstj->XctLarij", T2C, f, f.conj(), optimize=[(0,1),(0,1)], backend="torch").reshape(D2, D2, d_PHYS, d_PHYS)
    raise ValueError(f"Unknown site '{site}' for env2; expected one of A-F")


def build_single_open_env3(site: str, a, b, c, d, e, f, chi, D_bond, d_PHYS,
                          C21AF, C32CB, C13ED, T1B, T2E, T2D, T3A, T3F, T1C):
    """Return the open tensor for ONE site in environment 3 (cdefab)."""
    T1B = T1B.reshape(chi, chi, D_bond, D_bond)
    T2E = T2E.reshape(chi, chi, D_bond, D_bond)
    T2D = T2D.reshape(chi, chi, D_bond, D_bond)
    T3A = T3A.reshape(chi, chi, D_bond, D_bond)
    T3F = T3F.reshape(chi, chi, D_bond, D_bond)
    T1C = T1C.reshape(chi, chi, D_bond, D_bond)
    D2 = chi * D_bond * D_bond
    if site == 'C':
        return oe.contract("YX,MYar,abci,rstj->MbsXctij", C21AF, T1B, c, c.conj(), optimize=[(0,1),(0,1),(0,1)], backend="torch").reshape(D2, D2, d_PHYS, d_PHYS)
    if site == 'F':
        return oe.contract("MYct,abci,rstj->YarMbsij", T3A, f, f.conj(), optimize=[(0,1),(0,1)], backend="torch").reshape(D2, D2, d_PHYS, d_PHYS)
    if site == 'E':
        return oe.contract("ZY,NZbs,abci,rstj->NctYarij", C32CB, T2D, e, e.conj(), optimize=[(0,1),(0,1),(0,1)], backend="torch").reshape(D2, D2, d_PHYS, d_PHYS)
    if site == 'B':
        return oe.contract("NZar,abci,rstj->ZbsNctij", T1C, b, b.conj(), optimize=[(0,1),(0,1)], backend="torch").reshape(D2, D2, d_PHYS, d_PHYS)
    if site == 'A':
        return oe.contract("XZ,LXct,abci,rstj->LarZbsij", C13ED, T3F, a, a.conj(), optimize=[(0,1),(0,1),(0,1)], backend="torch").reshape(D2, D2, d_PHYS, d_PHYS)
    if site == 'D':
        return oe.contract("LXbs,abci,rstj->XctLarij", T2E, d, d.conj(), optimize=[(0,1),(0,1)], backend="torch").reshape(D2, D2, d_PHYS, d_PHYS)
    raise ValueError(f"Unknown site '{site}' for env3; expected one of A-F")


def build_heisenberg_H(J: float = 1.0, d: int = 2) -> torch.Tensor:
    # use spin s=(d-1)/2 to build spin-s operators sx, sy, sz:
    spin = (d - 1) / 2
    spin = torch.tensor(spin, dtype=RDTYPE)
    Splus = torch.zeros((d, d), dtype=RDTYPE)
    Sminus = torch.zeros((d, d), dtype=RDTYPE)
    Sz = torch.zeros((d, d), dtype=RDTYPE)
    for i in range(d):
        m = float(spin - i)                   # numeric m-value (float)
        Sz[i, i] = m                          # diagonal Sz

        if i < d - 1:
            # coefficient for the transition m -> m-1
            CScoeff = torch.sqrt(spin * (spin + 1) - m * (m - 1))  # real CS-coeff/2
            Splus[i, i + 1] = CScoeff   # S+ raises m by 1
            Sminus[i + 1, i] = CScoeff  # S- lowers m by 1

    print(Splus,Sminus,Sz)

    # Perform contractions on CPU with explicit backend, then move to device
    SdotS = (oe.contract("ij,kl->ijkl", Splus, Sminus, backend="torch") * 0.5
            +oe.contract("ij,kl->ijkl", Sminus, Splus, backend="torch") * 0.5
            +oe.contract("ij,kl->ijkl", Sz, Sz, backend="torch")
            ).to(dtype=TENSORDTYPE, device=DEVICE)
    return SdotS



