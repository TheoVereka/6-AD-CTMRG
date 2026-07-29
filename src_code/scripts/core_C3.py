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
# _SVD_CPU_OFFLOAD_THRESHOLD: int = 0

# ── Truncation error recording ─────────────────────────────────────────────
# Set _RECORD_TRUNC_ERROR=True (via set_record_trunc_error) before a clean-
# evaluation CTMRG to record the SVD truncation errors.  Zero cost when False
# (the recording branch is never entered during normal optimisation).  The
# accumulated errors come from the last CTMRG iteration's three trunc_rhoC3 calls (3 calls × 1 SVD = 3 values), averaged to produce one scalar.
_RECORD_TRUNC_ERROR: bool = False
_TRUNC_ERRORS_ACC: list = []   # populated by trunc_rhoC3 when flag is True


def set_record_trunc_error(flag: bool) -> None:
    """Enable/disable SVD truncation-error recording in trunc_rhoC3.

    Call with flag=True immediately before CTMRG_from_init_to_stop in a
    clean (no-grad) evaluation; call with flag=False afterwards.  When True,
    each trunc_rhoC3 call appends one truncation error to the internal
    accumulator.  After CTMRG completes, call get_last_trunc_error() to
    retrieve the average over the last 3 recorded values (= last full
    C3-CTMRG iteration).
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

    Returns the mean over the last 3 recorded values (last full C3-CTMRG
    iteration: 3 environments × 1 SVD each), or None if no data.
    """
    if not _TRUNC_ERRORS_ACC:
        return None
    recent = _TRUNC_ERRORS_ACC[-3:]
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


def _record_trunc_err_from_kept(
        frobenius_sq: torch.Tensor, S_kept: torch.Tensor) -> None:
    """Record discarded weight without computing a full singular spectrum."""
    with torch.no_grad():
        total_sq = frobenius_sq.real.clamp(min=1e-30)
        kept_sq = S_kept.real.square().sum()
        discarded_sq = (total_sq - kept_sq).clamp(min=0.0)
        _TRUNC_ERRORS_ACC.append(
            (torch.sqrt(discarded_sq) / torch.sqrt(total_sq)).item())

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

# ── CTM convergence mode ──────────────────────────────────────────────────────
#
# Controls which criterion(a) must be satisfied for CTMRG_from_init_to_stop
# to declare convergence.
#
#   'SVdifference'  (default) — original criterion: converge when max change in
#       normalised corner singular values between consecutive CTMRG steps
#       is below env_conv_threshold (passed to CTMRG_from_init_to_stop).
#       Fast: O(chi) per check.  No extra tensor builds needed.
#
#   'Edifference'   — physics-grounded criterion: converge when the change
#       in the 1-bond energy proxy (NN EB bond from env1) between consecutive
#       checks is below _CTM_E_CONV_THRESHOLD.  Slightly slower (builds and
#       frees 6 open tensors per check, ~1/27 of full energy cost), but
#       guarantees that observables are converged, not just SVs.
#       SV check is still computed every iteration for zero-collapse detection.
#
#   'both'          — require BOTH SV convergence AND E convergence.
#       The SV criterion uses env_conv_threshold; the E criterion uses
#       _CTM_E_CONV_THRESHOLD.  Strictest mode: eliminates cases where SVs
#       converge but energy still drifts (possible near phase boundaries).
#
# Set via set_ctm_conv_mode(); defaults preserved for backward compatibility.
#
# Energy proxy cost analysis (D=8, chi=80):
#   open tensor size = D2² · d² · 8B = (80·64)² · 4 · 8 ≈ 686 MB
#   proxy builds 6 open tensors sequentially (each freed before the next) +
#   _build_nn_rho_seq (peak = W + oB = 2 opens, ~1.4 GB simultaneously).
#   With _CTM_E_PROXY_INTERVAL=5, cost = 6 builds / 5 iter × CTMRG_steps.
#   For CTMRG_steps=20: 24 builds total ≈ 20 ms/build × 24 ≈ 480 ms extra.
#   Full energy = 27 bonds × ~20 ms/bond ≈ 540 ms → proxy = ~89% of 1 energy.
#   Over 20 CTMRG steps: 89% × (20/5) = ~356% of 1 energy as overhead.
#   NOT recommended for optimization CTMRG (pass energy_proxy_fn=None).
#   Only enable for evaluation CTMRG (evaluate_energy_clean, evaluate_observables).
_CTM_CONV_MODE: str = 'both'
#   Active convergence mode.  Changed by set_ctm_conv_mode().

_CTM_E_CONV_THRESHOLD: float = 1e-8
#   Energy-proxy convergence threshold (|ΔE_proxy| < this → E-converged).
#   Only used when _CTM_CONV_MODE is 'Edifference' or 'both'.

# ── Rho side-channel cache ────────────────────────────────────────────────────
# When _CACHE_RHOS is True, each energy function writes the public 12-rho
# layout (6 NN + 6 NNN) for its environment into _RHO_CACHE.  C3-equivalent
# entries alias one of four representative small density matrices, so caching
# does not trigger any extra large open-tensor construction.
# Usage: set _CACHE_RHOS=True before the optimizer loop; read _RHO_CACHE after.
_CACHE_RHOS: bool = False
_RHO_CACHE: dict = {}   # 'env1'/'env2'/'env3' → (rho_list_12, mag_dict_6sites)
_RHO_ACC: list = []

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


def set_ctm_conv_mode(mode: str,
                      e_threshold: float = 1e-8,) -> None:
    """Set the CTMRG convergence criterion mode and associated parameters.

    Args:
        mode:             One of 'SVdifference', 'Edifference', 'both'.
            'SVdifference'  — (default) converge on max|ΔSV| < env_conv_threshold.
            'Edifference'   — converge on |ΔE_proxy| < e_threshold.
                              SV check still runs every iteration for collapse
                              detection (returns None when collapsed).
            'both'          — converge only when BOTH SV AND E criteria are met.
        e_threshold:      Energy-proxy convergence threshold for 'Edifference'
                          and 'both' modes.  Default 1e-8.
        e_proxy_interval: Number of CTMRG iterations between consecutive energy
                          proxy evaluations.  interval=1 checks every step;
                          interval=5 (default) checks every 5th step.
                          Higher = faster CTMRG; lower = tighter tracking.
                          Minimum 1.

    Note: The energy proxy is the NN EB bond energy from env1.  Building it
    requires 6 sequential open tensor builds (each ~D²·χ²·d²·bytes) under
    torch.no_grad().  Pass energy_proxy_fn=None to CTMRG_from_init_to_stop
    during optimization to avoid this overhead; only pass a real callable
    during clean evaluation (evaluate_energy_clean, evaluate_observables).
    """
    global _CTM_CONV_MODE, _CTM_E_CONV_THRESHOLD
    if mode not in ('SVdifference', 'Edifference', 'both'):
        raise ValueError(
            f"Unknown CTM convergence mode {mode!r}; "
            f"choose from 'SVdifference', 'Edifference', 'both'"
        )
    _CTM_CONV_MODE = mode
    _CTM_E_CONV_THRESHOLD = float(e_threshold)





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


        if _RSVD_BACKWARD_MODE != 'full_svd' and not bool(use_full_svd):
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
            if dev.type == 'cuda' :
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
    # Normalise to unit sphere so that the Riemannian Adam retraction (which
    # expects the parameter to start on S^(N_tri3·d-1)) is a no-op on the first
    # step.  Without this, ||a|| ≈ √(N_tri3·d) ≫ 1 and the first Adam step is
    # computed at the wrong geometric basepoint (though the retraction corrects
    # it one step later, the first gradient direction is slightly off-sphere).
    _n = a.norm()
    if _n > 1e-30:
        a = a / _n
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
    a_raw = torch.randn(
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
    a_raw = torch.randn(
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
    a_raw = torch.randn(
        D_bond, D_bond, D_bond, d_PHYS,
        dtype=TENSORDTYPE, device=DEVICE)
    b_raw = torch.randn(
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


# ── Free-parameter morphisms for symmetry-projected ansatze ──────────────────
#
# For ansatze with exact symmetrize_fn constraints (neel, sym6), optimizing the
# full D³·d parameter tensor and projecting at every forward pass has two costs:
#   1. Gradient waste: the gradient component outside the symmetric subspace
#      is silently discarded, so the effective learning rate is reduced.
#   2. Spurious flat directions: the optimizer can move freely in the null space
#      of the projection without changing the energy, wasting line-search steps.
#
# The fix: reparametrize so the optimizer's parameters ARE the free coordinates
# of the symmetric subspace.  The morphism (a gather-based index expansion) maps
# each free parameter tensor directly to the exact symmetric full tensor:
#
#   sym6 — 3 local mirror types, each with one 2-index swap:
#     a, d  (leg1↔leg2):  full[i,j,k,s] = h[i, tri_idx[j,k], s]
#     b, e  (leg0↔leg1):  full[i,j,k,s] = h[tri_idx[i,j], k, s]
#     c, f  (leg0↔leg2):  full[i,j,k,s] = h[tri_idx[i,k], j, s]
#   where tri_idx[p,q] = tri_idx[q,p] is a precomputed (D,D) table of unique
#   indices 0..D*(D+1)//2-1.  Free param shapes (half of D³):
#     h_a, h_d:  (D,  D*(D+1)//2, d_PHYS)
#     h_b, h_e:  (D*(D+1)//2, D,  d_PHYS)
#     h_c, h_f:  (D*(D+1)//2, D,  d_PHYS)
#
#   neel — full S₃ symmetry on all 3 virtual legs:
#     full[i,j,k,s] = h[tri3_idx[i,j,k], s]
#   where tri3_idx is a precomputed (D,D,D) table invariant under all 6
#   permutations of (i,j,k).  Free param shape:
#     h:  (D*(D+1)*(D+2)//6, d_PHYS)
#
# Both morphisms are pure tensor indexing — autograd differentiates through
# them for free.  Index tables are cached in module-level dicts (computed once
# per D on first call).
#

_TRI_IDX_CACHE:  dict = {}   # (D, device_str) → (D,D) long tensor
_TRI3_IDX_CACHE: dict = {}   # (D, device_str) → (D,D,D) long tensor
_TRI3_MULT_CACHE: dict = {}  # (D, device_str) → (D,D,D) float64 tensor: sqrt(#permutations)


def _get_tri_idx(D: int, device: torch.device) -> torch.Tensor:
    """Return (cached) (D,D) symmetric triangle-index table.

    tri[j,k] == tri[k,j] == unique index in [0, D*(D+1)//2) for pair {j,k}.
    Computed once per (D, device) then stored in _TRI_IDX_CACHE.
    """
    key = (D, str(device))
    if key not in _TRI_IDX_CACHE:
        idx = torch.zeros(D, D, dtype=torch.long, device=device)
        m = 0
        for j in range(D):
            for k in range(j, D):
                idx[j, k] = m
                idx[k, j] = m
                m += 1
        _TRI_IDX_CACHE[key] = idx
    return _TRI_IDX_CACHE[key]


def _get_tri3_idx(D: int, device: torch.device) -> torch.Tensor:
    """Return (cached) (D,D,D) S₃-symmetric triangle-index table.

    tri3[i,j,k] is the same for all 6 permutations of (i,j,k).
    Values in [0, D*(D+1)*(D+2)//6).
    Computed once per (D, device) then stored in _TRI3_IDX_CACHE.
    """
    from itertools import permutations as _perms
    key = (D, str(device))
    if key not in _TRI3_IDX_CACHE:
        idx  = torch.zeros(D, D, D, dtype=torch.long,  device=device)
        mult = torch.zeros(D, D, D, dtype=torch.float64, device=device)
        m = 0
        for i in range(D):
            for j in range(i, D):
                for k in range(j, D):
                    perms = set(_perms((i, j, k)))
                    n_perm = len(perms)
                    for p in perms:
                        idx[p]  = m
                        mult[p] = n_perm
                    m += 1
        _TRI3_IDX_CACHE[key] = idx
        _TRI3_MULT_CACHE[key] = mult.sqrt()   # store sqrt(multiplicity)
    elif key not in _TRI3_MULT_CACHE:
        # Rebuild mult if only idx was cached (older session)
        from itertools import permutations as _perms2
        mult = torch.zeros(D, D, D, dtype=torch.float64, device=device)
        for i in range(D):
            for j in range(i, D):
                for k in range(j, D):
                    perms = set(_perms2((i, j, k)))
                    for p in perms:
                        mult[p] = len(perms)
        _TRI3_MULT_CACHE[key] = mult.sqrt()
    return _TRI3_IDX_CACHE[key]


def _get_tri3_sqrt_mult(D: int, device: torch.device) -> torch.Tensor:
    """Return (cached) (D,D,D) tensor of sqrt(#permutations) for each (i,j,k).

    Used by neel_param_to_a to give every free parameter h[m] a unit-scale
    gradient: a[i,j,k,s] = h[tri3[i,j,k], s] / sqrt_mult[i,j,k]  so that
    ∂L/∂h[m] = Σ_{perms} ∂L/∂a[perm] / sqrt(n_perm)  has ||∂L/∂h||_F
    equalized across m regardless of how many permutations class m has.
    May trigger computation of tri3 idx as a side-effect.
    """
    key = (D, str(device))
    if key not in _TRI3_MULT_CACHE:
        _get_tri3_idx(D, device)   # fills both caches
    return _TRI3_MULT_CACHE[key]


# ── sym6 morphism ─────────────────────────────────────────────────────────────

def sym6_params_to_abcdef(
        h_a: torch.Tensor, h_b: torch.Tensor, h_c: torch.Tensor,
        h_d: torch.Tensor, h_e: torch.Tensor, h_f: torch.Tensor,
) -> tuple:
    """Map unconstrained sym6 free parameters → 6 exactly-symmetric tensors.

    The map is a pure gather (advanced indexing) — autograd differentiates it
    for free.  No projection needed; every value of h gives a valid symmetric
    tensor.

    Free parameter shapes:
        h_a, h_d : (D, D*(D+1)//2, d_PHYS)   — leg1↔leg2 symmetry
        h_b, h_e : (D*(D+1)//2, D,  d_PHYS)   — leg0↔leg1 symmetry
        h_c, h_f : (D*(D+1)//2, D,  d_PHYS)   — leg0↔leg2 symmetry

    Morphisms:
        a[i,j,k,s] = h_a[i, tri[j,k], s]
        b[i,j,k,s] = h_b[tri[i,j], k,  s]
        c[i,j,k,s] = h_c[tri[i,k], j,  s]
        (same for d,e,f with h_d,h_e,h_f)

    Returns:
        (a, b, c, d, e, f) each of shape (D, D, D, d_PHYS), exactly symmetric.
    """
    D   = h_a.shape[0]
    tri = _get_tri_idx(D, h_a.device)

    # a,d : h[:,  tri[j,k], :] — tri is (D,D); h[:,tri,:] broadcasts to (D,D,D,d)
    a = h_a[:, tri, :]
    d = h_d[:, tri, :]

    # b,e : h[tri[i,j], :, :] — shape (D,D,D,d) with indices (i,j,k,s) directly
    b = h_b[tri, :, :]
    e = h_e[tri, :, :]

    # c,f : h[tri[i,k], :, :] has index order (i,k,j,s); permute to (i,j,k,s)
    c = h_c[tri, :, :].permute(0, 2, 1, 3)
    f = h_f[tri, :, :].permute(0, 2, 1, 3)

    return a, b, c, d, e, f


def sym6_abcdef_to_free_params(
        a: torch.Tensor, b: torch.Tensor, c: torch.Tensor,
        d: torch.Tensor, e: torch.Tensor, f: torch.Tensor,
) -> tuple:
    """Extract sym6 free parameters from 6 already-symmetric site tensors.

    Left inverse of sym6_params_to_abcdef: extracts the D*(D+1)//2 independent
    entries for each tensor by selecting the upper-triangle (j ≤ k, i ≤ j,
    i ≤ k respectively).  Used to convert full padded tensors back to the free
    parametrization during warm-start.

    Returns:
        (h_a, h_b, h_c, h_d, h_e, h_f) with reduced shapes as above.
    """
    D   = a.shape[0]
    dev = a.device
    rows, cols = torch.triu_indices(D, D, device=dev)   # row ≤ col  (N_tri pairs)

    h_a = a[:, rows, cols, :]      # (D, N_tri, d) : upper-tri of (j,k) for each i
    h_d = d[:, rows, cols, :]

    h_b = b[rows, cols, :, :]      # (N_tri, D, d) : upper-tri of (i,j)
    h_e = e[rows, cols, :, :]

    # c[i,j,k] symmetric in (i,k): extract upper-tri of (i,k) for each j
    h_c = c[rows, :, cols, :]      # (N_tri, D, d)
    h_f = f[rows, :, cols, :]

    return h_a, h_b, h_c, h_d, h_e, h_f


def initialize_sym6_free_params(D: int, d_PHYS: int,
                                noise_scale: float = 1.0) -> tuple:
    """Random initialization of the sym6 free parameters (6-tuple of reduced tensors).

    Sets _USE_FULL_SVD = True so the first L-BFGS step uses full deterministic
    SVD (avoids rSVD gradient noise on the fresh random point).

    Returns:
        (h_a, h_b, h_c, h_d, h_e, h_f) with shapes as in sym6_params_to_abcdef.
    """
    global _USE_FULL_SVD
    N = D * (D + 1) // 2
    kw = dict(dtype=TENSORDTYPE, device=DEVICE)
    h_a = torch.randn(D, N, d_PHYS, **kw) * noise_scale
    h_b = torch.randn(N, D, d_PHYS, **kw) * noise_scale
    h_c = torch.randn(N, D, d_PHYS, **kw) * noise_scale
    h_d = torch.randn(D, N, d_PHYS, **kw) * noise_scale
    h_e = torch.randn(N, D, d_PHYS, **kw) * noise_scale
    h_f = torch.randn(N, D, d_PHYS, **kw) * noise_scale
    _USE_FULL_SVD = True
    return (h_a, h_b, h_c, h_d, h_e, h_f)


def pad_sym6_free_params(
        old_params: tuple,
        old_D: int, new_D: int,
        d_PHYS: int, noise: float,
) -> tuple:
    """Warm-start sym6 free params from D=old_D to D=new_D.

    1. Expand old free params → 6 full (old_D, old_D, old_D, d) tensors.
    2. Normalize + zero-pad each to (new_D, new_D, new_D, d) with noise.
    3. Project via symmetrize_six_local_reflections (exact symmetry).
    4. Extract new free params.

    Returns:
        (h_a, h_b, h_c, h_d, h_e, h_f) for the new D.
    """
    global _USE_FULL_SVD
    h_a, h_b, h_c, h_d, h_e, h_f = old_params
    # Expand to full tensors (detached — warm-start is outside autograd)
    a6 = sym6_params_to_abcdef(
        h_a.detach(), h_b.detach(), h_c.detach(),
        h_d.detach(), h_e.detach(), h_f.detach())
    kw = dict(dtype=TENSORDTYPE, device=DEVICE)
    padded = []
    scale = float(torch.sqrt(torch.tensor(old_D**3 * d_PHYS, dtype=RDTYPE)))
    for t in a6:
        t_norm = normalize_tensor(t) * scale
        new_t = noise * torch.randn(new_D, new_D, new_D, d_PHYS, **kw)
        new_t[:old_D, :old_D, :old_D, :] += t_norm
        padded.append(new_t)
    # Project onto symmetric subspace and extract free params
    sym6 = symmetrize_six_local_reflections(*padded)
    _USE_FULL_SVD = True
    return sym6_abcdef_to_free_params(*sym6)


# ── neel morphism ─────────────────────────────────────────────────────────────

def neel_param_to_a(h: torch.Tensor, D: int) -> torch.Tensor:
    """Map Néel free parameter h → S₃-symmetric unit-norm tensor a.

    Forward pass:
        h_unit = h / ||h||      (project h onto S^(N-1))
        a[i,j,k,s] = h_unit[tri3[i,j,k], s] / sqrt(mult[i,j,k])

    **Why normalise h here?**
    This is the ONLY place h enters the computation graph.  Normalising
    h → h_unit = h/||h|| makes autograd automatically include the factor

        ∂h_unit/∂h = (I − ĥĥᵀ) / ||h||

    in every gradient, i.e. the gradient is always projected onto the
    tangent plane of S^(N-1) at ĥ.  This gives L-BFGS correct Riemannian
    gradients (and correct curvature pairs (s,y)) for every line-search
    closure call, without requiring an explicit hook.  It is *orthogonal*
    to the Riemannian Adam hook registered in optimize_at_chi, which
    additionally projects the first moment.

    **Isometric property** (when ||h||=1, so h_unit=h):
        ||a||² = Σ_{i,j,k} h[tri3[i,j,k],s]² / mult[i,j,k]
               = Σ_m  n_m · |h[m,:]|² / n_m  =  ||h||²  =  1  ✓
    The scaling also makes the gradient magnitudes uniform across
    permutation classes (all classes contribute equally to ||∇h||).

    Args:
        h : Free parameter of shape (D*(D+1)*(D+2)//6, d_PHYS), any norm.
        D : Virtual bond dimension.

    Returns:
        Tensor of shape (D, D, D, d_PHYS), exactly S₃-symmetric,
        unit Frobenius norm.
    """
    tri3      = _get_tri3_idx(D, h.device)
    sqrt_mult = _get_tri3_sqrt_mult(D, h.device).to(dtype=h.dtype)
    # Project h onto the unit sphere before gathering.  This ensures that:
    #   (a) The physical tensor a has unit Frobenius norm regardless of ||h||.
    #   (b) Autograd automatically includes (I - ĥĥᵀ)/||h|| in ∂a/∂h, which
    #       is the tangent-space projection at ĥ.  This gives L-BFGS the
    #       correct Riemannian gradient for every closure call without needing
    #       an explicit hook — orthogonal to the Adam-specific hook + retraction
    #       registered in optimize_at_chi.
    h_unit = h / (h.norm() + 1e-30)
    return h_unit[tri3, :] / sqrt_mult.unsqueeze(-1)


def neel_a_to_free_param(a_sym: torch.Tensor) -> torch.Tensor:
    """Extract Néel free parameter from an S₃-symmetric tensor a_sym.

    Left inverse of neel_param_to_a (with sqrt-mult scaling): picks the
    canonical (i ≤ j ≤ k) entry and multiplies by sqrt(multiplicity) so
    that neel_param_to_a(neel_a_to_free_param(a_sym), D) recovers a_sym.

    Returns:
        h of shape (D*(D+1)*(D+2)//6, d_PHYS).
    """
    D   = a_sym.shape[0]
    dev = a_sym.device
    triples = [(i, j, k)
               for i in range(D) for j in range(i, D) for k in range(j, D)]
    i_idx = torch.tensor([x[0] for x in triples], dtype=torch.long, device=dev)
    j_idx = torch.tensor([x[1] for x in triples], dtype=torch.long, device=dev)
    k_idx = torch.tensor([x[2] for x in triples], dtype=torch.long, device=dev)
    # Extract canonical values and multiply by sqrt(#perms) to invert the scaling
    sqrt_mult = _get_tri3_sqrt_mult(D, dev).to(dtype=a_sym.dtype)
    scale = sqrt_mult[i_idx, j_idx, k_idx]       # (N3,)
    return a_sym[i_idx, j_idx, k_idx, :] * scale.unsqueeze(-1)


def initialize_neel_free_param(D: int, d_PHYS: int,
                               noise_scale: float = 1.0) -> torch.Tensor:
    """Random initialization of the Néel free parameter on the unit sphere.

    Returns h of shape (D*(D+1)*(D+2)//6, d_PHYS) with ||h||_F = 1.
    The unit-sphere constraint is maintained throughout optimisation by
    Riemannian Adam retraction in optimize_at_chi.
    Sets _USE_FULL_SVD = True for the first L-BFGS step.
    """
    global _USE_FULL_SVD
    N = D * (D + 1) * (D + 2) // 6
    h = torch.randn(N, d_PHYS, dtype=TENSORDTYPE, device=DEVICE)
    h = h / (h.norm() + 1e-30)   # initialise on unit sphere in h-space
    _USE_FULL_SVD = True
    return h


def pad_neel_free_param(
        h_old: torch.Tensor,
        old_D: int, new_D: int,
        d_PHYS: int, noise: float,
) -> torch.Tensor:
    """Warm-start Néel free param from D=old_D to D=new_D.

    1. Expand h_old → a_sym (old_D, old_D, old_D, d).
    2. Normalize + zero-pad to (new_D, new_D, new_D, d) with noise.
    3. Re-symmetrize (S₃) to land exactly in the symmetric subspace.
    4. Extract new free param.

    Returns:
        h of shape (new_D*(new_D+1)*(new_D+2)//6, d_PHYS).
    """
    global _USE_FULL_SVD
    a_old  = neel_param_to_a(h_old.detach(), old_D)
    scale  = float(torch.sqrt(torch.tensor(old_D**3 * d_PHYS, dtype=RDTYPE)))
    a_norm = normalize_tensor(a_old) * scale
    kw     = dict(dtype=TENSORDTYPE, device=DEVICE)
    a_new  = noise * torch.randn(new_D, new_D, new_D, d_PHYS, **kw)
    a_new[:old_D, :old_D, :old_D, :] += a_norm
    a_new  = symmetrize_virtual_legs(a_new)
    _USE_FULL_SVD = True
    h_new  = neel_a_to_free_param(a_new)
    return h_new / (h_new.norm() + 1e-30)   # warm-start h on unit sphere


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










def _cubic_apply(c_transpose: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Apply ``(C.T)^3`` to a thin matrix without materialising either cube."""
    return c_transpose @ (c_transpose @ (c_transpose @ x))


def _cubic_adjoint_apply(
        c_transpose: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Apply the Hermitian adjoint of ``(C.T)^3`` to a thin matrix."""
    ch = c_transpose.mH
    return ch @ (ch @ (ch @ x))


def _matrix_free_cubic_rsvd(
        matC: torch.Tensor, chi: int) -> tuple:
    """Leading cubic-corner singular triples with O(N*chi) workspace.

    This is algebraically the same range finder used by ``SVD_PROPACK`` in
    augmented mode, but all products with ``M=(C.T)^3`` are applied as three
    skinny GEMMs.  ``Q`` remains detached, matching the existing augmented
    backward convention; autograd follows the projected matrix back through
    all three factors of C.
    """
    n = matC.shape[0]
    q = min(2 * chi, n) if _RSVD_BACKWARD_MODE == 'augmented' else chi
    niter = (_adaptive_power_iters(chi, n)
             if _RSVD_POWER_ITERS is None else _RSVD_POWER_ITERS)
    ct = matC.T

    with torch.no_grad():
        random = torch.randn(n, q, dtype=matC.dtype, device=matC.device)
        Q, _ = torch.linalg.qr(_cubic_apply(ct, random))
        for _ in range(niter):
            Q, _ = torch.linalg.qr(_cubic_adjoint_apply(ct, Q))
            Q, _ = torch.linalg.qr(_cubic_apply(ct, Q))

    # B = Q.mH @ M = (M.mH @ Q).mH.  The latter form is matrix-free and
    # differentiable with respect to matC while Q is intentionally constant.
    B = _cubic_adjoint_apply(ct, Q).mH
    U_b, S_all, Vh_all = torch.linalg.svd(B, full_matrices=False)
    U = Q @ U_b[:, :chi]
    V = Vh_all[:chi].mH
    return U, S_all[:chi], V


def _cubic_frobenius_sq(
        matC: torch.Tensor, block_cols: int) -> torch.Tensor:
    """Exact ``||(C.T)^3||_F^2`` using only O(N*block_cols) workspace."""
    ct = matC.T
    n = matC.shape[0]
    total = matC.real.new_zeros(())
    # The first multiplication by an identity block is just a column slice.
    for start in range(0, n, block_cols):
        stop = min(start + block_cols, n)
        block = ct @ (ct @ ct[:, start:stop])
        total = total + block.abs().square().sum()
    return total


def trunc_rhoC3(matC: torch.Tensor, chi: int, D_squared: int):
    """C3-symmetric corner truncation using one enlarged corner only.

    Under exact C3 symmetry the three enlarged corners entering the original
    ``trunc_rhoCCC`` are identical.  The three SVDs and six projectors therefore
    collapse to one SVD and one incoming/outgoing projector pair.  No copies of
    the other two corners are materialised.

    Args:
        matC: enlarged corner, shape ``(n, n)``.  During CTMRG updates
            ``n = chi * D_squared``; during contraction-based initialization
            ``n = D_squared**2``.
        chi: retained CTM bond dimension.
        D_squared: fused double-layer virtual dimension.

    Returns:
        ``(P_in, C, P_out)`` with shapes ``(chi,n)``, ``(chi,chi)``,
        ``(n,chi)``.  ``P_in`` is the projector used on the first CTM index of
        a transfer tensor and ``P_out`` on the second one, matching the index
        orientation of the original C21 representative.
    """
    if matC.ndim != 2 or matC.shape[0] != matC.shape[1]:
        raise ValueError(f"matC must be square, got shape {tuple(matC.shape)}")
    n = matC.shape[0]
    if chi > n:
        raise ValueError(f"chi={chi} exceeds enlarged corner dimension n={n}")

    # Accuracy baseline: preserve the exact 0717 arithmetic and autograd graph.
    # Algebraic reassociation and the matrix-free projected SVD passed an
    # isolated truncation test but changed (and for augmented destabilised)
    # gradients after repeated differentiable CTMRG steps.  The matrix-free
    # helpers therefore remain experimental until their custom backward passes
    # the full-pipeline regression.
    R1 = matC.T
    R2 = torch.mm(matC, matC)
    M = torch.mm(R1, R2.T)

    U, S, V = truncated_svd_propack(
        M, chi,
        chi_extra=round(2*np.sqrt(D_squared)),
        rel_cutoff=1e-12,
        v0=None,
        keep_multiplets=False,
        abs_tol=1e-14,
        eps_multiplet=1e-12,
    )
    if _RECORD_TRUNC_ERROR:
        # Recording remains diagnostic-only and does not switch the requested
        # production mode to full SVD as the 0717 implementation did.
        _record_trunc_err_from_kept(M.detach().abs().square().sum(), S)

    sqrtInvTruncS = _safe_sqrt_inv_diag(S[:chi])
    P_out = torch.mm(R2.T, torch.mm(V, sqrtInvTruncS))
    P_in = torch.mm(torch.mm(sqrtInvTruncS, U.conj().T), R1)

    C = torch.mm(P_out.T, torch.mm(matC, P_in.T))
    C = normalize_tensor(C)
    return P_in, C, P_out





def initialize_envCTs_C3(A, B, C, D, E, F, chi, D_squared, identity_init=False):
    """Initialize the C3 CTM with only ``C21CD, T1F, T2A``.

    ``T1F`` represents the orbit ``{T1F,T2B,T3D}``, while ``T2A`` represents
    ``{T2A,T3C,T1E}``.  The other rotated tensors are never allocated.
    """
    D_bond = round(D_squared ** 0.5)
    if D_bond * D_bond != D_squared:
        raise ValueError(f"D_squared={D_squared} is not a perfect square")

    if identity_init:
        noise = 1e-2
        C21CD = torch.ones((chi, chi), dtype=A.dtype, device=A.device)
        T1F = torch.ones((chi, chi, D_squared), dtype=A.dtype, device=A.device)
        T2A = torch.ones((chi, chi, D_squared), dtype=A.dtype, device=A.device)
        C21CD = normalize_tensor(C21CD + noise * torch.randn_like(C21CD))
        T1F   = normalize_tensor(T1F   + noise * torch.randn_like(T1F))
        T2A   = normalize_tensor(T2A   + noise * torch.randn_like(T2A))
        return C21CD, T1F, T2A

    # Direct contraction-based C3 initialization.  The old implementation
    # first materialised three D^8 pair tensors and two D^10 raw edges.  Fuse
    # the original double-layer tensors and projectors into the networks
    # instead; the largest unavoidable outputs are now D^8 (corner) and
    # chi*D^6 (projected edge).
    A23 = A.reshape(D_bond,D_bond,D_squared,D_squared).diagonal(dim1=0, dim2=1).sum(-1)
    F31 = F.permute(1,2,0).reshape(D_bond,D_bond,D_squared,D_squared).diagonal(dim1=0, dim2=1).sum(-1)
    E23 = E.reshape(D_bond,D_bond,D_squared,D_squared).diagonal(dim1=0, dim2=1).sum(-1)
    B31 = B.permute(1,2,0).reshape(D_bond,D_bond,D_squared,D_squared).diagonal(dim1=0, dim2=1).sum(-1)

    C_raw = oe.contract(
        "yj,pij,pYk,in,nm,Xqk,mql,lx->YyXx",
        E23, B, C, A23, F31, D, E, B31,
        optimize=[(2,5),(0,1),(0,1),(0,4),(0,3),(1,2),(0,1)],
        backend='torch'
    ).reshape(D_squared*D_squared, D_squared*D_squared)

    C23 = C.reshape(D_bond,D_bond,D_squared,D_squared).diagonal(dim1=0, dim2=1).sum(-1)
    F12 = F.permute(2,0,1).reshape(D_bond,D_bond,D_squared,D_squared).diagonal(dim1=0, dim2=1).sum(-1)

    P_in, C21CD, P_out = trunc_rhoC3(C_raw, chi, D_squared)
    n0 = D_squared * D_squared

    TB = oe.contract(
        "mi,jyi,aYq,jMq,pYy->Mmpa",
        C23, D, F, A, P_in.reshape(chi, D_squared, D_squared),
        optimize=[(0,1),(0,1),(0,1),(0,1)], backend='torch'
    ).reshape(n0, chi*D_squared)
    # C3-rotated representative of the T2A orbit, directly in the orientation
    # paired with T1F by the boundary SVD.
    TA = oe.contract(
        "im,iyj,qMj,qYg,Yyp->Mmpg",
        F12, E, B, C, P_out.reshape(D_squared, D_squared, chi),
        optimize=[(0,1),(0,1),(0,1),(0,1)], backend='torch'
    ).reshape(n0, chi*D_squared)

    U, S, V = truncated_svd_propack(
        torch.mm(TB.T, TA), chi,
        chi_extra=round(2*np.sqrt(D_squared)), rel_cutoff=1e-12, v0=None,
        keep_multiplets=False, abs_tol=1e-14, eps_multiplet=1e-12)
    sqrtInvTruncS = _safe_sqrt_inv_diag(S[:chi])
    TB_old = TB
    T1F = torch.mm((torch.mm(TA, torch.mm(V, sqrtInvTruncS))).T, TB)
    T2A = torch.mm(torch.mm(torch.mm(sqrtInvTruncS, U.conj().T), TB_old.T), TA)

    T1F = normalize_tensor(T1F.reshape(chi, chi, D_squared))
    T2A = normalize_tensor(T2A.reshape(chi, chi, D_squared))
    return C21CD, T1F, T2A











def check_env_CV_C3(lastC21CD, nowC21CD,
                    lastC21EB, nowC21EB,
                    lastC21AF, nowC21AF,
                    env_conv_threshold):
    """Convergence test for the three C3 corner representatives.

    The old three cyclic one-cut density matrices of each environment all
    reduce to ``C @ C @ C``.  We therefore evaluate one per environment.
    """
    max_delta = lastC21CD.new_tensor(0.0).real
    now_spectra = []
    for lastC, nowC in ((lastC21CD, nowC21CD),
                        (lastC21EB, nowC21EB),
                        (lastC21AF, nowC21AF)):
        last_rho = torch.mm(torch.mm(lastC, lastC), lastC)
        now_rho  = torch.mm(torch.mm(nowC,  nowC),  nowC)
        sv_last = torch.linalg.svdvals(last_rho).real
        sv_now  = torch.linalg.svdvals(now_rho).real
        now_spectra.append(sv_now)
        sv_last = sv_last / (sv_last[0] + 1e-30)
        sv_now  = sv_now  / (sv_now[0] + 1e-30)
        sv_weight = (sv_now + sv_last) * 0.5
        max_delta = torch.maximum(max_delta,
                                  ((sv_now - sv_last).abs() * sv_weight).max())

    _ZERO_NORM_THR = 1e-15
    _FLAT_SPEC_THR = 0.85
    all_bad = True
    for sv in now_spectra:
        s0 = sv[0].item()
        if s0 < _ZERO_NORM_THR:
            continue
        if (sv[-1] / (sv[0] + 1e-30)).item() < _FLAT_SPEC_THR:
            all_bad = False
            break
    if all_bad:
        return None
    return (max_delta < env_conv_threshold).item()






def update_environmentCTs_1to2_C3(C21CD, T1F, T2A,
                                   A, B, C, D, E, F, chi, D_squared):
    """C3-symmetric update ``env1 -> env2``.

    Inputs are the three representatives of env1.  The returned tensors are
    exactly ``C21EB, T1D, T2C``; no rotated copies are constructed.
    """
    C21EB_grown = oe.contract(
        "YX,MYa,LXb,amg,lbg->MmLl", C21CD, T1F, T2A, E, B,
        optimize=[(0,1),(1,3),(0,1),(0,1)], backend='torch'
    ).reshape(chi*D_squared, chi*D_squared)

    P_in, C21EB, P_out = trunc_rhoC3(C21EB_grown, chi, D_squared)
    V = P_in.reshape(chi, chi, D_squared)
    U = P_out.reshape(chi, D_squared, chi)

    # T3C belongs to the T2A orbit; T3D belongs to the T1F orbit.
    T1D = oe.contract("OYg,abg,MOb->YMa", T2A, D, V,
                      optimize=[(0,1),(0,1)], backend='torch')
    T2C = oe.contract("OXg,abg,OaL->XLb", T1F, C, U,
                      optimize=[(0,1),(0,1)], backend='torch')
    return C21EB, normalize_tensor(T1D), normalize_tensor(T2C)







def update_environmentCTs_2to3_C3(C21EB, T1D, T2C,
                                   A, B, C, D, E, F, chi, D_squared):
    """C3-symmetric update ``env2 -> env3`` returning ``C21AF,T1B,T2E``."""
    C21AF_grown = oe.contract(
        "YX,MYa,LXb,amg,lbg->MmLl", C21EB, T1D, T2C, A, F,
        optimize=[(0,1),(1,3),(0,1),(0,1)], backend='torch'
    ).reshape(chi*D_squared, chi*D_squared)

    P_in, C21AF, P_out = trunc_rhoC3(C21AF_grown, chi, D_squared)
    V = P_in.reshape(chi, chi, D_squared)
    U = P_out.reshape(chi, D_squared, chi)

    # T3E belongs to the T2C orbit; T3B belongs to the T1D orbit.
    T1B = oe.contract("OYg,abg,MOb->YMa", T2C, B, V,
                      optimize=[(0,1),(0,1)], backend='torch')
    T2E = oe.contract("OXg,abg,OaL->XLb", T1D, E, U,
                      optimize=[(0,1),(0,1)], backend='torch')
    return C21AF, normalize_tensor(T1B), normalize_tensor(T2E)







def update_environmentCTs_3to1_C3(C21AF, T1B, T2E,
                                   A, B, C, D, E, F, chi, D_squared):
    """C3-symmetric update ``env3 -> env1`` returning ``C21CD,T1F,T2A``."""
    C21CD_grown = oe.contract(
        "YX,MYa,LXb,amg,lbg->MmLl", C21AF, T1B, T2E, C, D,
        optimize=[(0,1),(1,3),(0,1),(0,1)], backend='torch'
    ).reshape(chi*D_squared, chi*D_squared)

    P_in, C21CD, P_out = trunc_rhoC3(C21CD_grown, chi, D_squared)
    V = P_in.reshape(chi, chi, D_squared)
    U = P_out.reshape(chi, D_squared, chi)

    # T3A belongs to the T2E orbit; T3F belongs to the T1B orbit.
    T1F = oe.contract("OYg,abg,MOb->YMa", T2E, F, V,
                      optimize=[(0,1),(0,1)], backend='torch')
    T2A = oe.contract("OXg,abg,OaL->XLb", T1B, A, U,
                      optimize=[(0,1),(0,1)], backend='torch')
    return C21CD, normalize_tensor(T1F), normalize_tensor(T2A)






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


def CTMRG_from_init_to_stop(A, B, C, D, E, F,
                            chi: int,
                            D_squared: int,
                            a_third_max_iterations: int,
                            env_conv_threshold: float,
                            identity_init: bool = False,
                            energy_proxy_fn=None) -> tuple:
    """Run C3-symmetric CTMRG with exactly 3 corners and 6 edges.

    Return order is
    ``C21CD,T1F,T2A,C21EB,T1D,T2C,C21AF,T1B,T2E,ctm_steps``.
    One loop iteration performs the complete ``1->2->3->1`` cycle.
    """
    _ZERO_COLLAPSE_THR = 1e-20
    _NOISE_SCALES = (0.0, 0.1, 1.0, 10.0)

    for restart, noise_scale in enumerate(_NOISE_SCALES):
        use_identity = identity_init if restart == 0 else True
        with torch.no_grad():
            D_bond = round(D_squared ** 0.5)
            if D_bond >= 7 and not use_identity:
                use_identity = True
            nowC21CD, nowT1F, nowT2A = initialize_envCTs_C3(
                A, B, C, D, E, F, chi, D_squared, identity_init=use_identity)
            if noise_scale > 0.0:
                for tensor in (nowC21CD, nowT1F, nowT2A):
                    scale = torch.linalg.norm(tensor).clamp(min=1.0).item()
                    tensor.add_(noise_scale * scale * torch.randn_like(tensor))
                    tensor.copy_(normalize_tensor(tensor))

        nowC21EB = nowT1D = nowT2C = None
        nowC21AF = nowT1B = nowT2E = None
        lastC21CD = lastC21EB = lastC21AF = None
        collapsed = False
        ctm_steps = a_third_max_iterations
        last_e_proxy = None
        sv_met = False
        effective_mode = 'SVdifference' if energy_proxy_fn is None else _CTM_CONV_MODE

        for iteration in range(a_third_max_iterations):
            if not (lastC21CD is None or lastC21EB is None or lastC21AF is None):
                cv = check_env_CV_C3(
                    lastC21CD.detach(), nowC21CD.detach(),
                    lastC21EB.detach(), nowC21EB.detach(),
                    lastC21AF.detach(), nowC21AF.detach(),
                    env_conv_threshold)
                if cv is None:
                    collapsed = True
                    break
                if effective_mode == 'SVdifference' and cv:
                    ctm_steps = iteration + 1
                    break
                if effective_mode == 'both':
                    sv_met = sv_met or bool(cv)
                should_check_energy = (
                    energy_proxy_fn is not None
                    and effective_mode != 'SVdifference'
                    and (effective_mode != 'both' or sv_met)
                )
                if should_check_energy:
                    curr_e = energy_proxy_fn(
                        nowC21CD, nowT1F, nowT2A,
                        nowC21EB, nowT1D, nowT2C,
                        nowC21AF, nowT1B, nowT2E)
                    e_met = (last_e_proxy is not None and
                             abs(curr_e - last_e_proxy) < _CTM_E_CONV_THRESHOLD)
                    last_e_proxy = curr_e
                    if effective_mode == 'Edifference' and e_met:
                        ctm_steps = iteration + 1
                        break
                    if effective_mode == 'both' and sv_met and e_met:
                        ctm_steps = iteration + 1
                        break

            lastC21EB = nowC21EB
            nowC21EB, nowT1D, nowT2C = _ckpt(
                update_environmentCTs_1to2_C3,
                nowC21CD, nowT1F, nowT2A,
                A, B, C, D, E, F, chi, D_squared,
                use_reentrant=False)

            lastC21AF = nowC21AF
            nowC21AF, nowT1B, nowT2E = _ckpt(
                update_environmentCTs_2to3_C3,
                nowC21EB, nowT1D, nowT2C,
                A, B, C, D, E, F, chi, D_squared,
                use_reentrant=False)

            lastC21CD = nowC21CD
            nowC21CD, nowT1F, nowT2A = _ckpt(
                update_environmentCTs_3to1_C3,
                nowC21AF, nowT1B, nowT2E,
                A, B, C, D, E, F, chi, D_squared,
                use_reentrant=False)

            if (torch.linalg.norm(nowC21CD).item() < _ZERO_COLLAPSE_THR and
                torch.linalg.norm(nowC21EB).item() < _ZERO_COLLAPSE_THR and
                torch.linalg.norm(nowC21AF).item() < _ZERO_COLLAPSE_THR):
                collapsed = True
                break

        if not collapsed:
            break

    return (nowC21CD, nowT1F, nowT2A,
            nowC21EB, nowT1D, nowT2C,
            nowC21AF, nowT1B, nowT2E,
            ctm_steps)












def _legacy_energy_expectation_nearest_neighbor_3ebadcf_bonds(
                a,b,c,d,e,f, 
                Jeb,Jad,Jcf,
                Jfa,Jde,Jbc,
                Jae,Jec,Jca,Jdb,Jbf,Jfd,
                SdotS,
                chi, D_bond, d_PHYS, 
                C21CD,T1F,T2A):
    """Energy for 3 ebadcf bonds.

    Memory strategy (GPU only):
      6 opens < 15% GPU : pre-build all 6 once, _ckpt per bond.
        (D≤7 on 32 GB: 6 opens ≤ 2.7 GB = 8.6% — pre-build, 5× faster.)
      6 opens ≥ 15% GPU : lazy-open closures, at most 2 opens alive at once.
        (D≥8 on 32 GB: 6 opens ≥ 6.1 GB = 19% — lazy required to fit backward.)
    CPU: no checkpoint, no closure — pre-build all 6 once, direct calls.
    """
    _on_gpu = DEVICE.type == 'cuda'
    _D2 = chi * D_bond * D_bond
    _use_lazy = _on_gpu and (
        6 * _D2 * _D2 * d_PHYS * d_PHYS * a.element_size()
        > torch.cuda.get_device_properties(DEVICE).total_memory * 0.15
    ) if _on_gpu else False

    if _CACHE_RHOS:
        _RHO_ACC.clear()

    # ── GPU + large-D: lazy-open closures + _ckpt ────────────────────────
    if _use_lazy:
        T1F_r = T1F.reshape(chi,chi,D_bond,D_bond)
        T2A_r = T2A.reshape(chi,chi,D_bond,D_bond)

        D2 = chi * D_bond * D_bond
        _bop = lambda einstr, *args, opt: oe.contract(einstr, *args,
                                                       optimize=opt, backend="torch")
        # Build closed tensors from cheaply-constructed opens (freed immediately)
        _open_E = _bop("YX,MYar,abci,rstj->MbsXctij", C21CD, T1F_r, e, e.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
        closed_E = oe.contract("MXii->MX", _open_E, backend="torch"); del _open_E
        _open_D = _bop("MYct,abci,rstj->YarMbsij",    T2A_r, d, d.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
        closed_D = oe.contract("YMii->YM", _open_D, backend="torch"); del _open_D
        _open_A = _bop("ZY,NZbs,abci,rstj->NctYarij", C21CD, T1F_r, a, a.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
        closed_A = oe.contract("NYii->NY", _open_A, backend="torch"); del _open_A
        _open_F = _bop("NZar,abci,rstj->ZbsNctij",    T2A_r, f, f.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
        closed_F = oe.contract("ZNii->ZN", _open_F, backend="torch"); del _open_F
        _open_C = _bop("XZ,LXct,abci,rstj->LarZbsij", C21CD, T1F_r, c, c.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
        closed_C = oe.contract("LZii->LZ", _open_C, backend="torch"); del _open_C
        _open_B = _bop("LXbs,abci,rstj->XctLarij",    T2A_r, b, b.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
        closed_B = oe.contract("XLii->XL", _open_B, backend="torch"); del _open_B

        # ── NN bond closures (pair products computed ON-THE-FLY inside _ckpt
        #    so they are never pinned in saved_tensors; saves 6×(χD²)²) ──
        def _nn_AD(cE, cB, cC, cF, SdotS):
            _oA = _bop("ZY,NZbs,abci,rstj->NctYarij", C21CD, T1F_r, a, a.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
            _oD = _bop("MYct,abci,rstj->YarMbsij",    T2A_r, d, d.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
            return _compute_nn_bond_energy(_oA, _oD, torch.mm(cE, cB), torch.mm(cC, cF), SdotS)
        def _nn_CF(cA, cD, cE, cB, SdotS):
            _oC = _bop("XZ,LXct,abci,rstj->LarZbsij", C21CD, T1F_r, c, c.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
            _oF = _bop("NZar,abci,rstj->ZbsNctij",    T2A_r, f, f.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
            return _compute_nn_bond_energy(_oC, _oF, torch.mm(cA, cD), torch.mm(cE, cB), SdotS)
        def _nn_EB(cC, cF, cA, cD, SdotS):
            _oE = _bop("YX,MYar,abci,rstj->MbsXctij", C21CD, T1F_r, e, e.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
            _oB = _bop("LXbs,abci,rstj->XctLarij",    T2A_r, b, b.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
            return _compute_nn_bond_energy(_oE, _oB, torch.mm(cC, cF), torch.mm(cA, cD), SdotS)
        def _nn_FA(cD, cE, cB, cC, SdotS):
            _oF = _bop("NZar,abci,rstj->ZbsNctij",    T2A_r, f, f.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
            _oA = _bop("ZY,NZbs,abci,rstj->NctYarij", C21CD, T1F_r, a, a.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
            return _compute_nn_bond_energy(_oF, _oA, torch.mm(cD, cE), torch.mm(cB, cC), SdotS)
        def _nn_DE(cB, cC, cF, cA, SdotS):
            _oD = _bop("MYct,abci,rstj->YarMbsij",    T2A_r, d, d.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
            _oE = _bop("YX,MYar,abci,rstj->MbsXctij", C21CD, T1F_r, e, e.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
            return _compute_nn_bond_energy(_oD, _oE, torch.mm(cB, cC), torch.mm(cF, cA), SdotS)
        def _nn_BC(cF, cA, cD, cE, SdotS):
            _oB = _bop("LXbs,abci,rstj->XctLarij",    T2A_r, b, b.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
            _oC = _bop("XZ,LXct,abci,rstj->LarZbsij", C21CD, T1F_r, c, c.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
            return _compute_nn_bond_energy(_oB, _oC, torch.mm(cF, cA), torch.mm(cD, cE), SdotS)

        # ── NNN bond closures (sequential open builds: peak = 2·open+W, not 4·open) ──
        def _nnn_AE(cD, cB, cC, cF, SdotS):
            return _compute_nnn_bond_energy_seq(
                lambda: _bop("ZY,NZbs,abci,rstj->NctYarij", C21CD, T1F_r, a, a.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS),
                cD,
                lambda: _bop("YX,MYar,abci,rstj->MbsXctij", C21CD, T1F_r, e, e.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS),
                cB, torch.mm(cC, cF), SdotS)
        def _nnn_EC(cB, cF, cA, cD, SdotS):
            return _compute_nnn_bond_energy_seq(
                lambda: _bop("YX,MYar,abci,rstj->MbsXctij", C21CD, T1F_r, e, e.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS),
                cB,
                lambda: _bop("XZ,LXct,abci,rstj->LarZbsij", C21CD, T1F_r, c, c.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS),
                cF, torch.mm(cA, cD), SdotS)
        def _nnn_CA(cF, cD, cE, cB, SdotS):
            return _compute_nnn_bond_energy_seq(
                lambda: _bop("XZ,LXct,abci,rstj->LarZbsij", C21CD, T1F_r, c, c.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS),
                cF,
                lambda: _bop("ZY,NZbs,abci,rstj->NctYarij", C21CD, T1F_r, a, a.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS),
                cD, torch.mm(cE, cB), SdotS)
        def _nnn_DB(cE, cC, cF, cA, SdotS):
            return _compute_nnn_bond_energy_seq(
                lambda: _bop("MYct,abci,rstj->YarMbsij",    T2A_r, d, d.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS),
                cE,
                lambda: _bop("LXbs,abci,rstj->XctLarij",    T2A_r, b, b.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS),
                cC, torch.mm(cF, cA), SdotS)
        def _nnn_BF(cC, cA, cD, cE, SdotS):
            return _compute_nnn_bond_energy_seq(
                lambda: _bop("LXbs,abci,rstj->XctLarij",    T2A_r, b, b.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS),
                cC,
                lambda: _bop("NZar,abci,rstj->ZbsNctij",    T2A_r, f, f.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS),
                cA, torch.mm(cD, cE), SdotS)
        def _nnn_FD(cA, cE, cB, cC, SdotS):
            return _compute_nnn_bond_energy_seq(
                lambda: _bop("NZar,abci,rstj->ZbsNctij",    T2A_r, f, f.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS),
                cA,
                lambda: _bop("MYct,abci,rstj->YarMbsij",    T2A_r, d, d.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS),
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

        if _CACHE_RHOS:
            # NN order: AD, CF, EB, FA, DE, BC → _r[0..5]
            # NNN order: AE, EC, CA, DB, BF, FD → _r[6..11]
            # Mag dict: one-body rho from the NN bond containing that site.
            #   AD bond (_r[0]): A=idx0, D=idx1
            #   CF bond (_r[1]): C=idx0, F=idx1
            #   EB bond (_r[2]): E=idx0, B=idx1
            _r = list(_RHO_ACC)  # copy for safety
            _RHO_CACHE['env1'] = (
                _r,
                {'A': (_r[0], 0), 'D': (_r[0], 1),
                 'C': (_r[1], 0), 'F': (_r[1], 1),
                 'E': (_r[2], 0), 'B': (_r[2], 1)})
        return torch.real((E_AD+E_CF+E_EB + E_FA+E_DE+E_BC)*0.5 +E_AE+E_EC+E_CA +E_DB+E_BF+E_FD)

    # ── Pre-build path: CPU (any D) or GPU (6 opens fit in memory) ────────
    o, cl = build_open_closed_env1(a, b, c, d, e, f, chi, D_bond, d_PHYS,
                                   C21CD, T1F, T2A)
    AD = torch.mm(cl['A'], cl['D']);  CF = torch.mm(cl['C'], cl['F'])
    EB = torch.mm(cl['E'], cl['B']);  FA = torch.mm(cl['F'], cl['A'])
    DE = torch.mm(cl['D'], cl['E']);  BC = torch.mm(cl['B'], cl['C'])

    if _on_gpu:
        # GPU pre-build: sub-checkpoint each bond (saves ~9 GB backward memory)
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


    if _CACHE_RHOS:
        # NN order: AD, CF, EB, FA, DE, BC → _r[0..5]
        # NNN order: AE, EC, CA, DB, BF, FD → _r[6..11]
        _r = list(_RHO_ACC)  # copy for safety
        _RHO_CACHE['env1'] = (
            _r,
            {'A': (_r[0], 0), 'D': (_r[0], 1),
             'C': (_r[1], 0), 'F': (_r[1], 1),
             'E': (_r[2], 0), 'B': (_r[2], 1)})
    return torch.real((E_AD+E_CF+E_EB + E_FA+E_DE+E_BC)*0.5 +E_AE+E_EC+E_CA +E_DB+E_BF+E_FD)




def _legacy_energy_expectation_nearest_neighbor_3afcbed_bonds(a,b,c,d,e,f,
                Jaf,Jcb,Jed, 
                Jdc,Jba,Jfe,
                Jca,Jae,Jec,Jbf,Jfd,Jdb,
                SdotS,
                chi, D_bond, d_PHYS, 
                C21EB,T1D,T2C):
    """Energy for 3 afcbed bonds.  GPU D>=8: lazy-open+_ckpt (6 opens>=15% GPU).  CPU: no checkpoint."""
    _on_gpu = DEVICE.type == 'cuda'
    _D2 = chi * D_bond * D_bond
    _use_lazy = _on_gpu and (
        6 * _D2 * _D2 * d_PHYS * d_PHYS * a.element_size()
        > torch.cuda.get_device_properties(DEVICE).total_memory * 0.15
    ) if _on_gpu else False

    if _CACHE_RHOS:
        _RHO_ACC.clear()

    # ── GPU + large-D: lazy-open closures + _ckpt ────────────────────────
    if _use_lazy:
        T1D_r = T1D.reshape(chi,chi,D_bond,D_bond)
        T2C_r = T2C.reshape(chi,chi,D_bond,D_bond)

        D2 = chi * D_bond * D_bond
        _bop = lambda einstr, *args, opt: oe.contract(einstr, *args,
                                                       optimize=opt, backend="torch")
        _open_A = _bop("YX,MYar,abci,rstj->MbsXctij", C21EB, T1D_r, a, a.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
        closed_A = oe.contract("MXii->MX", _open_A, backend="torch"); del _open_A
        _open_B = _bop("MYct,abci,rstj->YarMbsij",    T2C_r, b, b.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
        closed_B = oe.contract("YMii->YM", _open_B, backend="torch"); del _open_B
        _open_C = _bop("ZY,NZbs,abci,rstj->NctYarij", C21EB, T1D_r, c, c.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
        closed_C = oe.contract("NYii->NY", _open_C, backend="torch"); del _open_C
        _open_D = _bop("NZar,abci,rstj->ZbsNctij",    T2C_r, d, d.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
        closed_D = oe.contract("ZNii->ZN", _open_D, backend="torch"); del _open_D
        _open_E = _bop("XZ,LXct,abci,rstj->LarZbsij", C21EB, T1D_r, e, e.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
        closed_E = oe.contract("LZii->LZ", _open_E, backend="torch"); del _open_E
        _open_F = _bop("LXbs,abci,rstj->XctLarij",    T2C_r, f, f.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
        closed_F = oe.contract("XLii->XL", _open_F, backend="torch"); del _open_F

        # ── NN bond closures (pair products computed on-the-fly inside _ckpt) ──
        def _nn_CB(cA, cF, cE, cD, SdotS):
            _oC = _bop("ZY,NZbs,abci,rstj->NctYarij", C21EB, T1D_r, c, c.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
            _oB = _bop("MYct,abci,rstj->YarMbsij",    T2C_r, b, b.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
            return _compute_nn_bond_energy(_oC, _oB, torch.mm(cA, cF), torch.mm(cE, cD), SdotS)
        def _nn_AF(cE, cD, cC, cB, SdotS):
            _oA = _bop("YX,MYar,abci,rstj->MbsXctij", C21EB, T1D_r, a, a.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
            _oF = _bop("LXbs,abci,rstj->XctLarij",    T2C_r, f, f.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
            return _compute_nn_bond_energy(_oA, _oF, torch.mm(cE, cD), torch.mm(cC, cB), SdotS)
        def _nn_ED(cC, cB, cA, cF, SdotS):
            _oE = _bop("XZ,LXct,abci,rstj->LarZbsij", C21EB, T1D_r, e, e.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
            _oD = _bop("NZar,abci,rstj->ZbsNctij",    T2C_r, d, d.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
            return _compute_nn_bond_energy(_oE, _oD, torch.mm(cC, cB), torch.mm(cA, cF), SdotS)
        def _nn_DC(cB, cA, cF, cE, SdotS):
            _oD = _bop("NZar,abci,rstj->ZbsNctij",    T2C_r, d, d.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
            _oC = _bop("ZY,NZbs,abci,rstj->NctYarij", C21EB, T1D_r, c, c.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
            return _compute_nn_bond_energy(_oD, _oC, torch.mm(cB, cA), torch.mm(cF, cE), SdotS)
        def _nn_BA(cF, cE, cD, cC, SdotS):
            _oB = _bop("MYct,abci,rstj->YarMbsij",    T2C_r, b, b.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
            _oA = _bop("YX,MYar,abci,rstj->MbsXctij", C21EB, T1D_r, a, a.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
            return _compute_nn_bond_energy(_oB, _oA, torch.mm(cF, cE), torch.mm(cD, cC), SdotS)
        def _nn_FE(cD, cC, cB, cA, SdotS):
            _oF = _bop("LXbs,abci,rstj->XctLarij",    T2C_r, f, f.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
            _oE = _bop("XZ,LXct,abci,rstj->LarZbsij", C21EB, T1D_r, e, e.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
            return _compute_nn_bond_energy(_oF, _oE, torch.mm(cD, cC), torch.mm(cB, cA), SdotS)

        # ── NNN bond closures (sequential open builds: peak = 2·open+W, not 4·open) ──
        def _nnn_CA(cB, cF, cE, cD, SdotS):
            return _compute_nnn_bond_energy_seq(
                lambda: _bop("ZY,NZbs,abci,rstj->NctYarij", C21EB, T1D_r, c, c.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS),
                cB,
                lambda: _bop("YX,MYar,abci,rstj->MbsXctij", C21EB, T1D_r, a, a.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS),
                cF, torch.mm(cE, cD), SdotS)
        def _nnn_AE(cF, cD, cC, cB, SdotS):
            return _compute_nnn_bond_energy_seq(
                lambda: _bop("YX,MYar,abci,rstj->MbsXctij", C21EB, T1D_r, a, a.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS),
                cF,
                lambda: _bop("XZ,LXct,abci,rstj->LarZbsij", C21EB, T1D_r, e, e.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS),
                cD, torch.mm(cC, cB), SdotS)
        def _nnn_EC(cD, cB, cA, cF, SdotS):
            return _compute_nnn_bond_energy_seq(
                lambda: _bop("XZ,LXct,abci,rstj->LarZbsij", C21EB, T1D_r, e, e.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS),
                cD,
                lambda: _bop("ZY,NZbs,abci,rstj->NctYarij", C21EB, T1D_r, c, c.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS),
                cB, torch.mm(cA, cF), SdotS)
        def _nnn_BF(cA, cE, cD, cC, SdotS):
            return _compute_nnn_bond_energy_seq(
                lambda: _bop("MYct,abci,rstj->YarMbsij",    T2C_r, b, b.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS),
                cA,
                lambda: _bop("LXbs,abci,rstj->XctLarij",    T2C_r, f, f.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS),
                cE, torch.mm(cD, cC), SdotS)
        def _nnn_FD(cE, cC, cB, cA, SdotS):
            return _compute_nnn_bond_energy_seq(
                lambda: _bop("LXbs,abci,rstj->XctLarij",    T2C_r, f, f.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS),
                cE,
                lambda: _bop("NZar,abci,rstj->ZbsNctij",    T2C_r, d, d.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS),
                cC, torch.mm(cB, cA), SdotS)
        def _nnn_DB(cC, cA, cF, cE, SdotS):
            return _compute_nnn_bond_energy_seq(
                lambda: _bop("NZar,abci,rstj->ZbsNctij",    T2C_r, d, d.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS),
                cC,
                lambda: _bop("MYct,abci,rstj->YarMbsij",    T2C_r, b, b.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS),
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
        
        if _CACHE_RHOS:
            _r = list(_RHO_ACC)
            _RHO_CACHE['env2'] = (
                _r,
                {'C': (_r[0], 0), 'B': (_r[0], 1),
                 'A': (_r[1], 0), 'F': (_r[1], 1),
                 'E': (_r[2], 0), 'D': (_r[2], 1)})
        return torch.real((E_AF+E_CB+E_ED + E_DC+E_BA+E_FE)*0.5 +E_CA+E_AE+E_EC +E_BF+E_FD+E_DB)

    # ── Pre-build path: CPU (any D) or GPU (6 opens fit in memory) ────────
    o, cl = build_open_closed_env2(a, b, c, d, e, f, chi, D_bond, d_PHYS,
                                   C21EB, T1D, T2C)
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

    if _CACHE_RHOS:
        _r = list(_RHO_ACC)
        _RHO_CACHE['env2'] = (
            _r,
            {'C': (_r[0], 0), 'B': (_r[0], 1),
                'A': (_r[1], 0), 'F': (_r[1], 1),
                'E': (_r[2], 0), 'D': (_r[2], 1)})
    return torch.real((E_AF+E_CB+E_ED + E_DC+E_BA+E_FE)*0.5 +E_CA+E_AE+E_EC +E_BF+E_FD+E_DB)






def _legacy_energy_expectation_nearest_neighbor_other_3_bonds(a,b,c,d,e,f, 
                    Jcd,Jef,Jab,
                    Jbe,Jfc,Jda,
                    Jec,Jca,Jae,Jfd,Jdb,Jbf,
                    SdotS,
                    chi, D_bond, d_PHYS,
                    C21AF,T1B,T2E):
    """
    Compute the variational energy expectation value for the remaining 3 bonds
    in the type-3 environment.

    This function covers bonds C-D, E-F, A-B (the bonds not handled by
    ``energy_expectation_nearest_neighbor_6_bonds``).  The same open/closed
    tensor construction is used, but the environment here is the type-3 set
    ``(C21AF, T1B, T2E)``; the three rotated placements are contracted
    directly with these representatives.

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
        C21AF (torch.Tensor): Type-3 corner representative, shape ``(chi, chi)``.
        T1B, T2E (torch.Tensor): The two type-3 transfer representatives.

    Returns:
        torch.Tensor: Scalar energy per bond (float32).
    """
    _on_gpu = DEVICE.type == 'cuda'
    _D2 = chi * D_bond * D_bond
    _use_lazy = _on_gpu and (
        6 * _D2 * _D2 * d_PHYS * d_PHYS * a.element_size()
        > torch.cuda.get_device_properties(DEVICE).total_memory * 0.15
    ) if _on_gpu else False

    if _CACHE_RHOS:
        _RHO_ACC.clear()

    # ── GPU + large-D: lazy-open closures + _ckpt ────────────────────────
    if _use_lazy:
        T1B_r = T1B.reshape(chi,chi,D_bond,D_bond)
        T2E_r = T2E.reshape(chi,chi,D_bond,D_bond)

        D2 = chi * D_bond * D_bond
        _bop = lambda einstr, *args, opt: oe.contract(einstr, *args,
                                                       optimize=opt, backend="torch")
        _open_C = _bop("YX,MYar,abci,rstj->MbsXctij", C21AF, T1B_r, c, c.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
        closed_C = oe.contract("MXii->MX", _open_C, backend="torch"); del _open_C
        _open_F = _bop("MYct,abci,rstj->YarMbsij",    T2E_r, f, f.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
        closed_F = oe.contract("YMii->YM", _open_F, backend="torch"); del _open_F
        _open_E = _bop("ZY,NZbs,abci,rstj->NctYarij", C21AF, T1B_r, e, e.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
        closed_E = oe.contract("NYii->NY", _open_E, backend="torch"); del _open_E
        _open_B = _bop("NZar,abci,rstj->ZbsNctij",    T2E_r, b, b.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
        closed_B = oe.contract("ZNii->ZN", _open_B, backend="torch"); del _open_B
        _open_A = _bop("XZ,LXct,abci,rstj->LarZbsij", C21AF, T1B_r, a, a.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
        closed_A = oe.contract("LZii->LZ", _open_A, backend="torch"); del _open_A
        _open_D = _bop("LXbs,abci,rstj->XctLarij",    T2E_r, d, d.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
        closed_D = oe.contract("XLii->XL", _open_D, backend="torch"); del _open_D

        # ── NN closures (pair products computed on-the-fly inside _ckpt) ──
        def _nn_EF(cC, cD, cA, cB, SdotS):
            _oE = _bop("ZY,NZbs,abci,rstj->NctYarij", C21AF, T1B_r, e, e.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
            _oF = _bop("MYct,abci,rstj->YarMbsij",    T2E_r, f, f.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
            return _compute_nn_bond_energy(_oE, _oF, torch.mm(cC, cD), torch.mm(cA, cB), SdotS)
        def _nn_AB(cE, cF, cC, cD, SdotS):
            _oA = _bop("XZ,LXct,abci,rstj->LarZbsij", C21AF, T1B_r, a, a.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
            _oB = _bop("NZar,abci,rstj->ZbsNctij",    T2E_r, b, b.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
            return _compute_nn_bond_energy(_oA, _oB, torch.mm(cE, cF), torch.mm(cC, cD), SdotS)
        def _nn_CD(cA, cB, cE, cF, SdotS):
            _oC = _bop("YX,MYar,abci,rstj->MbsXctij", C21AF, T1B_r, c, c.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
            _oD = _bop("LXbs,abci,rstj->XctLarij",    T2E_r, d, d.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
            return _compute_nn_bond_energy(_oC, _oD, torch.mm(cA, cB), torch.mm(cE, cF), SdotS)
        def _nn_BE(cF, cC, cD, cA, SdotS):
            _oB = _bop("NZar,abci,rstj->ZbsNctij",    T2E_r, b, b.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
            _oE = _bop("ZY,NZbs,abci,rstj->NctYarij", C21AF, T1B_r, e, e.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
            return _compute_nn_bond_energy(_oB, _oE, torch.mm(cF, cC), torch.mm(cD, cA), SdotS)
        def _nn_FC(cD, cA, cB, cE, SdotS):
            _oF = _bop("MYct,abci,rstj->YarMbsij",    T2E_r, f, f.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
            _oC = _bop("YX,MYar,abci,rstj->MbsXctij", C21AF, T1B_r, c, c.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
            return _compute_nn_bond_energy(_oF, _oC, torch.mm(cD, cA), torch.mm(cB, cE), SdotS)
        def _nn_DA(cB, cE, cF, cC, SdotS):
            _oD = _bop("LXbs,abci,rstj->XctLarij",    T2E_r, d, d.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
            _oA = _bop("XZ,LXct,abci,rstj->LarZbsij", C21AF, T1B_r, a, a.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS)
            return _compute_nn_bond_energy(_oD, _oA, torch.mm(cB, cE), torch.mm(cF, cC), SdotS)

        # ── NNN closures (sequential open builds: peak = 2·open+W, not 4·open) ─────
        def _nnn_EC(cF, cD, cA, cB, SdotS):
            return _compute_nnn_bond_energy_seq(
                lambda: _bop("ZY,NZbs,abci,rstj->NctYarij", C21AF, T1B_r, e, e.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS),
                cF,
                lambda: _bop("YX,MYar,abci,rstj->MbsXctij", C21AF, T1B_r, c, c.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS),
                cD, torch.mm(cA, cB), SdotS)
        def _nnn_CA(cD, cB, cE, cF, SdotS):
            return _compute_nnn_bond_energy_seq(
                lambda: _bop("YX,MYar,abci,rstj->MbsXctij", C21AF, T1B_r, c, c.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS),
                cD,
                lambda: _bop("XZ,LXct,abci,rstj->LarZbsij", C21AF, T1B_r, a, a.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS),
                cB, torch.mm(cE, cF), SdotS)
        def _nnn_AE(cB, cF, cC, cD, SdotS):
            return _compute_nnn_bond_energy_seq(
                lambda: _bop("XZ,LXct,abci,rstj->LarZbsij", C21AF, T1B_r, a, a.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS),
                cB,
                lambda: _bop("ZY,NZbs,abci,rstj->NctYarij", C21AF, T1B_r, e, e.conj(), opt=[(0,1),(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS),
                cF, torch.mm(cC, cD), SdotS)
        def _nnn_FD(cC, cA, cB, cE, SdotS):
            return _compute_nnn_bond_energy_seq(
                lambda: _bop("MYct,abci,rstj->YarMbsij",    T2E_r, f, f.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS),
                cC,
                lambda: _bop("LXbs,abci,rstj->XctLarij",    T2E_r, d, d.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS),
                cA, torch.mm(cB, cE), SdotS)
        def _nnn_DB(cA, cE, cF, cC, SdotS):
            return _compute_nnn_bond_energy_seq(
                lambda: _bop("LXbs,abci,rstj->XctLarij",    T2E_r, d, d.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS),
                cA,
                lambda: _bop("NZar,abci,rstj->ZbsNctij",    T2E_r, b, b.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS),
                cE, torch.mm(cF, cC), SdotS)
        def _nnn_BF(cE, cC, cD, cA, SdotS):
            return _compute_nnn_bond_energy_seq(
                lambda: _bop("NZar,abci,rstj->ZbsNctij",    T2E_r, b, b.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS),
                cE,
                lambda: _bop("MYct,abci,rstj->YarMbsij",    T2E_r, f, f.conj(), opt=[(0,1),(0,1)]).reshape(D2,D2,d_PHYS,d_PHYS),
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
        
        if _CACHE_RHOS:
            _r = list(_RHO_ACC)
            _RHO_CACHE['env3'] = (
                _r,
                {'E': (_r[0], 0), 'F': (_r[0], 1),
                 'A': (_r[1], 0), 'B': (_r[1], 1),
                 'C': (_r[2], 0), 'D': (_r[2], 1)})
            
        return torch.real((E_EF+E_AB+E_CD + E_BE+E_FC+E_DA)*0.5 +E_EC+E_CA+E_AE +E_FD+E_DB+E_BF)

    # ── Pre-build path: CPU (any D) or GPU (6 opens fit in memory) ─────────
    o, cl = build_open_closed_env3(a, b, c, d, e, f, chi, D_bond, d_PHYS,
                                   C21AF, T1B, T2E)
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

    if _CACHE_RHOS:
        _r = list(_RHO_ACC)
        _RHO_CACHE['env3'] = (
            _r,
            {'E': (_r[0], 0), 'F': (_r[0], 1),
                'A': (_r[1], 0), 'B': (_r[1], 1),
                'C': (_r[2], 0), 'D': (_r[2], 1)})
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


def _build_nn_rho_seq(build_open_X, build_open_Y, pair1, pair2, d_PHYS):
    """Sequential NN rho: build open_X, compute W, free open_X, then build open_Y.

    Peak memory = W + open_Y (= 2·D2²·d²) instead of 3·D2²·d² from _build_nn_rho
    (which requires both opens alive simultaneously as function arguments).
    For D=11, chi=110: saves ~5.7 GB per NN rho call.

    Args:
        build_open_X: callable () → (D2,D2,d,d) tensor for site X
        build_open_Y: callable () → (D2,D2,d,d) tensor for site Y
        pair1, pair2:  (D2,D2) pair-product matrices
        d_PHYS:        physical dimension d
    """
    closure = torch.mm(pair1, pair2)                                           # (D2,D2) negligible
    oX  = build_open_X()                                                      # (D2,D2,d,d)
    W   = oe.contract("MN,NYij->MYij", closure, oX, backend="torch")          # (D2,D2,d,d)
    del oX                                                                     # free before building oY
    oY  = build_open_Y()                                                   # (D2,D2,d,d)
    rho = oe.contract("MYij,YMkl->ikjl", W, oY, backend="torch")              # (d,d,d,d)
    del W, oY
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
    if _CACHE_RHOS:
        _RHO_ACC.append(rho_nrm.detach())
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
    if _CACHE_RHOS:
        _RHO_ACC.append(rho.detach())
    return oe.contract("ikjl,ijkl->", rho, SdotS, backend="torch")


def _compute_nnn_bond_energy(open_X, closed_Y, open_Z, closed_W,
                             large_pair, SdotS):
    """Compute PSD-normalized nnn bond energy  tr(rho · SdotS).

    Same sub-checkpoint rationale as ``_compute_nn_bond_energy``.
    """
    d_PHYS = open_X.shape[2]
    rho = _build_nnn_rho(open_X, closed_Y, open_Z, closed_W,
                         large_pair, d_PHYS)
    if _CACHE_RHOS:
        _RHO_ACC.append(rho.detach())
    return oe.contract("ikjl,ijkl->", rho, SdotS, backend="torch")


def build_open_closed_env1(a, b, c, d, e, f, chi, D_bond, d_PHYS,
                           C21CD, T1F, T2A):
    """Construct open/closed tensors for environment 1 (ebadcf).

    Returns (opens, closeds) — dicts keyed by site label ('A'..'F').
    """
    T1F = T1F.reshape(chi,chi,D_bond,D_bond)
    T2A = T2A.reshape(chi,chi,D_bond,D_bond)
    D2 = chi*D_bond*D_bond
    o = {}
    o['E'] = oe.contract("YX,MYar,abci,rstj->MbsXctij", C21CD, T1F, e, e.conj(), optimize=[(0,1),(0,1),(0,1)], backend="torch").reshape(D2,D2,d_PHYS,d_PHYS)
    o['D'] = oe.contract("MYct,abci,rstj->YarMbsij", T2A, d, d.conj(), optimize=[(0,1),(0,1)], backend="torch").reshape(D2,D2,d_PHYS,d_PHYS)
    o['A'] = oe.contract("ZY,NZbs,abci,rstj->NctYarij", C21CD, T1F, a, a.conj(), optimize=[(0,1),(0,1),(0,1)], backend="torch").reshape(D2,D2,d_PHYS,d_PHYS)
    o['F'] = oe.contract("NZar,abci,rstj->ZbsNctij", T2A, f, f.conj(), optimize=[(0,1),(0,1)], backend="torch").reshape(D2,D2,d_PHYS,d_PHYS)
    o['C'] = oe.contract("XZ,LXct,abci,rstj->LarZbsij", C21CD, T1F, c, c.conj(), optimize=[(0,1),(0,1),(0,1)], backend="torch").reshape(D2,D2,d_PHYS,d_PHYS)
    o['B'] = oe.contract("LXbs,abci,rstj->XctLarij", T2A, b, b.conj(), optimize=[(0,1),(0,1)], backend="torch").reshape(D2,D2,d_PHYS,d_PHYS)
    cl = {k: oe.contract("ABii->AB", o[k], backend="torch") for k in o}
    return o, cl




def build_open_closed_env2(a, b, c, d, e, f, chi, D_bond, d_PHYS,
                           C21EB, T1D, T2C):
    """Construct open/closed tensors for environment 2 (afcbed)."""
    T1D = T1D.reshape(chi,chi,D_bond,D_bond)
    T2C = T2C.reshape(chi,chi,D_bond,D_bond)
    D2 = chi*D_bond*D_bond
    o = {}
    o['A'] = oe.contract("YX,MYar,abci,rstj->MbsXctij", C21EB, T1D, a, a.conj(), optimize=[(0,1),(0,1),(0,1)], backend="torch").reshape(D2,D2,d_PHYS,d_PHYS)
    o['B'] = oe.contract("MYct,abci,rstj->YarMbsij", T2C, b, b.conj(), optimize=[(0,1),(0,1)], backend="torch").reshape(D2,D2,d_PHYS,d_PHYS)
    o['C'] = oe.contract("ZY,NZbs,abci,rstj->NctYarij", C21EB, T1D, c, c.conj(), optimize=[(0,1),(0,1),(0,1)], backend="torch").reshape(D2,D2,d_PHYS,d_PHYS)
    o['D'] = oe.contract("NZar,abci,rstj->ZbsNctij", T2C, d, d.conj(), optimize=[(0,1),(0,1)], backend="torch").reshape(D2,D2,d_PHYS,d_PHYS)
    o['E'] = oe.contract("XZ,LXct,abci,rstj->LarZbsij", C21EB, T1D, e, e.conj(), optimize=[(0,1),(0,1),(0,1)], backend="torch").reshape(D2,D2,d_PHYS,d_PHYS)
    o['F'] = oe.contract("LXbs,abci,rstj->XctLarij", T2C, f, f.conj(), optimize=[(0,1),(0,1)], backend="torch").reshape(D2,D2,d_PHYS,d_PHYS)
    cl = {k: oe.contract("ABii->AB", o[k], backend="torch") for k in o}
    return o, cl




def build_open_closed_env3(a, b, c, d, e, f, chi, D_bond, d_PHYS,
                           C21AF, T1B, T2E):
    """Construct open/closed tensors for environment 3 (cdefab)."""
    T1B = T1B.reshape(chi,chi,D_bond,D_bond)
    T2E = T2E.reshape(chi,chi,D_bond,D_bond)
    D2 = chi*D_bond*D_bond
    o = {}
    o['C'] = oe.contract("YX,MYar,abci,rstj->MbsXctij", C21AF, T1B, c, c.conj(), optimize=[(0,1),(0,1),(0,1)], backend="torch").reshape(D2,D2,d_PHYS,d_PHYS)
    o['F'] = oe.contract("MYct,abci,rstj->YarMbsij", T2E, f, f.conj(), optimize=[(0,1),(0,1)], backend="torch").reshape(D2,D2,d_PHYS,d_PHYS)
    o['E'] = oe.contract("ZY,NZbs,abci,rstj->NctYarij", C21AF, T1B, e, e.conj(), optimize=[(0,1),(0,1),(0,1)], backend="torch").reshape(D2,D2,d_PHYS,d_PHYS)
    o['B'] = oe.contract("NZar,abci,rstj->ZbsNctij", T2E, b, b.conj(), optimize=[(0,1),(0,1)], backend="torch").reshape(D2,D2,d_PHYS,d_PHYS)
    o['A'] = oe.contract("XZ,LXct,abci,rstj->LarZbsij", C21AF, T1B, a, a.conj(), optimize=[(0,1),(0,1),(0,1)], backend="torch").reshape(D2,D2,d_PHYS,d_PHYS)
    o['D'] = oe.contract("LXbs,abci,rstj->XctLarij", T2E, d, d.conj(), optimize=[(0,1),(0,1)], backend="torch").reshape(D2,D2,d_PHYS,d_PHYS)
    cl = {k: oe.contract("ABii->AB", o[k], backend="torch") for k in o}
    return o, cl




def build_single_open_env1(site: str, a, b, c, d, e, f, chi, D_bond, d_PHYS,
                          C21CD, T1F, T2A):
    """Return the open tensor for ONE site in environment 1 (ebadcf).

    Memory-efficient single-site version of build_open_closed_env1.
    Used by evaluate_observables on GPU+D>=8 so that only 2 open tensors
    (~chi^2 D^4 d^2) are ever alive simultaneously instead of all 6.
    """
    T1F = T1F.reshape(chi, chi, D_bond, D_bond)
    T2A = T2A.reshape(chi, chi, D_bond, D_bond)
    D2 = chi * D_bond * D_bond
    if site == 'E':
        return oe.contract("YX,MYar,abci,rstj->MbsXctij", C21CD, T1F, e, e.conj(), optimize=[(0,1),(0,1),(0,1)], backend="torch").reshape(D2, D2, d_PHYS, d_PHYS)
    if site == 'D':
        return oe.contract("MYct,abci,rstj->YarMbsij", T2A, d, d.conj(), optimize=[(0,1),(0,1)], backend="torch").reshape(D2, D2, d_PHYS, d_PHYS)
    if site == 'A':
        return oe.contract("ZY,NZbs,abci,rstj->NctYarij", C21CD, T1F, a, a.conj(), optimize=[(0,1),(0,1),(0,1)], backend="torch").reshape(D2, D2, d_PHYS, d_PHYS)
    if site == 'F':
        return oe.contract("NZar,abci,rstj->ZbsNctij", T2A, f, f.conj(), optimize=[(0,1),(0,1)], backend="torch").reshape(D2, D2, d_PHYS, d_PHYS)
    if site == 'C':
        return oe.contract("XZ,LXct,abci,rstj->LarZbsij", C21CD, T1F, c, c.conj(), optimize=[(0,1),(0,1),(0,1)], backend="torch").reshape(D2, D2, d_PHYS, d_PHYS)
    if site == 'B':
        return oe.contract("LXbs,abci,rstj->XctLarij", T2A, b, b.conj(), optimize=[(0,1),(0,1)], backend="torch").reshape(D2, D2, d_PHYS, d_PHYS)
    raise ValueError(f"Unknown site '{site}' for env1; expected one of A-F")




def build_single_open_env2(site: str, a, b, c, d, e, f, chi, D_bond, d_PHYS,
                          C21EB, T1D, T2C):
    """Return the open tensor for ONE site in environment 2 (afcbed)."""
    T1D = T1D.reshape(chi, chi, D_bond, D_bond)
    T2C = T2C.reshape(chi, chi, D_bond, D_bond)
    D2 = chi * D_bond * D_bond
    if site == 'A':
        return oe.contract("YX,MYar,abci,rstj->MbsXctij", C21EB, T1D, a, a.conj(), optimize=[(0,1),(0,1),(0,1)], backend="torch").reshape(D2, D2, d_PHYS, d_PHYS)
    if site == 'B':
        return oe.contract("MYct,abci,rstj->YarMbsij", T2C, b, b.conj(), optimize=[(0,1),(0,1)], backend="torch").reshape(D2, D2, d_PHYS, d_PHYS)
    if site == 'C':
        return oe.contract("ZY,NZbs,abci,rstj->NctYarij", C21EB, T1D, c, c.conj(), optimize=[(0,1),(0,1),(0,1)], backend="torch").reshape(D2, D2, d_PHYS, d_PHYS)
    if site == 'D':
        return oe.contract("NZar,abci,rstj->ZbsNctij", T2C, d, d.conj(), optimize=[(0,1),(0,1)], backend="torch").reshape(D2, D2, d_PHYS, d_PHYS)
    if site == 'E':
        return oe.contract("XZ,LXct,abci,rstj->LarZbsij", C21EB, T1D, e, e.conj(), optimize=[(0,1),(0,1),(0,1)], backend="torch").reshape(D2, D2, d_PHYS, d_PHYS)
    if site == 'F':
        return oe.contract("LXbs,abci,rstj->XctLarij", T2C, f, f.conj(), optimize=[(0,1),(0,1)], backend="torch").reshape(D2, D2, d_PHYS, d_PHYS)
    raise ValueError(f"Unknown site '{site}' for env2; expected one of A-F")




def build_single_open_env3(site: str, a, b, c, d, e, f, chi, D_bond, d_PHYS,
                          C21AF, T1B, T2E):
    """Return the open tensor for ONE site in environment 3 (cdefab)."""
    T1B = T1B.reshape(chi, chi, D_bond, D_bond)
    T2E = T2E.reshape(chi, chi, D_bond, D_bond)
    D2 = chi * D_bond * D_bond
    if site == 'C':
        return oe.contract("YX,MYar,abci,rstj->MbsXctij", C21AF, T1B, c, c.conj(), optimize=[(0,1),(0,1),(0,1)], backend="torch").reshape(D2, D2, d_PHYS, d_PHYS)
    if site == 'F':
        return oe.contract("MYct,abci,rstj->YarMbsij", T2E, f, f.conj(), optimize=[(0,1),(0,1)], backend="torch").reshape(D2, D2, d_PHYS, d_PHYS)
    if site == 'E':
        return oe.contract("ZY,NZbs,abci,rstj->NctYarij", C21AF, T1B, e, e.conj(), optimize=[(0,1),(0,1),(0,1)], backend="torch").reshape(D2, D2, d_PHYS, d_PHYS)
    if site == 'B':
        return oe.contract("NZar,abci,rstj->ZbsNctij", T2E, b, b.conj(), optimize=[(0,1),(0,1)], backend="torch").reshape(D2, D2, d_PHYS, d_PHYS)
    if site == 'A':
        return oe.contract("XZ,LXct,abci,rstj->LarZbsij", C21AF, T1B, a, a.conj(), optimize=[(0,1),(0,1),(0,1)], backend="torch").reshape(D2, D2, d_PHYS, d_PHYS)
    if site == 'D':
        return oe.contract("LXbs,abci,rstj->XctLarij", T2E, d, d.conj(), optimize=[(0,1),(0,1)], backend="torch").reshape(D2, D2, d_PHYS, d_PHYS)
    raise ValueError(f"Unknown site '{site}' for env3; expected one of A-F")




def _c3_representative_energy(
        build_even, build_odd,
        closed_even: torch.Tensor, closed_odd: torch.Tensor,
        nn_even_odd_couplings, nn_odd_even_couplings,
        nnn_even_couplings, nnn_odd_couplings,
        SdotS: torch.Tensor, cache_key: str) -> torch.Tensor:
    """Evaluate four C3 bond-orbit representatives with streamed opens.

    The public cache still contains 12 entries per environment.  Three entries
    in each C3 orbit alias the same tiny detached rho; no large tensor is copied.
    Closed representatives are built directly by the caller, so the peak here
    is the sequential rho contraction rather than six resident open tensors.
    """
    def _nn_eo(c_even, c_odd, spin_op):
        pair = c_even @ c_odd
        rho = _build_nn_rho_seq(
            build_even, build_odd, pair, pair, spin_op.shape[0])
        energy = oe.contract("ikjl,ijkl->", rho, spin_op, backend="torch")
        return energy, rho.detach()

    def _nn_oe(c_even, c_odd, spin_op):
        pair = c_odd @ c_even
        rho = _build_nn_rho_seq(
            build_odd, build_even, pair, pair, spin_op.shape[0])
        energy = oe.contract("ikjl,ijkl->", rho, spin_op, backend="torch")
        return energy, rho.detach()

    def _nnn_even(c_even, c_odd, spin_op):
        rho = _build_nnn_rho_seq(
            build_even, c_odd, build_even, c_odd,
            c_even @ c_odd, spin_op.shape[0])
        energy = oe.contract("ikjl,ijkl->", rho, spin_op, backend="torch")
        return energy, rho.detach()

    def _nnn_odd(c_even, c_odd, spin_op):
        rho = _build_nnn_rho_seq(
            build_odd, c_even, build_odd, c_even,
            c_odd @ c_even, spin_op.shape[0])
        energy = oe.contract("ikjl,ijkl->", rho, spin_op, backend="torch")
        return energy, rho.detach()

    def _run(fn):
        if DEVICE.type == 'cuda' and torch.is_grad_enabled():
            return _ckpt(
                fn, closed_even, closed_odd, SdotS,
                use_reentrant=False)
        return fn(closed_even, closed_odd, SdotS)

    e_eo, rho_eo = _run(_nn_eo)
    e_oe, rho_oe = _run(_nn_oe)
    e_even, rho_even = _run(_nnn_even)
    e_odd, rho_odd = _run(_nnn_odd)

    if _CACHE_RHOS:
        # Only 4*d^4 scalars cross devices when an environment was evaluated
        # on a secondary GPU.
        r_eo = rho_eo.to(DEVICE)
        r_oe = rho_oe.to(DEVICE)
        r_even = rho_even.to(DEVICE)
        r_odd = rho_odd.to(DEVICE)
        rhos = ([r_eo] * 3 + [r_oe] * 3
                + [r_even] * 3 + [r_odd] * 3)
        mag = {
            'A': (r_eo, 0), 'C': (r_eo, 0), 'E': (r_eo, 0),
            'B': (r_eo, 1), 'D': (r_eo, 1), 'F': (r_eo, 1),
        }
        _RHO_CACHE[cache_key] = (rhos, mag)

    energy = (
        0.5 * (sum(nn_even_odd_couplings) * e_eo
               + sum(nn_odd_even_couplings) * e_oe)
        + sum(nnn_even_couplings) * e_even
        + sum(nnn_odd_couplings) * e_odd
    )
    return torch.real(energy)


def energy_expectation_nearest_neighbor_3ebadcf_bonds(
        a, b, c, d, e, f,
        Jeb, Jad, Jcf, Jfa, Jde, Jbc,
        Jae, Jec, Jca, Jdb, Jbf, Jfd,
        SdotS, chi, D_bond, d_PHYS, C21CD, T1F, T2A):
    """C3-reduced env1 energy: 12 bond contractions become 4."""
    t1 = T1F.reshape(chi, chi, D_bond, D_bond)
    t2 = T2A.reshape(chi, chi, D_bond, D_bond)
    d2 = chi * D_bond * D_bond

    def build_even():
        return oe.contract(
            "ZY,NZbs,abci,rstj->NctYarij", C21CD, t1, a, a.conj(),
            optimize=[(0,1),(0,1),(0,1)], backend="torch"
        ).reshape(d2, d2, d_PHYS, d_PHYS)

    def build_odd():
        return oe.contract(
            "MYct,abci,rstj->YarMbsij", t2, d, d.conj(),
            optimize=[(0,1),(0,1)], backend="torch"
        ).reshape(d2, d2, d_PHYS, d_PHYS)

    closed_even = oe.contract(
        "ZY,NZbs,abci,rsti->NctYar", C21CD, t1, a, a.conj(),
        optimize=[(0,1),(0,1),(0,1)], backend="torch"
    ).reshape(d2, d2)
    closed_odd = oe.contract(
        "MYct,abci,rsti->YarMbs", t2, d, d.conj(),
        optimize=[(0,1),(0,1)], backend="torch"
    ).reshape(d2, d2)
    return _c3_representative_energy(
        build_even, build_odd, closed_even, closed_odd,
        (Jeb, Jad, Jcf), (Jfa, Jde, Jbc),
        (Jae, Jec, Jca), (Jdb, Jbf, Jfd),
        SdotS, 'env1')


def energy_expectation_nearest_neighbor_3afcbed_bonds(
        a, b, c, d, e, f,
        Jaf, Jcb, Jed, Jdc, Jba, Jfe,
        Jca, Jae, Jec, Jbf, Jfd, Jdb,
        SdotS, chi, D_bond, d_PHYS, C21EB, T1D, T2C):
    """C3-reduced env2 energy: 12 bond contractions become 4."""
    t1 = T1D.reshape(chi, chi, D_bond, D_bond)
    t2 = T2C.reshape(chi, chi, D_bond, D_bond)
    d2 = chi * D_bond * D_bond

    def build_even():
        return oe.contract(
            "YX,MYar,abci,rstj->MbsXctij", C21EB, t1, a, a.conj(),
            optimize=[(0,1),(0,1),(0,1)], backend="torch"
        ).reshape(d2, d2, d_PHYS, d_PHYS)

    def build_odd():
        return oe.contract(
            "LXbs,abci,rstj->XctLarij", t2, f, f.conj(),
            optimize=[(0,1),(0,1)], backend="torch"
        ).reshape(d2, d2, d_PHYS, d_PHYS)

    closed_even = oe.contract(
        "YX,MYar,abci,rsti->MbsXct", C21EB, t1, a, a.conj(),
        optimize=[(0,1),(0,1),(0,1)], backend="torch"
    ).reshape(d2, d2)
    closed_odd = oe.contract(
        "LXbs,abci,rsti->XctLar", t2, f, f.conj(),
        optimize=[(0,1),(0,1)], backend="torch"
    ).reshape(d2, d2)
    return _c3_representative_energy(
        build_even, build_odd, closed_even, closed_odd,
        (Jaf, Jcb, Jed), (Jdc, Jba, Jfe),
        (Jca, Jae, Jec), (Jbf, Jfd, Jdb),
        SdotS, 'env2')


def energy_expectation_nearest_neighbor_other_3_bonds(
        a, b, c, d, e, f,
        Jcd, Jef, Jab, Jbe, Jfc, Jda,
        Jec, Jca, Jae, Jfd, Jdb, Jbf,
        SdotS, chi, D_bond, d_PHYS, C21AF, T1B, T2E):
    """C3-reduced env3 energy: 12 bond contractions become 4."""
    t1 = T1B.reshape(chi, chi, D_bond, D_bond)
    t2 = T2E.reshape(chi, chi, D_bond, D_bond)
    d2 = chi * D_bond * D_bond

    def build_even():
        return oe.contract(
            "YX,MYar,abci,rstj->MbsXctij", C21AF, t1, c, c.conj(),
            optimize=[(0,1),(0,1),(0,1)], backend="torch"
        ).reshape(d2, d2, d_PHYS, d_PHYS)

    def build_odd():
        return oe.contract(
            "LXbs,abci,rstj->XctLarij", t2, d, d.conj(),
            optimize=[(0,1),(0,1)], backend="torch"
        ).reshape(d2, d2, d_PHYS, d_PHYS)

    closed_even = oe.contract(
        "YX,MYar,abci,rsti->MbsXct", C21AF, t1, c, c.conj(),
        optimize=[(0,1),(0,1),(0,1)], backend="torch"
    ).reshape(d2, d2)
    closed_odd = oe.contract(
        "LXbs,abci,rsti->XctLar", t2, d, d.conj(),
        optimize=[(0,1),(0,1)], backend="torch"
    ).reshape(d2, d2)
    return _c3_representative_energy(
        build_even, build_odd, closed_even, closed_odd,
        (Jcd, Jef, Jab), (Jbe, Jfc, Jda),
        (Jec, Jca, Jae), (Jfd, Jdb, Jbf),
        SdotS, 'env3')


def _alias_c3_open_closed(open_even, open_odd):
    """Expose legacy A..F keys while storing only two representatives."""
    opens = {
        'A': open_even, 'C': open_even, 'E': open_even,
        'B': open_odd, 'D': open_odd, 'F': open_odd,
    }
    closed_even = oe.contract("ABii->AB", open_even, backend="torch")
    closed_odd = oe.contract("ABii->AB", open_odd, backend="torch")
    closeds = {
        'A': closed_even, 'C': closed_even, 'E': closed_even,
        'B': closed_odd, 'D': closed_odd, 'F': closed_odd,
    }
    return opens, closeds


def build_open_closed_env1(a, b, c, d, e, f, chi, D_bond, d_PHYS,
                           C21CD, T1F, T2A):
    """Build only the A/C/E and B/D/F representatives for environment 1."""
    open_even = build_single_open_env1(
        'A', a, b, c, d, e, f, chi, D_bond, d_PHYS, C21CD, T1F, T2A)
    open_odd = build_single_open_env1(
        'D', a, b, c, d, e, f, chi, D_bond, d_PHYS, C21CD, T1F, T2A)
    return _alias_c3_open_closed(open_even, open_odd)


def build_open_closed_env2(a, b, c, d, e, f, chi, D_bond, d_PHYS,
                           C21EB, T1D, T2C):
    """Build only the A/C/E and B/D/F representatives for environment 2."""
    open_even = build_single_open_env2(
        'A', a, b, c, d, e, f, chi, D_bond, d_PHYS, C21EB, T1D, T2C)
    open_odd = build_single_open_env2(
        'F', a, b, c, d, e, f, chi, D_bond, d_PHYS, C21EB, T1D, T2C)
    return _alias_c3_open_closed(open_even, open_odd)


def build_open_closed_env3(a, b, c, d, e, f, chi, D_bond, d_PHYS,
                           C21AF, T1B, T2E):
    """Build only the A/C/E and B/D/F representatives for environment 3."""
    open_even = build_single_open_env3(
        'C', a, b, c, d, e, f, chi, D_bond, d_PHYS, C21AF, T1B, T2E)
    open_odd = build_single_open_env3(
        'D', a, b, c, d, e, f, chi, D_bond, d_PHYS, C21AF, T1B, T2E)
    return _alias_c3_open_closed(open_even, open_odd)


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

    #print(Splus,Sminus,Sz)

    # Perform contractions on CPU with explicit backend, then move to device
    SdotS = (oe.contract("ij,kl->ijkl", Splus, Sminus, backend="torch") * 0.5
            +oe.contract("ij,kl->ijkl", Sminus, Splus, backend="torch") * 0.5
            +oe.contract("ij,kl->ijkl", Sz, Sz, backend="torch")
            ).to(dtype=TENSORDTYPE, device=DEVICE)
    return SdotS
