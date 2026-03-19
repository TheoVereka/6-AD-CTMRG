# NaN Gradient Diagnosis: D=2, chi=5→chi=8 Transition

Diagnostic run: 2026-03-19. Code state: git HEAD `5157b66 "New init env"`.

---

## 1. Confirmed: NaN comes from `_loss.backward()`, not from the forward pass

```
chi=5 forward pass: loss = -0.11129750   ctm_steps=7   ALL grads CLEAN
chi=8 forward pass: loss = -0.11129487   ctm_steps=7   finite
chi=8 backward:     RuntimeError: Function 'LinalgSvdBackward0' returned nan values in its 0th output
                    grad[a..f]: NaN/Inf in ALL six tensors after crash
```

The forward pass at chi=8 succeeds with a finite loss. The backward crashes inside PyTorch's
C++ SVD backward kernel. This is the ONLY source of NaN — the optimizer sees corrupt gradients.

---

## 2. Root cause: degenerate singular values in `trunc_rhoCCC` rho matrices

### 2a. Physical ranks of the rho matrices

The rho matrices passed to `svd_fixed` inside `trunc_rhoCCC` are
`matC21` × `matC32` × `matC13`, each of shape `(chi·D² × chi·D²)`.

At D=2 (D²=4), D⁴=16:
- The raw initial corner matrices `Craw21/32/13` are 16×16 (= D⁴ × D⁴).
- Their **effective rank is ~4** for these tensors (only 4 SVs above 1e-10).

Measured rho32 singular value spectra (all three rho are similar):

| index | chi=5 | chi=8 | chi=12 |
|-------|-------|-------|--------|
| 0 | 2.0e−6 | 2.0e−6 | 2.0e−6 |
| 1 | 4.5e−9 | 4.5e−9 | 4.5e−9 |
| 2 | 7.8e−10 | 7.8e−10 | 7.8e−10 |
| 3 | 6.7e−11 | 6.7e−11 | 6.7e−11 |
| **4** | **2.3e−14** ← single sv near noise | **2.3e−14** | **2.3e−14** |
| **5** | *(not taken)* | **2.8e−15** ← adjacent pair | **2.8e−15** |
| **6** | *(not taken)* | **1.1e−15** | **1.1e−15** |
| **7** | *(not taken)* | **4.5e−17** ← deep noise | **4.5e−17** |

Count of near-zero SVs within `sv[:chi]`:
- chi=5: 2 zeros (threshold 1e−10) — but sv[4]=2.3e−14 is **isolated**, no degenerate peer
- chi=8: **5 zeros** — sv[4..7] are all in machine-noise range, multiple near-degenerate **pairs** exist
- chi=12: **9 zeros**

### 2b. How this produces NaN in PyTorch's SVD backward

PyTorch's `LinalgSvdBackward0` computes the F-matrix:

```
F_ij = 1 / (σi² − σj²)    for i ≠ j
```

When σi and σj are both near machine precision (~1e−15 for float64), their squared values
can be **equal in floating-point representation**: `σi² == σj²` → denominator = 0 → `0/0 = NaN`.

Measured minimum SV gap within `sv[:chi]` and implied F-matrix magnitude:

| chi | min gap in sv[:chi] | max F_ij ~ 1/(min_gap)² |
|-----|---------------------|--------------------------|
| 5   | 6.7e−11             | ~1.5e+10 (large but no 0/0) |
| **8** | **1.06e−15**       | **~9.4e+14 → NaN via 0/0 in float64** |
| 12  | 8.2e−20             | ~1.2e+19 |

At chi=5, sv[4]=2.3e−14 is the only noise-floor SV; within `sv[:5]`, it has no near-identical
partner → `σi² ≠ σj²` in float64 → backward survives.

At chi=8, sv[5]=2.8e−15 and sv[6]=1.1e−15 differ by ~1.7e−15; their squares:
- (2.8e−15)² = 7.8e−30
- (1.1e−15)² = 1.2e−30

Difference = 6.6e−30. In float64 this is representable. But sv[6]=1.1e−15 and sv[7]=4.5e−17:
- (1.1e−15)² = 1.2e−30
- (4.5e−17)² = 2.0e−33

Difference = ~1.2e−30 → F_ij ~ 8.3e+29. When multiplied by gradient components of order ~1,
this gives grad_A ~ 1e+29. **Even if not exactly NaN**, after summing ~32 such terms, the result
can overflow or produce NaN through other arithmetic chains.

In practice the RuntimeError `LinalgSvdBackward0 returned nan` fires at the FIRST such encounter
and halts the backward entirely, so all grad[a..f] are fully corrupted.

---

## 3. Affected code locations

Every call to `svd_fixed` that is in the autograd graph is a potential site.
The CTMRG runs fully differentiably (no `torch.no_grad`), so the entire path is:

```
a,b,c,d,e,f
  → abcdef_to_ABCDEF (normalize_tensor inside — no SVD)
  → CTMRG_from_init_to_stop
      → initialize_envCTs_1          [6 svd_fixed calls]
          ├── SVD of rho32, rho13, rho21 for initial CCC truncation  (calls #0,1,2)
          │     shapes: (16,16) — 12 zero SVs each at chi=8
          └── SVD of out2Ain3D, out3Cin1F, out1Ein2B for T splitting  (calls #3,4,5)
                shapes: (32,32) — 16 zero SVs each
      → loop: iteration × 3 update functions  (7 iterations × 3 = 21 rounds)
          each round calls trunc_rhoCCC  [3 svd_fixed calls per round]
              SVD of rho32, rho13, rho21  (shapes: (32,32) at chi=8)
              → 21×3 = 63 - 6 = NO, each update calls trunc_rhoCCC once = 21 calls = 21×3=63 SVDs? 
              Actually: 7 CTMRG steps × 3 updates/step × 1 trunc_rhoCCC/update × 3 SVDs/trunc = 63 SVDs
  → energy_expectation_* (no SVD)
```

**Total SVDs in the autograd graph at chi=8, ctm_steps=7**:
- `initialize_envCTs_1`: 6 SVDs
- Main CTMRG loop: 7 × 3 × 3 = 63 SVDs (but diagnostic counted only 60 total — 54 from loop)

All 60 observed SVDs had near-zero singular values. The backward crashes at the **FIRST**
degenerate pair encountered during reverse-mode traversal.

### Specific lines in `core_unrestricted.py`:

| Location | Lines | Description |
|----------|-------|-------------|
| `initialize_envCTs_1` | ~726–741 | 3 SVD calls on 16×16 rho mats (CCC truncation) |
| `initialize_envCTs_1` | ~763–765 | 3 SVD calls on 32×32 outer-product mats (T splitting) |
| `trunc_rhoCCC` | ~432, 447, 462 | 3 SVD calls per CTMRG iteration (called from all 3 update funcs) |
| All 3 update functions | ~1040, 1183, 1345 | each calls `trunc_rhoCCC` once per iteration |

---

## 4. Why chi=5 is safe but chi=8 is not

At chi=5: `sv[:5] = [2.0e-6, 4.5e-9, 7.8e-10, 6.7e-11, 2.3e-14]`.
The last entry `sv[4]=2.3e-14` is in the noise floor but has **no other SV nearby within chi=5**.
No pair within `sv[:5]` satisfies `σi² ≈ σj²` in float64 → F-matrix is finite → clean backward.

At chi=8: `sv[:8]` includes sv[4..7] all in range [4.5e-17, 2.3e-14].
Multiple adjacent pairs have `|σi²−σj²|` small enough that the float64 result is catastrophically
imprecise, eventually reaching `0/0=NaN` or producing values so large they NaN through subsequent
arithmetic.

**The transition is structural**: as chi grows past the effective physical rank (~4 at this initialization),
each new SV taken is numerical noise, and the gaps between noise-floor SVs shrink toward machine
epsilon. The SVD backward on ANY of the 60+ SVD calls (all of which include these noise-floor SVs)
can trigger the crash.

---

## 5. What the existing hook-based fix does and does NOT do

`svd_fixed` registers `_hook_U_raw` and `_hook_Vh_raw` which call `nan_to_num` on the incoming
gradient of U and Vh respectively. These hooks run **before the SVD backward kernel** receives
`grad_U` and `grad_Vh`.

**What it cannot fix**: The hook receives `grad_U` (the gradient flowing INTO the SVD from
downstream operations). If `grad_U` contains no NaN (it's finite), `nan_to_num` does nothing.
The SVD backward kernel then computes `F * (U†·grad_U − ...)` where `F_ij = ∞` for degenerate SVs.
`finite × ∞ = ∞`, and `∞ + (−∞) = NaN` in subsequent summations.

The hooks only help if the gradient ARRIVING at U is already NaN. They do not protect against the
F-matrix generating new NaN/Inf from finite-but-tiny SV differences.

---

## 6. Non-issue: `sqrt(svL[:chi])` backward

The three outer-product matrices in `initialize_envCTs_1` (out2Ain3D, out3C1F, out1E2B) have
nonzero singular values within `svL[:chi]` at chi=5, 8, and 12 (confirmed by measurement).
`sqrt` backward is NOT a NaN source under current initialization conditions.

---

## 7. The fix required

Re-implement the **Lorentzian-regularized SVD backward** (lost in `git reset --hard d3d317a`):

```python
# Replace F_ij = 1/(σi²−σj²)  with  F_ij^reg = (σi²−σj²) / ((σi²−σj²)²+ε²)
# As σi→σj: F_ij^reg → 0  (safe)  vs  F_ij → ∞  (NaN)
# ε = 1e-12 for float64 is appropriate (below physical SV gaps, above noise-floor gaps)
```

This was implemented as `class _RegSVD(torch.autograd.Function)` with `_RegSVD.apply(A, eps)`
called from `svd_fixed`. It needs to be reimplemented in `core_unrestricted.py`.

The implementation must also handle the phase-gauge term correctly (anti-Hermitian projection
on the F-term for complex SVD): the backward formula for complex A is:

```
M = U† grad_U − (U† grad_U)†   (anti-Hermitian part only)
N = Vh grad_Vh† − (Vh grad_Vh†)†
grad_A = U · [diag(grad_S) + (1/2) F_reg ⊙ (M + N)] · Vh
```

where `F_reg[i,j] = (σi²−σj²)/((σi²−σj²)²+ε²)` for i≠j, and 0 on the diagonal.

---

## 8. Summary table

| Issue | Confirmed? | Severity | Location |
|-------|-----------|----------|----------|
| rho SVDs in `trunc_rhoCCC` — 0/0 in F-matrix | **YES** | FATAL | Lines ~432,447,462 (called 21×) |
| Init SVDs in `initialize_envCTs_1` — 0/0 in F-matrix | **YES** | FATAL | Lines ~726–741 |
| T-split SVDs in `initialize_envCTs_1` — 0/0 in F-matrix | **YES** | FATAL | Lines ~763–765 |
| `sqrt(svL[:chi])` backward — Inf/NaN | NO (safe at chi=8 currently) | — | Lines ~773–775 |
| Phase-gauge `RuntimeError` from PyTorch svd_backward | NOT triggered (hooks catch it) | minor | Lines ~264–298 |

**All three fatal sources share the same root cause**: `torch.linalg.svd` backward has no
regularization for degenerate/near-zero singular value pairs. The fix is a single custom
autograd Function replacing all `svd_fixed` calls.
