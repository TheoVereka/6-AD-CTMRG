"""
SUMMARY: Diagnostic Code Installation & Next Steps
===================================================

STATUS: ✅ Diagnostic code successfully installed in main_sweep_CPU.py

CRITICAL FINDING: Your normalization test revealed norm2 is NEGATIVE (-1.89e-08)
                  This is physically impossible and indicates a bug!

═══════════════════════════════════════════════════════════════════════════════
WHAT HAS BEEN DONE
═══════════════════════════════════════════════════════════════════════════════

1. ✅ Diagnostic code inserted into main_sweep_CPU.py (lines ~580-670)
   - Automatically detects when loss exceeds physical bounds
   - Prints detailed diagnostics (E1, E2, E3 components, tensor norms, CTMRG status)
   - Saves JSON data to diagnostics/ folder
   - SavesPyTorch tensors for post-mortem analysis

2. ✅ Created test_positive_norms.py
   - Tests if normalizations are always positive across multiple trials
   - Will help confirm if norm2 consistently comes out negative

═══════════════════════════════════════════════════════════════════════════════
HOW TO USE THE DIAGNOSTIC
═══════════════════════════════════════════════════════════════════════════════

STEP 1: Run your sweep normally
----------------------------------------
cd /home/chye/6ADctmrg/6-AD-CTMRG/src_code
python scripts/main_sweep_CPU.py

STEP 2: Watch for diagnostic output
----------------------------------------
When loss exceeds bounds, you'll see:

    ======================================================================
    ⚠️  EXTREME LOSS DETECTED AT STEP 212
    ======================================================================
    Loss: +52.666603 (per bond: +5.851845)
    Physical bounds: [-0.75, +0.25] per bond
    
    Energy components:
      E1 (bonds EB,AD,CF): +18.245  (per bond: +6.082)
      E2 (bonds AF,CB,ED): +19.835  (per bond: +6.612)  ← largest component
      E3 (bonds CD,EF,AB): +14.587  (per bond: +4.862)
      Sum: +52.666603
      Reported loss_item: +52.666603
      Discrepancy: 0.000e+00
    
    Tensor statistics:
      ||a|| = 1.000234
      ||b|| = 0.999876
      ...
      Contains NaN: False
      Contains Inf: False
    
    Optimization state:
      D_bond=3, chi=29
      CTMRG steps: 70 (max=70)
      ⚠️  CTMRG DID NOT CONVERGE  ← KEY INDICATOR
      Optimizer: lbfgs
    ======================================================================
    
    Diagnostic data saved to: ../diagnostics/extreme_loss_step_212.json

STEP 3: Share the diagnostic output
----------------------------------------
Send me:
1. The printed diagnostic text when extreme loss appears
2. The JSON file from diagnostics/extreme_loss_step_XXX.json
3. Any patterns you notice (e.g., "always E2 is largest", "always when CTMRG hits max steps")

═══════════════════════════════════════════════════════════════════════════════
WHAT TO LOOK FOR
═══════════════════════════════════════════════════════════════════════════════

KEY INDICATOR 1: "CTMRG DID NOT CONVERGE"
   → Environment is stale/incorrect for current tensors
   → FIX: Try increasing CTM_MAX_STEPS from 70 to 150

KEY INDICATOR 2: E2 is consistently the largest/most extreme
   → Confirms your finding that energy_expectation_nearest_neighbor_3afcbed_bonds has a bug
   → The negative norm2 you discovered is likely the root cause
   → Need to fix contraction logic in that function

KEY INDICATOR 3: Tensor norms >> 1.0
   → Normalization failing during L-BFGS line search
   → May need to normalize inside closure() before each energy evaluation

KEY INDICATOR 4: Discrepancy > 1e-6
   → loss_item ≠ E1+E2+E3 when recomputed
   → Suggests optimizer is caching wrong value or race condition

═══════════════════════════════════════════════════════════════════════════════
HYPOTHESIS: Root Cause of +80, +52, -19 Bug
═══════════════════════════════════════════════════════════════════════════════

Based on your normalization test showing norm2 is NEGATIVE:

PRIMARY SUSPECT: energy_expectation_nearest_neighbor_3afcbed_bonds
   - This function produces negative normalization (impossible!)
   - Likely has wrong environment tensor assignment
   - Or wrong index ordering in one of the contractions
   
SECONDARY SUSPECT: CTMRG not converging
   - If CTMRG hits max_steps without converging
   - Environment doesn't match tensors
   - All energy functions will give wrong results

To test: Run test_positive_norms.py
   cd /home/chye/6ADctmrg/6-AD-CTMRG/src_code/tests
   python test_positive_norms.py
   
   If norm2 is consistently negative → confirms the bug is structural
   If norm2 is sometimes negative → suggests numerical instability

═══════════════════════════════════════════════════════════════════════════════
NEXT ACTIONS
═══════════════════════════════════════════════════════════════════════════════

IMMEDIATE:
1. Run: python tests/test_positive_norms.py
   → Confirms if norm2 is consistently negative

2. Run: python scripts/main_sweep_CPU.py
   → Captures diagnostic data when extreme loss appears

3. Share results with me

AFTER DIAGNOSTICS:
- If norm2 is negative → Fix energy_expectation_nearest_neighbor_3afcbed_bonds
- If CTMRG doesn't converge → Increase CTM_MAX_STEPS or investigate why
- If tensors denormalized → Add normalization inside optimizer closure

═══════════════════════════════════════════════════════════════════════════════
FILES CREATED/MODIFIED
═══════════════════════════════════════════════════════════════════════════════

MODIFIED:
- scripts/main_sweep_CPU.py (added diagnostic code at line ~580)

CREATED:
- tests/DIAGNOSTIC_INSTRUCTIONS.py (detailed guide)
- tests/DIAGNOSTIC_PATCH.py (original patch code)
- tests/test_positive_norms.py (test for negative normalizations)
- tests/test_denormalized_tensors.py (test denorm effect)
- tests/test_nan_inf_propagation.py (test numerical issues)
- tests/test_imaginary_component.py (test Hermiticity)

All test results so far:
✅ Imaginary components negligible (< 1e-10)
✅ No NaN/Inf in normal operation
✅ Denormalization doesn't change energy (division by norm works correctly)
❌ norm2 is NEGATIVE (bug confirmed!)

═══════════════════════════════════════════════════════════════════════════════
"""

print(__doc__)
