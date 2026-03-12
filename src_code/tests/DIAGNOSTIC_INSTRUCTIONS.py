"""
STEP-BY-STEP GUIDE: How to Add Diagnostic Code to main_sweep_CPU.py

This will help identify the cause of extreme loss values (+80, +52, -19).
"""

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Open main_sweep_CPU.py
# ══════════════════════════════════════════════════════════════════════════════

# Location: /home/chye/6ADctmrg/6-AD-CTMRG/src_code/scripts/main_sweep_CPU.py

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Find the target location (around line 580)
# ══════════════════════════════════════════════════════════════════════════════

# Look for these lines:
"""
        delta     = (loss_item - prev_loss) if prev_loss is not None else float('inf')
        elapsed   = time.perf_counter() - t_start

        print(f"    step {step:5d}  ctm={ctm_steps:3d}  loss={loss_item:+.10f}"
              f"  Δ={delta:+.3e}  {elapsed:.0f}/{budget_seconds:.0f}s")
"""

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Add the import at the TOP of the file (after other imports)
# ══════════════════════════════════════════════════════════════════════════════

# Add these lines after the existing imports (around line 40-60):
"""
import os
import json
"""

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: Insert diagnostic code BETWEEN delta/elapsed calculation and print
# ══════════════════════════════════════════════════════════════════════════════

# INSERT THIS ENTIRE BLOCK after line 580 (after "elapsed = ..." line)
# and BEFORE the print statement:

DIAGNOSTIC_CODE = '''
        # ──────────────────────────────────────────────────────────────────────
        # DIAGNOSTIC CODE: Detect and log extreme loss values
        # ──────────────────────────────────────────────────────────────────────
        
        # Compute individual energy components
        with torch.no_grad():
            E1_diag = energy_expectation_nearest_neighbor_3ebadcf_bonds(
                a.detach(),b.detach(),c.detach(),d.detach(),e.detach(),f.detach(),
                Hs[0],Hs[1],Hs[2], chi, D_bond,
                C21CD,C32EF,C13AB,T1F,T2A,T2B,T3C,T3D,T1E).real.item()
            
            E2_diag = energy_expectation_nearest_neighbor_3afcbed_bonds(
                a.detach(),b.detach(),c.detach(),d.detach(),e.detach(),f.detach(),
                Hs[3],Hs[4],Hs[5], chi, D_bond,
                C21EB,C32AD,C13CF,T1D,T2C,T2F,T3E,T3B,T1A).real.item()
            
            E3_diag = energy_expectation_nearest_neighbor_other_3_bonds(
                a.detach(),b.detach(),c.detach(),d.detach(),e.detach(),f.detach(),
                Hs[6],Hs[7],Hs[8], chi, D_bond,
                C21AF,C32CB,C13ED,T1B,T2E,T2D,T3A,T3F,T1C).real.item()
        
        # Check if loss exceeds physical bounds
        N_BONDS = 9
        PHYSICAL_MIN_PER_BOND = -0.75
        PHYSICAL_MAX_PER_BOND = +0.25
        
        loss_per_bond = loss_item / N_BONDS
        is_extreme = (loss_per_bond < PHYSICAL_MIN_PER_BOND - 0.1 or 
                      loss_per_bond > PHYSICAL_MAX_PER_BOND + 0.1)
        
        if is_extreme:
            print("\\n" + "="*70)
            print(f"⚠️  EXTREME LOSS DETECTED AT STEP {step}")
            print("="*70)
            print(f"Loss: {loss_item:+.6f} (per bond: {loss_per_bond:+.6f})")
            print(f"Physical bounds: [{PHYSICAL_MIN_PER_BOND:.2f}, {PHYSICAL_MAX_PER_BOND:.2f}] per bond")
            
            print(f"\\nEnergy components:")
            print(f"  E1 (bonds EB,AD,CF): {E1_diag:+.6f}  (per bond: {E1_diag/3:+.6f})")
            print(f"  E2 (bonds AF,CB,ED): {E2_diag:+.6f}  (per bond: {E2_diag/3:+.6f})")
            print(f"  E3 (bonds CD,EF,AB): {E3_diag:+.6f}  (per bond: {E3_diag/3:+.6f})")
            print(f"  Sum: {E1_diag + E2_diag + E3_diag:+.6f}")
            print(f"  Reported loss_item: {loss_item:+.6f}")
            print(f"  Discrepancy: {abs(loss_item - (E1_diag + E2_diag + E3_diag)):.3e}")
            
            print(f"\\nTensor statistics:")
            tensor_norms = {
                'a': torch.norm(a).item(),
                'b': torch.norm(b).item(),
                'c': torch.norm(c).item(),
                'd': torch.norm(d).item(),
                'e': torch.norm(e).item(),
                'f': torch.norm(f).item()
            }
            for name, norm in tensor_norms.items():
                print(f"  ||{name}|| = {norm:.6f}")
            
            has_nan = any(torch.isnan(t).any().item() for t in [a,b,c,d,e,f])
            has_inf = any(torch.isinf(t).any().item() for t in [a,b,c,d,e,f])
            print(f"  Contains NaN: {has_nan}")
            print(f"  Contains Inf: {has_inf}")
            
            print(f"\\nOptimization state:")
            print(f"  D_bond={D_bond}, chi={chi}")
            print(f"  CTMRG steps: {ctm_steps} (max={CTM_MAX_STEPS})")
            if ctm_steps >= CTM_MAX_STEPS:
                print(f"  ⚠️  CTMRG DID NOT CONVERGE")
            print(f"  Optimizer: {OPTIMIZER}")
            print("="*70 + "\\n")
            
            # Save diagnostic data
            diag_dir = os.path.join(os.path.dirname(__file__), "..", "diagnostics")
            os.makedirs(diag_dir, exist_ok=True)
            diag_file = os.path.join(diag_dir, f"extreme_loss_step_{step}.json")
            
            diag_data = {
                'step': step,
                'D_bond': D_bond,
                'chi': chi,
                'loss': loss_item,
                'loss_per_bond': loss_per_bond,
                'E1': E1_diag,
                'E2': E2_diag,
                'E3': E3_diag,
                'ctm_steps': ctm_steps,
                'ctm_converged': ctm_steps < CTM_MAX_STEPS,
                'tensor_norms': tensor_norms,
                'has_nan': has_nan,
                'has_inf': has_inf,
                'optimizer': OPTIMIZER
            }
            
            with open(diag_file, 'w') as f:
                json.dump(diag_data, f, indent=2)
            
            print(f"Diagnostic data saved to: {diag_file}")
            torch.save({
                'a': a.detach().cpu(),
                'b': b.detach().cpu(),
                'c': c.detach().cpu(),
                'd': d.detach().cpu(),
                'e': e.detach().cpu(),
                'f': f.detach().cpu(),
            }, os.path.join(diag_dir, f"tensors_step_{step}.pt"))
        
        # ────────────────────────────────────────────────────────────────────────
        # END DIAGNOSTIC CODE
        # ────────────────────────────────────────────────────────────────────────
'''

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: Run your sweep
# ══════════════════════════════════════════════════════════════════════════════

# Command:
# cd /home/chye/6ADctmrg/6-AD-CTMRG/src_code
# python scripts/main_sweep_CPU.py

# ══════════════════════════════════════════════════════════════════════════════
# WHAT TO EXPECT
# ══════════════════════════════════════════════════════════════════════════════

"""
When the code runs normally, you'll see the regular output:
    step     0  ctm= 23  loss=-0.3167492718  Δ=+inf  5/600s
    step     1  ctm= 15  loss=-0.3245831298  Δ=-7.83e-03  12/600s
    ...

When an extreme loss appears, you'll see:
    step   212  ctm=  7  loss=+52.6666028016  Δ=+3.2e+01  145/600s
    
    ======================================================================
    ⚠️  EXTREME LOSS DETECTED AT STEP 212
    ======================================================================
    Loss: +52.666603 (per bond: +5.851845)
    Physical bounds: [-0.75, +0.25] per bond
    
    Energy components:
      E1 (bonds EB,AD,CF): +18.245123  (per bond: +6.081708)
      E2 (bonds AF,CB,ED): +19.834562  (per bond: +6.611521)
      E3 (bonds CD,EF,AB): +14.586918  (per bond: +4.862306)
      Sum: +52.666603
      Reported loss_item: +52.666603
      Discrepancy: 0.000e+00
    
    Tensor statistics:
      ||a|| = 1.000234
      ||b|| = 0.999876
      ||c|| = 1.000456
      ||d|| = 0.999234
      ||e|| = 1.000123
      ||f|| = 0.999987
      Contains NaN: False
      Contains Inf: False
    
    Optimization state:
      D_bond=3, chi=29
      CTMRG steps: 70 (max=70)
      ⚠️  CTMRG DID NOT CONVERGE
      Optimizer: lbfgs
    ======================================================================
    
    Diagnostic data saved to: ../diagnostics/extreme_loss_step_212.json
    
Normal output continues:
    step   213  ctm= 18  loss=-0.3421298452  Δ=-5.3e+01  152/600s
    ...
"""

# ══════════════════════════════════════════════════════════════════════════════
# INTERPRETING THE RESULTS
# ══════════════════════════════════════════════════════════════════════════════

"""
KEY INDICATORS to look for:

1. ⚠️ "CTMRG DID NOT CONVERGE"
   → The environment is stale/incorrect for the current tensors
   → FIX: Increase CTM_MAX_STEPS or relax CTM_CONV_THR
   → Or: CTMRG is truly unstable for this tensor configuration

2. Discrepancy > 1e-6
   → loss_item doesn't match E1+E2+E3 recomputation
   → Suggests race condition or caching issue in optimizer

3. Any tensor norm >> 2.0 or << 0.5
   → Normalization is failing during optimization
   → Tensors are growing/shrinking despite normalize_tensor() calls

4. Contains NaN: True or Contains Inf: True
   → Numerical overflow/underflow
   → Gradient explosion in L-BFGS

5. One of E1, E2, E3 is >> others
   → Specific bond set has wrong Hamiltonian assignment
   → Or: specific energy function has a bug

6. ctm_steps is very low (< 10) when extreme loss appears
   → Environment updated too quickly, didn't converge
   → Could be from warm-start with wrong initial environment
"""

# ══════════════════════════════════════════════════════════════════════════════
# NEXT STEPS AFTER GETTING DIAGNOSTIC OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

"""
Based on what you see in the diagnostic output, we can:

CASE 1: CTMRG didn't converge
   → Increase CTM_MAX_STEPS from 70 to 150
   → Or check if tensors changed too much between steps

CASE 2: E1, E2, E3 are all large and similar
   → All bonds have wrong energies → environment mismatch
   → Check environment tensor assignments

CASE 3: Only E2 is extreme (and negative norm as you found)
   → Bug in energy_expectation_nearest_neighbor_3afcbed_bonds
   → Need to check contraction logic in that function

CASE 4: Tensor norms are not ~1.0
   → normalize_tensor() is not being called properly
   → Or: optimizer step size is too large

CASE 5: Happens only after warm-start
   → Old checkpoint has incompatible environment or wrong D/chi
   → Start from random initialization instead
"""

print(__doc__)
