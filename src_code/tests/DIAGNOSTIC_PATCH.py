"""
DIAGNOSTIC PATCH for main_sweep_CPU.py

Add this code RIGHT BEFORE the print statement that shows loss values
(around line 583) to capture diagnostic information when extreme loss
values occur.

This will help identify:
1. What causes +80, +52, -19 loss values
2. Whether tensors are corrupted
3. Whether environment is unconverged
4. Which energy components are problematic
"""

# ──────────────────────────────────────────────────────────────────────────────
# INSERT THIS CODE BLOCK AT LINE ~580 in main_sweep_CPU.py,
# RIGHT BEFORE: print(f"    step {step:5d}  ctm={ctm_steps:3d}...")
# ──────────────────────────────────────────────────────────────────────────────

import os
import json

# Compute individual energy components for diagnosis
with torch.no_grad():  # Don't need gradients for diagnosis
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

# Check if loss is physically impossible
PHYSICAL_MIN_PER_BOND = -0.75  # Heisenberg AFM singlet limit
PHYSICAL_MAX_PER_BOND = +0.25  # Heisenberg AFM triplet limit
N_BONDS = 9

loss_per_bond = loss_item / N_BONDS
is_extreme = (loss_per_bond < PHYSICAL_MIN_PER_BOND - 0.1 or 
              loss_per_bond > PHYSICAL_MAX_PER_BOND + 0.1)

if is_extreme:
    print("\n" + "="*70)
    print(f"⚠️  EXTREME LOSS DETECTED AT STEP {step}")
    print("="*70)
    print(f"Loss: {loss_item:+.6f} (per bond: {loss_per_bond:+.6f})")
    print(f"Physical bounds: [{PHYSICAL_MIN_PER_BOND:.2f}, {PHYSICAL_MAX_PER_BOND:.2f}] per bond")
    print(f"Violation: {abs(loss_per_bond - PHYSICAL_MAX_PER_BOND if loss_per_bond > PHYSICAL_MAX_PER_BOND else loss_per_bond - PHYSICAL_MIN_PER_BOND):.3f}")
    
    print(f"\nEnergy components:")
    print(f"  E1 (ebadcf bonds): {E1_diag:+.6f} (per bond: {E1_diag/3:+.6f})")
    print(f"  E2 (afcbed bonds): {E2_diag:+.6f} (per bond: {E2_diag/3:+.6f})")
    print(f"  E3 (other bonds):  {E3_diag:+.6f} (per bond: {E3_diag/3:+.6f})")
    print(f"  Sum: {E1_diag + E2_diag + E3_diag:+.6f}")
    print(f"  Reported loss_item: {loss_item:+.6f}")
    print(f"  Discrepancy: {abs(loss_item - (E1_diag + E2_diag + E3_diag)):.3e}")
    
    print(f"\nTensor statistics:")
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
    
    print(f"\nOptimization state:")
    print(f"  D_bond={D_bond}, chi={chi}")
    print(f"  CTMRG steps: {ctm_steps} (max={CTM_MAX_STEPS})")
    if ctm_steps >= CTM_MAX_STEPS:
        print(f"  ⚠️  CTMRG DID NOT CONVERGE (hit max iterations)")
    print(f"  Optimizer: {OPTIMIZER}")
    
    print("="*70 + "\n")
    
    # Save diagnostic data
    diag_dir = "diagnostics"
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
    
    # Optionally save tensors for debugging
    torch.save({
        'a': a.detach().cpu(),
        'b': b.detach().cpu(),
        'c': c.detach().cpu(),
        'd': d.detach().cpu(),
        'e': e.detach().cpu(),
        'f': f.detach().cpu(),
    }, os.path.join(diag_dir, f"tensors_step_{step}.pt"))

# ──────────────────────────────────────────────────────────────────────────────
# END OF DIAGNOSTIC PATCH
# ──────────────────────────────────────────────────────────────────────────────

# INSTRUCTIONS FOR USE:
# 1. Open main_sweep_CPU.py
# 2. Find line ~583 where it prints:
#      print(f"    step {step:5d}  ctm={ctm_steps:3d}  loss={loss_item:+.10f}"...)
# 3. INSERT the entire diagnostic block ABOVE that line
# 4. Run your sweep normally
# 5. When extreme loss appears, diagnostic info will be printed and saved
# 6. Check the "diagnostics/" folder for saved data
# 7. Share the diagnostic output to identify the root cause

# WHAT TO LOOK FOR in the diagnostic output:
# - If "CTMRG DID NOT CONVERGE" appears → environment is stale/bad
# - If "Discrepancy" is large → loss_item doesn't match recomputed energy
# - If any tensor norm >> 1.0 → normalization failed
# - If any energy component is extreme → specific bond set has issues
# - Compare E1, E2, E3 to see which bonds are problematic
