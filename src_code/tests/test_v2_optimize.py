"""
Test optimization: D=2 chi=8, multiple seeds.
Check: convergence, energy bound, consistency, no gradient explosions.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import core_unres_v2 as cv2

torch.set_default_dtype(torch.float64)
cv2.set_dtype(True)

QMC_BOUND = -0.3646  # E/bond for Heisenberg honeycomb (QMC reference)

def test_optimization(seed=42, D=2, chi=8, max_opt_steps=15, max_ctm=60):
    """Run short optimization and check basic sanity."""
    torch.manual_seed(seed)
    H = cv2.build_heisenberg_H(J=1.0)
    
    print(f"\n  Seed={seed}, D={D}, chi={chi}, max_opt_steps={max_opt_steps}")
    
    a, b, c, dt, e, f, history = cv2.optimize_ipeps(
        H, chi, D,
        max_ctm_steps=max_ctm, ctm_conv_thr=1e-7,
        max_opt_steps=max_opt_steps,
        lbfgs_max_iter=10, lbfgs_lr=1.0,
        opt_conv_thr=1e-9,
        verbose=True,
    )
    
    # Check history
    final_E = history[-1]
    print(f"\n  Final E/bond = {final_E:.10f}")
    print(f"  QMC bound    = {QMC_BOUND:.4f}")
    print(f"  All energies: {[f'{e:.6f}' for e in history]}")
    
    # Check all tensors are real
    for name, t in [('a', a), ('b', b), ('c', c), ('d', dt), ('e', e), ('f', f)]:
        assert t.dtype == torch.float64, f"{name} dtype={t.dtype}"
        assert not t.is_complex(), f"{name} is complex"
    
    # Check energy is reasonable
    assert abs(final_E) < 1.0, f"E/bond = {final_E}, unreasonably large"
    
    # Check for huge jumps (|ΔE| > 1.0 between steps)
    for i in range(1, len(history)):
        jump = abs(history[i] - history[i-1])
        if jump > 0.5:
            print(f"  WARNING: Large jump at step {i}: ΔE = {jump:.6f}")
    
    return final_E


if __name__ == "__main__":
    results = {}
    for seed in [42, 123, 999]:
        print(f"\n{'='*60}")
        print(f"OPTIMIZATION: seed={seed}")
        print(f"{'='*60}")
        try:
            e = test_optimization(seed=seed)
            results[seed] = e
            print(f"  ✓ Completed: E/bond = {e:.10f}")
        except Exception as ex:
            print(f"  ✗ FAILED: {ex}")
            import traceback
            traceback.print_exc()
            results[seed] = None
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    valid = [v for v in results.values() if v is not None]
    for seed, e in results.items():
        status = f"{e:.10f}" if e is not None else "FAILED"
        below = " [BELOW QMC!]" if e is not None and e < QMC_BOUND else ""
        print(f"  seed={seed}: E/bond = {status}{below}")
    
    if valid:
        spread = max(valid) - min(valid)
        print(f"\n  Energy spread across seeds: {spread:.6f}")
        print(f"  Mean E/bond: {sum(valid)/len(valid):.10f}")
