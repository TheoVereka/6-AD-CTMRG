import numpy as np
import opt_einsum as oe

def get_cost(expr, *shapes):
    """Returns (flops, max_memory) for a given einsum expression and shapes."""
    views = [np.ones(s) for s in shapes]
    # opt_einsum's 'optimal' does a true brute-force search for small tensor counts
    path, path_info = oe.contract_path(expr, *views, optimize='optimal')
    
    # Extract total FLOPs and the largest intermediate tensor size
    flops = path_info.opt_cost
    max_mem = path_info.largest_intermediate
    return flops, max_mem

def evaluate_strategies(D, chi):
    # D is for Latin (Upper), chi is for Greek (lower)
    
    # ---------------------------------------------------------
    # STRATEGY A: THE "REUSE" STRATEGY (Shared Intermediates)
    # ---------------------------------------------------------
    # 1. Form shared P (T and E) -> M Y delta gamma
    flops_P, mem_P = get_cost("MYa, adg -> MYdg", (D, D, chi), (chi, chi, chi))
    
    # 2. Form shared Q (S and B) -> L X epsilon gamma
    flops_Q, mem_Q = get_cost("LXb, ebg -> LXeg", (D, D, chi), (chi, chi, chi))
    
    # 3. Finish Task 2 (T_new)
    flops_T2, mem_T2 = get_cost("MYdg, MdG -> GYg", (D, D, chi, chi), (D, chi, D))
    
    # 4. Finish Task 3 (S_new)
    flops_T3, mem_T3 = get_cost("LXeg, LeH -> HXg", (D, D, chi, chi), (D, chi, D))
    
    # 5. Finish Task 1 (C_new) using C, P, and Q
    flops_T1_reuse, mem_T1_reuse = get_cost("YX, MYdg, LXeg -> MdLe", 
                                            (D, D), (D, D, chi, chi), (D, D, chi, chi))
    
    total_flops_reuse = flops_P + flops_Q + flops_T2 + flops_T3 + flops_T1_reuse
    peak_mem_reuse = max(mem_P, mem_Q, mem_T2, mem_T3, mem_T1_reuse)

    # ---------------------------------------------------------
    # STRATEGY B: THE "INDEPENDENT" STRATEGY (No sharing for C_new)
    # ---------------------------------------------------------
    # 1. Task 1 (C_new) solved purely independently
    flops_T1_ind, mem_T1_ind = get_cost("YX, MYa, LXb, adg, ebg -> MdLe", 
                                        (D, D), (D, D, chi), (D, D, chi), (chi, chi, chi), (chi, chi, chi))
    
    # Task 2 and 3 are calculated independently (same as P/Q + finish steps above, 
    # but their FLOPs are double counted since we aren't reusing them for Task 1)
    total_flops_ind = flops_T1_ind + (flops_P + flops_T2) + (flops_Q + flops_T3)
    peak_mem_ind = max(mem_T1_ind, mem_P, mem_Q, mem_T2, mem_T3)

    return {
        "Reuse": {"flops": total_flops_reuse, "mem": peak_mem_reuse},
        "Independent": {"flops": total_flops_ind, "mem": peak_mem_ind}
    }

# --- Sweep the Parameter Space ---
D_vals = [50, 75, 100, 125, 150, 175, 200]
chi_vals = [30, 40, 50, 60, 70, 80, 90, 100]

print(f"{'D':<5} | {'chi':<5} | {'Winner':<15} | {'Mem Ceiling (D^2 chi^2)':<25} | {'Peak Mem Used':<15}")
print("-" * 75)

for D in D_vals:
    for chi in chi_vals:
        costs = evaluate_strategies(D, chi)
        
        mem_ceiling = (D**2) * (chi**2)
        
        # Determine winner
        if costs["Independent"]["flops"] < costs["Reuse"]["flops"]:
            winner = "Independent"
            speedup = costs["Reuse"]["flops"] / costs["Independent"]["flops"]
        else:
            winner = "Reuse"
            speedup = costs["Independent"]["flops"] / costs["Reuse"]["flops"]
            
        peak_mem = costs[winner]["mem"]
        
        # Verify it does not blow up your memory constraint
        if peak_mem > 2 * mem_ceiling: 
            mem_status = "VIOLATION!"
        else:
            mem_status = "OK"

        print(f"{D:<5} | {chi:<5} | {winner:<15} | {mem_ceiling:<25} | {peak_mem:<15} ({mem_status})")