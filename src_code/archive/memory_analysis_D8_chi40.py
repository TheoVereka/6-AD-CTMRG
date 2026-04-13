#!/usr/bin/env python3
"""
GPU Memory Budget Analysis for iPEPS 6-site CTMRG at D=8, chi=40, d_PHYS=2.

This script computes the exact memory footprint (in bytes) for every tensor
class that appears during forward, backward, and checkpoint-recomputation
phases of the CTMRG + energy optimization pipeline.

Run:  python memory_analysis_D8_chi40.py [--D 8] [--chi 40] [--d 2] [--dtype float64]
"""
import argparse, textwrap

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--D',   type=int, default=8,  help='Bond dimension D_bond')
    p.add_argument('--chi', type=int, default=40, help='Environment bond dimension chi')
    p.add_argument('--d',   type=int, default=2,  help='Physical dimension d_PHYS')
    p.add_argument('--dtype', default='float64',
                   choices=['float32','float64','complex64','complex128'])
    p.add_argument('--ctm-steps', type=int, default=50,
                   help='CTM_MAX_STEPS (= a_third_max_iterations)')
    args = p.parse_args()

    D     = args.D
    chi   = args.chi
    d     = args.d
    D_sq  = D * D          # double-layer bond dim
    D4    = D_sq * D_sq     # D^4
    bytes_per = {'float32':4, 'float64':8, 'complex64':8, 'complex128':16}[args.dtype]
    ctm_sub_steps = 3 * args.ctm_steps   # total _ckpt calls in CTMRG loop

    def fmt(b):
        if b < 1024:       return f"{b} B"
        if b < 1024**2:    return f"{b/1024:.1f} KB"
        if b < 1024**3:    return f"{b/1024**2:.1f} MB"
        return f"{b/1024**3:.2f} GB"

    def entry(label, shape, count=1, *, note=''):
        import math
        n = math.prod(shape) * count
        b = n * bytes_per
        dims = 'x'.join(str(s) for s in shape)
        c = f" x{count}" if count > 1 else ""
        nn = f"  ({note})" if note else ""
        return (label, dims, c, n, b, nn)

    rows = []

    # ═══════════════════════════════════════════════════════════════
    #  Section 1:  Persistent tensors (alive for the whole optimization step)
    # ═══════════════════════════════════════════════════════════════
    rows.append(('=== PERSISTENT (alive whole step) ===','','','','',''))
    rows.append(entry('Single-layer tensors a..f',
                       (D, D, D, d), 6, note='iPEPS params'))
    rows.append(entry('Double-layer tensors A..F',
                       (D_sq, D_sq, D_sq), 6, note='a⊗a* contractions'))
    rows.append(entry('Hamiltonian SdotS',
                       (d, d, d, d), 1))

    # ═══════════════════════════════════════════════════════════════
    #  Section 2:  CTMRG iteration chain  (autograd graph output tensors)
    # ═══════════════════════════════════════════════════════════════
    rows.append(('','','','','',''))
    rows.append(('=== CTMRG CHAIN (autograd keeps checkpoint outputs) ===','','','','',''))
    rows.append(entry('Corner matrices per step',
                       (chi, chi), 3, note='3 corners'))
    rows.append(entry('Transfer tensors per step',
                       (chi, chi, D_sq), 6, note='6 transfers'))
    import math
    per_step_entries = 3*chi*chi + 6*chi*chi*D_sq
    per_step_bytes   = per_step_entries * bytes_per
    total_chain      = ctm_sub_steps * per_step_bytes
    rows.append(('  -> per CTMRG sub-step', '', '', per_step_entries, per_step_bytes, ''))
    rows.append((f'  -> {ctm_sub_steps} sub-steps TOTAL',
                 '', '', ctm_sub_steps * per_step_entries, total_chain, ''))

    # ═══════════════════════════════════════════════════════════════
    #  Section 3:  CTMRG update intermediates (per checkpointed step)
    # ═══════════════════════════════════════════════════════════════
    rows.append(('','','','','',''))
    rows.append(('=== CTMRG UPDATE (per _ckpt recomputation in backward) ===','','','','',''))
    rows.append(entry('Grown corner (before truncation)',
                       (chi*D_sq, chi*D_sq), 3, note='absorbed C×T×A×T'))
    rows.append(entry('Corner absorption intermediate',
                       (chi, chi, D_sq, D_sq), 1,
                       note='largest einsum intermediate'))
    rows.append(entry('SVD matrix in trunc_rhoCCC',
                       (chi*D_sq, chi*D_sq), 1, note='R1 @ R2.T'))
    rows.append(entry('SVD output U',
                       (chi*D_sq, chi), 1, note='truncated to chi cols'))
    rows.append(entry('SVD output V',
                       (chi*D_sq, chi), 1))
    rows.append(entry('Projectors P (reshpaed)',
                       (chi, D_sq, chi), 6, note='3 pairs: U-type + V-type'))

    # ═══════════════════════════════════════════════════════════════
    #  Section 4:  Energy function intermediates (THE BIG ONES)
    # ═══════════════════════════════════════════════════════════════
    rows.append(('','','','','',''))
    rows.append(('=== ENERGY FUNCTION (per _ckpt recomputation in backward) ===','','','','',''))
    chiD = chi * D       # = chi * D_bond  (NOT chi * D_sq)
    # WAIT — in the code,  open tensors have shape (chi*D_bond*D_bond, …)
    # chi*D_bond*D_bond = chi * D_sq.  So:
    chiDsq = chi * D_sq   # = chi * D_bond^2 = 2560  for D=8,chi=40

    rows.append(entry('Open tensors (open_A..open_F)',
                       (chiDsq, chiDsq, d, d), 6,
                       note='6 alive simultaneously'))
    rows.append(entry('Closed tensors (closed_A..F)',
                       (chiDsq, chiDsq), 6))
    rows.append(entry('Composite matrices (AD,CF,EB,FA,DE,BC)',
                       (chiDsq, chiDsq), 6,
                       note='mm of closed pairs'))
    rows.append(entry('NN rho tensor (open_X ⊗ open_Y)',
                       (d*d, d*d, chiDsq, chiDsq), 1,
                       note='per bond, ~800 MB each'))
    rows.append(entry('  -> 6 NN rho tensors (AD,CF,EB,FA,DE,BC)',
                       (d*d, d*d, chiDsq, chiDsq), 6,
                       note='ALL held by autograd'))
    rows.append(entry('NNN rho tensor (open_X × closed_Y × open_Z)',
                       (d*d, d*d, chiDsq, chiDsq), 1,
                       note='per bond'))
    rows.append(entry('  -> 6 NNN rho tensors (AE,EC,CA,DB,BF,FD)',
                       (d*d, d*d, chiDsq, chiDsq), 6,
                       note='ALL held by autograd'))
    rows.append(entry('Reduced density matrix rhoXY',
                       (d*d, d*d), 12,
                       note='tiny 4x4'))

    # ═══════════════════════════════════════════════════════════════
    #  Section 5:  rSVD backward 5th term (per SVD)
    # ═══════════════════════════════════════════════════════════════
    rows.append(('','','','','',''))
    rows.append(('=== rSVD BACKWARD (per SVD in trunc_rhoCCC) ===','','','','',''))
    N_svd = chi * D_sq  # matrix size for SVD
    k_svd = chi          # truncation rank
    rows.append(entry('A_input (saved for 5th term)',
                       (N_svd, N_svd), 1, note='detached original matrix'))
    rows.append(entry('B = A - USVh (residual)',
                       (N_svd, N_svd), 1))
    rows.append(entry('R = BhB (for eigh)',
                       (N_svd, N_svd), 1, note='exact path only'))
    rows.append(entry('Q eigenvectors of R',
                       (N_svd, N_svd), 1, note='exact path only'))
    rows.append(entry('Neumann rhs (iterative path)',
                       (N_svd, k_svd), 1))

    # ═══════════════════════════════════════════════════════════════
    #  Section 6:  PEAK MEMORY ESTIMATES
    # ═══════════════════════════════════════════════════════════════
    rows.append(('','','','','',''))
    rows.append(('=== PEAK MEMORY ESTIMATES ===','','','','',''))

    # CUDA context baseline
    cuda_ctx = 750 * 1024**2  # ~750 MB typical

    # Persistent tensors
    persist  = (6 * D**3 * d + 6 * D_sq**3 + d**4) * bytes_per

    # CTMRG chain
    chain    = total_chain

    # Energy backward peak (one _ckpt recomputation):
    #   6 open + 6 closed + 6 composite + 12 rho (all held by autograd)
    open_sz = 6 * chiDsq * chiDsq * d * d * bytes_per
    closed_sz = 6 * chiDsq * chiDsq * bytes_per
    composite_sz = 6 * chiDsq * chiDsq * bytes_per
    rho_sz  = 12 * d*d * d*d * chiDsq * chiDsq * bytes_per

    energy_peak = open_sz + closed_sz + composite_sz + rho_sz

    # Gradient tensors during backward (roughly equal to forward intermediates)
    grad_overhead = energy_peak * 0.3  # conservative estimate

    total_peak = cuda_ctx + persist + chain + energy_peak + grad_overhead

    rows.append(('CUDA context (approx)', '', '', '', cuda_ctx, ''))
    rows.append(('Persistent tensors', '', '', '', persist, ''))
    rows.append(('CTMRG chain (all steps)', '', '', '', chain, ''))
    rows.append(('Energy open tensors (6)', '', '', '', open_sz, ''))
    rows.append(('Energy closed tensors (6)', '', '', '', closed_sz, ''))
    rows.append(('Energy composite matrices (6)', '', '', '', composite_sz, ''))
    rows.append(('Energy rho tensors (12)', '', '', '', rho_sz, '  *** DOMINANT ***'))
    rows.append(('Gradient overhead (~30%)', '', '', '', int(grad_overhead), ''))
    rows.append(('─'*40, '', '', '', '', ''))
    rows.append(('ESTIMATED PEAK', '', '', '', int(total_peak), ''))

    # ═══════════════════════════════════════════════════════════════
    #  Print table
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"  GPU Memory Budget:  D={D},  chi={chi},  d_PHYS={d},  dtype={args.dtype}")
    print(f"  D_sq={D_sq},  chi*D_sq={chiDsq},  CTM sub-steps={ctm_sub_steps}")
    print(f"{'='*80}\n")

    for label, dims, cnt, n, b, note in rows:
        if label.startswith('===') or label.startswith('─'):
            print(f"\n{label}")
            continue
        if not label:
            continue
        size_str = fmt(b) if isinstance(b, (int, float)) and b > 0 else ''
        if isinstance(n, (int, float)) and n > 0:
            print(f"  {label:<50s}  {dims:>20s}{cnt:>4s}  = {n:>15,d} entries  → {size_str:>10s}{note}")
        else:
            print(f"  {label:<50s}  {'':>20s}{'':>4s}    {'':>15s}    {size_str:>10s}{note}")

    print(f"\n{'='*80}")
    print(f"  BOTTLENECK ANALYSIS")
    print(f"{'='*80}")
    print(textwrap.dedent(f"""\
    The 12 rho tensors ({fmt(rho_sz)} total) are the dominant cost.

    Each rho tensor has shape:
      (d²×d², d²×d², chi*D², chi*D²) = ({d*d}, {d*d}, {chiDsq}, {chiDsq})
      = {d*d * d*d * chiDsq * chiDsq:,d} entries = {fmt(d*d * d*d * chiDsq * chiDsq * bytes_per)}

    During checkpoint recomputation of one energy function (backward pass),
    ALL 12 rho tensors are alive simultaneously because:
      1. The variable 'rho' is reassigned 12 times, BUT
      2. Autograd retains a reference to each old tensor (it's saved as input
         to the downstream 'rhoXY = einsum(rho, composite1, composite2)' node
         for gradient computation w.r.t. composite1 and composite2).
      3. Only AFTER the autograd engine processes each rhoXY node can the
         corresponding rho tensor be freed — but all 12 nodes are at the
         same graph depth, so none can be freed until backward reaches them.

    With 3 energy functions checkpointed independently:
      - Only ONE energy function is recomputed at a time (sequential backward
        through the sum E1 + E2 + E3).
      - Peak = one energy function's 12 rho tensors = {fmt(rho_sz)}.

    The "Tried to allocate 8.00 GiB" error:
      - If 8.00 GiB = TOTAL GPU CAPACITY (e.g., RTX 2080/3070), then the OOM
        occurs while trying to allocate one of the late rho tensors (~{fmt(d*d*d*d*chiDsq*chiDsq*bytes_per)})
        when cumulative memory already exceeds the GPU's {fmt(8*1024**3)} limit.
      - If 8.00 GiB = the SINGLE allocation that failed: this doesn't match any
        individual tensor shape at D={D}, chi={chi}. Most likely the error
        message reports the GPU's total capacity, not the allocation size.
        (PyTorch error format: "Tried to allocate X; GPU Y has Z total capacity")

    Multi-GPU (N_GPUS ≥ 2) doesn't help because:
      - CTMRG runs entirely on GPU:0 — all 27 environment tensors stay there.
      - Multi-GPU dispatches the 3 energy functions to different GPUs, but
        each energy function ALONE needs ~{fmt(int(energy_peak))} peak memory.
      - The bottleneck is per-function memory, not total across functions.
    """))

    # ═══════════════════════════════════════════════════════════════
    #  Mitigation strategies
    # ═══════════════════════════════════════════════════════════════
    print(f"{'='*80}")
    print(f"  MITIGATION STRATEGIES")
    print(f"{'='*80}")
    print(textwrap.dedent(f"""\
    1. REUSE rho tensor memory (most impactful):
       Instead of computing all 12 rho tensors first and then backward,
       restructure the energy function to compute each bond's energy
       WITHIN a nested checkpoint. This way only 1 rho (~{fmt(d*d*d*d*chiDsq*chiDsq*bytes_per)})
       is alive at a time → saves {fmt(rho_sz - d*d*d*d*chiDsq*chiDsq*bytes_per)}.

    2. Free open tensors after building closed tensors:
       Currently open_A..F are all alive throughout because they're used
       later for rho computations. If each rho computation is checkpointed
       independently, open tensors can be recomputed on demand.

    3. Use float32 instead of float64:
       Halves all tensor sizes. Peak drops from {fmt(int(total_peak))} to ~{fmt(int(total_peak/2))}.
       BUT: CTMRG convergence may suffer (environment convergence threshold
       needs to be relaxed).

    4. Reduce chi:  chi=30 instead of 40 cuts chiDsq from {chiDsq} to {30*D_sq}:
       rho: ({d*d},{d*d},{30*D_sq},{30*D_sq}) = {d*d*d*d*30*D_sq*30*D_sq:,d} entries
       12 rho → {fmt(12*d*d*d*d*30*D_sq*30*D_sq*bytes_per)} vs current {fmt(rho_sz)}.

    5. Gradient accumulation per bond:
       Compute E_AD, backward through it (accumulating gradients to a..f),
       then E_CF, backward, etc. This requires manual gradient accumulation
       but keeps only 1 bond's rho and open tensors alive at once.
    """))


if __name__ == '__main__':
    main()
