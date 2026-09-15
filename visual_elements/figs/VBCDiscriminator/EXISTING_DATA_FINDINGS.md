# What the existing twoC3 data can and cannot decide

No `figs0906`, restricted/VUMPS result, or competitor data are used here.
The input is the project's own `data/0713summary/2tensor_twoC3` observations,
with `sym2_free_param` used only as an internal variational consistency check.

## Correct observable

Sorting three NN correlations and retaining only
`Delta = rank3-rank1` removes precisely the information needed here. Define

```text
q = (rank2-rank1)/(rank3-rank1)
Z6 = (27/2) product_i(Gi-mean(G)) / Delta^3 .
```

For AF correlations (more negative means stronger):

```text
plaquette:        rank1 = rank2 < rank3     q=0, Z6=+1
dimer-plaquette:  rank1 < rank2 = rank3     q=1, Z6=-1
```

The continuous values `q` and `Z6`, rather than a thresholded label, should be
plotted and extrapolated.

A genuine three-distinct texture is not excluded by construction. It would
appear as an interior limiting value `0<q<1` with both rank gaps much larger
than the duplicate-environment scatter, and it must be stable under increasing
chi, D, initial state, and orientation. The present interior-q points fail the
last requirement: q moves strongly with D and sometimes changes clock sector.

## Existing high-D evidence

Selected values from the generated CSV are:

| J2 | q(D8) | q(D9) | q(D10) | q(D11) |
|---:|---:|---:|---:|---:|
| 0.28 | 0.075 | 0.161 | 0.150 | 0.309 |
| 0.30 | 1.000 | 0.003 | 0.153 | 0.062 |
| 0.32 | 0.028 | 0.111 | 0.209 | 0.213 |
| 0.34 | 0.998 | 0.060 | 0.149 | 0.506 |

Among all 35 points with `D>=8` and `0.27<=J2<=0.34`, the conservative
threshold gives 19 plaquette-like, 4 dimer-like, and 12 mixed/three-distinct.
In particular, the branch changes with D at fixed J2. A rank-sorted Delta
scaling therefore combines different clock sectors and cannot decide which
sector is the ground state.

There is nevertheless a strong consistency clue at `J2=0.30`:

| ansatz | D | chi | E/site | q | texture |
|---|---:|---:|---:|---:|---|
| twoC3 | 8 | 104 | -0.4246130247 | 0.9997 | dimer-like |
| sym2_free_param | 8 | 104 | -0.4246701540 | 0.0000 | plaquette |

`sym2_free_param` is a mirror-symmetric subset of twoC3. Its energy is lower
by `5.71e-5` per site at the same D and chi, which is impossible if the twoC3
dimer point were the optimized minimum. The D8 dimer point is therefore a
trapped optimization branch, not evidence for a dimer ground state. The D9,
D10, and D11 twoC3 points at J2=0.30 are all plaquette-like.

## Present conclusion

The existing data favor a plaquette VBC, especially around J2=0.28--0.32, but
do not yet constitute a controlled ground-state discrimination. They lack
two separately followed textures at identical `(J2,D,chi)` and hence lack a
dimer variational upper bound against which the plaquette state can be tested.

The decisive calculation is the trace-free source continuation implemented
in `models/0907core`: seed both textures inside the identical twoC3 manifold,
take `h -> 0`, verify the final q/Z6 sector, and compare only the h=0 energies.
If the energy gap is smaller than the chi-lookahead/replica resolution, the
proper result is a quantitative non-resolution bound, not a phase label.
