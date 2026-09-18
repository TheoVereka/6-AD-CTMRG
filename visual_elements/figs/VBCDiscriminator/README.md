# twoC3 VBC discriminator

The old `Delta = rank3-rank1` is only the magnitude of the three-group NN
bond order.  It cannot distinguish these two textures:

| texture | sorted AF correlations | middle fraction | normalized Z6 clock |
|---|---|---:|---:|
| plaquette VBC | `r1 = r2 < r3` | 0 | +1 |
| dimer-plaquette VBC | `r1 < r2 = r3` | 1 | -1 |

Here a more negative correlation is a stronger AF bond and

```text
middle_fraction = (r2-r1)/(r3-r1)
clock_z6 = (27/2) product_i(Gi-mean(G)) / (r3-r1)^3 .
```

`analyze_existing_twoc3.py` applies both diagnostics to existing observations.
It keeps the geometrical groups instead of throwing them away by rank:

```text
G0 = {AD, CF, EB}
G1 = {AF, BC, DE}
G2 = {AB, CD, EF}
```

## Replica-1 supervisor figures

To redraw the two requested summary PDFs for every currently available
cluster/J2 combination (partial continuations are included), run from the
repository root:

```bash
python visual_elements/figs/VBCDiscriminator/plot_pinning_supervisor.py
```

Outputs are under `visual_elements/figs/VBCDiscriminator/replica1_supervisor/`
as `Kuma/J2_0p30/` and `Izar/J2_0p29/` etc., **exactly two PDFs per J2**.
The first is a three-column pinning-field figure with one row per available
`D`, shared axes, and sorted NN correlations. The second has one row per
`h=0.08,0.04,0.02,0.01,0`, shared `1/D` limits and one shared vertical range
per observable column. Izar's unrun rank-split column is left blank.

For sorted correlations `C1 <= C2 <= C3` (more negative is stronger), the
signed texture coordinate in the second figure is explicitly

```text
omega1 = C2-C1, omega2 = C3-C2,
eta = (omega1-omega2)/(omega1+omega2).
```

Thus ideal dimer-plaquette is `eta=+1`, ideal plaquette is `eta=-1`, and
`eta` is undefined if all three correlations coincide. Only the `h=0`
energies compare the same Hamiltonian across pinning sources.

## Decisive calculation

`main_C3.py` now accepts a trace-free NN pinning source.  The plaquette and
dimer-plaquette jobs use exactly the same twoC3 degrees of freedom and CTMRG;
only the temporary source is different.  Each job follows
`h = 0.08, 0.04, 0.02, 0.01, 0` by checkpoint continuation.  Thus the final
states are competing minima of the same unbiased Hamiltonian.

On the cluster, copy the four files in `models/0907core` together and run:

```bash
cd /your/scratch/copy/of/0907core
bash submit_vbc_branches.sh
```

The default pilot submits 16 H100 jobs: `J2={0.30,0.32}`,
`D={8,10}`, two branches, and two replicas.  A publication sweep is:

```bash
J2_VALUES_TEXT="0.28 0.30 0.32 0.34" \
D_VALUES_TEXT="8 9 10 11" \
bash submit_vbc_branches.sh
```

To reuse the already optimized `0713summary` twoC3 tensors in replica 1,
first copy that tree to scratch and set `SEED_ROOT`:

```bash
SEED_ROOT=/scratch/you/0713summary bash submit_vbc_branches.sh
```

The initial `h=0.08` source is intentionally large enough to move either old
texture into the requested sector; later stages continue from the previous
field. Replica 2 remains an independent deterministic random start. Leaving
`SEED_ROOT` unset makes both replicas deterministic random starts.

After copying `Results_VBC_branches` back, run:

```bash
python visual_elements/figs/VBCDiscriminator/analyze_branch_runs.py \
  --input models/0907core/Results_VBC_branches
```

The automatic label is deliberately conservative. If both textures survive
to `h=0`, it reports a lower-energy phase only when both replicas exist and
`abs(Edimer-Eplaquette) > 3 epsilon`, where `epsilon` is the maximum of the
chi-lookahead energy shift, within-branch replica spread, and an absolute
per-site floor. If every continuation instead collapses reproducibly to the
same texture, that common texture is reported as selected. All inconsistent
or sub-resolution outcomes are reported as unresolved.
