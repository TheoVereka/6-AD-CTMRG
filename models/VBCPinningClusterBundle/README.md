# Self-contained twoC3 VBC pinning bundle

Copy this whole directory to either cluster. It contains the pinned twoC3
driver, hexagonal CTMRG core, Izar/Kuma launchers, and the existing twoC3
checkpoints needed by the default Izar and Kuma sweeps (plus the earlier even-
`J2` high-`D` seeds).

The submitters explicitly export the absolute bundle path because Slurm runs a
spooled copy of each batch script; the jobs therefore do not resolve helper
scripts relative to `/var/spool/slurmd`.

The physics is unchanged from the VBC discriminator protocol:

```text
fields: 0.08 -> 0.04 -> 0.02 -> 0.01 -> 0
PVB source:    (Jweak, Jstrong, Jstrong) = (J1-2h, J1+h, J1+h)
dimer source:  (Jstrong, Jweak, Jweak)   = (J1+2h, J1-h, J1-h)
```

Only the final `h=0` energies are physical and comparable. Both branches use
the identical twoC3 variational manifold.

By default, `seed_orientations.csv` aligns the source with the NN pattern of
the bundled tensor. Antiferromagnetic correlations are stronger when they are
more negative. For a plaquette-like seed, `orientation` is its unique weak
group; for a dimer-like seed, it is its unique strong group. The matching
branch reinforces the seed while the competing branch reverses strong/weak
couplings on the *same* clock axis. Replica 2 uses that same axis. Set, for
example, `ORIENTATIONS_TEXT="0 1 2"` only when an explicit all-domain sweep is
wanted.

## Izar

```bash
scp -r VBCPinningClusterBundle chye@izar:/scratch/izar/chye/
ssh chye@izar
cd /scratch/izar/chye/VBCPinningClusterBundle
bash submit_izar_vbc.sh
```

Defaults: `J2={0.29,0.30,0.31,0.32}`, `D={8,9}`, seed-aware orientation. This
submits 32 jobs: 16 replica-1 jobs through the three-day launcher and 16
replica-2 jobs through the seven-day launcher. For every branch:

- replica 1 uses the bundled existing tensor and `3daysResumingjob.run`;
- replica 2 uses an independent deterministic random tensor and
  `7daysResumingjob.run`.

The default Izar sweep can therefore be launched simply with:

```bash
bash submit_izar_vbc.sh
```

Resubmitting the same command is safe: finished field stages are skipped and
an interrupted field resumes from its best checkpoint.

The Izar launchers accept only `D=8,chi=104` and `D=9,chi=108` by default.
See `IZAR_MEMORY_REPORT.md` before overriding this guard.

## Kuma

```bash
scp -r VBCPinningClusterBundle pghosh@kuma:/scratch/pghosh/
ssh pghosh@kuma
cd /scratch/pghosh/VBCPinningClusterBundle
bash submit_kuma_vbc.sh
```

Defaults: `J2={0.30,0.32}`, `D={9,10,11}`, seed-aware orientation, for 24
jobs. Launch it with:

```bash
bash submit_kuma_vbc.sh
```

On Kuma, replica 1 also uses the bundled tensor and replica 2 is random.

### Kuma three-source seeded sweep

`submit_vbc_three.sh` runs only `J2=0.30`, `D={5,6,...,11}`, and replica 1. It
submits 21 jobs: the plaquette and dimer-plaquette sources above plus

```text
rank-split source on (rank1, rank2, rank3): (J1+h, J1, J1-h)
```

Here rank1 is the strongest (most-negative) seed correlation, so this third
source reinforces the complete existing bond hierarchy. The full geometrical
rank order is read from the bundled seed correlations and held fixed throughout
`h=0.08 -> 0.04 -> 0.02 -> 0.01 -> 0`. Results use the
separate `Results_VBC_three` tree, so they cannot collide with the standard
Kuma sweep.

Each field stage receives 26 optimization hours explicitly (130 h total inside
the 168 h allocation). An intentional override uses `THREE_STAGE_HOURS`; an
unrelated inherited `STAGE_HOURS` cannot shorten this sweep at submission.

```bash
bash submit_vbc_three.sh
```

Copy the whole `Results_VBC_three` tree back to
`models/VBCPinningClusterBundle/Results_VBC_three` on the local machine,
including unfinished stage folders. The analysis is local-only and can be
rerun while cluster jobs are still running:

```bash
python visual_elements/figs/VBCDiscriminator/analyze_three_source_runs.py \
  --input models/VBCPinningClusterBundle/Results_VBC_three
```

Do not use `analyze_branch_runs.py` for this tree: that script is for the
separate two-source/two-replica `Results_VBC_branches` experiment.

This writes `stage_status.csv`, all readable stage data and available `h=0`
endpoints, energy/order parameters versus `h`, labelled NN correlations versus
`h`, and fixed-`h` trends versus
`1/D` under `visual_elements/figs/VBCDiscriminator/three_source_comparison`.
When all three branches for at least one `D` reach `h=0`, it additionally
writes a common-Hamiltonian energy comparison and `h=0` plots. Finite-`h`
energies from different sources are *not* phase-energy comparisons.

## Results

Both clusters write the same tree under `Results_VBC_branches`. After copying
it back, analyze it with:

```bash
python visual_elements/figs/VBCDiscriminator/analyze_branch_runs.py \
  --input models/VBCPinningClusterBundle/Results_VBC_branches
```

Run `sha256sum -c SHA256SUMS` after SCP to verify all tensor seeds.
