# Izar J2 continuation from manually selected h=0 tensors

This directory follows each manually selected `D<=9` tensor away from its
native `J2` on the fixed grid

```text
0.26, 0.265, 0.27, 0.275, 0.28, 0.29, 0.30, 0.31, 0.32
```

The left and right directions are independent. Every J2 point is a distinct
Slurm job. Within one direction, jobs are linked by `afterok:<jobid>` and each
new point resumes the completed best tensor from the immediately preceding
point. Every directional sequence is submitted twice into separate
`insurance_1` and `insurance_2` trees. Dependent jobs remain pending and do
not hold a GPU while their predecessor runs.

Every stage uses the original Hamiltonian (`h=0`), the two-C3 ansatz, pure
L-BFGS, and `--resume-tensors-only`. The per-point internal limits are exactly
`(D/5) days`: D7=33.6 h, D8=38.4 h, D9=43.2 h.
D7 uses Izar's three-day/normal launcher; D8 and D9 use its seven-day/long
launcher.  The validated CTM dimensions are D7/chi91, D8/chi104, D9/chi108.

The selected `J2=0.33, D=9` tensor lies to the right of the requested grid, so
it has a left sequence only. Consequently there are 11 directional sequences
and 22 insurance-chain heads rather than 24. Expanding all grid points gives
98 Slurm jobs: 22 heads can queue normally, while 76 wait on dependencies.
There are 32 D7/three-day stage jobs and 66 D8-D9/seven-day stage jobs.

Before upload, regenerate and validate the bundled seeds from the manual CSV:

```powershell
python .\models\VBCJ2SeedContinuationIzar\prepare_selected_seeds.py
```

Upload the self-contained seed/launcher folder (the numerical code remains in
the already deployed `~/VBCPinningLyraLBFGS` folder):

```powershell
scp -r .\models\VBCJ2SeedContinuationIzar chye@izar.hpc.epfl.ch:~/
```

If an obsolete one-allocation version was already submitted, first cancel its
jobs and remove only this workflow's results/logs:

```bash
cd ~/VBCJ2SeedContinuationIzar
bash clear_j2_continuation_izar.sh --yes
```

Then inspect the exact dependency graph without submitting:

```bash
cd ~/VBCJ2SeedContinuationIzar
bash submit_selected_j2_sequences.sh --dry-run
```

The final two lines must report 98 jobs, 22 chain heads, and 76 dependent
stages. Submit:

```bash
bash submit_selected_j2_sequences.sh
```

Results are isolated in `Results_Izar_J2_sequences`. The submitter refuses to
run while any old or new workflow jobs remain in the queue. A failed stage
does not release its `afterok` successors; the independent insurance chain is
unaffected. A later clean resubmission resumes a partial stage from
`latest.pt` and immediately exits stages that already have both best tensor
and observation.

## Snapshot and plot completed individual J2 stages

There is no need to wait for an entire directional chain. From the repository
root on Windows, this one command uploads the packer, snapshots only stage
directories containing the terminal `sweep_results.json`, downloads them,
merges them with earlier snapshots, and regenerates the fixed-D figures:

```powershell
powershell -ExecutionPolicy Bypass -File .\models\VBCJ2SeedContinuationIzar\download_completed_j2_and_plot.ps1
```

The accumulated raw results go to
`data/distinVBCsJ2Continuation/Results_Izar_J2_sequences` outside this code
repository. The figures and their source CSVs go to
`visual_elements/figs/VBCDiscriminator/j2_seed_continuations`. Each
`2C3_NN_ranks_vs_J2_Dx.pdf` has shared-y dimer-plaquette-seed and
plaquette-seed panels. Left and right continuations use the same rank markers
and meet at the actual selected seed observation; insurance replicas differ
only by line style/opacity.

The same command also updates one two-panel extrapolation PDF per eligible J2.
The left panel uses the dimer-plaquette seed and the right panel the plaquette
seed. Panel eligibility is independent: a panel is fitted once that seed has
D=7,8,9, while the other panel remains explicitly blank if it is incomplete.
A J2 is skipped only while neither seed has all three D values. Each populated
panel averages available insurance replicas at each D, then linearly
extrapolates the three sorted NN ranks to `1/D=0`. These extrapolations produce
PDF only: no Delta construction and no PNG/CSV side products.

## Supplemental D9 starts on the three-day normal QoS

Three additional h=0 seeds are bundled independently of the original six:

| seed | seed J2 | texture | eta | abs(dE) | relative Delta difference |
|---|---:|---|---:|---:|---:|
| s101 | 0.28 | dimer-plaquette | +0.8975 | 1.09e-5 | 18.49% |
| s102 | 0.30 | dimer-plaquette | +0.7577 | 8.30e-6 | 8.68% |
| s103 | 0.31 | plaquette | -0.9335 | 2.25e-5 | 7.00% |

The s101 Delta mismatch is deliberately accepted despite slightly exceeding
the former 15% cutoff because it is the purest available non-J2=0.33 D9
dimer seed.  The J2=0.31 plaquette seed is used instead of the even purer
J2=0.33 original tensor because the fixed grid ends at 0.32 and the latter
cannot have the requested right branch. Its source directory says orientation
2 because that was the original dimer pin; the final h=0 correlations identify
the plaquette tensor itself as orientation 0, which is what the new run uses.

Regenerate and validate the self-contained supplemental seeds locally:

```powershell
python .\models\VBCJ2SeedContinuationIzar\prepare_d9_supplemental_seeds.py
```

Upload the supplemental files into the existing remote bundle:

```powershell
$b = ".\models\VBCJ2SeedContinuationIzar"
scp -r "$b\d9_supplemental_seeds" `
  "$b\d9_supplemental_seed_manifest.csv" `
  "$b\d9_supplemental_submission_plan.tsv" `
  "$b\submit_d9_supplemental_sequences.sh" `
  "$b\run_one_j2_stage_izar.sh" `
  "$b\izar_3days_sequence.run" `
  "$b\pack_completed_j2_stages.sh" `
  chye@izar.hpc.epfl.ch:~/VBCJ2SeedContinuationIzar/
```

On Izar, verify and submit:

```bash
cd ~/VBCJ2SeedContinuationIzar
bash submit_d9_supplemental_sequences.sh --dry-run
bash submit_d9_supplemental_sequences.sh
squeue -u "$USER" -o '%.18i %.18j %.2t %.10M %.20R' | grep -E 'vjd10[123]'
```

The dry-run must report 48 three-day stage jobs: 12 chain heads and 36
afterok dependents. All points use D9/chi108, the external normal QoS with
71:59:50 walltime, and the unchanged 43.2-hour internal D/5 limit. Existing
vjc jobs are allowed to coexist; only duplicate vjd jobs are refused.

Supplemental results share `Results_Izar_J2_sequences`, so the existing
snapshot command downloads them automatically. Fixed-D plots distinguish
each D9 start with a separate marker and labelled seed line. Inverse-D plots
do not average different starting basins: each available D9 seed produces a
separate linear-fit possibility.

## Connected NN correlations

The same Windows download-and-plot command also generates a connected copy of
every VJC NN-correlation figure. For a bond `xy` in environment `e`, the value
is computed directly from the observation file as

```text
connected_corr(e,xy) = corr(e,xy) - dot(mag(e,x), mag(e,y))
```

Thus all three Sx/Sy/Sz components are included. Outputs use the
`connected_NN` filename marker and mirror both fixed-D continuation plots and
the per-J2 two-panel inverse-D fits. Raw connected-correlation points have no
error bars, while regression shading and extrapolated-intercept uncertainty
remain visible. Connected extrapolations, like the ordinary ones, write PDF
only.
