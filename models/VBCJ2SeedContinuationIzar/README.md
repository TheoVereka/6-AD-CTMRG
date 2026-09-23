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
