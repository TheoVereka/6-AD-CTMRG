# Plaquette continuation from both 62.5% tensors

`submit_plaq_refinement.sh` submits 16 independent heads: D=6 or 7, original
62.5% source F or R, and four replicas per source. Each head starts at 70% and
continues through 75, 80, 84, 88, 91, 94, 96, 98, 99, and 100%, with `afterok`
between adjacent stages. This is 176 jobs (16 heads and 160 dependent jobs).

Every stage uses `main_C3_LBFGS.py`, the plaquette connection, one GPU,
`normal` QOS, a 71:59:50 Slurm limit, and a 70-hour optimizer budget. The
same-D original F/R `COMPLETED.stage` file identifies the precise starting
checkpoint. D=6 stays at chi=108 and D=7 stays at chi=126, matching those
original optimized tensors. The four replicas share the corresponding source
tensor but have distinct fixed RNG seeds. Output and the submission manifest
are kept in a new bundle, separate from the original results.

From the repository root in PowerShell, upload only these five files:

```powershell
ssh.exe chye@izar.hpc.epfl.ch 'mkdir -p /scratch/izar/chye/plaq_refinement_20261006'
scp.exe .\models\connectionPlaqDimer\core_C3.py .\models\connectionPlaqDimer\main_C3_LBFGS.py .\models\connectionPlaqDimer\run_plaq_refinement.sh .\models\connectionPlaqDimer\plaq_refinement_3days.run .\models\connectionPlaqDimer\submit_plaq_refinement.sh chye@izar.hpc.epfl.ch:/scratch/izar/chye/plaq_refinement_20261006/
ssh.exe chye@izar.hpc.epfl.ch 'cd /scratch/izar/chye/plaq_refinement_20261006 && bash submit_plaq_refinement.sh --dry-run'
ssh.exe chye@izar.hpc.epfl.ch 'cd /scratch/izar/chye/plaq_refinement_20261006 && bash submit_plaq_refinement.sh'
```

The last command submits the jobs. The live submission first verifies all
four original F/R checkpoints under
`/scratch/izar/chye/connectionPlaqDimer_trees_20261003`, then writes
`submitted_plaq_refinement_20261006.tsv`. The manifest prevents accidental
resubmission. Results go to `results_plaq_refinement_20261006/` in the new
remote bundle.
