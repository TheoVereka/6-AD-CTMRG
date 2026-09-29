# LastIzar replacement workload

This directory is a static Izar bundle.  All 226 new `.run` files already
exist under `jobs/`; the cluster does not generate or rewrite run files.

## Optimizer behavior

- `main_C3.py`: Adam warmup then L-BFGS.  On a non-resumed run, the
  `near-optimal -> final L-BFGS` shortcut is suppressed during the first ten
  steps of the first Adam optimizer only.  Resumed tensors and every later
  Adam optimizer retain the shortcut.
- `main_C3_LBFGS.py`: pure L-BFGS.
- Mean-field tensor noise, padding noise, and CTMRG initialization/restart
  noise are at most `1e-3`.  Randomized SVD is unchanged.
- D=5 uses chi=50; D=6 uses chi=72; every job uses CTM max steps 50.

## New jobs

- Four new D=6 plaquette adiabatic chains start from the immutable original
  D=6, J2=.265 tensor and run `.27,.275,.28,.29,.30,.31,.32`.  They use four
  distinct fixed seeds.  The existing direct task1 chain remains insurance 1;
  the new chains are insurances 2--5.  QOS `long`, Slurm wall time 24 h.
- D=5 dimer chain: original J2=.29 seed -> `.30,.31,.32`; QOS `long`, 12 h.
- D=6 dimer chain: original J2=.275 seed -> `.27,.265,.26`; QOS `long`, 24 h.
- Independent plaquette preparations use h=.02, .03, and .01.  For every h,
  both Adam->L-BFGS and pure-LBFGS mean-field preparations are followed by a
  pure-LBFGS h=0 `afterok` job.
  - D=6 grid: `.32,.31,.30,.29,.28,.275,.27`; QOS `normal`, 36 h.
  - D=5 grid: `.26,.265,.27,.275,.28,.29,.30,.31,.32`; QOS `long`, 24 h.
- Submission order is h=.02, .03, .01, with all D=6 pairs before D=5 pairs
  for each h.

The new submission consists of 226 jobs: 102 heads and 124 `afterok` jobs.
The already completed insurance-1 chain is not resubmitted.

## Replace the bundle on Izar

From Windows PowerShell at the repository root:

```powershell
scp -r .\models\LastIzar chye@izar.hpc.epfl.ch:~/
```

Then on Izar:

```bash
cd ~/LastIzar
bash cleanup_obsolete_h005.sh
bash submit_all.sh --dry-run
bash submit_all.sh
```

The cleanup script cancels only pending L2--L5 jobs whose reason contains
`DependencyNeverSatisfied`, deletes only the obsolete task2--task5 result
directories, and removes only their L2--L5 logs.  It never touches task1.

Every stage writes under `Results_LastIzar/` and can be downloaded while the
remaining jobs continue to run.
