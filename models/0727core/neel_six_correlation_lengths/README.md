# Néel straight-row correlation-length cluster bundle

This directory is self-contained. It contains the available Néel checkpoints,
the audited straight-row transfer code, and the Slurm submission files.

For every available checkpoint with `D>=3`, one job computes:

1. `env2(a,b) × env2(b,a)`;
2. `env1(a,b) × env3(b,a)`;
3. `env3(a,b) × env1(b,a)`;

both as the physical corner-metric generalized eigenproblem and, for
diagnostics only, as an ordinary eigenproblem. The output schema is
`three_geometric_straight_rows_v3`.

The default CTMRG tolerance is `1e-11`. D=3 and D=4 force full-SVD CTMRG;
D>=5 keeps the production augmented-rSVD policy. D=2 is rejected.

## Upload and submit

Copy this one directory to scratch:

```powershell
scp -r models\0727core\neel_six_correlation_lengths chye@CLUSTER:/scratch/chye/
```

Then run exactly one submission command on the cluster:

```bash
cd /scratch/chye/neel_six_correlation_lengths
bash submit_cluster.sh --dry-run
bash submit_cluster.sh
```

The script submits every available `D>=3` checkpoint as a separate Slurm job.
Completed files are written only below:

```text
/scratch/chye/neel_six_correlation_lengths/results_straight_rows_v3/
```

Only valid, converged schema-v3 files in that new directory are skipped, so
the submission command can be rerun after a partial batch.

## Download

Copy the whole bundle back, or only the new result directory:

```powershell
scp -r chye@CLUSTER:/scratch/chye/neel_six_correlation_lengths D:\destination\
```

After download, `plot_six_inverse_xi.py` recursively finds only schema-v3
results and ignores all obsolete schema-v2 JSON files.
