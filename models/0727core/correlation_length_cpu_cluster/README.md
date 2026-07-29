# 2C3 straight-row correlation-length cluster bundle

This directory is self-contained: it contains all collected 2C3 checkpoints,
the audited straight-row env2 transfer solver, the manifest, and the Slurm
files. It no longer requires a second source-code copy below `/home/chye`.

The output schema is `straight_row_env2_v3`. Every job uses a strict CTMRG
tolerance of `1e-11` with at most 300 steps. D=3 and D=4 force full-SVD CTMRG,
D>=5 uses the production augmented-rSVD policy, and D=2 is never submitted.
Results that hit the CTMRG step limit are rejected rather than imported.

## Upload and submit

Copy this one directory to scratch:

```powershell
scp -r models\0727core\correlation_length_cpu_cluster chye@CLUSTER:/scratch/chye/
```

Then run exactly one submission command:

```bash
cd /scratch/chye/correlation_length_cpu_cluster
bash submit_correlation_lengths.sh
```

With no arguments, it submits every manifest checkpoint with `D>=3`, one
Slurm job per `(J2,D)`. It discovers the available combinations rather than
assuming every rectangular-grid entry exists.

Useful optional checks and subsets are:

```bash
bash submit_correlation_lengths.sh --dry-run
bash submit_correlation_lengths.sh --J2 0.24,0.26,0.30 --D 3,4,5
bash submit_correlation_lengths.sh --min-D 7
```

All new outputs are written only below:

```text
/scratch/chye/correlation_length_cpu_cluster/results_straight_rows_v3/
```

The command may be rerun after a partial batch; valid completed v3 results
are skipped.

## Download and import

Copy the entire scratch bundle back to the existing data directory:

```powershell
scp -r chye@CLUSTER:/scratch/chye/correlation_length_cpu_cluster D:\HyraiOn\ENS_Lyon\Internship\2026-EPFL\data\
```

Then run:

```powershell
D:\Programs\Python312\python.exe models\0727core\correlation_length_cpu_cluster\import_completed_results.py
```

The importer now reads
`data/correlation_length_cpu_cluster/results_straight_rows_v3/`, validates the
v3 schema and manifest metadata, and moves completed JSON files into the
corresponding `0713summary/J2_*/2tensor_twoC3/D_*/correlation_length.json`
locations ready for `plot_analysis_Windows.py`.
