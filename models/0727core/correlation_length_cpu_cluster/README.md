# 2C3 three-direction generalized correlation-length cluster bundle

This directory is self-contained: it contains all collected 2C3 checkpoints,
the audited three-straight-row generalized solver, the manifest, and the Slurm
files. It no longer requires a second source-code copy below `/home/chye`.

The output schema is `three_geometric_straight_rows_generalized_v4`.  Every
job computes the env2, env1(a,b)-env3(b,a), and env3(a,b)-env1(b,a)
corner-metric generalized spectra.  CTMRG now uses the actual production
settings (`max_steps=70`, SV tolerance `1e-7`, mode `both`, energy tolerance
`2e-8`) instead of the erroneous `300/1e-11` override. D=3 and D=4 force
full-SVD CTMRG, D>=5 uses the production augmented-rSVD policy, and D=2 is
never submitted. A single sentinel iteration distinguishes convergence on the
final allowed check from true loop exhaustion; only the latter is rejected.

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
/scratch/chye/correlation_length_cpu_cluster/results_three_env_generalized_v4/
```

The command may be rerun after a partial batch; valid completed v4 results
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

The importer recursively finds every
`results_three_env_generalized_v4/` below
`data/correlation_length_cpu_cluster/` (so an accidental nested directory
from repeated `scp -r` is harmless), validates the v4 schema and manifest
metadata, and moves completed JSON files into the corresponding
`0713summary/J2_*/2tensor_twoC3/D_*/correlation_length.json` locations ready
for `plot_analysis_Windows.py`.
