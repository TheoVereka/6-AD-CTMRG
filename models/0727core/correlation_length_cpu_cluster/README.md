# C3-CTM three-direction ordinary correlation-length cluster bundle

This directory is self-contained: it contains all collected C3-compatible checkpoints,
the three-straight-row ordinary solver, the manifest, and the Slurm
files. It no longer requires a second source-code copy below `/home/chye`.

New output uses `three_geometric_straight_rows_ordinary_v6`. Every
job computes the env2, env1(a,b)-env3(b,a), and env3(a,b)-env1(b,a)
raw row-transfer spectra without `C tensor C`. These are fixed-CTM-gauge
diagnostics rather than gauge-invariance claims. CTMRG uses the production
settings (`max_steps=70`, SV tolerance `1e-7`, mode `both`, energy tolerance
`2e-8`) instead of the erroneous `300/1e-11` override. D=3 and D=4 force
full-SVD CTMRG, D>=5 uses the production augmented-rSVD policy, and D=2 is
never submitted. A single sentinel iteration distinguishes convergence on the
final allowed check from true loop exhaustion; only the latter is rejected.

## Upload and submit

First refresh the local checkpoint manifest. The collector accepts exactly
the C3-compatible `main_C3.py` ansatze `neel_free_param`,
`neel_symmetrized`, `1tensor_C6Ypi`, `1tensor_C3Vypi`, and
`2tensor_twoC3`; every other ansatz directory is ignored:

```powershell
D:\Programs\Python312\python.exe models\0727core\correlation_length_cpu_cluster\collect_checkpoints.py
```

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
Slurm job per `(ansatz,J2,D)`. It iterates only combinations actually present
in the manifest rather than assuming a rectangular grid.

Useful optional checks and subsets are:

```bash
bash submit_correlation_lengths.sh --dry-run
bash submit_correlation_lengths.sh --ansatz 1tensor_C6Ypi,2tensor_twoC3 --min-D 7
bash submit_correlation_lengths.sh --J2 0.24,0.26,0.30 --D 3,4,5
bash submit_correlation_lengths.sh --min-D 7
```

All new outputs are written only below:

```text
/scratch/chye/correlation_length_cpu_cluster/results_three_env_ordinary_v5/
```

The command may be rerun after a partial batch. It skips both valid completed
ordinary results and matching active Slurm jobs. New jobs use names such as
`clo6-2c3-0p29-D10` and encode the ansatz. During migration, active two-C3
names `clo5-2c3-*` are recognized directly. For still older `cl-*` jobs, the
submitter reads their `job-*-$JOB_ID.out` logs and reports whether an ordinary,
generalized-only, or no recognizable marker has appeared. Every still-active
legacy job is skipped conservatively: a generalized marker seen so far does
not prove that the same batch script will not run ordinary next. Once that job
leaves `squeue`, a missing valid ordinary JSON becomes eligible automatically.
RUNNING, PENDING, CONFIGURING, and COMPLETING jobs reported by `squeue` are all
active. `--dependency=singleton` plus the worker's result recheck also prevents
two simultaneous submit invocations from performing the same calculation.

Old generalized result directories are neither read nor deleted. Existing
two-C3 ordinary-v5 JSON remains valid and prevents recomputation; new generic
C3-CTM results use schema v6 in the same ordinary result directory.

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
`results_three_env_ordinary_v5/` below
`data/correlation_length_cpu_cluster/` (so an accidental nested directory
from repeated `scp -r` is harmless), validates compatible ordinary-v5/v6 data and manifest
metadata, and moves each completed JSON into the ansatz path recorded in the
manifest, namely
`0713summary/J2_*/<C3-compatible-ansatz>/D_*/correlation_length.json`, ready
for `plot_analysis_Windows.py`.
