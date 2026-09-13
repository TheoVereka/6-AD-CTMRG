# D345678910 legacy Neel ordinary correlation lengths

Run the collector on Windows from the repository root:

```text
python models/0727core/neel_legacy_ordinary_correlation_lengths/collect_checkpoints.py
```

It scans only the `(J2,D)` observables that still exist in `D345678910`.
An old `neel_six_correlation_lengths/results` ordinary result is copied beside
the observable only when its recorded source-checkpoint SHA-256 equals the
currently selected `.pt`. There are no J2-specific exceptions: a matching hash
is reused and a different hash is staged for a fresh ordinary calculation.

Copy the entire `neel_legacy_ordinary_correlation_lengths` directory to the
cluster, enter that directory, and submit:

```text
bash submit_all.sh
```

D>=8 (including D=8 and D=9) is submitted as three independent direction jobs:
`env2`, `env1_ab_env3_ba`, and `env3_ab_env1_ba`. Slurm can schedule these
concurrently; actual start times depend on available resources. Re-running
`bash submit_all.sh` skips completed directions and active jobs for the same
checkpoint. Once all three directions have valid results or accepted jobs,
the launcher cancels matching sequential jobs. A submission failure leaves
the sequential job intact; rerun the command to submit the remaining directions.
Older names without a checkpoint hash are cancelled only when their logs
confirm the current checkpoint (and ordinary calculation for generic `cl-` jobs).

Wait until all Slurm jobs finish. Copy the entire directory, including the
manifest and `results_three_env_ordinary_v5` with all direction JSON files,
back to `D:\HyraiOn\ENS_Lyon\Internship\2026-EPFL\data\` and run from the
repository root:

```text
python models/0727core/neel_legacy_ordinary_correlation_lengths/import_completed_results.py
```

The importer reads the returned bundle from that data directory by default
and writes `correlation_length_D_<D>.json` beside the exact selected legacy
observable. `plot_analysis_neel_legacy.py` only plots correlation lengths after
these JSON files exist. Imports are idempotent: a valid destination is reported
as `ALREADY IMPORTED`, and downloaded results are retained unless
`--delete-source` is explicitly passed.

Use `--incoming PATH` if the downloaded bundle is elsewhere. If the cluster
did not create the combined JSON, the importer assembles it from all three
valid direction files, checking each against the manifest checkpoint hash.
Missing or stale directions are reported as `INCOMPLETE`, including in dry runs.

`bash submit_all.sh --dry-run` previews submissions and sequential cancellations
without changing the queue.
