# D345678910 legacy Neel ordinary correlation lengths

Run the collector on Windows from the repository root:

```text
python models/0727core/neel_legacy_ordinary_correlation_lengths/collect_checkpoints.py
```

It scans only the `(J2,D)` observables that still exist in `D345678910`.
An old `neel_six_correlation_lengths/results` ordinary result is copied beside
the observable only when its recorded source-checkpoint SHA-256 equals the
currently selected `.pt`. J2=0.255 is never reused from `neel_six`; a completed
new-format result for its current tensor is still recognized on later runs.

Copy the entire `neel_legacy_ordinary_correlation_lengths` directory to the
cluster, enter that directory, and submit:

```text
bash submit_all.sh
```

Wait until all Slurm jobs finish. D=10 and D=11 are submitted as three
independent direction jobs. Copy the entire directory, including
`results_three_env_ordinary_v5`, back over the local directory and run from the
repository root:

```text
python models/0727core/neel_legacy_ordinary_correlation_lengths/import_completed_results.py
```

The importer now reads the returned results from its own directory by default
and writes `correlation_length_D_<D>.json` beside the exact selected legacy
observable. `plot_analysis_neel_legacy.py` only plots correlation lengths after
these JSON files exist.

`submit_all.sh --dry-run` prints every Slurm job without submitting it.
