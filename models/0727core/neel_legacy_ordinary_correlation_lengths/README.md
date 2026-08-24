# D345678910 legacy Neel ordinary correlation lengths

1. Run `python collect_checkpoints.py`. It imports compatible ordinary spectra
   already present in the old `neel_six_correlation_lengths/results` tree and
   stages only missing checkpoints.
2. Copy this directory to the cluster and run `bash submit_all.sh`. D=10 and
   D=11 are automatically submitted as three independent direction jobs.
3. Copy the directory back and run `python import_completed_results.py` to
   place `correlation_length_D_<D>.json` beside each selected legacy observable.

`submit_all.sh --dry-run` prints every Slurm job without submitting it.
