# Sep-27 h=0 J2 adiabatic continuation (Kuma)

This directory is a self-contained pure-LBFGS two-C3 bundle.  It launches the
best plaquette and genuine dimer seed at each of the two D classes.  Every seed
has two directions and two independent insurance copies, hence 16 dependency
chains.  Every J2 point is its own Slurm job; later points use `afterok` and
resume the preceding point.

The physical grid is:

`0.26, 0.265, 0.27, 0.275, 0.28, 0.29, 0.30, 0.31, 0.32`.

The external launcher exactly follows `0730core/singleFileSbatchTwoC3.run`:
Kuma `long` QoS, `h100` partition, one GPU, 16 CPU cores, 90 GB host RAM and
`167:59:50` wall time.  Each Python stage stops internally after D/5 days:
48 h for the lower-D class and 52.8 h for the higher-D class.  The environment
is `/home/pghosh/venvs/6adctmrg_Kuma` with `nvhpc/24.7-mpi` and CUDA 12.5.1.

`selection.tsv` records the four selected seeds. Candidate IDs and numerical
diagnostics are in `candidate_catalog.tsv`. The launcher automatically runs
`prepare_selection.py`, enforcing one seed per D/family and both numerical
tolerances before it submits anything.

Cluster-visible Slurm names all start with `D10`, including the higher-D
class.  Result directories contain only aliases `a01` to `a04`, and log
filenames use only the Slurm job ID.  The real mapping is retained in
`private_manifest.tsv` for decoding after download.

## Give the extracted folder to another user

The recipient locally extracts the archive, then copies the complete folder:

```powershell
tar -xzf .\Working_AD_Honeycomb_Sep27.tar.gz
scp -r .\Working_AD_Honeycomb_Sep27 pghosh@kuma:/scratch/pghosh/
ssh pghosh@kuma
```

Then on Kuma:

```bash
cd /scratch/pghosh/Working_AD_Honeycomb_Sep27
bash submit_16_sequences.sh
```

That single command validates/materialises the seeds and submits exactly
`16 chain heads + 48 afterok jobs = 64 jobs`. An optional non-submitting check
is `bash submit_16_sequences.sh --dry-run`.
Inspect the live dependency graph with:

```bash
squeue -u "$USER" -o '%.18i %.16j %.10T %.20R'
```

Results remain under `Results_Sep27/aXX/{left,right}/J2_*`.  Preserve
`private_manifest.tsv` when downloading; it is the alias-to-physical-D key.
