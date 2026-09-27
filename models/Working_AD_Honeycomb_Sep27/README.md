# Sep-27 h=0 J2 adiabatic continuation (Kuma)

This directory is a self-contained pure-LBFGS two-C3 bundle.  It launches the
best D10 plaquette, best D10 genuine dimer and best D11 plaquette seed. The D11
dimer branch is excluded. Every seed has two directions and no replica or
insurance duplicate, hence 6 dependency chains. Every J2 point is its own
Slurm job; later points use `afterok` and resume the guarded predecessor.

After each optimization, `eta_guard.py` evaluates the raw-NN texture coordinate
and requires `|eta| >= 0.35`. A mixed result is retained for analysis but its
tensor is not propagated: `resume_for_next.pt` is copied from the preceding
stage. If the first target in a direction is mixed (there is no earlier
J2-stage result), it is recomputed once with a different RNG seed. If that
retry is also mixed, the immutable starting seed is propagated.

The physical grid is:

`0.26, 0.265, 0.27, 0.275, 0.28, 0.29, 0.30, 0.31, 0.32`.

Every one of the 24 J2 stages has its own standalone file under `jobs/`. Each
file directly follows `0730core/singleFileSbatchTwoC3.run`: Kuma `long` QoS,
`h100` partition, one GPU, 16 CPU cores, 90 GB host RAM and `167:59:50` wall
time are written in its `#SBATCH` header. Each Python stage stops internally
after D/5 days:
48 h for the lower-D class and 52.8 h for the higher-D class.  The environment
is `/home/pghosh/venvs/6adctmrg_Kuma` with `nvhpc/24.7-mpi` and CUDA 12.5.1.
No account is hard-coded. The submit command is intentionally minimal, like
the established Kuma launchers: `sbatch --chdir=... job.run` for a chain head,
and the same command plus `--dependency=afterok:...` for later stages. No
resources, job variables, or job names are sent through CLI overrides.

The selected seeds and all 24 run files are already materialised inside the
archive. The six chains are written explicitly in `submit_6_sequences.sh`.
Cluster launch performs no plan generation and no seed copying.

Cluster-visible Slurm names all start with `D10`, including the higher-D
class. Result directories contain only aliases `a01` to `a03`; Slurm logs use
the masked D10 job name. The real mapping is retained in
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
bash submit_6_sequences.sh
```

That single command submits exactly
`6 chain heads + 18 afterok jobs = 24 jobs`.
Inspect the live dependency graph with:

```bash
squeue -u "$USER" -o '%.18i %.16j %.10T %.20R'
```

Results remain under `Results_Sep27/aXX/{left,right}/J2_*`.  Preserve
`private_manifest.tsv` when downloading; it is the alias-to-physical-D key.
