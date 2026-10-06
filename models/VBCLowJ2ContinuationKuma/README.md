# Low-J2 fixed-pinning continuation on Kuma

This self-contained bundle launches four Slurm dependency chains:

- `D=8,9`, with exactly one environment dimension per D:
  `chi=144,162`;
- dimer (`signed h=-0.08`) and plaquette (`signed h=+0.08`);
- `J2 = 0.24, 0.22, 0.20, 0.18, 0.16, 0.14`.

Every first stage resumes the matching `J2=0.26` optimized pinning tensor
with `--resume-tensors-only`; its old CTM environment/chi is not loaded.
Every later stage has an `afterok` dependency and resumes the best tensor from
the immediately preceding J2.  All stages use pure L-BFGS, fixed D/chi,
`CTM_MAX_STEPS=70`, double precision, and the same pinning field throughout.

The Kuma launcher deliberately matches the known working Kuma configuration:
`qos=long`, `partition=h100`, one GPU, 16 CPUs, 90 GB, NVHPC 24.7, CUDA 12.5.1,
and `/home/pghosh/venvs/6adctmrg_Kuma`.  It does not add an account directive.

```bash
# From local PowerShell:
scp -r .\models\VBCLowJ2ContinuationKuma pghosh@kuma:/scratch/pghosh/

# On Kuma:
cd /scratch/pghosh/VBCLowJ2ContinuationKuma
bash submit_all.sh --dry-run
bash submit_all.sh
```

Expected submission: **24 Slurm jobs = 4 heads + 20 afterok jobs**.  Results are
written under `Results_Kuma_lowJ2/`.
