# Low-J2 fixed-pinning continuation on Izar

This self-contained bundle launches 30 Slurm dependency-chain heads.  The
first stage of every chain resumes its matching optimized `J2=0.26` tensor;
every later stage is submitted with `afterok` and resumes the immediately
preceding J2 best tensor.

Exactly one environment dimension is run for each D: `D=6 -> chi=108`,
`D=7 -> chi=126`, and `D=8 -> chi=144`.  The initial checkpoint is loaded
with `--resume-tensors-only`, so its historical CTM environment/chi is ignored.

| signed field | target J2 sequence |
|---|---|
| `-0.01,+0.01` | `0.25,0.24,...,0.16` in steps of `0.01` |
| `-0.02,+0.02` | `0.24,0.22,...,0.10` |
| `-0.04,+0.04` | `0.24,0.22,...,0.10` |
| `-0.08,+0.08` main | `0.23,0.20,...,0.02` in steps of `0.03` |
| `-0.08,+0.08` extra | `0.24,0.22,...,0.12` |

For `D=6,7`, the main grids use Izar's three-day `normal` QOS; `D=8` uses
the seven-day `long` QOS.  Every extra `|h|=0.08` grid uses the seven-day
configuration for all D.  The `.run` files preserve the known working Izar
stack: `partition=gpu`, one GPU, one CPU, 40 GB, `exclude=i39`, GCC 11.3,
CUDA 11.8, and `/home/chye/venvs/6adctmrg_Izar`.

Numerics are pure L-BFGS at fixed D/chi, double precision,
`CTM_MAX_STEPS=70`, and CTMRG/padding noise `0.001`.  Negative signed h is
implemented by the `dimer-plaquette` branch with positive field magnitude;
positive signed h uses the `plaquette` branch.

```bash
# From local PowerShell:
scp -r .\models\VBCLowJ2ContinuationIzar chye@izar.hpc.epfl.ch:~/

# On Izar:
cd ~/VBCLowJ2ContinuationIzar
bash submit_all.sh --dry-run
bash submit_all.sh
```

Expected submission: **246 Slurm jobs = 30 heads + 216 afterok jobs**.
Results are written under `Results_Izar_lowJ2/`.
