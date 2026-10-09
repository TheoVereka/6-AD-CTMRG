# Extended low-J2 fixed-pinning continuations on Kuma

This directory is a self-contained Kuma bundle.  It contains immutable copies
of the already tested `main_C3_LBFGS.py`, `core_C3.py`, Kuma `.run` file, stage
runner, and all 16 initial tensors.  Nothing outside this directory is needed
after it is copied to `/scratch/pghosh`.

It submits **80 jobs in 16 independent heads plus 64 afterok jobs**:

| signed h | D | initial tensor | adiabatic target J2 sequence |
|---|---:|---:|---|
| `-0.08,+0.08` | 8, 9 | optimized `J2=0.14` Kuma tensor | `0.12,0.10,0.08,0.06` |
| `-0.04,+0.04` | 8, 9 | optimized `J2=0.26` tensor | `0.24,0.22,0.20,0.18,0.16,0.14,0.12` |
| `-0.02,+0.02` | 8, 9 | optimized `J2=0.26` tensor | `0.24,0.22,0.20,0.18,0.16` |
| `-0.01,+0.01` | 8, 9 | optimized `J2=0.26` tensor | `0.24,0.22,0.21,0.20` |

Every continuation uses the previous J2 stage's optimized tensor with an
`afterok` dependency.  `D=8` runs only `chi=144`; `D=9` runs only `chi=162`.
All stages use pure L-BFGS, `CTM_MAX_STEPS=70`, double precision, CTMRG noise
`0.001`, and a 48-hour internal budget.

The launcher is copied from the successful Kuma configuration: `qos=long`,
`partition=h100`, one GPU, 16 CPUs, 90 GB, outer wall time `167:59:50`, NVHPC
24.7, CUDA 12.5.1, and `/home/pghosh/venvs/6adctmrg_Kuma`.  It deliberately
has no account directive.

```bash
# From local PowerShell:
scp -r .\models\VBCLowJ2ContinuationKumaExtended pghosh@kuma:/scratch/pghosh/

# On Kuma:
cd /scratch/pghosh/VBCLowJ2ContinuationKumaExtended
bash submit_all.sh --dry-run
bash submit_all.sh
```

Results are written under `Results_Kuma_lowJ2_extended/`.

