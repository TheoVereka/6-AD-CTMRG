# Lyra pure-LBFGS VBC continuation bundle

Target sweep:

- replica 1 only, resumed from the `0713summary` twoC3 tensors;
- exact J2 priority order: `0.275, 0.27, 0.28, 0.265, 0.29, 0.26, 0.30, 0.31, 0.32, 0.33, 0.34`;
- `D = 6, 7, ..., 11`;
- no synthetic D11 start: skip a `(J2,D=11)` point when `0713summary`
  has no native D11 tensor;
- physical pinning sources `plaquette` and `dimer-plaquette` only;
- continuation fields `h = 0.08, 0.04, 0.02, 0.01, 0.005` (no `h=0`);
- pure L-BFGS: Adam must be impossible to select;
- 126 independent branch jobs in total (`63` native seeds times two branches).

## Wall-time classes

The production array is split into two Slurm resource classes:

| D | QOS | requested wall time | per-field optimizer budget | five-field total | reserve |
|---|---|---:|---:|---:|---:|
| 6, 7, 8, 9 | `normal` | `2-23:59:00` | `14.0 h` | `70 h` | `1 h 59 min` |
| 10, 11 | `long` | `6-23:59:00` | `33.0 h` | `165 h` | `2 h 59 min` |

These budgets use about 97--98% of the allocation. A literal division of the
Slurm limit by five is unsafe: `--hours` limits the optimizer loop only. Each
stage may overrun by its final L-BFGS outer step and then performs a chi
lookahead, observable/checkpoint writes, plots, cleanup, and Python startup
outside that budget. The small reserve protects the final `h=0.005` stage.
The Lyra D11 pilot will measure this overhead; budgets may only be increased
if the measured worst-case reserve remains safe.

The final production layout will use one restartable job per
`(J2,D,branch)`. A re-submission of the same job skips completed field stages
and resumes an interrupted stage from its latest/best checkpoint. This is
important during Lyra beta testing because running jobs may be terminated.

## Seed provenance

`prepare_0713_seeds.py` creates a self-contained `seeds/` tree and
`seed_manifest.csv`. It keeps the requested J2 priority order, and D is
ascending inside each J2. There are 63 native same-`J2`, same-`D` tensors.
`D=11` is absent for `J2=0.265`, `0.27`, and `0.26`, so those three jobs are
omitted rather than initialized by padding a D10 tensor. The manifest records
both target D and actual seed D; they must be identical for every row.

## Staged deployment plan

1. Run `probe_lyra.sh` from the Lyra frontend. It records Slurm associations,
   QOS/partition configuration, storage variables, available compiler/CUDA/
   Python modules, and a short GPU-node probe on both Blackwell partitions.
2. Generate `setup_venv_Lyra.sh` from the observed Lyra module names and test
   a Blackwell-capable PyTorch build on both partitions.
3. Run one one-hour debug pilot per partition at a small D, then a production
   pilot at D11 to measure GPU/host memory and iteration speed.
4. Select the faster stable partition and set its correct CPU ratio (12 cores
   per B200 or 16 per RTX6000). Do not request a guessed `--mem`; Lyra assigns
   host RAM from the CPU request.
5. Dry-run the 126-entry job manifest, submit jobs in the requested J2 order
   and low-to-high D order within each J2,
   and verify early logs before releasing the full array.
6. Re-submit failed/preempted array elements against the same output root;
   checkpoint continuation prevents completed stages from being repeated.

No production launcher is finalized before step 1 because module names,
account/QOS access, and the usable PyTorch/CUDA combination must be observed
on Lyra rather than copied from Izar or Kuma.
