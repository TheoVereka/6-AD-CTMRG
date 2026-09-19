# Izar pure-LBFGS VBC continuation bundle

The directory keeps its existing name, but the production launchers now
target Izar exclusively. Nothing here imports code from
`VBCPinningClusterBundle`; only the one-time local raw-data migration reads
that legacy directory.

## Physics and numerical protocol

- replica 1 only, resumed from native `0713summary` twoC3 tensors;
- physical `plaquette` and `dimer-plaquette` pinning sources only;
- `h = 0.08, 0.04, 0.02, 0.01, 0.005`, with no `h=0` optimization;
- pure L-BFGS; the local `main_C3_LBFGS.py` rejects Adam;
- fixed Izar chi: `D6/chi72`, `D7/chi91`, `D8/chi104`, `D9/chi108`;
- every field resumes the preceding field, while a resubmitted job resumes an
  interrupted field from its own latest checkpoint.

The 80 independent jobs are submitted in this printed order:

1. Seven-day D9 jobs for
   `J2 = 0.275, 0.27, 0.28, 0.265, 0.26, 0.33, 0.34`.
2. Three-day D8, D7, then D6 jobs for
   `0.275, 0.27, 0.28, 0.265, 0.26, 0.29, 0.30, 0.31, 0.32, 0.33, 0.34`.
3. At each `(J2,D)`, plaquette is submitted before dimer-plaquette.

Slurm does not guarantee FIFO start order; the order above is submission and
queue-age priority, not a dependency chain that would leave GPUs idle.

## Izar allocations

| D | QOS | wall time | optimizer budget per h | total optimizer budget | reserve |
|---|---|---:|---:|---:|---:|
| 6, 7, 8 | `normal` | `71:59:50` | 14 h | 70 h | 1 h 59 min 50 s |
| 9 | `long` | `167:59:50` | 33 h | 165 h | 2 h 59 min 50 s |

Both launchers inherit the established Izar resource configuration:
`partition=gpu`, one GPU, one CPU, 40 GB host RAM, and `--exclude=i39`.
The reserve covers L-BFGS outer-step overshoot, lookahead observables,
checkpoint writes, plotting, cleanup, and five Python startups.

## Submit

Copy this whole directory to Izar, then preview all 80 jobs:

```bash
cd VBCPinningLyraLBFGS
bash submit_izar_replica1.sh --dry-run
```

Submit after checking the printed order:

```bash
bash submit_izar_replica1.sh
```

Results are written to `Results_Izar_replica1`. Re-running the submit command
is safe: completed h stages are skipped, and interrupted stages resume.

## Local archive and postprocessing

The authoritative local raw archive is:

```text
D:\HyraiOn\ENS_Lyon\Internship\2026-EPFL\data\distinVBCs
```

Its two discovery roots are `Results_Izar_replica1` and
`Results_Kuma_replica1`. Copy future Izar results directly into the former,
merging with the existing tree. Then run:

```powershell
python visual_elements\figs\VBCDiscriminator\plot_pinning_supervisor.py
```

That single command non-destructively imports any newer replica-1 files still
present under either model bundle, reads every available raw observation in
the archive, updates supervisor figures and processed CSVs, and regenerates
`quadratic_pinning_extrapolation`. Incomplete continuations remain visible in
the supervisor plots. A quadratic fit is included dynamically once it has at
least four distinct positive fields; insufficient groups are recorded in
`omitted_incomplete_fits.csv` rather than silently hard-coded away.
