# LastIzar

Static Izar bundle for the final D=5 and D=6 plaquette-state tests.  No run
files are generated on the cluster: `jobs/` already contains every Slurm job,
and `submit_all.sh` only submits them and adds `afterok` dependencies.

## Numerical choices

- D=5: chi=50; D=6: chi=72.
- Every run passes `--ctm-max-steps 50`; no stage may use a larger value.
- Pin: plaquette, orientation 0, h=0.005; the second stage restores h=0.
- Adam/L-BFGS learning rates, tolerances, history, and related optimizer
  hyperparameters come from the two bundled main files; the launcher does not
  override them.  Randomized SVD is unchanged.
- `main_C3.py` is the normal Adam-warmup then L-BFGS implementation.
- `main_C3_LBFGS.py` rejects Adam and runs pure L-BFGS.  For its independent
  h=0.005 jobs it still constructs the requested mean-field starting tensor.
- D=6 adiabatic task starts from the bundled original 0713summary tensor at
  J2=0.265.  Every later J2 stage resumes the preceding stage's best tensor.
- Every h=0 job has an `afterok` dependency on its own h=0.005 job.

## Izar resources

- 3-day jobs: qos `normal`, partition `gpu`, wall time `71:59:50`.
- 7-day jobs: qos `long`, partition `gpu`, wall time `167:59:50`.
- One GPU, one CPU, 40G RAM, and node `i39` excluded.
- No Slurm account is forced; this is identical to the previously successful
  Izar launch headers.

## Upload and launch

From Windows PowerShell, from the repository root:

```powershell
scp -r .\models\LastIzar chye@izar.hpc.epfl.ch:~/
```

On Izar:

```bash
cd ~/LastIzar
bash submit_all.sh --dry-run
bash submit_all.sh
```

The dry run must end with `71 jobs = 33 heads + 38 afterok jobs`.

Results are written under `Results_LastIzar/`; logs are written under
`slurm_logs/`.  A manually resubmitted stage skips a complete tensor/observable
pair or resumes its own latest checkpoint.
