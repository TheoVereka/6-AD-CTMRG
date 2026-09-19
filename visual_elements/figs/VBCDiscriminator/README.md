# twoC3 VBC discriminator

For sorted NN correlations `C1 <= C2 <= C3` (more negative means the
stronger AF bond), the ordinary splitting

```text
Delta = C3-C1
```

measures the VBC amplitude but does not distinguish plaquette from
dimer-plaquette texture. The signed coordinate used here is

```text
omega1 = C2-C1
omega2 = C3-C2
eta = (omega1-omega2)/(omega1+omega2)
```

Thus ideal dimer-plaquette has `eta=+1`, ideal plaquette has `eta=-1`, and
eta is undefined when all three correlations coincide.

## Authoritative raw archive

All replica-1 raw observations, logs, hyperparameters, and tensors live under

```text
D:\HyraiOn\ENS_Lyon\Internship\2026-EPFL\data\distinVBCs
  Results_Izar_replica1\...
  Results_Kuma_replica1\...
```

`sync_distin_vbcs.py` non-destructively imports replica-1 files from the old
and new model bundles. It never deletes archive files, and an older legacy
file cannot overwrite a newer file copied directly from a cluster.

Future `Results_Izar_replica1` downloads should be merged directly into the
archive directory of the same name.

## One-command postprocessing

From the repository root, run:

```powershell
python visual_elements\figs\VBCDiscriminator\plot_pinning_supervisor.py
```

This command performs the complete update:

1. synchronize any newer local replica-1 raw files into `distinVBCs`;
2. strictly parse every completed observation in both archive trees;
3. regenerate the two supervisor PDFs for every cluster/J2;
4. update per-pin processed CSVs in
   `data/processed/VBCPinningSupervisorCSV`;
5. refit all dynamically eligible positive-field continuations;
6. regenerate `quadratic_pinning_extrapolation` and update
   `data/processed/VBCPinningQuadraticExtrapolation`.

Available fields are discovered from the data, including both the legacy
`h=0` endpoints and new `h=0.005` endpoints. J2 tags use at least two and at
most three decimal places: for example `J2_0p30`, `J2_0p265`, and
`J2_0p275`. The default supervisor command plots and fits D6 through D10;
D5 and D11 are excluded. This can be overridden with `--dimensions`.

An unfinished job directory without an observation is harmless. A discovered
observation missing its `hyperparams.yaml`, or one that cannot be parsed,
aborts before new outputs are written; it is never silently skipped.

## Quadratic continuation fit

The three sorted correlations are fitted independently using only `h>0`:

```text
C(h) = C0 + c1 h + c2 h^2 .
```

The extrapolated splitting is

```text
Delta0 = Cweakest(0)-Cstrongest(0).
```

The linear-response diagnostic compares the two fitted contributions at the
reference field `h_ref=0.02`, not the dimensionful coefficients in isolation:

```text
R(h_ref) = |c2 h_ref^2| / |c1 h_ref| = |c2/c1| h_ref .
```

The quadratic correction is classified as subleading when `R(0.02) < 1`.

Every cluster/J2/D/physical-pin group with at least four distinct positive
fields is fitted. This is a data-driven eligibility condition, not a hardcoded
J2 or D skip. Partial groups are listed in
`omitted_incomplete_fits.csv`, including exactly which fields exist, and join
the fit automatically after enough SCP data arrives.

`rank-split` is displayed by the supervisor when available but is excluded
from the two physical-branch quadratic extrapolation. Measured h=0 points are
shown as diagnostics and are not included in the fit.
