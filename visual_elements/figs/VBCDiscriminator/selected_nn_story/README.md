# Selected NN-correlation story plots

Run from the repository root:

```powershell
python .\visual_elements\figs\VBCDiscriminator\selected_nn_story\select_and_plot.py
```

The script scans the original 0713 2C3 data, all currently downloaded D=5--9
Izar continuations, the D=10/11 Sep27 bundle, the previous h=0 candidate pool,
and the dedicated D7/J2=.26 repair snapshot.  For that live repair snapshot it
requires the atomic `COMPLETED.stage` marker in addition to the tensor and
observation, so it is safe to run on half-finished cluster downloads.

Outputs under `plots/` include one selected-data CSV, a complete candidate
audit, the fixed-energy-length diagnostic table, one two-panel NN-vs-J2 PDF,
one two-panel raw NN-vs-1/D PDF per J2, and separate D-by-J2 provenance tables
for the dimer-plaquette and plaquette configurations.  No fit is drawn in any
figure.

`plots/energy_vs_inverse_D/` contains exactly two energy-vs-1/D PDFs, one per
configuration, with every J2 and D included.  Dimer-plaquette J2 values use a
red gradient and plaquette J2 values use a blue gradient.  Selected energies
are scatter-only.  For each J2, the corresponding archived figure-24
original-2C3 gapped curve is drawn in black, with J2 encoded by transparency;
these curves are not refits of the selected data.

The three NN ranks use the same encoding in every PDF: strongest is a large
circle with a solid line, middle is a moderately enlarged square with a dashed
line, and weakest is an upward triangle at the original size with a dotted
line.  The D=11 plaquette continuation at J2=.26, .265, and .27 is excluded
because 0713 has no energy/Delta reference at those points.

## Signed pinning-energy surface and crossing line

Run the independent signed-field analysis with:

```powershell
python .\visual_elements\figs\VBCDiscriminator\selected_nn_story\pinning_energy_phase_boundary.py
```

It maps plaquette pinning to positive `h` and dimer-plaquette pinning to
negative `h`.  Set A requires at least two distinct nonzero fields on each
side for the same `(J2,D)`; `h=0` does not count towards this coverage test.
Set B retains every observation belonging to Set A.  Rank-split pinning is
excluded because it is a different direction in order-parameter space.

`plots/pinning_energy_phase_boundary/01_signed_h_energy_surface.pdf` is the
literal all-B plot, including failed runs.  The `01b_...` companion applies
the documented variational, texture, chi, and energy screening.  Only points
with identical `(J2,D)` are joined.  `fit_diagnostics/` contains one PDF per
J2 showing the finite-D points/curves and the two black D-to-infinity branch
curves used to infer the crossing.  The final `02_...` plot connects only
crossings that pass the fit and signed-response checks; rejected estimates
remain visible as grey crosses.  Every inclusion, rejection, fit residual,
finite-D crossing, and D-to-infinity result is exported to the adjacent CSV
audit files.

`03_all_good_points.pdf` is the deliberately inclusive comparison surface.
It uses every Set-B point that passes the broad original-2C3 energy envelope,
the pinning-texture consistency check, and a permissive smooth/monotonic E(h)
curve check at fixed `(J2,D,branch)`.  It performs no chi cutoff, no
CTMRG-lookahead cutoff, and no lowest-energy/variational-winner selection.  All
points, including `h=0`, are small circles.  The exact 03 membership is
exported as `all_good_points.csv`; every rejection and its curve residual are
recorded in `all_good_points_audit.csv`.  A thick original-2C3 D=10 reference
curve is drawn first at h=0, underneath every pinning layer: solid for
`0.24<=J2<=0.275` and dashed for `0.275<=J2<=0.34`.

For `02_hc_vs_J2_phase_boundary.pdf`, `04_phasediagram.pdf`, and
`05_E_crossing_vs_original_2C3.pdf`, reruns are first median-reduced at fixed
`(J2,D,h,branch)`.  Every finite-D branch is fitted first as
`E_s,D(h)=A_s,D+B_s,D h+C_s,D h^2` using equal field weight.  An h=0 datum is
discarded only if a leave-zero-out quadratic misses it by more than both
`3e-4` and six times the nonzero-field RMS.  The two branches are crossed at
each D to obtain `(h_c,D +/- sigma_h,D, E_crossing,D +/- sigma_E,D)`.
`E_crossing,D` is fitted as `E_crossing+k exp[-a_g(J2)D]` with
`1/sigma_E,D^2`; the covariance is conservatively inflated by reduced chi2 and
includes archived-`a_g` uncertainty.  No D fit is applied to `h_c,D`.  Instead,
`w_D=1/[sigma_h,D |E_crossing,D-E_crossing|]` gives its normalized statistical
weight.  The central value is the weighted mean, and asymmetric 1-sigma bars
are the 15.87%--84.13% quantiles of the corresponding Gaussian mixture.
Every local crossing and final weight is recorded in
`energy_crossings_by_D.csv`.
For the block background, reruns are first averaged within D.  All contiguous
high-D suffixes containing the two largest available D values are tested with
equal D weight and one common window for all three ranks.  The retained suffix
minimizes the largest displacement between the extrapolated triplet and the
actually observed largest-D triplet; fit residuals do not choose the window.
The three correlations are separately extrapolated by
`C_r(D)=C_r,inf+k_r exp[-a_g(J2)D]`, with `a_g` fixed by the original-2C3
gapped energy fit; only then are Delta and q formed.  The hue is
`q=(Cweak+Cstrong-2Cmid)/(Cweak-Cstrong)`: dimer-plaquette is `q=-1` (red),
plaquette is `q=+1` (blue).  The linear brightness scale is the extrapolated
`Delta=Cweak-Cstrong`.  No h=0 value of q or Delta is imposed.  At h=0, the
three selected dimer and three selected plaquette correlations are fitted
separately, corresponding ranks are averaged, and q/Delta are computed from
that averaged triplet.  At J2 values absent from the selected table, the same
two-sector construction uses the all-good h=0 data.
Every D and every sampled field has equal weight, independent of how many
reruns are present.

`Delta_vs_J2_selected.pdf` is generated by `select_and_plot.py` from exactly
the selected data underlying `NN_corr_vs_J2_selected.pdf`, replaces each
correlation triplet by `Delta=Cweak-Cstrong`, and overlays the independently
reconstructed h=0 thermodynamic Delta in both texture panels.  The energy
phase-boundary script does not write or overwrite this selected-story figure.

Completed targeted Kuma repairs are read from
`data/distinVBCsKumaTargetedRepairs/Results_Kuma_TargetedRepairs`.  The h=0
D10/J2=.28 dimer point is added; the h=0 D10 and D11/J2=.32 plaquette points
replace their previous selections.  Repair candidates are applied only after
the established global selection, so they cannot change unrelated grid points.
`05_E_crossing_vs_original_2C3.pdf` compares the energy at the fitted pinning
branch crossing with the archived original-2C3 thermodynamic energy.
