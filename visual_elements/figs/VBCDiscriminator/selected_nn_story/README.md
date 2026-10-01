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

The three NN ranks use the same encoding in every PDF: strongest is a large
circle with a solid line, middle is a moderately enlarged square with a dashed
line, and weakest is an upward triangle at the original size with a dotted
line.  The D=11 plaquette continuation at J2=.26, .265, and .27 is excluded
because 0713 has no energy/Delta reference at those points.
