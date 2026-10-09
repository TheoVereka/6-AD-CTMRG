# Low-J2 fixed-pinning plots

From the repository root on Windows, run:

```powershell
powershell -ExecutionPolicy Bypass -File .\visual_elements\figs\VBCDiscriminator\low_j2_pinning_continuation\download_izar_and_plot.ps1
```

The command uploads a tiny read-only packer, snapshots every completed Izar
stage, downloads the archive into
`data/external/VBCLowJ2ContinuationIzar`, combines it with the extracted Kuma
results in `data/external/VBCLowJ2ContinuationKuma/Results_Kuma_lowJ2`, and
regenerates all plots.

Izar `D=8` is rejected before CSV collection and plotting.  If the extended
Kuma bundle has been downloaded to
`data/external/VBCLowJ2ContinuationKumaExtended/Results_Kuma_lowJ2_extended`,
the same command discovers and merges it automatically.

For every requested signed field the output contains an NN-correlation plot,
an NN-splitting plot, and a staggered-magnetization plot.  The latter uses the
exact 2C3 publication definition: all 18 site/environment vectors are aligned
by the ACE/BDF sublattice sign, the central value is the x-z norm of their
mean, and the error bar is their full-vector RMS spread about that mean.

Completion requires the plain observation, final best tensor,
`sweep_results.json`, and `hyperparams.yaml`.  Lookahead observations are not
counted as separate jobs.  If Kuma and Izar both contain the same
`(signed h,D,J2)` coordinate, the plotted representative is the one with the
lowest variational energy; the full decision is recorded in the two CSV files.

