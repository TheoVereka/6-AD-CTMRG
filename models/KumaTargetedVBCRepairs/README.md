# Targeted Kuma VBC repairs

This self-contained bundle launches three pure-LBFGS two-stage chains.  Each
chain first optimizes the target J2 with the appropriate h=0.02 pinning field,
then an `afterok` job resumes that tensor at h=0.

| alias | seed | target | branch | D, chi | action in selected story |
|---|---|---|---|---|---|
| r01 | J2=0.29 dimer | J2=0.28 | dimer-plaquette | 10, 120 | add |
| r02 | J2=0.31 plaquette | J2=0.32 | plaquette | 10, 120 | replace |
| r03 | J2=0.31 plaquette | J2=0.32 | plaquette | 11, 140 | replace |

The static `.run` files exactly follow the previously successful Kuma setup:
`long` QoS, `h100` partition, one GPU, 16 CPU cores, 90 GB host memory,
`167:59:50` external wall time, NVHPC 24.7 MPI, CUDA 12.5.1 and the existing
`/home/pghosh/venvs/6adctmrg_Kuma` environment.  No account is hard-coded.
The D=10 stages have a 48 h internal limit and D=11 stages 52.8 h.  Tensor and
CTM initialization noise are at most 1e-3; rSVD approximation is unchanged.

Upload the complete folder to `/scratch/pghosh` and launch it with one command:

```powershell
scp -r .\models\KumaTargetedVBCRepairs pghosh@kuma:/scratch/pghosh/
```

```bash
cd /scratch/pghosh/KumaTargetedVBCRepairs
bash submit_all.sh
```

This submits exactly `3 heads + 3 afterok jobs`.  Results are written beneath
`Results_Kuma_TargetedRepairs` and incomplete stages safely resume from their
latest checkpoint when resubmitted.

After completion, download exactly this result directory to:

```text
D:\HyraiOn\ENS_Lyon\Internship\2026-EPFL\data\distinVBCsKumaTargetedRepairs\Results_Kuma_TargetedRepairs
```

For example, from the repository root in PowerShell:

```powershell
New-Item -ItemType Directory -Force `
  "D:\HyraiOn\ENS_Lyon\Internship\2026-EPFL\data\distinVBCsKumaTargetedRepairs" | Out-Null
scp -r pghosh@kuma:/scratch/pghosh/KumaTargetedVBCRepairs/Results_Kuma_TargetedRepairs `
  "D:\HyraiOn\ENS_Lyon\Internship\2026-EPFL\data\distinVBCsKumaTargetedRepairs\"
```

Then run `refresh_selected_nn_story.ps1` from this folder on the local machine.
The refresh first rebuilds the selected points and crosses the fitted energy
branches at every D.  It error-weights the gapped extrapolation of
`E_crossing,D`, statistically combines `h_c,D` by uncertainty and energy
proximity, and finally regenerates all dependent plots.
It then applies the canonical `PublicationPlots` stylesheet and writes the six
final PDFs plus the refreshed 180-degree MP4 to
`visual_elements\figs\VBCDiscriminator\selected_nn_story\pubPlots`.
The MP4 step requires `ffmpeg` on `PATH`.
