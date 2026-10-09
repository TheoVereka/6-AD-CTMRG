# Local GPU and true checkpoint inventory

GPU: `NVIDIA GeForce RTX 4060 Laptop GPU, 551.23, 8188, 7545`.
Default Python: `D:\Programs\Python312\python.exe`; torch `2.6.0+cpu`; CUDA available: `False`.
Python 3.10 exists at D:/Programs/Python310/python.exe but has no torch. No conda or repository venv was found.

## Actual local-test tensors

All five are selected 0713summary two-C3 checkpoints at J2=0.26, with finite real float64 a_raw/b_raw tensors.
Fractional 1.25 D² for odd D requires integer convention; the table records ceiling, without imposing it on production code.

| D | 1.25 D² | ceiling χ | optimized χ | original file |
|---:|---:|---:|---:|---|
| 2 | 5.0 | 5 | 20 | `2tensor_twoC3__J2_0p26_20260714_100445/sweep_D2_chi20_best.pt` |
| 3 | 11.25 | 12 | 24 | `2tensor_twoC3__J2_0p26_20260714_100445/sweep_D3_chi24_best.pt` |
| 4 | 20.0 | 20 | 36 | `2tensor_twoC3__J2_0p26_20260714_100445/sweep_D4_chi36_best.pt` |
| 5 | 31.25 | 32 | 50 | `0730newdata/Working_AD_Honeycomb_August19/Results_J2_twoc3_0.26/D_5/sweep_D5_chi50_best.pt` |
| 6 | 45.0 | 45 | 72 | `0730newdata/Working_AD_Honeycomb_August19/Results_J2_twoc3_0.26/D_6/sweep_D6_chi72_best.pt` |

## Eventual D8 target inventory

7 actual tensors in inclusive J2=[0.24,0.275], hence 21 eventual jobs if and only if the test gate passes.

- J2=0.24: `D:\HyraiOn\ENS_Lyon\Internship\2026-EPFL\data\0713summary\J2_0p24\2tensor_twoC3\D_8\tensor_best.pt`; optimized χ=104.
- J2=0.245: `D:\HyraiOn\ENS_Lyon\Internship\2026-EPFL\data\0713summary\J2_0p245\2tensor_twoC3\D_8\tensor_best.pt`; optimized χ=112.
- J2=0.25: `D:\HyraiOn\ENS_Lyon\Internship\2026-EPFL\data\0713summary\J2_0p25\2tensor_twoC3\D_8\tensor_best.pt`; optimized χ=104.
- J2=0.26: `D:\HyraiOn\ENS_Lyon\Internship\2026-EPFL\data\0713summary\J2_0p26\2tensor_twoC3\D_8\tensor_best.pt`; optimized χ=104.
- J2=0.265: `D:\HyraiOn\ENS_Lyon\Internship\2026-EPFL\data\0713summary\J2_0p265\2tensor_twoC3\D_8\tensor_best.pt`; optimized χ=104.
- J2=0.27: `D:\HyraiOn\ENS_Lyon\Internship\2026-EPFL\data\0713summary\J2_0p27\2tensor_twoC3\D_8\tensor_best.pt`; optimized χ=104.
- J2=0.275: `D:\HyraiOn\ENS_Lyon\Internship\2026-EPFL\data\0713summary\J2_0p275\2tensor_twoC3\D_8\tensor_best.pt`; optimized χ=112.

## Existing environment and disk

The GPU and driver are present, but default PyTorch is CPU-only. A CUDA-capable isolated environment is needed for actual GPU tests; no package was installed during this inventory.
Driver 551.23 advertises CUDA 12.4. An official PyTorch 2.6 cu124 build is the matching candidate; verify its import/device probe before running tests.
An isolated env can reuse CPU-installed scientific packages via system-site-packages, with torch overridden inside that env, to avoid mutating the existing global environment.

- C:/: free 50.50 GiB, total 200.00 GiB.
- D:/: free 56.14 GiB, total 551.64 GiB.

No cluster job was created. No checkpoint, source manifest or global environment was modified.
