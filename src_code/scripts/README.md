* add here example files showing how to run the code

* `renyi2_twoc3.py` computes real float64 CTM replica spectra using a repaired
  rank-revealing block solver and isometric packed replica sectors. Normal/swap
  pairs are 1/1, 2/3, 3/2. Run
  `python -B src_code/scripts/renyi2_twoc3.py --help`; examples and validation
  are in `../../tmp_tee_schematic_20261009/核心脚本说明.md`.
  Actual local GPU validation and its explicit pass gate are recorded in
  `../../tmp_tee_schematic_20261009/local_gpu_validation_20261009/本机GPU验证结果.md`.
  Low-D validation does not claim that D=8 has already been run locally.
