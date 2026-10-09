"""Cross-check final CLI output against the actual GPU validation evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'src_code' / 'scripts'))
import renyi2_twoc3 as routine
sys.path.insert(0, str(ROOT / 'tmp_tee_schematic_20261009' / 'algorithm_checks'))
from verify_renyi2_core import explicit_transfers

rows = []
for d, chi in ((2, 5), (4, 20)):
    path = HERE / 'production_final' / f'D{d}_chi{chi}_pair1.json'
    result = json.loads(path.read_text(encoding='utf-8'))
    csv = np.loadtxt(path.with_suffix('.csv'), delimiter=',', skiprows=1)
    reference_path = HERE / f'D{d}_chi{chi}' / 'pair1_validation.json'
    reference = json.loads(reference_path.read_text(encoding='utf-8'))
    expected = (reference['dense_reference']['S2'] if d == 2
                else reference['stages'][-1]['S2_with_full_T1'])
    backend_error = float(np.max(np.abs(csv[:, 1] - expected)))
    if d == 2:
        with np.load(HERE / 'production_final' / 'D2_chi5_edges.npz') as archive:
            values = [archive[name] / np.linalg.norm(archive[name]) for name in 'ABCD']
        one, two = explicit_transfers(values)
        expected, _ = routine.entropy_from_spectra(np.linalg.eigvals(one),
                                                  np.linalg.eigvals(two), csv[:, 0])
    error = float(np.max(np.abs(csv[:, 1] - expected)))
    stages = [stage['modes_requested_per_sector'] for stage in result['stages']]
    passed = (result['status'] == 'spectrally_stable_estimate'
              and stages == [8, 16]
              and result.get('stopped_after_stable_mode_increase') is True
              and len(csv) == 451 and error < 1e-4)
    rows.append({'D': d, 'chi': chi, 'status': result['status'],
                 'executed_modes': stages, 'max_absolute_S2_error': error,
                 'reference': 'independent_full_dense_T1_T2_same_CPU_edges' if d == 2 else 'actual_GPU_driver_full_T1',
                 'CPU_vs_GPU_CTM_S2_difference': backend_error if d == 2 else None,
                 'passed': passed, 'runtime': result['runtime']})

cached = routine._obtain_edges(argparse.Namespace(
    edge_file=str(HERE / 'production_final' / 'D2_chi5_edges.npz'), pair=1, device='cpu'))
cache_passed = cached.metadata.get('ctm_converged') is True and len(cached.metadata.get('ctm', [])) == 2
summary = json.loads((HERE / 'validation_summary.json').read_text(encoding='utf-8'))
solver_path = ROOT / 'src_code' / 'scripts' / 'renyi2_spectral.py'
solver_sha = hashlib.sha256(solver_path.read_bytes()).hexdigest()
same_solver = solver_sha == summary['runtime']['source_files_sha256'][r'src_code\scripts\renyi2_spectral.py']
report = {'passed': all(row['passed'] for row in rows) and cache_passed and same_solver,
          'cases': rows, 'edge_cache_preserves_CTM_metadata': cache_passed,
          'solver_identical_to_full_D2_D6_GPU_test': same_solver,
          'post_full_test_changes': 'Only stop-after-stable mode control flow, source hashes and runtime/VRAM logging; T1/T2 contractions, packed coordinates and eigensolver unchanged.',
          'final_source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (solver_path, solver_path.with_name('renyi2_twoc3.py'))}}
out = HERE / 'production_final' / 'final_cli_validation.json'
out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(report, ensure_ascii=True, indent=2))
raise SystemExit(0 if report['passed'] else 2)
