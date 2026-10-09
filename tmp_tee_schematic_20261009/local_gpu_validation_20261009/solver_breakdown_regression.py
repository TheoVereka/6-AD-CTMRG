"""Dense reference checks for the previously failing CTM input and tiny scales."""
from pathlib import Path
import importlib.util
import json
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src_code/scripts"))
from renyi2_spectral import solve_block
import renyi2_twoc3 as core

old = ROOT / "tmp_tee_schematic_20261009/algorithm_checks/verify_renyi2_core.py"
spec = importlib.util.spec_from_file_location("explicit_reference", old)
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)

torch.set_num_threads(2)
rows = []
lengths = np.arange(100, 1001, 2)
for pair in (1, 2, 3):
    archive = np.load(old.parent / f"tiny_checkpoint_pair{pair}_edges.npz")
    edges = core.Edges(*(torch.tensor(archive[name], dtype=torch.float64) for name in "ABCD")).normalized()
    t1, t2 = reference.explicit_transfers([getattr(edges, name).numpy() for name in "ABCD"])
    target, _ = core.entropy_from_spectra(np.linalg.eigvals(t1), np.linalg.eigvals(t2), lengths)
    for count in (8, 16, 32):
        block = max(4, count // 2)
        one = solve_block(lambda v: torch.from_numpy(t1) @ v, edges.chi**2,
                          min(count, edges.chi**2), block_size=block,
                          subspace=64, max_matvec=1600)
        results = []
        for parity in (1, -1):
            operator = core.ReplicaTransfer(edges, batch=2, parity=parity)
            result = solve_block(operator, edges.chi**4,
                                 min(count, edges.chi**2 * (edges.chi**2 + parity) // 2),
                                 block_size=block, subspace=64, max_matvec=1600,
                                 projector=operator.project)
            results.append(result)
        all_results = [one] + results
        assert all(r.converged for r in all_results), [(r.reason, r.relative_residuals) for r in all_results]
        got, _ = core.entropy_from_spectra(one.eigenvalues,
                                          np.concatenate([r.eigenvalues for r in results]), lengths)
        error = float(np.max(np.abs(got - target)))
        row = {"pair": pair, "modes": count, "L_min": 100, "L_max": 1000,
               "lengths_compared": len(lengths), "max_absolute_S2_dense_error": error,
               "solvers": [{"matvec": r.matvec_count, "restarts": r.restarts,
                            "max_residual": float(max(r.relative_residuals)),
                            "gram_error": r.orthogonality_error,
                            "deflations": r.rank_deflations,
                            "completions": r.random_completions,
                            "gram_repairs": r.gram_repairs} for r in all_results]}
        print(json.dumps(row), flush=True)
        assert error < 1e-4, row
        assert all(r.orthogonality_error < 1e-9 for r in all_results), row
        rows.append(row)

# Rank decisions must be relative to the operator, never an absolute 1e-14.
diagonal = torch.tensor([1., .8, .6, .4, .2, .1, .05, .01], dtype=torch.float64)
for scale in (1e-100, 1., 1e100):
    answer = solve_block(lambda v: scale * diagonal * v, 8, 4,
                         subspace=8, block_size=2, max_matvec=100)
    assert answer.converged, answer
    assert np.allclose(np.sort(answer.eigenvalues.real / scale), [.4, .6, .8, 1.], atol=1e-12)
    rows.append({"name": "scale_invariance", "operator_scale": scale,
                 "max_residual": float(max(answer.relative_residuals)),
                 "gram_error": answer.orthogonality_error})

Path(__file__).with_name("solver_breakdown_regression.json").write_text(
    json.dumps({"all_checks_passed": True, "rows": rows}, indent=2), encoding="utf-8")
