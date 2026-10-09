from pathlib import Path
import sys
import json
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src_code/scripts"))
from renyi2_spectral import solve_block
torch.set_num_threads(2)
diagonal = torch.tensor(np.r_[np.ones(7), .94, np.linspace(.7, .02, 88)])
rows = []
for block in (4, 8):
    answer = solve_block(lambda v: diagonal * v, 96, 8, block_size=block,
                         subspace=32, max_matvec=1600)
    assert answer.converged and np.count_nonzero(abs(answer.eigenvalues - 1) < 1e-8) == 7
    rows.append({"name": "sevenfold_larger_than_starting_block", "block": block,
                 "matvec": answer.matvec_count, "restarts": answer.restarts,
                 "returned_roots": answer.eigenvalues.real.tolist(),
                 "gram_error": answer.orthogonality_error,
                 "random_completions": answer.random_completions,
                 "rank_deflations": answer.rank_deflations})

for dimension in (3, 35):
    n = 80
    diagonal = torch.tensor(np.r_[np.linspace(1., .01, dimension), np.zeros(n - dimension)])
    mask = torch.tensor(np.r_[np.ones(dimension), np.zeros(n - dimension)])
    answer = solve_block(lambda v: diagonal * v, n, dimension,
                         block_size=4, subspace=64, max_matvec=300,
                         projector=lambda v: mask * v)
    assert answer.converged and len(answer.eigenvalues) == dimension
    assert np.max(abs(np.sort(answer.eigenvalues.real) - np.sort(diagonal[:dimension].numpy()))) < 1e-12
    rows.append({"name": "exhausted_projected_space", "image_dimension": dimension,
                 "reason": answer.reason, "matvec": answer.matvec_count,
                 "gram_error": answer.orthogonality_error})

Path(__file__).with_suffix(".json").write_text(
    json.dumps({"all_checks_passed": True, "rows": rows}, indent=2), encoding="utf-8")
print(json.dumps(rows, indent=2), flush=True)
