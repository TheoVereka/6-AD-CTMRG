"""Reproduce the archived solver failure, then apply the repaired solver."""
from dataclasses import asdict
from pathlib import Path
import importlib.util
import json
import sys
import numpy as np
import torch


def record_basis(*unused):
    pass


def record_orth(*unused):
    pass


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src_code/scripts"))
import renyi2_twoc3 as core
import renyi2_spectral as repaired
archived = ROOT / "tmp_tee_schematic_20261009/algorithm_checks/krylov_diagnosis_20261009/renyi2_spectral_instrumented.py"
spec = importlib.util.spec_from_file_location("original_solver_archived", archived)
original = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = original
spec.loader.exec_module(original)
archive = np.load(archived.parent.parent / "tiny_checkpoint_pair1_edges.npz")
edges = core.Edges(*(torch.tensor(archive[name], dtype=torch.float64) for name in "ABCD")).normalized()
torch.set_num_threads(2)
rows = []
for name, solver in (("original", original), ("repaired", repaired)):
    for parity in (1, -1):
        operator = core.ReplicaTransfer(edges, batch=2, parity=parity)
        result = solver.solve_block(operator, edges.chi**4, 8,
                                     subspace=80, block_size=4, max_matvec=800,
                                     projector=operator.project)
        row = {"solver": name, "parity": parity, "converged": result.converged,
               "matvec": result.matvec_count, "restarts": result.restarts,
               "largest_returned_modulus": float(max(abs(result.eigenvalues))),
               "maximum_residual": float(max(result.relative_residuals)),
               "gram_error": result.orthogonality_error, "reason": result.reason}
        print(json.dumps(row), flush=True)
        rows.append(row)
assert all(not row["converged"] for row in rows if row["solver"] == "original")
assert all(row["converged"] for row in rows if row["solver"] == "repaired")
Path(__file__).with_suffix(".json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
