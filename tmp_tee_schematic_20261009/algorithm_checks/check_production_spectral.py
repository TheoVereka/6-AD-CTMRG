"""Small independent dense checks for the matrix-free block solver."""
import json
from pathlib import Path

import numpy as np
import scipy.linalg as sla
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src_code" / "scripts"))
from renyi2_spectral import (
    entropy_from_spectra,
    solve_block,
    tail_radius_ratio_limit,
)


def main():
    torch.set_num_threads(2)
    rng = np.random.default_rng(1178)
    checks = []

    def run(name, matrix, *, k=6, block_size=4, subspace=24, projector=None):
        tensor = torch.as_tensor(matrix, dtype=torch.float64)
        result = solve_block(lambda vector: tensor @ vector, len(matrix), k,
                             block_size=block_size, subspace=subspace,
                             max_matvec=1600, tol=1e-10, seed=917,
                             projector=projector)
        exact = sla.eigvals(matrix)
        exact = exact[np.argsort(-np.abs(exact), kind="stable")]
        distances = [float(np.min(np.abs(exact - value))) for value in result.eigenvalues]
        lead = abs(exact[k - 1])
        wanted = exact[np.abs(exact) >= lead - 1e-9]
        # Cluster cardinality and optimal assignment catch duplicate roots.
        from scipy.optimize import linear_sum_assignment
        costs = np.abs(wanted[:, None] - result.eigenvalues[None, :])
        rows, columns = linear_sum_assignment(costs)
        matching_error = float(np.max(costs[rows, columns]))
        record = {"name": name, "converged": result.converged,
                  "reason": result.reason, "roots_returned": len(result.eigenvalues),
                  "max_relative_residual": float(np.max(result.relative_residuals)),
                  "max_eigenvalue_match_error": matching_error,
                  "wanted_cluster_size": len(wanted),
                  "matvec_count": result.matvec_count,
                  "restarts": result.restarts,
                  "orthogonality_error": result.orthogonality_error}
        print(record, flush=True)
        assert result.converged, record
        assert len(wanted) <= len(result.eigenvalues), record
        assert matching_error < 1e-7, record
        assert result.orthogonality_error < 1e-9, record
        checks.append(record)

    n = 96
    rotation = sla.qr(rng.normal(size=(n, n)))[0]
    values = np.r_[1., .99, .98, .96, .94, .91, np.linspace(.78, .03, n - 6)]
    run("normal_real_requires_restarts", rotation @ np.diag(values) @ rotation.T)

    values = np.r_[np.ones(5), .94, np.linspace(.70, .02, n - 6)]
    run("exact_degeneracy_5_block_8", rotation @ np.diag(values) @ rotation.T,
        k=5, block_size=8, subspace=32)

    matrix = np.diag(np.r_[1., np.linspace(.70, .02, n - 1)])
    for start, radius, theta in ((1, .98, .30), (3, .95, .51)):
        matrix[start:start + 2, start:start + 2] = radius * np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    run("complex_pairs_real_basis", rotation @ matrix @ rotation.T, k=4)

    values = np.r_[1., .99, .98, .96, .94, .91, np.linspace(.78, .03, n - 6)]
    matrix = np.diag(values) + np.diag(np.full(n - 1, .025), 1)
    run("nonnormal_real_matrix", rotation @ matrix @ rotation.T)

    image = 35
    diagonal = torch.as_tensor(np.r_[np.linspace(1., .01, image), np.zeros(45)])
    mask = torch.as_tensor(np.r_[np.ones(image), np.zeros(45)])
    projected = solve_block(lambda vector: diagonal * vector, 80, 4,
                            subspace=18, block_size=4, max_matvec=1000,
                            tol=1e-10, projector=lambda vector: vector * mask)
    assert projected.converged
    assert np.allclose(np.sort(projected.eigenvalues.real), np.sort(diagonal[:4].numpy()), atol=1e-9)
    checks.append({"name": "orthogonal_projector", "converged": projected.converged,
                   "matvec_count": projected.matvec_count, "restarts": projected.restarts,
                   "max_relative_residual": float(max(projected.relative_residuals))})

    failed = solve_block(lambda vector: diagonal * vector, 80, 8, max_matvec=5,
                        subspace=24)
    assert not failed.converged and failed.reason == "max_matvec_reached"
    checks.append({"name": "failure_not_silently_successful", "converged": failed.converged,
                   "reason": failed.reason})

    # Directly summing lambda**1000 would overflow these two scaled spectra.
    t1 = np.array([1000., 1000., 700.])
    t2 = np.array([500000., 500000., 500000., 100000.])
    entropy = entropy_from_spectra(t1, t2, [50, 500, 1000])
    expected = [power * (2 * np.log(1000) - np.log(500000)) + 2 * np.log(2) - np.log(3)
                for power in (50, 500, 1000)]
    assert np.allclose([item["S2"] for item in entropy], expected, atol=2e-8)
    checks.append({"name": "scaled_trace_powers_no_overflow", "entropies": entropy})

    # Tiny backward residual can accompany a very inaccurate eigenvalue.
    a = np.array([[1., 1e6], [0., 1. - 1e-4]])
    approximation = 1. + 1e-3
    v = np.array([1., 1e-3 / 1e6])
    residual = np.linalg.norm(a @ v - approximation * v) / (
        np.linalg.norm(a @ v) + approximation * np.linalg.norm(v))
    assert residual < 1e-10 and min(abs(sla.eigvals(a) - approximation)) > 1e-4
    checks.append({"name": "nonnormal_residual_is_not_forward_accuracy",
                   "relative_residual": residual, "eigenvalue_error": 1e-3})

    limits = {str(power): tail_radius_ratio_limit(64**4, 32, power, 2e-5)
              for power in (50, 100, 500, 1000)}
    checks.append({"name": "dimension_only_tail_radius_limits", "limits": limits})
    output = Path(__file__).with_name("production_spectral_results.json")
    output.write_text(json.dumps({"all_checks_passed": True, "checks": checks}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
