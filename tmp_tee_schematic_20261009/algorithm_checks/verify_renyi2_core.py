"""Independent checks of production renyi2_twoc3; no large-D PEPS job.

Small explicit NumPy transfer matrices are the reference, independently of
the production contraction path.  Long circumferences use full spectra and
direct scaled matrix powers, not the iterative eigensolver under test.
"""
from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src_code" / "scripts"))
import renyi2_twoc3 as production


def rel_error(value, reference):
    return float(np.linalg.norm(value - reference) / max(np.linalg.norm(reference), 1e-300))


def explicit_transfers(values):
    a, b, c, d = values
    chi = a.shape[2]
    # Contract all four tensors directly; do not call production.build_t1.
    t1 = np.einsum("abpi,baqj,cdkp,dclq->ijkl", d, c, a, b, optimize=True)
    upper = np.einsum("abip,bcjq,cdkr,dals->pqrsijkl", a, b, a, b, optimize=True)
    lower = np.einsum("abpi,bcqj,cdrk,dasl->ijklpqrs", d, c, d, c, optimize=True)
    return t1.reshape(chi**2, chi**2), lower.reshape(chi**4, chi**4) @ upper.reshape(chi**4, chi**4)


def purified_real_edge(rng, bond_D=2, virtual=2, purification=2):
    raw = rng.normal(size=(virtual, virtual, bond_D, purification))
    edge = np.einsum("uvap,UVbp->abuUvV", raw, raw).reshape(bond_D, bond_D, virtual**2, virtual**2)
    return edge / np.linalg.norm(edge)


def check_random_paths(rng):
    rows = []
    for bond_D, chi in itertools.product((2, 8), (2, 3)):
        values = [rng.normal(size=(bond_D, bond_D, chi, chi)) for _ in range(4)]
        values = [value / np.linalg.norm(value) for value in values]
        edges = production.Edges(*(torch.tensor(value, dtype=torch.float64) for value in values))
        t1, t2 = explicit_transfers(values)
        expected_vector = rng.normal(size=chi**4)
        vector = torch.tensor(expected_vector, dtype=torch.float64)
        permutation = np.arange(chi**4).reshape((chi,) * 4).transpose(2, 3, 0, 1).reshape(-1)
        one = rel_error(production.build_t1(edges).numpy(), t1)
        assert one < 2e-12
        commutator = rel_error(t2[:, permutation], t2[permutation, :])
        assert commutator < 2e-12
        for batch in (1, 2, chi, chi + 3):
            operator = production.ReplicaTransfer(edges, batch=batch)
            actual = operator(vector).numpy()
            full_error = rel_error(actual, t2 @ expected_vector)
            assert full_error < 2e-12
            sector_errors = {}
            for parity in (1, -1):
                pure = .5 * (expected_vector + parity * expected_vector[permutation])
                sector_operator = production.ReplicaTransfer(edges, batch=batch, parity=parity)
                observed = sector_operator(torch.tensor(pure, dtype=torch.float64)).numpy()
                reference = t2 @ pure
                sector_errors[str(parity)] = rel_error(observed, reference)
                assert sector_errors[str(parity)] < 2e-12
                assert rel_error(observed[permutation], parity * observed) < 2e-12
            rows.append({"bond_D": bond_D, "chi": chi, "requested_batch": batch,
                         "effective_batch": operator.batch, "T1_relative_error": one,
                         "T2_matvec_relative_error": full_error,
                         "swap_commutator_relative_error": commutator,
                         "sector_relative_errors": sector_errors})
    return rows


def check_marked_extraction():
    rows = []
    for bond_D in (2, 8):
        chi = 3
        normal, swapped = [], []
        for collection, tag in ((normal, 1), (swapped, 2)):
            for environment in (1, 2, 3):
                collection.append(torch.zeros((chi, chi), dtype=torch.float64))
                for edge in (1, 2):
                    marked = np.empty((chi, chi, bond_D**2), dtype=float)
                    for m, x, q in np.ndindex(marked.shape):
                        marked[m, x, q] = tag * 1e7 + environment * 1e5 + edge * 1e4 + m * 1e3 + x * 100 + q
                    collection.append(torch.tensor(marked, dtype=torch.float64))
            collection.append(7)
        # Explicit independently derived normal/swap index pairs.
        for pair, ni, si in ((1, 0, 0), (2, 3, 6), (3, 6, 3)):
            got = production.extract_pair(normal, swapped, pair)
            count = 0
            for a, b, i, p in itertools.product(range(bond_D), range(bond_D), range(chi), range(chi)):
                assert got.A[a, b, i, p] == normal[ni + 1][p, i, a * bond_D + b]
                assert got.D[a, b, i, p] == normal[ni + 2][i, p, a * bond_D + b]
                assert got.B[a, b, i, p] == swapped[si + 1][i, p, b * bond_D + a]
                assert got.C[a, b, i, p] == swapped[si + 2][p, i, b * bond_D + a]
                count += 4
            rows.append({"bond_D": bond_D, "chi": chi, "pair": pair,
                         "normal_env": ni // 3 + 1, "swap_env": si // 3 + 1,
                         "exact_marked_element_comparisons": count})
    return rows


def check_long_circumferences(rng):
    values = [purified_real_edge(rng) for _ in range(4)]
    edges = production.Edges(*(torch.tensor(value, dtype=torch.float64) for value in values))
    t1, t2 = explicit_transfers(values)
    eigen1, eigen2 = np.linalg.eigvals(t1), np.linalg.eigvals(t2)
    lengths = np.arange(100, 1001, 2)
    entropies, imaginary = production.entropy_from_spectra(eigen1, eigen2, lengths)
    assert np.isfinite(entropies).all()
    assert entropies.min() >= -1e-10
    assert np.all(entropies <= lengths * math.log(2) + 1e-10)
    radius1, radius2 = max(abs(eigen1)), max(abs(eigen2))
    direct_rows = []
    for length in (100, 102, 128, 200, 256, 500, 768, 998, 1000):
        power = length // 2
        z1 = np.trace(np.linalg.matrix_power(t1 / radius1, power))
        z2 = np.trace(np.linalg.matrix_power(t2 / radius2, power))
        expected = 2 * (power * np.log(radius1) + np.log(z1)) - (power * np.log(radius2) + np.log(z2))
        index = (length - 100) // 2
        error = float(abs(entropies[index] - expected))
        assert error < 2e-8
        direct_rows.append({"cut_bonds": length, "S2": float(entropies[index]),
                            "scaled_matrix_power_absolute_error": error})
    rescale_errors = {}
    for exponent in (-140, 140):
        rescaled, _ = production.entropy_from_spectra(eigen1 * 10.**exponent, eigen2 * 10.**(2 * exponent), lengths)
        error = float(np.max(abs(rescaled - entropies)))
        assert error < 2e-8
        rescale_errors[str(exponent)] = error
    np.savez(HERE / "positive_edges_cli.npz", **dict(zip("ABCD", values)))
    return {"bond_D": 2, "chi": edges.chi, "number_of_even_lengths": len(lengths),
            "length_min": 100, "length_max": 1000, "full_spectrum": True,
            "maximum_imaginary_trace_relative": imaginary,
            "matrix_power_checks": direct_rows,
            "large_scalar_rescaling_absolute_errors": rescale_errors}


def check_rejected_inputs():
    edge = torch.ones((2, 2, 2, 2), dtype=torch.float64)
    outcomes = []
    for label, operation, exception in (
        ("float32 edge", lambda: production.Edges(*(edge.float() for _ in range(4))), TypeError),
        ("complex128 edge", lambda: production.Edges(*(edge.to(torch.complex128) for _ in range(4))), TypeError),
        ("zero edge normalization", lambda: production.Edges(*(torch.zeros_like(edge) for _ in range(4))).normalized(), ValueError),
        ("odd cut count", lambda: production.entropy_from_spectra([1.], [1.], [101]), ValueError),
        ("nonpositive trace", lambda: production.log_trace_powers([-1.], [102]), RuntimeError),
    ):
        try:
            operation()
        except exception:
            outcomes.append({"input": label, "rejected_as": exception.__name__})
        else:
            raise AssertionError(label + " was silently accepted")
    return outcomes


def main():
    torch.set_num_threads(2)
    torch.manual_seed(20261009)
    rng = np.random.default_rng(20261009)
    # Avoid oversized BLAS pools for these deliberately tiny matrices.
    try:
        from threadpoolctl import threadpool_limits
        pool = threadpool_limits(limits=2)
    except ImportError:
        pool = None
    report = {"scope": "Production core contractions, index extraction and full-spectrum stability; not a high-D benchmark or an iterative-solver certificate.",
              "random_dense_reference_checks": check_random_paths(rng),
              "marked_extraction_checks": check_marked_extraction(),
              "long_circumference_checks": check_long_circumferences(rng),
              "rejected_invalid_inputs": check_rejected_inputs()}
    tiny_a, tiny_b = [.2 + rng.random((2, 2, 2, 2)) for _ in range(2)]
    torch.save({"a_raw": torch.tensor(tiny_a, dtype=torch.float64),
                "b_raw": torch.tensor(tiny_b, dtype=torch.float64), "J2": .26,
                "ansatz": "twoc3"}, HERE / "tiny_checkpoint_D2.pt")
    (HERE / "verify_renyi2_core.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if pool is not None:
        pool.restore_original_limits()


if __name__ == "__main__":
    main()
