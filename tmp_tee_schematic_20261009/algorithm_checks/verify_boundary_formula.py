"""Independent small-matrix checks of the supplied ABCD replica construction.

This validates boundary-MPO algebra, not the finite-cylinder CTMRG approximation.
Locally purified random MPOs give positive physical boundary operators.
"""
from pathlib import Path
import itertools
import json
import numpy as np

ROOT = Path(__file__).resolve().parent


def purified_edge(rng, d=2, virtual=2, purification=2):
    k = rng.normal(size=(virtual, virtual, d, purification)) + 1j * rng.normal(size=(virtual, virtual, d, purification))
    e = np.einsum("uvap,UVbp->abuUvV", k, k.conj()).reshape(d, d, virtual**2, virtual**2)
    return e / np.linalg.norm(e)


def physical_operator(tensors):
    d, _, chi, _ = tensors[0].shape
    words = list(itertools.product(range(d), repeat=len(tensors)))
    operator = np.empty((len(words), len(words)), complex)
    for i, ket in enumerate(words):
        for j, bra in enumerate(words):
            value = np.eye(chi, dtype=complex)
            for tensor, a, b in zip(tensors, ket, bra):
                value = value @ tensor[a, b]
            operator[i, j] = np.trace(value)
    return operator


def transfer_matrices(a, b, c, d):
    chi = a.shape[2]
    ab = np.einsum("abip,bajq->ijpq", a, b)
    dc = np.einsum("abpi,baqj->pqij", d, c)
    t1 = np.einsum("pqij,klpq->ijkl", dc, ab).reshape(chi**2, chi**2)
    upper = np.einsum("abip,bcjq,cdkr,dals->pqrsijkl", a, b, a, b, optimize=True).reshape(chi**4, chi**4)
    lower = np.einsum("abpi,bcqj,cdrk,dasl->ijklpqrs", d, c, d, c, optimize=True).reshape(chi**4, chi**4)
    return t1, lower @ upper


def eight_steps(v, a, b, c, d):
    w = np.einsum("abip,ijkl->abpjkl", a, v)
    w = np.einsum("bcjq,abpjkl->acpqkl", b, w)
    w = np.einsum("cdkr,acpqkl->adpqrl", a, w)
    out = np.einsum("dals,adpqrl->pqrs", b, w)
    w = np.einsum("abpi,pqrs->abiqrs", d, out)
    w = np.einsum("bcqj,abiqrs->acijrs", c, w)
    w = np.einsum("cdrk,acijrs->adijks", d, w)
    return np.einsum("dasl,adijks->ijkl", c, w)


def sector_ring(v, first, second, parity):
    out = np.zeros_like(v)
    for a in range(first.shape[0]):
        for c in range(a, first.shape[0]):
            contribution = np.einsum("bip,bjq,dkr,dls,ijkl->pqrs",
                first[a], second[:, c], first[c], second[:, a], v, optimize=True)
            out += contribution * (.5 if a == c else 1.)
    return out + parity * out.transpose(2, 3, 0, 1)


def relative_error(x, y):
    return float(np.linalg.norm(x - y) / max(np.linalg.norm(y), 1e-300))


def main():
    rng = np.random.default_rng(20261009)
    a, b, c, d = [purified_edge(rng) for _ in range(4)]
    chi = a.shape[2]
    t1, t2 = transfer_matrices(a, b, c, d)
    v = rng.normal(size=(chi,)*4) + 1j * rng.normal(size=(chi,)*4)
    expected = (t2 @ v.ravel()).reshape(v.shape)
    actual = eight_steps(v, a, b, c, d)
    report = {"D": a.shape[0], "chi": chi, "matvec_relative_error": relative_error(actual, expected)}
    report["replica_commutator_relative_error"] = relative_error(
        eight_steps(v.transpose(2, 3, 0, 1), a, b, c, d), actual.transpose(2, 3, 0, 1))
    report["sector_pruning_relative_errors"] = {}
    for parity in (1, -1):
        vec = v + parity * v.transpose(2, 3, 0, 1)
        upper = sector_ring(vec, a, b, parity)
        pruned = sector_ring(upper, d, c, parity)
        report["sector_pruning_relative_errors"][str(parity)] = relative_error(pruned, eight_steps(vec, a, b, c, d))
    eig1, eig2 = np.linalg.eigvals(t1), np.linalg.eigvals(t2)
    report["circumferences"] = []
    for cells in (1, 2, 3):
        left = physical_operator([a, d] * cells)
        right = physical_operator([b, c] * cells)
        vals, vectors = np.linalg.eigh(left)
        assert vals.min() > -1e-12
        assert np.linalg.eigvalsh(right).min() > -1e-12
        root_left = (vectors * np.sqrt(np.maximum(vals, 0))) @ vectors.conj().T
        rho = root_left @ right @ root_left
        z = np.trace(rho)
        rho /= z
        physical_purity = np.trace(rho @ rho)
        z1 = np.trace(np.linalg.matrix_power(t1, cells))
        z2 = np.trace(np.linalg.matrix_power(t2, cells))
        trace_purity = z2 / z1**2
        spectrum_purity = np.sum(eig2**cells) / np.sum(eig1**cells)**2
        entry = dict(cut_bonds=2*cells,
            entropy_physical=float(-np.log(physical_purity.real)),
            entropy_transfer=float(-np.log(trace_purity.real)),
            trace_purity_relative_error=relative_error(trace_purity, physical_purity),
            spectrum_purity_relative_error=relative_error(spectrum_purity, physical_purity))
        assert entry["trace_purity_relative_error"] < 1e-10
        assert entry["spectrum_purity_relative_error"] < 1e-10
        report["circumferences"].append(entry)
    assert report["matvec_relative_error"] < 1e-11
    assert report["replica_commutator_relative_error"] < 1e-11
    assert max(report["sector_pruning_relative_errors"].values()) < 1e-11
    report["scope"] = "Exact boundary-MPO algebra; does not certify CTMRG finite-cylinder fixed points, physical topology, or solver tail coverage."
    (ROOT / "boundary_formula_checks.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
