"""Tiny, geometry-matched armchair-cylinder check; not a production solver.

The physical cut is between columns 5 and 6 of the requested cfabed/bedcfa
honeycomb.  It crosses gamma, beta, gamma, beta, ... legs.  The exact channel
uses 6 columns and a periodic vertical circumference of 2*L sites.  It does
NOT compare this geometry with the archived, differently oriented cylinder.
"""
from __future__ import annotations

import json
import argparse
import sys
from itertools import product
from pathlib import Path

import numpy as np
import opt_einsum as oe
import torch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src_code" / "scripts"))
import correlation_length as cl


def column_kraus(sites, column, cells):
    """Explicit one-column Kraus operators, exact only for tiny cylinders."""
    rows = 2 * cells
    side = sites["a"].shape[0]
    pairs = [(r, (r + 1) % rows) for r in range(column % 2, rows, 2)]
    arguments = []
    for first, second in pairs:
        tensors = []
        for row in (first, second):
            labels = "cfabed" if row % 2 == 0 else "bedcfa"
            tensor = sites[labels[column % 6]]
            # Stored axes alpha,beta,gamma,spin; alpha is the blue vertical
            # link.  On an even row+column, beta is the right zigzag link.
            if (row + column) % 2 == 0:
                tensor = tensor.transpose(0, 2, 1, 3)
            tensors.append(tensor)
        pair = np.einsum("alrp,amnq->lmrnpq", *tensors)
        arguments += [pair, [first, second, rows + first, rows + second,
                             2 * rows + first, 2 * rows + second]]
    arguments += [list(range(3 * rows))]
    explicit = oe.contract(*arguments).reshape(side**rows, side**rows, 2**rows)
    return explicit.transpose(2, 1, 0)  # spin, output, input


def channel(sites, cells):
    count = sites["a"].shape[0] ** (2 * cells)
    period = np.eye(count * count)
    for column in range(6, 12):
        kraus = column_kraus(sites, column, cells)
        local = np.einsum("soi,spj->opij", kraus, kraus.conj()).reshape(count**2, count**2)
        local /= np.linalg.norm(local)
        period = local @ period
        period /= np.linalg.norm(period)
    return period


def fixed_point(transfer):
    count = round(np.sqrt(len(transfer)))
    x = np.eye(count) / count
    for step in range(2000):
        output = (transfer @ x.reshape(-1)).reshape(count, count)
        output /= np.trace(output)
        difference = np.linalg.norm(output - x)
        x = output
        if difference < 1e-12:
            break
    return x, step + 1, float(difference)


def mpo(edge1, edge2, cells, reverse_chi=False):
    side = round(np.sqrt(edge1.shape[-1]))
    width = edge1.shape[0]
    # Pair nearest neighbors by first-first and then second-second chi legs.
    first = edge1.reshape(width, width, side, side).transpose(2, 3, 1, 0)
    second = edge2.reshape(width, width, side, side).transpose(2, 3, 0, 1)
    if reverse_chi:
        first = first.swapaxes(2, 3)
        second = second.swapaxes(2, 3)
    configurations = list(product(range(side), repeat=2 * cells))
    out = np.empty((len(configurations), len(configurations)), dtype=edge1.dtype)
    for i, ket in enumerate(configurations):
        for j, bra in enumerate(configurations):
            current = np.eye(width)
            for position in range(2 * cells):
                tensor = first if position % 2 == 0 else second
                current = current @ tensor[ket[position], bra[position]]
            out[i, j] = np.trace(current)
    return out


def entropy(left, right):
    combined = left @ right
    z1 = np.trace(combined)
    z2 = np.trace(combined @ combined)
    ratio = z2 / z1**2
    return {"Z1": float(np.real(z1)), "purity": float(np.real(ratio)),
            "S2": float(-np.log(np.real(ratio))) if np.real(ratio) > 0 else None,
            "imaginary_purity": float(abs(np.imag(ratio)))}


def residual(transfer, x):
    x = x / np.trace(x)
    out = (transfer @ x.reshape(-1)).reshape(x.shape)
    multiplier = np.trace(out)
    return float(np.linalg.norm(out - multiplier * x) / max(abs(multiplier), 1e-30))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("positive", "signed"), default="positive")
    parser.add_argument("--max-steps", type=int, default=120)
    options = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(20261009)
    core = cl._core
    core.set_dtype(True, use_real=True)
    core.set_device("cpu")
    core._USE_FULL_SVD = True
    core.set_ctm_conv_mode("SVdifference", e_threshold=1e-10)
    rng = np.random.default_rng(20261009)
    # Positive but leg-asymmetric tensors make the convergence check cheap
    # without imposing a=b, reflection symmetry, or cyclic invariance.
    a, b = [(0.2 + rng.random((2, 2, 2, 2)) if options.family == "positive"
             else rng.normal(size=(2, 2, 2, 2))) for _ in range(2)]
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    derived = core.twoc3_abcdef_from_ab(torch.tensor(a), torch.tensor(b))
    sites = {name: tensor.numpy() for name, tensor in zip("abcdef", derived)}
    exact = {}
    for cells in (1, 2):
        transfer = channel(sites, cells)
        left, steps_l, err_l = fixed_point(transfer)
        right, steps_r, err_r = fixed_point(transfer.conj().T)
        exact[cells] = (transfer, left, right)
    product_a = np.array([0.6, 0.8]).reshape(1, 1, 1, 2)
    product_b = np.array([0.8, -0.6]).reshape(1, 1, 1, 2)
    product_sites = {name: value.numpy() for name, value in zip(
        "abcdef", core.twoc3_abcdef_from_ab(torch.tensor(product_a), torch.tensor(product_b)))}
    product_transfer = channel(product_sites, 2)
    product_left, _, _ = fixed_point(product_transfer)
    product_right, _, _ = fixed_point(product_transfer.conj().T)
    baseline = entropy(product_left, product_right)["S2"]
    assert abs(baseline) < 1e-14
    report = {"purpose": __doc__, "seed": 20261009, "baseline_D1_S2": baseline,
              "ctm_max_steps": options.max_steps, "ctm_spectrum_tolerance": 1e-10,
              "tensor_family": options.family + ", leg-asymmetric generic two-C3, D=2", "checks": []}
    for chi in (2, 4, 8):
        environments = []
        for raw_a, raw_b in ((a, b), (b, a)):
            _, double_layers = cl._build_ctm_layers(torch.tensor(raw_a), torch.tensor(raw_b))
            result = core.CTMRG_from_init_to_stop(*double_layers, chi, 4, options.max_steps, 1e-10, True)
            environments.append(result)
        for phase, first_index, second_index in (("env1_T1F_T2A", 1, 2),
                                                 ("env2_T1D_T2C", 4, 5),
                                                 ("env3_T1B_T2E", 7, 8)):
            for cells in (1, 2):
                transfer, exact_left, exact_right = exact[cells]
                left = mpo(environments[0][first_index].numpy(), environments[0][second_index].numpy(), cells)
                right = mpo(environments[1][first_index].numpy(), environments[1][second_index].numpy(), cells, reverse_chi=True).T
                wrong_right = mpo(environments[1][first_index].numpy(), environments[1][second_index].numpy(), cells).T
                observed = entropy(left, right)
                reference = entropy(exact_left, exact_right)
                wrong_direction = entropy(left, wrong_right)
                report["checks"].append({
                    "chi": chi, "phase": phase, "L_cells": cells, "cut_bonds_2L": 2 * cells,
                    "ctm_steps_ab": int(environments[0][-1]), "ctm_steps_ba": int(environments[1][-1]),
                    "ctm_stopped_before_limit_ab": int(environments[0][-1]) < options.max_steps,
                    "ctm_stopped_before_limit_ba": int(environments[1][-1]) < options.max_steps,
                    "S2_from_periodized_edges": observed["S2"], "S2_exact_matched_cylinder": reference["S2"],
                    "S2_difference": observed["S2"] - reference["S2"],
                    "wrong_right_chi_direction_S2": wrong_direction["S2"],
                    "wrong_right_chi_direction_difference": wrong_direction["S2"] - reference["S2"],
                    "left_cylinder_eigenvector_residual": residual(transfer, left),
                    "right_cylinder_eigenvector_residual": residual(transfer.conj().T, right),
                    "left_hermiticity_relative": float(np.linalg.norm(left - left.conj().T) / np.linalg.norm(left)),
                    "right_hermiticity_relative": float(np.linalg.norm(right - right.conj().T) / np.linalg.norm(right)),
                    "exact_left_eigenvector_residual": residual(transfer, exact_left),
                    "exact_right_eigenvector_residual": residual(transfer.conj().T, exact_right),
                })
    destination = Path(__file__).with_name("code_boundary_check" + ("_signed" if options.family == "signed" else "") + ".json")
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
