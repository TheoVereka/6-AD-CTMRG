"""Tensor-independent reproduction of the first normalization failure."""
from pathlib import Path
import json

import torch


def example(gram_error):
    torch.set_num_threads(2)
    q = torch.zeros((4, 1), dtype=torch.float64)
    q[0, 0] = (1. + gram_error)**.5
    # A vector already almost in the old basis; its genuine new component is
    # tiny. This is the state reached near Krylov breakdown, without any PEPS.
    candidate = 1e-4 * q
    candidate[1, 0] = 1e-17
    before = torch.linalg.vector_norm(candidate).item()
    for _ in range(2):
        overlap = q.T @ candidate
        candidate.addmm_(q, overlap, beta=1, alpha=-1)
    new, r = torch.linalg.qr(candidate, mode="reduced")
    threshold = max(50 * torch.finfo(torch.float64).eps * before, 1e-14 * before)
    keep = (torch.abs(torch.diagonal(r)) > threshold)
    return dict(old_gram_error=(q.T @ q - torch.eye(1)).norm().item(),
                before_norm=before, after_norm=candidate.norm().item(),
                raw_old_basis_component=(q.T @ candidate).norm().item(),
                threshold=threshold, r_diagonal=r.diagonal().tolist(),
                keep=keep.tolist(),
                normalized_old_basis_overlap=(q.T @ new[:, keep]).norm().item())


def main():
    result={"exact_old_basis":example(0.), "slightly_inexact_old_basis":example(5e-9)}
    assert result["exact_old_basis"]["normalized_old_basis_overlap"] == 0.
    assert result["slightly_inexact_old_basis"]["keep"] == [True]
    assert result["slightly_inexact_old_basis"]["normalized_old_basis_overlap"] > 1e-4
    Path(__file__).with_suffix(".json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":main()
