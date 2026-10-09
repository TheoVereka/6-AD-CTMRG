"""Real block Krylov / thick-restart Rayleigh--Ritz for a real linear map.

Rank-revealing orthogonalization rejects numerical breakdown directions
before they can be normalized. Sector projection is repeated after basis
updates; restarts check and repair the basis Gram matrix. Residuals are
scaled to the leading spectrum, so negligible roots are not required to
have a meaningless relative-to-themselves accuracy.

Q and A Q are preallocated torch.float64 bases. Small eigensolves and an
ordered real Schur decomposition run on CPU. Restarts transform basis rows
in tiles, so no second full basis is allocated. No spectral-tail guarantee
is implied by ``converged``: it means the returned Ritz pairs passed the
explicit full-space residual criterion.

This file has no import-time work and no dependence on the surrounding repo.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Callable

import numpy as np
import scipy.linalg as sla
import torch


@dataclass(frozen=True)
class BlockResult:
    eigenvalues: np.ndarray
    relative_residuals: np.ndarray
    converged: bool
    matvec_count: int
    restarts: int
    subspace_dim: int
    reason: str
    orthogonality_error: float
    requested_k: int
    rank_deflations: int = 0
    random_completions: int = 0
    gram_repairs: int = 0

    def __getitem__(self, name):
        return getattr(self, name)


def _ordered_indices(values):
    # Real matrices produce complex conjugate pairs. Stable ordering is useful
    # for diagnostics, but boundary selection below keeps complete |lambda|
    # clusters and therefore never intentionally drops one conjugate partner.
    return np.argsort(-np.abs(values), kind="stable")


def _complete_cluster_count(values, count, *, rtol):
    if count >= len(values):
        return len(values)
    boundary = abs(values[count - 1])
    if boundary == 0:
        # Arbitrarily many zero roots are irrelevant to every positive power.
        return count
    # A tolerance referenced to lambda_max groups *all* tiny roots into one
    # fictitious cluster and can demand the entire numerical null space.
    scale = float(boundary)
    while count < len(values) and abs(abs(values[count]) - boundary) <= rtol * scale:
        count += 1
    return count


def solve_block(
    operator_matvec: Callable[[torch.Tensor], torch.Tensor],
    n: int,
    k: int,
    block_size: int = 4,
    subspace: int = 64,
    tol: float = 1e-10,
    max_matvec: int = 2000,
    device: str | torch.device = "cpu",
    seed: int = 20261009,
    projector: Callable[[torch.Tensor], torch.Tensor] | None = None,
    progress_every: int = 0,
) -> BlockResult:
    """Find leading-magnitude Ritz pairs using an actual block Krylov basis.

    ``operator_matvec`` accepts and returns a 1D float64 torch tensor on
    ``device``. ``projector``, if supplied, is an orthogonal real projector;
    every randomized start and every randomized rank-completion vector is
    projected before orthogonalization. The supplied map should preserve its
    image. Returned eigenvalues are complex128 numpy scalars. The returned
    count can exceed k to preserve a conjugate pair or detected boundary
    cluster. Missing multiplicities / omitted tails remain an independent
    diagnostic; passing residuals does not certify their absence.
    """
    n, k = int(n), int(k)
    block_size, subspace, max_matvec = int(block_size), int(subspace), int(max_matvec)
    if n < 1 or k < 1 or k > n or block_size < 1:
        raise ValueError("Require n>=k>=1 and block_size>=1.")
    if tol <= 0 or max_matvec < 1:
        raise ValueError("tol and max_matvec must be positive.")
    capacity = min(n, subspace)
    if capacity < k or (capacity < n and capacity < k + 2):
        raise ValueError("subspace must leave room beyond k (unless n is small).")
    block_size = min(block_size, capacity)
    # Resolve 'cuda' to the indexed device used by tensors (e.g. cuda:0).
    dev = torch.empty(0, device=torch.device(device)).device
    generator = torch.Generator(device=dev)
    generator.manual_seed(int(seed))
    q_basis = torch.empty((n, capacity), dtype=torch.float64, device=dev)
    aq_basis = torch.empty_like(q_basis)
    projected = torch.zeros((capacity, capacity), dtype=torch.float64, device=dev)
    calls = 0
    started = time.perf_counter()
    restarts = 0
    m = 0
    eps = torch.finfo(torch.float64).eps
    cluster_rtol = max(100 * eps, min(1e-8, 10 * tol))
    last_values = np.empty(0, dtype=np.complex128)
    last_residuals = np.empty(0, dtype=float)
    rank_deflations = 0
    random_completions = 0
    gram_repairs = 0

    def apply(vector):
        nonlocal calls
        if calls >= max_matvec:
            return None
        answer = operator_matvec(vector)
        calls += 1
        if progress_every and calls % progress_every == 0:
            print(f"  block Krylov: matvec={calls}, basis={m}/{capacity}, restarts={restarts}, "
                  f"elapsed={time.perf_counter()-started:.1f}s", flush=True)
        if not isinstance(answer, torch.Tensor):
            answer = torch.as_tensor(answer, device=dev)
        if answer.shape != (n,) or answer.device != dev:
            raise ValueError("operator_matvec must return an n-vector on the same device.")
        if answer.is_complex() or answer.dtype != torch.float64:
            raise TypeError("operator_matvec must preserve real float64 dtype.")
        if not torch.isfinite(answer).all():
            raise FloatingPointError("Nonfinite operator output.")
        return answer

    def project_block(candidate):
        if projector is not None:
            for j in range(candidate.shape[1]):
                value = projector(candidate[:, j])
                if value.shape != (n,) or value.dtype != torch.float64 or value.device != dev:
                    raise ValueError("projector must preserve vector shape, dtype and device.")
                candidate[:, j].copy_(value)
        return candidate

    def random_block(width):
        return project_block(torch.randn((n, width), dtype=torch.float64,
                                        device=dev, generator=generator))

    def subtract_old(candidate):
        for _ in range(2):
            if m:
                overlap = q_basis[:, :m].T @ candidate
                candidate.addmm_(q_basis[:, :m], overlap, beta=1, alpha=-1)
        return candidate

    def rank_reveal(candidate, reference_norm, relative_cutoff):
        # QR alone is not rank revealing: its arbitrary completion columns
        # must never be mistaken for independent Krylov directions. The SVD
        # is only on the small block R, not on the large operator or basis.
        if candidate.shape[1] == 0 or reference_norm == 0:
            return candidate[:, :0]
        remaining = float(torch.linalg.vector_norm(candidate).item())
        if remaining <= relative_cutoff * reference_norm:
            return candidate[:, :0]
        block_q, block_r = torch.linalg.qr(candidate, mode="reduced")
        left, singular, _ = torch.linalg.svd(block_r, full_matrices=False)
        keep = singular > relative_cutoff * reference_norm
        return block_q @ left[:, keep]

    def orthogonalize(candidate):
        nonlocal rank_deflations
        original_width = candidate.shape[1]
        candidate = project_block(candidate)
        before = float(torch.linalg.vector_norm(candidate).item())
        if before == 0:
            rank_deflations += original_width
            return candidate[:, :0]
        # Relative to the incoming block: this also works for an operator
        # whose entire scale is 1e-100. Near-zero *new* information is deflated.
        candidate = rank_reveal(subtract_old(candidate), before, 4096 * eps)
        if candidate.shape[1]:
            # Unit scaling exposes any amplified roundoff in the first pass.
            # Project again because QR completion can leak out of an exact
            # symmetry sector at numerical rank loss.
            candidate = project_block(candidate)
            candidate = rank_reveal(subtract_old(candidate),
                                    math.sqrt(candidate.shape[1]), 4096 * eps)
        rank_deflations += original_width - candidate.shape[1]
        return candidate

    def append(candidate):
        nonlocal m
        width = min(candidate.shape[1], capacity - m, max_matvec - calls)
        if width <= 0:
            return 0
        old = m
        q_basis[:, old:old + width].copy_(candidate[:, :width])
        for j in range(old, old + width):
            value = apply(q_basis[:, j])
            if value is None:
                break
            aq_basis[:, j].copy_(value)
        m = old + width
        projected[:m, old:m] = q_basis[:, :m].T @ aq_basis[:, old:m]
        if old:
            projected[old:m, :old] = q_basis[:, old:m].T @ aq_basis[:, :old]
        return width

    def ritz():
        h = projected[:m, :m].detach().cpu().numpy().copy()
        values, coefficients = sla.eig(h, left=False, right=True,
                                       check_finite=False)
        order = _ordered_indices(values)
        return h, values[order], coefficients[:, order]

    def residuals(values, coefficients, count):
        result = np.empty(count, dtype=float)
        leading_scale = max(float(np.max(np.abs(values))), np.finfo(float).tiny)
        for j in range(count):
            y = coefficients[:, j]
            real_coeff = torch.as_tensor(np.array(y.real, dtype=np.float64, copy=True, order="C"), device=dev)
            imag_coeff = torch.as_tensor(np.array(y.imag, dtype=np.float64, copy=True, order="C"), device=dev)
            xr = q_basis[:, :m] @ real_coeff
            xi = q_basis[:, :m] @ imag_coeff
            ar = aq_basis[:, :m] @ real_coeff
            ai = aq_basis[:, :m] @ imag_coeff
            x_norm = float(torch.sqrt(torch.dot(xr, xr) + torch.dot(xi, xi)).item())
            a_norm = float(torch.sqrt(torch.dot(ar, ar) + torch.dot(ai, ai)).item())
            real_lambda, imag_lambda = float(values[j].real), float(values[j].imag)
            ar.add_(xr, alpha=-real_lambda).add_(xi, alpha=imag_lambda)
            ai.add_(xi, alpha=-real_lambda).add_(xr, alpha=-imag_lambda)
            residual_norm = float(torch.sqrt(torch.dot(ar, ar) + torch.dot(ai, ai)).item())
            denominator = max(a_norm + abs(values[j]) * x_norm,
                              2 * leading_scale * x_norm)
            result[j] = residual_norm / max(denominator, np.finfo(float).tiny)
        return result

    def finish(success, reason):
        if m:
            gram = q_basis[:, :m].T @ q_basis[:, :m]
            gram.diagonal().sub_(1)
            ortho = float(torch.linalg.matrix_norm(gram).item())
        else:
            ortho = float("inf")
        # A small projected residual has no meaning in a corrupt basis.
        if success and (not math.isfinite(ortho) or ortho > 1e-8):
            success, reason = False, "basis_orthogonality_not_recovered"
        return BlockResult(last_values.copy(), last_residuals.copy(), bool(success),
                           calls, restarts, m, reason, ortho, k,
                           rank_deflations, random_completions, gram_repairs)

    # Restart uses no n*keep temporary: each row tile is independent under
    # right multiplication. At most 128 MiB total transform scratch is live.
    def transform_basis(keep_matrix, retained):
        transform = torch.as_tensor(np.ascontiguousarray(keep_matrix), device=dev)
        rows = max(1, min(n, (64 * 1024**2) // (8 * (m + retained))))
        for basis in (q_basis, aq_basis):
            for start in range(0, n, rows):
                stop = min(n, start + rows)
                tile = basis[start:stop, :m].clone()
                changed = tile @ transform
                basis[start:stop, :retained].copy_(changed)

    def repair_gram():
        nonlocal m, gram_repairs
        gram = q_basis[:, :m].T @ q_basis[:, :m]
        error = float(torch.linalg.matrix_norm(gram - torch.eye(m, dtype=torch.float64, device=dev)).item())
        if error > 100 * eps * max(1, m):
            # Synchronizing Q and AQ under the same right transformation
            # preserves the exact cached relationship AQ = operator(Q).
            small = gram.detach().cpu().numpy()
            eigenvalues, eigenvectors = sla.eigh((small + small.T) * .5,
                                                 check_finite=False)
            keep = eigenvalues > 4096 * eps * max(float(eigenvalues[-1]), 1.)
            transform = eigenvectors[:, keep] / np.sqrt(eigenvalues[keep])[None, :]
            transform_basis(transform, int(keep.sum()))
            m = int(keep.sum())
            gram_repairs += 1
        projected[:m, :m] = q_basis[:, :m].T @ aq_basis[:, :m]

    with torch.no_grad():
        first = orthogonalize(random_block(block_size))
        added = append(first)
        del first
        if not added:
            return finish(False, "projector_has_no_nonzero_start")
        frontier_start, frontier_stop = 0, m
        while True:
            if m >= k:
                h, values, vectors = ritz()
                count = _complete_cluster_count(values, k, rtol=cluster_rtol)
                last_values = values[:count]
                last_residuals = residuals(values, vectors, count)
                enough_space = (m >= min(capacity, max(k + block_size, 2 * k)))
                if enough_space and np.all(last_residuals <= tol):
                    return finish(True, "ritz_residuals_converged_tail_not_certified")
            if calls >= max_matvec:
                return finish(False, "max_matvec_reached")
            if m == n:
                # Exact full-space reduction; reaching it is meaningful even
                # for a highly nonnormal matrix, although rounding persists.
                return finish(bool(len(last_residuals) >= k and np.all(last_residuals <= tol)),
                              "full_space_reduction")
            if m == capacity:
                h, values, _ = ritz()
                target = min(capacity - block_size, max(k + block_size, 2 * k))
                retained = _complete_cluster_count(values, target, rtol=cluster_rtol)
                if retained >= capacity:
                    # Try an earlier complete boundary; never split a detected
                    # cluster just to make the requested storage budget fit.
                    retained = target
                    while retained >= k and abs(abs(values[retained - 1]) - abs(values[retained])) <= cluster_rtol * max(abs(values[retained - 1]), 1e-300):
                        retained -= 1
                    if retained < k:
                        return finish(False, "subspace_too_small_for_detected_boundary_cluster")
                threshold = 0.5 * (abs(values[retained - 1]) + abs(values[retained]))
                _, schur_vectors, actual = sla.schur(
                    h, output="real", sort=lambda real, imag: math.hypot(real, imag) > threshold,
                    check_finite=False)
                if actual < k or actual >= capacity:
                    return finish(False, "ordered_schur_retention_failed")
                retained = int(actual)
                u = schur_vectors[:, :retained]
                transform_basis(u, retained)
                m = retained
                repair_gram()
                restarts += 1
                frontier_start, frontier_stop = max(0, m - block_size), m
            width = min(block_size, capacity - m, max_matvec - calls)
            candidate = aq_basis[:, frontier_start:frontier_stop].clone()
            candidate = orthogonalize(candidate)
            if candidate.shape[1] < width:
                # The original block has hit an invariant subspace or dropped
                # rank. Add independent projected directions explicitly.
                if candidate.shape[1]:
                    append(candidate[:, :width])
                for _ in range(3):
                    available = min(block_size, capacity - m, max_matvec - calls)
                    if available <= 0:
                        break
                    supplement = orthogonalize(random_block(available))
                    if supplement.shape[1]:
                        old = m
                        append(supplement)
                        random_completions += supplement.shape[1]
                        frontier_start, frontier_stop = old, m
                        break
                else:
                    if m >= k:
                        _, values, vectors = ritz()
                        count = _complete_cluster_count(values, k, rtol=cluster_rtol)
                        last_values = values[:count]
                        last_residuals = residuals(values, vectors, count)
                        return finish(bool(np.all(last_residuals <= tol)), "projected_space_exhausted")
                    return finish(False, "projected_space_dimension_below_k")
            else:
                old = m
                append(candidate[:, :width])
                frontier_start, frontier_stop = old, m


def real_log_trace_power(eigenvalues, power, *, imaginary_tolerance=1e-10):
    """Stable log(trace(T**power)); retain phases, reject nonphysical sign.

    This sums the provided spectrum, not an unknown omitted tail. The second
    return value is the sum-cancellation factor, which amplifies spectral
    errors. Negative/complex traces are never silently replaced by abs().
    """
    values = np.asarray(eigenvalues, dtype=np.complex128)
    if int(power) != power or power < 1 or not len(values):
        raise ValueError("A positive integer power and a nonempty spectrum are required.")
    if not np.isfinite(values).all():
        raise FloatingPointError("Nonfinite eigenvalue.")
    radius = float(np.max(np.abs(values)))
    if not radius:
        raise FloatingPointError("Trace vanishes for the supplied zero spectrum.")
    terms = np.power(values / radius, int(power))
    total = complex(math.fsum(terms.real), math.fsum(terms.imag))
    absolute_sum = math.fsum(np.abs(terms))
    if total.real <= 0 or abs(total.imag) > imaginary_tolerance * abs(total.real):
        raise FloatingPointError("Trace power is not positive real within tolerance.")
    return int(power) * math.log(radius) + math.log(total.real), absolute_sum / abs(total)


def entropy_from_spectra(t1_eigenvalues, t2_eigenvalues, powers):
    """Return S2(2L) estimates and cancellation diagnostics for integer L."""
    output = []
    for power in powers:
        log1, cancellation1 = real_log_trace_power(t1_eigenvalues, power)
        log2, cancellation2 = real_log_trace_power(t2_eigenvalues, power)
        output.append({"L_cells": int(power), "cut_bonds_2L": 2 * int(power),
                       "S2": 2 * log1 - log2,
                       "trace1_cancellation": cancellation1,
                       "trace2_cancellation": cancellation2})
    return output


def tail_radius_ratio_limit(dimension, retained, power, relative_trace_budget,
                            scaled_kept_trace=1.0):
    """Largest certified |omitted lambda|/rho under the dimension-only bound.

    The caller must actually establish that all omitted eigenvalues obey the
    resulting radius limit. A first omitted Ritz value alone is no proof.
    """
    missing = int(dimension) - int(retained)
    if missing <= 0:
        return float("inf")
    if power <= 0 or relative_trace_budget <= 0 or scaled_kept_trace <= 0:
        raise ValueError("Positive power and budgets are required.")
    return math.exp(math.log(relative_trace_budget * scaled_kept_trace / missing) / power)
