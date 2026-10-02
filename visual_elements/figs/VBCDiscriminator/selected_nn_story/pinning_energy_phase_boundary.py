#!/usr/bin/env python3
"""Signed-pinning-field energy surface and D->infinity crossing line.

Plaquette pinning is assigned signed h>0 and dimer-plaquette pinning signed
h<0.  Nonzero rank-split fields are a different direction in order-parameter
space and are deliberately excluded.  Set A contains (J2,D) pairs with at
least two distinct positive and two distinct negative fields.  Set B contains
every discovered scalar-field observation for the pairs in A, including h=0.

Figure 03 shows all physically screened B points.  For figure 04, reruns at a
fixed (J2,D,h) are reduced before fitting, each finite-D energy branch uses an
ordinary quadratic E(h), and the finite-D roots enter an equal-D constant
extrapolation.  Each of the three NN correlations is separately extrapolated
with the fixed-a_g gapped form on an adaptive contiguous high-D suffix; Delta
and q are calculated only from the three extrapolated correlations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patheffects as path_effects
import numpy as np
from scipy.interpolate import griddata


HERE = Path(__file__).resolve().parent
VBC_DIR = HERE.parent
REPO = HERE.parents[3]
DATA = REPO.parent / "data"

import sys

sys.path.insert(0, str(VBC_DIR))
from analyze_branch_runs import find_lookahead, read_scalar_hyperparams  # noqa: E402
from analyze_existing_twoc3 import parse_observation  # noqa: E402


DEFAULT_OUTPUT = HERE / "plots" / "pinning_energy_phase_boundary"
ENERGY_FITS = DATA / "processed" / "publicationPlots" / "figure_24_fits.csv"
OBS_RE = re.compile(
    r"^D_(\d+)_chi_(\d+)_energy_magnetization_correlation\.txt$"
)
REPLICA_RE = re.compile(r"replica_(\d+)")
SCALAR_BRANCHES = ("dimer-plaquette", "plaquette")
ROOTS = (
    ("synced_large_h", DATA / "distinVBCs"),
    ("synced_small_h", DATA / "distinVBCsSmallH"),
    ("last_izar_pins", DATA / "distinVBCsJ2Continuation" / "LastIzar"
     / "Results_LastIzar"),
    ("d7_repair", DATA / "distinVBCsJ2Continuation" / "D7DimerJ2_0p26"
     / "Results_D7DimerJ2_0p26"),
    ("izar_j2_sequences", DATA / "distinVBCsJ2Continuation"
     / "Results_Izar_J2_sequences"),
    ("external_sep27", DATA / "external" / "Working_AD_Honeycomb_Sep27"),
    ("bundle_replica12", REPO / "models" / "VBCPinningClusterBundle"
     / "Results_VBC_branches"),
    ("bundle_kuma_three", REPO / "models" / "VBCPinningClusterBundle"
     / "Results_VBC_three"),
)


@dataclass(frozen=True)
class FieldPoint:
    point_id: str
    source_root: str
    path: str
    J2: float
    D: int
    chi: int
    branch: str
    field_magnitude: float
    signed_h: float
    orientation: int
    replica: int
    energy: float
    chi_energy_shift: float
    strongest: float
    middle: float
    weakest: float
    delta: float
    middle_fraction: float
    texture: str
    content_sha256: str


@dataclass
class RepresentativeDecision:
    point: FieldPoint
    selected: bool
    reason: str


@dataclass
class RobustFit:
    degree: int
    beta: np.ndarray
    covariance: np.ndarray
    residuals: np.ndarray
    inliers: np.ndarray
    rms: float
    scale: float
    n: int
    n_inlier: int


@dataclass
class CrossingResult:
    J2: float
    accepted: bool
    reason: str
    h_window: float
    h_c: float
    h_c_error: float
    h_c_q16: float
    h_c_q84: float
    slope_difference: float
    plaquette_degree: int
    dimer_degree: int
    plaquette_n: int
    dimer_n: int
    plaquette_inliers: int
    dimer_inliers: int
    plaquette_rms: float
    dimer_rms: float
    Ds: str
    per_D_crossing_count: int
    per_D_crossing_spread: float


def close(left: float, right: float, tolerance: float = 1.0e-10) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=tolerance)


def field_sign(branch: str, magnitude: float) -> float:
    if abs(magnitude) <= 1.0e-14:
        return 0.0
    return magnitude if branch == "plaquette" else -magnitude


def root_source_label(label: str, path: Path) -> str:
    for part in path.parts:
        if part.startswith("Results_"):
            return f"{label}:{part}"
    return label


def discover_points() -> tuple[list[FieldPoint], list[str], int, int]:
    points: list[FieldPoint] = []
    failures: list[str] = []
    seen: set[tuple[str, str, float]] = set()
    duplicate_count = 0
    rank_split_count = 0
    for label, root in ROOTS:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("D_*_chi_*_energy_magnetization_correlation.txt")):
            if OBS_RE.fullmatch(path.name) is None:
                continue
            hyperparams_path = path.parent / "hyperparams.yaml"
            if not hyperparams_path.is_file():
                failures.append(f"{path}: missing hyperparams.yaml")
                continue
            try:
                params = read_scalar_hyperparams(hyperparams_path)
                branch = params.get("vbc_branch", "none")
                if branch == "rank-split":
                    rank_split_count += 1
                    continue
                if branch not in SCALAR_BRANCHES:
                    continue
                magnitude = float(params["vbc_field"])
                if magnitude < -1.0e-14:
                    raise ValueError("negative stored vbc_field")
                observation = parse_observation(path)
                # Sequence paths contain both the seed J2 and the current J2;
                # the generic observation parser takes the first path token.
                # The colocated immutable run metadata are authoritative.
                current_J2 = float(params["J2"])
                if current_J2 < 0.26 - 1.0e-12:
                    continue
                lookahead_path = find_lookahead(path)
                chi_shift = math.nan
                if lookahead_path is not None:
                    lookahead = parse_observation(lookahead_path)
                    chi_shift = abs(
                        lookahead.energy_per_site - observation.energy_per_site
                    )
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                dedup = (digest, branch, round(magnitude, 12))
                if dedup in seen:
                    duplicate_count += 1
                    continue
                seen.add(dedup)
                replica_match = REPLICA_RE.search(str(path))
                replica = int(replica_match.group(1)) if replica_match else 0
                orientation = int(params.get("vbc_orientation", -1))
                point_hash = hashlib.sha1(
                    f"{path.resolve()}|{branch}|{magnitude}".encode("utf-8")
                ).hexdigest()[:14]
                points.append(FieldPoint(
                    point_id=point_hash,
                    source_root=root_source_label(label, path),
                    path=str(path.resolve()), J2=current_J2,
                    D=observation.D, chi=observation.chi, branch=branch,
                    field_magnitude=magnitude,
                    signed_h=field_sign(branch, magnitude),
                    orientation=orientation, replica=replica,
                    energy=observation.energy_per_site,
                    chi_energy_shift=chi_shift, delta=observation.delta,
                    strongest=observation.rank1,
                    middle=observation.rank2,
                    weakest=observation.rank3,
                    middle_fraction=observation.middle_fraction,
                    texture=observation.texture, content_sha256=digest,
                ))
            except (KeyError, OSError, TypeError, ValueError) as error:
                failures.append(f"{path}: {error}")
    return sorted(points, key=lambda row: (
        row.J2, row.D, row.signed_h, row.branch, -row.chi, row.path,
    )), failures, duplicate_count, rank_split_count


def construct_sets(points: list[FieldPoint]) -> tuple[list[dict], list[FieldPoint]]:
    keys = sorted({(row.J2, row.D) for row in points})
    set_a: list[dict] = []
    accepted_keys: set[tuple[float, int]] = set()
    for J2, D in keys:
        subset = [row for row in points if close(row.J2, J2) and row.D == D]
        negative = sorted({round(row.signed_h, 12) for row in subset
                           if row.signed_h < -1.0e-12})
        positive = sorted({round(row.signed_h, 12) for row in subset
                           if row.signed_h > 1.0e-12})
        zeros = sum(abs(row.signed_h) <= 1.0e-12 for row in subset)
        if len(negative) >= 2 and len(positive) >= 2:
            accepted_keys.add((J2, D))
            set_a.append({
                "J2": J2, "D": D,
                "negative_fields": " ".join(f"{value:g}" for value in negative),
                "positive_fields": " ".join(f"{value:g}" for value in positive),
                "n_negative_fields": len(negative),
                "n_positive_fields": len(positive), "n_h0_observations": zeros,
                "n_all_observations": len(subset),
            })
    set_b = [row for row in points if (row.J2, row.D) in accepted_keys]
    return set_a, set_b


def texture_consistent(row: FieldPoint) -> bool:
    if row.delta <= 0.012 or not math.isfinite(row.middle_fraction):
        return True
    if row.branch == "plaquette":
        return row.middle_fraction <= 0.62
    return row.middle_fraction >= 0.38


def load_original_energies() -> dict[tuple[float, int], float]:
    energies: dict[tuple[float, int], float] = {}
    root = DATA / "0713summary"
    for path in root.glob(
        "J2_*/2tensor_twoC3/D_*/energy_magnetization_correlation.txt"
    ):
        try:
            row = parse_observation(path)
        except (OSError, TypeError, ValueError):
            continue
        energies[(row.J2, row.D)] = row.energy_per_site
    return energies


def energy_plausible(
    row: FieldPoint, references: dict[tuple[float, int], float],
) -> bool:
    reference = references.get((row.J2, row.D))
    if reference is None:
        return True
    # A pinning term can lower the finite-h Hamiltonian energy linearly in h.
    # This deliberately generous envelope catches failed optimizations without
    # prejudging the physical susceptibility or the branch slope.
    tolerance = 0.003 + 0.90 * abs(row.signed_h)
    return abs(row.energy - reference) <= tolerance


def screen_normal_energy_curves(
    rows: list[FieldPoint], references: dict[tuple[float, int], float],
) -> tuple[list[FieldPoint], list[dict]]:
    """Apply exactly the three permissive screens defining figure 03.

    There is deliberately no chi, lookahead, replica, or variational-energy
    ranking.  After the broad original-2C3 energy envelope and texture check,
    a robust low-order E(h) curve is constructed independently for every
    (J2,D,branch).  Only unmistakable curve outliers are removed.  The 1e-3
    residual floor makes this a visual/physical zigzag screen rather than a
    precision-data selection.
    """
    audit: list[dict] = []
    candidates: list[FieldPoint] = []
    for row in rows:
        if not energy_plausible(row, references):
            audit.append({
                **asdict(row), "included_in_03": False,
                "curve_residual": math.nan, "curve_threshold": math.nan,
                "screen_reason": "outside broad original-2C3 energy envelope",
            })
        elif not texture_consistent(row):
            audit.append({
                **asdict(row), "included_in_03": False,
                "curve_residual": math.nan, "curve_threshold": math.nan,
                "screen_reason": "pinning texture flipped to opposite branch",
            })
        else:
            candidates.append(row)

    retained: list[FieldPoint] = []
    keys = sorted({(row.J2, row.D, row.branch) for row in candidates})
    for J2, D, branch in keys:
        group = [row for row in candidates if (
            close(row.J2, J2) and row.D == D and row.branch == branch
        )]
        distinct_h = {round(row.signed_h, 12) for row in group}
        if len(distinct_h) < 2:
            for row in group:
                audit.append({
                    **asdict(row), "included_in_03": False,
                    "curve_residual": math.nan, "curve_threshold": math.nan,
                    "screen_reason": "isolated point: no E(h) curve",
                })
            continue

        degree = 3 if len(distinct_h) >= 6 else 2 if len(distinct_h) >= 4 else 1
        h_scale = max(abs(row.signed_h) for row in group) or 1.0
        scaled_h = np.asarray([row.signed_h / h_scale for row in group])
        matrix = np.column_stack([
            scaled_h ** power for power in range(degree + 1)
        ])
        energy = np.asarray([row.energy for row in group])
        fit = robust_fit(matrix, energy, degree)
        if fit is None:
            # Two- or three-field curves do not contain enough redundancy for
            # an outlier decision; retain them rather than silently ranking.
            residuals = energy - matrix @ np.linalg.lstsq(
                matrix, energy, rcond=None,
            )[0]
            threshold = math.inf
        else:
            residuals = fit.residuals
            threshold = max(1.0e-3, 4.5 * fit.scale)

        for row, residual in zip(group, residuals):
            included = abs(float(residual)) <= threshold
            if included:
                retained.append(row)
            audit.append({
                **asdict(row), "included_in_03": included,
                "curve_residual": float(residual),
                "curve_threshold": threshold,
                "screen_reason": (
                    "included: broad energy + texture + normal E(h) curve"
                    if included else "obvious E(h) curve outlier/zigzag"
                ),
            })

    # Remove the remaining literal zigzags.  The aligned plaquette branch
    # must be non-increasing as positive h grows; the aligned dimer branch is
    # non-decreasing when signed h moves from negative values towards zero.
    # When a pair violates that response, discard the point less compatible
    # with the robust smooth curve, then re-check.  This is not an energy-rank
    # choice between replicas at a fixed h.
    residual_by_id = {
        row["point_id"]: abs(float(row["curve_residual"]))
        for row in audit
        if row["included_in_03"] and math.isfinite(float(row["curve_residual"]))
    }
    removed_for_zigzag: set[str] = set()
    for J2, D, branch in sorted({
        (row.J2, row.D, row.branch) for row in retained
    }):
        active = [row for row in retained if (
            close(row.J2, J2) and row.D == D and row.branch == branch
        )]
        while True:
            by_h: dict[float, list[FieldPoint]] = {}
            for row in active:
                by_h.setdefault(round(row.signed_h, 12), []).append(row)
            fields = sorted(by_h)
            violation = None
            for left_h, right_h in zip(fields, fields[1:]):
                left_E = float(np.median([row.energy for row in by_h[left_h]]))
                right_E = float(np.median([row.energy for row in by_h[right_h]]))
                if branch == "plaquette":
                    bad = right_E > left_E + 5.0e-5
                else:
                    bad = right_E < left_E - 5.0e-5
                if bad:
                    violation = (left_h, right_h)
                    break
            if violation is None:
                break
            alternatives = [
                row for field in violation for row in by_h[field]
            ]
            victim = max(
                alternatives,
                key=lambda row: residual_by_id.get(row.point_id, 0.0),
            )
            active.remove(victim)
            removed_for_zigzag.add(victim.point_id)

    if removed_for_zigzag:
        retained = [row for row in retained
                    if row.point_id not in removed_for_zigzag]
        for row in audit:
            if row["point_id"] in removed_for_zigzag:
                row["included_in_03"] = False
                row["screen_reason"] = "obvious nonmonotonic E(h) zigzag"
    return sorted(retained, key=lambda row: (
        row.J2, row.D, row.branch, row.signed_h, row.energy,
    )), audit


def choose_representatives(
    rows: list[FieldPoint], references: dict[tuple[float, int], float],
) -> tuple[list[FieldPoint], list[RepresentativeDecision]]:
    decisions: list[RepresentativeDecision] = []
    selected: list[FieldPoint] = []
    keys = sorted({(row.J2, row.D, row.branch, round(row.signed_h, 12))
                   for row in rows})
    for J2, D, branch, signed_h in keys:
        group = [row for row in rows if (
            close(row.J2, J2) and row.D == D and row.branch == branch
            and close(row.signed_h, signed_h)
        )]
        plausible = [row for row in group if energy_plausible(row, references)]
        if not plausible:
            decisions.extend(RepresentativeDecision(
                row, False, "energy outside original-2C3 physical envelope"
            ) for row in group)
            continue
        consistent = [row for row in plausible if texture_consistent(row)]
        if not consistent:
            decisions.extend(RepresentativeDecision(
                row, False, "branch texture flipped") for row in group
            )
            continue
        max_chi = max(row.chi for row in consistent)
        high_chi = [row for row in consistent if row.chi >= 0.75 * max_chi]
        converged = [row for row in high_chi
                     if not math.isfinite(row.chi_energy_shift)
                     or row.chi_energy_shift <= 5.0e-4]
        pool = converged or high_chi
        winner = min(pool, key=lambda row: (
            row.energy, -row.chi,
            row.chi_energy_shift if math.isfinite(row.chi_energy_shift)
            else math.inf,
        ))
        selected.append(winner)
        for row in group:
            if row.point_id == winner.point_id:
                reason = "selected variational/high-chi representative"
                decisions.append(RepresentativeDecision(row, True, reason))
            elif not energy_plausible(row, references):
                decisions.append(RepresentativeDecision(
                    row, False, "energy outside original-2C3 physical envelope",
                ))
            elif not texture_consistent(row):
                decisions.append(RepresentativeDecision(
                    row, False, "branch texture flipped",
                ))
            elif row.chi < 0.75 * max_chi:
                decisions.append(RepresentativeDecision(
                    row, False, "chi below 75% of group maximum",
                ))
            elif (math.isfinite(row.chi_energy_shift)
                  and row.chi_energy_shift > 5.0e-4):
                decisions.append(RepresentativeDecision(
                    row, False, "large chi-lookahead energy shift",
                ))
            else:
                decisions.append(RepresentativeDecision(
                    row, False, "higher-energy duplicate/protocol",
                ))
    return sorted(selected, key=lambda row: (
        row.J2, row.D, row.branch, row.signed_h,
    )), decisions


def load_gapped_a() -> dict[float, float]:
    values: dict[float, float] = {}
    with ENERGY_FITS.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if (row.get("ansatz") == "2C3" and row.get("model") == "gapped"
                    and row.get("parameter") == "a"):
                values[float(row["J2"])] = float(row["central"])
    return values


def robust_fit(X: np.ndarray, y: np.ndarray, degree: int) -> RobustFit | None:
    n, p = X.shape
    if n < p + 2 or np.linalg.matrix_rank(X) < p:
        return None
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    weights = np.ones(n)
    scale = 0.0
    for _ in range(40):
        residuals = y - X @ beta
        median = float(np.median(residuals))
        scale = 1.4826 * float(np.median(np.abs(residuals - median)))
        scale = max(scale, 2.0e-7)
        ratio = np.abs(residuals - median) / (1.5 * scale)
        new_weights = np.ones(n)
        mask = ratio > 1.0
        new_weights[mask] = 1.0 / ratio[mask]
        weighted_X = X * np.sqrt(new_weights)[:, None]
        weighted_y = y * np.sqrt(new_weights)
        new_beta, *_ = np.linalg.lstsq(weighted_X, weighted_y, rcond=None)
        if np.max(np.abs(new_beta - beta)) <= 1.0e-12:
            beta, weights = new_beta, new_weights
            break
        beta, weights = new_beta, new_weights
    residuals = y - X @ beta
    median = float(np.median(residuals))
    scale = max(1.4826 * float(np.median(np.abs(residuals - median))), 2.0e-7)
    inliers = np.abs(residuals - median) <= max(4.5 * scale, 2.0e-6)
    if int(np.sum(inliers)) < p + 1 or np.linalg.matrix_rank(X[inliers]) < p:
        inliers = np.ones(n, dtype=bool)
    beta, *_ = np.linalg.lstsq(X[inliers], y[inliers], rcond=None)
    residuals = y - X @ beta
    rms = float(np.sqrt(np.mean(residuals[inliers] ** 2)))
    dof = max(1, int(np.sum(inliers)) - p)
    sigma2 = float(np.sum(residuals[inliers] ** 2) / dof)
    covariance = sigma2 * np.linalg.pinv(X[inliers].T @ X[inliers])
    return RobustFit(
        degree=degree, beta=beta, covariance=covariance,
        residuals=residuals, inliers=inliers, rms=rms, scale=scale,
        n=n, n_inlier=int(np.sum(inliers)),
    )


def robust_weighted_fit_all(
    X: np.ndarray, y: np.ndarray, degree: int,
) -> RobustFit | None:
    """Huber-weighted fit in which every supplied point retains nonzero weight."""
    n, p = X.shape
    if n < p or np.linalg.matrix_rank(X) < p:
        return None
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    weights = np.ones(n)
    scale = 2.0e-7
    for _ in range(60):
        residuals = y - X @ beta
        median = float(np.median(residuals))
        scale = max(
            1.4826 * float(np.median(np.abs(residuals - median))), 2.0e-7,
        )
        ratio = np.abs(residuals - median) / (1.5 * scale)
        new_weights = np.ones(n)
        mask = ratio > 1.0
        new_weights[mask] = 1.0 / ratio[mask]
        weighted_X = X * np.sqrt(new_weights)[:, None]
        weighted_y = y * np.sqrt(new_weights)
        new_beta, *_ = np.linalg.lstsq(weighted_X, weighted_y, rcond=None)
        if np.max(np.abs(new_beta - beta)) <= 1.0e-12:
            beta, weights = new_beta, new_weights
            break
        beta, weights = new_beta, new_weights
    residuals = y - X @ beta
    effective_dof = max(1.0, float(np.sum(weights)) - p)
    sigma2 = float(np.sum(weights * residuals ** 2) / effective_dof)
    covariance = sigma2 * np.linalg.pinv(X.T @ (weights[:, None] * X))
    return RobustFit(
        degree=degree, beta=beta, covariance=covariance,
        residuals=residuals, inliers=np.ones(n, dtype=bool),
        rms=float(np.sqrt(np.mean(residuals ** 2))), scale=scale,
        n=n, n_inlier=n,
    )


def design(rows: list[FieldPoint], a_g: float, degree: int) -> np.ndarray:
    columns = []
    for row in rows:
        finite_D = math.exp(-a_g * row.D)
        values = [1.0, finite_D, row.signed_h, row.signed_h * finite_D]
        if degree == 2:
            values.append(row.signed_h ** 2)
        columns.append(values)
    return np.asarray(columns, dtype=float)


def aicc(fit: RobustFit) -> float:
    n = fit.n_inlier
    k = len(fit.beta)
    rss = max(float(np.sum(fit.residuals[fit.inliers] ** 2)), 1.0e-30)
    score = n * math.log(rss / n) + 2.0 * k
    if n > k + 1:
        score += 2.0 * k * (k + 1) / (n - k - 1)
    else:
        score += 1.0e6
    return score


def fit_branch(rows: list[FieldPoint], a_g: float) -> RobustFit | None:
    y = np.asarray([row.energy for row in rows], dtype=float)
    linear = robust_fit(design(rows, a_g, 1), y, 1)
    quadratic = robust_fit(design(rows, a_g, 2), y, 2)
    if linear is None:
        return quadratic
    if quadratic is None:
        return linear
    hscale = max(abs(row.signed_h) for row in rows)
    linear_term = abs(quadratic.beta[2] * hscale)
    quadratic_term = abs(quadratic.beta[4] * hscale * hscale)
    physically_perturbative = quadratic_term <= max(linear_term, 1.0e-8)
    if physically_perturbative and aicc(quadratic) + 2.0 < aicc(linear):
        return quadratic
    return linear


def polynomial_coefficients(fit: RobustFit) -> tuple[float, float, float]:
    constant = float(fit.beta[0])
    linear = float(fit.beta[2])
    quadratic = float(fit.beta[4]) if fit.degree == 2 else 0.0
    return constant, linear, quadratic


def crossing_from_coefficients(
    plaquette: tuple[float, float, float],
    dimer: tuple[float, float, float],
    limit: float,
) -> float:
    coefficients = np.asarray(plaquette) - np.asarray(dimer)
    constant, linear, quadratic = coefficients
    roots: list[float] = []
    if abs(quadratic) <= 1.0e-10:
        if abs(linear) > 1.0e-12:
            roots = [-constant / linear]
    else:
        discriminant = linear * linear - 4.0 * quadratic * constant
        if discriminant >= 0.0:
            root = math.sqrt(discriminant)
            roots = [(-linear - root) / (2.0 * quadratic),
                     (-linear + root) / (2.0 * quadratic)]
    physical = [value for value in roots if math.isfinite(value)
                and abs(value) <= limit]
    return min(physical, key=abs) if physical else math.nan


def crossing_uncertainty(
    plaquette: RobustFit, dimer: RobustFit, limit: float,
    seed: int,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    samples = []
    try:
        p_samples = rng.multivariate_normal(
            plaquette.beta, plaquette.covariance, size=2500,
            check_valid="ignore",
        )
        d_samples = rng.multivariate_normal(
            dimer.beta, dimer.covariance, size=2500,
            check_valid="ignore",
        )
    except (ValueError, np.linalg.LinAlgError):
        return math.nan, math.nan, math.nan
    for p_beta, d_beta in zip(p_samples, d_samples):
        p_coeff = (p_beta[0], p_beta[2], p_beta[4]
                   if plaquette.degree == 2 else 0.0)
        d_coeff = (d_beta[0], d_beta[2], d_beta[4]
                   if dimer.degree == 2 else 0.0)
        root = crossing_from_coefficients(p_coeff, d_coeff, limit)
        if math.isfinite(root):
            samples.append(root)
    if len(samples) < 200:
        return math.nan, math.nan, math.nan
    q16, q84 = np.quantile(samples, [0.16, 0.84])
    return float(0.5 * (q84 - q16)), float(q16), float(q84)


def fit_crossing_for_j2(
    J2: float, rows: list[FieldPoint], a_g: float,
) -> tuple[CrossingResult, list[dict], list[dict]]:
    branch_rows = {
        branch: [row for row in rows if row.branch == branch]
        for branch in SCALAR_BRANCHES
    }
    candidate_windows = (0.003, 0.005, 0.01, 0.02, 0.04, 0.08)
    chosen = None
    for window in candidate_windows:
        subsets = {
            branch: [row for row in values
                     if abs(row.signed_h) <= window + 1.0e-12]
            for branch, values in branch_rows.items()
        }
        sufficient = True
        for branch, subset in subsets.items():
            nonzero = {round(abs(row.signed_h), 12) for row in subset
                       if abs(row.signed_h) > 1.0e-12}
            dimensions = {row.D for row in subset}
            if len(nonzero) < 2 or len(dimensions) < 2 or len(subset) < 7:
                sufficient = False
                break
        if not sufficient:
            continue
        fits = {branch: fit_branch(subset, a_g)
                for branch, subset in subsets.items()}
        if all(fit is not None for fit in fits.values()):
            chosen = (window, subsets, fits)
            break

    if chosen is None:
        result = CrossingResult(
            J2, False, "no common near-zero window with >=2 fields and >=2 D",
            math.nan, math.nan, math.nan, math.nan, math.nan, math.nan,
            0, 0, 0, 0, 0, 0, math.nan, math.nan,
            " ".join(str(D) for D in sorted({row.D for row in rows})),
            0, math.nan,
        )
        return result, [], []

    window, subsets, fits_any = chosen
    fits: dict[str, RobustFit] = {
        branch: fit for branch, fit in fits_any.items() if fit is not None
    }
    pfit = fits["plaquette"]
    dfit = fits["dimer-plaquette"]
    limit = max(0.012, 2.0 * window)
    p_coeff = polynomial_coefficients(pfit)
    d_coeff = polynomial_coefficients(dfit)
    h_c = crossing_from_coefficients(p_coeff, d_coeff, limit)
    slope_difference = p_coeff[1] - d_coeff[1]
    error, q16, q84 = crossing_uncertainty(
        pfit, dfit, limit, seed=int(round(J2 * 1.0e6)),
    )

    reasons = []
    if not math.isfinite(h_c):
        reasons.append("no near-zero real crossing")
    if abs(slope_difference) < 0.02:
        reasons.append("branch slopes nearly parallel")
    # With the signed-field convention used here, increasing positive h must
    # lower the plaquette branch, whereas moving towards more negative h must
    # lower the dimer-plaquette branch.  A violation at x_D=0 is not exotic
    # response: it means the D extrapolation is being driven by unbalanced
    # branch/D coverage (the present J2=0.29 data are the canonical example).
    if p_coeff[1] >= 0.0 or d_coeff[1] <= 0.0:
        reasons.append("D-infinity branch slope has the wrong pinning-field sign")
    if max(pfit.rms, dfit.rms) > 2.5e-4:
        reasons.append("robust-fit RMS exceeds 2.5e-4")
    if pfit.n_inlier / pfit.n < 0.65 or dfit.n_inlier / dfit.n < 0.65:
        reasons.append("fewer than 65% robust inliers")
    if math.isfinite(error) and error > 0.006:
        reasons.append("crossing uncertainty exceeds 0.006")

    fit_audit = []
    for branch in SCALAR_BRANCHES:
        fit = fits[branch]
        for row, residual, inlier in zip(
            subsets[branch], fit.residuals, fit.inliers
        ):
            fit_audit.append({
                "J2": J2, "branch": branch, "D": row.D,
                "signed_h": row.signed_h, "energy": row.energy,
                "residual": residual, "robust_inlier": bool(inlier),
                "h_window": window, "degree": fit.degree,
                "point_id": row.point_id, "path": row.path,
            })

    per_D_rows = []
    per_D_crossings = []
    for D in sorted({row.D for row in rows}):
        local_fits = {}
        for branch in SCALAR_BRANCHES:
            subset = [row for row in subsets[branch] if row.D == D]
            distinct = {round(row.signed_h, 12) for row in subset}
            if len(distinct) < 3:
                continue
            h_values = np.asarray([row.signed_h for row in subset])
            energy = np.asarray([row.energy for row in subset])
            matrix = np.column_stack((np.ones(len(subset)), h_values))
            local = robust_fit(matrix, energy, 1)
            if local is not None:
                local_fits[branch] = local
        if set(local_fits) != set(SCALAR_BRANCHES):
            continue
        local_p = (local_fits["plaquette"].beta[0],
                   local_fits["plaquette"].beta[1], 0.0)
        local_d = (local_fits["dimer-plaquette"].beta[0],
                   local_fits["dimer-plaquette"].beta[1], 0.0)
        local_hc = crossing_from_coefficients(local_p, local_d, limit)
        if math.isfinite(local_hc):
            per_D_crossings.append(local_hc)
        per_D_rows.append({
            "J2": J2, "D": D, "h_c_D": local_hc,
            "plaquette_rms": local_fits["plaquette"].rms,
            "dimer_rms": local_fits["dimer-plaquette"].rms,
            "h_window": window,
        })
    spread = (float(np.std(per_D_crossings))
              if len(per_D_crossings) >= 2 else math.nan)
    if math.isfinite(spread) and spread > 0.006:
        reasons.append("finite-D crossings spread by more than 0.006")

    result = CrossingResult(
        J2=J2, accepted=not reasons,
        reason="accepted" if not reasons else "; ".join(reasons),
        h_window=window, h_c=h_c, h_c_error=error,
        h_c_q16=q16, h_c_q84=q84,
        slope_difference=slope_difference,
        plaquette_degree=pfit.degree, dimer_degree=dfit.degree,
        plaquette_n=pfit.n, dimer_n=dfit.n,
        plaquette_inliers=pfit.n_inlier, dimer_inliers=dfit.n_inlier,
        plaquette_rms=pfit.rms, dimer_rms=dfit.rms,
        Ds=" ".join(str(D) for D in sorted({row.D for row in rows})),
        per_D_crossing_count=len(per_D_crossings),
        per_D_crossing_spread=spread,
    )
    return result, fit_audit, per_D_rows


def plot_energy_surface(
    rows: list[FieldPoint], representatives: list[FieldPoint], output: Path,
    *, title: str | None = None,
) -> None:
    dimensions = sorted({row.D for row in rows})
    norm = matplotlib.colors.Normalize(min(dimensions), max(dimensions))
    cmap = plt.get_cmap("viridis")
    figure = plt.figure(figsize=(11.0, 8.2), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    for D in dimensions:
        alpha = 0.18 + 0.78 * norm(D)
        color = cmap(norm(D))
        subset = [row for row in rows if row.D == D]
        for sign, marker in ((-1, "s"), (0, "x"), (1, "o")):
            signed = [row for row in subset if (
                (-1 if row.signed_h < -1e-12 else
                 1 if row.signed_h > 1e-12 else 0) == sign
            )]
            if signed:
                axis.scatter(
                    [row.J2 for row in signed],
                    [row.signed_h for row in signed],
                    [row.energy for row in signed], color=[color],
                    alpha=alpha, s=15, marker=marker, depthshade=False,
                )
        for J2 in sorted({row.J2 for row in subset}):
            curve = sorted(
                [row for row in representatives
                 if row.D == D and close(row.J2, J2)],
                key=lambda row: row.signed_h,
            )
            if len(curve) >= 2:
                axis.plot(
                    [row.J2 for row in curve],
                    [row.signed_h for row in curve],
                    [row.energy for row in curve],
                    color=color, alpha=alpha, linewidth=1.0,
                )
    axis.set_xlabel(r"$J_2$")
    axis.set_ylabel(r"signed pinning field $h$")
    axis.set_zlabel(r"$E$ per site")
    axis.set_title(title or (
        r"All scalar pinning results in $B$: plaquette $h>0$, "
        r"dimer-plaquette $h<0$"
    ))
    axis.view_init(elev=24, azim=-58)
    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    colorbar = figure.colorbar(scalar, ax=axis, pad=0.10, shrink=0.70)
    colorbar.set_label(r"$D$ (also encoded by opacity)")
    colorbar.set_ticks(dimensions)
    handles = [
        Line2D([], [], linestyle="none", marker="s", color="0.25",
               label="dimer pin, signed h<0"),
        Line2D([], [], linestyle="none", marker="o", color="0.25",
               label="plaquette pin, signed h>0"),
        Line2D([], [], linestyle="none", marker="x", color="0.25",
               label="h=0 endpoint"),
    ]
    axis.legend(handles=handles, loc="upper left", frameon=False, fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    plt.close(figure)


def plot_all_good_points(
    rows: list[FieldPoint], original_energies: dict[tuple[float, int], float],
    output: Path, video_output: Path | None = None,
) -> None:
    """Plot every non-obvious-bad B point without convergence ranking.

    Unlike the fit-representative surface, this deliberately applies no chi
    cutoff, no lookahead cutoff, and no variational winner selection.  Lines
    are only visual guides: for duplicate protocols at the same h they pass
    through the median energy, separately for each branch and (J2,D).
    """
    dimensions = sorted({row.D for row in rows})
    norm = matplotlib.colors.Normalize(min(dimensions), max(dimensions))
    cmap = plt.get_cmap("viridis")
    figure = plt.figure(figsize=(11.0, 8.2), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    axis.computed_zorder = False

    # Draw the unbiased D=10 reference first so every pinning curve/point is
    # layered above it.  Include J2=.275 in both pieces for visual continuity.
    reference = sorted(
        (J2, energy) for (J2, D), energy in original_energies.items()
        if D == 10 and 0.24 - 1.0e-12 <= J2 <= 0.34 + 1.0e-12
    )
    solid = [(J2, energy) for J2, energy in reference
             if J2 <= 0.275 + 1.0e-12]
    dashed = [(J2, energy) for J2, energy in reference
              if J2 >= 0.275 - 1.0e-12]
    for segment, linestyle, line_color in (
        (solid, "-", "#c62828"), (dashed, "--", "0.10"),
    ):
        if len(segment) >= 2:
            axis.plot(
                [item[0] for item in segment], [0.0] * len(segment),
                [item[1] for item in segment], color=line_color,
                linestyle=linestyle, linewidth=3.2, alpha=0.90, zorder=0,
            )
    for D in dimensions:
        alpha = 0.18 + 0.78 * norm(D)
        color = cmap(norm(D))
        subset = [row for row in rows if row.D == D]
        axis.scatter(
            [row.J2 for row in subset],
            [row.signed_h for row in subset],
            [row.energy for row in subset],
            color=[color], alpha=alpha, s=7, marker="o",
            linewidths=0.0, depthshade=False, zorder=2,
        )
        for J2 in sorted({row.J2 for row in subset}):
            for branch in SCALAR_BRANCHES:
                curve = [row for row in subset
                         if close(row.J2, J2) and row.branch == branch]
                by_h: dict[float, list[float]] = {}
                for row in curve:
                    by_h.setdefault(round(row.signed_h, 12), []).append(row.energy)
                if len(by_h) < 2:
                    continue
                fields = sorted(by_h)
                energies = [float(np.median(by_h[field])) for field in fields]
                axis.plot(
                    [J2] * len(fields), fields, energies,
                    color=color, alpha=alpha, linewidth=0.55, zorder=1,
                )
    axis.set_xlabel(r"$J_2$")
    axis.set_ylabel(r"signed pinning field $h$")
    axis.set_zlabel(r"$E$ per site")
    axis.set_title(
        r"All non-obvious-bad points in $B$: no $\chi$/lookahead/"
        r"variational-winner filtering"
    )
    axis.view_init(elev=24, azim=-58)
    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    colorbar = figure.colorbar(scalar, ax=axis, pad=0.10, shrink=0.70)
    colorbar.set_label(r"$D$ (also encoded by opacity)")
    colorbar.set_ticks(dimensions)
    axis.legend(handles=[
        Line2D([], [], color="#c62828", linewidth=3.2, linestyle="-",
               label=r"original 2C3 $D=10$, $0.24\leq J_2\leq0.275$"),
        Line2D([], [], color="0.10", linewidth=3.2, linestyle="--",
               label=r"original 2C3 $D=10$, $0.275\leq J_2\leq0.34$"),
    ], loc="upper left", frameon=False, fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    if video_output is not None:
        from matplotlib.animation import FFMpegWriter, FuncAnimation

        video_output.parent.mkdir(parents=True, exist_ok=True)
        # Rotate clockwise through the same half turn used in the original
        # movie; only the direction is reversed.
        azimuths = np.linspace(-58.0, -238.0, 61)

        def rotate(frame: int) -> tuple:
            axis.view_init(elev=24, azim=float(azimuths[frame]))
            return (axis,)

        animation = FuncAnimation(
            figure, rotate, frames=len(azimuths), interval=100, blit=False,
        )
        animation.save(
            video_output,
            writer=FFMpegWriter(
                fps=10, bitrate=2400,
                metadata={"title": "03 all-good pinning-energy surface"},
            ),
            dpi=120,
        )
    plt.close(figure)


def evaluate_fit(fit: RobustFit, h: np.ndarray, finite_D: float) -> np.ndarray:
    values = (
        fit.beta[0] + fit.beta[1] * finite_D
        + fit.beta[2] * h + fit.beta[3] * h * finite_D
    )
    if fit.degree == 2:
        values = values + fit.beta[4] * h * h
    return values


def plot_fit_diagnostic(
    result: CrossingResult, rows: list[FieldPoint], a_g: float, output: Path,
) -> None:
    """Show exactly which near-zero data determine one reported crossing."""
    if not math.isfinite(result.h_window):
        return
    window = result.h_window
    subsets = {
        branch: [row for row in rows if row.branch == branch
                 and abs(row.signed_h) <= window + 1.0e-12]
        for branch in SCALAR_BRANCHES
    }
    fits = {branch: fit_branch(values, a_g)
            for branch, values in subsets.items()}
    if any(fit is None for fit in fits.values()):
        return

    dimensions = sorted({row.D for values in subsets.values() for row in values})
    norm = matplotlib.colors.Normalize(min(dimensions), max(dimensions))
    cmap = plt.get_cmap("viridis")
    figure, axis = plt.subplots(figsize=(7.2, 5.8), constrained_layout=True)
    styles = {
        "dimer-plaquette": ("s", "--", "dimer-plaquette branch"),
        "plaquette": ("o", "-", "plaquette branch"),
    }
    grid = np.linspace(-window, window, 301)
    for branch in SCALAR_BRANCHES:
        fit = fits[branch]
        assert fit is not None
        marker, linestyle, _ = styles[branch]
        for D in dimensions:
            subset = [row for row in subsets[branch] if row.D == D]
            if not subset:
                continue
            color = cmap(norm(D))
            alpha = 0.28 + 0.68 * norm(D)
            axis.scatter(
                [row.signed_h for row in subset],
                [row.energy for row in subset], marker=marker,
                color=[color], alpha=alpha, s=31, zorder=3,
            )
            finite_D = math.exp(-a_g * D)
            axis.plot(
                grid, evaluate_fit(fit, grid, finite_D),
                color=color, alpha=alpha, linestyle=linestyle,
                linewidth=0.85, zorder=1,
            )
        axis.plot(
            grid, evaluate_fit(fit, grid, 0.0), color="black",
            linestyle=linestyle, linewidth=2.0, zorder=2,
        )
    if math.isfinite(result.h_c):
        axis.axvline(result.h_c, color="#9b2f6b", linewidth=1.3,
                     alpha=0.85, zorder=0)
    axis.axvline(0.0, color="0.55", linewidth=0.8, zorder=0)
    status = "accepted" if result.accepted else f"rejected: {result.reason}"
    axis.set_title(
        rf"$J_2={result.J2:g}$; $h_c={result.h_c:+.3g}$; {status}",
        fontsize=10,
    )
    axis.set_xlabel(r"signed pinning field $h$")
    axis.set_ylabel(r"$E$ per site")
    axis.grid(alpha=0.16)
    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    colorbar = figure.colorbar(scalar, ax=axis, pad=0.02)
    colorbar.set_label(r"$D$")
    colorbar.set_ticks(dimensions)
    handles = [
        Line2D([], [], color="black", linestyle="-", marker="o",
               label=r"plaquette; black curve is $D\to\infty$"),
        Line2D([], [], color="black", linestyle="--", marker="s",
               label=r"dimer-plaquette; black curve is $D\to\infty$"),
        Line2D([], [], color="#9b2f6b", linestyle="-", label=r"reported $h_c$"),
    ]
    axis.legend(handles=handles, frameon=False, fontsize=8, loc="best")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    plt.close(figure)


def plot_phase_boundary(results: list[CrossingResult], output: Path) -> None:
    ordered = sorted(results, key=lambda row: row.J2)
    accepted = [row for row in ordered
                if row.accepted and math.isfinite(row.h_c)]
    figure, axis = plt.subplots(figsize=(6.8, 6.0), constrained_layout=True)
    if accepted:
        h = np.asarray([row.h_c for row in accepted])
        J2 = np.asarray([row.J2 for row in accepted])
        errors = np.asarray([
            row.h_c_error if math.isfinite(row.h_c_error) else 0.0
            for row in accepted
        ])
        # A rejected J2 is a genuine hole in the inferred boundary, not a
        # license to interpolate a visually smooth curve across it.
        segment: list[CrossingResult] = []
        for row in ordered + [CrossingResult(
            math.nan, False, "sentinel", math.nan, math.nan, math.nan,
            math.nan, math.nan, math.nan, 0, 0, 0, 0, 0, 0,
            math.nan, math.nan, "", 0, math.nan,
        )]:
            if row.accepted and math.isfinite(row.h_c):
                segment.append(row)
                continue
            if len(segment) >= 2:
                axis.plot(
                    [item.h_c for item in segment],
                    [item.J2 for item in segment],
                    color="#6b2f8a", linewidth=1.8, zorder=3,
                )
            segment = []
        axis.errorbar(
            h, J2, xerr=errors, linestyle="none", marker="o",
            markersize=6.0, color="#6b2f8a", capsize=2.5, zorder=4,
            label=r"$D\to\infty$ robust crossing",
        )
    axis.plot(
        [0.0, 0.0], [0.24, 0.275], color="#159e83", linewidth=7.0,
        solid_capstyle="butt", zorder=2,
        label=r"$h=0$, $0.24\leq J_2\leq0.275$",
    )
    rejected = [row for row in results if not row.accepted]
    if rejected:
        rejected_x = [row.h_c if math.isfinite(row.h_c) else 0.0
                      for row in rejected]
        axis.scatter(
            rejected_x, [row.J2 for row in rejected],
            color="0.55", marker="x", s=35,
            label="rejected/unstable fit", zorder=3,
        )
        for x_value, row in zip(rejected_x, rejected):
            if not math.isfinite(row.h_c):
                axis.annotate(
                    "no real root", (x_value, row.J2), xytext=(5, 0),
                    textcoords="offset points", color="0.45", fontsize=7,
                    va="center",
                )
    all_h = [0.0] + [row.h_c for row in results if math.isfinite(row.h_c)]
    limit = max(0.004, 1.25 * max(abs(value) for value in all_h))
    axis.set_xlim(-limit, limit)
    axis.set_ylim(0.235, max(0.345, max((row.J2 for row in results), default=.34) + .005))
    axis.set_xlabel(r"crossing field $h_c$")
    axis.set_ylabel(r"$J_2$")
    axis.grid(alpha=0.20)
    axis.legend(frameon=False, fontsize=9, loc="best")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    plt.close(figure)


def full_energy_design(rows: list[FieldPoint], a_g: float) -> np.ndarray:
    """Cubic-in-h energy surface with finite-D corrections through h^2."""
    matrix = []
    for row in rows:
        x_D = math.exp(-a_g * row.D)
        h = row.signed_h
        matrix.append([
            1.0, x_D, h, h * x_D, h * h, h * h * x_D, h ** 3,
        ])
    return np.asarray(matrix, dtype=float)


def full_energy_coefficients(fit: RobustFit) -> np.ndarray:
    """Return ascending D=infinity polynomial coefficients in h."""
    return np.asarray([
        fit.beta[0], fit.beta[2], fit.beta[4], fit.beta[6],
    ], dtype=float)


def crossing_from_polynomials(
    plaquette: np.ndarray, dimer: np.ndarray, limit: float = 0.01,
) -> float:
    difference = np.asarray(plaquette) - np.asarray(dimer)
    while len(difference) > 1 and abs(difference[-1]) < 1.0e-12:
        difference = difference[:-1]
    if len(difference) <= 1:
        return math.nan
    roots = np.roots(difference[::-1])
    real = [float(root.real) for root in roots
            if abs(root.imag) <= 1.0e-8 and abs(root.real) <= limit]
    return min(real, key=abs) if real else math.nan


def full_crossing_uncertainty(
    plaquette: RobustFit, dimer: RobustFit, seed: int,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    try:
        p_samples = rng.multivariate_normal(
            plaquette.beta, plaquette.covariance, size=5000,
            check_valid="ignore",
        )
        d_samples = rng.multivariate_normal(
            dimer.beta, dimer.covariance, size=5000,
            check_valid="ignore",
        )
    except (ValueError, np.linalg.LinAlgError):
        return math.nan, math.nan, math.nan
    roots = []
    for p_beta, d_beta in zip(p_samples, d_samples):
        p_coeff = p_beta[[0, 2, 4, 6]]
        d_coeff = d_beta[[0, 2, 4, 6]]
        root = crossing_from_polynomials(p_coeff, d_coeff)
        if math.isfinite(root):
            roots.append(root)
    if len(roots) < 500:
        return math.nan, math.nan, math.nan
    q025, q975 = np.quantile(roots, [0.025, 0.975])
    sigma = (float(q975) - float(q025)) / (2.0 * 1.959963984540054)
    return sigma, float(q025), float(q975)


def local_full_h_crossings(rows: list[FieldPoint]) -> list[dict]:
    """Return one ordinary quadratic crossing per bond dimension.

    Replicas/protocols at the same field are first replaced by their median,
    so every sampled field and every D enters exactly once.  This is the
    finite-D construction that is visually represented by figure 03.
    """
    output = []
    for D in sorted({row.D for row in rows}):
        coefficients: dict[str, np.ndarray] = {}
        diagnostics: dict[str, tuple[int, float]] = {}
        for branch in SCALAR_BRANCHES:
            subset = [row for row in rows
                      if row.D == D and row.branch == branch]
            by_h: dict[float, list[float]] = {}
            for row in subset:
                by_h.setdefault(round(row.signed_h, 12), []).append(row.energy)
            if len(by_h) < 3:
                continue
            fields = np.asarray(sorted(by_h), dtype=float)
            energy = np.asarray([
                float(np.median(by_h[field])) for field in fields
            ])
            degree = 2
            h_scale = float(np.max(np.abs(fields))) or 1.0
            scaled = fields / h_scale
            matrix = np.column_stack([
                scaled ** power for power in range(degree + 1)
            ])
            if np.linalg.matrix_rank(matrix) < degree + 1:
                continue
            beta, *_ = np.linalg.lstsq(matrix, energy, rcond=None)
            residuals = energy - matrix @ beta
            coeff = np.zeros(3)
            for power, value in enumerate(beta):
                coeff[power] = value / (h_scale ** power)
            coefficients[branch] = coeff
            diagnostics[branch] = (
                len(fields), float(np.sqrt(np.mean(residuals ** 2))),
            )
        if set(coefficients) != set(SCALAR_BRANCHES):
            continue
        crossing = crossing_from_polynomials(
            coefficients["plaquette"], coefficients["dimer-plaquette"],
        )
        if not math.isfinite(crossing):
            continue
        output.append({
            "D": D, "h_c_D": crossing,
            "plaquette_n_fields": diagnostics["plaquette"][0],
            "dimer_n_fields": diagnostics["dimer-plaquette"][0],
            "plaquette_rms": diagnostics["plaquette"][1],
            "dimer_rms": diagnostics["dimer-plaquette"][1],
            "plaquette_E0": coefficients["plaquette"][0],
            "plaquette_E1": coefficients["plaquette"][1],
            "plaquette_E2": coefficients["plaquette"][2],
            "dimer_E0": coefficients["dimer-plaquette"][0],
            "dimer_E1": coefficients["dimer-plaquette"][1],
            "dimer_E2": coefficients["dimer-plaquette"][2],
        })
    return output


def fit_all_good_crossing(
    J2: float, rows: list[FieldPoint], _a_g: float,
) -> tuple[dict, list[dict], list[dict]]:
    """Constant high-D extrapolation of the equally weighted finite-D roots.

    This intentionally contains no robust weighting.  Repeated runs cannot
    change the answer because they were median-reduced before each D fit.
    """
    local = local_full_h_crossings(rows)
    for item in local:
        item["J2"] = J2
    high_D = [item for item in local if item["D"] >= 6]
    if not high_D:
        high_D = local
    finite_values = np.asarray(
        [item["h_c_D"] for item in high_D], dtype=float,
    )
    if not len(finite_values):
        return ({
            "J2": J2, "h_c": math.nan, "error95": math.nan,
            "contains_h0_95": False,
            "reason": "no D with two three-field quadratic branches",
        }, [], local)
    # The final branch energies are the equal-D constant extrapolations of
    # the three fitted coefficients.  Their closest-to-zero intersection is
    # the reported central h_c; finite-D roots determine its uncertainty.
    plaquette_coefficients = np.asarray([
        float(np.mean([item[f"plaquette_E{power}"] for item in high_D]))
        for power in range(3)
    ])
    dimer_coefficients = np.asarray([
        float(np.mean([item[f"dimer_E{power}"] for item in high_D]))
        for power in range(3)
    ])
    h_c = crossing_from_polynomials(
        plaquette_coefficients, dimer_coefficients,
    )
    finite_spread = (float(np.std(finite_values, ddof=1))
                     if len(finite_values) >= 2 else math.nan)
    stat_sigma = (finite_spread / math.sqrt(len(finite_values))
                  if math.isfinite(finite_spread) else math.nan)
    error95 = (1.959963984540054 * stat_sigma
               if math.isfinite(stat_sigma) else math.nan)
    conditional_q025 = h_c - error95 if math.isfinite(error95) else math.nan
    conditional_q975 = h_c + error95 if math.isfinite(error95) else math.nan
    contains_zero = (math.isfinite(h_c) and math.isfinite(error95)
                     and abs(h_c) <= error95)
    point_audit = [{
        "J2": J2, "branch": row.branch, "D": row.D,
        "signed_h": row.signed_h, "energy": row.energy,
        "used": True, "aggregation": "median within (D, branch, h)",
        "point_id": row.point_id, "path": row.path,
    } for row in rows]
    result = {
        "J2": J2, "h_c": h_c, "error95": error95,
        "contains_h0_95": contains_zero,
        "conditional_stat_sigma": stat_sigma,
        "conditional_q025": conditional_q025,
        "conditional_q975": conditional_q975,
        "finite_D_crossing_std": finite_spread,
        "finite_D_crossing_count": len(high_D),
        "Ds": " ".join(str(item["D"]) for item in high_D),
        "plaquette_n": sum(item["plaquette_n_fields"] for item in high_D),
        "dimer_n": sum(item["dimer_n_fields"] for item in high_D),
        "plaquette_rms": float(np.mean([
            item["plaquette_rms"] for item in high_D
        ])),
        "dimer_rms": float(np.mean([
            item["dimer_rms"] for item in high_D
        ])),
        "plaquette_E0": plaquette_coefficients[0],
        "plaquette_E1": plaquette_coefficients[1],
        "plaquette_E2": plaquette_coefficients[2],
        "dimer_E0": dimer_coefficients[0],
        "dimer_E1": dimer_coefficients[1],
        "dimer_E2": dimer_coefficients[2],
        "reason": ("median per (D,h); ordinary quadratic E_D(h); "
                   "equal-D coefficient extrapolation; finite-D-root 95% SEM"),
    }
    return result, point_audit, local


def averaged_correlations_by_D(rows: list[FieldPoint]) -> dict[int, np.ndarray]:
    """Average reruns first, giving every D exactly one triplet and one weight."""
    samples: dict[int, list[np.ndarray]] = {}
    for row in rows:
        samples.setdefault(row.D, []).append(np.asarray([
            row.strongest, row.middle, row.weakest,
        ], dtype=float))
    return {
        D: np.mean(np.asarray(values), axis=0)
        for D, values in samples.items()
    }


def fit_correlation_window(
    by_D: dict[int, np.ndarray], a_g: float, selected: list[int],
) -> dict:
    """Unweighted fixed-a_g fits of all three ranks on one shared D window."""
    x_D = np.exp(-a_g * np.asarray(selected, dtype=float))
    matrix = np.column_stack((np.ones(len(selected)), x_D))
    values = np.asarray([by_D[D] for D in selected], dtype=float)
    beta, *_ = np.linalg.lstsq(matrix, values, rcond=None)
    residuals = values - matrix @ beta
    errors = np.full(3, math.nan)
    if len(selected) > 2:
        inverse = np.linalg.pinv(matrix.T @ matrix)
        for rank in range(3):
            sigma2 = float(np.sum(residuals[:, rank] ** 2)
                           / (len(selected) - 2))
            errors[rank] = math.sqrt(max(0.0, sigma2 * inverse[0, 0]))
    ordering = np.argsort(beta[0])
    correlations = np.asarray(beta[0])[ordering]
    largest_D_values = np.asarray(by_D[max(selected)], dtype=float)
    return {
        "Ds": list(selected),
        "correlations": correlations,
        "errors": errors[ordering],
        "rms": np.sqrt(np.mean(residuals ** 2, axis=0)),
        "max_abs_residual": float(np.max(np.abs(residuals))),
        "extrapolation_distance": float(np.max(np.abs(
            correlations - largest_D_values
        ))),
    }


def candidate_high_D_correlation_fits(
    by_D: dict[int, np.ndarray], a_g: float,
) -> list[dict]:
    """All contiguous high-D windows, from the two largest D values onward."""
    dimensions = sorted(by_D)
    return [
        fit_correlation_window(by_D, a_g, dimensions[-count:])
        for count in range(2, len(dimensions) + 1)
    ] if len(dimensions) >= 2 else []


def fit_high_D_correlations(
    by_D: dict[int, np.ndarray], a_g: float,
) -> dict | None:
    """Choose the physically stable thermodynamic extrapolation.

    Every contiguous high-D window containing at least the two largest D
    values is tried.  The chosen common window for the three ranks is the one
    whose extrapolated correlation triplet moves least from the actually
    observed largest-D triplet.  This rejects exact but violently overshooting
    two-point fits; the fit residual is retained only as an audit diagnostic.
    """
    candidates = candidate_high_D_correlation_fits(by_D, a_g)
    if not candidates:
        return None
    return min(candidates, key=lambda fit: (
        fit["extrapolation_distance"],
        -len(fit["Ds"]),
    ))


def correlation_node(
    J2: float, signed_h: float, branch: str, fit: dict, handling: str,
) -> dict:
    strongest, middle, weakest = map(float, fit["correlations"])
    errors = np.asarray(fit["errors"], dtype=float)
    delta = weakest - strongest
    q = ((weakest + strongest - 2.0 * middle) / delta
         if delta > 1.0e-12 else 0.0)
    delta_error = (float(math.hypot(errors[0], errors[2]))
                   if np.all(np.isfinite(errors[[0, 2]])) else math.nan)
    if delta > 1.0e-12 and np.all(np.isfinite(errors)):
        numerator = weakest + strongest - 2.0 * middle
        denominator2 = delta * delta
        derivatives = np.asarray([
            (delta + numerator) / denominator2,
            -2.0 / delta,
            (delta - numerator) / denominator2,
        ])
        q_error = float(np.sqrt(np.sum((derivatives * errors) ** 2)))
    else:
        q_error = math.nan
    return {
        "J2": J2, "signed_h": signed_h, "branch": branch,
        "q": q, "q_error": q_error,
        "q_fraction": 0.5 * (1.0 - q),
        "strongest_extrapolated": strongest,
        "middle_extrapolated": middle,
        "weakest_extrapolated": weakest,
        "extrapolated_Delta": delta,
        "Delta_error": delta_error,
        "handling": handling,
        "n": len(fit["Ds"]),
        "Ds": " ".join(str(D) for D in fit["Ds"]),
        "fit_rms_max": float(np.max(fit["rms"])),
        "fit_max_abs_residual": fit["max_abs_residual"],
        "extrapolation_distance": fit["extrapolation_distance"],
    }


def selected_h0_fits(gapped_a: dict[float, float]) -> dict[tuple[float, str], dict]:
    selected_csv = HERE / "plots" / "selected_nn_data.csv"
    if not selected_csv.is_file():
        return {}
    with selected_csv.open(encoding="utf-8-sig", newline="") as stream:
        rows = [row for row in csv.DictReader(stream)
                if row.get("selected", "").lower() == "true"]
    fits: dict[tuple[float, str], dict] = {}
    keys = sorted({(float(row["J2"]), row["texture"]) for row in rows})
    for J2, texture in keys:
        if J2 not in gapped_a:
            continue
        subset = [row for row in rows
                  if close(float(row["J2"]), J2) and row["texture"] == texture]
        by_D: dict[int, list[np.ndarray]] = {}
        for row in subset:
            by_D.setdefault(int(row["D"]), []).append(np.asarray([
                float(row["strongest"]), float(row["middle"]),
                float(row["weakest"]),
            ]))
        averaged = {D: np.mean(values, axis=0) for D, values in by_D.items()}
        fit = fit_high_D_correlations(averaged, gapped_a[J2])
        if fit is not None:
            fits[(J2, texture)] = fit
    return fits


def build_texture_nodes(
    rows: list[FieldPoint], gapped_a: dict[float, float],
) -> tuple[list[dict], list[dict]]:
    nodes: list[dict] = []
    candidates: list[dict] = []
    keys = sorted({(row.J2, round(row.signed_h, 12)) for row in rows
                   if abs(row.signed_h) > 1.0e-12})
    for J2, signed_h in keys:
        if J2 not in gapped_a:
            continue
        subset = [row for row in rows if (
            close(row.J2, J2) and close(row.signed_h, signed_h)
        )]
        fit = fit_high_D_correlations(
            averaged_correlations_by_D(subset), gapped_a[J2],
        )
        if fit is None:
            continue
        node = correlation_node(
            J2, signed_h, subset[0].branch, fit,
            "per-D mean; endpoint-stable high-D fixed-a_g fits of three NN correlations",
        )
        nodes.append(node)
        candidates.append(node)

    # h=0: extrapolate all three ranks independently in both preparation
    # sectors, average corresponding thermodynamic ranks, then form Delta/q.
    component_fits = selected_h0_fits(gapped_a)
    all_J2 = sorted({row.J2 for row in rows})
    for J2 in all_J2:
        if J2 not in gapped_a:
            continue
        components: list[tuple[str, dict]] = []
        for branch in SCALAR_BRANCHES:
            fit = component_fits.get((J2, branch))
            if fit is None:
                subset = [row for row in rows if (
                    close(row.J2, J2) and abs(row.signed_h) <= 1.0e-12
                    and row.branch == branch
                )]
                fit = fit_high_D_correlations(
                    averaged_correlations_by_D(subset), gapped_a[J2],
                ) if subset else None
            if fit is not None:
                components.append((branch, fit))
                candidates.append(correlation_node(
                    J2, 0.0, branch, fit,
                    "h=0 component: selected NN data when available; "
                    "otherwise all-good branch data",
                ))
        if len(components) != 2:
            continue
        correlations = np.mean(
            np.asarray([fit["correlations"] for _, fit in components]), axis=0,
        )
        component_errors = np.asarray(
            [fit["errors"] for _, fit in components], dtype=float,
        )
        errors = np.sqrt(np.nansum(component_errors ** 2, axis=0)) / 2.0
        if not np.any(np.isfinite(component_errors)):
            errors[:] = math.nan
        combined = {
            "correlations": correlations,
            "errors": errors,
            "Ds": sorted({D for _, fit in components for D in fit["Ds"]}),
            "rms": np.mean(np.asarray([fit["rms"] for _, fit in components]),
                           axis=0),
            "max_abs_residual": max(
                fit["max_abs_residual"] for _, fit in components
            ),
            "extrapolation_distance": max(
                fit["extrapolation_distance"] for _, fit in components
            ),
        }
        nodes.append(correlation_node(
            J2, 0.0, "average of dimer/plaquette h=0 sectors", combined,
            "fit 2x3 h=0 correlations, then average corresponding ranks",
        ))
    return sorted(nodes, key=lambda row: (row["J2"], row["signed_h"])), candidates


def texture_rgb(q: np.ndarray, delta: np.ndarray, delta_max: float) -> np.ndarray:
    q = np.clip(q, -1.0, 1.0)
    strength = np.clip(delta / max(delta_max, 1.0e-12), 0.0, 1.0)
    red = np.asarray([0.92, 0.10, 0.10])
    purple = np.asarray([0.60, 0.16, 0.72])
    blue = np.asarray([0.10, 0.30, 1.00])
    negative = np.clip(-q, 0.0, 1.0)[..., None]
    positive = np.clip(q, 0.0, 1.0)[..., None]
    neutral = 1.0 - negative - positive
    base = negative * red + neutral * purple + positive * blue
    return base * strength[..., None]


def field_plot_coordinate(value: float | np.ndarray) -> float | np.ndarray:
    """Signed log-|h| coordinate with a quarter-decade gap around h=0."""
    values = np.asarray(value, dtype=float)
    magnitude = np.abs(values)
    result = np.zeros_like(values)
    small = (magnitude > 0.0) & (magnitude < 1.0e-3)
    result[small] = np.sign(values[small]) * 0.25 * magnitude[small] / 1.0e-3
    logarithmic = magnitude >= 1.0e-3
    result[logarithmic] = np.sign(values[logarithmic]) * (
        0.25 + np.log10(magnitude[logarithmic] / 1.0e-3)
    )
    return float(result) if result.ndim == 0 else result


def centered_edges(
    centers: np.ndarray, lower: float | None = None,
    upper: float | None = None,
) -> np.ndarray:
    centers = np.asarray(centers, dtype=float)
    middle = 0.5 * (centers[:-1] + centers[1:])
    first = (lower if lower is not None
             else centers[0] - 0.5 * (centers[1] - centers[0]))
    last = (upper if upper is not None
            else centers[-1] + 0.5 * (centers[-1] - centers[-2]))
    return np.concatenate(([first], middle, [last]))


def plot_selected_delta_comparison(
    texture_nodes: list[dict], output: Path,
) -> None:
    """Replot the selected-NN data as Delta and add the h=0 extrapolation."""
    selected_csv = HERE / "plots" / "selected_nn_data.csv"
    if not selected_csv.is_file():
        raise FileNotFoundError(
            f"selected NN table is missing: {selected_csv}"
        )
    with selected_csv.open(encoding="utf-8-sig", newline="") as stream:
        rows = [row for row in csv.DictReader(stream)
                if row.get("selected", "").lower() == "true"]

    textures = ("dimer-plaquette", "plaquette")
    titles = {
        "dimer-plaquette": "Dimer-plaquette sector",
        "plaquette": "Plaquette sector",
    }
    figure, axes = plt.subplots(
        1, 2, figsize=(16.0, 7.7), sharex=True, sharey=True,
        constrained_layout=True,
    )
    for axis, texture in zip(axes, textures):
        texture_rows = [row for row in rows if row["texture"] == texture]
        dimensions = sorted({int(row["D"]) for row in texture_rows})
        cmap = plt.get_cmap("YlOrRd" if texture == "dimer-plaquette" else "PuBu")
        for index, D in enumerate(dimensions):
            subset = sorted(
                [row for row in texture_rows if int(row["D"]) == D],
                key=lambda row: float(row["J2"]),
            )
            if not subset:
                continue
            fraction = (0.35 if len(dimensions) == 1 else
                        0.30 + 0.68 * index / (len(dimensions) - 1))
            alpha = (0.18 if len(dimensions) == 1 else
                     0.18 + 0.82 * (D - dimensions[0])
                     / max(1, dimensions[-1] - dimensions[0]))
            axis.errorbar(
                [float(row["J2"]) for row in subset],
                [float(row["delta"]) for row in subset],
                yerr=[float(row["delta_error"]) for row in subset],
                color=cmap(fraction), alpha=alpha, marker="o",
                markersize=5.2, linestyle="-", linewidth=1.05,
                elinewidth=0.7, capsize=1.8, label=f"D={D}",
            )

        maximum_selected_J2 = max(float(row["J2"]) for row in texture_rows)
        extrapolated = sorted(
            [row for row in texture_nodes
             if abs(float(row["signed_h"])) <= 1.0e-12
             and float(row["J2"]) <= maximum_selected_J2 + 1.0e-12],
            key=lambda row: float(row["J2"]),
        )
        axis.plot(
            [float(row["J2"]) for row in extrapolated],
            [float(row["extrapolated_Delta"]) for row in extrapolated],
            color="0.08", marker="D", markerfacecolor="white",
            markeredgewidth=1.2, markersize=5.5, linewidth=2.2,
            label=r"extrapolated $\Delta(J_2,h=0)$", zorder=10,
        )
        axis.set_title(titles[texture], fontsize=13)
        axis.set_xlabel(r"$J_2$", fontsize=12)
        axis.grid(alpha=0.18)
        axis.legend(loc="best", fontsize=8.0, frameon=False, ncol=2,
                    handlelength=1.7, columnspacing=0.8)
    axes[0].set_ylabel(
        r"$\Delta=C_{\rm weak}-C_{\rm strong}$", fontsize=12,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    plt.close(figure)


def plot_04_phase_diagram(
    crossings: list[dict], texture_nodes: list[dict], output: Path,
) -> None:
    valid_crossings = [row for row in crossings
                       if math.isfinite(row.get("h_c", math.nan))]
    h_values = np.asarray([row["signed_h"] for row in texture_nodes])
    j_values = np.asarray([row["J2"] for row in texture_nodes])
    q_values = np.asarray([row["q"] for row in texture_nodes])
    delta_values = np.asarray([
        row["extrapolated_Delta"] for row in texture_nodes
    ])

    # Evaluate only at the actually sampled rectangular grid and render each
    # value as one flat cell.  Nearest fill covers the few missing edge nodes;
    # it does not create a smooth interpolation between physical samples.
    h_centers = np.asarray(sorted(set(h_values)), dtype=float)
    x_centers = np.asarray(field_plot_coordinate(h_centers), dtype=float)
    j_centers = np.asarray(sorted(set(j_values)), dtype=float)
    H, J = np.meshgrid(x_centers, j_centers)
    points = np.column_stack((field_plot_coordinate(h_values), j_values))
    q_grid = griddata(points, q_values, (H, J), method="nearest")
    delta_grid = griddata(points, delta_values, (H, J), method="nearest")
    delta_max = math.ceil(float(np.nanmax(delta_values)) * 20.0) / 20.0
    rgb = texture_rgb(q_grid, delta_grid, delta_max)
    h_edges = centered_edges(
        x_centers, field_plot_coordinate(-1.0e-1),
        field_plot_coordinate(1.0e-1),
    )
    j_edges = centered_edges(j_centers)

    figure = plt.figure(figsize=(9.6, 7.2), constrained_layout=True)
    grid = figure.add_gridspec(1, 3, width_ratios=(12.0, 0.55, 0.55))
    axis = figure.add_subplot(grid[0])
    hue_axis = figure.add_subplot(grid[1])
    brightness_axis = figure.add_subplot(grid[2])
    cell_count = rgb.shape[0] * rgb.shape[1]
    cell_cmap = matplotlib.colors.ListedColormap(
        rgb.reshape(cell_count, 3), name="texture_cells",
    )
    cell_norm = matplotlib.colors.BoundaryNorm(
        np.arange(cell_count + 1) - 0.5, cell_count,
    )
    axis.pcolormesh(
        h_edges, j_edges, np.arange(cell_count).reshape(rgb.shape[:2]),
        cmap=cell_cmap, norm=cell_norm, shading="flat",
        antialiased=False, rasterized=True, zorder=0,
    )
    ordered = sorted(valid_crossings, key=lambda row: row["J2"])
    hc = [row["h_c"] for row in ordered]
    hc_x = [field_plot_coordinate(value) for value in hc]
    J2 = [row["J2"] for row in ordered]
    errors = [row["error95"] for row in ordered]
    boundary, = axis.plot(
        hc_x, J2, color="#fff176", linewidth=2.2, marker="o",
        markersize=4.8, zorder=5,
        label=r"extrapolated energy crossing $h_c$ (95% error)",
    )
    boundary.set_path_effects([
        path_effects.Stroke(linewidth=4.0, foreground="black"),
        path_effects.Normal(),
    ])
    for value, J2_value, error in zip(hc, J2, errors):
        if not math.isfinite(error):
            continue
        left = field_plot_coordinate(value - error)
        right = field_plot_coordinate(value + error)
        axis.plot([left, right], [J2_value, J2_value], color="#fff176",
                  linewidth=1.4, zorder=4)
        axis.plot([left, left], [J2_value - 0.0006, J2_value + 0.0006],
                  color="#fff176", linewidth=1.2, zorder=4)
        axis.plot([right, right], [J2_value - 0.0006, J2_value + 0.0006],
                  color="#fff176", linewidth=1.2, zorder=4)
    qsl, = axis.plot(
        [0.0, 0.0], [0.24, 0.275], color="#00d000", linewidth=7.0,
        solid_capstyle="butt", zorder=2,
        label=r"QSL: $h=0$, $0.24\leq J_2\leq0.275$",
    )
    axis.text(
        field_plot_coordinate(-2.5e-2), 0.334, "dimer-plaquette",
        color="white", fontsize=11, fontweight="semibold",
        ha="center", va="center", zorder=3,
    )
    axis.text(
        field_plot_coordinate(2.5e-2), 0.334, "plaquette",
        color="white", fontsize=11, fontweight="semibold",
        ha="center", va="center", zorder=3,
    )
    ticks_h = np.asarray([-1.0e-1, -1.0e-2, -1.0e-3, 0.0,
                          1.0e-3, 1.0e-2, 1.0e-1])
    axis.set_xticks(field_plot_coordinate(ticks_h))
    axis.set_xticklabels([
        r"$-10^{-1}$", r"$-10^{-2}$", r"$-10^{-3}$", "$0$",
        r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$",
    ])
    axis.set_xlim(field_plot_coordinate(-1.0e-1),
                  field_plot_coordinate(1.0e-1))
    axis.set_ylim(0.238, 0.343)
    axis.set_xlabel(r"signed pinning field $h$")
    axis.set_ylabel(r"$J_2$")
    axis.grid(axis="y", color="white", alpha=0.22, linewidth=0.7)
    axis.legend(loc="lower right", framealpha=0.88, fontsize=8.5)

    hue_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "dimer_purple_plaquette",
        [(0.92, 0.10, 0.10), (0.60, 0.16, 0.72), (0.10, 0.30, 1.00)],
    )
    hue_bar = matplotlib.colorbar.ColorbarBase(
        hue_axis, cmap=hue_cmap,
        norm=matplotlib.colors.Normalize(-1.0, 1.0),
        orientation="vertical", ticks=[-1.0, 0.0, 1.0],
    )
    hue_bar.set_ticklabels([
        r"$-1$", "$0$", r"$+1$",
    ])
    hue_bar.set_label(
        r"extrapolated $q=(C_{\rm weak}+C_{\rm strong}-2C_{\rm mid})/"
        r"(C_{\rm weak}-C_{\rm strong})$"
    )
    brightness_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "delta_brightness", ["black", "white"],
    )
    brightness_bar = matplotlib.colorbar.ColorbarBase(
        brightness_axis, cmap=brightness_cmap,
        norm=matplotlib.colors.Normalize(vmin=0.0, vmax=delta_max),
        orientation="vertical",
        ticks=np.linspace(0.0, delta_max, 4),
    )
    brightness_bar.set_label(
        r"extrapolated $\Delta=C_{\rm weak}-C_{\rm strong}$"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    plt.close(figure)


def write_dict_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as stream:
        if not fields:
            return
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--make-video", action="store_true",
        help="also render the short 180-degree z-axis rotation of figure 03",
    )
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    points, failures, duplicates, rank_split = discover_points()
    set_a, set_b = construct_sets(points)
    original_energies = load_original_energies()
    representatives, decisions = choose_representatives(set_b, original_energies)
    all_good_points, all_good_audit = screen_normal_energy_curves(
        set_b, original_energies,
    )
    gapped_a = load_gapped_a()

    results: list[CrossingResult] = []
    fit_audit: list[dict] = []
    per_D: list[dict] = []
    for J2 in sorted({row.J2 for row in set_b}):
        if J2 not in gapped_a:
            results.append(CrossingResult(
                J2, False, "missing figure-24 a_g", math.nan, math.nan,
                math.nan, math.nan, math.nan, math.nan, 0, 0, 0, 0, 0, 0,
                math.nan, math.nan, "", 0, math.nan,
            ))
            continue
        subset = [row for row in representatives if close(row.J2, J2)]
        result, audit, local = fit_crossing_for_j2(J2, subset, gapped_a[J2])
        results.append(result)
        fit_audit.extend(audit)
        per_D.extend(local)

    all_good_crossings: list[dict] = []
    all_good_crossing_audit: list[dict] = []
    all_good_local_crossings: list[dict] = []
    for J2 in sorted({row.J2 for row in all_good_points}):
        if J2 not in gapped_a:
            continue
        subset = [row for row in all_good_points if close(row.J2, J2)]
        crossing, audit, local = fit_all_good_crossing(
            J2, subset, gapped_a[J2],
        )
        all_good_crossings.append(crossing)
        all_good_crossing_audit.extend(audit)
        all_good_local_crossings.extend(local)
    texture_nodes, texture_candidates = build_texture_nodes(
        all_good_points, gapped_a,
    )

    write_dict_csv(output / "set_A.csv", set_a)
    write_dict_csv(output / "set_B_all_observations.csv",
                   [asdict(row) for row in set_b])
    write_dict_csv(output / "all_good_points.csv",
                   [asdict(row) for row in all_good_points])
    write_dict_csv(output / "all_good_points_audit.csv", all_good_audit)
    write_dict_csv(output / "representative_selection_audit.csv", [
        {**asdict(item.point), "selected_for_fit": item.selected,
         "selection_reason": item.reason} for item in decisions
    ])
    write_dict_csv(output / "robust_fit_point_audit.csv", fit_audit)
    write_dict_csv(output / "finite_D_crossing_diagnostics.csv", per_D)
    write_dict_csv(output / "hc_D_infinity.csv", [asdict(row) for row in results])
    write_dict_csv(output / "hc_03_all_good_D_infinity.csv",
                   all_good_crossings)
    write_dict_csv(output / "hc_03_all_good_fit_point_audit.csv",
                   all_good_crossing_audit)
    write_dict_csv(output / "hc_03_all_good_finite_D_crossings.csv",
                   all_good_local_crossings)
    write_dict_csv(output / "texture_D_infinity_nodes.csv", texture_nodes)
    write_dict_csv(output / "texture_D_infinity_candidates.csv",
                   texture_candidates)
    write_dict_csv(output / "discovery_failures.csv",
                   [{"failure": failure} for failure in failures], ["failure"])

    plot_energy_surface(
        set_b, representatives, output / "01_signed_h_energy_surface.pdf",
    )
    plot_energy_surface(
        representatives, representatives,
        output / "01b_fit_representative_energy_surface.pdf",
        title=(r"Fit representatives after physical/numerical screening: "
               r"plaquette $h>0$, dimer-plaquette $h<0$"),
    )
    diagnostics = output / "fit_diagnostics"
    for result in results:
        if result.J2 not in gapped_a:
            continue
        subset = [row for row in representatives if close(row.J2, result.J2)]
        token = f"{result.J2:.6f}".rstrip("0").rstrip(".").replace(".", "p")
        plot_fit_diagnostic(
            result, subset, gapped_a[result.J2],
            diagnostics / f"J2_{token}.pdf",
        )
    plot_phase_boundary(results, output / "02_hc_vs_J2_phase_boundary.pdf")
    plot_all_good_points(
        all_good_points, original_energies,
        output / "03_all_good_points.pdf",
        (output / "03_all_good_points_z_rotation.mp4"
         if args.make_video else None),
    )
    plot_04_phase_diagram(
        all_good_crossings, texture_nodes,
        output / "04_phasediagram.pdf",
    )
    plot_selected_delta_comparison(
        texture_nodes, output.parent / "Delta_vs_J2_selected.pdf",
    )

    accepted = sum(row.accepted for row in results)
    print(f"Discovered scalar pinning observations: {len(points)}")
    print(f"Deduplicated copied observations: {duplicates}")
    print(f"Excluded non-scalar rank-split observations: {rank_split}")
    print(f"Set A: {len(set_a)} (J2,D) pairs")
    print(f"Set B: {len(set_b)} observations")
    print(f"All non-obvious-bad B points: {len(all_good_points)}")
    print(f"D->infinity crossings: {accepted}/{len(results)} accepted")
    zero_covered = sum(bool(row.get("contains_h0_95"))
                       for row in all_good_crossings)
    print("03 full-data 95% crossings containing h=0: "
          f"{zero_covered}/{len(all_good_crossings)}")
    print(f"Output: {output}")
    if failures:
        print(f"Discovery failures recorded: {len(failures)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
