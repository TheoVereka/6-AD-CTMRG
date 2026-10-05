#!/usr/bin/env python3
"""Signed-pinning-field energy surface and D->infinity crossing line.

Plaquette pinning is assigned signed h>0 and dimer-plaquette pinning signed
h<0.  Nonzero rank-split fields are a different direction in order-parameter
space and are deliberately excluded.  Set A contains (J2,D) pairs with at
least two distinct positive and two distinct negative fields.  Set B contains
every discovered scalar-field observation for the pairs in A, including h=0.

Figure 03 shows all physically screened B points.  For figures 02/04 and the
crossing-energy comparison, reruns at fixed (J2,D,h) are reduced first.  At
each (J2,D,branch), energy is fitted quadratically in h; a visibly inconsistent
h=0 point is omitted by an explicit leave-zero-out test.  The two fitted
finite-D branches give h_c,D and E_crossing,D, which are then extrapolated in D
in two different ways: E_crossing,D enters an error-weighted fixed-a_g gapped
fit, while h_c,D is statistically combined using its uncertainty and its
energy distance from the extrapolated crossing energy.  Each of the three NN
correlations is
separately extrapolated with the same fixed-a_g construction; Delta and q are
calculated only from the three extrapolated correlations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

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
sys.path.insert(0, str(HERE))
from analyze_branch_runs import find_lookahead, read_scalar_hyperparams  # noqa: E402
from analyze_existing_twoc3 import parse_observation  # noqa: E402
import select_and_plot as selected_story  # noqa: E402


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
    ("kuma_targeted_repairs", DATA / "distinVBCsKumaTargetedRepairs"
     / "Results_Kuma_TargetedRepairs"),
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
            # r03 is the rejected J2=.32, D=11 plaquette repair.  It is not
            # part of either the selected h=0 story or the energy surface.
            if label == "kuma_targeted_repairs" and "r03" in path.parts:
                continue
            # A copied observation can exist while the cluster process is
            # still writing the stage.  Targeted repairs become visible to
            # every downstream fit only after their atomic completion marker.
            if (label == "kuma_targeted_repairs"
                    and not (path.parent / "COMPLETED.stage").is_file()):
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


def load_gapped_a_errors() -> dict[float, float]:
    values: dict[float, float] = {}
    with ENERGY_FITS.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if (row.get("ansatz") == "2C3" and row.get("model") == "gapped"
                    and row.get("parameter") == "a"):
                values[float(row["J2"])] = abs(float(row["error"]))
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


def plot_two_stage_energy_diagnostic(
    crossing: dict, nodes: list[dict], finite_D: list[dict], output: Path,
) -> None:
    """Show finite-D h fits, their crossings, and the combined h_c."""
    figure, axis = plt.subplots(figsize=(6.8, 5.0), constrained_layout=True)
    styles = {
        "dimer-plaquette": ("#d62728", "o", "dimer-plaquette"),
        "plaquette": ("#1f77b4", "s", "plaquette"),
    }
    for branch in SCALAR_BRANCHES:
        color, marker, label = styles[branch]
        rows = sorted([row for row in nodes if row["branch"] == branch],
                      key=lambda row: row["D"])
        if rows:
            D_min = min(int(row["D"]) for row in rows)
            D_max = max(int(row["D"]) for row in rows)
        for row in rows:
            D = int(row["D"])
            alpha = (0.22 if D_max == D_min else
                     0.16 + 0.34 * (D - D_min) / (D_max - D_min))
            h_bound = min(float(row["h_max"]), ENERGY_DIAGNOSTIC_H_LIMIT)
            if branch == "dimer-plaquette":
                fields = np.linspace(-h_bound, 0.0, 100)
            else:
                fields = np.linspace(0.0, h_bound, 100)
            energy = (float(row["E0"]) + float(row["E1"]) * fields
                      + float(row["E2"]) * fields ** 2)
            axis.plot(fields, energy, color=color, alpha=alpha,
                      linewidth=0.9)
            raw_h = np.asarray([
                float(value) for value in str(row["h_values"]).split()
            ])
            raw_E = np.asarray([
                float(value) for value in str(row["E_values"]).split()
            ])
            near = np.abs(raw_h) <= ENERGY_DIAGNOSTIC_H_LIMIT + 1.0e-12
            axis.scatter(raw_h[near], raw_E[near], marker=marker, s=12,
                         color=color, alpha=alpha, linewidths=0)
        axis.plot([], [], color=color, linewidth=2.0, label=label)
    if finite_D:
        D_min = min(int(row["D"]) for row in finite_D)
        D_max = max(int(row["D"]) for row in finite_D)
        for row in finite_D:
            D = int(row["D"])
            alpha = (0.65 if D_max == D_min else
                     0.35 + 0.55 * (D - D_min) / (D_max - D_min))
            axis.errorbar(
                float(row["h_c_D"]), float(row["E_crossing_D"]),
                xerr=float(row["h_c_D_sigma"]),
                yerr=float(row["E_crossing_D_sigma"]),
                linestyle="none", marker="x", markersize=5.0,
                color="black", alpha=alpha, capsize=1.5, zorder=5,
            )
    h_c = float(crossing.get("h_c", math.nan))
    if math.isfinite(h_c):
        axis.axvline(h_c, color="#8a5a00", linewidth=1.5, linestyle="--",
                     label=r"$h_c$")
    axis.set_xlim(-1.08 * ENERGY_DIAGNOSTIC_H_LIMIT,
                  1.08 * ENERGY_DIAGNOSTIC_H_LIMIT)
    axis.set_xlabel(r"signed pinning field $h$")
    axis.set_ylabel("energy per site")
    axis.set_title(
        rf"$J_2={float(crossing['J2']):g}$: cross at each $D$, then combine"
    )
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, fontsize=8.5)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    plt.close(figure)


def plot_two_stage_phase_boundary(crossings: list[dict], output: Path) -> None:
    valid = [row for row in sorted(crossings, key=lambda row: row["J2"])
             if math.isfinite(row.get("h_c", math.nan))]
    figure, axis = plt.subplots(figsize=(6.8, 6.0), constrained_layout=True)
    if valid:
        h = np.asarray([row["h_c"] for row in valid], dtype=float)
        J2 = np.asarray([row["J2"] for row in valid], dtype=float)
        lower = np.asarray([
            row.get("h_c_error_low", row.get("error_1sigma", 0.0)) for row in valid
        ], dtype=float)
        upper = np.asarray([
            row.get("h_c_error_high", row.get("error_1sigma", 0.0)) for row in valid
        ], dtype=float)
        axis.plot(h, J2, color="#6b2f8a", linewidth=1.8, zorder=3)
        axis.errorbar(
            h, J2, xerr=np.vstack((lower, upper)), linestyle="none",
            marker="o", markersize=6.0, color="#6b2f8a", capsize=2.5,
            zorder=4, label=r"statistical $h_c$",
        )
    axis.plot(
        [0.0, 0.0], [0.24, 0.275], color="#00d000", linewidth=7.0,
        solid_capstyle="butt", zorder=2,
        label=r"$h=0$, $0.24\leq J_2\leq0.275$",
    )
    all_h = [0.0] + [float(row["h_c"]) for row in valid]
    all_errors = [0.0] + [max(
        float(row.get("h_c_error_low", 0.0)),
        float(row.get("h_c_error_high", 0.0)),
    ) for row in valid]
    limit = max(0.001, 1.2 * max(
        abs(value) + error for value, error in zip(all_h, all_errors)
    ))
    axis.set_xlim(-limit, limit)
    axis.set_ylim(0.235, 0.345)
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
    q16, q84 = np.quantile(roots, [0.15865525393145707,
                                   0.8413447460685429])
    sigma = float(np.std(roots, ddof=1))
    return sigma, float(q16), float(q84)


ENERGY_H_FIT_LIMIT = 8.0e-2
ENERGY_DIAGNOSTIC_H_LIMIT = 5.0e-3
ENERGY_MIN_D = 6
ENERGY_H0_ABS_TOL = 3.0e-4
ENERGY_H0_RMS_FACTOR = 6.0
ENERGY_MC_SAMPLES = 12000
ENERGY_H_MIXTURE_SAMPLES = 100000


def _quadratic_h_fit(fields: np.ndarray, energies: np.ndarray) -> tuple[
    np.ndarray, np.ndarray, np.ndarray
]:
    """Equal-h OLS in a scaled coordinate, returned in physical h units."""
    h_scale = float(np.max(np.abs(fields)))
    if h_scale <= 0.0:
        raise ValueError("quadratic h fit has no nonzero field")
    scaled = fields / h_scale
    design = np.column_stack((np.ones(len(fields)), scaled, scaled ** 2))
    if len(fields) < 3 or np.linalg.matrix_rank(design) < 3:
        raise ValueError("quadratic h fit is rank deficient")
    beta_scaled, *_ = np.linalg.lstsq(design, energies, rcond=None)
    residual = energies - design @ beta_scaled
    inverse = np.linalg.pinv(design.T @ design)
    dof = max(1, len(fields) - 3)
    covariance_scaled = float(np.sum(residual ** 2) / dof) * inverse
    transform = np.diag([1.0, 1.0 / h_scale, 1.0 / h_scale ** 2])
    return (transform @ beta_scaled,
            positive_semidefinite(transform @ covariance_scaled @ transform),
            residual)


def fit_energy_h_at_D(
    J2: float, D: int, branch: str, rows: list[FieldPoint],
) -> dict | None:
    """First stage: median per h, then the smallest supported near-zero fit."""
    by_h: dict[float, list[float]] = {}
    for row in rows:
        if (row.D == D and row.branch == branch
                and abs(row.signed_h) <= ENERGY_H_FIT_LIMIT + 1.0e-12):
            by_h.setdefault(round(row.signed_h, 12), []).append(row.energy)
    all_fields = np.asarray(sorted(by_h), dtype=float)
    if len(all_fields) < 3 or not np.any(np.abs(all_fields) > 1.0e-14):
        return None
    all_energies = np.asarray([
        float(np.median(by_h[float(field)])) for field in all_fields
    ], dtype=float)
    # Four distinct fields give one residual degree of freedom.  Use the
    # smallest near-zero window that supports this; fall back to three fields
    # only when no four-point window exists.
    chosen = None
    for minimum_count in (4, 3):
        for window in (0.005, 0.01, 0.02, 0.04, ENERGY_H_FIT_LIMIT):
            mask = np.abs(all_fields) <= window + 1.0e-12
            candidate = all_fields[mask]
            if (len(candidate) >= minimum_count
                    and np.linalg.matrix_rank(np.column_stack((
                        np.ones(len(candidate)), candidate, candidate ** 2,
                    ))) == 3):
                chosen = mask
                break
        if chosen is not None:
            break
    if chosen is None:
        return None
    fields = all_fields[chosen]
    energies = all_energies[chosen]

    excluded_h0 = False
    h0_residual = math.nan
    h0_threshold = math.nan
    zero_indices = np.flatnonzero(np.abs(fields) <= 1.0e-14)
    nonzero = np.abs(fields) > 1.0e-14
    if len(zero_indices) == 1 and int(np.count_nonzero(nonzero)) >= 4:
        try:
            nonzero_beta, _, nonzero_residual = _quadratic_h_fit(
                fields[nonzero], energies[nonzero],
            )
            h0_residual = abs(float(
                energies[zero_indices[0]] - nonzero_beta[0]
            ))
            nonzero_rms = float(np.sqrt(np.mean(nonzero_residual ** 2)))
            h0_threshold = max(
                ENERGY_H0_ABS_TOL, ENERGY_H0_RMS_FACTOR * nonzero_rms,
            )
            excluded_h0 = h0_residual > h0_threshold
        except ValueError:
            pass
    used = nonzero if excluded_h0 else np.ones(len(fields), dtype=bool)
    try:
        beta, covariance, residual = _quadratic_h_fit(
            fields[used], energies[used],
        )
    except ValueError:
        return None
    h_max = float(np.max(np.abs(fields[used])))
    return {
        "J2": J2, "D": D, "branch": branch,
        "E0": float(beta[0]), "E1": float(beta[1]),
        "E2": float(beta[2]),
        "cov00": float(covariance[0, 0]),
        "cov01": float(covariance[0, 1]),
        "cov02": float(covariance[0, 2]),
        "cov11": float(covariance[1, 1]),
        "cov12": float(covariance[1, 2]),
        "cov22": float(covariance[2, 2]),
        "h_max": h_max,
        "n_h": int(np.count_nonzero(used)),
        "h_values": " ".join(f"{value:g}" for value in fields[used]),
        "E_values": " ".join(f"{value:.16g}" for value in energies[used]),
        "h_fit_rms": float(np.sqrt(np.mean(residual ** 2))),
        "h0_excluded": excluded_h0,
        "h0_residual": h0_residual, "h0_threshold": h0_threshold,
        "n_raw": sum(len(by_h[float(field)]) for field in fields),
    }


def build_energy_h_fits(J2: float, rows: list[FieldPoint]) -> list[dict]:
    nodes = []
    dimensions = sorted({row.D for row in rows if row.D >= ENERGY_MIN_D})
    for branch in SCALAR_BRANCHES:
        for D in dimensions:
            fit = fit_energy_h_at_D(J2, D, branch, rows)
            if fit is not None:
                nodes.append(fit)
    return nodes


def _energy_node_beta(node: dict) -> np.ndarray:
    return np.asarray([node["E0"], node["E1"], node["E2"]], dtype=float)


def _energy_node_covariance(node: dict) -> np.ndarray:
    return np.asarray([
        [node["cov00"], node["cov01"], node["cov02"]],
        [node["cov01"], node["cov11"], node["cov12"]],
        [node["cov02"], node["cov12"], node["cov22"]],
    ], dtype=float)


def positive_semidefinite(covariance: np.ndarray) -> np.ndarray:
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    return eigenvectors @ np.diag(np.maximum(eigenvalues, 0.0)) @ eigenvectors.T


def build_finite_D_energy_crossings(
    J2: float, nodes: list[dict],
) -> list[dict]:
    """Cross the two fitted h branches independently at every D."""
    output = []
    for D in sorted({int(node["D"]) for node in nodes}):
        by_branch = {
            branch: next((node for node in nodes if (
                int(node["D"]) == D and node["branch"] == branch
            )), None)
            for branch in SCALAR_BRANCHES
        }
        if any(node is None for node in by_branch.values()):
            continue
        plaquette = by_branch["plaquette"]
        dimer = by_branch["dimer-plaquette"]
        assert plaquette is not None and dimer is not None
        p_beta = _energy_node_beta(plaquette)
        d_beta = _energy_node_beta(dimer)
        h_c_D = crossing_from_polynomials(p_beta, d_beta)
        if not math.isfinite(h_c_D):
            continue
        E_p = float(sum(p_beta[power] * h_c_D ** power
                        for power in range(3)))
        E_d = float(sum(d_beta[power] * h_c_D ** power
                        for power in range(3)))
        E_crossing_D = 0.5 * (E_p + E_d)

        rng = np.random.default_rng(
            510000 + int(round(1000 * J2)) * 100 + D
        )
        p_samples = rng.multivariate_normal(
            p_beta, positive_semidefinite(_energy_node_covariance(plaquette)),
            size=4000, check_valid="ignore",
        )
        d_samples = rng.multivariate_normal(
            d_beta, positive_semidefinite(_energy_node_covariance(dimer)),
            size=4000, check_valid="ignore",
        )
        samples = []
        for sampled_p, sampled_d in zip(p_samples, d_samples):
            root = crossing_from_polynomials(sampled_p, sampled_d)
            if not math.isfinite(root):
                continue
            sampled_E_p = float(sum(
                sampled_p[power] * root ** power for power in range(3)
            ))
            sampled_E_d = float(sum(
                sampled_d[power] * root ** power for power in range(3)
            ))
            samples.append((root, 0.5 * (sampled_E_p + sampled_E_d)))
        if len(samples) >= 500:
            local_covariance = positive_semidefinite(
                np.cov(np.asarray(samples, dtype=float), rowvar=False, ddof=1)
            )
        else:
            local_covariance = np.zeros((2, 2), dtype=float)
        h_max = max(float(plaquette["h_max"]), float(dimer["h_max"]))
        output.append({
            "J2": J2, "D": D,
            "h_c_D": h_c_D, "E_crossing_D": E_crossing_D,
            "h_c_D_sigma": math.sqrt(max(0.0, local_covariance[0, 0])),
            "E_crossing_D_sigma": math.sqrt(
                max(0.0, local_covariance[1, 1])
            ),
            "hE_covariance": float(local_covariance[0, 1]),
            "h_max_D": h_max,
            "plaquette_h_max": float(plaquette["h_max"]),
            "dimer_h_max": float(dimer["h_max"]),
            "plaquette_h0_excluded": bool(plaquette["h0_excluded"]),
            "dimer_h0_excluded": bool(dimer["h0_excluded"]),
            "branch_energy_mismatch": abs(E_p - E_d),
            "mc_accepted": len(samples),
        })
    return output


def fit_crossing_observables_vs_D(
    rows: list[dict], a_g: float, a_g_error: float,
) -> dict | None:
    """Error-weight E_c(D) fit, then energy-aware h_c statistics."""
    selected = sorted(rows, key=lambda row: row["D"])
    if len(selected) < 3:
        return None
    dimensions = np.asarray([row["D"] for row in selected], dtype=float)
    energies = np.asarray([
        row["E_crossing_D"] for row in selected
    ], dtype=float)
    energy_errors = np.asarray([
        row["E_crossing_D_sigma"] for row in selected
    ], dtype=float)
    positive_E_errors = energy_errors[
        np.isfinite(energy_errors) & (energy_errors > 0.0)
    ]
    E_error_floor = (0.1 * float(np.min(positive_E_errors))
                     if len(positive_E_errors) else 1.0e-10)
    energy_errors = np.where(
        np.isfinite(energy_errors) & (energy_errors > 0.0),
        energy_errors, E_error_floor,
    )
    objective_weights = 1.0 / energy_errors ** 2

    def solve_energy(exponent: float) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, float
    ]:
        design = np.column_stack((
            np.ones(len(dimensions)), np.exp(-exponent * dimensions),
        ))
        inverse = np.linalg.pinv(
            design.T @ (objective_weights[:, None] * design)
        )
        mapping = inverse @ design.T @ np.diag(objective_weights)
        parameters = mapping @ energies
        residual = energies - design @ parameters
        dof = max(1, len(selected) - 2)
        chi2_reduced = float(
            np.sum((residual / energy_errors) ** 2) / dof
        )
        covariance = inverse * max(1.0, chi2_reduced)
        return parameters, covariance, residual, chi2_reduced

    parameters, covariance, residual, chi2_reduced = solve_energy(a_g)
    a_systematic = 0.0
    if math.isfinite(a_g_error) and a_g_error > 0.0:
        lower, _, _, _ = solve_energy(max(1.0e-8, a_g - a_g_error))
        upper, _, _, _ = solve_energy(a_g + a_g_error)
        a_systematic = 0.5 * float(upper[0] - lower[0])
    E_crossing = float(parameters[0])
    E_sigma = math.hypot(
        math.sqrt(max(0.0, float(covariance[0, 0]))), a_systematic,
    )

    h_values = np.asarray([row["h_c_D"] for row in selected], dtype=float)
    h_errors = np.asarray([row["h_c_D_sigma"] for row in selected], dtype=float)
    positive_h_errors = h_errors[
        np.isfinite(h_errors) & (h_errors > 0.0)
    ]
    h_error_floor = (0.1 * float(np.min(positive_h_errors))
                     if len(positive_h_errors) else 1.0e-10)
    h_errors = np.where(
        np.isfinite(h_errors) & (h_errors > 0.0), h_errors, h_error_floor,
    )
    energy_distance = np.abs(energies - E_crossing)
    distance_floor = max(1.0e-12, 1.0e-6 * E_sigma)
    raw_h_weights = 1.0 / (
        h_errors * np.maximum(energy_distance, distance_floor)
    )
    probabilities = raw_h_weights / np.sum(raw_h_weights)
    h_c = float(np.sum(probabilities * h_values))

    rng = np.random.default_rng(
        610000 + int(round(1000.0 * float(selected[0]["J2"])))
    )
    choices = rng.choice(
        len(selected), size=ENERGY_H_MIXTURE_SAMPLES, p=probabilities,
    )
    h_samples = rng.normal(h_values[choices], h_errors[choices])
    h_q16, h_q84 = map(float, np.quantile(
        h_samples, [0.15865525393145707, 0.8413447460685429]
    ))
    for index, row in enumerate(selected):
        row["E_distance_to_extrapolated"] = float(energy_distance[index])
        row["h_weight_raw"] = float(raw_h_weights[index])
        row["h_weight_normalized"] = float(probabilities[index])
    return {
        "h_c": h_c, "h_c_q16": h_q16, "h_c_q84": h_q84,
        "E_crossing": E_crossing, "E_crossing_sigma": E_sigma,
        "n_D": len(selected),
        "Ds": " ".join(str(int(value)) for value in dimensions),
        "h_weights": " ".join(
            f"{value:.8g}" for value in probabilities
        ),
        "E_crossing_D_fit_rms": float(np.sqrt(np.mean(residual ** 2))),
        "E_crossing_reduced_chi2": chi2_reduced,
        "a_g": a_g, "a_g_error": a_g_error,
        "a_g_systematic_E_crossing": a_systematic,
    }


def fit_all_good_crossing(
    J2: float, rows: list[FieldPoint], a_g: float, a_g_error: float,
) -> tuple[dict, list[dict], list[dict]]:
    """Cross at each D, extrapolate E_c, then statistically combine h_c,D."""
    nodes = build_energy_h_fits(J2, rows)
    finite_D = build_finite_D_energy_crossings(J2, nodes)
    extrapolation = fit_crossing_observables_vs_D(
        finite_D, a_g, a_g_error,
    )
    if extrapolation is None:
        return ({
            "J2": J2, "h_c": math.nan, "error_1sigma": math.nan,
            "contains_h0_1sigma": False,
            "reason": "fewer than three finite-D energy crossings",
        }, nodes, finite_D)
    h_c = float(extrapolation["h_c"])
    h_q16 = float(extrapolation["h_c_q16"])
    h_q84 = float(extrapolation["h_c_q84"])
    h_error_low = max(0.0, h_c - h_q16)
    h_error_high = max(0.0, h_q84 - h_c)
    h_sigma = 0.5 * (h_error_low + h_error_high)
    E_crossing = float(extrapolation["E_crossing"])
    E_sigma = float(extrapolation["E_crossing_sigma"])
    E_q16, E_q84 = E_crossing - E_sigma, E_crossing + E_sigma
    result = {
        "J2": J2, "h_c": h_c, "error_1sigma": h_sigma,
        "h_c_error_low": h_error_low, "h_c_error_high": h_error_high,
        "h_c_q16": h_q16, "h_c_q84": h_q84,
        "contains_h0_1sigma": bool(
            math.isfinite(h_q16) and h_q16 <= 0.0 <= h_q84
        ),
        "conditional_stat_sigma": h_sigma,
        "conditional_q16": h_q16, "conditional_q84": h_q84,
        "E_crossing": E_crossing, "E_crossing_sigma": E_sigma,
        "E_crossing_error_low": E_sigma,
        "E_crossing_error_high": E_sigma,
        "E_crossing_q16": E_q16, "E_crossing_q84": E_q84,
        "finite_D_crossing_std": float(np.std(
            [row["h_c_D"] for row in finite_D], ddof=1
        )),
        "finite_D_crossing_count": len(finite_D),
        "Ds": extrapolation["Ds"],
        "h_weights": extrapolation["h_weights"],
        "E_crossing_D_fit_rms": extrapolation["E_crossing_D_fit_rms"],
        "E_crossing_reduced_chi2": extrapolation[
            "E_crossing_reduced_chi2"
        ],
        "a_g": extrapolation["a_g"],
        "a_g_error": extrapolation["a_g_error"],
        "a_g_systematic_E_crossing": extrapolation[
            "a_g_systematic_E_crossing"
        ],
        "reason": ("median per h; quadratic branch fits and energy crossing "
                   "at each D; optional h=0 outlier removal; E_crossing,D "
                   "uses an error-weighted fixed-a_g fit; h_c is an "
                   "energy-proximity/error weighted Gaussian mixture"),
    }
    return result, nodes, finite_D


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
    a_g_error: float = 0.0,
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
    delta = float(correlations[2] - correlations[0])
    symmetric_delta_error = (
        float(math.hypot(errors[ordering][0], errors[ordering][2]))
        if np.all(np.isfinite(errors[ordering][[0, 2]])) else math.nan
    )
    if len(selected) == 2:
        # A two-point exponential has no residual degree of freedom.  Its
        # symmetric model error is the distance from the extrapolate to the
        # largest-D splitting, combined in quadrature with the nonlinear
        # one-sigma propagation of the published uncertainty of a_g.  This is
        # the two-point-only analogue of comparing with an adjacent three-D
        # window when such a window exists.
        largest_D_delta = float(values[-1, 2] - values[-1, 0])
        endpoint_systematic = abs(delta - largest_D_delta)
        a_g_systematic = 0.0
        if math.isfinite(a_g_error) and a_g_error > 0.0:
            for trial_a in (max(1.0e-8, a_g - a_g_error),
                            a_g + a_g_error):
                trial_x = np.exp(-trial_a * np.asarray(selected, dtype=float))
                trial_matrix = np.column_stack((
                    np.ones(len(selected)), trial_x,
                ))
                trial_beta, *_ = np.linalg.lstsq(
                    trial_matrix, values, rcond=None,
                )
                trial_correlations = np.sort(trial_beta[0])
                trial_delta = float(
                    trial_correlations[2] - trial_correlations[0]
                )
                a_g_systematic = max(
                    a_g_systematic, abs(trial_delta - delta),
                )
        symmetric_delta_error = float(math.hypot(
            endpoint_systematic, a_g_systematic,
        ))
    largest_D_values = np.asarray(by_D[max(selected)], dtype=float)
    return {
        "Ds": list(selected),
        "correlations": correlations,
        "errors": errors[ordering],
        "Delta_error": symmetric_delta_error,
        "rms": np.sqrt(np.mean(residuals ** 2, axis=0)),
        "max_abs_residual": float(np.max(np.abs(residuals))),
        "extrapolation_distance": float(np.max(np.abs(
            correlations - largest_D_values
        ))),
    }


def candidate_high_D_correlation_fits(
    by_D: dict[int, np.ndarray], a_g: float, a_g_error: float = 0.0,
) -> list[dict]:
    """All contiguous high-D windows, from the two largest D values onward."""
    dimensions = sorted(by_D)
    return [
        fit_correlation_window(
            by_D, a_g, dimensions[-count:], a_g_error,
        )
        for count in range(2, len(dimensions) + 1)
    ] if len(dimensions) >= 2 else []


def fit_high_D_correlations(
    by_D: dict[int, np.ndarray], a_g: float, a_g_error: float = 0.0,
) -> dict | None:
    """Use the common high-D suffix with the stable physical endpoint.

    The decision is made from the motion of the extrapolated triplet away
    from the largest-D observation, not from the residual of the fit itself.
    This rejects exact but violently overshooting two-point extrapolations.
    """
    candidates = candidate_high_D_correlation_fits(
        by_D, a_g, a_g_error,
    )
    if not candidates:
        return None
    return min(candidates, key=lambda fit: (
        fit["extrapolation_distance"], -len(fit["Ds"]),
    ))


def correlation_node(
    J2: float, signed_h: float, branch: str, fit: dict, handling: str,
) -> dict:
    strongest, middle, weakest = map(float, fit["correlations"])
    errors = np.asarray(fit["errors"], dtype=float)
    delta = weakest - strongest
    q = ((weakest + strongest - 2.0 * middle) / delta
         if delta > 1.0e-12 else 0.0)
    delta_error = float(fit.get(
        "Delta_error",
        (float(math.hypot(errors[0], errors[2]))
         if np.all(np.isfinite(errors[[0, 2]])) else math.nan),
    ))
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
        fit_rows = []
        for row in subset:
            fit_rows.append(SimpleNamespace(
                D=int(row["D"]), J2=J2, texture=texture,
                ranks=tuple(
                    (float(row[name]), float(row[f"{name}_error"]))
                    for name in ("strongest", "middle", "weakest")
                ),
            ))
        selected_fit = selected_story.stable_high_D_rank_fit(
            fit_rows, gapped_a[J2],
        )
        fit = None if selected_fit is None else {
            "Ds": selected_fit["Ds"],
            "correlations": selected_fit["correlations"],
            "errors": selected_fit["errors"],
            "Delta_error": selected_fit["Delta_error"],
            "rms": np.full(3, selected_fit["fit_rms_max"]),
            "max_abs_residual": selected_fit["fit_rms_max"],
            "extrapolation_distance": selected_fit[
                "extrapolation_distance"
            ],
        }
        if fit is not None:
            fits[(J2, texture)] = fit
    return fits


def build_texture_nodes(
    rows: list[FieldPoint], gapped_a: dict[float, float],
    gapped_a_errors: dict[float, float],
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
            gapped_a_errors.get(J2, 0.0),
        )
        if fit is None:
            continue
        node = correlation_node(
            J2, signed_h, subset[0].branch, fit,
            "per-D mean; endpoint-stable common high-D fixed-a_g window",
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
                if subset:
                    by_D = averaged_correlations_by_D(subset)
                    audited_window = selected_story.EXTRAPOLATION_WINDOWS.get(
                        (branch, J2)
                    )
                    if (audited_window is not None
                            and all(D in by_D for D in audited_window)):
                        fit = fit_correlation_window(
                            by_D, gapped_a[J2], list(audited_window),
                            gapped_a_errors.get(J2, 0.0),
                        )
                    else:
                        fit = fit_high_D_correlations(
                            by_D, gapped_a[J2],
                            gapped_a_errors.get(J2, 0.0),
                        )
                else:
                    fit = None
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
            "Delta_error": float(math.sqrt(sum(
                fit["Delta_error"] ** 2 for _, fit in components
            )) / 2.0),
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
    boundary, = axis.plot(
        hc_x, J2, color="#fff176", linewidth=2.2, marker="o",
        markersize=4.8, zorder=5,
        label=r"statistical energy crossing $h_c$",
    )
    boundary.set_path_effects([
        path_effects.Stroke(linewidth=4.0, foreground="black"),
        path_effects.Normal(),
    ])
    for row, value, J2_value in zip(ordered, hc, J2):
        error_low = float(row.get(
            "h_c_error_low", row.get("error_1sigma", math.nan)
        ))
        error_high = float(row.get(
            "h_c_error_high", row.get("error_1sigma", math.nan)
        ))
        if not math.isfinite(error_low) or not math.isfinite(error_high):
            continue
        left = field_plot_coordinate(value - error_low)
        right = field_plot_coordinate(value + error_high)
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
    # Energy extrapolation may use a high-D targeted repair even when that
    # (J2,D) pair lacks the four-sided field coverage required for Set A.
    energy_fit_points, energy_fit_audit = screen_normal_energy_curves(
        points, original_energies,
    )
    gapped_a = load_gapped_a()
    gapped_a_errors = load_gapped_a_errors()
    all_good_crossings: list[dict] = []
    energy_h_fits: list[dict] = []
    finite_D_energy_crossings: list[dict] = []
    for J2 in sorted({row.J2 for row in energy_fit_points}):
        if J2 not in gapped_a:
            continue
        subset = [row for row in energy_fit_points if close(row.J2, J2)]
        crossing, h_nodes, branch_fits = fit_all_good_crossing(
            J2, subset, gapped_a[J2], gapped_a_errors.get(J2, 0.0),
        )
        all_good_crossings.append(crossing)
        energy_h_fits.extend(h_nodes)
        finite_D_energy_crossings.extend(branch_fits)
    texture_nodes, texture_candidates = build_texture_nodes(
        all_good_points, gapped_a, gapped_a_errors,
    )

    write_dict_csv(output / "set_A.csv", set_a)
    write_dict_csv(output / "set_B_all_observations.csv",
                   [asdict(row) for row in set_b])
    write_dict_csv(output / "all_good_points.csv",
                   [asdict(row) for row in all_good_points])
    write_dict_csv(output / "all_good_points_audit.csv", all_good_audit)
    write_dict_csv(output / "energy_fit_points.csv",
                   [asdict(row) for row in energy_fit_points])
    write_dict_csv(output / "energy_fit_points_audit.csv", energy_fit_audit)
    write_dict_csv(output / "representative_selection_audit.csv", [
        {**asdict(item.point), "selected_for_fit": item.selected,
         "selection_reason": item.reason} for item in decisions
    ])
    write_dict_csv(output / "hc_D_infinity.csv", all_good_crossings)
    write_dict_csv(output / "hc_03_all_good_D_infinity.csv",
                   all_good_crossings)
    write_dict_csv(output / "energy_h_quadratic_by_D.csv", energy_h_fits)
    write_dict_csv(output / "energy_crossings_by_D.csv",
                   finite_D_energy_crossings)
    for stale_name in (
        "robust_fit_point_audit.csv", "finite_D_crossing_diagnostics.csv",
        "hc_03_all_good_fit_point_audit.csv",
        "hc_03_all_good_finite_D_crossings.csv",
        "energy_D_gapped_nodes.csv", "energy_quadratic_branch_fits.csv",
        "energy_gapped_coefficients_D_infinity.csv",
    ):
        (output / stale_name).unlink(missing_ok=True)
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
    for crossing in all_good_crossings:
        J2 = float(crossing["J2"])
        nodes = [row for row in energy_h_fits
                 if close(float(row["J2"]), J2)]
        branches = [row for row in finite_D_energy_crossings
                    if close(float(row["J2"]), J2)]
        token = f"{J2:.6f}".rstrip("0").rstrip(".").replace(".", "p")
        plot_two_stage_energy_diagnostic(
            crossing, nodes, branches, diagnostics / f"J2_{token}.pdf",
        )
    plot_two_stage_phase_boundary(
        all_good_crossings, output / "02_hc_vs_J2_phase_boundary.pdf",
    )
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
    accepted = sum(math.isfinite(row.get("h_c", math.nan))
                   for row in all_good_crossings)
    print(f"Discovered scalar pinning observations: {len(points)}")
    print(f"Deduplicated copied observations: {duplicates}")
    print(f"Excluded non-scalar rank-split observations: {rank_split}")
    print(f"Set A: {len(set_a)} (J2,D) pairs")
    print(f"Set B: {len(set_b)} observations")
    print(f"All non-obvious-bad B points: {len(all_good_points)}")
    print(f"finite-D fitted crossings: "
          f"{accepted}/{len(all_good_crossings)} accepted")
    zero_covered = sum(bool(row.get("contains_h0_1sigma"))
                       for row in all_good_crossings)
    print("statistical 1-sigma crossings containing h=0: "
          f"{zero_covered}/{len(all_good_crossings)}")
    print(f"Output: {output}")
    if failures:
        print(f"Discovery failures recorded: {len(failures)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
