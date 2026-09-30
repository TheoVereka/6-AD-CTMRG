#!/usr/bin/env python3
"""Select one physical NN-correlation observation per (texture, J2, D).

The selection is global rather than pointwise.  Hard energy eligibility is
applied first, followed by the requested Delta preference/window.  A
deterministic coordinate search then minimizes fixed-D roughness, fixed-J2
finite-D inconsistency, low-J2 residual splitting, high-J2 drift, and the
residual of C(D)=C_inf+k exp(-a_g D), with a_g fixed by the published 2C3
gapped energy fit.  The final figures contain data only, never these fits.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


HERE = Path(__file__).resolve().parent
VBC_DIR = HERE.parent
FIGS = VBC_DIR.parent
REPO = HERE.parents[3]
DATA = REPO.parent / "data"
sys.path.insert(0, str(VBC_DIR))
sys.path.insert(0, str(FIGS / "PublicationPlots"))

from publication_common import parse_observable  # noqa: E402
from plot_j2_seed_continuations import (  # noqa: E402
    Point,
    Seed,
    discover_last_izar,
    discover_legacy,
    discover_sep27,
    ranked_nn,
    read_seeds,
    read_sep27_seeds,
)


J2_GRID = (0.26, 0.265, 0.27, 0.275, 0.28, 0.29, 0.30, 0.31, 0.32)
TEXTURES = ("dimer-plaquette", "plaquette")
TEXTURE_TITLES = {
    "dimer-plaquette": "Dimer-plaquette sector",
    "plaquette": "Plaquette sector",
}
D_RANGES = {"dimer-plaquette": range(5, 11), "plaquette": range(6, 12)}
RANK_NAMES = ("strongest", "middle", "weakest")
RANK_COLORS = ("#c92535", "#2f9855", "#2878b8")
RANK_MARKERS = ("o", "s", "^")
RANK_LINESTYLES = ("-", "--", ":")
# Strongest is deliberately much more prominent, middle moderately more
# prominent, and weakest retains the previous marker size.
COMBINED_MARKER_SIZES = (7.2, 5.2, 3.2)
INVERSE_D_MARKER_SIZES = (8.2, 6.1, 4.2)

ORIGINAL_ROOT = DATA / "0713summary"
LEGACY_INPUT = DATA / "distinVBCsJ2Continuation" / "Results_Izar_J2_sequences"
LEGACY_MANIFEST = (
    REPO / "models" / "VBCJ2SeedContinuationIzar" / "selected_seed_manifest.csv"
)
LAST_INPUT = (
    DATA / "distinVBCsJ2Continuation" / "LastIzar" / "Results_LastIzar"
)
SEP27_INPUT = DATA / "external" / "Working_AD_Honeycomb_Sep27"
CATALOG = (
    DATA / "distinVBCsH0TensorCandidates" / "selection_20260923_170530"
    / "all_candidates.csv"
)
D7_REPAIR_ROOT = (
    DATA / "distinVBCsJ2Continuation" / "D7DimerJ2_0p26"
    / "Results_D7DimerJ2_0p26"
)
ENERGY_FITS = DATA / "processed" / "publicationPlots" / "figure_24_fits.csv"
DEFAULT_OUTPUT = HERE / "plots"

ENERGY_HARD = 3.0e-4
ENERGY_PREFERRED = 2.0e-4
DELTA_PREFERRED = 0.25
DELTA_WIDE = 0.35


@dataclass
class Candidate:
    candidate_id: str
    texture: str
    J2: float
    D: int
    chi: int
    source: str
    observation: Path
    tensor: Path
    energy: float
    ranks: tuple[tuple[float, float], ...]
    delta: float
    delta_error: float
    eta: float
    reference_available: bool
    reference_energy: float
    reference_delta: float
    energy_difference: float
    relative_delta_difference: float
    local_score: float = math.inf
    eligible: bool = False
    delta_exception: bool = False
    rejection_reason: str = ""

    @property
    def key(self) -> tuple[str, int, float]:
        return self.texture, self.D, self.J2


@dataclass
class Discovery:
    candidates: list[Candidate] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    seen: set[tuple[str, str]] = field(default_factory=set)


def close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=0.0, abs_tol=1.0e-10)


def j2_tag(value: float) -> str:
    thousandths = int(round(value * 1000))
    decimals = 3 if thousandths % 10 else 2
    return f"{value:.{decimals}f}".replace(".", "p")


def texture_from_eta(eta: float) -> str:
    return "dimer-plaquette" if eta >= 0.0 else "plaquette"


def summarize_observation(path: Path) -> tuple[
    float, tuple[tuple[float, float], ...], float, float, float
]:
    energy = float(parse_observable(path)["E"])
    ranks = ranked_nn(path)
    delta = float(ranks[2][0] - ranks[0][0])
    delta_error = float(math.hypot(ranks[2][1], ranks[0][1]))
    if delta <= 1.0e-14:
        eta = 0.0
    else:
        eta = float(2.0 * (ranks[1][0] - ranks[0][0]) / delta - 1.0)
    return energy, ranks, delta, delta_error, eta


def infer_tensor(observation: Path, D: int, chi: int) -> Path | None:
    folder = observation.parent
    direct = (
        folder / f"sweep_D{D}_chi{chi}_best.pt",
        folder / f"sweep_D{D}_chi_{chi}_best.pt",
        folder / "tensor_best.pt",
    )
    for path in direct:
        if path.is_file():
            return path
    matches = sorted(folder.glob(f"sweep_D{D}_chi*_best.pt"))
    return matches[-1] if matches else None


def load_references() -> dict[tuple[int, float], dict]:
    references: dict[tuple[int, float], dict] = {}
    for path in sorted(ORIGINAL_ROOT.glob(
        "J2_*/2tensor_twoC3/D_*/energy_magnetization_correlation.txt"
    )):
        try:
            J2 = float(path.parents[2].name.removeprefix("J2_").replace("p", "."))
            D = int(path.parent.name.removeprefix("D_"))
        except ValueError:
            continue
        if J2 not in J2_GRID or not 5 <= D <= 11:
            continue
        energy, ranks, delta, delta_error, eta = summarize_observation(path)
        references[(D, J2)] = {
            "energy": energy, "ranks": ranks, "delta": delta,
            "delta_error": delta_error, "eta": eta, "observation": path,
            "tensor": path.parent / "tensor_best.pt",
        }
    return references


def add_candidate(
    discovery: Discovery,
    references: dict[tuple[int, float], dict],
    *, texture: str, J2: float, D: int, chi: int, source: str,
    observation: Path, tensor: Path | None = None,
) -> None:
    if texture not in TEXTURES or J2 not in J2_GRID:
        return
    reference = references.get((D, J2))
    # 0713 has no D=11 reference at these three low-J2 values, but the
    # completed a03 plaquette continuation exists and the user explicitly
    # requires it to remain visible.  Keep it as unreferenced data rather than
    # inventing zero energy/Delta differences.
    allow_unreferenced = (
        texture == "plaquette" and D == 11
        and any(close(J2, value) for value in (0.26, 0.265, 0.27))
    )
    if reference is None and not allow_unreferenced:
        return
    try:
        observation = observation.resolve()
        tensor = (tensor or infer_tensor(observation, D, chi))
        if not observation.is_file():
            raise FileNotFoundError(f"missing observation: {observation}")
        if tensor is None or not tensor.is_file():
            raise FileNotFoundError(f"missing tensor next to {observation}")
        tensor = tensor.resolve()
        dedup = (str(observation).lower(), texture)
        if dedup in discovery.seen:
            return
        energy, ranks, delta, delta_error, eta = summarize_observation(observation)
    except (OSError, ValueError, KeyError) as error:
        discovery.errors.append(f"{source}: {error}")
        return
    discovery.seen.add(dedup)
    reference_available = reference is not None
    if reference_available:
        ref_energy = float(reference["energy"])
        ref_delta = float(reference["delta"])
        energy_difference = energy - ref_energy
        relative_delta = abs(delta - ref_delta) / max(abs(ref_delta), 1.0e-15)
    else:
        ref_energy = math.nan
        ref_delta = math.nan
        energy_difference = math.nan
        relative_delta = math.nan
    digest = hashlib.sha1(
        f"{texture}|{observation}".encode("utf-8")
    ).hexdigest()[:12]
    discovery.candidates.append(Candidate(
        candidate_id=f"{source}:{digest}", texture=texture, J2=J2, D=D,
        chi=chi, source=source, observation=observation, tensor=tensor,
        energy=energy, ranks=ranks, delta=delta, delta_error=delta_error,
        eta=eta, reference_available=reference_available,
        reference_energy=ref_energy,
        reference_delta=ref_delta,
        energy_difference=energy_difference,
        relative_delta_difference=relative_delta,
    ))


def discover_original(discovery: Discovery, references: dict) -> None:
    for (D, J2), row in sorted(references.items()):
        # Below/at the transition the texture label is not physical once the
        # splitting collapses: the same symmetry-restored tensor is a valid
        # endpoint of either continuation basin.  At larger J2 retain the eta
        # classification and require genuinely distinct VBC textures.
        textures = (TEXTURES if J2 <= 0.275
                    else (texture_from_eta(float(row["eta"])),))
        for texture in textures:
            add_candidate(
                discovery, references,
                texture=texture, J2=J2, D=D, chi=0,
                source=("original_2c3:symmetry_restored"
                        if J2 <= 0.275 else "original_2c3"),
                observation=row["observation"], tensor=row["tensor"],
            )


def discover_catalog(discovery: Discovery, references: dict) -> None:
    if not CATALOG.is_file():
        discovery.errors.append(f"candidate catalog missing: {CATALOG}")
        return
    with CATALOG.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            try:
                add_candidate(
                    discovery, references,
                    texture=row["selected_texture"], J2=float(row["J2"]),
                    D=int(row["D"]), chi=int(row["chi"]),
                    source=f"catalog:{row['source']}:{row['seed_branch']}",
                    observation=Path(row["observation_path"]),
                    tensor=Path(row["tensor_path"]),
                )
            except (KeyError, TypeError, ValueError) as error:
                discovery.errors.append(f"catalog row: {error}")


def add_points(
    discovery: Discovery, references: dict, points: list[Point], prefix: str
) -> None:
    for point in points:
        add_candidate(
            discovery, references,
            texture=point.seed_texture, J2=point.J2, D=point.D,
            chi=point.chi,
            source=(f"{prefix}:{point.seed_id}:{point.direction}:"
                    f"insurance{point.insurance}"),
            observation=Path(point.observation),
        )


def add_seeds(
    discovery: Discovery, references: dict, seeds: dict[str, Seed], prefix: str
) -> None:
    for seed in seeds.values():
        add_candidate(
            discovery, references,
            texture=seed.texture, J2=seed.J2, D=seed.D, chi=seed.chi,
            source=f"{prefix}:{seed.seed_id}", observation=seed.observation,
        )


def discover_continuations(discovery: Discovery, references: dict) -> None:
    if LEGACY_MANIFEST.is_file():
        seeds = read_seeds(LEGACY_MANIFEST)
        points, _ = discover_legacy(LEGACY_INPUT, seeds)
        add_seeds(discovery, references, seeds, "legacy_seed")
        add_points(discovery, references, points, "legacy")
    if LAST_INPUT.is_dir():
        seeds, points, _ = discover_last_izar(LAST_INPUT, ORIGINAL_ROOT)
        add_seeds(discovery, references, seeds, "lastizar_seed")
        add_points(discovery, references, points, "lastizar")
    sep_manifest = SEP27_INPUT / "private_manifest.tsv"
    if sep_manifest.is_file():
        seeds = read_sep27_seeds(sep_manifest)
        points, _ = discover_sep27(SEP27_INPUT, seeds)
        add_seeds(discovery, references, seeds, "sep27_seed")
        add_points(discovery, references, points, "sep27")


def discover_d7_repair(discovery: Discovery, references: dict) -> None:
    if not D7_REPAIR_ROOT.is_dir():
        return
    for observation in sorted(D7_REPAIR_ROOT.rglob(
        "D_7_chi_91_energy_magnetization_correlation.txt"
    )):
        # A snapshot can be taken while optimization is writing this folder.
        # Consume the stage only after run_stage.sh atomically publishes its
        # completion marker; a merely existing observation is insufficient.
        if not (observation.parent / "COMPLETED.stage").is_file():
            continue
        parts = {part.lower() for part in observation.parts}
        # The finite-field precursor is never eligible; only the h=0 child or
        # one of the four original-Hamiltonian adiabatic duplicates is used.
        if "adiabatic" not in parts and observation.parent.name.lower() != "h0":
            continue
        label = observation.parent.relative_to(D7_REPAIR_ROOT).as_posix()
        add_candidate(
            discovery, references, texture="dimer-plaquette", J2=0.26,
            D=7, chi=91, source=f"d7_repair:{label}",
            observation=observation,
        )


def local_score(candidate: Candidate) -> float:
    if candidate.reference_available:
        energy = abs(candidate.energy_difference) / ENERGY_PREFERRED
        delta = candidate.relative_delta_difference / DELTA_PREFERRED
        score = 2.0 * energy * energy + 0.8 * delta * delta
    else:
        score = 0.0
    if candidate.delta_exception:
        score += 12.0 + 4.0 * max(
            0.0, candidate.relative_delta_difference - DELTA_WIDE
        )
    if candidate.J2 >= 0.28:
        signed_eta = (candidate.eta if candidate.texture == "dimer-plaquette"
                      else -candidate.eta)
        score += 3.0 * (max(0.0, 0.35 - signed_eta) / 0.35) ** 2
    elif close(candidate.J2, 0.275):
        signed_eta = (candidate.eta if candidate.texture == "dimer-plaquette"
                      else -candidate.eta)
        score += 0.4 * (max(0.0, -signed_eta) / 0.5) ** 2
    # At equal physical quality, retain the higher-chi observation.
    score -= min(candidate.chi, 200) * 2.0e-4
    return score


def make_eligible(candidates: list[Candidate]) -> dict[
    tuple[str, int, float], list[Candidate]
]:
    energy_qualified: dict[tuple[str, int, float], list[Candidate]] = {}
    eligible: dict[tuple[str, int, float], list[Candidate]] = {}
    for candidate in candidates:
        if candidate.D not in D_RANGES[candidate.texture]:
            candidate.rejection_reason = "outside requested texture/D range"
            continue
        if candidate.texture == "plaquette" and candidate.D == 5:
            candidate.rejection_reason = "all D=5 plaquette data are excluded"
            continue
        if (candidate.texture == "dimer-plaquette" and candidate.D == 7
                and close(candidate.J2, 0.26)
                and not candidate.source.startswith("d7_repair:")):
            candidate.rejection_reason = "retired D7/J2=.26 dimer data"
            continue
        if candidate.J2 >= 0.28:
            signed_eta = (candidate.eta
                          if candidate.texture == "dimer-plaquette"
                          else -candidate.eta)
            if signed_eta < 0.25:
                candidate.rejection_reason = (
                    f"wrong/insufficient {candidate.texture} texture: "
                    f"signed eta={signed_eta:.3f}<0.25"
                )
                continue
        if not candidate.reference_available:
            signed_eta = (-candidate.eta if candidate.texture == "plaquette"
                          else candidate.eta)
            if signed_eta < 0.25:
                candidate.rejection_reason = (
                    "unreferenced point has wrong/insufficient texture: "
                    f"signed eta={signed_eta:.3f}<0.25"
                )
                continue
            candidate.eligible = True
            candidate.local_score = local_score(candidate)
            eligible.setdefault(candidate.key, []).append(candidate)
            continue
        if abs(candidate.energy_difference) > ENERGY_HARD:
            candidate.rejection_reason = (
                f"|dE|={abs(candidate.energy_difference):.6g}>0.0003"
            )
            continue
        energy_qualified.setdefault(candidate.key, []).append(candidate)

    for key, rows in energy_qualified.items():
        normal = [row for row in rows
                  if row.relative_delta_difference <= DELTA_WIDE]
        # A relative-Delta exception is only defensible in the restoration
        # regime when its absolute splitting is itself small enough to support
        # D->infinity equality.  Never use a large-splitting exception merely
        # to fill an otherwise absent grid point.
        texture, D, J2 = key
        base_limit = {5: 0.12, 6: 0.10, 7: 0.075, 8: 0.050,
                      9: 0.035, 10: 0.025, 11: 0.020}[D]
        if close(J2, 0.275):
            base_limit *= 2.2
        c_exceptions = [
            row for row in rows
            if J2 <= 0.275 and row.delta <= base_limit
        ]
        chosen_pool = normal if normal else c_exceptions
        if not chosen_pool:
            for row in rows:
                row.rejection_reason = (
                    "no <=35% Delta candidate and splitting is not small "
                    "enough for the restoration-regime exception"
                )
            continue
        for row in chosen_pool:
            row.eligible = True
            row.delta_exception = row.relative_delta_difference > DELTA_WIDE
            row.local_score = local_score(row)
        for row in rows:
            if row not in chosen_pool:
                row.rejection_reason = (
                    f"relative Delta difference "
                    f"{row.relative_delta_difference:.1%}>35%"
                )
        eligible[key] = sorted(
            chosen_pool,
            key=lambda row: (row.local_score, abs(row.energy_difference),
                             row.source, str(row.observation)),
        )
    return eligible


def load_fixed_a() -> dict[float, float]:
    values: dict[float, float] = {}
    if not ENERGY_FITS.is_file():
        return values
    with ENERGY_FITS.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if (row.get("ansatz") == "2C3" and row.get("model") == "gapped"
                    and row.get("parameter") == "a"):
                J2 = float(row["J2"])
                if J2 in J2_GRID:
                    values[J2] = float(row["central"])
    return values


def fixed_a_fit(rows: list[Candidate], rank: int, a: float) -> dict:
    D = np.asarray([row.D for row in rows], dtype=float)
    y = np.asarray([row.ranks[rank][0] for row in rows], dtype=float)
    basis = np.exp(-a * D)
    design = np.column_stack((np.ones(len(D)), basis))
    parameters, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ parameters
    return {
        "C_inf": float(parameters[0]), "k": float(parameters[1]),
        "rms": float(np.sqrt(np.mean(residual ** 2))),
        "max_abs": float(np.max(np.abs(residual))),
    }


def objective(selection: dict, fixed_a: dict[float, float]) -> float:
    # Unreferenced D=11 low-J2 points must be shown, but must not pull the
    # selection of the 0713-compared grid toward themselves.
    rows = [row for row in selection.values() if row.reference_available]
    score = sum(row.local_score for row in rows)

    # Fixed D: suppress unphysical zigzags and strongly non-monotone splitting.
    for texture in TEXTURES:
        for D in D_RANGES[texture]:
            curve = sorted(
                [row for row in rows if row.texture == texture and row.D == D],
                key=lambda row: row.J2,
            )
            for left, middle, right in zip(curve, curve[1:], curve[2:]):
                if not (middle.J2 - left.J2 <= 0.011
                        and right.J2 - middle.J2 <= 0.011):
                    continue
                fraction = ((middle.J2 - left.J2) / (right.J2 - left.J2))
                for rank in range(3):
                    prediction = (left.ranks[rank][0]
                                  + fraction * (right.ranks[rank][0]
                                                - left.ranks[rank][0]))
                    score += 0.7 * ((middle.ranks[rank][0] - prediction)
                                    / 0.018) ** 2
            high = [row for row in curve if row.J2 >= 0.275]
            for first, second in zip(high, high[1:]):
                reversal = first.delta - second.delta - 0.012
                if reversal > 0.0:
                    score += 2.5 * (reversal / 0.025) ** 2
            if D >= 8 and len(high) >= 3:
                x = np.sqrt(np.maximum(
                    np.asarray([row.J2 for row in high]) - 0.27, 0.0
                ))
                y = np.asarray([row.delta for row in high])
                design = np.column_stack((np.ones(len(x)), x))
                fit, *_ = np.linalg.lstsq(design, y, rcond=None)
                rms = float(np.sqrt(np.mean((y - design @ fit) ** 2)))
                score += 1.0 * (rms / 0.025) ** 2

    # Fixed J2: use the energy gapped-fit length a_g, but only as a selection
    # diagnostic/regularizer.  No fitted curve is ever drawn.
    for texture in TEXTURES:
        for J2 in J2_GRID:
            curve = sorted(
                [row for row in rows if row.texture == texture
                 and close(row.J2, J2)], key=lambda row: row.D,
            )
            if len(curve) >= 3 and J2 in fixed_a:
                fits = [fixed_a_fit(curve, rank, fixed_a[J2])
                        for rank in range(3)]
                rms_scale = 0.012 if J2 >= 0.275 else 0.018
                score += 0.8 * sum((fit["rms"] / rms_scale) ** 2
                                   for fit in fits)
                delta_inf = max(fit["C_inf"] for fit in fits) - min(
                    fit["C_inf"] for fit in fits
                )
                if J2 < 0.275:
                    score += 4.5 * (delta_inf / 0.025) ** 2
                elif close(J2, 0.275):
                    score += 1.5 * (delta_inf / 0.055) ** 2
            if J2 <= 0.275:
                for first, second in zip(curve, curve[1:]):
                    increase = second.delta - first.delta - 0.012
                    if increase > 0.0:
                        score += 1.5 * (increase / 0.025) ** 2
            if J2 >= 0.275:
                high_D = [row for row in curve if row.D >= 8]
                if len(high_D) >= 2:
                    for rank in range(3):
                        spread = float(np.std(
                            [row.ranks[rank][0] for row in high_D]
                        ))
                        score += 0.8 * (spread / 0.015) ** 2
    return float(score)


def select_globally(
    eligible: dict[tuple[str, int, float], list[Candidate]],
    fixed_a: dict[float, float],
) -> tuple[dict[tuple[str, int, float], Candidate], float]:
    keys = sorted(eligible, key=lambda key: (key[0], key[1], key[2]))
    rng = random.Random(260930)
    best_selection = None
    best_score = math.inf
    for restart in range(6):
        if restart == 0:
            selection = {key: eligible[key][0] for key in keys}
        else:
            selection = {
                key: rng.choice(eligible[key][:min(4, len(eligible[key]))])
                for key in keys
            }
        current_score = objective(selection, fixed_a)
        for _sweep in range(8):
            changed = False
            order = list(keys)
            rng.shuffle(order)
            for key in order:
                original = selection[key]
                local_best = original
                local_best_score = current_score
                for candidate in eligible[key]:
                    if candidate is original:
                        continue
                    selection[key] = candidate
                    trial = objective(selection, fixed_a)
                    if trial < local_best_score - 1.0e-10:
                        local_best, local_best_score = candidate, trial
                selection[key] = local_best
                if local_best is not original:
                    changed = True
                    current_score = local_best_score
            if not changed:
                break
        current_score = objective(selection, fixed_a)
        if current_score < best_score:
            best_selection = dict(selection)
            best_score = current_score
    assert best_selection is not None
    return best_selection, best_score


def fit_diagnostics(
    selection: dict, fixed_a: dict[float, float]
) -> list[dict]:
    output = []
    rows = list(selection.values())
    for texture in TEXTURES:
        for J2 in J2_GRID:
            curve = sorted(
                [row for row in rows if row.texture == texture
                 and close(row.J2, J2)], key=lambda row: row.D,
            )
            if len(curve) < 3 or J2 not in fixed_a:
                continue
            fits = [fixed_a_fit(curve, rank, fixed_a[J2])
                    for rank in range(3)]
            delta_inf = max(fit["C_inf"] for fit in fits) - min(
                fit["C_inf"] for fit in fits
            )
            for rank, fit in enumerate(fits):
                output.append({
                    "texture": texture, "J2": J2,
                    "Ds": " ".join(str(row.D) for row in curve),
                    "fixed_a_g_2C3": fixed_a[J2], "rank": RANK_NAMES[rank],
                    **fit, "extrapolated_splitting": delta_inf,
                })
    return output


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def candidate_record(row: Candidate, selected: set[str]) -> dict:
    record = {
        "selected": row.candidate_id in selected,
        "eligible": row.eligible,
        "delta_exception": row.delta_exception,
        "rejection_reason": row.rejection_reason,
        "candidate_id": row.candidate_id, "texture": row.texture,
        "J2": row.J2, "D": row.D, "chi": row.chi,
        "source": row.source, "energy": row.energy,
        "reference_available": row.reference_available,
        "reference_energy": row.reference_energy,
        "energy_difference": row.energy_difference,
        "delta": row.delta, "delta_error": row.delta_error,
        "reference_delta": row.reference_delta,
        "relative_delta_difference": row.relative_delta_difference,
        "eta": row.eta, "local_score": row.local_score,
        "strongest": row.ranks[0][0], "strongest_error": row.ranks[0][1],
        "middle": row.ranks[1][0], "middle_error": row.ranks[1][1],
        "weakest": row.ranks[2][0], "weakest_error": row.ranks[2][1],
        "observation": str(row.observation), "tensor": str(row.tensor),
    }
    return record


CANDIDATE_FIELDS = [
    "selected", "eligible", "delta_exception", "rejection_reason",
    "candidate_id", "texture", "J2", "D", "chi", "source", "energy",
    "reference_available", "reference_energy", "energy_difference", "delta", "delta_error",
    "reference_delta", "relative_delta_difference", "eta", "local_score",
    "strongest", "strongest_error", "middle", "middle_error", "weakest",
    "weakest_error", "observation", "tensor",
]


def selected_records(selection: dict) -> list[dict]:
    rows = sorted(selection.values(), key=lambda row: (row.texture, row.D, row.J2))
    return [candidate_record(row, {row.candidate_id}) for row in rows]


def combined_color(texture: str, index: int, total: int):
    cmap = plt.get_cmap("YlOrRd" if texture == "dimer-plaquette" else "PuBu")
    fraction = 0.35 if total == 1 else 0.30 + 0.68 * index / (total - 1)
    return cmap(fraction)


def plot_vs_j2(selection: dict, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16.0, 7.7), sharex=True, sharey=True,
                             constrained_layout=True)
    for ax, texture in zip(axes, TEXTURES):
        Ds = list(D_RANGES[texture])
        total_curves = len(Ds) * 3
        curve_index = 0
        for D in Ds:
            subset = sorted(
                [row for row in selection.values()
                 if row.texture == texture and row.D == D],
                key=lambda row: row.J2,
            )
            alpha = 0.18 + 0.82 * (D - Ds[0]) / max(1, Ds[-1] - Ds[0])
            for rank in range(3):
                color = combined_color(texture, curve_index, total_curves)
                curve_index += 1
                if not subset:
                    continue
                ax.errorbar(
                    [row.J2 for row in subset],
                    [row.ranks[rank][0] for row in subset],
                    yerr=[row.ranks[rank][1] for row in subset],
                    color=color, alpha=alpha, marker=RANK_MARKERS[rank],
                    markersize=COMBINED_MARKER_SIZES[rank],
                    linestyle=RANK_LINESTYLES[rank], linewidth=1.05,
                    elinewidth=0.7, capsize=1.8,
                    label=f"D={D} {RANK_NAMES[rank]}",
                )
        ax.set_title(TEXTURE_TITLES[texture], fontsize=13)
        ax.set_xlabel(r"$J_2$", fontsize=12)
        ax.grid(alpha=0.18)
        ax.legend(loc="best", fontsize=6.2, frameon=False, ncol=2,
                  handlelength=1.5, columnspacing=0.7)
    axes[0].set_ylabel("NN correlation", fontsize=12)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def plot_inverse_D(selection: dict, J2: float, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.3), sharex=True, sharey=True,
                             constrained_layout=True)
    for ax, texture in zip(axes, TEXTURES):
        subset = sorted(
            [row for row in selection.values()
             if row.texture == texture and close(row.J2, J2)],
            key=lambda row: 1.0 / row.D,
        )
        for rank, color in enumerate(RANK_COLORS):
            if not subset:
                continue
            ax.errorbar(
                [1.0 / row.D for row in subset],
                [row.ranks[rank][0] for row in subset],
                yerr=[row.ranks[rank][1] for row in subset],
                color=color, marker=RANK_MARKERS[rank],
                markersize=INVERSE_D_MARKER_SIZES[rank],
                linestyle=RANK_LINESTYLES[rank], linewidth=1.05,
                elinewidth=0.75, capsize=2.0,
            )
        ax.set_title(rf"{TEXTURE_TITLES[texture]}, $J_2={J2:g}$", fontsize=12)
        ax.set_xlabel(r"$1/D$", fontsize=12)
        ax.grid(alpha=0.18)
    axes[0].set_ylabel("NN correlation", fontsize=12)
    handles = [
        Line2D([], [], color=color, marker=RANK_MARKERS[rank],
               linestyle=RANK_LINESTYLES[rank], linewidth=1.05,
               markersize=INVERSE_D_MARKER_SIZES[rank], label=name)
        for rank, (color, name) in enumerate(zip(RANK_COLORS, RANK_NAMES))
    ]
    fig.legend(handles=handles, loc="outside upper center", ncol=3,
               frameon=False, fontsize=9)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def expected_keys() -> list[tuple[str, int, float]]:
    return [
        (texture, D, J2)
        for texture in TEXTURES for D in D_RANGES[texture] for J2 in J2_GRID
    ]


def write_report(
    path: Path, discovery: Discovery, selection: dict, score: float,
    diagnostics: list[dict], fixed_a: dict[float, float],
) -> None:
    missing = [key for key in expected_keys() if key not in selection]
    exceptions = [row for row in selection.values() if row.delta_exception]
    compared = [row for row in selection.values() if row.reference_available]
    unreferenced = [row for row in selection.values()
                    if not row.reference_available]
    preferred_energy = sum(
        abs(row.energy_difference) <= ENERGY_PREFERRED
        for row in compared
    )
    preferred_delta = sum(
        row.relative_delta_difference <= DELTA_PREFERRED
        for row in compared
    )
    lines = [
        "# Selected NN-correlation data audit", "",
        f"- Discovered candidates: {len(discovery.candidates)}",
        f"- Selected grid points: {len(selection)} / {len(expected_keys())}",
        f"- Global selection objective: {score:.8g}",
        f"- points with a 0713 reference: {len(compared)}",
        f"- unreferenced points retained explicitly: {len(unreferenced)}",
        f"- |dE| <= 0.0002: {preferred_energy}/{len(compared)} compared points",
        f"- relative Delta difference <= 25%: {preferred_delta}/{len(compared)} compared points",
        f"- relative Delta difference > 35% exceptions: {len(exceptions)}",
        f"- fixed-a diagnostics: {len(diagnostics)} rank fits",
        "",
        "The hard energy cutoff is 0.0003. Delta <=25% is preferred and <=35% "
        "is the normal wide window. A >35% candidate can enter only in the "
        "low-J2 restoration regime, only when its absolute splitting is small "
        "enough to support D->infinity equality, and only when the grid point "
        "has no energy-qualified <=35% option; every such exception is exposed "
        "explicitly.",
        "",
        "D=5 plaquette data are excluded unconditionally. The old D=7, J2=.26 "
        "dimer data are also excluded; that point remains missing until a complete "
        "candidate appears under the dedicated D7 repair snapshot root.",
        "D=11 plaquette points at J2=.26, .265, and .27 are retained without "
        "energy/Delta comparison because no corresponding 0713 reference exists; "
        "their CSV reference fields are NaN.",
        "",
        "## Missing grid points", "",
    ]
    lines.extend(
        f"- {texture}, D={D}, J2={J2:g}" for texture, D, J2 in missing
    )
    if not missing:
        lines.append("- none")
    lines += ["", "## Delta-window exceptions", ""]
    lines.extend(
        f"- {row.texture}, D={row.D}, J2={row.J2:g}: "
        f"{row.relative_delta_difference:.1%}, {row.source}"
        for row in exceptions
    )
    if not exceptions:
        lines.append("- none")
    lines += ["", "## Fixed energy-fit length", ""]
    lines.extend(f"- J2={J2:g}: a_g={fixed_a[J2]:.9g}"
                 for J2 in sorted(fixed_a))
    if discovery.errors:
        lines += ["", "## Discovery warnings", ""]
        lines.extend(f"- {error}" for error in discovery.errors)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    references = load_references()
    discovery = Discovery()
    discover_original(discovery, references)
    discover_catalog(discovery, references)
    discover_continuations(discovery, references)
    discover_d7_repair(discovery, references)
    eligible = make_eligible(discovery.candidates)
    fixed_a = load_fixed_a()
    selection, score = select_globally(eligible, fixed_a)
    diagnostics = fit_diagnostics(selection, fixed_a)
    selected_ids = {row.candidate_id for row in selection.values()}

    write_csv(
        output / "candidate_audit.csv",
        [candidate_record(row, selected_ids) for row in sorted(
            discovery.candidates,
            key=lambda row: (row.texture, row.D, row.J2, row.source),
        )],
        CANDIDATE_FIELDS,
    )
    write_csv(
        output / "selected_nn_data.csv", selected_records(selection),
        CANDIDATE_FIELDS,
    )
    diagnostic_fields = [
        "texture", "J2", "Ds", "fixed_a_g_2C3", "rank", "C_inf", "k",
        "rms", "max_abs", "extrapolated_splitting",
    ]
    write_csv(output / "fixed_a_fit_diagnostics.csv", diagnostics,
              diagnostic_fields)
    write_report(output / "selection_report.md", discovery, selection, score,
                 diagnostics, fixed_a)

    plot_vs_j2(selection, output / "NN_corr_vs_J2_selected.pdf")
    for J2 in J2_GRID:
        plot_inverse_D(
            selection, J2,
            output / f"NN_corr_vs_inverse_D_J2_{j2_tag(J2)}.pdf",
        )

    missing = len(expected_keys()) - len(selection)
    print(f"Discovered {len(discovery.candidates)} candidates")
    print(f"Selected {len(selection)} grid points; {missing} missing")
    print(f"Global objective: {score:.8g}")
    print(f"Plots and audits: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
