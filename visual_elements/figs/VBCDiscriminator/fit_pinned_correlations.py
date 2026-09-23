#!/usr/bin/env python3
"""Quadratically extrapolate pinned NN correlations to zero field.

Only positive-field replica-1 observations from the plaquette and
dimer-plaquette continuations enter the fits.  The rank-split source and the
measured h=0 values are excluded.  Every positive h that is present is used,
so a future h=0.005 stage is picked up automatically.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

from analyze_three_source_runs import COLORS, discover, select_highest_chi
from analyze_existing_twoc3 import (
    DEFAULT_INPUT as DEFAULT_ORIGINAL_INPUT,
    discover as discover_original,
)
from plot_pinning_replica1 import DEFAULT_BUNDLE, HERE, REPO, RankedStage, rank_stage
from sync_distin_vbcs import DEFAULT_ARCHIVE


DEFAULT_OUTPUT = HERE / "quadratic_pinning_extrapolation"
DEFAULT_CSV_OUTPUT = (REPO.parent / "data" / "processed"
                      / "VBCPinningQuadraticExtrapolation")
BRANCHES = ("dimer-plaquette", "plaquette")
BRANCH_LABELS = {
    "dimer-plaquette": "Dimer-plaquette pin",
    "plaquette": "Plaquette pin",
}
RANKS = (
    ("strongest", "rank1", "s", "-"),
    ("middle", "rank2", "o", "--"),
    ("weakest", "rank3", "^", ":"),
)
CLUSTER_STYLE = {
    "Izar": ("o", "-"),
    "Kuma": ("X", "--"),
}
LINEAR_RESPONSE_REFERENCE_H = 0.02


@dataclass(frozen=True)
class RankFit:
    cluster: str
    J2: float
    D: int
    branch: str
    rank: str
    n_positive_fields: int
    h_min: float
    h_max: float
    C0: float
    C0_stderr: float
    c1: float
    c2: float
    quadratic_over_linear_coefficient: float
    quadratic_over_linear_at_hmax: float
    linear_response_reference_h: float
    quadratic_over_linear_at_reference_h: float
    quadratic_correction_smaller_at_reference_h: bool
    r_squared: float
    rmse: float
    max_abs_residual: float
    C0_without_largest_h: float
    C0_window_shift: float
    observed_h0: float
    observed_minus_extrapolated: float


@dataclass(frozen=True)
class ExtrapolatedSet:
    cluster: str
    J2: float
    D: int
    branch: str
    C0_strongest: float
    C0_middle: float
    C0_weakest: float
    Delta0: float
    Delta0_fit_stderr: float
    E0: float
    E0_fit_stderr: float
    energy_c1: float
    energy_c2: float
    energy_r_squared: float
    energy_rmse: float
    energy_max_abs_residual: float
    E0_without_largest_h: float
    E0_window_shift: float
    min_r_squared: float
    max_rmse: float
    max_abs_residual: float
    max_quadratic_over_linear_coefficient: float
    max_quadratic_over_linear_at_hmax: float
    linear_response_reference_h: float
    max_quadratic_over_linear_at_reference_h: float
    all_quadratic_corrections_smaller_at_reference_h: bool
    max_C0_window_shift: float
    Delta0_window_shift_bound: float
    positive_fields: str


@dataclass(frozen=True)
class FitOmission:
    cluster: str
    J2: float
    D: int
    branch: str
    n_positive_fields: int
    positive_fields: str
    reason: str


@dataclass(frozen=True)
class EnergyFit:
    E0: float
    E0_stderr: float
    c1: float
    c2: float
    r_squared: float
    rmse: float
    max_abs_residual: float
    E0_without_largest_h: float
    E0_window_shift: float


def _finite_ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) <= 1e-14:
        return math.inf
    return abs(numerator / denominator)


def fit_one_rank(rows: list[RankedStage], rank_name: str,
                 attribute: str) -> RankFit:
    positive = sorted((row for row in rows if row.h > 0), key=lambda row: row.h)
    fields = np.asarray([row.h for row in positive], dtype=float)
    values = np.asarray([getattr(row, attribute) for row in positive], dtype=float)
    if len(set(fields)) < 4:
        key = (rows[0].cluster, rows[0].J2, rows[0].D, rows[0].branch)
        raise RuntimeError(
            f"{key}: quadratic quality assessment requires at least four "
            f"distinct h>0 points, found {sorted(set(fields))}"
        )

    coefficients, covariance = np.polyfit(fields, values, 2, cov=True)
    c2, c1, C0 = (float(value) for value in coefficients)
    predicted = np.polyval(coefficients, fields)
    residuals = values - predicted
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((values - np.mean(values)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else math.nan

    without_largest = fields < max(fields)
    C0_restricted = float(np.polyfit(
        fields[without_largest], values[without_largest], 2,
    )[2]) if np.count_nonzero(without_largest) >= 3 else math.nan

    zero_rows = [row for row in rows if math.isclose(row.h, 0.0, abs_tol=1e-14)]
    observed_h0 = (float(getattr(zero_rows[0], attribute))
                   if zero_rows else math.nan)
    coefficient_ratio = _finite_ratio(c2, c1)
    range_ratio = coefficient_ratio * float(max(fields))
    reference_h = min(LINEAR_RESPONSE_REFERENCE_H, float(max(fields)))
    reference_ratio = coefficient_ratio * reference_h
    first = rows[0]
    return RankFit(
        cluster=first.cluster, J2=first.J2, D=first.D, branch=first.branch,
        rank=rank_name, n_positive_fields=len(fields),
        h_min=float(min(fields)), h_max=float(max(fields)), C0=C0,
        C0_stderr=float(math.sqrt(max(float(covariance[2, 2]), 0.0))),
        c1=c1, c2=c2,
        quadratic_over_linear_coefficient=coefficient_ratio,
        quadratic_over_linear_at_hmax=range_ratio,
        linear_response_reference_h=reference_h,
        quadratic_over_linear_at_reference_h=reference_ratio,
        quadratic_correction_smaller_at_reference_h=reference_ratio < 1.0,
        r_squared=r_squared,
        rmse=float(math.sqrt(ss_res / len(fields))),
        max_abs_residual=float(np.max(np.abs(residuals))),
        C0_without_largest_h=C0_restricted,
        C0_window_shift=abs(C0 - C0_restricted),
        observed_h0=observed_h0,
        observed_minus_extrapolated=(observed_h0 - C0
                                     if math.isfinite(observed_h0) else math.nan),
    )


def fit_energy(rows: list[RankedStage]) -> EnergyFit:
    """Fit E(h)=E0+c1*h+c2*h^2 using the same positive fields as C(h)."""
    positive = sorted((row for row in rows if row.h > 0), key=lambda row: row.h)
    fields = np.asarray([row.h for row in positive], dtype=float)
    values = np.asarray([row.energy_per_site for row in positive], dtype=float)
    if len(set(fields)) < 4:
        key = (rows[0].cluster, rows[0].J2, rows[0].D, rows[0].branch)
        raise RuntimeError(
            f"{key}: energy quadratic covariance fit requires at least four "
            f"distinct h>0 points, found {sorted(set(fields))}"
        )

    coefficients, covariance = np.polyfit(fields, values, 2, cov=True)
    c2, c1, E0 = (float(value) for value in coefficients)
    residuals = values - np.polyval(coefficients, fields)
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((values - np.mean(values)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else math.nan
    without_largest = fields < max(fields)
    E0_restricted = float(np.polyfit(
        fields[without_largest], values[without_largest], 2,
    )[2]) if np.count_nonzero(without_largest) >= 3 else math.nan
    return EnergyFit(
        E0=E0,
        E0_stderr=float(math.sqrt(max(float(covariance[2, 2]), 0.0))),
        c1=c1,
        c2=c2,
        r_squared=r_squared,
        rmse=float(math.sqrt(ss_res / len(fields))),
        max_abs_residual=float(np.max(np.abs(residuals))),
        E0_without_largest_h=E0_restricted,
        E0_window_shift=abs(E0 - E0_restricted),
    )


def fit_group(rows: list[RankedStage]) -> tuple[list[RankFit], ExtrapolatedSet]:
    fits = [fit_one_rank(rows, name, attribute)
            for name, attribute, _, _ in RANKS]
    by_rank = {fit.rank: fit for fit in fits}
    strongest, middle, weakest = (by_rank[name]
                                  for name in ("strongest", "middle", "weakest"))
    delta = weakest.C0 - strongest.C0
    delta_stderr = math.hypot(strongest.C0_stderr, weakest.C0_stderr)
    energy = fit_energy(rows)
    fields = sorted({row.h for row in rows if row.h > 0}, reverse=True)
    summary = ExtrapolatedSet(
        cluster=rows[0].cluster, J2=rows[0].J2, D=rows[0].D,
        branch=rows[0].branch,
        C0_strongest=strongest.C0, C0_middle=middle.C0,
        C0_weakest=weakest.C0, Delta0=delta,
        Delta0_fit_stderr=delta_stderr,
        E0=energy.E0, E0_fit_stderr=energy.E0_stderr,
        energy_c1=energy.c1, energy_c2=energy.c2,
        energy_r_squared=energy.r_squared, energy_rmse=energy.rmse,
        energy_max_abs_residual=energy.max_abs_residual,
        E0_without_largest_h=energy.E0_without_largest_h,
        E0_window_shift=energy.E0_window_shift,
        min_r_squared=min(fit.r_squared for fit in fits),
        max_rmse=max(fit.rmse for fit in fits),
        max_abs_residual=max(fit.max_abs_residual for fit in fits),
        max_quadratic_over_linear_coefficient=max(
            fit.quadratic_over_linear_coefficient for fit in fits),
        max_quadratic_over_linear_at_hmax=max(
            fit.quadratic_over_linear_at_hmax for fit in fits),
        linear_response_reference_h=strongest.linear_response_reference_h,
        max_quadratic_over_linear_at_reference_h=max(
            fit.quadratic_over_linear_at_reference_h for fit in fits),
        all_quadratic_corrections_smaller_at_reference_h=all(
            fit.quadratic_correction_smaller_at_reference_h for fit in fits),
        max_C0_window_shift=max(fit.C0_window_shift for fit in fits),
        Delta0_window_shift_bound=(strongest.C0_window_shift
                                   + weakest.C0_window_shift),
        positive_fields=";".join(f"{field:g}" for field in fields),
    )
    return fits, summary


def write_dataclasses(rows: list, path: Path, row_type=None) -> None:
    if not rows and row_type is None:
        raise ValueError(f"refusing to write empty table: {path}")
    fieldnames = (list(asdict(rows[0])) if rows
                  else list(row_type.__dataclass_fields__))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def plot_fit_pages(groups: dict[tuple, list[RankedStage]],
                   fit_lookup: dict[tuple, list[RankFit]], path: Path,
                   png_dir: Path | None = None,
                   h_max: float | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if png_dir is not None:
        png_dir.mkdir(parents=True, exist_ok=True)
    page_keys = sorted({(key[0], key[1], key[2]) for key in groups})
    with PdfPages(path) as pdf:
        for cluster, J2, D in page_keys:
            page_groups = {branch: groups.get((cluster, J2, D, branch), [])
                           for branch in BRANCHES}
            all_values = [getattr(row, attribute)
                          for rows in page_groups.values() for row in rows
                          for _, attribute, _, _ in RANKS]
            lo, hi = min(all_values), max(all_values)
            pad = max(0.015, 0.05 * (hi - lo))
            max_field = max(row.h for rows in page_groups.values()
                            for row in rows)
            h_pad = max(1.0e-4, 0.05 * max_field)
            fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.3), sharex=True,
                                     sharey=True)
            for ax, branch in zip(axes, BRANCHES):
                rows = page_groups[branch]
                if not rows:
                    ax.text(0.5, 0.5, "No completed observation",
                            transform=ax.transAxes, ha="center", va="center")
                    continue
                fits = {fit.rank: fit
                        for fit in fit_lookup[(cluster, J2, D, branch)]}
                positive = sorted((row for row in rows if row.h > 0),
                                  key=lambda row: row.h)
                h_curve = np.linspace(0, max(row.h for row in positive), 200)
                for rank_name, attribute, marker, line in RANKS:
                    fit = fits[rank_name]
                    color = COLORS[branch]
                    ax.plot(h_curve,
                            fit.C0 + fit.c1 * h_curve + fit.c2 * h_curve ** 2,
                            color=color, linestyle=line, linewidth=1.6)
                    ax.plot([row.h for row in positive],
                            [getattr(row, attribute) for row in positive],
                            linestyle="none", marker=marker, markersize=7,
                            markerfacecolor="white", markeredgewidth=1.4,
                            color=color, label=rank_name)
                    ax.plot(0, fit.C0, marker="*", markersize=10, color=color,
                            linestyle="none")
                    zero = [row for row in rows
                            if math.isclose(row.h, 0.0, abs_tol=1e-14)]
                    if zero:
                        ax.plot(0, getattr(zero[0], attribute), marker="x",
                                markersize=7, markeredgewidth=1.4, color="0.25",
                                linestyle="none")
                ax.set_title(BRANCH_LABELS[branch], color=COLORS[branch])
                ax.set_xlabel("pinning field $h$")
                ax.set_xlim(-h_pad, max_field + h_pad)
                ax.set_ylim(lo - pad, hi + pad)
                ax.grid(alpha=0.22)
                summary_fits = list(fits.values())
                ax.text(
                    0.03, 0.04,
                    f"min $R^2$={min(f.r_squared for f in summary_fits):.5f}\n"
                    f"max $|c_2h/c_1|$ at "
                    f"$h={summary_fits[0].linear_response_reference_h:g}$="
                    f"{max(f.quadratic_over_linear_at_reference_h for f in summary_fits):.3f}\n"
                    f"max $C_0$ window shift="
                    f"{max(f.C0_window_shift for f in summary_fits):.4f}",
                    transform=ax.transAxes, fontsize=8.5, va="bottom",
                    bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.85},
                )
            axes[0].set_ylabel("sorted NN correlation")
            rank_handles = [Line2D([], [], color="0.2", linestyle=line,
                                   marker=marker, markerfacecolor="white",
                                   label=name)
                            for name, _, marker, line in RANKS]
            extra_handles = [
                Line2D([], [], color="0.25", marker="*", linestyle="none",
                       label="quadratic extrapolation at $h=0$"),
                Line2D([], [], color="0.25", marker="x", linestyle="none",
                       label="measured $h=0$ (not fitted)"),
            ]
            fig.legend(handles=rank_handles + extra_handles, loc="lower center",
                       ncol=5, frameon=False, fontsize=8.5)
            fig.suptitle(
                f"{cluster} replica 1: $J_2/J_1={J2:g}$, $D={D}$; "
                r"$C(h)=C_0+c_1h+c_2h^2$ from $h>0$ only"
                + (fr", $h\leq {h_max:g}$" if h_max is not None else ""),
                fontsize=13,
            )
            fig.tight_layout(rect=(0, 0.11, 1, 0.94))
            pdf.savefig(fig)
            if png_dir is not None:
                j2_label = f"{J2:.3f}".rstrip("0").rstrip(".").replace(".", "p")
                fig.savefig(
                    png_dir / f"{cluster}_J2_{j2_label}_D_{D}.png",
                    dpi=220, bbox_inches="tight",
                )
            plt.close(fig)


def load_original_energies(input_dir: Path,
                           dimensions: tuple[int, ...]) -> dict[tuple[float, int], float]:
    """Read the original, unbiased 0713summary twoC3 observations."""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"original 0713summary directory is missing: {input_dir}")
    selected = {}
    for row in discover_original(input_dir, "2tensor_twoC3", False):
        if row.D not in dimensions:
            continue
        key = (row.J2, row.D)
        if key not in selected or row.chi > selected[key].chi:
            selected[key] = row
    if not selected:
        raise RuntimeError(f"no original twoC3 energies found in {input_dir}")
    return {key: row.energy_per_site for key, row in selected.items()}


def plot_extrapolated(summaries: list[ExtrapolatedSet], path: Path,
                      dimensions: tuple[int, ...],
                      original_energies: dict[tuple[float, int], float],
                      png_path: Path | None = None,
                      h_max: float | None = None) -> None:
    fig, axes = plt.subplots(
        3, len(dimensions), figsize=(5.0 * len(dimensions), 12.0),
        squeeze=False, sharex="col", sharey="row",
    )
    all_fit_j2 = sorted({row.J2 for row in summaries})
    j2_min, j2_max = min(all_fit_j2), max(all_fit_j2)
    energy_differences: list[tuple[float, float]] = []
    for column, D in enumerate(dimensions):
        split_ax, energy_ax, difference_ax = axes[:, column]
        panel = [row for row in summaries if row.D == D]
        for cluster in ("Izar", "Kuma"):
            marker, line = CLUSTER_STYLE[cluster]
            offset = -0.00025 if cluster == "Kuma" else 0.00025
            for branch in BRANCHES:
                subset = sorted((row for row in panel
                                 if row.cluster == cluster and row.branch == branch),
                                key=lambda row: row.J2)
                if not subset:
                    continue
                split_ax.errorbar(
                    [row.J2 + offset for row in subset],
                    [row.Delta0 for row in subset],
                    yerr=[row.Delta0_fit_stderr for row in subset],
                    color=COLORS[branch], marker=marker, linestyle=line,
                    linewidth=1.35, markersize=7, markerfacecolor="white",
                    capsize=2.5,
                    label=f"{BRANCH_LABELS[branch]} ({cluster})",
                )

        original = sorted(
            (J2, energy) for (J2, original_D), energy in original_energies.items()
            if original_D == D and j2_min <= J2 <= j2_max
        )
        if original:
            energy_ax.plot(
                [item[0] for item in original], [item[1] for item in original],
                color="black", marker="o", markersize=4.5, linewidth=1.35,
                label="original 0713summary twoC3 energy", zorder=1,
            )
        for cluster in ("Izar", "Kuma"):
            marker, line = CLUSTER_STYLE[cluster]
            offset = -0.00025 if cluster == "Kuma" else 0.00025
            for branch in BRANCHES:
                subset = sorted((row for row in panel
                                 if row.cluster == cluster and row.branch == branch),
                                key=lambda row: row.J2)
                if not subset:
                    continue
                energy_ax.errorbar(
                    [row.J2 + offset for row in subset],
                    [row.E0 for row in subset],
                    yerr=[row.E0_fit_stderr for row in subset],
                    color=COLORS[branch], marker=marker, linestyle=line,
                    linewidth=1.35, markersize=7, markerfacecolor="white",
                    elinewidth=1.35, capsize=4.0, capthick=1.35, zorder=2,
                    label=f"{BRANCH_LABELS[branch]} ({cluster})",
                )
                difference_rows = [
                    (row, original_energies.get((row.J2, D))) for row in subset
                ]
                difference_rows = [
                    (row, original_energy)
                    for row, original_energy in difference_rows
                    if original_energy is not None
                ]
                if difference_rows:
                    differences = [row.E0 - original_energy
                                   for row, original_energy in difference_rows]
                    errors = [row.E0_fit_stderr
                              for row, _ in difference_rows]
                    energy_differences.extend(zip(differences, errors))
                    difference_ax.errorbar(
                        [row.J2 + offset for row, _ in difference_rows],
                        differences, yerr=errors,
                        color=COLORS[branch], marker=marker, linestyle=line,
                        linewidth=1.35, markersize=7, markerfacecolor="white",
                        elinewidth=1.35, capsize=4.0, capthick=1.35,
                        label=f"{BRANCH_LABELS[branch]} ({cluster})",
                    )

        split_ax.set_title(f"$D={D}$")
        split_ax.grid(alpha=0.22)
        energy_ax.grid(alpha=0.22)
        difference_ax.set_xlabel("$J_2/J_1$")
        difference_ax.grid(alpha=0.22)
        difference_ax.axhline(0.0, color="black", linewidth=0.8, zorder=0)
        difference_ax.tick_params(axis="x", labelrotation=45, labelsize=8)
        if panel:
            ticks = sorted({row.J2 for row in panel}
                           | {J2 for J2, _ in original})
            energy_ax.set_xticks(ticks)
        else:
            split_ax.text(0.5, 0.5, "No fitted pinning data",
                          transform=split_ax.transAxes,
                          ha="center", va="center")
            energy_ax.text(0.5, 0.5, "No fitted pinning data",
                           transform=energy_ax.transAxes,
                           ha="center", va="center")
            difference_ax.text(0.5, 0.5, "No fitted pinning data",
                               transform=difference_ax.transAxes,
                               ha="center", va="center")
    axes[0, 0].set_ylabel(
        r"extrapolated splitting $\Delta_0=C_{\rm weakest}(0)-C_{\rm strongest}(0)$"
    )
    axes[1, 0].set_ylabel(r"energy/site at $h\to0$, $E_0$")
    axes[2, 0].set_ylabel(
        r"$E_{h\mathrm{-extrapolation}}-E_{\mathrm{original}}$"
    )
    if energy_differences:
        upper = max(value + error for value, error in energy_differences)
        axes[2, 0].set_ylim(0.0, max(1.0e-6, 1.08 * upper))
    handles, labels = [], []
    for ax in axes.flat:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               fontsize=9)
    window = (fr", $h\leq {h_max:g}$ only" if h_max is not None else "")
    fig.suptitle(
        "Quadratic $h>0$ extrapolation" + window
        + "; original twoC3 energy in black",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0.075, 1, 0.965))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    if png_path is not None:
        png_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def load_groups(roots: tuple[tuple[str, Path], ...],
                dimensions: tuple[int, ...] | None = None,
                h_max: float | None = None,
                ) -> dict[tuple, list[RankedStage]]:
    groups: dict[tuple, list[RankedStage]] = defaultdict(list)
    for cluster, root in roots:
        if not root.is_dir():
            raise FileNotFoundError(f"{cluster} input directory is missing: {root}")
        stages = [stage for stage in select_highest_chi(discover(root, strict=True))
                  if stage.replica == 1]
        for stage in stages:
            if (stage.branch in BRANCHES
                    and (dimensions is None or stage.D in dimensions)
                    and (h_max is None or stage.field <= h_max + 1.0e-12)):
                ranked = rank_stage(stage, cluster)
                groups[(cluster, stage.J2, stage.D, stage.branch)].append(ranked)
    if not groups:
        raise RuntimeError("no requested plaquette/dimer-plaquette observations found")
    return groups


def run_analysis(roots: tuple[tuple[str, Path], ...], output_dir: Path,
                 csv_output_dir: Path,
                 dimensions: tuple[int, ...] | None = None,
                 original_input: Path = DEFAULT_ORIGINAL_INPUT,
                 h_max: float | None = None,
                 write_png: bool = False,
                 ) -> tuple[list[ExtrapolatedSet], list[FitOmission]]:
    """Fit every dynamically eligible group and report incomplete groups."""
    groups = load_groups(roots, dimensions, h_max)
    rank_fits: list[RankFit] = []
    summaries: list[ExtrapolatedSet] = []
    omissions: list[FitOmission] = []
    fit_lookup: dict[tuple, list[RankFit]] = {}
    eligible_groups: dict[tuple, list[RankedStage]] = {}
    for key, rows in sorted(groups.items()):
        fields = sorted({row.h for row in rows if row.h > 0}, reverse=True)
        if len(fields) < 4:
            omissions.append(FitOmission(
                cluster=key[0], J2=key[1], D=key[2], branch=key[3],
                n_positive_fields=len(fields),
                positive_fields=";".join(f"{field:g}" for field in fields),
                reason="quadratic covariance fit requires at least four distinct h>0 points",
            ))
            continue
        fits, summary = fit_group(rows)
        rank_fits.extend(fits)
        summaries.append(summary)
        fit_lookup[key] = fits
        eligible_groups[key] = rows

    write_dataclasses(omissions, csv_output_dir / "omitted_incomplete_fits.csv",
                      FitOmission)
    if not summaries:
        raise RuntimeError(
            "no group has the four distinct positive fields required for a "
            "quadratic covariance fit; see omitted_incomplete_fits.csv"
        )
    fitted_dimensions = tuple(sorted({row.D for row in summaries}))
    original_energies = load_original_energies(original_input, fitted_dimensions)
    plot_fit_pages(
        eligible_groups, fit_lookup,
        output_dir / "01_correlation_fits.pdf",
        output_dir / "01_correlation_fits_png" if write_png else None,
        h_max,
    )
    new_summary_path = output_dir / "02_extrapolated_vs_J2.pdf"
    plot_extrapolated(
        summaries, new_summary_path, fitted_dimensions, original_energies,
        new_summary_path.with_suffix(".png") if write_png else None,
        h_max,
    )
    legacy_summary_path = output_dir / "02_extrapolated_splitting_vs_J2.pdf"
    if legacy_summary_path.is_file():
        legacy_summary_path.unlink()
    write_dataclasses(rank_fits,
                      csv_output_dir / "quadratic_fit_coefficients.csv")
    write_dataclasses(
        summaries,
        csv_output_dir / "extrapolated_correlations_and_splitting.csv",
    )
    return summaries, omissions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kuma", type=Path,
                        default=DEFAULT_ARCHIVE / "Results_Kuma_replica1")
    parser.add_argument("--izar", type=Path,
                        default=DEFAULT_ARCHIVE / "Results_Izar_replica1")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--csv-output-dir", type=Path, default=DEFAULT_CSV_OUTPUT)
    parser.add_argument("--original-input", type=Path,
                        default=DEFAULT_ORIGINAL_INPUT,
                        help="0713summary root for the unbiased twoC3 energy curve")
    parser.add_argument("--h-max", type=float, default=None,
                        help="include only observations with h <= this value")
    parser.add_argument("--png", action="store_true",
                        help="also write per-fit-page and summary PNG files")
    parser.add_argument(
        "--dimensions", type=int, nargs="+", default=(6, 7, 8, 9, 10),
        help="D filter; default excludes D5 and D11",
    )
    args = parser.parse_args()
    dimensions = (tuple(sorted(set(args.dimensions)))
                  if args.dimensions is not None else None)

    # Strictly validate all discovered observations before creating outputs.
    summaries, omissions = run_analysis(
        (("Kuma", args.kuma), ("Izar", args.izar)),
        args.output_dir, args.csv_output_dir, dimensions, args.original_input,
        args.h_max, args.png,
    )

    print(f"Fitted {len(summaries)} cluster/J2/D/pin combinations "
          f"({3 * len(summaries)} individual correlation fits).")
    print(f"Figures: {args.output_dir}")
    print(f"Tables:  {args.csv_output_dir}")
    print(f"Dynamically omitted incomplete groups: {len(omissions)}")
    print(f"Worst R^2: {min(row.min_r_squared for row in summaries):.6f}")
    reference_fields = sorted({row.linear_response_reference_h
                               for row in summaries})
    reference_label = ",".join(f"{field:g}" for field in reference_fields)
    print(f"Combinations passing |c2*h^2| < |c1*h| at h="
          f"{reference_label} for all three correlations: "
          f"{sum(row.all_quadratic_corrections_smaller_at_reference_h for row in summaries)}"
          f"/{len(summaries)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
