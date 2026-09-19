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
from plot_pinning_replica1 import DEFAULT_BUNDLE, HERE, REPO, RankedStage, rank_stage


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
    quadratic_coefficient_smaller: bool
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
    min_r_squared: float
    max_rmse: float
    max_abs_residual: float
    max_quadratic_over_linear_coefficient: float
    max_quadratic_over_linear_at_hmax: float
    all_quadratic_coefficients_smaller: bool
    max_C0_window_shift: float
    Delta0_window_shift_bound: float
    positive_fields: str


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
    first = rows[0]
    return RankFit(
        cluster=first.cluster, J2=first.J2, D=first.D, branch=first.branch,
        rank=rank_name, n_positive_fields=len(fields),
        h_min=float(min(fields)), h_max=float(max(fields)), C0=C0,
        C0_stderr=float(math.sqrt(max(float(covariance[2, 2]), 0.0))),
        c1=c1, c2=c2,
        quadratic_over_linear_coefficient=coefficient_ratio,
        quadratic_over_linear_at_hmax=range_ratio,
        quadratic_coefficient_smaller=abs(c2) < abs(c1),
        r_squared=r_squared,
        rmse=float(math.sqrt(ss_res / len(fields))),
        max_abs_residual=float(np.max(np.abs(residuals))),
        C0_without_largest_h=C0_restricted,
        C0_window_shift=abs(C0 - C0_restricted),
        observed_h0=observed_h0,
        observed_minus_extrapolated=(observed_h0 - C0
                                     if math.isfinite(observed_h0) else math.nan),
    )


def fit_group(rows: list[RankedStage]) -> tuple[list[RankFit], ExtrapolatedSet]:
    fits = [fit_one_rank(rows, name, attribute)
            for name, attribute, _, _ in RANKS]
    by_rank = {fit.rank: fit for fit in fits}
    strongest, middle, weakest = (by_rank[name]
                                  for name in ("strongest", "middle", "weakest"))
    delta = weakest.C0 - strongest.C0
    delta_stderr = math.hypot(strongest.C0_stderr, weakest.C0_stderr)
    fields = sorted({row.h for row in rows if row.h > 0}, reverse=True)
    summary = ExtrapolatedSet(
        cluster=rows[0].cluster, J2=rows[0].J2, D=rows[0].D,
        branch=rows[0].branch,
        C0_strongest=strongest.C0, C0_middle=middle.C0,
        C0_weakest=weakest.C0, Delta0=delta,
        Delta0_fit_stderr=delta_stderr,
        min_r_squared=min(fit.r_squared for fit in fits),
        max_rmse=max(fit.rmse for fit in fits),
        max_abs_residual=max(fit.max_abs_residual for fit in fits),
        max_quadratic_over_linear_coefficient=max(
            fit.quadratic_over_linear_coefficient for fit in fits),
        max_quadratic_over_linear_at_hmax=max(
            fit.quadratic_over_linear_at_hmax for fit in fits),
        all_quadratic_coefficients_smaller=all(
            fit.quadratic_coefficient_smaller for fit in fits),
        max_C0_window_shift=max(fit.C0_window_shift for fit in fits),
        Delta0_window_shift_bound=(strongest.C0_window_shift
                                   + weakest.C0_window_shift),
        positive_fields=";".join(f"{field:g}" for field in fields),
    )
    return fits, summary


def write_dataclasses(rows: list, path: Path) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def plot_fit_pages(groups: dict[tuple, list[RankedStage]],
                   fit_lookup: dict[tuple, list[RankFit]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
                ax.set_xlim(-0.004, 0.084)
                ax.set_ylim(lo - pad, hi + pad)
                ax.grid(alpha=0.22)
                summary_fits = list(fits.values())
                ax.text(
                    0.03, 0.04,
                    f"min $R^2$={min(f.r_squared for f in summary_fits):.5f}\n"
                    f"max $|c_2h_{{max}}/c_1|$="
                    f"{max(f.quadratic_over_linear_at_hmax for f in summary_fits):.3f}\n"
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
                f"{cluster} replica 1: $J_2/J_1={J2:.2f}$, $D={D}$; "
                r"$C(h)=C_0+c_1h+c_2h^2$ from $h>0$ only",
                fontsize=13,
            )
            fig.tight_layout(rect=(0, 0.11, 1, 0.94))
            pdf.savefig(fig)
            plt.close(fig)


def plot_splitting(summaries: list[ExtrapolatedSet], path: Path,
                   dimensions: tuple[int, ...]) -> None:
    fig, axes = plt.subplots(1, len(dimensions), figsize=(5.0 * len(dimensions), 4.8),
                             squeeze=False, sharey=True)
    for ax, D in zip(axes[0], dimensions):
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
                ax.errorbar(
                    [row.J2 + offset for row in subset],
                    [row.Delta0 for row in subset],
                    yerr=[row.Delta0_fit_stderr for row in subset],
                    color=COLORS[branch], marker=marker, linestyle=line,
                    linewidth=1.35, markersize=7, markerfacecolor="white",
                    capsize=2.5,
                    label=f"{BRANCH_LABELS[branch]} ({cluster})",
                )
        ax.set_title(f"$D={D}$")
        ax.set_xlabel("$J_2/J_1$")
        ax.grid(alpha=0.22)
        if panel:
            ax.set_xticks(sorted({row.J2 for row in panel}))
        else:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center")
    axes[0, 0].set_ylabel(
        r"extrapolated splitting $\Delta_0=C_{\rm weakest}(0)-C_{\rm strongest}(0)$"
    )
    handles, labels = [], []
    for ax in axes[0]:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               fontsize=9)
    fig.suptitle("Quadratic $h>0$ extrapolation; rank-split source excluded",
                 fontsize=14)
    fig.tight_layout(rect=(0, 0.14, 1, 0.93))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def load_groups(roots: tuple[tuple[str, Path], ...],
                dimensions: tuple[int, ...]) -> dict[tuple, list[RankedStage]]:
    groups: dict[tuple, list[RankedStage]] = defaultdict(list)
    for cluster, root in roots:
        if not root.is_dir():
            raise FileNotFoundError(f"{cluster} input directory is missing: {root}")
        stages = [stage for stage in select_highest_chi(discover(root, strict=True))
                  if stage.replica == 1]
        for stage in stages:
            if stage.branch in BRANCHES and stage.D in dimensions:
                ranked = rank_stage(stage, cluster)
                groups[(cluster, stage.J2, stage.D, stage.branch)].append(ranked)
    if not groups:
        raise RuntimeError("no requested plaquette/dimer-plaquette observations found")
    return groups


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kuma", type=Path,
                        default=DEFAULT_BUNDLE / "Results_VBC_three")
    parser.add_argument("--izar", type=Path,
                        default=DEFAULT_BUNDLE / "Results_VBC_branches")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--csv-output-dir", type=Path, default=DEFAULT_CSV_OUTPUT)
    parser.add_argument("--dimensions", type=int, nargs="+", default=(7, 8, 9))
    args = parser.parse_args()
    dimensions = tuple(sorted(set(args.dimensions)))

    # Strictly validate all discovered observations before creating outputs.
    groups = load_groups((("Kuma", args.kuma), ("Izar", args.izar)), dimensions)
    rank_fits: list[RankFit] = []
    summaries: list[ExtrapolatedSet] = []
    fit_lookup: dict[tuple, list[RankFit]] = {}
    for key, rows in sorted(groups.items()):
        fits, summary = fit_group(rows)
        rank_fits.extend(fits)
        summaries.append(summary)
        fit_lookup[key] = fits

    plot_fit_pages(groups, fit_lookup, args.output_dir / "01_correlation_fits.pdf")
    plot_splitting(summaries, args.output_dir / "02_extrapolated_splitting_vs_J2.pdf",
                   dimensions)
    write_dataclasses(rank_fits, args.csv_output_dir / "quadratic_fit_coefficients.csv")
    write_dataclasses(summaries,
                      args.csv_output_dir / "extrapolated_correlations_and_splitting.csv")

    print(f"Fitted {len(summaries)} cluster/J2/D/pin combinations "
          f"({len(rank_fits)} individual correlation fits).")
    print(f"Figures: {args.output_dir}")
    print(f"Tables:  {args.csv_output_dir}")
    print(f"Worst R^2: {min(row.min_r_squared for row in summaries):.6f}")
    print("Combinations passing |c2| < |c1| for all three correlations: "
          f"{sum(row.all_quadratic_coefficients_smaller for row in summaries)}"
          f"/{len(summaries)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
