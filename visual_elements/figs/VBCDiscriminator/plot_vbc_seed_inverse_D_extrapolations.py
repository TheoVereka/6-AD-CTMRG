#!/usr/bin/env python3
"""Two-panel linear 1/D extrapolations for both selected VBC seed textures."""

from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from plot_j2_seed_continuations import (
    DATA,
    DEFAULT_INPUT,
    DEFAULT_LAST_IZAR_INPUT,
    DEFAULT_MANIFEST,
    DEFAULT_ORIGINAL_ROOT,
    DEFAULT_OUTPUT,
    J2_GRID,
    Point,
    RANK_COLORS,
    RANK_LABELS,
    SEED_MARKERS,
    Seed,
    TEXTURE_ORDER,
    TEXTURE_TITLES,
    discover_legacy,
    discover_last_izar,
    discover_original_d5_d6,
    discover_sep27,
    point_ranks,
    read_seeds,
    read_sep27_seeds,
    seed_point,
)


DEFAULT_DS = (5, 6, 7, 8, 9, 10, 11)
DEFAULT_SEP27_INPUT = DATA / "external" / "Working_AD_Honeycomb_Sep27"
SHADING_ALPHA = 0.025


def j2_tag(value: float) -> str:
    thousandths = int(round(value * 1000))
    decimals = 3 if thousandths % 10 else 2
    return f"{value:.{decimals}f}".replace(".", "p")


VARIANT_LINESTYLES = ("--", "-.", ":", (0, (5, 1, 1, 1)))


def aggregate_seed_replicas(rows: list[Point], D: int, seed_id: str,
                            *, connected: bool = False) -> dict:
    replicas = [row for row in rows if row.D == D and row.seed_id == seed_id]
    if not replicas:
        raise ValueError(f"missing D={D}, seed={seed_id}")
    rank_values = np.asarray(
        [[rank[0] for rank in point_ranks(row, connected)] for row in replicas],
        dtype=float,
    )
    rank_errors = np.asarray(
        [[rank[1] for rank in point_ranks(row, connected)] for row in replicas],
        dtype=float,
    )
    means = np.mean(rank_values, axis=0)
    # The error bars are diagnostics, not statistical sampling errors:
    # retain the intra-group RMS and insurance-replica spread.
    errors = np.sqrt(np.mean(
        rank_errors ** 2 + (rank_values - means[None, :]) ** 2, axis=0
    ))
    return {
        "D": D,
        "inverse_D": 1.0 / D,
        "seed_id": seed_id,
        "means": means,
        "errors": errors,
        "n_replicas": len(replicas),
    }


def build_variants(rows: list[Point], dimensions: tuple[int, ...],
                   *, connected: bool = False) -> list[list[dict]]:
    options = []
    for D in dimensions:
        seed_ids = sorted({row.seed_id for row in rows if row.D == D})
        if not seed_ids:
            raise ValueError(f"missing D={D}")
        options.append([
            aggregate_seed_replicas(rows, D, seed_id, connected=connected)
            for seed_id in seed_ids
        ])
    return [list(choice) for choice in itertools.product(*options)]


def linear_fit(x: np.ndarray, y: np.ndarray) -> dict:
    coefficients, covariance = np.polyfit(x, y, 1, cov=True)
    slope, intercept = map(float, coefficients)
    prediction = slope * x + intercept
    ss_res = float(np.sum((y - prediction) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "slope": slope,
        "intercept": intercept,
        "covariance": covariance,
        "intercept_stderr": float(math.sqrt(max(0.0, covariance[1, 1]))),
        "r_squared": math.nan if ss_tot == 0 else 1.0 - ss_res / ss_tot,
    }


def variant_label(rows: list[dict], seeds: dict[str, Seed]) -> str:
    # Historically only D=9 had multiple seeds.  LastIzar introduces distinct
    # D=5/6 Adam, pure-LBFGS, and adiabatic series, so those choices must also
    # appear in the label or several fit variants become indistinguishable.
    changing = [row for row in rows if row["D"] in {5, 6, 9}]
    if not changing:
        changing = [rows[-1]]
    return ", ".join(
        f"D{row['D']} {row['seed_id']} (seed $J_2={seeds[row['seed_id']].J2:g}$)"
        for row in changing
    )


def plot_panel(ax: plt.Axes, variants: list[list[dict]], texture: str,
               seeds: dict[str, Seed], *, connected: bool = False) -> list[dict]:
    x = np.asarray([row["inverse_D"] for row in variants[0]], dtype=float)
    x_fit = np.linspace(0.0, max(x) * 1.04, 300)
    summaries = []
    for variant_index, rows in enumerate(variants):
        linestyle = VARIANT_LINESTYLES[variant_index % len(VARIANT_LINESTYLES)]
        marker = SEED_MARKERS[variant_index % len(SEED_MARKERS)]
        fits = []
        for rank, color in enumerate(RANK_COLORS):
            y = np.asarray([row["means"][rank] for row in rows], dtype=float)
            yerr = np.asarray([row["errors"][rank] for row in rows], dtype=float)
            fit = linear_fit(x, y)
            fits.append(fit)
            y_fit = fit["slope"] * x_fit + fit["intercept"]
            design = np.column_stack((x_fit, np.ones_like(x_fit)))
            fit_sigma = np.sqrt(np.maximum(
                0.0, np.einsum("ij,jk,ik->i", design, fit["covariance"], design)
            ))
            ax.fill_between(x_fit, y_fit - fit_sigma, y_fit + fit_sigma,
                            color=color, alpha=SHADING_ALPHA, linewidth=0)
            ax.plot(x_fit, y_fit, linestyle=linestyle, color=color,
                    linewidth=1.25, alpha=0.9)
            ax.errorbar(x, y, yerr=None if connected else yerr,
                        fmt=marker, color=color,
                        markersize=5.0, elinewidth=0.9, capsize=2.5, zorder=4)
            ax.errorbar([0.0], [fit["intercept"]],
                        yerr=[fit["intercept_stderr"]], fmt="*", color=color,
                        markeredgecolor="black", markeredgewidth=0.5,
                        markersize=9.5, elinewidth=0.9, capsize=2.5, zorder=5)
        summaries.append({
            "label": variant_label(rows, seeds),
            "fits": fits,
            "linestyle": linestyle,
            "marker": marker,
        })

    for row in variants[0]:
        ax.annotate(f"D={row['D']}", (row["inverse_D"], row["means"][0]),
                    xytext=(0, -12), textcoords="offset points",
                    ha="center", va="top", fontsize=8, color="0.3")
    ax.axvline(0.0, color="0.35", linewidth=0.9, linestyle=":")
    ax.set_xlabel(r"$1/D$", fontsize=12)
    ax.grid(alpha=0.2)
    if len(summaries) > 1:
        handles = [
            Line2D([], [], color="0.3", marker=item["marker"],
                   linestyle=item["linestyle"], markersize=4.5,
                   linewidth=1.0, label=item["label"])
            for item in summaries
        ]
        ax.legend(handles=handles, loc="best", frameon=False, fontsize=7.2)
    return summaries


def make_figure(J2: float, by_texture: dict[str, list[list[dict]] | None],
                missing_by_texture: dict[str, list[int]], seeds: dict[str, Seed],
                output: Path, *, connected: bool = False) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.3), sharex=True, sharey=True,
                             constrained_layout=True)
    summaries = {}
    for ax, texture in zip(axes, TEXTURE_ORDER):
        ax.set_title(rf"{TEXTURE_TITLES[texture]}, $J_2={J2:g}$", fontsize=12)
        rows = by_texture[texture]
        if rows is None:
            ax.set_xlabel(r"$1/D$", fontsize=12)
            ax.grid(alpha=0.2)
            continue
        summaries[texture] = plot_panel(
            ax, rows, texture, seeds, connected=connected
        )
    inverse_D_max = max(
        row["inverse_D"]
        for variants in by_texture.values() if variants is not None
        for row in variants[0]
    )
    for ax in axes:
        ax.set_xlim(-0.004, inverse_D_max * 1.07)
    axes[0].set_ylabel(
        "connected NN correlation" if connected else "NN correlation",
        fontsize=12,
    )
    handles = [Line2D([], [], color=color, marker="o", linestyle="--",
                      markersize=5, linewidth=1.25, label=label)
               for color, label in zip(RANK_COLORS, RANK_LABELS)]
    handles.append(Line2D([], [], color="0.35", marker="*", linestyle="--",
                          markersize=8, linewidth=1.0,
                          label=r"linear fit; star: $1/D\to0$"))
    fig.legend(handles=handles, loc="outside upper center", ncol=3,
               frameon=False, fontsize=9)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)

    print(f"Saved {output}")
    for texture in TEXTURE_ORDER:
        if texture not in summaries:
            print(f"  {texture}: panel left empty; missing "
                  + ",".join(f"D={D}" for D in missing_by_texture[texture]))
            continue
        for variant in summaries[texture]:
            intercepts = ", ".join(
                f"r{rank}={fit['intercept']:.6f}+/-{fit['intercept_stderr']:.6f}"
                for rank, fit in enumerate(variant["fits"], start=1)
            )
            print(f"  {texture} [{variant['label']}]: {intercepts}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--sep27-input", type=Path, default=DEFAULT_SEP27_INPUT,
        help="Optional Kuma Sep27 bundle/result root to merge as D=10/11 data",
    )
    parser.add_argument(
        "--sep27-manifest", type=Path, default=None,
        help="Default: <sep27-input>/private_manifest.tsv",
    )
    parser.add_argument("--original-root", type=Path, default=DEFAULT_ORIGINAL_ROOT)
    parser.add_argument(
        "--last-izar-input", type=Path, default=DEFAULT_LAST_IZAR_INPUT,
        help="Optional LastIzar root (or its Results_LastIzar directory)",
    )
    parser.add_argument("--Ds", type=int, nargs="+", default=DEFAULT_DS)
    parser.add_argument("--J2", type=float, nargs="*", default=J2_GRID,
                        help="J2 values to consider; default: full continuation grid")
    args = parser.parse_args()

    dimensions = tuple(args.Ds)
    seeds = read_seeds(args.manifest)
    points, _ = discover_legacy(args.input, seeds)
    original_seeds, original_points, original_audit = discover_original_d5_d6(
        args.original_root
    )
    overlap = sorted(set(seeds) & set(original_seeds))
    if overlap:
        raise ValueError(f"duplicate original/continuation seed ids: {overlap}")
    seeds.update(original_seeds)
    points.extend(original_points)
    last_seeds, last_points, last_partial = discover_last_izar(
        args.last_izar_input, args.original_root
    )
    overlap = sorted(set(seeds) & set(last_seeds))
    if overlap:
        raise ValueError(f"duplicate LastIzar seed ids: {overlap}")
    seeds.update(last_seeds)
    points.extend(last_points)
    if args.last_izar_input.is_dir():
        print(
            f"Merged LastIzar data: {len(last_points)} continuation points, "
            f"{len(last_seeds)} series; {last_partial} incomplete h=0 stages ignored"
        )
    original_counts = {
        texture: sum(row["texture"] == texture for row in original_audit)
        for texture in ("dimer-plaquette", "plaquette", "mixed")
    }
    print(
        "Merged original D=5,6 states: "
        f"dimer={original_counts['dimer-plaquette']}, "
        f"plaquette={original_counts['plaquette']}, "
        f"mixed={original_counts['mixed']}"
    )
    if args.sep27_input.is_dir():
        sep27_manifest = (args.sep27_manifest
                          or args.sep27_input / "private_manifest.tsv")
        sep27_seeds = read_sep27_seeds(sep27_manifest)
        overlap = sorted(set(seeds) & set(sep27_seeds))
        if overlap:
            raise ValueError(f"duplicate seed ids across inputs: {overlap}")
        sep27_points, sep27_partial = discover_sep27(
            args.sep27_input, sep27_seeds
        )
        seeds.update(sep27_seeds)
        points.extend(sep27_points)
        print(
            f"Merged Sep27 data: {len(sep27_points)} completed stages; "
            f"{sep27_partial} partial stages ignored"
        )
    else:
        print(f"Sep27 input absent; using legacy data only: {args.sep27_input}")
    all_rows = [*points, *(seed_point(seed) for seed in seeds.values())]
    generated = 0
    for J2 in args.J2:
        by_texture: dict[str, list[list[dict]] | None] = {}
        connected_by_texture: dict[str, list[list[dict]] | None] = {}
        missing_by_texture: dict[str, list[int]] = {}
        for texture in TEXTURE_ORDER:
            selected = [row for row in all_rows
                        if row.seed_texture == texture
                        and row.D in dimensions
                        and math.isclose(row.J2, J2, abs_tol=1e-12)]
            present = {row.D for row in selected}
            absent = [D for D in dimensions if D not in present]
            missing_by_texture[texture] = absent
            present_dimensions = tuple(D for D in dimensions if D in present)
            if len(present_dimensions) < 3:
                by_texture[texture] = None
                connected_by_texture[texture] = None
                continue
            by_texture[texture] = build_variants(selected, present_dimensions)
            connected_by_texture[texture] = build_variants(
                selected, present_dimensions, connected=True
            )
        if all(rows is None for rows in by_texture.values()):
            missing = [
                f"{texture}: D={','.join(map(str, missing_by_texture[texture]))}"
                for texture in TEXTURE_ORDER
            ]
            print(f"Skipping J2={J2:g}; neither panel can be extrapolated ("
                  + "; ".join(missing) + ")")
            continue
        output = args.output_dir / (
            f"2C3_VBC_NN_ranks_vs_inverse_D_J2_{j2_tag(J2)}.pdf"
        )
        make_figure(J2, by_texture, missing_by_texture, seeds, output)
        connected_output = args.output_dir / (
            f"2C3_VBC_connected_NN_ranks_vs_inverse_D_J2_{j2_tag(J2)}.pdf"
        )
        make_figure(
            J2, connected_by_texture, missing_by_texture, seeds,
            connected_output, connected=True,
        )
        generated += 1
    if generated == 0:
        raise RuntimeError("no J2 value has at least three D values for either seed texture")
    print(
        f"Generated {generated} NN and {generated} connected-NN two-panel PDFs; "
        "no PNG/fit CSV files were written"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
