#!/usr/bin/env python3
"""Plot all completed low-J2 fixed-pinning continuations from Kuma and Izar.

For duplicated (signed h, D, J2) points, for example the D=8, |h|=0.08
overlap between Kuma and Izar, the lowest variational energy is plotted.
Every parsed candidate and the selected representative are written to CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D


REQUESTED_H = (-0.08, -0.04, -0.02, -0.01, 0.01, 0.02, 0.04, 0.08)
OBS_RE = re.compile(
    r"^D_(?P<D>\d+)_chi_(?P<chi>\d+)_energy_magnetization_correlation\.txt$"
)
ENERGY_RE = re.compile(
    r"^energy_per_site\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)",
    re.MULTILINE,
)
CORR_RE = re.compile(
    r"^corr_env(\d+)_([A-F]{2})\s*=\s*"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)",
    re.MULTILINE,
)
MAG_RE = re.compile(
    r"^mag_env(\d+)_([A-F])\s+"
    r"Sx=([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s+"
    r"Sy=([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s+"
    r"Sz=([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)",
    re.MULTILINE,
)

# The three physical NN directions, each averaged over its six equivalent
# bonds.  This is the same grouping used by the established 2C3 analysis.
NN_GROUPS = (
    ((1, "EB"), (1, "AD"), (1, "CF"),
     (3, "BE"), (3, "FC"), (3, "DA")),
    ((2, "CB"), (2, "AF"), (2, "ED"),
     (1, "FA"), (1, "DE"), (1, "BC")),
    ((3, "EF"), (3, "AB"), (3, "CD"),
     (2, "DC"), (2, "BA"), (2, "FE")),
)

RANK_STYLE = {
    "strongest": {"base": "#D73027", "marker": "o", "linestyle": "-"},
    "middle": {"base": "#1A9850", "marker": "s", "linestyle": "--"},
    "weakest": {"base": "#2166AC", "marker": "^", "linestyle": ":"},
}
DELTA_BASE = "#E66101"
MAGNETIZATION_BASE = "#762A83"


@dataclass(frozen=True)
class Observation:
    source: str
    path: str
    D: int
    chi: int
    J2: float
    signed_h: float
    branch: str
    energy_per_site: float
    strongest: float
    middle: float
    weakest: float
    delta: float
    staggered_magnetization: float
    staggered_magnetization_error: float


def read_hyperparams(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        return payload

    values: dict[str, object] = {}
    for line in text.splitlines():
        match = re.match(r"^([A-Za-z0-9_]+)\s*:\s*(.*?)\s*$", line)
        if not match:
            continue
        key, raw = match.groups()
        raw = raw.strip("'\"")
        try:
            values[key] = float(raw)
        except ValueError:
            values[key] = raw
    return values


def signed_field(branch: str, magnitude: float, path: Path) -> float:
    magnitude = abs(magnitude)
    if branch == "dimer-plaquette":
        return -magnitude
    if branch == "plaquette":
        return magnitude
    parts = set(path.parts)
    if any(part.startswith("h_m") for part in parts):
        return -magnitude
    if any(part.startswith("h_p") for part in parts):
        return magnitude
    raise ValueError(f"cannot infer signed field from branch={branch!r}")


def arithmetic_mean(values: list[float]) -> float:
    if not values:
        raise ValueError("empty correlation group")
    return sum(values) / len(values)


def publication_staggered_magnetization(
    magnetizations: dict[tuple[int, str], tuple[float, float, float]],
) -> tuple[float, float]:
    """Exact 2C3 central value and RMS spread from publication_common.py."""
    aligned: list[tuple[float, float, float]] = []
    for env in (1, 2, 3):
        for site in "ABCDEF":
            key = (env, site)
            if key not in magnetizations:
                continue
            sign = 1.0 if site in "ACE" else -1.0
            aligned.append(tuple(sign * value for value in magnetizations[key]))
    if not aligned:
        raise ValueError("missing site magnetizations")
    n_vectors = len(aligned)
    mean = tuple(
        sum(vector[component] for vector in aligned) / n_vectors
        for component in range(3)
    )
    # The 2C3 publication central value intentionally uses the real-tensor
    # x-z plane.  The spread retains all x/y/z components.
    central = math.hypot(mean[0], mean[2])
    squared_spread = sum(
        sum((vector[component] - mean[component]) ** 2 for component in range(3))
        for vector in aligned
    ) / n_vectors
    return central, math.sqrt(squared_spread)


def parse_observation(path: Path, source: str) -> Observation:
    name_match = OBS_RE.fullmatch(path.name)
    if name_match is None:
        raise ValueError("not a plain observation filename")
    D = int(name_match.group("D"))
    chi = int(name_match.group("chi"))

    hyperparams_path = path.parent / "hyperparams.yaml"
    if not hyperparams_path.is_file():
        raise ValueError("missing hyperparams.yaml")
    params = read_hyperparams(hyperparams_path)
    J2 = float(params["J2"])
    branch = str(params.get("vbc_branch", ""))
    field = signed_field(branch, float(params["vbc_field"]), path)

    text = path.read_text(encoding="utf-8", errors="replace")
    energy_match = ENERGY_RE.search(text)
    if energy_match is None:
        raise ValueError("missing energy_per_site")
    energy = float(energy_match.group(1))

    raw_corr = {
        (int(match.group(1)), match.group(2)): float(match.group(3))
        for match in CORR_RE.finditer(text)
    }
    raw_mag = {
        (int(match.group(1)), match.group(2)): (
            float(match.group(3)), float(match.group(4)), float(match.group(5))
        )
        for match in MAG_RE.finditer(text)
    }
    group_means: list[float] = []
    for group in NN_GROUPS:
        missing = [key for key in group if key not in raw_corr]
        if missing:
            raise ValueError(f"missing NN correlations {missing}")
        group_means.append(arithmetic_mean([raw_corr[key] for key in group]))
    strongest, middle, weakest = sorted(group_means)
    staggered_magnetization, staggered_magnetization_error = (
        publication_staggered_magnetization(raw_mag)
    )
    return Observation(
        source=source,
        path=str(path.resolve()),
        D=D,
        chi=chi,
        J2=J2,
        signed_h=field,
        branch=branch,
        energy_per_site=energy,
        strongest=strongest,
        middle=middle,
        weakest=weakest,
        delta=weakest - strongest,
        staggered_magnetization=staggered_magnetization,
        staggered_magnetization_error=staggered_magnetization_error,
    )


def is_requested_h(value: float) -> bool:
    return any(math.isclose(value, target, rel_tol=0.0, abs_tol=1.0e-10)
               for target in REQUESTED_H)


def discover(source: str, root: Path) -> tuple[list[Observation], list[str]]:
    rows: list[Observation] = []
    failures: list[str] = []
    if not root.is_dir():
        return rows, [f"{source}: root does not exist: {root}"]
    for path in sorted(root.rglob("D_*_chi_*_energy_magnetization_correlation.txt")):
        match = OBS_RE.fullmatch(path.name)
        if match is None:  # excludes all lookahead observations
            continue
        D = int(match.group("D"))
        chi = int(match.group("chi"))
        # Izar D=8 is permanently excluded from this project.  Keep this at
        # discovery time so it cannot enter either plots or audit CSV files,
        # even when an old local archive still contains those stages.
        if source == "Izar" and D == 8:
            continue
        # This is the exact completion condition used by run_one_stage.sh.
        best_tensor = path.parent / f"sweep_D{D}_chi{chi}_best.pt"
        if not best_tensor.is_file() or best_tensor.stat().st_size == 0:
            continue
        try:
            row = parse_observation(path, source)
            if is_requested_h(row.signed_h):
                rows.append(row)
        except (KeyError, OSError, TypeError, ValueError) as error:
            failures.append(f"{path}: {error}")
    return rows, failures


def point_key(row: Observation) -> tuple[float, int, float]:
    return (round(row.signed_h, 10), row.D, round(row.J2, 10))


def select_lowest_energy(rows: list[Observation]) -> tuple[list[Observation], dict[str, str]]:
    groups: dict[tuple[float, int, float], list[Observation]] = {}
    for row in rows:
        groups.setdefault(point_key(row), []).append(row)

    selected: list[Observation] = []
    reasons: dict[str, str] = {}
    for key in sorted(groups):
        candidates = sorted(
            groups[key],
            key=lambda row: (row.energy_per_site, -row.chi, row.source, row.path),
        )
        winner = candidates[0]
        selected.append(winner)
        reasons[winner.path] = "selected: lowest variational energy"
        for row in candidates[1:]:
            reasons[row.path] = (
                "duplicate coordinate: higher energy than "
                f"{winner.source} ({winner.energy_per_site:.12g})"
            )
    return selected, reasons


def write_csvs(
    rows: list[Observation],
    selected: list[Observation],
    reasons: dict[str, str],
    output: Path,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fields = list(Observation.__dataclass_fields__) + ["selected", "selection_reason"]
    selected_paths = {row.path for row in selected}
    with (output / "all_completed_observations.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (
            r.signed_h, r.D, r.J2, r.energy_per_site, r.source, r.path
        )):
            payload = asdict(row)
            payload["selected"] = row.path in selected_paths
            payload["selection_reason"] = reasons.get(row.path, "")
            writer.writerow(payload)

    selected_fields = list(Observation.__dataclass_fields__)
    with (output / "selected_lowest_energy_observations.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=selected_fields)
        writer.writeheader()
        for row in sorted(selected, key=lambda r: (r.signed_h, r.D, r.J2)):
            writer.writerow(asdict(row))


def mix_with_white(color: str, strength: float) -> tuple[float, float, float]:
    base = to_rgb(color)
    strength = min(1.0, max(0.0, strength))
    return tuple(1.0 - strength * (1.0 - component) for component in base)


def d_visual(D: int, d_min: int, d_max: int) -> tuple[float, float]:
    fraction = 1.0 if d_max == d_min else (D - d_min) / (d_max - d_min)
    color_strength = 0.48 + 0.52 * fraction
    alpha = 0.52 + 0.48 * fraction
    return color_strength, alpha


def h_token(value: float) -> str:
    sign = "m" if value < 0.0 else "p"
    return f"{sign}{abs(value):.2f}".replace(".", "p")


def h_math(value: float) -> str:
    return f"{value:+.2f}" if value > 0.0 else f"{value:.2f}"


def save_figure(fig: plt.Figure, output_stem: Path) -> None:
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=260, bbox_inches="tight")
    plt.close(fig)


def add_no_data(ax: plt.Axes, h: float) -> None:
    ax.text(
        0.5, 0.5,
        f"No completed observations yet for $h={h_math(h)}$",
        transform=ax.transAxes, ha="center", va="center", color="0.35",
    )
    ax.set_xlabel(r"$J_2/J_1$")


def plot_nn(rows: list[Observation], h: float, output: Path,
            d_min: int, d_max: int) -> None:
    subset = [row for row in rows if math.isclose(
        row.signed_h, h, rel_tol=0.0, abs_tol=1.0e-10
    )]
    fig, ax = plt.subplots(figsize=(7.4, 5.8))
    if not subset:
        add_no_data(ax, h)
    else:
        for D in sorted({row.D for row in subset}):
            points = sorted((row for row in subset if row.D == D), key=lambda row: row.J2)
            strength, alpha = d_visual(D, d_min, d_max)
            for rank in ("strongest", "middle", "weakest"):
                style = RANK_STYLE[rank]
                ax.plot(
                    [row.J2 for row in points],
                    [getattr(row, rank) for row in points],
                    color=mix_with_white(style["base"], strength),
                    alpha=alpha,
                    marker=style["marker"],
                    linestyle=style["linestyle"],
                    linewidth=1.55,
                    markersize=5.3,
                    markeredgewidth=0.8,
                    zorder=2 + D,
                )

        rank_handles = [
            Line2D(
                [0], [0], color=RANK_STYLE[rank]["base"],
                marker=RANK_STYLE[rank]["marker"],
                linestyle=RANK_STYLE[rank]["linestyle"],
                linewidth=1.7, markersize=5.5, label=rank.capitalize(),
            )
            for rank in ("strongest", "middle", "weakest")
        ]
        rank_legend = ax.legend(
            handles=rank_handles, loc="best", frameon=False, title="NN group"
        )
        ax.add_artist(rank_legend)
        d_handles = []
        for D in sorted({row.D for row in subset}):
            strength, alpha = d_visual(D, d_min, d_max)
            d_handles.append(Line2D(
                [0], [0], color=mix_with_white("#202020", strength), alpha=alpha,
                marker="o", linestyle="-", linewidth=1.55, markersize=5.0,
                label=f"D={D}",
            ))
        ax.legend(handles=d_handles, loc="lower right", frameon=False, title="Bond dimension")
        ax.set_ylabel(r"Ranked NN correlation $\langle \mathbf{S}_i\!\cdot\!\mathbf{S}_j\rangle$")
        ax.set_xlabel(r"$J_2/J_1$")
        ax.grid(alpha=0.18, linewidth=0.6)
        ax.margins(x=0.035, y=0.08)
    ax.set_title(rf"NN correlations at $h={h_math(h)}$")
    fig.tight_layout()
    save_figure(fig, output / f"NN_corr_vs_J2_h_{h_token(h)}")


def plot_delta(rows: list[Observation], h: float, output: Path,
               d_min: int, d_max: int) -> None:
    subset = [row for row in rows if math.isclose(
        row.signed_h, h, rel_tol=0.0, abs_tol=1.0e-10
    )]
    fig, ax = plt.subplots(figsize=(7.4, 5.8))
    if not subset:
        add_no_data(ax, h)
    else:
        for D in sorted({row.D for row in subset}):
            points = sorted((row for row in subset if row.D == D), key=lambda row: row.J2)
            strength, alpha = d_visual(D, d_min, d_max)
            ax.plot(
                [row.J2 for row in points],
                [row.delta for row in points],
                color=mix_with_white(DELTA_BASE, strength),
                alpha=alpha,
                marker="o",
                linestyle="-",
                linewidth=1.75,
                markersize=5.4,
                markeredgewidth=0.8,
                label=f"D={D}",
                zorder=2 + D,
            )
        ax.legend(loc="best", frameon=False, title="Bond dimension")
        ax.set_xlabel(r"$J_2/J_1$")
        ax.set_ylabel(r"$\Delta=C_{\rm weakest}-C_{\rm strongest}$")
        ax.set_ylim(bottom=0.0)
        ax.grid(alpha=0.18, linewidth=0.6)
        ax.margins(x=0.035, y=0.08)
    ax.set_title(rf"NN splitting at $h={h_math(h)}$")
    fig.tight_layout()
    save_figure(fig, output / f"Delta_vs_J2_h_{h_token(h)}")


def plot_staggered_magnetization(
    rows: list[Observation], h: float, output: Path, d_min: int, d_max: int,
) -> None:
    subset = [row for row in rows if math.isclose(
        row.signed_h, h, rel_tol=0.0, abs_tol=1.0e-10
    )]
    fig, ax = plt.subplots(figsize=(7.4, 5.8))
    if not subset:
        add_no_data(ax, h)
    else:
        for D in sorted({row.D for row in subset}):
            points = sorted(
                (row for row in subset if row.D == D), key=lambda row: row.J2
            )
            strength, alpha = d_visual(D, d_min, d_max)
            color = mix_with_white(MAGNETIZATION_BASE, strength)
            ax.errorbar(
                [row.J2 for row in points],
                [row.staggered_magnetization for row in points],
                yerr=[row.staggered_magnetization_error for row in points],
                color=color,
                ecolor=color,
                alpha=alpha,
                marker="o",
                linestyle="-",
                linewidth=1.75,
                markersize=5.4,
                markeredgewidth=0.8,
                elinewidth=0.9,
                capsize=2.3,
                capthick=0.9,
                label=f"D={D}",
                zorder=2 + D,
            )
        ax.legend(loc="best", frameon=False, title="Bond dimension")
        ax.set_xlabel(r"$J_2/J_1$")
        ax.set_ylabel(r"Staggered magnetization $m_{\rm stag}$")
        ax.set_ylim(bottom=0.0)
        ax.grid(alpha=0.18, linewidth=0.6)
        ax.margins(x=0.035, y=0.10)
    ax.set_title(rf"Staggered magnetization at $h={h_math(h)}$")
    fig.tight_layout()
    save_figure(fig, output / f"Staggered_magnetization_vs_J2_h_{h_token(h)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kuma", type=Path, action="append", required=True,
                        help="Kuma result root; repeat for additional bundles")
    parser.add_argument("--izar", type=Path, required=True,
                        help="Path to Results_Izar_lowJ2")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    all_rows: list[Observation] = []
    all_failures: list[str] = []
    per_source: dict[str, int] = {}
    sources: list[tuple[str, Path]] = []
    for index, root in enumerate(args.kuma, start=1):
        source = "Kuma" if len(args.kuma) == 1 else f"Kuma:{root.parent.name}"
        sources.append((source, root))
    sources.append(("Izar", args.izar))
    for source, root in sources:
        rows, failures = discover(source, root)
        all_rows.extend(rows)
        all_failures.extend(failures)
        per_source[source] = per_source.get(source, 0) + len(rows)

    selected, reasons = select_lowest_energy(all_rows)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csvs(all_rows, selected, reasons, args.output)

    observed_Ds = sorted({row.D for row in selected})
    d_min = min(observed_Ds) if observed_Ds else 6
    d_max = max(observed_Ds) if observed_Ds else 9
    plt.rcParams.update({
        "font.size": 12.0,
        "axes.labelsize": 13.0,
        "axes.titlesize": 13.5,
        "legend.fontsize": 9.2,
        "legend.title_fontsize": 9.5,
        "xtick.labelsize": 11.0,
        "ytick.labelsize": 11.0,
        "axes.linewidth": 0.9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    for h in REQUESTED_H:
        plot_nn(selected, h, args.output, d_min, d_max)
        plot_delta(selected, h, args.output, d_min, d_max)
        plot_staggered_magnetization(selected, h, args.output, d_min, d_max)

    print(
        "Completed observations: "
        + ", ".join(f"{source}={count}" for source, count in per_source.items())
    )
    print(f"Unique plotted (h,D,J2) points: {len(selected)}")
    print(f"Duplicate candidates rejected by variational energy: {len(all_rows) - len(selected)}")
    print(f"Output: {args.output.resolve()}")
    if all_failures:
        print(f"Warnings: {len(all_failures)}")
        for failure in all_failures[:12]:
            print(f"  {failure}")
    return 0 if selected else 2


if __name__ == "__main__":
    raise SystemExit(main())

