#!/usr/bin/env python3
"""Compare 0713-summary 2C3 energies with two restricted ansatze.

The figure contains four energy-per-site curves over 0.24 <= J2 <= 0.30:

* 2C3, D=8 (loaded from the 0713 summary)
* 2C3, D=9 (loaded from the 0713 summary)
* less restricted, D=10 (the supplied Neel data)
* more restricted, D=8 (the supplied plaquette data)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


DEFAULT_DATA_DIR = Path(
    r"D:\HyraiOn\ENS_Lyon\Internship\2026-EPFL\data\0713summary"
)
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent
    / "analysis_plots_0713summary"
    / "energy_twoC3_D8_D9_restrictions.pdf"
)
DEFAULT_DIFFERENCE_OUTPUT = (
    Path(__file__).resolve().parent
    / "analysis_plots_0713summary"
    / "energy_difference_twoC3_vs_less_restricted_D10.pdf"
)

J2_MIN = 0.24
J2_MAX = 0.30
TWO_C3_ANSATZ = "2tensor_twoC3"

# Supplied as "Neel"; plotted with the requested legend name.
LESS_RESTRICTED_D10 = {
    0.24: -0.440164615,
    0.26: -0.434247830,
    0.28: -0.428867853,
    0.30: -0.424076147,
}

# Supplied as "plaquette"; plotted with the requested legend name.
MORE_RESTRICTED_D8 = {
    0.24: -0.439142524,
    0.26: -0.432680783,
    0.28: -0.426623461,
    0.30: -0.421025828,
}


def parse_j2_folder(name: str) -> float | None:
    """Parse a summary directory name such as J2_0p24."""
    if not name.startswith("J2_"):
        return None
    try:
        return float(name.removeprefix("J2_").replace("p", "."))
    except ValueError:
        return None


def load_two_c3_curve(data_dir: Path, D: int) -> tuple[list[float], list[float]]:
    """Load one 2C3 fixed-D curve from summary manifest files."""
    points: list[tuple[float, float]] = []
    for j2_dir in data_dir.iterdir():
        if not j2_dir.is_dir():
            continue
        j2 = parse_j2_folder(j2_dir.name)
        if j2 is None or not J2_MIN <= j2 <= J2_MAX:
            continue

        manifest_path = (
            j2_dir / TWO_C3_ANSATZ / f"D_{D}" / "manifest.json"
        )
        if not manifest_path.is_file():
            continue
        with manifest_path.open(encoding="utf-8") as stream:
            manifest = json.load(stream)
        points.append((j2, float(manifest["energy_per_site"])))

    points.sort()
    if not points:
        raise FileNotFoundError(
            f"No {TWO_C3_ANSATZ} D={D} summary data found in {data_dir}"
        )
    return [point[0] for point in points], [point[1] for point in points]


def load_two_c3_energy(data_dir: Path, j2: float, D: int) -> float:
    """Load one 2C3 energy, allowing canonical summary names such as 0p3."""
    for j2_dir in data_dir.iterdir():
        folder_j2 = parse_j2_folder(j2_dir.name) if j2_dir.is_dir() else None
        if folder_j2 is None or abs(folder_j2 - j2) > 1e-12:
            continue
        manifest_path = (
            j2_dir / TWO_C3_ANSATZ / f"D_{D}" / "manifest.json"
        )
        if manifest_path.is_file():
            with manifest_path.open(encoding="utf-8") as stream:
                return float(json.load(stream)["energy_per_site"])
    raise FileNotFoundError(f"No {TWO_C3_ANSATZ} J2={j2:g}, D={D} data in {data_dir}")


def mapping_as_xy(data: dict[float, float]) -> tuple[list[float], list[float]]:
    points = sorted(
        (j2, energy)
        for j2, energy in data.items()
        if J2_MIN <= j2 <= J2_MAX
    )
    return [point[0] for point in points], [point[1] for point in points]


def plot_energy_comparison(data_dir: Path, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    curve_specs = [
        (*load_two_c3_curve(data_dir, 8), "2C3 (D=8)", "tab:blue", "o", "-"),
        (*load_two_c3_curve(data_dir, 9), "2C3 (D=9)", "tab:orange", "s", "-"),
        (
            *mapping_as_xy(LESS_RESTRICTED_D10),
            "less restricted (D=10)",
            "tab:green",
            "^",
            "--",
        ),
        (
            *mapping_as_xy(MORE_RESTRICTED_D8),
            "more restricted (D=8)",
            "tab:red",
            "D",
            "--",
        ),
    ]

    for j2s, energies, label, color, marker, linestyle in curve_specs:
        ax.plot(
            j2s,
            energies,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.5,
            markersize=6,
            label=label,
        )

    ax.set_xlim(J2_MIN, J2_MAX)
    ax.set_xticks([0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30])
    ax.set_xlabel(r"$J_2$")
    ax.set_ylabel("Energy per site")
    ax.set_title("2C3 and restricted-ansatz energy comparison")
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.5f"))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}")


def plot_energy_difference(data_dir: Path, output: Path) -> None:
    """Plot E_2C3(D) - E_less-restricted(D=10) against 1/D."""
    Ds = [6, 7, 8, 9]
    styles = [
        (0.24, "tab:blue", "o"),
        (0.26, "tab:orange", "s"),
        (0.28, "tab:green", "^"),
        (0.30, "tab:red", "D"),
    ]

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for j2, color, marker in styles:
        reference_energy = LESS_RESTRICTED_D10[j2]
        inverse_D = [1.0 / D for D in Ds]
        differences = [
            load_two_c3_energy(data_dir, j2, D) - reference_energy
            for D in Ds
        ]
        ax.plot(
            inverse_D,
            differences,
            color=color,
            marker=marker,
            linewidth=1.5,
            markersize=6,
            label=rf"$J_2={j2:.2f}$",
        )

    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_xlim(left=0.1)
    ax.set_xticks([1.0 / D for D in reversed(Ds)])
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, -3))
    ax.set_xlabel(r"$1/D$")
    ax.set_ylabel(
        r"$E_{\mathrm{2C3}}(D)-E_{\mathrm{less\ restricted}}(D=10)$"
    )
    ax.set_title("Energy-per-site difference")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--difference-output", type=Path, default=DEFAULT_DIFFERENCE_OUTPUT
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_energy_comparison(args.data_dir, args.output)
    plot_energy_difference(args.data_dir, args.difference_output)
