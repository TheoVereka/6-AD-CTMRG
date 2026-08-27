#!/usr/bin/env python3
"""Plot D345678910 legacy-Neel observables with ordinary correlation lengths."""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import plot_analysis_Windows as base


DATA_ROOT = Path(r"D:\HyraiOn\ENS_Lyon\Internship\2026-EPFL\data\D345678910")
OUT_DIR = Path(__file__).resolve().parent / "analysis_plots_neel_legacy"
FOLDER_RE = re.compile(
    r"^(?:neel_symmetrized|neel_legacy)__J2_([0-9p]+)_", re.IGNORECASE
)
OBS_RE = re.compile(r"^D_(\d+)_chi_(\d+)_energy_magnetization_correlation\.txt$")
D_MIN, D_MAX = 3, 11


def parse_j2(label: str) -> float:
    return round(float(label.replace("p", ".")), 6)


def selected_observables() -> dict[float, dict[int, tuple[Path, dict]]]:
    """Select each (J2,D) independently across all surviving legacy folders."""
    choices = {}
    for folder in sorted(DATA_ROOT.iterdir()):
        folder_match = FOLDER_RE.match(folder.name)
        if folder_match is None or not folder.is_dir():
            continue
        j2 = parse_j2(folder_match.group(1))
        for path in sorted(folder.iterdir()):
            observable_match = OBS_RE.match(path.name)
            if observable_match is None:
                continue
            D, chi = int(observable_match.group(1)), int(observable_match.group(2))
            if not D_MIN <= D <= D_MAX:
                continue
            try:
                parsed = base.parse_plain_file(str(path))
            except (OSError, AttributeError, TypeError, ValueError):
                continue
            rank = (float(parsed["energy_per_site"]), chi, str(path).lower())
            key = (j2, D)
            if key not in choices or rank < choices[key][0]:
                choices[key] = (rank, folder, parsed)
    selected = {}
    for (j2, D), (_rank, folder, parsed) in sorted(choices.items()):
        selected.setdefault(j2, {})[D] = (folder, parsed)
    return selected


def inverse_xi_entry(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    directional = {}
    for direction in ("env2", "env1_ab_env3_ba", "env3_ab_env1_ba"):
        values = payload["spectra"][direction]["eigenvalues"][:2]
        magnitudes = sorted(
            (math.hypot(float(value["real"]), float(value["imag"])) for value in values),
            reverse=True,
        )
        directional[direction] = math.log(magnitudes[0] / magnitudes[1])
    lower, center, upper = sorted(directional.values())
    return {
        "inverse_xi": center,
        "inverse_xi_lower": lower,
        "inverse_xi_upper": upper,
        "inverse_xi_lower_error": center - lower,
        "inverse_xi_upper_error": upper - center,
        "directional_inverse_xi": directional,
        "source": str(path),
    }


def aligned_magnetization_statistics(
    D_data: dict[int, dict],
) -> dict[int, tuple[float, float]]:
    """Return |mean aligned vector| and the 3D RMS spread around it.

    Every one of the 3 environments contributes all 6 site magnetization
    vectors. B/D/F vectors are reversed, so the resulting 18 vectors estimate
    the same staggered Neel vector. The scalar error is the root-mean-square
    Euclidean distance of those vectors from their mean vector.
    """
    statistics = {}
    for D, values in D_data.items():
        mag = values.get("mag", {})
        aligned = []
        for env in (1, 2, 3):
            for site in "ABCDEF":
                entry = mag.get((env, site))
                if entry is None:
                    continue
                sign = 1.0 if site in "ACE" else -1.0
                aligned.append(
                    sign
                    * np.asarray(
                        [entry["Sx"], entry["Sy"], entry["Sz"]], dtype=float
                    )
                )
        if not aligned:
            statistics[D] = (float("nan"), float("nan"))
            continue
        vectors = np.vstack(aligned)
        mean_vector = np.mean(vectors, axis=0)
        magnitude = float(np.linalg.norm(mean_vector))
        deviations = vectors - mean_vector
        error = float(np.sqrt(np.mean(np.sum(deviations * deviations, axis=1))))
        statistics[D] = (magnitude, error)
    return statistics


def load_all() -> dict[float, dict[str, dict]]:
    output = {}
    for j2, points in sorted(selected_observables().items()):
        D_data = {D: parsed for D, (_folder, parsed) in points.items()}
        if not D_data:
            continue
        values = base.process_parsed_D_data(D_data)
        mag_statistics = aligned_magnetization_statistics(D_data)
        values["mneel_list"] = [mag_statistics[D][0] for D in values["Ds"]]
        values["mneel_error_by_D"] = {
            D: mag_statistics[D][1] for D in values["Ds"]
        }
        m_lin2, m_lin3, m_c2, m_c3 = base.compute_mag_extrap(
            values["Ds"], values["mneel_list"]
        )
        values.update(
            {
                "m_lin2": m_lin2,
                "m_lin3": m_lin3,
                "m_c2": m_c2,
                "m_c3": m_c3,
            }
        )
        values["inverse_correlation_lengths"] = {}
        for D in values["Ds"]:
            folder = points[D][0]
            correlation = folder / f"correlation_length_D_{D}.json"
            if correlation.is_file():
                try:
                    values["inverse_correlation_lengths"][D] = inverse_xi_entry(correlation)
                except (OSError, KeyError, TypeError, ValueError, ZeroDivisionError, json.JSONDecodeError) as exc:
                    print(f"WARNING: skipped {correlation}: {exc}")
        output[j2] = {"neel_symmetrized": values}
    return output


def delta_and_error(entry: dict) -> tuple[float, float]:
    rank1 = next(index for index, rank in enumerate(entry["ranks"]) if rank == 1)
    rank3 = next(index for index, rank in enumerate(entry["ranks"]) if rank == 3)
    delta = abs(float(entry["means"][rank3]) - float(entry["means"][rank1]))
    error = math.sqrt(float(entry["stds"][rank1]) ** 2 + float(entry["stds"][rank3]) ** 2) / math.sqrt(2.0)
    return delta, error


def plot_scatter_column(axes, values: dict) -> None:
    correlation = values["inverse_correlation_lengths"]
    energies = dict(zip(values["Ds"], values["energy_per_site"]))
    mags = dict(zip(values["Ds"], values["mneel_list"]))
    for row, (ylabel, color) in enumerate(
        (("Energy per site", "#1f77b4"), (r"m$_\mathrm{N\acute{e}el}$", "#9467bd"), (r"$\Delta_\mathrm{NN}$", "tab:orange"))
    ):
        x, xlo, xhi, y, yerr = [], [], [], [], []
        for D in sorted(correlation):
            xi = correlation[D]
            if row == 0 and D in energies:
                value, error = energies[D], 0.0
            elif row == 1 and D in mags:
                value = mags[D]
                error = values["mneel_error_by_D"].get(D, 0.0)
            elif row == 2 and D in values["nn_groups"]:
                value, error = delta_and_error(values["nn_groups"][D])
            else:
                continue
            x.append(xi["inverse_xi"])
            xlo.append(xi["inverse_xi_lower_error"])
            xhi.append(xi["inverse_xi_upper_error"])
            y.append(value)
            yerr.append(error)
        ax = axes[row]
        if x:
            ax.errorbar(x, y, xerr=[xlo, xhi], yerr=yerr, fmt="o", color=color, capsize=3)
        ax.set_xlim(left=0.0)
        if row == 1:
            ax.set_ylim(bottom=0.0)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.18)
    axes[2].set_xlabel(r"$1/\xi$")


def plot_per_j2(j2: float, values: dict) -> None:
    figure, axes = plt.subplots(4, 2, figsize=(10.2, 13.8), gridspec_kw={"hspace": 0.28, "wspace": 0.30})
    base.plot_col_energy(axes[0, 0], values, show_xlabel=False)
    base.plot_col_mag(axes[1, 0], values, show_xlabel=False)
    Ds = values["Ds"]
    axes[1, 0].errorbar(
        [1.0 / D for D in Ds],
        values["mneel_list"],
        yerr=[values["mneel_error_by_D"].get(D, 0.0) for D in Ds],
        fmt="none", ecolor="#9467bd", capsize=2, alpha=0.75,
    )
    axes[1, 0].set_ylim(bottom=0.0)
    base.plot_col_nn(axes[2, 0], values, show_xlabel=True)
    if values["inverse_correlation_lengths"]:
        base.plot_col_inverse_xi(axes[3, 0], values)
        axes[3, 0].set_ylim(bottom=0.0)
    else:
        axes[3, 0].axis("off")
    plot_scatter_column(axes[:3, 1], values)
    axes[3, 1].axis("off")
    axes[0, 0].set_title("vs $1/D$")
    axes[0, 1].set_title("observables vs $1/\\xi$")
    axes[0, 0].set_ylabel("Energy per site")
    axes[1, 0].set_ylabel(r"m$_\mathrm{N\acute{e}el}$")
    axes[2, 0].set_ylabel(r"$\langle S_i\cdot S_j\rangle$ (NN)")
    figure.suptitle(rf"legacy Néel, $J_2={j2:g}$")
    base._save(figure, str(OUT_DIR / f"J2_{base._j2_fname(j2)}.pdf"))


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_data = load_all()
    if not all_data:
        raise RuntimeError(f"No legacy Neel observations found below {DATA_ROOT}")
    for j2, ansatz_map in sorted(all_data.items()):
        plot_per_j2(j2, ansatz_map["neel_symmetrized"])
    base.plot_e_vs_j2_figures(all_data, str(OUT_DIR))
    base.plot_m_vs_j2_figures(all_data, str(OUT_DIR))
    print(f"Wrote per-J2 plots and six summary plots to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
