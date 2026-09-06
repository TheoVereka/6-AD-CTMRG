"""Shared data, fitting, plotting, and export machinery for figures 1--28."""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.lines import Line2D
import numpy as np
from scipy.optimize import least_squares

import publication_config as cfg


ANSATZ_NEEL = "Neel"
ANSATZ_TWOC3 = "2C3"
TWOC3_DIRECTORY = "2tensor_twoC3"
NEEL_FOLDER_RE = re.compile(
    r"^(?:neel_symmetrized|neel_legacy)__J2_([0-9p]+)_", re.IGNORECASE
)
OBS_FILE_RE = re.compile(
    r"^D_(\d+)_chi_(\d+)_energy_magnetization_correlation\.txt$"
)
ENERGY_RE = re.compile(
    r"^energy_per_site\s*=\s*([+-]?[\d.eE+-]+)", re.MULTILINE
)
CORR_RE = re.compile(
    r"^corr_env(\d+)_([A-F]{2})\s*=\s*([+-]?[\d.eE+-]+)", re.MULTILINE
)
MAG_RE = re.compile(
    r"^mag_env(\d+)_([A-F])\s+Sx=([+-]?[\d.eE+-]+)\s+"
    r"Sy=([+-]?[\d.eE+-]+)\s+Sz=([+-]?[\d.eE+-]+)",
    re.MULTILINE,
)
NN_GROUPS = (
    ((1, "EB"), (1, "AD"), (1, "CF"), (3, "BE"), (3, "FC"), (3, "DA")),
    ((2, "CB"), (2, "AF"), (2, "ED"), (1, "FA"), (1, "DE"), (1, "BC")),
    ((3, "EF"), (3, "AB"), (3, "CD"), (2, "DC"), (2, "BA"), (2, "FE")),
)

PURPLE = "#6f2dbd"
ORANGE = "#d95f02"
BLUE = "#2166ac"
RED = "#b2182b"

# Finite-D palettes follow the light-to-dark family used by
# summary_deltaNNN_vs_J2.pdf, rather than encoding D only through alpha.
PURPLE_D_COLORS = {
    3: "#fde0ff", 4: "#e7b5ff", 5: "#cf72e8", 6: "#b536c9",
    7: "#8c2fc2", 8: "#6339bd", 9: "#3f51b5", 10: "#174ea6",
    11: "#082567",
}
ORANGE_D_COLORS = {
    3: "#fbfabc", 4: "#ffefa9", 5: "#fed78d", 6: "#f9b664",
    7: "#db7a24", 8: "#ae4829", 9: "#731608", 10: "#3b0202",
    11: "#160000",
}


NEEL_ENERGY_CMAP = colors.LinearSegmentedColormap.from_list(
    "publication_purple_to_blue",
    ("#3b0f70", "#365c9d", "#38a6d8", "#79d4f2"),
)
TWOC3_ENERGY_CMAP = colors.LinearSegmentedColormap.from_list(
    "publication_red_to_dark_orange",
    ("#6C0B13", "#b50802", "#fe5821", "#e4975f"),
)


def _key(j2: float, D: int) -> tuple[float, int]:
    return round(float(j2), 6), int(D)


def _banned(ansatz: str, j2: float, D: int) -> bool:
    return (
        int(D) in cfg.GLOBAL_BANNED_DS[ansatz]
        or _key(j2, D) in {_key(*point) for point in cfg.GLOBAL_BANS[ansatz]}
    )


def _fit_banned(figure: int, j2: float, D: int) -> bool:
    return _key(j2, D) in {_key(*point) for point in cfg.FIT_BANS.get(figure, set())}


def _plot_allowed(j2: float, ansatz: str) -> bool:
    return round(float(j2), 6) not in {
        round(float(value), 6) for value in cfg.PLOT_BANNED_J2[ansatz]
    }


def rms(values: list[float] | np.ndarray) -> float:
    """sqrt(sum((x-xbar)^2)/N), exactly as requested."""
    array = np.asarray(values, dtype=float)
    if not len(array):
        return float("nan")
    return float(np.sqrt(np.sum((array - np.mean(array)) ** 2) / len(array)))


def parse_observable(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    energy_match = ENERGY_RE.search(text)
    if energy_match is None:
        raise ValueError(f"No energy in {path}")
    corr = {
        (int(match.group(1)), match.group(2)): float(match.group(3))
        for match in CORR_RE.finditer(text)
    }
    mag = {
        (int(match.group(1)), match.group(2)): np.asarray(
            [float(match.group(3)), float(match.group(4)), float(match.group(5))]
        )
        for match in MAG_RE.finditer(text)
    }
    return {"E": float(energy_match.group(1)), "corr": corr, "mag": mag}


def magnetization(observation: dict, *, legacy_neel: bool) -> tuple[float, float]:
    mag = observation["mag"]
    aligned = []
    for env in (1, 2, 3):
        for site in "ABCDEF":
            if (env, site) in mag:
                aligned.append((1.0 if site in "ACE" else -1.0) * mag[(env, site)])
    if not aligned:
        return float("nan"), float("nan")
    vectors = np.vstack(aligned)
    mean_vector = np.mean(vectors, axis=0)
    error = float(np.sqrt(np.sum((vectors - mean_vector) ** 2) / len(vectors)))
    if legacy_neel:
        central = float(np.linalg.norm(mean_vector))
    else:
        # This is the central-value definition used by plot_analysis_Windows.
        central = float(np.linalg.norm(mean_vector[[0, 2]]))
    return central, error


def delta(observation: dict) -> tuple[float, float]:
    groups = []
    for group in NN_GROUPS:
        values = [observation["corr"][item] for item in group if item in observation["corr"]]
        if len(values) == 6:
            groups.append((float(np.mean(values)), rms(values), float(np.sum(values))))
    if len(groups) != 3:
        return float("nan"), float("nan")
    order = np.argsort([entry[0] for entry in groups])
    rank1, rank3 = groups[int(order[0])], groups[int(order[-1])]
    central = (rank3[2] - rank1[2]) / 24.0
    error = math.sqrt(rank3[1] ** 2 + rank1[1] ** 2) / 2.0
    return float(central), float(error)


def inverse_xi(path: Path) -> tuple[float, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    directional = []
    for direction in ("env2", "env1_ab_env3_ba", "env3_ab_env1_ba"):
        eigenvalues = payload["spectra"][direction]["eigenvalues"][:2]
        magnitudes = sorted(
            [math.hypot(float(value["real"]), float(value["imag"])) for value in eigenvalues],
            reverse=True,
        )
        if len(magnitudes) != 2 or magnitudes[1] <= 0.0:
            raise ValueError(f"Invalid transfer spectrum in {path}")
        directional.append(math.log(magnitudes[0] / magnitudes[1]))
    return float(np.mean(directional)), rms(directional)


def _record(ansatz: str, j2: float, D: int, observation: dict, xi_path: Path | None) -> dict:
    m, m_error = magnetization(observation, legacy_neel=ansatz == ANSATZ_NEEL)
    dlt, dlt_error = delta(observation)
    inv_xi = inv_xi_error = float("nan")
    if xi_path is not None and xi_path.is_file():
        try:
            inv_xi, inv_xi_error = inverse_xi(xi_path)
        except (OSError, KeyError, TypeError, ValueError, ZeroDivisionError, json.JSONDecodeError):
            pass
    return {
        "ansatz": ansatz, "J2": float(j2), "D": int(D),
        "E": observation["E"], "m": m, "m_error": m_error,
        "delta": dlt, "delta_error": dlt_error,
        "inverse_xi": inv_xi, "inverse_xi_error": inv_xi_error,
    }


def load_neel() -> list[dict]:
    choices: dict[tuple[float, int], tuple[tuple, Path, dict]] = {}
    if not cfg.NEEL_DATA_ROOT.is_dir():
        return []
    for folder in sorted(cfg.NEEL_DATA_ROOT.iterdir()):
        match = NEEL_FOLDER_RE.match(folder.name)
        if match is None or not folder.is_dir():
            continue
        j2 = round(float(match.group(1).replace("p", ".")), 6)
        for path in folder.iterdir():
            file_match = OBS_FILE_RE.match(path.name)
            if file_match is None:
                continue
            D, chi = int(file_match.group(1)), int(file_match.group(2))
            if not cfg.MIN_D <= D <= cfg.MAX_D or _banned(ANSATZ_NEEL, j2, D):
                continue
            try:
                observation = parse_observable(path)
            except (OSError, ValueError):
                continue
            rank = (observation["E"], -chi, str(path).lower())
            if (j2, D) not in choices or rank < choices[(j2, D)][0]:
                choices[(j2, D)] = (rank, folder, observation)
    return [
        _record(ANSATZ_NEEL, j2, D, observation, folder / f"correlation_length_D_{D}.json")
        for (j2, D), (_rank, folder, observation) in sorted(choices.items())
    ]


def load_twoc3() -> list[dict]:
    records = []
    if not cfg.TWOC3_DATA_ROOT.is_dir():
        return records
    for j2_folder in sorted(cfg.TWOC3_DATA_ROOT.glob("J2_*")):
        if not j2_folder.is_dir():
            continue
        try:
            j2 = round(float(j2_folder.name[3:].replace("p", ".")), 6)
        except ValueError:
            continue
        ansatz_folder = j2_folder / TWOC3_DIRECTORY
        if not ansatz_folder.is_dir():
            continue
        for d_folder in sorted(ansatz_folder.glob("D_*")):
            try:
                D = int(d_folder.name[2:])
            except ValueError:
                continue
            if not cfg.MIN_D <= D <= cfg.MAX_D or _banned(ANSATZ_TWOC3, j2, D):
                continue
            path = d_folder / "energy_magnetization_correlation.txt"
            if not path.is_file():
                continue
            try:
                observation = parse_observable(path)
            except (OSError, ValueError):
                continue
            records.append(_record(
                ANSATZ_TWOC3, j2, D, observation, d_folder / "correlation_length.json"
            ))
    return records


def load_dataset() -> dict[str, list[dict]]:
    return {ANSATZ_NEEL: load_neel(), ANSATZ_TWOC3: load_twoc3()}


def _finite(rows: list[dict], *fields: str) -> list[dict]:
    return [row for row in rows if all(np.isfinite(row[field]) for field in fields)]


def _groups(rows: list[dict], field: str) -> dict[float | int, list[dict]]:
    output = {}
    for row in rows:
        output.setdefault(row[field], []).append(row)
    return {key: sorted(values, key=lambda item: (item["J2"], item["D"])) for key, values in sorted(output.items())}


def _safe_sigma(values: np.ndarray) -> np.ndarray:
    positive = values[np.isfinite(values) & (values > 0.0)]
    fallback = float(np.median(positive)) if len(positive) else 1.0
    return np.where(np.isfinite(values) & (values > 0.0), values, fallback)


def fit_m_xi(rows: list[dict], *, absolute: bool, power: bool) -> dict | None:
    rows = _finite(rows, "inverse_xi", "inverse_xi_error", "m", "m_error")
    minimum = 3 if power else 2
    if len(rows) < minimum:
        return None
    x = np.asarray([row["inverse_xi"] for row in rows])
    sx = _safe_sigma(np.asarray([row["inverse_xi_error"] for row in rows]))
    y = np.asarray([row["m"] for row in rows])
    sy = _safe_sigma(np.asarray([row["m_error"] for row in rows]))
    slope, intercept = np.polyfit(x, y, 1)

    def model(parameters, xx):
        m0, c = parameters[:2]
        alpha = parameters[2] if power else 1.0
        return m0 + c * np.maximum(xx, 0.0) ** alpha

    def residual(parameters):
        alpha = parameters[2] if power else 1.0
        c = parameters[1]
        derivative = c * alpha * np.maximum(x, 1e-15) ** (alpha - 1.0)
        sigma = np.sqrt(sy**2 + (derivative * sx) ** 2)
        return (y - model(parameters, x)) / sigma

    lower = [0.0 if absolute else -np.inf, 0.0]
    upper = [np.inf, np.inf]
    guesses = [[
        max(0.0, intercept) if absolute else intercept,
        max(float(slope), 1e-8),
    ]]
    if power:
        lower.append(cfg.POWER_ALPHA_BOUNDS[0])
        upper.append(cfg.POWER_ALPHA_BOUNDS[1])
        guesses = [guess + [alpha] for alpha in (0.5, 1.0, 1.5, 2.0) for guess in guesses]
    best = None
    for guess in guesses:
        try:
            result = least_squares(
                residual, guess, bounds=(lower, upper), max_nfev=cfg.FIT_MAX_NFEV
            )
            if result.success and (best is None or np.sum(result.fun**2) < np.sum(best.fun**2)):
                best = result
        except ValueError:
            continue
    if best is None:
        return None
    covariance = _covariance(best, len(rows), absolute_sigma=True)
    errors = np.sqrt(np.maximum(0.0, np.diag(covariance)))
    names = ["m0", "c"] + (["alpha"] if power else [])
    return {"names": names, "values": best.x, "errors": errors, "model": model, "n": len(rows)}


def _covariance(result, n_data: int, *, absolute_sigma: bool = False) -> np.ndarray:
    try:
        inverse = np.linalg.pinv(result.jac.T @ result.jac)
        if absolute_sigma:
            return inverse
        dof = max(1, n_data - len(result.x))
        return inverse * (2.0 * result.cost / dof)
    except np.linalg.LinAlgError:
        return np.full((len(result.x), len(result.x)), np.nan)


def fit_linear_invD(rows: list[dict]) -> dict | None:
    rows = _finite(rows, "m", "m_error")
    if len(rows) < 2:
        return None
    x = np.asarray([1.0 / row["D"] for row in rows])
    y = np.asarray([row["m"] for row in rows])
    sigma = _safe_sigma(np.asarray([row["m_error"] for row in rows]))
    design = np.column_stack([np.ones(len(x)), x])
    weight = 1.0 / sigma**2
    normal = design.T @ (weight[:, None] * design)
    covariance = np.linalg.pinv(normal)
    parameters = covariance @ (design.T @ (weight * y))
    return {"names": ["m0", "c"], "values": parameters,
            "errors": np.sqrt(np.maximum(0.0, np.diag(covariance))), "n": len(rows)}


def fit_energy(rows: list[dict], *, gapped: bool) -> dict | None:
    rows = _finite(rows, "E")
    if len(rows) < 3:
        return None
    x = np.asarray([1.0 / row["D"] for row in rows])
    y = np.asarray([row["E"] for row in rows])

    def model(parameters, xx):
        E0, k, a = parameters
        if gapped:
            return E0 + k * np.exp(-a / np.maximum(xx, 1e-15))
        return E0 + k * np.maximum(xx, 0.0) ** a

    scale = max(float(np.ptp(y)), 1e-10)
    guesses = []
    for a0 in (0.25, 0.5, 1.0, 2.0, 4.0):
        basis = np.exp(-a0 / x) if gapped else x**a0
        c1, c0 = np.polyfit(basis, y, 1)
        guesses.append([c0, c1, a0])
    best = None
    for guess in guesses:
        result = least_squares(
            lambda p: (y - model(p, x)) / scale,
            guess,
            bounds=([-np.inf, -np.inf, 1e-6], [np.inf, np.inf, 20.0 if gapped else 8.0]),
            max_nfev=cfg.FIT_MAX_NFEV,
        )
        if result.success and (best is None or result.cost < best.cost):
            best = result
    if best is None:
        return None
    covariance = _covariance(best, len(rows))
    return {"names": ["E0", "k", "a"], "values": best.x,
            "errors": np.sqrt(np.maximum(0.0, np.diag(covariance))),
            "model": model, "n": len(rows)}


def delta_statistic(rows: list[dict]) -> dict | None:
    rows = sorted(_finite(rows, "delta", "delta_error"), key=lambda row: row["D"])
    rows = rows[-cfg.DELTA_STATISTIC_N_LARGEST_D:]
    if len(rows) < 2:
        return None
    values = np.asarray([row["delta"] for row in rows])
    errors = np.asarray([row["delta_error"] for row in rows])
    # The three largest-D values are equal contributors.  This is a
    # conservative uncertainty of the representative constant, not the
    # uncertainty of an infinitely repeatable weighted mean:
    #   sigma_total^2 = mean(sigma_i^2) + mean((Delta_i - Delta_bar)^2).
    central = float(np.mean(values))
    measurement_rms = float(np.sqrt(np.mean(errors**2)))
    spreading_rms = rms(values)
    error = float(np.hypot(measurement_rms, spreading_rms))
    return {"names": ["delta_extrap"], "values": np.asarray([central]),
            "errors": np.asarray([error]), "n": len(rows),
            "Ds": [row["D"] for row in rows],
            "measurement_error_rms": measurement_rms,
            "spreading_rms": spreading_rms}


def _new_figure(*, twin: bool = False, extra_width: float = 0.0):
    width, height = cfg.FIGSIZE[0] + extra_width, cfg.FIGSIZE[1]
    left, bottom, axes_width, axes_height = cfg.MAIN_AXES_INCHES
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([
        left / width, bottom / height,
        axes_width / width, axes_height / height,
    ])
    return fig, ax, ax.twinx() if twin else None


def _errorbar(ax, x, y, *, xerr=None, yerr=None, color="k", alpha=1.0,
              label=None, marker="o", linestyle="none", zorder=2):
    return ax.errorbar(
        x, y, xerr=xerr, yerr=yerr, fmt=marker, linestyle=linestyle,
        color=color, alpha=alpha, markersize=cfg.MARKER_SIZE,
        capsize=cfg.CAPSIZE, label=label, zorder=zorder,
    )


def _alpha(index: int, count: int) -> float:
    return cfg.RAW_ALPHA_MIN + (1.0 - cfg.RAW_ALPHA_MIN) * index / max(count - 1, 1)


def _row(figure: int, source: dict, *, series: str, x: float, y: float,
         x_error: float | None = None, y_error: float | None = None) -> dict:
    return {"figure": figure, "ansatz": source.get("ansatz", ""), "series": series,
            "J2": source.get("J2", ""), "D": source.get("D", ""),
            "x": x, "x_error": x_error, "y": y, "y_error": y_error}


def _fit_rows(figure: int, ansatz: str, j2: float, result: dict,
              model: str, excluded: list[tuple[float, int]]) -> list[dict]:
    return [
        {"figure": figure, "ansatz": ansatz, "J2": j2, "model": model,
         "n_points": result["n"], "parameter": name, "central": value,
         "error": error, "fit_bans": repr(excluded),
         "statistic_Ds": repr(result.get("Ds", "")),
         "measurement_error_rms": result.get("measurement_error_rms", ""),
         "spreading_rms": result.get("spreading_rms", "")}
        for name, value, error in zip(result["names"], result["values"], result["errors"])
    ]


def _colorbar(fig, ax, cmap, values):
    if values:
        norm = colors.Normalize(vmin=min(values), vmax=max(values) if max(values) > min(values) else min(values) + 1e-12)
        figure_width, figure_height = fig.get_size_inches()
        left, bottom, axes_width, axes_height = cfg.MAIN_AXES_INCHES
        cax = fig.add_axes([
            (left + axes_width + cfg.COLORBAR_GAP) / figure_width,
            bottom / figure_height,
            cfg.COLORBAR_WIDTH / figure_width,
            axes_height / figure_height,
        ])
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, label=r"$J_2$")
        return norm
    return colors.Normalize(0.0, 1.0)


def _fit_formula(ax, formula: str) -> None:
    ax.text(
        0.04, 0.96, formula, transform=ax.transAxes,
        ha="left", va="top",
    )


def _outside_legend(ax, *, ncols: int = 1, handles=None, labels=None) -> None:
    ax.legend(
        handles=handles, labels=labels,
        loc="center left", bbox_to_anchor=(1.02, 0.5),
        ncols=ncols,
    )


def _add_vertical_lines(ax) -> None:
    for value in cfg.VERTICAL_LINES_J2:
        ax.axvline(value, color="0.35", linestyle="--", linewidth=1.5)
        ax.text(
            value, 0.98, rf"$J_2={value:.2f}$",
            transform=ax.get_xaxis_transform(), rotation=90,
            ha="right", va="top", color="0.35",
        )


def _plot_m_xi(figure: int, rows: list[dict], *, fit_kind: str | None):
    fig, ax, _ = _new_figure(extra_width=cfg.COLORBAR_EXTRA_WIDTH)
    data_out, fits_out = [], []
    by_j2 = _groups(_finite(rows, "inverse_xi", "inverse_xi_error", "m", "m_error"), "J2")
    cmap = cm.viridis
    ansatz = rows[0]["ansatz"] if rows else ANSATZ_NEEL
    plotted_j2 = [j2 for j2 in by_j2 if _plot_allowed(j2, ansatz)]
    norm = colors.Normalize(min(plotted_j2, default=0.0), max(plotted_j2, default=1.0) or 1.0)
    for j2, group in by_j2.items():
        color = cmap(norm(j2))
        x = [row["inverse_xi"] for row in group]
        y = [row["m"] for row in group]
        if _plot_allowed(j2, ansatz):
            _errorbar(ax, x, y, xerr=[row["inverse_xi_error"] for row in group],
                      yerr=[row["m_error"] for row in group], color=color)
        data_out.extend(_row(figure, row, series="raw", x=row["inverse_xi"], y=row["m"],
                             x_error=row["inverse_xi_error"], y_error=row["m_error"]) for row in group)
        if fit_kind is None:
            continue
        fit_group = [row for row in group if not _fit_banned(figure, j2, row["D"])]
        absolute = fit_kind.startswith("abs")
        power = fit_kind.endswith("power")
        result = fit_m_xi(fit_group, absolute=absolute, power=power)
        if result is None:
            continue
        xmax = max(x)
        xline = np.linspace(0.0, xmax, cfg.FIT_CURVE_POINTS)
        if _plot_allowed(j2, ansatz):
            ax.plot(xline, result["model"](result["values"], xline), color=color)
        m0, m0_error = result["values"][0], result["errors"][0]
        if _plot_allowed(j2, ansatz):
            _errorbar(ax, [0.0], [m0], yerr=[m0_error], color=color, marker="s", zorder=4)
        excluded = [_key(j2, row["D"]) for row in group if _fit_banned(figure, j2, row["D"])]
        fits_out.extend(_fit_rows(figure, group[0]["ansatz"], j2, result, fit_kind, excluded))
        data_out.append(_row(figure, {"ansatz": group[0]["ansatz"], "J2": j2, "D": 0},
                             series="intercept", x=0.0, y=m0, y_error=m0_error))
    _colorbar(fig, ax, cmap, plotted_j2)
    if fit_kind is not None:
        ansatz_tag = r"\mathrm{N}" if rows and rows[0]["ansatz"] == ANSATZ_NEEL else r"2\mathrm{C}3"
        formulas = {
            "linear": rf"$m=m_{{0,\mathrm{{lin}}}}^{{{ansatz_tag}}}+c_{{\mathrm{{lin}}}}^{{{ansatz_tag}}}\xi^{{-1}}$",
            "abs_linear": rf"$m=\left|m_{{0,\mathrm{{abs\,lin}}}}^{{{ansatz_tag}}}\right|+c_{{\mathrm{{abs\,lin}}}}^{{{ansatz_tag}}}\xi^{{-1}}$",
            "power": rf"$m=m_{{0,\mathrm{{pow}}}}^{{{ansatz_tag}}}+c_{{\mathrm{{pow}}}}^{{{ansatz_tag}}}(\xi^{{-1}})^{{\alpha_{{\mathrm{{pow}}}}^{{{ansatz_tag}}}}}$",
            "abs_power": rf"$m=\left|m_{{0,\mathrm{{abs\,pow}}}}^{{{ansatz_tag}}}\right|+c_{{\mathrm{{abs\,pow}}}}^{{{ansatz_tag}}}(\xi^{{-1}})^{{\alpha_{{\mathrm{{abs\,pow}}}}^{{{ansatz_tag}}}}}$",
        }
        _fit_formula(ax, formulas[fit_kind])
    ax.set_xlabel(r"$1/\xi$")
    ax.set_ylabel(r"$m$")
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    return fig, data_out, fits_out


def _raw_vs_j2(ax, figure: int, rows: list[dict], *, observable: str,
               error_field: str | None, color: str, label_prefix: str = r"$D=",
               palette: dict[int, str] | None = None,
               connect: bool = False) -> list[dict]:
    output = []
    groups = _groups(_finite(rows, observable, *([error_field] if error_field else [])), "D")
    for index, (D, group) in enumerate(groups.items()):
        series_color = palette.get(int(D), color) if palette else color
        alpha = 1.0 if palette else _alpha(index, len(groups))
        plot_group = [row for row in group if _plot_allowed(row["J2"], row["ansatz"])]
        yerr = [row[error_field] for row in plot_group] if error_field else None
        if plot_group:
            _errorbar(ax, [row["J2"] for row in plot_group], [row[observable] for row in plot_group],
                      yerr=yerr, color=series_color, alpha=alpha,
                      label=rf"{label_prefix}{D}$",
                      linestyle="-" if connect else "none")
        output.extend(_row(figure, row, series=f"raw_D{D}", x=row["J2"], y=row[observable],
                           y_error=row[error_field] if error_field else None) for row in group)
    return output


def _m_extrap(figure: int, rows: list[dict]) -> tuple[list[dict], list[dict]]:
    points, fit_rows = [], []
    for j2, group in _groups(rows, "J2").items():
        fit_group = [row for row in group if not _fit_banned(figure, j2, row["D"])]
        result = fit_linear_invD(fit_group)
        if result is None:
            continue
        central = max(0.0, float(result["values"][0]))
        sigma = float(result["errors"][0])
        lower, upper = max(0.0, central - sigma), max(0.0, central + sigma)
        points.append({"ansatz": ANSATZ_NEEL, "J2": j2, "D": 0, "value": central,
                       "lower_error": central - lower, "upper_error": upper - central})
        excluded = [_key(j2, row["D"]) for row in group if _fit_banned(figure, j2, row["D"])]
        fit_rows.extend(_fit_rows(figure, ANSATZ_NEEL, j2, result, "linear_m_vs_invD", excluded))
    return points, fit_rows


def _delta_extrap(figure: int, rows: list[dict]) -> tuple[list[dict], list[dict]]:
    points, fit_rows = [], []
    for j2, group in _groups(rows, "J2").items():
        fit_group = [row for row in group if not _fit_banned(figure, j2, row["D"])]
        result = delta_statistic(fit_group)
        if result is None:
            continue
        central, error = float(result["values"][0]), float(result["errors"][0])
        points.append({"ansatz": ANSATZ_TWOC3, "J2": j2, "D": 0,
                       "value": central, "error": error})
        excluded = [_key(j2, row["D"]) for row in group if _fit_banned(figure, j2, row["D"])]
        fit_rows.extend(_fit_rows(figure, ANSATZ_TWOC3, j2, result,
                                  "inverse_variance_constant", excluded))
    return points, fit_rows


def _draw_m_extrap(ax, figure: int, rows: list[dict], *, label=r"$m_{\mathrm{extrap}}$"):
    points, fits = _m_extrap(figure, rows)
    plotted = [point for point in points if _plot_allowed(point["J2"], point["ansatz"])]
    if plotted:
        _errorbar(ax, [p["J2"] for p in plotted], [p["value"] for p in plotted],
                  yerr=[[p["lower_error"] for p in plotted], [p["upper_error"] for p in plotted]],
                  color=PURPLE, label=label, marker="s", linestyle="-", zorder=5)
    data = [_row(figure, p, series="m_extrap", x=p["J2"], y=p["value"],
                 y_error=max(p["lower_error"], p["upper_error"])) for p in points]
    return data, fits


def _draw_delta_extrap(ax, figure: int, rows: list[dict], *, label=r"$\Delta_{\mathrm{extrap}}$"):
    points, fits = _delta_extrap(figure, rows)
    plotted = [point for point in points if _plot_allowed(point["J2"], point["ansatz"])]
    if plotted:
        _errorbar(ax, [p["J2"] for p in plotted], [p["value"] for p in plotted],
                  yerr=[p["error"] for p in plotted], color=ORANGE, label=label,
                  marker="s", linestyle="-", zorder=5)
    data = [_row(figure, p, series="delta_extrap", x=p["J2"], y=p["value"],
                 y_error=p["error"]) for p in points]
    return data, fits


def _plot_m_delta(figure: int, neel: list[dict], twoc3: list[dict], *, mode: str):
    extra_width = cfg.TWO_COLUMN_LEGEND_EXTRA_WIDTH if figure in (15, 16) else 0.0
    fig, left, right = _new_figure(twin=True, extra_width=extra_width)
    data, fits = [], []
    if mode in ("raw", "raw_fit"):
        data += _raw_vs_j2(left, figure, neel, observable="m", error_field="m_error",
                           color=PURPLE, palette=PURPLE_D_COLORS)
        data += _raw_vs_j2(right, figure, twoc3, observable="delta", error_field="delta_error",
                           color=ORANGE, palette=ORANGE_D_COLORS)
    if mode in ("raw_fit", "fit"):
        new, fit = _draw_m_extrap(left, figure, neel)
        data += new; fits += fit
        new, fit = _draw_delta_extrap(right, figure, twoc3)
        data += new; fits += fit
    if mode == "fit":
        _add_vertical_lines(left)
    left.set_xlabel(r"$J_2$")
    left.set_ylabel(r"$m$", color=PURPLE)
    right.set_ylabel(r"$\Delta$", color=ORANGE)
    left.tick_params(axis="y", colors=PURPLE)
    right.tick_params(axis="y", colors=ORANGE)
    left.set_ylim(bottom=0.0); right.set_ylim(bottom=0.0)
    handles1, labels1 = left.get_legend_handles_labels()
    handles2, labels2 = right.get_legend_handles_labels()
    if handles1 or handles2:
        if figure in (15, 16):
            target = max(len(handles1), len(handles2))
            blank_count = target - len(handles1)
            padded_handles1 = handles1 + [
                Line2D([], [], linestyle="none") for _ in range(blank_count)
            ]
            padded_labels1 = labels1 + [""] * blank_count
            _outside_legend(
                left, ncols=2,
                handles=padded_handles1 + handles2,
                labels=padded_labels1 + labels2,
            )
        else:
            left.legend(handles1 + handles2, labels1 + labels2)
    return fig, data, fits


def _energy_fits(figure: int, rows: list[dict], *, gapped: bool):
    points, fits = [], []
    for j2, group in _groups(rows, "J2").items():
        fit_group = [row for row in group if not _fit_banned(figure, j2, row["D"])]
        result = fit_energy(fit_group, gapped=gapped)
        if result is None:
            continue
        points.append({"ansatz": group[0]["ansatz"], "J2": j2, "D": 0,
                       "value": float(result["values"][0]), "error": float(result["errors"][0]),
                       "result": result})
        excluded = [_key(j2, row["D"]) for row in group if _fit_banned(figure, j2, row["D"])]
        fits.extend(_fit_rows(figure, group[0]["ansatz"], j2, result,
                              "gapped" if gapped else "gapless", excluded))
    return points, fits


def _plot_energy_invD(figure: int, rows: list[dict], *, fit: str | None, cmap):
    fig, ax, _ = _new_figure(extra_width=cfg.COLORBAR_EXTRA_WIDTH)
    data, fits = [], []
    by_j2 = _groups(_finite(rows, "E"), "J2")
    ansatz = rows[0]["ansatz"] if rows else ANSATZ_NEEL
    plotted_j2 = [j2 for j2 in by_j2 if _plot_allowed(j2, ansatz)]
    norm = colors.Normalize(min(plotted_j2, default=0.0), max(plotted_j2, default=1.0) or 1.0)
    for j2, group in by_j2.items():
        color = cmap(norm(j2))
        x = [1.0 / row["D"] for row in group]
        if _plot_allowed(j2, ansatz):
            _errorbar(ax, x, [row["E"] for row in group], color=color)
        data.extend(_row(figure, row, series="raw", x=1.0 / row["D"], y=row["E"]) for row in group)
        if fit is None:
            continue
        fit_group = [row for row in group if not _fit_banned(figure, j2, row["D"])]
        result = fit_energy(fit_group, gapped=fit == "gapped")
        if result is None:
            continue
        xline = np.linspace(0.0, max(x), cfg.FIT_CURVE_POINTS)
        if _plot_allowed(j2, ansatz):
            ax.plot(xline, result["model"](result["values"], xline), color=color)
        E0, E0_error = result["values"][0], result["errors"][0]
        if _plot_allowed(j2, ansatz):
            _errorbar(ax, [0.0], [E0], yerr=[E0_error], color=color, marker="s", zorder=4)
        excluded = [_key(j2, row["D"]) for row in group if _fit_banned(figure, j2, row["D"])]
        fits.extend(_fit_rows(figure, group[0]["ansatz"], j2, result, fit, excluded))
        data.append(_row(figure, {"ansatz": group[0]["ansatz"], "J2": j2, "D": 0},
                         series=f"{fit}_E0", x=0.0, y=E0, y_error=E0_error))
    _colorbar(fig, ax, cmap, plotted_j2)
    if fit is not None:
        ansatz_tag = r"\mathrm{N}" if rows and rows[0]["ansatz"] == ANSATZ_NEEL else r"2\mathrm{C}3"
        if fit == "gapped":
            formula = rf"$E=E_{{0,\mathrm{{g}}}}^{{{ansatz_tag}}}+k_{{\mathrm{{g}}}}^{{{ansatz_tag}}}\exp\!\left[-a_{{\mathrm{{g}}}}^{{{ansatz_tag}}}/(1/D)\right]$"
        else:
            formula = rf"$E=E_{{0,\mathrm{{gl}}}}^{{{ansatz_tag}}}+k_{{\mathrm{{gl}}}}^{{{ansatz_tag}}}(1/D)^{{a_{{\mathrm{{gl}}}}^{{{ansatz_tag}}}}}$"
        _fit_formula(ax, formula)
    ax.set_xlabel(r"$1/D$"); ax.set_ylabel(r"$E$"); ax.set_xlim(left=0.0)
    return fig, data, fits


def _segmented_curve(ax, points: list[dict], *, color: str, gapped: bool, label: str):
    points = sorted(
        (point for point in points if _plot_allowed(point["J2"], point["ansatz"])),
        key=lambda point: point["J2"],
    )
    if not points:
        return
    x = np.asarray([p["J2"] for p in points]); y = np.asarray([p["value"] for p in points])
    _errorbar(ax, x, y, yerr=[p["error"] for p in points], color=color,
              marker="s" if gapped else "o", label=label, zorder=4)
    switch = cfg.ENERGY_SWITCH_J2
    for first, second in zip(points[:-1], points[1:]):
        after = first["J2"] >= switch
        solid = after if gapped else not after
        ax.plot([first["J2"], second["J2"]], [first["value"], second["value"]],
                color=color, linestyle="-" if solid else "--")


def _plot_energy_j2(figure: int, rows: list[dict], *, color: str,
                    include_fits: bool, raw: bool = True):
    fig, ax, _ = _new_figure(extra_width=cfg.LEGEND_EXTRA_WIDTH)
    data, fits = [], []
    if raw:
        data += _raw_vs_j2(ax, figure, rows, observable="E", error_field=None, color=color)
    if include_fits:
        gapless, fit_gapless = _energy_fits(figure, rows, gapped=False)
        gapped, fit_gapped = _energy_fits(figure, rows, gapped=True)
        _segmented_curve(ax, gapless, color=color, gapped=False, label=r"gapless $E_{\mathrm{extrap}}$")
        _segmented_curve(ax, gapped, color=color, gapped=True, label=r"gapped $E_{\mathrm{extrap}}$")
        fits += fit_gapless + fit_gapped
        for name, points in (("gapless_E_extrap", gapless), ("gapped_E_extrap", gapped)):
            data.extend(_row(figure, p, series=name, x=p["J2"], y=p["value"], y_error=p["error"]) for p in points)
    ax.set_xlabel(r"$J_2$"); ax.set_ylabel(r"$E$")
    if ax.get_legend_handles_labels()[0]:
        _outside_legend(ax)
    return fig, data, fits


def _plot_energy_combined(figure: int, neel: list[dict], twoc3: list[dict]):
    fig, ax, _ = _new_figure()
    data, fits = [], []
    for rows, color, ansatz in ((neel, BLUE, ANSATZ_NEEL), (twoc3, RED, ANSATZ_TWOC3)):
        gapless, fit_gapless = _energy_fits(figure, rows, gapped=False)
        gapped, fit_gapped = _energy_fits(figure, rows, gapped=True)
        _segmented_curve(ax, gapless, color=color, gapped=False,
                         label=rf"{ansatz} gapless")
        _segmented_curve(ax, gapped, color=color, gapped=True,
                         label=rf"{ansatz} gapped")
        fits += fit_gapless + fit_gapped
        for name, points in (("gapless_E_extrap", gapless), ("gapped_E_extrap", gapped)):
            data.extend(_row(figure, p, series=name, x=p["J2"], y=p["value"], y_error=p["error"]) for p in points)
    _add_vertical_lines(ax)
    ax.set_xlabel(r"$J_2$"); ax.set_ylabel(r"$E$"); ax.legend()
    return fig, data, fits


def render_figure(figure: int, dataset: dict[str, list[dict]]):
    neel, twoc3 = dataset[ANSATZ_NEEL], dataset[ANSATZ_TWOC3]
    if figure in range(1, 6):
        kinds = {1: None, 2: "linear", 3: "abs_linear", 4: "power", 5: "abs_power"}
        return _plot_m_xi(figure, neel, fit_kind=kinds[figure])
    if figure in range(6, 11):
        limited = [row for row in twoc3 if row["J2"] <= cfg.TWOC3_M_VS_XI_J2_MAX]
        kinds = {6: None, 7: "linear", 8: "abs_linear", 9: "power", 10: "abs_power"}
        return _plot_m_xi(figure, limited, fit_kind=kinds[figure])
    if figure == 11:
        fig, ax, _ = _new_figure(extra_width=cfg.LEGEND_EXTRA_WIDTH); data = _raw_vs_j2(ax, figure, neel, observable="m", error_field="m_error", color=PURPLE, palette=PURPLE_D_COLORS)
        ax.set_xlabel(r"$J_2$"); ax.set_ylabel(r"$m$"); ax.set_ylim(bottom=0.0); _outside_legend(ax); return fig, data, []
    if figure == 12:
        fig, ax, _ = _new_figure(extra_width=cfg.LEGEND_EXTRA_WIDTH); data = _raw_vs_j2(ax, figure, twoc3, observable="delta", error_field="delta_error", color=ORANGE, palette=ORANGE_D_COLORS)
        ax.set_xlabel(r"$J_2$"); ax.set_ylabel(r"$\Delta$"); ax.set_ylim(bottom=0.0); _outside_legend(ax); return fig, data, []
    if figure == 13:
        fig, ax, _ = _new_figure(extra_width=cfg.LEGEND_EXTRA_WIDTH); data = _raw_vs_j2(ax, figure, neel, observable="m", error_field="m_error", color=PURPLE, palette=PURPLE_D_COLORS)
        extra, fits = _draw_m_extrap(ax, figure, neel); data += extra
        ax.set_xlabel(r"$J_2$"); ax.set_ylabel(r"$m$"); ax.set_ylim(bottom=0.0); _outside_legend(ax); return fig, data, fits
    if figure == 14:
        fig, ax, _ = _new_figure(extra_width=cfg.LEGEND_EXTRA_WIDTH); data = _raw_vs_j2(ax, figure, twoc3, observable="delta", error_field="delta_error", color=ORANGE, palette=ORANGE_D_COLORS)
        extra, fits = _draw_delta_extrap(ax, figure, twoc3); data += extra
        ax.set_xlabel(r"$J_2$"); ax.set_ylabel(r"$\Delta$"); ax.set_ylim(bottom=0.0); _outside_legend(ax); return fig, data, fits
    if figure in (15, 16, 17):
        return _plot_m_delta(figure, neel, twoc3, mode={15: "raw", 16: "raw_fit", 17: "fit"}[figure])
    if figure in (18, 19, 20):
        return _plot_energy_invD(figure, neel, fit={18: None, 19: "gapped", 20: "gapless"}[figure], cmap=NEEL_ENERGY_CMAP)
    if figure == 21:
        return _plot_energy_j2(figure, neel, color=BLUE, include_fits=False)
    if figure == 22:
        return _plot_energy_j2(figure, neel, color=BLUE, include_fits=True)
    if figure in (23, 24, 25):
        return _plot_energy_invD(figure, twoc3, fit={23: None, 24: "gapped", 25: "gapless"}[figure], cmap=TWOC3_ENERGY_CMAP)
    if figure == 26:
        return _plot_energy_j2(figure, twoc3, color=RED, include_fits=False)
    if figure == 27:
        return _plot_energy_j2(figure, twoc3, color=RED, include_fits=True)
    if figure == 28:
        return _plot_energy_combined(figure, neel, twoc3)
    raise ValueError(f"Unknown figure {figure}")


def render_figure_12_bis(dataset: dict[str, list[dict]]):
    fig, ax, _ = _new_figure(extra_width=cfg.LEGEND_EXTRA_WIDTH)
    data = _raw_vs_j2(
        ax, 12, dataset[ANSATZ_TWOC3], observable="delta",
        error_field="delta_error", color=ORANGE,
        palette=ORANGE_D_COLORS, connect=True,
    )
    ax.set_xlabel(r"$J_2$")
    ax.set_ylabel(r"$\Delta$")
    ax.set_ylim(bottom=0.0)
    _outside_legend(ax)
    return fig, data


def render_figure_11_bis(dataset: dict[str, list[dict]]):
    fig, ax, _ = _new_figure(extra_width=cfg.LEGEND_EXTRA_WIDTH)
    data = _raw_vs_j2(
        ax, 11, dataset[ANSATZ_NEEL], observable="m",
        error_field="m_error", color=PURPLE,
        palette=PURPLE_D_COLORS, connect=True,
    )
    ax.set_xlabel(r"$J_2$")
    ax.set_ylabel(r"$m$")
    ax.set_ylim(bottom=0.0)
    _outside_legend(ax)
    return fig, data


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        if path.exists():
            path.unlink()
        return
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def run_figure(figure: int, dataset: dict[str, list[dict]] | None = None) -> Path:
    if figure == 14 and len(cfg.FIT_BANS[14]) > 1:
        raise ValueError("Figure 14 FIT_BANS may contain at most one (J2, D) point")
    plt.style.use(cfg.STYLE_PATH)
    cfg.FIGURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg.PROCESSED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset() if dataset is None else dataset
    fig, data, fits = render_figure(figure, dataset)
    figure_path = cfg.FIGURE_OUTPUT_DIR / f"figure_{figure:02d}.pdf"
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)
    _write_csv(cfg.PROCESSED_OUTPUT_DIR / f"figure_{figure:02d}_data.csv", data)
    _write_csv(cfg.PROCESSED_OUTPUT_DIR / f"figure_{figure:02d}_fits.csv", fits)
    if figure == 11:
        bis, bis_data = render_figure_11_bis(dataset)
        bis_path = cfg.FIGURE_OUTPUT_DIR / "figure_11_bis.pdf"
        bis.savefig(bis_path, bbox_inches="tight")
        plt.close(bis)
        _write_csv(cfg.PROCESSED_OUTPUT_DIR / "figure_11_bis_data.csv", bis_data)
    if figure == 12:
        bis, bis_data = render_figure_12_bis(dataset)
        bis_path = cfg.FIGURE_OUTPUT_DIR / "figure_12_bis.pdf"
        bis.savefig(bis_path, bbox_inches="tight")
        plt.close(bis)
        _write_csv(cfg.PROCESSED_OUTPUT_DIR / "figure_12_bis_data.csv", bis_data)
    print(figure_path)
    return figure_path


def run_figures(figures=range(1, 29)) -> None:
    dataset = load_dataset()
    if not dataset[ANSATZ_NEEL]:
        raise RuntimeError(f"No Neel data found below {cfg.NEEL_DATA_ROOT}")
    if not dataset[ANSATZ_TWOC3]:
        raise RuntimeError(f"No 2C3 data found below {cfg.TWOC3_DATA_ROOT}")
    for figure in figures:
        run_figure(int(figure), dataset)


def single_main(figure: int) -> int:
    run_figure(figure)
    return 0
