"""Build a signed-h-symmetric Delta surface directly from extrapolated data.

The plotting coordinate is x=sqrt(abs(h)). Opposite field signs at the same
(abs(h), J2) are averaged with equal weight before fitting, so the inferred
surface is explicitly even in h. Raw positive- and negative-h observations
remain separate in the plot and retain their signs through a blue/red colour
gradient. No h=0 phase boundary or critical coupling is imposed.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.lines import Line2D
from scipy.interpolate import PchipInterpolator


HERE = Path(__file__).resolve().parent
INPUT = (
    HERE / "plots" / "pinning_energy_phase_boundary"
    / "texture_D_infinity_nodes.csv"
)
POSTER = HERE / "Delta_symmetric_sqrt_h_surface.png"
VIDEO = HERE / "Delta_symmetric_sqrt_h_surface_rotation.mp4"
H_LIMIT = 0.10
X_LIMIT = math.sqrt(H_LIMIT)
J_LIMIT = 0.35
MIN_J2_PER_FIELD = 4


def load_rows() -> list[dict[str, float | str]]:
    with INPUT.open(encoding="utf-8-sig", newline="") as stream:
        return [{
            "J2": float(row["J2"]),
            "h": float(row["signed_h"]),
            "Delta": float(row["extrapolated_Delta"]),
            "branch": row["branch"],
        } for row in csv.DictReader(stream)]


def symmetrise(
    rows: list[dict[str, float | str]],
) -> list[dict[str, float]]:
    """Give each (|h|,J2) one value, independent of rerun/sign counts."""
    buckets: dict[tuple[float, float], list[float]] = {}
    for row in rows:
        key = (round(abs(float(row["h"])), 12),
               round(float(row["J2"]), 12))
        buckets.setdefault(key, []).append(float(row["Delta"]))
    return [{"abs_h": field, "x": math.sqrt(field), "J2": coupling,
             "Delta": float(np.mean(values))}
            for (field, coupling), values in sorted(buckets.items())]


def groups_by_field(
    rows: list[dict[str, float]],
) -> dict[float, list[dict[str, float]]]:
    groups: dict[float, list[dict[str, float]]] = {}
    for row in rows:
        groups.setdefault(row["abs_h"], []).append(row)
    return groups


def power_cv_rmse(groups: dict, power: float) -> float:
    """Leave one whole J2 value out of each fixed-|h| regression."""
    residuals: list[float] = []
    for subset in groups.values():
        if len(subset) < MIN_J2_PER_FIELD:
            continue
        coupling = np.asarray([row["J2"] for row in subset])
        delta = np.asarray([row["Delta"] for row in subset])
        transformed = delta ** power
        for held_out in range(len(subset)):
            keep = np.arange(len(subset)) != held_out
            slope, intercept = np.polyfit(
                coupling[keep], transformed[keep], 1,
            )
            prediction = max(
                0.0, slope * coupling[held_out] + intercept,
            ) ** (1.0 / power)
            residuals.append(prediction - delta[held_out])
    return float(np.sqrt(np.mean(np.asarray(residuals) ** 2)))


def choose_power(groups: dict) -> tuple[float, list[tuple[float, float]]]:
    candidates = np.arange(0.5, 3.01, 0.25)
    scores = [(float(power), power_cv_rmse(groups, float(power)))
              for power in candidates]
    return min(scores, key=lambda item: item[1])[0], scores


def fixed_field_fits(groups: dict, power: float) -> list[dict[str, float]]:
    """Fit Delta**p=a(|h|)[J2-Jc(|h|)] at each measured |h|."""
    fits: list[dict[str, float]] = []
    for field, subset in sorted(groups.items()):
        if len(subset) < MIN_J2_PER_FIELD:
            continue
        coupling = np.asarray([row["J2"] for row in subset])
        delta = np.asarray([row["Delta"] for row in subset])
        transformed = delta ** power
        slope, intercept = np.polyfit(coupling, transformed, 1)
        if not math.isfinite(slope) or slope <= 0.0:
            continue
        prediction = slope * coupling + intercept
        total = float(np.sum((transformed - np.mean(transformed)) ** 2))
        residual = float(np.sum((transformed - prediction) ** 2))
        fits.append({
            "abs_h": float(field),
            "x": math.sqrt(float(field)),
            "slope": float(slope),
            "root": float(-intercept / slope),
            "r2": float(1.0 - residual / total) if total > 0.0 else math.nan,
            "n": float(len(subset)),
        })
    return fits


class SurfaceModel:
    def __init__(self, fits: list[dict[str, float]]):
        ordered = sorted(fits, key=lambda row: row["x"])
        self.x = np.asarray([row["x"] for row in ordered])
        self.root = np.asarray([row["root"] for row in ordered])
        self.log_slope = np.log(np.asarray([row["slope"] for row in ordered]))
        self.root_interpolator = PchipInterpolator(
            self.x, self.root, extrapolate=False,
        )
        self.slope_interpolator = PchipInterpolator(
            self.x, self.log_slope, extrapolate=False,
        )

    @staticmethod
    def _linear_outer(values: np.ndarray, x: np.ndarray,
                      y: np.ndarray) -> np.ndarray:
        result = np.empty_like(values, dtype=float)
        inside = (values >= x[0]) & (values <= x[-1])
        result[inside] = np.nan
        left = values < x[0]
        right = values > x[-1]
        left_slope = (y[1] - y[0]) / (x[1] - x[0])
        right_slope = (y[-1] - y[-2]) / (x[-1] - x[-2])
        result[left] = y[0] + left_slope * (values[left] - x[0])
        result[right] = y[-1] + right_slope * (values[right] - x[-1])
        return result

    def roots(self, x_values: np.ndarray) -> np.ndarray:
        x_values = np.asarray(x_values, dtype=float)
        result = self._linear_outer(x_values, self.x, self.root)
        inside = (x_values >= self.x[0]) & (x_values <= self.x[-1])
        result[inside] = self.root_interpolator(x_values[inside])
        return result

    def slopes(self, x_values: np.ndarray) -> np.ndarray:
        x_values = np.asarray(x_values, dtype=float)
        result = self._linear_outer(x_values, self.x, self.log_slope)
        inside = (x_values >= self.x[0]) & (x_values <= self.x[-1])
        result[inside] = self.slope_interpolator(x_values[inside])
        return np.exp(result)

    def delta(self, X: np.ndarray, J: np.ndarray,
              power: float) -> np.ndarray:
        transformed = self.slopes(X.ravel()) * (
            J.ravel() - self.roots(X.ravel())
        )
        return (np.maximum(0.0, transformed) ** (1.0 / power)).reshape(X.shape)


def signed_colour(field: float, maximum: float) -> tuple[float, float, float, float]:
    if abs(field) <= 1.0e-14:
        return (0.02, 0.02, 0.02, 1.0)
    strength = min(1.0, math.sqrt(abs(field) / maximum))
    target = (0.86, 0.08, 0.08) if field < 0.0 else (0.05, 0.28, 0.88)
    start = np.asarray((0.06, 0.06, 0.06))
    rgb = (1.0 - strength) * start + strength * np.asarray(target)
    return (float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0)


def common_style() -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.size": 14,
        "axes.labelsize": 18, "xtick.labelsize": 12,
        "ytick.labelsize": 12, "legend.fontsize": 10.5,
    })


def construct_figure(
    rows: list[dict[str, float | str]], model: SurfaceModel, power: float,
) -> tuple[plt.Figure, plt.Axes]:
    x_grid = np.linspace(0.0, X_LIMIT, 170)
    j_grid = np.linspace(0.0, J_LIMIT, 170)
    X, J = np.meshgrid(x_grid, j_grid)
    D = model.delta(X, J, power)
    boundary_x = np.linspace(0.0, X_LIMIT, 600)
    boundary_j = model.roots(boundary_x)
    boundary_valid = (boundary_j >= 0.0) & (boundary_j <= J_LIMIT)
    maximum_field = max(abs(float(row["h"])) for row in rows)

    figure = plt.figure(figsize=(10.6, 7.6), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    axis.computed_zorder = False
    axis.plot_surface(
        X, J, D, cmap="viridis", alpha=0.38, linewidth=0.0,
        antialiased=True, rcount=105, ccount=105, zorder=1,
    )
    axis.plot_surface(
        X, J, np.zeros_like(D), color="0.72", alpha=0.13,
        linewidth=0.0, antialiased=False, zorder=0,
    )

    # Stems make the three-dimensional location of every observation explicit.
    for row in rows:
        field = float(row["h"])
        x_value = math.sqrt(abs(field))
        coupling = float(row["J2"])
        delta = float(row["Delta"])
        colour = signed_colour(field, maximum_field)
        axis.plot(
            [x_value, x_value], [coupling, coupling], [0.0, delta],
            color=colour, alpha=0.38, linewidth=0.75, zorder=2,
        )
        axis.scatter(
            [x_value], [coupling], [delta], color=[colour], s=18,
            marker="o", edgecolors="white", linewidths=0.20,
            depthshade=False, alpha=0.94, zorder=4,
        )

    axis.plot(
        boundary_x[boundary_valid], boundary_j[boundary_valid],
        np.zeros(np.count_nonzero(boundary_valid)),
        color="#ffcc00", linewidth=4.0, zorder=6,
    )
    measured_boundary = (model.root >= 0.0) & (model.root <= J_LIMIT)
    axis.scatter(
        model.x[measured_boundary], model.root[measured_boundary],
        np.zeros(np.count_nonzero(measured_boundary)),
        color="#ffcc00", marker="x",
        s=34, linewidths=1.4, depthshade=False, zorder=7,
    )

    axis.set_xlim(0.0, X_LIMIT)
    axis.set_ylim(0.0, J_LIMIT)
    axis.set_zlim(0.0, max(0.62, float(np.max(D)) * 1.04))
    axis.set_xlabel(r"$\sqrt{|h|}$", labelpad=11)
    axis.set_ylabel(r"$J_2$", labelpad=11)
    axis.set_zlabel(r"extrapolated $\Delta$", labelpad=9)
    axis.tick_params(which="major", labelsize=12)
    axis.view_init(elev=25, azim=-58)
    axis.legend(handles=[
        Line2D([], [], color=(0.86, 0.08, 0.08), marker="o",
               linestyle="none", markersize=6,
               label=r"extrapolated $h<0$ data"),
        Line2D([], [], color="black", marker="o", linestyle="none",
               markersize=6, label=r"extrapolated $h=0$ data"),
        Line2D([], [], color=(0.05, 0.28, 0.88), marker="o",
               linestyle="none", markersize=6,
               label=r"extrapolated $h>0$ data"),
        Line2D([], [], color="#ffcc00", linewidth=4,
               label=r"data-inferred $\Delta=0$ boundary"),
    ], loc="upper left", fontsize=10)
    return figure, axis


def plot_poster(
    rows: list[dict[str, float | str]], model: SurfaceModel, power: float,
) -> None:
    figure, _ = construct_figure(rows, model, power)
    figure.savefig(POSTER, dpi=300)
    plt.close(figure)


def plot_video(
    rows: list[dict[str, float | str]], model: SurfaceModel, power: float,
) -> None:
    if not FFMpegWriter.isAvailable():
        raise RuntimeError("ffmpeg is unavailable; Delta surface video not written")
    figure, axis = construct_figure(rows, model, power)
    temporary = VIDEO.with_name(f".{VIDEO.stem}.tmp{VIDEO.suffix}")
    azimuths = np.linspace(-58.0, -238.0, 61)

    def rotate(frame: int) -> tuple:
        axis.view_init(elev=25, azim=float(azimuths[frame]))
        return (axis,)

    animation = FuncAnimation(
        figure, rotate, frames=len(azimuths), interval=100, blit=False,
    )
    animation.save(
        temporary,
        writer=FFMpegWriter(
            fps=10, bitrate=3400,
            metadata={"title": "signed-h-symmetric data-inferred Delta surface"},
        ),
        dpi=145,
    )
    temporary.replace(VIDEO)
    plt.close(figure)


def main() -> int:
    common_style()
    raw_rows = load_rows()
    symmetric_rows = symmetrise(raw_rows)
    groups = groups_by_field(symmetric_rows)
    power, scores = choose_power(groups)
    fits = fixed_field_fits(groups, power)
    model = SurfaceModel(fits)
    plot_poster(raw_rows, model, power)
    plot_video(raw_rows, model, power)

    best_rmse = next(score for candidate, score in scores
                     if math.isclose(candidate, power))
    print(f"cross-validated power p={power:g}; Delta RMSE={best_rmse:.8g}")
    print("fixed-|h| roots inferred from symmetrised data:")
    for row in fits:
        print(
            f"  |h|={row['abs_h']:.3g}: Jc={row['root']:.6f}, "
            f"R2={row['r2']:.4f}, nJ2={int(row['n'])}"
        )
    print(f"Poster: {POSTER}")
    print(f"Video: {VIDEO}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
