#!/usr/bin/env python3
"""Production-style energy comparisons requested for figs0906.

Every plotted number is hardcoded in this file so that the figures can be
edited and regenerated without access to the original data directories.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


HERE = Path(__file__).resolve().parent
STYLE = HERE.parent / "PublicationPlots" / "plottingStyle" / "everyday_stylesheet.mplstyle"
# Latest 2C3 energy-per-site data; requested comparison values are all literal.
TWO_C3 = {
    0.24: {5: -0.4390220048111, 6: -0.4397241791201, 7: -0.4400332326324,
           8: -0.4402652370156, 9: -0.4403730014637, 10: -0.4404665074930},
    0.245: {8: -0.4387385592657, 10: -0.4389741054558},
    0.25: {8: -0.4372386733638, 10: -0.4374758198990},
    0.26: {5: -0.4327170350214, 6: -0.4336099793702, 7: -0.4341051449691,
           8: -0.4343870123737, 9: -0.4346062193824, 10: -0.4346810460999},
    0.265: {8: -0.4330308105537, 10: -0.4333419227966},
    0.27: {8: -0.4316550562418, 10: -0.4320399938912},
    0.275: {8: -0.4303882490411, 10: -0.4307514424297},
    0.28: {5: -0.4277485793095, 6: -0.4284879576922, 7: -0.4289195349411,
           8: -0.4291893313359, 9: -0.4294108101266, 10: -0.4295193839466},
    0.29: {8: -0.4268483479623, 10: -0.4271813261246},
    0.30: {2: -0.3950449798745, 3: -0.4168204665912, 4: -0.4195672180976,
           5: -0.4228575645321, 6: -0.4239107969320, 7: -0.4242729290438,
           8: -0.4246130246619, 9: -0.4248712511494, 10: -0.4250070508240,
           11: -0.4250380256901},
    0.31: {8: -0.4225934640509, 10: -0.4229589784093},
    0.32: {5: -0.4183991752452, 6: -0.4198309439578, 7: -0.4203108986799,
           8: -0.4207280842888, 9: -0.4209327748874, 10: -0.4210683557824},
}

LESS_RESTRICTED_D10 = {
    0.24: -0.440164615, 0.26: -0.434247830, 0.28: -0.428867853,
    0.30: -0.424076147, 0.32: -0.419633676,
}

MORE_RESTRICTED_D8 = {
    0.24: -0.439142524, 0.26: -0.432680783, 0.28: -0.426623461,
    0.30: -0.421025828, 0.32: -0.415818627,
}

# J2=0.30 external variational data (called VUMPS in the older script).
LESS_RESTRICTED_J2_030 = {
    4: -0.41956662, 5: -0.42177429, 6: -0.42275089,
    7: -0.42352259, 8: -0.42421711, 9: -0.42464710,
}

MORE_RESTRICTED_J2_030 = {
    4: -0.41969802, 5: -0.42040762, 6: -0.42113174,
    7: -0.42125176, 8: -0.42127372, 9: -0.42124687,
}

COLUMNAR_J2_030 = {
    2: -0.3950449758864, 3: -0.4168206838974, 4: -0.4195672772825,
    5: -0.4216474423767, 6: -0.4227095635775, 7: -0.4236111328932,
    9: -0.4247688279767,
}

BLUE = "#2166ac"
ORANGE = "#d95f02"
RED = "#b2182b"
PURPLE = "#6f2dbd"
GREEN = "#1b9e77"


def new_figure(extra_width: float = 3.5):
    width, height = 6.65 + extra_width, 5.2
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([1.15 / width, 0.85 / height, 5.0 / width, 4.0 / height])
    return fig, ax


def finish(fig, ax, basename: str, *, ncols: int = 1) -> None:
    ax.legend(loc="center left", bbox_to_anchor=(1.03, 0.5), ncols=ncols, fontsize=16)
    output = HERE / f"{basename}.pdf"
    fig.savefig(output, bbox_inches="tight")
    print(output)
    plt.close(fig)


def format_inverse_d_axis(ax) -> None:
    ax.set_xlim(left=0.0)
    ax.set_xlabel(r"$1/D$")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.10))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))


def plot_2c3_minus_less_restricted() -> None:
    fig, ax = new_figure()
    specs = [
        (0.24, BLUE, "o"), (0.26, ORANGE, "s"), (0.28, GREEN, "^"),
        (0.30, RED, "D"), (0.32, PURPLE, "v"),
    ]
    for j2, color, marker in specs:
        ds = list(range(5, 11))
        difference = [TWO_C3[j2][d] - LESS_RESTRICTED_D10[j2] for d in ds]
        ax.plot([1.0 / d for d in ds], difference, marker=marker, color=color,
                markersize=6, label=rf"$J_2={j2:.2f}$")
    ax.axhline(0.0, color="black", linewidth=1.1)
    format_inverse_d_axis(ax)
    ax.set_ylabel(r"$E_{\mathrm{2C3}}(D)-E_{\mathrm{less}}(10)$")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, -3), useMathText=True)
    finish(fig, ax, "energy_difference_2C3_D5_D10_vs_less_restricted_D10")


def plot_restriction_comparison() -> None:
    fig, ax = new_figure()
    j2s = sorted(TWO_C3)
    restricted_j2s = sorted(LESS_RESTRICTED_D10)
    curves = [
        (j2s, [TWO_C3[j2][8] for j2 in j2s], r"2C3, $D=8$", BLUE, "o", "-"),
        (j2s, [TWO_C3[j2][10] for j2 in j2s], r"2C3, $D=10$", RED, "D", "-"),
        (restricted_j2s, [LESS_RESTRICTED_D10[j2] for j2 in restricted_j2s],
         r"less restricted, $D=10$", GREEN, "^", "--"),
        (restricted_j2s, [MORE_RESTRICTED_D8[j2] for j2 in restricted_j2s],
         r"more restricted, $D=8$", ORANGE, "s", "--"),
    ]
    for x_values, values, label, color, marker, linestyle in curves:
        ax.plot(x_values, values, label=label, color=color, marker=marker,
                markersize=6, linestyle=linestyle)
    ax.set_xlim(0.24, 0.32)
    ax.set_xticks(restricted_j2s)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    ax.set_xlabel(r"$J_2$")
    ax.set_ylabel(r"$E$ per site")
    finish(fig, ax, "energy_2C3_D8_D10_and_restricted_ansatze")


def plot_four_energies() -> None:
    fig, ax = new_figure()
    curves = [
        (LESS_RESTRICTED_J2_030, r"less restricted ($E_{\mathrm{unif}}$)", BLUE, "o"),
        (MORE_RESTRICTED_J2_030, r"more restricted ($E_{\mathrm{PVB}}$)", ORANGE, "s"),
        (COLUMNAR_J2_030, r"2tensor columnar", PURPLE, "^"),
        (TWO_C3[0.30], r"2C3", RED, "D"),
    ]
    for data, label, color, marker in curves:
        ds = sorted(d for d in data if d >= 5)
        ax.plot([1.0 / d for d in ds], [data[d] for d in ds],
                label=label, color=color, marker=marker, markersize=6)
    format_inverse_d_axis(ax)
    ax.set_ylabel(r"$E$ per site")
    finish(fig, ax, "energy_J2_0p30_four_ansatze")


def plot_three_differences() -> None:
    fig, ax = new_figure()
    curve_specs = [
        (COLUMNAR_J2_030, LESS_RESTRICTED_J2_030,
         r"$E_{\mathrm{columnar}}-E_{\mathrm{less}}$", BLUE, "o"),
        (TWO_C3[0.30], MORE_RESTRICTED_J2_030,
         r"$E_{\mathrm{2C3}}-E_{\mathrm{more}}$", ORANGE, "s"),
    ]
    for lhs, rhs, label, color, marker in curve_specs:
        ds = sorted(d for d in set(lhs) & set(rhs) if d >= 5)
        values = [lhs[d] - rhs[d] for d in ds]
        ax.plot([1.0 / d for d in ds], values, label=label, color=color,
                marker=marker, markersize=6)
    ax.axhline(0.0, color="black", linewidth=1.1)
    format_inverse_d_axis(ax)
    ax.set_ylabel(r"$E_{\mathrm{iPEPS}}-E_{\mathrm{VUMPS}}$")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, -3), useMathText=True)
    finish(fig, ax, "energy_difference_J2_0p30_three_ansatze")


def main() -> None:
    plt.style.use(STYLE)
    plot_2c3_minus_less_restricted()
    plot_restriction_comparison()
    plot_four_energies()
    plot_three_differences()


if __name__ == "__main__":
    main()
