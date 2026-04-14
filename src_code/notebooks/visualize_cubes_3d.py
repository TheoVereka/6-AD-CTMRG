"""
Interactive 3D cube visualization of D=2 (chi=4) and D=3 (chi=9) iPEPS tensors.

  D=2 → 2×2×2  cube, 8  corner nodes
  D=3 → 3×3×3  grid, 27 corner nodes

Layout per figure: 6 rows (sites a–f) × 2 cols (spin ↑ / spin ↓).
Node colour → tensor value  T[i₀,i₁,i₂,s]  (RdBu_r, shared scale per figure).
Node size   → |value| (bigger = larger magnitude).
Text labels (value) shown for D=2; omitted for D=3 (too crowded).

Colorbar is placed to the RIGHT of the subplots, never overlapping.
Rotate any subplot by click-and-drag.
"""

import warnings
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D             # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)

# ── data ─────────────────────────────────────────────────────────────────────
DATA_DIR = Path("/home/chye/6ADctmrg/data/raw/6tensors_20260414_145446")

CHECKPOINTS = {
    2: DATA_DIR / "sweep_D2_chi4_best.pt",
    3: DATA_DIR / "sweep_D3_chi9_best.pt",
    4: DATA_DIR / "sweep_D4_chi16_best.pt",
}

SITES = list("abcdef")
SPIN_LABELS = ["spin ↑  (s=0)", "spin ↓  (s=1)"]
CMAP = matplotlib.cm.RdBu_r


# ── geometry helpers ──────────────────────────────────────────────────────────

def make_corners(D: int) -> np.ndarray:
    """All D³ corners (i₀,i₁,i₂) ∈ {0,…,D-1}³, shape (D³, 3)."""
    coords = np.arange(D, dtype=float)
    g = np.stack(np.meshgrid(coords, coords, coords, indexing="ij"), axis=-1)
    return g.reshape(-1, 3)


def make_edges(D: int, corners: np.ndarray) -> list[list]:
    """Grid edges: pairs that differ in exactly one axis by exactly 1."""
    idx = {tuple(c): i for i, c in enumerate(corners.astype(int).tolist())}
    segs = []
    for ci in corners.astype(int):
        for ax in range(3):
            nj = ci.copy(); nj[ax] += 1
            if tuple(nj) in idx:
                segs.append([ci.tolist(), nj.tolist()])
    return segs


# ── figure builder ────────────────────────────────────────────────────────────

def build_figure(D: int) -> plt.Figure:
    pt_path = CHECKPOINTS[D]
    ckpt    = torch.load(pt_path, map_location="cpu", weights_only=False)
    tensors = {k: ckpt[k].numpy() for k in SITES}
    energy  = ckpt["energy"]
    chi     = ckpt["chi"]

    corners = make_corners(D)   # (D³, 3)
    segs    = make_edges(D, corners)
    annotate = (D == 2)         # annotate values only for D=2

    # global colour scale over all 6 sites × 2 spins
    all_vals = np.concatenate([tensors[s].ravel() for s in SITES])
    vmax = float(np.max(np.abs(all_vals)))
    vmin = -vmax
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    n_rows, n_cols = len(SITES), 2
    fig_w = 5.2 * n_cols + 1.8   # extra right margin for colorbar
    fig_h = 4.2 * n_rows + 0.8

    fig = plt.figure(figsize=(fig_w, fig_h))
    energy_str = f"{energy:.8f}" if energy is not None else "None"
    fig.suptitle(
        f"D={D}  chi={chi}   energy = {energy_str}\n"
        f"Node colour & size → T[a,b,c,s]     (drag to rotate)",
        fontsize=11, fontweight="bold", y=0.995,
    )

    # leave ~14% of figure width on the right for the colorbar
    fig.subplots_adjust(left=0.05, right=0.84, top=0.955, bottom=0.02,
                        hspace=0.30, wspace=0.05)

    axes_list = []
    for row, site in enumerate(SITES):
        T = tensors[site]   # (D, D, D, 2)

        for col, s in enumerate([0, 1]):
            ax: Axes3D = fig.add_subplot(n_rows, n_cols,
                                         row * n_cols + col + 1,
                                         projection="3d")

            # values at every corner
            vals = np.array([T[int(c[0]), int(c[1]), int(c[2]), s]
                              for c in corners.astype(int)])

            # wireframe
            lc = Line3DCollection(segs, colors="lightgray",
                                  linewidths=0.8, zorder=1)
            ax.add_collection3d(lc)

            # scatter: size ∝ |value|
            sizes = 190 * (np.abs(vals) / max(vmax, 1e-12)) + 45
            ax.scatter(
                corners[:, 0], corners[:, 1], corners[:, 2],
                c=vals, cmap=CMAP, norm=norm,
                s=sizes, depthshade=True,
                edgecolors="gray", linewidths=0.3, zorder=5,
            )

            if annotate:
                # outward text offset from cube centre
                centre = np.full(3, (D - 1) / 2.0)
                diffs  = corners - centre
                norms_ = np.linalg.norm(diffs, axis=1, keepdims=True)
                norms_[norms_ == 0] = 1.0
                offsets = diffs / norms_ * 0.18
                for c, val, off in zip(corners, vals, offsets):
                    ax.text(
                        c[0] + off[0], c[1] + off[1], c[2] + off[2],
                        f"{val:+.3f}",
                        fontsize=5.8, ha="center", va="center",
                        color="black", zorder=10,
                    )

            # axis ticks
            ticks = list(range(D))
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.set_zticks(ticks)
            ax.set_xticklabels([f"a={v}" for v in ticks], fontsize=5.5)
            ax.set_yticklabels([f"b={v}" for v in ticks], fontsize=5.5)
            ax.set_zticklabels([f"c={v}" for v in ticks], fontsize=5.5)
            pad = 0.35
            ax.set_xlim(-pad, D - 1 + pad)
            ax.set_ylim(-pad, D - 1 + pad)
            ax.set_zlim(-pad, D - 1 + pad)
            ax.set_box_aspect([1, 1, 1])

            ax.set_title(f"site  {site}   –   {SPIN_LABELS[s]}",
                         fontsize=8.5, fontweight="bold", pad=3)
            ax.view_init(elev=22, azim=40)
            axes_list.append(ax)

    # ── colorbar: placed in own axes to the right, well clear of the 3D panels
    #    We add it AFTER subplots_adjust so the right=0.84 boundary is set.
    cbar_ax = fig.add_axes([0.87, 0.12, 0.022, 0.76])   # [left, bottom, w, h]
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax,
                      label=f"tensor value  T[a,b,c,s]")
    cb.ax.tick_params(labelsize=8)

    return fig


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    fig2 = build_figure(D=2)
    fig3 = build_figure(D=3)
    plt.show()
