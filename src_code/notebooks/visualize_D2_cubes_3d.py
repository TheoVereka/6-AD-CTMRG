"""
Interactive 3D cube visualization of D=2 iPEPS tensors a, b, c, d  (chi=4).

Each tensor T has shape (D=2, D=2, D=2, d_phys=2).
The 8 corners of a unit cube sit at (i₀,i₁,i₂) ∈ {0,1}³.
Corner colour + size  →  tensor value  T[i₀, i₁, i₂, s].

Layout: 4 rows (sites a,b,c,d)  ×  2 cols (spin ↑, spin ↓)

Rotate each subplot interactively by clicking and dragging.
"""

import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401 (registers projection)
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from pathlib import Path

# ── data ─────────────────────────────────────────────────────────────────────
DATA_DIR = Path("/home/chye/6ADctmrg/data/raw/6tensors_20260414_143622")
PT_FILE  = DATA_DIR / "sweep_D2_chi4_best.pt"

SITES = list("abcd")                 # 4 sites to show
SPIN_LABELS = ["spin ↑  (s=0)", "spin ↓  (s=1)"]

ckpt    = torch.load(PT_FILE, map_location="cpu", weights_only=False)
tensors = {k: ckpt[k].numpy() for k in SITES}
energy  = ckpt["energy"]
chi     = ckpt["chi"]

# ── unit-cube geometry ────────────────────────────────────────────────────────
# 8 corners  (i₀, i₁, i₂) ∈ {0,1}³  —  iterate in numpy / torch order
corners = np.array([
    [i0, i1, i2]
    for i0 in range(2)
    for i1 in range(2)
    for i2 in range(2)
], dtype=float)                       # shape (8, 3)

# 12 edges: pairs of corner indices that differ in exactly one coordinate
_CUBE_EDGES = [
    (0, 1), (0, 2), (0, 4),
    (1, 3), (1, 5),
    (2, 3), (2, 6),
    (3, 7),
    (4, 5), (4, 6),
    (5, 7),
    (6, 7),
]

def _edge_segments():
    return [[corners[a].tolist(), corners[b].tolist()] for a, b in _CUBE_EDGES]

# anchor text labels outward from cube centre (0.5, 0.5, 0.5)
_CENTRE = np.array([0.5, 0.5, 0.5])
_OFFSETS = ((corners - _CENTRE) / np.linalg.norm(corners - _CENTRE, axis=1, keepdims=True) * 0.16)

# ── global colour scale (over all 4 sites, both spins) ───────────────────────
all_vals = np.concatenate([tensors[s].ravel() for s in SITES])
vmax = float(np.max(np.abs(all_vals)))
vmin = -vmax
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
cmap = matplotlib.cm.RdBu_r

# ── figure ────────────────────────────────────────────────────────────────────
n_rows, n_cols = len(SITES), 2
fig = plt.figure(figsize=(11, 17))
fig.suptitle(
    f"D=2 iPEPS tensors  a, b, c, d     chi={chi}     energy = {energy:.10f}\n"
    "Corner colour & size  →  T[i₀, i₁, i₂, s]     (click & drag to rotate)",
    fontsize=11, fontweight="bold", y=0.995,
)

segs = _edge_segments()
axes_3d: dict = {}

for row, site in enumerate(SITES):
    T = tensors[site]                 # (2, 2, 2, 2)

    for col, s in enumerate([0, 1]):
        ax: Axes3D = fig.add_subplot(n_rows, n_cols,
                                     row * n_cols + col + 1,
                                     projection="3d")

        # ── values at the 8 corners ──────────────────────────────────────────
        vals = np.array([T[int(c[0]), int(c[1]), int(c[2]), s] for c in corners])

        # ── wireframe cube ───────────────────────────────────────────────────
        lc = Line3DCollection(segs, colors="lightgray", linewidths=0.9, zorder=1)
        ax.add_collection3d(lc)

        # ── corner scatter ───────────────────────────────────────────────────
        # size: linear in |value| so zero is still visible
        sizes = 200 * (np.abs(vals) / vmax) + 60
        ax.scatter(
            corners[:, 0], corners[:, 1], corners[:, 2],
            c=vals, cmap=cmap, norm=norm,
            s=sizes, depthshade=True,
            edgecolors="gray", linewidths=0.4, zorder=5,
        )

        # ── corner value annotations ─────────────────────────────────────────
        for k, (c, val, off) in enumerate(zip(corners, vals, _OFFSETS)):
            ax.text(
                c[0] + off[0], c[1] + off[1], c[2] + off[2],
                f"{val:+.3f}",
                fontsize=6.2, ha="center", va="center",
                color="black", zorder=10,
            )

        # ── corner index labels on axis ticks ────────────────────────────────
        ax.set_xticks([0, 1]); ax.set_xticklabels(["i₀=0", "i₀=1"], fontsize=6.5)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["i₁=0", "i₁=1"], fontsize=6.5)
        ax.set_zticks([0, 1]); ax.set_zticklabels(["i₂=0", "i₂=1"], fontsize=6.5)

        ax.set_xlim(-0.25, 1.25)
        ax.set_ylim(-0.25, 1.25)
        ax.set_zlim(-0.25, 1.25)
        ax.set_box_aspect([1, 1, 1])

        ax.set_title(f"site  {site}   –   {SPIN_LABELS[s]}",
                     fontsize=9, fontweight="bold", pad=4)

        # nice initial viewing angle
        ax.view_init(elev=22, azim=40)
        axes_3d[(site, s)] = ax

# ── shared colour bar ─────────────────────────────────────────────────────────
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=list(axes_3d.values()),
                    shrink=0.35, pad=0.06, aspect=20,
                    label="tensor value  T[i₀,i₁,i₂,s]")
cbar.ax.tick_params(labelsize=8)

fig.subplots_adjust(left=0.04, right=0.88, top=0.955, bottom=0.03,
                    hspace=0.35, wspace=0.15)
plt.show()
