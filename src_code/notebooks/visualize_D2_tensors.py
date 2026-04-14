"""
Visualize the 6 iPEPS site tensors (a,b,c,d,e,f) from a D=2 checkpoint.

Tensor shape: (D, D, D, d_phys) = (2, 2, 2, 2)
  axis 0: virtual bond 0
  axis 1: virtual bond 1
  axis 2: virtual bond 2
  axis 3: physical (spin) index

Layout per tensor:
  - columns → physical index s ∈ {0 (↑), 1 (↓)}
  - rows    → first virtual index i₀ ∈ {0, 1}
  - cell    → heatmap of T[i₀, :, :, s]  (2×2 matrix)
"""

import sys
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

DATA_DIR = Path("/home/chye/6ADctmrg/data/raw/6tensors_20260414_143622")

FILES = {
    "D=2, chi=3 (best)": DATA_DIR / "sweep_D2_chi3_best.pt",
    "D=2, chi=4 (best)": DATA_DIR / "sweep_D2_chi4_best.pt",
}

SITE_LABELS = list("abcdef")
SPIN_LABELS  = ["↑ (s=0)", "↓ (s=1)"]
VIRT_LABELS  = ["i₀=0", "i₀=1"]


# ──────────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_checkpoint(path: Path) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    tensors = {k: ckpt[k].numpy() for k in SITE_LABELS}
    meta = {k: ckpt[k] for k in ("D_bond", "chi", "energy", "step", "timestamp")}
    return tensors, meta


def print_tensors(tensors: dict, meta: dict, label: str):
    """Pretty-print all tensor values."""
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  D={meta['D_bond']}  chi={meta['chi']}  energy={meta['energy']:.10f}")
    print(f"  step={meta['step']}  time={meta['timestamp']}")
    print(f"{'='*70}")
    for name, T in tensors.items():
        print(f"\n  Tensor {name}  shape={T.shape}  (virt0, virt1, virt2, phys)")
        for s in range(T.shape[3]):
            spin = "↑" if s == 0 else "↓"
            print(f"    phys s={s} ({spin}):  T[i0,i1,i2,{s}]")
            for i0 in range(T.shape[0]):
                mat = T[i0, :, :, s]       # shape (D, D)
                for i1 in range(mat.shape[0]):
                    row = "  ".join(f"{mat[i1,i2]:+.6f}" for i2 in range(mat.shape[1]))
                    print(f"      i0={i0} i1={i1}: [{row}]")


def plot_tensors(tensors: dict, meta: dict, title: str, out_path: Path | None = None):
    """
    6×4 grid of heatmaps:
      - 6 rows  → sites a…f
      - 4 cols  → (i₀=0,s=↑), (i₀=0,s=↓), (i₀=1,s=↑), (i₀=1,s=↓)
    Each cell shows T[i₀, :, :, s] as a 2×2 heatmap.
    """
    n_sites = len(SITE_LABELS)
    D = list(tensors.values())[0].shape[0]
    d = list(tensors.values())[0].shape[3]
    n_cols = D * d   # 2 * 2 = 4

    fig_w = 2.2 * n_cols + 1.2
    fig_h = 2.2 * n_sites + 1.0
    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.01)

    gs = gridspec.GridSpec(n_sites, n_cols, figure=fig,
                           hspace=0.55, wspace=0.35)

    # global colour scale (shared across all heatmaps for fair comparison)
    all_vals = np.concatenate([T.ravel() for T in tensors.values()])
    vmax = np.max(np.abs(all_vals))
    vmin = -vmax

    col_titles = [
        f"i₀=0, s=↑", f"i₀=0, s=↓",
        f"i₀=1, s=↑", f"i₀=1, s=↓",
    ]

    for row, name in enumerate(SITE_LABELS):
        T = tensors[name]          # (D, D, D, d_phys)
        col = 0
        for i0 in range(D):
            for s in range(d):
                ax = fig.add_subplot(gs[row, col])
                mat = T[i0, :, :, s]
                im = ax.imshow(mat, cmap="RdBu_r", vmin=vmin, vmax=vmax,
                               aspect="equal", interpolation="nearest")

                # annotate each cell with its value
                for r in range(D):
                    for c_ in range(D):
                        ax.text(c_, r, f"{mat[r,c_]:+.3f}",
                                ha="center", va="center",
                                fontsize=7.5, color="black")

                # axis decoration
                if row == 0:
                    ax.set_title(col_titles[col], fontsize=8)
                if col == 0:
                    ax.set_ylabel(f"site {name}", fontsize=9, fontweight="bold")
                ax.set_xticks(range(D), [f"i₂={v}" for v in range(D)], fontsize=6)
                ax.set_yticks(range(D), [f"i₁={v}" for v in range(D)], fontsize=6)
                col += 1

    # shared colour bar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.018, 0.7])
    sm = plt.cm.ScalarMappable(cmap="RdBu_r",
                                norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label="tensor value")

    plt.tight_layout(rect=[0, 0, 0.91, 1])
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {out_path}")
    return fig


def plot_comparison_bar(tensors_chi3: dict, tensors_chi4: dict,
                        out_path: Path | None = None):
    """
    Bar chart: Frobenius norm of each tensor at chi=3 vs chi=4,
    and a side-by-side scatter of all 16 elements per site.
    """
    n_sites = len(SITE_LABELS)
    fig, axes = plt.subplots(2, n_sites, figsize=(14, 5),
                             gridspec_kw=dict(hspace=0.5, wspace=0.35))
    fig.suptitle("D=2 tensor comparison: chi=3 vs chi=4", fontsize=11,
                 fontweight="bold")

    for col, name in enumerate(SITE_LABELS):
        T3 = tensors_chi3[name].ravel()
        T4 = tensors_chi4[name].ravel()
        n = len(T3)
        idx = np.arange(n)

        # top row: element-by-element scatter
        ax = axes[0, col]
        ax.scatter(T3, T4, s=20, color="steelblue", edgecolors="none", alpha=0.8)
        lim = max(np.max(np.abs(T3)), np.max(np.abs(T4))) * 1.15
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8, label="y=x")
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_title(f"site {name}", fontsize=9)
        if col == 0:
            ax.set_xlabel("chi=3 value", fontsize=8)
            ax.set_ylabel("chi=4 value", fontsize=8)
        ax.tick_params(labelsize=7)

        # bottom row: element values side by side
        ax2 = axes[1, col]
        ax2.bar(idx - 0.2, T3, 0.35, label="chi=3", color="steelblue", alpha=0.8)
        ax2.bar(idx + 0.2, T4, 0.35, label="chi=4", color="tomato",    alpha=0.8)
        ax2.axhline(0, color="k", lw=0.5)
        ax2.set_xticks(idx, [str(i) for i in idx], fontsize=5.5)
        ax2.set_xlabel("element index", fontsize=8)
        if col == 0:
            ax2.set_ylabel("value", fontsize=8)
            ax2.legend(fontsize=7)
        ax2.tick_params(labelsize=7)

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {out_path}")
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print_only = "--print-only" in sys.argv

    tensors_chi3, meta_chi3 = load_checkpoint(FILES["D=2, chi=3 (best)"])
    tensors_chi4, meta_chi4 = load_checkpoint(FILES["D=2, chi=4 (best)"])

    # ── text output ──────────────────────────────────────────────────────────
    print_tensors(tensors_chi3, meta_chi3, "D=2, chi=3 (best)")
    print_tensors(tensors_chi4, meta_chi4, "D=2, chi=4 (best)")

    # ── Frobenius norms ──────────────────────────────────────────────────────
    print("\n── Frobenius norms ──")
    for name in SITE_LABELS:
        n3 = np.linalg.norm(tensors_chi3[name])
        n4 = np.linalg.norm(tensors_chi4[name])
        print(f"  site {name}:  chi=3 ‖T‖={n3:.6f}   chi=4 ‖T‖={n4:.6f}")

    # ── pairwise Frobenius distances ─────────────────────────────────────────
    print("\n── Frobenius distances chi3 vs chi4 ──")
    for name in SITE_LABELS:
        diff = np.linalg.norm(tensors_chi3[name] - tensors_chi4[name])
        print(f"  site {name}:  ‖T_chi3 − T_chi4‖={diff:.6f}")

    if print_only:
        sys.exit(0)

    # ── heatmap figures ──────────────────────────────────────────────────────
    out_chi3 = DATA_DIR / "viz_D2_chi3_tensors.png"
    out_chi4 = DATA_DIR / "viz_D2_chi4_tensors.png"
    out_cmp  = DATA_DIR / "viz_D2_chi3_vs_chi4.png"

    title_chi3 = (f"D=2, chi=3  |  energy={meta_chi3['energy']:.8f}  "
                  f"|  step={meta_chi3['step']}  |  {meta_chi3['timestamp']}")
    title_chi4 = (f"D=2, chi=4  |  energy={meta_chi4['energy']:.8f}  "
                  f"|  step={meta_chi4['step']}  |  {meta_chi4['timestamp']}")

    print("\nGenerating heatmaps …")
    fig1 = plot_tensors(tensors_chi3, meta_chi3, title_chi3, out_chi3)
    fig2 = plot_tensors(tensors_chi4, meta_chi4, title_chi4, out_chi4)
    fig3 = plot_comparison_bar(tensors_chi3, tensors_chi4, out_cmp)
    plt.show()
    print("Done.")
