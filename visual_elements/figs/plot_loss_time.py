#!/usr/bin/env python3
"""
Plot log|loss - E_ref| vs log(cumulative time) for 6 AD-CTMRG runs.

For each .out file, extracts all 'step ... loss=... N/Bs' lines,
glues all (D, chi) blocks in appearance order, and makes the time
monotonically increasing: each new block's time starts from the
last time of the previous block.

Penalty lines (loss = +1e6 from OOM) are excluded.

E_ref = global minimum loss across all 6 runs (best energy reached).
Both axes are log-scaled.
"""

import re
import glob
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_FILES = sorted(glob.glob(os.path.join(DATA_DIR, "*.out")))

# Regex for step lines:
#   step   242  ctm=  0  loss=+1000000.0000000000  Δ=+inf  164/8999910s
STEP_RE = re.compile(
    r"^\s*step\s+\d+\s+ctm=\s*\d+\s+loss=([+-]?\d+\.\d+)\s+"
    r".*?\s+(\d+)/\d+s\s*$"
)
# Block header:  ┌── D=2  chi=3  budget=...
BLOCK_RE = re.compile(r"D=(\d+)\s+chi=(\d+)\s+budget=")

PENALTY = 999_999.0  # threshold: anything above this is an OOM penalty


def parse_file(path):
    """Return (times, losses) arrays with cumulative time across blocks."""
    times = []
    losses = []
    cumulative_offset = 0.0
    last_time_in_block = 0.0
    in_block = False

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            # Detect block header → advance offset
            if BLOCK_RE.search(line):
                if in_block:
                    # Close previous block: add its last time to offset
                    cumulative_offset += last_time_in_block
                in_block = True
                last_time_in_block = 0.0
                continue

            m = STEP_RE.match(line)
            if m:
                loss = float(m.group(1))
                t = float(m.group(2))
                # Skip OOM penalty points
                if loss > PENALTY:
                    continue
                last_time_in_block = t
                times.append(cumulative_offset + t)
                losses.append(loss)

    return np.array(times), np.array(losses)


# ── Load all data ─────────────────────────────────────────────────────────
all_data = []
for fpath in OUT_FILES:
    label = os.path.splitext(os.path.basename(fpath))[0]
    t, loss = parse_file(fpath)
    if len(t) == 0:
        print(f"WARNING: no data in {fpath}")
    all_data.append((label, t, loss))

# E_ref = global minimum energy across every run (best achieved value)
all_losses = np.concatenate([loss for _, _, loss in all_data if len(loss) > 0])
E_ref = float(all_losses.min())
print(f"E_ref (global min) = {E_ref:.10f}")

# ── Plotting ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))

colors = plt.cm.tab10.colors
for i, (label, t, loss) in enumerate(all_data):
    if len(t) == 0:
        continue

    # log|loss - E_ref|; drop points where |loss - E_ref| == 0 (log undefined)
    diff = np.abs(loss - E_ref)
    mask = diff > 0
    t_plot = t[mask]
    y_plot = np.log10(diff[mask])

    if len(t_plot) == 0:
        print(f"WARNING: all points equal E_ref in {label}, nothing to plot.")
        continue

    ax.plot(t_plot, y_plot, linewidth=0.8, label=label,
            color=colors[i % len(colors)], alpha=0.85)

ax.set_xscale("log")
ax.set_xlabel("Cumulative wall time  (seconds, log scale)", fontsize=12)
ax.set_ylabel(r"$\log_{10}\,|E - E_{\rm ref}|$", fontsize=12)
ax.set_title(
    f"AD-CTMRG convergence  —  all D, χ blocks glued sequentially\n"
    f"$E_{{\\rm ref}}$ = {E_ref:.6f}  (global minimum)",
    fontsize=13,
)
ax.legend(fontsize=9, loc="best")
ax.grid(True, which="both", alpha=0.3)
ax.set_ylim(bottom=-2.1)
ax.set_xlim(left=4,right=1e5)  # start from 0.01s to avoid clutter at very early times
fig.tight_layout()

out_png = os.path.join(DATA_DIR, "loss_vs_time.png")
out_pdf = os.path.join(DATA_DIR, "loss_vs_time.pdf")
fig.savefig(out_png, dpi=200)
fig.savefig(out_pdf)
print(f"Saved → {out_png}")
print(f"Saved → {out_pdf}")
plt.close(fig)
