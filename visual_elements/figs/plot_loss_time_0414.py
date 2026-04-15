#!/usr/bin/env python3
"""
Plot log|loss - E_ref| vs log(cumulative time) for AD-CTMRG runs in 0414core.

Two figures are produced:
  • Figure 1 – all job-*.out files whose Output dir maps to a  6tensors__*  folder
  • Figure 2 – all job-*.out files whose Output dir maps to a  neel_symmetrized__*  folder

Matching uses the "Output dir" line written by main_unrestricted.py.
If that line is absent, a ±1 s timestamp fallback is used (Started vs folder name).

Penalty lines (loss > 999 999) are excluded.
E_ref = minimum loss across the runs in each figure.
"""

import re
import os
import glob
from datetime import datetime, timedelta
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────
HERE      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.normpath(os.path.join(HERE, '..', '..', '..', 'data', '0414core'))
OUT_FILES = sorted(glob.glob(os.path.join(DATA_DIR, "job-*.out")))

# ── Regexes ───────────────────────────────────────────────────────────────
# Step lines:   step   242  ctm=  0  loss=+1000000.0000000000  …  164/8999910s
STEP_RE = re.compile(
    r"^\s*step\s+\d+\s+ctm=\s*\d+\s+loss=([+-]?\d+\.\d+)\s+"
    r".*?\s+(\d+)/\d+s\s*$"
)
# Block header:  ┌── D=2  chi=3  budget=...
BLOCK_RE = re.compile(r"D=\d+\s+chi=\d+\s+budget=")
# Output dir:   Output dir   : /scratch/.../RUN_FOLDER_NAME
OUTDIR_RE = re.compile(r"Output dir\s*:\s*(\S+)")
# Started:      Started      : 2026-04-14 18:20:11
STARTED_RE = re.compile(r"Started\s*:\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})")
# J2 value from folder name:  J2_0p23  →  0.23
J2_RE = re.compile(r"J2_(0p\d+)")

PENALTY = 999_999.0   # OOM penalty threshold

# ── Parse folder timestamps from local data dir ───────────────────────────
# Maps  folder_basename → (type_prefix, datetime)
local_folders: dict[str, tuple[str, datetime]] = {}
TS_RE = re.compile(r"(\d{8})_(\d{6})$")   # YYYYMMDD_HHMMSS at end of name

for entry in os.scandir(DATA_DIR):
    if not entry.is_dir():
        continue
    name = entry.name
    m = TS_RE.search(name)
    if not m:
        continue
    dt = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    if name.startswith("6tensors__"):
        local_folders[name] = ("6tensors", dt)
    elif name.startswith("neel_symmetrized__"):
        local_folders[name] = ("neel_symmetrized", dt)
    # sym6 intentionally omitted


def _folder_from_started(started: datetime) -> tuple[str, str] | None:
    """Return (run_type, folder_basename) for the local folder whose timestamp
    is within 1 second of *started*, or None."""
    for basename, (rtype, fdt) in local_folders.items():
        if abs((started - fdt).total_seconds()) <= 1.0:
            return rtype, basename
    return None


# ── Step-data parser (identical logic to plot_loss_time.py) ───────────────

def parse_file(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (times, losses) with cumulative wall-time across D/chi blocks."""
    times: list[float] = []
    losses: list[float] = []
    cumulative_offset = 0.0
    last_time_in_block = 0.0
    in_block = False

    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if BLOCK_RE.search(line):
                if in_block:
                    cumulative_offset += last_time_in_block
                in_block = True
                last_time_in_block = 0.0
                continue
            m = STEP_RE.match(line)
            if m:
                loss = float(m.group(1))
                t    = float(m.group(2))
                if loss > PENALTY:
                    continue
                last_time_in_block = t
                times.append(cumulative_offset + t)
                losses.append(loss)

    return np.array(times), np.array(losses)


# ── Read header info + step data for every job file ───────────────────────

def _j2_label(folder_basename: str) -> str:
    """Convert  J2_0p23  →  J₂=0.23, J2_0p2 → J₂=0.2, etc."""
    m = J2_RE.search(folder_basename)
    if not m:
        return folder_basename
    digits = m.group(1)[2:]          # strip leading "0p"
    # '23' → '0.23',  '2' → '0.2',  '3' → '0.3'
    return rf"$J_2={0}.{digits}$"


records: dict[str, list[tuple[str, str, np.ndarray, np.ndarray]]] = {
    "6tensors": [],
    "neel_symmetrized": [],
}

for fpath in OUT_FILES:
    # ── scan header (first ~40 lines for speed) ──────────────────────────
    run_type = None
    folder_basename = None
    started_dt = None

    with open(fpath, "r", encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            if i > 80:
                break
            m = OUTDIR_RE.search(line)
            if m:
                # Output dir may be split across two lines — reconstruct
                raw = m.group(1).rstrip()
                # some paths are line-wrapped with a whitespace continuation
                folder_basename = os.path.basename(raw)
                if folder_basename in local_folders:
                    run_type = local_folders[folder_basename][0]
                break
            m = STARTED_RE.search(line)
            if m and started_dt is None:
                started_dt = datetime.strptime(m.group(1).strip(), "%Y-%m-%d %H:%M:%S")

    # Fallback: use ±1 s timestamp match
    if run_type is None and started_dt is not None:
        result = _folder_from_started(started_dt)
        if result:
            run_type, folder_basename = result

    if run_type not in records:
        continue   # sym6 or no match → skip

    t, loss = parse_file(fpath)
    if len(t) == 0:
        print(f"  skip (no data): {os.path.basename(fpath)}")
        continue

    label = _j2_label(folder_basename)
    records[run_type].append((label, folder_basename, t, loss))

# ── Sort each group by J2 value for consistent colour ordering ────────────
def _sort_key(item):
    m = J2_RE.search(item[1])
    return float("0." + m.group(1)[2:]) if m else 0.0

for key in records:
    records[key].sort(key=_sort_key)

# ── Plotting helper ───────────────────────────────────────────────────────
_TITLES = {
    "6tensors":          "Fully unrestricted 6-tensor ansatz",
    "neel_symmetrized":  "Néel-symmetrized single-tensor ansatz",
}
_FNAMES = {
    "6tensors":         "loss_vs_time_6tensors.{ext}",
    "neel_symmetrized": "loss_vs_time_neel.{ext}",
}

COLORS = plt.cm.tab10.colors

for run_type, items in records.items():
    if not items:
        print(f"No data for {run_type}, skipping figure.")
        continue

    # E_ref for this figure
    all_losses = np.concatenate([loss for _, _, _, loss in items])
    E_ref = float(all_losses.min())
    print(f"\n[{run_type}]  E_ref = {E_ref:.10f}  ({len(items)} runs)")

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (label, folder, t, loss) in enumerate(items):
        diff = np.abs(loss - E_ref)
        mask = diff > 0
        t_plot = t[mask]
        y_plot = np.log10(diff[mask])
        if len(t_plot) == 0:
            print(f"  skip (all == E_ref): {folder}")
            continue
        ax.plot(t_plot, y_plot, linewidth=0.9, label=label,
                color=COLORS[i % len(COLORS)], alpha=0.88)
        print(f"  {label:18s}  n_steps={len(t):5d}  t_max={t[-1]:,.0f}s  "
              f"E_min={loss.min():.8f}")

    ax.set_xscale("log")
    ax.set_xlabel("Cumulative wall time  (seconds, log scale)", fontsize=12)
    ax.set_ylabel(r"$\log_{10}\,|E - E_{\rm ref}|$", fontsize=12)
    ax.set_title(
        f"AD-CTMRG convergence — {_TITLES[run_type]}\n"
        rf"$E_{{\rm ref}}$ = {E_ref:.6f}  (min across all $J_2$ runs)",
        fontsize=13,
    )
    ax.legend(fontsize=10, loc="upper right", framealpha=0.8,
              title=r"$J_2$  values")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_ylim(bottom=-2.1)

    fig.tight_layout()

    for ext in ("png", "pdf"):
        out = os.path.join(DATA_DIR, _FNAMES[run_type].format(ext=ext))
        fig.savefig(out, dpi=200 if ext == "png" else 72)
        print(f"  Saved → {out}")

    plt.close(fig)

print("\nDone.")
