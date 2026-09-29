#!/usr/bin/env python3
"""Diagnose LastIzar stages from downloaded results, Slurm logs, and sacct."""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA = REPO.parent / "data"
DEFAULT_SNAPSHOT = DATA / "distinVBCsJ2Continuation" / "LastIzar"
DEFAULT_OUTPUT = (
    REPO / "visual_elements" / "figs" / "VBCDiscriminator"
    / "j2_seed_continuations"
)
OBS_TEMPLATE = "D_{D}_chi_{CHI}_energy_magnetization_correlation.txt"


def parse_run(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    values = dict(re.findall(
        r'^export ([A-Z0-9_]+)="([^"]*)"$', text, re.MULTILINE
    ))
    match = re.search(r"^#SBATCH --job-name=(.+)$", text, re.MULTILINE)
    if match is None:
        raise ValueError(f"job name absent from {path}")
    values["JOB_NAME"] = match.group(1).strip()
    values["RUN_FILE"] = path.name
    return values


def read_sacct(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    rows: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8", errors="replace", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="|"):
            job_name = (row.get("JobName") or "").strip()
            job_id = (row.get("JobIDRaw") or "").strip()
            if not re.fullmatch(r"L[1-5].+", job_name):
                continue
            # Prefer the allocation row, not .batch/.extern steps.
            if not job_id.isdigit():
                continue
            previous = rows.get(job_name)
            if previous is None or int(job_id) > int(previous["JobIDRaw"]):
                rows[job_name] = row
    return rows


def read_squeue(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    rows: dict[str, dict[str, str]] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.strip().split("|", maxsplit=4)
        if len(fields) != 5:
            continue
        job_id, job_name, state, elapsed, reason = fields
        if not re.fullmatch(r"L[1-5].+", job_name):
            continue
        rows[job_name] = {
            "JobIDRaw": job_id, "JobName": job_name, "State": state,
            "Elapsed": elapsed, "Reason": reason,
        }
    return rows


def latest_log(log_root: Path, job_name: str, suffix: str) -> Path | None:
    candidates = list(log_root.glob(f"{job_name}-*-*.{suffix}"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def useful_tail(path: Path | None, limit: int = 24) -> str:
    if path is None or not path.is_file():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    lines = [line.strip() for line in lines if line.strip()]
    return "\n".join(lines[-limit:])


def classify_reason(state: str, error: str, stdout: str) -> str:
    text = f"{error}\n{stdout}"
    rules = (
        (r"CUDA out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED", "GPU out of memory"),
        (r"oom-kill|OUT_OF_MEMORY", "Slurm/system-memory OOM"),
        (r"No space left on device", "filesystem full"),
        (r"Required predecessor/seed tensor is missing|Configured input tensor is missing", "missing predecessor tensor"),
        (r"Stage returned without a complete tensor\+observables pair", "optimizer returned without complete outputs"),
        (r"FloatingPointError|non-finite|NaN|nan detected|environment collapsed", "numerical/CTMRG failure"),
        (r"ModuleNotFoundError|ImportError", "Python environment/import failure"),
        (r"unrecognized arguments|error: argument", "CLI mismatch"),
        (r"No such file or directory|FileNotFoundError", "missing file"),
        (r"TIMEOUT", "Slurm wall-time timeout"),
        (r"CANCELLED", "job cancelled"),
    )
    for pattern, label in rules:
        if re.search(pattern, text, re.IGNORECASE):
            return label
    exceptions = re.findall(
        r"^(?:[A-Za-z_][\w.]*Error|Exception|RuntimeError|AssertionError):?.*$",
        text, re.MULTILINE,
    )
    if exceptions:
        return exceptions[-1][:240]
    if state:
        return f"Slurm state {state}; inspect log excerpt"
    return "no decisive error found in downloaded logs"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    result_root = args.snapshot / "Results_LastIzar"
    log_root = args.snapshot / "slurm_logs"
    sacct = read_sacct(
        args.snapshot / "download_diagnostics" / "sacct_LastIzar.psv"
    )
    squeue = read_squeue(
        args.snapshot / "download_diagnostics" / "squeue_LastIzar.psv"
    )
    rows: list[dict[str, str]] = []
    for run_file in sorted((HERE / "jobs").glob("*.run")):
        config = parse_run(run_file)
        output = args.snapshot / config["OUTPUT_REL"]
        best = output / f'sweep_D{config["D"]}_chi{config["CHI"]}_best.pt'
        observation = output / OBS_TEMPLATE.format(**config)
        latest = output / f'sweep_D{config["D"]}_chi{config["CHI"]}_latest.pt'
        if best.is_file() and observation.is_file():
            result_status = "complete"
        elif output.is_dir() and (best.is_file() or latest.is_file()
                                  or any(output.iterdir())):
            result_status = "partial"
        else:
            result_status = "no_output"

        allocation = sacct.get(config["JOB_NAME"], {})
        queued = squeue.get(config["JOB_NAME"], {})
        state = (allocation.get("State") or queued.get("State") or "").strip()
        queue_reason = (queued.get("Reason") or "").strip()
        error_path = latest_log(log_root, config["JOB_NAME"], "error")
        out_path = latest_log(log_root, config["JOB_NAME"], "out")
        error_tail = useful_tail(error_path)
        out_tail = useful_tail(out_path)
        reason = ""
        if (result_status != "complete"
                and (state not in {"PENDING", "RUNNING"}
                     or "DependencyNeverSatisfied" in queue_reason)):
            reason = classify_reason(state, error_tail, out_tail)
        rows.append({
            "job_name": config["JOB_NAME"],
            "run_file": config["RUN_FILE"],
            "task": config["TASK_FAMILY"],
            "stage": config["STAGE_KIND"],
            "D": config["D"], "J2": config["J2"],
            "h": config["VBC_FIELD"], "optimizer": config["MAIN_KIND"],
            "result_status": result_status,
            "slurm_job_id": (allocation.get("JobIDRaw")
                             or queued.get("JobIDRaw") or "").strip(),
            "slurm_state": state,
            "queue_reason": queue_reason,
            "exit_code": (allocation.get("ExitCode") or "").strip(),
            "reason": reason,
            "error_log": str(error_path or ""),
            "stdout_log": str(out_path or ""),
            "error_excerpt": error_tail[-3000:],
            "stdout_excerpt": out_tail[-3000:],
        })

    by_name = {row["job_name"]: row for row in rows}
    for row in rows:
        if row["stage"] != "pin_h0" or row["result_status"] == "complete":
            continue
        parent_name = row["job_name"][:-1] + "p" if row["job_name"].endswith("z") else ""
        parent = by_name.get(parent_name)
        if parent and ("DEPENDENCY" in row["queue_reason"].upper()
                       or "DEPENDENCY" in row["slurm_state"].upper()):
            row["reason"] = (
                f"afterok blocked by {parent_name}: "
                f"{parent['slurm_state'] or parent['result_status']}; "
                f"{parent['reason'] or 'parent did not complete'}"
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "LastIzar_job_diagnostics.csv"
    fields = list(rows[0])
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    md_path = args.output_dir / "LastIzar_job_diagnostics.md"
    status_counts = Counter(row["result_status"] for row in rows)
    state_counts = Counter(row["slurm_state"] or "unknown" for row in rows)
    failures = [row for row in rows
                if row["result_status"] != "complete"
                and (row["slurm_state"] not in {"PENDING", "RUNNING"}
                     or "DependencyNeverSatisfied" in row["queue_reason"])]
    with md_path.open("w", encoding="utf-8") as stream:
        stream.write("# LastIzar job diagnostics\n\n")
        stream.write(f"Result states: `{dict(status_counts)}`  \n")
        stream.write(f"Slurm states: `{dict(state_counts)}`\n\n")
        stream.write("| job | task/stage | D | J2 | h | result | Slurm | cause |\n")
        stream.write("|---|---|---:|---:|---:|---|---|---|\n")
        for row in failures:
            cause = row["reason"].replace("|", "\\|").replace("\n", " ")
            stream.write(
                f"| {row['job_name']} | {row['task']}/{row['stage']} | "
                f"{row['D']} | {row['J2']} | {row['h']} | "
                f"{row['result_status']} | {row['slurm_state'] or 'unknown'} | "
                f"{cause} |\n"
            )

    print(f"Result states: {dict(status_counts)}")
    print(f"Slurm states: {dict(state_counts)}")
    print(f"Failure/blocked rows: {len(failures)}")
    for row in failures:
        print(
            f"  {row['job_name']} D={row['D']} J2={row['J2']} "
            f"h={row['h']}: {row['reason']}"
        )
    print(f"CSV: {csv_path}")
    print(f"Report: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
