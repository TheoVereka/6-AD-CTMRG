"""Prepare the user's fixed five-normal/five-long J2 allocation and account for QOS slots."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

BUNDLE = Path(__file__).resolve().parent
ORDER = ["0.26", "0.25", "0.265", "0.32", "0.20", "0.27", "0.245", "0.275", "0.24", "0.28"]
LIMIT = 16
WALLTIME = {"normal": "71:59:50", "long": "167:59:50"}


def allocation(normal_running: int, long_running: int) -> list[dict]:
    if not 0 <= normal_running <= LIMIT or not 0 <= long_running <= LIMIT:
        raise ValueError("Each running count must be between 0 and 16.")
    jobs = []
    for j2 in ORDER:
        tag = j2.replace(".", "p")
        for pair in (1, 2, 3):
            qos = "normal" if len(jobs) < 15 else "long"
            jobs.append({
                "priority": len(jobs) + 1, "J2": j2, "D": 8, "chi": 80,
                "pair": pair, "normal_env": pair, "swapped_env": {1: 1, 2: 3, 3: 2}[pair],
                "qos": qos, "walltime": WALLTIME[qos],
                "job_name": f"r2j{tag.replace('p', '')}D8p{pair}",
                "job_path": f"jobs/J2_{tag}_D8_pair{pair}.run",
                "output_directory": f"Results/J2_{tag}/D_8/chi_80/pair_{pair}",
            })
    return jobs


def job_text(job: dict) -> str:
    return f'''#!/usr/bin/env bash
#SBATCH --qos={job["qos"]}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --time={job["walltime"]}
#SBATCH --exclude=i39
#SBATCH --job-name={job["job_name"]}
#SBATCH --output=slurm_logs/%x-%N-%j.out
#SBATCH --error=slurm_logs/%x-%N-%j.error

set -euo pipefail
if [[ -n "${{BUNDLE_DIR:-}}" ]]; then
    BUNDLE_DIR="$(cd -- "${{BUNDLE_DIR}}" && pwd)"
elif [[ -f "${{PWD}}/run_pair.sh" ]]; then
    BUNDLE_DIR="${{PWD}}"
elif [[ -n "${{SLURM_SUBMIT_DIR:-}}" && -f "${{SLURM_SUBMIT_DIR}}/run_pair.sh" ]]; then
    BUNDLE_DIR="${{SLURM_SUBMIT_DIR}}"
else
    BUNDLE_DIR="$(cd -- "$(dirname -- "${{BASH_SOURCE[0]}}")/.." && pwd)"
fi
export BUNDLE_DIR
exec bash "${{BUNDLE_DIR}}/run_pair.sh" {job["J2"]} {job["pair"]}
'''


def validate_seeds() -> None:
    with (BUNDLE / "seed_manifest.csv").open(encoding="utf-8", newline="") as handle:
        seeds = list(csv.DictReader(handle))
    if [row["J2"] for row in seeds] != ORDER:
        raise ValueError("seed_manifest.csv must contain the ten requested J2 values in priority order.")
    for row in seeds:
        path = BUNDLE / row["relative_path"]
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"Missing or modified seed: {row['relative_path']}")


def read_queue(user: str) -> list[dict]:
    command = ["squeue", "--user", user, "--states", "RUNNING", "--noheader", "--format", "%i|%j|%q"]
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    jobs = []
    for line in result.stdout.splitlines():
        if line.strip():
            job_id, name, qos = (part.strip() for part in line.split("|", 2))
            jobs.append({"job_id": job_id, "name": name, "qos": qos})
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--auto", action="store_true", help="Query actual squeue on the Izar login node.")
    parser.add_argument("--user", default="chye", help="Slurm user for --auto.")
    parser.add_argument("--normal-running", type=int, help="Explicit current normal-QOS running-job count.")
    parser.add_argument("--long-running", type=int, help="Explicit current long-QOS running-job count.")
    parser.add_argument("--source", default="user-supplied current QOS counts",
                        help="Describe the actual queue query or user confirmation.")
    parser.add_argument("--known-job", action="append", default=[], metavar="ID|NAME|QOS|WALLTIME",
                        help="Record an actual user-provided job snapshot with explicit counts.")
    parser.add_argument("--preview", action="store_true", help="Print the allocation without changing files.")
    args = parser.parse_args()
    if args.auto:
        if args.normal_running is not None or args.long_running is not None or args.known_job:
            parser.error("Use --auto or the two explicit counts, not both.")
        try:
            snapshot = read_queue(args.user)
        except (OSError, subprocess.CalledProcessError, ValueError) as error:
            parser.exit(2, f"Cannot obtain current QOS from squeue; no allocation changed: {error}\n")
        normal_running = sum(job["qos"] == "normal" for job in snapshot)
        long_running = sum(job["qos"] == "long" for job in snapshot)
        source = f"Live squeue --user {args.user} --states RUNNING --format %i|%j|%q"
    else:
        if args.normal_running is None or args.long_running is None:
            parser.error("Supply --auto or BOTH --normal-running and --long-running.")
        snapshot = []
        for record in args.known_job:
            parts = record.split("|")
            if len(parts) != 4 or parts[2] not in WALLTIME:
                parser.error("--known-job requires ID|NAME|QOS|WALLTIME with normal or long QOS.")
            snapshot.append(dict(zip(("job_id", "name", "qos", "walltime"), parts)))
        if not snapshot:
            snapshot = None
        normal_running, long_running = args.normal_running, args.long_running
        if snapshot and any(sum(job["qos"] == qos for job in snapshot) > count
                            for qos, count in [("normal", normal_running), ("long", long_running)]):
            parser.error("Known-job snapshot is inconsistent with the supplied running counts.")
        source = args.source
    try:
        jobs = allocation(normal_running, long_running)
        validate_seeds()
    except ValueError as error:
        parser.exit(2, f"No allocation changed: {error}\n")
    counts = {qos: sum(job["qos"] == qos for job in jobs) for qos in WALLTIME}
    split = [j2 for j2 in ORDER if len({job["qos"] for job in jobs if job["J2"] == j2}) > 1]
    free = {"normal": LIMIT - normal_running, "long": LIMIT - long_running}
    report = {
        "ready": True, "created_utc": datetime.now(timezone.utc).isoformat(), "source": source,
        "queue_snapshot": snapshot, "normal_running": normal_running, "long_running": long_running,
        "qos_running_limit": LIMIT, "available_slots_at_snapshot": free, "new_jobs": counts,
        "combined_requested_slots": {"normal": normal_running + counts["normal"],
                                     "long": long_running + counts["long"]},
        "minimum_jobs_waiting_for_QOS_slots": sum(max(0, counts[qos] - free[qos]) for qos in counts),
        "total_new_jobs": len(jobs), "J2_priority": ORDER, "J2_split_between_QOS": split,
        "allocation_rule": "User-directed fixed allocation: first five J2 values normal (three days), last five long (seven days). Queue refresh only updates capacity accounting; it never changes this allocation.",
        "normal_J2": ORDER[:5], "long_J2": ORDER[5:],
        "scope": "QOS-slot accounting only; available GPUs, priorities and partition limits can still cause waiting.",
    }
    if not args.preview:
        (BUNDLE / "jobs").mkdir(exist_ok=True)
        for job in jobs:
            (BUNDLE / job["job_path"]).write_text(job_text(job), encoding="utf-8", newline="\n")
        with (BUNDLE / "job_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(jobs[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(jobs)
        (BUNDLE / "qos_allocation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**report, "preview_only": args.preview}, indent=2))


if __name__ == "__main__":
    main()
