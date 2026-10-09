"""Validate launch preparation in disposable copies; never query or submit to Izar."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BUNDLE = ROOT / "models" / "Renyi2Twoc3Izar"
BASH = Path(r"D:\Programs\Git\bin\bash.exe")
spec = importlib.util.spec_from_file_location("prepare_jobs", BUNDLE / "prepare_jobs.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)
checks = []

for normal, long, expected in [(1, 0, (15, 15)), (1, 1, (15, 15)), (0, 2, (15, 15)), (2, 0, (15, 15)), (3, 1, (15, 15))]:
    jobs = prepare.allocation(normal, long)
    count = (sum(job["qos"] == "normal" for job in jobs), sum(job["qos"] == "long" for job in jobs))
    assert count == expected, (normal, long, count)
    assert [job["J2"] for job in jobs] == [j2 for j2 in prepare.ORDER for _ in range(3)]
    assert [job["pair"] for job in jobs] == [1, 2, 3] * 10
    assert all(job["qos"] == "normal" for job in jobs[:15])
    assert all(job["qos"] == "long" for job in jobs[15:])
    if (normal, long) == (1, 0):
        assert count[0] + normal == 16 and count[1] + long == 15
    checks.append({"case": "fixed user allocation and capacity arithmetic", "normal_running": normal, "long_running": long,
                   "normal_new": count[0], "long_new": count[1], "hypothetical_test_only": True})

completed = subprocess.CompletedProcess([], 0, " 123|old-a|normal\n124|old-b|long\n", "")
with patch.object(prepare.subprocess, "run", return_value=completed) as mocked:
    queue = prepare.read_queue("chye")
    assert [job["qos"] for job in queue] == ["normal", "long"]
    assert mocked.call_args.args[0] == ["squeue", "--user", "chye", "--states", "RUNNING", "--noheader", "--format", "%i|%j|%q"]
checks.append({"case": "parse actual squeue format", "mocked_only": True})

production_before = {path.relative_to(BUNDLE).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in BUNDLE.rglob("*") if path.is_file()}
with tempfile.TemporaryDirectory(prefix="fixture_", dir=HERE) as directory:
    temporary = Path(directory).resolve()
    assert temporary.is_relative_to(HERE.resolve())
    fake = temporary / "bundle"
    shutil.copytree(BUNDLE, fake)
    (fake / "qos_allocation.json").write_text('{"ready": false}\n', encoding="utf8")
    pending = subprocess.run([str(BASH), "--noprofile", "--norc", "submit_all.sh", "--dry-run"],
                             cwd=fake, capture_output=True, text=True)
    assert pending.returncode == 2 and "QOS allocation pending" in pending.stderr
    checks.append({"case": "pending allocation prevents submission", "passed": True})

    for normal, long in [(1, 0), (1, 1), (0, 2), (2, 0), (3, 1)]:
        generated = subprocess.run([sys.executable, "-B", "prepare_jobs.py", "--normal-running", str(normal),
                                    "--long-running", str(long), "--source", "TEST FIXTURE ONLY, not current Izar"],
                                   cwd=fake, capture_output=True, text=True, check=True)
        report = json.loads(generated.stdout)
        assert report["total_new_jobs"] == 30
        assert report["minimum_jobs_waiting_for_QOS_slots"] == max(0, normal - 1) + max(0, long - 1)
        assert report["new_jobs"] == {"normal": 15, "long": 15}
        with (fake / "job_manifest.csv").open(encoding="utf8", newline="") as handle:
            jobs = list(csv.DictReader(handle))
        assert len(jobs) == 30
        for job in jobs:
            text = (fake / job["job_path"]).read_text(encoding="utf8")
            assert f'#SBATCH --qos={job["qos"]}\n' in text
            assert f'#SBATCH --time={job["walltime"]}\n' in text
            assert f'run_pair.sh" {job["J2"]} {job["pair"]}\n' in text
            subprocess.run([str(BASH), "--noprofile", "--norc", "-n", job["job_path"]], cwd=fake, check=True)
        dry = subprocess.run([str(BASH), "--noprofile", "--norc", "submit_all.sh", "--dry-run"],
                             cwd=fake, capture_output=True, text=True, check=True)
        assert "Dry run complete: 30 jobs; nothing submitted." in dry.stdout
        assert len([line for line in dry.stdout.splitlines() if "J2=" in line]) == 30
        checks.append({"case": "generated 30 headers and dry run", "normal_running": normal,
                       "long_running": long, "passed": True, "disposable_fixture_only": True})

    for name in ["run_pair.sh", "submit_all.sh"]:
        subprocess.run([str(BASH), "--noprofile", "--norc", "-n", name], cwd=fake, check=True)
    assert "#SBATCH" not in (fake / "run_pair.sh").read_text(encoding="utf8")

    # A Slurm-spooled wrapper has no reliable path to the original bundle.
    original = fake / "jobs" / "J2_0p26_D8_pair1.run"
    stub_text = original.read_text(encoding="utf8").split("exec bash ", 1)[0] + 'cygpath -w "${BUNDLE_DIR}"\n'
    spool = temporary / "spool"
    spool.mkdir()
    stub = spool / "slurm_script"
    stub.write_text(stub_text, encoding="utf8", newline="\n")
    local_stub = fake / "jobs" / "test_directory.run"
    local_stub.write_text(stub_text, encoding="utf8", newline="\n")
    for name, script, cwd, additions in [
        ("exported bundle", stub, temporary, {"BUNDLE_DIR": fake.as_posix()}),
        ("sbatch chdir", stub, fake, {}),
        ("Slurm submit directory", stub, temporary, {"SLURM_SUBMIT_DIR": fake.as_posix()}),
        ("local wrapper fallback", local_stub, temporary, {}),
    ]:
        environment = os.environ.copy()
        for key in ["BUNDLE_DIR", "SLURM_SUBMIT_DIR"]:
            environment.pop(key, None)
        environment.update(additions)
        result = subprocess.run([str(BASH), "--noprofile", "--norc", str(script)], cwd=cwd,
                                env=environment, capture_output=True, text=True, check=True)
        assert Path(result.stdout.strip()).resolve() == fake.resolve(), (name, result.stdout)
        checks.append({"case": name, "passed": True})

production_after = {path.relative_to(BUNDLE).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in BUNDLE.rglob("*") if path.is_file()}
assert production_before == production_after, "Tests must not alter the real bundle allocation."
for line in (BUNDLE / "code_manifest.sha256").read_text(encoding="utf8").splitlines():
    digest, relative = line.split("  ", 1)
    assert hashlib.sha256((BUNDLE / relative).read_bytes()).hexdigest() == digest
actual = json.loads((BUNDLE / "qos_allocation.json").read_text(encoding="utf8"))
assert actual["ready"] is True
assert actual["normal_running"] == 1 and actual["long_running"] == 0
assert actual["queue_snapshot"][0]["job_id"] == "3204848"
assert actual["new_jobs"] == {"normal": 15, "long": 15}

report = {"all_checks_passed": True, "checks": checks, "production_allocation_ready": True,
          "fixed_user_allocation": "first five J2 normal, last five long; capacity refresh cannot change allocation",
          "numerical_four_file_hashes_unchanged": True, "actual_cluster_queries": 0, "actual_submissions": 0,
          "disposable_test_copies_removed": True}
(HERE / "checks.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
print(json.dumps(report, indent=2))
