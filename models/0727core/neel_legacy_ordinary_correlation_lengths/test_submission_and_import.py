"""Queue migration and downloaded split-result integration tests (no Slurm needed)."""

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from bundle_utils import ORDINARY_DIRECTIONS, partial_result_name, result_name

HERE = Path(__file__).resolve().parent
ANSATZ = "neel_symmetrized"
HASH = "a" * 64
NAME = "clo7-neels-0p23-D8"
BASH = shutil.which("bash") or next(
    (str(p) for p in (Path("D:/Programs/Git/bin/bash.exe"),
                      Path("C:/Program Files/Git/bin/bash.exe")) if p.is_file()), None
)


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


@unittest.skipUnless(BASH, "Bash is required for submission tests")
class SubmissionTests(unittest.TestCase):
    def run_queue(self, queue="", fail_at=0, args=(), complete="", bad_id=False):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for filename in ("submit_all.sh", "submit_correlation_lengths.sh"):
                write(root / filename, (HERE / filename).read_text(encoding="utf-8"))
            for filename in ("correlation_length_job.run", "run_one_correlation_length.py", "checkpoints/test.pt"):
                write(root / filename, "")
            write(root / "checkpoint_manifest.tsv",
                  f"J2_0p23\t{ANSATZ}\t0.23\t8\ttest.pt\toriginal.pt\t{HASH}\n")
            mocks = {
                "squeue": 'printf "%s\\n" "$QUEUE"',
                "python": '[[ -n "$COMPLETE" && " $* " == *" --direction $COMPLETE "* ]]',
                "sbatch": 'echo "submit $*" >> actions\n'
                          'n=$(wc -l < actions)\n'
                          '[[ "$n" != "$FAIL_AT" ]] || exit 1\n'
                          'if [[ "$BAD_ID" == 1 ]]; then echo invalid; else echo "$((100+n))"; fi',
                "scancel": 'echo "cancel $*" >> actions',
            }
            for name, body in mocks.items():
                target = root / "bin" / name
                write(target, "#!/bin/bash\n" + body + "\n")
                target.chmod(0o755)
            process = subprocess.run(
                [BASH, "-c", 'export PATH="$PWD/bin:$PATH"; bash submit_all.sh "$@"', "test", *args],
                cwd=root, env={**os.environ, "QUEUE": queue, "FAIL_AT": str(fail_at),
                               "COMPLETE": complete, "BAD_ID": str(int(bad_id))},
                capture_output=True, text=True,
            )
            actions = (root / "actions").read_text() if (root / "actions").exists() else ""
            return process, actions.splitlines()

    def test_submit_before_cancel_all_duplicate_sequential_jobs(self):
        queue = "\n".join(f"{i}|{NAME}-{HASH[:12]}|RUNNING" for i in (42, 43))
        process, actions = self.run_queue(queue)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(len(actions), 5)
        for action, direction in zip(actions, ORDINARY_DIRECTIONS):
            self.assertTrue(action.startswith("submit "))
            self.assertTrue(action.endswith(direction))
        self.assertEqual(actions[3:], ["cancel 42", "cancel 43"])

    def test_failed_submission_or_invalid_id_preserves_sequential(self):
        for options in ({"fail_at": 2}, {"bad_id": True}):
            with self.subTest(options=options):
                process, actions = self.run_queue(f"42|{NAME}-{HASH[:12]}|RUNNING", **options)
                self.assertNotEqual(process.returncode, 0)
                self.assertFalse(any(a.startswith("cancel") for a in actions))

    def test_existing_and_completed_directions_are_reused(self):
        queue = f"42|{NAME}-{HASH[:12]}|RUNNING\n44|{NAME}-e13-{HASH[:12]}|PENDING"
        process, actions = self.run_queue(queue, complete="env2")
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(len(actions), 2)
        self.assertTrue(actions[0].endswith("env3_ab_env1_ba"))
        self.assertEqual(actions[1], "cancel 42")

    def test_dry_run_has_no_queue_side_effects(self):
        process, actions = self.run_queue(f"42|{NAME}-{HASH[:12]}|RUNNING", args=("--dry-run",))
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(actions, [])
        self.assertEqual(process.stdout.count("WOULD SUBMIT"), 3)
        self.assertIn("WOULD CANCEL sequential job 42", process.stdout)

    def test_unrelated_hash_is_not_cancelled(self):
        process, actions = self.run_queue(f"42|{NAME}-bbbbbbbbbbbb|RUNNING")
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(len(actions), 3)


class ImportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.bundle = root / "incoming" / "nested_bundle"
        self.legacy = root / "D345678910"
        write(self.legacy / "run/tensor.pt", "source tensor")
        item = dict(selected_for_rerun=True, ansatz_directory=ANSATZ,
                    j2_directory="J2_0p23", D=8, j2=0.23, sha256=HASH,
                    original_relative_path="D345678910/run/tensor.pt",
                    source_checkpoint_sha256=hashlib.sha256(b"source tensor").hexdigest(),
                    legacy_run_relative_path="run", legacy_correlation_filename="correlation_length_D_8.json")
        write(self.bundle / "checkpoint_manifest.json", json.dumps(dict(
            bundle_kind="D345678910_neel_legacy_ordinary_only", items=[item])))
        self.results = self.bundle / "results_three_env_ordinary_v5"
        self.destination = self.legacy / "run/correlation_length_D_8.json"
        for direction in ORDINARY_DIRECTIONS:
            payload = dict(schema="c3ctm_single_ordinary_correlation_length_direction",
                           schema_version=1, transfer_network_schema="three_geometric_straight_rows_ordinary_v6",
                           ansatz_directory=ANSATZ, direction=direction, D_bond=8, chi=128,
                           calculation_hyperparameters={"J2": 0.23},
                           spectra={direction: dict(eigenvalues=[dict(real=2, imag=0), dict(real=1, imag=0)],
                                                    inverse_correlation_length=math.log(2))},
                           cluster_bundle_provenance={"checkpoint_sha256": HASH},
                           ctm=dict(steps_ab=10, steps_ba=10), seed=123,
                           completed_at_utc="2026-09-13T00:00:00Z")
            write(self.partial(direction), json.dumps(payload))

    def partial(self, direction):
        return self.results / partial_result_name(ANSATZ, "J2_0p23", 8, direction)

    def run_import(self, *args):
        return subprocess.run([sys.executable, str(HERE / "import_completed_results.py"),
                               "--incoming", str(self.bundle.parent), "--legacy-root", str(self.legacy), *args],
                              capture_output=True, text=True)

    def test_partial_only_import_and_repeat(self):
        dry = self.run_import("--dry-run")
        self.assertEqual(dry.returncode, 0, dry.stderr)
        self.assertIn("WOULD ASSEMBLE AND IMPORT", dry.stdout)
        self.assertFalse(self.destination.exists())
        process = self.run_import()
        self.assertEqual(process.returncode, 0, process.stderr)
        payload = json.loads(self.destination.read_text())
        self.assertEqual(set(payload["spectra"]), set(ORDINARY_DIRECTIONS))
        self.assertEqual(payload["cluster_bundle_provenance"]["checkpoint_sha256"], HASH)
        self.assertTrue(all(self.partial(d).exists() for d in ORDINARY_DIRECTIONS))
        self.assertIn("ALREADY IMPORTED", self.run_import().stdout)

    def test_missing_or_stale_partial_is_incomplete_even_in_dry_run(self):
        path = self.partial("env2")
        payload = json.loads(path.read_text())
        payload["cluster_bundle_provenance"]["checkpoint_sha256"] = "b" * 64
        write(path, json.dumps(payload))
        for args in ((), ("--dry-run",)):
            result = self.run_import(*args)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn("INCOMPLETE", result.stdout)
            self.assertNotIn("WOULD ASSEMBLE", result.stdout)
        path.unlink()
        self.assertEqual(self.run_import("--dry-run").returncode, 1)
        self.assertFalse(self.destination.exists())

    def test_invalid_combined_file_dry_run_uses_valid_partials(self):
        write(self.results / result_name(ANSATZ, "J2_0p23", 8), "{}")
        result = self.run_import("--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("WOULD ASSEMBLE AND IMPORT", result.stdout)


if __name__ == "__main__":
    unittest.main()
