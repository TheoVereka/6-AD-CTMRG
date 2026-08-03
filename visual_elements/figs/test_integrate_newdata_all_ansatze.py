from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import integrate_newdata_all_ansatze as integration

TWOC3 = "2tensor_twoC3"


def _observable(energy: float) -> str:
    return f"energy_per_site = {energy:.12f}\n"


def _make_run(
    root: Path,
    source: str,
    name: str,
    j2: float,
    D: int,
    *,
    energies_by_chi: dict[int, float],
    checkpoint_chis: list[int],
    ansatz: str = "2tensor_twoC3",
) -> Path:
    run = root / source / name
    run.mkdir(parents=True)
    params = {
        "ansatz": ansatz,
        "J2": j2,
        "run_timestamp": "20260730_120000",
    }
    (run / "hyperparams.yaml").write_text(json.dumps(params), encoding="utf-8")
    (run / "run.log").write_text(
        "Started : 2026-07-30 12:00:00\n"
        f"D={D} complete in 12.50 h\n"
        "Total wall time: 12.50 h (45000 s)\n",
        encoding="utf-8",
    )
    for chi, energy in energies_by_chi.items():
        (run / f"D_{D}_chi_{chi}_energy_magnetization_correlation.txt").write_text(
            _observable(energy),
            encoding="utf-8",
        )
    for chi in checkpoint_chis:
        (run / f"sweep_D{D}_chi{chi}_best.pt").write_bytes(b"checkpoint")
    return run


def _make_summary(
    summary: Path,
    j2: str,
    D: int,
    energy: float,
    ansatz: str = TWOC3,
) -> None:
    target = summary / f"J2_{j2}" / ansatz / f"D_{D}"
    target.mkdir(parents=True)
    (target / "energy_magnetization_correlation.txt").write_text(
        _observable(energy),
        encoding="utf-8",
    )
    (target / "manifest.json").write_text(
        json.dumps(
            {
                "J2_label": j2,
                "D": D,
                "chi": 30,
                "energy_per_site": energy,
                "source_job": "old_source/run",
            }
        ),
        encoding="utf-8",
    )


def test_prepare_classifies_imports_conflicts_and_unfinished(tmp_path: Path) -> None:
    new = tmp_path / "0730newdata"
    summary = tmp_path / "0713summary"
    rerun = new / integration.DEFAULT_RERUN_FOLDER
    new.mkdir()
    summary.mkdir()

    # One run, two chi files: it must remain one candidate and select chi=40.
    _make_run(
        new,
        "source_unique",
        "unique",
        0.31,
        4,
        energies_by_chi={30: -0.41, 40: -0.42},
        checkpoint_chis=[30, 40],
    )
    # Two different output directories for the same pair are a real conflict.
    _make_run(
        new,
        "source_B",
        "duplicate_1",
        0.32,
        5,
        energies_by_chi={50: -0.40},
        checkpoint_chis=[50],
    )
    _make_run(
        new,
        "source_C",
        "duplicate_2",
        0.32,
        5,
        energies_by_chi={50: -0.405},
        checkpoint_chis=[50],
    )
    # Duplicate unfinished attempts are both archived when summary lacks the pair.
    _make_run(
        new,
        "source_D",
        "unfinished_1",
        0.33,
        6,
        energies_by_chi={},
        checkpoint_chis=[60],
    )
    _make_run(
        new,
        "source_E",
        "unfinished_2",
        0.33,
        6,
        energies_by_chi={},
        checkpoint_chis=[60],
    )
    # Existing summary plus one new result is also a conflict.
    _make_summary(summary, "0p3", 3, -0.43)
    _make_run(
        new,
        "source_overlap",
        "overlap",
        0.30,
        3,
        energies_by_chi={30: -0.431},
        checkpoint_chis=[30],
    )
    # Another ansatz is independently unique even when J2 and D are identical.
    _make_run(
        new,
        "source_other",
        "other",
        0.31,
        4,
        energies_by_chi={40: -0.39},
        checkpoint_chis=[40],
        ansatz="1tensor_C6Ypi",
    )

    preparation = integration.prepare(new, summary, rerun, dry_run=False)

    assert set(preparation.unique) == {
        (TWOC3, "0p31", 4),
        ("1tensor_C6Ypi", "0p31", 4),
    }
    assert preparation.unique[(TWOC3, "0p31", 4)].candidate.chi == 40
    imported = summary / "J2_0p31" / TWOC3 / "D_4"
    assert (imported / "energy_magnetization_correlation.txt").is_file()
    assert (imported / "tensor_best.pt").is_file()
    assert set(preparation.conflicts) == {
        (TWOC3, "0p3", 3),
        (TWOC3, "0p32", 5),
    }
    assert len(preparation.conflicts[(TWOC3, "0p3", 3)]) == 2
    assert len(preparation.conflicts[(TWOC3, "0p32", 5)]) == 2
    assert len(preparation.archived) == 2
    assert len(list(rerun.glob("*.pt"))) == 2
    assert len(list(rerun.glob("*.log"))) == 2

    # The generated rerun folder is excluded on subsequent scans.
    second_scan = integration.discover_new_runs(new)
    assert sum(len(items) for items in second_scan.incomplete.values()) == 2


def test_unique_labels_disambiguate_identical_display_text(tmp_path: Path) -> None:
    candidate = integration.Candidate(
        ansatz=TWOC3,
        j2="0p3",
        timestamp="20260730_120000",
        D=4,
        chi=40,
        energy=-0.4,
        job_dir=tmp_path,
        observation=tmp_path / "obs.txt",
        source_job="source/a",
        lookahead_chi=None,
        lookahead_energy=None,
    )
    choices = [
        integration.Choice(candidate, "new", "same", "source/a", 1.0),
        integration.Choice(candidate, "new", "same", "source/b", 1.0),
    ]
    labels = integration._unique_display_labels(choices)
    assert len(labels) == len(set(labels)) == 2


def test_every_0801_yaml_ansatz_is_kept_distinct(tmp_path: Path) -> None:
    new = tmp_path / "newdata"
    new.mkdir()
    for index, ansatz in enumerate(integration.KNOWN_YAML_ANSATZE):
        _make_run(
            new,
            f"source_{index}",
            f"run_{index}",
            0.20 + index * 0.001,
            3,
            energies_by_chi={30: -0.5 + index * 0.001},
            checkpoint_chis=[30],
            ansatz=ansatz,
        )
    _make_run(
        new,
        "future_source",
        "future_run",
        0.399,
        4,
        energies_by_chi={40: -0.399},
        checkpoint_chis=[40],
        ansatz="future_registry_ansatz",
    )
    scan = integration.discover_new_runs(new)
    discovered = {ansatz for ansatz, _j2, _D in scan.completed}
    assert discovered == set(integration.KNOWN_YAML_ANSATZE) | {
        "future_registry_ansatz"
    }
    assert len(scan.completed) == len(integration.KNOWN_YAML_ANSATZE) + 1


def test_group_confirm_stages_every_D_before_replacing_summary(tmp_path: Path) -> None:
    new = tmp_path / "newdata"
    summary = tmp_path / "summary"
    new.mkdir()
    summary.mkdir()
    for D, energy in ((4, -0.44), (5, -0.45)):
        _make_summary(summary, "0p3", D, -0.40 - D * 0.001)
        _make_run(
            new,
            f"source_D{D}",
            f"run_D{D}",
            0.30,
            D,
            energies_by_chi={40: energy},
            checkpoint_chis=[40],
        )
    scan = integration.discover_new_runs(new)
    choices = [
        scan.completed[(TWOC3, "0p3", D)][0]
        for D in (4, 5)
    ]

    original_copy = integration.copy_candidate
    calls = {"count": 0}

    def fail_second(candidate, target):
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("synthetic staging failure")
        return original_copy(candidate, target)

    integration.copy_candidate = fail_second
    try:
        try:
            integration.import_choices_atomically(choices, summary)
        except RuntimeError as exc:
            assert "synthetic staging failure" in str(exc)
        else:
            raise AssertionError("atomic import should have failed")
    finally:
        integration.copy_candidate = original_copy

    # Neither old D was touched because all selected data stages before commit.
    for D in (4, 5):
        energy = integration.read_energy(
            summary
            / "J2_0p3"
            / TWOC3
            / f"D_{D}"
            / "energy_magnetization_correlation.txt"
        )
        assert energy == -0.40 - D * 0.001

    integration.import_choices_atomically(choices, summary)
    for D, expected in ((4, -0.44), (5, -0.45)):
        energy = integration.read_energy(
            summary
            / "J2_0p3"
            / TWOC3
            / f"D_{D}"
            / "energy_magnetization_correlation.txt"
        )
        assert energy == expected


def test_conflict_figures_are_partitioned_by_ansatz_and_j2(tmp_path: Path) -> None:
    def choice(ansatz: str, j2: str, D: int) -> integration.Choice:
        candidate = integration.Candidate(
            ansatz=ansatz,
            j2=j2,
            timestamp="20260801_120000",
            D=D,
            chi=40,
            energy=-0.4,
            job_dir=tmp_path,
            observation=tmp_path / "obs.txt",
            source_job=f"{ansatz}/{j2}/D{D}",
            lookahead_chi=None,
            lookahead_energy=None,
        )
        return integration.Choice(candidate, "new", "source", candidate.source_job, 1.0)

    conflicts = {
        (TWOC3, "0p3", 4): [choice(TWOC3, "0p3", 4)],
        (TWOC3, "0p3", 5): [choice(TWOC3, "0p3", 5)],
        ("2tensor_columnar", "0p3", 4): [choice("2tensor_columnar", "0p3", 4)],
        (TWOC3, "0p31", 4): [choice(TWOC3, "0p31", 4)],
    }
    grouped = integration.group_conflicts_by_ansatz_j2(conflicts)
    assert set(grouped) == {
        (TWOC3, "0p3"),
        ("2tensor_columnar", "0p3"),
        (TWOC3, "0p31"),
    }
    assert set(grouped[(TWOC3, "0p3")]) == {4, 5}


def test_olddata_is_an_exact_reviewed_ledger_for_full_scp_copies(tmp_path: Path) -> None:
    old = tmp_path / "0730olddata"
    new = tmp_path / "0730newdata"
    old.mkdir()

    # Byte-identical completed and incomplete runs represent already reviewed data.
    _make_run(
        old,
        "source",
        "identical_completed",
        0.20,
        3,
        energies_by_chi={30: -0.50},
        checkpoint_chis=[30],
    )
    _make_run(
        old,
        "source",
        "identical_incomplete",
        0.21,
        4,
        energies_by_chi={},
        checkpoint_chis=[40],
    )
    _make_run(
        old,
        "source",
        "old_incomplete_now_completed",
        0.22,
        5,
        energies_by_chi={},
        checkpoint_chis=[50],
    )
    _make_run(
        old,
        "source",
        "changed_checkpoint",
        0.23,
        6,
        energies_by_chi={60: -0.47},
        checkpoint_chis=[60],
    )
    _make_run(
        old,
        "source",
        "same_checkpoint_regenerated_observable",
        0.235,
        6,
        energies_by_chi={60: -0.471},
        checkpoint_chis=[60],
    )
    shutil.copytree(old, new)

    # The old checkpoint becomes a newly completed pair when its observable arrives.
    newly_completed = new / "source" / "old_incomplete_now_completed"
    (newly_completed / "D_5_chi_50_energy_magnetization_correlation.txt").write_text(
        _observable(-0.48),
        encoding="utf-8",
    )
    # Same relative name but changed checkpoint content must be reviewed again.
    changed = new / "source" / "changed_checkpoint" / "sweep_D6_chi60_best.pt"
    changed.write_bytes(b"changed checkpoint contents")
    regenerated = (
        new
        / "source"
        / "same_checkpoint_regenerated_observable"
        / "D_6_chi_60_energy_magnetization_correlation.txt"
    )
    regenerated.write_text(_observable(-0.472), encoding="utf-8")
    # A genuinely new run in the re-scp'd source folder is also visible.
    _make_run(
        new,
        "source",
        "brand_new",
        0.24,
        7,
        energies_by_chi={70: -0.46},
        checkpoint_chis=[70],
    )

    scan = integration.discover_new_runs(new, old_root=old)
    assert set(scan.completed) == {
        (TWOC3, "0p22", 5),
        (TWOC3, "0p23", 6),
        (TWOC3, "0p24", 7),
    }
    assert not scan.incomplete
    skipped_kinds = [record["kind"] for record in scan.reviewed_skipped]
    assert skipped_kinds.count("completed") == 2
    assert skipped_kinds.count("incomplete_checkpoint") == 1


def test_single_D_conflict_plot_helpers() -> None:
    import plot_analysis_Windows as analysis
    import matplotlib.pyplot as plt

    corr = {
        key: -0.30 + index * 0.001
        for index, key in enumerate(
            {
                pair
                for group in analysis.NN_GROUPS_RAW
                for pair in group
            }
        )
    }
    mag = {
        (env, site): {
            "Sx": 0.01 if site in "ACE" else -0.01,
            "Sy": 0.0,
            "Sz": 0.10 if site in "ACE" else -0.10,
        }
        for env in (1, 2, 3)
        for site in "ABCDEF"
    }
    processed = analysis.process_parsed_D_data(
        {4: {"energy_per_site": -0.4, "corr": corr, "mag": mag}}
    )
    assert processed is not None
    fig, axes = plt.subplots(3, 1)
    analysis.plot_col_energy(axes[0], processed, show_xlabel=False)
    analysis.plot_col_mag(axes[1], processed, show_xlabel=False)
    analysis.plot_col_nn(axes[2], processed, show_xlabel=True)
    fig.canvas.draw()
    plt.close(fig)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as directory:
        test_prepare_classifies_imports_conflicts_and_unfinished(Path(directory))
    with tempfile.TemporaryDirectory() as directory:
        test_unique_labels_disambiguate_identical_display_text(Path(directory))
    with tempfile.TemporaryDirectory() as directory:
        test_every_0801_yaml_ansatz_is_kept_distinct(Path(directory))
    with tempfile.TemporaryDirectory() as directory:
        test_group_confirm_stages_every_D_before_replacing_summary(Path(directory))
    with tempfile.TemporaryDirectory() as directory:
        test_conflict_figures_are_partitioned_by_ansatz_and_j2(Path(directory))
    with tempfile.TemporaryDirectory() as directory:
        test_olddata_is_an_exact_reviewed_ledger_for_full_scp_copies(Path(directory))
    test_single_D_conflict_plot_helpers()
    print("All all-ansatz integration tests passed.")
