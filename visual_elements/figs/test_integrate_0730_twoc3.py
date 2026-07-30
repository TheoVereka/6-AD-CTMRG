from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import integrate_0730_twoc3 as integration


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


def _make_summary(summary: Path, j2: str, D: int, energy: float) -> None:
    target = summary / f"J2_{j2}" / integration.ANSATZ / f"D_{D}"
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
    # Non-2C3 data is ignored.
    _make_run(
        new,
        "source_other",
        "other",
        0.34,
        7,
        energies_by_chi={70: -0.39},
        checkpoint_chis=[70],
        ansatz="1tensor_C6Ypi",
    )

    preparation = integration.prepare(new, summary, rerun, dry_run=False)

    assert set(preparation.unique) == {("0p31", 4)}
    assert preparation.unique[("0p31", 4)].candidate.chi == 40
    imported = summary / "J2_0p31" / integration.ANSATZ / "D_4"
    assert (imported / "energy_magnetization_correlation.txt").is_file()
    assert (imported / "tensor_best.pt").is_file()
    assert set(preparation.conflicts) == {("0p3", 3), ("0p32", 5)}
    assert len(preparation.conflicts[("0p3", 3)]) == 2
    assert len(preparation.conflicts[("0p32", 5)]) == 2
    assert len(preparation.archived) == 2
    assert len(list(rerun.glob("*.pt"))) == 2
    assert len(list(rerun.glob("*.log"))) == 2

    # The generated rerun folder is excluded on subsequent scans.
    second_scan = integration.discover_new_runs(new)
    assert sum(len(items) for items in second_scan.incomplete.values()) == 2


def test_unique_labels_disambiguate_identical_display_text(tmp_path: Path) -> None:
    candidate = integration.Candidate(
        ansatz=integration.ANSATZ,
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
    test_single_D_conflict_plot_helpers()
    print("All integrate_0730_twoc3 tests passed.")
