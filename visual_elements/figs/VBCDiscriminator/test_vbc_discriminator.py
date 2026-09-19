from __future__ import annotations

import sys
import tempfile
import unittest
import csv
import math
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_branch_runs import StageRow, compare_zero_field
from analyze_existing_twoc3 import GROUP_KEYS, parse_observation
from analyze_three_source_runs import Stage, discover as discover_three_source
from plot_pinning_replica1 import rank_stage
from plot_pinning_supervisor import omega_order, write_source_csvs
from fit_pinned_correlations import fit_one_rank


def write_observation(path: Path, groups: tuple[float, float, float], energy: float) -> None:
    lines = ["# D=8  chi=160", "", f"energy_per_site = {energy:+.12e}"]
    for keys, value in zip(GROUP_KEYS, groups):
        lines.extend(f"corr_{key} = {value:+.12e}" for key in keys)
    path.write_text("\n".join(lines), encoding="utf-8")


def stage(branch: str, replica: int, texture: str, middle: float,
          clock: float, energy: float) -> StageRow:
    return StageRow(
        path="synthetic", branch=branch, orientation=0, replica=replica,
        field=0.0, J2=0.30, D=8, chi=160, energy_per_site=energy,
        chi_energy_shift=2.0e-7, G0=-0.3, G1=-0.2, G2=-0.2,
        delta=0.1, middle_fraction=middle, clock_z6=clock,
        texture=texture,
    )


class VBCDiscriminatorTest(unittest.TestCase):
    def test_strict_discovery_refuses_to_skip_malformed_observation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "D_8_chi_160_energy_magnetization_correlation.txt"
            path.write_text("incomplete download", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "refusing to use an incomplete dataset"):
                discover_three_source(Path(tmp), strict=True)

    def test_ranked_correlations_keep_geometry(self) -> None:
        row = Stage(
            path="synthetic", branch="rank-split", replica=1,
            field=0.04, J2=0.30, D=9, chi=180,
            energy_per_site=-0.42, chi_energy_shift=1.0e-7,
            G0=-0.20, G1=-0.40, G2=-0.30,
            delta=0.20, middle_fraction=0.50, clock_z6=0.0,
            texture="three-distinct/mixed",
        )
        ranked = rank_stage(row, "Kuma")
        self.assertEqual((ranked.rank1, ranked.rank2, ranked.rank3),
                         (-0.40, -0.30, -0.20))
        self.assertEqual((ranked.rank1_group, ranked.rank2_group,
                          ranked.rank3_group), (1, 2, 0))

    def test_ideal_clock_signs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "J2_0p3"
            root.mkdir()
            p_path = root / "p_energy_magnetization_correlation.txt"
            d_path = root / "d_energy_magnetization_correlation.txt"
            write_observation(p_path, (-0.35, -0.35, -0.20), -0.42)
            write_observation(d_path, (-0.35, -0.20, -0.20), -0.42)
            p = parse_observation(p_path)
            d = parse_observation(d_path)
        self.assertAlmostEqual(p.middle_fraction, 0.0)
        self.assertAlmostEqual(p.clock_z6, 1.0)
        self.assertEqual(p.texture, "plaquette")
        self.assertAlmostEqual(d.middle_fraction, 1.0)
        self.assertAlmostEqual(d.clock_z6, -1.0)
        self.assertEqual(d.texture, "dimer-plaquette")

    def test_signed_omega_order_ideal_textures(self) -> None:
        row = Stage(
            path="synthetic", branch="plaquette", replica=1,
            field=0.0, J2=0.30, D=8, chi=160,
            energy_per_site=-0.42, chi_energy_shift=0.0,
            G0=-0.3, G1=-0.3, G2=-0.2,
            delta=0.1, middle_fraction=0.0, clock_z6=1.0,
            texture="plaquette",
        )
        ranked = rank_stage(row, "Kuma")
        self.assertAlmostEqual(omega_order(ranked), -1.0)
        self.assertAlmostEqual(omega_order(replace(ranked, rank2=-0.2)), 1.0)
        self.assertAlmostEqual(omega_order(replace(ranked, rank2=-0.25)), 0.0)
        self.assertTrue(math.isnan(omega_order(replace(
            ranked, rank1=-0.2, rank2=-0.2, rank3=-0.2))))

    def test_supervisor_csv_schema_and_sorted_correlations(self) -> None:
        row = Stage(
            path="synthetic", branch="rank-split", replica=1,
            field=0.04, J2=0.30, D=9, chi=180,
            energy_per_site=-0.42, chi_energy_shift=0.0,
            G0=-0.20, G1=-0.40, G2=-0.30,
            delta=0.20, middle_fraction=0.50, clock_z6=0.0,
            texture="three-distinct/mixed",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(write_source_csvs([rank_stage(row, "Kuma")], root), 2)
            with (root / "rank-split" / "energy.csv").open(
                    encoding="utf-8", newline="") as stream:
                energy = list(csv.DictReader(stream))
            with (root / "rank-split" / "nn_correlations.csv").open(
                    encoding="utf-8", newline="") as stream:
                correlations = list(csv.DictReader(stream))
        self.assertEqual(list(energy[0]), ["D", "h", "E"])
        self.assertEqual(
            list(correlations[0]),
            ["D", "h", "NNcorrStrongest", "NNcorrMiddle", "NNcorrWeakest"],
        )
        self.assertEqual(
            tuple(float(correlations[0][key]) for key in
                  ("NNcorrStrongest", "NNcorrMiddle", "NNcorrWeakest")),
            (-0.40, -0.30, -0.20),
        )

    def test_quadratic_fit_excludes_measured_zero_field(self) -> None:
        source = Stage(
            path="synthetic", branch="plaquette", replica=1,
            field=0.08, J2=0.30, D=8, chi=160,
            energy_per_site=-0.42, chi_energy_shift=0.0,
            G0=-0.40, G1=-0.30, G2=-0.20,
            delta=0.20, middle_fraction=0.50, clock_z6=0.0,
            texture="three-distinct/mixed",
        )
        base = rank_stage(source, "Kuma")
        rows = [replace(base, h=h, rank1=-0.4 + 2.0 * h - 3.0 * h * h)
                for h in (0.08, 0.04, 0.02, 0.01)]
        rows.append(replace(base, h=0.0, rank1=-9.0))
        fit = fit_one_rank(rows, "strongest", "rank1")
        self.assertAlmostEqual(fit.C0, -0.4)
        self.assertAlmostEqual(fit.c1, 2.0)
        self.assertAlmostEqual(fit.c2, -3.0)
        self.assertAlmostEqual(fit.linear_response_reference_h, 0.02)
        self.assertAlmostEqual(fit.quadratic_over_linear_at_reference_h, 0.03)
        self.assertTrue(fit.quadratic_correction_smaller_at_reference_h)
        self.assertEqual(fit.n_positive_fields, 4)
        self.assertAlmostEqual(fit.observed_h0, -9.0)

    def test_resolved_energy_competition(self) -> None:
        rows = [
            stage("plaquette", 1, "plaquette", 0.05, 0.92, -0.4200100),
            stage("plaquette", 2, "plaquette", 0.04, 0.94, -0.4200102),
            stage("dimer-plaquette", 1, "dimer-plaquette", 0.95, -0.92, -0.4200000),
            stage("dimer-plaquette", 2, "dimer-plaquette", 0.96, -0.94, -0.4199998),
        ]
        result = compare_zero_field(rows, 1.0e-6)[0]
        self.assertGreater(result.gap_Ed_minus_Ep, 0.0)
        self.assertEqual(result.decision, "PLAQUETTE lower")


if __name__ == "__main__":
    unittest.main()
