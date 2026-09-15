from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_branch_runs import StageRow, compare_zero_field
from analyze_existing_twoc3 import GROUP_KEYS, parse_observation


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

