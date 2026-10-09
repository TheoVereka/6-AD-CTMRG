"""Run the established production checks without replacing historical evidence."""
from pathlib import Path

root = Path(__file__).resolve().parents[2]
source = root / "tmp_tee_schematic_20261009/algorithm_checks/check_production_spectral.py"
script = source.read_text(encoding="utf-8")
destination = Path(__file__).with_name("solver_production_regression.json")
script = script.replace('output = Path(__file__).with_name("production_spectral_results.json")',
                        f"output = Path({str(destination)!r})")
exec(compile(script, str(source), "exec"),
     {"__file__": str(source), "__name__": "__main__"})
