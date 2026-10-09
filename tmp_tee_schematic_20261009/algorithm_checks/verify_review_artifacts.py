"""Check the exported review files without rerunning tensor numerics."""
from pathlib import Path
import json
import re
import sys
from urllib.parse import unquote

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from assemble_review import ORDER


def pdf_counts(path):
    data = path.read_bytes()
    return (len(re.findall(rb"/Type\s*/Page\b", data)),
            len(re.findall(rb"/Type\s*/Font\b", data)))


def main():
    dimensions = {}
    for name in ORDER:
        png = ROOT / "figures" / f"step_{name}.png"
        pdf = png.with_suffix(".pdf")
        with Image.open(png) as image:
            dimensions[name] = list(image.size)
        assert pdf_counts(pdf) == (1, 0), pdf
    pages, fonts = pdf_counts(ROOT / "all_steps.pdf")
    assert (pages, fonts) == (len(ORDER), 0)
    panel_pdf = ROOT / "figures/four_panel_bis.pdf"
    assert pdf_counts(panel_pdf)[0] == 1
    assert b"TimesNewRomanPSMT" in panel_pdf.read_bytes()
    layout = json.loads((ROOT / "four_panel_layout.json").read_text(encoding="utf-8"))
    assert layout["label_font"] == "Times New Roman" and layout["label_points"] == 10.5
    for left, right in ((0, 1), (2, 3)):
        a, b = [layout["panels"][i]["axes_box"] for i in (left, right)]
        assert abs(a[1] - b[1]) < 1e-12 and abs(a[3] - b[3]) < 1e-12
        assert abs(a[0]) < 1e-12 and abs(b[0]+b[2]-1) < 1e-12

    documents = [ROOT / name for name in (
        "逐步图解.md", "Krylov错误说明.md", "核心脚本说明.md", "S2_routine明确结论.md",
        "env_pair_mapping.md", "新算法_S2核查.md",
        "algorithm_checks/krylov_diagnosis_20261009/诊断结论.md",
        "algorithm_checks/krylov_diagnosis_20261009/evidence_scope.md",
    )]
    missing = []
    for path in documents:
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            target = target.strip().strip("<>")
            if "://" in target or target.startswith("#"):
                continue
            target = re.sub(r":\d+$", "", unquote(target.split("#")[0]))
            if not (path.parent / target).exists():
                missing.append({"document": str(path.relative_to(ROOT)), "target": target})
    assert not missing, missing
    geometry = json.loads((ROOT / "algorithm_checks/projection_verification_v6.json").read_text(encoding="utf-8"))
    assert geometry["all_checks_passed"]
    result = dict(figure_count=len(ORDER), ordered_steps=ORDER,
                  png_dimensions=dimensions, combined_pdf_pages=pages,
                  combined_pdf_font_objects=fonts, missing_markdown_links=missing,
                  projection="xz square; xy 60/120 parallelogram, depth length .5",
                  geometry_check_count=geometry["check_count"],
                  diagram_revision="v6", figure_white_margin_pixels="1-2",
                  composite_font="Times New Roman", composite_label_points=10.5,
                  composite_width_in=layout["width_in"],
                  composite_first_panel=layout["panels"][0]["source"])
    (ROOT / "artifact_checks.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
