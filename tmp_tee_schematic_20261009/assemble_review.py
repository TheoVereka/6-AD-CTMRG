"""Export the depth-rendered schematics through Matplotlib, without labels."""
from pathlib import Path
import argparse
import html
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties

ROOT = Path(__file__).resolve().parent
ORDER = ("01", "02", "02_bis", "03", "03_bis", "04", "05", "05_bis", "06", "06_bis", "07", "07_bis")
CAPTIONS = {
    "01": "1 · 单层蜂窝", "02": "2 · 两层与等分切口", "03_bis": "3_bis · 物理腿画出后的延伸示意",
    "03": "3 · 物理腿相连／开放", "04": "4 · 复制与移位（右下层下移6）",
    "05": "5 · 红绿飞线", "06": "6 · 单份 CTM 边界", "07": "7 · 两份 CTM 边界与视觉腰斩",
    "02_bis": "2_bis · 双层网格延伸（不含灰腿）",
    "05_bis": "5_bis · 五个网络区域各三组省略号",
    "06_bis": "6_bis · CTM边界及左侧25%省略号",
    "07_bis": "7_bis · 五条柱链各两组省略号",
}


def assemble_panels(first_panel):
    """Two equal-width rows with equal image heights within each row.

    Column divisions are independent, so the four panels are not forced
    into a rectangular two-by-two grid. Labels stay vector text in PDF.
    """
    names = (first_panel, "06_bis", "05_bis", "07_bis")
    images = [mpimg.imread(ROOT / "figures" / f"step_{name}.png") for name in names]
    ratios = [data.shape[1] / data.shape[0] for data in images]
    # At this display width a 10.5pt label remains comparable to body type
    # in a journal column. Its size is physical, independent of PNG pixels.
    width_in, dpi = 3.4, 600
    gap_in = .014
    row_heights = [(width_in - gap_in) / (ratios[i] + ratios[i+1]) for i in (0, 2)]
    height_in = sum(row_heights) + gap_in
    font_path = Path("C:/Windows/Fonts/times.ttf")
    if not font_path.exists():
        raise FileNotFoundError("Times New Roman regular font is required: " + str(font_path))
    font = FontProperties(fname=str(font_path), size=10.5)
    matplotlib.rcParams["pdf.fonttype"] = 42
    figure = plt.figure(figsize=(width_in, height_in), dpi=dpi, facecolor="white")
    panels = []
    for row in range(2):
        h = row_heights[row]
        bottom = height_in - h if row == 0 else 0
        left = 0.
        for column in range(2):
            i = row * 2 + column
            w = ratios[i] * h
            box = [left / width_in, bottom / height_in, w / width_in, h / height_in]
            ax = figure.add_axes(box)
            ax.imshow(images[i], interpolation="none", aspect="equal")
            ax.set_axis_off()
            ax.text(.012, .985, f"({chr(97+i)})", transform=ax.transAxes,
                    fontproperties=font, ha="left", va="top", color="black")
            panels.append(dict(label=f"({chr(97+i)})", source=f"step_{names[i]}.png",
                               axes_box=box))
            left += w + gap_in
    destination = ROOT / "figures" / "four_panel_bis"
    for suffix in (".png", ".pdf"):
        figure.savefig(destination.with_suffix(suffix), dpi=dpi, pad_inches=0,
                       facecolor="white", metadata={"Title": ""})
    plt.close(figure)
    metadata = dict(width_in=width_in, height_in=height_in, dpi=dpi,
                    label_font="Times New Roman", label_font_file=str(font_path),
                    label_points=10.5, gap_in=gap_in, row_heights_in=row_heights,
                    panels=panels)
    (ROOT / "four_panel_layout.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-panel", choices=("02_bis", "03_bis"), default="03_bis")
    args = parser.parse_args()
    # The requested true-square front faces need oblique projection. Export
    # the already depth-rendered pixels at their original aspect in Matplotlib.
    view = json.loads((ROOT / "coordinates" / "camera_01.json").read_text(encoding="utf-8"))
    view = {key: view[key] for key in ("projection", "depth_scale", "depth_angle", "screen_projection_matrix")}
    (ROOT / "matplotlib_view.json").write_text(json.dumps(view, indent=2), encoding="utf-8")

    cards = []
    with PdfPages(ROOT / "all_steps.pdf", metadata={"Title": "", "Author": "", "Subject": ""}) as pages:
        for name in ORDER:
            path = ROOT / "figures" / f"step_{name}.png"
            if not path.exists():
                continue
            image = mpimg.imread(path)
            height, width = image.shape[:2]
            figure = plt.figure(figsize=(width / 160, height / 160), dpi=160, facecolor="white")
            ax = figure.add_axes([0, 0, 1, 1])
            ax.imshow(image, interpolation="none")
            ax.set_axis_off()
            pdf = path.with_suffix(".pdf")
            figure.savefig(pdf, dpi=160, pad_inches=0, facecolor="white")
            pages.savefig(figure, dpi=160, pad_inches=0, facecolor="white")
            plt.close(figure)
            cards.append(f'<section id="step-{name}"><h2>{html.escape(CAPTIONS[name])}</h2>'
                         f'<a href="figures/step_{name}.png"><img src="figures/step_{name}.png" '
                         f'alt="{name}"></a><p><a href="figures/step_{name}.pdf">PDF</a></p></section>')
    document = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>蜂窝边界网络逐步绘图</title>
<style>body{margin:2rem auto;max-width:1450px;padding:0 1rem;font:17px/1.65 system-ui,sans-serif;color:#222}
img{display:block;width:100%;height:auto}section{margin:3rem 0}h2{font-size:1.2rem;font-weight:500}
a{color:#1358a0}nav{display:flex;gap:1rem;flex-wrap:wrap}</style>
<h1>逐步绘图修订版</h1><p>蜂窝边长1，球直径0.10，ket/bra球心距0.19；开放灰腿上、下、留白各0.03。方形边长0.3，中心为双小球中点。所有圆柱与方形按真实深度相互遮挡，整体在25%透明底图之上；方形仅对相连细腿使用符号遮罩。xz面保持正方形；xy面短边为长边一半，内角60°/120°。蓝色(110,150,255)，纯点线；灰腿(160,160,160)。复制位移为4/2/6，向右位移2。y省略号投影点距与x一致。2_bis与3_bis都保留。PNG与PDF仅保留极小边距；单图无文字、坐标轴或背景网格，合图仅有(a–d)。</p>
<p>数值状态：三pair与八步收缩核对通过；默认求谱器在基扩展时接纳近零块，导致正交性崩坏。原因已定位，求谱器尚未修复，当前原型不可生产使用。</p>
<nav><a href="逐步图解.md">图中对象说明</a><a href="S2_routine明确结论.md">S₂ routine与精度的明确结论</a><a href="Krylov错误说明.md">本次求谱错误逐步解释</a><a href="核心脚本说明.md">核心原型及停止原因</a><a href="env_pair_mapping.md">三个env pair及LMN/XYZ核对</a><a href="all_steps.pdf">全部单图 PDF</a><a href="figures/four_panel_bis.pdf">四panel合图 PDF</a>'''
    document += "".join(f'<a href="#step-{name}">{name}</a>' for name in ORDER)
    layout = assemble_panels(args.first_panel)
    document += '</nav><section><h2>四panel合图</h2><a href="figures/four_panel_bis.png"><img src="figures/four_panel_bis.png" alt="four panels"></a><p><a href="figures/four_panel_bis.pdf">PDF</a></p></section>' + "".join(cards) + "</html>"
    (ROOT / "review.html").write_text(document, encoding="utf-8")
    print(f"Exported {len(cards)} figures and the combined PDF; Matplotlib view {view}.")
    print(f"Four panels: first={args.first_panel}, width={layout['width_in']}in, labels={layout['label_points']}pt Times New Roman.")


if __name__ == "__main__":
    main()
