#!/usr/bin/env python3
"""Build an eight-slide meeting deck from this repository's saved evidence.

All diagrams are editable PowerPoint shapes. Result plots are drawn with
Matplotlib from saved experiment data; no generated artwork is used.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
ASSETS = OUT / "assets"
ASSETS.mkdir(exist_ok=True)
DATA = json.loads((ROOT / "doc/assets/figure_data.json").read_text())
ARCHIVE = ROOT / 'results/archive/2026-10-19'
SUMMARY = [row for row in json.loads(((ARCHIVE / 'summary.json') if (ARCHIVE / 'summary.json').exists()
                                     else ROOT / "results/summary.json").read_text())
           if row['model'] in ('granite', 'olmoe', 'deepseek')]
MANIFEST = json.loads(((ARCHIVE / 'experiment_manifest.json') if (ARCHIVE / 'experiment_manifest.json').exists()
                       else ROOT / "results/experiment_manifest.json").read_text())
VALIDATION = json.loads(((ARCHIVE / 'nvmevirt_validation.json') if (ARCHIVE / 'nvmevirt_validation.json').exists()
                         else ROOT / "results/nvmevirt_validation.json").read_text())
# Sum of the actual GGUF tensor element counts in the three metadata files.
PARAMETERS = {"granite": 1334628352, "olmoe": 6919161856, "deepseek": 15706484224}

FONT = "Noto Sans CJK TC"
BLACK, GRAY, LINE = "111111", "585858", "D4DAE0"
BLUE, LIGHT_BLUE, PALE_BLUE = "62AADB", "9CCDEE", "EFF7FC"
RED, PALE_RED = "C83D47", "FBEFF0"
WHITE, MID_GRAY = "FFFFFF", "ACB3BA"


def rgb(value):
    return RGBColor.from_string(value)


prs = Presentation()
prs.slide_width, prs.slide_height = Inches(16), Inches(9)
prs.core_properties.title = "AiF Flash：MoE expert 分組與放置實驗"
prs.core_properties.subject = "架構、cr-read、模型尺寸、分組延遲與實驗結果"
prs.core_properties.author = "expert_flash_experiment"
prs.core_properties.keywords = "AiF, MoE, cr-read, NVMeVirt, INT8"
prs.core_properties.comments = "Based on saved experiments dated 2026-10-19."


def rect(s, x, y, w, h, fill=WHITE, edge=LINE, radius=False, lw=1):
    kind = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
    q = s.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    q.fill.solid()
    q.fill.fore_color.rgb = rgb(fill)
    q._element.spPr.append(OxmlElement("a:effectLst"))
    for ref in q._element.xpath("./p:style/a:effectRef"):
        ref.set("idx", "0")
    if edge is None:
        q.line.fill.background()
    else:
        q.line.color.rgb = rgb(edge)
        q.line.width = Pt(lw)
    if radius:
        q.adjustments[0] = .08
    return q


def txt(s, x, y, w, h, text, size=22, color=BLACK, bold=False,
        align=PP_ALIGN.LEFT, valign=MSO_ANCHOR.MIDDLE, margin=0):
    q = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = q.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.margin_left = tf.margin_right = Inches(margin)
    tf.margin_top = tf.margin_bottom = Inches(margin)
    tf.vertical_anchor = valign
    for index, line in enumerate(str(text).split("\n")):
        p = tf.paragraphs[0] if index == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_before, p.space_after = Pt(0), Pt(0)
        p.line_spacing = 1.12
        r = p.add_run()
        r.text = line
        r.font.name, r.font.size = FONT, Pt(size)
        r.font.bold = bold
        r.font.color.rgb = rgb(color)
        rpr = r._r.get_or_add_rPr()
        ea = OxmlElement("a:ea")
        ea.set("typeface", FONT)
        rpr.append(ea)
    return q


def box(s, x, y, w, h, text, size=22, fill=WHITE, edge=LINE,
        color=BLACK, bold=False, radius=False):
    q = rect(s, x, y, w, h, fill, edge, radius)
    txt(s, x+.09, y+.04, w-.18, h-.08, text, size, color, bold, PP_ALIGN.CENTER)
    return q


def line(s, x1, y1, x2, y2, color=LINE, width=1.2, arrow=False):
    q = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1),
                               Inches(x2), Inches(y2))
    q.line.color.rgb, q.line.width = rgb(color), Pt(width)
    q._element.spPr.append(OxmlElement("a:effectLst"))
    for ref in q._element.xpath("./p:style/a:effectRef"):
        ref.set("idx", "0")
    if arrow:
        end = OxmlElement("a:tailEnd")
        end.set("type", "triangle")
        q._element.spPr.get_or_add_ln().append(end)
    return q


def arrow(s, x1, y1, x2, y2, color=BLACK, width=1.6):
    return line(s, x1, y1, x2, y2, color, width, True)


def new_slide(number, section, title=None, source="", notes=""):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = rgb(WHITE)
    if title:
        txt(s, .65, .32, 14.7, .35, section, 14, BLUE, True)
        txt(s, .65, .85, 14.7, .73, title, 33, BLACK, True)
    line(s, .65, 8.48, 15.35, 8.48, LINE, .8)
    txt(s, .65, 8.59, 13.85, .25, source, 10.5, GRAY)
    txt(s, 14.55, 8.55, .8, .33, f"{number:02d} / 08", 11, BLACK,
        align=PP_ALIGN.RIGHT)
    s.notes_slide.notes_text_frame.text = notes
    return s


def table(s, x, y, widths, rows, heights, header=True, sizes=None,
          fills=None, alignments=None, colors=None):
    yy = y
    for r, row in enumerate(rows):
        xx = x
        for c, value in enumerate(row):
            fill = PALE_BLUE if r == 0 and header else WHITE
            if fills and (r, c) in fills:
                fill = fills[r, c]
            rect(s, xx, yy, widths[c], heights[r], fill=fill, edge=LINE, lw=.65)
            fs = (sizes or {}).get((r, c), 18 if r == 0 and header else 21)
            color = (colors or {}).get((r, c), BLACK)
            alignment = (alignments or {}).get(c, PP_ALIGN.CENTER)
            txt(s, xx+.12, yy+.07, widths[c]-.24, heights[r]-.14,
                value, fs, color, r == 0 and header, alignment)
            xx += widths[c]
        yy += heights[r]


def chip_row(s, x, y, count, w, h=.34, fill=PALE_BLUE, start=0):
    gap = .035
    cw = (w-gap*(count-1))/count
    for i in range(count):
        box(s, x+i*(cw+gap), y, cw, h, str(start+i), 10,
            fill, BLUE, radius=False)


def save_plot(fig, name):
    for suffix in ("png", "svg"):
        fig.savefig(ASSETS / f"{name}.{suffix}", dpi=240,
                    facecolor="white", transparent=False)
    plt.close(fig)


def plot_results():
    font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    font_manager.fontManager.addfont(str(font_path))
    plt.rcParams.update({"font.family": "Noto Sans CJK JP", "font.size": 17,
                         "axes.unicode_minus": False, "svg.fonttype": "path",
                         "text.color": "#111111", "axes.labelcolor": "#111111",
                         "xtick.color": "#111111", "ytick.color": "#111111"})
    fig, ax = plt.subplots(figsize=(11.1, 5.35))
    fig.subplots_adjust(left=.075, right=.997, top=.98, bottom=.145)
    x = np.arange(3)*1.43
    keys = ["baseline_16_separate_ms", "baseline_16_stacked_ms",
            "selected_group_stacked_ms", "learned_group_stacked_ms"]
    cols = [BLACK, MID_GRAY, LIGHT_BLUE, RED]
    for j, (key, color) in enumerate(zip(keys, cols)):
        vals = [d[key] for d in SUMMARY]
        positions = x+(j-1.5)*.245
        ax.bar(positions, vals, width=.225, color="#"+color, zorder=3)
        for xx, value in zip(positions, vals):
            ax.text(xx, value+.28, f"{value:.2f}", ha="center", va="bottom", fontsize=12.9)
    ax.set_xticks(x, ["Granite", "OLMoE", "DeepSeek V2 Lite"], fontsize=20)
    ax.set_ylim(0, 28)
    ax.set_yticks([0, 5, 10, 15, 20, 25])
    ax.set_ylabel("routed FFN 服務時間（ms/token）", fontsize=17, labelpad=9)
    ax.grid(axis="y", color="#E1E5E9", lw=.8, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#C1C7CD")
    ax.tick_params(axis="y", length=0, labelsize=14)
    ax.tick_params(axis="x", length=0, pad=13)
    save_plot(fig, "07_main_results")

    fig, axes = plt.subplots(1, 3, figsize=(14.7, 3.85))
    fig.subplots_adjust(left=.045, right=.99, top=.76, bottom=.20, wspace=.30)
    names = ["Granite", "OLMoE", "DeepSeek V2 Lite"]
    limits = [(7, 14), (12, 33), (20, 61)]
    for ax, model, name, lim in zip(axes, DATA["models"], names, limits):
        vals = [v["ms"] for v in DATA["models"][model]["fixed_stacked"]]
        gs = [v["g"] for v in DATA["models"][model]["fixed_stacked"]]
        xx = np.arange(len(vals))
        ax.plot(xx, vals, color="#111111", lw=2.1, marker="o", ms=6, zorder=3)
        idx = int(np.argmin(vals))
        ax.scatter([4], [vals[-1]], color="#"+BLUE, s=105, zorder=4)
        ax.scatter([idx], [vals[idx]], color="#"+RED, s=135, zorder=5)
        for i, val in enumerate(vals):
            offset = 10 if i != idx else 15
            ax.annotate(f"{val:.2f}", (i, val), xytext=(0, offset),
                        textcoords="offset points", ha="center", fontsize=14.5,
                        color="#"+(RED if i == idx else BLACK),
                        fontweight="bold" if i == idx else "normal")
        ax.set_title(name, loc="left", fontsize=22, pad=34, fontweight="bold")
        ax.text(0, 1.10, f"最低 g = {gs[idx]}", transform=ax.transAxes,
                ha="left", va="center", fontsize=15.5, color="#"+RED,
                fontweight="bold")
        ax.set_xticks(xx, [str(g) for g in gs], fontsize=17)
        ax.set_ylim(*lim)
        ax.set_xlim(-.32, 4.32)
        ax.set_xlabel("g（chips / expert）", fontsize=15, labelpad=9)
        ax.set_ylabel("ms/token", fontsize=14)
        ax.grid(axis="y", color="#E1E5E9", lw=.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#C1C7CD")
        ax.tick_params(axis="both", length=0)
        ax.tick_params(axis="y", labelsize=12)
    save_plot(fig, "08_group_sweep")


def slide1():
    s = new_slide(1, "", source="AiF §4.1–4.4、Table 2；docs/aif_model_notes.md；scripts/materialize_plan.py",
        notes="第一頁直接開始內容，不設封面或頁面標題。\n"
        "AiF 在 flash chip 內加入數位 PE：NAND array → page buffer → ECCLITE → INT8 PE。"
        "主機傳輸向量，權重不必每次離開 chip。\n"
        "幾何依據：原論文 Table 2（PDF 第10頁）：8 channels、每 channel 2 chips、每 chip 4 planes、16 KiB/page。"
        "本專案 chip ID mod 8 對應 channel，因此 channel 0 連 chip 0/8。"
        "一個 block 的192 WL 是由論文量測資料推得的專案設定，並非 Table 2 公布的模擬常數。"
        "圖中僅展開部分 blocks，block 總數未假定。未單獨建模 die。\n"
        "來源：paper/aif.txt:442–455、664–667、705–783；docs/aif_model_notes.md。")
    rect(s, .65, .55, 2.35, 1.02, PALE_BLUE, BLUE)
    txt(s, .79, .68, 2.07, .34, "Host", 22, align=PP_ALIGN.CENTER)
    txt(s, .79, 1.13, 2.07, .29, "輸入 x／結果 y", 18, align=PP_ALIGN.CENTER)
    box(s, 4.0, .55, 3.05, 1.02, "SSD controller\n命令與向量分送", 22)
    arrow(s, 3.02, .88, 3.95, .88)
    txt(s, 3.02, .40, .93, .28, "x", 18, align=PP_ALIGN.CENTER)
    arrow(s, 3.95, 1.30, 3.02, 1.30, BLUE)
    txt(s, 3.05, 1.37, .9, .30, "y", 18, BLUE, align=PP_ALIGN.CENTER)
    rect(s, 8.05, .45, 7.3, 1.6, WHITE, BLUE, lw=1.4)
    txt(s, 8.27, .52, 6.8, .30, "AiFChip 內部的數位 GEMV 資料路徑", 17, bold=True)
    stages = [(8.28, 1.35, "NAND"), (10.0, 1.60, "Page buffer"),
              (11.98, 1.22, "ECCLITE"), (13.57, 1.48, "INT8 PE")]
    for x, w, text in stages:
        box(s, x, 1.04, w, .62, text, 16, PALE_BLUE if "PE" in text else WHITE, LINE,
            color=RED if "PE" in text else BLACK, bold="PE" in text)
    for a, b in zip(stages, stages[1:]):
        arrow(s, a[0]+a[1]+.03, 1.35, b[0]-.04, 1.35, BLACK, 1.2)
    arrow(s, 7.10, .90, 7.98, .90)
    txt(s, 7.07, .43, .96, .28, "x", 17, align=PP_ALIGN.CENTER)
    arrow(s, 7.98, 1.60, 7.10, 1.60, BLUE)
    txt(s, 7.1, 1.65, .89, .28, "y", 17, BLUE, align=PP_ALIGN.CENTER)
    txt(s, .65, 1.98, 14.7, .55, "權重留在 chip 內讀取、修正錯誤與計算；主機主要交換向量。", 26, bold=True)

    rect(s, .65, 2.8, 6.5, 4.55, WHITE, BLACK, lw=1.3)
    txt(s, .92, 2.98, 5.97, .43, "AiF SSD", 25, bold=True)
    txt(s, .92, 3.49, 6.0, .40, "8 channels × 2 chips = 16 chips", 21)
    for channel in range(8):
        col, row = channel % 4, channel // 4
        x, y = .92+col*1.51, 4.17+row*1.4
        rect(s, x, y, 1.35, 1.16, PALE_BLUE, BLUE)
        txt(s, x+.08, y+.06, 1.19, .30, f"Channel {channel}", 14, bold=True, align=PP_ALIGN.CENTER)
        for j, chip in enumerate((channel, channel+8)):
            box(s, x+.10+j*.60, y+.50, .55, .43, f"C{chip}", 12, WHITE,
                RED if chip == 0 else LINE, color=RED if chip == 0 else BLACK, bold=chip == 0)
    txt(s, .92, 7.5, 6.2, .50, "chip 內頻寬：6.4 GB/s × 16 = 102.4 GB/s", 20)
    arrow(s, 7.2, 5.03, 7.63, 5.03, BLUE, 1.8)

    rect(s, 7.72, 2.8, 7.63, 4.55, WHITE, BLUE, lw=1.3)
    txt(s, 7.98, 2.98, 7.07, .43, "Chip 0 放大：4 planes / chip", 24, bold=True)
    px = 7.98
    for p in range(4):
        pw = 3.10 if p == 0 else 1.10
        rect(s, px, 3.61, pw, 3.40, PALE_BLUE, BLUE)
        txt(s, px+.08, 3.72, pw-.16, .37, f"Plane {p}", 18 if p == 0 else 15,
            bold=True, align=PP_ALIGN.CENTER)
        if p == 0:
            rect(s, px+.16, 4.23, pw-.32, 1.83, WHITE, BLACK)
            txt(s, px+.30, 4.30, pw-.60, .37, "Block A", 19, bold=True)
            for j in range(3):
                box(s, px+.31, 4.82+j*.34, pw-.62, .29,
                    f"WL {j}  →  LSB page", 13, PALE_BLUE, LINE)
            box(s, px+.16, 6.28, pw-.32, .46, "Block B …", 18, WHITE, LINE)
        else:
            for j in range(3):
                box(s, px+.13, 4.29+j*.78, pw-.26, .55, "Block", 12, WHITE, LINE)
        px += pw+.17
    txt(s, 7.98, 7.50, 7.1, .48, "LSB page：16 KiB；4-plane wave 最多讀 64 KiB", 19)
    txt(s, .65, 8.13, 14.7, .25, "本專案採 192 WL/block（由論文量測資料推得）；die 未獨立建模。", 12.5, GRAY)


def slide2():
    s = new_slide(2, "01 / AiF FLASH 架構與讀取條件", "cr-read：同一 block 內，不中斷地連續讀取",
        "AiF §4.2、Table 2；scripts/aif_model.py:Demand；docs/aif_model_notes.md",
        "cr-read 的必要條件：同一 plane、同一 block 內的連續 LSB pages，無任意間隔。"
        "跨 block、新 GEMV stream 或中斷後，不假設可以保留原讀取狀態。"
        "28 μs 是 be-enc LSB 普通讀取；9.7 μs 是後續 cr-read。"
        "T_run=28+(P−1)×9.7 是專案依論文相位推導的模型，28已含開始與結束成本，不再加尾端28。"
        "四 planes 取最慢 plane 並與 6.4GB/s/chip 管線上限取 max。"
        "4 pages 示例：普通 4×28=112μs；連續 28+3×9.7=57.1μs，僅解析讀取示例。")
    txt(s, .65, 1.70, 14.7, .48, "權重放在 LSB pages；第一頁付普通讀取成本，後續頁才能使用 cr-read。", 22)
    rect(s, .65, 2.54, 10.50, 2.52, WHITE, BLUE, lw=1.3)
    txt(s, .9, 2.69, 9.9, .42, "同一 plane ／ Block A ／ 連續 LSB pages", 22, bold=True)
    xs = [1.02, 3.52, 6.02, 8.52]
    for i, x in enumerate(xs):
        box(s, x, 3.35, 1.95, .77, f"Page {i}", 23, PALE_RED if i == 0 else PALE_BLUE,
            RED if i == 0 else BLUE)
        txt(s, x, 4.33, 1.95, .43, "28 μs" if i == 0 else "9.7 μs", 25,
            RED if i == 0 else BLUE, True, PP_ALIGN.CENTER)
        if i < 3:
            arrow(s, x+1.99, 3.74, xs[i+1]-.07, 3.74)
    rect(s, 12.05, 2.54, 3.3, 2.52, WHITE, RED, lw=1.3)
    txt(s, 12.22, 2.69, 2.96, .42, "Block B", 23, bold=True, align=PP_ALIGN.CENTER)
    box(s, 12.55, 3.35, 2.30, .77, "Page 0", 23, PALE_RED, RED)
    txt(s, 12.55, 4.33, 2.30, .43, "28 μs", 25, RED, True, PP_ALIGN.CENTER)
    arrow(s, 11.23, 3.74, 11.96, 3.74, RED)
    txt(s, 11.18, 3.01, .80, .46, "跨 block", 13.5, RED, True, PP_ALIGN.CENTER)
    txt(s, .65, 5.30, 14.7, .47, "跨 block 或開始新 stream，就要重新付一次 28 μs。", 25, RED, True)

    rect(s, .65, 6.13, 8.60, 1.78, PALE_BLUE, None)
    txt(s, .92, 6.32, 8.03, .62, "T_run(P) = 28 + (P − 1) × 9.7 μs", 29, bold=True)
    txt(s, .92, 7.10, 8.03, .47, "P：同一 plane／block 中連續讀取的頁數（P ≥ 1）", 18)
    box(s, 9.65, 6.13, 2.65, 1.78, "普通讀 4 頁\n112 μs", 24, WHITE, LINE)
    box(s, 12.70, 6.13, 2.65, 1.78, "連續讀 4 頁\n57.1 μs", 24, PALE_BLUE, BLUE)
    txt(s, .65, 8.10, 14.7, .28, "解析示例；28 μs 已包含整段的開始與結束成本。四 planes 可並行，chip 另受 6.4 GB/s 上限限制。", 13, GRAY)


def slide3():
    s = new_slide(3, "02 / 模型大小與分組延遲", "要看單個 expert 的 projection 大小，才能判斷切分是否太細",
        "GGUF metadata；results/experiment_manifest.json；docs/aif_model_notes.md",
        "尺寸由實際GGUF矩陣維度與INT8 1 byte/weight推導。"
        "總參數以GGUF metadata所有tensor shape_ggml元素加總：Granite1334628352、OLMoE6919161856、DeepSeek15706484224。"
        "Granite H1024 I512 E32 K8 L24；OLMoE H2048 I1024 E64 K8 L16；"
        "DeepSeek H2048 I1408 E64 K6 routed L26（總L27）。"
        "projection H×I；expert3×H×I；全部 routed=L×E×3HI。"
        "DeepSeek全部routed為13.40625GiB，表格四捨五入為13.406。"
        "模型檔案大小由manifest.model_bytes計算，屬實際GGUF格式：821847360、4213511776、16702520416bytes，"
        "分別0.765405/3.924139/15.555434GiB。"
        "GGUF Q4/Q8只用於真實routing；不能與INT8模型服務的權重大小混用。"
        "不含attention/shared experts/dense FFN/scale metadata。")
    rect(s, .65, 1.80, 14.70, .77, PALE_BLUE, None)
    txt(s, .91, 1.90, 14.2, .55, "INT8：1 projection = H × I bytes；1 expert = 3 × H × I bytes", 26, bold=True)
    rows = [
        ["模型／總參數", "H × I", "Experts\nE / top-k", "Routed\n層數", "單 projection", "單 expert", "全部 routed\nINT8 權重"],
        ["Granite 3.1 A400M\n1.335 B", "1024 × 512", "32 / 8", "24", "0.5 MiB", "1.5 MiB", "1.125 GiB"],
        ["OLMoE 1B 7B\n6.919 B", "2048 × 1024", "64 / 8", "16", "2 MiB", "6 MiB", "6 GiB"],
        ["DeepSeek V2 Lite\n15.706 B", "2048 × 1408", "64 / 6", "26", "2.75 MiB", "8.25 MiB", "13.406 GiB"],
    ]
    sizes = {(0,c):17 for c in range(7)}
    sizes.update({(r,1):19 for r in range(1,4)})
    sizes.update({(r,6):20 for r in range(1,4)})
    sizes.update({(r,0):19 for r in range(1,4)})
    fills = {(r,4):PALE_BLUE for r in range(1,4)}
    colors = {(r,4):RED for r in range(1,4)}
    table(s, .65, 2.93, [3.0,2.2,1.7,1.25,2.15,2.05,2.35], rows,
          [.82,1.02,1.02,1.02], sizes=sizes, fills=fills, colors=colors,
          alignments={0:PP_ALIGN.LEFT})
    txt(s, .65, 7.04, 14.7, .55, "Granite 的 0.5 MiB projection 切到 16 chips：每 chip 只有 32 KiB。", 26, RED, True)
    txt(s, .65, 7.82, 14.7, .32,
        "Routing 用 GGUF：Granite Q4 0.765 GiB ｜ OLMoE Q4 3.924 GiB ｜ DeepSeek Q8 15.555 GiB", 16.5)
    txt(s, .65, 8.19, 14.7, .23, "總參數由 GGUF tensors 加總。INT8 欄只含 routed experts；DeepSeek 排除首層 dense 與 shared experts。", 12.5, GRAY)


def slide4():
    s = new_slide(4, "02 / 模型大小與分組延遲", "分組可以讓單條 stream 變長，也讓不同 experts 並行",
        "scripts/aif_model.py:phase_demand、Demand；Granite 尺寸的解析示例",
        "此頁為解析示例，單一projection phase、8個相同expert、Granite S512KiB、理想均衡routing；"
        "未串接gate/up且不含控制或傳輸。所有stream均在block內。"
        "w=ceil((S/g)/(4×16KiB))。g16 shard32KiB pages2 wave1 t28；"
        "g8 shard64KiB pages4 wave1 t28；g4 shard128KiB pages8 wave2 t37.7。"
        "本例bandwidth下限均小於NAND讀取時間，因此T_stream採28+(w−1)9.7。"
        "均衡nmax分別8、4、2，所以phase224、112、75.4us。"
        "真實routing不保證平均；專案同chip streams串行、不同groups並行；未做跨expert plane packing。"
        "16chips始終固定，g只代表每expert使用chip數，groups=16/g。")
    txt(s, .65, 1.70, 14.7, .45, "g = 每個 expert 使用的 chip 數；總共固定 16 chips，因此 groups = 16 / g。", 22)
    rect(s, .65, 2.38, 14.7, 1.20, PALE_BLUE, None)
    txt(s, .92, 2.53, 7.0, .43, "T_stream = 28 + (w − 1) × 9.7 μs", 26, bold=True)
    txt(s, .92, 3.02, 6.8, .32, "w = 向上取整［每 chip bytes / 64 KiB］", 17)
    txt(s, 8.27, 2.53, 6.80, .43, "T_phase ≈ n_max × T_stream", 27, bold=True)
    txt(s, 8.27, 3.02, 6.80, .32, "n_max：同時選中 experts 在最忙 group 的數量", 17)

    txt(s, .65, 3.90, 7.75, .46, "同樣 8 個 experts：g = 16", 24, bold=True)
    chip_row(s, .8, 4.49, 16, 7.45, .35)
    for i in range(8):
        box(s, .8+i*.95, 5.00, .83, .52, f"E{i}", 17, PALE_RED, RED)
    txt(s, .8, 5.65, 7.45, .35, "1 個 group，8 個 experts 依序處理 → 224 μs", 20, RED, True)
    txt(s, .65, 6.20, 7.75, .46, "改為 g = 4：4 個 groups 可同時工作", 23, bold=True)
    for group in range(4):
        x = .8+group*1.91
        rect(s, x-.07, 6.76, 1.84, 1.08, WHITE, BLUE, lw=.9)
        chip_row(s, x, 6.87, 4, 1.72, .29, start=group*4)
        box(s, x, 7.30, .79, .47, f"E{group}", 16, PALE_BLUE, BLUE)
        box(s, x+.90, 7.30, .79, .47, f"E{group+4}", 16, PALE_BLUE, BLUE)
    txt(s, .8, 7.92, 7.45, .35, "每組 2 個 experts → 2 × 37.7 = 75.4 μs", 21, BLUE, True)

    rows = [
        ["Granite：512 KiB / projection", "g = 16", "g = 8", "g = 4"],
        ["每 chip 權重", "32 KiB", "64 KiB", "128 KiB"],
        ["四-plane waves：w", "1", "1", "2"],
        ["單條 stream（μs）", "28", "28", "37.7"],
        ["groups", "1", "2", "4"],
        ["最忙 group：n_max", "8", "4", "2"],
        ["整個 phase（μs）", "224", "112", "75.4"],
    ]
    sizes={(0,0):14,(0,1):17,(0,2):17,(0,3):17}
    sizes.update({(r,c):17 for r in range(1,7) for c in range(4)})
    table(s, 8.80, 4.07, [3.0,1.16,1.16,1.23], rows, [.62]+[.49]*6,
          sizes=sizes, fills={(6,c):PALE_BLUE for c in range(4)},
          colors={(6,3):RED}, alignments={0:PP_ALIGN.LEFT})
    txt(s, 8.82, 7.82, 6.40, .43, "單 expert 較慢，整批仍可能更快。", 23, RED, True)
    txt(s, .65, 8.25, 14.7, .19, "示意假設：8 experts 理想均衡、單一 projection phase；未串接 gate/up，不含控制／向量成本。真實 routing 可能碰撞。", 11.2, GRAY)


def slide5():
    s = new_slide(5, "02 / 模型大小與分組延遲", "gate/up 串接延長連續讀取；固定排列減少 group 排隊",
        "scripts/aif_model.py:stack_gate_up；scripts/optimize_placement.py；docs/placement_permutation_findings.md",
        "gate/up共用輸入x，實體垂直堆疊2I×H後形成一條連續stream，並非只在軟體同phase發命令。"
        "Granite g16：每chip gate32KiB、up32KiB，原本兩條stream各28us；串接64KiB只一波28us。"
        "權重總量不變，down依然等activation完成。activation/combine只為依賴關係圖示，其計算時間未納入結果。"
        "下半部固定排列是教學例；選中IDs為0,4,8,12,1,5,2,3，原分布[4,2,1,1]，"
        "交換實體slots後[2,2,2,2]。未選中slots也交換以維持每組容量。"
        "learned placement只學固定expert到group映射，不修改model權重或routing，屬heuristic非全域最優。")
    txt(s, .65, 1.73, 14.7, .40, "Granite、g = 16 的簡單例子：gate/up 共用同一份輸入 x。", 22)
    rect(s, .65, 2.35, 6.7, 1.76, WHITE, LINE)
    txt(s, .89, 2.51, 6.22, .32, "分開存放", 20, bold=True)
    box(s, .98, 3.02, 2.65, .67, "gate：32 KiB", 21, PALE_RED, RED)
    box(s, 4.10, 3.02, 2.85, .67, "up：32 KiB", 21, PALE_RED, RED)
    txt(s, 3.66, 3.08, .4, .51, "+", 23, align=PP_ALIGN.CENTER)
    txt(s, 3.9, 2.48, 3.0, .35, "28 + 28 = 56 μs", 20, RED, True,
        align=PP_ALIGN.RIGHT)
    arrow(s, 7.48, 3.24, 8.16, 3.24)
    rect(s, 8.3, 2.35, 7.05, 1.76, PALE_BLUE, BLUE)
    txt(s, 8.56, 2.51, 6.5, .32, "實體串接成連續 2I × H 矩陣", 20, bold=True)
    box(s, 8.64, 3.02, 3.78, .67, "[ gate ; up ]：64 KiB", 21, WHITE, BLUE)
    txt(s, 12.62, 3.04, 2.40, .57, "一次 28 μs", 25, RED, True, PP_ALIGN.CENTER)
    txt(s, .65, 4.34, 14.7, .40,
        "x → [gate ; up] → 拆回輸出 → activation(gate) ⊙ up → down", 23, bold=True)
    line(s, .65, 5.04, 15.35, 5.04, LINE)

    txt(s, .65, 5.28, 14.7, .42, "固定排列：把常一起啟用的 experts 放到不同 groups", 26, bold=True)
    configurations = [(0.85, "原排列", [4,2,1,1], RED),
                      (8.65, "調整固定 slots 後", [2,2,2,2], BLUE)]
    for x, title, counts, accent in configurations:
        txt(s, x, 5.92, 6.55, .40, title, 21, bold=True)
        for group, count in enumerate(counts):
            yy=6.42+group*.34
            txt(s, x, yy, 1.08, .27, f"Group {group}", 13)
            for j in range(count):
                rect(s, x+1.20+j*.65, yy, .55, .27,
                     PALE_RED if accent == RED else LIGHT_BLUE, accent, lw=.8)
            txt(s, x+4.18, yy, 1.9, .27, f"{count} expert"+("s" if count>1 else ""), 14)
        txt(s, x, 7.72, 6.55, .35, f"最忙 group：{max(counts)} 個 experts 排隊", 19, accent, True)
    arrow(s, 7.32, 7.02, 8.13, 7.02)
    txt(s, .65, 8.18, 14.7, .23, "下半部為示意；選中的 experts 不變，僅交換固定位置，保留各組容量。權重與 router 都不重新訓練。", 12, GRAY)


def slide6():
    s = new_slide(6, "03 / 實驗方法與結果", "固定真實 routing，比較配置，再以 NVMeVirt 重播核對",
        "README.md；scripts/evaluate_learned_overhead.py；results/nvmevirt_validation.json；results/experiment_manifest.json",
        "三模型均使用explanation/programming/mathematics三個plain-text prompts，各32decode steps。"
        "真實llama.cpp擷取expert IDs，開關tracing的greedy輸出逐byte一致。"
        "每折兩prompts學placement並選g/stacking，第三prompt只驗證，共三折。"
        "固定g搜尋1,2,4,8,16，沒有改選中expert IDs。"
        "v3 controller fanout每logical input只跨PCIe一次，controller再分送chips。"
        "chip command0.25us、host phase2us、input1B/output4B為展示假設；PCIe8GB/s、8channels×2GB/s。"
        "v3已保存48workloads、183744 NVMe commands，所有kernel counters吻合。"
        "這驗證Python/kernel模型實作一致，不構成獨立實體AiF效能量測。"
        "服務時間含flash/control/host/input/output，不含attention/router/activation/combine/shared/dense。"
        "資料日期2026-10-19；本簡報製作未重跑推論或啟動VM。")
    xs = [.65,4.44,8.23,12.02]
    labels = ["真實 llama.cpp", "掃描配置", "保留 prompt 驗證", "NVMeVirt 重播"]
    descs = ["3 模型、3 prompts\n每組 32 次 decode", "g = 1 / 2 / 4 / 8 / 16\n串接與固定排列", "2 組訓練、1 組保留\n輪流做 3 折", "實際送入 NVMe queue\n核對 kernel counters"]
    for i,x in enumerate(xs):
        rect(s, x, 1.97, 3.33, 1.82, PALE_BLUE if i in (0,2) else WHITE,
             BLUE if i in (0,2) else LINE)
        txt(s, x+.18, 2.12, 2.97, .50, labels[i], 23, bold=True,
            align=PP_ALIGN.CENTER)
        txt(s, x+.16, 2.84, 3.01, .65, descs[i], 17,
            align=PP_ALIGN.CENTER)
        if i<3:
            arrow(s, x+3.38, 2.89, xs[i+1]-.07, 2.89, BLACK)

    txt(s, .65, 4.12, 7.1, .40, "三折：只用訓練 prompts 選配置", 24, bold=True)
    cols=[2.07,4.07,6.07]
    for x, name in zip(cols,["說明", "程式", "數學"]):
        txt(s, x, 4.76, 1.7, .35, name, 20, bold=True, align=PP_ALIGN.CENTER)
    for fold in range(3):
        y=5.36+fold*.64
        txt(s, .72, y, 1.22, .46, f"Fold {fold+1}", 17)
        for j,x in enumerate(cols):
            held=j==fold
            box(s, x, y, 1.70, .46, "保留驗證" if held else "訓練", 17,
                PALE_RED if held else PALE_BLUE, RED if held else BLUE)

    txt(s, 8.65, 4.12, 6.70, .40, "v3：controller 複製輸入向量", 24, bold=True)
    txt(s, 8.65, 4.77, 6.70, .45, "每份 logical input 只經 PCIe 傳一次", 21)
    table(s, 8.65, 5.38, [3.5,3.20], [
        ["Chip command", "0.25 μs"], ["Host phase", "2 μs"],
        ["PCIe / channel", "8 / 2 GB/s"], ["輸入 / 輸出元素", "1 / 4 bytes"]],
        [.40]*4, header=False, sizes={(r,c):17 for r in range(4) for c in range(2)},
        alignments={0:PP_ALIGN.LEFT})
    txt(s, 8.65, 7.10, 6.70, .39, "命令延遲與向量精度為展示假設，待硬體校準。", 15, RED)
    rect(s, .65, 7.45, 14.70, .60, PALE_BLUE, None)
    txt(s, .90, 7.51, 14.20, .43,
        f"v3 驗證：{VALIDATION['v3_workloads']} workloads ／ {VALIDATION['v3_commands']:,} NVMe commands ／ counters 全部相符", 24, bold=True)
    txt(s, .65, 8.17, 14.7, .22, "以下為建模的 routed FFN 服務時間；非實體 AiF 硬體或完整模型推論速度。", 13, GRAY)


def slide7():
    s = new_slide(7, "03 / 實驗方法與結果", "Granite 改善最明顯；相同方法對三個模型的收益不同",
        "results/summary.csv、summary.json；v3 controller-fanout；保存結果日期 2026-10-19",
        "全部數字為每decoded token全部納入routed layers之FFN服務時間ms，越低越好。"
        "四組配置：16分開、16串接、訓練選g串接原排列、訓練選g串接學習排列。"
        "Granite 18.920064/12.703488/8.3781698333/7.4515605，原/學g4/4，"
        "相對16已串接改善41.3424%。OLMoE16.567552/13.648384/13.3899251667/12.8242079167，g8/8，6.0386%。"
        "DeepSeek24.916736/21.332688/21.332688/21.0150557083，原g16、學g8，1.4889%。"
        "選配置後為三折heldout平均。DeepSeek同時改g與排列，不能將差全部歸因於同g下的排列。"
        "控制/傳輸成本假設同第6頁。僅routed FFN，不能換算完整tokens/s。")
    legends=[(.72,BLACK,"16 chips・gate/up 分開"), (4.60,MID_GRAY,"16 chips・gate/up 串接"),
             (8.50,LIGHT_BLUE,"選 g＋串接・原排列"), (12.1,RED,"選 g＋串接・學習排列")]
    for x,color,text in legends:
        rect(s, x, 1.88, .20, .20, color, None)
        txt(s, x+.30, 1.78, 3.12, .43, text, 16.3)
    s.shapes.add_picture(str(ASSETS/"07_main_results.png"), Inches(.53), Inches(2.37),
                         width=Inches(11.18), height=Inches(5.38))
    for i, (row, name, g) in enumerate(zip(SUMMARY,["Granite", "OLMoE", "DeepSeek"],[4,8,8])):
        y=2.55+i*1.57
        rect(s, 12.0, y, 3.35, 1.37, WHITE, LINE)
        txt(s, 12.18, y+.11, 2.99, .33, f"{name}  ·  g = {g}", 18.5, bold=True)
        txt(s, 12.18, y+.45, 2.99, .48, f"−{row['improvement_vs_16_stacked']*100:.2f}%", 34, RED, True)
        txt(s, 12.18, y+1.02, 2.99, .25,
            f"{row['baseline_16_stacked_ms']:.3f} → {row['learned_group_stacked_ms']:.3f} ms", 15.5)
    txt(s, 12.0, 7.42, 3.35, .58, "降低幅度皆相對\n16-chip 已串接基準", 17, bold=True)
    txt(s, .65, 7.93, 11.0, .37, "三折保留 prompt 平均。DeepSeek 原排列選 g=16，學習排列選 g=8。", 16.5)


def slide8():
    s = new_slide(8, "03 / 實驗方法與結果", "g 沒有通用答案：expert 尺寸、碰撞與控制成本要一起看",
        "doc/assets/figure_data.json；NFS runs/*-aif/overhead_fanout/；learned_overhead_fanout/",
        "上圖為固定原排列、gate/up串接的g掃描，三組prompts合併平均，非使用測試集選擇的heldout配置。"
        "Granite g1/2/4/8/16:11.4129990833/9.1095918333/8.3781698333/9.0552495833/12.703488。"
        "OLMoE30.326588125/20.6209638333/15.61653475/13.3899251667/13.648384。"
        "DeepSeek56.6565426667/37.4578758333/26.5263968333/21.6045643333/21.332688。"
        "各圖y軸依模型縮放，曲線不是端到端測速。"
        "DeepSeek下方成本範例為選定配置比較：原g16串接，flash19.3284、control1.248、host.104、input.272896、output.379392；"
        "學習g8串接，flash19.7449010417、control.624、host.104、input.1627626667、output.379392。"
        "delta flash+.4165010417、control−.624、input−.1101333333，host/output不變；total−.3176322917ms。"
        "全部為服務模型與展示成本假設；僅三個短prompts，未經實體AiF校準，不含attention等其他計算。")
    txt(s, .65, 1.72, 14.7, .42,
        "固定原排列、gate/up 已串接；三組 prompts 合併平均（各圖 y 軸依模型縮放）。", 20.5)
    s.shapes.add_picture(str(ASSETS/"08_group_sweep.png"), Inches(.65), Inches(2.40),
                         width=Inches(14.7), height=Inches(3.85))
    rect(s, .65, 6.34, 14.7, 1.48, PALE_BLUE, None)
    txt(s, .91, 6.49, 14.18, .38,
        "DeepSeek：原排列 g=16 → 學習排列 g=8，總時間仍可從 21.333 降至 21.015 ms。", 23, bold=True)
    blocks=[(.95,3.0,"Flash  +0.417 ms",RED), (4.46,3.38,"控制  −0.624 ms",BLACK),
            (8.28,3.20,"輸入  −0.110 ms",BLACK), (12.05,2.95,"合計  −0.318 ms",RED)]
    for x,w,label,color in blocks:
        txt(s, x, 7.11, w, .42, label, 22.5, color, True)
    txt(s, .65, 8.01, 14.7, .35,
        "配置選擇需同時看 flash、命令與傳輸；本輪只有三個短 prompts，控制成本仍待硬體校準。", 17)


def verify_data():
    assert len(SUMMARY) == 3
    for row in SUMMARY:
        expected=1-row["learned_group_stacked_ms"]/row["baseline_16_stacked_ms"]
        assert math.isclose(expected, row["improvement_vs_16_stacked"], rel_tol=1e-12)
    assert math.isclose(28+3*9.7, 57.1)
    assert [8*28, 4*28, 2*(28+9.7)] == [224,112,75.4]
    assert VALIDATION["all_kernel_counters_match"]
    for row,model in zip(SUMMARY,DATA["main"]):
        assert row == model


def main():
    verify_data()
    plot_results()
    for function in (slide1,slide2,slide3,slide4,slide5,slide6,slide7,slide8):
        function()
    assert len(prs.slides) == 8
    # Guard against accidental geometry outside the presentation canvas.
    for index, slide in enumerate(prs.slides,1):
        for shape in slide.shapes:
            assert shape.left >= 0 and shape.top >= 0, (index,shape.name)
            assert shape.left+shape.width <= prs.slide_width+5, (index,shape.name,"right")
            assert shape.top+shape.height <= prs.slide_height+5, (index,shape.name,"bottom")
    target=OUT/"AiF_Flash_Meeting_8slides.pptx"
    prs.save(target)
    snapshot={"experiment_date":MANIFEST["date"], "deck_date":"2026-10-27",
              "results":SUMMARY, "figure_data":DATA,
              "model_files":[{"name":m["name"], "parameters":PARAMETERS[m["name"]], "bytes":m["model_bytes"],
                              "GiB":m["model_bytes"]/2**30} for m in MANIFEST["models"]],
              "verification":{"v3_workloads":VALIDATION["v3_workloads"],
                              "v3_commands":VALIDATION["v3_commands"],
                              "all_kernel_counters_match":VALIDATION["all_kernel_counters_match"]}}
    (OUT/"data_snapshot.json").write_text(json.dumps(snapshot,ensure_ascii=False,indent=2)+"\n")
    print(target)


if __name__ == "__main__":
    main()
