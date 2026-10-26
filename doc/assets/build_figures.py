#!/usr/bin/env python3
"""Build the report's static figures from its saved source-data snapshot."""
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np

OUT = Path(__file__).resolve().parent
DATA = json.loads((OUT / 'figure_data.json').read_text())
FONT = Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
if FONT.exists():
    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(FONT)).get_name()
plt.rcParams.update({'font.size': 11, 'axes.unicode_minus': False,
                     'savefig.facecolor': '#ffffff', 'svg.fonttype': 'path'})
INK, MUTED, BLUE, GREEN, ORANGE = '#183248', '#5c6e7b', '#3478ac', '#25877b', '#d68c32'
LIGHT, GRAY = '#edf3f7', '#a4adb5'


def canvas(width=12, height=6):
    fig, ax = plt.subplots(figsize=(width, height))
    fig.patch.set_facecolor('white')
    ax.set(xlim=(0, 12), ylim=(0, height))
    ax.axis('off')
    fig.subplots_adjust(left=.025, right=.975, top=.97, bottom=.035)
    return fig, ax


def label(ax, x, y, text, size=11, color=INK, **kw):
    return ax.text(x, y, text, fontsize=size, color=color, va='center', **kw)


def box(ax, x, y, w, h, text, color=LIGHT, edge=None, size=11):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.04,rounding_size=.09',
                               linewidth=1, facecolor=color, edgecolor=edge or color))
    label(ax, x+w/2, y+h/2, text, size=size, ha='center')


def arrow(ax, a, b, color=MUTED):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=14,
                                linewidth=1.5, color=color))


def save(fig, stem):
    for ext in ('png', 'svg'):
        fig.savefig(OUT / f'{stem}.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)


def datapath():
    fig, ax = canvas(height=5.2)
    label(ax, .2, 4.85, 'AiF 把矩陣乘法移到 flash chip 內，減少權重搬運', 17, weight='bold')
    label(ax, .2, 4.27, '一般 SSD offloading', 12, weight='bold')
    for x, w, text in [(0.3, 2.3, 'NAND flash\n存放大量權重'), (4.2, 2.6, 'SSD controller\n／外部介面'),
                       (8.5, 3, '主機 DRAM ＋ CPU\n計算 W × x')]:
        box(ax, x, 2.95, w, .95, text)
    arrow(ax, (2.65, 3.42), (4.12, 3.42))
    arrow(ax, (6.86, 3.42), (8.43, 3.42))
    label(ax, 7.65, 3.88, '搬整批權重', 10, ha='center', color=ORANGE)
    label(ax, .2, 2.35, 'AiF：chip 內讀取、修正錯誤，再做 GEMV', 12, weight='bold')
    box(ax, .3, .86, 6.1, 1.08, 'NAND → page buffer → ECCLITE → PE\n權重留在 chip 內；PE 計算 W × x', '#e5f1ee', size=12)
    box(ax, 8.5, .86, 3, 1.08, '主機／controller\n送輸入 x，收結果 y')
    arrow(ax, (8.43, 1.62), (6.47, 1.62), GREEN)
    arrow(ax, (6.47, 1.1), (8.43, 1.1), GREEN)
    label(ax, 7.45, 2.0, '向量輸入', 10, ha='center')
    label(ax, 7.45, .65, '向量結果', 10, ha='center')
    label(ax, .3, .25, '概念圖；省略協定細節。AiF 仍須支付 NAND 讀取、控制與向量傳輸成本。', 10, color=MUTED)
    save(fig, '01_aif_datapath')


def geometry():
    fig, ax = canvas(height=6.4)
    label(ax, .2, 6.05, '這裡的「位置」有三層：chip 群組、plane 分頁、block 內連續位址', 16, weight='bold')
    box(ax, .3, 4.7, 2.1, .8, 'SSD\n8 條 channels')
    arrow(ax, (2.48, 5.1), (3.0, 5.1))
    box(ax, 3.05, 4.7, 2.6, .8, '每 channel：2 chips\n共 16 chips')
    arrow(ax, (5.72, 5.1), (6.2, 5.1))
    box(ax, 6.25, 4.7, 2.5, .8, '每 chip：4 planes\n共 64 planes')
    arrow(ax, (8.83, 5.1), (9.3, 5.1))
    box(ax, 9.35, 4.7, 2.3, .8, 'plane 內有 blocks\nLSB page：16 KiB')
    label(ax, .3, 4.2, '分組範例 g = 4：每個 expert 只切到固定的 4 顆 chips，權重不複製', 12)
    colors = ['#dcebf5', '#e0f1e9', '#fff0d8', '#ebe4f4']
    for i in range(16):
        x = .35 + i*.715
        box(ax, x, 3.23, .61, .62, str(i), colors[i//4], size=10)
    for j in range(4):
        label(ax, 1.72+j*2.86, 2.94, f'group {j}', 10, ha='center')
    label(ax, .3, 2.4, '同一 chip 內的連續 stream：一波最多讀 4 個 pages（每 plane 一頁）', 12)
    for plane in range(4):
        label(ax, 1.2+plane*1.15, 1.92, f'plane {plane}', 10, ha='center')
        for wave in range(2):
            box(ax, .75+plane*1.15, 1.28-wave*.55, .9, .42, f'page {plane+4*wave}', '#dcebf5', size=9)
    label(ax, 5.55, 1.7, 'block 內連續讀：第一波 28 μs', 11)
    label(ax, 5.55, 1.15, 'NAND 後續每波 9.7 μs；跨 block 重新開始', 10.5)
    label(ax, 5.55, .72, 'chip 服務仍受 6.4 GB/s 串流上限限制', 10.5)
    label(ax, .3, .27, '本專案未把 die、實體字線快慢差異或 FTL 搬移列為獨立模型；192 WL/block 是依論文量測資料推得。', 10, color=MUTED)
    save(fig, '02_flash_geometry')


def short_reads():
    fig, ax = canvas(height=6.2)
    label(ax, .2, 5.85, 'Granite 單一 projection：512 KiB，切得越碎，每次讀取越短', 16, weight='bold')
    for row, (g, n, waves, t, color) in enumerate([(16, 2, 1, 28, BLUE), (4, 8, 2, 37.7, GREEN)]):
        yy = 4.6-row*1.55
        label(ax, .35, yy+.15, f'g = {g}\n每 chip：{512//g} KiB', 12)
        for wave in range(waves):
            for p in range(4):
                filled = wave*4+p < n
                x = 2.5+p*.7
                y = yy-.18-wave*.49
                ax.add_patch(Rectangle((x, y), .58, .39, facecolor=color if filled else 'white',
                                       edgecolor=color, linewidth=1.3))
        label(ax, 5.8, yy+.02, f'{n} pages／chip，{waves} 波讀取', 11)
        label(ax, 9.0, yy+.02, f'{t:g} μs／stream', 12, color=color, weight='bold')
    label(ax, .35, 2.03, 'g = 4 的單一 stream 較長；不同 experts 可分散到 4 個 groups 並行。', 11)
    label(ax, .35, 1.57, 'g = 16 時所有 experts 都用同一批 chips；本專案每 chip 依序處理各 stream。', 11)
    box(ax, .4, .43, 11.1, .68, 'gate/up 串接：g = 16 時，兩個 32 KiB streams → 一個 64 KiB stream\n兩次 28 μs → 一次 28 μs；down 仍須等 activation 完成', '#fff1dc', size=11)
    label(ax, .35, .03, '解析示例，僅為權重服務時間；不含命令與向量成本，也不代表獨立 expert 可共用空 plane。', 9, color=MUTED)
    save(fig, '03_short_reads')


def collisions():
    fig, ax = canvas(height=5.1)
    label(ax, .2, 4.75, 'learned placement：讓常一起選中的 experts 少擠在同一組', 16, weight='bold')
    before = [[0, 4, 8, 12], [1, 5], [2], [3]]
    after = [[0, 4], [1, 5], [2, 12], [3, 8]]
    colors = ['#dcebf5', '#e0f1e9', '#fff0d8', '#ebe4f4']
    for side, (xx, title, groups) in enumerate([(0.4, '原始排列：group = expert ID mod 4', before),
                                              (6.5, '學到的固定排列：分散共同啟用', after)]):
        label(ax, xx, 4.1, title, 11, weight='bold')
        for j, experts in enumerate(groups):
            yy=3.48-j*.59
            box(ax, xx, yy-.18, .94, .4, f'組 {j}', colors[j], size=10)
            for k, e in enumerate(experts):
                box(ax, xx+1.16+k*.66, yy-.18, .55, .4, str(e), colors[j], size=10)
        label(ax, xx, .88, '最忙的 group：'+('4' if side==0 else '2')+' 個 experts 要排隊', 12,
              color=ORANGE if side==0 else GREEN)
    arrow(ax, (5.25, 2.54), (6.14, 2.54))
    label(ax, .4, .25, '教學例：選中的 8 個 experts 不變。只交換固定 slots，未選中的 experts 也要交換以保持各組容量。', 10, color=MUTED)
    save(fig, '04_expert_collisions')


def main_results():
    fig, ax = plt.subplots(figsize=(12, 6.3), layout='constrained')
    keys = ['baseline_16_separate_ms', 'baseline_16_stacked_ms', 'selected_group_stacked_ms', 'learned_group_stacked_ms']
    labels = ['16 chips，gate/up 分開', '16 chips，gate/up 串接', '訓練選 g＋串接，原排列', '訓練選 g＋串接＋學習排列']
    colors = [GRAY, '#69a9d2', '#e0b466', GREEN]
    x=np.arange(3)
    for j, (key, title, color) in enumerate(zip(keys, labels, colors)):
        values=[r[key] for r in DATA['main']]
        bars=ax.bar(x+(j-1.5)*.19, values, .18, label=title, color=color)
        ax.bar_label(bars, fmt='%.2f', fontsize=10, padding=3)
    ax.set_xticks(x, ['Granite\n學習配置 g = 4', 'OLMoE\n學習配置 g = 8', 'DeepSeek V2 Lite\n學習配置 g = 8'])
    ax.set_ylabel('模型估算的 routed FFN 服務時間（ms/token；越低越好）')
    ax.set_ylim(0, 32)
    ax.legend(frameon=False, fontsize=10, ncol=2, loc='upper left')
    ax.set_title('相同 routing，逐步改變權重配置與放置', fontsize=16, color=INK, pad=16)
    ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    ax.spines[['right', 'top']].set_visible(False)
    fig.supxlabel('v3 controller-fanout；三折保留 prompt 平均；0.25 μs/chip command、2 μs/host phase 為假設\n僅含 routed FFN projections，無完整模型 tokens/s 或實體硬體量測', fontsize=10)
    save(fig, '05_main_results')


def group_sizes():
    fig, axes=plt.subplots(1, 3, figsize=(12, 4.6), layout='constrained', sharey=True)
    for ax, (name, title) in zip(axes, [('granite','Granite'),('olmoe','OLMoE'),('deepseek','DeepSeek V2 Lite')]):
        values=DATA['models'][name]['fixed_stacked']
        xx=[v['g'] for v in values]; yy=[v['ms'] for v in values]
        ax.plot(xx, yy, '-o', color=BLUE, lw=2)
        for x,y in zip(xx,yy):
            ax.annotate(f'{y:.2f}', (x,y), xytext=(0,7), textcoords='offset points', ha='center', fontsize=9)
        ax.set_xscale('log', base=2); ax.set_xticks(xx, labels=xx)
        ax.set(xlabel='每個 expert 使用的 chip 數 g', title=title, ylim=(0,64))
        ax.grid(alpha=.2); ax.spines[['right','top']].set_visible(False)
    axes[0].set_ylabel('routed FFN 模型時間（ms/token）')
    fig.suptitle('同樣串接 gate/up、保留原始排列：最佳 g 隨模型而異', fontsize=15, color=INK)
    fig.supxlabel('固定配置掃描；三組 prompts 合併平均，尚未使用 learned placement。v3 成本與主結果相同。', fontsize=10)
    save(fig, '06_group_size')


def control_costs():
    fig, axes=plt.subplots(1, 3, figsize=(12, 4.9), layout='constrained')
    for ax, (name, title) in zip(axes, [('granite','Granite'),('olmoe','OLMoE'),('deepseek','DeepSeek V2 Lite')]):
        values=DATA['models'][name]['control_sweep']
        x=[v['chip_control_us'] for v in values]
        ax.plot(x, [v['g16_stacked_ms'] for v in values], '-o', color=BLUE, label='16 chips＋串接')
        ax.plot(x, [v['selected_learned_ms'] for v in values], '-o', color=GREEN, label='訓練選 g＋串接＋學習排列')
        for xx,v in zip(x, values):
            ax.annotate('g='+('/'.join(map(str, v['selected_g']))), (xx,v['selected_learned_ms']),
                        xytext=(0,-15), textcoords='offset points', ha='center', fontsize=8)
        ax.set_xscale('symlog', linthresh=.1)
        ax.set_xticks(x, labels=['0','.1','.25','.5','1','2','5'])
        ax.set(xlabel='每 chip command 的假設成本（μs）', title=title)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=.2); ax.spines[['right','top']].set_visible(False)
        ax.legend(fontsize=8, frameon=False, loc='upper left')
    axes[0].set_ylabel('routed FFN 模型時間（ms/token）')
    fig.suptitle('控制成本變高，最佳 chip group 也可能變小', fontsize=15, color=INK)
    fig.supxlabel('每個成本點都重新用訓練 prompts 選 g，再評估保留 prompt；host 成本固定 2 μs。\n這是參數敏感度分析，未經實體 AiF 校準；保留資料上也可能出現負收益。', fontsize=10)
    save(fig, '07_control_sensitivity')


if __name__ == '__main__':
    for build in (datapath, geometry, short_reads, collisions, main_results, group_sizes, control_costs):
        build()
    print(f'Wrote 7 PNG figures and 7 SVG figures to {OUT}')
