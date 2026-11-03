#!/usr/bin/env python3
"""Static report figures for the plane-wave study; values come from artifacts."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from run_corpus_traces import ROOT

MODELS = ('granite', 'olmoe', 'deepseek', 'qwen3', 'mixtral')
NAMES = ('Granite', 'OLMoE', 'DeepSeek', 'Qwen3', 'Mixtral')
COLORS = ('#8c99a2', '#247e85', '#d18a35')


def save(fig, path):
    for suffix in ('png', 'svg', 'pdf'):
        fig.savefig(path.with_suffix('.' + suffix), dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=ROOT/'results/plane-wave-v1')
    args = parser.parse_args()
    output = args.results/'figures'
    output.mkdir(exist_ok=True)
    all_candidates = json.loads((args.results/'heldout_candidates.json').read_text())
    summary = json.loads((args.results/'summary.json').read_text())
    rows = [r for r in all_candidates if r['timing'] == 'ideal_overlap' and r['window'] == 128 and r['group_chips'] == 16]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    x = np.arange(5)
    for i, width in enumerate((4, 2, 1)):
        values = [next(r for r in rows if r['model'] == name and r['plane_width'] == width) for name in MODELS]
        base = [next(r['mean_ms_per_token'] for r in rows if r['model'] == name and r['plane_width'] == 4) for name in MODELS]
        latency = [100*v['mean_ms_per_token']/b for v,b in zip(values,base)]
        fill = [100*v['plane_slot_fill'] for v in values]
        for ax, y in zip(axes, (latency, fill)):
            ax.bar(x+(i-1)*.24, y, width=.23, color=COLORS[i], label=f'{width} planes / stream')
    axes[0].axhline(100, color='#555555', linewidth=.8, linestyle='--')
    axes[0].set_ylabel('Latency (% of fixed-g=16, width=4)')
    axes[0].set_title('Fixed g=16: relative model latency')
    axes[1].set_ylabel('Useful bytes / allocated wave capacity (%)')
    axes[1].set_title('Fixed g=16: useful plane-slot fill')
    axes[1].set_ylim(0, 110)
    for ax in axes:
        ax.set_xticks(x, NAMES)
        ax.grid(axis='y', alpha=.18)
        ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)
    axes[0].legend(fontsize=8)
    fig.suptitle('100 held-out prompts/model, 128 decode inputs; conditional ideal-overlap model', fontsize=11)
    fig.tight_layout()
    save(fig, output/'fixed_g16')

    rows = [r for r in summary if r['timing'] == 'ideal_overlap' and r['window'] == 128]
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, policy in enumerate(('width2_trainselected','width1_trainselected')):
        values = [next(r for r in rows if r['model']==name and r['policy']==policy) for name in MODELS]
        ax.bar(x+(i-.5)*.3, [-100*r['improvement_vs_old_frozen'] for r in values], width=.29,
               color=COLORS[i+1], label=f'{2 if i==0 else 1} planes; g selected on training')
    ax.set_xticks(x, NAMES)
    ax.set_ylabel('Latency increase vs previous best (%)')
    ax.set_title('Allow g=1/2/4/8/16: narrower planes do not beat previous best')
    ax.grid(axis='y',alpha=.18)
    ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False)
    ax.legend(fontsize=9)
    fig.tight_layout()
    save(fig, output/'best_group_comparison')

    synthetic = json.loads((args.results/'synthetic/results.json').read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, k in zip(axes, (2, 8)):
        for policy, color in zip(('pair','single'),COLORS[1:]):
            values = [r for r in synthetic if r['selected_experts']==k and r['policy']==policy]
            ax.plot([r['shard_kib'] for r in values], [100*r['ideal_reduction_vs_baseline'] for r in values],
                    marker='o', markersize=3, color=color, label=policy)
        ax.axhline(0,color='#777777',linewidth=.8)
        ax.set_xscale('log',base=2)
        ax.set_xticks([16,32,64,128,256,1024],['16','32','64','128','256','1024'])
        ax.set_xlabel('Per-expert, per-chip shard (KiB)')
        ax.set_title(f'{k} selected out of 32 stored experts')
        ax.grid(alpha=.18)
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Flash latency reduction vs width=4 (%)')
    axes[0].legend()
    fig.suptitle('Exact uniform routing, fixed equal-capacity bins; ideal flash/ECC bound only',fontsize=11)
    fig.tight_layout()
    save(fig, output/'synthetic_size_sweep')


if __name__ == '__main__':
    main()
