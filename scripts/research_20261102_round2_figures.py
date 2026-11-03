#!/usr/bin/env python3
"""Plot scoped public-routing timing replay, with explicit precision caveat."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/research-2026-11-02/round2'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    original = BASE / 'public-validation/summary.json'
    geometry = BASE / 'geometry-study/heldout_summary.json'
    results = []
    names = {'granite': 'Granite 3.1', 'deepseek': 'DeepSeek V2 Lite', 'qwen3': 'Qwen3'}
    for item in json.loads(original.read_text())['models']:
        results.append(dict(model=names[item['model']],
            baseline_ms=item['means_ms']['legacy'], candidate_ms=item['means_ms']['chosen'],
            gain_pct=item['relative_gain_pct'], ci_low_pct=item['paired_prompt_ci_pct'][0],
            ci_high_pct=item['paired_prompt_ci_pct'][1], prompts=60,
            baseline='Previously strongest one-copy group layout',
            precision='Actual CPU GGUF routes; nominal raw-I8 timing',
            frozen_sha256=item['frozen_plan_sha256']))
    doc = json.loads(geometry.read_text())
    item = next(r for r in doc['results'] if r['split'] == 'external60')
    results.append(dict(model='Granite 4 H Tiny',
        baseline_ms=item['mean_ms']['g16_down'], candidate_ms=item['mean_ms']['chosen'],
        gain_pct=100*item['chosen_gain_over_fixed_g16_down'],
        ci_low_pct=item['paired_prompt_bootstrap_fixed_g16_down'][0],
        ci_high_pct=item['paired_prompt_bootstrap_fixed_g16_down'][1], prompts=60,
        baseline='Same trained GU; fixed g16 down (stronger than fitted phase-group)',
        precision='Actual CPU GGUF routes; nominal raw-I8 timing',
        frozen_sha256=doc['frozen_plan_sha256']))
    target = BASE / 'public_routing_replay.csv'
    with target.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(results[0]))
        writer.writeheader();writer.writerows(results)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                         'axes.spines.top':False,'axes.spines.right':False})
    fig, ax = plt.subplots(figsize=(8.8,4.8))
    x=np.arange(len(results));values=np.array([r['gain_pct'] for r in results])
    lower=values-np.array([r['ci_low_pct'] for r in results])
    upper=np.array([r['ci_high_pct'] for r in results])-values
    ax.bar(x,values,color=['#3476ad']*3+['#d18a35'],width=.57,zorder=3)
    ax.errorbar(x,values,yerr=np.vstack([lower,upper]),fmt='none',color='#222222',
                capsize=5,elinewidth=1.1,zorder=4)
    for i,value in enumerate(values):
        ax.text(i,value+.18,f'{value:.2f}%',ha='center',va='bottom',weight='bold')
    ax.set_xticks(x,[r['model'] for r in results]);ax.set_ylim(0,6.5)
    ax.set_ylabel('Reduction in modeled routed-FFN service (%)')
    ax.grid(axis='y',alpha=.23,zorder=0)
    ax.set_title('Ordinary-read placement: replay on 60 public prompts per model',
                 loc='left',pad=14,weight='bold')
    fig.text(.105,.105,'Paired prompt bootstrap 95% CI; per-model baselines are specified in the CSV.',fontsize=8.4)
    fig.text(.105,.065,'Q4/Q8 CPU routing replayed under a raw-I8 timing assumption. Not device or full-model speed.',fontsize=8.4)
    fig.text(.105,.029,'No novelty claim: overlapping striping and query-aware declustering have direct prior art.',fontsize=8.4,color='#555555')
    fig.subplots_adjust(left=.105,right=.985,top=.85,bottom=.22)
    for ext in ('png','pdf'):
        fig.savefig(BASE/f'public_routing_replay.{ext}',dpi=220)
    receipt=dict(source_sha256=sha(__file__),
                 inputs={str(p.relative_to(ROOT)):sha(p) for p in (original,geometry)},
                 outputs={str(p.relative_to(ROOT)):sha(p) for p in (
                     target,BASE/'public_routing_replay.png',BASE/'public_routing_replay.pdf')},
                 all_models_or_datasets_blind=False,
                 quantization_and_native_format_matched=False,
                 physical_device_or_full_model_timing=False)
    (BASE/'public_routing_replay.receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('Wrote public_routing_replay.{csv,png,pdf,receipt.json}')


if __name__=='__main__':main()
