#!/usr/bin/env python3
"""Scientific figures from sealed same-format heldout and capacity controls."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/research-2026-11-02/round2'
REPLAY=BASE/'ordinary-read-compiler/second-heldout-replay-v2'
OUT=REPLAY/'figures'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    paths=[REPLAY/'summary.json',REPLAY/'per_prompt.json',BASE/'capacity-symmetry-control/summary.json',Path(__file__)]
    before={str(p):sha(p) for p in paths}
    report=json.loads(paths[0].read_text());prompts=json.loads(paths[1].read_text());capacity=json.loads(paths[2].read_text())
    assert report['status']=='complete'
    paired=[r for r in report['paired'] if r['baseline']=='cyclic']
    data=[]
    for r in paired:
        campaign='full-campaign-v5-W8-training' if r['input_channel_copies']==1 else 'full-campaign-v6-W8-input4'
        lookup={x['family']:x for x in capacity['records'] if x['campaign']==campaign and x['model']==r['model'] and x['command_ns']==r['command_ns']}
        totals={x['family']:x for x in report['totals'] if x['model']==r['model'] and x['input_channel_copies']==r['input_channel_copies'] and x['command_ns']==r['command_ns']}
        assert lookup['cyclic']['useful_bytes']==lookup['whole_rows']['useful_bytes']
        a,b=totals['cyclic']['complete_ffn_service'],totals['whole_rows']['complete_ffn_service']
        data.append(dict(model=r['model'],copies=r['input_channel_copies'],command_ns=r['command_ns'],
                         mean_gain_pct=100*(1-b['mean_ns']/a['mean_ns']),p99_gain_pct=100*(1-b['p99_ns']/a['p99_ns']),
                         moe_capacity_change_pct=100*(lookup['whole_rows']['new_allocated_bytes']/lookup['cyclic']['new_allocated_bytes']-1),
                         baseline_complete_ffn_mean_ms=a['mean_ns']/1e6,whole_complete_ffn_mean_ms=b['mean_ns']/1e6))
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.right':False,'axes.spines.top':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,2,figsize=(12.8,8.2),layout='constrained')
    colors={1:'#2468a0',4:'#c46027'}
    for model,ax,title in zip(['deepseek','granite4tiny'],axes[0],['DeepSeek V2 Lite','Granite 4 H Tiny']):
        for copies in (1,4):
            rows=[p for p in prompts if p['model']==model and p['command_ns']==250 and p['input_channel_copies']==copies and p['baseline']=='cyclic']
            assert len(rows)==16
            ax.plot(np.arange(1,17),[100*p['complete_ffn_mean_reduction_fraction'] for p in rows],marker='o',ms=4,color=colors[copies],label=f'ONFI input ×{copies}')
        ax.axhline(0,color='.5',lw=.7);ax.axvline(8.5,color='.7',ls=':',lw=1)
        ax.set(xlim=(.5,16.5),xticks=[1,4,8,9,12,16],xlabel='Prompt index: OpenBookQA 1–8, BoolQ 9–16',ylabel='Per-prompt mean reduction (%)',title=title+' · command 0.25 μs')
        ax.legend(frameon=False,fontsize=9);ax.grid(axis='y',alpha=.16)
    ax=axes[1,0];groups=[('deepseek',250),('deepseek',5000),('granite4tiny',250),('granite4tiny',5000)]
    x=np.arange(4)
    for copies,delta in [(1,-.18),(4,.18)]:
        rr=[next(d for d in data if d['model']==m and d['command_ns']==c and d['copies']==copies) for m,c in groups]
        ax.bar(x+delta,[d['mean_gain_pct'] for d in rr],width=.33,color=colors[copies],alpha=.8,label=f'input ×{copies}: mean')
        ax.scatter(x+delta,[d['p99_gain_pct'] for d in rr],color=colors[copies],marker='D',s=28,edgecolors='black',linewidths=.4,label=f'input ×{copies}: P99')
    ax.axhline(0,color='.5',lw=.7);ax.set(xticks=x,xticklabels=['Deep\n0.25 μs','Deep\n5 μs','G4\n0.25 μs','G4\n5 μs'],ylabel='Complete FFN service reduction (%)',title='Pooled descriptive mean and P99')
    ax.legend(frameon=False,fontsize=8,ncol=2);ax.grid(axis='y',alpha=.16)
    ax=axes[1,1]
    for i,d in enumerate(data):
        marker='o' if d['model']=='deepseek' else 's'
        ax.scatter(d['moe_capacity_change_pct'],d['mean_gain_pct'],c=colors[d['copies']],marker=marker,s=55,edgecolors='white',lw=.5,zorder=3)
        if d['mean_gain_pct']>0:
            label=('Deep' if d['model']=='deepseek' else 'G4')+f" c{d['copies']} / {d['command_ns']/1000:g}μs"
            offset=(-118,9) if d['model']=='deepseek' and d['command_ns']==5000 and d['copies']==1 else (7,-12)
            ax.annotate(label,(d['moe_capacity_change_pct'],d['mean_gain_pct']),xytext=offset,textcoords='offset points',fontsize=8)
    ax.axhline(0,color='.5',lw=.7);ax.axvline(0,color='.5',lw=.7)
    ax.set(xlabel='MoE + shared LSB reservation change (%)',ylabel='Complete FFN mean reduction (%)',title='Physical capacity and service tradeoff',xlim=(-6.2,5.4),ylim=(-.15,4.1))
    ax.grid(alpha=.16)
    fig.suptitle('Frozen W8 layouts vs. strong cyclic layouts · 16 heldout prompts × 16 decode positions',fontsize=13)
    for ext in ('png','pdf','svg'):fig.savefig(OUT/('same_format_heldout.'+ext),dpi=180)
    plt.close(fig)
    captions='Uncalibrated ordinary-read service model, not device or full-model time. DeepSeek service adds the same raw0 dense prefix once; footprint covers MoE+shared coefficient LSB blocks only. Layouts were selected on TRAIN and frozen before text acquisition. P99 is descriptive across256 correlated positions. Per-prompt gains have16 units, not256 independent tasks. No confidence intervals or accuracy inference. G4 lowcommand input4 has66 slower token pairs despite positive16prompt means. Same numeric reference preserves logits/routes/tokens. Capacity symmetry is route-blind and applies to all layout families.\n'
    (OUT/'caption.txt').write_text(captions)
    assert all(sha(p)==h for p,h in before.items())
    (OUT/'provenance.json').write_text(json.dumps(dict(inputs_sha256=before,plotted_data=data,caption=captions,outputs_sha256={p.name:sha(p) for p in OUT.iterdir() if p.name!='provenance.json'}),indent=2)+'\n')
    print(json.dumps(dict(outputs=[p.name for p in OUT.iterdir()],data_points=len(data)),indent=2))


if __name__=='__main__':main()
