#!/usr/bin/env python3
"""Export layout results and control-latency sensitivity as scientific figures."""
import argparse
import csv
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/home/nfs_client/ericyu/expert_flash_experiment/tmp/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def records(path):
    with path.open() as file:
        return list(csv.DictReader(file))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('/home/nfs_client/ericyu/expert_flash_experiment/runs'))
    p.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'results')
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    models=[(name,label) for name,label in [('granite','Granite 3.1'),('olmoe','OLMoE'),('deepseek','DeepSeek V2 Lite')]
            if ((a.data/(name+'-aif')/'learned_overhead_fanout/summary.json').exists()
                or (a.data/(name+'-aif')/'learned_overhead/summary.json').exists())]
    fig,ax=plt.subplots(figsize=(10,5.2),layout='constrained')
    series=[[],[],[],[]]; selected_groups=[]; summary=[]
    for name,label in models:
        root=a.data/(name+'-aif')
        traffic='overhead_fanout' if (root/'overhead_fanout').exists() else 'overhead'
        placement='learned_overhead_fanout' if (root/'learned_overhead_fanout').exists() else 'learned_overhead'
        rows=records(root/traffic/'overhead_placement.csv')
        separate=next(r for r in rows if r['group_chips']=='16' and r['stack_gate_up']=='False')
        stacked=next(r for r in rows if r['group_chips']=='16' and r['stack_gate_up']=='True')
        learned=json.loads((root/placement/'summary.json').read_text())
        modulo=next(r for r in learned if r['policy']=='modulo')
        trained=next(r for r in learned if r['policy']=='learned')
        values=[float(separate['ffn_ms_per_token']),float(stacked['ffn_ms_per_token']),
                modulo['mean_ffn_ms_per_token'],trained['mean_ffn_ms_per_token']]
        for dst,value in zip(series,values):
            dst.append(value)
        gs=sorted({r['group_chips'] for r in trained['folds']})
        selected_groups.append('/'.join(map(str,gs)))
        summary.append(dict(model=name,baseline_16_separate_ms=values[0],baseline_16_stacked_ms=values[1],
                            selected_group_stacked_ms=values[2],learned_group_stacked_ms=values[3],
                            selected_groups=gs,speedup_vs_16_separate=values[0]/values[3],
                            improvement_vs_16_stacked=1-values[3]/values[1],
                            improvement_vs_group_stacked=1-values[3]/values[2]))
    x=np.arange(len(models)); width=.18
    labels=['16 chips, separate gate/up','16 chips, stacked gate/up',
            'Training-selected group + stacking','Selected group + stacking + learned placement']
    colors=['#999999','#56B4E9','#E69F00','#009E73']
    for j,(ys,label,color) in enumerate(zip(series,labels,colors)):
        bars=ax.bar(x+(j-1.5)*width,ys,width,label=label,color=color)
        ax.bar_label(bars,fmt='%.2f',fontsize=8,padding=2)
    ax.set_xticks(x,labels=[label+'\n'+g+' chips/expert selected' for (_,label),g in zip(models,selected_groups)])
    ax.set_ylabel('Modeled routed FFN service (ms/token)')
    ax.set_title('Fewer chip contexts + longer read runs + fewer expert collisions\nReal routes; group/layout selection evaluated on held-out prompts')
    ax.grid(axis='y',alpha=.2); ax.set_axisbelow(True)
    ax.set_ylim(0,max(max(values) for values in series)*1.28)
    ax.legend(loc='upper right',fontsize=8,frameon=False)
    fig.supxlabel('Illustrative control cost: 0.25 us / chip-matrix command + 2 us / host phase; vector traffic included',fontsize=9)
    for extension in ('png','pdf'):
        fig.savefig(a.output/('optimized_layouts.'+extension),dpi=180,bbox_inches='tight')
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (a.output/'summary.csv').open('w',newline='') as file:
        w=csv.DictWriter(file,fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
    fig,axes=plt.subplots(1,len(models),figsize=(5*len(models),4.7),layout='constrained',squeeze=False)
    costs=[0,100,250,500,1000,2000,5000]
    for axis,(name,label) in zip(axes[0],models):
        root=a.data/(name+'-aif')
        traffic='overhead_fanout' if (root/'overhead_fanout').exists() else 'overhead'
        placement='learned_overhead_fanout' if (root/'learned_overhead_fanout').exists() else 'learned_overhead'
        fixed=records(root/traffic/'control_sensitivity.csv')
        heldout=records(root/placement/'heldout_control_sensitivity.csv')
        for stacked,color,tag in ((False,'#999999','16 chips, separate'),(True,'#56B4E9','16 chips, stacked')):
            values=[float(next(r for r in fixed if r['group_chips']=='16' and r['stack_gate_up']==str(stacked)
                               and int(r['per_chip_control_ns'])==cost and r['per_host_command_ns']=='2000')['ffn_ms_per_token']) for cost in costs]
            axis.plot(np.array(costs)/1000,values,marker='o',color=color,label=tag)
        selected=[]; groups=[]
        for cost in costs:
            rows=[r for r in heldout if r['policy']=='learned' and int(r['per_chip_control_ns'])==cost
                  and r['per_host_command_ns']=='2000']
            selected.append(sum(float(r['ffn_ms_per_token']) for r in rows)/len(rows))
            groups.append('/'.join(map(str,sorted({int(r['group_chips']) for r in rows}))))
        axis.plot(np.array(costs)/1000,selected,marker='o',color='#009E73',label='Training-selected learned layout')
        for c,value,g in zip(costs,selected,groups):
            axis.annotate('g='+g,(c/1000,value),xytext=(0,-14),textcoords='offset points',fontsize=7,ha='center')
        axis.set_xscale('symlog',linthresh=.1)
        axis.set_xticks(np.array(costs)/1000,labels=['0','.1','.25','.5','1','2','5'])
        axis.set_xlabel('Control time per chip-matrix command (us)')
        axis.set_ylabel('Modeled routed FFN service (ms/token)')
        axis.set_title(label); axis.grid(alpha=.2)
        axis.legend(fontsize=8,frameon=False)
    fig.suptitle('The best group size depends on command overhead\nParameters are sensitivity assumptions; these are not physical hardware timings',fontsize=11)
    for extension in ('png','pdf'):
        fig.savefig(a.output/('control_tradeoff.'+extension),dpi=180,bbox_inches='tight')
    print(a.output/'optimized_layouts.png')


if __name__=='__main__':
    main()
