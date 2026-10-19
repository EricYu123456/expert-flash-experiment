#!/usr/bin/env python3
"""Export static scientific plots of trace-driven AiF service estimates."""
import argparse
import csv
from pathlib import Path
import os

os.environ.setdefault('MPLCONFIGDIR','/home/nfs_client/ericyu/expert_flash_experiment/tmp/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('/home/nfs_client/ericyu/expert_flash_experiment/runs'))
    p.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'results')
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(2,2,figsize=(10.5,7),layout='constrained')
    colors=['#0072B2','#D55E00']
    for model,label,color in zip(('granite','olmoe'),('Granite 3.1 (0.5 MiB / projection)','OLMoE (2 MiB / projection)'),colors):
        records=list(csv.DictReader((a.data/(model+'-aif')/'placement.csv').open()))
        ready=[r for r in records if r['serial_experts']=='False']
        serial=[r for r in records if r['serial_experts']=='True']
        x=[int(r['group_chips']) for r in ready]
        for axis,key,factor in ((axes[0,0],'ffn_ms_per_token',1),
                                (axes[0,1],'useful_bandwidth_fraction',100),
                                (axes[1,0],'active_chip_fraction',100),
                                (axes[1,1],'plane_slot_fill',100)):
            axis.plot(x,[float(r[key])*factor for r in ready],marker='o',color=color,label=label)
        axes[0,0].plot(x,[float(r['ffn_ms_per_token']) for r in serial],linestyle='--',color=color,alpha=.5)
        axes[0,0].fill_between(x,[float(r['ffn_p50_ms']) for r in ready],
                              [float(r['ffn_p95_ms']) for r in ready],color=color,alpha=.12)
    axes[0,0].set_ylabel('Modeled routed FFN service (ms/token)')
    axes[0,0].set_title('Solid: ready experts; dashed: serial experts')
    axes[0,1].set_ylabel('Useful bandwidth / 102.4 GB/s (%)')
    axes[0,1].set_title('Startup + page tails + group collisions')
    axes[1,0].set_ylabel('Active chip-command fraction (%)')
    axes[1,0].set_title('Chip participation does not imply saturation')
    axes[1,1].set_ylabel('Useful bytes / read-wave slots (%)')
    axes[1,1].set_title('Four 16 KiB planes per chip')
    for axis in axes.flat:
        axis.set_xscale('log',base=2)
        axis.set_xticks([1,2,4,8,16],labels=['1','2','4','8','16'])
        axis.set_xlabel('Fixed chips per expert')
        axis.grid(alpha=.2)
        axis.set_ylim(bottom=0)
    for axis in (axes[0,1],axes[1,0],axes[1,1]):
        axis.set_ylim(0,105)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=2,frameon=False)
    fig.suptitle('Real llama.cpp routes; INT8 AiF flash-service model\n96 decode inputs / model, batch 1; no hardware MAC-lane measurement',fontsize=12)
    for extension in ('png','pdf'):
        fig.savefig(a.output/('aif_moe_results.'+extension),dpi=180)
    print(a.output/'aif_moe_results.png')


if __name__=='__main__':
    main()
