#!/usr/bin/env python3
"""Standalone scientific figure, with cost/provenance limits in the caption."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/research-2026-11-02/round2/qwen38-placement'


def main():
    train=json.loads((BASE/'native-full48/screen.json').read_text())
    external=json.loads((BASE/'external-doc-grounded/summary.json').read_text())
    cases=['group_g16','group_best','overlap_best','whole_row','rect_min_io4']
    names=['Fixed g16','Train-selected\ngroups','Overlapping\nstripes','Whole rows','Rectangles\n(ABI condition)']
    t={r['case']:r for r in train['totals']};e={r['case']:r for r in external['totals']}
    fig,ax=plt.subplots(1,2,figsize=(12,4.9),layout='constrained')
    x=np.arange(len(cases));w=.36
    for a,metric,title in zip(ax,('mean','p99'),('Mean','Empirical token P99')):
        tv=[t[c][metric+'_ns']/1e6 for c in cases]
        ev=[e[c]['token_weighted_mean_ns' if metric=='mean' else 'p99_ns']/1e6 for c in cases]
        b0=a.bar(x-w/2,tv,w,label='Train: one repeated C++ task',color='#8097a7')
        b1=a.bar(x+w/2,ev,w,label='External: one document-QA session',color='#226f70')
        for bars in (b0,b1):bars[-1].set_hatch('///')
        a.bar_label(b0,fmt='%.2f',fontsize=8,padding=3);a.bar_label(b1,fmt='%.2f',fontsize=8,padding=3)
        a.set_xticks(x,names,fontsize=9);a.set_ylim(0,104);a.set_ylabel('Sum of 48 FFN services (ms)')
        a.set_title(title);a.spines[['top','right']].set_visible(False);a.grid(axis='y',alpha=.2);a.set_axisbelow(True)
    ax[0].legend(frameon=False,fontsize=8,loc='lower left')
    fig.suptitle('Qwen3.8-Flash-Next routes: frozen physical-layout controls',fontsize=13)
    fig.supxlabel('Nominal I8/A8 software cost on NVFP4 routes; shared FFN included; no CR-read.\n'
                  '4B valid-row output assumed; 1 μs reconstruction, 0 activation/combine. Not full-model or AiF measurements.',fontsize=9)
    out=BASE/'figures';out.mkdir(exist_ok=True)
    for ext in ('png','pdf','svg'):fig.savefig(out/('frozen48_controls.'+ext),dpi=180)
    plt.close(fig)
    note=dict(figure='frozen48_controls',train_tokens=12784,external_tokens=5519,external_sessions=1,external_work_requests=41,
              claim='Existing geometry controls, not a new-method result. Rectangular native numeric ABI remains unverified.',
              source_statistics=[str(BASE/'native-full48/screen.json'),str(BASE/'external-doc-grounded/summary.json')])
    (out/'caption.json').write_text(json.dumps(note,indent=2)+'\n')
    sensitivity=BASE/'parameter-sensitivity/summary.json'
    if sensitivity.exists():
        d=json.loads(sensitivity.read_text());fig,a=plt.subplots(figsize=(7.8,4.8),layout='constrained')
        for c,label,color,style in zip(cases,names,('#a5a5a5','#7471aa','#bf7c32','#226f70','#3e8ca5'),(':','--','-','-','--')):
            rows=sorted((r for r in d['totals'] if r['case']==c and r['label'].startswith('command_')),key=lambda r:r['command_ns'])
            a.plot([r['command_ns']/1000 for r in rows],[r['mean_ns']/1e6 for r in rows],marker='o',markersize=3,label=label.replace('\n',' '),color=color,linestyle=style)
        a.set_xlabel('Assumed serialized controller cost per native matrix command (μs)')
        a.set_ylabel('48-layer FFN mean service (ms)');a.spines[['top','right']].set_visible(False);a.grid(alpha=.2)
        a.set_title('The winning geometry depends on controller cost')
        a.legend(frameon=False,fontsize=8,loc='upper left')
        fig.supxlabel('Same frozen layouts; 256 fixed external tokens; page-fringe output paid.\nNo refit, hardware calibration, end-to-end timing, or new-method claim.',fontsize=9)
        for ext in ('png','pdf','svg'):fig.savefig(out/('controller_sensitivity.'+ext),dpi=180)
        plt.close(fig)


if __name__=='__main__':main()
