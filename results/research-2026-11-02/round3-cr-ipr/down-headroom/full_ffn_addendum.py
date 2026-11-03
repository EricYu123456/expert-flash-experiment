#!/usr/bin/env python3
"""Frozen standard resource-partition baseline replay; no new kernel feature."""
from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'full-ffn'
sys.path.insert(0,str(BASE))
import asymmetric_down as old
f=old.f
DEST=HERE/'full-ffn-addendum'
LAYOUTS=['candidate','RR','g8_fixed']
INCUMBENT={'candidate':'original_g4_shared16','RR':'original_g4_shared16','g8_fixed':'original_g2_shared16'}

def freeze():
    DEST.mkdir(exist_ok=True);assert not (DEST/'protocol.frozen.json').exists()
    _,routes,menus,groups=old.checked_inputs()
    new={}
    new['original_g4_shared16']=menus['g4_reused_TRAIN_labels']
    new['original_g2_shared16']=menus['g2_reused_TRAIN_labels']
    with np.load(HERE/'plateau_arrays.npz') as z:
        pages=np.array([z[f'L{l}.dedicated_shared_g4.pages'] for l in range(3)])
    rows=pages*np.array([32]*64+[16])[None,:,None]
    new['dedicated_shared4_learned']=rows.astype(np.int32)
    static=np.zeros((3,65,16),dtype=np.int32);cold=[]
    for l in range(3):
        incidence=np.bincount(routes[l].ravel(),minlength=64);e=int(np.argmin(incidence));cold.append(e)
        assert np.array_equal(pages[l,e],np.array([0]*12+[12]*4))
        ids=[x for x in range(64) if x!=e]
        for i,x in enumerate(ids):static[l,x,4*(i//21):4*(i//21)+4]=384
        static[l,e,12:16]=384;static[l,64,12:16]=384
    new['dedicated_shared4_static']=static
    for name,m in new.items():
        assert m.shape==(3,65,16) and np.all(m.sum(axis=2)==1536)
        C=np.array([512]*64+[1024])[None,:,None]
        pp=(m*C+16383)//16384;reserve=((pp+3)//4).sum(axis=1)
        assert np.all(128+reserve<=192),(name,reserve.max())
    np.savez_compressed(DEST/'menus.frozen.npz',**new)
    paths=[Path(__file__),HERE/'plateau_addendum.py',HERE/'plateau_report.json',HERE/'plateau_arrays.npz',
           HERE/'plateau_addendum.frozen.json',BASE/'asymmetric_down.py',BASE/'policy_addendum.py',BASE/'shaping_addendum.py',
           old.a.SHAPER,f.PIPE,f.CPP,f.ROOT/'scripts/research_20261102_cr_ipr_full_ffn.py',f.TRAIN,
           BASE/'g8-addendum/group_maps.frozen.npz',DEST/'menus.frozen.npz']
    f.dump(DEST/'protocol.frozen.json',dict(status='frozen_before_fullFFN_replay',frozen_utc=f.utc(),
        inputs_sha256={str(p):f.sha(p) for p in paths},GU=LAYOUTS,menus=list(new),incumbents=INCUMBENT,
        screen_indices=old.f.read(old.DEST/'protocol.frozen.json')['screen_indices'],screen_layers=[0,1,2],
        screen_axes=dict(input_copies=[1],priority=['GF','DF']),cold_expert_by_layer=cold,
        fit_scope='Only existing TRAIN512 used for work-objective fitting and rare-expert choice; screen64 and full512 are descriptive TRAIN reuse, not newvalidation.',
        static_control='ID-sorted63experts split into3groups of21; same TRAIN-selected rare expert as learned control. Grouping is coactivation-blind, rare selection is TRAIN-informed.',
        mechanism='Fixed onecopy complete-output-row maps; three four-chip private groups and a fourth shared group hosting one rare routed expert. Same fullprivate C512/C1024 inputs and original GU kernels.',
        kernel='Unchanged asymmetric_down.KernelRecipes/graph_for; strict500ns fixed0123 privateDown. Common GU retains existing deterministic PBpolicy. No private coldreset rescue.',
        capacity='Everymenu onecopy, exactrowcoverage,128GUWL+Downceilpages/4<=192WL/chip. Same existing capacity budget; no relocation duringqueries.',
        selection='For each GU, require all192screen events PBvalid and128BFIFOvalid. Select smallest arithmeticmean three-layer sum over the same64queries among menus andGF/DF, tie by menu thenpriority. Freeze before full512. Run full only if a selectednewmenu improves thatfamily incumbent bestpriority by>1%; run frozen familywinner and originalincumbent for allfamilies atinput1/4 GF/DF.',
        interpretation='Standard resourcepartition/reoptimized generic baseline; no mappingalgorithm novelty or measuredhardwarespeed claim.',
        limits='Pageburst outputFIFO, conditionalcopy/CR/PB/source controls, noheldout, noMACcalibration. Work-objective improvement is not assumed to be fullFFN speedup.'))
    print(json.dumps(dict(status='frozen',menus=list(new),cold=cold)),flush=True)

def checked():
    protocol=f.read(DEST/'protocol.frozen.json')
    for path,h in protocol['inputs_sha256'].items():assert f.sha(path)==h,path
    _,routes,_,groups=old.checked_inputs()
    with np.load(DEST/'menus.frozen.npz') as z:menus={k:z[k].copy() for k in z.files}
    return protocol,routes,menus,groups

def execute(mode):
    protocol,routes,menus,groups=checked();screen=mode=='screen'
    ids=np.array(protocol['screen_indices']) if screen else np.arange(512)
    dest=DEST/('screen' if screen else 'TRAIN512');dest.mkdir(exist_ok=True)
    assert not (dest/'report.json').exists()
    if screen:configs=[(g,m,1,df) for g in LAYOUTS for m in menus for df in [0,1]]
    else:
        selection=f.read(DEST/'selection.frozen.json');assert selection['full512_gate_passed']
        assert selection['screen_report_sha256']==f.sha(DEST/'screen/report.json')
        configs=[(g,m,cp,df) for g in LAYOUTS for m in sorted(set([selection['winners'][g],INCUMBENT[g]])) for cp in [1,4] for df in [0,1]]
    f.dump(dest/'run.frozen.json',dict(protocol_sha256=f.sha(DEST/'protocol.frozen.json'),source_sha256=f.sha(__file__),indices=ids.tolist(),configurations=configs,selection_sha256=None if screen else f.sha(DEST/'selection.frozen.json')))
    recipes=old.KernelRecipes(f.load_pipeline(),old.a.load_shaper())
    build=DEST/'executor';build.mkdir(exist_ok=True);executor=f.Executor(build)
    arrays={};records=[];cache={};unique=0;start=time.monotonic()
    for ci,(GU,menu,copies,df) in enumerate(configs):
        name=f'{GU}.{menu}.input{copies}.{"DF" if df else "GF"}'
        cost=np.full((3,len(ids)),-1,dtype=np.int64);fifo=cost.copy();work=Counter();before=recipes.decisions.copy()
        for layer in range(3):
            for q,idx in enumerate(ids):
                graph=old.graph_for(routes[layer,idx],layer,GU,menus[menu][layer],f.Service(copies,df,0),recipes,groups)
                if graph is None:continue
                key,data=graph.signature();key=(key,df,json.dumps(dict(graph.pipeline_totals),sort_keys=True))
                if key in cache:value,audit=cache[key]
                else:
                    result=executor.run(graph,data);audit=f.audit(graph,result,full=(q in [0,len(ids)-1]));value=result['cost_ns'];cache[key]=(value,audit);unique+=1
                if q==0:
                    result=executor.run(graph,data)
                    f.dump_trace(dest/f'trace.L{layer}.{name}.json',graph,result,audit)
                cost[layer,q]=value;fifo[layer,q]=audit['peak_output_fifo_bytes']
                for k,v in audit['resource_work_ns'].items():work[k]+=v
        valid=cost>=0;arrays[name+'.cost']=cost;arrays[name+'.fifo']=fifo
        records.append(dict(name=name,GU=GU,menu=menu,input_copies=copies,priority='DF' if df else 'GF',
            completed=int(valid.sum()),requested=int(cost.size),mean_ns=float(cost[valid].mean()) if valid.any() else None,
            cost_ns=f.stats(cost[valid]) if valid.any() else None,
            three_layer_sum_ns=f.stats(cost.sum(axis=0)) if valid.all() else None,
            FIFO_peak_bytes=f.stats(fifo[valid]) if valid.any() else None,
            valid_FIFO_capacities={str(k):int(np.count_nonzero(valid&(fifo<=k))) for k in [64,128,256,2048]},
            resource_work_per_valid_query={k:v/int(valid.sum()) for k,v in work.items()} if valid.any() else {},policy_calls=dict(recipes.decisions-before)))
        progress=dict(finished=ci+1,total=len(configs),last=name,unique_graphs=unique,elapsed_seconds=time.monotonic()-start)
        f.dump(dest/'progress.json',progress);print(json.dumps(progress),flush=True)
    np.savez_compressed(dest/'service_arrays.npz',**arrays)
    f.dump(dest/'private_failures.json',recipes.failure_samples)
    f.dump(dest/'report.json',dict(status='PASS',mode=mode,protocol_sha256=f.sha(DEST/'protocol.frozen.json'),source_sha256=f.sha(__file__),indices=ids.tolist(),records=records,executor=executor.receipt,elapsed_seconds=time.monotonic()-start,scope='TRAINreuse conditional causal fullFFN, noheldout or newalgorithm.'))
    if screen:
        winners={};comparisons={};gate=False
        for GU in LAYOUTS:
            good=[r for r in records if r['GU']==GU and r['completed']==192 and r['valid_FIFO_capacities']['128']==192]
            assert good,GU
            best=min(good,key=lambda r:(r['three_layer_sum_ns']['mean'],r['menu'],r['priority']))
            anchor=min([r for r in good if r['menu']==INCUMBENT[GU]],key=lambda r:r['three_layer_sum_ns']['mean'])
            gain=1-best['mean_ns']/anchor['mean_ns'];winners[GU]=best['menu']
            comparisons[GU]=dict(best=best['name'],incumbent=anchor['name'],mean_gain=gain,tail_gain=1-best['cost_ns']['p99']/anchor['cost_ns']['p99'])
            gate|=best['menu'].startswith('dedicated') and gain>0.01
        f.dump(DEST/'selection.frozen.json',dict(frozen_utc=f.utc(),source_sha256=f.sha(__file__),screen_report_sha256=f.sha(dest/'report.json'),screen_arrays_sha256=f.sha(dest/'service_arrays.npz'),winners=winners,comparisons=comparisons,full512_gate_passed=bool(gate)))
        print(json.dumps(dict(status='selection_frozen',winners=winners,comparisons=comparisons,gate=bool(gate))),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['freeze','screen','run']);args=ap.parse_args()
    freeze() if args.mode=='freeze' else execute(args.mode)
