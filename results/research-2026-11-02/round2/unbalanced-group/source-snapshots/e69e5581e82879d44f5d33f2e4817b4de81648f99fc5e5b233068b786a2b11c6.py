#!/usr/bin/env python3
"""Strengthen ordinary-read disjoint-group baselines by releasing equal counts.

Exploratory training-only audit. Same g within each candidate, unchanged complete
INT8 matrices/pages, no replica or new hardware. Actual chip capacity is checked
on a whole-model virtual address plan, separately from the old mean payload cap.
"""
import argparse
import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from materialize_plan import WaveAllocator
from research_20261102_projection_study import equal_rows, evaluate, scalar_cost
from run_corpus_traces import ROOT, sha256

DEST=ROOT/'results/research-2026-11-02/round2/unbalanced-group'
PHYSICAL_CAP=20*(1<<30)


def write(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def library():
    source=Path(__file__).with_suffix('.cpp')
    target=DEST/('single-relabel-'+sha256(source)+'.so')
    if not target.exists():
        temporary=target.with_name(target.name+'.'+str(os.getpid())+'.tmp')
        subprocess.run(['g++','-std=c++17','-O3','-march=native','-shared','-fPIC',str(source),'-o',str(temporary)],check=True)
        os.replace(temporary,target)
    lib=ctypes.CDLL(str(target))
    p32=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')
    p64=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS')
    lib.optimize_unbalanced_group.argtypes=[p32,p32,p32,p64]+[ctypes.c_int]*8+[ctypes.c_uint64,ctypes.c_double,p64]
    lib.optimize_unbalanced_group.restype=ctypes.c_double
    return lib


def inputs(name):
    if name=='granite4tiny':
        path=ROOT/'results/research-2026-11-02/round2/geometry-study/granite4tiny.frozen.json'
        doc=json.loads(path.read_text());route_path=Path(doc['training_routes'])
        phases=[]
        for p in doc['phases']:
            candidates=[c for c in p['candidates'] if c['group_equivalent']]
            phases.append(dict(layer=p['layer'],name=p['name'],total_rows=p['total_rows'],columns=p['columns'],common=p['common'],
                seeds=[dict(label=c['label'],rows=c['rows']) for c in candidates],
                rows=p['group_rows'],main_rows=p['rows']))
    else:
        path=ROOT/'results/research-2026-11-02/projection-study'/(name+'.frozen_plan.json')
        doc=json.loads(path.read_text());route_path=Path(doc['train_path'])
        legacy_path=Path(doc['legacy_path']);assert sha256(legacy_path)==doc['legacy_sha256']
        legacy=json.loads(legacy_path.read_text());phases=[]
        for p in doc['phases']:
            seeds=[]
            for c in legacy['training_candidates']:
                if c['family']=='group_only_learned' and c['plane_width']==4:
                    rows=equal_rows(p['total_rows'],doc['experts'],c['group_chips'],legacy['mappings'][c['key']][str(p['layer'])])
                    seeds.append(dict(label=c['key'],rows=rows.tolist()))
            scores={c['label']:c['mean_ns'] for c in p['candidates']}
            original=min(seeds,key=lambda s:scores[s['label']])
            phases.append(dict(layer=p['layer'],name=p['name'],total_rows=p['total_rows'],columns=p['columns'],common=p['common'],
                seeds=seeds,rows=original['rows'],main_rows=p['chosen_rows']))
    expected=doc.get('training_routes_sha256',doc.get('train_sha256'))
    assert sha256(route_path)==expected
    with np.load(route_path) as f:
        routes=f['routes'][:32,:64].copy();ids=f['prompt_ids'][:32].tolist()
        layers=f['layers'].tolist()
    assert len(ids)==32 and all(p.startswith('train_') for p in ids)
    return path,doc,route_path,routes,ids,layers,phases


def inventory(phases,experts):
    alloc=[WaveAllocator() for _ in range(16)]
    for p in phases:
        rows=np.asarray(p['rows'],dtype=np.int64)
        assert rows.shape==(experts,16) and np.all(rows.sum(axis=1)==p['total_rows'])
        for e in range(experts):
            for c in range(16):
                if rows[e,c]:alloc[c].allocate(int(rows[e,c])*p['columns'])
    chips=[a.summary(c) for c,a in enumerate(alloc)]
    total_useful=sum(c['useful_int8_bytes'] for c in chips)
    return dict(chips=chips,total_useful_bytes=total_useful,
        total_physical_pages=sum(c['physical_pages'] for c in chips),
        total_reserved_lsb_bytes=sum(c['reserved_lsb_bytes'] for c in chips),
        old_mean_useful_budget_per_chip_bytes=total_useful/16,
        max_useful_bytes_per_chip=max(c['useful_int8_bytes'] for c in chips),
        max_reserved_lsb_bytes_per_chip=max(c['reserved_lsb_bytes'] for c in chips),
        actual_lsb_capacity_per_chip_bytes=PHYSICAL_CAP,
        actual_lsb_capacity_assumption='20 GiB/chip scenario supplied by task; not a vendor measured usable-capacity disclosure. Only routed FFN weights placed; remaining capacity must cover unrelated weights, ECC and spare.',
        within_actual_lsb_cap=all(c['reserved_lsb_bytes']<=PHYSICAL_CAP for c in chips),
        within_old_mean_useful_cap=all(c['useful_int8_bytes']<=total_useful/16 for c in chips))


def mean(selected,weights,rows,p):
    result=evaluate(selected,rows,p['columns'],p['common'])
    return float(result['model_ns']@weights/weights.sum())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',choices=['granite4tiny','granite','qwen3'],required=True)
    ap.add_argument('--all-layers',action='store_true');ap.add_argument('--iterations',type=int,default=6000)
    ap.add_argument('--sweeps',type=int,default=10);ap.add_argument('--temperatures',nargs='+',type=float,default=[0.,1000.,5000.])
    args=ap.parse_args();DEST.mkdir(parents=True,exist_ok=True);lib=library()
    source_path,doc,route_path,routes,ids,layers,phases=inputs(args.model)
    source_sha=sha256(source_path);baseline_inventory=inventory(phases,doc['experts'])
    limit=None if args.all_layers else (3 if args.model=='granite4tiny' else 2)
    selected_layers=layers if limit is None else layers[:limit]
    output=[];updated=[dict(p) for p in phases]
    for pi,p in enumerate(phases):
        if p['layer'] not in selected_layers:continue
        li=layers.index(p['layer'])
        selected,weights=np.unique(np.sort(routes[:,:,li].reshape(-1,doc['active']),axis=1),axis=0,return_counts=True)
        selected=np.ascontiguousarray(selected,dtype=np.int32);weights=np.ascontiguousarray(weights,dtype=np.int64)
        seed_by_g={}
        for seed in p['seeds']:
            rows=np.asarray(seed['rows'],dtype=np.int32);g=int(np.count_nonzero(rows[0]))
            value=mean(selected,weights,rows,p)
            if g not in seed_by_g or value<seed_by_g[g]['mean_ns']:
                seed_by_g[g]=dict(rows=rows,mean_ns=value,label=seed['label'])
        candidates=[]
        for g,seed in sorted(seed_by_g.items()):
            templates=np.ascontiguousarray(equal_rows(p['total_rows'],16//g,g),dtype=np.int32)
            lookup={tuple(r):i for i,r in enumerate(templates)}
            initial=np.ascontiguousarray([lookup[tuple(r)] for r in seed['rows']],dtype=np.int32)
            best=initial.copy();best_mean=seed['mean_ns'];search=[]
            for ti,temp in enumerate(args.temperatures):
                if g==16:break
                labels=initial.copy();stats=np.zeros(3,dtype=np.int64)
                cost=lib.optimize_unbalanced_group(labels,templates,selected,weights,doc['experts'],len(templates),len(selected),doc['active'],p['columns'],int(p['common']),args.iterations,args.sweeps,20261102+pi*1009+g*31+ti,temp,stats)
                rows=templates[labels];reference=mean(selected,weights,rows,p)
                assert abs(reference-cost)<1e-6,(reference,cost)
                assert np.all(rows.sum(axis=1)==p['total_rows'])
                # Relabeling changes ownership among groups, never matrix geometry/pages.
                np.testing.assert_array_equal(np.sort(((rows*p['columns']+16383)//16384).sum(axis=1)),np.sort(((seed['rows']*p['columns']+16383)//16384).sum(axis=1)))
                for ei in sorted({0,len(selected)//2,len(selected)-1}):
                    vector=evaluate(selected[ei:ei+1],rows,p['columns'],p['common'])
                    for k,v in scalar_cost(selected[ei],rows,p['columns'],p['common']).items():assert int(vector[k][0])==v
                candidate_phases=[dict(q) for q in updated];candidate_phases[pi]['rows']=rows.tolist()
                cap=inventory(candidate_phases,doc['experts'])
                search.append(dict(temperature_ns=temp,mean_ns=reference,valid=int(stats[0]),anneal_accepted=int(stats[1]),greedy_moves=int(stats[2]),within_actual_cap=cap['within_actual_lsb_cap'],max_reserved_lsb_bytes_per_chip=cap['max_reserved_lsb_bytes_per_chip']))
                if cap['within_actual_lsb_cap'] and reference<best_mean:best_mean=reference;best=labels.copy()
            rows=templates[best]
            candidates.append(dict(label='unbalanced_group_g'+str(g),g=g,balanced_seed_label=seed['label'],balanced_seed_ns=seed['mean_ns'],mean_ns=best_mean,rows=rows.tolist(),group_counts=np.bincount(best,minlength=16//g).tolist(),search=search))
        winner=min(candidates,key=lambda c:c['mean_ns']);balanced=min(candidates,key=lambda c:c['balanced_seed_ns'])
        updated[pi]['rows']=winner['rows']
        phase=dict(layer=p['layer'],name=p['name'],total_rows=p['total_rows'],columns=p['columns'],common=p['common'],candidates=candidates,
            group_ns=balanced['balanced_seed_ns'],group_rows=seed_by_g[balanced['g']]['rows'].tolist(),
            chosen_ns=winner['mean_ns'],chosen_rows=winner['rows'],chosen_g=winner['g'],
            original_main_rows=p['main_rows'],original_main_ns=mean(selected,weights,np.asarray(p['main_rows']),p))
        output.append(phase)
        print(args.model,p['layer'],p['name'],'balanced',round(phase['group_ns'],3),'unbalanced',round(phase['chosen_ns'],3),'g',winner['g'],'gain',round(100*(1-phase['chosen_ns']/phase['group_ns']),5),'counts',winner['group_counts'],flush=True)
        write(DEST/(args.model+'.'+('full' if args.all_layers else 'screen')+'.partial.json'),output)
    final_inventory=inventory(updated,doc['experts']);assert final_inventory['within_actual_lsb_cap']
    assert final_inventory['total_useful_bytes']==baseline_inventory['total_useful_bytes']
    assert final_inventory['total_physical_pages']==baseline_inventory['total_physical_pages']
    assert sha256(source_path)==source_sha
    result=dict(model=args.model,status='training-only exploratory baseline strengthening; main frozen plan unchanged',
        phases=output,train_ids=ids,steps=64,training_routes=str(route_path),training_routes_sha256=sha256(route_path),
        selected_training_slice='[:32,:64] of existing training cache; no heldout opened',
        source_plan=str(source_path),source_plan_sha256=source_sha,frozen_utc=datetime.now(timezone.utc).isoformat(),
        experts=doc['experts'],active=doc['active'],layers=selected_layers,
        training_totals_ns={k:sum(p[k] for p in output) for k in ['group_ns','chosen_ns','original_main_ns']},
        baseline_whole_model_inventory=baseline_inventory,updated_whole_model_inventory=final_inventory,
        whole_model_rows=[{k:p[k] for k in ['layer','name','total_rows','columns','common','rows']} for p in updated],
        actual_cap_check='After each anneal+descent candidate, evaluate exact WaveAllocator whole-model reserved bytes with all other currently committed phase layouts. Discard candidates above actual chip capacity. Search transients are not physical deployments.',
        seed_provenance='Frozen legacy/group candidates may have been fitted on more training routes; all new relabel decisions and phase comparisons use only this 32x64 training slice.',
        limitations=['Only routed FFN W8/FP32 service model; not measured full-model speed.','Same total useful bytes and physical page count does not imply same per-chip inventory or reserved LSB bytes.','No new placement method claim: frequency/coactivation-aware unconstrained-count grouping is a known strong baseline.','The original main plan remains frozen; this baseline was strengthened after earlier aggregate heldout results were visible.'],
        arguments=vars(args),source_sha256={p.name:sha256(p) for p in [Path(__file__),Path(__file__).with_suffix('.cpp')]})
    target=DEST/(args.model+'.'+('full' if args.all_layers else 'screen')+'.frozen.json');write(target,result)
    print(str(target),json.dumps(result['training_totals_ns']),flush=True)


if __name__=='__main__':main()
