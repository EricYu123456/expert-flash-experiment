#!/usr/bin/env python3
"""Stronger geometry controls and joined-load choice for paired shared rows.

Keeps the first shared-pipeline study/source untouched.  The copy policy solves
each fixed chip pair's three legal complete-row assignments.  It optimizes a
necessary occupancy bound, then separately replays actual native dependencies.
"""
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import json
from pathlib import Path

import numpy as np

from research_20261102_shared_pipeline import ROOT,BASE,SOURCE,EXPECTED,JOINT,sha,build_jobs

DEST=BASE/'shared-pipeline-more'
DEPTH=BASE/'pipeline-joint-depth/screen.json'


def row_work(rows,columns):
    rows=np.asarray(rows,dtype=np.int64)
    return ((rows*columns+65535)//65536)*28000+np.where(rows>0,250+(columns+1)//2+rows*2,0)


def profiles(layer,name,source,joint,depth):
    ps={p['name']:p for p in source['phases'] if p['layer']==layer}
    gu=np.asarray(ps['gate_up']['rows']);down=np.asarray(ps['down']['rows'])
    if name=='joint':
        labels=np.asarray(next(x for x in joint['results'] if x['layer']==layer)['joint_best']['labels'])
        down=np.where(np.arange(16)[None,:]//4==labels[:,None],0,128)
    elif name=='opposite':
        down=np.where(gu>0,0,192)
    elif name=='wave':
        gu=np.asarray(next(x for x in ps['gate_up']['candidates'] if x['label']=='wave_rotations')['rows'])
    elif name.startswith('depth_'):
        best=next(x for x in depth['results'] if x['layer']==layer)[name[6:]]
        gu=np.asarray(best['gu_rows']);down=np.asarray(best['down_rows'])
    else:assert name=='source'
    assert np.all(gu.sum(axis=1)==1024) and np.all(down.sum(axis=1)==1536)
    return gu,down


def assign_shared(occupied,replica):
    sr=np.zeros(16,dtype=np.int64);ranges=[];choices=[]
    for p in range(8):
        candidates=[]
        for a in ((0,1,2) if replica else (1,)):
            rows=np.zeros(16,dtype=np.int64);rows[p]=128*a;rows[p+8]=128*(2-a)
            added=row_work(rows,1536)
            candidates.append((max(occupied[p]+added[p],occupied[p+8]+added[p+8]),
                               added[p]+added[p+8],abs(a-1),a))
        a=min(candidates)[-1];choices.append(a)
        first=256*p
        if a==1:ranges.extend([(p,first,128),(p+8,first+128,128)])
        else:ranges.append((p if a==2 else p+8,first,256))
        sr[p]=128*a;sr[p+8]=128*(2-a)
    intervals=sorted((start,start+size) for _,start,size in ranges)
    assert intervals[0][0]==0 and intervals[-1][1]==2048
    assert all(b==c for (_,b),(c,_) in zip(intervals,intervals[1:]))
    return sr,ranges,choices


def worker(task):
    layer,name,replica,side=task
    source=json.loads(SOURCE.read_text());joint=json.loads(JOINT.read_text());depth=json.loads(DEPTH.read_text())
    with np.load(source['training_routes']) as f:
        selected=f['routes'][:32,:64,f['layers'].tolist().index(layer)].reshape(-1,6)
    gu,down=profiles(layer,name,source,joint,depth)
    sd=np.full(16,64,dtype=np.int64);sd[side:side+8]=128
    rw=row_work(gu,1536)+row_work(down,512)
    values=[];bounds=[];choices=[];trace=None
    for i,route in enumerate(selected):
        occupied=rw[route].sum(axis=0)+row_work(sd,1024)
        sg,ranges,decision=assign_shared(occupied,replica)
        items=[dict(gu_rows=gu[e].tolist(),down_rows=down[e].tolist(),down_columns=512) for e in route]
        items.append(dict(gu_rows=sg.tolist(),down_rows=sd.tolist(),down_columns=1024))
        g=build_jobs(items);r=g.run(False,trace=i==0)
        values.append(r['makespan_ns']);bounds.append(2000+int((occupied+row_work(sg,1536)).max()));choices.append(decision)
        assert values[-1]>=bounds[-1]
        if trace is None:
            trace=dict(layer=layer,mapping=name,replica=replica,shared_down_high_half=side,
                selected=route.tolist(),shared_gu_ranges=ranges,shared_gu_choices=decision,items=items,**r)
    label=f'L{layer}.{name}.rep{int(replica)}.sd{side}'
    np.savez_compressed(DEST/(label+'.npz'),cost=np.asarray(values),native_chip_bound=np.asarray(bounds),choices=np.asarray(choices))
    (DEST/(label+'.trace.json')).write_text(json.dumps(trace,indent=2)+'\n')
    return dict(layer=layer,mapping=name,replica=replica,side=side,label=label,
        mean_ns=float(np.mean(values)),p99_ns=float(np.percentile(values,99)),mean_bound_ns=float(np.mean(bounds)),
        npz_sha256=sha(DEST/(label+'.npz')),trace_sha256=sha(DEST/(label+'.trace.json')))


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    assert sha(SOURCE)==EXPECTED
    source=json.loads(SOURCE.read_text());assert sha(source['training_routes'])==source['training_routes_sha256']
    with np.load(source['training_routes']) as f:assert all(x.startswith('train_') for x in f['prompt_ids'].tolist())
    names=('source','joint','opposite','wave','depth_barrier','depth_joined')
    tasks=[(layer,name,replica,side) for layer in source['layers'][:3] for name in names for replica in (False,True) for side in (0,8)]
    records=[]
    with ProcessPoolExecutor(max_workers=8) as pool:
        for future in as_completed([pool.submit(worker,t) for t in tasks]):
            r=future.result();records.append(r)
            print(r['label'],r['mean_ns']/1e6,r['p99_ns']/1e6,flush=True)
            (DEST/'partial.json').write_text(json.dumps(records,indent=2)+'\n')
    totals=[]
    for name in names:
        for replica in (False,True):
            for side in (0,8):
                records_here=[r for r in records if r['mapping']==name and r['replica']==replica and r['side']==side]
                loaded=[np.load(DEST/(r['label']+'.npz')) for r in records_here]
                x=np.stack([r['cost'] for r in loaded]).sum(axis=0);b=np.stack([r['native_chip_bound'] for r in loaded]).sum(axis=0)
                totals.append(dict(mapping=name,replica=replica,side=side,mean_ns=float(x.mean()),
                    p99_ns=float(np.percentile(x,99)),mean_native_bound_ns=float(b.mean())))
    out=dict(status='complete train-only3layer stronger shared geometry screen',created_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sha(__file__),imported_sources_sha256={n:sha(Path(__file__).with_name(n)) for n in (
            'research_20261102_shared_pipeline.py','research_20261102_granite4_pipeline.py')},
        inputs_sha256={str(p):sha(p) for p in (SOURCE,JOINT,DEPTH)},training_routes_sha256=sha(source['training_routes']),
        protocol='All2048training route vectors,3layers; each geometry has matched onecopy/pairedcopy and both fixed sharedDown high halves. Gate-first strict scheduler, activation/combine0.',
        copy_policy='Per fixed paired chips, enumerate0/1/2 shared128-row tiles at halfA, remaining at halfB; minimize exact sum(native job command+input+NAND+channel output) chip bound. Full schedule separately replayed.',
        numeric_scope='Complete row relocation only; same nominal raw-I8 timing caveats. No verified native numeric ABI, device or full-model claim.',
        records=records,totals=totals,
        novelty='No novelty claim: shared-expert replication and load balancing have direct prior. Same-budget generic/routed replicas, host pinning and different scheduler controls still required.')
    assert sha(SOURCE)==EXPECTED
    (DEST/'screen.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(totals,indent=2))


if __name__=='__main__':main()
