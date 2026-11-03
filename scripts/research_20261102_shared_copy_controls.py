#!/usr/bin/env python3
"""Matched-capacity routed/hybrid copies under the same native FFN scheduler."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import itertools
import json

import numpy as np

from research_20261102_shared_pipeline import BASE,SOURCE,EXPECTED,JOINT,sha,build_jobs
from research_20261102_shared_more_profiles import row_work

DEST=BASE/'shared-copy-controls'
ORACLE=BASE/'shared-oracle/screen.json'


def shared_assignment(occupied,allowed):
    rows=np.zeros(16,dtype=np.int64);decisions=[]
    for p in range(8):
        choices=[]
        for a in allowed:
            pair_rows=np.array([128*a,128*(2-a)])
            added=row_work(pair_rows,1536)
            choices.append((max(occupied[p]+added[0],occupied[p+8]+added[1]),
                int(added.sum()),abs(a-1),a))
        a=min(choices)[-1];decisions.append(a);rows[p]=128*a;rows[p+8]=128*(2-a)
    return rows,decisions


def choose_jobs(route,gu,down,sd,copied,allowed):
    available=[i for i,e in enumerate(route) if int(e) in copied]
    best=None
    for flips in itertools.product((False,True),repeat=len(available)):
        rows=gu[route].copy()
        for ei,flip in zip(available,flips):
            if flip:rows[ei]=np.roll(rows[ei],8)
        occupied=row_work(rows,1536).sum(axis=0)+row_work(down[route],512).sum(axis=0)+row_work(sd,1024)
        sr,decisions=shared_assignment(occupied,allowed)
        full=occupied+row_work(sr,1536)
        key=(int(full.max()),int(full.sum()),sum(flips),tuple(flips))
        if best is None or key<best[0]:best=(key,rows,sr,decisions,available,flips)
    key,rows,sr,decisions,available,flips=best
    items=[dict(gu_rows=rows[i].tolist(),down_rows=down[e].tolist(),down_columns=512) for i,e in enumerate(route)]
    items.append(dict(gu_rows=sr.tolist(),down_rows=sd.tolist(),down_columns=1024))
    return items,key[0]+2000,dict(shared_choices=decisions,flipped_logical_experts=[int(route[i]) for i,f in zip(available,flips) if f])


def worker(task):
    layer,mapping,policy,side=task
    p=json.loads(SOURCE.read_text());o=json.loads(ORACLE.read_text());j=json.loads(JOINT.read_text())
    phase={x['name']:x for x in p['phases'] if x['layer']==layer}
    gu=np.asarray(phase['gate_up']['rows']);down=np.asarray(phase['down']['rows'])
    if mapping=='joint':
        labels=np.asarray(next(x for x in j['results'] if x['layer']==layer)['joint_best']['labels'])
        down=np.where(np.arange(16)[None,:]//4==labels[:,None],0,128)
    elif mapping=='opposite':down=np.where(gu>0,0,192)
    else:assert mapping=='source'
    plan=next(x for x in o['layers'] if x['layer']==layer)
    if policy=='hybrid':
        copied=[plan['optimal_hybrid']['routed_expert']]
        allowed=(0,1) if plan['optimal_hybrid']['copy_original_shared_half']=='A' else (1,2)
    else:
        copied=plan[policy];allowed=(1,)
    assert len(copied)==(1 if policy=='hybrid' else 2)
    sd=np.full(16,64);sd[side:side+8]=128
    with np.load(p['training_routes']) as f:selected=f['routes'][:32,:64,f['layers'].tolist().index(layer)].reshape(-1,6)
    costs=[];bounds=[];trace=None;all_decisions=[]
    for idx,route in enumerate(selected):
        items,bound,decision=choose_jobs(route,gu,down,sd,copied,allowed)
        r=build_jobs(items).run(False,trace=idx==0)
        costs.append(r['makespan_ns']);bounds.append(bound);all_decisions.append(decision)
        assert costs[-1]>=bound
        if trace is None:trace=dict(layer=layer,mapping=mapping,policy=policy,side=side,
            copied_logical_experts=copied,shared_allowed_a=list(allowed),selected=route.tolist(),items=items,
            choice=decision,**r)
    label=f'L{layer}.{mapping}.{policy}.sd{side}'
    np.savez_compressed(DEST/(label+'.npz'),cost=np.asarray(costs),native_chip_bound=np.asarray(bounds))
    (DEST/(label+'.decisions.json')).write_text(json.dumps(all_decisions)+'\n')
    (DEST/(label+'.trace.json')).write_text(json.dumps(trace,indent=2)+'\n')
    return dict(layer=layer,mapping=mapping,policy=policy,side=side,label=label,
        mean_ns=float(np.mean(costs)),p99_ns=float(np.percentile(costs,99)),mean_bound_ns=float(np.mean(bounds)),
        copied_experts=copied,shared_allowed_a=list(allowed),
        npz_sha256=sha(DEST/(label+'.npz')),trace_sha256=sha(DEST/(label+'.trace.json')),
        decisions_sha256=sha(DEST/(label+'.decisions.json')))


def main():
    DEST.mkdir(parents=True,exist_ok=True);assert sha(SOURCE)==EXPECTED
    frozen_inputs={str(p):sha(p) for p in (SOURCE,JOINT,ORACLE)}
    p=json.loads(SOURCE.read_text());assert sha(p['training_routes'])==p['training_routes_sha256']
    tasks=[(layer,mapping,policy,side) for layer in p['layers'][:3]
        for mapping in ('source','joint','opposite') for policy in ('hot2','optimal2','hybrid') for side in (0,8)]
    records=[]
    with ProcessPoolExecutor(max_workers=8) as pool:
        for future in as_completed([pool.submit(worker,t) for t in tasks]):
            r=future.result();records.append(r);print(r['label'],r['mean_ns']/1e6,r['p99_ns']/1e6,flush=True)
            (DEST/'partial.json').write_text(json.dumps(records,indent=2)+'\n')
    totals=[]
    for mapping in ('source','joint','opposite'):
        for policy in ('hot2','optimal2','hybrid'):
            for side in (0,8):
                rr=[r for r in records if r['mapping']==mapping and r['policy']==policy and r['side']==side]
                data=[np.load(DEST/(r['label']+'.npz')) for r in rr]
                x=np.stack([a['cost'] for a in data]).sum(axis=0);b=np.stack([a['native_chip_bound'] for a in data]).sum(axis=0)
                totals.append(dict(mapping=mapping,policy=policy,side=side,mean_ns=float(x.mean()),p99_ns=float(np.percentile(x,99)),mean_bound_ns=float(b.mean())))
    for path,digest in frozen_inputs.items():assert sha(path)==digest
    out=dict(status='complete train3layer matched16tile extra capacity native replay',
        created_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),inputs_sha256=frozen_inputs,
        training_routes_sha256=sha(p['training_routes']),records=records,totals=totals,
        storage='Every policy adds16 GU128x1536 tile copies=3MiB/layer,192pages,48storedwaves. Routed2 means2 whole GU copies; hybrid means8 shared tiles plus1 whole8tile routedGU.',
        selection='Enumerate legal copied routedGU whole-owner choices and legal perpair shared0/1/2 tiles; minimize nativechip occupancy bound, then run the same complete dependency schedule.',
        limitations=['Training copy plans were selected by GU NAND-only oracle, not globally optimized native FFN cost.',
            'No general optimality certificate over arbitrary per-chip copy locations or replica budgets.',
            'Same raw-I8 timing/numeric-ABI caveats and activation/combine0 as shared_pipeline. Not device measurements or full-model execution.',
            'One full-row native operator, one context, complete expert GU dependency, private down inputs, host/control/channel/PCIe costs preserved.'])
    (DEST/'screen.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(totals,indent=2))


if __name__=='__main__':main()
