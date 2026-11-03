#!/usr/bin/env python3
"""Whole-row shared/routed FFN native-job replay; no new chip operation.

Shared GU replicas are two adjacent 128-row tiles at each paired owner.  A
query can choose one tile at each owner or both at one owner.  Every row is
executed exactly once.  The published strict scheduler owns a chip for the
entire native command/input/compute/output interval.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from research_20261102_granite4_pipeline import Graph, SOURCE, EXPECTED

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/research-2026-11-02/round2'
DEST=BASE/'shared-pipeline'
JOINT=BASE/'pipeline-joint/joint_work_screen.json'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build_jobs(items,activation_ns=0,combine_ns=0):
    g=Graph();submit=g.add(0,2000,kind='initial_host_submission')
    upload=g.add(2,(1536+7)//8,[submit],kind='common_input_to_controller')
    ready=[]
    for ei,item in enumerate(items):
        outputs=[]
        for c,r in enumerate(item['gu_rows']):
            if r:
                w=(int(r)*1536+65535)//65536
                outputs.append(g.matrix(ei,c,0,int(r),1536,w,submit,upload))
        ready.append(g.add(27,0,outputs,expert=ei,kind='whole_expert_gu_host_ready'))
    terminal=[]
    for ei,item in enumerate(items):
        act=g.add(0,activation_ns,[ready[ei]],1,ei,'host_activation')
        ds=g.add(0,2000,[act],1,ei,'expert_down_host_submission')
        cols=item['down_columns']
        up=g.add(2,(cols+7)//8,[ds],1,ei,'private_input_to_controller')
        for c,r in enumerate(item['down_rows']):
            if r:
                w=(int(r)*cols+65535)//65536
                terminal.append(g.matrix(ei,c,1,int(r),cols,w,ds,up))
    g.add(0,combine_ns,terminal,1,kind='host_combine')
    return g


def shared_gu_ranges(n_a,replica):
    """Actual logical rows and fixed stored offsets, not only row counts."""
    ranges=[]
    for pair in range(8):
        first=pair*256
        if not replica or n_a==3:
            ranges.extend([(pair,first,128),(pair+8,first+128,128)])
        else:
            owner=pair+8 if n_a>3 else pair
            ranges.append((owner,first,256))
    ordered=sorted((start,start+count) for _,start,count in ranges)
    assert ordered[0][0]==0 and ordered[-1][1]==2048
    assert all(b==c for (_,b),(c,_) in zip(ordered,ordered[1:]))
    return ranges


def worker(task):
    layer,mapping,replica,shared_down_side,samples,down_first,outdir=task
    d=json.loads(SOURCE.read_text());joint=json.loads(JOINT.read_text())
    with np.load(d['training_routes']) as f:
        li=f['layers'].tolist().index(layer)
        selected=f['routes'][:32,:64,li].reshape(-1,6)
    indices=np.arange(len(selected)) if samples==0 else np.unique(np.linspace(0,len(selected)-1,samples,dtype=int))
    phases={(p['layer'],p['name']):p for p in d['phases']}
    gu=np.asarray(phases[layer,'gate_up']['rows']);down=np.asarray(phases[layer,'down']['rows'])
    if mapping=='joint':
        labels=np.asarray(next(x for x in joint['results'] if x['layer']==layer)['joint_best']['labels'])
        down=np.where(np.arange(16)[None,:]//4==labels[:,None],0,128)
    elif mapping=='opposite':
        home=(gu[:,:8].sum(axis=1)>0)
        down=np.zeros((64,16),dtype=np.int64)
        for e in range(64):down[e,8:]=192 if home[e] else 0;down[e,:8]=0 if home[e] else 192
    assert np.all(gu.sum(axis=1)==1024) and np.all(down.sum(axis=1)==1536)
    shared_down=np.full(16,64,dtype=np.int64);shared_down[shared_down_side:shared_down_side+8]=128
    assert shared_down.sum()==1536
    rows=[];lb=[];nands=[];trace=None
    for idx in indices:
        route=selected[idx]
        items=[dict(gu_rows=gu[e].tolist(),down_rows=down[e].tolist(),down_columns=512) for e in route]
        n_a=int((gu[route,:8].sum(axis=1)>0).sum())
        ranges=shared_gu_ranges(n_a,replica)
        sr=np.zeros(16,dtype=np.int64)
        for c,start,count in ranges:sr[c]+=count
        items.append(dict(gu_rows=sr.tolist(),down_rows=shared_down.tolist(),down_columns=1024))
        g=build_jobs(items)
        r=g.run(down_first,trace=trace is None)
        rows.append(r['makespan_ns'])
        chip=np.zeros(16,dtype=np.int64);total=0
        for item in items:
            for key,cols in [('gu_rows',1536),('down_rows',item['down_columns'])]:
                rr=np.asarray(item[key]);waves=(rr*cols+65535)//65536
                total+=int(waves.sum())
                chip+=waves*28000+np.where(rr>0,250+(cols+1)//2+rr*2,0)
        lb.append(2000+int(chip.max()));nands.append(total)
        assert r['makespan_ns']>=lb[-1]
        if trace is None:
            trace=dict(layer=layer,mapping=mapping,shared_gu_replica=replica,shared_down_high_half=shared_down_side,
                route_index=int(idx),selected=route.tolist(),shared_gu_ranges=ranges,items=items,**r)
    label=f'L{layer}.{mapping}.rep{int(replica)}.sd{shared_down_side}.df{int(down_first)}'
    target=Path(outdir)
    np.savez_compressed(target/(label+'.npz'),indices=indices,cost=np.asarray(rows,dtype=np.int64),
        native_chip_bound=np.asarray(lb,dtype=np.int64),total_chip_waves=np.asarray(nands,dtype=np.int64))
    (target/(label+'.trace.json')).write_text(json.dumps(trace,indent=2)+'\n')
    return dict(layer=layer,mapping=mapping,shared_gu_replica=replica,shared_down_high_half=shared_down_side,
        down_first=down_first,label=label,mean_ns=float(np.mean(rows)),p99_ns=float(np.percentile(rows,99)),
        mean_native_chip_bound_ns=float(np.mean(lb)),mean_chip_waves=float(np.mean(nands)),
        numpy_sha256=sha(target/(label+'.npz')),trace_sha256=sha(target/(label+'.trace.json')))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--samples',type=int,default=0)
    ap.add_argument('--workers',type=int,default=8);ap.add_argument('--layers',type=int,default=3)
    ap.add_argument('--output',type=Path,default=DEST);args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    assert sha(SOURCE)==EXPECTED;d=json.loads(SOURCE.read_text())
    assert sha(d['training_routes'])==d['training_routes_sha256']
    # Joint fit is currently available for only the first three layers.
    assert 1<=args.layers<=3
    tasks=[(layer,mapping,replica,side,args.samples,False,str(args.output))
        for layer in d['layers'][:args.layers] for mapping in ('source','joint','opposite')
        for replica in (False,True) for side in (0,8)]
    records=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(worker,task) for task in tasks]
        for future in as_completed(futures):
            r=future.result();records.append(r)
            print(r['label'],r['mean_ns']/1e6,r['p99_ns']/1e6,flush=True)
            (args.output/'partial.json').write_text(json.dumps(records,indent=2)+'\n')
    totals=[]
    for mapping in ('source','joint','opposite'):
        for replica in (False,True):
            for side in (0,8):
                cases=[r for r in records if r['mapping']==mapping and r['shared_gu_replica']==replica and r['shared_down_high_half']==side]
                values=np.stack([np.load(args.output/(r['label']+'.npz'))['cost'] for r in cases]).sum(axis=0)
                totals.append(dict(mapping=mapping,shared_gu_replica=replica,shared_down_high_half=side,
                    mean_ns=float(values.mean()),p99_ns=float(np.percentile(values,99))))
    out=dict(status='Complete train-only first3layer whole-row shared/routed replay',
        created_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),
        strict_scheduler_source_sha256=sha(Path(__file__).with_name('research_20261102_granite4_pipeline.py')),
        frozen_plan_sha256=EXPECTED,root_joint_sha256=sha(JOINT),training_routes_sha256=sha(d['training_routes']),
        arguments={**vars(args),'output':str(args.output)},records=records,totals=totals,
        shared_geometry=dict(gu_rows=2048,gu_columns=1536,down_rows=1536,down_columns=1024,
            shared_down='8x128rows/2waves plus8x64rows/1wave:24total waves. High half is a matched fixed control.',
            shared_gu_onecopy_bytes_per_layer=2048*1536,extra_gu_replica_bytes_per_layer=2048*1536,
            extra_gu_replica_bytes_full40=40*2048*1536,
            replica='Each paired chip stores the same two contiguous128row tiles. One selected copy of every row is computed.',
            no_partial_dot_products=True,no_additional_native_context=True),
        limitations=['Cost replay, not measured NAND/PE or full-model execution; shared metadata already inspected separately.',
            'Same nominal raw-I8 A8/output4B contract caveat; no unchanged-GGUF-on-AiF numeric claim.',
            'Activation/combine=0 is optimistic; native commands/submissions, full input reload, channels and PCIe are charged.',
            'Fixed gate-first priority; schedules are feasible, not globally optimal. Other schedules and generic same-budget replication remain necessary baselines.',
            'Only three train layers. No heldout or quality claims. Host shared-expert pinning and partial-copy controllers remain strong comparisons.',
            'Shared-expert replication/load balancing is prior art, not a novelty claim.'])
    assert sha(SOURCE)==EXPECTED
    (args.output/'screen.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(totals,indent=2))


if __name__=='__main__':main()
