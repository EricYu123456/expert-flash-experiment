#!/usr/bin/env python3
"""TRAIN-only Q38 complete-row boundary view screening.

Mandatory-work selector is an exact enumeration. Native costs preserve the
sealed backend; physical prefix/row-alias ABI and numerical validation are
separate prerequisites, not established by this counterfactual replay.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from research_20261103_topology_placement import sha
from research_20261103_topology_placement_replay import Replay,load_python

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/research-2026-11-02/round2'
OUT=BASE/'routed-boundary-views'
CAMPAIGN=BASE/'ordinary-read-compiler/full-campaign-v4-deduplicated/summary.json'
BITS=((np.arange(1024)[:,None]>>np.arange(10))&1).astype(np.int64)


def gu_cost(rows):
    rows=np.asarray(rows,np.int64)
    pages=(rows*2560+16383)//16384;waves=(pages+3)//4
    return 250+1280+waves*28000+2*(pages*16384//2560)


def load(layer):
    d=json.loads(CAMPAIGN.read_text())
    rec=next(r for r in d['records'] if r['model']=='qwen38' and r['layer']==layer and r['command_ns']==250)
    art=next(a for a in rec['artifacts'] if a['family']=='whole_rows')
    assert sha(ROOT/art['arrays'])==art['arrays_sha256']
    with np.load(ROOT/art['arrays']) as z:plan={k:z[k] for k in z.files}
    rows=np.zeros((513,2,16),np.int64);costs=np.zeros_like(rows)
    for e in range(513):
        for p in range(2):
            lo,hi=plan['offsets'][e,p]
            assert hi-lo==16 and np.array_equal(plan['jobs'][lo:hi,0],np.arange(16))
            for c,r,C,pages,waves in plan['jobs'][lo:hi]:
                rows[e,p,c]+=r;costs[e,p,c]+=250+C//2+28000*waves+2*(pages*16384//C)
    assert all(len(np.flatnonzero(r==101))==3 for r in rows[:512,0])
    routes=plan['selection_routes'];base=costs[routes].sum((1,2))+costs[-1].sum(0)
    return rec,art,plan,rows,costs,routes,base


def edges_and_delta(rows,strategy,routes,base):
    choices=[]
    for e in range(512):
        high=np.flatnonzero(rows[e,0]==101);low=np.flatnonzero(rows[e,0]<101)
        if strategy=='xor8':cc=[(int(s),int(s^8)) for s in high if s^8 in low]
        else:cc=[(int(s),int(t)) for s in high for t in low]
        choices.append(cc)
    freq=np.bincount(routes.ravel(),minlength=512)
    mean_load=base.mean(0)
    edges=[]
    for e,cc in enumerate(choices):
        if strategy=='xor8':edge=cc[0]
        elif strategy=='frequency':edge=max(cc,key=lambda st:(mean_load[st[0]]-mean_load[st[1]],-st[0],-st[1]))
        else:
            # Prefer eliminating a6/4 imbalance, then5/4; no hardware xor restriction.
            dw=(rows[e,1]*640+65535)//65536
            edge=min(cc,key=lambda st:(dw[st[1]],-dw[st[0]],st))
        edges.append(edge)
    edges=np.asarray(edges,np.int32);delta=delta_for_edges(rows,edges)
    return choices,edges,delta


def delta_for_edges(rows,edges):
    alt=rows[:512,0].copy()
    for e,(s,t) in enumerate(edges):alt[e,s],alt[e,t]=alt[e,t],alt[e,s]
    return gu_cost(alt)-gu_cost(rows[:512,0])


def oracle(base,routes,delta):
    loads=np.stack([base[q]+BITS@delta[r] for q,r in enumerate(routes)])
    scores=loads.max(2);which=scores.argmin(1)
    return dict(loads=loads,work=scores[np.arange(len(routes)),which],bits=which)


def learn_edges(rows,routes,base,choices,initial,passes=2):
    edges=initial.copy();delta=delta_for_edges(rows,edges);o=oracle(base,routes,delta)
    loads=o['loads'];values=o['work'].copy();members=[np.flatnonzero((routes==e).any(1)) for e in range(512)]
    order=sorted(range(512),key=lambda e:(-len(members[e]),e));moves=trials=0
    for sweep in range(passes):
        changed=0
        for e in order:
            ii=members[e]
            if not len(ii):continue
            old=delta[e];best=int(values[ii].sum());choice=None
            for edge in choices[e]:
                if np.array_equal(edge,edges[e]):continue
                alt=rows[e,0].copy();s,t=edge;alt[s],alt[t]=alt[t],alt[s]
                new=gu_cost(alt)-gu_cost(rows[e,0]);trial=[]
                for q in ii:
                    k=int(np.flatnonzero(routes[q]==e)[0])
                    ll=loads[q]+BITS[:,k,None]*(new-old)
                    trial.append((q,ll,ll.max(1).min()))
                trials+=1;value=sum(int(x[2]) for x in trial)
                if value<best:best=value;choice=(edge,new,trial)
            if choice:
                edges[e]=choice[0];delta[e]=choice[1]
                for q,ll,value in choice[2]:loads[q]=ll;values[q]=value
                changed+=1
        moves+=changed
        if not changed:break
    direct=oracle(base,routes,delta);assert np.array_equal(direct['work'],values)
    return edges,delta,dict(passes=sweep+1,changed_experts=moves,candidate_trials=trials,
                            mean_mandatory_ns=float(values.mean()),candidate_limit=39)


def static_joint(base,routes,delta):
    members=[np.flatnonzero((routes==e).any(1)) for e in range(512)]
    dynamic=oracle(base,routes,delta)
    majority=np.zeros(512,np.int8)
    for e,ii in enumerate(members):
        if len(ii):
            votes=[(int(dynamic['bits'][q])>>int(np.flatnonzero(routes[q]==e)[0]))&1 for q in ii]
            majority[e]=int(sum(votes)*2>=len(votes))
    starts=[np.zeros(512,np.int8),np.ones(512,np.int8),majority]
    best=None
    for seed,initial in enumerate(starts):
        bit=initial.copy();load=base+(delta*bit[:,None])[routes].sum(1);cost=load.max(1)
        accepted=0
        for sweep in range(12):
            changed=0
            for e,ii in enumerate(members):
                if not len(ii):continue
                dl=delta[e]*(1-2*int(bit[e]));ll=load[ii]+dl;value=ll.max(1)
                if value.sum()<cost[ii].sum():bit[e]^=1;load[ii]=ll;cost[ii]=value;changed+=1
            accepted+=changed
            if not changed:break
        candidate=(float(cost.mean()),bit,dict(start=seed,sweeps=sweep+1,accepted=accepted))
        if best is None or candidate[0]<best[0]:best=candidate
    return best[1],dict(mean_mandatory_ns=best[0],**best[2])


def compact(plan,route):
    chosen=np.r_[route,512];jobs=[];offsets=np.zeros((11,2,2),np.int32)
    for e,source in enumerate(chosen):
        for p in range(2):
            lo,hi=plan['offsets'][source,p];offsets[e,p,0]=len(jobs)
            jobs.extend(plan['jobs'][lo:hi]);offsets[e,p,1]=len(jobs)
    return dict(offsets=offsets,jobs=np.asarray(jobs,np.int32),
                input_columns=np.ascontiguousarray(plan['input_columns'][chosen]),total_rows=plan['total_rows'][chosen])


def variant(plan,route,edges,bits):
    p=dict(plan);p['jobs']=plan['jobs'].copy()
    for k,e in enumerate(route):
        if not bits[k]:continue
        s,t=edges[e];lo=plan['offsets'][k,0,0]
        jobs=p['jobs'];jobs[lo+s,1],jobs[lo+t,1]=jobs[lo+t,1],jobs[lo+s,1]
        for c in [s,t]:
            j=lo+c;jobs[j,3]=(int(jobs[j,1])*2560+16383)//16384;jobs[j,4]=(int(jobs[j,3])+3)//4
    return p


def native_policy(native,compacts,routes,edges,querybits,service):
    result=[];local=np.arange(10,dtype=np.int32)[None]
    for q,route in enumerate(routes):
        p=variant(compacts[q],route,edges,querybits[q])
        result.append(int(native.run(p,local,service)[0][0]))
    return np.asarray(result,np.int64)


def static_native(native,compacts,routes,edges,initial,service,passes=3):
    bit=initial.copy();cost=native_policy(native,compacts,routes,edges,bit[routes],service)
    members=[np.flatnonzero((routes==e).any(1)) for e in range(512)]
    local=np.arange(10,dtype=np.int32)[None];moves=evaluations=0
    order=sorted(range(512),key=lambda e:(-len(members[e]),e))
    for sweep in range(passes):
        changed=0
        for e in order:
            ii=members[e]
            if not len(ii):continue
            bit[e]^=1;trial=[]
            for q in ii:
                p=variant(compacts[q],routes[q],edges,bit[routes[q]])
                trial.append(int(native.run(p,local,service)[0][0]));evaluations+=1
            if sum(trial)<cost[ii].sum():cost[ii]=trial;changed+=1
            else:bit[e]^=1
        moves+=changed
        if not changed:break
    return bit,cost,dict(sweeps=sweep+1,accepted=moves,native_query_evaluations=evaluations,
                         objective='TRAIN native mean; same fixed A/B views, no replicas')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--layer',type=int,default=0);args=ap.parse_args()
    start=time.monotonic();dest=OUT/f'L{args.layer}';dest.mkdir(parents=True,exist_ok=True)
    rec,art,plan,rows,costs,routes,base=load(args.layer);native=Replay();backend=load_python()
    compacts=[compact(plan,r) for r in routes];identity=np.zeros((64,10),np.int8)
    costs_identity=native_policy(native,compacts,routes,np.zeros((512,2),np.int32),identity,rec['service'])
    assert np.array_equal(costs_identity,plan['cost'])
    cases={};arrays=dict(routes=routes,base_rows=rows,base_cost=costs_identity,base_joined_work=base)
    for strategy in ['xor8','priority','frequency','learned']:
        init='priority' if strategy=='learned' else strategy
        choices,edges,delta=edges_and_delta(rows,init,routes,base);fit=None
        if strategy=='learned':edges,delta,fit=learn_edges(rows,routes,base,choices,edges)
        dyn=oracle(base,routes,delta)
        static,staticfit=static_joint(base,routes,delta)
        static,staticcost,nativefit=static_native(native,compacts,routes,edges,static,rec['service'])
        dynbits=BITS[dyn['bits']];dynamiccost=native_policy(native,compacts,routes,edges,dynbits,rec['service'])
        baseline=min(float(staticcost.mean()),float(costs_identity.mean()))
        payload=sum(int((rows[e,0,s]-rows[e,0,t])*2560) for e,(s,t) in enumerate(edges))
        cases[strategy]=dict(edge_fit=fit,static_joint_fit=staticfit,static_native_fit=nativefit,
                            mean_static_native_ns=float(staticcost.mean()),p99_static_native_ns=float(np.percentile(staticcost,99)),
                            mean_dynamic_mandatory_ns=float(dyn['work'].mean()+2000),
                            mean_dynamic_native_ns=float(dynamiccost.mean()),p99_dynamic_native_ns=float(np.percentile(dynamiccost,99)),
                            dynamic_vs_best_static_mean_gain_pct=100*(1-dynamiccost.mean()/baseline),
                            possible_dynamic_vs_best_static_upper_pct=100*(1-(dyn['work'].mean()+2000)/baseline),
                            duplicate_coefficient_bytes=payload,extra_programmed_page_bytes=512*65536,
                            whole_GU_copy_budget_payload=payload//3276800,whole_GU_copy_budget_pages=(512*65536)//(51*65536))
        for key,value in [('edges',edges),('delta',delta),('dynamic_state',dyn['bits']),('dynamic_work',dyn['work']),
                          ('dynamic_cost',dynamiccost),('static_bits',static),('static_cost',staticcost)]:arrays[strategy+'__'+key]=value
        # Direct graph spot checks include the physical descriptor-size changes.
        for q in [0,63]:
            pp=variant(compacts[q],routes[q],edges,dynbits[q])
            got=backend.ReferenceGraph.build(pp,np.arange(11),backend.Service(**rec['service'])).run()
            assert got['cost']==dynamiccost[q]
    np.savez_compressed(dest/'screen.npz',**arrays)
    result=dict(layer=args.layer,scope='Only frozen TRAIN64 selection queries; no heldout.',source_artifact=art,
                source_campaign_sha256=sha(CAMPAIGN),service=rec['service'],source_mean_native_ns=float(costs_identity.mean()),
                cases=cases,elapsed_seconds=time.monotonic()-start,script_sha256=sha(__file__),
                limitations='Host selector/scatter time not charged; native prefix/row-map/physical view legality under separate audit. Logical B target is coreT+donor suffix, not a relabeled contiguous global rowrange. Useful results once, fringe outputs truly computed/transferred then discarded. Static controls retain exactly same A/B views but repack one view without replicas.')
    (dest/'screen.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['layer','source_mean_native_ns','cases','elapsed_seconds']},indent=2))


if __name__=='__main__':main()
