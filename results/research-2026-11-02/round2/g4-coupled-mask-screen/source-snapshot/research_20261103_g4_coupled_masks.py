#!/usr/bin/env python3
"""G4 W8TRAIN L0 coupled-coloring control, free vs aligned omit masks."""
import json
import math
import time
from pathlib import Path

import numpy as np

from research_20261103_arbitrary_rows_headroom import row_cost
from research_20261103_topology_placement import sha
from research_20261103_topology_placement_replay import Replay,load_python

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/research-2026-11-02/round2'
OUT=BASE/'g4-coupled-mask-screen'


def fit(routes,shared,service,seed,free,iterations=20000):
    rng=np.random.default_rng(seed);E=64
    cats=np.zeros((E,16),np.int8)
    for e in range(E):
        omit=np.arange(4*(e%4),4*(e%4)+4)
        cats[e,omit]=1
        available=np.flatnonzero(cats[e]==0)
        cats[e,rng.choice(available,5,replace=False)]=2
    # category0:lowGU+Down;1:highGU+omittedDown;2:highGU+Down.
    g=np.asarray([row_cost(r,1536,service) for r in [40,83,83]])
    d=np.asarray([row_cost(r,512,service) for r in [128,0,128]])
    table=g+d
    assert row_cost(82,1536,service)==row_cost(83,1536,service)
    members=[np.flatnonzero((routes==e).any(1)) for e in range(E)]
    loads=table[cats][routes].sum(1)+shared
    costs=loads.max(1);total=int(costs.sum());best=total;witness=cats.copy();accepted=valid=0
    for it in range(iterations):
        e=int(rng.integers(E));old=cats[e].copy();new=old.copy()
        if free:
            a,b=rng.choice(16,2,replace=False)
            if old[a]==old[b]:continue
            new[a],new[b]=old[b],old[a]
        elif rng.random()<.8:
            a=int(rng.choice(np.flatnonzero(old==2)))
            b=int(rng.choice(np.flatnonzero(old==0)))
            new[a],new[b]=old[b],old[a]
        else:
            oldgroup=int(np.flatnonzero(old==1)[0]//4)
            group=int(rng.integers(3));group+=group>=oldgroup
            new[old==1]=0;new[4*group:4*group+4]=1
            needed=5-int((new==2).sum())
            available=np.flatnonzero(new==0)
            if needed:new[rng.choice(available,needed,replace=False)]=2
        valid+=1;ii=members[e]
        delta=table[new]-table[old]
        proposed=loads[ii]+delta;pc=proposed.max(1)
        change=int((pc-costs[ii]).sum())
        temperature=2000*(.0001**(it/max(1,iterations-1)))
        if change<=0 or rng.random()<math.exp(-change/(len(routes)*temperature)):
            cats[e]=new;loads[ii]=proposed;costs[ii]=pc;total+=change;accepted+=1
            if total<best:best=total;witness=cats.copy()
    # Deterministic best-improvement one-chip exchanges within the legal family.
    cats=witness.copy();loads=table[cats][routes].sum(1)+shared;costs=loads.max(1)
    greedy_moves=0
    for sweep in range(4):
        changed=0
        for e in range(E):
            old=cats[e].copy();ii=members[e];choice=None
            for a in range(16):
                for b in range(a+1,16):
                    if old[a]==old[b] or (not free and (old[a]==1 or old[b]==1)):continue
                    delta=np.zeros(16,dtype=np.int64);delta[a]=table[old[b]]-table[old[a]];delta[b]=-delta[a]
                    pp=loads[ii]+delta;pc=pp.max(1);change=int((pc-costs[ii]).sum())
                    if change<0 and (choice is None or change<choice[0]):choice=change,a,b,pp,pc
            if choice:
                _,a,b,pp,pc=choice;cats[e,a],cats[e,b]=cats[e,b],cats[e,a]
                loads[ii]=pp;costs[ii]=pc;changed+=1
        greedy_moves+=changed
        if not changed:break
    assert np.all((cats==2).sum(1)==5) and np.all((cats==1).sum(1)==4)
    if not free:assert all(len(set(np.flatnonzero(c==1)//4))==1 for c in cats)
    gu=np.full((E,16),40,np.int64);down=np.where(cats==1,0,128)
    for e in range(E):
        high=np.flatnonzero(cats[e]!=0);gu[e,high]=82;gu[e,high[:6]]=83
    assert np.all(gu.sum(1)==1024) and np.all(down.sum(1)==1536)
    gw=(gu*1536+65535)//65536;dw=(down*512+65535)//65536
    assert np.array_equal(gw+dw,2+(cats==2))
    direct=(row_cost(gu,1536,service)[routes].sum(1)+row_cost(down,512,service)[routes].sum(1)+shared).max(1)
    assert np.array_equal(direct,costs)
    return gu,down,cats,dict(seed=seed,proposals=iterations,valid=valid,accepted=accepted,
                             greedy_mask_moves=greedy_moves,greedy_sweeps=sweep+1,
                             mean_joint_chip_ns=float(direct.mean()),p99_joint_chip_ns=float(np.percentile(direct,99)),
                             mean_joined_NAND_waves=float((gw[routes].sum(1)+dw[routes].sum(1)+shared_waves).max(1).mean()))


def main():
    global shared_waves
    start=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
    source=BASE/'ordinary-read-compiler/full-campaign-v5-W8-training/summary.json'
    campaign=json.loads(source.read_text());rec=next(r for r in campaign['records'] if r['model']=='granite4tiny' and r['layer']==0 and r['command_ns']==250)
    s=rec['service'];shared_rows=np.asarray(rec['shared_rows'])
    shared=sum(row_cost(shared_rows[p],C,s) for p,C in enumerate([1536,1024]))
    shared_waves=(shared_rows[0]*1536+65535)//65536+(shared_rows[1]*1024+65535)//65536
    with np.load(rec['training']['path']) as f:routes=np.ascontiguousarray(f['routes'][:,:,0].reshape(-1,6),np.int32)
    assert sha(rec['training']['path'])==rec['training']['sha256']
    native=Replay();backend=load_python();runs=[];arrays={'routes':routes};checks=[]
    for free in [False,True]:
        for seed in [20261103,20261104]:
            gu,down,cats,fitstats=fit(routes,shared,s,seed,free)
            allgu=np.r_[gu,shared_rows[0][None]];alldown=np.r_[down,shared_rows[1][None]]
            plan=backend.lower_rows(allgu,alldown,1536,[512]*64+[1024],backend.Service(**s))
            cost,bounds=native.run(plan,routes,s)
            key=('free' if free else 'aligned')+'_'+str(seed)
            for q in [0,255,511]:
                result=backend.ReferenceGraph.build(plan,np.r_[routes[q],64],backend.Service(**s)).run()
                assert result['cost']==cost[q];checks.append(dict(case=key,query=q,cost_ns=int(cost[q])))
            physical=backend.physical_addresses(plan,backend.Service(**s));physical.pop('addresses')
            runs.append(dict(key=key,free_omit=free,fit=fitstats,mean_native_ns=float(cost.mean()),p99_native_ns=float(np.percentile(cost,99)),footprint=physical))
            arrays[key+'__gu_rows']=allgu;arrays[key+'__down_rows']=alldown;arrays[key+'__categories']=cats;arrays[key+'__cost']=cost
    previous=BASE/'arbitrary-row-headroom/granite4tiny/result.json';baseline=json.loads(previous.read_text())
    strongest=min(v['mean_ns'] for k,v in baseline['records'].items())
    bestfree=min((r for r in runs if r['free_omit']),key=lambda r:r['mean_native_ns'])
    bestaligned=min((r for r in runs if not r['free_omit']),key=lambda r:r['mean_native_ns'])
    result=dict(scope='G4 L0 W8TRAIN512 only; shared fixed; no test; matched20kproposals+4greedysweeps; no algorithm novelty claimed.',
                old_coverage=dict(path=str(ROOT/'scripts/research_20261102_joint_depth.py'),sha256=sha(ROOT/'scripts/research_20261102_joint_depth.py'),
                                  restricted_extra_masks=2832,all_extra_masks=math.comb(16,5),missing_masks=1536,
                                  old_omit_choices=4,new_decompositions_per_extra_mask=math.comb(11,4)),
                source_summary_sha256=sha(source),training=rec['training'],service=s,
                strongest_previous_native_mean_ns=strongest,best_free_vs_previous_mean_gain_pct=100*(1-bestfree['mean_native_ns']/strongest),
                best_free_vs_aligned_mean_gain_pct=100*(1-bestfree['mean_native_ns']/bestaligned['mean_native_ns']),
                runs=runs,checks=checks,elapsed_seconds=time.monotonic()-start,script_sha256=sha(__file__))
    np.savez_compressed(OUT/'arrays.npz',**arrays);(OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['training','old_coverage','checks','service']},indent=2))


if __name__=='__main__':main()
