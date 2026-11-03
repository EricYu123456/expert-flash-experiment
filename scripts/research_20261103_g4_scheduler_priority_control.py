#!/usr/bin/env python3
"""Five frozen G4 policies, two static phase priorities; no geometry/bit fit."""
import argparse
from dataclasses import replace
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import research_20261103_ordinary_read_backend as backend

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/research-2026-11-02/round2'
JOINT=BASE/'g4-joint-prefix-replica-screen'
GU=BASE/'g4-elastic-prefix-screen/work-v3-independent-prefix'
DOWN=BASE/'g4-down-paired-merge-screen'
INC=BASE/'compact-coupled-solver/continuation/native/0107.npz'
CFG=BASE/'compact-coupled-solver/protocol.frozen.json'
OUT=JOINT/'scheduler-priority-control'
SELF=Path(__file__).resolve()
FAMILIES=['B','GU_allA','GU_greedy','Down14_unitflow','joint_greedy']


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def load(p):
    with np.load(p) as z:return {k:z[k].copy() for k in z.files}
def stats(x):return dict(mean_ns=float(x.mean()),p99_ns=float(np.percentile(x,99)))


def freeze():
    OUT.mkdir(exist_ok=False)
    paths=[SELF,INC,CFG,GU/'contract_arrays.npz',GU/'rowid_ledgers.json',GU/'greedy_arrays.npz',
           GU/'native-v1/arrays.npz',DOWN/'arrays.npz',DOWN/'native-graph14/arrays.npz',
           JOINT/'work_arrays.npz',JOINT/'native-v1/arrays.npz',
           ROOT/'scripts/research_20261103_ordinary_read_backend.py',ROOT/'scripts/research_20261103_ordinary_read_native.cpp']
    dump(OUT/'protocol.frozen.json',dict(inputs=[dict(path=str(p),sha256=sha(p)) for p in paths],
        scope='Fivealreadyfrozen G4L0 TRAIN512policies xdown_first0/1=5120costs. No new geometry/sourcegraph/runtimebit fitting, nativeCP, selectorbenchmark, weights orheldout.',
        families=FAMILIES,choice='GUallA=ones8/layout3; GUgreedy=originalfeasible_bits[3]; Down14=originalpure-restored-native merge_bits; joint=originalchosen_u/chosen_v. Runtime bits never change with priority.',
        service='Same16chip onecontext/fullinput/channel/PCIe/host/barrier cost. ONLY down_first0->1 changes. Lowerbounds mustbe identical betweenpriorities.',
        bridge='Everyfamilydf0cost andLB512exact againstownpriorfrozenreplay; Bdf0cost/LB againstCP0107. Everyfamily/priority q0fullReferenceGraph trace exact, owner/reload/wholeGU/A8 barriers.',
        reporting='Bothstaticpriorities foreveryfamily. Selectonestaticpriority per familybyTRAINmean forcomparison; no querywise freepriorityoracle.',
        claim='Baseline scheduler-sensitivity control, not a new method or device performance result. ReportlargeallApriorityeffectbeforenewwork.'))
    print(sha(OUT/'protocol.frozen.json'))


def run():
    proto=json.loads((OUT/'protocol.frozen.json').read_text())
    for x in proto['inputs']:assert sha(x['path'])==x['sha256']
    assert not(OUT/'report.json').exists();began=time.perf_counter()
    b=load(INC);w=load(JOINT/'work_arrays.npz');g=load(GU/'greedy_arrays.npz');gn=load(GU/'native-v1/arrays.npz')
    dn=load(DOWN/'native-graph14/arrays.npz');jn=load(JOINT/'native-v1/arrays.npz')
    entries=json.loads((GU/'rowid_ledgers.json').read_text())[3]['experts']
    service=backend.Service(**json.loads(CFG.read_text())['service']);services=[replace(service,down_first=i) for i in (0,1)]
    assert all(vars(services[0])[k]==vars(services[1])[k] for k in vars(service) if k!='down_first')
    native=backend.NativeReplay(OUT/'build');routes=w['routes'];rows=b['rows']
    expected_cost=[b['cost'],gn['allA'],gn['greedy'],dn['cost'],jn['cost']]
    expected_LB=[b['lower_bound'],gn['allA_LB'],gn['greedy_LB'],dn['lower_bound'],jn['lower_bound']]
    costs=np.zeros((5,2,512),np.int64);bounds=costs.copy();trace_checks=[];owner_checks=0
    all_u=np.zeros((5,512,6,8),np.int8);all_v=np.zeros((5,512,6,6),np.int8)
    all_u[1]=1;all_u[2]=g['feasible_bits'][3];all_v[3]=dn['merge_bits'].reshape(512,6,6)
    all_u[4]=w['chosen_u'];all_v[4]=w['chosen_v']
    resident={}
    for e in range(65):
        for phase in range(2):
            for j in range(*w['resident_offsets'][e,phase]):resident[e,phase,int(w['resident_jobs'][j,0])]=w['resident_addresses'][j]
    down_images=[]
    for e in range(64):
        starts=np.r_[0,np.cumsum(rows[e,1])];original=[np.arange(starts[c],starts[c+1],dtype=np.int32) for c in range(16)]
        image=[x.copy() for x in original]
        for src,dest in w['Down_edges'][e]:image[dest]=np.r_[original[dest],original[src]]
        down_images.append(image)
    for fi,family in enumerate(FAMILIES):
        for q,route in enumerate(routes):
            selected=np.r_[route,64];local=rows[selected].copy()
            for k,e in enumerate(route):
                entry=entries[e];u=all_u[fi,q,k];v=all_v[fi,q,k]
                image={int(c):np.asarray(ids,np.int32) for c,ids in entry['resident_prefix_image'].items()}
                for i,c in enumerate(entry['sources']):local[k,0,c]=128 if u[i] else rows[e,0,c]
                for i,c in enumerate(entry['low_destinations']):local[k,0,c]=0 if u[i] else 40
                hd=entry['high_destination'];local[k,0,hd]=rows[e,0,hd] if not u[7] else entry['high_prefix_rows'] if not u.all() else 0
                guids=np.concatenate([image[c][:local[k,0,c]] for c in range(16)])
                assert np.array_equal(np.unique(guids),np.arange(1024))
                for i,(src,dest) in enumerate(w['Down_edges'][e]):
                    if v[i]:local[k,1,src]=0;local[k,1,dest]=256
                downids=np.concatenate([down_images[e][c][:local[k,1,c]] for c in range(16)])
                assert np.array_equal(np.sort(downids),np.arange(1536));owner_checks+=1
            assert np.array_equal(local[6],rows[64])
            plan=backend.lower_rows(local[:,0],local[:,1],1536,[512]*6+[1024],service)
            for k,e in enumerate(selected):
                for phase in (0,1):
                    for c,nr,nc,pages,waves in plan['jobs'][slice(*plan['offsets'][k,phase])]:
                        addr=resident[int(e),phase,int(c)]
                        assert pages<=addr[4] and waves<=addr[3] and addr[2]+waves<=192
            for priority,s in enumerate(services):
                got=native.run(plan,np.arange(6,dtype=np.int32)[None],6,1,s,trace=q==0)
                costs[fi,priority,q]=got['cost'][0];bounds[fi,priority,q]=got['lower_bound'][0]
                if q==0:
                    graph=backend.ReferenceGraph.build(plan,np.arange(7),s);ref=graph.run();tr=ref['trace']
                    assert ref['cost']==got['cost'][0] and np.array_equal(tr,got['trace'])
                    owner_intervals=[[]for _ in range(16)]
                    for i,node in enumerate(graph.nodes):assert all(tr[d,3]<=tr[i,2] for d in node['deps'])
                    for job in range(graph.job_count):
                        indexes=[i for i,n in enumerate(graph.nodes) if n['job']==job];assert len(indexes)==4
                        cmd,inp,comp,out=indexes;n=graph.nodes[cmd];chip=n['chip'];e=n['expert'];phase=n['phase']
                        assert n['acquire'] and graph.nodes[out]['release']
                        width=int(plan['input_columns'][e,phase]);assert graph.nodes[inp]['duration']==s.transfer(width,s.channel_bytes_per_second)
                        owner_intervals[chip].append((int(tr[cmd,2]),int(tr[out,3])))
                        if phase==1:
                            ready=[i for i,x in enumerate(graph.nodes) if x['resource']==s.resources-1 and x['expert']==e and x['phase']==0]
                            assert len(ready)==1 and tr[cmd,2]>=tr[ready[0],3]+s.host_submit_ns+s.activation_ns
                    for intervals in owner_intervals:
                        intervals.sort();assert all(x[1]<=y[0] for x,y in zip(intervals,intervals[1:]))
                    np.savez_compressed(OUT/f'{family}.df{priority}.q0.npz',rows=local,offsets=plan['offsets'],jobs=plan['jobs'],input_columns=plan['input_columns'],total_rows=plan['total_rows'],native_trace=got['trace'],ReferenceGraph_trace=tr)
                    trace_checks.append(dict(family=family,priority=priority,query=0,cost_ns=int(got['cost'][0]),nodes=len(tr),exact=True))
        assert np.array_equal(costs[fi,0],expected_cost[fi]) and np.array_equal(bounds[fi,0],expected_LB[fi])
        assert np.array_equal(bounds[fi,0],bounds[fi,1])
        print(json.dumps(dict(family=family,GU_first=stats(costs[fi,0]),Down_first=stats(costs[fi,1]),downfirst_mean_change_pct=float(100*(1-costs[fi,1].mean()/costs[fi,0].mean())))),flush=True)
    best_priority=costs.mean(2).argmin(1);records=[]
    bestB=costs[0,best_priority[0]]
    for fi,family in enumerate(FAMILIES):
        chosen=costs[fi,best_priority[fi]]
        records.append(dict(family=family,priority_cases=[dict(down_first=p,**stats(costs[fi,p]))for p in(0,1)],
                            selected_static_priority=int(best_priority[fi]),mean_gain_vs_best_priority_B_pct=float(100*(1-chosen.mean()/bestB.mean())),
                            p99_gain_vs_best_priority_B_pct=float(100*(1-np.percentile(chosen,99)/np.percentile(bestB,99)))))
    np.savez_compressed(OUT/'arrays.npz',cost=costs,lower_bound=bounds,GU_bits=all_u,Down_bits=all_v,routes=routes,best_static_priority=best_priority)
    report=dict(status='PASS fixed-bit/fixed-geometry scheduler priority sensitivity',protocol_sha256=sha(OUT/'protocol.frozen.json'),
                families=FAMILIES,native_costs=5120,df0_cost_and_LB_bridge_values=5120,
                unchanged_priority_independent_lowerbounds=2560,owner_views_each_phase=owner_checks,
                representative_traces=trace_checks,records=records,native_provenance=native.provenance,
                allA_priority_mean_change_pct=float(100*(1-costs[1,1].mean()/costs[1,0].mean())),
                scope='Onlydown_first changes; everyfamily givenboth. Selectedstaticpriority byTRAINmean, noquerywise freeoracle/new placementorbits/selectorfit. No weights/heldout/nativeCP.',seconds=time.perf_counter()-began)
    for x in proto['inputs']:assert sha(x['path'])==x['sha256']
    dump(OUT/'report.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','run']);args=ap.parse_args()
    freeze()if args.action=='freeze'else run()
