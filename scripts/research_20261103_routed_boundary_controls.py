#!/usr/bin/env python3
"""Same-budget whole-GU replication and exact native selector spot checks."""
import itertools
import json
import time
from pathlib import Path

import numpy as np

from research_20261103_routed_boundary_views import (
    OUT,load,compact,variant,gu_cost,oracle,BITS,native_policy,
)
from research_20261103_topology_placement import sha
from research_20261103_topology_placement_replay import Replay,load_python

PROFILE=np.full((560,16),75,np.int64)
for j,high in enumerate(itertools.combinations(range(16),3)):
    PROFILE[j,list(high)]=101
    low=np.flatnonzero(PROFILE[j]==75);PROFILE[j,low[:2]]=76
PROFILE_COST=gu_cost(PROFILE)


def states(base,routes,selected,delta):
    loads=[];values=[];indices=[];localbits=[]
    for q,route in enumerate(routes):
        kk=np.asarray([k for k,e in enumerate(route) if int(e) in selected],np.int32)
        bits=((np.arange(1<<len(kk))[:,None]>>np.arange(len(kk)))&1).astype(np.int64)
        ll=base[q]+bits@delta[route[kk]];score=ll.max(1)
        loads.append(ll);values.append(int(score.min()));indices.append(kk);localbits.append(bits)
    return loads,np.asarray(values),indices,localbits


def best_profile(e,base,routes,selected,delta,base_gu,cache):
    loads,current,_,_=cache;ii=np.flatnonzero((routes==e).any(1))
    scores=np.zeros(560,np.int64)
    dd=PROFILE_COST-gu_cost(base_gu[e])
    for q in ii:
        alt=(loads[q][None]+dd[:,None]).max(2).min(1)
        scores+=np.minimum(alt,current[q])-current[q]
    best=int(scores.argmin())
    return best,int(scores[best])


def fit_whole(base,routes,base_gu,budget,frequency_only=False):
    delta=np.zeros((512,16),np.int64);selected=set();profiles={};moves=[]
    freq=np.bincount(routes.ravel(),minlength=512)
    candidates=np.flatnonzero(freq).tolist()
    frequency=sorted(candidates,key=lambda e:(-freq[e],e))[:budget]
    for step in range(budget):
        cache=states(base,routes,selected,delta)
        eligible=[frequency[step]] if frequency_only else [e for e in candidates if e not in selected]
        options=[]
        for e in eligible:
            p,change=best_profile(e,base,routes,selected,delta,base_gu,cache)
            options.append((change,e,p))
        change,e,p=min(options)
        selected.add(e);profiles[e]=p;delta[e]=PROFILE_COST[p]-gu_cost(base_gu[e])
        moves.append(dict(expert=e,profile=p,change_sum_ns=change,frequency=int(freq[e])))
    # Matched load-aware retargeting of full-GU backup locations, not only copies
    # using the proposed packet edge; every3-high mask is a legal full copy.
    for sweep in range(2):
        changed=0
        for e in sorted(selected):
            others=selected-{e};cache=states(base,routes,others,delta)
            oldvalue=int(states(base,routes,selected,delta)[1].sum())
            p,_=best_profile(e,base,routes,others,delta,base_gu,cache)
            new=PROFILE_COST[p]-gu_cost(base_gu[e]);old=delta[e].copy();delta[e]=new
            newvalue=int(states(base,routes,selected,delta)[1].sum())
            if newvalue<oldvalue:profiles[e]=p;changed+=1
            else:delta[e]=old
        if not changed:break
    cache=states(base,routes,selected,delta);loads,work,indices,bits=cache
    querybits=np.zeros((len(routes),10),np.int8)
    for q in range(len(routes)):
        which=int(loads[q].max(1).argmin());querybits[q,indices[q]]=bits[q][which]
    return selected,profiles,delta,querybits,work,moves


def whole_native(native,compacts,routes,profiles,querybits,service):
    values=[];local=np.arange(10,dtype=np.int32)[None]
    for q,route in enumerate(routes):
        pp=dict(compacts[q]);pp['jobs']=compacts[q]['jobs'].copy()
        for k,e in enumerate(route):
            if not querybits[q,k]:continue
            rr=PROFILE[profiles[int(e)]];lo=pp['offsets'][k,0,0]
            pp['jobs'][lo:lo+16,1]=rr
            pp['jobs'][lo:lo+16,3]=(rr*2560+16383)//16384
            pp['jobs'][lo:lo+16,4]=(pp['jobs'][lo:lo+16,3]+3)//4
        values.append(int(native.run(pp,local,service)[0][0]))
    return np.asarray(values,np.int64)


def main():
    start=time.monotonic();dest=OUT/'L0';screen=json.loads((dest/'screen.json').read_text())
    z=np.load(dest/'screen.npz');rec,art,plan,rows,costs,routes,base=load(0)
    native=Replay();backend=load_python();edges=z['learned__edges'];static=z['learned__static_bits']
    original=[compact(plan,r) for r in routes]
    # Strong one-copy static repack is the common base for whole-GU controls.
    compacts=[variant(original[q],route,edges,static[route]) for q,route in enumerate(routes)]
    base_gu=rows[:512,0].copy()
    for e,(s,t) in enumerate(edges):
        if static[e]:base_gu[e,s],base_gu[e,t]=base_gu[e,t],base_gu[e,s]
    static_delta=gu_cost(base_gu)-gu_cost(rows[:512,0]);base=base+static_delta[routes].sum(1)
    budget=min(screen['cases']['learned']['whole_GU_copy_budget_payload'],screen['cases']['learned']['whole_GU_copy_budget_pages'])
    records={};arrays={};checks=[]
    for name,frequency in [('frequency_whole_GU',True),('greedy_whole_GU',False)]:
        selected,profiles,delta,qbits,work,moves=fit_whole(base,routes,base_gu,budget,frequency)
        value=whole_native(native,compacts,routes,profiles,qbits,rec['service'])
        records[name]=dict(selected_experts=sorted(selected),profiles={str(e):p for e,p in profiles.items()},
                           fitting='TRAIN exact mandatory-work marginal selection +two coordinate retarget sweeps;560 arbitrary3-high whole-GU profiles.',
                           mean_mandatory_ns=float(work.mean()+2000),mean_native_ns=float(value.mean()),p99_native_ns=float(np.percentile(value,99)),
                           extra_coefficient_bytes=budget*3276800,extra_programmed_page_bytes=budget*51*65536,
                           mean_gain_vs_static_pct=100*(1-value.mean()/screen['cases']['learned']['mean_static_native_ns']),
                           additions=moves)
        arrays[name+'__delta']=delta;arrays[name+'__query_bits']=qbits;arrays[name+'__work']=work;arrays[name+'__cost']=value
        # Independently compare native descriptors to direct graph for3 queries.
        for q in [0,31,63]:
            pp=dict(compacts[q]);pp['jobs']=compacts[q]['jobs'].copy()
            for k,e in enumerate(routes[q]):
                if not qbits[q,k]:continue
                rr=PROFILE[profiles[int(e)]];lo=pp['offsets'][k,0,0]
                pp['jobs'][lo:lo+16,1]=rr;pp['jobs'][lo:lo+16,3]=(rr*2560+16383)//16384
                pp['jobs'][lo:lo+16,4]=(pp['jobs'][lo:lo+16,3]+3)//4
            got=backend.ReferenceGraph.build(pp,np.arange(11),backend.Service(**rec['service'])).run()
            assert got['cost']==value[q];checks.append(dict(case=name,query=q,native_ns=int(value[q])))
    native_oracles=[]
    for q in [0,63]:
        values=[]
        for bits in BITS:
            pp=variant(original[q],routes[q],edges,bits)
            values.append(int(native.run(pp,np.arange(10,dtype=np.int32)[None],rec['service'])[0][0]))
        values=np.asarray(values,np.int64);arrays[f'query{q}__all1024_native_costs']=values
        chosen=int(z['learned__dynamic_state'][q]);best=int(values.argmin())
        native_oracles.append(dict(query=q,all_native_assignments=1024,work_selector_state=chosen,
                                   work_selector_native_ns=int(values[chosen]),exact_native_best_state=best,
                                   exact_native_best_ns=int(values[best]),selector_gap_ns=int(values[chosen]-values[best])))
    packet=screen['cases']['learned'];bestwhole=min(r['mean_native_ns'] for r in records.values())
    result=dict(scope='L0 TRAIN64 only; no new data/fitting outside frozenTRAIN; no heldout.',
                same_capacity=dict(packet_extra_coefficient_bytes=packet['duplicate_coefficient_bytes'],packet_extra_programmed_page_bytes=packet['extra_programmed_page_bytes'],
                                   full_GU_copies=budget,GU_only_no_down_copies=True,unused_whole_GU_page_budget_bytes=packet['extra_programmed_page_bytes']-budget*51*65536),
                base='Same strongest zero-copy static repack from learned A/B views, not the older phase-only plan.',
                controls=records,native_oracle_checks=native_oracles,python_checks=checks,
                packet_mean_native_ns=packet['mean_dynamic_native_ns'],packet_vs_best_whole_mean_gain_pct=100*(1-packet['mean_dynamic_native_ns']/bestwhole),
                elapsed_seconds=time.monotonic()-start,script_sha256=sha(__file__),
                limits='Whole-copy location/ID fit is greedy+coordinate, not globally optimal. All selectors have uncharged host planning/scatter; no new selector/replication novelty. Native row-alias ABI and numeric proof remain external prerequisites.')
    np.savez_compressed(dest/'controls.npz',**arrays);(dest/'controls.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['controls']},indent=2))
    print(json.dumps({k:{j:v for j,v in r.items() if j in ['mean_native_ns','p99_native_ns','mean_gain_vs_static_pct','mean_mandatory_ns']} for k,r in records.items()},indent=2))


if __name__=='__main__':main()
