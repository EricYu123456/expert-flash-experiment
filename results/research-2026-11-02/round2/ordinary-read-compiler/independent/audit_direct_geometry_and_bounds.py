#!/usr/bin/env python3
"""Independent row-partition, physical-interval, trace and finite-pool audit.

No old campaigns/routes are rerun. The guarded source snapshot is executable
provenance; direct integer partitions are the row-splitting oracle.
"""
import copy
import hashlib
import itertools
import json
import sys
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime,timezone
from functools import lru_cache
from pathlib import Path

import numpy as np

OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT/'guarded-snapshots'))
import research_20261103_ordinary_read_backend as backend
import research_20261103_ordinary_read_compile as compiler


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ceildiv(a,b):return (a+b-1)//b


@lru_cache(None)
def row_partitions(rows,max_row,least=1):
    if rows==0:return ((),)
    answer=[]
    for first in range(least,min(rows,max_row)+1):
        answer.extend((first,)+tail for tail in row_partitions(rows-first,max_row,first))
    return tuple(answer)


def independent_nodes(plan,chosen,s):
    """Required serial operations/edges, without running an event scheduler."""
    nodes=[];locks=[]
    def add(resource,duration,deps=(),phase=0,expert=-1,chip=-1,job=-1):
        index=len(nodes);nodes.append(dict(resource=resource,duration=duration,deps=tuple(deps),phase=phase,expert=expert,chip=chip,job=job));return index
    def transfer(n,rate):return ceildiv(n*10**9,rate)
    def matrix(ei,phase,d,submit,upload):
        chip,r,c,pages,waves=map(int,d);job=len(locks)
        assert pages==ceildiv(r*c,s.page_bytes) and waves==ceildiv(pages,s.planes)
        assert waves<=s.block_wordlines
        emitted=r if s.output_fringe==0 else (pages*s.page_bytes//c if s.output_fringe==1 else waves*s.wave_bytes//c)
        cmd=add(1,s.command_ns,[submit],phase,ei,chip,job)
        inp=add(3+chip%s.channels,transfer(c*s.input_bytes,s.channel_bytes_per_second),[cmd,upload],phase,ei,chip,job)
        comp=add(3+s.channels+chip,waves*s.ordinary_read_ns,[inp],phase,ei,chip,job)
        end=add(3+chip%s.channels,transfer(emitted*s.output_bytes,s.channel_bytes_per_second),[comp],phase,ei,chip,job)
        host=add(2,transfer(emitted*s.output_bytes,s.pcie_bytes_per_second),[end],phase,ei)
        locks.append((chip,cmd,end))
        return host
    submit=add(0,s.host_submit_ns)
    upload=add(2,transfer(int(plan['input_columns'][0,0])*s.input_bytes,s.pcie_bytes_per_second),[submit])
    barriers=[]
    for ei,e in enumerate(chosen):
        start,end=plan['offsets'][e,0]
        complete=[matrix(ei,0,d,submit,upload) for d in plan['jobs'][start:end]]
        barriers.append(add(s.resources-1,0,complete,expert=ei))
    complete=[]
    for ei,e in enumerate(chosen):
        activation=add(0,s.activation_ns,[barriers[ei]],1,ei)
        ds=add(0,s.host_submit_ns,[activation],1,ei)
        up=add(2,transfer(int(plan['input_columns'][e,1])*s.input_bytes,s.pcie_bytes_per_second),[ds],1,ei)
        start,end=plan['offsets'][e,1]
        complete.extend(matrix(ei,1,d,ds,up) for d in plan['jobs'][start:end])
    add(0,s.combine_ns,complete,1)
    return nodes,locks


def audit_trace(plan,chosen,s,trace,reported_bound):
    nodes,locks=independent_nodes(plan,chosen,s)
    assert trace.shape==(len(nodes),8)
    work=[0]*s.resources;chipwork=[0]*s.chips;critical=[];resource_intervals=defaultdict(list)
    for i,n in enumerate(nodes):
        r,d,start,end,phase,expert,chip,job=map(int,trace[i])
        assert (r,d,phase,expert,chip,job)==tuple(n[k] for k in ['resource','duration','phase','expert','chip','job'])
        assert end-start==d and start>=0
        assert all(int(trace[p,3])<=start for p in n['deps'])
        work[r]+=d
        if job>=0:chipwork[chip]+=d
        critical.append(d+max((critical[p] for p in n['deps']),default=0))
        if d:resource_intervals[r].append((start,end))
    for spans in resource_intervals.values():
        spans.sort();assert all(a[1]<=b[0] for a,b in zip(spans,spans[1:]))
    owners=defaultdict(list)
    for chip,begin,end in locks:owners[chip].append((int(trace[begin,2]),int(trace[end,3])))
    for spans in owners.values():
        spans.sort();assert all(a[1]<=b[0] for a,b in zip(spans,spans[1:]))
    lb=max(max(work),max(critical),s.host_submit_ns+max(chipwork),s.host_submit_ns+work[1])
    assert lb==reported_bound and int(trace[:,3].max())>=lb
    return dict(nodes=len(nodes),jobs=len(locks),lower_bound=lb)


def audit_physical(plan,s):
    physical=backend.physical_addresses(plan,s)
    independent_blocks=[0]*s.chips;intervals=defaultdict(list)
    for j,a,d in zip(plan['logical'],physical['addresses'],plan['jobs']):
        assert j['chip']==a['chip']==int(d[0])
        assert a['first_plane']==0 and a['reserved_wordlines']==j['waves']
        begin=a['wordline']*s.wave_bytes
        reserved=j['waves']*s.wave_bytes;useful=j['rows']*j['columns']
        assert begin%s.wave_bytes==0 and useful<=j['pages']*s.page_bytes<=reserved
        assert begin+reserved<=s.block_wordlines*s.wave_bytes
        intervals[a['chip'],a['block']].append((begin,begin+reserved))
        independent_blocks[a['chip']]=max(independent_blocks[a['chip']],a['block']+1)
    for spans in intervals.values():
        spans.sort();assert all(a[1]<=b[0] for a,b in zip(spans,spans[1:]))
    for e in range(len(plan['offsets'])):
        for phase in (0,1):
            start,end=plan['offsets'][e,phase];cover=[]
            for j in plan['logical'][start:end]:cover.extend(range(j['row_start'],j['row_start']+j['rows']))
            assert cover==list(range(int(plan['total_rows'][e,phase])))
    assert independent_blocks==physical['chip_blocks']
    backend.physical_addresses(plan,s,chip_blocks=independent_blocks)
    smaller=independent_blocks.copy();hot=next(i for i,n in enumerate(smaller) if n);smaller[hot]-=1
    try:backend.physical_addresses(plan,s,chip_blocks=smaller)
    except AssertionError:pass
    else:raise AssertionError('insufficient block budget accepted')
    return max(independent_blocks),len(plan['jobs'])


def main():
    native=backend.NativeReplay(OUT/'guarded-build')
    split_cases=0;partition_classes=0;reject_rows=0
    for b,c,depth,r in itertools.product(range(1,7),range(1,11),range(1,6),range(1,15)):
        maximum=depth*b//c
        s=backend.Service(chips=2,channels=1,planes=1,page_bytes=b,block_wordlines=depth)
        if not maximum:
            try:backend.split_complete_rows(r,c,s)
            except AssertionError:reject_rows+=1;continue
            raise AssertionError('row exceeding a block accepted')
        parts=row_partitions(r,maximum);partition_classes+=len(parts)
        exact=min((len(p),sum(ceildiv(n*c,b) for n in p)) for p in parts)
        got=backend.split_complete_rows(r,c,s)
        assert sum(got)==r and all(0<n<=maximum for n in got)
        assert (len(got),sum(ceildiv(n*c,b) for n in got))==exact
        split_cases+=1
    rng=np.random.default_rng(2026110317);fuzz=[]
    for index in range(128):
        chips=[1,2,4,6][index%4];channels=int(rng.choice([c for c in range(1,chips+1) if chips%c==0]))
        planes=int(rng.integers(1,5));page=int(rng.choice([3,8,16,64]));block=int(rng.choice([1,3,7,192]))
        capacity=planes*page*block;h=int(rng.integers(1,min(71,capacity)+1));i=int(rng.integers(1,min(37,capacity)+1));si=int(rng.integers(1,min(53,capacity)+1))
        experts=3;shared=index%3;count=experts+shared
        gu=np.asarray([rng.multinomial(2*(i if e<experts else si),np.full(chips,1/chips)) for e in range(count)])
        down=np.asarray([rng.multinomial(h,np.full(chips,1/chips)) for _ in range(count)])
        s=backend.Service(chips=chips,channels=channels,planes=planes,page_bytes=page,block_wordlines=block,
           input_bytes=[1,2,4][index%3],output_bytes=[1,4,8][index%3],
           ordinary_read_ns=0 if index%5==0 else 17,command_ns=0 if index%4==0 else 3,
           host_submit_ns=0 if index%3==0 else 5,activation_ns=0 if index%2==0 else 7,combine_ns=0 if index%2==0 else 11,
           pcie_bytes_per_second=3_000_000_000,channel_bytes_per_second=1_300_000_000,
           output_fringe=index%3,down_first=index%2)
        plan=backend.lower_rows(gu,down,h,[i]*experts+[si]*shared,s)
        blocks,jobs=audit_physical(plan,s)
        active=1+index%3;routes=np.asarray([rng.choice(experts,active,replace=False) for _ in range(2)],dtype=np.int32)
        got=native.run(plan,routes,experts,shared,s,trace=True)
        chosen=np.r_[routes[0],np.arange(experts,experts+shared)]
        checked=audit_trace(plan,chosen,s,got['trace'],int(got['lower_bound'][0]))
        fuzz.append(dict(index=index,service=asdict(s),hidden=h,intermediate=i,shared_intermediate=si,
                         shared=shared,lowered_jobs=jobs,max_chip_blocks=blocks,**checked))
    shapes=[]
    for name,(h,i,_,_,_,si) in compiler.SHAPES.items():
        s=backend.Service(output_fringe=1,input_buffer_bytes=32768)
        count=2+int(bool(si));gu=np.zeros((count,16),dtype=np.int64);down=gu.copy()
        gu[:,0]=[2*i]*2+([2*si] if si else []);down[:,0]=h
        plan=backend.lower_rows(gu,down,h,[i]*2+([si] if si else []),s)
        blocks,jobs=audit_physical(plan,s);got=native.run(plan,[[0],[1]],2,int(bool(si)),s,True)
        checked=audit_trace(plan,np.r_[0,np.arange(2,count)],s,got['trace'],int(got['lower_bound'][0]))
        shapes.append(dict(model=name,full_original_rows_on_one_chip=True,lowered_jobs=jobs,max_chip_blocks=blocks,**checked))
    # The actual reported imported-plan bug must now reject before entering C++.
    s=backend.Service(chips=1,channels=1,planes=1,page_bytes=1024,block_wordlines=2,
       ordinary_read_ns=0,command_ns=0,host_submit_ns=0,pcie_bytes_per_second=10**9,channel_bytes_per_second=10**9)
    good=backend.lower_rows([[2]],[[1024]],1024,1,s);bad=copy.deepcopy(good);bad['input_columns'][0,0]=1
    rejected=[]
    cases={'wrong_input_columns':bad}
    for label,change in [('offset_gap',lambda p:p['offsets'].__setitem__((0,1,0),0)),
                          ('wrong_pages',lambda p:p['jobs'].__setitem__((0,3),1)),
                          ('wrong_wave_count',lambda p:p['jobs'].__setitem__((0,4),1)),
                          ('out_of_range_chip',lambda p:p['jobs'].__setitem__((0,0),1))]:
        p=copy.deepcopy(good);change(p);cases[label]=p
    for name,p in cases.items():
        try:native.run(p,[[0]],1,0,s)
        except AssertionError as error:rejected.append(dict(case=name,rejected=True,exception=str(error)))
        else:raise AssertionError(name+' accepted')
    try:backend.lower_rows([[2]],[[1024]],1024,1,backend.Service(chips=1,channels=1,planes=1,page_bytes=1024,block_wordlines=2,input_buffer_bytes=512))
    except AssertionError as error:rejected.append(dict(case='input_buffer_too_small',rejected=True,exception=str(error)))
    else:raise AssertionError('input buffer overflow accepted')
    # Tiny fully replayed fitted pool: exact means use integer sums; no fitter
    # or root's bound is used as a cost oracle.
    s=backend.Service(channels=4,planes=2,page_bytes=8,block_wordlines=3,
       ordinary_read_ns=7,command_ns=2,host_submit_ns=3,activation_ns=1,combine_ns=2)
    shape=(4,3,3,2,1,2);routes=np.asarray([[0,1],[1,2],[2,0]],dtype=np.int32)
    def candidate(label,family,base,labels):
        variants=np.asarray([np.roll(base,v) for v in range(16)],dtype=np.int64)
        c=dict(label=label,family=family,variants=variants,labels=np.asarray(labels),rows=variants[labels])
        return c
    gu=[candidate('g_disjoint','disjoint',[3,3]+[0]*14,[0,4,8]),candidate('g_cyclic','cyclic',[2,2,2]+[0]*13,[1,5,9]),candidate('g_whole','whole_rows',[2,2,1,1]+[0]*12,[2,6,10])]
    down=[candidate('d_disjoint','disjoint',[4]+[0]*15,[0,4,8]),candidate('d_cyclic','cyclic',[2,1,1]+[0]*13,[1,5,9]),candidate('d_whole','whole_rows',[1,1,1,1]+[0]*12,[2,6,10])]
    for p,cols in [(gu,4),(down,3)]:
        for c in p:c['costs']=compiler.candidate_cost(c,cols,s)
    shared=compiler.shared_rows(4,2,s);complete={}
    for gi,di,shift in itertools.product(range(3),range(3),range(16)):
        p=compiler.make_plan(gu[gi],down[di],shift,shape,shared,s)
        v=native.run(p,routes,3,1,s)['cost'];complete[gi,di,shift]=int(v.sum())
    pool=[]
    for budget in (1,1000):
        reports,_=compiler.select_pairs(native,gu,down,routes,shape,shared,s,budget)
        for report in reports:
            allowed={'disjoint'} if report['family']=='disjoint' else ({'disjoint','cyclic'} if report['family']=='cyclic' else {'disjoint','cyclic','whole_rows'})
            exact=min(v for (g,d,_),v in complete.items() if gu[g]['family'] in allowed and down[d]['family'] in allowed)
            chosen=complete[tuple(report['key'])]
            if report['finite_pool_certified_best']:assert chosen==exact
            assert chosen>=exact
            pool.append(dict(budget=budget,family=report['family'],certified=report['finite_pool_certified_best'],
                             exact_mean_ns=exact/len(routes),chosen_mean_ns=chosen/len(routes),
                             exhausted=report['replay_budget_exhausted']))
    # Explicit BFD packing counterexample is policy scope, not invalid addresses.
    s=backend.Service(chips=1,channels=1,planes=1,page_bytes=1,block_wordlines=10)
    p=backend.lower_rows([[5],[3],[3]],[[4],[3],[2]],1,1,s)
    packing=backend.physical_addresses(p,s)
    assert packing['chip_blocks']==[3]
    report=dict(status='PASS independent guarded-backend geometry/trace/LB/fitted-pool audit',
      created_utc=datetime.now(timezone.utc).isoformat(),
      guarded_sources=json.loads((OUT/'guarded_sources.frozen.json').read_text()),
      build=native.provenance,split_cases=split_cases,direct_partition_classes=partition_classes,
      too_large_row_rejections=reject_rows,synthetic_graphs=len(fuzz),synthetic_costs=2*len(fuzz),fuzz=fuzz,
      seven_original_shapes=shapes,malformed_rejections=rejected,
      tiny_finite_pool=dict(pairs=144,queries=3,fully_replayed_costs=432,certificates=pool),
      BFD_counterexample=dict(capacity=10,waves=[5,4,3,3,3,2],emitted_blocks=3,feasible_two_blocks=[[5,3,2],[4,3,3]],scope='BFD allocation is a legal heuristic; rejected BFD capacity does not certify all packings infeasible.'),
      scope='No real route/model campaigns rerun. Direct integer partitions are independent of capacity DP; trace audit checks mandatory operations/edges/locks and recomputes LB without the Python event scheduler. Numerical PE outputs and real-device contracts remain outside this software audit.')
    path=OUT/'geometry_bounds_audit.json';path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('fuzz','guarded_sources','build')},indent=2))


if __name__=='__main__':main()
