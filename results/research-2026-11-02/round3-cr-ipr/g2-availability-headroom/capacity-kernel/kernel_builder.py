#!/usr/bin/env python3
"""Standard complete-row g2 Down copy bands under a GU<=128WL budget."""
from collections import Counter
from itertools import combinations

PAGE=16384

def canonical_supports():
    supports=[list(c) for c in combinations(range(6),4) for _ in range(4)]
    supports += [[0,1,2],[3,4,5],[0,3,4],[1,2,5]]
    assert len(supports)==64
    return supports

def build_layout(supports=None,gu_reserved_wl=128):
    """Accept a frozen expert-label permutation of the canonical support map."""
    supports=canonical_supports() if supports is None else [sorted(map(int,s)) for s in supports]
    if len(supports)!=64 or any(len(s)!=len(set(s)) or min(s)<0 or max(s)>=6 for s in supports):
        raise ValueError('Supply64nonemptyexpertgroup-supportlistswithgroupIDs0..5.')
    counts=Counter(g for s in supports for g in s)
    if counts!=Counter({g:42 for g in range(6)}):raise ValueError('Eachgroupmustcontainexactly42residentexpertcopies.')
    if sorted(map(len,supports))!=[3]*4+[4]*60:raise ValueError('Expected60degree4and4degree3expertcopies.')
    if not 0<=gu_reserved_wl<=192:raise ValueError('GUreservationmustbe0..192WL.')
    bands0=(192-gu_reserved_wl)//6
    if bands0+32<42:
        raise ValueError(f'Local2blockcapacityfails:GU{gu_reserved_wl}+Down252WL>{384};only{bands0+32}whole6WLjobsfit.')
    groups=[]
    for g in range(6):
        ids=[e for e,s in enumerate(supports) if g in s];jobs=[]
        for rank,e in enumerate(ids):
            block=0 if rank<bands0 else 1
            wl=(gu_reserved_wl+6*rank) if block==0 else 6*(rank-bands0)
            assert 0<=wl and wl+6<=192
            jobs.append(dict(expert=e,group=g,block=block,WL_start=wl,WL_count=6,pages=24,rows=768,columns=512,rank=rank,commands=1,cold_starts=1))
        groups.append(dict(group=g,chips=[2*g,2*g+1],resident_experts=ids,jobs=jobs))
    return dict(supports=supports,GU_reserved_WL=gu_reserved_wl,groups=groups,copy_instances=252,
                Down_WL_per_private_chip=252,Down_pages_per_private_chip=1008,
                private_Down_MiB=189.0,shared_Down_MiB=1.5,weight_bits=8,activation_bits=8)

def build_private_job(expert,group,local_chip=0,layout=None):
    """24pages/6native same-WL waves, complete768rows ononechip."""
    assert local_chip in (0,1)
    L=build_layout() if layout is None else layout
    job=next((j for j in L['groups'][group]['jobs'] if j['expert']==expert),None)
    if job is None:raise ValueError('Expertcopyisnotresidentinthechosengroup.')
    pages=[]
    for j in range(24):
        pages.append(dict(page_id=f'e{expert}.p{j}',owner=f'e{expert}',page_index=j,expert=expert,
            plane=j%4,block=job['block'],wl=job['WL_start']+j//4,bytes=PAGE,
            source_row_start=local_chip*768,source_byte_offset=local_chip*768*512+j*PAGE,
            source_row_bytes=512,copy_group=group,physical_chip=2*group+local_chip))
    waves=[dict(pages=pages[i:i+4],read_pages=pages[i:i+4],source_order=[0,1,2,3]) for i in range(0,24,4)]
    return dict(waves=waves,metadata=dict(job,physical_chip=2*group+local_chip,source_row_start=local_chip*768,
            source_row_end=(local_chip+1)*768,row_bytes=512,fixed_source_0123=True,sameWL=True,requires_IPR=False))

def build_shared_job(local_chip=0,gu_reserved_wl=128):
    assert 0<=local_chip<4 and gu_reserved_wl+6<=192
    pages=[dict(page_id=f's.p{j}',owner='s',expert=64,page_index=j,plane=j%4,block=0,
        wl=gu_reserved_wl+j//4,bytes=PAGE,source_row_start=local_chip*384,
        source_byte_offset=local_chip*384*1024+j*PAGE,source_row_bytes=1024,physical_chip=12+local_chip) for j in range(24)]
    waves=[dict(pages=pages[i:i+4],read_pages=pages[i:i+4],source_order=[0,1,2,3]) for i in range(0,24,4)]
    return dict(waves=waves,metadata=dict(expert=64,pages=24,rows=384,columns=1024,row_bytes=1024,
        physical_chip=12+local_chip,source_row_start=local_chip*384,source_row_end=(local_chip+1)*384,
        commands=1,cold_starts=1,block=0,WL_start=gu_reserved_wl,WL_count=6,sameWL=True,requires_IPR=False))

def assign_route(route,supports=None,capacity=1):
    """Deterministic ordinary augmenting-path matching, not a newsolver."""
    ids=sorted(map(int,route));supports=canonical_supports() if supports is None else supports
    assert len(ids)==6 and len(set(ids))==6 and all(0<=e<64 for e in ids) and capacity in (1,2)
    occupants={};assignment={}
    def augment(e,seen):
        for g in sorted(supports[e]):
            for slot in range(capacity):
                key=(g,slot)
                if key in seen:continue
                seen.add(key)
                if key not in occupants or augment(occupants[key],seen):
                    occupants[key]=e;assignment[e]=g;return True
        return False
    for e in ids:
        if not augment(e,set()):return dict(feasible=False,capacity=capacity,partial_assignment=assignment,route=ids)
    assert all(assignment[e] in supports[e] for e in ids)
    load=Counter(assignment.values());assert max(load.values())<=capacity
    return dict(feasible=True,capacity=capacity,assignment=assignment,group_load=[load[g] for g in range(6)],route=ids)
