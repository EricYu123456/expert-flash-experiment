#!/usr/bin/env python3
"""Standard balanced degree4 CBC with movable shared-Down row halves.

No numerical or full-FFN execution. Importable joint layout/job/dispatch API.
"""
from collections import Counter
import datetime
import hashlib
from itertools import combinations
import json
from pathlib import Path

PAGE=16384
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/research-2026-11-02/round3-cr-ipr/eight-pair-cbc-degree4'
REMOVED=((0,1,2,3),(4,5,6,7),(0,2,4,6),(1,3,5,7),(0,1,4,5),(2,3,6,7))


def canonical_supports():
    return [list(s) for s in combinations(range(8),4) if s not in REMOVED]


def build_layout(supports=None,gu_reserved_wl=128):
    """64unique degree4 supports,32experts/pair;34native resident jobs/pair."""
    supports=canonical_supports() if supports is None else [sorted(map(int,s)) for s in supports]
    if len(supports)!=64 or any(len(s)!=4 or len(set(s))!=4 or min(s)<0 or max(s)>=8 for s in supports):
        raise ValueError('Require64expert supports of four distinct groups0..7.')
    if len({tuple(s) for s in supports})!=64:
        raise ValueError('Unique degree4 support sets are required for the all-top6 guarantee.')
    if Counter(g for s in supports for g in s)!=Counter({g:32 for g in range(8)}):
        raise ValueError('Eachpair must contain32routed experts.')
    if not isinstance(gu_reserved_wl,int) or not 0<=gu_reserved_wl<=192:
        raise ValueError('GU reservation must be an integer0..192WL.')
    bands0=min(34,(192-gu_reserved_wl)//6)
    if bands0+32<34:
        raise ValueError('Local two-block capacity cannot hold34whole six-WLjobs with this GU reservation.')
    groups=[]
    for g in range(8):
        experts=[e for e in range(64) if g in supports[e]];jobs=[]
        owners=[(f'e{e}',e,None,768,512) for e in experts]+[('S0',64,0,384,1024),('S1',64,1,384,1024)]
        for rank,(owner,e,half,rows,columns) in enumerate(owners):
            block=0 if rank<bands0 else 1
            wl=gu_reserved_wl+6*rank if block==0 else 6*(rank-bands0)
            if wl<0 or wl+6>192:raise ValueError('Native job crosses a block boundary.')
            jobs.append(dict(owner=owner,expert=e,shared_half=half,group=g,rank=rank,block=block,
                WL_start=wl,WL_count=6,pages=24,rows=rows,columns=columns,commands=1,cold_starts=1))
        groups.append(dict(group=g,chips=[2*g,2*g+1],resident_experts=experts,jobs=jobs))
    return dict(supports=supports,GU_reserved_WL=gu_reserved_wl,groups=groups,
        routed_copy_instances=256,shared_half_copy_instances=16,Down_WL_per_chip=204,
        routed_Down_MiB=192,shared_Down_MiB=12,weight_bits=8,activation_bits=8,
        claim='Standard CBC/replicateddeclustering baseline, not a new theorem or solver.')


def _build(owner,group,local_chip,layout):
    if not 0<=group<8 or local_chip not in (0,1):raise ValueError('group0..7/local_chip0..1 required.')
    entry=next((j for j in layout['groups'][group]['jobs'] if j['owner']==owner),None)
    if entry is None:raise ValueError('Requested copy is not resident in this pair.')
    rows,columns=entry['rows'],entry['columns'];half=entry['shared_half']
    source_start=local_chip*768 if half is None else half*768+local_chip*384
    chip=2*group+local_chip;pages=[]
    for p in range(24):
        pages.append(dict(page_id=f'{owner}.p{p}',physical_page_id=f'{owner}.g{group}.c{chip}.p{p}',
            owner=owner,expert=entry['expert'],shared_half=half,page_index=p,plane=p%4,block=entry['block'],
            wl=entry['WL_start']+p//4,bytes=PAGE,source_row_start=source_start,
            source_byte_offset=source_start*columns+p*PAGE,source_row_bytes=columns,copy_group=group,physical_chip=chip))
    waves=[dict(pages=pages[p:p+4],read_pages=pages[p:p+4],source_order=[0,1,2,3]) for p in range(0,24,4)]
    meta=dict(entry,physical_chip=chip,source_row_start=source_start,source_row_end=source_start+rows,
              row_bytes=columns,input_bytes=columns,fixed_source_0123=True,sameWL=True,requires_IPR=False,
              resident_job_id=f'{owner}.g{group}.c{chip}')
    return dict(waves=waves,metadata=meta)


def build_private_job(expert,group,local_chip=0,layout=None):
    if not 0<=int(expert)<64:raise ValueError('Privateexpert0..63 required.')
    return _build(f'e{int(expert)}',int(group),int(local_chip),build_layout() if layout is None else layout)


def build_shared_job(half,group,local_chip=0,layout=None):
    if half not in (0,1):raise ValueError('Sharedhalf must be0or1.')
    return _build(f'S{half}',int(group),int(local_chip),build_layout() if layout is None else layout)


def buildjob(owner,group,local_chip=0,layout=None):
    L=build_layout() if layout is None else layout
    if isinstance(owner,int):return build_private_job(owner,group,local_chip,L)
    if owner in ('S0','S1'):return build_shared_job(int(owner[1]),group,local_chip,L)
    if isinstance(owner,str) and owner.startswith('e'):return build_private_job(int(owner[1:]),group,local_chip,L)
    raise ValueError('Owner must be expertID/eID/S0/S1.')


def execution_view(bundle):
    """Existing FFN backend view: shared halves are both expert64,copy0."""
    m=bundle['metadata']
    return dict(expert=m['expert'],copy=0,resident_copy=m['group'],shared_half=m['shared_half'],
        resident_job_id=m['resident_job_id'],chip=m['physical_chip'],rows=m['rows'],columns=m['columns'],
        block=m['block'],wl_start=m['WL_start'],pages=24,row_start=m['source_row_start'],reserved_WL=6)


def resident_jobs(layout=None):
    L=build_layout() if layout is None else layout
    return [execution_view(buildjob(j['owner'],g,local,L)) for g in range(8)
            for j in L['groups'][g]['jobs'] for local in range(2)]


def assign_route(route,layout=None):
    """Match6distinct experts to6pairs, then place S0/S1 in the2unused pairs."""
    ids=sorted(map(int,route));L=build_layout() if layout is None else layout
    if len(ids)!=6 or len(set(ids))!=6 or min(ids)<0 or max(ids)>=64:
        raise ValueError('Supply six distinct expertIDs0..63.')
    occupants={}
    def augment(e,seen):
        for g in L['supports'][e]:
            if g in seen:continue
            seen.add(g)
            if g not in occupants or augment(occupants[g],seen):occupants[g]=e;return True
        return False
    for e in ids:
        if not augment(e,set()):return dict(feasible=False,capacity=1,route=ids)
    assignment={e:g for g,e in occupants.items()};unused=sorted(set(range(8))-set(occupants))
    if len(assignment)!=6 or len(unused)!=2:raise RuntimeError('Incomplete matching inventory.')
    halves={0:unused[0],1:unused[1]};selected=[];group_owners=[None]*8
    for e in ids:
        g=assignment[e];group_owners[g]=f'e{e}'
        selected.extend(execution_view(build_private_job(e,g,local,L)) for local in range(2))
    for half in (0,1):
        g=halves[half];group_owners[g]=f'S{half}'
        selected.extend(execution_view(build_shared_job(half,g,local,L)) for local in range(2))
    return dict(feasible=True,capacity=1,route=ids,assignment=assignment,shared_half_assignment=halves,
                group_owners=group_owners,group_load=[1]*8,selected_jobs=selected,
                input_contract='Everyroutedchip job takes fullC512; eachsharedhalfchip job takes fullC1024. No fragmented input or shared dot context.')


dispatch=assign_route


def freeze():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.frozen.json').exists():raise RuntimeError('Alreadyfrozen; do not overwrite sealed builder artifacts.')
    layout=build_layout()
    (OUT/'layout.frozen.json').write_text(json.dumps(layout,indent=2)+'\n')
    (OUT/'resident_jobs.frozen.json').write_text(json.dumps(resident_jobs(layout),indent=2)+'\n')
    h=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    protocol=dict(status='frozen_before_verification_or_FFN',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        inputs_sha256={str(p):h(p) for p in [Path(__file__).resolve(),OUT/'layout.frozen.json',OUT/'resident_jobs.frozen.json']},
        design='AllC(8,4) supports except three balanced complementary pairs;64unique supports,32experts/group,degree4.',
        removal=[list(s) for s in REMOVED],dispatch='Deterministic sortedID augmenting cap1; S0/S1 assigned to sortedunusedpairs.',
        hardware='Existing two-block/chip budget, fullC512/C1024 inputs, one rowaccumulator/nativejob, onecold sixwave sameWL fixed0123 GEMV per selectedchip.',
        timing='No fullFFN execution or measured hardware timing in this builder. Inherits conditional native24page/sixwave contract.',
        chronology='Designed after prospective failures were observed. Any later replay on previously observed prospective data is diagnosticreuse, not a new heldout evaluation.',
        novelty='Standard CBC/replicateddeclustering baseline; no newtheorem,solver,hardware or globalstorage-optimality claim.')
    (OUT/'protocol.frozen.json').write_text(json.dumps(protocol,indent=2)+'\n')
    print(json.dumps(dict(status=protocol['status'],builder_sha256=h(__file__),protocol_sha256=h(OUT/'protocol.frozen.json'))))


if __name__=='__main__':freeze()
