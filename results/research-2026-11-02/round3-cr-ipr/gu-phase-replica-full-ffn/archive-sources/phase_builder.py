#!/usr/bin/env python3
"""Physical two-phase GU replica builder; fixed0123 ECC only.

API returns actual resident addresses and complete logical page-source identity.
The existing conditional pre-ECC drop and synchronous IPR premise are explicit.
"""
from collections import Counter
from copy import deepcopy
from functools import lru_cache

PAGE=16384
PHASE0_ONLY=(0,1,2)
PHASE2_ONLY=(3,4,5)

@lru_cache(None)
def _layout():
    next_wl=[0]*4;resident=[];copies={}
    for expert in range(64):
        phases=[0] if expert in PHASE0_ONLY else [2] if expert in PHASE2_ONLY else [0,2]
        copies[expert]={}
        for phase in phases:
            pages=[]
            for j in range(6):
                plane=(phase+j)%4;wl=next_wl[plane];next_wl[plane]+=1
                p=dict(page_id=f'e{expert}.p{j}',physical_page_id=f'e{expert}.c{phase}.p{j}',
                    owner=f'e{expert}',expert=expert,copy_phase=phase,page_index=j,
                    source_page_id=f'e{expert}.p{j}',source_byte_offset=j*PAGE,
                    source_row_bytes=1536,plane=plane,block=0,wl=wl,bytes=PAGE)
                pages.append(p);resident.append(p)
            copies[expert][phase]=pages
    assert next_wl==[183]*4
    shared=[]
    for j in range(12):
        plane=j%4;wl=next_wl[plane];next_wl[plane]+=1
        p=dict(page_id=f's.p{j}',physical_page_id=f's.p{j}',owner='s',expert=64,
            copy_phase=None,page_index=j,source_page_id=f's.p{j}',source_byte_offset=j*PAGE,
            source_row_bytes=1536,plane=plane,block=0,wl=wl,bytes=PAGE)
        shared.append(p);resident.append(p)
    assert next_wl==[186]*4 and len(resident)==744
    assert len({(p['plane'],p['block'],p['wl']) for p in resident})==744
    return dict(phase0_only=list(PHASE0_ONLY),phase2_only=list(PHASE2_ONLY),copies=copies,
                shared_pages=shared,resident_pages=resident,WL_per_plane=next_wl,
                routed_copy_instances=122,routed_extra_copy_instances=58)

def build_layout():
    """Full per-chip GU ledger (same address map on all16 chip shards)."""
    return deepcopy(_layout())

def assign_phases(route):
    ids=sorted(map(int,route))
    if len(ids)!=6 or len(set(ids))!=6 or ids[0]<0 or ids[-1]>=64:
        raise ValueError('A route must contain six distinctexpertIDs0..63.')
    a=[e for e in ids if e in PHASE0_ONLY];b=[e for e in ids if e in PHASE2_ONLY]
    dual=[e for e in ids if e not in PHASE0_ONLY and e not in PHASE2_ONLY]
    a+=dual[:3-len(a)];b+=dual[3-len([e for e in ids if e in PHASE0_ONLY]):]
    a.sort();b.sort();assert len(a)==len(b)==3 and set(a+b)==set(ids)
    chosen={e:0 for e in a}|{e:2 for e in b}
    order=[e for pair in zip(a,b) for e in pair]
    return dict(phase_by_expert=chosen,phase0_experts=a,phase2_experts=b,expert_order=order)

def resident_pool(route):
    selected=set(map(int,route));pool=[]
    for source in _layout()['resident_pages']:
        if source['expert'] not in selected and source['expert']!=64:
            p=dict(source);p['page_id']='drop.'+p['physical_page_id'];p['unselected']=True
            p['dummy_unselected_expert']=p['expert'];pool.append(p)
    return pool

def _plain(route):
    assignment=assign_phases(route);L=_layout();stream=[]
    for e in assignment['expert_order']:stream += deepcopy(L['copies'][e][assignment['phase_by_expert'][e]])
    stream += deepcopy(L['shared_pages'])
    assert len(stream)==48 and [p['plane'] for p in stream]==list(range(4))*12
    waves=[dict(pages=stream[i:i+4],read_pages=stream[i:i+4],source_order=[0,1,2,3]) for i in range(0,48,4)]
    return waves,assignment

def _split(waves,pool):
    """Fixed originalwave10/prefix1, samefour-dummy conditional lowering."""
    out=[];used=set()
    for wi,w in enumerate(waves):
        chunks=[w['pages'][:1],w['pages'][1:]] if wi==10 else [w['pages']]
        for pages in chunks:
            read=list(pages)
            for plane in sorted(set(range(4))-{p['plane'] for p in pages}):
                p=next(p for p in pool if p['plane']==plane and p['physical_page_id'] not in used)
                used.add(p['physical_page_id']);read.append(deepcopy(p))
            out.append(dict(pages=pages,read_pages=read,source_order=[0,1,2,3],original_wave=wi,shaping_split=wi==10))
    return out

def certify(route,waves,assignment):
    expected=[p for e in assignment['expert_order'] for p in _layout()['copies'][e][assignment['phase_by_expert'][e]]]+_layout()['shared_pages']
    actual=[p for w in waves for p in sorted(w['pages'],key=lambda p:p['plane'])]
    key=lambda p:(p['source_page_id'],p['physical_page_id'],p['plane'],p['block'],p['wl'])
    assert list(map(key,expected))==list(map(key,actual))
    assert [p['page_index'] for e in sorted(route) for p in actual if p['expert']==e]==list(range(6))*6
    assert Counter(p['expert'] for p in actual)==Counter({**{int(e):6 for e in route},64:12})
    assert all(p['source_page_id']==p['page_id'] for p in actual)
    pool={key(p) for p in resident_pool(route)};physical=[];dummy=[]
    for w in waves:
        assert {p['plane'] for p in w['read_pages']}==set(range(4))
        useful={p['page_id'] for p in w['pages']}
        for p in w['read_pages']:
            physical.append((p['plane'],p['block'],p['wl']))
            if p['page_id'] not in useful:assert key(p) in pool;dummy.append(p)
    assert len(physical)==len(set(physical)) and len(dummy)==4*(len(waves)-12)
    return dict(status='PASS',useful_pages=48,physical_pages=len(physical),dummy_pages=len(dummy),
                fixed_source_0123=True,logical_page_source_identity=True,complete_owner_rows=True,
                routed_rows_per_chip=64,shared_rows_per_chip=128,GU_row_bytes=1536,
                all_dummy_addresses_real_unselected=True,same_block=True,no_extra_PEselector=True)

def build_fixedphase_plain(route):
    waves,assignment=_plain(route)
    return dict(waves=waves,assignment=assignment,owner_ranks={f'e{e}':i for i,e in enumerate(sorted(map(int,route)))}|{'s':6},
                metadata=certify(route,waves,assignment),pool=resident_pool(route))

def build_fixedphase_split(route):
    """Primary strict500 kernel; returnwaves plusassignment/pool/scattermetadata."""
    waves,assignment=_plain(route);pool=resident_pool(route);waves=_split(waves,pool)
    return dict(waves=waves,assignment=assignment,owner_ranks={f'e{e}':i for i,e in enumerate(sorted(map(int,route)))}|{'s':6},
                metadata=certify(route,waves,assignment),pool=pool)
