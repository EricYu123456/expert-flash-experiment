#!/usr/bin/env python3
"""Equal-byte standard whole-GU g8 replicas, separately frozen strong control."""
from collections import Counter
from copy import deepcopy
from functools import lru_cache
from phase_builder import assign_phases

PAGE=16384

@lru_cache(None)
def _group(group):
    assert group in (0,1)
    excluded=(3,4,5) if group==0 else (0,1,2)
    ids=[e for e in range(64) if e not in excluded];copies={};resident=[]
    for rank,e in enumerate(ids):
        pages=[]
        for j in range(12):
            p=dict(page_id=f'e{e}.p{j}',physical_page_id=f'g{group}.e{e}.p{j}',owner=f'e{e}',
                expert=e,copy_group=group,page_index=j,source_page_id=f'e{e}.g8shard.p{j}',
                source_byte_offset=j*PAGE,source_row_bytes=1536,source_shard_rows=128,
                plane=j%4,block=0,wl=3*rank+j//4,bytes=PAGE)
            pages.append(p);resident.append(p)
        copies[e]=pages
    shared=[]
    for j in range(12):
        p=dict(page_id=f's.p{j}',physical_page_id=f'g{group}.s.p{j}',owner='s',expert=64,
            copy_group=group,page_index=j,source_page_id=f's.g16shard.p{j}',source_byte_offset=j*PAGE,
            source_row_bytes=1536,source_shard_rows=128,plane=j%4,block=0,wl=183+j//4,bytes=PAGE)
        shared.append(p);resident.append(p)
    assert len(resident)==744 and len({(p['plane'],p['wl']) for p in resident})==744
    return dict(group=group,chips=list(range(8*group,8*group+8)),exclusive_experts=[0,1,2] if group==0 else [3,4,5],
                resident_experts=ids,copies=copies,shared_pages=shared,resident_pages=resident,
                WL_per_plane=[186]*4,routed_rows_per_chip=128,shared_rows_per_chip=128)

def build_layout(group=None):
    """Templates correspond to chiplocalindex0..7; actualsources use thatshard."""
    return deepcopy({0:_group(0),1:_group(1)} if group is None else _group(group))

def resident_pool(route,group):
    selected=set(map(int,route));pool=[]
    for source in _group(group)['resident_pages']:
        if source['expert']!=64 and source['expert'] not in selected:
            p=dict(source,page_id='drop.'+source['physical_page_id'],unselected=True,
                   dummy_unselected_expert=source['expert']);pool.append(p)
    return pool

def _plain(route,group):
    assignment=assign_phases(route);ids=assignment['phase0_experts'] if group==0 else assignment['phase2_experts']
    L=_group(group);stream=[]
    for e in ids:stream+=deepcopy(L['copies'][e])
    stream+=deepcopy(L['shared_pages']);assert len(stream)==48
    waves=[dict(pages=stream[i:i+4],read_pages=stream[i:i+4],source_order=[0,1,2,3]) for i in range(0,48,4)]
    assert all(len({p['wl'] for p in w['read_pages']})==1 for w in waves)
    return waves,assignment,ids

def _result(route,group,mode):
    assert group in (0,1) and mode in ('plain','split','whole_dummy')
    waves,assignment,ids=_plain(route,group);pool=resident_pool(route,group)
    if mode=='split':
        from phase_builder import _split
        waves=_split(waves,pool)
    elif mode=='whole_dummy':
        source=next(p for p in pool if p['page_index']==0);wl=source['wl']
        read=[deepcopy(next(p for p in pool if p['wl']==wl and p['plane']==plane)) for plane in range(4)]
        waves=waves[:10]+[dict(pages=[],read_pages=read,source_order=[0,1,2,3],all_dummy_wave=True)]+waves[10:]
    expected=[p for e in ids for p in _group(group)['copies'][e]]+_group(group)['shared_pages']
    actual=[p for w in waves for p in sorted(w['pages'],key=lambda p:p['plane'])]
    key=lambda p:(p['page_id'],p['physical_page_id'],p['plane'],p['wl'])
    assert list(map(key,actual))==list(map(key,expected))
    assert Counter(p['expert'] for p in actual)==Counter({**{e:12 for e in ids},64:12})
    physical=[(p['plane'],p['block'],p['wl']) for w in waves for p in w['read_pages']]
    assert len(physical)==len(set(physical))
    sameWL=all(len({(p['block'],p['wl']) for p in w['read_pages']})==1 for w in waves)
    assert sameWL==(mode!='split')
    return dict(waves=waves,assignment=assignment,group=group,group_expert_order=ids,
        owner_ranks={f'e{e}':i for i,e in enumerate(sorted(map(int,route)))}|{'s':6},pool=pool,
        metadata=dict(status='PASS',useful_pages=48,physical_pages=len(physical),dummy_pages=len(physical)-48,
            GU_row_bytes=1536,routed_rows_per_chip=128,shared_rows_per_chip=128,chip_span=8,
            fixed_source_0123=True,same_WL_every_physical_wave=sameWL,requires_IPR=(mode=='split'),
            all_dummy_addresses_real_unselected=True,copy_scope='58extrafullGUmatrices',
            source_shard='Expertlocalchip=chip%8;sharedlocalchip=chip. Bothgroupcopies refertothesamecompleteexpert tensor.',
            no_extra_PEselector=True))

def build_fixedgroup_plain(route,group=0):return _result(route,group,'plain')
def build_fixedgroup_split(route,group=0):return _result(route,group,'split')
def build_fixedgroup_whole_dummy(route,group=0):
    """Same-WL strongcontrol; emptyusefulwave needs explicitall-dummysimulator."""
    return _result(route,group,'whole_dummy')
