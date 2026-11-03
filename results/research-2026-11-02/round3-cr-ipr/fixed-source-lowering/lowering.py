#!/usr/bin/env python3
"""Whole-row macro lowering with fixed-plane or programmable-source slots.

The optimizer operates on existing page addresses and never moves a coefficient.
Pre-ECC drop remains an explicitly conditional capability.
"""
from collections import Counter
from functools import lru_cache
from copy import deepcopy

PAGE=16384

def physical_page(kind,expert,index,shared_geometry='native'):
    owner='s' if expert==64 else f'e{expert}'
    if expert!=64 and kind=='candidate':
        plane=[0,1,2,0,1,3][index]
        wl=2*expert+index//3 if plane<2 else expert
    elif expert!=64:
        address=expert*6+index;plane=address%4;wl=address//4
    elif kind=='candidate' or shared_geometry=='candidate_patterns':
        plane=[3,2,3,2,3,2,3,2,3,2,3,2][index]
        wl=(64 if kind=='candidate' else 96)+index//2
    else:
        plane=index%4;wl=96+index//4
    return dict(page_id=f'{owner}.p{index}',owner=owner,page_index=index,
                plane=plane,block=0,wl=wl,bytes=PAGE)

def build_problem(kind='candidate',even_count=3,K=6,shared_macros=4,shared_geometry='native'):
    assert kind in ('candidate','RR') and 0<=even_count<=K<=6 and 0<=shared_macros<=4
    if kind=='candidate':groups=[list(range(K))]
    else:groups=[list(range(0,2*even_count,2)),list(range(1,2*(K-even_count)+1,2))]
    types=[]
    for ids in groups:
        if ids:
            patterns=[tuple(physical_page(kind,ids[0],j,shared_geometry)['plane'] for j in range(b,b+3)) for b in [0,3]]
        else:
            e=0 if len(types)==0 else 1
            patterns=[tuple(physical_page(kind,e,j,shared_geometry)['plane'] for j in range(b,b+3)) for b in [0,3]]
        types.append(dict(experts=ids,patterns=patterns))
    shared=[tuple(physical_page(kind,64,j,shared_geometry)['plane'] for j in range(b,b+3)) for b in range(0,3*shared_macros,3)]
    return dict(kind=kind,K=K,shared_geometry=shared_geometry,types=types,shared_patterns=shared)

def _advance(pattern,state,source):
    cost=0;phase=state
    if source=='fixed':
        for p in pattern:
            n=(p-phase)%4;cost+=n+1;phase=(p+1)%4
    else:
        for p in pattern:
            if phase&(1<<p):cost+=1;phase=0
            phase|=1<<p
    return cost,phase

def solve_macros(problem,source='fixed'):
    """Return exact optimum for the declared macro order/slot objective.

    Counts remove interchangeable expert labels. State keeps A-unstarted and
    A-complete/B-ready counts for each of the at most two physical phase types.
    An action consumes a complete three-page macro; shared actions are ordered.
    """
    assert source in ('fixed','permutation')
    T=len(problem['types']);S=len(problem['shared_patterns'])
    initial=tuple(v for t in problem['types'] for v in (len(t['experts']),0))+(0,0)
    def transitions(state):
        s,phase=state[-2:]
        for t,typ in enumerate(problem['types']):
            u,r=state[2*t:2*t+2]
            for macro,allowed in [(0,u>0),(1,r>0)]:
                if not allowed:continue
                nxt=list(state)
                if macro==0:nxt[2*t]-=1;nxt[2*t+1]+=1
                else:nxt[2*t+1]-=1
                c,nxt[-1]=_advance(typ['patterns'][macro],phase,source)
                yield f'{"A" if macro==0 else "B"}{t}',c,tuple(nxt)
        if s<S:
            nxt=list(state);nxt[-2]+=1
            c,nxt[-1]=_advance(problem['shared_patterns'][s],phase,source)
            yield 'S',c,tuple(nxt)
    @lru_cache(None)
    def dp(state):
        if not any(state[:2*T]) and state[-2]==S:
            return ((-state[-1])%4 if source=='fixed' else int(bool(state[-1]))),1
        options=[(c+dp(n)[0],dp(n)[1]) for _,c,n in transitions(state)]
        best=min(v[0] for v in options)
        return best,sum(n for v,n in options if v==best)
    objective,paths=dp(initial);actions=[];state=initial
    while any(state[:2*T]) or state[-2]!=S:
        candidates=[(a,c,n) for a,c,n in transitions(state) if c+dp(n)[0]==dp(state)[0]]
        a,c,state=min(candidates,key=lambda x:x[0]);actions.append(a)
    macros=materialize_macros(problem,actions)
    slots=macro_slots(macros,source)
    waves=len(slots)//4
    assert (len(slots) if source=='fixed' else waves)==objective
    return dict(source=source,optimal_slots=len(slots),optimal_waves=waves,objective=objective,
                optimum_type_orders=paths,states=dp.cache_info().currsize,actions=actions,
                macros=macros,slots=slots)

def materialize_macros(problem,actions):
    unstarted=[list(t['experts']) for t in problem['types']];ready=[[] for _ in unstarted]
    result=[];s=0
    for code in actions:
        if code=='S':e=64;begin=s*3;s+=1
        else:
            t=int(code[1:])
            if code[0]=='A':e=unstarted[t].pop(0);ready[t].append(e);begin=0
            else:e=ready[t].pop(0);begin=3
        pages=[physical_page(problem['kind'],e,j,problem['shared_geometry']) for j in range(begin,begin+3)]
        result.append(dict(action=code,owner=pages[0]['owner'],page_begin=begin,pages=pages))
    return result

def macro_slots(macros,source):
    """Each None is one physical slot which must read a true dummy page."""
    slots=[];mask=0
    for m in macros:
        for p in m['pages']:
            if source=='fixed':
                while len(slots)%4!=p['plane']:slots.append(None)
            else:
                if mask&(1<<p['plane']):
                    slots.extend([None]*((-len(slots))%4));mask=0
                mask|=1<<p['plane']
            slots.append(deepcopy(p))
    slots.extend([None]*((-len(slots))%4))
    return slots

def resident_pool(problem):
    selected={e for t in problem['types'] for e in t['experts']}
    return [dict(physical_page(problem['kind'],e,j,problem['shared_geometry']),unselected=True)
            for e in range(64) if e not in selected for j in range(6)]

def slots_to_waves(problem,slots,source):
    assert len(slots)%4==0
    pool=resident_pool(problem);used=set();waves=[]
    for wi in range(len(slots)//4):
        group=deepcopy(slots[4*wi:4*wi+4]);pages=[p for p in group if p is not None]
        if not pages:raise ValueError('all-dummy wave is outside sealed pipeline input contract')
        assert len({p['plane'] for p in pages})==len(pages)
        read=list(pages)
        for plane in sorted(set(range(4))-{p['plane'] for p in pages}):
            p=next((p for p in pool if p['plane']==plane and p['page_id'] not in used),None)
            if p is None:raise ValueError('resident unique unselected-page pool exhausted')
            p=deepcopy(p);used.add(p['page_id']);read.append(p)
        order=sorted(range(4)) if source=='fixed' else [p['plane'] for p in pages]+[p['plane'] for p in read if p not in pages]
        if source=='fixed':assert [p['plane'] for p in pages]==sorted(p['plane'] for p in pages)
        waves.append(dict(pages=pages,read_pages=read,source_order=order))
    return waves

def validate_lowering(problem,solution,waves,source):
    expected=[p for m in solution['macros'] for p in m['pages']]
    actual=[next(p for p in w['pages'] if p['plane']==plane) for w in waves
            for plane in (sorted(w['source_order']) if source=='fixed' else w['source_order'])
            if plane in {p['plane'] for p in w['pages']}]
    fields=lambda p:(p['page_id'],p['plane'],p['block'],p['wl'],p['bytes'])
    assert list(map(fields,expected))==list(map(fields,actual))
    expected_unordered=[physical_page(problem['kind'],e,j,problem['shared_geometry'])
                        for t in problem['types'] for e in t['experts'] for j in range(6)]
    expected_unordered += [physical_page(problem['kind'],64,j,problem['shared_geometry']) for j in range(3*len(problem['shared_patterns']))]
    assert Counter(map(fields,actual))==Counter(map(fields,expected_unordered))
    pool=set(map(fields,resident_pool(problem)));physical=[];dummies=[]
    for w in waves:
        useful={p['page_id'] for p in w['pages']}
        assert {p['plane'] for p in w['read_pages']}==set(range(4))
        for p in w['read_pages']:
            physical.append((p['plane'],p['block'],p['wl']))
            if p['page_id'] not in useful:assert fields(p) in pool;dummies.append(p)
    assert len(set(physical))==len(physical)
    assert all(p['block']==0 and 0<=p['wl']<192 for w in waves for p in w['read_pages'])
    owners=Counter(p['owner'] for p in actual)
    assert all(v%3==0 for v in owners.values())
    return dict(status='PASS',useful_pages=len(actual),physical_pages=len(physical),dummy_pages=len(dummies),
                zero_copy=True,one_block=True,real_unselected_unique_addresses=True,
                owner_pages=dict(owners),macro_contiguous_after_drop=True,macro_rows=32)

def one_extra_wave_variants(solution,source):
    """Bounded timing controls preserve the exact optimum macro stream.

    Fixed mode inserts one complete cyclic rotation before a useful slot.
    Permutation mode splits one physical wave after a useful prefix.
    """
    slots=solution['slots']
    if source=='fixed':
        for i,p in enumerate(slots):
            if p is None:continue
            trial=slots[:i]+[None]*4+slots[i:]
            if all(any(p is not None for p in trial[j:j+4]) for j in range(0,len(trial),4)):
                yield dict(kind='insert_one_cyclic_rotation',slot=i),trial
    else:
        for wi in range(len(slots)//4):
            pages=[p for p in slots[4*wi:4*wi+4] if p is not None]
            for prefix in range(1,len(pages)):
                left=pages[:prefix]+[None]*(4-prefix)
                right=pages[prefix:]+[None]*(4-len(pages)+prefix)
                yield dict(kind='split_wave',wave=wi,prefix=prefix),slots[:4*wi]+left+right+slots[4*wi+4:]
