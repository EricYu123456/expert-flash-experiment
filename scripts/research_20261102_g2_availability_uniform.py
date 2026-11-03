#!/usr/bin/env python3
"""Exact weighted cap1 availability over all top6 for the frozen standard CBC.

Separate addendum: does not modify the selected maps or the TRAIN protocol.
"""
from collections import Counter
import ctypes
import datetime
import hashlib
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/research-2026-11-02/round3-cr-ipr/g2-availability-headroom'
OUT=BASE/'uniform-addendum'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def need(ok,msg):
    if not ok:raise RuntimeError(msg)
def write(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')


def main():
    OUT.mkdir(parents=True,exist_ok=True);began=time.monotonic()
    paths=[Path(__file__).resolve(),BASE/'availability.frozen.npz',BASE/'protocol.frozen.json',BASE/'selection.frozen.json',
           BASE/'matching.so',ROOT/'scripts/research_20261102_g2_availability_headroom.cpp']
    hashes={str(p):sha(p) for p in paths}
    write(OUT/'protocol.frozen.json',dict(status='frozen_before_exact_weighted_class_enumeration',
          utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),inputs_sha256=hashes,
          operation='Enumerate bounded occupancy counts of all19support types summing6. Weight by product binomial(multiplicity,count). Full cap1 matching and independent capacitated Hall classification must agree.',
          claim='Uniform distribution over all C(64,6) unordered top6, invariant under expert label permutations. Not a TRAIN/heldout quality test or an observed workload probability estimate.'))
    with np.load(BASE/'availability.frozen.npz') as z:
        all_maps=z['masks'].copy();names=z['names'].copy()
    masks=np.ascontiguousarray(all_maps[0,0],dtype=np.uint8)
    histogram=Counter(map(int,masks))
    need(sorted(histogram.values())==[1]*4+[4]*15,'nineteen expected support types')
    for mm in all_maps.reshape(-1,64):need(Counter(map(int,mm))==histogram,'label permutations preserve uniform incidence')
    types=sorted(histogram);members=[[e for e in range(64) if masks[e]==m] for m in types]
    mult=[len(x) for x in members]
    lib=ctypes.CDLL(str(BASE/'matching.so'))
    u8=np.ctypeslib.ndpointer(dtype=np.uint8,flags='C_CONTIGUOUS');i32=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')
    lib.match_route.argtypes=[u8,i32,i32,i32];lib.match_route.restype=ctypes.c_int
    caps=np.ones(6,dtype=np.int32);answer=np.empty(6,dtype=np.int32)
    group_subsets=[s for s in range(1,64) if s.bit_count() in [4,5]]
    # Degree>=3 rules out Hall failures below3neighbors; each3support has
    # multiplicity1, so3neighbor subsets also cannot contain4experts.
    classes=0;total=0;fail=0;groupmiss=0;fourhall=0;both=0;maxdef=Counter();examples=[]
    counts=[0]*19
    def visit(index,left,weight):
        nonlocal classes,total,fail,groupmiss,fourhall,both
        if index==19:
            if left:return
            route=np.asarray([e for i,n in enumerate(counts) for e in members[i][:n]],dtype=np.int32)
            selected=[int(masks[e]) for e in route]
            matched=bool(lib.match_route(masks,route,caps,answer))
            deficits=[sum((m&~s)==0 for m in selected)-s.bit_count() for s in group_subsets]
            hall_failure=max(deficits)>0
            need(matched!=hall_failure,'matching/Hall classification disagreement')
            union=0
            for m in selected:union|=m
            missing=union!=63
            four=any(d>0 for d,s in zip(deficits,group_subsets) if s.bit_count()==4)
            classes+=1;total+=weight
            if hall_failure:
                fail+=weight;groupmiss+=weight*missing;fourhall+=weight*four;both+=weight*(missing and four)
                maxdef[max(deficits)]+=weight
                if len(examples)<6:examples.append(dict(experts=route.tolist(),support_masks=selected,weight=weight,missing_groups=[g for g in range(6) if not union&(1<<g)],max_Hall_deficit=max(deficits)))
            return
        for n in range(min(mult[index],left)+1):
            counts[index]=n;visit(index+1,left-n,weight*math.comb(mult[index],n))
        counts[index]=0
    visit(0,6,1)
    need(total==math.comb(64,6)==74974368,'weighted enumeration must cover every top6 once')
    per_group_miss=[math.comb(sum(not(int(m)&(1<<g)) for m in masks),6) for g in range(6)]
    need(per_group_miss==[math.comb(22,6)]*6,'each group unavailable to22experts')
    need(hashes=={p:sha(p) for p in hashes},'frozen source/maps changed')
    report=dict(status='complete_exact_uniform_availability',inputs_sha256=hashes,
        support_types=[dict(mask=m,groups=[g for g in range(6) if m&(1<<g)],multiplicity=histogram[m]) for m in types],
        bounded_composition_classes=classes,weighted_top6_count=total,perfect_cap1_count=total-fail,failed_cap1_count=fail,
        uniform_cap1_failure_probability=fail/total,uniform_cap1_perfect_probability=1-fail/total,
        failed_due_missing_group=groupmiss,failed_due_4group_Hall_subset=fourhall,intersection_of_failure_types=both,
        weighted_max_Hall_deficit={str(k):v for k,v in maxdef.items()},per_group_missing_all6_count=per_group_miss,
        all_frozen_maps_same_uniform_probability=True,map_labels=names.tolist(),examples=examples,
        interpretation='TRAIN1536/1536 on a selected label permutation is not universal availability. Uniform failure probability is a combinatorial property of the frozen19type incidence and is unchanged by label permutations; it is not a heldout workload forecast.',
        limitations=['Standard CBC/capacitated Hall counting, no new theorem or solver claim.',
                     'Cap1 exact availability only; cap2 remains guaranteed for alltop6 by degree>=3 with six groups of capacity2.',
                     'No new routes or activations are captured; no heldout, taskquality, hardware or fullFFN experiment.',
                     'This addendum leaves frozen maps, selectedTRAINpolicy and original source untouched.'],
        elapsed_seconds=time.monotonic()-began,actual_command=['python',str(Path(__file__).resolve())])
    write(OUT/'report.json',report)
    (OUT/'report.md').write_text(
        f"Exact uniform availability: {classes:,} bounded support-count classes weight to all{total:,} unordered top6. {fail:,} fail cap1 ({100*fail/total:.8f}%); {total-fail:,} admit sixdistinct groups. Augmenting-path matching agrees with independent Hall subset counts for everyclass.\n\n"
        f"{groupmiss:,} failures miss at leastone group; {fourhall:,} contain a Hall-overfull fourgroup subset, with overlap{both:,}. Eachgroup is unavailable to22experts, giving C(22,6)={per_group_miss[0]:,} top6missing thatgroup before union deduplication.\n\n"
        'All22frozen label maps/layer have the same19support multiplicities, hence the same uniform probability. SelectedTRAIN1536/1536 is not a universal cap1guarantee; this uniform distribution is not a heldout workload forecast. Cap2 guarantees everytop6. StandardCBC/Hall counting is prior, not a noveltyclaim. Original maps/source/protocol are unchanged.\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['inputs_sha256','support_types','examples','map_labels','limitations']},indent=2))


if __name__=='__main__':main()
