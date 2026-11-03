#!/usr/bin/env python3
"""TRAIN-only standard six-group availability/cap2 headroom screen.

New native job geometry is retained; this is a conditional chip-calendar
oracle, not full-FFN execution, heldout validation or a new CBC result.
"""
from collections import Counter
import ctypes
import datetime
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/research-2026-11-02/round3-cr-ipr'
OUT=BASE/'g2-availability-headroom'
TRAIN=ROOT/'results/research-2026-11-02/round2/w8-format-training/granite4tiny/decode.routes.npz'
OLD=BASE/'down-replica-full-ffn/TRAIN512'
PIPE=ROOT/'scripts/research_20261102_cr_ipr_pipeline.py'
SEEDS=list(range(2026110200,2026110220))
SWAP_SEED=2026110264
SWAP_BUDGET=2048


def need(ok,msg):
    if not ok:raise RuntimeError(msg)


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')


def id_slots():
    rows=[]
    for subset in itertools.combinations(range(6),4):rows += [sum(1<<g for g in subset)]*4
    rows += [sum(1<<g for g in subset) for subset in [(0,1,2),(3,4,5),(0,3,4),(1,2,5)]]
    masks=np.asarray(rows,dtype=np.uint8)
    check_map(masks)
    return masks


def adjacency(masks):return ((masks[...,None]>>np.arange(6))&1).astype(np.uint8)


def check_map(masks):
    a=adjacency(masks)
    need(np.array_equal(a.sum(axis=0),np.full(6,42)),'group capacity42 invariant')
    need(np.array_equal(a.sum(axis=1),np.r_[np.full(60,4),np.full(4,3)]),'per-expert rowdegree invariant')


def matching_assignment(adj,expert_ids,caps=(1,1,1,1,1,1)):
    """Public deterministic online augmenting-path API; original input slots."""
    expert_ids=list(map(int,expert_ids));owners=[-1]*12
    def place(rank,seen):
        for g in range(6):
            if not adj[expert_ids[rank],g]:continue
            for j in range(int(caps[g])):
                slot=2*g+j
                if slot in seen:continue
                seen.add(slot)
                if owners[slot]<0 or place(owners[slot],seen):owners[slot]=rank;return True
        return False
    for rank in range(6):
        if not place(rank,set()):return None
    result=[-1]*6
    for slot,rank in enumerate(owners):
        if rank>=0:result[rank]=slot//2
    return result


def plan_route(adj,expert_ids):
    """Canonical online policy: sorted expert IDs, cap1 then earliest-group cap2.

    Return group IDs aligned with expert_ids as provided. This policy uses no
    query index or saved TRAIN assignment lookup and is shared by input1/4.
    """
    order=np.argsort(expert_ids);sorted_ids=np.asarray(expert_ids)[order]
    caps=[1]*6
    for prefix in range(7):
        answer=matching_assignment(adj,sorted_ids,caps)
        if answer is not None:
            result=np.empty(6,dtype=np.int32);result[order]=answer
            return result.tolist()
        if prefix<6:caps[prefix]=2
    raise RuntimeError('cap2 Hall guarantee violated')


def hall_perfect(masks,route):
    """Independent finite subset Hall oracle, no augmenting-path state."""
    selected=[int(masks[e]) for e in route]
    for subset in range(1,64):
        if sum((m&~subset)==0 for m in selected)>subset.bit_count():return False
    return True


def native_kernel():
    spec=importlib.util.spec_from_file_location('g2_native_cost_only',PIPE)
    p=importlib.util.module_from_spec(spec);sys.modules[spec.name]=p;spec.loader.exec_module(p)
    waves=[]
    for w in range(6):
        pages=[dict(page_id=f'e.p{k}',owner='e',page_index=k,plane=k%4,block=0,wl=k//4,bytes=16384) for k in range(4*w,4*w+4)]
        waves.append(dict(pages=pages,source_order=[0,1,2,3]))
    trace=p.simulate(waves,cache_copy_ns=500,source_policy='fixed',issue_policy='strict_no_interval',row_bytes=512)
    need(trace['feasible'] and trace['makespan_ns']==89940,'new native768row C512 fixed-source job')
    write(OUT/'native_g2_kernel.trace.json',dict(waves=waves,trace=trace))
    return trace['makespan_ns']


def capacity(masks):
    own=set();jobs=[]
    for chip in range(16):
        for e in range(64):
            for lp,plane in enumerate([0,1,2,0,1,3]):
                wl=2*e+lp//3 if plane<2 else e
                own.add((chip,0,plane,wl))
        for lp in range(12):own.add((chip,0,[3,2][lp%2],64+lp//2))
    need(len(own)==6336,'existing candidateGU exact addresses')
    for g in range(6):
        experts=[e for e in range(64) if masks[e]&(1<<g)];need(len(experts)==42,'group42resident experts')
        for slot,e in enumerate(experts):
            block=0 if slot<10 else 1;wl=128+6*slot if slot<10 else 6*(slot-10)
            for chip in [2*g,2*g+1]:
                jobs.append(dict(expert=e,copy_group=g,chip=chip,rows=768,columns=512,row_start=(chip%2)*768,
                                 block=block,wl_start=wl,pages=24,reserved_WL=6))
                for page in range(24):
                    key=(chip,block,page%4,wl+page//4)
                    need(key not in own and key[3]<192,'physical band collision or crossblock job')
                    own.add(key)
    for chip in range(12,16):
        jobs.append(dict(expert=64,copy_group=6,chip=chip,rows=384,columns=1024,row_start=(chip-12)*384,
                         block=0,wl_start=128,pages=24,reserved_WL=6))
        for page in range(24):
            key=(chip,0,page%4,128+page//4);need(key not in own,'sharedDown collision');own.add(key)
    need(len(own)*16384==int(289.5*(1<<20)),'total useful coefficient bytes')
    need(len(jobs)==508,'252 full routedDown copies + shared')
    blocks=len({(c,b) for c,b,p,w in own})
    return dict(unique_pages=len(own),programmed_bytes=len(own)*16384,programmed_MiB=289.5,
                full_routedDown_copy_count=252,extra_full_routedDown_copies=188,extra_MiB_vs_onecopy=141,
                extra_MiB_vs_g4_triples=45,actual_nonempty_chipblocks=blocks,common_budget_chipblocks=32,
                routed_chip_max_pages_per_plane=[380,380,322,322],native_jobs_stay_in_one_block=True,
                address_sha256=hashlib.sha256(np.asarray(sorted(own),dtype='<i4').tobytes()).hexdigest(),jobs=jobs)


def main():
    OUT.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    traces={cp:OLD/f'trace.L0.candidate.triple_private_shared4.input{cp}.GF.json' for cp in [1,4]}
    paths=[Path(__file__).resolve(),Path(__file__).with_suffix('.cpp'),TRAIN,PIPE,OLD/'report.json',OLD/'service_arrays.npz',*traces.values()]
    hashes={str(p):sha(p) for p in paths}
    protocol=dict(status='frozen_before_TRAIN_matching',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),inputs_sha256=hashes,
        seeds=SEEDS,seed_generator='NumPy PCG64; layer seed=seed+10000*layer',swap_seed=SWAP_SEED,
        swap_budget_per_layer=SWAP_BUDGET,swap_rule='Uniform distinct same-degree expert pairs; accept only strictly more perfect cap1 routes. Stop useful work once all512 match; no cost fitting.',
        fixed_degrees='E0..59degree4,E60..63degree3 for all controls/seeds/swaps; each column42. Degree3 identities are not fitted.',
        selection='Per layer choose greatest perfect count among IDslot+20seeds, ties by listed order; bounded swaps start there.',
        groups=[list(range(2*g,2*g+2)) for g in range(6)],shared_chips=list(range(12,16)),
        scope='Only first3layers of previously observed TRAIN512; no heldout, new arithmetic/quantization or fullFFN execution. Standard CBC/replicateddeclustering baseline, no theorem/solver novelty claim.')
    write(OUT/'protocol.frozen.json',protocol)
    kernel=native_kernel()
    with np.load(TRAIN) as z:routes=np.ascontiguousarray(z['routes'].transpose(2,0,1,3).reshape(40,512,6)[:3],dtype=np.int32)
    command=['g++','-O3','-std=c++17','-shared','-fPIC',str(Path(__file__).with_suffix('.cpp')),'-o',str(OUT/'matching.so')]
    build=subprocess.run(command,text=True,capture_output=True);write(OUT/'build_command.json',dict(command=command,returncode=build.returncode,stdout=build.stdout,stderr=build.stderr));need(build.returncode==0,'compile matcher')
    lib=ctypes.CDLL(str(OUT/'matching.so'))
    u8=np.ctypeslib.ndpointer(dtype=np.uint8,flags='C_CONTIGUOUS');i32=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS');i8=np.ctypeslib.ndpointer(dtype=np.int8,flags='C_CONTIGUOUS')
    lib.score_map.argtypes=[u8,i32,ctypes.c_int,i8];lib.score_map.restype=ctypes.c_int
    lib.train_swaps.argtypes=[u8,i32,ctypes.c_int,i32,ctypes.c_int,i32,i8,i32];lib.train_swaps.restype=ctypes.c_int
    lib.match_route.argtypes=[u8,i32,i32,i32];lib.match_route.restype=ctypes.c_int
    initial=id_slots();maps=[np.tile(initial,(3,1))];names=['IDslot']
    for seed in SEEDS:
        mm=[]
        for layer in range(3):
            rng=np.random.Generator(np.random.PCG64(seed+10000*layer))
            mm.append(initial[np.r_[rng.permutation(60),60+rng.permutation(4)]])
        maps.append(np.asarray(mm,dtype=np.uint8));names.append(f'seed{seed}')
    scores=np.zeros((21,3),dtype=np.int32)
    for i,mm in enumerate(maps):
        for layer in range(3):
            check_map(mm[layer]);flags=np.empty(512,dtype=np.int8);scores[i,layer]=lib.score_map(mm[layer],routes[layer],512,flags)
    fitted=[];fit_records=[];pair_records=[]
    for layer in range(3):
        winner=int(np.argmax(scores[:,layer]));m=maps[winner][layer].copy()
        rng=np.random.Generator(np.random.PCG64(SWAP_SEED+10000*layer));pairs=[]
        for k in range(SWAP_BUDGET):
            group=np.arange(60) if rng.integers(64)<60 else np.arange(60,64)
            pairs.append(rng.choice(group,2,replace=False))
        pairs=np.ascontiguousarray(pairs,dtype=np.int32);pair_records.append(pairs)
        hist=np.zeros(SWAP_BUDGET,dtype=np.int32);accepted=np.zeros(SWAP_BUDGET,dtype=np.int8);attempted=np.zeros(1,dtype=np.int32)
        score=lib.train_swaps(m,routes[layer],512,pairs,SWAP_BUDGET,hist,accepted,attempted)
        check_map(m);need(score==sum(hall_perfect(m,r) for r in routes[layer]),'fitted score independent Hall crosscheck')
        fitted.append(m);fit_records.append(dict(layer=layer,start_map=names[winner],start_score=int(scores[winner,layer]),final_score=score,
                     attempted=int(attempted[0]),accepted_swaps=[dict(proposal=int(k),experts=pairs[k].tolist(),score=int(hist[k])) for k in np.flatnonzero(accepted)]))
    maps.append(np.asarray(fitted,dtype=np.uint8));names.append('TRAIN_fitted');maps=np.asarray(maps,dtype=np.uint8)
    frozen_adj=adjacency(maps)
    np.savez_compressed(OUT/'availability.frozen.npz',names=np.asarray(names),masks=maps,adjacency=frozen_adj,
                        row_degrees=np.r_[np.full(60,4),np.full(4,3)],swap_pairs=np.asarray(pair_records))
    write(OUT/'selection.frozen.json',dict(winner='TRAIN_fitted',fit_records=fit_records,initial_scores=scores.tolist(),
          availability_sha256=sha(OUT/'availability.frozen.npz'),protocol_sha256=sha(OUT/'protocol.frozen.json')))
    # Deterministic matching API uses sorted IDs internally, then scatters to
    # the original router slots. Prefix cap2 enables the earliest-free group.
    assignments=np.empty((22,3,512,6),dtype=np.int32);perfect=np.zeros((22,3,512),dtype=np.uint8);records=[]
    for i,m in enumerate(maps):
        for layer in range(3):
            for q,route in enumerate(routes[layer]):
                answer=plan_route(frozen_adj[i,layer],route);assignments[i,layer,q]=answer
                loads=np.bincount(answer,minlength=6)
                need(loads.max()<=2 and all(m[layer,e]&(1<<g) for e,g in zip(route,answer)),'complete legal cap2 assignment')
                perfect[i,layer,q]=int(loads.max()==1)
                need(bool(perfect[i,layer,q])==hall_perfect(m[layer],route),'all-control Hall crosscheck')
        records.append(dict(name=names[i],perfect=int(perfect[i].sum()),events=1536,perfect_by_layer=perfect[i].sum(axis=1).astype(int).tolist(),
                            fallback_cap2=int(1536-perfect[i].sum())))
        if i in [0,20,21]:print('MATCHED',names[i],records[-1]['perfect'],'/1536',flush=True)
    caps=capacity(maps[-1,0]);write(OUT/'capacity.json',caps)
    np.savez_compressed(OUT/'assignments.TRAIN.frozen.npz',names=np.asarray(names),routes=routes,group_assignments=assignments,
                        perfect_cap1=perfect,selected_adjacency=frozen_adj[-1],selected_assignments=assignments[-1],
                        selected_group_loads=np.eye(6,dtype=np.int32)[assignments[-1]].sum(axis=2))
    calendar_cases=[];cost_arrays={}
    with np.load(OLD/'service_arrays.npz') as z:
        for copies,path in traces.items():
            trace=json.loads(path.read_text());ns=trace['nodes'];A=[0]*16
            for job in trace['jobs']:
                if job['phase']==0:A[job['chip']]=ns[job['release']]['end_ns']
            ds=[next(n['end_ns'] for n in ns if n['kind']=='Down_submit' and n['expert_rank']==rank) for rank in range(7)]
            up=[next(n['end_ns'] for n in ns if n['kind']=='Down_input_PCIe' and n['expert_rank']==rank) for rank in range(7)]
            groupA=[max(A[2*g:2*g+2]) for g in range(6)]
            B=250+256*copies+kernel+64;S=250+512*copies+89940+32;shared=max(A[12:])+S
            need(max(ds[:5])<=min(A[:12]) and ds[5]<min(A[:12])+B and ds[6]<min(A[12:]),'release-calendar ordering for one/two-job formula')
            need(all(u<=r+250 for r,u in zip(ds,up)),'full private PCIe input precedes local input after command')
            costs=np.empty((22,3,512),dtype=np.int64);loads=np.eye(6,dtype=np.int32)[assignments].sum(axis=3)
            for i in range(22):
                for layer in range(3):
                    for q,route in enumerate(routes[layer]):
                        final=shared;late_slot=int(np.argmax(route))
                        for g in range(6):
                            ranks=np.flatnonzero(assignments[i,layer,q]==g)
                            if len(ranks)==1:finish=max(groupA[g],ds[5] if ranks[0]==late_slot else 0)+B
                            elif len(ranks)==2:finish=groupA[g]+2*B
                            else:finish=groupA[g]
                            final=max(final,finish)
                        costs[i,layer,q]=final
            base=z[f'candidate.triple_private_shared4.input{copies}.GF.cost']
            cost_arrays[f'input{copies}.fixed_calendar_LB']=costs
            means=costs.mean(axis=(1,2));ideal=max(shared,ds[5]+B,max(groupA)+B)
            cases=[dict(name=names[i],mean_ns=float(means[i]),min_ns=int(costs[i].min()),max_ns=int(costs[i].max()),
                        headroom_upper_pct=float(100*(1-means[i]/base.mean()))) for i in range(22)]
            calendar_cases.append(dict(input_copies=copies,GU_release_ns=A,private_Down_submit_by_sorted_rank_ns=ds,
                private_upload_complete_by_sorted_rank_ns=up,g2_job_owner_ns=B,shared_g4_job_owner_ns=S,
                all_cap1_ideal_calendar_ns=ideal,existing_g4_triples_actual_mean_ns=float(base.mean()),maps=cases))
    np.savez_compressed(OUT/'oracle_costs.npz',**cost_arrays)
    # Manual capacity witnesses distinguish cap1 success from guaranteed cap2.
    bad=[8,9,10,11,60,63]
    need(not hall_perfect(initial,bad),'manual Hall counterexample')
    fallback=plan_route(adjacency(initial),bad);need(max(np.bincount(fallback,minlength=6))<=2,'cap2 manual witness')
    need(hashes=={p:sha(p) for p in hashes},'frozen inputs changed')
    chosen=calendar_cases[0]['maps'][-1];worth=chosen['headroom_upper_pct']>5
    report=dict(status='potential_above5pct_requires_actual' if worth else 'negative_below5pct_stop',inputs_sha256=hashes,
        matching_records=records,fit_records=fit_records,calendar_cases=calendar_cases,capacity={k:v for k,v in caps.items() if k!='jobs'},
        manual_nonperfect_top6=bad,manual_cap2_assignment=fallback,
        cap2_all_top6_proof='Every expert has at least3neighbors. Every nonempty expert subset of a top6 has at least3neighbor groups, whose capacity2 supports at least6experts >= subset size. Capacitated Hall therefore guarantees a complete assignment for everytop6. Known elementary CBC fact, no new theorem.',
        API=dict(source=str(Path(__file__).resolve()),function='plan_route(adjacency64x6,expert_ids6)',
                 availability=str(OUT/'availability.frozen.npz'),adjacency_shape=[22,3,64,6],selected='adjacency[-1]',
                 assignments=str(OUT/'assignments.TRAIN.frozen.npz'),assignment_shape=[22,3,512,6],
                 slot_order='Assignments align with original router slots; the online API sorts internally then scatters back.',
                 warning='TRAIN arrays are audit artifacts, not a query-index-dependent runtime policy. Actual adapter should call the deterministic online matcher with frozen adjacency.'),
        elapsed_seconds=time.monotonic()-started,actual_commands=[command,['python',str(Path(__file__).resolve())]],
        source_sha256=sha(__file__),matcher_binary_sha256=sha(OUT/'matching.so'),
        limits=['Standard CBC/replicateddeclustering control; neither new replication theorem nor new solver is claimed.',
                'TRAIN-only labels/swap fit; no heldout evidence, new numeric/quality test or fullFFN execution.',
                'The restricted oracle freezes current candidateGU release and full-private-submit/upload calendars and native768row g2 jobs. It is not global over GU/Down geometry.',
                'FullC512 routed/privateC1024 shared vectors and command250ns are charged; latest completeGU→A8→privateupload readiness is retained. Globalcommand/channel/PCIe contention, FIFO and runtime matcher cost are relaxed.',
                'The primary ready-aware bound includes late sorted-rank5private readiness; a bare GUrelease+90us estimate would be more optimistic.',
                'Per-expert degrees stay4for IDs0..59 and3for60..63; swaps never change these identities. Stronger fits moving low-degree identities are outside this frozen screen.',
                'Logical fullrow arithmetic uses existingI32 bounds:512*128*128=8388608. No partial input, host partial sums, extra accumulator context or new native feature is used.',
                'Payload289.5MiB is45MiB more than g4triples; both fit the same maximum2blocks/chip budget. More programmed coefficient bytes are not free.',
                'Physical certificate checks unique whole-job page coordinates and complete row bands, not coefficient programming. PB/source timing remains the previously conditional strict500ns model; no measured hardware timing claim.'])
    write(OUT/'report.json',report)
    (OUT/'report.md').write_text(
        f"Standard g2 availability oracle: {report['status']}. IDslot perfect matching {records[0]['perfect']}/1536; frozen TRAIN_fitted {records[-1]['perfect']}/1536; all remaining events have a legal cap2 assignment. Twenty fixed PCG64 seed controls and at most2048same-degree swap proposals/layer are retained. Per-expert rowdegrees and every column42stay unchanged.\n\n"
        f"Native routedDown is768rows/chip,C512,24pages,6waves,89.940us. Ready-aware fixed-calendar means for fitted input1/4 are {calendar_cases[0]['maps'][-1]['mean_ns']/1000:.6f}/{calendar_cases[1]['maps'][-1]['mean_ns']/1000:.6f}us, against existing g4triple actual means280.526/285.438us: optimistic headroom{calendar_cases[0]['maps'][-1]['headroom_upper_pct']:.4f}%/{calendar_cases[1]['maps'][-1]['headroom_upper_pct']:.4f}%. These retain fullprivate submit/upload readiness and minimum command/input/lastONFI ownership, while relaxing other shared resources/FIFO and matcher cost. They are not achieved fullFFN costs.\n\n"
        'Availability is60degree4experts, four per4-of-6subset, plus fourdegree3experts on012/345/034/125. Eachgroup has42completeDown experts. Everytop6 has a cap2matching by elementary capacitatedHall; cap1 is not guaranteed, as IDslot route8,9,10,11,60,63 shows. This is a standard CBC/replicateddeclustering baseline, not a new theorem.\n\n'
        'Coefficient payload289.5MiB/layer: original148.5+188extraDowncopies×0.75=141MiB. This is45MiB more than g4triples. Each routedchip stores10intact6WLbands in block0WL128..187 and32in block1WL0..191; maxplane count380<384. Sharedg4 stays onchips12..15. All addresses are unique and every nativejob stays within oneblock;28nonempty chipblocks fit the common32block budget.\n\n'
        'API: import plan_route from this source and pass selected_adjacency[layer] with sixexpertIDs. Returned groupindices align with the supplied router order. The frozen TRAIN assignment arrays document the screen; the runtime policy never looks up queryindices.\n\n'+ '\n'.join(report['limits'])+'\n')
    print(json.dumps(dict(status=report['status'],matching_records=records,fit_records=fit_records,
                         selected_costs=[x['maps'][-1] for x in calendar_cases],elapsed_seconds=report['elapsed_seconds']),indent=2))


if __name__=='__main__':main()
