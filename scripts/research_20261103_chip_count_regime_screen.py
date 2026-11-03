#!/usr/bin/env python3
"""Geometry/resource necessary-headroom stage for existing ordinary-read scaling.

No native replay, label fitting, replica selection, inference, or external data.
"""
import argparse
import hashlib
import json
import math
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from research_20261102_whole_row_compiler import Capability, minimum_work, materialize

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'results/research-2026-11-02/round2'
OUT=BASE/'chip-count-regime-screen/stage-a'
MODELS={'granite4tiny':dict(H=1536,I=512,SI=1024,E=64,K=6),
        'deepseek':dict(H=2048,I=1408,SI=2816,E=64,K=6)}
CHIPS=[8,16,32,64]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def stats(x):return dict(mean_ns=float(np.mean(x)),p99_ns=float(np.percentile(x,99)))


def freeze():
    OUT.mkdir(parents=True,exist_ok=False);(OUT/'source-snapshot').mkdir()
    paths=[Path(__file__),ROOT/'scripts/research_20261102_whole_row_compiler.py',
           BASE/'w8-format-training/protocol.frozen.json',BASE/'w8-format-training/selection_protocol.frozen.json',
           BASE/'full-ffn-integer-reference/frozen_protocol.json',BASE/'deepseek-full-ffn-integer-reference/protocol.frozen.json']
    paths += [BASE/'compact-coupled-solver/continuation/native/0107.npz',
              BASE/'native-joint-refinement/pilot-v1/deepseek.L0.input4.independent_paired.seed20261105/arrays.npz']
    paths += [BASE/f'w8-format-training/{m}/decode.routes.npz' for m in MODELS]
    inputs=[]
    for path in paths:
        item=dict(path=str(path),sha256=sha(path))
        if path.suffix=='.py':
            snap=OUT/'source-snapshot'/(item['sha256'][:16]+'-'+path.name);shutil.copy2(path,snap);item['snapshot']=str(snap)
        inputs.append(item)
    p=dict(status='frozen_before_regime_geometry_and_TRAIN_ledger',utc=datetime.now(timezone.utc).isoformat(),
           source_sha256=sha(__file__),inputs=inputs,models=MODELS,chips=CHIPS,
           hardware=dict(channels='min(8,chips)',planes=4,page_bytes=16384,block_wordlines=192,read_wave_ns=28000,
                         command_ns=250,host_submit_ns=2000,input_bytes=1,output_bytes=4,input_copies=[1,4],
                         output_fringe='allcomplete rows in programmed pages; floor(pages*16KiB/C)',
                         contexts='single perchip through channel drain; fullGU/A8 before privateDown'),
           interpretation='8/32/64chip scalevariants, not measuredAiF16chip implementation or evidence for64independentPEs. Never compareperformanceacrossdifferenthardware asplacementgain.',
           stage='A:exactwhole-row work-depth frontiers plus bounded deterministic profile resource ledgers. No nativeGF/DFcosts, label fits, replicas or inference.',
           menu='For eachrouted/sharedprojection:allwidth1..M equalrows plus everyminimum-work-at-depth witness fromminimumdepth throughexact-workdepth. Canonical rowmultisets; allsingleblock.',
           mappings='Cyclic shifts withfixedsteps1/3/5/7 andM/2,M/4,M/8; disjointwidths dividingM withwithin-groupceilrowrotation. Deduplicate full64expert rowarrays. No learnedexpertlabels.',
           shared='Samefair2geometries (minimumwork/tieslowestdepth, minimumdepth/tieslowestwork) andshifts0/Mhalf for allroutedcontrols.',
           finite_selection='Rankphasecontrols bymeanownedchipwork onTRAIN512; keep4each. Combine4x4 withallglobalDownrotations,rankmeanjointownedwork. This finiteproxysearch isnotnativeruntimeoptimization orallprofile/globaloptimality.',
           bounds='Universallowermax(initialhost+ceil(totalminimumwaves/M)*readwave,minimumwholeexpertchain,minimumcontroller/channel/PCIevolume). Exactfullrow aggregatefloors,notattainableschedule. Allwidth/fullinput/commands/fringecharged incontrolledgers.',
           budget=dict(max_DP_calls=120,target_wall_seconds=90,native_queries=0,label_fit_proposals=0,
                       possible_next_stage='Onlyafterreport:max4(profile,inputcopy) pairs/physicalM-modelcase,includinginput1/4and16chiplegacyanchors x2prioritiesGF/DF x512 x8hardwarecases=32768nativequeries; notrunbythisscript.'),
           exclusions='NoQ38yet,noheldout/newroutes/models/quantization/PE/FIFO/context/MSB/partials orcapacityclaim. L0coefficientinventory canfitmoreimagesinthesameblock;replicaclaimsneedexplicitstrongcapacitycontrol.')
    dump(OUT/'protocol.frozen.json',p);print(json.dumps(dict(protocol_sha256=sha(OUT/'protocol.frozen.json'),source_sha256=sha(__file__))))


def geometry(R,C,M):
    cap=Capability(chips=M);B=65536
    dmin=((R+M-1)//M*C+B-1)//B
    wfloor=(R*C+B-1)//B;period=C//math.gcd(C,B)
    assert R*C%B==0
    exactdepth=period*((wfloor//period+M-1)//M)
    end=max(dmin,exactdepth)
    frontier=[];menu={}
    def add(rr,kind,meta):
        rr=np.asarray(sorted(map(int,rr),reverse=True),np.int64)
        assert len(rr)==M and rr.sum()==R
        pg=(rr*C+16383)//16384;wa=(pg+3)//4
        assert np.all(wa<=192)
        key=tuple(rr.tolist())
        entry=dict(rows=rr.tolist(),positive_jobs=int(np.count_nonzero(rr)),pages=int(pg.sum()),waves=int(wa.sum()),depth=int(wa.max()),origins=[])
        if key not in menu:menu[key]=entry
        menu[key]['origins'].append(dict(kind=kind,**meta))
    for g in range(1,M+1):
        rr=np.zeros(M,np.int64);rr[:g]=R//g;rr[:R%g]+=1;add(rr,'allwidth',dict(width=g))
    for depth in range(dmin,end+1):
        result=minimum_work(R,C,depth,cap);assert result is not None
        witness=result['witness'];add(witness['rows'],'exact_at_depth',dict(depth=depth))
        frontier.append(dict(depth=depth,waves=result['optimal_total_waves'],rows=witness['rows'],positive_jobs=witness['positive_jobs']))
    balanced=None
    for work in range(wfloor,((np.full(M,R//M)+np.r_[np.ones(R%M,dtype=int),np.zeros(M-R%M,dtype=int)])*C+B-1).__floordiv__(B).sum()+1):
        ww=np.full(M,int(work)//M,np.int64);ww[:int(work)%M]+=1
        candidate=materialize(R,C,ww,cap)
        if candidate is not None:
            balanced=candidate;add(candidate['rows'],'simple_balanced_wave',dict());break
    return dict(R=R,C=C,chips=M,minimum_depth=dmin,exact_work_depth=exactdepth,
                aggregate_wave_floor=wfloor,period=period,frontier=frontier,balanced=balanced,
                menu=list(menu.values()),DP_calls=len(frontier))


def one_table(rr,C,copies):
    rr=np.asarray(rr,np.int64);active=rr>0;pages=(rr*C+16383)//16384;waves=(pages+3)//4;emit=pages*16384//C
    work=active*(250+(C*copies+1)//2)+waves*28000+emit*2
    return work,dict(jobs=active.sum(-1),pages=pages.sum(-1),waves=waves.sum(-1),
                     input_bytes=active.sum(-1)*C*copies,output_bytes=emit.sum(-1)*4)


def routed_maps(geo,E):
    M=geo['chips'];maps={}
    steps=sorted(set([1,3,5,7]+[M//d for d in (2,4,8) if M//d>0]))
    def add(rr,description):
        key=np.asarray(rr,np.int64).tobytes()
        if key not in maps:maps[key]=dict(rows=np.asarray(rr,np.int64),description=description)
    for j,profile in enumerate(geo['menu']):
        base=np.asarray(profile['rows'])
        for step in steps:add(np.stack([np.roll(base,e*step%M) for e in range(E)]),dict(kind='cyclic',profile=j,step=step))
        widths=[x['width'] for x in profile['origins'] if x['kind']=='allwidth' and M%x['width']==0]
        for g in widths:
            groups=M//g;rr=np.zeros((E,M),np.int64)
            band=base[:g]
            for e in range(E):rr[e,(e%groups)*g:(e%groups+1)*g]=np.roll(band,(e//groups)%g)
            add(rr,dict(kind='disjoint',profile=j,width=g))
    return list(maps.values())


def shared_choices(geo):
    menu=geo['menu'];M=geo['chips']
    selected=[min(range(len(menu)),key=lambda j:(menu[j]['waves'],menu[j]['depth'],menu[j]['positive_jobs'])),
              min(range(len(menu)),key=lambda j:(menu[j]['depth'],menu[j]['waves'],menu[j]['positive_jobs']))]
    out=[];seen=set()
    for j in selected:
        for shift in [0,M//2]:
            rr=np.roll(np.asarray(menu[j]['rows']),shift);key=rr.tobytes()
            if key not in seen:seen.add(key);out.append(dict(rows=rr,description=dict(profile=j,shift=shift)))
    return out


def phase_candidates(geo,sg,routes,copies):
    pool=routed_maps(geo,64);shared=shared_choices(sg);records=[];best=[]
    for item in pool:
        rw,counts=one_table(item['rows'],geo['C'],copies);rload=rw[routes].sum(1)
        for sh in shared:
            sw,sc=one_table(sh['rows'],sg['C'],copies);loads=rload+sw
            values=loads.max(1);score=float(values.mean())
            record=dict(mapping=item['description'],shared=sh['description'],owned_work=stats(values),
                        commands_mean=float((counts['jobs'][routes].sum(1)+sc['jobs']).mean()),
                        waves_mean=float((counts['waves'][routes].sum(1)+sc['waves']).mean()),
                        input_bytes_mean=float((counts['input_bytes'][routes].sum(1)+sc['input_bytes']).mean()),
                        output_bytes_mean=float((counts['output_bytes'][routes].sum(1)+sc['output_bytes']).mean()))
            records.append(record)
            candidate=dict(score=score,rows=np.vstack([item['rows'],sh['rows']]),loads=loads,record=record)
            best.append(candidate);best.sort(key=lambda x:x['score']);best=best[:4]
    return records,best,len(pool),len(shared)


def necessary(shape,M):
    H,I,SI,K=[shape[k] for k in ('H','I','SI','K')]
    waves=(K*(2*I*H+H*I)+2*SI*H+H*SI)//65536
    assert (K*(2*I*H+H*I)+2*SI*H+H*SI)%65536==0
    read_waves=(waves+M-1)//M
    routed_depth=((2*I+M-1)//M*H+65535)//65536+((H+M-1)//M*I+65535)//65536
    shared_depth=((2*SI+M-1)//M*H+65535)//65536+((H+M-1)//M*SI+65535)//65536
    # This chain deliberately omits command/input/output costs, making it a
    # necessarybound independent of which integerprofile achieves its depth.
    chain=4000+max(routed_depth,shared_depth)*28000
    outputs=4*(K*(2*I+H)+2*SI+H);uploads=H+K*I+SI
    pcie=(outputs+uploads+7)//8
    commands=2*(K+1)*250
    chan=(outputs+(K+1)*H+K*I+SI+2*min(8,M)-1)//(2*min(8,M))
    return dict(total_minimum_read_waves=waves,pooled_integer_chip_read_waves=read_waves,
                read_only_initial_host_LB_ns=2000+read_waves*28000,
                minimum_routed_expert_chain_depth_waves=routed_depth,
                minimum_shared_expert_chain_depth_waves=shared_depth,depth_only_chain_LB_ns=chain,
                minimum_PCIe_volume_ns=pcie,minimum_controller_work_ns=commands,minimum_channel_volume_ns=chan,
                universal_necessary_ns=max(2000+read_waves*28000,chain,2000+pcie,2000+commands,2000+chan),
                limit='Theseindependentminima neednotbejointlyattainable. Actualcommands/reloads/queues/readydepth maydominatewideprofile service.')


def run():
    p=json.loads((OUT/'protocol.frozen.json').read_text())
    for item in p['inputs']:assert sha(item['path'])==item['sha256']
    assert sha(__file__)==p['source_sha256'] and not (OUT/'report.json').exists()
    t0=time.monotonic();cases=[];geometry_records=[];npz={};calls=0
    for model,shape in MODELS.items():
        with np.load(BASE/f'w8-format-training/{model}/decode.routes.npz') as z:
            assert z['split'].item()=='train' and z['completed'].all();routes=z['routes'][:,:,0].reshape(512,6)
        H,I,SI=[shape[k] for k in ('H','I','SI')]
        for M in CHIPS:
            gs={name:geometry(R,C,M) for name,R,C in [('GU',2*I,H),('Down',H,I),('sharedGU',2*SI,H),('sharedDown',H,SI)]}
            calls+=sum(g['DP_calls'] for g in gs.values());assert calls<=120
            geometry_records.append(dict(model=model,chips=M,shapes=gs))
            lower=necessary(shape,M)
            for copies in (1,4):
                gr,gb,ng,nsg=phase_candidates(gs['GU'],gs['sharedGU'],routes,copies)
                dr,db,nd,nsd=phase_candidates(gs['Down'],gs['sharedDown'],routes,copies)
                best=None
                for ig,gg in enumerate(gb):
                    for idd,dd in enumerate(db):
                        for shift in range(M):
                            loads=gg['loads']+np.roll(dd['loads'],shift,axis=1);values=loads.max(1);score=float(values.mean())
                            if best is None or score<best['score']:
                                best=dict(score=score,GU_rank=ig,Down_rank=idd,Down_shift=shift,values=values,
                                          rows=np.stack([gg['rows'],np.roll(dd['rows'],shift,axis=1)],axis=1))
                tag=f'{model}.M{M}.input{copies}'
                npz[tag+'.rows']=best['rows'];npz[tag+'.owned_work']=best['values'];npz[tag+'.routes']=routes
                counts_by_phase=[]
                for phase,C,SC in [(0,H,H),(1,I,SI)]:
                    _,count=one_table(best['rows'][:64,phase],C,copies);_,sc=one_table(best['rows'][64,phase],SC,copies)
                    counts_by_phase.append({k:float((v[routes].sum(1)+sc[k]).mean()) for k,v in count.items()})
                case=dict(model=model,chips=M,channels=min(8,M),input_copies=copies,necessary=lower,
                          GU_pool=dict(routed_maps=ng,shared_choices=nsg,candidates=len(gr)),Down_pool=dict(routed_maps=nd,shared_choices=nsd,candidates=len(dr)),
                          best_finite_proxy=dict(GU_rank=best['GU_rank'],Down_rank=best['Down_rank'],Down_global_shift=best['Down_shift'],
                                                owned_chip_work=stats(best['values']),phase_counters=counts_by_phase,
                                                GU=gb[best['GU_rank']]['record'],Down=db[best['Down_rank']]['record']),
                          owned_work_optimization_ceiling_pct=100*(1-lower['pooled_integer_chip_read_waves']*28000/best['values'].mean()),
                          limit='Ceilingappliesonlyowned-chipwork objective,not nativeT. Bestfiniteproxy isnotbestnative or globallyoptimizedrouting. No schedule/replay/placementgainacrossMclaimed.')
                cases.append(case);dump(OUT/(tag+'.phase_pool.json'),dict(GU=gr,Down=dr))
                print(json.dumps(dict(model=model,M=M,input_copies=copies,necessary_ns=lower['universal_necessary_ns'],owned_proxy=stats(best['values']),counts=counts_by_phase)),flush=True)
    np.savez_compressed(OUT/'candidate_arrays.npz',**npz)
    dump(OUT/'geometry.json',geometry_records)
    report=dict(status='stageA_geometry_and_ownedwork_only',protocol_sha256=sha(OUT/'protocol.frozen.json'),source_sha256=sha(__file__),
                geometry_DP_calls=calls,cases=cases,seconds=time.monotonic()-t0,
                next_stage_budget=dict(max_candidates_per_hardware=4,priorities=['GF','DF'],queries_per_case=512,total_max_native_queries=32768),
                limits='No nativeperf/fit/replica/heldout/Q38/deviceclaim. SameMfullnative GF/DF,strongfinitegeometry andsamecapacity controls neededbefore adaptiveclaim. Hardwaretargetsdo notprovePEcount.')
    dump(OUT/'report.json',report)
    print(json.dumps(dict(status=report['status'],DP_calls=calls,seconds=report['seconds']),indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['freeze','run']);args=ap.parse_args()
    freeze() if args.mode=='freeze' else run()
