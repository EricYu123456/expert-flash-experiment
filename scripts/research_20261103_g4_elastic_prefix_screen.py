#!/usr/bin/env python3
"""Bounded TRAIN-only elastic g8/g16 GU prefix-replica work oracle."""
import argparse
import ctypes
import hashlib
import json
import shutil
import subprocess
import time
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path

import numpy as np

from research_20261103_topology_placement_replay import CPP as NATIVE_CPP,PYTHON,load_python

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'results/research-2026-11-02/round2'
OUT=BASE/'g4-elastic-prefix-screen/work-v1'
B=BASE/'compact-coupled-solver/continuation/native/0107.npz'
SERVICE=BASE/'compact-coupled-solver/protocol.frozen.json'
TRAIN=BASE/'w8-format-training/granite4tiny/decode.routes.npz'
ORACLE=ROOT/'scripts/research_20261103_g4_down_copy_oracle.cpp'
SEEDS=list(range(202611081,202611089))


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def freeze():
    assert not OUT.exists();OUT.mkdir(parents=True);(OUT/'source-snapshot').mkdir()
    service=json.loads(SERVICE.read_text())['service']
    with np.load(B) as z:rows=z['rows'];routes=z['routes'];mean=float(z['cost'].mean())
    assert rows.shape==(65,2,16) and routes.shape==(512,6) and mean==561099.01953125
    assert np.all(np.sort(rows[:64,0],axis=1)==[40]*7+[82]*3+[83]*6)
    with np.load(TRAIN) as z:assert z['split'].item()=='train' and z['completed'].all() and np.array_equal(routes,z['routes'][:,:,0].reshape(512,6))
    assert service['command_ns']==250 and service['input_channel_copies']==1 and service['output_fringe']==1
    paths=[Path(__file__),ORACLE,B,SERVICE,TRAIN,PYTHON,NATIVE_CPP,ROOT/'scripts/research_20261103_topology_placement_replay.py',ROOT/'scripts/research_20261102_whole_row_compiler.py']
    p=dict(status='frozen_before_TRAIN_fit',utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),baseline_path=str(B),baseline_sha256=sha(B),baseline_native_mean_ns=mean,service=service,inputs=[dict(path=str(x),sha256=sha(x)) for x in paths],routeblind=dict(fixed_excluded_high_rank=list(range(9)),random_seeds=SEEDS,total=17),fit=dict(start='best17routeblind mean exact64-mask work oracle',passes=2,expert_order='descendingTRAIN frequency,tieID',choices_per_expert=9,objective='Mean exact min maxchip fullGU+privateDown+shared native ownershipwork'),static_controls=['SamefixedA/B layout;2passes perexpertbinary choice fromallB/allA,zerocopyrepack','Stronger zerocopy B+all9A geometrymenu;2passes fromallB/allA-selectedlayout,explicitadditionalcontrolbudget'],gate='host2us+minimumwork only,without chipindexdispatch. Also compute independent all64mask sortdescendingW+rank*250 optimalfirststart LB. Report >=2%possibleadvantage before fullnative; no automaticnative/alllayer/heldout/copycontrol expansion.',scope='OnlyG4L0 actualW8TRAIN512. Standard prefixreplication/finitecoordinatesearch,notnewalgorithm/theorem. Shared/Down fixed;completeGU->A8 barrier,fullH input,onecontext retained.',physical='B remains9high82/83+7low40. Auses8of9high128rows. CommonBsource rows remainprefix;copyoffsource rows ascontiguous appended suffix intoAsources. Bfringe85/42rows charged/drop; A128exact. All64experts copies resident evenifneverused. Usefulcopybytes!=page/waveincrement because oldpaddingfills.')
    for item in p['inputs']:
        path=Path(item['path'])
        if path.suffix in ['.py','.cpp']:
            snap=OUT/'source-snapshot'/(item['sha256'][:16]+'-'+path.name);shutil.copy2(path,snap);item['snapshot']=str(snap)
    (OUT/'protocol.frozen.json').write_text(json.dumps(p,indent=2)+'\n')
    print(json.dumps(dict(protocol_sha256=sha(OUT/'protocol.frozen.json'),source_sha256=sha(__file__))))


def logical_image(rows,choice):
    ledger=[];resident=rows.copy();copied=0
    for e in range(64):
        original={c:list(range(int(rows[e,0,:c].sum()),int(rows[e,0,:c+1].sum()))) for c in range(16)}
        high=np.flatnonzero(rows[e,0]>40);excluded=int(high[choice[e]])
        active=[int(c) for c in high if c!=excluded]
        donors=[r for c in range(16) if c not in active for r in original[c]]
        image={c:v.copy() for c,v in original.items()};A={};at=0
        for c in active:
            n=128-len(original[c]);image[c]+=donors[at:at+n];at+=n;A[c]=image[c].copy();resident[e,0,c]=128
        assert at==len(donors) and len(donors) in [362,363]
        assert sorted(r for rr in A.values() for r in rr)==list(range(1024))
        counts=Counter(r for rr in image.values() for r in rr)
        assert all(counts[r]==(2 if r in donors else 1) for r in range(1024))
        assert all(image[c][:len(original[c])]==original[c] for c in range(16))
        copied+=len(donors)
        ledger.append(dict(expert=e,excluded_high_rank=int(choice[e]),excluded_chip=excluded,active_A_chips=active,B_logical_rows=original,A_logical_rows=A,resident_prefix_image=image,copied_logical_rows=donors,copied_rows=len(donors),copied_bytes=len(donors)*1536,base_GU_rows=1024,A_GU_rows=1024,resident_GU_rows=1024+len(donors),GU_resident_waves=33,extra_waves=8,B_emitted_high_rows=85,B_emitted_low_rows=42,A_emitted_rows_per_job=128))
    return ledger,resident,copied


def run():
    p=json.loads((OUT/'protocol.frozen.json').read_text());assert sha(__file__)==p['source_sha256']
    for x in p['inputs']:assert sha(x['path'])==x['sha256']
    started=time.monotonic();cpu=time.process_time();backend=load_python();s=backend.Service(**p['service'])
    with np.load(B) as z:rows=z['rows'];routes=np.ascontiguousarray(z['routes'],np.int32);native=z['cost'];nativeLB=z['lower_bound'];original_plan={k:z[k] for k in ['offsets','jobs','input_columns','total_rows']}
    table=backend.occupancy_table(original_plan,s)['chip_ns'];base=np.ascontiguousarray(table[routes].sum((1,2))+table[64].sum(0),np.int64);baseline_work=base.max(1)
    high=np.stack([np.flatnonzero(rr>40) for rr in rows[:64,0]])
    delta_menu=np.zeros((64,9,16),np.int64);A_menu=np.zeros((64,9,16),np.int64)
    for e in range(64):
        for j in range(9):
            active=np.delete(high[e],j);A_menu[e,j,active]=128
            plan=backend.lower_rows(A_menu[e,j:j+1],rows[e:e+1,1],1536,[512],s)
            cc=backend.occupancy_table(plan,s)['chip_ns'][0].sum(0)
            delta_menu[e,j]=cc-table[e].sum(0)
    binary=OUT/'oracle.so';assert not binary.exists();subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC',str(ORACLE),'-o',str(binary)],check=True)
    lib=ctypes.CDLL(str(binary));fn=lib.down_copy_oracle;i64=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS');i32=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS');fn.argtypes=[i64,i32,i64,i32,ctypes.c_int,i64,i32]
    allq=np.arange(512,dtype=np.int32);members=[np.flatnonzero((routes==e).any(1)).astype(np.int32) for e in range(64)];order=sorted(range(64),key=lambda e:(-len(members[e]),e));calls=queries=0
    def evaluate(delta,ii=allq):
        nonlocal calls,queries
        values=np.zeros(len(ii),np.int64);masks=np.zeros(len(ii),np.int32)
        fn(base,routes,np.ascontiguousarray(delta),ii,len(ii),values,masks);calls+=1;queries+=len(ii)
        return values,masks
    layouts=[np.full(64,j,np.int32) for j in range(9)]
    layouts.extend(np.random.default_rng(seed).integers(0,9,size=64,dtype=np.int32) for seed in SEEDS)
    blind=[];blind_values=[]
    for j,layout in enumerate(layouts):
        delta=delta_menu[np.arange(64),layout];v,m=evaluate(delta);blind_values.append(v)
        blind.append(dict(index=j,rule='fixed_rank' if j<9 else 'random_seed',rank=j if j<9 else None,seed=None if j<9 else SEEDS[j-9],mean_work_ns=float(v.mean()),p99_work_ns=float(np.percentile(v,99))))
    best=min(range(17),key=lambda j:blind[j]['mean_work_ns']);choice=layouts[best].copy();delta=delta_menu[np.arange(64),choice].copy();values=blind_values[best].copy();passes=[];proposals=0
    for pass_index in range(2):
        changes=0
        for e in order:
            ii=members[e];win=int(choice[e]);winv=values[ii].copy();score=int(winv.sum())
            for j in range(9):
                delta[e]=delta_menu[e,j];v,m=evaluate(delta,ii);proposals+=1
                if int(v.sum())<score:score=int(v.sum());win=j;winv=v.copy()
            changes+=win!=choice[e];choice[e]=win;delta[e]=delta_menu[e,win];values[ii]=winv
        full,masks=evaluate(delta);assert np.array_equal(full,values)
        passes.append(dict(pass_index=pass_index+1,changed_experts=int(changes),mean_work_ns=float(values.mean()),p99_work_ns=float(np.percentile(values,99))))
        print('coordinate',passes[-1],flush=True)
    values,masks=evaluate(delta)
    def static(allchoices=False):
        results=[]
        for start in [0,1]:
            selected=choice.copy() if start else np.full(64,-1,np.int32)
            loads=base.copy()
            for e in range(64):
                if selected[e]>=0:loads[members[e]]+=delta_menu[e,selected[e]]
            vv=loads.max(1);hist=[]
            for pi in range(2):
                changes=0
                for e in order:
                    ii=members[e];old=np.zeros(16,np.int64) if selected[e]<0 else delta_menu[e,selected[e]]
                    target=[-1]+list(range(9)) if allchoices else [-1,int(choice[e])]
                    win=int(selected[e]);winloads=loads[ii].copy();winv=vv[ii].copy();score=int(winv.sum())
                    for j in target:
                        new=np.zeros(16,np.int64) if j<0 else delta_menu[e,j];ll=loads[ii]-old+new;v=ll.max(1)
                        if int(v.sum())<score:score=int(v.sum());win=j;winloads=ll;winv=v
                    changes+=win!=selected[e];selected[e]=win;loads[ii]=winloads;vv[ii]=winv
                hist.append(dict(pass_index=pi+1,changed_experts=int(changes),mean_work_ns=float(vv.mean())))
            results.append(dict(start=start,mean_work_ns=float(vv.mean()),p99_work_ns=float(np.percentile(vv,99)),choices=selected,values=vv,passes=hist))
        return min(results,key=lambda r:r['mean_work_ns'])
    fixed_static=static(False);menu_static=static(True)
    bits=((np.arange(64)[:,None]>>np.arange(6))&1).astype(np.int64)
    loads=base[:,None,:]+np.einsum('mk,qkc->qmc',bits,delta[routes])
    exhaustive=loads.max(2);assert np.array_equal(exhaustive.min(1),values)
    assert np.all(exhaustive[np.arange(512),masks]==values)
    # Any first chip-context acquisition is one of16serialized commands. Pair
    # descending work with ascending first-start rank; no fixed chip-ID bias.
    firststart=s.host_submit_ns+(np.sort(loads,axis=2)[:,:,::-1]+np.arange(16)*s.command_ns).max(2)
    sortedLB=firststart.min(1);sortedMasks=firststart.argmin(1).astype(np.int32)
    plainLB=s.host_submit_ns+values;assert np.all(sortedLB>=plainLB)
    ledger,resident,copyrows=logical_image(rows,choice)
    resident_plan=backend.lower_rows(resident[:,0],resident[:,1],1536,[512]*64+[1024],s)
    physical=backend.physical_addresses(resident_plan,s)
    original_rows_plan=backend.lower_rows(rows[:,0],rows[:,1],1536,[512]*64+[1024],s)
    original_physical=backend.physical_addresses(original_rows_plan,s)
    addresses=np.asarray([[a[k] for k in ['chip','block','wordline','reserved_wordlines','pages','first_plane']] for a in physical.pop('addresses')],np.int32)
    original_physical.pop('addresses')
    extra_pages=physical['programmed_page_bytes']-original_physical['programmed_page_bytes'];extra_waves=(physical['reserved_wave_bytes']-original_physical['reserved_wave_bytes'])//65536
    assert extra_pages==33554432 and extra_waves==512
    # Native view positions reuse each resident band's start; shortB prefixes
    # remain legal at the same address and fullA prefixes never span blocks.
    views=[]
    for e in range(64):
        for j in range(*resident_plan['offsets'][e,0]):
            chip=int(resident_plan['jobs'][j,0]);nB=int(rows[e,0,chip]);nA=int(A_menu[e,choice[e],chip])
            views.append(dict(expert=e,chip=chip,address=addresses[j].tolist(),B_rows=nB,B_pages=(nB*1536+16383)//16384,B_waves=(nB*1536+65535)//65536,A_rows=nA,A_pages=(nA*1536+16383)//16384,A_waves=(nA*1536+65535)//65536,logical_row_map_ledger=e))
    saved_static={}
    for name,result in [('same_layout_static',fixed_static),('all9_menu_static',menu_static)]:
        saved_static[name]=dict(result);saved_static[name]['choices']=result['choices'].tolist();saved_static[name].pop('values')
    np.savez_compressed(OUT/'work_arrays.npz',rows_B=rows,routes=routes,base=base,baseline_work=baseline_work,baseline_native=native,baseline_native_LB=nativeLB,A_menu=A_menu,delta_menu=delta_menu,routeblind_layouts=np.stack(layouts),routeblind_values=np.stack(blind_values),selected_layout=choice,selected_delta=delta,selected_values=values,selected_masks=masks,plain_host_LB=plainLB,sorted_firststart_LB=sortedLB,sorted_firststart_masks=sortedMasks,same_layout_static_choices=fixed_static['choices'],same_layout_static_values=fixed_static['values'],all9_static_choices=menu_static['choices'],all9_static_values=menu_static['values'],resident_rows=resident,resident_jobs=resident_plan['jobs'],resident_offsets=resident_plan['offsets'],resident_addresses=addresses)
    native_ceiling=100*(1-plainLB.mean()/native.mean());sorted_ceiling=100*(1-sortedLB.mean()/native.mean())
    budget=dict(copied_rows=copyrows,copied_coefficient_bytes=copyrows*1536,extra_programmed_page_bytes=extra_pages,extra_resident_waves=int(extra_waves),extra_reserved_wave_bytes=int(extra_waves)*65536,padding_reused_bytes=copyrows*1536-extra_pages,baseline_physical=original_physical,resident_physical=physical,wholeGU_comparison=dict(useful_bytes_per_full_GU=1024*1536,min_waves_per_full_GU=24,min_programmed_bytes_per_full_GU=24*65536,full_copies_if_only_coefficient_budget=(copyrows*1536)//(1024*1536),full_copies_if_page_wave_budget=512//24,leftover_waves_after21=512-21*24,warning='Independent wholeGU backup maychoose24wave geometry; do not force25wave B orcopyDown. Blockallocation/perchipcapacity mustalso match; strongcontrols notrun in workstage.'))
    report=dict(status='bounded_work_only',protocol_sha256=sha(OUT/'protocol.frozen.json'),source_sha256=sha(__file__),oracle_source_sha256=sha(ORACLE),oracle_binary_sha256=sha(binary),baseline_native_mean_ns=float(native.mean()),baseline_chip_work_mean_ns=float(baseline_work.mean()),baseline_native_LB_mean_ns=float(nativeLB.mean()),routeblind=blind,best_routeblind_index=best,coordinate_passes=passes,coordinate_proposals=proposals,oracle_calls=calls,oracle_query_evaluations=queries,selected_work_mean_ns=float(values.mean()),selected_work_p99_ns=float(np.percentile(values,99)),work_improvement_pct=100*(1-values.mean()/baseline_work.mean()),plain_host_LB_mean_ns=float(plainLB.mean()),plain_native_possible_ceiling_pct=native_ceiling,sorted_firststart_LB_mean_ns=float(sortedLB.mean()),sorted_native_possible_ceiling_pct=sorted_ceiling,static_controls=saved_static,replica_vs_same_layout_static_work_pct=100*(1-values.mean()/fixed_static['values'].mean()),replica_vs_all9_static_work_pct=100*(1-values.mean()/menu_static['values'].mean()),budget=budget,exhaustive64_masks_checked=32768,gate_possible_ge2pct=bool(native_ceiling>=2),stricter_gate_possible_ge2pct=bool(sorted_ceiling>=2),seconds=time.monotonic()-started,cpu_seconds=time.process_time()-cpu,scope='TRAIN-only necessarywork/lowerbound, notnativeT. No selector/native/largecopycontrols/heldout executed. Finite search only, no globaloptimality/algorithmnovelty claim.')
    for name,obj in [('work_report.json',report),('copy_rowid_ledger.json',ledger),('physical_prefix_views.json',views)]:
        (OUT/name).write_text(json.dumps(obj,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['selected_work_mean_ns','work_improvement_pct','plain_native_possible_ceiling_pct','sorted_native_possible_ceiling_pct','replica_vs_same_layout_static_work_pct','replica_vs_all9_static_work_pct','gate_possible_ge2pct','stricter_gate_possible_ge2pct','seconds']},indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['freeze','run']);a=ap.parse_args();freeze() if a.mode=='freeze' else run()
