#!/usr/bin/env python3
"""Train32x64 screen of full-profile copies and generic replicated declustering.

No heldout data, GU copies, extra contexts, new arithmetic, or decoder operands.
"""
import ctypes
from datetime import datetime,timezone
from itertools import product
import json
import os
from pathlib import Path
import subprocess

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow

from research_20261102_projection_study import evaluate
from run_corpus_traces import ROOT,sha256

DEST=ROOT/'results/research-2026-11-02/round2/compound-baseline-screen'
THM=ROOT/'results/research-2026-11-02/round2/compound-replica/theorem.json'
MAIN=ROOT/'results/research-2026-11-02/round2/geometry-study/granite4tiny.frozen.json'
ORACLE_COST=166912


def summary(a):
    return dict(mean_ns=float(a.mean()),p95_ns=float(np.percentile(a,95)),p99_ns=float(np.percentile(a,99)),
        min_ns=int(a.min()),max_ns=int(a.max()),fraction_at_5wave_paired_bound=float(np.mean(a==ORACLE_COST)))


def loadlib():
    cpp=Path(__file__).with_suffix('.cpp');target=DEST/('selector-'+sha256(cpp)+'.so')
    if not target.exists():
        temp=target.with_name(target.name+'.'+str(os.getpid())+'.tmp')
        subprocess.run(['g++','-std=c++17','-O3','-march=native','-shared','-fPIC',str(cpp),'-o',str(temp)],check=True)
        os.replace(temp,target)
    lib=ctypes.CDLL(str(target));p32=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS');p64=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS')
    lib.optimize_whole_copy.argtypes=[p32,p32,p32,p64]+[ctypes.c_int]*4+[ctypes.c_uint64,p64]
    lib.optimize_whole_copy.restype=ctypes.c_double
    lib.evaluate_whole_copy.argtypes=[p32,p32,p32,ctypes.c_int,ctypes.c_int,p64]
    return lib


def flow_query(route,locations):
    # task -> (expert,chip) cap1 -> chip cap5 -> physical channel cap9.
    # Thus no expert has two native tasks on the same chip, and all inputs
    # are paid once per task. All tasks are ordinary 128x512 dense GEMVs.
    K=len(route);ntasks=K*12;ec0=ntasks;chip0=ec0+K*16;chan0=chip0+16;src=chan0+8;sink=src+1
    row=[];col=[];val=[]
    def edge(a,b,c):row.append(a);col.append(b);val.append(c)
    for k,e in enumerate(route):
        for t in range(12):
            task=k*12+t;edge(src,task,1)
            for chip in locations[e][t]:edge(task,ec0+k*16+chip,1)
        for chip in range(16):edge(ec0+k*16+chip,chip0+chip,1)
    for chip in range(16):edge(chip0+chip,chan0+chip%8,5)
    for ch in range(8):edge(chan0+ch,sink,9)
    graph=csr_matrix((np.asarray(val,dtype=np.int64),(row,col)),shape=(sink+1,sink+1))
    solved=maximum_flow(graph,src,sink);assert solved.flow_value==72
    loads=np.zeros(16,dtype=int);jobs=[];seen=set()
    for k,e in enumerate(route):
        for t in range(12):
            candidates=[c for c in locations[e][t] if solved.flow[k*12+t,ec0+k*16+c]>0]
            assert len(candidates)==1;c=candidates[0]
            assert (int(e),c) not in seen;seen.add((int(e),c));loads[c]+=1;jobs.append((int(e),t,c))
    assert loads.max()==5 and np.all(loads[:8]+loads[8:]==9)
    return loads,jobs


def python_profile(route,base,alt):
    best=10**9
    for choices in product(*[(int(base[e]),int(alt[e])) if alt[e]>=0 else (int(base[e]),) for e in route]):
        counts=np.bincount(choices,minlength=4);peak=6-counts.min();pair=min(counts[0]+counts[2],counts[1]+counts[3])
        best=min(best,int(peak*28000+18000+2000+256*(12-pair)+4608))
    return best


def main():
    DEST.mkdir(parents=True,exist_ok=True);lib=loadlib()
    theorem=json.loads(THM.read_text());main_plan=json.loads(MAIN.read_text());initial_hashes={str(THM):sha256(THM),str(MAIN):sha256(MAIN)}
    assert theorem['primary_tiles']==768 and theorem['extra_copies']==688
    route_path=Path(main_plan['training_routes']);assert sha256(route_path)==main_plan['training_routes_sha256']
    with np.load(route_path) as f:
        routes=f['routes'][:32,:64].copy();ids=f['prompt_ids'][:32].tolist();layers=f['layers'].tolist()
    assert len(ids)==32 and all(i.startswith('train_') for i in ids)
    locations=[[None]*12 for _ in range(64)]
    for tile in theorem['tiles']:
        locations[tile['expert']][tile['row_start']//128]=tile['locations']
    copied_count=688//12;assert copied_count==57
    results=[];archive={};rng=np.random.default_rng(20261102)
    for li,layer in enumerate(layers[:3]):
        route_all=np.ascontiguousarray(np.sort(routes[:,:,li].reshape(-1,6),axis=1),dtype=np.int32)
        unique,inverse,weights=np.unique(route_all,axis=0,return_inverse=True,return_counts=True)
        unique=np.ascontiguousarray(unique,dtype=np.int32);weights=np.ascontiguousarray(weights,dtype=np.int64)
        phase=next(p for p in main_plan['phases'] if p['layer']==layer and p['name']=='down')
        rows=np.asarray(phase['rows']);assert np.all(np.count_nonzero(rows,axis=1)==12)
        base=np.ascontiguousarray([next(g for g in range(4) if not rows[e,g*4:g*4+4].any()) for e in range(64)],dtype=np.int32)
        popularity=np.bincount(route_all.ravel(),minlength=64)
        copied=np.argsort(-popularity,kind='stable')[:copied_count]
        best_cost=np.inf;best_alt=None;search=[]
        for restart in range(5):
            alt=np.full(64,-1,dtype=np.int32)
            alt[copied]=(base[copied]+rng.integers(1,4,size=copied_count))%4
            stats=np.zeros(2,dtype=np.int64)
            cost=lib.optimize_whole_copy(base,alt,unique,weights,64,len(unique),6,12,20261102+li*101+restart,stats)
            search.append(dict(restart=restart,mean_ns=cost,proposals=int(stats[0]),accepted=int(stats[1])))
            if cost<best_cost:best_cost=cost;best_alt=alt.copy()
        costs=np.zeros(len(unique),dtype=np.int64);lib.evaluate_whole_copy(base,best_alt,unique,len(unique),6,costs)
        assert abs(float(costs@weights/weights.sum())-best_cost)<1e-6
        for t in sorted({0,len(unique)//4,len(unique)//2,len(unique)-1}):assert python_profile(unique[t],base,best_alt)==costs[t]
        no_replica=evaluate(route_all,rows,512,False)['model_ns']
        assert abs(float(no_replica.mean())-phase['chosen_ns'])<1e-6
        # Generic RDA is evaluated on the exact same theorem locations. It can
        # reproduce the compound selector and is not handicapped to whole profiles.
        flow_cost=np.full(len(unique),ORACLE_COST,dtype=np.int64);samples=[]
        for qi,query in enumerate(unique):
            loads,jobs=flow_query(query,locations)
            if qi in {0,len(unique)//2,len(unique)-1}:samples.append(dict(query=query.tolist(),chip_loads=loads.tolist(),jobs=jobs))
        whole=costs[inverse];generic=flow_cost[inverse];compound=np.full(len(route_all),ORACLE_COST,dtype=np.int64)
        archive['layer'+str(layer)+'_whole']=whole;archive['layer'+str(layer)+'_generic_flow']=generic
        archive['layer'+str(layer)+'_no_replica']=no_replica
        stored=np.zeros(16,dtype=int)
        for e in range(64):
            for omit in ([int(base[e]),int(best_alt[e])] if best_alt[e]>=0 else [int(base[e])]):
                for c in range(16):stored[c]+=c//4!=omit
        assert stored.sum()==768+57*12
        result=dict(layer=layer,unique_queries=len(unique),tokens=len(route_all),base_omitted_color=base.tolist(),
            copied_experts=copied.tolist(),unreplicated_experts=np.flatnonzero(best_alt<0).tolist(),secondary_omitted_color=best_alt.tolist(),
            whole_copy_extra_tiles=684,whole_copy_unused_budget_tiles=4,whole_copy_stored_tiles_per_chip=stored.tolist(),
            no_replica_g12=summary(no_replica),whole_expert_copy=summary(whole),compound_shared_profile=summary(compound),generic_RDA_tile_flow=summary(generic),
            whole_mean_gap_vs_compound_ns=float(whole.mean()-ORACLE_COST),whole_p99_gap_vs_compound_ns=float(np.percentile(whole,99)-ORACLE_COST),
            generic_flow_gap_vs_compound_ns=0.,flow_all_train_unique_queries_verified=True,flow_assignment_samples=samples,search=search)
        results.append(result)
        print(layer,'whole',summary(whole),'genericflow',summary(generic),'unique',len(unique),flush=True)
        (DEST/'partial.json').write_text(json.dumps(results,indent=2)+'\n')
    npz=DEST/'train_costs.npz';np.savez_compressed(npz,**archive)
    assert all(sha256(Path(p))==digest for p,digest in initial_hashes.items())
    output=dict(status='complete exploratory train-only baseline screen',frozen_utc=datetime.now(timezone.utc).isoformat(),
        results=results,train_ids=ids,steps=64,layers=layers[:3],training_routes=str(route_path),training_routes_sha256=sha256(route_path),
        input_hashes=initial_hashes,cost_archive=str(npz),cost_archive_sha256=sha256(npz),
        storage_budget=dict(base_down_tiles=768,extra_down_tiles=688,extra_bytes=688*65536,
            extra_down_payload_fraction=688/768,full_routed_FFN_extra_payload_fraction=(688/768)/3,
            GU_extra_copies=0,explanation='GU has twice the down payload for Granite4Tiny; only down is replicated. Whole-copy policy uses57 complete12tile backups=684extra tiles and cannot use4remaining tiles as another whole profile.'),
        prior_theory=dict(generic_RDA_superset='With the same688 physical tile locations, generic integral task-to-chip flow contains every compound profile assignment, even after imposing at most one tile per expert/chip and exact9task/channel input balance. Mean/p99 improvement over this baseline is impossible in the modeled resource costs.',
            whole_profile_necessary_availability='For every6-expert query to cover every omission color, at most5 experts may lack each color. Hence at least4*(64-5)=236 logical expert-profile incidences are necessary. Given64 baseprofiles, at least172 extra fullprofiles (2064extra tiles if stored as independent12tile copies) are necessary in this restricted4-profile family. A2profile-per-expert policy cannot give the universal guarantee even if every expert is copied.',
            distinction='The profile-incidence bound concerns independent complete stored profiles; selective shared-row replicas and generic replicated declustering evade this duplication cost. It is not a global lower bound beyond the canonical12tile and4omit-profile family.'),
        native_costs='Every query executes72 ordinary128x512 GEMVs, one task/chip/expert;250ns each, all512B task inputs andFP32 outputs paid.5waves*28us+18uscommands+2ushost+2304nschannelinput+4608nsoutput=166912ns. No extra execution contexts or GU copies.',
        limitations=['Whole-copy hot57 membership is a frequency baseline, with5 restarts and exact coordinate optimization of the second omit color; no global optimality or optimized membership claim.',
            'Generic flow is deliberately given exactly the same fixed physical locations as compound, exposing the direct prior-art superset rather than handicapping placement.',
            'Tile payload/pages are counted; allocator block padding, GGUF scales, unrelated weights and device capacity are not equated to payload.',
            'No real model numeric or physical latency experiment; true released-checkpoint routes drive an ordinary-native-command arithmetic model.',
            'Neither main plans nor theorem artifacts changed, and no heldout data was opened.'],
        source_sha256={p.name:sha256(p) for p in [Path(__file__),Path(__file__).with_suffix('.cpp')]})
    (DEST/'completion.json').write_text(json.dumps(output,indent=2)+'\n')
    print('complete',sha256(DEST/'completion.json'),flush=True)


if __name__=='__main__':main()
