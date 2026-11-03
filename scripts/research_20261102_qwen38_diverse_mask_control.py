#!/usr/bin/env python3
"""Route-blind balanced distinct masks: ordinary design/max-flow controls.

Only the original train corpus is read. Construction uses no routes; finite
seed selection uses train objectives, never the32step smoke cost or external.
"""
from collections import deque
from datetime import datetime,timezone
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from research_20261102_qwen38_fit import TRAIN,TRAIN_SHA,sha
from research_20261102_qwen38_replay import inventories,build
from research_20261102_qwen38_fast_replay import encode,evaluate
from research_20261102_qwen38_compiled_campaign import library
from research_20261102_qwen38_independent_audit import coverage,ledger,SHAPES

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/research-2026-11-02/round2/qwen38-placement'
OUT=BASE/'diverse-mask-control'
SEEDS=(20261102,20261103,20261104,20261105)
REMOVED=((0,1,2),(0,1,3),(0,1,4))


class Flow:
    def __init__(self,n):self.adj=[[] for _ in range(n)]
    def edge(self,a,b,cap):
        i=len(self.adj[a]);j=len(self.adj[b])
        self.adj[a].append([b,j,cap]);self.adj[b].append([a,i,0]);return i
    def run(self,source,sink):
        total=0;n=len(self.adj)
        while True:
            level=[-1]*n;level[source]=0;queue=deque([source])
            while queue:
                u=queue.popleft()
                for v,_,cap in self.adj[u]:
                    if cap and level[v]<0:level[v]=level[u]+1;queue.append(v)
            if level[sink]<0:return total
            cursor=[0]*n
            def send(u,cap):
                if u==sink:return cap
                while cursor[u]<len(self.adj[u]):
                    edge=self.adj[u][cursor[u]];v,rev,left=edge
                    if left and level[v]==level[u]+1:
                        got=send(v,min(cap,left))
                        if got:
                            edge[2]-=got;self.adj[v][rev][2]+=got;return got
                    cursor[u]+=1
                return 0
            while True:
                got=send(source,10**9)
                if not got:break
                total+=got


def bmatching(allowed,left_degree,right_degree,rng):
    assert allowed.shape==(512,16)
    assert int(allowed.sum(axis=1).min())==int(allowed.sum(axis=1).max())
    assert int(allowed.sum(axis=0).min())==int(allowed.sum(axis=0).max())
    assert 512*left_degree==16*right_degree
    source=528;sink=529;flow=Flow(530);references={}
    for e in rng.permutation(512):flow.edge(source,int(e),left_degree)
    for c in rng.permutation(16):flow.edge(512+int(c),sink,right_degree)
    for e in rng.permutation(512):
        for c in rng.permutation(np.flatnonzero(allowed[e])):
            references[int(e),int(c)]=flow.edge(int(e),512+int(c),1)
    got=flow.run(source,sink);assert got==512*left_degree
    chosen=np.zeros((512,16),dtype=bool)
    for (e,c),edge in references.items():chosen[e,c]=flow.adj[e][edge][2]==0
    assert np.all(chosen<=allowed) and np.all(chosen.sum(axis=1)==left_degree) and np.all(chosen.sum(axis=0)==right_degree)
    return chosen,dict(total_flow=got,left_degree=left_degree,right_degree=right_degree,
        allowed_left_degree=int(allowed.sum(axis=1)[0]),allowed_right_degree=int(allowed.sum(axis=0)[0]),
        fractional_feasibility=f'{left_degree}/{int(allowed.sum(axis=1)[0])} on every allowed edge; integral max-flow saturates all demands.')


def pool():
    removed=set();orbits=[]
    for triple in REMOVED:
        orbit={tuple(sorted((c+shift)%16 for c in triple)) for shift in range(16)}
        assert len(orbit)==16 and not (removed&orbit);removed|=orbit;orbits.append(sorted(orbit))
    triples=[x for x in combinations(range(16),3) if x not in removed]
    assert len(triples)==512
    masks=np.zeros((512,16),dtype=bool)
    for i,triple in enumerate(triples):masks[i,list(triple)]=True
    assert np.all(masks.sum(axis=0)==96)
    return masks,orbits


def construct(masks,seed,layer):
    rng=np.random.default_rng(np.random.SeedSequence([seed,layer]))
    gu=masks[rng.permutation(512)].copy()
    down,down_flow=bmatching(~gu,10,320,rng)
    gu_plus,gu_flow=bmatching(~gu,2,64,rng)
    down_plus,down_plus_flow=bmatching(down,4,128,rng)
    rows=np.stack([np.where(gu,101,75)+gu_plus,np.where(down,198,96)+down_plus]).astype(np.int32)
    assert np.all(~(gu&down)) and np.all(rows.sum(axis=2)==np.array([1280,2560])[:,None])
    assert np.all(rows.sum(axis=1)==np.array([40960,81920])[:,None])
    assert len({tuple(np.flatnonzero(g)) for g in gu})==512
    cert=dict(seed=seed,layer=layer,distinct_gu_masks=512,gu_high_per_chip=gu.sum(axis=0).tolist(),
        down_high_per_chip=down.sum(axis=0).tolist(),gu_plus_per_chip=gu_plus.sum(axis=0).tolist(),
        down_plus_per_chip=down_plus.sum(axis=0).tolist(),all_gu_down_high_disjoint=True,
        per_expert_rows=rows.sum(axis=2).tolist(),per_chip_rows=rows.sum(axis=1).tolist(),
        flows=dict(down_high=down_flow,gu_low_increment=gu_flow,down_high_increment=down_plus_flow))
    return rows,dict(gu_high=gu,down_high=down,gu_plus=gu_plus,down_plus=down_plus),cert


def rows_from(desc):
    rows=np.zeros((2,512,16),dtype=np.int32)
    for e in range(512):
        for p in range(2):
            for chip,r,c,w in desc[e,p]:
                if r:rows[p,e,chip]=r
    return rows


def descriptor(rows,original):
    desc=original.copy()
    for e in range(512):
        for p,c in enumerate((2560,640)):
            for chip,r in enumerate(rows[p,e]):desc[e,p,chip]=(chip,r,c,(int(r)*c+65535)//65536)
    return np.ascontiguousarray(desc,dtype=np.int32)


def items(desc,selected):
    result=[]
    for e in list(selected)+[512]:
        item=dict(gu_partial=False,down_partial=False)
        for p,name in enumerate(('gate_up','down')):
            cursor=0;jobs=[]
            for chip,r,c,w in desc[e,p]:
                if not r:continue
                jobs.append(dict(chip=int(chip),row_start=cursor,rows=int(r),column_start=0,columns=int(c)));cursor+=int(r)
            assert cursor==(1280,2560)[p];coverage(jobs,SHAPES[name]);item[name]=jobs
        result.append(item)
    return result


def services(desc):
    costs=np.zeros((513,2,16),dtype=np.int64)
    for e in range(513):
        for p in range(2):
            for chip,r,c,w in desc[e,p]:
                if r:costs[e,p,chip]+=int(w)*28000+250+(int(c)+1)//2+2*int(r)
    return costs


def score(cost,selected):
    loads=cost[selected].sum(axis=1)+cost[512]
    phase=loads[:,0].max(axis=1)+loads[:,1].max(axis=1)
    joined=loads.sum(axis=1).max(axis=1)
    return dict(objective=float((joined*1000+phase).mean()),mean_phase_bound_ns=float(phase.mean()),
                mean_joined_bound_ns=float(joined.mean()))


def inventory(rows):
    output={}
    for p,(name,c) in enumerate([('gate_up',2560),('down',640)]):
        a=rows[p].astype(np.int64);pages=(a*c+16383)//16384;waves=(a*c+65535)//65536
        assert np.all(pages.sum(axis=1)==(204,104)[p]) and np.all(waves.sum(axis=1)==(51,26)[p])
        assert np.all(a.sum(axis=0)==(40960,81920)[p])
        output[name]=dict(per_expert_waves=int(waves.sum(axis=1)[0]),per_expert_pages=int(pages.sum(axis=1)[0]),
            per_chip_rows=a.sum(axis=0).tolist(),per_chip_useful_bytes=(a.sum(axis=0)*c).tolist(),
            per_chip_pages=pages.sum(axis=0).tolist(),per_chip_waves=waves.sum(axis=0).tolist(),
            per_chip_reserved_wave_bytes=(waves.sum(axis=0)*65536).tolist(),
            rowtype_counts_per_chip={str(r):(a==r).sum(axis=0).tolist() for r in np.unique(a)})
    return output


def run_case(name,layer,desc,partial,fit,smoke,steps,lib):
    fit_cost=evaluate(lib,desc,partial,fit,1000,0,False);smoke_cost=evaluate(lib,desc,partial,smoke,1000,0,False)
    cost=services(desc);fit_bound=cost[fit].sum(axis=1).sum(axis=1)+cost[512].sum(axis=0)
    fit_bound=fit_bound.max(axis=1)+2000
    smoke_bound=(cost[smoke].sum(axis=1).sum(axis=1)+cost[512].sum(axis=0)).max(axis=1)+2000
    assert np.all(fit_cost>=fit_bound) and np.all(smoke_cost>=smoke_bound)
    checks=[]
    for ix in (0,len(smoke)//2,len(smoke)-1):
        diagnostic=build(items(desc,smoke[ix]),1000,0).run(False,trace=ix==0)
        assert diagnostic['makespan_ns']==int(smoke_cost[ix]);checks.append(dict(index=ix,exact=True))
        if ix==0:
            (OUT/f'L{layer}.{name}.trace.json').write_text(json.dumps(dict(layer=layer,case=name,selected=smoke[ix].tolist(),items=items(desc,smoke[ix]),**diagnostic),indent=2)+'\n')
    target=OUT/f'L{layer}.{name}.npz'
    np.savez_compressed(target,descriptor=desc,partial=partial,rows=rows_from(desc),train_cost=fit_cost,
        cost=smoke_cost,native_chip_bound=smoke_bound,train_native_chip_bound=fit_bound,token_steps=steps)
    return dict(layer=layer,name=name,train_score=score(cost,fit),train_mean_ns=float(fit_cost.mean()),
        train_p99_ns=float(np.percentile(fit_cost,99)),smoke_mean_ns=float(smoke_cost.mean()),
        smoke_p99_ns=float(np.percentile(smoke_cost,99)),smoke_mean_bound_ns=float(smoke_bound.mean()),
        python_checks=checks,npz=str(target),npz_sha256=sha(target),trace_sha256=sha(OUT/f'L{layer}.{name}.trace.json'))


def main():
    assert not (OUT/'summary.json').exists();OUT.mkdir(parents=True,exist_ok=True)
    assert sha(TRAIN)==TRAIN_SHA
    baseplans=[BASE/f'joint-smoke/L{l}.whole.global_complement.r1000.plan.json' for l in range(3)]
    matched=[BASE/f'residue-mask-pilot/L{l}.m0.o1.npz' for l in range(3)]
    sources=[Path(__file__),ROOT/'scripts/research_20261102_native_replay_v2.cpp',ROOT/'scripts/research_20261102_qwen38_replay.py',
             ROOT/'scripts/research_20261102_qwen38_fast_replay.py',ROOT/'scripts/research_20261102_granite4_pipeline.py',
             ROOT/'scripts/research_20261102_qwen38_compiled_campaign.py',ROOT/'scripts/research_20261102_qwen38_independent_audit.py',
             BASE/'native_replay_v2.so']+baseplans+matched
    hashes={str(p):sha(p) for p in sources}
    protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),seeds=SEEDS,removed_orbit_representatives=REMOVED,
        construction='All560 GU triples minus3distinct16-element cyclic orbits. Random assignment to expert IDs, then integral regular bipartite b-matchings; no route used by construction.',
        fit='All16 original train requests at token0,4,...796:3200queries/layer. Shared expert fixed to source complement plan.',
        primary_selection='Per-layer minimum mean(joined-native-chip-bound*1000 + sum-phase-chip-bounds) over4fixedseed plans, matching pilotobjective1. Frozen before evaluating smoke.',
        secondary_selection='Also report finite4seed per-layer minimum actual native training mean and global best seed by training mean; never choose by smoke.',
        evaluation='Exactly32linspace token steps/request:512queries/layer, three FFN layers, gate-first,0activation,1000ns reconstruction parameter (unused by whole rows).',
        baseline_order='diverse and matchedcyclic use pilotcanonical chip0..15 job order. Original complement is replayed in both originaltemplate and canonicalchip orders.',
        source_inputs_sha256=hashes,training_sha256=TRAIN_SHA,external_payloads_opened=False,
        novelty='Ordinary balanced design/max-flow baseline; no new algorithm, hardware, quantization or native context.')
    (OUT/'protocol.before_generation.json').write_text(json.dumps(protocol,indent=2)+'\n')
    masks,orbits=pool();all_rows={};certificates=[]
    for layer in range(3):
        for seed in SEEDS:
            rows,mask_arrays,cert=construct(masks,seed,layer);all_rows[layer,seed]=rows
            target=OUT/f'L{layer}.seed{seed}.construction.npz';np.savez_compressed(target,rows=rows,**mask_arrays)
            cert.update(inventory=inventory(rows),npz_sha256=sha(target));certificates.append(cert)
    generated=dict(created_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),orbits=orbits,
        certificates=certificates,construction_complete_before_training_routes_loaded=True)
    (OUT/'construction.frozen.json').write_text(json.dumps(generated,indent=2)+'\n')
    # Only now load routes, after every seed's physical construction is fixed.
    with np.load(TRAIN) as f:routes=f['routes'].copy()
    inv=inventories();lib=library();records=[];chosen={};contexts={}
    steps=np.unique(np.linspace(0,798,32,dtype=int))
    for layer in range(3):
        plan=json.loads(baseplans[layer].read_text());original,partial=encode(plan,inv);initial=rows_from(original)
        assert np.all((initial[0]>=101)&(initial[1]>=198)==False)
        fit=routes[:,::4,layer].reshape(-1,10);smoke=routes[:,steps,layer].reshape(-1,10)
        candidates=[]
        for seed in SEEDS:
            rows=all_rows[layer,seed]
            assert np.array_equal(np.sort(rows,axis=2),np.sort(initial,axis=2))
            assert np.array_equal(np.sort(rows,axis=1),np.sort(initial,axis=1))
            desc=descriptor(rows,original);candidates.append((score(services(desc),fit)['objective'],seed))
        chosen[layer]=min(candidates)[1];contexts[layer]=(original,partial,initial,fit,smoke)
    selection=dict(created_utc=datetime.now(timezone.utc).isoformat(),primary_seeds_by_layer=chosen,
        construction_sha256=sha(OUT/'construction.frozen.json'),selected_using_train_only=True,
        smoke_evaluated_before_selection=False)
    (OUT/'selection.before_smoke.json').write_text(json.dumps(selection,indent=2)+'\n')
    for layer in range(3):
        original,partial,initial,fit,smoke=contexts[layer]
        for seed in SEEDS:
            desc=descriptor(all_rows[layer,seed],original)
            records.append(run_case(f'seed{seed}',layer,desc,partial,fit,smoke,steps,lib))
        with np.load(matched[layer]) as a:
            desc=a['descriptor'].copy();old_cost=a['cost'].copy();old_steps=a['token_steps'].copy()
        assert np.array_equal(desc[512],original[512]) and np.array_equal(old_steps,steps)
        r=run_case('matched_cyclic',layer,desc,partial,fit,smoke,steps,lib)
        assert np.array_equal(np.load(r['npz'])['cost'],old_cost);records.append(r)
        records.append(run_case('original_complement_nativeorder',layer,original,partial,fit,smoke,steps,lib))
        records.append(run_case('original_complement_chiporder',layer,descriptor(initial,original),partial,fit,smoke,steps,lib))
    by={(r['layer'],r['name']):r for r in records}
    combinations_to_report={f'seed{seed}':{l:f'seed{seed}' for l in range(3)} for seed in SEEDS}
    combinations_to_report['train_bound_selected']={l:f'seed{chosen[l]}' for l in range(3)}
    native_chosen={l:min(SEEDS,key=lambda s:by[l,f'seed{s}']['train_mean_ns']) for l in range(3)}
    combinations_to_report['train_native_mean_selected']={l:f'seed{native_chosen[l]}' for l in range(3)}
    global_chosen=min(SEEDS,key=lambda s:sum(by[l,f'seed{s}']['train_mean_ns'] for l in range(3)))
    for name in ['matched_cyclic','original_complement_nativeorder','original_complement_chiporder']:
        combinations_to_report[name]={l:name for l in range(3)}
    totals=[];paired={}
    for name,choices in combinations_to_report.items():
        train=[];smoke=[]
        for l,c in choices.items():
            with np.load(by[l,c]['npz']) as a:train.append(a['train_cost']);smoke.append(a['cost'])
        t=np.stack(train).sum(axis=0);s=np.stack(smoke).sum(axis=0);paired[name]=s
        totals.append(dict(name=name,choices=choices,train_mean_ns=float(t.mean()),train_p99_ns=float(np.percentile(t,99)),
            smoke_mean_ns=float(s.mean()),smoke_p99_ns=float(np.percentile(s,99))))
    matched_cost=paired['matched_cyclic'];comparisons=[]
    for name,value in paired.items():
        comparisons.append(dict(candidate=name,baseline='matched_cyclic',smoke_mean_gain_pct=100*(1-value.mean()/matched_cost.mean()),
            smoke_p99_gain_pct=100*(1-np.percentile(value,99)/np.percentile(matched_cost,99)),
            smoke_queries_improved=int((value<matched_cost).sum()),smoke_queries_worsened=int((value>matched_cost).sum())))
    assert all(sha(p)==h for p,h in hashes.items()) and sha(TRAIN)==TRAIN_SHA
    np.savez_compressed(OUT/'paired_smoke_costs.npz',token_steps=steps,**paired)
    output=dict(status='complete train-only route-blind diverse-mask strong control',created_utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=sha(OUT/'protocol.before_generation.json'),construction_sha256=sha(OUT/'construction.frozen.json'),
        selection_sha256=sha(OUT/'selection.before_smoke.json'),primary_seeds_by_layer=chosen,native_mean_seeds_by_layer=native_chosen,
        best_global_seed_by_train_native_mean=global_chosen,records=records,totals=totals,comparisons=comparisons,
        independent_python_checks=sum(len(r['python_checks']) for r in records),source_inputs_sha256=hashes,
        limits=['All routes are original training task; smoke is in-task descriptive validation, not heldout/external.',
            'Construction is route-blind but choosing among4fixedseeds uses training routes. No subsequent swaps or learning.',
            'Rows are onecopy full-column ordinary densejobs; no partial sums, changed precision, extra copies or native contexts.',
            'NominalI8/A8/4B-output service and28us NAND are uncalibrated cost-model assumptions, not originalNVFP4checkpoint execution or silicon measurements.',
            'Only3FFNlayers including sourcefixedsharedexpert; not whole-model latency or publication novelty.'])
    (OUT/'summary.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(dict(totals=totals,comparisons=comparisons,python_checks=output['independent_python_checks'],summary_sha256=sha(OUT/'summary.json')),indent=2))


if __name__=='__main__':main()
