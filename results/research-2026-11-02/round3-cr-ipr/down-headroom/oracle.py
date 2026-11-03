#!/usr/bin/env python3
"""TRAIN-only fixed-copy complete-row Down chip-work headroom screen.

All objectives below are chip occupation relaxations, never fullFFN costs.
"""
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

DEST=Path(__file__).resolve().parent
ROOT=DEST.parents[3]
BASE=ROOT/'results/research-2026-11-02/round3-cr-ipr/full-ffn'
PIPE=ROOT/'scripts/research_20261102_cr_ipr_pipeline.py'
TRAIN=ROOT/'results/research-2026-11-02/round2/w8-format-training/granite4tiny/decode.routes.npz'
MENU=BASE/'asymmetric-down/Down_menus.frozen.npz'
PAGES=48

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stats(x):
    a=np.asarray(x);return dict(mean=float(a.mean()),p99=float(np.percentile(a,99)),minimum=float(a.min()),maximum=float(a.max()))
def save(name,x):(DEST/name).write_text(json.dumps(x,indent=2)+'\n')

def load_pipeline():
    assert sha(PIPE)=='ad35d08e17b4458cb5a0f54732aafce338d0526f5888fb02fcfdcc507ad99905'
    sp=importlib.util.spec_from_file_location('down_oracle_pipe',PIPE)
    p=importlib.util.module_from_spec(sp);sys.modules[sp.name]=p;sp.loader.exec_module(p);return p

def price_tables(pipe,copies):
    tables=[];proof=[]
    for columns in (512,1024):
        prices=np.full(97,np.nan);prices[0]=0
        for count in range(1,97):
            waves=[]
            for begin in range(0,count,4):
                pages=[dict(page_id=f'x.p{j}',owner='x',page_index=j,plane=j%4,block=0,wl=j//4,bytes=16384) for j in range(begin,min(begin+4,count))]
                waves.append(dict(pages=pages,source_order=[p['plane'] for p in pages]))
            r=pipe.simulate(waves,cache_copy_ns=500,source_policy='fixed',row_bytes=columns)
            if r['feasible']:
                # One owner: command + input + compute with final ONFI tail.
                prices[count]=250+columns*copies//2+r['makespan_ns']+(16384//columns*4)//2
                assert r['makespan_ns']==28500+2560*count
        tables.append(prices)
        proof.append(dict(columns=columns,copy500_valid_page_counts=np.flatnonzero(np.isfinite(prices)).tolist(),
                          job_work_formula=f'28500+2560*pages+250+{columns}*copies/2+{(16384//columns*4)//2}',
                          interpretation='Mandatory chip-owner occupation with no channel waiting. Last-page output4B/row; earlier output can overlap compute.'))
    return tables,proof

def incidence(routes):
    A=np.zeros((len(routes),65),dtype=np.int64)
    for i,route in enumerate(routes):A[i,route]=1
    A[:,64]=1;return A

def costs(pages,tables):
    return np.concatenate((tables[0][pages[:64]],tables[1][pages[64:65]]),axis=0)

def load(A,pages,tables):return A@costs(pages,tables)
def reserve(pages):return ((pages+3)//4).sum(axis=0)
def capacity(pages):return bool(np.all(128+reserve(pages)<=192))

def shared_search(A,start,tables):
    groups=np.argmax(start[:64]>0,axis=1)//4
    n=A[:,:64]@np.eye(4,dtype=np.int64)[groups]
    assert np.all(start[:64].sum(axis=1)==48)
    private=tables[0][12]*n
    best=None;all_records=[]
    for p0 in range(25):
      for p1 in range(25-p0):
       for p2 in range(25-p0-p1):
        p=np.array([p0,p1,p2,24-p0-p1-p2],dtype=np.int64)
        candidate=start.copy();candidate[64]=np.repeat(p,4)
        if not capacity(candidate):continue
        critical=(private+tables[1][p][None,:]).max(axis=1)
        key=(float(critical.mean()),tuple(p))
        if best is None or key<best[0]:best=(key,candidate,critical)
        all_records.append(dict(group_pages=p.tolist(),mean_ns=float(critical.mean()),p99_ns=float(np.percentile(critical,99))))
    assert len(all_records)==2925
    return best[1],best[2],all_records

def refine(A,start,tables,seed,variable=False,budget=4096):
    rng=np.random.default_rng(seed);pages=start.copy();W=load(A,pages,tables);critical=W.max(axis=1);score=float(critical.sum())
    active=[np.flatnonzero(A[:,e]) for e in range(65)];accepted=valid=0
    for _ in range(budget):
        e=int(rng.integers(65 if variable else 64));positive=np.flatnonzero(pages[e]>0)
        src=int(rng.choice(positive));dst=int(rng.integers(16))
        if dst==src:continue
        amount=int(rng.choice([1,4,int(pages[e,src])])) if variable else 12
        if not variable and pages[e,dst]!=0:continue
        if amount>pages[e,src]:continue
        price=tables[int(e==64)];old_src=int(pages[e,src]);old_dst=int(pages[e,dst]);new_src=old_src-amount;new_dst=old_dst+amount
        if new_dst>=len(price) or not np.isfinite(price[new_src]) or not np.isfinite(price[new_dst]):continue
        old_res=reserve(pages)[[src,dst]]
        new_res=old_res+np.array([(new_src+3)//4-(old_src+3)//4,(new_dst+3)//4-(old_dst+3)//4])
        if np.any(128+new_res>192):continue
        ids=active[e];trial=W[ids].copy();trial[:,src]+=price[new_src]-price[old_src];trial[:,dst]+=price[new_dst]-price[old_dst]
        values=trial.max(axis=1);delta=float((values-critical[ids]).sum());valid+=1
        if delta < -0.5:
            pages[e,src]=new_src;pages[e,dst]=new_dst;W[ids]=trial;critical[ids]=values;score+=delta;accepted+=1
    assert capacity(pages) and np.array_equal(pages.sum(axis=1),start.sum(axis=1))
    assert np.array_equal(W,load(A,pages,tables))
    return pages,critical,dict(proposals=budget,valid=valid,accepted=accepted,mean_ns=score/len(A))

def dynamic_bound(tables):
    br=tables[0][1]-2560;bs=tables[1][1]-2560
    def required(T):
        pr=max([0]+[p for p in range(1,97) if np.isfinite(tables[0][p]) and tables[0][p]<=T])
        ps=max([0]+[p for p in range(1,97) if np.isfinite(tables[1][p]) and tables[1][p]<=T])
        if not pr or not ps:return False
        nr=6*((48+pr-1)//pr);ns=(96+ps-1)//ps
        return 384*2560+nr*br+ns*bs<=16*T
    lo,hi=0,500000
    while lo+1<hi:
        mid=(lo+hi)//2
        if required(mid):hi=mid
        else:lo=mid
    assert hi==tables[1][24]
    assert 12*tables[0][24]+4*tables[1][24]<=16*hi
    return dict(exact_relaxed_minmax_ns=int(hi),previous_integer_ns_rejected=not required(hi-1),
                witness='Each of six selected routed experts takes two disjoint chips,24pages/chip. Shared takes the remaining4chips,24pages/chip. All16chips have exactlyone job; six CR waves/job pass finitePB.',
                caveat='Route-dependent support assignment ignores fixed single-copy placement and GU/channel/command dependencies. This is a lower bound, not an executable fixed mapping.')

def single_chip_finish(local_ready,jobs):
    finish=local_ready
    for release,duration in sorted(jobs):finish=max(finish,release)+duration
    return finish

def g8_release_oracle(routes,down_pages,groups,tables):
    early=[];late=[];exposure=[]
    for route in routes:
        count=np.bincount(groups[route],minlength=2)
        gu=28500+(count+1)*12*2560+250+768
        shared_first=28500+12*2560+250+768
        result=[]
        for shared_ready in (shared_first,float(gu.max())):
            per_chip=[]
            for chip in range(16):
                jobs=[(float(gu[groups[e]]),float(tables[0][down_pages[e,chip]])) for e in route if down_pages[e,chip]]
                if down_pages[64,chip]:jobs.append((shared_ready,float(tables[1][down_pages[64,chip]])))
                per_chip.append(single_chip_finish(float(gu[chip//8]),jobs))
            result.append(max(per_chip)+2000)
        early.append(result[0]);late.append(result[1]);exposure.append(result[1]-result[0])
    return dict(shared_first_relaxation_ns=stats(early),shared_last_relaxation_ns=stats(late),
                shared_last_minus_first_relaxed_ns=stats(exposure),
                interpretation='Optimistic chip-exclusive completion floors: GU uses unlimited-buffer ECC floor; ignores global resources and host barriers. Difference is exposure within this relaxation, not a bound or measurement of real scheduling speedup.')

def main():
    inputs=[PIPE,TRAIN,MENU,DEST/'plan.frozen.json',BASE/'g8-addendum/group_maps.frozen.npz']
    save('inputs.json',dict(inputs_sha256={str(p):sha(p) for p in inputs},source_sha256=sha(__file__)))
    with np.load(TRAIN) as z:routes=z['routes'].transpose(2,0,1,3).reshape(40,512,6)[:3].astype(np.int64)
    with np.load(MENU) as z:maps=z['g4_reused_TRAIN_labels'].copy();g8down=z['g2_reused_TRAIN_labels'].copy()
    with np.load(BASE/'g8-addendum/group_maps.frozen.npz') as z:g8groups=z['learned'].copy()
    pipe=load_pipeline();tables,proof=price_tables(pipe,1);tables4,_=price_tables(pipe,4)
    records=[];arrays={};witnesses={};g8=[]
    for layer in range(3):
        A=incidence(routes[layer]);start=np.concatenate((maps[layer,:64]//32,maps[layer,64:65]//16),axis=0).astype(np.int64)
        assert capacity(start)
        initial=load(A,start,tables).max(axis=1)
        shared,shared_critical,all_shared=shared_search(A,start,tables)
        free,free_critical,free_receipt=refine(A,shared,tables,20261102+layer,False)
        variable,variable_critical,var_receipt=refine(A,free,tables,20261202+layer,True)
        for label,pages,critical,receipt in [('baseline_g4_shared16',start,initial,None),('shared_exact',shared,shared_critical,None),('free4_equal_rows',free,free_critical,free_receipt),('variable_rows_support',variable,variable_critical,var_receipt)]:
            W=load(A,pages,tables);four=load(A,pages,tables4).max(axis=1)
            assert np.array_equal(W.max(axis=1),critical)
            assert pages[:64].sum()==64*48 and pages[64].sum()==96
            records.append(dict(layer=layer,label=label,critical_work_ns=stats(critical),input4_same_map_critical_work_ns=stats(four),
                                total_work_ns=stats(W.sum(axis=1)),mean_reduction_vs_baseline=1-float(critical.mean()/initial.mean()),
                                resident_combined_max_WL=int((128+reserve(pages)).max()),
                                positive_jobs_per_resident_expert=np.count_nonzero(pages[:64],axis=1).tolist(),
                                shared_pages_per_chip=pages[64].tolist(),receipt=receipt))
            arrays[f'L{layer}.{label}.critical']=critical;arrays[f'L{layer}.{label}.pages']=pages
            witnesses[f'L{layer}.{label}']=dict(routed_rows=(pages[:64]*32).tolist(),shared_rows=(pages[64]*16).tolist(),
                                              private_row_bytes=512,shared_row_bytes=1024,capacity_reserve_WL=reserve(pages).tolist(),
                                              useful_bytes=int(pages.sum()*16384),stored_copies=1)
        save(f'layer{layer}.shared_compositions.json',all_shared)
        down=np.concatenate((g8down[layer,:64]//32,g8down[layer,64:65]//16),axis=0).astype(np.int64)
        g8.append(dict(layer=layer,**g8_release_oracle(routes[layer],down,g8groups[layer],tables)))
    np.savez_compressed(DEST/'oracle_arrays.npz',**arrays)
    save('row_witnesses.json',witnesses)
    report=dict(status='PASS',TRAIN_queries=1536,layers=[0,1,2],oracle='Mandatory chip occupation; not fullFFN replay',
                source_sha256=sha(__file__),kernel_prices=proof,relaxed_dynamic_bound_input1=dynamic_bound(tables),
                relaxed_dynamic_bound_input4=dynamic_bound(tables4),records=records,g8_shared_order_release_oracle=g8,
                prior='Existing whole-row compiler already allows unequal row counts and independent phases. Standard local search and support balancing are strong controls, not novel algorithms.',
                limitations=['No heldout or new routing collection.','All candidate maps fixed across512TRAIN routes/layer, not perquery dynamic remapping.',
                             'Price table is a necessary owner-work envelope. Global command/channel/PCIe contention, GU readiness, full A8 dependencies and finite outputFIFO are not replayed.',
                             'One-block reserve uses hottestGU planes128WL plus ceil(job_pages/4) for each Down job; zero coefficient copies and useful bytes are unchanged.',
                             'Local search4096proposals/stage is finite and not an optimum proof. Shared composition is exact only within equal-four-chip learned groups.',
                             'Dynamic lower-bound witness changes support perroute and cannot be executed by a fixed onecopy placement.'])
    save('report.json',report)
    lines=['# Down work/headroom oracle', '', 'This is TRAIN-only chip-occupation analysis, not a new fullFFN replay or a latency claim. All rows retain their full private input: routed C512, shared C1024. All native jobs use fixed0123 drain and strict500ns finite PB. Existing variable-row compiler capability is reused.', '',
           '| Layer | Fixed map | Mean critical us | P99 us | Reduction vs g4/shared16 |', '|---:|---|---:|---:|---:|']
    for r in records:lines.append(f"| {r['layer']} | {r['label']} | {r['critical_work_ns']['mean']/1000:.3f} | {r['critical_work_ns']['p99']/1000:.3f} | {100*r['mean_reduction_vs_baseline']:.3f}% |")
    lines += ['', f"The route-dependent collision-free Down relaxation has exact minmax {dynamic_bound(tables)['exact_relaxed_minmax_ns']/1000:.3f}us (input1); it assigns two chips per routed expert and four to shared, all24pages/chip. It is only a lower bound because fixed one-copy weights cannot move per query.", '',
              'Shared compaction enumerates all2925 integer four-group compositions, including unequal group row counts. Free4 support then moves equal384row shards, exposing ordinary declustering headroom. The variable stage permits complete-page row transfers and changing support, while checking the same one-block capacity. All maps are fixed across TRAIN and use exactly the same coefficient bytes.', '',
              'The g8 shared-first/shared-last side calculation uses optimistic GU service floors and per-chip release scheduling. Its difference is exposure in a relaxation, not demonstrated FFN speedup. Root may evaluate shared-GU-first scheduling separately.', '',
              'Every result must be compared fairly across GU layouts by giving each the same sharedDown/variable-row/support compiler menu and fitting budget. No sharding, local-search or mapping novelty is claimed. Native replay is required before treating any work reduction as a fullFFN improvement.']
    (DEST/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status='PASS',records=records,dynamic=dynamic_bound(tables)),indent=2))

if __name__=='__main__':main()
