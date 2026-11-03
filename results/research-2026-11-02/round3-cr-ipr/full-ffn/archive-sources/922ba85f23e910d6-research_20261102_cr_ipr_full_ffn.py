#!/usr/bin/env python3
"""Conditional finite-PB CR/IPR full-FFN event replay; TRAIN only.

Public API: Graph, run, audit, build_graph, main. Model service times are not
measured native hardware performance. Immutable source/input receipts precede
all execution. Existing baselines and pipeline artifacts are never modified.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import ctypes
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/research-2026-11-02/round3-cr-ipr'
OUT=BASE/'full-ffn'
TRAIN=ROOT/'results/research-2026-11-02/round2/w8-format-training/granite4tiny/decode.routes.npz'
PIPE=ROOT/'scripts/research_20261102_cr_ipr_pipeline.py'
CPP=Path(__file__).with_suffix('.cpp')
PIPE_SHA='ad35d08e17b4458cb5a0f54732aafce338d0526f5888fb02fcfdcc507ad99905'
PAGE=16384

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def read(p):return json.loads(Path(p).read_text())
def utc():return datetime.now(timezone.utc).isoformat()
def stats(x):
    a=np.asarray(x);return dict(mean=float(a.mean()),p99=float(np.percentile(a,99)),minimum=float(a.min()),maximum=float(a.max()))

def load_pipeline():
    assert sha(PIPE)==PIPE_SHA, 'Pipeline changed: preserve inputs and create a new explicit freeze.'
    import sys
    spec=importlib.util.spec_from_file_location('cr_ipr_pipeline_frozen',PIPE)
    p=importlib.util.module_from_spec(spec);sys.modules[spec.name]=p;spec.loader.exec_module(p)
    return p


@dataclass(frozen=True)
class Service:
    input_copies:int=1
    down_first:int=0
    activation_ns:int=0
    command_ns:int=250
    submit_ns:int=2000
    combine_ns:int=0


class Graph:
    def __init__(self,service=Service()):
        self.service=service;self.nodes=[];self.deps=[];self.labels=[]
        self.jobs=[];self.outputs=[];self.pipeline_totals=Counter()
    def add(self,r,d,deps=(),phase=0,expert=-1,chip=-1,job=-1,acquire=0,release=0,kind='',nbytes=0):
        i=len(self.nodes);deps=list(deps)
        assert all(0<=j<i for j in deps) and d>=0
        self.nodes.append((r,int(d),phase,expert,chip,job,acquire,release))
        self.deps.append(deps);self.labels.append(dict(kind=kind,bytes=int(nbytes)))
        return i
    def matrix(self,chip,phase,columns,recipe,submit,upload):
        s=self.service;j=len(self.jobs)
        # Recipe: duration and row-complete milestones(owner-rank,t,bytes).
        priority=min(x[0] for x in recipe['milestones'])
        cmd=self.add(1,s.command_ns,[submit],phase,priority,chip,j,1,kind='command')
        nb=columns*s.input_copies
        inp=self.add(3+chip%8,(nb+1)//2,[cmd,upload],phase,priority,chip,j,kind='input_ONFI',nbytes=nb)
        comp=self.add(11+chip,recipe['duration'],[inp],phase,priority,chip,j,kind='array_ECC')
        channel=[];owner_results=defaultdict(list)
        for owner,t,nbytes in recipe['milestones']:
            produce=self.add(-1,t,[inp],phase,owner,chip,j,kind='row_produced',nbytes=nbytes)
            onfi=self.add(3+chip%8,(nbytes+1)//2,[produce],phase,owner,chip,j,kind='output_ONFI',nbytes=nbytes)
            pcie=self.add(2,(nbytes+7)//8,[onfi],phase,owner,kind='output_PCIe',nbytes=nbytes)
            channel.append(onfi);owner_results[owner].append(pcie)
            self.outputs.append((chip,produce,onfi,nbytes))
        release=self.add(-1,0,[comp,*channel],phase,priority,chip,j,release=1,kind='owner_release')
        self.jobs.append(dict(chip=chip,phase=phase,columns=columns,command=cmd,input=inp,
                              compute=comp,release=release,output_ONFI=channel,
                              recipe_signature=recipe['signature'],pipeline=recipe['summary']))
        self.pipeline_totals.update(recipe['counts'])
        return owner_results
    def arrays(self):
        n=np.ascontiguousarray(self.nodes,dtype=np.int64)
        offsets=np.r_[0,np.cumsum([len(d) for d in self.deps])].astype(np.int32)
        deps=np.fromiter((v for d in self.deps for v in d),dtype=np.int32)
        return n,offsets,np.ascontiguousarray(deps)
    def signature(self):
        n,o,d=self.arrays()
        h=hashlib.sha256(n.tobytes()+o.tobytes()+d.tobytes()).hexdigest()
        return h,(n,o,d)


class Executor:
    def __init__(self,out):
        binary=out/'build'/('graph-'+sha(CPP)[:16]+'.so');binary.parent.mkdir(exist_ok=True)
        command=['c++','-std=c++17','-O3','-shared','-fPIC',str(CPP),'-o',str(binary)]
        if not binary.exists():subprocess.run(command,check=True)
        self.lib=ctypes.CDLL(str(binary));self.call=self.lib.run_graph
        a64=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS')
        a32=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')
        self.call.argtypes=[ctypes.c_int,a64,a32,a32,ctypes.c_int,a64];self.call.restype=ctypes.c_int64
        self.receipt=dict(source=str(CPP),source_sha256=sha(CPP),binary=str(binary),binary_sha256=sha(binary),command=command)
    def run(self,g,arrays=None):
        n,o,d=g.arrays() if arrays is None else arrays
        trace=np.zeros((len(n),2),dtype=np.int64)
        cost=int(self.call(len(n),n,o,d,g.service.down_first,trace))
        return dict(cost_ns=cost,trace=trace)


def run(graph,executor):return executor.run(graph)


def audit(g,result,full=True):
    trace=result['trace'];nodes=g.nodes
    assert trace.shape==(len(nodes),2)
    if full:
        resources=defaultdict(list)
        for i,(node,dep) in enumerate(zip(nodes,g.deps)):
            r,d,phase,e,c,j,a,release=node;start,end=map(int,trace[i])
            assert end-start==d and all(trace[q,1]<=start for q in dep)
            if r>=0 and d:resources[r].append((start,end))
        for intervals in resources.values():
            v=sorted(intervals);assert all(a[1]<=b[0] for a,b in zip(v,v[1:]))
        owners=defaultdict(list)
        for j,job in enumerate(g.jobs):
            cmd=job['command'];release=job['release'];inp=job['input'];comp=job['compute']
            lo,hi=int(trace[cmd,0]),int(trace[release,1]);owners[job['chip']].append((lo,hi))
            assert trace[comp,0]==trace[inp,1]
            assert all(trace[q,1]<=hi for q in job['output_ONFI']) and trace[comp,1]<=hi
            for i,node in enumerate(nodes):
                if node[5]==j:assert lo<=trace[i,0]<=trace[i,1]<=hi
        for intervals in owners.values():
            v=sorted(intervals);assert all(a[1]<=b[0] for a,b in zip(v,v[1:]))
    # Exact continuous channel drain, rather than assuming a page disappears
    # instantaneously on either input or channel start.
    events=defaultdict(lambda:defaultdict(lambda:[0,0]))
    for chip,produce,onfi,nbytes in g.outputs:
        pt=int(trace[produce,1]);a,b=map(int,trace[onfi])
        assert b-a==(nbytes+1)//2 and a>=pt and nbytes%2==0
        events[chip][pt][0]+=nbytes
        events[chip][a][1]+=2;events[chip][b][1]-=2
    peaks=np.zeros(16,dtype=np.int64)
    for chip,ee in events.items():
        previous=rate=backlog=0
        for t,(arrivals,delta_rate) in sorted(ee.items()):
            backlog-=rate*(t-previous);assert backlog>=0
            backlog+=arrivals;peaks[chip]=max(peaks[chip],backlog)
            rate+=delta_rate;assert rate>=0
            previous=t
        assert backlog==rate==0
    counts=Counter();work=Counter()
    for n,l in zip(g.nodes,g.labels):
        counts[l['kind']]+=1;counts[l['kind']+'_bytes']+=l['bytes']
        if n[0]>=0:work[l['kind']+'_busy_ns']+=n[1]
    return dict(nodes=len(nodes),jobs=len(g.jobs),peak_output_fifo_bytes_per_chip=peaks.tolist(),
                peak_output_fifo_bytes=int(peaks.max()),
                valid_FIFO_capacities={str(k):bool(peaks.max()<=k) for k in [64,128,256,2048]},
                counts=dict(counts),resource_work_ns=dict(work),pipeline_counts=dict(g.pipeline_totals),
                all_dependencies_resources_chip_owners_checked=full)


def dump_trace(path,g,result,summary):
    dump(path,dict(service=asdict(g.service),cost_ns=result['cost_ns'],audit=summary,
                   nodes=[dict(index=i,resource=n[0],duration=n[1],phase=n[2],expert_rank=n[3],
                               chip=n[4],job=n[5],acquire=bool(n[6]),release=bool(n[7]),
                               deps=g.deps[i],start_ns=int(result['trace'][i,0]),
                               end_ns=int(result['trace'][i,1]),**g.labels[i])
                          for i,n in enumerate(g.nodes)],jobs=g.jobs))


class Recipes:
    def __init__(self,pipeline):self.p=pipeline;self.cache={};self.failures=[]
    def make(self,waves,profile,columns,owner_ranks,validate=True):
        settings=dict(cache_copy_ns=100 if profile=='strict100' else 500,
                      source_policy='permutation',issue_policy='strict_no_interval' if profile=='strict100' else 'cold_reset',
                      row_bytes=columns,validate_row_stream=validate,
                      unselected_policy='drop_before_ecc_conditional')
        signature=(profile,columns,validate,tuple((tuple((owner_ranks[x['owner']],x['page_index'],x['plane'],x.get('bytes',PAGE)) for x in w['pages']),
                                               tuple(w['source_order']),tuple(w.get('mask_active_planes',[]))) for w in waves))
        if signature in self.cache:return self.cache[signature]
        result=self.p.simulate(waves,settings)
        if not result['feasible']:
            if len(self.failures)<20:self.failures.append(dict(settings=settings,summary={k:result[k] for k in ['functional_ok','timing_feasible','makespan_ns','violations','stream_validation']},waves=waves))
            self.cache[signature]=None;return None
        milestones=[];rows=Counter()
        for x in result['page_completions']:
            owner=x['owner'];before=rows[owner]//columns;rows[owner]+=x['bytes'];after=rows[owner]//columns
            if after>before:milestones.append((owner_ranks[owner],int(x['end_ns']),int((after-before)*4)))
        counts={k:result[k] for k in ['waves','pages','bytes','physical_read_pages','physical_read_bytes','ecc_pages','ecc_bytes','cold_waves','cr_waves']}
        recipe=dict(duration=int(result['makespan_ns']),milestones=milestones,
                    signature=hashlib.sha256(repr(signature).encode()).hexdigest(),counts=counts,
                    summary=dict(profile=profile,settings=settings,**counts,
                                 finite_PB_feasible=True,complete_row_stream=validate,
                                 conditional_capabilities=result['conditional_capabilities']))
        self.cache[signature]=recipe;return recipe
    def private(self,rows,columns,owner,profile):
        pages=(rows*columns+PAGE-1)//PAGE
        waves=[]
        for start in range(0,pages,4):
            pp=[dict(page_id=f'v.p{j}',owner='v',page_index=j,plane=j%4,block=0,wl=j//4,bytes=PAGE) for j in range(start,min(start+4,pages))]
            waves.append(dict(pages=pp,source_order=[x['plane'] for x in pp]))
        # Full vector remains columns wide. The native private matrix resets
        # final coefficient padding; emitted fringe rows are charged in output.
        rec=self.make(waves,profile,columns,{'v':0},validate=False)
        if rec is None:return None
        return dict(rec,milestones=[(owner,t,n) for _,t,n in rec['milestones']])


def packed_waves(pipeline,route,order,keep_hot):
    inverse=np.argsort(order);by_wl=defaultdict(list)
    for e in route:
        for j in range(6):
            page=int(inverse[e])*6+j
            by_wl[page//4].append(dict(page_id=f'e{e}.p{j}',owner=f'e{e}',page_index=j,plane=page%4,block=0,wl=page//4,bytes=PAGE))
    for j in range(12):
        page=384+j
        by_wl[page//4].append(dict(page_id=f's.p{j}',owner='s',page_index=j,plane=page%4,block=0,wl=page//4,bytes=PAGE))
    waves=[dict(pages=sorted(v,key=lambda p:p['plane']),source_order=sorted(p['plane'] for p in v)) for _,v in sorted(by_wl.items())]
    return pipeline.with_keep_hot(waves) if keep_hot else waves


def rr_greedy_waves(route):
    # Pair opposite native phases when possible, then preserve every selected
    # expert's complete six-page order. This is an actual row-safe construction,
    # not the max-plane page-work bound used as a fabricated schedule.
    a=sorted(e for e in route if e%2==0);b=sorted(e for e in route if e%2)
    order=[]
    while a and b:order.extend((a.pop(0),b.pop(0)))
    order+=a+b;stream=[]
    for e in order:
        for j in range(6):
            page=e*6+j;stream.append(dict(page_id=f'e{e}.p{j}',owner=f'e{e}',page_index=j,plane=page%4,block=0,wl=page//4,bytes=PAGE))
    for j in range(12):stream.append(dict(page_id=f's.p{j}',owner='s',page_index=j,plane=j%4,block=0,wl=96+j//4,bytes=PAGE))
    waves=[];pending=[];planes=set()
    for page in stream:
        if page['plane'] in planes:
            waves.append(dict(pages=pending,source_order=[p['plane'] for p in pending]));pending=[];planes=set()
        pending.append(page);planes.add(page['plane'])
    if pending:waves.append(dict(pages=pending,source_order=[p['plane'] for p in pending]))
    return waves


def down_rows(chosen,mode,strong):
    if mode=='old_strong':return strong[chosen,1].copy()
    rows=np.zeros((7,16),dtype=np.int32)
    for rank,e in enumerate(chosen):
        if e==64 or mode=='g16_equal':rows[rank]=96
        else:rows[rank,(np.arange(12)+e)%16]=128
    assert np.all(rows.sum(axis=1)==1536)
    return rows


def build_graph(route,layer,gu,down,profile,service,recipes,orders,strong):
    route=sorted(map(int,route));chosen=route+[64];ranks={f'e{e}':i for i,e in enumerate(route)};ranks['s']=6
    p=recipes.p;g=Graph(service);submit=g.add(0,service.submit_ns,kind='GU_submit')
    upload=g.add(2,192,[submit],kind='GU_input_PCIe',nbytes=1536)
    outputs=defaultdict(list)
    if gu in ['separate_GU','legacy_strong_whole_rows']:
        for rank,e in enumerate(chosen):
            for chip in range(16):
                rows=int(strong[e,0,chip]) if gu=='legacy_strong_whole_rows' else (128 if e==64 else 64)
                if not rows:continue
                recipe=recipes.private(rows,1536,rank,profile)
                if recipe is None:return None
                oo=g.matrix(chip,0,1536,recipe,submit,upload)
                for owner,ids in oo.items():outputs[owner]+=ids
    else:
        if gu=='candidate':waves=p.build_g4_candidate(route)
        elif gu.startswith('sameWL_'):
            order=orders[layer] if 'matching' in gu else np.arange(64)
            waves=packed_waves(p,route,order,'keep_hot' in gu)
        elif gu=='IPR_standard_RR_row_safe':waves=rr_greedy_waves(route)
        elif gu=='IPR_phase_aligned_balanced_shared':waves=p.build_balanced_ipr(route)
        else:raise ValueError(gu)
        recipe=recipes.make(waves,profile,1536,ranks,validate=True)
        if recipe is None:return None
        for chip in range(16):
            oo=g.matrix(chip,0,1536,recipe,submit,upload)
            for owner,ids in oo.items():outputs[owner]+=ids
    downmap=down_rows(chosen,down,strong)
    complete=[]
    for rank,e in enumerate(chosen):
        assert outputs[rank]
        ready=g.add(-1,0,outputs[rank],expert=rank,kind='full_GU_at_host')
        act=g.add(0,service.activation_ns,[ready],phase=1,expert=rank,kind='full_vector_activation_A8')
        ds=g.add(0,service.submit_ns,[act],phase=1,expert=rank,kind='Down_submit')
        columns=1024 if e==64 else 512
        up=g.add(2,(columns+7)//8,[ds],phase=1,expert=rank,kind='Down_input_PCIe',nbytes=columns)
        for chip,rows in enumerate(downmap[rank]):
            if not rows:continue
            recipe=recipes.private(int(rows),columns,rank,profile)
            if recipe is None:return None
            oo=g.matrix(chip,1,columns,recipe,ds,up)
            complete+=oo[rank]
    g.add(0,service.combine_ns,complete,phase=1,kind='weighted_combine')
    return g


def load_strong():
    api=read(BASE/'baselines/provenance_and_old_plan_api.json')['old_strong_plan_api']
    rows=[];inputs=[]
    for layer in range(3):
        record=next(r for r in api['latest_three_layer_arrays'] if r['model']=='granite4tiny' and r['moe_layer']==layer and r['input_channel_copies']==1 and r['family']=='independent_paired')
        path=record['arrays'];expected=record['sha256']
        if layer==0:path=api['G4_L0_strong_CP_incumbent']['path'];expected=api['G4_L0_strong_CP_incumbent']['sha256']
        assert sha(path)==expected
        with np.load(path) as z:rows.append(z['rows'].copy())
        inputs.append(dict(layer=layer,path=path,sha256=expected))
    return np.asarray(rows),inputs


def freeze(out):
    out.mkdir(parents=True,exist_ok=True)
    assert not (out/'execution_plan.frozen.json').exists()
    p=load_pipeline();strong,strong_inputs=load_strong()
    input_paths=[Path(__file__),CPP,PIPE,TRAIN,BASE/'experiment_plan.frozen.json',BASE/'baselines/protocol.frozen.json',BASE/'baselines/frozen_TRAIN_packed_expert_order.npz']
    plan=dict(frozen_utc=utc(),status='frozen_before_full_FFN_execution',
              inputs_sha256={str(p):sha(p) for p in input_paths},strong_inputs=strong_inputs,
              data='Previously observed W8 TRAIN only:32prompts x16steps x first3MoE layers. No heldout data opened.',
              primary_profile='strict100',secondary_profile='cold_reset500',
              selection_disclosure='100ns strict and500ns cold-reset profiles chosen after completed pipeline sensitivity showed exact320ns pass/321ns fail; this is supported conditional parameter analysis, not a prospectively unseen-hardware or heldout selection.',
              primary_GU=['candidate','sameWL_matching_keep_hot','sameWL_matching_sparse','IPR_standard_RR_row_safe','IPR_phase_aligned_balanced_shared','separate_GU','legacy_strong_whole_rows'],
              secondary_GU=['candidate','sameWL_matching_keep_hot'],
              generic_control='exact alias of candidate layout/execution under same capabilities; zero difference, no independent repeated samples',
              main_down=['g16_equal','g12_cyclic'],old_strong_down='smoke and explicit supplementary comparisons on same layer; no refit',
              copies=[1,4],priority=['GF','DF'],activation_ns=[0,2000],
              activation_sensitivity='Main512 uses0. 2us sensitivity is smoke and main paired candidate/matchingkeep-hot if computationally practical; any reduced scope explicitly reported.',
              output_FIFO_capacities_bytes=[64,128,256,2048],
              source_order='programmable conditional shared by all main controls; fixed order negative cases retained',
              resources=dict(host_submit_ns=2000,command_ns=250,pcie_Bps=8000000000,channels=8,channel_Bps=2000000000,chip_to_channel='chip%8; two chips per channel',one_GEMV_owner_until='array/ECC compute end AND last ONFI result drained',timers='resource-free causal page-completion offsets'),
              pipeline=dict(cache_slots=2,overwrite_at='evaluation_start',copy_overlaps_recycle=True,copy_scope='per_plane',skip_WL_CR='user premise',cold_reset='cold28us explicitly paid, no paced retention rescue',keep_hot='all4plane array reads; inactive discarded beforeECC, conditional feature; physical overfetch paid'),
              numeric_limits='Replay performs no GEMV arithmetic. Complete-row and fragment correctness delegated to independent numeric evidence. Privatejob final padding resets; native emitted floor(pages*16384/columns) rows charged. Full input dimension/bytes never shortened.',
              host_assumptions='Activation/A8 serialized host resource at declared0/2us, scales/address metadata resident host and no per-row measured cost, combine0ns; no original-GGUF-quality or measured-hardware claim.',
              FIFO='Posthoc exact perchip produced full-row bytes minus continuous ONFI drained bytes. Above budget means invalid for that budget; no silent backpressure or large FIFO.',
              scheduler='Fixed deterministic greedy ready queues, GF/DF phase priority; not global optimal schedule.')
    dump(out/'execution_plan.frozen.json',plan)
    return plan


def negative_cases(out,p):
    cases={}
    for name,cfg in [
        ('candidate_strict500',dict(cache_copy_ns=500)),
        ('candidate_fixed_source100',dict(cache_copy_ns=100,source_policy='fixed')),
        ('candidate_strict0_ideal_copy',dict(cache_copy_ns=0)),
        ('candidate_strict100',dict(cache_copy_ns=100)),
    ]:
        r=p.simulate(p.build_g4_candidate(),cfg)
        cases[name]={k:r[k] for k in ['functional_ok','timing_feasible','feasible','makespan_ns','hypothetical_makespan_ns','violations','stream_validation','config']}
        dump(out/(name+'.pipeline.json'),r)
    assert cases['candidate_strict500']['makespan_ns'] is None
    assert cases['candidate_fixed_source100']['makespan_ns'] is None
    assert cases['candidate_strict0_ideal_copy']['feasible'] and cases['candidate_strict100']['feasible']
    dump(out/'negative_and_copy0_cases.json',cases)
    return cases


def campaign(out,phase,limit=512):
    plan=read(out/'execution_plan.frozen.json')
    for p,h in plan['inputs_sha256'].items():assert sha(p)==h,p
    p=load_pipeline();recipes=Recipes(p);executor=Executor(out)
    strong,_=load_strong()
    with np.load(TRAIN) as z:routes=z['routes'].transpose(2,0,1,3).reshape(40,512,6)[:3]
    with np.load(BASE/'baselines/frozen_TRAIN_packed_expert_order.npz') as z:orders=z['slot_to_expert']
    dest=out/phase;dest.mkdir(exist_ok=True)
    assert not (dest/'report.json').exists()
    dump(dest/'run.frozen.json',dict(utc=utc(),source_sha256=sha(__file__),plan_sha256=sha(out/'execution_plan.frozen.json'),
                                   phase=phase,queries_per_layer=limit,layers=[0,1,2],executor=executor.receipt))
    negative=negative_cases(dest,p)
    cache={};records=[];arrays={};started=time.monotonic();total_replayed=0
    configurations=[]
    for profile in ['strict100','cold_reset500']:
        gu_list=plan['primary_GU'] if profile=='strict100' else plan['secondary_GU']
        for down in ['g16_equal','g12_cyclic']:
            for copies in [1,4]:
                for df in [0,1]:
                    for gu in gu_list:configurations.append((profile,down,copies,df,0,gu))
    # Extra matched Down control and host activation sensitivity are finite,
    # predeclared pair comparisons; all requested query indices retained.
    for copies in [1,4]:
        for df in [0,1]:
            for gu in ['candidate','sameWL_matching_keep_hot']:
                configurations.append(('strict100','old_strong',copies,df,0,gu))
                configurations.append(('strict100','g12_cyclic',copies,df,2000,gu))
    for ci,(profile,down,copies,df,activation,gu) in enumerate(configurations):
        name=f'{profile}.{down}.input{copies}.{("DF" if df else "GF")}.a{activation}.{gu}'
        cost=np.full((3,limit),-1,dtype=np.int64);fifo=cost.copy();node_counts=cost.copy()
        aggregate=Counter();failures=Counter();audited=0
        for layer in range(3):
            for q,route in enumerate(routes[layer,:limit]):
                service=Service(copies,df,activation)
                graph=build_graph(route,layer,gu,down,profile,service,recipes,orders,strong[layer])
                if graph is None:failures['pipeline_infeasible']+=1;continue
                signature,graph_arrays=graph.signature();signature=(signature,df)
                if signature in cache:
                    value,report=cache[signature]
                else:
                    result=executor.run(graph,graph_arrays)
                    checkfull=(q<2 or q==limit-1)
                    report=audit(graph,result,full=checkfull)
                    value=result['cost_ns'];cache[signature]=(value,report)
                    total_replayed+=1
                    if len(records)<1 and layer==0 and q==0:
                        dump_trace(dest/('small_full_trace.'+gu+'.json'),graph,result,report)
                    if phase=='smoke' and q==0 and layer==0:
                        dump_trace(dest/('trace.'+name+'.json'),graph,result,report)
                cost[layer,q]=value;fifo[layer,q]=report['peak_output_fifo_bytes'];node_counts[layer,q]=report['nodes']
                for k,v in report['counts'].items():aggregate[k]+=v
                for k,v in report['resource_work_ns'].items():aggregate[k]+=v
                for k,v in report['pipeline_counts'].items():aggregate['pipeline_'+k]+=v
                audited+=report['all_dependencies_resources_chip_owners_checked']
        good=cost>=0
        rec=dict(name=name,profile=profile,down=down,copies=copies,priority='DF' if df else 'GF',activation_ns=activation,GU=gu,
                 requested_queries=int(cost.size),completed_queries=int(good.sum()),failures=dict(failures),
                 cost_ns=stats(cost[good]) if good.any() else None,
                 per_token_three_layers_ns=stats(cost.sum(axis=0)) if good.all() else None,
                 FIFO_peak_bytes=stats(fifo[good]) if good.any() else None,
                 valid_queries_by_FIFO={str(k):int(np.count_nonzero(good&(fifo<=k))) for k in [64,128,256,2048]},
                 event_counts_and_work_per_query={k:v/int(good.sum()) for k,v in aggregate.items()} if good.any() else {},
                 fully_audited_queries_or_cached_equivalent=int(audited),nodes=stats(node_counts[good]) if good.any() else None)
        records.append(rec);arrays[name+'.cost']=cost;arrays[name+'.fifo']=fifo
        if ci%4==0 or ci==len(configurations)-1:
            progress=dict(phase=phase,finished_configurations=ci+1,total_configurations=len(configurations),
                          actual_unique_graph_replays=total_replayed,cached_graphs=len(cache),elapsed_seconds=time.monotonic()-started,last=name)
            dump(dest/'progress.json',progress);print(json.dumps(progress),flush=True)
    comparisons=[]
    for r in records:
        if r['GU']!='candidate':continue
        for baseline in ['sameWL_matching_keep_hot','sameWL_matching_sparse','IPR_standard_RR_row_safe','IPR_phase_aligned_balanced_shared','separate_GU','legacy_strong_whole_rows']:
            match=next((x for x in records if all(x[k]==r[k] for k in ['profile','down','copies','priority','activation_ns']) and x['GU']==baseline),None)
            if match is None:continue
            a=arrays[r['name']+'.cost'];b=arrays[match['name']+'.cost'];valid=(a>=0)&(b>=0)
            if not valid.any():continue
            av,bv=a[valid],b[valid];gain=1-float(av.mean()/bv.mean())
            compare=dict(candidate=r['name'],baseline=match['name'],paired_queries=int(valid.sum()),
                         mean_gain_pct=100*gain,p99_gain_pct=100*(1-float(np.percentile(av,99)/np.percentile(bv,99))),
                         paired_slower_queries=int(np.count_nonzero(av>bv)),paired_slower_fraction=float(np.mean(av>bv)),
                         same_queries_equal=int(np.count_nonzero(av==bv)),
                         FIFO_joint_valid={str(k):int(np.count_nonzero(valid&(arrays[r['name']+'.fifo']<=k)&(arrays[match['name']+'.fifo']<=k))) for k in [64,128,256,2048]})
            comparisons.append(compare)
    np.savez_compressed(dest/'service_arrays.npz',**arrays)
    dump(dest/'pipeline_failures.json',recipes.failures)
    report=dict(status='complete_conditional_TRAIN_full_FFN_service_model',phase=phase,queries_per_layer=limit,layers=[0,1,2],
                plan_sha256=sha(out/'execution_plan.frozen.json'),records=records,comparisons=comparisons,
                generic_compiler_control='candidate exact same layout and graph; all costs, FIFO counts and bytes identical; gain0%',
                actual_unique_graph_replays=total_replayed,elapsed_seconds=time.monotonic()-started,
                conditional_profiles=plan['pipeline'],host_assumptions=plan['host_assumptions'],
                limitations=['No heldout data or inference run; previously observed TRAIN only.','No measured hardware latency or native ABI claim.','Above specified output FIFO budget is invalid for that budget; no backpressure rescue.','Programmable source order and drop-before-ECC common-input GEMV continuation remain conditional.','Strict500/fixed-source failures retained separately; no conditional pacing used.'],negative_cases=negative)
    dump(dest/'report.json',report)
    lines=['# Conditional full-FFN TRAIN service replay','',f'First3MoE layers × {limit}decode inputs/layer. Fixed source/input hashes precede execution. These are explicit conditional service models, not measured hardware latency.','',
           '| Profile / Down / copies / priority / activation | Mean gain vs exact matching keep-hot | P99 gain | Slower queries | Joint valid FIFO64/128/256/2048 |','|---|---:|---:|---:|---|']
    for c in comparisons:
        if 'sameWL_matching_keep_hot' not in c['baseline']:continue
        name=c['candidate'].replace('.candidate','');fv='/'.join(str(c['FIFO_joint_valid'][str(k)]) for k in [64,128,256,2048])
        lines.append(f"| {name} | {c['mean_gain_pct']:.4f}% | {c['p99_gain_pct']:.4f}% | {c['paired_slower_queries']}/{c['paired_queries']} | {fv} |")
    lines+=['','The generic sparse compiler uses the exact candidate layout and graph:0%gain. Strict100ns copy is selected after prior pipeline sensitivity; it is a declared unmeasured conditional timing value. Cold-reset500ns explicitly pays28us cold reads. Strict500ns and fixed-source failures are preserved.','',
            'All private Down jobs wait for full expert GU at host, whole-vector activation/A8 and a full private input upload. Chip owner remains held until compute end and final ONFI output. FIFO validity is posthoc by64/128/256/2048B; costs with an insufficient FIFO are not feasible bounded-FIFO schedules.','',
            'Event counts and attributed work are in report.json; all paired arrays in service_arrays.npz. No new heldout data was opened.']
    (dest/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status='complete',phase=phase,configurations=len(records),actual_unique_graph_replays=total_replayed,elapsed_seconds=report['elapsed_seconds'])),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['freeze','smoke','run']);parser.add_argument('--output-dir',type=Path,default=OUT);args=parser.parse_args();out=args.output_dir.resolve()
    if args.action=='freeze':freeze(out)
    elif args.action=='smoke':campaign(out,'smoke',8)
    else:campaign(out,'TRAIN512',512)

if __name__=='__main__':main()
