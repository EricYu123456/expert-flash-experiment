#!/usr/bin/env python3
"""Independent saved-trace/resource/resident-binding audit plus cap2 micro replay.

Reuses the earlier independent scalar_execute/semantic_audit, not target audit.
Target code is imported only to construct12synthetic cap2 fallback graphs.
"""
from collections import Counter,defaultdict
import bisect
import copy
import ctypes
from dataclasses import asdict
import datetime
import hashlib
import heapq
import importlib.util
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/research-2026-11-02/round3-cr-ipr'
TARGET=ROOT/'scripts/research_20261102_cr_g2_availability_ffn.py'
DATA=BASE/'g2-availability-full-ffn'
OUT=BASE/'g2-availability-full-ffn-independent'
CHECKER=ROOT/'scripts/research_20261102_cr_ipr_full_ffn_audit.py'
KERNEL=BASE/'g2-availability-headroom/capacity-kernel/kernel_builder.py'
TRAIN=ROOT/'results/research-2026-11-02/round2/w8-format-training/granite4tiny/decode.routes.npz'


def need(ok,msg):
    if not ok:raise RuntimeError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m


def expected_residents(adj):
    """Independent closed coordinate formula, no capacity-kernel import."""
    need(adj.shape==(64,6) and np.all(adj.sum(axis=0)==42),'frozen support columns')
    expected={}
    for g in range(6):
        experts=np.flatnonzero(adj[:,g]).tolist()
        for rank,e in enumerate(experts):
            b=0 if rank<10 else 1;wl=128+rank*6 if rank<10 else (rank-10)*6
            for local in range(2):
                chip=2*g+local;j=dict(expert=e,copy=0,resident_copy=g,resident_job_id=f'e{e}.g{g}.c{chip}',
                    chip=chip,rows=768,columns=512,block=b,wl_start=wl,pages=24,row_start=local*768,reserved_WL=6)
                expected[j['resident_job_id']]=j
    for chip in range(12,16):
        j=dict(expert=64,copy=0,resident_copy=0,resident_job_id=f'shared.c{chip}',chip=chip,rows=384,columns=1024,
               block=0,wl_start=128,pages=24,row_start=(chip-12)*384,reserved_WL=6)
        expected[j['resident_job_id']]=j
    need(len(expected)==508,'all508resident chip jobs')
    own=set();logical=defaultdict(list)
    for j in expected.values():
        need(j['wl_start']+6<=192,'native job crosses block')
        for page in range(24):
            key=(j['chip'],j['block'],page%4,j['wl_start']+page//4)
            need(key not in own,'resident Down collision');own.add(key)
        logical[j['expert'],j['resident_copy']].append((j['row_start'],j['row_start']+j['rows']))
    for intervals in logical.values():
        cursor=0
        for start,end in sorted(intervals):need(start==cursor,'resident row gap/overlap');cursor=end
        need(cursor==1536,'resident fullcopy row coverage')
    need(len(own)==12192,'resident Down pages')
    return expected


def resident_recipe_signature(p):
    """Independently bind the sealed recipe digest to absolute page extents."""
    waves=[]
    for wave in range(6):
        pages=[dict(page_id=f'v.p{page}',owner='v',page_index=page,plane=page%4,block=p['block'],
                    wl=p['wl_start']+page//4,bytes=16384) for page in range(4*wave,4*wave+4)]
        waves.append(dict(pages=pages,source_order=[0,1,2,3]))
    settings=dict(cache_copy_ns=500,source_policy='fixed',issue_policy='strict_no_interval',row_bytes=p['columns'],
                  validate_row_stream=True,unselected_policy='drop_before_ecc_conditional')
    payload=dict(waves=waves,settings=settings,ranks={'v':0},pool=None)
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest(),settings


def binding_audit(record,route,adj,expected,capacity=1):
    nodes=record['nodes'];chosen=sorted(map(int,route))+[64]
    groups={};byexpert=defaultdict(list);seen=set();rowcount=Counter()
    for n in nodes:
        if n['kind']=='output_ONFI':rowcount[n['phase'],n['expert_rank']]+=n['bytes']//4
    for rank in range(7):
        need(rowcount[0,rank]==(2048 if rank==6 else 1024) and rowcount[1,rank]==1536,'exact fullrow coverage')
    for job in record['jobs']:
        if job['phase']==0:continue
        p=job['physical_down_job'];rid=p['resident_job_id']
        need(rid in expected and p==expected[rid],'selected resident_job_id or physical metadata differs from independent frozen layout')
        need(rid not in seen,'resident job selected twice');seen.add(rid)
        rank=nodes[job['command']]['expert_rank'];e=chosen[rank]
        need(e==p['expert'],'rank/expert binding')
        need(job['chip']==p['chip'] and job['columns']==p['columns'],'actual physical chip/operand dimension binding')
        emitted=sum(nodes[i]['bytes'] for i in job['output_ONFI'])//4
        need(emitted==p['rows'] and job['pipeline']['bytes']==p['rows']*p['columns'],'native row/byte binding')
        need(nodes[job['compute']]['duration']==89940,'native sixwave kernel duration')
        pipe=job['pipeline']
        digest,settings=resident_recipe_signature(p)
        need(job['recipe_signature']==digest,'actual native recipe digest differs from certified absolute block/WL pages')
        need(pipe['settings']==settings,'native private operand/PB settings')
        need(pipe['finite_PB_feasible'] and pipe['source_policy']=='fixed' and pipe['policy']=='private_strict','fixed-source finite PB Down contract')
        need(pipe['pages']==pipe['physical_read_pages']==pipe['ecc_pages']==24 and pipe['waves']==6 and pipe['cold_waves']==1 and pipe['cr_waves']==5,'native cold/CR page accounting')
        byexpert[e].append(p)
        if e<64:
            g=p['resident_copy'];need(adj[e,g] and p['chip']//2==g,'expert/group/physical chip incidence')
            need(e not in groups or groups[e]==g,'expert scattered across multiple selected replica groups');groups[e]=g
    need(set(byexpert)==set(chosen) and len(seen)==16,'allsix experts andshared exactly16Down chip jobs')
    for e,js in byexpert.items():
        need(len(js)==(4 if e==64 else 2),'full expert band count')
        cursor=0
        for j in sorted(js,key=lambda x:x['row_start']):need(j['row_start']==cursor,'selected output-row scatter gap/overlap');cursor+=j['rows']
        need(cursor==1536,'selected full expert output coverage')
    loads=[sum(g==x for g in groups.values()) for x in range(6)]
    need(sum(loads)==6 and max(loads)<=capacity,'matching job capacity')
    need(len(record['jobs'])==32,'sixteenGU plus16Down jobs, none missing')
    need(sum(n['bytes'] for n in nodes if n['kind']=='input_ONFI')==34816*record['service']['input_copies'],'complete input ONFI conservation')
    work=Counter()
    for n in nodes:
        if n['resource']>=0:work[n['kind']+'_busy_ns']+=n['duration']
    need(dict(work)==record['audit']['resource_work_ns'],'resource busy excludes resource-free timers')
    return dict(selected_resident_jobs=16,selected_expert_groups={str(e):g for e,g in groups.items()},group_loads=loads,
                rows=18944,GU_rows=8192,Down_rows=10752,input_ONFI_bytes=34816*record['service']['input_copies'])


def critical_path(record):
    """Longest path after adding observed resource and chip-lease orders."""
    nodes=record['nodes'];pred=[set(n['deps']) for n in nodes];positive=defaultdict(list)
    for i,n in enumerate(nodes):
        if n['resource']>=0 and n['duration']>0:positive[n['resource']].append((n['end_ns'],i))
    for r in positive:positive[r].sort()
    for i,n in enumerate(nodes):
        if n['resource']<0:continue
        entries=positive[n['resource']];pos=bisect.bisect_right(entries,(n['start_ns'],len(nodes)))-1
        if pos>=0:pred[i].add(entries[pos][1])
    owners=defaultdict(list)
    for j in record['jobs']:owners[j['chip']].append(j)
    for jobs in owners.values():
        jobs.sort(key=lambda j:nodes[j['command']]['start_ns'])
        for prev,current in zip(jobs,jobs[1:]):pred[current['command']].add(prev['release'])
    children=[[] for _ in nodes];degree=[]
    for i,pp in enumerate(pred):
        degree.append(len(pp))
        for p in pp:children[p].append(i)
    ready=[i for i,d in enumerate(degree) if d==0];heapq.heapify(ready);end=[0]*len(nodes);parent=[None]*len(nodes);done=0
    while ready:
        i=heapq.heappop(ready);p=max(pred[i],key=lambda j:(end[j],j)) if pred[i] else None
        start=end[p] if p is not None else 0
        need(start==nodes[i]['start_ns'],'augmented event/resource DAG earliest start differs')
        end[i]=start+nodes[i]['duration'];parent[i]=p;done+=1
        for child in children[i]:
            degree[child]-=1
            if degree[child]==0:heapq.heappush(ready,child)
    need(done==len(nodes) and max(end)==record['cost_ns'],'critical-path conservation')
    i=max(range(len(nodes)),key=lambda j:(end[j],j));chain=[]
    while i is not None:
        n=nodes[i]
        if n['duration']>0:chain.append(dict(index=i,kind=n['kind'],resource=n['resource'],chip=n['chip'],rank=n['expert_rank'],
            start_ns=n['start_ns'],end_ns=n['end_ns'],duration_ns=n['duration']))
        i=parent[i]
    chain.reverse();need(sum(n['duration_ns'] for n in chain)==record['cost_ns'],'critical chain durations')
    last=max((n for n in nodes if n['kind']=='output_PCIe'),key=lambda n:n['end_ns'])
    onfi=nodes[last['deps'][0]];job=record['jobs'][onfi['job']]
    return dict(cost_ns=record['cost_ns'],critical_chain=chain,critical_final_resident_job=job['physical_down_job'],
                final_command_start_ns=nodes[job['command']]['start_ns'],final_command_end_ns=nodes[job['command']]['end_ns'],
                terminal_ONFI_end_ns=onfi['end_ns'],terminal_PCIe_duration_ns=last['duration'],
                terminal_PCIe_bytes=last['bytes'])


def hall_failure(adj,route):
    for subset in range(1,64):
        count=sum(all(not adj[e,g] or subset&(1<<g) for g in range(6)) for e in route)
        if count>subset.bit_count():return True
    return False


def generated_record(graph,trace,cost,checker):
    """Serialize C++ result and independently derive its inventory/FIFO flags."""
    nodes=[dict(index=i,resource=n[0],duration=n[1],phase=n[2],expert_rank=n[3],chip=n[4],job=n[5],
                acquire=bool(n[6]),release=bool(n[7]),deps=graph.deps[i],start_ns=int(trace[i,0]),end_ns=int(trace[i,1]),**graph.labels[i]) for i,n in enumerate(graph.nodes)]
    arrivals=defaultdict(list);transfers=defaultdict(list);counts=Counter();work=Counter();pipes=Counter()
    for n in nodes:
        counts[n['kind']]+=1;counts[n['kind']+'_bytes']+=n['bytes']
        if n['resource']>=0:work[n['kind']+'_busy_ns']+=n['duration']
        if n['kind']=='row_produced':arrivals[n['chip']].append((n['end_ns'],n['bytes']))
        if n['kind']=='output_ONFI':transfers[n['chip']].append((n['start_ns'],n['end_ns'],n['bytes']))
    peaks=[checker.fifo_integral(arrivals[c],transfers[c])[0] for c in range(16)]
    for j in graph.jobs:
        for key in ['waves','pages','bytes','physical_read_pages','physical_read_bytes','ecc_pages','ecc_bytes','cold_waves','cr_waves']:pipes[key]+=j['pipeline'][key]
    audit=dict(peak_output_fifo_bytes_per_chip=peaks,peak_output_fifo_bytes=max(peaks),
               valid_FIFO_capacities={str(k):max(peaks)<=k for k in [64,128,256,2048]},
               counts=dict(counts),resource_work_ns=dict(work),pipeline_counts=dict(pipes))
    return dict(service=asdict(graph.service),cost_ns=cost,audit=audit,nodes=nodes,jobs=graph.jobs)


def main():
    OUT.mkdir(parents=True,exist_ok=True);began=time.monotonic();checker=load('prior_independent_g2_checker',CHECKER)
    protocol=read(DATA/'protocol.frozen.json')
    for p,h in protocol['inputs_sha256'].items():need(sha(p)==h,'sealed input changed: '+p)
    files=sorted((DATA/'TRAIN512').glob('trace.*json'));need(len(files)==20,'expected20 saved representative traces')
    paths=[Path(__file__).resolve(),CHECKER,TARGET,KERNEL,DATA/'protocol.frozen.json',DATA/'resident_jobs.frozen.json',
           DATA/'resident_layouts.frozen.json',DATA/'support_maps.frozen.npz',DATA/'TRAIN512/report.json',DATA/'TRAIN512/service_arrays.npz',
           DATA/'TRAIN512/run.frozen.json',TRAIN,*files]
    paths+=list((DATA/'build').glob('*.so'))
    hashes={str(p):sha(p) for p in paths};write(OUT/'protocol.frozen.json',dict(status='frozen_before_independent_saved_trace_replay',inputs_sha256=hashes,
        scope='Twenty saved qindex0 traces, fullsummaryarray arithmetic only, and12 synthetic cap2 micrographs; no wholeTRAIN rerun or heldout. Target imports confined to fallback graph construction.'))
    with np.load(DATA/'support_maps.frozen.npz') as z:adj=z['selected'].copy()
    original_key=protocol['selected_input_map_key'];original_index=protocol['selected_input_map_index']
    origin=[p for p in protocol['inputs_sha256'] if p.endswith('/availability.frozen.npz') or p.endswith('/selected_adjacency.frozen.npz')]
    need(len(origin)==1,'unique frozen support source')
    with np.load(origin[0]) as z:origin_map=z[original_key] if original_index is None else z[original_key][original_index]
    need(np.array_equal(adj,origin_map),'actual frozen map differs from selected oracle source')
    expected=[expected_residents(a) for a in adj];layouts=read(DATA/'resident_layouts.frozen.json');sealed=read(DATA/'resident_jobs.frozen.json')
    for layer in range(3):
        need({j['resident_job_id']:j for j in sealed[layer]['entries']}==expected[layer],'sealed resident jobs differ from coordinate formula')
        need(layouts[layer]['supports']==[np.flatnonzero(row).tolist() for row in adj[layer]],'sealed layout support map')
        for g in range(6):
            group=layouts[layer]['groups'][g];need(group['chips']==[2*g,2*g+1] and group['resident_experts']==np.flatnonzero(adj[layer,:,g]).tolist(),'sealed physical group membership')
            for rank,j in enumerate(group['jobs']):
                p=expected[layer][f"e{j['expert']}.g{g}.c{2*g}"]
                need(j['block']==p['block'] and j['WL_start']==p['wl_start'] and j['rows']==768 and j['columns']==512 and j['rank']==rank,'certified group job extent')
    with np.load(TRAIN) as z:routes=z['routes'].transpose(2,0,1,3).reshape(40,512,6)[:3]
    with np.load(DATA/'TRAIN512/service_arrays.npz') as z:arrays={k:z[k].copy() for k in z.files}
    report_source=read(DATA/'TRAIN512/report.json');summary=[]
    for rec in report_source['records']:
        name=rec['name'];cost=arrays[name+'.cost'];fifo=arrays[name+'.fifo'];cap=arrays[name+'.matching_capacity']
        need(cost.shape==(3,512) and np.all(cost>=0) and np.all(fifo<=128) and np.all(cap==1),'fullsaved summary inventory')
        need(float(cost.mean())==rec['cost_ns']['mean'] and int(np.count_nonzero(cap==1))==rec['perfect_matching_events'],'fullsaved reported mean/capacity')
        summary.append(dict(name=name,events=1536,mean_ns=float(cost.mean()),all_FIFO128=True,all_cap1=True))
    records=[];replay={};critical={};candidate_accounting=[]
    for path in files:
        saved=read(path);parts=path.name.split('.');layer=int(parts[1][1:]);name='.'.join(parts[2:-1])
        own,cost=checker.scalar_execute(saved['nodes'],saved['service']['down_first']);want=np.array([[n['start_ns'],n['end_ns']] for n in saved['nodes']],dtype=np.int64)
        need(np.array_equal(own,want) and cost==saved['cost_ns'],'saved node timestamp mismatch')
        basic=checker.semantic_audit(saved,own);bind=binding_audit(saved,routes[layer,0],adj[layer],expected[layer]);cp=critical_path(saved)
        need(cost==int(arrays[name+'.cost'][layer,0]),'trace/summary query binding')
        if 'candidate' in name:
            submit=next(n['end_ns'] for n in saved['nodes'] if n['kind']=='Down_submit' and n['expert_rank']==5)
            copies=saved['service']['input_copies'];owner_bound=submit+250+256*copies+89940+64
            need(cp['critical_final_resident_job']['expert']==max(routes[layer,0]),'candidate critical expert is sortedrank5')
            need(cp['final_command_start_ns']==submit+250 and cp['final_command_end_ns']==submit+500,'candidate siblingcommand serialization')
            need(cost==owner_bound+266 and cp['terminal_PCIe_duration_ns']==16,'candidate exact266ns oracle gap')
            candidate_accounting.append(dict(path=str(path),private_submit_ns=submit,minimum_single_job_owner_end_ns=owner_bound,
                                            sibling_command_serialization_ns=250,terminal_PCIe_ns=16,actual_cost_ns=cost,gap_ns=266))
        records.append(dict(path=str(path),sha256=sha(path),layer=layer,query_index=0,name=name,cost_ns=cost,**basic,binding=bind))
        replay[path.name]=own;critical[path.name]=cp
    # Mutations are checked against the actual saved independent-address binder.
    baseline=read(DATA/'TRAIN512/trace.L0.candidate.g2availability.input1.GF.json');controls=[]
    mutations=dict(wrong_block=lambda p:p.__setitem__('block',1-p['block']),wrong_chip=lambda p:p.__setitem__('chip',(p['chip']+2)%12),
                   wrong_WL=lambda p:p.__setitem__('wl_start',p['wl_start']+1),wrong_row=lambda p:p.__setitem__('row_start',p['row_start']+1),
                   wrong_group=lambda p:p.__setitem__('resident_copy',(p['resident_copy']+1)%6),wrong_job_id=lambda p:p.__setitem__('resident_job_id','e63.g0.c0'),
                   wrong_expert=lambda p:p.__setitem__('expert',(p['expert']+1)%64))
    for name,mutate in mutations.items():
        bad=copy.deepcopy(baseline);p=next(j['physical_down_job'] for j in bad['jobs'] if j['phase']==1);mutate(p)
        try:binding_audit(bad,routes[0,0],adj[0],expected[0])
        except RuntimeError as e:controls.append(dict(name=name,detected=True,diagnostic=str(e)))
        else:raise RuntimeError('undetected physical mutation '+name)
    for name in ['wrong_actual_chip','wrong_actual_columns','wrong_native_recipe_block']:
        bad=copy.deepcopy(baseline);job=next(j for j in bad['jobs'] if j['phase']==1)
        if name=='wrong_actual_chip':job['chip']=(job['chip']+2)%12
        elif name=='wrong_actual_columns':job['columns']=256
        else:
            p=copy.deepcopy(job['physical_down_job']);p['block']=1-p['block'];job['recipe_signature']=resident_recipe_signature(p)[0]
        try:binding_audit(bad,routes[0,0],adj[0],expected[0])
        except RuntimeError as e:controls.append(dict(name=name,detected=True,diagnostic=str(e)))
        else:raise RuntimeError('undetected actual-builder mutation '+name)
    # Only12 micrographs: all8service axes on one failing route plus4 checks
    # across the other layers. C++ is the sealed executor, never target audit.
    driver=load('g2_actual_builder_for_cap2_micrographs_only',TARGET)
    binary=next((DATA/'build').glob('*.so'));lib=ctypes.CDLL(str(binary));call=lib.run_graph
    a64=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS');a32=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')
    call.argtypes=[ctypes.c_int,a64,a32,a32,ctypes.c_int,a64];call.restype=ctypes.c_int64
    recipes=driver.r.Recipes(driver.f.load_pipeline(),driver.a.load_shaper());fallback=[]
    for layer in range(3):
        missing=np.flatnonzero(~adj[layer,:,0]);route=np.r_[missing[:3],missing[-3:]].astype(int).tolist()
        need(len(set(route))==6 and hall_failure(adj[layer],route),'synthetic fallback route must failcap1')
        decision=driver.dispatch(route,layouts[layer]);need(decision['capacity']==2 and decision['feasible'],'actual cap2 dispatch')
        resident=driver.resident_jobs(layouts[layer]);need({j['resident_job_id']:j for js in resident.values() for j in js}==expected[layer],'actual runtime resident builder binding')
        selected=[j for e,g in decision['assignment'].items() for j in resident[int(e),int(g)]]+resident[64,0]
        axes=list(itertools.product(['candidate','RR'],[1,4],[0,1])) if layer==0 else [(f,1,0) for f in ['candidate','RR']]
        for family,copies,df in axes:
            maps={family+'.g2availability':[selected]*3}
            graph=driver.r.graph_for(route,layer,family,'g2availability',driver.f.Service(copies,df,0),recipes,None,maps)
            need(graph is not None,'fallback strict PB must construct')
            n,o,d=graph.arrays();times=np.zeros((len(n),2),dtype=np.int64);cost=int(call(len(n),n,o,d,df,times));need(cost>0,'C++ fallback did not finish')
            saved=generated_record(graph,times,cost,checker);own,owncost=checker.scalar_execute(saved['nodes'],df)
            need(np.array_equal(own,times) and owncost==cost,'fallback C++ vs scalar replay')
            basic=checker.semantic_audit(saved,own);need(basic['FIFO128_valid'],'fallback exceeds128B FIFO')
            bind=binding_audit(saved,route,adj[layer],expected[layer],2);need(max(bind['group_loads'])==2,'fallback must execute serial cap2jobs')
            cp=critical_path(saved);name=f'fallback.L{layer}.{family}.input{copies}.{"DF" if df else "GF"}'
            write(OUT/(name+'.trace.json'),saved);replay[name]=own;critical[name]=cp
            fallback.append(dict(name=name,layer=layer,route=route,dispatch=decision,cost_ns=cost,**basic,binding=bind))
    np.savez_compressed(OUT/'independent_node_times.npz',**replay);write(OUT/'critical_paths.json',critical)
    need(hashes=={p:sha(p) for p in hashes},'sealed source or inputs changed during audit')
    out=dict(status='pass_saved_traces_and_cap2_micrographs',new_audit_source_sha256=sha(__file__),inputs_sha256=hashes,
        reused_independent_checker_sha256=sha(CHECKER),saved_trace_count=len(records),saved_traces=records,
        saved_total_nodes=sum(r['nodes'] for r in records),saved_total_edges=sum(r['edges'] for r in records),saved_total_jobs=sum(r['jobs'] for r in records),
        fullarray_arithmetic_only=summary,resident_jobs_independently_bound=508*3,physical_mutation_controls=controls,
        candidate_critical_accounting=candidate_accounting,
        fallback_micrographs=fallback,critical_paths_sha256=sha(OUT/'critical_paths.json'),replay_times_sha256=sha(OUT/'independent_node_times.npz'),
        actual_command=['python',str(Path(__file__).resolve())],elapsed_seconds=time.monotonic()-began,
        limitations=['Reuses the earlier independent scalar scheduler/semantic checker; this is a new binding/criticalpath driver, not a second new independent scheduler.',
            'Only20saved qindex0representatives are replayed; candidateL2 lacks four files due targetcache reuse. All12288TRAINcost/FIFO/capacity entries are read for summary arithmetic, not independently reexecuted.',
            'Target imports are confined to constructing12synthetic cap2micrographs. The sealed C++ executor and independent scalar oracle both execute them; targetaudit and wholecampaign execute are never called.',
            'Cap2routes are deterministic constructed counterexamples using frozen maps, not captured heldout data. Allsixexperts retain fulltwo768rowjobs plus sharedfour384rowjobs,32totalchipjobs includingGU.',
            'Physical verification binds selected residentjobIDs, copies, expert/group/chip/block/WL androwextents to closed-form coordinates and frozenmaps; native recipe digests are independently recomputed from absolute block/WL page extents. It is not coefficient programming or a new numerical-quality/ABI claim.',
            'Existing conditional skipWLCR/IPR/commonGU source permutation/preECCdrop, twoPBslots, implicit6.4GB/s PE sustain, cold28us availability and unknown firstphase/discharge split remain assumptions.',
            'CompleteC512private/C1024shared input bytes and singleI32whole-row semantics remain unchanged. StandardCBC/replicateddeclustering is prior; matchingruntime overhead is not measured.'])
    write(OUT/'report.json',out)
    (OUT/'report.md').write_text(
        f"All20saved g2actual representatives pass exact independent scalar replay: {out['saved_total_nodes']:,}nodes, {out['saved_total_edges']:,}edges, {out['saved_total_jobs']:,}chipjobs. Every start/end andmakespan matches. FullGU→hostA8→fullprivateDown→all7combine, channel/resource exclusion, chipownership throughcompute+ONFI, continuousdrain128BFIFO and exact8192GU+10752Down rows pass. FourcandidateL2files were notsaved duecache reuse; this is not a replay of everyTRAINquery.\n\n"
        'All1524residentchipjob records (508/layer) match independent support/coordinate formulas. Everyselected resident_job_id binds to its exactexpert/group/physicalchip/block/WL andcanonicalrowband. Native recipe digests are independently reconstructed from their absolute page extents andPBsettings, binding the instantiated timingrecipe as well as metadata. Ten mutations ofmetadata, actualchip/columns andnative-recipe block are allcaught.\n\n'
        'Twelve syntheticcap2micrographs execute with the actualdispatch/builder andsealedC++executor, then match independent scalar replay andcausal/physical/FIFO checks. These contain32chipjobs each, preserving allsixfullprivate experts andsharedDown; groupcapacity2 causes serialjobs rather than droppedwork. They are constructedfallback tests, not heldoutobservations.\n\n'
        'The augmented event/resource/owner DAG reproduces every node earlieststart andprovides exactcritical chains. Candidate254712/258856ns endswith thelateexpert\'s secondchipcommand, fullinput,89940ns nativejob,last64nsONFI and16nsPCIe; the266ns gap above theearlierready-aware oracle is250ns siblingcommand serialization plus16ns terminalPCIe.\n\n'
        + '\n'.join(out['limitations'])+'\n')
    print(json.dumps(dict(status=out['status'],saved_traces=len(records),fallback_micrographs=len(fallback),nodes=out['saved_total_nodes'],
                         fallback_costs=[dict(name=r['name'],cost_ns=r['cost_ns'],FIFO=r['peak_FIFO_bytes']) for r in fallback],elapsed_seconds=out['elapsed_seconds']),indent=2))


if __name__=='__main__':main()
