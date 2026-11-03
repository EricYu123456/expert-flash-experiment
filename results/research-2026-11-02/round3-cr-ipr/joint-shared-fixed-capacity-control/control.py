#!/usr/bin/env python3
"""Posthoc equal-payload fixed-shared ablation, unchanged sealed causal engine."""
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
import time
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];BASE=HERE.parent
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
J=module('fixed_shared_joint_driver',ROOT/'scripts/research_20261102_cr_joint_shared_ffn.py')
f,r,a,native,joint=J.f,J.r,J.a,J.native,J.joint
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def axes():return [(family,'cr',copies) for family in ['candidate','RR'] for copies in [1,4]]+[("g8_fixed",array,copies) for array in ['cr','ordinary'] for copies in [1,4]]
def dump(p,x):f.dump(p,x)

def dispatch(route,layout):
 ids=sorted(map(int,route));supports=[[g for g in S if g<6] for S in layout['supports']]
 choice=J.g2.k.assign_route(ids,supports,1)
 if not choice['feasible']:choice=J.g2.k.assign_route(ids,supports,2)
 assert choice['feasible'] and len(choice['assignment'])==6
 selected=[]
 for e in ids:
  g=int(choice['assignment'][e]);assert 0<=g<6 and g in layout['supports'][e]
  selected += [joint.execution_view(joint.build_private_job(e,g,local,layout)) for local in [0,1]]
 for half,g in [(0,6),(1,7)]:
  selected += [joint.execution_view(joint.build_shared_job(half,g,local,layout)) for local in [0,1]]
 assert len(selected)==16 and all(j['copy']==0 for j in selected)
 for e in ids+[64]:
  bands=sorted((j['row_start'],j['row_start']+j['rows']) for j in selected if j['expert']==e)
  assert bands[0][0]==0 and bands[-1][1]==1536 and all(x[1]==y[0] for x,y in zip(bands,bands[1:]))
 return dict(choice,shared_half_assignment={0:6,1:7},selected_jobs=selected,
    full_group_load=choice['group_load']+[1,1],input_contract='Onecompleteprivateinputanddotpernativejob;cap2jobsserialonchip.')

def freeze():
 assert not (HERE/'protocol.frozen.json').exists()
 shutil.copyfile(joint.OUT/'layout.frozen.json',HERE/'layout.reference.frozen.json')
 shutil.copyfile(joint.OUT/'resident_jobs.frozen.json',HERE/'resident_jobs.reference.frozen.json')
 layout=f.read(HERE/'layout.reference.frozen.json');resident=f.read(HERE/'resident_jobs.reference.frozen.json')
 assert len(resident)==544 and joint.resident_jobs(layout)==resident
 supports=[[g for g in S if g<6] for S in layout['supports']]
 hall=[]
 for T in range(64):
  n=sum(all(T&(1<<g) for g in S) for S in supports);bound=2*T.bit_count()
  assert min(n,6)<=bound
  hall.append(dict(T=T,size=T.bit_count(),contained_experts=n,capacity2_Hall_lhs=min(n,6),capacity=bound))
 dump(HERE/'dispatch_certificate.json',dict(status='PASS',private_groups=list(range(6)),shared_fixed_groups=[6,7],
   restricted_support_min_degree=min(map(len,supports)),all64_private_T_Hall_capacity2=hall,
   original_resident_jobs=544,payload_MiB=303,layout_unchanged=True,addresses_unchanged=True,
   no_refit=True,proof='Restricteddegree>=2;withinanytwo privategroups atmostoneuniqueoriginal4supportcanfit (bothsharedgroupsmustbeincluded). For>=3privategroups capacity2>=6.'))
 paths=[Path(__file__),ROOT/'scripts/research_20261102_cr_joint_shared_ffn.py',
   ROOT/'scripts/research_20261102_eight_pair_cbc.py',ROOT/'scripts/research_20261102_cr_g2_availability_ffn.py',
   ROOT/'scripts/research_20261102_cr_native_group_control.py',J.g2.DRIVER,J.g2.KERNEL,
   r.BASE/'asymmetric_down.py',r.BASE/'policy_addendum.py',r.BASE/'shaping_addendum.py',a.SHAPER,
   ROOT/'scripts/research_20261102_cr_ipr_full_ffn.py',f.PIPE,f.CPP,f.TRAIN,
   r.BASE/'g8-addendum/group_maps.frozen.npz',r.SHARED/'menus.frozen.npz',r.SHARED/'selection.frozen.json',
   joint.OUT/'layout.frozen.json',joint.OUT/'resident_jobs.frozen.json',
   HERE/'layout.reference.frozen.json',HERE/'resident_jobs.reference.frozen.json',HERE/'dispatch_certificate.json',
   J.OUT/'TRAIN512/service_arrays.npz']
 dump(HERE/'protocol.frozen.json',dict(status='frozen_before_equal_payload_ablation',utc=f.utc(),
   inputs_sha256={str(p):sha(p) for p in paths},axes=axes(),priority='GFonly',screen_indices=list(range(64)),full_indices=list(range(512)),layers=[0,1,2],
   gate='All8axes×192screen events mustcompletefinitePBvalid andFIFO<=128. Correctnessgateonly;nogainthreshold/selection. Runall8fullaxes ifpass; retainallslower/P99/failures.',
   ablation='Identical303MiB/544residentjobs/fullNANDaddresses versusjoint8. FixS0togroup6,S1togroup7;expertsonlygroups0..5,deterministiccap1then2matching. Unusedreplicasremainresident. Executionviewcopy0;allper-expertrowscomplete.',
   GU='candidate/RR inheritedconditionalcommonGU source/drop policy;g8fixed CR/ordinaryusessealednativeRecipes. Same Service(inputcopies,GF,activation0),r.graph_for,causalExecutor,PB/ECC/FIFO.',
   comparison='Onlysamefamily,arraymode,inputcopies,GFprimaryjoint8TRAIN512arrays. No244.5/289.5MiBreferences ornewmobility/layoutfit.',
   traces='Everylayerq0andfirstcapacity2eventperaxis/stage,savedwithdispatch/selectedresidentjobs;runactualbackendfulleventaudit evenifcostsignaturecached.',
   chronology='Posthocstructuralablationdesignedafterjointprimary; previouslyobservedTRAINonly. No independentvalidationclaim.',
   blind_scope='OnlyTRAINroutes,synthetic/layout/sourceartifacts,andjointprimaryTRAINarraysallowed. No prospectiveinputs/outcomesorprimarydiagnosticreplayread.',
   assumptions='Inheritedconditionalserviceenvelope/W8_A8/fullprivateinputs/oneGEMVperchip/128B outputFIFO;noextraMACcontext/FIFO/partialdot.',
   source_integrity='Allsealeddependenciesread-only;localreferenceledgersbyteidentical.'))
 print(json.dumps(dict(status='frozen',axes=axes(),resident_jobs=544,protocol_sha256=sha(HERE/'protocol.frozen.json'))),flush=True)

def checked():
 protocol=f.read(HERE/'protocol.frozen.json')
 for path,h in protocol['inputs_sha256'].items():assert sha(path)==h,path
 routes,_,_,groups=r.inputs()
 layout=f.read(HERE/'layout.reference.frozen.json')
 resident={j['resident_job_id']:j for j in f.read(HERE/'resident_jobs.reference.frozen.json')}
 return protocol,routes,groups,layout,resident

def paired(fixed,joint_cost):
 valid=(fixed>=0)&(joint_cost>=0)
 out=dict(paired_events=int(valid.sum()),requested=int(fixed.size))
 if valid.any():
  x=fixed[valid];y=joint_cost[valid];d=x-y
  out.update(joint_mean_ns=float(y.mean()),fixed_mean_ns=float(x.mean()),
    fixed_mean_increase_pct=100*(float(x.mean()/y.mean())-1),joint_mean_reduction_vs_fixed_pct=100*(1-float(y.mean()/x.mean())),
    joint_event_P99_reduction_vs_fixed_pct=100*(1-float(np.percentile(y,99)/np.percentile(x,99))),
    fixed_slower_events=int(np.count_nonzero(d>0)),fixed_equal_events=int(np.count_nonzero(d==0)),fixed_faster_events=int(np.count_nonzero(d<0)),
    worst_fixed_regression_ns=int(d.max()),best_fixed_improvement_ns=int(-d.min()))
 if valid.all():out['joint_3layer_sum_P99_reduction_vs_fixed_pct']=100*(1-float(np.percentile(joint_cost.sum(axis=0),99)/np.percentile(fixed.sum(axis=0),99)))
 return out

def execute(mode):
 protocol,routes,groups,layout,resident=checked();screen=mode=='screen'
 if not screen:
  gate=f.read(HERE/'correctness_gate.frozen.json');assert gate['passed'] and gate['screen_report_sha256']==sha(HERE/'screen/report.json')
 indices=np.array(protocol['screen_indices'] if screen else protocol['full_indices']);dest=HERE/('screen' if screen else 'TRAIN512');dest.mkdir(exist_ok=True)
 assert not (dest/'report.json').exists()
 dump(dest/'run.frozen.json',dict(utc=f.utc(),source_sha256=sha(__file__),protocol_sha256=sha(HERE/'protocol.frozen.json'),axes=axes(),indices=indices.tolist(),gate_rule='PB/FIFOonly,nooutcome selection'))
 with np.load(J.OUT/'TRAIN512/service_arrays.npz') as z:
  anchors={f'{family}.{array}.joint8.input{copies}.GF.cost':z[f'{family}.{array}.joint8.input{copies}.GF.cost'].copy() for family,array,copies in axes()}
 executor=f.Executor(HERE);pipeline={};cache={};arrays={};records=[];failures=[];trace_index=[];dispatch_cache={};started=time.monotonic()
 for ci,(family,array,copies) in enumerate(axes()):
  name=f'{family}.{array}.fixedShared.input{copies}.GF';key=(family=='g8_fixed',array)
  if key not in pipeline:pipeline[key]=native.Recipes(f.load_pipeline(),a.load_shaper(),array) if key[0] else r.Recipes(f.load_pipeline(),a.load_shaper())
  recipes=pipeline[key];cost=np.full((3,len(indices)),-1,dtype=np.int64);fifo=cost.copy();caps=cost.copy();work=Counter();counts=Counter()
  for layer in range(3):
   seen_cap2=False
   for qi,index in enumerate(indices):
    route=routes[layer,index];dkey=tuple(sorted(map(int,route)))
    if dkey not in dispatch_cache:
     decision=dispatch(route,layout)
     for job in decision['selected_jobs']:assert job==resident[job['resident_job_id']]
     dispatch_cache[dkey]=decision
    decision=dispatch_cache[dkey];caps[layer,qi]=decision['capacity'];jobs=decision['selected_jobs']
    maps={family+'.fixedShared':[jobs]*3}
    graph=r.graph_for(route,layer,family,'fixedShared',f.Service(copies,0,0),recipes,groups,maps)
    if graph is None:
     failures.append(dict(axis=name,layer=layer,index=int(index),route=list(map(int,route)),kind='finitePB_graph_rejected',dispatch=decision));continue
    signature,data=graph.signature();signature=(signature,array,json.dumps(dict(graph.pipeline_totals),sort_keys=True))
    tags=[]
    if qi==0:tags.append('q0')
    if decision['capacity']==2 and not seen_cap2:tags.append('first_cap2');seen_cap2=True
    if signature in cache and not tags:value,audit=cache[signature]
    else:
     result=executor.run(graph,data);audit=f.audit(graph,result,full=bool(tags) or qi==len(indices)-1);value=result['cost_ns'];cache[signature]=(value,audit)
     if tags:
      stem=f'trace.L{layer}.{name}.{"_".join(tags)}'
      f.dump_trace(dest/(stem+'.json'),graph,result,audit)
      dump(dest/(stem+'.dispatch.json'),dict(axis=name,layer=layer,index=int(index),route=list(map(int,route)),tags=tags,decision=decision))
      trace_index.append(dict(axis=name,layer=layer,index=int(index),capacity=decision['capacity'],tags=tags,path=str(dest/(stem+'.json'))))
    cost[layer,qi]=value;fifo[layer,qi]=audit['peak_output_fifo_bytes']
    for k,v in audit['resource_work_ns'].items():work[k]+=v
    for k,v in graph.pipeline_totals.items():counts[k]+=v
  valid=cost>=0;name_joint=f'{family}.{array}.joint8.input{copies}.GF.cost';anchor=anchors[name_joint][:,indices]
  arrays[name+'.cost']=cost;arrays[name+'.fifo']=fifo;arrays[name+'.matching_capacity']=caps
  n=int(valid.sum());rec=dict(name=name,family=family,array_mode=array,input_copies=copies,priority='GF',requested=int(cost.size),completed=n,
    valid_FIFO128=int(np.count_nonzero(valid&(fifo<=128))),cost_ns=f.stats(cost[valid]) if n else None,
    three_layer_sum_ns=f.stats(cost.sum(axis=0)) if valid.all() else None,FIFO_peak_bytes=f.stats(fifo[valid]) if n else None,
    cap1_events=int(np.count_nonzero(caps==1)),cap2_events=int(np.count_nonzero(caps==2)),payload_MiB=303,resident_jobs=544,
    resource_work_per_completed={k:v/n for k,v in work.items()} if n else {},pipeline_counts_per_completed={k:v/n for k,v in counts.items()} if n else {},
    joint_anchor=name_joint,comparison=paired(cost,anchor))
  records.append(rec)
  progress=dict(finished=ci+1,total=len(axes()),last=name,unique_graphs=len(cache),elapsed_seconds=time.monotonic()-started)
  dump(dest/'progress.json',progress);print(json.dumps(progress),flush=True)
 np.savez_compressed(dest/'service_arrays.npz',**arrays)
 dump(dest/'failures.json',failures);dump(dest/'trace_index.json',trace_index)
 valid=all(x['completed']==x['requested'] and x['valid_FIFO128']==x['requested'] for x in records)
 report=dict(status='PASS' if valid else 'RETAINED_FAILURES',mode=mode,source_sha256=sha(__file__),protocol_sha256=sha(HERE/'protocol.frozen.json'),indices=indices.tolist(),records=records,
   finitePB_and_FIFO128_all_valid=valid,failures=len(failures),elapsed_seconds=time.monotonic()-started,unique_graphs=len(cache),saved_audit_traces=len(trace_index),
   scope='PosthocTRAINequal303MiB/544residentjobfixedsharedvsjointmovabilityablation;no layout/address/refit/dispatchcostoracle. No prospective read.')
 dump(dest/'report.json',report)
 if screen:dump(HERE/'correctness_gate.frozen.json',dict(passed=valid,screen_report_sha256=sha(dest/'report.json'),protocol_sha256=sha(HERE/'protocol.frozen.json'),rule='All8axescompletePB+128B FIFO;no performance threshold orselection.'))
 lines=[f'# Equal-payload fixed-shared control ({mode})','',
   'The same303MiB/544residentjobs and exactaddresses are retained. Onlyexecution isrestricted: S0 ongroup6,S1 ongroup7; routedexperts cap1then2 ongroups0..5. Positive mobilityreduction means thejointdispatch isfaster.','',
   '| GU | array | inputcopies | fixedmean µs | jointmean µs | jointmean reduction vsfixed | eventP99 reduction | cap2events |','|---|---|---:|---:|---:|---:|---:|---:|']
 for x in records:
  c=x['comparison'];lines.append(f'| {x["family"]} | {x["array_mode"]} | {x["input_copies"]} | {c.get("fixed_mean_ns",0)/1000:.3f} | {c.get("joint_mean_ns",0)/1000:.3f} | {c.get("joint_mean_reduction_vs_fixed_pct",0):.3f}% | {c.get("joint_event_P99_reduction_vs_fixed_pct",0):.3f}% | {x["cap2_events"]} |')
 lines += ['',f'Completed {sum(x["completed"] for x in records)}events; allfinitePB/FIFO128valid={valid}. Everyregression,aligned3layerP99,resourcecount,andfailureremainsinthereport. Q0andfirstcap2tracesinclude exactselectedresidentjobs.', '',
   'Screenindices0..63 are solelyaPB/FIFOcorrectnessgate. All8GFaxes arefrozen; no gain-basedselection orfitting. The fullcomparison reusesonlymatchedjoint8TRAINarrays. This is a posthoc standardresourcepartition/sharedmobilityablation, notnewsolver/CBC/hardwareorindependentvalidation. ConditionalGUfeaturesandnativeg8CR/ordinarycontractsareinheritedunchanged. No prospective data/outcomes are read.']
 (dest/'report.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps(dict(status=report['status'],all_valid=valid,mode=mode,records=len(records))),flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['freeze','screen','full']);args=ap.parse_args()
 freeze() if args.mode=='freeze' else execute(args.mode)
