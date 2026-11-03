#!/usr/bin/env python3
"""No new search; row/owner audit plus one independent B2 and B8 timeline."""
import collections,ctypes,hashlib,json,pathlib,sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];BASE=ROOT/'results/research-2026-11-02/round2';OUT=BASE/'multi-request-onecopy-screen';sys.path.insert(0,str(ROOT/'scripts'))
import research_20261103_ordinary_read_backend as backend
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def main():
 report=json.loads((OUT/'report.json').read_text());proto=json.loads((OUT/'protocol.frozen.json').read_text());saved=np.load(OUT/'arrays.npz');inputs={p:sha(p) for p in proto['inputs_sha256']};assert inputs==proto['inputs_sha256'];geometry=[];models={};relax=[]
 for model,copies,path in [('granite4tiny',1,BASE/'compact-coupled-solver/continuation/native/0107.npz'),('deepseek',4,BASE/'native-joint-refinement/pilot-v1/deepseek.L0.input4.independent_paired.seed20261106/arrays.npz')]:
  z=np.load(path);rows=z['rows'];routes=z['routes'];assert np.array_equal(saved[model+'.single_native'],z['cost'])
  campaign=BASE/f'ordinary-read-compiler/full-campaign-v{5 if copies==1 else 6}-W8-{ "training" if copies==1 else "input4"}/summary.json';rec=next(r for r in json.loads(campaign.read_text())['records'] if r['model']==model and r['layer']==0 and r['command_ns']==250);D,I,E,K,L,SI=rec['shape'];s=backend.Service(**rec['service']);plan=backend.lower_rows(rows[:,0],rows[:,1],D,[I]*64+[SI],s);models[model]=(rec,rows,routes,plan)
  # All selected alternative layouts retain every logical full row exactly once.
  for r in [x for x in report['records'] if x['model']==model]:
   B=r['batch'];cohort=routes.reshape(32//B,B,16,K).transpose(0,2,1,3).reshape(-1,B,K)
   for mapping in r['maps']:
    rr=rows.copy()
    if mapping['name']!='strong':
     for p,label in enumerate(mapping['phase_labels']):
      item=next(x for x in rec['fitting'][p] if x['label']==label);base=np.asarray(item['base_rows']);width=np.count_nonzero(base);shifts=np.asarray(item['labels'])*(width if item['family']=='disjoint' else 1);rr[:64,p]=np.stack([np.roll(base,int(k)) for k in shifts])
    pplan=backend.lower_rows(rr[:,0],rr[:,1],D,[I]*64+[SI],s);total=0
    for e in range(65):
     for p in [0,1]:
      lo,hi=pplan['offsets'][e,p];cursor=0;chips=set();C=D if p==0 else SI if e==64 else I
      for logical,job in zip(pplan['logical'][lo:hi],pplan['jobs'][lo:hi]):
       c,n,columns,pages,waves=map(int,job);assert c not in chips;chips.add(c);assert columns==C and logical.get('column_start',0)==0
       assert logical['row_start']==cursor and logical['rows']==n and n==rr[e,p,c];cursor+=n;total+=n;assert pages==(n*C+16383)//16384 and waves==(pages+3)//4 and waves<=192
      assert cursor==(2*(SI if e==64 else I) if p==0 else D)
    assert np.array_equal(rr[-1],rows[-1]);geometry.append(dict(model=model,batch=B,name=mapping['name'],all_fullrows_once=True,onecopy=True,one_job_per_active_chip_phase=True,fullinput_columns=True,total_logical_weight_rows=total))
   table=backend.occupancy_table(plan,s);selected=np.concatenate([cohort,np.full((len(cohort),B,1),64,np.int32)],axis=2);load=table['chip_ns'][selected].sum((1,2,3));count=table['command_counts'][selected].sum((1,2,3));cols=np.column_stack((np.full(65,D),np.r_[[I]*64,SI]));inp=table['command_counts']*(cols[:,:,None]*s.input_channel_copies//2);inputload=inp[selected].sum((1,2,3));best=r['maps'][0]['best_policy']['mean_cohort_ns'];zero_command=(load-count*250).max(1).mean()+2000;zero_reload=(load-inputload).max(1).mean()+2000;zero_both=(load-count*250-inputload).max(1).mean()+2000;relax.append(dict(model=model,batch=B,feasible_strong_mean_ns=best,zero_command_only_gain_ceiling_pct=float(100*(1-zero_command/best)),zero_input_reload_only_gain_ceiling_pct=float(100*(1-zero_reload/best)),zero_both_gain_ceiling_pct=float(100*(1-zero_both/best)),scope='Optimistic zero-cost relaxations on currentfixedrowmap, not implementations or newhardware. Actualgroupwidth changes cannotremoveallcostand mayraiseweightimbalance.'))
 # Exactly two independent timelines: use saved G4B2 and a new single DeepB8.
 lib=ctypes.CDLL(str(OUT/'cohort.so'));fn=lib.eval_ready_cohorts;p32=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS');p64=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS');fn.argtypes=[p32,p32,p32,p32,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,p64,p64,p64,ctypes.c_void_p,p64];timelines=[];outtraces={}
 for model,B in [('granite4tiny',2),('deepseek',8)]:
  rec,rows,routes,plan=models[model];K=6;s=backend.Service(**rec['service']);cohort=np.ascontiguousarray(routes.reshape(32//B,B,16,K).transpose(0,2,1,3).reshape(-1,B,K)[:1],np.int32)
  if B==2:trace=saved[f'{model}.B2.strong.DF0.trace0'];cost=int(saved[f'{model}.B2.strong.DF0.cost'][0])
  else:
   buf=np.zeros((B*1400,8),np.int64);n=np.zeros(1,np.int64);v=np.zeros(1,np.int64);lb=v.copy();fn(plan['offsets'],plan['jobs'],plan['input_columns'],cohort,64,1,B,6,s.array(),v,lb,ctypes.c_void_p(buf.ctypes.data),n);trace=buf[:n[0]];cost=int(v[0])
  graph=backend.ReferenceGraph(s);joboff=0;request_inputs=[]
  for request in range(B):
   sub=backend.ReferenceGraph.build(plan,np.r_[cohort[0,request],64],s);offset=len(graph.nodes);request_inputs.append(dict(request=request,GU_submit_node=offset,GU_upload_node=offset+1,logical_experts=np.r_[cohort[0,request],64].tolist()))
   for ni,node in enumerate(sub.nodes):
    x=dict(node);x['deps']=[d+offset for d in x['deps']];x['job']=x['job']+joboff if x['job']>=0 else -1;x['expert']=x['expert']+request*7 if x['expert']>=0 else request*7 if ni==len(sub.nodes)-1 else -1;graph.nodes.append(x)
   joboff+=sub.job_count
  direct=graph.run();assert direct['cost']==cost;assert np.array_equal(direct['trace'],trace);resource=collections.defaultdict(list);owners=collections.defaultdict(list);jobs=collections.defaultdict(list)
  for i,(node,t) in enumerate(zip(graph.nodes,trace)):
   r,d,start,end,phase,ordinal,chip,job=map(int,t);assert end-start==d and all(trace[z,3]<=start for z in node['deps']);resource[r].append((start,end));assert (r,d,phase,ordinal,chip,job)==tuple(node[k] for k in ['resource','duration','phase','expert','chip','job'])
   if job>=0:jobs[job].append(i)
  for job,ii in jobs.items():
   cmd=[i for i in ii if graph.nodes[i]['acquire']];out=[i for i in ii if graph.nodes[i]['release']];assert len(cmd)==len(out)==1;cmd,out=cmd[0],out[0];c=graph.nodes[cmd]['chip'];start,end=int(trace[cmd,2]),int(trace[out,3]);owners[c].append((start,end));assert all(start<=trace[i,2]<=trace[i,3]<=end for i in ii)
   inp=next(i for i in ii if graph.nodes[i]['resource']==3+c%8 and len(graph.nodes[i]['deps'])==2);upload=next(d for d in graph.nodes[inp]['deps'] if d!=cmd);assert graph.nodes[upload]['resource']==2;ordinal=graph.nodes[cmd]['expert'];request=ordinal//7
   if graph.nodes[cmd]['phase']==0:assert upload==request_inputs[request]['GU_upload_node']
   else:assert upload!=request_inputs[request]['GU_upload_node'];assert graph.nodes[upload]['expert']==ordinal and graph.nodes[upload]['phase']==1
  for intervals in list(resource.values())+list(owners.values()):
   intervals=sorted((a,b) for a,b in intervals if a<b);assert all(b<=c for (_,b),(c,_) in zip(intervals,intervals[1:]))
  timelines.append(dict(model=model,batch=B,native_ns=cost,ReferenceGraph_ns=direct['cost'],nodes=len(trace),native_jobs=len(jobs),every_dependency_resource_and_chipowner_checked=True,every_request_own_GU_input_upload=True,every_expert_private_Down_input=True,request_input_nodes=request_inputs));outtraces[model+f'.B{B}.native']=trace;outtraces[model+f'.B{B}.ReferenceGraph']=direct['trace']
 assert inputs=={p:sha(p) for p in inputs};np.savez_compressed(OUT/'independent_timeline_traces.npz',**outtraces);audit=dict(status='PASS',source_sha256=sha(__file__),single_request_512_regressions_exact_models=2,all_fullrow_layout_checks=len(geometry),geometry=geometry,timelines=timelines,input_command_zero_cost_relaxations=relax,all_inputs_unchanged=True,no_weights_reuse_across_requests=True,no_PB_reuse=True,no_multicontext=True,no_heldout=True);dump(OUT/'independent_contract_audit.json',audit);print(json.dumps({k:audit[k] for k in ['status','single_request_512_regressions_exact_models','all_fullrow_layout_checks','timelines','input_command_zero_cost_relaxations']},indent=2))
if __name__=='__main__':main()
