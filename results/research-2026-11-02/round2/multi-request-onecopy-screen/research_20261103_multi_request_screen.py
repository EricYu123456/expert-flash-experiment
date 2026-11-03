#!/usr/bin/env python3
"""Bounded TRAIN-only independent-request multiplexing/width feasibility screen."""
import argparse,ctypes,hashlib,json,pathlib,subprocess,sys,time
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];BASE=ROOT/'results/research-2026-11-02/round2';OUT=BASE/'multi-request-onecopy-screen';SELF=pathlib.Path(__file__).resolve();ADAPTER=SELF.with_name('research_20261103_multi_request_adapter.cpp');CORE=SELF.with_name('research_20261103_ordinary_read_native.cpp')
sys.path.insert(0,str(ROOT/'scripts'));import research_20261103_ordinary_read_backend as b
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
CASES=[('granite4tiny',1,BASE/'compact-coupled-solver/continuation/native/0107.npz'),('deepseek',4,BASE/'native-joint-refinement/pilot-v1/deepseek.L0.input4.independent_paired.seed20261106/arrays.npz')]
def freeze():
 OUT.mkdir(exist_ok=False);files=[SELF,ADAPTER,CORE,pathlib.Path(b.__file__),ROOT/'docs/research_2026_11_02.md',ROOT/'docs/research_2026_11_02_round2.md',ROOT/'scripts/research_20261102_replay_oracle.py',BASE/'replay-oracle/summary.json']
 for model,copies,path in CASES:files.extend([path,BASE/f'ordinary-read-compiler/full-campaign-v{5 if copies==1 else 6}-W8-{ "training" if copies==1 else "input4"}/summary.json',BASE/f'w8-format-training/{model}/decode.routes.npz'])
 dump(OUT/'protocol.frozen.json',dict(inputs_sha256={str(p):sha(p) for p in files},models=[dict(model=m,copies=c,layer=0,baseline=str(p)) for m,c,p in CASES],cohorts=[2,4,8],cohort_rule='Adjacent distinctTRAIN promptIDs groupedB; allrequests atsame decodeindex ready at time0.32prompts×16steps, no real arrivals or SLO.',families='Allfrozen cyclic/disjoint equalwidth1..16 plusbalanced/integer rowprofiles perphase; staticbestpair selected bymean cohort necessarybound, not queryadaptiveaddresses.',scheduler='Existing singleinput-context wholejob Graph ownership engine; lightweightadapter merelyreplicates independentrequest DAGs. Everyrequest retains ownGUinput/upload andeachprivateDown. GF/DF ordinaryphasepriority, same for allmaps.',forbidden='No PB/weight/inputreuse acrossjobs, multicastdedup, newMAC, independentplane context, copiedweights, changedroutes or futuretokens.',scope='Bounded feasibility/prior screen, two layer0cases only; no fit, heldout, modelinference or serving/queueSLO claim.'))
 print('FROZEN',sha(OUT/'protocol.frozen.json'))
def run():
 proto=json.loads((OUT/'protocol.frozen.json').read_text());assert not (OUT/'report.json').exists()
 for p,h in proto['inputs_sha256'].items():assert sha(p)==h,p
 start=time.monotonic();generated=OUT/'cohort_adapter.cpp';generated.write_text(CORE.read_text()+'\n'+ADAPTER.read_text());binary=OUT/'cohort.so';subprocess.run(['g++','-O3','-std=c++17','-shared','-fPIC',str(generated),'-o',str(binary)],check=True);lib=ctypes.CDLL(str(binary));fn=lib.eval_ready_cohorts;p32=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS');p64=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS');fn.argtypes=[p32,p32,p32,p32,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,p64,p64,p64,ctypes.c_void_p,p64];fn.restype=None
 records=[];arrays={};tracechecks=[]
 for model,copies,path in CASES:
  campaign=BASE/f'ordinary-read-compiler/full-campaign-v{5 if copies==1 else 6}-W8-{ "training" if copies==1 else "input4"}/summary.json';rec=next(x for x in json.loads(campaign.read_text())['records'] if x['model']==model and x['layer']==0 and x['command_ns']==250);D,I,E,K,L,SI=rec['shape'];s=b.Service(**rec['service']);z=np.load(path);strong=z['rows'];shared=strong[64]
  with np.load(BASE/f'w8-format-training/{model}/decode.routes.npz') as f:assert f['split'].item()=='train' and f['completed'].all();routes=f['routes'][:,:,0].copy();assert all(str(x).startswith('train_') for x in f['prompt_ids'])
  assert np.array_equal(routes.reshape(512,K),z['routes'])
  candidates=[]
  for phase in [0,1]:
   cc=[]
   for item in rec['fitting'][phase]:
    base=np.asarray(item['base_rows']);width=int(np.count_nonzero(base));labels=np.asarray(item['labels']);shifts=labels*width if item['family']=='disjoint' else labels;rr=np.stack([np.roll(base,int(k)) for k in shifts]);assert rr.shape==(64,16);cc.append(dict(label=item['label'],width=width,family=item['family'],rows=rr))
   assert set(range(1,17))<={c['width'] for c in cc};candidates.append(cc)
  profiles=[]
  for phase in [0,1]:
   arr=[]
   for candidate in candidates[phase]:
    rr=strong.copy();rr[:64,phase]=candidate['rows'];plan=b.lower_rows(rr[:,0],rr[:,1],D,[I]*64+[SI],s);table=b.occupancy_table(plan,s);arr.append(table['chip_ns'][:,phase])
   profiles.append(arr)
  def cohort_data(B):return routes.reshape(32//B,B,16,K).transpose(0,2,1,3).reshape(-1,B,K)
  def plan_for(rows):return b.lower_rows(rows[:,0],rows[:,1],D,[I]*64+[SI],s)
  def replay(plan,rr,downfirst):
   cfg=dict(rec['service']);cfg['down_first']=downfirst;service=b.Service(**cfg);rr=np.ascontiguousarray(rr,np.int32);v=np.zeros(len(rr),np.int64);lb=v.copy();trace=np.zeros((rr.shape[1]*1400,8),np.int64);n=np.zeros(1,np.int64);fn(plan['offsets'],plan['jobs'],plan['input_columns'],rr,64,len(rr),rr.shape[1],K,service.array(),v,lb,ctypes.c_void_p(trace.ctypes.data),n);return v,lb,trace[:n[0]]
  # B1 adapter regression must match the ordinary frozen baseline exactly.
  v,lb,tr=replay(plan_for(strong),routes.reshape(512,1,K),0);assert np.array_equal(v,z['cost']);arrays[model+'.single_native']=v
  for B in [2,4,8]:
   rr=cohort_data(B);ng=len(rr);sharedload=(profiles[0][0][64]+profiles[1][0][64])*B;allpairs=[]
   for gi,G in enumerate(profiles[0]):
    g=G[rr].sum((1,2))
    for di,Q in enumerate(profiles[1]):
     work=(g+Q[rr].sum((1,2))+sharedload).max(1)+s.host_submit_ns;allpairs.append((float(work.mean()),gi,di,work))
   best=min(allpairs,key=lambda x:x[0]);group=min((x for x in allpairs if candidates[0][x[1]]['family'] in ['cyclic','disjoint'] and candidates[1][x[2]]['family'] in ['cyclic','disjoint']),key=lambda x:x[0]);small=min((x for x in allpairs if candidates[0][x[1]]['width']<=4 and candidates[1][x[2]]['width']<=4),key=lambda x:x[0]);maps={'strong':strong.copy()}
   labels={}
   for name,item in [('best_simple_width',group),('best_phase_profile',best),('gsmall_le4',small)]:
    r=strong.copy();r[:64,0]=candidates[0][item[1]]['rows'];r[:64,1]=candidates[1][item[2]]['rows'];maps[name]=r;labels[name]=[candidates[0][item[1]]['label'],candidates[1][item[2]]['label']]
   serial=z['cost'].reshape(32//B,B,16).transpose(0,2,1).reshape(-1,B).sum(1);useful_per_request=K*3*D*I+3*D*SI;globalLB=s.host_submit_ns+B*useful_per_request/65536/16*28000
   results=[]
   for name,r in maps.items():
    plan=plan_for(r);table=b.occupancy_table(plan,s);selected=np.concatenate([rr,np.full((ng,B,1),64,np.int32)],axis=2);chip=table['chip_ns'][selected].sum((1,2,3));commandcounts=table['command_counts'][selected].sum((1,2,3,4));commandLB=commandcounts*250+s.host_submit_ns;channel=table['channel_ns'][selected].sum((1,2,3)).max(1);pcieout=table['pcie_ns'][selected].sum((1,2,3));pcieinput=B*(D+K*I+SI)/8;host=B*(K+2)*2000;necessary=np.maximum.reduce([chip.max(1)+2000,commandLB,channel,pcieout+pcieinput,np.full(ng,host)]);policies=[]
    for df in [0,1]:
     costs,bounds,trace=replay(plan,rr,df);assert np.all(costs>=necessary) and np.all(bounds>=necessary);key=f'{model}.B{B}.{name}.DF{df}';arrays[key+'.cost']=costs;arrays[key+'.LB']=bounds
     if name=='strong' and B==2 and df==0:
      # Every native job has exactly one acquire/release interval on one chip.
      intervals={c:[] for c in range(16)}
      for job in set(trace[trace[:,7]>=0,7]):
       jt=trace[trace[:,7]==job];cmd=jt[jt[:,0]==1];out=jt[(jt[:,0]>=3)&(jt[:,0]<11)&(jt[:,1]>0)];assert len(cmd)==1;chipid=int(cmd[0,6]);intervals[chipid].append((int(cmd[0,2]),int(jt[jt[:,0]>=3,3].max())))
      # Exact Python independent merged graph check on first ready cohort.
      merged=b.ReferenceGraph(b.Service(**rec['service']));joboffset=0
      for req in range(B):
       sub=b.ReferenceGraph.build(plan,np.r_[rr[0,req],64],merged.service);offset=len(merged.nodes)
       for node in sub.nodes:
        n=dict(node);n['deps']=[d+offset for d in n['deps']];n['job']=n['job']+joboffset if n['job']>=0 else -1;n['expert']=n['expert']+req*(K+1) if n['expert']>=0 else -1;merged.nodes.append(n)
       joboffset+=sub.job_count
      direct=merged.run();assert direct['cost']==int(costs[0]);tracechecks.append(dict(model=model,batch=B,cost_ns=int(costs[0]),python_CPP_exact=True,nodes=len(trace)));arrays[key+'.trace0']=trace;arrays[key+'.pythontrace0']=direct['trace']
     policies.append(dict(down_first=df,mean_cohort_ns=float(costs.mean()),p99_cohort_ns=float(np.percentile(costs,99)),mean_native_LB_ns=float(bounds.mean())))
    bestpolicy=min(policies,key=lambda x:x['mean_cohort_ns']);results.append(dict(name=name,phase_labels=labels.get(name,['frozen_strong','frozen_strong']),mean_necessary_LB_ns=float(necessary.mean()),mean_commands=float(commandcounts.mean()),mean_command_resource_ns=float(commandLB.mean()),mean_input_output_channel_resource_ns=float(channel.mean()),mean_PCIe_resource_ns=float((pcieout+pcieinput).mean()),mean_owned_chip_work_ns=float(chip.max(1).mean()),policies=policies,best_policy=bestpolicy,remaining_any_layout_schedule_gain_ceiling_pct=float(100*(1-globalLB/bestpolicy['mean_cohort_ns']))))
   records.append(dict(model=model,input_channel_copies=copies,layer=0,batch=B,cohorts=ng,requests=ng*B,synthetic_all_ready=True,serial_strong_mean_cohort_ns=float(serial.mean()),universal_useful_NAND_work_LB_ns=float(globalLB),no_crossrequest_weight_sensing_saved=True,maps=results));print(model,B,[(r['name'],r['best_policy']['mean_cohort_ns'],r['remaining_any_layout_schedule_gain_ceiling_pct']) for r in results],flush=True)
 for p,h in proto['inputs_sha256'].items():assert sha(p)==h,p
 np.savez_compressed(OUT/'arrays.npz',**arrays);report=dict(status='PASS bounded feasibility',source_sha256=sha(SELF),protocol_sha256=sha(OUT/'protocol.frozen.json'),native_core_sha256=sha(CORE),adapter_sha256=sha(ADAPTER),records=records,independent_python_CPP_checks=tracechecks,seconds=time.monotonic()-start,scope='Trace-derived simultaneous ready cohorts fromsame TRAIN decode index/differentprompts, not realarrivals orSLO. Sameonecopy layouts withfullweights/fullinputperjob. Width/rowprofile controls and ordinaryGF/DF scheduling only; no newphysical mechanism identified.');dump(OUT/'report.json',report)
def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run']);a=p.parse_args();freeze() if a.action=='freeze' else run()
if __name__=='__main__':main()
