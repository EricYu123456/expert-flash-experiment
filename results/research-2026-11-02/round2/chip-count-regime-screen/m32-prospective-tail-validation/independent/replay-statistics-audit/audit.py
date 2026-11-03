#!/usr/bin/env python3
"""Independent fixed-plan prospective cost and cluster-statistics audit; no fit."""
from pathlib import Path
from datetime import datetime,timezone
from dataclasses import replace
import argparse,ctypes,hashlib,importlib.util,json,math,shutil,subprocess,sys
import numpy as np
OUT=Path(__file__).resolve().parent;BASE=OUT.parents[1];ROOT=Path('/home/ericyu/expert_flash_experiment');R=ROOT/'results/research-2026-11-02/round2';PROD=BASE/'replay-adapter-v2/prospective'
NAMES=('free','common_cyclic','phase_independent');FANOUT=(1,4)
ORDER=('chips','channels','planes','page_bytes','block_wordlines','input_bytes','output_bytes','ordinary_read_ns','command_ns','host_submit_ns','pcie_bytes_per_second','channel_bytes_per_second','activation_ns','combine_ns','output_fringe','down_first','input_buffer_bytes')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def module(p,name):
 sp=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m

def freeze():
 assert not (OUT/'protocol.frozen.json').exists();snap=OUT/'source-snapshot';snap.mkdir()
 deps={}
 for key,f in [('wave_cpp',R/'wave-drain/snapshots/research_20261103_wave_drain_native.cpp'),('wave_python',R/'wave-drain/snapshots/research_20261103_wave_drain.py'),('backend',BASE/'replay-adapter-v2/source-snapshot/7c61f09756e24f11-7c61f09756e24f11-research_20261103_ordinary_read_backend.py'),('compiler',BASE/'replay-adapter-v2/source-snapshot/23cfb0c655567802-research_20261102_whole_row_compiler.py')]:
  dest=snap/({'backend':'research_20261103_ordinary_read_backend.py','compiler':'research_20261102_whole_row_compiler.py'}.get(key,f.name));shutil.copy2(f,dest);deps[key]={'path':str(dest),'sha256':sha(dest),'origin':str(f)}
 inputs=[BASE/'protocol.frozen.json',BASE/'root_release.json',PROD/'completion.json',PROD/'binding.frozen.json',PROD/'costs.npz',PROD/'L0.routes.npz',PROD/'replay_report.json',BASE/'analysis/report.json',BASE/'data-capture/granite4tiny/decode.routes.npz',BASE/'data-capture/manifest.json']+[BASE/f'plans/{n}.frozen.npz' for n in NAMES]
 dump(OUT/'protocol.frozen.json',{'utc':datetime.now(timezone.utc).isoformat(),'source_sha256':sha(__file__),'inputs':{str(p):sha(p) for p in inputs},'dependencies':deps,'rules':['Verify entire producer completion and binding hashes and unchanged three pre-data fixed static maps; exact route/prompt/source/token order and shared inclusion.','All3072 costs and3072lowerbounds compared with independently maintained sealed wave-drain C++ in postcompute mode, input fanout1/4; no perwave experiment here.','18 samples (3maps x2fanouts xq0/255/511) compare all trace fields with frozen PythonReferenceGraph and separately implemented scalar heap graph; audit dependencies/resource/chipowner/fullGU barriers.','Independently recompute all report metrics,32 per-prompt and two source summaries; explicit paired improve/tie/regress and negative prompt counts.','Reproduce fixed PCG64 seed20261115,2000 draws of32 prompt clusters retaining16queries; integer-weighted means and direct sorted linear interpolation forP99/CI, not512IID. Primaryfreevsphase_independent only; commoncontrol descriptive.','No changing plans/priority/analysis/gate/selection or reading old test; prospectiveL0 service-model audit, notnative/quality/fullmodelproof.']})
 print(json.dumps({'protocol_sha256':sha(OUT/'protocol.frozen.json'),'source_sha256':sha(__file__)}))

def quantile(x,p,axis=None):
 a=np.asarray(x)
 if axis is None:a=np.sort(a.reshape(-1));n=len(a)
 else:a=np.sort(a,axis=axis);n=a.shape[axis]
 h=(n-1)*p;i=math.floor(h);f=h-i
 if axis is None:lo,hi=a[i],a[min(i+1,n-1)]
 else:lo=np.take(a,i,axis=axis);hi=np.take(a,min(i+1,n-1),axis=axis)
 return np.asarray(lo,dtype=np.float64)+(np.asarray(hi,dtype=np.float64)-lo)*f

def stat(x):return {'mean_ns':float(np.sum(x,dtype=np.int64)/np.size(x)),'p99_ns':float(quantile(x,.99))}
def ci(x):return [float(quantile(x,.025)),float(quantile(x,.975))]
def compare(actual,expected,path='root',diffs=None):
 if diffs is None:diffs=[]
 if isinstance(expected,dict):
  assert set(actual)==set(expected),(path,set(actual)^set(expected))
  for k in expected:compare(actual[k],expected[k],path+'.'+k,diffs)
 elif isinstance(expected,list):
  assert len(actual)==len(expected),(path,len(actual),len(expected))
  for k,(a,e) in enumerate(zip(actual,expected)):compare(a,e,f'{path}[{k}]',diffs)
 elif isinstance(expected,float):
  error=abs(float(actual)-expected);assert error<=1e-8,(path,actual,expected,error);diffs.append(error)
 else:assert actual==expected,(path,actual,expected)
 return diffs

class IndependentNative:
 def __init__(self,source):
  build=OUT/'build';build.mkdir();self.path=build/'independent_wave_postcompute.so'
  argv=['c++','-O3','-std=c++17','-shared','-fPIC',str(source),'-o',str(self.path)];subprocess.run(argv,check=True)
  self.lib=ctypes.CDLL(str(self.path));self.fn=self.lib.eval_wave_drain
  p32=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS');p64=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS')
  self.fn.argtypes=[p32,p32,p32,p32,*([ctypes.c_int]*5),p64,p64,p64,ctypes.c_void_p,p64];self.fn.restype=None
  dump(OUT/'build_receipt.json',{'source_sha256':sha(source),'argv':argv,'binary_sha256':sha(self.path)})
 def run(self,plan,routes,service,fanout,trace=False):
  routes=np.ascontiguousarray(routes,np.int32);cfg=np.array([service[k] for k in ORDER]+[0,fanout],np.int64);cost=np.zeros(len(routes),np.int64);lb=cost.copy();log=np.zeros((10000 if trace else 1,8),np.int64);nlog=np.zeros(1,np.int64)
  self.fn(plan['offsets'],plan['jobs'],plan['input_columns'],routes,64,1,len(plan['jobs']),len(routes),6,cfg,cost,lb,ctypes.c_void_p(log.ctypes.data) if trace else None,nlog)
  return cost,lb,log[:nlog[0]].copy() if trace else None

def run():
 proto=json.loads((OUT/'protocol.frozen.json').read_text());assert sha(__file__)==proto['source_sha256']
 for f,h in proto['inputs'].items():assert sha(f)==h
 for dep in proto['dependencies'].values():assert sha(dep['path'])==dep['sha256']
 for rec in json.loads((PROD/'completion.json').read_text())['files']:assert sha(rec['path'])==rec['sha256']
 for rec in json.loads((PROD/'binding.frozen.json').read_text())['files']:assert sha(rec['path'])==rec['sha256']
 assert sha(PROD/'completion.json')=='ee5e2c92965d78bad5ea64699eb247c81e5b55fac8aa18b958a68a16c9f3788d'
 anchor=json.loads((BASE/'protocol.frozen.json').read_text());service=anchor['hardware_primary'];assert service['down_first']==0 and service['input_channel_copies']==1
 maps={n:load(BASE/f'plans/{n}.frozen.npz') for n in NAMES};fresh=load(BASE/'data-capture/granite4tiny/decode.routes.npz');costs=load(PROD/'costs.npz');l0=load(PROD/'L0.routes.npz');report=json.loads((BASE/'analysis/report.json').read_text());producer=json.loads((PROD/'replay_report.json').read_text())
 routes=fresh['routes'][:,:,0,:].reshape(512,6);assert np.array_equal(routes,l0['routes'])
 assert np.array_equal(costs['prompt_ids'],fresh['prompt_ids']) and np.array_equal(costs['source_ids'],fresh['source_keys']) and np.array_equal(costs['dataset_item_ids'],fresh['source_ids'])
 assert np.array_equal(costs['token_positions'],fresh['token_positions']) and np.array_equal(costs['token_ids'],fresh['token_ids'])
 sys.path.insert(0,str(OUT/'source-snapshot'));b=module(proto['dependencies']['backend']['path'],'independent_prospective_backend');wave=module(proto['dependencies']['wave_python']['path'],'independent_prospective_wave');native=IndependentNative(proto['dependencies']['wave_cpp']['path'])
 allcheck=[];traces=[];actual_arrays={};nodecount=0
 for n,plan in maps.items():
  assert sha(BASE/f'plans/{n}.frozen.npz')==anchor['plans'][n]['sha256'];assert 'routes' not in plan
  assert np.array_equal(plan['rows'][64],maps['free']['rows'][64]);assert np.array_equal(plan['total_rows'][:64],np.tile([1024,1536],(64,1))) and np.array_equal(plan['total_rows'][64],[2048,1536])
  assert np.array_equal(plan['input_columns'][:64],np.tile([1536,512],(64,1))) and np.array_equal(plan['input_columns'][64],[1536,1024])
  for fanout in FANOUT:
   key=f'input{fanout}.{n}';actual,lb,_=native.run(plan,routes,service,fanout)
   expected=costs[key+'.cost'].reshape(-1);saved_lb=costs[key+'.bound'].reshape(-1)
   assert np.array_equal(actual,expected) and np.array_equal(lb,saved_lb) and np.all(actual>=lb)
   actual_arrays[key+'.cost']=actual.reshape(32,16);actual_arrays[key+'.bound']=lb.reshape(32,16)
   allcheck.append({'plan':n,'input_copies':fanout,'costs_exact':512,'lowerbounds_exact':512,'native_shared_experts':1,'whole_expert_GU_barrier':True})
   srv=dict(service);srv['input_channel_copies']=fanout
   for q in (0,255,511):
    c,l,t=native.run(plan,routes[q:q+1],service,fanout,True);saved=costs[f'{key}.q{q}.trace'];assert np.array_equal(t,saved)
    ref=b.ReferenceGraph.build(plan,np.r_[routes[q],64],b.Service(**srv)).run();assert ref['cost']==int(c[0]) and np.array_equal(ref['trace'],saved)
    sg=wave.ScalarGraph.build(plan,list(map(int,routes[q]))+[64],srv,0,fanout);sc,st=sg.run();assert sc==int(c[0]) and np.array_equal(st,saved);checks=sg.audit(saved)
    traceitem=next(x for x in producer['trace_checks'] if x['plan']==n and x['input_copies']==fanout and x['query']==q);assert traceitem['cost_ns']==int(c[0]) and traceitem['nodes']==len(t)
    nodecount+=len(t);traces.append({'plan':n,'input_copies':fanout,'query':q,'cost_ns':int(c[0]),'all_three_trace_implementations_exact':True,**checks})
 ids=fresh['prompt_ids'].astype(str);sources=fresh['source_keys'].astype(str)
 draw=np.random.Generator(np.random.PCG64(20261115)).integers(0,32,size=(2000,32));weights=np.asarray([np.bincount(row,minlength=32) for row in draw],np.int64)
 recomputed={'status':'prospective_frozen_analysis','protocol_sha256':sha(BASE/'protocol.frozen.json'),'costs_archive_sha256':sha(PROD/'costs.npz'),'profiles':{}}
 negative=[]
 for fanout in FANOUT:
  prof='input'+str(fanout);values={n:actual_arrays[f'{prof}.{n}.cost'] for n in NAMES};baseline=values['phase_independent'];base_mean=weights@baseline.sum(1)/512;base_p99=quantile(baseline[draw].reshape(2000,512),.99,axis=1)
  result={'fixed_primary_baseline':'phase_independent','plans':{}}
  for n,v in values.items():
   diff=baseline-v;boot_mean=weights@v.sum(1)/512;boot_p99=quantile(v[draw].reshape(2000,512),.99,axis=1)
   pp=[]
   for i in range(32):pp.append({'prompt_id':str(ids[i]),'source_id':str(sources[i]),**stat(v[i]),'max_ns':int(v[i].max()),'paired_mean_gain_ns':float(diff[i].sum()/16),'paired_p99_gain_ns':float(quantile(baseline[i],.99)-quantile(v[i],.99))})
   result['plans'][n]={**stat(v),'mean_gain_pct':100*(1-v.sum()/baseline.sum()),'p99_gain_pct':100*(1-float(quantile(v,.99))/float(quantile(baseline,.99))),'improved_queries':int((diff>0).sum()),'tied_queries':int((diff==0).sum()),'regressed_queries':int((diff<0).sum()),'max_regression_ns':int(max(0,-diff.min())),'perprompt':pp,'by_source':{s:stat(v[sources==s]) for s in sorted(set(sources))},'descriptive_cluster_CI95':{'mean_gain_ns':ci(base_mean-boot_mean),'p99_gain_ns':ci(base_p99-boot_p99),'mean_gain_pct':ci(100*(1-boot_mean/base_mean)),'p99_gain_pct':ci(100*(1-boot_p99/base_p99))}}
   prompt_delta=v.sum(1)-baseline.sum(1)
   negative.append({'profile':prof,'plan':n,'prompt_mean_improve':int((prompt_delta<0).sum()),'prompt_mean_tie':int((prompt_delta==0).sum()),'prompt_mean_regress':int((prompt_delta>0).sum()),'all_prompt_mean_deltas_ns':(prompt_delta/16).tolist(),'by_source':{s:{'query_improve':int((diff[sources==s]>0).sum()),'query_tie':int((diff[sources==s]==0).sum()),'query_regress':int((diff[sources==s]<0).sum()),'prompt_mean_improve':int((prompt_delta[sources==s]<0).sum()),'prompt_mean_tie':int((prompt_delta[sources==s]==0).sum()),'prompt_mean_regress':int((prompt_delta[sources==s]>0).sum()),'mean_gain_pct':100*(1-v[sources==s].sum()/baseline[sources==s].sum()),'p99_gain_pct':100*(1-float(quantile(v[sources==s],.99))/float(quantile(baseline[sources==s],.99)))} for s in sorted(set(sources))}})
  recomputed['profiles'][prof]=result
 diffs=compare(recomputed,report)
 for f,h in proto['inputs'].items():assert sha(f)==h
 dump(OUT/'statistics_recomputed.json',recomputed);dump(OUT/'negative_cases.json',negative);np.savez_compressed(OUT/'independent_arrays.npz',**actual_arrays,cluster_draws=draw)
 primary=recomputed['profiles']['input1']['plans']['free'];sensitivity=recomputed['profiles']['input4']['plans']['free']
 audit={'status':'PASS_FIXED_PROSPECTIVE_COST_TRACE_AND_CLUSTER_STATISTICS','utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':sha(OUT/'protocol.frozen.json'),'native_cost_checks':allcheck,'independent_CPP_costs':3072,'independent_CPP_lowerbounds':3072,'sampled_Python_and_independent_scalar_traces':traces,'trace_nodes_compared':nodecount,'all_saved_statistics_reproduced':True,'numeric_comparison_max_abs_error':max(diffs,default=0),'bootstrap':{'draws':2000,'prompt_clusters':32,'queries_per_cluster':16,'seed':20261115,'rng':'PCG64','mean_method':'Integer cluster-sum weighted counts, not flattened bootstrap mean','quantile_method':'Independently sorted linear interpolation'},'primary_free_vs_predeclared_phase_independent':{k:primary[k] for k in ['mean_ns','p99_ns','mean_gain_pct','p99_gain_pct','improved_queries','tied_queries','regressed_queries','max_regression_ns','descriptive_cluster_CI95']},'input4_same_maps_sensitivity':{k:sensitivity[k] for k in ['mean_ns','p99_ns','mean_gain_pct','p99_gain_pct','improved_queries','tied_queries','regressed_queries','max_regression_ns','descriptive_cluster_CI95']},'interpretation':'Negative prospective L0 generalization for free map vs fixed phase-independent baseline. Do not replace baseline/select another map or novelty subset. TRAIN OR gate pass remains true but does not imply prospective success. No device, all-layer speed or paired full-model numerical proof.'}
 dump(OUT/'audit.json',audit)
 dump(OUT/'completion.json',{'status':'SEALED_PASS_NEGATIVE_RESULT_RETAINED','files':{str(f.relative_to(OUT)):sha(f) for f in sorted(OUT.rglob('*')) if f.is_file() and f.name!='completion.json' and '__pycache__' not in str(f)}})
 print(json.dumps({'status':audit['status'],'CPP_costs':3072,'trace_nodes':nodecount,'max_statistics_error':max(diffs),'primary_mean_gain_pct':primary['mean_gain_pct'],'primary_p99_gain_pct':primary['p99_gain_pct'],'audit_sha256':sha(OUT/'audit.json'),'completion_sha256':sha(OUT/'completion.json')}))
if __name__=='__main__':{'freeze':freeze,'run':run}[sys.argv[1]]()
