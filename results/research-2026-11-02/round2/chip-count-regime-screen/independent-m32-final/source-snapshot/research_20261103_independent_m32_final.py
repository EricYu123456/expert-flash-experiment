#!/usr/bin/env python3
"""Read sealed CP maps; independently audit bounded metadata and native samples."""
from pathlib import Path
from datetime import datetime,timezone
from dataclasses import replace
import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/research-2026-11-02/round2/chip-count-regime-screen'
SRC=BASE/'g4-M32-coupled-cp';OUT=BASE/'independent-m32-final'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def load(p):
 with np.load(p) as z:return {k:z[k] for k in z.files}
def up(n,d):return (int(n)+int(d)-1)//int(d)
def stats(a):return {'mean_ns':float(np.mean(a)),'p99_ns':float(np.percentile(a,99))}

def freeze_source():
 p=json.loads((OUT/'protocol.frozen.json').read_text());d=OUT/'source-snapshot';d.mkdir(exist_ok=False)
 source=d/Path(__file__).name;shutil.copy2(__file__,source)
 deps={}
 for x in p['inputs']:
  for suffix in ['ordinary_read_backend.py','whole_row_compiler.py','ordinary_read_native.cpp']:
   if x['path'].endswith(suffix):
    q=Path(x['snapshot']);assert sha(q)==x['sha256']
    name={'ordinary_read_backend.py':'research_20261103_ordinary_read_backend.py',
      'whole_row_compiler.py':'research_20261102_whole_row_compiler.py','ordinary_read_native.cpp':'ordinary_read_native.cpp'}[suffix]
    dest=d/name;shutil.copy2(q,dest);deps[name]={'path':str(dest),'sha256':sha(dest)}
 dump(OUT/'implementation.frozen.json',{'status':'frozen_before_final_results_binding','utc':datetime.now(timezone.utc).isoformat(),
  'producer_source':str(source),'producer_source_sha256':sha(source),'dependencies':deps,'protocol_sha256':sha(OUT/'protocol.frozen.json')})
 print(json.dumps({'implementation_sha256':sha(OUT/'implementation.frozen.json'),'source_sha256':sha(__file__)}))

def bind():
 assert (SRC/'completion.json').exists(),'Wait for producer completion/seal notification.'
 files=[SRC/x for x in ['protocol.frozen.json','fit_report.json','native_protocol.frozen.json','native_report.json',
  'candidates.frozen.npz','native_arrays.npz','input4_selection.frozen.json','completion.json']]
 files+=sorted((SRC/'native').glob('*.plan.npz'))
 assert all(p.is_file() for p in files)
 dump(OUT/'results_binding.frozen.json',{'utc':datetime.now(timezone.utc).isoformat(),'status':'producer_complete_frozen_before_audit',
  'implementation_sha256':sha(OUT/'implementation.frozen.json'),'files':{str(f):sha(f) for f in files}})
 print(json.dumps({'binding_sha256':sha(OUT/'results_binding.frozen.json'),'bound_files':len(files)}))

def run():
 protocol=json.loads((OUT/'protocol.frozen.json').read_text());impl=json.loads((OUT/'implementation.frozen.json').read_text())
 binding=json.loads((OUT/'results_binding.frozen.json').read_text())
 assert sha(__file__)==impl['producer_source_sha256']
 for f,h in binding['files'].items():assert sha(f)==h
 assert sha(SRC/'protocol.frozen.json')==protocol['producer_protocol_sha256']
 for x in impl['dependencies'].values():assert sha(x['path'])==x['sha256']
 sys.path.insert(0,str(OUT/'source-snapshot'));f=impl['dependencies']['research_20261103_ordinary_read_backend.py']['path']
 spec=importlib.util.spec_from_file_location('m32_final_backend',f);b=importlib.util.module_from_spec(spec)
 sys.modules[spec.name]=b;spec.loader.exec_module(b)
 pp=json.loads((SRC/'protocol.frozen.json').read_text());fit=json.loads((SRC/'fit_report.json').read_text())
 nr=json.loads((SRC/'native_report.json').read_text());sel=json.loads((SRC/'input4_selection.frozen.json').read_text())
 z=load(SRC/'candidates.frozen.npz');na=load(SRC/'native_arrays.npz')
 original=load(BASE/'stage-b/granite4tiny.M32.minimum_depth.plan.npz');old=load(BASE/'stage-b/cost_arrays.npz')
 ids=[c['id'] for c in fit['candidates']];assert len(ids)<=8
 routes=z['routes'];assert np.array_equal(routes,original['routes']) and routes.shape==(512,6)
 base=b.Service(**pp['service']);assert base.chips==32 and base.output_fringe==1
 coefficients={1:(29102,28762,-340),4:(31406,29530,-1876)}
 result=[];traces=[];peak_witnesses=[];saved={};all_stats=[]

 def manual_table(plan,s):
  a=np.zeros((65,2,32),np.int64);counters={'jobs':0,'pages':0,'waves':0,'useful_bytes':0,'valid_rows':0}
  for e in range(65):
   for phase in range(2):
    C=1536 if phase==0 else (512 if e<64 else 1024)
    R=(1024 if phase==0 else 1536) if e<64 else (2048 if phase==0 else 1536)
    begin,end=map(int,plan['offsets'][e,phase]);covered=np.zeros(R,np.uint8);cursor=0
    for j in range(begin,end):
     chip,n,c,p,w=map(int,plan['jobs'][j]);assert c==C
     assert p==up(n*C,16384) and w==up(p,4) and w<=192
     emit=p*16384//C;duration=s.command_ns+up(C*s.input_bytes*s.input_channel_copies*10**9,s.channel_bytes_per_second)+w*s.ordinary_read_ns+up(emit*s.output_bytes*10**9,s.channel_bytes_per_second)
     a[e,phase,chip]+=duration;covered[cursor:cursor+n]+=1;cursor+=n
     for k,v in [('jobs',1),('pages',p),('waves',w),('useful_bytes',n*C),('valid_rows',n)]:counters[k]+=v
    assert cursor==R and np.all(covered==1)
  return a,counters

 for ident in ids:
  rows=z[ident+'.rows'];X=z[ident+'.X'];O=z[ident+'.O'];assert X.dtype==O.dtype==np.bool_
  assert np.all(X.sum(1)==5) and np.all(O.sum(1)==7) and not np.any(X&O)
  A=rows[:64,0]>0;D=rows[:64,1]>0
  assert np.array_equal(A,~O) and np.array_equal(D,X|O)
  assert np.all(np.sort(rows[:64,0],axis=1)==[0]*7+[40]+[41]*24)
  assert np.all(np.sort(rows[:64,1],axis=1)==[0]*20+[128]*12)
  assert np.array_equal(rows[64],original['rows'][64])
  plan=b.lower_rows(rows[:,0],rows[:,1],1536,[512]*64+[1024],base)
  stored=load(SRC/'native'/(ident+'.plan.npz'))
  for key in ['jobs','offsets','input_columns','total_rows']:assert np.array_equal(plan[key],stored[key])
  assert np.array_equal(stored['rows'],rows) and np.array_equal(stored['routes'],routes)
  addr=stored['addresses'];assert addr.shape==(2424,6)
  waves_per_chip=np.zeros(32,np.int64)
  for chip in range(32):
   inds=np.flatnonzero(plan['jobs'][:,0]==chip);intervals=[]
   for j in inds:
    c,block,wl,res,pages,plane=map(int,addr[j]);job=plan['jobs'][j]
    assert c==chip and block==plane==0 and res==int(job[4]) and pages==int(job[3])
    assert wl>=0 and wl+res<=192;intervals.append((wl,wl+res));waves_per_chip[chip]+=res
   intervals.sort();assert all(t[1]<=u[0] for t,u in zip(intervals,intervals[1:]))
  assert np.all(waves_per_chip<=131)
  for copies in [1,4]:
   s=replace(base,input_channel_copies=copies);table,inv=manual_table(plan,s)
   assert np.array_equal(table,b.occupancy_table(plan,s)['chip_ns'])
   intercept,kx,ko=coefficients[copies];formula=intercept+kx*X+ko*O
   assert np.array_equal(table[:64].sum(1),formula)
   allquery=table[routes].sum((1,2))+table[64].sum(0)
   assert np.array_equal(allquery,formula[routes].sum(1)+table[64].sum(0))
   if copies==1:
    assert np.array_equal(allquery.max(1),z[ident+'.work'])
    candidate=next(c for c in fit['candidates'] if c['id']==ident)
    assert int(allquery.max(1).sum())==candidate['score']
    for k,v in stats(allquery.max(1)).items():assert abs(candidate['work'][k]-v)<1e-6
    nand=(~O[routes]).sum(1)+(X[routes]|O[routes]).sum(1)+z['sharedwaves']
    peak=int(nand.max())
    if 'peak' in candidate['stage']:
     assert peak<=10;peak_witnesses.append({'id':ident,'max_read_waves':peak})
    saved[ident+'.all_query_chip_owned_ns']=allquery
    for priority in [0,1]:
     key=ident+'.'+('DF' if priority else 'GF');cost=na[key+'.cost']
     assert len(cost)==512 and np.all(cost>=na[key+'.bound'])
     record=next(r for r in nr['records'] if r['id']==ident and r['input_copies']==1 and r['priority']==('DF' if priority else 'GF'))
     for k,v in stats(cost).items():assert abs(record['cost'][k]-v)<1e-6
     all_stats.append(record)
     for q in [0,255,511]:
      got=b.ReferenceGraph.build(plan,np.r_[routes[q],64],replace(s,down_first=priority)).run()
      assert got['cost']==int(cost[q])
      if q==0:assert np.array_equal(got['trace'],na[key+'.q0trace'])
      traces.append({'map':ident,'copies':1,'priority':priority,'q':q,'cost_ns':got['cost'],'nodes':len(got['trace']),'q0_all_native_trace_fields_exact':q==0})
   assert inv=={'jobs':2424,'pages':9764,'waves':2441,'useful_bytes':int(148.5*2**20),'valid_rows':64*2560+3584}
  result.append({'id':ident,'all512_chip_services_exact_both_fanouts':True,'shared_unchanged':True,'complete_rows_one_copy':True,
   'inventory':inv,'chip_reserved_waves':waves_per_chip.tolist(),'single_block_addresses_valid':True,'TRAIN_max_NANDwaves':peak})

 best=min(all_stats,key=lambda r:r['cost']['mean_ns']);assert best==nr['best_coupled_input1']
 sourcekeys=[f'granite4tiny.M32.minimum_depth.{v}.cost' for v in ['GF','DF']]
 sourcekey=min(sourcekeys,key=lambda k:float(old[k].mean()));sourcestat=stats(old[sourcekey])
 assert sourcekey==nr['source_priority'] and sourcestat==nr['source_uncoupled']
 chosen='uncoupled_source' if sourcestat['mean_ns']<best['cost']['mean_ns'] else best['id']
 assert chosen==sel['winner']==nr['final_selected_id']
 rows=original['rows'] if chosen=='uncoupled_source' else z[chosen+'.rows']
 assert hashlib.sha256(rows.tobytes()).hexdigest()==sel['rows_sha256']
 plan=b.lower_rows(rows[:,0],rows[:,1],1536,[512]*64+[1024],base)
 for priority in [0,1]:
  s=replace(base,input_channel_copies=4,down_first=priority);key='selected_input4.'+('DF' if priority else 'GF')
  rec=next(r for r in nr['records'] if r['input_copies']==4 and r['priority']==('DF' if priority else 'GF'))
  assert rec['id']==chosen and rec['cost']==stats(na[key+'.cost'])
  for q in [0,255,511]:
   got=b.ReferenceGraph.build(plan,np.r_[routes[q],64],s).run();assert got['cost']==int(na[key+'.cost'][q])
   if q==0:assert np.array_equal(got['trace'],na[key+'.q0trace'])
   traces.append({'map':chosen,'copies':4,'priority':priority,'q':q,'cost_ns':got['cost'],'nodes':len(got['trace']),'q0_all_native_trace_fields_exact':q==0})
 assert len(traces)<=54
 for f,h in binding['files'].items():assert sha(f)==h
 saved['routes']=routes;np.savez_compressed(OUT/'audit_arrays.npz',**saved)
 report={'status':'PASS_RETAINED_M32_CP_GEOMETRY_WORK_ADDRESS_AND_NATIVE_SAMPLES','utc':datetime.now(timezone.utc).isoformat(),
  'protocol_sha256':sha(OUT/'protocol.frozen.json'),'binding_sha256':sha(OUT/'results_binding.frozen.json'),'maps':result,
  'reference_native_checks':traces,'new_cpp_costs':0,'input4_selected_id':chosen,'peak10_retained_witnesses':peak_witnesses,
  'solver_stages_as_reported':fit['stages'],'scope':['FeasibilityOPTIMAL provesSAT only for the fixed peak10 constraints, not native latency optimality.',
    'Mean status/bound concerns fixed TRAIN owned-work surrogate, not native makespan.','All geometry/core/command/fringe assumptions are unchanged32chip scale-model conditions.',
    'All512query owned work checked; native costs sampledq0/255/511 only, not a new full campaign.','Physical/logical metadata proof, not raw hardware numeric execution.','StandardCP/hypergraph strongcontrol and TRAIN-only fit; no first/new algorithm or generalIFPbreakthrough.']}
 dump(OUT/'audit.json',report)
 files=[f for f in OUT.rglob('*') if f.is_file() and f.name!='completion.json']
 dump(OUT/'completion.json',{'status':'SEALED_PASS','utc':datetime.now(timezone.utc).isoformat(),'files':{str(f.relative_to(OUT)):sha(f) for f in sorted(files)}})
 print(json.dumps({'status':report['status'],'maps':len(result),'reference_graphs':len(traces),'audit_sha256':sha(OUT/'audit.json'),'completion_sha256':sha(OUT/'completion.json')},indent=2))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['freeze-source','bind','run']);a=ap.parse_args()
 {'freeze-source':freeze_source,'bind':bind,'run':run}[a.mode]()
