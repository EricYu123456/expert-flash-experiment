#!/usr/bin/env python3
"""Independent exact-orbit/row/address/native-sample audit, no CPP or fitting."""
from dataclasses import replace
from datetime import datetime,timezone
from pathlib import Path
import argparse,hashlib,importlib.util,json,shutil,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
B=ROOT/'results/research-2026-11-02/round2/chip-count-regime-screen'
SRC=B/'g4-M32-cyclic-rotation-cp';OUT=B/'independent-m32-rotation-final'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def load(p):
 with np.load(p) as z:return {k:z[k] for k in z.files}
def stats(x):return {'mean_ns':float(np.mean(x)),'p99_ns':float(np.percentile(x,99))}
def up(x,d):return (int(x)+int(d)-1)//int(d)

def freeze():
 p=json.loads((OUT/'protocol.frozen.json').read_text());d=OUT/'source-snapshot';d.mkdir(exist_ok=False)
 shutil.copy2(__file__,d/Path(__file__).name);deps={}
 for x in p['inputs']:
  f=Path(x.get('snapshot',x['path']));assert sha(f)==x['sha256']
  if x['path'].endswith('ordinary_read_backend.py'):
   dest=d/'research_20261103_ordinary_read_backend.py';shutil.copy2(f,dest);deps['backend']={'path':str(dest),'sha256':sha(dest)}
 c=B/'independent-m32-final/source-snapshot/research_20261102_whole_row_compiler.py'
 dest=d/c.name;shutil.copy2(c,dest);deps['compiler']={'path':str(dest),'sha256':sha(dest)}
 dump(OUT/'implementation.frozen.json',{'utc':datetime.now(timezone.utc).isoformat(),'source_sha256':sha(__file__),
  'snapshot':str(d/Path(__file__).name),'dependencies':deps,'all_frozen_input_hashes_checked':True,
  'protocol_sha256':sha(OUT/'protocol.frozen.json')})
 print(json.dumps({'implementation_sha256':sha(OUT/'implementation.frozen.json'),'source_sha256':sha(__file__)}))

def bind(seal_path):
 seal=Path(seal_path).resolve();x=json.loads(seal.read_text());records=x['files']
 if isinstance(records,dict):records=[{'path':str(seal.parent/k),'sha256':v} for k,v in records.items()]
 files={}
 for r in records:
  f=Path(r['path']);assert sha(f)==r['sha256'];files[str(f)]=r['sha256']
 for name in ['protocol.frozen.json','fit_report.json','native_report.json','native_protocol.frozen.json','candidates.frozen.npz','native_arrays.npz']:
  f=SRC/name;files[str(f)]=sha(f)
 for f in (SRC/'native').glob('*.plan.npz'):files[str(f)]=sha(f)
 files[str(seal)]=sha(seal)
 dump(OUT/'results_binding.frozen.json',{'utc':datetime.now(timezone.utc).isoformat(),'files':files,
  'implementation_sha256':sha(OUT/'implementation.frozen.json'),'producer_seal':str(seal)})
 print(json.dumps({'binding_sha256':sha(OUT/'results_binding.frozen.json'),'files_bound':len(files)}))

def run():
 p=json.loads((OUT/'protocol.frozen.json').read_text());im=json.loads((OUT/'implementation.frozen.json').read_text())
 binding=json.loads((OUT/'results_binding.frozen.json').read_text());assert sha(__file__)==im['source_sha256']
 for f,h in binding['files'].items():assert sha(f)==h
 for x in p['inputs']:
  assert sha(x.get('snapshot',x['path']))==x['sha256']
 assert sha(SRC/'protocol.frozen.json')==p['producer_protocol_sha256']
 for d in im['dependencies'].values():assert sha(d['path'])==d['sha256']
 sys.path.insert(0,str(OUT/'source-snapshot'));f=im['dependencies']['backend']['path']
 sp=importlib.util.spec_from_file_location('rotation_final_backend',f);b=importlib.util.module_from_spec(sp);sys.modules[sp.name]=b;sp.loader.exec_module(b)
 pp=json.loads((SRC/'protocol.frozen.json').read_text());inp=load(SRC/'inputs.frozen.npz')
 fit=json.loads((SRC/'fit_report.json').read_text());nr=json.loads((SRC/'native_report.json').read_text())
 z=load(SRC/'candidates.frozen.npz');na=load(SRC/'native_arrays.npz')
 original=load(B/'stage-b/granite4tiny.M32.minimum_depth.plan.npz');old=load(B/'stage-b/cost_arrays.npz')
 rotations=np.asarray([np.roll(original['rows'][0],k,axis=1) for k in range(32)])
 assert np.array_equal(rotations,inp['rotations']) and len(np.unique(rotations.reshape(32,-1),axis=0))==32
 source_labels=[]
 for e,r in enumerate(original['rows'][:64]):
  found=[k for k in range(32) if np.array_equal(rotations[k],r)];assert len(found)==1 and found[0]==e%32;source_labels+=found
 assert np.array_equal(source_labels,inp['labels'])
 assert np.array_equal(inp['shared'],original['rows'][64]) and np.array_equal(z['routes'],original['routes'])
 routes=z['routes'];ids=[c['id'] for c in fit['candidates']];assert len(ids)<=8
 base=b.Service(**pp['service']);assert base.chips==32 and base.output_fringe==1
 results=[];traces=[];records=[];saved={};source_bridge=[]

 for ident in ids:
  labels=z[ident+'.labels'];rows=z[ident+'.rows'];assert labels.shape==(64,) and np.all((labels>=0)&(labels<32))
  onehot=np.eye(32,dtype=np.uint8)[labels];assert onehot.shape==(64,32) and np.all(onehot.sum(1)==1)
  expected=np.concatenate((rotations[labels],inp['shared'][None]),axis=0);assert np.array_equal(rows,expected)
  # Exact rows check includes the40-row band's rotated chip, not just active masks.
  assert np.array_equal(np.argmax(rows[:64,0]==40,axis=1),(24+labels)%32)
  X=(rows[:64,0]>0)&(rows[:64,1]>0);O=rows[:64,0]==0
  assert np.all(X.sum(1)==5) and np.all(O.sum(1)==7) and not np.any(X&O)
  assert np.all(np.sort(rows[:64,0],axis=1)==[0]*7+[40]+[41]*24)
  assert np.all(np.sort(rows[:64,1],axis=1)==[0]*20+[128]*12)
  plan=b.lower_rows(rows[:,0],rows[:,1],1536,[512]*64+[1024],base);stored=load(SRC/'native'/(ident+'.plan.npz'))
  for k in ['jobs','offsets','input_columns','total_rows']:assert np.array_equal(stored[k],plan[k])
  assert np.array_equal(stored['rows'],rows) and np.array_equal(stored['labels'],labels)
  addresses=stored['addresses'];assert addresses.shape==(2424,6);chipwaves=np.zeros(32,np.int64)
  for chip in range(32):
   intervals=[]
   for j in np.flatnonzero(plan['jobs'][:,0]==chip):
    c,block,wl,res,pg,plane=map(int,addresses[j]);job=plan['jobs'][j]
    assert c==chip and block==plane==0 and res==int(job[4]) and pg==int(job[3]) and 0<=wl<wl+res<=192
    intervals.append((wl,wl+res));chipwaves[chip]+=res
   intervals.sort();assert all(a[1]<=q[0] for a,q in zip(intervals,intervals[1:]))
  assert np.all(chipwaves<=131)
  inventory={'jobs':0,'pages':0,'waves':0,'useful_bytes':0,'valid_rows':0}
  table_by_copy={};sharedwaves=np.zeros(32,np.int64)
  for copies,(g,d,o) in [(1,(29102,28762,-340)),(4,(31406,29530,-1876))]:
   s=replace(base,input_channel_copies=copies);table=np.zeros((65,2,32),np.int64)
   inv={k:0 for k in inventory}
   for e in range(65):
    for phase in range(2):
     C=1536 if phase==0 else (512 if e<64 else 1024)
     R=(1024 if phase==0 else 1536) if e<64 else (2048 if phase==0 else 1536)
     coverage=np.zeros(R,np.uint8);cursor=0;lo,hi=map(int,plan['offsets'][e,phase])
     for j in range(lo,hi):
      chip,n,c,pg,w=map(int,plan['jobs'][j]);assert c==C and n==int(rows[e,phase,chip])
      assert pg==up(n*C,16384) and w==up(pg,4)
      coverage[cursor:cursor+n]+=1;cursor+=n
      emit=pg*16384//C
      table[e,phase,chip]+=s.command_ns+up(C*s.input_bytes*copies*10**9,s.channel_bytes_per_second)+w*s.ordinary_read_ns+up(emit*s.output_bytes*10**9,s.channel_bytes_per_second)
      for k,v in [('jobs',1),('pages',pg),('waves',w),('useful_bytes',n*C),('valid_rows',n)]:inv[k]+=v
      if copies==1 and e==64:sharedwaves[chip]+=w
     assert cursor==R and np.all(coverage==1)
   assert np.array_equal(table,b.occupancy_table(plan,s)['chip_ns'])
   formula=g+d*X+o*O;assert np.array_equal(table[:64].sum(1),formula)
   actual=table[routes].sum((1,2))+table[64].sum(0)
   assert np.array_equal(actual,formula[routes].sum(1)+table[64].sum(0))
   assert inv=={'jobs':2424,'pages':9764,'waves':2441,'useful_bytes':int(148.5*2**20),'valid_rows':167424}
   table_by_copy[copies]=actual;inventory=inv
  work=table_by_copy[1].max(1);assert np.array_equal(work,z[ident+'.work'])
  cand=next(c for c in fit['candidates'] if c['id']==ident);assert cand['score']==int(work.sum()) and cand['work']==stats(work)
  peak=int((6+sharedwaves+X[routes].sum(1)).max());assert peak==cand['peak_readwaves']
  for priority in [0,1]:
   key=ident+'.'+('DF' if priority else 'GF');cost=na[key+'.cost'];bound=na[key+'.bound']
   assert cost.shape==(512,) and np.all(cost>=bound)
   r=next(r for r in nr['records'] if r['id']==ident and r['priority']==('DF' if priority else 'GF'))
   assert r['cost']==stats(cost) and r['bound']==stats(bound);records.append(r)
   if ident=='source':
    oldkey='granite4tiny.M32.minimum_depth.'+('DF' if priority else 'GF')+'.cost'
    assert np.array_equal(cost,old[oldkey]);source_bridge.append({'priority':priority,'all512costs_exact':True})
   for q in [0,255,511]:
    got=b.ReferenceGraph.build(plan,np.r_[routes[q],64],replace(base,down_first=priority)).run()
    assert got['cost']==int(cost[q])
    if q==0:assert np.array_equal(got['trace'],na[key+'.q0trace'])
    traces.append({'map':ident,'priority':priority,'q':q,'cost_ns':got['cost'],'nodes':len(got['trace']),'q0fulltrace_exact':q==0})
  saved[ident+'.onehot']=onehot;saved[ident+'.labels']=labels;saved[ident+'.owned_input1']=table_by_copy[1]
  results.append({'id':ident,'exact64_common_rotation_rows':True,'GU40chip_positions_exact':True,'onehot_shape':[64,32],
    'all512chip_work_exact_both_fanouts':True,'shared_unchanged':True,'complete_rows_onecopy':True,'same_inventory':inventory,
    'singleblock_addresses_valid':True,'chip_reserved_waves':chipwaves.tolist(),'peak_NANDwaves':peak})

 best=min(records,key=lambda r:r['cost']['mean_ns']);assert best==nr['best_cyclic']
 free=json.loads((B/'g4-M32-coupled-cp/native_report.json').read_text())['best_coupled_input1']
 free_arrays=load(B/'g4-M32-coupled-cp/native_arrays.npz');fv=free_arrays[free['id']+'.'+free['priority']+'.cost']
 assert free['cost']==stats(fv) and free==nr['previous_free']
 assert free['cost']=={'mean_ns':299342.484375,'p99_ns':300795.0}
 gain=100*(1-free['cost']['mean_ns']/best['cost']['mean_ns']);p99gain=100*(1-free['cost']['p99_ns']/best['cost']['p99_ns'])
 assert abs(gain-nr['free_mean_improvement_vs_cyclic_pct'])<1e-10
 assert abs(p99gain-nr['free_p99_improvement_vs_cyclic_pct'])<1e-10
 assert len(traces)<=48
 for f,h in binding['files'].items():assert sha(f)==h
 saved['routes']=routes;np.savez_compressed(OUT/'audit_arrays.npz',**saved)
 report={'status':'PASS_EXACT_LEARNED_ROTATION_ORBIT_AND_NATIVE_AUDIT','utc':datetime.now(timezone.utc).isoformat(),
  'protocol_sha256':sha(OUT/'protocol.frozen.json'),'binding_sha256':sha(OUT/'results_binding.frozen.json'),
  'source64expert_labels':source_labels,'maps':results,'native_reference_checks':traces,'source512_native_bridge':source_bridge,
  'best_learned_rotation':best,'previous_free_best':free,'free_vs_learned_rotation_mean_pct':gain,'free_vs_learned_rotation_p99_pct':p99gain,
  'reported_solver_stages':fit['stages'],'new_CPP_costs':0,
  'claim_limits':['Learnedrotation nativebest, not routeblind source, is the attribution baseline.','BothstandardCP, sameboundedsolverbudget; no globaloptimality without actual certificate.',
    'Input4chipwork checked but thisproducer hasno input4native replay; do not invent its latency.','TRAIN L0 only;32chip equalcapability scale-model, not manufacturedAiF.',
    'Samecomplete-row profile/capacity/shared, no extra copies/newgeometry/quantization/nativeABI.','No firstsolver/firstjoint/firsthypergraph claim. Smallremaininggain is a stop gate, not reason to expand all layers/heldout.']}
 dump(OUT/'audit.json',report)
 files=[f for f in OUT.rglob('*') if f.is_file() and f.name!='completion.json']
 dump(OUT/'completion.json',{'status':'SEALED_PASS','utc':datetime.now(timezone.utc).isoformat(),'files':{str(f.relative_to(OUT)):sha(f) for f in sorted(files)}})
 print(json.dumps({'status':report['status'],'maps':len(results),'reference_graphs':len(traces),'free_mean_gain_pct':gain,
  'free_p99_gain_pct':p99gain,'audit_sha256':sha(OUT/'audit.json'),'completion_sha256':sha(OUT/'completion.json')},indent=2))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['freeze','bind','run']);ap.add_argument('--seal');a=ap.parse_args()
 if a.mode=='bind':assert a.seal;bind(a.seal)
 else:{'freeze':freeze,'run':run}[a.mode]()
