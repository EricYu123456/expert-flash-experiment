#!/usr/bin/env python3
"""Bounded frozen-job service-contract audit; no native FIFO/PE implementation."""
from collections import defaultdict
import ctypes,hashlib,heapq,json,shutil,subprocess
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/research-2026-11-02/round2/ordinary-read-compiler/full-campaign-v4-deduplicated'
DEST=ROOT/'results/research-2026-11-02/round2/wave-drain'
ORDER=('chips','channels','planes','page_bytes','block_wordlines','input_bytes','output_bytes','ordinary_read_ns','command_ns','host_submit_ns','pcie_bytes_per_second','channel_bytes_per_second','activation_ns','combine_ns','output_fringe','down_first','input_buffer_bytes')
CASES=('postcompute_fanout1','perwave_fanout1','postcompute_fanout4','perwave_fanout4')
MODES=((0,1),(1,1),(0,4),(1,4))

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def cdiv(x,y):return (int(x)+int(y)-1)//int(y)
def transfer(n,rate):return cdiv(int(n)*10**9,rate)
def write(name,x):
 p=DEST/name;p.write_text(json.dumps(x,indent=2)+'\n');return p

def chunks(job,s):
 _,r,C,pages,waves=map(int,job);B=s['planes']*s['page_bytes'];prior=0;out=[]
 for w in range(1,waves+1):
  total=min(w*B,pages*s['page_bytes'])//C
  out.append((prior,total));prior=total
 assert prior==pages*s['page_bytes']//C and prior>=r
 return out


def validate(plan,logical,s):
 o,j,C,T=(plan[k] for k in ('offsets','jobs','input_columns','total_rows'))
 E=len(o);assert o.shape==(E,2,2) and C.shape==T.shape==(E,2)
 assert o.dtype==j.dtype==C.dtype==np.int32 and j.ndim==2 and j.shape[1]==5
 assert np.all(C[:,0]==C[0,0]) and np.all(C>0)
 assert len(logical)==len(j)
 pointer=0;max_job_fifo=max_wave_fifo=0;partial_rows=0
 for e in range(E):
  for p in range(2):
   lo,hi=map(int,o[e,p]);assert lo==pointer and hi>lo;cursor=0
   for z in range(lo,hi):
    chip,r,c,pages,waves=map(int,j[z]);assert 0<=chip<s['chips'] and r>0 and c==C[e,p]
    assert pages==cdiv(r*c,s['page_bytes']) and waves==cdiv(pages,s['planes']) and waves<=s['block_wordlines']
    if s['input_buffer_bytes']:assert c*s['input_bytes']<=s['input_buffer_bytes']
    want=dict(expert=e,phase=p,chip=chip,row_start=cursor,rows=r,columns=c,pages=pages,waves=waves,useful_bytes=r*c)
    assert all(logical[z][k]==v for k,v in want.items());cursor+=r
    cs=chunks(j[z],s);emitted=pages*s['page_bytes']//c
    # Independent row-end enumeration assigns each complete row to its wave.
    count=[0]*waves
    for row in range(emitted):
     w=cdiv((row+1)*c,s['planes']*s['page_bytes'])-1
     assert 0<=w<waves;count[w]+=1
    assert count==[b-a for a,b in cs]
    assert [x for a,b in cs for x in range(a,b)]==list(range(emitted))
    assert sum(min(b,r)-min(a,r) for a,b in cs)==r
    max_job_fifo=max(max_job_fifo,emitted*s['output_bytes']);max_wave_fifo=max(max_wave_fifo,max(count)*s['output_bytes'])
    partial_rows+=sum((w*s['planes']*s['page_bytes'])%c!=0 for w in range(1,waves))
   assert cursor==T[e,p];pointer=hi
 assert pointer==len(j)
 return dict(jobs=len(j),max_postcompute_batch_bytes=max_job_fifo,max_perwave_batch_bytes=max_wave_fifo,row_crossing_wave_boundaries=partial_rows)


class ScalarGraph:
 def __init__(self,s,mode,fanout):self.s=s;self.mode=mode;self.fanout=fanout;self.nodes=[];self.jobs=0
 def add(self,r,d,deps=(),phase=0,e=-1,chip=-1,job=-1,acquire=False,release=False,stage='',wave=-1,nbytes=0):
  i=len(self.nodes);self.nodes.append(dict(resource=r,duration=int(d),deps=list(deps),phase=phase,expert=e,chip=chip,job=job,acquire=acquire,release=release,stage=stage,wave=wave,bytes=int(nbytes)));return i
 def matrix(self,ei,ph,desc,submit,upload):
  s=self.s;chip,rows,C,pages,waves=map(int,desc);jid=self.jobs;self.jobs+=1;ch=3+chip%s['channels'];array=3+s['channels']+chip
  cmd=self.add(1,s['command_ns'],[submit],ph,ei,chip,jid,True,stage='command')
  inp=self.add(ch,self.fanout*transfer(C*s['input_bytes'],s['channel_bytes_per_second']),[cmd,upload],ph,ei,chip,jid,stage='input',nbytes=C*s['input_bytes']*self.fanout)
  portions=chunks(desc,s) if self.mode else [(0,pages*s['page_bytes']//C)]
  previous=inp;hostprevious=None
  for w,(a,b) in enumerate(portions):
   comp=self.add(array,s['ordinary_read_ns']*(1 if self.mode else waves),[previous],ph,ei,chip,jid,stage='read',wave=w)
   size=(b-a)*s['output_bytes']
   out=self.add(ch,transfer(size,s['channel_bytes_per_second']),[comp],ph,ei,chip,jid,release=w==len(portions)-1,stage='output_channel',wave=w,nbytes=size)
   deps=[out]+([hostprevious] if self.mode and hostprevious is not None else [])
   hostprevious=self.add(2,transfer(size,s['pcie_bytes_per_second']),deps,ph,ei,stage='output_host',wave=w,nbytes=size)
   previous=out
  return hostprevious
 @classmethod
 def build(cls,plan,selected,s,mode,fanout):
  g=cls(s,mode,fanout);submit=g.add(0,s['host_submit_ns'],stage='initial_submit')
  upload=g.add(2,transfer(int(plan['input_columns'][0,0])*s['input_bytes'],s['pcie_bytes_per_second']),[submit],stage='common_GU_upload')
  joins=[]
  for ei,e in enumerate(selected):
   lo,hi=plan['offsets'][e,0];outputs=[g.matrix(ei,0,x,submit,upload) for x in plan['jobs'][lo:hi]]
   joins.append(g.add(3+s['channels']+s['chips'],0,outputs,e=ei,stage='full_GU_ready'))
  complete=[]
  for ei,e in enumerate(selected):
   act=g.add(0,s['activation_ns'],[joins[ei]],1,ei,stage='host_activation')
   ds=g.add(0,s['host_submit_ns'],[act],1,ei,stage='down_submit')
   up=g.add(2,transfer(int(plan['input_columns'][e,1])*s['input_bytes'],s['pcie_bytes_per_second']),[ds],1,ei,stage='full_down_upload')
   lo,hi=plan['offsets'][e,1];complete.extend(g.matrix(ei,1,x,ds,up) for x in plan['jobs'][lo:hi])
  g.add(0,s['combine_ns'],complete,1,stage='combine');return g
 def run(self):
  s=self.s;n=self.nodes;R=4+s['channels']+s['chips'];ready=[[] for _ in range(R)];owner=[-1]*s['chips'];busy=[False]*R;nexts=[[] for _ in n];pending=[];events=[];start=[-1]*len(n);end=start.copy()
  def enqueue(i):
   x=n[i];priority=1-x['phase'] if s['down_first'] else x['phase'];heapq.heappush(ready[x['resource']],((priority,x['expert'],i),i))
  for i,x in enumerate(n):
   pending.append(len(x['deps']))
   for d in x['deps']:assert d<i;nexts[d].append(i)
   if not pending[i]:enqueue(i)
  now=done=0
  while done<len(n):
   for r,q in enumerate(ready):
    if busy[r]:continue
    held=[];chosen=None
    while q:
     item=heapq.heappop(q);x=n[item[1]]
     valid=x['job']<0 or (owner[x['chip']]<0 if x['acquire'] else owner[x['chip']]==x['job'])
     if valid:chosen=item[1];break
     held.append(item)
    for item in held:heapq.heappush(q,item)
    if chosen is None:continue
    x=n[chosen]
    if x['acquire']:owner[x['chip']]=x['job']
    start[chosen]=now;busy[r]=True;heapq.heappush(events,(now+x['duration'],chosen))
   assert events;now=events[0][0]
   while events and events[0][0]==now:
    _,i=heapq.heappop(events);x=n[i];end[i]=now;busy[x['resource']]=False
    if x['release']:assert owner[x['chip']]==x['job'];owner[x['chip']]=-1
    done+=1
    for j in nexts[i]:
     pending[j]-=1
     if not pending[j]:enqueue(j)
  return now,np.array([[x['resource'],x['duration'],start[i],end[i],x['phase'],x['expert'],x['chip'],x['job']] for i,x in enumerate(n)],dtype=np.int64)
 def audit(self,trace):
  assert trace.shape==(len(self.nodes),8)
  resources=defaultdict(list);owners=defaultdict(list);jobs=defaultdict(list)
  for i,(x,t) in enumerate(zip(self.nodes,trace)):
   r,d,a,b,ph,e,chip,job=map(int,t)
   assert (r,d,ph,e,chip,job)==(x['resource'],x['duration'],x['phase'],x['expert'],x['chip'],x['job']) and a>=0 and b-a==d
   assert all(int(trace[q,3])<=a for q in x['deps'])
   resources[r].append((a,b,i))
   if job>=0:jobs[job].append(i)
  for ranges in resources.values():
   nonempty=sorted(x for x in ranges if x[0]<x[1]);assert all(a[1]<=b[0] for a,b in zip(nonempty,nonempty[1:]))
  for job,ids in jobs.items():
   cs=[i for i in ids if self.nodes[i]['acquire']];ends=[i for i in ids if self.nodes[i]['release']]
   assert len(cs)==len(ends)==1
   begin=int(trace[cs[0],2]);end=int(trace[ends[0],3]);chip=self.nodes[cs[0]]['chip'];owners[chip].append((begin,end,job))
   assert sum(self.nodes[i]['stage']=='input' for i in ids)==1
   assert all(begin<=trace[i,2]<=trace[i,3]<=end for i in ids)
  for ranges in owners.values():
   ranges.sort();assert all(a[1]<=b[0] for a,b in zip(ranges,ranges[1:]))
  # Every expert's full GU host delivery precedes its activation and full down upload.
  for ei in {x['expert'] for x in self.nodes if x['stage']=='host_activation'}:
   gu=[i for i,x in enumerate(self.nodes) if x['expert']==ei and x['phase']==0 and x['stage']=='output_host']
   act=next(i for i,x in enumerate(self.nodes) if x['expert']==ei and x['stage']=='host_activation')
   assert max(int(trace[i,3]) for i in gu)<=trace[act,2]
  return dict(nodes=len(self.nodes),jobs=len(jobs),resource_intervals_checked=sum(map(len,resources.values())),owner_intervals_checked=sum(map(len,owners.values())),full_GU_barriers_checked=sum(x['stage']=='host_activation' for x in self.nodes))


class Native:
 def __init__(self,source):
  build=DEST/'build';build.mkdir();self.source=source;self.binary=build/('wave-drain-'+sha(source)[:16]+'.so')
  subprocess.run(['c++','-std=c++17','-O3','-shared','-fPIC',str(source),'-o',str(self.binary)],check=True)
  self.lib=ctypes.CDLL(str(self.binary));self.call=self.lib.eval_wave_drain
  p32=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS');p64=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS')
  self.call.argtypes=[p32,p32,p32,p32,*([ctypes.c_int]*5),p64,p64,p64,ctypes.c_void_p,p64];self.call.restype=None
 def run(self,plan,routes,experts,s,mode,fanout,trace=False):
  routes=np.ascontiguousarray(routes,dtype=np.int32);cfg=np.array([s[k] for k in ORDER]+[mode,fanout],dtype=np.int64)
  chosen=list(map(int,routes[0]))+list(range(experts,len(plan['offsets'])))
  desc=np.concatenate([plan['jobs'][a:b] for e in chosen for a,b in plan['offsets'][e]])
  maxnodes=3+4*len(chosen)+sum(2+3*int(d[4]) if mode else 5 for d in desc)
  log=np.zeros((maxnodes if trace else 1,8),dtype=np.int64);nlog=np.zeros(1,dtype=np.int64);cost=np.zeros(len(routes),dtype=np.int64);lb=cost.copy()
  self.call(plan['offsets'],plan['jobs'],plan['input_columns'],routes,experts,len(plan['offsets'])-experts,len(plan['jobs']),len(routes),routes.shape[1],cfg,cost,lb,ctypes.c_void_p(log.ctypes.data) if trace else None,nlog)
  assert np.all(cost>=lb) and nlog[0]==maxnodes
  return dict(cost=cost,lower_bound=lb,trace=log[:nlog[0]].copy() if trace else None)


def scalar_tests(native,service):
 rng=np.random.default_rng(11032026);results=[]
 for z in range(24):
  s=dict(service);s.update(chips=1+z%4,channels=1,planes=1+z%4,page_bytes=3+z%7,block_wordlines=7,input_buffer_bytes=0,ordinary_read_ns=0 if z%7==0 else 3,command_ns=0 if z%5==0 else 2,host_submit_ns=0 if z%3==0 else 1,activation_ns=z%2,combine_ns=z%3,pcie_bytes_per_second=10**9,channel_bytes_per_second=10**9,down_first=z%2)
  jobs=[];offsets=np.zeros((3,2,2),dtype=np.int32);columns=np.zeros((3,2),dtype=np.int32);total=np.zeros((3,2),dtype=np.int64);logical=[]
  Cgu=1+z%(s['page_bytes']*s['planes']+3)
  for e in range(3):
   for ph in range(2):
    C=Cgu if ph==0 else 1+(z+e)%(s['page_bytes']*s['planes']+4);columns[e,ph]=C;cursor=0;offsets[e,ph,0]=len(jobs)
    for chip in range(s['chips']):
     maxrows=s['block_wordlines']*s['planes']*s['page_bytes']//C;r=int(rng.integers(1,min(maxrows,9)+1));pages=cdiv(r*C,s['page_bytes']);waves=cdiv(pages,s['planes'])
     jobs.append((chip,r,C,pages,waves));logical.append(dict(expert=e,phase=ph,chip=chip,row_start=cursor,rows=r,columns=C,pages=pages,waves=waves,useful_bytes=r*C));cursor+=r
    total[e,ph]=cursor;offsets[e,ph,1]=len(jobs)
  plan=dict(offsets=offsets,jobs=np.array(jobs,dtype=np.int32),input_columns=columns,total_rows=total);validate(plan,logical,s)
  routes=np.array([[1,0]],dtype=np.int32)
  for mode,fanout in MODES:
   out=native.run(plan,routes,2,s,mode,fanout,True);g=ScalarGraph.build(plan,[1,0,2],s,mode,fanout);cost,tr=g.run()
   assert cost==out['cost'][0] and np.array_equal(tr,out['trace']);check=g.audit(out['trace'])
   results.append(dict(test=z,mode=mode,fanout=fanout,cost_ns=cost,**check))
 return results


def main():
 DEST.mkdir(parents=True,exist_ok=True);assert not (DEST/'protocol.frozen.json').exists(),'Keep frozen outputs; select a fresh directory.'
 snap=DEST/'snapshots';snap.mkdir()
 source=ROOT/'scripts/research_20261103_wave_drain_native.cpp';script=Path(__file__)
 inputs={str(x):sha(x) for x in [script,source,BASE/'protocol.frozen.json',BASE/'source-snapshot/manifest.json',ROOT/'paper/aif.txt']}
 records=[]
 for model in ('qwen38','granite4tiny','deepseek'):
  for layer in range(3):
   sp=BASE/f'{model}.L{layer}.cmd250/summary.json';d=json.loads(sp.read_text());inputs[str(sp)]=sha(sp);records.append(d)
   for a in d['artifacts']:
    for key in ('arrays','plan'):
     p=ROOT/a[key];assert sha(p)==a[key+'_sha256'];inputs[str(p)]=a[key+'_sha256']
 for name in ('research_20261103_ordinary_read_native.cpp','research_20261103_ordinary_read_backend.py'):
  p=BASE/'source-snapshot'/name;shutil.copy2(p,snap/name);inputs[str(p)]=sha(p)
 for p in (script,source):shutil.copy2(p,snap/p.name)
 write('protocol.frozen.json',dict(inputs_sha256=inputs,profiles=[dict(name=n,perwave=bool(m),ONFI_input_copies=f) for n,(m,f) in zip(CASES,MODES)],models=['qwen38','granite4tiny','deepseek'],layers=[0,1,2],command_ns=250,queries=64,
  semantics='One command/full input reload per complete-row single-block job. Perwave: read one ordinary wave, drain newly complete all-requested-page row outputs over ONFI, then next wave. Output PCIe may overlap next wave read but stays ordered within a job. Owner acquired at command and held through final ONFI drain. GU waits all emitted outputs at host, then host activation and complete private A8 Down input; no chunked nonlinear pipeline. Fourfold input sensitivity serializes four full-vector ONFI copies, common GU PCIe stays once and each private Down PCIe stays once.',
  limits='Perwave pause/backpressure at existing GEMV command is not documented by AiF, nor FIFO capacity, MAC throughput/output dtype. This is a finer service assumption sensitivity, not proven existing-hardware feasibility. A full wave of emitted outputs may still need buffering. No cycle/RTL/device measurements.',
  checks='Match all27x64 original postcompute/fanout1 saved costs. Independent scalar simulator and interval/DAG/owner/full-GU audits on first query of each profile/plan plus24tiny varied shapes including zero durations and rowstraddlingwaves. All job geometry/rows and fragments checked against array fields and independent row-end enumeration before replay.'))
 native=Native(source);write('build_receipt.json',dict(source_sha256=sha(source),binary=str(native.binary),binary_sha256=sha(native.binary)))
 scalar=scalar_tests(native,records[0]['service']);write('scalar_checks.json',dict(status='PASS',cases=len(scalar),records=scalar))
 diagnostics=[];results=[];cost_arrays={};geometry=[]
 for d in records:
  model,layer=d['model'],d['layer'];s=d['service'];E=d['shape'][2]
  selected_indices=None
  for a in d['artifacts']:
   family=a['family'];label=f'{model}.L{layer}.{family}'
   with np.load(ROOT/a['arrays']) as f:plan={k:f[k].copy() for k in ('offsets','jobs','input_columns','total_rows')};routes=f['selection_routes'].copy();old=f['cost'].copy()
   assert routes.shape==(64,d['shape'][3]);local=json.loads((ROOT/a['plan']).read_text());geom=validate(plan,local['logical'],s);geometry.append(dict(label=label,**geom))
   allcost=[]
   for name,(mode,fanout) in zip(CASES,MODES):
    out=native.run(plan,routes,E,s,mode,fanout,True);allcost.append(out['cost']);cost_arrays[label+'.'+name]=out['cost']
    if (mode,fanout)==(0,1):assert np.array_equal(out['cost'],old)
    chosen=list(map(int,routes[0]))+list(range(E,len(plan['offsets'])));g=ScalarGraph.build(plan,chosen,s,mode,fanout);ref,tr=g.run()
    assert ref==out['cost'][0] and np.array_equal(tr,out['trace']);check=g.audit(out['trace'])
    diagnostics.append(dict(label=label,profile=name,exact_scalar_and_trace_match=True,**check))
    results.append(dict(model=model,layer=layer,family=family,profile=name,mean_ns=float(out['cost'].mean()),p99_ns=float(np.percentile(out['cost'],99)),max_ns=int(out['cost'].max()),first_ns=int(out['cost'][0])))
    if layer==0:np.savez_compressed(DEST/(label+'.'+name+'.first_trace.npz'),trace=out['trace'])
   cost_arrays[label+'.perwave_delta_fanout1']=allcost[1]-allcost[0];cost_arrays[label+'.perwave_delta_fanout4']=allcost[3]-allcost[2]
   print(label,'PASS',*[round(float(x.mean())/1e6,6) for x in allcost],flush=True)
 np.savez_compressed(DEST/'costs.npz',**cost_arrays)
 aggregates=[]
 for model in ('qwen38','granite4tiny','deepseek'):
  for name in CASES:
   values={f:sum(cost_arrays[f'{model}.L{l}.{f}.{name}'] for l in range(3)) for f in ('disjoint','cyclic','whole_rows')}
   cyclic=values['cyclic'];whole=values['whole_rows']
   aggregates.append(dict(model=model,profile=name,layers=3,queries=64,means_ns={f:float(v.mean()) for f,v in values.items()},whole_reduction_vs_cyclic=1-float(whole.mean())/float(cyclic.mean()),whole_faster_queries=int(np.count_nonzero(whole<cyclic)),whole_equal_queries=int(np.count_nonzero(whole==cyclic))))
 assert all(sha(p)==h for p,h in inputs.items())
 write('summary.json',dict(status='PASS bounded contract sensitivity',original_saved_cost_matches=27*64,scalar_cases=len(scalar),real_firstquery_trace_checks=len(diagnostics),service_cost_evaluations=len(results)*64,records=results,three_layer_aggregates=aggregates,geometry=geometry,trace_diagnostics=diagnostics,
  interpretation='Compare all profiles for the same frozen geometries/labels/routes. No refit, native numeric proof, calibrated PE/FIFO/inputbroadcast conclusion or full-model speed claim. Perwave input/output fragmentation rounds each transfer separately.'))
 write('completion.json',dict(status='complete',outputs_sha256={str(p):sha(p) for p in sorted(DEST.rglob('*')) if p.is_file() and '__pycache__' not in str(p) and p.name!='completion.json'}))
 print(json.dumps(aggregates,indent=2),flush=True)

if __name__=='__main__':main()
