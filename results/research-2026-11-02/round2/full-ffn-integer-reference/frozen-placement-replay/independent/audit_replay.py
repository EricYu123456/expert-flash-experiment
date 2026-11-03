#!/usr/bin/env python3
"""Independent geometry compiler and set-queue event scheduler; imports no replay code."""
import collections, datetime, hashlib, json, pathlib, time
import numpy as np
OUT=pathlib.Path(__file__).resolve().parent
REPLAY=OUT.parent
REFERENCE=REPLAY.parent
PLAN=REFERENCE.parent/'geometry-study/granite4tiny.frozen.json'
NAMES=['host','controller','pcie']+[f'channel{i}' for i in range(8)]+[f'chip{i}' for i in range(16)]+['sync']
CASES={'frozen_phase_group':'group_rows','frozen_whole_rows':'rows'}
def sha(path):return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
def compile_jobs(items):
 nodes=[];jobs=[]
 def add(resource,duration,parents=(),phase=0,expert=-1,kind='',chip=-1,job=-1,acquire=False,release=False):
  idx=len(nodes);nodes.append(dict(resource=resource,duration=duration,deps=list(parents),phase=phase,expert=expert,kind=kind,chip=chip,job=job,acquire=acquire,release=release));return idx
 submit=add(0,2000,kind='initial_host_submission');input_ready=add(2,192,[submit],kind='common_input_to_controller')
 def matrix(e,chip,phase,rows,cols,submit_node,input_node):
  j=len(jobs)
  cmd=add(1,250,[submit_node],phase,e,'matrix_command',chip,j,True)
  inp=add(3+chip%8,(cols+1)//2,[cmd,input_node],phase,e,'input_channel_reload',chip,j)
  waves=(rows*cols+65535)//65536;assert 1<=waves<=192
  comp=add(11+chip,waves*28000,[inp],phase,e,'nonpreemptive_native_compute',chip,j)
  out=add(3+chip%8,2*rows,[comp],phase,e,'output_channel',chip,j,False,True)
  host=add(2,(rows+1)//2,[out],phase,e,'output_pcie')
  jobs.append(dict(expert=e,phase=phase,chip=chip,rows=rows,columns=cols,waves=waves,command=cmd,input=inp,compute=comp,release=out,host=host));return host
 gu_done=[]
 for e,item in enumerate(items):
  parents=[matrix(e,c,0,r,1536,submit,input_ready) for c,r in enumerate(item['gu_rows']) if r]
  gu_done.append(add(27,0,parents,expert=e,kind='whole_expert_gu_host_ready'))
 terminal=[]
 for e,item in enumerate(items):
  act=add(0,0,[gu_done[e]],1,e,'host_activation')
  down_submit=add(0,2000,[act],1,e,'expert_down_host_submission')
  cols=item['down_columns'];down_input=add(2,(cols+7)//8,[down_submit],1,e,'private_input_to_controller')
  terminal.extend(matrix(e,c,1,r,cols,down_submit,down_input) for c,r in enumerate(item['down_rows']) if r)
 add(0,0,terminal,1,kind='host_combine')
 return nodes,jobs

def independent_schedule(nodes):
 # Sets and a linear event list are deliberately different from the original
 # per-resource ready heaps and event heap. Priority is held identical.
 children=[[] for _ in nodes];remaining=[len(n['deps']) for n in nodes]
 available=[set() for _ in NAMES]
 for i,n in enumerate(nodes):
  for dep in n['deps']:children[dep].append(i)
  if not remaining[i]:available[n['resource']].add(i)
 busy=[False]*len(NAMES);owners=[None]*16
 starts=[-1]*len(nodes);ends=[-1]*len(nodes);events=[];now=0;done=0
 while done<len(nodes):
  for resource in range(len(NAMES)):
   if busy[resource]:continue
   eligible=[]
   for i in available[resource]:
    n=nodes[i];owner=owners[n['chip']] if n['chip']>=0 else None
    if n['job']<0 or (owner is None if n['acquire'] else owner==n['job']):eligible.append(i)
   if not eligible:continue
   i=min(eligible,key=lambda i:(nodes[i]['phase'],nodes[i]['expert'],i));n=nodes[i];available[resource].remove(i)
   if n['acquire']:assert owners[n['chip']] is None;owners[n['chip']]=n['job']
   starts[i]=now;busy[resource]=True;events.append((now+n['duration'],i))
  assert events,'Independent scheduler deadlock'
  now=min(e[0] for e in events);complete=sorted(e for e in events if e[0]==now);events=[e for e in events if e[0]!=now]
  for _,i in complete:
   n=nodes[i];ends[i]=now;busy[n['resource']]=False;done+=1
   if n['release']:assert owners[n['chip']]==n['job'];owners[n['chip']]=None
   for child in children[i]:
    remaining[child]-=1
    if remaining[child]==0:available[nodes[child]['resource']].add(child)
 assert owners==[None]*16
 return max(ends),starts,ends

def verify_trace(trace,nodes,jobs):
 actual=trace['trace'];assert len(actual)==len(nodes)==trace['nodes']
 intervals=collections.defaultdict(list);work=[0]*len(NAMES)
 for i,(a,n) in enumerate(zip(actual,nodes)):
  assert a['index']==i and a['resource']==NAMES[n['resource']]
  for field in ['deps','phase','expert','kind','chip','job','acquire','release']:assert a[field]==n[field],(i,field)
  start,end=a['start_ns'],a['end_ns'];assert start>=0 and end-start==n['duration']
  assert all(actual[d]['end_ns']<=start for d in n['deps'])
  if end>start:intervals[a['resource']].append((start,end,i))
  work[n['resource']]+=end-start
 for values in intervals.values():
  values.sort();assert all(a[1]<=b[0] for a,b in zip(values,values[1:]))
 owned=collections.defaultdict(list)
 for j in jobs:
  a=actual[j['command']]['start_ns'];b=actual[j['release']]['end_ns'];owned[j['chip']].append((a,b))
  chain=[j[x] for x in ['command','input','compute','release','host']]
  assert all(actual[x]['end_ns']<=actual[y]['start_ns'] for x,y in zip(chain,chain[1:]))
 for intervals in owned.values():
  intervals.sort();assert all(a[1]<=b[0] for a,b in zip(intervals,intervals[1:]))
 assert trace['resource_work_ns']==work
 assert trace['makespan_ns']==max(n['end_ns'] for n in actual)
 return len(jobs)

def main():
 began=time.monotonic();protocol=json.loads((REPLAY/'protocol.frozen.json').read_text());summary=json.loads((REPLAY/'summary.json').read_text())
 for name,expected in {**protocol['inputs_sha256'],**protocol['source_sha256']}.items():assert sha(name)==expected,name
 assert sha(PLAN)=='a55f17010b7b48c6c396dee92bb1d127a541e28a93b94aee2e42aac9f3dc2538'
 assert summary['protocol_sha256']==sha(REPLAY/'protocol.frozen.json')
 plan=json.loads(PLAN.read_text());phases={(p['layer'],p['name']):p for p in plan['phases']};assert len(phases)==80
 for layer in range(40):
  for name,cols,rows in [('gate_up',1536,1024),('down',512,1536)]:
   p=phases[layer,name];assert p['columns']==cols and p['total_rows']==rows
   for key in CASES.values():
    a=np.asarray(p[key]);assert a.shape==(64,16) and np.all(a>=0) and np.all(a.sum(axis=1)==rows)
 with np.load(REPLAY/'routes.npz') as f:routes=f['routes'];ids=f['prompt_ids'].tolist()
 assert routes.shape==(2,16,40,6) and np.all((routes>=0)&(routes<64))
 route_provenance=[]
 for pi,id in enumerate(ids):
  placed=REFERENCE/'main/traces'/f'{id}.placed.routing.jsonl';canonical=REFERENCE/'main/traces'/f'{id}.unsharded.routing.jsonl';assert placed.read_bytes()==canonical.read_bytes()
  rows=[json.loads(l) for l in placed.read_text().splitlines()];decode=[r for r in rows if r['phase']=='decode'];assert len(decode)==640
  seen=set()
  for r in decode:
   assert (r['step'],r['layer']) not in seen;seen.add((r['step'],r['layer']))
   assert np.array_equal(routes[pi,r['step'],r['layer']],r['expert_ids']) and len(set(r['expert_ids']))==6
  route_provenance.append(dict(prompt=id,decode_layer_records=640,placed_unsharded_byte_exact=True,source_sha256=sha(placed)))
 shared=[json.loads(l) for l in (REFERENCE/'main/weight_cache_ledger.jsonl').read_text().splitlines() if 'shexp' in json.loads(l)['weight']]
 assert len(shared)==120
 for s in shared:
  assert s['expert']==0 and s['shared_layout']
  if 'down_shexp' in s['weight']:assert s['columns']==1024 and s['rows']==1536 and s['chip_rows']==[96]*16
  else:assert s['columns']==1536 and s['rows']==1024 and s['chip_rows']==[64]*16
 records=[];independent_costs={case:np.zeros((40,32),dtype=np.int64) for case in CASES};trace_jobs=0;trace_nodes=0
 # Reconstruct every sample, not merely the 80 retained first-query traces.
 for layer in range(40):
  selected=routes[:,:,layer].reshape(32,6)
  for case,key in CASES.items():
   gu=phases[layer,'gate_up'][key];down=phases[layer,'down'][key]
   npz_path=REPLAY/f'L{layer}.{case}.npz';trace_path=REPLAY/f'L{layer}.{case}.trace.json'
   with np.load(npz_path) as f:stored=f['cost'];bounds=f['native_chip_bound']
   assert stored.shape==bounds.shape==(32,)
   meta=next(r for r in summary['records'] if r['layer']==layer and r['case']==case)
   assert meta['artifacts_sha256']['npz']==sha(npz_path) and meta['artifacts_sha256']['trace.json']==sha(trace_path)
   assert meta['samples']==32 and meta['mean_ns']==float(stored.mean()) and meta['p99_ns']==float(np.percentile(stored,99))
   first=json.loads(trace_path.read_text())
   for sample,experts in enumerate(selected):
    items=[dict(gu_rows=gu[int(e)],down_rows=down[int(e)],down_columns=512) for e in experts]
    items.append(dict(gu_rows=[128]*16,down_rows=[96]*16,down_columns=1024))
    assert len(items)==7 and sum(sum(x['gu_rows']) for x in items)==8192 and sum(sum(x['down_rows']) for x in items)==10752
    nodes,jobs=compile_jobs(items);cost,starts,ends=independent_schedule(nodes);independent_costs[case][layer,sample]=cost
    assert cost==int(stored[sample]),(layer,case,sample,cost,int(stored[sample]))
    chip_load=[0]*16
    for j in jobs:chip_load[j['chip']]+=250+(j['columns']+1)//2+j['waves']*28000+2*j['rows']
    bound=2000+max(chip_load);assert int(bounds[sample])==bound and cost>=bound
    if sample==0:
     assert first['layer']==layer and first['case']==case and first['selected']==experts.tolist() and first['items']==items
     trace_jobs+=verify_trace(first,nodes,jobs);trace_nodes+=len(nodes)
     assert first['makespan_ns']==cost and all(t['start_ns']==starts[i] and t['end_ns']==ends[i] for i,t in enumerate(first['trace']))
   records.append(dict(layer=layer,case=case,samples=32,independent_makespan_exact=True,retained_trace_all_start_end_exact=True,native_bounds_exact=True))
  if layer%10==9:print('Audited layers',layer+1,'elapsed',round(time.monotonic()-began,2),flush=True)
 totals=[]
 for case,array in independent_costs.items():
  sums=array.sum(axis=0);stored=np.load(REPLAY/f'{case}.sum40.npy');assert np.array_equal(sums,stored)
  total=next(r for r in summary['totals'] if r['case']==case);assert total['mean_ns']==float(sums.mean()) and total['p99_ns']==float(np.percentile(sums,99)) and total['per_prompt_mean_ns']==sums.reshape(2,16).mean(axis=1).tolist()
  np.save(OUT/f'{case}.independently_rescheduled.npy',array)
  totals.append(dict(case=case,mean_ns=float(sums.mean()),p99_ns=float(np.percentile(sums,99)),per_prompt_mean_ns=sums.reshape(2,16).mean(axis=1).tolist()))
 reduction=1-totals[1]['mean_ns']/totals[0]['mean_ns'];assert reduction==summary['mean_reduction']
 for name,expected in {**protocol['inputs_sha256'],**protocol['source_sha256']}.items():assert sha(name)==expected
 result=dict(status='PASS',completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),audit_source_sha256=sha(__file__),replay_protocol_sha256=sha(REPLAY/'protocol.frozen.json'),independent_scheduler='Separately implemented geometry compiler, set-based per-resource queues and linear event list; no replay/Graph/build_jobs imports.',frozen_geometry_sha256=sha(PLAN),refit_performed=False,route_provenance=route_provenance,actual_W8_decode_route_shape=list(routes.shape),shared_cache_entries_checked=120,shared_GU_rows_per_chip=128,shared_down_rows_per_chip=96,shared_down_columns=1024,independently_rescheduled_layer_queries=2560,retained_first_query_traces_checked=80,retained_native_jobs_checked=trace_jobs,retained_nodes_checked=trace_nodes,all_resource_and_dependency_intervals_valid=True,all_whole_chip_ownership_intervals_valid=True,all_native_job_geometry_and_timing_valid=True,all_native_lower_bounds_exact=True,all_40layer_sum_arrays_and_P99_exact=True,totals=totals,mean_reduction_fraction=reduction,mean_reduction_percent=100*reduction,wall_seconds=time.monotonic()-began,limits=['Uncalibrated software FFN service replay, not hardware execution or full-model latency.','Two earlier training prompts and32 decode positions; no heldout/quality/generalization inference.','Shared gate64+up64 rows per chip can form a128-row whole-row concatenated native job with logical scatter; audit checks job shape/counts and CPU-reference partition, not programmed flash weight addresses.','F32 return byte costs and nominal28us reads are modeling assumptions; native AiF output ABI and latency are not established.'])
 write(OUT/'audit.json',result);write(OUT/'layer_case_records.json',records);print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
