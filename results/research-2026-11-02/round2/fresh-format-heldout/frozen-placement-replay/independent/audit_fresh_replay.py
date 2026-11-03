#!/usr/bin/env python3
"""Independent provenance, whole40 aggregation and separate set-queue replay."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import collections,datetime,hashlib,importlib.util,json,pathlib,time
import numpy as np
OUT=pathlib.Path(__file__).resolve().parent
REPLAY=OUT.parent
FRESH=REPLAY.parent
BASE=FRESH.parent
PLAN=BASE/'geometry-study/granite4tiny.frozen.json'
INDEPENDENT_SOURCE=BASE/'full-ffn-integer-reference/frozen-placement-replay/independent/audit_replay.py'
CASES={'frozen_phase_group':'group_rows','frozen_whole_rows':'rows'}
PHASES={};ROUTES=None;ENGINE=None

def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def initialize():
 global PHASES,ROUTES,ENGINE
 assert sha(INDEPENDENT_SOURCE)=='36a53b2ef51c82677c9fd3440533edf7f14169df9227d43ad95686cbacfd11c6'
 module_spec=importlib.util.spec_from_file_location('older_independent_set_queue',INDEPENDENT_SOURCE);ENGINE=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(ENGINE)
 doc=json.loads(PLAN.read_text());PHASES={(p['layer'],p['name']):p for p in doc['phases']}
 with np.load(REPLAY/'routes.npz') as f:ROUTES=f['routes']

def audit_layer(task):
 layer,case=task;key=CASES[case];gu=PHASES[layer,'gate_up'][key];down=PHASES[layer,'down'][key];trace=json.loads((REPLAY/f'L{layer}.{case}.trace.json').read_text())
 with np.load(REPLAY/f'L{layer}.{case}.npz') as f:expected=f['cost'].reshape(-1);bounds=f['native_chip_bound'].reshape(-1);waves=f['total_chip_waves'].reshape(-1);jobcounts=f['native_jobs'].reshape(-1)
 actual=[];native_job_checks=0;retained_nodes=0;retained_jobs=0
 for sample,selected in enumerate(ROUTES[:,:,layer].reshape(-1,6)):
  items=[dict(gu_rows=gu[int(e)],down_rows=down[int(e)],down_columns=512) for e in selected];items.append(dict(gu_rows=[128]*16,down_rows=[96]*16,down_columns=1024))
  assert sum(sum(x['gu_rows']) for x in items)==8192 and sum(sum(x['down_rows']) for x in items)==10752
  nodes,jobs=ENGINE.compile_jobs(items);cost,starts,ends=ENGINE.independent_schedule(nodes);actual.append(cost);assert int(expected[sample])==cost
  chip=[0]*16;w=0
  for j in jobs:chip[j['chip']]+=250+(j['columns']+1)//2+28000*j['waves']+2*j['rows'];w+=j['waves']
  bound=2000+max(chip);assert int(bounds[sample])==bound and cost>=bound and int(waves[sample])==w and int(jobcounts[sample])==len(jobs);native_job_checks+=len(jobs)
  if sample==0:
   assert trace['layer']==layer and trace['case']==case and trace['selected']==selected.tolist() and trace['items']==items
   retained_jobs=ENGINE.verify_trace(trace,nodes,jobs);retained_nodes=len(nodes)
   assert trace['makespan_ns']==cost and all(t['start_ns']==starts[i] and t['end_ns']==ends[i] for i,t in enumerate(trace['trace']))
 return {'layer':layer,'case':case,'cost':actual,'native_jobs_checked':native_job_checks,'retained_trace_nodes_checked':retained_nodes,'retained_trace_jobs_checked':retained_jobs}

def main():
 start=time.monotonic();protocol=json.loads((REPLAY/'protocol.frozen.json').read_text());summary=json.loads((REPLAY/'summary.json').read_text());manifest=json.loads((FRESH/'manifest.json').read_text());items=manifest['items'];assert len(items)==16
 for p,h in {**protocol['inputs_sha256'],**protocol['source_sha256']}.items():assert sha(p)==h
 assert summary['protocol_sha256']==sha(REPLAY/'protocol.frozen.json') and sha(PLAN)=='a55f17010b7b48c6c396dee92bb1d127a541e28a93b94aee2e42aac9f3dc2538'
 with np.load(REPLAY/'routes.npz') as f:routes=f['routes'];ids=f['prompt_ids'].tolist();groups=f['dataset_groups'].tolist()
 assert routes.shape==(16,16,40,6) and ids==[j['id'] for j in items] and groups==[j['kind'] for j in items]
 assert protocol['prompt_ids']==ids and protocol['dataset_groups']==groups and groups==['lm']*8+['arc']*8
 provenance=[]
 for pi,j in enumerate(items):
  id=j['id'];placed=FRESH/'generation/run/traces'/f'{id}.placed.routing.jsonl';unsharded=FRESH/'generation/run/traces'/f'{id}.unsharded.routing.jsonl';assert placed.read_bytes()==unsharded.read_bytes()
  decoded={}
  for line in placed.read_text().splitlines():
   row=json.loads(line)
   if row['phase']=='decode':
    k=(row['step'],row['layer']);assert k not in decoded;decoded[k]=row
  assert set(decoded)=={(s,l) for s in range(16) for l in range(40)}
  for (step,layer),row in decoded.items():assert np.array_equal(routes[pi,step,layer],row['expert_ids']) and len(set(row['expert_ids']))==6 and all(0<=e<64 for e in row['expert_ids'])
  provenance.append({'prompt_id':id,'dataset_group':j['kind'],'decode_records':640,'placed_unsharded_byte_exact':True,'source_sha256':sha(placed)})
 extraction=json.loads((REPLAY/'route_extraction.json').read_text());assert extraction['routes_npz_sha256']==sha(REPLAY/'routes.npz') and extraction['raw_int32_sha256']==hashlib.sha256(routes.tobytes()).hexdigest()
 records=[]
 with ProcessPoolExecutor(max_workers=4,initializer=initialize) as pool:
  for future in as_completed([pool.submit(audit_layer,(layer,case)) for layer in range(40) for case in CASES]):
   records.append(future.result())
   if len(records)%10==0:print('Independent groups',len(records),'elapsed',round(time.monotonic()-start,2),flush=True)
 arrays={case:np.stack([np.asarray(next(r for r in records if r['layer']==layer and r['case']==case)['cost'],dtype=np.int64).reshape(16,16) for layer in range(40)]) for case in CASES}
 totals={case:a.sum(axis=0) for case,a in arrays.items()}
 for case,array in totals.items():
  assert np.array_equal(array,np.load(REPLAY/f'{case}.sum40.npy'))
  row=next(x for x in summary['totals'] if x['case']==case)
  assert row['samples']==256 and row['mean_ns']==float(array.mean()) and row['p99_ns']==float(np.percentile(array,99))
  np.save(OUT/f'{case}.independent_layers.npy',arrays[case])
 a,b=totals['frozen_phase_group'],totals['frozen_whole_rows'];assert summary['mean_reduction_fraction']==1-float(b.mean())/float(a.mean())
 for pi,row in enumerate(summary['per_prompt']):
  assert row['prompt_id']==ids[pi] and row['dataset_group']==groups[pi]
  assert row['phase_group']['mean_ns']==float(a[pi].mean()) and row['whole_rows']['mean_ns']==float(b[pi].mean())
  assert row['phase_group']['p99_ns']==float(np.percentile(a[pi],99)) and row['whole_rows']['p99_ns']==float(np.percentile(b[pi],99))
  assert row['mean_reduction_fraction']==1-float(b[pi].mean())/float(a[pi].mean())
 for row in summary['dataset_groups']:
  ix=[i for i,g in enumerate(groups) if g==row['dataset_group']];aa,bb=a[ix],b[ix];assert row['prompts']==len(ix)==8
  assert row['phase_group']['mean_ns']==float(aa.mean()) and row['whole_rows']['mean_ns']==float(bb.mean()) and row['phase_group']['p99_ns']==float(np.percentile(aa,99)) and row['whole_rows']['p99_ns']==float(np.percentile(bb,99)) and row['mean_reduction_fraction']==1-float(bb.mean())/float(aa.mean())
 import csv
 with open(REPLAY/'paired_per_token.csv',newline='') as f:csvrows=list(csv.DictReader(f))
 assert len(csvrows)==256
 for row in csvrows:
  i,t=int(row['prompt_index']),int(row['decode_step']);assert row['prompt_id']==ids[i] and row['dataset_group']==groups[i] and int(row['phase_group_ns'])==a[i,t] and int(row['whole_rows_ns'])==b[i,t] and int(row['delta_ns'])==a[i,t]-b[i,t]
 for p,h in {**protocol['inputs_sha256'],**protocol['source_sha256']}.items():assert sha(p)==h
 result={'status':'PASS','audit_source_sha256':sha(__file__),'independent_engine_source_sha256':sha(INDEPENDENT_SOURCE),'protocol_sha256':sha(REPLAY/'protocol.frozen.json'),'replay_inputs_and_sources_unchanged':True,'frozen_old_plan_unchanged':True,'route_extract_exact':True,'placed_unsharded_routes_byte_exact':True,'actual_route_shape':[16,16,40,6],'prompts':16,'dataset_prompts':{'lm':8,'arc':8},'decode_positions_per_prompt':16,'refit':False,'independently_recompiled_rescheduled_layer_queries':20480,'independent_scheduler':'Earlier separately implemented set-queue/linear-event engine; imports no authoritative Graph or new replay code.','native_jobs_checked':sum(r['native_jobs_checked'] for r in records),'retained_traces_checked':80,'retained_trace_nodes_checked':sum(r['retained_trace_nodes_checked'] for r in records),'retained_trace_jobs_checked':sum(r['retained_trace_jobs_checked'] for r in records),'all_native_job_shapes_waves_bounds_exact':True,'all_retained_trace_start_end_resource_lock_causality_checks_exact':True,'all_per_token_per_prompt_dataset_sum40_and_P99_exact':True,'mean_reduction_fraction':summary['mean_reduction_fraction'],'totals':summary['totals'],'dataset_groups':summary['dataset_groups'],'route_provenance':provenance,'wall_seconds':time.monotonic()-start,'limits':['16 prompts,256within-prompt-correlated positions; descriptive results, not a large-sample confidence claim.','Uncalibrated software FFN service model; no actual hardware/native-AiF ABI/full-model latency claim.']}
 write(OUT/'audit.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['dataset_groups','route_provenance']},indent=2),flush=True)
if __name__=='__main__':main()
