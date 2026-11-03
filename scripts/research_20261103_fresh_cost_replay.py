#!/usr/bin/env python3
"""Fresh fixed-format full40/shared replay using the unchanged Python Graph.

No fitting and no new hardware operation; actual W8A8 routes are read verbatim.
"""
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import collections,csv,hashlib,json,pathlib,platform,time
import numpy as np
from research_20261102_shared_pipeline import build_jobs
ROOT=pathlib.Path(__file__).resolve().parents[1]
BASE=ROOT/'results/research-2026-11-02/round2'
FRESH=BASE/'fresh-format-heldout'
DEST=FRESH/'frozen-placement-replay'
PLAN=BASE/'geometry-study/granite4tiny.frozen.json'
EXPECTED='a55f17010b7b48c6c396dee92bb1d127a541e28a93b94aee2e42aac9f3dc2538'
CASES={'frozen_phase_group':'group_rows','frozen_whole_rows':'rows'}
WORKER_PHASES={};WORKER_ROUTES=None

def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def initialize_worker():
 global WORKER_PHASES,WORKER_ROUTES
 doc=json.loads(PLAN.read_text());WORKER_PHASES={(p['layer'],p['name']):p for p in doc['phases']}
 with np.load(DEST/'routes.npz') as f:WORKER_ROUTES=f['routes']

def worker(task):
 layer,case=task;key=CASES[case];gu=np.asarray(WORKER_PHASES[layer,'gate_up'][key],dtype=np.int64);down=np.asarray(WORKER_PHASES[layer,'down'][key],dtype=np.int64)
 selected=WORKER_ROUTES[:,:,layer].reshape(-1,6);cost=[];bounds=[];waves=[];jobs=[];cache={};trace=None
 for sample,experts in enumerate(selected):
  ids=tuple(map(int,experts))
  if ids not in cache:
   items=[dict(gu_rows=gu[e].tolist(),down_rows=down[e].tolist(),down_columns=512) for e in ids]
   items.append(dict(gu_rows=[128]*16,down_rows=[96]*16,down_columns=1024))
   graph=build_jobs(items);result=graph.run(False,trace=sample==0);chip=np.zeros(16,dtype=np.int64);totalwaves=0;totaljobs=0
   for item in items:
    for name,C in [('gu_rows',1536),('down_rows',item['down_columns'])]:
     R=np.asarray(item[name],dtype=np.int64);W=(R*C+65535)//65536;assert np.all(W<=192)
     chip+=np.where(R>0,W*28000+250+(C+1)//2+2*R,0);totalwaves+=int(W.sum());totaljobs+=int((R>0).sum())
   lb=2000+int(chip.max());assert result['makespan_ns']>=lb
   cache[ids]=(result['makespan_ns'],lb,totalwaves,totaljobs)
   if sample==0:trace=dict(layer=layer,case=case,prompt_index=0,decode_step=0,selected=list(ids),items=items,**result)
  v,lb,w,j=cache[ids];cost.append(v);bounds.append(lb);waves.append(w);jobs.append(j)
 label=f'L{layer}.{case}';target=DEST/(label+'.npz');np.savez_compressed(target,cost=np.asarray(cost,dtype=np.int64).reshape(16,16),native_chip_bound=np.asarray(bounds,dtype=np.int64).reshape(16,16),total_chip_waves=np.asarray(waves,dtype=np.int64).reshape(16,16),native_jobs=np.asarray(jobs,dtype=np.int64).reshape(16,16))
 tracepath=DEST/(label+'.trace.json');write(tracepath,trace)
 return dict(layer=layer,case=case,label=label,samples=256,unique_ordered_routes=len(cache),mean_ns=float(np.mean(cost)),p99_ns=float(np.percentile(cost,99)),artifacts_sha256={'npz':sha(target),'trace.json':sha(tracepath)})

def stats(array):
 a=np.asarray(array,dtype=np.int64);return dict(samples=int(a.size),mean_ns=float(a.mean()),p50_ns=float(np.percentile(a,50)),p95_ns=float(np.percentile(a,95)),p99_ns=float(np.percentile(a,99)),min_ns=int(a.min()),max_ns=int(a.max()))

def main():
 began=time.monotonic();DEST.mkdir(exist_ok=True)
 if (DEST/'protocol.frozen.json').exists():raise RuntimeError('Never overwrite a frozen replay')
 assert sha(PLAN)==EXPECTED
 manifest=json.loads((FRESH/'manifest.json').read_text());items=manifest['items'];assert len(items)==16 and [j['kind'] for j in items]==['lm']*8+['arc']*8
 gen=json.loads((FRESH/'generation/summary.json').read_text());assert gen['status']=='PASS' and gen['all_routes_and_generated_token_ids_identical'] and not gen['placement_refit']
 paths=[FRESH/'generation/run/traces'/f"{j['id']}.placed.routing.jsonl" for j in items]
 sources=[pathlib.Path(__file__),ROOT/'scripts/research_20261102_shared_pipeline.py',ROOT/'scripts/research_20261102_granite4_pipeline.py',ROOT/'scripts/research_20261102_single_request.py',ROOT/'scripts/run_corpus_traces.py']
 inputs=[PLAN,FRESH/'manifest.json',FRESH/'acquisition_protocol.frozen.json',FRESH/'generation/protocol.frozen.json',FRESH/'generation/summary.json',FRESH/'generation/run/numeric_report.json',*paths]
 hashes={str(p):sha(p) for p in inputs};sourcehashes={str(p):sha(p) for p in sources}
 protocol=dict(frozen_utc=datetime.now(timezone.utc).isoformat(),inputs_sha256=hashes,source_sha256=sourcehashes,prompt_ids=[j['id'] for j in items],dataset_groups=[j['kind'] for j in items],dataset_names=[j['dataset'] for j in items],splits=[j['split'] for j in items],shape=[16,16,40,6],cases=CASES,
  fitting='None: apply pre-existing frozen group_rows/rows to all actual W8A8 placed decode routes in acquisition-frozen order. Preserve selected expert order.',
  shared='Same fixed one-copy numeric-reference shared partition:128GU fullrows/chip (64gate+64up),1536columns;96down rows/chip,1024columns. Count the concatenated GU once and shared down once.',
  scheduler='Unchanged old Python build_jobs/Graph; gate-first phase/expert/node priority; full native chip lock command-to-output-channel-drain; source backend authoritative, new native backend not used.',
  hardware_contract=dict(chips=16,channels=8,ordinary_wave_ns=28000,native_wave_limit=192,controller_command_ns=250,host_submission_ns=2000,pcie_bytes_per_ns=8,channel_bytes_per_ns=2,input_element_bytes=1,output_element_bytes=4,activation_ns=0,combine_ns=0),
  analysis='Paired whole40-layer cost for each of16prompts×16decode positions, per-prompt means/P99, dataset groups, pooled mean/P99. P99 after summing40layers. No large-sample confidence/generalization claims from16prompts.',
  trace_selection='Retain first acquisition prompt decode0 for every layer/layout, chosen before service results.',workers=8,python=platform.python_version(),numpy=np.__version__,limitations=['Uncalibrated software FFN service model, not measured hardware or full-model wall time.','Only16fresh prompts and256within-prompt-correlated decode positions; no population-level confidence inference.','Quality sensitivity relativeQ4GGUF is separately disclosed; generated-token equality is placement equivalence, not task accuracy.','Whole-row geometry/complementary striping are existing-method controls, not a new method claim.'])
 write(DEST/'protocol.frozen.json',protocol)
 # Extract only after the analysis protocol is frozen; no result-driven filtering.
 routes=np.full((16,16,40,6),-1,dtype=np.int32)
 for pi,path in enumerate(paths):
  decode=[json.loads(l) for l in path.read_text().splitlines() if json.loads(l)['phase']=='decode'];assert len(decode)==640
  for r in decode:
   assert np.all(routes[pi,r['step'],r['layer']]==-1);assert len(set(r['expert_ids']))==6;routes[pi,r['step'],r['layer']]=r['expert_ids']
 assert np.all((routes>=0)&(routes<64));np.savez_compressed(DEST/'routes.npz',routes=routes,prompt_ids=np.asarray([j['id'] for j in items]),dataset_groups=np.asarray([j['kind'] for j in items]))
 write(DEST/'route_extraction.json',dict(shape=list(routes.shape),routes_npz_sha256=sha(DEST/'routes.npz'),raw_int32_sha256=hashlib.sha256(routes.tobytes()).hexdigest(),protocol_sha256=sha(DEST/'protocol.frozen.json')))
 records=[]
 with ProcessPoolExecutor(max_workers=8,initializer=initialize_worker) as pool:
  for future in as_completed([pool.submit(worker,(layer,case)) for layer in range(40) for case in CASES]):
   r=future.result();records.append(r)
   if len(records)%10==0:print('Completed layer/layout groups',len(records),'elapsed',round(time.monotonic()-began,2),flush=True)
 records.sort(key=lambda r:(r['layer'],r['case']));costs={case:np.stack([np.load(DEST/f'L{layer}.{case}.npz')['cost'] for layer in range(40)]).sum(axis=0) for case in CASES}
 for case,array in costs.items():np.save(DEST/f'{case}.sum40.npy',array)
 a,b=costs['frozen_phase_group'],costs['frozen_whole_rows'];delta=a-b
 perprompt=[]
 for pi,j in enumerate(items):
  perprompt.append(dict(prompt_index=pi,prompt_id=j['id'],dataset_group=j['kind'],dataset=j['dataset'],split=j['split'],phase_group=stats(a[pi]),whole_rows=stats(b[pi]),mean_reduction_fraction=1-float(b[pi].mean())/float(a[pi].mean()),paired_mean_delta_ns=float(delta[pi].mean()),paired_faster_tokens=int((delta[pi]>0).sum()),paired_tied_tokens=int((delta[pi]==0).sum()),paired_slower_tokens=int((delta[pi]<0).sum())))
 groups=[]
 for group in ['lm','arc']:
  ix=np.array([i for i,j in enumerate(items) if j['kind']==group]);aa,bb=a[ix],b[ix];groups.append(dict(dataset_group=group,prompts=len(ix),phase_group=stats(aa),whole_rows=stats(bb),mean_reduction_fraction=1-float(bb.mean())/float(aa.mean()),paired_mean_delta_ns=float((aa-bb).mean()),per_prompt_mean_reduction_fractions=[perprompt[i]['mean_reduction_fraction'] for i in ix]))
 with open(DEST/'paired_per_token.csv','w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['prompt_index','prompt_id','dataset_group','decode_step','phase_group_ns','whole_rows_ns','delta_ns','reduction_fraction']);w.writeheader()
  for pi,j in enumerate(items):
   for step in range(16):w.writerow(dict(prompt_index=pi,prompt_id=j['id'],dataset_group=j['kind'],decode_step=step,phase_group_ns=int(a[pi,step]),whole_rows_ns=int(b[pi,step]),delta_ns=int(delta[pi,step]),reduction_fraction=1-float(b[pi,step])/float(a[pi,step])))
 report=dict(status='complete',completed_utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(DEST/'protocol.frozen.json'),routes_npz_sha256=sha(DEST/'routes.npz'),wall_seconds=time.monotonic()-began,records=records,totals=[dict(case=c,**stats(v)) for c,v in costs.items()],mean_reduction_fraction=1-float(b.mean())/float(a.mean()),paired_mean_delta_ns=float(delta.mean()),paired_faster_tokens=int((delta>0).sum()),paired_tied_tokens=int((delta==0).sum()),paired_slower_tokens=int((delta<0).sum()),per_prompt=perprompt,dataset_groups=groups,limits=protocol['limitations'])
 assert all(sha(p)==h for p,h in hashes.items()) and all(sha(p)==h for p,h in sourcehashes.items())
 write(DEST/'summary.json',report);print(json.dumps({k:v for k,v in report.items() if k not in ['records','per_prompt']},indent=2),flush=True)
if __name__=='__main__':main()
