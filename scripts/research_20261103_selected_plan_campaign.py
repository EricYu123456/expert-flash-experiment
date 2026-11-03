#!/usr/bin/env python3
"""Bounded four-view selected compiler-plan numeric validation campaign."""
import argparse,collections,hashlib,json,os,pathlib,signal,subprocess,time
import numpy as np
import research_20261103_selected_plan_numeric as base
ROOT=base.ROOT;BASE=base.BASE;OUT=base.OUT;SELF=pathlib.Path(__file__).resolve();SEAL=BASE/'ordinary-read-compiler/W8_selected_plans.frozen.json';EXPECTED_SEAL='00507ff56f9197fb18511ad58303919214c03cba6f0c1440f6733ffc42944d00';AUDIT=ROOT/'scripts/research_20261103_selected_plan_numeric_audit.py'
sha=base.sha;dump=base.dump

def freeze(phase,manifest=None):
 assert sha(SEAL)==EXPECTED_SEAL;seal=json.loads(SEAL.read_text());dest=OUT/('smoke-primary' if phase=='smoke' else 'second-heldout');dest.mkdir(exist_ok=True)
 if (dest/'protocol.frozen.json').exists():raise RuntimeError('Never overwrite campaign freeze')
 inputs=[SELF,pathlib.Path(base.__file__),AUDIT,SEAL,OUT/'build_inventory.json',OUT/'numeric_helper_identity.json'];inventory=json.loads((OUT/'build_inventory.json').read_text())
 if phase=='smoke':
  corpus=base.NFS/'runs/corpus-300-v1/corpus/prompts';prompts=[dict(id=p.stem,prompt_path=str(p),dataset='existingTRAIN',split='train',prompt_sha256=sha(p)) for p in [corpus/'train_programming_short_015.txt',corpus/'train_summarization_short_027.txt']];steps=4;budget=600
 else:
  prior=json.loads((OUT/'smoke-primary/summary.json').read_text());assert prior['status']=='complete' and prior['independent_audits_passed']
  path=pathlib.Path(manifest).resolve();payload=json.loads(path.read_text());prompts=payload['items'];assert len(prompts)==16 and len({p['id'] for p in prompts})==16;steps=16;budget=1800;inputs.append(path)
  assert len(collections.Counter(p['dataset'] for p in prompts))==2 and sorted(collections.Counter(p['dataset'] for p in prompts).values())==[8,8]
  for p in prompts:assert sha(p['prompt_path'])==p['prompt_sha256']
 inputs += [pathlib.Path(p['prompt_path']) for p in prompts];cases={}
 for campaign in seal['campaigns']:
  for field in ['summary','protocol','snapshot_manifest']:
   path=ROOT/campaign[field];assert sha(path)==campaign[field+'_sha256'];inputs.append(path)
  plans=(ROOT/campaign['summary']).parent;profile='input'+str(campaign['input_channel_copies'])
  for name,cfg in base.CFG.items():
   caseid=profile+'-'+name;d=dest/caseid;d.mkdir();bands=[];ledger=[];hashes={}
   for index,layer in enumerate(cfg['layers']):
    path=plans/f'{name}.L{index}.cmd250/whole_rows.plan.json';plan=json.loads(path.read_text());assert plan['model']==name and plan['layer']==index and plan['service']['command_ns']==250 and plan['selection']['family']=='whole_rows';bb,ll=base.lower_layer(plan,dict(cfg,layers=list(range(len(cfg['layers'])))));bb=[(layer,*b[1:]) for b in bb];ll=[dict(b,layer=layer,compiler_moe_index=index) for b in ll];bands.extend(bb);ledger.extend(ll);hashes[str(path)]=sha(path);inputs.append(path)
   if cfg['dense']:
    for p,C,R in [('gate',cfg['H'],cfg['dense']),('up',cfg['H'],cfg['dense']),('down',cfg['dense'],cfg['H'])]:
     for chip in range(16):bands.append((0,p,0,C,R,chip,chip*(R//16),R//16))
   tsv=d/'physical_rows.tsv';tsv.write_text(''.join('\t'.join(map(str,row))+'\n' for row in bands));dump(d/'mapping_ledger.json',ledger);jobs=d/'jobs.tsv';jobs.write_text(''.join(f"{p['prompt_path']}\t{p['id']}\n" for p in prompts));binary=inventory['models'][name]['binary'];model=dict(model=str(cfg['model']),binary=binary,rows_tsv=str(tsv),jobs_tsv=str(jobs),plans_sha256=hashes,raw_layers=cfg['layers'],compiler_to_raw_layer=cfg['layers']);cases[caseid]=dict(name=name,profile=profile,models={name:model},prompts=[dict(p,path=p['prompt_path']) for p in prompts],steps=steps,threads=8,plan_seal_sha256=EXPECTED_SEAL,canonical_scope='All FFN projections, including shared and DeepSeek dense raw0',native_address_claim=False)
   dump(d/'protocol.frozen.json',cases[caseid]);inputs += [cfg['source'],cfg['build'],cfg['protocol'],cfg['model'],pathlib.Path(inventory['models'][name]['source']),pathlib.Path(binary),tsv,jobs,d/'mapping_ledger.json',d/'protocol.frozen.json']
 inputs += [pathlib.Path(x) for x in json.loads(base.CFG['granite4tiny']['build'].read_text())['argv'] if x.endswith('.a')]
 spec=dict(phase=phase,steps=steps,threads=8,max_processes=4,budget_seconds=budget,frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),inputs_sha256={str(p):sha(p) for p in set(inputs)},cases=cases,prompts=prompts,plan_seal_sha256=EXPECTED_SEAL,output_policy='Full reference F32 FFN/logit streams retained for exact comparison, all callbacks/scale/route ledgers retained. No prompt truncation/reselection. Canonical is separately rerun under each layout profile; no cross-layout state reuse.',numeric_scope='Fixed original-GGUF fullrow W8 and whole-vector dynamicA8 CPU reference; only physical band packing/scatter changes. Host attention/router/SSM unchanged. DeepSeek raw0dense fixed equal16, sharedI2816single branch/down. No quantizer tuning, quality or native-performance claim.',partial_policy='Stop process groups at budget; completed paired prompts exported with completed mask; incomplete slots=-1. Preserve all partial logs and never shorten/reselect.')
 dump(dest/'protocol.frozen.json',spec);print(json.dumps({'phase':phase,'status':'frozen','protocol_sha256':sha(dest/'protocol.frozen.json'),'cases':list(cases),'prompts':len(prompts)},indent=2))

def verify(spec):
 for p,h in spec['inputs_sha256'].items():assert sha(p)==h,p

def execute(phase):
 dest=OUT/('smoke-primary' if phase=='smoke' else 'second-heldout');spec=json.loads((dest/'protocol.frozen.json').read_text());verify(spec)
 if (dest/'started.json').exists():raise RuntimeError('Never overwrite executed campaign')
 dump(dest/'started.json',dict(protocol_sha256=sha(dest/'protocol.frozen.json'),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())));active={};began=time.monotonic();finished=[];deadline=began+spec['budget_seconds'];next_progress=began+30
 for caseid,case in spec['cases'].items():
  name=case['name'];cfg=case['models'][name];d=dest/caseid/'run';d.mkdir();argv=[cfg['binary'],cfg['model'],cfg['jobs_tsv'],str(spec['steps']),'8',cfg['rows_tsv'],str(d)];dump(d/'run_command.json',dict(argv=argv,case_protocol_sha256=sha(dest/caseid/'protocol.frozen.json')));out=(d/'stdout.log').open('w');err=(d/'stderr.log').open('w');p=subprocess.Popen(argv,stdout=out,stderr=err,env=dict(os.environ,OMP_NUM_THREADS='8',OMP_DYNAMIC='FALSE',OPENBLAS_NUM_THREADS='1'),start_new_session=True);active[p.pid]=(p,caseid,out,err,time.monotonic())
 while active:
  for pid,(p,caseid,out,err,start) in list(active.items()):
   code=p.poll()
   if code is not None:out.close();err.close();entry=dict(case=caseid,exit_code=code,wall_seconds=time.monotonic()-start);finished.append(entry);dump(dest/caseid/'execution.json',entry);del active[pid];print('CASE_COMPLETE',json.dumps(entry),flush=True)
  now=time.monotonic()
  if now>=next_progress:
   counts={c:len(list((dest/c/'run/traces').glob('*.placed.summary.json'))) for c in spec['cases']};print('PROGRESS',json.dumps(dict(elapsed_seconds=round(now-began,1),completed_paired_prompts=counts,active=[v[1] for v in active.values()])),flush=True);next_progress=now+30
  if now>=deadline:
   for p,*_ in active.values():os.killpg(p.pid,signal.SIGTERM)
   for p,caseid,out,err,start in active.values():
    try:p.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
    out.close();err.close();entry=dict(case=caseid,exit_code='BUDGET_STOP',wall_seconds=time.monotonic()-start);finished.append(entry);dump(dest/caseid/'execution.json',entry)
   active.clear();break
  if active:time.sleep(2)
 dump(dest/'execution.json',dict(cases=finished,wall_seconds=time.monotonic()-began,budget_seconds=spec['budget_seconds']));verify(spec);extract(phase)

def extract(phase):
 dest=OUT/('smoke-primary' if phase=='smoke' else 'second-heldout');spec=json.loads((dest/'protocol.frozen.json').read_text());N=len(spec['prompts']);steps=spec['steps'];results={};canonical={}
 for caseid,case in spec['cases'].items():
  name=case['name'];layers=case['models'][name]['raw_layers'];routes=np.full((N,steps,len(layers),6),-1,dtype=np.int32);done=np.zeros(N,dtype=np.bool_);prefills=[];offsets=[0];tokenids=[];perprompt=[]
  for i,p in enumerate(spec['prompts']):
   run=dest/caseid/'run/traces';pid=p['id'];stems=[run/f'{pid}.{m}' for m in ['unsharded','placed']];paths=[pathlib.Path(str(stem)+'.summary.json') for stem in stems]
   if not all(x.exists() for x in paths):offsets.append(offsets[-1]);continue
   sums=[json.loads(x.read_text()) for x in paths];a,b=sums;n=a['prompt_tokens'];assert a['prompt_token_ids']==b['prompt_token_ids'] and a['generated_token_ids']==b['generated_token_ids'] and a['decoded_tokens']==b['decoded_tokens']==steps
   rpaths=[pathlib.Path(str(s)+'.routing.jsonl') for s in stems];assert rpaths[0].read_bytes()==rpaths[1].read_bytes();full=np.full((n,len(layers),6),-1,dtype=np.int32);seen=set()
   for raw in rpaths[0].read_text().splitlines():
    x=json.loads(raw);li=layers.index(x['layer']);e=x['expert_ids'];assert len(set(e))==6 and all(0<=v<64 for v in e)
    if x['phase']=='prefill':t=x['position'];assert 0<=t<n and x['token_id']==a['prompt_token_ids'][t];full[t,li]=e;key=('prefill',t,li)
    else:t=x['step'];assert x['phase']=='decode' and 0<=t<steps and x['position']==n+t and x['token_id']==a['generated_token_ids'][t];routes[i,t,li]=e;key=('decode',t,li)
    assert key not in seen;seen.add(key)
   assert np.all(full>=0) and np.all(routes[i]>=0) and len(seen)==(n+steps)*len(layers);done[i]=True;prefills.append(full);offsets.append(offsets[-1]+n);tokenids.extend(a['prompt_token_ids']);perprompt.append(dict(id=pid,prefill_tokens=n,generated_token_ids=a['generated_token_ids'],route_sha256=sha(rpaths[0])))
   logs=pathlib.Path(str(stems[0])+'.logits.f32');fingerprint=(sha(rpaths[0]),sha(logs),tuple(a['generated_token_ids']))
   if (name,pid) in canonical:assert canonical[name,pid]==fingerprint,'Across input1/input4 layouts numeric routes/logits must match'
   else:canonical[name,pid]=fingerprint
  d=dest/caseid;np.savez_compressed(d/'decode.routes.npz',routes=routes,layers=np.asarray(layers,dtype=np.int32),prompt_ids=np.asarray([p['id'] for p in spec['prompts']]),datasets=np.asarray([p['dataset'] for p in spec['prompts']]),completed=done,split=np.asarray('TRAIN_smoke' if phase=='smoke' else 'second_heldout'))
  np.savez_compressed(d/'prefill.routes.npz',routes=np.concatenate(prefills) if prefills else np.empty((0,len(layers),6),dtype=np.int32),prompt_offsets=np.asarray(offsets),token_ids=np.asarray(tokenids,dtype=np.int32),layers=np.asarray(layers,dtype=np.int32),prompt_ids=np.asarray([p['id'] for p in spec['prompts']]),completed=done)
  results[caseid]=dict(status='complete' if done.all() else 'partial',completed_prompts=int(done.sum()),decode_shape=list(routes.shape),prefill_tokens=int(offsets[-1]),decode_archive_sha256=sha(d/'decode.routes.npz'),prefill_archive_sha256=sha(d/'prefill.routes.npz'),prompts=perprompt)
 status='complete' if all(x['status']=='complete' for x in results.values()) else 'partial';dump(dest/'summary.json',dict(status=status,independent_audits_passed=False,all_views_same_routes_and_continuation_logits=True,protocol_sha256=sha(dest/'protocol.frozen.json'),cases=results));print(json.dumps({k:{key:v for key,v in x.items() if key!='prompts'} for k,x in results.items()},indent=2))

def audit(phase):
 import research_20261103_selected_plan_numeric_audit as independent
 dest=OUT/('smoke-primary' if phase=='smoke' else 'second-heldout');summary=json.loads((dest/'summary.json').read_text());assert summary['status']=='complete';audits={}
 for caseid in summary['cases']:
  independent.DEST=dest/caseid;independent.main();audits[caseid]=sha(dest/caseid/'independent/audit.json')
 # Only the audit-status summary is finalized; frozen inputs and arrays are unchanged.
 summary['independent_audits_passed']=True;summary['independent_audit_sha256']=audits;dump(dest/'summary.json',summary);print(json.dumps({'status':'pass','cases':audits},indent=2))

def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run','extract','audit']);p.add_argument('--phase',choices=['smoke','fresh'],required=True);p.add_argument('--manifest');a=p.parse_args()
 if a.action=='freeze':freeze(a.phase,a.manifest)
 elif a.action=='run':execute(a.phase)
 elif a.action=='extract':extract(a.phase)
 else:audit(a.phase)
if __name__=='__main__':main()
