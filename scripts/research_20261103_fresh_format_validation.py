#!/usr/bin/env python3
"""Frozen fresh-input teacher quality + unchanged-binary placement validation."""
import argparse,collections,datetime,hashlib,json,math,os,pathlib,subprocess,time
import numpy as np
ROOT=pathlib.Path('/home/ericyu/expert_flash_experiment')
OUT=ROOT/'results/research-2026-11-02/round2/fresh-format-heldout'
OLD=ROOT/'results/research-2026-11-02/round2/full-ffn-integer-reference'
CPP=ROOT/'scripts/research_20261103_teacher_forced_quality.cpp'
SELF=ROOT/'scripts/research_20261103_fresh_format_validation.py'
QUALITY_BIN=OUT/'quality-build/teacher-forced-quality'
GEN_BIN=OLD/'build/full-ffn-integer-reference'
MODEL=pathlib.Path('/home/nfs_client/ericyu/expert_flash_experiment/models/granite4tiny-q4_k_m.gguf')
PLACEMENT=OLD/'placement.tsv'
ACQUISITION_PROTOCOL=OUT/'acquisition_protocol.frozen.json'
ACQUISITION_SHA='b9286837ae30920efe9da96eccd12ec4e6ef51e7453b2de6f0fd56ffa3dfe8e2'
PROJECTIONS=['gate_exps','up_exps','down_exps','gate_shexp','up_shexp','down_shexp']
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def jsonlines(p):return [json.loads(l) for l in p.read_text().splitlines()]
def contract_audit():
 old=(ROOT/'scripts/research_20261102_full_ffn_integer_reference.cpp').read_text();new=CPP.read_text()
 spans=[('quantize','static float quantize(','static int64_t dot('),('dot','static int64_t dot(','static float scale_output('),('scale_output','static float scale_output(','static bool equal_bits('),('equal_bits','static bool equal_bits(','static bool target('),('target','static bool target(','static const Cache &get_cache('),('weight_cache','static const Cache &get_cache(','struct Stats'),('W8A8_audit','static void audit(','struct Trace')]
 records=[]
 for label,begin,end in spans:
  a=old[old.index(begin):old.index(end,old.index(begin))].strip()
  newend='static void audit_source_gguf' if label=='W8A8_audit' else end
  b=new[new.index(begin):new.index(newend,new.index(begin))].strip();assert a==b,label
  records.append({'function':label,'old_new_byte_exact':True,'body_sha256':hashlib.sha256(a.encode()).hexdigest()})
 return records

def prepare(manifest_path):
 assert sha(ACQUISITION_PROTOCOL)==ACQUISITION_SHA,'Acquisition must already be frozen'
 manifest_path=manifest_path.resolve();manifest=json.loads(manifest_path.read_text());items=manifest['items'];assert len(items)==16
 lm=[j for j in items if j['kind']=='lm'];arc=[j for j in items if j['kind']=='arc'];assert len(lm)==len(arc)==8
 assert [j['kind'] for j in items]==['lm']*8+['arc']*8 and len({j['id'] for j in items})==16 and [j['id'][-2:] for j in items]==[f'{i:02}' for i in range(8)]*2
 for j in items:assert sha(j['prompt_path'])==j['prompt_sha256'],j['id']
 original=json.loads((OLD/'frozen_protocol.json').read_text());assert sha(OLD/'frozen_protocol.json')=='3163341827c25b210c42126ae7465c1fe8be9827aab93d314a89ef8a54975f40'
 assert sha(GEN_BIN)=='1bc6714863968385e8fb90d67b4a2cde7d72c6976aa2dcafe5bb26777c3e8037'
 assert sha(PLACEMENT)=='a8853dab14bfa25e0d36597dac191138c260600f942553cbe2932a37d4160154'
 assert sha(MODEL)=='5a38b08c441ae1adbafb1d2b8a7167e0d48734d83af68b268cefea1eec553dcd'
 immutable_contract=contract_audit();write(OUT/'quality-build/unchanged_numeric_contract.json',immutable_contract)
 build=json.loads((OUT/'quality-build/build_command.json').read_text());build.update(source_sha256=sha(CPP),binary_sha256=sha(QUALITY_BIN),compiler=subprocess.check_output(['c++','--version'],text=True).splitlines()[0]);write(OUT/'quality-build/build_command.json',build)
 required=[manifest_path,ACQUISITION_PROTOCOL,CPP,SELF,QUALITY_BIN,GEN_BIN,PLACEMENT,OLD/'frozen_protocol.json',OUT/'quality-build/unchanged_numeric_contract.json',OUT/'quality-build/build_command.json']
 required.extend(pathlib.Path(p) for p in original['input_sha256'] if p.endswith('.a') or '/sources/' in p)
 inputs={str(p):sha(p) for p in required};inputs[str(MODEL)]=sha(MODEL)
 for kind,jobs in [('quality',lm),('generation',items)]:
  dest=OUT/kind;dest.mkdir(exist_ok=True)
  if (dest/'protocol.frozen.json').exists():raise RuntimeError('Never overwrite frozen validation protocol')
  (dest/'jobs.tsv').write_text(''.join(f"{j['prompt_path']}\t{j['id']}\n" for j in jobs))
  protocol={'frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'kind':kind,'inputs_sha256':inputs,'jobs_tsv_sha256':sha(dest/'jobs.tsv'),'manifest_path':str(manifest_path),'jobs':[{'id':j['id'],'kind':j['kind'],'dataset':j['dataset'],'split':j['split'],'prompt_path':j['prompt_path'],'prompt_sha256':j['prompt_sha256']} for j in jobs],'threads':8,'input_order':'Acquisition-frozen lm00..07 then arc00..07; no output-based selection','numeric_contract':'Exactly prior fixed GGUF-to_float row W8[-127,127], whole-vector dynamic A8, exact I64 dot, one F32 rescale; old function bodies byte-exact','attention_router_ssm':'Source-GGUF host operators, same in both references','placement_refit':False,'quantizer_tuning':False,'quality_claim_limits':['Sensitivity relative to the configured Q4 GGUF reference; not original BF16 quality, certified task accuracy or hardware performance.','Fresh heldout examples are bounded8 WikiText2 test chunks plus8 ARC-Challenge validation questions; no population generalization is established.']}
  if kind=='quality':protocol.update(binary=str(QUALITY_BIN),modes=['source-gguf','unsharded'],tokens_per_chunk=128,scored_next_tokens_per_chunk=127,total_scored_next_tokens=1016,tokenization='llama_tokenize add_special=true,parse_special=true; original GGUF BOS rules; first128 tokens; no chat template',loss='F64 max-shift logsumexp over full F32 vocabulary logits; NLL=logsumexp-targetLogit; corpus PPL=exp(totalNLL/1016)',router='Same fixed128 teacher input tokens; ordered top6 identity plus set intersection/6 and Jaccard; all40 layers',repeat='Repeat first acquisition-frozen WikiText item W8 in an independently initialized context; compare all128 vocabulary-logit vectors bitwise',snapshots='First2 full vocabulary-logit vectors per mode/prompt; full128 logit vectors for first W8 prompt')
  else:protocol.update(binary=str(GEN_BIN),modes=['unsharded','placed'],decode_steps=16,prompt_scope='Full acquisition prompt unchanged, including complete ARC options; actual GGUF token counts reported, no silent truncation',purpose='Exact same-format full-FFN placement equivalence on fresh inputs; generation token matches are not quality scores',old_binary_unchanged=True,old_placement_unchanged=True)
  write(dest/'protocol.frozen.json',protocol)
 print('Frozen quality and generation protocols before either execution',flush=True)

def verify(kind):
 protocol=json.loads((OUT/kind/'protocol.frozen.json').read_text())
 for p,h in protocol['inputs_sha256'].items():assert sha(p)==h,p
 assert sha(OUT/kind/'jobs.tsv')==protocol['jobs_tsv_sha256']
 for job in protocol['jobs']:assert sha(job['prompt_path'])==job['prompt_sha256']
 return protocol

def run(kind):
 spec=verify(kind);dest=OUT/kind/'run'
 if dest.exists():raise RuntimeError('Never overwrite validation run')
 dest.mkdir();n=128 if kind=='quality' else spec['decode_steps'];argv=[spec['binary'],str(MODEL),str(OUT/kind/'jobs.tsv'),str(n),'8',str(PLACEMENT),str(dest)]
 write(dest/'run_command.json',{'argv':argv,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'protocol_sha256':sha(OUT/kind/'protocol.frozen.json')})
 t=time.monotonic()
 with open(dest/'stdout.log','w') as out,open(dest/'stderr.log','w') as err:r=subprocess.run(argv,stdout=out,stderr=err,env=dict(os.environ,OMP_NUM_THREADS='8',OMP_DYNAMIC='FALSE',OPENBLAS_NUM_THREADS='1'))
 write(dest/'execution.json',{'exit_code':r.returncode,'wall_seconds':time.monotonic()-t,'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
 if r.returncode:raise RuntimeError(f'{kind} execution failed; artifacts retained')
 verify(kind)
 if kind=='quality':audit_quality()
 else:audit_generation()

def geometry(row,T):
 p=row['projection'];C=512 if p=='down_exps' else 1024 if p=='down_shexp' else 1536;R=512 if p in ['gate_exps','up_exps'] else 1024 if p in ['gate_shexp','up_shexp'] else 1536;K=6 if p.endswith('_exps') else 1
 assert row['tokens']==T and row['selected_experts']==K and row['output_floats']==T*K*R
 assert row['weight_shape']==[C,R,64 if K==6 else 1,1]
 assert row['input_shape']==([C,6 if p=='down_exps' else 1,T,1] if K==6 else [C,T,1,1])
 assert row['output_shape']==([R,K,T,1] if K==6 else [R,T,1,1])

def route_comparison(a,b):
 assert len(a)==len(b);ordered=sets=intersection=0;jaccard=0
 for x,y in zip(a,b):
  assert (x['position'],x['layer'],x['token_id'])==(y['position'],y['layer'],y['token_id'])
  s,t=set(x['expert_ids']),set(y['expert_ids']);assert len(s)==len(t)==6
  ordered+=x['expert_ids']==y['expert_ids'];sets+=s==t;k=len(s&t);intersection+=k;jaccard+=k/len(s|t)
 n=len(a);return {'records':n,'ordered_exact':ordered,'ordered_exact_fraction':ordered/n,'set_exact':sets,'set_exact_fraction':sets/n,'mean_intersection_over6':intersection/(6*n),'mean_jaccard':jaccard/n}

def audit_quality():
 spec=verify('quality');dest=OUT/'quality/run';items=spec['jobs'];rows=jsonlines(dest/'coverage_ledger.jsonl');bykey=collections.defaultdict(list)
 for r in rows:geometry(r,128);bykey[(r['mode'],r['job'],r['weight'])].append(r)
 expected_jobs={mode:[j['id'] for j in items]+([items[0]['id']+'.repeat'] if mode=='unsharded' else []) for mode in ['source-gguf','unsharded']}
 for mode,jobs in expected_jobs.items():
  for job in jobs:
   for layer in range(40):
    for p in PROJECTIONS:assert len(bykey[(mode,job,f'blk.{layer}.ffn_{p}.weight')])==1
 assert len(rows)==240*17 and len(bykey)==len(rows)
 sums={'source-gguf':0.,'unsharded':0.};top_matches=0;scored=0;all_routes={m:[] for m in sums};perprompt=[];snapshot_checks=[]
 for job in items:
  id=job['id'];summaries={m:json.loads((dest/'traces'/f'{id}.{m}.summary.json').read_text()) for m in sums};positions={m:jsonlines(dest/'traces'/f'{id}.{m}.positions.jsonl') for m in sums}
  assert summaries['source-gguf']['token_ids']==summaries['unsharded']['token_ids']
  for mode,s in summaries.items():
   assert s['input_tokens']==128 and s['scored_next_tokens']==127 and s['finite_logits']==128*s['vocabulary'];assert len(positions[mode])==127
   summed=math.fsum(p['nll_F64'] for p in positions[mode]);assert abs(summed-s['sum_nll_F64'])<1e-9;sums[mode]+=summed
   for i,p in enumerate(positions[mode]):assert p['position']==i and p['input_token_id']==s['token_ids'][i] and p['target_token_id']==s['token_ids'][i+1] and math.isfinite(p['nll_F64']) and abs(p['nll_F64']-(p['logsumexp_F64']-p['target_logit']))<1e-12
   snapshot=np.fromfile(dest/'traces'/f'{id}.{mode}.first2_logits.f32',dtype='<f4').reshape(2,s['vocabulary']);assert np.isfinite(snapshot).all()
   errors=[]
   for i,logits in enumerate(snapshot.astype(np.float64)):
    mx=logits.max();lse=float(mx+np.log(np.exp(logits-mx).sum(dtype=np.float64)));p=positions[mode][i];errors.append(abs(lse-p['logsumexp_F64']));assert errors[-1]<1e-10 and int(np.argmax(logits))==p['top1_token_id'] and float(logits[p['target_token_id']])==p['target_logit']
   snapshot_checks.append({'job':id,'mode':mode,'first2_F64_logsumexp_max_abs_difference':max(errors),'full_logits_finite_count':s['finite_logits']})
   routes=jsonlines(dest/'traces'/f'{id}.{mode}.routing.jsonl');assert len(routes)==5120
   assert {(r['position'],r['layer']) for r in routes}=={(p,l) for p in range(128) for l in range(40)}
   # Add job identity for aggregation while preserving pair order.
   all_routes[mode].extend(routes)
  a,b=positions['source-gguf'],positions['unsharded'];agree=sum(x['top1_token_id']==y['top1_token_id'] for x,y in zip(a,b));top_matches+=agree;scored+=127
  ma=summaries['source-gguf']['mean_nll_F64'];mb=summaries['unsharded']['mean_nll_F64']
  perprompt.append({'job':id,'scored_tokens':127,'source_gguf_mean_NLL':ma,'full_FFN_W8A8_mean_NLL':mb,'delta_NLL_W8_minus_GGUF':mb-ma,'perplexity_ratio_W8_over_GGUF':math.exp(mb-ma),'top1_agreement':agree,'router':route_comparison(jsonlines(dest/'traces'/f'{id}.source-gguf.routing.jsonl'),jsonlines(dest/'traces'/f'{id}.unsharded.routing.jsonl'))})
 first_id=items[0]['id'];original=dest/'traces'/f'{first_id}.unsharded.summary.json';repeat=dest/'traces'/f'{first_id}.repeat.unsharded.summary.json';a=json.loads(original.read_text());b=json.loads(repeat.read_text());assert a['all_logits_fnv1a64']==b['all_logits_fnv1a64'] and b['repeat_logits_bit_mismatches']==0 and a['token_ids']==b['token_ids']
 for suffix in ['positions.jsonl','routing.jsonl','first2_logits.f32']:assert (dest/'traces'/f'{first_id}.unsharded.{suffix}').read_bytes()==(dest/'traces'/f'{first_id}.repeat.unsharded.{suffix}').read_bytes()
 for layer in range(40):
  for p in PROJECTIONS:
   x=bykey[('unsharded',first_id,f'blk.{layer}.ffn_{p}.weight')][0];y=bykey[('unsharded',first_id+'.repeat',f'blk.{layer}.ffn_{p}.weight')][0];assert x['output_fnv1a64']==y['output_fnv1a64'] and x['input_fnv1a64']==y['input_fnv1a64']
 assert scored==1016
 means={m:v/scored for m,v in sums.items()};delta=means['unsharded']-means['source-gguf'];router=route_comparison(all_routes['source-gguf'],all_routes['unsharded']);bylayer=[]
 for layer in range(40):bylayer.append({'layer':layer,**route_comparison([r for r in all_routes['source-gguf'] if r['layer']==layer],[r for r in all_routes['unsharded'] if r['layer']==layer])})
 result={'status':'PASS','purpose':'Fixed-teacher quality sensitivity relative to Q4 GGUF; no original-BF16/task-accuracy/hardware inference','prompts':8,'input_tokens_per_prompt':128,'scored_next_tokens':scored,'source_gguf_mean_NLL_F64':means['source-gguf'],'full_FFN_W8A8_mean_NLL_F64':means['unsharded'],'delta_NLL_W8_minus_GGUF':delta,'source_gguf_perplexity':math.exp(means['source-gguf']),'full_FFN_W8A8_perplexity':math.exp(means['unsharded']),'perplexity_ratio_W8_over_GGUF':math.exp(delta),'teacher_top1_agreement_count':top_matches,'teacher_top1_agreement_fraction':top_matches/scored,'full_fixed_input_router_comparison':router,'router_per_layer':bylayer,'per_prompt':perprompt,'coverage_records':len(rows),'all40layers_all6projections_all128tokens_covered':True,'all_logits_finite':True,'independent_context_first_W8_prompt_bitwise_repeat':True,'repeat_full_vocabulary_logits_compared':a['finite_logits'],'snapshot_numeric_checks':snapshot_checks,'protocol_sha256':sha(OUT/'quality/protocol.frozen.json')}
 write(OUT/'quality/summary.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['router_per_layer','per_prompt','snapshot_numeric_checks']},indent=2),flush=True)

def audit_generation():
 spec=verify('generation');dest=OUT/'generation/run';report=json.loads((dest/'numeric_report.json').read_text());rows=jsonlines(dest/'coverage_ledger.jsonl');groups={m:[r for r in rows if r['mode']==m] for m in ['unsharded','placed']};assert len(groups['unsharded'])==len(groups['placed']);fields=['job','phase','step','weight','layer','projection','op','weight_shape','input_shape','output_shape','tokens','selected_experts','output_floats','input_fnv1a64','expert_ids_fnv1a64','output_fnv1a64']
 for a,b in zip(groups['unsharded'],groups['placed']):assert all(a[k]==b[k] for k in fields)
 totals=collections.Counter();coverage=collections.Counter();details=[]
 for r in rows:geometry(r,r['tokens']);totals[r['mode'],r['job'],r['phase'],r['weight']]+=r['tokens'];coverage[r['mode'],r['job'],r['phase'],r['weight']]+=1
 for job in spec['jobs']:
  id=job['id'];a=json.loads((dest/'traces'/f'{id}.unsharded.summary.json').read_text());b=json.loads((dest/'traces'/f'{id}.placed.summary.json').read_text());n=a['prompt_tokens'];assert a['prompt_token_ids']==b['prompt_token_ids'] and a['generated_token_ids']==b['generated_token_ids'] and a['decoded_tokens']==16
  ra=dest/'traces'/f'{id}.unsharded.routing.jsonl';rb=dest/'traces'/f'{id}.placed.routing.jsonl';assert ra.read_bytes()==rb.read_bytes();routes=jsonlines(ra);assert len(routes)==40*(n+16)
  for mode in groups:
   for layer in range(40):
    for projection in PROJECTIONS:
     w=f'blk.{layer}.ffn_{projection}.weight';assert totals[mode,id,'prefill',w]==n and totals[mode,id,'decode',w]==16 and coverage[mode,id,'prefill',w]==(n+511)//512 and coverage[mode,id,'decode',w]==16
  snapshots=[]
  for suffix in ['ffn_outputs','logits']:
   p=dest/'traces'/f'{id}.unsharded.{suffix}.f32';array=np.memmap(p,dtype='<f4',mode='r');assert np.isfinite(array).all();snapshots.append({'kind':suffix,'float_count':array.size,'sha256':sha(p)})
  details.append({'id':id,'kind':job['kind'],'prefill_tokens':n,'decode_tokens':16,'routes_byte_exact':True,'route_records_per_mode':len(routes),'route_sha256':sha(ra),'generated_tokens_identical':True,'snapshots':snapshots})
 for key in ['integer_mismatches','same_input_scaled_bit_mismatches','cross_run_output_bit_mismatches','cross_run_logits_bit_mismatches']:assert report[key]==0
 assert len(coverage)==2*16*2*240
 result={'status':'PASS','fresh_prompts':16,'WikiText_test_chunks':8,'ARC_Challenge_validation_questions':8,'old_binary_unchanged':True,'old_placement_unchanged':True,'placement_refit':False,'all40layers_all6projections_prefill_and_decode_replaced':True,'all_routes_and_generated_token_ids_identical':True,'coverage_callback_records':len(rows),'coverage_keys':len(coverage),'unique_placed_row_outputs':report['integer_checks'],'same_input_I64_comparisons':report['integer_checks'],'cross_run_F32_scalar_comparisons':report['cross_run_output_floats'],'cross_run_vocabulary_logits_compared':report['cross_run_logits_floats'],'integer_mismatches':0,'cross_run_F32_bit_mismatches':0,'cross_run_logit_bit_mismatches':0,'actual_total_prefill_tokens':sum(d['prefill_tokens'] for d in details),'actual_total_decode_tokens_per_mode':256,'details':details,'protocol_sha256':sha(OUT/'generation/protocol.frozen.json'),'limits':['Counts are same-format placement equivalence, not generation-quality scores.','No hardware/native-AiF ABI/performance claim.']}
 write(OUT/'generation/summary.json',result);print(json.dumps({k:v for k,v in result.items() if k!='details'},indent=2),flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','quality','generation','audit-quality','audit-generation']);p.add_argument('--manifest',type=pathlib.Path,default=OUT/'manifest.json');a=p.parse_args()
 if a.action=='prepare':prepare(a.manifest)
 elif a.action=='audit-quality':audit_quality()
 elif a.action=='audit-generation':audit_generation()
 else:run(a.action)
if __name__=='__main__':main()
