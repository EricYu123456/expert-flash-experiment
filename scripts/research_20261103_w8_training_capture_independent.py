#!/usr/bin/env python3
"""Read-only independent mapping/count/shape audit of frozen TRAIN W8 capture."""
import collections,hashlib,json,pathlib
import numpy as np
ROOT=pathlib.Path('/home/ericyu/expert_flash_experiment')
OUT=ROOT/'results/research-2026-11-02/round2/w8-format-training'
def digest(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for block in iter(lambda:f.read(4<<20),b''):h.update(block)
 return h.hexdigest()
def main():
 spec=json.loads((OUT/'protocol.frozen.json').read_text());selection=json.loads((OUT/'selection_protocol.frozen.json').read_text());prompts=spec['prompts'];assert len(prompts)==32
 # Reconstruct seeded balanced selection using filename metadata only.
 rows=json.loads((OUT/'eligible_train_filename_metadata.json').read_text());seed='20261103'
 rank=lambda domain,value:hashlib.sha256((seed+'\0'+domain+'\0'+value).encode()).hexdigest()
 tasks=sorted({x['task'] for x in rows},key=lambda t:(rank('task',t),t));buckets={t:sorted((x for x in rows if x['task']==t),key=lambda x:(rank('prompt',x['id']),x['id'])) for t in tasks};chosen=[]
 for k in range(6):
  for t in tasks:
   if len(chosen)<32:chosen.append(buckets[t][k]['id'])
 assert chosen==[x['id'] for x in prompts] and dict(collections.Counter(x['task'] for x in prompts))==selection['selected_task_counts']
 assert all(x['id'].startswith('train_') and x['split']=='train' for x in prompts)
 for x in prompts:assert digest(x['path'])==x['prompt_sha256']
 report={'status':'pass','audit_source_sha256':digest(__file__),'protocol_sha256':digest(OUT/'protocol.frozen.json'),'selected32_IDs_independently_reconstructed':True,'test_or_fresh_text_opened':False,'models':{}}
 for model,model_spec in spec['models'].items():
  dest=OUT/model;dec=dict(np.load(dest/'decode.routes.npz',allow_pickle=False));pre=dict(np.load(dest/'prefill.routes.npz',allow_pickle=False));summary=json.loads((dest/'summary.json').read_text());L=len(model_spec['routed_raw_layers']);layers=model_spec['routed_raw_layers'];mask=dec['completed'];assert np.array_equal(mask,pre['completed']) and mask.shape==(32,)
  assert dec['routes'].shape==(32,16,L,6) and pre['routes'].shape[1:]==(L,6) and np.array_equal(dec['layers'],layers) and np.array_equal(pre['layers'],layers)
  assert list(dec['prompt_ids'])==chosen==list(pre['prompt_ids']) and np.array_equal(dec['tasks'],[p['task'] for p in prompts]) and np.array_equal(dec['length_buckets'],[p['length_bucket'] for p in prompts])
  offs=pre['prompt_offsets'];assert len(offs)==33 and offs[0]==0 and offs[-1]==len(pre['routes'])==len(pre['token_ids'])
  vocab=102400 if model=='deepseek' else 100352
  names={f'blk.{l}.ffn_{p}.weight' for l in layers for p in ['gate_exps','up_exps','down_exps','gate_shexp','up_shexp','down_shexp']}
  if model=='deepseek':names|={f'blk.0.ffn_{p}.weight' for p in ['gate','up','down']}
  ledger={};batch_finite={};total_logits=0;total_routes=0;total_outputs=0;counts=collections.Counter();snapshot_hashes={};all_inputs={str(OUT/'protocol.frozen.json'):digest(OUT/'protocol.frozen.json')};weight_keys={};cache_records=0;duplicate_cache_checks=0
  for b in range(4):
   p=dest/f'batch{b}/coverage_ledger.jsonl'
   if p.exists():
    all_inputs[str(p)]=digest(p)
    with p.open() as f:
     for raw in f:
      try:x=json.loads(raw)
      except json.JSONDecodeError:continue
      ledger.setdefault(x['job'],[]).append(x)
   cachepath=dest/f'batch{b}/weight_cache_ledger.jsonl'
   if cachepath.exists():
    all_inputs[str(cachepath)]=digest(cachepath)
    for raw in cachepath.read_text().splitlines():
     try:w=json.loads(raw)
     except json.JSONDecodeError:continue
     key=(w['weight'],w['expert']);fingerprint=tuple(w[k] for k in ['columns','rows','source_type','source_byte_offset','canonical_bytes_fnv1a64','row_scales_fnv1a64']);cache_records+=1
     if key in weight_keys:assert weight_keys[key]==fingerprint;duplicate_cache_checks+=1
     else:weight_keys[key]=fingerprint
   p=dest/f'batch{b}/capture_report.json'
   if p.exists():batch_finite[b]=json.loads(p.read_text());all_inputs[str(p)]=digest(p)
  for i,p in enumerate(prompts):
   if not mask[i]:assert np.all(dec['routes'][i]==-1) and offs[i]==offs[i+1];continue
   stem=dest/f"batch{i%4}/traces/{p['id']}.unsharded";summary_path=pathlib.Path(str(stem)+'.summary.json');s=json.loads(summary_path.read_text());n=s['prompt_tokens'];assert n==len(s['prompt_token_ids'])==offs[i+1]-offs[i] and s['decoded_tokens']==16 and s['prefill_replaced'] and not s['prefill_state_reused']
   assert np.array_equal(pre['token_ids'][offs[i]:offs[i+1]],s['prompt_token_ids'])
   routepath=pathlib.Path(str(stem)+'.routing.jsonl');seen=set()
   with routepath.open() as f:
    for raw in f:
     x=json.loads(raw);l=layers.index(x['layer']);k=x['expert_ids'];assert len(set(k))==6 and all(0<=v<64 for v in k)
     if x['phase']=='prefill':
      pos=x['position'];key=('prefill',pos,l);assert x['step']==-1 and 0<=pos<n and x['token_id']==s['prompt_token_ids'][pos];arr=pre['routes'][offs[i]+pos,l]
     else:
      step=x['step'];key=('decode',step,l);assert x['phase']=='decode' and 0<=step<16 and x['position']==n+step and x['token_id']==s['generated_token_ids'][step];arr=dec['routes'][i,step,l]
     assert key not in seen and np.array_equal(arr,k);seen.add(key);total_routes+=1
   assert len(seen)==(n+16)*L
   all_inputs[str(summary_path)]=digest(summary_path);all_inputs[str(routepath)]=digest(routepath)
   per=collections.Counter()
   for x in ledger[p['id']]:
    assert x['weight'] in names and x['mode']=='unsharded' and x['integer_mismatches']==x['scaled_bit_mismatches']==x['cross_run_output_bit_mismatches']==0
    phase=x['phase'];assert phase in ['prefill','decode'];C,R,E,W=x['weight_shape'];K=6 if E==64 else 1;T=x['tokens'];assert W==1 and x['selected_experts']==K
    down=x['projection'].startswith('down');shared='shexp' in x['projection'];dense=model=='deepseek' and x['layer']==0
    if model=='deepseek':expected=(10944,2048) if dense and down else (2048,10944) if dense else (2816,2048) if shared and down else (2048,2816) if shared else (1408,2048) if down else (2048,1408)
    else:expected=(512,1536) if down and not shared else (1024,1536) if down else (1536,1024) if shared else (1536,512)
    assert (C,R)==expected,(model,x['weight'],(C,R),expected)
    assert x['output_floats']==T*K*R
    if K==6:assert x['op']=='MUL_MAT_ID' and x['input_shape'][0]==C and x['input_shape'][1] in [1,6] and x['input_shape'][2:]==[T,1] and x['output_shape']==[R,6,T,1]
    else:assert x['op']=='MUL_MAT' and x['input_shape']==[C,T,1,1] and x['output_shape']==[R,T,1,1]
    assert phase!='decode' or T==1;per[phase,x['weight']]+=T;total_outputs+=x['output_floats'];counts['dense' if dense else 'shared' if shared else 'routed']+=x['output_floats']
   for w in names:assert per['prefill',w]==n and per['decode',w]==16
   for suffix in ['prefill.-1','decode.0']:
    path=dest/f"batch{i%4}/snapshots/{p['id']}.{suffix}.logits.f32";v=np.fromfile(path,dtype=np.float32);assert len(v)==vocab and np.all(np.isfinite(v));snapshot_hashes[str(path)]=digest(path)
   total_logits+=17*vocab
  for b,cap in batch_finite.items():
   ids={p['id'] for p in prompts[b::4]};rows=sum(x['output_floats'] for pid,entries in ledger.items() if pid in ids for x in entries);assert cap['finite_FFN_values_checked']==rows and cap['continuation_logit_values_checked_finite']==8*17*vocab and cap['full_FFN_outputs_all_tokens_finite'] and cap['new_capture_routed_rows_paired_with_placed'] is False and set(cap['prompt_logit_stream_hashes'])==ids
  outputs_per_token=982400 if model=='deepseek' else 757760
  assert total_outputs==(int(offs[-1])+int(mask.sum())*16)*outputs_per_token
  assert total_outputs==sum(summary['FFN_output_rows_by_kind_completed_prompts'].values()) and counts==summary['FFN_output_rows_by_kind_completed_prompts']
  report['models'][model]={'status':summary['status'],'completed_prompts':int(mask.sum()),'raw_layers':layers,'decode_shape':list(dec['routes'].shape),'prefill_tokens':int(offs[-1]),'route_records_independently_compared':total_routes,'FFN_outputs_finite_by_instrumented_callback_completed_prompts':total_outputs,'full_FFN_weight_names':len(names),'output_rows_by_kind':dict(counts),'unique_FFN_scalar_outputs_per_prefill_or_decode_token':outputs_per_token,'continuation_logit_finite_values_complete_prompts':total_logits,'complete_batch_finite_reports':len(batch_finite),'cache_records':cache_records,'unique_weight_expert_keys':len(weight_keys),'cross_batch_identical_fixed_weight_and_scale_checks':duplicate_cache_checks,'snapshot_hashes':snapshot_hashes,'input_sha256':all_inputs,'decode_archive_sha256':digest(dest/'decode.routes.npz'),'prefill_archive_sha256':digest(dest/'prefill.routes.npz')}
 path=OUT/'independent';path.mkdir(exist_ok=True);target=path/'audit.json';assert not target.exists(),'Do not overwrite sealed audit';target.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:{key:v for key,v in m.items() if key not in ['input_sha256','snapshot_hashes']} for k,m in report['models'].items()},indent=2))
if __name__=='__main__':main()
