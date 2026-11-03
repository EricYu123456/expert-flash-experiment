#!/usr/bin/env python3
"""Bounded, frozen DeepSeekV2Lite full-FFN same-format placement smoke."""
import argparse,collections,datetime,hashlib,json,os,pathlib,subprocess,time
import numpy as np
ROOT=pathlib.Path('/home/ericyu/expert_flash_experiment')
NFS=pathlib.Path('/home/nfs_client/ericyu/expert_flash_experiment')
BASE=ROOT/'results/research-2026-11-02/round2'
OUT=BASE/'deepseek-full-ffn-integer-reference'
MODEL=NFS/'models/deepseek-v2-lite-chat-q8_0.gguf'
META=NFS/'models/deepseek-metadata.json'
CPP=ROOT/'scripts/research_20261103_deepseek_full_ffn.cpp'
SELF=ROOT/'scripts/research_20261103_deepseek_full_ffn.py'
BIN=OUT/'build/deepseek-full-ffn'
MODEL_SHA='09d9abe9170d14386b0a54fe3969035f6a770956cfefa7d2fbaa12e0900a2ba4'
PROJECTIONS=['gate_exps','up_exps','down_exps','gate_shexp','up_shexp','down_shexp']
IDS=['train_programming_short_015','train_summarization_short_027']
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def lines(p):return [json.loads(l) for l in p.read_text().splitlines()]
def prepare():
 if (OUT/'protocol.frozen.json').exists():raise RuntimeError('Never overwrite frozen protocol')
 OUT.mkdir(exist_ok=True);(OUT/'build').mkdir(exist_ok=True)
 metadata=json.loads(META.read_text());m=metadata['metadata'];assert metadata['sha256']==MODEL_SHA
 assert metadata['routed_layer_ids']==metadata['shared_ffn_layer_ids']==list(range(1,27)) and metadata['dense_ffn_layer_ids']==[0]
 assert m['deepseek2.block_count']==27 and m['deepseek2.leading_dense_block_count']==1 and m['deepseek2.expert_feed_forward_length']==1408 and m['deepseek2.expert_shared_count']==2 and m['deepseek2.embedding_length']==2048
 expected={f'blk.{layer}.ffn_{p}.weight' for layer in range(1,27) for p in PROJECTIONS}|{f'blk.0.ffn_{p}.weight' for p in ['gate','up','down']}
 tensors={t['name']:t for t in metadata['tensors'] if t['name'] in expected};assert len(tensors)==159
 inventory=[]
 for name,t in sorted(tensors.items()):
  C,R=t['shape_ggml'][:2];assert R%16==0
  inventory.append({'weight':name,'columns':C,'rows':R,'experts':t['shape_ggml'][2] if len(t['shape_ggml'])==3 else 1,'source_type':t['ggml_type'],'equal_rows_per_chip':[R//16]*16})
 write(OUT/'frozen_equal_projection_rows.json',inventory)
 old=ROOT/'scripts/research_20261102_full_ffn_integer_reference.cpp';oldtext=old.read_text();newtext=CPP.read_text();contract=[]
 for label,start,end in [('quantize','static float quantize(','static int64_t dot('),('dot','static int64_t dot(','static float scale_output('),('scale_output','static float scale_output(','static bool equal_bits('),('equal_bits','static bool equal_bits(','static bool target(')]:
  a=oldtext[oldtext.index(start):oldtext.index(end,oldtext.index(start))].strip();b=newtext[newtext.index(start):newtext.index(end,newtext.index(start))].strip();assert a==b,label;contract.append({'function':label,'old_new_byte_exact':True,'body_sha256':hashlib.sha256(a.encode()).hexdigest()})
 write(OUT/'unchanged_numeric_primitives.json',contract)
 oldbuild=json.loads((BASE/'full-ffn-integer-reference/build_command.json').read_text())['argv'];argv=list(oldbuild);argv[argv.index(str(old))]=str(CPP);argv[argv.index('-o')+1]=str(BIN);subprocess.run(argv,check=True)
 write(OUT/'build_command.json',{'argv':argv,'compiler':subprocess.check_output(['c++','--version'],text=True).splitlines()[0],'source_sha256':sha(CPP),'binary_sha256':sha(BIN)})
 jobs=[]
 for id in IDS:
  p=NFS/'runs/corpus-300-v1/corpus/prompts'/f'{id}.txt';jobs.append({'id':id,'path':str(p),'sha256':sha(p)})
 (OUT/'jobs.tsv').write_text(''.join(f"{j['path']}\t{j['id']}\n" for j in jobs))
 source=NFS/'sources/llama.cpp-927dada6c'
 inputs=[MODEL,META,CPP,SELF,BIN,old,OUT/'build_command.json',OUT/'jobs.tsv',OUT/'frozen_equal_projection_rows.json',OUT/'unchanged_numeric_primitives.json',source/'src/models/deepseek2.cpp',source/'src/llama-graph.cpp',source/'src/llama-model.cpp']+[pathlib.Path(p) for p in argv if p.endswith('.a')]
 hashes={str(p):sha(p) for p in inputs};assert hashes[str(MODEL)]==MODEL_SHA
 protocol={'frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'input_sha256':hashes,'jobs':jobs,'steps':4,'threads':8,'wall_budget_seconds':360,'model_alias':'deepseek-v2-lite-chat-q8_0','source_commit':'927dada6c9143ba4c940b72004d2698fa5e4e930','scope':{'raw_layers':27,'MoE_raw_layers':list(range(1,27)),'dense_leading_raw_layer':0,'MoE_target_projections':PROJECTIONS,'dense_target_projections':['gate','up','down'],'target_weight_names':159,'all_prefill_tokens_replaced':True,'all_decode_tokens_replaced':True,'attention_router':'Original GGUF host operators','shared':'One concatenated hidden2816=2*1408 branch per MoE layer, with exactly one shared down [2816,2048]; not two downs.'},'numeric_contract':'Original read-only GGUF per-row to_float; fixed row-W8[-127,127]; whole original input-vector dynamicA8[-127,127] once per operator input vector; F32 maxabs/127 (zero1), round ties-away then clip; exact full-C I64 dot, convert once toF32 and multiply once byF32(rowScale*inputScale). No column partials, no fast-math/FMA contraction.','physical_contract':'Fixed equal16 whole-row partition per logical projection, reverse within-chip physical row order, actual chip packed bytes + logical scatter. Dense/shared packed layout identical in both modes; routed canonical vs packed. No fitting.','modes':['unsharded','placed'],'verification':'Every placed I64/fullrow-rescale checked against canonical; every FFN intermediate F32 and final-prefill/decode vocabulary logits checked bitwise across fresh contexts; all routes and generated tokens equal.','no_new_model_download':True,'no_new_quantization_method_claim':True,'no_native_AiF_ABI_or_hardware_performance_claim':True,'no_original_checkpoint_quality_equivalence_claim':True,'prompt_scope':'Existing TRAIN corpus only; no new heldout payloads opened.'}
 write(OUT/'protocol.frozen.json',protocol);print('DeepSeek protocol frozen before execution',flush=True)
def verify():
 x=json.loads((OUT/'protocol.frozen.json').read_text())
 for p,h in x['input_sha256'].items():assert sha(p)==h,p
 for j in x['jobs']:assert sha(j['path'])==j['sha256']
 return x

def audit():
 spec=verify();dest=OUT/'smoke';report=json.loads((dest/'numeric_report.json').read_text());records=lines(dest/'coverage_ledger.jsonl');inventory={r['weight']:r for r in json.loads((OUT/'frozen_equal_projection_rows.json').read_text())};assert len(inventory)==159
 summaries={}
 for j in spec['jobs']:
  id=j['id'];summaries[id]=json.loads((dest/'traces'/f'{id}.unsharded.summary.json').read_text())
 grouped={m:[r for r in records if r['mode']==m] for m in ['unsharded','placed']};assert len(grouped['unsharded'])==len(grouped['placed'])
 fields=['job','phase','step','weight','layer','projection','op','weight_shape','input_shape','output_shape','tokens','selected_experts','output_floats','input_fnv1a64','expert_ids_fnv1a64','output_fnv1a64']
 for a,b in zip(grouped['unsharded'],grouped['placed']):assert all(a[k]==b[k] for k in fields)
 coverage=collections.Counter();tokencount=collections.Counter();kindout=collections.Counter()
 for r in records:
  w=inventory[r['weight']];C,R,E=w['columns'],w['rows'],w['experts'];K=6 if E==64 else 1;T=r['tokens'];p=r['projection'];routed=E==64
  assert r['selected_experts']==K and r['weight_shape']==[C,R,E,1] and r['output_floats']==T*K*R
  assert r['input_shape']==([C,6 if p=='down_exps' else 1,T,1] if routed else [C,T,1,1]) and r['output_shape']==([R,K,T,1] if routed else [R,T,1,1])
  assert (r['layer']==0)==(p in ['gate','up','down'])
  if r['phase']=='decode':assert T==1
  key=(r['mode'],r['job'],r['phase'],r['weight']);coverage[key]+=1;tokencount[key]+=T
  if r['mode']=='placed':kindout['dense' if r['layer']==0 else 'shared' if 'shexp' in p else 'routed']+=r['output_floats']
 details=[]
 for j in spec['jobs']:
  id=j['id'];a=summaries[id];b=json.loads((dest/'traces'/f'{id}.placed.summary.json').read_text());n=a['prompt_tokens'];assert a['generated_token_ids']==b['generated_token_ids'] and a['prompt_token_ids']==b['prompt_token_ids']
  for mode in grouped:
   for phase,total,callbacks in [('prefill',n,(n+511)//512),('decode',4,4)]:
    for w in inventory:assert tokencount[mode,id,phase,w]==total and coverage[mode,id,phase,w]==callbacks
  x=dest/'traces'/f'{id}.unsharded.routing.jsonl';y=dest/'traces'/f'{id}.placed.routing.jsonl';assert x.read_bytes()==y.read_bytes();routes=lines(x);assert len(routes)==26*(n+4) and {r['layer'] for r in routes}==set(range(1,27))
  for phase in ['prefill','decode']:
   per=collections.Counter(r['layer'] for r in routes if r['phase']==phase);assert all(per[l]==(n if phase=='prefill' else 4) for l in range(1,27))
  snapshots=[]
  for suffix in ['ffn_outputs','logits']:
   path=dest/'traces'/f'{id}.unsharded.{suffix}.f32';v=np.memmap(path,dtype='<f4',mode='r');assert np.isfinite(v).all();snapshots.append({'kind':suffix,'floats':v.size,'sha256':sha(path)})
  details.append({'id':id,'prefill_tokens':n,'decode_tokens':4,'MoE_route_records_per_mode':len(routes),'routes_byte_exact':True,'generated_tokens_identical':True,'snapshots':snapshots})
 assert len(coverage)==2*2*2*159
 for key in ['integer_mismatches','same_input_scaled_bit_mismatches','cross_run_output_bit_mismatches','cross_run_logits_bit_mismatches']:assert report[key]==0
 caches=lines(dest/'weight_cache_ledger.jsonl');assert len({(r['weight'],r['expert']) for r in caches})==len(caches)
 for r in caches:assert r['chip_rows']==[r['rows']//16]*16 and r['columns']==inventory[r['weight']]['columns']
 shared=[r for r in caches if 'shexp' in r['weight']];dense=[r for r in caches if r['weight'].startswith('blk.0.')];assert len(shared)==26*3 and len(dense)==3
 for r in shared:
  if 'down' in r['weight']:assert r['columns']==2816 and r['rows']==2048
  else:assert r['columns']==2048 and r['rows']==2816
 result={'status':'PASS','protocol_sha256':sha(OUT/'protocol.frozen.json'),'raw_layers':27,'MoE_raw_layer_ids':list(range(1,27)),'dense_leading_raw_layer':0,'target_FFN_weight_names':159,'MoE_weight_names':156,'dense_weight_names':3,'all_layer_token_shape_coverage_exact':True,'coverage_callback_records':len(records),'coverage_keys':len(coverage),'unique_placed_output_rows':report['integer_checks'],'cross_run_F32_scalars_compared':report['cross_run_output_floats'],'cross_run_vocabulary_logits_compared':report['cross_run_logits_floats'],'same_input_I64_mismatches':0,'cross_run_F32_bit_mismatches':0,'cross_run_logit_bit_mismatches':0,'placed_rows_by_kind':dict(kindout),'shared_cache_entries':len(shared),'shared_hidden2816_single_down_verified':True,'dense_cache_entries':len(dense),'cache_entries':len(caches),'details':details,'limits':['Same fixed-format CPU functional reference, not native AiF arithmetic/ABI or hardware performance.','Two existing TRAIN prompts and4decode steps; no quality or heldout generalization claim.','Dense raw0 included separately; it is not a27th MoE layer.']}
 write(OUT/'summary.json',result);print(json.dumps(result,indent=2),flush=True)

def run():
 spec=verify();dest=OUT/'smoke'
 if dest.exists():raise RuntimeError('Never overwrite smoke run')
 dest.mkdir();argv=[str(BIN),str(MODEL),str(OUT/'jobs.tsv'),'4','8',str(dest)];write(dest/'run_command.json',{'argv':argv,'protocol_sha256':sha(OUT/'protocol.frozen.json'),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()});t=time.monotonic()
 try:
  with open(dest/'stdout.log','w') as out,open(dest/'stderr.log','w') as err:r=subprocess.run(argv,stdout=out,stderr=err,env=dict(os.environ,OMP_NUM_THREADS='8',OMP_DYNAMIC='FALSE',OPENBLAS_NUM_THREADS='1'),timeout=spec['wall_budget_seconds'])
  status=r.returncode
 except subprocess.TimeoutExpired:status='TIMEOUT'
 write(dest/'execution.json',{'exit_code':status,'wall_seconds':time.monotonic()-t,'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
 if status!=0:raise RuntimeError('DeepSeek bounded smoke failed or timed out; artifacts retained')
 audit()
def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','audit']);a=p.parse_args()
 if a.action=='prepare':prepare()
 elif a.action=='run':run()
 else:audit()
if __name__=='__main__':main()
