#!/usr/bin/env python3
"""Independent row ownership and exact paired reference audit; no model execution."""
import collections,hashlib,json,pathlib,re,struct
import numpy as np
ROOT=pathlib.Path('/home/ericyu/expert_flash_experiment');BASE=ROOT/'results/research-2026-11-02/round2';DEST=BASE/'selected-plan-numeric/smoke';TRAIN=BASE/'w8-format-training'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def jsonl(p):return [json.loads(x) for x in p.read_text().splitlines()]
def fnv(bs):
 h=14695981039346656037
 for b in bs:h=((h^b)*1099511628211)&((1<<64)-1)
 return str(h)
def main():
 spec=json.loads((DEST/'protocol.frozen.json').read_text());report={'status':'pass','source_sha256':sha(__file__),'protocol_sha256':sha(DEST/'protocol.frozen.json'),'models':{},'new_quantization_method':False,'native_flash_programming_claim':False}
 for name,cfg in spec['models'].items():
  dest=DEST/name;run=dest/'run';numeric=json.loads((run/'numeric_report.json').read_text());assert numeric['status']=='complete' and numeric['all_FFN_canonical_baseline'] and numeric['shared_uses_selected_rows']
  for k in ['integer_mismatches','same_input_scaled_bit_mismatches','cross_run_output_bit_mismatches','cross_run_logits_bit_mismatches']:assert numeric[k]==0
  # Reconstruct native complete-row ownership directly in concatenated row space.
  native={};total_native_jobs=0
  for path,old_sha in cfg['plans_sha256'].items():
   assert sha(path)==old_sha;plan=json.loads(pathlib.Path(path).read_text());H,I,E,K,L,SI=plan['shape'];layer=cfg['raw_layers'][plan['layer']];owners={}
   for e in range(E+1):
    for phase in (0,1):owners[e,phase]=np.full(2*(SI if e==E else I) if phase==0 else H,-1,dtype=np.int16)
   for b in plan['logical']:
    owner=owners[b['expert'],b['phase']];a=b['row_start'];z=a+b['rows'];assert np.all(owner[a:z]==-1);owner[a:z]=b['chip'];total_native_jobs+=1
   assert all(np.all(a>=0) for a in owners.values())
   for e in range(E+1):
    mid=SI if e==E else I;suffix='_shexp' if e==E else '_exps';se=0 if e==E else e
    native[layer,'gate'+suffix,se]=(H,owners[e,0][:mid].copy());native[layer,'up'+suffix,se]=(H,owners[e,0][mid:].copy());native[layer,'down'+suffix,se]=(mid,owners[e,1].copy())
  if name=='deepseek':
   for p,C,R in [('gate',2048,10944),('up',2048,10944),('down',10944,2048)]:native[0,p,0]=(C,np.arange(R,dtype=np.int16)//(R//16))
  # TSV conversion is checked independently, without its GU intersection helper.
  tsv={};ordered=collections.defaultdict(lambda:[[] for _ in range(16)])
  with pathlib.Path(cfg['rows_tsv']).open() as f:
   for line in f:
    l,p,e,C,R,chip,start,n=line.split();l,e,C,R,chip,start,n=map(int,[l,e,C,R,chip,start,n]);key=l,p,e
    if key not in tsv:tsv[key]=(C,np.full(R,-1,dtype=np.int16))
    assert tsv[key][0]==C and len(tsv[key][1])==R and np.all(tsv[key][1][start:start+n]==-1);tsv[key][1][start:start+n]=chip;ordered[key][chip].extend(range(start,start+n))
  assert tsv.keys()==native.keys()
  logical_rows=0;zero_chips=0
  for key,(C,arr) in native.items():assert C==tsv[key][0] and np.array_equal(arr,tsv[key][1]);logical_rows+=len(arr);zero_chips+=int((np.bincount(arr,minlength=16)==0).sum())
  # Compare identical W8 bytes/scales to prior fixed-format TRAIN cache records.
  prior={};prior_inputs={}
  for path in sorted((TRAIN/name).glob('batch*/weight_cache_ledger.jsonl')):
   prior_inputs[str(path)]=sha(path)
   for c in jsonl(path):
    key=c['weight'],c['expert'];fp=c['canonical_bytes_fnv1a64'],c['row_scales_fnv1a64'];assert key not in prior or prior[key]==fp;prior[key]=fp
  physical_hashes={};used_rows=0;cache=jsonl(run/'weight_cache_ledger.jsonl')
  shared={}
  for c in cache:
   m=re.fullmatch(r'blk\.(\d+)\.ffn_(.+)\.weight',c['weight']);assert m;key=int(m[1]),m[2],c['expert'];C,arr=native[key]
   assert c['columns']==C and c['rows']==len(arr) and c['chip_rows']==np.bincount(arr,minlength=16).tolist()
   if key not in physical_hashes:
    pairs=[(chip,r) for chip in range(16) for r in reversed(ordered[key][chip])];physical_hashes[key]=fnv(np.asarray(pairs,dtype='<i4').tobytes())
   assert c['physical_logical_map_fnv1a64']==physical_hashes[key]
   assert (c['canonical_bytes_fnv1a64'],c['row_scales_fnv1a64'])==prior[c['weight'],c['expert']];used_rows+=len(arr)
   if 'shexp' in key[1]:shared[c['weight']]=c['chip_rows']
  # Match every original ledger operator across both modes, including all shared/dense.
  ledger=jsonl(run/'coverage_ledger.jsonl');group=collections.defaultdict(list)
  for x in ledger:group[x['job'],x['mode']].append(x)
  total_outputs=0;total_logits=0;route_records=0;vocab=102400 if name=='deepseek' else 100352;per_token=982400 if name=='deepseek' else 757760;expected_names={f'blk.{l}.ffn_{p}.weight' for l in cfg['raw_layers'] for p in ['gate_exps','up_exps','down_exps','gate_shexp','up_shexp','down_shexp']}
  if name=='deepseek':expected_names|={f'blk.0.ffn_{p}.weight' for p in ['gate','up','down']}
  for prompt in spec['prompts']:
   pid=prompt['id'];summaries=[];routefiles=[]
   for mode in ['unsharded','placed']:
    stem=run/'traces'/f'{pid}.{mode}';s=json.loads(pathlib.Path(str(stem)+'.summary.json').read_text());summaries.append(s);routefiles.append(pathlib.Path(str(stem)+'.routing.jsonl'));assert s['decoded_tokens']==spec['steps'] and s['prefill_replaced'] and not s['prefill_state_reused']
   a,b=summaries;assert a['prompt_token_ids']==b['prompt_token_ids'] and a['generated_token_ids']==b['generated_token_ids'];n=a['prompt_tokens'];assert n==b['prompt_tokens']
   assert routefiles[0].read_bytes()==routefiles[1].read_bytes();routes=jsonl(routefiles[0]);assert len(routes)==(n+spec['steps'])*len(cfg['raw_layers']);seen=set()
   for x in routes:
    l=x['layer'];assert l in cfg['raw_layers'] and len(set(x['expert_ids']))==6 and all(0<=e<64 for e in x['expert_ids']);position=x['position'];key=x['phase'],position,l;assert key not in seen;seen.add(key)
    if x['phase']=='prefill':assert 0<=position<n and x['token_id']==a['prompt_token_ids'][position]
    else:assert x['phase']=='decode' and position==n+x['step'] and 0<=x['step']<spec['steps'] and x['token_id']==a['generated_token_ids'][x['step']]
   route_records+=len(routes);left=group[pid,'unsharded'];right=group[pid,'placed'];assert len(left)==len(right);coverage=collections.Counter()
   for u,p in zip(left,right):
    for k in ['job','phase','step','weight','layer','projection','op','weight_shape','input_shape','output_shape','tokens','selected_experts','output_floats','input_fnv1a64','expert_ids_fnv1a64','output_fnv1a64']:assert u[k]==p[k],(pid,k)
    assert u['weight'] in expected_names
    for x in (u,p):assert x['integer_mismatches']==x['scaled_bit_mismatches']==x['cross_run_output_bit_mismatches']==0
    assert p['integer_checks']==p['output_floats'];total_outputs+=u['output_floats'];coverage[u['phase'],u['weight']]+=u['tokens']
   for w in expected_names:assert coverage['prefill',w]==n and coverage['decode',w]==spec['steps']
   assert sum(x['output_floats'] for x in left)==(n+spec['steps'])*per_token
   prefix=run/'traces'/f'{pid}.unsharded';ffn=pathlib.Path(str(prefix)+'.ffn_outputs.f32');logs=pathlib.Path(str(prefix)+'.logits.f32');assert ffn.stat().st_size==(n+spec['steps'])*per_token*4 and logs.stat().st_size==(spec['steps']+1)*vocab*4
   values=np.memmap(logs,dtype=np.float32,mode='r');assert np.all(np.isfinite(values));total_logits+=len(values)
   assert pathlib.Path(str(run/'traces'/f'{pid}.unsharded')+'.output.txt').read_bytes()==pathlib.Path(str(run/'traces'/f'{pid}.placed')+'.output.txt').read_bytes()
  assert numeric['cross_run_output_floats']==numeric['integer_checks']==total_outputs and numeric['finite_FFN_values_both_modes']==2*total_outputs
  assert numeric['cross_run_logits_floats']==total_logits and numeric['finite_continuation_logit_values_both_modes']==2*total_logits
  report['models'][name]=dict(status='pass',complete_prompts=len(spec['prompts']),steps=spec['steps'],routed_raw_layers=cfg['raw_layers'],native_logical_jobs=total_native_jobs,all_plan_rowmaps_independently_reconstructed=len(native),logical_weight_rows_all_experts=logical_rows,zero_row_projection_chip_slots=zero_chips,used_cache_entries=len(cache),used_cache_rows_with_verified_physical_map=used_rows,used_cache_W8_bytes_and_scales_identical_to_priorTRAIN=True,unique_paired_FFN_output_floats=total_outputs,placed_I64_checks=total_outputs,cross_run_F32_checks=total_outputs,unique_paired_logit_floats=total_logits,ordered_route_records_compared=route_records,all_FFN_weight_names=len(expected_names),selected_shared_chip_rows=shared,prior_TRAIN_cache_inputs_sha256=prior_inputs,numeric_report_sha256=sha(run/'numeric_report.json'),coverage_ledger_sha256=sha(run/'coverage_ledger.jsonl'),cache_ledger_sha256=sha(run/'weight_cache_ledger.jsonl'))
 out=DEST/'independent';out.mkdir(exist_ok=True);p=out/'audit.json';assert not p.exists(),'Do not overwrite sealed selected-plan audit';p.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({name:{k:v for k,v in x.items() if k not in ['selected_shared_chip_rows','prior_TRAIN_cache_inputs_sha256']} for name,x in report['models'].items()},indent=2))
if __name__=='__main__':main()
