#!/usr/bin/env python3
"""Read-only audit: exact executed tensor geometry and per-token FFN coverage."""
import collections, hashlib, json, pathlib, sys
import numpy as np
root=pathlib.Path(__file__).resolve().parent
stage=sys.argv[1]
p=root/stage
spec=json.loads((root/'frozen_protocol.json').read_text())['stages'][stage]
rows=[json.loads(line) for line in (p/'coverage_ledger.jsonl').read_text().splitlines()]
shapes={
 'gate_exps':([1536,512,64,1],1536,512,6),
 'up_exps':([1536,512,64,1],1536,512,6),
 'down_exps':([512,1536,64,1],512,1536,6),
 'gate_shexp':([1536,1024,1,1],1536,1024,1),
 'up_shexp':([1536,1024,1,1],1536,1024,1),
 'down_shexp':([1024,1536,1,1],1024,1536,1),
}
prompt_tokens={j['id']:json.loads((p/'traces'/f"{j['id']}.unsharded.summary.json").read_text())['prompt_tokens'] for j in spec['jobs']}
token_totals=collections.Counter()
for r in rows:
 projection=r['projection'];w,C,R,K=shapes[projection];T=prompt_tokens[r['job']] if r['phase']=='prefill' else 1
 assert r['tokens']==T and r['weight_shape']==w and r['selected_experts']==K,r
 assert r['output_floats']==T*K*R,r
 if projection.endswith('_exps'):
  V=6 if projection=='down_exps' else 1
  assert r['input_shape']==[C,V,T,1] and r['output_shape']==[R,K,T,1],r
 else:assert r['input_shape']==[C,T,1,1] and r['output_shape']==[R,T,1,1],r
 token_totals[(r['mode'],r['job'],r['phase'],r['weight'])]+=T
for mode in ['unsharded','placed']:
 for job in spec['jobs']:
  for layer in range(40):
   for projection in shapes:
    for phase,T in [('prefill',prompt_tokens[job['id']]),('decode',spec['decode_steps'])]:
     assert token_totals[(mode,job['id'],phase,f'blk.{layer}.ffn_{projection}.weight')]==T
finite=[]
for job in spec['jobs']:
 for suffix in ['ffn_outputs','logits']:
  file=p/'traces'/f"{job['id']}.unsharded.{suffix}.f32"
  a=np.memmap(file,dtype='<f4',mode='r');assert np.isfinite(a).all(),str(file)
  finite.append({'file':str(file),'float_count':a.size,'all_finite':True})
result={'status':'PASS','all_40_layers_all_six_projections_every_prefill_token_replaced':True,'all_decode_tokens_replaced':True,'prompt_tokens':prompt_tokens,'exact_tensor_geometries_checked':len(rows),'coverage_keys':len(token_totals),'finite_reference_snapshots':finite,'audit_source_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}
(p/'full_token_coverage_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
