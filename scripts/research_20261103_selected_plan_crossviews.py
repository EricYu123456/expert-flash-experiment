#!/usr/bin/env python3
"""Exact cross-profile comparison, linking the two paired validations to four views."""
import argparse,hashlib,json,pathlib
import numpy as np
from research_20261103_selected_plan_numeric import OUT,sha

def compare(a,b):
 assert a.stat().st_size==b.stat().st_size;total=0
 with a.open('rb') as left,b.open('rb') as right:
  while True:
   x=left.read(8<<20);y=right.read(8<<20);assert x==y,(a,b,total)
   if not x:break
   total+=len(x)
 return total

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['smoke','fresh'],required=True);phase=parser.parse_args().phase;dest=OUT/('smoke-primary' if phase=='smoke' else 'second-heldout');summary=json.loads((dest/'summary.json').read_text());assert summary['status']=='complete' and summary['independent_audits_passed'];spec=json.loads((dest/'protocol.frozen.json').read_text());results={}
 for name in ['granite4tiny','deepseek']:
  a=dest/('input1-'+name);b=dest/('input4-'+name);left=dict(np.load(a/'decode.routes.npz',allow_pickle=False));right=dict(np.load(b/'decode.routes.npz',allow_pickle=False));assert left.keys()==right.keys()
  for k in left:assert np.array_equal(left[k],right[k]),(name,k)
  leftp=dict(np.load(a/'prefill.routes.npz',allow_pickle=False));rightp=dict(np.load(b/'prefill.routes.npz',allow_pickle=False));assert leftp.keys()==rightp.keys()
  for k in leftp:assert np.array_equal(leftp[k],rightp[k]),(name,k)
  total_ffn=0;total_logits=0;per_prompt=[]
  for p in spec['prompts']:
   filename=p['id']+'.unsharded';ffn=compare(a/'run/traces'/(filename+'.ffn_outputs.f32'),b/'run/traces'/(filename+'.ffn_outputs.f32'));logits=compare(a/'run/traces'/(filename+'.logits.f32'),b/'run/traces'/(filename+'.logits.f32'));total_ffn+=ffn//4;total_logits+=logits//4;per_prompt.append(dict(id=p['id'],full_FFN_scalar_outputs_per_view=ffn//4,continuation_logit_scalars_per_view=logits//4))
  for d in [a,b]:
   report=json.loads((d/'run/numeric_report.json').read_text());assert report['cross_run_output_floats']==total_ffn and report['cross_run_output_bit_mismatches']==0 and report['cross_run_logits_floats']==total_logits and report['cross_run_logits_bit_mismatches']==0
  results[name]=dict(status='pass',views=['input1.canonical','input1.placed','input4.canonical','input4.placed'],full_FFN_scalar_outputs_per_view=total_ffn,continuation_logit_scalars_per_view=total_logits,exact_pairwise_links=['input1canonical==input1placed by fullF32 streaming comparison','input4canonical==input4placed by fullF32 streaming comparison','input1canonical==input4canonical by full filebyte comparison'],both_prefill_and_decode_routes_exact_across_profiles=True,decode_shape=list(left['routes'].shape),prefill_tokens=len(leftp['token_ids']),per_prompt=per_prompt)
 report=dict(status='pass',source_sha256=sha(__file__),protocol_sha256=sha(dest/'protocol.frozen.json'),models=results,scope='Fixed numeric reference; no sourceGGUF equivalence, quality or native hardware timing claim. Each of3 comparisons covers one fullview scalarcount; do not call it3x distinct outputs.')
 target=dest/'crossviews.exact.json';assert not target.exists(),'Never overwrite cross-view seal';target.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:{key:v for key,v in x.items() if key!='per_prompt'} for k,x in results.items()},indent=2))
if __name__=='__main__':main()
