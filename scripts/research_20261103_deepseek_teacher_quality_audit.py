#!/usr/bin/env python3
"""Coverage and independent F64 metric reconstruction for frozen Deep teacher windows."""
import collections,csv,json,math,pathlib
import numpy as np
from research_20261103_deepseek_teacher_quality import OUT,BASE,sha,lines,verify

def routecmp(a,b):
 assert len(a)==len(b);ordered=sets=intersection=0;jaccard=0.
 for x,y in zip(a,b):
  assert (x['position'],x['layer'],x['token_id'])==(y['position'],y['layer'],y['token_id']);s,t=set(x['expert_ids']),set(y['expert_ids']);assert len(s)==len(t)==6 and all(0<=e<64 for e in s|t)
  k=len(s&t);ordered+=x['expert_ids']==y['expert_ids'];sets+=s==t;intersection+=k;jaccard+=k/len(s|t)
 n=len(a);return dict(records=n,ordered_exact=ordered,ordered_exact_fraction=ordered/n,set_exact=sets,set_exact_fraction=sets/n,mean_intersection_over6=intersection/(6*n),mean_jaccard=jaccard/n)

def geometry(r):
 p=r['projection'];dense=r['layer']==0;routed=p.endswith('_exps');shared=p.endswith('_shexp');assert dense or routed or shared;down=p.startswith('down');C,R=(10944,2048) if dense and down else (2048,10944) if dense else (2816,2048) if shared and down else (2048,2816) if shared else (1408,2048) if down else (2048,1408);K=6 if routed else 1
 assert r['tokens']==128 and r['selected_experts']==K and r['output_floats']==128*K*R and r['weight_shape']==[C,R,64 if routed else 1,1]
 assert r['input_shape']==([C,6 if down else 1,128,1] if routed else [C,128,1,1]);assert r['output_shape']==([R,6,128,1] if routed else [R,128,1,1]);assert r['op']==('MUL_MAT_ID' if routed else 'MUL_MAT')

def main():
 spec=verify();execution=json.loads((OUT/'execution.json').read_text());assert len(execution['batches'])==2 and all(x['exit_code']==0 for x in execution['batches']);expected={f'blk.{l}.ffn_{p}.weight' for l in range(1,27) for p in ['gate_exps','up_exps','down_exps','gate_shexp','up_shexp','down_shexp']}|{f'blk.0.ffn_{p}.weight' for p in ['gate','up','down']};assert len(expected)==159
 grouped={};cache=[];all_rows=[]
 for batch in range(2):
  d=OUT/f'batch{batch}';rows=lines(d/'coverage_ledger.jsonl');all_rows.extend(rows);cache.extend(lines(d/'weight_cache_ledger.jsonl'))
  for r in rows:geometry(r);key=r['mode'],r['job'],r['weight'];assert key not in grouped;grouped[key]=r
  report=json.loads((d/'completion_numeric.json').read_text());assert report['status']=='complete' and report['FFN_weight_names']==159 and report['dense_raw0_included'] and report['all_logits_finite'];assert report['finite_W8_FFN_values']==(640 if batch==0 else 512)*982400
  for mode in ['source-gguf','unsharded']:
   ids=[j['id'] for j in spec['jobs'][batch::2]]+([spec['jobs'][0]['id']+'.repeat'] if mode=='unsharded' and batch==0 else [])
   for pid in ids:
    for w in expected:
     row=grouped[mode,pid,w]
     if mode=='source-gguf':assert row['all_finite']
     else:assert row['integer_mismatches']==row['scaled_bit_mismatches']==row['cross_run_output_bit_mismatches']==0
 assert len(all_rows)==159*17
 prior={}
 for path in (BASE/'w8-format-training/deepseek').glob('batch*/weight_cache_ledger.jsonl'):
  for c in lines(path):prior[c['weight'],c['expert']]=(c['canonical_bytes_fnv1a64'],c['row_scales_fnv1a64'])
 for c in cache:assert prior[c['weight'],c['expert']]==(c['canonical_bytes_fnv1a64'],c['row_scales_fnv1a64'])
 sums={'source-gguf':0.,'unsharded':0.};allroutes={m:[] for m in sums};chunk=[];snapshot_checks=[];top=0;tokenwindows=[]
 for index,j in enumerate(spec['jobs']):
  pid=j['id'];d=OUT/f'batch{index%2}/traces';summaries={m:json.loads((d/f'{pid}.{m}.summary.json').read_text()) for m in sums};positions={m:lines(d/f'{pid}.{m}.positions.jsonl') for m in sums};assert summaries['source-gguf']['token_ids']==summaries['unsharded']['token_ids'];tokenwindows.append(dict(id=pid,prompt_sha256=j['prompt_sha256'],Deep_GGUF_token_ids=summaries['source-gguf']['token_ids']))
  for mode,s in summaries.items():
   assert s['input_tokens']==128 and s['scored_next_tokens']==127 and len(s['token_ids'])==128 and s['vocabulary']==102400 and s['finite_logits']==128*102400 and len(positions[mode])==127
   summed=math.fsum(p['nll_F64'] for p in positions[mode]);assert abs(summed-s['sum_nll_F64'])<1e-9;sums[mode]+=summed
   for p,r in enumerate(positions[mode]):assert r['position']==p and r['input_token_id']==s['token_ids'][p] and r['target_token_id']==s['token_ids'][p+1] and math.isfinite(r['nll_F64']) and abs(r['nll_F64']-(r['logsumexp_F64']-r['target_logit']))<1e-12
   snap=np.fromfile(d/f'{pid}.{mode}.first2_logits.f32',dtype='<f4').reshape(2,102400);assert np.isfinite(snap).all();errs=[]
   for p,logits in enumerate(snap.astype(np.float64)):
    mx=logits.max();lse=float(mx+np.log(np.exp(logits-mx).sum(dtype=np.float64)));r=positions[mode][p];errs.append(abs(lse-r['logsumexp_F64']));assert errs[-1]<1e-10 and int(np.argmax(logits))==r['top1_token_id'] and float(logits[r['target_token_id']])==r['target_logit']
   snapshot_checks.append(dict(id=pid,mode=mode,first2_full_vocabulary_F64_lse_max_abs_error=max(errs)));routes=lines(d/f'{pid}.{mode}.routing.jsonl');assert len(routes)==128*26 and {(r['position'],r['layer']) for r in routes}=={(p,l) for p in range(128) for l in range(1,27)}
   for r in routes:assert r['phase']=='teacher_prefill' and r['step']==-1 and r['token_id']==s['token_ids'][r['position']]
   allroutes[mode].extend(routes)
  agree=sum(x['top1_token_id']==y['top1_token_id'] for x,y in zip(positions['source-gguf'],positions['unsharded']));top+=agree;a=summaries['source-gguf']['mean_nll_F64'];b=summaries['unsharded']['mean_nll_F64'];delta=b-a;chunk.append(dict(id=pid,scored_tokens=127,source_Q8_mean_NLL=a,W8_mean_NLL=b,delta_NLL_W8_minus_Q8=delta,source_Q8_PPL=math.exp(a),W8_PPL=math.exp(b),delta_PPL_W8_minus_Q8=math.exp(b)-math.exp(a),PPL_ratio_W8_over_Q8=math.exp(delta),top1_agreement_count=agree,top1_agreement_fraction=agree/127,router=routecmp(lines(d/f'{pid}.source-gguf.routing.jsonl'),lines(d/f'{pid}.unsharded.routing.jsonl'))))
 pid=spec['jobs'][0]['id'];d=OUT/'batch0/traces';a=json.loads((d/f'{pid}.unsharded.summary.json').read_text());b=json.loads((d/f'{pid}.repeat.unsharded.summary.json').read_text());assert a['all_logits_fnv1a64']==b['all_logits_fnv1a64'] and a['token_ids']==b['token_ids'] and b['repeat_logits_bit_mismatches']==0 and b['independent_context_repeat']
 for suffix in ['positions.jsonl','routing.jsonl','first2_logits.f32']:assert (d/f'{pid}.unsharded.{suffix}').read_bytes()==(d/f'{pid}.repeat.unsharded.{suffix}').read_bytes()
 for w in expected:
  x,y=grouped['unsharded',pid,w],grouped['unsharded',pid+'.repeat',w];assert x['input_fnv1a64']==y['input_fnv1a64'] and x['output_fnv1a64']==y['output_fnv1a64']
 source_mean=sums['source-gguf']/1016;w8_mean=sums['unsharded']/1016;delta=w8_mean-source_mean;bylayer=[dict(raw_layer=l,**routecmp([r for r in allroutes['source-gguf'] if r['layer']==l],[r for r in allroutes['unsharded'] if r['layer']==l])) for l in range(1,27)]
 report=dict(status='PASS',source_sha256=sha(__file__),protocol_sha256=sha(OUT/'protocol.frozen.json'),purpose='Supplemental quality sensitivity on8existing frozenWikiText texts, not a newblindtest. NativeDeep GGUFtokenization, sourceQ8GGUFbaseline, noBF16/taskaccuracy/hardware inference.',windows=8,input_tokens_per_window=128,scored_next_token_positions=1016,source_Q8_GGUF_mean_NLL_F64=source_mean,full_FFN_W8A8_mean_NLL_F64=w8_mean,delta_NLL_W8_minus_source_Q8=delta,source_Q8_GGUF_PPL=math.exp(source_mean),full_FFN_W8A8_PPL=math.exp(w8_mean),delta_PPL_W8_minus_Q8=math.exp(w8_mean)-math.exp(source_mean),PPL_ratio_W8_over_Q8=math.exp(delta),teacher_top1_agreement_count=top,teacher_top1_agreement_fraction=top/1016,full_fixed_teacher_router_comparison=routecmp(allroutes['source-gguf'],allroutes['unsharded']),router_per_raw_layer=bylayer,per_chunk=chunk,all159_FFN_weights_all128tokens_both_controls_covered=True,Deep_raw0_dense_included=True,sharedI2816_single_down=True,all_logits_and_FFN_outputs_finite=True,source_and_W8_unique_FFN_output_scalars_per_control=1024*982400,firstW8_context_repeat_full_logit_values_bitcompared=128*102400,repeat_bit_mismatches=0,coverage_callback_records=159*17,snapshot_independent_numeric_checks=snapshot_checks,cache_records=len(cache),cache_bytes_and_scales_hashes_match_priorTRAIN=True,inference_orchestration_wall_seconds=execution['wall_seconds'])
 p=OUT/'summary.json';assert not p.exists();dump=lambda p,x:p.write_text(json.dumps(x,indent=2)+'\n');dump(p,report);dump(OUT/'actual_token_windows.json',tokenwindows)
 keys=[k for k in chunk[0] if k!='router']
 with (OUT/'per_chunk_quality.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:r[k] for k in keys} for r in chunk)
 text=['Existing frozen WikiText windows; DeepSeek Q8_GGUF -> fullFFN W8A8 supplemental quality sensitivity.','No newblindtest/taskaccuracy/BF16/nativehardware claim. Positive deltaNLL orPPL is degradation; no quantizer/layout retuned.','','ID | Q8 PPL | W8 PPL | deltaPPL | ratio | deltaNLL']
 for r in chunk:text.append(f"{r['id']} | {r['source_Q8_PPL']:.6f} | {r['W8_PPL']:.6f} | {r['delta_PPL_W8_minus_Q8']:+.6f} | {r['PPL_ratio_W8_over_Q8']:.8f} | {r['delta_NLL_W8_minus_Q8']:+.8f}")
 text += ['',f'Aggregate(1016positions): Q8PPL{report["source_Q8_GGUF_PPL"]:.8f}, W8PPL{report["full_FFN_W8A8_PPL"]:.8f}, ratio{report["PPL_ratio_W8_over_Q8"]:.10f}, deltaNLL{delta:+.10f}.',f'Top1agreement{top}/1016; fullteacher router sites26624(raw1..26 atall128tokens). FirstW8window independentcontext bitrepeat:13,107,200logits.', 'SourceandW8 contexts are independent withidentical128native-token IDs perwindow. Deep tokenization/window endpoints differfromG4, soPPLvalues are not crossmodel comparisons. Allsourceweights unchanged, fullFFN W8 helpers byte-exact andfixed rowweights/scales hashesmatchTRAIN.']
 (OUT/'per_chunk_quality.txt').write_text('\n'.join(text)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ['router_per_raw_layer','per_chunk','snapshot_independent_numeric_checks']},indent=2))
if __name__=='__main__':main()
