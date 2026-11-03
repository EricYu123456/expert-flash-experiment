#!/usr/bin/env python3
"""Independent archive/raw-route/metadata checks after explicit root release."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import hashlib,json,sys
import numpy as np
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;CAP=BASE/'data-capture';OUT=HERE/'capture-integrity-audit'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def freeze():
 OUT.mkdir(exist_ok=False)
 paths=[BASE/'root_release.json',CAP/'acquisition_capture_protocol.frozen.json',CAP/'completion.frozen.json',CAP/'manifest.json',CAP/'capture_summary.json',CAP/'granite4tiny/decode.routes.npz',BASE/'protocol.frozen.json',CAP/'exclusion_catalog.json',CAP/'capture_execution.json']
 dump(OUT/'protocol.frozen.json',{'utc':datetime.now(timezone.utc).isoformat(),'source_sha256':sha(__file__),'inputs':{str(p):sha(p) for p in paths},'rules':['Verify explicit root release and pinned completion/protocol/manifest/archive hashes.','Full32x16x40x6 ordered archive: integer, range, unique top6, allcomplete masks, exact prompt/sourceIDs/layers/position/token identities.','Independently rebuild all decode and prefill route keys from32 raw JSONL files; reject duplicates or missing coordinates. Compare decode expert order exactly with archive.','Verify32 prompt file hashes, exclusions, source payload byte hashes, deterministic source-ID ranking, fixed batch assignment and no truncation/replacement. No old prompt text/test payload read.','Inspect all per-operator coverage metadata for complete40layers and all6projection types; this checks ledger provenance, not recomputation of1.4B numerical values.','Generate the binding for the separately pre-frozen descriptive route-shift rule, with no costs/selection/refit.']})
 print('frozen',sha(OUT/'protocol.frozen.json'))
def run():
 p=json.loads((OUT/'protocol.frozen.json').read_text());assert sha(__file__)==p['source_sha256']
 for f,h in p['inputs'].items():assert sha(f)==h
 rel=json.loads((BASE/'root_release.json').read_text());seal=json.loads((CAP/'completion.frozen.json').read_text());m=json.loads((CAP/'manifest.json').read_text());cp=json.loads((CAP/'acquisition_capture_protocol.frozen.json').read_text());summary=json.loads((CAP/'capture_summary.json').read_text());anchor=json.loads((BASE/'protocol.frozen.json').read_text());ex=json.loads((CAP/'exclusion_catalog.json').read_text());execution=json.loads((CAP/'capture_execution.json').read_text())
 assert sha(BASE/'root_release.json')=='07ea0440e55edaf3d48abd30094df7c0f7b9bae52577b6b91e8dbecf876e3884'
 assert sha(CAP/'completion.frozen.json')==rel['capture_completion_sha256']=='8f7b2389fb0f8e42f024dd73b7ad271969ca8749f77f8a759811981e1b407909'
 assert sha(CAP/'acquisition_capture_protocol.frozen.json')==rel['capture_protocol_sha256']==m['protocol_sha256']==seal['acquisition_capture_protocol_sha256']
 assert sha(CAP/'manifest.json')==rel['capture_manifest_sha256']==seal['manifest_sha256']==summary['manifest_sha256']
 assert sha(CAP/'granite4tiny/decode.routes.npz')==seal['decode_archive_sha256']==summary['decode_archive_sha256']=='d4f7cc54970ef1223668c92b68eea22f22c775be9f3293fa61203601df8535ca'
 assert sha(BASE/'protocol.frozen.json')==m['anchor_protocol_sha256']==seal['anchor_protocol_sha256']
 assert not cp['payloads_opened_before_freeze']
 assert datetime.fromisoformat(cp['frozen_utc'])<datetime.fromisoformat(m['acquisition_started_utc'])<datetime.fromisoformat(m['created_utc'])<datetime.fromisoformat(seal['sealed_utc'])<datetime.fromisoformat(rel['utc'])
 assert execution['status']=='complete' and all(x['exit_code']==0 for x in execution['batches']) and execution['all_original_selections_retained']
 with np.load(CAP/'granite4tiny/decode.routes.npz') as z:a={k:z[k] for k in z.files}
 r=a['routes'];assert r.shape==(32,16,40,6) and r.dtype==np.int32 and np.all((r>=0)&(r<64)) and np.all(np.diff(np.sort(r,axis=-1),axis=-1)>0)
 assert a['completed'].shape==(32,) and a['completed'].dtype==bool and a['completed'].all()
 assert np.array_equal(a['layers'],np.arange(40)) and a['representation'].item()=='fixed_full_FFN_W8A8_canonical_only' and a['split'].item()=='prospective_blind'
 assert a['token_ids'].shape==a['token_positions'].shape==(32,16) and np.all(a['token_ids']>=0)
 items=m['items'];assert len(items)==32 and len(set(a['prompt_ids'].tolist()))==32
 assert np.array_equal(a['prompt_ids'],[i['id'] for i in items]) and np.array_equal(a['source_ids'],[i['source_id'] for i in items]) and np.array_equal(a['source_keys'],[i['source_key'] for i in items])
 assert Counter(a['source_keys'].tolist())=={'piqa_validation':16,'winogrande_validation':16}
 oldhash={x['prompt_sha256'] for x in ex['records']};oldids={x['id'] for x in ex['records']};seenhash=set();raw_hashes={}
 for receipt in m['source_receipts']:
  assert sha(receipt['path'])==receipt['sha256'];raw_hashes[receipt['path']]=receipt['sha256']
 if cp['sources'][0].get('expected_git_blob_sha1'):
  b=Path(m['source_receipts'][0]['path']).read_bytes();assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==cp['sources'][0]['expected_git_blob_sha1']
 ranks={}
 for spec in cp['sources']:
  ranked=sorted(range(m['source_rows_counts'][spec['key']]),key=lambda i:(hashlib.sha256('\0'.join([cp['seed'],spec['key'],spec['revision'],spec['split'],str(i)]).encode()).hexdigest(),str(i)))
  selected=[i for i in items if i['source_key']==spec['key']];assert [int(i['source_id']) for i in selected]==ranked[:16]
  ranks[spec['key']]=[int(i['source_id']) for i in selected]
 expected={f'blk.{L}.ffn_{pr}.weight' for L in range(40) for pr in ['gate_exps','up_exps','down_exps','gate_shexp','up_shexp','down_shexp']}
 coverage={};coverage_records=0;coverage_numerical_claims=0
 for batch in range(4):
  pp=CAP/f'granite4tiny/batch{batch}/coverage_ledger.jsonl'
  for line in pp.read_text().splitlines():
   x=json.loads(line);assert x['weight'] in expected and x['mode']=='unsharded' and x['phase'] in ('decode','prefill')
   assert x['integer_mismatches']==x['scaled_bit_mismatches']==x['cross_run_output_bit_mismatches']==0
   C,R,E=x['weight_shape'][:3];K=6 if E==64 else 1;assert x['selected_experts']==K and x['input_shape'][0]==C and x['output_floats']==x['tokens']*K*R
   coverage.setdefault(x['job'],Counter())[x['phase'],x['weight']]+=x['tokens'];coverage_records+=1;coverage_numerical_claims+=x['output_floats']
 checks=[];raw_records=0;decode_records=0;prefill_records=0
 for idx,item in enumerate(items):
  ident=item['id'];assert ident not in oldids;h=sha(item['prompt_path']);assert h==item['prompt_sha256'] and h not in oldhash|seenhash;seenhash.add(h)
  assert item['removed_codepoints']==0 and item['prompt_sha256']==item['full_rendered_prompt_sha256'] and item['split']=='prospective_blind'
  spec=next(x for x in cp['sources'] if x['key']==item['source_key']);assert item['dataset_revision']==spec['revision'] and item['dataset']==spec['dataset'] and item['dataset_split']==spec['split']
  assert item['source_payload_sha256']==raw_hashes[item['source_payload_path']]
  batch=idx%4;stem=CAP/f'granite4tiny/batch{batch}/traces/{ident}.unsharded';ss=json.loads(Path(str(stem)+'.summary.json').read_text());n=ss['prompt_tokens']
  assert n==item['prompt_tokens_including_BOS']<=256 and ss['decoded_tokens']==16 and len(ss['generated_token_ids'])==16
  assert np.array_equal(a['token_positions'][idx],np.arange(n,n+16)) and np.array_equal(a['token_ids'][idx],ss['generated_token_ids'])
  keys=set();local_decode=0;local_prefill=0;routepath=Path(str(stem)+'.routing.jsonl')
  for line in routepath.read_text().splitlines():
   x=json.loads(line);L=x['layer'];ids=x['expert_ids'];assert 0<=L<40 and len(ids)==len(set(ids))==6 and all(isinstance(e,int) and 0<=e<64 for e in ids)
   if x['phase']=='decode':
    t=x['step'];assert 0<=t<16 and x['position']==n+t and x['token_id']==ss['generated_token_ids'][t] and np.array_equal(r[idx,t,L],ids);key=('decode',t,L);local_decode+=1
   else:
    assert x['phase']=='prefill';t=x['position'];assert 0<=t<n and x['token_id']==ss['prompt_token_ids'][t];key=('prefill',t,L);local_prefill+=1
   assert key not in keys;keys.add(key)
  assert local_decode==640 and local_prefill==n*40
  for w in expected:assert coverage[ident]['decode',w]==16 and coverage[ident]['prefill',w]==n
  rec=next(x for x in summary['prompt_records'] if x['id']==ident);assert sha(routepath)==rec['routing_sha256'] and rec['routing_records']==len(keys)
  raw_records+=len(keys);decode_records+=local_decode;prefill_records+=local_prefill
  checks.append({'id':ident,'source_key':item['source_key'],'raw_route_records':len(keys),'decode_records':local_decode,'prefill_records':local_prefill,'archive_order_exact':True,'complete_all240_projection_weights':True})
 assert decode_records==20480 and len(coverage)==32
 finitecount=0;logitcount=0
 for rec in summary['batch_reports']:
  assert sha(rec['path'])==rec['sha256'];br=json.loads(Path(rec['path']).read_text());assert br['status']=='complete' and br['mode']=='canonical_only' and br['jobs']==8 and br['decode_steps']==16 and br['full_FFN_outputs_all_tokens_finite'] and not br['new_capture_routed_rows_paired_with_placed']
  subset=items[rec['batch']::4];expected_count=sum(x['prompt_tokens_including_BOS']+16 for x in subset)*40*(6*(512+512+1536)+1024+1024+1536)
  assert expected_count==br['finite_FFN_values_checked']==rec['finite_FFN_values'];finitecount+=expected_count;logitcount+=br['continuation_logit_values_checked_finite']
 assert coverage_numerical_claims==finitecount==1438986240
 binding={'capture_complete':True,'root_confirmed_sealed_capture':True,'routes':{'path':str(CAP/'granite4tiny/decode.routes.npz'),'sha256':sha(CAP/'granite4tiny/decode.routes.npz'),'key':'routes'},'acquisition_protocol':{'path':str(CAP/'acquisition_capture_protocol.frozen.json'),'sha256':sha(CAP/'acquisition_capture_protocol.frozen.json')},'capture_protocol':{'path':str(CAP/'acquisition_capture_protocol.frozen.json'),'sha256':sha(CAP/'acquisition_capture_protocol.frozen.json')},'capture_manifest':{'path':str(CAP/'manifest.json'),'sha256':sha(CAP/'manifest.json')},'source_keys':['piqa_validation','winogrande_validation'],'prompt_sources':[x['source_key'] for x in items],'root_release':{'path':str(BASE/'root_release.json'),'sha256':sha(BASE/'root_release.json')},'capture_and_acquisition_share_one_frozen_protocol':True,'independent_archive_audit_protocol_sha256':sha(OUT/'protocol.frozen.json')}
 dump(HERE/'route-shift-diagnostic/sealed_capture_binding.json',binding)
 report={'status':'PASS_INDEPENDENT_NEW_ARCHIVE_AND_RAW_ROUTE_PROVENANCE','protocol_sha256':sha(OUT/'protocol.frozen.json'),'utc':datetime.now(timezone.utc).isoformat(),'archive_shape':list(r.shape),'uniqueK_all20480_layer_queries':True,'complete_mask_all32':True,'ordered_decode_route_records_exact':decode_records,'prefill_route_records':prefill_records,'all_raw_records':raw_records,'coverage_records':coverage_records,'declared_finite_FFN_scalar_ledger':finitecount,'declared_finite_continuation_logits':logitcount,'source_selected_row_IDs':ranks,'prompt_checks':checks,'scope':'Independent rawroute/archive identity and complete projection metadata/provenance. Did not recompute all FFN outputs or re-run inference, and does not prove numerical execution of the three new placements/native hardware. No previous-test payload read.'}
 dump(OUT/'audit.json',report)
 for f,h in p['inputs'].items():assert sha(f)==h
 dump(OUT/'completion.json',{'status':'SEALED_PASS','files':{str(f.relative_to(OUT)):sha(f) for f in sorted(OUT.iterdir()) if f.is_file() and f.name!='completion.json'},'source_sha256':sha(__file__),'route_shift_binding_sha256':sha(HERE/'route-shift-diagnostic/sealed_capture_binding.json')})
 print(json.dumps({'status':report['status'],'decode_records':decode_records,'raw_records':raw_records,'audit_sha256':sha(OUT/'audit.json'),'completion_sha256':sha(OUT/'completion.json')}))
if __name__=='__main__':{'freeze':freeze,'run':run}[sys.argv[1]]()
