#!/usr/bin/env python3
"""Freeze, execute and independently audit full FFN placement equivalence."""
import argparse, collections, datetime, hashlib, json, os, pathlib, subprocess, sys, time
ROOT=pathlib.Path('/home/ericyu/expert_flash_experiment')
OUT=ROOT/'results/research-2026-11-02/round2/full-ffn-integer-reference'
NFS=pathlib.Path('/home/nfs_client/ericyu/expert_flash_experiment')
CPP=ROOT/'scripts/research_20261102_full_ffn_integer_reference.cpp'
SELF=ROOT/'scripts/research_20261102_full_ffn_integer_reference.py'
MODEL=NFS/'models/granite4tiny-q4_k_m.gguf'
PLAN=ROOT/'results/research-2026-11-02/round2/geometry-study/granite4tiny.frozen.json'
OLD_BUILD=ROOT/'results/research-2026-11-02/round2/raw-w8a8-numeric/build_command.json'
BIN=OUT/'build/full-ffn-integer-reference'
IDS=['train_programming_short_015','train_summarization_short_027']
PROJECTIONS=['gate_exps','up_exps','down_exps','gate_shexp','up_shexp','down_shexp']
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
 return h.hexdigest()
def write(p,x): p.write_text(json.dumps(x,indent=2)+'\n')
def prepare():
 if (OUT/'frozen_protocol.json').exists(): raise RuntimeError('Frozen protocol already exists; never overwrite')
 OUT.mkdir(parents=True,exist_ok=True);(OUT/'build').mkdir(exist_ok=True)
 source=json.loads(PLAN.read_text());assert len(source['phases'])==80
 lines=[]
 for p in source['phases']:
  rows=p.get('rows64x16',p.get('rows'));assert len(rows)==64
  for e,rr in enumerate(rows):
   assert len(rr)==16 and sum(rr)==p['total_rows']
   lines.append('\t'.join(map(str,[p['layer'],p['name'],e,p['total_rows'],p['columns'],*rr])))
 (OUT/'placement.tsv').write_text('\n'.join(lines)+'\n')
 old=json.loads(OLD_BUILD.read_text());argv=old['argv'];argv[argv.index(str(ROOT/'scripts/research_20261102_raw_w8a8_numeric.cpp'))]=str(CPP);argv[argv.index('-o')+1]=str(BIN)
 subprocess.run(argv,check=True)
 write(OUT/'build_command.json',{'argv':argv,'compiler_version':subprocess.check_output(['c++','--version'],text=True).splitlines()[0],'source_sha256':sha(CPP),'binary_sha256':sha(BIN),'upstream_static_libraries_reused_unchanged':True})
 prompts=[]
 for id in IDS:
  p=NFS/'runs/corpus-300-v1/corpus/prompts'/f'{id}.txt';prompts.append({'id':id,'path':str(p),'sha256':sha(p)})
 stages={
  'pilot':{'jobs':[prompts[0]],'decode_steps':8,'threads':8},
  'main':{'jobs':prompts,'decode_steps':16,'threads':8}}
 for stage,data in stages.items():
  (OUT/f'{stage}.jobs.tsv').write_text(''.join(f"{j['path']}\t{j['id']}\n" for j in data['jobs']))
 files=[CPP,SELF,BIN,PLAN,OUT/'placement.tsv',OLD_BUILD]
 files += [pathlib.Path(x) for x in argv if x.endswith('.a')]
 source_root=NFS/'sources/llama.cpp-927dada6c'
 files += [source_root/'ggml/src/ggml-cpu/ggml-cpu.c',source_root/'ggml/include/ggml-cpu.h',source_root/'ggml/src/ggml-common.h',source_root/'src/models/granite-hybrid.cpp']
 hashes={str(p):sha(p) for p in files};hashes[str(MODEL)]=sha(MODEL)
 assert hashes[str(MODEL)]=='5a38b08c441ae1adbafb1d2b8a7167e0d48734d83af68b268cefea1eec553dcd'
 assert hashes[str(PLAN)]=='a55f17010b7b48c6c396dee92bb1d127a541e28a93b94aee2e42aac9f3dc2538'
 write(OUT/'frozen_protocol.json',{
  'frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_commit':'927dada6c9143ba4c940b72004d2698fa5e4e930','input_sha256':hashes,'stages':stages,
  'scope':'All 40-layer routed and shared gate/up/down FFN matrices during every prefill and decode. Attention, router and SSM remain original GGUF host operators.',
  'numeric_contract':{'weights':'Original GGUF to_float, fixed per-row symmetric signed W8 [-127,127]','activation':'Each whole original F32 vector dynamic symmetric signed A8 [-127,127], once before dot','scale':'F32 maxabs/127, all-zero scale 1','rounding':'F32 division then std::round halfway away from zero, clip [-127,127]','dot':'Exact I64 accumulation over all columns; no column partials','output':'One F32 conversion and one multiplication by F32(rowScale*inputScale); no fast-math or FMA contraction'},
  'physical_contract':{'routed':'Frozen expert chip-prefix row counts; gate/up interpreted as concatenated gate512 + up512; whole rows stored in actual per-chip buffers, reversed logical row order within each chip; compute bytes then scatter to logical row','shared':'One copy, equal contiguous full-row counts across16 chips, reversed physical slots, same packed layout computed in BOTH modes','unsharded':'Canonical row buffers for routed matrices; identical fixed shared packed buffers','placed':'Actual chip packed buffers for all routed matrices, plus same fixed shared buffers'},
  'verification':['Every placed I64 row dot compared directly with canonical same input bytes','Every intermediate FFN F32 output compared byte-exact across independently initialized unsharded and placed runs','Every vocabulary logit compared byte-exact after prefill and each decode','Every route and generated token compared between modes','Explicit six projections x40 layers xprefill/decode coverage; no prefill state reuse'],
  'harness_revision':2,'prefill_requests_logits_for_all_tokens_to_cover_last_layer_FFN':True,'test_prompts_read':False,'new_quantization_method_claimed':False,'native_aif_output_abi_claimed':False,'original_checkpoint_numerical_equivalence_claimed':False,'cpu_wall_times_are_flash_performance':False})
 print('Frozen protocol written before runs:',OUT/'frozen_protocol.json',flush=True)
def audit(stage):
 path=OUT/stage;frozen=json.loads((OUT/'frozen_protocol.json').read_text());spec=frozen['stages'][stage];report=json.loads((path/'numeric_report.json').read_text());rows=[json.loads(l) for l in (path/'coverage_ledger.jsonl').read_text().splitlines()]
 bymode={m:[r for r in rows if r['mode']==m] for m in ['unsharded','placed']};assert len(bymode['unsharded'])==len(bymode['placed'])
 mismatches=[]
 compare_fields=['job','phase','step','weight','layer','projection','op','weight_shape','input_shape','output_shape','tokens','selected_experts','output_floats','input_fnv1a64','expert_ids_fnv1a64','output_fnv1a64']
 for a,b in zip(bymode['unsharded'],bymode['placed']):
  if any(a[k]!=b[k] for k in compare_fields):mismatches.append({'unsharded':a,'placed':b})
 assert not mismatches,mismatches[:1]
 coverage=collections.Counter((r['mode'],r['job'],r['phase'],r['weight']) for r in rows)
 expected=[]
 for mode in bymode:
  for job in spec['jobs']:
   for layer in range(40):
    for p in PROJECTIONS:
     for phase,count in [('prefill',1),('decode',spec['decode_steps'])]:
      key=(mode,job['id'],phase,f'blk.{layer}.ffn_{p}.weight');actual=coverage[key];assert actual==count,(key,actual,count);expected.append({'mode':mode,'job':job['id'],'phase':phase,'weight':key[-1],'expected_callbacks':count,'actual_callbacks':actual})
 route_checks=[]
 for job in spec['jobs']:
  id=job['id'];a=path/'traces'/f'{id}.unsharded.routing.jsonl';b=path/'traces'/f'{id}.placed.routing.jsonl';assert a.read_bytes()==b.read_bytes();sa=json.loads((path/'traces'/f'{id}.unsharded.summary.json').read_text());sb=json.loads((path/'traces'/f'{id}.placed.summary.json').read_text());assert sa['generated_token_ids']==sb['generated_token_ids']
  route_checks.append({'job':id,'prompt_tokens':sa['prompt_tokens'],'decode_tokens':spec['decode_steps'],'route_records_per_mode':len(a.read_text().splitlines()),'routes_byte_exact':True,'generated_token_ids_identical':True,'route_sha256':sha(a),'reference_ffn_outputs_sha256':sha(path/'traces'/f'{id}.unsharded.ffn_outputs.f32'),'reference_logits_sha256':sha(path/'traces'/f'{id}.unsharded.logits.f32')})
 assert report['integer_mismatches']==report['same_input_scaled_bit_mismatches']==report['cross_run_output_bit_mismatches']==report['cross_run_logits_bit_mismatches']==0
 assert len(coverage)==len(expected)
 result={'status':'PASS','stage':stage,'frozen_protocol_sha256':sha(OUT/'frozen_protocol.json'),'fully_covered_FFN_weight_names':240,'expected_coverage_keys':len(expected),'observed_coverage_keys':len(coverage),'callback_records':len(rows),'cross_run_operator_records_identical':True,'routes':route_checks,'integer_checks':report['integer_checks'],'cross_run_output_floats':report['cross_run_output_floats'],'cross_run_logits_floats':report['cross_run_logits_floats'],'integer_mismatches':0,'cross_run_output_bit_mismatches':0,'cross_run_logits_bit_mismatches':0,'scope':'CPU functional same-format placement equivalence; no quality or native AiF ABI inference'}
 write(path/'independent_audit.json',result);write(path/'expected_coverage.json',expected);print(json.dumps(result,indent=2),flush=True)
def run(stage):
 frozen=json.loads((OUT/'frozen_protocol.json').read_text());spec=frozen['stages'][stage]
 for p in [str(CPP),str(SELF),str(BIN),str(PLAN),str(OUT/'placement.tsv')]:assert sha(p)==frozen['input_sha256'][p],p
 for job in spec['jobs']:assert sha(job['path'])==job['sha256']
 dest=OUT/stage
 if dest.exists():raise RuntimeError('Never overwrite an existing experiment stage')
 dest.mkdir();cmd=[str(BIN),str(MODEL),str(OUT/f'{stage}.jobs.tsv'),str(spec['decode_steps']),str(spec['threads']),str(OUT/'placement.tsv'),str(dest)]
 write(dest/'run_command.json',{'argv':cmd,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'frozen_protocol_sha256':sha(OUT/'frozen_protocol.json')})
 env=dict(os.environ,OMP_NUM_THREADS=str(spec['threads']),OMP_DYNAMIC='FALSE',OPENBLAS_NUM_THREADS='1')
 t=time.monotonic()
 with open(dest/'stdout.log','w') as stdout,open(dest/'stderr.log','w') as stderr:r=subprocess.run(cmd,stdout=stdout,stderr=stderr,env=env)
 write(dest/'completion.json',{'exit_code':r.returncode,'wall_seconds':time.monotonic()-t,'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
 if r.returncode:raise RuntimeError(f'{stage} failed; inspect stderr.log (failure retained)')
 audit(stage)
def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','pilot','main','audit-pilot','audit-main']);a=p.parse_args()
 if a.action=='prepare':prepare()
 elif a.action.startswith('audit-'):audit(a.action[6:])
 else:run(a.action)
if __name__=='__main__':main()
