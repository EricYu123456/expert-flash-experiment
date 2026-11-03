#!/usr/bin/env python3
"""Existing, unchanged source-GGUF trace executable; disclosure control only."""
import datetime,hashlib,json,os,pathlib,subprocess,time
ROOT=pathlib.Path(__file__).resolve().parent
CONTROL=ROOT/'source-gguf-control'
BINARY=pathlib.Path('/home/nfs_client/ericyu/expert_flash_experiment/build/llama-route-trace/llama-corpus-trace')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
if CONTROL.exists():raise RuntimeError('Never overwrite prior control')
CONTROL.mkdir();(CONTROL/'traces').mkdir()
frozen=json.loads((ROOT/'frozen_protocol.json').read_text());spec=frozen['stages']['main'];model='/home/nfs_client/ericyu/expert_flash_experiment/models/granite4tiny-q4_k_m.gguf'
(CONTROL/'jobs.tsv').write_text(''.join(f"{j['path']}\t{CONTROL/'traces'/j['id']}\n" for j in spec['jobs']))
cmd=[str(BINARY),model,str(CONTROL/'jobs.tsv'),'16','8','1','trace']
protocol={'frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'argv':cmd,'existing_binary_sha256':sha(BINARY),'existing_source_sha256':sha('/home/ericyu/expert_flash_experiment/scripts/llama_corpus_trace.cpp'),'control_driver_sha256':sha(__file__),'main_frozen_protocol_sha256':sha(ROOT/'frozen_protocol.json'),'prompts':spec['jobs'],'purpose':'Disclose source-GGUF vs fixed full-FFN W8A8 greedy-token changes; not a quality benchmark or placement-equivalence requirement','prefill_logits_requested':'Existing source-GGUF trace default: final token only; W8 reference requests all for coverage','reference_format':'Source GGUF for every operator, no W8A8 callback replacements','batch_sequences':1,'steps':16,'threads':8}
write(CONTROL/'frozen_control_protocol.json',protocol)
for j in spec['jobs']:assert sha(j['path'])==j['sha256']
t=time.monotonic()
with open(CONTROL/'stdout.log','w') as out,open(CONTROL/'stderr.log','w') as err:r=subprocess.run(cmd,stdout=out,stderr=err,env=dict(os.environ,OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='1'))
write(CONTROL/'completion.json',{'exit_code':r.returncode,'wall_seconds':time.monotonic()-t,'binary_unchanged':sha(BINARY)==protocol['existing_binary_sha256']})
assert r.returncode==0
comparisons=[]
for job in spec['jobs']:
 id=job['id'];gguf=json.loads((CONTROL/'traces'/f'{id}.trace.summary.json').read_text());w8=json.loads((ROOT/'main/traces'/f'{id}.unsharded.summary.json').read_text());assert gguf['prompt_token_ids']==w8['prompt_token_ids']
 a=gguf['generated_token_ids'];b=w8['generated_token_ids'];assert len(a)==len(b)==16
 agreement=[x==y for x,y in zip(a,b)];prefix=next((i for i,v in enumerate(agreement) if not v),16)
 ra=[json.loads(l) for l in (CONTROL/'traces'/f'{id}.trace.routing.jsonl').read_text().splitlines()];rb=[json.loads(l) for l in (ROOT/'main/traces'/f'{id}.unsharded.routing.jsonl').read_text().splitlines()];rb=[x for x in rb if x['phase']=='decode'];assert len(ra)==len(rb)==640
 matching_routes=sum(x['expert_ids']==y['expert_ids'] for x,y in zip(ra,rb))
 comparisons.append({'job':id,'prompt_tokens':len(gguf['prompt_token_ids']),'compared_generated_positions':16,'generated_token_id_agreement':sum(agreement),'generated_token_id_agreement_fraction':sum(agreement)/16,'identical_generated_prefix_tokens':prefix,'source_gguf_token_ids':a,'full_ffn_w8a8_token_ids':b,'source_gguf_text':(CONTROL/'traces'/f'{id}.trace.output.txt').read_text(),'full_ffn_w8a8_text':(ROOT/'main/traces'/f'{id}.unsharded.output.txt').read_text(),'decode_layer_route_records':640,'identical_selected_expert_sets_ordered':matching_routes,'route_comparison_condition':'Free-running continuations; inputs differ after generated-token divergence, so route changes are not isolated quantization error.'})
write(CONTROL/'comparison.json',{'status':'complete','purpose':protocol['purpose'],'unchanged_source_gguf_trace_binary':True,'comparisons':comparisons,'quality_inference_claimed':False,'placement_claim_conditioned_on_source_GGUF_agreement':False})
print(json.dumps(comparisons,indent=2))
