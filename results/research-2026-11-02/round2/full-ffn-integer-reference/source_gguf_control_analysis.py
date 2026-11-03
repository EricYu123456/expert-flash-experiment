#!/usr/bin/env python3
# Analyze the existing binary's actual .traced output label; no rerun or edit to frozen inputs.
import json,pathlib
ROOT=pathlib.Path(__file__).resolve().parent
CONTROL=ROOT/'source-gguf-control'
spec=json.loads((ROOT/'frozen_protocol.json').read_text())['stages']['main']
protocol=json.loads((CONTROL/'frozen_control_protocol.json').read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
comparisons=[]
for job in spec['jobs']:
 id=job['id'];gguf=json.loads((CONTROL/'traces'/f'{id}.traced.summary.json').read_text());w8=json.loads((ROOT/'main/traces'/f'{id}.unsharded.summary.json').read_text());assert gguf['prompt_token_ids']==w8['prompt_token_ids']
 a=gguf['generated_token_ids'];b=w8['generated_token_ids'];assert len(a)==len(b)==16
 agreement=[x==y for x,y in zip(a,b)];prefix=next((i for i,v in enumerate(agreement) if not v),16)
 ra=[json.loads(l) for l in (CONTROL/'traces'/f'{id}.routing.jsonl').read_text().splitlines()];rb=[json.loads(l) for l in (ROOT/'main/traces'/f'{id}.unsharded.routing.jsonl').read_text().splitlines()];rb=[x for x in rb if x['phase']=='decode'];assert len(ra)==len(rb)==640
 matching_routes=sum(x['expert_ids']==y['expert_ids'] for x,y in zip(ra,rb))
 comparisons.append({'job':id,'prompt_tokens':len(gguf['prompt_token_ids']),'compared_generated_positions':16,'generated_token_id_agreement':sum(agreement),'generated_token_id_agreement_fraction':sum(agreement)/16,'identical_generated_prefix_tokens':prefix,'source_gguf_token_ids':a,'full_ffn_w8a8_token_ids':b,'source_gguf_text':(CONTROL/'traces'/f'{id}.traced.output.txt').read_text(),'full_ffn_w8a8_text':(ROOT/'main/traces'/f'{id}.unsharded.output.txt').read_text(),'decode_layer_route_records':640,'identical_selected_expert_sets_ordered':matching_routes,'route_comparison_condition':'Free-running continuations; inputs differ after generated-token divergence, so route changes are not isolated quantization error.'})
write(CONTROL/'comparison.json',{'status':'complete','purpose':protocol['purpose'],'unchanged_source_gguf_trace_binary':True,'comparisons':comparisons,'quality_inference_claimed':False,'placement_claim_conditioned_on_source_GGUF_agreement':False})
print(json.dumps(comparisons,indent=2))
