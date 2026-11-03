#!/usr/bin/env python3
"""Read-only bridge from TRAIN canonical capture to new all-FFN canonical smoke."""
import hashlib,json,pathlib
import numpy as np
from research_20261103_selected_plan_numeric import BASE,OUT,sha
pid='train_programming_short_015';reports={}
for name,vocab in [('granite4tiny',100352),('deepseek',102400)]:
 train=BASE/'w8-format-training'/name/'batch3';smoke=OUT/'smoke-primary'/('input1-'+name)/'run';a=json.loads((train/'traces'/f'{pid}.unsharded.summary.json').read_text());b=json.loads((smoke/'traces'/f'{pid}.unsharded.summary.json').read_text());assert a['prompt_token_ids']==b['prompt_token_ids'] and a['generated_token_ids'][:4]==b['generated_token_ids']
 rows=lambda p:[json.loads(raw) for raw in p.read_text().splitlines()]
 old=[r for r in rows(train/'traces'/f'{pid}.unsharded.routing.jsonl') if r['phase']=='prefill' or r['step']<4];new=rows(smoke/'traces'/f'{pid}.unsharded.routing.jsonl');assert old==new
 all_logits=np.memmap(smoke/'traces'/f'{pid}.unsharded.logits.f32',mode='r',dtype=np.uint32).reshape(5,vocab)
 for index,suffix in enumerate(['prefill.-1','decode.0']):assert np.array_equal(np.fromfile(train/'snapshots'/f'{pid}.{suffix}.logits.f32',dtype=np.uint32),all_logits[index])
 reports[name]=dict(status='pass',TRAIN_prompt_id=pid,prefill_and_first4_decode_route_records_exact=len(new),generated_prefix_tokens_exact=4,last_prefill_and_first_decode_logit_values_bitexact=2*vocab,scope='TRAIN full-FFN canonical-only capture and selected-layout allFFN canonical have the same numeric reference on this bridge; not originalGGUF/quality equivalence.')
r=dict(status='pass',source_sha256=sha(__file__),capture_protocol_sha256=sha(BASE/'w8-format-training/protocol.frozen.json'),smoke_protocol_sha256=sha(OUT/'smoke-primary/protocol.frozen.json'),models=reports);p=OUT/'smoke-primary/TRAIN_reference_bridge.json';assert not p.exists();p.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(reports,indent=2))
