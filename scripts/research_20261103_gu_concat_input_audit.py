#!/usr/bin/env python3
"""Verify actual gate/up callbacks share bit-identical whole-vector A8 input."""
import argparse,collections,json,pathlib
from research_20261103_selected_plan_numeric import OUT,sha
p=argparse.ArgumentParser();p.add_argument('--phase',choices=['smoke','fresh'],required=True);phase=p.parse_args().phase;dest=OUT/('smoke-primary' if phase=='smoke' else 'second-heldout');spec=json.loads((dest/'protocol.frozen.json').read_text());reports={}
for case in spec['cases']:
 path=dest/case/'run/coverage_ledger.jsonl';group=collections.defaultdict(lambda:collections.defaultdict(list))
 with path.open() as f:
  for raw in f:
   r=json.loads(raw);projection=r['projection'];parts=projection.split('_',1);kind=parts[0]
   if kind not in ['gate','up']:continue
   suffix=parts[1] if len(parts)>1 else 'dense';key=r['job'],r['mode'],r['phase'],r['step'],r['layer'],suffix;group[key][kind].append((r['input_shape'],r['tokens'],r['selected_experts'],r['input_fnv1a64'],r['expert_ids_fnv1a64']))
 pairs=0
 for key,values in group.items():assert values['gate']==values['up'],(case,key);pairs+=len(values['gate'])
 reports[case]=dict(status='pass',gate_up_callback_pairs=pairs,input_contract='Input hash includes complete A8 vector bytes and F32 input scale for each full vector. Gate/up repeats are bit-identical; concatenation uses the same full input.',coverage_ledger_sha256=sha(path))
r=dict(status='pass',source_sha256=sha(__file__),protocol_sha256=sha(dest/'protocol.frozen.json'),cases=reports,scope='Read-only numerical compatibility of gate||up native rows; CPU computes separate projections. No physical flash programming or quantization-performance claim.');path=dest/'GU_input_identity.json';assert not path.exists();path.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(reports,indent=2))
