#!/usr/bin/env python3
"""Frozen new-root ABI crosscheck against independent sealed wave-drain costs."""
from pathlib import Path
from copy import deepcopy
import hashlib,json,shutil,sys
import numpy as np
ROOT=Path('/home/ericyu/expert_flash_experiment')
BASE=ROOT/'results/research-2026-11-02/round2'
DEST=Path(__file__).resolve().parent
WAVE=BASE/'wave-drain'
PLANS=BASE/'ordinary-read-compiler/full-campaign-v4-deduplicated'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,x):(DEST/n).write_text(json.dumps(x,indent=2)+'\n')

def main():
 assert not (DEST/'protocol.frozen.json').exists()
 snapshot=DEST/'snapshots';snapshot.mkdir()
 inputs={str(Path(__file__)):sha(Path(__file__))}
 for name in ('research_20261103_ordinary_read_backend.py','research_20261103_ordinary_read_native.cpp','research_20261102_whole_row_compiler.py'):
  p=ROOT/'scripts'/name;raw=p.read_bytes();(snapshot/name).write_bytes(raw);inputs[str(p)]=hashlib.sha256(raw).hexdigest()
 for p in (WAVE/'protocol.frozen.json',WAVE/'costs.npz',WAVE/'summary.json',BASE/'ordinary-read-compiler/independent/inconsistent_physical_metadata.json'):inputs[str(p)]=sha(p)
 records=[]
 for m in ('qwen38','granite4tiny','deepseek'):
  for l in range(3):
   p=PLANS/f'{m}.L{l}.cmd250/summary.json';d=json.loads(p.read_text());inputs[str(p)]=sha(p);records.append(d)
   for a in d['artifacts']:
    p=ROOT/a['arrays'];assert sha(p)==a['arrays_sha256'];inputs[str(p)]=a['arrays_sha256']
 write('protocol.frozen.json',dict(inputs_sha256=inputs,scope='Compare new root18field ServiceABI input_channel_copies1/4 against already-sealed independent postcompute fanout1/4 costs for all27old-v4 plans x64queries. Replay malformed physical metadata counterexample against snapshotted rootguard. No fit/global-campaign/event semantic changes.',numeric='Fixed I8 coefficients and original output_fringe1 service only; no PE numeric implementation or new calibration.'))
 sys.path.insert(0,str(snapshot))
 from research_20261103_ordinary_read_backend import Service,NativeReplay,lower_rows,physical_addresses
 assert Service().input_channel_copies==1 and len(Service().array())==18 and Service(input_channel_copies=4).array()[17]==4
 native=NativeReplay(DEST/'build');results=[]
 with np.load(WAVE/'costs.npz') as reference:
  for d in records:
   for a in d['artifacts']:
    label=f"{d['model']}.L{d['layer']}.{a['family']}"
    with np.load(ROOT/a['arrays']) as f:plan={k:f[k].copy() for k in ('offsets','jobs','input_columns','total_rows')};routes=f['selection_routes'].copy()
    for copies in (1,4):
     s=Service(**{**d['service'],'input_channel_copies':copies});got=native.run(plan,routes,d['shape'][2],1,s,trace=d['layer']==0)
     expected=reference[label+f'.postcompute_fanout{copies}'];assert np.array_equal(got['cost'],expected)
     trace_match=None
     if d['layer']==0:
      p=WAVE/(label+f'.postcompute_fanout{copies}.first_trace.npz');inputs[str(p)]=sha(p)
      with np.load(p) as f:assert np.array_equal(got['trace'],f['trace'])
      trace_match=True
     results.append(dict(label=label,input_channel_copies=copies,queries=len(routes),max_cost_difference_ns=0,first_query_full_trace_match=trace_match))
 # Reconstruct the exact archived counterexample; generated positive control must remain legal.
 original=json.loads((BASE/'ordinary-read-compiler/independent/inconsistent_physical_metadata.json').read_text());s=Service(**original['service'])
 base=lower_rows([[2]],[[2]],1,1,s);positive=physical_addresses(base,s);assert positive['chip_blocks']==[2] and positive['allocated_lsb_block_bytes']==4
 tests=[]
 def reject(name,change):
  p=deepcopy(base);change(p)
  try:physical_addresses(p,s)
  except (AssertionError,KeyError,TypeError,ValueError,IndexError) as e:tests.append(dict(name=name,rejected=True,exception=type(e).__name__,message=str(e)))
  else:raise AssertionError('Malformed physical metadata accepted: '+name)
 reject('archived_logical_waves_shrunk',lambda p:[x.update(waves=1) for x in p['logical']])
 for key,value in [('expert',1),('phase',1),('chip',1),('rows',1),('columns',2),('useful_bytes',1),('pages',1),('waves',1),('row_start',1),('column_start',1)]:
  reject('logical_'+key,lambda p,k=key,v=value:p['logical'][0].update({k:v}))
 reject('wrong_total_rows',lambda p:p['total_rows'].__setitem__((0,0),1))
 try:physical_addresses(base,s,chip_blocks=1)
 except AssertionError:tests.append(dict(name='insufficient_per_chip_bundle_budget',rejected=True))
 else:raise AssertionError('Capacity guard accepted1bundle for2bundle payload.')
 assert physical_addresses(base,s,chip_blocks=2)==positive
 # Existing transfer accounting rounds a combined byte budget, whereas our
 # earlier sensitivity rounded each plane copy independently. Real shapes
 # here align, but the distinction is documented rather than hidden.
 rounding=dict(example_full_input_bytes=1,input_copies=4,channel_bytes_per_second=2_000_000_000,root_combined_transfer_ns=s.transfer(4,2_000_000_000),prior_per_copy_rounding_ns=4*s.transfer(1,2_000_000_000),general_difference_bound_ns=3,actual27plans_difference_ns=0)
 assert all(sha(p)==h for p,h in inputs.items())
 write('results.json',dict(status='PASS newroot copy1/copy4 and physical importguard',costs_exact=sum(x['queries'] for x in results),first_query_full_traces_exact=sum(x['first_query_full_trace_match'] is True for x in results),records=results,guard_regressions=tests,positive_control_certificate=positive,rounding_contract=rounding,native_provenance=native.provenance))
 write('notes.json',dict(status='PASS',costs_exact=3456,traces_exact=18,guards_rejected=len(tests),source_sha256={k:v for k,v in inputs.items() if '/scripts/' in k},findings=['Default input_channel_copies=1 is backward compatible for all sealed27plans/64queries. Copies4 exactly matches the independently sealed postcompute fanout4 vectors.','Cfg18th field only multiplies ONFI full-vector job input bytes; PCIe common GU/private Down uploads and native command counts are unchanged. Full trace matches on18layer0profiles guard against hidden cost-axis changes.','The archived physical metadata shrink counterexample is now rejected, as are independent expert/phase/chip/rows/C/usefulbytes/pages/waves/rowstart/columnstart/totalrows mutations. Normal two-block placement and sufficient capacity pass.','For arbitrary fractional-nanosecond transfers, ceil(copies*C/rate) differs from copies*ceil(C/rate). All actual columns here align, so this causes zero discrepancy. Neither rounding model is native input distribution calibration.'],limits='No global/v5/v6/actual W8 capture quality claim here. This crosscheck is on old27frozen-v4 timing plans. New global capacity/training provenance audited separately.'))
 write('completion.json',dict(status='complete',outputs_sha256={str(p):sha(p) for p in sorted(DEST.rglob('*')) if p.is_file() and '__pycache__' not in str(p) and p.name!='completion.json'}))
 print(json.dumps({'costs':3456,'traces':18,'guards':len(tests),'backend_sha256':inputs[str(ROOT/'scripts/research_20261103_ordinary_read_backend.py')],'native_sha256':inputs[str(ROOT/'scripts/research_20261103_ordinary_read_native.cpp')],'notes_sha256':sha(DEST/'notes.json')},indent=2))

if __name__=='__main__':main()
