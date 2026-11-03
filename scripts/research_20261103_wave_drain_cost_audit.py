#!/usr/bin/env python3
"""Postprocess the frozen bounded wave-drain run; no event replays or refits."""
from pathlib import Path
from collections import defaultdict
import csv,hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'results/research-2026-11-02/round2/wave-drain'
BASE=ROOT/'results/research-2026-11-02/round2/ordinary-read-compiler/full-campaign-v4-deduplicated'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,x):(D/n).write_text(json.dumps(x,indent=2)+'\n')
s=json.loads((D/'summary.json').read_text());ledgers=[];deltas=[]
with np.load(D/'costs.npz') as z:
 for model in ('qwen38','granite4tiny','deepseek'):
  for family in ('disjoint','cyclic','whole_rows'):
   totals=defaultdict(lambda:np.zeros(64,dtype=np.int64))
   for layer in range(3):
    summary=json.loads((BASE/f'{model}.L{layer}.cmd250/summary.json').read_text());d,i,e,k,layers,si=summary['shape'];a=next(x for x in summary['artifacts'] if x['family']==family)
    with np.load(ROOT/a['arrays']) as f:jobs=f['jobs'];o=f['offsets'];C=f['input_columns'];T=f['total_rows'];routes=f['selection_routes']
    assert len(o)==e+1 and np.all(C[:e]==[d,i]) and np.array_equal(C[e],[d,si]) and np.all(T[:e]==[2*i,d]) and np.array_equal(T[e],[2*si,d])
    srv=summary['service']
    for q,route in enumerate(routes):
     chosen=list(map(int,route))+[e]
     totals['GU_common_PCIe_input_bytes'][q]+=d*srv['input_bytes']
     totals['Down_private_PCIe_input_bytes'][q]+=sum(int(C[x,1])*srv['input_bytes'] for x in chosen)
     for x in chosen:
      for ph in range(2):
       lo,hi=o[x,ph]
       for chip,rows,cols,pages,waves in jobs[lo:hi]:
        totals['commands'][q]+=1;totals['ONFI_full_input_bytes_fanout1'][q]+=int(cols)*srv['input_bytes'];totals['NAND_waves'][q]+=int(waves)
        totals['all_page_output_bytes'][q]+=int(pages)*srv['page_bytes']//int(cols)*srv['output_bytes']
   ledgers.append(dict(model=model,family=family,layers=3,queries=64,mean_inventory={n:float(v.mean()) for n,v in totals.items()},ONFI_input_bytes_fanout4_mean=float(totals['ONFI_full_input_bytes_fanout1'].mean()*4),extra_input_bytes_fanout4_mean=float(totals['ONFI_full_input_bytes_fanout1'].mean()*3)))
   for fanout in (1,4):
    before=sum(z[f'{model}.L{l}.{family}.postcompute_fanout{fanout}'] for l in range(3));after=sum(z[f'{model}.L{l}.{family}.perwave_fanout{fanout}'] for l in range(3));delta=after-before
    deltas.append(dict(model=model,family=family,fanout=fanout,perwave_minus_postcompute_mean_ns=float(delta.mean()),min_ns=int(delta.min()),max_ns=int(delta.max()),max_absolute_ns=int(abs(delta).max()),faster=int(np.count_nonzero(delta<0)),equal=int(np.count_nonzero(delta==0)),slower=int(np.count_nonzero(delta>0))))
write('cost_audit.json',dict(status='PASS exact known model dims and complete raw ledger',records=ledgers,three_layer_drain_cost_deltas=deltas,source_sha256=sha(Path(__file__))))
notes={
 'status':'PASS bounded frozen-v4 native-cost service-contract sensitivity',
 'result':'Per-wave ONFI output draining does not materially change these three-layer means. Four independent plane input transfers reverse the small G4 whole-row advantage in this fixed-plan screen. No full-campaign expansion performed.',
 'data_and_checks':{'models':['qwen38','granite4tiny','deepseek'],'layers':[0,1,2],'training_queries':64,'frozen_plans':27,'profiles':4,'costs':6912,'original_postcompute_costs_exactly_reproduced':1728,'real_trace_checks':108,'tiny_scalar_trace_checks':96,'tiny_cases':24,'all_pass':True,'tiny_scope':'1-4chips/planes,smallpages,variableC including C>wavebytes,rowstraddling,lastpartialwave,zero command/read/host durations and both phase priorities. Independent scalar heap scheduler matches full traces. Separate interval/dependency/owner/full-GU audit checks trace legality.'},
 'three_layer_whole_reduction_vs_cyclic':[{'model':r['model'],'profile':r['profile'],'reduction':r['whole_reduction_vs_cyclic'],'whole_faster_queries':r['whole_faster_queries']} for r in s['three_layer_aggregates']],
 'native_contract':[
  'Retain original int32 five-field job descriptors: chip, complete rows, full columns, requested physical page count, wave count. Preserve full-column row functions and all frozen chip assignments/labels; no column split, numerical merge, output quantization, additional context or native command is introduced.',
  'Each command acquires one chip owner, loads full input once, reads ordinary waves and retains ownership through final ONFI output drain. The four-plane input profile serializes four full-vector input transfers; it leaves common GU PCIe upload once and private Down PCIe upload once per selected expert.',
  'After wave w, completed rows=floor(min(w*planes*pagebytes,requested_pages*pagebytes)/fullC). Emit the new interval since w-1, including all requested-page fringe rows. Final wave uses actual requested pages, not a full-wave byte allowance. Independently enumerate each completed row endpoint to verify chunk coverage and useful-row identity.',
  'Perwave backpressure: the next array wave begins after the previous ONFI drain; PCIe delivery may overlap later array reads and is explicitly ordered within each job. GUready waits all job outputs at host, including the conservatively retained page fringe. Host activation, new Down submit and complete A8 Down upload follow full GU readiness. No neuron producer-consumer pipeline is used.',
  'Perwave transfers round channel and PCIe durations separately. All families use the same new service semantics and same fixed resource-priority policy; no geometry or labels are refitted for input fanout4.'
 ],
 'buffer_batch_sizes_bytes':{model:{'postcompute_max_job_emitted_batch':max(r['max_postcompute_batch_bytes'] for r in s['geometry'] if r['label'].startswith(model+'.')),'perwave_max_emitted_batch':max(r['max_perwave_batch_bytes'] for r in s['geometry'] if r['label'].startswith(model+'.'))} for model in ('qwen38','granite4tiny','deepseek')},
 'hardware_evidence':{'primary':'AiF ISCA2025 DOI10.1145/3695053.3731073, section4.4/Figure14, original paper/aif.txt lines776-823.','confirmed':'The controller supplies input dimension/block/offset/pages; input is loaded into plane PBs; matrix segments flow sequentially; output elements enter FIFO in order and are polled through SREG as generated.','unverified':'FIFO capacity, full/ready backpressure signal, command-internal pause at a wave boundary, scheduling/control of FIFO drain, plane input broadcast versus independent DATAIN copies, input/output dtype/scale/accumulator ABI, and measured PE throughput are not publicly established by the inspected source.','interpretation':'This finer service assumption is not proven existing-hardware feasibility or RTL/device execution. It examines the FIFO/input fanout dependency of the software replay.'},
 'limits':[
  'Even perwave drain can need up to one wave output batch in the finite FIFO/PE path. The maxima here are408B Qwen,512B G4,188B Deep. These are model requirements, not verified hardware capacities. An even smaller FIFO needs row-level backpressure evidence, which this model does not create.',
  'The old postcompute batch requirement is816B Qwen,1536B G4,1408B Deep among these27 frozen plans, so the potential buffer dependency is explicitly bounded here rather than assumed to equal a full FFN output.',
  'One charged command is the GEMV command. SREG polling, per-wave control, extra DATAIN addressing/command phases and host callbacks are not separately calibrated or charged. Therefore fanout4 is a bandwidth sensitivity scenario, not an upper bound on every implementation overhead.',
  'Read/ECC/PE use the same ordinary28us-wave service approximation. No new MAC latency, fixed-point numerical output contract, physical FIFO design or native data-path experiment is claimed.',
  'Three-layer development routes only, not heldout validation or full-model speed. Full48/40/26 costs are not extrapolated. G4 weak gain is sensitive to input distribution; fitting layouts again under that profile would be a separate experiment.',
  'Small output availability differences can perturb the fixed greedy scheduler and make a perwave profile slightly slower or faster. No claim that the old scheduler is optimal or that every native interleaving is modeled.'
 ],
 'files':{'full_results':'summary.json','actual_cost_arrays':'costs.npz','inventories_and_delta_counts':'cost_audit.json','protocol':'protocol.frozen.json','tiny_checks':'scalar_checks.json','native_cpp':'scripts/research_20261103_wave_drain_native.cpp','driver':'scripts/research_20261103_wave_drain.py','postprocess':'scripts/research_20261103_wave_drain_cost_audit.py'},
 'next_step':'Document input fanout as a capability/calibration parameter and retain the G4 reversal. Perwave drain alone does not justify a broader campaign here; first obtain native FIFO/backpressure/input-distribution documentation or measured evidence.'
}
write('notes.json',notes)
with (D/'three_layer_summary.csv').open('w') as f:
 fields=['model','profile','cyclic_mean_ms','whole_mean_ms','whole_reduction_vs_cyclic','whole_faster_queries'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
 for r in s['three_layer_aggregates']:w.writerow(dict(model=r['model'],profile=r['profile'],cyclic_mean_ms=r['means_ns']['cyclic']/1e6,whole_mean_ms=r['means_ns']['whole_rows']/1e6,whole_reduction_vs_cyclic=r['whole_reduction_vs_cyclic'],whole_faster_queries=r['whole_faster_queries']))
write('completion.json',dict(status='complete',outputs_sha256={str(p):sha(p) for p in sorted(D.rglob('*')) if p.is_file() and '__pycache__' not in str(p) and p.name!='completion.json'},postprocess_sha256=sha(Path(__file__))))
print(json.dumps({'notes_sha256':sha(D/'notes.json'),'summary_sha256':sha(D/'summary.json'),'cost_audit_sha256':sha(D/'cost_audit.json')},indent=2))
