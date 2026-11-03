#!/usr/bin/env python3
"""Summarize the completed independent replay; no new replay or fitting."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'results/research-2026-11-02/round2/ordinary-read-compiler/independent/second-heldout-v2'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(name,x):(D/name).write_text(json.dumps(x,indent=2)+'\n')
r=read(D/'results.json')
assert r['status']=='PASS'
cyclic=[x for x in r['paired'] if x['baseline']=='cyclic']
prompt=read(D/'per_prompt.recomputed.json')
for x in cyclic:
    a=[y['complete_ffn_mean_reduction_fraction'] for y in prompt if all(y[k]==x[k] for k in ('model','input_channel_copies','command_ns','baseline'))]
    x['minimum_prompt_mean_gain']=min(a);x['maximum_prompt_mean_gain']=max(a)
notes={
 'status':'PASS full second-heldout-v2 independent service-cost and statistics audit',
 'result':'All202752 saved costs,202752 lower bounds,792 complete first traces(675372nodes),24 pooled totals,256 per-prompt records and32 dataset records are reproduced. All792 traces independently pass dependency/resource/chip-owner/full-GU-barrier checks;24 representative native traces plus4 dense-prefix traces equal separate Python heap-scheduler traces.',
 'root_protocol_sha256':'cb3b22789217c6fc2b1658ac92537b9efe4f275cc347e213b6898ae6af764c96',
 'plan_anchor_sha256':'00507ff56f9197fb18511ad58303919214c03cba6f0c1440f6733ffc42944d00',
 'independent_results_sha256':sha(D/'results.json'),
 'inspection':[
   'Inspected the entire frozen root replay driver, including capture_routes, worker, dense_prefix and summarize. Its saved snapshot hash e6fd9082bc7ca2e283a0d8d9564e2c69b293c37f6df5c91ee81baeda3f864fbf equals the protocol source hash. The audit imports only the older sealed independent wave-drain engine, not root backend or its C++ library.',
   'Verified all1609 frozen input hashes, all root source-snapshot hashes, numeric-bound route hashes, all792 result artifact hashes and four dense-prefix array/address hashes. Checked all1143424 per-layer plan job descriptors against independent ceil-page/ceil-wave arithmetic, contiguous logical row identity, original input dimensions, complete row totals and one-block limits.',
   'Every decoded route contains exactly six distinct indices in0..63. Each layer independently appends expert64 exactly once. One GU common PCIe upload serves seven experts; each of seven experts has one activation, one full private Down upload and its own complete GU-host-delivery barrier. No shared omission or repetition found.',
   'Every raw MoE layer appears exactly once: Deep raw1..26, G4 raw0..39. Deep raw0 dense prefix appears only in final per-token sums, once per family, with common profile-specific cost. No prefix occurs inside the MoE/shared layer sum. G4 has zero prefix.',
   'Deep prefix GU is the cost-equivalent concatenation of10944gate+10944up rows:16*1368 complete rows withC2048;Down16*128rows withC10944. Numeric validation elsewhere retains separate gate/up. Its useful payload3*2048*10944=67239936B is verified; separate prefix physical map is not claimed to be jointly packed with the MoE maps.',
   'The independent engine uses fanout*ceil(single-input-transfer-time), root uses ceil(fanout*input-bytes/rate). Explicitly checked these are equal for every actual job; arbitrary odd/artificial input sizes need not be equivalent. Fanout multiplies ONFI input reload only, not shared PCIe input or command count.'
 ],
 'cyclic_comparisons_complete_ffn':cyclic,
 'deep_fixed_prefix_four_profiles':r['prefix_checks'],
 'statistics':'Sixteen prompts are8 OpenBookQA plus8 BoolQ;16 ordered decode positions per prompt. Independent vectors reproduce pooled means/P50/P95/P99/min/max, paired token/prompt counts and both dataset summaries. These256 positions are correlated, not256 independent task samples. No filtering or refit was performed.',
 'important_limits':[
   'Deep is faster than cyclic on all256 tokens and all16 prompt means in all four cost profiles. G4 at250ns/copy1 has3 slower tokens;copy4 has66 slower tokens, despite all16 prompt means improving. At5000ns G4 whole/cyclic costs tie on every token. Preserve all of these cases.',
   'Copy4 G4 low-command gain is0.5797469%, about0.145730ms mean service. Omitted activation/combine, host row-scale/gather and uncalibrated native control costs could affect this small margin. This audit does not charge an arbitrary host tax or infer measured device speed.',
   'The completed route-blind capacity symmetry control makes low-command G4 whole6.0GiB versus cyclic5.8125GiB;the6.0GiB floor is proven only inside the audited layerwise swap8 family. Thus this cost audit does not establish same-capacity superiority. Deep dense-prefix allocation remains separately reported.',
   'Full numerical AiF output/scale/rounding ABI, input broadcast, finite FIFO/backpressure and PE throughput remain unverified hardware contracts. Postcompute scheduling reproduces the declared model;it does not resolve those capabilities.',
   'The independent C++ engine shares a historical scheduler ancestor with root. Separate Python graph construction/heap scheduling and explicit trace invariants help detect implementation mistakes but do not independently establish the common service assumptions. No claim of globally optimal scheduling or real hardware timing.',
   'No model inference, model download, task-accuracy evaluation or original frozen-plan modification was performed.'
 ],
 'files':{'full_results':'results.json','vectors':'independent_total_vectors.npz','per_case_hashes':'layer_ledger.json','python_heap_checks':'scalar_trace_checks.json','per_prompt':'per_prompt.recomputed.json','dataset':'dataset_groups.recomputed.json','preexecution_protocol':'protocol.frozen.json'}
}
write('notes.json',notes)
rows=['獨立 heldout 成本審核通過。全部202,752成本與下界、792份完整trace（675,372節點），以及所有模型／每題／dataset統計均重現。檢查未發現shared或Deep dense prefix漏算／重算。','',
      '| Model | ONFI copies | Command ns | 完整FFN平均改善 vs cyclic | 慢token | 改善prompt均值 |',
      '|---|---:|---:|---:|---:|---:|']
for x in cyclic:
 rows.append(f"| {x['model']} | {x['input_channel_copies']} | {x['command_ns']} | {x['complete_ffn_mean_reduction_fraction']:.5%} | {x['slower_tokens']}/256 | {x['faster_prompt_means']}/16 |")
rows+=['','Deep四種固定prefix成本：copy1/250ns=1.852224ms，copy1/5000ns=1.994236ms，copy4/250ns=1.888128ms，copy4/5000ns=2.013724ms；每token只加一次。','',
       '這是未校準ordinary-read軟體服務模型的重現，非裝置量測。G4 copy4保留66個變慢token；G4高command全部相等。容量公平性亦未解：swap8強控制後G4 whole6.0GiB、cyclic5.8125GiB。16prompt各16decode位置有相關性，不能視為256獨立task。數值ABI、FIFO、輸入廣播與PE時序仍需硬體證據。']
(D/'notes.md').write_text('\n'.join(rows)+'\n')
write('report_completion.json',{'status':'complete','prior_execution_completion_sha256':sha(D/'completion.json'),'postprocess_source_sha256':sha(Path(__file__)),
       'outputs_sha256':{str(D/p):sha(D/p) for p in ('notes.json','notes.md','results.json','protocol.frozen.json')}})
print(json.dumps({'notes_sha256':sha(D/'notes.json'),'results_sha256':sha(D/'results.json'),'report_completion_sha256':sha(D/'report_completion.json')},indent=2))
