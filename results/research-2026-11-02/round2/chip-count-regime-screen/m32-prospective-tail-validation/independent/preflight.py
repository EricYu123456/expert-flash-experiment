#!/usr/bin/env python3
"""Read-only static preflight. Does not open test text/routes, infer or price jobs."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import numpy as np

HERE=Path(__file__).resolve().parent
PRODUCER=HERE.parent
ROOT=HERE.parents[5]
ROUND=ROOT/'results/research-2026-11-02/round2'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(name,x):(HERE/name).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def load(p):
 with np.load(p) as z:return {k:z[k] for k in z.files}
def up(n,d):return (int(n)+int(d)-1)//int(d)

P=PRODUCER/'protocol.frozen.json'
assert sha(P)=='911ecbdb77dcd18ccc880870f70a6a231c0a67730bfac7ac9144eea061e9b516'
p=json.loads(P.read_text())
dump('preflight_protocol.frozen.json',{
 'utc':datetime.now(timezone.utc).isoformat(),'source_sha256':sha(__file__),
 'plan_analysis_protocol':str(P),'plan_analysis_protocol_sha256':sha(P),
 'scope':'Static plan/source/metadata preflight only. No inference, fetching datasets, old test text/routes, new capture results or service-cost calculation.',
 'expected_maps':{'free':'incumbent4','common_cyclic':'incumbent0','phase_independent':'incumbent2'},
 'future_binding':'Do not audit new costs until acquisition and capture protocols plus completed capture manifest are frozen and explicitly provided.'})

checks=[]
for ref in p['references']:
 f=Path(ref.get('snapshot',ref['path']));assert sha(f)==ref['sha256']
 checks.append({'path':str(f),'sha256':ref['sha256'],'read_scope':'Hash only unless separately listed as code or metadata read.'})

maps=[];saved={}
for name,item in p['plans'].items():
 f=Path(item['path']);original=Path(item['original_path'])
 assert sha(f)==item['sha256'] and sha(original)==item['original_sha256']
 z=load(f);old=load(original)
 assert 'routes' not in z and set(z)==set(old)-{'routes'}
 for key in z:assert np.array_equal(z[key],old[key]),(name,key)
 assert item['id']=={'free':'incumbent4','common_cyclic':'incumbent0','phase_independent':'incumbent2'}[name]
 rows=z['rows'];assert rows.shape==(65,2,32)
 assert np.all(rows[:64,0].sum(1)==1024) and np.all(rows[:64,1].sum(1)==1536)
 assert np.array_equal(rows[64].sum(1),[2048,1536])
 jobs=z['jobs'];offsets=z['offsets'];C=z['input_columns']
 assert jobs.shape==(2424,5) and C.shape==(65,2)
 assert np.all(C[:,0]==1536) and np.all(C[:64,1]==512) and C[64,1]==1024
 inventory={'valid_rows':0,'useful_bytes':0,'pages':0,'waves':0,'jobs':0}
 for e in range(65):
  for phase in range(2):
   R=int(rows[e,phase].sum());cursor=0;coverage=np.zeros(R,np.uint8)
   lo,hi=map(int,offsets[e,phase])
   for j in range(lo,hi):
    chip,n,c,pg,w=map(int,jobs[j]);assert n==int(rows[e,phase,chip]) and c==int(C[e,phase])
    assert pg==up(n*c,16384) and w==up(pg,4) and w<=192
    coverage[cursor:cursor+n]+=1;cursor+=n
    for k,v in [('valid_rows',n),('useful_bytes',n*c),('pages',pg),('waves',w),('jobs',1)]:inventory[k]+=v
   assert cursor==R and np.all(coverage==1)
 addresses=z['addresses'];assert addresses.shape==(2424,6)
 for chip in range(32):
  ranges=[]
  for j in np.flatnonzero(jobs[:,0]==chip):
   c,block,wl,res,pg,plane=map(int,addresses[j]);assert c==chip and block==plane==0
   assert res==int(jobs[j,4]) and pg==int(jobs[j,3]) and 0<=wl<wl+res<=192
   ranges.append((wl,wl+res))
  ranges.sort();assert all(a[1]<=b[0] for a,b in zip(ranges,ranges[1:]))
 assert inventory=={'valid_rows':167424,'useful_bytes':int(148.5*2**20),'pages':9764,'waves':2441,'jobs':2424}
 saved[name+'.rows']=rows
 maps.append({'name':name,'id':item['id'],'static_archive':str(f),'archive_sha256':sha(f),
   'original_plan':str(original),'original_plan_sha256':sha(original),
   'all_static_arrays_exact_to_original':True,'TRAIN_routes_removed':True,
   'rows_sha256':hashlib.sha256(rows.tobytes()).hexdigest(),'one_copy_complete_rows':True,
   'inventory':inventory,'allocated_LSB_blocks':32,'fixed_priority':'GF'})
assert all(np.array_equal(saved[name+'.rows'][64],saved['free.rows'][64]) for name in p['plans'])
np.savez_compressed(HERE/'three_map_rows.frozen.npz',**saved)

capture_cpp=ROOT/'scripts/research_20261103_w8_training_capture_granite4tiny.cpp'
cpp=capture_cpp.read_text();helperfile=ROUND/'w8-format-training/numeric_helper_identity.json'
identity=json.loads(helperfile.read_text())['granite4tiny']
spans=[('quantize','static float quantize(','static int64_t dot('),
 ('dot','static int64_t dot(','static float scale_output('),
 ('scale_output','static float scale_output(','static bool equal_bits(')]
helpers=[]
for name,first,last in spans:
 body=cpp[cpp.index(first):cpp.index(last,cpp.index(first))].strip()
 h=hashlib.sha256(body.encode()).hexdigest();expected=next(x for x in identity if x['function']==name)
 assert h==expected['body_sha256'] and expected['byte_exact']
 helpers.append({'name':name,'body_sha256':h,'matches_existing_helper_identity':True})
training_protocol=ROUND/'w8-format-training/protocol.frozen.json'
training=json.loads(training_protocol.read_text())
build=ROUND/'w8-format-training/build/granite4tiny.build.json'
buildinfo=json.loads(build.read_text())
assert '-fno-fast-math' in buildinfo['argv'] and '-ffp-contract=off' in buildinfo['argv']
assert sha(capture_cpp)==buildinfo['source_sha256']
assert buildinfo['binary_sha256']==p['model']['capture_binary_sha256']
selected=ROUND/'w8-format-training/selected_train_prompts.json'
trainmeta=json.loads(selected.read_text());assert len(trainmeta)==32
assert all('prompt_sha256' in row and row['split']=='train' for row in trainmeta)

contract={
 'source':str(capture_cpp),'source_sha256':sha(capture_cpp),'helpers':helpers,
 'build_receipt':str(build),'build_receipt_sha256':sha(build),
 'existing_binary_sha256':p['model']['capture_binary_sha256'],
 'checkpoint_sha256_inherited':p['model']['checkpoint_sha256'],'checkpoint_reopened':False,
 'coefficient_model':'Existing GGUF to_float per complete row, then fixed signed W8 [-127,127]. This is the declared full-FFN W8 numeric model; it is not a claim of equivalence to original GGUF Q4 arithmetic.',
 'row_scale':'F32 maxabs(row)/127; zero row uses1. std::round then clamp [-127,127]. Original per-row scales remain associated with complete logical rows.',
 'activation':'Once per entire operator input vector: F32 maxabs/127 (zero vector scale1), round and clamp to A8 [-127,127]. No row/chip partial scales; Down waits for the whole nonlinear hidden vector.',
 'dot_and_output':'CPU exact I64 sum of I8 products; F32 sw*sx, cast integer sum to F32, then F32 multiply. Compile flags disable fast-math and FP contraction.',
 'routing':'Actual ffn_moe_topk callback IDs; all40 full-FFN layers execute, attention/router retain GGUF. Fixed16 decode steps; no synthetic topk sampler.',
 'capture_scope':training['capture_scope'],
 'critical_limit':'Collector unsharded mode computes routed canonical rows and does not execute these three M32 maps. Its legacy16-chip packed shared helper is not a new M32 paired proof.',
}

notes={
 'status':'PASS_STATIC_MAP_AND_NUMERIC_CONTRACT_PREFLIGHT; WAIT_NEW_ACQUISITION_CAPTURE_BINDING',
 'utc':datetime.now(timezone.utc).isoformat(),'producer_protocol_sha256':sha(P),
 'static_maps':maps,'numeric_contract':contract,
 'prompt_metadata_rules':{
  'new_payload_read_by_this_preflight':False,'old_test_payload_read_by_this_preflight':False,
  'existing_TRAIN_metadata':{'path':str(selected),'sha256':sha(selected),'items':32,'text_files_opened':False},
  'required_new_exclusion':'Freeze a metadata-only catalog of every previously captured source ID (with dataset/split/revision scope) and rendered prompt text SHA. Exclude matching IDs or text SHAs under the acquisition protocol; do not open old prompt text or test routes to do so.',
  'required_selection':'Two primary publisher-pinned sources,16 prompts each, deterministic frozen selection/rendering, no answers in prompts and no model/output-based filtering. Preserve all16 decode steps for each32 prompts.',
  'current_status':'Producer plan protocol specifies exclusions, but the new acquisition source/ID/hash exclusion manifest and capture protocol are not yet bound here. No claim of completed cross-campaign exclusion is made.',
  'failure_rule':'Archive incomplete capture without resampling, truncating or dropping unfavorable prompts/steps. Full512 primary metrics require complete32x16x40x6 routes and finite full-FFN coverage.',
 },
 'numeric_evidence_separation':[
  {'evidence':'Old full-model paired CPU W8 proof','scope':'Old frozen16-chip images/data only; not these new three M32 layouts.'},
  {'evidence':'Root planned M32 offline materialization','scope':'Not executed or bound by this preflight. If later passing, exact existing W8 bytes/scales, physical chipbanks, explicit fixtures/TRAIN routes can prove the specified FFN mapping, not new-prompt paired full-model inference.'},
  {'evidence':'Prospective32-prompt capture','scope':'Canonical full-model W8 routing only unless an explicitly separate paired run is later authorized and completed.'},
  {'evidence':'Native NAND numeric ABI','scope':'CPU I64/F32 semantics do not validate physical PE output type/scales, accumulator reset, FIFO flow control or ECC/final status.'},
 ],
 'publication_claim_risks':[
  'M32 is a scale variant with32 full per-chip compute owners and8 shared channels. Existing controller CE/way topology does not prove those compute contexts in manufactured AiF. Within-M32 comparisons are valid model comparisons;16→32 is not a placement-only speedup.',
  'Only G4 layer0 labels and nominal250ns command/28us ordinary-read costs are evaluated here. No all-model/all-layer/end-to-end or calibrated-device speed claim follows.',
  'Against phase-independent learned cyclic TRAIN control, residual mean is0.6061917% and P998.5142751%. The2%mean OR5%P99 resource gate passes through P99; this is separate from novelty or permission to expand.',
  'At512 queries the linear empirical P99 depends on roughly the slowest6 queries;32 prompts are clusters, not512 independent samples. Predeclared prompt-cluster bootstrap/descriptive source breakdown should remain intact.',
  'Free masks versus contiguous cyclic orbits tests a larger standard placement space. Both CP controls are time-bounded and FEASIBLE; UNKNOWN peak10 is not UNSAT. A gap is not an impossibility theorem for cyclic placement or proof of solver novelty.',
  'Weighted hypergraph/query-aware declustering can express correlated selected-set work balance. This is a strong prior risk, not by itself a disproof of a complete native lowering/certificate system contribution. First CP, first joint mapping and first hypergraph claims are unsupported.',
  'Addresses/counts and opaque complete-row relocation are checked, but CPU arithmetic and metadata proofs remain distinct from the unverified native output/reset contract. No independent plane inputs, free PB reuse or changed A8 quantization may be introduced.',
  'Hold fixed all three maps, GF, primary free-versus-phase-independent comparison and input4 sensitivity before new payload. Never choose maps/baselines/priority or drop prompts from validation outcomes.',
 ],
 'future_cost_audit_gate':'Wait for explicitly completed capture and immutable acquisition/capture manifests. Then bind their hashes, validate full route shape/order/coverage, and independently reconstruct costs under the already frozen plans/profiles. No cost evaluation occurs in this preflight.',
 'files_semantically_read':[str(P),str(capture_cpp),str(helperfile),str(training_protocol),str(build),str(selected),
   str(ROOT/'scripts/research_20261103_w8_training_capture.py'),str(ROOT/'scripts/research_20261103_second_heldout_sources.py')],
 'references_hash_checked':checks,
}
dump('notes.json',notes)
dump('contract_and_maps.json',{'maps':maps,'numeric_contract':contract})
(HERE/'report.md').write_text('''三張固定 M32 圖均逐元素吻合原 sealed TRAIN winner，原 TRAIN routes 已移除；完整 rows、shared、單副本與32-block地址帳通過。尚未讀取新 acquisition/capture payload，也未執行成本或推理。\n\n數值契約為既有完整 row-W8／完整向量 dynamic-A8／CPU I64 dot／一次 F32 rescale。canonical collector 不執行這三張圖；舊16-chip full-model paired proof、新M32離線 materialization 與新資料 canonical capture 必須分開引用，皆不構成 native PE output/reset 證明。\n\nTRAIN residual mean0.606%、P998.514%通過2%mean OR5%P99資源門檻，但不是新穎性證明。M32 scale variant、單層、名目成本、32個prompt clusters與標準declustering/CP prior均需明列。等待 acquisition 排除manifest與完整capture封存後才核成本。\n''')
files=[f for f in HERE.rglob('*') if f.is_file() and f.name!='completion.preflight.json']
dump('completion.preflight.json',{'status':'SEALED_READ_ONLY_PREFLIGHT_PENDING_CAPTURE','utc':datetime.now(timezone.utc).isoformat(),
 'files':{str(f.relative_to(HERE)):sha(f) for f in sorted(files)}})
print(json.dumps({'status':notes['status'],'notes_sha256':sha(HERE/'notes.json'),'completion_sha256':sha(HERE/'completion.preflight.json')},indent=2))
