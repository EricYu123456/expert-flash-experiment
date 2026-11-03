#!/usr/bin/env python3
"""Extract one specifically authorized post-freeze external QA trace.

Reuse the independently checked public v3 format; never import the unavailable
original recorder.  Warm-up requests are excluded from external evaluation.
"""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import requests
from threadpoolctl import threadpool_limits

from research_20261102_qwen38_trace_audit import dtypes,bf16_to_f32,bf16_round

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/research-2026-11-02/round2'
OUT=BASE/'qwen38-trace-audit/external-doc-grounded-stride1-short'
NFS=Path('/home/nfs_client/ericyu/expert_flash_experiment/research-20261102/qwen38-trace-audit')
REV='fee2af4ea0da212d853d82943dad3897653066a2'
TASK='diagnostic/03-doc-grounded-stride1-short'
URL=f'https://huggingface.co/datasets/aswinkumar99/qwen3.8-flash-next-expert-traces/resolve/{REV}/{TASK}/trace.bin'
FREEZE=BASE/'qwen38-placement/native-full48/plans.frozen.json'
FREEZE_SHA='9db047b5335c4d95e45755dc62ee1b545ba97b1119ec39294d09a86299b5e505'
WEIGHTS=NFS/'router_weights.npz'
WEIGHTS_SHA='c9914cf0d6021a7ad445a617ffd7fb2da3a28b57e3df48d69c929efcf4567c70'
PARSER_SHA='b0efa430cbb6445c1eb774fa4f8446e203f6dabe1cb8a750a0e7224d036abfa0'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())


def frozen():
    assert sha(FREEZE)==FREEZE_SHA
    doc=read(FREEZE);assert len(doc['records'])==240
    for r in doc['records']:assert sha(ROOT/r['plan'])==r['plan_sha256']


def arithmetic(tokens,indices,main_end,state_dtype):
    header=read(OUT/'trace_header.json');features=header['features']
    probes=features['state_probe_layers'];assert probes==list(range(48))+[255]
    assert features['state_prefill_stride']==features['state_decode_stride']==1
    weights=np.load(WEIGHTS);assert sha(WEIGHTS)==WEIGHTS_SHA
    chosen=indices[np.unique(np.linspace(0,len(indices)-1,4,dtype=int))]
    records=[];receipts=[]
    for ti in chosen:
        start=main_end+int(ti)*49*5780;end=start+49*5780-1
        path=NFS/'external-doc-grounded-stride1-short'/f'states.token-file-row-{int(ti)}.range.bin'
        if path.exists():raw=path.read_bytes()
        else:
            r=requests.get(URL,headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity'},timeout=40)
            r.raise_for_status();assert r.status_code==206 and r.headers['Content-Range']==f'bytes {start}-{end}/12238037388'
            raw=r.content;assert len(raw)==49*5780;path.write_bytes(raw)
        assert len(raw)==49*5780
        receipts.append(dict(url=URL,range_start=start,range_end=end,path=str(path),bytes=len(raw),sha256=sha(path)))
        states=np.frombuffer(raw,dtype=state_dtype);step=tokens[ti]['step']
        assert np.array_equal(states['layer'],probes)
        for key in ['req_id','phase','tok_idx']:assert np.all(states[key]==step[key])
        assert np.all(states[-1]['hc_stream_gates']==0) and np.all(states[-1]['hc_lowrank']==0)
        for layer in [0,12,24,36,47]:
            x=bf16_to_f32(states[layer]['block_input'])
            w=weights[f'model.language_model.layers.{layer}.mlp.gate.weight']
            assert w.shape==(512,2560) and np.all(np.isfinite(x)) and np.all(np.isfinite(w))
            assert np.all((w.view(np.uint32)&65535)==0)
            y=w@x;rounded=bf16_round(y);rank=np.lexsort((np.arange(512),-rounded))
            saved=tokens[ti]['layers'][layer];top32=saved['logit_top32_idx'];ids=saved['experts']
            records.append(dict(source_file_row=int(ti),req_id=int(step['req_id']),token_pos=int(step['tok_idx']),layer=layer,
                ordered_top10_exact=bool(np.array_equal(rank[:10],ids)),ordered_top32_exact=bool(np.array_equal(rank[:32],top32)),
                top10_set_agreement=len(set(rank[:10].tolist())&set(ids.tolist()))/10,
                centered_top32_max_abs_error=float(np.max(np.abs((rounded-rounded.max())[top32]-saved['logit_top32_val'].astype(np.float32)))),
                fp32_before_bf16_round_max_abs_error=float(np.max(np.abs((y-y.max())[top32]-saved['logit_top32_val'].astype(np.float32))))))
    return dict(method='Saved BF16 state and widened BF16 routerW; FP32 CPU GEMV then BF16 round-to-nearest-even, low-ID tie break.',
        records=records,range_receipts=receipts,ordered_top10_exact_fraction=float(np.mean([r['ordered_top10_exact'] for r in records])),
        ordered_top32_exact_fraction=float(np.mean([r['ordered_top32_exact'] for r in records])),
        max_centered_top32_error=max(r['centered_top32_max_abs_error'] for r in records),
        scope='Internal arithmetic consistency only; no independent original engine/model generation.')


def main():
    frozen();before=read(OUT/'before_payload.json');assert before['freeze_sha256']==FREEZE_SHA
    parser=Path(__file__).with_name('research_20261102_qwen38_trace_audit.py');assert sha(parser)==PARSER_SHA
    assert sha(BASE/'qwen38-trace-audit/split_plan.before_payload.json')=='6e5520ac3f51264cac9cb99ecd5ef93f302e4e13b48408833feb7a2d51aa8966'
    run=read(OUT/'source/run.json');warm=read(OUT/'source/warmup.json');n_warm=warm['warmup_requests'];assert n_warm==32
    receipts=read(OUT/'payload_receipts.json');path=Path(receipts[0]['path']);assert sha(path)==receipts[0]['sha256']
    with path.open('rb') as f:
        fields=struct.unpack('<8sHHIHHHHII',f.read(32));header=json.loads(f.read(fields[-1]))
    magic,version,endian,hbytes,sbytes,lbytes,layers,k,flags,jbytes=fields
    assert (magic,version,endian,sbytes,lbytes,layers,k,flags)==(b'FTEXTRC\0',3,258,176,210,48,10,0x2fbf)
    assert header['model_geometry']==dict(hidden_size=2560,moe_intermediate_size=640,num_experts=512,num_moe_layers=48,top_k=10)
    _,_,td,sd=dtypes();body=path.stat().st_size-hbytes;assert body==425952192 and body%td.itemsize==0
    tokens=np.memmap(path,mode='r',offset=hbytes,dtype=td,shape=(body//td.itemsize,));s=tokens['step'];r=tokens['layers'];n=len(tokens)
    assert n==41532 and np.all(np.isin(s['phase'],[1,2]))
    assert np.array_equal(r['layer'],np.broadcast_to(np.arange(48),(n,48)))
    for key in ['req_id','phase','tok_idx']:assert np.array_equal(r[key],np.broadcast_to(s[key][:,None],(n,48)))
    assert np.all(r['experts']<512) and np.all(r['logit_top32_idx']<512)
    assert np.all(np.diff(np.sort(r['experts'],axis=-1),axis=-1)>0)
    assert np.all(np.diff(np.sort(r['logit_top32_idx'],axis=-1),axis=-1)>0)
    assert np.array_equal(r['experts'],r['logit_top32_idx'][:,:,:10])
    assert np.all(np.isfinite(r['gates'])) and np.all(r['gates']>=0) and np.all(np.diff(r['gates'].astype(np.float32),axis=-1)<=0)
    sums=r['gates'].astype(np.float32).sum(axis=-1);assert np.max(np.abs(sums-1))<.001
    assert np.all(np.isfinite(r['logit_top32_val'])) and np.all(r['logit_top32_val'][:,:,0]==0)
    assert np.all(np.diff(r['logit_top32_val'].astype(np.float32),axis=-1)<=0)
    assert np.all(s['token_id']<248320) and np.all(s['slots_total']==24576)
    reqs=np.unique(s['req_id']);assert np.array_equal(reqs,np.arange(len(reqs)))
    warmup=s['req_id']<n_warm;work=~warmup
    assert np.flatnonzero(work)[0]==run['recorded']['first_workload_token_index']==14182
    assert np.all(warmup[:14182]) and np.all(work[14182:])
    assert int((warmup&(s['phase']==1)).sum())==warm['warmup_decode_tokens']==11866
    request_rows=[];history_total=0
    for req in reqs:
        ix=np.flatnonzero((s['phase']==1)&(s['req_id']==req))
        if len(ix):
            assert np.array_equal(s['tok_idx'][ix],np.arange(len(ix)))
            assert np.array_equal(s['prev_token_ids'][ix[1:],0],s['token_id'][ix[:-1]])
            history_total+=len(ix)-1
        request_rows.append(dict(req_id=int(req),warmup=bool(req<n_warm),decode_rows=len(ix),
            prefill_rows=int(((s['req_id']==req)&(s['phase']!=1)).sum()),
            first_file_row=int(np.flatnonzero(s['req_id']==req)[0])))
    batch_times,batch_ids=np.unique(s['step_wall_ns'],return_inverse=True)
    session=run['session_id'];views={}
    for name,mask in [('external_decode',work&(s['phase']==1)),('external_prefill',work&(s['phase']!=1)),
                      ('warmup_decode',warmup&(s['phase']==1)),('warmup_prefill',warmup&(s['phase']!=1))]:
        ix=np.flatnonzero(mask);target=OUT/(name+'.npz')
        np.savez_compressed(target,routes=r['experts'][ix],gates=r['gates'][ix],layers=np.arange(48,dtype=np.int64),
            req_ids=s['req_id'][ix],session_ids=np.full(len(ix),session),token_pos=s['tok_idx'][ix],tok_idx=s['tok_idx'][ix],
            token_ids=s['token_id'][ix],prev_token_ids=s['prev_token_ids'][ix],source_file_rows=ix,
            step_wall_ns=s['step_wall_ns'][ix],batch_ids=batch_ids[ix],phase=s['phase'][ix],warmup=warmup[ix],
            prompt_ids=np.asarray([f'qwen38_docgrounded_req_{int(req):03d}' for req in s['req_id'][ix]]))
        data=np.load(target);views[name]=dict(path=str(target),sha256=sha(target),bytes=target.stat().st_size,
            fields={key:dict(shape=list(data[key].shape),dtype=str(data[key].dtype)) for key in data.files})
    selected=np.flatnonzero(work&(s['phase']==1))
    assert len(selected)==5519
    with threadpool_limits(limits=1):math=arithmetic(tokens,selected,path.stat().st_size,sd)
    sections=read(OUT/'sections.json');state=next(x for x in sections if x['magic']=='FTEXST2')
    assert state['count']==n*49 and state['record_bytes']==sd.itemsize
    metadata=sum(x['bytes'] for x in read(OUT/'doc_receipts.json'))
    downloaded=path.stat().st_size+sum(x['bytes'] for x in math['range_receipts'])+65536+32*len(sections)+metadata
    assert downloaded<=2_000_000_000
    frozen()
    out=dict(status='complete one authorized external task extraction; frozen plans unchanged; no fit performed',
        created_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),parser_source_sha256=PARSER_SHA,
        dataset_revision=REV,source_task=TASK,before_payload_sha256=sha(OUT/'before_payload.json'),
        freeze_path=str(FREEZE),freeze_sha256_before_and_after=FREEZE_SHA,all240_plan_hashes_unchanged=True,
        main_stream_path=str(path),main_stream_sha256=sha(path),main_stream_bytes=path.stat().st_size,
        expected_full_trace_lfs_sha256=receipts[0]['expected_full_trace_lfs_sha256'],full_trace_hash_locally_verified=False,
        reused_router_weights_sha256=WEIGHTS_SHA,header=header,
        counts=dict(all_token_rows=n,all_decode_rows=int((s['phase']==1).sum()),all_prefill_rows=int((s['phase']!=1).sum()),
            warmup_requests=32,warmup_decode_rows=11866,external_decode_rows=5519,
            external_prefill_rows=int((work&(s['phase']!=1)).sum()),external_requests=int((reqs>=32).sum()),
            first_external_file_row=14182,source_client_session_count=1),
        request_summary=request_rows,checks=dict(complete48_layers=True,top10_unique_in_range=True,top32_unique_in_range=True,
            top32_first10_exact=True,all_decode_positions_contiguous=True,token_history_pairs_checked=history_total,
            all_token_history_pairs_exact=True,gate_sum_min=float(sums.min()),gate_sum_max=float(sums.max())),
        views=views,state_router_arithmetic=math,new_download_bytes_including_docs_header_footers=downloaded,
        new_download_budget_bytes=2_000_000_000,
        provenance=dict(dataset='https://huggingface.co/datasets/aswinkumar99/qwen3.8-flash-next-expert-traces/tree/'+REV,
            task='Document-grounded question answering and agentic follow-ups, different from exploratory C++ completion task.',
            segmentation='First32request IDs are explicit warm-up per warmup.json. Only req_id>=32,phase1 exported to external_decode.',
            request_to_prompt='req_id is a forward request within one tool-using client session, not an independent user/task. No unverified mapping to the16 scripted prompt indexes is assigned.',
            source_status=run['status'],
            source_counts='Final footer/recorded counts used, not stale live runner decode/prefill counters.',
            metadata_docs_sha256={x['path']:x['sha256'] for x in read(OUT/'doc_receipts.json')}),
        limitations=['Only this external task was opened; no other external binary task, model download, physical fitting or plan update.',
            'One external client session with multiple request/turn forwards is not thousands of independent QA tasks.',
            'Sourceallresidentcache/coldstart fields do not model deployment weight-cache behavior.',
            'Prefillphase2 is recorded forwarded uncached/D2D-eligible positions; omitted prefix-cache hits prevent complete original prompt reconstruction.',
            'Third-party public trace with arithmetic consistency, not independent reproduction of original unavailable recorder commit/branch.',
            'Real source routed weights NVFP4 and gates BF16; nominal I8 NAND placement timing is a conditional service model, not original checkpoint performance.',
            'Sharedexpert routing is not recorded; geometry640 is from separately pinned checkpoint config. Attention/SSM/HC/PLE/fullmodel performance is outside this dataset-derived FFN model.',
            'AiF numeric output datatype/scales and host partial reconstruction ABI remain unverified; this extraction adds no hardware capability.',
            'Decode indexes executed forwards and may include discarded inflight work, not exactly visible streamed completion tokens.'])
    (OUT/'audit.json').write_text(json.dumps(out,indent=2)+'\n')
    completion=dict(status='complete',audit_sha256=sha(OUT/'audit.json'),source_sha256=sha(__file__),
                    freeze_sha256=FREEZE_SHA,external_decode_npz_sha256=views['external_decode']['sha256'])
    (OUT/'completion.json').write_text(json.dumps(completion,indent=2)+'\n')
    print(json.dumps(dict(counts=out['counts'],math={k:v for k,v in math.items() if k not in ['records','range_receipts']},
                          external_decode=views['external_decode']['path'],**completion),indent=2))


if __name__=='__main__':main()
