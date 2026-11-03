#!/usr/bin/env python3
"""Independent read-only v3 route extraction from one fixed public train task.

The unavailable recorder branch is not imported or executed.  Format offsets,
joins and routes are checked independently.  Only selected hidden-state ranges
are fetched; the full3.926GB trace is intentionally not downloaded/hashed.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import requests
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
LOCAL=ROOT/'results/research-2026-11-02/round2/qwen38-trace-audit'
NFS=Path('/home/nfs_client/ericyu/expert_flash_experiment/research-20261102/qwen38-trace-audit')
REV='fee2af4ea0da212d853d82943dad3897653066a2'
URL=f'https://huggingface.co/datasets/aswinkumar99/qwen3.8-flash-next-expert-traces/resolve/{REV}/diagnostic/smoke-batched-8way/trace.bin'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def summary(x):
    x=np.asarray(x)
    return dict(min=float(x.min()),max=float(x.max()),mean=float(x.mean()))


def dtypes():
    # Extended step fields reuse compact's trailing word: token_id at28,
    # history at32, PLE at44; four trailing pad bytes make176.  The prose
    # spec is ambiguous about this reuse.  Offset32 would yield invalid
    # token_ids=0xffffffff on the first BOS row, and fails token-history joins.
    step=np.dtype(dict(names=['req_id','phase','rebuild_event','tok_idx','step_wall_ns',
        'slots_total','slots_in_use','token_id','prev_token_ids','ple_hash_ids'],
        formats=['<u4','u1','u1','<u4','<u8','<u4','<u4','<u4',('<u4',3),('<u8',16)],
        offsets=[0,4,5,8,12,20,24,28,32,44],itemsize=176))
    layer=np.dtype(dict(names=['req_id','phase','layer','hit_mask','tok_idx','experts','gates','slot_ids',
        'logit_top32_idx','logit_top32_val','router_entropy','top10_mass','norm_block_input','norm_routed_out','norm_shared_out'],
        formats=['<u4','u1','u1','<u2','<u4',('<u2',10),('<f2',10),('<u2',10),
            ('<u2',32),('<f2',32),'<f2','<f2','<f2','<f2','<f2'],
        offsets=[0,4,5,6,8,12,32,52,72,136,200,202,204,206,208],itemsize=210))
    token=np.dtype([('step',step),('layers',layer,(48,))])
    assert token.itemsize==10256
    state=np.dtype(dict(names=['req_id','phase','layer','tok_idx','block_input','hc_stream_gates','hc_lowrank'],
        formats=['<u4','u1','u1','<u4',('<u2',2560),('<u2',4),('<u2',320)],
        offsets=[0,4,5,8,12,5132,5140],itemsize=5780))
    return step,layer,token,state


def bf16_to_f32(a):return (np.asarray(a,dtype=np.uint32)<<16).view(np.float32)


def bf16_round(a):
    u=np.asarray(a,dtype=np.float32).view(np.uint32)
    return ((u+np.uint32(0x7fff)+((u>>16)&1))&np.uint32(0xffff0000)).view(np.float32)


def state_checks(tokens,header,main_end,state_dtype):
    s=tokens['step'];has=(s['phase']==1)|((s['phase']!=1)&(s['tok_idx']%4==0))
    state_tokens=np.flatnonzero(has)
    assert len(state_tokens)*49==650132
    decode=np.flatnonzero(s['phase']==1)
    # A fixed small arithmetic audit, no data-dependent selection or fitting.
    chosen=decode[np.unique(np.linspace(0,len(decode)-1,4,dtype=int))]
    taps=[0,12,24,36,47]
    weights=np.load(NFS/'router_weights.npz')
    records=[];receipts=[]
    for ti in chosen:
        ordinal=int(np.searchsorted(state_tokens,ti));assert state_tokens[ordinal]==ti
        start=main_end+ordinal*49*5780;end=start+49*5780-1
        response=requests.get(URL,headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity'},timeout=40)
        response.raise_for_status();assert response.status_code==206 and len(response.content)==49*5780
        assert response.headers['Content-Range'].startswith(f'bytes {start}-{end}/')
        path=NFS/'smoke-batched-8way'/f'states.token-file-row-{ti}.range.bin'
        path.write_bytes(response.content)
        receipts.append(dict(url=URL,range_start=start,range_end=end,path=str(path),sha256=sha(path),bytes=len(response.content)))
        states=np.frombuffer(response.content,dtype=state_dtype)
        assert np.array_equal(states['layer'],np.r_[np.arange(48),255])
        assert np.all(states['req_id']==s[ti]['req_id']) and np.all(states['phase']==1)
        assert np.all(states['tok_idx']==s[ti]['tok_idx'])
        assert np.all(states[-1]['hc_stream_gates']==0) and np.all(states[-1]['hc_lowrank']==0)
        for layer in taps:
            x=bf16_to_f32(states[layer]['block_input'])
            w=weights[f'model.language_model.layers.{layer}.mlp.gate.weight']
            assert w.shape==(512,2560) and np.all(np.isfinite(w))
            assert np.all((w.view(np.uint32)&np.uint32(65535))==0)
            y=w@x;rounded=bf16_round(y)
            original=tokens[ti]['layers'][layer]
            ids=original['experts'];top32=original['logit_top32_idx']
            rank=np.lexsort((np.arange(512),-rounded))
            centered=rounded-rounded.max()
            records.append(dict(file_token_row=int(ti),req_id=int(s[ti]['req_id']),tok_idx=int(s[ti]['tok_idx']),
                layer=layer,router_matrix_shape=list(w.shape),router_matrix_f32_hash=hashlib.sha256(w.tobytes()).hexdigest(),
                hidden_all_finite=bool(np.all(np.isfinite(x))),
                top10_order_equal=bool(np.array_equal(rank[:10],ids)),
                top10_set_agreement=len(set(rank[:10].tolist())&set(ids.tolist()))/10,
                top32_order_equal=bool(np.array_equal(rank[:32],top32)),
                max_abs_centered_logit_error=float(np.max(np.abs(centered[top32]-original['logit_top32_val'].astype(np.float32)))),
                fp32_raw_max_abs_centered_logit_error=float(np.max(np.abs((y-y.max())[top32]-original['logit_top32_val'].astype(np.float32)))),
                method='FP32 CPU dot of savedBF16values then BF16 round-to-nearest-even; deterministic low-expert-ID tie break.'))
    return dict(records=records,range_receipts=receipts,
        top10_order_equal_fraction=float(np.mean([r['top10_order_equal'] for r in records])),
        mean_top10_set_agreement=float(np.mean([r['top10_set_agreement'] for r in records])),
        max_centered_logit_error=max(r['max_abs_centered_logit_error'] for r in records),
        caveat='Internal arithmetic consistency with published routerW/state only. Does not reproduce unavailable original recorder or allNVFP4modelweights.')


def main():
    plan=LOCAL/'split_plan.before_payload.json';assert plan.exists()
    receipts=json.loads((LOCAL/'payload_receipts.json').read_text())
    path=NFS/'smoke-batched-8way/main_stream.range.bin'
    assert sha(path)==next(r['sha256'] for r in receipts if r['range_start']==0)
    with path.open('rb') as f:raw=f.read(32);fields=struct.unpack('<8sHHIHHHHII',raw);run=json.loads(f.read(fields[-1]))
    magic,version,endian,hbytes,sbytes,lbytes,layers,k,flags,jbytes=fields
    assert (magic,version,endian,sbytes,lbytes,layers,k)==(b'FTEXTRC\0',3,0x0102,176,210,48,10)
    assert flags==0x2fbf and run['model_geometry']==dict(hidden_size=2560,moe_intermediate_size=640,num_experts=512,num_moe_layers=48,top_k=10)
    _,_,token_dtype,state_dtype=dtypes()
    body=path.stat().st_size-hbytes;assert body==150804224 and body%token_dtype.itemsize==0
    tokens=np.memmap(path,mode='r',offset=hbytes,dtype=token_dtype,shape=(body//token_dtype.itemsize,))
    s=tokens['step'];r=tokens['layers'];n=len(tokens)
    assert n==14704 and np.all(np.isin(s['phase'],[0,1,2]))
    assert np.array_equal(r['layer'],np.broadcast_to(np.arange(48),(n,48)))
    for key in ['req_id','phase','tok_idx']:assert np.array_equal(r[key],np.broadcast_to(s[key][:,None],(n,48)))
    assert np.all(r['experts']<512) and np.all(r['logit_top32_idx']<512)
    assert np.all(np.diff(np.sort(r['experts'],axis=-1),axis=-1)>0)
    assert np.all(np.diff(np.sort(r['logit_top32_idx'],axis=-1),axis=-1)>0)
    assert np.array_equal(r['experts'],r['logit_top32_idx'][:,:,:10])
    assert np.all(np.isfinite(r['gates'])) and np.all(r['gates']>=0)
    assert np.all(np.diff(r['gates'].astype(np.float32),axis=-1)<=0)
    assert np.all(np.isfinite(r['logit_top32_val'])) and np.all(r['logit_top32_val'][:,:,0]==0)
    assert np.all(np.diff(r['logit_top32_val'].astype(np.float32),axis=-1)<=0)
    mass=r['gates'].astype(np.float32).sum(axis=-1)
    assert np.max(np.abs(mass-1))<0.001
    assert np.all(s['token_id']<248320)
    assert np.all(s['slots_total']==24576)
    reqs=np.unique(s['req_id'][s['phase']==1]);assert len(reqs)==16
    groups=[];request_summary=[];history_compared=history_equal=0
    for req in reqs:
        ix=np.flatnonzero((s['phase']==1)&(s['req_id']==req))
        assert np.array_equal(s['tok_idx'][ix],np.arange(len(ix)))
        assert len(np.unique(s['tok_idx'][ix]))==len(ix)
        groups.append(ix)
        previous=s['prev_token_ids'][ix[1:],0];current=s['token_id'][ix[:-1]]
        history_compared+=len(previous);history_equal+=int((previous==current).sum())
        assert np.array_equal(previous,current)
        request_summary.append(dict(req_id=int(req),decode_rows=len(ix),prefill_rows=int(((s['req_id']==req)&(s['phase']!=1)).sum()),
            first_file_row=int(ix[0]),last_file_row=int(ix[-1]),tok_idx_start=0,tok_idx_end=len(ix)-1))
    assert len({len(g) for g in groups})==1
    index=np.stack(groups);batch_times,batch_inv=np.unique(s['step_wall_ns'],return_inverse=True)
    target=LOCAL/'train_decode.npz'
    np.savez_compressed(target,routes=r['experts'][index],gates=r['gates'][index],
        prompt_ids=np.asarray([f'qwen38_smoke_batched_b01_req_{int(q):03d}' for q in reqs]),
        req_ids=reqs,layers=np.arange(48,dtype=np.int64),steps=np.arange(len(groups[0]),dtype=np.int64),
        token_ids=s['token_id'][index],prev_token_ids=s['prev_token_ids'][index],
        source_file_rows=index,step_wall_ns=s['step_wall_ns'][index],batch_ids=batch_inv[index],
        phase=s['phase'][index],valid=np.ones(index.shape,dtype=bool))
    pix=np.flatnonzero(s['phase']!=1);prefill=LOCAL/'train_prefill.npz'
    np.savez_compressed(prefill,routes=r['experts'][pix],gates=r['gates'][pix],
        req_ids=s['req_id'][pix],layers=np.arange(48,dtype=np.int64),tok_idx=s['tok_idx'][pix],
        token_ids=s['token_id'][pix],prev_token_ids=s['prev_token_ids'][pix],phase=s['phase'][pix],
        source_file_rows=pix,step_wall_ns=s['step_wall_ns'][pix],batch_ids=batch_inv[pix])
    with threadpool_limits(limits=1):arithmetic=state_checks(tokens,run,path.stat().st_size,state_dtype)
    vals=r['logit_top32_val'].astype(np.float32)
    tied=(np.diff(vals,axis=-1)==0)
    order_violation=tied & (np.diff(r['logit_top32_idx'].astype(np.int32),axis=-1)<0)
    out=dict(status='complete independent train-task extraction and arithmetic audit; no physical plan fit',
        created_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),dataset_revision=REV,
        training_split_plan_sha256=sha(plan),source_task='diagnostic/smoke-batched-8way',
        main_range_path=str(path),main_range_sha256=sha(path),main_range_bytes=path.stat().st_size,
        expected_full_trace_sha256='e97d6063aee6e4c7e620980193e850e58a6a100998dc0e024a0fcc0be6aabf95',
        full_trace_hash_locally_verified=False,router_weights_npz_sha256=sha(NFS/'router_weights.npz'),
        header=run,records=n,phase_counts={str(int(q)):int((s['phase']==q).sum()) for q in np.unique(s['phase'])},
        requests=request_summary,decode_routes_shape=list(r['experts'][index].shape),
        decode_npz=str(target),decode_npz_sha256=sha(target),prefill_npz=str(prefill),prefill_npz_sha256=sha(prefill),
        checks=dict(all48layers_per_token=True,top10_unique_and_in_range=True,top32_unique_and_in_range=True,
            top32_first10_exactly_execution_route=True,per_request_decode_tok_idx_contiguous=True,
            gate_sum=summary(mass),token_history_pairs_compared=history_compared,token_history_equal=history_equal,
            top32_float16_tie_order_violations=int(order_violation.sum()),
            note='Storedfloat16logit ties can differ from originalBF16/fullprecision ordering; orderedtop10 route equality is checked independently.'),
        state_router_arithmetic=arithmetic,
        format_note='Featurestep token_id offset28,previousIDs32,PLE44,tailpad4. Checked actualBOS/tokenhistory; naive appendat32 contradictsprose spec and fails.',
        caveats=['All16requests share one C++task and near-identicalvariants; all aretrain/exploratory. Other task binarypayloads remainunopened. No generalization claim.',
            'Sourceis8way batched NVFP4routedexperts withBF16routers. These are realrecordedroutes; usingI8NANDservicegeometry remainsa conditional costmodel, not execution ofthe originalcheckpoint onAiF.',
            'Sharedexpert isnot recorded; its640intermediate shape isverified onlyfrom publishedcheckpointconfig. No fullmodeltiming.',
            'Sourcecache is allresident, misses are compulsorycoldstart; sourcehit/slotvalues are not deploymentcache/performance evidence.',
            'Prefillrecordedpositions mayomit cachedprefixes. Phase0/2 rows are separate and are not decodedtokens or a completeprompt transcript.',
            'Headerenginecommit and recorderbranch/reader currentlyunresolvable inpublicupstream; enginegitdirty isnull. This auditdoesnot establish reproducibleoriginalengine.',
            'Publishedcheckpointcurrentrevision geometrychecked, butoriginalmodelrevision isnot inheader. OnlypublicsavedrouterW arithmetic consistency independentlytested.',
            'Decodeforwards caninclude finalinflight discardedwork; tok_idx indexesexecuted forwards, not clientvisiblecompletioncount.'])
    (LOCAL/'audit.json').write_text(json.dumps(out,indent=2)+'\n')
    (LOCAL/'completion.json').write_text(json.dumps(dict(status='complete',audit_sha256=sha(LOCAL/'audit.json'),
        decode_npz_sha256=sha(target),prefill_npz_sha256=sha(prefill),source_sha256=sha(__file__)),indent=2)+'\n')
    print(json.dumps(dict(audit_sha256=sha(LOCAL/'audit.json'),routes_shape=out['decode_routes_shape'],
        arithmetic={k:v for k,v in arithmetic.items() if k not in ('records','range_receipts')}),indent=2))


if __name__=='__main__':main()
