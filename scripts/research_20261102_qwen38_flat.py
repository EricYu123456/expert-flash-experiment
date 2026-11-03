#!/usr/bin/env python3
"""Create file-order convenience view; do not read any new source task."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/research-2026-11-02/round2/qwen38-trace-audit'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    source = OUT / 'train_decode.npz'
    assert sha(source) == '953e8db87d4bc4a5b5b9a77f5aebae1578b1e522df5bf3ac6a32a591f9e8b145'
    original = np.load(source)
    nreq, ntok, nlayers, k = original['routes'].shape
    assert (nreq, ntok, nlayers, k) == (16, 799, 48, 10)
    order = np.argsort(original['source_file_rows'].reshape(-1))
    assert len(np.unique(order)) == nreq * ntok

    def flat(key):
        value = original[key]
        assert value.shape[:2] == (nreq, ntok)
        return value.reshape((nreq * ntok,) + value.shape[2:])[order]

    req_ids = np.broadcast_to(original['req_ids'][:, None], (nreq, ntok)).reshape(-1)[order]
    token_pos = np.broadcast_to(original['steps'][None, :], (nreq, ntok)).reshape(-1)[order]
    prompt_ids = np.broadcast_to(original['prompt_ids'][:, None], (nreq, ntok)).reshape(-1)[order]
    assert np.all(np.diff(flat('source_file_rows')) > 0)
    # Timings belong to the original eight-request source batch, not latency of
    # an independently replayed request. Verify these groups before exporting.
    batches = flat('batch_ids')
    batch_rows = []
    for bid in np.unique(batches):
        ix = np.flatnonzero(batches == bid)
        assert len(ix) == 8
        assert len(np.unique(req_ids[ix])) == 8
        assert len(np.unique(token_pos[ix])) == 1
        assert len(np.unique(flat('step_wall_ns')[ix])) == 1
        batch_rows.append(dict(batch_id=int(bid), count=len(ix),
                               requests=req_ids[ix].astype(int).tolist(),
                               token_pos=int(token_pos[ix[0]])))
    target = OUT / 'train_decode_file_order.npz'
    np.savez_compressed(target, routes=flat('routes'), gates=flat('gates'),
        req_ids=req_ids, session_ids=prompt_ids, prompt_ids=prompt_ids,
        token_pos=token_pos, tok_idx=token_pos, layers=original['layers'],
        token_ids=flat('token_ids'), prev_token_ids=flat('prev_token_ids'),
        source_file_rows=flat('source_file_rows'), step_wall_ns=flat('step_wall_ns'),
        batch_ids=batches, phase=flat('phase'), valid=flat('valid'))
    views = {}
    for path in [source, target, OUT / 'train_prefill.npz']:
        data = np.load(path)
        views[path.name] = dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size,
            fields={key: dict(shape=list(data[key].shape), dtype=str(data[key].dtype))
                    for key in data.files})
    meta = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sha(__file__), views=views,
        batch_validation=dict(decode_batch_count=len(batch_rows), rows_per_batch=8,
                              all_batch_rows_have_same_token_pos=True,
                              request_sets=sorted({tuple(row['requests']) for row in batch_rows})),
        grouping=dict(source_task='diagnostic/smoke-batched-8way',
            task_name='b01-cpp-ring-threadpool', all_requests_train_exploratory=True,
            training_fixed_before_route_parse=True, external_task_payloads_opened=False,
            session_ids='Synthetic unique source-task/request identifiers, not independent tasks.',
            token_pos='Executed decode-forward index within request; contiguous 0..798.',
            file_order='Original decode token-block order after removing prefill blocks.',
            per_request_view='train_decode.npz: routes[request,token,layer,topk].',
            flat_view='train_decode_file_order.npz: routes[token,layer,topk], original file order.',
            fixed_stride4='Use routes[:,::4] in request view or token_pos%4==0 in flat view. Flat[::4] aliases request IDs in 8-way batches.',
            batch_fields='batch_ids and step_wall_ns describe source shared eight-request forward, not single-request latency.',
            confidence='Complete recorded routes with independent arithmetic spot-check; unavailable source recorder prevents full engine reproduction.'),
        download_budget=dict(initial_limit_bytes=10_000_000_000,
            main_stream_bytes=150_809_896, router_weights_bytes=309_433_101,
            state_spot_check_bytes=4*283_220,
            substantive_payload_bytes=150_809_896+309_433_101+4*283_220,
            full_trace_downloaded=False, full_checkpoint_downloaded=False),
        seals=dict(extraction_completion_sha256=sha(OUT/'completion.json'),
                   extraction_audit_sha256=sha(OUT/'audit.json'),
                   split_plan_sha256=sha(OUT/'split_plan.before_payload.json')))
    (OUT/'schema.json').write_text(json.dumps(meta, indent=2)+'\n')
    print(json.dumps(dict(flat_npz=str(target), sha256=sha(target),
                          shape=list(np.load(target)['routes'].shape),
                          schema_sha256=sha(OUT/'schema.json')), indent=2))


if __name__ == '__main__':
    main()
