#!/usr/bin/env python3
"""No-refit replay of all48 frozen physical views on one external task."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from research_20261102_qwen38_compiled_campaign import library, items_for, source_hashes
from research_20261102_qwen38_fast_replay import encode, evaluate
from research_20261102_qwen38_fit import sha
from research_20261102_qwen38_replay import BASE, inventories, build

ROOT = Path(__file__).resolve().parents[1]
PLANS = BASE/'qwen38-placement/native-full48/plans.frozen.json'
PLAN_SHA = '9db047b5335c4d95e45755dc62ee1b545ba97b1119ec39294d09a86299b5e505'
EXTERNAL = BASE/'qwen38-trace-audit/external-doc-grounded-stride1-short/external_decode.npz'
EXTERNAL_SHA = '2111de845e4e8938e2158d0fc778f7910c48879184aeb1900759cd764d5f4736'


def worker(task):
    record, output = task
    inv = inventories(); lib = library(); target = Path(output)
    planpath = Path(record['plan']); assert sha(planpath) == record['plan_sha256']
    plan = json.loads(planpath.read_text()); desc, partial = encode(plan, inv)
    with np.load(EXTERNAL) as f:
        selected = f['routes'][:, record['layer']].copy()
    values = evaluate(lib, desc, partial, selected, 1000, 0, False)
    service = np.zeros((513, 16), dtype=np.int64)
    for e in range(513):
        for phase in (0, 1):
            for chip, rows, columns, waves in desc[e, phase]:
                if rows:
                    service[e, chip] += int(waves)*28000+250+(int(columns)+1)//2+2*int(rows)
    bounds = (service[selected].sum(axis=1)+service[512]).max(axis=1)+2000
    assert np.all(values >= bounds)
    diagnostic = build(items_for(plan, inv, selected[0]), 1000, 0).run(False, trace=True)
    assert diagnostic['makespan_ns'] == int(values[0])
    label = record['label']; np.savez_compressed(target/(label+'.npz'), cost=values, native_chip_bound=bounds)
    trace = dict(layer=record['layer'], case=record['case'], selected=selected[0].tolist(),
                 items=items_for(plan, inv, selected[0]), fixed_shared=plan['fixed_shared'], **diagnostic)
    (target/(label+'.trace.json')).write_text(json.dumps(trace, indent=2)+'\n')
    return dict(**record, samples=len(values), mean_ns=float(values.mean()), p99_ns=float(np.percentile(values, 99)),
                mean_bound_ns=float(bounds.mean()), artifact_sha256=dict(npz=sha(target/(label+'.npz')),
                  **{'trace.json':sha(target/(label+'.trace.json')), 'plan.json':sha(planpath)}))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--output', type=Path, default=BASE/'qwen38-placement/external-doc-grounded')
    args = ap.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    assert sha(PLANS) == PLAN_SHA and sha(EXTERNAL) == EXTERNAL_SHA
    freeze = json.loads(PLANS.read_text()); assert len(freeze['records']) == 240
    assert all(sha(r['plan']) == r['plan_sha256'] for r in freeze['records'])
    with np.load(EXTERNAL) as f:
        assert f['routes'].shape == (5519, 48, 10)
        req = f['req_ids'].copy(); assert np.all(req >= 32)
        assert len(np.unique(req)) == 41
    hashes = source_hashes(); hashes[str(Path(__file__))] = sha(__file__)
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for f in as_completed([pool.submit(worker, (r, str(args.output))) for r in freeze['records']]):
            r = f.result(); records.append(r)
            if len(records)%20 == 0:
                print('complete', len(records), '/240', flush=True)
    totals = []; all_costs = {}
    for case in sorted({r['case'] for r in records}):
        rr = [r for r in records if r['case'] == case]
        values = np.stack([np.load(args.output/(r['label']+'.npz'))['cost'] for r in rr]).sum(axis=0)
        all_costs[case] = values
        request_means = np.asarray([values[req == r].mean() for r in np.unique(req)])
        totals.append(dict(case=case, token_weighted_mean_ns=float(values.mean()), p99_ns=float(np.percentile(values, 99)),
                           equal_request_mean_ns=float(request_means.mean()), min_request_mean_ns=float(request_means.min()),
                           max_request_mean_ns=float(request_means.max())))
    np.savez_compressed(args.output/'paired48layer_costs.npz', req_ids=req, **all_costs)
    assert sha(PLANS) == PLAN_SHA and sha(EXTERNAL) == EXTERNAL_SHA
    assert all(sha(r['plan']) == r['plan_sha256'] for r in freeze['records'])
    assert all(sha(p) == h for p, h in hashes.items())
    doc = dict(status='complete no-refit single-external-task frozen48layer FFN replay',
               created_utc=datetime.now(timezone.utc).isoformat(), source_sha256=hashes,
               plans_sha256=PLAN_SHA, external_path=str(EXTERNAL), external_sha256=EXTERNAL_SHA,
               tokens=5519, work_requests=41, independent_client_sessions=1,
               excluded_warmup_decode=11866, records=records, totals=totals,
               paired_costs_sha256=sha(args.output/'paired48layer_costs.npz'),
               controls='Exactly the same240plans frozen before external payload acquisition; no label/shared/layout/refit or per-query movement. Same native resource model,1us per routed partial reconstruction,0activation,gate-first.',
               limits=['One external document-QA task from one client session, not41 independent task domains.',
                       'Token-weighted and equal-request-weighted means are both reported. P99 is the empirical per-token sum across48FFN layers, not a queueing SLO or end-to-end model measurement.',
                       'Nominal rawI8/A8/4B-output costs on a public NVFP4 checkpoint trace. No original-checkpoint-on-AiF or calibrated hardware claim.',
                       'Rectangular controls require an appropriate native partial-sum/scale ABI, not verified for AiF.',
                       'Numerical router reconstruction validates sampled public records, not reproduction of the unavailable original recorder commit.',
                       'This validates existing geometry controls; no new-method or publication-ready novelty claim.'])
    (args.output/'summary.json').write_text(json.dumps(doc, indent=2)+'\n')
    print(json.dumps(totals, indent=2), flush=True)


if __name__ == '__main__':
    main()
