#!/usr/bin/env python3
"""Sealed static plans and compiled replay of the unchanged resource model."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import ctypes
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np

from research_20261102_qwen38_fast_replay import encode, evaluate
from research_20261102_qwen38_fit import TRAIN, TRAIN_SHA, sha
from research_20261102_qwen38_replay import BASE, RECT, SOURCE, build, choose_records, fit_shared, inventories

ROOT = Path(__file__).resolve().parents[1]
LIB = BASE/'qwen38-placement/native_replay_v2.so'


def library():
    lib = ctypes.CDLL(str(LIB)); p32 = np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
    p64 = np.ctypeslib.ndpointer(dtype=np.int64, flags='C_CONTIGUOUS')
    lib.eval_native_ffn.argtypes = [p32, p32, p32, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                   ctypes.c_int64, ctypes.c_int64, ctypes.c_int, p64]
    lib.eval_native_ffn.restype = None
    return lib


def validate(source, output):
    doc = json.loads((source/'screen.json').read_text()); lib = library(); inv = inventories(); results = []
    with np.load(TRAIN) as f:
        routes = f['routes'].copy()
    started = time.monotonic()
    for r in doc['records']:
        planpath = source/(r['label']+'.plan.json'); costpath = source/(r['label']+'.npz')
        assert sha(planpath) == r['artifact_sha256']['plan.json'] and sha(costpath) == r['artifact_sha256']['npz']
        plan = json.loads(planpath.read_text()); desc, partial = encode(plan, inv)
        with np.load(costpath) as f:
            selected = routes[:, f['token_steps'], r['layer']].reshape(-1, 10); reference = f['cost'].copy()
        got = evaluate(lib, desc, partial, selected, r['reduction_ns'], r.get('activation_ns', 0), r.get('down_first', False))
        assert np.array_equal(got, reference), (r['label'], int(np.abs(got-reference).max()))
        results.append(dict(label=r['label'], queries=len(got), maximum_difference_ns=0))
    output.mkdir(parents=True, exist_ok=True)
    report = dict(status='PASS exact compiledv2/Python equivalence', created_utc=datetime.now(timezone.utc).isoformat(),
                  source_screen=str(source/'screen.json'), source_screen_sha256=sha(source/'screen.json'),
                  sources_sha256=source_hashes(), queries=sum(r['queries'] for r in results),
                  elapsed_seconds=time.monotonic()-started, records=results,
                  scope='Equivalent software cost model only, not numerical/device verification.')
    (output/'equivalence.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'records'}, indent=2), flush=True)


def source_hashes():
    names = ('research_20261102_qwen38_compiled_campaign.py', 'research_20261102_native_replay_v2.cpp',
             'research_20261102_qwen38_fast_replay.py', 'research_20261102_qwen38_replay.py',
             'research_20261102_granite4_pipeline.py')
    return {str(p):sha(p) for p in [ROOT/'scripts'/n for n in names]+[LIB]}


def prepare_fit(fitdir, layers, cases, out):
    f = json.loads((fitdir/'fit.json').read_text())
    assert all(sha(p) == h for p, h in f['source_sha256'].items())
    assert sha(fitdir/'templates.json') == sha(SOURCE/'templates.json')
    inv = inventories(); records = []; paths = [TRAIN, fitdir/'fit.json', fitdir/'templates.json', RECT]
    for l in range(layers):
        with np.load(TRAIN) as data:
            fit_routes = data['routes'][:, ::4, l].reshape(-1, 10)
        rr = {p:json.loads((fitdir/f'layer{l}.{p}.json').read_text())['records'] for p in ('gate_up', 'down')}
        paths += [fitdir/f'layer{l}.{p}.json' for p in rr]
        for c in cases:
            plans = choose_records(rr, c); shared, history = fit_shared(fit_routes, plans, inv)
            name = f'L{l}.{c}'; path = out/(name+'.plan.json')
            path.write_text(json.dumps(dict(plans=plans, fixed_shared=shared, shared_fit_history=history), indent=2)+'\n')
            records.append(dict(layer=l, case=c, label=name, plan=str(path), plan_sha256=sha(path)))
    freeze = dict(status='frozen complete static physical views before replay', created_utc=datetime.now(timezone.utc).isoformat(),
                  fit_directory=str(fitdir), records=records, inputs_sha256={str(p):sha(p) for p in paths},
                  source_sha256=source_hashes(), external_task_payloads_opened=False,
                  protocol='Routed labels and one-copy shared geometry fitted exclusively on16same-task requests, steps0,4,...796. No per-query runtime remapping.')
    (out/'plans.frozen.json').write_text(json.dumps(freeze, indent=2)+'\n')
    print('plans frozen', sha(out/'plans.frozen.json'), flush=True)
    return freeze


def items_for(plan, inv, selected):
    items = []
    for e in selected:
        item = {}
        for p, name in enumerate(('gate_up', 'down')):
            r = plan['plans'][name]; candidate = inv[name][r['label']]
            item[name] = candidate['variants'][r['labels'][e]]
            item['gu_partial' if p == 0 else 'down_partial'] = candidate['conditional_partial_sum']
        items.append(item)
    item = dict(gu_partial=False, down_partial=False)
    for name, r in plan['fixed_shared'].items():
        item[name] = inv[name][r['label']]['variants'][r['color']]
    items.append(item)
    return items


def worker(task):
    record, reduction, activation, down_first, output = task
    target = Path(output); inv = inventories(); lib = library(); planpath = Path(record['plan'])
    assert sha(planpath) == record['plan_sha256']; plan = json.loads(planpath.read_text())
    desc, partial = encode(plan, inv)
    with np.load(TRAIN) as f:
        selected = f['routes'][:, :, record['layer']].reshape(-1, 10)
    started = time.monotonic(); costs = evaluate(lib, desc, partial, selected, reduction, activation, down_first)
    service = np.zeros((513, 16), dtype=np.int64)
    for e in range(513):
        for phase in (0, 1):
            for chip, rows, columns, waves in desc[e, phase]:
                if rows:
                    service[e, chip] += int(waves)*28000+250+(int(columns)+1)//2+2*int(rows)
    bound = (service[selected].sum(axis=1)+service[512]).max(axis=1)+2000
    assert np.all(costs >= bound)
    items = items_for(plan, inv, selected[0]); diagnostic = build(items, reduction, activation).run(down_first, trace=True)
    assert int(costs[0]) == diagnostic['makespan_ns']
    name = record['label']
    np.savez_compressed(target/(name+'.npz'), cost=costs, native_chip_bound=bound, token_steps=np.arange(799))
    trace = dict(layer=record['layer'], case=record.get('case'), selected=selected[0].tolist(),
                 items=items, fixed_shared=plan['fixed_shared'], **diagnostic)
    (target/(name+'.trace.json')).write_text(json.dumps(trace, indent=2)+'\n')
    result = dict(**record, reduction_ns=reduction, activation_ns=activation, down_first=down_first,
                  samples=len(costs), mean_ns=float(costs.mean()), p99_ns=float(np.percentile(costs, 99)),
                  mean_bound_ns=float(bound.mean()), compute_seconds=time.monotonic()-started,
                  artifact_sha256=dict(npz=sha(target/(name+'.npz')), **{'trace.json':sha(target/(name+'.trace.json')), 'plan.json':sha(planpath)}))
    print(name, result['mean_ns']/1e6, flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--validate', type=Path)
    ap.add_argument('--fit-directory', type=Path); ap.add_argument('--layers', type=int, default=3)
    ap.add_argument('--cases', nargs='+', default=['group_g16', 'group_best', 'overlap_best', 'whole_row', 'rect_min_io4'])
    ap.add_argument('--workers', type=int, default=8); ap.add_argument('--reduction-ns', type=int, default=1000)
    ap.add_argument('--activation-ns', type=int, default=0); ap.add_argument('--down-first', action='store_true')
    ap.add_argument('--output', type=Path, required=True); args = ap.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    assert sha(TRAIN) == TRAIN_SHA
    if args.validate:
        validate(args.validate, args.output); return
    assert args.fit_directory
    freeze = prepare_fit(args.fit_directory, args.layers, args.cases, args.output)
    frozen_hash = sha(args.output/'plans.frozen.json'); sources = source_hashes(); records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        tasks = [(r, args.reduction_ns, args.activation_ns, args.down_first, str(args.output)) for r in freeze['records']]
        for f in as_completed([pool.submit(worker, t) for t in tasks]):
            records.append(f.result()); (args.output/'partial.json').write_text(json.dumps(records, indent=2)+'\n')
    totals = []
    for c in args.cases:
        rr = [r for r in records if r['case'] == c]
        costs = np.stack([np.load(args.output/(r['label']+'.npz'))['cost'] for r in rr]).sum(axis=0)
        totals.append(dict(case=c, mean_ns=float(costs.mean()), p99_ns=float(np.percentile(costs, 99))))
    assert sha(args.output/'plans.frozen.json') == frozen_hash
    assert all(sha(p) == h for p, h in freeze['inputs_sha256'].items())
    assert all(sha(p) == h for p, h in sources.items())
    doc = dict(status='complete compiled replay of frozen full-FFN physical views', created_utc=datetime.now(timezone.utc).isoformat(),
               arguments={**vars(args), 'output':str(args.output), 'fit_directory':str(args.fit_directory)},
               plan_freeze_sha256=frozen_hash, inputs_sha256=freeze['inputs_sha256'], source_sha256=sources,
               records=records, totals=totals,
               limits=['Exact software replay of the earlier Graph contract, with a Python diagnostic trace per plan; not calibrated NAND or native numerical inference.',
                       'Same nominalI8/A8/4B-output costs on public NVFP4 routes. No checkpoint conversion, full-model quality or silicon precision claim.',
                       'Only FFN service for specified layers; router, attention, PLE and other model components omitted.',
                       'Only the original repeated task training stream; source engine provenance caveat remains.',
                       'All geometry/stripe families are baselines without a new-method claim; rectangular host sums require compatible output ABI.'])
    (args.output/'screen.json').write_text(json.dumps(doc, indent=2)+'\n')
    print(json.dumps(totals, indent=2), flush=True)


if __name__ == '__main__':
    main()
