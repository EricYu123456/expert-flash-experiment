#!/usr/bin/env python3
"""Replay matched phase/complement/joined plans on the same native Graph."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from research_20261102_qwen38_fit import TRAIN, TRAIN_SHA, sha
from research_20261102_qwen38_replay import BASE, SOURCE, RECT, inventories, fit_shared, build

ROOT = Path(__file__).resolve().parents[1]
FIT = BASE/'qwen38-placement/joint-fit3'


def worker(task):
    layer, geometry, strategy, samples, reduce_ns, output = task
    target = Path(output); inv = inventories()
    phase = json.loads((FIT/f'L{layer}.{geometry}.phase.json').read_text())
    joined = json.loads((FIT/f'L{layer}.{geometry}.joined.json').read_text())
    if strategy == 'phase_original':
        labels = phase['original_labels']
    elif strategy == 'global_complement':
        labels = phase['complement_labels']
    elif strategy == 'phase_matched':
        labels = phase['learned_labels']
    else:
        assert strategy == 'joined_matched'; labels = joined['learned_labels']
    plans = {p:dict(label=phase['inventories'][p], labels=labels[i]) for i, p in enumerate(('gate_up', 'down'))}
    with np.load(TRAIN) as f:
        fit_routes = f['routes'][:, ::4, layer].reshape(-1, 10)
        route = f['routes'][:, :, layer].copy()
    fixed_shared, fit_history = fit_shared(fit_routes, plans, inv)
    steps = np.arange(route.shape[1]) if not samples else np.unique(np.linspace(0, route.shape[1]-1, samples, dtype=int))
    selected = route[:, steps].reshape(-1, 10)
    variants = {p:inv[p][plans[p]['label']]['variants'] for p in plans}
    shared = {p:inv[p][v['label']]['variants'][v['color']] for p, v in fixed_shared.items()}
    values = []; bounds = []; trace = None
    for s in selected:
        items = []
        for e in s:
            items.append(dict(gate_up=variants['gate_up'][labels[0][e]], down=variants['down'][labels[1][e]],
                              gu_partial=geometry == 'rect', down_partial=geometry == 'rect'))
        items.append(dict(gate_up=shared['gate_up'], down=shared['down'], gu_partial=False, down_partial=False))
        result = build(items, reduce_ns, 0).run(False, trace=trace is None)
        occupied = np.zeros(16, dtype=np.int64)
        for item in items:
            for p in ('gate_up', 'down'):
                for job in item[p]:
                    r, c = job['rows'], job['columns']
                    occupied[job['chip']] += ((r*c+65535)//65536)*28000+250+(c+1)//2+2*r
        bound = 2000+int(occupied.max()); assert result['makespan_ns'] >= bound
        values.append(result['makespan_ns']); bounds.append(bound)
        if trace is None:
            trace = dict(layer=layer, geometry=geometry, strategy=strategy, selected=s.tolist(),
                         fixed_shared=fixed_shared, items=items, **result)
    name = f'L{layer}.{geometry}.{strategy}.r{reduce_ns}'
    np.savez_compressed(target/(name+'.npz'), token_steps=steps, cost=np.asarray(values), native_chip_bound=np.asarray(bounds))
    (target/(name+'.trace.json')).write_text(json.dumps(trace, indent=2)+'\n')
    (target/(name+'.plan.json')).write_text(json.dumps(dict(plans=plans, fixed_shared=fixed_shared, shared_fit_history=fit_history), indent=2)+'\n')
    result = dict(layer=layer, geometry=geometry, strategy=strategy, label=name, samples=len(values),
                  reduction_ns=reduce_ns, mean_ns=float(np.mean(values)), p99_ns=float(np.percentile(values, 99)),
                  mean_bound_ns=float(np.mean(bounds)), fixed_shared=fixed_shared,
                  artifact_sha256={ext:sha(target/(name+'.'+ext)) for ext in ('npz', 'trace.json', 'plan.json')})
    print(name, result['mean_ns']/1e6, result['p99_ns']/1e6, flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--layers', type=int, default=3)
    ap.add_argument('--samples-per-request', type=int, default=16); ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--reduction-ns', type=int, default=1000)
    ap.add_argument('--output', type=Path, default=BASE/'qwen38-placement/joint-smoke')
    ap.add_argument('--strategies', nargs='+', default=['phase_original', 'global_complement', 'phase_matched', 'joined_matched'])
    args = ap.parse_args(); args.output.mkdir(parents=True, exist_ok=True); assert sha(TRAIN) == TRAIN_SHA
    fit = json.loads((FIT/'fit.json').read_text()); assert all(sha(p) == h for p, h in fit['source_sha256'].items())
    paths = [FIT/'fit.json', SOURCE/'fit.json', SOURCE/'templates.json', TRAIN, RECT]
    paths += [FIT/f'L{l}.{g}.{obj}.json' for l in range(args.layers) for g in ('whole', 'rect') for obj in ('phase', 'joined')]
    hashes = {str(p):sha(p) for p in paths}
    sources = [Path(__file__), Path(__file__).with_name('research_20261102_qwen38_replay.py'),
               Path(__file__).with_name('research_20261102_granite4_pipeline.py')]
    source_hashes = {str(p):sha(p) for p in sources}
    tasks = [(l, g, s, args.samples_per_request, args.reduction_ns, str(args.output))
             for l in range(args.layers) for g in ('whole', 'rect') for s in args.strategies]
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for f in as_completed([pool.submit(worker, t) for t in tasks]):
            records.append(f.result())
    totals = []
    for g in ('whole', 'rect'):
        for s in args.strategies:
            rr = [r for r in records if r['geometry'] == g and r['strategy'] == s]
            values = np.stack([np.load(args.output/(r['label']+'.npz'))['cost'] for r in rr]).sum(axis=0)
            totals.append(dict(geometry=g, strategy=s, mean_ns=float(values.mean()), p99_ns=float(np.percentile(values, 99))))
    assert all(sha(p) == h for p, h in hashes.items()) and all(sha(p) == h for p, h in source_hashes.items())
    out = dict(status='complete train-only matched native joint geometry replay', created_utc=datetime.now(timezone.utc).isoformat(),
               arguments={**vars(args), 'output':str(args.output)}, inputs_sha256=hashes, source_sha256=source_hashes,
               records=records, totals=totals,
               limits=['Same one-copy shared-search, native locking, all transfer and FFN dependencies for every strategy.',
                       'Rectangular outputs require a compatible native partial-sum ABI; reduction_ns is a CPU sensitivity, not device calibration.',
                       'Train-only first3 layers, one repeated task; nominalI8 costs on NVFP4 routes, not native checkpoint performance.',
                       'Global complement and ordinary stripe shapes are existing-method controls; no new algorithm or publication claim.'])
    (args.output/'screen.json').write_text(json.dumps(out, indent=2)+'\n')
    print(json.dumps(totals, indent=2), flush=True)


if __name__ == '__main__':
    main()
