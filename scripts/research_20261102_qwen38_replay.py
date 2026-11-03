#!/usr/bin/env python3
"""Native full-FFN schedule controls for the frozen Qwen3.8 route study.

Whole rows are the established relocation family. Rectangular controls are
conditional on an output/scale ABI that permits host partial reconstruction.
Every native job owns its chip from command through output-channel drain.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from research_20261102_granite4_pipeline import Graph
from research_20261102_qwen38_fit import TRAIN, TRAIN_SHA, templates, table

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'results/research-2026-11-02/round2'
SOURCE = BASE/'qwen38-placement/phase-fit3'
RECT = BASE/'qwen38-placement/independent_rectangles.json'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build(items, reduction_ns, activation_ns=0):
    g = Graph(); submit = g.add(0, 2000, kind='initial_host_submission')
    upload = g.add(2, 320, [submit], kind='common_input_to_controller')
    ready = []
    for ei, item in enumerate(items):
        output = []
        for j in item['gate_up']:
            r, c = j['rows'], j['columns']
            output.append(g.matrix(ei, j['chip'], 0, r, c, (r*c+65535)//65536, submit, upload))
        barrier = g.add(27, 0, output, expert=ei, kind='whole_expert_gu_host_ready')
        if item['gu_partial']:
            barrier = g.add(0, reduction_ns, [barrier], expert=ei, kind='gu_partial_reconstruction')
        ready.append(barrier)
    complete = []
    for ei, item in enumerate(items):
        act = g.add(0, activation_ns, [ready[ei]], 1, ei, 'host_activation')
        ds = g.add(0, 2000, [act], 1, ei, 'expert_down_host_submission')
        up = g.add(2, 80, [ds], 1, ei, 'private_input_to_controller')
        output = []
        for j in item['down']:
            r, c = j['rows'], j['columns']
            output.append(g.matrix(ei, j['chip'], 1, r, c, (r*c+65535)//65536, ds, up))
        if item['down_partial']:
            complete.append(g.add(0, reduction_ns, output, 1, ei, 'down_partial_reconstruction'))
        else:
            complete += output
    g.add(0, 0, complete, 1, kind='host_combine')
    return g


def inventories():
    values = json.loads((SOURCE/'templates.json').read_text())
    rect = json.loads(RECT.read_text())
    controls = rect['controls']
    for label, key in [('rect_min_output', 'gate_up_two_stripes_min_output'),
                       ('rect_min_io4', 'gate_up_three_stripes_min_io4')]:
        jobs = controls[key]['rectangles']
        variants = [[dict(chip=(i+color)%16, row_start=r, rows=n, column_start=c, columns=m)
                     for i, (r, n, c, m) in enumerate(jobs)] for color in range(16)]
        values['gate_up'].append(dict(label=label, family='rectangular', conditional_partial_sum=True, variants=variants))
    return {p:{v['label']:v for v in vv} for p, vv in values.items()}


def records_for(layer):
    return {p:json.loads((SOURCE/f'layer{layer}.{p}.json').read_text())['records'] for p in ('gate_up', 'down')}


def choose_records(records, case):
    chosen = {}
    for p, values in records.items():
        if case == 'group_g16':
            r = next(v for v in values if v['label'] == 'disjoint_g16')
        elif case == 'group_best':
            r = min((v for v in values if v['family'] == 'disjoint'), key=lambda v:v['mean_bound_ns'])
        elif case == 'overlap_best':
            r = min((v for v in values if v['family'] in ('disjoint', 'overlap')), key=lambda v:v['mean_bound_ns'])
        elif case == 'whole_row':
            r = min((v for v in values if not v['conditional_partial_sum']), key=lambda v:v['mean_bound_ns'])
        elif (p == 'gate_up' and case == 'rect_down_only') or (p == 'down' and case == 'rect_gu_only'):
            r = min((v for v in values if not v['conditional_partial_sum']), key=lambda v:v['mean_bound_ns'])
        else:
            r = next(v for v in values if v['label'] == 'exact_work_rectangular')
        r = dict(r)
        if p == 'gate_up' and case in ('rect_min_output', 'rect_min_io4', 'rect_gu_only'):
            r['label'] = 'rect_min_io4' if case == 'rect_min_io4' else 'rect_min_output'
        chosen[p] = r
    return chosen


def fit_shared(selected, plans, inv):
    """Same static one-copy shared-expert controls for every routed family.

    Enumerate GU/down candidates alternately against the joined native-chip
    bound. The result is a legal fixed inventory; no per-query weight motion.
    """
    load = np.zeros((len(selected), 16), dtype=np.int64)
    for p in plans:
        load += table(inv[p][plans[p]['label']])[np.asarray(plans[p]['labels'])][selected].sum(axis=1)
    candidates = {}
    for p in ('gate_up', 'down'):
        keep = ('disjoint_g8', 'disjoint_g16', 'balanced_whole_row', 'cyclic_width10') if p == 'gate_up' else ('disjoint_g8', 'disjoint_g16', 'balanced_whole_row', 'cyclic_width13')
        candidates[p] = [(name, color, row) for name in keep for color, row in enumerate(table(inv[p][name]))]
    choices = {p:next(x for x in candidates[p] if x[:2] == ('balanced_whole_row', 0)) for p in candidates}
    improvements = []
    for sweep in range(8):
        changed = False
        for p in ('gate_up', 'down'):
            other = 'down' if p == 'gate_up' else 'gate_up'; base = load+choices[other][2]
            scored = []
            for name, color, row in candidates[p]:
                value = (base+row).max(axis=1)
                scored.append((int(value.sum()), name, color, row))
            best = min(scored, key=lambda x:(x[0], x[1], x[2]))
            changed |= choices[p][:2] != best[1:3]; choices[p] = best[1:]
            improvements.append(dict(sweep=sweep, phase=p, label=best[1], color=best[2], mean_bound_ns=best[0]/len(selected)))
        if not changed:
            break
    return {p:dict(label=x[0], color=x[1]) for p, x in choices.items()}, improvements


def worker(task):
    layer, case, samples, reduction_ns, activation_ns, down_first, output = task
    target = Path(output); inv = inventories(); records = records_for(layer); plans = choose_records(records, case)
    with np.load(TRAIN) as f:
        fit_routes = f['routes'][:, ::4, layer].reshape(-1, 10)
        all_routes = f['routes'][:, :, layer].copy()
    fixed_shared, search = fit_shared(fit_routes, plans, inv)
    steps = np.arange(all_routes.shape[1]) if not samples else np.unique(np.linspace(0, all_routes.shape[1]-1, samples, dtype=int))
    selected = all_routes[:, steps].reshape(-1, 10)
    labels = {p:np.asarray(v['labels']) for p, v in plans.items()}
    shared = {p:inv[p][v['label']]['variants'][v['color']] for p, v in fixed_shared.items()}
    values = []; bounds = []; trace = None
    for i, route in enumerate(selected):
        items = []
        for e in route:
            items.append(dict(gate_up=inv['gate_up'][plans['gate_up']['label']]['variants'][labels['gate_up'][e]],
                              down=inv['down'][plans['down']['label']]['variants'][labels['down'][e]],
                              gu_partial=inv['gate_up'][plans['gate_up']['label']]['conditional_partial_sum'],
                              down_partial=inv['down'][plans['down']['label']]['conditional_partial_sum']))
        items.append(dict(gate_up=shared['gate_up'], down=shared['down'], gu_partial=False, down_partial=False))
        r = build(items, reduction_ns, activation_ns).run(down_first, trace=trace is None)
        chip = np.zeros(16, dtype=np.int64)
        for item in items:
            for phase in ('gate_up', 'down'):
                for j in item[phase]:
                    row, col = j['rows'], j['columns']
                    chip[j['chip']] += ((row*col+65535)//65536)*28000+250+(col+1)//2+2*row
        bound = 2000+int(chip.max()); assert r['makespan_ns'] >= bound
        values.append(r['makespan_ns']); bounds.append(bound)
        if trace is None:
            trace = dict(layer=layer, case=case, selected=route.tolist(), fixed_shared=fixed_shared, items=items, **r)
    label = f'L{layer}.{case}.r{reduction_ns}.a{activation_ns}.df{int(down_first)}'
    np.savez_compressed(target/(label+'.npz'), cost=np.asarray(values), native_chip_bound=np.asarray(bounds), token_steps=steps)
    (target/(label+'.trace.json')).write_text(json.dumps(trace, indent=2)+'\n')
    (target/(label+'.plan.json')).write_text(json.dumps(dict(layer=layer, case=case, plans=plans, fixed_shared=fixed_shared,
                                                           shared_fit_history=search), indent=2)+'\n')
    return dict(layer=layer, case=case, reduction_ns=reduction_ns, activation_ns=activation_ns, down_first=down_first,
                label=label, samples=len(values), mean_ns=float(np.mean(values)), p99_ns=float(np.percentile(values, 99)),
                mean_bound_ns=float(np.mean(bounds)), routed_families={p:r['label'] for p, r in plans.items()}, fixed_shared=fixed_shared,
                artifact_sha256={ext:sha(target/(label+'.'+ext)) for ext in ('npz', 'trace.json', 'plan.json')})


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--samples-per-request', type=int, default=0)
    ap.add_argument('--layers', type=int, default=3); ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--reduction-ns', type=int, default=0); ap.add_argument('--activation-ns', type=int, default=0)
    ap.add_argument('--down-first', action='store_true')
    ap.add_argument('--cases', nargs='+', default=['group_g16', 'group_best', 'overlap_best', 'whole_row', 'rect_grid', 'rect_min_output', 'rect_min_io4', 'rect_gu_only', 'rect_down_only'])
    ap.add_argument('--output', type=Path, default=BASE/'qwen38-placement/native-smoke')
    args = ap.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    assert sha(TRAIN) == TRAIN_SHA
    fit = json.loads((SOURCE/'fit.json').read_text())
    assert all(sha(p) == h for p, h in fit['source_sha256'].items())
    paths = [SOURCE/'fit.json', SOURCE/'templates.json', TRAIN, RECT]
    paths += [SOURCE/f'layer{l}.{p}.json' for l in range(args.layers) for p in ('gate_up', 'down')]
    hashes = {str(p):sha(p) for p in paths}
    sources = [Path(__file__), Path(__file__).with_name('research_20261102_granite4_pipeline.py'), Path(__file__).with_name('research_20261102_qwen38_fit.py')]
    shashes = {str(p):sha(p) for p in sources}
    records = []
    tasks = [(l, c, args.samples_per_request, args.reduction_ns, args.activation_ns, args.down_first, str(args.output)) for l in range(args.layers) for c in args.cases]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for f in as_completed([pool.submit(worker, t) for t in tasks]):
            r = f.result(); records.append(r); print(r['label'], r['mean_ns']/1e6, r['p99_ns']/1e6, flush=True)
            (args.output/'partial.json').write_text(json.dumps(records, indent=2)+'\n')
    totals = []
    for case in args.cases:
        matching = [r for r in records if r['case'] == case]
        loaded = [np.load(args.output/(r['label']+'.npz')) for r in matching]
        values = np.stack([f['cost'] for f in loaded]).sum(axis=0)
        totals.append(dict(case=case, mean_ns=float(values.mean()), p99_ns=float(np.percentile(values, 99))))
    assert all(sha(p) == h for p, h in hashes.items()) and all(sha(p) == h for p, h in shashes.items())
    out = dict(status='complete train-only native-job full-FFN geometry replay', created_utc=datetime.now(timezone.utc).isoformat(),
               arguments={**vars(args), 'output':str(args.output)}, inputs_sha256=hashes, source_sha256=shashes,
               records=records, totals=totals,
               controls='Same one-copy shared-expert candidate inventory and coordinate search for every routed family. All16 same-task requests represented. Every matrix job reloads its whole local input view and locks one chip until output channel drain.',
               limits=['Train-only first layers and one repeated task; not external validation or full-model time.',
                       'Native service uses nominal rawI8/A8/4B output, while real trace came from NVFP4. No format conversion or accuracy claim.',
                       'Rectangular jobs require known numeric output/scales and host reconstruction, unproven for AiF. reduction_ns is a sensitivity parameter, not measured here.',
                       'Fixed gate/down priority, not schedule optimality. Native outputs, channels and PCIe charged; activation/reduction defaults optimistic.',
                       'Ordinary stripe/rectangle tilings are strong existing-method controls, not a novelty claim.'])
    (args.output/'screen.json').write_text(json.dumps(out, indent=2)+'\n')
    print(json.dumps(totals, indent=2), flush=True)


if __name__ == '__main__':
    main()
