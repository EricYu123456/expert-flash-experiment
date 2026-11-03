#!/usr/bin/env python3
"""Strict native-job replay of the frozen DeepSeek boundary-copy screen.

The independently checked oracle supplies complete-row owner choices, rather
than a latency estimate. This script evaluates those choices with full GU to
private-input down dependencies, one native context, channels and PCIe.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from research_20261102_granite4_pipeline import Graph

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/research-2026-11-02'
SOURCE = BASE / 'projection-study/deepseek.frozen_plan.json'
ORACLE = BASE / 'round2/deep-shared-boundary/full/oracle.json'
PLANS = BASE / 'round2/deep-shared-boundary/executable_row_plans.json'
DEST = BASE / 'round2/deep-shared-replay'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(items, activation_ns=0, combine_ns=0):
    g = Graph()
    submit = g.add(0, 2000, kind='initial_host_submission')
    upload = g.add(2, 256, [submit], kind='common_input_to_controller')
    ready = []
    for ei, item in enumerate(items):
        outputs = []
        for c, rows in enumerate(item['gu_rows']):
            if rows:
                w = (rows*2048+65535)//65536
                outputs.append(g.matrix(ei, c, 0, rows, 2048, w, submit, upload))
        ready.append(g.add(27, 0, outputs, expert=ei, kind='whole_expert_gu_host_ready'))
    terminal = []
    for ei, item in enumerate(items):
        act = g.add(0, activation_ns, [ready[ei]], 1, ei, 'host_activation')
        ds = g.add(0, 2000, [act], 1, ei, 'expert_down_host_submission')
        cols = item['down_columns']
        up = g.add(2, (cols+7)//8, [ds], 1, ei, 'private_input_to_controller')
        for c, rows in enumerate(item['down_rows']):
            if rows:
                w = (rows*cols+65535)//65536
                terminal.append(g.matrix(ei, c, 1, rows, cols, w, ds, up))
    g.add(0, combine_ns, terminal, 1, kind='host_combine')
    return g


def worker(task):
    layer, label, down_first, samples, activation, combine, output = task
    source = json.loads(SOURCE.read_text())
    oracle = json.loads(ORACLE.read_text())
    record = next(r for r in oracle['records'] if r['layer'] == layer)
    phases = {p['name']:p for p in source['phases'] if p['layer'] == layer}
    gu = np.asarray(phases['gate_up']['chosen_rows'])
    down = np.asarray(phases['down']['chosen_rows'])
    assert np.all(gu.sum(axis=1) == 2816)
    assert np.all(down.sum(axis=1) == 2048)
    with np.load(ROOT / record['npz']) as f:
        selected = f['selected'].copy()
        assert selected.shape == (2048, 6)
        if label == 'onecopy':
            shared_gu = np.broadcast_to(f['onecopy_gu_rows'], (2048, 16)).copy()
            shared_down = f['onecopy_down_rows'].copy()
        else:
            shared_gu = f[label+'_gu_rows'].copy()
            shared_down = f[label+'_down_rows'].copy()
        bound = f[('onecopy' if label == 'onecopy' else label)+'_native_chip_bound_ns'].copy()
    indices = np.arange(2048) if not samples else np.unique(np.linspace(0, 2047, samples, dtype=int))
    assert np.all(shared_gu.sum(axis=1) == 5632) and shared_down.sum() == 2048
    values = []; trace = None; work = None
    for i in indices:
        route = selected[i]
        items = [dict(gu_rows=gu[e].tolist(), down_rows=down[e].tolist(), down_columns=1408) for e in route]
        items.append(dict(gu_rows=shared_gu[i].tolist(), down_rows=shared_down.tolist(), down_columns=2816))
        result = build(items, activation, combine).run(down_first, trace=trace is None)
        assert result['makespan_ns'] >= int(bound[i])
        values.append(result['makespan_ns'])
        if work is None:
            work = np.zeros(len(result['resource_work_ns']), dtype=np.int64)
        work += result['resource_work_ns']
        if trace is None:
            trace = dict(layer=layer, policy=label, down_first=down_first,
                         selected=route.tolist(), route_index=int(i), items=items, **result)
    target = Path(output)
    name = f'L{layer}.{label}.df{int(down_first)}'
    np.savez_compressed(target/(name+'.npz'), indices=indices, cost=np.asarray(values),
                        native_chip_bound=bound[indices])
    (target/(name+'.trace.json')).write_text(json.dumps(trace, indent=2)+'\n')
    return dict(layer=layer, policy=label, down_first=down_first, label=name,
                mean_ns=float(np.mean(values)), p99_ns=float(np.percentile(values, 99)),
                mean_native_bound_ns=float(bound[indices].mean()),
                resource_work_mean_ns=(work/len(indices)).tolist(),
                npz_sha256=sha(target/(name+'.npz')), trace_sha256=sha(target/(name+'.trace.json')))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', type=int, default=0)
    ap.add_argument('--layers', type=int, default=3)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--activation-ns', type=int, default=0)
    ap.add_argument('--combine-ns', type=int, default=0)
    ap.add_argument('--output', type=Path, default=DEST)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = json.loads(SOURCE.read_text()); oracle = json.loads(ORACLE.read_text())
    assert sha(SOURCE) == oracle['source_plan_sha256']
    assert sha(source['train_path']) == source['train_sha256'] == oracle['train_sha256']
    inputs = [SOURCE, ORACLE, PLANS, Path(source['train_path'])]
    inputs += [ROOT/r['npz'] for r in oracle['records'][:args.layers]]
    hashes = {str(p):sha(p) for p in inputs}
    scripts = [Path(__file__), Path(__file__).with_name('research_20261102_granite4_pipeline.py')]
    scripts += [Path(__file__).with_name('research_20261102_single_request.py')]
    script_hashes = {str(p):sha(p) for p in scripts}
    assert 1 <= args.layers <= 26
    policies = ('onecopy', 'r0', 'r1', 'r2', 'r3', 'r11')
    tasks = [(layer, policy, df, args.samples, args.activation_ns, args.combine_ns, str(args.output))
             for layer in source['layers'][:args.layers] for policy in policies for df in (False, True)]
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(worker, task) for task in tasks]):
            rec = future.result(); records.append(rec)
            print(rec['label'], rec['mean_ns']/1e6, rec['p99_ns']/1e6, flush=True)
            (args.output/'partial.json').write_text(json.dumps(records, indent=2)+'\n')
    totals = []
    for policy in policies:
        for df in (False, True):
            matching = [r for r in records if r['policy'] == policy and r['down_first'] == df]
            loaded = [np.load(args.output/(r['label']+'.npz')) for r in matching]
            values = np.stack([a['cost'] for a in loaded]).sum(axis=0)
            bounds = np.stack([a['native_chip_bound'] for a in loaded]).sum(axis=0)
            totals.append(dict(policy=policy, down_first=df, mean_ns=float(values.mean()),
                               p99_ns=float(np.percentile(values, 99)), mean_bound_ns=float(bounds.mean())))
    assert all(sha(p) == h for p, h in hashes.items())
    assert all(sha(p) == h for p, h in script_hashes.items())
    result = dict(status='complete train-only strict DeepSeek shared-FFN replay',
                  created_utc=datetime.now(timezone.utc).isoformat(),
                  arguments={**vars(args), 'output':str(args.output)}, inputs_sha256=hashes,
                  source_sha256=script_hashes, records=records, totals=totals,
                  contract='Same strict Graph for all layouts: full-row GU, whole-expert GU receipt before activation, private down input reload, one native context held command through channel output, host/controller/channel/PCIe charged.',
                  capacity='onecopy/r0 have no duplicates. r copies r/11 of sharedGU; r1=26MiB across all26 layers. Only selected full rows execute, no partial sums or new native operation.',
                  limits=['Training replay, not heldout or full-model latency; attention, router, dense first FFN and other layers excluded.',
                          'Nominal rawI8/A8 and4B outputs remain an uncalibrated numeric/service contract; not original GGUF native inference.',
                          'Frozen occupancy-selected row choices are feasible, not globally optimal schedule choices or placements.',
                          'Both priorities are controls; zero activation/combine default is optimistic.',
                          'Partial overlap replicas and their fixed-owner minmax selector have direct generic RDA/range-overlap priors; no new-method claim.'])
    (args.output/'screen.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(totals, indent=2))


if __name__ == '__main__':
    main()
