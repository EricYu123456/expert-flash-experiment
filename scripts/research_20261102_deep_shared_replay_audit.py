#!/usr/bin/env python3
"""Independent shape/resource/precedence audit of DeepSeek replay artifacts."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'results/research-2026-11-02/round2'
RES = ['host', 'controller', 'pcie']+[f'channel{i}' for i in range(8)]+[f'chip{i}' for i in range(16)]+['sync']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ceildiv(a, b):
    return (a+b-1)//b


def verify_trace(t, expected_items, cost, activation, combine):
    assert t['items'] == expected_items and t['makespan_ns'] == int(cost)
    ns = t['trace']; assert len(ns) == t['nodes']
    assert max(n['end_ns'] for n in ns) == int(cost)
    work = Counter(); resources = defaultdict(list)
    for i, n in enumerate(ns):
        assert n['index'] == i and 0 <= n['start_ns'] <= n['end_ns']
        assert all(0 <= d < i and ns[d]['end_ns'] <= n['start_ns'] for d in n['deps'])
        dt = n['end_ns']-n['start_ns']; work[n['resource']] += dt
        if dt:
            resources[n['resource']].append((n['start_ns'], n['end_ns']))
    for intervals in resources.values():
        intervals.sort(); assert all(b <= c for (_, b), (c, _) in zip(intervals, intervals[1:]))
    assert t['resource_work_ns'] == [work[x] for x in RES]
    assert work['host'] == 16000+7*activation+combine
    shapes = []
    for phase in (0, 1):
        for e, item in enumerate(expected_items):
            rows = item['gu_rows'] if phase == 0 else item['down_rows']
            columns = 2048 if phase == 0 else item['down_columns']
            shapes += [(phase, e, c, r, columns) for c, r in enumerate(rows) if r]
    stages = ['matrix_command', 'input_channel_reload', 'nonpreemptive_native_compute', 'output_channel', 'output_pcie']
    jobs = defaultdict(list); output = {}; ready = {}; submissions = {}; guoutputs = defaultdict(list); locks = defaultdict(list)
    for n in ns:
        if n['kind'] in stages[:4]:
            jobs[n['job']].append(n)
        if n['kind'] == 'output_pcie':
            output[n['deps'][0]] = n
        if n['kind'] == 'whole_expert_gu_host_ready':
            ready[n['expert']] = n
        if n['kind'] == 'expert_down_host_submission':
            submissions[n['expert']] = n
    assert len(jobs) == len(shapes)
    for j, (phase, e, c, r, columns) in enumerate(shapes):
        nodes = jobs[j]; assert len(nodes) == 4
        nodes = nodes+[output[nodes[-1]['index']]]
        assert [n['kind'] for n in nodes] == stages
        assert [n['resource'] for n in nodes] == ['controller', f'channel{c%8}', f'chip{c}', f'channel{c%8}', 'pcie']
        assert all(n['expert'] == e and n['phase'] == phase for n in nodes)
        assert all(n['chip'] == c and n['job'] == j for n in nodes[:4])
        assert [n['end_ns']-n['start_ns'] for n in nodes] == [250, ceildiv(columns, 2), 28000*ceildiv(r*columns, 65536), r*2, ceildiv(r*4, 8)]
        assert all(a['end_ns'] <= b['start_ns'] for a, b in zip(nodes, nodes[1:]))
        assert nodes[0]['acquire'] and nodes[3]['release']
        assert not any(n['acquire'] for n in nodes[1:]) and not any(n['release'] for n in nodes[:3]+nodes[4:])
        locks[c].append((nodes[0]['start_ns'], nodes[3]['end_ns']))
        if phase == 0:
            guoutputs[e].append(nodes[-1]['index'])
        else:
            ds = submissions[e]; act = ns[ds['deps'][0]]
            assert act['kind'] == 'host_activation' and act['deps'] == [ready[e]['index']]
            assert nodes[0]['deps'] == [ds['index']]
    for intervals in locks.values():
        intervals.sort(); assert all(b <= c for (_, b), (c, _) in zip(intervals, intervals[1:]))
    assert len(ready) == len(submissions) == 7
    for e in range(7):
        assert sorted(ready[e]['deps']) == sorted(guoutputs[e])
    assert work['pcie'] == 256+6*176+352+sum(ceildiv(r*4, 8) for _, _, _, r, _ in shapes)
    return len(ns), len(shapes)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--input', type=Path, default=BASE/'deep-shared-replay/full3')
    args = ap.parse_args(); path = args.input/'screen.json'; doc = json.loads(path.read_text())
    assert all(sha(p) == h for p, h in doc['inputs_sha256'].items())
    assert all(sha(p) == h for p, h in doc['source_sha256'].items())
    frozen = json.loads((ROOT/'results/research-2026-11-02/projection-study/deepseek.frozen_plan.json').read_text())
    oracle = json.loads((BASE/'deep-shared-boundary/full/oracle.json').read_text())
    phases = {(p['layer'], p['name']):p for p in frozen['phases']}
    allcosts = defaultdict(list); traces = nodes = jobs = scalars = 0
    for rec in doc['records']:
        layer = rec['layer']; policy = rec['policy']; name = rec['label']
        npz = args.input/(name+'.npz'); tracepath = args.input/(name+'.trace.json')
        assert sha(npz) == rec['npz_sha256'] and sha(tracepath) == rec['trace_sha256']
        origin = next(r for r in oracle['records'] if r['layer'] == layer)
        with np.load(ROOT/origin['npz']) as f, np.load(npz) as out:
            indices = out['indices'].copy(); actual = out['cost'].copy()
            expected_bound = f[policy+'_native_chip_bound_ns'][indices]
            assert np.array_equal(expected_bound, out['native_chip_bound']) and np.all(actual >= expected_bound)
            gu = np.asarray(phases[layer, 'gate_up']['chosen_rows']); down = np.asarray(phases[layer, 'down']['chosen_rows'])
            selected = f['selected'][indices[0]]
            if policy == 'onecopy':
                sg = f['onecopy_gu_rows']; sd = f['onecopy_down_rows']
            else:
                sg = f[policy+'_gu_rows'][indices[0]]; sd = f[policy+'_down_rows']
            assert sg.sum() == 5632 and sd.sum() == 2048
            items = [dict(gu_rows=gu[e].tolist(), down_rows=down[e].tolist(), down_columns=1408) for e in selected]
            items.append(dict(gu_rows=sg.tolist(), down_rows=sd.tolist(), down_columns=2816))
        trace = json.loads(tracepath.read_text())
        assert trace['selected'] == selected.tolist() and trace['route_index'] == int(indices[0])
        n, j = verify_trace(trace, items, actual[0], doc['arguments']['activation_ns'], doc['arguments']['combine_ns'])
        traces += 1; nodes += n; jobs += j; scalars += len(actual)
        assert rec['mean_ns'] == float(actual.mean()) and rec['p99_ns'] == float(np.percentile(actual, 99))
        allcosts[policy, rec['down_first']].append(actual)
    for total in doc['totals']:
        values = np.stack(allcosts[total['policy'], total['down_first']]).sum(axis=0)
        assert total['mean_ns'] == float(values.mean()) and total['p99_ns'] == float(np.percentile(values, 99))
    best_onecopy = min((r for r in doc['totals'] if r['policy'] in ('onecopy', 'r0')), key=lambda r:r['mean_ns'])
    comparison = []
    for r in (1, 2, 3, 11):
        best = min((t for t in doc['totals'] if t['policy'] == 'r'+str(r)), key=lambda t:t['mean_ns'])
        comparison.append(dict(r=r, total_extra_gu_bytes_per_layer=r*1048576,
                               evaluated_layers_extra_bytes=r*1048576*doc['arguments']['layers'],
                               mean_ns=best['mean_ns'], p99_ns=best['p99_ns'], down_first=best['down_first'],
                               gain_vs_best_onecopy_mean_pct=100*(1-best['mean_ns']/best_onecopy['mean_ns']),
                               gain_vs_best_onecopy_p99_pct=100*(1-best['p99_ns']/best_onecopy['p99_ns'])))
    result = dict(status='PASS', created_utc=datetime.now(timezone.utc).isoformat(),
                  source_sha256=sha(__file__), screen_sha256=sha(path), trace_count=traces,
                  trace_nodes=nodes, trace_jobs=jobs, scalar_cases=scalars, best_onecopy=best_onecopy,
                  comparisons=comparison, source_artifact_hashes_verified=True,
                  limits=['Only saved representative full traces have this independent trace check; all queries also carry scheduler internal resource/ownership assertions.',
                          'All scalar costs and oracle bounds checked. The separate boundary audit covers all physical row inventories and legal split choices.',
                          'No new hardware or numeric capability is established by this timing audit. Train-only first3layer, not full-model or heldout speed.'])
    (args.input/'independent_audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
