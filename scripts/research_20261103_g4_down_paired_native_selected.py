#!/usr/bin/env python3
"""Authorized graph14 pure-unitflow-bit native replay; no graph/selector fitting."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import research_20261103_ordinary_read_backend as backend

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/research-2026-11-02/round2'
SOURCE = BASE / 'g4-down-paired-merge-screen'
OUT = SOURCE / 'native-graph14'
INC = BASE / 'compact-coupled-solver/continuation/native/0107.npz'
CFG = BASE / 'compact-coupled-solver/protocol.frozen.json'
TRAIN = BASE / 'w8-format-training/granite4tiny/decode.routes.npz'
SELF = Path(__file__).resolve()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, x):
    Path(p).write_text(json.dumps(x, indent=2) + '\n')


def freeze():
    OUT.mkdir(exist_ok=False)
    paths = [SELF, SOURCE / 'arrays.npz', SOURCE / 'report.json', SOURCE / 'unitflow-bound/arrays.npz',
             SOURCE / 'unitflow-bound/report.json', INC, CFG, TRAIN,
             ROOT / 'scripts/research_20261103_ordinary_read_backend.py', ROOT / 'scripts/research_20261103_ordinary_read_native.cpp']
    dump(OUT / 'protocol.frozen.json', dict(inputs=[dict(path=str(p), sha256=sha(p)) for p in paths],
        graph=14, selection='Fixed graph14 selected by TRAIN restored chip-work. Runtime never choosesamong32graphs. Pure unitflow endpoints only; no bestofgreedy/postfit substitution.',
        scope='512original Bcost/LBbridge +512restored graph14 native costs. Six Python ReferenceGraph traces: B/new q0,255,511. No other graph/native/CP/weights/heldout/selector benchmark.',
        selector='Zero selector time is optimistic sensitivity only. Unitflow source506ns waived cost restored before ordinary row lowering and independently rechecked.',
        contract='OriginalGU/shared and privatefull512Down input, wholeGU hostactivation/A8 barrier, onecontext/chip, full reload everyjob. Downpair128+128 or0+256 completewave prefixes; immutable originalrow IDs/scales.',
        gate='If native gain<2%mean and<5%P99, stopselectorbench. If >= either, reportbefore same24MiB32wholeDown/partialRDA/staticguards/CPUselector controls.'))
    print(sha(OUT / 'protocol.frozen.json'))


def trace_audit(plan, native, service, expected, label):
    local = np.arange(6, dtype=np.int32)[None]
    got = native.run(plan, local, 6, 1, service, trace=True)
    graph = backend.ReferenceGraph.build(plan, np.arange(7), service)
    reference = graph.run()
    assert int(got['cost'][0]) == reference['cost'] == int(expected)
    assert np.array_equal(got['trace'], reference['trace'])
    tr = reference['trace']; nodes = graph.nodes
    for i, node in enumerate(nodes):
        assert tr[i, 3] - tr[i, 2] == node['duration']
        assert all(tr[d, 3] <= tr[i, 2] for d in node['deps'])
    intervals = [[] for _ in range(16)]
    for job in range(graph.job_count):
        indexes = [i for i, n in enumerate(nodes) if n['job'] == job]
        assert len(indexes) == 4
        cmd, inp, comp, out = indexes
        n = nodes[cmd]; chip = n['chip']; e = n['expert']; phase = n['phase']
        assert nodes[cmd]['acquire'] and nodes[out]['release']
        width = int(plan['input_columns'][e, phase])
        assert nodes[inp]['duration'] == service.transfer(width * service.input_channel_copies, service.channel_bytes_per_second)
        assert width == (1536 if phase == 0 else (512 if e < 6 else 1024))
        intervals[chip].append((int(tr[cmd, 2]), int(tr[out, 3]), job))
        if phase == 1:
            ready = [i for i, x in enumerate(nodes) if x['resource'] == service.resources - 1 and x['expert'] == e and x['phase'] == 0]
            assert len(ready) == 1 and tr[cmd, 2] >= tr[ready[0], 3] + service.host_submit_ns + service.activation_ns
    for spans in intervals:
        spans.sort(); assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))
    cmds = tr[(tr[:, 0] == 1) & (tr[:, 4] == 0) & (tr[:, 5] == 0)]
    assert np.array_equal(cmds[:, 6], np.arange(16))
    assert np.array_equal(cmds[:, 2], 2000 + np.arange(16) * 250)
    dump(OUT / (label + '.nodes.json'), nodes)
    np.savez_compressed(OUT / (label + '.traces.npz'), native=got['trace'], ReferenceGraph=reference['trace'])
    return dict(label=label, cost_ns=int(expected), nodes=len(nodes), jobs=graph.job_count,
                every_node_trace_exact=True, dependencies_resources_owner_reload_barrier=True)


def run():
    protocol = json.loads((OUT / 'protocol.frozen.json').read_text())
    for item in protocol['inputs']:
        assert sha(item['path']) == item['sha256']
    assert not (OUT / 'report.json').exists(); began = time.perf_counter()
    with np.load(INC) as z:
        original = {k: z[k].copy() for k in z.files}
    with np.load(SOURCE / 'arrays.npz') as z:
        screen = {k: z[k].copy() for k in z.files}
    with np.load(SOURCE / 'unitflow-bound/arrays.npz') as z:
        flow = {k: z[k].copy() for k in z.files}
    with np.load(TRAIN) as z:
        assert z['split'].item() == 'train' and z['completed'].all()
        assert np.array_equal(original['routes'], z['routes'][:, :, 0].reshape(512, 6))
    service = backend.Service(**json.loads(CFG.read_text())['service'])
    native = backend.NativeReplay(OUT / 'build')
    base_plan = {k: original[k] for k in ['offsets', 'jobs', 'input_columns', 'total_rows']}
    bridge = native.run(base_plan, original['routes'], 64, 1, service)
    assert np.array_equal(bridge['cost'], original['cost']) and np.array_equal(bridge['lower_bound'], original['lower_bound'])
    routes = original['routes']; rows = original['rows']; edges = screen['layouts'][14]
    pairs = edges[routes].reshape(512, 36, 2); endpoints = flow['assignments'][14]
    bits = (endpoints == pairs[:, :, 1]).astype(np.int8)
    assert np.all((endpoints == pairs[:, :, 0]) | (endpoints == pairs[:, :, 1]))
    expected_work = screen['base'] + np.einsum('qj,qjc->qc', bits, screen['effects'][14][routes].reshape(512, 36, 16))
    restored = flow['fixed_base_with_dispatch'][14] - np.arange(16) * 250
    for q in range(512):
        restored[q] += np.bincount(endpoints[q], minlength=16) * 28256
        restored[q] += np.bincount(endpoints[q][endpoints[q] == pairs[q, :, 0]], minlength=16) * 506
    assert np.array_equal(expected_work, restored) and expected_work.max(1).mean() == 545769.66015625
    costs, lbs, compact_rows, traces = [], [], [], []
    row_view_checks = 0
    for q, route in enumerate(routes):
        selected = np.r_[route, 64]; local_rows = rows[selected].copy()
        for k, e in enumerate(route):
            starts = np.r_[0, np.cumsum(rows[e, 1])]
            images = [np.arange(starts[c], starts[c+1], dtype=np.int32) for c in range(16)]
            for j, (src, dest) in enumerate(edges[e]):
                if bits[q, k*6+j]:
                    local_rows[k, 1, src] = 0; local_rows[k, 1, dest] = 256
                    images[dest] = np.r_[images[dest], images[src]]; images[src] = np.array([], np.int32)
            assert np.array_equal(np.sort(np.concatenate(images)), np.arange(1536))
            assert local_rows[k, 1].sum() == 1536
            assert np.all(local_rows[k, 1] * 512 % 65536 == 0)
            row_view_checks += 1
        assert np.array_equal(local_rows[:, 0], rows[selected, 0]) and np.array_equal(local_rows[6], rows[64])
        plan = backend.lower_rows(local_rows[:, 0], local_rows[:, 1], 1536, [512] * 6 + [1024], service)
        work = backend.occupancy_table(plan, service)['chip_ns'].sum((0, 1))
        assert np.array_equal(work, expected_work[q])
        result = native.run(plan, np.arange(6, dtype=np.int32)[None], 6, 1, service)
        costs.append(result['cost'][0]); lbs.append(result['lower_bound'][0]); compact_rows.append(local_rows)
        if q in (0, 255, 511):
            traces.append(trace_audit(plan, native, service, costs[-1], f'q{q}.paired'))
            b_rows = rows[selected]
            b_plan = backend.lower_rows(b_rows[:, 0], b_rows[:, 1], 1536, [512] * 6 + [1024], service)
            traces.append(trace_audit(b_plan, native, service, bridge['cost'][q], f'q{q}.baseline'))
    costs, lbs = np.asarray(costs), np.asarray(lbs)
    mean_gain = 100 * (1 - costs.mean() / bridge['cost'].mean())
    p99_gain = 100 * (1 - np.percentile(costs, 99) / np.percentile(bridge['cost'], 99))
    np.savez_compressed(OUT / 'arrays.npz', routes=routes, merge_bits=bits, actual_chip_work=expected_work,
                        compact_rows=np.asarray(compact_rows), cost=costs, lower_bound=lbs,
                        baseline_cost=bridge['cost'], baseline_lower_bound=bridge['lower_bound'])
    report = dict(status='PASS complete fixedgraph14 native replay with restored real costs', protocol_sha256=sha(OUT / 'protocol.frozen.json'),
                  graph=14, query_count=512, baseline_cost_and_LB_exact_values=1024, symbolic_expert_view_checks=row_view_checks,
                  baseline_native_mean_ns=float(bridge['cost'].mean()), new_native_mean_ns=float(costs.mean()),
                  baseline_native_p99_ns=float(np.percentile(bridge['cost'], 99)), new_native_p99_ns=float(np.percentile(costs, 99)),
                  mean_gain_pct=float(mean_gain), p99_gain_pct=float(p99_gain), actual_chip_work_mean_ns=float(expected_work.max(1).mean()),
                  representative_node_trace_checks=traces, native_provenance=native.provenance,
                  gate_ge2mean_or5tail=bool(mean_gain >= 2 or p99_gain >= 5),
                  selector_cost_ns=0, selector_scope='Zero-time optimistic sensitivity; not measured runtime selection.',
                  next_step='Stop selectorbench if gate fails; otherwise reportbefore same24MiB wholeDown32/partialRDA/staticcontrols/CPUselector. No native-device or novelty claim.',
                  seconds=time.perf_counter()-began)
    for item in protocol['inputs']:
        assert sha(item['path']) == item['sha256']
    dump(OUT / 'report.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['freeze','run']); args = parser.parse_args()
    freeze() if args.action == 'freeze' else run()
