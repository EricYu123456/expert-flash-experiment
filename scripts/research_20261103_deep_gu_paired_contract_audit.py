#!/usr/bin/env python3
"""Independent TRAIN-only row/flow/residency audit; no native replay or weights."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/research-2026-11-02/round2'
SOURCE = BASE / 'deep-gu-paired-replica-screen'
OUT = SOURCE / 'independent-contract'
INC = BASE / 'native-joint-refinement/pilot-v1/deepseek.L0.input4.independent_paired.seed20261106'
TRAIN = BASE / 'w8-format-training/deepseek/decode.routes.npz'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, value):
    Path(p).write_text(json.dumps(value, indent=2) + '\n')


def flow_value(pairs, caps):
    # Edmonds-Karp in SciPy, independent of the screened C++ Dinic code.
    if min(caps) < 0:
        return 0
    ii, jj, vv = [], [], []
    for j, endpoints in enumerate(pairs):
        ii.append(0); jj.append(1 + j); vv.append(1)
        for c in endpoints:
            ii.append(1 + j); jj.append(49 + int(c)); vv.append(1)
    for c, cap in enumerate(caps):
        ii.append(49 + c); jj.append(65); vv.append(int(min(6, cap)))
    graph = csr_matrix((np.asarray(vv, np.int64), (ii, jj)), shape=(66, 66))
    return maximum_flow(graph, 0, 65, method='edmonds_karp').flow_value


def euler_assignment(pairs):
    # Orient an Euler tour of every even-degree component. Repeated edges retain IDs.
    adj = [[] for _ in range(16)]
    for j, (a, b) in enumerate(pairs):
        adj[int(a)].append((int(b), j)); adj[int(b)].append((int(a), j))
    used = np.zeros(48, bool); take = np.full(48, -1, np.int32)
    for start in range(16):
        stack = [start]
        while stack:
            a = stack[-1]
            while adj[a] and used[adj[a][-1][1]]:
                adj[a].pop()
            if not adj[a]:
                stack.pop(); continue
            b, edge = adj[a].pop(); used[edge] = True; take[edge] = b; stack.append(b)
    assert used.all() and np.all(np.bincount(take, minlength=16) == 3)
    return take


def free_bound(base, delta, cap=6):
    # Relax all endpoint restrictions: 48 independent unit packets, each chip <= cap.
    lo, hi = int(max(base)), int(max(base) + 48 * delta)
    while lo < hi:
        mid = (lo + hi) // 2
        capacity = np.minimum(cap, np.maximum(0, (mid - base) // delta)).sum()
        if capacity >= 48:
            hi = mid
        else:
            lo = mid + 1
    return lo


def address_ledger(rows, columns):
    jobs = []
    for e in range(65):
        for p in range(2):
            logical_cursor = 0
            for c in range(16):
                n, width = int(rows[e, p, c]), int(columns[e, p])
                if n:
                    pages = (n * width + 16383) // 16384
                    waves = (pages + 3) // 4
                    assert waves <= 192
                    jobs.append(dict(job=len(jobs), expert=e, phase=p, chip=c,
                                     stored_rows=n, columns=width, useful_bytes=n * width,
                                     pages=pages, waves=waves, original_row_start=logical_cursor))
                logical_cursor += n
    blocks = []
    for c in range(16):
        remaining = []
        for job in sorted((j for j in jobs if j['chip'] == c), key=lambda j: (-j['waves'], j['job'])):
            fits = [b for b, left in enumerate(remaining) if left >= job['waves']]
            if fits:
                b = min(fits, key=lambda b: (remaining[b] - job['waves'], b))
            else:
                b = len(remaining); remaining.append(192)
            start = 192 - remaining[b]
            job.update(block=b, wordline=start, end_wordline=start + job['waves'], first_plane=0)
            remaining[b] -= job['waves']
        blocks.append(len(remaining))
    for c in range(16):
        for b in range(blocks[c]):
            spans = sorted((j['wordline'], j['end_wordline']) for j in jobs if j['chip'] == c and j['block'] == b)
            assert all(a[1] <= z[0] for a, z in zip(spans, spans[1:]))
            assert all(0 <= a < z <= 192 for a, z in spans)
    ledger = dict(jobs=len(jobs), useful_bytes=sum(j['useful_bytes'] for j in jobs),
                  pages=sum(j['pages'] for j in jobs), waves=sum(j['waves'] for j in jobs),
                  chip_blocks=blocks, allocated_lsb_block_bytes=sum(blocks) * 192 * 65536)
    ledger['programmed_page_bytes'] = ledger['pages'] * 16384
    ledger['reserved_wave_bytes'] = ledger['waves'] * 65536
    return ledger, jobs


def main():
    OUT.mkdir(exist_ok=False)
    began = time.perf_counter()
    protocol = json.loads((SOURCE / 'protocol.frozen.json').read_text())
    report = json.loads((SOURCE / 'work_report.json').read_text())
    assert sha(SOURCE / 'protocol.frozen.json') == report['protocol_sha256']
    paths = [Path(__file__).resolve(), SOURCE / 'protocol.frozen.json', SOURCE / 'work_arrays.npz',
             SOURCE / 'work_report.json', INC / 'arrays.npz', INC / 'result.json', TRAIN,
             ROOT / 'paper/aif.txt', ROOT / 'kernel/aif.h',
             ROOT / 'scripts/research_20261103_routed_boundary_views.py',
             BASE / 'routed-boundary-views/report.md']
    inputs = [dict(path=str(p), sha256=sha(p)) for p in paths]
    for x in protocol['inputs']:
        assert sha(x['path']) == x['sha256']
    dump(OUT / 'inputs.frozen.json', dict(inputs=inputs, root_protocol_sha256=report['protocol_sha256'],
                                        scope='TRAIN only; no weights, heldout, fitting, native replay or source changes'))
    with np.load(SOURCE / 'work_arrays.npz') as z:
        arrays = {k: z[k].copy() for k in z.files}
    with np.load(INC / 'arrays.npz') as z:
        original = {k: z[k].copy() for k in ['rows', 'routes', 'jobs', 'offsets', 'input_columns', 'total_rows', 'cost']}
    with np.load(TRAIN) as z:
        assert z['split'].item() == 'train' and z['completed'].all()
        assert np.array_equal(z['routes'][:, :, 0].reshape(512, 6), original['routes'])
    rows, columns, routes = original['rows'], original['input_columns'], original['routes']
    assert rows.shape == (65, 2, 16) and np.array_equal(rows, arrays['rows'])
    assert np.all(np.sort(rows[:64, 0], axis=1) == [160] * 8 + [192] * 8)
    assert np.all(columns[:64] == [2048, 1408]) and np.array_equal(columns[64], [2048, 2816])
    assert all(len(set(r.tolist())) == 6 for r in routes)
    svc = protocol['service']
    transfer = lambda n, rate: (int(n) * 1000000000 + rate - 1) // rate
    occupancy = np.zeros((65, 2, 16), np.int64)
    regenerated = []
    for e in range(65):
        for p in range(2):
            for c, n in enumerate(rows[e, p]):
                if not n:
                    continue
                width = int(columns[e, p]); pages = (int(n) * width + 16383) // 16384; waves = (pages + 3) // 4
                regenerated.append([c, int(n), width, pages, waves])
                emitted = int(n) if svc['output_fringe'] == 0 else (pages * 16384 if svc['output_fringe'] == 1 else waves * 65536) // width
                occupancy[e, p, c] = svc['command_ns'] + transfer(width * svc['input_bytes'] * svc['input_channel_copies'], svc['channel_bytes_per_second']) + waves * svc['ordinary_read_ns'] + transfer(emitted * svc['output_bytes'], svc['channel_bytes_per_second'])
    assert np.array_equal(regenerated, original['jobs'])
    original_work = occupancy[routes].sum((1, 2)) + occupancy[64].sum(0)
    assert np.array_equal(original_work, arrays['original_chip_ns'])
    delta = svc['ordinary_read_ns'] + transfer(32 * svc['output_bytes'], svc['channel_bytes_per_second'])
    assert delta == 28064
    base = original_work - (rows[:64, 0] == 192)[routes].sum(1) * delta
    assert np.array_equal(base, arrays['base_chip_ns'])
    free = np.asarray([free_bound(b, delta) for b in base], np.int64)
    dispatch_base = base + np.arange(16) * svc['command_ns']
    free_dispatch = np.asarray([free_bound(b, delta) for b in dispatch_base], np.int64)
    free_unlimited = np.asarray([free_bound(b, delta, 48) for b in dispatch_base], np.int64)
    stats, all_euler = [], []
    geometry_cases = 0
    row_ids = []
    resident_rows = rows.copy(); resident_rows[:64, 0] = 192
    for layout_index, edges in enumerate(arrays['layouts']):
        # Full per-expert prefix/logical-ID check for all 256 endpoint choices.
        for e in range(64):
            starts = np.r_[0, np.cumsum(rows[e, 0])]
            original_ids = [np.arange(starts[c], starts[c + 1], dtype=np.int32) for c in range(16)]
            physical = [x[:160].copy() for x in original_ids]
            assert np.array_equal(np.sort(edges[e, :, 0]), np.flatnonzero(rows[e, 0] == 192))
            assert np.array_equal(np.sort(edges[e, :, 1]), np.flatnonzero(rows[e, 0] == 160))
            for src, dst in edges[e]:
                packet = original_ids[src][160:192]
                physical[src] = np.r_[physical[src], packet]
                physical[dst] = np.r_[physical[dst], packet]
            counts = np.bincount(np.concatenate(physical), minlength=2816)
            assert (counts == 2).sum() == 256 and (counts == 1).sum() == 2560
            assert all(len(x) == 192 for x in physical)
            if layout_index == 0:
                for c in range(16):
                    packet_src = int(edges[e, np.flatnonzero((edges[e] == c).any(1))[0], 0])
                    row_ids.append(dict(expert=e, chip=c, core_original_ids=physical[c][:160].tolist(),
                                        suffix_original_ids=physical[c][160:].tolist(), suffix_source_chip=packet_src,
                                        scale_metadata='Original expert/gate-up/logical-row F32 scale; identity indexed. No re-quantization.'))
            for mask in range(256):
                lengths = np.full(16, 160, np.int32)
                for j, (src, dst) in enumerate(edges[e]):
                    lengths[dst if mask >> j & 1 else src] = 192
                selected = np.concatenate([physical[c][:lengths[c]] for c in range(16)])
                assert np.array_equal(np.sort(selected), np.arange(2816))
                assert np.array_equal(np.sort(lengths), np.sort(rows[e, 0]))
                assert lengths.sum() == 2816 and (lengths // 32).sum() == 88
                geometry_cases += 1
        euler = []
        for q, route in enumerate(routes):
            pairs = edges[route].reshape(48, 2)
            assert np.all(np.bincount(pairs.ravel(), minlength=16) == 6)
            euler.append(euler_assignment(pairs))
            for label, bb, values, choices in [('', base, arrays[f'layout{layout_index}_work'], arrays[f'layout{layout_index}_assignment']),
                                               ('dispatch_', dispatch_base, arrays[f'layout{layout_index}_dispatch_work'], arrays[f'layout{layout_index}_dispatch_assignment'])]:
                take = choices[q]
                assert np.all((take == pairs[:, 0]) | (take == pairs[:, 1]))
                counts = np.bincount(take, minlength=16)
                bound = int(values[q])
                assert max(bb[q] + counts * delta) == bound
                assert flow_value(pairs, (bound - bb[q]) // delta) == 48
                assert flow_value(pairs, (bound - 1 - bb[q]) // delta) < 48
            ga = arrays[f'layout{layout_index}_balanced_GU_assignment'][q]
            assert np.all(np.bincount(ga, minlength=16) == 3)
            assert np.all((ga == pairs[:, 0]) | (ga == pairs[:, 1]))
        # Static endpoint choices retain the exact original per-expert row multiset.
        bits = arrays[f'layout{layout_index}_static_bits']
        static_rows = rows[:64, 0].copy()
        for e in range(64):
            for j, (src, dst) in enumerate(edges[e]):
                static_rows[e, src] = 160 if bits[e, j] else 192
                static_rows[e, dst] = 192 if bits[e, j] else 160
        assert np.array_equal(np.sort(static_rows, axis=1), np.sort(rows[:64, 0], axis=1))
        static = base + ((static_rows == 192)[routes].sum(1)) * delta
        assert np.array_equal(static.max(1), arrays[f'layout{layout_index}_static_work'])
        value = arrays[f'layout{layout_index}_dispatch_work']
        stats.append(dict(layout=layout_index, independent_flow_cases=1024, exact_feasibility_and_one_ns_below_infeasibility=True,
                          mean_dispatch_work_ns=float(value.mean()), mean_gap_over_endpoint_free_ns=float((value - free_dispatch).mean()),
                          max_gap_over_endpoint_free_ns=int((value - free_dispatch).max())))
        all_euler.append(euler)
    baseline_ledger, _ = address_ledger(rows, columns)
    resident_ledger, addresses = address_ledger(resident_rows, columns)
    assert resident_ledger['useful_bytes'] - baseline_ledger['useful_bytes'] == 33554432
    assert resident_ledger['pages'] - baseline_ledger['pages'] == 2048
    assert resident_ledger['waves'] - baseline_ledger['waves'] == 512
    assert baseline_ledger['useful_bytes'] == int(544.5 * 1048576)
    # Stored GU slots are a multiset, not 3072 independent logical rows.
    for j in addresses:
        if j['expert'] < 64 and j['phase'] == 0:
            j['original_row_start'] = None
            j['logical_row_identity'] = 'See layout0_row_ids.json; core160 + duplicated donor32, 2816 logical rows/expert.'
    dump(OUT / 'layout0_row_ids.json', row_ids)
    dump(OUT / 'resident_addresses.json', dict(placement='Static repack; original source addresses are not preserved.',
                                             ledger=resident_ledger, jobs=addresses))
    np.savez_compressed(OUT / 'independent_arrays.npz', euler_assignment=np.asarray(all_euler),
                        endpoint_free_work=free, endpoint_free_dispatch_work=free_dispatch,
                        endpoint_and_chipcount_free_dispatch_work=free_unlimited)
    results = dict(status='PASS bounded independent contract audit', scipy_version=scipy.__version__,
                   root_protocol_sha256=report['protocol_sha256'], inputs_frozen_sha256=sha(OUT / 'inputs.frozen.json'),
                   flow_algorithm='SciPy Edmonds-Karp; root C++ uses Dinic. Every optimum independently checked feasible and optimum-1ns infeasible.',
                   full_flow_optimality_checks=6144, maxflow_calls=12288, euler_degree3_cases=3072,
                   all_prefix_endpoint_masks_checked=geometry_cases, numeric_scope='Logical row/scale identity only; no coefficient arithmetic or native output-ABI certification.',
                   per_layout=stats, base_occupancy_recomputed=True, static_row_families_and_work_recomputed=True,
                   baseline=baseline_ledger, resident=resident_ledger,
                   extra=dict(useful_bytes=33554432, pages=2048, waves=512,
                              coefficient_percentage=100 * 33554432 / baseline_ledger['useful_bytes'],
                              per_expert_stored_row_slots=3072, per_expert_unique_logical_rows=2816,
                              per_expert_duplicate_row_slots=256),
                   endpoint_free_mean_work_ns=float(free.mean()), endpoint_free_mean_dispatch_work_ns=float(free_dispatch.mean()),
                   endpoint_free_native_mean_LB_ns=float(free_dispatch.mean() + svc['host_submit_ns']),
                   endpoint_free_optimistic_native_gain_upper_pct=float(100 * (1 - (free_dispatch.mean() + svc['host_submit_ns']) / original['cost'].mean())),
                   endpoint_chipcount_relaxation_gap_ns=float((free_dispatch - free_unlimited).mean()),
                   whole_GU_budget=dict(bytes_per_copy=5767168, wave_per_copy=88, complete_copies=5,
                                        complete_copy_bytes=28835840, residual_bytes=4718592, residual_waves=72,
                                        six_copy_bytes=34603008, six_copy_over_budget_bytes=1048576),
                   gate='All six finite layouts below 2% optimistic fixed-executor ceiling. No-go for this bounded screen; not global replication impossibility.',
                   seconds=time.perf_counter() - began)
    for x in inputs:
        assert sha(x['path']) == x['sha256'], x['path']
    dump(OUT / 'audit.json', results)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
