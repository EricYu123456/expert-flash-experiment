#!/usr/bin/env python3
"""Independent complete-bipartite maxflow certificate of the free-A8 relaxation."""
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
SOURCE = BASE / 'g4-elastic-prefix-screen/free-span-bound'
OUT = SOURCE / 'independent'
INC = BASE / 'compact-coupled-solver/continuation/native/0107.npz'
TRAIN = BASE / 'w8-format-training/granite4tiny/decode.routes.npz'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, x):
    Path(p).write_text(json.dumps(x, indent=2) + '\n')


def flow(n, caps):
    # source->n experts(cap8); each expert->16chips(cap1); chip->sink(cap_c).
    # This explicitly enforces eight DISTINCT chips for every A expert.
    sink = n + 17
    graph = np.zeros((sink + 1, sink + 1), np.int64)
    graph[0, 1:n+1] = 8
    graph[1:n+1, n+1:n+17] = 1
    graph[n+1:n+17, sink] = caps
    result = maximum_flow(csr_matrix(graph), 0, sink, method='edmonds_karp')
    return int(result.flow_value), result.flow[1:n+1, n+1:n+17].toarray()


def main():
    OUT.mkdir(exist_ok=False)
    began = time.perf_counter()
    protocol = json.loads((SOURCE / 'protocol.frozen.json').read_text())
    report = json.loads((SOURCE / 'report.json').read_text())
    assert sha(SOURCE / 'protocol.frozen.json') == report['protocol_sha256']
    paths = [Path(__file__).resolve(), SOURCE / 'protocol.frozen.json', SOURCE / 'report.json',
             SOURCE / 'arrays.npz', INC, TRAIN,
             ROOT / 'scripts/research_20261103_ordinary_read_native.cpp']
    frozen = dict(scope='All32768 original TRAIN mask/query values; independent Edmonds-Karp. No fitting, weights, native replay or heldout.',
                  inputs=[dict(path=str(p), sha256=sha(p)) for p in paths])
    dump(OUT / 'protocol.frozen.json', frozen)
    for item in protocol['inputs']:
        assert sha(item['path']) == item['sha256']
    with np.load(SOURCE / 'arrays.npz') as z:
        arrays = {k: z[k].copy() for k in z.files}
    with np.load(INC) as z:
        rows = z['rows'].copy(); columns = z['input_columns'].copy(); routes = z['routes'].copy(); native = z['cost'].copy()
    with np.load(TRAIN) as z:
        assert z['split'].item() == 'train' and z['completed'].all()
        assert np.array_equal(routes, z['routes'][:, :, 0].reshape(512, 6))
    assert np.array_equal(rows, arrays['rows']) and np.array_equal(routes, arrays['routes'])
    assert np.array_equal(native, arrays['baseline_native'])
    # Independent ordinary chip-owned service; no import of occupancy_table.
    table = np.zeros((65, 2, 16), np.int64)
    for e in range(65):
        for phase in range(2):
            for c, r in enumerate(rows[e, phase]):
                if not r:
                    continue
                width = int(columns[e, phase]); pages = (int(r) * width + 16383) // 16384
                waves = (pages + 3) // 4; emitted = pages * 16384 // width
                table[e, phase, c] = 250 + (width + 1) // 2 + waves * 28000 + emitted * 2
    fixed = table[routes, 1].sum(1) + table[64].sum(0)
    gu = table[routes, 0]
    job = 250 + 768 + 3 * 28000 + 128 * 2
    assert job == 85274 == report['A_job_ns']
    witnesses, counts, calls = [], {str(n): 0 for n in range(7)}, 0
    checked_values = np.zeros((64, 512), np.int64)
    for mask in range(64):
        n = mask.bit_count(); keep = [k for k in range(6) if not (mask >> k & 1)]
        base = fixed + gu[:, keep].sum(1)
        assert np.array_equal(base, arrays['allmask_base'][mask])
        for q in range(512):
            target = int(arrays['allmask_work'][mask, q]); floor = int(max(base[q]))
            assert target >= floor
            counts[str(n)] += 1
            if n == 0:
                assert target == floor and target - 1 < floor
                assert not arrays['allmask_chip_capacities'][mask, q].any()
                continue
            caps = np.minimum(n, np.maximum(0, (target - base[q]) // job))
            assert np.array_equal(caps, arrays['allmask_chip_capacities'][mask, q])
            value, assigned = flow(n, caps); calls += 1
            assert value == 8 * n
            assert np.all((assigned == 0) | (assigned == 1))
            assert np.all(assigned.sum(1) == 8) and np.all(assigned.sum(0) <= caps)
            previous_caps = np.minimum(n, np.maximum(0, (target - 1 - base[q]) // job))
            previous_value, _ = flow(n, previous_caps); calls += 1
            assert target - 1 < floor or previous_value < 8 * n
            for k in range(1, n + 1):
                assert np.minimum(caps, k).sum() >= 8 * k
            if q == 0 and mask == (1 << n) - 1:
                witnesses.append(dict(mask=mask, query=q, nA=n, base=base[q].tolist(), optimum=target,
                                      chip_capacities=caps.tolist(), expert_chip_incidence=assigned.tolist(),
                                      previous_ns=target-1, previous_maxflow=previous_value))
            checked_values[mask, q] = target
    checked_values[0] = arrays['allmask_work'][0]
    best = checked_values.min(0) + 2000
    assert np.array_equal(best, arrays['best_LB'])
    assert np.array_equal(checked_values.argmin(0), arrays['winner'])
    assert np.all(best <= native)
    quantiles = []
    for percentile in (0, 1, 50, 90, 95, 99, 100):
        lb, baseline = np.percentile(best, percentile), np.percentile(native, percentile)
        assert lb <= baseline
        quantiles.append(dict(percentile=percentile, relaxed_LB_ns=float(lb), baseline_native_ns=float(baseline)))
    assert best.mean() == report['relaxed_native_mean_LB_ns']
    assert np.percentile(best, 99) == report['relaxed_native_p99_LB_ns']
    mean_upper = 100 * (1 - best.mean() / native.mean())
    p99_upper = 100 * (1 - np.percentile(best, 99) / np.percentile(native, 99))
    assert mean_upper == report['mean_improvement_upper_pct']
    assert p99_upper == report['p99_improvement_upper_pct']
    for item in frozen['inputs'] + protocol['inputs']:
        assert sha(item['path']) == item['sha256']
    dump(OUT / 'flow_witnesses.json', witnesses)
    result = dict(status='PASS full independent relaxed-flow audit', scipy_version=scipy.__version__,
                  root_protocol_sha256=report['protocol_sha256'], independent_protocol_sha256=sha(OUT / 'protocol.frozen.json'),
                  allmask_query_values_checked=32768, nA0_exact_fixed_work_cases=512,
                  distinct_chip_flow_optima_checked=32256, maxflow_calls=calls,
                  every_nonzero_A_optimum_and_T_minus_1_checked=True, cases_per_nA=counts,
                  base_service_independently_recomputed=True, A_job_ns=job,
                  relaxed_native_mean_LB_ns=float(best.mean()), relaxed_native_p99_LB_ns=float(np.percentile(best, 99)),
                  mean_improvement_upper_pct=float(mean_upper), p99_improvement_upper_pct=float(p99_upper),
                  quantile_monotonicity=quantiles,
                  caveat='Optimistic query-dependent complete-bipartite placement relaxation with unbounded copies/addresses. No attainable layout, native gain or new hardware claim.',
                  seconds=time.perf_counter()-began)
    dump(OUT / 'audit.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
