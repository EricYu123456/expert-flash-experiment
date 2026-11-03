#!/usr/bin/env python3
"""Integral whole-packet necessary bound; source packet's506ns overhead waived."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/research-2026-11-02/round2/g4-down-paired-merge-screen'
OUT = SOURCE / 'unitflow-bound'
SELF = Path(__file__).resolve()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, x):
    Path(p).write_text(json.dumps(x, indent=2) + '\n')


def freeze():
    OUT.mkdir(exist_ok=False)
    paths = [SELF, SOURCE / 'protocol.frozen.json', SOURCE / 'report.json', SOURCE / 'arrays.npz',
             ROOT / 'scripts/research_20261103_ordinary_read_native.cpp']
    dump(OUT / 'protocol.frozen.json', dict(inputs=[dict(path=str(p), sha256=sha(p)) for p in paths],
        scope='All32 frozen routeblind graphs x512TRAIN. No fitting, native, weights, heldout or old source/output edits.',
        relaxation='Each pair keeps dest128 guaranteed28762ns. Original source128 packet mayexecute at either endpoint withuniform28256ns; waive extra506ns ifpacketat source. Exactly36integral packets/query. Necessarybound only, notreal work.',
        executor='GUunchanged andfirstexpert16positive chips in originalcanonicalchiporder. Initialcommandsstart host2000+c*250before any28usreadcomplete; addc*250tofixedbase. Fixedexecutoronly.',
        algorithm='Explicit edge->two endpoint maxflow, integral chip capacities floor((T-base_c)/28256), clippedto graphdegree. Binarysearch exact base_c+k*28256 levels. CertifyT feasible andT-1infeasible.',
        gate='If all32selected graphs have optimisticnative mean<2% andP99<5%, sealfinite no-go withoutnative/CP/heavycontrols. No globalreplicationimpossibility claim.'))
    print(sha(OUT / 'protocol.frozen.json'))


def run():
    protocol = json.loads((OUT / 'protocol.frozen.json').read_text())
    for item in protocol['inputs']:
        assert sha(item['path']) == item['sha256']
    assert not (OUT / 'report.json').exists()
    began = time.perf_counter()
    with np.load(SOURCE / 'arrays.npz') as z:
        arrays = {k: z[k].copy() for k in z.files}
    original = json.loads((SOURCE / 'report.json').read_text())
    base, routes, layouts, native = [arrays[k] for k in ['base', 'routes', 'layouts', 'baseline_native']]
    values = np.zeros((32, 512), np.int64); assignments = np.zeros((32, 512, 36), np.int32)
    fixed_bases = np.zeros((32, 512, 16), np.int64)
    records, calls = [], 0
    order = [original['best_feasible_layout'], original['best_LP_layout']]
    order = list(dict.fromkeys(order + list(range(32))))
    for ordinal, layout in enumerate(order):
        pairs_all = layouts[layout][routes].reshape(512, 36, 2)
        for q in range(512):
            pairs = pairs_all[q]; degree = np.bincount(pairs.ravel(), minlength=16)
            assert max(degree) <= 6
            fixed = base[q] - np.bincount(pairs[:, 0], minlength=16) * 28762 + np.arange(16) * 250
            fixed_bases[layout, q] = fixed
            graph = np.zeros((54, 54), np.int64)
            graph[0, 1:37] = 1
            for j, (src, dest) in enumerate(pairs):
                graph[1+j, 37+src] = 1; graph[1+j, 37+dest] = 1
            def solve(target, output=False):
                nonlocal calls
                if target < max(fixed):
                    return False
                caps = np.minimum(degree, (target - fixed) // 28256)
                graph[37:53, 53] = caps
                answer = maximum_flow(csr_matrix(graph), 0, 53, method='edmonds_karp'); calls += 1
                if answer.flow_value != 36:
                    return False
                if output:
                    incidence = answer.flow[1:37, 37:53].toarray()
                    assert np.all(incidence.sum(1) == 1) and np.all(incidence.sum(0) <= caps)
                    take = incidence.argmax(1)
                    assert np.all((take == pairs[:, 0]) | (take == pairs[:, 1]))
                    assignments[layout, q] = take
                return True
            levels = np.unique(np.concatenate([fixed[c] + np.arange(degree[c] + 1) * 28256 for c in range(16)]))
            lo, hi = 0, len(levels) - 1
            assert solve(int(levels[hi]))
            while lo < hi:
                mid = (lo + hi) // 2
                if solve(int(levels[mid])):
                    hi = mid
                else:
                    lo = mid + 1
            bound = int(levels[lo]); assert solve(bound, True) and not solve(bound - 1)
            values[layout, q] = bound
            counts = np.bincount(assignments[layout, q], minlength=16)
            assert max(fixed + counts * 28256) == bound
        LB = values[layout] + 2000
        rec = dict(layout=layout, mean_work_plus_dispatch_LB_ns=float(values[layout].mean()),
                   mean_native_LB_ns=float(LB.mean()), p99_native_LB_ns=float(np.percentile(LB, 99)),
                   optimistic_native_mean_gain_upper_pct=float(100 * (1 - LB.mean() / native.mean())),
                   optimistic_native_p99_gain_upper_pct=float(100 * (1 - np.percentile(LB, 99) / np.percentile(native, 99))))
        records.append(rec)
        print(json.dumps(dict(done=ordinal+1, record=rec)), flush=True)
    report = dict(status='Exact uniform integral packet relaxation; not actual native work', protocol_sha256=sha(OUT / 'protocol.frozen.json'),
                  scipy_version=scipy.__version__, layouts=32, queries_per_layout=512, exact_optima_checked=16384,
                  maxflow_calls=calls, every_optimum_T_feasible_and_T_minus_1_infeasible=True,
                  packet_relaxed_ns=28256, source_packet_actual_ns=28762, waived_source_packet_ns=506,
                  baseline_native_mean_ns=float(native.mean()), baseline_native_p99_ns=float(np.percentile(native, 99)),
                  records=sorted(records, key=lambda r:r['layout']),
                  max_optimistic_native_mean_gain_upper_pct=max(r['optimistic_native_mean_gain_upper_pct'] for r in records),
                  max_optimistic_native_p99_gain_upper_pct=max(r['optimistic_native_p99_gain_upper_pct'] for r in records),
                  all_finite_layouts_below_gate=all(r['optimistic_native_mean_gain_upper_pct'] < 2 and r['optimistic_native_p99_gain_upper_pct'] < 5 for r in records),
                  limitations='Fixed32graphs/currentexecutor only. Waived source command/input cost means returned assignments are not claimed actual work. No new hardware, native, weights, fitting, heldout or globalimpossibility.',
                  seconds=time.perf_counter()-began)
    np.savez_compressed(OUT / 'arrays.npz', uniform_packet_bound=values, assignments=assignments, fixed_base_with_dispatch=fixed_bases)
    for item in protocol['inputs']:
        assert sha(item['path']) == item['sha256']
    dump(OUT / 'report.json', report)
    print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['freeze','run']); args = parser.parse_args()
    freeze() if args.action == 'freeze' else run()
