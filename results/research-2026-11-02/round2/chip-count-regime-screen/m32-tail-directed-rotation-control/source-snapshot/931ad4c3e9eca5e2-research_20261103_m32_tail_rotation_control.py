#!/usr/bin/env python3
"""TRAIN-only tail-directed cyclic control; supplementary to frozen primary."""
import argparse
import ctypes
import hashlib
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from research_20261103_topology_placement_replay import CPP, PYTHON, load_python

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/research-2026-11-02/round2/chip-count-regime-screen'
IND = BASE / 'g4-M32-independent-rotation-cp'
COMMON = BASE / 'g4-M32-cyclic-rotation-cp'
FREE = BASE / 'g4-M32-coupled-cp'
OUT = BASE / 'm32-tail-directed-rotation-control'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def stats(x):
    return {'mean_ns': float(np.mean(x)),
            'p99_ns': float(np.quantile(x, .99, method='linear')),
            'max_ns': int(np.max(x))}


def rows(z, labels):
    r = np.empty((65, 2, 32), np.int64)
    r[:64, 0] = z['gu_rotations'][labels[:, 0]]
    r[:64, 1] = z['down_rotations'][labels[:, 1]]
    r[64] = z['shared']
    return r


def freeze():
    OUT.mkdir(exist_ok=False)
    (OUT / 'source-snapshot').mkdir()
    z = load(IND / 'inputs.frozen.npz')
    independent = load(IND / 'candidates.frozen.npz')
    common = load(COMMON / 'candidates.frozen.npz')
    z['independent_labels'] = independent['incumbent2.labels']
    z['common_labels'] = np.repeat(common['incumbent0.labels'][:, None], 2, axis=1)
    assert np.array_equal(rows(z, z['independent_labels']), independent['incumbent2.rows'])
    assert np.array_equal(rows(z, z['common_labels']), common['incumbent0.rows'])
    assert z['routes'].shape == (512, 6)
    assert all(len(set(route)) == 6 for route in z['routes'])
    np.savez_compressed(OUT / 'inputs.frozen.npz', **z)
    inputs = [Path(__file__), CPP, PYTHON,
              ROOT / 'scripts/research_20261103_topology_placement_replay.py',
              IND / 'inputs.frozen.npz', IND / 'candidates.frozen.npz',
              IND / 'protocol.frozen.json', IND / 'native_arrays.npz',
              COMMON / 'candidates.frozen.npz', COMMON / 'native_arrays.npz',
              FREE / 'native_report.json', OUT / 'inputs.frozen.npz',
              BASE / 'm32-prospective-tail-validation/protocol.frozen.json',
              BASE / 'm32-prospective-tail-validation/completion.frozen.json']
    ledger = []
    for path in inputs:
        item = {'path': str(path), 'sha256': sha(path)}
        if path.suffix in ('.py', '.cpp'):
            target = OUT / 'source-snapshot' / (item['sha256'][:16] + '-' + path.name)
            shutil.copy2(path, target)
            item['snapshot'] = str(target)
        ledger.append(item)
    service = json.loads((IND / 'protocol.frozen.json').read_text())['service']
    dump(OUT / 'protocol.frozen.json', {
        'utc': datetime.now(timezone.utc).isoformat(), 'source_sha256': sha(__file__),
        'inputs': ledger, 'service': service,
        'scope': 'Existing TRAIN512 only, no new prompt/capture/routes/cost payload opened. '
                 'Supplementary strong control; original three plans and prospective primary unchanged.',
        'search': {'starts': ['independent_labels', 'common_labels'],
                   'objectives': ['peak', 'top6_work'], 'seed': 20261116,
                   'single_phase_passes': 4, 'joint_phase_passes': 2,
                   'all_labels_per_single': 32, 'all_pairs_per_joint': 1024,
                   'updates': 'Strict lexicographic descent; deterministic per-fit PCG64 permutations. '
                              'All experts visited on every fixed pass, no restart or new fit.',
                   'peak_score': ['count(query NAND peak > 10)',
                                  'sum(max(chip NAND waves-10,0))', 'sum(query max owned work)'],
                   'top6_work_score': ['sum(largest six query max owned work)',
                                      'sum(query max owned work)', 'count(query NAND peak > 10)'],
                   'guard': 'Sum query max owned work <= ceil(1.02 * frozen independent anchor sum). '
                            'This is a work proxy, not native latency or an optimality certificate.',
                   'retention': 'Two anchors plus final map of each of four fixed fits; deduplicate exact labels.'},
        'native': {'inputs': [1, 4], 'priorities': ['GF', 'DF'],
                   'maximum_costs': 6 * 2 * 2 * 512,
                   'python_trace_queries': [0, 255, 511],
                   'selection': 'Supplementary winner minimizes TRAIN input1 GF P99, then mean; '
                                'input4 uses identical plan. Never change primary comparison.'},
        'interpretation': 'Standard cyclic-coordinate control for a possible objective mismatch. '
                          'No new method claim, hardware change, shared refit, copies, or new-data tuning.'})
    print(json.dumps({'protocol_sha256': sha(OUT / 'protocol.frozen.json')}), flush=True)


def inputs():
    p = json.loads((OUT / 'protocol.frozen.json').read_text())
    assert sha(__file__) == p['source_sha256']
    for item in p['inputs']:
        assert sha(item['path']) == item['sha256'], item['path']
    return p, load(OUT / 'inputs.frozen.npz')


def fit():
    p, z = inputs()
    assert not (OUT / 'fit_report.json').exists()
    backend = load_python()
    service = backend.Service(**p['service'])
    anchor = backend.lower_rows(*rows(z, z['independent_labels']).transpose(1, 0, 2),
                                1536, [512] * 64 + [1024], service)
    occupancy = backend.occupancy_table(anchor, service)['chip_ns']
    shared = occupancy[64].sum(0)
    sw = np.zeros(32, np.int64)
    for phase in range(2):
        for chip, nr, nc, pages, waves in anchor['jobs'][slice(*anchor['offsets'][64, phase])]:
            sw[chip] += waves
    G, D = (z['gu_rotations'] > 0).astype(np.int64), (z['down_rotations'] > 0).astype(np.int64)
    routes = z['routes']
    def full(labels):
        g, d = G[labels[:, 0]][routes].sum(1), D[labels[:, 1]][routes].sum(1)
        return g, d, 29102 * g + 28762 * d + shared, g + d + sw
    _, _, aw, _ = full(z['independent_labels'])
    assert np.array_equal(aw, occupancy[routes].sum((1, 2)) + shared)
    cap = (int(aw.max(1).sum()) * 102 + 99) // 100
    candidates = {name: z[name + '_labels'].copy() for name in ('independent', 'common')}
    records = []
    begin = time.monotonic()
    pair_labels = np.asarray([(a, b) for a in range(32) for b in range(32)], np.int64)
    for start_index, start_name in enumerate(('independent', 'common')):
        for objective_index, objective in enumerate(('peak', 'top6_work')):
            name = start_name + '_' + objective
            labels = z[start_name + '_labels'].copy()
            rng = np.random.default_rng(20261116 + 10 * start_index + objective_index)
            counts_g, counts_d, work, waves = full(labels)
            wmax, nmax = work.max(1), waves.max(1)
            excess = np.maximum(waves - 10, 0).sum(1)
            def scores(cw, cn, ce):
                total = cw.sum(1)
                over = (cn > 10).sum(1)
                if objective == 'peak':
                    return [over, ce.sum(1), total]
                return [np.partition(cw, 506, axis=1)[:, 506:].sum(1), total, over]
            initial_score = [int(x[0]) for x in scores(wmax[None], nmax[None], excess[None])]
            passes = []
            for pass_index in range(6):
                joint = pass_index >= 4
                changes, proposals = 0, 0
                for expert in rng.permutation(64):
                    query_ids = np.flatnonzero(np.any(routes == expert, axis=1))
                    if not len(query_ids):
                        continue
                    phase_order = [None] if joint else rng.permutation(2)
                    for phase in phase_order:
                        if joint:
                            choices = pair_labels
                        else:
                            choices = np.repeat(labels[expert][None], 32, axis=0)
                            choices[:, phase] = np.arange(32)
                        cg = counts_g[query_ids][None] - G[labels[expert, 0]] + G[choices[:, 0], None]
                        cd = counts_d[query_ids][None] - D[labels[expert, 1]] + D[choices[:, 1], None]
                        sub_w = 29102 * cg + 28762 * cd + shared
                        sub_n = cg + cd + sw
                        cw = np.broadcast_to(wmax, (len(choices), 512)).copy()
                        cn = np.broadcast_to(nmax, (len(choices), 512)).copy()
                        ce = np.broadcast_to(excess, (len(choices), 512)).copy()
                        cw[:, query_ids] = sub_w.max(2)
                        cn[:, query_ids] = sub_n.max(2)
                        ce[:, query_ids] = np.maximum(sub_n - 10, 0).sum(2)
                        score_arrays = scores(cw, cn, ce)
                        feasible = cw.sum(1) <= cap
                        score_arrays = [np.where(feasible, x, np.iinfo(np.int64).max) for x in score_arrays]
                        best = int(np.lexsort(tuple(score_arrays[::-1]))[0])
                        old_score = tuple(int(x[0]) for x in scores(wmax[None], nmax[None], excess[None]))
                        new_score = tuple(int(x[best]) for x in score_arrays)
                        proposals += len(choices)
                        if new_score < old_score:
                            labels[expert] = choices[best]
                            counts_g[query_ids], counts_d[query_ids] = cg[best], cd[best]
                            work[query_ids], waves[query_ids] = sub_w[best], sub_n[best]
                            wmax, nmax, excess = cw[best], cn[best], ce[best]
                            changes += 1
                assert np.array_equal(work, full(labels)[2])
                record = {'pass': pass_index, 'joint': joint, 'changes': changes, 'proposals': proposals,
                          'score': [int(x[0]) for x in scores(wmax[None], nmax[None], excess[None])],
                          'work': stats(wmax), 'peak10_queries': int((nmax <= 10).sum())}
                passes.append(record)
                print(json.dumps({'fit': name, **record}), flush=True)
            candidates[name] = labels.copy()
            records.append({'id': name, 'initial_score': initial_score, 'passes': passes})
    saved = {'routes': routes, 'shared_work': shared, 'shared_waves': sw}
    retained, seen, alias = [], {}, {}
    for ident, labels in candidates.items():
        key = labels.tobytes()
        if key in seen:
            alias[ident] = seen[key]
            continue
        seen[key] = ident
        _, _, work, waves = full(labels)
        saved[ident + '.labels'], saved[ident + '.rows'] = labels, rows(z, labels)
        saved[ident + '.work'] = work.max(1)
        retained.append({'id': ident, 'work': stats(work.max(1)),
                         'peak10_queries': int((waves.max(1) <= 10).sum())})
    np.savez_compressed(OUT / 'candidates.frozen.npz', **saved)
    dump(OUT / 'fit_report.json', {'utc': datetime.now(timezone.utc).isoformat(),
         'protocol_sha256': sha(OUT / 'protocol.frozen.json'), 'fit_seconds': time.monotonic() - begin,
         'guard_sum_work_ns': cap, 'fits': records, 'retained': retained, 'aliases': alias,
         'scope': 'TRAIN only; no optimality or fresh-data claim.'})


def native():
    p, z = inputs()
    assert not (OUT / 'native_report.json').exists()
    fit = json.loads((OUT / 'fit_report.json').read_text())
    c = load(OUT / 'candidates.frozen.npz')
    ids = [x['id'] for x in fit['retained']]
    dump(OUT / 'native_protocol.frozen.json', {
        'utc': datetime.now(timezone.utc).isoformat(), 'source_sha256': sha(__file__),
        'candidates_sha256': sha(OUT / 'candidates.frozen.npz'),
        'fit_report_sha256': sha(OUT / 'fit_report.json'), 'ids': ids,
        'input_copies': [1, 4], 'priorities': ['GF', 'DF'], 'queries': 512,
        'trace_queries': [0, 255, 511], 'no_new_routes_opened': True})
    directory = OUT / 'native'
    directory.mkdir()
    binary = directory / 'native.so'
    subprocess.run(['c++', '-std=c++17', '-O3', '-shared', '-fPIC', str(CPP), '-o', str(binary)], check=True)
    backend = load_python()
    lib = ctypes.CDLL(str(binary))
    fn = lib.eval_ordinary_read
    i32 = np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
    i64 = np.ctypeslib.ndpointer(dtype=np.int64, flags='C_CONTIGUOUS')
    fn.argtypes = [i32, i32, i32, i32, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                   ctypes.c_int, ctypes.c_int, i64, i64, i64, ctypes.c_void_p, i64]
    route = np.ascontiguousarray(z['routes'], np.int32)
    def evaluate(plan, r, service, trace=False):
        cost, bound, nlog = np.zeros(len(r), np.int64), np.zeros(len(r), np.int64), np.zeros(1, np.int64)
        log = np.zeros((6000 if trace else 1, 8), np.int64)
        fn(plan['offsets'], plan['jobs'], plan['input_columns'], r, 64, 1, len(plan['jobs']), len(r), 6,
           service.array(), cost, bound, ctypes.c_void_p(log.ctypes.data) if trace else None, nlog)
        assert np.all(cost >= bound)
        return cost, bound, log[:nlog[0]]
    records, arrays, traces, physical = [], {}, [], {}
    old = {'independent': load(IND / 'native_arrays.npz'), 'common': load(COMMON / 'native_arrays.npz')}
    for ident in ids:
        r = c[ident + '.rows']
        base_service = backend.Service(**p['service'])
        plan = backend.lower_rows(r[:, 0], r[:, 1], 1536, [512] * 64 + [1024], base_service)
        backend.validate_plan(plan, 65, base_service)
        ledger = backend.physical_addresses(plan, base_service)
        address = ledger.pop('addresses')
        assert ledger['chip_blocks'] == [1] * 32
        physical[ident] = ledger
        np.savez_compressed(directory / (ident + '.plan.npz'), rows=r, labels=c[ident + '.labels'],
             offsets=plan['offsets'], jobs=plan['jobs'], input_columns=plan['input_columns'],
             total_rows=plan['total_rows'],
             addresses=np.asarray([[a[k] for k in ('chip', 'block', 'wordline', 'reserved_wordlines',
                                                 'pages', 'first_plane')] for a in address], np.int32))
        for copies in (1, 4):
            for priority in (0, 1):
                cfg = dict(p['service'], input_channel_copies=copies, down_first=priority)
                service = backend.Service(**cfg)
                label = 'DF' if priority else 'GF'
                cost, bound, _ = evaluate(plan, route, service)
                key = f'input{copies}.{ident}.{label}'
                arrays[key + '.cost'], arrays[key + '.bound'] = cost, bound
                for q in (0, 255, 511):
                    tc, tb, log = evaluate(plan, route[q:q + 1], service, True)
                    py = backend.ReferenceGraph.build(plan, np.r_[route[q], 64], service).run()
                    assert cost[q] == tc[0] == py['cost'] and np.array_equal(log, py['trace'])
                    arrays[key + f'.q{q}trace'] = log
                    traces.append({'id': key, 'q': q, 'nodes': len(log), 'all_nodes_exact': True})
                if copies == 1 and ident in old:
                    old_id = 'incumbent2' if ident == 'independent' else 'incumbent0'
                    assert np.array_equal(cost, old[ident][old_id + '.' + label + '.cost'])
                records.append({'id': ident, 'input_copies': copies, 'priority': label, 'cost': stats(cost)})
    best = min((r for r in records if r['input_copies'] == 1 and r['priority'] == 'GF'),
               key=lambda r: (r['cost']['p99_ns'], r['cost']['mean_ns']))
    free = json.loads((FREE / 'native_report.json').read_text())['best_coupled_input1']
    report = {'utc': datetime.now(timezone.utc).isoformat(),
              'protocol_sha256': sha(OUT / 'protocol.frozen.json'), 'records': records,
              'selected_train_tail_control': best, 'frozen_free': free,
              'free_vs_tail_control_mean_gain_pct': 100 * (1 - free['cost']['mean_ns'] / best['cost']['mean_ns']),
              'free_vs_tail_control_p99_gain_pct': 100 * (1 - free['cost']['p99_ns'] / best['cost']['p99_ns']),
              'native_costs': len(ids) * 2048, 'traces': traces, 'physical': physical,
              'scope': 'Supplementary TRAIN-only control; prospective original primary unchanged. '
                       'Finite work-proxy coordinate descent, not optimality; no fresh-data claim.'}
    np.savez_compressed(OUT / 'native_arrays.npz', **arrays)
    dump(OUT / 'native_report.json', report)
    dump(OUT / 'completion.json', {'utc': datetime.now(timezone.utc).isoformat(),
         'status': 'TRAIN_supplementary_control_complete_no_new_payload',
         'files': [{'path': str(f), 'sha256': sha(f)} for f in sorted(OUT.rglob('*')) if f.is_file()]})
    print(json.dumps({k: report[k] for k in ('selected_train_tail_control',
          'free_vs_tail_control_mean_gain_pct', 'free_vs_tail_control_p99_gain_pct', 'native_costs')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('freeze', 'fit', 'native'))
    args = parser.parse_args()
    {'freeze': freeze, 'fit': fit, 'native': native}[args.mode]()
