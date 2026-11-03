#!/usr/bin/env python3
"""Matched bounded ordinary cyclic-rotation CP control, never a new algorithm."""
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
import ortools
from ortools.sat.python import cp_model
from research_20261103_topology_placement_replay import CPP, PYTHON, load_python

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/research-2026-11-02/round2/chip-count-regime-screen'
STAGE = BASE / 'stage-b'
FREE = BASE / 'g4-M32-coupled-cp'
OUT = BASE / 'g4-M32-cyclic-rotation-cp'
SOURCE = STAGE / 'granite4tiny.M32.minimum_depth.plan.npz'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def load(path):
    with np.load(path) as z:
        return {key: z[key] for key in z.files}


def stats(values):
    return {'mean_ns': float(np.mean(values)), 'p99_ns': float(np.percentile(values, 99))}


def template_rows(rotations, labels, shared):
    result = np.concatenate([rotations[np.asarray(labels, np.int64)], shared[None]], axis=0)
    assert result.shape == (65, 2, 32)
    assert np.all(result[:64, 0].sum(1) == 1024)
    assert np.all(result[:64, 1].sum(1) == 1536)
    return result


def freeze():
    OUT.mkdir(exist_ok=False)
    (OUT / 'source-snapshot').mkdir()
    z = load(SOURCE)
    rotations = np.asarray([np.roll(z['rows'][0], label, axis=1) for label in range(32)])
    labels = []
    for rows in z['rows'][:64]:
        found = np.flatnonzero(np.all(rotations == rows, axis=(1, 2)))
        assert len(found) == 1
        labels.append(found[0])
    labels = np.asarray(labels, np.int64)
    assert np.array_equal(template_rows(rotations, labels, z['rows'][64]), z['rows'])
    X = (rotations[:, 0] > 0) & (rotations[:, 1] > 0)
    O = rotations[:, 0] == 0
    assert np.all(X.sum(1) == 5) and np.all(O.sum(1) == 7) and not np.any(X & O)
    np.savez_compressed(OUT / 'inputs.frozen.npz', rotations=rotations, labels=labels,
                        shared=z['rows'][64], routes=z['routes'], X=X, O=O)
    files = [Path(__file__), SOURCE, CPP, PYTHON,
             ROOT / 'scripts/research_20261103_topology_placement_replay.py',
             STAGE / 'protocol.frozen.json', STAGE / 'report.json', STAGE / 'cost_arrays.npz',
             FREE / 'protocol.frozen.json', FREE / 'native_report.json', FREE / 'native_arrays.npz',
             FREE / 'candidates.frozen.npz', OUT / 'inputs.frozen.npz']
    inputs = []
    for path in files:
        item = {'path': str(path), 'sha256': sha(path)}
        if path.suffix in ('.py', '.cpp'):
            snap = OUT / 'source-snapshot' / (item['sha256'][:16] + '-' + path.name)
            shutil.copy2(path, snap)
            item['snapshot'] = str(snap)
        inputs.append(item)
    service = dict(json.loads((STAGE / 'protocol.frozen.json').read_text())['service_base'])
    service.update(chips=32, channels=8, input_channel_copies=1, down_first=0)
    protocol = dict(status='frozen_before_cyclic_label_fit', utc=datetime.now(timezone.utc).isoformat(),
                    source_sha256=sha(__file__), inputs=inputs, service=service,
                    model='G4 actualW8 L0 TRAIN512, no heldout',
                    space='For each64 experts choose oneof32 common GU+Down rotations of exactStageB expert0 rowtemplate. All originalStageB rows inthis orbit, labels=e mod32. GU40row rotates too. Frozen shared unchanged.',
                    work='Exact table 29102+28762 X(label,chip)-340 O(label,chip) plus actualshared. No newgeometry, rowcount/perphase multisets equal free-mask experiment, but cyclic orbit strictlyrestricted.',
                    solver=dict(version=ortools.__version__, workers=8, seed=20261114,
                                peak_seconds=30, mean_seconds=90, bools=2048,
                                peak='PureNANDpeak<=10 feasibility, no incumbentmean constraint.',
                                mean='Nopeakconstraint; minimize sum512T, global objectivecap at originalStageB input1work. Exactrows and eachlabel hint fromsource.'),
                    retention='Originalsource+atmost7distinct improving/peak labelmaps; topactualworkpool. Originalpeakmap maybesuperseded; exactpeak verifiedafterfit for everyretainedmap. No free-maskincumbent relabeled ascyclic.',
                    native='Atmost8maps xGF/DF x512 <=8192; existing unchangedCPP. Sourcevectorbridge all512. q0PythonGraph permap/priority. Noinput4 fit/replay inthisstage.',
                    controls='Report source, strongestStageBfinite, and previouslyfitted free-mask coupled. BothCP use same30+90s/workers8/seed; model/space sizes differ, no optimality claim absentOPTIMAL.',
                    scope='Standardcyclic-label CP strongcontrol, not newmethod or breakthrough. 32PE hardware-scale sensitivity, not actualAiF measured hardware. No newsharedfit/replicas/quantization/ABI/heldout/alllayers.')
    dump(OUT / 'protocol.frozen.json', protocol)
    print(json.dumps(dict(protocol_sha256=sha(OUT / 'protocol.frozen.json'), source_sha256=sha(__file__))))


def inputs():
    protocol = json.loads((OUT / 'protocol.frozen.json').read_text())
    assert sha(__file__) == protocol['source_sha256']
    for item in protocol['inputs']:
        assert sha(item['path']) == item['sha256'], item['path']
    return protocol, load(OUT / 'inputs.frozen.npz')


def fit():
    protocol, z = inputs()
    assert not (OUT / 'fit_report.json').exists()
    backend = load_python()
    service = backend.Service(**protocol['service'])
    original_rows = template_rows(z['rotations'], z['labels'], z['shared'])
    plan = backend.lower_rows(original_rows[:, 0], original_rows[:, 1], 1536, [512] * 64 + [1024], service)
    table = backend.occupancy_table(plan, service)['chip_ns']
    shared_work = table[64].sum(0)
    shared_waves = np.zeros(32, np.int64)
    for phase in range(2):
        for chip, nr, nc, pages, waves in plan['jobs'][slice(*plan['offsets'][64, phase])]:
            shared_waves[chip] += waves
    assert shared_waves.sum() == 73
    routes, X, O = z['routes'], z['X'], z['O']
    def actual(labels):
        return (6 * 29102 + shared_work + 28762 * X[labels][routes].sum(1)
                - 340 * O[labels][routes].sum(1))
    source_load = actual(z['labels'])
    assert np.array_equal(source_load, table[routes].sum((1, 2)) + shared_work)
    source_score = int(source_load.max(1).sum())
    pool, events = {}, []
    best_score, start = source_score, time.monotonic()
    def retain(labels, stage, force=False, objective=None):
        nonlocal best_score
        labels = np.asarray(labels, np.int64)
        loads = actual(labels)
        score = int(loads.max(1).sum())
        improved = score < best_score
        best_score = min(best_score, score)
        events.append(dict(stage=stage, actual_score_ns=score, improved=improved,
                           solver_objective=objective, elapsed_seconds=time.monotonic() - start))
        key = labels.tobytes()
        if score < source_score or force:
            pool[key] = dict(labels=labels.copy(), work=loads.max(1), score=score,
                             stage=stage, forced=force)
            chosen = sorted(pool, key=lambda k: pool[k]['score'])[:7]
            if force and key not in chosen:
                chosen = chosen[:6] + [key]
            for old in list(pool):
                if old not in chosen:
                    del pool[old]
    def build(peak, initial):
        model = cp_model.CpModel()
        Y = [[model.NewBoolVar(f'y{e}_{label}') for label in range(32)] for e in range(64)]
        for e in range(64):
            model.AddExactlyOne(Y[e])
            for label in range(32):
                model.AddHint(Y[e][label], int(initial[e] == label))
        xc = [[sum(Y[e][int(v)] for v in np.flatnonzero(X[:, chip])) for chip in range(32)] for e in range(64)]
        oc = [[sum(Y[e][int(v)] for v in np.flatnonzero(O[:, chip])) for chip in range(32)] for e in range(64)]
        high = int(6 * 29102 + shared_work.max() + 6 * 28762)
        T = [model.NewIntVar(280000, high, f'T{q}') for q in range(512)]
        initial_work = actual(initial).max(1)
        for q, route in enumerate(routes):
            for chip in range(32):
                xs = sum(xc[int(e)][chip] for e in route)
                os = sum(oc[int(e)][chip] for e in route)
                model.Add(6 * 29102 + int(shared_work[chip]) + 28762 * xs - 340 * os <= T[q])
                if peak:
                    model.Add(6 + int(shared_waves[chip]) + xs <= 10)
            model.AddHint(T[q], int(initial_work[q]))
        if not peak:
            model.Add(sum(T) <= source_score)
            model.Minimize(sum(T))
        return model, Y
    initial = z['labels'].copy()
    stages = []
    for peak, limit in ((True, 30), (False, 90)):
        before_build = time.monotonic()
        model, Y = build(peak, initial)
        build_seconds = time.monotonic() - before_build
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = limit
        solver.parameters.num_search_workers = 8
        solver.parameters.random_seed = protocol['solver']['seed']
        def decode(value):
            labels = np.asarray([next(j for j, var in enumerate(row) if value(var)) for row in Y], np.int64)
            return labels
        class Callback(cp_model.CpSolverSolutionCallback):
            def on_solution_callback(self):
                retain(decode(self.Value), 'peak' if peak else 'mean', force=peak,
                       objective=float(self.ObjectiveValue()))
        status = solver.Solve(model, Callback())
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            labels = decode(solver.Value)
            retain(labels, 'peak_final' if peak else 'mean_final', force=peak,
                   objective=float(solver.ObjectiveValue()))
            if peak:
                initial = labels
        stage = dict(stage='peak10_feasibility' if peak else 'mean_no_peak',
                     status=solver.StatusName(status), build_seconds=build_seconds,
                     solver_wall_seconds=solver.WallTime(), objective=None if peak else float(solver.ObjectiveValue()),
                     best_bound=None if peak else float(solver.BestObjectiveBound()),
                     warning='UNKNOWN is not infeasibility; peakOPTIMAL means purefeasibility only.')
        stages.append(stage)
        print(json.dumps(stage), flush=True)
    candidates = [dict(id='source', labels=z['labels'], work=source_load.max(1), score=source_score, stage='source')]
    candidates += [dict(id=f'incumbent{j}', **item) for j, item in enumerate(sorted(pool.values(), key=lambda x: x['score']))]
    assert len(candidates) <= 8
    saved = dict(routes=routes, sharedwork=shared_work, sharedwaves=shared_waves)
    for candidate in candidates:
        ident = candidate['id']
        saved[ident + '.labels'] = candidate['labels']
        saved[ident + '.rows'] = template_rows(z['rotations'], candidate['labels'], z['shared'])
        saved[ident + '.work'] = candidate['work']
    np.savez_compressed(OUT / 'candidates.frozen.npz', **saved)
    report = dict(status='bounded_rotation_CP_fitted', protocol_sha256=sha(OUT / 'protocol.frozen.json'),
                  stages=stages, events=events, source_work=stats(source_load.max(1)),
                  candidates=[{k: v for k, v in candidate.items() if k not in ('labels', 'work')} |
                              dict(work=stats(candidate['work']),
                                   peak_readwaves=int((6 + shared_waves + X[candidate['labels']][routes].sum(1)).max()))
                              for candidate in candidates],
                  seconds=time.monotonic() - start,
                  limits='Onlycycliclabel CP; no external/free-map inserted as cyclic. Work objective not native. TRAIN only.')
    dump(OUT / 'fit_report.json', report)
    print(json.dumps({k: v for k, v in report.items() if k not in ('events', 'candidates')}, indent=2))


def native():
    protocol, z = inputs()
    assert not (OUT / 'native_report.json').exists()
    fit = json.loads((OUT / 'fit_report.json').read_text())
    candidates = load(OUT / 'candidates.frozen.npz')
    ids = [x['id'] for x in fit['candidates']]
    assert len(ids) <= 8
    dump(OUT / 'native_protocol.frozen.json', dict(status='frozen_before_native', source_sha256=sha(__file__),
         fit_report_sha256=sha(OUT / 'fit_report.json'), candidates_sha256=sha(OUT / 'candidates.frozen.npz'),
         ids=ids, priorities=['GF', 'DF'], queries=len(ids) * 1024, no_extra_fit=True))
    backend = load_python()
    service = backend.Service(**protocol['service'])
    directory = OUT / 'native'
    directory.mkdir()
    binary = directory / 'native.so'
    subprocess.run(['c++', '-std=c++17', '-O3', '-shared', '-fPIC', str(CPP), '-o', str(binary)], check=True)
    lib = ctypes.CDLL(str(binary))
    fn = lib.eval_ordinary_read
    i32 = np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
    i64 = np.ctypeslib.ndpointer(dtype=np.int64, flags='C_CONTIGUOUS')
    fn.argtypes = [i32, i32, i32, i32, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                   ctypes.c_int, i64, i64, i64, ctypes.c_void_p, i64]
    def evaluate(plan, routes, config, trace=False):
        backend.validate_plan(plan, 65, config)
        routes = np.ascontiguousarray(routes, np.int32)
        cost = np.zeros(len(routes), np.int64)
        bound = cost.copy()
        nlog = np.zeros(1, np.int64)
        tr = np.zeros((6000 if trace else 1, 8), np.int64)
        fn(plan['offsets'], plan['jobs'], plan['input_columns'], routes, 64, 1, len(plan['jobs']),
           len(routes), 6, config.array(), cost, bound,
           ctypes.c_void_p(tr.ctypes.data) if trace else None, nlog)
        assert np.all(cost >= bound)
        return cost, bound, tr[:nlog[0]].copy() if trace else None
    output, records, physical, trace_checks = {}, [], {}, []
    stage_arrays = load(STAGE / 'cost_arrays.npz')
    for ident in ids:
        rows = candidates[ident + '.rows']
        labels = candidates[ident + '.labels']
        assert np.array_equal(rows, template_rows(z['rotations'], labels, z['shared']))
        plan = backend.lower_rows(rows[:, 0], rows[:, 1], 1536, [512] * 64 + [1024], service)
        ledger = backend.physical_addresses(plan, service)
        addresses = ledger.pop('addresses')
        assert ledger['chip_blocks'] == [1] * 32
        physical[ident] = ledger
        np.savez_compressed(directory / (ident + '.plan.npz'), rows=rows, labels=labels, routes=z['routes'],
                            offsets=plan['offsets'], jobs=plan['jobs'], input_columns=plan['input_columns'],
                            total_rows=plan['total_rows'], addresses=np.asarray([[a[k] for k in ['chip', 'block', 'wordline', 'reserved_wordlines', 'pages', 'first_plane']] for a in addresses], np.int32))
        for priority in (0, 1):
            cfg = dict(protocol['service'])
            cfg['down_first'] = priority
            config = backend.Service(**cfg)
            cost, bound, _ = evaluate(plan, z['routes'], config)
            key = ident + '.' + ('DF' if priority else 'GF')
            output[key + '.cost'], output[key + '.bound'] = cost, bound
            qcost, qbound, trace = evaluate(plan, z['routes'][:1], config, True)
            py = backend.ReferenceGraph.build(plan, np.r_[z['routes'][0], 64], config).run()
            assert qcost[0] == cost[0] == py['cost'] and np.array_equal(py['trace'], trace)
            output[key + '.q0trace'] = trace
            trace_checks.append(dict(id=key, nodes=len(trace), all_nodes_exact=True))
            if ident == 'source':
                assert np.array_equal(cost, stage_arrays['granite4tiny.M32.minimum_depth.' + ('DF' if priority else 'GF') + '.cost'])
            records.append(dict(id=ident, priority='DF' if priority else 'GF', cost=stats(cost), bound=stats(bound)))
    np.savez_compressed(OUT / 'native_arrays.npz', **output)
    best = min(records, key=lambda x: x['cost']['mean_ns'])
    free = json.loads((FREE / 'native_report.json').read_text())['best_coupled_input1']
    dump(OUT / 'native_report.json', dict(status='bounded_standard_cyclic_native_control',
         protocol_sha256=sha(OUT / 'protocol.frozen.json'), native_protocol_sha256=sha(OUT / 'native_protocol.frozen.json'),
         records=records, best_cyclic=best, previous_free=free,
         free_mean_improvement_vs_cyclic_pct=100 * (1 - free['cost']['mean_ns'] / best['cost']['mean_ns']),
         free_p99_improvement_vs_cyclic_pct=100 * (1 - free['cost']['p99_ns'] / best['cost']['p99_ns']),
         native_costs=len(ids) * 1024, source_bridge='All512 sourceGF/DF costs exact against frozenStageB',
         physical=physical, traces=trace_checks,
         limits='Same32chiphardware androwmultisets/shared. StandardCP vsstandardCP heuristic controls, no equal-optimality guarantee. Noheldout, newgeometry, copies, nativecode or input4fit.'))
    print(json.dumps(dict(best_cyclic=best, previous_free=free,
          free_mean_improvement_vs_cyclic_pct=100 * (1 - free['cost']['mean_ns'] / best['cost']['mean_ns'])), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('freeze', 'fit', 'native'))
    args = parser.parse_args()
    {'freeze': freeze, 'fit': fit, 'native': native}[args.mode]()
