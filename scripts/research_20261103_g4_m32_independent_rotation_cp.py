#!/usr/bin/env python3
"""Bounded phase-independent cyclic control: a standard stronger label space."""
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
COMMON = BASE / 'g4-M32-cyclic-rotation-cp'
FREE = BASE / 'g4-M32-coupled-cp'
OUT = BASE / 'g4-M32-independent-rotation-cp'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def load(path):
    with np.load(path) as z:
        return {key: z[key] for key in z.files}


def stats(values):
    return dict(mean_ns=float(np.mean(values)), p99_ns=float(np.percentile(values, 99)))


def rows_from(z, labels):
    labels = np.asarray(labels, np.int64)
    assert labels.shape == (64, 2) and np.all((labels >= 0) & (labels < 32))
    rows = np.empty((65, 2, 32), np.int64)
    rows[:64, 0] = z['gu_rotations'][labels[:, 0]]
    rows[:64, 1] = z['down_rotations'][labels[:, 1]]
    rows[64] = z['shared']
    assert np.all(rows[:64, 0].sum(1) == 1024) and np.all(rows[:64, 1].sum(1) == 1536)
    return rows


def freeze():
    OUT.mkdir(exist_ok=False)
    (OUT / 'source-snapshot').mkdir()
    old = load(COMMON / 'inputs.frozen.npz')
    fitted = load(COMMON / 'candidates.frozen.npz')
    common = json.loads((COMMON / 'native_report.json').read_text())['best_cyclic']
    hint = fitted[common['id'] + '.labels']
    z = dict(gu_rotations=old['rotations'][:, 0], down_rotations=old['rotations'][:, 1],
             source_labels=np.repeat(old['labels'][:, None], 2, axis=1),
             hint_labels=np.repeat(hint[:, None], 2, axis=1), shared=old['shared'], routes=old['routes'])
    assert np.array_equal(rows_from(z, z['hint_labels']), fitted[common['id'] + '.rows'])
    assert np.array_equal(rows_from(z, z['source_labels']), fitted['source.rows'])
    np.savez_compressed(OUT / 'inputs.frozen.npz', **z)
    files = [Path(__file__), CPP, PYTHON,
             ROOT / 'scripts/research_20261103_topology_placement_replay.py',
             STAGE / 'protocol.frozen.json', STAGE / 'cost_arrays.npz',
             COMMON / 'protocol.frozen.json', COMMON / 'inputs.frozen.npz', COMMON / 'candidates.frozen.npz',
             COMMON / 'native_report.json', COMMON / 'native_arrays.npz', COMMON / 'producer-seal/completion.json',
             FREE / 'native_report.json', FREE / 'native_arrays.npz', OUT / 'inputs.frozen.npz']
    inputs = []
    for path in files:
        item = dict(path=str(path), sha256=sha(path))
        if path.suffix in ('.py', '.cpp'):
            snap = OUT / 'source-snapshot' / (item['sha256'][:16] + '-' + path.name)
            shutil.copy2(path, snap)
            item['snapshot'] = str(snap)
        inputs.append(item)
    service = json.loads((COMMON / 'protocol.frozen.json').read_text())['service']
    protocol = dict(status='frozen_before_independent_phase_rotation_fit', utc=datetime.now(timezone.utc).isoformat(),
                    source_sha256=sha(__file__), inputs=inputs, service=service,
                    scope='G4 M32/input1 L0actualW8 TRAIN512 only; no sharedfit/heldout/alllayers/input4/nativechanges/replicas.',
                    labels='GU andDown each64experts choose anyof32 exactexpert0 phase rotations independently. GU40row rotates exactly; GU25active andDown12active, overlapmayexceed5/unionmayomit chips.',
                    costs='Direct exact29102*G+28762*D+actualshared perchip. DirectwavesG+D, neverreuse6+X assumption.',
                    anchors=dict(original_source='stageB minimum_depth commonlabels', common_hint=common),
                    solver=dict(version=ortools.__version__, workers=8, seed=20261114, peak_seconds=30, mean_seconds=90,
                                onehot_bools=4096, peak='sum_selected(G+D)+actualsharedwaves<=10, purefeasibility.',
                                mean='Removepeakconstraints,minimize sumT. Globalobjectivecap atretainedlearnedcommonhint, no perquerycap. UNKNOWN isnotinfeasible.'),
                    retention='Atmost6newCP maps+source+commonhint=8. Bothanchors retained regardlessCPstatus. Sameboundedactualworkpool policy as earlier; peakmapmaybesuperseded.',
                    native='<=8maps xGF/DF x512=8192costs. Bothsource/commonhint bridgeall512againstsealedanchors. q0allnodesPythonGraph. No extra2kcoordinate or input4fit.',
                    interpretation='Necessary standardstrongcontrol, not newalgorithm. EqualCPtime doesnotproveequalconvergence oroptimality inlarger space. Comparefree onlyto strongestlearnedcontrol andkeepnegativecases.')
    dump(OUT / 'protocol.frozen.json', protocol)
    print(json.dumps(dict(protocol_sha256=sha(OUT / 'protocol.frozen.json'), source_sha256=sha(__file__))))


def inputs():
    protocol = json.loads((OUT / 'protocol.frozen.json').read_text())
    assert sha(__file__) == protocol['source_sha256']
    for item in protocol['inputs']:
        assert sha(item['path']) == item['sha256'], item['path']
    return protocol, load(OUT / 'inputs.frozen.npz')


def fit():
    p, z = inputs()
    assert not (OUT / 'fit_report.json').exists()
    backend = load_python()
    service = backend.Service(**p['service'])
    hint_rows = rows_from(z, z['hint_labels'])
    plan = backend.lower_rows(hint_rows[:, 0], hint_rows[:, 1], 1536, [512] * 64 + [1024], service)
    table = backend.occupancy_table(plan, service)['chip_ns']
    shared_work = table[64].sum(0)
    shared_waves = np.zeros(32, np.int64)
    for phase in range(2):
        for chip, nr, nc, pages, waves in plan['jobs'][slice(*plan['offsets'][64, phase])]:
            shared_waves[chip] += waves
    G, D = z['gu_rotations'] > 0, z['down_rotations'] > 0
    assert np.all(G.sum(1) == 25) and np.all(D.sum(1) == 12) and shared_waves.sum() == 73
    routes = z['routes']
    def counts(labels):
        return G[labels[:, 0]][routes].sum(1), D[labels[:, 1]][routes].sum(1)
    def actual(labels):
        gu, down = counts(labels)
        return 29102 * gu + 28762 * down + shared_work
    hint_work = actual(z['hint_labels'])
    assert np.array_equal(hint_work, table[routes].sum((1, 2)) + shared_work)
    cap = int(hint_work.max(1).sum())
    pool, events = {}, []
    start, best_score = time.monotonic(), cap
    def retain(labels, stage, force=False, objective=None):
        nonlocal best_score
        labels = np.asarray(labels, np.int64)
        work = actual(labels).max(1)
        score = int(work.sum())
        improved = score < best_score
        best_score = min(best_score, score)
        events.append(dict(stage=stage, actual_score_ns=score, improved=improved,
                           solver_objective=objective, elapsed_seconds=time.monotonic() - start))
        if np.array_equal(labels, z['hint_labels']) or np.array_equal(labels, z['source_labels']):
            return
        key = labels.tobytes()
        if score < cap or force:
            pool[key] = dict(labels=labels.copy(), work=work, score=score, stage=stage, forced=force)
            chosen = sorted(pool, key=lambda k: pool[k]['score'])[:6]
            if force and key not in chosen:
                chosen = chosen[:5] + [key]
            for old in list(pool):
                if old not in chosen:
                    del pool[old]
    def build(peak, initial):
        model = cp_model.CpModel()
        Y = [[[model.NewBoolVar(f'y{e}_{phase}_{label}') for label in range(32)] for phase in range(2)] for e in range(64)]
        for e in range(64):
            for phase in range(2):
                model.AddExactlyOne(Y[e][phase])
                for label in range(32):
                    model.AddHint(Y[e][phase][label], int(initial[e, phase] == label))
        # Complement GU uses7ratherthan25 onehotterms; directDown uses12.
        gu_omit = [[sum(Y[e][0][int(label)] for label in np.flatnonzero(~G[:, chip])) for chip in range(32)] for e in range(64)]
        down_on = [[sum(Y[e][1][int(label)] for label in np.flatnonzero(D[:, chip])) for chip in range(32)] for e in range(64)]
        high = int(6 * (29102 + 28762) + shared_work.max())
        T = [model.NewIntVar(280000, high, f'T{q}') for q in range(512)]
        initial_work = actual(initial).max(1)
        for q, route in enumerate(routes):
            for chip in range(32):
                g = 6 - sum(gu_omit[int(e)][chip] for e in route)
                d = sum(down_on[int(e)][chip] for e in route)
                model.Add(29102 * g + 28762 * d + int(shared_work[chip]) <= T[q])
                if peak:
                    model.Add(g + d + int(shared_waves[chip]) <= 10)
            model.AddHint(T[q], int(initial_work[q]))
        if not peak:
            model.Add(sum(T) <= cap)
            model.Minimize(sum(T))
        return model, Y
    initial = z['hint_labels'].copy()
    stages = []
    for peak, limit in ((True, 30), (False, 90)):
        before_build = time.monotonic()
        model, Y = build(peak, initial)
        build_seconds = time.monotonic() - before_build
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = limit
        solver.parameters.num_search_workers = 8
        solver.parameters.random_seed = p['solver']['seed']
        def decode(value):
            return np.asarray([[next(label for label, var in enumerate(phase) if value(var)) for phase in expert] for expert in Y], np.int64)
        class Callback(cp_model.CpSolverSolutionCallback):
            def on_solution_callback(self):
                retain(decode(self.Value), 'peak' if peak else 'mean', force=peak,
                       objective=float(self.ObjectiveValue()))
        status = solver.Solve(model, Callback())
        if status in (cp_model.FEASIBLE, cp_model.OPTIMAL):
            labels = decode(solver.Value)
            retain(labels, 'peak_final' if peak else 'mean_final', force=peak,
                   objective=float(solver.ObjectiveValue()))
            if peak:
                initial = labels
        stage = dict(stage='peak10_feasibility' if peak else 'mean_no_peak', status=solver.StatusName(status),
                     build_seconds=build_seconds, solver_wall_seconds=solver.WallTime(),
                     objective=None if peak else float(solver.ObjectiveValue()),
                     best_bound=None if peak else float(solver.BestObjectiveBound()),
                     warning='UNKNOWN isnotinfeasible; peakOPTIMAL means feasibilityonly, not latencyoptimum.')
        stages.append(stage)
        print(json.dumps(stage), flush=True)
    candidates = []
    for ident, labels in (('source', z['source_labels']), ('common_hint', z['hint_labels'])):
        work = actual(labels).max(1)
        candidates.append(dict(id=ident, labels=labels, work=work, score=int(work.sum()), stage=ident))
    candidates += [dict(id=f'incumbent{j}', **item) for j, item in enumerate(sorted(pool.values(), key=lambda item: item['score']))]
    assert len(candidates) <= 8
    saved = dict(routes=routes, sharedwork=shared_work, sharedwaves=shared_waves)
    summaries = []
    for candidate in candidates:
        ident, labels = candidate['id'], candidate['labels']
        saved[ident + '.labels'], saved[ident + '.rows'], saved[ident + '.work'] = labels, rows_from(z, labels), candidate['work']
        gu, down = counts(labels)
        summaries.append({k: v for k, v in candidate.items() if k not in ('labels', 'work')} |
                         dict(work=stats(candidate['work']), peak_readwaves=int((gu + down + shared_waves).max())))
    np.savez_compressed(OUT / 'candidates.frozen.npz', **saved)
    report = dict(status='bounded_independent_phase_rotation_fit', protocol_sha256=sha(OUT / 'protocol.frozen.json'),
                  stages=stages, events=events, candidates=summaries, seconds=time.monotonic() - start,
                  limits='Retainsource/commonhint, nofree-map isinserted into cyclicspace. MeanFEASIBLE notoptimum, worknotnative. Noheldout/newfitforinput4.')
    dump(OUT / 'fit_report.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'events'}, indent=2))


def native():
    p, z = inputs()
    assert not (OUT / 'native_report.json').exists()
    fit = json.loads((OUT / 'fit_report.json').read_text())
    c = load(OUT / 'candidates.frozen.npz')
    ids = [x['id'] for x in fit['candidates']]
    assert len(ids) <= 8
    dump(OUT / 'native_protocol.frozen.json', dict(status='frozen_before_native', source_sha256=sha(__file__),
         fit_report_sha256=sha(OUT / 'fit_report.json'), candidates_sha256=sha(OUT / 'candidates.frozen.npz'),
         ids=ids, priorities=['GF', 'DF'], queries=len(ids) * 1024, no_extra_fit=True))
    backend = load_python()
    service = backend.Service(**p['service'])
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
        cost, bound, nlog = np.zeros(len(routes), np.int64), np.zeros(len(routes), np.int64), np.zeros(1, np.int64)
        tr = np.zeros((6000 if trace else 1, 8), np.int64)
        fn(plan['offsets'], plan['jobs'], plan['input_columns'], routes, 64, 1, len(plan['jobs']), len(routes), 6,
           config.array(), cost, bound, ctypes.c_void_p(tr.ctypes.data) if trace else None, nlog)
        assert np.all(cost >= bound)
        return cost, bound, tr[:nlog[0]].copy() if trace else None
    common = json.loads((COMMON / 'native_report.json').read_text())['best_cyclic']
    anchors = dict(source=load(STAGE / 'cost_arrays.npz'), common_hint=load(COMMON / 'native_arrays.npz'))
    output, records, physical, traces = {}, [], {}, []
    for ident in ids:
        rows = c[ident + '.rows']
        assert np.array_equal(rows, rows_from(z, c[ident + '.labels']))
        plan = backend.lower_rows(rows[:, 0], rows[:, 1], 1536, [512] * 64 + [1024], service)
        ledger = backend.physical_addresses(plan, service)
        addresses = ledger.pop('addresses')
        assert ledger['chip_blocks'] == [1] * 32
        physical[ident] = ledger
        np.savez_compressed(directory / (ident + '.plan.npz'), rows=rows, labels=c[ident + '.labels'], routes=z['routes'],
             offsets=plan['offsets'], jobs=plan['jobs'], input_columns=plan['input_columns'], total_rows=plan['total_rows'],
             addresses=np.asarray([[a[k] for k in ('chip', 'block', 'wordline', 'reserved_wordlines', 'pages', 'first_plane')] for a in addresses], np.int32))
        for priority in (0, 1):
            cfg = dict(p['service'])
            cfg['down_first'] = priority
            config = backend.Service(**cfg)
            label = 'DF' if priority else 'GF'
            cost, bound, _ = evaluate(plan, z['routes'], config)
            key = ident + '.' + label
            output[key + '.cost'], output[key + '.bound'] = cost, bound
            qcost, qbound, trace = evaluate(plan, z['routes'][:1], config, True)
            py = backend.ReferenceGraph.build(plan, np.r_[z['routes'][0], 64], config).run()
            assert qcost[0] == cost[0] == py['cost'] and np.array_equal(py['trace'], trace)
            output[key + '.q0trace'] = trace
            traces.append(dict(id=key, nodes=len(trace), all_nodes_exact=True))
            if ident in anchors:
                anchor_key = ('granite4tiny.M32.minimum_depth' if ident == 'source' else common['id']) + '.' + label + '.cost'
                assert np.array_equal(cost, anchors[ident][anchor_key])
            records.append(dict(id=ident, priority=label, cost=stats(cost), bound=stats(bound)))
    np.savez_compressed(OUT / 'native_arrays.npz', **output)
    best = min(records, key=lambda item: item['cost']['mean_ns'])
    free = json.loads((FREE / 'native_report.json').read_text())['best_coupled_input1']
    report = dict(status='bounded_phase_independent_cyclic_native_control', protocol_sha256=sha(OUT / 'protocol.frozen.json'),
                  native_protocol_sha256=sha(OUT / 'native_protocol.frozen.json'), records=records, best_independent=best,
                  common_anchor=common, free=free,
                  free_mean_improvement_vs_independent_pct=100 * (1 - free['cost']['mean_ns'] / best['cost']['mean_ns']),
                  free_p99_improvement_vs_independent_pct=100 * (1 - free['cost']['p99_ns'] / best['cost']['p99_ns']),
                  native_costs=len(ids) * 1024, physical=physical, traces=traces,
                  anchor_bridges='All512 GF/DF source andlearnedcommonhint exact versus sealed earlierresults.',
                  limits='Standardlargercyclicspace withsamefiniteCPbudget, notglobaloptimum. Sameprofiles/shared/copies0. Noheldout/alllayers/nativechanges orinput4fit. Workfitmaynotimprovenative; anchorsretained.')
    dump(OUT / 'native_report.json', report)
    print(json.dumps({k: report[k] for k in ('best_independent', 'common_anchor', 'free', 'free_mean_improvement_vs_independent_pct', 'free_p99_improvement_vs_independent_pct')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('freeze', 'fit', 'native'))
    args = parser.parse_args()
    {'freeze': freeze, 'fit': fit, 'native': native}[args.mode]()
