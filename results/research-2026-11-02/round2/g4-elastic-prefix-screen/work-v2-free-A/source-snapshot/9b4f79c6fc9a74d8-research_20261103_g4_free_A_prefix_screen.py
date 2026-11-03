#!/usr/bin/env python3
"""Bounded arbitrary-eight-chip A / frozen B GU prefix-view TRAIN work screen.

All searches are offline. Each expert has exactly one resident A plus B view.
This is a necessary-work screen, not native performance or a new algorithm.
"""
import argparse
import ctypes
import hashlib
import json
import shutil
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import networkx as nx

from research_20261103_topology_placement_replay import CPP as NATIVE_CPP, PYTHON, load_python

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'results/research-2026-11-02/round2'
OUT = BASE/'g4-elastic-prefix-screen/work-v2-free-A'
B = BASE/'compact-coupled-solver/continuation/native/0107.npz'
SERVICE = BASE/'compact-coupled-solver/protocol.frozen.json'
TRAIN = BASE/'w8-format-training/granite4tiny/decode.routes.npz'
ORACLE = ROOT/'scripts/research_20261103_g4_down_copy_oracle.cpp'
RELAXED = BASE/'g4-elastic-prefix-screen/free-span-bound/report.json'
SEEDS = {h: [2026110800+h*10+j for j in range(8)] for h in (2, 4, 6, 8)}
MATCH_SEEDS = [2026110901, 2026110902, 2026110903, 2026110904]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2)+'\n')


def metrics(x):
    return dict(mean_ns=float(np.mean(x)), p99_ns=float(np.percentile(x, 99)))


def freeze():
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT/'source-snapshot').mkdir()
    service = json.loads(SERVICE.read_text())['service']
    with np.load(B) as z:
        rows, routes = z['rows'], z['routes']
        assert float(z['cost'].mean()) == 561099.01953125
    assert rows.shape == (65, 2, 16) and routes.shape == (512, 6)
    assert np.all(np.sort(rows[:64, 0], axis=1) == [40]*7+[82]*3+[83]*6)
    with np.load(TRAIN) as z:
        assert z['split'].item() == 'train' and z['completed'].all()
        assert np.array_equal(routes, z['routes'][:, :, 0].reshape(512, 6))
    assert service['command_ns'] == 250 and service['input_channel_copies'] == 1
    assert service['output_fringe'] == 1
    inputs = []
    paths = [Path(__file__), ORACLE, B, SERVICE, TRAIN, RELAXED, PYTHON, NATIVE_CPP,
             ROOT/'scripts/research_20261103_topology_placement_replay.py',
             ROOT/'scripts/research_20261102_whole_row_compiler.py']
    for path in paths:
        item = dict(path=str(path), sha256=sha(path))
        if path.suffix in ('.py', '.cpp'):
            snapshot = OUT/'source-snapshot'/(item['sha256'][:16]+'-'+path.name)
            shutil.copy2(path, snapshot)
            item['snapshot'] = str(snapshot)
        inputs.append(item)
    protocol = dict(status='frozen_before_offline_TRAIN_search',
                    utc=datetime.now(timezone.utc).isoformat(),
                    source_sha256=sha(__file__), baseline_path=str(B),
                    inputs=inputs, service=service,
                    scope='G4 L0 actualW8 TRAIN512; frozen B/Down/shared; no heldout/native expansion.',
                    routeblind=dict(high_source_counts=[2, 4, 6, 8], seeds=SEEDS, count=32,
                                    construction='Uniform distinct source subset of h Bhigh and 8-h Blow perexpert.'),
                    structured_initialization=dict(seeds=MATCH_SEEDS, networkx_version=nx.__version__,
                        matching='TRAIN maximum-coactivation-weight perfect matching; random integer secondary ties have total less than one primary unit.',
                        placement='Each pair uses complementary A8 masks maximizing retained B core: top8 B_rows[e]-B_rows[f], random secondary ties. Four TRAIN-derived initial layouts, separately labeled, not routeblind.'),
                    dynamic_fit=dict(starts='best two among 32routeblind+4TRAINmatching mean minmaxwork', passes=2,
                                     expert_order='descending TRAIN frequency, tie ID',
                                     proposals='All 8x8 selected-source/unselected-target swaps per expert; no fixed h histogram.',
                                     objective='Mean exact 64-mask minimum maxchip context-ownership work; full commands/input/page-fringe output paid.'),
                    static_controls=[
                        dict(name='same_fixed_A_B', passes=2, starts=['allB', 'allA'],
                             proposals='Perexpert fixed offline choice of B or selected A; zero replica.'),
                        dict(name='free_A_or_B', passes=2, starts=['allB', 'allA'],
                             proposals='Perexpert B or current A or any 64 one-source swaps of latent A; no replica; extra control budget reported.')],
                    residency='Each expert fixes ONE A8 source set. B original rows stay as prefixes; offsource original rows are copied once into active prefixes up to128. Runtime only A/B, not all layouts. copyRows=1024-sum(B[source]); extraWaves=sum_source(3-Bwaves[source]), so search is NOT same-copy-budget constrained.',
                    gate=dict(mean_threshold_pct=2, p99_threshold_pct=5,
                              necessary_LB='host_submit + exact min maxchip work; also allmask optimal first-acquisition rank assignment.',
                              action='Report if either ceiling passes, but do not automatically run native/copy controls/alllayers/heldout.'),
                    limitations='Finite heuristic search; no optimum over offline A positions. Ordinary synthetic service and existing page-fringe output assumptions. Prefix capacity saving and adaptive geometry latency must be attributed separately.')
    dump(OUT/'protocol.frozen.json', protocol)
    print(json.dumps(dict(protocol_sha256=sha(OUT/'protocol.frozen.json'), source_sha256=sha(__file__))))


def logical_image(rows, active):
    resident = rows.copy()
    ledger = []
    for e in range(64):
        original = {c: list(range(int(rows[e, 0, :c].sum()), int(rows[e, 0, :c+1].sum()))) for c in range(16)}
        donors = [r for c in range(16) if not active[e, c] for r in original[c]]
        image = {c: rr.copy() for c, rr in original.items()}
        av = {}
        cursor = 0
        for c in np.flatnonzero(active[e]):
            c = int(c)
            nr = 128-len(original[c])
            image[c] += donors[cursor:cursor+nr]
            cursor += nr
            av[c] = image[c].copy()
            resident[e, 0, c] = 128
        assert cursor == len(donors)
        assert sorted(r for rr in av.values() for r in rr) == list(range(1024))
        expected = Counter({r: 2 if r in donors else 1 for r in range(1024)})
        assert Counter(r for rr in image.values() for r in rr) == expected
        ledger.append(dict(expert=e, A_source_chips=np.flatnonzero(active[e]).tolist(),
                           B_original_rows=original, resident_prefix_image=image,
                           A_rows=av, copied_logical_rows=donors, copied_rows=len(donors),
                           copied_bytes=len(donors)*1536,
                           high_sources=int(np.sum(active[e] & (rows[e, 0] > 40)))))
    return resident, ledger


def run():
    p = json.loads((OUT/'protocol.frozen.json').read_text())
    assert not (OUT/'work_report.json').exists() and sha(__file__) == p['source_sha256']
    for item in p['inputs']:
        assert sha(item['path']) == item['sha256']
    started, cpu = time.monotonic(), time.process_time()
    backend = load_python()
    s = backend.Service(**p['service'])
    with np.load(B) as z:
        rows, routes, native = z['rows'], np.ascontiguousarray(z['routes'], np.int32), z['cost']
        plan = {k: z[k] for k in ['offsets', 'jobs', 'input_columns', 'total_rows']}
    table = backend.occupancy_table(plan, s)['chip_ns']
    base = np.ascontiguousarray(table[routes].sum((1, 2))+table[64].sum(0), np.int64)
    Bgu = table[:64, 0]
    Ajob = 85274
    # Verify the row-independent A-job cost against the declared ordinary backend.
    check = backend.lower_rows(np.full((1, 16), 128), np.full((1, 16), 96), 1536, [512], s)
    assert np.all(backend.occupancy_table(check, s)['chip_ns'][0, 0] == Ajob)
    binary = OUT/'oracle.so'
    subprocess.run(['c++', '-O3', '-std=c++17', '-shared', '-fPIC', str(ORACLE), '-o', str(binary)], check=True)
    fn = ctypes.CDLL(str(binary)).down_copy_oracle
    i64 = np.ctypeslib.ndpointer(dtype=np.int64, flags='C_CONTIGUOUS')
    i32 = np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
    fn.argtypes = [i64, i32, i64, i32, ctypes.c_int, i64, i32]
    allq = np.arange(512, dtype=np.int32)
    members = [np.flatnonzero((routes == e).any(1)).astype(np.int32) for e in range(64)]
    order = sorted(range(64), key=lambda e: (-len(members[e]), e))
    calls = query_evaluations = 0

    def evaluate(delta, indices=allq):
        nonlocal calls, query_evaluations
        v, masks = np.zeros(len(indices), np.int64), np.zeros(len(indices), np.int32)
        fn(base, routes, np.ascontiguousarray(delta, np.int64), indices, len(indices), v, masks)
        calls += 1
        query_evaluations += len(indices)
        return v, masks

    layouts, blind, blind_values = [], [], []
    for h, seeds in SEEDS.items():
        for seed in seeds:
            rng = np.random.default_rng(seed)
            active = np.zeros((64, 16), bool)
            for e in range(64):
                high, low = np.flatnonzero(rows[e, 0] > 40), np.flatnonzero(rows[e, 0] == 40)
                chosen = np.r_[rng.choice(high, h, replace=False), rng.choice(low, 8-h, replace=False)]
                active[e, chosen] = True
            v, m = evaluate(active*Ajob-Bgu)
            layouts.append(active)
            blind_values.append(v)
            blind.append(dict(index=len(blind), h=h, seed=seed, work=metrics(v)))
    coact = np.zeros((64, 64), np.int64)
    for route in routes:
        for i in range(6):
            for j in range(i+1, 6):
                e, f = sorted(map(int, (route[i], route[j])))
                coact[e, f] += 1
    structured = []
    for seed in MATCH_SEEDS:
        rng = np.random.default_rng(seed)
        graph = nx.Graph()
        for e in range(64):
            for f in range(e+1, 64):
                graph.add_edge(e, f, weight=int(coact[e, f])*1_000_000+int(rng.integers(0, 1000)))
        pairs = sorted(tuple(sorted(edge)) for edge in nx.max_weight_matching(graph, maxcardinality=True))
        assert len(pairs) == 32 and sorted(e for edge in pairs for e in edge) == list(range(64))
        active = np.zeros((64, 16), bool)
        for e, f in pairs:
            difference = rows[e, 0]-rows[f, 0]
            tie = rng.permutation(16)
            sources = np.lexsort((tie, -difference))[:8]
            active[e, sources] = True
            active[f] = ~active[e]
            assert np.sum(difference[sources]) == np.sort(difference)[-8:].sum()
        v, m = evaluate(active*Ajob-Bgu)
        layouts.append(active)
        blind_values.append(v)
        structured.append(dict(index=len(layouts)-1, seed=seed, pairs=pairs,
                               co_selected_pairs=int(sum(coact[e, f] for e, f in pairs)), work=metrics(v)))
    pool = blind+structured
    start_indices = sorted(range(len(layouts)), key=lambda j: (pool[j]['work']['mean_ns'], j))[:2]
    finals = []
    for start_index in start_indices:
        active = layouts[start_index].copy()
        delta = active*Ajob-Bgu
        values = blind_values[start_index].copy()
        initial = metrics(values)
        history = []
        proposed = 0
        for pi in range(2):
            changed = 0
            for e in order:
                ii = members[e]
                oldactive = active[e].copy()
                winner = oldactive.copy()
                winvalues = values[ii].copy()
                score = int(winvalues.sum())
                for source in np.flatnonzero(oldactive):
                    for target in np.flatnonzero(~oldactive):
                        candidate = oldactive.copy()
                        candidate[source], candidate[target] = False, True
                        delta[e] = candidate*Ajob-Bgu[e]
                        vv, mm = evaluate(delta, ii)
                        proposed += 1
                        if int(vv.sum()) < score:
                            winner, winvalues, score = candidate, vv.copy(), int(vv.sum())
                changed += not np.array_equal(winner, oldactive)
                active[e] = winner
                delta[e] = winner*Ajob-Bgu[e]
                values[ii] = winvalues
            full, masks = evaluate(delta)
            assert np.array_equal(full, values)
            history.append(dict(pass_index=pi+1, changed_experts=int(changed), work=metrics(values)))
            print(json.dumps(dict(start_index=start_index, **history[-1])), flush=True)
        values, masks = evaluate(delta)
        finals.append(dict(start_index=start_index, initial=initial, history=history,
                           proposals=proposed, source_mask=active, delta=delta,
                           values=values, masks=masks, final=metrics(values)))
    selected_index = min(range(2), key=lambda j: finals[j]['final']['mean_ns'])
    selected = finals[selected_index]
    active, delta, values, masks = [selected[k] for k in ('source_mask', 'delta', 'values', 'masks')]

    # Same-layout binary static and a stronger zero-copy local free-A geometry
    # control. The latter's additional static search budget is reported, not hidden.
    def static(free_sources):
        results = []
        for start in (0, 1):
            aa = active.copy()
            isA = np.full(64, bool(start))
            loads = base.copy()
            for e in range(64):
                if isA[e]:
                    loads[members[e]] += aa[e]*Ajob-Bgu[e]
            vv = loads.max(1)
            initial = metrics(vv)
            history, proposals = [], 0
            for pi in range(2):
                changes = 0
                for e in order:
                    ii = members[e]
                    old = aa[e]*Ajob-Bgu[e] if isA[e] else np.zeros(16, np.int64)
                    candidates = [aa[e]]
                    if free_sources:
                        for source in np.flatnonzero(aa[e]):
                            for target in np.flatnonzero(~aa[e]):
                                candidate = aa[e].copy()
                                candidate[source], candidate[target] = False, True
                                candidates.append(candidate)
                    score = int(vv[ii].sum())
                    winA, winsources = bool(isA[e]), aa[e].copy()
                    winloads, winvalues = loads[ii].copy(), vv[ii].copy()
                    for candidate in [None]+candidates:
                        new = np.zeros(16, np.int64) if candidate is None else candidate*Ajob-Bgu[e]
                        ll = loads[ii]-old+new
                        ev = ll.max(1)
                        proposals += 1
                        if int(ev.sum()) < score:
                            winA = candidate is not None
                            winsources = aa[e].copy() if candidate is None else candidate.copy()
                            winloads, winvalues, score = ll, ev, int(ev.sum())
                    changes += winA != isA[e] or not np.array_equal(winsources, aa[e])
                    isA[e], aa[e], loads[ii], vv[ii] = winA, winsources, winloads, winvalues
                # Full independent accumulation checkpoint, no incremental cache.
                full = base.copy()
                for e in range(64):
                    if isA[e]:
                        full[members[e]] += aa[e]*Ajob-Bgu[e]
                assert np.array_equal(full, loads) and np.array_equal(full.max(1), vv)
                history.append(dict(pass_index=pi+1, changed_experts=int(changes), work=metrics(vv)))
            results.append(dict(start=start, initial=initial, history=history, final=metrics(vv),
                                proposals=proposals, source_mask=aa, isA=isA, values=vv))
        return results, min(range(2), key=lambda j: results[j]['final']['mean_ns'])
    fixed_static, fixed_winner = static(False)
    free_static, free_winner = static(True)

    bits = ((np.arange(64)[:, None] >> np.arange(6)) & 1).astype(np.int64)
    loads = base[:, None, :]+np.einsum('mk,qkc->qmc', bits, delta[routes])
    enumerated = loads.max(2)
    assert np.array_equal(enumerated.min(1), values)
    assert np.array_equal(enumerated.argmin(1), masks)
    plain = s.host_submit_ns+values
    firststart = s.host_submit_ns+(np.sort(loads, axis=2)[:, :, ::-1]+np.arange(16)*s.command_ns).max(2)
    sortedLB = firststart.min(1)
    assert np.all(sortedLB >= plain)
    resident, ledger = logical_image(rows, active)
    resident_plan = backend.lower_rows(resident[:, 0], resident[:, 1], 1536, [512]*64+[1024], s)
    original_plan = backend.lower_rows(rows[:, 0], rows[:, 1], 1536, [512]*64+[1024], s)
    physical = backend.physical_addresses(resident_plan, s)
    baseline_physical = backend.physical_addresses(original_plan, s)
    addresses = np.asarray([[item[k] for k in ('chip', 'block', 'wordline', 'reserved_wordlines', 'pages', 'first_plane')]
                            for item in physical.pop('addresses')], np.int32)
    baseline_physical.pop('addresses')
    copyrows = sum(item['copied_rows'] for item in ledger)
    extra_waves = int(np.sum(active*(3-np.ceil(rows[:64, 0]*1536/65536).astype(np.int64))))
    assert extra_waves == (physical['reserved_wave_bytes']-baseline_physical['reserved_wave_bytes'])//65536
    assert physical['useful_bytes']-baseline_physical['useful_bytes'] == copyrows*1536
    views = []
    for e in range(64):
        for j in range(*resident_plan['offsets'][e, 0]):
            c = int(resident_plan['jobs'][j, 0])
            nrB, nrA = int(rows[e, 0, c]), 128 if active[e, c] else 0
            views.append(dict(expert=e, chip=c, address=addresses[j].tolist(), B_rows=nrB, A_rows=nrA,
                              B_pages=(nrB*1536+16383)//16384, A_pages=(nrA*1536+16383)//16384,
                              B_waves=(nrB*1536+65535)//65536, A_waves=(nrA*1536+65535)//65536))

    def summary(fit):
        return {k: v for k, v in fit.items() if k not in ('source_mask', 'delta', 'values', 'masks', 'isA')}

    def budget_for_sources(aa):
        copies = 1024-np.sum(aa*rows[:64, 0], axis=1)
        h = np.sum(aa & (rows[:64, 0] > 40), axis=1)
        ew = np.sum(aa*(3-(rows[:64, 0]*1536+65535)//65536), axis=1)
        return dict(copied_rows=int(copies.sum()), copied_coefficient_bytes=int(copies.sum()*1536),
                    extra_waves=int(ew.sum()), extra_programmed_bytes=int(ew.sum()*65536),
                    high_source_histogram={str(k): int(np.sum(h == k)) for k in range(9)},
                    copied_rows_per_expert=copies.tolist(), extra_waves_per_expert=ew.tolist())

    budget = budget_for_sources(active)
    budget.update(baseline_physical=baseline_physical, resident_physical=physical,
                  padding_reused_bytes=budget['copied_coefficient_bytes']-budget['extra_programmed_bytes'],
                  whole_GU_useful_budget_copies=budget['copied_coefficient_bytes']//1572864,
                  whole_GU_page_wave_budget_copies=budget['extra_waves']//24,
                  whole_GU_page_wave_remainder_waves=budget['extra_waves']%24,
                  warning='Per-layer varying budgets, not equal-budget optimization. WholeGU minimum24waves may beat fixed B25; no Down copy requirement. Block/perchip constraints also need fair controls.')
    ceilings = {}
    for name, lb in [('host_plus_work', plain), ('optimal_firststart', sortedLB)]:
        ceilings[name] = dict(LB=metrics(lb), mean_improvement_upper_pct=100*(1-lb.mean()/native.mean()),
                              p99_improvement_upper_pct=100*(1-np.percentile(lb, 99)/np.percentile(native, 99)))
    # Both trained endpoints get their tail/work ceilings recorded; fitting did
    # not change to a tail objective or refit based on this report.
    for ff in finals:
        lb = s.host_submit_ns+ff['values']
        ff['plain_ceiling'] = dict(mean_pct=100*(1-lb.mean()/native.mean()),
                                   p99_pct=100*(1-np.percentile(lb, 99)/np.percentile(native, 99)))
    gate = any(ff['plain_ceiling']['mean_pct'] >= 2 or ff['plain_ceiling']['p99_pct'] >= 5 for ff in finals)
    np.savez_compressed(OUT/'work_arrays.npz', rows_B=rows, routes=routes, base=base,
                        baseline_native=native, B_GU_work=Bgu, routeblind_source_masks=np.stack(layouts[:32]),
                        routeblind_values=np.stack(blind_values[:32]), initial_source_masks=np.stack(layouts),
                        initial_values=np.stack(blind_values), selected_source_mask=active,
                        selected_delta=delta, selected_values=values, selected_masks=masks,
                        plain_host_LB=plain, sorted_firststart_LB=sortedLB,
                        sorted_firststart_masks=firststart.argmin(1).astype(np.int32),
                        final_source_masks=np.stack([ff['source_mask'] for ff in finals]),
                        final_values=np.stack([ff['values'] for ff in finals]),
                        final_runtime_masks=np.stack([ff['masks'] for ff in finals]),
                        same_layout_static_source_masks=np.stack([ff['source_mask'] for ff in fixed_static]),
                        same_layout_static_isA=np.stack([ff['isA'] for ff in fixed_static]),
                        same_layout_static_values=np.stack([ff['values'] for ff in fixed_static]),
                        free_static_source_masks=np.stack([ff['source_mask'] for ff in free_static]),
                        free_static_isA=np.stack([ff['isA'] for ff in free_static]),
                        free_static_values=np.stack([ff['values'] for ff in free_static]),
                        resident_rows=resident, resident_jobs=resident_plan['jobs'],
                        resident_offsets=resident_plan['offsets'], resident_addresses=addresses)
    report = dict(status='bounded_work_only', protocol_sha256=sha(OUT/'protocol.frozen.json'),
                  source_sha256=sha(__file__), oracle_binary_sha256=sha(binary), baseline_native=metrics(native),
                  baseline_chip_work=metrics(base.max(1)), baseline_pooled_chip_work_ns=float(base.mean()),
                  A_job_ns=Ajob, routeblind=blind, TRAIN_matching_initializations=structured, start_indices=start_indices,
                  final_fits=[dict(**summary(ff), budget=budget_for_sources(ff['source_mask'])) for ff in finals],
                  selected_final_index=selected_index, selected_work=metrics(values), ceilings=ceilings,
                  work_improvement_pct=100*(1-values.mean()/base.max(1).mean()),
                  static_controls=dict(same_layout=dict(starts=list(map(summary, fixed_static)), winner=fixed_winner),
                                       free_A_or_B=dict(starts=list(map(summary, free_static)), winner=free_winner)),
                  replica_vs_same_layout_static_work_pct=100*(1-values.mean()/fixed_static[fixed_winner]['values'].mean()),
                  replica_vs_free_static_work_pct=100*(1-values.mean()/free_static[free_winner]['values'].mean()),
                  budget=budget, exact64_masks_checked=32768, oracle_calls=calls,
                  oracle_query_evaluations=query_evaluations, seconds=time.monotonic()-started,
                  cpu_seconds=time.process_time()-cpu, gate_passed=bool(gate),
                  scope='TRAIN512 finiteoffline heuristic workscreen; not achievednativeT, globaloptimum, equalcopybudgetfit, newalgorithm or hardwareexecution. No native/selector/wholecopycontrols/heldout executed.')
    dump(OUT/'work_report.json', report)
    dump(OUT/'copy_rowid_ledger.json', ledger)
    dump(OUT/'physical_prefix_views.json', views)
    print(json.dumps({k: report[k] for k in ('selected_work', 'ceilings', 'budget', 'gate_passed', 'seconds')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['freeze', 'run'])
    args = parser.parse_args()
    freeze() if args.mode == 'freeze' else run()
