#!/usr/bin/env python3
"""Alias-aware computed-row lowering for one sealed independent-prefix layout.

Uses the unmodified ordinary native Graph; no new execution context or early A8.
Every read has computed-row total_rows. Unique consumed row IDs are separate.
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

from research_20261103_topology_placement_replay import CPP, PYTHON, load_python

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'results/research-2026-11-02/round2'
WORK = BASE/'g4-elastic-prefix-screen/work-v3-independent-prefix'
OUT = WORK/'native-v1'
B = BASE/'compact-coupled-solver/continuation/native/0107.npz'
V1 = BASE/'g4-elastic-prefix-screen/work-v1/work_arrays.npz'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2)+'\n')


def stats(x):
    return dict(mean_ns=float(np.mean(x)), p99_ns=float(np.percentile(x, 99)))


def freeze():
    OUT.mkdir(exist_ok=False)
    (OUT/'source-snapshot').mkdir()
    files = [Path(__file__), CPP, PYTHON, B, V1, WORK/'protocol.frozen.json',
             WORK/'contract_arrays.npz', WORK/'rowid_ledgers.json',
             WORK/'greedy_arrays.npz', WORK/'greedy_report.json',
             ROOT/'scripts/research_20261103_topology_placement_replay.py',
             ROOT/'scripts/research_20261102_whole_row_compiler.py']
    inputs = []
    for path in files:
        item = dict(path=str(path), sha256=sha(path))
        if path.suffix in ('.py', '.cpp'):
            snapshot = OUT/'source-snapshot'/(item['sha256'][:16]+'-'+path.name)
            shutil.copy2(path, snapshot)
            item['snapshot'] = str(snapshot)
        inputs.append(item)
    service = json.loads((WORK/'protocol.frozen.json').read_text())['service']
    assert json.loads((WORK/'greedy_report.json').read_text())['selected_fixed_layout'] == 3
    dump(OUT/'protocol.frozen.json', dict(status='frozen_before_native_replay',
         utc=datetime.now(timezone.utc).isoformat(), source_sha256=sha(__file__), inputs=inputs,
         layout=3, queries=512, service=service, cases=['originalB','allA','greedy_fixed_layout3'],
         lowering='Native matrix descriptors have computed rowcounts, including duplicate highprefix computations. total_rows is computed count, not nominal modelR. Separate consumed-row ownership proves exactly1024 unique rows and drops all duplicate/fringe outputs. Existing lower_rows/validate_plan unchanged.',
         aliases='Native read starts at the original resident band address. Short/long views only read prefixes, at most the same stored allocation. Read overlap is not additional coefficient copies.',
         dependencies='Unmodified Graph waits all GU job PCIe outputs of an expert, thenactivation/fullA8 barrier/originalprivateDown. No early-ready/partialdot/newcontext.',
         checks='B512 exact cost bridge; q0/255/511 all3cases fullPython trace equality; allselected query/expert logical row owners + physical read prefixes + occupancy match.',
         costs='Zero runtime selector/scatter/extra software cost is an optimistic sensitivity, not a production runtime. All commands/fullinputreload/pages/waves/emittedfringe/channel/PCIe/submit costs remain.',
         gate=dict(mean_pct=2, p99_pct=5, action='Ifneitherpasses stop selectorbench. Ifpasses report rootbefore strongstatic/samebudgetcopy controls or furtherexpansion.'),
         limits='TRAIN-only fixedpairing, synthetic service, no newnumerical or hardwareABI proof. No heldout/alllayer/refit.'))
    print(json.dumps(dict(protocol_sha256=sha(OUT/'protocol.frozen.json'), source_sha256=sha(__file__))))


def run():
    protocol = json.loads((OUT/'protocol.frozen.json').read_text())
    assert sha(__file__) == protocol['source_sha256'] and not (OUT/'report.json').exists()
    for item in protocol['inputs']:
        assert sha(item['path']) == item['sha256']
    backend = load_python()
    service = backend.Service(**protocol['service'])
    contract = np.load(WORK/'contract_arrays.npz')
    greedy = np.load(WORK/'greedy_arrays.npz')
    v1, b = np.load(V1), np.load(B)
    rows, routes = contract['rows'], np.ascontiguousarray(contract['routes'], np.int32)
    ledger = json.loads((WORK/'rowid_ledgers.json').read_text())[3]['experts']
    chosen_bits = greedy['feasible_bits'][3]
    t0 = time.monotonic()
    binary = OUT/'native.so'
    subprocess.run(['c++', '-std=c++17', '-O3', '-shared', '-fPIC', str(CPP), '-o', str(binary)], check=True)
    lib = ctypes.CDLL(str(binary))
    fn = lib.eval_ordinary_read
    p32 = np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
    p64 = np.ctypeslib.ndpointer(dtype=np.int64, flags='C_CONTIGUOUS')
    fn.argtypes = [p32,p32,p32,p32,ctypes.c_int,ctypes.c_int,ctypes.c_int,
                   ctypes.c_int,ctypes.c_int,p64,p64,p64,ctypes.c_void_p,p64]
    fn.restype = None

    def evaluate(plan, rr, trace=False):
        backend.validate_plan(plan, 65, service)
        rr = np.ascontiguousarray(rr, np.int32)
        values, lb = np.zeros(len(rr), np.int64), np.zeros(len(rr), np.int64)
        capacity = 5000 if trace else 1
        log, nlog = np.zeros((capacity, 8), np.int64), np.zeros(1, np.int64)
        fn(plan['offsets'], plan['jobs'], plan['input_columns'], rr, 64, 1, len(plan['jobs']),
           len(rr), 6, service.array(), values, lb,
           ctypes.c_void_p(log.ctypes.data) if trace else None, nlog)
        assert np.all(values >= lb)
        if trace:
            assert nlog[0] <= capacity
        return values, lb, log[:nlog[0]].copy() if trace else np.empty((0, 8), np.int64)

    # Fixed coefficient-image allocations are reused, not reallocated perquery.
    resident = {}
    for e in range(65):
        for phase in range(2):
            for j in range(*v1['resident_offsets'][e, phase]):
                c = int(v1['resident_jobs'][j, 0])
                resident[e, phase, c] = v1['resident_addresses'][j].copy()

    ownership_checks = 0
    def computed_rows_and_owners(q, bits):
        nonlocal ownership_checks
        computed = rows.copy()
        owners = []
        for ordinal, e in enumerate(routes[q]):
            e = int(e)
            entry = ledger[e]
            source = entry['sources']
            lows = entry['low_destinations']
            hd = entry['high_destination']
            original = {int(c): rr for c, rr in entry['B_original_rows'].items()}
            image = {int(c): rr for c, rr in entry['resident_prefix_image'].items()}
            u = list(map(int, bits[ordinal]))
            gu = rows[e, 0].copy()
            row_owner = {}
            # Every rowowner is a (chip, rowoffset) in an actual native output.
            for i, c in enumerate(source):
                gu[c] = 128 if u[i] else rows[e, 0, c]
                for position, row_id in enumerate(image[c][:int(gu[c])]):
                    assert row_id not in row_owner
                    row_owner[row_id] = (c, position)
            for i, c in enumerate(lows):
                gu[c] = 0 if u[i] else 40
                if not u[i]:
                    for position, row_id in enumerate(original[c]):
                        assert row_id not in row_owner
                        row_owner[row_id] = (c, position)
            gu[hd] = len(original[hd]) if not u[7] else entry['high_prefix_rows'] if not all(u) else 0
            if gu[hd]:
                for position, row_id in enumerate(original[hd][:int(gu[hd])]):
                    if row_id not in row_owner:
                        row_owner[row_id] = (hd, position)
            assert sorted(row_owner) == list(range(1024))
            for row_id, (c, position) in row_owner.items():
                assert position < gu[c] and image[c][position] == row_id
            assert gu.sum() >= 1024
            computed[e, 0] = gu
            owners.append(dict(expert=e, ordinal=ordinal, computed_rows=int(gu.sum()), unique_consumed_rows=1024,
                               owner_map=np.asarray([row_owner[row_id] for row_id in range(1024)], np.int16)))
            ownership_checks += 1
        return computed, owners

    def lower(computed):
        plan = backend.lower_rows(computed[:, 0], computed[:, 1], 1536, [512]*64+[1024], service)
        assert np.array_equal(plan['total_rows'], computed.sum(2))
        aliases = np.zeros((len(plan['jobs']), 6), np.int32)
        for e in range(65):
            for phase in range(2):
                for j in range(*plan['offsets'][e, phase]):
                    c, nr, nc, pages, waves = map(int, plan['jobs'][j])
                    alias = resident[e, phase, c]
                    ac, block, wl, reserved, resident_pages, firstplane = map(int, alias)
                    assert ac == c and firstplane == 0 and pages <= resident_pages
                    assert waves <= reserved and wl+waves <= 192
                    aliases[j] = alias
        return plan, aliases

    Bplan, Baliases = lower(rows)
    assert all(np.array_equal(Bplan[k], b[k]) for k in ['jobs','offsets','input_columns','total_rows'])
    baseline, baselineLB, _ = evaluate(Bplan, routes)
    assert np.array_equal(baseline, b['cost']) and np.array_equal(baselineLB, b['lower_bound'])
    allArows, _ = computed_rows_and_owners(0, np.ones((6, 8), np.int8))
    # Complete all64 expert image to the allA endpoint; shared staysfixed.
    allArows[:64, 0] = 0
    for e in range(64):
        allArows[e, 0, ledger[e]['sources']] = 128
    Aplan, Aaliases = lower(allArows)
    allA, allALB, _ = evaluate(Aplan, routes)
    costs, bounds, work = np.zeros(512, np.int64), np.zeros(512, np.int64), np.zeros(512, np.int64)
    query_ledgers, trace_checks = [], []
    owner_maps = np.zeros((512, 6, 1024, 2), np.int16)
    counters = np.zeros((512, 2, 7), np.int64)  # jobs/usefulcomputed/pages/waves/input/out/channelns
    traces = {}
    for q in range(512):
        computed, owners = computed_rows_and_owners(q, chosen_bits[q])
        plan, aliases = lower(computed)
        costs[q], bounds[q] = [v[0] for v in evaluate(plan, routes[q:q+1])[:2]]
        table = backend.occupancy_table(plan, service)['chip_ns']
        selected = np.r_[routes[q], 64]
        work[q] = (table[routes[q]].sum((0,1))+table[64].sum(0)).max()
        assert work[q] == greedy['feasible_work'][3, q]
        for e in selected:
            for phase in range(2):
                for job in plan['jobs'][slice(*plan['offsets'][e, phase])]:
                    c, nr, nc, pages, waves = map(int, job)
                    emitted = pages*16384//nc
                    counters[q, phase] += [1, nr*nc, pages, waves, nc, emitted*4, (nc+1)//2+2*emitted]
        owner_summaries = []
        for ordinal, owner in enumerate(owners):
            owner_maps[q, ordinal] = owner.pop('owner_map')
            owner_summaries.append(owner)
        query_ledgers.append(dict(query=q, selected_experts=routes[q].tolist(), experts=owner_summaries,
                                  ownership_array='query_row_owners.npz:owner_map[query,ordinal,row_id,(chip,position)]'))
        if q in (0,255,511):
            for tag, pp, rr, aa in [('B',Bplan,rows,Baliases), ('allA',Aplan,allArows,Aaliases), ('greedy',plan,computed,aliases)]:
                ncost, nlb, ntrace = evaluate(pp, routes[q:q+1], True)
                graph = backend.ReferenceGraph.build(pp, selected, service)
                result = graph.run()
                assert result['cost'] == ncost[0] and np.array_equal(result['trace'], ntrace)
                traces[f'{tag}_q{q}'] = ntrace
                np.savez_compressed(OUT/f'{tag}_q{q}.npz', rows=rr, offsets=pp['offsets'], jobs=pp['jobs'],
                                    input_columns=pp['input_columns'], total_rows=pp['total_rows'],
                                    aliases=aa, trace=ntrace, cost=ncost, lower_bound=nlb)
                trace_checks.append(dict(case=tag, query=q, cost_ns=int(ncost[0]), nodes=len(ntrace)))
    gain = dict(mean_pct=100*(1-costs.mean()/baseline.mean()),
                p99_pct=100*(1-np.percentile(costs,99)/np.percentile(baseline,99)))
    passed = gain['mean_pct'] >= 2 or gain['p99_pct'] >= 5
    np.savez_compressed(OUT/'arrays.npz', baseline=baseline, baseline_LB=baselineLB,
                        allA=allA, allA_LB=allALB, greedy=costs, greedy_LB=bounds,
                        greedy_work=work, counters=counters, **traces)
    dump(OUT/'query_row_owners.json', query_ledgers)
    np.savez_compressed(OUT/'query_row_owners.npz', owner_map=owner_maps,
                        logical_row_ids=np.arange(1024, dtype=np.int16), routes=routes)
    result = dict(status='zero_selector_software_optimistic_native_model_sensitivity',
                  protocol_sha256=sha(OUT/'protocol.frozen.json'), source_sha256=sha(__file__),
                  native_cpp_sha256=sha(CPP), binary_sha256=sha(binary),
                  baseline=stats(baseline), allA=stats(allA), greedy=stats(costs),
                  native_improvement=gain, gate_passed=bool(passed),
                  paired_queries=dict(improved=int(np.sum(costs<baseline)), equal=int(np.sum(costs==baseline)),
                                      worsened=int(np.sum(costs>baseline)), max_saving_ns=int(np.max(baseline-costs)),
                                      max_regression_ns=int(np.max(costs-baseline))),
                  checks=dict(baseline_costs_exact=512, baseline_bounds_exact=512,
                              expert_query_unique_row_checks=ownership_checks,
                              computed_work_exact=512, representative_traces=trace_checks),
                  counter_columns=['jobs','computed_coefficient_bytes','pages','waves','fullinputbytes','emittedoutputbytes','channel_ns'],
                  counters_mean_by_phase=counters.mean(0).tolist(), seconds=time.monotonic()-t0,
                  limits='Not hardware execution or numericquality. Aliascomputedrow totals are billing descriptors, not proof of nominalmodelshape. Independentrowowners supply1024 distinct originalGU rows; repeated/fringe outputs aredrained/drop. Zero runtime selector/scatter software is optimistic; fullGU A8 dependency retained. No strongcopycontrols/heldout/alllayer.')
    dump(OUT/'report.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['freeze','run'])
    args = parser.parse_args()
    freeze() if args.mode == 'freeze' else run()
