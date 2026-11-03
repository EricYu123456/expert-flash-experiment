#!/usr/bin/env python3
"""Reuse the sealed independent scheduler on saved native/prospective/joint traces.

New bindings and mutation driver; NOT a new independent event executor. No
target builder, simulator, campaign, target audit, or target executor is called.
Page coordinates below reconstruct digests only; page-level PB deadlines were
verified earlier and are outside this saved-actual event audit.
"""
from collections import Counter, defaultdict
import copy
import datetime
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/research-2026-11-02/round3-cr-ipr'
OUT = BASE / 'joint-shared-full-ffn-independent'
CHECKER = ROOT / 'scripts/research_20261102_cr_ipr_full_ffn_audit.py'
BINDER = ROOT / 'scripts/research_20261102_g2_availability_full_ffn_audit.py'
TRAIN = ROOT / 'results/research-2026-11-02/round2/w8-format-training/granite4tiny/decode.routes.npz'
DIAGNOSTIC = BASE / 'prospective-routes/decode.routes.npz'
G2 = BASE / 'g2-availability-full-ffn'
JOINT = BASE / 'eight-pair-cbc-degree4'
DATA = dict(native=BASE / 'native-group-control/v2', prospective=BASE / 'prospective-full-ffn',
            joint_TRAIN=BASE / 'joint-shared-full-ffn/TRAIN512',
            joint_diagnostic=BASE / 'joint-shared-full-ffn/diagnostic-reuse')


def need(ok, msg):
    if not ok:
        raise RuntimeError(msg)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


def page(owner, j, plane, wl, block=0):
    return dict(page_id=f'{owner}.p{j}', owner=owner, page_index=j, plane=plane,
                block=int(block), wl=int(wl), bytes=16384)


def cut(stream):
    waves, pending = [], []
    for p in stream:
        if p['plane'] in [x['plane'] for x in pending]:
            waves.append(dict(pages=pending, source_order=[x['plane'] for x in pending]))
            pending = []
        pending.append(p)
    if pending:
        waves.append(dict(pages=pending, source_order=[x['plane'] for x in pending]))
    return waves


def common_pages(route, family, labels=None, group=None):
    """Address reconstruction only, independently of target recipe classes."""
    if family == 'g8_fixed':
        resident = np.flatnonzero(labels == group).tolist()
        need(len(resident) == 32, 'g8 balanced physical resident group')
        selected = [e for e in resident if e in route]
        stream = [page(f'e{e}', j, j % 4, 3 * resident.index(e) + j // 4)
                  for e in selected for j in range(12)]
        stream += [page('s', j, j % 4, 96 + j // 4) for j in range(12)]
        waves = [dict(pages=stream[k:k + 4], source_order=[0, 1, 2, 3])
                 for k in range(0, len(stream), 4)]
        return waves, None
    pool = []
    for e in range(64):
        if e in route:
            continue
        for j in range(6):
            if family == 'candidate':
                plane = [0, 1, 2, 0, 1, 3][j]
                wl = 2 * e + j // 3 if plane < 2 else e
            else:
                plane, wl = (e * 6 + j) % 4, (e * 6 + j) // 4
            pool.append(dict(page(f'e{e}', j, plane, wl), unselected=True,
                             dummy_unselected_expert=e))
    if family == 'RR':
        evens = [e for e in route if e % 2 == 0]
        odds = [e for e in route if e % 2]
        order = [e for pair in zip(evens, odds) for e in pair]
        order += evens[min(len(evens), len(odds)):] + odds[min(len(evens), len(odds)):]
        stream = [page(f'e{e}', j, (e * 6 + j) % 4, (e * 6 + j) // 4)
                  for e in order for j in range(6)]
        stream += [page('s', j, j % 4, 96 + j // 4) for j in range(12)]
        return cut(stream), pool
    stream, shared_count, shared_j = [], Counter(), 0
    def macro(e, offset):
        pp = [0, 1, 2] if offset == 0 else [0, 1, 3]
        return [page(f'e{e}', offset + j, p, 2 * e + offset // 3 if p < 2 else e)
                for j, p in enumerate(pp)]
    def shared(planes):
        nonlocal shared_j
        result = []
        for p in planes:
            result.append(page('s', shared_j, p, 64 + shared_count[p]))
            shared_count[p] += 1
            shared_j += 1
        return result
    for a, b, c in [route[:3], route[3:]]:
        d, cc = shared([3, 2, 3]), shared([2, 3, 2])
        stream += macro(a, 0) + d + macro(b, 0) + macro(a, 3) + macro(b, 3) + cc + macro(c, 0) + macro(c, 3)
    return [dict(pages=stream[k:k + 4], source_order=[p['plane'] for p in stream[k:k + 4]])
            for k in range(0, len(stream), 4)], pool


def digest(waves, columns, ranks, native=None, GU=False, pool=None):
    settings = dict(cache_copy_ns=500, source_policy='fixed' if native is not None or not GU else 'permutation',
                    issue_policy='strict_no_interval', row_bytes=columns, validate_row_stream=True,
                    unselected_policy='reject' if native is not None else 'drop_before_ecc_conditional')
    if native is not None:
        settings['array_mode'] = native
        payload = dict(waves=waves, settings=settings, ranks=ranks, GU=GU)
        encoded = json.dumps(payload, sort_keys=True)
    else:
        payload = dict(waves=waves, settings=settings, ranks=ranks, pool=pool)
        encoded = json.dumps(payload, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(encoded.encode()).hexdigest(), settings


def native_waves_contract(waves, route=None):
    for wave in waves:
        pp = wave['pages']
        need(len(pp) == 4 and wave['source_order'] == [0, 1, 2, 3], 'whole4plane fixed0123 wave')
        need([p['plane'] for p in pp] == [0, 1, 2, 3], 'native source-plane correspondence')
        need(len({(p['block'], p['wl']) for p in pp}) == 1, 'native sameblock sameWL across4planes')
        need(not any(p.get('unselected', False) for p in pp), 'native rejects unselected read')
        if route is not None:
            need(all(p['owner'] == 's' or int(p['owner'][1:]) in route for p in pp), 'native read of unselected expert')


def joint_residents():
    layout = read(JOINT / 'layout.frozen.json')
    supports = layout['supports']
    removed = [{0, 1, 2, 3}, {4, 5, 6, 7}, {0, 2, 4, 6}, {1, 3, 5, 7}, {0, 1, 4, 5}, {2, 3, 6, 7}]
    canonical = [list(s) for s in itertools.combinations(range(8), 4) if set(s) not in removed]
    need(supports == canonical, 'joint frozen supports differ from certified degree4 incidence')
    expected = {}
    for group in range(8):
        experts = [e for e in range(64) if group in supports[e]]
        need(len(experts) == 32, 'joint pergroup32experts')
        for rank, owner in enumerate(experts + ['S0', 'S1']):
            half = int(owner[1]) if isinstance(owner, str) else None
            expert = 64 if half is not None else owner
            block = 0 if rank < 10 else 1
            wl = 128 + 6 * rank if rank < 10 else 6 * (rank - 10)
            for local in range(2):
                chip = 2 * group + local
                p = dict(expert=expert, copy=0, resident_copy=group, shared_half=half,
                         resident_job_id=f'{owner if half is not None else "e" + str(owner)}.g{group}.c{chip}',
                         chip=chip, rows=384 if half is not None else 768,
                         columns=1024 if half is not None else 512, block=block, wl_start=wl,
                         pages=24, row_start=half * 768 + local * 384 if half is not None else local * 768,
                         reserved_WL=6)
                expected[p['resident_job_id']] = p
    sealed = read(JOINT / 'resident_jobs.frozen.json')
    need(len(sealed) == 544 and {p['resident_job_id']: p for p in sealed} == expected,
         'all544joint resident jobs exact independent coordinate formula')
    occupied = set()
    for p in expected.values():
        need(p['wl_start'] + 6 <= 192 and p['rows'] * p['columns'] == 24 * 16384, 'joint native extent')
        for j in range(24):
            address = (p['chip'], p['block'], j % 4, p['wl_start'] + j // 4)
            need(address not in occupied, 'joint resident Down address overlap')
            occupied.add(address)
    return expected, np.array([[g in s for g in range(8)] for s in supports], dtype=bool)


def cap1_feasible(route, adjacency):
    masks = {0}
    for e in sorted(map(int, route)):
        masks = {m | (1 << g) for m in masks for g in np.flatnonzero(adjacency[e]) if not m & (1 << g)}
        if not masks:
            return False
    return True


def bind(record, route, family, expected=None, adjacency=None, cap=0, native=None, groups=None,
         joint=False, triple=False):
    ns = record['nodes']
    route = sorted(map(int, route))
    chosen = route + [64]
    ranks = {f'e{e}': i for i, e in enumerate(route)}
    ranks['s'] = 6
    rows, work = Counter(), Counter()
    for n in ns:
        if n['kind'] == 'output_ONFI':
            rows[n['phase'], n['expert_rank']] += n['bytes'] // 4
        if n['resource'] >= 0:
            work[n['kind'] + '_busy_ns'] += n['duration']
    need(dict(work) == record['audit']['resource_work_ns'], 'actual busy resource accounting (timers excluded)')
    for rank in range(7):
        need(rows[0, rank] == (2048 if rank == 6 else 1024) and rows[1, rank] == 1536, 'exact logical fullGU/Down row total')
    actual, byexpert, expertgroups, chiploads = {}, defaultdict(list), {}, Counter()
    GUaddresses = []
    for job in record['jobs']:
        pipe = job['pipeline']
        need(pipe['finite_PB_feasible'] and pipe['settings']['cache_copy_ns'] == 500, 'certified cache/PB settings')
        need(pipe['pages'] * 16384 == pipe['bytes'] == pipe['ecc_bytes'] and pipe['ecc_pages'] == pipe['pages'], 'useful/ECC byte conservation')
        need(pipe['physical_read_bytes'] == 16384 * pipe['physical_read_pages'], 'physical-read byte conservation')
        if job['phase'] == 0:
            need(job['columns'] == 1536 and 0 <= job['chip'] < 16, 'GU actual chip/full input')
            group = job['chip'] // 8 if family == 'g8_fixed' else None
            waves, pool = common_pages(route, family, groups, group)
            key, settings = digest(waves, 1536, ranks, native=native, GU=True, pool=pool)
            need(job['recipe_signature'] == key, 'GU physical group/address/selected expert recipe binding')
            perrank = Counter()
            for i in job['output_ONFI']:
                perrank[ns[i]['expert_rank']] += ns[i]['bytes'] // 4
            if family == 'g8_fixed':
                desired = {ranks[f'e{e}']: 128 for e in route if groups[e] == group}
                desired[6] = 128
                need(dict(perrank) == desired, 'g8 8chip128rows shard/group/scatter ownership')
            else:
                need(dict(perrank) == {**{i: 64 for i in range(6)}, 6: 128}, 'g4 16chip64rows commonGU ownership')
            GUaddresses.append(dict(chip=job['chip'], physical_group=group, recipe_signature=key,
                                    logical_shard_row_start=(job['chip'] % 8) * 128 if family == 'g8_fixed' else job['chip'] * 64))
            if native is not None:
                native_waves_contract(waves, route)
                need(pipe['no_IPR_or_dummy_or_drop'] and pipe['policy'] in ['strict', 'GU_explicit_cold_reset'], 'native no shape/drop/IPR contract')
                need(pipe['pages'] == pipe['physical_read_pages'] == len(waves) * 4 and pipe['waves'] == len(waves), 'native whole selected waves no dummy/drop')
                if pipe['policy'] == 'GU_explicit_cold_reset':
                    settings['issue_policy'] = 'cold_reset'
                need(pipe['settings'] == settings and pipe['source_policy'] == 'fixed', 'native GU array/source/settings contract')
                if native == 'ordinary':
                    need(pipe['policy'] == 'strict' and pipe['cold_waves'] == pipe['waves'] and pipe['cr_waves'] == 0, 'ordinary allcold no shape/reset/drop')
                    need(ns[job['compute']]['duration'] == 28500 * pipe['waves'] + 10240, 'ordinary complete-wave kernel duration')
            else:
                if pipe['policy'] == 'GU_explicit_cold_reset':
                    settings['issue_policy'] = 'cold_reset'
                need(pipe['settings'] == settings and pipe['source_policy'] == 'permutation', 'conditional GU source/settings binding')
            continue
        p = job['physical_down_job']
        rank = ns[job['command']]['expert_rank']
        e = chosen[rank]
        if triple:
            g = rank // 2 if e < 64 else 3
            block, wl = (1, 3 * e) if e < 64 else (0, 128)
            desired = dict(expert=e, copy=g if e < 64 else 0, chip=job['chip'], rows=384,
                           columns=512 if e < 64 else 1024, block=block, wl_start=wl,
                           row_start=(job['chip'] % 4) * 384, pages=12 if e < 64 else 24,
                           reserved_WL=3 if e < 64 else 6)
            need(job['chip'] // 4 == g and p == desired, 'triple selected resident copy/chip/block/WL/rows binding')
        else:
            rid = p['resident_job_id']
            need(rid in expected and p == expected[rid], 'selected resident_job_id/chip/block/WL/row/C differs from frozen layout')
        key = (rank, p['chip'])
        need(key not in actual, 'duplicate selected Down job')
        actual[key] = p
        need(p['expert'] == e and job['chip'] == p['chip'] and job['columns'] == p['columns'], 'actual Down chip/expert/full operand binding')
        need(sum(ns[i]['bytes'] for i in job['output_ONFI']) == p['rows'] * 4 and pipe['bytes'] == p['rows'] * p['columns'], 'complete native Down band rows/coeff bytes')
        waves = [dict(pages=[page('v', j, j % 4, p['wl_start'] + j // 4, p['block'])
                            for j in range(k, k + 4)], source_order=[0, 1, 2, 3])
                 for k in range(0, p['pages'], 4)]
        recipe, settings = digest(waves, p['columns'], {'v': 0}, native=native)
        need(job['recipe_signature'] == recipe, 'actual native recipe digest differs from certified absolute block/WL pages')
        need(pipe['settings'] == settings and pipe['source_policy'] == 'fixed', 'native Down settings/array_mode binding')
        need(pipe['policy'] == ('strict' if native is not None else 'private_strict'), 'strict unshaped Down policy')
        need(pipe['pages'] == pipe['physical_read_pages'] == pipe['ecc_pages'] == p['pages'] and pipe['waves'] == p['reserved_WL'], 'Down selected wholewave accounting')
        if native == 'ordinary':
            need(pipe['cold_waves'] == pipe['waves'] and pipe['cr_waves'] == 0, 'ordinary Down allcold')
            need(ns[job['compute']]['duration'] == 28500 * pipe['waves'] + 10240, 'ordinary Down kernel duration')
        else:
            need(pipe['cold_waves'] == 1 and pipe['cr_waves'] == pipe['waves'] - 1, 'Down onecold per independent native job')
            need(ns[job['compute']]['duration'] == 28500 + 10240 * pipe['waves'], 'certified native CR Down duration')
        if native is not None:
            native_waves_contract(waves)
            need(pipe['no_IPR_or_dummy_or_drop'], 'native Down no drop/shape/IPR')
        byexpert[e].append(p)
        chiploads[p['chip']] += 1
        if not triple:
            group = p['resident_copy']
            if joint or e < 64:
                need(p['chip'] // 2 == group, 'actual Down pair binding')
            if e < 64:
                need(adjacency[e, group], 'unavailable routed copy selected')
                need(e not in expertgroups or expertgroups[e] == group, 'routed expert split across copy groups')
                expertgroups[e] = group
    need(len(GUaddresses) == 16 and len({j['chip'] for j in GUaddresses}) == 16, 'oneGU job per physical chip')
    for e in chosen:
        js = byexpert[e]
        need(len(js) == (4 if e == 64 or triple else 2), 'complete private/shared Down band count')
        cursor = 0
        for p in sorted(js, key=lambda p: p['row_start']):
            need(p['row_start'] == cursor, 'selected Down scatter gap/overlap')
            cursor += p['rows']
        need(cursor == 1536, 'selected Down complete1536rows')
    if joint:
        need(len(actual) == 16 and len(chiploads) == 16 and set(chiploads.values()) == {1}, 'joint exactlyoneDown perphysicalchip')
        loads = Counter(expertgroups.values())
        need(len(loads) == 6 and set(loads.values()) == {1}, 'joint routed sixdistinctpairs')
        shared = byexpert[64]
        halfgroups = {}
        for half in [0, 1]:
            js = [p for p in shared if p['shared_half'] == half]
            need(len(js) == 2 and {p['columns'] for p in js} == {1024}, 'shared twohalves use completeC1024')
            need({p['row_start'] for p in js} == {half * 768, half * 768 + 384}, 'sharedhalf canonical row positions')
            need(len({p['resident_copy'] for p in js}) == 1, 'sharedhalf split across groups')
            halfgroups[half] = js[0]['resident_copy']
        need(set(halfgroups.values()) == set(range(8)) - set(expertgroups.values()), 'shared halves occupy exactlyunusedpairs')
    elif not triple:
        need(len(actual) == 16 and max(Counter(expertgroups.values()).values()) <= cap, 'g2 cap1/cap2 complete assignment')
    return dict(selected_Down_jobs=len(actual), selected_routed_groups=expertgroups,
                physical_chip_Down_job_counts=dict(chiploads), exact_full_rows=18944,
                shared_rows=1536, shared_columns=1024, GU_address_bindings=GUaddresses)


def stats(values):
    return dict(mean=float(values.mean()), p99=float(np.percentile(values, 99)),
                minimum=float(values.min()), maximum=float(values.max()))


def array_audit(kind, arrays, report, routes, maps, jointadj):
    results = []
    for rec in report['records']:
        name = rec['name']
        cost, fifo = arrays[name + '.cost'], arrays[name + '.fifo']
        need(cost.shape == routes.shape[:2] and np.all(cost >= 0) and np.all((fifo >= 0) & (fifo <= 128)), 'all saved query arrays complete/FIFO128')
        need(rec.get('completed_events', rec.get('events')) == cost.size, 'all array sample count')
        need(stats(cost) == rec['cost_ns'] and stats(cost.sum(axis=0)) == rec['three_layer_sum_ns'] and stats(fifo) == rec['FIFO_peak_bytes'], 'saved report distribution arithmetic')
        fallbacks = None
        if kind == 'prospective':
            plan = name.split('.')[1]
            capacity = arrays[name + '.matching_capacity']
            expected = np.zeros_like(cost) if plan == 'g4triples' else np.array([
                [1 if cap1_feasible(route, maps[plan][layer]) else 2 for route in routes[layer]]
                for layer in range(3)], dtype=np.int64)
            need(np.array_equal(expected, capacity), 'all768 arrays bound to correct frozen route/support minimum capacity')
            fallbacks = int(np.count_nonzero(capacity == 2))
            need(fallbacks == rec['fallback_events'], 'all768 fallback count')
        elif kind.startswith('joint'):
            need(all(cap1_feasible(route, jointadj) for layer in routes for route in layer), 'joint allsaved routes match certified supports')
            need(rec['matching_failures'] == 0 and rec['payload_MiB'] == 303, 'joint matching/capacity accounting')
        results.append(dict(name=name, entries=int(cost.size), mean_ns=float(cost.mean()),
                            all_FIFO128=True, fallback_entries=fallbacks,
                            scope='Array arithmetic and route/map capacity binding; these entries were not all event-replayed.'))
    return results


def main():
    began = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    need(not (OUT / 'manifest.frozen.json').exists(), 'final audit directory already sealed')
    checker = module('joint_reused_independent_scalar', CHECKER)
    binder = module('joint_reused_binding_helpers', BINDER)
    available = {k: p for k, p in DATA.items() if (p / 'report.json').exists()}
    need(len(available) == 4, 'all four saved campaign reports required before final replay/seal')
    with np.load(TRAIN) as z:
        train = z['routes'].transpose(2, 0, 1, 3).reshape(40, 512, 6)[:3]
    with np.load(DIAGNOSTIC) as z:
        diagnostic = z['routes'].transpose(2, 0, 1, 3).reshape(40, 256, 6)[:3]
    with np.load(BASE / 'full-ffn/g8-addendum/group_maps.frozen.npz') as z:
        groups = z['learned'].copy()
    with np.load(G2 / 'support_maps.frozen.npz') as z:
        fitted = z['selected'].copy()
    with np.load(BASE / 'g2-availability-headroom/selected_adjacency.frozen.npz') as z:
        maps = dict(g2_fitted=fitted, g2_IDslot=z['IDslot'].copy(), g2_seed=z['best_single_seed'].copy())
    residents = {name: [binder.expected_residents(adj) for adj in matrix] for name, matrix in maps.items()}
    expected_joint, jointadj = joint_residents()
    joint_frozen = read(JOINT / 'verification_report.json')
    need(joint_frozen['status'] == 'PASS', 'prior independent joint builder verification receipt')
    paths = [Path(__file__).resolve(), CHECKER, BINDER, TRAIN, DIAGNOSTIC,
             JOINT / 'layout.frozen.json', JOINT / 'resident_jobs.frozen.json', JOINT / 'verification_report.json',
             JOINT / 'verification_manifest.json', BASE / 'full-ffn/g8-addendum/group_maps.frozen.npz',
             BASE / 'g2-availability-headroom/selected_adjacency.frozen.npz', G2 / 'support_maps.frozen.npz']
    hashes = {str(p): sha(p) for p in paths}
    campaigns, allfiles = {}, []
    for kind, directory in available.items():
        protocol_path = (directory.parent if kind.startswith('joint') else directory) / 'protocol.frozen.json'
        protocol = read(protocol_path)
        for p, h in protocol['inputs_sha256'].items():
            need(sha(p) == h, 'sealed campaign input changed: ' + p)
            hashes[p] = h
        files = sorted(directory.glob('trace.*json'))
        for p in [protocol_path, directory / 'report.json', directory / 'service_arrays.npz', *files]:
            hashes[str(p)] = sha(p)
        with np.load(directory / 'service_arrays.npz') as z:
            arrays = {key: z[key].copy() for key in z.files}
        route_array = diagnostic if kind in ['prospective', 'joint_diagnostic'] else train
        campaigns[kind] = dict(arrays=arrays, routes=route_array, report=read(directory / 'report.json'), files=files)
        allfiles += [(kind, p) for p in files]
    need(len(campaigns['native']['files']) == 12 and len(campaigns['prospective']['files']) == 60,
         'actual saved native12/prospective60 inventory')
    write(OUT / 'protocol.frozen.json', dict(status='frozen_before_saved_event_replay', inputs_sha256=hashes,
          distinction='New binding/mutation driver; reused independent scalar_execute, semantic_audit, critical_path. No new independent scheduler claim.',
          scope='Every saved representative in four completed campaigns, not every TRAIN/diagnostic query. All summary arrays read, not event-replayed.'))
    summaries = {kind: array_audit(kind, c['arrays'], c['report'], c['routes'], maps, jointadj)
                 for kind, c in campaigns.items()}
    records, critical, times = [], {}, {}
    for index, (kind, path) in enumerate(allfiles):
        parts = path.stem.split('.')
        layer = int(parts[1][1:])
        if kind == 'prospective':
            family, plan, copypart, priority, sample, qpart = parts[2:]
            query = int(qpart[1:]); array_mode = None; native = None
            config = '.'.join(parts[2:6])
        elif kind == 'native':
            array_mode, plan, copypart, priority = parts[2:]
            family, query, native, sample = 'g8_fixed', 0, array_mode, 'first_saved'
            config = '.'.join(parts[2:])
        else:
            family, array_mode, plan, copypart, priority = parts[2:]
            query, sample = 0, 'first_saved'
            native = array_mode if family == 'g8_fixed' else None
            config = '.'.join(parts[2:])
        campaign = campaigns[kind]
        saved = read(path)
        route = campaign['routes'][layer, query]
        need(saved['service']['input_copies'] == int(copypart[5:]) and saved['service']['down_first'] == int(priority == 'DF'), 'file/service binding')
        own, cost = checker.scalar_execute(saved['nodes'], saved['service']['down_first'])
        want = np.array([[n['start_ns'], n['end_ns']] for n in saved['nodes']], dtype=np.int64)
        need(np.array_equal(own, want) and cost == saved['cost_ns'], 'every node saved start/end differs from independent scalar replay')
        semantic = checker.semantic_audit(saved, own)
        need(semantic['FIFO128_valid'], 'representative output FIFO exceeds128')
        isjoint = kind.startswith('joint')
        triple = plan == 'g4triples'
        mapname = 'g2_fitted' if plan == 'g2availability' else plan
        adj = jointadj if isjoint else None if triple else maps[mapname][layer]
        resident = expected_joint if isjoint else None if triple else residents[mapname][layer]
        capacity = 0 if triple else 1 if cap1_feasible(route, adj) else 2
        binding = bind(saved, route, family, resident, adj, capacity, native, groups[layer], isjoint, triple)
        cp = binder.critical_path(saved)
        need(cost == int(campaign['arrays'][config + '.cost'][layer, query]) and semantic['peak_FIFO_bytes'] == int(campaign['arrays'][config + '.fifo'][layer, query]), 'saved trace exact768/1536 array query binding')
        if sample == 'fallback':
            need(capacity == 2 and max(Counter(binding['selected_routed_groups'].values()).values()) == 2,
                 'fallback representative must execute all cap2jobs')
        key = kind + '.' + path.name
        records.append(dict(campaign=kind, path=str(path), sha256=sha(path), layer=layer, query_index=query,
                            config=config, sample=sample, matching_capacity=capacity, route=route.tolist(),
                            cost_ns=cost, **semantic, binding=binding))
        critical[key] = cp
        times[key] = own
        if (index + 1) % 12 == 0:
            print(json.dumps(dict(replayed=index + 1, total=len(allfiles), last=key)), flush=True)
    mutations = []
    def reject(name, original, change, record_info):
        bad = copy.deepcopy(original)
        change(bad)
        try:
            bind(bad, **record_info)
        except RuntimeError as error:
            mutations.append(dict(name=name, detected=True, diagnostic=str(error)))
        else:
            raise RuntimeError('undetected negative mutation: ' + name)
    joint_record = next(r for r in records if r['campaign'] == 'joint_TRAIN' and r['config'] == 'candidate.cr.joint8.input1.GF')
    baseline = read(joint_record['path'])
    info = dict(route=joint_record['route'], family='candidate', expected=expected_joint, adjacency=jointadj,
                cap=1, native=None, groups=groups[joint_record['layer']], joint=True)
    def down(r):
        return next(j for j in r['jobs'] if j['phase'] == 1)
    def shared(r):
        return next(j for j in r['jobs'] if j['phase'] == 1 and j['physical_down_job']['expert'] == 64)
    reject('shared_half_row_shift', baseline, lambda r: shared(r)['physical_down_job'].__setitem__('row_start', shared(r)['physical_down_job']['row_start'] + 1), info)
    reject('shared_half_incomplete_C512', baseline, lambda r: shared(r)['physical_down_job'].__setitem__('columns', 512), info)
    reject('wrong_actual_chip', baseline, lambda r: down(r).__setitem__('chip', (down(r)['chip'] + 2) % 16), info)
    reject('wrong_native_block_metadata', baseline, lambda r: down(r)['physical_down_job'].__setitem__('block', 1 - down(r)['physical_down_job']['block']), info)
    def recipeblock(r):
        job = down(r);p = copy.deepcopy(job['physical_down_job']);p['block'] = 1 - p['block']
        ww = [dict(pages=[page('v', j, j % 4, p['wl_start'] + j // 4, p['block']) for j in range(k, k + 4)], source_order=[0, 1, 2, 3]) for k in range(0, 24, 4)]
        job['recipe_signature'] = digest(ww, p['columns'], {'v': 0})[0]
    reject('wrong_actual_native_recipe_block_metadata_unchanged', baseline, recipeblock, info)
    ordinary_record = next(r for r in records if r['campaign'] == 'native' and r['config'] == 'ordinary.g2availability.input1.GF')
    ordinary = read(ordinary_record['path']);layer = ordinary_record['layer']
    ordinfo = dict(route=ordinary_record['route'], family='g8_fixed', expected=residents['g2_fitted'][layer],
                   adjacency=fitted[layer], cap=ordinary_record['matching_capacity'], native='ordinary', groups=groups[layer])
    reject('ordinary_array_mode_changed_to_CR', ordinary, lambda r: down(r)['pipeline']['settings'].__setitem__('array_mode', 'cr'), ordinfo)
    reject('ordinary_GU_unselected_policy_drop', ordinary, lambda r: r['jobs'][0]['pipeline']['settings'].__setitem__('unselected_policy', 'drop_before_ecc_conditional'), ordinfo)
    reject('ordinary_GU_shape_or_dummy_count', ordinary, lambda r: r['jobs'][0]['pipeline'].__setitem__('physical_read_pages', r['jobs'][0]['pipeline']['physical_read_pages'] + 4), ordinfo)
    reject('wrong_actual_GU_physical_group', ordinary, lambda r: r['jobs'][0].__setitem__('chip', 8), ordinfo)
    # Shared row mutation uses a value different from whichever half was first.
    need(len(mutations) == 9, 'all requested physical/native mutations detected')
    need(hashes == {p: sha(p) for p in hashes}, 'sealed input changed during independent replay')
    np.savez_compressed(OUT / 'independent_node_times.npz', **times)
    write(OUT / 'critical_paths.json', critical)
    total = Counter()
    for r in records:
        for key in ['nodes', 'edges', 'jobs', 'input_ONFI_bytes', 'output_bytes']:
            total[key] += r[key]
    report = dict(status='pass_all_saved_representatives', utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  source_sha256=sha(__file__), reused_scalar_semantic_checker_sha256=sha(CHECKER),
                  reused_critical_path_helper_sha256=sha(BINDER), trace_counts=dict(Counter(r['campaign'] for r in records)),
                  total_traces=len(records), totals=dict(total), all_node_start_end_exact=True,
                  all_FIFO128_valid=True, all_saved_physical_bindings_exact=True,
                  fallback_traces=sum(r['sample'] == 'fallback' for r in records),
                  joint_resident_jobs_reconstructed=544, arrays=summaries, records=records, mutations=mutations,
                  development_errata=read(OUT / 'development_errata.json') if (OUT / 'development_errata.json').exists() else [],
                  elapsed_seconds=time.monotonic() - began,
                  source_distinction='New binding driver and requested negative controls; reused prior independently implemented scalar executor, semantic/FIFO checker and critical-path helper, not a newly independent implementation.',
                  limits=['Only saved representative traces were event-replayed. Every saved summary entry was arithmetically checked and prospective route/map minimum capacities checked, not every query event-replayed.',
                          'Native v2 has12 actual saved traces rather than24 because normalized graph cache suppresses duplicate q0 saves. Joint saved counts similarly reflect cache; missing files were not regenerated.',
                          'Fixed conditional service model: 6.4GB/s implicit MAC/ECC throughput, cold28us including unidentified phases, CR9.7us, overwrite/copy/finitePB/output ABI assumptions. No flash measurement, quality or original-model numerical equivalence.',
                          'Candidate/RR GU permutation and preECCdrop/shaping remain conditional. Native g8 uses fixed0123 sameWL selected4plane waves, rejects unselected pages, and pays explicit CR coldreset; ordinary uses allcold28us model, not all architectures.',
                          'Recipe payload address digests rebind already verified kernels; this audit does not independently re-prove every page-level PB deadline.',
                          'Joint design follows observed prospective failures; joint diagnostic replay is diagnostic reuse, not a fresh heldout. Known CBC/group replication, no new theorem or globalgeometry optimum.',
                          'Matching computation runtime is excluded from service costs. Absolute page/row metadata is certified; saved trace alone is not actual-device physical programming or coefficient execution.'])
    write(OUT / 'report.json', report)
    text = ('# Independent saved actual trace audit\n\n'
            f'PASS: {len(records)} saved traces; every node start/end and makespan exactly replays the previously independent scalar executor. All128B continuous-drain FIFO, resource/chip leases, fullGU→A8→privateDown→combine dependencies and exact18,944outputrows pass.\n\n'
            f'Trace inventory: {dict(Counter(r["campaign"] for r in records))}. Prospective includes48 first plus12 fallback traces; all cap2 jobs finish with full rows. Native v2 saved12traces because cache merged q0 graphs; no missing representatives were regenerated. Summary arrays are arithmetic/route-binding checks only, not full event replay.\n\n'
            'All544joint resident jobs independently reconstruct exact jobID/group/chip/block/WL/row/C. Selected sharedS0/S1 provide1536rows total with completeC1024; each joint chip runs exactlyoneDown. Native g8 GU binds the learned physical8chipgroups, fixed addresses and128row/chip scatter; all selected waves are whole0123/sameWL, with no dummy/drop/shape. Ordinary is entirely cold.\n\n'
            'Nine negative controls reject sharedhalf row/C, actualchip, nativeblock including changed recipe digest with unchanged metadata, ordinaryarray_mode, drop, extra physicalwaves, and wrongGUgroup.\n\n'
            'This is a new binding/mutation driver reusing the existing independent scalar/semantic/FIFO/criticalpath checkers. No new independent executor and no whole campaign rerun. Kernel digests bind already verified recipes; page-level deadlines are not re-proved. Conditional service model and existing source/drop/phase/MAC assumptions remain. Standard CBC/group replication; no newtheorem/hardware/quality/globaloptimality claim. Joint diagnostic uses already observed prompts after design, not fresh validation.\n\n'
            'Reproduce: `python scripts/research_20261102_joint_shared_full_ffn_audit.py` in a new/unsealed output directory. A sealed manifest intentionally prevents rerun in place. See report.json for pertrace detail and critical_paths.json for exact resource-augmented chains.\n')
    (OUT / 'README.md').write_text(text)
    outputs = sorted(p for p in OUT.iterdir() if p.is_file() and p.name != 'manifest.frozen.json')
    write(OUT / 'manifest.frozen.json', dict(status='sealed', inputs_sha256=hashes,
          outputs_sha256={str(p): sha(p) for p in outputs}, new_source_sha256=sha(__file__),
          no_further_inplace_changes=True))
    print(json.dumps(dict(status='PASS', traces=len(records), report_sha256=sha(OUT / 'report.json'),
                          source_sha256=sha(__file__), manifest_sha256=sha(OUT / 'manifest.frozen.json'))), flush=True)


if __name__ == '__main__':
    main()
