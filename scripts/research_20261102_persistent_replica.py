#!/usr/bin/env python3
"""Independent uncoded fixed-copy resource bounds on the chosen data layouts.

The foreground and background calendar match coded_latency.py.  Unlike the
earlier homogeneous closure, a 34-node maxflow handles unequal chip work.
One-wave tiles give integral assignments. Larger whole-row tiles may be split
by this relaxation, so these results are optimistic uncoded resource bounds,
not implemented replica schedules or an optimal-placement theorem.
"""
import argparse
from collections import Counter
import heapq
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow

import research_20261102_coded_latency as coded
from run_corpus_traces import ROOT, sha256

READ = 28000
DEST = ROOT / 'results/research-2026-11-02/round2/interference/persistent'
FIELDS = coded.FIELDS + ('host_input', 'service_ns')


def flow_service(loads, copies, release_ns=0, blocked=None):
    """Minimum NAND service in a fixed-copy wave relaxation, independent of I/O.

    Original-owner nodes send either original or disjoint single-alternate
    tile work to executor nodes. Executor capacity is floor((T-release)/tR).
    The optimum is on either the healthy or the one-blocked-chip lattice.
    """
    loads = np.asarray(loads, dtype=np.int64)
    copies = np.asarray(copies, dtype=np.int64)
    assert copies.shape == (16, 16) and np.all(copies.diagonal() == 0)
    assert np.all(copies.sum(axis=1) <= loads)
    if blocked is None or release_ns <= 0:
        blocked, release_ns = None, 0.
    initial = max((release_ns if c == blocked else 0) + loads[c] * READ
                  for c in range(16) if loads[c] > 0)
    total = int(loads.sum())
    if not np.any(copies):
        return float(initial), loads.copy()
    cut = (total + 15) // 16
    if initial <= cut * READ and release_ns == 0:
        return float(initial), loads.copy()
    template = np.zeros((34, 34), dtype=np.int64)
    template[0, 1:17] = loads
    template[1:17, 17:33] = copies
    for chip in range(16):
        template[1 + chip, 17 + chip] = loads[chip]

    def feasible(duration, return_flow=False):
        available = np.full(16, max(0, int(math.floor((duration + 1e-6) / READ))), dtype=np.int64)
        if blocked is not None:
            available[blocked] = max(0, int(math.floor((duration - release_ns + 1e-6) / READ)))
        capacity = template.copy()
        capacity[17:33, 33] = available
        solved = maximum_flow(csr_matrix(capacity), 0, 33)
        okay = int(solved.flow_value) == total
        if return_flow:
            return okay, np.asarray(solved.flow[17:33, 33].toarray()).ravel()
        return okay

    remainder = release_ns % READ if blocked is not None else 0.
    lo = max(0, cut - 1)
    hi = int(math.ceil(initial / READ))
    while lo < hi:
        mid = (lo + hi) // 2
        if feasible(mid * READ + remainder):
            hi = mid
        else:
            lo = mid + 1
    candidate = lo * READ
    if not feasible(candidate):
        candidate += remainder
    okay, placement = feasible(candidate, True)
    assert okay and candidate <= initial + 1e-5
    return float(candidate), placement


def independent_checks():
    rng = np.random.default_rng(2026110210)
    checks = 0
    for _ in range(80):
        jobs = [(int(rng.integers(0, 4)), int(rng.integers(0, 4))) for _ in range(6)]
        jobs = [(a, b if a != b else (b + 1) % 4) for a, b in jobs]
        loads = np.bincount([a for a, b in jobs], minlength=16)
        copies = np.zeros((16, 16), dtype=int)
        for a, b in jobs:
            copies[a, b] += 1
        for release in (0, 13000, 100000, 3000000):
            blocked = int(rng.integers(0, 4)) if release else None
            fast, _ = flow_service(loads, copies, release, blocked)
            exact = float('inf')
            for choices in itertools.product((0, 1), repeat=len(jobs)):
                work = np.zeros(16, dtype=int)
                for (a, b), alternate in zip(jobs, choices):
                    work[b if alternate else a] += 1
                finish = max((release if c == blocked else 0) + work[c] * READ
                             for c in range(16) if work[c] > 0)
                exact = min(exact, finish)
            assert abs(fast - exact) < 1e-5, (loads, copies, release, blocked, fast, exact)
            checks += 1
    return checks


def tile_specs(rows, columns):
    unit_rows = 65536 // math.gcd(columns, 65536)
    result = []
    for expert in range(len(rows)):
        cursor = 0
        for source, count in enumerate(rows[expert]):
            count = int(count)
            first = cursor
            for offset in range(0, count, unit_rows):
                nrows = min(unit_rows, count - offset)
                pages = coded.ceildiv(nrows * columns, 16384)
                result.append(dict(expert=expert, source=source, row_start=first + offset,
                    rows=nrows, bytes=nrows * columns, pages=pages, waves=coded.ceildiv(pages, 4)))
            cursor += count
    # Each source's tile stream preserves the original wave count exactly.
    counted = np.zeros_like(rows)
    for tile in result:
        counted[tile['expert'], tile['source']] += tile['waves']
    np.testing.assert_array_equal(counted, coded.ceildiv(coded.ceildiv(rows * columns, 16384), 4))
    return result


def fixed_copy_plan(rows, columns, frequencies, budget_bytes, budget_pages, kind):
    tiles = tile_specs(rows, columns)
    experts = len(rows)
    selected = []
    remaining = int(budget_bytes)
    remaining_pages = int(budget_pages)
    if kind == 'whole_frequency':
        for expert in np.argsort(-frequencies, kind='stable'):
            indices = [i for i, tile in enumerate(tiles) if tile['expert'] == expert]
            cost_bytes = sum(tiles[i]['bytes'] for i in indices)
            cost_pages = sum(tiles[i]['pages'] for i in indices)
            if cost_bytes <= remaining and cost_pages <= remaining_pages:
                selected.extend(indices)
                remaining -= cost_bytes
                remaining_pages -= cost_pages
    else:
        # Frequency-only weighted-fair partial allocation with source diversity.
        # No coactivation or future background interval enters this policy.
        by_expert = [[] for _ in range(experts)]
        for i, tile in enumerate(tiles):
            by_expert[tile['expert']].append(i)
        for expert, indices in enumerate(by_expert):
            indices.sort(key=lambda i: ((tiles[i]['source'] - expert) % 16, tiles[i]['row_start']))
        allocated = np.zeros(experts, dtype=float)
        positions = np.zeros(experts, dtype=int)
        weight = np.sqrt(np.maximum(frequencies, 1e-9))
        heap = []
        for expert, indices in enumerate(by_expert):
            if indices:
                tile = tiles[indices[0]]
                heapq.heappush(heap, (-weight[expert] / tile['bytes'], expert))
        while heap:
            _, expert = heapq.heappop(heap)
            index = by_expert[expert][positions[expert]]
            tile = tiles[index]
            if tile['bytes'] <= remaining and tile['pages'] <= remaining_pages:
                selected.append(index)
                remaining -= tile['bytes']
                remaining_pages -= tile['pages']
                allocated[expert] += tile['bytes']
            positions[expert] += 1
            if positions[expert] < len(by_expert[expert]):
                nxt = tiles[by_expert[expert][positions[expert]]]
                heapq.heappush(heap, (-weight[expert] / (allocated[expert] + nxt['bytes']), expert))
    expected_load = np.zeros(16)
    copy_bytes = np.zeros(16, dtype=np.int64)
    matrix = np.zeros((experts, 16, 16), dtype=np.int64)
    owners = []
    for index in selected:
        tile = tiles[index]
        choices = [c for c in range(16) if c != tile['source']]
        destination = min(choices, key=lambda c: (expected_load[c], copy_bytes[c], c))
        expected_load[destination] += frequencies[tile['expert']] * tile['waves']
        copy_bytes[destination] += tile['bytes']
        matrix[tile['expert'], tile['source'], destination] += tile['waves']
        owners.append(dict(**tile, alternate_chip=destination))
    actual = sum(o['bytes'] for o in owners)
    actual_pages = sum(o['pages'] for o in owners)
    assert actual <= budget_bytes and actual_pages <= budget_pages
    return matrix, dict(kind=kind, budget_bytes=int(budget_bytes), actual_copy_bytes=int(actual),
                        budget_pages=int(budget_pages), actual_copy_pages=int(actual_pages),
                        owners=owners, copy_bytes_per_chip=copy_bytes.tolist(),
                        tile_wave_grains=sorted({t['waves'] for t in tiles}),
                        exact_unit_wave_assignment=all(t['waves'] == 1 for t in tiles))


def ledger_at(cache, prefix, index):
    return {field: cache[prefix + field][index] for field in FIELDS}


def simulate(ledgers, copy_work, layers, background, cap_ns, clean_cache):
    now = 0.
    token_start = 0.
    latencies = []
    stats = Counter()
    for pi, base in enumerate(ledgers):
        busy = background.current(now)
        inputs, outputs = base['inputs'], base['outputs']
        prefix = 2000 + 250 * base['commands'].sum() + max(coded.ceildiv(base['host_input'], 8),
                    coded.ceildiv((inputs[:8] + inputs[8:]).max(), 2))
        post = max(coded.ceildiv(outputs.sum(), 8), coded.ceildiv((outputs[:8] + outputs[8:]).max(), 2))
        release, chip = 0., None
        if busy is not None:
            chip, end, event = busy
            ready = end if cap_ns is None else min(end, now + cap_ns)
            release = max(0., ready - now - prefix)
        if release == 0 and pi in clean_cache:
            duration, placement = clean_cache[pi]
        else:
            duration, placement = flow_service(base['waves'], copy_work[pi], release, chip)
            if release == 0:
                clean_cache[pi] = duration, placement
        # NAND dominates the original shared-ECC/PE pipeline in these shapes.
        assert np.all(base['service_ns'] <= base['waves'] * READ + 1e-6)
        total = prefix + duration + post
        if busy is not None:
            stats['phase_start_busy'] += 1
            stats['phase_start_busy_original_data_active'] += int(base['waves'][chip] > 0)
            stats['phase_start_busy_flow_executor_active'] += int(placement[chip] > 0)
            if cap_ns is not None and now + cap_ns < busy[1] and placement[chip] > 0:
                stats['suspend_bound_phases'] += 1
                stats['foreground_erase_service_debt_lower_bound_ns'] += int(placement[chip] * READ)
        stats['new_erase_start_inside_phase_uncertainty'] += int(background.future_start_crossed(now, now + total))
        now += total
        if (pi + 1) % (layers * 2) == 0:
            latencies.append((now - token_start) / 1e6)
            token_start = now
    a = np.asarray(latencies)
    return dict(mean_ms=float(a.mean()), p95_ms=float(np.quantile(a, .95)),
                p99_ms=float(np.quantile(a, .99)), total_elapsed_ms=now / 1e6,
                per_token_ms=latencies, stats=dict(stats))


def run(args):
    DEST.mkdir(parents=True, exist_ok=True)
    checks = independent_checks()
    coded_doc = json.loads((coded.DEST / 'summary.json').read_text())
    missing = set(args.models) - {m['model'] for m in coded_doc['models']}
    if missing:
        raise ValueError('The shared coded reference is incomplete for: ' + ', '.join(sorted(missing)))
    # Preserve the exact comparison snapshot even if another experiment later
    # writes a different summary in the source study directory.
    (DEST / 'coded_reference.json').write_text(json.dumps(coded_doc, indent=2) + '\n')
    result = []
    for name in args.models:
        started = time.time()
        frozen_path = ROOT / f'results/research-2026-11-02/projection-study/{name}.frozen_plan.json'
        frozen = json.loads(frozen_path.read_text())
        doc = next(m for m in coded_doc['models'] if m['model'] == name)
        with np.load(Path(frozen['train_path'])) as z:
            routes = z['routes']
        with np.load(coded.DEST / 'phase_ledgers' / (name + '.npz')) as z:
            cache = {key: z[key] for key in z.files}
        phases = frozen['phases']
        ledgers = [ledger_at(cache, 'data_', i) for i in range(len(cache['data_waves']))]
        token_count = len(cache['sample_indices'])
        expected = routes.reshape(-1, len(frozen['layers']), frozen['active'])[cache['sample_indices']]
        np.testing.assert_array_equal(expected, cache['selected'])
        ledgers = ledgers[:args.tokens * len(phases)]
        work = {}
        plan_receipts = {}
        for budget in ('quarter', 'same_as_coded', 'full'):
            for kind in ('whole_frequency', 'fair_partial_tiles'):
                matrices = []
                receipts = []
                for pi, phase in enumerate(phases):
                    rows = np.asarray(phase['chosen_rows'], dtype=np.int64)
                    li = frozen['layers'].index(phase['layer'])
                    frequencies = np.bincount(routes[:, :, li].ravel(), minlength=frozen['experts']) / (200 * 128)
                    data_bytes = int(rows.sum() * phase['columns'])
                    capacity = doc['capacity']['phases'][pi]
                    assert capacity['layer'] == phase['layer'] and capacity['phase'] == phase['name']
                    budget_bytes = data_bytes // 4 if budget == 'quarter' else \
                                   capacity['parity_weight_bytes'] if budget == 'same_as_coded' else data_bytes
                    data_pages = int(coded.ceildiv(rows * phase['columns'], 16384).sum())
                    budget_pages = data_pages // 4 if budget == 'quarter' else \
                                   capacity['parity_physical_pages'] if budget == 'same_as_coded' else data_pages
                    matrix, receipt = fixed_copy_plan(rows, phase['columns'], frequencies, budget_bytes, budget_pages, kind)
                    matrices.append(matrix)
                    receipts.append(dict(layer=phase['layer'], phase=phase['name'], **receipt))
                copy_work = []
                for ti, token in enumerate(cache['selected'][:args.tokens]):
                    for pi, phase in enumerate(phases):
                        selected = token[pi // 2]
                        copy_work.append(matrices[pi][selected].sum(axis=0))
                key = budget + '_' + kind
                work[key] = copy_work
                path = DEST / (name + '.' + key + '.frozen.json')
                path.write_text(json.dumps(dict(model=name, kind=kind, budget=budget,
                    base_projection_sha256=sha256(frozen_path),
                    fit_routes_sha256=sha256(Path(frozen['train_path'])), phases=receipts), separators=(',', ':')) + '\n')
                plan_receipts[key] = dict(path=str(path), sha256=sha256(path),
                    actual_copy_bytes=sum(r['actual_copy_bytes'] for r in receipts),
                    budget_bytes=sum(r['budget_bytes'] for r in receipts),
                    actual_copy_pages=sum(r['actual_copy_pages'] for r in receipts),
                    budget_pages=sum(r['budget_pages'] for r in receipts),
                    exact_all_tiles_unit_wave=all(r['exact_unit_wave_assignment'] for r in receipts))
        scenarios = []
        clean_cache = {key: {} for key in work}
        for length_ms in args.lengths:
            for duty in args.duties:
                background = coded.Background(length_ms * 1000000, duty)
                policies = {}
                for cap in args.caps:
                    cap_ns = None if cap < 0 else cap * 1000
                    label = 'none' if cap < 0 else str(cap_ns)
                    for key, copies in work.items():
                        value = simulate(ledgers, copies, len(frozen['layers']), background, cap_ns, clean_cache[key])
                        policies[key + '_cap' + label] = value
                    # Independent exact reproduction of the no-copy baseline.
                    zero = [np.zeros((16, 16), dtype=np.int64) for _ in ledgers]
                    value = simulate(ledgers, zero, len(frozen['layers']), background, cap_ns, {})
                    policies['chosen_wait_cap' + label] = value
                    if args.tokens == token_count:
                        original = next(s for s in doc['scenarios'] if s['erase_ms'] == length_ms and s['duty'] == duty)
                        expected_wait = original['policies']['chosen_wait_cap' + label + '_decode0']
                        np.testing.assert_allclose(value['per_token_ms'], expected_wait['per_token_ms'], rtol=0, atol=1e-7)
                scenarios.append(dict(erase_ms=length_ms, duty=duty, policies=policies))
                print(name, length_ms, duty, 'elapsed', time.time() - started, flush=True)
        result.append(dict(model=name, tokens=min(args.tokens, token_count), plans=plan_receipts,
                           phase_ledger_sha256=sha256(coded.DEST / 'phase_ledgers' / (name + '.npz')),
                           coded_actual_extra_weight_fraction=doc['capacity']['extra_weight_fraction'],
                           chosen_projection_sha256=sha256(frozen_path), scenarios=scenarios))
        output = dict(study='Persistent same-layout uncoded fixed-copy resource relaxation',
            source_sha256=sha256(Path(__file__)), coded_source_sha256=sha256(ROOT / 'scripts/research_20261102_coded_latency.py'),
            coded_reference_sha256=sha256(DEST / 'coded_reference.json'),
            independent_unit_job_bruteforce_checks=checks,
            assumptions=['Same chosen gate_up/down row ownership and 32 routing samples as coded study.',
                         'Identical absolute Background.current and phase-prefix/post cost frame.',
                         'Each uncoded tile has exactly one immutable alternate chip, no inference writes.',
                         'Maxflow can divide a multi-wave whole-row tile. Such costs are optimistic bounds, not executable schedules.',
                         'Each phase enforces both useful INT8-byte and physical-page capacity budgets. same_as_coded matches actual padded parity bytes/pages.',
                         'Extra copy descriptors, copy-input fanout changes and output-channel changes are omitted to strengthen uncoded controls.',
                         'Background caps are ideal read-availability bounds, not free erase progress. Foreground debt is a lower bound.',
                         'Busy intervals starting inside a phase are audit flags, matching coded phase-start admission model.',
                         'Static copy placement uses frequency-only whole hot experts or fair partial tiles; not optimal among all fixed layouts.',
                         'No measured NAND/PE/FTL or full-model latency claims.'], models=result)
        (DEST / 'summary.json').write_text(json.dumps(output, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=['granite', 'deepseek', 'qwen3'])
    parser.add_argument('--tokens', type=int, default=32)
    parser.add_argument('--lengths', nargs='+', type=int, default=[3, 5])
    parser.add_argument('--duties', nargs='+', type=float, default=[0, .02, .1, .25])
    parser.add_argument('--caps', nargs='+', type=int, default=[-1, 0, 10, 100, 1000])
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    if args.audit_only:
        print(independent_checks())
    else:
        run(args)
