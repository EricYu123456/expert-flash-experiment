#!/usr/bin/env python3
"""Replay saved final partition/replica representatives with the old independent oracle.

This is a new driver and extension checker, not a new independent scheduler.
The previously frozen independent scalar_execute/semantic_audit are reused.
No target graph builder, scheduler or target audit is imported or executed.
"""
from collections import Counter, defaultdict
import copy
import datetime
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/research-2026-11-02/round3-cr-ipr'
OUT = BASE / 'final-extensions-independent'
CHECKER = ROOT / 'scripts/research_20261102_cr_ipr_full_ffn_audit.py'
PART = BASE / 'down-headroom/final-run'
REPLICA = BASE / 'down-replica-full-ffn'
ROUTES = ROOT / 'results/research-2026-11-02/round2/w8-format-training/granite4tiny/decode.routes.npz'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def write(p, x):
    Path(p).write_text(json.dumps(x, indent=2, ensure_ascii=False) + '\n')


def require(ok, msg):
    if not ok:
        raise RuntimeError(msg)


def load_checker():
    spec = importlib.util.spec_from_file_location('reused_independent_checker', CHECKER)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def parsed_name(p):
    layer, family, menu, copies, priority = p.name.removeprefix('trace.').removesuffix('.json').split('.')
    return int(layer[1:]), family, menu, int(copies[5:]), priority


def extension_audit(record, route, menu_rows=None, map_jobs=None, triple=False):
    """Exact complete rows, resource work and selected resident-job binding."""
    nodes = record['nodes']
    row_counts = Counter()
    work = Counter()
    busy = Counter()
    for n in nodes:
        if n['kind'] == 'output_ONFI':
            row_counts[n['phase'], n['expert_rank']] += n['bytes'] // 4
        if n['resource'] >= 0:
            work[n['kind'] + '_busy_ns'] += n['duration']
            busy[n['resource']] += n['duration']
    require(dict(work) == record['audit']['resource_work_ns'], 'saved busy work excludes timers incorrectly')
    for rank in range(7):
        require(row_counts[0, rank] == (2048 if rank == 6 else 1024), 'exact logical GU row count')
        require(row_counts[1, rank] == 1536, 'exact logical Down row count')
    require(sum(row_counts.values()) == 18944, 'exact full-FFN row count')
    chosen = sorted(map(int, route)) + [64]
    expected = {}
    if map_jobs is not None:
        for rank, e in enumerate(chosen):
            copy_id = rank // 2 if triple and e != 64 else 0
            for job in map_jobs:
                if job['expert'] == e and job['copy'] == copy_id:
                    key = (rank, job['chip'])
                    require(key not in expected, 'duplicate selected resident job')
                    expected[key] = job
    actual = {}
    intervals = defaultdict(list)
    physical_bindings = 0
    for job in record['jobs']:
        pipe = job['pipeline']
        require(pipe['finite_PB_feasible'], 'non-executable PB recipe')
        require(pipe['settings']['cache_copy_ns'] == 500, 'unexpected cache-copy timing')
        require(pipe['bytes'] == pipe['pages'] * 16384 == pipe['ecc_bytes'], 'useful coefficient byte conservation')
        require(pipe['physical_read_bytes'] == pipe['physical_read_pages'] * 16384, 'physical-read byte conservation')
        require(pipe['ecc_pages'] == pipe['pages'], 'ECC useful-page conservation')
        if job['phase'] == 0:
            continue
        require(pipe['source_policy'] == 'fixed' and pipe['policy'] == 'private_strict', 'private Down fixed-source contract')
        require(pipe['cold_waves'] == 1 and pipe['cr_waves'] == pipe['waves'] - 1, 'native private-job cold/CR accounting')
        rank = nodes[job['command']]['expert_rank']
        key = (rank, job['chip'])
        require(key not in actual, 'duplicate selected rank/chip job')
        output_rows = sum(nodes[i]['bytes'] for i in job['output_ONFI']) // 4
        actual[key] = output_rows
        require(output_rows * job['columns'] == pipe['bytes'], 'native full-row exact size without fringe')
        if menu_rows is not None:
            require(output_rows == int(menu_rows[chosen[rank], job['chip']]), 'Down chip row allocation differs from frozen menu')
        if map_jobs is not None:
            require(key in expected, 'unexpected/unselected physical Down job')
            p = job['physical_down_job']
            require(p == expected[key], 'physical Down trace differs from frozen resident map')
            require(p['expert'] == chosen[rank] and p['columns'] == job['columns'] and p['chip'] == job['chip'], 'private operand/expert/chip binding')
            require(p['rows'] == output_rows and p['pages'] == pipe['pages'], 'physical native-row/byte binding')
            require(0 <= p['block'] < 2 and 0 <= p['wl_start'] and p['wl_start'] + p['reserved_WL'] <= 192, 'native job crosses block boundary')
            require(p['reserved_WL'] == pipe['waves'] == (p['pages'] + 3) // 4, 'physical fixed-source extent including partial final WL')
            intervals[rank].append((p['row_start'], p['row_start'] + p['rows']))
            physical_bindings += 1
    if menu_rows is not None:
        desired = {(rank, chip): int(menu_rows[e, chip]) for rank, e in enumerate(chosen)
                   for chip in range(16) if menu_rows[e, chip]}
        require(actual == desired, 'missing or unexpected frozen-menu Down shard')
    if map_jobs is not None:
        require(set(actual) == set(expected), 'missing selected resident Down job')
        for rank in range(7):
            cursor = 0
            for start, end in sorted(intervals[rank]):
                require(start == cursor and end > start, 'Down row scatter gap/overlap')
                cursor = end
            require(cursor == 1536, 'Down row scatter coverage')
    return dict(exact_logical_rows=18944, GU_rows=8192, Down_rows=10752,
                physical_Down_bindings=physical_bindings,
                exclusive_busy_ns_per_resource={str(k): int(v) for k, v in busy.items()})


def capacity_audit(ledger, maps):
    """Reconstruct unique physical page addresses from sealed GU and whole jobs."""
    results = []
    for entry in ledger:
        total_pages = 0
        nonempty_blocks = 0
        flat_jobs = []
        logical = defaultdict(list)
        for chip in entry['chips']:
            addresses = set()
            counts = [[0] * 4 for _ in range(2)]
            def add(block, plane, wl):
                require(0 <= block < 2 and 0 <= plane < 4 and 0 <= wl < 192, 'capacity address outside common budget')
                key = (block, plane, wl)
                require(key not in addresses, 'resident GU/Down address collision')
                addresses.add(key)
                counts[block][plane] += 1
            for p in chip['GU']:
                require(p['bytes'] == 16384, 'resident GU page size')
                add(p['block'], p['plane'], p['wl'])
            for j in chip['Down_jobs']:
                require(j['chip'] == chip['chip'], 'ledger chip ownership')
                require(j['pages'] * 16384 == j['rows'] * j['columns'], 'resident full rows')
                require(j['reserved_WL'] == (j['pages'] + 3) // 4, 'private-job WL reservation including partial final WL')
                for page in range(j['pages']):
                    add(j['block'], page % 4, j['wl_start'] + page // 4)
                logical[j['expert'], j['copy']].append((j['row_start'], j['row_start'] + j['rows']))
                flat_jobs.append(j)
            require(counts == chip['actual_pages_per_block_plane'], 'ledger per-plane count')
            blocks = sum(any(row) for row in counts)
            require(blocks == chip['nonempty_blocks'], 'ledger per-chip block count')
            require(len(addresses) * 16384 == chip['programmed_bytes'], 'ledger per-chip bytes')
            nonempty_blocks += blocks
            total_pages += len(addresses)
        for e in range(65):
            copies = range(3) if entry['plan'] == 'triple_private_shared4' and e < 64 else range(1)
            for copy_id in copies:
                cursor = 0
                for start, end in sorted(logical[e, copy_id]):
                    require(start == cursor and end > start, 'resident replica row gap/overlap')
                    cursor = end
                require(cursor == 1536, 'resident full replica coverage')
        require(len(logical) == (193 if entry['plan'] == 'triple_private_shared4' else 65), 'unexpected logical replica')
        map_key = entry['GU'] + '.' + entry['plan']
        canonical = lambda j: json.dumps(j, sort_keys=True)
        require(sorted(map(canonical, flat_jobs)) == sorted(map(canonical, maps[map_key][entry['layer']])), 'resident ledger/map agreement')
        require(nonempty_blocks == entry['actual_nonempty_blocks'] <= entry['allocated_budget_blocks'] == 32, 'total common block budget')
        require(total_pages * 16384 == entry['programmed_bytes'], 'total programmed bytes')
        results.append(dict(GU=entry['GU'], plan=entry['plan'], layer=entry['layer'],
                            unique_pages=total_pages, programmed_bytes=total_pages * 16384,
                            actual_nonempty_blocks=nonempty_blocks, common_budget_blocks=32))
    return results


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    began = time.monotonic()
    checker = load_checker()
    files = [('partition', p) for p in sorted((PART / 'TRAIN512').glob('trace.*json'))]
    files += [('replica', p) for p in sorted((REPLICA / 'TRAIN512').glob('trace.*json'))]
    require(Counter(k for k, p in files) == {'partition': 72, 'replica': 108}, 'all representative traces must be present')
    hashes = {}
    for base in [PART, REPLICA]:
        protocol_path = base / 'protocol.frozen.json'
        for p, h in read(protocol_path)['inputs_sha256'].items():
            require(sha(p) == h, 'sealed campaign input changed: ' + p)
            hashes[p] = h
        for p in [protocol_path, base / 'TRAIN512/report.json', base / 'TRAIN512/service_arrays.npz']:
            hashes[str(p)] = sha(p)
    paths = [Path(__file__).resolve(), CHECKER, ROUTES, PART / 'menus.frozen.npz',
             REPLICA / 'Down_job_maps.frozen.json', REPLICA / 'physical_ledger.frozen.json.gz',
             BASE / 'down-replica-headroom/shared-g4-standard-triples/report.json', *[p for k, p in files]]
    hashes.update({str(p): sha(p) for p in paths})
    write(OUT / 'protocol.frozen.json', dict(status='frozen_before_replay', inputs_sha256=hashes,
          reused_checker=str(CHECKER), new_source=str(Path(__file__).resolve()),
          scope='All saved qindex0 representatives: partition24configs*3layers; replica36configs*3layers. No other TRAIN queries replayed.'))
    with np.load(ROUTES) as z:
        routes = z['routes'].transpose(2, 0, 1, 3).reshape(40, 512, 6)[:3]
    with np.load(PART / 'menus.frozen.npz') as z:
        menus = {k: z[k].copy() for k in z.files}
    maps = read(REPLICA / 'Down_job_maps.frozen.json')
    ledger = json.loads(gzip.decompress((REPLICA / 'physical_ledger.frozen.json.gz').read_bytes()))
    capacities = capacity_audit(ledger, maps)
    arrays = {}
    for key, base in [('partition', PART), ('replica', REPLICA)]:
        with np.load(base / 'TRAIN512/service_arrays.npz') as z:
            arrays[key] = {k: z[k].copy() for k in z.files}
    records = []
    replay_arrays = {}
    for i, (kind, p) in enumerate(files):
        r = read(p)
        layer, family, menu, copies, priority = parsed_name(p)
        require(r['service']['input_copies'] == copies and r['service']['down_first'] == int(priority == 'DF'), 'file/service metadata')
        independent, cost = checker.scalar_execute(r['nodes'], r['service']['down_first'])
        saved = np.array([[n['start_ns'], n['end_ns']] for n in r['nodes']], dtype=np.int64)
        require(np.array_equal(independent, saved), 'node timestamp mismatch: ' + str(p))
        require(cost == r['cost_ns'], 'makespan mismatch')
        basic = checker.semantic_audit(r, independent)
        require(basic['FIFO128_valid'], 'representative exceeds 128B FIFO')
        ext = extension_audit(r, routes[layer, 0],
              menu_rows=menus[menu][layer] if kind == 'partition' else None,
              map_jobs=maps[family + '.' + menu][layer] if kind == 'replica' else None,
              triple=menu == 'triple_private_shared4')
        config = '.'.join([family, menu, 'input' + str(copies), priority])
        require(int(arrays[kind][config + '.cost'][layer, 0]) == cost, 'representative differs from saved TRAIN cost')
        require(int(arrays[kind][config + '.fifo'][layer, 0]) == basic['peak_FIFO_bytes'], 'representative differs from saved TRAIN FIFO')
        records.append(dict(campaign=kind, path=str(p), sha256=sha(p), layer=layer, query_index=0,
                            family=family, menu=menu, input_copies=copies, priority=priority,
                            cost_ns=cost, **basic, **ext))
        replay_arrays[kind + '.' + p.name] = independent
        if (i + 1) % 24 == 0:
            print('REPLAYED', i + 1, '/', len(files), flush=True)
    controls = checker.corruption_controls(read(files[0][1]))
    baseline = read(REPLICA / 'TRAIN512/trace.L0.candidate.triple_private_shared4.input1.GF.json')
    for name, mutate in [('wrong_resident_copy', lambda r: r['jobs'][16]['physical_down_job'].__setitem__('copy', 2)),
                         ('row_scatter_shift', lambda r: r['jobs'][16]['physical_down_job'].__setitem__('row_start', 1))]:
        changed = copy.deepcopy(baseline)
        mutate(changed)
        try:
            extension_audit(changed, routes[0, 0], map_jobs=maps['candidate.triple_private_shared4'][0], triple=True)
        except RuntimeError as e:
            controls.append(dict(name=name, detected=True, diagnostic=str(e)))
        else:
            raise RuntimeError('corruption undetected: ' + name)
    # Restricted-calendar bound: preserve the existing GU release calendar,
    # same native384row private jobs, fixed shared-g4 and three replica groups.
    oracle = read(BASE / 'down-replica-headroom/shared-g4-standard-triples/report.json')
    bound_checks = []
    for case in oracle['cases']:
        copies = case['input_copies']
        A = case['GU_calendar_ns']
        B = 250 + 512 * copies // 2 + 59220 + 64
        S = 250 + 1024 * copies // 2 + 89940 + 32
        bound = max(max(A[:12]) + 2 * B, max(A[12:]) + S)
        require(bound == case['triple_chip_calendar_LB_ns'], 'restricted-calendar arithmetic')
        rr = [r for r in records if r['campaign'] == 'replica' and r['family'] == 'candidate'
              and r['menu'] == 'triple_private_shared4' and r['input_copies'] == copies]
        require(len(rr) == 6 and {r['cost_ns'] for r in rr} == {bound + 16}, 'candidate representative bound gap')
        for r in rr:
            saved = read(r['path'])
            gu = [j for j in saved['jobs'] if j['phase'] == 0]
            require(len(gu) == 16 and all(saved['nodes'][j['release']]['end_ns'] == A[j['chip']] for j in gu), 'bound GU release calendar changed')
            last_owner = max(saved['nodes'][j['release']]['end_ns'] for j in saved['jobs'])
            last_onfi = max(n['end_ns'] for n in saved['nodes'] if n['kind'] == 'output_ONFI')
            final_pcie = [n for n in saved['nodes'] if n['kind'] == 'output_PCIe' and n['end_ns'] == bound + 16]
            require(last_owner == last_onfi == bound, 'restricted-calendar owner/ONFI completion witness')
            require(final_pcie and all(n['start_ns'] == bound and n['duration'] == 16 and n['bytes'] == 128 for n in final_pcie), 'sixteen-ns terminal PCIe witness')
        costs = arrays['replica'][f'candidate.triple_private_shared4.input{copies}.GF.cost']
        require(np.all(costs == bound + 16), 'saved full TRAIN candidate cost differs from bound gap')
        bound_checks.append(dict(input_copies=copies, restricted_calendar_LB_ns=bound,
                                 representative_replay_cost_ns=bound + 16, gap_ns=16,
                                 terminal_ONFI_and_owner_end_ns=bound, terminal_PCIe_bytes=128,
                                 terminal_PCIe_duration_ns=16,
                                 all1536_saved_cost_entries_equal=True,
                                 scope='Fixed GU releases, native384row jobs, fixed sharedg4, three complete private replica groups. Not global over geometry.'))
    np.savez_compressed(OUT / 'independent_node_times.npz', **replay_arrays)
    require(hashes == {p: sha(p) for p in hashes}, 'input changed during replay')
    report = dict(status='pass_saved_representatives', utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        new_driver_sha256=sha(__file__), reused_independent_checker_sha256=sha(CHECKER), inputs_sha256=hashes,
        source_distinction='New driver/extensions, reused previously independent scalar executor and semantic checker; no second newly independent scheduler is claimed.',
        trace_counts=dict(Counter(r['campaign'] for r in records)), records=records,
        total_nodes=sum(r['nodes'] for r in records), total_edges=sum(r['edges'] for r in records),
        total_jobs=sum(r['jobs'] for r in records), all_saved_node_times_exact=True,
        FIFO128_valid_traces=sum(r['FIFO128_valid'] for r in records),
        capacity_records=capacities, restricted_calendar_checks=bound_checks, corruption_controls=controls,
        development_failures=[dict(stage='initial_new_driver_setup', diagnostic="KeyError: topk_ids is not a file in the archive",
                                   correction='Read the existing routes NPZ field; no target source or result changed.'),
                              dict(stage='initial_new_driver_capacity_check', diagnostic='whole-WL private job',
                                   correction='Allow the existing sharedg16 six-page job to reserve ceil(6/4)=2WL; full rows do not require four active planes in its final wave.')],
        independent_node_times_sha256=sha(OUT / 'independent_node_times.npz'),
        actual_command=['python', str(Path(__file__).resolve())], elapsed_seconds=time.monotonic()-began,
        limitations=[
            'Only qindex0 representative traces are replayed:72 partition and108 replica. EveryTRAINquery is not independently replayed; fullarrays are read only for representative binding and the explicitly named candidate constant-cost check.',
            'Saved nodes/edges are the replay input. This audits conditional scheduling, complete-row counts, dependencies, FIFO and resident metadata; it performs no GEMV arithmetic or task-quality test.',
            'The scheduler and primary semantic checker are reused independent code, not a new second independent scheduling implementation. Audited sources and their own audits are not imported.',
            'FIFO uses continuous2B/ns drain, not instantaneous clearing at ONFI start. Resource-free row timers contribute no resource busy work. PCIe does not retain chip ownership.',
            'PB feasibility is the sealed job-recipe declaration, whose timing contract was independently audited previously; this pass does not reconstruct every array/ECC/cachecopy page deadline.',
            'Skip-WL CR/IPR, source permutation and preECC dummy dropping remain conditional; native PE sustain6.4GB/s and cold28us firstavailability are modeled accounting, not measured firstphase/discharge evidence.',
            'Resident coordinates, row extents and fixed operand dimensions are checked; physical coefficient programming and original-model numeric/ABI equivalence are not established.',
            'Anchor Down jobs are placed intact in block1, thus32nonempty chipblocks in this sealed layout; triples use28. The earlier independent firstfit witness used fewer anchor blocks. Both fit the same32block budget.',
            'The280510/285422ns bound preserves current GU release calendars, sharedg4 and384row private-job geometry. Its16ns gap is not a proof of optimality over other GU/Down geometries or hardware.'])
    write(OUT / 'report.json', report)
    (OUT / 'report.md').write_text(
        f"All180 saved qindex0 representatives pass exact scalar replay:72 final partition and108 replica traces, {report['total_nodes']:,} nodes, {report['total_edges']:,} edges and {report['total_jobs']:,} chip jobs. Every saved start/end and makespan matches. All180 fit128B outputFIFO under continuous2B/ns drain.\n\n"
        'This new driver reuses the earlier independent scalar_execute and semantic_audit. It does not claim a new independent scheduler or replay of everyTRAINquery. Additional checks bind each representative to its saved TRAIN cost/FIFO, exact8192GU+10752Down complete rows, frozen partition chip-row menus, replica expert/copy/row scatter, full private inputs, fixed-source native Down jobs and resource busy work excluding timers. FullGU-at-host→activation/A8→fullprivateDown upload and allsevenDown→combine, channel exclusion and chip ownership through compute+ONFI are checked; PCIe is outside ownership. Seven corruption controls are detected.\n\n'
        'All27 frozen replica resident layouts are independently expanded to unique page coordinates with no GU/Down collisions, complete1536row expert-copy coverage and at most2blocks/chip. Onecopy anchors program155713536B/layer and use32chipblocks in this sealed intact-block1 placement; triples program256376832B, +96MiB, and use28. This differs from the earlier valid compact-firstfit oracle placement, not from its common32block budget.\n\n'
        'The candidate restricted-calendar lower bound is280510ns for input1 and285422ns for input4. Replayed representatives cost280526/285438ns, a16ns gap; the saved1536cost entries per profile agree. The bound fixes GU releases, sharedg4 and native384row jobs and is not global over arbitrary geometry.\n\n'
        + '\n'.join(report['limitations']) + '\n\nReproduce: `python ' + str(Path(__file__).resolve()) + '`\n')
    print(json.dumps(dict(status=report['status'],traces=len(records),nodes=report['total_nodes'],edges=report['total_edges'],jobs=report['total_jobs'],elapsed_seconds=report['elapsed_seconds']),indent=2))


if __name__ == '__main__':
    main()
