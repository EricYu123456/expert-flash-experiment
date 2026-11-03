#!/usr/bin/env python3
"""TRAIN-only causal FFN replay of standard complete private-Down copies."""
from collections import Counter, defaultdict
from pathlib import Path
import argparse, gzip, hashlib, json, sys, time
import numpy as np

OUT = Path(__file__).resolve().parent
BASE = OUT.parent / 'full-ffn'
sys.path.insert(0, str(BASE))
import asymmetric_down as old
f, a = old.f, old.a
SHARED = OUT.parent / 'down-headroom/final-run'
FAMILIES = ['candidate', 'RR', 'g8_fixed']
PLANS = ['original_g4_shared16', 'onecopy_shared4', 'triple_private_shared4']


def inputs():
    selection = f.read(SHARED / 'selection.frozen.json')
    with np.load(SHARED / 'menus.frozen.npz') as z:
        original = z['original_g4_shared16'].copy()
        shared = {g: z[selection['winners'][g]].copy() for g in FAMILIES}
    with np.load(BASE / 'g8-addendum/group_maps.frozen.npz') as z:
        groups = z['learned'].copy()
    with np.load(f.TRAIN) as z:
        assert z['split'].item() == 'train' and z['completed'].all()
        routes = z['routes'].transpose(2, 0, 1, 3).reshape(40, 512, 6)[:3]
    for g, m in shared.items():
        assert np.all(m.sum(axis=2) == 1536)
        assert np.all(m[:, 64, :12] == 0) and np.all(m[:, 64, 12:] == 384)
        assert np.all(m[:, :64, 12:] == 0)
        for l in range(3):
            for e in range(64):
                occupied = np.flatnonzero(m[l, e])
                assert len(occupied) == 4 and np.all(m[l, e, occupied] == 384)
                assert np.array_equal(occupied, np.arange(4) + (occupied[0] // 4) * 4)
    return routes, original, shared, groups


def GU_pages(family, layer, chip, groups, pipeline):
    if family == 'g8_fixed':
        group = chip // 8
        resident = np.flatnonzero(groups[layer] == group)
        result = []
        for position, e in enumerate(resident):
            for j in range(12):
                result.append(dict(owner=int(e), local_page=j, plane=j % 4,
                                   block=0, wl=position * 3 + j // 4, bytes=16384))
        result += [dict(owner=64, local_page=j, plane=j % 4, block=0,
                        wl=96 + j // 4, bytes=16384) for j in range(12)]
        return result
    pages = a.resident_pool([], family, np.arange(64))
    result = [dict(owner=int(p['owner'][1:]), local_page=p['page_index'],
                   plane=p['plane'], block=p['block'], wl=p['wl'], bytes=16384)
              for p in pages]
    if family == 'candidate':
        shared_pages = [p for w in pipeline.build_g4_candidate(range(6)) for p in w['pages'] if p['owner'] == 's']
    else:
        shared_pages = [dict(page_index=j, plane=j % 4, block=0, wl=96 + j // 4) for j in range(12)]
    result += [dict(owner=64, local_page=p['page_index'], plane=p['plane'],
                    block=p['block'], wl=p['wl'], bytes=16384) for p in shared_pages]
    return result


def allocate_down(family, plan, layer, original, shared):
    jobs = []
    cursor = np.zeros(16, dtype=int)
    if plan == 'triple_private_shared4':
        for group in range(3):
            for e in range(64):
                for chip in range(4 * group, 4 * group + 4):
                    jobs.append(dict(expert=e, copy=group, chip=chip, rows=384,
                                     columns=512, block=1, wl_start=3 * e,
                                     row_start=(chip % 4) * 384, pages=12, reserved_WL=3))
        for chip in range(12, 16):
            jobs.append(dict(expert=64, copy=0, chip=chip, rows=384,
                             columns=1024, block=0, wl_start=128,
                             row_start=(chip - 12) * 384, pages=24, reserved_WL=6))
    else:
        matrix = original[layer] if plan == 'original_g4_shared16' else shared[family][layer]
        row_cursor = np.zeros(65, dtype=int)
        for e in range(65):
            columns = 1024 if e == 64 else 512
            for chip in range(16):
                rows = int(matrix[e, chip])
                if not rows:
                    continue
                pages = (rows * columns + 16383) // 16384
                assert rows * columns == pages * 16384
                reserved = (pages + 3) // 4
                jobs.append(dict(expert=e, copy=0, chip=chip, rows=rows,
                                 columns=columns, block=1, wl_start=int(cursor[chip]),
                                 row_start=int(row_cursor[e]), pages=pages, reserved_WL=reserved))
                cursor[chip] += reserved
                row_cursor[e] += rows
        assert np.all(row_cursor == 1536)
        assert np.all(cursor <= 192)
    return jobs


def materialize():
    routes, original, shared, groups = inputs()
    pipeline = f.load_pipeline()
    records, maps = [], {}
    for family in FAMILIES:
        for plan in PLANS:
            all_jobs = []
            for layer in range(3):
                jobs = allocate_down(family, plan, layer, original, shared)
                all_jobs.append(jobs)
                chips = []
                total_bytes = 0
                for chip in range(16):
                    gu = GU_pages(family, layer, chip, groups, pipeline)
                    down_jobs = [j for j in jobs if j['chip'] == chip]
                    pages = [dict(p, matrix='GU', copy=0) for p in gu]
                    for job in down_jobs:
                        for j in range(job['pages']):
                            pages.append(dict(owner=job['expert'], local_page=j,
                                plane=j % 4, block=job['block'], wl=job['wl_start'] + j // 4,
                                bytes=16384, matrix='Down', copy=job['copy']))
                    addresses = [(p['block'], p['plane'], p['wl']) for p in pages]
                    assert len(addresses) == len(set(addresses))
                    assert all(0 <= p['wl'] < 192 and p['block'] in [0, 1] for p in pages)
                    counts = [[sum(p['block'] == b and p['plane'] == q for p in pages)
                               for q in range(4)] for b in range(2)]
                    total_bytes += len(pages) * 16384
                    chips.append(dict(chip=chip, actual_pages_per_block_plane=counts,
                        nonempty_blocks=sum(any(row) for row in counts),
                        programmed_bytes=len(pages) * 16384, GU=gu, Down_jobs=down_jobs))
                expected = 155713536 + (100663296 if plan == 'triple_private_shared4' else 0)
                assert total_bytes == expected, (family, plan, layer, total_bytes)
                for e in range(65):
                    copies = 3 if plan == 'triple_private_shared4' and e < 64 else 1
                    for copy in range(copies):
                        bands = [j for j in jobs if j['expert'] == e and j['copy'] == copy]
                        intervals = sorted((j['row_start'], j['row_start'] + j['rows']) for j in bands)
                        assert intervals[0][0] == 0 and intervals[-1][1] == 1536
                        assert all(lo[1] == hi[0] for lo, hi in zip(intervals, intervals[1:]))
                records.append(dict(GU=family, plan=plan, layer=layer,
                    allocated_budget_blocks=32, actual_nonempty_blocks=sum(c['nonempty_blocks'] for c in chips),
                    programmed_bytes=total_bytes, extra_bytes=expected - 155713536, chips=chips))
            maps[family + '.' + plan] = all_jobs
    return records, maps


def freeze():
    OUT.mkdir(exist_ok=True)
    assert not (OUT / 'protocol.frozen.json').exists()
    records, maps = materialize()
    with gzip.open(OUT / 'physical_ledger.frozen.json.gz', 'wt') as z:
        json.dump(records, z, separators=(',', ':'))
    f.dump(OUT / 'Down_job_maps.frozen.json', maps)
    paths = [Path(__file__), BASE / 'asymmetric_down.py', BASE / 'policy_addendum.py',
             BASE / 'shaping_addendum.py', a.SHAPER, f.PIPE, f.CPP,
             f.ROOT / 'scripts/research_20261102_cr_ipr_full_ffn.py', f.TRAIN,
             BASE / 'g8-addendum/group_maps.frozen.npz', SHARED / 'menus.frozen.npz',
             SHARED / 'selection.frozen.json', SHARED / 'protocol.frozen.json',
             OUT / 'physical_ledger.frozen.json.gz', OUT / 'Down_job_maps.frozen.json']
    f.dump(OUT / 'protocol.frozen.json', dict(status='frozen_before_execution', frozen_utc=f.utc(),
        inputs_sha256={str(p): f.sha(p) for p in paths}, families=FAMILIES, plans=PLANS,
        routes='Previously observed W8/A8 TRAIN first three layers, 512 inputs/layer. No heldout or fitting in this experiment.',
        anchors='Each GU family receives original g4 TRAIN-learned routedDown/sharedg16, and its unchanged frozen all64 onecopy sharedg4 winner. Prior g8GU originalg2 remains an additional sealed reference.',
        replica='Standard whole privateDown complete copies: all64 experts copied onto each of three disjoint g4 groups chips0..3,4..7,8..11. SharedDown exclusively chips12..15. Sorted router-known expert IDs assigned ranks0/1 to group0,2/3 to group1,4/5 to group2. No cost oracle, copy optimizer, new CBC, or perquery migration.',
        storage='Original onecopy155713536B/layer; triples256376832B/layer, +100663296B=96MiB two extra complete privateDown copies. Same common two-block/chip capacity budget for all controls; actual nonempty blocks reported separately.',
        addresses='GU unchanged block0. Triple privateDown eachchip has64 contiguous3WL bands filling block1 exactly192WL; each12page GEMV remains within one block and starts cold. Triple shared24pages/chip occupies block0 WL128..133. Anchors whole Down jobs placed intact into block1; every private job independently cold with fullinput, so relocation adds no crossblock read or hidden reuse.',
        native_contract='Full common-input GU through host full-expert output barrier, activation/A8, then each expert private complete C512 input; shared C1024. One active GEMV/chip until compute and finalONFIoutput. Outputs stream only at complete rows; no private context sharing.',
        profiles='Existing fixed9.7us CR/copy500ns conditional finitePB model; commonGU same first-feasible fill/split/coldreset policy and granted source/drop capabilities for all families. g8GU fixed0123. PrivateDown fixed0123 strict500, no shaping/coldreset rescue, one initialcold per job.',
        resources=dict(host_submit_ns=2000,command_ns=250,PCIe_bytes_per_ns=8,channels=8,channel_bytes_per_ns=2,chip_channel='chip%8',activation_ns=0),
        FIFO='Exact perchip produced-minus-continuously-channel-drained output backlog; report64/128/256/2048B. Primary validity requires128B for every query. No implicit backpressure or extraFIFO.',
        screen_indices=np.linspace(0, 511, 64, dtype=int).tolist(),screen_axes=dict(input_copies=[1],priority=['GF']),
        gate='Freeze all plans before screen. Run full512 iff at least one GU family tripleplan is PBvalid and128BFIFOvalid on all192events AND mean fullFFN improvement exceeds5% relative to samefamily onecopyshared4. No layout/assignment change after screen. Full results are descriptive TRAIN reuse.',
        full_axes=dict(input_copies=[1,4],priority=['GF','DF']),
        route_independence='For every replay retain normalized timing-DAG signature counts, excluding physical-address labels but preserving all event/resource/dependency durations and service priority. If all1536 candidate/triple queries share one DAG and one cost, report route independence only inside this formal contract; not hardwareconstant or newCBC.',
        interpretation='Standard replication plus shared/private resource partition, no new algorithm or hardwaremeasurement. New geometry invalidates prior oracle bound assuming unchanged sharedg16; stored bytes increase, query mathematical coefficients/readbytes do not. Conditional timings, no fullweights/inference/numericalchanges.'))
    print(json.dumps(dict(status='frozen',source_sha256=f.sha(__file__),protocol_sha256=f.sha(OUT / 'protocol.frozen.json'))), flush=True)


class Recipes(old.KernelRecipes):
    def private_at(self, job, rank):
        waves = []
        for start in range(0, job['pages'], 4):
            pages = [dict(page_id=f'v.p{j}', owner='v', page_index=j, plane=j % 4,
                          block=job['block'], wl=job['wl_start'] + j // 4, bytes=16384)
                     for j in range(start, min(start + 4, job['pages']))]
            waves.append(dict(pages=pages, source_order=[p['plane'] for p in pages]))
        recipe = self.make(waves, job['columns'], {'v': 0}, 'fixed')
        if recipe is None:
            return None
        return dict(recipe, milestones=[(rank, t, b) for _, t, b in recipe['milestones']])


def graph_for(route, layer, family, plan, service, recipes, groups, maps):
    route = sorted(map(int, route))
    ranks = {f'e{e}': i for i, e in enumerate(route)}
    ranks['s'] = 6
    chosen = route + [64]
    jobs = maps[family + '.' + plan][layer]
    selected_jobs = []
    for rank, e in enumerate(chosen):
        copy = rank // 2 if plan == 'triple_private_shared4' and e < 64 else 0
        js = [j for j in jobs if j['expert'] == e and j['copy'] == copy]
        assert sum(j['rows'] for j in js) == 1536
        for job in js:
            recipe = recipes.private_at(job, rank)
            if recipe is None:
                return None
            selected_jobs.append((rank, e, job, recipe))
    g = f.Graph(service)
    submit = g.add(0, service.submit_ns, kind='GU_submit')
    upload = g.add(2, 192, [submit], kind='GU_input_PCIe', nbytes=1536)
    outputs = defaultdict(list)
    if family == 'g8_fixed':
        for group in [0, 1]:
            waves, pool = old.group_waves(route, groups[layer], group)
            recipe = recipes.make(waves, 1536, ranks, 'fixed', pool)
            if recipe is None:
                return None
            for chip in range(8 * group, 8 * group + 8):
                for owner, nodes in g.matrix(chip, 0, 1536, recipe, submit, upload).items():
                    outputs[owner] += nodes
    else:
        waves = recipes.p.build_g4_candidate(route) if family == 'candidate' else f.rr_greedy_waves(route)
        pool = a.resident_pool(route, family, np.arange(64))
        recipe = recipes.make(waves, 1536, ranks, 'permutation', pool)
        if recipe is None:
            return None
        for chip in range(16):
            for owner, nodes in g.matrix(chip, 0, 1536, recipe, submit, upload).items():
                outputs[owner] += nodes
    complete = []
    for rank, e in enumerate(chosen):
        ready = g.add(-1, 0, outputs[rank], expert=rank, kind='full_GU_at_host')
        act = g.add(0, service.activation_ns, [ready], 1, rank, kind='full_vector_activation_A8')
        down_submit = g.add(0, service.submit_ns, [act], 1, rank, kind='Down_submit')
        columns = 1024 if e == 64 else 512
        up = g.add(2, (columns + 7) // 8, [down_submit], 1, rank, kind='Down_input_PCIe', nbytes=columns)
        for rr, ee, job, recipe in selected_jobs:
            if rr != rank:
                continue
            before = len(g.jobs)
            complete += g.matrix(job['chip'], 1, columns, recipe, down_submit, up)[rank]
            g.jobs[before]['physical_down_job'] = job
    g.add(0, service.combine_ns, complete, 1, kind='weighted_combine')
    return g


def checked():
    protocol = f.read(OUT / 'protocol.frozen.json')
    for path, h in protocol['inputs_sha256'].items():
        assert f.sha(path) == h, path
    routes, _, _, groups = inputs()
    return protocol, routes, groups, f.read(OUT / 'Down_job_maps.frozen.json')


def execute(mode):
    spec, routes, groups, maps = checked()
    screen = mode == 'screen'
    indices = np.asarray(spec['screen_indices']) if screen else np.arange(512)
    if not screen:
        selection = f.read(OUT / 'selection.frozen.json')
        assert selection['gate_passed'] and selection['screen_report_sha256'] == f.sha(OUT / 'screen/report.json')
    configs = [(gu, plan, cp, df) for gu in FAMILIES for plan in PLANS
               for cp in ([1] if screen else [1, 4]) for df in ([0] if screen else [0, 1])]
    dest = OUT / ('screen' if screen else 'TRAIN512')
    dest.mkdir(exist_ok=True)
    assert not (dest / 'report.json').exists()
    f.dump(dest / 'run.frozen.json', dict(frozen_utc=f.utc(),source_sha256=f.sha(__file__),
        protocol_sha256=f.sha(OUT / 'protocol.frozen.json'),indices=indices.tolist(),configurations=configs,
        selection_sha256=None if screen else f.sha(OUT / 'selection.frozen.json')))
    recipes = Recipes(f.load_pipeline(), a.load_shaper())
    executor = f.Executor(OUT)
    arrays, records, cache = {}, [], {}
    unique, started = 0, time.monotonic()
    for ci, (gu, plan, cp, df) in enumerate(configs):
        name = f'{gu}.{plan}.input{cp}.{"DF" if df else "GF"}'
        cost = np.full((3, len(indices)), -1, dtype=np.int64)
        fifo = cost.copy()
        aggregate = Counter()
        dag_counts = Counter()
        before = recipes.decisions.copy()
        for layer in range(3):
            for q, index in enumerate(indices):
                g = graph_for(routes[layer,index], layer, gu, plan, f.Service(cp,df,0), recipes, groups, maps)
                if g is None:
                    continue
                key, data = g.signature()
                dag_counts[key] += 1
                key = (key, df, json.dumps(dict(g.pipeline_totals), sort_keys=True))
                if key in cache:
                    value, audit = cache[key]
                else:
                    result = executor.run(g,data)
                    audit = f.audit(g,result,full=q in [0,len(indices)-1])
                    value = result['cost_ns']
                    cache[key] = value, audit
                    unique += 1
                if q == 0:
                    result = executor.run(g,data)
                    f.dump_trace(dest / f'trace.L{layer}.{name}.json',g,result,audit)
                cost[layer,q], fifo[layer,q] = value, audit['peak_output_fifo_bytes']
                for k,v in audit['counts'].items():
                    aggregate[k] += v
                for k,v in audit['pipeline_counts'].items():
                    aggregate['pipeline_' + k] += v
                for k,v in audit['resource_work_ns'].items():
                    aggregate[k] += v
        good = cost >= 0
        arrays[name + '.cost'], arrays[name + '.fifo'] = cost, fifo
        records.append(dict(name=name,GU=gu,plan=plan,input_copies=cp,priority='DF' if df else 'GF',
            completed=int(good.sum()),requested=int(cost.size),
            cost_ns=f.stats(cost[good]) if good.any() else None,
            three_layer_sum_ns=f.stats(cost.sum(axis=0)) if good.all() else None,
            FIFO_peak_bytes=f.stats(fifo[good]) if good.any() else None,
            valid_FIFO_capacities={str(k):int(np.count_nonzero(good & (fifo <= k))) for k in [64,128,256,2048]},
            counts_and_work_per_valid_query={k:v/int(good.sum()) for k,v in aggregate.items()} if good.any() else {},
            policy_calls=dict(recipes.decisions - before)))
        records[-1]['route_independence'] = dict(
            normalized_timing_DAG_counts=dict(dag_counts),
            distinct_DAGs=len(dag_counts),
            distinct_completed_costs=int(len(np.unique(cost[good]))),
            all_queries_same_DAG_and_cost=bool(good.all() and len(dag_counts)==1 and len(np.unique(cost[good]))==1),
            scope='Resource/duration/dependency/rank DAG with physical-address labels excluded; service priority fixed by this configuration. Formal conditional model only.')
        progress = dict(finished=ci+1,total=len(configs),unique_graphs=unique,
                        elapsed_seconds=time.monotonic()-started,last=name)
        f.dump(dest / 'progress.json', progress)
        print(json.dumps(progress),flush=True)
    np.savez_compressed(dest / 'service_arrays.npz',**arrays)
    f.dump(dest / 'private_failures.json',recipes.failure_samples)
    report = dict(status='complete_conditional_TRAIN_private_Down_replica',mode=mode,records=records,
        indices=indices.tolist(),source_sha256=f.sha(__file__),protocol_sha256=f.sha(OUT / 'protocol.frozen.json'),
        unique_graphs=unique,elapsed_seconds=time.monotonic()-started,executor=executor.receipt,
        scope='Standard wholeDown replication with extra96MiB/layer. Samecapabilities GU controls; noheldout or hardwaremeasurement.')
    f.dump(dest / 'report.json',report)
    if screen:
        checks = {}
        for gu in FAMILIES:
            anchor = next(r for r in records if r['GU']==gu and r['plan']=='onecopy_shared4')
            triple = next(r for r in records if r['GU']==gu and r['plan']=='triple_private_shared4')
            valid = all(r['completed']==192 and r['valid_FIFO_capacities']['128']==192 for r in [anchor,triple])
            gain = 1-triple['cost_ns']['mean']/anchor['cost_ns']['mean'] if valid else None
            checks[gu] = dict(all192_PB_FIFO128_valid=valid,mean_gain_vs_onecopyshared4=gain,
                             passes=bool(valid and gain > .05))
        f.dump(OUT / 'selection.frozen.json',dict(frozen_utc=f.utc(),source_sha256=f.sha(__file__),
            screen_report_sha256=f.sha(dest / 'report.json'),screen_arrays_sha256=f.sha(dest / 'service_arrays.npz'),
            gate_passed=any(c['passes'] for c in checks.values()),checks=checks,
            selected_plans_unchanged=PLANS,rule=spec['gate']))
        print(json.dumps(dict(status='gate_frozen',checks=checks)),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode',choices=['freeze','screen','run'])
    args = parser.parse_args()
    freeze() if args.mode=='freeze' else execute(args.mode)
