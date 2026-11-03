#!/usr/bin/env python3
"""Supplementary sealed-map all-layer transfer; no fitting or map selection."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time
from datetime import datetime, timezone
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT/'results/research-2026-11-02/round2/chip-count-regime-screen/m32-prospective-tail-validation'
OUT = PARENT/'all-layer-fixed-map-transfer'
CAPTURE = PARENT/'data-capture'
ADAPTER = ROOT/'scripts/research_20261103_m32_prospective_replay_v2.py'
NAMES = ('free', 'common_cyclic', 'phase_independent', 'tail_cyclic')
ROOT_SHA = '961b0657048e706d1389b9924431b24d68544770b2492ecb4b01f998013e9ccc'
RELEASE_SHA = '07ea0440e55edaf3d48abd30094df7c0f7b9bae52577b6b91e8dbecf876e3884'
COMPARISONS = ('common_cyclic', 'phase_independent', 'tail_cyclic')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2)+'\n')


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def csvwrite(path, records):
    records = list(records)
    if not records:
        Path(path).write_text('')
        return
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)


def adapter():
    spec = importlib.util.spec_from_file_location('all_layer_frozen_adapter', ADAPTER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module; spec.loader.exec_module(module)
    return module


def freeze():
    assert sha(OUT/'protocol.frozen.json') == ROOT_SHA
    assert sha(PARENT/'root_release.json') == RELEASE_SHA
    assert not (OUT/'execution_protocol.frozen.json').exists()
    root = json.loads((OUT/'protocol.frozen.json').read_text())
    a = adapter()
    files = [Path(__file__), ADAPTER, a.CPP, a.PYTHON,
             ROOT/'scripts/research_20261102_whole_row_compiler.py',
             OUT/'protocol.frozen.json', PARENT/'root_release.json',
             PARENT/'protocol.frozen.json', CAPTURE/'acquisition_capture_protocol.frozen.json',
             CAPTURE/'manifest.json', CAPTURE/'capture_summary.json', CAPTURE/'completion.frozen.json',
             CAPTURE/'granite4tiny/decode.routes.npz',
             PARENT/'replay-adapter-v2/prospective/costs.npz',
             PARENT/'replay-adapter-v2/prospective/completion.json']
    files += [Path(root['plans'][name]['path']) for name in NAMES]
    snapshot = OUT/'source-snapshot'; snapshot.mkdir(exist_ok=False)
    references = []
    for path in files:
        item = dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)
        if path.suffix in ('.py', '.cpp'):
            target = snapshot/(item['sha256'][:16]+'-'+path.name)
            shutil.copy2(path, target); item['snapshot'] = str(target)
        references.append(item)
    dump(OUT/'execution_protocol.frozen.json', dict(
        status='source_and_execution_rules_frozen_before_secondary_cost_computation',
        utc=datetime.now(timezone.utc).isoformat(), source_sha256=sha(__file__),
        parent_protocol_sha256=ROOT_SHA, release_sha256=RELEASE_SHA, references=references,
        costs='163840: 4 fixed maps * 2 input profiles * 40 layers * 512 original-order queries; GF only. No fit or choice changes.',
        parallel='Eight independent ctypes calls, one per fixed map/profile; unchanged native source has per-call local graph/state. No query reduction or time-based exclusion.',
        physical='Separate metadata image per map: 40 distinct layers, per-chip decreasing-wave first-fit (ties layer then job index); commands stay within 192 wordlines. Logical coefficient/scale ID=(layer,expert,phase,row), all columns, shared once/layer. Addresses may change; homogeneous service ignores physical block address.',
        traces='96 complete native/Python traces q0/255/511 x layers0/13/26/39 x4maps x2profiles; compare every node and audit resource locks/fullGU barrier/private input reload.',
        analysis='Sum40 layers per query first, linear P99. All40 layers/32prompts/two sources/all negative free-v-control cases. Bootstrap 2000 PCG64 seed20261115 paired32prompt clusters retaining16steps; CIs only predeclared phase_independent and tail_cyclic comparisons.',
        limits='Supplementary after capture/before cost rules, not original prospective primary. Synthetic sum of FFN service estimates, not full-model token latency, paired placed numeric proof, or device measurement. No heldout reuse, inference, weights, new ABI, or map fitting.'))
    print(json.dumps(dict(execution_protocol_sha256=sha(OUT/'execution_protocol.frozen.json'), source_sha256=sha(__file__))), flush=True)


def check_frozen():
    p = json.loads((OUT/'execution_protocol.frozen.json').read_text())
    assert p['source_sha256'] == sha(__file__)
    for item in p['references']:
        assert sha(item['path']) == item['sha256'], item['path']
    assert sha(OUT/'protocol.frozen.json') == ROOT_SHA
    assert sha(PARENT/'root_release.json') == RELEASE_SHA
    return json.loads((OUT/'protocol.frozen.json').read_text())


def physical_image(name, plan, backend, service, directory):
    lowered = backend.lower_rows(plan['rows'][:, 0], plan['rows'][:, 1], 1536,
                                [512]*64+[1024], service)
    for key in ('offsets', 'jobs', 'input_columns', 'total_rows'):
        assert np.array_equal(plan[key], lowered[key]), key
    # Each record is one entire full-column logical row interval, never a replica.
    records = []
    for layer in range(40):
        for job, item in enumerate(lowered['logical']):
            records.append([layer, job, item['expert'], item['phase'], item['chip'],
                            item['row_start'], item['rows'], item['columns'], item['pages'],
                            item['waves'], -1, -1, 0])
    ledger = np.asarray(records, np.int64)
    blocks = []; used_waves = []
    for chip in range(32):
        indices = np.flatnonzero(ledger[:, 4] == chip)
        indices = sorted(indices, key=lambda i: (-int(ledger[i, 9]), int(ledger[i, 0]), int(ledger[i, 1])))
        used = []
        for i in indices:
            waves = int(ledger[i, 9])
            block = next((b for b, usage in enumerate(used) if usage+waves <= 192), len(used))
            if block == len(used): used.append(0)
            ledger[i, 10:12] = block, used[block]
            used[block] += waves
        blocks.append(len(used)); used_waves.append(sum(used))
        for block in range(len(used)):
            intervals = sorted((int(ledger[i, 11]), int(ledger[i, 11]+ledger[i, 9])) for i in indices if ledger[i, 10] == block)
            assert all(a >= 0 and b <= 192 for a, b in intervals)
            assert all(x[1] <= y[0] for x, y in zip(intervals, intervals[1:]))
    for layer in range(40):
        rows = ledger[ledger[:, 0] == layer]
        for expert in range(65):
            for phase in (0, 1):
                bands = rows[(rows[:, 2] == expert) & (rows[:, 3] == phase)]
                assert bands[0, 5] == 0
                assert np.all(bands[:-1, 5]+bands[:-1, 6] == bands[1:, 5])
                assert bands[-1, 5]+bands[-1, 6] == plan['total_rows'][expert, phase]
    columns = ['layer', 'layer_job', 'expert', 'phase', 'chip', 'logical_row_start', 'rows',
               'full_input_columns', 'pages', 'waves', 'block', 'wordline', 'first_plane']
    np.savez_compressed(directory/(name+'.resident.npz'), ledger=ledger, columns=np.asarray(columns),
                        chip_blocks=np.asarray(blocks), chip_waves=np.asarray(used_waves))
    useful = int(np.sum(ledger[:, 6]*ledger[:, 7])); pages = int(ledger[:, 8].sum()); waves = int(ledger[:, 9].sum())
    emitted = (ledger[:, 8]*16384)//ledger[:, 7]
    partial = (ledger[:, 8]*16384)%ledger[:, 7]
    assert len(set(tuple(row) for row in ledger[:, [0, 2, 3, 5]])) == len(ledger)
    expected_useful = 40*(64*(1024*1536+1536*512)+2048*1536+1536*1024)
    assert useful == expected_useful
    return dict(jobs=len(ledger), logical_rows=int(ledger[:, 6].sum()), coefficient_useful_bytes=useful,
                coefficient_copies_bytes=0, programmed_pages=pages, programmed_page_bytes=pages*16384,
                reserved_waves=waves, reserved_wave_bytes=waves*65536, per_chip_waves=used_waves,
                per_chip_blocks=blocks, total_blocks=sum(blocks), allocated_bytes=sum(blocks)*192*65536,
                whole_fringe_rows_drained_and_dropped=int(np.sum(emitted-ledger[:, 6])),
                partial_row_bytes_charged_and_dropped=int(partial.sum()),
                shared_included_once_per_layer=True, scale_ID='(layer,expert,phase,original logical row); host metadata, unchanged scales',
                all_full_columns=True, distinct_resident_intervals=True, all_commands_one_block=True,
                coefficient_storage='Logical original rows exactly once; padding/fringe not independent logical weights.',
                ABI_limit='No device valid-row/reset/output ABI proof; page-fringe output charged as frozen backend output_fringe=1, complete fringe rows dropped; partial byte tail charged as pages.')


def audit_trace(g, result):
    trace = result['trace']; nodes = g.nodes
    assert len(trace) == len(nodes)
    for resource in range(g.service.resources):
        intervals = sorted((int(t[2]), int(t[3])) for t in trace if t[0] == resource and t[1] > 0)
        assert all(a[1] <= b[0] for a, b in zip(intervals, intervals[1:]))
    jobs = sorted({n['job'] for n in nodes if n['job'] >= 0})
    owners = {}
    for job in jobs:
        ids = [i for i, n in enumerate(nodes) if n['job'] == job]
        acquire = [i for i in ids if nodes[i]['acquire']]; release = [i for i in ids if nodes[i]['release']]
        assert len(acquire) == len(release) == 1
        a, r = acquire[0], release[0]; chip = nodes[a]['chip']
        owners.setdefault(chip, []).append((trace[a, 2], trace[r, 3]))
        inputs = [i for i in ids if nodes[i]['resource'] == 3+chip % g.service.channels and i != r]
        assert len(inputs) == 1
        inp = inputs[0]; assert a in nodes[inp]['deps'] and len(nodes[inp]['deps']) == 2
        assert trace[inp, 2] >= max(trace[d, 3] for d in nodes[inp]['deps'])
    for intervals in owners.values():
        intervals.sort(); assert all(a[1] <= b[0] for a, b in zip(intervals, intervals[1:]))
    acts = [i for i, n in enumerate(nodes) if n['phase'] == 1 and n['resource'] == 0 and n['duration'] == g.service.activation_ns and len(n['deps']) == 1]
    # The six selected routed experts plus shared each have a whole-GU join.
    assert len(acts) == 7
    for act in acts:
        ready = nodes[act]['deps'][0]
        gu = nodes[ready]['deps']; assert gu
        assert all(nodes[i]['phase'] == 0 and nodes[i]['resource'] == 2 for i in gu)
        assert trace[act, 2] >= max(trace[i, 3] for i in gu)
    return dict(resource_exclusion=True, single_owner_until_output_drain=True,
                all_jobs_input_reloaded=True, full_GU_before_A8_and_private_Down=True,
                selected_experts_including_shared=7, jobs=len(jobs), nodes=len(nodes))


def stats(values):
    v = np.asarray(values)
    return dict(mean_ns=float(v.mean()), p99_ns=float(np.quantile(v, .99, method='linear')), max_ns=int(v.max()))


def analyze(arrays, raw, directory):
    ids = raw['prompt_ids'].astype(str); sources = raw['source_keys'].astype(str)
    draws = np.random.default_rng(20261115).integers(0, 32, size=(2000, 32))
    np.savez_compressed(directory/'bootstrap_clusters.npz', prompt_indices=draws)
    tables = []; prompts = []; source_tables = []; aggregate_negative = []; layer_negative = []; summaries = {}
    for copies in (1, 4):
        totals = {name: arrays[f'input{copies}.{name}.cost'].sum(axis=2) for name in NAMES}
        profile = {}
        for name in NAMES:
            costs = arrays[f'input{copies}.{name}.cost']; total = totals[name]
            profile[name] = stats(total)
            for layer in range(40):
                tables.append(dict(input_copies=copies, plan=name, layer=layer, **stats(costs[:, :, layer])))
            for p in range(32):
                prompts.append(dict(input_copies=copies, plan=name, prompt_index=p, prompt_id=ids[p], source=sources[p], **stats(total[p])))
            for source in sorted(set(sources)):
                source_tables.append(dict(input_copies=copies, plan=name, source=source, prompts=int(np.sum(sources == source)), **stats(total[sources == source])))
        comparisons = {}
        for control in COMPARISONS:
            free = totals['free']; other = totals[control]
            difference = free-other
            neg = np.argwhere(difference > 0)
            comparisons[control] = dict(free_minus_control_mean_ns=float(difference.mean()),
                mean_gain_percent=100*(other.mean()-free.mean())/other.mean(),
                free_minus_control_p99_ns=float(np.quantile(free, .99)-np.quantile(other, .99)),
                p99_gain_percent=100*(np.quantile(other, .99)-np.quantile(free, .99))/np.quantile(other, .99),
                regressed_queries=len(neg), tied_queries=int(np.sum(difference == 0)), improved_queries=int(np.sum(difference < 0)),
                max_regression_ns=max(0, int(difference.max())),
                max_regression_percent=max(0., float(np.max(100*difference/other))))
            for p, step in neg:
                aggregate_negative.append(dict(input_copies=copies, control=control, prompt_index=int(p), prompt_id=ids[p],
                    source=sources[p], decode_index=int(step), token_position=int(raw['token_positions'][p, step]),
                    free_ns=int(free[p, step]), control_ns=int(other[p, step]), regression_ns=int(difference[p, step])))
            layer_diff = arrays[f'input{copies}.free.cost']-arrays[f'input{copies}.{control}.cost']
            for p, step, layer in np.argwhere(layer_diff > 0):
                layer_negative.append(dict(input_copies=copies, control=control, prompt_index=int(p), prompt_id=ids[p],
                    source=sources[p], decode_index=int(step), layer=int(layer), regression_ns=int(layer_diff[p, step, layer])))
            if control in ('phase_independent', 'tail_cyclic'):
                f = free[draws].reshape(2000, -1); b = other[draws].reshape(2000, -1)
                fm = f.mean(axis=1); bm = b.mean(axis=1)
                fq = np.quantile(f, .99, axis=1); bq = np.quantile(b, .99, axis=1)
                metrics = dict(free_minus_control_mean_ns=fm-bm, mean_gain_percent=100*(bm-fm)/bm,
                    free_minus_control_p99_ns=fq-bq, p99_gain_percent=100*(bq-fq)/bq)
                comparisons[control]['paired_prompt_cluster_95pct_CI'] = {k: np.quantile(v, [.025, .975]).tolist() for k, v in metrics.items()}
        summaries[f'input{copies}'] = dict(plans=profile, comparisons=comparisons)
    csvwrite(directory/'all_40_layer_stats.csv', tables)
    csvwrite(directory/'all_32_prompt_stats.csv', prompts)
    csvwrite(directory/'both_source_stats.csv', source_tables)
    csvwrite(directory/'all_regressed_40layer_sum_queries.csv', aggregate_negative)
    csvwrite(directory/'all_regressed_layer_queries.csv', layer_negative)
    dump(directory/'analysis.json', dict(status='complete_supplementary_fixed_map_transfer', profiles=summaries,
        statistic='Per query sum of all40 FFN service costs, then mean/linearP99; no sum of layerP99s.',
        bootstrap=dict(draws=2000, clusters=32, retained_steps_per_prompt=16, seed=20261115, rng='PCG64', CI='percentile2.5/97.5 descriptive paired prompt clusters; layers not independent'),
        scope='After-capture/before-cost supplementary analysis. Not original prospective primary, fullmodel token time, quality, placed fullmodel numeric proof, or hardware measurement.',
        all_queries_preserved=True, no_fits_or_map_selection=True, negative_sum_records=len(aggregate_negative), negative_layer_records=len(layer_negative)))
    return summaries


def run():
    root = check_frozen(); directory = OUT/'execution'; directory.mkdir(exist_ok=False)
    t0 = time.monotonic()
    a = adapter(); raw = load(CAPTURE/'granite4tiny/decode.routes.npz'); a.check_routes(raw, True)
    summary = json.loads((CAPTURE/'capture_summary.json').read_text())
    assert summary['status'] == 'complete' and summary['all40layersfinite'] and summary['complete_queries'] == 512
    manifest = json.loads((CAPTURE/'manifest.json').read_text())
    assert summary['manifest_sha256'] == sha(CAPTURE/'manifest.json')
    assert summary['decode_archive_sha256'] == sha(CAPTURE/'granite4tiny/decode.routes.npz')
    for p, item in enumerate(manifest['items']):
        assert str(item['id']) == str(raw['prompt_ids'][p])
        assert str(item['source_key']) == str(raw['source_keys'][p])
    routes = np.ascontiguousarray(raw['routes'].transpose(2, 0, 1, 3).reshape(40*512, 6), np.int32)
    native = a.Native(directory/'build'); plans = {name: load(root['plans'][name]['path']) for name in NAMES}
    base_config = json.loads((PARENT/'protocol.frozen.json').read_text())['hardware_primary']
    for name in NAMES: assert sha(root['plans'][name]['path']) == root['plans'][name]['sha256']
    arrays = {}; durations = []
    def evaluate(name, copies):
        start = time.monotonic(); config = dict(base_config, input_channel_copies=copies, down_first=0)
        cost, bound, _ = native.evaluate(plans[name], routes, native.backend.Service(**config))
        return name, copies, cost, bound, time.monotonic()-start
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(evaluate, name, copies) for name in NAMES for copies in (1, 4)]
        for f in as_completed(futures):
            name, copies, cost, bound, elapsed = f.result()
            key = f'input{copies}.{name}'
            arrays[key+'.cost'] = cost.reshape(40, 32, 16).transpose(1, 2, 0).copy()
            arrays[key+'.bound'] = bound.reshape(40, 32, 16).transpose(1, 2, 0).copy()
            durations.append(dict(plan=name, input_copies=copies, seconds=elapsed))
            np.savez_compressed(directory/(key+'.partial.npz'), cost=arrays[key+'.cost'], bound=arrays[key+'.bound'])
            print(json.dumps(dict(completed=key, seconds=elapsed)), flush=True)
    cost_seconds = time.monotonic()-t0
    arrays.update(prompt_ids=raw['prompt_ids'], source_keys=raw['source_keys'], dataset_item_ids=raw['source_ids'],
                  token_positions=raw['token_positions'], layers=raw['layers'])
    np.savez_compressed(directory/'costs.npz', **arrays)
    primary = load(PARENT/'replay-adapter-v2/prospective/costs.npz'); bridges = []
    for name in NAMES[:3]:
        for copies in (1, 4):
            key = f'input{copies}.{name}'
            for suffix in ('cost', 'bound'):
                assert np.array_equal(arrays[key+'.'+suffix][:, :, 0], primary[key+'.'+suffix]), key
            bridges.append(dict(plan=name, input_copies=copies, queries=512, all_costs_and_bounds_exact=True))
    physical_dir = directory/'physical'; physical_dir.mkdir()
    service = native.backend.Service(**base_config)
    physical = {name: physical_image(name, plan, native.backend, service, physical_dir) for name, plan in plans.items()}
    dump(physical_dir/'report.json', physical)
    trace_dir = directory/'traces'; trace_dir.mkdir(); traces = []
    for name in NAMES:
        for copies in (1, 4):
            service = native.backend.Service(**dict(base_config, input_channel_copies=copies, down_first=0))
            for layer in (0, 13, 26, 39):
                for query in (0, 255, 511):
                    route = routes[layer*512+query:layer*512+query+1]
                    cost, bound, tr = native.evaluate(plans[name], route, service, True)
                    g = native.backend.ReferenceGraph.build(plans[name], np.r_[route[0], 64], service); py = g.run()
                    assert py['cost'] == int(cost[0]) and np.array_equal(tr, py['trace'])
                    p, step = divmod(query, 16)
                    assert cost[0] == arrays[f'input{copies}.{name}.cost'][p, step, layer]
                    audit = audit_trace(g, py)
                    stem = f'input{copies}.{name}.L{layer}.q{query}'
                    np.savez_compressed(trace_dir/(stem+'.npz'), native=tr, python=py['trace'], selected=np.r_[route[0], 64])
                    dump(trace_dir/(stem+'.nodes.json'), g.nodes)
                    traces.append(dict(plan=name, input_copies=copies, layer=layer, query=query, cost_ns=int(cost[0]),
                                       bound_ns=int(bound[0]), all_nodes_exact=True, **audit))
    trace_seconds = time.monotonic()-t0-cost_seconds
    dump(directory/'verification.json', dict(status='PASS', L0_primary_bridges=bridges, traces=traces,
        exact_python_traces=len(traces), costs=163840, native_seconds=cost_seconds, post_cost_physical_and_trace_seconds=trace_seconds,
        eight_calls=durations, native_binary_sha256=sha(native.binary), native_source_sha256=sha(a.CPP),
        all40layer_coverage=True, no_new_model_inference=True, physical_address_relabel_only=True))
    analysis = analyze(arrays, raw, directory)
    check_frozen()
    dump(directory/'completion.frozen.json', dict(status='sealed_complete_secondary_all40_fixed_map_transfer',
        utc=datetime.now(timezone.utc).isoformat(), execution_protocol_sha256=sha(OUT/'execution_protocol.frozen.json'),
        parent_protocol_sha256=ROOT_SHA, root_release_sha256=RELEASE_SHA, total_wall_seconds=time.monotonic()-t0,
        files=[dict(path=str(p), sha256=sha(p), bytes=p.stat().st_size) for p in sorted(directory.rglob('*')) if p.is_file()],
        no_fit=True, primary_untouched=True, supplementary_only=True))
    print(json.dumps(dict(completion_sha256=sha(directory/'completion.frozen.json'), total_wall_seconds=time.monotonic()-t0, analysis=analysis), indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=('freeze', 'run'))
    args = parser.parse_args(); {'freeze': freeze, 'run': run}[args.mode]()
