#!/usr/bin/env python3
"""Post-run verification using the existing independently written trace checker.

Not a new independent scheduler. No capture, target executor, or target graph
builder is called. No sample, layout, comparison or fit is selected here.
"""
from collections import Counter
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/research-2026-11-02/round3-cr-ipr'
OUT = BASE / 'second-prospective-routes'
DATA = OUT / 'comparison'
DEST = OUT / 'verification'


def need(ok, msg):
    if not ok:
        raise RuntimeError(msg)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


def main():
    DEST.mkdir(exist_ok=True)
    need(not (OUT / 'manifest.frozen.json').exists(), 'second diagnostic already sealed')
    receipt = read(OUT / 'comparison.protocol.frozen.json')
    capture = read(OUT / 'report.json')
    original = read(DATA / 'report.json')
    pre = read(OUT / 'pre_execution.receipt.json')
    need(receipt['utc'] < pre['utc'] and pre['comparison_protocol_sha256'] == sha(OUT / 'comparison.protocol.frozen.json'), 'pre-execution comparison/prompt chronology')
    need(capture['protocol_sha256'] == sha(OUT / 'protocol.frozen.json') and capture['prompt_count'] == 16 and capture['decode_inputs'] == 256, 'all16 capture inventory')
    need(original['comparison_protocol_sha256'] == sha(OUT / 'comparison.protocol.frozen.json') and original['route_sha256'] == capture['decode_sha256'] == sha(OUT / 'decode.routes.npz'), 'new routes bound to fixed comparison')
    for path, expected in receipt['inputs_sha256'].items():
        need(sha(path) == expected, 'pre-model frozen dependency changed: ' + path)
    for job in receipt['prompts']:
        need(sha(job['path']) == job['prompt_sha256'], 'prompt changed after freeze')
    for batch in range(2):
        cmd = read(OUT / f'batch{batch}.command.json')['argv']
        need(cmd[3:5] == ['16', '8'] and sha(cmd[0]) == receipt['inputs_sha256'][cmd[0]], 'unchanged16steps8threads helper binary')
        need(Path(cmd[2]).read_text().count('\n') == 8, 'each batch8prompts')
    binder_path = ROOT / 'scripts/research_20261102_joint_shared_full_ffn_audit.py'
    base_path = ROOT / 'scripts/research_20261102_cr_ipr_full_ffn_audit.py'
    helper_path = ROOT / 'scripts/research_20261102_g2_availability_full_ffn_audit.py'
    b = load('second_reused_address_binder', binder_path)
    c = load('second_reused_scalar_checker', base_path)
    h = load('second_reused_critical_helper', helper_path)
    with np.load(OUT / 'decode.routes.npz') as z:
        need(z['routes'].shape == (16, 16, 40, 6) and z['completed'].all(), 'all256 complete inputs40layers')
        routes = z['routes'].transpose(2, 0, 1, 3).reshape(40, 256, 6)[:3]
    with np.load(DATA / 'service_arrays.npz') as z:
        arrays = {key: z[key].copy() for key in z.files}
    with np.load(BASE / 'g2-availability-full-ffn/support_maps.frozen.npz') as z:
        adjacency = z['selected'].copy()
    with np.load(BASE / 'full-ffn/g8-addendum/group_maps.frozen.npz') as z:
        groups = z['learned'].copy()
    g2_residents = [h.expected_residents(adj) for adj in adjacency]
    joint_residents, jointadj = b.joint_residents()
    need(len(original['records']) == receipt['configuration_count'] == 16, 'all16profiles retained')
    need(set(x['name'] for x in original['records']) == set(x['name'] for x in receipt['axes']), 'exactpredeclaredaxes')
    summary = []
    for record in original['records']:
        name = record['name']
        family, array, plan, copies, priority = name.split('.')
        cost, fifo, cap = [arrays[name + '.' + suffix] for suffix in ['cost', 'fifo', 'matching_capacity']]
        need(cost.shape == fifo.shape == cap.shape == (3, 256), '768entriesperprofile')
        need(record['requested_events'] == 768 and np.count_nonzero(cost >= 0) == record['completed_events'], 'allfailuresretained')
        valid = (cost >= 0) & (fifo >= 0) & (fifo <= 128)
        need(int(valid.sum()) == record['FIFO128_valid_events'], 'exactFIFOvalidcount')
        if np.all(cost >= 0):
            need(b.stats(cost) == record['cost_ns'] and b.stats(cost.sum(axis=0)) == record['three_layer_sum_ns'], 'allquerymean/P99/aligned3layerarithmetic')
        want = np.zeros((3, 256), dtype=np.int64)
        if plan != 'g4triples':
            for layer in range(3):
                adj = jointadj if plan == 'joint8' else adjacency[layer]
                want[layer] = [1 if b.cap1_feasible(route, adj) else 2 for route in routes[layer]]
        need(np.array_equal(cap, want), 'all768route/supportcapacitybinding')
        need(record['fallback_events'] == int(np.count_nonzero(cap == 2)), 'everycap2fallbackretained')
        summary.append(dict(name=name, events=768, mean_ns=float(cost.mean()), event_P99_ns=float(np.percentile(cost, 99)),
                            aligned_three_layer_P99_ns=float(np.percentile(cost.sum(axis=0), 99)),
                            fallback_events=int(np.count_nonzero(cap == 2)), FIFO128_valid_events=int(valid.sum())))
    expected_pairs = {(x['candidate'], x['reference']) for x in receipt['comparisons']}
    need({(x['candidate'], x['reference']) for x in original['comparisons']} == expected_pairs, 'all14predeclaredpairedcomparisons')
    for pair in original['comparisons']:
        new, old = pair['candidate'], pair['reference']
        nv, ov = arrays[new + '.cost'], arrays[old + '.cost']
        valid = (nv >= 0) & (ov >= 0) & (arrays[new + '.fifo'] <= 128) & (arrays[old + '.fifo'] <= 128)
        need(pair['jointly_valid_events'] == int(valid.sum()), 'pairedvalidsamplecount')
        if valid.any():
            x, y = nv[valid], ov[valid]
            expected = dict(mean_difference_ns=float((x - y).mean()), mean_reduction_pct=100 * (1 - float(x.mean() / y.mean())),
                            event_P99_reduction_pct=100 * (1 - float(np.percentile(x, 99) / np.percentile(y, 99))),
                            slower_events=int(np.count_nonzero(x > y)), equal_events=int(np.count_nonzero(x == y)), faster_events=int(np.count_nonzero(x < y)))
            need(all(pair[k] == v for k, v in expected.items()), 'pairedmean/P99/slower/equal/fasterarithmetic')
            if valid.all():
                need(pair['three_layer_sum_P99_reduction_pct'] == 100 * (1 - float(np.percentile(nv.sum(axis=0), 99) / np.percentile(ov.sum(axis=0), 99))), 'aligned3layerP99comparison')
    records, times, critical = [], {}, {}
    files = sorted(DATA.glob('trace.*json'))
    need(sum('.first.q0.' in p.name for p in files) == 48, 'everyprofile/layerfirsttracewithoutcacheomission')
    for index, path in enumerate(files):
        parts = path.stem.split('.')
        layer, family, array, plan, copies, priority, sample, qpart = parts[1:]
        layer, query = int(layer[1:]), int(qpart[1:])
        name = '.'.join(parts[2:7])
        record = read(path)
        native = array if family == 'g8_fixed' else None
        triple, isjoint = plan == 'g4triples', plan == 'joint8'
        cap = int(arrays[name + '.matching_capacity'][layer, query])
        resident = None if triple else joint_residents if isjoint else g2_residents[layer]
        adj = None if triple else jointadj if isjoint else adjacency[layer]
        need(record['service']['input_copies'] == int(copies[5:]) and record['service']['down_first'] == 0, 'predeclaredinputcopies/GF')
        trace, cost = c.scalar_execute(record['nodes'], 0)
        want = np.array([[n['start_ns'], n['end_ns']] for n in record['nodes']], dtype=np.int64)
        need(np.array_equal(trace, want) and cost == record['cost_ns'] == int(arrays[name + '.cost'][layer, query]), 'everysavednodeexacteventreplay')
        semantic = c.semantic_audit(record, trace)
        binding = b.bind(record, routes[layer, query], family, resident, adj, cap, native, groups[layer], isjoint, triple)
        need(semantic['peak_FIFO_bytes'] == int(arrays[name + '.fifo'][layer, query]), 'exactcontinuousFIFOquerybinding')
        if sample == 'fallback':
            need(cap == 2, 'savedfallbackcapacity')
        records.append(dict(path=str(path), sha256=sha(path), name=name, layer=layer, query_index=query,
                            sample=sample, matching_capacity=cap, cost_ns=cost, **semantic, binding=binding))
        times[path.name] = trace;critical[path.name] = h.critical_path(record)
        if (index + 1) % 12 == 0:
            print(json.dumps(dict(independent_saved_traces=index + 1, total=len(files))), flush=True)
    np.savez_compressed(DEST / 'independent_node_times.npz', **times)
    write(DEST / 'critical_paths.json', critical)
    write(DEST / 'report.json', dict(status='PASS', source_sha256=sha(__file__),
          reused_checker_sha256={str(p): sha(p) for p in [base_path, binder_path, helper_path]},
          prompts=16, configurations=16, array_entries=16 * 768, saved_traces=len(files), records=records,
          total_nodes=sum(x['nodes'] for x in records), total_jobs=sum(x['jobs'] for x in records),
          all_saved_start_end_exact=True, summaries=summary,
          scope='Existing independent scalar/semantic/FIFO/physicalbind checker reused on all saved representatives; all arrays arithmetic/routebindingchecked, not everyqueryeventreplayed. No new independent executor, newnumeric/quality study or newCBC theorem.'))
    comparisons = original['comparisons']
    lines = ['# Second authored diagnostic\n',
             'All16newprompts and16comparisonaxes were frozen before this model execution. No prompts, maps, fits, profiles or winners were selected or replaced after results.\n',
             'This is a second authored diagnostic, not a representative benchmark, task-quality result or extrapolation guarantee. Universal top6matching comes from the existing Hall certificate. CanonicalW8/A8 routes and conditional timing retain the earlier numerical/hardware limits.\n',
             f'Capture:16prompts×16steps×40layers; comparison:first3layers×256inputs×16profiles={16*768}events. All {original["failures"]} failures are retained. Saved-trace independent replay covers{len(files)}representatives, not allquerygraphs.\n',
             '| Configuration | Mean µs | Event P99 µs | 3layer P99 µs | Fallbacks | FIFO128valid |',
             '|---|---:|---:|---:|---:|---:|']
    for record in summary:
        lines.append(f'| {record["name"]} | {record["mean_ns"]/1000:.6f} | {record["event_P99_ns"]/1000:.6f} | {record["aligned_three_layer_P99_ns"]/1000:.6f} | {record["fallback_events"]} | {record["FIFO128_valid_events"]}/768 |')
    lines += ['', '| New / reference | Mean reduction % | Event P99 reduction % | 3layer P99 reduction % | Slower/equal/faster | MiB new/reference |',
              '|---|---:|---:|---:|---|---|']
    for pair in comparisons:
        lines.append(f'| {pair["candidate"]} / {pair["reference"]} | {pair.get("mean_reduction_pct",float("nan")):.6f} | {pair.get("event_P99_reduction_pct",float("nan")):.6f} | {pair.get("three_layer_sum_P99_reduction_pct",float("nan")):.6f} | {pair.get("slower_events",0)}/{pair.get("equal_events",0)}/{pair.get("faster_events",0)} | {pair["candidate_payload_MiB"]}/{pair["reference_payload_MiB"]} |')
    lines += ['', 'Sources are separate from all earlier sealed campaigns. comparison.protocol.frozen.json is the authoritative evaluation plan, superseding only the old evaluation paragraph inherited by the exact capture clone. The clone differs only in OUT/PROMPTS; its capture/extraction/run logic is identical.',
              '', 'Commands:', '', '```sh',
              'python scripts/research_20261102_cr_second_prospective_replay.py freeze',
              'python scripts/research_20261102_cr_second_prospective_replay.py capture',
              'python scripts/research_20261102_cr_second_prospective_replay.py run',
              'python scripts/research_20261102_cr_second_prospective_audit.py', '```',
              '', 'Final manifest hashes the two protocols, prompttexts, rawcapture reports/logs/routes/ledgers, frozen dependencies, all comparison arrays and savedtraces, and independent verification. No existing sealed source or campaign was changed.']
    (OUT / 'README.md').write_text('\n'.join(lines) + '\n')
    # Hash every produced artifact, retaining full provenance without embedding
    # large raw weights, generated tokens or text in the human report.
    for path, expected in receipt['inputs_sha256'].items():
        need(sha(path) == expected, 'frozen dependency changed during independent audit: ' + path)
    paths = sorted(p for p in OUT.rglob('*') if p.is_file() and p.name != 'manifest.frozen.json')
    write(OUT / 'manifest.frozen.json', dict(status='sealed_second_authored_diagnostic', utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
          inputs_sha256=receipt['inputs_sha256'], verification_source_sha256=sha(__file__),
          outputs_sha256={str(p): sha(p) for p in paths}, no_inplace_modifications=True))
    print(json.dumps(dict(status='PASS', saved_traces=len(files), verification_sha256=sha(DEST / 'report.json'),
                          report_sha256=sha(DATA / 'report.json'), manifest_sha256=sha(OUT / 'manifest.frozen.json'))), flush=True)


if __name__ == '__main__':
    main()
