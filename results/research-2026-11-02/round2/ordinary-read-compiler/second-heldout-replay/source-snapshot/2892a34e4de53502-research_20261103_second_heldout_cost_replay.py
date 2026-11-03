#!/usr/bin/env python3
"""Replay sealed TRAIN-selected physical plans on the second W8 heldout.

This is a declared ordinary-read service model, not measured NAND performance.
All layout families and cost axes are fixed before reading the new routes.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from research_20261103_ordinary_read_backend import (
    ROOT, NativeReplay, ReferenceGraph, Service, lower_rows, physical_addresses,
    sha, validate_plan,
)

BASE = ROOT / 'results/research-2026-11-02/round2'
SEAL = BASE / 'ordinary-read-compiler/W8_selected_plans.frozen.json'
EXPECTED_SEAL = '00507ff56f9197fb18511ad58303919214c03cba6f0c1440f6733ffc42944d00'
MANIFEST = BASE / 'second-format-heldout/manifest.json'
EXPECTED_MANIFEST = 'ccfabf3b3e1a4293f2d0354cdea289b7d94197fc6733253601ca97448dabcd8f'
NUMERIC = BASE / 'selected-plan-numeric/second-heldout'
DEST = BASE / 'ordinary-read-compiler/second-heldout-replay'
SHAPES = {'deepseek': (2048, 1408, 64, 6, 26, 2816),
          'granite4tiny': (1536, 512, 64, 6, 40, 1024)}
FAMILIES = ('disjoint', 'cyclic', 'whole_rows')
SELF = Path(__file__).resolve()


def write(path, data):
    path = Path(path)
    temp = path.with_name(path.name + '.writing')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    temp.replace(path)


def utc():
    return datetime.now(timezone.utc).isoformat()


def stats(values):
    a = np.asarray(values, dtype=np.int64)
    return dict(samples=int(a.size), mean_ns=float(a.mean()),
                p50_ns=float(np.percentile(a, 50)), p95_ns=float(np.percentile(a, 95)),
                p99_ns=float(np.percentile(a, 99)), min_ns=int(a.min()), max_ns=int(a.max()))


def freeze():
    DEST.mkdir(parents=True, exist_ok=True)
    target = DEST / 'protocol.frozen.json'
    if target.exists():
        raise RuntimeError('Never overwrite a frozen analysis protocol.')
    assert sha(SEAL) == EXPECTED_SEAL and sha(MANIFEST) == EXPECTED_MANIFEST
    seal = json.loads(SEAL.read_text())
    manifest = json.loads(MANIFEST.read_text())
    items = manifest['items']
    assert len(items) == 16 and len({p['id'] for p in items}) == 16
    assert sorted(set(p['dataset'] for p in items)) == ['allenai/openbookqa', 'google/boolq']
    sources = [SELF, Path(__file__).with_name('research_20261103_ordinary_read_backend.py'),
               Path(__file__).with_name('research_20261103_ordinary_read_native.cpp'),
               Path(__file__).with_name('research_20261102_whole_row_compiler.py')]
    inventory = BASE / 'deepseek-full-ffn-integer-reference/frozen_equal_projection_rows.json'
    inputs = [SEAL, MANIFEST, inventory]
    inputs += [Path(p['prompt_path']) for p in items]
    tasks = []
    for campaign in seal['campaigns']:
        for key in ('summary', 'protocol', 'snapshot_manifest'):
            path = ROOT / campaign[key]
            assert sha(path) == campaign[key + '_sha256']
            inputs.append(path)
        summary = json.loads((ROOT / campaign['summary']).read_text())
        copies = campaign['input_channel_copies']
        assert copies in (1, 4)
        for record in summary['records']:
            model = record['model']
            assert model in SHAPES and record['shape'] == list(SHAPES[model])
            assert record['command_ns'] in (250, 5000)
            raw = record['training']['physical_model_layer']
            assert raw == record['layer'] + (model == 'deepseek')
            service = Service(**record['service'])
            assert service.input_channel_copies == copies and service.output_fringe == 1
            for artifact in record['artifacts']:
                assert artifact['family'] in FAMILIES
                for key in ('plan', 'arrays'):
                    path = ROOT / artifact[key]
                    assert sha(path) == artifact[key + '_sha256']
                    inputs.append(path)
                tasks.append(dict(model=model, moe_layer_index=record['layer'], raw_layer=raw,
                                  input_channel_copies=copies, command_ns=record['command_ns'],
                                  family=artifact['family'], service=asdict(service),
                                  arrays=str(ROOT / artifact['arrays']), plan=str(ROOT / artifact['plan'])))
    assert len(tasks) == (26 + 40) * 2 * 2 * 3
    sources_dir = DEST / 'source-snapshot'
    sources_dir.mkdir()
    for p in sources:
        (sources_dir / (sha(p)[:16] + '-' + p.name)).write_bytes(p.read_bytes())
    protocol = dict(frozen_utc=utc(), plan_seal_sha256=EXPECTED_SEAL,
                    manifest_sha256=EXPECTED_MANIFEST,
                    inputs_sha256={str(p): sha(p) for p in sorted(set(inputs))},
                    sources_sha256={str(p): sha(p) for p in sources},
                    tasks=tasks, workers=8, prompts=items,
                    route_shape={m: [16, 16, cfg[4], cfg[3]] for m, cfg in SHAPES.items()},
                    numeric_completion_requirement='All four paired canonical/placed campaigns complete; independent numeric audits pass; input1/input4 canonical routes, logits and continuation tokens identical. Require all completed masks true. No filtering of prompts or decode positions.',
                    fit='None. Read all sealed selected plans verbatim; no scheduling search, retile, relabel or parameter adjustment on heldout.',
                    analysis='Sum every MoE+shared layer for each aligned prompt/decode position, then calculate pooled, dataset and paired per-prompt means/P99. All three TRAIN-selected families, command250/5000ns and ONFI inputcopy1/4 scenarios reported. Preserve ordered selected experts. Descriptive16prompt sample;256 correlated decode positions are not independent tasks.',
                    dense_prefix='DeepSeek raw0 dense fixed equal complete rows: GU1368rows/chip,C2048;Down128rows/chip,C10944. Same prefix exactly once per token for every family. Report MoE-only and complete-FFN-service separately. Cost-equivalent GU concatenation is a row-order/scatter abstraction; numeric harness retains separate gate/up operators. Prefix address ledger is separate, not falsely included in MoE global maps.',
                    trace_validation='Save first prompt/decode0 native trace for all792 layer/layout/profile cases. Independent ReferenceGraph exact trace check on the first MoE layer of every24 model/layout/profile combination, plus all four dense-prefix scenarios. Selection fixed in advance.',
                    limitations=['Uncalibrated software service and virtual addresses, not measured device, PE or full-model wall time.',
                                 'Fixed W8 full-row/dynamicA8 CPU reference; source attention/router unchanged. No task accuracy or original-checkpoint equivalence claim.',
                                 'Direct numeric selected-rowmap validation is four whole_rows lowcommand maps, not every counterfactual baseline map.',
                                 'FFN attention/router/SSM, activation/combine time(default0), row-scale/gather CPU costs and device overprovisioning are not covered by modeled FFN service.',
                                 'Native full-vector, complete-row and output-drain contract remains explicit; AiF output ABI, controller cost and per-plane broadcast are not calibrated.',
                                 'Finite fitted-pool TRAIN mean certificates do not prove globally optimal placement or heldout/tail performance.'])
    write(target, protocol)
    print(json.dumps(dict(status='frozen', protocol_sha256=sha(target), tasks=len(tasks), prompts=16)), flush=True)


def verify(protocol):
    for group in ('inputs_sha256', 'sources_sha256'):
        for path, digest in protocol[group].items():
            assert sha(path) == digest, path


def capture_routes(protocol):
    summary_path = NUMERIC / 'summary.json'
    summary = json.loads(summary_path.read_text())
    assert summary['status'] == 'complete' and summary['independent_audits_passed']
    assert summary['all_views_same_routes_and_continuation_logits']
    numeric_spec = json.loads((NUMERIC / 'protocol.frozen.json').read_text())
    assert numeric_spec['plan_seal_sha256'] == EXPECTED_SEAL
    assert [p['id'] for p in numeric_spec['prompts']] == [p['id'] for p in protocol['prompts']]
    assert [p['prompt_sha256'] for p in numeric_spec['prompts']] == [p['prompt_sha256'] for p in protocol['prompts']]
    hashes = {str(summary_path): sha(summary_path),
              str(NUMERIC / 'protocol.frozen.json'): sha(NUMERIC / 'protocol.frozen.json')}
    for model, cfg in SHAPES.items():
        arrays = []
        for copies in (1, 4):
            caseid = f'input{copies}-{model}'
            path = NUMERIC / caseid / 'decode.routes.npz'
            assert sha(path) == summary['cases'][caseid]['decode_archive_sha256']
            with np.load(path) as z:
                assert z['routes'].shape == (16, 16, cfg[4], cfg[3])
                assert z['completed'].shape == (16,) and z['completed'].all()
                assert z['split'].item() == 'second_heldout'
                assert z['prompt_ids'].tolist() == [p['id'] for p in protocol['prompts']]
                assert z['datasets'].tolist() == [p['dataset'] for p in protocol['prompts']]
                layers = np.arange(cfg[4], dtype=np.int32) + (model == 'deepseek')
                assert np.array_equal(z['layers'], layers)
                a = z['routes'].copy()
            assert np.all((a >= 0) & (a < cfg[2]))
            assert all(len(set(row.tolist())) == cfg[3] for row in a.reshape(-1, cfg[3]))
            arrays.append(a)
            hashes[str(path)] = sha(path)
        assert np.array_equal(*arrays), model
        np.save(DEST / (model + '.routes.npy'), arrays[0])
        hashes[str(DEST / (model + '.routes.npy'))] = sha(DEST / (model + '.routes.npy'))
    write(DEST / 'numeric_inputs.bound.json', dict(bound_utc=utc(), sha256=hashes,
          protocol_sha256=sha(DEST / 'protocol.frozen.json'), fitting='None; binding completed output hashes only.'))
    return hashes


_NATIVE = None
_ROUTES = {}


def initialize_worker():
    global _NATIVE, _ROUTES
    _NATIVE = NativeReplay(DEST / 'build')
    _ROUTES = {m: np.load(DEST / (m + '.routes.npy')) for m in SHAPES}


def worker(task):
    model = task['model']
    s = Service(**task['service'])
    path = Path(task['arrays'])
    with np.load(path) as z:
        plan = {k: z[k].copy() for k in ('offsets', 'jobs', 'input_columns', 'total_rows')}
    validate_plan(plan, SHAPES[model][2] + 1, s)
    routes = np.ascontiguousarray(_ROUTES[model][:, :, task['moe_layer_index']].reshape(-1, 6))
    result = _NATIVE.run(plan, routes, SHAPES[model][2], 1, s, trace=True)
    checked = task['moe_layer_index'] == 0
    if checked:
        reference = ReferenceGraph.build(plan, routes[0].tolist() + [SHAPES[model][2]], s).run()
        assert np.array_equal(result['trace'], reference['trace'])
        assert int(result['cost'][0]) == reference['makespan_ns']
    label = f"input{s.input_channel_copies}.{model}.cmd{s.command_ns}.{task['family']}.L{task['moe_layer_index']}"
    target = DEST / 'layers' / (label + '.npz')
    np.savez_compressed(target, cost=result['cost'].reshape(16, 16),
                        lower_bound=result['lower_bound'].reshape(16, 16),
                        first_trace=result['trace'], first_route=routes[0])
    return dict(**task, artifact=str(target), artifact_sha256=sha(target),
                mean_ns=float(result['cost'].mean()), p99_ns=float(np.percentile(result['cost'], 99)),
                reference_trace_exact=checked, first_trace_nodes=len(result['trace']))


def dense_prefix(native, copies, command):
    inventory = json.loads((BASE / 'deepseek-full-ffn-integer-reference/frozen_equal_projection_rows.json').read_text())
    byname = {entry['weight']: entry for entry in inventory}
    for phase, rows, columns in [('gate', 10944, 2048), ('up', 10944, 2048), ('down', 2048, 10944)]:
        entry = byname[f'blk.0.ffn_{phase}.weight']
        assert (entry['rows'], entry['columns'], entry['experts']) == (rows, columns, 1)
        assert entry['equal_rows_per_chip'] == [rows // 16] * 16
    s = Service(command_ns=command, input_channel_copies=copies, output_fringe=1, input_buffer_bytes=32768)
    plan = lower_rows(np.full((1, 16), 1368), np.full((1, 16), 128), 2048, 10944, s)
    result = native.run(plan, np.asarray([[0]], dtype=np.int32), 1, 0, s, trace=True)
    reference = ReferenceGraph.build(plan, [0], s).run()
    assert np.array_equal(result['trace'], reference['trace'])
    assert int(result['cost'][0]) == reference['makespan_ns']
    addresses = physical_addresses(plan, s)
    label = f'input{copies}.cmd{command}'
    arrays = DEST / 'dense-prefix' / (label + '.npz')
    np.savez_compressed(arrays, **{k: plan[k] for k in ('offsets', 'jobs', 'input_columns', 'total_rows')}, trace=result['trace'])
    address_path = DEST / 'dense-prefix' / (label + '.physical.json')
    write(address_path, dict(raw_model_layer=0, service=asdict(s), logical=plan['logical'], **addresses))
    return dict(input_channel_copies=copies, command_ns=command, cost_ns=int(result['cost'][0]),
                arrays_sha256=sha(arrays), physical_sha256=sha(address_path),
                useful_bytes=addresses['useful_bytes'], programmed_page_bytes=addresses['programmed_page_bytes'],
                reserved_wave_bytes=addresses['reserved_wave_bytes'],
                standalone_allocated_lsb_block_bytes=addresses['allocated_lsb_block_bytes'],
                note='Separate fixed-prefix map. Do not add standalone allocated blocks to claim a joint-packing optimum.')


def summarize(records, prefixes, protocol):
    allrows, paired, perprompt, groups = [], [], [], []
    items = protocol['prompts']
    for model in SHAPES:
        for copies in (1, 4):
            for command in (250, 5000):
                prefix = 0 if model != 'deepseek' else next(p['cost_ns'] for p in prefixes
                           if p['input_channel_copies'] == copies and p['command_ns'] == command)
                arrays = {}
                for family in FAMILIES:
                    selected = sorted([r for r in records if r['model'] == model and r['input_channel_copies'] == copies
                                       and r['command_ns'] == command and r['family'] == family], key=lambda r: r['moe_layer_index'])
                    assert [r['moe_layer_index'] for r in selected] == list(range(SHAPES[model][4]))
                    costs = np.stack([np.load(r['artifact'])['cost'] for r in selected]).sum(0)
                    arrays[family] = costs
                    label = f'input{copies}.{model}.cmd{command}.{family}'
                    np.savez_compressed(DEST / 'totals' / (label + '.npz'), moe_shared_ns=costs, complete_ffn_ns=costs+prefix)
                    allrows.append(dict(model=model, input_channel_copies=copies, command_ns=command, family=family,
                                        moe_shared=stats(costs), fixed_dense_prefix_ns=prefix, complete_ffn_service=stats(costs+prefix)))
                for baseline in ('disjoint', 'cyclic'):
                    a, b = arrays[baseline], arrays['whole_rows']
                    pair = dict(model=model, input_channel_copies=copies, command_ns=command, baseline=baseline,
                                moe_mean_reduction_fraction=1-float(b.mean())/float(a.mean()),
                                complete_ffn_mean_reduction_fraction=1-float(b.mean()+prefix)/float(a.mean()+prefix),
                                paired_mean_delta_ns=float((a-b).mean()),
                                faster_tokens=int((a>b).sum()), tied_tokens=int((a==b).sum()), slower_tokens=int((a<b).sum()),
                                faster_prompt_means=int((a.mean(1)>b.mean(1)).sum()), tied_prompt_means=int((a.mean(1)==b.mean(1)).sum()),
                                slower_prompt_means=int((a.mean(1)<b.mean(1)).sum()))
                    paired.append(pair)
                    for pi, item in enumerate(items):
                        perprompt.append(dict(model=model, input_channel_copies=copies, command_ns=command, baseline=baseline,
                            prompt_id=item['id'], dataset=item['dataset'], baseline_moe=stats(a[pi]), whole_rows_moe=stats(b[pi]),
                            moe_mean_reduction_fraction=1-float(b[pi].mean())/float(a[pi].mean()),
                            complete_ffn_mean_reduction_fraction=1-float(b[pi].mean()+prefix)/float(a[pi].mean()+prefix)))
                    for dataset in sorted(set(p['dataset'] for p in items)):
                        indices = [i for i,p in enumerate(items) if p['dataset']==dataset]
                        aa, bb = a[indices], b[indices]
                        groups.append(dict(model=model,input_channel_copies=copies,command_ns=command,baseline=baseline,dataset=dataset,
                            prompts=len(indices),baseline_moe=stats(aa),whole_rows_moe=stats(bb),
                            moe_mean_reduction_fraction=1-float(bb.mean())/float(aa.mean()),
                            complete_ffn_mean_reduction_fraction=1-float(bb.mean()+prefix)/float(aa.mean()+prefix)))
    write(DEST / 'per_prompt.json', perprompt)
    write(DEST / 'dataset_groups.json', groups)
    with (DEST / 'comparison.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(paired[0]))
        w.writeheader(); w.writerows(paired)
    table = ['| Model | ONFI input copies | Command ns | Cyclic MoE+shared ms | Whole MoE+shared ms | MoE gain | Complete FFN gain |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for row in paired:
        if row['baseline'] != 'cyclic': continue
        lookup = {r['family']:r for r in allrows if all(r[k]==row[k] for k in ('model','input_channel_copies','command_ns'))}
        table.append(f"| {row['model']} | {row['input_channel_copies']} | {row['command_ns']} | {lookup['cyclic']['moe_shared']['mean_ns']/1e6:.6f} | {lookup['whole_rows']['moe_shared']['mean_ns']/1e6:.6f} | {row['moe_mean_reduction_fraction']:.4%} | {row['complete_ffn_mean_reduction_fraction']:.4%} |")
    (DEST / 'comparison.md').write_text('\n'.join(table)+'\n\nUncalibrated service model; complete FFN adds DeepSeek raw0 dense only. Attention/router/SSM and host row-scale/gather excluded. No heldout fitting.\n')
    return allrows, paired


def run():
    protocol_path = DEST / 'protocol.frozen.json'
    protocol = json.loads(protocol_path.read_text()); verify(protocol)
    if (DEST / 'started.json').exists():
        raise RuntimeError('Never overwrite an executed heldout campaign.')
    bound = capture_routes(protocol)
    for name in ('layers', 'dense-prefix', 'totals'):
        (DEST / name).mkdir()
    native = NativeReplay(DEST / 'build')
    write(DEST / 'started.json', dict(started_utc=utc(), protocol_sha256=sha(protocol_path), native=native.provenance))
    began = time.monotonic()
    prefixes = [dense_prefix(native, copies, command) for copies in (1,4) for command in (250,5000)]
    records=[]
    with ProcessPoolExecutor(max_workers=protocol['workers'], initializer=initialize_worker) as pool:
        for future in as_completed([pool.submit(worker,t) for t in protocol['tasks']]):
            records.append(future.result())
            if len(records)%66==0:
                print('PROGRESS', json.dumps(dict(completed_layer_cases=len(records),total=len(protocol['tasks']),elapsed_seconds=round(time.monotonic()-began,1))), flush=True)
    records.sort(key=lambda r:(r['model'],r['input_channel_copies'],r['command_ns'],r['family'],r['moe_layer_index']))
    totals, paired = summarize(records,prefixes,protocol)
    verify(protocol)
    assert all(sha(p)==h for p,h in bound.items())
    report=dict(status='complete',completed_utc=utc(),protocol_sha256=sha(protocol_path),elapsed_seconds=time.monotonic()-began,
                costs=len(records)*256,full_reference_traces=sum(r['reference_trace_exact'] for r in records),
                saved_native_traces=len(records),saved_native_trace_nodes=sum(r['first_trace_nodes'] for r in records),
                native=native.provenance,records=records,dense_prefix=prefixes,totals=totals,paired=paired,
                limits=protocol['limitations'])
    write(DEST / 'summary.json',report)
    print(json.dumps(dict(status=report['status'],costs=report['costs'],elapsed_seconds=report['elapsed_seconds'],paired=paired),indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','run']);args=parser.parse_args()
    if args.action=='freeze':freeze()
    else:run()
