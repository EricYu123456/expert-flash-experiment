#!/usr/bin/env python3
"""Train-only ordinary-read geometry/label compiler with executable addresses.

Standard covering DP, balanced expert-label swaps, cyclic declustering and
branch-and-bound are controls/components, not new algorithms. This integrated
prototype tests whether native read/dispatch costs select useful physical
whole-row geometries. No CR-read, column partials or weight replication.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import datetime, timezone
import ctypes
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from research_20261102_whole_row_compiler import Capability, certificate, minimum_work
from research_20261103_ordinary_read_backend import (
    ROOT, NativeReplay, Service, ceildiv, emitted_rows, lower_rows,
    physical_addresses, sha, split_complete_rows,
)

BASE = ROOT / 'results/research-2026-11-02/round2'
DEST = BASE / 'ordinary-read-compiler/development'
SHAPES = {
    'granite': (1024, 512, 32, 8, 24, 0),
    'olmoe': (2048, 1024, 64, 8, 16, 0),
    'deepseek': (2048, 1408, 64, 6, 26, 2816),
    'qwen3': (2048, 768, 128, 8, 48, 0),
    'mixtral': (4096, 14336, 8, 2, 32, 0),
    'granite4tiny': (1536, 512, 64, 6, 40, 1024),
    'qwen38': (2560, 640, 512, 10, 48, 640),
}


def write(path, value):
    path = Path(path); temporary = path.with_name(path.name + '.writing')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def training(model, layer, fit_limit, manifest=None):
    if manifest is not None:
        specification = json.loads(Path(manifest).read_text())
        record = specification['models'][model]
        p = Path(record['archive']); assert sha(p) == record['sha256']
        assert record['status'] == 'complete'
        with np.load(p) as f:
            assert f['split'].item() == 'train'
            assert f['representation'].item() == 'fixed_full_FFN_W8A8_canonical_only'
            assert f['completed'].all(), 'Do not fit incomplete capture slots.'
            assert f['routes'].shape == (32, 16, SHAPES[model][4], SHAPES[model][3])
            assert f['layers'].tolist() == record['layers']
            assert len(set(f['prompt_ids'].tolist())) == 32
            assert all(str(s).startswith('train_') for s in f['prompt_ids'])
            physical_layer = int(f['layers'][layer])
            a = f['routes'][:, :, layer].reshape(-1, SHAPES[model][3])
            assert np.all((a >= 0) & (a < SHAPES[model][2]))
            assert all(len(set(r.tolist())) == len(r) for r in a)
        protocol = 'Frozen actual complete-FFN W8A8 TRAIN capture,32balanced TRAIN prompts×16decode; no prefill/test/fresh data. Source attention/router unchanged GGUF.'
    elif model == 'qwen38':
        p = BASE / 'qwen38-trace-audit/train_decode.npz'
        with np.load(p) as f:
            a = f['routes'][:, ::4, layer].reshape(-1, 10)
        protocol = 'All16requests represented; original token stride4 inside every request; development source previously used.'
    elif model == 'granite4tiny':
        d = json.loads((BASE / 'geometry-study/granite4tiny.frozen.json').read_text())
        p = Path(d['training_routes'])
        with np.load(p) as f:
            physical_layer = int(f['layers'][layer])
            a = f['routes'][:, :, layer].reshape(-1, 6)
        protocol = 'Existing32training prompts×64decodes, allrepresented before deterministic uniform subsampling.'
    else:
        d = json.loads((ROOT / 'results/plane-wave-v2/no-cr' / (model + '.frozen_plan.json')).read_text())
        p = Path(d['training_cache']['path'])
        assert sha(p) == d['training_cache']['sha256']
        with np.load(p) as f:
            physical_layer = int(f['layers'][layer])
            a = f['routes'][:, :, layer].reshape(-1, SHAPES[model][3])
        protocol = 'Existing200training prompts×128decodes; deterministic uniform training-only subsampling.'
    indices = np.unique(np.linspace(0, len(a) - 1, min(fit_limit, len(a)), dtype=np.int64))
    return np.ascontiguousarray(a[indices], dtype=np.int32), dict(path=str(p), sha256=sha(p),
             moe_layer_index=layer, physical_model_layer=layer if model == 'qwen38' else physical_layer,
             representation='fixed_full_FFN_W8A8_canonical_only' if manifest is not None else 'source-checkpoint routing with nominal I8 service',
             selection=protocol, original_events=len(a), selected_indices=indices.tolist())


def candidates(rows, columns, service):
    m = service.chips; assert m == 16, 'Label fitter currently supports16chips; backend is general.'
    cap = Capability(chips=m, planes=service.planes, page_bytes=service.page_bytes,
                     block_wordlines=service.block_wordlines)
    cert = certificate(rows, columns, cap); out = []
    for family, widths in [('disjoint', [g for g in range(1, m+1) if m % g == 0]),
                            ('cyclic', range(1, m+1))]:
        for width in widths:
            base = np.zeros(m, dtype=np.int64)
            base[:width] = rows // width; base[:rows % width] += 1
            variants = []
            for v in range(m):
                shift = (v % (m // width)) * width if family == 'disjoint' else v
                variants.append(np.roll(base, shift))
            out.append(dict(label=f'{family}_{width}', family=family,
                            variants=np.asarray(variants, dtype=np.int64)))
    profiles = [('equal_wave_profile', cert['equal_wave_profile']),
                ('balanced_wave_profile', cert['balanced_wave_profile'])]
    profiles += [(f'depth_{c["requested_max_job_waves"]}', c['witness'])
                 for c in cert['depth_constrained_certificates']]
    # Neighboring legal depths make work/depth alternatives explicit. This is
    # a bounded candidate family, not all integer partitions or native jobs.
    dmin = cert['arithmetic']['minimum_job_depth']
    for depth in range(dmin + 1, min(dmin + 3, service.block_wordlines + 1)):
        profiles.append((f'depth_{depth}', minimum_work(rows, columns, depth, cap)['witness']))
    seen = set()
    for name, profile in profiles:
        if profile is None:
            continue
        base = np.asarray(profile['rows'], dtype=np.int64); key = tuple(base.tolist())
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(label=name, family='whole_rows',
                        variants=np.asarray([np.roll(base, v) for v in range(m)])))
    return out, cert


def candidate_cost(candidate, columns, service):
    work = np.zeros((16, 16), dtype=np.int64)
    transfer = work.copy(); commands = work.copy()
    outputs = np.zeros(16, dtype=np.int64)
    cache = {}
    for label, rows in enumerate(candidate['variants']):
        for chip, n in enumerate(rows):
            if not n:
                continue
            if int(n) not in cache:
                pieces = split_complete_rows(int(n), columns, service)
                w = xfer = outpcie = 0
                for piece in pieces:
                    pages = ceildiv(piece * columns, service.page_bytes)
                    waves = ceildiv(pages, service.planes)
                    d = [chip, piece, columns, pages, waves]
                    ibytes = columns * service.input_bytes * service.input_channel_copies
                    obytes = emitted_rows(d, service) * service.output_bytes
                    tr = service.transfer(ibytes, service.channel_bytes_per_second) + service.transfer(obytes, service.channel_bytes_per_second)
                    w += waves * service.ordinary_read_ns + service.command_ns + tr
                    xfer += tr; outpcie += service.transfer(obytes, service.pcie_bytes_per_second)
                cache[int(n)] = (w, xfer, len(pieces), outpcie)
            w, xfer, count, outpcie = cache[int(n)]
            work[label, chip] = w; transfer[label, chip] = xfer
            commands[label, chip] = count; outputs[label] += outpcie
    assert np.all(commands.sum(1) == commands.sum(1)[0])
    return dict(chip=work, transfer=transfer, commands=commands, pcie=outputs)


def initial_labels(candidate, experts):
    if candidate['family'] == 'disjoint':
        width = int(candidate['label'].split('_')[-1])
        return np.arange(experts, dtype=np.int32) % (16 // width)
    return (np.arange(experts, dtype=np.int32) * max(1, 16 // experts)) % 16


def deduplicate(candidates_, experts):
    """One fit per identical physical label-search space and inventory.

    Under unrestricted pair swaps, the assigned physical-rowvector histogram
    exactly determines reachable assignments. Unused ordinal labels can never
    enter a fit; duplicate ordinal labels can always be distributed back to
    their prescribed counts. This equivalence would need revision for single
    label moves, expert-specific constraints or a different optimizer.
    """
    groups = {}; out = []
    for ci, c in enumerate(candidates_):
        variants = [tuple(row.tolist()) for row in c['variants']]
        assigned = {}
        for v in initial_labels(c, experts):
            row = variants[v]; assigned[row] = assigned.get(row, 0) + 1
        key = tuple(sorted(assigned.items()))
        if key in groups:
            canonical = groups[key]
            canonical['aliases'].append(c['label'])
            if c['family'] not in canonical['families']:
                canonical['families'].append(c['family'])
        else:
            canonical = dict(c, seed_index=ci, aliases=[c['label']], families=[c['family']],
                             geometry_sha256=hashlib.sha256(repr(key).encode()).hexdigest())
            groups[key] = canonical; out.append(canonical)
    return out


def fit_phase(native_labels, candidate, columns, routes, experts, service, iterations, seed):
    costs = candidate_cost(candidate, columns, service)
    unique, weights = np.unique(np.sort(routes, axis=1), axis=0, return_counts=True)
    unique = np.ascontiguousarray(unique, dtype=np.int32); weights = weights.astype(np.int64)
    initial = initial_labels(candidate, experts)
    table = np.zeros((2, 16, 16), dtype=np.int64); table[0] = costs['chip']
    labels = np.tile(initial, (2, 1)); best = np.inf; runs = []
    for si, temperature in enumerate((0., 2000.)):
        fitted = np.tile(initial, (2, 1)); stats = np.zeros(2, dtype=np.int64)
        value = native_labels.fit_native_labels(fitted, table, unique, weights,
                    experts, len(unique), routes.shape[1], 16, 0, iterations,
                    seed + si, temperature, stats)
        actual = costs['chip'][fitted[0]][routes].sum(1).max(1).mean()
        assert abs(value / 1001 - actual) < 1e-5
        assert np.array_equal(np.sort(initial), np.sort(fitted[0]))
        runs.append(dict(temperature_ns=temperature, valid=int(stats[0]), accepted=int(stats[1])))
        if actual < best:
            best = actual; labels = fitted
    c = dict(candidate); c.update(costs=costs, labels=labels[0].copy(),
                                 rows=candidate['variants'][labels[0]], mean_phase_chip_bound_ns=float(best), fit=runs)
    return c


def reuse_phase(candidate, columns, routes, experts, service, saved):
    assert candidate['family'] == saved['family']
    assert candidate['variants'][0].tolist() == saved['base_rows']
    labels = np.asarray(saved['labels'], dtype=np.int32)
    assert labels.shape == (experts,) and np.all((labels >= 0) & (labels < 16))
    assert np.array_equal(np.sort(labels), np.sort(initial_labels(candidate, experts)))
    costs = candidate_cost(candidate, columns, service)
    actual = float(costs['chip'][labels][routes].sum(1).max(1).mean())
    assert abs(actual - saved['mean_phase_chip_bound_ns']) < 1e-5
    return dict(candidate, costs=costs, labels=labels,
                rows=candidate['variants'][labels], mean_phase_chip_bound_ns=actual,
                fit=saved['fit'])


def shared_rows(hidden, intermediate, service):
    if not intermediate:
        return None
    result = []
    for r, c in ((2 * intermediate, hidden), (hidden, intermediate)):
        cap = Capability(chips=service.chips, planes=service.planes,
                         page_bytes=service.page_bytes, block_wordlines=service.block_wordlines)
        cert = certificate(r, c, cap)
        # Fixed compiler-derived minimum-depth/minimum-work geometry, identical
        # for every routed family. No replica, per-query choice or fitting.
        result.append(np.asarray(cert['depth_constrained_certificates'][0]['witness']['rows']))
    return result


def make_plan(g, d, shift, shape, shared, service):
    h, i, experts, _, _, si = shape
    gu = g['rows']; down = np.roll(d['rows'], shift, axis=1)
    if shared is not None:
        gu = np.r_[gu, shared[0][None]]; down = np.r_[down, shared[1][None]]
    return lower_rows(gu, down, h, [i] * experts + ([si] if si else []), service)


def selected_load(candidate, routes):
    labels = candidate['labels'][routes]; costs = candidate['costs']
    return dict(chip=costs['chip'][labels].sum(1), transfer=costs['transfer'][labels].sum(1),
                commands=costs['commands'][labels].sum(1), pcie=costs['pcie'][labels].sum(1))


def select_pairs(native, gu, down, routes, shape, shared, service, budget):
    """LB-ordered exact replay within a stated, fitted finite candidate pool."""
    _, _, experts, active, _, si = shape
    shared_work = np.zeros(16, dtype=np.int64); shared_transfer = shared_work.copy()
    shared_commands = 0; shared_pcie = 0
    if shared is not None:
        for rr, c in zip(shared, (shape[0], si)):
            sc = candidate_cost(dict(variants=np.tile(rr, (16, 1))), c, service)
            shared_work += sc['chip'][0]; shared_transfer += sc['transfer'][0]
            shared_commands += int(sc['commands'][0].sum()); shared_pcie += int(sc['pcie'][0])
    gloads = {g['label']: selected_load(g, routes) for g in gu}
    dloads = {d['label']: selected_load(d, routes) for d in down}
    host_lb = (1 + active + bool(si)) * service.host_submit_ns + (active + bool(si)) * service.activation_ns + service.combine_ns
    input_pcie = service.transfer(shape[0] * service.input_bytes, service.pcie_bytes_per_second)
    input_pcie += active * service.transfer(shape[1] * service.input_bytes, service.pcie_bytes_per_second)
    if si:
        input_pcie += service.transfer(si * service.input_bytes, service.pcie_bytes_per_second)
    pairs = []
    for gi, g in enumerate(gu):
        gl = gloads[g['label']]
        for di, d in enumerate(down):
            dl = dloads[d['label']]
            controller = service.host_submit_ns + (gl['commands'].sum(1) + dl['commands'].sum(1) + shared_commands) * service.command_ns
            pcie = service.host_submit_ns + gl['pcie'] + dl['pcie'] + shared_pcie + input_pcie
            for shift in range(service.chips):
                chip = service.host_submit_ns + (gl['chip'] + np.roll(dl['chip'], shift, axis=1) + shared_work).max(1)
                tr = gl['transfer'] + np.roll(dl['transfer'], shift, axis=1) + shared_transfer
                channel = service.host_submit_ns + tr.reshape(len(routes), service.chips // service.channels, service.channels).sum(1).max(1)
                lb = np.maximum.reduce((chip, controller, pcie, channel, np.full(len(routes), host_lb)))
                pairs.append((float(lb.mean()), gi, di, shift))
    pairs.sort(); evaluated = {}; reports = []; winners = {}
    for family, allowed in [('disjoint', {'disjoint'}), ('cyclic', {'disjoint', 'cyclic'}),
                             ('whole_rows', {'disjoint', 'cyclic', 'whole_rows'})]:
        eligible = [x for x in pairs if gu[x[1]]['family'] in allowed and down[x[2]]['family'] in allowed]
        best = np.inf; winner = None; replays = 0; pruned = 0; stopped = False
        # Strong controls are available as feasible initial incumbents.
        for prev in winners.values():
            key = tuple(prev['key'])
            if gu[key[0]]['family'] in allowed and down[key[1]]['family'] in allowed:
                value = evaluated[key]['mean_ns']
                if value < best:
                    best = value; winner = key
        for pi, (lb, gi, di, shift) in enumerate(eligible):
            key = gi, di, shift
            if lb >= best:
                pruned = len(eligible) - pi; break
            if key not in evaluated:
                if replays >= budget:
                    stopped = True; break
                plan = make_plan(gu[gi], down[di], shift, shape, shared, service)
                got = native.run(plan, routes, experts, int(bool(si)), service)
                mean = float(got['cost'].mean())
                assert mean + 1e-8 >= lb, (mean, lb)
                evaluated[key] = dict(mean_ns=mean, p99_ns=float(np.percentile(got['cost'], 99)), cost=got['cost'])
                replays += 1
            value = evaluated[key]['mean_ns']
            if value < best:
                best = value; winner = key
        assert winner is not None
        gi, di, shift = winner
        rec = dict(family=family, key=list(winner), gu_label=gu[gi]['label'], down_label=down[di]['label'],
                   down_rotation=shift, mean_ns=best, p99_ns=evaluated[winner]['p99_ns'],
                   finite_pool_pairs=len(eligible), additional_replays=replays,
                   lower_bound_pruned_pairs=pruned, replay_budget_exhausted=stopped,
                   finite_pool_certified_best=not stopped,
                   qualification='Only this fitted finite pool, these selection queries and this fixed greedy schedule; not a global placement/schedule optimum.')
        reports.append(rec); winners[family] = rec
    return reports, evaluated


def run_one(model, layer, command, args, native, labels):
    shape = SHAPES[model]; h, i, experts, _, _, si = shape
    service = Service(command_ns=command, output_fringe=1, input_buffer_bytes=32768,
                      input_channel_copies=args.input_copies)
    target = args.output / f'{model}.L{layer}.cmd{command}'; target.mkdir(parents=True, exist_ok=True)
    routes, provenance = training(model, layer, args.fit_queries, args.training_manifest)
    # Selection and label-fitting both use TRAIN. This screen reuses previously
    # examined data and cannot constitute a newly blind benchmark.
    indices = np.unique(np.linspace(0, len(routes)-1, min(args.select_queries, len(routes)), dtype=np.int64))
    selected = np.ascontiguousarray(routes[indices])
    phases, certs = [], []
    reused = None
    if args.reuse_fitting:
        reuse_path = args.reuse_fitting / f'{model}.L{layer}.cmd{command}/summary.json'
        reused = json.loads(reuse_path.read_text())
        assert reused['training']['sha256'] == provenance['sha256']
        assert reused['training']['selected_indices'] == provenance['selected_indices']
        assert reused['service'] == asdict(service)
    for p, (r, c) in enumerate(((2*i, h), (h, i))):
        candidates_, cert = candidates(r, c, service); fitted = []
        for candidate in deduplicate(candidates_, experts):
            if reused is not None:
                old = next(v for v in reused['fitting'][p] if v['label'] == candidate['label'])
                fitted.append(reuse_phase(candidate, c, routes, experts, service, old))
            else:
                fitted.append(fit_phase(labels, candidate, c, routes, experts, service,
                                         args.iterations, 20261103 + layer*10007 + p*1009 + candidate['seed_index']*31))
        phases.append(fitted); certs.append(cert)
    shared = shared_rows(h, si, service)
    reports, evaluated = select_pairs(native, *phases, selected, shape, shared, service, args.replay_budget)
    artifacts = []
    for report in reports:
        gi, di, shift = report['key']; plan = make_plan(phases[0][gi], phases[1][di], shift, shape, shared, service)
        addresses = physical_addresses(plan, service)
        label = report['family']; p = target / (label + '.npz')
        np.savez_compressed(p, **{k:plan[k] for k in ('offsets', 'jobs', 'input_columns', 'total_rows')},
                            selection_routes=selected, cost=evaluated[tuple(report['key'])]['cost'])
        doc = dict(model=model, layer=layer, shape=shape, service=asdict(service), selection=report,
                   gu_labels=phases[0][gi]['labels'].tolist(), down_labels=phases[1][di]['labels'].tolist(),
                   logical=plan['logical'], **addresses)
        write(target / (label + '.plan.json'), doc)
        artifacts.append(dict(family=label, arrays=str(p), arrays_sha256=sha(p),
            plan=str(target / (label + '.plan.json')), plan_sha256=sha(target / (label + '.plan.json')),
            useful_bytes=addresses['useful_bytes'], allocated_lsb_block_bytes=addresses['allocated_lsb_block_bytes'],
            chip_blocks=addresses['chip_blocks']))
    fit_report = []
    for p, phase in enumerate(phases):
        fit_report.append([dict(label=c['label'], family=c['family'], labels=c['labels'].tolist(),
                               aliases=c['aliases'], families=c['families'], geometry_sha256=c['geometry_sha256'],
                               base_rows=c['variants'][0].tolist(),
                               mean_phase_chip_bound_ns=c['mean_phase_chip_bound_ns'], fit=c['fit']) for c in phase])
    result = dict(model=model, layer=layer, command_ns=command, shape=shape,
                  training=provenance, service=asdict(service), fitting=fit_report,
                  selection_training_indices=indices.tolist(), selection=reports, artifacts=artifacts,
                  total_distinct_replays=len(evaluated), shared_rows=None if shared is None else [r.tolist() for r in shared],
                  fitting_reused_from=None if reused is None else dict(path=str(reuse_path), sha256=sha(reuse_path),
                      policy='Retain canonical first geometry fit exactly; remove duplicate geometries and their extra random-seed searches. Recomputed and checked every training phase objective and label histogram.'),
                  shared_policy='Same fixed minimum-depth/minimum-work onecopy geometry for every routed family; no shared fit.',
                  metadata_policy='Coefficient I8 bytes only in flash. Host row scales/address metadata require separate ledger; not inserted as MAC coefficients.')
    write(target / 'summary.json', result)
    print(model, layer, command, [(r['family'], round(r['mean_ns']/1e6, 6), r['gu_label'], r['down_label'], r['down_rotation'], r['additional_replays'], r['finite_pool_certified_best']) for r in reports], flush=True)
    return result


def initialize_worker(build, labels_path):
    global WORKER_NATIVE, WORKER_LABELS
    from research_20261102_qwen38_fit import load_library
    WORKER_NATIVE = NativeReplay(build)
    WORKER_LABELS = load_library(labels_path)


def worker(job):
    model, layer, command, args = job
    return run_one(model, layer, command, args, WORKER_NATIVE, WORKER_LABELS)


def assemble_models(records, output):
    """Pack all selected layers jointly; single-layer block IDs are local."""
    totals = []
    for model in sorted({r['model'] for r in records}):
        for command in sorted({r['command_ns'] for r in records if r['model'] == model}):
            layers = sorted([r for r in records if r['model'] == model and r['command_ns'] == command], key=lambda r:r['layer'])
            for family in ('disjoint', 'cyclic', 'whole_rows'):
                logical, offsets, jobs, inputs, rows, costs = [], [], [], [], [], []
                expert_base = job_base = 0
                for layer in layers:
                    artifact = next(a for a in layer['artifacts'] if a['family'] == family)
                    assert sha(artifact['arrays']) == artifact['arrays_sha256']
                    assert sha(artifact['plan']) == artifact['plan_sha256']
                    plan = json.loads(Path(artifact['plan']).read_text())
                    with np.load(artifact['arrays']) as a:
                        offsets.append(a['offsets'].astype(np.int64) + job_base)
                        jobs.append(a['jobs'].copy()); inputs.append(a['input_columns'].copy())
                        rows.append(a['total_rows'].copy()); costs.append(a['cost'].copy())
                        count = len(a['offsets']); job_count = len(a['jobs'])
                    logical.extend(dict(j, expert=j['expert'] + expert_base,
                                        moe_layer_index=layer['layer'], physical_model_layer=layer['training']['physical_model_layer']) for j in plan['logical'])
                    expert_base += count; job_base += job_count
                assert all(len(c) == len(costs[0]) for c in costs)
                service = Service(**layers[0]['service'])
                merged = dict(offsets=np.concatenate(offsets).astype(np.int32),
                              jobs=np.concatenate(jobs), input_columns=np.concatenate(inputs),
                              total_rows=np.concatenate(rows), logical=logical)
                addresses = physical_addresses(merged, service)
                target = output / 'models'; target.mkdir(exist_ok=True)
                label = f'{model}.cmd{command}.{family}'
                path = target / (label + '.physical.json.gz')
                document = dict(model=model, family=family, command_ns=command,
                                service=asdict(service), layers=[r['layer'] for r in layers],
                                complete_all_MoE_layers=len(layers) == SHAPES[model][4],
                                logical=logical, **addresses,
                                block_unit='One block-index bundle across the same-address planes; each occupies planes native blocks.',
                                capacity='No physical device budget supplied; this is a ledger, not measured capacity. Best-fit decreasing is a legal heuristic, not a minimum-block or infeasibility proof.')
                with gzip.GzipFile(filename=str(path), mode='wb', mtime=0) as f:
                    f.write(json.dumps(document, separators=(',', ':')).encode())
                values = np.stack(costs).sum(0)
                np.save(target / (label + '.selection-cost.npy'), values)
                totals.append(dict(model=model, command_ns=command, family=family,
                                   layers=len(layers), complete_all_MoE_layers=document['complete_all_MoE_layers'],
                                   training_selection_queries=len(values), mean_ns=float(values.mean()),
                                   p99_ns=float(np.percentile(values, 99)),
                                   physical_plan=str(path), physical_plan_sha256=sha(path),
                                   useful_bytes=addresses['useful_bytes'],
                                   programmed_page_bytes=addresses['programmed_page_bytes'],
                                   reserved_wave_bytes=addresses['reserved_wave_bytes'],
                                   allocated_lsb_block_bytes=addresses['allocated_lsb_block_bytes'],
                                   chip_blocks=addresses['chip_blocks'],
                                   finite_pool_mean_certified_layers=sum(next(s for s in r['selection'] if s['family'] == family)['finite_pool_certified_best'] for r in layers)))
    write(output / 'model_totals.json', dict(status='complete virtual multilayer addresses and aligned training-selection service sums', records=totals,
          limits='No full-model time, heldout test or physical programming. Attention/router/SSM/dense first layers, row-scale metadata and device overprovisioning excluded. Simultaneous-resident coefficient capacity is reported in same-address LSB block bundles.'))
    return totals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', choices=SHAPES, default=list(SHAPES))
    ap.add_argument('--layers', nargs='+', type=int, help='MoE-layer indices, not raw model layer IDs (DeepSeek starts with one dense layer). Default first MoE layer.')
    ap.add_argument('--all-layers', action='store_true')
    ap.add_argument('--command-ns', nargs='+', type=int, default=[250, 5000])
    ap.add_argument('--fit-queries', type=int, default=1024)
    ap.add_argument('--select-queries', type=int, default=64)
    ap.add_argument('--iterations', type=int, default=2000)
    ap.add_argument('--replay-budget', type=int, default=128)
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--reuse-fitting', type=Path, help='Reuse exact canonical phase fits from an older sealed campaign; geometry, service, training indices/objective and budgets must agree.')
    ap.add_argument('--training-manifest', type=Path, help='Frozen manifest of complete, same-format W8A8 TRAIN routing archives. Otherwise use the earlier development sources.')
    ap.add_argument('--input-copies', type=int, default=1, help='Full-vector ONFI loads per native job;1assumes internal replication,4charges eachplane independently. PCIe upload remains once.')
    ap.add_argument('--output', type=Path, default=DEST)
    args = ap.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    # Refuse to silently replace a previous sealed compile experiment.
    protocol = args.output / 'protocol.frozen.json'
    assert not protocol.exists(), 'Choose a new output directory to retain previous provenance.'
    native = NativeReplay(args.output / 'build')
    source = Path(__file__).with_name('research_20261102_native_labels.cpp')
    so = args.output / 'build/native_labels.so'
    subprocess.run(['c++', '-std=c++17', '-O3', '-shared', '-fPIC', str(source), '-o', str(so)], check=True)
    from research_20261102_qwen38_fit import load_library
    labels = load_library(so)
    paths = [Path(__file__), Path(__file__).with_name('research_20261103_ordinary_read_backend.py'),
             Path(__file__).with_name('research_20261103_ordinary_read_native.cpp'),
             Path(__file__).with_name('research_20261102_whole_row_compiler.py'),
             Path(__file__).with_name('research_20261102_qwen38_fit.py'), source]
    hashes = {str(p):sha(p) for p in paths}
    training_manifest = None
    if args.training_manifest:
        training_manifest = json.loads(args.training_manifest.read_text())
        assert all(m in training_manifest['models'] for m in args.models)
        hashes[str(args.training_manifest)] = sha(args.training_manifest)
        for m in args.models:
            record = training_manifest['models'][m]
            assert record['status'] == 'complete' and sha(record['archive']) == record['sha256']
            hashes[record['archive']] = record['sha256']
    reuse_manifest = None
    if args.reuse_fitting:
        old_protocol = args.reuse_fitting / 'protocol.frozen.json'
        old_summary = args.reuse_fitting / 'summary.json'
        d = json.loads(old_protocol.read_text())
        assert d['arguments']['iterations'] == args.iterations
        assert d['arguments']['fit_queries'] == args.fit_queries
        reuse_manifest = {str(p):sha(p) for p in (old_protocol, old_summary)}
    assert not (args.all_layers and args.layers) and min(args.workers, args.replay_budget, args.fit_queries, args.select_queries, args.iterations, args.input_copies) > 0
    jobs = [(m, l, c) for m in args.models
            for l in (range(SHAPES[m][4]) if args.all_layers else (args.layers or [0]))
            for c in args.command_ns]
    assert len(jobs) == len(set(jobs))
    assert all(0 <= l < SHAPES[m][4] for m, l, _ in jobs)
    write(protocol, dict(created_utc=datetime.now(timezone.utc).isoformat(),
          sources_sha256=hashes, arguments={**vars(args), 'output':str(args.output),
             'reuse_fitting':None if args.reuse_fitting is None else str(args.reuse_fitting),
             'training_manifest':None if args.training_manifest is None else str(args.training_manifest)}, build=native.provenance,
          reused_fitting_inputs_sha256=reuse_manifest,
          label_fitter=dict(binary=str(so), binary_sha256=sha(so), source_sha256=sha(source),
                            command='c++ -std=c++17 -O3 -shared -fPIC SOURCE -o BINARY'),
          controls='Same label-swap iterations and temperatures pergeometry; same complete-row native lowering, controller, channels, PCIe, output-fringe and single chip context. Disjoint, every cyclic width and whole-row work/depth alternatives. All16relative down rotations offered to each family.',
          selection='Training-only finite candidate pairs, valid resource/ownership lowerbounds ordered exact event replay with explicit optional budget. Earlier control is an incumbent for larger pools.',
          geometry_deduplication='Within unrestricted expert-pair label swaps, the initial assigned physical-rowvector histogram defines the exact reachable assignment set. Fit its canonical first candidate once and retain aliases; unused or duplicate ordinal variants add no extra seeded search.',
          input_buffer='32768B is the described AiF two16KiB input read latches perplane, inferred from paper Section4.4/Table2; not an exposed ISA maximum or cross-plane input-capacity aggregation. All seven A8 shapes fit.',
          input_fanout='input_copies counts explicit full-vector ONFI loads perjob, while controller/common PCIe upload staysonce. Actual AiF input-broadcast implementation is unverified; four copies remove the free-broadcast cost assumption without adding independent contexts.',
          limits='Development data reused. No external validation, priority/new-algorithm claim, calibration, native PE arithmetic or full-model speedup. A label fitted for a phase bound need not be a global best label for full FFN service.'))
    started = time.monotonic(); records = []
    if args.workers == 1:
        for job in jobs:
            records.append(run_one(*job, args, native, labels))
            write(args.output / 'partial.json', records)
    else:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize_worker,
                                 initargs=(args.output / 'build', so)) as pool:
            for finished in as_completed([pool.submit(worker, (*job, args)) for job in jobs]):
                records.append(finished.result())
                records.sort(key=lambda r:(r['model'], r['layer'], r['command_ns']))
                write(args.output / 'partial.json', records)
    totals = assemble_models(records, args.output)
    assert all(sha(p) == h for p, h in hashes.items())
    if reuse_manifest:
        assert all(sha(p) == h for p, h in reuse_manifest.items())
    write(args.output / 'summary.json', dict(status='complete train-only executable ordinary-read compiler development screen',
          protocol_sha256=sha(protocol), elapsed_seconds=time.monotonic()-started, records=records,
          model_totals=totals))
    print('complete', len(records), 'elapsed', time.monotonic()-started, flush=True)


if __name__ == '__main__':
    main()
