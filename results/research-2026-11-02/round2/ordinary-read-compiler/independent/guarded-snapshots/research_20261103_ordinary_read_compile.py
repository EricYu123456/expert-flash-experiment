#!/usr/bin/env python3
"""Train-only ordinary-read geometry/label compiler with executable addresses.

Standard covering DP, balanced expert-label swaps, cyclic declustering and
branch-and-bound are controls/components, not new algorithms. This integrated
prototype tests whether native read/dispatch costs select useful physical
whole-row geometries. No CR-read, column partials or weight replication.
"""
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import ctypes
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
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def training(model, layer, fit_limit):
    if model == 'qwen38':
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
                    ibytes = columns * service.input_bytes
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


def fit_phase(native_labels, candidate, columns, routes, experts, service, iterations, seed):
    costs = candidate_cost(candidate, columns, service)
    unique, weights = np.unique(np.sort(routes, axis=1), axis=0, return_counts=True)
    unique = np.ascontiguousarray(unique, dtype=np.int32); weights = weights.astype(np.int64)
    if candidate['family'] == 'disjoint':
        width = int(candidate['label'].split('_')[-1])
        initial = np.arange(experts, dtype=np.int32) % (16 // width)
    else:
        initial = (np.arange(experts, dtype=np.int32) * max(1, 16 // experts)) % 16
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
    service = Service(command_ns=command, output_fringe=1, input_buffer_bytes=32768)
    target = args.output / f'{model}.L{layer}.cmd{command}'; target.mkdir(parents=True, exist_ok=True)
    routes, provenance = training(model, layer, args.fit_queries)
    # Selection and label-fitting both use TRAIN. This screen reuses previously
    # examined data and cannot constitute a newly blind benchmark.
    indices = np.unique(np.linspace(0, len(routes)-1, min(args.select_queries, len(routes)), dtype=np.int64))
    selected = np.ascontiguousarray(routes[indices])
    phases, certs = [], []
    for p, (r, c) in enumerate(((2*i, h), (h, i))):
        candidates_, cert = candidates(r, c, service); fitted = []
        for ci, candidate in enumerate(candidates_):
            fitted.append(fit_phase(labels, candidate, c, routes, experts, service,
                                     args.iterations, 20261103 + layer*10007 + p*1009 + ci*31))
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
                               base_rows=c['variants'][0].tolist(),
                               mean_phase_chip_bound_ns=c['mean_phase_chip_bound_ns'], fit=c['fit']) for c in phase])
    result = dict(model=model, layer=layer, command_ns=command, shape=shape,
                  training=provenance, service=asdict(service), fitting=fit_report,
                  selection_training_indices=indices.tolist(), selection=reports, artifacts=artifacts,
                  total_distinct_replays=len(evaluated), shared_rows=None if shared is None else [r.tolist() for r in shared],
                  shared_policy='Same fixed minimum-depth/minimum-work onecopy geometry for every routed family; no shared fit.',
                  metadata_policy='Coefficient I8 bytes only in flash. Host row scales/address metadata require separate ledger; not inserted as MAC coefficients.')
    write(target / 'summary.json', result)
    print(model, layer, command, [(r['family'], round(r['mean_ns']/1e6, 6), r['gu_label'], r['down_label'], r['down_rotation'], r['additional_replays'], r['finite_pool_certified_best']) for r in reports], flush=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', choices=SHAPES, default=list(SHAPES))
    ap.add_argument('--layers', nargs='+', type=int, default=[0], help='MoE-layer indices, not raw model layer IDs (DeepSeek starts with one dense layer).')
    ap.add_argument('--command-ns', nargs='+', type=int, default=[250, 5000])
    ap.add_argument('--fit-queries', type=int, default=1024)
    ap.add_argument('--select-queries', type=int, default=64)
    ap.add_argument('--iterations', type=int, default=2000)
    ap.add_argument('--replay-budget', type=int, default=128)
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
             Path(__file__).with_name('research_20261102_whole_row_compiler.py'), source]
    hashes = {str(p):sha(p) for p in paths}
    jobs = [(m, l, c) for m in args.models for l in args.layers for c in args.command_ns]
    assert all(0 <= l < SHAPES[m][4] for m, l, _ in jobs)
    write(protocol, dict(created_utc=datetime.now(timezone.utc).isoformat(),
          sources_sha256=hashes, arguments={**vars(args), 'output':str(args.output)}, build=native.provenance,
          controls='Same label-swap iterations and temperatures pergeometry; same complete-row native lowering, controller, channels, PCIe, output-fringe and single chip context. Disjoint, every cyclic width and whole-row work/depth alternatives. All16relative down rotations offered to each family.',
          selection='Training-only finite candidate pairs, valid resource/ownership lowerbounds ordered exact event replay with explicit optional budget. Earlier control is an incumbent for larger pools.',
          input_buffer='32768B is the described AiF two16KiB input read latches perplane, inferred from paper Section4.4/Table2; not an exposed ISA maximum or cross-plane input-capacity aggregation. All seven A8 shapes fit.',
          limits='Development data reused. No external validation, priority/new-algorithm claim, calibration, native PE arithmetic or full-model speedup. A label fitted for a phase bound need not be a global best label for full FFN service.'))
    started = time.monotonic(); records = []
    for job in jobs:
        records.append(run_one(*job, args, native, labels))
        write(args.output / 'partial.json', records)
    assert all(sha(p) == h for p, h in hashes.items())
    write(args.output / 'summary.json', dict(status='complete train-only executable ordinary-read compiler development screen',
          protocol_sha256=sha(protocol), elapsed_seconds=time.monotonic()-started, records=records))
    print('complete', len(records), 'elapsed', time.monotonic()-started, flush=True)


if __name__ == '__main__':
    main()
