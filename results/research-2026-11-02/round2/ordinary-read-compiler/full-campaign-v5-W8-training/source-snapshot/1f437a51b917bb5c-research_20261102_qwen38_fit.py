#!/usr/bin/env python3
"""Exploratory train-only physical geometries for a new real route corpus.

One common cost and matched label-swap budget apply to disjoint groups,
all-width cyclic overlapping stripes, complete-row wave templates and a
conditional exact-work rectangular control. No evaluation task is opened.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import ctypes
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'results/research-2026-11-02/round2'
DEST = BASE/'qwen38-placement'
TRAIN = BASE/'qwen38-trace-audit/train_decode.npz'
TRAIN_SHA = '953e8db87d4bc4a5b5b9a77f5aebae1578b1e522df5bf3ac6a32a591f9e8b145'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rectangles(phase):
    if phase == 'gate_up':
        # Low-output exact50-wave tiling, independently constructed by the
        # architecture auditor. Unit grid128 rows x512 columns.
        out = [(r*128, 128, 1024, 1536) for r in range(10)]
        out += [(r*128, 384, c*512, 512) for r in (0, 3) for c in (0, 1)]
        out += [(r*128, 256, 0, 1024) for r in (6, 8)]
    else:
        # Unit grid128 rows x128 columns;9 two-wave and7 one-wave jobs.
        out = [(r*128, 256, 0, 512) for r in range(0, 18, 2)]
        out += [(r*128, 128, 0, 512) for r in (18, 19)]
        out += [(r*128, 512, 512, 128) for r in range(0, 20, 4)]
    assert len(out) == 16
    return out


def templates(phase):
    geo = json.loads((DEST/'geometry_certificates.json').read_text())
    p = next(x for x in geo['phases'] if x['name'] == phase)
    rows, cols = p['rows'], p['columns']
    result = []
    for g in (1, 2, 4, 8, 16):
        variants = []
        for label in range(16):
            n = np.full(g, rows//g); n[:rows%g] += 1
            start = (label % (16//g))*g; cursor = 0; jobs = []
            for c, r in enumerate(n):
                jobs.append(dict(chip=start+c, row_start=cursor, rows=int(r), column_start=0, columns=cols)); cursor += int(r)
            variants.append(jobs)
        result.append(dict(label=f'disjoint_g{g}', family='disjoint', conditional_partial_sum=False, variants=variants))
    for g in range(1, 16):
        variants = []
        for label in range(16):
            n = np.full(g, rows//g); n[:rows%g] += 1
            cursor = 0; jobs = []
            for j, r in enumerate(n):
                jobs.append(dict(chip=(j+label)%16, row_start=cursor, rows=int(r), column_start=0, columns=cols)); cursor += int(r)
            variants.append(jobs)
        result.append(dict(label=f'cyclic_width{g}', family='overlap', conditional_partial_sum=False, variants=variants))
    rr = p['balanced_whole_wave']['rows']; variants = []
    for label in range(16):
        jobs = []; cursor = 0
        for j, r in enumerate(rr):
            if r:
                jobs.append(dict(chip=(j+label)%16, row_start=cursor, rows=r, column_start=0, columns=cols))
            cursor += r
        variants.append(jobs)
    result.append(dict(label='balanced_whole_row', family='geometry', conditional_partial_sum=False, variants=variants))
    rect = rectangles(phase); variants = []
    for label in range(16):
        variants.append([dict(chip=(j+label)%16, row_start=r0, rows=r, column_start=c0, columns=c)
                         for j, (r0, r, c0, c) in enumerate(rect)])
    result.append(dict(label='exact_work_rectangular', family='rectangular', conditional_partial_sum=True, variants=variants))
    # Independently check every candidate's coefficient coverage, rectangular
    # bounds and one job per chip; all are static locations, no runtime moves.
    for candidate in result:
        a = np.zeros((rows, cols), dtype=np.uint8)
        for job in candidate['variants'][0]:
            r, n, c, m = (job[k] for k in ('row_start', 'rows', 'column_start', 'columns'))
            assert 0 <= r < r+n <= rows and 0 <= c < c+m <= cols
            a[r:r+n, c:c+m] += 1
        assert np.all(a == 1)
        for jobs in candidate['variants']:
            assert len({j['chip'] for j in jobs}) == len(jobs)
    return result


def table(candidate):
    result = np.zeros((16, 16), dtype=np.int64)
    for label, jobs in enumerate(candidate['variants']):
        for j in jobs:
            r, c = j['rows'], j['columns']
            result[label, j['chip']] += ((r*c+65535)//65536)*28000+250+(c+1)//2+2*r
    return result


def load_library(path):
    lib = ctypes.CDLL(str(path))
    ptr32 = np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
    ptr64 = np.ctypeslib.ndpointer(dtype=np.int64, flags='C_CONTIGUOUS')
    lib.fit_native_labels.argtypes = [ptr32, ptr64, ptr32, ptr64]+[ctypes.c_int]*6+[ctypes.c_uint64, ctypes.c_double, ptr64]
    lib.fit_native_labels.restype = ctypes.c_double
    return lib


def score(selected, weights, labels, costs, objective):
    loads = [costs[p][labels[p]][selected].sum(axis=1) for p in range(2)]
    phase = loads[0].max(axis=1)+loads[1].max(axis=1)
    joined = (loads[0]+loads[1]).max(axis=1)
    objective_value = phase*1000+joined if objective == 0 else joined*1000+phase
    return float(objective_value@weights/weights.sum()), float(phase@weights/weights.sum()), float(joined@weights/weights.sum())


def worker(task):
    layer, phase, stride, iterations, output = task
    target = Path(output); lib = load_library(target/'native_labels.so')
    with np.load(TRAIN) as f:
        selected = f['routes'][:, ::stride, layer].reshape(-1, 10)
    selected, weights = np.unique(np.sort(selected, axis=1), axis=0, return_counts=True)
    selected = np.ascontiguousarray(selected, dtype=np.int32); weights = np.ascontiguousarray(weights, dtype=np.int64)
    records = []
    for ci, candidate in enumerate(templates(phase)):
        cost = np.zeros((2, 16, 16), dtype=np.int64); cost[0] = table(candidate)
        initial = np.tile(np.arange(512, dtype=np.int32)%16, (2, 1))
        before = score(selected, weights, initial, cost, 0)
        best, best_labels = before[0], initial.copy(); runs = []
        for si, temperature in enumerate((0.0, 2000.0)):
            fitted = initial.copy(); stats = np.zeros(2, dtype=np.int64)
            value = lib.fit_native_labels(fitted, cost, selected, weights, 512, len(selected), 10, 16, 0,
                                          iterations, 20261102+layer*1009+ci*31+si, temperature, stats)
            reference = score(selected, weights, fitted, cost, 0)
            assert abs(reference[0]-value) < 1e-5
            assert np.array_equal(np.sort(fitted, axis=1), np.sort(initial, axis=1))
            runs.append(dict(temperature_ns=temperature, valid=int(stats[0]), accepted=int(stats[1]), native_phase_bound_ns=reference[1]))
            if value < best:
                best, best_labels = value, fitted.copy()
        final = score(selected, weights, best_labels, cost, 0)
        records.append(dict(label=candidate['label'], family=candidate['family'], conditional_partial_sum=candidate['conditional_partial_sum'],
                            labels=best_labels[0].tolist(), initial_mean_bound_ns=before[1], mean_bound_ns=final[1], search=runs))
    result = dict(layer=layer, phase=phase, fit_stride=stride, fit_tokens=int(weights.sum()), unique_routes=len(selected), records=records)
    (target/f'layer{layer}.{phase}.json').write_text(json.dumps(result, indent=2)+'\n')
    whole = min((r for r in records if not r['conditional_partial_sum']), key=lambda r:r['mean_bound_ns'])
    disjoint = min((r for r in records if r['family'] == 'disjoint'), key=lambda r:r['mean_bound_ns'])
    print(layer, phase, 'group', disjoint['label'], disjoint['mean_bound_ns'], 'whole', whole['label'], whole['mean_bound_ns'],
          'rect', next(r['mean_bound_ns'] for r in records if r['family'] == 'rectangular'), flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--layers', type=int, default=3)
    ap.add_argument('--stride', type=int, default=4); ap.add_argument('--iterations', type=int, default=8000)
    ap.add_argument('--workers', type=int, default=8); ap.add_argument('--output', type=Path, default=DEST/'phase-fit3')
    args = ap.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    assert sha(TRAIN) == TRAIN_SHA
    cpp = Path(__file__).with_name('research_20261102_native_labels.cpp'); so = args.output/'native_labels.so'
    subprocess.run(['g++', '-std=c++17', '-O3', '-march=native', '-shared', '-fPIC', str(cpp), '-o', str(so)], check=True)
    inputs = [TRAIN, DEST/'geometry_certificates.json', DEST/'official_config.json']
    hashes = {str(p):sha(p) for p in inputs}
    source_hashes = {str(p):sha(p) for p in [Path(__file__), cpp]}
    t = {phase:templates(phase) for phase in ('gate_up', 'down')}
    (args.output/'templates.json').write_text(json.dumps(t, indent=2)+'\n')
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        tasks = [(l, p, args.stride, args.iterations, str(args.output)) for l in range(args.layers) for p in t]
        for f in as_completed([pool.submit(worker, x) for x in tasks]):
            results.append(f.result())
    assert all(sha(p) == h for p, h in hashes.items()) and all(sha(p) == h for p, h in source_hashes.items())
    doc = dict(status='complete train-only phase geometry fit, no latency claim', created_utc=datetime.now(timezone.utc).isoformat(),
               arguments={**vars(args), 'output':str(args.output)}, training_routes_sha256=TRAIN_SHA,
               inputs_sha256=hashes, source_sha256=source_hashes, templates_sha256=sha(args.output/'templates.json'),
               results=results, objective='Minimize mean max per-chip nonpreemptive command+input+ordinaryNAND+output occupancy. Necessary phase bound, not feasible full dependency latency.',
               protocol='All16 same-task requests, token steps0,4,...798. Both temperatures and identical label swap budgets for every geometry. Balanced label histograms preserve complete per-chip physical inventory. No external task payload opened.',
               limits=['Exploratory fit on NVFP4 checkpoint routes, with nominal I8/A8/4B-output AiF cost. Not deployment inference, precision conversion or quality evidence.',
                       'Rectangular family requires host partial sums and known native output/scale semantics, not established for AiF. Its necessary bound excludes reduction and shared expert.',
                       'Native full-job dependencies, channels/PCIe contention and reduction must be replayed separately. No publication novelty or global optimum claim.'])
    (args.output/'fit.json').write_text(json.dumps(doc, indent=2)+'\n')
    print('frozen', sha(args.output/'fit.json'), flush=True)


if __name__ == '__main__':
    main()
