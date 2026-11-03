#!/usr/bin/env python3
"""Matched phase/joined fit, including simple global-complement controls.

Reuses the sealed label-swap kernel and physical inventories. New learning is
restricted to balanced permutations of those existing layouts; no new native
operation, weight format, replica or decoder is introduced.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from research_20261102_qwen38_fit import TRAIN, TRAIN_SHA, load_library, score, table, sha
from research_20261102_qwen38_replay import BASE, SOURCE, RECT, inventories, records_for

DEST = BASE/'qwen38-placement/joint-fit3'
ROOT = Path(__file__).resolve().parents[1]


def worker(task):
    layer, geometry, objective, iterations, output = task
    inv = inventories(); records = records_for(layer)
    names = dict(gate_up='balanced_whole_row', down='balanced_whole_row') if geometry == 'whole' else dict(gate_up='rect_min_io4', down='exact_work_rectangular')
    old_labels = []
    for p in ('gate_up', 'down'):
        label = 'balanced_whole_row' if geometry == 'whole' else 'exact_work_rectangular'
        old_labels.append(next(r['labels'] for r in records[p] if r['label'] == label))
    old_labels = np.asarray(old_labels, dtype=np.int32)
    with np.load(TRAIN) as f:
        selected = f['routes'][:, ::4, layer].reshape(-1, 10)
    selected, weights = np.unique(np.sort(selected, axis=1), axis=0, return_counts=True)
    selected = np.ascontiguousarray(selected, dtype=np.int32); weights = np.ascontiguousarray(weights, dtype=np.int64)
    cost = np.stack([table(inv[p][names[p]]) for p in ('gate_up', 'down')])
    # Exhaust all sixteen global relative shifts. A useful simple complement
    # must be a baseline rather than attributed to the learned optimizer.
    controls = []
    for shift in range(16):
        labels = old_labels.copy(); labels[1] = (labels[0]+shift)%16
        v = score(selected, weights, labels, cost, 1)
        controls.append(dict(shift=shift, mean_phase_bound_ns=v[1], mean_joined_bound_ns=v[2]))
    best_control = min(controls, key=lambda x:(x['mean_joined_bound_ns'], x['mean_phase_bound_ns']))
    complement = old_labels.copy(); complement[1] = (complement[0]+best_control['shift'])%16
    original_score = score(selected, weights, old_labels, cost, objective)
    best_labels = old_labels.copy(); best_score = original_score[0]; runs = []
    lib = load_library(SOURCE/'native_labels.so')
    for init_name, initial in [('original_phase', old_labels), ('global_complement', complement)]:
        initial_score = score(selected, weights, initial, cost, objective)
        if initial_score[0] < best_score:
            best_score, best_labels = initial_score[0], initial.copy()
        for si, temperature in enumerate((0.0, 2000.0)):
            labels = initial.copy(); stats = np.zeros(2, dtype=np.int64)
            value = lib.fit_native_labels(labels, cost, selected, weights, 512, len(selected), 10, 16,
                                          objective, iterations, 20261102+layer*1009+si*31+objective*97, temperature, stats)
            v = score(selected, weights, labels, cost, objective)
            assert abs(value-v[0]) < 1e-5
            assert np.all(np.bincount(labels[0], minlength=16) == 32)
            assert np.all(np.bincount(labels[1], minlength=16) == 32)
            runs.append(dict(init=init_name, temperature_ns=temperature, mean_phase_bound_ns=v[1],
                             mean_joined_bound_ns=v[2], valid=int(stats[0]), accepted=int(stats[1])))
            if value < best_score:
                best_score, best_labels = value, labels.copy()
    final = score(selected, weights, best_labels, cost, objective)
    result = dict(layer=layer, geometry=geometry, objective='phase' if objective == 0 else 'joined',
                  inventories=names, original_labels=old_labels.tolist(), complement_labels=complement.tolist(),
                  learned_labels=best_labels.tolist(), global_shift_controls=controls, best_global_control=best_control,
                  original_mean_phase_bound_ns=original_score[1], original_mean_joined_bound_ns=original_score[2],
                  learned_mean_phase_bound_ns=final[1], learned_mean_joined_bound_ns=final[2], search=runs,
                  fit_tokens=int(weights.sum()), unique_routes=len(selected))
    p = Path(output)/f'L{layer}.{geometry}.{result["objective"]}.json'
    p.write_text(json.dumps(result, indent=2)+'\n')
    print(layer, geometry, result['objective'], final[1]/1e6, final[2]/1e6,
          'complement', best_control['mean_joined_bound_ns']/1e6, flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--layers', type=int, default=3)
    ap.add_argument('--iterations', type=int, default=12000); ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--output', type=Path, default=DEST); args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True); assert sha(TRAIN) == TRAIN_SHA
    prior = json.loads((SOURCE/'fit.json').read_text())
    assert all(sha(p) == h for p, h in prior['source_sha256'].items())
    paths = [TRAIN, SOURCE/'fit.json', SOURCE/'native_labels.so', SOURCE/'templates.json', RECT]
    paths += [SOURCE/f'layer{l}.{p}.json' for l in range(args.layers) for p in ('gate_up', 'down')]
    hashes = {str(p):sha(p) for p in paths}
    src = [Path(__file__), Path(__file__).with_name('research_20261102_qwen38_fit.py'),
           Path(__file__).with_name('research_20261102_qwen38_replay.py'), Path(__file__).with_name('research_20261102_native_labels.cpp')]
    source_hashes = {str(p):sha(p) for p in src}; results = []
    tasks = [(l, g, obj, args.iterations, str(args.output)) for l in range(args.layers) for g in ('whole', 'rect') for obj in (0, 1)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for f in as_completed([pool.submit(worker, t) for t in tasks]):
            results.append(f.result())
    assert all(sha(p) == h for p, h in hashes.items()) and all(sha(p) == h for p, h in source_hashes.items())
    doc = dict(status='complete train-only same-budget phase/joined geometry fit',
               created_utc=datetime.now(timezone.utc).isoformat(), arguments={**vars(args), 'output':str(args.output)},
               inputs_sha256=hashes, source_sha256=source_hashes, records=results,
               protocol='All16 same-task requests, token steps0,4,...796. Both objectives share both original-phase and best global-shift initials, both temperatures,12000 swaps/run. Only balanced labels change, exact per-chip inventory preserved.',
               limits=['Necessary native-chip occupancy fits; not feasible full-DAG latency.',
                       'All16 global relative shifts are simple existing-method controls, not a new contribution.',
                       'Always-active shared expert, channels/PCIe and host reconstruction must be included in the separate replay.',
                       'Same nominalI8/NVFP4 provenance, numeric-ABI and train-only restrictions as parent study. No external task opened.'])
    path = args.output/'fit.json'; path.write_text(json.dumps(doc, indent=2)+'\n')
    print('frozen', sha(path), flush=True)


if __name__ == '__main__':
    main()
