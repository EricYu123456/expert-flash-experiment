#!/usr/bin/env python3
"""Screen arbitrary chip-subset exchanges against already frozen layouts.

This is an exploratory train-only search. Each accepted 2x2 move preserves
every expert/chip's stored shard-size inventory, so no replication or hidden
capacity is introduced. Public test routes are never used for optimization.
"""
import argparse
import ctypes
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from run_corpus_traces import ROOT, sha256
from research_20261102_projection_study import evaluate, scalar_cost

DEST = ROOT / 'results/research-2026-11-02/round2/free-rows'
PLAN = ROOT / 'results/research-2026-11-02/projection-study'


def library():
    DEST.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).with_suffix('.cpp')
    target = DEST / 'free_rows.so'
    subprocess.run(['g++', '-std=c++17', '-O3', '-march=native', '-shared',
                    '-fPIC', str(source), '-o', str(target)], check=True)
    lib = ctypes.CDLL(str(target))
    ptr32 = np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
    ptr64 = np.ctypeslib.ndpointer(dtype=np.int64, flags='C_CONTIGUOUS')
    lib.optimize_rows.argtypes = [ptr32, ptr32, ptr64] + [ctypes.c_int] * 6 + [
        ctypes.c_uint64, ctypes.c_double, ptr64]
    lib.optimize_rows.restype = ctypes.c_double
    return lib


def variable_cost(selected, rows, phase):
    values = evaluate(selected, rows, phase['columns'], phase['common'])
    return values['flash_ns'] + values['input_ns'] + values['output_ns']


def run(args):
    lib = library()
    result = []
    for name in args.models:
        plan = json.loads((PLAN / (name + '.frozen_plan.json')).read_text())
        with np.load(plan['train_path']) as f:
            routes = f['routes']
            assert f['prompt_ids'].tolist() == plan['train_ids']
        assert sha256(Path(plan['train_path'])) == plan['train_sha256']
        phases = []
        for index, phase in enumerate(plan['phases']):
            if args.layers and phase['layer'] not in args.layers:
                continue
            li = plan['layers'].index(phase['layer'])
            unique, weights = np.unique(np.sort(routes[:, :, li].reshape(-1, plan['active']),
                                             axis=1), axis=0, return_counts=True)
            unique = np.ascontiguousarray(unique, dtype=np.int32)
            weights = np.ascontiguousarray(weights, dtype=np.int64)
            chosen = np.ascontiguousarray(phase['chosen_rows'], dtype=np.int32)
            best = chosen.copy()
            best_mean = float(evaluate(unique, best, phase['columns'], phase['common'])[
                'model_ns'] @ weights / weights.sum())
            initial_mean = best_mean
            records = []
            # Two distinct starting geometries; both were frozen in round one.
            starts = [('chosen', chosen), ('row_only', np.asarray(phase['row_only_rows'], dtype=np.int32))]
            for si, (label, start) in enumerate(starts):
                if si and np.array_equal(start, chosen):
                    continue
                if np.all(start == start[:, :1]):
                    continue
                for temperature in args.temperatures:
                    begin = time.time()
                    rows = np.ascontiguousarray(start.copy())
                    stats = np.zeros(3, dtype=np.int64)
                    mean = lib.optimize_rows(rows, unique, weights, plan['experts'], len(unique),
                        plan['active'], phase['columns'], int(phase['common']), args.iterations,
                        20261102 + index * 1009 + si * 31 + int(temperature), temperature, stats)
                    # The C++ variable objective has an independent NumPy reference.
                    reference = float(variable_cost(unique, rows, phase) @ weights / weights.sum())
                    assert abs(reference - mean) < 1e-6, (reference, mean)
                    np.testing.assert_array_equal(np.sort(rows, axis=0), np.sort(start, axis=0))
                    np.testing.assert_array_equal(np.sort(rows, axis=1), np.sort(start, axis=1))
                    for t in sorted({0, len(unique)//2, len(unique)-1}):
                        slow = scalar_cost(unique[t], rows, phase['columns'], phase['common'])
                        fast = evaluate(unique[t:t+1], rows, phase['columns'], phase['common'])
                        for k, v in slow.items():
                            assert v == int(fast[k][0])
                    full_mean = float(evaluate(unique, rows, phase['columns'], phase['common'])[
                        'model_ns'] @ weights / weights.sum())
                    if full_mean < best_mean:
                        best_mean, best = full_mean, rows.copy()
                    records.append(dict(start=label, temperature_ns=temperature,
                        valid=int(stats[0]), accepted=int(stats[1]), improving=int(stats[2]),
                        mean_ns=full_mean, seconds=time.time()-begin))
            row = dict(layer=phase['layer'], phase=phase['name'], chosen_mean_ns=initial_mean,
                       new_mean_ns=best_mean, gain_fraction=1-best_mean/initial_mean,
                       rows=best.tolist(), search=records)
            phases.append(row)
            print(name, phase['layer'], phase['name'], round(100*row['gain_fraction'], 5),
                  'percent', flush=True)
            (DEST / (name + '.partial.json')).write_text(json.dumps(phases, indent=2)+'\n')
        output = dict(model=name, optimization_data='original 200 training prompts only',
            status='screening only, no optimality proof or hardware measurement',
            protocol=dict(iterations=args.iterations, temperatures=args.temperatures,
                preserves='per-expert and per-chip shard-row/page/wave inventory',
                ordinary_read_ns=28000, no_cr=True, no_replication=True),
            frozen_utc=datetime.now(timezone.utc).isoformat(),
            source_sha256={p.name: sha256(p) for p in [Path(__file__), Path(__file__).with_suffix('.cpp')]},
            phases=phases)
        (DEST / (name + '.frozen.json')).write_text(json.dumps(output, indent=2)+'\n')
        result.append(output)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=['granite', 'deepseek', 'qwen3'])
    parser.add_argument('--layers', type=int, nargs='*')
    parser.add_argument('--iterations', type=int, default=20000)
    parser.add_argument('--temperatures', type=float, nargs='+', default=[0., 1000.])
    run(parser.parse_args())
