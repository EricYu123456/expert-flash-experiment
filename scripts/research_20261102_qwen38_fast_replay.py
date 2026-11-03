#!/usr/bin/env python3
"""Compiled equivalent replay, validated against sealed Python cost vectors.

This accelerates the same uncalibrated software service model; it does not
add hardware capability or supply new latency measurements.
"""
import argparse
import ctypes
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np

from research_20261102_qwen38_fit import TRAIN, TRAIN_SHA, sha
from research_20261102_qwen38_replay import BASE, inventories

ROOT = Path(__file__).resolve().parents[1]
DEST = BASE/'qwen38-placement'
LIB = DEST/'native_replay.so'


def load_library():
    lib = ctypes.CDLL(str(LIB)); p32 = np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
    p64 = np.ctypeslib.ndpointer(dtype=np.int64, flags='C_CONTIGUOUS')
    lib.eval_native_ffn.argtypes = [p32, p32, p32, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                   ctypes.c_int64, ctypes.c_int64, ctypes.c_int, p64]
    lib.eval_native_ffn.restype = None
    return lib


def encode(plan, inv):
    desc = np.zeros((513, 2, 16, 4), dtype=np.int32)
    partial = np.zeros((513, 2), dtype=np.int32)
    for e in range(513):
        for p, name in enumerate(('gate_up', 'down')):
            if e < 512:
                r = plan['plans'][name]; candidate = inv[name][r['label']]; color = r['labels'][e]
                partial[e, p] = candidate['conditional_partial_sum']
            else:
                r = plan['fixed_shared'][name]; candidate = inv[name][r['label']]; color = r['color']
            for s, j in enumerate(candidate['variants'][color]):
                rows, cols = j['rows'], j['columns']
                desc[e, p, s] = (j['chip'], rows, cols, (rows*cols+65535)//65536)
    return desc, partial


def evaluate(lib, desc, partial, selected, reduction_ns=0, activation_ns=0, down_first=False):
    selected = np.ascontiguousarray(selected, dtype=np.int32)
    out = np.zeros(len(selected), dtype=np.int64)
    lib.eval_native_ffn(desc, partial, selected, 512, len(selected), 10, reduction_ns, activation_ns, int(down_first), out)
    return out


def validate(directory):
    doc = json.loads((directory/'screen.json').read_text()); lib = load_library(); inv = inventories()
    with np.load(TRAIN) as f:
        routes = f['routes'].copy()
    assert sha(TRAIN) == TRAIN_SHA
    records = []; started = time.monotonic()
    for r in doc['records']:
        planpath = directory/(r['label']+'.plan.json'); npz = directory/(r['label']+'.npz')
        assert sha(planpath) == r['artifact_sha256']['plan.json'] and sha(npz) == r['artifact_sha256']['npz']
        plan = json.loads(planpath.read_text()); desc, partial = encode(plan, inv)
        with np.load(npz) as f:
            steps = f['token_steps'].copy(); reference = f['cost'].copy()
        selected = routes[:, steps, r['layer']].reshape(-1, 10)
        got = evaluate(lib, desc, partial, selected, r['reduction_ns'], r.get('activation_ns', 0), r.get('down_first', False))
        assert np.array_equal(got, reference), (r['label'], int(np.abs(got-reference).max()))
        records.append(dict(label=r['label'], queries=len(got), maximum_difference_ns=0))
    report = dict(status='PASS exact compiled/Python cost-vector equivalence',
                  created_utc=datetime.now(timezone.utc).isoformat(), screen_sha256=sha(directory/'screen.json'),
                  sources_sha256={str(p):sha(p) for p in (Path(__file__), Path(__file__).with_name('research_20261102_native_replay.cpp'), LIB)},
                  queries=sum(r['queries'] for r in records), elapsed_seconds=time.monotonic()-started,
                  records=records, scope='Software model equivalence only; not device, numerical ABI, hardware calibration or new placement evidence.')
    target = directory/'compiled_equivalence.json'; target.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'records'}, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--validate', type=Path, required=True)
    args = ap.parse_args(); validate(args.validate)


if __name__ == '__main__':
    main()
