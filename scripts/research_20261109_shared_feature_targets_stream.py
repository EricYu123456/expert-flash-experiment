#!/usr/bin/env python3
"""Streaming-I/O revision of offline native target generation.

Arithmetic comes from the already frozen 11/09 Numeric library. Sequential
write(2) avoids write faults on new NFS-backed mmap output pages. Completed
expert receipts from the initial generator are verified and reused.
"""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits, threadpool_info

from research_20261109_shared_feature_targets import (
    Numeric, DATA, capture_arrays, load_weights, prototype_selection, dump, sha)


def preserve_partial(path):
    if not path.exists():
        return
    n = 1
    while path.with_name(path.name + f'.incomplete-attempt{n}').exists():
        n += 1
    path.rename(path.with_name(path.name + f'.incomplete-attempt{n}'))


def compute(a):
    out, layer = a.output, a.layer
    out.mkdir(parents=True, exist_ok=True)
    manifests = sorted(a.captures.glob('shard*/capture_manifest.json'))
    assert len(manifests) == 4
    parts = [capture_arrays(p.parent, layer) for p in manifests]
    x = np.concatenate([p[2]['x'] for p in parts])
    hidden_experts, selection = prototype_selection(a.save_hidden_experts, a.prototype_train_captures, layer)
    hidden_experts = set(hidden_experts)
    numeric = Numeric(a.threads)
    input_identity = {str(p): sha(p) for p in manifests}
    for p, (_, info, _) in zip(manifests, parts):
        path = p.parent / info['arrays']['x']['path']
        input_identity[str(path)] = sha(path)
    experts = list(range(64)) if a.experts == 'all' else list(map(int, a.experts.split(',')))
    assert hidden_experts <= set(experts)
    fixed = [e for e in {1: [14, 40], 13: [10, 56], 26: [0, 1]}[layer] if e in experts]
    order = list(dict.fromkeys(fixed + sorted(hidden_experts) + experts))
    index = dict(status='complete', layer=layer, rows=len(x), hidden=2048,
                 expert_ids=experts, layout='expert-major independent F32[N,H] files',
                 input_sha256=input_identity, capture_manifest_sha256={str(p): sha(p) for p in manifests},
                 targets={}, prototype_hidden={}, numpy_version=np.__version__,
                 CPU_affinity=sorted(os.sched_getaffinity(0)), BLAS=threadpool_info(),
                 generator_revision='streaming output; frozen canonical arithmetic unchanged')
    dump(out / 'prototype_selection.json', selection)
    for expert in order:
        dest, receipt = out / f'expert{expert}.f32', out / f'expert{expert}.json'
        weights_identity = {str(a.weights / (f'expert{expert}.{role}' + suffix)): sha(a.weights / (f'expert{expert}.{role}' + suffix))
                            for role in ('gate', 'up', 'down') for suffix in ('.i8', '.scales.f32')}
        if receipt.exists():
            existing = json.loads(receipt.read_text())
            assert existing['input_sha256'] == input_identity and existing['target_sha256'] == sha(dest)
            assert existing['weights_payload_sha256'] == weights_identity
            assert existing['numeric_binary_sha256'] == numeric.binary_sha256
            if expert in hidden_experts:
                rec = existing['prototype_hidden']
                assert sha(rec['path']) == rec['sha256']
                index['prototype_hidden'][str(expert)] = rec
            index['targets'][str(expert)] = existing
            continue
        temp = dest.with_suffix('.f32.partial')
        hidden_dest = out / f'prototype{expert}.hidden.f32'
        hidden_temp = hidden_dest.with_suffix('.f32.partial')
        for p in (dest, temp, hidden_dest, hidden_temp):
            preserve_partial(p)
        w = load_weights(a.weights, expert)
        start = time.monotonic()
        proto_file = hidden_temp.open('wb', buffering=0) if expert in hidden_experts else None
        try:
            with temp.open('wb', buffering=0) as f, threadpool_limits(limits=a.threads, user_api='blas'):
                for offset in range(0, len(x), a.batch):
                    values = numeric.expert(np.ascontiguousarray(x[offset:offset + a.batch]), w,
                                            return_hidden=proto_file is not None)
                    if proto_file is not None:
                        y, phi = values
                        assert proto_file.write(memoryview(phi).cast('B')) == phi.nbytes
                    else:
                        y = values
                    assert f.write(memoryview(y).cast('B')) == y.nbytes
        finally:
            if proto_file is not None:
                proto_file.close()
        assert temp.stat().st_size == len(x) * 2048 * 4
        temp.replace(dest)
        checked = 0
        if expert in fixed:
            y = np.memmap(dest, dtype=np.float32, shape=(len(x), 2048), mode='r')
            cursor = 0
            for _, info, arrays in parts:
                native = np.ascontiguousarray(arrays['expert_outputs'][:, info['expert_ids'].index(expert)])
                assert np.array_equal(np.asarray(y[cursor:cursor + len(native)]).view(np.uint32), native.view(np.uint32))
                checked += native.size
                cursor += len(native)
            del y
        identity = dict(expert=expert, rows=len(x), shape=[len(x), 2048], dtype='float32',
                        path=str(dest), seconds=time.monotonic() - start,
                        target_sha256=sha(dest), input_sha256=input_identity,
                        weights_manifest_sha256=sha(a.weights / 'weights.json'),
                        weights_payload_sha256=weights_identity,
                        numeric_binary_sha256=numeric.binary_sha256,
                        all_native_capture_fixed_target_floats_bitchecked=checked,
                        generator_script_sha256=sha(__file__), output_IO='sequential unbuffered write, no output mmap')
        if expert in hidden_experts:
            assert hidden_temp.stat().st_size == len(x) * 1408 * 4
            hidden_temp.replace(hidden_dest)
            rec = dict(path=str(hidden_dest), shape=[len(x), 1408], dtype='float32',
                       sha256=sha(hidden_dest), meaning='Native routed expert pre-Down SwiGLU phi; before its own vector-A8. Not native shared branch.',
                       native_D_prefix=str(a.weights / f'expert{expert}.down'),
                       own_native_D_positive_control='Its own W8 D on its own raw phi, with native vector-A8 and zero bias, is canonical native output; this is a native/hot-GU control, not cross-expert reuse.')
            identity['prototype_hidden'] = rec
            index['prototype_hidden'][str(expert)] = rec
        dump(receipt, identity)
        index['targets'][str(expert)] = identity
        print(json.dumps(dict(layer=layer, expert=expert, rows=len(x), seconds=identity['seconds'],
                              native_bitchecks=checked)), flush=True)
    dump(out / 'targets.json', index)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--captures', type=Path, required=True)
    p.add_argument('--layer', type=int, required=True)
    p.add_argument('--experts', default='all')
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--batch', type=int, default=2048)
    p.add_argument('--save-hidden-experts', default='train-top8')
    p.add_argument('--prototype-train-captures', type=Path, required=True)
    a = p.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    compute(a)


if __name__ == '__main__':
    main()
