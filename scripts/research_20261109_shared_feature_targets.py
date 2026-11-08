#!/usr/bin/env python3
"""Export and compute all native expert targets from CPU-captured x, offline.

FP64 BLAS operates on exactly representable A8/W8 integers. Every product and
every possible partial sum is an integer bounded by C*127**2 (C<=10944), well
inside FP64's exact range. Final scaling and SwiGLU run through frozen C++.
The output is canonical W8A8, not original GGUF or BF16 arithmetic.
"""
from __future__ import annotations
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from threadpoolctl import threadpool_limits, threadpool_info

ROOT = Path('/home/ericyu/expert_flash_experiment')
OLD = Path('/home/nfs_client/ericyu/expert_flash_experiment/research-2026-11-08-shared-features')
DATA = Path('/home/nfs_client/ericyu/expert_flash_experiment/research-2026-11-09-shared-features')
MODEL = Path('/home/nfs_client/ericyu/expert_flash_experiment/models/deepseek-v2-lite-chat-q8_0.gguf')


def dump(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.partial')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def build():
    folder = DATA / 'build'
    folder.mkdir(parents=True, exist_ok=True)
    original = json.loads((OLD / 'build/capture_build.json').read_text())['argv']
    capture_src = str(ROOT / 'scripts/research_20261108_shared_feature_capture.cpp')
    records = {}
    for name, src, shared in [('target-export', 'targets', False), ('shared-feature-numeric.so', 'numeric', True)]:
        cmd = list(original)
        source = ROOT / f'scripts/research_20261109_shared_feature_{src}.cpp'
        cmd[cmd.index(capture_src)] = str(source)
        cmd[cmd.index('-o') + 1] = str(folder / name)
        if shared:
            # Existing static GGML libraries are not PIC. Compile the unchanged
            # pinned vector SwiGLU body with its original SIMD header instead.
            llama = Path('/home/nfs_client/ericyu/expert_flash_experiment/sources/llama.cpp-927dada6c')
            vector = llama / 'ggml/src/ggml-cpu/vec.cpp'
            text = vector.read_text()
            body = text[text.index('void ggml_vec_swiglu_f32('):text.index('ggml_float ggml_vec_cvar_f32(')]
            pinned = folder / 'pinned_swiglu.cpp'
            pinned.write_text('#include "vec.h"\n' + body)
            cmd = ['c++', '-shared', '-fPIC', '-O3', '-march=native', '-fno-fast-math',
                   '-ffp-contract=off', '-std=c++17', '-fopenmp',
                   '-I' + str(llama / 'ggml/include'), '-I' + str(llama / 'ggml/src'),
                   '-I' + str(llama / 'ggml/src/ggml-cpu'), str(source), str(pinned),
                   '-o', str(folder / name), '-lm']
            records['pinned_SwiGLU'] = dict(upstream=str(vector), upstream_sha256=sha(vector),
                                           body_sha256=hashlib.sha256(body.encode()).hexdigest(),
                                           extracted_body_byte_exact=True)
        destination = folder / name
        cmd[cmd.index('-o') + 1] = str(destination) + '.building'
        run = subprocess.run(cmd, capture_output=True, text=True)
        if run.returncode:
            raise RuntimeError(run.stderr)
        Path(str(destination) + '.building').replace(destination)
        records[name] = dict(argv=cmd, destination=str(destination), source_sha256=sha(source), binary_sha256=sha(destination))
    records['frozen_11_08_capture_sha256'] = sha(ROOT / 'scripts/research_20261108_shared_feature_capture.cpp')
    records['transform_header_sha256'] = sha(ROOT / 'scripts/research_20261109_shared_feature_transform.hpp')
    dump(folder / 'target_build.json', records)
    print(json.dumps(records, indent=2))


def build_quality():
    folder = DATA / 'build'
    folder.mkdir(parents=True, exist_ok=True)
    cmd = list(json.loads((OLD / 'build/capture_build.json').read_text())['argv'])
    source = ROOT / 'scripts/research_20261109_shared_feature_quality.cpp'
    cmd[cmd.index(str(ROOT / 'scripts/research_20261108_shared_feature_capture.cpp'))] = str(source)
    dest = folder / 'shared-feature-quality'
    cmd[cmd.index('-o') + 1] = str(dest) + '.building'
    run = subprocess.run(cmd, capture_output=True, text=True)
    if run.returncode:
        raise RuntimeError(run.stderr)
    Path(str(dest) + '.building').replace(dest)
    record = dict(argv=cmd, destination=str(dest), source_sha256=sha(source), binary_sha256=sha(dest),
                  transform_header_sha256=sha(ROOT / 'scripts/research_20261109_shared_feature_transform.hpp'),
                  frozen_capture_sha256=sha(ROOT / 'scripts/research_20261108_shared_feature_capture.cpp'))
    dump(folder / 'quality_build.json', record)
    print(json.dumps(record, indent=2))


class Numeric:
    def __init__(self, threads=8):
        library = DATA / 'build/shared-feature-numeric.so'
        self.binary_sha256 = sha(library)
        self.lib = ctypes.CDLL(str(library))
        f32 = np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS')
        f64 = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
        i8 = np.ctypeslib.ndpointer(dtype=np.int8, flags='C_CONTIGUOUS')
        i32 = np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
        u8 = np.ctypeslib.ndpointer(dtype=np.uint8, flags='C_CONTIGUOUS')
        self.lib.sf_set_threads.argtypes = [ctypes.c_int]
        self.lib.sf_set_threads(threads)
        self.lib.sf_quantize.argtypes = [ctypes.c_int, ctypes.c_int, f32, i8, f32]
        self.lib.sf_rescale_exact_dots.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, f64, f32, f32, f32]
        self.lib.sf_swiglu.argtypes = [ctypes.c_int, ctypes.c_int, f32, f32, f32]
        self.lib.sf_transform.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, f32, f32, u8, i32, f32, f32, f32]
        self.lib.sf_predict.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, f32, i8, f32, f32, f32]

    def quantize(self, x):
        x = np.ascontiguousarray(x, dtype=np.float32)
        q, s = np.empty(x.shape, np.int8), np.empty(len(x), np.float32)
        assert self.lib.sf_quantize(*x.shape, x, q, s) == 0
        return q, s

    def matvecs(self, x, w, sw):
        q, sx = self.quantize(x)
        assert q.shape[1] == w.shape[1] and q.shape[1] <= 10944
        d = np.ascontiguousarray(q.astype(np.float64) @ w.astype(np.float64).T)
        # Independent I64 dot probes include boundary and middle coordinates.
        ts = sorted({0, len(q) // 2, len(q) - 1})
        rs = sorted({0, len(w) // 2, len(w) - 1})
        exact = q[ts].astype(np.int64) @ w[rs].astype(np.int64).T
        assert np.array_equal(d[np.ix_(ts, rs)], exact)
        out = np.empty(d.shape, np.float32)
        assert self.lib.sf_rescale_exact_dots(len(x), len(w), w.shape[1], d, sw, sx, out) == 0
        return out

    def expert(self, x, weights, return_hidden=False):
        gate = self.matvecs(x, *weights['gate'])
        up = self.matvecs(x, *weights['up'])
        z = np.empty(gate.shape, np.float32)
        assert self.lib.sf_swiglu(*z.shape, gate, up, z) == 0
        y = self.matvecs(z, *weights['down'])
        return (y, z) if return_hidden else y


def export(layer, experts, out, threads):
    cmd = [str(DATA / 'build/target-export'), str(MODEL), str(layer),
           ','.join(map(str, experts)), str(threads), str(out)]
    if (out / 'weights.json').exists():
        m = json.loads((out / 'weights.json').read_text())
        assert m['layer'] == layer and set(map(int, m['experts'])) == set(experts)
        return
    subprocess.run(cmd, check=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
    dump(out / 'export_command.json', dict(argv=cmd))


def load_weights(folder, expert):
    m = json.loads((folder / 'weights.json').read_text())['experts'][str(expert)]
    result = {}
    for role, r in m.items():
        w = np.fromfile(folder / (r['prefix'] + '.i8'), np.int8).reshape(r['rows'], r['columns'])
        s = np.fromfile(folder / (r['prefix'] + '.scales.f32'), np.float32)
        assert s.shape == (len(w),)
        result[role] = w, s
    return result


def capture_arrays(folder, layer):
    manifest = json.loads((folder / 'capture_manifest.json').read_text())
    assert manifest['status'] == 'complete' and manifest['mode'] == 'w8'
    info = manifest['layers'][str(layer)]
    arrays = {}
    for k in ('x', 'expert_outputs'):
        rec = info['arrays'][k]
        arrays[k] = np.memmap(folder / rec['path'], dtype=rec['dtype'], shape=tuple(rec['shape']), mode='r')
    return manifest, info, arrays


def benchmark(threads):
    numeric = Numeric(threads)
    report = dict(status='passed', CPU_only=True, threads=threads, exact_integer_backend='FP64 CPU BLAS; C++ F32 rescale; pinned llama.cpp SwiGLU', cases=[])
    for layer, experts in [(1, [14, 40]), (13, [10, 56]), (26, [0, 1])]:
        folder = DATA / f'benchmark/weights/layer{layer}'
        start = time.monotonic()
        export(layer, experts, folder, threads)
        export_seconds = time.monotonic() - start
        _, meta, arrays = capture_arrays(OLD / 'captures/train/shard0', layer)
        for expert in experts:
            weights = load_weights(folder, expert)
            target = arrays['expert_outputs'][:, meta['expert_ids'].index(expert)]
            for N in (256, len(target)):
                start = time.monotonic()
                with threadpool_limits(limits=threads, user_api='blas'):
                    out = numeric.expert(np.ascontiguousarray(arrays['x'][:N]), weights)
                seconds = time.monotonic() - start
                assert np.array_equal(out.view(np.uint32), np.ascontiguousarray(target[:N]).view(np.uint32)), (layer, expert, N)
                row = dict(layer=layer, expert=expert, rows=N, seconds=seconds,
                           row_floats_bitchecked=int(out.size), all_output_bits_equal_old_native_W8=True,
                           two_expert_weight_export_seconds=export_seconds)
                report['cases'].append(row)
                print(json.dumps(row), flush=True)
    dump(DATA / 'benchmark/native_target_benchmark.json', report)


def prototype_selection(spec, train_captures, layer):
    if not spec:
        return [], None
    if spec != 'train-top8':
        ids = list(map(int, spec.split(',')))
        assert len(set(ids)) == len(ids) and all(0 <= e < 64 for e in ids)
        return ids, dict(rule='Explicit predeclared prototype expert IDs', expert_ids=ids)
    assert train_captures and train_captures.name == 'train', 'train-top8 requires a TRAIN capture directory'
    counts = np.zeros(64, np.int64)
    identities = {}
    manifests = sorted(train_captures.glob('shard*/capture_manifest.json'))
    assert len(manifests) == 4
    for p in manifests:
        m = json.loads(p.read_text())
        assert m['status'] == 'complete' and m['mode'] == 'w8'
        rec = m['layers'][str(layer)]['arrays']['selected_experts']
        path = p.parent / rec['path']
        ids = np.memmap(path, dtype=rec['dtype'], shape=tuple(rec['shape']), mode='r')
        counts += np.bincount(np.asarray(ids).ravel(), minlength=64)
        identities[str(p)] = sha(p)
        identities[str(path)] = sha(path)
    chosen = sorted(range(64), key=lambda e: (-counts[e], e))[:8]
    return chosen, dict(rule='TRAIN native router frequency descending, expert ID ascending tie-break; no VALID/TEST values used',
                        expert_ids=chosen, TRAIN_counts=counts.tolist(), input_sha256=identities,
                        extra_host_GU_bytes_per_prototype=2 * 2048 * 1408 + 2 * 1408 * 4)


def compute(captures, layer, experts, weights, out, threads, batch, hidden_experts=None, prototype_meta=None):
    """Write expert-major targets; each expert is an independent contiguous file."""
    out.mkdir(parents=True, exist_ok=True)
    children = sorted(captures.glob('shard*/capture_manifest.json')) if captures.is_dir() else [captures]
    assert children
    parts = [capture_arrays(p.parent, layer) for p in children]
    x = np.concatenate([p[2]['x'] for p in parts])
    numeric = Numeric(threads)
    input_identity = {str(p): sha(p) for p in children}
    for p, (_, info, _) in zip(children, parts):
        path = p.parent / info['arrays']['x']['path']
        input_identity[str(path)] = sha(path)
    index = dict(status='complete', layer=layer, rows=len(x), hidden=2048,
                 expert_ids=experts, layout='expert-major independent F32[N,H] files',
                 input_sha256=input_identity, capture_manifest_sha256={str(p): sha(p) for p in children},
                 targets={}, prototype_hidden={}, numpy_version=np.__version__,
                 CPU_affinity=sorted(os.sched_getaffinity(0)), BLAS=threadpool_info())
    hidden_experts = set(hidden_experts or [])
    assert hidden_experts <= set(experts)
    if prototype_meta:
        dump(out / 'prototype_selection.json', prototype_meta)
    for expert in experts:
        dest = out / f'expert{expert}.f32'
        receipt = out / f'expert{expert}.json'
        weights_identity = {str(weights / (f'expert{expert}.{role}' + suffix)): sha(weights / (f'expert{expert}.{role}' + suffix))
                            for role in ('gate', 'up', 'down') for suffix in ('.i8', '.scales.f32')}
        if receipt.exists():
            existing = json.loads(receipt.read_text())
            assert existing['input_sha256'] == input_identity and existing['target_sha256'] == sha(dest)
            assert existing['weights_payload_sha256'] == weights_identity
            if expert in hidden_experts:
                assert existing.get('prototype_hidden'), 'Cannot add hidden after a completed target without an explicit new output'
                assert sha(existing['prototype_hidden']['path']) == existing['prototype_hidden']['sha256']
                index['prototype_hidden'][str(expert)] = existing['prototype_hidden']
            index['targets'][str(expert)] = existing
            continue
        assert not dest.exists(), 'Do not overwrite unreceipted target'
        w = load_weights(weights, expert)
        temp = dest.with_suffix('.f32.partial')
        y = np.memmap(temp, dtype=np.float32, shape=(len(x), 2048), mode='w+')
        hidden_dest = out / f'prototype{expert}.hidden.f32'
        hidden_temp = hidden_dest.with_suffix('.f32.partial')
        hidden = np.memmap(hidden_temp, dtype=np.float32, shape=(len(x), 1408), mode='w+') if expert in hidden_experts else None
        checked = 0
        start = time.monotonic()
        with threadpool_limits(limits=threads, user_api='blas'):
            for a in range(0, len(x), batch):
                generated = numeric.expert(np.ascontiguousarray(x[a:a + batch]), w, return_hidden=hidden is not None)
                if hidden is not None:
                    yy, zz = generated
                    hidden[a:a + batch] = zz
                else:
                    yy = generated
                y[a:a + batch] = yy
        y.flush()
        del y
        temp.replace(dest)
        # Contemporary fixed-expert graph targets independently validate this
        # offline path over every row, including the longer 64-token contexts.
        if all(expert in info['expert_ids'] for (_, info, _) in parts):
            generated = np.memmap(dest, dtype=np.float32, shape=(len(x), 2048), mode='r')
            cursor = 0
            for (_, info, arrays) in parts:
                native = np.ascontiguousarray(arrays['expert_outputs'][:, info['expert_ids'].index(expert)])
                assert np.array_equal(np.asarray(generated[cursor:cursor + len(native)]).view(np.uint32), native.view(np.uint32))
                checked += native.size
                cursor += len(native)
        identity = dict(expert=expert, rows=len(x), shape=[len(x), 2048], dtype='float32',
                        path=str(dest), seconds=time.monotonic() - start,
                        target_sha256=sha(dest), input_sha256=input_identity,
                        weights_manifest_sha256=sha(weights / 'weights.json'),
                        weights_payload_sha256=weights_identity,
                        numeric_binary_sha256=numeric.binary_sha256,
                        all_native_capture_fixed_target_floats_bitchecked=checked)
        if hidden is not None:
            hidden.flush()
            del hidden
            hidden_temp.replace(hidden_dest)
            rec = dict(path=str(hidden_dest), shape=[len(x), 1408], dtype='float32',
                       sha256=sha(hidden_dest), meaning='Native routed expert pre-Down SwiGLU phi; before its own vector-A8. Not native shared branch.',
                       native_D_prefix=str(weights / f'expert{expert}.down'),
                       own_native_D_positive_control='Its own W8 D on its own raw phi, with native vector-A8 and zero bias, is canonical native output; this is a native/hot-GU control, not cross-expert reuse.')
            identity['prototype_hidden'] = rec
            index['prototype_hidden'][str(expert)] = rec
        dump(receipt, identity)
        index['targets'][str(expert)] = identity
        print(json.dumps(dict(layer=layer, expert=expert, rows=len(x), seconds=identity['seconds'])), flush=True)
    dump(out / 'targets.json', index)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('action', choices=['build', 'build-quality', 'benchmark', 'export', 'compute'])
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--layer', type=int)
    p.add_argument('--experts', default='all')
    p.add_argument('--captures', type=Path)
    p.add_argument('--weights', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--batch', type=int, default=2048)
    p.add_argument('--save-hidden-experts', default='')
    p.add_argument('--prototype-train-captures', type=Path)
    a = p.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    experts = list(range(64)) if a.experts == 'all' else list(map(int, a.experts.split(',')))
    if a.action == 'build':
        build()
    elif a.action == 'build-quality':
        build_quality()
    elif a.action == 'benchmark':
        benchmark(a.threads)
    elif a.action == 'export':
        export(a.layer, experts, a.output, a.threads)
    else:
        hidden, selection = prototype_selection(a.save_hidden_experts, a.prototype_train_captures, a.layer)
        compute(a.captures, a.layer, experts, a.weights, a.output, a.threads, a.batch, hidden, selection)


if __name__ == '__main__':
    main()
