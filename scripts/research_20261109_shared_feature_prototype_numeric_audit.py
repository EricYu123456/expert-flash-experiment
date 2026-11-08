#!/usr/bin/env python3
"""Validate prototype source remapping and native-D ownership control."""
import json
import numpy as np
from research_20261109_shared_feature_targets import Numeric, DATA, dump, load_weights, sha
from research_20261109_shared_feature_prototype_transform import apply_cpp, apply_numpy


def main():
    numeric = Numeric(1)
    rng = np.random.default_rng(20261109)
    cases = []
    for width in (1408, 2560, 2816):
        N = 3
        x = rng.normal(size=(N, 2048)).astype(np.float32)
        shared = rng.normal(size=(N, 2816)).astype(np.float32)
        phi = {e: rng.normal(size=(N, 1408)).astype(np.float32) for e in (14, 40)}
        sources = rng.integers(0, 3, width, dtype=np.uint8)
        prototypes = np.asarray([(-1 if s != 2 else rng.choice([14, 40])) for s in sources], np.int32)
        indices = np.asarray([rng.integers([2048, 2816, 1408][s]) for s in sources], np.int32)
        center = rng.normal(size=width).astype(np.float32)
        inv = np.exp(rng.uniform(-5, 5, width)).astype(np.float32)
        a = apply_cpp(numeric, x, shared, phi, sources, indices, prototypes, center, inv)
        b = apply_numpy(x, shared, phi, sources, indices, prototypes, center, inv)
        assert np.array_equal(a.view(np.uint32), b.view(np.uint32))
        cases.append(dict(width=width, mixed_raw_shared_two_prototypes=True, transformed_floats_bitchecked=a.size))
    controls = []
    for layer in (1, 13, 26):
        folder = DATA / f'targets64/train/layer{layer}'
        selected = json.loads((folder / 'prototype_selection.json').read_text())['expert_ids']
        for e in selected:
            receipt = folder / f'expert{e}.json'
            if not receipt.exists():
                continue
            m = json.loads(receipt.read_text())
            if 'prototype_hidden' not in m:
                continue
            N = min(64, m['rows'])
            phi = np.ascontiguousarray(np.memmap(m['prototype_hidden']['path'], dtype=np.float32, shape=(m['rows'], 1408), mode='r')[:N])
            target = np.ascontiguousarray(np.memmap(m['path'], dtype=np.float32, shape=(m['rows'], 2048), mode='r')[:N])
            w, sw = load_weights(DATA / f'weights/layer{layer}', e)['down']
            bias = np.zeros(2048, np.float32)
            out = np.empty(target.shape, np.float32)
            assert numeric.lib.sf_predict(N, 1408, 2048, phi, w, sw, bias, out) == 0
            assert np.array_equal(out.view(np.uint32), target.view(np.uint32))
            controls.append(dict(layer=layer, owner=e, rows=N, native_D_on_own_phi_floats_bitchecked=out.size,
                                 interpretation='Native/hot-GU ownership positive control; no cross-expert reuse or approximation claim'))
            break
    assert len(controls) == 3
    report = dict(status='passed', CPU_only=True, cases=cases, native_D_ownership_controls=controls,
                  numerical_binary_sha256=numeric.binary_sha256,
                  prototype_wrapper_sha256=sha(__file__.replace('_prototype_numeric_audit.py', '_prototype_transform.py')),
                  new_quantization_boundaries=False,
                  contract='Gather source pool, F32 subtract and multiply, whole transformed vector A8, canonical row-W8 C. Native shared Down and each native owner D retain their own original vector-A8 boundary.')
    dump(DATA / 'benchmark/prototype_numeric_audit.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
