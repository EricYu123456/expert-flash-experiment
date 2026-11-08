#!/usr/bin/env python3
"""Small independent fixtures for deployed feature transforms and W8A8 order."""
import json
from pathlib import Path
import numpy as np

from research_20261109_shared_feature_targets import Numeric, DATA, ROOT, dump, sha


def quantize(x):
    x = np.asarray(x, np.float32)
    maximum = np.max(np.abs(x), axis=1)
    s = np.where(maximum == 0, np.float32(1), maximum / np.float32(127)).astype(np.float32)
    ratio = np.divide(x, s[:, None], dtype=np.float32).astype(np.float64)
    q = np.clip(np.copysign(np.floor(np.abs(ratio) + .5), ratio), -127, 127).astype(np.int8)
    return q, s


def main():
    numeric = Numeric(1)
    rng = np.random.default_rng(20261109)
    near = np.array([[0] * 8, [-127, 127, -.5, .5, np.nextafter(np.float32(.5), np.float32(0)),
                             np.nextafter(np.float32(-.5), np.float32(0)), -2.5, 2.5]], np.float32)
    q, s = numeric.quantize(near)
    qq, ss = quantize(near)
    assert np.array_equal(q, qq) and np.array_equal(s.view(np.uint32), ss.view(np.uint32))
    cases = []
    for width in (1408, 2048, 2560, 2816):
        N, H, Z, O = 5, 2048, 2816, 19
        x = rng.normal(size=(N, H)).astype(np.float32)
        z = (rng.normal(size=(N, Z)) * np.exp(rng.uniform(-5, 5, size=Z))).astype(np.float32)
        sources = rng.integers(0, 2, width, dtype=np.uint8)
        indices = np.asarray([rng.integers(Z if src else H) for src in sources], np.int32)
        center = rng.normal(scale=.3, size=width).astype(np.float32)
        inv = np.exp(rng.uniform(-4, 4, size=width)).astype(np.float32)
        out = np.empty((N, width), np.float32)
        assert numeric.lib.sf_transform(N, H, Z, width, x, z, sources, indices, center, inv, out) == 0
        raw = np.stack([z[:, idx] if src else x[:, idx] for src, idx in zip(sources, indices)], axis=1)
        expected = np.multiply(np.subtract(raw, center, dtype=np.float32), inv, dtype=np.float32)
        assert np.array_equal(out.view(np.uint32), expected.view(np.uint32))
        qx, sx = quantize(expected)
        qw, sw = quantize(rng.normal(size=(O, width)).astype(np.float32))
        bias = rng.normal(size=O).astype(np.float32)
        prediction = np.empty((N, O), np.float32)
        assert numeric.lib.sf_predict(N, width, O, out, qw, sw, bias, prediction) == 0
        d = qx.astype(np.int64) @ qw.astype(np.int64).T
        scale = np.multiply(sx[:, None], sw[None, :], dtype=np.float32)
        ref = np.add(np.multiply(d.astype(np.float32), scale, dtype=np.float32), bias, dtype=np.float32)
        assert np.array_equal(prediction.view(np.uint32), ref.view(np.uint32))
        cases.append(dict(width=width, transformed_floats_bitchecked=N * width,
                          predictor_floats_bitchecked=N * O, order='F32 subtract, F32 multiply, vector A8, exact integer dot, F32 scales/product, separate F32 bias'))
    report = dict(status='passed', CPU_only=True, seed=20261109, rounding_fixture_values=near.size,
                  cases=cases, numeric_binary_sha256=numeric.binary_sha256,
                  numeric_source_sha256=sha(ROOT / 'scripts/research_20261109_shared_feature_numeric.cpp'),
                  shared_transform_header_sha256=sha(ROOT / 'scripts/research_20261109_shared_feature_transform.hpp'),
                  scope='Numeric fixtures only; native expert-output equivalence is independently measured in native_target_benchmark.json.')
    dump(DATA / 'benchmark/numeric_audit.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
