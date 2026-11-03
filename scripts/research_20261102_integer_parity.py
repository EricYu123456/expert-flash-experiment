#!/usr/bin/env python3
"""Exact integer row-output recovery for a proposed digital in-flash layout.

Radix decomposition and coded matvec are prior art. This executable checks the
specific 16-data-chip / opposite-half parity placement and its limitations.
It is not an implementation of a fabricated NAND accelerator.
"""
from itertools import combinations
import json
from pathlib import Path
import time

import numpy as np

from run_corpus_traces import ROOT, sha256

DEST = ROOT / 'results/research-2026-11-02/round2/integer-parity'


def split_i8(value):
    value = np.asarray(value, dtype=np.int64)
    low = ((value + 128) % 256) - 128
    high = (value - low) // 256
    assert np.all((-128 <= high) & (high <= 127))
    assert np.array_equal(value, low + 256 * high)
    return low.astype(np.int8), high.astype(np.int8)


def raw_gemv(weight, vector):
    # The proposed interface returns I32 accumulators; host recovery uses I64.
    value = np.asarray(weight, dtype=np.int64) @ np.asarray(vector, dtype=np.int64)
    assert np.all((-2**31 <= value) & (value < 2**31))
    return value.astype(np.int32)


def encode(weights, parity_count):
    assert weights.dtype == np.int8 and weights.shape[0] == 16
    result = {}
    for half in range(2):
        w = weights[half*8:half*8+8].astype(np.int64)
        for parity in range(parity_count):
            coefficient = np.ones(8, dtype=np.int64) if parity == 0 else np.arange(1, 9)
            value = np.einsum('i,ijk->jk', coefficient, w, dtype=np.int64)
            low, high = split_i8(value)
            # Four distinct destinations for Plo/Phi/Qlo/Qhi, outside data half.
            base = (1 - half) * 8
            result[(half, parity)] = dict(low=low, high=high,
                owners=(base + 2*parity, base + 2*parity + 1),
                coefficient=coefficient)
    return result


def recover(weights, vector, encoded, unavailable):
    expected = np.stack([raw_gemv(w, vector) for w in weights])
    outputs = {chip: expected[chip].astype(np.int64) for chip in range(16) if chip not in unavailable}
    parity_gemvs = 0
    for half in range(2):
        missing = [c for c in range(half*8, half*8+8) if c in unavailable]
        if not missing:
            continue
        equations = []
        for (h, p), entry in encoded.items():
            if h != half or set(entry['owners']) & set(unavailable):
                continue
            y = raw_gemv(entry['low'], vector).astype(np.int64)
            y += 256 * raw_gemv(entry['high'], vector).astype(np.int64)
            parity_gemvs += 2
            for chip in range(half*8, half*8+8):
                if chip not in missing:
                    y -= entry['coefficient'][chip-half*8] * outputs[chip]
            equations.append((entry['coefficient'], y))
            if len(equations) == len(missing):
                break
        if len(equations) < len(missing):
            raise ValueError('Available parity cannot recover these missing data shards')
        if len(missing) == 1:
            a, y = equations[0]
            coefficient = int(a[missing[0]-half*8])
            assert np.all(y % coefficient == 0)
            outputs[missing[0]] = y // coefficient
        elif len(missing) == 2:
            (a, s), (b, t) = equations
            i, j = [c-half*8 for c in missing]
            determinant = int(a[i]*b[j] - a[j]*b[i])
            assert determinant != 0
            yi = int(b[j])*s - int(a[j])*t
            yj = int(a[i])*t - int(b[i])*s
            assert np.all(yi % determinant == 0) and np.all(yj % determinant == 0)
            outputs[missing[0]], outputs[missing[1]] = yi // determinant, yj // determinant
        else:
            raise ValueError('At most two erased chips supported')
    rebuilt = np.stack([outputs[c] for c in range(16)])
    np.testing.assert_array_equal(rebuilt, expected.astype(np.int64))
    return rebuilt, parity_gemvs


def float32_counterexample():
    # A legal INT8 row at D=2048 loses low bits when its raw accumulator is
    # converted to FP32 before decoding. Output byte width 4 alone is insufficient.
    rng = np.random.default_rng(20261102)
    w = rng.integers(100, 128, (16, 1, 2048), dtype=np.int16).astype(np.int8)
    x = rng.integers(100, 128, 2048, dtype=np.int16).astype(np.int8)
    y = np.stack([raw_gemv(a, x) for a in w]).astype(np.int64)
    p = w[:8].astype(np.int64).sum(axis=0)
    low, high = split_i8(p)
    py = raw_gemv(low, x).astype(np.float32).astype(np.float64)
    py += 256 * raw_gemv(high, x).astype(np.float32).astype(np.float64)
    rebuilt = py - y[1:8].astype(np.float32).astype(np.float64).sum(axis=0)
    assert not np.array_equal(rebuilt, y[0])
    return dict(exact_raw_output=int(y[0, 0]), fp32_return_reconstruction=float(rebuilt[0]),
                demonstrates='I32 raw-output mode is required for exact reconstruction; FP32 is insufficient.')


def groupwise_scale_counterexample():
    # Summed block-scaled results cannot be decoded using unrelated row/group
    # scales. A common post-dot scale per row works; column-group scales do not.
    q = np.array([[1, 2, 3, 4], [3, 1, 2, 1]], dtype=np.int64)
    x = np.array([2, 1, 3, 2], dtype=np.int64)
    scale = np.array([[1., 2.], [3., 1.]])
    outputs = np.array([sum((q[r, g*2:g*2+2] @ x[g*2:g*2+2]) * scale[r, g]
                            for g in range(2)) for r in range(2)])
    parity_raw = (q[0] + q[1]) @ x
    naive = parity_raw - outputs[1]
    assert naive != outputs[0]
    return dict(actual_first_output=float(outputs[0]), invalid_naive_reconstruction=float(naive),
                demonstrates='Column-group quantization needs per-group partial recovery or a different code.')


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261102)
    cases = []
    started = time.time()
    for rows, columns in ((1, 1), (5, 31), (4, 129), (3, 2048)):
        weights = rng.integers(-128, 128, (16, rows, columns), dtype=np.int16).astype(np.int8)
        vector = rng.integers(-128, 128, columns, dtype=np.int16).astype(np.int8)
        for count in (1, 2):
            encoded = encode(weights, count)
            capacity = sum(e['low'].nbytes + e['high'].nbytes for e in encoded.values())
            assert capacity == weights.nbytes * count // 4
            checks = 0
            for nmissing in range(1, count + 1):
                for missing in combinations(range(16), nmissing):
                    rebuilt, _ = recover(weights, vector, encoded, missing)
                    # Distinct row scales applied after raw recovery preserve output.
                    scales = rng.uniform(.001, 1, (16, rows))
                    direct = np.stack([raw_gemv(w, vector) for w in weights])
                    np.testing.assert_array_equal(rebuilt * scales, direct.astype(np.int64) * scales)
                    checks += 1
            cases.append(dict(rows_per_shard=rows, columns=columns, parity_count=count,
                              all_chip_erasure_patterns_checked=checks,
                              extra_weight_fraction=capacity/weights.nbytes,
                              exact_integer_outputs=True, distinct_post_row_scales_supported=True))
    result = dict(cases=cases, total_erasure_checks=sum(c['all_chip_erasure_patterns_checked'] for c in cases),
                  fp32_failure=float32_counterexample(),
                  groupwise_scale_failure=groupwise_scale_counterexample(),
                  assumptions=['Equal-shaped sixteen data shards; original g8/unequal row layouts require padding/repartitioning.',
                    'Native signed INT8 weights and INT8 vector, raw I32 dot outputs before nonlinearities/scales.',
                    'No claim of bitwise parity for BF16/FP16 inputs or current GGUF groupwise Q8 arithmetic.',
                    'Two opposite data halves; P/Q each have two radix parts on disjoint destinations.',
                    'Recovery uses measured-or-modeled extra GEMV, transfers and I64 host work, not zero time.'],
                  prior_art=['Coded matrix-vector multiplication is established.',
                             'TR-ABFT already uses this signed radix-256 checksum decomposition.',
                             'RAIL/TTFLASH already use NAND parity to avoid blocking.'],
                  source_sha256=sha256(Path(__file__)), seconds=time.time()-started)
    (DEST / 'codec_checks.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
