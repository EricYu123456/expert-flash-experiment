#!/usr/bin/env python3
"""Independent NumPy/GGUF check of real cached expert quantization, not inference."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
GGUF_SOURCE = Path('/home/nfs_client/ericyu/expert_flash_experiment/sources/llama.cpp-927dada6c/gguf-py')
sys.path.insert(0, str(GGUF_SOURCE))
from gguf import GGMLQuantizationType
from gguf.quants import dequantize


def fnv(data):
    result = 14695981039346656037
    for value in memoryview(data).cast('B'):
        result = ((result ^ value) * 1099511628211) & ((1 << 64) - 1)
    return result


def main():
    checks = []
    for model in ('granite4tiny', 'qwen3'):
        d = json.loads((HERE / model / 'numeric.json').read_text())
        metadata_path = Path('/home/nfs_client/ericyu/expert_flash_experiment/models') / (model + '-metadata.json')
        metadata = json.loads(metadata_path.read_text())
        # One actual cached expert per targeted matrix, all its rows and weights.
        chosen = {}
        for entry in d['weight_cache']:
            tensor = entry['key'].rsplit('/', 1)[0]
            chosen.setdefault(tensor, entry)
        for tensor, entry in chosen.items():
            C, R = entry['columns'], entry['rows']
            dtype = {'q4_K': GGMLQuantizationType.Q4_K, 'q8_0': GGMLQuantizationType.Q8_0}[entry['source_type']]
            row_bytes = C // (256 if dtype == GGMLQuantizationType.Q4_K else 32) * (144 if dtype == GGMLQuantizationType.Q4_K else 34)
            packed = np.memmap(metadata['model_path'], mode='r', dtype=np.uint8,
                               offset=entry['source_byte_offset'], shape=(R, row_bytes))
            original = dequantize(np.array(packed), dtype).astype(np.float32)
            assert original.shape == (R, C)
            amax = np.max(np.abs(original), axis=1)
            scales = np.where(amax == 0, np.float32(1), amax / np.float32(127)).astype(np.float32)
            normalized = (original / scales[:, None]).astype(np.float32)
            # Convert normalized F32 exactly to F64 before adding .5, avoiding an
            # extra F32 rounding at just-below-half values; std::round ties away.
            rounded = np.sign(normalized) * np.floor(np.abs(normalized.astype(np.float64)) + .5)
            quant = np.clip(rounded, -127, 127).astype(np.int8)
            scale_path = Path(entry['scales_f32le_path'])
            if not scale_path.is_absolute():
                scale_path = ROOT / scale_path
            cached_scales = np.fromfile(scale_path, dtype='<f4')
            scale_exact = np.array_equal(scales.view(np.uint32), cached_scales.view(np.uint32))
            quant_digest = fnv(quant)
            digest_exact = quant_digest == int(entry['weights_fnv1a64'])
            record = dict(model=model, tensor=tensor, expert=int(entry['key'].rsplit('/', 1)[1]),
                rows=R, columns=C, weights_checked=R * C, scale_bits_exact=scale_exact,
                quantized_weights_digest_exact=digest_exact, quantized_i8_sha256=hashlib.sha256(quant.tobytes()).hexdigest(),
                scales_sha256=hashlib.sha256(scales.tobytes()).hexdigest(),
                cache_fnv1a64=entry['weights_fnv1a64'], independent_fnv1a64=str(quant_digest))
            checks.append(record)
    report = dict(status='PASS' if all(c['scale_bits_exact'] and c['quantized_weights_digest_exact'] for c in checks) else 'FAIL',
                  real_expert_matrices_checked=len(checks), rows_checked=sum(c['rows'] for c in checks),
                  weights_checked=sum(c['weights_checked'] for c in checks), checks=checks,
                  method='Independent Python GGUF dequantizer and NumPy F32 maxabs/scale/division; half-away rounding from exact normalized F32.',
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (HERE / 'independent_quantization_check.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'checks'}, indent=2))
    assert report['status'] == 'PASS'


if __name__ == '__main__':
    main()
