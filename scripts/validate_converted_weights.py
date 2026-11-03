#!/usr/bin/env python3
"""Check sampled Q8_0 weights against local BF16 / block-scaled FP8 sources.

This validates the numerical conversion, including FP8 scales, independently
of the converter's tensor transformations. It is not an accuracy benchmark.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from safetensors import safe_open


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-model', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--blocks-per-tensor', type=int, default=96)
    parser.add_argument('--llama-source', type=Path, default=Path(
        '/home/nfs_client/ericyu/expert_flash_experiment/sources/llama.cpp-927dada6c'))
    args = parser.parse_args()
    if args.blocks_per_tensor < 2:
        parser.error('Need at least two sampled blocks')
    sys.path.insert(0, str(args.llama_source / 'gguf-py'))
    from gguf import GGUFReader, GGMLQuantizationType

    config = json.loads((args.source_model / 'config.json').read_text())
    index = json.loads((args.source_model / 'model.safetensors.index.json').read_text())['weight_map']
    reader = GGUFReader(str(args.model))
    tensors = {t.name: t for t in reader.tensors}
    h = config['hidden_size']
    intermediate = config.get('moe_intermediate_size', config['intermediate_size'])
    experts = config.get('num_experts', config.get('num_local_experts'))
    layers = config['num_hidden_layers']
    is_mixtral = config['model_type'] == 'mixtral'
    block_size = config.get('quantization_config', {}).get('weight_block_size')
    checks = [('model.embed_tokens.weight', 'token_embd.weight', 0)]
    for layer in (0, layers-1):
        for expert in (0, experts-1):
            for projection in ('gate', 'up', 'down'):
                part = dict(gate='w1', up='w3', down='w2')[projection] if is_mixtral else projection+'_proj'
                module = 'block_sparse_moe.experts' if is_mixtral else 'mlp.experts'
                source = f'model.layers.{layer}.{module}.{expert}.{part}.weight'
                target = f'blk.{layer}.ffn_{projection}_exps.weight'
                checks.append((source, target, expert*h*intermediate//32))
    records = []
    for position, (source_name, target_name, offset) in enumerate(checks):
        target = tensors[target_name]
        assert target.tensor_type == GGMLQuantizationType.Q8_0, target_name
        raw = target.data.reshape(-1, 34)
        with safe_open(args.source_model / index[source_name], framework='pt', device='cpu') as source_file:
            source = source_file.get_slice(source_name)
            rows, columns = source.get_shape()
            assert columns % 32 == 0, source_name
            nblocks = rows*columns//32
            rng = np.random.default_rng(position)
            samples = np.unique(np.concatenate((np.array([0, nblocks-1]),
                rng.integers(0, nblocks, args.blocks_per_tensor-2))))
            reference = []
            scale_name = source_name + '_scale_inv'
            scale = None
            if scale_name in index:
                with safe_open(args.source_model / index[scale_name], framework='pt', device='cpu') as scale_file:
                    scale = scale_file.get_tensor(scale_name).float().numpy()
                assert block_size == [128, 128], block_size
            for block in samples:
                row, column = divmod(int(block)*32, columns)
                values = source[row:row+1, column:column+32].float().numpy().reshape(32)
                if scale is not None:
                    values = values * scale[row//128, column//128]
                reference.append(values)
        reference = np.asarray(reference, dtype=np.float32)
        packed = np.asarray(raw[offset+samples]).copy()
        scales = packed[:, :2].copy().view(np.float16).astype(np.float32)
        actual = scales * packed[:, 2:].view(np.int8).astype(np.float32)
        assert np.isfinite(reference).all() and np.isfinite(actual).all(), source_name
        denominator = float(np.linalg.norm(reference))
        relative = float(np.linalg.norm(actual-reference))/max(denominator, 1e-30)
        assert relative < .02, (source_name, relative)
        records.append(dict(source_tensor=source_name, gguf_tensor=target_name,
                            sampled_blocks=len(samples), fp8_block_scales_applied=scale is not None,
                            relative_l2_error=relative,
                            max_absolute_error=float(np.max(np.abs(actual-reference))),
                            all_values_finite=True))
    result = dict(source_model=str(args.source_model.resolve()), model=str(args.model.resolve()),
                  interpretation='Deterministic numerical conversion samples, not a model accuracy evaluation.',
                  checks=records, total_sampled_weights=sum(r['sampled_blocks'] for r in records)*32,
                  all_checks_passed=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output), checks=len(records),
                          max_relative_l2_error=max(r['relative_l2_error'] for r in records),
                          all_checks_passed=True)))


if __name__ == '__main__':
    main()
