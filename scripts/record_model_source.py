#!/usr/bin/env python3
"""Record local source provenance without copying or modifying model weights.

Large safetensors payloads are not rehashed: this records each file's size,
header hash and the existing Hugging Face download metadata when available.
The converted GGUF is fully hashed separately by inspect_gguf.py.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--llama-source', type=Path, default=Path(
        '/home/nfs_client/ericyu/expert_flash_experiment/sources/llama.cpp-927dada6c'))
    args = parser.parse_args()
    index_path = args.source / 'model.safetensors.index.json'
    index = json.loads(index_path.read_text())
    config = json.loads((args.source / 'config.json').read_text())
    headers, weight_files, dtype_counts = {}, [], {}
    for name in sorted(set(index['weight_map'].values())):
        path = args.source / name
        with path.open('rb') as stream:
            size_bytes = stream.read(8)
            header_size = struct.unpack('<Q', size_bytes)[0]
            if header_size > path.stat().st_size - 8:
                raise ValueError(f'Invalid safetensors header: {path}')
            header_bytes = stream.read(header_size)
        header = json.loads(header_bytes)
        payload_bytes = path.stat().st_size - 8 - header_size
        for tensor, value in header.items():
            if tensor == '__metadata__':
                continue
            start, end = value['data_offsets']
            assert 0 <= start <= end <= payload_bytes, (path, tensor)
            dtype_counts[value['dtype']] = dtype_counts.get(value['dtype'], 0) + 1
        headers[name] = header
        entry = dict(name=name, size=path.stat().st_size,
                     safetensors_header_sha256=hashlib.sha256(header_bytes).hexdigest(),
                     full_payload_sha256_status='not_rehashed')
        downloaded = args.source / '.cache/huggingface/download' / (name + '.metadata')
        if downloaded.exists():
            entry['download_metadata'] = downloaded.read_text().splitlines()
        weight_files.append(entry)
    for tensor, name in index['weight_map'].items():
        assert tensor in headers[name], (tensor, name)
    small_files = {name: digest(args.source / name) for name in (
        'config.json', 'model.safetensors.index.json', 'tokenizer.json',
        'tokenizer_config.json', 'tokenizer.model', 'special_tokens_map.json')
        if (args.source / name).exists()}
    result = dict(source_directory=str(args.source.resolve()), source_files_read_only=True,
                  config=config, source_file_sha256=small_files, weight_files=weight_files,
                  indexed_tensors=len(index['weight_map']), dtype_tensor_counts=dtype_counts,
                  index_tensors_and_payload_ranges_valid=True,
                  converter_commit=(args.llama_source / 'EXPERIMENT_SOURCE_COMMIT.txt').read_text().strip(),
                  converter_sha256=digest(args.llama_source / 'convert_hf_to_gguf.py'))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(output=str(args.output), shards=len(weight_files),
                          tensors=len(index['weight_map']), dtype_counts=dtype_counts)))


if __name__ == '__main__':
    main()
