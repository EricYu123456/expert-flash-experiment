#!/usr/bin/env python3
"""Read-only source audit of the existing real-routing NPZ caches.

Verifies corpus/split IDs and reconstructs the first and last prompt of every
model/split from raw JSONL, without calling the producer's validation function.
"""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = Path('/home/nfs_client/ericyu/expert_flash_experiment')
RUN = DATA / 'runs/corpus-300-v1'
MODELS = ('granite', 'olmoe', 'deepseek', 'qwen3', 'mixtral')


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            value.update(chunk)
    return value.hexdigest()


def main():
    corpus_path = RUN / 'corpus/prompts.jsonl'
    corpus = [json.loads(line) for line in corpus_path.read_text().splitlines()]
    split_ids = {split: [row['id'] for row in corpus if row['split'] == split]
                 for split in ('train', 'test')}
    assert len(split_ids['train']) == 200 and len(split_ids['test']) == 100
    assert not set(split_ids['train']) & set(split_ids['test'])
    receipts = []
    for model in MODELS:
        original_path = RUN / model / 'analysis/frozen_plan.json'
        original = json.loads(original_path.read_text())
        metadata_path = DATA / 'models' / (model + '-metadata.json')
        metadata = json.loads(metadata_path.read_text())
        assert original['corpus_sha256'] == digest(corpus_path)
        assert original['model_sha256'] == metadata['sha256']
        assert {row['id'] for row in original['training_routing']} == set(split_ids['train'])
        for mappings in original['mappings'].values():
            for mapping in mappings.values():
                assert sorted(mapping) == list(range(len(mapping)))
        training_hashes = {row['id']: row['routing_sha256'] for row in original['training_routing']}
        for split in ('train', 'test'):
            cache_path = RUN / model / 'analysis' / (split + '_routes.npz')
            with np.load(cache_path, allow_pickle=False) as data:
                ids, layers, routes = data['prompt_ids'].tolist(), data['layers'].tolist(), data['routes']
            assert ids == split_ids[split]
            assert routes.shape[:2] == (len(ids), 128)
            selected_receipts = []
            for index in (0, len(ids) - 1):
                prompt_id = ids[index]
                raw = RUN / model / 'traces' / (prompt_id + '.routing.jsonl')
                reconstructed = np.full(routes.shape[1:], -1, dtype=routes.dtype)
                layer_index = {layer: i for i, layer in enumerate(layers)}
                seen = set()
                with raw.open() as stream:
                    for line in stream:
                        event = json.loads(line)
                        token, layer = event['token_index'], event['layer']
                        key = token, layer
                        assert key not in seen and 0 <= token < 128 and layer in layer_index
                        assert event['event'] == 'routing' and event['phase'] == 'decode'
                        assert len(set(event['expert_ids'])) == routes.shape[-1]
                        reconstructed[token, layer_index[layer]] = event['expert_ids']
                        seen.add(key)
                assert len(seen) == 128 * len(layers)
                np.testing.assert_array_equal(reconstructed, routes[index])
                raw_hash = digest(raw)
                if split == 'train':
                    assert raw_hash == training_hashes[prompt_id]
                selected_receipts.append(dict(prompt_id=prompt_id, raw_sha256=raw_hash,
                                              reconstructed_routing_events=len(seen),
                                              full_128_tokens_bit_exact=True))
            receipts.append(dict(model=model, split=split, cache_sha256=digest(cache_path),
                                 shape=list(routes.shape), corpus_ids_order_exact=True,
                                 original_frozen_plan_sha256=digest(original_path),
                                 raw_prompt_checks=selected_receipts))
    target = Path(__file__).with_name('audit_trace_provenance.json')
    target.write_text(json.dumps(dict(corpus_sha256=digest(corpus_path),
                                     disjoint_train_test_ids=True,
                                     scope='All cache IDs; first and last raw prompt of each model/split',
                                     checks=receipts), indent=2) + '\n')
    print(json.dumps(dict(models=len(MODELS), splits=len(receipts), raw_prompts=20,
                          raw_routing_events=sum(p['reconstructed_routing_events']
                                                 for r in receipts for p in r['raw_prompt_checks']),
                          result=str(target)), indent=2))


if __name__ == '__main__':
    main()
