#!/usr/bin/env python3
"""Independently bind packed routes to original prompts and raw trace events."""
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DATA = Path('/home/nfs_client/ericyu/expert_flash_experiment')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    source = json.loads((HERE / 'granite4tiny-source/source-proof.json').read_text())
    metadata_path = DATA / 'models/granite4tiny-metadata.json'
    metadata = json.loads(metadata_path.read_text())
    assert metadata['sha256'] == source['sha256']
    assert metadata['model_bytes'] == source['model_bytes']
    model_path = Path(metadata['model_path'])
    assert model_path.stat().st_size == source['model_bytes']
    # Download and metadata inspection independently hashed the complete GGUF.
    assert source['status'] == 'official_IBM_GGUF_pinned_LFS_hash_verified'
    arch = metadata['metadata']['general.architecture']
    assert arch == 'granitehybrid'
    assert metadata['metadata'][arch + '.expert_count'] == 64
    assert metadata['metadata'][arch + '.expert_used_count'] == 6
    assert metadata['routed_layer_ids'] == list(range(40))
    corpus = DATA / 'runs/corpus-300-v1/corpus/prompts.jsonl'
    rows = [json.loads(line) for line in corpus.read_text().splitlines()]
    train = [row for row in rows if row['split'] == 'train']
    result = []
    for mode, count, steps in [('smoke', 2, 16), ('screen', 32, 64)]:
        directory = HERE / 'granite4tiny-routes' / mode
        validation = json.loads((directory / 'validation.json').read_text())
        assert validation['original_corpus_sha256'] == sha(corpus)
        assert validation['metadata_sha256'] == sha(metadata_path)
        assert validation['ids'] == [row['id'] for row in train[:count]]
        route_path = directory / 'routes.npz'
        assert validation['route_sha256'] == sha(route_path)
        with np.load(route_path) as packed:
            array = packed['routes']
            ids = packed['prompt_ids'].tolist()
            layers = packed['layers'].tolist()
            assert str(packed['split']) == 'train'
        assert array.shape == (count, steps, 40, 6)
        assert ids == validation['ids'] and layers == list(range(40))
        assert array.min() >= 0 and array.max() < 64
        assert np.all(np.diff(np.sort(array, axis=-1), axis=-1) > 0)
        layer_index = {layer: i for i, layer in enumerate(layers)}
        events = 0
        run = Path(validation['run_directory']) / 'granite4tiny/traces'
        for pi, row in enumerate(train[:count]):
            prompt = corpus.parent / 'prompts' / (row['id'] + '.txt')
            assert sha(prompt) == row['prompt_sha256']
            stem = run / row['id']
            traced = json.loads(Path(str(stem) + '.traced.summary.json').read_text())
            baseline = json.loads(Path(str(stem) + '.baseline.summary.json').read_text())
            assert traced['trace_enabled'] and not baseline['trace_enabled']
            assert traced['generated_token_ids'] == baseline['generated_token_ids']
            assert traced['prompt_token_ids'] == baseline['prompt_token_ids']
            assert Path(str(stem) + '.traced.output.txt').read_bytes() == Path(str(stem) + '.baseline.output.txt').read_bytes()
            raw = Path(str(stem) + '.routing.jsonl')
            receipt = next(v for v in validation['validation'] if v['id'] == row['id'])
            assert receipt['routing_sha256'] == sha(raw)
            seen = set()
            for line in raw.read_text().splitlines():
                event = json.loads(line)
                ti, layer = event['token_index'], event['layer']
                assert (ti, layer) not in seen
                seen.add((ti, layer))
                assert event['event'] == 'routing' and event['phase'] == 'decode'
                assert event['position'] == traced['prompt_tokens'] + ti
                assert event['token_id'] == traced['generated_token_ids'][ti]
                np.testing.assert_array_equal(event['expert_ids'], array[pi, ti, layer_index[layer]])
                events += 1
            assert len(seen) == steps * 40
        assert events == count * steps * 40
        result.append(dict(mode=mode, shape=list(array.shape), route_sha256=sha(route_path),
            raw_events=events, prompts=count, steps=steps, actual_experts_seen=int(np.unique(array).size),
            original_train_identity_verified=True, prompt_file_hashes_verified=True,
            packed_array_matches_every_raw_event=True, traced_untraced_ids_and_bytes_equal=True,
            no_test_inference=True,
            prefix_note='Legacy collector calls the capped prefix prefix32; the smoke run contains only 16 decode steps. Full IDs/bytes are checked separately.'))
    output = dict(status='passed', model_path=str(model_path), model_sha256=source['sha256'],
        official_GGUF_revision=source['source_records']['ibm-granite/granite-4.0-h-tiny-GGUF']['revision'],
        metadata_path=str(metadata_path), metadata_sha256=sha(metadata_path),
        corpus_sha256=sha(corpus), collections=result,
        modeling_scope='Actual released Q4_K_M checkpoint routes; no numerical claim for a hypothetical W8 checkpoint. Placement service models must declare their own fixed weight format and only compare identical policies at that format.',
        architecture_exclusions=dict(always_active_shared_ffn_layers=metadata['shared_ffn_layer_ids'],
            shared_intermediate_width=metadata['metadata'][arch + '.expert_shared_feed_forward_length'],
            ssm_tensor_count=sum('ssm' in tensor['name'] for tensor in metadata['tensors']),
            interpretation='Routed FFN placement cannot be labeled full-model speedup.'),
        files_sha256={str(path.relative_to(ROOT)): sha(path) for path in [
            HERE / 'collect_granite4tiny.py', HERE / 'download_granite4tiny.py',
            HERE / 'granite4tiny-source/source-proof.json', HERE / 'granite4tiny-source/binary-source-proof.json',
            ROOT / 'scripts/run_corpus_traces.py', ROOT / 'scripts/llama_corpus_trace.cpp',
            ROOT / 'scripts/inspect_gguf.py']})
    (HERE / 'granite4tiny-routes/audit.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({key:value for key,value in output.items() if key not in ('architecture_exclusions','files_sha256')}, indent=2))


if __name__ == '__main__':
    main()
