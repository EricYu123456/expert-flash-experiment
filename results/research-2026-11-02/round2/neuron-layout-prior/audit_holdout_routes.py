#!/usr/bin/env python3
"""Audit frozen-plan confirmation routes independently of collector validation."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DATA = Path('/home/nfs_client/ericyu/expert_flash_experiment')
FROZEN = ROOT / 'results/research-2026-11-02/round2/geometry-study/granite4tiny.frozen.json'
FROZEN_SHA = 'a55f17010b7b48c6c396dee92bb1d127a541e28a93b94aee2e42aac9f3dc2538'
TRAIN_SHA = 'a6349fdb17ef09df0faa53cb4a6d5084445d65fbc2d94ef45c46da411e912f87'
SMOKE_SHA = '93c1bd27022e6577d6e8f6d5449e03d7267044b33933d1785471ae680aaed796'
METADATA_SHA = 'e9b83008e3128440af9f65b2c771e52a5892085e8151bf4c8918397bbfbf96e6'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    assert sha(FROZEN) == FROZEN_SHA
    frozen = json.loads(FROZEN.read_text())
    assert len(frozen['phases']) == 80
    assert frozen['training_routes_sha256'] == TRAIN_SHA
    assert sha(HERE / 'granite4tiny-routes/screen/routes.npz') == TRAIN_SHA
    assert sha(HERE / 'granite4tiny-routes/smoke/routes.npz') == SMOKE_SHA
    metadata_path = DATA / 'models/granite4tiny-metadata.json'
    assert sha(metadata_path) == METADATA_SHA
    metadata = json.loads(metadata_path.read_text())
    assert metadata['routed_layer_ids'] == list(range(40))
    collections = []
    observed_ids = set()
    for mode, count in [('test16', 16), ('external60', 60)]:
        directory = HERE / 'granite4tiny-routes' / mode
        validation_path = directory / 'validation.json'
        validation = json.loads(validation_path.read_text())
        selection = json.loads((directory / 'selection.json').read_text())
        corpus = Path(validation['corpus_path'])
        assert validation['corpus_sha256'] == sha(corpus)
        all_rows = [json.loads(line) for line in corpus.read_text().splitlines()]
        if mode == 'test16':
            rows = [row for row in all_rows if row['split'] == 'test'][:16]
            split = 'old_test_reuse_new_model'
        else:
            rows = all_rows
            manifest = json.loads(corpus.parent.parent.joinpath('corpus_manifest.json').read_text())
            assert sha(corpus) == manifest['corpus_sha256']
            split = 'fixed_public_confirmation_new_model'
        ids = [row['id'] for row in rows]
        assert len(ids) == count and len(set(ids)) == count
        assert not observed_ids.intersection(ids)
        assert not set(frozen['train_ids']).intersection(ids)
        observed_ids.update(ids)
        assert ids == validation['prompt_ids'] == selection['prompt_ids']
        assert validation['frozen_plan_sha256_before'] == FROZEN_SHA
        assert validation['frozen_plan_sha256_after'] == FROZEN_SHA
        assert validation['frozen_utc'] == frozen['frozen_utc']
        freeze_time = datetime.fromisoformat(frozen['frozen_utc'].replace('Z', '+00:00'))
        start_time = datetime.fromisoformat(validation['collection_started_utc'])
        finish_time = datetime.fromisoformat(validation['collection_finished_utc'])
        assert freeze_time < start_time < finish_time
        assert validation['fit_allowed'] is False
        assert validation['layout_modified'] is False
        assert validation['baseline_prefill_state_reused'] is False
        assert validation['metadata_sha256'] == METADATA_SHA
        route_path = directory / 'routes.npz'
        assert sha(route_path) == validation['route_sha256']
        with np.load(route_path) as packed:
            array = packed['routes']
            assert packed['prompt_ids'].tolist() == ids
            assert packed['layers'].tolist() == list(range(40))
            assert str(packed['split']) == split
        assert array.shape == (count, 64, 40, 6)
        assert array.min() >= 0 and array.max() < 64
        assert np.all(np.diff(np.sort(array, axis=-1), axis=-1) > 0)
        run = Path(validation['run_directory']) / 'granite4tiny/traces'
        binaries = list(Path(validation['run_directory']).glob('llama-corpus-trace-*'))
        assert len(binaries) == 1 and sha(binaries[0]) == validation['binary_sha256']
        events = 0
        for pi, row in enumerate(rows):
            prompt = corpus.parent / 'prompts' / (row['id'] + '.txt')
            assert sha(prompt) == row['prompt_sha256']
            stem = run / row['id']
            traced = json.loads(Path(str(stem) + '.traced.summary.json').read_text())
            baseline = json.loads(Path(str(stem) + '.baseline.summary.json').read_text())
            assert traced['trace_enabled'] and not baseline['trace_enabled']
            assert traced['generated_token_ids'] == baseline['generated_token_ids']
            assert traced['prompt_token_ids'] == baseline['prompt_token_ids']
            assert len(traced['generated_token_ids']) == 64
            assert Path(str(stem) + '.traced.output.txt').read_bytes() == Path(str(stem) + '.baseline.output.txt').read_bytes()
            raw = Path(str(stem) + '.routing.jsonl')
            receipt = next(v for v in validation['validation'] if v['id'] == row['id'])
            assert receipt['routing_sha256'] == sha(raw)
            seen = set()
            for line in raw.read_text().splitlines():
                event = json.loads(line)
                ti, layer = event['token_index'], event['layer']
                assert (ti, layer) not in seen
                assert event['event'] == 'routing' and event['phase'] == 'decode'
                assert event['position'] == traced['prompt_tokens'] + ti
                assert event['token_id'] == traced['generated_token_ids'][ti]
                np.testing.assert_array_equal(event['expert_ids'], array[pi, ti, layer])
                seen.add((ti, layer))
                events += 1
            assert len(seen) == 64 * 40
        assert events == count * 64 * 40
        collections.append(dict(mode=mode, shape=list(array.shape),
            route_path=str(route_path), route_sha256=sha(route_path),
            raw_events=events, actual_experts_seen=int(np.unique(array).size),
            prompt_ids_match_immutable_selection=True,
            prompt_file_hashes_verified=True,
            packed_array_matches_every_raw_event=True,
            traced_untraced_full_prompt_ids_decode_ids_and_output_bytes_equal=True,
            trace_enabled_vs_disabled_controls=True,
            fresh_baseline_prefill=True,
            collection_after_frozen_plan_verified=True,
            frozen_before_and_after=FROZEN_SHA,
            fit_allowed=False, layout_modified=False,
            qualification=validation['qualification'],
            selection_sha256=sha(directory / 'selection.json'),
            validation_sha256=sha(validation_path)))
    assert sha(FROZEN) == FROZEN_SHA
    output = dict(status='passed', audit_utc=datetime.now(timezone.utc).isoformat(),
        frozen_plan_path=str(FROZEN), frozen_plan_sha256=FROZEN_SHA,
        training_routes_unchanged_sha256=TRAIN_SHA,
        smoke_routes_unchanged_sha256=SMOKE_SHA,
        metadata_path=str(metadata_path), metadata_sha256=METADATA_SHA,
        model_sha256=metadata['sha256'], collections=collections,
        source_sha256=sha(Path(__file__)),
        routed_ffn_only=True,
        limitation='Released fixed Q4_K_M checkpoint routing is measured. W8 physical service is a separate arithmetic model, not measured end-to-end latency or a quality claim. Both heldout corpora were previously used with other models; neither is a blind new source-family test.')
    out_path = HERE / 'granite4tiny-routes/holdout_audit.json'
    out_path.write_text(json.dumps(output, indent=2) + '\n')
    completion = dict(status='complete', audit_path=str(out_path), audit_sha256=sha(out_path),
        frozen_plan_sha256=FROZEN_SHA,
        collections=[{key:c[key] for key in ['mode','shape','route_path','route_sha256','raw_events']} for c in collections],
        no_fit_or_plan_changes=True,
        limitation=output['limitation'])
    (HERE / 'holdout_completion.json').write_text(json.dumps(completion, indent=2) + '\n')
    print(json.dumps(completion, indent=2))


if __name__ == '__main__':
    main()
