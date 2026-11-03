#!/usr/bin/env python3
"""Collect frozen-plan confirmation routes without fitting or changing layouts."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'scripts'))
from run_corpus_traces import DATA, run_model, sha256, validate_prompt

HERE = Path(__file__).resolve().parent
FROZEN = ROOT / 'results/research-2026-11-02/round2/geometry-study/granite4tiny.frozen.json'
FROZEN_SHA = 'a55f17010b7b48c6c396dee92bb1d127a541e28a93b94aee2e42aac9f3dc2538'
TRAIN_SHA = 'a6349fdb17ef09df0faa53cb4a6d5084445d65fbc2d94ef45c46da411e912f87'
METADATA_SHA = 'e9b83008e3128440af9f65b2c771e52a5892085e8151bf4c8918397bbfbf96e6'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['test16', 'external60'])
    a = parser.parse_args()
    assert sha256(FROZEN) == FROZEN_SHA
    frozen = json.loads(FROZEN.read_text())
    assert len(frozen['phases']) == 80 and frozen['training_routes_sha256'] == TRAIN_SHA
    metadata_path = DATA / 'models/granite4tiny-metadata.json'
    assert sha256(metadata_path) == METADATA_SHA
    if a.mode == 'test16':
        corpus = DATA / 'runs/corpus-300-v1/corpus/prompts.jsonl'
        rows = [json.loads(line) for line in corpus.read_text().splitlines()]
        rows = [row for row in rows if row['split'] == 'test'][:16]
        split = 'old_test_reuse_new_model'
        selection = 'First 16 old-test rows in immutable original corpus order.'
        qualification = 'Old corpus aggregate outcomes have been used with other models; this is not a blind source-family test. Granite4 routes were not collected before layout freeze.'
    else:
        corpus = ROOT / 'results/research-2026-11-02/round2/public-validation/corpus/prompts.jsonl'
        manifest = json.loads(corpus.parent.parent.joinpath('corpus_manifest.json').read_text())
        assert sha256(corpus) == manifest['corpus_sha256']
        rows = [json.loads(line) for line in corpus.read_text().splitlines()]
        assert len(rows) == 60
        split = 'fixed_public_confirmation_new_model'
        selection = 'All 60 prompts from previously frozen public corpus; no reselection.'
        qualification = 'The same public corpus was evaluated on other models; Granite4 routes were not observed before its layout freeze. This is not a new never-used benchmark family.'
    assert set(row['id'] for row in rows).isdisjoint(frozen['train_ids'])
    for row in rows:
        assert sha256(corpus.parent / 'prompts' / (row['id'] + '.txt')) == row['prompt_sha256']
    run_dir = DATA / 'runs/research-2026-11-02/granite4tiny' / a.mode
    run_dir.mkdir(parents=True, exist_ok=True)
    evidence = HERE / 'granite4tiny-routes' / a.mode
    evidence.mkdir(parents=True, exist_ok=True)
    binary = DATA / 'build/llama-route-trace/llama-corpus-trace'
    pinned = run_dir / ('llama-corpus-trace-' + sha256(binary))
    if not pinned.exists():
        shutil.copy2(binary, pinned)
    assert sha256(binary) == sha256(pinned)
    provenance = dict(mode=a.mode, split=split, corpus_path=str(corpus),
        corpus_sha256=sha256(corpus), prompt_ids=[row['id'] for row in rows],
        prompts=len(rows), steps=64, batch_size=8, threads=8,
        frozen_plan_path=str(FROZEN), frozen_plan_sha256_before=FROZEN_SHA,
        frozen_utc=frozen['frozen_utc'], collection_started_utc=datetime.now(timezone.utc).isoformat(),
        selection=selection, qualification=qualification,
        fit_allowed=False, layout_modified=False,
        baseline_prefill_state_reused=False,
        metadata_sha256=METADATA_SHA, binary_sha256=sha256(pinned),
        collector_source_sha256=sha256(ROOT / 'scripts/run_corpus_traces.py'),
        isolated_script_sha256=sha256(Path(__file__)),
        prompt_composition=dict(Counter(row['task'] for row in rows)))
    (evidence / 'selection.json').write_text(json.dumps(provenance, indent=2) + '\n')
    args = SimpleNamespace(data=DATA, run_dir=run_dir, corpus=corpus, binary=pinned,
        steps=64, batch_size=8, threads=8, inference_workers=1, reuse_prefill=False)
    run_model('granite4tiny', args, rows)
    assert sha256(FROZEN) == FROZEN_SHA, 'Frozen layout changed during collection'
    assert sha256(metadata_path) == METADATA_SHA
    metadata = json.loads(metadata_path.read_text())
    validations, values = [], []
    for row in rows:
        receipt, route = validate_prompt(run_dir / 'granite4tiny/traces' / row['id'], row, metadata, 64)
        validations.append(receipt)
        values.append(route)
    routes = np.stack(values)
    assert routes.shape == (len(rows), 64, 40, 6)
    route_path = evidence / 'routes.npz'
    np.savez_compressed(route_path, routes=routes, prompt_ids=np.asarray(provenance['prompt_ids']),
        layers=np.asarray(metadata['routed_layer_ids']), split=np.asarray(split))
    final = dict(**provenance, frozen_plan_sha256_after=sha256(FROZEN),
        validation=validations, route_path=str(route_path), route_sha256=sha256(route_path),
        routes_shape=list(routes.shape), metadata_path=str(metadata_path),
        run_directory=str(run_dir), all_greedy_ids_and_bytes_identical=True,
        collection_finished_utc=datetime.now(timezone.utc).isoformat(),
        status='complete frozen-plan heldout route collection; no fitting or latency analysis')
    (evidence / 'validation.json').write_text(json.dumps(final, indent=2) + '\n')
    print(json.dumps({key:value for key,value in final.items() if key not in ('validation','prompt_ids')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
