#!/usr/bin/env python3
"""Isolated true-route smoke/screen using the unchanged pinned collector."""
import argparse
from collections import Counter
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['smoke', 'screen'])
    args = parser.parse_args()
    original = DATA / 'runs/corpus-300-v1/corpus/prompts.jsonl'
    all_rows = [json.loads(line) for line in original.read_text().splitlines()]
    train = [row for row in all_rows if row['split'] == 'train']
    # Preserve original order and prompt-level train identity; never inspect test routes.
    steps, count, batch_size = (16, 2, 2) if args.mode == 'smoke' else (64, 32, 8)
    rows = train[:count]
    run_dir = DATA / 'runs/research-2026-11-02/granite4tiny' / args.mode
    run_dir.mkdir(parents=True, exist_ok=True)
    evidence = HERE / 'granite4tiny-routes' / args.mode
    evidence.mkdir(parents=True, exist_ok=True)
    binary = DATA / 'build/llama-route-trace/llama-corpus-trace'
    pinned = run_dir / ('llama-corpus-trace-' + sha256(binary))
    if not pinned.exists():
        shutil.copy2(binary, pinned)
    assert sha256(binary) == sha256(pinned)
    manifest = dict(mode=args.mode, split='original train only', original_corpus_path=str(original),
        original_corpus_sha256=sha256(original), sample_policy='First N train rows in immutable original corpus order.',
        ids=[row['id'] for row in rows], steps=steps, batch_size=batch_size, threads=8,
        collector_script_sha256=sha256(ROOT / 'scripts/run_corpus_traces.py'),
        trace_source_sha256=sha256(ROOT / 'scripts/llama_corpus_trace.cpp'),
        binary_path=str(pinned), binary_sha256=sha256(pinned),
        prompt_composition={str(key):value for key,value in Counter((r['task'],r['length_bucket']) for r in rows).items()},
        baseline='Independent full untraced prefill and decode; no prefill cache reuse.',
        routed_selection='Actual ffn_moe_topk callback; no synthetic routes or masks.')
    (evidence / 'selection.json').write_text(json.dumps(manifest, indent=2) + '\n')
    namespace = SimpleNamespace(data=DATA, run_dir=run_dir, corpus=original, binary=pinned,
        steps=steps, batch_size=batch_size, threads=8, inference_workers=1, reuse_prefill=False)
    run_model('granite4tiny', namespace, rows)
    metadata = json.loads((DATA / 'models/granite4tiny-metadata.json').read_text())
    validations, routes = [], []
    for row in rows:
        result, selected = validate_prompt(run_dir / 'granite4tiny/traces' / row['id'], row, metadata, steps)
        validations.append(result)
        routes.append(selected)
    array = np.stack(routes)
    assert array.shape[0:2] == (count, steps) and array.shape[-1] == 6
    assert all(row['experts'] == 64 and row['active_experts'] == 6 for row in validations)
    route_path = evidence / 'routes.npz'
    np.savez_compressed(route_path, routes=array,
        layers=np.asarray(metadata['routed_layer_ids']),
        prompt_ids=np.asarray([row['id'] for row in rows]), split=np.asarray('train'))
    output = dict(**manifest, validation=validations, routes_shape=list(array.shape),
        route_path=str(route_path), route_sha256=sha256(route_path),
        metadata_path=str(DATA / 'models/granite4tiny-metadata.json'),
        metadata_sha256=sha256(DATA / 'models/granite4tiny-metadata.json'),
        run_directory=str(run_dir),
        status='Actual 64-expert/top6 routes; traced/untraced greedy IDs, prompt IDs and output bytes identical.')
    (evidence / 'validation.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({key:value for key,value in output.items() if key not in ('validation','ids')}, indent=2))


if __name__ == '__main__':
    main()
