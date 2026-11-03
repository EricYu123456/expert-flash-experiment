#!/usr/bin/env python3
"""Collect a new public math/code routing set with traced/untraced controls.

Prompts and selection are fixed before any new routing. This does not measure
benchmark answer quality or use reference solutions in inference prompts.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import random
import types

import numpy as np

import aif_model as flash
from run_corpus_traces import DATA, ROOT, run_model, sha256
from research_20261102_projection_study import evaluate, bootstrap

DEST = ROOT / 'results/research-2026-11-02/round2/public-validation'
RAW = DEST.parent / 'public-data'


def prepare():
    corpus = DEST / 'corpus'
    prompts = corpus / 'prompts'
    prompts.mkdir(parents=True, exist_ok=True)
    rows = []
    sources = [('gsm8k_test', 'https://github.com/openai/grade-school-math/blob/master/grade_school_math/data/test.jsonl'),
               ('mbpp', 'https://github.com/google-research/google-research/blob/master/mbpp/mbpp.jsonl')]
    for source, url in sources:
        data = [json.loads(line) for line in (RAW / (source + '.jsonl')).read_text().splitlines()]
        rng = random.Random(20261102 + (0 if source == 'gsm8k_test' else 1009))
        indices = sorted(rng.sample(range(len(data)), 30))
        for index in indices:
            item = data[index]
            if source == 'gsm8k_test':
                text = 'Solve the following mathematical problem and explain your reasoning.\n\n' + item['question'] + '\n'
            else:
                text = 'Write a Python function for the following task and explain your solution.\n\n' + item['text'] + '\n'
            pid = f'public_{source}_{index:04d}'
            target = prompts / (pid + '.txt')
            if target.exists():
                assert target.read_text() == text
            else:
                target.write_text(text)
            rows.append(dict(id=pid, split='external_confirmation', task=source,
                             length_bucket='source_native', prompt_sha256=hashlib.sha256(text.encode()).hexdigest(),
                             source_url=url, source_index=index))
    rng = random.Random(20261102)
    rng.shuffle(rows)
    path = corpus / 'prompts.jsonl'
    contents = ''.join(json.dumps(row) + '\n' for row in rows)
    if path.exists():
        assert path.read_text() == contents
    else:
        path.write_text(contents)
    manifest = dict(prompts=60, decode_steps=64, selection_seed=20261102,
                    source_hashes={s: sha256(RAW / (s + '.jsonl')) for s, _ in sources},
                    corpus_sha256=sha256(path), templates_include_answers=False,
                    purpose='new source-family confirmation of previously frozen projection layouts',
                    source_sha256=sha256(Path(__file__)))
    (DEST / 'corpus_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return rows, path


def collect(models, threads, workers):
    rows, corpus = prepare()
    args = types.SimpleNamespace(data=DATA, run_dir=DEST, corpus=corpus,
              binary=DATA / 'build/llama-route-trace/llama-corpus-trace',
              steps=64, threads=threads, batch_size=8, inference_workers=1, reuse_prefill=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_model, model, args, rows): model for model in models}
        for future in as_completed(futures):
            future.result()
            print('PUBLIC_MODEL_COMPLETE', futures[future], flush=True)


def analyze(models):
    rows, _ = prepare()
    results = []
    for name in models:
        plan_path = ROOT / 'results/research-2026-11-02/projection-study' / (name + '.frozen_plan.json')
        plan = json.loads(plan_path.read_text())
        metadata = json.loads((DATA / 'models' / (name + '-metadata.json')).read_text())
        validation = json.loads((DEST / name / 'validation.json').read_text())
        assert {r['id'] for r in validation} == {r['id'] for r in rows}
        assert all(r['baseline_output_identical'] and r['baseline_token_ids_identical'] for r in validation)
        ids = [r['id'] for r in rows]
        routes = np.zeros((len(rows), 64, len(plan['layers']), plan['active']), dtype=np.int16)
        li = {layer: i for i, layer in enumerate(plan['layers'])}
        for pi, row in enumerate(rows):
            path = DEST / name / 'traces' / (row['id'] + '.routing.jsonl')
            assert sha256(path) == next(v for v in validation if v['id'] == row['id'])['routing_sha256']
            for line in path.read_text().splitlines():
                event = json.loads(line)
                routes[pi, event['token_index'], li[event['layer']]] = event['expert_ids']
        np.savez_compressed(DEST / name / 'external_routes.npz', routes=routes,
                            prompt_ids=np.array(ids), layers=np.array(plan['layers']))
        totals = {label: np.zeros(routes.shape[:2], dtype=np.int64) for label in ('baseline', 'legacy', 'chosen')}
        for phase in plan['phases']:
            selected = routes[:, :, li[phase['layer']]].reshape(-1, plan['active'])
            for label in totals:
                values = evaluate(selected, phase[label + '_rows'], phase['columns'], phase['common'])
                totals[label] += values['model_ns'].reshape(routes.shape[:2])
        result = dict(model=name, means_ms={k: float(v.mean() / 1e6) for k, v in totals.items()},
                      relative_gain_pct=100 * (1 - totals['chosen'].mean() / totals['legacy'].mean()),
                      paired_prompt_ci_pct=bootstrap(totals['legacy'], totals['chosen']),
                      frozen_plan_sha256=sha256(plan_path), no_new_training=True,
                      source_families={task: {label: float(value[np.array([r['task'] == task for r in rows])].mean()/1e6)
                                             for label, value in totals.items()}
                                       for task in ('gsm8k_test', 'mbpp')})
        results.append(result)
        print(json.dumps(result), flush=True)
    (DEST / 'summary.json').write_text(json.dumps(dict(models=results,
          purpose='previously frozen layouts evaluated on new public prompts',
          full_model_latency_measured=False, output_controls_passed=True), indent=2) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('prepare', 'collect', 'analyze'), required=True)
    p.add_argument('--models', nargs='+', default=['granite', 'deepseek', 'qwen3'])
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--workers', type=int, default=2)
    a = p.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    if a.stage == 'prepare': prepare()
    elif a.stage == 'collect': collect(a.models, a.threads, a.workers)
    else: analyze(a.models)


if __name__ == '__main__':
    main()
