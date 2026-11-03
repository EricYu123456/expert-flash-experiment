#!/usr/bin/env python3
"""Summarize real decode callbacks; no model or method fitting."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(model, prompt, left, right):
    directory = HERE / model
    a = json.loads((directory / f'{prompt}.{left}.summary.json').read_text())
    b = json.loads((directory / f'{prompt}.{right}.summary.json').read_text())
    ra = [json.loads(x) for x in (directory / f'{prompt}.{left}.routing.jsonl').read_text().splitlines()]
    rb = [json.loads(x) for x in (directory / f'{prompt}.{right}.routing.jsonl').read_text().splitlines()]
    assert a['prompt_token_ids'] == b['prompt_token_ids']
    assert len(ra) == len(rb)
    for x, y in zip(ra, rb):
        assert (x['token_index'], x['layer']) == (y['token_index'], y['layer'])
    tokens_a, tokens_b = a['generated_token_ids'], b['generated_token_ids']
    first_token_difference = next((i for i, (x, y) in enumerate(zip(tokens_a, tokens_b)) if x != y), None)
    ordered = [x['expert_ids'] != y['expert_ids'] for x, y in zip(ra, rb)]
    sets = [set(x['expert_ids']) != set(y['expert_ids']) for x, y in zip(ra, rb)]
    prefix_limit = len(tokens_a) if first_token_difference is None else first_token_difference
    prefix = [(x, y) for x, y in zip(ra, rb) if x['token_index'] < prefix_limit]
    return dict(prompt_id=prompt, left=left, right=right, generated_token_ids_left=tokens_a,
                generated_token_ids_right=tokens_b, generated_tokens_exact=tokens_a == tokens_b,
                matching_token_positions=sum(x == y for x, y in zip(tokens_a, tokens_b)),
                first_different_generated_token_index=first_token_difference,
                routing_events=len(ra), ordered_route_differences=sum(ordered),
                route_set_differences=sum(sets), routing_exact=not any(ordered),
                routing_sets_exact=not any(sets),
                common_token_prefix_routing=dict(events=len(prefix),
                    ordered_route_differences=sum(x['expert_ids'] != y['expert_ids'] for x, y in prefix),
                    route_set_differences=sum(set(x['expert_ids']) != set(y['expert_ids']) for x, y in prefix)),
                first_changed_expert_set=next((dict(token_index=x['token_index'], layer=x['layer'],
                    left=x['expert_ids'], right=y['expert_ids']) for x, y in zip(ra, rb)
                    if set(x['expert_ids']) != set(y['expert_ids'])), None))


def main():
    frozen = json.loads((HERE / 'frozen_inputs.json').read_text())
    models = []
    for model in ('granite4tiny', 'qwen3'):
        numeric = json.loads((HERE / f'{model}.numeric.json').read_text())
        expected_phases = 6 if model == 'granite4tiny' else 2
        assert len(numeric['phases']) == expected_phases
        assert not numeric['full_model_values_modified']
        expected_rows = 98304 if model == 'granite4tiny' else 524288
        for phase in numeric['phases']:
            assert phase['callbacks'] == 16 and phase['dot_rows'] == expected_rows
            assert phase['full_vs_dst']['nonfinite'] == 0
        comparisons = []
        for prompt in frozen['prompts']:
            for left, right in (('control', 'capture'), ('control', 'overwrite'),
                                ('control', 'overwrite-full'), ('overwrite-full', 'overwrite')):
                comparisons.append(compare(model, prompt['id'], left, right))
        assert all(c['generated_tokens_exact'] and c['routing_exact']
                   for c in comparisons if c['right'] == 'capture')
        metrics = {}
        for name in ('full_vs_dst', 'split2_vs_full', 'split2_vs_dst', 'split3_vs_full'):
            stats = [phase[name] for phase in numeric['phases'] if phase[name]['count']]
            if stats:
                metrics[name] = dict(dot_rows=sum(s['count'] for s in stats),
                    bitwise_different=sum(s['bitwise_different'] for s in stats),
                    max_abs=max(s['max_abs'] for s in stats),
                    max_relative_unclipped=max(s['max_relative_unclipped'] for s in stats),
                    max_ulp=max(s['max_ulp'] for s in stats),
                    minimum_phase_relative_l2=min(s['relative_l2'] for s in stats),
                    maximum_phase_relative_l2=max(s['relative_l2'] for s in stats))
        models.append(dict(model=model, captured_numeric_metrics=metrics,
                           numerical_phases=numeric['phases'], comparisons=comparisons))
    checks = {p: sha(Path(p)) == h for p, h in frozen['input_sha256'].items()}
    assert all(checks.values())
    report = dict(status='completed_small_train_smoke', models=models,
        original_inputs_hash_unchanged=checks, settings=frozen,
        no_downloaded_weights=True, no_test_prompts=True, no_method_fitting=True,
        scope='Original GGUF numerical software composition only, not native AiF format equivalence.',
        reduction='Original full input quantized once with CPU vec_dot_type.from_float; original immutable GGUF blocks retained; two float32 partial dots added in float32. G4 three-part 512+512+512 measured read-only.',
        controls='Independent native control and read-only capture are token/routing exact. Full-dot overwrite additionally isolates original-row vec_dot versus optimized/repacked CPU reduction. Split overwrite is compared both to native and full-dot overwrite.',
        limitations=['Two original-training prompts and 16 decode steps per prompt/model are insufficient for model quality claims.',
            'Only G4 first-three-layer gate/up and Qwen first-two-layer down outputs were replaced; remaining model operators are unchanged.',
            'G4 three-part metrics are read-only; propagation run uses 1024+512.',
            'Tiny local relative L2 does not imply exact downstream routing or token equality.',
            'Maximum relative errors and ULP differences are reported without threshold filtering; near-zero outputs amplify these metrics.',
            'Native AiF accumulator/output/scales remain undocumented; CPU float32 outputs do not prove native support.',
            'Initial invalid runtime-repacked-buffer interpretation is archived separately and excluded from all valid numerical summaries.'])
    (HERE / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({r['model']: dict(metrics=r['captured_numeric_metrics'],
        comparisons=[{k: v for k, v in c.items() if k in ('prompt_id', 'left', 'right',
            'generated_tokens_exact', 'first_different_generated_token_index', 'routing_events',
            'ordered_route_differences', 'route_set_differences')} for c in r['comparisons']])
        for r in models}, indent=2))


if __name__ == '__main__':
    main()
