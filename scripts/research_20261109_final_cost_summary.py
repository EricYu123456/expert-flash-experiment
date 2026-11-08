#!/usr/bin/env python3
"""Summarize completed final TEST costs using only frozen VALID placements.

This wrapper adds completion/provenance guards around the frozen cost audit.
It neither fits models nor selects cache placements from TEST frequencies.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import time


HERE = Path(__file__).resolve().parent
COST_SOURCE = HERE / 'research_20261109_cost_audit.py'
DATA = Path('/home/nfs_client/ericyu/expert_flash_experiment/research-2026-11-09-shared-features')
SCOPE_NAMES = {'TF63': 'NLL_prefix63', 'nonBOS62': 'nonBOS_NLL_prefix62',
               'BOS': 'BOS', 'generation_causal': 'generation_causal'}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            value.update(block)
    return value.hexdigest()


def normalized(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def same(actual, expected, label):
    assert normalized(actual) == normalized(expected), label


def save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def verify_frozen(receipt):
    for path, expected in receipt['files_sha256'].items():
        assert sha(path) == expected, f'Changed frozen input: {path}'
    for path, expected in receipt['source_snapshots_sha256'].items():
        assert sha(path) == expected, f'Changed source snapshot: {path}'
    assert receipt['files_sha256'][str(COST_SOURCE)] == sha(COST_SOURCE)


def load_cost(receipt):
    assert sha(COST_SOURCE) == receipt['files_sha256'][str(COST_SOURCE)]
    # Set these before importing the frozen audit and its NumPy dependency.
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[key] = '1'
    spec = importlib.util.spec_from_file_location('frozen_final_cost_audit', COST_SOURCE)
    cost = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cost)
    assert sha(cost.qa.__file__) == receipt['files_sha256'][str(Path(cost.qa.__file__).resolve())]
    return cost


def completion_gate(plan_path):
    plan = read(plan_path)
    assert plan['stage'] == 'test' and plan['prompts'] == 1000
    assert plan['max_tokens'] == 64 and plan['steps'] == 16
    folder = Path(plan['output'])
    # This must precede opening any prompt summary or routing file.
    execution = read(folder / 'execution.json')
    assert execution['status'] == 'complete', 'Final TEST is still active or failed'
    assert execution['all_input_hashes_rechecked'] is True
    arms = {a['name']: a['registry'] for a in plan['arms']}
    assert len(plan['arms']) == len(arms) == 5 and arms['native'] == '-'
    assert len(execution['batches']) == 5 * plan['shards_per_arm']
    assert all(b['exit_code'] == 0 for b in execution['batches'])
    receipt_path = Path(plan['selection_receipt']).resolve()
    receipt = read(receipt_path)
    assert receipt['status'] == 'frozen_before_new_TEST'
    assert receipt['TEST_results_used_for_selection'] is False
    protocol = read(folder / 'protocol.frozen.json')
    same(protocol['plan'], plan, 'Executed plan differs from supplied plan')
    assert protocol['inputs_sha256'][str(receipt_path)] == sha(receipt_path)
    assert receipt['files_sha256'][str(Path(plan_path).resolve())] == sha(plan_path)
    verify_frozen(receipt)
    manifest_path = DATA / 'corpus103/manifest.json'
    records = [r for r in read(manifest_path)['records'] if r['split'] == 'test']
    ids = {r['id'] for r in records}
    assert len(ids) == len(records) == len({r['article_id'] for r in records}) == 1000
    finished = {}
    for name in arms:
        arm_folder = folder / name
        completions = sorted(arm_folder.glob('shard*/data/completion.json'))
        assert len(completions) == plan['shards_per_arm'], name
        seen = set()
        generated = 0
        for completion_path in completions:
            done = read(completion_path)
            assert done['status'] == 'complete' and done['gpu_layers'] == 0
            assert done['max_tokens'] == 64 and done['decode_steps'] == 16
            summaries = list(completion_path.parent.glob('*.summary.json'))
            assert len(summaries) == done['jobs']
            for path in summaries:
                value = read(path)
                prompt = value['prompt_id']
                assert prompt not in seen and prompt in ids
                seen.add(prompt)
                assert value['input_tokens'] == 64 and value['scored_tokens'] == 63
                assert value['decode_steps'] == 16
                length = value['actual_generated_length']
                assert 1 <= length <= 16 and len(value['generated_token_ids']) == length
                assert path.with_name(prompt + '.routing.jsonl').is_file()
                generated += length
        assert seen == ids, f'Incomplete 1000-prompt arm: {name}'
        finished[name] = dict(prompts=len(seen), expected_scope_events={
            'NLL_prefix63': 1000 * 63 * 26 * 6,
            'nonBOS_NLL_prefix62': 1000 * 62 * 26 * 6,
            'BOS': 1000 * 26 * 6, 'generation_causal': generated * 26 * 6})
    return plan, receipt, finished


def check_placements(cost, receipt, budget_name, registry, linear_registry, policy_path):
    frozen = receipt['cost']['placements'][budget_name]
    heads = cost.qa.registry(registry)
    raw_heads = cost.qa.registry(linear_registry)
    owners = {(l, p) for (l, _), m in heads.items() for p in m['prototypes']}
    assert owners == {(1, 25)} and set(heads) == set(raw_heads)
    host = cost.inventory(heads, owners)
    raw = cost.inventory(raw_heads)
    same(host, frozen['nonlinear_inventory'], 'Nonlinear inventory changed')
    same(raw, frozen['raw_inventory_without_forced_GU'], 'Linear inventory changed')
    assert host['total_method_host_bytes'] - raw['total_method_host_bytes'] == cost.GU
    policy = read(policy_path)
    assert policy['status'] == 'frozen_before_TEST' and policy['TEST_opened'] is False
    assert sha(policy_path) == receipt['files_sha256'][policy_path]
    ranking = policy['primary_ranking']
    order = [(r['layer'], r['expert']) for r in ranking]
    assert len(order) == len(set(order)) == 26 * 64
    assert set(order) == {(l, e) for l in range(1, 27) for e in range(64)}
    budget = host['total_method_host_bytes']
    _, mixed, left = cost.mixed_cache(ranking, budget)
    _, raw_mixed, raw_left = cost.mixed_cache(
        [r for r in ranking if (r['layer'], r['expert']) not in heads], cost.GU)
    recomputed = dict(same_budget_mixed_native=mixed, mixed_unused_bytes=left,
        same_budget_GU_cache=order[:budget // cost.GU],
        same_budget_full_expert_cache=order[:budget // cost.EXPERT],
        raw_readouts_plus_native_cache=raw_mixed, raw_unused_bytes=raw_left)
    same(recomputed, frozen['policies'][policy_path], 'Frozen cache placements changed')
    return host, raw, recomputed


def worker(task):
    receipt = read(task['receipt'])
    cost = load_cost(receipt)
    host, raw, placements = check_placements(cost, receipt, task['budget'],
        task['registry'], task['linear_registry'], task['policy'])
    result = cost.audit(task['registry'], task['quality_folder'],
        linear_registry=task['linear_registry'], policy_path=task['policy'])
    assert result['source_sha256'] == receipt['files_sha256'][str(COST_SOURCE)]
    same(result['host_inventory'], host, 'audit inventory differs from receipt')
    cache = result['cache_policies']
    assert cache['policy_sha256'] == receipt['files_sha256'][task['policy']]
    same(sorted(cache['hot_GU']), sorted(placements['same_budget_GU_cache']), 'GU placement')
    same(sorted(cache['full_expert']), sorted(placements['same_budget_full_expert_cache']), 'Full placement')
    same(cache['mixed_GU_D'], placements['same_budget_mixed_native'], 'Mixed placement')
    assert cache['mixed_unused_bytes'] == placements['mixed_unused_bytes']
    linear = result['actual_matched_linear_control_inventory']
    assert linear['runtime_coefficient_budget_equal_to_method'] is True
    same(linear['without_forced_shared_GU_bank'], raw, 'Raw inventory differs')
    spare = linear['optimized_native_cache_in_remaining_budget']
    assert spare['available_bytes'] == cost.GU
    same(spare['selected'], placements['raw_readouts_plus_native_cache'], 'Raw-plus-cache placement')
    assert spare['unused_bytes'] == placements['raw_unused_bytes']
    scopes = result['modeled_routed_FFN_coefficient_work']
    for scope, expected in task['expected_scope_events'].items():
        assert scopes[scope]['events'] == expected, f'Incomplete routing scope: {scope}'
    result.update(job={k: task[k] for k in ('name', 'budget', 'routing', 'policy_role',
                                          'registry', 'linear_registry', 'quality_folder')},
                  frozen_inventory_and_placements_rechecked=True,
                  worker_threads=1, selected_scopes={name: scopes[key] for name, key in SCOPE_NAMES.items()})
    return task['name'], result


def run(args):
    started = time.monotonic()
    plan_path = Path(args.plan).resolve()
    plan, receipt, finished = completion_gate(plan_path)
    arms = {a['name']: a['registry'] for a in plan['arms']}
    q = receipt['quality']
    primary, dense, linear = q['primary_arm'], q['dense_same_basis_arm'], q['matched_linear_arm']
    assert primary == 'prototype25_rank256_k12' and dense == 'prototype25_dense_k12'
    assert linear == 'raw_x_select1408_eq_rank256_k12'
    cost_rules = receipt['cost']
    assert cost_rules['primary_scope'] == 'All26 native-reference nonBOS NLL input positions1..62'
    policies = [('primary_v2', cost_rules['primary_cache_policy']),
                ('sensitivity_v1', cost_rules['sensitivity_cache_policy'])]
    raw_dense = str(DATA / 'proto25-cohort12/registries/registry.raw_x_select1408_eq.k12.tsv')
    assert raw_dense in receipt['files_sha256']
    tasks = []
    for budget_name, arm_name, raw_registry in (
            ('dense', dense, raw_dense), ('rank256', primary, arms[linear])):
        for routing, reference, choices in (
                ('fixed_native', 'native', policies), ('actual_nonlinear', arm_name, policies[:1])):
            for role, policy in choices:
                tasks.append(dict(name=f'{budget_name}.{routing}.{role}', budget=budget_name,
                    routing=routing, policy_role=role, policy=policy, registry=arms[arm_name],
                    linear_registry=raw_registry, quality_folder=str(Path(plan['output']) / reference),
                    receipt=plan['selection_receipt'],
                    expected_scope_events=finished[reference]['expected_scope_events']))
    assert len(tasks) == 6
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    save(out / 'command.json', dict(plan=str(plan_path), output=str(out), processes=args.processes,
        threads_per_process=1, wrapper_sha256=sha(__file__), completed_arms=finished,
        selection_receipt=plan['selection_receipt'], selection_receipt_sha256=sha(plan['selection_receipt']),
        cost_source_sha256=receipt['files_sha256'][str(COST_SOURCE)],
        tasks=tasks, at_utc=datetime.now(timezone.utc).isoformat()))
    audits = {}
    with ProcessPoolExecutor(max_workers=args.processes,
                             mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(worker, task) for task in tasks]
        for future in as_completed(futures):
            name, result = future.result()
            save(out / (name + '.json'), result)
            audits[name] = result
            print(json.dumps(dict(completed=name, audits=len(audits), total=6)), flush=True)
    verify_frozen(receipt)
    assert sha(plan['selection_receipt']) == read(out / 'command.json')['selection_receipt_sha256']
    same(read(Path(plan['output']) / 'execution.json')['status'], 'complete', 'Execution state changed')
    rows = []
    for name in sorted(audits):
        result = audits[name]
        for scope, values in result['selected_scopes'].items():
            rows.append(dict(audit=name, budget=result['job']['budget'], routing=result['job']['routing'],
                policy_role=result['job']['policy_role'], scope=scope,
                method_host_bytes=result['host_inventory']['total_method_host_bytes'], **values))
    with (out / 'scope_summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = dict(status='complete', plan=str(plan_path), selection_receipt=plan['selection_receipt'],
        selection_receipt_sha256=sha(plan['selection_receipt']), wrapper_sha256=sha(__file__),
        frozen_cost_source_sha256=receipt['files_sha256'][str(COST_SOURCE)],
        frozen_inputs_rechecked_before_and_after=True, all_five_arms_finished_1000=True,
        primary='fixed_native + primary_v2 + nonBOS62, for each preregistered host budget',
        sensitivity='fixed_native + sensitivity_v1; actual nonlinear routing is descriptive',
        placement_selection='Frozen VALID rankings and receipt placements only; no TEST-frequency selection',
        budget_scope=receipt['cost']['budget'], seconds=time.monotonic() - started,
        audits={name: dict(path=str(out / (name + '.json')), sha256=sha(out / (name + '.json')))
                for name in sorted(audits)}, scopes=rows,
        not_measured=['physical flash IO', 'NVMeVirt', 'TPOT', 'speedup'],
        completed_utc=datetime.now(timezone.utc).isoformat())
    save(out / 'summary.json', summary)
    print(json.dumps(dict(status='complete', output=str(out), audits=len(audits),
                          seconds=summary['seconds'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', default=str(DATA / 'final1000-plan.json'))
    parser.add_argument('--output', required=True, help='New output directory; must not exist')
    parser.add_argument('--processes', type=int, default=4, choices=range(1, 5))
    run(parser.parse_args())
