#!/usr/bin/env python3
"""Fixed, single-context gate/up supermatrices with explicit inactive overfetch.

Only common-input gate/up shards are concatenated. Down stays expert-specific.
Every activated bundle is read in full, with all outputs transferred. Fixed chip
groups come from the previous training-only experiment; no weights are replicated.
"""
import argparse
import csv
import json
from pathlib import Path
import random
import time

import numpy as np

import aif_model as flash
import corpus_analysis as corpus
from run_corpus_traces import DATA, MODELS, ROOT, geometry, sha256, write_json

COMPONENTS = corpus.COMPONENTS
EXTRA = ('physical_pages', 'read_waves', 'cold_runs', 'selected_useful_bytes',
         'read_weight_bytes', 'gate_returned_experts', 'gate_selected_experts')


def fixed_bundles(routes, placement, group_chips, size, trials=64, seed=20261030):
    """Greedy coactivation clustering, followed by capacity-preserving swaps.

    Objective: training-frequency-weighted number of touched bundles. Selecting
    the bundle size and g subsequently uses full flash/control/vector latency.
    """
    placement = np.asarray(placement, dtype=int)
    expert_count = len(placement)
    groups = 16 // group_chips
    flat = routes.reshape(-1, routes.shape[-1]).astype(int)
    pairs = flat[:, :, None] * expert_count + flat[:, None, :]
    affinity = np.bincount(pairs.ravel(), minlength=expert_count ** 2).reshape(expert_count, expert_count)
    labels = np.full(expert_count, -1, dtype=int)
    members = []
    group_members = []
    for group in range(groups):
        experts = sorted(np.flatnonzero(placement % groups == group).tolist(), key=lambda e: int(placement[e]))
        group_members.append(experts)
        left = set(experts)
        while left:
            seed_expert = max(left, key=lambda e: (int(affinity[e, e]), -int(placement[e])))
            current = [seed_expert]
            left.remove(seed_expert)
            while left and len(current) < size:
                chosen = max(left, key=lambda e: (sum(int(affinity[e, x]) for x in current), -int(placement[e])))
                current.append(chosen)
                left.remove(chosen)
            label = len(members)
            for expert in current:
                labels[expert] = label
            members.append(current)
    unique, weights = np.unique(np.sort(flat, axis=1), axis=0, return_counts=True)
    incidence = np.zeros((len(unique), expert_count), dtype=np.int8)
    incidence[np.arange(len(unique))[:, None], unique] = 1
    counts = np.stack([incidence[:, member].sum(axis=1) for member in members], axis=1)
    initial = score = int((counts > 0).sum(axis=1) @ weights)
    rng = random.Random(seed)
    accepted = 0
    eligible = [group for group in group_members if len({int(labels[e]) for e in group}) > 1]
    for _ in range(trials if size > 1 and eligible else 0):
        a, b = rng.sample(rng.choice(eligible), 2)
        la, lb = int(labels[a]), int(labels[b])
        if la == lb:
            continue
        delta = incidence[:, b] - incidence[:, a]
        before = ((counts[:, la] > 0).astype(int) + (counts[:, lb] > 0)) @ weights
        ca, cb = counts[:, la] + delta, counts[:, lb] - delta
        after = ((ca > 0).astype(int) + (cb > 0)) @ weights
        if after < before:
            counts[:, la], counts[:, lb] = ca, cb
            labels[a], labels[b] = lb, la
            score += int(after - before)
            accepted += 1
    assert np.all(labels >= 0)
    final_members = [np.flatnonzero(labels == b).tolist() for b in range(len(members))]
    assert sorted(e for bundle in final_members for e in bundle) == list(range(expert_count))
    assert all(len({int(placement[e] % groups) for e in bundle}) == 1 for bundle in final_members)
    assert sorted(map(len, final_members)) == sorted(map(len, members))
    return labels.tolist(), dict(bundle_size=size, bundles=len(members), trials=trials,
                                 accepted=accepted, initial_touched_bundles=initial,
                                 final_touched_bundles=score, seed=seed)


def layer_costs(routes, placement, group_chips, dimensions, labels):
    """Exact vectorization of full-bundle, four-plane serialized reads."""
    shape = routes.shape[:-1]
    flat = routes.reshape(-1, routes.shape[-1])
    selected = flat.shape[-1]
    placement = np.asarray(placement, dtype=int)
    labels = np.asarray(labels, dtype=int)
    groups = 16 // group_chips
    number = int(labels.max()) + 1
    sizes = np.bincount(labels, minlength=number)
    bundle_groups = np.empty(number, dtype=int)
    for bundle in range(number):
        members = np.flatnonzero(labels == bundle)
        owner = placement[members] % groups
        if len(members) == 0 or not np.all(owner == owner[0]):
            raise ValueError('Every nonempty bundle must remain within one fixed chip group')
        bundle_groups[bundle] = owner[0]
    selected_bundles = labels[flat]
    occupied = np.stack([(selected_bundles == b).any(axis=1) for b in range(number)], axis=1)
    gate_rows = dimensions['gate'][0] + dimensions['up'][0]
    columns = dimensions['up'][1]
    down_profile = corpus.profiles(dimensions, group_chips, True)[1]
    costs = corpus.layer_costs(flat, placement, group_chips, [down_profile])
    costs = {key: value.copy() for key, value in costs.items()}
    conservative = {key: value.copy() for key, value in costs.items()}
    # Drain gaps are not proven to preserve CR state. The conservative schedule
    # therefore uses ordinary 28us reads for EVERY wave, followed by ECC drain.
    down_rows, down_columns = dimensions['down']
    down_counts = corpus.counts_for(flat, placement, groups)
    down_nand = []
    down_ecc = []
    down_pages_sum = np.zeros(len(flat), dtype=np.int64)
    down_waves_sum = np.zeros(len(flat), dtype=np.int64)
    down_cold_sum = np.zeros(len(flat), dtype=np.int64)
    for shard in range(group_chips):
        count = down_rows // group_chips + int(shard < down_rows % group_chips)
        pages = flash.ceildiv(count * down_columns, flash.PAGE)
        waves = flash.ceildiv(pages, 4)
        cold = flash.ceildiv(waves, 192)
        nand = waves * flash.CR_NS + cold * (flash.COLD_NS - flash.CR_NS)
        down_nand.append(waves * flash.COLD_NS)
        down_ecc.append(pages * 2560)
        total_selected = down_counts.sum(axis=1)
        down_pages_sum += total_selected * pages
        down_waves_sum += total_selected * waves
        down_cold_sum += total_selected * cold
    conservative['flash_ns'] = (down_counts[:, :, None] * (np.asarray(down_nand) + down_ecc)[None, None, :]).max(axis=(1, 2))
    in_chips = np.zeros((len(flat), 16), dtype=np.int64)
    out_chips = np.zeros_like(in_chips)
    gate_ideal = np.zeros(len(flat), dtype=np.int64)
    gate_conservative = np.zeros_like(gate_ideal)
    command_count = np.zeros_like(gate_ideal)
    pages_sum = np.zeros_like(gate_ideal)
    waves_sum = np.zeros_like(gate_ideal)
    cold_sum = np.zeros_like(gate_ideal)
    returned_experts = occupied.astype(np.int64) @ sizes
    for group in range(groups):
        indices = np.flatnonzero(bundle_groups == group)
        if not len(indices):
            continue
        active = occupied[:, indices].astype(np.int64)
        for shard in range(group_chips):
            count = gate_rows // group_chips + int(shard < gate_rows % group_chips)
            if count == 0:
                continue
            chip = group * group_chips + shard
            nbytes = sizes[indices] * count * columns
            pages = (nbytes + flash.PAGE - 1) // flash.PAGE
            waves = (pages + 3) // 4
            cold = (waves + 191) // 192
            nand = waves * flash.CR_NS + cold * (flash.COLD_NS - flash.CR_NS)
            page_count = active @ pages
            nand_ns = active @ nand
            ecc_ns = page_count * 2560
            gate_ideal = np.maximum(gate_ideal, np.maximum(nand_ns, ecc_ns))
            ordinary_ns = (active @ waves) * flash.COLD_NS
            gate_conservative = np.maximum(gate_conservative, ordinary_ns + ecc_ns)
            commands = active @ cold
            command_count += commands
            pages_sum += page_count
            waves_sum += active @ waves
            cold_sum += commands
            in_chips[:, chip] = active.any(axis=1) * columns
            out_chips[:, chip] = (active @ sizes[indices]) * count * 4
    channel_in = in_chips[:, :8] + in_chips[:, 8:]
    channel_out = out_chips[:, :8] + out_chips[:, 8:]
    gate_components = dict(flash_ns=gate_ideal, control_ns=command_count * 250,
                           host_ns=np.full(len(flat), 2000, dtype=np.int64),
                           input_ns=np.maximum(flash.ceildiv(columns, 8), (channel_in.max(axis=1) + 1) // 2),
                           output_ns=np.maximum((out_chips.sum(axis=1) + 7) // 8, (channel_out.max(axis=1) + 1) // 2))
    for key, value in gate_components.items():
        costs[key] += value
        conservative[key] += gate_conservative if key == 'flash_ns' else value
    costs['model_ns'] = sum(costs.values())
    conservative['model_ns'] = sum(conservative.values())
    selected_bytes = selected * (gate_rows * columns + down_rows * down_columns)
    metrics = dict(physical_pages=pages_sum + down_pages_sum,
                   read_waves=waves_sum + down_waves_sum,
                   cold_runs=cold_sum + down_cold_sum,
                   selected_useful_bytes=np.full(len(flat), selected_bytes, dtype=np.int64),
                   read_weight_bytes=returned_experts * gate_rows * columns + selected * down_rows * down_columns,
                   gate_returned_experts=returned_experts,
                   gate_selected_experts=np.full(len(flat), selected, dtype=np.int64))
    reshape = lambda values: {key: value.reshape(shape) for key, value in values.items()}
    return reshape(costs), reshape(conservative), reshape(metrics)


def evaluate(routes, layers, dimensions, placement_by_layer, labels_by_layer, group_chips):
    costs = {key: np.zeros(routes.shape[:2], dtype=np.int64) for key in (*COMPONENTS, 'model_ns')}
    conservative = {key: np.zeros(routes.shape[:2], dtype=np.int64) for key in (*COMPONENTS, 'model_ns')}
    metrics = {key: np.zeros(routes.shape[:2], dtype=np.int64) for key in EXTRA}
    for li, layer in enumerate(layers):
        c, upper, m = layer_costs(routes[:, :, li], placement_by_layer[str(layer)], group_chips,
                                  dimensions[layer], labels_by_layer[str(layer)])
        for key in costs:
            costs[key] += c[key]
            conservative[key] += upper[key]
        for key in metrics:
            metrics[key] += m[key]
    return costs, conservative, metrics


def describe(costs, conservative, metrics, window, baseline=None, baseline_upper=None):
    average = lambda value: float(value[:, :window].mean())
    total = lambda value: int(value[:, :window].sum())
    useful = total(metrics['selected_useful_bytes'])
    pages = total(metrics['physical_pages'])
    waves = total(metrics['read_waves'])
    result = dict(ms_per_token=average(costs['model_ns']) / 1e6,
                  conservative_ms_per_token=average(conservative['model_ns']) / 1e6,
                  components_ms={key: average(costs[key]) / 1e6 for key in COMPONENTS},
                  conservative_components_ms={key: average(conservative[key]) / 1e6 for key in COMPONENTS},
                  useful_slot_fill=useful / (waves * 4 * flash.PAGE),
                  occupied_slot_fill=pages / (waves * 4),
                  selected_bytes_per_token=useful / costs['model_ns'][:, :window].size,
                  read_amplification=total(metrics['read_weight_bytes']) / useful,
                  physical_page_read_amplification=pages * flash.PAGE / useful,
                  gate_expert_amplification=total(metrics['gate_returned_experts']) / total(metrics['gate_selected_experts']),
                  physical_pages_per_token=average(metrics['physical_pages']),
                  cold_runs_per_token=average(metrics['cold_runs']))
    if baseline is not None:
        base = baseline['model_ns'][:, :window].mean(axis=1)
        value = costs['model_ns'][:, :window].mean(axis=1)
        result.update(improvement=1 - value.mean() / base.mean(),
                      improvement_ci95=corpus.paired_interval(base, value, seed=20261030),
                      fraction_prompts_improved=float((value < base).mean()))
    if baseline_upper is not None:
        base = baseline_upper['model_ns'][:, :window].mean(axis=1)
        value = conservative['model_ns'][:, :window].mean(axis=1)
        result.update(conservative_improvement=1 - value.mean() / base.mean(),
                      conservative_improvement_ci95=corpus.paired_interval(base, value, seed=20261030))
    return result


def capacity(dimensions, mappings, labels, group_chips):
    useful = physical = reserved = cold = 0
    for layer, dims in dimensions.items():
        placement = np.asarray(mappings[str(layer)])
        bundle_labels = np.asarray(labels[str(layer)])
        sizes = np.bincount(bundle_labels)
        for projection, rows, columns in (('gate_up', dims['gate'][0] + dims['up'][0], dims['up'][1]),
                                          ('down', *dims['down'])):
            counts = sizes if projection == 'gate_up' else np.ones(len(placement), dtype=np.int64)
            for shard in range(group_chips):
                row_count = rows // group_chips + int(shard < rows % group_chips)
                nbytes = counts * row_count * columns
                pages = (nbytes + flash.PAGE - 1) // flash.PAGE
                waves = (pages + 3) // 4
                useful += int(nbytes.sum())
                physical += int(pages.sum()) * flash.PAGE
                reserved += int(waves.sum()) * 4 * flash.PAGE
                cold += int(((waves + 191) // 192).sum())
    return dict(unique_weight_bytes=useful, physical_page_bytes=physical,
                allocated_wave_bytes=reserved, cold_runs_if_all_requested=cold,
                weights_replicated=False,
                capacity_scope='Sum of allocated stream waves; block-end/placement-boundary padding excluded, no NAND program executed.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=DATA / 'runs/corpus-300-v1')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results/plane-wave-v1/static-bundles')
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    parser.add_argument('--trials', type=int, default=64)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    summaries = []
    candidates_all = []
    checks = []
    per_prompt = []
    sources = {}
    for name in args.models:
        base_dir = args.run_dir / name / 'analysis'
        old_path = base_dir / 'frozen_plan.json'
        old = json.loads(old_path.read_text())
        metadata_path = DATA / 'models' / f'{name}-metadata.json'
        metadata = json.loads(metadata_path.read_text())
        dimensions = flash.tensor_dimensions(metadata)
        layers, experts, _ = geometry(metadata)
        train_path = base_dir / 'train_routes.npz'
        with np.load(train_path) as data:
            train = data['routes']
            train_ids = data['prompt_ids'].tolist()
            assert data['layers'].tolist() == layers
        assert train.shape[:2] == (200, 128)
        assert set(train_ids) == {r['id'] for r in old['training_routing']}
        labels_by_spec = {}
        stats = {}
        train_candidates = []
        for g in (1, 2, 4, 8, 16):
            mappings = old['mappings'][f'g{g}']
            for size in (1, 2, 4):
                tag = f'g{g}_b{size}'
                labels, current_stats = {}, {}
                for li, layer in enumerate(layers):
                    if size == 1:
                        labels[str(layer)] = list(range(experts))
                    else:
                        labels[str(layer)], current_stats[str(layer)] = fixed_bundles(
                            train[:, :, li], mappings[str(layer)], g, size, args.trials,
                            seed=20261030 + layer * 131 + g * 17 + size)
                labels_by_spec[tag] = labels
                stats[tag] = current_stats
                costs, upper, metrics = evaluate(train, layers, dimensions, mappings, labels, g)
                if size == 1:
                    original = corpus.evaluate_routes(train, layers, dimensions, g, True, mappings)
                    for key in (*COMPONENTS, 'model_ns'):
                        assert np.array_equal(costs[key], original[key]), (name, g, key)
                    checks.append(dict(model=name, g=g, phase='train', all_tokens_baseline_exact=True))
                record = dict(model=name, tag=tag, g=g, bundle_size=size,
                              training_ms=float(costs['model_ns'].mean() / 1e6),
                              training_conservative_ms=float(upper['model_ns'].mean() / 1e6),
                              training_read_amplification=float(metrics['read_weight_bytes'].sum() / metrics['selected_useful_bytes'].sum()))
                train_candidates.append(record)
                print(f'{name}: {tag} train={record["training_ms"]:.5f} ms amp={record["training_read_amplification"]:.3f}', flush=True)
        winner = min(train_candidates, key=lambda row: (row['training_ms'], row['bundle_size'], row['g']))
        winner_upper = min(train_candidates, key=lambda row: (row['training_conservative_ms'], row['bundle_size'], row['g']))
        old_g = old['winners']['learned']['group_chips']
        fixed_g_winner = min((r for r in train_candidates if r['g'] == old_g), key=lambda r: (r['training_ms'], r['bundle_size']))
        frozen_path = args.output_dir / f'{name}.frozen_plan.json'
        frozen = dict(model=name, date='2026-10-30', training_only=True,
                      test_routes_opened_before_freeze=False, training_ids=train_ids,
                      source_frozen_plan_sha256=sha256(old_path), training_cache_sha256=sha256(train_path),
                      winner=winner, winner_conservative=winner_upper, winner_at_old_g=fixed_g_winner,
                      candidates=train_candidates, labels=labels_by_spec, learning_stats=stats,
                      fitting_objective='Greedy coactivation then capacity-preserving swaps minimize touched bundles; full latency selects g/B.',
                      assumptions=dict(single_context=True, read_entire_active_bundle=True,
                                       transfer_all_bundle_outputs=True, down_unchanged=True,
                                       chip_major_static_row_permutation=True,
                                       extra_host_output_demux_cost_ns=0,
                                       conservative_timing='Every wave ordinary 28us read, then serialized shared ECC drain; no retained CR state assumed.',
                                       existing_control_ns=250, existing_host_phase_ns=2000))
        write_json(frozen_path, frozen)
        frozen_hash = sha256(frozen_path)
        test_path = base_dir / 'test_routes.npz'
        with np.load(test_path) as data:
            test = data['routes']
            test_ids = data['prompt_ids'].tolist()
            assert data['layers'].tolist() == layers
        assert test.shape[:2] == (100, 128) and set(test_ids).isdisjoint(train_ids)
        policies = dict(old_baseline=f'g{old_g}_b1', selected=winner['tag'],
                        selected_at_old_g=fixed_g_winner['tag'], conservative_selected=winner_upper['tag'],
                        b2_at_old_g=f'g{old_g}_b2', b4_at_old_g=f'g{old_g}_b4')
        results = {}
        for tag in sorted(set(policies.values())):
            g = int(tag.split('_')[0][1:])
            mappings = old['mappings'][f'g{g}']
            results[tag] = evaluate(test, layers, dimensions, mappings, labels_by_spec[tag], g)
        base, base_upper, _ = results[policies['old_baseline']]
        original = corpus.evaluate_routes(test, layers, dimensions, old_g, True, old['mappings'][f'g{old_g}'])
        for key in (*COMPONENTS, 'model_ns'):
            assert np.array_equal(base[key], original[key]), (name, key)
        checks.append(dict(model=name, g=old_g, phase='test', all_tokens_baseline_exact=True))
        for window in (32, 128):
            summaries.append(dict(model=name, window=window, old_g=old_g,
                policies={policy: dict(tag=tag, **describe(*results[tag], window, base, base_upper)) for policy, tag in policies.items()}))
        for i, prompt_id in enumerate(test_ids):
            for window in (32, 128):
                per_prompt.append(dict(model=name, prompt_id=prompt_id, window=window,
                    **{policy: float(results[tag][0]['model_ns'][i, :window].mean()/1e6) for policy, tag in policies.items()}))
        capacities = {}
        for policy in ('old_baseline', 'selected', 'selected_at_old_g'):
            tag = policies[policy]
            g = int(tag.split('_')[0][1:])
            capacities[policy] = capacity(dimensions, old['mappings'][f'g{g}'], labels_by_spec[tag], g)
        assert len({v['unique_weight_bytes'] for v in capacities.values()}) == 1
        write_json(args.output_dir / f'{name}.capacity.json', capacities)
        assert sha256(frozen_path) == frozen_hash
        sources[name] = dict(metadata=str(metadata_path), metadata_sha256=sha256(metadata_path),
                             old_plan=str(old_path), old_plan_sha256=sha256(old_path),
                             train_cache=str(train_path), train_cache_sha256=sha256(train_path),
                             test_cache=str(test_path), test_cache_sha256=sha256(test_path),
                             frozen_plan_sha256=frozen_hash, training_ids=train_ids, test_ids=test_ids)
        candidates_all.extend(train_candidates)
        write_json(args.output_dir / 'summary.json', summaries)
        write_json(args.output_dir / 'training_candidates.json', candidates_all)
        write_json(args.output_dir / 'validation.json', checks)
        print(f'{name}: heldout complete, winner {winner["tag"]}', flush=True)
    with (args.output_dir / 'per_prompt.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_prompt[0]))
        writer.writeheader()
        writer.writerows(per_prompt)
    write_json(args.output_dir / 'manifest.json', dict(date='2026-10-30', timezone='Asia/Taipei',
        method='Fixed gate/up supermatrices; all members read and returned; down unchanged.',
        corpus_sha256=sha256(args.run_dir / 'corpus/prompts.jsonl'), sources=sources,
        software_sha256={p:sha256(ROOT/'scripts'/p) for p in
                         ('static_wave_bundle_study.py', 'corpus_analysis.py', 'aif_model.py', 'overhead_model.py')},
        command=dict(models=args.models, trials=args.trials, run_dir=str(args.run_dir)),
        measurements='Python architectural latency model; no new NVMeVirt replay or physical NAND programming.',
        wall_seconds=time.time()-started))


if __name__ == '__main__':
    main()
