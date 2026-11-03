#!/usr/bin/env python3
"""Train-only no-CR workload bounds and fixed, one-copy row-shard diagnostics.

Does not open test routes, run models, modify existing plans, or invoke a VM.
The row experiment uses full output rows and ordinary four-plane streams.
Gate/up and down retain the existing host barrier and vector traffic model.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from math import ceil
from pathlib import Path
import random
import time

import numpy as np

import aif_model as flash
import joint_plane_search as joint
from run_corpus_traces import DATA, MODELS, ROOT, geometry, sha256

PAGE = 16384
WAVE = PAGE * 4
COMPONENTS = ('flash_ns', 'control_ns', 'host_ns', 'input_ns', 'output_ns')


def ceildiv(a, b):
    return (a + b - 1) // b


def row_template(rows, columns):
    """Smallest balanced integer wave budget that holds all complete rows.

    This is a restricted template construction, not a proof of optimality over
    all 16-chip partitions. Actual rows may cross individual page boundaries.
    """
    for total in range(ceildiv(rows * columns, WAVE), rows * 16 + 1):
        waves = np.full(16, total // 16, dtype=np.int64)
        waves[:total % 16] += 1
        capacity = waves * WAVE // columns
        minimum = np.where(waves > 0, ((waves - 1) * WAVE) // columns + 1, 0)
        if capacity.sum() < rows or minimum.sum() > rows:
            continue
        counts = minimum.copy()
        remaining = rows - int(counts.sum())
        # Fill the available shard capacity round-robin to avoid an accidental
        # large result-byte imbalance unrelated to wave allocation.
        while remaining:
            eligible = np.flatnonzero(counts < capacity)
            take = eligible[:remaining]
            counts[take] += 1
            remaining -= len(take)
        assert counts.sum() == rows
        assert np.array_equal(ceildiv(counts * columns, WAVE), waves)
        return counts
    raise ValueError('No whole-row template')


def phase_tables(rows_by_expert_chip, columns, common):
    rows = np.asarray(rows_by_expert_chip, dtype=np.int64)
    pages = ceildiv(rows * columns, PAGE)
    waves = ceildiv(pages, 4)
    cold = ceildiv(waves, 192)
    return dict(rows=rows, pages=pages, waves=waves, cold=cold,
                inputs=(rows > 0).astype(np.int64) * columns,
                common=common, columns=columns)


def costs_aggregated(tables, work):
    waves, pages, cold, inputs, outputs = work
    active = work[-1].shape[0]  # Only the first dimension; expert K is separate.
    channel_input = inputs[:, :8] + inputs[:, 8:]
    channel_output = outputs[:, :8] + outputs[:, 8:]
    host_input = tables['columns'] * (1 if tables['common'] else tables['active'])
    result = dict(
        flash_ns=np.maximum(28000 * waves.max(axis=1), 2560 * pages.max(axis=1)),
        control_ns=250 * cold.sum(axis=1),
        host_ns=np.full(active, 2000, dtype=np.int64),
        input_ns=np.maximum(ceildiv(host_input, 8), ceildiv(channel_input.max(axis=1), 2)),
        output_ns=np.maximum(ceildiv(outputs.sum(axis=1), 8), ceildiv(channel_output.max(axis=1), 2)))
    result['model_ns'] = sum(result.values())
    return result


def aggregate(unique, tables):
    waves = tables['waves'][unique].sum(axis=1)
    pages = tables['pages'][unique].sum(axis=1)
    cold = tables['cold'][unique].sum(axis=1)
    if tables['common']:
        inputs = (tables['inputs'][unique] > 0).any(axis=1) * tables['columns']
    else:
        inputs = tables['inputs'][unique].sum(axis=1)
    outputs = tables['rows'][unique].sum(axis=1) * 4
    return [waves, pages, cold, inputs, outputs]


def summarize_arrays(values, weights):
    return {k: float(v @ weights / weights.sum() / 1e6) for k, v in values.items()}


def scalar_reference(selected, rows, columns, common):
    """Independent small event check, deriving byte work directly from rows."""
    wave_work, page_work, commands, inputs, outputs = [], [], [], [], []
    for chip in range(16):
        counts = [int(rows[expert, chip]) for expert in selected]
        pages = [ceil(count * columns / 16384) for count in counts]
        waves = [ceil(count / 4) for count in pages]
        wave_work.append(sum(waves))
        page_work.append(sum(pages))
        commands.append(sum(ceil(count / 192) for count in waves))
        inputs.append(columns * (int(any(counts)) if common else sum(count > 0 for count in counts)))
        outputs.append(sum(counts) * 4)
    return dict(flash_ns=max(max(wave_work) * 28000, max(page_work) * 2560),
                control_ns=sum(commands) * 250, host_ns=2000,
                input_ns=max(ceil(columns * (1 if common else len(selected)) / 8),
                             max(ceil((inputs[c] + inputs[c+8]) / 2) for c in range(8))),
                output_ns=max(ceil(sum(outputs) / 8),
                              max(ceil((outputs[c] + outputs[c+8]) / 2) for c in range(8))))


def optimize_offsets(unique, weights, experts, template, columns, common, trials, seed):
    """Swap expert offsets, preserving per-chip stored row/page/wave capacity.

    The sixteen allowed placements are cyclic rotations of one row template.
    Label multiplicity is immutable, so swapping labels preserves every chip's
    complete stored-weight inventory; runtime selection never moves weights.
    """
    offsets = np.arange(experts) % 16
    variants = np.array([np.roll(template, int(o)) for o in range(16)])
    tables = phase_tables(variants[offsets], columns, common)
    tables['active'] = unique.shape[1]
    initial_capacity = {k: v.sum(axis=0).copy() for k, v in tables.items()
                        if isinstance(v, np.ndarray)}
    work = aggregate(unique, tables)
    values = costs_aggregated(tables, work)
    current = int(values['model_ns'] @ weights)
    initial = current
    incidence = np.zeros((len(unique), experts), dtype=np.int8)
    incidence[np.arange(len(unique))[:, None], unique] = 1
    accepted = 0
    rng = random.Random(seed)
    mutable = len(np.unique(variants, axis=0)) > 1
    for _ in range(trials if mutable else 0):
        a, b = rng.sample(range(experts), 2)
        if offsets[a] == offsets[b]:
            continue
        difference = incidence[:, b] - incidence[:, a]
        affected = np.flatnonzero(difference)
        delta = difference[affected, None]
        proposed = []
        for key, old in zip(('waves', 'pages', 'cold', 'inputs', 'rows'), work):
            if key == 'inputs' and common:
                # All gate_up templates in this study occupy every chip.
                assert np.all(tables['inputs'] > 0)
                proposed.append(old[affected].copy())
            else:
                change = tables[key][a] - tables[key][b]
                proposed.append(old[affected] + delta * change * (4 if key == 'rows' else 1))
        new_values = costs_aggregated(tables, proposed)
        candidate = current + int((new_values['model_ns'] - values['model_ns'][affected]) @ weights[affected])
        if candidate < current:
            current = candidate
            accepted += 1
            offsets[a], offsets[b] = offsets[b], offsets[a]
            for key in ('rows', 'pages', 'waves', 'cold', 'inputs'):
                tables[key][[a, b]] = tables[key][[b, a]]
            for old, new in zip(work, proposed):
                old[affected] = new
            for key in values:
                values[key][affected] = new_values[key]
    assert current <= initial
    assert np.all(tables['rows'].sum(axis=1) == template.sum())
    for key, capacity in initial_capacity.items():
        np.testing.assert_array_equal(tables[key].sum(axis=0), capacity)
    recomputed = aggregate(unique, tables)
    for a, b in zip(work, recomputed):
        np.testing.assert_array_equal(a, b)
    check = costs_aggregated(tables, recomputed)
    for key in values:
        np.testing.assert_array_equal(values[key], check[key])
    scalar_cases = sorted({0, len(unique)//2, len(unique)-1})
    for index in scalar_cases:
        scalar = scalar_reference(unique[index], tables['rows'], columns, common)
        for component in COMPONENTS:
            assert int(values[component][index]) == scalar[component]
    # Pack immutable streams consecutively in same-WL four-plane stripes;
    # move to a fresh block when a complete stream will not fit the current one.
    reserved_waves = []
    boundary = []
    for chip in range(16):
        cursor = pad = 0
        for w in tables['waves'][:, chip]:
            assert w <= 192
            if cursor % 192 + w > 192:
                pad += 192 - cursor % 192
                cursor += 192 - cursor % 192
            cursor += int(w)
        reserved_waves.append(int(ceildiv(cursor, 192) * 192))
        boundary.append(int(pad))
    peak_mean = float(np.average(work[0].max(axis=1), weights=weights))
    average_work = float(np.average(work[0].sum(axis=1) / 16, weights=weights))
    return dict(components_ms=summarize_arrays(values, weights),
                template_rows=template.tolist(), template_waves=ceildiv(template * columns, WAVE).tolist(),
                offsets=offsets.tolist(), accepted=accepted, proposals=trials if mutable else 0,
                initial_model_ms=float(initial / weights.sum() / 1e6),
                mean_critical_chip_waves=peak_mean, mean_chip_waves=average_work,
                mean_peak_over_mean_chip_work=peak_mean / average_work,
                mean_physical_pages=float(np.average(work[1].sum(axis=1), weights=weights)),
                commands_per_event=float(np.average(work[2].sum(axis=1), weights=weights)),
                useful_rows_per_chip=tables['rows'].sum(axis=0).tolist(),
                stored_pages_per_chip=tables['pages'].sum(axis=0).tolist(),
                stored_waves_per_chip=tables['waves'].sum(axis=0).tolist(),
                reserved_block_waves_per_chip=reserved_waves, block_boundary_padding_waves_per_chip=boundary,
                rows_once=True, offsets_fixed_after_training=True, capacity_preserved_by_swaps=True,
                independent_scalar_cases=len(scalar_cases))


def analyze_model(name, args, corpus_ids):
    start = time.time()
    metadata_path = DATA / 'models' / (name + '-metadata.json')
    metadata = json.loads(metadata_path.read_text())
    layers, experts, active = geometry(metadata)
    dimensions = flash.tensor_dimensions(metadata)
    frozen_path = ROOT / 'results/plane-wave-v2/no-cr' / (name + '.frozen_plan.json')
    frozen = json.loads(frozen_path.read_text())
    cache_path = Path(frozen['training_cache']['path'])
    assert cache_path.name == 'train_routes.npz'
    assert sha256(cache_path) == frozen['training_cache']['sha256']
    with np.load(cache_path) as cached:
        routes = cached['routes']
        ids = cached['prompt_ids'].tolist()
        assert cached['layers'].tolist() == layers
    assert ids == frozen['training_prompt_ids']
    assert set(ids) == corpus_ids and all(x.startswith('train_') for x in ids)
    assert routes.shape == (200, 128, len(layers), active)
    np.testing.assert_array_equal(np.sort(routes, axis=-1)[..., 1:] > np.sort(routes, axis=-1)[..., :-1], True)
    bounds = dict(baseline_flash_ms=0., ideal_pooled_nand_ms=0., integer_epoch_pooled_nand_ms=0.,
                  ideal_pooled_ecc_ms=0., baseline_commands_ms=0., baseline_vectors_ms=0., host_ms=0.)
    layer_rows = []
    phase_results = []
    group_total = dict(group_only_learned=Counter(), joint_group_plane=Counter())
    layerwise_means = Counter()
    baseline_components = Counter()
    modelwide_recomputed = Counter()
    for li, layer in enumerate(layers):
        unique, weights = np.unique(np.sort(routes[:, :, li].reshape(-1, active), axis=1),
                                    axis=0, return_counts=True)
        base_profiles = joint.phase_profiles(dimensions[layer], 16, 4, use_cr=False)
        base_counts = joint.counts_for(unique, np.arange(experts), 16, 4)
        base_values = joint.costs_from_counts(base_counts, base_profiles, 16, 4, active)
        base_ms = summarize_arrays({k: base_values[k] for k in (*COMPONENTS, 'model_ns')}, weights)
        baseline_components.update(base_ms)
        bounds['baseline_flash_ms'] += base_ms['flash_ns']
        bounds['baseline_commands_ms'] += base_ms['control_ns']
        bounds['baseline_vectors_ms'] += base_ms['input_ns'] + base_ms['output_ns']
        bounds['host_ms'] += base_ms['host_ns']
        for p in base_profiles:
            full_rows = int(p['rows'].sum())
            pages = ceildiv(full_rows * p['columns'], PAGE)
            useful_bytes = full_rows * p['columns'] * active
            bounds['ideal_pooled_nand_ms'] += active * pages / 64 * 28000 / 1e6
            bounds['integer_epoch_pooled_nand_ms'] += ceildiv(active * pages, 64) * 28000 / 1e6
            bounds['ideal_pooled_ecc_ms'] += useful_bytes / (16 * 6.4e9) * 1e3
        result = dict(model=name, layer=int(layer), baseline_ms=base_ms['model_ns'],
                      baseline_flash_ms=base_ms['flash_ns'], candidates=[])
        for family in ('group_only_learned', 'joint_group_plane'):
            candidates = []
            for cell in frozen['training_candidates']:
                if cell['family'] != family:
                    continue
                g, width = cell['group_chips'], cell['plane_width']
                counts = joint.counts_for(unique, frozen['mappings'][cell['key']][str(layer)], g, width)
                ns = joint.model_ns_from_counts(counts, joint.phase_profiles(dimensions[layer], g, width, False),
                                                g, width, active)
                ms = float(ns @ weights / weights.sum() / 1e6)
                candidates.append((ms, g, width, cell['key']))
                if cell['key'] == frozen['winners'][family]['key']:
                    modelwide_recomputed[family] += ms
            winner = min(candidates)
            layerwise_means[family] += winner[0]
            group_total[family][f'g{winner[1]}w{winner[2]}'] += 1
            result[family + '_best_ms'] = winner[0]
            result[family + '_best_g'] = winner[1]
            result[family + '_best_width'] = winner[2]
            result['candidates'].extend(dict(family=family, model_ms=x[0], g=x[1], width=x[2], key=x[3]) for x in candidates)
        phase_entry = dict(layer=int(layer))
        # Independent fixed row shards for each sequential projection phase.
        for p in base_profiles:
            template = row_template(int(p['rows'].sum()), p['columns'])
            optimized = optimize_offsets(unique, weights, experts, template, p['columns'], p['common'],
                                         args.trials, args.seed + layer * 131 + (0 if p['common'] else 7919))
            optimized['baseline_pages_per_expert'] = int(p['pages'].sum())
            optimized['baseline_waves_per_expert'] = int(p['waves'].sum())
            optimized['stored_pages_inflation_vs_baseline'] = sum(optimized['stored_pages_per_chip']) / (experts * int(p['pages'].sum())) - 1
            optimized['stored_wave_reservations_change_vs_baseline'] = sum(optimized['stored_waves_per_chip']) / (experts * int(p['waves'].sum())) - 1
            phase_entry[p['name']] = optimized
            result[p['name'] + '_fractional_ms'] = optimized['components_ms']['model_ns']
        result['fractional_ms'] = result['gate_up_fractional_ms'] + result['down_fractional_ms']
        layer_rows.append(result)
        phase_results.append(phase_entry)
        if (li + 1) % 8 == 0:
            print(f'{name}: {li+1}/{len(layers)} layers complete', flush=True)
    fractional = Counter()
    fractional_initial_ms = 0.
    for phase_entry in phase_results:
        for p in ('gate_up', 'down'):
            fractional.update(phase_entry[p]['components_ms'])
            fractional_initial_ms += phase_entry[p]['initial_model_ms']
    winner = frozen['winners']['joint_group_plane']
    group_winner = frozen['winners']['group_only_learned']
    baseline_ms = baseline_components['model_ns']
    for component in COMPONENTS:
        assert abs(baseline_components[component] - frozen['baseline_training']['components_ms'][component]) < 1e-8
    for family, recomputed in modelwide_recomputed.items():
        assert abs(recomputed - frozen['winners'][family]['mean_ms_per_token']) < 1e-8
    result = dict(model=name, layers=len(layers), experts=experts, active=active,
                  train_prompts=200, tokens_per_prompt=128,
                  provenance=dict(metadata_path=str(metadata_path), metadata_sha256=sha256(metadata_path),
                                  frozen_plan_path=str(frozen_path), frozen_plan_sha256=sha256(frozen_path),
                                  train_cache_path=str(cache_path), train_cache_sha256=sha256(cache_path),
                                  prompt_ids=ids),
                  baseline_components_ms=dict(baseline_components), workload_bounds=bounds,
                  pooled_nand_gap_fraction=1-bounds['ideal_pooled_nand_ms']/bounds['baseline_flash_ms'],
                  command_fraction_of_baseline=baseline_components['control_ns']/baseline_ms,
                  total_nonflash_fraction_of_baseline=1-bounds['baseline_flash_ms']/baseline_ms,
                  modelwide_group_ms=group_winner['mean_ms_per_token'],
                  modelwide_group_and_plane_ms=winner['mean_ms_per_token'],
                  layerwise_group_ms=layerwise_means['group_only_learned'],
                  layerwise_group_and_plane_ms=layerwise_means['joint_group_plane'],
                  layerwise_configs={k:dict(v) for k,v in group_total.items()},
                  fractional_components_ms=dict(fractional), fractional_initial_ms=fractional_initial_ms,
                  fractional_improvement_vs_baseline=1-fractional['model_ns']/baseline_ms,
                  fractional_improvement_vs_modelwide_best=1-fractional['model_ns']/winner['mean_ms_per_token'],
                  row_shard_plans=phase_results, layer_results=layer_rows,
                  checks=dict(baseline_components_match_existing_frozen_plan=True,
                              modelwide_candidate_matches_existing_frozen_plan=True,
                              independent_scalar_cases=sum(v[p]['independent_scalar_cases'] for v in phase_results for p in ('gate_up','down'))),
                  seconds=time.time()-start)
    print(json.dumps({k:result[k] for k in ('model','pooled_nand_gap_fraction','command_fraction_of_baseline',
                                           'modelwide_group_and_plane_ms','layerwise_group_and_plane_ms',
                                           'fractional_components_ms','fractional_improvement_vs_baseline','seconds')}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    parser.add_argument('--trials', type=int, default=256)
    parser.add_argument('--seed', type=int, default=20261102)
    parser.add_argument('--output', type=Path, default=ROOT/'results/research-2026-11-02')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    corpus_path = DATA/'runs/corpus-300-v1/corpus/prompts.jsonl'
    corpus = [json.loads(x) for x in corpus_path.read_text().splitlines()]
    train_ids = {r['id'] for r in corpus if r['split'] == 'train'}
    assert len(train_ids) == 200
    results = []
    for name in args.models:
        results.append(analyze_model(name, args, train_ids))
        document = dict(study='2026-11-02 train-only no-CR diagnostics',
                        source_sha256=sha256(Path(__file__)), corpus_sha256=sha256(corpus_path),
                        dependency_source_sha256={s:sha256(ROOT/'scripts'/s) for s in
                                                  ('aif_model.py','joint_plane_search.py','run_plane_wave_study.py','run_corpus_traces.py')},
                        seed=args.seed, proposals_per_layer_phase=args.trials,
                        formulas=dict(
                            pooled_nand='sum_over_layers_and_phases K*ceil(R*C/16384)/64*28000 ns; fractional work bound',
                            integer_pooled='sum ceil(K*ceil(R*C/16384)/64)*28000 ns; synchronous epoch oracle',
                            fixed_phase='max_c(max(28000*sum_selected ceil(rows_e,c*C/65536),2560*sum_selected ceil(rows_e,c*C/16384))) + 250*sum_e,c ceil(waves_e,c/192) + 2000 + max(host_input/8,max_channel_input/2) + max(host_output/8,max_channel_output/2), rounded up',
                            dependency='T_layer=T_gate_up+T_down; gate_up results go to host, activation forms each expert down input, then fanout to its immutable shards; activation/combine compute unmeasured as in existing model'),
                        limitations=['Train-only exploratory results are not held-out gains.',
                                     'Pooled NAND oracle relaxes immutable placement, plane addresses and per-expert output rows; it is an opportunity ceiling, not an implementation.',
                                     'The integer pooled oracle assumes synchronized 28us epochs; asynchronous service can relax this.',
                                     'Row search uses sixteen cyclic rotations per phase, fixed offset multiplicities, 256 default swaps. It does not search every overlapping chip subset.',
                                     'No weight copies, dynamic weight relocation, cross-expert multi-context plane read, or cr-read.',
                                     'Existing ideal NAND/ECC overlap and uncalibrated command/link constants retained.',
                                     'Gate/up and down may use different immutable row maps; host row-reassembly/activation and final expert combine compute time are unmeasured.',
                                     'Block packing reports per-layer per-phase reserved block waves; no raw NAND capacity budget or physical device calibration claimed.'],
                        test_routes_opened=False, models=results)
        (args.output/'diagnostics.json').write_text(json.dumps(document, indent=2)+'\n')
    columns = ['model','baseline_ms','baseline_flash_ms','pooled_nand_ms','integer_pooled_nand_ms','nand_gap_pct',
               'command_pct','vector_ms','modelwide_best_ms','layerwise_group_ms','layerwise_joint_ms',
               'fractional_initial_ms','fractional_optimized_ms','fractional_flash_ms','fractional_command_ms',
               'fractional_vector_ms','fractional_improvement_pct']
    with (args.output/'diagnostics.csv').open('w') as stream:
        writer=csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for r in results:
            b=r['baseline_components_ms']; w=r['workload_bounds']; f=r['fractional_components_ms']
            writer.writerow(dict(model=r['model'],baseline_ms=b['model_ns'],baseline_flash_ms=b['flash_ns'],
                                 pooled_nand_ms=w['ideal_pooled_nand_ms'],integer_pooled_nand_ms=w['integer_epoch_pooled_nand_ms'],
                                 nand_gap_pct=100*r['pooled_nand_gap_fraction'],command_pct=100*r['command_fraction_of_baseline'],
                                 vector_ms=b['input_ns']+b['output_ns'],modelwide_best_ms=r['modelwide_group_and_plane_ms'],
                                 layerwise_group_ms=r['layerwise_group_ms'],layerwise_joint_ms=r['layerwise_group_and_plane_ms'],
                                 fractional_initial_ms=r['fractional_initial_ms'],fractional_optimized_ms=f['model_ns'],
                                 fractional_flash_ms=f['flash_ns'],fractional_command_ms=f['control_ns'],
                                 fractional_vector_ms=f['input_ns']+f['output_ns'],
                                 fractional_improvement_pct=100*r['fractional_improvement_vs_baseline']))


if __name__ == '__main__':
    main()
