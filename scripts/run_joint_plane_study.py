#!/usr/bin/env python3
"""Frozen train/test replay for joint group/plane placement (CR retained)."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np

import aif_model as flash
import joint_plane_search as joint
import plane_wave_model as plane
import run_plane_wave_study as previous
from run_corpus_traces import DATA, MODELS, ROOT, geometry, sha256, write_json

SEED = 20261101
FAMILIES = ('modulo', 'warm_start', 'staged_equal', 'joint')


def source_hashes():
    names = ('joint_plane_search.py', 'run_joint_plane_study.py',
             'run_plane_wave_study.py', 'plane_wave_model.py',
             'aif_model.py', 'overhead_model.py', 'corpus_analysis.py')
    return {name: sha256(ROOT / 'scripts' / name) for name in names}


def weighted_metrics(unique, weights, slots, g, width, profiles):
    values = joint.costs_from_counts(joint.counts_for(unique, slots, g, width),
                                     profiles, g, width, unique.shape[-1])
    return dict(sums={k: int(v @ weights) for k, v in values.items() if k != 'phases'},
                phases={p: {k: int(v @ weights) for k, v in d.items()}
                        for p, d in values['phases'].items()})


def fit_layer(task):
    layer, dims, unique, weights, experts, g, width, warm, trials, seed, use_cr = task
    profiles = joint.phase_profiles(dims, g, width, use_cr)
    starts = (list(range(experts)), warm)
    outputs = {'modulo': starts[0], 'warm_start': warm}
    stats = []
    joint_candidates, staged_candidates = [], []
    for si, initial in enumerate(starts):
        local_seed = seed + si * 10_000_019
        fitted, jstat = joint.optimize_slots(unique, weights, experts, initial,
                                             g, width, profiles, trials, local_seed)
        if g == 16 or width == 4:
            staged, sstats = fitted, dict(degenerate_same_search=True, final=jstat)
        else:
            group_budget = trials // 2
            grouped, group_stats = joint.optimize_slots(
                unique, weights, experts, initial, g, 4,
                joint.phase_profiles(dims, g, 4, use_cr), group_budget, local_seed)
            staged, plane_stats = joint.optimize_slots(
                unique, weights, experts, grouped, g, width, profiles,
                trials - group_budget, local_seed + 1, allow_group_moves=False)
            initial_score = joint.score_slots(unique, weights, initial, g, width, profiles)
            staged_score = joint.score_slots(unique, weights, staged, g, width, profiles)
            retained_initial = staged_score > initial_score
            if retained_initial:
                staged = initial
            sstats = dict(degenerate_same_search=False, group=group_stats,
                          plane=plane_stats, retained_initial=retained_initial,
                          initial_actual_score=list(initial_score),
                          final_actual_score=list(joint.score_slots(unique, weights, staged,
                                                                   g, width, profiles)))
        jscore = joint.score_slots(unique, weights, fitted, g, width, profiles)
        sscore = joint.score_slots(unique, weights, staged, g, width, profiles)
        joint_candidates.append((jscore, si, fitted))
        staged_candidates.append((sscore, si, staged))
        stats.append(dict(start=('identity', 'v1_trained_group_and_plane')[si],
                          joint=jstat, staged=sstats))
    best_joint = min(joint_candidates, key=lambda x: (x[0], x[1]))
    best_staged = min(staged_candidates, key=lambda x: (x[0], x[1]))
    outputs['joint'], outputs['staged_equal'] = best_joint[2], best_staged[2]
    metrics = {family: weighted_metrics(unique, weights, slots, g, width, profiles)
               for family, slots in outputs.items()}
    return dict(layer=layer, g=g, width=width, mappings=outputs, metrics=metrics,
                optimization=dict(starts=stats, selected_joint_start=best_joint[1],
                                  selected_staged_start=best_staged[1]))


def summary_from_sums(sums, phases, observations):
    counts = {k: sums[k] / observations for k in joint.COUNTERS}
    return dict(mean_ms_per_token=sums['model_ns'] / observations / 1e6,
                components_ms={k: sums[k] / observations / 1e6 for k in joint.COMPONENTS},
                counters_per_token=counts,
                plane_slot_fill=counts['useful_bytes'] / (counts['read_waves'] * 4 * flash.PAGE),
                physical_page_fill=counts['useful_bytes'] / (counts['physical_pages'] * flash.PAGE),
                phase_components_ms={p: {k: v / observations / 1e6 for k, v in d.items()}
                                     for p, d in phases.items()})


def choose_winners(candidates):
    def winner(family, condition=lambda c: True):
        eligible = [c for c in candidates if c['family'] == family and condition(c)]
        return min(eligible, key=lambda c: (c['mean_ms_per_token'], -c['plane_width'],
                                           c['group_chips']))
    return dict(baseline=winner('modulo', lambda c: c['group_chips'] == 16 and c['plane_width'] == 4),
                modulo=winner('modulo'),
                group_only=winner('warm_start', lambda c: c['plane_width'] == 4),
                plane_only=winner('joint', lambda c: c['group_chips'] == 16 and c['plane_width'] < 4),
                historical_staged=winner('warm_start'),
                historical_forced_plane=winner('warm_start', lambda c: c['plane_width'] < 4),
                staged_equal=winner('staged_equal'),
                staged_equal_forced_plane=winner('staged_equal', lambda c: c['plane_width'] < 4),
                joint=winner('joint'),
                joint_forced_plane=winner('joint', lambda c: c['plane_width'] < 4),
                joint_width4=winner('joint', lambda c: c['plane_width'] == 4),
                **{f'joint_width{w}': winner('joint', lambda c, w=w: c['plane_width'] == w)
                   for w in (1, 2)})


def fit_model(args, name, rows, metadata, old, v1):
    started = time.time()
    layers, experts, active = geometry(metadata)
    dimensions = flash.tensor_dimensions(metadata)
    training = [r for r in rows if r['split'] == 'train']
    routes, receipt = previous.load_cache(args.run_dir / name / 'analysis/train_routes.npz',
                                          training, layers, experts, active)
    assert [r['id'] for r in old['training_routing']] == receipt['prompt_ids']
    unique_by_layer = [np.unique(np.sort(routes[:, :, li].reshape(-1, active), axis=1),
                                 axis=0, return_counts=True) for li in range(len(layers))]
    jobs = []
    mappings = {family: {} for family in FAMILIES}
    stats, totals = {}, {}
    for g in joint.GROUPS:
        stats[f'g{g}'] = {}
        for family in FAMILIES:
            mappings[family][f'g{g}'] = {}
        for width in joint.WIDTHS:
            stats[f'g{g}'][f'w{width}'] = {}
            for family in FAMILIES:
                mappings[family][f'g{g}'][f'w{width}'] = {}
                totals[(family, g, width)] = dict(
                    sums={k: 0 for k in (*joint.COMPONENTS, *joint.COUNTERS, 'model_ns')},
                    phases={p: {k: 0 for k in joint.COMPONENTS} for p in ('gate_up', 'down')})
            for li, layer in enumerate(layers):
                warm = joint.slots_from_group_bins(
                    v1['mappings'][f'g{g}'][str(layer)],
                    v1['train_only_plane_assignments'][f'g{g}'][f'w{width}'][str(layer)],
                    g, width)
                unique, weights = unique_by_layer[li]
                jobs.append((layer, dimensions[layer], unique, weights, experts, g, width,
                             warm, args.swap_trials, args.seed + layer * 100_003 + g * 1009 + width * 101,
                             args.use_cr))
    completed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fit_layer, task) for task in jobs]
        for future in as_completed(futures):
            result = future.result()
            layer, g, width = result['layer'], result['g'], result['width']
            stats[f'g{g}'][f'w{width}'][str(layer)] = result['optimization']
            for family in FAMILIES:
                mappings[family][f'g{g}'][f'w{width}'][str(layer)] = result['mappings'][family]
                target, values = totals[(family, g, width)], result['metrics'][family]
                for k, v in values['sums'].items():
                    target['sums'][k] += v
                for p, d in values['phases'].items():
                    for k, v in d.items():
                        target['phases'][p][k] += v
            completed += 1
            if completed % max(1, len(layers)) == 0:
                print(f'{name}: {completed}/{len(jobs)} layer/configuration searches; '
                      f'{time.time() - started:.1f}s', flush=True)
    candidates = [dict(model=name, family=family, group_chips=g, plane_width=width,
                       **summary_from_sums(total['sums'], total['phases'], 200 * 128))
                  for (family, g, width), total in totals.items()]
    plan = dict(schema_version=2, model=name, use_cr=args.use_cr, seed=args.seed,
                model_sha256=metadata['sha256'],
                metadata_sha256=sha256(args.data / 'models' / f'{name}-metadata.json'),
                corpus_sha256=sha256(args.run_dir / 'corpus/prompts.jsonl'),
                old_plan_sha256=sha256(args.old_results / f'{name}.frozen_plan.json'),
                v1_plan_sha256=sha256(args.v1_results / f'{name}.frozen_plan.json'),
                training_cache=receipt, training_prompt_ids=receipt['prompt_ids'],
                mappings=mappings, optimization=stats, training_candidates=candidates,
                winners=choose_winners(candidates),
                assumptions=dict(chips=16, planes=4, page_bytes=flash.PAGE,
                                 block_wordlines=192, cold_ns=28000, cr_ns=9700,
                                 use_cr=args.use_cr, ideal_overlap=True,
                                 nand='waves*9700+cold*(28000-9700)' if args.use_cr else 'waves*28000',
                                 pipeline='max(NAND lane load, shared ECC bytes/6.4GBps)',
                                 matrix_block_command_ns=250, host_phase_ns=2000,
                                 channel_bytes_per_ns=2, pcie_bytes_per_ns=8,
                                 gate_up_stacked=True, whole_output_row_shards=True,
                                 fixed_per_expert_per_layer=True, no_weight_replication=True,
                                 independently_addressed_plane_bins_required=True),
                search=dict(groups=list(joint.GROUPS), widths=list(joint.WIDTHS),
                            starts=['identity', 'v1_trained_group_and_plane'],
                            proposals_per_start=args.swap_trials,
                            staged_group_proposals=args.swap_trials // 2,
                            staged_plane_proposals=args.swap_trials - args.swap_trials // 2,
                            degenerate_cases='g16 allocates all proposals to plane; w4 all to group',
                            acceptance='Complete frequency-weighted model latency; squared occupancy tie-breaker',
                            fixed_before_test=True),
                test_routes_loaded_before_freeze=False, test_routes_used_for_selection=False,
                frozen_unix=time.time(), fit_seconds=time.time() - started,
                software_sha256=source_hashes())
    target = args.output / f'{name}.frozen_plan.json'
    write_json(target, plan)
    write_json(args.output / f'{name}.freeze_receipt.json',
               dict(frozen_plan_sha256=sha256(target), frozen_unix=plan['frozen_unix'],
                    test_routes_loaded_before_freeze=False, test_routes_used_for_selection=False))
    print(f'{name}: frozen {sha256(target)}; '
          f'joint=(g{plan["winners"]["joint"]["group_chips"]},w{plan["winners"]["joint"]["plane_width"]})',
          flush=True)
    return plan


def scalar_checks(routes, layers, dimensions, plan):
    """Independent per-chip layout/scheduler check for every g/width/family."""
    result = []
    for family in FAMILIES:
        for g in joint.GROUPS:
            for width in joint.WIDTHS:
                for pi, ti, li in ((0, 0, 0), (1, 31, len(layers) // 2),
                                   (len(routes) - 1, 127, len(layers) - 1)):
                    layer = layers[li]
                    slots = plan['mappings'][family][f'g{g}'][f'w{width}'][str(layer)]
                    selected = routes[pi, ti, li].tolist()
                    groups, bins = joint.decode_slots(slots, g, width)
                    profiles = joint.phase_profiles(dimensions[layer], g, width, plan['use_cr'])
                    fast = joint.costs_from_counts(joint.counts_for(np.array([selected]), slots, g, width),
                                                   profiles, g, width, len(selected))
                    scalar_flash = 0
                    for p in profiles:
                        chip_service = []
                        for group in range(16 // g):
                            resident = np.flatnonzero(groups == group).tolist()
                            active = [e for e in selected if groups[e] == group]
                            masks = {e: tuple(range(int(bins[e]) * width,
                                                   (int(bins[e]) + 1) * width)) for e in resident}
                            for shard in range(g):
                                layout = plane.build_layout({e: int(p['useful'][shard]) for e in resident},
                                                            policy={4: 'baseline', 2: 'pair', 1: 'single'}[width],
                                                            masks=masks)
                                value = plane.evaluate_chip(layout, active)
                                if plan['use_cr']:
                                    chip_service.append(value['service_ns'])
                                else:
                                    lane_waves = {}
                                    for e in active:
                                        s = layout.shards[e]
                                        lane_waves[s.mask] = lane_waves.get(s.mask, 0) + s.waves
                                    chip_service.append(max(max(lane_waves.values(), default=0) * 28000,
                                                            plane.transfer_ns(sum(layout.shards[e].pages
                                                                                  for e in active) * flash.PAGE)))
                        scalar_flash += max(chip_service)
                    assert scalar_flash == int(fast['flash_ns'][0]), (family, g, width, li)
                result.append(dict(family=family, g=g, width=width, events=3,
                                   exact_per_chip_flash_match=True))
    return result


def layout_audit(metadata, plan):
    dimensions = flash.tensor_dimensions(metadata)
    layers, experts, _ = geometry(metadata)
    rows = []
    for family in FAMILIES:
        for g in joint.GROUPS:
            for width in joint.WIDTHS:
                useful = pages = reserved = slots_total = 0
                for layer in layers:
                    slots = plan['mappings'][family][f'g{g}'][f'w{width}'][str(layer)]
                    assert sorted(slots) == list(range(experts))
                    groups, bins = joint.decode_slots(slots, g, width)
                    expected_group, expected_bin = joint.decode_slots(np.arange(experts), g, width)
                    np.testing.assert_array_equal(
                        np.bincount(groups * (4 // width) + bins, minlength=(16 // g) * (4 // width)),
                        np.bincount(expected_group * (4 // width) + expected_bin,
                                    minlength=(16 // g) * (4 // width)))
                    for p in joint.phase_profiles(dimensions[layer], g, width, plan['use_cr']):
                        for group in range(16 // g):
                            resident = np.flatnonzero(groups == group).tolist()
                            masks = {e: tuple(range(int(bins[e]) * width,
                                                   (int(bins[e]) + 1) * width)) for e in resident}
                            for shard in range(g):
                                layout = plane.build_layout({e: int(p['useful'][shard]) for e in resident},
                                                            policy={4: 'baseline', 2: 'pair', 1: 'single'}[width],
                                                            masks=masks)
                                summary = layout.summary()
                                useful += summary['useful_bytes']
                                pages += summary['physical_page_bytes']
                                reserved += summary['reserved_lsb_bytes']
                                slots_total += summary['allocated_plane_slots']
                rows.append(dict(family=family, g=g, width=width, useful_weight_bytes=useful,
                                 physical_page_bytes=pages, reserved_lsb_bytes=reserved,
                                 reserved_tlc_raw_bytes=reserved * 3,
                                 allocated_plane_slots=slots_total,
                                 lsb_byte_utilization=useful / reserved,
                                 immutable_permutation_and_group_bin_capacity=True,
                                 page_bytes_not_matrix_rows=True))
    assert len({r['useful_weight_bytes'] for r in rows}) == 1
    return rows


def evaluate_model(args, name, rows, metadata, plan):
    frozen = args.output / f'{name}.frozen_plan.json'
    frozen_hash = sha256(frozen)
    receipt = json.loads((args.output / f'{name}.freeze_receipt.json').read_text())
    assert receipt['frozen_plan_sha256'] == frozen_hash
    assert not receipt['test_routes_loaded_before_freeze']
    assert not plan['test_routes_used_for_selection']
    layers, experts, active = geometry(metadata)
    dimensions = flash.tensor_dimensions(metadata)
    heldout = [r for r in rows if r['split'] == 'test']
    routes, test_receipt = previous.load_cache(args.run_dir / name / 'analysis/test_routes.npz',
                                               heldout, layers, experts, active)
    loaded_unix = time.time()
    baseline_maps = {str(layer): list(range(experts)) for layer in layers}
    base_values = joint.evaluate_routes(routes, layers, dimensions, 16, 4,
                                       baseline_maps, plan['use_cr'])
    reference = {}
    if plan['use_cr']:
        with np.load(args.run_dir / name / 'analysis/test_token_costs.npz') as data:
            for k in (*joint.COMPONENTS, 'model_ns'):
                np.testing.assert_array_equal(base_values[k], data[f'baseline_16_stacked__{k}'])
                reference[k] = True
    candidate_values, grid = {}, []
    for family in FAMILIES:
        for g in joint.GROUPS:
            for width in joint.WIDTHS:
                mapping = plan['mappings'][family][f'g{g}'][f'w{width}']
                values = joint.evaluate_routes(routes, layers, dimensions, g, width, mapping, plan['use_cr'])
                candidate_values[(family, g, width)] = values
                for window in (32, 128):
                    grid.append(dict(model=name, family=family, group_chips=g, plane_width=width,
                                     window=window, diagnostic_only_test_never_selects=True,
                                     **previous.summarize(values, window)))
    summary, per_prompt, arrays = [], [], {}
    selected = {}
    for label, config in plan['winners'].items():
        values = candidate_values[(config['family'], config['group_chips'], config['plane_width'])]
        selected[label] = values
        for k, v in values.items():
            if k != 'phases':
                arrays[f'{label}__{k}'] = v
        for window in (32, 128):
            base = base_values['model_ns'][:, :window].mean(axis=1)
            current = values['model_ns'][:, :window].mean(axis=1)
            equal = selected.get('staged_equal')
            staged_config = plan['winners']['staged_equal']
            staged_values = candidate_values[(staged_config['family'], staged_config['group_chips'],
                                              staged_config['plane_width'])]
            staged = staged_values['model_ns'][:, :window].mean(axis=1)
            item = dict(model=name, policy=label, use_cr=plan['use_cr'], window=window,
                        group_chips=config['group_chips'], plane_width=config['plane_width'],
                        training_mean_ms=config['mean_ms_per_token'],
                        **previous.summarize(values, window),
                        improvement_vs_baseline=1 - float(current.mean() / base.mean()),
                        improvement_vs_baseline_ci95=previous.paired_interval(
                            base, current, seed=args.seed, samples=args.bootstrap),
                        improvement_vs_equal_staged=1 - float(current.mean() / staged.mean()),
                        improvement_vs_equal_staged_ci95=previous.paired_interval(
                            staged, current, seed=args.seed, samples=args.bootstrap),
                        prompt_fraction_improved=float((current < base).mean()))
            summary.append(item)
            for pi, row in enumerate(heldout):
                per_prompt.append(dict(model=name, policy=label, id=row['id'], task=row['task'],
                                       length_bucket=row['length_bucket'], window=window,
                                       mean_ms=float(current[pi] / 1e6),
                                       baseline_mean_ms=float(base[pi] / 1e6)))
    np.savez_compressed(args.output / f'{name}.token_costs.npz',
                        prompt_ids=np.array(test_receipt['prompt_ids']), **arrays)
    checks = scalar_checks(routes, layers, dimensions, plan)
    capacity = layout_audit(metadata, plan)
    write_json(args.output / f'{name}.layout_audit.json', capacity)
    validation = dict(model=name, frozen_plan_sha256=frozen_hash,
                      test_routes_loaded_unix=loaded_unix, training_cache=plan['training_cache'],
                      test_cache=test_receipt, baseline_cr_exact_old_array_match=reference,
                      scalar_checks=checks, layout_audit_passed=True,
                      fixed_plan_for_prefix32_and_full128=True,
                      test_routes_used_for_selection=False)
    write_json(args.output / f'{name}.validation.json', validation)
    assert frozen_hash == sha256(frozen)
    print(f'{name}: held-out grid and independent layout checks complete', flush=True)
    return summary, grid, per_prompt, validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=DATA / 'runs/corpus-300-v1')
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--old-results', type=Path, default=ROOT / 'results/corpus-300-v1')
    parser.add_argument('--v1-results', type=Path, default=ROOT / 'results/plane-wave-v1')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/plane-wave-v2/joint-cr')
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    parser.add_argument('--swap-trials', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--stage', choices=('fit', 'test', 'both'), default='both')
    parser.add_argument('--no-cr', action='store_false', dest='use_cr', default=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in (args.run_dir / 'corpus/prompts.jsonl').read_text().splitlines()]
    assert Counter(r['split'] for r in rows) == {'train': 200, 'test': 100}
    summaries, grids, per_prompt, validations, training = [], [], [], [], []
    started = time.time()
    for name in args.models:
        metadata = json.loads((args.data / 'models' / f'{name}-metadata.json').read_text())
        old = json.loads((args.old_results / f'{name}.frozen_plan.json').read_text())
        v1 = json.loads((args.v1_results / f'{name}.frozen_plan.json').read_text())
        assert old['corpus_sha256'] == sha256(args.run_dir / 'corpus/prompts.jsonl')
        if args.stage in ('fit', 'both'):
            plan = fit_model(args, name, rows, metadata, old, v1)
        else:
            plan = json.loads((args.output / f'{name}.frozen_plan.json').read_text())
        training.extend(plan['training_candidates'])
        if args.stage in ('test', 'both'):
            s, grid, prompts, validation = evaluate_model(args, name, rows, metadata, plan)
            summaries.extend(s)
            grids.extend(grid)
            per_prompt.extend(prompts)
            validations.append(validation)
            write_json(args.output / 'summary.json', summaries)
            write_json(args.output / 'heldout_candidates.json', grids)
            write_json(args.output / 'per_prompt.json', per_prompt)
            write_json(args.output / 'validation.json', validations)
        write_json(args.output / 'training_candidates.json', training)
    manifest = dict(schema_version=2, date='2026-11-01', command=[sys.executable, *sys.argv],
                    models=args.models, use_cr=args.use_cr, seed=args.seed,
                    configuration={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    software_sha256=source_hashes(), elapsed_seconds=time.time() - started,
                    bootstrap=dict(unit='whole prompt; paired', resamples=args.bootstrap, seed=args.seed),
                    measured_hardware=False,
                    limitations=['Analytical ideal NAND/ECC overlap; independent subset-plane addressing and contexts required.',
                                 'Existing 300-prompt real routes and INT8 whole-output-row byte geometry; no new model inference.',
                                 'Heuristic fixed-budget train-only search; no global optimum claim.',
                                 'Excludes attention, prefill, routing, shared experts, DRAM cache, GC, and bus timing for NAND command headers.'])
    manifest['configuration_sha256'] = hashlib.sha256(json.dumps(manifest['configuration'], sort_keys=True).encode()).hexdigest()
    write_json(args.output / 'manifest.json', manifest)


if __name__ == '__main__':
    main()
