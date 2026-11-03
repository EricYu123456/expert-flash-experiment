#!/usr/bin/env python3
"""Replay frozen real routes with ordinary reads on both sides of every comparison.

Only NAND sense changes: every wave takes 28 us; the same max(NAND, ECC)
pipeline, gate/up stacking, vectors, host phases, and block commands remain.
Training and configuration selection use 200 prompts.  All requested models
are frozen before the first held-out route cache is opened.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
from math import ceil
from pathlib import Path
import sys
import time

import numpy as np

import aif_model as flash
import joint_plane_search as joint
import run_plane_wave_study as previous
from run_corpus_traces import DATA, MODELS, ROOT, geometry, sha256, write_json

SEED = 20261101
GROUPS, WIDTHS = joint.GROUPS, joint.WIDTHS
COMPONENTS, COUNTERS = joint.COMPONENTS, joint.COUNTERS
FAMILIES = ('group_only_modulo', 'group_only_learned',
            'staged_group_then_plane', 'joint_group_plane')


def ordinary_stream_cost(nbytes, width, block_pages=192):
    """An ordinary-read stream; block boundaries still issue block commands."""
    pages = (nbytes + flash.PAGE - 1) // flash.PAGE
    waves = (pages + width - 1) // width
    cold = (waves + block_pages - 1) // block_pages
    return dict(physical_pages=pages, waves=waves, cold_reads=cold,
                nand_ns=waves * flash.COLD_NS, matrix_commands=cold)


def scalar_event(selected, dimensions, g, width, slots, use_cr=False):
    """Independent per-chip/per-stream loop; no vectorized cost/profile calls."""
    result = {k: 0 for k in (*COMPONENTS, *COUNTERS)}
    selected = list(selected)
    ngroups, nbin = 16 // g, 4 // width
    gate, up, down = (dimensions[p] for p in ('gate', 'up', 'down'))
    phases = ((gate[0] + up[0], up[1], True), (down[0], down[1], False))
    for rows, columns, common in phases:
        chip_in, chip_out, chip_services, chip_nand, chip_ecc = [], [], [], [], []
        phase_contexts = 0
        phase_context_bytes = 0
        for group in range(ngroups):
            residents = [e for e in selected if slots[e] % ngroups == group]
            occupied = {((slots[e] // ngroups) % nbin) for e in residents}
            extras = max(len(occupied) - 1, 0)
            phase_contexts += extras * g
            phase_context_bytes += extras * (rows * 4 + (0 if common else columns * g))
            for shard in range(g):
                shard_rows = rows // g + (shard < rows % g)
                nbytes = shard_rows * columns
                p = ordinary_stream_cost(nbytes, width)
                lanes, lane_waves = [0] * nbin, [0] * nbin
                for expert in residents:
                    lane = (slots[expert] // ngroups) % nbin
                    lanes[lane] += (p['waves'] * flash.CR_NS + p['cold_reads'] *
                                    (flash.COLD_NS - flash.CR_NS)) if use_cr else p['nand_ns']
                    lane_waves[lane] += p['waves']
                pages = len(residents) * p['physical_pages']
                nand, ecc = max(lanes), pages * 2560
                chip_services.append(max(nand, ecc))
                chip_nand.append(nand)
                chip_ecc.append(ecc)
                chip_in.append(columns * ((len(residents) > 0) if common else len(residents)))
                chip_out.append(len(residents) * shard_rows * 4)
                result['useful_bytes'] += len(residents) * nbytes
                result['physical_pages'] += pages
                result['read_waves'] += max(lane_waves)
                result['ordinary_read_epochs'] += (max(lane_waves) if not use_cr else
                                                  max((lane_waves[b] // p['waves'] if p['waves'] else 0)
                                                      * p['cold_reads'] for b in range(nbin)))
                result['cold_reads'] += len(residents) * p['cold_reads']
                result['matrix_commands'] += len(residents) * p['cold_reads']
                result['nand_read_commands'] += len(residents) * p['waves']
                result['plane_read_commands'] += pages
        result['flash_ns'] += max(chip_services)
        result['nand_bound_ns'] += max(chip_nand)
        result['ecc_bound_ns'] += max(chip_ecc)
        result['host_ns'] += 2000
        result['extra_contexts'] += phase_contexts
        result['context_bytes'] += phase_context_bytes
        host_in = columns * (1 if common else len(selected))
        result['input_ns'] += max(ceil(host_in / 8),
                                  ceil(max(chip_in[c] + chip_in[c + 8] for c in range(8)) / 2))
        result['output_ns'] += max(ceil(sum(chip_out) / 8),
                                   ceil(max(chip_out[c] + chip_out[c + 8] for c in range(8)) / 2))
    result['control_ns'] = result['matrix_commands'] * 250
    result['model_ns'] = sum(result[k] for k in COMPONENTS)
    return result


def protocol(args):
    return dict(seed=args.seed, groups=list(GROUPS), widths=list(WIDTHS),
                use_cr=False, proposals_per_start=args.swap_trials, starts=2,
                initializations=['identity', 'v1_train_only_group_and_plane'],
                staged_budget='g<16,w<4: half group(full w4 cost), half fixed-group plane; '
                              'g16: all plane; w4: all group',
                proposals_count='All randomly sampled swaps, including equal-slot-bin no-ops.',
                full_latency_objective=True, input_tokens=128, training_prompts=200,
                test_prompts=100, test_windows=[32, 128],
                original_initialization_kept_if_staged_final_objective_regresses=True,
                analytical_timing='max(NAND lane completion, shared ECC work); ordinary 28 us each wave',
                assumptions=dict(chips=16, planes=4, page_bytes=flash.PAGE, block_wordlines=192,
                                 ordinary_read_ns=flash.COLD_NS, shared_chip_ecc_bytes_per_second=6_400_000_000,
                                 matrix_block_command_ns=250, host_phase_ns=2000,
                                 input_element_bytes=1, output_element_bytes=4,
                                 host_bandwidth_bytes_per_ns=8, channel_bandwidth_bytes_per_ns=2,
                                 channels=8, chip_channel_pair='c and c+8', gate_up_stacked=True,
                                 independent_plane_addressing_and_multiple_contexts_required=True))


def source_hashes():
    names = ('run_no_cr_study.py', 'joint_plane_search.py', 'run_plane_wave_study.py',
             'plane_wave_model.py', 'corpus_analysis.py', 'aif_model.py', 'overhead_model.py')
    return {name: sha256(ROOT / 'scripts' / name) for name in names}


def seed_for(seed, layer, g, width, start):
    return seed + layer * 100003 + g * 1009 + width * 101 + start * 10000019


def fit_layer_group(task):
    """One layer/group task; identical starts and budgets in both approaches."""
    layer, g, dimensions, experts, unique, weights, old, budget, seed = task
    mappings = {family: {} for family in FAMILIES}
    stats = {}
    identity = list(range(experts))
    group_profile = joint.phase_profiles(dimensions, g, 4, use_cr=False)
    group_starts = [identity, joint.slots_from_group_bins(old['mapping'], [0] * experts, g, 4)]
    group_runs = []
    for si, initial in enumerate(group_starts):
        slots, receipt = joint.optimize_slots(unique, weights, experts, initial, g, 4,
                                              group_profile, budget,
                                              seed_for(seed, layer, g, 4, si), True)
        group_runs.append((joint.score_slots(unique, weights, slots, g, 4, group_profile),
                           slots, receipt))
    best = min(range(2), key=lambda i: (group_runs[i][0], i))
    mappings['group_only_modulo']['4'] = identity
    for family in ('group_only_learned', 'staged_group_then_plane', 'joint_group_plane'):
        mappings[family]['4'] = group_runs[best][1]
    stats['4'] = dict(selected_start=best, group_runs=[r[2] for r in group_runs],
                      staged_equals_joint=True)
    for width in (2, 1):
        profile = joint.phase_profiles(dimensions, g, width, use_cr=False)
        starts = [identity, joint.slots_from_group_bins(old['mapping'], old['bins'][str(width)], g, width)]
        runs = {'joint': [], 'staged': []}
        for si, initial in enumerate(starts):
            rng_seed = seed_for(seed, layer, g, width, si)
            slots, receipt = joint.optimize_slots(unique, weights, experts, initial, g, width,
                                                  profile, budget, rng_seed, True)
            runs['joint'].append((joint.score_slots(unique, weights, slots, g, width, profile), slots, receipt))
            if g == 16:
                runs['staged'].append(runs['joint'][-1])
                continue
            group_slots, group_receipt = joint.optimize_slots(
                unique, weights, experts, initial, g, 4, group_profile,
                budget // 2, rng_seed, True)
            staged, plane_receipt = joint.optimize_slots(
                unique, weights, experts, group_slots, g, width, profile,
                budget - budget // 2, rng_seed + 1, False)
            initial_score = joint.score_slots(unique, weights, initial, g, width, profile)
            staged_score = joint.score_slots(unique, weights, staged, g, width, profile)
            kept_initial = initial_score < staged_score
            if kept_initial:
                staged, staged_score = initial, initial_score
            staged_receipt = dict(group=group_receipt, plane=plane_receipt,
                                  actual_width_initial_score=list(initial_score),
                                  actual_width_final_score=list(staged_score),
                                  kept_initial_seed=kept_initial)
            runs['staged'].append((staged_score, staged, staged_receipt))
        stats[str(width)] = {}
        for method, family in (('joint', 'joint_group_plane'), ('staged', 'staged_group_then_plane')):
            chosen = min(range(2), key=lambda i: (runs[method][i][0], i))
            mappings[family][str(width)] = runs[method][chosen][1]
            stats[str(width)][method] = dict(selected_start=chosen,
                                             runs=[r[2] for r in runs[method]],
                                             selected_score=list(runs[method][chosen][0]))
    return layer, g, mappings, stats


def candidate_key(family, g, width):
    return f'{family}__g{g}__w{width}'


def candidate_row(family, g, width, values, window=128):
    return dict(key=candidate_key(family, g, width), family=family,
                group_chips=g, plane_width=width, **previous.summarize(values, window))


def choose_winners(candidates):
    def choose(population):
        return min(population, key=lambda c: (c['mean_ms_per_token'], -c['plane_width'], c['group_chips']))
    winners = {'baseline': dict(key='baseline', family='baseline', group_chips=16, plane_width=4)}
    for family in FAMILIES:
        population = [c for c in candidates if c['family'] == family]
        winners[family] = choose(population)
        if family in ('staged_group_then_plane', 'joint_group_plane'):
            winners[family + '_forced_plane'] = choose([c for c in population if c['plane_width'] < 4])
    plane = [c for c in candidates if c['family'] == 'joint_group_plane' and c['group_chips'] == 16]
    # g=16,w4 is identity-equivalent regardless of expert slot permutation.
    winners['plane_only_g16'] = choose(plane)
    winners['plane_only_g16_forced_plane'] = choose([c for c in plane if c['plane_width'] < 4])
    return winners


def fit_model(args, name, rows, metadata):
    started = time.time()
    layers, experts, active = geometry(metadata)
    dimensions = flash.tensor_dimensions(metadata)
    old_path = args.old_results / (name + '.frozen_plan.json')
    old = json.loads(old_path.read_text())
    training = [r for r in rows if r['split'] == 'train']
    routes, receipt = previous.load_cache(args.run_dir / name / 'analysis/train_routes.npz',
                                          training, layers, experts, active)
    assert receipt['prompt_ids'] == old['training_prompt_ids']
    unique_layers = [np.unique(np.sort(routes[:, :, li].reshape(-1, active), axis=1),
                               axis=0, return_counts=True) for li in range(len(layers))]
    tasks = []
    for g in GROUPS:
        for li, layer in enumerate(layers):
            old_seed = dict(mapping=old['mappings'][f'g{g}'][str(layer)],
                            bins={str(w): old['train_only_plane_assignments'][f'g{g}'][f'w{w}'][str(layer)]
                                  for w in (2, 1)})
            tasks.append((layer, g, dimensions[layer], experts, *unique_layers[li],
                          old_seed, args.swap_trials, args.seed))
    mappings = {family: {} for family in FAMILIES}
    optimization = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, (layer, g, learned, stats) in enumerate(pool.map(fit_layer_group, tasks)):
            optimization.setdefault(f'g{g}', {})[str(layer)] = stats
            for family, widths in learned.items():
                for width, slots in widths.items():
                    key = candidate_key(family, g, int(width))
                    mappings[family].setdefault(key, {})[str(layer)] = slots
            if (index + 1) % len(layers) == 0:
                print(f'{name}: no-CR fit g={g} complete ({index + 1}/{len(tasks)} layer-group tasks)', flush=True)
    flat_maps = {key: value for family in mappings.values() for key, value in family.items()}
    candidates = []
    for family in FAMILIES:
        for g in GROUPS:
            for width in (WIDTHS if family in ('staged_group_then_plane', 'joint_group_plane') else (4,)):
                key = candidate_key(family, g, width)
                values = joint.evaluate_routes(routes, layers, dimensions, g, width,
                                               flat_maps[key], use_cr=False)
                candidates.append(candidate_row(family, g, width, values))
    identity = {str(layer): list(range(experts)) for layer in layers}
    baseline = joint.evaluate_routes(routes, layers, dimensions, 16, 4, identity, use_cr=False)
    plan = dict(schema_version=2, study='ordinary_read_only', model=name,
                protocol=protocol(args), model_sha256=metadata['sha256'],
                metadata_sha256=sha256(args.data / 'models' / (name + '-metadata.json')),
                corpus_sha256=sha256(args.run_dir / 'corpus/prompts.jsonl'),
                old_seed_plan_sha256=sha256(old_path), training_cache=receipt,
                training_prompt_ids=receipt['prompt_ids'], mappings=flat_maps,
                optimization=optimization, training_candidates=candidates,
                baseline_training=previous.summarize(baseline, 128), winners=choose_winners(candidates),
                source_sha256=source_hashes(), frozen_unix=time.time(),
                test_routes_loaded_before_freeze=False, test_routes_used_for_selection=False,
                fit_seconds=time.time() - started)
    plan['configuration_sha256'] = hashlib.sha256(json.dumps(plan['protocol'], sort_keys=True).encode()).hexdigest()
    write_json(args.output / (name + '.frozen_plan.json'), plan)
    print(f'{name}: frozen no-CR plan after {plan["fit_seconds"]:.1f}s; '
          f'joint g={plan["winners"]["joint_group_plane"]["group_chips"]},'
          f'w={plan["winners"]["joint_group_plane"]["plane_width"]}', flush=True)
    return plan


def scalar_checks(routes, layers, dimensions, plan):
    checks = []
    picks = [(0, 0, 0), (min(1, len(routes) - 1), 31, len(layers) // 2),
             (len(routes) - 1, 127, len(layers) - 1)]
    for cell in plan['training_candidates']:
        g, width = cell['group_chips'], cell['plane_width']
        mapping = plan['mappings'][cell['key']]
        for pi, ti, li in picks:
            layer, selected = layers[li], routes[pi, ti, li].tolist()
            scalar = scalar_event(selected, dimensions[layer], g, width, mapping[str(layer)])
            fast = joint.costs_from_counts(
                joint.counts_for(np.array([selected]), mapping[str(layer)], g, width),
                joint.phase_profiles(dimensions[layer], g, width, use_cr=False), g, width, len(selected))
            for k, value in scalar.items():
                if int(fast[k][0]) != value:
                    raise AssertionError(f'{cell["key"]}/{layer}/{k}: {fast[k][0]} != {value}')
            checks.append(dict(key=cell['key'], layer=layer, prompt=pi, token=ti,
                               all_components_and_counters_equal=True))
    return checks


def add_derived(row, baseline_prompt, prompt, args):
    row['improvement_vs_no_cr_baseline'] = float(1 - prompt.mean() / baseline_prompt.mean())
    row['improvement_95ci'] = previous.paired_interval(baseline_prompt, prompt,
                                                      seed=args.seed, samples=args.bootstrap)
    row['fraction_prompts_improved'] = float((prompt < baseline_prompt).mean())
    return row


def evaluate_model(args, name, rows, metadata, plan):
    frozen = args.output / (name + '.frozen_plan.json')
    frozen_hash = sha256(frozen)
    assert not plan['test_routes_loaded_before_freeze'] and not plan['test_routes_used_for_selection']
    assert plan['source_sha256'] == source_hashes(), 'Software changed after freeze; repeat fit before held-out evaluation'
    heldout = [r for r in rows if r['split'] == 'test']
    layers, experts, active = geometry(metadata)
    dimensions = flash.tensor_dimensions(metadata)
    routes, receipt = previous.load_cache(args.run_dir / name / 'analysis/test_routes.npz',
                                          heldout, layers, experts, active)
    assert not set(receipt['prompt_ids']) & set(plan['training_prompt_ids'])
    checks = scalar_checks(routes, layers, dimensions, plan)
    identity = {str(layer): list(range(experts)) for layer in layers}
    baseline = joint.evaluate_routes(routes, layers, dimensions, 16, 4, identity, use_cr=False)
    cache = {'baseline': baseline}
    candidates = []
    for cell in plan['training_candidates']:
        value = joint.evaluate_routes(routes, layers, dimensions, cell['group_chips'], cell['plane_width'],
                                      plan['mappings'][cell['key']], use_cr=False)
        cache[cell['key']] = value
        for window in (32, 128):
            candidates.append(dict(model=name, window=window,
                                   **candidate_row(cell['family'], cell['group_chips'], cell['plane_width'], value, window)))
    summaries, per_prompt, sensitivity, arrays, comparisons = [], [], [], {}, []
    for label, winner in plan['winners'].items():
        value = cache[winner['key']]
        arrays.update(previous.flatten_values(label, value))
        for window in (32, 128):
            base_prompt = baseline['model_ns'][:, :window].mean(axis=1)
            prompt = value['model_ns'][:, :window].mean(axis=1)
            row = dict(model=name, policy=label, group_chips=winner['group_chips'],
                       plane_width=winner['plane_width'], window=window, prompts=len(heldout),
                       **previous.summarize(value, window))
            summaries.append(add_derived(row, base_prompt, prompt, args))
            for i, sample in enumerate(heldout):
                per_prompt.append(dict(model=name, policy=label, group_chips=winner['group_chips'],
                                       plane_width=winner['plane_width'], window=window, id=sample['id'],
                                       task=sample['task'], length_bucket=sample['length_bucket'],
                                       mean_ms_per_token=float(prompt[i] / 1e6),
                                       baseline_ms_per_token=float(base_prompt[i] / 1e6)))
        for context_ns in (0, 250, 1000, 5000, 20000):
            for command_ns in (0, 50, 250, 1000):
                base = baseline['model_ns'] + context_ns * baseline['extra_contexts'] + command_ns * baseline['nand_read_commands']
                adjusted = value['model_ns'] + context_ns * value['extra_contexts'] + command_ns * value['nand_read_commands']
                sensitivity.append(dict(model=name, policy=label, window=128,
                                        extra_context_setup_ns=context_ns, extra_nand_lane_command_ns=command_ns,
                                        mean_ms_per_token=float(adjusted.mean() / 1e6),
                                        baseline_ms_per_token=float(base.mean() / 1e6),
                                        improvement=float(1 - adjusted.mean() / base.mean())))
    for label, reference_label in (('joint_group_plane', 'staged_group_then_plane'),
                                    ('joint_group_plane_forced_plane', 'staged_group_then_plane_forced_plane')):
        candidate = cache[plan['winners'][label]['key']]
        reference = cache[plan['winners'][reference_label]['key']]
        for window in (32, 128):
            b = reference['model_ns'][:, :window].mean(axis=1)
            c = candidate['model_ns'][:, :window].mean(axis=1)
            comparisons.append(dict(model=name, candidate=label, reference=reference_label, window=window,
                                    improvement=float(1 - c.mean() / b.mean()),
                                    improvement_95ci=previous.paired_interval(b, c, seed=args.seed, samples=args.bootstrap)))
    np.savez_compressed(args.output / (name + '.token_costs.npz'),
                        prompt_ids=np.array(receipt['prompt_ids']), **arrays)
    # Fully specified baseline accounting; no-cr sense must obey both lower bounds.
    nand_min = ecc_min = useful_total = pages_total = commands_total = 0
    for layer in layers:
        for p in joint.phase_profiles(dimensions[layer], 16, 4, use_cr=False):
            nand_min += active * int(p['waves'].max(initial=0)) * flash.COLD_NS
            ecc_min += active * int(p['pages'].max(initial=0)) * 2560
            useful_total += active * int(p['useful'].sum())
            pages_total += active * int(p['pages'].sum())
            commands_total += active * int(p['cold'].sum())
    assert (baseline['flash_ns'] >= nand_min).all() and (baseline['flash_ns'] >= ecc_min).all()
    assert (baseline['useful_bytes'] == useful_total).all()
    assert (baseline['physical_pages'] == pages_total).all()
    assert (baseline['matrix_commands'] == commands_total).all()
    validation = dict(model=name, frozen_plan_sha256=frozen_hash, test_cache=receipt,
                      no_cr_on_baseline_and_all_candidates=True, overlap_preserved=True,
                      ordinary_read_epochs_equal_read_waves=all(np.array_equal(v['ordinary_read_epochs'], v['read_waves'])
                                                               for v in cache.values()),
                      all_models_frozen_before_first_heldout_open=True,
                      test_routes_used_for_training_or_selection=False,
                      fixed_plan_for_prefix32_and_full128=True, scalar_reference_checks=checks,
                      baseline_bounds=dict(nand_ns_per_token=nand_min, ecc_ns_per_token=ecc_min,
                                           flash_ns_per_token=float(baseline['flash_ns'].mean()),
                                           useful_bytes_per_token=useful_total, physical_pages_per_token=pages_total,
                                           matrix_block_commands_per_token=commands_total),
                      source_sha256=source_hashes(), evaluated_unix=time.time())
    assert validation['ordinary_read_epochs_equal_read_waves']
    write_json(args.output / (name + '.validation.json'), validation)
    assert sha256(frozen) == frozen_hash
    print(f'{name}: held-out no-CR evaluation complete; {len(checks)} scalar checks', flush=True)
    return summaries, per_prompt, sensitivity, candidates, comparisons, validation


def write_csv(path, rows):
    if rows:
        with path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def write_report(output, summaries, comparisons):
    selected = {(r['model'], r['policy']): r for r in summaries if r['window'] == 128}
    lines = ['# 無 cr-read：固定配置實驗', '',
             '兩側全部使用普通讀取：每波 28 μs。baseline 為 g=16、四 planes、原始 expert ID、'
             '無學習，gate/up 已串接。其餘 NAND/ECC 重疊、向量與命令成本相同。', '',
             '200 個 training prompts（各 128 tokens）學配置與選 g/plane 寬度；全部模型凍結後，'
             '評估獨立 100 個 test prompts。下表為 test 的 ms/input；括號為相對各模型 baseline 的降低百分比。', '',
             '| 模型 | baseline | 只分組 modulo | 分組＋學排列 | 只 plane g16 | 先 group 再 plane | 聯合 group＋plane |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for model in MODELS:
        if (model, 'baseline') not in selected:
            continue
        cells = [model, f'{selected[model, "baseline"]["mean_ms_per_token"]:.4f}']
        for label in ('group_only_modulo', 'group_only_learned', 'plane_only_g16',
                      'staged_group_then_plane', 'joint_group_plane'):
            row = selected[model, label]
            cells.append(f'{row["mean_ms_per_token"]:.4f} ({100 * row["improvement_vs_no_cr_baseline"]:+.2f}%, '
                         f'g={row["group_chips"]},w={row["plane_width"]})')
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines.extend(['', '上述策略均允許退回 w=4（無 plane 分工）。以下強制使用 w<4，避免把 fallback 當成 plane 的收益。', '',
                  '| 模型 | 聯合搜尋強制 plane | 相對 baseline 降低 | paired 95% CI |',
                  '|---|---:|---:|---:|'])
    for model in MODELS:
        row = selected.get((model, 'joint_group_plane_forced_plane'))
        if row:
            ci = row['improvement_95ci']
            lines.append(f'| {model} | {row["mean_ms_per_token"]:.4f} (g={row["group_chips"]},w={row["plane_width"]}) | '
                         f'{100 * row["improvement_vs_no_cr_baseline"]:+.2f}% | '
                         f'[{100 * ci[0]:+.2f}%, {100 * ci[1]:+.2f}%] |')
    lines.extend(['', '每層、每個 g/w 採相同兩個起點（原始 ID、舊 training-only group/plane 配置），'
                  '每起點 1,024 次 swap proposals。聯合搜尋可跨 group/bin 交換；staged 在 g<16,w<4 時分 '
                  '512 次 group、512 次固定 group 的 plane 搜尋；g=16 或 w=4 則把全部預算用於唯一可學的維度。'
                  '完整延遲作為 training objective。這是有限預算搜尋，無法證明全域最佳。', '',
                  '模型仍假設獨立 plane 定址、多 contexts 及 max(NAND,ECC) 管線。額外 context/每 lane read 命令成本的'
                  '敏感度保存在 `sensitivity.csv`；prefix32、完整 128、paired prompt bootstrap 2,000 次 CI、每 prompt 和 '
                  'per-token 成本分別見 `summary.json`、`per_prompt.csv`、`*.token_costs.npz`。'
                  '250 ns/matrix-block 的既有命令成本一直計入；跨 block 不再改變 sense 延遲。', '',
                  '所有 g/w cells 見 `training_candidates.json` 與 `heldout_candidates.json`；held-out grid 僅用於描述，'
                  '沒有用來重選配置。搜尋／資料 hashes 與凍結紀錄見 `*.frozen_plan.json`、`*.validation.json`。', ''])
    (output / 'report.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=DATA / 'runs/corpus-300-v1')
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--old-results', type=Path, default=ROOT / 'results/plane-wave-v1')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/plane-wave-v2/no-cr')
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    parser.add_argument('--swap-trials', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--stage', choices=('fit', 'test', 'both'), default='both')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in (args.run_dir / 'corpus/prompts.jsonl').read_text().splitlines()]
    assert Counter(r['split'] for r in rows) == {'train': 200, 'test': 100}
    assert len({r['id'] for r in rows}) == 300
    started = time.time()
    plans, metadata = {}, {}
    for name in args.models:
        metadata[name] = json.loads((args.data / 'models' / (name + '-metadata.json')).read_text())
        if args.stage in ('fit', 'both'):
            plans[name] = fit_model(args, name, rows, metadata[name])
        else:
            plans[name] = json.loads((args.output / (name + '.frozen_plan.json')).read_text())
    training = [row for plan in plans.values() for row in plan['training_candidates']]
    write_json(args.output / 'training_candidates.json', training)
    freeze_receipt = {name: sha256(args.output / (name + '.frozen_plan.json')) for name in args.models}
    write_json(args.output / 'all_models_freeze_receipt.json',
               dict(frozen_plans=freeze_receipt, before_any_heldout_cache_open=True, created_unix=time.time()))
    summaries, per_prompt, sensitivity, candidates, comparisons, validations = [], [], [], [], [], []
    if args.stage in ('test', 'both'):
        for name in args.models:
            values = evaluate_model(args, name, rows, metadata[name], plans[name])
            for target, rows_out in zip((summaries, per_prompt, sensitivity, candidates, comparisons, validations), values):
                target.extend(rows_out if isinstance(rows_out, list) else [rows_out])
            write_json(args.output / 'summary.json', summaries)
            write_csv(args.output / 'per_prompt.csv', per_prompt)
            write_csv(args.output / 'sensitivity.csv', sensitivity)
            write_json(args.output / 'heldout_candidates.json', candidates)
            write_json(args.output / 'joint_vs_staged.json', comparisons)
            write_json(args.output / 'validation.json', validations)
            write_report(args.output, summaries, comparisons)
    manifest = dict(schema_version=2, study='ordinary_read_only', date='2026-11-01',
                    command=[sys.executable, *sys.argv], models=args.models,
                    protocol=protocol(args), source_sha256=source_hashes(),
                    corpus_sha256=sha256(args.run_dir / 'corpus/prompts.jsonl'),
                    elapsed_seconds=time.time() - started,
                    all_models_freeze_receipt=freeze_receipt,
                    bootstrap=dict(unit='whole prompt; paired', resamples=args.bootstrap, seed=args.seed),
                    no_new_inference_or_nvme_measurements=True,
                    timing_claim='Conditional analytical FFN model latency; not measured end-to-end latency.')
    write_json(args.output / 'manifest.json', manifest)


if __name__ == '__main__':
    main()
