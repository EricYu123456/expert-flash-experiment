#!/usr/bin/env python3
"""Train fixed plane subsets, then evaluate disjoint cached real MoE routes.

The ideal-overlap result is an analytical lower bound. Retaining cr-read state
across ECC stalls is a separate assumption. Conservative-drain uses an ordinary
28 us read for every wave followed by its ECC drain. The independent plane/context
architecture is not supported by the old NVMe kernel.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np

import aif_model as flash
import corpus_analysis as corpus
from run_corpus_traces import DATA, MODELS, ROOT, geometry, sha256, write_json

GROUPS = (1, 2, 4, 8, 16)
WIDTHS = (4, 2, 1)
TIMINGS = ('ideal_overlap', 'retained_cr_schedule', 'conservative_drain')
SEED = 20261030
BLOCK_PAGES = 192
COMPONENTS = ('flash_ns', 'control_ns', 'host_ns', 'input_ns', 'output_ns')
COUNTERS = ('useful_bytes', 'physical_pages', 'read_waves', 'cold_reads',
            'matrix_commands', 'nand_read_commands', 'plane_read_commands',
            'extra_contexts', 'context_bytes', 'nand_bound_ns', 'ecc_bound_ns',
            'ordinary_read_epochs')


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def phase_profiles(dimensions, g, width):
    """Same whole-output-row shards and INT8 bytes as the frozen experiment."""
    result = []
    gate, up, down = (dimensions[p] for p in ('gate', 'up', 'down'))
    for name, (rows, columns), common in (
            ('gate_up', (gate[0] + up[0], up[1]), True),
            ('down', down, False)):
        row_counts = np.array([rows // g + (i < rows % g) for i in range(g)], dtype=np.int64)
        useful = row_counts * columns
        pages = (useful + flash.PAGE - 1) // flash.PAGE
        waves = (pages + width - 1) // width
        cold = (waves + BLOCK_PAGES - 1) // BLOCK_PAGES
        nand = waves * flash.CR_NS + cold * (flash.COLD_NS - flash.CR_NS)
        ecc = pages * 2560  # 16 KiB / 6.4 GB/s, exactly integral nanoseconds.
        result.append(dict(name=name, common=common, columns=columns,
                           rows=row_counts, useful=useful, pages=pages,
                           waves=waves, cold=cold, nand=nand, ecc=ecc))
    return result


def counts_for(routes, placement, bins, g, width):
    ngroups, nbin = 16 // g, 4 // width
    groups = np.asarray(placement, dtype=np.int16)[routes] % ngroups
    sub = np.asarray(bins, dtype=np.int16)[routes]
    codes = groups * nbin + sub
    counts = np.stack([(codes == c).sum(axis=-1) for c in range(ngroups * nbin)], axis=-1)
    return counts.reshape((*routes.shape[:-1], ngroups, nbin))


def costs_from_counts(counts, profiles, g, width, active, timing='ideal_overlap'):
    """Closed form for homogeneous streams; scalar scheduler cross-checks it.

    Each bin executes its fixed assigned experts serially; bins use disjoint
    physical planes. No active expert moves to a different bin at runtime.
    """
    if timing not in TIMINGS:
        raise ValueError(timing)
    shape = counts.shape[:-2]
    counts = counts.reshape((-1, *counts.shape[-2:]))
    group_counts = counts.sum(axis=-1)
    peak_bins = counts.max(axis=-1)
    global_bin_peak = peak_bins.max(axis=-1)
    global_group_peak = group_counts.max(axis=-1)
    chip_counts = np.repeat(group_counts, g, axis=1)
    result = {k: np.zeros(len(counts), dtype=np.int64) for k in (*COMPONENTS, *COUNTERS)}
    phase_result = {}
    for p in profiles:
        nand = global_bin_peak * p['nand'].max(initial=0)
        ecc = global_group_peak * p['ecc'].max(initial=0)
        if timing == 'ideal_overlap':
            service = np.maximum(nand, ecc)
        elif timing == 'retained_cr_schedule':
            # All shards have the same dimensions within each group. Largest
            # row count maximizes both NAND and ECC, including uneven shards.
            service = (peak_bins * p['nand'].max(initial=0) +
                       group_counts * p['ecc'].max(initial=0)).max(axis=-1)
        else:
            # ECC drain interrupts the read chain. A fresh ordinary read for
            # every synchronized epoch avoids assuming cr-read state retention.
            service = (peak_bins * p['waves'].max(initial=0) * flash.COLD_NS +
                       group_counts * p['ecc'].max(initial=0)).max(axis=-1)
        inputs = ((chip_counts > 0) if p['common'] else chip_counts) * p['columns']
        outputs = chip_counts * np.tile(p['rows'] * 4, 16 // g)
        host_input = p['columns'] * (1 if p['common'] else active)
        channel_in = inputs[:, :8] + inputs[:, 8:]
        channel_out = outputs[:, :8] + outputs[:, 8:]
        streams = active * int(p['cold'].sum())
        phase = dict(flash_ns=service,
                     control_ns=np.full(len(counts), 250 * streams, dtype=np.int64),
                     host_ns=np.full(len(counts), 2000, dtype=np.int64),
                     input_ns=np.maximum((host_input + 7) // 8, (channel_in.max(axis=1) + 1) // 2),
                     output_ns=np.maximum((outputs.sum(axis=1) + 7) // 8,
                                          (channel_out.max(axis=1) + 1) // 2))
        for k, v in phase.items():
            result[k] += v
        used_bins = (counts > 0).sum(axis=-1)
        extras = np.maximum(used_bins - 1, 0).sum(axis=-1)
        result['useful_bytes'] += active * int(p['useful'].sum())
        result['physical_pages'] += active * int(p['pages'].sum())
        result['read_waves'] += peak_bins.sum(axis=-1) * int(p['waves'].sum())
        result['ordinary_read_epochs'] += peak_bins.sum(axis=-1) * int(
            (p['waves'] if timing == 'conservative_drain' else p['cold']).sum())
        result['cold_reads'] += streams
        result['matrix_commands'] += streams
        result['nand_read_commands'] += active * int(p['waves'].sum())
        result['plane_read_commands'] += active * int(p['pages'].sum())
        result['extra_contexts'] += extras * g
        # Each extra context holds partial FP32 outputs; down also needs its
        # expert-specific INT8 input. This excludes implementation metadata.
        result['context_bytes'] += extras * (int(p['rows'].sum()) * 4 +
                                             (0 if p['common'] else p['columns'] * g))
        result['nand_bound_ns'] += nand
        result['ecc_bound_ns'] += ecc
        phase_result[p['name']] = {k: v.reshape(shape) for k, v in phase.items()}
    result['model_ns'] = sum(result[k] for k in COMPONENTS)
    result = {k: v.reshape(shape) for k, v in result.items()}
    result['phases'] = phase_result
    return result


def evaluate_routes(routes, layers, dimensions, g, width, mappings, bins,
                    timing='ideal_overlap'):
    shape = routes.shape[:2]
    result = {k: np.zeros(shape, dtype=np.int64) for k in (*COMPONENTS, *COUNTERS, 'model_ns')}
    result['phases'] = {p: {k: np.zeros(shape, dtype=np.int64) for k in COMPONENTS}
                        for p in ('gate_up', 'down')}
    for li, layer in enumerate(layers):
        layer_routes = routes[:, :, li, :]
        counts = counts_for(layer_routes, mappings[str(layer)], bins[str(layer)], g, width)
        values = costs_from_counts(counts, phase_profiles(dimensions[layer], g, width),
                                  g, width, routes.shape[-1], timing)
        for k in result:
            if k != 'phases':
                result[k] += values[k]
        for p in result['phases']:
            for k in COMPONENTS:
                result['phases'][p][k] += values['phases'][p][k]
    return result


def optimize_bins(unique, weights, experts, placement, g, width, profiles, trials, seed):
    """Fixed-budget train-only capacity-preserving swaps inside chip groups."""
    ngroups, nbin = 16 // g, 4 // width
    placement = np.asarray(placement)
    bins = ((placement // ngroups) % nbin).astype(np.int16)
    if nbin == 1:
        return bins.tolist(), dict(attempted=0, accepted=0, unique_routes=len(unique))
    incidence = np.zeros((len(unique), experts), dtype=np.int16)
    incidence[np.arange(len(unique))[:, None], unique] = 1
    group_ids = placement % ngroups
    codes = group_ids * nbin + bins
    counts = np.stack([incidence[:, codes == code].sum(axis=1)
                       for code in range(ngroups * nbin)], axis=1)
    group_counts = counts.reshape(-1, ngroups, nbin).sum(axis=-1)
    fixed_ecc = [group_counts.max(axis=1) * p['ecc'].max(initial=0) for p in profiles]
    nand_coeff = [p['nand'].max(initial=0) for p in profiles]

    def score():
        peak = counts.max(axis=1)
        latency = sum(np.maximum(peak * t, e) for t, e in zip(nand_coeff, fixed_ecc))
        return int(latency @ weights), int(np.square(counts).sum(axis=1) @ weights)

    initial = current = score()
    accepted = attempted = 0
    # At most one resident expert per bin already attains the occupancy bound.
    eligible = [np.flatnonzero(group_ids == group).tolist() for group in range(ngroups)
                if (np.bincount(bins[group_ids == group], minlength=nbin) > 1).any()]
    rng = random.Random(seed)
    for _ in range(trials if eligible else 0):
        group_experts = rng.choice(eligible)
        a, b = rng.sample(group_experts, 2)
        if bins[a] == bins[b]:
            continue
        attempted += 1
        ca, cb = int(codes[a]), int(codes[b])
        delta = incidence[:, b] - incidence[:, a]
        counts[:, ca] += delta
        counts[:, cb] -= delta
        candidate = score()
        if candidate < current:
            bins[a], bins[b] = bins[b], bins[a]
            codes[a], codes[b] = codes[b], codes[a]
            current = candidate
            accepted += 1
        else:
            counts[:, ca] -= delta
            counts[:, cb] += delta
    assert current <= initial
    return bins.tolist(), dict(attempted=attempted, accepted=accepted, unique_routes=len(unique),
                              initial_flash_ns_weighted=initial[0], final_flash_ns_weighted=current[0],
                              initial_squared_occupancy=initial[1], final_squared_occupancy=current[1])


def summarize(values, window):
    sl = (slice(None), slice(0, window))
    result = dict(mean_ms_per_token=float(values['model_ns'][sl].mean() / 1e6),
                  components_ms={k: float(values[k][sl].mean() / 1e6) for k in COMPONENTS},
                  counters_per_token={k: float(values[k][sl].mean()) for k in COUNTERS})
    counts = result['counters_per_token']
    result['plane_slot_fill'] = counts['useful_bytes'] / (counts['read_waves'] * 4 * flash.PAGE)
    result['physical_page_fill'] = counts['useful_bytes'] / (counts['physical_pages'] * flash.PAGE)
    result['phase_components_ms'] = {
        p: {k: float(v[sl].mean() / 1e6) for k, v in components.items()}
        for p, components in values['phases'].items()}
    return result


def paired_interval(baseline, candidate, *, seed=SEED, samples=2000):
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, len(baseline), size=(samples, len(baseline)))
    improvement = 1 - candidate[ids].mean(axis=1) / baseline[ids].mean(axis=1)
    return [float(v) for v in np.quantile(improvement, [.025, .975])]


def load_cache(path, rows, layers, experts, active):
    with np.load(path, allow_pickle=False) as data:
        ids, cached_layers, routes = data['prompt_ids'], data['layers'], data['routes']
    assert ids.tolist() == [r['id'] for r in rows]
    assert cached_layers.tolist() == layers
    assert routes.shape == (len(rows), 128, len(layers), active)
    assert 0 <= routes.min() <= routes.max() < experts
    assert (np.diff(np.sort(routes, axis=-1), axis=-1) > 0).all()
    return routes, dict(path=str(path.resolve()), sha256=sha256(path), routes_sha256=array_hash(routes),
                        shape=list(routes.shape), prompt_ids=ids.tolist())


def source_hashes():
    names = ('run_plane_wave_study.py', 'plane_wave_model.py', 'aif_model.py',
             'overhead_model.py', 'corpus_analysis.py')
    return {name: sha256(ROOT / 'scripts' / name) for name in names}


def fit_model(args, name, rows, metadata, old):
    started = time.time()
    layers, experts, active = geometry(metadata)
    dimensions = flash.tensor_dimensions(metadata)
    training = [r for r in rows if r['split'] == 'train']
    routes, receipt = load_cache(args.run_dir / name / 'analysis/train_routes.npz',
                                 training, layers, experts, active)
    expected_ids = [r['id'] for r in old['training_routing']]
    assert expected_ids == receipt['prompt_ids']
    unique_by_layer = []
    for li in range(len(layers)):
        unique_by_layer.append(np.unique(np.sort(routes[:, :, li, :].reshape(-1, active), axis=1),
                                        axis=0, return_counts=True))
    candidates, assignments, optimization = [], {}, {}
    for g in GROUPS:
        assignments[f'g{g}'], optimization[f'g{g}'] = {}, {}
        for width in WIDTHS:
            bins, stats = {}, {}
            for li, layer in enumerate(layers):
                unique, weights = unique_by_layer[li]
                bins[str(layer)], stats[str(layer)] = optimize_bins(
                    unique, weights, experts, old['mappings'][f'g{g}'][str(layer)],
                    g, width, phase_profiles(dimensions[layer], g, width),
                    args.swap_trials, args.seed + layer * 131 + g * 17 + width)
            assignments[f'g{g}'][f'w{width}'] = bins
            optimization[f'g{g}'][f'w{width}'] = stats
            for timing in TIMINGS:
                values = evaluate_routes(routes, layers, dimensions, g, width,
                                         old['mappings'][f'g{g}'], bins, timing)
                candidates.append(dict(model=name, group_chips=g, plane_width=width,
                                       timing=timing, **summarize(values, 128)))
        print(f'{name}: trained fixed plane assignments for g={g}', flush=True)
    winners = {}
    for timing in TIMINGS:
        population = [c for c in candidates if c['timing'] == timing]
        winners[timing] = {f'width{w}': min((c for c in population if c['plane_width'] == w),
                                          key=lambda c: (c['mean_ms_per_token'], c['group_chips']))
                           for w in WIDTHS}
        winners[timing]['joint'] = min(population,
                                       key=lambda c: (c['mean_ms_per_token'], -c['plane_width'], c['group_chips']))
    plan = dict(schema_version=1, model=name, seed=args.seed,
                assumptions=dict(chips=16, planes=4, page_bytes=flash.PAGE,
                                 block_pages=BLOCK_PAGES, cold_ns=flash.COLD_NS, cr_ns=flash.CR_NS,
                                 chip_ecc_bytes_per_ns=6.4, per_matrix_block_command_ns=250,
                                 host_phase_ns=2000, input_element_bytes=1, output_element_bytes=4,
                                 fixed_gate_up_stacking=True,
                                 width4_one_context_serial_experts=True,
                                 width2_width1_independent_plane_addressing_and_contexts=True,
                                 ideal_overlap='Lower bound: max(NAND lane completion, aggregate ECC bytes/BW).',
                                 retained_cr_schedule='NAND sense and ECC drain serialized; requires retention of cr-read state across drain gaps.',
                                 conservative_drain='Every synchronized wave uses a 28 us ordinary read followed by chip ECC drain.',
                                 cold_reads_counter='Logical expert/block starts; not the number of ordinary read epochs.',
                                 ordinary_read_epochs_counter='Chip epochs using ordinary read; all epochs in conservative_drain.'),
                model_sha256=metadata['sha256'], metadata_sha256=sha256(args.data / 'models' / (name + '-metadata.json')),
                corpus_sha256=sha256(args.run_dir / 'corpus/prompts.jsonl'),
                old_frozen_plan_sha256=sha256(args.old_results / (name + '.frozen_plan.json')),
                old_winner=old['winners']['learned'], mappings=old['mappings'],
                train_only_plane_assignments=assignments, optimization=optimization,
                training_cache=receipt, training_prompt_ids=receipt['prompt_ids'],
                training_steps=128, training_candidates=candidates, winners=winners,
                bin_objective='Weighted ideal-overlap flash latency, then squared occupancy; same-group capacity-preserving swaps.',
                selection_objective='Mean 128-step training routed FFN model latency including vectors, commands and phases; per timing.',
                search=dict(swap_trials_per_layer_g_width=args.swap_trials,
                            groups=list(GROUPS), widths=list(WIDTHS), seed=args.seed),
                test_routes_loaded_before_freeze=False, test_routes_used_for_selection=False,
                frozen_unix=time.time(), software_sha256=source_hashes(), fit_seconds=time.time() - started)
    config = dict(seed=args.seed, swap_trials=args.swap_trials, groups=list(GROUPS),
                  widths=list(WIDTHS), timings=list(TIMINGS), block_pages=BLOCK_PAGES,
                  assumptions=plan['assumptions'])
    plan['configuration_sha256'] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    target = args.output / (name + '.frozen_plan.json')
    write_json(target, plan)
    print(f'{name}: frozen {sha256(target)}; winners={[(t, v["joint"]["group_chips"], v["joint"]["plane_width"]) for t, v in winners.items()]}', flush=True)
    return plan


def scalar_checks(routes, layers, dimensions, plan):
    """Check every g/width/phase against independent event-level scheduler."""
    import plane_wave_model as plane
    checks = []
    picks = [(0, 0, 0), (min(1, len(routes) - 1), 31, len(layers) // 2),
             (len(routes) - 1, 127, len(layers) - 1)]
    for g in GROUPS:
        mappings = plan['mappings'][f'g{g}']
        for width in WIDTHS:
            bins = plan['train_only_plane_assignments'][f'g{g}'][f'w{width}']
            for pi, ti, li in picks:
                layer = layers[li]
                selected = routes[pi, ti, li].tolist()
                counts = counts_for(np.asarray([selected]), mappings[str(layer)], bins[str(layer)], g, width)
                profiles = phase_profiles(dimensions[layer], g, width)
                for timing in TIMINGS:
                    fast = costs_from_counts(counts, profiles, g, width, len(selected), timing)
                    phases = []
                    for profile in profiles:
                        services, waves, pages, commands = [], 0, 0, 0
                        for group in range(16 // g):
                            resident = [e for e, slot in enumerate(mappings[str(layer)]) if slot % (16 // g) == group]
                            active = [e for e in selected if e in resident]
                            masks = {e: tuple(range(bins[str(layer)][e] * width,
                                                   (bins[str(layer)][e] + 1) * width)) for e in resident}
                            for shard in range(g):
                                layout = plane.build_layout({e: int(profile['useful'][shard]) for e in resident},
                                                            policy={4:'baseline', 2:'pair', 1:'single'}[width],
                                                            masks=masks, block_pages=BLOCK_PAGES)
                                exact = plane.evaluate_chip(layout, active, common_input=profile['common'])
                                key = {'ideal_overlap':'service_ns',
                                       'retained_cr_schedule':'retained_cr_schedule_ns',
                                       'conservative_drain':'conservative_service_ns'}[timing]
                                services.append(exact[key])
                                waves += exact['read_waves']
                                pages += exact['physical_pages']
                                commands += exact['command_count']
                        phases.append((max(services, default=0), waves, pages, commands))
                    expected = dict(flash_ns=sum(p[0] for p in phases),
                                    read_waves=sum(p[1] for p in phases),
                                    physical_pages=sum(p[2] for p in phases),
                                    matrix_commands=sum(p[3] for p in phases))
                    for k, value in expected.items():
                        if int(fast[k][0]) != value:
                            raise AssertionError(f'Scalar mismatch {g}/{width}/{timing}/{layer}/{k}: {fast[k][0]} != {value}')
                    checks.append(dict(group=g, width=width, layer=layer, timing=timing,
                                       prompt_index=pi, token_index=ti))
    return checks


def baseline_checks(routes, layers, dimensions, plan):
    checks = []
    # Match all training/test tokens and all components to the established
    # exact-vector baseline, for every g (not just hand-picked scalar cases).
    for g in GROUPS:
        mapping = plan['mappings'][f'g{g}']
        bins = plan['train_only_plane_assignments'][f'g{g}']['w4']
        fast = evaluate_routes(routes, layers, dimensions, g, 4, mapping, bins)
        old = corpus.evaluate_routes(routes, layers, dimensions, g, True, mapping)
        for k in (*COMPONENTS, 'model_ns'):
            np.testing.assert_array_equal(fast[k], old[k])
        checks.append(dict(group=g, all_tokens_all_components_equal=True,
                           routing_events=int(np.prod(routes.shape[:-1]))))
    return checks


def flatten_values(prefix, value):
    out = {prefix + '__' + k: v for k, v in value.items() if k != 'phases'}
    for phase, components in value['phases'].items():
        out.update({prefix + '__' + phase + '__' + k: v for k, v in components.items()})
    return out


def evaluate_model(args, name, rows, metadata, plan):
    frozen = args.output / (name + '.frozen_plan.json')
    frozen_hash = sha256(frozen)
    assert not plan['test_routes_loaded_before_freeze'] and not plan['test_routes_used_for_selection']
    assert plan['software_sha256'] == source_hashes(), 'Software changed after training freeze'
    heldout = [r for r in rows if r['split'] == 'test']
    layers, experts, active = geometry(metadata)
    dimensions = flash.tensor_dimensions(metadata)
    routes, receipt = load_cache(args.run_dir / name / 'analysis/test_routes.npz',
                                 heldout, layers, experts, active)
    assert not set(receipt['prompt_ids']) & set(plan['training_prompt_ids'])
    summary, per_prompt, sensitivity, candidates_out, arrays = [], [], [], [], {}
    scalar = scalar_checks(routes, layers, dimensions, plan)
    baseline = baseline_checks(routes, layers, dimensions, plan)
    old_g = plan['old_winner']['group_chips']
    legacy_overlap = evaluate_routes(routes, layers, dimensions, old_g, 4,
                                     plan['mappings'][f'g{old_g}'],
                                     plan['train_only_plane_assignments'][f'g{old_g}']['w4'],
                                     'ideal_overlap')
    for timing in TIMINGS:
        specs = {'old_frozen_baseline': (old_g, 4)}
        specs.update({f'width{w}_trainselected': (plan['winners'][timing][f'width{w}']['group_chips'], w)
                      for w in WIDTHS})
        specs.update({f'width{w}_at_old_g': (old_g, w) for w in (2, 1)})
        joint = plan['winners'][timing]['joint']
        specs['selected_joint'] = (joint['group_chips'], joint['plane_width'])
        # Evaluate all candidate cells for descriptive held-out heatmaps, but
        # use only winners already committed in the training freeze.
        values = {}
        for g in GROUPS:
            for width in WIDTHS:
                values[(g, width)] = evaluate_routes(
                    routes, layers, dimensions, g, width, plan['mappings'][f'g{g}'],
                    plan['train_only_plane_assignments'][f'g{g}'][f'w{width}'], timing)
                for window in (32, 128):
                    candidates_out.append(dict(model=name, timing=timing, group_chips=g, plane_width=width,
                                               window=window, **summarize(values[(g, width)], window)))
        reference = values[specs['old_frozen_baseline']]
        for label, spec in specs.items():
            value = values[spec]
            arrays.update(flatten_values(timing + '__' + label, value))
            for window in (32, 128):
                base_prompt = reference['model_ns'][:, :window].mean(axis=1)
                prompt = value['model_ns'][:, :window].mean(axis=1)
                gain = float(1 - prompt.mean() / base_prompt.mean())
                summary.append(dict(model=name, timing=timing, policy=label,
                                    group_chips=spec[0], plane_width=spec[1], window=window,
                                    prompts=len(heldout), **summarize(value, window),
                                    improvement_vs_old_frozen=gain,
                                    legacy_overlap_baseline_ms=float(legacy_overlap['model_ns'][:, :window].mean() / 1e6),
                                    improvement_vs_legacy_overlap=float(1 - prompt.mean() /
                                                                         legacy_overlap['model_ns'][:, :window].mean()),
                                    improvement_95ci=paired_interval(base_prompt, prompt,
                                                                   seed=args.seed, samples=args.bootstrap)))
                for i, row in enumerate(heldout):
                    per_prompt.append(dict(model=name, timing=timing, policy=label,
                                           group_chips=spec[0], plane_width=spec[1],
                                           id=row['id'], task=row['task'], length_bucket=row['length_bucket'],
                                           window=window, mean_ms_per_token=float(prompt[i] / 1e6),
                                           baseline_ms_per_token=float(base_prompt[i] / 1e6)))
            # Charge identical metrics to both baseline and candidate. Extra
            # context setup is a setup per additional simultaneous plane bin.
            for context_ns in (0, 250, 1000, 5000, 20000):
                for plane_command_ns in (0, 50, 250, 1000):
                    base = reference['model_ns'] + context_ns * reference['extra_contexts'] + plane_command_ns * reference['nand_read_commands']
                    candidate = value['model_ns'] + context_ns * value['extra_contexts'] + plane_command_ns * value['nand_read_commands']
                    sensitivity.append(dict(model=name, timing=timing, policy=label,
                                            group_chips=spec[0], plane_width=spec[1],
                                            extra_context_setup_ns=context_ns,
                                            extra_nand_command_ns=plane_command_ns,
                                            window=128, mean_ms_per_token=float(candidate.mean() / 1e6),
                                            baseline_ms_per_token=float(base.mean() / 1e6),
                                            improvement=float(1 - candidate.mean() / base.mean())))
        del values
    np.savez_compressed(args.output / (name + '.token_costs.npz'),
                        prompt_ids=np.array(receipt['prompt_ids']), **arrays)
    validation = dict(model=name, frozen_plan_sha256=frozen_hash, test_cache=receipt,
                      test_routes_loaded_unix=time.time(), fixed_plan_for_prefix32_and_full128=True,
                      baseline_full_array_checks=baseline, scalar_scheduler_checks=scalar,
                      test_routes_used_for_training_or_selection=False,
                      old_nvme_kernel_supports_new_timing=False)
    write_json(args.output / (name + '.validation.json'), validation)
    assert sha256(frozen) == frozen_hash
    print(f'{name}: held-out evaluation and {len(scalar)} scalar checks complete', flush=True)
    return summary, per_prompt, sensitivity, candidates_out, validation


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=DATA / 'runs/corpus-300-v1')
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--old-results', type=Path, default=ROOT / 'results/corpus-300-v1')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/plane-wave-v1')
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    parser.add_argument('--swap-trials', type=int, default=256)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--stage', choices=('fit', 'test', 'both'), default='both')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in (args.run_dir / 'corpus/prompts.jsonl').read_text().splitlines()]
    assert Counter(r['split'] for r in rows) == {'train': 200, 'test': 100}
    assert len({r['id'] for r in rows}) == 300
    summaries, per_prompt, sensitivities, candidates, validations, training = [], [], [], [], [], []
    started = time.time()
    for name in args.models:
        metadata = json.loads((args.data / 'models' / (name + '-metadata.json')).read_text())
        old = json.loads((args.old_results / (name + '.frozen_plan.json')).read_text())
        assert old['corpus_sha256'] == sha256(args.run_dir / 'corpus/prompts.jsonl')
        if args.stage in ('fit', 'both'):
            plan = fit_model(args, name, rows, metadata, old)
        else:
            plan = json.loads((args.output / (name + '.frozen_plan.json')).read_text())
        training.extend(plan['training_candidates'])
        if args.stage in ('test', 'both'):
            s, p, c, h, v = evaluate_model(args, name, rows, metadata, plan)
            summaries.extend(s); per_prompt.extend(p); sensitivities.extend(c)
            candidates.extend(h); validations.append(v)
            # Useful partial artifacts if a long run is interrupted.
            write_json(args.output / 'summary.json', summaries)
            write_csv(args.output / 'per_prompt.csv', per_prompt)
            write_csv(args.output / 'sensitivity.csv', sensitivities)
            write_json(args.output / 'heldout_candidates.json', candidates)
            write_json(args.output / 'validation.json', validations)
    write_json(args.output / 'training_candidates.json', training)
    manifest = dict(schema_version=1, date='2026-10-30', timezone='Asia/Taipei',
                    command=[sys.executable, *sys.argv], models=args.models, seed=args.seed,
                    corpus_sha256=sha256(args.run_dir / 'corpus/prompts.jsonl'),
                    configuration={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    software_sha256=source_hashes(), elapsed_seconds=time.time() - started,
                    bootstrap=dict(unit='whole prompt; paired', resamples=args.bootstrap, seed=args.seed),
                    timing_claims=dict(ideal_overlap='Conditional analytical lower bound; not guaranteed schedulable.',
                                       retained_cr_schedule='Synchronous sense epochs plus serialized ECC drain; requires retention of cr-read state across gaps.',
                                       conservative_drain='Every epoch uses a 28 us ordinary read followed by ECC drain; independent plane addressing and multiple contexts still required.',
                                       measured_hardware=False, nvme_kernel_supports_new_timing=False),
                    limitations=['Real routes from existing synthetic 300-prompt corpus; no new model inference.',
                                 'Fixed INT8 dimensions, not original GGUF quantized weight byte size.',
                                 'No prefill, attention, routing, shared experts, combine, DRAM cache, GC or channel bus timing for NAND command headers.',
                                 'Training mask search is heuristic; test routes never select masks, g, or width.',
                                 'Cache SHA-256 and original frozen-plan routing hashes are retained; full raw trace validation inherited from original corpus study.'])
    manifest['configuration_sha256'] = hashlib.sha256(
        json.dumps(manifest['configuration'], sort_keys=True).encode()).hexdigest()
    write_json(args.output / 'manifest.json', manifest)


if __name__ == '__main__':
    main()
