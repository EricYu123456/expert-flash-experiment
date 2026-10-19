#!/usr/bin/env python3
"""Fit capacity-preserving static MoE placement with prompt-level held-out tests.

Two prompts train each fold, the third tests it. Swaps change fixed expert-to-slot
placement only. Original routing files, selected logical IDs, and weights stay fixed.
Reported timings are flash-only; mappings can feed a separate overhead model.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np

import aif_model as model

GROUPS = (1, 2, 4, 8, 16)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(counts):
    return int(counts.max(axis=1).sum()), int(np.square(counts).sum())


def optimize(routes, expert_count, group_chips, trials, seed):
    """Accept improving expert-slot swaps; group capacity never changes."""
    placement = list(range(expert_count))
    ngroups = model.CHIPS // group_chips
    if ngroups == 1:
        return placement, dict(attempted=0, accepted=0)
    incidence = np.zeros((len(routes), expert_count), dtype=np.int16)
    for i, experts in enumerate(routes):
        incidence[i, experts] = 1
    counts = np.zeros((len(routes), ngroups), dtype=np.int16)
    for e in range(expert_count):
        counts[:, e % ngroups] += incidence[:, e]
    initial = current = score(counts)
    rng, accepted = random.Random(seed), 0
    for _ in range(trials):
        a, b = rng.sample(range(expert_count), 2)
        ga, gb = placement[a] % ngroups, placement[b] % ngroups
        if ga == gb:
            continue
        candidate = counts.copy()
        delta = incidence[:, b] - incidence[:, a]
        candidate[:, ga] += delta
        candidate[:, gb] -= delta
        proposed = score(candidate)
        if proposed < current:
            placement[a], placement[b] = placement[b], placement[a]
            counts, current = candidate, proposed
            accepted += 1
    assert sorted(placement) == list(range(expert_count))
    assert current <= initial
    return placement, dict(attempted=trials, accepted=accepted,
                           initial_peak_sum=initial[0], final_peak_sum=current[0],
                           initial_square_sum=initial[1], final_square_sum=current[1])


def phases(dims, stacked):
    if stacked:
        return [[(dims['gate'][0] + dims['up'][0], dims['up'][1])], [dims['down']]]
    return [[dims['gate']], [dims['up']], [dims['down']]]


def layer_coefficient(dims, group_chips, stacked):
    # Homogeneous expert streams give an exact coefficient for the largest
    # selected-expert count in any fixed group. Source all costs from the model.
    return sum(max(d.service_ns for d in model.phase_demand([0], phase, group_chips))
               for phase in phases(dims, stacked))


def event_ns(event, placement, group_chips, coefficient):
    counts = [0] * (model.CHIPS // group_chips)
    for e in event['expert_ids']:
        counts[placement[e] % len(counts)] += 1
    return max(counts) * coefficient


def layer_sum(events, placements, group_chips, coefficients):
    return sum(event_ns(event, placements[event['layer']], group_chips,
                        coefficients[event['layer']][group_chips]) for event in events)


def evaluate(events, dimensions, placements, group_by_layer, coefficients):
    tokens, useful = {}, 0
    for event in events:
        layer = event['layer']
        ns = event_ns(event, placements[layer], group_by_layer[layer],
                      coefficients[layer][group_by_layer[layer]])
        key = event['prompt'], event['token_index']
        tokens[key] = tokens.get(key, 0) + ns
        useful += len(event['expert_ids']) * sum(r*c for r, c in dimensions[layer].values())
    ns = sum(tokens.values())
    latencies = sorted(tokens.values())
    return dict(model_ns=ns, decode_tokens=len(tokens), routing_records=len(events),
                useful_bytes=useful, ffn_ms_per_token=ns / len(tokens) / 1e6,
                ffn_p50_ms=latencies[math.ceil(.50*len(tokens))-1]/1e6,
                ffn_p95_ms=latencies[math.ceil(.95*len(tokens))-1]/1e6,
                useful_bandwidth_fraction=useful*1e9/(model.CHIPS*model.CHIP_BW*ns))


def verify_fast_model(events, dimensions, placements, group_by_layer, coefficients, stacked):
    # Check one real held-out route per layer against full phase_demand. This
    # validation never affects the optimizer or training-only group selection.
    checked = set()
    for event in events:
        layer = event['layer']
        if layer in checked:
            continue
        checked.add(layer)
        group = group_by_layer[layer]
        slots = [placements[layer][e] for e in event['expert_ids']]
        exact = sum(max(d.service_ns for d in model.phase_demand(slots, phase, group))
                    for phase in phases(dimensions[layer], stacked))
        fast = event_ns(event, placements[layer], group, coefficients[layer][group])
        assert exact == fast, (layer, group, exact, fast)


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--swap-trials', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if not 0 <= args.swap_trials <= 100:
        parser.error('--swap-trials must be in [0, 100]')
    source = json.loads(args.run_manifest.read_text())
    metadata_path = Path(source['metadata'])
    trace_paths = [Path(path) for path in source['traces']]
    metadata = json.loads(metadata_path.read_text())
    dimensions = model.tensor_dimensions(metadata)
    architecture = metadata['metadata']['general.architecture']
    expert_count = metadata['metadata'][architecture + '.expert_count']
    events = model.load_events(trace_paths)
    prompts = sorted({event['prompt'] for event in events})
    if len(prompts) != 3:
        raise ValueError('Exactly three distinct prompt traces are required for three-fold evaluation')
    layers = sorted(dimensions)
    identity = {layer: list(range(expert_count)) for layer in layers}
    for event in events:
        if not event['expert_ids'] or event['layer'] not in dimensions:
            raise ValueError('Missing expert route or tensor dimensions')
        if any(e < 0 or e >= expert_count for e in event['expert_ids']):
            raise ValueError('Logical expert ID outside metadata bounds')
    coefficients = {layout: {layer: {g: layer_coefficient(dimensions[layer], g, layout == 'stacked')
                                            for g in GROUPS} for layer in layers}
                    for layout in ('separate', 'stacked')}
    rows, folds, optimization = [], [], []
    for fold_index, heldout in enumerate(prompts):
        training = [event for event in events if event['prompt'] != heldout]
        testing = [event for event in events if event['prompt'] == heldout]
        assert {e['prompt'] for e in training}.isdisjoint({heldout})
        train_by_layer = {layer: [e for e in training if e['layer'] == layer] for layer in layers}
        learned = {}
        for g in GROUPS:
            learned[g] = {}
            for layer in layers:
                routes = [e['expert_ids'] for e in train_by_layer[layer]]
                if not routes:
                    raise ValueError(f'No training routes for layer {layer}')
                permutation, info = optimize(routes, expert_count, g, args.swap_trials,
                                              args.seed + fold_index*100000 + layer*100 + g)
                learned[g][layer] = permutation
                optimization.append(dict(heldout_prompt=heldout, layer=layer, group_chips=g, **info))
        fold = dict(heldout_prompt=heldout,
                    train_prompts=sorted({e['prompt'] for e in training}),
                    placements={f'g{g}': {str(layer): learned[g][layer] for layer in layers}
                                for g in GROUPS}, layout_choices={})
        for layout in ('separate', 'stacked'):
            coeff = coefficients[layout]
            candidates, choices = [], {}
            for name, mappings in [('modulo', {g: identity for g in GROUPS}), ('learned', learned)]:
                totals = {g: layer_sum(training, mappings[g], g, coeff) for g in GROUPS}
                global_g = min(GROUPS, key=lambda g: (totals[g], g))
                layer_g = {layer: min(GROUPS, key=lambda g: (
                    layer_sum(train_by_layer[layer], mappings[g], g, coeff), g)) for layer in layers}
                choices[name + '_global_group_chips'] = global_g
                choices[name + '_layer_group_chips'] = {str(k): v for k, v in layer_g.items()}
                for g in GROUPS:
                    candidates.append((name + f'_g{g}', mappings[g], {layer: g for layer in layers}, g))
                candidates.append((name + '_training_selected_global_g', mappings[global_g],
                                   {layer: global_g for layer in layers}, global_g))
                candidates.append((name + '_training_selected_layer_g',
                                   {layer: mappings[layer_g[layer]][layer] for layer in layers}, layer_g, 'adaptive'))
            fold['layout_choices'][layout] = choices
            for policy, placement, group_by_layer, group_label in candidates:
                verify_fast_model(testing, dimensions, placement, group_by_layer, coeff, layout == 'stacked')
                for split, selected in [('train', training), ('heldout', testing)]:
                    result = evaluate(selected, dimensions, placement, group_by_layer, coeff)
                    rows.append(dict(heldout_prompt=heldout, split=split, layout=layout,
                                     policy=policy, group_chips=group_label, **result))
        folds.append(fold)
    output_dir = args.output_dir or args.run_manifest.parent / 'learned_placement'
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, records in [('cross_validation.csv', rows), ('optimization.csv', optimization)]:
        with (output_dir / name).open('w', newline='') as output:
            fields = list(dict.fromkeys(key for record in records for key in record))
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
    mappings_path = output_dir / 'mappings.json'
    mappings_path.write_text(json.dumps(dict(
        schema_version=1, placement_semantics='placements[g][layer][logical_expert_id] -> placement_slot',
        group_rule='placement_slot % (16 / group_chips)',
        flash_coefficient_ns=coefficients,
        fit_cost='Flash-only; group choices must be reselected on training routes when overhead assumptions change',
        folds=folds), indent=2) + '\n')
    manifest = dict(
        source_run_manifest=str(args.run_manifest.resolve()),
        source_run_manifest_sha256=digest(args.run_manifest),
        metadata=str(metadata_path.resolve()), metadata_sha256=digest(metadata_path), model_sha256=metadata['sha256'],
        traces=[dict(path=str(path.resolve()), sha256=digest(path)) for path in trace_paths],
        script_sha256=digest(Path(__file__)), timing_model_sha256=digest(Path(model.__file__)),
        python_version=sys.version, numpy_version=np.__version__,
        elapsed_seconds=time.monotonic()-started, folds=len(folds), prompt_names=prompts,
        seed=args.seed, swap_trials_per_layer_per_g=args.swap_trials,
        algorithm='Deterministic local swaps of fixed placement slots, starting from modulo placement',
        objective='Lexicographically minimize training sum(max selected experts/group), then sum squared group counts',
        training_scope='Two complete prompt traces per fold; held-out prompt never used for swaps or group selection',
        capacity_constraint='Every mapping is a permutation of all expert slots; fixed per-group expert capacity is preserved',
        cost_source='aif_model.phase_demand; homogeneous same-shaped experts yield exact per-layer occupancy coefficient',
        layout_options=['Separate gate/up/down', 'Physically stacked gate+up followed by down'],
        cold_per_block=1, block_lsb_pages=192,
        controls=['Modulo at every fixed g', 'Learned at every fixed g',
                  'Training-selected global g for modulo and learned',
                  'Training-selected per-layer g for modulo and learned'],
        limitations=['Flash-only cost excludes command, vector communication, and host work.',
                     '100 swap samples are a heuristic search, not a global optimum.',
                     'Only three prompts; folds are correlated and no universal workload claim is justified.',
                     'Per-layer g can alter storage allocation; capacity and block fragmentation still need hardware modeling.'])
    (output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    summary = []
    for layout in ('separate', 'stacked'):
        for policy in ('modulo_training_selected_global_g', 'learned_training_selected_global_g',
                       'modulo_training_selected_layer_g', 'learned_training_selected_layer_g'):
            selected = [r for r in rows if r['split'] == 'heldout' and r['layout'] == layout and r['policy'] == policy]
            summary.append(dict(layout=layout, policy=policy,
                                mean_heldout_ffn_ms_per_token=sum(r['ffn_ms_per_token'] for r in selected)/len(selected),
                                per_fold_ffn_ms_per_token={r['heldout_prompt']: r['ffn_ms_per_token'] for r in selected}))
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(dict(output_dir=str(output_dir), elapsed_seconds=manifest['elapsed_seconds'], summary=summary), indent=2))


if __name__ == '__main__':
    main()
