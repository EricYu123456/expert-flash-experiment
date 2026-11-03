#!/usr/bin/env python3
"""Structural screen of equal-grain g8 copies on healthy ordinary-read chips.

No optimization or test data: prove the NAND dominance construction, materialize
an input-reuse counterexample, and bound residual costs using 32 train events.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from research_20261102_projection_study import evaluate

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/research-2026-11-02'
DEST = BASE / 'round2/clean-replica'
PAGE = 16384


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def down_counterexample(columns, tile_rows, wave_count):
    """K8, four selected experts/group; exact makespan 4 tiles on all chips.

    At eight chips, four copied adjacent tiles from one expert replace the four
    original distinct experts.  Opposite chips retain their four original jobs.
    Every channel thus has five expert-chip inputs instead of eight.
    """
    jobs = []
    for expert in range(8):
        half = expert // 4
        for tile in range(8):
            source = half * 8 + tile
            copied = tile < 4 if half == 0 else tile >= 4
            if copied:
                owner = 12 + expert if half == 0 else expert - 4
            else:
                owner = source
            jobs.append(dict(expert=expert, tile=tile, source=source, owner=owner,
                             copied=copied, row_start=tile * tile_rows,
                             rows=tile_rows, columns=columns))
    partial = assignment_cost(jobs, columns, tile_rows, wave_count, coalesce=False)
    merged = assignment_cost(jobs, columns, tile_rows, wave_count, coalesce=True)
    original_jobs = [dict(j, owner=j['source']) for j in jobs]
    whole = assignment_cost(original_jobs, columns, tile_rows, wave_count, coalesce=False)
    assert partial['flash_ns'] == whole['flash_ns']
    assert partial['control_ns'] == whole['control_ns']
    assert partial['output_ns'] == whole['output_ns']
    assert partial['input_ns'] < whole['input_ns']
    assert merged['flash_ns'] == whole['flash_ns']
    assert merged['control_ns'] < whole['control_ns']
    for expert in range(8):
        assert sorted(j['tile'] for j in jobs if j['expert'] == expert) == list(range(8))
    copies = [j for j in jobs if j['copied']]
    per_source_sets = [[sorted({j['expert'] for j in copies if j['source'] == 8 * h + p})
                        for p in range(8)] for h in range(2)]
    smallest = [min(sets, key=len) for sets in per_source_sets]
    assert smallest == [[], []]
    return dict(active_experts=8, selected_experts_per_original_group=4,
                logical_tiles=64, copied_tiles=len(copies),
                makespan_lower_bound_tiles=4, exact_integral_assignment=jobs,
                minimum_source_copy_expert_sets=smallest,
                native_command_per_tile=dict(partial=partial, whole_groups=whole),
                row_run_coalescing=dict(partial=merged, whole_groups=whole),
                native_interface='Each job is a whole-row shard. Optional coalescing '
                    'only combines four adjacent copied output-row ranges of the same '
                    'expert into one dense GEMV. No partial dot sums or weight changes.',
                scope='This is a numerical structural counterexample, not a model route '
                    'frequency or a learned layout. Unused model experts can retain '
                    'their original one-copy placement.',
                theorem_result='NAND dominance holds; full service termwise dominance '
                    'fails if same-expert input reuse or adjacent-row command coalescing is allowed.')


def assignment_cost(jobs, columns, tile_rows, waves, coalesce):
    counts = [0] * 16
    experts = [set() for _ in range(16)]
    row_ranges = defaultdict(list)
    for job in jobs:
        c, e = job['owner'], job['expert']
        counts[c] += 1
        experts[c].add(e)
        row_ranges[c, e].append((job['row_start'], job['row_start'] + tile_rows))
    commands = len(jobs)
    if coalesce:
        commands = 0
        for intervals in row_ranges.values():
            end = None
            for start, stop in sorted(intervals):
                if start != end:
                    commands += 1
                end = stop
    pages = (tile_rows * columns + PAGE - 1) // PAGE
    assert (pages + 3) // 4 == waves
    host_input = 8 * columns
    inputs = [len(e) * columns for e in experts]
    outputs = [n * tile_rows * 4 for n in counts]
    flash = max(max(counts) * waves * 28000, max(counts) * pages * 2560)
    control = commands * 250
    input_ns = max((host_input + 7) // 8,
                   max((inputs[c] + inputs[c + 8] + 1) // 2 for c in range(8)))
    output_ns = max((sum(outputs) + 7) // 8,
                    max((outputs[c] + outputs[c + 8] + 1) // 2 for c in range(8)))
    return dict(chip_tile_counts=counts, chip_unique_expert_inputs=[len(e) for e in experts],
                channel_input_bytes=[inputs[c] + inputs[c + 8] for c in range(8)],
                native_commands=commands, flash_ns=flash, control_ns=control,
                input_ns=input_ns, output_ns=output_ns, host_ns=2000,
                model_ns=flash + control + input_ns + output_ns + 2000,
                assumes_input_reuse_between_same_expert_runs=True)


def ideal_g8(phase, active):
    rows, columns = phase['total_rows'] // 8, phase['columns']
    assert phase['total_rows'] % 8 == 0 and active % 2 == 0
    pages = (rows * columns + PAGE - 1) // PAGE
    waves = (pages + 3) // 4
    n = active // 2
    flash = max(n * waves * 28000, n * pages * 2560)
    control = active * 8 * 250
    inputs = columns if phase['common'] else n * columns
    host_input = columns if phase['common'] else active * columns
    input_ns = max((host_input + 7) // 8, inputs)
    host_output = active * phase['total_rows'] * 4
    output_ns = max((host_output + 7) // 8, n * rows * 4)
    return flash + control + input_ns + output_ns + 2000


def model_bounds(name):
    if name == 'granite4tiny':
        plan_path = BASE / 'round2/geometry-study/granite4tiny.frozen.json'
        raw = plan_path.read_bytes()
        plan = json.loads(raw)
        routes_path = Path(plan['training_routes'])
        row_key, target = 'rows', 'gate_up'
    else:
        plan_path = BASE / 'projection-study' / (name + '.frozen_plan.json')
        raw = plan_path.read_bytes()
        plan = json.loads(raw)
        routes_path = Path(plan['train_path'])
        row_key, target = 'chosen_rows', 'down'
    with np.load(routes_path) as data:
        ids = data['prompt_ids'][:32].tolist()
        assert len(ids) == 32 and all(p.startswith('train_') for p in ids)
        selected = data['routes'][:32, 0].copy()
    layers, active = plan['layers'], plan['active']
    assert selected.shape == (32, len(layers), active)
    assert len(plan['phases']) == len(layers) * 2
    baseline = np.zeros(32, dtype=np.int64)
    unlimited_copy_oracle = np.zeros(32, dtype=np.int64)
    input_gap = 0
    control_gap = 0
    phase_records = []
    for phase in plan['phases']:
        li = layers.index(phase['layer'])
        values = evaluate(selected[:, li], np.asarray(phase[row_key]),
                          phase['columns'], phase['common'])
        baseline += values['model_ns']
        if phase['name'] != target:
            unlimited_copy_oracle += values['model_ns']
            continue
        oracle = ideal_g8(phase, active)
        unlimited_copy_oracle += np.minimum(values['model_ns'], oracle)
        columns = phase['columns']
        # This is an intentionally optimistic lower bound on partial input cost:
        # retain the mandatory PCIe transfer and grant zero ONFI input cost.
        host_bytes = columns if phase['common'] else active * columns
        minimum_input = (host_bytes + 7) // 8
        whole_max_input = columns if phase['common'] else (active * columns + 1) // 2
        input_gap += whole_max_input - minimum_input
        # Out of scope for the one-tile/one-command dominance proof: grant one
        # full expert GEMV per selected expert, a loose coalescing upper bound.
        coalescing_gap = active * 7 * 250
        control_gap += coalescing_gap
        phase_records.append(dict(layer=phase['layer'], phase=target,
            tile_rows=phase['total_rows'] // 8, columns=columns,
            tile_pages=(phase['total_rows'] // 8 * columns + PAGE - 1) // PAGE,
            tile_waves=(phase['total_rows'] // 8 * columns + 65535) // 65536,
            unlimited_copy_ideal_ns=oracle,
            maximum_additional_input_gain_ns=whole_max_input - minimum_input,
            loose_additional_coalescing_gain_ns=coalescing_gap))
    mean_baseline = float(baseline.mean())
    mean_oracle = float(unlimited_copy_oracle.mean())
    return dict(model=name, target_phase=target, sample='first 32 training prompts, decode step 0',
                prompt_ids=ids, routes_path=str(routes_path), routes_sha256=digest(routes_path),
                plan_path=str(plan_path), plan_snapshot_sha256=hashlib.sha256(raw).hexdigest(),
                active_experts=active, layers=len(layers), experts=plan['experts'],
                strong_frozen_ffn_mean_ns=mean_baseline,
                unlimited_copy_with_phase_fallback_oracle_ns=mean_oracle,
                unlimited_copy_headroom_fraction=1 - mean_oracle / mean_baseline,
                partial_over_whole_input_only_maximum_ns=input_gap,
                partial_over_whole_input_only_fraction_of_frozen=input_gap / mean_baseline,
                partial_over_whole_input_only_fraction_of_optimistic_whole=input_gap / mean_oracle,
                out_of_scope_coalescing_plus_input_loose_maximum_ns=control_gap + input_gap,
                out_of_scope_coalescing_plus_input_fraction_of_frozen=(control_gap + input_gap) / mean_baseline,
                coalescing_bound_assumption='At least one native command per selected '
                    'expert; no cross-expert row stacking/co-loading. This is not a '
                    'universal bound once arbitrary command regrouping is permitted.',
                phases=phase_records,
                caveat='Only routed FFN service is modeled. Not full-model execution, '
                    'not a quality experiment, not a 25/50% capacity-aware optimal solver. '
                    'The full-copy oracle is mechanism headroom, not new partial-placement gain.')


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    proof = dict(
        claim='Equal-grain g8 partial copies have no NAND-makespan advantage over some '
              'whole-expert opposite-group replication plan of no greater useful/page copy budget.',
        assumptions=['16 homogeneous healthy chips in two groups of eight',
            'each expert has exactly one equal-size complete-row tile on every chip of its original group',
            'all phase tiles have identical pages, waves and output row count',
            'no tile is divided, copied weights remain immutable and every selected tile executes once',
            'copy destinations arbitrary; multiple alternate copies are allowed and charged',
            'integer waves and integral tile assignment, no fractional multiwave splitting'],
        construction='For each original group h independently, select a source j_h minimizing '
            'the number of distinct experts with any tile copy away from j_h. Replicate each '
            'such expert completely to the opposite group. If its set is S_h and C_h copies '
            'were stored for source group h, 8|S_h| <= C_h. Thus both group-wise useful bytes '
            'and physical pages of the new copy inventory are no greater.',
        lower_bound='For every selected route, let n_h be the selected experts originally in '
            'group h and s_h the selected experts in S_h. The original partial plan cannot '
            'move the n_h-s_h mandatory jobs off source j_h, and must execute 8K total jobs. '
            'Its maximum per-chip tile count is at least L=max(ceil(K/2),n_0-s_0,n_1-s_1).',
        achiever='Whole expert copies allow each selected copied expert to choose either '
            'group. Choose the group0 expert count in [n_0-s_0, K-(n_1-s_1)] nearest K/2. '
            'This is an integral assignment with maximum L on all eight chips/group. '
            'Exactly the same sets and proof work for every route, not only training routes.',
        cost_terms={'NAND':'Dominated pointwise; multiply tile count by its unchanged wave service.',
            'output':'Whole-group assignment gives exactly K tiles per channel-pair, hence '
                     'minimum maximum channel output and the same mandatory PCIe output.',
            'commands':'Equal only when one native command is charged per original tile. '
                       'Adjacent whole-row coalescing is a separate, legal controller operation.',
            'host':'Same selected experts, input/output tensors and GU->host activation->down barriers.',
            'input':'Full termwise dominance is false with same-expert input reuse after tile '
                    'colocation. Counterexample supplied. The possible input gain is tiny here.'},
        capacity_qualification='The theorem covers useful bytes and pages, not arbitrary '
            'per-chip capacity caps or reserved block stripes. Whole copies can distribute '
            'copies differently and block rounding can change. A strong whole-copy baseline '
            'must be allowed to allocate its budget across layers and pack globally; imposing '
            'a fixed fraction in each layer can introduce artificial reservation losses.',
        applicability='Does not cover unequal row sizes, heterogeneous chips, changing command '
            'granularity, online background, or a different pre-existing base placement.',
        verdict='No-go for a new g8 partial-copy NAND balancing claim. Under one tile per '
            'native command, remaining gain is bounded by the input term. Do not claim a '
            'universal full-cost dominance theorem or no-go for row-run coalescing.')
    counters = {name: down_counterexample(*args) for name, args in
                [('granite', (512, 128, 1)), ('qwen3', (768, 256, 3))]}
    models = [model_bounds(name) for name in ('granite', 'qwen3', 'granite4tiny')]
    value = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                 source_sha256=digest(Path(__file__)),
                 source_hashes={str(p): digest(p) for p in [Path(__file__),
                    ROOT / 'scripts/research_20261102_projection_study.py',
                    ROOT / 'kernel/aif.c', ROOT / 'kernel/aif.h',
                    ROOT / 'docs/vm-aif-abi.md']},
                 test_routes_opened=False, weights_or_vm_modified=False,
                 protocol='Structural no-go screen; no partial-copy optimizer was run '
                     'after the pointwise NAND dominance proof.',
                 proof=proof, model_bounds=models, input_reuse_counterexamples=counters)
    save(DEST / 'summary.json', value)
    for m in models:
        print(m['model'], 'fullcopy_oracle_headroom_percent=',
              100 * m['unlimited_copy_headroom_fraction'],
              'partial_input_extra_upper_percent=',
              100 * m['partial_over_whole_input_only_fraction_of_optimistic_whole'])
    for name, c in counters.items():
        fixed = c['native_command_per_tile']
        coalesced = c['row_run_coalescing']
        print(name, 'counterexample_input_gain_ns=', fixed['whole_groups']['model_ns'] - fixed['partial']['model_ns'],
              'with_legal_row_coalescing_gain_ns=', coalesced['whole_groups']['model_ns'] - coalesced['partial']['model_ns'])


if __name__ == '__main__':
    main()
