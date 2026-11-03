#!/usr/bin/env python3
"""Read-count-derived disturbance work bounds on frozen physical MoE layouts.

Sensitivity only: no device RBER calibration, stochastic retry injection, new
decoder, changed quantization, replication, or additional NAND command assumed.
The fluid WL-reclaim and fractional-packing results are unattainable lower
bounds, not implementations of reclaim in the current AiF ABI.
"""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np

from materialize_plan import WaveAllocator


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/research-2026-11-02/projection-study'
DEST = ROOT / 'results/research-2026-11-02/round2/disturb-oracle'
WL = 192
PLANES = 4
READ_US = 28


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def stress(counts, alpha):
    """Non-target base stress plus adjacent excess, independently per block."""
    result = counts.sum(axis=-1, keepdims=True) - counts
    neighbors = np.zeros_like(counts, dtype=np.float64)
    neighbors[..., 1:] += counts[..., :-1]
    neighbors[..., :-1] += counts[..., 1:]
    return result + (alpha - 1) * neighbors


def fractional_base_bound(values, blocks, length):
    """Relax page grouping to fractional blocks with occupancy n in [1,L].

    A page of read rate f assigned to an n-page block contributes (n-1)*f
    to total non-target stress. Each page consumes 1/n blocks. Relaxing
    integral grouping gives min sum((n-1)*f) with sum(1/n)<=B. In x=1/n
    this is convex; n=clip(sqrt(mu/f),1,L). Zero-rate stored pages still
    consume 1/L blocks. This lower bound allows arbitrary fragmentation.
    """
    values = np.sort(np.asarray(values, dtype=np.float64))
    if not len(values):
        return 0.0
    assert len(values) <= blocks * length
    if len(values) == blocks * length:
        return float((length - 1) * values.sum())
    positive = values[values > 0]
    if not len(positive):
        return 0.0
    zero_blocks = (len(values) - len(positive)) / length
    if zero_blocks + len(positive) <= blocks:
        return 0.0
    lo, hi = 0.0, float(positive[-1] * length * length)
    for _ in range(64):
        mu = (lo + hi) / 2
        occupancy = np.clip(np.sqrt(mu / positive), 1, length)
        if zero_blocks + np.sum(1 / occupancy) > blocks:
            lo = mu
        else:
            hi = mu
    # Use the feasible solution of the convex relaxation. The dual certificate
    # avoids numerical overestimating the lower bound at the boundary.
    mu = (lo + hi) / 2
    occupancy = np.clip(np.sqrt(mu / positive), 1, length)
    dual = np.sum((occupancy - 1) * positive + mu / occupancy)
    dual += mu * zero_blocks - mu * blocks
    return max(0.0, float(dual) - 1e-8 * max(1.0, float(values.sum())))


def neighbor_bound(values, blocks, length):
    """Relaxed minimum sum f[source]*valid_neighbor_count on B paths."""
    values = np.sort(np.asarray(values, dtype=np.float64))
    n = len(values)
    if not n:
        return 0.0
    holes = blocks * length - n
    if not holes:
        # Full blocks: exactly 2B endpoint pages have degree 1, all others 2.
        return float(2 * values.sum() - values[-min(n, 2 * blocks):].sum())
    # B+H valid runs at most => at least N-B-H valid adjacency edges.
    # Assign all necessary degrees to the lowest-rate pages (capacity 2 each),
    # even when this graph is not constructible; still a safe lower bound.
    degrees = 2 * max(0, n - blocks - holes)
    whole, half = divmod(degrees, 2)
    return float(2 * values[:whole].sum() + (values[whole] if half else 0))


def audits():
    rng = np.random.default_rng(2026110219)
    direct_cases = 0
    for length in (3, 4, 7):
        events = rng.integers(0, length, size=27)
        counts = np.bincount(events, minlength=length)[None, :]
        for alpha in (1, 2.5, 9):
            actual = np.zeros(length)
            for source in events:
                for victim in range(length):
                    if source != victim:
                        actual[victim] += alpha if abs(source-victim) == 1 else 1
            np.testing.assert_allclose(stress(counts, alpha)[0], actual)
            shuffled = events.copy()
            rng.shuffle(shuffled)
            np.testing.assert_array_equal(np.bincount(shuffled, minlength=length), counts[0])
            direct_cases += 1
    brute_cases = 0
    for n in range(1, 7):
        values = rng.integers(0, 7, size=n).astype(float)
        for alpha in (1, 9):
            best = math.inf
            for positions in itertools.permutations(range(6), n):
                valid = np.zeros(6, dtype=bool)
                rates = np.zeros(6)
                valid[list(positions)] = True
                rates[list(positions)] = values
                cost = stress(rates.reshape(2, 3), alpha).ravel()[valid].sum()
                best = min(best, cost)
            lower = fractional_base_bound(values, 2, 3)
            lower += (alpha - 1) * neighbor_bound(values, 2, 3)
            assert lower <= best + 1e-6, (values, alpha, lower, best)
            brute_cases += 1
    uniform = np.ones((2, 192))
    expected = 191 + 16
    assert np.all(stress(uniform, 9)[:, 1:-1] == expected)
    return dict(direct_counter_cases=direct_cases,
                exhaustive_small_packing_lower_bounds=brute_cases,
                full_scan_formula=True, marginal_shuffle_invariant=True)


def make_arrays(streams, frequencies, experts, layer_index, blocks):
    counts = [np.zeros((PLANES, b, WL), dtype=np.int32) for b in blocks]
    owners = [np.full((PLANES, b, WL), -1, dtype=np.int32) for b in blocks]
    segment_stats = Counter()
    expected = Counter()
    for row in streams:
        layer, expert, phase, chip, start, rows, cols, pages, waves, runs, segments = row
        frequency = int(frequencies[layer_index[layer], expert])
        owner = layer_index[layer] * experts + expert
        assert pages == (rows * cols + 16383) // 16384
        assert waves == (pages + 3) // 4 and runs == len(segments)
        covered = 0
        for block, wl, nwaves, page_offset, npages, per_plane in segments:
            assert covered == page_offset and sum(per_plane) == npages
            assert 0 <= wl < WL and wl + nwaves <= WL
            for plane, number in enumerate(per_plane):
                target = owners[chip][plane, block, wl:wl+number]
                assert np.all(target == -1), (chip, plane, block, wl)
                target[:] = owner
                counts[chip][plane, block, wl:wl+number] = frequency
                segment_stats['weighted_segment_page_reads'] += number * frequency
                if number == WL:
                    segment_stats['weighted_full_physical_block_scan_reads'] += number * frequency
            covered += npages
        assert covered == pages
        expected['stored_pages'] += pages
        expected['foreground_page_reads_per_training_pass'] += pages * frequency
        expected['stream_waves_per_training_pass'] += waves * frequency
    assert sum(np.count_nonzero(o >= 0) for o in owners) == expected['stored_pages']
    assert sum(c.sum(dtype=np.int64) for c in counts) == expected['foreground_page_reads_per_training_pass']
    return counts, owners, dict(segment_stats), dict(expected)


def hotness_slots(streams, frequencies, layer_index):
    """Sort all model streams by marginal rate within equal address shapes.

    A permutation of existing same-page-count slots guarantees identical
    per-chip capacity, padding, block runs, whole rows and foreground work.
    """
    groups = defaultdict(list)
    result = [list(x) for x in streams]
    for index, row in enumerate(streams):
        groups[(row[3], row[7], row[8], row[9], tuple(s[4] for s in row[10]),
                tuple(tuple(s[5]) for s in row[10]))].append(index)
    for indices in groups.values():
        logical = sorted(indices, key=lambda i: (-int(frequencies[layer_index[streams[i][0]], streams[i][1]]), i))
        physical = sorted(indices, key=lambda i: (streams[i][10][0][0], streams[i][10][0][1]))
        for source, destination in zip(logical, physical):
            result[source] = list(streams[source])
            result[source][10] = streams[destination][10]
    return result


def hotness_repack(streams, frequencies, layer_index):
    allocators = [WaveAllocator() for _ in range(16)]
    result = []
    order = sorted(range(len(streams)), key=lambda i: (
        -int(frequencies[layer_index[streams[i][0]], streams[i][1]]), i))
    for index in order:
        row = list(streams[index])
        pages, waves, runs, segments = allocators[row[3]].allocate(row[5] * row[6])
        assert (pages, waves, runs) == tuple(row[7:10])
        row[10] = segments
        result.append(row)
    return result, [a.summary(c)['block_stripes'] for c, a in enumerate(allocators)]


def rate_statistics(values):
    values = np.asarray(values, dtype=float).ravel()
    total = values.sum()
    ordered = np.sort(values)[::-1]
    return dict(min=float(values.min()), max=float(values.max()), mean=float(values.mean()),
                cv=float(values.std() / values.mean()) if total else 0,
                zero_fraction=float(np.mean(values == 0)),
                top_10pct_read_share=float(ordered[:max(1, math.ceil(.1*len(values)))].sum()/total) if total else 0)


def analyze_policy(counts, owners, tokens, alphas, thresholds, lower_bounds):
    result = dict(blocks=0, stored_pages=0, reserved_slots=0,
                  weighted_reads=0, uniform_heat_reads=0, single_expert_reads=0,
                  fully_occupied_block_reads=0, multi_expert_block_reads=0,
                  heat_cv_read_weighted_sum=0.0, mean_occupancy_page_weighted_sum=0.0)
    costs = {str(a): Counter() for a in alphas}
    profiles = []
    for chip, (count, owner) in enumerate(zip(counts, owners)):
        valid = owner >= 0
        n = valid.sum(axis=-1)
        block_reads = count.sum(axis=-1, dtype=np.int64)
        nonempty = n > 0
        result['blocks'] += int(nonempty.sum())
        result['stored_pages'] += int(valid.sum())
        result['reserved_slots'] += int(valid.size)
        result['weighted_reads'] += int(block_reads.sum())
        result['fully_occupied_block_reads'] += int(block_reads[n == WL].sum())
        result['mean_occupancy_page_weighted_sum'] += float(np.square(n).sum())
        for plane, block in np.argwhere(nonempty):
            rates = count[plane, block, valid[plane, block]]
            same_heat = bool(np.all(rates == rates[0]))
            same_owner = len(np.unique(owner[plane, block, valid[plane, block]])) == 1
            reads = int(block_reads[plane, block])
            result['uniform_heat_reads'] += reads if same_heat else 0
            result['single_expert_reads'] += reads if same_owner else 0
            result['multi_expert_block_reads'] += 0 if same_owner else reads
            cv = float(rates.std() / rates.mean()) if rates.sum() else 0
            result['heat_cv_read_weighted_sum'] += cv * reads
            profiles.append((chip, int(plane), int(block), int(n[plane, block]),
                             reads / tokens, cv, same_owner))
        for alpha in alphas:
            s = np.where(valid, stress(count.astype(float), alpha) / tokens, 0)
            peak = s.max(axis=-1)
            out = costs[str(alpha)]
            out['block_rr_copy_numerator'] += float((n * peak).sum())
            out['block_rr_erase_numerator'] += float(peak.sum())
            out['ideal_wl_rr_copy_numerator'] += float(s.sum())
            for threshold in thresholds:
                # Deterministic reset-rate projection. Counts depend exclusively
                # on accumulated real training marginal reads, never iid noise.
                out['finite_block_copies_at_trainpass_T' + str(threshold)] += float(
                    (n * np.floor(peak * tokens / threshold)).sum())
                out['finite_wl_copies_at_trainpass_T' + str(threshold)] += float(
                    np.floor(s * tokens / threshold).sum())
    foreground = result['weighted_reads'] / tokens
    result['foreground_page_reads_per_token'] = foreground
    for key in ('uniform_heat_reads', 'single_expert_reads', 'fully_occupied_block_reads', 'multi_expert_block_reads'):
        result[key + '_fraction'] = result[key] / result['weighted_reads'] if result['weighted_reads'] else 0
    result['block_heat_cv_read_weighted'] = result['heat_cv_read_weighted_sum'] / result['weighted_reads']
    result['mean_valid_pages_per_page_block'] = result['mean_occupancy_page_weighted_sum'] / result['stored_pages']
    for alpha in alphas:
        out = costs[str(alpha)]
        baseline = out['block_rr_copy_numerator']
        wl = out['ideal_wl_rr_copy_numerator']
        lb = lower_bounds[str(alpha)]
        assert lb <= wl + 1e-4, (alpha, lb, wl)
        out['universal_same_capacity_copy_numerator_lower_bound'] = lb
        out['premature_block_rr_copy_numerator'] = max(0, baseline - wl)
        out['block_rr_overcopy_factor'] = baseline / wl if wl else 1
        out['block_rr_premature_copy_fraction'] = 1 - wl / baseline if baseline else 0
        out['any_placement_wl_rr_savings_fraction_upper_bound'] = max(0, 1 - lb/wl) if wl else 0
        out['any_placement_block_rr_copy_savings_fraction_upper_bound'] = max(0, 1-lb/baseline) if baseline else 0
    result['stress_costs'] = {k:dict(v) for k,v in costs.items()}
    return result, profiles


def model(name, args):
    frozen_path = SOURCE / (name + '.frozen_plan.json')
    frozen = json.loads(frozen_path.read_text())
    physical_path = Path(frozen['physical']['path'])
    assert sha256(physical_path) == frozen['physical']['sha256']
    train_path = Path(frozen['train_path'])
    assert train_path.name == 'train_routes.npz' and sha256(train_path) == frozen['train_sha256']
    with np.load(train_path) as data:
        routes = data['routes']
        assert data['prompt_ids'].tolist() == frozen['train_ids']
        assert len(frozen['train_ids']) == 200 and all(x.startswith('train_') for x in frozen['train_ids'])
        assert data['layers'].tolist() == frozen['layers']
    tokens = routes.shape[0] * routes.shape[1]
    experts = frozen['experts']
    frequencies = np.stack([np.bincount(routes[:, :, i].ravel(), minlength=experts)
                            for i in range(len(frozen['layers']))])
    assert np.all(frequencies.sum(axis=1) == tokens * frozen['active'])
    layer_index = {layer:i for i,layer in enumerate(frozen['layers'])}
    with gzip.open(physical_path, 'rt') as stream:
        physical = json.load(stream)
    streams = physical['streams']
    blocks = [x['block_stripes'] for x in physical['chips']]
    original = make_arrays(streams, frequencies, experts, layer_index, blocks)
    lower_base = lower_adj = 0.0
    lower_cache = {}
    for chip, (count, owner) in enumerate(zip(original[0], original[1])):
        for plane in range(PLANES):
            values = np.sort(count[plane][owner[plane] >= 0]).astype(float) / tokens
            key = (blocks[chip], hashlib.sha256(values.tobytes()).hexdigest())
            if key not in lower_cache:
                lower_cache[key] = (fractional_base_bound(values, blocks[chip], WL),
                                    neighbor_bound(values, blocks[chip], WL))
            base, adj = lower_cache[key]
            lower_base += base
            lower_adj += adj
    lower = {str(a): lower_base + (a-1)*lower_adj for a in args.alphas}
    policies = {'original_allocator': original,
                'global_hotness_same_slots': make_arrays(hotness_slots(streams, frequencies, layer_index),
                    frequencies, experts, layer_index, blocks)}
    repack, repack_blocks = hotness_repack(streams, frequencies, layer_index)
    repack_fits = all(a <= b for a,b in zip(repack_blocks,blocks))
    if repack_fits:
        policies['global_hotness_repack_same_capacity'] = make_arrays(
            repack, frequencies, experts, layer_index, blocks)
    output = {}
    for policy, arrays in policies.items():
        assert arrays[3] == original[3], (policy, arrays[3], original[3])
        value, profiles = analyze_policy(arrays[0], arrays[1], tokens, args.alphas, args.thresholds, lower)
        value['full_physical_block_single_segment_read_fraction'] = arrays[2].get(
            'weighted_full_physical_block_scan_reads', 0) / value['weighted_reads']
        output[policy] = value
        np.savez_compressed(args.output_dir / (name + '.' + policy + '.blocks.npz'),
                            columns=np.array(['chip','plane','block','valid_pages','reads_per_token','heat_cv','single_expert']),
                            profiles=np.asarray(profiles))
    sensitivity = []
    foreground_us = output['original_allocator']['foreground_page_reads_per_token'] * READ_US
    for alpha in args.alphas:
        strong_name = min(output, key=lambda p: output[p]['stress_costs'][str(alpha)]['ideal_wl_rr_copy_numerator'])
        strong = output[strong_name]['stress_costs'][str(alpha)]
        original_cost = output['original_allocator']['stress_costs'][str(alpha)]
        for threshold in args.thresholds:
            for program_us in args.program_us:
                for erase_us in args.erase_us:
                    original_work = (original_cost['block_rr_copy_numerator'] * (READ_US+program_us)
                                     + original_cost['block_rr_erase_numerator'] * erase_us) / threshold
                    strong_work = (strong['ideal_wl_rr_copy_numerator'] * (READ_US+program_us)
                                   + strong['ideal_wl_rr_copy_numerator'] / WL * erase_us) / threshold
                    lower_work = (lower[str(alpha)] * (READ_US+program_us+erase_us/WL)) / threshold
                    sensitivity.append(dict(alpha=alpha, erc_threshold=threshold,
                        program_us=program_us, erase_us=erase_us, strong_policy=strong_name,
                        original_block_rr_maintenance_work_us_per_token=original_work,
                        strong_ideal_wl_rr_maintenance_work_lower_estimate_us_per_token=strong_work,
                        universal_placement_maintenance_work_lower_bound_us_per_token=lower_work,
                        original_block_rr_serialized_nand_duty=original_work/(foreground_us+original_work),
                        all_maintenance_removal_work_savings_ceiling=original_work/(foreground_us+original_work),
                        placement_after_strong_wl_rr_work_savings_upper_bound=max(0,(strong_work-lower_work)/(foreground_us+strong_work)),
                        original_to_universal_work_savings_upper_bound=max(0,(original_work-lower_work)/(foreground_us+original_work))))
    report = dict(model=name, source=dict(frozen_path=str(frozen_path), frozen_sha256=sha256(frozen_path),
        physical_path=str(physical_path), physical_sha256=sha256(physical_path),
        train_path=str(train_path), train_sha256=sha256(train_path)),
        training=dict(prompts=200, decode_steps=int(routes.shape[1]), events_per_layer=tokens,
                      layers=frozen['layers'], experts=experts, active=frozen['active'],
                      expert_marginals_per_token=(frequencies/tokens).tolist(),
                      marginal_heterogeneity=rate_statistics(frequencies/tokens)),
        original_blocks_per_chip=blocks, repack_blocks_per_chip=repack_blocks,
        repack_same_capacity=repack_fits,
        stream_accounting=original[3],
        policy_results=output, fractional_packing_lower_bound=dict(base=lower_base, adjacent_degree=lower_adj),
        sensitivity=sensitivity)
    write_json(args.output_dir / (name + '.json'), report)
    print(name, 'stored_pages', original[3]['stored_pages'],
          'repack_fits', repack_fits,
          'max_post_WL_resource_work_bound',
          max(s['placement_after_strong_wl_rr_work_savings_upper_bound'] for s in sensitivity), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', default=['granite','qwen3','olmoe','deepseek','mixtral'])
    parser.add_argument('--output-dir', type=Path, default=DEST)
    parser.add_argument('--alphas', nargs='+', type=float, default=[1,8.4,9])
    parser.add_argument('--thresholds', nargs='+', type=int, default=[10000,50000,100000,500000,767000,1000000])
    parser.add_argument('--program-us', nargs='+', type=float, default=[300,380,700])
    parser.add_argument('--erase-us', nargs='+', type=float, default=[3500,5000])
    args = parser.parse_args()
    assert all(a >= 1 for a in args.alphas)
    assert all(t > 0 for t in args.thresholds)
    assert all(t >= 0 for t in args.program_us + args.erase_us)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit = audits()
    write_json(args.output_dir/'audit.json', audit)
    protocol = dict(scope=__doc__, script_sha256=sha256(Path(__file__)),
        read_us=READ_US, chips=16, planes_per_chip=PLANES, lsb_wl_per_block=WL,
        use_cr=False, copies=1, alphas=args.alphas, erc_thresholds=args.thresholds,
        program_us=args.program_us, erase_us=args.erase_us,
        source='Frozen true-chosen physical_plan files and all 200 train prompt marginals; no test_routes opened.',
        stress_model='ERC[j]=Rblock-R[j]+(alpha-1)*(R[j-1]+R[j+1]); only valid expert LSB pages need protection.',
        interpretation=[
            'Parameters are sensitivity axes, not measured AiF device calibration. Alpha9/767K resembles one STRAW Good@2K point from different NAND, not a joint calibration.',
            'All physical ordinary-read pages, including intra-page byte padding, are counted. Empty wave/block slots are invalid and not copied.',
            'Global hotness sorting changes only existing stream addresses, preserving rows/chips/page counts/waves/block-run counts. Equal-shape slot sort has exactly identical capacity; repack is included only if every chip fits original capacity.',
            'Block RR copies all valid pages when maximum WL stress reaches threshold. Fluid ideal WL-RR copies only each WL at its own stress rate. Finite training-pass projections use floor(accumulated stress/threshold), not iid faults.',
            'Ideal WL-RR and fractional grouping are lower bounds with free metadata, no new implemented ABI functionality. Their use only strengthens the no-go screen.',
            'Fractional bound permits arbitrary page fragmentation and fractional block occupancy, relaxing whole-row stream/program-order constraints; it cannot overstate minimum base stress.',
            'Read+program per copied page and erase per block are exposed sensitivity costs. Ideal WL erase floor is copies/192. Copy-induced extra disturbance, retention loss, retry feedback, ECC parity, ordinary CSB/MSB data and spare-space GC are omitted.',
            'No RBER-to-ECC-retry mapping is calibrated. No external retry noise or new decoder is added. This estimates read-count-derived reclaim work, not numerical ECC failures or correctability.',
            'The current ABI is not claimed to implement per-WL reclaim. Real page copying/erase must use existing controller paths, legal program order and spare blocks; no voltage/decoder/CIM changes are proposed.',
            'Serialized NAND resource work/duty is not a bound on token p99 or an end-to-end performance measurement. Background overlap and bounded suspend require an event simulator.',
            'Linear marginal counts cannot reveal coactivation benefit; threshold timing/reset/queue effects need separate validation if this work bound is material.'])
    write_json(args.output_dir/'protocol.json', protocol)
    reports = [model(name,args) for name in args.models]
    summary = []
    findings = []
    artifact_checks = []
    for report in reports:
        original = report['policy_results']['original_allocator']
        accounting = report['stream_accounting']
        # The current frozen plans occupy four real pages in every read wave;
        # verify this from independent stream and physical-page accounting.
        complete_wave_pages = (accounting['foreground_page_reads_per_training_pass'] ==
                               4*accounting['stream_waves_per_training_pass'])
        full_capacity = original['stored_pages'] == original['reserved_slots']
        base_identity = None
        if full_capacity and 1 in args.alphas:
            key = next(k for k in original['stress_costs'] if float(k) == 1)
            expected = (WL-1)*original['foreground_page_reads_per_token']
            base_identity = True
            for policy in report['policy_results'].values():
                actual = policy['stress_costs'][key]['ideal_wl_rr_copy_numerator']
                assert math.isclose(actual, expected, rel_tol=1e-12), (report['model'],actual,expected)
        artifact_checks.append(dict(model=report['model'], all_stream_wave_pages_real=complete_wave_pages,
                                    full_capacity_base_stress_identity=base_identity))
        row = dict(model=report['model'], marginal_cv=report['training']['marginal_heterogeneity']['cv'],
                   full_physical_block_scan_reads_fraction=original['full_physical_block_single_segment_read_fraction'],
                   single_expert_block_reads_fraction=original['single_expert_reads_fraction'],
                   fully_occupied_block_reads_fraction=original['fully_occupied_block_reads_fraction'],
                   block_heat_cv_read_weighted=original['block_heat_cv_read_weighted'],
                   reserved_utilization=original['stored_pages']/original['reserved_slots'],
                   maximum_after_strong_wl_rr_total_work_bound=max(s['placement_after_strong_wl_rr_work_savings_upper_bound'] for s in report['sensitivity']),
                   reference_alpha9_threshold767k_program380_erase3500=next((s for s in report['sensitivity'] if
                       s['alpha']==9 and s['erc_threshold']==767000 and s['program_us']==380 and s['erase_us']==3500), None))
        summary.append(row)
        maximum = max(report['sensitivity'], key=lambda s:s['placement_after_strong_wl_rr_work_savings_upper_bound'])
        reference = row['reference_alpha9_threshold767k_program380_erase3500']
        findings.append(dict(model=report['model'],
            reference_post_strong_WL_total_work_bound_percent=(
                100*reference['placement_after_strong_wl_rr_work_savings_upper_bound'] if reference else None),
            sweep_max_post_strong_WL_total_work_bound_percent=100*maximum['placement_after_strong_wl_rr_work_savings_upper_bound'],
            sweep_max_parameters={k:maximum[k] for k in ('alpha','erc_threshold','program_us','erase_us','strong_policy')},
            full_physical_block_single_stream_scan_fraction=row['full_physical_block_scan_reads_fraction'],
            explanation=('All physical blocks are fully occupied: base WL stress is permutation invariant; remaining adjacency-endpoint freedom is tiny.'
                         if full_capacity else 'Unused slots permit a looser occupancy/fragmentation oracle; its remaining work bound is still reported with actual maintenance duty.')))
    audit['real_artifact_checks'] = artifact_checks
    write_json(args.output_dir/'audit.json', audit)
    write_json(args.output_dir/'summary.json',dict(models=summary,audit=audit,
        conclusion='Inspect bounded total NAND work after strong WL reclaim; no hardware-performance or novel-hotness claim.'))
    write_json(args.output_dir/'findings.json', dict(
        disposition='archive_this_total_work_direction_for_the_stated_grid'
            if all(x['sweep_max_post_strong_WL_total_work_bound_percent'] < 15 for x in findings)
            else 'material_work_bound_needs_a_realizable_placement_and_strong_maintenance_baselines',
        results=findings,
        distinctions=[
            'Hotness substantially reduces premature block-reclaim copies on some models, but that is known prior and much of this credit disappears with ideal WL reclaim.',
            'All frozen stream segments are shorter than a whole physical 192-WL block; a zero full-block single-stream scan fraction must not be described as full-block expert scans.',
            'Full physical block occupancy, full single-stream scan, and single-expert block ownership are different metrics.',
            'All bounds concern homogeneous-threshold first-order serialized NAND resource work; no tail-latency, calibrated device or end-to-end speedup claim.',
            'Extreme low ERC thresholds are sensitivity only; no AiF LSB calibration or extra read-retry decoder is introduced.',
            'Fine-grain reclaim and arbitrary fractional packing are oracle relaxations; they do not authorize new hardware functionality.']))
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__':
    main()
