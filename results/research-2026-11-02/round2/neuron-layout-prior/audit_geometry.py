#!/usr/bin/env python3
"""Independent g12 costs, exact uniform-routing null and physical-witness audit."""
from collections import Counter
from fractions import Fraction
import gzip
import hashlib
import json
from math import comb
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DEST = ROOT / 'results/research-2026-11-02/round2/geometry-study'
FROZEN = DEST / 'granite4tiny.frozen.json'
EXPECTED = 'a55f17010b7b48c6c396dee92bb1d127a541e28a93b94aee2e42aac9f3dc2538'
PAGE, PLANES, BLOCK = 16384, 4, 192


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ceildiv(a, b):
    return (a + b - 1) // b


def ledger(selected, rows, columns, common):
    """No imports from the implementation's model/evaluation helpers."""
    rows = np.asarray(rows, dtype=np.int64)
    pages = ceildiv(rows * columns, PAGE)
    waves = ceildiv(pages, PLANES)
    selected_rows = rows[selected]
    chip_pages = pages[selected].sum(axis=1)
    chip_waves = waves[selected].sum(axis=1)
    chip_inputs = (np.any(selected_rows > 0, axis=1) if common else
                   np.count_nonzero(selected_rows, axis=1)) * columns
    chip_outputs = selected_rows.sum(axis=1) * 4
    commands = ceildiv(waves, BLOCK)[selected].sum(axis=(1, 2))
    values = dict(flash_ns=np.maximum(chip_waves.max(axis=1) * 28000,
                                     chip_pages.max(axis=1) * 2560),
        control_ns=commands * 250,
        host_ns=np.full(len(selected), 2000, dtype=np.int64),
        input_ns=np.maximum(ceildiv(columns * (1 if common else selected.shape[1]), 8),
            ceildiv((chip_inputs[:, :8] + chip_inputs[:, 8:]).max(axis=1), 2)),
        output_ns=np.maximum(ceildiv(chip_outputs.sum(axis=1), 8),
            ceildiv((chip_outputs[:, :8] + chip_outputs[:, 8:]).max(axis=1), 2)))
    values['model_ns'] = sum(values.values())
    return values, chip_waves, chip_pages, commands


def count_vectors(groups, members, k, prefix=(), weight=1):
    if groups == 1:
        if k <= members:
            yield prefix + (k,), weight * comb(members, k)
    else:
        for n in range(min(k, members) + 1):
            yield from count_vectors(groups - 1, members, k - n,
                prefix + (n,), weight * comb(members, n))


def exact_null():
    denominator = comb(64, 6)
    power2 = []
    for g in (1, 2, 4, 8, 16):
        rows = 1536 // g
        waves = ceildiv(ceildiv(rows * 512, PAGE), 4)
        counts_total = cost_total = 0
        distribution = Counter()
        for n, weight in count_vectors(16 // g, 4 * g, 6):
            loads = [n[c // g] for c in range(16)]
            pair = max(loads[c] + loads[c + 8] for c in range(8))
            cost = max(loads) * waves * 28000 + 6 * g * 250 + 2000 + \
                max(384, ceildiv(pair * 512, 2)) + max(4608, ceildiv(pair * rows * 4, 2))
            counts_total += weight
            cost_total += cost * weight
            distribution[cost] += weight
        assert counts_total == denominator
        power2.append(dict(g=g, mean_service_ns=float(Fraction(cost_total, denominator)),
            exact_mean_numerator=cost_total, exact_mean_denominator=denominator))
    counts_total = cost_total = wave_total = five_count = input_total = 0
    occupancy = []
    for n, weight in count_vectors(4, 16, 6):
        peak = 6 - min(n)
        assert peak in (5, 6)
        pair_min = min(n[0] + n[2], n[1] + n[3])
        input_ns = 256 * (12 - pair_min)
        cost = peak * 28000 + 18000 + 2000 + input_ns + 4608
        counts_total += weight
        wave_total += peak * weight
        input_total += input_ns * weight
        cost_total += cost * weight
        five_count += (peak == 5) * weight
        occupancy.append(dict(omitted_group_counts=list(n), combinations=weight,
            peak_waves=peak, input_ns=input_ns, service_ns=cost))
    assert counts_total == denominator
    numerator = denominator - 4 * comb(48, 6) + 6 * comb(32, 6) - 4 * comb(16, 6)
    assert numerator == five_count
    p = Fraction(numerator, denominator)
    assert float(Fraction(wave_total, denominator)) == 6 - float(p)
    return dict(null='Uniform sampling of six distinct experts from 64; exactly 16 experts in each omitted color.',
        formula='P5 = [C(64,6)-4C(48,6)+6C(32,6)-4C(16,6)] / C(64,6)',
        exact_P5=str(p), probability_five_waves=float(p), probability_six_waves=1-float(p),
        iid_color_approximation_P5=float(Fraction(195, 512)),
        mean_peak_waves=6-float(p), mean_input_ns=float(Fraction(input_total, denominator)),
        g12_mean_service_ns=float(Fraction(cost_total, denominator)),
        power2_uniform_means=power2, occupancy_cases=occupancy,
        random_balanced_mapping_interpretation='The same expectation holds for any fixed trace of six distinct experts when expert IDs are mapped uniformly at random to four labeled omitted colors of exactly 16 experts each. Each fixed selected set sees the same multivariate hypergeometric color law; linearity of expectation removes any assumption about expert popularity or query independence.',
        controls=['Any capacity-balanced fixed expert permutation has exactly the same uniform expectation.',
            'This is an exact expectation over balanced random layouts, not generated routes. A single random layout has trace-dependent variance; confidence intervals across layouts require repeated layout sampling.',
            'Comparing fitted train routes with this null does not establish out-of-sample improvement.'])


def physical_audit(path, phases, policy):
    with gzip.open(path, 'rt') as stream:
        doc = json.load(stream)
    phase_map = {(p['layer'], p['name']): p for p in phases}
    coverage = {}
    last = [0] * 16
    totals = Counter()
    chip_bytes = np.zeros(16, dtype=np.int64)
    for layer, name, expert, chip, start, rows, columns, pages, waves, commands, segments in doc['streams']:
        phase = phase_map[layer, name]
        expected = phase['rows'] if policy == 'chosen' else phase['group_rows']
        assert rows == expected[expert][chip] > 0 and columns == phase['columns']
        key = layer, name, expert
        assert start == coverage.get(key, 0)
        coverage[key] = start + rows
        assert pages == ceildiv(rows * columns, PAGE)
        assert waves == ceildiv(pages, 4)
        assert commands == len(segments) == ceildiv(waves, BLOCK)
        consumed = 0
        for block, wl, rounds, page_offset, count, plane_counts in segments:
            assert 0 <= wl < BLOCK and wl + rounds <= BLOCK
            assert block * BLOCK + wl >= last[chip]
            last[chip] = block * BLOCK + wl + rounds
            assert page_offset == consumed
            assert sum(plane_counts) == count and max(plane_counts) == rounds
            assert plane_counts == [ceildiv(count - plane, 4) for plane in range(4)]
            # The active pages of each wave share this block/WL across planes.
            assert all(0 <= n <= rounds and wl + n <= BLOCK for n in plane_counts)
            consumed += count
        assert consumed == pages
        chip_bytes[chip] += rows * columns
        totals['useful_bytes'] += rows * columns
        totals['physical_pages'] += pages
        totals['chip_waves'] += waves
        totals['commands'] += commands
    for phase in phases:
        for expert in range(64):
            assert coverage[phase['layer'], phase['name'], expert] == phase['total_rows']
    expected_bytes = sum(p['total_rows'] * p['columns'] * 64 for p in phases)
    assert totals['useful_bytes'] == expected_bytes
    for chip, summary in enumerate(doc['chips']):
        assert summary['useful_int8_bytes'] == chip_bytes[chip]
    return dict(path=str(path), sha256=sha(path), stream_count=len(doc['streams']),
        logical_matrix_coverage_once=True, per_chip_reserved_wave_ranges_nonoverlapping=True,
        ordinary_same_block_same_wordline_plane_groups=True,
        useful_bytes_per_chip=chip_bytes.tolist(), totals=dict(totals),
        reserved_lsb_bytes=sum(c['reserved_lsb_bytes'] for c in doc['chips']),
        idle_plane_slot_bytes=sum(c['idle_plane_slot_bytes'] for c in doc['chips']),
        limitation='Virtual address and row-coverage witness only; no programmed NAND or vendor capacity measurement.')


def main():
    before = sha(FROZEN)
    assert before == EXPECTED
    frozen = json.loads(FROZEN.read_text())
    assert len(frozen['phases']) == 80
    with np.load(frozen['training_routes']) as f:
        routes = f['routes']
        assert f['prompt_ids'].tolist() == frozen['train_ids']
    assert sha(Path(frozen['training_routes'])) == frozen['training_routes_sha256']
    assert all(pid.startswith('train_') for pid in frozen['train_ids'])
    phase_results = []
    sums = Counter()
    for phase in frozen['phases']:
        layer = frozen['layers'].index(phase['layer'])
        selected = routes[:, :, layer].reshape(-1, 6)
        for policy, key in [('chosen', 'rows'), ('group', 'group_rows')]:
            matrix = np.asarray(phase[key], dtype=np.int64)
            assert matrix.shape == (64, 16) and np.all(matrix >= 0)
            assert np.all(matrix.sum(axis=1) == phase['total_rows'])
            values, waves, pages, commands = ledger(selected, matrix, phase['columns'], phase['common'])
            assert abs(float(values['model_ns'].mean()) - phase[policy + '_ns']) < 1e-6
            for metric, value in values.items():
                sums[policy + '_' + metric] += float(value.mean())
        if phase['name'] != 'down':
            continue
        candidate = next(c for c in phase['candidates'] if c['label'] == 'complement_g12')
        matrix = np.asarray(candidate['rows'], dtype=np.int64)
        assert np.all(np.isin(matrix, [0, 128]))
        assert np.all((matrix == 0).sum(axis=1) == 4)
        assert np.all((matrix > 0).sum(axis=0) == 48)
        omitted = []
        for row in matrix:
            zeros = np.flatnonzero(row == 0)
            assert zeros[0] % 4 == 0 and np.array_equal(zeros, np.arange(zeros[0], zeros[0]+4))
            omitted.append(int(zeros[0] // 4))
        assert np.bincount(omitted, minlength=4).tolist() == [16] * 4
        counts = np.stack([(np.asarray(omitted)[selected] == j).sum(axis=1) for j in range(4)], axis=1)
        peak = 6 - counts.min(axis=1)
        input_ns = 256 * (12 - np.minimum(counts[:,0]+counts[:,2], counts[:,1]+counts[:,3]))
        exact = peak * 28000 + 18000 + 2000 + input_ns + 4608
        values, waves, pages, commands = ledger(selected, matrix, 512, False)
        np.testing.assert_array_equal(exact, values['model_ns'])
        np.testing.assert_array_equal(peak, waves.max(axis=1))
        assert np.all(pages.sum(axis=1) == 6 * 48) and np.all(commands == 72)
        assert abs(float(exact.mean()) - candidate['mean_ns']) < 1e-6
        phase_results.append(dict(layer=phase['layer'], trained_P5=float(np.mean(peak == 5)),
            trained_P6=float(np.mean(peak == 6)), mean_input_ns=float(input_ns.mean()),
            g12_mean_ns=float(exact.mean()), strongest_group_ns=phase['group_ns'],
            g12_gain_vs_group_pct=100*(1-float(exact.mean())/phase['group_ns']),
            physical_pages_per_expert=48, occupied_chip_waves_per_expert=12,
            commands_per_request=72, input_logical_host_bytes=3072, output_logical_host_bytes=36864,
            output_service_ns=4608))
    physical = []
    for policy, key in [('chosen','chosen_physical'),('group','phase_group_physical')]:
        witness = frozen[key]
        assert sha(Path(witness['path'])) == witness['sha256']
        physical.append(dict(policy=policy, **physical_audit(Path(witness['path']), frozen['phases'], policy)))
    assert sha(FROZEN) == before
    null = exact_null()
    null['frozen_plan_sha256'] = before
    (DEST / 'audit_uniform_routing.json').write_text(json.dumps(null, indent=2) + '\n')
    chosen_down = sum(p['g12_mean_ns'] for p in phase_results)
    random_down = len(phase_results) * null['g12_mean_service_ns']
    output = dict(status='passed', frozen_plan_sha256=before,
        train_routes_sha256=frozen['training_routes_sha256'], source_sha256=sha(Path(__file__)),
        physical=physical, component_training_totals_ns=dict(sums), down_phases=phase_results,
        down_training_decomposition=dict(chosen_ns=chosen_down,
            strongest_phase_group_ns=sum(p['strongest_group_ns'] for p in phase_results),
            exact_random_balanced_g12_expectation_ns=random_down,
            learned_vs_random_layout_pct=100*(1-chosen_down/random_down),
            mean_trained_P5=float(np.mean([p['trained_P5'] for p in phase_results])),
            qualification='Training-set decomposition only; no heldout performance or sampling-variance claim.'),
        g12_closed_form='T = 28000*(6-min n_j)+18000+2000+256*(12-min(n0+n2,n1+n3))+4608 ns',
        g16_down_closed_form_ns=201680,
        comparisons='g16 reads 3 pages on each of 16 chips; g12 reads 4 pages on each of 12 chips. Both are 48 physical pages/expert. g12 has 72 vs 96 commands/request.',
        qualifications=['All costs are fixed W8/FP32 service-model arithmetic, not physical or full-model latency.',
            'Reserved wave/LSB capacity includes unused plane slots and allocator padding; it is not measured NAND-vendor usable capacity.',
            'Shared FFN, SSM/attention, host row reassembly/activation/combine execution and ECC spare storage are outside these totals.',
            'The no-replica matrix rows are complete, unique, immutable and never duplicated; no routing, decoder, operator or ABI changes are proposed.',
            'Hypergraph selected-set minmax and step-cost declustering are established prior art; this audit establishes feasibility/accounting, not novelty.'])
    (DEST / 'audit_g12_physical_service.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(dict(status=output['status'], phases=len(frozen['phases']),
        down_layers=len(phase_results), mean_trained_P5=float(np.mean([p['trained_P5'] for p in phase_results])),
        null_P5=null['probability_five_waves'], null_service_ns=null['g12_mean_service_ns'],
        uniform_power2=null['power2_uniform_means'], totals_ns=dict(sums)), indent=2))


if __name__ == '__main__':
    main()
