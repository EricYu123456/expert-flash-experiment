#!/usr/bin/env python3
"""Independent legacy-address and exact combinatorial reference checks.

This audits outputs without modifying the model, runner, plans, or routes.
Run from the repository root: python results/plane-wave-v1/audit_reference.py
"""
from itertools import combinations, product
from math import comb
from pathlib import Path
import hashlib
import json
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import aif_model as legacy
import materialize_plan as legacy_placement
import plane_wave_model as plane


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def old_addresses(segments):
    result = []
    for block, wl, waves, offset, pages, per_plane in segments:
        result.extend((p % 4, block, wl + p // 4) for p in range(pages))
    return result


def compare_legacy_placement():
    rng = random.Random(85373)
    receipts = []
    for block_pages in (4, 7, 192):
        values = [1, plane.PAGE - 1, plane.PAGE, plane.PAGE + 1,
                  4 * plane.PAGE - 1, 4 * plane.PAGE + 1,
                  4 * block_pages * plane.PAGE - 1,
                  4 * block_pages * plane.PAGE + 1]
        values += [rng.randrange(1, (8 * block_pages + 3) * plane.PAGE)
                   for _ in range(25)]
        old_allocator = legacy_placement.WaveAllocator(block_pages)
        new_layout = plane.build_layout(dict(enumerate(values)), block_pages=block_pages)
        seen = set()
        checked_pages = 0
        for expert, nbytes in enumerate(values):
            old_pages, old_waves, old_cold, old_segments = old_allocator.allocate(nbytes)
            shard = new_layout.shards[expert]
            got = [shard.page_address(p) for p in range(shard.pages)]
            assert got == old_addresses(old_segments)
            assert (shard.pages, shard.waves, len(shard.segments)) == (old_pages, old_waves, old_cold)
            assert not (seen & set(got))
            seen.update(got)
            checked_pages += len(got)
        old_summary, new_summary = old_allocator.summary(0), new_layout.summary()
        for key in ('reserved_lsb_bytes', 'reserved_tlc_raw_bytes',
                    'physical_page_bytes', 'intra_page_padding_bytes'):
            assert old_summary[key] == new_summary[key]
        # Old Demand is independent of the new allocator's segment bookkeeping.
        for _ in range(50):
            selected = rng.sample(range(len(values)), rng.randrange(1, len(values) + 1))
            demand = legacy.Demand()
            for e in selected:
                demand.add_stream(values[e], block_pages, 1)
            result = plane.evaluate_chip(new_layout, selected)
            assert result['service_ns'] == demand.service_ns
            assert result['physical_pages'] == demand.pages
            assert result['cold_reads'] == demand.cold
        receipts.append(dict(block_pages=block_pages, streams=len(values), checked_pages=checked_pages,
                             selected_subsets=50, all_addresses_and_selected_service_equal=True))
    return receipts


def exact_fixed_bins():
    """Hypergeometric occupancy gives an oracle independent of scheduling code."""
    experts, active = 8, 4
    results = []
    for width in (2, 1):
        bins = 4 // width
        resident = experts // bins
        # Each expert consumes exactly one read wave in its assigned bin.
        layout = plane.build_layout({e: width * plane.PAGE for e in range(experts)},
                                    policy={2: 'pair', 1: 'single'}[width])
        before = {e: [s.page_address(p) for p in range(s.pages)]
                  for e, s in layout.shards.items()}
        population = comb(experts, active)
        # Count subset probabilities from bin occupancies, without consulting
        # expert IDs or outputs of the model being audited.
        expected_sum = 0
        expected_histogram = {}
        for occupancy in product(range(resident + 1), repeat=bins):
            if sum(occupancy) != active:
                continue
            ways = 1
            for count in occupancy:
                ways *= comb(resident, count)
            latency = max(occupancy) * 28_000
            expected_sum += ways * latency
            expected_histogram[latency] = expected_histogram.get(latency, 0) + ways
        actual_sum = 0
        actual_histogram = {}
        for selected in combinations(range(experts), active):
            measured = plane.evaluate_chip(layout, selected, include_timeline=True)
            actual_sum += measured['service_ns']
            actual_histogram[measured['service_ns']] = actual_histogram.get(measured['service_ns'], 0) + 1
            assert measured['conservative_service_ns'] == (
                measured['read_waves'] * 28_000 + measured['physical_pages'] * 2560)
            assert measured['service_ns'] * plane.CHIP_BW >= measured['physical_pages'] * plane.PAGE * 1_000_000_000
            for timeline in measured['plane_timeline']:
                for previous, following in zip(timeline, timeline[1:]):
                    assert previous['drain_end_ns'] <= following['nand_start_ns']
                assert all(x['nand_end_ns'] - x['nand_start_ns'] == 28_000 for x in timeline)
        assert actual_sum == expected_sum
        assert actual_histogram == expected_histogram
        assert before == {e: [s.page_address(p) for p in range(s.pages)]
                          for e, s in layout.shards.items()}
        results.append(dict(experts=experts, selected=active, width=width, exact_subsets=population,
                            mean_service_ns=expected_sum / population,
                            latency_histogram=expected_histogram,
                            fixed_addresses_preserved=True,
                            conservative_ordinary_reads_and_buffer_safety=True))
    return results


def bundle_closed_form():
    # Finite population formula: a bundle is touched unless all K samples avoid
    # its B members. This explicitly charges reads of inactive bundle members.
    experts, active = 32, 8
    results = []
    for bundle in (1, 2, 4):
        touched = experts / bundle * (1 - comb(experts - bundle, active) / comb(experts, active))
        cost_us = 28 + 9.7 * (bundle - 1)
        results.append(dict(experts=experts, selected=active, bundle_size=bundle,
                            expected_touched=touched,
                            read_amplification=bundle * touched / active,
                            gate_up_us_per_layer=touched * cost_us,
                            speedup_vs_unbundled=active * 28 / (touched * cost_us)))
    return results


def audit_static_bundle_counts():
    """Exact finite-population expectations include inactive outputs/weights."""
    import numpy as np
    import static_wave_bundle_study as bundle_study

    experts, active, hidden, intermediate = 8, 4, 1024, 512
    population = comb(experts, active)
    routes = np.asarray(list(combinations(range(experts), active)))
    dimensions = dict(gate=(intermediate, hidden), up=(intermediate, hidden),
                      down=(hidden, intermediate))
    receipts = []
    for bundle in (1, 2, 4):
        ideal, conservative, metrics = bundle_study.layer_costs(
            routes, list(range(experts)), 16, dimensions,
            [e // bundle for e in range(experts)])
        touched_sum = (experts // bundle) * (population - comb(experts - bundle, active))
        expected_flash = touched_sum * (28_000 + (bundle - 1) * 9700) + population * active * 28_000
        expected_weights = (touched_sum * bundle * 2 + population * active) * hidden * intermediate
        assert int(ideal['flash_ns'].sum()) == expected_flash
        assert int(metrics['read_weight_bytes'].sum()) == expected_weights
        assert int(metrics['physical_pages'].sum()) * 16_384 == expected_weights
        assert int(metrics['gate_returned_experts'].sum()) == bundle * touched_sum
        assert int(metrics['gate_selected_experts'].sum()) == active * population
        # 4096 output bytes per expert in each phase; PCIe/8 dominates channels.
        assert int(ideal['output_ns'].sum()) == (bundle * touched_sum + active * population) * 512
        expected_conservative = (int(metrics['read_waves'].sum()) // 16 * 28_000 +
                                 int(metrics['physical_pages'].sum()) // 16 * 2560)
        assert int(conservative['flash_ns'].sum()) == expected_conservative
        receipts.append(dict(experts=experts, active=active, exact_subsets=population,
                             bundle=bundle, flash_ns_mean=expected_flash / population,
                             all_inactive_weights_and_outputs_charged=True,
                             conservative_all_waves_cold=True))
    return receipts


def main():
    receipt = dict(audit_kind='independent prior-implementation comparison and exact combinatorial oracles',
                   source_sha256={name: file_hash(ROOT / 'scripts' / name)
                                  for name in ('plane_wave_model.py', 'materialize_plan.py', 'aif_model.py',
                                               'static_wave_bundle_study.py')},
                   legacy_address_comparison=compare_legacy_placement(),
                   exact_fixed_bins=exact_fixed_bins(),
                   uniform_granite_bundle_reference=bundle_closed_form(),
                   static_bundle_exact_overfetch_audit=audit_static_bundle_counts())
    target = Path(__file__).with_name('audit_reference.json')
    target.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
