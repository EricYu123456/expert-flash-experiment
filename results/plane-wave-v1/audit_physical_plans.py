#!/usr/bin/env python3
"""Independent row coverage, NAND interval and exact capacity plan audit."""
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import aif_model as flash

OUT = ROOT / 'results/plane-wave-v1'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def main():
    source = read_json(OUT / 'static-bundles/manifest.json')
    physical = read_json(OUT / 'static-bundles/physical-plans/manifest.json')
    for name, expected in physical['source_sha256'].items():
        assert digest(ROOT / 'scripts' / name) == expected
    receipts = []
    for metadata in physical['plans_metadata']:
        path = Path(metadata['path'])
        assert digest(path) == metadata['sha256']
        plan = json.loads(gzip.decompress(path.read_bytes()))
        model, tag = plan['model'], plan['tag']
        g = int(tag.split('_')[0][1:])
        old = read_json(Path(source['sources'][model]['old_plan']))
        dimensions = flash.tensor_dimensions(read_json(Path(source['sources'][model]['metadata'])))
        frozen_path = OUT / 'static-bundles' / (model + '.frozen_plan.json')
        frozen = read_json(frozen_path)
        assert plan['source_frozen_plan_sha256'] == digest(frozen_path)
        labels = frozen['labels'][tag]
        coverage = defaultdict(list)
        chip_ranges = defaultdict(list)
        chip_pages, chip_waves, chip_useful = defaultdict(int), defaultdict(int), defaultdict(int)
        explicit_addresses = set()
        explicit_pages = 0
        for record in plan['streams']:
            layer, projection, bundle, chip, columns, nbytes, pages, waves, runs, outputs, segments = record
            dims = dimensions[layer]
            rows, wanted_columns = ((dims['gate'][0] + dims['up'][0], dims['up'][1])
                                     if projection == 'gate_up' else dims['down'])
            assert columns == wanted_columns
            assert sum(out[2] for out in outputs) * columns == nbytes
            assert pages == (nbytes + 16383) // 16384
            assert waves == (pages + 3) // 4
            assert runs == (waves + 191) // 192 == len(segments)
            assert outputs
            expected_local_row = 0
            for expert, row_start, count, local_row in outputs:
                assert local_row == expected_local_row
                expected_local_row += count
                assert count > 0 and 0 <= row_start < row_start + count <= rows
                mapping = old['mappings'][f'g{g}'][str(layer)][expert]
                assert chip // g == mapping % (16 // g)
                if projection == 'gate_up':
                    assert labels[str(layer)][expert] == bundle
                else:
                    assert len(outputs) == 1 and expert == bundle
                coverage[layer, projection, expert].append((row_start, row_start + count))
            offset, wave_sum = 0, 0
            for block, wl, count, page_offset, segment_pages, plane_counts in segments:
                assert page_offset == offset
                assert sum(plane_counts) == segment_pages
                assert max(plane_counts) == count
                assert all(0 <= wl + p <= 192 for p in plane_counts)
                assert all(a >= b and a - b <= 1 for a, b in zip(plane_counts, plane_counts[1:]))
                assert (segment_pages + 3) // 4 == count
                assert len(plane_counts) == 4
                chip_ranges[chip].append((block * 192 + wl, block * 192 + wl + count))
                # Enumerate two representative plans, including a chip of the
                # larger DeepSeek bundle; all other addresses get an exact
                # interval proof because whole four-plane waves are reserved.
                if (model == 'granite' and tag.endswith('b2')) or (model == 'deepseek' and tag.endswith('b4') and chip == 0):
                    for page in range(segment_pages):
                        address = chip, page % 4, block, wl + page // 4
                        assert address not in explicit_addresses
                        explicit_addresses.add(address)
                        explicit_pages += 1
                offset += segment_pages
                wave_sum += count
            assert offset == pages and wave_sum == waves
            chip_pages[chip] += pages
            chip_waves[chip] += waves
            chip_useful[chip] += nbytes
        expected_useful = 0
        expected_logical = 0
        for layer, dims in dimensions.items():
            expert_count = len(old['mappings'][f'g{g}'][str(layer)])
            for projection, rows, columns in (('gate_up', dims['gate'][0] + dims['up'][0], dims['up'][1]),
                                               ('down', *dims['down'])):
                expected_useful += expert_count * rows * columns
                for expert in range(expert_count):
                    intervals = sorted(coverage[layer, projection, expert])
                    assert intervals and intervals[0][0] == 0 and intervals[-1][1] == rows
                    assert all(a[1] == b[0] for a, b in zip(intervals, intervals[1:]))
                    expected_logical += 1
        assert len(coverage) == expected_logical
        total_reserved = 0
        total_boundary, total_final = 0, 0
        for chip in range(16):
            intervals = sorted(chip_ranges[chip])
            assert all(a[1] <= b[0] for a, b in zip(intervals, intervals[1:]))
            cursor = intervals[-1][1] if intervals else 0
            reserved = ((cursor + 191) // 192) * 192
            boundary = cursor - chip_waves[chip]
            final = reserved - cursor
            summary = plan['chip_summary'][chip]
            assert summary['useful_int8_bytes'] == chip_useful[chip]
            assert summary['physical_pages'] == chip_pages[chip]
            assert summary['read_rounds'] == chip_waves[chip]
            assert summary['boundary_padding_waves'] == boundary
            assert summary['final_block_tail_waves'] == final
            assert summary['reserved_lsb_bytes'] == reserved * 65536
            total_reserved += reserved * 65536
            total_boundary += boundary
            total_final += final
        assert sum(chip_useful.values()) == expected_useful == plan['totals']['useful_int8_bytes']
        assert total_reserved == plan['totals']['reserved_lsb_bytes']
        assert sum(chip_pages.values()) == plan['totals']['physical_pages']
        assert sum(chip_waves.values()) == plan['totals']['read_rounds']
        receipts.append(dict(model=model, tag=tag, streams=len(plan['streams']),
                             covered_expert_projections=expected_logical,
                             physical_pages=sum(chip_pages.values()), explicitly_enumerated_pages=explicit_pages,
                             all_rows_once=True, nonoverlapping_wave_ranges=True,
                             same_wl_per_stripe=True, exact_capacity_including_boundary_and_final_padding=True,
                             boundary_padding_waves=total_boundary, final_block_tail_waves=total_final))
    target = OUT / 'audit_physical_plans.json'
    target.write_text(json.dumps(dict(scope='All 15 plans: row/interval/capacity proof; two representative page enumerations',
                                     plans=receipts), indent=2) + '\n')
    print(json.dumps(dict(plans=len(receipts), covered_physical_pages=sum(r['physical_pages'] for r in receipts),
                          explicitly_enumerated_pages=sum(r['explicitly_enumerated_pages'] for r in receipts),
                          result=str(target)), indent=2))


if __name__ == '__main__':
    main()
