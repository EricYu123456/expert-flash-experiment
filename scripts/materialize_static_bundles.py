#!/usr/bin/env python3
"""Produce fixed same-WL NAND address metadata for trained gate/up bundles."""
import argparse
import gzip
import json
from pathlib import Path

import aif_model as flash
from materialize_plan import WaveAllocator
from run_corpus_traces import DATA, MODELS, ROOT, sha256, write_json


def materialize(dimensions, mappings, labels_by_layer, g):
    allocators = [WaveAllocator() for _ in range(16)]
    records = []
    expected_bytes = 0
    coverage = set()
    for layer in sorted(dimensions):
        dims = dimensions[layer]
        mappings_layer = mappings[str(layer)]
        labels = labels_by_layer[str(layer)]
        experts = len(mappings_layer)
        gate_rows, columns = dims['gate'][0] + dims['up'][0], dims['up'][1]
        down_rows, down_columns = dims['down']
        expected_bytes += experts * (gate_rows * columns + down_rows * down_columns)
        bundles = {}
        for expert, bundle in enumerate(labels):
            bundles.setdefault(bundle, []).append(expert)
        projections = [('gate_up', gate_rows, columns, bundles)]
        projections.append(('down', down_rows, down_columns, {e:[e] for e in range(experts)}))
        for projection, rows, cols, members_by_bundle in projections:
            for bundle, members in sorted(members_by_bundle.items()):
                groups = {mappings_layer[e] % (16 // g) for e in members}
                assert len(groups) == 1
                first_chip = groups.pop() * g
                for shard in range(g):
                    count = rows // g + int(shard < rows % g)
                    if count == 0:
                        continue
                    start = shard * (rows // g) + min(shard, rows % g)
                    chip = first_chip + shard
                    nbytes = count * len(members) * cols
                    pages, waves, cold, segments = allocators[chip].allocate(nbytes)
                    outputs = []
                    for member_index, expert in enumerate(members):
                        key = (layer, projection, expert, shard)
                        assert key not in coverage
                        coverage.add(key)
                        outputs.append([expert, start, count, member_index * count])
                    assert sum(s[4] for s in segments) == pages
                    assert sum(s[2] for s in segments) == waves
                    assert len(segments) == cold
                    records.append([layer, projection, bundle, chip, cols, nbytes,
                                    pages, waves, cold, outputs, segments])
    summaries = [a.summary(chip) for chip, a in enumerate(allocators)]
    totals = {key:sum(s[key] for s in summaries) for key in summaries[0] if key != 'chip'}
    assert totals['useful_int8_bytes'] == expected_bytes
    assert totals['reserved_lsb_bytes'] == (totals['physical_page_bytes'] +
        totals['idle_plane_slot_bytes'] + (totals['boundary_padding_waves'] +
                                         totals['final_block_tail_waves']) * 4 * flash.PAGE)
    return dict(artifact_type='Virtual address metadata only; no NAND program or weight copy.',
        geometry=dict(chips=16, planes=4, page_bytes=flash.PAGE, lsb_wordlines_per_block=192),
        stream_columns=['layer','projection','bundle','chip','input_dimension','int8_bytes',
                        'pages','waves','block_runs','outputs','segments'],
        output_columns=['expert','source_projection_row_start','row_count','bundle_output_row_start'],
        segment_columns=['block_id_per_plane','wordline_offset','read_waves','stream_page_offset',
                         'physical_pages','plane_page_counts'],
        address_rule='Within each segment: local page p -> plane p%4, WL wordline_offset+floor(p/4), same block stripe.',
        data_order='Chip-major row permutation, then expert-major rows; down remains singleton.',
        checks=dict(unique_logical_shards=True, useful_bytes_conserved=True,
                    nonoverlapping_whole_wave_ranges=True, all_planes_same_block_wl_per_wave=True,
                    page_wave_block_counts_match_timing=True, no_weight_replication=True),
        chip_summary=summaries, totals=totals, streams=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=ROOT/'results/plane-wave-v1/static-bundles')
    parser.add_argument('--run-dir', type=Path, default=DATA/'runs/corpus-300-v1')
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    args = parser.parse_args()
    destination = args.results/'physical-plans'
    destination.mkdir(exist_ok=True)
    rows = []
    for name in args.models:
        frozen_path = args.results/f'{name}.frozen_plan.json'
        frozen = json.loads(frozen_path.read_text())
        old_path = args.run_dir/name/'analysis/frozen_plan.json'
        old = json.loads(old_path.read_text())
        assert frozen['source_frozen_plan_sha256'] == sha256(old_path)
        dims = flash.tensor_dimensions(json.loads((DATA/'models'/f'{name}-metadata.json').read_text()))
        old_g = old['winners']['learned']['group_chips']
        tags = {f'g{old_g}_b{size}' for size in (1,2,4)} | {frozen['winner']['tag']}
        for tag in sorted(tags):
            g = int(tag.split('_')[0][1:])
            plan = materialize(dims, old['mappings'][f'g{g}'], frozen['labels'][tag], g)
            plan.update(model=name, tag=tag, source_frozen_plan_sha256=sha256(frozen_path))
            path = destination/f'{name}_{tag}.json.gz'
            # mtime=0 makes compressed artifacts deterministic across reruns.
            with path.open('wb') as stream:
                stream.write(gzip.compress((json.dumps(plan,separators=(',',':'))+'\n').encode(), mtime=0))
            rows.append(dict(model=name, tag=tag, path=str(path.resolve()), sha256=sha256(path),
                             totals=plan['totals'], checks=plan['checks']))
    write_json(args.results/'physical_capacity.json', rows)
    write_json(destination/'manifest.json', dict(plans=len(rows),
        source_sha256={name:sha256(ROOT/'scripts'/name) for name in
                       ('materialize_static_bundles.py','materialize_plan.py')},
        artifact_type='Virtual fixed same-WL address plans; no physical NAND programmed.', plans_metadata=rows))
    print(f'Wrote {len(rows)} fixed address plans and exact capacity accounts.')


if __name__ == '__main__':
    main()
