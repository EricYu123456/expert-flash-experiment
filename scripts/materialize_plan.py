#!/usr/bin/env python3
"""Materialize virtual INT8 MoE stream placement into nonoverlapping NAND slots.

This writes address metadata only, never actual weights or NVMe namespace data.
Complete short streams share a block; streams never cross an extra block boundary
relative to aif_model.Demand.add_stream's cold-run count.
"""
import argparse
import hashlib
import json
from pathlib import Path

import aif_model as model

BLOCK_PAGES = 192
STREAM_COLUMNS = ['layer', 'logical_expert', 'projection', 'placement_slot', 'chip',
                  'row_start', 'row_count', 'input_dimension', 'useful_int8_bytes',
                  'physical_pages', 'read_rounds', 'cold_runs', 'segments']
SEGMENT_COLUMNS = ['block_id_per_plane', 'wordline_offset', 'read_waves',
                   'stream_page_offset', 'physical_pages', 'plane_page_counts']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class WaveAllocator:
    """Reserve whole four-plane waves, sharing block stripes between streams."""
    def __init__(self, block_pages=BLOCK_PAGES):
        self.block_pages = block_pages
        self.cursor = self.padding = self.useful = self.pages = self.rounds = self.cold = 0
        self.streams = self.nonzero_starts = self.last_end = 0

    def allocate(self, nbytes):
        if nbytes <= 0:
            raise ValueError('Cannot allocate an empty stream')
        pages = model.ceildiv(nbytes, model.PAGE)
        rounds = model.ceildiv(pages, model.PLANES)
        offset = self.cursor % self.block_pages
        remaining = self.block_pages - offset
        # Long streams start at a block boundary. A short stream is placed only
        # if it fits wholly in this block; otherwise pad its unused suffix.
        if offset and (rounds > self.block_pages or rounds > remaining):
            self.cursor += remaining
            self.padding += remaining
        first_start = self.cursor
        assert first_start >= self.last_end
        self.nonzero_starts += first_start % self.block_pages != 0
        segments, left_rounds, left_pages, consumed_pages = [], rounds, pages, 0
        while left_rounds:
            block, wl = divmod(self.cursor, self.block_pages)
            segment_rounds = min(left_rounds, self.block_pages - wl)
            segment_pages = min(left_pages, segment_rounds * model.PLANES)
            per_plane = [(segment_pages + model.PLANES - 1 - plane) // model.PLANES
                         for plane in range(model.PLANES)]
            assert sum(per_plane) == segment_pages
            assert max(per_plane) == segment_rounds
            assert all(wl + count <= self.block_pages for count in per_plane)
            segments.append([block, wl, segment_rounds, consumed_pages, segment_pages, per_plane])
            self.cursor += segment_rounds
            left_rounds -= segment_rounds
            left_pages -= segment_pages
            consumed_pages += segment_pages
        self.last_end = self.cursor
        expected = model.Demand()
        expected.add_stream(nbytes, self.block_pages, 1)
        assert (pages, rounds, len(segments)) == (expected.pages, expected.rounds, expected.cold)
        assert left_pages == 0 and consumed_pages == pages
        if rounds <= self.block_pages:
            assert len(segments) == 1
        self.useful += nbytes
        self.pages += pages
        self.rounds += rounds
        self.cold += len(segments)
        self.streams += 1
        return pages, rounds, len(segments), segments

    def summary(self, chip):
        stripes = model.ceildiv(self.cursor, self.block_pages)
        reserved_waves = stripes * self.block_pages
        tail_waves = reserved_waves - self.cursor
        wave_bytes = model.PLANES * model.PAGE
        assert reserved_waves == self.rounds + self.padding + tail_waves
        assert self.pages <= self.rounds * model.PLANES
        return dict(chip=chip, streams=self.streams, block_stripes=stripes,
                    useful_int8_bytes=self.useful, physical_page_bytes=self.pages * model.PAGE,
                    allocated_read_wave_bytes=self.rounds * wave_bytes,
                    idle_plane_slot_bytes=(self.rounds * model.PLANES - self.pages) * model.PAGE,
                    intra_page_padding_bytes=self.pages * model.PAGE - self.useful,
                    boundary_padding_waves=self.padding, final_block_tail_waves=tail_waves,
                    nonzero_wordline_stream_starts=self.nonzero_starts,
                    reserved_lsb_bytes=reserved_waves * wave_bytes,
                    reserved_tlc_raw_bytes=3 * reserved_waves * wave_bytes,
                    physical_pages=self.pages, read_rounds=self.rounds, cold_runs=self.cold)


def self_check():
    # Small streams share a block; a non-fitting short stream pads to the next
    # block. A large stream starts afresh and crosses exactly the modeled runs.
    allocator = WaveAllocator(4)
    a = allocator.allocate(5 * model.PAGE)       # two waves, WL 0
    b = allocator.allocate(model.PAGE)           # one wave, WL 2
    c = allocator.allocate(5 * model.PAGE)       # two waves, new block WL 0
    d = allocator.allocate(9 * model.PAGE)       # three waves, next block WL 0
    e = allocator.allocate(17 * model.PAGE)      # five waves, two block runs
    assert a[3][0][:2] == [0, 0] and b[3][0][:2] == [0, 2]
    assert c[3][0][:2] == [1, 0] and d[3][0][:2] == [2, 0]
    assert e[2] == 2 and [segment[0] for segment in e[3]] == [3, 4]
    assert allocator.padding == 4
    allocator.summary(0)


def make_plan(metadata, specification, source_refs):
    dims = model.tensor_dimensions(metadata)
    architecture = metadata['metadata']['general.architecture']
    expert_count = metadata['metadata'][architecture + '.expert_count']
    group = specification['group_chips']
    if group not in (1, 2, 4, 8, 16):
        raise ValueError('Expected a fixed supported group_chips value')
    mappings = specification['mappings']
    allocators = [WaveAllocator() for _ in range(model.CHIPS)]
    records, logical_useful = [], 0
    for layer in sorted(dims):
        current = dims[layer]
        slots = mappings[str(layer)] if mappings is not None else list(range(expert_count))
        if sorted(slots) != list(range(expert_count)):
            raise ValueError(f'Layer {layer} mapping is not capacity-preserving')
        if specification['stack_gate_up']:
            projections = [('gate_up', current['gate'][0] + current['up'][0], current['up'][1])]
            projections.append(('down', *current['down']))
        else:
            projections = [(name, *current[name]) for name in ('gate', 'up', 'down')]
        for expert in range(expert_count):
            slot = slots[expert]
            first_chip = slot % (model.CHIPS // group) * group
            for projection, rows, columns in projections:
                logical_useful += rows * columns
                covered = 0
                for shard in range(group):
                    row_count = rows // group + (shard < rows % group)
                    if not row_count:
                        continue
                    chip = first_chip + shard
                    row_start = shard * (rows // group) + min(shard, rows % group)
                    assert row_start == covered
                    nbytes = row_count * columns
                    pages, rounds, cold, segments = allocators[chip].allocate(nbytes)
                    records.append([layer, expert, projection, slot, chip, row_start,
                                    row_count, columns, nbytes, pages, rounds, cold, segments])
                    covered += row_count
                assert covered == rows
    chip_summary = [allocator.summary(chip) for chip, allocator in enumerate(allocators)]
    total_keys = [key for key in chip_summary[0] if key != 'chip']
    totals = {key: sum(record[key] for record in chip_summary) for key in total_keys}
    assert totals['useful_int8_bytes'] == logical_useful
    assert totals['reserved_lsb_bytes'] == (totals['physical_page_bytes'] +
        totals['idle_plane_slot_bytes'] + (totals['boundary_padding_waves'] +
        totals['final_block_tail_waves']) * model.PLANES * model.PAGE)
    totals.update(lsb_fraction_of_1tb=totals['reserved_lsb_bytes'] / 1_000_000_000_000,
                  tlc_raw_fraction_of_1tb=totals['reserved_tlc_raw_bytes'] / 1_000_000_000_000,
                  lsb_utilization=logical_useful / totals['reserved_lsb_bytes'])
    ordering = {str(layer): dict(gate_rows=[0, current['gate'][0]],
                                up_rows=[current['gate'][0], current['gate'][0]+current['up'][0]],
                                input_dimension=current['up'][1])
                for layer, current in dims.items()}
    return dict(schema_version=1, artifact_type='virtual_int8_nand_address_plan_no_weight_copy',
                heldout_prompt=specification['heldout_prompt'], policy=specification['policy'],
                trained_configuration=specification, source=source_refs,
                geometry=dict(chips=model.CHIPS, planes_per_chip=model.PLANES,
                              bytes_per_lsb_page=model.PAGE, lsb_wordlines_per_block=BLOCK_PAGES,
                              block_stripe='Same block_id_per_plane in all four planes of one chip',
                              chip_channel='chip % 8', reference_ssd_bytes=1_000_000_000_000),
                stream_columns=STREAM_COLUMNS, segment_columns=SEGMENT_COLUMNS,
                address_rule='Stream page p maps to plane p%4 and WL segment.wordline_offset + floor((p-segment.stream_page_offset)/4); page numbering restarts at plane 0 per segment.',
                stream_data_order='Contiguous row-major INT8 shard; physical last page can contain byte padding',
                stacked_output_order=ordering if specification['stack_gate_up'] else None,
                stacked_output_rule='gate_up output[0:I] is gate and output[I:2I] is up; apply activation/multiply before dependent down' if specification['stack_gate_up'] else None,
                streams=records, chip_summary=chip_summary, totals=totals,
                checks=dict(complete_rows_covered_once=True, fixed_mapping_capacity_preserved=True,
                            nonoverlapping_wave_ranges_per_chip=True,
                            all_page_round_cold_counts_match_aif_model=True,
                            short_streams_are_within_one_block=True),
                limitations=['Virtual addresses and INT8 size only; no actual model weights or namespace pages were written.',
                             'Only routed expert FFN weights are placed; router, attention, embeddings, shared experts, ECC parity, and SSD spare capacity are excluded.',
                             '192 LSB WLs/block is inferred from characterized paper NAND, not a disclosed emulator constant.',
                             'Three times reserved LSB bytes is raw TLC cell capacity; CSB/MSB pages could hold unrelated data in the paper design.',
                             'Existing 2 GB virtual NVMe namespace is a metadata replay test device, not claimed storage for these weights.'])


def main():
    self_check()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-manifest', type=Path, required=True)
    parser.add_argument('--plans', type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    source = json.loads(args.run_manifest.read_text())
    metadata_path = Path(source['metadata'])
    metadata = json.loads(metadata_path.read_text())
    plans_path = args.plans or args.run_manifest.parent / 'learned_overhead/plans.json'
    specifications = json.loads(plans_path.read_text())
    output_dir = args.output_dir or plans_path.parent / 'physical_plans'
    output_dir.mkdir(parents=True, exist_ok=True)
    references = dict(run_manifest=str(args.run_manifest.resolve()),
                      run_manifest_sha256=digest(args.run_manifest),
                      metadata=str(metadata_path.resolve()), metadata_sha256=digest(metadata_path),
                      model_sha256=metadata['sha256'], plans=str(plans_path.resolve()),
                      plans_sha256=digest(plans_path), script_sha256=digest(Path(__file__)),
                      timing_model_sha256=digest(Path(model.__file__)))
    summary = []
    for specification in specifications:
        name = specification['heldout_prompt'] + '_' + specification['policy']
        if not name.replace('_', '').isalnum():
            raise ValueError('Unsafe output plan name')
        plan = make_plan(metadata, specification, references)
        destination = output_dir / (name + '.json')
        destination.write_text(json.dumps(plan, separators=(',', ':')) + '\n')
        summary.append(dict(heldout_prompt=specification['heldout_prompt'],
                            policy=specification['policy'], group_chips=specification['group_chips'],
                            stack_gate_up=specification['stack_gate_up'],
                            path=str(destination), file_bytes=destination.stat().st_size,
                            **plan['totals']))
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
