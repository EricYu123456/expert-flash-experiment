#!/usr/bin/env python3
"""Audit existing AiF placement interfaces and whole-stream block packing.

No routes or weights are opened.  Existing projection-study plans supply the
immutable row ownership and stream dimensions.  Only virtual NAND addresses are
reordered.  This is an offline capacity certificate, not a device measurement.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
from itertools import combinations
import json
from pathlib import Path

from materialize_plan import WaveAllocator

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'results/research-2026-11-02/round2/existing-hardware'
SOURCE = ROOT / 'results/research-2026-11-02/projection-study'
PAGE = 16384
PLANES = 4
BLOCK_WAVES = 192
WAVE_BYTES = PAGE * PLANES
MODELS = ('granite', 'olmoe', 'deepseek', 'qwen3', 'mixtral')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def read_plan(path):
    with gzip.open(path, 'rt') as stream:
        return json.load(stream)


def make_segments(block, offset, pages, waves):
    assert 0 < waves <= BLOCK_WAVES and offset + waves <= BLOCK_WAVES
    counts = [(pages + PLANES - 1 - p) // PLANES for p in range(PLANES)]
    return [[block, offset, waves, 0, pages, counts]]


def expert_first(plan):
    """One global matrix post order: layer, expert, gate_up before down."""
    allocators = [WaveAllocator() for _ in range(16)]
    order = sorted(range(len(plan['streams'])), key=lambda i: (
        plan['streams'][i][0], plan['streams'][i][1],
        {'gate_up': 0, 'down': 1}[plan['streams'][i][2]], plan['streams'][i][3]))
    records = [None] * len(order)
    for i in order:
        old = plan['streams'][i]
        p, w, runs, segments = allocators[old[3]].allocate(old[5] * old[6])
        assert [p, w, runs] == old[7:10]
        records[i] = old[:10] + [segments]
    return records


def per_chip_ffd(plan):
    """Allocator-only first-fit-decreasing; each immutable stream stays intact.

    Different chips can allocate the same posted matrix into different holes.
    This needs an FTL allocation policy, not a new chip command or PE operation.
    It is not an operand offered by the paper's host-facing aif_post API.
    """
    groups = defaultdict(list)
    for i, old in enumerate(plan['streams']):
        assert old[8] <= BLOCK_WAVES and old[9] == 1
        groups[old[3]].append(i)
    records = [None] * len(plan['streams'])
    for chip, indices in sorted(groups.items()):
        used = []
        for i in sorted(indices, key=lambda j: (-plan['streams'][j][8], j)):
            old = plan['streams'][i]
            waves = old[8]
            block = next((b for b, fill in enumerate(used)
                          if fill + waves <= BLOCK_WAVES), len(used))
            if block == len(used):
                used.append(0)
            segments = make_segments(block, used[block], old[7], waves)
            used[block] += waves
            records[i] = old[:10] + [segments]
    return records


def mixtral_row_safe_wave_fill(plan):
    """Keep down whole; fill remaining holes with GU runs of 16 whole rows/wave.

    Arbitrarily cutting a byte stream at a block boundary would be invalid when
    a down dot product straddles the cut.  This construction never cuts down;
    each GU cut is an exact row boundary with no per-run page/wave padding.
    """
    assert plan['model'] == 'mixtral'
    groups = defaultdict(list)
    for i, old in enumerate(plan['streams']):
        groups[old[3]].append(i)
    records = [None] * len(plan['streams'])
    for chip, indices in sorted(groups.items()):
        total_waves = sum(plan['streams'][i][8] for i in indices)
        assert total_waves % BLOCK_WAVES == 0
        used = [0] * (total_waves // BLOCK_WAVES)
        for i in indices:
            old = plan['streams'][i]
            if old[2] != 'down':
                continue
            assert (old[5], old[6], old[7], old[8]) == (256, 14336, 224, 56)
            block = next(b for b, fill in enumerate(used) if fill + 56 <= BLOCK_WAVES)
            records[i] = old[:10] + [make_segments(block, used[block], old[7], old[8])]
            used[block] += 56
        for i in indices:
            old = plan['streams'][i]
            if old[2] != 'gate_up':
                continue
            assert (old[5], old[6], old[7], old[8]) == (1792, 4096, 448, 112)
            left_waves, page_offset, segments = 112, 0, []
            while left_waves:
                block = next(b for b, fill in enumerate(used) if fill < BLOCK_WAVES)
                w = min(left_waves, BLOCK_WAVES - used[block])
                p = w * PLANES
                segment = make_segments(block, used[block], p, w)[0]
                segment[3] = page_offset
                segments.append(segment)
                used[block] += w
                page_offset += p
                left_waves -= w
            records[i] = old[:9] + [len(segments), segments]
        assert all(fill == BLOCK_WAVES for fill in used)
    return records


def certificate(original, records, allow_extra_commands=False):
    """Independent occupancy, coverage and demand audit of serialized addresses."""
    assert len(records) == len(original['streams'])
    occupied = defaultdict(list)
    coverage = defaultdict(list)
    totals = [Counter() for _ in range(16)]
    stream_sizes = [Counter() for _ in range(16)]
    for old, new in zip(original['streams'], records):
        # This proves demand equivalence for ANY routed selection, not a sample.
        assert new[:9] == old[:9]
        assert new[9] >= old[9]
        if not allow_extra_commands:
            assert new[9] == old[9]
        layer, expert, phase, chip, start, nrows, cols, pages, waves, runs, segments = new
        assert pages == (nrows * cols + PAGE - 1) // PAGE
        assert waves == (pages + PLANES - 1) // PLANES
        assert runs == len(segments)
        coverage[layer, expert, phase].append((start, start + nrows))
        total_pages, total_waves = 0, 0
        for block, wl, sw, page_offset, sp, counts in segments:
            assert page_offset == total_pages
            assert 0 <= wl < BLOCK_WAVES and wl + sw <= BLOCK_WAVES
            assert sum(counts) == sp and max(counts) == sw
            for plane, count in enumerate(counts):
                if count:
                    occupied[chip, plane].append((block * BLOCK_WAVES + wl,
                                                  block * BLOCK_WAVES + wl + count))
            total_pages += sp
            total_waves += sw
            totals[chip]['block_stripes'] = max(totals[chip]['block_stripes'], block + 1)
        assert (total_pages, total_waves) == (pages, waves)
        if len(segments) > old[9]:
            # Native whole-output-row GEMVs only: host concatenates, never sums
            # partial dot products or relies on state persisting across commands.
            row_cursor = start
            for block, wl, sw, page_offset, sp, counts in segments:
                assert page_offset * PAGE % cols == 0
                assert sp * PAGE % cols == 0
                assert page_offset * PAGE // cols == row_cursor - start
                row_cursor += sp * PAGE // cols
            assert row_cursor == start + nrows
        totals[chip].update(streams=1, useful_bytes=nrows * cols,
                            pages=pages, waves=waves, commands=runs)
        stream_sizes[chip][waves] += 1
    for key, ranges in coverage.items():
        end = 0
        for begin, stop in sorted(ranges):
            assert begin == end
            end = stop
    for key, ranges in occupied.items():
        last = 0
        for first, end in sorted(ranges):
            assert first >= last
            last = end
    chips = []
    for chip, counter in enumerate(totals):
        item = dict(chip=chip, **counter)
        item['stream_wave_histogram'] = dict(sorted(stream_sizes[chip].items()))
        item['reserved_lsb_bytes'] = item['block_stripes'] * BLOCK_WAVES * WAVE_BYTES
        item['allocated_wave_bytes'] = item['waves'] * WAVE_BYTES
        item['reserved_idle_waves'] = item['block_stripes'] * BLOCK_WAVES - item['waves']
        item['whole_stream_bin_lower_bound'] = max(
            (item['waves'] + BLOCK_WAVES - 1) // BLOCK_WAVES,
            sum(count for size, count in stream_sizes[chip].items() if size > BLOCK_WAVES // 2))
        item['reaches_bin_lower_bound'] = item['block_stripes'] == item['whole_stream_bin_lower_bound']
        item['wave_volume_block_lower_bound'] = (item['waves'] + BLOCK_WAVES - 1) // BLOCK_WAVES
        item['reaches_wave_volume_bound'] = item['block_stripes'] == item['wave_volume_block_lower_bound']
        chips.append(item)
    total = {k: sum(c[k] for c in chips) for k in (
        'streams', 'useful_bytes', 'pages', 'waves', 'commands', 'block_stripes',
        'reserved_lsb_bytes', 'allocated_wave_bytes', 'reserved_idle_waves')}
    total['reserved_lsb_gib'] = total['reserved_lsb_bytes'] / (1 << 30)
    total['whole_stream_bin_lower_bound'] = sum(c['whole_stream_bin_lower_bound'] for c in chips)
    total['reaches_bin_lower_bound'] = all(c['reaches_bin_lower_bound'] for c in chips)
    total['reaches_wave_volume_bound'] = all(c['reaches_wave_volume_bound'] for c in chips)
    extra_commands = sum(new[9] - old[9] for old, new in zip(original['streams'], records))
    by_expert_phase = Counter()
    extra_inputs = defaultdict(lambda: [0] * 16)
    for old, new in zip(original['streams'], records):
        key = tuple(old[:3])
        delta = new[9] - old[9]
        by_expert_phase[key] += delta
        # Do not assume the chip retains its vector across native GEMV runs.
        # Host/controller already owns the same input: additional PCIe is zero.
        extra_inputs[key][old[3]] += delta * old[6]
    extra_input_ns = {key: max((b[c] + b[c + 8] + 1) // 2 for c in range(8))
                      for key, b in extra_inputs.items()}
    return dict(total=total, chips=chips,
                unchanged_stream_identity_rows_columns_pages_waves=True,
                unchanged_commands=extra_commands == 0,
                extra_inventory_commands=extra_commands,
                max_extra_commands_per_selected_expert_phase=max(by_expert_phase.values()),
                max_extra_control_ns_per_selected_expert_phase=max(by_expert_phase.values()) * 250,
                conservative_extra_native_input_bytes_inventory=sum(sum(b) for b in extra_inputs.values()),
                extra_pcie_input_bytes=0, extra_output_bytes=0,
                max_extra_native_input_ns_per_selected_expert_phase=max(extra_input_ns.values()),
                max_extra_control_plus_input_ns_per_selected_expert_phase=max(
                    by_expert_phase[k] * 250 + extra_input_ns[k] for k in by_expert_phase),
                all_additional_native_runs_have_complete_rows=True,
                cross_run_partial_output_sum_needed=False,
                tensor_row_coverage_once=True, nonoverlapping_chip_plane_addresses=True,
                any_routing_selection_has_identical_modeled_service=extra_commands == 0)


def persist_plan(name, policy, old, records):
    path = DEST / f'{name}.{policy}.physical_plan.json.gz'
    value = {**old, 'artifact_type': 'virtual_addresses_only_no_programmed_weights',
             'policy': policy, 'streams': records}
    # Original chip summaries describe obsolete addresses; replace them.
    allow_extra = policy == 'row_safe_wave_fill'
    value['chips'] = certificate(old, records, allow_extra)['chips']
    value['native_run_row_columns'] = ['stream_index', 'segment_index', 'row_start', 'row_count']
    run_rows = []
    for i, record in enumerate(records):
        for j, segment in enumerate(record[10]):
            if len(record[10]) == 1:
                start, count = record[4:6]
            else:
                assert segment[3] * PAGE % record[6] == 0
                assert segment[4] * PAGE % record[6] == 0
                start = record[4] + segment[3] * PAGE // record[6]
                count = segment[4] * PAGE // record[6]
            run_rows.append([i, j, start, count])
    value['native_run_rows'] = run_rows
    with gzip.GzipFile(filename=str(path), mode='wb', mtime=0) as stream:
        stream.write(json.dumps(value, separators=(',', ':')).encode())
    reread = read_plan(path)
    audit = certificate(old, reread['streams'], allow_extra)
    coverage = defaultdict(list)
    for i, j, start, count in reread['native_run_rows']:
        coverage[i].append((start, start + count))
    for i, record in enumerate(reread['streams']):
        cursor = record[4]
        for start, end in sorted(coverage[i]):
            assert cursor == start
            cursor = end
        assert cursor == record[4] + record[5]
    return dict(path=str(path), sha256=sha256(path), serialized_address_audit=audit)


def mixtral_route_free_overhead(old, records):
    """Enumerate all 28 top-2 sets/layer; no observed routes or hotness fitting."""
    extra = defaultdict(lambda: defaultdict(lambda: [0] * 16))
    for before, after in zip(old['streams'], records):
        layer, expert, phase, chip = before[:4]
        delta = after[9] - before[9]
        if phase == 'down':
            assert delta == 0
        else:
            extra[layer][expert][chip] = delta
    minimum, maximum, uniform_mean = 0, 0, 0.0
    for layer, experts in extra.items():
        costs = []
        for selected in combinations(sorted(experts), 2):
            delta = [sum(experts[e][c] for e in selected) for c in range(16)]
            control = sum(delta) * 250
            vector_reload = max((delta[c] + delta[c + 8]) * 4096 // 2 for c in range(8))
            costs.append(control + vector_reload)
        assert len(costs) == 28
        minimum += min(costs)
        maximum += max(costs)
        uniform_mean += sum(costs) / len(costs)
    nand_ns = len(extra) * 2 * (112 + 56) * 28000
    return dict(active_experts=2, layers=len(extra), observed_routes_opened=False,
                combinations_per_layer=28, original_ffn_nand_ns_per_token=nand_ns,
                extra_control_plus_conservative_input_ns_per_token_minimum=minimum,
                extra_control_plus_conservative_input_ns_per_token_maximum=maximum,
                uniform_top2_mean_extra_ns_per_token=uniform_mean,
                maximum_extra_fraction_of_original_nand=maximum / nand_ns,
                note='Bounds concern the current additive service model, not hardware or '
                     'end-to-end inference. Uniform top-2 is a synthetic expectation, '
                     'not an empirical model route distribution. Host activation is excluded.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEST)
    args = parser.parse_args()
    # Keep a fixed own destination: no input artifact can be overwritten.
    assert args.output.resolve() == DEST.resolve()
    DEST.mkdir(parents=True, exist_ok=True)
    source_paths = [Path(__file__), ROOT / 'kernel/aif.c', ROOT / 'kernel/aif.h',
                    ROOT / 'paper/aif.txt', ROOT / 'docs/vm-aif-abi.md',
                    ROOT / 'scripts/materialize_plan.py', ROOT / 'scripts/aif_model.py',
                    ROOT / 'scripts/research_20261102_projection_study.py']
    hashes_before = {str(p): sha256(p) for p in source_paths}
    results = []
    for name in MODELS:
        path = SOURCE / f'{name}.physical_plan.json.gz'
        before_sha = sha256(path)
        old = read_plan(path)
        baseline = certificate(old, old['streams'])
        for chip, saved in zip(baseline['chips'], old['chips']):
            assert chip['reserved_lsb_bytes'] == saved['reserved_lsb_bytes']
            assert chip['waves'] == saved['read_rounds']
            assert chip['commands'] == saved['cold_runs']
        model_result = dict(model=name, source_path=str(path), source_sha256=before_sha,
                            baseline=baseline, policies={})
        candidates = [('global_expert_first', expert_first(old)),
                      ('per_chip_ffd', per_chip_ffd(old))]
        if name == 'mixtral':
            candidates.append(('row_safe_wave_fill', mixtral_row_safe_wave_fill(old)))
        for policy, records in candidates:
            audit = certificate(old, records, policy == 'row_safe_wave_fill')
            base = baseline['total']['reserved_lsb_bytes']
            packed = audit['total']['reserved_lsb_bytes']
            result = dict(audit=audit, capacity_saving_bytes=base - packed,
                          capacity_saving_fraction=(base - packed) / base,
                          modeled_latency_change_fraction=(0.0 if audit['unchanged_commands'] else None),
                          model_command_change=audit['extra_inventory_commands'], nand_wave_change=0)
            if policy == 'row_safe_wave_fill':
                result['route_free_overhead_bounds'] = mixtral_route_free_overhead(old, records)
            if packed < base:
                result['physical_plan'] = persist_plan(name, policy, old, records)
            model_result['policies'][policy] = result
        assert sha256(path) == before_sha
        results.append(model_result)
    hashes_after = {str(p): sha256(p) for p in source_paths}
    assert hashes_before == hashes_after
    value = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        scope='Existing INT8 whole-row streams; normal reads; virtual physical addresses only.',
        routes_opened=False, weights_opened=False, vm_or_hardware_modified=False,
        source_sha256=hashes_after,
        geometry=dict(chips=16, planes=4, page_bytes=PAGE, wordlines_per_block=BLOCK_WAVES,
                      bytes_per_four_plane_wave=WAVE_BYTES),
        costs=dict(normal_read_ns_per_wave=28000, ecc_mac_ceiling_bytes_per_ns=6.4,
                   chip_matrix_command_ns=250, host_phase_ns=2000,
                   latency_assumptions_not_hardware_calibrated=True),
        invariants='Each stream retains its chip, exact contiguous logical rows, columns, '
                   'page count and wave count. The expert-first/FFD policies also retain '
                   'the block-command count. Row-safe fill instead pays additional '
                   'per-block commands, 250ns each, and conservatively reloads the full '
                   'input subvector over ONFI per additional native run. No extra PCIe '
                   'input or output bytes; the original complete rows are concatenated.',
        global_expert_first_scope='A single matrix posting order; software allocation only. '
            'The paper does not guarantee a particular free-block policy or short-stream '
            'cohabitation, so public aif_post alone cannot guarantee these physical addresses.',
        per_chip_ffd_scope='A controller/FTL free-space allocation policy. Uses existing '
            'block/offset/page-count chip operands; no chip PE/ISA addition. Not exposed '
            'by the published host API, and its unequal chosen row ownership already '
            'requires controller layout support beyond the paper\'s equal-row aif_post.',
        row_safe_wave_fill_scope='Mixtral only: keep each down stream whole and fill '
            'the remaining wave slots with GU runs, 16 complete output rows per wave. '
            'All extra runs return distinct complete rows; host/controller only concatenate. '
            'No partial-dot sum, extra row padding, resident accumulator or new chip ISA. '
            'Uses the existing per-block GEMV command described in paper footnote 9; requires '
            'controller mapping/allocation software to follow the row-run table. No added '
            'host command is assumed because the controller handles the runs. Only metadata '
            'command overhead is modeled; mapping-table lookup cost is unmeasured.',
        capacity_scope='IFP LSB reserved stripes; CSB/MSB remain available for general data. '
            'Three times this value is covered TLC raw address space, not three times '
            'additional occupied model bytes. This inherits the existing planner\'s I8 '
            'payload accounting and excludes GGUF/Q8 scale arrays, FTL tables and other '
            'model tensors. Runtime free-space/GC effects are unmodeled.',
        no_latency_claim='Boundary holes are not read. Reducing reserved holes does not '
            'reduce the NAND work bound. Whole-stream packing keeps commands unchanged; '
            'compact crossing can only increase modeled control cost.',
        models=results)
    write_json(DEST / 'packing.json', value)
    for result in results:
        print(result['model'], 'baseline_GiB=', result['baseline']['total']['reserved_lsb_gib'],
              {p: dict(saving_percent=round(r['capacity_saving_fraction'] * 100, 6),
                       GiB=r['audit']['total']['reserved_lsb_gib'],
                       optimal_whole_stream=r['audit']['total']['reaches_bin_lower_bound'])
               for p, r in result['policies'].items()})


if __name__ == '__main__':
    main()
