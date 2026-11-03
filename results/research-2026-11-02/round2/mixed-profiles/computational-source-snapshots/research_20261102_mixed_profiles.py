#!/usr/bin/env python3
"""Small train-only mixed whole-row profile screen on existing AiF primitives.

All widths 1..16 use cyclic contiguous masks; integer rows are distributed by
floor/ceil, not fractionally.  Fixed-width cyclic controls distinguish changing
job granularity from merely relaxing old disjoint group masks/counts.
"""
import argparse
from collections import Counter
import ctypes
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from materialize_plan import WaveAllocator
from research_20261102_diagnostics import row_template
from research_20261102_projection_study import evaluate, scalar_cost

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/research-2026-11-02'
DEST = BASE / 'round2/mixed-profiles'
CAPACITY = 20 << 30


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_snapshot(path):
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def profile_menu(rows, columns):
    result, labels, widths = [], [], []
    for width in range(1, 17):
        initial = np.zeros(16, dtype=np.int32)
        initial[:width] = rows // width
        initial[:rows % width] += 1
        for offset in range(16):
            value = np.roll(initial, offset)
            if any(np.array_equal(value, old) for old in result):
                continue
            result.append(value); labels.append(f'g{width}_offset{offset}'); widths.append(width)
    template = np.asarray(row_template(rows, columns), dtype=np.int32)
    for offset in range(16):
        value = np.roll(template, offset)
        if any(np.array_equal(value, old) for old in result):
            continue
        result.append(value); labels.append(f'wave_template_offset{offset}'); widths.append(0)
    values = np.ascontiguousarray(result, dtype=np.int32)
    assert np.all(values.sum(axis=1) == rows) and np.all(values >= 0)
    return values, labels, np.asarray(widths)


def load_library():
    source = Path(__file__).with_suffix('.cpp')
    target = DEST / 'mixed_profiles.so'
    subprocess.run(['g++', '-O3', '-std=c++17', '-shared', '-fPIC', str(source), '-o', str(target)], check=True)
    lib = ctypes.CDLL(str(target))
    ptr32 = np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
    ptr64 = np.ctypeslib.ndpointer(dtype=np.int64, flags='C_CONTIGUOUS')
    lib.optimize_mixed_profiles.argtypes = [ptr32, ptr32, ctypes.c_int, ptr32, ptr64] + [ctypes.c_int] * 6 + [ctypes.c_uint64, ptr64]
    lib.optimize_mixed_profiles.restype = ctypes.c_double
    return lib


def optimize(lib, initial, menu, unique, weights, phase, passes, seed):
    rows = np.ascontiguousarray(initial, dtype=np.int32)
    menu = np.ascontiguousarray(menu, dtype=np.int32)
    stats = np.zeros(6, dtype=np.int64)
    start = time.time()
    mean = lib.optimize_mixed_profiles(rows, menu, len(menu), unique, weights,
        len(rows), len(unique), unique.shape[1], phase['columns'], int(phase['common']), passes, seed, stats)
    ref = evaluate(unique, rows, phase['columns'], phase['common'])
    reference = float(ref['model_ns'] @ weights / weights.sum())
    assert abs(mean - reference) < 1e-6, (mean, reference)
    assert np.all(rows.sum(axis=1) == phase['total_rows'])
    for index in sorted({0, len(unique) // 2, len(unique) - 1}):
        scalar = scalar_cost(unique[index], rows, phase['columns'], phase['common'])
        assert all(int(ref[k][index]) == value for k, value in scalar.items())
    return dict(mean_ns=reference, rows=rows.tolist(),
        components_ns={k: float(v @ weights / weights.sum()) for k, v in ref.items()},
        passes=int(stats[0]), accepted=int(stats[1]), candidate_evaluations=int(stats[2]),
        converged=bool(stats[5]), seconds=time.time() - start)


def mean_cost(unique, weights, rows, phase):
    return float(evaluate(unique, np.asarray(rows), phase['columns'], phase['common'])['model_ns'] @ weights / weights.sum())


def lower_bound(menu, active, phase):
    """Optimistic profile-aware work/command bound, ignoring chip masks.

    DP picks one menu geometry per selected expert. It retains each expert's
    minimum possible critical chip demand, total waves, and native commands.
    Aggregated work may be divided ideally across 16 chips in this relaxation.
    """
    waves = (menu.astype(np.int64) * phase['columns'] + 65535) // 65536
    options = sorted(set((int(w.sum()), int(w.max()), int(((w + 191) // 192).sum())) for w in waves))
    states = {(0, 0): 0}
    for _ in range(active):
        next_states = {}
        for (total, peak), commands in states.items():
            for work, maximum, cmd in options:
                key = (total + work, max(peak, maximum))
                next_states[key] = min(next_states.get(key, 1 << 60), commands + cmd)
        states = next_states
    best = min((max(peak, (total + 15) // 16) * 28000 + commands * 250,
                total, peak, commands) for (total, peak), commands in states.items())
    host_input = phase['columns'] * (1 if phase['common'] else active)
    host_output = active * phase['total_rows'] * 4
    return dict(mean_ns_lower_bound=best[0] + 2000 + (host_input + 7) // 8 + (host_output + 7) // 8,
                flash_plus_control_ns_lower_bound=best[0], total_wave_work=best[1],
                single_expert_peak_waves=best[2], commands=best[3],
                oracle='Route-adaptive profile choices and ideally balanced aggregate work; '
                       'fixed masks, ONFI fanout and cross-event consistency are relaxed. Not executable.')


def capacity_plan(name, phases, output=False):
    alloc = [WaveAllocator() for _ in range(16)]
    records = []
    for phase in phases:
        for expert, row in enumerate(phase['rows']):
            cursor = 0
            for chip, count in enumerate(row):
                count = int(count)
                if count:
                    pages, waves, commands, segments = alloc[chip].allocate(count * phase['columns'])
                    if output:
                        records.append([phase['layer'], phase['name'], expert, chip, cursor, count,
                            phase['columns'], pages, waves, commands, segments])
                cursor += count
            assert cursor == phase['total_rows']
    chips = [a.summary(c) for c, a in enumerate(alloc)]
    useful = sum(c['useful_int8_bytes'] for c in chips)
    value = dict(chips=chips, useful_int8_bytes=useful,
        physical_page_bytes=sum(c['physical_page_bytes'] for c in chips),
        allocated_wave_bytes=sum(c['allocated_read_wave_bytes'] for c in chips),
        reserved_lsb_bytes=sum(c['reserved_lsb_bytes'] for c in chips),
        maximum_chip_reserved_lsb_bytes=max(c['reserved_lsb_bytes'] for c in chips),
        actual_chip_limit_bytes=CAPACITY, actual_device_total_bytes=16 * CAPACITY,
        actual_capacity_assumption='20GiB usable LSB/chip scenario supplied by task, '
            'not a vendor measured disclosure. Covered TLC raw address space is 3x '
            'LSB reservation; all reported candidates also fit 20GiB raw/chip.',
        maximum_chip_covered_tlc_raw_bytes=3 * max(c['reserved_lsb_bytes'] for c in chips),
        fits_actual_20gib_per_chip=all(c['reserved_lsb_bytes'] <= CAPACITY for c in chips),
        model_payload_equal_share_reference_bytes=useful / 16,
        maximum_useful_to_equal_share_ratio=max(c['useful_int8_bytes'] for c in chips) / (useful / 16),
        namespace_warning='The logical shared namespace has no equal-share physical quota '
            'per chip. useful/16 is a comparison reference, not the 20GiB actual chip limit. '
            'I8 routed FFN payload only; GGUF scales, other tensors, FTL metadata and '
            'CSB/MSB capacity are not included.')
    assert value['fits_actual_20gib_per_chip']
    assert value['maximum_chip_covered_tlc_raw_bytes'] <= CAPACITY
    if output:
        path = DEST / f'{name}.physical_plan.json.gz'
        payload = dict(artifact='virtual addresses, no programmed weights',
            columns=['layer', 'phase', 'expert', 'chip', 'row_start', 'rows', 'columns',
                     'pages', 'waves', 'commands', 'segments'], streams=records, chips=chips)
        with gzip.GzipFile(filename=str(path), mode='wb', mtime=0) as stream:
            stream.write(json.dumps(payload, separators=(',', ':')).encode())
        # Check serialized row and address coverage independent of allocation cursor.
        with gzip.open(path, 'rt') as stream:
            reread = json.load(stream)
        ranges, tensor_rows = {}, {}
        for layer, pname, e, c, first, count, cols, pages, waves, commands, segments in reread['streams']:
            key = layer, pname, e
            assert tensor_rows.get(key, 0) == first
            tensor_rows[key] = first + count
            assert pages == (count * cols + 16383) // 16384
            assert waves == (pages + 3) // 4 and commands == len(segments)
            consumed = 0
            for block, wl, w, offset, p, plane_counts in segments:
                assert offset == consumed and sum(plane_counts) == p and max(plane_counts) == w
                assert wl + w <= 192
                for plane, length in enumerate(plane_counts):
                    if length:
                        ranges.setdefault((c, plane), []).append((block * 192 + wl, block * 192 + wl + length))
                consumed += p
            assert consumed == pages
        for phase in phases:
            for expert in range(len(phase['rows'])):
                assert tensor_rows[phase['layer'], phase['name'], expert] == phase['total_rows']
        for intervals in ranges.values():
            end = 0
            for first, stop in sorted(intervals):
                assert first >= end
                end = stop
        value['physical_plan'] = dict(path=str(path), sha256=sha(path),
            row_coverage_once=True, nonoverlapping_chip_plane_addresses=True)
    return value


def run_model(name, lib, passes, controls):
    baseline_path = BASE / 'round2/unbalanced-group' / f'{name}.screen.strong.frozen.json'
    baseline, baseline_sha = read_snapshot(baseline_path)
    source_path = Path(baseline['source_plan'])
    source, source_sha = read_snapshot(source_path)
    with np.load(baseline['training_routes']) as data:
        ids = data['prompt_ids'][:32].tolist()
        routes = data['routes'][:32, :64].copy()
    assert len(ids) == 32 and all(p.startswith('train_') for p in ids)
    assert routes.shape[:2] == (32, 64)
    source_rows_key = 'rows' if name == 'granite4tiny' else 'chosen_rows'
    full_phases = [dict(layer=p['layer'], name=p['name'], total_rows=p['total_rows'],
        columns=p['columns'], common=p['common'], rows=p[source_rows_key]) for p in source['phases']]
    original_capacity = capacity_plan(name, full_phases)
    results = []
    for pi, saved_phase in enumerate(baseline['phases']):
        phase = dict(saved_phase)
        source_phase = next(p for p in source['phases'] if
            (p['layer'], p['name']) == (phase['layer'], phase['name']))
        phase['original_main_rows'] = source_phase[source_rows_key]
        phase['candidates'] = sum((phase.get(key, []) for key in
            ('candidates', 'balanced_controls', 'unbalanced_candidates',
             'greedy_from_balanced_controls')), [])
        li = baseline['layers'].index(phase['layer'])
        unique, weights = np.unique(np.sort(routes[:, :, li].reshape(-1, baseline['active']), axis=1),
                                    axis=0, return_counts=True)
        unique = np.ascontiguousarray(unique, dtype=np.int32)
        weights = np.ascontiguousarray(weights, dtype=np.int64)
        menu, labels, widths = profile_menu(phase['total_rows'], phase['columns'])
        starts = [('original_main', phase['original_main_rows']), ('unbalanced_disjoint', phase['chosen_rows'])]
        original_ns = mean_cost(unique, weights, starts[0][1], phase)
        unbalanced_ns = mean_cost(unique, weights, starts[1][1], phase)
        assert abs(unbalanced_ns - phase['chosen_ns']) < 1e-6
        assert abs(original_ns - phase['original_main_ns']) < 1e-6
        fixed = []
        if controls:
            for width in range(1, 17):
                candidates = [c for c in phase['candidates'] if c['g'] == width]
                subset = menu[widths == width]
                initial = [subset[e % len(subset)].tolist() for e in range(baseline['experts'])]
                if candidates:
                    initial = min(candidates, key=lambda c: c['mean_ns'])['rows']
                else:
                    current = np.asarray(phase['original_main_rows'])
                    if np.all((current > 0).sum(axis=1) == width) and all(
                        any(np.array_equal(row, p) for p in subset) for row in current):
                        initial = current
                fitted = optimize(lib, initial, subset, unique, weights, phase, passes,
                                  20261102 + pi * 1009 + width * 31)
                fixed.append(dict(width=width, **fitted))
            if np.any(widths == 0):
                subset = menu[widths == 0]
                initial = [subset[e % len(subset)].tolist() for e in range(baseline['experts'])]
                fitted = optimize(lib, initial, subset, unique, weights, phase, passes,
                                  20261102 + pi * 1009 + 701)
                fixed.append(dict(width=0, family='wave_template_rotations', **fitted))
            best_fixed = min(fixed, key=lambda c: c['mean_ns'])
            starts.append(('best_fixed_width', best_fixed['rows']))
            waves = (menu.astype(np.int64) * phase['columns'] + 65535) // 65536
            minimum_work = int(waves.sum(axis=1).min())
            for cell in fixed:
                family_menu = menu[widths == cell['width']]
                family_waves = (family_menu.astype(np.int64) * phase['columns'] + 65535) // 65536
                # Include ALL minimum-work width seeds, even coarse stationary
                # starts, to expose rather than conceal coordinate barriers.
                if cell['width'] == 0 or int(family_waves.sum(axis=1).min()) == minimum_work:
                    starts.append(('minimum_work_g' + str(cell['width']), cell['rows']))
        else:
            best_fixed = None
        searches = []
        seen = set()
        for si, (label, initial) in enumerate(starts):
            key = np.asarray(initial, dtype=np.int32).tobytes()
            if key in seen:
                continue
            seen.add(key)
            fitted = optimize(lib, initial, menu, unique, weights, phase, passes,
                              20261102 + pi * 1009 + si * 53)
            searches.append(dict(start=label, **fitted))
        winner = min(searches, key=lambda s: s['mean_ns'])
        # Preserve original/main if the restricted search cannot improve it.
        assert winner['mean_ns'] <= min(original_ns, unbalanced_ns) + 1e-6
        winner_rows = np.asarray(winner['rows'])
        assignment = []
        for row in winner_rows:
            match = [i for i, profile in enumerate(menu) if np.array_equal(row, profile)]
            assignment.append(labels[match[0]] if match else 'unchanged_original_main_profile')
        families = Counter(p.split('_offset')[0] for p in assignment)
        raw_reference = min(original_ns, unbalanced_ns,
            best_fixed['mean_ns'] if best_fixed else float('inf'))
        same_family_feasible = min(raw_reference,
            winner['mean_ns'] if len(families) == 1 else float('inf'))
        geometry_table = []
        for width in sorted(set(widths.tolist())):
            sample = menu[np.flatnonzero(widths == width)[0]].astype(np.int64)
            pages = (sample * phase['columns'] + 16383) // 16384
            waves = (pages + 3) // 4
            geometry_table.append(dict(width=width, family='wave_template' if width == 0 else 'equal_rows',
                row_multiset=sorted(map(int, sample)), total_pages=int(pages.sum()),
                total_waves=int(waves.sum()), maximum_tile_waves=int(waves.max()),
                native_commands=int(((waves + 191) // 192).sum())))
        result = dict(layer=phase['layer'], name=phase['name'], total_rows=phase['total_rows'],
            columns=phase['columns'], common=phase['common'], profiles=len(menu),
            original_main_ns=original_ns, unbalanced_disjoint_ns=unbalanced_ns,
            best_fixed_width_ns=best_fixed['mean_ns'] if best_fixed else None,
            best_fixed_width=best_fixed['width'] if best_fixed else None,
            chosen_ns=winner['mean_ns'], chosen_rows=winner['rows'],
            profiles_by_expert=assignment, profile_histogram=dict(Counter(assignment)),
            profile_family_histogram=dict(families),
            all_final_experts_same_profile_family=len(families) == 1,
            feasible_single_profile_control_ns=same_family_feasible,
            attributable_mixed_profile_gain_fraction=1 - winner['mean_ns'] / same_family_feasible,
            uniform_control_note='If the final mixed-search result uses one profile family '
                'for all experts, it is itself a feasible single-profile control. Its '
                'extra search cannot be credited to allowing mixed profiles.',
            width_histogram=dict(Counter(map(int, (winner_rows > 0).sum(axis=1)))),
            geometry_table=geometry_table,
            components_ns=winner['components_ns'],
            gain_vs_strongest_reference_fraction=1 - winner['mean_ns'] / min(
                original_ns, unbalanced_ns, best_fixed['mean_ns'] if best_fixed else float('inf')),
            profile_work_command_relaxation=lower_bound(menu, baseline['active'], phase),
            fixed_width_controls=fixed, mixed_searches=searches)
        results.append(result)
        for whole in full_phases:
            if (whole['layer'], whole['name']) == (phase['layer'], phase['name']):
                whole['rows'] = winner['rows']
        print(name, phase['layer'], phase['name'], 'main', round(original_ns),
              'unbalanced', round(unbalanced_ns), 'same_width', round(best_fixed['mean_ns']) if best_fixed else None,
              'mixed', round(winner['mean_ns']), 'width_counts', result['width_histogram'], flush=True)
        write(DEST / (name + '.partial.json'), results)
    final_capacity = capacity_plan(name, full_phases, True)
    assert final_capacity['useful_int8_bytes'] == original_capacity['useful_int8_bytes']
    totals = {k: sum(p[k] for p in results) for k in
              ('original_main_ns', 'unbalanced_disjoint_ns', 'chosen_ns')}
    if controls:
        totals['best_fixed_width_ns'] = sum(p['best_fixed_width_ns'] for p in results)
        totals['strongest_per_phase_reference_ns'] = sum(min(p['original_main_ns'],
            p['unbalanced_disjoint_ns'], p['best_fixed_width_ns']) for p in results)
        totals['gain_vs_strongest_reference_fraction'] = 1 - totals['chosen_ns'] / totals['strongest_per_phase_reference_ns']
        totals['single_profile_feasible_control_ns'] = sum(p['feasible_single_profile_control_ns'] for p in results)
        totals['attributable_mixed_profile_gain_fraction'] = 1 - totals['chosen_ns'] / totals['single_profile_feasible_control_ns']
    output = dict(model=name, status='exploratory train-only small-layer screen',
        sample='32 original training prompts x first 64 decode steps, exact selected routes',
        train_ids=ids, training_routes=baseline['training_routes'],
        training_routes_sha256=sha(baseline['training_routes']),
        baseline_snapshot_path=str(baseline_path), baseline_snapshot_sha256=baseline_sha,
        baseline_snapshot=baseline, original_source_path=str(source_path), original_source_sha256=source_sha,
        source_sha256={str(p): sha(p) for p in [Path(__file__), Path(__file__).with_suffix('.cpp'),
            ROOT / 'scripts/research_20261102_projection_study.py', ROOT / 'scripts/materialize_plan.py']},
        frozen_utc=datetime.now(timezone.utc).isoformat(), test_routes_opened=False,
        phases=results, training_totals_ns=totals,
        original_full_model_capacity=original_capacity, new_full_model_capacity=final_capacity,
        full_model_rows=full_phases,
        protocol='One immutable complete copy of every I8 row. Expert-wise changes of '
            'width, row granularity and chip window; floor/ceil rows for widths1..16, '
            '16 cyclic masks plus wave-template rotations. True normal-read flash, '
            'commands, channel/PCIe input and output costs; GU->host activation->down retained. '
            'Mixed starts include every minimum-work fixed-width local optimum, '
            'plus the fixed wave-template solution; each mixed run gets the same pass budget.',
        limitations='Coordinate local minimum only; no global/MILP optimality claim. '
            'No test, quality inference or hardware measurement. Host activation/reassembly '
            'compute and controller lookup are unmeasured. Variable geometry is a mechanism '
            'to screen, not a novelty claim. Fixed-width controls use overlapping masks '
            'and unconstrained expert counts, unlike the earlier disjoint-groups baseline.')
    write(DEST / (name + '.screen.frozen.json'), output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', default=['granite4tiny', 'granite', 'qwen3'])
    parser.add_argument('--passes', type=int, default=32)
    parser.add_argument('--no-controls', action='store_true')
    args = parser.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    lib = load_library()
    for name in args.models:
        output = run_model(name, lib, args.passes, not args.no_controls)
        print(name, output['training_totals_ns'], flush=True)


if __name__ == '__main__':
    main()
