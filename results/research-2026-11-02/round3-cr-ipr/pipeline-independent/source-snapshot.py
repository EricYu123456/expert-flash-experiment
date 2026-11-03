#!/usr/bin/env python3
"""Causal finite-PB, synchronous IPR, shared-ECC timing envelope.

This is a capability sensitivity, not a device/RTL implementation. Public API:
``build_g4_candidate``, ``build_ordinary_balanced_shared``,
``build_ordinary_packed``, ``build_compensator_ipr``, ``simulate``,
``validate_stream``. All inputs and outputs are JSON-compatible dictionaries;
``PipelineConfig`` is also accepted by ``simulate``. Time is integer ns.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'results/research-2026-11-02/round3-cr-ipr/pipeline'
PAGE_BYTES = 16384
ROW_BYTES = 1536
MACRO_ROWS = 32
COLD_NS = 28000
CR_NS = 9700
RECYCLE_NS = 6040
ECC_BYTES_PER_SECOND = 6400000000


@dataclass(frozen=True)
class PipelineConfig:
    page_bytes: int = PAGE_BYTES
    row_bytes: int = ROW_BYTES
    cold_ns: int = COLD_NS
    cr_ns: int = CR_NS
    recycle_ns: int = RECYCLE_NS
    ecc_bytes_per_second: int = ECC_BYTES_PER_SECOND
    cache_copy_ns: int = 0
    weight_slots: int = 2
    overwrite_at: str = 'evaluation_start'
    copy_overlaps_recycle: bool = True
    availability_phase_ns: int = 0
    source_policy: str = 'permutation'
    issue_policy: str = 'strict_no_interval'
    array_mode: str = 'cr'
    copy_scope: str = 'per_plane'
    validate_row_stream: bool = True
    unselected_policy: str = 'reject'
    cold_discharge_ns: int = 0


def _page(owner: str, index: int, plane: int, wl: int, block: int = 0) -> dict:
    return dict(page_id=f'{owner}.p{index}', owner=owner, page_index=index,
                plane=plane, wl=wl, block=block, bytes=PAGE_BYTES)


def _wave(pages: Sequence[dict], order: Sequence[int] | None = None) -> dict:
    planes = [p['plane'] for p in pages]
    assert len(set(planes)) == len(planes) and len(pages) <= 4
    return dict(pages=list(pages), source_order=list(order if order is not None else planes))


def build_g4_candidate(expert_ids: Sequence[int] | None = None, repeats: int = 2) -> list[dict]:
    """One physical copy, 64 resident experts, 3-page complete-row macros.

    A = 012, B = 013, D = 323, C = 232. Each three-expert group
    consumes A1,D,A2,B1,B2,C,A3,B3. Four-page cuts are synchronous IPR
    waves, never independently advancing planes. Expert IDs may be any 0..63.
    """
    ids = list(range(3 * repeats)) if expert_ids is None else list(map(int, expert_ids))
    assert len(ids) > 0 and len(ids) % 3 == 0 and len(set(ids)) == len(ids)
    assert min(ids) >= 0 and max(ids) < 64
    stream = []
    shared_plane_counts = Counter()
    shared_index = 0
    for group in range(len(ids) // 3):
        a, b, c = ids[3 * group:3 * group + 3]
        def macro(e: int, second: bool) -> list[dict]:
            planes = (0, 1, 3) if second else (0, 1, 2)
            start = 3 if second else 0
            return [_page(f'e{e}', start + j, p,
                          2 * e + int(second) if p < 2 else e)
                    for j, p in enumerate(planes)]
        def shared(planes: Sequence[int]) -> list[dict]:
            nonlocal shared_index
            out = []
            for p in planes:
                out.append(_page('s', shared_index, p, 64 + shared_plane_counts[p]))
                shared_plane_counts[p] += 1
                shared_index += 1
            return out
        d, cc = shared((3, 2, 3)), shared((2, 3, 2))
        stream.extend(macro(a, False) + d + macro(b, False) + macro(a, True)
                      + macro(b, True) + cc + macro(c, False) + macro(c, True))
    waves = [_wave(stream[i:i + 4]) for i in range(0, len(stream), 4)]
    assert all(len(w['pages']) == 4 for w in waves)
    return waves


def build_ordinary_balanced_shared(expert_ids: Sequence[int] | None = None,
                                   ipr: bool = False) -> list[dict]:
    """Six 4+2 page matrices plus three full shared waves: 15 waves.

    same-WL version reserves two WLs per expert. The balanced IPR version
    compacts the singly-used p2/p3 planes; demand and logical stream stay equal.
    """
    ids = list(range(6)) if expert_ids is None else list(map(int, expert_ids))
    out = []
    for e in ids:
        pages = [_page(f'e{e}', j, j % 4,
                       e if ipr and j % 4 >= 2 else 2 * e + j // 4)
                 for j in range(6)]
        out.extend((_wave(pages[:4]), _wave(pages[4:])))
    shared = [_page('s', j, j % 4, (64 if ipr and j % 4 >= 2 else 128) + j // 4)
              for j in range(12)]
    out.extend(_wave(shared[j:j + 4]) for j in range(0, 12, 4))
    return out


def build_balanced_ipr(expert_ids: Sequence[int] | None = None) -> list[dict]:
    return build_ordinary_balanced_shared(expert_ids, ipr=True)


def build_ordinary_packed(expert_ids: Sequence[int] | None = None) -> list[dict]:
    """One fixed ordinary same-WL 6-page/expert packing of all 64 experts.

    Requested pages share a WL only when resident neighbors are co-selected.
    Unrequested pages are not charged/read. A partial wave remains partial;
    there is no fictitious cross-WL gather. The physical order is row-safe.
    """
    ids = list(range(6)) if expert_ids is None else list(map(int, expert_ids))
    by_wl: dict[int, list] = defaultdict(list)
    for e in ids:
        for j in range(6):
            address = 6 * e + j
            by_wl[address // 4].append(_page(f'e{e}', j, address % 4, address // 4))
    out = [_wave(sorted(pages, key=lambda p: p['plane']))
           for _, pages in sorted(by_wl.items())]
    out.extend(_wave([_page('s', j + k, k, 96 + j // 4) for k in range(4)])
               for j in range(0, 12, 4))
    return out


def build_compensator_ipr(expert_ids: Sequence[int] | None = None) -> list[dict]:
    """12 full waves by filling every expert's sparse wave with shared pages.

    Deliberately strong geometry control, but its 2-page shared interruptions
    fail the complete-row stream contract. No partial-accumulator/FIFO rescue.
    """
    ids = list(range(6)) if expert_ids is None else list(map(int, expert_ids))
    assert len(ids) == 6
    out = []
    for i, e in enumerate(ids):
        out.append(_wave([_page(f'e{e}', j, j, 2 * e) for j in range(4)]))
        out.append(_wave([_page(f'e{e}', 4, 0, 2 * e + 1),
                          _page(f'e{e}', 5, 1, 2 * e + 1),
                          _page('s', 2 * i, 2, 64 + i),
                          _page('s', 2 * i + 1, 3, 64 + i)]))
    return out


def with_keep_hot(waves: Sequence[dict]) -> list[dict]:
    """Read all four physical planes, including unrequested same-WL pages.

    This requires an explicit simulate(unselected_policy=...) decision. Merely
    reading the unrequested pages does not silently exempt them from ECC.
    """
    out = []
    for w in waves:
        assert len({(p['block'], p['wl']) for p in w['pages']}) == 1
        out.append(dict(w, mask_active_planes=[0, 1, 2, 3], source_order=[0, 1, 2, 3]))
    return out


def normalize_waves(waves: Sequence[dict], config: PipelineConfig) -> list[dict]:
    out, seen, physical = [], set(), set()
    for wi, w in enumerate(waves):
        pages, read_pages = [], []
        requested_ids = {p.get('page_id', p.get('id', f"{p['owner']}.p{p['page_index']}")) for p in w['pages']}
        raw = list(w.get('read_pages', w['pages']))
        for requested in w['pages']:
            if requested['plane'] not in {p['plane'] for p in raw}:
                raw.append(requested)
        mask = w.get('mask_active_planes')
        if mask is not None:
            mask = [p for p in range(4) if mask & (1 << p)] if isinstance(mask, int) else list(mask)
            assert set(p['plane'] for p in raw) <= set(mask)
            if set(mask) != set(p['plane'] for p in raw):
                assert len({(p.get('block', 0), p.get('wl', wi)) for p in raw}) == 1, 'supply explicit read_pages for non-same-WL dummy reads'
                first = raw[0]
                for plane in sorted(set(mask) - set(p['plane'] for p in raw)):
                    raw.append(dict(page_id=f'drop.w{wi}.p{plane}', owner='_drop', page_index=wi * 4 + plane,
                                    plane=plane, wl=first.get('wl', wi), block=first.get('block', 0),
                                    bytes=config.page_bytes, useful=False))
        for p in raw:
            x = dict(p)
            if 'page_id' not in x:
                x['page_id'] = x.get('id', f"{x['owner']}.p{x['page_index']}")
            x.setdefault('bytes', config.page_bytes)
            x.setdefault('block', 0)
            x.setdefault('wl', wi)
            x['useful'] = x['page_id'] in requested_ids
            assert x['page_id'] not in seen, f"duplicate logical page {x['page_id']}"
            seen.add(x['page_id'])
            address = (x['plane'], x['block'], x['wl'])
            assert address not in physical, f'duplicate physical LSB address {address}'
            physical.add(address)
            assert 0 <= x['plane'] < 4 and 0 <= x['wl'] < 192
            assert x['bytes'] > 0 and x['bytes'] <= config.page_bytes
            read_pages.append(x)
            if x['useful']:
                pages.append(x)
        assert pages and len(pages) <= 4
        assert {p['page_id'] for p in pages} == requested_ids, 'physical reads must contain all requested logical pages'
        by_plane = {p['plane']: p for p in read_pages}
        assert len(by_plane) == len(read_pages), 'synchronous IPR has at most one page per plane'
        assert len({p['block'] for p in read_pages}) == 1, 'a wave must stay within one block'
        if len(read_pages) > len(pages):
            assert config.unselected_policy != 'reject', 'select explicit unselected-page drop or decode capability'
        order = list(w.get('source_order', sorted(by_plane)))
        order += [p for p in sorted(by_plane) if p not in order]
        assert set(order) == set(by_plane) and len(order) == len(by_plane)
        actual = sorted(by_plane) if config.source_policy == 'fixed' else order
        out.append(dict(wave=wi, pages=pages, read_pages=read_pages, source_order=order,
                        actual_order=actual, block=pages[0]['block']))
    return out


def validate_stream(pages: Iterable[dict], row_bytes: int = ROW_BYTES) -> dict:
    """Owner switches occur only at full output-row boundaries.

    This checks page order and dot-product row continuity; it does not certify
    PE/command support for changing output-owner tags at a macro boundary.
    """
    pages = list(pages)
    errors, counts, byte_counts = [], Counter(), Counter()
    previous = None
    for i, p in enumerate(pages):
        owner = p['owner']
        if previous is not None and owner != previous and byte_counts[previous] % row_bytes:
            errors.append(dict(kind='owner_switch_inside_row', position=i,
                               owner=previous, bytes=byte_counts[previous], next_owner=owner))
        if p['page_index'] != counts[owner]:
            errors.append(dict(kind='per_owner_page_order', position=i, owner=owner,
                               expected=counts[owner], actual=p['page_index']))
        counts[owner] += 1
        byte_counts[owner] += p['bytes']
        previous = owner
    for owner, nbytes in byte_counts.items():
        if nbytes % row_bytes:
            errors.append(dict(kind='final_incomplete_row', owner=owner, bytes=nbytes))
    return dict(ok=not errors, errors=errors, owner_pages=dict(counts),
                owner_bytes=dict(byte_counts),
                owner_rows={k: v // row_bytes for k, v in byte_counts.items()})


def _ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _avoid_blackouts(start: int, duration: int, intervals: Sequence[tuple[int, int]]) -> int:
    if not duration:
        return start
    for a, b in intervals:
        if start + duration <= a:
            return start
        if start < b and start + duration > a:
            start = b
    return start


def simulate(waves: Sequence[dict], config: PipelineConfig | dict | None = None,
             **overrides: Any) -> dict:
    """Schedule finite read/cache slots and a chip-shared serial ECC.

    Copies are started as soon as their read page and destination cache are
    available. ECC follows the actual wave-source order, with zero implicit
    reorder storage. There are no weight FIFOs and no read/cache source mux.
    A one-slot sensitivity reads directly from its sole latch and is optimistic
    about availability of that direct ECC path. Positive-copy two-slot models
    always require read->cache before ECC. Scheduled copies and ECC retain
    their source slots until completion (no hidden incremental freeing).

    Strict failures return makespan_ns=None; their full hypothetical trace is
    retained for diagnosis only. Page completion timestamps are useful to the
    caller only when timing_feasible and functional_ok both hold.
    """
    cfg = config if isinstance(config, PipelineConfig) else PipelineConfig(**(config or {}))
    if overrides:
        cfg = replace(cfg, **overrides)
    assert cfg.weight_slots in (1, 2)
    assert cfg.overwrite_at in ('sense_start', 'evaluation_start')
    assert cfg.source_policy in ('permutation', 'fixed')
    assert cfg.issue_policy in ('strict_no_interval', 'cold_reset', 'paced_retention_conditional')
    assert cfg.array_mode in ('cr', 'ordinary')
    assert cfg.copy_scope in ('per_plane', 'chip_serial_conditional')
    assert cfg.unselected_policy in ('reject', 'drop_before_ecc_conditional', 'decode_and_discard_conditional')
    assert 0 <= cfg.cold_discharge_ns < cfg.cold_ns
    assert min(cfg.cache_copy_ns, cfg.recycle_ns) >= 0
    assert cfg.cold_ns > 0 and cfg.cr_ns > cfg.recycle_ns >= 0
    assert cfg.ecc_bytes_per_second > 0 and cfg.row_bytes > 0
    ws = normalize_waves(waves, cfg)
    if not ws:
        raise ValueError('at least one wave is required')
    ordered = [next(p for p in w['pages'] if p['plane'] == plane)
               for w in ws for plane in w['actual_order']
               if plane in {p['plane'] for p in w['pages']}]
    stream = validate_stream(ordered, cfg.row_bytes)
    events, arrays, completions, violations, read_intervals, cache_intervals = [], [], [], [], [], []
    read_release, cache_release = [0] * 4, [0] * 4
    read_page = [None] * 4
    ecc_free = copy_free = previous_end = 0
    previous_block, previous_planes = None, set()
    total_stall = 0
    nominal_starts, nominal_ends = [], []
    # Only the fixed no-interval schedule needs future recycle blackouts.
    # Mutating schedules instead delay the next issue until all copies finish
    # when copy/recycle overlap is disabled.
    t, previous_nominal_planes = 0, set()
    for wi, w in enumerate(ws):
        active = {p['plane'] for p in w['read_pages']}
        use_cr = cfg.array_mode == 'cr' and wi > 0 and w['block'] == ws[wi - 1]['block'] and active <= previous_nominal_planes
        if wi and not use_cr:
            t += cfg.cold_discharge_ns
        nominal_starts.append(t)
        t += cfg.cr_ns if use_cr else cfg.cold_ns - cfg.cold_discharge_ns
        nominal_ends.append(t)
        previous_nominal_planes = active
    blackouts = [(a, a + cfg.recycle_ns) for wi, a in enumerate(nominal_starts)
                 if wi and cfg.array_mode == 'cr' and ws[wi]['block'] == ws[wi - 1]['block']
                 and {p['plane'] for p in ws[wi]['read_pages']} <= {p['plane'] for p in ws[wi - 1]['read_pages']}]
    for wi, w in enumerate(ws):
        active = {p['plane'] for p in w['read_pages']}
        idle_cold_planes = sorted(active - previous_planes) if wi else sorted(active)
        can_cr = cfg.array_mode == 'cr' and wi > 0 and previous_block == w['block'] and not idle_cold_planes
        start = previous_end
        selected = sorted(active)
        earliest_clear = max((read_release[p] for p in selected), default=0)
        reuse_offset = cfg.recycle_ns if can_cr and cfg.overwrite_at == 'evaluation_start' else 0
        if not cfg.copy_overlaps_recycle:
            reuse_offset = 0
        required_start = max(start, earliest_clear - reuse_offset)
        read_kind = 'cr' if can_cr else 'cold'
        cold_reason = None if can_cr else ('first_wave' if wi == 0 else 'ordinary_mode' if cfg.array_mode == 'ordinary' else 'block_change' if previous_block != w['block'] else 'plane_idle')
        if cfg.array_mode == 'ordinary':
            # Ordinary reads may wait without requiring retained CR voltages.
            start = max(start, earliest_clear)
            read_kind = 'cold'
        elif cfg.issue_policy == 'cold_reset' and required_start > start:
            start = max(start, earliest_clear)
            read_kind = 'cold'
            cold_reason = 'finite_latch_backpressure'
        elif cfg.issue_policy == 'paced_retention_conditional':
            start = required_start
        # A block change also requires that its selected read latches be free.
        elif not can_cr and wi:
            start = max(start, earliest_clear)
        if read_kind == 'cold' and wi:
            events.append(dict(stage='array_discharge', wave=wi - 1, start_ns=previous_end,
                               end_ns=previous_end + cfg.cold_discharge_ns,
                               duration_ns=cfg.cold_discharge_ns))
            start = max(start, previous_end + cfg.cold_discharge_ns)
        total_stall += start - previous_end
        duration = cfg.cr_ns if read_kind == 'cr' else cfg.cold_ns - cfg.cold_discharge_ns
        end = start + duration
        overwrite = start + (cfg.recycle_ns if read_kind == 'cr' and cfg.overwrite_at == 'evaluation_start' else 0)
        for p in selected:
            if read_release[p] > overwrite:
                violations.append(dict(kind='read_latch_overwrite', wave=wi, plane=p,
                                       old_page=read_page[p], release_ns=read_release[p],
                                       overwrite_ns=overwrite, excess_ns=read_release[p] - overwrite))
        ready = end + cfg.availability_phase_ns
        assert ready >= start, 'availability phase cannot precede the read operation'
        record = dict(wave=wi, kind=read_kind, start_ns=start, end_ns=end,
                      duration_ns=duration, ready_ns=ready, overwrite_ns=overwrite,
                      block=w['block'], pages=len(w['read_pages']), useful_pages=len(w['pages']),
                      active_planes=selected, previously_idle_planes=idle_cold_planes,
                      cold_reason=cold_reason, idle_before_ns=start - previous_end)
        arrays.append(record)
        events.append(dict(stage='array_' + read_kind, **record))
        if read_kind == 'cr':
            events.append(dict(stage='wl_recycle', wave=wi, start_ns=start,
                               end_ns=start + cfg.recycle_ns, duration_ns=cfg.recycle_ns))
        for p in w['read_pages']:
            if not p['useful'] and cfg.unselected_policy == 'drop_before_ecc_conditional':
                plane = p['plane']
                read_release[plane] = ready
                read_page[plane] = p['page_id']
                events.append(dict(stage='drop_before_ecc', wave=wi, plane=plane,
                                   page_id=p['page_id'], start_ns=ready, end_ns=ready, duration_ns=0,
                                   bytes=p['bytes'], useful=False))
                read_intervals.append(dict(wave=wi, plane=plane, page_id=p['page_id'],
                                           start_ns=overwrite, ready_ns=ready, end_ns=ready, useful=False))
        for plane in w['actual_order']:
            p = next(p for p in w['read_pages'] if p['plane'] == plane)
            if not p['useful'] and cfg.unselected_policy == 'drop_before_ecc_conditional':
                continue
            service = _ceildiv(p['bytes'] * 10**9, cfg.ecc_bytes_per_second)
            common = {k: p[k] for k in ('page_id', 'owner', 'page_index', 'plane', 'bytes', 'block', 'wl', 'useful')}
            common['wave'] = wi
            if cfg.weight_slots == 2:
                copy_start = max(ready, cache_release[plane], copy_free if cfg.copy_scope != 'per_plane' else 0)
                if not cfg.copy_overlaps_recycle and cfg.issue_policy == 'strict_no_interval':
                    copy_start = _avoid_blackouts(copy_start, cfg.cache_copy_ns, blackouts)
                copy_end = copy_start + cfg.cache_copy_ns
                if cfg.copy_scope != 'per_plane':
                    copy_free = copy_end
                events.append(dict(stage='cache_copy', start_ns=copy_start, end_ns=copy_end,
                                   duration_ns=cfg.cache_copy_ns, **common))
                ecc_start = max(ecc_free, copy_end)
                ecc_end = ecc_start + service
                read_release[plane] = copy_end
                cache_release[plane] = ecc_end
                cache_intervals.append(dict(start_ns=copy_start, end_ns=ecc_end, **common))
            else:
                ecc_start = max(ecc_free, ready)
                ecc_end = ecc_start + service
                read_release[plane] = ecc_end
            read_page[plane] = p['page_id']
            read_intervals.append(dict(start_ns=overwrite, ready_ns=ready, end_ns=read_release[plane], **common))
            events.append(dict(stage='ecc', start_ns=ecc_start, end_ns=ecc_end,
                               duration_ns=service, source='cache' if cfg.weight_slots == 2 else 'read', **common))
            completions.append(dict(start_ns=ecc_start, end_ns=ecc_end, **common))
            ecc_free = ecc_end
        previous_end, previous_block, previous_planes = end, w['block'], active
    events.append(dict(stage='array_discharge', wave=len(ws) - 1, start_ns=previous_end,
                       end_ns=previous_end + cfg.cold_discharge_ns, duration_ns=cfg.cold_discharge_ns))
    previous_end += cfg.cold_discharge_ns
    # A common availability lag reserves the read latch even if the array has
    # already started the next read. Include that pending old page in hazards.
    # It is represented by the same [ready,release] lifecycle and deadlines.
    ecc_intervals = sorted((e['start_ns'], e['end_ns']) for e in completions)
    assert all(b <= c for (a, b), (c, d) in zip(ecc_intervals, ecc_intervals[1:]))
    for intervals in (cache_intervals,):
        by_plane = defaultdict(list)
        for x in intervals:
            by_plane[x['plane']].append((x['start_ns'], x['end_ns']))
        assert all(all(b <= c for (a, b), (c, d) in zip(sorted(rr), sorted(rr)[1:]))
                   for rr in by_plane.values())
    if cfg.weight_slots == 2:
        copies = [x for x in events if x['stage'] == 'cache_copy']
        for x in copies:
            assert x['start_ns'] >= arrays[x['wave']]['ready_ns']
        copy_groups = defaultdict(list)
        for x in copies:
            copy_groups[x['plane'] if cfg.copy_scope == 'per_plane' else -1].append(x)
        assert all(all(a['end_ns'] <= b['start_ns'] for a, b in zip(
                       sorted(rr, key=lambda x: x['start_ns']), sorted(rr, key=lambda x: x['start_ns'])[1:]))
                   for rr in copy_groups.values())
        if not cfg.copy_overlaps_recycle:
            for x in copies:
                for rec in (e for e in events if e['stage'] == 'wl_recycle'):
                    if max(x['start_ns'], rec['start_ns']) < min(x['end_ns'], rec['end_ns']):
                        violations.append(dict(kind='copy_recycle_overlap', page_id=x['page_id'],
                                               copy_start_ns=x['start_ns'], copy_end_ns=x['end_ns'],
                                               recycle_wave=rec['wave']))
    total_bytes = sum(p['bytes'] for p in ordered)
    physical_bytes = sum(p['bytes'] for w in ws for p in w['read_pages'])
    ecc_busy = sum(x['end_ns'] - x['start_ns'] for x in completions)
    array_busy = sum(x['duration_ns'] for x in arrays) + cfg.cold_discharge_ns * sum(x['kind'] == 'cold' for x in arrays)
    first_ready = arrays[0]['ready_ns']
    release_lb_end = 0
    for x in completions:
        release_lb_end = max(release_lb_end, arrays[x['wave']]['ready_ns']) + x['end_ns'] - x['start_ns']
    lower = max(array_busy, release_lb_end)
    hypothetical = max(previous_end, ecc_free)
    assert hypothetical >= lower
    end_events = defaultdict(int)
    for x in completions:
        end_events[x['end_ns']] += 1
    outstanding = peak_outstanding = 0
    occupancy_events = []
    for x in arrays:
        serviced = x['pages'] if cfg.unselected_policy == 'decode_and_discard_conditional' else x['useful_pages']
        occupancy_events.append((x['ready_ns'], 1, serviced))
    for x in completions:
        occupancy_events.append((x['end_ns'], 0, -1))
    for _, _, delta in sorted(occupancy_events):
        outstanding += delta
        peak_outstanding = max(peak_outstanding, outstanding)
    assert outstanding == 0
    row_completions, owner_cumulative = [], Counter()
    for x in completions:
        if not x['useful']:
            continue
        old_rows = owner_cumulative[x['owner']] // cfg.row_bytes
        owner_cumulative[x['owner']] += x['bytes']
        new_rows = owner_cumulative[x['owner']] // cfg.row_bytes
        if new_rows > old_rows:
            row_completions.append(dict(owner=x['owner'], row_start=old_rows,
                                        row_end=new_rows, end_ns=x['end_ns'], page_id=x['page_id']))
    functional = stream['ok'] or not cfg.validate_row_stream
    timing = not violations
    max_slots = []
    for plane in range(4):
        occupancy = []
        for x in read_intervals + cache_intervals:
            if x['plane'] == plane and x['end_ns'] > x['start_ns']:
                occupancy.extend(((x['start_ns'], 1), (x['end_ns'], -1)))
        live = peak = 0
        for _, delta in sorted(occupancy):
            live += delta
            peak = max(peak, live)
        assert live == 0
        max_slots.append(peak)
    if timing:
        assert max(max_slots) <= cfg.weight_slots, max_slots
    conditional = ['skip-WL CR and synchronized IPR accepted as user premise',
                   'arbitrary plane source order' if cfg.source_policy == 'permutation' else 'fixed plane drain 0123',
                   'PB read-to-cache copy path and selected copy timing' if cfg.weight_slots == 2 else 'direct read-latch ECC path (one-slot sensitivity)',
                   'per-plane copy concurrency' if cfg.copy_scope == 'per_plane' else 'chip-serialized copy sensitivity']
    if cfg.issue_policy == 'paced_retention_conditional':
        conditional.append('CR voltage retention across idle intervals; not established existing capability')
    if cfg.availability_phase_ns:
        conditional.append('common page-availability offset from nominal sense completion')
    if cfg.unselected_policy != 'reject':
        conditional.append(cfg.unselected_policy)
    if cfg.cold_discharge_ns:
        conditional.append('unknown cold-read discharge phase split; each run keeps total28us startup+final discharge')
    return dict(config=asdict(cfg), waves=len(ws), pages=len(ordered), bytes=total_bytes,
                physical_read_pages=sum(len(w['read_pages']) for w in ws), physical_read_bytes=physical_bytes,
                ecc_pages=len(completions), ecc_bytes=sum(x['bytes'] for x in completions),
                functional_ok=functional, timing_feasible=timing, feasible=functional and timing,
                makespan_ns=hypothetical if functional and timing else None,
                hypothetical_makespan_ns=hypothetical, array_end_ns=previous_end,
                array_busy_ns=array_busy, ecc_busy_ns=ecc_busy, issue_stall_ns=total_stall,
                cold_waves=sum(x['kind'] == 'cold' for x in arrays), cr_waves=sum(x['kind'] == 'cr' for x in arrays),
                first_page_ready_ns=first_ready, first_ecc_page_complete_ns=completions[0]['end_ns'],
                page_completions=[x for x in completions if x['useful']], ecc_page_completions=completions, row_completions=row_completions,
                owner_completion_ns={k: max(x['end_ns'] for x in completions if x['owner'] == k) for k in owner_cumulative},
                read_latch_releases=[dict(plane=p, page_id=read_page[p], release_ns=read_release[p]) for p in range(4)],
                read_latch_intervals=read_intervals, cache_latch_intervals=cache_intervals,
                peak_unconsumed_pages=peak_outstanding, events=sorted(events, key=lambda x: (x['start_ns'], x['end_ns'], x['stage'])),
                max_weight_slots_by_plane=max_slots,
                array_waves=arrays, violations=violations, stream_validation=stream,
                resource_bounds=dict(array_service_ns=array_busy, ecc_service_ns=ecc_busy,
                                     first_release_plus_ecc_ns=first_ready + ecc_busy,
                                     unlimited_weight_buffer_release_ecc_lb_ns=release_lb_end,
                                     combined_lower_bound_ns=lower,
                                     interpretation='Necessary resource/release bounds only; finite PB scheduling is checked separately.'),
                conservation=dict(requested_pages=len(ordered), array_read_pages=sum(x['pages'] for x in arrays),
                                  unselected_read_pages=sum(x['pages'] - x['useful_pages'] for x in arrays),
                                  ecc_completed_pages=len(completions), useful_ecc_completed_pages=sum(x['useful'] for x in completions), array_bytes=physical_bytes,
                                  ecc_bytes=sum(x['bytes'] for x in completions),
                                  cache_copies=len(completions) if cfg.weight_slots == 2 else 0,
                                  each_page_once=len({x['page_id'] for x in completions if x['useful']}) == len(ordered),
                                  resource_intervals_audited=True,
                                  physical_trace_valid=timing),
                conditional_capabilities=conditional)


def effective_bandwidth_model(waves: Sequence[dict], bandwidth: int = ECC_BYTES_PER_SECOND) -> dict:
    """A distinct abstract service model: no 9.7us array or finite-PB claims."""
    nbytes = sum(p.get('bytes', PAGE_BYTES) for w in waves for p in w['pages'])
    return dict(model='effective_bandwidth_only', bytes=nbytes, bytes_per_second=bandwidth,
                abstract_service_ns=_ceildiv(nbytes * 10**9, bandwidth),
                first_page_phase_and_startup='not specified',
                finite_PB_feasibility='not implied',
                warning='Do not combine this effective rate with a second 9.7us/wave charge or treat it as the explicit array+ECC simulator.')


def _summary(result: dict) -> dict:
    keys = ('config', 'waves', 'pages', 'bytes', 'physical_read_pages', 'physical_read_bytes', 'ecc_pages', 'ecc_bytes', 'functional_ok', 'timing_feasible', 'feasible',
            'makespan_ns', 'hypothetical_makespan_ns', 'array_end_ns', 'array_busy_ns', 'ecc_busy_ns',
            'issue_stall_ns', 'cold_waves', 'cr_waves', 'first_page_ready_ns',
            'first_ecc_page_complete_ns', 'peak_unconsumed_pages', 'resource_bounds', 'conservation')
    d = {k: result[k] for k in keys}
    d['violation_count'] = len(result['violations'])
    d['first_violation'] = result['violations'][0] if result['violations'] else None
    d['stream_errors'] = result['stream_validation']['errors'][:4]
    return d


def run_self_tests() -> list[dict]:
    tests = []
    def check(name: str, pred: bool, details: Any = None) -> None:
        assert pred, (name, details)
        tests.append(dict(name=name, status='PASS', details=details))
    tiny = [_wave([_page('x', 0, 0, 0)])]
    t = simulate(tiny, validate_row_stream=False, cache_copy_ns=100)
    check('single page hand calculation', t['makespan_ns'] == 28000 + 100 + 2560, t['makespan_ns'])
    full = [_wave([_page('x', j, j, 0) for j in range(4)])]
    t = simulate(full, validate_row_stream=False, cache_copy_ns=500)
    check('four parallel copies and serial ECC hand calculation', t['makespan_ns'] == 38740, t['makespan_ns'])
    two = full + [_wave([_page('x', j + 4, j, 1) for j in range(4)])]
    bad = simulate(two, validate_row_stream=False, overwrite_at='sense_start', cache_copy_ns=100)
    good = simulate(two, validate_row_stream=False, overwrite_at='evaluation_start', cache_copy_ns=100)
    check('positive copy cannot precede immediate sense overwrite', not bad['timing_feasible'], bad['violations'][0])
    check('recycling gives cache-copy window', good['timing_feasible'], good['makespan_ns'])
    no_overlap = simulate(two, validate_row_stream=False, cache_copy_ns=100, copy_overlaps_recycle=False)
    check('no recycle overlap denies positive copy window', not no_overlap['timing_feasible'], no_overlap['violations'][0])
    one = simulate(two, validate_row_stream=False, weight_slots=1)
    check('one slot cannot drain four pages within recycle window', not one['timing_feasible'], one['violations'][0])
    fast = simulate(two, validate_row_stream=False, overwrite_at='sense_start', cache_copy_ns=0,
                    ecc_bytes_per_second=10**12)
    check('zero copy and fast ECC have no fabricated hazard', fast['timing_feasible'], fast['makespan_ns'])
    slow_copy = simulate(two, validate_row_stream=False, cache_copy_ns=6041, ecc_bytes_per_second=10**12)
    check('copy exceeding recycle window fails even with fast ECC', not slow_copy['timing_feasible'], slow_copy['violations'][0])
    candidate = build_g4_candidate()
    check('candidate source-order stencil', [w['source_order'] for w in candidate[:6]] ==
          [[0, 1, 2, 3], [2, 3, 0, 1], [2, 0, 1, 3], [0, 1, 3, 2], [3, 2, 0, 1], [2, 0, 1, 3]])
    cand = simulate(candidate)
    check('candidate stream and demand conservation', cand['functional_ok'] and cand['pages'] == 48 and
          cand['bytes'] == 786432 and cand['conservation']['each_page_once'], _summary(cand))
    fixed = simulate(candidate, source_policy='fixed')
    check('fixed drain detects physical candidate stream failure', not fixed['functional_ok'], fixed['stream_validation']['errors'][:2])
    edge_good = simulate(candidate, cache_copy_ns=320)
    edge_bad = simulate(candidate, cache_copy_ns=321)
    check('exact 320ns copy timing boundary', edge_good['feasible'] and not edge_bad['timing_feasible'],
          dict(pass_ns=320, fail_ns=321, violation=edge_bad['violations'][0]))
    comp = simulate(build_compensator_ipr())
    check('shared compensator interruption is not a legal row stream', not comp['functional_ok'], comp['stream_validation']['errors'][:2])
    ordinary = build_ordinary_balanced_shared()
    check('ordinary balanced has fifteen waves', len(ordinary) == 15)
    check('packed same-WL counts actual selected resident WLs',
          len(build_ordinary_packed(range(6))) == 12 and len(build_ordinary_packed([0, 3, 6, 9, 12, 15])) == 15)
    masks = [full[0], _wave([_page('x', 4, 0, 1)]),
             _wave([_page('x', 5 + j, j, 2) for j in range(4)])]
    r = simulate(masks, validate_row_stream=False)
    check('inactive plane cannot silently retain CR for the next wave',
          [a['kind'] for a in r['array_waves']] == ['cold', 'cr', 'cold'] and
          r['array_waves'][2]['previously_idle_planes'] == [1, 2, 3])
    hot = with_keep_hot(masks)
    dropped = simulate(hot, validate_row_stream=False, unselected_policy='drop_before_ecc_conditional')
    decoded = simulate(hot, validate_row_stream=False, unselected_policy='decode_and_discard_conditional')
    check('keep-hot drop reads all physical pages but only requested ECC pages',
          dropped['physical_read_pages'] == 12 and dropped['ecc_pages'] == 9 and
          dropped['cold_waves'] == 1 and dropped['bytes'] == 9 * PAGE_BYTES)
    check('keep-hot decode explicitly pays for all physical ECC pages',
          decoded['physical_read_pages'] == decoded['ecc_pages'] == 12 and len(decoded['page_completions']) == 9)
    drop_rejected = False
    try:
        simulate(hot, validate_row_stream=False)
    except AssertionError:
        drop_rejected = True
    check('keep-hot dummy drop never implicitly granted', drop_rejected)
    split = simulate(two, validate_row_stream=False, cold_discharge_ns=1000)
    unsplit = simulate(two, validate_row_stream=False)
    check('cold discharge split moves first availability and preserves run array total',
          split['array_end_ns'] == unsplit['array_end_ns'] and split['first_page_ready_ns'] == 27000 and
          split['makespan_ns'] == unsplit['makespan_ns'] - 1000)
    reset = simulate(two, validate_row_stream=False, overwrite_at='sense_start', cache_copy_ns=100, issue_policy='cold_reset')
    paced = simulate(two, validate_row_stream=False, overwrite_at='sense_start', cache_copy_ns=100, issue_policy='paced_retention_conditional')
    check('cold reset explicitly charges another cold read', reset['feasible'] and reset['cold_waves'] == 2 and reset['issue_stall_ns'] == 100)
    check('paced CR names retained-bias assumption', paced['feasible'] and paced['cr_waves'] == 1 and paced['issue_stall_ns'] == 100)
    # Arbitrary valid short streams, no row validation: monotone ECC/bounds and
    # no silent rescue when ECC bandwidth is reduced by 64x.
    for pages_per_wave in (1, 2, 3, 4):
        waves = [_wave([_page('x', wi * pages_per_wave + j, j, wi) for j in range(pages_per_wave)]) for wi in range(5)]
        for policy in ('strict_no_interval', 'cold_reset', 'paced_retention_conditional'):
            r = simulate(waves, validate_row_stream=False, cache_copy_ns=500,
                         ecc_bytes_per_second=10**8, issue_policy=policy)
            check(f'finite backpressure {pages_per_wave} pages {policy}',
                  (not r['timing_feasible']) if policy == 'strict_no_interval' else r['timing_feasible'],
                  dict(stall_ns=r['issue_stall_ns'], violations=len(r['violations'])))
    # Single-wave no copy-overwrite hazard, any availability lag just shifts ECC.
    delayed = simulate(full, validate_row_stream=False, availability_phase_ns=1000)
    plain = simulate(full, validate_row_stream=False)
    check('availability phase is a real dependency', delayed['makespan_ns'] == plain['makespan_ns'] + 1000)
    duplicate_rejected = False
    try:
        simulate(full + full, validate_row_stream=False)
    except AssertionError:
        duplicate_rejected = True
    check('duplicate logical/physical reads rejected', duplicate_rejected)
    return tests


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    tests = run_self_tests()
    candidate = build_g4_candidate()
    layouts = dict(candidate12=candidate,
                   ordinary_sameWL_balanced15=build_ordinary_balanced_shared(),
                   ordinary_IPR_balanced15=build_balanced_ipr(),
                   ordinary_packed_adjacent12=build_ordinary_packed(range(6)),
                   ordinary_packed_spaced15=build_ordinary_packed([0, 3, 6, 9, 12, 15]),
                   IPR_compensator12_functional_negative=build_compensator_ipr())
    copy_values = [0, 100, 320, 321, 500, 1000, 2000, 3000, 4000, 5000, 6040, 6500, 10000]
    sweep = []
    for layout, waves in layouts.items():
        for overwrite in ('sense_start', 'evaluation_start'):
            for copy in copy_values:
                r = simulate(waves, cache_copy_ns=copy, overwrite_at=overwrite)
                sweep.append(dict(layout=layout, **_summary(r)))
    variants = []
    variant_configs = dict(
        candidate_permuted_two_slots=PipelineConfig(),
        candidate_fixed_two_slots=PipelineConfig(source_policy='fixed'),
        candidate_permuted_one_slot=PipelineConfig(weight_slots=1),
        candidate_no_recycle_copy=PipelineConfig(cache_copy_ns=500, copy_overlaps_recycle=False),
        candidate_paced_conditional=PipelineConfig(cache_copy_ns=1000, issue_policy='paced_retention_conditional'),
        candidate_cold_reset=PipelineConfig(cache_copy_ns=1000, issue_policy='cold_reset'),
        candidate_sense_start_cold_reset=PipelineConfig(cache_copy_ns=500, overwrite_at='sense_start', issue_policy='cold_reset'),
        candidate_serial_copy_conditional=PipelineConfig(cache_copy_ns=500, copy_scope='chip_serial_conditional'),
    )
    for name, cfg in variant_configs.items():
        variants.append(dict(name=name, **_summary(simulate(candidate, cfg))))
    strong_keep_hot = []
    for name in ('ordinary_sameWL_balanced15', 'ordinary_packed_adjacent12', 'ordinary_packed_spaced15'):
        for policy in ('drop_before_ecc_conditional', 'decode_and_discard_conditional'):
            for copy in (0, 100, 320, 321, 500, 1000):
                r = simulate(with_keep_hot(layouts[name]), unselected_policy=policy, cache_copy_ns=copy)
                strong_keep_hot.append(dict(layout=name, **_summary(r)))
    # Capability matched explicit ordinary-vs-CR controls. Ordinary array mode
    # can pace because it makes no retained-voltage claim.
    mode_controls = []
    for layout, waves in layouts.items():
        for mode in ('ordinary', 'cr'):
            r = simulate(waves, array_mode=mode, cache_copy_ns=500)
            mode_controls.append(dict(layout=layout, array_mode=mode, **_summary(r)))
    phase_sweep = []
    for phase in (-1000, 0, 500, 1000, 2000, 4000):
        for copy in (0, 500, 1000):
            r = simulate(candidate, cache_copy_ns=copy, availability_phase_ns=phase)
            phase_sweep.append(dict(phase_ns=phase, copy_ns=copy, **_summary(r)))
    discharge_sweep = []
    for discharge in (0, 500, 1000, 2000, 3000):
        for copy in (0, 320, 500):
            r = simulate(candidate, cache_copy_ns=copy, cold_discharge_ns=discharge)
            discharge_sweep.append(dict(discharge_ns=discharge, **_summary(r)))
    # Full resident capacity audit uses all 64 experts plus the single shared
    # twelve-page set; not six demand pages multiplied into physical replicas.
    resident = defaultdict(set)
    for e in range(64):
        for second in (False, True):
            for p in ((0, 1, 3) if second else (0, 1, 2)):
                resident[p].add(2 * e + int(second) if p < 2 else e)
    for w in candidate:
        for p in w['pages']:
            if p['owner'] == 's':
                resident[p['plane']].add(p['wl'])
    counts = [len(resident[p]) for p in range(4)]
    assert counts == [128, 128, 70, 70] and max(max(v) for v in resident.values()) < 192
    ledger = dict(
        accepted_user_premises=['skip-WL CR within one block', 'synchronized IPR selecting one WL per active plane'],
        paper_facts=[dict(claim='CR requires successive same-block reads without any interval', source='paper/aif.txt:529'),
                     dict(claim='cold single-sensing read 28us, CR 9.7us, recycling6.04us', source='paper/aif.txt:538-550 and924'),
                     dict(claim='four page latches/plane: cache + one inference read + two input latches', source='paper/aif.txt:733-743'),
                     dict(claim='on-chip ECC throughput6.4GB/s; output FIFO is after computation', source='paper/aif.txt:815-821')],
        finite_storage=['only one read latch and one cache latch per plane hold weights',
                        'two other latches remain input vector storage',
                        'no ECC-input FIFO, no arbitrary read/cache source mux, no partial-row output accumulator contexts'],
        unresolved_capabilities=['read latch overwritten at sense start or only at evaluation start',
                                 'read-to-cache copy latency and per-plane copy concurrency',
                                 'copy overlapping WL recycling',
                                 'programmable plane-source order and complete-row owner tagging',
                                 'page availability phase relative to reported9.7us',
                                 'array pacing retaining CR bias is conditional, never granted by default'],
        per_plane_CR_continuity=['a selected plane is CR-eligible only if it was read in the immediately preceding synchronous wave in the same block',
                                 'if any selected plane was idle, the whole synchronized wave uses cold28us; no asynchronous plane finish or AIPR',
                                 'keep-hot physical reads include unselected pages; before-ECC drop and after-ECC discard are separate conditional controls',
                                 'physical read pages/bytes, useful pages/bytes and ECC pages/bytes are conserved separately'],
        conservative_service=['cache latch remains occupied until whole-page ECC completion',
                              'read latch remains occupied until whole-page copy completion',
                              'all pages of a synchronous IPR wave become available together',
                              'first read is cold28us, final ECC drain is included',
                              'one shared ECC serializes all plane pages at2.56us/page'],
        cold_phase_accounting=['default first availability28us/final discharge0 is a conservative service convention; measured28us is tPRE+tEVAL+tDISCH',
                               'cold_discharge_ns=d changes each cold start sense duration to28us-d and charges discharge d at each run end; a no-stall chain retains total28us+(n-1)*9.7us',
                               'numeric tPRE and tDISCH split is not disclosed; d values and common availability_phase_ns offsets are conditional sensitivities',
                               'copy/ECC independence from discharge is assumed in the optional phase split'],
        ordinary_controls=['balanced matrix-aligned4+2 pages/expert +12sharedpages =15waves',
                           'fixed packed one-copy placement can be12waves on adjacent co-selection,15on spaced selection',
                           'IPR balanced15 keeps useful pages/row stream identical',
                           '12wave compensator fills empty planes but fails complete-row shared stream without extra state'],
        model_boundary=['explicit model charges array and ECC on separate overlapping resources',
                        'effective6.4GB/s-only model has unspecified startup and does not prove finite-PB feasibility',
                        'no AIPR: every plane in a wave shares one array start and end',
                        'cold_reset waits and restarts a28us operation; no undocumented hold-bias rescue',
                        'feasible means feasible within named timing/control assumptions, not measured existing-hardware support'])
    report = dict(status='PASS', model='finite-PB synchronous-IPR array+shared-ECC',
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  geometry=dict(qbits=4, H=1536, paired_GU_row_bytes=ROW_BYTES,
                                page_bytes=PAGE_BYTES, macro_pages=3, macro_rows=MACRO_ROWS,
                                resident_experts=64, selected_experts=6, shared_pages=12,
                                resident_plane_WLs=counts, block_WLs=192),
                  self_tests=tests, layouts=layouts, sweep=sweep, variants=variants,
                  availability_phase_sweep=phase_sweep, mode_controls=mode_controls,
                  cold_discharge_phase_sweep=discharge_sweep, strong_keep_hot_controls=strong_keep_hot,
                  effective_bandwidth_only=[dict(layout=k, **effective_bandwidth_model(v)) for k, v in layouts.items()],
                  assumption_ledger=ledger)
    (DEST / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    (DEST / 'assumption_ledger.json').write_text(json.dumps(ledger, indent=2) + '\n')
    traces = dict(candidate12_copy0_eval=simulate(candidate),
                  candidate12_copy500_eval=simulate(candidate, cache_copy_ns=500),
                  candidate12_copy500_sense_FAIL=simulate(candidate, cache_copy_ns=500, overwrite_at='sense_start'),
                  ordinary15_copy500_eval=simulate(build_ordinary_balanced_shared(), cache_copy_ns=500),
                  candidate12_copy1000_cold_reset=simulate(candidate, cache_copy_ns=1000, issue_policy='cold_reset'))
    traces['ordinary15_keep_hot_drop_copy500'] = simulate(with_keep_hot(build_ordinary_balanced_shared()), unselected_policy='drop_before_ecc_conditional', cache_copy_ns=500)
    traces['ordinary15_keep_hot_decode_copy500_FAIL'] = simulate(with_keep_hot(build_ordinary_balanced_shared()), unselected_policy='decode_and_discard_conditional', cache_copy_ns=500)
    for name, r in traces.items():
        (DEST / (name + '.events.json')).write_text(json.dumps(r, indent=2) + '\n')
    small = simulate(candidate[:2], validate_row_stream=False, cache_copy_ns=500)
    (DEST / 'small_two_wave_trace.json').write_text(json.dumps(small, indent=2) + '\n')
    with (DEST / 'small_two_wave_trace.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['stage', 'wave', 'plane', 'page_id', 'start_ns', 'end_ns', 'duration_ns'], extrasaction='ignore')
        writer.writeheader()
        writer.writerows(small['events'])
    feasible = [r for r in sweep if r['layout'] == 'candidate12' and r['feasible']]
    caps = {overwrite: max((r['config']['cache_copy_ns'] for r in feasible if r['config']['overwrite_at'] == overwrite), default=None)
            for overwrite in ('sense_start', 'evaluation_start')}
    base = simulate(candidate)
    ordinary = simulate(build_ordinary_balanced_shared())
    lines = [
        '# Finite PB / CR / IPR pipeline audit', '',
        'This is a timing envelope under explicitly named capabilities, not a device feasibility certificate.', '',
        f"All {len(tests)} meaningful checks passed. The G4 resident construction occupies {counts} WLs per plane in one192-WL block.", '',
        'The explicit model has a28us cold first wave,9.7us subsequent synchronized CR waves, a6.04us recycling phase, two weight latches per plane, and one6.4GB/s ECC. All four page latches are accounted for; two remain input latches. Four-page ECC service is10.24us. Pages are copied read→cache and held through full-page ECC completion; no weight FIFO or cache/read mux is supplied.', '',
        'A plane must have been read in the immediately preceding wave to remain CR-eligible. If any selected plane was idle, the synchronous wave is cold28us. The default ordinary balanced15 case consequently has7cold/8CR waves. Keep-hot controls instead read all4planes, then either conditionally drop unselected pages before ECC or explicitly decode them before discarding.', '',
        f"For the sampled copy sweep, candidate maximum feasible cache-copy latency is {caps['sense_start']}ns at sense-start overwrite and {caps['evaluation_start']}ns at evaluation-start overwrite (copy/recycle overlap enabled). A null cap means no sampled point was feasible.", '',
        '| Layout / default conditional capabilities | Waves | Functional | Finite timing | Makespan us |',
        '|---|---:|---|---|---:|',
    ]
    for name, waves in layouts.items():
        r = simulate(waves)
        lines.append(f"| {name} | {r['waves']} | {r['functional_ok']} | {r['timing_feasible']} | {r['makespan_ns'] / 1000 if r['makespan_ns'] is not None else 'FAIL'} |")
    lines += ['',
              'The defaults assume programmable plane-source order, evaluation-start overwrite, zero copy latency and copy/recycle overlap. These are sensitivity settings, not established AiF controls. Readout with fixed0123 order fails the candidate complete-row stream check. A one-weight-slot control fails finite timing. Shared compensator packing is also functionally invalid because its two-page shared runs interrupt a row.', '',
              'A physical ordinary packed layout can use12waves for adjacent co-selected experts and15waves for spaced ones. The15wave balanced matrix-aligned control is therefore not the only ordinary layout. No route-independent latency advantage over every ordinary packing is claimed.', '',
              'Strict no-interval failures remain failures. The cold-reset control waits for latch release and pays another28us read. A paced9.7us continuation is separately labeled as conditional CR bias retention. Copies that cannot overlap recycling are evaluated separately.', '',
              'At copy500ns, keep-hot balanced15 with before-ECC drop is feasible at175.620us, physically reads60pages, and decodes48pages. Decoding all60pages instead fails the finite-PB schedule. These capabilities are explicit controls, not granted to the candidate alone.', '',
              'The exact default candidate cache-copy boundary is320ns: the minimum latch-deadline slack is640−2*copy ns. Both320ns pass and321ns fail are tested. Ordinary packed adjacent12 has the same boundary and same150.880us zero-copy cost.', '',
              'Cold first availability28us and final discharge0 is a conservative service accounting convention. The paper gives28us=tPRE+tEVAL+tDISCH and does not disclose the numeric phase split. Optional cold_discharge_ns sensitivities move d from initial availability to final run discharge while retaining the same total array service. No measured first-page phase is claimed.', '',
              'The independent byte/service and release-time bounds are necessary bounds only. The event schedule verifies latch deadlines, serial ECC, per-plane copy serialization, exact page coverage, startup and drain. Hypothetical times for failed schedules are diagnostic and must not be used as feasible costs.', '',
              'The effective-bandwidth-only6.4GB/s cost is a separate model with unspecified startup. It is never added to a second9.7us/wave charge and does not certify this pipeline.', '',
              'The public Python API returns page_completions, full-row completion ranges, per-owner completion timestamps, read_latch_releases, array_waves, finite-storage lifecycle intervals and event traces. Caller-supplied waves are accepted. See assumption_ledger.json and small_two_wave_trace.csv.']
    (DEST / 'report.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(status='PASS', self_tests=len(tests), candidate=_summary(base),
                          ordinary=_summary(ordinary), sampled_copy_caps_ns=caps), indent=2))


if __name__ == '__main__':
    main()
