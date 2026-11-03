#!/usr/bin/env python3
"""Conditional CR-preserving overfetch/drop follow-up; sealed v1 stays intact."""
from collections import Counter
from functools import lru_cache
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import sys

DEST = Path(__file__).resolve().parent
ROOT = DEST.parents[4]
SOURCE = ROOT / 'scripts/research_20261102_cr_ipr_pipeline.py'
SEALED_SHA = 'ad35d08e17b4458cb5a0f54732aafce338d0526f5888fb02fcfdcc507ad99905'
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SEALED_SHA
spec = importlib.util.spec_from_file_location('sealed_shaping_pipeline', SOURCE)
v1 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = v1
spec.loader.exec_module(v1)


def build_shaped_candidate(expert_ids=None, splits=((10, 1),)):
    """Return original useful page stream with explicit, real dummy reads.

    splits is an iterable of (original_wave_index, prefix_page_count), or dict.
    Page identifiers and addresses for dummy reads are actual unselected
    resident expert weights. No vacant physical placeholder is allocated.
    """
    ids = list(range(6)) if expert_ids is None else list(map(int, expert_ids))
    assert len(ids) == 6 and len(set(ids)) == 6 and min(ids) >= 0 and max(ids) < 64
    split_map = dict(splits)
    assert len(split_map) == len(splits)
    assert all(0 <= int(w) < 12 and 1 <= int(k) <= 3 for w, k in split_map.items())
    return shape_existing(v1.build_g4_candidate(ids), splits, ids, 'candidate')


def shape_existing(original, splits, expert_ids, dummy_layout='candidate'):
    """Same shaping capability applied to candidate or fixed packed placement."""
    ids = list(expert_ids)
    split_map = dict(splits)
    unused = [e for e in range(64) if e not in ids]
    pools = {plane: [] for plane in range(4)}
    for e in unused:
        if dummy_layout == 'candidate':
            for plane, index in enumerate((0, 1, 2, 5)):
                pools[plane].append(dict(page_id=f'e{e}.p{index}', owner=f'e{e}', page_index=index,
                                         plane=plane, block=0, wl=2 * e if plane < 2 else e,
                                         bytes=16384, dummy_unselected_expert=e, dummy_layout=dummy_layout))
        else:
            assert dummy_layout == 'packed'
            for index in range(6):
                physical = 6 * e + index
                plane = physical % 4
                pools[plane].append(dict(page_id=f'e{e}.p{index}', owner=f'e{e}', page_index=index,
                                         plane=plane, block=0, wl=physical // 4, bytes=16384,
                                         dummy_unselected_expert=e, dummy_layout=dummy_layout))
    next_dummy = Counter()
    out = []
    for wi, wave in enumerate(original):
        by_plane = {p['plane']: p for p in wave['pages']}
        ordered = [by_plane[p] for p in wave['source_order']]
        chunks = [ordered] if wi not in split_map else [ordered[:split_map[wi]], ordered[split_map[wi]:]]
        for chunk in chunks:
            present = {p['plane'] for p in chunk}
            missing = sorted(set(range(4)) - present)
            reads = list(chunk)
            for plane in missing:
                p = pools[plane][next_dummy[plane]]
                next_dummy[plane] += 1
                reads.append(p)
            out.append(dict(pages=list(chunk), read_pages=reads,
                            source_order=[p['plane'] for p in chunk] + missing,
                            original_wave=wi, shaping_split=wi in split_map))
    return out


def certify(waves, expert_ids=None, expected=None):
    ids = list(range(6)) if expert_ids is None else list(map(int, expert_ids))
    expected = v1.build_g4_candidate(ids) if expected is None else expected
    def useful_stream(ws):
        return [next(p['page_id'] for p in w['pages'] if p['plane'] == plane)
                for w in ws for plane in w['source_order'] if plane in {p['plane'] for p in w['pages']}]
    assert useful_stream(waves) == useful_stream(expected)
    physical, dummy = set(), []
    for w in waves:
        assert {p['plane'] for p in w['read_pages']} == set(range(4))
        useful = {p['page_id'] for p in w['pages']}
        for p in w['read_pages']:
            addr = (p['plane'], p['block'], p['wl'])
            assert addr not in physical
            physical.add(addr)
            assert p['block'] == 0 and 0 <= p['wl'] < 192
            if p['page_id'] not in useful:
                e, plane = p['dummy_unselected_expert'], p['plane']
                assert e in set(range(64)) - set(ids)
                if p['dummy_layout'] == 'candidate':
                    assert p['page_id'] == f"e{e}.p{[0, 1, 2, 5][plane]}"
                    assert p['wl'] == (2 * e if plane < 2 else e)
                else:
                    index = p['page_index']
                    assert p['page_id'] == f'e{e}.p{index}' and (6 * e + index) % 4 == plane
                    assert p['wl'] == (6 * e + index) // 4
                dummy.append(p)
    assert len(dummy) == 4 * (len(waves) - 12)
    return dict(status='PASS', identical_useful_consume_order=True, useful_pages=48,
                physical_read_pages=len(physical), dummy_pages=len(dummy),
                existing_unselected_resident_addresses=True, unique_physical_reads=True,
                all_four_planes_read_every_wave=True, dummy_addresses=dummy,
                no_new_weight_copies=True)


def evaluate(splits, cache_copy_ns=500, expert_ids=None, **overrides):
    waves = build_shaped_candidate(expert_ids, splits)
    certificate = certify(waves, expert_ids)
    result = v1.simulate(waves, cache_copy_ns=cache_copy_ns,
                         unselected_policy='drop_before_ecc_conditional', **overrides)
    assert result['pages'] == result['ecc_pages'] == 48
    assert result['physical_read_pages'] == 4 * len(waves)
    assert result['cold_waves'] == 1 and result['cr_waves'] == len(waves) - 1
    assert result['issue_stall_ns'] == 0
    return dict(splits=[list(x) for x in sorted(dict(splits).items())], waves=waves,
                result=result, certificate=certificate)


def specs(added_waves):
    for indices in itertools.combinations(range(12), added_waves):
        for prefixes in itertools.product((1, 2, 3), repeat=added_waves):
            yield tuple(zip(indices, prefixes))


@lru_cache(maxsize=64)
def _canonical_selection(cache_copy_ns, max_added_waves, override_items):
    best = None
    overrides = dict(override_items)
    for added in range(max_added_waves + 1):
        for splits in specs(added):
            out = evaluate(splits, cache_copy_ns=cache_copy_ns, **overrides)
            r = out['result']
            key = (r['makespan_ns'], added, splits) if r['feasible'] else None
            if key is not None and (best is None or key < best[0]):
                best = (key, splits)
    return best[1] if best is not None else None


def select_shaped_candidate(cache_copy_ns=500, expert_ids=None, max_added_waves=2, **overrides):
    """Select best structural split and instantiate true addresses for a route.

    Timing depends on wave/source structure, not selected expert WL numbers,
    under the same-block skip-WL premise. Canonical search is cached; actual
    route instantiation and physical-address certification are repeated.
    """
    assert max_added_waves in (0, 1, 2)
    selected = _canonical_selection(cache_copy_ns, max_added_waves, tuple(sorted(overrides.items())))
    if selected is None:
        return dict(feasible=False, selected_splits=None, waves=None, result=None,
                    conditional_capability='drop_before_ecc_conditional')
    out = evaluate(selected, cache_copy_ns=cache_copy_ns, expert_ids=expert_ids, **overrides)
    return dict(feasible=out['result']['feasible'], selected_splits=out['splits'], **out)


def summarize(out):
    r = out['result']
    return dict(splits=out['splits'], **v1._summary(r),
                dummy_addresses=out['certificate']['dummy_addresses'])


def _consume(waves):
    return [next(p for p in w['pages'] if p['plane'] == plane)
            for w in waves for plane in w['source_order']
            if plane in {p['plane'] for p in w['pages']}]


def shape_kernel(waves, splits, unselected_page_pool):
    """Preserve arbitrary true useful addresses/order and fill four planes.

    Pool pages must declare unselected=True or dummy_unselected_expert. Caller
    certifies the supplied pool against its physical layout. Missing planes
    use same-block real pool pages; changing a packed same-WL mask to differing
    WLs invokes the granted synchronous IPR capability. This does not remain
    an ordinary same-WL schedule. Empty splits provides fill-only control.
    """
    split_map = dict(splits)
    assert len(split_map) == len(splits)
    assert all(0 <= i < len(waves) for i in split_map)
    used = {(p['plane'], p.get('block', 0), p['wl'])
            for w in waves for p in w.get('read_pages', w['pages'])}
    pools = {}
    for source in unselected_page_pool:
        p = dict(source)
        assert p.get('unselected') is True or 'dummy_unselected_expert' in p
        p.setdefault('block', 0)
        p.setdefault('bytes', 16384)
        if (p['plane'], p['block'], p['wl']) not in used:
            pools.setdefault((p['block'], p['plane']), []).append(p)
    out = []
    for wi, wave in enumerate(waves):
        useful = _consume([wave])
        if wi in split_map:
            k = split_map[wi]
            assert 1 <= k < len(useful)
            chunks = (useful[:k], useful[k:])
        else:
            chunks = (useful,)
        for chunk in chunks:
            block = chunk[0].get('block', 0)
            assert all(p.get('block', 0) == block for p in chunk)
            reads, present = list(chunk), {p['plane'] for p in chunk}
            if wi not in split_map:
                for p in wave.get('read_pages', []):
                    if p['plane'] not in present:
                        assert p.get('block', 0) == block
                        reads.append(p)
                        present.add(p['plane'])
            for plane in sorted(set(range(4)) - present):
                pool = pools.get((block, plane), [])
                while pool and (plane, block, pool[0]['wl']) in used:
                    pool.pop(0)
                if not pool:
                    raise ValueError(f'no unused real dummy page for block{block}/plane{plane}')
                p = pool.pop(0)
                used.add((plane, block, p['wl']))
                reads.append(p)
            missing = sorted(set(range(4)) - {p['plane'] for p in chunk})
            out.append(dict(pages=list(chunk), read_pages=reads,
                            source_order=[p['plane'] for p in chunk] + missing,
                            original_wave=wi, shaping_split=wi in split_map))
    def identity(ws):
        return [(p.get('page_id', p.get('id')), p['plane'], p.get('block', 0), p['wl']) for p in _consume(ws)]
    assert identity(out) == identity(waves)
    assert all({p['plane'] for p in w['read_pages']} == set(range(4)) for w in out)
    return out


def shape_wave(waves, index, prefix, unselected_page_pool):
    return shape_kernel(waves, ((index, prefix),), unselected_page_pool)


if __name__ == '__main__':
    best = select_shaped_candidate(500, max_added_waves=1)
    print(json.dumps(summarize(best) if best['feasible'] else best, indent=2))
