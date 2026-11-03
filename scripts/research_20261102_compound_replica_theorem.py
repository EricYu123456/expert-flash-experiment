#!/usr/bin/env python3
"""Construct and exhaustively certify a restricted, uncoded placement bound.

This is a combinatorial certificate, not a novelty or device-performance claim.
The atoms are twelve fixed 128-row INT8 shards per expert.  Only the four
omit-one-group execution configurations are used by the construction.  No
routing traces or model weights are read by this program.
"""
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import itertools
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'results/research-2026-11-02/round2/compound-replica'
E, K, GROUPS, LANES = 64, 6, 4, 4
ROWS, COLS, TILE_ROWS = 1536, 512, 128
TILE_BYTES = TILE_ROWS * COLS


def construct():
    default = [e % GROUPS for e in range(E)]
    forbidden = [-1] * E
    for color in range(GROUPS):
        eligible = [e for e in range(E)
                    if default[e] != color and forbidden[e] == -1]
        assert len(eligible) >= K - 1
        for e in eligible[:K - 1]:
            forbidden[e] = color
    tiles = []
    for e in range(E):
        row = 0
        for group in range(GROUPS):
            if group == default[e]:
                continue
            for lane in range(LANES):
                locations = [group * LANES + lane]
                if group != forbidden[e]:
                    locations.append(default[e] * LANES + lane)
                tiles.append(dict(expert=e, row_start=row, rows=TILE_ROWS,
                                  primary_group=group, lane=lane,
                                  locations=locations))
                row += TILE_ROWS
        assert row == ROWS
    return default, forbidden, tiles


def assignment(experts, forbidden):
    """Tiny integral matching, with balanced ONFI channel-pair demand."""
    experts = sorted(experts, key=lambda e: (forbidden[e] == -1, e))
    # Channels connect c and c+8: group pairs are {0,2} and {1,3}.
    for doubled in itertools.product((0, 2), (1, 3)):
        target = tuple(2 if g in doubled else 1 for g in range(GROUPS))

        @lru_cache(None)
        def search(i, remaining):
            if i == len(experts):
                return () if not any(remaining) else None
            for color in range(GROUPS):
                if remaining[color] == 0 or forbidden[experts[i]] == color:
                    continue
                rem = list(remaining)
                rem[color] -= 1
                suffix = search(i + 1, tuple(rem))
                if suffix is not None:
                    return (color,) + suffix
            return None

        colors = search(0, target)
        if colors is not None:
            return dict(zip(experts, colors)), target
    raise AssertionError(('No balanced assignment', experts))


def physical_query(experts, omitted, default, tiles):
    jobs = []
    coverage = Counter()
    for tile in tiles:
        e = tile['expert']
        if e not in omitted:
            continue
        group = tile['primary_group']
        chip = tile['locations'][0]
        if omitted[e] != default[e] and group == omitted[e]:
            chip = default[e] * LANES + tile['lane']
        assert chip in tile['locations']
        assert chip // LANES != omitted[e]
        jobs.append((e, tile['row_start'], chip))
        coverage[e] += tile['rows']
    assert len(jobs) == K * 12
    assert coverage == Counter({e: ROWS for e in experts})
    assert len({(e, r) for e, r, _ in jobs}) == K * 12
    assert len({(e, c) for e, _, c in jobs}) == K * 12
    counts = Counter(c for _, _, c in jobs)
    waves = [counts[c] for c in range(16)]
    assert max(waves) == 5
    channel_jobs = [waves[c] + waves[c + 8] for c in range(8)]
    assert channel_jobs == [9] * 8
    components = dict(flash_ns=5 * 28000,
                      control_ns=len(jobs) * 250, host_ns=2000,
                      input_ns=max(math.ceil(K * COLS / 8), 9 * COLS // 2),
                      output_ns=max(math.ceil(K * ROWS * 4 / 8),
                                    9 * TILE_ROWS * 4 // 2))
    return jobs, waves, components


def compositions(total, parts):
    if parts == 1:
        yield (total,)
    else:
        for n in range(total + 1):
            for rest in compositions(total - n, parts - 1):
                yield (n,) + rest


def main():
    default, forbidden, tiles = construct()
    groups = [[e for e in range(E) if forbidden[e] == f]
              for f in (-1, 0, 1, 2, 3)]
    assert [len(g) for g in groups] == [44, 5, 5, 5, 5]
    primary = len(tiles)
    copies = sum(len(t['locations']) - 1 for t in tiles)
    lower_bound = primary - 16 * 5
    assert primary == 768 and copies == lower_bound == 688
    unreplicated = Counter(t['locations'][0] for t in tiles
                           if len(t['locations']) == 1)
    assert [unreplicated[c] for c in range(16)] == [5] * 16
    per_chip = Counter(c for t in tiles for c in t['locations'])
    witnesses, weighted_queries = [], 0
    for counts in compositions(K, 5):
        if any(n > len(g) for n, g in zip(counts, groups)):
            continue
        multiplicity = math.prod(math.comb(len(g), n)
                                 for n, g in zip(counts, groups))
        query = [e for n, g in zip(counts, groups) for e in g[:n]]
        omitted, target = assignment(query, forbidden)
        jobs, waves, components = physical_query(query, omitted, default, tiles)
        assert sum(components.values()) == 166912
        witnesses.append(dict(availability_type_counts=counts,
                              route_multiplicity=multiplicity,
                              example_experts=query, omitted=omitted,
                              omitted_group_counts=target,
                              chip_waves=waves, components_ns=components))
        weighted_queries += multiplicity
    assert len(witnesses) == 206
    assert weighted_queries == math.comb(E, K) == 74974368
    # Model-independent exact integer GEMV check, including relocated rows.
    import numpy as np
    rng = np.random.default_rng(20261102)
    x = rng.integers(-128, 128, size=COLS, dtype=np.int8).astype(np.int64)
    matrix = rng.integers(-128, 128, size=(ROWS, COLS), dtype=np.int8).astype(np.int64)
    reference = matrix @ x
    numerical_cases = 0
    for e in range(E):
        for omit in range(GROUPS):
            if forbidden[e] == omit:
                continue
            reconstructed = np.empty(ROWS, dtype=np.int64)
            for t in tiles:
                if t['expert'] != e:
                    continue
                chip = t['locations'][0]
                if omit != default[e] and t['primary_group'] == omit:
                    chip = default[e] * LANES + t['lane']
                assert chip in t['locations']
                a, b = t['row_start'], t['row_start'] + TILE_ROWS
                reconstructed[a:b] = matrix[a:b] @ x
            assert np.array_equal(reference, reconstructed)
            numerical_cases += 1
    DEST.mkdir(parents=True, exist_ok=True)
    data = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        hardware='16 chips, four existing same-address planes/chip, 16KiB pages; '
                 'one dense 128x512 INT8 GEMV per selected expert/chip; host row scatter.',
        scope='Canonical twelve 64KiB full-row atoms/expert, 64 experts, any six '
              'distinct experts. Lower bound concerns physical extra atom copies '
              'for worst-case <=5 NAND read waves/chip, not global optimality '
              'over re-sharding, coding, other operators or all model geometries.',
        numerical_scope='INT64 oracle on unchanged INT8 rows, exact row reconstruction; '
                        'this does not measure or certify a physical AiF accumulator.',
        model_weights_or_routing_opened=False,
        experts=E, active_experts=K, total_rows=ROWS, columns=COLS,
        tile_rows=TILE_ROWS, tile_bytes=TILE_BYTES,
        primary_tiles=primary, extra_copies=copies,
        necessary_extra_copies=lower_bound,
        total_storage_ratio=(primary + copies) / primary,
        stored_physical_pages=4 * (primary + copies),
        per_chip_stored_tiles=[per_chip[c] for c in range(16)],
        per_chip_unreplicated_tiles=[unreplicated[c] for c in range(16)],
        default_omitted_group=default, forbidden_omitted_group=forbidden,
        type_compositions_checked=len(witnesses),
        all_distinct_expert_queries_covered=weighted_queries,
        exact_synthetic_integer_reconstruction_cases=numerical_cases,
        guarantee='For any route, assign omission counts 2/2/1/1 with the two doubled '
                  'colors on opposite ONFI pairs. All 72 original atoms execute '
                  'once, chip peak is 5 waves, every channel carries 9 input vectors.',
        lower_bound_proof='A chip cannot hold six mandatory unreplicated atoms: '
                          'their owner experts number at most six, so a six-expert '
                          'route containing those owners forces >=6 waves. Hence '
                          'at most 5 unreplicated atoms/chip, at most 80 total. '
                          'Each of the other 768-80 atoms needs at least one extra copy.',
        construction_proof='Twenty distinct special experts each forbid one '
                           'nondefault omitted group, five per color. All others '
                           'allow all four groups. Any six-expert subset has a '
                           'color-capacity matching for a paired 2/2/1/1 target: '
                           'each color forbids <=5 experts; if 5 forbid a color, '
                           'give that color target 1. Every union of >=2 color '
                           'neighborhoods contains all six experts.',
        components_scope='166912ns is the exact arithmetic for this restricted '
                         'one-atom/command four-profile construction at 250ns '
                         'command and 2us host phase, both uncalibrated assumptions. '
                         'Coalescing multiple atom commands or alternative profiles '
                         'can alter command/input costs. No device latency claim.',
        novelty='Not established: replicated declustering, combinatorial batch '
                'codes, batch array codes and singleton storage bounds are direct '
                'prior-art checks. A constructive numerical result alone is not a paper breakthrough.',
        tiles=tiles, witnesses=witnesses)
    (DEST / 'theorem.json').write_text(json.dumps(data, indent=2) + '\n')
    print(json.dumps({k: data[k] for k in (
        'primary_tiles', 'extra_copies', 'total_storage_ratio',
        'type_compositions_checked', 'all_distinct_expert_queries_covered',
        'exact_synthetic_integer_reconstruction_cases')}))


if __name__ == '__main__':
    main()
