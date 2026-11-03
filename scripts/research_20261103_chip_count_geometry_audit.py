#!/usr/bin/env python3
"""Independent inverse-capacity DP audit of the chip-count geometry frontier.

No producer/solver import, native replay, routes, fitting, or heldout data.
For j job slots and a total w waves, maximize complete-row capacity. This is
the inverse of the producer's minimum-work covering formulation.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/research-2026-11-02/round2/chip-count-regime-screen'
INPUT = BASE / 'stage-a/geometry.json'
OUT = BASE / 'independent-geometry'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def freeze():
    OUT.mkdir(exist_ok=False)
    dump(OUT / 'protocol.frozen.json', dict(
        utc=datetime.now(timezone.utc).isoformat(), source_sha256=sha(__file__),
        geometry_path=str(INPUT), geometry_sha256=sha(INPUT),
        formulation='capacity[j,w] = max over k in [0,depth] of capacity[j-1,w-k] + floor(k*65536/C); inactive slots have k=0.',
        proof='Each complete-row job taking k waves has at most floor(k*B/C) rows. Conversely choose those capacities, then remove whole rows until total R; reads can only shrink. The least w with capacity[M,w]>=R is therefore the exact minimum aggregate waves with at most M single-block jobs of depth<=d.',
        scope='Only row geometry. No command-cost/native-latency optimality, physical allocation, full-model inference, or hardware ABI claim. No producer import or heldout data.',
    ))


def inverse_dp(rows, columns, slots, depth):
    assert 0 < depth <= 192
    limit = slots * depth
    previous = [-1] * (limit + 1)
    previous[0] = 0
    capacities = [k * 65536 // columns for k in range(depth + 1)]
    for j in range(1, slots + 1):
        current = [-1] * (limit + 1)
        for total in range(j * depth + 1):
            for k in range(min(depth, total) + 1):
                value = previous[total-k]
                if value >= 0:
                    current[total] = max(current[total], value + capacities[k])
        previous = current
    optimum = next(w for w, capacity in enumerate(previous) if capacity >= rows)
    assert all(capacity < rows for capacity in previous[:optimum])
    return optimum, previous


def run():
    protocol = json.loads((OUT / 'protocol.frozen.json').read_text())
    assert sha(__file__) == protocol['source_sha256']
    assert sha(INPUT) == protocol['geometry_sha256']
    assert not (OUT / 'audit.json').exists()
    records = []
    frontier_checks = menu_checks = 0
    for case in json.loads(INPUT.read_text()):
        for name, g in case['shapes'].items():
            rows, columns, slots = (g[k] for k in ('R', 'C', 'chips'))
            for profile in g['menu']:
                rr = profile['rows']
                waves = [(r * columns + 65535) // 65536 for r in rr]
                pages = [(r * columns + 16383) // 16384 for r in rr]
                assert len(rr) == slots and sum(rr) == rows and min(rr) >= 0
                assert sum(waves) == profile['waves']
                assert sum(pages) == profile['pages']
                assert max(waves) == profile['depth'] <= 192
                assert sum(r > 0 for r in rr) == profile['positive_jobs']
                menu_checks += 1
            for item in g['frontier']:
                depth = item['depth']
                optimum, cap = inverse_dp(rows, columns, slots, depth)
                witness_waves = [(r * columns + 65535) // 65536 for r in item['rows']]
                assert len(item['rows']) == slots and sum(item['rows']) == rows
                assert min(item['rows']) >= 0 and max(witness_waves) <= depth
                assert sum(witness_waves) == item['waves'] == optimum
                records.append(dict(model=case['model'], phase=name, chips=slots,
                                    rows=rows, columns=columns, depth=depth,
                                    exact_minimum_waves=optimum,
                                    maximum_row_capacity_by_total_waves=cap))
                frontier_checks += 1
    audit = dict(status='PASS independent inverse-capacity integer DP',
                 protocol_sha256=sha(OUT / 'protocol.frozen.json'),
                 source_sha256=sha(__file__), geometry_sha256=sha(INPUT),
                 frontier_checks=frontier_checks, menu_profiles_checked=menu_checks,
                 records=records, limits=protocol['scope'])
    dump(OUT / 'audit.json', audit)
    dump(OUT / 'completion.json', dict(utc=datetime.now(timezone.utc).isoformat(),
         source_sha256=sha(__file__), inputs_unchanged=sha(INPUT) == protocol['geometry_sha256'],
         files={p.name: sha(p) for p in sorted(OUT.iterdir()) if p.is_file()}))
    print(json.dumps({k: v for k, v in audit.items() if k != 'records'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['freeze', 'run'])
    args = parser.parse_args()
    freeze() if args.action == 'freeze' else run()
