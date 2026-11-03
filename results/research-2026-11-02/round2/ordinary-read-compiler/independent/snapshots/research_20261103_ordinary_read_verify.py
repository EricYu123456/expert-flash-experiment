#!/usr/bin/env python3
"""Regression of the generic backend against sealed, independent replays."""
from dataclasses import replace
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path

import numpy as np

from research_20261103_ordinary_read_backend import (
    ROOT, NativeReplay, ReferenceGraph, Service, ceildiv, lower_rows,
    physical_addresses, sha, split_complete_rows,
)

BASE = ROOT / 'results/research-2026-11-02/round2'
DEST = BASE / 'ordinary-read-compiler/backend-validation'


def qwen_import(record):
    from research_20261102_qwen38_fast_replay import encode
    from research_20261102_qwen38_replay import inventories
    plan = json.loads(Path(record['plan']).read_text())
    assert sha(record['plan']) == record['plan_sha256']
    desc, partial = encode(plan, inventories())
    assert not partial.any()
    offsets = np.zeros((513, 2, 2), dtype=np.int32)
    inputs = np.zeros((513, 2), dtype=np.int32)
    jobs = []
    for e in range(513):
        for p in range(2):
            offsets[e, p, 0] = len(jobs)
            for chip, rows, columns, waves in desc[e, p]:
                if rows:
                    jobs.append([int(chip), int(rows), int(columns),
                                 ceildiv(int(rows) * int(columns), 16384), int(waves)])
                    assert inputs[e, p] in (0, columns); inputs[e, p] = columns
            offsets[e, p, 1] = len(jobs)
    return dict(offsets=offsets, jobs=np.asarray(jobs, dtype=np.int32), input_columns=inputs)


def compare_python(native, plan, routes, experts, shared, service):
    got = native.run(plan, routes, experts, shared, service, trace=True)
    selected = np.r_[routes[0], np.arange(experts, experts + shared)]
    ref = ReferenceGraph.build(plan, selected, service).run()
    assert np.array_equal(got['trace'], ref['trace'])
    for i, route in enumerate(routes):
        ref = ReferenceGraph.build(plan, np.r_[route, np.arange(experts, experts + shared)], service).run()
        assert int(got['cost'][i]) == ref['cost']
    return got


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    native = NativeReplay(DEST / 'build'); s = Service(); records = []
    frozen = BASE / 'geometry-study/granite4tiny.frozen.json'
    doc = json.loads(frozen.read_text())
    reference = BASE / 'full-ffn-integer-reference/frozen-placement-replay'
    routes = np.load(reference / 'routes.npz')['routes'].reshape(32, 40, 6)
    for layer in range(40):
        phases = {p['name']: p for p in doc['phases'] if p['layer'] == layer}
        for case, key in [('frozen_phase_group', 'group_rows'), ('frozen_whole_rows', 'rows')]:
            gu = np.r_[np.asarray(phases['gate_up'][key]), np.full((1, 16), 128)]
            down = np.r_[np.asarray(phases['down'][key]), np.full((1, 16), 96)]
            plan = lower_rows(gu, down, 1536, [512] * 64 + [1024], s)
            got = native.run(plan, routes[:, layer], 64, 1, s, trace=layer == 0)
            old = np.load(reference / f'L{layer}.{case}.npz')['cost']
            assert np.array_equal(got['cost'], old)
            if layer == 0:
                compare_python(native, plan, routes[:3, layer], 64, 1, s)
            records.append(dict(model='granite4tiny', layer=layer, case=case,
                                events=32, cost_max_difference_ns=0))
    print('G4 sealed regression: 2560 costs exact', flush=True)
    campaign = BASE / 'qwen38-placement/native-full48/plans.frozen.json'
    frozen_qwen = json.loads(campaign.read_text())
    external = np.load(BASE / 'qwen38-trace-audit/external-doc-grounded-stride1-short/external_decode.npz')['routes']
    steps = np.unique(np.linspace(0, len(external) - 1, 96, dtype=np.int64))
    for record in frozen_qwen['records']:
        if record['layer'] % 8 or record['case'] not in ('group_g16', 'overlap_best', 'whole_row'):
            continue
        plan = qwen_import(record); layer = record['layer']
        for fringe in (0, 1, 2):
            profile = replace(s, output_fringe=fringe)
            got = native.run(plan, external[steps, layer], 512, 1, profile)
            if fringe:
                oldpath = BASE / 'qwen38-placement/output-fringe' / (record['label'] + f'.mode{fringe}.npz')
            else:
                oldpath = BASE / 'qwen38-placement/native-full48-external' / (record['label'] + '.npz')
                if not oldpath.exists():
                    # Locate the sealed external campaign without assuming a directory alias.
                    matches = list((BASE / 'qwen38-placement').glob('external*/*' + record['label'] + '.npz'))
                    assert len(matches) == 1, (oldpath, matches)
                    oldpath = matches[0]
            old = np.load(oldpath)['cost'][steps]
            assert np.array_equal(got['cost'], old), (record['label'], fringe, int(np.abs(got['cost'] - old).max()))
            compare_python(native, plan, external[steps[:1], layer], 512, 1, profile)
            records.append(dict(model='qwen38', layer=layer, case=record['case'], fringe=fringe,
                                events=len(steps), cost_max_difference_ns=0))
    print('Q38 sealed regression: 5184 costs exact', flush=True)
    # Non-Qwen hidden/input lengths, shared/nonshared, different channel/plane
    # counts, page fringes, zero-duration work, GF/DF, and multiple jobs/chip.
    rng = np.random.default_rng(20261103); synthetic = 0; max_jobs = 0
    for chips, planes, fringe, down_first, shared in itertools.product((4, 8), (1, 2), (0, 1, 2), (0, 1), (0, 1)):
        profile = Service(chips=chips, channels=chips // 2, planes=planes,
                          page_bytes=128, block_wordlines=7, input_bytes=2,
                          output_bytes=8, ordinary_read_ns=0 if synthetic % 7 == 0 else 37,
                          command_ns=5, host_submit_ns=13,
                          pcie_bytes_per_second=3_000_000_000,
                          channel_bytes_per_second=1_000_000_000,
                          activation_ns=11, combine_ns=19,
                          output_fringe=fringe, down_first=down_first)
        e = 8; count = e + shared
        gu = np.zeros((count, chips), dtype=np.int64)
        down = gu.copy()
        # Whole rows can exceed one block, so ragged lowering is exercised.
        for x in range(count):
            gu[x] = rng.multinomial(79, np.full(chips, 1 / chips))
            down[x] = rng.multinomial(41, np.full(chips, 1 / chips))
        columns = [64] * e + ([96] if shared else [])
        plan = lower_rows(gu, down, 96, columns, profile)
        physical = physical_addresses(plan, profile)
        assert physical['row_coverage_and_address_nonoverlap']
        r = np.asarray([rng.choice(e, 3, replace=False) for _ in range(3)], dtype=np.int32)
        compare_python(native, plan, r, e, shared, profile)
        max_jobs = max(max_jobs, len(plan['jobs'])); synthetic += 1
    # Exact tiny partitions independently enumerate row chunks: verify the
    # lowerer's minimum-command then minimum-wave contract, not just coverage.
    partitions = 0
    for b, c, depth, rows in itertools.product(range(1, 5), range(1, 7), range(1, 5), range(1, 19)):
        maxr = depth * b // c
        if not maxr:
            continue
        profile = Service(chips=2, channels=1, planes=1, page_bytes=b, block_wordlines=depth)
        pieces = split_complete_rows(rows, c, profile)
        commands = ceildiv(rows, maxr)
        dp = [[None] * (rows + 1) for _ in range(commands + 1)]; dp[0][0] = 0
        for n in range(1, commands + 1):
            for total in range(1, rows + 1):
                costs = [dp[n-1][total-r] + ceildiv(r*c, b)
                         for r in range(1, min(maxr, total)+1)
                         if dp[n-1][total-r] is not None]
                if costs:
                    dp[n][total] = min(costs)
        assert len(pieces) == commands
        assert sum(ceildiv(r*c, b) for r in pieces) == dp[commands][rows]
        partitions += 1
    report = dict(status='PASS generic ordinary-read backend regression',
                  created_utc=datetime.now(timezone.utc).isoformat(),
                  sources_sha256={str(p): sha(p) for p in (
                      Path(__file__), Path(__file__).with_name('research_20261103_ordinary_read_backend.py'),
                      Path(__file__).with_name('research_20261103_ordinary_read_native.cpp'),
                      Path(__file__).with_name('research_20261102_whole_row_compiler.py'))},
                  build=native.provenance, records=records,
                  sealed_costs=sum(r['events'] for r in records),
                  synthetic_graphs=synthetic, synthetic_costs=3*synthetic,
                  synthetic_max_job_count=max_jobs, independently_enumerated_row_split_cases=partitions,
                  limits='Software contract and lowering equivalence only. No native PE arithmetic, calibration, numeric ABI, quantization metadata or device execution.')
    (DEST / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('records', 'sources_sha256', 'build')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
