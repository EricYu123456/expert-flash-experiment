#!/usr/bin/env python3
"""Exact uniform-route occupancy sweep of fixed plane bins, without relocation."""
import argparse
import csv
import itertools
import json
from math import comb
from pathlib import Path

import plane_wave_model as plane
from run_corpus_traces import ROOT, sha256, write_json


def occupancy_distribution(experts, active, lanes):
    capacity = experts // lanes
    denominator = comb(experts, active)
    result = []
    for prefix in itertools.product(range(min(capacity, active) + 1), repeat=lanes - 1):
        last = active - sum(prefix)
        if not 0 <= last <= capacity:
            continue
        counts = prefix + (last,)
        numerator = 1
        for count in counts:
            numerator *= comb(capacity, count)
        result.append((counts, numerator / denominator))
    assert abs(sum(p for _, p in result) - 1) < 1e-12
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results/plane-wave-v1/synthetic')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    experts = 32
    records = []
    for kib in (16, 24, 32, 40, 48, 64, 80, 96, 128, 256, 1024):
        nbytes = kib * 1024
        for active in (1, 2, 4, 8, 16):
            for policy, width in plane.POLICY_WIDTH.items():
                lanes = 4 // width
                cost = plane.stream_cost(nbytes, width)
                row = dict(experts=experts, selected_experts=active, shard_kib=kib,
                           policy=policy, width=width, mean_ideal_ns=0.,
                           mean_conservative_ns=0., mean_read_waves=0.,
                           mean_extra_contexts=0., probability_all_selected_fit_one_wave=0.)
                for counts, probability in occupancy_distribution(experts, active, lanes):
                    peak = max(counts)
                    waves = peak * cost['stream_waves']
                    ecc = active * cost['physical_pages'] * 2560
                    row['mean_ideal_ns'] += probability * max(peak * cost['nand_ns'], ecc)
                    row['mean_conservative_ns'] += probability * (waves * plane.COLD_NS + ecc)
                    row['mean_read_waves'] += probability * waves
                    row['mean_extra_contexts'] += probability * (sum(c > 0 for c in counts) - 1)
                    row['probability_all_selected_fit_one_wave'] += probability * (waves == 1)
                row['useful_slot_fill'] = active * nbytes / (row['mean_read_waves'] * 4 * plane.PAGE)
                row['occupied_slot_fill'] = active * cost['physical_pages'] / (row['mean_read_waves'] * 4)
                row['logical_cold_starts'] = active * cost['cold_reads']
                records.append(row)
    for row in records:
        baseline = next(r for r in records if r['shard_kib'] == row['shard_kib'] and
                        r['selected_experts'] == row['selected_experts'] and r['policy'] == 'baseline')
        row['ideal_reduction_vs_baseline'] = 1 - row['mean_ideal_ns'] / baseline['mean_ideal_ns']
        row['conservative_reduction_vs_baseline'] = 1 - row['mean_conservative_ns'] / baseline['mean_conservative_ns']
    write_json(args.output_dir / 'results.json', records)
    with (args.output_dir / 'results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    write_json(args.output_dir / 'manifest.json', dict(experts=experts, cases=len(records),
        routing='Exact multivariate hypergeometric occupancy for uniform K-subsets of E=32; no random sampling.',
        placement='Fixed expert e -> lane e % (4/width), equal stored expert count per lane.',
        timing='Per-chip flash/ECC only; excludes control/vector/host. Every case uses identical useful bytes.',
        software_sha256={name:sha256(ROOT/'scripts'/name) for name in
                         ('plane_wave_synthetic_sweep.py', 'plane_wave_model.py')}))
    print(json.dumps([r for r in records if r['shard_kib'] in (32, 40, 80) and
                     r['selected_experts'] in (1, 2, 8)], indent=2))


if __name__ == '__main__':
    main()
