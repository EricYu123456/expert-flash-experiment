#!/usr/bin/env python3
"""Replay real MoE routes under deterministic fixed expert-placement permutations.

Only the expert-to-placement-slot mapping changes. Input routing JSONL, original
logical expert identities, tensor dimensions, and selected weight volume do not.
Results use the same INT8 timing model as aif_model.py, with ready-phase scheduling.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys

import aif_model as model


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_mapping(events, dimensions, placement, group_chips):
    # evaluate() uses expert_ids only to select a physical group, never to fetch
    # weights. Retain logical IDs in a separate field in these in-memory records;
    # the original events and trace files are never mutated.
    remapped = [dict(event, logical_expert_ids=event['expert_ids'],
                     expert_ids=[placement[e] for e in event['expert_ids']])
                for event in events]
    return model.evaluate(remapped, dimensions, group_chips,
                          cold_per_block=1, serial_experts=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-manifest', type=Path, required=True)
    parser.add_argument('--seeds', type=int, default=8,
                        help='Use permutation seeds 0 through SEEDS-1 (default: 8)')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error('--seeds must be positive')
    source = json.loads(args.run_manifest.read_text())
    metadata_path = Path(source['metadata'])
    trace_paths = [Path(path) for path in source['traces']]
    metadata = json.loads(metadata_path.read_text())
    dimensions = model.tensor_dimensions(metadata)
    events = model.load_events(trace_paths)
    arch = metadata['metadata']['general.architecture']
    expert_count = metadata['metadata'][arch + '.expert_count']
    for event in events:
        if event['layer'] not in dimensions:
            raise ValueError('Routing layer absent from metadata')
        if any(e < 0 or e >= expert_count for e in event['expert_ids']):
            raise ValueError('Logical expert ID outside metadata bounds')
    output_dir = args.output_dir or args.run_manifest.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, permutations = [], {}
    identity = list(range(expert_count))
    reference = model.evaluate(events, dimensions, 16, cold_per_block=1,
                               serial_experts=False)
    for seed in range(args.seeds):
        placement = identity.copy()
        random.Random(seed).shuffle(placement)
        assert sorted(placement) == identity
        permutations[str(seed)] = placement
        for group in (1, 2, 4, 8, 16):
            result = (reference if group == 16 else
                      evaluate_mapping(events, dimensions, placement, group))
            assert result['useful_bytes'] == reference['useful_bytes']
            assert result['routing_records'] == len(events)
            rows.append(dict(
                seed=seed, group_chips=group,
                ffn_ms_per_token=result['ffn_ms_per_token'],
                ffn_p50_ms=result['ffn_p50_ms'], ffn_p95_ms=result['ffn_p95_ms'],
                useful_bandwidth_fraction=result['useful_bandwidth_fraction'],
                active_chip_fraction=result['active_chip_fraction'],
                service_balance=result['service_balance'],
                useful_bytes=result['useful_bytes'], model_ns=result['model_ns'],
                routing_records=result['routing_records'],
                decode_tokens=result['decode_tokens'],
                cold_per_block=1, placement='fixed_permuted_expert_slots',
                schedule='ready_per_projection_phase'))
    csv_path = output_dir / 'placement_permutations.csv'
    with csv_path.open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summaries = []
    for group in (1, 2, 4, 8, 16):
        selected = [row for row in rows if row['group_chips'] == group]
        latency = [row['ffn_ms_per_token'] for row in selected]
        efficiency = [row['useful_bandwidth_fraction'] for row in selected]
        summaries.append(dict(group_chips=group,
                              min_ffn_ms_per_token=min(latency),
                              mean_ffn_ms_per_token=statistics.mean(latency),
                              max_ffn_ms_per_token=max(latency),
                              min_useful_bandwidth_fraction=min(efficiency),
                              max_useful_bandwidth_fraction=max(efficiency)))
    manifest = dict(
        source_run_manifest=str(args.run_manifest.resolve()),
        source_run_manifest_sha256=sha256(args.run_manifest),
        metadata=str(metadata_path.resolve()), metadata_sha256=sha256(metadata_path),
        model_sha256=metadata['sha256'],
        traces=[dict(path=str(path.resolve()), sha256=sha256(path)) for path in trace_paths],
        script_sha256=sha256(Path(__file__)),
        timing_model_sha256=sha256(Path(model.__file__)),
        python_version=sys.version,
        permutation_algorithm='random.Random(seed).shuffle(list(range(expert_count)))',
        permutation_scope='Same fixed permutation at every layer; expert IDs remain logical identities.',
        placement_rule='group = placement_slot[logical_expert_id] % (16 / group_chips)',
        original_id_baseline=str((args.run_manifest.parent / 'placement.csv').resolve()),
        seeds=list(range(args.seeds)), expert_count=expert_count,
        logical_expert_id_to_placement_slot=permutations,
        all_chip_result_reused=True,
        unchanged=['Original input trace files and selected logical expert sets',
                   'Original per-layer projection dimensions and INT8 weight volume',
                   'One cold ordinary-read-equivalent per block run',
                   'Ready scheduling inside each projection phase'],
        limitations=['Eight placement samples are robustness evidence, not a statistical population bound.',
                     'No new routing inference is performed and no weights are moved on physical hardware.',
                     'Timing-model limitations from the source run still apply.'],
        summaries=summaries)
    manifest_path = output_dir / 'placement_permutations.manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(csv=str(csv_path), manifest=str(manifest_path),
                          rows=len(rows), summaries=summaries), indent=2))


if __name__ == '__main__':
    main()
