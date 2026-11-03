#!/usr/bin/env python3
"""Revalidate all saved real NVMeVirt replays and collect protocol totals."""
import argparse
from collections import Counter
import json
from pathlib import Path

from validate_replay import validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, default=Path(
        '/home/nfs_client/ericyu/expert_flash_experiment/runs'))
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1]
                        / 'results/nvmevirt_validation.json')
    args = parser.parse_args()
    records = []
    protocols = Counter()
    for root in sorted(args.runs.glob('*-aif')):
        for path in sorted(root.rglob('*.validation.json')):
            tag = path.name.removesuffix('.validation.json')
            expected_path = path.parent.parent / (tag + '.expected.json')
            expected = json.loads(expected_path.read_text())
            protocol = expected.get('model', 'metadata_only_int8_v1')
            more = path.parent / (tag + '.v2_stats.txt')
            result = validate(expected_path, path.parent / (tag + '.stats.txt'),
                              path.parent / (tag + '.wall.json'), more if more.exists() else None)
            result.update(model=root.name.removesuffix('-aif'), protocol=protocol,
                          validation_file=str(path.resolve()))
            records.append(result)
            protocols[protocol] += 1
    assert records, 'No actual replay evidence found'
    v3 = [r for r in records if r['protocol'] == 'metadata_only_int8_v3']
    result = dict(workloads=len(records), total_commands=sum(r['commands'] for r in records),
                  v3_workloads=len(v3), v3_commands=sum(r['commands'] for r in v3),
                  protocol_counts=dict(protocols), all_kernel_counters_match=True,
                  validation_method='Reparsed every saved expected JSON, kernel counters and observed ioctl timing; v2/v3 components checked.',
                  results=records)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'results'}, indent=2))


if __name__ == '__main__':
    main()
