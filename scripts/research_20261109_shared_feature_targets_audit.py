#!/usr/bin/env python3
"""Lightweight final audit of offline native targets and execution provenance.

Generation receipts contain full payload SHA256 checks. This final audit checks
their identity, all payload sizes, source freezes and TRAIN-only prototype IDs;
it deliberately does not reread the roughly 80 GiB of target/hidden payloads.
"""
from collections import Counter
import json
from pathlib import Path

import numpy as np

from research_20261109_shared_feature_targets import DATA, ROOT, dump, sha


def load(path):
    return json.loads(path.read_text())


def main():
    combined_path = DATA / 'targetgen-stream/execution.json'
    combined = load(combined_path)
    assert combined['status'] == 'complete'
    freezes = {}
    for name in ('targetgen', 'targetgen-stream'):
        protocol = load(DATA / name / 'protocol.frozen.json')
        for path, expected in protocol['inputs_sha256'].items():
            actual = sha(Path(path))
            assert actual == expected, path
            freezes[path] = actual
    assert load(DATA / 'targetgen/superseded.json')['status'] == 'superseded_partial'
    assert load(DATA / 'targetgen/execution.json')['status'] == 'failed'
    rows, counts, total_bytes, checked = {}, Counter(), 0, 0
    prototype_ids, manifests = {}, {}
    for layer in (1, 13, 26):
        frequency = np.zeros(64, dtype=np.int64)
        for folder in sorted((DATA / 'captures64/train').glob('shard*')):
            capture = load(folder / 'capture_manifest.json')
            item = capture['layers'][str(layer)]['arrays']['selected_experts']
            ids = np.memmap(folder / item['path'], dtype=item['dtype'], mode='r', shape=tuple(item['shape']))
            frequency += np.bincount(ids.ravel(), minlength=64)
        owners = sorted(range(64), key=lambda e: (-int(frequency[e]), e))[:8]
        prototype_ids[str(layer)] = owners
        for split in ('train', 'validation'):
            key = f'{split}/layer{layer}'
            path = DATA / f'targets64/{key}/targets.json'
            manifest = load(path)
            assert sha(path) == combined['manifests'][key]['sha256']
            assert manifest['status'] == 'complete'
            assert manifest['expert_ids'] == list(range(64))
            assert set(map(int, manifest['prototype_hidden'])) == set(owners)
            rows[key] = manifest['rows']
            for source, identity in manifest['capture_manifest_sha256'].items():
                assert sha(Path(source)) == identity
            for e in range(64):
                target = manifest['targets'][str(e)]
                receipt = load(path.parent / f'expert{e}.json')
                assert target == receipt
                assert target['shape'] == [manifest['rows'], 2048]
                assert target['dtype'] == 'float32'
                size = Path(target['path']).stat().st_size
                assert size == manifest['rows'] * 2048 * 4
                total_bytes += size
                assert target['numeric_binary_sha256'] == combined['numeric_binary_sha256']
                provenance = combined['expert_provenance'][f'{key}/expert{e}']
                assert provenance['target_sha256'] == target['target_sha256']
                counts[provenance['revision']] += 1
                if e in {1: (14, 40), 13: (10, 56), 26: (0, 1)}[layer]:
                    assert target['all_native_capture_fixed_target_floats_bitchecked'] == manifest['rows'] * 2048
                    checked += target['all_native_capture_fixed_target_floats_bitchecked']
            for item in manifest['prototype_hidden'].values():
                assert item['shape'] == [manifest['rows'], 1408]
                assert item['dtype'] == 'float32'
                size = Path(item['path']).stat().st_size
                assert size == manifest['rows'] * 1408 * 4
                total_bytes += size
            manifests[key] = combined['manifests'][key]
    assert sum(counts.values()) == 384
    report = dict(status='passed', source_freezes_unchanged=freezes,
                  combined_execution_path=str(combined_path), combined_execution_sha256=sha(combined_path),
                  manifests=manifests, rows=rows, target_files=384, prototype_hidden_files=48,
                  target_and_hidden_payload_bytes=total_bytes,
                  target_revisions=dict(counts), prototype_ids_TRAIN_frequency_top8=prototype_ids,
                  native_capture_fixed_target_floats_bitchecked=checked,
                  numeric_binary_sha256=combined['numeric_binary_sha256'],
                  original_attempt='superseded_partial; VALID completed and TRAIN complete receipts preserved',
                  final_audit_scope='All final file sizes and receipt/manifest/provenance identities checked; frozen source/binary and capture manifests rehashed. Full target/hidden payload hashes are recorded by their generating receipts, not reread by this final lightweight audit.',
                  arithmetic='Canonical W8 weights and full-vector A8; exact integer-valued FP64 BLAS dot, frozen C++ F32 rescale, pinned native SwiGLU; no attention rerun for offpath targets.')
    dump(DATA / 'targetgen-stream/final_audit.json', report)
    dump(ROOT / 'results/research-2026-11-09-shared-features/native_target_audit.json', report)
    print(json.dumps(dict(status=report['status'], targets=384, prototypes=48, bitchecked_floats=checked,
                          payload_GiB=total_bytes / 2**30, revisions=dict(counts))))


if __name__ == '__main__':
    main()
