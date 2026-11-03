#!/usr/bin/env python3
"""Seal the finite paired-Down screen and fixed graph14 no-go decision."""
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, x):
    Path(p).write_text(json.dumps(x, indent=2) + '\n')


def main():
    protocols = [HERE / 'protocol.frozen.json', HERE / 'unitflow-bound/protocol.frozen.json',
                 HERE / 'unitflow-bound/restored-feasible/protocol.frozen.json', HERE / 'native-graph14/protocol.frozen.json']
    for p in protocols:
        for item in json.loads(p.read_text())['inputs']:
            assert sha(item['path']) == item['sha256']
    original_hashes = json.loads((HERE / 'audit_inputs.frozen.json').read_text())
    for item in original_hashes:
        assert sha(item['path']) == item['sha256']
    result = json.loads((HERE / 'native-graph14/report.json').read_text())
    assert not result['gate_ge2mean_or5tail']
    assert result['query_count'] == 512 and result['baseline_cost_and_LB_exact_values'] == 1024
    assert result['mean_gain_pct'] < 2 and result['p99_gain_pct'] < 5
    with np.load(HERE / 'arrays.npz') as z:
        rows = z['rows'].copy(); edges = z['layouts'][14].copy()
    copied, cases, ledger = 0, 0, []
    for e in range(64):
        starts = np.r_[0, np.cumsum(rows[e, 1])]
        original = [np.arange(starts[c], starts[c+1], dtype=np.int32) for c in range(16)]
        physical = [x.copy() for x in original]; inventory = []
        for j, (src, dest) in enumerate(edges[e]):
            packet = original[src]
            assert len(packet) == len(original[dest]) == 128
            physical[dest] = np.r_[original[dest], packet]
            inventory.append(dict(edge=j, source_chip=int(src), destination_chip=int(dest),
                                  copied_original_Down_row_ids=packet.tolist(), coefficient_bytes=65536))
            copied += 65536
        counts = np.bincount(np.concatenate(physical), minlength=1536)
        assert np.all((counts == 1) | (counts == 2)) and (counts == 2).sum() == 768
        for mask in range(64):
            lengths = rows[e, 1].copy()
            for j, (src, dest) in enumerate(edges[e]):
                if mask >> j & 1:
                    lengths[src] = 0; lengths[dest] = 256
            selected = np.concatenate([physical[c][:lengths[c]] for c in range(16)])
            assert np.array_equal(np.sort(selected), np.arange(1536))
            assert (lengths * 512 % 65536 == 0).all()
            cases += 1
        ledger.append(dict(expert=e, original_unique_rows=1536, resident_row_slots=2304,
                           physical_original_Down_row_ids=[x.tolist() for x in physical], copies=inventory,
                           scales='Original (expert,Down,logical row0..1535) immutable F32 row scale. Duplicated slots reference same scale identity.',
                           views='Each edge0: source128+dest128. Edge1: source0+dest256. Retain core before copied suffix. Complete private512 input.'))
    assert copied == 24 * 1048576 and cases == 4096
    with gzip.open(HERE / 'native-graph14/physical_row_identity_ledger.json.gz', 'wt') as f:
        json.dump(ledger, f)
    decision = dict(decision='SEALED_FINITE_NO_GO', selected_fixed_graph=14,
                    reason='Zero-selector complete native mean1.94856983695% andP99 0.716982063945% improvements fail2mean/5tail screen gate.',
                    result=result, original_screen_protocol_sha256=sha(protocols[0]),
                    unitflow_protocol_sha256=sha(protocols[1]), native_protocol_sha256=sha(protocols[3]),
                    selected_graph14_symbolic_all64_masks_cases=cases,
                    coefficient_copy_bytes=copied, extra_pages=1536, extra_reserved_waves=384,
                    stop='No selectorbenchmark, strongwholeDown32/partialRDA fit, othergraphnative, CP, weights, all-layer or heldout expansion.',
                    scope='Finite32routeblind work/LP andunitflow screen, oneTRAINselected graph14/nativepolicy. Notglobal replica/scheduler impossibility, realdevice proof, newalgorithm or publication claim.')
    dump(HERE / 'decision.json', decision)
    target = HERE / 'completion.json'; assert not target.exists()
    files = sorted(p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p != target)
    dump(target, dict(status='SEALED_FINITE_NO_GO', root_inputs_unchanged=True,
                      no_weights_or_heldout=True, selected_native_only_graph14=True,
                      files=[dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p)) for p in files]))
    print(json.dumps(dict(completion=str(target), completion_sha256=sha(target), files=len(files)), indent=2))


if __name__ == '__main__':
    main()
