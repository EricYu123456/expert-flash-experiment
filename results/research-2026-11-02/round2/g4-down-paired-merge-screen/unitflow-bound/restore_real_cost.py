#!/usr/bin/env python3
"""Restore source506ns overhead to flow endpoints; arithmetic, not native replay."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent
ROOT = HERE.parents[4]
OUT = HERE / 'restored-feasible'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, x):
    Path(p).write_text(json.dumps(x, indent=2) + '\n')


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def main():
    OUT.mkdir(exist_ok=False)
    main_source = ROOT / 'scripts/research_20261103_g4_down_paired_merge_screen.py'
    pack_source = ROOT / 'scripts/research_20261103_g4_elastic_prefix_contract_audit.py'
    paths = [Path(__file__).resolve(), HERE / 'arrays.npz', HERE / 'report.json', SOURCE / 'arrays.npz',
             SOURCE / 'report.json', main_source, pack_source]
    protocol = dict(scope='Restore omitted506ns/source packet and evaluate already available legal bits. No LP/flow refit or native/weights/heldout. Two-pass samegraph static control only for newly best graph.',
                    inputs=[dict(path=str(p), sha256=sha(p)) for p in paths])
    dump(OUT / 'protocol.frozen.json', protocol)
    with np.load(SOURCE / 'arrays.npz') as z:
        arrays = {k: z[k].copy() for k in z.files}
    with np.load(HERE / 'arrays.npz') as z:
        flow = {k: z[k].copy() for k in z.files}
    records, candidates, bits_list = [], [], []
    for j in range(32):
        pairs = arrays['layouts'][j][arrays['routes']].reshape(512, 36, 2)
        take = flow['assignments'][j]
        assert np.all((take == pairs[:, :, 0]) | (take == pairs[:, :, 1]))
        bits = (take == pairs[:, :, 1]).astype(np.int8)
        effects = arrays['effects'][j][arrays['routes']].reshape(512, 36, 16)
        actual = arrays['base'] + np.einsum('qj,qjc->qc', bits, effects)
        counts = np.asarray([np.bincount(a, minlength=16) for a in take])
        source_counts = np.asarray([np.bincount(a[a == pairs[q, :, 0]], minlength=16) for q, a in enumerate(take)])
        independently_restored = flow['fixed_base_with_dispatch'][j] - np.arange(16) * 250 + counts * 28256 + source_counts * 506
        assert np.array_equal(actual, independently_restored)
        restored_work = actual.max(1)
        better = arrays['feasible_values'][j] < restored_work
        bits[better] = arrays['feasible_bits'][j, better]
        best_available = np.minimum(restored_work, arrays['feasible_values'][j])
        assert np.all(best_available <= arrays['base'].max(1))
        candidates.append(best_available); bits_list.append(bits)
        records.append(dict(layout=j, restored_endpoint_mean_work_ns=float(restored_work.mean()),
                            best_available_mean_work_ns=float(best_available.mean()), best_available_p99_work_ns=float(np.percentile(best_available, 99)),
                            best_available_work_gain_pct=float(100 * (1 - best_available.mean() / arrays['base'].max(1).mean()))))
    best = min(range(32), key=lambda j: records[j]['best_available_mean_work_ns'])
    main_module = load(main_source, 'frozen_paired_merge_main')
    static = main_module.static_fit(arrays['base'], arrays['effects'][best], arrays['routes'])
    packing = load(pack_source, 'independent_generic_packing')
    resident = arrays['rows'].copy()
    for e in range(64):
        for src, dest in arrays['layouts'][best, e]:
            resident[e, 1, dest] = 256
    ledger, jobs = packing.pack(resident, arrays['input_columns'])
    assert ledger['coefficient_bytes'] == 172.5 * 1048576
    assert ledger['pages'] == 11296 and ledger['waves'] == 2824
    dump(OUT / 'best_resident_addresses.json', dict(layout=best, ledger=ledger, jobs=jobs,
                                                  caveat='Static repack of stored row slots. Core/donor row-ID rule from original six matching edges, not new independent logical rows.'))
    np.savez_compressed(OUT / 'arrays.npz', best_available_actual_work=np.asarray(candidates), legal_merge_bits=np.asarray(bits_list),
                        best_layout_static_bits=static[1], best_layout_static_values=static[2])
    report = dict(status='Actual chip-owned work of legal bits after restoring source506ns; not native performance',
                  protocol_sha256=sha(OUT / 'protocol.frozen.json'), records=records, best_layout=best,
                  best_record=records[best], all32x512_chip_vectors_crosschecked_two_formulas=True,
                  best_layout_static_control=dict(mean_work_ns=static[0], p99_work_ns=float(np.percentile(static[2], 99)),
                                                  passes=static[3], initial=static[4],
                                                  dynamic_vs_static_work_pct=float(100 * (1 - candidates[best].mean() / static[0]))),
                  best_resident_ledger=ledger,
                  warning='Flow LB still excludes506ns/source; only this restored evaluation is feasible chip work. No actual latency/scatter/selector claim or strongwholeDown32 comparison.')
    for item in protocol['inputs']:
        assert sha(item['path']) == item['sha256']
    dump(OUT / 'report.json', report)
    print(json.dumps({k:v for k,v in report.items() if k!='records'}, indent=2))


if __name__ == '__main__':
    main()
