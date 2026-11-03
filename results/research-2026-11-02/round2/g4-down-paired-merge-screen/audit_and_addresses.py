#!/usr/bin/env python3
"""Exact-integer dual/feasible certificates and symbolic single-block addresses."""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, x):
    Path(p).write_text(json.dumps(x, indent=2) + '\n')


def main():
    protocol = json.loads((HERE / 'protocol.frozen.json').read_text())
    report = json.loads((HERE / 'report.json').read_text())
    inputs = [dict(path=str(p), sha256=sha(p)) for p in [Path(__file__).resolve(), HERE / 'arrays.npz', HERE / 'report.json', HERE / 'protocol.frozen.json']]
    dump(HERE / 'audit_inputs.frozen.json', inputs)
    with np.load(HERE / 'arrays.npz') as z:
        arrays = {k: z[k].copy() for k in z.files}
    base, routes = arrays['base'], arrays['routes']
    num_cases = 0
    for layout, edges in enumerate(arrays['layouts']):
        effect = arrays['effects'][layout][routes].reshape(512, 36, 16)
        bits = arrays['feasible_bits'][layout]
        loads = base + np.einsum('qj,qjc->qc', bits, effect)
        assert np.array_equal(loads.max(1), arrays['feasible_values'][layout])
        assert np.all(arrays['feasible_values'][layout] <= base.max(1))
        for q in range(512):
            w = [int(x) for x in arrays['lp_weights'][layout, q]]
            assert min(w) >= 0 and sum(w) > 0
            numerator = sum(w[c] * int(base[q, c]) for c in range(16))
            pairs = edges[routes[q]].reshape(36, 2)
            numerator += sum(min(0, -28762 * w[int(src)] + 28256 * w[int(dest)]) for src, dest in pairs)
            certificate = numerator // sum(w)
            assert certificate == arrays['lp_safe_bounds'][layout, q]
            assert certificate <= arrays['feasible_values'][layout, q]
            num_cases += 1
    # Reuse only our independent generic BFD implementation, not the main fitter.
    packing_source = ROOT / 'scripts/research_20261103_g4_elastic_prefix_contract_audit.py'
    spec = importlib.util.spec_from_file_location('independent_generic_pack', packing_source)
    packing = importlib.util.module_from_spec(spec); spec.loader.exec_module(packing)
    baseline_ledger, _ = packing.pack(arrays['rows'], arrays['input_columns'])
    physical, ledger_records = [], []
    checked_views = 0
    for layout in sorted({report['best_feasible_layout'], report['best_LP_layout']}):
        resident = arrays['rows'].copy(); rowids = []
        for e in range(64):
            starts = np.r_[0, np.cumsum(arrays['rows'][e, 1])]
            original = [np.arange(starts[c], starts[c+1], dtype=np.int32) for c in range(16)]
            image = [x.copy() for x in original]
            for src, dest in arrays['layouts'][layout, e]:
                assert len(original[src]) == len(original[dest]) == 128
                image[dest] = np.r_[original[dest], original[src]]; resident[e, 1, dest] = 256
            counts = np.bincount(np.concatenate(image), minlength=1536)
            assert (counts == 1).sum() == (counts == 2).sum() == 768
            for mask in range(64):
                lengths = arrays['rows'][e, 1].copy()
                for j, (src, dest) in enumerate(arrays['layouts'][layout, e]):
                    if mask >> j & 1:
                        lengths[src] = 0; lengths[dest] = 256
                ids = np.concatenate([image[c][:lengths[c]] for c in range(16)])
                assert np.array_equal(np.sort(ids), np.arange(1536))
                assert (lengths * 512 % 65536 == 0).all()
                assert lengths.sum() == 1536 and (lengths // 128).sum() == 12
                checked_views += 1
            rowids.append(dict(expert=e, edges=arrays['layouts'][layout, e].tolist(),
                               stored_original_Down_row_ids=[x.tolist() for x in image],
                               scale_ID='Original expert/Down/logicalrow scale; never physical row position.'))
        ledger, jobs = packing.pack(resident, arrays['input_columns'])
        assert ledger['coefficient_bytes'] - baseline_ledger['coefficient_bytes'] == 24 * 1048576
        assert ledger['pages'] - baseline_ledger['pages'] == 1536
        assert ledger['waves'] - baseline_ledger['waves'] == 384
        # Every runtime view is a whole-wave prefix of the resident matrix band.
        for job in jobs:
            if job['expert'] < 64 and job['phase'] == 1:
                assert job['stored_row_slots'] in (128, 256) and job['waves'] in (1, 2)
                job['logical_identity'] = 'See rowIDs; these are stored slots, including copied suffixes.'
        ledger_records.append(dict(layout=layout, **ledger))
        physical.append(dict(layout=layout, ledger=ledger, jobs=jobs, row_ids=rowids,
                             address_contract='Static repack of all jobs; original source addresses need not remain fixed. Single-block wave-aligned prefixes.'))
    with gzip.open(HERE / 'physical_address_and_row_ledger.json.gz', 'wt') as f:
        json.dump(physical, f)
    result = dict(status='PASS exact certificates and physical prefix construction',
                  exact_Python_integer_dual_certificates=num_cases, feasible_query_load_checks=num_cases,
                  symbolic_view_cases=checked_views, baseline_ledger=baseline_ledger,
                  selected_resident_ledgers=ledger_records, generic_BFD_source=str(packing_source),
                  generic_BFD_source_sha256=sha(packing_source),
                  row_scale_checks='Symbolic original row identity; not numerical weights/scales replay or native output ABI.',
                  raw_original_rows=1536, stored_row_slots_per_expert=2304, duplicate_row_slots_per_expert=768,
                  every_executed_row_useful=True, partialrows_or_fringe=0)
    for item in protocol['inputs'] + inputs:
        assert sha(item['path']) == item['sha256']
    dump(HERE / 'audit.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
