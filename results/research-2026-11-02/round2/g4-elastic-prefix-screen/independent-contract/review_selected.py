#!/usr/bin/env python3
"""Read-only cross-check of the now-completed work-v1 selected physical image."""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
WORK = HERE.parent / 'work-v1'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, x):
    Path(p).write_text(json.dumps(x, indent=2) + '\n')


def main():
    audit = ROOT / 'scripts/research_20261103_g4_elastic_prefix_contract_audit.py'
    spec = importlib.util.spec_from_file_location('independent_elastic_contract', audit)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    paths = [Path(__file__).resolve(), audit, WORK / 'protocol.frozen.json', WORK / 'work_arrays.npz',
             WORK / 'work_report.json', WORK / 'copy_rowid_ledger.json', WORK / 'physical_prefix_views.json']
    inputs = [dict(path=str(p), sha256=sha(p)) for p in paths]
    dump(HERE / 'selected_inputs.frozen.json', inputs)
    protocol = json.loads((WORK / 'protocol.frozen.json').read_text())
    report = json.loads((WORK / 'work_report.json').read_text())
    assert sha(WORK / 'protocol.frozen.json') == report['protocol_sha256']
    for item in protocol['inputs']:
        assert sha(item['path']) == item['sha256']
    with np.load(WORK / 'work_arrays.npz') as z:
        arrays = {k: z[k].copy() for k in z.files}
    with np.load(HERE / 'resident_rows.npz') as z:
        columns = z['input_columns'].copy(); resident_examples = z['resident_rows'].copy()
    with gzip.open(HERE / 'row_identity_and_fringe.json.gz', 'rt') as f:
        identities = {(x['omit_rank'], x['expert']): x for x in json.load(f)}
    rowids = json.loads((WORK / 'copy_rowid_ledger.json').read_text())
    chosen = arrays['selected_layout']; rows = arrays['rows_B']; resident = rows.copy()
    for e, rank in enumerate(chosen):
        resident[e, 0] = resident_examples[rank, e, 0]
        independent = identities[int(rank), e]['physical_original_logical_row_ids']
        root = rowids[e]['resident_prefix_image']
        assert all(root[str(c)] == independent[c] for c in range(16))
        assert rowids[e]['excluded_high_rank'] == rank
        assert rowids[e]['copied_rows'] == sum(len(x) for x in independent) - 1024
    assert np.array_equal(resident, arrays['resident_rows'])
    ledger, jobs = module.pack(resident, columns)
    assert np.array_equal([[j[k] for k in ['chip', 'stored_row_slots', 'columns', 'pages', 'waves']] for j in jobs], arrays['resident_jobs'])
    assert np.array_equal([[j[k] for k in ['chip', 'block', 'wordline', 'waves', 'pages', 'first_plane']] for j in jobs], arrays['resident_addresses'])
    assert ledger['coefficient_bytes'] == report['budget']['resident_physical']['useful_bytes']
    assert ledger['allocated_lsb_block_bytes'] == 240 * 1048576
    # Exact declared GU-only service independently rebuilt from pages and emitted rows.
    B_pages = (rows[:64, 0] * 1536 + 16383) // 16384
    B_emitted = B_pages * 16384 // 1536
    B_cost = 250 + 768 + (B_pages + 3) // 4 * 28000 + B_emitted * 2
    assert np.all(np.unique(B_cost) == [29102, 57188])
    A = arrays['A_menu']
    A_cost = np.where(A > 0, 250 + 768 + 3 * 28000 + 128 * 2, 0)
    assert np.array_equal(A_cost - B_cost[:, None, :], arrays['delta_menu'])
    assert np.all(A_cost.sum(2) - B_cost.sum(1)[:, None] == -36214)
    bits = ((np.arange(64)[:, None] >> np.arange(6)) & 1)
    loads = arrays['base'][:, None, :] + np.einsum('mk,qkc->qmc', bits, arrays['selected_delta'][arrays['routes']])
    values = loads.max(2).min(1)
    assert np.array_equal(values, arrays['selected_values'])
    assert np.array_equal(loads.max(2)[np.arange(512), arrays['selected_masks']], values)
    sorted_LB = 2000 + (np.sort(loads, axis=2)[:, :, ::-1] + np.arange(16) * 250).max(2).min(1)
    assert np.array_equal(sorted_LB, arrays['sorted_firststart_LB'])
    for item in inputs:
        assert sha(item['path']) == item['sha256']
    for item in protocol['inputs']:
        assert sha(item['path']) == item['sha256']
    dump(HERE / 'selected_audit.json', dict(status='PASS read-only selected-image and declared-service audit',
                                           root_protocol_sha256=report['protocol_sha256'],
                                           selected_prefix_row_maps_checked=64, service_menu_cases=576,
                                           exhaustive_A_B_mask_cases=32768,
                                           per_expert_A_minus_B_total_chip_owned_ns=-36214,
                                           physical_BFD_ledger=ledger,
                                           copied_coefficient_bytes=report['budget']['copied_coefficient_bytes'],
                                           extra_programmed_bytes=33554432, extra_reserved_waves=512,
                                           source_padding_reused_bytes=2083840,
                                           necessary_work_mean_ns=float(values.mean()),
                                           sorted_firststart_LB_mean_ns=float(sorted_LB.mean()),
                                           root_gate_passed=report['stricter_gate_possible_ge2pct'],
                                           scope='No source mutation, fitting, native replay, weights or heldout. Geometric scale identities only.'))
    print((HERE / 'selected_audit.json').read_text())


if __name__ == '__main__':
    main()
