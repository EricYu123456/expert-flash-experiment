#!/usr/bin/env python3
"""Fixed GU3/Down14 row ownership, prefix ranges, joint work and BFD audit."""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/research-2026-11-02/round2'
SOURCE = BASE / 'g4-joint-prefix-replica-screen'
OUT = SOURCE / 'independent-contract'
GU = BASE / 'g4-elastic-prefix-screen/work-v3-independent-prefix'
DOWN = BASE / 'g4-down-paired-merge-screen'
INC = BASE / 'compact-coupled-solver/continuation/native/0107.npz'
PACK = ROOT / 'scripts/research_20261103_g4_elastic_prefix_contract_audit.py'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, x):
    Path(p).write_text(json.dumps(x, indent=2) + '\n')


def job(n, c):
    if not n:
        return 0, 0, 0, 0
    pages = (int(n) * c + 16383) // 16384
    waves = (pages + 3) // 4; emitted = pages * 16384 // c
    return 250 + (c + 1) // 2 + waves * 28000 + 2 * emitted, pages, waves, emitted


def main():
    assert (SOURCE / 'protocol.frozen.json').exists()
    OUT.mkdir(exist_ok=False)
    producer = json.loads((SOURCE / 'protocol.frozen.json').read_text())
    paths = [Path(__file__).resolve(), SOURCE / 'protocol.frozen.json', SOURCE / 'work_arrays.npz', SOURCE / 'work_report.json',
             SOURCE / 'LP_arrays.npz', GU / 'contract_arrays.npz', GU / 'rowid_ledgers.json', GU / 'protocol.frozen.json',
             DOWN / 'arrays.npz', DOWN / 'completion.json', INC, PACK]
    protocol = dict(scope='Fixed GUlayout3 and Downgraph14 only. Exhaust each phase view separately; no new images, fit, native or heldout.',
                    producer_protocol_sha256=sha(SOURCE / 'protocol.frozen.json'),
                    inputs=[dict(path=str(p), sha256=sha(p)) for p in paths])
    dump(OUT / 'protocol.frozen.json', protocol)
    for item in producer['inputs']:
        assert sha(item['path']) == item['sha256']
    with np.load(SOURCE / 'work_arrays.npz') as z:
        work = {k: z[k].copy() for k in z.files}
    with np.load(GU / 'contract_arrays.npz') as z:
        gu = {k: z[k].copy() for k in z.files}
    with np.load(DOWN / 'arrays.npz') as z:
        down = {k: z[k].copy() for k in z.files}
    with np.load(INC) as z:
        rows, columns = z['rows'].copy(), z['input_columns'].copy()
    entries = json.loads((GU / 'rowid_ledgers.json').read_text())[3]['experts']
    assert np.array_equal(gu['base'], down['base']) and np.array_equal(work['base'], gu['base'])
    assert np.array_equal(work['routes'], gu['routes']) and np.array_equal(work['routes'], down['routes'])
    assert np.array_equal(work['DG'], gu['D'][3]) and np.array_equal(work['DZ'], gu['Z'])
    assert np.array_equal(work['DD'], down['effects'][14]) and np.array_equal(work['Down_edges'], down['layouts'][14])
    table = np.zeros((65, 2, 16), np.int64)
    for e in range(65):
        for p in range(2):
            for c, n in enumerate(rows[e, p]):
                table[e, p, c] = job(n, int(columns[e, p]))[0]
    routes = work['routes']; base = table[routes].sum((1, 2)) + table[64].sum(0)
    assert np.array_equal(base, work['base'])
    resident = gu['resident_rows'].copy()
    for e in range(64):
        for src, dest in down['layouts'][14, e]:
            resident[e, 1, dest] = 256
    assert np.array_equal(resident, work['resident_rows'])
    assert np.array_equal(resident[64], rows[64])
    spec = importlib.util.spec_from_file_location('independent_joint_BFD', PACK)
    packing = importlib.util.module_from_spec(spec); spec.loader.exec_module(packing)
    baseline_ledger, _ = packing.pack(rows, columns)
    resident_ledger, jobs = packing.pack(resident, columns)
    assert np.array_equal([[j[k] for k in ['chip', 'stored_row_slots', 'columns', 'pages', 'waves']] for j in jobs], work['resident_jobs'])
    assert np.array_equal([[j[k] for k in ['chip', 'block', 'wordline', 'waves', 'pages', 'first_plane']] for j in jobs], work['resident_addresses'])
    assert [(n + 191) // 192 for n in resident_ledger['chip_reserved_waves']] == resident_ledger['chip_blocks'] == [2] * 16
    addresses = {(j['expert'], j['phase'], j['chip']): j for j in jobs}
    gu_counts, down_counts, row_ledgers = [], [], []
    gu_copies = down_copies = prefix_checks = 0
    for e, entry in enumerate(entries):
        image = {int(c): np.asarray(v, np.int32) for c, v in entry['resident_prefix_image'].items()}
        original = {int(c): np.asarray(v, np.int32) for c, v in entry['B_original_rows'].items()}
        multiplicity = np.bincount(np.concatenate(list(image.values())), minlength=1024)
        assert (multiplicity == 2).sum() == entry['copied_rows']
        assert np.all((multiplicity == 1) | (multiplicity == 2))
        gu_copies += entry['copied_rows'] * 1536
        metrics = []
        for mask in range(256):
            u = np.array([(mask >> i) & 1 for i in range(8)], np.int64)
            nr = rows[e, 0].copy()
            for i, c in enumerate(entry['sources']):
                nr[c] = 128 if u[i] else rows[e, 0, c]
            for i, c in enumerate(entry['low_destinations']):
                nr[c] = 0 if u[i] else 40
            hd = entry['high_destination']
            nr[hd] = len(original[hd]) if not u[7] else entry['high_prefix_rows'] if not u.all() else 0
            owners = {}; actual = np.zeros(16, np.int64)
            pg = wa = emitted = 0
            for c, n in enumerate(nr):
                if not n:
                    continue
                for position, logical in enumerate(image[c][:n]):
                    owners.setdefault(int(logical), (c, position))
                cost, pages, waves, emit = job(n, 1536); actual[c] = cost
                address = addresses[e, 0, c]
                assert pages <= address['pages'] and waves <= address['waves']
                assert address['wordline'] + waves <= 192
                pg += pages; wa += waves; emitted += emit; prefix_checks += 1
            assert sorted(owners) == list(range(1024))
            for logical, (chip, position) in owners.items():
                assert position < nr[chip] and image[chip][position] == logical
                assert (logical // 512, logical % 512) in ((0, logical % 512), (1, logical % 512))
            expected = table[e, 0] + np.einsum('i,ic->c', u, gu['D'][3, e]) + int(u.all()) * gu['Z'][e]
            assert np.array_equal(actual, expected)
            metric = [int((nr > 0).sum()), pg, wa, int(nr.sum()), emitted, len(owners)]
            assert np.array_equal(metric, gu['view_counts'][3, e, mask])
            metrics.append(metric)
        gu_counts.append(metrics)
        starts = np.r_[0, np.cumsum(rows[e, 1])]
        original_down = [np.arange(starts[c], starts[c+1], dtype=np.int32) for c in range(16)]
        down_image = [v.copy() for v in original_down]
        for src, dest in down['layouts'][14, e]:
            down_image[dest] = np.r_[original_down[dest], original_down[src]]
            down_copies += 65536
        counts = np.bincount(np.concatenate(down_image), minlength=1536)
        assert (counts == 2).sum() == (counts == 1).sum() == 768
        metrics_down = []
        for mask in range(64):
            v = np.array([(mask >> i) & 1 for i in range(6)], np.int64)
            nr = rows[e, 1].copy()
            for i, (src, dest) in enumerate(down['layouts'][14, e]):
                if v[i]:
                    nr[src] = 0; nr[dest] = 256
            selected = np.concatenate([down_image[c][:nr[c]] for c in range(16)])
            assert np.array_equal(np.sort(selected), np.arange(1536))
            actual = np.asarray([job(n, 512)[0] for n in nr])
            assert np.array_equal(actual, table[e, 1] + np.einsum('i,ic->c', v, down['effects'][14, e]))
            assert nr.sum() == 1536 and (nr * 512 % 65536 == 0).all()
            for c, n in enumerate(nr):
                if n:
                    _, pages, waves, emitted = job(n, 512)
                    address = addresses[e, 1, c]
                    assert pages <= address['pages'] and waves <= address['waves'] and address['wordline'] + waves <= 192
                    assert emitted == n; prefix_checks += 1
            metrics_down.append([int((nr > 0).sum()), 48, 12, 1536, 1536])
        down_counts.append(metrics_down)
        row_ledgers.append(dict(expert=e, GU_physical_original_row_ids={str(c): v.tolist() for c, v in image.items()},
                                GU_scale_rule='Original logicalGU row<512 is gate, >=512 is up; operator row modulo512. Host consumes1024unique only.',
                                Down_physical_original_row_ids=[v.tolist() for v in down_image],
                                Down_scale_rule='Original expert/Down/row0..1535. Full private512input after uniqueGU activation/A8.',
                                GU_unique_rows=1024, Down_unique_rows=1536, copied_GU_rows=entry['copied_rows'], copied_Down_rows=768))
    assert gu_copies == 35638272 and down_copies == 25165824
    assert resident_ledger['coefficient_bytes'] - baseline_ledger['coefficient_bytes'] == gu_copies + down_copies
    assert resident_ledger['pages'] - baseline_ledger['pages'] == 3584
    assert resident_ledger['waves'] - baseline_ledger['waves'] == 896
    with gzip.open(OUT / 'physical_rows_and_addresses.json.gz', 'wt') as f:
        json.dump(dict(ledger=resident_ledger, jobs=jobs, row_images=row_ledgers,
                       semantics='Static joint repack. Storedrow slots/copies distinct from logical dimensions; views read prefixes and scatter immutable rowIDs.'), f)
    np.savez_compressed(OUT / 'all_view_counters.npz', GU=np.asarray(gu_counts), Down=np.asarray(down_counts))
    # Check each existing chosen joint query against ordinary row counts, not only the algebra.
    chosen_vectors = []
    for q, selected in enumerate(routes):
        loads = table[64].sum(0).copy()
        for k, e in enumerate(selected):
            u, v = work['chosen_u'][q, k], work['chosen_v'][q, k]
            mask = sum(int(x) << i for i, x in enumerate(u)); entry = entries[e]
            actual_gu = table[e, 0] + np.einsum('i,ic->c', u, gu['D'][3, e]) + int(u.all()) * gu['Z'][e]
            actual_down = table[e, 1] + np.einsum('i,ic->c', v, down['effects'][14, e])
            assert gu_counts[e][mask][-1] == 1024
            loads += actual_gu + actual_down
        formula = base[q] + np.einsum('ei,eic->c', work['chosen_u'][q], work['DG'][selected]) + (work['chosen_u'][q].all(1)[:, None] * work['DZ'][selected]).sum(0) + np.einsum('ei,eic->c', work['chosen_v'][q], work['DD'][selected])
        assert np.array_equal(loads, formula) and loads.max() == work['chosen_work'][q]
        chosen_vectors.append(loads)
    # All eight producer paths retain correct AND and additive phase work.
    for order in range(2):
        for start in range(4):
            u, v = work['policy_u'][order, start], work['policy_v'][order, start]
            values = base + np.einsum('qei,qeic->qc', u, work['DG'][routes]) + (u.all(2)[:, :, None] * work['DZ'][routes]).sum(1) + np.einsum('qei,qeic->qc', v, work['DD'][routes])
            assert np.array_equal(values.max(1), work['policy_values'][order, start])
    np.savez_compressed(OUT / 'chosen_chip_vectors.npz', chip_work=np.asarray(chosen_vectors), routes=routes)
    result = dict(status='PASS fixed-image independent symbolic/address/work contract', producer_protocol_sha256=sha(SOURCE / 'protocol.frozen.json'),
                  protocol_sha256=sha(OUT / 'protocol.frozen.json'), fixed_GU_layout=3, fixed_Down_graph=14,
                  GU_view_cases=64*256, Down_view_cases=64*64, every_prefix_address_checks=prefix_checks,
                  chosen_joint_query_vectors=512, all_policy_vectors=4096,
                  logical_rows=dict(routedGU=1024, routedDown=1536, GU_computed_billing_may_exceed_unique=True),
                  baseline=baseline_ledger, joint_resident=resident_ledger,
                  copies=dict(GU_bytes=gu_copies, Down_bytes=down_copies, total_bytes=gu_copies+down_copies,
                              programmed_page_increment=3584, reserved_wave_increment=896, programmed_byte_increment=56*1048576),
                  equation='W=CPbase+sum(DG*u)+sum(DZ*AND8u)+sum(DD*v); actual source506ns included in DD.',
                  dispatcher='Only host+maxwork used. Fixedchip c*250 Down-only bound doesnotapply when GUbits skip jobs.',
                  shared_and_input_dimensions_unchanged=True, native_replay_performed=False,
                  limitation='No weights, nativeoutput/scaling/resetABI proof, new placementfit or jointnative result audit. Joint native traces to audit separately when produced.')
    for item in protocol['inputs'] + producer['inputs']:
        assert sha(item['path']) == item['sha256']
    dump(OUT / 'audit.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
