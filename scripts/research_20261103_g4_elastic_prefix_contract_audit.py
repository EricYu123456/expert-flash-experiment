#!/usr/bin/env python3
"""Bounded symbolic/address audit of fixed G4 8-source A / original B prefixes."""
import gzip
import hashlib
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/research-2026-11-02/round2'
OUT = BASE / 'g4-elastic-prefix-screen/independent-contract'
MAP = BASE / 'compact-coupled-solver/continuation/native/0107.npz'
TRAIN = BASE / 'w8-format-training/granite4tiny/decode.routes.npz'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, x):
    Path(p).write_text(json.dumps(x, indent=2) + '\n')


def pack(rows, columns):
    jobs = []
    for e in range(65):
        for phase in range(2):
            for chip, n in enumerate(rows[e, phase]):
                if n:
                    c = int(columns[e, phase]); n = int(n)
                    pages = (n * c + 16383) // 16384
                    waves = (pages + 3) // 4
                    assert 0 < waves <= 192
                    jobs.append(dict(job=len(jobs), expert=e, phase=phase, chip=chip,
                                     stored_row_slots=n, columns=c, coefficient_bytes=n * c,
                                     pages=pages, waves=waves))
    blocks, chip_waves = [], []
    for c in range(16):
        remaining = []
        for j in sorted((j for j in jobs if j['chip'] == c), key=lambda j: (-j['waves'], j['job'])):
            options = [b for b, left in enumerate(remaining) if left >= j['waves']]
            if options:
                b = min(options, key=lambda b: (remaining[b] - j['waves'], b))
            else:
                b = len(remaining); remaining.append(192)
            start = 192 - remaining[b]
            j.update(block=b, wordline=start, end_wordline=start + j['waves'], first_plane=0)
            remaining[b] -= j['waves']
        blocks.append(len(remaining))
        chip_waves.append(sum(j['waves'] for j in jobs if j['chip'] == c))
        for b in range(len(remaining)):
            spans = sorted((j['wordline'], j['end_wordline']) for j in jobs if j['chip'] == c and j['block'] == b)
            assert all(0 <= x < y <= 192 for x, y in spans)
            assert all(x[1] <= y[0] for x, y in zip(spans, spans[1:]))
    ledger = dict(stored_jobs=len(jobs), coefficient_bytes=sum(j['coefficient_bytes'] for j in jobs),
                  pages=sum(j['pages'] for j in jobs), waves=sum(j['waves'] for j in jobs),
                  chip_blocks=blocks, chip_reserved_waves=chip_waves,
                  allocated_lsb_block_bytes=sum(blocks) * 192 * 65536)
    ledger['programmed_page_bytes'] = ledger['pages'] * 16384
    ledger['reserved_wave_bytes'] = ledger['waves'] * 65536
    return ledger, jobs


def main():
    OUT.mkdir(exist_ok=False)
    began = time.perf_counter()
    paths = [Path(__file__).resolve(), MAP, TRAIN, BASE / 'compact-coupled-solver/protocol.frozen.json',
             BASE / 'g4-down-tile-copy-screen/protocol.frozen.json', ROOT / 'paper/aif.txt',
             ROOT / 'kernel/aif.h', ROOT / 'scripts/research_20261103_ordinary_read_backend.py',
             ROOT / 'scripts/research_20261103_ordinary_read_native.cpp']
    protocol = dict(scope='Symbolic and virtual single-block address construction only. TRAIN provenance; no weights, heldout, native replay or fitting.',
                    source_choices='Nine separately materialized offline candidates/expert: omit one of nine B-high chips from eight A sources. One fixed source choice supports A/B only.',
                    append_policy='Offsource original logical rows in chip/row order partitioned over ascending source chips to fill each source core to128rows.',
                    inputs=[dict(path=str(p), sha256=sha(p)) for p in paths],
                    numerical_contract='Original full-row W8 bytes and original gate/up row-scale IDs; full1536 input A8 unchanged; useful output scatter by immutable logical IDs.',
                    fringe='B source reads8pages, emits85complete rows plus512B partial next row; all page/output service charged. Only original B core rows consumed.',
                    address='Wave-aligned static BFD repack of resident row slots; Down/shared unchanged. Duplicated slots are not independent logical rows.')
    dump(OUT / 'protocol.frozen.json', protocol)
    with np.load(MAP) as z:
        rows = z['rows'].copy(); columns = z['input_columns'].copy(); routes = z['routes'].copy()
        original_jobs = z['jobs'].copy()
    assert rows.shape == (65, 2, 16) and np.all(columns[:64] == [1536, 512])
    expected = np.array([40] * 7 + [82] * 3 + [83] * 6)
    assert np.all(np.sort(rows[:64, 0], axis=1) == expected)
    assert np.all(rows[:64, 0].sum(1) == 1024)
    with np.load(TRAIN) as z:
        assert z['split'].item() == 'train' and z['completed'].all()
        assert np.array_equal(routes, z['routes'][:, :, 0].reshape(512, 6))
    baseline, original_addresses = pack(rows, columns)
    descriptors = [[j['chip'], j['stored_row_slots'], j['columns'], j['pages'], j['waves']] for j in original_addresses]
    assert np.array_equal(descriptors, original_jobs)
    cases, physical_layouts, address_ledgers, row_ledgers = [], [], [], []
    donor_copy_bytes = np.zeros((9, 64), np.int64)
    for omit_rank in range(9):
        resident = rows.copy()
        for e in range(64):
            br = rows[e, 0]; high = np.flatnonzero(br > 40); omitted = int(high[omit_rank])
            sources = [int(c) for c in high if c != omitted]
            offsource = [c for c in range(16) if c not in sources]
            starts = np.r_[0, np.cumsum(br)]
            b_ids = [np.arange(starts[c], starts[c + 1], dtype=np.int32) for c in range(16)]
            donor = np.concatenate([b_ids[c] for c in offsource])
            copy_rows = len(donor); assert copy_rows in (362, 363)
            assert copy_rows == 1024 - sum(int(br[c]) for c in sources)
            physical = [x.copy() for x in b_ids]; cursor = 0
            for c in sources:
                take = 128 - int(br[c])
                physical[c] = np.r_[physical[c], donor[cursor:cursor + take]]
                cursor += take; resident[e, 0, c] = 128
            assert cursor == len(donor)
            counts = np.bincount(np.concatenate(physical), minlength=1024)
            assert np.all(counts >= 1) and np.all(counts <= 2)
            assert (counts == 2).sum() == copy_rows
            a_ids = np.concatenate([physical[c] for c in sources])
            selected_b = np.concatenate([physical[c][:int(br[c])] for c in range(16)])
            assert np.array_equal(np.sort(a_ids), np.arange(1024))
            assert np.array_equal(np.sort(selected_b), np.arange(1024))
            # Explicit scale IDs are (gate/up, original row in that operator).
            for ids in [a_ids, selected_b]:
                scale_ids = np.column_stack([ids // 512, ids % 512])
                assert len(np.unique(scale_ids, axis=0)) == 1024
            a_rows = np.zeros(16, np.int64); a_rows[sources] = 128
            b_pages = (br * 1536 + 16383) // 16384
            a_pages = (a_rows * 1536 + 16383) // 16384
            resident_pages = (resident[e, 0] * 1536 + 16383) // 16384
            assert b_pages.sum() == 100 and ((b_pages + 3) // 4).sum() == 25
            assert a_pages.sum() == 96 and ((a_pages + 3) // 4).sum() == 24
            assert resident_pages.sum() == 132 and ((resident_pages + 3) // 4).sum() == 33
            assert sum(a_rows > 0) == 8 and sum(br > 0) == 16
            outputs = b_pages * 16384 // 1536
            assert outputs.sum() == 1059 and (outputs - br).sum() == 35
            fringe = []
            source_fringe = 0
            for c in range(16):
                complete = int(outputs[c]); partial_bytes = int(b_pages[c] * 16384 % 1536)
                if c in sources:
                    fringe_ids = physical[c][int(br[c]):complete].tolist()
                    assert len(fringe_ids) == (3 if br[c] == 82 else 2)
                    assert partial_bytes == 512
                    source_fringe += len(fringe_ids)
                    tail = dict(kind='real appended copied row fragment', logical_row=int(physical[c][complete]), coefficient_bytes=512)
                else:
                    fringe_ids = []
                    tail = dict(kind='physical page padding fragment', coefficient_bytes=partial_bytes)
                fringe.append(dict(chip=c, B_required_rows=int(br[c]), physical_stored_rows=len(physical[c]),
                                   B_read_pages=int(b_pages[c]), B_emitted_complete_rows=complete,
                                   discarded_complete_rows=complete-int(br[c]),
                                   discarded_real_copy_logical_ids=fringe_ids,
                                   discarded_padding_rows=complete-int(br[c])-len(fringe_ids), partial_row_tail=tail))
            assert source_fringe == (19 if br[omitted] == 83 else 18)
            assert sum(f['partial_row_tail']['coefficient_bytes'] for f in fringe) == 11776
            donor_copy_bytes[omit_rank, e] = copy_rows * 1536
            cases.append(dict(expert=e, omit_rank=omit_rank, omitted_high_chip=omitted,
                              source_chips=sources, offsource_chips=offsource, copy_rows=copy_rows,
                              copy_bytes=copy_rows*1536, resident_row_slots=1024+copy_rows,
                              A_jobs=8, A_pages=96, A_waves=24, A_useful_rows=1024, A_emitted_rows=1024,
                              B_jobs=16, B_pages=100, B_waves=25, B_useful_rows=1024, B_emitted_rows=1059,
                              B_discarded_real_copy_rows=source_fringe, B_discarded_padding_rows=35-source_fringe,
                              B_partial_row_tail_coefficient_bytes=11776, resident_pages=132, resident_waves=33))
            row_ledgers.append(dict(expert=e, omit_rank=omit_rank,
                                    physical_original_logical_row_ids=[x.tolist() for x in physical],
                                    scale_id_rule='flat logical GU row i -> gate if i<512 else up; row=i%512; original F32 row scale unchanged',
                                    fringe=fringe))
        ledger, addresses = pack(resident, columns)
        assert ledger['pages'] - baseline['pages'] == 2048
        assert ledger['waves'] - baseline['waves'] == 512
        assert ledger['coefficient_bytes'] - baseline['coefficient_bytes'] == donor_copy_bytes[omit_rank].sum()
        assert np.array_equal(resident[:, 1], rows[:, 1]) and np.array_equal(resident[64], rows[64])
        for j in addresses:
            if j['expert'] < 64 and j['phase'] == 0:
                j['identity'] = 'Original B core followed by offsource copies if source. See row_identity_and_fringe.json.gz for this omit_rank.'
        address_ledgers.append(dict(omit_rank=omit_rank, ledger=ledger, jobs=addresses))
        physical_layouts.append(resident)
    with gzip.open(OUT / 'row_identity_and_fringe.json.gz', 'wt') as f:
        json.dump(row_ledgers, f)
    with gzip.open(OUT / 'resident_addresses.json.gz', 'wt') as f:
        json.dump(address_ledgers, f)
    dump(OUT / 'symbolic_cases.json', cases)
    np.savez_compressed(OUT / 'resident_rows.npz', baseline_rows=rows, resident_rows=np.asarray(physical_layouts),
                        copy_bytes=donor_copy_bytes, input_columns=columns)
    result = dict(status='PASS bounded symbolic/address contract', protocol_sha256=sha(OUT / 'protocol.frozen.json'),
                  cases_checked=576, independently_materialized_global_layout_examples=9,
                  nine_source_choices_are_offline_alternatives_not_free_runtime_views=True,
                  coefficient_numeric_replay=False, native_output_ABI_certified=False,
                  baseline=baseline, resident_layout_ledgers=[dict(omit_rank=x['omit_rank'], **x['ledger']) for x in address_ledgers],
                  extra_copy_bytes_min=int(donor_copy_bytes.sum(1).min()), extra_copy_bytes_max=int(donor_copy_bytes.sum(1).max()),
                  extra_pages_each_layout=2048, extra_waves_each_layout=512, extra_page_wave_bytes=33554432,
                  per_expert_copy_rows_counts={str(n):sum(c['copy_rows']==n for c in cases) for n in (362,363)},
                  B_source_fringe='85 complete outputs plus512B real appended-row fragment; consume core82/83 only. No validR, rowstop, skipped MAC, extra context or partial-dot contribution.',
                  B_all_chips='1059 complete outputs,1024 useful.35 complete fringe rows dropped,11776B total partial-row tails read. Entire read pages and complete outputs charged.',
                  A='8jobs x128 complete outputs,12pages/3waves each. Whole1024 GU outputs gathered before host activation and full512-vector A8 privateDown.',
                  scope='No baseline fitting or performance conclusion. No weights, heldout or native replay. Virtual addresses allow static repacking.', seconds=time.perf_counter()-began)
    for item in protocol['inputs']:
        assert sha(item['path']) == item['sha256']
    dump(OUT / 'audit.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'resident_layout_ledgers'}, indent=2))


if __name__ == '__main__':
    main()
