#!/usr/bin/env python3
"""Replay frozen Deep physical addresses without importing the fitter."""
from pathlib import Path
from collections import Counter
import gzip, hashlib, json
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = next(p for p in OUT.parents if (p / 'scripts/plane_wave_model.py').exists())
TRAIN = ROOT / 'results/research-2026-11-02/round2/w8-format-training/deepseek/decode.routes.npz'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')

def main():
    protocol = json.loads((OUT / 'protocol.frozen.json').read_text())
    assert sha(TRAIN) == protocol['TRAIN']['sha256']
    assert sha(OUT / 'deep_block_headroom.py') == protocol['source_sha256']
    with np.load(TRAIN) as z:
        routes = z['routes'].transpose(2, 0, 1, 3).reshape(26, 512, 6)
        assert z['completed'].all() and z['split'].item() == 'train'
    with gzip.open(OUT / 'actual_page_address_ledgers.json.gz', 'rt') as f:
        ledgers = json.load(f)
    archived = np.load(OUT / 'TRAIN_maps_and_metrics.frozen.npz')
    report = json.loads((OUT / 'report.json').read_text())
    totals = Counter()
    by_control = {}
    for record in ledgers:
        name, layer = record['control'], record['layer']
        a = np.asarray(record['addresses'], dtype=np.int64)
        blocks = 2 if record['matched_two_blocks'] else 3
        assert a.shape == (1452, 5)
        assert np.unique(a[:, [3, 2, 4]], axis=0).shape[0] == 1452
        assert (a[:, 4] >= 0).all() and (a[:, 4] < 192).all()
        assert (a[:, 2] >= 0).all() and (a[:, 2] < 4).all()
        assert (a[:, 3] >= 0).all() and (a[:, 3] < blocks).all()
        observed = np.zeros((65, blocks, 4), dtype=np.int64)
        for owner in range(65):
            local_pages = a[a[:, 0] == owner, 1]
            expected_pages = 44 if owner == 64 else 22
            assert np.array_equal(np.sort(local_pages), np.arange(expected_pages))
            # C=2048 W8: every physical page covers eight complete, unique rows.
            row_ids = (8 * local_pages[:, None] + np.arange(8)).reshape(-1)
            assert np.array_equal(np.sort(row_ids), np.arange(expected_pages * 8))
        np.add.at(observed, (a[:, 0], a[:, 3], a[:, 2]), 1)
        assert np.array_equal(observed, archived[name + '.counts'][layer])
        assert (observed.sum(axis=0) <= 192).all()
        r = routes[layer]
        assert np.all(np.diff(np.sort(r, axis=1), axis=1) > 0)
        requested = np.zeros((512, 65), dtype=bool)
        requested[np.arange(512)[:, None], r] = True
        requested[:, 64] = True
        # Count each concrete requested physical address, independently of fitted counts.
        per_bin = np.zeros((512, blocks, 4), dtype=np.int64)
        for b in range(blocks):
            for p in range(4):
                owners = a[(a[:, 3] == b) & (a[:, 2] == p), 0]
                per_bin[:, b, p] = requested[:, owners].sum(axis=1)
        assert np.all(per_bin.sum(axis=(1, 2)) == 176)
        waves_per_block = per_bin.max(axis=2)
        waves = waves_per_block.sum(axis=1)
        active_blocks = (waves_per_block > 0).sum(axis=1)
        nand_ns = waves * 9700 + active_blocks * 18300
        global_plane = per_bin.sum(axis=1).max(axis=1)
        integer_block = ((per_bin.sum(axis=2) + 3) // 4).sum(axis=1)
        independent_plane = (per_bin.sum(axis=1) * 9700 + (per_bin > 0).sum(axis=1) * 18300).max(axis=1)
        necessary_resource = np.maximum(nand_ns, 28000 + 176 * 16384 * 10 // 64)
        rebuilt = np.column_stack((waves, active_blocks, nand_ns, global_plane,
                                   integer_block, independent_plane, necessary_resource))
        assert np.array_equal(rebuilt, archived[name + '.metrics'][layer])
        # The kth actual-WL-sorted requested page in each plane forms wave k.
        # Its active set is exactly {p: count[p] > k}; these sets only shrink.
        # Therefore all 512 routes have a constructive same-block schedule,
        # without an idle plane silently resuming CR hot in that block.
        for q in range(512):
            emitted = []
            for b in range(blocks):
                queues = []
                for p in range(4):
                    rows = a[(a[:, 3] == b) & (a[:, 2] == p) & requested[q, a[:, 0]]]
                    rows = rows[np.argsort(rows[:, 4], kind='stable')]
                    assert len(rows) == per_bin[q, b, p]
                    queues.append(rows)
                previous = {p for p in range(4) if len(queues[p])}
                for k in range(int(waves_per_block[q, b])):
                    active = {p for p in range(4) if len(queues[p]) > k}
                    assert active and active <= previous
                    for p in sorted(active):
                        emitted.append(tuple(queues[p][k]))
                    previous = active
            assert len(emitted) == 176 and len(set(emitted)) == 176
            assert Counter(page[0] for page in emitted) == Counter({**{int(e): 22 for e in r[q]}, 64: 44})
        totals.update(plans=1, physical_addresses=1452, route_schedules=512,
                      selected_page_events=512 * 176)
        by_control.setdefault(name, []).append(rebuilt)
    paired = {}
    base = np.asarray(by_control['flat_global_RR'])
    for name, rows in by_control.items():
        v = np.asarray(rows)
        delta = v[:, :, 2] - base[:, :, 2]
        paired[name] = dict(
            mean_waves=float(v[:, :, 0].mean()),
            exact_44_queries=int((v[:, :, 0] == 44).sum()),
            NAND_faster_than_flat_RR=int((delta < 0).sum()),
            NAND_equal_to_flat_RR=int((delta == 0).sum()),
            NAND_slower_than_flat_RR=int((delta > 0).sum()),
            mean_NAND_saved_fraction=float((base[:, :, 2].mean() - v[:, :, 2].mean()) / base[:, :, 2].mean()),
            mean_wave_gap_to_global_44=float(v[:, :, 0].mean() - 44),
            mean_wave_gap_to_actual_block_integer_LB=float((v[:, :, 0] - v[:, :, 4]).mean()),
            mean_wave_gap_to_global_plane_LB=float((v[:, :, 0] - v[:, :, 3]).mean()))
    result = dict(status='PASS', source_sha256=sha(__file__),
                  source_fitter_sha256=sha(OUT / 'deep_block_headroom.py'),
                  TRAIN_sha256=sha(TRAIN), checks=dict(totals), controls=paired,
                  fixed_source_order='0123; physical page contains eight complete rows',
                  scope='Independent address and synchronous same-block NAND geometry replay. No finite-PB/native timing or fullFFN proof.')
    write('independent_audit.json', result)
    print(json.dumps(dict(status=result['status'], **dict(totals))), flush=True)

if __name__ == '__main__':
    main()
