#!/usr/bin/env python3
"""Independent integer/row-address audit of the sealed elastic GU work screen.

Does not import the producer, occupancy_table, native runner, or oracle library.
No fitting, scheduling, external inputs, or numeric model evaluation is run.
"""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/research-2026-11-02/round2/g4-elastic-prefix-screen/work-v1'
AUDIT = OUT / 'independent'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def stats(x):
    return dict(mean_ns=float(np.mean(x)), p99_ns=float(np.percentile(x, 99)))


def main():
    AUDIT.mkdir(exist_ok=False)
    p = json.loads((OUT / 'protocol.frozen.json').read_text())
    r = json.loads((OUT / 'work_report.json').read_text())
    for item in p['inputs']:
        assert sha(item['path']) == item['sha256']
        if 'snapshot' in item:
            assert sha(item['snapshot']) == item['sha256']
    assert sha(OUT / 'protocol.frozen.json') == r['protocol_sha256']
    a = np.load(OUT / 'work_arrays.npz')
    b = np.load(p['baseline_path'])
    rows, routes = a['rows_B'], a['routes']
    assert np.array_equal(rows, b['rows']) and np.array_equal(routes, b['routes'])
    assert rows.shape == (65, 2, 16) and routes.shape == (512, 6)
    assert np.all(np.sort(rows[:64, 0], axis=1) == [40]*7+[82]*3+[83]*6)
    assert all(len(set(q)) == 6 for q in routes)
    assert np.all((routes >= 0) & (routes < 64))
    service = p['service']
    assert service['command_ns'] == 250 and service['output_fringe'] == 1
    assert service['input_channel_copies'] == 1

    # Derive every chip's occupation time from jobs, without producer helpers.
    occupancy = np.zeros((65, 2, 16), np.int64)
    counters = np.zeros((65, 2, 5), np.int64)  # jobs/pages/waves/inputB/outputB
    for e in range(65):
        for phase in range(2):
            band_rows = np.zeros(16, np.int64)
            for chip, nr, nc, pages, waves in b['jobs'][slice(*b['offsets'][e, phase])]:
                chip, nr, nc, pages, waves = map(int, (chip, nr, nc, pages, waves))
                assert pages == (nr*nc+16383)//16384
                assert waves == (pages+3)//4 and waves <= 192
                assert nc == b['input_columns'][e, phase]
                emitted = pages*16384//nc
                inp = (nc+1)//2
                out = emitted*2
                occupancy[e, phase, chip] += 250+inp+waves*28000+out
                counters[e, phase] += [1, pages, waves, nc, emitted*4]
                band_rows[chip] += nr
            assert np.array_equal(band_rows, rows[e, phase])
    base = occupancy[routes].sum(axis=(1, 2)) + occupancy[64].sum(axis=0)
    assert np.array_equal(base, a['base'])
    assert np.array_equal(base.max(1), a['baseline_work'])
    assert float(base.mean()) == 527059.25

    # Nine offline A geometries; no query may choose a different omitted rank.
    high = [np.flatnonzero(rows[e, 0] > 40) for e in range(64)]
    manual_delta = np.zeros((64, 9, 16), np.int64)
    for e in range(64):
        for j in range(9):
            aa = np.zeros(16, np.int64)
            aa[np.delete(high[e], j)] = 128
            assert np.array_equal(aa, a['A_menu'][e, j])
            assert aa.sum() == 1024 and np.count_nonzero(aa) == 8
            # 128x1536 -> 12 pages/3 waves/128 emitted; full input on each job.
            work = np.where(aa > 0, 250+768+3*28000+256, 0)
            manual_delta[e, j] = work-occupancy[e, 0]
    assert np.array_equal(manual_delta, a['delta_menu'])
    assert set(np.unique(manual_delta)) == {-57188, -29102, 28086}
    assert np.all(manual_delta.sum(2) == -36214)
    assert 527059.25+6*(-36214)/16 == 513479.0
    choice = a['selected_layout']
    assert choice.shape == (64,) and np.all((choice >= 0) & (choice < 9))
    delta = manual_delta[np.arange(64), choice]
    assert np.array_equal(delta, a['selected_delta'])
    bits = ((np.arange(64)[:, None] >> np.arange(6)) & 1).astype(np.int64)
    loads = base[:, None, :] + np.einsum('mk,qkc->qmc', bits, delta[routes])
    maxwork = loads.max(2)
    assert np.array_equal(maxwork.min(1), a['selected_values'])
    assert np.array_equal(maxwork.argmin(1), a['selected_masks'])
    plain = 2000+maxwork.min(1)
    # All chips have shared GU, so earliest first-acquisition ranks are 0..15.
    assert np.all(occupancy[64, 0] > 0)
    sorted_lb = 2000+(np.sort(loads, axis=2)[:, :, ::-1]+np.arange(16)*250).max(2)
    sorted_lb = sorted_lb.min(1)
    assert np.array_equal(plain, a['plain_host_LB'])
    assert np.array_equal(sorted_lb, a['sorted_firststart_LB'])
    assert np.all(sorted_lb >= plain)

    # Check each retained static incumbent, independent of the search trace.
    for name in ['same_layout', 'all9']:
        ch = a[name+'_static_choices']
        ll = base.copy()
        for e in range(64):
            if ch[e] >= 0:
                if name == 'same_layout':
                    assert ch[e] == choice[e]
                ll[(routes == e).any(1)] += manual_delta[e, ch[e]]
        assert np.array_equal(ll.max(1), a[name+'_static_values'])

    # Exact logical identity: B's original row vector is each resident prefix;
    # A's eight long prefixes contain each original row once, including G/U IDs.
    ledger = json.loads((OUT / 'copy_rowid_ledger.json').read_text())
    copyrows = 0
    for e, entry in enumerate(ledger):
        original = {int(c): rr for c, rr in entry['B_logical_rows'].items()}
        image = {int(c): rr for c, rr in entry['resident_prefix_image'].items()}
        av = {int(c): rr for c, rr in entry['A_logical_rows'].items()}
        active = set(map(int, np.delete(high[e], choice[e])))
        assert set(av) == active and entry['excluded_high_rank'] == choice[e]
        donors = [rr for c in range(16) if c not in active for rr in original[c]]
        assert donors == entry['copied_logical_rows'] and len(donors) in (362, 363)
        assert sorted(rr for vv in original.values() for rr in vv) == list(range(1024))
        assert sorted(rr for vv in av.values() for rr in vv) == list(range(1024))
        for c in range(16):
            assert len(original[c]) == rows[e, 0, c]
            assert image[c][:len(original[c])] == original[c]
            assert len(image[c]) == a['resident_rows'][e, 0, c]
            if c in active:
                assert len(image[c]) == 128 and av[c] == image[c]
            else:
                assert image[c] == original[c]
        multiplicity = Counter(rr for vv in image.values() for rr in vv)
        assert multiplicity == Counter({rr: 2 if rr in donors else 1 for rr in range(1024)})
        copyrows += len(donors)
    assert copyrows == r['budget']['copied_rows'] == 23202

    # Check independent physical intervals and view prefixes, including all fixed
    # Down/shared jobs. This is a virtual address certificate, not device I/O.
    bins = defaultdict(list)
    perchip = [dict(chip=c, useful_bytes=0, pages=0, resident_waves=0, blocks=0) for c in range(16)]
    for job, addr in zip(a['resident_jobs'], a['resident_addresses']):
        c, nr, nc, pages, waves = map(int, job)
        ac, block, wl, reserve, apages, plane = map(int, addr)
        assert ac == c and pages == apages and waves == reserve and plane == 0
        assert pages == (nr*nc+16383)//16384 and waves == (pages+3)//4
        assert wl >= 0 and wl+waves <= 192
        bins[c, block].append((wl, wl+waves))
        perchip[c]['useful_bytes'] += nr*nc
        perchip[c]['pages'] += pages
        perchip[c]['resident_waves'] += waves
        perchip[c]['blocks'] = max(perchip[c]['blocks'], block+1)
    for intervals in bins.values():
        intervals.sort()
        assert all(x[1] <= y[0] for x, y in zip(intervals, intervals[1:]))
    views = json.loads((OUT / 'physical_prefix_views.json').read_text())
    assert len(views) == 1024
    for v in views:
        e, c = v['expert'], v['chip']
        ac, block, wl, reserve, pages, plane = v['address']
        assert ac == c and plane == 0
        assert v['B_rows'] == rows[e, 0, c]
        assert v['A_rows'] == a['A_menu'][e, choice[e], c]
        for phase in ['A', 'B']:
            nr = v[phase+'_rows']
            expected_pages = (nr*1536+16383)//16384
            expected_waves = (expected_pages+3)//4
            assert v[phase+'_pages'] == expected_pages <= pages
            assert v[phase+'_waves'] == expected_waves <= reserve
            assert wl+expected_waves <= 192
    assert sum(x['pages']*16384 for x in perchip) == 193462272
    assert sum(x['resident_waves'] for x in perchip) == 2952
    assert [x['blocks'] for x in perchip] == r['budget']['resident_physical']['chip_blocks']

    bn = a['baseline_native']
    metrics = dict(baseline_native=stats(bn), baseline_maxchip_work=stats(base.max(1)),
                   selected_maxchip_work=stats(maxwork.min(1)), host_plus_work_LB=stats(plain),
                   optimal_first_start_LB=stats(sorted_lb))
    ceilings = {}
    for label, lb in [('host_plus_work', plain), ('optimal_first_start', sorted_lb)]:
        ceilings[label] = dict(mean_improvement_upper_pct=100*(1-lb.mean()/bn.mean()),
                              p99_improvement_upper_pct=100*(1-np.percentile(lb, 99)/np.percentile(bn, 99)))
    assert ceilings['host_plus_work']['mean_improvement_upper_pct'] < 2
    assert ceilings['host_plus_work']['p99_improvement_upper_pct'] < 5
    payload = dict(status='PASS_bounded_no_go', source_sha256=sha(__file__),
                   producer_protocol_sha256=sha(OUT/'protocol.frozen.json'),
                   metrics=metrics, ceilings=ceilings, all_masks_checked=32768,
                   delta_regression=dict(per_expert_aggregate_ns=-36214, pooled_B_ns=527059.25,
                                         pooled_all_A_ns=513479.0),
                   runtime_choices='Exactly two views per expert, one offline chosen A plus B; not all nine As.',
                   copied_rows=copyrows, copied_coefficient_bytes=copyrows*1536,
                   extra_pages=2048, extra_waves=512, extra_programmed_bytes=33554432,
                   perchip_resident=perchip,
                   physical_capacity_scope='These are per-layer GU copies; full-layer ledgers also include unchanged Down/shared. No full-model extrapolation or average-chip capacity constraint.',
                   full_backup_service_equivalence='A full separate A image plus a full separate B image has the identical selected row/column/page/wave/native-job descriptors for each mask. The current timing model is address-invariant within a block; hence the same selector/scatter gives the same work and native DAG. Prefix sharing changes residency, not service. This is structural equivalence, not an executed device test.',
                   limits=['Only finite layouts/coordinate search were screened; no optimum over all offline copy choices is claimed.',
                           'Chip ownership work omits queue/dependency gaps, PCIe conflicts and selector costs; the report is a necessary lower bound, never achieved latency.',
                           'Neither native replay nor big whole-GU replica controls are warranted by the failed necessary-work gate.',
                           'Synthetic ordinary-read model and page-fringe output contract; real AiF output/scatter/numeric ABI is not newly validated.',
                           'P99 inequality follows pointwise T>=LB using the same empirical quantile definition. No tail generalization beyond TRAIN512.'],
                   gate=dict(mean_threshold_pct=2, p99_threshold_pct=5, passed=False))
    (AUDIT/'audit.json').write_text(json.dumps(payload, indent=2)+'\n')
    report = f'''The bounded G4 L0 TRAIN512 elastic-prefix screen is sealed as a negative result. Runtime has exactly A/B for each expert; the nine omitted-high ranks are offline candidates. Independent integer calculations reproduced all 32,768 masks, row-ID coverage, short/long prefix views, physical interval nonoverlap, and the -36,214 ns aggregate per-expert A-minus-B regression. No native replay, heldout, or extra replica campaign was run.

| Quantity | Mean (ms) | P99 (ms) |
|---|---:|---:|
| Existing B native | {bn.mean()/1e6:.9f} | {np.percentile(bn,99)/1e6:.9f} |
| Selected min maxchip work | {maxwork.min(1).mean()/1e6:.9f} | {np.percentile(maxwork.min(1),99)/1e6:.9f} |
| Host + minimum work LB | {plain.mean()/1e6:.9f} | {np.percentile(plain,99)/1e6:.9f} |
| Optimal first-start assignment LB | {sorted_lb.mean()/1e6:.9f} | {np.percentile(sorted_lb,99)/1e6:.9f} |

Even assuming zero selector cost and no further queue/dependency gaps, the plain LB permits at most {ceilings['host_plus_work']['mean_improvement_upper_pct']:.6f}% mean / {ceilings['host_plus_work']['p99_improvement_upper_pct']:.6f}% P99 improvement relative to B. The stronger valid first-start LB reduces those ceilings to {ceilings['optimal_first_start']['mean_improvement_upper_pct']:.6f}% / {ceilings['optimal_first_start']['p99_improvement_upper_pct']:.6f}%. Both miss the 2% mean or 5% tail expansion gate. Static same-layout A/B and stronger B-plus-all-nine-A coordinate controls retain B. These finite-search observations are not a universal no-go theorem for all copy assignments.

The selected image copies {copyrows} complete rows = {copyrows*1536} useful coefficient bytes ({copyrows*1536/1048576:.6f} MiB), but uses only 512 extra resident waves / 2,048 pages = 32 MiB by consuming old padding. This is one layer, not all forty. Full-layer per-chip physical allocation grows from sixteen to twenty LSB blocks (192 waves/block), with the exact per-chip inventory in audit.json. Same-resident-budget whole-GU alternatives can hold 21 minimum-24-wave copies plus eight waves, or 22 copies if only useful coefficient bytes are compared; those are distinct budgets and were not used as weakened controls. Whole A+B duplication is service-equivalent to prefix sharing under the declared address-invariant native model; prefix sharing saves capacity, not a separate latency mechanism.

B prefix reads still drain all 85/42 page-fringe rows per high/low job, including appended values in discarded slots. A's eight jobs each drain 128 rows. Stored original row IDs and host scatter preserve all complete GU rows; Down, shared, full-input A8 barrier, and single-context behavior are unchanged. This is a synthetic necessary-work screen, not measured hardware or model quality evidence.
'''
    (AUDIT/'report.md').write_text(report)
    files = [OUT/'protocol.frozen.json', OUT/'work_report.json', OUT/'work_arrays.npz',
             OUT/'copy_rowid_ledger.json', OUT/'physical_prefix_views.json',
             AUDIT/'audit.json', AUDIT/'report.md', Path(__file__)]
    completion = dict(status='sealed_bounded_no_go', files=[dict(path=str(x), sha256=sha(x)) for x in files])
    (AUDIT/'completion.json').write_text(json.dumps(completion, indent=2)+'\n')
    print(json.dumps(dict(status=payload['status'], ceilings=ceilings,
                         completion_sha256=sha(AUDIT/'completion.json')), indent=2))


if __name__ == '__main__':
    main()
