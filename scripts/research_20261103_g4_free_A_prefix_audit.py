#!/usr/bin/env python3
"""Independent integer/coverage/address checker for sealed free-A work-v2.

No producer/backend/oracle imports. No placement search or native replay.
"""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'results/research-2026-11-02/round2'
OUT = BASE/'g4-elastic-prefix-screen/work-v2-free-A'
AUDIT = OUT/'independent'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(x):
    return dict(mean_ns=float(np.mean(x)), p99_ns=float(np.percentile(x, 99)))


def main():
    AUDIT.mkdir(exist_ok=False)
    p = json.loads((OUT/'protocol.frozen.json').read_text())
    r = json.loads((OUT/'work_report.json').read_text())
    assert sha(OUT/'protocol.frozen.json') == r['protocol_sha256']
    for item in p['inputs']:
        assert sha(item['path']) == item['sha256']
        if 'snapshot' in item:
            assert sha(item['snapshot']) == item['sha256']
    a = np.load(OUT/'work_arrays.npz')
    b = np.load(p['baseline_path'])
    rows, routes = a['rows_B'], a['routes']
    assert np.array_equal(rows, b['rows']) and np.array_equal(routes, b['routes'])
    assert rows.shape == (65, 2, 16) and routes.shape == (512, 6)
    assert all(len(set(q)) == 6 for q in routes)
    s = p['service']
    assert s['command_ns'] == 250 and s['input_channel_copies'] == 1
    assert s['output_fringe'] == 1 and s['host_submit_ns'] == 2000
    table = np.zeros((65, 2, 16), np.int64)
    baseline_inventory = np.zeros((16, 3), np.int64)  # usefulB, pages, waves
    for e in range(65):
        for phase in range(2):
            covered = np.zeros(16, np.int64)
            for c, nr, nc, pages, waves in b['jobs'][slice(*b['offsets'][e, phase])]:
                c, nr, nc, pages, waves = map(int, (c, nr, nc, pages, waves))
                assert pages == (nr*nc+16383)//16384 and waves == (pages+3)//4
                assert nc == b['input_columns'][e, phase]
                emitted = pages*16384//nc
                table[e, phase, c] += 250+(nc+1)//2+waves*28000+2*emitted
                covered[c] += nr
                baseline_inventory[c] += [nr*nc, pages, waves]
            assert np.array_equal(covered, rows[e, phase])
    base = table[routes].sum((1, 2))+table[64].sum(0)
    assert np.array_equal(base, a['base'])
    assert np.array_equal(table[:64, 0], a['B_GU_work'])
    assert base.mean() == 527059.25
    Bgu = table[:64, 0]
    bits = ((np.arange(64)[:, None] >> np.arange(6)) & 1).astype(np.int64)

    def enumerate_mask(active):
        assert active.shape == (64, 16) and np.all(active.sum(1) == 8)
        delta = active*85274-Bgu
        loads = base[:, None, :]+np.einsum('mk,qkc->qmc', bits, delta[routes])
        maxwork = loads.max(2)
        return loads, maxwork.min(1), maxwork.argmin(1)

    checks = 0
    for i, active in enumerate(a['initial_source_masks']):
        loads, v, masks = enumerate_mask(active)
        assert np.array_equal(v, a['initial_values'][i])
        if i < 32:
            assert np.array_equal(active, a['routeblind_source_masks'][i])
            assert np.array_equal(v, a['routeblind_values'][i])
            h = r['routeblind'][i]['h']
            assert np.all((active & (rows[:64, 0] > 40)).sum(1) == h)
        checks += 32768
    initial_mean = a['initial_values'].mean(1)
    assert r['start_indices'] == sorted(range(36), key=lambda j: (initial_mean[j], j))[:2]
    for i, active in enumerate(a['final_source_masks']):
        ll, vv, mm = enumerate_mask(active)
        assert np.array_equal(vv, a['final_values'][i])
        assert np.array_equal(mm, a['final_runtime_masks'][i])
        assert vv.mean() <= initial_mean[r['start_indices'][i]]
        assert r['final_fits'][i]['proposals'] == 8192
        previous = r['final_fits'][i]['initial']['mean_ns']
        for step in r['final_fits'][i]['history']:
            assert step['work']['mean_ns'] <= previous
            previous = step['work']['mean_ns']
        checks += 32768
    selected = int(np.argmin(a['final_values'].mean(1)))
    assert selected == r['selected_final_index']
    active = a['selected_source_mask']
    assert np.array_equal(active, a['final_source_masks'][selected])
    loads, values, masks = enumerate_mask(active)
    assert np.array_equal(values, a['selected_values'])
    assert np.array_equal(masks, a['selected_masks'])
    assert np.array_equal(active*85274-Bgu, a['selected_delta'])
    plain = values+2000
    assert np.array_equal(plain, a['plain_host_LB'])
    assert np.all(table[64, 0] > 0)
    ranked = 2000+(np.sort(loads, axis=2)[:, :, ::-1]+np.arange(16)*250).max(2)
    sortedlb = ranked.min(1)
    assert np.array_equal(sortedlb, a['sorted_firststart_LB'])
    assert np.array_equal(ranked.argmin(1), a['sorted_firststart_masks'])
    assert np.all(sortedlb >= plain)
    assert np.all(sortedlb <= a['baseline_native'])
    relaxed = np.load(BASE/'g4-elastic-prefix-screen/free-span-bound/arrays.npz')
    assert np.array_equal(relaxed['routes'], routes)
    assert np.all(relaxed['best_LB'] <= plain)

    # Independently verify maximum primary coactivation matching (no secondary
    # ties used here), and each pair's complementary minimum-copy placement.
    coact = np.zeros((64, 64), np.int64)
    for query in routes:
        for i in range(6):
            for j in range(i+1, 6):
                e, f = sorted(map(int, (query[i], query[j])))
                coact[e, f] += 1
    graph = nx.Graph()
    for e in range(64):
        for f in range(e+1, 64):
            graph.add_edge(e, f, weight=int(coact[e, f]))
    optimal_matching = nx.max_weight_matching(graph, maxcardinality=True)
    primary = sum(coact[min(e, f), max(e, f)] for e, f in optimal_matching)
    assert primary == 685
    for entry in r['TRAIN_matching_initializations']:
        initial = a['initial_source_masks'][entry['index']]
        pairs = entry['pairs']
        assert len(pairs) == 32 and sorted(e for pair in pairs for e in pair) == list(range(64))
        assert sum(coact[e, f] for e, f in pairs) == entry['co_selected_pairs'] == primary
        for e, f in pairs:
            assert np.array_equal(initial[e], ~initial[f])
            diff = rows[e, 0]-rows[f, 0]
            assert np.sum(diff[initial[e]]) == np.sort(diff)[-8:].sum()

    # Static controls have no replica: one fixed GU row geometry per expert.
    static_metrics = {}
    for name in ['same_layout_static', 'free_static']:
        ms = []
        for start in range(2):
            aa = a[name+'_source_masks'][start]
            isA = a[name+'_isA'][start]
            assert np.all(aa.sum(1) == 8)
            if name == 'same_layout_static':
                assert np.array_equal(aa, active)
            ll = base.copy()
            for e in range(64):
                if isA[e]:
                    ll[(routes == e).any(1)] += aa[e]*85274-Bgu[e]
            assert np.array_equal(ll.max(1), a[name+'_values'][start])
            ms.append(stats(ll.max(1)))
        static_metrics[name] = ms

    # Stored originals are unique B prefixes. Copied donor rows fill each A128
    # prefix; for every view exactly the original complete logical GU is read.
    ledger = json.loads((OUT/'copy_rowid_ledger.json').read_text())
    copyrows = 0
    for e, entry in enumerate(ledger):
        original = {int(c): rr for c, rr in entry['B_original_rows'].items()}
        image = {int(c): rr for c, rr in entry['resident_prefix_image'].items()}
        av = {int(c): rr for c, rr in entry['A_rows'].items()}
        assert set(av) == set(np.flatnonzero(active[e]))
        assert sorted(rr for band in original.values() for rr in band) == list(range(1024))
        assert sorted(rr for band in av.values() for rr in band) == list(range(1024))
        donors = [rr for c in range(16) if not active[e, c] for rr in original[c]]
        assert donors == entry['copied_logical_rows']
        assert len(donors) == 1024-np.sum(rows[e, 0, active[e]]) == entry['copied_rows']
        for c in range(16):
            assert len(original[c]) == rows[e, 0, c]
            assert image[c][:len(original[c])] == original[c]
            assert len(image[c]) == a['resident_rows'][e, 0, c]
            if active[e, c]:
                assert av[c] == image[c] and len(image[c]) == 128
            else:
                assert image[c] == original[c]
        assert Counter(rr for band in image.values() for rr in band) == Counter({rr: 2 if rr in donors else 1 for rr in range(1024)})
        copyrows += len(donors)
    assert copyrows == r['budget']['copied_rows'] == 28015
    expected_waves = int(np.sum(active*(3-(rows[:64, 0]*1536+65535)//65536)))
    assert expected_waves == r['budget']['extra_waves'] == 625
    assert np.array_equal(a['resident_rows'][:, 1], rows[:, 1])
    assert np.array_equal(a['resident_rows'][64], rows[64])
    intervals = defaultdict(list)
    inventory = np.zeros((16, 4), np.int64)  # useful, pages, waves, blocks
    for job, addr in zip(a['resident_jobs'], a['resident_addresses']):
        c, nr, nc, pages, waves = map(int, job)
        ac, block, wl, reserve, apages, plane = map(int, addr)
        assert ac == c and reserve == waves and apages == pages and plane == 0
        assert pages == (nr*nc+16383)//16384 and waves == (pages+3)//4
        assert 0 <= wl and wl+waves <= 192
        intervals[c, block].append((wl, wl+waves))
        inventory[c, :3] += [nr*nc, pages, waves]
        inventory[c, 3] = max(inventory[c, 3], block+1)
    for bands in intervals.values():
        bands.sort()
        assert all(x[1] <= y[0] for x, y in zip(bands, bands[1:]))
    assert int(inventory[:, 0].sum()-baseline_inventory[:, 0].sum()) == copyrows*1536
    assert int(inventory[:, 1].sum()-baseline_inventory[:, 1].sum()) == 4*expected_waves
    assert int(inventory[:, 2].sum()-baseline_inventory[:, 2].sum()) == expected_waves
    assert np.all((baseline_inventory[:, 2] > 0) & (baseline_inventory[:, 2] <= 192))
    assert inventory[:, 3].tolist() == r['budget']['resident_physical']['chip_blocks']
    views = json.loads((OUT/'physical_prefix_views.json').read_text())
    assert len(views) == 1024
    for view in views:
        e, c = view['expert'], view['chip']
        ac, block, wl, waves, pages, plane = view['address']
        assert ac == c and plane == 0
        assert view['B_rows'] == rows[e, 0, c]
        assert view['A_rows'] == (128 if active[e, c] else 0)
        for viewname in ['A', 'B']:
            nr = view[viewname+'_rows']
            assert view[viewname+'_pages'] == (nr*1536+16383)//16384 <= pages
            assert view[viewname+'_waves'] == (nr*1536+65535)//65536 <= waves
            assert wl+view[viewname+'_waves'] <= 192

    bn = a['baseline_native']
    ceiling = {}
    for label, lb in [('host_plus_work', plain), ('optimal_firststart', sortedlb)]:
        ceiling[label] = dict(LB=stats(lb), mean_pct=100*(1-lb.mean()/bn.mean()),
                             p99_pct=100*(1-np.percentile(lb, 99)/np.percentile(bn, 99)))
    final_gates = []
    for vv in a['final_values']:
        lb = 2000+vv
        final_gates.append(dict(mean_pct=100*(1-lb.mean()/bn.mean()),
                                p99_pct=100*(1-np.percentile(lb, 99)/np.percentile(bn, 99))))
    assert all(g['mean_pct'] < 2 and g['p99_pct'] < 5 for g in final_gates)
    assert not r['gate_passed']
    audit = dict(status='PASS_bounded_no_go', source_sha256=sha(__file__),
                 producer_protocol_sha256=sha(OUT/'protocol.frozen.json'),
                 independently_enumerated_query_masks=checks,
                 matching_primary_optimum=int(primary), static_controls=static_metrics,
                 baseline_native=stats(bn), selected_maxchip_work=stats(values),
                 ceilings=ceiling, all_trained_endpoints=final_gates,
                 copied_rows=copyrows, copied_coefficient_bytes=copyrows*1536,
                 extra_waves=expected_waves, extra_pages=4*expected_waves,
                 extra_programmed_bytes=expected_waves*65536,
                 perchip_inventory_columns=['coefficient_bytes', 'pages', 'waves', 'LSB_blocks'],
                 baseline_perchip_inventory=baseline_inventory.tolist(),
                 resident_perchip_inventory=inventory.tolist(),
                 replica_views='Each expert has one fixed A8 source set and B. 32routeblind+4TRAINmatching are offline alternatives, not free runtime views.',
                 attribution='Adaptive geometry is the possible latency mechanism. Identical A/B geometry stored as two full images has the same descriptors/work/DAG; prefix sharing only reduces capacity.',
                 limitations=['Finite two-start/two-pass local search, not global offline-copy optimum.',
                              'Variable copy budgets; not same-budget comparison with v1 or any copy competitor.',
                              'Work/firststart are necessary lower bounds, not achieved latency; no native replay, selector timings, full-copy/partial-copy controls or heldout.',
                              'Output page-fringe/native ABI and performance are the declared synthetic model, not newly measured device capability.',
                              'All copy budgets are per layer, not the full 40-layer model.'],
                 gate=dict(mean_threshold_pct=2, p99_threshold_pct=5, passed=False))
    (AUDIT/'audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    report = f'''The broader fixed-A8 placement work screen is sealed as a bounded negative result. Thirty-two routeblind layouts (h=2/4/6/8 high sources, eight seeds each) and four TRAIN-derived maximum-coactivation perfect-matching layouts were evaluated. Matching covers 685 co-selected pairs across TRAIN512; each paired initial A geometry is complementary and minimizes copied core rows for that complementary choice. The two best starts were matching layouts, followed by two coordinate passes and 8,192 source swaps each.

| Quantity | Mean (ms) | Empirical P99 (ms) |
|---|---:|---:|
| Frozen B native incumbent | {bn.mean()/1e6:.9f} | {np.percentile(bn,99)/1e6:.9f} |
| Selected exact min maxchip work | {values.mean()/1e6:.9f} | {np.percentile(values,99)/1e6:.9f} |
| Host + work necessary LB | {plain.mean()/1e6:.9f} | {np.percentile(plain,99)/1e6:.9f} |
| Optimal first-start assignment LB | {sortedlb.mean()/1e6:.9f} | {np.percentile(sortedlb,99)/1e6:.9f} |

The selected layout permits at most {ceiling['host_plus_work']['mean_pct']:.6f}% mean / {ceiling['host_plus_work']['p99_pct']:.6f}% P99 native improvement, even with zero selection overhead and no additional scheduling stalls. The stronger first-start bound reduces those ceilings to {ceiling['optimal_firststart']['mean_pct']:.6f}% / {ceiling['optimal_firststart']['p99_pct']:.6f}%. The other trained endpoint permits 1.386813% / 0.970757%. Both fail the 2% mean or 5% tail expansion gate. Same-layout static A/B and broader zero-copy B/free-A local-search controls retain the original all-B incumbent; additional static search is explicitly reported in work_report.json. No native/selector/full-copy/partial-copy/heldout campaign was run.

The independent checker reproduced {checks:,} query-mask values across all 36 initial layouts and both trained endpoints, both final output mask tie choices, static final costs, original and copied row identities, prefix coverage and physical interval nonoverlap. Each expert retains exactly one A plus B at runtime. Nine/sixteen/all possible A geometries are never supplied free to the query.

This selected layout copies {copyrows:,} complete rows = {copyrows*1536:,} useful coefficient bytes ({copyrows*1536/1048576:.6f} MiB) per layer. It needs {expected_waves} additional resident waves / {expected_waves*4} pages = {expected_waves/16:.6f} MiB programmed storage. Whole-layer L0 address allocation, including unchanged Down/shared, grows from 16 to {int(inventory[:,3].sum())} LSB blocks. Per-chip inventory is explicit in audit.json; the fit imposed no fixed-average chip-capacity budget. This variable budget cannot be called equal to v1's 512-wave budget. It could buy at most 27 minimum-work full-GU images under only a useful-byte bound or 26 under the page/wave bound, leaving one wave; no feasible whole-copy placement/control is claimed from those volume counts.

The unrestricted per-query free-span relaxation retains its 3.011906% mean / 6.164532% P99 possible ceilings. It removes fixed address and replica-capacity constraints and is not an implemented layout. The finite heuristic result here does not prove all offline A source placements fail, nor does prefix sharing itself add speed relative to storing identical A/B views separately. Complete-row reads, page-fringe drains/drop, full H input, single native context, unchanged shared/Down and full-GU A8 dependency remain the declared contract.
'''
    (AUDIT/'report.md').write_text(report)
    files = [OUT/'protocol.frozen.json', OUT/'work_report.json', OUT/'work_arrays.npz',
             OUT/'copy_rowid_ledger.json', OUT/'physical_prefix_views.json',
             AUDIT/'audit.json', AUDIT/'report.md', Path(__file__)]
    completion = dict(status='sealed_bounded_no_go', files=[dict(path=str(path), sha256=sha(path)) for path in files])
    (AUDIT/'completion.json').write_text(json.dumps(completion, indent=2)+'\n')
    print(json.dumps(dict(status=audit['status'], ceilings=ceiling,
                         completion_sha256=sha(AUDIT/'completion.json')), indent=2))


if __name__ == '__main__':
    main()
