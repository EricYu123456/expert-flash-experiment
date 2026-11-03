#!/usr/bin/env python3
"""Seal the stronger independent-phase cyclic control without modifying fits."""
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/research-2026-11-02/round2/chip-count-regime-screen'
OUT = BASE / 'g4-M32-independent-rotation-cp'
COMMON = BASE / 'g4-M32-cyclic-rotation-cp'
FREE = BASE / 'g4-M32-coupled-cp'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(p):
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


def stats(x):
    return dict(mean_ns=float(np.mean(x)), p99_ns=float(np.percentile(x, 99)))


def dump(p, x):
    Path(p).write_text(json.dumps(x, indent=2) + '\n')


def main():
    target = OUT / 'producer-seal'
    target.mkdir(exist_ok=False)
    p = json.loads((OUT / 'protocol.frozen.json').read_text())
    for item in p['inputs']:
        assert sha(item['path']) == item['sha256']
    z, c, a = load(OUT / 'inputs.frozen.npz'), load(OUT / 'candidates.frozen.npz'), load(OUT / 'native_arrays.npz')
    fit, native = (json.loads((OUT / name).read_text()) for name in ('fit_report.json', 'native_report.json'))
    np_ = json.loads((OUT / 'native_protocol.frozen.json').read_text())
    assert np_['fit_report_sha256'] == sha(OUT / 'fit_report.json')
    assert np_['candidates_sha256'] == sha(OUT / 'candidates.frozen.npz')
    checks = []
    for record in fit['candidates']:
        ident = record['id']
        rows, labels = c[ident + '.rows'], c[ident + '.labels']
        assert labels.shape == (64, 2) and np.all((labels >= 0) & (labels < 32))
        assert np.array_equal(rows[:64, 0], z['gu_rotations'][labels[:, 0]])
        assert np.array_equal(rows[:64, 1], z['down_rotations'][labels[:, 1]])
        assert np.array_equal(rows[64], z['shared'])
        assert np.all(np.sort(rows[:64, 0], axis=1) == [0] * 7 + [40] + [41] * 24)
        assert np.all(np.sort(rows[:64, 1], axis=1) == [0] * 20 + [128] * 12)
        table = np.zeros((65, 2, 32), np.int64)
        waves = np.zeros_like(table)
        for e in range(65):
            for phase in range(2):
                columns = 1536 if phase == 0 else (1024 if e == 64 else 512)
                for chip in range(32):
                    count = int(rows[e, phase, chip])
                    if count:
                        pages = (count * columns + 16383) // 16384
                        wave = (pages + 3) // 4
                        emitted = pages * 16384 // columns
                        table[e, phase, chip] = 250 + columns // 2 + wave * 28000 + 2 * emitted
                        waves[e, phase, chip] = wave
        loads = table[c['routes']].sum((1, 2)) + table[64].sum(0)
        G, D = rows[:64, 0] > 0, rows[:64, 1] > 0
        formula = 29102 * G[c['routes']].sum(1) + 28762 * D[c['routes']].sum(1) + c['sharedwork']
        assert np.array_equal(loads, formula)
        assert np.array_equal(loads.max(1), c[ident + '.work'])
        query_waves = waves[c['routes']].sum((1, 2)) + waves[64].sum(0)
        unique, count = np.unique(query_waves.max(1), return_counts=True)
        plan = load(OUT / 'native' / (ident + '.plan.npz'))
        assert np.array_equal(plan['rows'], rows)
        occupied = set()
        for chip, block, wl, span, pages, plane in plan['addresses']:
            assert block == plane == 0 and wl + span <= 192
            for wl_ in range(int(wl), int(wl + span)):
                key = int(chip), wl_
                assert key not in occupied
                occupied.add(key)
        for priority in ('GF', 'DF'):
            actual = next(x for x in native['records'] if x['id'] == ident and x['priority'] == priority)
            assert stats(a[ident + '.' + priority + '.cost']) == actual['cost']
        overlaps = (G & D).sum(1)
        checks.append(dict(id=ident, exact_independent_orbits_with40row=True, onecopy=True,
                           shared_unchanged=True, addresses_singleblock_nonoverlap=True,
                           work=stats(loads.max(1)), overlap_histogram={str(int(k)): int(v) for k, v in zip(*np.unique(overlaps, return_counts=True))},
                           peak_readwave_histogram={str(int(k)): int(v) for k, v in zip(unique, count)},
                           perchip_resident_waves=waves.sum((0, 1)).tolist(),
                           perchip_useful_bytes=(rows[:, 0].sum(0) * 1536 + rows[:64, 1].sum(0) * 512 + rows[64, 1] * 1024).tolist()))
    common_report = json.loads((COMMON / 'native_report.json').read_text())
    common = common_report['best_cyclic']
    common_arrays = load(COMMON / 'native_arrays.npz')
    original = load(BASE / 'stage-b/cost_arrays.npz')
    for priority in ('GF', 'DF'):
        assert np.array_equal(a['source.' + priority + '.cost'], original['granite4tiny.M32.minimum_depth.' + priority + '.cost'])
        assert np.array_equal(a['common_hint.' + priority + '.cost'], common_arrays[common['id'] + '.' + priority + '.cost'])
    best = native['best_independent']
    independent_cost = a[best['id'] + '.' + best['priority'] + '.cost']
    free = json.loads((FREE / 'native_report.json').read_text())['best_coupled_input1']
    free_cost = load(FREE / 'native_arrays.npz')[free['id'] + '.' + free['priority'] + '.cost']
    difference = independent_cost - free_cost
    comparison = dict(independent=best, common=common, free=free,
                      free_mean_reduction_pct=100 * (1 - free_cost.mean() / independent_cost.mean()),
                      free_p99_reduction_pct=100 * (1 - np.percentile(free_cost, 99) / np.percentile(independent_cost, 99)),
                      free_improved_queries=int((difference > 0).sum()), tied_queries=int((difference == 0).sum()),
                      free_regressed_queries=int((difference < 0).sum()), max_free_regression_ns=int(max(0, -difference.min())),
                      perrequest_mean_free_gain_ns=difference.reshape(32, 16).mean(1).tolist(),
                      independent_best_tail_record=min(native['records'], key=lambda x: x['cost']['p99_ns']))
    result = dict(status='producer_checks_pass_independent_audit_separate', utc=datetime.now(timezone.utc).isoformat(),
                  protocol_sha256=sha(OUT / 'protocol.frozen.json'), source_sha256=sha(__file__),
                  fit_stages=fit['stages'], maps=checks, native_costs=native['native_costs'], comparison=comparison,
                  attribution='AllstandardfixedprofileTRAINlabelcontrols. Originalsource/commonhint retained; no performance loss attributedto failedsearch. Freegeometrynotnew: hereallphase rowmultisets andshared identical, onlyallowedlabelmaps differ.',
                  limits='Equal30+90s/workers8/seed butlargermodel/convergence differs. UNKNOWNnotUNSAT; FEASIBLEnotglobaloptimum. L0onlyTRAIN512 andnominal32PEhardware sensitivity, not device/quality/end-to-end/heldout evidence. No input4refit.',
                  validation='Producer elementary page/fringe service andonecopy/address checks, all512anchorbridges, savedq0C++/Python traces. Independent reviewer separately verifies final artifacts.')
    dump(target / 'report.json', result)
    text = ['Independent-phase cyclic strong control, G4 M32 L0 TRAIN512', '',
            '| Learned label family | Mean (us) | P99 (us) |', '|---|---:|---:|']
    for label, record in (('common cyclic', common), ('independent-phase cyclic with common retained', best), ('free coupled masks', free)):
        text.append(f"| {label} | {record['cost']['mean_ns']/1000:.6f} | {record['cost']['p99_ns']/1000:.6f} |")
    text += ['', f"Free relative to the strongest retained independent control: mean {comparison['free_mean_reduction_pct']:.6f}%, P99 {comparison['free_p99_reduction_pct']:.6f}%. All maps use one copy, identical GU/Down row multisets, and the same shared rows.", '',
             'Search is bounded and not globally optimal. Larger independent-phase space may converge less well under the same solver time; both old anchors are retained. UNKNOWN feasibility is not a negative proof. No heldout, all-layer or implemented32PE AiF claim.', '']
    (target / 'report.md').write_text('\n'.join(text))
    shutil.copy2(__file__, target / Path(__file__).name)
    files = [x for x in OUT.rglob('*') if x.is_file() and 'producer-seal' not in x.parts]
    files += [target / 'report.json', target / 'report.md', target / Path(__file__).name]
    dump(target / 'completion.json', dict(status='sealed_producer', utc=datetime.now(timezone.utc).isoformat(),
                                          files=[dict(path=str(x), sha256=sha(x), bytes=x.stat().st_size) for x in sorted(files)],
                                          independent_review='separate directory; not claimed inproducerseal'))
    print(json.dumps(dict(completion=str(target / 'completion.json'), sha256=sha(target / 'completion.json'), comparison=comparison), indent=2))


if __name__ == '__main__':
    main()
