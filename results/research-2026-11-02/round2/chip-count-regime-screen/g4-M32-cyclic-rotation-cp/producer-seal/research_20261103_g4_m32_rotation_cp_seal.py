#!/usr/bin/env python3
"""Producer integrity/attribution seal for the matched cyclic CP control."""
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/research-2026-11-02/round2/chip-count-regime-screen'
OUT = BASE / 'g4-M32-cyclic-rotation-cp'
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
    z, c = load(OUT / 'inputs.frozen.npz'), load(OUT / 'candidates.frozen.npz')
    costs = load(OUT / 'native_arrays.npz')
    native = json.loads((OUT / 'native_report.json').read_text())
    fit = json.loads((OUT / 'fit_report.json').read_text())
    np_ = json.loads((OUT / 'native_protocol.frozen.json').read_text())
    assert np_['fit_report_sha256'] == sha(OUT / 'fit_report.json')
    assert np_['candidates_sha256'] == sha(OUT / 'candidates.frozen.npz')
    ids = [x['id'] for x in fit['candidates']]
    checks = []
    for ident in ids:
        labels, rows = c[ident + '.labels'], c[ident + '.rows']
        assert labels.shape == (64,) and np.all((0 <= labels) & (labels < 32))
        assert np.array_equal(rows[:64], z['rotations'][labels])
        assert np.array_equal(rows[64], z['shared'])
        assert np.all(rows[:64, 0].sum(1) == 1024) and np.all(rows[:64, 1].sum(1) == 1536)
        X, O = z['X'][labels], z['O'][labels]
        work = 6 * 29102 + c['sharedwork'] + 28762 * X[c['routes']].sum(1) - 340 * O[c['routes']].sum(1)
        assert np.array_equal(work.max(1), c[ident + '.work'])
        waves = 6 + c['sharedwaves'] + X[c['routes']].sum(1)
        unique, count = np.unique(waves.max(1), return_counts=True)
        plan = load(OUT / 'native' / (ident + '.plan.npz'))
        assert np.array_equal(plan['rows'], rows)
        occupied = set()
        for chip, block, wl, span, pages, first_plane in plan['addresses']:
            assert block == first_plane == 0 and wl + span <= 192
            for address in range(int(wl), int(wl + span)):
                key = int(chip), address
                assert key not in occupied
                occupied.add(key)
        for priority in ('GF', 'DF'):
            record = next(x for x in native['records'] if x['id'] == ident and x['priority'] == priority)
            assert stats(costs[ident + '.' + priority + '.cost']) == record['cost']
        checks.append(dict(id=ident, exact_orbit_with40row=True, onecopy_coverage=True,
                           unchanged_shared=True, address_nonoverlap_singleblock=True,
                           work=stats(work.max(1)), peak_readwaves_histogram={str(int(k)): int(v) for k, v in zip(unique, count)},
                           label_histogram=np.bincount(labels, minlength=32).tolist()))
    source = load(BASE / 'stage-b/cost_arrays.npz')
    for priority in ('GF', 'DF'):
        assert np.array_equal(costs['source.' + priority + '.cost'], source['granite4tiny.M32.minimum_depth.' + priority + '.cost'])
    best = native['best_cyclic']
    cyclic_cost = costs[best['id'] + '.' + best['priority'] + '.cost']
    f = json.loads((FREE / 'native_report.json').read_text())['best_coupled_input1']
    free_cost = load(FREE / 'native_arrays.npz')[f['id'] + '.' + f['priority'] + '.cost']
    diff = cyclic_cost - free_cost
    comparison = dict(cyclic=best, free=f, mean_reduction_pct=100 * (1 - free_cost.mean() / cyclic_cost.mean()),
                      p99_reduction_pct=100 * (1 - np.percentile(free_cost, 99) / np.percentile(cyclic_cost, 99)),
                      improved_queries=int((diff > 0).sum()), tied_queries=int((diff == 0).sum()),
                      regressed_queries=int((diff < 0).sum()), max_regression_ns=int(max(0, -diff.min())),
                      perrequest_mean_gain_ns=diff.reshape(32, 16).mean(1).tolist())
    report = dict(status='producer_checks_pass_independent_review_separate', utc=datetime.now(timezone.utc).isoformat(),
                  source_sha256=sha(__file__), fit_stages=fit['stages'], checks=checks, comparison=comparison,
                  same_budget='Bothfree/cyclic use peak30s+mean90s,8workers,seed20261114. Free4096Bool vs cyclic2048onehot; modelcomplexity and heuristic convergence differ. Bothnative GF/DF512 offered equally.',
                  attribution='Original~9% versus unlearnedsource is largely explained by standardlearnedcycliclabels. Remaining free advantage is0.621774%mean/8.514275%P99 onTRAIN; not a newgeometry oralgorithm claim.',
                  limits='Nopeak10 witness found bycyclic30s => UNKNOWN, not infeasibility. MeanFEASIBLE, notoptimal. No claimbestcyclicorbit orfree optimum. Noheldout/alllayers/newweights/32PEimplementation. Input4 wasnotfitted/replayed forthiscycliccontrol.',
                  next='Do notattribute remaining gap to free geometry before a stronger phase-independent cyclic/joint native control. Only root can authorize furtherbounded work.')
    dump(target / 'report.json', report)
    (target / 'report.md').write_text(
        'Matched learned cyclic control, G4 M32 L0 TRAIN512\n\n'
        f"Cyclic best GF: mean {cyclic_cost.mean()/1000:.6f} us, P99 {np.percentile(cyclic_cost,99)/1000:.6f} us.\n\n"
        f"Free-mask GF: mean {free_cost.mean()/1000:.6f} us, P99 {np.percentile(free_cost,99)/1000:.6f} us; relative reductions {comparison['mean_reduction_pct']:.6f}% / {comparison['p99_reduction_pct']:.6f}%.\n\n"
        'Both use the same solver-time budget and fixed shared rows; every expert remains one copy of exactly the same row multisets. Cyclic has one common rotation for both phases, with the 40-row GU slot rotating exactly. Source64 maps lie in that orbit and both512-source GF/DF cost vectors match StageB.\n\n'
        'Cyclic peak10 feasibility is UNKNOWN, mean is FEASIBLE. Free peak10 on all512queries is a saved feasible witness, not proof that cyclic cannot achieve it. This is a finite TRAIN comparison of standard controls, not a new geometry or algorithm claim. No heldout or all-layer evaluation.\n')
    shutil.copy2(__file__, target / Path(__file__).name)
    files = [x for x in OUT.rglob('*') if x.is_file() and 'producer-seal' not in x.parts]
    files += [target / 'report.json', target / 'report.md', target / Path(__file__).name]
    dump(target / 'completion.json', dict(status='sealed_producer', utc=datetime.now(timezone.utc).isoformat(),
                                          files=[dict(path=str(x), sha256=sha(x), bytes=x.stat().st_size) for x in sorted(files)],
                                          independent_review='separate independent-m32-rotation-final directory, not claimed bythisseal'))
    print(json.dumps(dict(completion=str(target / 'completion.json'), sha256=sha(target / 'completion.json'), comparison=comparison), indent=2))


if __name__ == '__main__':
    main()
