#!/usr/bin/env python3
"""Read-only producer checks and attribution seal for the bounded M32 control."""
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/research-2026-11-02/round2/chip-count-regime-screen'
OUT = BASE / 'g4-M32-coupled-cp'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    with np.load(path) as z:
        return {key: z[key] for key in z.files}


def stats(values):
    return {'mean_ns': float(np.mean(values)), 'p99_ns': float(np.percentile(values, 99))}


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def main():
    target = OUT / 'producer-seal'
    target.mkdir(exist_ok=False)
    protocol = json.loads((OUT / 'protocol.frozen.json').read_text())
    for item in protocol['inputs']:
        assert sha(item['path']) == item['sha256'], item['path']
    fit = json.loads((OUT / 'fit_report.json').read_text())
    native = json.loads((OUT / 'native_report.json').read_text())
    native_protocol = json.loads((OUT / 'native_protocol.frozen.json').read_text())
    assert native_protocol['fit_report_sha256'] == sha(OUT / 'fit_report.json')
    assert native_protocol['candidates_sha256'] == sha(OUT / 'candidates.frozen.npz')
    candidates = load(OUT / 'candidates.frozen.npz')
    arrays = load(OUT / 'native_arrays.npz')
    stage = json.loads((BASE / 'stage-b/report.json').read_text())
    stage_arrays = load(BASE / 'stage-b/cost_arrays.npz')
    shared = load(BASE / 'stage-b/granite4tiny.M32.minimum_depth.plan.npz')['rows'][64]
    ids = [record['id'] for record in fit['candidates']]
    routes = candidates['routes']
    assert routes.shape == (512, 6)
    assert all(len(set(route)) == 6 for route in routes)
    checks = []
    for ident in ids:
        X, O, rows = (candidates[ident + '.' + k] for k in ('X', 'O', 'rows'))
        assert np.all(X.sum(1) == 5) and np.all(O.sum(1) == 7) and not np.any(X & O)
        assert np.all(np.sort(rows[:64, 0], axis=1) == [0] * 7 + [40] + [41] * 24)
        assert np.all(np.sort(rows[:64, 1], axis=1) == [0] * 20 + [128] * 12)
        assert np.array_equal(rows[:64, 0] == 0, O)
        assert np.array_equal(rows[:64, 1] > 0, X | O)
        assert np.array_equal(rows[64], shared)
        # Independently expand scalar command/input/read/page-fringe-output costs.
        table = np.zeros((65, 2, 32), dtype=np.int64)
        waves = np.zeros_like(table)
        pages = np.zeros_like(table)
        for e in range(65):
            for phase in range(2):
                columns = 1536 if phase == 0 else (1024 if e == 64 else 512)
                for chip in range(32):
                    count = int(rows[e, phase, chip])
                    if not count:
                        continue
                    page = (count * columns + 16383) // 16384
                    wave = (page + 3) // 4
                    emitted = page * 16384 // columns
                    table[e, phase, chip] = 250 + columns // 2 + 28000 * wave + 2 * emitted
                    pages[e, phase, chip], waves[e, phase, chip] = page, wave
        shared_work = table[64].sum(0)
        exact = table[routes].sum((1, 2)) + shared_work
        formula = 6 * 29102 + shared_work + 28762 * X[routes].sum(1) - 340 * O[routes].sum(1)
        assert np.array_equal(exact, formula)
        assert np.array_equal(exact.max(1), candidates[ident + '.work'])
        query_waves = waves[routes].sum((1, 2)) + waves[64].sum(0)
        resident_waves = waves.sum((0, 1))
        assert np.all(resident_waves <= 192)
        plan = load(OUT / 'native' / (ident + '.plan.npz'))
        assert np.array_equal(plan['rows'], rows)
        addresses = plan['addresses']
        occupied = set()
        for chip, block, wl, span, page, plane in addresses:
            assert block == 0 and plane == 0 and wl + span <= 192 and 0 < page <= span * 4
            for location in range(int(wl), int(wl + span)):
                key = int(chip), int(block), location
                assert key not in occupied
                occupied.add(key)
        checks.append(dict(id=ident, work=stats(exact.max(1)), peak_readwaves=int(query_waves.max()),
                           all_queries_peak_readwaves_10=bool(np.all(query_waves.max(1) == 10)),
                           resident_perchip_waves=resident_waves.tolist(),
                           resident_perchip_useful_bytes=(rows[:, 0].sum(0) * 1536 +
                                                         rows[:64, 1].sum(0) * 512 + rows[64, 1] * 1024).tolist(),
                           useful_bytes=int((rows[:, 0] * 1536).sum() + (rows[:64, 1] * 512).sum() + (rows[64, 1] * 1024).sum()),
                           programmed_pages=int(pages.sum()), reserved_waves=int(waves.sum()),
                           allocated_blocks=32, copies=0, source_addresses_nonoverlap=True))
        for priority in ('GF', 'DF'):
            record = next(x for x in native['records'] if x['id'] == ident and x['input_copies'] == 1 and x['priority'] == priority)
            assert stats(arrays[ident + '.' + priority + '.cost']) == record['cost']
    comparisons = []
    selected = native['final_selected_id']
    for copies in (1, 4):
        controls = [x for x in stage['records'] if x['model'] == 'granite4tiny' and x['chips'] == 32 and x['input_copies'] == copies]
        control = min(controls, key=lambda x: x['cost']['mean_ns'])
        proposed = min((x for x in native['records'] if x['input_copies'] == copies), key=lambda x: x['cost']['mean_ns'])
        key = (proposed['id'] + '.' if copies == 1 else 'selected_input4.') + proposed['priority'] + '.cost'
        proposed_cost = arrays[key]
        control_cost = stage_arrays[control['id'] + '.' + control['priority'] + '.cost']
        assert stats(proposed_cost) == proposed['cost']
        difference = control_cost - proposed_cost
        requests = difference.reshape(32, 16).mean(1)
        comparisons.append(dict(input_copies=copies, strongest_finite_control=control['id'] + '.' + control['priority'],
                                control=stats(control_cost), coupled=stats(proposed_cost),
                                coupled_id=proposed['id'], priority=proposed['priority'],
                                mean_reduction_pct=100 * (1 - proposed_cost.mean() / control_cost.mean()),
                                p99_reduction_pct=100 * (1 - np.percentile(proposed_cost, 99) / np.percentile(control_cost, 99)),
                                improved_queries=int((difference > 0).sum()), tied_queries=int((difference == 0).sum()),
                                regressed_queries=int((difference < 0).sum()), maximum_regression_ns=int(max(0, -difference.min())),
                                request_mean_gain_ns=requests.tolist(), requests_improved=int((requests > 0).sum())))
    report = dict(status='producer_checks_pass_external_independent_review_pending', utc=datetime.now(timezone.utc).isoformat(),
                  source_sha256=sha(__file__), protocol_sha256=sha(OUT / 'protocol.frozen.json'),
                  selected=selected, native_costs=native['native_costs'], maps=len(ids), all_rowmaps_distinct=len({candidates[x + '.rows'].tobytes() for x in ids}) == len(ids),
                  fit_stages=fit['stages'], candidate_checks=checks, comparisons=comparisons,
                  numerical_status='Peak10 feasibility OPTIMAL is a fixed-model feasibility certificate, not optimal latency. Mean FEASIBLE after90s, objective148200340/bound143360000; no global mean optimum.',
                  retention_note='Original peak-stage map was superseded by the bounded best-work pool; every seven retained mean map independently satisfies peak10 on all512queries. Canonical rows differ even for equal-work callbacks.',
                  attribution='Zero-replica standard coupled TRAIN coloring with unchanged GU/Down row multisets and shared geometry. Samehardware M32; all families GF/DF. Report uses strongest StageB finite control, not only weaker samegeometry source in native_report.',
                  limits='L0 TRAIN512 synthetic native service. No model inference/quality, heldout, full-layer claim, new algorithm, hardware measurement or proof that AiF has32PEs. FullGU/A8 barrier and nominal sameformat W8 retained. Input4 is sensitivity on the input1-selected map, no refit.',
                  fairness='Finite deterministic profiles and bounded standardCP have different search budgets. Gains demonstrate sparse-regime label-placement opportunity, not superiority over equally fitted unrestricted cyclic/free-mask methods.')
    dump(target / 'report.json', report)
    lines = ['M32 standard coupled-coloring control: producer seal', '',
             'G4 actual W8 TRAIN, L0, 32 chips / 8 channels (hardware scale sensitivity). No replicas; frozen shared rows and routed row multisets unchanged.', '',
             '| Input copies | Strong finite control mean/P99 (us) | Coupled mean/P99 (us) | Mean / P99 reduction |',
             '|---|---:|---:|---:|']
    for x in comparisons:
        lines.append(f"| {x['input_copies']} | {x['control']['mean_ns']/1000:.6f} / {x['control']['p99_ns']/1000:.6f} | {x['coupled']['mean_ns']/1000:.6f} / {x['coupled']['p99_ns']/1000:.6f} | {x['mean_reduction_pct']:.6f}% / {x['p99_reduction_pct']:.6f}% |")
    lines += ['', 'Peak10 feasibility succeeded; all seven retained mean maps independently have exactly ten maximum readwaves in every TRAIN query. Mean solver status is FEASIBLE, not OPTIMAL. Native input1 selected incumbent4/GF; input4 reuses that exact map.', '',
              'The primitive is ordinary zero-copy coloring, not a new algorithm. Search budgets differ from finite deterministic controls. The result motivates a fair matched-search control; it does not establish a publication claim, full-model benefit, or an implemented 32-PE AiF device.', '',
              'The original native report compares with the weaker minimum_depth source. This report corrects attribution to the strongest StageB finite candidate. No original artifact was changed.', '',
              'Producer verification: row-count coverage, shared identity, all512 chip-work coefficients, page/fringe billing, 32 nonoverlapping single-block inventories; saved q0 C++/Python trace equality from native runner. Independent review is separately owned by the architecture agent.', '']
    (target / 'report.md').write_text('\n'.join(lines))
    shutil.copy2(__file__, target / Path(__file__).name)
    files = [p for p in OUT.rglob('*') if p.is_file() and 'producer-seal' not in p.parts]
    files += [target / 'report.json', target / 'report.md', target / Path(__file__).name]
    dump(target / 'completion.json', dict(status='sealed_producer_result', utc=datetime.now(timezone.utc).isoformat(),
                                         files=[dict(path=str(p), sha256=sha(p), bytes=p.stat().st_size) for p in sorted(files)],
                                         note='Independent review is not included in this producer seal and may be added in a separate directory.'))
    print(json.dumps(dict(report=str(target / 'report.json'), completion_sha256=sha(target / 'completion.json'), comparisons=comparisons), indent=2))


if __name__ == '__main__':
    main()
