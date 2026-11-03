#!/usr/bin/env python3
"""Execute the frozen shaping plan and emit reviewable certificates/traces."""
import hashlib
import json
from pathlib import Path
import random
import shaping as sh

DEST = Path(__file__).resolve().parent
PLAN = DEST / 'plan.frozen.json'
COPY_ONE = [0, 100, 320, 321, 500, 1000, 2000, 3000, 4000, 5000]
COPY_TWO = [500, 1000, 2000, 3000, 4000, 5000]


def compact(out):
    d = sh.summarize(out)
    d.pop('dummy_addresses')
    return d


def main():
    plan_hash = hashlib.sha256(PLAN.read_bytes()).hexdigest()
    records, best, controls = [], [], []
    for added, copies in ((1, COPY_ONE), (2, COPY_TWO)):
        for c in copies:
            feasible = []
            for splits in sh.specs(added):
                out = sh.evaluate(splits, cache_copy_ns=c)
                record = dict(added_waves=added, copy_ns=c, **compact(out))
                records.append(record)
                if record['feasible']:
                    feasible.append(record)
            chosen = min(feasible, key=lambda r: (r['makespan_ns'], r['splits'])) if feasible else None
            best.append(dict(added_waves=added, copy_ns=c, feasible_configurations=len(feasible),
                             tested=36 if added == 1 else 594, best=chosen))
            print('sweep', added, c, 'feasible', len(feasible), 'best',
                  chosen['makespan_ns'] if chosen else None, chosen['splits'] if chosen else None, flush=True)
    for c in COPY_ONE:
        for policy in ('strict_no_interval', 'cold_reset'):
            r = sh.v1.simulate(sh.v1.build_g4_candidate(), cache_copy_ns=c, issue_policy=policy)
            controls.append(dict(layout='candidate12', issue_policy=policy, copy_ns=c, **sh.v1._summary(r)))
        for label, ids in (('packed_adjacent12', list(range(6))),
                           ('packed_spaced15', [0, 3, 6, 9, 12, 15])):
            base = sh.v1.build_ordinary_packed(ids)
            r = sh.v1.simulate(sh.v1.with_keep_hot(base), cache_copy_ns=c,
                               unselected_policy='drop_before_ecc_conditional')
            controls.append(dict(layout=label + '_keep_hot_drop', copy_ns=c, **sh.v1._summary(r)))
            # Adjacent packed12 receives exactly the same one-split shaping
            # search and real packed-placement dummy address allocation.
            if len(base) == 12:
                feasible = []
                for splits in sh.specs(1):
                    waves = sh.shape_existing(base, splits, ids, 'packed')
                    certificate = sh.certify(waves, ids, expected=base)
                    r = sh.v1.simulate(waves, cache_copy_ns=c,
                                       unselected_policy='drop_before_ecc_conditional')
                    if r['feasible']:
                        feasible.append(dict(layout='packed_adjacent13_shaped', copy_ns=c,
                                             splits=[list(x) for x in splits], **sh.v1._summary(r)))
                controls.append(min(feasible, key=lambda r: (r['makespan_ns'], r['splits'])) if feasible else
                                dict(layout='packed_adjacent13_shaped', copy_ns=c, feasible=False, makespan_ns=None))
    # Public selector addresses are replayed on varied true route IDs, proving
    # that the structural plan does not borrow a selected expert as a dummy.
    rng = random.Random(1122026)
    route_checks = []
    for ri in range(12):
        ids = rng.sample(range(64), 6)
        out = sh.select_shaped_candidate(500, ids, max_added_waves=1)
        assert out['feasible'] and out['result']['makespan_ns'] == 155140
        route_checks.append(dict(expert_ids=ids, splits=out['splits'],
                                 makespan_ns=out['result']['makespan_ns'], certificate=out['certificate']))
    selected = sh.select_shaped_candidate(500, max_added_waves=1)
    assert selected['result']['physical_read_pages'] == 52 and selected['result']['ecc_pages'] == 48
    assert selected['result']['cold_waves'] == 1 and selected['result']['cr_waves'] == 12
    original = sh.v1.simulate(sh.v1.build_g4_candidate(), cache_copy_ns=500)
    assert not original['feasible']
    (DEST / 'selected_copy500.events.json').write_text(json.dumps(selected, indent=2) + '\n')
    for c in (1000, 2000, 3000):
        out = sh.select_shaped_candidate(c, max_added_waves=2)
        (DEST / f'selected_copy{c}.events.json').write_text(json.dumps(out, indent=2) + '\n')
    report = dict(status='PASS', frozen_plan_sha256=plan_hash, sealed_v1_sha256=sh.SEALED_SHA,
                  shaping_api_sha256=hashlib.sha256(Path(sh.__file__).read_bytes()).hexdigest(),
                  driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  sweep=records, best=best, controls=controls, route_checks=route_checks,
                  priority_copy500=compact(selected),
                  wavecount_certificate=dict(contract='48 useful pages with at most4useful pages/wave, cache-only whole-page read/cache model, strict no-interval CR',
                                             twelve_wave_impossibility='all12waves must be full useful waves; 2*500+(12-2)*540=6400>6040ns',
                                             thirteen_wave_feasibility='selected finite-latch event schedule passes',
                                             optimal_wavecount=13,
                                             caveat='Only within this conditional contract; not a new general queue theorem or unconditional hardware certificate.'),
                  interpretation=['52physicalpage reads versus48useful/ECC pages is8.333%overfetch; no stored-weight replication.',
                                  'Dummy pages are real unselected resident weights, with unique physical addresses in block0.',
                                  'Every wave reads all4planes, so no plane is idled and no CR bias retention across a gap is assumed.',
                                  'Useful consume order is unchanged, including owner switches only at complete rows.',
                                  'Drop-before-ECC and programmable plane-source order remain conditional controls shared with baselines.',
                                  'Packed-adjacent receives the same shaping search and can tie this pipeline envelope.',
                                  'Ends at whole-page ECC completion; source-switch gaps, MAC throughput, power/read-disturb effects and output buffering are not measured.'])
    (DEST / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    lines = ['# CR-preserving overfetch/drop shaping', '',
             'A13wave schedule is feasible at500ns copy under the stated two-weight-latch, cache-only, evaluation-start overwrite model. It uses one28us cold start and twelve uninterrupted9.7us CR waves. Missing planes read real unselected resident expert pages and conditionally drop them before ECC.', '',
             '**Priority result:** split original wave10 into1+3useful pages. Useful/ECC pages remain48; physical reads become52 (+8.333%). The finite-latch event schedule completes at155.140us, versus170.140us for reactive cold reset. Original12waves fail the6400>6040ns necessary deadline bound. Thus13waves are minimal within this contract.', '',
             'No new stored weights, weight FIFO, read/cache mux, idle CR bias hold, or extra partial-row contexts are introduced. Useful page order is identical. Drop-before-ECC and programmable source order are explicit conditional capabilities.', '',
             '| Copy ns | Best13waves us | Feasible splits/36 | Best14waves us | Feasible splits/594 |',
             '|---:|---:|---:|---:|---:|']
    for c in COPY_ONE:
        one = next(x for x in best if x['added_waves'] == 1 and x['copy_ns'] == c)
        two = next((x for x in best if x['added_waves'] == 2 and x['copy_ns'] == c), None)
        def val(x):
            return f"{x['best']['makespan_ns'] / 1000:.3f}" if x and x['best'] else 'FAIL' if x else 'not swept'
        lines.append(f"| {c} | {val(one)} | {one['feasible_configurations']} | {val(two)} | {two['feasible_configurations'] if two else '—'} |")
    lines += ['', 'Controls use the same drop capability. Adjacent ordinary packed12 can receive the same shaping transformation and match this envelope; no stencil-exclusive benefit is claimed. Spaced packed15 keep-hot/drop is also reported. Twelve varied route-address instantiations validate that dummies always belong to unselected experts.', '',
              'The public API is shaping.py: build_shaped_candidate(expert_ids, splits), select_shaped_candidate(cache_copy_ns, expert_ids, max_added_waves), evaluate. Canonical structural selection is cached; each actual route is rebuilt and address-certified. Results are an addendum; sealed v1 and stream-burst outputs are unchanged.', '',
              'This is conditional timing feasibility ending at ECC completion. Native support for drop/source control, source-switch overhead, MAC throughput, power, read disturb and output buffering require separate evidence.']
    (DEST / 'report.md').write_text('\n'.join(lines) + '\n')
    assert hashlib.sha256(PLAN.read_bytes()).hexdigest() == plan_hash
    assert hashlib.sha256(sh.SOURCE.read_bytes()).hexdigest() == sh.SEALED_SHA


if __name__ == '__main__':
    main()
