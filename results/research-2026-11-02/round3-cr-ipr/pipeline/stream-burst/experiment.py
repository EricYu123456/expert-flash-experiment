#!/usr/bin/env python3
"""Follow-up only: finite-bank full-wave chain bound and explicit cold cuts.

Does not change the sealed v1 simulator or its outputs. The planned-cold
reference below is independent of v1's scheduling implementation.
"""
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys

DEST = Path(__file__).resolve().parent
ROOT = DEST.parents[4]
SOURCE = ROOT / 'scripts/research_20261102_cr_ipr_pipeline.py'
SEALED_SHA = 'ad35d08e17b4458cb5a0f54732aafce338d0526f5888fb02fcfdcc507ad99905'
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SEALED_SHA
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location('sealed_pipeline', SOURCE)
v1 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = v1
spec.loader.exec_module(v1)

S, R, U, COLD, DELTA = 2560, 9700, 6040, 28000, 540
COPIES = [0, 100, 320, 321, 500, 1000, 2000]


def synthetic(n, order='fixed'):
    rng = random.Random(20261102)
    out = []
    for wi in range(n):
        planes = list(range(4))
        if order == 'alternate_reverse' and wi % 2:
            planes.reverse()
        if order == 'random':
            rng.shuffle(planes)
        pages = [dict(page_id=f'x.p{4 * wi + p}', owner='x', page_index=4 * wi + p,
                      plane=p, block=0, wl=wi, bytes=16384) for p in range(4)]
        out.append(dict(pages=pages, source_order=planes))
    return out


def necessary(n, c):
    if n == 1:
        return True
    if n == 2:
        return c <= U
    return c <= U and 2 * c + (n - 2) * DELTA <= U


def planned(waves, c, cuts=(), trace=False):
    """Independent causal per-page recurrence, explicit cold at supplied cuts.

    Every cold start waits for the old read latches to clear, then performs a
    fresh28us operation. It never extends a9.7us CR across an idle interval.
    Input latches are retained, and the same GEMV command/PE state is assumed
    to continue through the explicit cold-voltage restart. This control is
    conditional; there is no hardware/API proof of its trigger mechanism.
    """
    cache_free, read_free = [0] * 4, [0] * 4
    ecc = array_end = 0
    violations, events, cold_indices = [], [], []
    copy_count = ecc_count = 0
    cuts = set(cuts)
    for wi, w in enumerate(waves):
        assert len(w['pages']) == 4 and {p['plane'] for p in w['pages']} == set(range(4))
        cold = wi == 0 or wi in cuts
        start = max(array_end, max(read_free)) if cold else array_end
        end = start + (COLD if cold else R)
        overwrite = start if cold else start + U
        if cold:
            cold_indices.append(wi)
        for plane in range(4):
            if read_free[plane] > overwrite:
                violations.append(dict(wave=wi, plane=plane, release_ns=read_free[plane],
                                       overwrite_ns=overwrite, excess_ns=read_free[plane] - overwrite))
        if trace:
            events.append(dict(stage='array_cold' if cold else 'array_cr', wave=wi,
                               start_ns=start, end_ns=end, overwrite_ns=overwrite))
        for plane in w['source_order']:
            page = next(p for p in w['pages'] if p['plane'] == plane)
            copy_start = max(end, cache_free[plane])
            copy_end = copy_start + c
            begin = max(ecc, copy_end)
            done = begin + S
            if trace:
                common = dict(wave=wi, plane=plane, page_id=page['page_id'])
                events.append(dict(stage='cache_copy', start_ns=copy_start, end_ns=copy_end, **common))
                events.append(dict(stage='ecc', start_ns=begin, end_ns=done, **common))
            read_free[plane], cache_free[plane], ecc = copy_end, done, done
            copy_count += 1
            ecc_count += 1
        array_end = end
    return dict(feasible=not violations, makespan_ns=max(array_end, ecc) if not violations else None,
                hypothetical_makespan_ns=max(array_end, ecc), array_end_ns=array_end,
                cold_indices=cold_indices, cuts=sorted(cuts), violations=violations,
                conservation=dict(read_pages=4 * len(waves), copied_pages=copy_count, ecc_pages=ecc_count),
                events=events)


def all_row_boundaries(waves):
    counts = Counter()
    valid = []
    for wi, w in enumerate(waves[:-1]):
        for p in w['pages']:
            counts[p['owner']] += p['bytes']
        if all(n % 1536 == 0 for n in counts.values()):
            valid.append(wi + 1)
    return valid


def best_cuts(waves, c, allowed):
    best = None
    for bits in range(1 << len(allowed)):
        cuts = [x for j, x in enumerate(allowed) if bits & (1 << j)]
        out = planned(waves, c, cuts)
        if out['feasible'] and (best is None or (out['makespan_ns'], len(cuts), cuts) <
                               (best['makespan_ns'], len(best['cuts']), best['cuts'])):
            best = out
    assert best is not None
    return best


def main():
    chain = []
    for c in COPIES:
        for order in ('fixed', 'alternate_reverse', 'random'):
            for n in range(1, 65):
                waves = synthetic(n, order)
                out = v1.simulate(waves, cache_copy_ns=c, validate_row_stream=False)
                independent = planned(waves, c)
                assert out['timing_feasible'] == independent['feasible']
                assert out['hypothetical_makespan_ns'] == independent['hypothetical_makespan_ns']
                bound = necessary(n, c)
                assert not out['timing_feasible'] or bound
                if order == 'fixed':
                    assert out['timing_feasible'] == bound
                chain.append(dict(copy_ns=c, source_order=order, waves=n,
                                  necessary_bound_satisfied=bound,
                                  bound_lhs_ns=2 * c + (n - 2) * DELTA if n >= 3 else None,
                                  feasible=out['timing_feasible'], makespan_ns=out['makespan_ns'],
                                  hypothetical_makespan_ns=out['hypothetical_makespan_ns'],
                                  violation_count=len(out['violations']),
                                  first_violation=out['violations'][0] if out['violations'] else None))
    maxima = [dict(copy_ns=c, **{order: max(x['waves'] for x in chain if x['copy_ns'] == c and
                                          x['source_order'] == order and x['feasible'])
                               for order in ('fixed', 'alternate_reverse', 'random')}) for c in COPIES]
    candidate = v1.build_g4_candidate()
    boundaries = all_row_boundaries(candidate)
    assert boundaries == [3, 6, 9]
    planned_records = []
    for c in COPIES:
        reactive = v1.simulate(candidate, cache_copy_ns=c, issue_policy='cold_reset')
        cold_indices = [x['wave'] for x in reactive['array_waves'] if x['kind'] == 'cold' and x['wave']]
        ref = planned(candidate, c, cold_indices)
        assert ref['feasible'] and ref['makespan_ns'] == reactive['makespan_ns']
        best = best_cuts(candidate, c, list(range(1, 12)))
        row_best = best_cuts(candidate, c, boundaries)
        six_six = planned(candidate, c, [6])
        strict = planned(candidate, c)
        planned_records.append(dict(copy_ns=c,
                                    strict={k: v for k, v in strict.items() if k != 'events'},
                                    reactive_cold_reset={k: v for k, v in ref.items() if k != 'events'},
                                    best_any_cut={k: v for k, v in best.items() if k != 'events'},
                                    best_full_row_cut={k: v for k, v in row_best.items() if k != 'events'},
                                    fixed_six_six_cut={k: v for k, v in six_six.items() if k != 'events'},
                                    any_cut_improvement_ns=ref['makespan_ns'] - best['makespan_ns'],
                                    full_row_improvement_ns=ref['makespan_ns'] - row_best['makespan_ns']))
    # Boundary tests of the derived bound independent of the coarse copy grid.
    six_pass = planned(candidate[:6], 1940)
    six_fail = planned(candidate[:6], 1941)
    assert six_pass['feasible'] and not six_fail['feasible']
    for c in (0, 100, 500):
        expect = {0: 13, 100: 12, 500: 11}[c]
        assert next(x for x in maxima if x['copy_ns'] == c)['fixed'] == expect
    report = dict(status='PASS', sealed_v1_sha256=SEALED_SHA,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  model='whole-page cache-only two-weight-bank synchronous full-wave envelope',
                  constants_ns=dict(page_ECC=S, CR=R, recycle=U, cold=COLD, backlog_per_full_wave=DELTA),
                  derivation=dict(necessary_for_N_at_least_3='2*c + (N-2)*(4*s-r) <= u, with initial c <= u',
                                  N1='no subsequent overwrite deadline', N2='c <= u',
                                  explanation='After N-2 complete prior waves, whichever plane was serviced last by ECC has a cache-free time no earlier than first_ready+c+(N-2)*4s. Its new read page must copy for another c before the following synchronous overwrite at first_ready+(N-2)*r+u. Source permutation cannot remove this last-plane necessary bound.',
                                  sufficiency='Fixed0123 full-wave order attains the bound in the tested range. Source permutations can create extra copy/ECC bubbles, so the necessary bound is not a blanket sufficiency claim for every order.',
                                  scope='This is a bound on the stated whole-page cache-only model, not a contradiction of AiF hardware or a model that treats reported effective6.4GB/s as a separate array read charge.'),
                  chain_cases=chain, chain_max_full_waves=maxima,
                  G4_full_row_boundaries=boundaries, G4_planned_cold=planned_records,
                  six_wave_copy_boundary_ns=dict(pass_ns=1940, fail_ns=1941),
                  limitations=['Explicit in-command cold-voltage restart requires a controller capability not established by this experiment.',
                               'All-copy cuts retain existing GEMV input latches and PE row state across the cold wait; full-row cuts avoid an interrupted partial-row accumulator.',
                               'Cold restart waits for old read latches before starting conservative28us cold operation; no retained CR bias over a wait is assumed.',
                               'MAC throughput and output FIFO/channel draining are outside this array/ECC follow-up.',
                               'Copy/recycle overlap, evaluation-start overwrite, programmable source order and zero extra source-switch overhead remain conditional.'])
    (DEST / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    best500 = next(x for x in planned_records if x['copy_ns'] == 500)
    for name, cuts in [('G4_copy500_best_any', best500['best_any_cut']['cuts']),
                       ('G4_copy500_best_full_row', best500['best_full_row_cut']['cuts']),
                       ('G4_copy500_six_six', [6])]:
        (DEST / (name + '.events.json')).write_text(json.dumps(planned(candidate, 500, cuts, trace=True), indent=2) + '\n')
    lines = ['# Full-wave chain and explicit cold-burst follow-up', '',
             'The sealed v1 simulator and its reports are unchanged. This follow-up independently reconstructs finite-latch copy/ECC timing and checks1..64full waves at seven copy latencies and three source-order sequences (1344cases).', '',
             'For N>=3, a necessary condition is **2c+(N−2)×540ns<=6040ns**, plus initial c<=6040ns. It follows from the last prior-wave ECC plane, irrespective of source permutation. Fixed0123 attains the bound here; other orders can add bubbles. This is a model-specific whole-page cache-only limit.', '',
             '| Copy ns | Maximum fixed-order full-wave chain | Alternating reverse | Random order |',
             '|---:|---:|---:|---:|']
    lines += [f"| {x['copy_ns']} | {x['fixed']} | {x['alternate_reverse']} | {x['random']} |" for x in maxima]
    lines += ['', 'N12 allows at most320ns copy; N6 allows1940ns. N44 necessarily fails this finite-bank model even at zero copy. This does not show the published hardware is wrong: it shows why raw9.7us array timing plus a separate6.4GB/s whole-page ECC cannot be assumed to pipeline arbitrarily long with two latches.', '',
              'An explicit cold-voltage restart is a separate conditional control. It waits for the read latches and pays28us, never extending a9.7us CR through an idle interval. Every possible cold-cut subset for G4’s12waves was enumerated; full-row-safe cut indices are3,6,9.', '',
              '| Copy ns | Reactive cold reset us | Best arbitrary cuts us | Best full-row cuts us | Fixed6+6 us |',
              '|---:|---:|---:|---:|---:|---:|']
    for x in planned_records:
        def us(k):
            value = x[k]['makespan_ns']
            return f'{value / 1000:.3f}' if value is not None else 'FAIL'
        lines.append(f"| {x['copy_ns']} | {us('reactive_cold_reset')} | {us('best_any_cut')} | {us('best_full_row_cut')} | {us('fixed_six_six_cut')} |")
    lines += ['', 'At500ns, reactive reset is170.140us. A full-row cut at3/6/9ties this timing; fixed6+6 is feasible. A one-wave early cut saves500ns (169.640us) if the existing partial-row GEMV state survives the cold restart; that extra capability is explicit. At2000ns, full-row-only cuts need four3wave bursts and are slower than reactive cuts.', '',
              'These are conditional service envelopes ending at weight ECC completion. Source-switch overhead, MAC throughput, output buffering and the existence of an in-command cold restart trigger are not certified.']
    (DEST / 'report.md').write_text('\n'.join(lines) + '\n')
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SEALED_SHA
    print(json.dumps(dict(status='PASS', cases=len(chain), chain_max_full_waves=maxima,
                          G4_planned_cold=planned_records), indent=2))


if __name__ == '__main__':
    main()
