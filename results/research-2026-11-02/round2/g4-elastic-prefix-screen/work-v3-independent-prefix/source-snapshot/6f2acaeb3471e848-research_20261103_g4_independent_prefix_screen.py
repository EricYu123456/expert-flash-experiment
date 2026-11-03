#!/usr/bin/env python3
"""TRAIN-only independent long/short prefix contract and necessary-work screen.

Storage layout is fixed; runtime chooses 8 independent bits per selected expert.
No backend/numerical guard is changed. Computed/consumed rows are separate ledgers.
"""
import argparse
import hashlib
import json
import shutil
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'results/research-2026-11-02/round2'
OUT = BASE/'g4-elastic-prefix-screen/work-v3-independent-prefix'
V1 = BASE/'g4-elastic-prefix-screen/work-v1'
B = BASE/'compact-coupled-solver/continuation/native/0107.npz'
CFG = BASE/'compact-coupled-solver/protocol.frozen.json'
TRAIN = BASE/'w8-format-training/granite4tiny/decode.routes.npz'
PAIR_SEEDS = [2026111101, 2026111102]
SCALE = 28000


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2)+'\n')


def stats(x):
    return dict(mean_ns=float(np.mean(x)), p99_ns=float(np.percentile(x, 99)))


def ceiling(lb, incumbent):
    return dict(LB=stats(lb), mean_pct=100*(1-float(np.mean(lb))/float(np.mean(incumbent))),
                p99_pct=100*(1-float(np.percentile(lb, 99))/float(np.percentile(incumbent, 99))))


def freeze():
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT/'source-snapshot').mkdir()
    inputs = []
    paths = [Path(__file__), B, CFG, TRAIN, V1/'work_arrays.npz', V1/'protocol.frozen.json',
             V1/'independent/completion.json']
    for path in paths:
        item = dict(path=str(path), sha256=sha(path))
        if path.suffix == '.py':
            snapshot = OUT/'source-snapshot'/(item['sha256'][:16]+'-'+path.name)
            shutil.copy2(path, snapshot)
            item['snapshot'] = str(snapshot)
        inputs.append(item)
    with np.load(B) as b, np.load(V1/'work_arrays.npz') as v1, np.load(TRAIN) as train:
        assert np.array_equal(b['routes'], v1['routes'])
        assert np.array_equal(b['routes'], train['routes'][:, :, 0].reshape(512, 6))
        assert np.array_equal(b['rows'], v1['rows_B'])
        assert train['split'].item() == 'train' and train['completed'].all()
    p = dict(status='frozen_before_contract_and_TRAIN_work', utc=datetime.now(timezone.utc).isoformat(),
             source_sha256=sha(__file__), inputs=inputs, scipy_version=scipy.__version__,
             service=json.loads(CFG.read_text())['service'],
             scope='G4 L0 actualW8 TRAIN512; same fixed v1 A8 positions; unchanged Down/shared; no native/heldout/alllayer.',
             layouts=[dict(index=0, special='last source chip', low_pairing='ascending source/low chips'),
                      dict(index=1, special='first source chip', low_pairing='ascending others, low chips rotated1'),
                      dict(index=2, seed=PAIR_SEEDS[0], special='uniform source', low_pairing='uniform permutation'),
                      dict(index=3, seed=PAIR_SEEDS[1], special='uniform source', low_pairing='uniform permutation')],
             storage='For 7 sources: original Bcore | one low40 original row range | 5/6-row highprefix piece. Special eighth source: Bcore | 45/46-row highsuffix. Original low/high destinations remain resident. Everyexpert33waves; same v1 copied bytes/512 extra waves perlayer.',
             contract='u_i independently chooses long128 or shortBcore. Low_i job executes iff notu_i. Original highdestination fullh if notu_special; prefix36..38 ifu_special and notalllong; absentifalllong. Every read/computed duplicate/fringe output ischarged; logical ownership consumes each original GU row exactlyonce. FullGU->A8 barrier remains.',
             delta='For i0..6: +28086 source,-29102 low. i7special:+28086 source,-28086 highdest. z=AND8u gives additional-29102 highdest. Baseincludes B GU+privateDown+shared.',
             LP=dict(queries_per_layout=512, layouts=4, variables='48u+6z+T', constraints='16chip+48 z<=u+6 sumu-z<=7',
                     bounds='u,z in[0,1]; T>=0; continuous necessary relaxation', solver='scipy.linprog HiGHS',
                     certificate='Recompute box-dual bound with nonpositive dual multipliers, sumchip lambda<=1-1e-10. Residuals over[0,1] contribute min(residual,0); T residual nonnegative. Convert to ns and floor after1e-4ns safety. No primal LP value used as certified LB.'),
             gate=dict(mean_pct=2, p99_pct=5, action='Stop this finite fixedpairing family if all LPceilings belowboth thresholds. Otherwise run bounded feasible greedy and report; no automaticnative.'),
             greedy=dict(starts=['allB','allA'], passes=2, bit_order='selectedexpert ordinal, source bits0..7',
                         objective='lexicographic (maxchipwork,sumSquaredChipwork,descendingChiploads)',
                         pairing_choice='One offline fixedpairing selected by mean feasible work acrossTRAIN. Runtime cannot mixpairings.',
                         scope='Atmost4x512x2startsx2passesx48flips; workonly. No selector/native costs waived in performance claims.'),
             limitations='Published prefixreplication/orientation principles; no novelalgorithm claim. LP/greedy work is not achieved latency. Outputfringe, ordinaryview/nativeABI remain modeled not measured. Complete image rows are W8 ordinary fullrows, not partialdots or new inputformat.')
    dump(OUT/'protocol.frozen.json', p)
    print(json.dumps(dict(protocol_sha256=sha(OUT/'protocol.frozen.json'), source_sha256=sha(__file__))))


def load_inputs():
    p = json.loads((OUT/'protocol.frozen.json').read_text())
    assert sha(__file__) == p['source_sha256']
    for item in p['inputs']:
        assert sha(item['path']) == item['sha256']
    b = np.load(B)
    v1 = np.load(V1/'work_arrays.npz')
    return p, b, v1


def job_work(nr, nc=1536):
    if nr == 0:
        return 0, 0, 0, 0
    pages = (nr*nc+16383)//16384
    waves = (pages+3)//4
    emitted = pages*16384//nc
    return 250+(nc+1)//2+28000*waves+2*emitted, pages, waves, emitted


def contract():
    p, b, v1 = load_inputs()
    assert not (OUT/'contract.json').exists()
    rows, routes = b['rows'], b['routes']
    # Independent job accounting, not the existing occupancy helper.
    table = np.zeros((65, 2, 16), np.int64)
    for e in range(65):
        for phase in range(2):
            for c, nr, nc, pages, waves in b['jobs'][slice(*b['offsets'][e, phase])]:
                work, pg, wa, emit = job_work(int(nr), int(nc))
                assert (pg, wa) == (pages, waves)
                table[e, phase, c] += work
    base = table[routes].sum((1, 2))+table[64].sum(0)
    assert np.array_equal(base, v1['base'])
    sources = np.zeros((4, 64, 8), np.int32)
    lowdest = np.zeros((4, 64, 7), np.int32)
    highdest = np.zeros(64, np.int32)
    D = np.zeros((4, 64, 8, 16), np.int64)
    Z = np.zeros((64, 16), np.int64)
    ledger = []
    view_counts = np.zeros((4, 64, 256, 6), np.int64)  # jobs/pages/waves/meaningfulcomputed/emitted/unique
    for layout in range(4):
        rng = np.random.default_rng(PAIR_SEEDS[layout-2]) if layout >= 2 else None
        per_expert = []
        for e in range(64):
            high = np.flatnonzero(rows[e, 0] > 40)
            low = np.flatnonzero(rows[e, 0] == 40)
            hd = int(high[v1['selected_layout'][e]])
            aa = [int(c) for c in high if c != hd]
            special = aa[-1] if layout == 0 else aa[0] if layout == 1 else int(rng.choice(aa))
            other = sorted(c for c in aa if c != special)
            ll = low if layout == 0 else np.roll(low, -1) if layout == 1 else rng.permutation(low)
            sources[layout, e] = other+[special]
            lowdest[layout, e] = ll
            highdest[e] = hd
            original = {c: list(range(int(rows[e, 0, :c].sum()), int(rows[e, 0, :c+1].sum()))) for c in range(16)}
            image = {c: rr.copy() for c, rr in original.items()}
            a8 = 128-len(original[special])
            prefix_n = len(original[hd])-a8
            prefix_pieces = []
            at = 0
            for i, c in enumerate(other):
                n = 128-len(original[c])-40
                piece = original[hd][at:at+n]
                at += n
                prefix_pieces.append(piece)
                image[c] += original[int(ll[i])]+piece
                D[layout, e, i, c] = 28086
                D[layout, e, i, ll[i]] = -29102
            assert at == prefix_n and prefix_n in (36, 37, 38)
            suffix = original[hd][prefix_n:]
            assert len(suffix) == a8 and a8 in (45, 46)
            image[special] += suffix
            D[layout, e, 7, special] = 28086
            D[layout, e, 7, hd] = -28086
            Z[e, hd] = -29102
            assert all(len(image[c]) == 128 for c in aa)
            assert np.array_equal(D[layout, e].sum(0)+Z[e], v1['selected_delta'][e])
            # Exhaust all256 native view semantics and consumed logical rows.
            for mask in range(256):
                u = [(mask >> i) & 1 for i in range(8)]
                computed, consumed, jobs = [], [], []
                for i, c in enumerate(other+[special]):
                    nr = 128 if u[i] else len(original[c])
                    computed.extend(image[c][:nr])
                    consumed.extend(image[c][:nr])
                    jobs.append((c, nr))
                for i in range(7):
                    if not u[i]:
                        computed.extend(original[int(ll[i])])
                        consumed.extend(original[int(ll[i])])
                        jobs.append((int(ll[i]), 40))
                hn = len(original[hd]) if not u[7] else prefix_n if not all(u) else 0
                if hn:
                    computed.extend(original[hd][:hn])
                    for i in range(7):
                        if not u[i]:
                            consumed.extend(prefix_pieces[i])
                    if not u[7]:
                        consumed.extend(suffix)
                    jobs.append((hd, hn))
                assert sorted(consumed) == list(range(1024))
                assert len(computed) >= 1024
                actual = np.zeros(16, np.int64)
                pages = waves = emitted = 0
                for c, nr in jobs:
                    work, pg, wa, em = job_work(nr)
                    actual[c] += work
                    pages += pg
                    waves += wa
                    emitted += em
                expected = table[e, 0]+np.einsum('i,ic->c', u, D[layout, e])+int(all(u))*Z[e]
                assert np.array_equal(actual, expected)
                view_counts[layout, e, mask] = [len(jobs), pages, waves, len(computed), emitted, len(consumed)]
            copyrows = sum(len(image[c])-len(original[c]) for c in aa)
            assert copyrows in (362, 363)
            multiplicity = Counter(rr for band in image.values() for rr in band)
            donors = [rr for c in range(16) if c not in aa for rr in original[c]]
            assert multiplicity == Counter({r: 2 if r in donors else 1 for r in range(1024)})
            per_expert.append(dict(expert=e, sources=other+[special], low_destinations=ll.tolist(),
                                   high_destination=hd, high_prefix_rows=prefix_n,
                                   special_suffix_rows=a8, B_original_rows=original,
                                   resident_prefix_image=image, prefix_piece_rows=prefix_pieces,
                                   special_suffix=suffix, copied_rows=copyrows))
        ledger.append(dict(layout=layout, experts=per_expert))
    np.savez_compressed(OUT/'contract_arrays.npz', rows=rows, routes=routes, baseline_native=b['cost'],
                        base=base, table=table, sources=sources, lowdest=lowdest, highdest=highdest,
                        D=D, Z=Z, view_counts=view_counts, resident_rows=v1['resident_rows'],
                        resident_jobs=v1['resident_jobs'], resident_addresses=v1['resident_addresses'])
    dump(OUT/'rowid_ledgers.json', ledger)
    report = dict(status='all256views_perexpert_PASS', protocol_sha256=sha(OUT/'protocol.frozen.json'),
                  layouts=4, experts_per_layout=64, view_instances_checked=4*64*256,
                  computed_GU_rows_min=int(view_counts[..., 3].min()), computed_GU_rows_max=int(view_counts[..., 3].max()),
                  emitted_rows_min=int(view_counts[..., 4].min()), emitted_rows_max=int(view_counts[..., 4].max()),
                  unique_consumed_rows=1024, jobs_min=int(view_counts[..., 0].min()), jobs_max=int(view_counts[..., 0].max()),
                  copied_rows_per_layer=sum(x['copied_rows'] for x in ledger[0]['experts']),
                  extra_waves_per_layer=512, extra_pages_per_layer=2048,
                  same_resident_addresses_as_v1=True,
                  limits='Meaningful computed count includes duplicate rows; emitted count additionally includes pagefringe. No backendguard was weakened and no malformed plan imported. Native execution/output ABI remains modeled.')
    dump(OUT/'contract.json', report)
    print(json.dumps(report, indent=2))


def matrix_for(D, Z, base):
    # Variables [6x8 u,6 z,T], inequalities [16chip,48ANDupper,6ANDlower].
    A = np.zeros((70, 55), np.float64)
    rhs = np.zeros(70, np.float64)
    A[:16, :48] = D.reshape(48, 16).T/SCALE
    A[:16, 48:54] = Z.T/SCALE
    A[:16, 54] = -1
    rhs[:16] = -base/SCALE
    for e in range(6):
        for i in range(8):
            A[16+8*e+i, 48+e] = 1
            A[16+8*e+i, 8*e+i] = -1
        A[64+e, 8*e:8*e+8] = 1
        A[64+e, 48+e] = -1
        rhs[64+e] = 7
    return A, rhs


def lp():
    p, b, v1 = load_inputs()
    assert not (OUT/'LP_report.json').exists()
    a = np.load(OUT/'contract_arrays.npz')
    D, Z, base, routes, native = [a[k] for k in ('D','Z','base','routes','baseline_native')]
    objective = np.zeros(55)
    objective[-1] = 1
    primal = np.zeros((4, 512))
    lower = np.zeros((4, 512), np.int64)
    duals = np.zeros((4, 512, 70))
    residualT = np.zeros((4, 512))
    maximum_violation = 0.0
    t0 = time.monotonic()
    for layout in range(4):
        for q, selected in enumerate(routes):
            A, rhs = matrix_for(D[layout, selected], Z[selected], base[q])
            result = linprog(objective, A_ub=A, b_ub=rhs,
                             bounds=[(0, 1)]*54+[(0, None)], method='highs')
            assert result.success and result.status == 0
            primal[layout, q] = result.fun*SCALE
            y = np.minimum(result.ineqlin.marginals, 0)
            chipmass = -float(y[:16].sum())
            if chipmass:
                y *= min(1.0, (1-1e-10)/chipmass)
            residual = objective-A.T@y
            assert residual[-1] >= 0
            # For bounded u,z, exact box minimization supplies a dual certificate
            # even if solver multipliers have reduced-cost numerical residuals.
            lb = float(y@rhs+np.minimum(residual[:54], 0).sum())*SCALE
            lower[layout, q] = int(np.floor(lb-1e-4))
            assert -1e-4 <= primal[layout, q]-lb < 0.1
            duals[layout, q] = y
            residualT[layout, q] = residual[-1]
            maximum_violation = max(maximum_violation, float(np.max(A@result.x-rhs)))
        print(json.dumps(dict(layout=layout, primal=stats(primal[layout]), necessary=ceiling(lower[layout]+2000, native))), flush=True)
    assert np.all(lower+2000 <= native)
    ceilings = [ceiling(x+2000, native) for x in lower]
    passed = any(x['mean_pct'] >= 2 or x['p99_pct'] >= 5 for x in ceilings)
    np.savez_compressed(OUT/'LP_arrays.npz', primal_work=primal, necessary_work_LB=lower,
                        dual_multipliers=duals, residual_T=residualT)
    report = dict(status='continuous_necessary_LP_not_runtime', protocol_sha256=sha(OUT/'protocol.frozen.json'),
                  layout_ceilings=ceilings, gate_passed=passed, solves=2048, variables=55, inequalities=70,
                  seconds=time.monotonic()-t0, maximum_scaled_primal_violation=maximum_violation,
                  floor_safety_ns=1e-4, maximum_primal_minus_integer_lower_ns=float(np.max(primal-lower)),
                  limits='Each layout is fixed. LP permits fractional bits and relaxes readiness/queue/selector. Empirical quantile LB follows pointwiseT>=LP; no native performance claim.')
    dump(OUT/'LP_report.json', report)
    print(json.dumps(report, indent=2))


def greedy():
    p, b, v1 = load_inputs()
    lpr = json.loads((OUT/'LP_report.json').read_text())
    assert lpr['gate_passed'] and not (OUT/'greedy_report.json').exists()
    a = np.load(OUT/'contract_arrays.npz')
    D, Z, base, routes, native = [a[k] for k in ('D','Z','base','routes','baseline_native')]
    values = np.zeros((4, 2, 512), np.int64)
    bits = np.zeros((4, 2, 512, 6, 8), np.int8)
    pass_changes = np.zeros((4, 2, 512, 2), np.int16)
    passes_work = np.zeros((4, 2, 512, 2), np.int64)
    initial_work = np.zeros((4, 2, 512), np.int64)
    t0 = time.monotonic()

    def key(loads):
        return (int(loads.max()), int(loads@loads), *sorted(map(int, loads), reverse=True))

    for layout in range(4):
        for start in range(2):
            for q, selected in enumerate(routes):
                u = np.full((6, 8), start, np.int8)
                de, ze = D[layout, selected], Z[selected]
                loads = base[q]+np.einsum('ei,eic->c', u, de)+(np.all(u, axis=1)[:, None]*ze).sum(0)
                initial_work[layout, start, q] = loads.max()
                for pi in range(2):
                    changes = 0
                    for e in range(6):
                        for i in range(8):
                            old_z = int(u[e].all())
                            sign = 1 if not u[e, i] else -1
                            u[e, i] ^= 1
                            new_z = int(u[e].all())
                            candidate = loads+sign*de[e, i]+(new_z-old_z)*ze[e]
                            if key(candidate) < key(loads):
                                loads = candidate
                                changes += 1
                            else:
                                u[e, i] ^= 1
                    pass_changes[layout, start, q, pi] = changes
                    passes_work[layout, start, q, pi] = loads.max()
                full = base[q]+np.einsum('ei,eic->c', u, de)+(np.all(u, axis=1)[:, None]*ze).sum(0)
                assert np.array_equal(full, loads)
                values[layout, start, q] = loads.max()
                bits[layout, start, q] = u
        print(json.dumps(dict(layout=layout, start0=stats(values[layout, 0]), start1=stats(values[layout, 1]),
                              best=stats(values[layout].min(0)))), flush=True)
    best_starts = values.argmin(1)
    feasible = values.min(1)
    best_bits = np.stack([bits[layout, best_starts[layout], np.arange(512)] for layout in range(4)])
    assert np.all(feasible <= base.max(1))
    best_layout = int(np.argmin(feasible.mean(1)))
    lower = np.load(OUT/'LP_arrays.npz')['necessary_work_LB']
    assert np.all(lower <= feasible)
    np.savez_compressed(OUT/'greedy_arrays.npz', initial_work=initial_work, values=values, bits=bits,
                        pass_changes=pass_changes, passes_work=passes_work,
                        best_starts=best_starts, feasible_work=feasible, feasible_bits=best_bits)
    report = dict(status='feasible_work_not_native_latency', protocol_sha256=sha(OUT/'protocol.frozen.json'),
                  layout_results=[dict(layout=layout, work=stats(feasible[layout]),
                      work_improvement_pct=100*(1-feasible[layout].mean()/base.max(1).mean()),
                      host_plus_feasible_work_ceiling=ceiling(feasible[layout]+2000, native),
                      LP_to_greedy_gap=stats(feasible[layout]-lower[layout]),
                      start_initial=[stats(x) for x in initial_work[layout]],
                      start_final=[stats(x) for x in values[layout]],
                      start_pass_changed_bits=[pass_changes[layout, s].sum(0).tolist() for s in range(2)]) for layout in range(4)],
                  selected_fixed_layout=best_layout, flips_attempted=4*2*512*2*48,
                  seconds=time.monotonic()-t0,
                  limits='The host+feasiblework values are lower bounds for those feasible masks, not an upper bound on the globallybest selector and not achieved nativeT. Choosingbestof2greedystarts is perquery; choosingpairing is offline only. Fullselectorcost/strongsamebudgetcopycontrols notrun.')
    dump(OUT/'greedy_report.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['freeze', 'contract', 'lp', 'greedy'])
    args = parser.parse_args()
    {'freeze': freeze, 'contract': contract, 'lp': lp, 'greedy': greedy}[args.mode]()
