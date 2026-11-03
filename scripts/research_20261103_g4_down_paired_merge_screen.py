#!/usr/bin/env python3
"""Bounded ordinary Down128/256 paired-prefix work/LP screen, TRAIN only."""
import argparse
import gzip
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/research-2026-11-02/round2'
OUT = BASE / 'g4-down-paired-merge-screen'
INC = BASE / 'compact-coupled-solver/continuation/native/0107.npz'
CFG = BASE / 'compact-coupled-solver/protocol.frozen.json'
TRAIN = BASE / 'w8-format-training/granite4tiny/decode.routes.npz'
SELF = Path(__file__).resolve()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, x):
    Path(p).write_text(json.dumps(x, indent=2) + '\n')


def freeze():
    OUT.mkdir(exist_ok=False)
    paths = [SELF, INC, CFG, TRAIN, ROOT / 'scripts/research_20261103_ordinary_read_native.cpp',
             ROOT / 'scripts/research_20261103_ordinary_read_backend.py', ROOT / 'paper/aif.txt', ROOT / 'kernel/aif.h']
    dump(OUT / 'protocol.frozen.json', dict(scope='G4L0 CP0107 TRAIN512 input1/command250. Work/continuous-LP screen only; no native, weights, heldout or existing-seal changes.',
        inputs=[dict(path=str(p), sha256=sha(p)) for p in paths],
        layouts='32 routeblind:16 seeded random directed matchings,8 seeded matchings directed to lower expertGU+shared chip work,8 deterministic low/high matching variants.',
        seed=202611100, per_expert='12positive Down128chips paired6edges. Each fixed dest core128+copied source128; originalsource retained. Split128+128 or merged0+256.',
        physical='6copies x128rows x512columns x64experts=24MiB/384waves. All jobs read full private512 input after wholeGU/A8. Source/dest128/256 are exact1/2waves; no fringe or partial dots.',
        feasible='For each query/layout, allsplit and allmerged starts, each two complete36bit greedy coordinate passes; strict maxchip-work improvement only.',
        LP='All32layouts x512queries continuous x[0,1] minimax. Safe certificate from exact integer-quantized nonnegative chip dual weights: floor((w.base+sum min(0,w.effect))/sumw). No floating solver fun used as a safe bound.',
        static='Selected best-feasible and best-LP layouts: same edge bits fixed/expert, zero-copy repack, two complete64x6 coordinate passes fromallsplit/allmerged. Work objective only; no globaloptimality.',
        gate='Report any >=2% mean or >=5% P99 optimistic native headroom before native/strong copy controls. Lower bound is host+LP dual, no selector penalty and no assumed attainability.',
        strong_controls='Same24MiB useful/page/wave budget can fund32 complete Down backups (each0.75MiB/12waves), not wholeFFN. Must fit IDs/locations/profiles with comparable TRAIN effort plus strong existing g6/down256 and mixed/static onecopy controls before a claim.',
        novelty='Standard replicated-job allocation and LP; not a new theorem. Old4MiB single-edge and other sealed branches remain unchanged.'))
    print(sha(OUT / 'protocol.frozen.json'))


def independent_table(rows, columns):
    table = np.zeros((65, 2, 16), np.int64)
    for e in range(65):
        for p in range(2):
            for c, n in enumerate(rows[e, p]):
                if n:
                    width = int(columns[e, p]); pages = (int(n) * width + 16383) // 16384
                    emitted = pages * 16384 // width
                    table[e, p, c] = 250 + (width + 1) // 2 + ((pages + 3) // 4) * 28000 + 2 * emitted
    return table


def greedy(base, effect):
    best = None
    for initial in (0, 1):
        bits = np.full(effect.shape[:2], initial, np.int8)
        loads = base + effect.sum(1) * initial
        values = loads.max(1)
        changes = []
        for _ in range(2):
            changed = 0
            for j in range(36):
                proposed = loads + effect[:, j] * (1 - 2 * bits[:, j, None])
                score = proposed.max(1); take = score < values
                changed += int(take.sum()); loads[take] = proposed[take]; values[take] = score[take]; bits[take, j] ^= 1
            changes.append(changed)
        if best is None:
            best = (values.copy(), bits.copy(), loads.copy(), [changes])
        else:
            take = values < best[0]; best[0][take] = values[take]; best[1][take] = bits[take]; best[2][take] = loads[take]; best[3].append(changes)
    assert np.array_equal(best[2], base + np.einsum('qj,qjc->qc', best[1], effect))
    return best


def static_fit(base, effects, routes):
    members = [np.flatnonzero((routes == e).any(1)) for e in range(64)]
    order = sorted(range(64), key=lambda e: (-len(members[e]), e)); results = []
    for initial in (0, 1):
        bits = np.full((64, 6), initial, np.int8)
        loads = base + (effects * initial)[routes].sum((1, 2)); values = loads.max(1); passes = []
        for _ in range(2):
            changed = 0
            for e in order:
                qq = members[e]
                for j in range(6):
                    proposed = loads[qq] + effects[e, j] * (1 - 2 * int(bits[e, j])); vv = proposed.max(1)
                    if vv.sum() < values[qq].sum():
                        bits[e, j] ^= 1; loads[qq] = proposed; values[qq] = vv; changed += 1
            passes.append(dict(changed_bits=changed, mean_work_ns=float(values.mean())))
        assert np.array_equal(loads, base + np.einsum('qkj,qkjc->qc', bits[routes], effects[routes]))
        results.append((float(values.mean()), bits, values, passes, initial))
    return min(results, key=lambda x: x[0])


def run():
    protocol = json.loads((OUT / 'protocol.frozen.json').read_text())
    for item in protocol['inputs']:
        assert sha(item['path']) == item['sha256']
    assert not (OUT / 'report.json').exists()
    began = time.perf_counter()
    with np.load(INC) as z:
        rows = z['rows'].copy(); columns = z['input_columns'].copy(); routes = z['routes'].copy(); native = z['cost'].copy()
    with np.load(TRAIN) as z:
        assert z['split'].item() == 'train' and z['completed'].all()
        assert np.array_equal(routes, z['routes'][:, :, 0].reshape(512, 6))
    service = json.loads(CFG.read_text())['service']
    assert service['command_ns'] == 250 and service['input_channel_copies'] == 1 and service['output_fringe'] == 1
    assert np.all(np.sort(rows[:64, 1], axis=1) == [0] * 4 + [128] * 12)
    table = independent_table(rows, columns)
    base = table[routes].sum((1, 2)) + table[64].sum(0)
    assert np.all(table[:64, 1][rows[:64, 1] > 0] == 28762)
    assert 250 + 256 + 56000 + 512 == 57018
    layouts, names = [], []
    for mode in range(32):
        edges = []
        rng = np.random.default_rng(protocol['seed'] + mode)
        for e in range(64):
            positive = np.flatnonzero(rows[e, 1] == 128)
            weights = table[e, 0] + table[64].sum(0)
            if mode < 24:
                paired = rng.permutation(positive).reshape(6, 2)
                if mode >= 16:
                    for pair in paired:
                        if weights[pair[0]] < weights[pair[1]]:
                            pair[:] = pair[::-1]
            else:
                ordered = np.asarray(sorted(positive, key=lambda c: (weights[c], c)))
                dest = ordered[:6]; src = np.roll(ordered[6:], (mode - 24) % 6)
                if mode >= 30:
                    src = src[::-1]
                paired = np.column_stack([src, dest])
            assert np.array_equal(np.sort(paired.ravel()), positive)
            edges.append(paired)
        names.append('random_' + str(mode) if mode < 16 else ('GUshared_directed_' + str(mode) if mode < 24 else 'lowhigh_' + str(mode)))
        layouts.append(np.asarray(edges, np.int32))
    records, effects_list, feasible_bits, feasible_values = [], [], [], []
    for j, edges in enumerate(layouts):
        effects = np.zeros((64, 6, 16), np.int64)
        for e in range(64):
            for k, (src, dest) in enumerate(edges[e]):
                effects[e, k, src] = -28762; effects[e, k, dest] = 28256
        assert np.all(effects.sum(2) == -506)
        effect = effects[routes].reshape(512, 36, 16)
        vals, bits, _, changes = greedy(base, effect)
        records.append(dict(index=j, name=names[j], feasible_mean_work_ns=float(vals.mean()), feasible_p99_work_ns=float(np.percentile(vals, 99)),
                            feasible_work_gain_pct=float(100 * (1 - vals.mean() / base.max(1).mean())), greedy_changes=changes,
                            allsplit_mean_work_ns=float(base.max(1).mean()), allmerged_mean_work_ns=float((base + effect.sum(1)).max(1).mean())))
        effects_list.append(effects); feasible_bits.append(bits); feasible_values.append(vals)
    order = sorted(range(32), key=lambda j: records[j]['feasible_mean_work_ns'])
    lp_bounds = np.zeros((32, 512), np.int64); lp_weights = np.zeros((32, 512, 16), np.int64)
    lp_choices = np.zeros((32, 512, 36), np.float64); max_gap = 0.0
    objective = np.r_[np.zeros(36), 1.0]
    for position, j in enumerate(order):
        effect = effects_list[j][routes].reshape(512, 36, 16)
        for q in range(512):
            matrix = np.column_stack([effect[q].T, -np.ones(16)])
            solution = linprog(objective, A_ub=matrix, b_ub=-base[q], bounds=[(0, 1)] * 36 + [(None, None)], method='highs')
            assert solution.success, solution.message
            weights = np.rint(np.maximum(0, -solution.ineqlin.marginals) * (1 << 32)).astype(np.int64)
            denominator = int(weights.sum()); assert denominator > 0
            numerator = int(weights @ base[q]) + int(np.minimum(0, effect[q] @ weights).sum())
            certificate = numerator // denominator
            lp_bounds[j, q] = certificate; lp_weights[j, q] = weights; lp_choices[j, q] = solution.x[:36]
            assert certificate <= feasible_values[j][q]
            assert certificate <= solution.fun + 0.1
            max_gap = max(max_gap, float(solution.fun - certificate))
        LB = lp_bounds[j] + 2000
        records[j].update(lp_safe_mean_work_LB_ns=float(lp_bounds[j].mean()), lp_safe_p99_work_LB_ns=float(np.percentile(lp_bounds[j], 99)),
                          optimistic_native_mean_LB_ns=float(LB.mean()), optimistic_native_p99_LB_ns=float(np.percentile(LB, 99)),
                          optimistic_native_mean_gain_upper_pct=float(100 * (1 - LB.mean() / native.mean())),
                          optimistic_native_p99_gain_upper_pct=float(100 * (1 - np.percentile(LB, 99) / np.percentile(native, 99))))
        if position in (0, 7, 15, 23, 31):
            print(json.dumps(dict(completed_layouts=position+1, record=records[j])), flush=True)
    best_feasible = min(range(32), key=lambda j: records[j]['feasible_mean_work_ns'])
    best_LP = min(range(32), key=lambda j: records[j]['lp_safe_mean_work_LB_ns'])
    statics = {}
    for j in sorted({best_feasible, best_LP}):
        mean, bits, vals, passes, initial = static_fit(base, effects_list[j], routes)
        statics[str(j)] = dict(mean_work_ns=mean, p99_work_ns=float(np.percentile(vals, 99)), passes=passes, initial=initial,
                               dynamic_vs_static_work_pct=float(100 * (1 - feasible_values[j].mean() / mean)),
                               bits=bits.tolist())
    # Symbolic copy/prefix proof for every64edge-state/expert in selected layout.
    rowids = []
    for e in range(64):
        starts = np.r_[0, np.cumsum(rows[e, 1])]
        original = [np.arange(starts[c], starts[c+1], dtype=np.int32) for c in range(16)]
        image = [x.copy() for x in original]
        for src, dest in layouts[best_feasible][e]:
            image[dest] = np.r_[image[dest], original[src]]
        counts = np.bincount(np.concatenate(image), minlength=1536)
        assert (counts == 2).sum() == 768 and (counts == 1).sum() == 768
        for mask in range(64):
            selected = [x.copy() for x in original]
            for k, (src, dest) in enumerate(layouts[best_feasible][e]):
                if mask >> k & 1:
                    selected[src] = np.array([], np.int32); selected[dest] = image[dest]
            assert np.array_equal(np.sort(np.concatenate(selected)), np.arange(1536))
            assert all(len(x) in (0, 128, 256) for x in selected)
        rowids.append(dict(expert=e, edges=layouts[best_feasible][e].tolist(), physical_logical_row_ids=[x.tolist() for x in image],
                           row_scale='Original expert Down logical row F32 scale; copied slots reference same identity.'))
    with gzip.open(OUT / 'selected_copy_rowids.json.gz', 'wt') as f:
        json.dump(rowids, f)
    np.savez_compressed(OUT / 'arrays.npz', rows=rows, input_columns=columns, routes=routes, baseline_native=native, base=base,
                        layouts=np.asarray(layouts), effects=np.asarray(effects_list), feasible_values=np.asarray(feasible_values),
                        feasible_bits=np.asarray(feasible_bits), lp_safe_bounds=lp_bounds, lp_weights=lp_weights, lp_continuous_choices=lp_choices)
    report = dict(status='Bounded TRAIN work/continuous-LP screen only', protocol_sha256=sha(OUT / 'protocol.frozen.json'),
                  scipy_version=scipy.__version__, baseline_native_mean_ns=float(native.mean()), baseline_native_p99_ns=float(np.percentile(native, 99)),
                  baseline_mean_work_ns=float(base.max(1).mean()), LP_cases=32*512, LP_dual_safe_floor=True,
                  max_float_optimum_minus_integer_dual_certificate_ns=max_gap, records=records,
                  best_feasible_layout=best_feasible, best_LP_layout=best_LP, static_controls=statics,
                  copy_budget=dict(experts=64, copies_per_expert=6, rows_per_copy=128, columns=512,
                                   coefficient_bytes=25165824, pages=1536, waves=384, complete_Down_backups_same_budget=32),
                  symbolic_view_cases_checked=4096, geometric_partial_rows=0,
                  gate_possible_ge2mean_or5tail=any(r['optimistic_native_mean_gain_upper_pct'] >= 2 or r['optimistic_native_p99_gain_upper_pct'] >= 5 for r in records),
                  limitations='LP fractional edge choices, readiness/resources/selector/outputABI not established; feasible means chip-work only. No native speed. Static two-pass control not globally optimal. Same-budget wholeDown32 and existing g6/mixed controls required if continuing.',
                  seconds=time.perf_counter()-began)
    for item in protocol['inputs']:
        assert sha(item['path']) == item['sha256']
    dump(OUT / 'report.json', report)
    print(json.dumps(dict(best_feasible=records[best_feasible], best_LP=records[best_LP], gate=report['gate_possible_ge2mean_or5tail'],
                          statics={k:{kk:vv for kk,vv in v.items() if kk != 'bits'} for k,v in statics.items()}, seconds=report['seconds']), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['freeze', 'run']); args = parser.parse_args()
    freeze() if args.action == 'freeze' else run()
