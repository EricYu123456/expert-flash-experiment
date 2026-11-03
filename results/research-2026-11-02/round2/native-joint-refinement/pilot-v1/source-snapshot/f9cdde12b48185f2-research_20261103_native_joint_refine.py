#!/usr/bin/env python3
"""Bounded TRAIN-only refinement using the complete native service objective.

This is a standard local-search control, not a new optimization algorithm.
Every move preserves each projection's row-count multiset and complete rows.
Ordered routing tuples are retained because scheduler priorities use ordinals.
"""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

from research_20261103_topology_placement_replay import Replay, load_python

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/research-2026-11-02/round2'
OUT = BASE / 'native-joint-refinement'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ExactFit:
    def __init__(self, rows, shape, service, routes, kind, seed,
                 global_permutations=None):
        self.backend = load_python()
        self.native = Replay()
        self.rows = np.array(rows, np.int64, copy=True)
        self.initial_rows = self.rows.copy()
        self.shape = shape
        self.service = self.backend.Service(**service)
        self.config = service
        self.kind = kind
        self.rng = np.random.default_rng(seed)
        self.routes = np.ascontiguousarray(routes, np.int32)
        self.unique, self.weights = np.unique(self.routes, axis=0,
                                             return_counts=True)
        self.unique = np.ascontiguousarray(self.unique, np.int32)
        self.members = [np.flatnonzero((self.unique == e).any(1))
                        for e in range(shape[2])]
        self.plan = self.lower()
        self.values, self.bounds = self.native.run(self.plan, self.unique, service)
        self.total = int(self.values @ self.weights)
        self.initial_total = self.total
        self.best = self.total
        self.best_rows = self.rows.copy()
        self.initial_values = self.values.copy()
        self.calls = 1
        self.evaluated_queries = len(self.unique)
        self.valid = self.accepted = 0
        self.moves = {}
        self.variants = []
        self.permutations = np.array(global_permutations if
                                     global_permutations is not None else
                                     [list(range(16))] * 2, np.int32)
        if kind == 'cyclic':
            for p in range(2):
                # Undo the inherited common chip permutation before generating
                # rotations, then restore it. A roll after a general permutation
                # would silently change the legal geometry family.
                inv = np.argsort(self.permutations[p])
                base = self.rows[0, p, inv]
                vv = np.unique(np.stack([np.roll(base, j) for j in range(16)]),
                               axis=0)[:, self.permutations[p]]
                assert all(np.any(np.all(vv == rr, axis=1)) for rr in self.rows[:-1, p])
                self.variants.append(vv)
        self.best_permutations = self.permutations.copy()

    def lower(self):
        D, I, E, K, L, SI = self.shape
        return self.backend.lower_rows(self.rows[:, 0], self.rows[:, 1],
                                       D, [I] * E + [SI], self.service)

    def update_phase(self, e, p):
        """Rebuild one band in canonical chip order, preserving fixed offsets."""
        c = int(self.plan['input_columns'][e, p])
        rr = self.rows[e, p]
        chips = np.flatnonzero(rr)
        lo, hi = self.plan['offsets'][e, p]
        assert hi - lo == len(chips)
        pages = (rr[chips] * c + self.service.page_bytes - 1) // self.service.page_bytes
        waves = (pages + self.service.planes - 1) // self.service.planes
        assert np.all(waves <= self.service.block_wordlines)
        self.plan['jobs'][lo:hi] = np.column_stack(
            (chips, rr[chips], np.full(len(chips), c), pages, waves))

    def proposal(self, temperature):
        global_move = self.rng.random() < .04
        paired = self.kind == 'coupled' or self.rng.random() < .5
        phases = [0, 1] if paired else [int(self.rng.integers(2))]
        a, b = map(int, self.rng.choice(16, 2, replace=False))
        experts = list(range(self.shape[2])) if global_move else [int(self.rng.integers(self.shape[2]))]
        if global_move:
            tag = 'global_paired' if paired else 'global_single'
            changes = [(e, p, self.rows[e, p].copy(),
                        self.rows[e, p, np.array([b if j == a else a if j == b else j
                                                 for j in range(16)])])
                       for e in experts for p in phases]
        elif self.kind == 'cyclic':
            tag = 'label_paired' if paired else 'label_single'
            changes = [(experts[0], p, self.rows[experts[0], p].copy(),
                        self.variants[p][int(self.rng.integers(len(self.variants[p])))].copy())
                       for p in phases]
        else:
            tag = 'swap_paired' if paired else 'swap_single'
            changes = []
            for p in phases:
                old = self.rows[experts[0], p].copy()
                new = old.copy()
                new[[a, b]] = new[[b, a]]
                changes.append((experts[0], p, old, new))
        changes = [(e, p, old, new) for e, p, old, new in changes
                   if not np.array_equal(old, new)]
        self.moves.setdefault(tag, dict(attempts=0, valid=0, accepted=0))
        self.moves[tag]['attempts'] += 1
        if not changes:
            return
        affected = sorted(set(e for e, p, old, new in changes))
        ii = (np.arange(len(self.unique)) if global_move else self.members[affected[0]])
        if not len(ii):
            return
        for e, p, old, new in changes:
            self.rows[e, p] = new
            self.update_phase(e, p)
        values, bounds = self.native.run(self.plan, self.unique[ii], self.config)
        self.calls += 1
        self.evaluated_queries += len(ii)
        change = int((values - self.values[ii]) @ self.weights[ii])
        accept = change <= 0 or (temperature > 0 and self.rng.random() <
                                 math.exp(-change / (len(self.routes) * temperature)))
        self.valid += 1
        self.moves[tag]['valid'] += 1
        if accept:
            self.accepted += 1
            self.moves[tag]['accepted'] += 1
            self.values[ii], self.bounds[ii] = values, bounds
            self.total += change
            if global_move:
                for p in phases:
                    self.permutations[p, [a, b]] = self.permutations[p, [b, a]]
                    if self.kind == 'cyclic':
                        self.variants[p][:, [a, b]] = self.variants[p][:, [b, a]]
            if self.total < self.best:
                self.best = self.total
                self.best_rows = self.rows.copy()
                self.best_permutations = self.permutations.copy()
        else:
            for e, p, old, new in changes:
                self.rows[e, p] = old
                self.update_phase(e, p)

    def run(self, iterations, initial_temperature_ns):
        started = time.monotonic()
        for it in range(iterations):
            temperature = initial_temperature_ns * (.0001 ** (it / max(1, iterations - 1)))
            self.proposal(temperature)
        terminal = self.total / len(self.routes)
        self.rows = self.best_rows.copy()
        final_plan = self.lower()
        values, bounds = self.native.run(final_plan, self.routes, self.config)
        self.calls += 1
        self.evaluated_queries += len(self.routes)
        assert int(values.sum()) == self.best
        assert np.array_equal(np.sort(self.rows, axis=2),
                              np.sort(self.initial_rows, axis=2))
        assert np.array_equal(self.rows[-1], self.initial_rows[-1])
        return final_plan, values, bounds, dict(
            objective='exact native TRAIN mean, ordered routing tuples',
            proposals=iterations, valid=self.valid, accepted=self.accepted,
            move_counts=self.moves, native_calls=self.calls,
            native_query_evaluations=self.evaluated_queries,
            initial_temperature_ns=initial_temperature_ns,
            initial_mean_ns=self.initial_total / len(self.routes),
            terminal_mean_ns=terminal, final_mean_ns=self.best / len(self.routes),
            initial_retained=True, unique_ordered_routes=len(self.unique),
            original_routes=len(self.routes), seconds=time.monotonic() - started,
            final_global_permutations=self.best_permutations.tolist(),
            same_row_multisets=True, shared_unchanged=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', required=True, type=Path)
    ap.add_argument('--iterations', type=int, default=2048)
    ap.add_argument('--seed', type=int, default=20261105)
    ap.add_argument('--temperature', type=float, default=500.)
    ap.add_argument('--family', choices=['joint_cyclic', 'joint_simple_menu',
                                        'independent_permutation', 'independent_paired',
                                        'counterbalanced'], required=True)
    ap.add_argument('--protocol', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    assert args.iterations > 0
    protocol = json.loads(args.protocol.read_text())
    assert protocol['source_sha256'] == sha(__file__)
    assert args.iterations == protocol['iterations']
    assert args.seed in protocol['seeds']
    assert args.temperature == protocol['initial_temperature_ns']
    assert args.family in protocol['families']
    assert str(args.case.resolve()) in [str(Path(x).resolve()) for x in protocol['cases']]
    for source in protocol['inputs']:
        assert sha(source['path']) == source['sha256']
    rec = json.loads(args.case.read_text())
    winner = rec['winners'][args.family]
    art = winner[winner['selected']]
    assert sha(art['arrays']) == art['sha256']
    with np.load(art['arrays'], allow_pickle=False) as z:
        rows, routes, original_values = z['rows'], z['routes'], z['cost']
    kind = {'joint_cyclic': 'cyclic', 'joint_simple_menu': 'cyclic',
            'independent_permutation': 'independent', 'independent_paired': 'independent',
            'counterbalanced': 'coupled'}[args.family]
    selected_record, selected_stage = winner, winner['selected']
    if selected_stage == 'inherited':
        old = json.loads(Path(rec['v1_case_path']).read_text())
        matching = [(r, stage) for r in old['records'] for stage in ['initial', 'final']
                    if r[stage]['arrays'] == art['arrays']]
        assert matching
        selected_record, selected_stage = matching[0]
    perm = (selected_record['fit']['final_global_permutations']
            if selected_stage == 'final' else None)
    assert sha(rec['training']['path']) == rec['training']['sha256']
    with np.load(rec['training']['path'], allow_pickle=False) as z:
        assert z['split'].item() == 'train' and z['completed'].all()
        original_routes = z['routes'][:, :, rec['layer']].reshape(-1, rec['shape'][3])
        assert np.array_equal(routes, original_routes)
    fit = ExactFit(rows, rec['shape'], rec['service'], routes, kind, args.seed, perm)
    assert fit.initial_total == int(original_values.sum())
    plan, values, bounds, stats = fit.run(args.iterations, args.temperature)
    args.output.mkdir(parents=True, exist_ok=False)
    arrays = args.output / 'arrays.npz'
    physical = fit.backend.physical_addresses(plan, fit.service)
    addresses = np.asarray([[a[k] for k in ['chip', 'block', 'wordline', 'reserved_wordlines', 'pages', 'first_plane']]
                            for a in physical.pop('addresses')], np.int32)
    logical = np.asarray([[a[k] for k in ['expert', 'phase', 'chip', 'row_start', 'rows', 'columns']]
                          for a in plan['logical']], np.int32)
    np.savez_compressed(arrays, rows=fit.rows, initial_rows=rows, routes=routes,
                        initial_cost=original_values, cost=values, lower_bound=bounds,
                        offsets=plan['offsets'], jobs=plan['jobs'],
                        input_columns=plan['input_columns'], total_rows=plan['total_rows'],
                        addresses=addresses, logical=logical)
    checks = []
    for q in [0, len(routes) - 1]:
        direct = fit.backend.ReferenceGraph.build(plan, np.r_[routes[q], rec['shape'][2]], fit.service).run()
        assert direct['cost'] == int(values[q])
        checks.append(dict(query=q, cost_ns=int(values[q]), nodes=len(direct['trace'])))
    result = dict(case=str(args.case), case_sha256=sha(args.case),
                  protocol=str(args.protocol), protocol_sha256=sha(args.protocol),
                  family=args.family, kind=kind, seed=args.seed,
                  inherited=winner, shape=rec['shape'], service=rec['service'],
                  training=rec['training'], fit=stats, physical=physical,
                  mean_ns=float(values.mean()), p99_ns=float(np.percentile(values, 99)),
                  arrays=str(arrays), arrays_sha256=sha(arrays), checks=checks,
                  script_sha256=sha(__file__), scope='TRAIN only; standard finite local-search control; no device timing or algorithm novelty claim.')
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(family=args.family, fit=stats, mean_ns=result['mean_ns'],
                          p99_ns=result['p99_ns'])), flush=True)


if __name__ == '__main__':
    main()
