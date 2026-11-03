#!/usr/bin/env python3
"""Train-only optimistic bound for native dense-GEMV block-diagonal fringes.

Candidate scope is exactly 11 full-wave + 4 quarter-wave + 1 empty chip per
expert. Pair two quarter-wave rows with structural zeros, using one larger
input vector and an ordinary dense GEMV. This is a screen, not a physical fit.
The fractional degree-4 matching ignores chip realizability and therefore
upper-bounds expected coactive fringe pairs of every fixed admissible layout.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'results/research-2026-11-02/round2/geometry-study/granite4tiny.frozen.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def solve_bound(weights, edges, incidence, degree=4):
    result = linprog(-weights, A_ub=incidence,
                     b_ub=np.full(incidence.shape[0], degree),
                     bounds=(0, None), method='highs')
    assert result.success, result.message
    # Explicit feasible dual certificate. Any residual roundoff is repaired by
    # increasing all vertex prices; every edge contains exactly two vertices.
    prices = np.maximum(-result.ineqlin.marginals, 0)
    deficit = max(0.0, float(np.max(weights - prices[edges].sum(axis=1))))
    prices += deficit / 2 + 1e-12
    upper = degree * float(prices.sum())
    slack = prices[edges].sum(axis=1) - weights
    assert float(slack.min()) >= -1e-12
    assert float(np.max(incidence @ result.x)) <= degree + 1e-8
    assert upper + 1e-8 >= -float(result.fun)
    return dict(expected_pairs_upper=upper,
                primal_expected_pairs=-float(result.fun),
                dual_prices=prices.tolist(),
                dual_min_edge_slack=float(slack.min()),
                dual_primal_gap=upper + float(result.fun),
                max_vertex_degree=float(np.max(incidence @ result.x)),
                nonzero_fractional_edges=int(np.count_nonzero(result.x > 1e-9)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frozen', type=Path, default=DEFAULT)
    parser.add_argument('--out', type=Path, default=DEFAULT.parent / 'blockdiag_bound.json')
    args = parser.parse_args()
    frozen = json.loads(args.frozen.read_text())
    routes_path = Path(frozen['training_routes'])
    assert sha(routes_path) == frozen['training_routes_sha256']
    with np.load(routes_path) as data:
        routes = data['routes']
        layers = data['layers'].tolist()
        ids = data['prompt_ids'].tolist()
    assert all(x.startswith('train_') for x in ids)
    assert ids == frozen['train_ids']
    experts, active = frozen['experts'], frozen['active']
    assert (experts, active, routes.shape) == (64, 6, (32, 64, 40, 6))
    edges = np.asarray(list(itertools.combinations(range(experts), 2)))
    incidence = coo_matrix((np.ones(2 * len(edges)),
                           (edges.T.reshape(-1), np.tile(np.arange(len(edges)), 2))),
                          shape=(experts, len(edges))).tocsr()
    # An explicit fixed top-6 query admits a degree-four graph with 12 edges.
    fixed_weights = ((edges[:, 0] < 6) & (edges[:, 1] < 6)).astype(float)
    fixed_check = solve_bound(fixed_weights, edges, incidence)
    assert abs(fixed_check['expected_pairs_upper'] - 12) < 1e-7
    uniform_w = active * (active - 1) / (experts * (experts - 1))
    uniform_pairs = experts * 4 / 2 * uniform_w
    uniform_check = solve_bound(np.full(len(edges), uniform_w), edges, incidence)
    assert abs(uniform_check['expected_pairs_upper'] - uniform_pairs) < 1e-7
    pair_indices = np.asarray(list(itertools.combinations(range(active), 2)))
    phases = {p['layer']: p for p in frozen['phases'] if p['name'] == 'down'}
    results = []
    for li, layer in enumerate(layers):
        selected = routes[:, :, li].reshape(-1, active).astype(np.int64)
        assert np.all(np.diff(np.sort(selected, axis=1), axis=1) > 0)
        a = selected[:, pair_indices[:, 0]].reshape(-1)
        b = selected[:, pair_indices[:, 1]].reshape(-1)
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        counts = np.bincount(lo * experts + hi, minlength=experts**2).reshape(experts, experts)
        assert int(counts.sum()) == len(selected) * 15
        weights = counts[edges[:, 0], edges[:, 1]] / len(selected)
        lp = solve_bound(weights, edges, incidence)
        rows = np.asarray(phases[layer]['rows'], dtype=np.int64)
        waves = (rows * 512 + 65535) // 65536
        g12_max = waves[selected].sum(axis=1).max(axis=1)
        assert np.isin(g12_max, [5, 6]).all()
        p5_upper = min(1.0, lp['expected_pairs_upper'] / 10)
        nand_lower = 6 - p5_upper
        results.append(dict(layer=layer, samples=len(selected), **lp,
                            optimistic_p5_upper=p5_upper,
                            optimistic_mean_maxwave_lower=nand_lower,
                            frozen_g12_mean_maxwave=float(g12_max.mean()),
                            frozen_g12_p5=float(np.mean(g12_max == 5)),
                            nand_wave_lower_gap_vs_g12=nand_lower-float(g12_max.mean())))
    new_lb = sum(r['optimistic_mean_maxwave_lower'] for r in results)
    old = sum(r['frozen_g12_mean_maxwave'] for r in results)
    document = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        study='Optimistic native block-diagonal quarter-fringe bound',
        source=str(Path(__file__).resolve()), source_sha256=sha(__file__),
        frozen_path=str(args.frozen.resolve()), frozen_sha256=sha(args.frozen),
        training_routes_path=str(routes_path), training_routes_sha256=sha(routes_path),
        heldout_opened=False, actual_weights_opened=False,
        hardware_change=False, fitted_physical_layout=False,
        candidate=dict(input_columns=512, full_wave_rows=128, quarter_rows=32,
                       full_wave_tiles_per_expert=11, quarter_tiles_per_expert=4,
                       empty_chips_per_expert=1, ordinary_wave_ns=28000,
                       dense_pair_shape=[64,1024], dense_pair_physical_pages=4,
                       pairs_inventory=128, structural_zero_bytes_per_layer=4194304,
                       down_unique_weight_bytes_per_layer=50331648,
                       zero_overhead_fraction=1/12),
        derivation=[
            'There are 15 per-expert chip jobs before fringe coactivation; top6 gives 90 jobs.',
            'A selected pair in the fixed quarter-fringe pairing graph saves exactly one job. Each expert has four fringes, so its graph degree is four; repeated expert pairs are permitted.',
            'For max chip load <=5, total jobs must be <=16*5=80, requiring at least ten coactive pairs on that query.',
            'Thus P(max_load<=5) <= E[pairs]/10 by Markov. Also max_load>=5 since even twelve coactive pairs leave 78 jobs. E[max_load]>=6-E[pairs]/10.',
            'The degree-at-most-four fractional matching LP maximizes expected coactive pairs using training pair frequencies. Dropping integer, chip, and placement constraints makes it an upper bound.',
            'Explicit nonnegative dual vertex prices certify each edge weight <= sum of endpoint prices; four times sum of prices is an expected-pairs upper bound.',
            'This lower bound ignores block splitting, ECC/MAC work on zeros, inactive-row outputs, input concatenation, inactive partner input slots and gang readiness. They cannot improve this NAND lower bound.'
        ],
        uniform_control=dict(expected_coactive_pairs=uniform_pairs,
                             optimistic_p5_upper=uniform_pairs/10,
                             optimistic_mean_maxwave_lower=6-uniform_pairs/10,
                             reason='Every four-regular multigraph has 128 edges; uniform top6 coactivation frequency is 30/(64*63).'),
        controls=dict(fixed_top6_expected_pairs_upper=fixed_check['expected_pairs_upper'],
                      uniform_expected_pairs_upper=uniform_check['expected_pairs_upper']),
        aggregate=dict(layers=len(results),
                       candidate_optimistic_sum_mean_maxwave_lower=new_lb,
                       frozen_g12_sum_mean_maxwave=old,
                       candidate_optimistic_nand_lower_ms=new_lb*.028,
                       frozen_g12_nand_ms=old*.028,
                       nand_lower_gap_ms=(new_lb-old)*.028,
                       nand_lower_ratio_vs_g12=new_lb/old,
                       candidate_worse_bound_layers=sum(r['nand_wave_lower_gap_vs_g12']>0 for r in results)),
        verdict=('NO-GO for this exact 11-full/4-quarter/1-empty family on the frozen training screen: even the relaxed NAND bound loses to frozen g12.'
                 if new_lb>old else
                 'Bound does not rule out the candidate; this is not positive evidence of a realizable layout.'),
        scope_limitations=[
            'Not a bound on arbitrary different shard geometries, three-way block-diagonal joins or arbitrary replicated layouts.',
            'Not actual PE, NAND, ONFI or host timing; only the ordinary-wave lower bound.',
            'Does not select layouts or inspect heldout data.',
            'Input concatenation and structural-zero representation use algebraically ordinary dense GEMV, but exact native dimension and output correctness remain a separate implementation condition.'
        ], layers=results)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(document,indent=2)+'\n')
    print(json.dumps({'output':str(args.out),'aggregate':document['aggregate'],
                      'uniform':document['uniform_control'],'verdict':document['verdict']},indent=2))

if __name__=='__main__':
    main()
