#!/usr/bin/env python3
"""Synthetic-only pre-data contract checks. No model/prompt access."""
import sys
import tempfile
import numpy as np
from research_20261103_local_affine_stage0_core import *
from research_20261103_local_affine_stage0_pipeline import score_point


def main(path):
    num = Numeric(path)
    rng = np.random.default_rng(SEED)
    checks = {}
    # F32 quotient followed by exact-away rounding; adversarial neighbors of
    # every half must agree with C++, while ties-even is demonstrably different.
    h = np.arange(-127, 128, dtype=np.float32) + np.float32(.5)
    v = np.concatenate([h, np.nextafter(h, np.float32(-np.inf)), np.nextafter(h, np.float32(np.inf)), rng.standard_normal(10000).astype(np.float32)])
    out = np.empty_like(v)
    num.lib.la_round(num.ptr(v), len(v), num.ptr(out))
    expected = np.copysign(np.floor(np.abs(v.astype(np.float64)) + .5), v).astype(np.float32)
    assert np.array_equal(out.view(np.uint32), expected.view(np.uint32))
    assert np.any(np.rint(h) != out[:len(h)])
    for scale in [1e-20, 1., 1e20]:
        a = (rng.uniform(-127, 127, (5, 2048)) * scale).astype(np.float32)
        a[:, 0] = np.float32(127 * scale)
        q, sx = num.quantize(a)
        ratio = np.divide(a, sx[:, None], dtype=np.float32)
        ref = np.copysign(np.floor(np.abs(ratio.astype(np.float64)) + .5), ratio)
        assert np.array_equal(q, np.clip(ref, -127, 127).astype(np.int8))
    q, sx = num.quantize(np.zeros((3, 64), np.float32))
    assert not np.any(q) and np.all(sx == 1)
    checks['near_half_cpp_python_round_cases'] = len(v)
    for width in [64, 128, 512, 1024, 1408, 2048, 10944]:
        q = rng.integers(-127, 128, (7, width), dtype=np.int16).astype(np.int8)
        w = rng.integers(-127, 128, (9, width), dtype=np.int16).astype(np.int8)
        q[0] = 127; q[1] = -127; w[0] = 127; w[1] = -127
        q[2, ::2] = 127; q[2, 1::2] = -127
        assert np.array_equal(num.dots(q, w), q.astype(np.int64) @ w.astype(np.int64).T)
    checks['DGEMM_signed_worst_random_widths'] = [64, 128, 512, 1024, 1408, 2048, 10944]
    for I in [1024, 1408]:
        weights = {}
        for role, R, C in [('gate', I, 2048), ('up', I, 2048), ('down', 2048, I)]:
            weights[role] = num.quantize(rng.normal(0, .02, (R, C)).astype(np.float32))
        y = teacher(num, rng.standard_normal((3, 2048)).astype(np.float32), weights)
        assert y.shape == (3, 2048) and np.isfinite(y).all()
    checks['synthetic_full_teacher_intermediates'] = [1024, 1408]
    x = rng.standard_normal((256, 8)).astype(np.float32)
    x[32:64] = x[0]
    tree = fit_tree(num, x, 7)
    for C in [1, 4, 16, 32]:
        ids = select_tree(num, tree, x, C)
        assert np.all(ids[32:64] == ids[0])
        assert np.array_equal(select_tree(num, tree, x[::-1], C)[::-1], ids)
    identical = np.ones((64, 8), np.float32)
    t = fit_tree(num, identical, 9)
    assert np.all(select_tree(num, t, identical, 32) == -1)
    checks['runtime_tree_pure_input_ties_and_unsplittable'] = True
    basis, sigma = fit_projection(x)
    for fmt in ['normal', 'balanced']:
        d = design(num, x, tree, 4, basis, sigma, 4, fmt)
        assert np.array_equal(d['v'], d['sx'].astype(np.float64)[:, None] * d['q'].astype(np.float64))
    d = design(num, x, tree, 4, basis, sigma, 8, 'identity')
    assert d['P'] is None
    # Same-leaf demean recovers a common planted slope despite disparate means.
    v = rng.standard_normal((200, 4))
    leaf = np.repeat(np.arange(4), 50)
    A = rng.standard_normal((4, 7))
    y = (v @ A + np.array([100., -200., 400., -800.])[leaf, None]).astype(np.float32)
    weights = np.where(np.arange(200) % 3, .25, 1.)
    cov, cross, local = statistics(v, y, leaf, weights, 4, np.arange(200) // 5)
    fit, info = solve(cov, cross, 0.)
    assert np.max(np.abs(fit - A)) < 1e-5
    inf, _ = solve(cov, cross, 'infinity')
    assert np.all(inf == 0)
    prior = rng.standard_normal(A.shape)
    l, _ = solve(cov, cross, 'infinity', prior)
    assert np.array_equal(l, prior)
    zero, _ = solve(np.zeros((4, 4)), np.zeros((4, 7)), 0., prior)
    assert np.array_equal(zero, prior)
    singular = np.diag([2., 1., 0., 0.])
    fit, _ = solve(singular, np.ones((4, 7)), 0., prior)
    assert np.array_equal(fit[2:], prior[2:])
    checks['weighted_within_leaf_ridge_infinity_trace0_min_norm'] = True
    # Actual integer/F32 deployment and residual recalibration.
    q, sx = num.quantize(v.astype(np.float32))
    df = {'q': q, 'sx': sx, 'v': sx.astype(np.float64)[:, None] * q, 'leaf': leaf}
    model = lower(num, [A] * 4, df, y, weights, 4)
    p = predict(num, model, df)
    assert np.isfinite(p).all() and len(set(model['map_hashes'])) == 1
    for c in range(4):
        ix = np.flatnonzero(leaf == c)
        pre = num.linear(q[ix], sx[ix], model['qmaps'][c], model['scales'][c])
        sums = np.zeros(7, np.float64)
        sw = 0.
        for j in ix:
            sums = sums + weights[j] * (y[j].astype(np.float64) - pre[np.flatnonzero(ix == j)[0]].astype(np.float64))
            sw += weights[j]
        assert np.array_equal(model['bias'][c].view(np.uint32), (sums / sw).astype(np.float32).view(np.uint32))
    checks['actual_A8_map_lowering_bias_and_map_dedup'] = True
    assert quantile_threshold(np.array([0., 1., 2., 3.]), np.ones(4, bool), 50) == 1.5
    model['map_hashes'] = ['nonzero', 'zero', 'nonzero', 'zero']
    use = np.ones(200, bool); covered = np.ones(200, bool)
    co = cost(model, 4, 4, 7, use, covered, leaf, 16, 'normal')
    assert co['active_coefficient_macs'] == 14
    assert cost(model, 7, 4, 7, use, covered, leaf, 16, 'normal')['host_projection_macs'] == 49
    assert cost(model, 7, 4, 7, use, covered, leaf, 16, 'identity')['host_projection_macs'] == 0
    checks['zero_leaf_work_fullH_format_cost_and_linear_guard'] = True
    ids = [str(i) for i in range(40)]
    a = {'sse': list(np.arange(40) + 1.), 'energy': [100.] * 40}
    b = {'sse': list(np.arange(40) * .5), 'energy': [100.] * 40}
    comp = [{'id': 'a-b', 'candidate': 'a', 'control': 'b'}]
    boot = bootstrap(comp, {'a': a, 'b': b}, ids, set())
    assert not boot['family_inconclusive'] and boot['max_t_cutoff95'] > 0
    zero_sd = bootstrap(comp, {'a': {'sse': [1.] * 40, 'energy': [1.] * 40}, 'b': {'sse': [0.] * 40, 'energy': [1.] * 40}}, ids, set())
    assert zero_sd['family_inconclusive']
    undefined = bootstrap(comp, {'a': {'sse': [1.] + [0.] * 39, 'energy': [1.] + [0.] * 39}, 'b': b}, ids, set())
    assert undefined['family_inconclusive'] and undefined['comparisons'][0]['undefined_bootstrap_draws'] > 0
    literal = bootstrap(comp, {'a': a, 'b': a}, ids, {'a-b'})
    assert literal['comparisons'][0]['simultaneous_band'] == [0., 0.]
    checks['paired40_maxT_undefined_no_redraw_nonliteral_SD0_literal_tie'] = True
    ds = {'prompt': np.arange(40), 'phase': np.zeros(40, np.int8)}
    metric = score_point(np.ones((40, 2), np.float32), np.zeros((40, 2), np.float32), ds, np.ones(40, bool), np.ones(40), np.arange(40) < 20)
    assert sum(metric['prefill']['natural']['sse']) == 40 and sum(metric['prefill']['natural']['covered_hits']) == 20
    checks['fallback_error_zero_paid_coverage_and_empty_decode'] = True
    print(json.dumps({'status': 'passed', 'checks': checks, 'audited_integer_dot_entries': num.dot_audits}, indent=2))


if __name__ == '__main__':
    main(sys.argv[1])
