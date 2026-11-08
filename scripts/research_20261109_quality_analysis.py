#!/usr/bin/env python3
"""Audit the 64-token live CPU runs and compare paired article-heldout quality.

Coefficient/MAC totals are counterfactual deployment models, not measured IO
or latency. The harness still computes the native FFNs for verification.
"""
from collections import Counter
import argparse
import json
import math
from pathlib import Path

import numpy as np

H, I, K, L = 2048, 1408, 6, 26
NATIVE = 3 * H * I


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def registry(path):
    result = {}
    if str(path) == '-':
        return result
    for line in Path(path).read_text().splitlines():
        if not line or line.startswith('#'):
            continue
        layer, expert, basis, width, transform, head = line.split('\t')
        key = int(layer), int(expert)
        assert key not in result
        sources = np.fromfile(transform + '.sources.u8', dtype=np.uint8)
        proto_path = Path(transform + '.prototypes.i32')
        owners = np.fromfile(proto_path, dtype=np.int32) if proto_path.exists() else np.full(len(sources), -1)
        factor_path = Path(head + '.factor.json')
        factor_rank, factor_files = None, []
        if factor_path.exists():
            cfg = read(factor_path)
            factor_width = cfg.get('input_width', cfg.get('width'))
            assert cfg['version'] == 1 and cfg['H'] == H and factor_width == int(width)
            assert 'width' not in cfg or cfg['width'] == factor_width
            assert not cfg.get('V_bias', False)
            assert 'input_transform_prefix' not in cfg or cfg['input_transform_prefix'] == transform
            factor_rank = int(cfg['rank'])
            assert 0 < factor_rank <= 1408
            expected = {head + s for s in ('.V.i8', '.V.scales.f32', '.latent.center.f32',
                                           '.latent.inv_scale.f32', '.U.i8', '.U.scales.f32', '.bias.f32')}
            assert set(cfg['payload_sha256']) == expected
            factor_files = list(cfg['payload_sha256']) + [str(factor_path)]
            assert all(Path(p).is_file() for p in factor_files)
        result[key] = dict(width=int(width), basis=basis, transform=transform, head=head,
                           prototypes=set(map(int, owners[sources == 2])), factor_rank=factor_rank,
                           factor_payload_files=factor_files,
                           readout_coefficient_MAC=(factor_rank * (H + int(width))
                                                    if factor_rank is not None else H * int(width)))
        assert len(sources) == int(width) and all(e >= 0 for e in result[key]['prototypes'])
    return result


def phase(route, summary):
    if route['phase'] == 'prefill':
        return ('NLL_prefix'+str(summary['scored_tokens']) if route['token_in_batch'] < summary['scored_tokens']
                else 'last_prefix_for_first_generated')
    return ('decode_causal' if route['step'] + 1 < summary['actual_generated_length']
            else 'decode_last_unused')


def load_arm(folder, registry_path='-', expected_ids=None, hot_pools=None):
    folder = Path(folder)
    heads = registry(registry_path)
    summaries, positions, routes, exposure = {}, {}, {}, {}
    completions = list(folder.glob('**/completion.json'))
    assert completions, f'No completion manifests: {folder}'
    for completion in completions:
        done = read(completion)
        assert done['status'] == 'complete' and done['gpu_layers'] == 0
        limit=done['max_tokens']
        assert limit in (16,64)
        files = list(completion.parent.glob('*.summary.json'))
        assert len(files) == done['jobs']
        for path in files:
            s = read(path)
            key = s['prompt_id']
            assert key not in summaries
            assert s['input_tokens'] == limit and s['scored_tokens'] == limit-1
            assert s['decode_steps'] == done['decode_steps']
            assert len(s['generated_token_ids']) == s['actual_generated_length'] <= s['decode_steps']
            assert s['stopped_eog'] or s['actual_generated_length'] == s['decode_steps']
            assert s['decoded_generated_inputs'] == s['actual_generated_length'] - int(s['stopped_eog'])
            pp = list(lines(path.with_name(key + '.positions.jsonl')))
            assert [r['position'] for r in pp] == list(range(limit-1))
            assert [r['input_token_id'] for r in pp] == s['token_ids'][:-1]
            assert [r['target_token_id'] for r in pp] == s['token_ids'][1:]
            assert abs(sum(r['nll_F64'] for r in pp) - s['sum_nll_F64']) < 1e-10
            assert abs(s['sum_nll_F64'] / (limit-1) - s['mean_nll_F64']) < 1e-13
            seen, rr, ee = set(), {}, {}
            for route in lines(path.with_name(key + '.routing.jsonl')):
                p, step, token, layer = (route[a] for a in ('phase', 'step', 'token_in_batch', 'layer'))
                identity = p, step, token, layer
                assert identity not in seen and 1 <= layer <= L
                seen.add(identity)
                ids = route['expert_ids']
                assert len(ids) == len(set(ids)) == K and all(0 <= e < 64 for e in ids)
                if p == 'prefill':
                    assert step == -1 and 0 <= token < limit
                else:
                    assert p == 'decode' and token == 0 and 0 <= step < s['decoded_generated_inputs']
                bucket = phase(route, s)
                count = ee.setdefault(bucket, Counter())
                count['routed_expert_events'] += K
                count['native_coefficient_MAC'] += K * NATIVE
                selected = [heads[(layer, e)] for e in ids if (layer, e) in heads]
                count['replaced_events'] += len(selected)
                count['readout_coefficient_MAC'] += sum(m['readout_coefficient_MAC'] for m in selected)
                factors = [m for m in selected if m['factor_rank'] is not None]
                count['private_latent_A8_rows'] += len(factors)
                count['latent_affine_values'] += sum(m['factor_rank'] for m in factors)
                count['factorized_V_coefficient_MAC'] += sum(m['factor_rank'] * m['width'] for m in factors)
                count['factorized_U_coefficient_MAC'] += sum(m['factor_rank'] * H for m in factors)
                required = set().union(*(m['prototypes'] for m in selected)) if selected else set()
                if 'required_prototype_ids' in route:
                    assert set(route['required_prototype_ids']) == required
                    assert route['eligible_replacement_expert_ids'] == [e for e in ids if (layer,e) in heads]
                if 'factorized_replacement_expert_ids' in route:
                    assert route['factorized_replacement_expert_ids'] == [e for e in ids if (layer,e) in heads and heads[layer,e]['factor_rank'] is not None]
                count['required_prototype_rows'] += len(required)
                count['prototype_rows_already_naturally_selected'] += len(required.intersection(ids))
                count['incremental_prototype_GU_MAC'] += len(required.difference(ids)) * 2 * H * I
                count['coefficient_MAC_after_replacement_and_incremental_GU'] += (
                    (K - len(selected)) * NATIVE + sum(m['readout_coefficient_MAC'] for m in selected)
                    + len(required.difference(ids)) * 2 * H * I)
                count['unique_basis_A8_rows'] += len({m['basis'] for m in selected})
                count['cross_expert_events'] += sum(bool(m['prototypes']) for m in selected)
                if hot_pools:
                    hot = set(hot_pools.get(layer, []))
                    count['native_hot_GU_cache_hits'] += len(hot.intersection(ids))
                rr[identity] = tuple(ids)
            assert len(seen) == L * (limit + s['decoded_generated_inputs'])
            assert sum(x['replaced_events'] for x in ee.values()) == s['replaced_expert_token_events']
            assert s['native_shared_z_floats_bitchecked']==len({l for l,e in heads})*2816*(limit+s['decoded_generated_inputs'])
            if 'prototype_GU_row_pairs_computed' in s:
                assert sum(x['required_prototype_rows'] for x in ee.values()) == s['prototype_GU_row_pairs_computed']
                assert sum(x['prototype_rows_already_naturally_selected'] for x in ee.values())*I == s['prototype_owner_selected_phi_floats_bitchecked']
                assert sum(x['cross_expert_events'] for x in ee.values()) == s['prototype_cross_replaced_events']
                assert sum(x['unique_basis_A8_rows'] for x in ee.values()) == s['basis_A8_unique_rows']
            if 'private_latent_A8_rows' in s:
                assert sum(x['private_latent_A8_rows'] for x in ee.values()) == s['private_latent_A8_rows']
                assert sum(x['latent_affine_values'] for x in ee.values()) == s['factorized_latent_affine_values']
                assert sum(x['factorized_V_coefficient_MAC'] for x in ee.values()) == s['factorized_V_coefficient_MAC']
                assert sum(x['factorized_U_coefficient_MAC'] for x in ee.values()) == s['factorized_U_coefficient_MAC']
            summaries[key], positions[key], routes[key], exposure[key] = s, pp, rr, ee
    if expected_ids is not None:
        assert set(summaries) == set(expected_ids)
    assert len({tuple(s['token_ids']) for s in summaries.values()}) == len(summaries)
    return dict(summaries=summaries, positions=positions, routes=routes, exposure=exposure,
                registry=str(registry_path), heads=heads)


def aggregate(arm):
    result = {}
    for per_prompt in arm['exposure'].values():
        for phase_name, values in per_prompt.items():
            counter = result.setdefault(phase_name, Counter())
            counter.update(values)
    for values in result.values():
        values['event_replacement_fraction'] = values['replaced_events'] / values['routed_expert_events']
        values['coefficient_MAC_saving_fraction_including_incremental_GU'] = (
            1 - values['coefficient_MAC_after_replacement_and_incremental_GU'] / values['native_coefficient_MAC'])
    return result


def restrict(arm, ids):
    """Select an explicit preregistered subset of an audited reference arm."""
    ids=set(ids)
    assert ids and ids <= set(arm['summaries'])
    return {**arm, **{field:{key:arm[field][key] for key in ids}
                     for field in ('summaries','positions','routes','exposure')}}


def compare(candidate, native, seed=20261109, bootstrap=10000, compare_to_native=True, article_groups=None):
    a, b = candidate['summaries'], native['summaries']
    assert set(a) == set(b)
    ids = sorted(a)
    assert all(a[i]['token_ids'] == b[i]['token_ids'] for i in ids)
    delta = np.asarray([a[i]['mean_nll_F64'] - b[i]['mean_nll_F64'] for i in ids])
    if article_groups is None:
        sums,counts=delta,np.ones(len(ids))
    else:
        names=sorted({article_groups[i] for i in ids})
        sums=np.array([sum(delta[j] for j,i in enumerate(ids) if article_groups[i]==name) for name in names])
        counts=np.array([sum(article_groups[i]==name for i in ids) for name in names])
    rng = np.random.default_rng(seed)
    draws = np.empty(bootstrap)
    for start in range(0, bootstrap, 100):
        ix=rng.integers(len(sums),size=(min(100,bootstrap-start),len(sums)))
        draws[start:start+100] = sums[ix].sum(axis=1)/counts[ix].sum(axis=1)
    lo, hi = map(float, np.quantile(draws, [.025, .975]))
    margin = math.log(1.02)
    exact, matching, length, top1, route_equal, route_total = 0, 0, 0, 0, 0, 0
    for key in ids:
        aa, bb = a[key]['generated_token_ids'], b[key]['generated_token_ids']
        exact += aa == bb
        length += max(len(aa), len(bb))
        matching += sum(j < len(aa) and j < len(bb) and aa[j] == bb[j] for j in range(max(len(aa), len(bb))))
        top1 += sum(x['top1_token_id'] == y['top1_token_id'] for x, y in zip(candidate['positions'][key], native['positions'][key]))
        for identity, experts in candidate['routes'][key].items():
            if identity[0] == 'prefill':
                route_equal += experts == native['routes'][key][identity]
                route_total += 1
    scored=sum(s['scored_tokens'] for s in a.values())
    return dict(prompts=len(ids), scored_targets=scored,
                native_subset_PPL=math.exp(np.mean([b[i]['mean_nll_F64'] for i in ids])),
                candidate_subset_PPL=math.exp(np.mean([a[i]['mean_nll_F64'] for i in ids])),
                mean_delta_nll=float(delta.mean()), ci95_delta_nll=[lo, hi],
                relative_subset_PPL_change=math.expm1(float(delta.mean())),
                ci95_relative_subset_PPL_change=[math.expm1(lo), math.expm1(hi)],
                margin_relative_subset_PPL=.02 if compare_to_native else None,
                verdict=('pass' if hi <= margin else ('fail' if lo > margin else 'inconclusive'))
                if compare_to_native else 'descriptive_secondary_comparison',
                bootstrap=dict(replicates=bootstrap, unit='article',articles=len(sums),seed=seed,
                               one_prompt_per_article=article_groups is None),
                prefill_top1_agreement=top1/scored,
                prefill_routing_slot_order_agreement=route_equal/route_total,
                generation_exact_prompt_fraction=exact/len(ids) if length else None,
                generation_positional_agreement=matching/length if length else None,
                generation_is_task_accuracy=False,
                exposure=aggregate(candidate))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--registry', default='-')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    native = load_arm(args.native)
    candidate = load_arm(args.candidate, args.registry, native['summaries'])
    result = compare(candidate, native)
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
