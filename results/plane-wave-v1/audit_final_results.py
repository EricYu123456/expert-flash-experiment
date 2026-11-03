#!/usr/bin/env python3
"""Read-only final artifact integrity and held-out baseline reproduction audit."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import aif_model as flash
import corpus_analysis as previous

OUT = ROOT / 'results/plane-wave-v1'
MODELS = ('granite', 'olmoe', 'deepseek', 'qwen3', 'mixtral')


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            value.update(chunk)
    return value.hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def audit_source_hashes(manifest):
    for source, expected in manifest['software_sha256'].items():
        assert digest(ROOT / 'scripts' / source) == expected, source


def main():
    plane_manifest = read_json(OUT / 'manifest.json')
    bundle_manifest = read_json(OUT / 'static-bundles/manifest.json')
    provenance = read_json(OUT / 'audit_trace_provenance.json')
    audit_source_hashes(plane_manifest)
    audit_source_hashes(bundle_manifest)
    assert plane_manifest['corpus_sha256'] == bundle_manifest['corpus_sha256'] == provenance['corpus_sha256']
    provenance_caches = {(r['model'], r['split']): r['cache_sha256'] for r in provenance['checks']}
    receipts = []
    for name in MODELS:
        plan_path = OUT / (name + '.frozen_plan.json')
        plan = read_json(plan_path)
        validation = read_json(OUT / (name + '.validation.json'))
        audit_source_hashes(plan)
        assert digest(plan_path) == validation['frozen_plan_sha256']
        assert not plan['test_routes_loaded_before_freeze'] and not plan['test_routes_used_for_selection']
        assert plan['training_cache']['sha256'] == provenance_caches[name, 'train']
        assert validation['test_cache']['sha256'] == provenance_caches[name, 'test']
        assert set(plan['training_prompt_ids']).isdisjoint(validation['test_cache']['prompt_ids'])
        source = bundle_manifest['sources'][name]
        old = read_json(Path(source['old_plan']))
        metadata = read_json(Path(source['metadata']))
        assert digest(Path(source['old_plan'])) == source['old_plan_sha256'] == plan['old_frozen_plan_sha256']
        assert digest(Path(source['metadata'])) == source['metadata_sha256']
        assert metadata['sha256'] == plan['model_sha256']
        assert plan['mappings'] == old['mappings']
        with np.load(Path(source['test_cache']), allow_pickle=False) as cache:
            routes, layers = cache['routes'], cache['layers'].tolist()
            assert cache['prompt_ids'].tolist() == validation['test_cache']['prompt_ids']
        assert digest(Path(source['test_cache'])) == source['test_cache_sha256'] == provenance_caches[name, 'test']
        assert digest(Path(source['train_cache'])) == source['train_cache_sha256'] == provenance_caches[name, 'train']
        mask_checks = 0
        for g in (1, 2, 4, 8, 16):
            ngroups = 16 // g
            for width in (4, 2, 1):
                for layer, mapping in plan['mappings'][f'g{g}'].items():
                    bins = plan['train_only_plane_assignments'][f'g{g}'][f'w{width}'][layer]
                    assert len(bins) == len(mapping)
                    assert all(0 <= b < 4 // width for b in bins)
                    for group in range(ngroups):
                        resident = [e for e, slot in enumerate(mapping) if slot % ngroups == group]
                        actual = Counter(bins[e] for e in resident)
                        initial = Counter((mapping[e] // ngroups) % (4 // width) for e in resident)
                        assert actual == initial
                        mask_checks += 1
        # Recompute the old ideal-overlap baseline from the independent previous
        # vector model and compare every held-out token/component with saved NPZ.
        g = plan['old_winner']['group_chips']
        dimensions = flash.tensor_dimensions(metadata)
        expected = previous.evaluate_routes(routes, layers, dimensions, g, True, old['mappings'][f'g{g}'])
        with np.load(OUT / (name + '.token_costs.npz'), allow_pickle=False) as costs:
            assert costs['prompt_ids'].tolist() == validation['test_cache']['prompt_ids']
            for component in (*previous.COMPONENTS, 'model_ns'):
                np.testing.assert_array_equal(costs['ideal_overlap__old_frozen_baseline__' + component], expected[component])
            for timing in ('ideal_overlap', 'retained_cr_schedule', 'conservative_drain'):
                candidates = [r for r in plan['training_candidates'] if r['timing'] == timing]
                winner = min(candidates, key=lambda r: (r['mean_ms_per_token'], -r['plane_width'], r['group_chips']))
                assert winner == plan['winners'][timing]['joint']
                winner_label = f'width{winner["plane_width"]}_trainselected'
                np.testing.assert_array_equal(costs[timing + '__selected_joint__model_ns'],
                                              costs[timing + '__' + winner_label + '__model_ns'])
        # The bundle freeze is written before its test cache is opened.
        bundle_path = OUT / 'static-bundles' / (name + '.frozen_plan.json')
        bundle = read_json(bundle_path)
        assert digest(bundle_path) == source['frozen_plan_sha256']
        assert bundle['source_frozen_plan_sha256'] == source['old_plan_sha256']
        assert bundle['training_cache_sha256'] == provenance_caches[name, 'train']
        assert bundle['training_only'] and not bundle['test_routes_opened_before_freeze']
        assert set(bundle['training_ids']) == set(plan['training_prompt_ids'])
        assert set(bundle['training_ids']).isdisjoint(source['test_ids'])
        assert bundle['winner'] == min(bundle['candidates'],
                                       key=lambda r: (r['training_ms'], r['bundle_size'], r['g']))
        bundle_checks = 0
        for tag, labels_by_layer in bundle['labels'].items():
            g, size = (int(field[1:]) for field in tag.split('_'))
            groups = 16 // g
            for layer, labels in labels_by_layer.items():
                mapping = old['mappings'][f'g{g}'][layer]
                assert len(labels) == len(mapping)
                members = {}
                for expert, label in enumerate(labels):
                    members.setdefault(label, []).append(expert)
                assert sorted(e for experts in members.values() for e in experts) == list(range(len(mapping)))
                for experts in members.values():
                    assert 1 <= len(experts) <= size
                    assert len({mapping[e] % groups for e in experts}) == 1
                    bundle_checks += 1
        capacities = read_json(OUT / 'static-bundles' / (name + '.capacity.json'))
        assert len({r['unique_weight_bytes'] for r in capacities.values()}) == 1
        assert all(not r['weights_replicated'] for r in capacities.values())
        receipts.append(dict(model=name, heldout_prompts=routes.shape[0], heldout_tokens_per_prompt=routes.shape[1],
                             every_saved_baseline_component_bit_exact=True,
                             frozen_source_and_route_hashes_match=True,
                             train_only_winners_reproduced=True,
                             fixed_mask_group_capacity_checks=mask_checks,
                             fixed_bundle_membership_checks=bundle_checks,
                             bundle_unique_capacity_preserved=True))
    target = OUT / 'audit_final_results.json'
    target.write_text(json.dumps(dict(audit_scope='Saved artifacts, independent previous baseline, frozen hashes and fixed memberships',
                                     models=receipts), indent=2) + '\n')
    print(json.dumps(dict(models=len(receipts), heldout_prompt_model_pairs=sum(r['heldout_prompts'] for r in receipts),
                          heldout_tokens=sum(r['heldout_prompts'] * r['heldout_tokens_per_prompt'] for r in receipts),
                          result=str(target)), indent=2))


if __name__ == '__main__':
    main()
