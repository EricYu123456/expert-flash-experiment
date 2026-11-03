#!/usr/bin/env python3
"""Reproducible selection sensitivity for the fixed-copy marginal baselines.

Uses the same first/last halves of training and exact selector as the parent
screen.  Adds train p99 and top-1%-CVaR selection so a mean-selected conventional
baseline is not silently treated as the strongest available p99 comparator.
"""
import gzip
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'scripts'))
import research_20261102_interference as study
import aif_model as flash
from research_20261102_projection_study import model_phases
from run_corpus_traces import DATA, geometry, sha256


def summaries(values):
    return (float(np.mean(values)), float(np.quantile(values, .99)),
            float(np.sort(values)[-max(1, int(np.ceil(.01 * len(values)))):].mean()))


def main():
    records = []
    for name in ('granite', 'olmoe', 'deepseek', 'qwen3', 'mixtral'):
        metadata = json.loads((DATA / 'models' / (name + '-metadata.json')).read_text())
        layers, experts, active = geometry(metadata)
        dims = flash.tensor_dimensions(metadata)
        with np.load(DATA / 'runs/corpus-300-v1' / name / 'analysis/train_routes.npz') as cache:
            routes = cache['routes']
        with np.load(study.DEST / (name + '.all_layer_baseline_tokens.npz')) as saved:
            uniforms = saved['nuisance_uniforms']
            originals = {key: saved[key] for key in saved.files if key.startswith('b')}
        plans = []
        for li, layer in enumerate(layers):
            fit = np.sort(routes[:100, :, li].reshape(-1, active), axis=1)
            indexes = np.sort(np.random.default_rng(study.SEED + li).choice(len(fit), 512, replace=False))
            fit = fit[indexes]
            frequencies = np.bincount(fit.ravel(), minlength=experts).astype(float) / len(fit)
            for phase, rows, columns, common in model_phases(dims, layer):
                tiles, q, grain = study.canonical_tiles(rows, columns)
                by_budget = {}
                for budget in (.1, .25, .5, 1):
                    quota = min(experts * len(tiles), int(budget * experts * q) // grain)
                    masks = study.marginal_masks(experts, len(tiles), quota, frequencies)
                    candidates = {}
                    scores = {}
                    for rule, mask in masks.items():
                        for cseed in (0, 1):
                            key = rule + '_color' + str(cseed)
                            candidate = study.balanced_colors(mask, frequencies, cseed)
                            candidates[key] = candidate
                            extra = study.completion_extra(study.copies_to_counts(candidate, fit),
                                       active * len(tiles), grain, 3000000)
                            scores[key] = summaries(extra)
                    selected = dict(
                        mean=min(scores, key=lambda n: (scores[n][0], n)),
                        p99=min(scores, key=lambda n: (scores[n][1], scores[n][2], scores[n][0], n)),
                        cvar99=min(scores, key=lambda n: (scores[n][2], scores[n][1], scores[n][0], n)))
                    by_budget[str(budget)] = dict(selected=selected, scores=scores,
                        colors={metric: candidates[rule].tolist() for metric, rule in selected.items()})
                plans.append(dict(layer=layer, layer_index=li, phase=phase, grain=grain,
                                  base_tiles=active * len(tiles), budgets=by_budget))
        frozen_path = study.DEST / 'plans' / (name + '.p99_selected_baselines.json.gz')
        with gzip.GzipFile(filename=str(frozen_path), mode='wb', mtime=0) as stream:
            stream.write(json.dumps(plans, separators=(',', ':')).encode())
        sums = {}
        original_base = json.loads((study.DEST / (name + '.token_baselines.json')).read_text())['records'][0]['original_ffn_ms']
        base_ns = int(round(original_base * 1e6))
        for budget in (.1, .25, .5, 1):
            for metric in ('mean', 'p99', 'cvar99'):
                for stall in (3000000, 5000000):
                    for probability in (.0001, .001, .01):
                        sums[(budget, metric, stall, probability)] = np.full((100, 128), base_ns, dtype=np.int64)
        for pi, plan in enumerate(plans):
            selected = np.sort(routes[100:, :, plan['layer_index']].reshape(-1, active), axis=1)
            for budget in (.1, .25, .5, 1):
                for metric in ('mean', 'p99', 'cvar99'):
                    colors = np.asarray(plan['budgets'][str(budget)]['colors'][metric], dtype=int)
                    counts = study.copies_to_counts(colors, selected)
                    for stall in (3000000, 5000000):
                        extra = study.completion_extra(counts, plan['base_tiles'], plan['grain'], stall).reshape(100, 128)
                        for probability in (.0001, .001, .01):
                            sums[(budget, metric, stall, probability)] += extra * (uniforms[:, :, pi] < probability)
        for (budget, metric, stall, probability), array in sums.items():
            original = originals[f'b{budget:g}_strongest_marginal_d{stall}_p{probability:g}']
            no_replica = originals[f'b{budget:g}_no_replica_d{stall}_p{probability:g}']
            if metric == 'mean':
                np.testing.assert_array_equal(array, original)
            records.append(dict(model=name, capacity=budget, training_selector=metric,
                stall_ns=stall, nuisance_probability_per_phase=probability,
                mean_ms=float(array.mean() / 1e6), p99_ms=float(np.quantile(array, .99) / 1e6),
                mean_gain_vs_no_replica_pct=float(100 * (1 - array.mean() / no_replica.mean())),
                p99_gain_vs_no_replica_pct=float(100 * (1 - np.quantile(array, .99) / np.quantile(no_replica, .99))),
                p99_gain_vs_original_mean_selected_pct=float(100 * (1 - np.quantile(array, .99) /
                                                                          np.quantile(original, .99))),
                frozen_plan_sha256=sha256(frozen_path)))
        print(name, 'p99 selection control complete', flush=True)
        study.write_json(study.DEST / (name + '.p99_selection_control.json'),
                         [r for r in records if r['model'] == name])
    study.write_json(study.DEST / 'p99_selection_control.json', dict(
        records=records, source_sha256=sha256(Path(__file__)),
        parent_source_sha256=sha256(ROOT / 'scripts/research_20261102_interference.py'),
        uses_original_test=False, selection_on_fit_only=True,
        exactly_reproduces_original_mean_selected_token_arrays=True))


if __name__ == '__main__':
    main()
