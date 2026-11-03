#!/usr/bin/env python3
"""Reproduce fixed-g16 held-out plane diagnostics from existing frozen plans.

Run from any directory, for example:
  python3 scripts/evaluate_fixed_g16_planes.py --results results/plane-wave-v1

The results directory must contain the main study's manifest and five frozen
plans. Its manifest supplies data/cache locations, models, seeds and bootstrap
count. This wrapper performs no training or configuration selection and does
not change the main manifest, study sources or frozen plans. It rewrites only
the fixed_g16_* numerical outputs and fixed_g16_diagnostics.json provenance.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

import aif_model as flash
import run_plane_wave_study as study
from run_corpus_traces import ROOT, geometry, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=ROOT / 'results/plane-wave-v1',
                        help='Main study output directory containing manifest.json and frozen plans.')
    args = parser.parse_args()
    out = args.results.resolve()
    manifest_path = out / 'manifest.json'
    manifest_hash = sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    assert study.source_hashes() == manifest['software_sha256']
    rows = [json.loads(line) for line in
            (Path(manifest['configuration']['run_dir']) / 'corpus/prompts.jsonl').read_text().splitlines()]
    heldout = [r for r in rows if r['split'] == 'test']
    summary, prompts, sensitivity, arrays, provenance = [], [], [], {}, []
    started = time.time()
    for name in manifest['models']:
        plan_path = out / (name + '.frozen_plan.json')
        frozen_hash = sha256(plan_path)
        plan = json.loads(plan_path.read_text())
        metadata_path = Path(manifest['configuration']['data']) / 'models' / (name + '-metadata.json')
        metadata = json.loads(metadata_path.read_text())
        layers, experts, active = geometry(metadata)
        dims = flash.tensor_dimensions(metadata)
        routes, cache = study.load_cache(
            Path(manifest['configuration']['run_dir']) / name / 'analysis/test_routes.npz',
            heldout, layers, experts, active)
        values = {}
        for timing in study.TIMINGS:
            for width in study.WIDTHS:
                values[(timing, width)] = study.evaluate_routes(
                    routes, layers, dims, 16, width, plan['mappings']['g16'],
                    plan['train_only_plane_assignments']['g16'][f'w{width}'], timing)
        ideal_g16 = values[('ideal_overlap', 4)]['model_ns']
        old_g = plan['old_winner']['group_chips']
        old = study.evaluate_routes(
            routes, layers, dims, old_g, 4, plan['mappings'][f'g{old_g}'],
            plan['train_only_plane_assignments'][f'g{old_g}']['w4'], 'ideal_overlap')['model_ns']
        for timing in study.TIMINGS:
            baseline = values[(timing, 4)]
            for width in study.WIDTHS:
                value = values[(timing, width)]
                np.testing.assert_array_equal(value['useful_bytes'], baseline['useful_bytes'])
                np.testing.assert_array_equal(value['physical_pages'], baseline['physical_pages'])
                arrays[f'{name}__{timing}__width{width}__model_ns'] = value['model_ns']
                for window in (32, 128):
                    bp = baseline['model_ns'][:, :window].mean(axis=1)
                    cp = value['model_ns'][:, :window].mean(axis=1)
                    summary.append(dict(
                        model=name, timing=timing, plane_width=width, group_chips=16,
                        window=window, prompts=len(heldout), **study.summarize(value, window),
                        improvement_vs_g16_width4=float(1 - cp.mean() / bp.mean()),
                        improvement_95ci=study.paired_interval(
                            bp, cp, seed=manifest['seed'], samples=manifest['bootstrap']['resamples']),
                        legacy_g16_overlap_baseline_ms=float(ideal_g16[:, :window].mean() / 1e6),
                        improvement_vs_g16_legacy_overlap=float(1 - cp.mean() / ideal_g16[:, :window].mean()),
                        old_selected_overlap_baseline_ms=float(old[:, :window].mean() / 1e6),
                        improvement_vs_old_selected_overlap=float(1 - cp.mean() / old[:, :window].mean())))
                    for i, row in enumerate(heldout):
                        prompts.append(dict(
                            model=name, timing=timing, plane_width=width, group_chips=16,
                            window=window, id=row['id'], task=row['task'], length_bucket=row['length_bucket'],
                            mean_ms_per_token=float(cp[i] / 1e6), baseline_ms_per_token=float(bp[i] / 1e6)))
                for context_ns in (0, 250, 1000, 5000, 20000):
                    for command_ns in (0, 50, 250, 1000):
                        bcost = (baseline['model_ns'] + context_ns * baseline['extra_contexts'] +
                                 command_ns * baseline['nand_read_commands'])
                        ccost = (value['model_ns'] + context_ns * value['extra_contexts'] +
                                 command_ns * value['nand_read_commands'])
                        sensitivity.append(dict(
                            model=name, timing=timing, plane_width=width, group_chips=16,
                            extra_context_setup_ns=context_ns, extra_nand_command_ns=command_ns,
                            window=128, mean_ms_per_token=float(ccost.mean() / 1e6),
                            baseline_ms_per_token=float(bcost.mean() / 1e6),
                            improvement=float(1 - ccost.mean() / bcost.mean())))
        assert sha256(plan_path) == frozen_hash
        provenance.append(dict(model=name, frozen_plan_sha256=frozen_hash, test_cache=cache))
        print(name + ' g16 diagnostics complete', flush=True)
    write_json(out / 'fixed_g16_summary.json', summary)
    study.write_csv(out / 'fixed_g16_per_prompt.csv', prompts)
    study.write_csv(out / 'fixed_g16_sensitivity.csv', sensitivity)
    np.savez_compressed(out / 'fixed_g16_token_costs.npz',
                        prompt_ids=np.array([r['id'] for r in heldout]), **arrays)
    assert sha256(manifest_path) == manifest_hash
    diagnostics = dict(
        schema_version=1, source_sha256=study.source_hashes(),
        diagnostics_source=str(Path(__file__).resolve()),
        diagnostics_source_sha256=sha256(Path(__file__).resolve()),
        command=[sys.executable, str(Path(__file__).resolve()), '--results', str(out)],
        main_manifest_sha256=manifest_hash, provenance=provenance, selection_changed=False,
        fixed_group_chips=16, widths=list(study.WIDTHS), timings=list(study.TIMINGS),
        bootstrap=manifest['bootstrap'], rows=len(summary), per_prompt_rows=len(prompts),
        elapsed_seconds=time.time() - started)
    original_command = out / 'fixed_g16.command.json'
    if original_command.exists():
        diagnostics['command_file_sha256'] = sha256(original_command)
    write_json(out / 'fixed_g16_diagnostics.json', diagnostics)


if __name__ == '__main__':
    main()
