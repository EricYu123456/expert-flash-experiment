#!/usr/bin/env python3
"""Audit functional equality and estimate full-routed-FFN reference inventory."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from independent_quantization_check import fnv

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
MODELS = Path('/home/nfs_client/ericyu/expert_flash_experiment/models')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    fixed = json.loads((HERE / 'frozen_inputs.json').read_text())
    reports = []
    for model in ('granite4tiny', 'qwen3'):
        d = json.loads((HERE / model / 'numeric.json').read_text())
        phases = {x['phase']: x for x in d['phases']}
        full = [v for k, v in phases.items() if k.startswith('raw-full/')]
        assert len(full) == (6 if model == 'granite4tiny' else 2)
        for p in full:
            split = phases[p['phase'].replace('raw-full/', 'raw-split/', 1)]
            assert p['callbacks'] == 16
            assert p['integer_mismatches'] == split['integer_mismatches'] == 0
            assert p['scaled_bit_mismatches'] == split['scaled_bit_mismatches'] == 0
            for metric in ('split2_vs_full', 'split3_vs_full'):
                assert p[metric]['bitwise_different'] == 0
                assert p[metric]['nonfinite'] == 0
            assert p['input_fnv1a64'] == split['input_fnv1a64']
            assert p['output_fnv1a64'] == split['output_fnv1a64']
        prompt_checks = []
        for prompt in fixed['prompts']:
            stem = prompt['id']
            directory = HERE / model / 'traces'
            full_summary = json.loads((directory / f'{stem}.raw-full.summary.json').read_text())
            split_summary = json.loads((directory / f'{stem}.raw-split.summary.json').read_text())
            route_a = (directory / f'{stem}.raw-full.routing.jsonl').read_bytes()
            route_b = (directory / f'{stem}.raw-split.routing.jsonl').read_bytes()
            output_a = (directory / f'{stem}.raw-full.output.txt').read_bytes()
            output_b = (directory / f'{stem}.raw-split.output.txt').read_bytes()
            assert full_summary['generated_token_ids'] == split_summary['generated_token_ids']
            assert route_a == route_b and output_a == output_b
            previous = ROOT / 'results/research-2026-11-02/round2/split-numeric' / model
            native_summary = json.loads((previous / f'{stem}.control.summary.json').read_text())
            native_routes = [json.loads(v) for v in (previous / f'{stem}.control.routing.jsonl').read_text().splitlines()]
            raw_routes = [json.loads(v) for v in route_a.splitlines()]
            assert len(native_routes) == len(raw_routes)
            x, y = native_summary['generated_token_ids'], full_summary['generated_token_ids']
            prompt_checks.append(dict(prompt_id=stem,
                generated_token_ids_full=full_summary['generated_token_ids'],
                generated_token_ids_split=split_summary['generated_token_ids'],
                tokens_exact=True, output_bytes_exact=True, routing_records_exact=True,
                routing_records=len(route_a.splitlines()),
                original_GGUF_native_control_token_ids=x,
                raw_vs_original_GGUF_tokens_exact=x == y,
                raw_vs_original_GGUF_ordered_route_differences=sum(a['expert_ids'] != b['expert_ids'] for a, b in zip(native_routes, raw_routes)),
                raw_vs_original_GGUF_route_set_differences=sum(set(a['expert_ids']) != set(b['expert_ids']) for a, b in zip(native_routes, raw_routes)),
                first_raw_vs_GGUF_token_difference=next((i for i, (a, b) in enumerate(zip(x, y)) if a != b), None)))
        cache_checks = []
        for e in d['weight_cache']:
            assert e['created_variant'] == 'raw-full'
            path = Path(e['scales_f32le_path'])
            if not path.is_absolute():
                path = ROOT / path
            scales = np.fromfile(path, dtype='<f4')
            assert len(scales) == e['rows'] and np.all(np.isfinite(scales)) and np.all(scales > 0)
            assert fnv(scales) == int(e['scales_fnv1a64'])
            cache_checks.append(dict(key=e['key'], rows=len(scales), min_scale=float(scales.min()),
                max_scale=float(scales.max()), scales_sha256=sha(path)))
        meta_path = MODELS / (model + '-metadata.json')
        meta = json.loads(meta_path.read_text())
        tensors = [t for t in meta['tensors'] if '_exps.' in t['name'] and t.get('projection') in ('gate', 'up', 'down')]
        elements = sum(math.prod(t['shape_ggml']) for t in tensors)
        rows = sum(t['shape_ggml'][1] * t['shape_ggml'][2] for t in tensors)
        active = 6 if model == 'granite4tiny' else 8
        per_token_macs = active * sum(t['shape_ggml'][0] * t['shape_ggml'][1] for t in tensors)
        per_token_outputs = active * sum(t['shape_ggml'][1] for t in tensors)
        reports.append(dict(model=model, prompt_checks=prompt_checks,
            numeric_dots_per_mode=sum(p['dot_rows'] for p in full),
            integer_mismatches=0, scaled_F32_bit_mismatches=0,
            input_and_output_phase_digests_exact=True,
            observed_integer_maxima={key: max(p[key] for p in full) for key in
                ('max_abs_full_integer', 'max_abs_partial2', 'max_abs_partial3')},
            raw_vs_original_local_phase_relative_l2=[p['raw_vs_original']['relative_l2'] for p in full],
            cache=dict(expert_matrix_entries=len(d['weight_cache']),
                raw_i8_bytes=sum(e['raw_i8_bytes'] for e in d['weight_cache']),
                scale_bytes=sum(e['scale_bytes'] for e in d['weight_cache']),
                all_entries_created_during_full_reference=True,
                all_scale_files_FNV_and_positive_finite_validated=True,
                row_scale_checks=cache_checks),
            all_routed_FFN_inventory=dict(metadata_path=str(meta_path), metadata_sha256=sha(meta_path),
                original_checkpoint_sha256=meta['sha256'], routed_raw_i8_bytes=elements,
                routed_raw_i8_GiB=elements / 2**30, host_row_scale_bytes=rows * 4,
                host_row_scale_MiB=rows * 4 / 2**20,
                logical_GEMV_row_outputs_per_decode_token=per_token_outputs,
                host_scale_read_bytes_per_decode_token_if_each_once=per_token_outputs * 4,
                integer_MACs_per_decode_token=per_token_macs,
                integer_MACs_for_same_32_decode_tokens=per_token_macs * 32,
                caveat='Inventory only, not a measured all-layer runtime. Native hardware input/output/scales and quality remain unvalidated.')))
    inputs_unchanged = {p: sha(Path(p)) == h for p, h in fixed['input_sha256'].items()}
    assert all(inputs_unchanged.values())
    independent = json.loads((HERE / 'independent_quantization_check.json').read_text())
    assert independent['status'] == 'PASS'
    report = dict(status='PASS_small_functional_operator_smoke', models=reports,
        frozen_inputs=fixed, original_input_hashes_unchanged=inputs_unchanged,
        independent_original_GGUF_quantization_check=independent,
        theoretical_integer_bounds=dict(signed_range=[-127, 127], maximum_abs_product=16129,
            G4_full_C1536=24774144, G4_partial_C1024=16516096,
            partial_C512=8258048, Qwen_full_C768=12387072, partial_C256=4129024,
            int32_limit=2147483647, all_partial_integers_exact_in_FP32_if_unscaled_native_output=True,
            FP32_exact_integer_limit=16777216,
            native_output_contract_proven=False),
        next_step_feasibility=dict(all_routed_reference_run=False,
            software_feasible=True,
            requirements=['Generalize interception to all routed projections and columns; verify row/ID/input layout for every shape.',
                'For a consistent prefill/decode reference, also substitute routed FFN prefill or explicitly specify a mixed prefill/decode compute contract. Current smoke leaves prefill original.',
                'Do not retain original routing as a substitute for routing produced by the raw reference.',
                'Quantization construction is offline setup, not placement speedup. Host row scales, vector quantization, integer recovery and output rescaling require accounted memory/latency.',
                'Obtain native AiF accumulation/output/scale contract before claiming direct execution. Conditional unscaled FP32 recovery does not prove that interface exists.',
                'Freeze compute settings and placements before using public prompts; evaluate model quality separately from full/split equality.'],
            optimization_note='Current callback computes original GGUF GEMV plus full and split integer controls, and lazily constructs weights. Its wall time is not a device or placement performance estimate.'),
        no_new_quantization_method_claim=True, no_new_hardware_claim=True,
        same_original_GGUF_values_preserved_claim=False, model_quality_validated=False)
    (HERE / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps([dict(model=r['model'], scalar_dots_per_mode=r['numeric_dots_per_mode'],
        maxima=r['observed_integer_maxima'], inventory=r['all_routed_FFN_inventory'],
        token_checks=[{k: v for k, v in c.items() if k not in ('generated_token_ids_full',
            'generated_token_ids_split', 'original_GGUF_native_control_token_ids')} for c in r['prompt_checks']]) for r in reports], indent=2))


if __name__ == '__main__':
    main()
