#!/usr/bin/env python3
"""Account for the fixed six-expert pilot; no latency or speedup inference."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path('/home/ericyu/expert_flash_experiment')
DATA = Path('/home/nfs_client/ericyu/expert_flash_experiment/research-2026-11-08-shared-features')
METADATA = Path('/home/nfs_client/ericyu/expert_flash_experiment/models/deepseek-metadata.json')
ELIGIBLE = {1: [14, 40], 13: [10, 56], 26: [0, 1]}
H, I, R, TOPK, LAYERS, EXPERTS = 2048, 1408, 2816, 6, 26, 64


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def size(n):
    return {'bytes': int(n), 'MiB': n / (1 << 20)}


def static_cost(metadata):
    tensors = metadata['tensors']
    role_names = ('gate', 'up', 'down', 'gate_exps', 'up_exps', 'down_exps',
                  'gate_shexp', 'up_shexp', 'down_shexp')
    ffn = [t for t in tensors if any('.ffn_' + p + '.weight' in t['name'] for p in role_names)]
    coefficients = sum(math.prod(t['shape_ggml']) for t in ffn)
    scales = sum(math.prod(t['shape_ggml'][1:]) * 4 for t in ffn)
    nonffn = sum(t['n_bytes'] for t in tensors if t not in ffn)
    gguf_overhead = metadata['model_bytes'] - sum(t['n_bytes'] for t in tensors)
    native = 3 * H * I
    native_scales = (2 * I + H) * 4
    readout = H * R
    readout_scales = H * 4
    bias = H * 4
    original_six = 6 * (native + native_scales)
    readout_six = 6 * (readout + readout_scales + bias)
    native_shared_gu = len(ELIGIBLE) * (2 * H * R + 2 * R * 4)
    native_shared_down = len(ELIGIBLE) * (H * R + H * 4)
    all_routed = LAYERS * EXPERTS * native
    all_routed_metadata = LAYERS * EXPERTS * native_scales
    baseline = coefficients + scales + nonffn + gguf_overhead
    replacement = baseline - original_six + readout_six
    return {
        'precision': 'One byte per canonical W8 coefficient; F32 complete-row scales and F32 readout biases. Non-FFN tensors retain source GGUF bytes. No AiF page padding or ECC is assumed.',
        'per_expert': {
            'native_GU_coefficients': 2 * H * I, 'native_D_coefficients': H * I,
            'native_total_coefficients': native, 'C_coefficients': readout,
            'net_coefficient_saving': native - readout,
            'coefficient_saving_fraction': (native - readout) / native,
            'native_W8_scale_bytes': native_scales,
            'C_W8_scale_bytes': readout_scales, 'C_F32_bias_bytes': bias,
        },
        'fixed_six_pilot': {
            'eligible': ELIGIBLE, 'original_coefficients': 6 * native,
            'replacement_coefficients': 6 * readout,
            'net_coefficient_saving': 6 * (native - readout),
            'coefficient_saving_fraction_within_each_measured_layer_routed_bank':
                2 * (native - readout) / (EXPERTS * native),
            'coefficient_saving_fraction_across_all_26_routed_banks':
                6 * (native - readout) / all_routed,
            'original_six_with_scales': size(original_six),
            'all_six_C_with_scales_and_bias': size(readout_six),
            'net_capacity_saving_if_original_six_removed': size(original_six - readout_six),
            'additional_capacity_if_all_original_fallbacks_retained': size(readout_six),
            'fallback_retained_total_expert_capacity': size(original_six + readout_six),
        },
        'host_residency': {
            'native_shared_GU_for_three_target_layers': size(native_shared_gu),
            'native_shared_D_for_three_target_layers_if_host_resident': size(native_shared_down),
            'incremental_host_bytes_to_keep_all_six_C_instead_of_flash': size(readout_six),
            'host_shared_GU_plus_all_six_C': size(native_shared_gu + readout_six),
            'host_shared_GU_D_plus_all_six_C': size(native_shared_gu + native_shared_down + readout_six),
            'interpretation': 'Shared GU/D already belong to the checkpoint; residency moves these weights rather than creating them. If the host budget can hold shared GU plus all six C matrices, the strongest same-representation control keeps C on host and pays no C flash reads. All C are not free under a budget that only fits shared GU. Biases/scales are included; KV, activations, workspace and allocator overhead need additional host capacity.',
        },
        'full_model_weight_accounting': {
            'conversion_materialized_in_loaded_GGUF': False,
            'actual_source_GGUF_file_bytes_retained_by_functional_harness': metadata['model_bytes'],
            'functional_harness_note': 'The current quality harness loads the complete original GGUF, builds native W8 caches, and adds readouts. Reported removal savings describe a possible tensor conversion after quality eligibility is accepted, not measured process RSS or a smaller GGUF already created.',
            'canonical_FFN_coefficients': coefficients,
            'canonical_FFN_row_scale_bytes': scales,
            'all_26_routed_banks_coefficients': all_routed,
            'all_26_routed_banks_row_scale_bytes': all_routed_metadata,
            'non_FFN_source_GGUF_tensor_bytes': nonffn,
            'retained_GGUF_header_and_padding_bytes': gguf_overhead,
            'baseline_modeled_weight_bytes': size(baseline),
            'pilot_replacement_originals_removed_modeled_weight_bytes': size(replacement),
            'pilot_original_fallbacks_retained_modeled_weight_bytes': size(baseline + readout_six),
            'pilot_fraction_of_modeled_weight_bytes_saved_if_originals_removed': (baseline - replacement) / baseline,
            'exclusions': ['KV cache', 'activations and workspace', 'physical flash page/alignment/ECC overhead', 'runtime metadata beyond row scales and biases'],
        },
        'hypothetical_all64_all26_NOT_VALIDATED': {
            'replaced_experts': LAYERS * EXPERTS,
            'coefficient_saving_fraction_in_routed_banks': (native - readout) / native,
            'net_coefficient_saving': LAYERS * EXPERTS * (native - readout),
            'all_26_native_shared_GU_residency': size(LAYERS * (2 * H * R + 2 * R * 4)),
            'all_1664_C_with_scales_and_bias_residency': size(LAYERS * EXPERTS * (readout + readout_scales + bias)),
            'shared_GU_plus_all_1664_C_residency': size(LAYERS * (2 * H * R + 2 * R * 4) + LAYERS * EXPERTS * (readout + readout_scales + bias)),
            'quality_evidence': False,
            'interpretation': 'Arithmetic extrapolation only. The pilot measures six fixed experts across three layers, so it cannot support this all-expert/all-layer conversion.',
        },
    }


def load_array(root, entry):
    path = root / entry['path']
    dtype = np.dtype(entry['dtype'])
    shape = tuple(entry['shape'])
    assert path.stat().st_size == math.prod(shape) * dtype.itemsize, path
    return np.memmap(path, dtype=dtype, shape=shape, mode='r')


def exposure(data, expected_prompts):
    jobs = [line.split('\t')[1] for line in (data / 'corpus/test.jobs.tsv').read_text().splitlines()]
    assert len(jobs) == len(set(jobs)) == expected_prompts
    expected = set(jobs)
    observed = set()
    collected = {l: [] for l in ELIGIBLE}
    identities = {}
    for shard in range(4):
        folder = data / f'captures/test/shard{shard}'
        manifest_path = folder / 'capture_manifest.json'
        manifest = json.loads(manifest_path.read_text())
        assert manifest['status'] == 'complete' and manifest['mode'] == 'w8'
        assert manifest['gpu_layers'] == 0 and manifest['max_tokens'] == 16
        identities[str(manifest_path)] = sha(manifest_path)
        prompt_set = set()
        for p in sorted((folder / 'scores').glob('*.summary.json')):
            score = json.loads(p.read_text())
            pid = score['prompt_id']
            assert pid not in observed and pid in expected
            assert score['tokens'] == 16 and score['scored_tokens'] == 15 and len(score['token_ids']) == 16
            observed.add(pid)
            prompt_set.add(pid)
        assert len(prompt_set) == manifest['jobs'] == expected_prompts // 4
        for layer, expert_ids in ELIGIBLE.items():
            item = manifest['layers'][str(layer)]
            assert item['expert_ids'] == expert_ids
            ids = np.array(load_array(folder, item['arrays']['selected_experts']))
            alpha = np.array(load_array(folder, item['arrays']['routing_weights']))
            assert ids.shape == alpha.shape == (item['rows'], TOPK)
            assert np.isfinite(alpha).all() and (alpha >= 0).all() and (alpha <= 1).all()
            assert ((ids >= 0) & (ids < EXPERTS)).all()
            assert all(len(set(row)) == TOPK for row in ids)
            rows_path = folder / f'layer{layer}/rows.jsonl'
            rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
            assert len(rows) == item['rows'] == len(prompt_set) * 16
            by_prompt = {}
            for row in rows:
                by_prompt.setdefault(row['prompt_id'], []).append(row['position'])
            assert set(by_prompt) == prompt_set and all(v == list(range(16)) for v in by_prompt.values())
            for key in ('selected_experts', 'routing_weights'):
                p = folder / item['arrays'][key]['path']
                identities[str(p)] = sha(p)
            identities[str(rows_path)] = sha(rows_path)
            collected[layer].append((ids, alpha))
    assert observed == expected
    result = {}
    total_events, weighted_mass, total_alpha = 0, 0.0, 0.0
    N = expected_prompts * 16
    for layer, parts in collected.items():
        ids = np.concatenate([p[0] for p in parts])
        alpha = np.concatenate([p[1] for p in parts]).astype(np.float64)
        assert len(ids) == N
        eligible = np.isin(ids, ELIGIBLE[layer])
        events = int(eligible.sum())
        mass = float(alpha[eligible].sum())
        alpha_total = float(alpha.sum())
        experts = {}
        for expert in ELIGIBLE[layer]:
            mask = ids == expert
            experts[str(expert)] = dict(selected_expert_token_events=int(mask.sum()),
                                        fraction_of_token_positions_selected=float(mask.any(axis=1).mean()),
                                        sum_native_router_alpha=float(alpha[mask].sum()),
                                        native_alpha_mass_fraction_within_layer=float(alpha[mask].sum() / alpha_total))
        result[str(layer)] = dict(rows=N, experts=experts,
                                 eligible_expert_token_events=events,
                                 fraction_of_top6_execution_slots=events / (N * TOPK),
                                 fraction_of_tokens_with_any_eligible_expert=float(eligible.any(axis=1).mean()),
                                 mean_eligible_slots_per_token=events / N,
                                 eligible_router_alpha_mass=mass,
                                 all_top6_router_alpha_mass=alpha_total,
                                 weighted_eligible_alpha_fraction=mass / alpha_total,
                                 coefficient_read_saving_fraction_for_this_layer_routed_FFN=
                                 events * (3 * H * I - H * R) / (N * TOPK * 3 * H * I))
        total_events += events
        weighted_mass += mass
        total_alpha += alpha_total
    return dict(status='complete', prompts=expected_prompts, positions_per_prompt=16,
                retained_positions_include_BOS=True, total_model_token_positions=N,
                source='Native W8 capture routing, teacher-forced first 16 tokens. This is a workload diagnostic, not post-replacement routing or timing.',
                layers=result, eligible_expert_token_events_three_layers=total_events,
                execution_slot_exposure_across_three_captured_layers=total_events / (N * len(ELIGIBLE) * TOPK),
                weighted_native_alpha_exposure_across_three_captured_layers=weighted_mass / total_alpha,
                execution_slot_exposure_across_all_26_MoE_layers=total_events / (N * LAYERS * TOPK),
                all_26_MoE_routed_FFN_coefficient_read_saving_fraction=
                total_events * (3 * H * I - H * R) / (N * LAYERS * TOPK * 3 * H * I),
                all_26_MoE_weighted_alpha_exposure=None,
                uncaptured_alpha_limitation='Only three layers store router alpha. The full-model execution-slot denominator is known (26 x 6 per token), but full-model alpha-weighted exposure is unavailable and is not extrapolated.',
                timing_limitation='Coefficient work counts assume selected native FFNs read once without caching and omit shared work, host GU, dependencies, row scales/biases, transfer, control and overlap. They are not speedups.',
                input_sha256=identities)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--data-root', type=Path, default=DATA)
    p.add_argument('--metadata', type=Path, default=METADATA)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--static-only', action='store_true')
    p.add_argument('--expected-test-prompts', type=int, default=1000)
    a = p.parse_args()
    metadata = json.loads(a.metadata.read_text())
    report = dict(status='static_only' if a.static_only else 'complete',
                  scope='Fixed six-expert shared_z pilot; coefficient/capacity and native-routing exposure only.',
                  metadata_sha256=sha(a.metadata), source_sha256=sha(__file__),
                  static_cost=static_cost(metadata), system_speedup_claim=False)
    if not a.static_only:
        report['native_TEST_exposure'] = exposure(a.data_root, a.expected_test_prompts)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(status=report['status'], output=str(a.output),
                          fixed_six_coefficients_saved=report['static_cost']['fixed_six_pilot']['net_coefficient_saving'],
                          all_26_routed_bank_saving_fraction=report['static_cost']['fixed_six_pilot']['coefficient_saving_fraction_across_all_26_routed_banks'])))


if __name__ == '__main__':
    main()
