#!/usr/bin/env python3
"""Audit completed CPU experiments and summarize paired, article-cluster quality."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path('/home/ericyu/expert_flash_experiment')
DATA = Path('/home/nfs_client/ericyu/expert_flash_experiment/research-2026-11-08-shared-features')
OUT = ROOT / 'results/research-2026-11-08-shared-features'
ELIGIBLE = {1: {14, 40}, 13: {10, 56}, 26: {0, 1}}


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def captured(split):
    result = {}
    for shard in range(4):
        folder = DATA / f'captures/{split}/shard{shard}'
        manifest = read(folder / 'capture_manifest.json')
        assert manifest['status'] == 'complete' and manifest['gpu_layers'] == 0 and manifest['mode'] == 'w8'
        for path in sorted((folder / 'scores').glob('*.summary.json')):
            row = read(path)
            key = row['prompt_id']
            assert key not in result and row['tokens'] == 16 and row['scored_tokens'] == 15
            positions = lines(path.with_name(key + '.jsonl'))
            assert [p['position'] for p in positions] == list(range(15))
            assert abs(sum(p['nll_F64'] for p in positions) - row['sum_nll_F64']) < 1e-12
            result[key] = dict(summary=row, positions=positions)
    return result


def quality(arm, expected, reference, eligible):
    result, exposures, route_keys = {}, {}, {}
    expected_steps = 0 if arm.endswith('_valid') else 8
    for shard in range(4):
        folder = DATA / f'quality/{arm}/shard{shard}/data'
        manifest = read(folder / 'completion.json')
        assert manifest['status'] == 'complete' and manifest['gpu_layers'] == 0
        assert manifest['decode_steps'] == expected_steps
        for path in sorted(folder.glob('*.summary.json')):
            row = read(path)
            key = row['prompt_id']
            assert key not in result and key in expected
            assert row['input_tokens'] == 16 and row['scored_tokens'] == 15
            assert row['decode_steps'] == expected_steps
            assert row['token_ids'] == reference[key]['summary']['token_ids']
            positions = lines(folder / (key + '.positions.jsonl'))
            assert [p['position'] for p in positions] == list(range(15))
            assert [p['target_token_id'] for p in positions] == row['token_ids'][1:]
            assert [p['input_token_id'] for p in positions] == row['token_ids'][:-1]
            assert abs(sum(p['nll_F64'] for p in positions) - row['sum_nll_F64']) < 1e-12
            assert abs(row['mean_nll_F64'] - row['sum_nll_F64'] / 15) < 1e-14
            assert len(row['generated_token_ids']) == row['actual_generated_length']
            assert row['actual_generated_length'] <= row['decode_steps']
            assert row['stopped_eog'] or row['actual_generated_length'] == row['decode_steps']
            assert row['decoded_generated_inputs'] == row['actual_generated_length'] - int(row['stopped_eog'])
            seen, exp = set(), dict(prefill_all16=0, prefill_NLL_causal15=0,
                                    prefill_last_for_first_generated_token=0,
                                    decode_all=0, decode_causal_to_reported_generated_tokens=0,
                                    decode_last_unused_prediction=0)
            ordered_prefill = {}
            for route in lines(folder / (key + '.routing.jsonl')):
                phase, step, pos, layer = (route[k] for k in ('phase', 'step', 'token_in_batch', 'layer'))
                identity = (phase, step, pos, layer)
                assert identity not in seen and 1 <= layer <= 26
                seen.add(identity)
                ids = route['expert_ids']
                assert len(ids) == len(set(ids)) == 6 and all(0 <= e < 64 for e in ids)
                hits = sum(e in eligible.get(layer, set()) for e in ids)
                if phase == 'prefill':
                    assert step == -1 and 0 <= pos < 16
                    exp['prefill_all16'] += hits
                    exp['prefill_NLL_causal15' if pos < 15 else 'prefill_last_for_first_generated_token'] += hits
                    ordered_prefill[(pos, layer)] = ids
                else:
                    assert phase == 'decode' and pos == 0 and 0 <= step < row['decoded_generated_inputs']
                    exp['decode_all'] += hits
                    causal = step + 1 < row['actual_generated_length']
                    exp['decode_causal_to_reported_generated_tokens' if causal else 'decode_last_unused_prediction'] += hits
            assert len(ordered_prefill) == 16 * 26
            assert len(seen) == 26 * (16 + row['decoded_generated_inputs'])
            assert exp['prefill_all16'] + exp['decode_all'] == row['replaced_expert_token_events']
            if not eligible:
                assert row['native_shared_z_floats_bitchecked'] == 0
                assert positions == reference[key]['positions'], f'Native capture/quality mismatch: {key}'
            else:
                assert row['native_shared_z_floats_bitchecked'] == len(eligible) * 2816 * (16 + row['decoded_generated_inputs'])
            result[key], exposures[key], route_keys[key] = row, exp, ordered_prefill
    assert set(result) == set(expected), f'{arm} does not contain the exact expected prompt set'
    return result, exposures, route_keys


def paired(a, b, ids, groups, boot, seed=20261108, compare_to_native=True):
    delta = np.asarray([a[i]['mean_nll_F64'] - b[i]['mean_nll_F64'] for i in ids])
    names = sorted(set(groups))
    sums = np.asarray([delta[np.asarray(groups) == g].sum() for g in names])
    counts = np.asarray([groups.count(g) for g in names])
    draws = sums[boot].sum(axis=1) / counts[boot].sum(axis=1)
    lo, hi = np.quantile(draws, [.025, .975])
    mean = float(delta.mean())
    margin = math.log(1.02)
    verdict = ('passes_this_pilot_margin' if hi <= margin else ('fails_this_configuration' if lo > margin else 'inconclusive')) if compare_to_native else 'descriptive_secondary_comparison'
    return dict(mean_delta_nll=mean, ci95_delta_nll=[float(lo), float(hi)],
                relative_subset_ppl_change=math.expm1(mean),
                ci95_relative_subset_ppl_change=[math.expm1(float(lo)), math.expm1(float(hi))],
                margin_delta_nll=margin if compare_to_native else None, verdict=verdict,
                bootstrap=f'10000 percentile article-cluster resamples; variable cluster sizes retained; seed{seed}')


def generation(a, b, ids):
    exact, positional, total, first = 0, 0, 0, []
    for key in ids:
        aa, bb = a[key]['generated_token_ids'], b[key]['generated_token_ids']
        exact += aa == bb
        n = max(len(aa), len(bb))
        matches = [j < len(aa) and j < len(bb) and aa[j] == bb[j] for j in range(n)]
        positional += sum(matches)
        total += n
        first.append(next((j for j, same in enumerate(matches) if not same), n))
    return dict(prompts=len(ids), exact_continuation_prompts=exact,
                exact_continuation_fraction=exact / len(ids), positional_token_match_fraction=positional / total,
                mean_common_prefix_tokens=float(np.mean(first)),
                interpretation='Agreement with native greedy continuation; not task accuracy or a correctness label')


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    corpus = read(DATA / 'corpus/manifest.json')
    registry = {r['id']: r for r in corpus['records']}
    captures, native_prefixes = {}, {}
    for split, count in [('train', 600), ('validation', 200), ('test', 1000)]:
        cap = captured(split)
        assert set(cap) == {k for k, v in registry.items() if v['split'] == split}
        assert len(cap) == count
        prefixes = {tuple(v['summary']['token_ids']) for v in cap.values()}
        assert len(prefixes) == count, 'Native 16-token duplicate prompt'
        for previous in native_prefixes.values():
            assert not (prefixes & previous), 'Native train/validation/test context overlap'
        native_prefixes[split], captures[split] = prefixes, cap
    assert read(DATA / 'quality/execution.json')['status'] == 'complete'
    frozen_quality = read(DATA / 'quality/protocol.frozen.json')
    for path, expected_hash in frozen_quality['inputs_sha256'].items():
        assert sha(path) == expected_hash, f'Changed frozen quality input: {path}'
    ids = sorted(captures['test'])
    groups = [registry[i]['article_id'] for i in ids]
    rng = np.random.default_rng(20261108)
    boot = rng.integers(0, len(set(groups)), size=(10000, len(set(groups))))
    arms, exposures, routes = {}, {}, {}
    for arm in ('native', 'shared_z_all6', 'raw_x_all6'):
        arms[arm], exposures[arm], routes[arm] = quality(arm, ids, captures['test'], {} if arm == 'native' else ELIGIBLE)
    valid_ids = sorted(captures['validation'])
    valid, valid_exp, _ = quality('shared_z_layer1_valid', valid_ids, captures['validation'], {1: ELIGIBLE[1]})
    report = dict(status='complete', at_utc=datetime.now(timezone.utc).isoformat(), cpu_only=True,
                  train_prompts=600, validation_prompts=200, heldout_test_prompts=1000,
                  test_articles=len(set(groups)), native_contexts16_disjoint=True,
                  native_capture_and_quality_all_15000_NLL_top1_records_identical=True,
                  test_scored_tokens_per_arm=15000, test_max_greedy_tokens_per_prompt=8,
                  readout_fit='TRAIN-only coefficients, dense VALID-only 10-value ridge grid; all choices frozen before TEST candidate outputs',
                  inputs=dict(corpus_manifest_sha256=sha(DATA / 'corpus/manifest.json'),
                              quality_protocol_sha256=sha(DATA / 'quality/protocol.frozen.json'),
                              all_frozen_inputs_rehashed_after_completion=True,
                              frozen_input_files_verified=len(frozen_quality['inputs_sha256'])),
                  arms={}, paired={}, generation={})
    for arm, values in arms.items():
        nll = sum(values[i]['sum_nll_F64'] for i in ids) / 15000
        report['arms'][arm] = dict(mean_nll=nll, subset_ppl=math.exp(nll), prompts=len(values),
                                  generated_tokens=sum(len(v['generated_token_ids']) for v in values.values()),
                                  stopped_eog_prompts=sum(v['stopped_eog'] for v in values.values()),
                                  exposure={k: sum(v[k] for v in exposures[arm].values()) for k in next(iter(exposures[arm].values()))},
                                  native_shared_z_floats_bitchecked=sum(v['native_shared_z_floats_bitchecked'] for v in values.values()))
    for a, b, label in [('shared_z_all6', 'native', 'shared_vs_native_primary'),
                        ('raw_x_all6', 'native', 'raw_x_vs_native_secondary'),
                        ('shared_z_all6', 'raw_x_all6', 'shared_vs_raw_x_secondary')]:
        report['paired'][label] = paired(arms[a], arms[b], ids, groups, boot, compare_to_native=b == 'native')
        report['generation'][label] = generation(arms[a], arms[b], ids)
    native_valid = {i: captures['validation'][i]['summary'] for i in valid_ids}
    valid_groups = [registry[i]['article_id'] for i in valid_ids]
    valid_boot = np.random.default_rng(20261109).integers(0, len(set(valid_groups)), size=(10000, len(set(valid_groups))))
    report['single_layer_VALID_diagnostic'] = paired(valid, native_valid, valid_ids, valid_groups, valid_boot, seed=20261109)
    report['single_layer_VALID_diagnostic']['selection_role'] = 'Diagnostic on the same VALID set used for ridge; not an independent quality confirmation and not used to select TEST eligibility'
    for arm in ('shared_z_all6', 'raw_x_all6'):
        total, exact, overlap = 0, 0, 0
        for key in ids:
            for identity, nn in routes['native'][key].items():
                aa = routes[arm][key][identity]
                total += 1
                exact += nn == aa
                overlap += len(set(nn) & set(aa))
        report['arms'][arm]['prefill_same_input_routing'] = dict(layer_token_events=total,
            exact_ordered_top6_fraction=exact / total, selected_expert_set_overlap=overlap / (6 * total),
            interpretation='Same teacher input tokens with each model live hidden states; no native route pinning')
    local = {str(l): read(DATA / f'fits-refined/layer{l}/test_report.json') for l in (1, 13, 26)}
    assert all(v['test_prompts'] == 1000 for v in local.values())
    report['local_readout_test'] = {l: v['per_expert'] for l, v in local.items()}
    report['local_readout_aggregates'] = {l: v['aggregates'] for l, v in local.items()}
    report['coefficient_cost_audit'] = read(OUT / 'cost_audit.json')
    quantization = read(OUT / 'layer26.quantization_VALID_diagnosis.json')
    report['VALID_quantization_attribution'] = dict(
        same_frozen_C=True, new_candidate=False,
        median_A8_zero_feature_fraction=quantization['activation_stats']['A8_zero_fraction_per_row']['p50'],
        median_amax_to_rms=quantization['activation_stats']['amax_to_rms_per_row']['p50'],
        per_expert=[dict(expert=r['expert'], nmse={k: v['nmse'] for k, v in r['versus_native_target'].items()})
                    for r in quantization['experts']])
    report['limitations'] = [
        '16-token English WikiText paragraph starts; 61 correlated article clusters, not long-context/multitask accuracy or standard benchmark PPL.',
        'Only six fixed experts in raw layers1,13,26 are replaced; other experts/layers and the native shared branch remain active.',
        'Native reference is canonical FFN row-W8/vector-A8 over a Q8_0 GGUF checkpoint, not BF16 or bit-identical original GGUF inference.',
        'FP readouts use a separate declared diagnostic accumulation contract; only W8A8 readouts enter live quality arms.',
        'Functional harness still computes original FFN and duplicates shared-GU verification; wall times are not compressed deployment performance.',
        'Eight greedy tokens measure early divergence only; sequence agreement is not answer correctness.',
        'NLL exposure counts prefill positions0..14 only; final prefill and final unused decode are listed separately.',
        'Secondary intervals are descriptive, individually95%; primary quality endpoint is shared_vs_native only.',
        'No actual flash reads, NVMeVirt latency, complete-model speedup or novelty claim is established by these experiments.'
    ]
    save(DATA / 'quality_summary.json', report)
    save(OUT / 'quality_summary.json', report)
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file('/home/nfs_client/ymtuan2/models/DeepSeek-V2-Lite-Chat/tokenizer.json')
    examples = []
    for key in ids[:10]:
        examples.append(dict(prompt_id=key, article_id=registry[key]['article_id'],
                             input=tokenizer.decode(arms['native'][key]['token_ids']),
                             outputs={a: tokenizer.decode(arms[a][key]['generated_token_ids']) for a in arms},
                             selection='First ten lexicographic fixed TEST IDs; no outcome selection'))
    save(OUT / 'generation_examples.json', examples)
    md = ['# Native shared features：CPU 實驗結果', '',
          f"已完成 600 TRAIN、200 VALID、1000 TEST prompts；TEST 來自 {report['test_articles']} 個 WikiText-2 articles。",
          '每題使用含 BOS 的前 16 個 tokens，評 15 個 next-token targets，另生成最多 8 tokens。', '',
          f"主要品質判定：**{report['paired']['shared_vs_native_primary']['verdict']}**。這只適用固定六 experts 的短上下文配置。", '',
          '| 真正執行的模型 | Mean NLL | Subset PPL | 相對 native PPL | Article-cluster 95% CI |',
          '|---|---:|---:|---:|---:|']
    for arm in arms:
        val = report['arms'][arm]
        pair = report['paired'].get('shared_vs_native_primary' if arm == 'shared_z_all6' else 'raw_x_vs_native_secondary') if arm != 'native' else None
        change = f"{pair['relative_subset_ppl_change'] * 100:+.3f}%" if pair else 'reference'
        ci = ('[' + ', '.join(f'{x * 100:+.3f}%' for x in pair['ci95_relative_subset_ppl_change']) + ']') if pair else '—'
        md.append(f"| {arm} | {val['mean_nll']:.6f} | {val['subset_ppl']:.4f} | {change} | {ci} |")
    md += ['', '2% relative subset-PPL 容忍值在測試前固定；僅 primary shared-vs-native interval 用於該品質判定。',
           'Native capture 與 quality runner 的全部 15000 個 NLL／top1／token records 完全一致。', '',
           '## 本輪能回答什麼', '',
           '在這個 checkpoint／校準資料／完整 native shared basis 的配置中，shared pre-Down readout 的 W8A8 NMSE 六點全輸更小的 raw x readout。',
           '因此，即使少量替換的整體品質落在容忍值內，也不能據此證明保存這份 shared nonlinear basis 比便宜的線性 input control 更有價值。',
           '本輪未訓練新 feature pool、未做 router-weighted readout、未做 quantization-aware fitting，也沒有否定這些其他配置。', '',
           '## 局部 W8A8 readout NMSE（1000 個 TEST prompts）', '',
           '| Layer | Expert | raw x | shared output | x+shared output | shared pre-Down | random SwiGLU |',
           '|---:|---:|---:|---:|---:|---:|---:|']
    for layer, values in local.items():
        for expert in sorted({r['expert'] for r in values['per_expert']}):
            row = {r['basis']: r['nmse'] for r in values['per_expert'] if r['expert'] == expert and r['precision'] == 'w8a8'}
            md.append(f"| {layer} | {expert} | " + ' | '.join(f'{row[b]:.4f}' for b in ('x', 'shared_output', 'x_plus_shared_output', 'shared_z', 'random_swiglu')) + ' |')
    md += ['', 'x+shared output 為 4096 維，其他新 nonlinear bases 為 2816 維；不能將較大容量當同成本優勢。',
           '完整 FP／W8A8、selected-only NMSE、cosine、各層 aggregate、ridge 與容量數值見 quality_summary.json 及 NFS fits-refined。', '',
           '以下只評 native router 真正選中該 expert 的 token rows：', '',
           '| Layer | Expert | Selected rows | raw x NMSE | shared pre-Down NMSE |',
           '|---:|---:|---:|---:|---:|']
    for layer, values in local.items():
        for expert in sorted({r['expert'] for r in values['per_expert']}):
            selected = {r['basis']: r for r in values['per_expert'] if r['expert'] == expert and r['precision'] == 'w8a8'}
            x, z = selected['x'], selected['shared_z']
            md.append(f"| {layer} | {expert} | {z['native_selected_rows']} | {x['native_selected_only']['nmse']:.4f} | {z['native_selected_only']['nmse']:.4f} |")
    md += ['', 'TRAIN 擬合對所有捕捉 x 使用相同權重，未依 router eligibility／alpha 加權。更大量的 routed-only 校準是尚未驗證的其他配置。', '',
           '## 真正生成與路由', '', '| 替換版本 | 與 native 完整 continuation 相同率（最多8 tokens，EOG停止） | positional token match | NLL causal replacements |',
           '|---|---:|---:|---:|']
    for arm, key in [('shared_z_all6', 'shared_vs_native_primary'), ('raw_x_all6', 'raw_x_vs_native_secondary')]:
        gen = report['generation'][key]
        exp = report['arms'][arm]['exposure']
        md.append(f"| {arm} | {gen['exact_continuation_fraction'] * 100:.2f}% | {gen['positional_token_match_fraction'] * 100:.2f}% | {exp['prefill_NLL_causal15']} |")
    md += ['', '後續 hidden states、routing、KV 與 greedy token 均重新執行；生成一致率不是任務 accuracy。',
           'NLL exposure 只算 input positions0..14，最後 prefill 及未用的最後 decode prediction 分開列在 JSON。', '',
           '## 為何晚層的 W8A8 特別差', '',
           'VALID-only 歸因顯示：layer26 的 shared z 經 whole-vector A8 後，每向量中位 90.13% features 變零；amax/RMS 中位為 41.95。',
           '下表固定同一份 W8A8-selected C/bias，僅改診斷量化邊界，未訓練新候選：', '',
           '| Expert | 同 C 全精度 | 僅 C row-W8 | 僅 z vector-A8 | 正式 W8A8 |',
           '|---:|---:|---:|---:|---:|']
    for row in quantization['experts']:
        mm = row['versus_native_target']
        md.append(f"| {row['expert']} | " + ' | '.join(f"{mm[k]['nmse']:.5f}" for k in
            ('same_C_full_precision', 'C_weight_W8_only', 'z_activation_A8_only', 'canonical_W8A8')) + ' |')
    cost = report['coefficient_cost_audit']
    static, native_exposure = cost['static_cost'], cost['native_TEST_exposure']
    pilot, resident = static['fixed_six_pilot'], static['host_residency']
    md += ['', '## 容量與 flash 工作量界線', '',
           f"六個原生 experts 含 W8 scales 共 {pilot['original_six_with_scales']['MiB']:.4f} MiB；改為六個 full-width C/bias/scales 共 {pilot['all_six_C_with_scales_and_bias']['MiB']:.4f} MiB。",
           f"刪掉對應原生 FFN 才淨省 {pilot['net_capacity_saving_if_original_six_removed']['MiB']:.4f} MiB；保留全部原生 fallback 則額外增加 {pilot['additional_capacity_if_all_original_fallbacks_retained']['MiB']:.4f} MiB。",
           f"三層 resident shared GU 為 {resident['native_shared_GU_for_three_target_layers']['MiB']:.4f} MiB；再加六個 C 共 {resident['host_shared_GU_plus_all_six_C']['MiB']:.4f} MiB，另需 shared Down／非 FFN／KV／workspace。",
           '這些 C 可全放 host 的預算下，必須比較 host-resident readout，不能刻意送 flash；同樣六個 raw-x C 只需約24.09 MiB。',
           f"本 pilot 的靜態 routed bank coefficient saving 只有 {pilot['coefficient_saving_fraction_across_all_26_routed_banks'] * 100:.6f}%。",
           f"固定 native TEST routing、無 cache、只計 routed 係數的反事實工作量節省為 {native_exposure['all_26_MoE_routed_FFN_coefficient_read_saving_fraction'] * 100:.6f}%；這不是 live candidate 的實測 flash 讀量，也不是 TPOT speedup。",
           '全64 experts／26層的寬度與host容量外推另列cost_audit.json，未驗證其品質。', '',
           '## 解讀與限制', '']
    md += ['- ' + s for s in report['limitations']]
    md += ['', '## 重現與產物', '',
           '- 協定：`docs/research_2026_11_08_shared_features_protocol.md`。',
           '- corpus → capture_run → fit --defer-test → evaluate → quality_run → report；各實際指令與 SHA 保存在 NFS。',
           f'- 大型產物：`{DATA}`。',
           '- `cost_audit.json`／`cost_static.json` 區分 fixed-six 真正成本與全層／全 expert 假想外推。',
           '- 前次較粗 ridge grid 只用於 VALID 診斷，保留於 `fits`；正式 `fits-refined` 使用同樣十個 λ 比較所有 bases。',
           '- 第一個 quality launch 在任何 TEST candidate 啟動／TEST 結果檢視前中止，保留於 `quality-first-grid-cancelled`；正式結果只使用 refined selections。',
           '- 資料：[Salesforce WikiText pinned source](https://huggingface.co/datasets/Salesforce/wikitext/tree/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-2-raw-v1)。', '']
    (OUT / 'report.zh.md').write_text('\n'.join(md))
    print(json.dumps(dict(report=str(OUT / 'report.zh.md'), primary=report['paired']['shared_vs_native_primary']), indent=2))


if __name__ == '__main__':
    run()
