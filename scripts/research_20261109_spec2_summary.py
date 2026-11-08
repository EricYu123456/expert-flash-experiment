#!/usr/bin/env python3
"""Summarize the fixed-grid, legacy TRAIN/VALID-only spec2 diagnosis."""
import json
from pathlib import Path
import shutil

from research_20261109_shared_feature_targets import DATA, ROOT, dump, sha


def main():
    results = ROOT / 'results/research-2026-11-09-shared-features'
    rows, inputs, source_hashes = [], {}, {}
    for layer in (1, 13, 26):
        path = results / f'spec2_conditional_layer{layer}.json'
        report = json.loads(path.read_text())
        assert report['status'] == 'complete'
        inputs[str(path)] = sha(path)
        for expert in report['fixed_experts']:
            support = [r for r in report['support'] if r['split'] == 'train' and r['expert'] == expert]
            models = [r for r in report['conditional']['records'] if r['expert'] == expert]
            row = {'layer': layer, 'expert': expert, 'TRAIN_articles': support[0]['article']['groups'],
                   'TRAIN_selected_articles': support[0]['selected_articles'],
                   'TRAIN_weight_only_row_N_eff_by_beta': {str(r['beta']): r['weight_only_row_N_eff'] for r in support},
                   'TRAIN_article_mass_N_eff_by_beta': {str(r['beta']): r['article']['mass_aggregated_N_eff'] for r in support},
                   'gain_by_width': {}}
            for width in (128, 256):
                group = {}
                for method in ('FP_frozen_raw_x_plus_conditional_shared_residual',
                               'W8A8_folded_conditional_no_joint_refit', 'W8A8_QDQ_joint_hybrid'):
                    group[method] = [{'ridge_B': r.get('ridge_B'),
                                      'relative_VALID_selected_alpha2_NMSE_reduction': r['VALID_relative_selected_alpha2_NMSE_reduction_vs_matched_raw_x']['all'],
                                      'early_pos1_7_reduction': r['VALID_relative_selected_alpha2_NMSE_reduction_vs_matched_raw_x']['early_pos1_7'],
                                      'late_pos8_15_reduction': r['VALID_relative_selected_alpha2_NMSE_reduction_vs_matched_raw_x']['late_pos8_15'],
                                      'VALID_minus_TRAIN_selected_alpha2_NMSE': r['VALID_minus_TRAIN_selected_alpha2_NMSE']}
                                     for r in models if r['method'] == method and r.get('width_shared') == width]
                row['gain_by_width'][str(width)] = group
            rows.append(row)
        source_hashes[report['source_sha256']] = True
    script = ROOT / 'scripts/research_20261109_spec2_diagnostics.py'
    assert set(source_hashes) == {sha(script)}
    freeze = DATA / 'spec2-diagnostics/executed-freeze'
    freeze.mkdir(exist_ok=True)
    identities = {}
    for name in ('research_20261109_spec2_diagnostics.py', 'research_20261109_spec2_numeric_run.py',
                 'research_20261109_spec2_summary.py', 'research_20261109_shared_feature_fit.py',
                 'research_20261108_shared_feature_fit.py'):
        path = ROOT / 'scripts' / name
        identities[str(path)] = sha(path)
        shutil.copy2(path, freeze / name)
    report = {'status': 'complete', 'input_report_sha256': inputs, 'source_sha256': identities,
              'cohorts': {'TRAIN_prompts': 600, 'TRAIN_articles': 321, 'TRAIN_rows': 9600,
                          'VALID_prompts': 200, 'VALID_rows': 3200},
              'fixed_hyperparameters': {'raw_x_ridge': 1, 'z_nuisance_ridge': 1,
                                       'residual_B_ridges': [.03, .3, 3], 'OOF_article_folds': 3,
                                       'conditional_weight_beta': 0, 'shared_feature_widths': [128, 256]},
              'rows': rows,
              'limitations': [
                  'Only calibration TRAIN/VALID, 16-token prefixes and six fixed experts; no new TEST, full-model PPL, or latency result.',
                  'OOF nuisance residuals select indices and fit B; scores on TRAIN are fitting diagnostics, not an unbiased OOF evaluation of B.',
                  'Fixed-ridge raw-x is a diagnostic baseline, not the optimized raw-x chosen by the separate screening experiment.',
                  'Ridge-regularized z-on-x nuisance retains some linearly predictable signal. Conditional gain above this fixed model does not by itself establish purely nonlinear information.',
                  'The grid is reported in full without a blind-test winner. No article-cluster confidence intervals were estimated for this local diagnostic.',
                  'All six experts have zero TRAIN BOS selected-alpha-squared mass; BOS selected-weighted NMSE is undefined, not zero.',
                  'FP residual folding is real-algebra equivalent; finite-precision differences and actual W8A8 outputs were evaluated separately.',
                  'QDQ joint hybrid readouts refit raw-x coefficients and have a different objective from the frozen-raw-x residual model.'
              ],
              'mechanism_scope': 'At the tested fixed grid, only late layer26 expert1 reaches a 10% conditional gain. No tested nonfinal-layer expert reaches that resource-allocation gate.',
              'numeric_audit': str(results / 'spec2_numeric_layer26.json')}
    dump(DATA / 'spec2-diagnostics/summary.json', report)
    dump(results / 'spec2_diagnostics_summary.json', report)
    lines = ['原六 experts、16-token TRAIN/VALID 的 shared residual 診斷。TRAIN 為 600 prompts／321 articles／9600 rows；VALID 為 200 prompts／3200 rows。所有 feature selection 與擬合僅使用 TRAIN。', '',
             '下表列出完整 B ridge {.03, .3, 3} 網格的增益範圍；正數代表較匹配 raw-x 的 VALID selected-alpha² NMSE 降低，沒有據此凍結新的部署 winner。raw-x 與 z nuisance ridge 固定為 1，conditional fit beta=0。', '',
             '| Layer／Expert | β=.99 row N_eff | FP conditional 128 | FP conditional 256 | W8A8 folded 128 | W8A8 folded 256 |',
             '|---|---:|---:|---:|---:|---:|']
    def interval(row, width, method):
        values = [100*r['relative_VALID_selected_alpha2_NMSE_reduction'] for r in row['gain_by_width'][str(width)][method]]
        return f'{min(values):+.2f}% 至 {max(values):+.2f}%'
    for row in rows:
        fp = 'FP_frozen_raw_x_plus_conditional_shared_residual'
        q = 'W8A8_folded_conditional_no_joint_refit'
        lines.append(f"| {row['layer']}／{row['expert']} | {row['TRAIN_weight_only_row_N_eff_by_beta']['0.99']:.1f} | {interval(row,128,fp)} | {interval(row,256,fp)} | {interval(row,128,q)} | {interval(row,256,q)} |")
    lines.extend(['', '只有 layer26／expert1 在此網格達到 10% 級別的局部改善；前兩層尚未達新版規格的非最後層機制門檻。所有六個 TRAIN experts 在 BOS 都未被選中，因此 BOS selected-alpha² NMSE 為 null，另保留 all-row NMSE。', '',
                  '這個結果只說明相對固定 ridge 模型的條件增益。z residualization 使用 ridge，仍可能保留因 shrinkage 而未移除的線性 x 訊號；不能直接宣稱額外訊號全來自非線性。B 與 indices 看過整份 TRAIN OOF nuisance residual，TRAIN 分數也不是 B 的無偏 OOF 泛化估計。獨立 VALID 才用來觀察未參與擬合的增益。', '',
                  'FP folding、直接 W8A8 folding、QDQ joint refit 分開報告。實際六個 canonical 模型（2048／2176／2304 三種寬度、各兩個 expert）的 86,016 個輸出 floats 與 frozen C++ 逐 bit 一致；A8 codes／scales 亦相同。此檢查覆蓋數值規則，沒有測量壓縮部署速度。', '',
                  '完整各 λ、位置切片、TRAIN／VALID gap、β 支持度與文章／prompt 聚合 N_eff 保存在同目錄 spec2_conditional_layer*.json 與 spec2_support_layer*.json。未使用新的 TEST。'])
    (results / 'spec2_diagnostics.zh.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'status': 'complete', 'summary': str(results / 'spec2_diagnostics_summary.json')}))


if __name__ == '__main__':
    main()
