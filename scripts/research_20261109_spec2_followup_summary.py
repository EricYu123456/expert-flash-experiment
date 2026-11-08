#!/usr/bin/env python3
"""Combine spec2 sensitivity and frozen-raw residual calibration artifacts."""
import json
from pathlib import Path

from research_20261109_shared_feature_targets import DATA, ROOT, dump, sha


def main():
    base=DATA/'spec2-frozen-raw-residual'
    inputs,rows={},[]
    checked=0
    for layer in (1,13,26):
        path=base/f'layer{layer}/selection.json'
        x=json.loads(path.read_text())
        assert x['status']=='complete'
        inputs[str(path)]=sha(path)
        checked+=x['canonical_CPP_prediction_floats_bitchecked']
        raw={r['expert']:r for r in x['raw_baselines']}
        for r in x['models']:
            rows.append({'layer':layer,'expert':r['expert'],'width_shared':r['shared_width'],
                         'chosen_B_beta':r['beta'],'chosen_B_ridge':r['ridge_B'],
                         'original_raw_beta':raw[r['expert']]['original_beta'],
                         'original_raw_lambda':raw[r['expert']]['original_lambda'],
                         'raw_VALID_selected_alpha2_nmse':raw[r['expert']]['metrics']['validation']['all']['selected_alpha2_nmse'],
                         'candidate_VALID_selected_alpha2_nmse':r['metrics']['validation']['all']['selected_alpha2_nmse'],
                         'relative_VALID_reduction_vs_raw':r['VALID_relative_selected_alpha2_NMSE_reduction_vs_original_raw'],
                         'relative_VALID_reduction_vs_beta_matched_B0':r['VALID_relative_selected_alpha2_NMSE_reduction_vs_matched_B0'],
                         'fallback_to_raw':r['fallback_to_original_raw'],
                         'readout_prefix':r['readout_prefix'],'transform_prefix':r['transform_prefix']})
    registries={}
    names=['registry.raw_fallback.tsv']+[f'registry.z{w}.{kind}.tsv' for w in (128,256) for kind in ('candidates','with_raw_fallback','B0_control')]
    for name in names:
        path=base/name
        lines=[]
        for layer in (1,13,26):
            lines.extend((base/f'layer{layer}'/name).read_text().splitlines())
        assert len(lines)==6 and len({tuple(line.split('\t')[:2]) for line in lines})==6
        path.write_text('\n'.join(lines)+'\n')
        registries[name]={'path':str(path),'sha256':sha(path)}
    sensitivity_path=DATA/'spec2-nuisance-sensitivity/layer26/report.json'
    sensitivity=json.loads(sensitivity_path.read_text())
    sensitivity_rows=[{'P_ridge':r['P_ridge'],'P_effective_df':r['P_effective_df'],
                       'shared_width':r['width'],'B_ridge':r['B_ridge'],
                       'relative_VALID_reduction':r['relative_VALID_selected_alpha2_NMSE_reduction']['all']}
                      for r in sensitivity['records']]
    result={'status':'complete','source_sha256':sha(Path(__file__)),'input_sha256':{**inputs,str(sensitivity_path):sha(sensitivity_path)},
            'grid_models_evaluated':144,'B0_controls_evaluated':48,'chosen_models':rows,
            'canonical_CPP_prediction_floats_bitchecked':checked,'registries':registries,
            'nuisance_ridge_sensitivity':sensitivity_rows,
            'mechanism_result':'No optimized-raw comparison reaches 10% VALID local selected-alpha2 error reduction. Nonfinal layers improve at most 0.685%; layer26/E1 improves at most 2.218%.',
            'limits':['These are VALID-selected local candidates, not blind TEST estimates or full-model NLL results.',
                      'A8 context, re-W8 and bias calibration make B=0 concat differ from original raw. Exact original raw remains available as a separate fallback.',
                      'Original raw W8 dequantized coefficients are frozen before concat re-quantization; their post-requantization bits can change.',
                      'No new runtime, additive dual contexts, learned GU, TEST access or system-speed measurement.'],
            'coefficient_cost':{'readout_width_raw':2048,'readout_width_hybrid128':2176,'readout_width_hybrid256':2304,
                                'per_event_original_expert_coefficient_count':3*2048*1408,
                                'per_event_raw_coefficient_saving_fraction':1-2048/(3*1408),
                                'per_event_hybrid128_coefficient_saving_fraction':1-2176/(3*1408),
                                'per_event_hybrid256_coefficient_saving_fraction':1-2304/(3*1408),
                                'hybrid128_coverage_ratio_needed_vs_raw':1.0625,
                                'hybrid256_coverage_ratio_needed_vs_raw':1.1333333333333333,
                                'scope':'Coefficient work only. No equal-quality coverage improvement has been demonstrated; retaining raw+C simultaneously adds fallback residency cost.'}}
    dump(base/'summary.json',result)
    results=ROOT/'results/research-2026-11-09-shared-features'
    dump(results/'spec2_followup_summary.json',result)
    lines=['新的公平比較把原六 experts 的 optimized raw-x（各自已選 β／λ）完全保留為獨立 fallback。用該模型的實際 canonical W8A8 TRAIN 殘差擬合 shared128／256 小 B，再把 head fold 成單一 concat readout、重新 row-W8、whole-vector A8 與 TRAIN bias 校準。β={0,.5,.9,.99}、B ridge={.03,.3,3} 的選擇只用 calibration-VALID。', '',
           '| Layer／Expert | shared128 最佳 VALID 誤差改善 | shared256 最佳 VALID 誤差改善 |',
           '|---|---:|---:|']
    for layer in (1,13,26):
        for expert in {1:(14,40),13:(10,56),26:(0,1)}[layer]:
            pair=[next(r for r in rows if r['layer']==layer and r['expert']==expert and r['width_shared']==w) for w in (128,256)]
            lines.append(f"| {layer}／{expert} | {100*pair[0]['relative_VALID_reduction_vs_raw']:+.4f}% | {100*pair[1]['relative_VALID_reduction_vs_raw']:+.4f}% |")
    lines.extend(['', '改善均以 selected-alpha² 加權的原生 expert 輸出 NMSE 計算。前兩層最大只有 0.685%，晚層 E1 最大 2.218%，此組候選未達約 10% 的投入門檻。VALID 選過超參數，所以這些數字也不是 blind TEST 的改善估計。', '',
                  '原先固定 ridge=1 的晚層 FP conditional 增益有明顯 shrinkage 混淆。保持 raw-x ridge=1 與原 indices，僅把 z-on-x 的 P ridge 從 1 降為 .01／.001 後，E1 的 shared128／256 最佳增益由約 14.56%／18.79% 降為 4.05%／4.97% 與 3.51%／4.21%。這比直接把舊增益歸為共享非線性訊號更符合證據。', '',
                  '144 個 B 候選與 48 個 β 匹配 B=0 控制均已評估；匯出各 width 的候選、帶原 raw fallback 的 registry 與 B0 registry。原 raw 所有 payload hash 不變，且獨立重算的 VALID 分數完全相同。所選 heads 與 B0 共 344,064 個輸出 floats 與 frozen C++ 逐 bit 一致；concat gather／F32 transform 亦已核對。B=0 concat 和原 raw 仍有小差異，因此沒有假定兩者 bit-equivalent。', '',
                  'shared128／256 每次替換的 raw 係數節省率為 48.48%／45.45%，raw-x 為 51.52%。因此還需約 6.25%／13.33% 更高、且品質相當的替換覆蓋率才能補回寬 head 的係數工作成本；本次沒有證明這種 coverage 改善，也沒有測量 flash 或部署速度。保留 raw fallback 的額外 resident bytes 仍須另計。', '',
                  '全部大產物與可直接供既有 adapter 使用的六列 registry 位於 NFS 的 spec2-frozen-raw-residual；BOS／非 BOS／早晚位置切片與 TRAIN／VALID 指標可由各 layer 的 selection.json 讀取。此次沒有開新 TEST。'])
    (results/'spec2_followup.zh.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'status':'complete','registries':registries,'canonical_CPP_floats':checked}))


if __name__=='__main__':main()
