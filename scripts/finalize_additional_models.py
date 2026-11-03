#!/usr/bin/env python3
"""Publish the Qwen3/Mixtral extension only after real traces and replays pass."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LABELS = {'granite':'Granite 3.1 1B A400M', 'olmoe':'OLMoE 1B 7B',
          'deepseek':'DeepSeek V2 Lite', 'qwen3':'Qwen3 30B A3B', 'mixtral':'Mixtral 8x7B'}
NEW = {'qwen3':('Qwen3-30B-A3B-Instruct-2507-FP8', 'qwen3-30b-a3b-instruct-2507-q8_0.gguf'),
       'mixtral':('Mixtral-8x7B-Instruct-v0.1', 'mixtral-8x7b-instruct-v0.1-q8_0.gguf')}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def costs(value):
    return {key:value[key]/value['decode_tokens']/1e6 for key in
            ('flash_ns','control_ns','host_ns','input_ns','output_ns','model_ns')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('/home/nfs_client/ericyu/expert_flash_experiment'))
    parser.add_argument('--date', default='2026-10-27')
    args = parser.parse_args()
    output = ROOT/'results'
    original = read(output/'archive/2026-10-19/experiment_manifest.json')
    manifest = dict(original)
    manifest.update(date=args.date, previous_experiment_date=original['date'],
                    original_results_archive=str(output/'archive/2026-10-19'))
    manifest['models'] = list(original['models'])
    added = {}
    for name, (directory, filename) in NEW.items():
        source = args.data/'models'/(name+'-source-manifest.json')
        assert read(source)['index_tensors_and_payload_ranges_valid']
        metadata_path = args.data/'models'/(name+'-metadata.json')
        metadata = read(metadata_path)
        numerical_path = args.data/'models'/(name+'-conversion-validation.json')
        assert read(numerical_path)['all_checks_passed']
        trace_dir = args.data/'runs'/(name+'-q8')
        checks = read(trace_dir/'validation.json')
        assert len(checks) == 3 and all(c['baseline_output_identical'] and c['decode_tokens']==32 for c in checks)
        tokenization = read(trace_dir/'hf-tokenization-validation.json')
        assert len(tokenization['checks']) == 3 and all(c['hf_llama_token_ids_identical'] for c in tokenization['checks'])
        run = args.data/'runs'/(name+'-aif')
        replays = []
        for phase, expected_count in (('overhead_fanout',10),('learned_overhead_fanout',6)):
            expected_files = sorted((run/phase).glob('*.expected.json'))
            assert len(expected_files) == expected_count, (name,phase,len(expected_files))
            for expected_path in expected_files:
                tag = expected_path.name.removesuffix('.expected.json')
                validation_path = run/phase/'nvmevirt'/(tag+'.validation.json')
                result = read(validation_path)
                assert result['all_kernel_counters_match'] and result['overhead_counters_checked']
                assert read(expected_path)['model']=='metadata_only_int8_v3'
                replays.append(validation_path)
        model = dict(name=name, added_date=args.date, model_path=str((args.data/'models'/filename).resolve()),
                     model_bytes=metadata['model_bytes'], model_sha256=metadata['sha256'],
                     routing_files={str(p.resolve()):sha(p) for p in sorted(trace_dir.glob('*.routing.jsonl'))},
                     trace_validation=checks, source_directory='/home/nfs_client/ymtuan2/models/'+directory,
                     source_manifest=str(source.resolve()), source_files_read_only=True,
                     tokenizer_validation=str((trace_dir/'hf-tokenization-validation.json').resolve()),
                     sampled_weight_validation=str(numerical_path.resolve()),
                     routing_weight_format='Q8_0 GGUF', timing_weight_format='INT8 one byte per weight')
        if name == 'mixtral':
            repair_path = args.data/'models/mixtral-tokenizer-compat-validation.json'
            repair = read(repair_path)
            assert repair['changed_header_bytes']==1 and repair['tensor_payload_not_written']
            repair['full_model_sha256_status'] = 'computed_after_metadata_repair'
            repair['full_model_sha256_after'] = metadata['sha256']
            previous_metadata = args.data/'models/mixtral-metadata-before-tokenizer-fix.json'
            if previous_metadata.exists():
                repair['full_model_sha256_before'] = read(previous_metadata)['sha256']
            repair['hf_llama_token_ids_identical_for_all_three_prompts'] = True
            repair_path.write_text(json.dumps(repair,indent=2)+'\n')
            model['tokenizer_compatibility_validation'] = str(repair_path.resolve())
        manifest['models'].append(model)
        architecture = metadata['metadata']['general.architecture']
        expert = next(t for t in metadata['tensors'] if t.get('projection')=='gate')
        h,i,e = expert['shape_ggml']
        group_summary = read(run/'learned_overhead_fanout/summary.json')
        plans = read(run/'learned_overhead_fanout/plans.json')
        assert all(p['stack_gate_up'] for p in plans), 'Report must distinguish unstacked selected layouts'
        fixed = rows(run/'overhead_fanout/overhead_placement.csv')
        fixed = [r for r in fixed if r['stack_gate_up']=='True']
        base = costs(read(run/'overhead_fanout/g16_stacked_overhead.expected.json'))
        learned = [costs(read(p)) for p in sorted((run/'learned_overhead_fanout').glob('*_learned_selected.expected.json'))]
        selected = {key:sum(v[key] for v in learned)/len(learned) for key in base}
        added[name] = dict(hidden=h, intermediate=i, experts=e,
                           selected_experts=metadata['metadata'][architecture+'.expert_used_count'],
                           routed_layers=len(metadata['routed_layer_ids']), projection_mib=h*i/2**20,
                           total_routing_records=sum(c['routing_records'] for c in checks),
                           base_costs=base, learned_costs=selected,
                           original_groups=sorted({p['group_chips'] for p in plans if p['policy']=='modulo'}),
                           learned_groups=sorted({p['group_chips'] for p in plans if p['policy']=='learned'}),
                           group_sweep=[dict(g=int(r['group_chips']),ms=float(r['ffn_ms_per_token'])) for r in fixed],
                           replays=len(replays), physical_plan_summary=read(run/'learned_overhead_fanout/physical_plans/summary.json'))
    pid_path = args.data/'vm/qemu.pid'
    if pid_path.exists():
        try:
            os.kill(int(pid_path.read_text().strip()),0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError('Shut down the experiment VM before final publication')
    manifest['vm_state'] = 'stopped'
    scripts = ROOT/'scripts'
    subprocess.run([sys.executable,str(scripts/'collect_replay_validation.py'),
                    '--runs',str(args.data/'runs'),'--output',str(output/'nvmevirt_validation.json')],check=True)
    replay = read(output/'nvmevirt_validation.json')
    manifest['nvmevirt_validation'] = {k:v for k,v in replay.items() if k!='results'}
    for key in list(manifest['source_sha256']):
        manifest['source_sha256'][key] = sha(ROOT/key)
    for name in ('record_model_source.py','validate_converted_weights.py','fix_mixtral_tokenizer.py','collect_replay_validation.py','finalize_additional_models.py'):
        manifest['source_sha256']['scripts/'+name] = sha(scripts/name)
    manifest['extension_checks'] = dict(models=list(NEW), numerical_conversion_checks_passed=True,
                                       all_hf_tokenizations_match=True, all_greedy_outputs_unchanged=True,
                                       all_v3_replay_counters_match=True,
                                       prior_timing_and_kernel_sources_unchanged=all(
                                           manifest['source_sha256'][key]==original['source_sha256'][key]
                                           for key in ('scripts/aif_model.py','scripts/overhead_model.py','kernel/aif.c','kernel/aif.h')))
    assert manifest['extension_checks']['prior_timing_and_kernel_sources_unchanged']
    final_checks = read(args.data/'runs/additional-models-checks.json')
    manifest['previous_checks'] = original['checks']
    manifest['checks'] = final_checks['checks']
    manifest['extension_software_versions'] = final_checks['versions']
    manifest['extension_guest_environment'] = str(args.data/'runs/additional-models-final-guest-environment.json')
    (output/'experiment_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    subprocess.run([sys.executable,str(scripts/'plot_optimized.py'),
                    '--data',str(args.data/'runs'),'--output',str(output)],check=True)
    summary = read(output/'summary.json')
    assert {r['model'] for r in summary} == set(LABELS)
    for row in read(output/'archive/2026-10-19/summary.json'):
        assert row==next(r for r in summary if r['model']==row['model']), 'Original results changed'
    (output/'additional_models.json').write_text(json.dumps(added,indent=2)+'\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,2,figsize=(13,8),layout='constrained')
    keys = ['baseline_16_separate_ms','baseline_16_stacked_ms','selected_group_stacked_ms','learned_group_stacked_ms']
    labels = ['16 separate','16 stacked','Selected g','Learned + g']
    for row,(name, geometry) in enumerate(added.items()):
        current = next(v for v in summary if v['model']==name)
        values = [current[key] for key in keys]
        bars = axes[row,0].bar(np.arange(4),values,color=['#111111','#ADB4BA','#9CCDEE','#C83D47'])
        axes[row,0].bar_label(bars,fmt='%.3f',padding=4,fontsize=10)
        axes[row,0].set_xticks(np.arange(4),labels,fontsize=10)
        axes[row,0].set_ylim(0,max(values)*1.17)
        axes[row,0].set_title(LABELS[name]+' — held-out selected layouts')
        sweep = sorted(geometry['group_sweep'],key=lambda r:r['g'])
        axes[row,1].plot(np.arange(5),[p['ms'] for p in sweep],color='#111111',marker='o')
        axes[row,1].set_xticks(np.arange(5),[p['g'] for p in sweep])
        axes[row,1].set_xlabel('g: chips per expert')
        axes[row,1].set_title('Original fixed placement; stacked gate/up')
        for x,v in enumerate(sweep):
            axes[row,1].annotate(f"{v['ms']:.2f}",(x,v['ms']),xytext=(0,8),textcoords='offset points',ha='center',fontsize=10)
        axes[row,1].margins(y=.22)
        for axis in axes[row]:
            axis.set_ylabel('Modeled routed FFN service (ms/token)')
            axis.grid(axis='y',color='#E2E6EA'); axis.set_axisbelow(True)
            axis.spines[['top','right']].set_visible(False)
    fig.suptitle('Qwen3 / Mixtral extension — same real-route protocol and v3 costs\nLeft: three-fold held-out means. Right: all three prompts pooled; not held-out selection.',fontsize=13)
    for ext in ('png','pdf','svg'):
        fig.savefig(output/('additional_models.'+ext),dpi=200,bbox_inches='tight')
    plt.close(fig)
    lines = ['# MoE routed FFN 的 AiF 分組與放置實驗', '',
             f'更新日期：**{args.date}**。本次補測 Qwen3 與 Mixtral；Granite、OLMoE、DeepSeek 沿用 2026-10-19 保存的結果。原三模型數值已逐項核對未變，原始報告與圖表封存在 [archive/2026-10-19](archive/2026-10-19/)。', '',
             '全部比較使用相同 v3 controller-fanout 計時模型、三組 plain-text prompts、各 32 次 greedy decode。每折只用兩組 prompts 學習固定排列與選擇 g／串接，第三組保留驗證；報告三折保留平均。', '',
             '## 五模型主要結果', '',
             '單位：**routed FFN 服務時間，ms／decoded token；越低越好**。降低幅度比較「16-chip、gate/up 已串接」基準。', '',
             '| 模型 | 16-chip 分開 | 16-chip 串接 | 選 g＋串接，原排列 | 選 g＋串接＋學習排列 | 學習配置 g | 較已串接基準降低 |',
             '| --- | ---: | ---: | ---: | ---: | --- | ---: |']
    for row in summary:
        gs='/'.join(map(str,row['selected_groups']))
        lines.append(f"| {LABELS[row['model']]} | {row['baseline_16_separate_ms']:.3f} | {row['baseline_16_stacked_ms']:.3f} | {row['selected_group_stacked_ms']:.3f} | **{row['learned_group_stacked_ms']:.3f}** | {gs} | {row['improvement_vs_16_stacked']*100:.2f}% |")
    lines += ['', 'DeepSeek 的原排列選 g=16，學習排列選 g=8；其差異同時包含 g 與排列，不能完全歸給同一 g 下的 placement 交換。', '',
              '![五模型配置比較](optimized_layouts.png)', '',
              '## 補測得到的差異', '']
    q = next(r for r in summary if r['model']=='qwen3')
    m = next(r for r in summary if r['model']=='mixtral')
    lines += [f"**Qwen3：三折皆選 g=8。** 16-chip 已串接為 {q['baseline_16_stacked_ms']:.3f} ms，選分組後 {q['selected_group_stacked_ms']:.3f} ms，加學習排列後 {q['learned_group_stacked_ms']:.3f} ms，較已串接基準降低 {q['improvement_vs_16_stacked']*100:.2f}%。它的單 projection 是 1.5 MiB，補上原有 0.5／2／2.75 MiB 之間的尺寸。", '',
              f"**Mixtral：三折皆選 g=16，學習排列沒有額外改善。** 已串接與訓練選配置均為 {m['learned_group_stacked_ms']:.3f} ms。單 projection 為 56 MiB；即使分到 16 chips，每 chip 仍有 3.5 MiB、56 個四-plane read waves，已是長讀取。每次只選 2 個 experts，較小的 group 也會降低可用的 chip 並行度。", '',
              '上述是不同真實模型的觀察，同時包含 expert 大小、top-k、層數與 routing 差異，並非只改尺寸的控制實驗。', '',
              '![新增兩模型的配置與 g 掃描](additional_models.png)', '',
              '右側 g 掃描固定原排列、gate/up 已串接，使用三組 prompts 合併平均；與左側訓練選配置後的保留驗證程序不同。', '',
              '| 原排列已串接 | g=1 | g=2 | g=4 | g=8 | g=16 |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name, info in added.items():
        sweep=sorted(info['group_sweep'],key=lambda v:v['g'])
        lines.append('| '+LABELS[name]+' | '+' | '.join(f"{r['ms']:.3f}" for r in sweep)+' |')
    lines += ['', '## 模型尺寸與轉檔驗證', '',
              '| 模型 | H × I | Experts／top-k | Routed 層數 | INT8／projection | Routing records |',
              '| --- | --- | --- | ---: | ---: | ---: |']
    for model in manifest['models']:
        name=model['name']; md=read(args.data/'models'/(name+'-metadata.json'))
        tensor=next(t for t in md['tensors'] if t.get('projection')=='gate')
        h,i,e=tensor['shape_ggml']; arch=md['metadata']['general.architecture']
        k=md['metadata'][arch+'.expert_used_count']
        count=len({t['layer'] for t in md['tensors'] if t.get('projection')})
        records=sum(c['routing_records'] for c in model['trace_validation'])
        lines.append(f'| {LABELS[name]} | {h} × {i} | {e}／{k} | {count} | {h*i/2**20:g} MiB | {records:,} |')
    lines += ['', 'Qwen3 的既有 FP8 checkpoint 依 128×128 block scales 解量化後轉為 Q8_0；Mixtral 由既有 BF16 轉為 Q8_0。兩者各完成 13 組獨立權重抽查，最大相對 L2 誤差為 0.6873%／0.5693%，所有樣本皆為有限值。這是轉檔檢查，不是模型品質評測。AiF 時序仍使用 INT8，不能拿 GGUF 檔案 bytes 當作 INT8 權重大小。', '',
              'Mixtral 首次轉檔的前綴空白 metadata 與來源 tokenizer 不符。已修正唯一的一個 header byte、重算完整 hash，並重新擷取全部 routing。修正後兩模型的三組 prompt token IDs 均與本機 Hugging Face tokenizer 相同，tracing 開關的生成文字逐 byte 相同。原始來源與 pinned llama.cpp 未修改；詳見 [tokenizer 與 tracing 紀錄](../docs/llama_trace.md#mixtral-tokenizer-metadata-correction)。', '',
              '## 命令與傳輸成本', '',
              '沿用 0.25 μs／chip-block command、2 μs／host phase、輸入 1 byte／輸出 4 bytes 每元素。這些是展示假設，未經實體 AiF 校準。每份 logical input 經 PCIe 傳一次，再由 controller 跨 channels 複製；PCIe／channel 上限為 8／2 GB/s。', '',
              '| 模型與配置 | Flash | Chip commands | Host | 輸入 | 輸出 | 合計 |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name, info in added.items():
        for key,label in (('base_costs','16-chip 已串接'),('learned_costs','訓練選學習配置')):
            value=info[key]
            lines.append('| '+LABELS[name]+' '+label+' | '+' | '.join(f'{value[k]:.4f}' for k in
                ('flash_ns','control_ns','host_ns','input_ns','output_ns','model_ns'))+' |')
    lines += ['', '分項單位同為 ms/token。', '', '![控制成本敏感度](control_tradeoff.png)', '',
              '## 虛擬地址與容量', '',
              '| 新增模型 | 有用 routed INT8 | 保留 LSB 空間 | 對應 raw TLC cells |',
              '| --- | ---: | ---: | ---: |']
    for name, info in added.items():
        plans=[p for p in info['physical_plan_summary'] if p['policy']=='learned']
        values=[max(p[k] for p in plans)/2**30 for k in ('useful_int8_bytes','reserved_lsb_bytes','reserved_tlc_raw_bytes')]
        lines.append('| '+LABELS[name]+' | '+' | '.join(f'{v:g} GiB' for v in values)+' |')
    lines += ['', '容量取各折學習配置的最大值，包含所有 routed experts。已檢查 row 覆蓋、位址不重疊、pages、cold runs 與 bytes 守恆。地址為 metadata 計畫，未把模型權重寫進約 2 GB 的 replay namespace；raw TLC 以保留 LSB 的三倍計算，未含 ECC、備用 blocks 或其他模型權重。', '',
              '## 重播證據與解讀範圍', '',
              f"本次新增 **32 組 v3 workloads、222,720 個 NVMe commands**，全部經 QEMU/KVM guest 的 NVMeVirt queue。連同原三模型，已保存 **{replay['v3_workloads']} 組 v3、{replay['v3_commands']:,} 個命令**；所有版本合計 **{replay['workloads']} 組、{replay['total_commands']:,} 個命令**。已重新解析全部證據，逐 chip bytes/pages/read rounds/cold reads/service，以及控制、host、PCIe／channel 分項均與預期吻合；實際 ioctl wall time 另存。", '',
              '這證明 Python 與 kernel 的模型實作一致，並非獨立的實體 AiF 效能量測。服務時間排除 attention、router、activation/combine、shared/dense FFN、GC、DRAM cache 與跨 host/SSD tensor overlap，不能換算完整模型 tokens/s。三個短 prompts 是可重現的先導實驗，不保證其他流量的最佳 g 相同。', '',
              'VM 已正常關閉。原始模型與所有大型 artifacts 保留於 NFS；模型、程式與 routing hashes 見 [experiment_manifest.json](experiment_manifest.json)，重播逐項證據見 [nvmevirt_validation.json](nvmevirt_validation.json)，五模型數字見 [summary.csv](summary.csv)，新增模型分項見 [additional_models.json](additional_models.json)。', '']
    (output/'report.md').write_text('\n'.join(lines))
    print(json.dumps(dict(report_data=str(output/'additional_models.json'), models=len(summary),
                          v3_workloads=replay['v3_workloads'], v3_commands=replay['v3_commands']),indent=2))


if __name__ == '__main__':
    main()
