#!/usr/bin/env python3
"""Publish five-model held-out figures from verified, frozen corpus-300 runs."""
import argparse
import csv
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from run_corpus_traces import DATA, MODELS, ROOT, sha256, write_json
from validate_replay import validate

LABELS=dict(granite='Granite',olmoe='OLMoE',deepseek='DeepSeek',qwen3='Qwen3',mixtral='Mixtral')
TASK_LABELS=dict(qa='Q&A',programming='Code',mathematics='Math',summarization='Summary',
                 translation='Translation',long_context='Long-context')
POLICIES=('baseline_16_separate','baseline_16_stacked','modulo_selected','learned_selected')
COLORS=('#a4adb5','#5f7b92','#edb458','#287e79')
NAMES=('g=16, separate','g=16, stacked','Selected g/layout, ID order','Selected g/layout, learned placement')


def save(fig,output,name):
    for extension in ('png','pdf','svg'):
        fig.savefig(output/(name+'.'+extension),dpi=180,bbox_inches='tight')
    plt.close(fig)


def gather(run_dir,require_replays):
    all_models={}
    for name in MODELS:
        analysis=run_dir/name/'analysis'
        plan_path=analysis/'frozen_plan.json'; plan=json.loads(plan_path.read_text())
        receipt=json.loads((analysis/'evaluation_receipt.json').read_text())
        assert receipt['frozen_plan_sha256']==sha256(plan_path)
        validation=json.loads((run_dir/name/'validation.json').read_text())
        assert len(validation)==300 and all(r['baseline_output_identical'] and r['baseline_token_ids_identical'] for r in validation)
        replays=[]
        manifest_path=run_dir/name/'replays/manifest.json'
        if require_replays:
            manifest=json.loads(manifest_path.read_text())
            assert manifest['frozen_plan_sha256']==sha256(plan_path)
            for job in manifest['workloads']:
                expected=Path(job['expected']); tag=Path(job['binary']).stem
                actual=manifest_path.parent/'nvmevirt'
                alias_path=actual/(tag+'.alias.json');source_tag=tag
                if alias_path.exists():
                    source_tag=json.loads(alias_path.read_text())['source_tag']
                    assert sha256(Path(job['binary']))==sha256(manifest_path.parent/(source_tag+'.bin'))
                result=validate(expected,actual/(source_tag+'.stats.txt'),actual/(source_tag+'.wall.json'),
                                actual/(source_tag+'.v2_stats.txt'))
                if source_tag!=tag:result['replay_alias_of']=source_tag
                replays.append(result)
        all_models[name]=dict(plan=plan,receipt=receipt,validation=validation,replays=replays,
                             collection=json.loads((run_dir/name/'provenance.json').read_text()),
                             summary=json.loads((analysis/'summary.json').read_text()),
                             strata=json.loads((analysis/'strata.json').read_text()),
                             per_prompt=json.loads((analysis/'per_prompt.json').read_text()))
    return all_models


def model_figure(name,model,output):
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    x=np.arange(4)
    for window,offset,color in ((32,-.17,'#709daf'),(128,.17,'#287e79')):
        row=next(r for r in model['summary'] if r['window']==window)
        bars=axes[0].bar(x+offset,[row['means_ms'][p] for p in POLICIES],width=.32,
                         color=color,label=f'First {window} inputs')
        axes[0].bar_label(bars,fmt='%.2f',padding=3,fontsize=8)
    axes[0].set_xticks(x,['g=16\nseparate','g=16\nstacked','Train-selected\nID order','Train-selected\nlearned'])
    axes[0].set_ylabel('Routed FFN ms / decode input');axes[0].legend(fontsize=8)
    labels=list(TASK_LABELS)
    rows=[next(r for r in model['strata'] if r['window']==128 and r['stratum']=='task' and r['label']==t) for t in labels]
    y=np.array([100*r['improvement_vs_16_stacked'] for r in rows])
    lo=np.array([100*r['improvement_vs_16_stacked_ci95'][0] for r in rows])
    hi=np.array([100*r['improvement_vs_16_stacked_ci95'][1] for r in rows])
    task_bars=axes[1].bar(np.arange(6),y,color='#287e79',yerr=np.maximum(0,np.stack([y-lo,hi-y])),capsize=3)
    if not np.any(y) and not np.any(lo) and not np.any(hi):
        axes[1].set_ylim(0,1)
        axes[1].bar_label(task_bars,labels=['0.0%']*len(labels),padding=3)
    axes[1].set_xticks(np.arange(6),[TASK_LABELS[t] for t in labels],rotation=20,ha='right')
    axes[1].set_ylabel('Reduction vs g=16 stacked (%)');axes[1].set_title('All 128 inputs; 95% prompt-bootstrap CI')
    for ax in axes:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
    winner=model['plan']['winners']['learned']
    fig.suptitle(f'{LABELS[name]}: 200 train / 100 held-out prompts; frozen g={winner["group_chips"]}, '
                 f'gate/up {"stacked" if winner["stack_gate_up"] else "separate"}')
    fig.tight_layout();save(fig,output,'model_'+name)


def figures(models,output,legacy):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'figure.facecolor':'white','axes.titleweight':'bold'})
    x=np.arange(5)
    # Normalization permits comparison across models with very different I/H.
    fig,axes=plt.subplots(1,2,figsize=(13,4),sharey=True)
    for ax,window in zip(axes,(32,128)):
        for j,(policy,color,label) in enumerate(zip(POLICIES,COLORS,NAMES)):
            values=[]
            for name in MODELS:
                r=next(r for r in models[name]['summary'] if r['window']==window)
                values.append(r['means_ms'][policy]/r['means_ms']['baseline_16_separate'])
            bars=ax.bar(x+(j-1.5)*.19,values,width=.18,color=color,label=label)
            if policy=='learned_selected':
                ax.bar_label(bars,labels=[f'{v:.3f}' for v in values],padding=3,fontsize=8)
        ax.set_xticks(x,[LABELS[n] for n in MODELS]);ax.set_ylim(0,1.16)
        ax.set_title(f'First {window} decode inputs');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0].set_ylabel('Routed FFN latency / g=16 separate latency')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=2,bbox_to_anchor=(.5,-.1))
    fig.suptitle('100 held-out prompts per model; one configuration trained on 200 prompts')
    fig.tight_layout();save(fig,output,'heldout_latency')

    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for window,offset,color in ((32,-.16,'#709daf'),(128,.16,'#287e79')):
        rows=[next(r for r in models[n]['summary'] if r['window']==window) for n in MODELS]
        y=np.array([r['improvement_vs_16_stacked']*100 for r in rows])
        lo=np.array([r['improvement_vs_16_stacked_ci95'][0]*100 for r in rows])
        hi=np.array([r['improvement_vs_16_stacked_ci95'][1]*100 for r in rows])
        axes[0].bar(x+offset,y,width=.30,color=color,label=f'Test: {window} steps',
                    yerr=np.maximum(0,np.stack([y-lo,hi-y])),capsize=3)
        placement=[r['placement_improvement']*100 for r in rows]
        axes[1].bar(x+offset,placement,width=.30,color=color,label=f'Test: {window} steps')
    train_gain=[];train_placement=[]
    for name in MODELS:
        plan=models[name]['plan'];winner=plan['winners']['learned']
        baseline=next(r for r in plan['candidates'] if r['policy']=='modulo' and
                      r['group_chips']==16 and r['stack_gate_up'])
        same_config=next(r for r in plan['candidates'] if r['policy']=='modulo' and
                         r['group_chips']==winner['group_chips'] and
                         r['stack_gate_up']==winner['stack_gate_up'])
        train_gain.append(100*(1-winner['training_ffn_ms_per_token']/baseline['training_ffn_ms_per_token']))
        train_placement.append(100*(1-winner['training_ffn_ms_per_token']/same_config['training_ffn_ms_per_token']))
    for ax,values in zip(axes,(train_gain,train_placement)):
        ax.scatter(x,values,marker='D',facecolors='white',edgecolors='#333',s=36,
                   zorder=5,label='Train: 128 steps')
    axes[0].set_title('Improvement over g=16 stacked');axes[1].set_title('Expert placement gain at same g/layout')
    for ax in axes:
        ax.set_xticks(x,[LABELS[n] for n in MODELS]);ax.set_ylabel('Latency reduction (%)')
        ax.axhline(0,color='#555',linewidth=.8);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0].legend();fig.suptitle('200 train / 100 held-out prompts; fixed configuration; 95% prompt-bootstrap CI')
    fig.tight_layout();save(fig,output,'heldout_improvement')

    fig,axes=plt.subplots(1,2,figsize=(13,4),gridspec_kw={'width_ratios':[2,1]})
    panels=(('task',list(TASK_LABELS),axes[0]),('length_bucket',['short','medium','long'],axes[1]))
    matrices=[]
    for field,labels,ax in panels:
        matrix=np.array([[next(r['improvement_vs_16_stacked']*100 for r in models[n]['strata']
                              if r['window']==128 and r['stratum']==field and r['label']==label)
                          for label in labels] for n in MODELS]);matrices.append(matrix)
    maximum=max(1,max(float(np.abs(a).max()) for a in matrices))
    for (field,labels,ax),matrix in zip(panels,matrices):
        im=ax.imshow(matrix,cmap='BrBG',vmin=-maximum,vmax=maximum,aspect='auto')
        ax.set_xticks(np.arange(len(labels)),[TASK_LABELS.get(l,l.title()) for l in labels],rotation=20,ha='right')
        ax.set_yticks(x,[LABELS[n] for n in MODELS])
        for i in range(5):
            for j in range(len(labels)):
                ax.text(j,i,f'{matrix[i,j]:.1f}%',ha='center',va='center',
                        color='white' if abs(matrix[i,j])>.65*maximum else '#202020')
        ax.set_title('By task' if field=='task' else 'By input length')
    fig.colorbar(im,ax=axes.tolist(),label='Reduction vs g=16 stacked (%)',fraction=.02,pad=.03)
    fig.suptitle('All 128 decode inputs on held-out prompts',y=1.02)
    save(fig,output,'task_length_generalization')

    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for ax,name in zip(axes.flat,MODELS):
        candidates=models[name]['plan']['candidates']
        for policy,color in (('modulo','#d99530'),('learned','#287e79')):
            for stacked,style in ((False,'--'),(True,'-')):
                rows=sorted((r for r in candidates if r['policy']==policy and r['stack_gate_up']==stacked),
                            key=lambda r:r['group_chips'])
                ax.plot([r['group_chips'] for r in rows],[r['training_ffn_ms_per_token'] for r in rows],
                        marker='o',color=color,linestyle=style,
                        label=policy+(' / stacked' if stacked else ' / separate'))
        winner=models[name]['plan']['winners']['learned']
        ax.scatter([winner['group_chips']],[winner['training_ffn_ms_per_token']],marker='*',s=180,
                   color='#b94a44',zorder=5,label='Training winner')
        ax.set_xscale('log',base=2);ax.set_xticks([1,2,4,8,16],[1,2,4,8,16]);ax.set_yscale('log')
        ax.set_title(LABELS[name]);ax.set_xlabel('Chips per expert (g)');ax.set_ylabel('Training FFN ms / input')
        ax.grid(alpha=.2)
    axes.flat[5].axis('off');h,l=axes.flat[0].get_legend_handles_labels();axes.flat[5].legend(h,l,loc='center')
    fig.suptitle('Configuration selection uses only the 200 training prompts')
    fig.tight_layout();save(fig,output,'training_group_sweep')

    fig,axes=plt.subplots(1,2,figsize=(12,4),sharey=True)
    for ax,split in zip(axes,('train','test')):
        positions=[];data=[]
        for i,name in enumerate(MODELS):
            validation=models[name]['validation']
            for j,length in enumerate(('short','medium','long')):
                data.append([r['prompt_tokens'] for r in validation if r['split']==split and r['length_bucket']==length])
                positions.append(i+(j-1)*.22)
        b=ax.boxplot(data,positions=positions,widths=.19,patch_artist=True,showfliers=False,
                     medianprops={'color':'#222'})
        for i,box in enumerate(b['boxes']):box.set_facecolor(('#b4d4c3','#70a5ad','#3c687f')[i%3])
        ax.set_xticks(x,[LABELS[n] for n in MODELS]);ax.set_title('200 train prompts' if split=='train' else '100 test prompts')
        ax.set_ylabel('Actual prompt tokens');ax.grid(axis='y',alpha=.2)
    from matplotlib.patches import Patch
    fig.legend([Patch(facecolor=c) for c in ('#b4d4c3','#70a5ad','#3c687f')],['Short','Medium','Long'],
               ncol=3,loc='lower center',bbox_to_anchor=(.5,-.05))
    fig.suptitle('Input lengths measured by each model tokenizer')
    fig.tight_layout();save(fig,output,'input_lengths')

    fig,ax=plt.subplots(figsize=(10,4))
    old=[100*r['improvement_vs_16_stacked'] for name in MODELS for r in legacy if r['model']==name]
    new=[100*next(r['improvement_vs_16_stacked'] for r in models[n]['summary'] if r['window']==32) for n in MODELS]
    ax.bar(x-.18,old,width=.34,color='#a4adb5',label='Prior: 3 prompts, 32 inputs, three-fold selection')
    ax.bar(x+.18,new,width=.34,color='#287e79',label='New: 100 held-out prompts, first 32 of 128 inputs')
    ax.set_xticks(x,[LABELS[n] for n in MODELS]);ax.set_ylabel('Reduction vs g=16 stacked (%)')
    ax.set_title('32-step context comparison: different workloads and training protocols')
    ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.legend(fontsize=8)
    fig.tight_layout();save(fig,output,'legacy_32_comparison')


def report(models,output,run_dir,replays_verified):
    lines=['# 300-prompt 五模型固定配置實驗','',
           '五個模型各用同一組 300 題：200 題學習 expert 固定排列與選擇 g、gate/up 串接，100 題保留測試。每題執行 128 次實際 decode；前 32 步直接取自同一次執行。','',
           '題庫涵蓋問答、程式、數學、摘要、翻譯、長文理解，各 50 題；short／medium／long 共 104／100／96 題。題庫是本地編寫、可重現的合成工作負載，並非公開品質基準。訓練與測試情境 ID、名稱與數值分開；模板與部分題型共用，因此結論範圍是這組工作負載的新輸入。任務／長度分組是描述統計，各長度不保證相同子題型組成，不能當作純輸入長度的因果比較。','',
           '配置只依 200 題的完整 128 步選定。配置與 SHA-256 先寫入 `frozen_plan.json`，之後才讀入測試 routing 做評估；前 32 步與完整 128 步均使用相同配置。每層每個 g 執行 512 次固定種子的容量守恆 swaps；這是啟發式搜尋，不保證全域最優。','',
           '| 模型 | g | gate/up | 16-chip 串接基線 ms/input | 學習配置 ms/input | 改善（95% CI） | 相同 g/layout 的排列增益 |',
           '|---|---:|---|---:|---:|---:|---:|']
    for n in MODELS:
        model=models[n];r=next(r for r in model['summary'] if r['window']==128);w=model['plan']['winners']['learned']
        lo,hi=r['improvement_vs_16_stacked_ci95']
        lines.append(f'| {LABELS[n]} | {w["group_chips"]} | {"串接" if w["stack_gate_up"] else "分開"} | '
                     f'{r["means_ms"]["baseline_16_stacked"]:.4f} | {r["means_ms"]["learned_selected"]:.4f} | '
                     f'{100*r["improvement_vs_16_stacked"]:.2f}% ({100*lo:.2f}–{100*hi:.2f}%) | {100*r["placement_improvement"]:.2f}% |')
    lines += ['', '信賴區間以整個 prompt 為抽樣單位、配對 bootstrap 2,000 次；同一題的 128 個 decode inputs 與各層不當作獨立樣本。輸出包含每題、每個 token、任務與輸入長度分組結果，並保留無改善或負改善。','',
              '所有 1,500 次 traced 推論各有一個完整 128 步 untraced greedy 對照，token IDs 與輸出 bytes 必須一致；每筆 routing 檢查 layer、token position、實際 token ID、expert 數量及範圍。Mixtral 使用 32 個獨立 sequence 的批次；其 untraced 對照從相同 prefill KV snapshot 啟動，再獨立執行全部 128 步。其他四模型使用 8 個 sequence 並各自重新 prefill。批次化會影響數值運算次序，因此不宣稱與舊單序列推論有逐位元相同的 routing。','',
              '向量化時序模型依每個 chip 的 expert 數量計算 flash、控制及向量傳輸；每個 g、layout、policy、layer 均與原始 scalar 模型核對，涵蓋六任務、三長度與第 1／32／128 步。','']
    if replays_verified:
        count=sum(len(m['replays']) for m in models.values())
        unique=[r for m in models.values() for r in m['replays'] if 'replay_alias_of' not in r]
        commands=sum(r['commands'] for r in unique)
        lines += [f'NVMeVirt 已涵蓋全部 100 題測試輸入：每模型固定學習配置與 16-chip 串接基線各一套，分成 prefix32 與 tail96，共 {count} 個配置／步段核對、{len(unique)} 個不同 workloads、{commands:,} 次真實 NVMe ioctl。完全相同的 descriptor stream 以逐 byte／SHA-256 等價驗證共用一次真實重播，另存 alias 記錄；不同的配置完整重播。全部 kernel counters 與 Python 相符；prefix32 和 tail96 合起來正好是完整 128 步。VM ioctl wall time 另外保留，不能作為實體 AiF 延遲。','']
    lines += ['![Held-out latency](heldout_latency.png)','![Improvement](heldout_improvement.png)',
              '![Task and length](task_length_generalization.png)','![Training sweep](training_group_sweep.png)',
              '![Input lengths](input_lengths.png)','![Legacy comparison](legacy_32_comparison.png)','',
              '舊三題結果與新的前 32 步比較圖使用相同時序參數，但輸入分布與配置學習方法不同；差異不能單獨歸因於 decode 長度。','',
              '延遲是 routed FFN projection 的 AiF 時序模型值，非端到端 tokens/s 或實體 flash PE/MAC 量測。INT8 權重、16 chips、4 planes/chip、16 KiB page、28 µs cold read、9.7 µs cr-read、6.4 GB/s/chip 沿用舊實驗。控制成本 250 ns/chip command、2 µs/host phase、input INT8／output FP32 是未校準的假設。Attention、router、shared experts、dense layers、activation/combine、DRAM caching、GC 與跨層 overlap 未納入。','',
              f'原始 traces、文字輸出、128／32 token IDs、配置、時序、虛擬 NAND 地址計畫與重播證據：`{run_dir}`。所有圖同時提供 PNG、PDF、SVG。']
    (output/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,default=DATA/'runs/corpus-300-v1')
    p.add_argument('--output-dir',type=Path,default=ROOT/'results/corpus-300-v1')
    p.add_argument('--without-replays',action='store_true',help='Explicit preliminary analysis; marks manifest accordingly.')
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    models=gather(a.run_dir,not a.without_replays)
    legacy=json.loads((ROOT/'results/summary.json').read_text())
    figures(models,a.output_dir,legacy);report(models,a.output_dir,a.run_dir,not a.without_replays)
    for name in MODELS:model_figure(name,models[name],a.output_dir)
    write_json(a.output_dir/'summary.json',[r for n in MODELS for r in models[n]['summary']])
    write_json(a.output_dir/'strata.json',[r for n in MODELS for r in models[n]['strata']])
    flat=[]
    for name in MODELS:
        winner=models[name]['plan']['winners']['learned']
        for row in models[name]['summary']:
            flat.append(dict(model=name,window=row['window'],prompts=row['prompts'],
                group_chips=winner['group_chips'],stack_gate_up=winner['stack_gate_up'],
                **{p+'_ms':v for p,v in row['means_ms'].items()},
                improvement_vs_16_stacked=row['improvement_vs_16_stacked'],
                ci95_low=row['improvement_vs_16_stacked_ci95'][0],ci95_high=row['improvement_vs_16_stacked_ci95'][1],
                placement_improvement=row['placement_improvement']))
    with (a.output_dir/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    placements=[]
    for name in MODELS:
        plan=models[name]['plan'];winner=plan['winners']['learned'];g=winner['group_chips']
        for layer,mapping in plan['mappings'][f'g{g}'].items():
            for expert,slot in enumerate(mapping):
                group=slot%(16//g)
                placements.append(dict(model=name,layer=int(layer),logical_expert=expert,slot=slot,
                    group_id=group,first_chip=group*g,last_chip=group*g+g-1,group_chips=g,
                    stack_gate_up=winner['stack_gate_up']))
    with (a.output_dir/'winner_placements.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(placements[0]));writer.writeheader();writer.writerows(placements)
    shutil.copyfile(a.run_dir/'corpus/prompts.jsonl',a.output_dir/'prompts.jsonl')
    prompt_stats=[]
    for name in MODELS:
        shutil.copyfile(a.run_dir/name/'analysis/frozen_plan.json',a.output_dir/(name+'.frozen_plan.json'))
        for row in models[name]['validation']:
            prompt_stats.append(dict(model=name,**{k:row[k] for k in
                ('id','split','task','length_bucket','prompt_tokens','decode_tokens','routing_records',
                 'sequence_batch','baseline_output_identical')}))
    with (a.output_dir/'prompt_stats.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(prompt_stats[0]));writer.writeheader();writer.writerows(prompt_stats)
    write_json(a.output_dir/'experiment_manifest.json',dict(schema_version=1,date='2026-10-27',timezone='Asia/Taipei',
               run_dir=str(a.run_dir),corpus_sha256=sha256(a.run_dir/'corpus/prompts.jsonl'),
               corpus=json.loads((a.run_dir/'corpus/manifest.json').read_text()),
               models={n:dict(frozen_plan_sha256=models[n]['receipt']['frozen_plan_sha256'],
                         collection=models[n]['collection'],
                         validation_prompts=len(models[n]['validation']),replays=models[n]['replays'],
                         winners=models[n]['plan']['winners'],evaluation=models[n]['receipt']) for n in MODELS},
               all_1500_traced_greedy_controls_pass=True,all_nvmevirt_replays_verified=not a.without_replays,
               code_checks=json.loads((a.run_dir/'code_checks.json').read_text()) if (a.run_dir/'code_checks.json').exists() else None,
               graph_script_sha256=sha256(Path(__file__)),legacy_summary_sha256=sha256(ROOT/'results/summary.json')))
    print(a.output_dir,flush=True)


if __name__=='__main__':main()
