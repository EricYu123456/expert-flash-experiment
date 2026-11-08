#!/usr/bin/env python3
"""Render completed TEST results and preregistered costs; never select a model."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


LABELS={
    'prototype25_dense_k12':'Dense GU25',
    'prototype25_rank256_k12':'GU25 rank256',
    'raw_x_select1408_eq_rank256_k12':'Raw-x rank256',
    'selected_alpha2_mean_k12':'TRAIN constant'}
ZH={
    'prototype25_dense_k12':'Dense 共享 GU25',
    'prototype25_rank256_k12':'共享 GU25＋rank256（primary）',
    'raw_x_select1408_eq_rank256_k12':'同寬 raw-x＋rank256',
    'selected_alpha2_mean_k12':'TRAIN selected α² 常數'}


def read(p):return json.loads(Path(p).read_text())
def pct(x):return f'{100*x:+.3f}%'
def ci(values):return f'{pct(values[0])}～{pct(values[1])}'


def row(cost,budget,scope='nonBOS62',routing='fixed_native',policy='primary_v2'):
    name=f'{budget}.{routing}.{policy}'
    values=[r for r in cost['scopes'] if r['audit']==name and r['scope']==scope]
    assert len(values)==1
    return values[0]


def generation_exposure(item):
    phases=item['vs_native']['exposure']
    names=('last_prefix_for_first_generated','decode_causal')
    events=sum(phases.get(n,{}).get('routed_expert_events',0) for n in names)
    replaced=sum(phases.get(n,{}).get('replaced_events',0) for n in names)
    private=sum(phases.get(n,{}).get('private_latent_A8_rows',0) for n in names)
    return events,replaced,private


def figures(quality,cost,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    names=list(LABELS)
    fig,ax=plt.subplots(figsize=(8.0,3.5))
    for j,name in enumerate(names):
        q=quality['arms'][name]['vs_native'];point=q['relative_subset_PPL_change']*100
        lo,hi=np.asarray(q['ci95_relative_subset_PPL_change'])*100
        color=('#2468b4' if name==quality['primary_arm'] else '#686868')
        ax.errorbar(point,j,xerr=[[point-lo],[hi-point]],fmt='o',capsize=4,color=color)
    ax.axvline(0,color='black',lw=.8);ax.axvline(2,color='#b34a42',linestyle='--',label='Primary 2% margin')
    ax.set_yticks(range(len(names)),[LABELS[n] for n in names]);ax.invert_yaxis()
    ax.set_xlabel('Relative subset-PPL change vs native (%)\n95% paired article bootstrap; secondary intervals descriptive')
    ax.set_title('Fresh 1,000-article TEST: 64 inputs / 63 targets')
    ax.grid(axis='x',alpha=.2);ax.legend(fontsize=8,loc='best');fig.tight_layout()
    fig.savefig(out/'TEST_quality.png',dpi=180);fig.savefig(out/'TEST_quality.svg');plt.close(fig)
    columns=[('candidate_saving_fraction','Shared GU25'),
        ('same_budget_mixed_native_cache_saving_fraction','Native mixed cache'),
        ('same_budget_raw_readouts_plus_optimized_native_cache_saving_fraction','Raw readouts + cache')]
    fig,ax=plt.subplots(figsize=(8.0,3.6));x=np.arange(2);width=.24
    for k,(key,label) in enumerate(columns):
        values=[100*row(cost,b)[key] for b in ('rank256','dense')]
        bars=ax.bar(x+(k-1)*width,values,width,label=label)
        ax.bar_label(bars,fmt='%.3f',fontsize=8,padding=3)
    budgets=[row(cost,b)['method_host_bytes']/(1<<20) for b in ('rank256','dense')]
    ax.set_xticks(x,[f'{label}: {budget:.2f} MiB' for label,budget in zip(('rank256','dense'),budgets)])
    ax.set_ylabel('Avoided routed coefficient byte-work (%)')
    ax.set_title('Same coefficient-residency budget: all26 / nonBOS62\nFrozen VALID-v2 placements, fixed native TEST routing')
    ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.legend(fontsize=8)
    ax.margins(y=.2);fig.tight_layout();fig.savefig(out/'TEST_cost.png',dpi=180);fig.savefig(out/'TEST_cost.svg');plt.close(fig)


def run(args):
    quality=read(args.quality);cost=read(args.cost);audit=read(args.audit)
    assert quality['stage']=='test' and quality['execution']['status']=='complete'
    assert quality['native']['unique_prompts']==1000
    assert cost['status']=='complete' and cost['all_five_arms_finished_1000']
    assert all(a['vs_native']['prompts']==1000 and a['vs_native']['scored_targets']==63000 for a in quality['arms'].values())
    assert audit.get('status') in ('passed','PASS','complete','pass')
    assert audit['plan_sha256']==quality['plan_sha256']
    assert audit['reference_summary_comparison']['sha256']==hashlib.sha256(Path(args.quality).read_bytes()).hexdigest()
    assert audit['selection_receipt_sha256']==cost['selection_receipt_sha256']
    assert cost['plan']==quality['plan']
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    primary=quality['arms'][quality['primary_arm']];p=primary['vs_native'];exposure=p['exposure']['NLL_prefix63']
    linear=primary['vs_matched_linear_descriptive'];compression=primary['vs_dense_same_basis_descriptive']
    verdict={'pass':'通過','fail':'未通過','inconclusive':'仍未定'}[p['verdict']]
    small=row(cost,'rank256');large=row(cost,'dense');generated=generation_exposure(primary)
    raw_quality=quality['arms']['raw_x_select1408_eq_rank256_k12']['vs_native']
    strong_raw_better=(small['same_budget_raw_readouts_plus_optimized_native_cache_saving_fraction']>small['candidate_saving_fraction']
        and raw_quality['relative_subset_PPL_change']<p['relative_subset_PPL_change'])
    conclusion=('在本次固定配置及係數成本模型下，強raw-x控制同時有較小PPL退化與較多避免工作；'
                '本輪尚不支持優先部署共享GU25。低秩化確實改善容量，但沒有證明新增非線性bank的額外價值。'
                if strong_raw_better else
                '本輪需綜合品質與同預算係數工作量判讀；primary品質門檻本身不足以證明非線性共享的部署優勢。')
    lines=['# 共享非線性第二輪：完整1000題 TEST 結果','',
        f'固定單一 layer1 GU25、12個 nonowner experts、rank256 的品質 primary **{verdict}2% subset-PPL門檻**：'
        f'{pct(p["relative_subset_PPL_change"])}，paired article 95% CI {ci(p["ci95_relative_subset_PPL_change"])}。',
        f'相較 dense共享方案，係數常駐量由{large["method_host_bytes"]/(1<<20):.4f}縮至'
        f'{small["method_host_bytes"]/(1<<20):.4f} MiB，下降{100*(1-small["method_host_bytes"]/large["method_host_bytes"]):.2f}%。',
        '這個容量改善不等同非線性共享已勝過強線性控制；下方同預算比較與低 exposure 控制一併判讀。','',
        conclusion,'',
        '## 完整模型品質','',
        '五組皆完整1000篇、每篇一prompt；64 input tokens（含BOS），每組63,000個next-token targets，'
        '另做最多16 token greedy generation、fresh KV及live routing。只有壓縮共享對原生為primary；其他區間均描述性。','',
        '| 配置 | subset PPL | 相對原生變化／95% CI | 生成全序列一致率 |',
        '|---|---:|---|---:|',
        f'| 原生 canonical W8A8 FFN | {p["native_subset_PPL"]:.6f} | 參考 | 100% |']
    for name in LABELS:
        q=quality['arms'][name]['vs_native']
        lines.append(f'| {ZH[name]} | {q["candidate_subset_PPL"]:.6f} | {pct(q["relative_subset_PPL_change"])}／{ci(q["ci95_relative_subset_PPL_change"])} | {100*q["generation_exact_prompt_fraction"]:.2f}% |')
    lines+=['',f'壓縮共享相對同寬、同rank raw-x的paired PPL差異為{pct(linear["relative_subset_PPL_change"])}'
        f'（{ci(linear["ci95_relative_subset_PPL_change"])}）；相對同basis dense為{pct(compression["relative_subset_PPL_change"])}'
        f'（{ci(compression["ci95_relative_subset_PPL_change"])}）。兩者皆為描述性，不套primary的2%判定。','',
        '預先指定的position sensitivity也顯示BOS target的影響，以下為描述性平均ΔNLL：','',
        '| 配置 | BOS，1個target／prompt | NonBOS，62個targets／prompt |',
        '|---|---:|---:|']
    for name in LABELS:
        s=quality['arms'][name]['position_sensitivity']
        lines.append(f'| {ZH[name]} | {s["BOS"]["mean_delta_nll"]:+.6f} | {s["nonBOS"]["mean_delta_nll"]:+.6f} |')
    lines+=['',
        '壓縮共享相對dense的整體PPL增加主要集中在BOS所預測的第一個target；'
        'nonBOS平均ΔNLL反而略小。這提示低秩量化對此位置敏感，不是全部位置都等幅惡化。'
        'primary仍按凍結的全63個targets判定；本輪沒有看TEST後新增BOS fallback或精確cache來改寫結果。','',
        '生成一致率是對原生 greedy tokens的一致性，並非task accuracy。此處是短prefix subset PPL，'
        '不是標準串接WikiText PPL、長上下文或pretraining-unseen能力證明。','',
        '![TEST品質](TEST_quality.png)','',
        '## 同預算係數工作量','',
        '主成本scope在TEST前固定為全部26層、native-reference routing、nonBOS位置1..62，'
        '使用quality-VALID nonBOS v2所選的cache配置。下表是避免的routed coefficient byte-work比例，並非實測flash bytes。','',
        '| 係數常駐預算 | 共享方法 | 原生GU cache | 原生完整expert cache | 原生mixed cache | Raw readouts＋剩餘native cache |',
        '|---|---:|---:|---:|---:|---:|']
    for b in ('rank256','dense'):
        r=row(cost,b);native=r['native_no_cache_bytes']
        lines.append(f'| {r["method_host_bytes"]/(1<<20):.4f} MiB（{b}） | {100*r["candidate_saving_fraction"]:.4f}% | '
            f'{100*(1-r["same_budget_hot_GU_cache_bytes"]/native):.4f}% | '
            f'{100*(1-r["same_budget_full_expert_cache_bytes"]/native):.4f}% | '
            f'{100*r["same_budget_mixed_native_cache_saving_fraction"]:.4f}% | '
            f'{100*r["same_budget_raw_readouts_plus_optimized_native_cache_saving_fraction"]:.4f}% |')
    lines+=['',f'Rank256下，raw readouts不需GU25，省出的5,778,432 bytes可另放一份原生GU；'
        '配置由VALID先選，沒有用TEST頻率重排。原生cache不改數值，因此其品質對應原生參考。',
        'Dense預算下的dense-raw-plus-cache僅作成本控制；dense raw的品質只跑VALID，沒有新增其TEST arm。'
        'rank256 raw的品質則有完整1000題TEST，兩者不能混稱同一份品質證據。',
        '同預算係數工作比較不包含總host peak memory；method-specific phi、transform、private latent與quant workspace仍額外佔用。','',
        '![TEST成本](TEST_cost.png)','',
        '## 覆蓋率與生成階段','',
        f'壓縮共享在TF63實際替換{exposure["replaced_events"]:,}／{exposure["routed_expert_events"]:,}個expert/token事件'
        f'（{100*exposure["event_replacement_fraction"]:.4f}%）。'
        f'生成因果階段另替換{generated[1]:,}／{generated[0]:,}個事件，private latent A8 rows={generated[2]:,}。',
        'TRAIN常數控制的描述性95%區間也全部低於2%。在此低替換比例下，品質門檻本身不能單獨證明有效的expert重建，'
        '也不能把本配置的通過外推到更廣coverage。',
        '固定native routing的預先指定generation成本與actual nonlinear routing另列如下；最後未使用decode不混入因果收益。','',
        '| scope／routing | 共享避免byte-work | 同預算mixed cache | raw＋cache，該routing反事實 |',
        '|---|---:|---:|---:|']
    for scope,routing in (('TF63','fixed_native'),('BOS','fixed_native'),('generation_causal','fixed_native'),('generation_causal','actual_nonlinear')):
        r=row(cost,'rank256',scope,routing)
        lines.append(f'| {scope}／{routing} | {100*r["candidate_saving_fraction"]:.4f}% | '
                     f'{100*r["same_budget_mixed_native_cache_saving_fraction"]:.4f}% | '
                     f'{100*r["same_budget_raw_readouts_plus_optimized_native_cache_saving_fraction"]:.4f}% |')
    lines+=['','每個scope完整GU／D／MAC／新增A8 contexts、含BOS v1敏感度、actual routing與unused decode見 `final_cost_summary.json`。'
        'actual-nonlinear欄的raw比較是套用相同routing的係數反事實，並非raw自己的生成trace。'
        'byte-work按expert/token事件累加，未模擬prefill跨token矩陣重用、page或flash queue；host coefficient MAC也不包含全部quant／affine工作。','',
        '## 方法判讀與限制','',
        '原native shared-z／小residual分支在強raw控制與article-OOF後未達機制門檻。'
        '另一個native routed GU25對三個早層experts有非BOS局部正例；擴成12個後，pooled local NMSE仍是raw略好。'
        'GPU短訓僅改善0.526%，沒有部署learned-GU。有效的工程改善是private readout的TRAIN activation-aware低秩化，'
        '同時也適用raw控制，不能把這個壓縮能力全歸功於非線性共享。','',
        '功能性harness仍計算原生FFN作核對、保留原模型權重；本輪未實測flash、NVMeVirt、TPOT或端到端加速。'
        '原生fallback的模型容量未被刪除。B只計增量係數／scales／bias／metadata，公共backbone、KV及workspace均需另外容量。'
        '若host放得下完整canonical model約14.72 GiB weights加KV/workspace，完整native常駐本身就是合法零routed-flash係數讀取控制。','',
        '模型與資料、原六experts控制、全部VALID數字、batch-shape reference修正，以及精確W8A8契約見 [方法詳解](method_details.zh.md)。','',
        '## 驗證與產物','',
        '- 正式五組各1000篇、合計5000次fresh推論；80個shards全部成功，逐題IDs／targets／EOG／生成長度／routes／replacement與private-A8計數核對。',
        '- 343個frozen inputs在開始及結束重查，另有source snapshots與獨立audit；沒有依TEST結果改rank、eligibility、payload、cache配置或統計規則。',
        '- 72個factor模型逾103萬output floats整kernel逐bit一致；新runtime的native arm在VALID200／12,600位置與原生64-token參考一致。','',
        '小型結果：`final_quality_summary.json`、`final_cost_summary.json`、`final_independent_audit.json`、'
        '`final-selection.frozen.json`及圖表保存在本目錄。大型模型、captures、targets、payloads、logs位於NFS研究根目錄。','',
        '原始推論命令（拒絕覆寫既有輸出）：','',
        '```bash',
        'python scripts/research_20261109_quality_run.py \\',
        '  --plan /home/nfs_client/ericyu/expert_flash_experiment/research-2026-11-09-shared-features/final1000-plan.json',
        '```','']
    (out/'report.zh.md').write_text('\n'.join(lines))
    for source,name in ((args.quality,'final_quality_summary.json'),(args.cost,'final_cost_summary.json'),(args.audit,'final_independent_audit.json')):
        assert Path(source).resolve()!=out/name
        shutil.copyfile(source,out/name)
    figures(quality,cost,out)
    state=dict(status='complete',unique_TEST_articles=1000,evaluations_completed=5000,
        prompts_completed_by_arm={name:1000 for name in ('native',*quality['arms'])},
        primary_arm=quality['primary_arm'],quality_primary_verdict=p['verdict'],
        primary_relative_PPL_change=p['relative_subset_PPL_change'],
        primary_CI95_relative_PPL_change=p['ci95_relative_subset_PPL_change'],
        compressed_coefficient_bank_MiB=small['method_host_bytes']/(1<<20),
        coefficient_bank_reduction_fraction=1-small['method_host_bytes']/large['method_host_bytes'],
        TEST_results_used_for_selection=False,measured_flash_IO=False,measured_end_to_end_speedup=False,
        independent_audit_status=audit['status'],report=str((out/'report.zh.md').resolve()),
        provenance=dict(quality_sha256=hashlib.sha256(Path(args.quality).read_bytes()).hexdigest(),
                        cost_sha256=hashlib.sha256(Path(args.cost).read_bytes()).hexdigest(),
                        independent_audit_sha256=hashlib.sha256(Path(args.audit).read_bytes()).hexdigest(),
                        report_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    (out/'state.json').write_text(json.dumps(state,indent=2,allow_nan=False)+'\n')
    print(out/'report.zh.md')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--quality',required=True);p.add_argument('--cost',required=True);p.add_argument('--audit',required=True);p.add_argument('--output',required=True)
    run(p.parse_args())
