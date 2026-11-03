#!/usr/bin/env python3
"""Summarize sealed standard-copy replay and equivalent whole-job packing."""
from pathlib import Path
from collections import defaultdict
import csv,gzip,hashlib,json
import numpy as np

OUT=Path(__file__).resolve().parent
BASE=OUT.parent
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def packed_witness():
    with gzip.open(OUT/'physical_ledger.frozen.json.gz','rt') as z:records=json.load(z)
    result=[]
    for record in records:
        chips=[]
        for chip in record['chips']:
            # Reserve one common WL band across all four planes up to the
            # highest real GU WL. Whole private matrices never cross a block.
            high=max(p['wl'] for p in chip['GU'])+1
            fill=[high,0];jobs=[];occupied={(p['block'],p['plane'],p['wl']) for p in chip['GU']}
            for job in chip['Down_jobs']:
                n=job['reserved_WL'];b=next(b for b in [0,1] if fill[b]+n<=192)
                shifted=dict(job,block=b,wl_start=fill[b]);jobs.append(shifted);fill[b]+=n
                for j in range(job['pages']):
                    key=(b,j%4,shifted['wl_start']+j//4)
                    assert key not in occupied;occupied.add(key)
            assert len(occupied)*16384==chip['programmed_bytes']
            chips.append(dict(chip=chip['chip'],GU_common_band_reserved=high,band_fill=fill,
                              nonempty_blocks=len({p[0] for p in occupied}),jobs=jobs))
        result.append(dict(GU=record['GU'],plan=record['plan'],layer=record['layer'],
            allocated_budget_blocks=32,executed_ledger_nonempty_blocks=record['actual_nonempty_blocks'],
            equivalent_packed_witness_nonempty_blocks=sum(c['nonempty_blocks'] for c in chips),chips=chips))
    dump(OUT/'equivalent_whole_job_packing_certificate.json',dict(status='PASS',records=result,
        scope='Additional whole-job relocation witness, not the executed physical ledger and not a minimum-block theorem. Every private GEMV remains within one block and starts cold with fullinput; the model has no cold-WL-distance term. Therefore block/WL renaming preserves each private recipe duration and whole timing DAG. Shared/private coefficients and query reads unchanged.'))


def main():
    report=read(OUT/'TRAIN512/report.json')
    assert all(r['completed']==1536 and r['valid_FIFO_capacities']['128']==1536 for r in report['records'])
    records={r['name']:r for r in report['records']};paired=[];cross=[]
    with np.load(OUT/'TRAIN512/service_arrays.npz') as z:
        for gu in ['candidate','RR','g8_fixed']:
            for cp in [1,4]:
                for priority in ['GF','DF']:
                    prefix=f'{gu}.triple_private_shared4.input{cp}.{priority}'
                    value=z[prefix+'.cost']
                    for plan in ['original_g4_shared16','onecopy_shared4']:
                        anchor=f'{gu}.{plan}.input{cp}.{priority}';base=z[anchor+'.cost'];delta=value-base
                        newrec,oldrec=records[prefix],records[anchor]
                        work_changes={k:newrec['counts_and_work_per_valid_query'].get(k,0)-oldrec['counts_and_work_per_valid_query'].get(k,0)
                                      for k in set(newrec['counts_and_work_per_valid_query'])|set(oldrec['counts_and_work_per_valid_query'])}
                        work_changes={k:v for k,v in sorted(work_changes.items()) if v}
                        if plan=='onecopy_shared4':assert not work_changes,(prefix,work_changes)
                        paired.append(dict(GU=gu,input_copies=cp,priority=priority,anchor=plan,
                            mean_gain=float(1-value.mean()/base.mean()),
                            event_p99_gain=float(1-np.percentile(value,99)/np.percentile(base,99)),
                            three_layer_sum_p99_gain=float(1-np.percentile(value.sum(axis=0),99)/np.percentile(base.sum(axis=0),99)),
                            faster=int((delta<0).sum()),equal=int((delta==0).sum()),slower=int((delta>0).sum()),
                            per_layer_mean_gain=(1-value.mean(axis=1)/base.mean(axis=1)).tolist(),
                            changed_mean_event_counts_work=work_changes))
        for cp in [1,4]:
            for pr in ['GF','DF']:
                candidate=z[f'candidate.triple_private_shared4.input{cp}.{pr}.cost']
                for other in ['RR','g8_fixed']:
                    baseline=z[f'{other}.triple_private_shared4.input{cp}.{pr}.cost']
                    cross.append(dict(candidate='candidate',baseline=other,input_copies=cp,priority=pr,
                        mean_gain=float(1-candidate.mean()/baseline.mean()),
                        event_p99_gain=float(1-np.percentile(candidate,99)/np.percentile(baseline,99)),
                        slower=int((candidate>baseline).sum()),equal=int((candidate==baseline).sum()),faster=int((candidate<baseline).sum())))
        priority=[]
        for gu in ['candidate','RR','g8_fixed']:
            for plan in ['original_g4_shared16','onecopy_shared4','triple_private_shared4']:
                for cp in [1,4]:
                    a=f'{gu}.{plan}.input{cp}.GF';b=f'{gu}.{plan}.input{cp}.DF'
                    exact=all(np.array_equal(z[a+'.'+suffix],z[b+'.'+suffix]) for suffix in ['cost','fifo'])
                    priority.append(dict(GU=gu,plan=plan,input_copies=cp,GF_DF_cost_FIFO_exact=exact))
        # Keep the previously TRAIN-selected g8GU/g2 incumbent visible.
        with np.load(BASE/'full-ffn/asymmetric-down/TRAIN512/service_arrays.npz') as old:
            g8=[]
            for cp in [1,4]:
                for pr in ['GF','DF']:
                    a=z[f'g8_fixed.triple_private_shared4.input{cp}.{pr}.cost']
                    b=old[f'g8_fixed.g2_reused_TRAIN_labels.input{cp}.{pr}.cost']
                    g8.append(dict(input_copies=cp,priority=pr,mean_gain=float(1-a.mean()/b.mean()),
                                   slower=int((a>b).sum()),equal=int((a==b).sum()),faster=int((a<b).sum())))
    packed_witness()
    candidate_cert=[dict(name=r['name'],**r['route_independence']) for r in report['records']
                    if r['GU']=='candidate' and r['plan']=='triple_private_shared4']
    result=dict(status='complete_TRAIN_reuse',source_sha256=sha(__file__),execution_source_sha256=report['source_sha256'],
        full_report_sha256=sha(OUT/'TRAIN512/report.json'),full_arrays_sha256=sha(OUT/'TRAIN512/service_arrays.npz'),
        stored_onecopy_bytes=155713536,stored_triple_bytes=256376832,extra_stored_bytes=100663296,
        extra_stored_MiB=96,query_mathematical_coefficients_and_useful_reads_unchanged=True,
        paired=paired,cross_GU_same_triple_Down=cross,priority_identity=priority,g8_previous_TRAIN_selected_g2_reference=g8,
        candidate_formal_route_independence=candidate_cert,
        interpretation='Standard complete privateDown copies and resource partition. Relative to onecopyshared4, only chip placement/load balance changes; all command/input/output/page/ECC counts and total resource work are exactlyequal. Strong RR receives the same copies. No newlayoutalgorithm/CBC/IRexpressiveness novelty or measuredhardware performance.',
        model_limits='Conditional fixed9.7us/copy500ns PB/source/drop and128BFIFO event model. Tiny producer-period changes can alter the shaping comparison. Entire shared/private operand vectors and chipownership paid. No heldout, inference, fullweight numericalchange, or hardwareconstant claim.')
    dump(OUT/'paired_report.json',result)
    with (OUT/'small_table.csv').open('w') as file:
        w=csv.writer(file);w.writerow(['GU','plan','input_copies','priority','mean_ns','p99_event_ns','p99_3layer_sum_ns','FIFO_peak_B','valid_queries_128B'])
        for r in report['records']:w.writerow([r['GU'],r['plan'],r['input_copies'],r['priority'],r['cost_ns']['mean'],r['cost_ns']['p99'],r['three_layer_sum_ns']['p99'],r['FIFO_peak_bytes']['maximum'],r['valid_FIFO_capacities']['128']])
    rows=[]
    for r in report['records']:
        if r['input_copies']==1 and r['priority']=='GF':
            rows.append(f"| {r['GU']} | {r['plan']} | {r['cost_ns']['mean']/1000:.6f} | {r['cost_ns']['p99']/1000:.6f} | {int(r['FIFO_peak_bytes']['maximum'])} |")
    gains=[]
    for x in paired:
        if x['input_copies']==1 and x['priority']=='GF' and x['anchor']=='onecopy_shared4':
            gains.append(f"{x['GU']} {100*x['mean_gain']:.3f}%（slower {x['slower']}/1536）")
    text='''# Standard private-Down replicas：完整 FFN 模型

三組完整 Down 副本提供明確的模型收益；這是標準 replication 與資源分組，代價為每層多存 **96 MiB**，沒有新的 layout 演算法或量測硬體速度主張。既有 W8/A8 TRAIN 前三層各 512 個輸入，共 1536 個 layer/query；同一批已見 TRAIN 用於 64-input 篩選和完整描述性評估，沒有 heldout。

| GU family | Down plan | Mean μs | Event P99 μs | Peak FIFO B |
|---|---|---:|---:|---:|
'''+ '\n'.join(rows)+'''

相對各 family 已重配置的 one-copy/shared-g4 強控制，input1/GF 的平均模型改善：'''+ '；'.join(gains)+'''。

同樣取得三副本能力後，candidate 相對標準 RR 的平均收益約 0.484%，相對固定 source g8 GU 約 4.398%。主要新收益是標準 Down replication 的負載平衡，不能包裝成 shared-layout 突破。

所有 36 個配置均完成 1536 個 queries，且全部在有限 PB 與 128B output FIFO 模型下有效。Input copies=1/4、GF/DF 都有逐 query 結果；完整 paired P99、slower/equal/faster、各層、counts/work 歸因及既有 TRAIN-selected g8GU/g2 reference 見 [paired_report.json](paired_report.json)。先前 64-input screen 選中的 g2 不保證在完整 512 個已見 TRAIN 輸入仍優於 g4；本輪兩個 references 都保留。

配置先凍結於 [protocol.frozen.json](protocol.frozen.json)。Shared Down 固定在 chips12..15，每 chip 384rows/C1024/24pages/6waves。其餘三組 g4 各儲存 64 個 routed experts 的完整 Down copy；已知六個 expert IDs 排序後，rank0/1、2/3、4/5 固定派至各組，不搜索 query completion cost。每 private chip job 384rows/C512/12pages/3waves，完整 upload 自己的 A8 vector、fresh cold，沒有 private context sharing。Candidate、標準 RR 與固定 source g8 GU 取得相同選單和能力。Generic IR 可以表達這些操作；表達能力本身不是新貢獻，也不等同文獻已預先提出全部結果。

原 one-copy 係數共 155713536B/layer，triples 為 256376832B/layer；每個 query 的數學 coefficients、useful reads 與 output bytes 不變。相對 onecopy/shared-g4，三副本的 commands、input/output bytes、physical/ECC pages 及各 resource busy work 都逐項相同，改善來自不同 chips 上的負載分配。相對原 shared-g16 anchor 還混合 shared Down 分組改善，不能全歸因 replication，更不能歸因 shared layout。

共同預算是每 chip 兩個 192-WL/four-plane blocks，共 32 blocks/layer。實際執行 ledger 中，GU 在 block0，完整 private Down bands 在 block1；三副本 private chips 的 64×3WL 正好填滿 block1，shared Down 在 block0 WL128..133，合計 28 個非空 blocks。單副本 anchors 為便於核對也全部把 Down 放 block1，因此本次 ledger 為 32 個非空 blocks；這不是其最省 block 的配置。另附 [whole-job packing certificate](equivalent_whole_job_packing_certificate.json)，可在不跨 block、不加 cold/command/input 的前提下重排完整冷啟動 jobs，保留相同模型時序。Allocated capacity budget、actual occupied bytes、nonempty blocks 三者分開列出。

Candidate/triples 所有 1536 個 queries 的 normalized timing DAG 與完成成本相同；每個 input/service 設定各有獨立證書。此證書保留 event duration、resource、dependencies 和 expert ranks，排除物理地址標籤。它只證明這個 formal service contract 的 route independence，沒有證明真實硬體延遲恆定或新的 CBC 性質。

模型沿用 host submit2μs、command250ns、PCIe8GB/s、八條2GB/s channels、每 chip 到 compute 加最後 ONFI output 才釋放 ownership。完整 GU outputs 到 host、activation/A8、private full-vector upload 均有因果 barrier。FIFO 是每 chip 產生 bytes 減去連續 channel drain 的 backlog，沒有隱含 backpressure 或新增大 FIFO。

9.7μs CR cadence、copy500ns、source permutation/drop-before-ECC 及 GU shaping/cold-reset 契約仍是未量測的條件模型；private Down 固定0123且嚴格執行，沒有額外 rescue。Cadence sensitivity 已知可改變 GU shaping 結果；本實驗不把此條件模型叫 native measurement。沒有讀取或更改完整 FFN weights、沒有品質/inference/heldout 評估。

可重用程式是 [replica_full_ffn.py](replica_full_ffn.py) 的 `graph_for`、`Recipes.private_at`、`allocate_down`；執行前來源/輸入、layout ledger 和 gate receipt 均已封存。代表性完整 traces 在 `TRAIN512/trace.*.json`，獨立地址、係數身分、既有 anchor 及 route-independent DAG audit 在 [independent_audit.json](independent_audit.json)。
'''
    (OUT/'README.md').write_text(text)
    print(json.dumps(dict(status='summarized',gains_input1_GF=gains)),flush=True)

if __name__=='__main__':main()
