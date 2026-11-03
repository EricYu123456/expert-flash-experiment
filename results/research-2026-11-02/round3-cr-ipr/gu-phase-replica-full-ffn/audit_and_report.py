#!/usr/bin/env python3
"""Independent address/source-identity audit and fixed-source tradeoff report."""
from pathlib import Path
from collections import Counter,defaultdict
import csv,gzip,hashlib,json
import numpy as np

OUT=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    spec=read(OUT/'protocol.frozen.json')
    for p,h in spec['inputs_sha256'].items():assert sha(p)==h,p
    with gzip.open(OUT/'physical_ledger.frozen.json.gz','rt') as z:ledgers=json.load(z)
    totals=Counter();capacity=[]
    for record in ledgers:
        kind=spec['cases'][record['case']][0];gu_bands=defaultdict(list);down_bands=defaultdict(list)
        total=0;blocks=0
        for chip in record['chips']:
            addresses=set();by_owner=defaultdict(list)
            for p in chip['GU']:
                key=(p['block'],p['plane'],p['wl']);assert key not in addresses;addresses.add(key)
                assert p['source_row_bytes']==1536 and p['source_byte_offset']==p['page_index']*16384
                e=p['expert'];copy=0 if e==64 else p['copy_phase'] if kind=='phase' else p['copy_group']
                by_owner[e,copy].append(p)
            for (e,copy),pages in by_owner.items():
                count=12 if e==64 or kind=='group' else 6
                assert len(pages)==count and sorted(p['page_index'] for p in pages)==list(range(count))
                rows=128 if e==64 or kind=='group' else 64
                start=chip['shared_row_start'] if e==64 else chip['routed_row_start']
                assert count*16384==rows*1536
                gu_bands[e,copy].append((start,start+rows))
            for job in chip['Down_jobs']:
                down_bands[job['expert'],job['copy']].append((job['row_start'],job['row_start']+job['rows']))
                assert job['columns']==(1024 if job['expert']==64 else 512)
                assert job['rows']*job['columns']==job['pages']*16384
                for j in range(job['pages']):
                    key=(job['block'],j%4,job['wl_start']+j//4)
                    assert key not in addresses;addresses.add(key)
            assert all(0<=b<2 and 0<=p<4 and 0<=w<192 for b,p,w in addresses)
            assert len(addresses)*16384==chip['programmed_bytes']
            blocks+=len({b for b,p,w in addresses});total+=len(addresses)*16384
            totals['physical_pages']+=len(addresses)
        assert blocks==record['nonempty_blocks']==28 and total==record['programmed_bytes']
        for e in range(65):
            copies=[k for k in gu_bands if k[0]==e]
            assert len(copies)==(1 if e<6 or e==64 else 2)
            for key in copies:
                intervals=sorted(gu_bands[key]);assert intervals[0][0]==0 and intervals[-1][1]==(2048 if e==64 else 1024)
                assert all(a[1]==b[0] for a,b in zip(intervals,intervals[1:]))
            dcopies=[k for k in down_bands if k[0]==e]
            assert len(dcopies)==(3 if record['extra_Down_bytes'] and e<64 else 1)
            for key in dcopies:
                intervals=sorted(down_bands[key]);assert intervals[0][0]==0 and intervals[-1][1]==1536
                assert all(a[1]==b[0] for a,b in zip(intervals,intervals[1:]))
        totals['layer_plans']+=1
        capacity.append(dict(case=record['case'],layer=record['layer'],stored_MiB=total/1048576,
                             extra_GU_MiB=87,extra_Down_MiB=record['extra_Down_bytes']/1048576,
                             nonempty_blocks=blocks,allocated_budget_blocks=32))
    mode='TRAIN512' if (OUT/'TRAIN512/report.json').exists() else 'screen'
    report=read(OUT/mode/'report.json');events=1536 if mode=='TRAIN512' else 192
    records={r['name']:r for r in report['records']};comparisons=[]
    with np.load(OUT/mode/'service_arrays.npz') as z:
        for cp in ([1,4] if mode=='TRAIN512' else [1]):
            for pr in (['GF','DF'] if mode=='TRAIN512' else ['GF']):
                phase=z[f'phase_triples.input{cp}.{pr}.cost'];group=z[f'group_triples.input{cp}.{pr}.cost']
                assert np.array_equal(phase,group)
                old=z[f'reference.candidate.triple_private_shared4.input{cp}.{pr}.cost']
                sameold=np.array_equal(phase,old)
                for name,base in [('equal_byte_group_triples',group),('old_candidateperm_triples',old),
                    ('old_RRperm_triples',z[f'reference.RR.triple_private_shared4.input{cp}.{pr}.cost']),
                    ('old_g8fixed_triples',z[f'reference.g8_fixed.triple_private_shared4.input{cp}.{pr}.cost'])]:
                    comparisons.append(dict(input_copies=cp,priority=pr,baseline=name,
                        cost_arrays_exact=bool(np.array_equal(phase,base)),mean_gain=float(1-phase.mean()/base.mean()),
                        event_p99_gain=float(1-np.percentile(phase,99)/np.percentile(base,99)),
                        slower=int((phase>base).sum()),equal=int((phase==base).sum()),faster=int((phase<base).sum())))
                assert records[f'phase_triples.input{cp}.{pr}']['counts_and_work_per_valid_query']==records[f'group_triples.input{cp}.{pr}']['counts_and_work_per_valid_query']
    for r in report['records']:
        assert r['completed']==events and r['valid_FIFO_capacities']['128']==events
        assert r['fixed_source_0123'] and not r['requires_programmable_PBsource']
    # Independent saved trace semantics: output shapes, copy choices, fullinput,
    # and the conventional canonical command-issue order are explicit.
    for path in sorted((OUT/mode).glob('trace.*.json')):
        trace=read(path);gu=[j for j in trace['jobs'] if j['phase']==0]
        assert len(gu)==16
        produced=Counter()
        for n in trace['nodes']:
            if n['kind']=='row_produced' and n['phase']==0:produced[n['expert_rank']]+=n['bytes']
        assert produced==Counter({**{i:4096 for i in range(6)},6:8192})
        for job in gu:
            assert job['columns']==1536 and job['pipeline']['source_policy']=='fixed'
            assert trace['nodes'][job['command']]['expert_rank']==0
            assert trace['nodes'][job['input']]['bytes']==1536*trace['service']['input_copies']
            meta=job['GU_copy_metadata'];assert meta['fixed_source_0123']
            if meta['kind']=='phase':assert meta['routed_rows_per_chip']==64
            else:assert meta['routed_rows_per_chip']==128
        totals['representative_traces']+=1
    result=dict(status='PASS',audit_source_sha256=sha(__file__),execution_source_sha256=report['source_sha256'],
        mode=mode,checks=dict(totals),capacity=capacity,paired=comparisons,
        contribution_scope='Phase placement is not uniquelyneeded: equalbyte standard wholeGU group copies reproduce everymeasured FFN cost and eventcount. Both replace programmablePBsource by extra87MiBGUstorage underconditionalfixedsource/IPR/drop/PB contract. No placement/CBCalgorithmnovelty.',
        limits='PreviouslyobservedTRAIN, noheldout/fullweights/nativehardwaremeasurement. SameWL groupwhole-dummyoptionalkernel is not splitIPRprimary.')
    dump(OUT/'audit_and_paired_report.json',result)
    with (OUT/'small_table.csv').open('w') as file:
        w=csv.writer(file);w.writerow(['case','input_copies','priority','mean_ns','p99_ns','completed','FIFO128_valid'])
        for r in report['records']:w.writerow([r['case'],r['input_copies'],r['priority'],r['cost_ns']['mean'],r['cost_ns']['p99'],r['completed'],r['valid_FIFO_capacities']['128']])
    lines=['# Fixed0123 GU replicas：能力與儲存取捨','','Phase copies 和同儲存量的標準 g8 whole-GU group replicas，在本輪所有 queries 的 FFN 模型結果逐項相同。它們移除 programmable PB source order 的假設，代價是每層額外 **87 MiB GU**；沒有 phase placement 的獨有收益或新 CBC 演算法主張。',
           '',f'目前範圍：{mode}，每 case {events} 個 layer/query，均為既有 W8/A8 TRAIN 前三層。沒有 heldout、完整權重 inference 或硬體量測。',
           '','| GU / Down 配置 | Mean μs | Event P99 μs | 儲存 MiB/layer | Source |',
           '|---|---:|---:|---:|---|']
    for r in report['records']:
        if r['input_copies']==1 and r['priority']=='GF':lines.append(f"| {r['case']} | {r['cost_ns']['mean']/1000:.6f} | {r['cost_ns']['p99']/1000:.6f} | {235.5 if r['case']=='phase_onecopy' else 331.5} | fixed0123 |")
    lines+=['','Phase+triples、group+triples 與舊 candidate-permutation+triples 的成本相同；舊方案只需 244.5 MiB，但依賴 source permutation。標準 group replication 因此完整解釋此 fixed-source 取捨，不能把 phase copies 包裝成唯一可達成 route-independent tail 的新配置。',
        '',f'所有新配置的 {events} 個 queries 均通過有限 PB 和 128B output FIFO。Phase/group triples 的 event counts、input/output bytes、physical/ECC pages 及總 resource work 也完全相同。完整 paired slower/equal/faster 見 [audit_and_paired_report.json](audit_and_paired_report.json)。',
        '','[protocol.frozen.json](protocol.frozen.json) 在 execution 前綁定 builder、source、TRAIN、舊 controls arrays 與實體 ledger。每顆 chip 的 GU 使用四 planes 各 186 WL；shared Down 在 chips12..15 的 WL186..191，private Down 完整 jobs 在 block1。全部新 cases 為 28 個非空 blocks，共同預算為 32；phase/group+triples 存 331.5 MiB，phase+onecopy 存 235.5 MiB。',
        '','Phase control 將六個 experts 配成三個 phase0、三個 phase2，交替六頁 expert stream；標準 group control 將六個 experts 配成兩個八-chip groups 各三個，每 expert 每 chip 十二頁。三個 group0-only、三個 group1-only、其餘 58 個 dual copies，為既有 capacity3 copy-placement 原理的直接實例。兩者都用固定 lex assignment，沒有 TRAIN fitting、query completion-time search 或 relocation。',
        '','主 profile 是既有 sealed pipeline 的固定0123 splitwave10/prefix1，13 waves、52 physical pages、48 ECC pages、四頁真未選 expert dummy，strict copy500ns；此 split 需要原已授予的 IPR/drop-before-ECC 能力，不標成 same-WL。Group 的未 split 12-wave 與全 dummy-wave kernel 可逐波 same-WL；whole-dummy 使用另一獨立 recurrence，沒有混入本輪 FFN。copy100 的 12-wave/no-drop 只作既有 kernel sensitivity。',
        '','原 expert/coefficient 身分與 rows 都保留：phase 每 routed expert 為16chips×64rows，group 為8chips×128rows，總共1024 GU rows；shared 為16chips×128rows=2048。C1536 的跨頁 row 依原 page index 連續讀完；expert邊界恰為完整 dot。固定0123輸出按 metadata scatter，沒有新增 PE selector、private input context 或部分 column。',
        '','FFN 仍繳完整 GU→host/A8 barrier、各 private C512/shared C1024 vector、hostsubmit、commands、PCIe、channel、chipownership 與 row-complete output FIFO。為保留強控制，所有 common-GU jobs 以固定 chip 順序、rank0 發 command/input，輸出 owner priority 保持原 expert rank；沒有根據 query latency 調順序。',
        '','9.7μs CR、copy500ns、IPR/drop、PB 和 FIFO 都是條件模型，並非量測 native performance；cadence sensitivity 限制仍適用。Equal-byte generic group baseline 已吃掉 phase layout novelty，本輪目的只剩辨認 execution contract、能力與容量取捨。',
        '','可重用 [phase_full_ffn.py](phase_full_ffn.py) 的 `graph_for`、`get_inputs`、`make_ledger`；builder 的 fixed stream/source identities 由前置凍結檔提供。所有 source/input receipts、copy/row metadata 與代表 traces 保留，舊封存檔未改寫。']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status='PASS',mode=mode,checks=dict(totals))),flush=True)

if __name__=='__main__':main()
