# Fixed0123 GU replicas：能力與儲存取捨

Phase copies 和同儲存量的標準 g8 whole-GU group replicas，在本輪所有 queries 的 FFN 模型結果逐項相同。它們移除 programmable PB source order 的假設，代價是每層額外 **87 MiB GU**；沒有 phase placement 的獨有收益或新 CBC 演算法主張。

目前範圍：screen，每 case 192 個 layer/query，均為既有 W8/A8 TRAIN 前三層。沒有 heldout、完整權重 inference 或硬體量測。

| GU / Down 配置 | Mean μs | Event P99 μs | 儲存 MiB/layer | Source |
|---|---:|---:|---:|---|
| phase_triples | 280.526000 | 280.526000 | 331.5 | fixed0123 |
| phase_onecopy | 320.394229 | 400.106000 | 235.5 | fixed0123 |
| group_triples | 280.526000 | 280.526000 | 331.5 | fixed0123 |

Phase+triples、group+triples 與舊 candidate-permutation+triples 的成本相同；舊方案只需 244.5 MiB，但依賴 source permutation。標準 group replication 因此完整解釋此 fixed-source 取捨，不能把 phase copies 包裝成唯一可達成 route-independent tail 的新配置。

所有新配置的 192 個 queries 均通過有限 PB 和 128B output FIFO。Phase/group triples 的 event counts、input/output bytes、physical/ECC pages 及總 resource work 也完全相同。完整 paired slower/equal/faster 見 [audit_and_paired_report.json](audit_and_paired_report.json)。

[protocol.frozen.json](protocol.frozen.json) 在 execution 前綁定 builder、source、TRAIN、舊 controls arrays 與實體 ledger。每顆 chip 的 GU 使用四 planes 各 186 WL；shared Down 在 chips12..15 的 WL186..191，private Down 完整 jobs 在 block1。全部新 cases 為 28 個非空 blocks，共同預算為 32；phase/group+triples 存 331.5 MiB，phase+onecopy 存 235.5 MiB。

Phase control 將六個 experts 配成三個 phase0、三個 phase2，交替六頁 expert stream；標準 group control 將六個 experts 配成兩個八-chip groups 各三個，每 expert 每 chip 十二頁。三個 group0-only、三個 group1-only、其餘 58 個 dual copies，為既有 capacity3 copy-placement 原理的直接實例。兩者都用固定 lex assignment，沒有 TRAIN fitting、query completion-time search 或 relocation。

主 profile 是既有 sealed pipeline 的固定0123 splitwave10/prefix1，13 waves、52 physical pages、48 ECC pages、四頁真未選 expert dummy，strict copy500ns；此 split 需要原已授予的 IPR/drop-before-ECC 能力，不標成 same-WL。Group 的未 split 12-wave 與全 dummy-wave kernel 可逐波 same-WL；whole-dummy 使用另一獨立 recurrence，沒有混入本輪 FFN。copy100 的 12-wave/no-drop 只作既有 kernel sensitivity。

原 expert/coefficient 身分與 rows 都保留：phase 每 routed expert 為16chips×64rows，group 為8chips×128rows，總共1024 GU rows；shared 為16chips×128rows=2048。C1536 的跨頁 row 依原 page index 連續讀完；expert邊界恰為完整 dot。固定0123輸出按 metadata scatter，沒有新增 PE selector、private input context 或部分 column。

FFN 仍繳完整 GU→host/A8 barrier、各 private C512/shared C1024 vector、hostsubmit、commands、PCIe、channel、chipownership 與 row-complete output FIFO。為保留強控制，所有 common-GU jobs 以固定 chip 順序、rank0 發 command/input，輸出 owner priority 保持原 expert rank；沒有根據 query latency 調順序。

9.7μs CR、copy500ns、IPR/drop、PB 和 FIFO 都是條件模型，並非量測 native performance；cadence sensitivity 限制仍適用。Equal-byte generic group baseline 已吃掉 phase layout novelty，本輪目的只剩辨認 execution contract、能力與容量取捨。

可重用 [phase_full_ffn.py](phase_full_ffn.py) 的 `graph_for`、`get_inputs`、`make_ledger`；builder 的 fixed stream/source identities 由前置凍結檔提供。所有 source/input receipts、copy/row metadata 與代表 traces 保留，舊封存檔未改寫。
