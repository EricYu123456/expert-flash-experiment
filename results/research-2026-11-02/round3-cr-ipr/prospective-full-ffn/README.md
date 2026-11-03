# 凍結後的新自編診斷路由

16個新自編 prompts，四個 tasks各四題；全部使用既有、已驗證來源身分的 canonical full-FFN W8/A8 CPU helper，16decode steps。沒有打開舊heldout或重新挑prompt，沒有任務品質測量，且新路由未配對placed operator。

配置與比較規則在模型執行前已寫入捕捉協定；重播程式是在archive產出後、尚未查看路由或matching/performance結果前完成。這個時間順序在protocol中明列，未宣稱code早於捕捉凍結。

主效能樣本為前3layers的768events／256對齊inputs／16prompt clusters。下表為input1、GF。

| GU / Down | mean µs | event P99 µs | fallback events | 相對同 GU g4 triples mean | 配對較慢 |
|---|---:|---:|---:|---:|---:|
| candidate / g4triples | 280.526000 | 280.526 | 0/768 | reference | — |
| candidate / g2_fitted | 255.048279 | 254.712 | 3/768 | 9.082% | 3 |
| candidate / g2_IDslot | 255.944154 | 340.466 | 11/768 | 8.763% | 11 |
| candidate / g2_seed | 254.824310 | 254.712 | 1/768 | 9.162% | 1 |
| RR / g4triples | 281.506677 | 296.966 | 0/768 | reference | — |
| RR / g2_fitted | 256.742159 | 275.908 | 3/768 | 8.797% | 3 |
| RR / g2_IDslot | 257.630576 | 338.006 | 11/768 | 8.482% | 11 |
| RR / g2_seed | 256.520055 | 273.201 | 1/768 | 8.876% | 1 |

G4 triples payload244.5MiB；三種g2 maps均289.5MiB。所有配置使用相同input／FIFO／排程政策，每個事件FIFO≤128B。

TRAIN-fitted candidate配置平均改善9.082%，prompt-cluster bootstrap 95%描述區間8.922–9.202%；event P99改善9.202%，但三層對齊服務和P99只改善4.617%。3個fallback的最大值340.966µs，高於基準280.526µs。不能以pooled event P99掩蓋整體服務和或最壞值。

固定IDslot的11次fallback使event P99退步21.367%，三層P99退步1.047%。固定seed在這批資料只失敗一次，表現甚至略優於TRAIN-fitted配置；不支持複雜學習的必要性。此處沒有把結果最佳者重新選為新主方法。

固定IDslot在全部40layers共10,240events中有83次cap1失敗。這只是matching統計，不是全40layers FFN效能。

[資料與凍結捕捉](../prospective-routes/protocol.frozen.json)、[重播凍結協定](protocol.frozen.json)、[全部16個profiles與input1/4](report.json)、[完整cost/FIFO/capacity陣列](service_arrays.npz)。

看過這些失敗之後才提出的 shared可移動／八-pair建構，若重播此批資料，一律標成diagnostic reuse，不能再稱新未見驗證。
