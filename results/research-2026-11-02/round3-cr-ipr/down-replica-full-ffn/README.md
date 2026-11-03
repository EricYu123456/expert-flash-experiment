# Standard private-Down replicas：完整 FFN 模型

三組完整 Down 副本提供明確的模型收益；這是標準 replication 與資源分組，代價為每層多存 **96 MiB**，沒有新的 layout 演算法或量測硬體速度主張。既有 W8/A8 TRAIN 前三層各 512 個輸入，共 1536 個 layer/query；同一批已見 TRAIN 用於 64-input 篩選和完整描述性評估，沒有 heldout。

| GU family | Down plan | Mean μs | Event P99 μs | Peak FIFO B |
|---|---|---:|---:|---:|
| candidate | original_g4_shared16 | 338.596219 | 385.962000 | 128 |
| candidate | onecopy_shared4 | 318.719906 | 400.106000 | 128 |
| candidate | triple_private_shared4 | 280.526000 | 280.526000 | 128 |
| RR | original_g4_shared16 | 339.961401 | 395.392000 | 128 |
| RR | onecopy_shared4 | 320.085089 | 401.186000 | 128 |
| RR | triple_private_shared4 | 281.891182 | 296.966000 | 128 |
| g8_fixed | original_g4_shared16 | 347.228263 | 444.174500 | 128 |
| g8_fixed | onecopy_shared4 | 324.953651 | 428.826000 | 128 |
| g8_fixed | triple_private_shared4 | 293.431521 | 340.966000 | 128 |

相對各 family 已重配置的 one-copy/shared-g4 強控制，input1/GF 的平均模型改善：candidate 11.984%（slower 0/1536）；RR 11.932%（slower 0/1536）；g8_fixed 9.701%（slower 84/1536）。

同樣取得三副本能力後，candidate 相對標準 RR 的平均收益約 0.484%，相對固定 source g8 GU 約 4.398%。主要新收益是標準 Down replication 的負載平衡，不能包裝成 shared-layout 突破。

所有 36 個配置均完成 1536 個 queries，且全部在有限 PB 與 128B output FIFO 模型下有效。Input copies=1/4、GF/DF 都有逐 query 結果；完整 paired P99、slower/equal/faster、各層、counts/work 歸因及既有 TRAIN-selected g8GU/g2 reference 見 [paired_report.json](paired_report.json)。先前 64-input screen 選中的 g2 不保證在完整 512 個已見 TRAIN 輸入仍優於 g4；本輪兩個 references 都保留。

配置先凍結於 [protocol.frozen.json](protocol.frozen.json)。Shared Down 固定在 chips12..15，每 chip 384rows/C1024/24pages/6waves。其餘三組 g4 各儲存 64 個 routed experts 的完整 Down copy；已知六個 expert IDs 排序後，rank0/1、2/3、4/5 固定派至各組，不搜索 query completion cost。每 private chip job 384rows/C512/12pages/3waves，完整 upload 自己的 A8 vector、fresh cold，沒有 private context sharing。Candidate、標準 RR 與固定 source g8 GU 取得相同選單和能力。Generic IR 可以表達這些操作；表達能力本身不是新貢獻，也不等同文獻已預先提出全部結果。

原 one-copy 係數共 155713536B/layer，triples 為 256376832B/layer；每個 query 的數學 coefficients、useful reads 與 output bytes 不變。相對 onecopy/shared-g4，三副本的 commands、input/output bytes、physical/ECC pages 及各 resource busy work 都逐項相同，改善來自不同 chips 上的負載分配。相對原 shared-g16 anchor 還混合 shared Down 分組改善，不能全歸因 replication，更不能歸因 shared layout。

共同預算是每 chip 兩個 192-WL/four-plane blocks，共 32 blocks/layer。實際執行 ledger 中，GU 在 block0，完整 private Down bands 在 block1；三副本 private chips 的 64×3WL 正好填滿 block1，shared Down 在 block0 WL128..133，合計 28 個非空 blocks。單副本 anchors 為便於核對也全部把 Down 放 block1，因此本次 ledger 為 32 個非空 blocks；這不是其最省 block 的配置。另附 [whole-job packing certificate](equivalent_whole_job_packing_certificate.json)，可在不跨 block、不加 cold/command/input 的前提下重排完整冷啟動 jobs，保留相同模型時序。Allocated capacity budget、actual occupied bytes、nonempty blocks 三者分開列出。

Candidate/triples 所有 1536 個 queries 的 normalized timing DAG 與完成成本相同；每個 input/service 設定各有獨立證書。此證書保留 event duration、resource、dependencies 和 expert ranks，排除物理地址標籤。它只證明這個 formal service contract 的 route independence，沒有證明真實硬體延遲恆定或新的 CBC 性質。

模型沿用 host submit2μs、command250ns、PCIe8GB/s、八條2GB/s channels、每 chip 到 compute 加最後 ONFI output 才釋放 ownership。完整 GU outputs 到 host、activation/A8、private full-vector upload 均有因果 barrier。FIFO 是每 chip 產生 bytes 減去連續 channel drain 的 backlog，沒有隱含 backpressure 或新增大 FIFO。

9.7μs CR cadence、copy500ns、source permutation/drop-before-ECC 及 GU shaping/cold-reset 契約仍是未量測的條件模型；private Down 固定0123且嚴格執行，沒有額外 rescue。Cadence sensitivity 已知可改變 GU shaping 結果；本實驗不把此條件模型叫 native measurement。沒有讀取或更改完整 FFN weights、沒有品質/inference/heldout 評估。

可重用程式是 [replica_full_ffn.py](replica_full_ffn.py) 的 `graph_for`、`Recipes.private_at`、`allocate_down`；執行前來源/輸入、layout ledger 和 gate receipt 均已封存。代表性完整 traces 在 `TRAIN512/trace.*.json`，獨立地址、係數身分、既有 anchor 及 route-independent DAG audit 在 [independent_audit.json](independent_audit.json)。
