# Chip 內 plane read wave：架構依據與獨立審核

研究日期：2026-10-30。這份文件補充主研究報告的硬體前提；模擬結果不等於實體 AiF 晶片支援了新的命令或多 context 執行。

## 先區分兩種浪費

令單一 chip 上一條矩陣 stream 有 `S` useful bytes，page 大小 `L=16 KiB`，一個 wave 最多讀四個 planes：

```text
P = ceil(S / L)                  實際存放的 pages
W = ceil(P / 4)                  baseline read waves
page padding = P*L - S           page 裡的無用 bytes
idle plane slots = 4*W - P       最後一波沒放 page 的位置
page fill = S / (P*L)
page-slot occupancy = P / (4*W)
useful wave fill = S / (4*W*L) = page fill * page-slot occupancy
```

例如 40 KiB stream 需要三個 pages：8 KiB 是最後一個 page 內的 padding，另有一個 16 KiB plane slot 沒用。96 KiB 需要六個完整 pages，page fill 是 100%，但兩個 waves 的 page-slot occupancy 是 75%。縮小 plane stripe 或合併 streams 主要處理後者；除非允許跨 stream 混放同一 page，不能自動消除 page padding。

## 原論文實際提供了什麼

以下頁碼指本 repository 的完整原論文：`AiF - Accelerating On-Device LLM Inference Using In-Flash Processing.pdf`，printed p.533 對應 PDF 第 5 頁。正式出版資訊見 [ACM 論文頁](https://doi.org/10.1145/3695053.3731073)。

| 依據 | 可以據此建模的內容 | 不能直接推論的內容 |
| --- | --- | --- |
| §4.1/Fig.6，p.533 | 矩陣分成 chip submatrices；PE 做 GEMV | 同 chip 任意四個 experts 的命令可獨立執行 |
| §4.2，p.534；`paper/aif.txt:517–545` | 同 block 連續讀取、沒有間隔，才可用 cr-read | 切到另一個 expert/block 仍可延續 9.7 μs；任意 ECC stall 後可無代價保留 recycling 狀態 |
| §4.4，p.536；`paper/aif.txt:724–743` | 每 plane 有 PB；推論時兩個原有 page latches 可放 input vector | 已提供每 plane 任意向量選擇、獨立矩陣 context、獨立累加器與回傳標記 |
| §4.4，pp.536–537；`paper/aif.txt:746–749,761–782,816–823` | 一條命令帶 input dimension、block ID、offset、page count；PB 資料經 ECC/PE、output FIFO | 多組 context 的切換延遲為零，或每 plane 各有 6.4 GB/s ECC/PE |
| §4.4/Table 1，pp.536–537 | ECCLITE 6.4 GB/s；PE 200 MHz；PE 3.98 mW、ECC 45.1 mW/chip | 公開了 MAC lane 數或新增 plane scheduling 的面積、功率與實測能量 |
| §5.2 footnote 9，p.538 | submatrix 跨 IFP blocks 時拆成不同 GEMV operations | 所有 blocks 一律只有一個 cold read |
| Table 2，p.538 | 四 planes/chip、16 KiB/page、28 μs cold、9.7 μs cr-read、2 GB/s/channel | channel 或 PCIe 的輸入輸出傳輸可忽略於完整推論延遲 |

因此多個 PB 提供了可行的資料儲存位置，但**資料有地方放，不等於命令與資料路徑已支援多個 experts 的獨立執行**。

## Multi-plane 位址限制：不能假定任意尾頁可合讀

[ONFI 3.0 §3.1.1，p.51](https://onfi.org/files/onfi_3_0_gold.pdf) 的 multi-plane operations 要求不同 plane、相同 page address，並列出無限制、完整、XNOR 三類 block-address 限制。[ONFI 5.1 §3.1.1，p.92](https://onfi.org/files/onfi_5_1_final_1%2C-d-%2C0%20%281%29.pdf) 則把 page-address 限制交由 device/operation 與 vendor datasheet 決定。兩者都不能用來保證 AiF 的四個 planes 可任意定址。

相同 block stripe、相同 LSB WL 是本研究最保守的合波配置。較新的獨立 plane read 並非空想：[Micron 2022-07-26 官方公告](https://investors.micron.com/news/press-release/2022/Micron-Ships-Worlds-First-232-Layer-NAND-Extends-Technology-Leadership-07-26-2022/default.aspx) 明確指出其六-plane TLC 有每 plane independent read。該公告證明這種能力存在，並不能證明原 AiF 的四-plane NAND 或 cr-read 已具有同等能力。

以上 ONFI 規則由官方 PDF 的搜尋索引正文核對；研究時官方直連回覆 `Request Rejected`，未把其他網站的轉載當成特定 NAND 產品 datasheet。實作前仍須確認目標 NAND 的完整 datasheet、multi-plane 功能位與 read command 行為。

## 三類方法的可實現前提

### 1. 固定一個或兩個 planes 給一條 expert stream

將每 stream 的 striping width 從四降為一或二，讓不同 experts 永久使用不同 plane bins。要以固定權重映射測試：同 token 選到兩個位於同一 bin 的 experts，仍會相撞；不能看到 routing 後才重新指派。

這能增加短 stream 的可併行數，但較長 stream 的每-plane 連續讀取也變長。某一 bin 的兩個 experts 需要兩次 cold run，不會因另一個 bin 空著而自動消失。放入互補 plane bins 也不代表 WL 一定相同；若地址不相容，必須串行，或明確假設 independent-address plane read。多 expert 並行還須 input/context/accumulator 選擇功能。

因此，這是有固定 placement 衝突的**條件式架構擴充模型**，不是 AiF 原論文已證明的功能。尤其 down projection 的每個 expert 使用自己的 activation vector；即使 gate/up 共用 layer input，down 的多 context 需求仍存在。

### 2. 靜態、共同 WL 的單一 supermatrix bundle

gate 和 up 共用同一個 layer input `x`，可垂直串接成普通矩陣。再把同 chip group 內固定 `B` 個 experts 的 gate/up shards 做 row concatenation，也仍是同一個 input dimension 的普通 GEMV。只要實際讀取整個 bundle，包含未被 router 選中的成員，就不需要 sparse plane command 或多個 input contexts。

必要的靜態 metadata 是 `output row -> (expert, gate/up, 原 row)`，host 按此解包，只替 selected experts 執行後續 activation/down。若先對每個 expert 做 chip sharding 再串接 shard，需以 chip-major 順序重排 global supermatrix rows，或由 controller 保存相同的 shard 描述；這是 row permutation，不改變 GEMV 的數學內容。目標模型的 `H=1024/2048/4096` 都整除 16 KiB，完整 output rows 可與 page 邊界對齊。

成本不可省略：bundle 任何成員被選中就讀整包，未選 experts 仍有真實 page 讀取、INT8 乘加、output FIFO 與 result bytes；輸出被 host 丟棄不代表這些工作免費。完整 bundle 命令跨 block 時仍必須分段與重收 cold start。若跳過未選 pages，方案就變成需要新 descriptor/plane mask 的架構擴充。

down 不能直接做相同的 vertical stack，因為其各 expert 的 input vectors 不同。用 block-diagonal matrix 加 concatenated inputs 雖可表達相同運算，卻引入大量零權重與讀取；省掉零元素又需要 sparse/multi-context datapath。保守實驗維持 down 為原本單 expert stream。

### 3. 動態 tail packing oracle

把此 token 所有 ready streams 的尾頁隨意湊成四-page waves，可估計剩餘機會，但若沒有把權重搬移、複製容量、plane 固定性、相同 WL/block、context 切換、block cold starts 及 dependencies 一起建模，它只是理想下界。不能報成可直接排程的方案，也不能與固定 placement 當成同一可實現條件。

## Timing、容量與功率審核原則

對一個連續 block run，`T(q)=28+(q-1)*9.7 μs`；不同 block 或保守的新 stream 另算 cold。對 chip 的最低 service 約束為：

```text
T_chip >= max(每個 plane/bin 的 read service,
              這顆 chip 真正讀取的 physical bytes / 6.4 GB/s)
```

四個 pages 每 9.7 μs 相當於 6.756 GB/s，超過公開的 6.4 GB/s ECC/PE path。必須封頂，不能給每個 bin 各自完整的 chip bandwidth。上述 `max` 是可重疊流水線的約束；在異質 releases、FIFO 不足或 ECC stall 下，未必是可達排程。模擬沿用它時應稱條件式 service model，不能保證實體 pipeline 無氣泡。

獨立審核發現，若先讀一波、停下來 drain ECC、再讀下一波，間隔不符合原文的 cr-read 條件。`plane_wave_model.py` 已依此修正：可明確列出 timeline 的保守版本，每波付 28 μs ordinary read，之後才做共享 ECC drain；9.7 μs 跨 drain 版本另列 `retained_cr_schedule_ns`，明示需要尚未證明的 hold-state 能力。`service_ns` 保留原 baseline 相容的 ideal pipeline bound；兩個時間不可混用。

固定 placement 必須跨 tokens 保持不變；training 選 layout，held-out prompt 只評估。每個 selected projection 的 useful INT8 bytes 必須恰好等於 `H*I`，rows 不漏不重，虛擬 `(chip,plane,block,WL)` 不重疊。對 supermatrix，額外 actual read bytes 要與 selected useful bytes 分開，避免 overfetch 被算成 useful utilization。

容量需分列 actual pages、idle plane reservations、block boundary padding、最後 block tail；LSB 可用容量也不等於完整 TLC raw capacity。未使用 CSB/MSB 可否存其他資料，須與 AiF 原方案及 namespace 隔離條件一致。沒有權重複製的方案不應把動態 remapping 當作免費能力。

目前沒有原 AiF 的多 context RTL，也沒有新增控制器與 independent-plane array 的功率模型。結果最多可報 read pages、cold starts、bytes 的變化和條件式延遲。新增 PPA、每 token Joules 與 MAC lane utilization 不可由 throughput 或 Table 1 靜態功率直接推得。

## 獨立數學反例：bundle 不是總會變快

Granite 的 `g=16` 下，每 expert 的 gate/up 垂直串接後是 64 KiB/chip，一次 cold wave 為 28 μs。兩 experts 的 bundle 是 128 KiB，兩 waves 為 37.7 μs；只有一個成員 active 時，會比單 expert 28 μs 慢，兩者都 active 才比兩條 stream 的 56 μs 快。

對 `E=32, K=8` 的均勻無放回 routing，bundle 寬度 `B` 整除 `E`，觸及的 bundle 數期望可獨立計算為：

```text
E[touched] = (E/B) * (1 - C(E-B,K) / C(E,K))
read amplification = B * E[touched] / K
```

只算 gate/up flash service、忽略 command/transfer 的解析值如下；所有 g16 chip 工作相同：

| B | 期望 touched bundles/layer | actual/useful gate_up bytes | 期望 gate_up μs/layer | 相對 B=1 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 8.000000 | 1.000000 | 224.000000 | 1.000000× |
| 2 | 7.096774 | 1.774194 | 267.548387 | 0.837232× |
| 4 | 5.636040 | 2.818020 | 321.817887 | 0.696046× |

這個例子證明，共選局部性與 inactive overfetch 會決定跨 expert bundling 是否有利。真實 trace 的學習結果應與這類均勻、集中和衝突案例一起解讀；training 最優結果本身不能證明 held-out 收益。

## 實作與結果的獨立審核

可重跑的獨立參考程式為 `results/plane-wave-v1/audit_reference.py`，輸出 `audit_reference.json` 並記錄受審 source SHA-256。它沒有修改 core、runner、routes 或 training plans。

| 獨立檢查 | 結果 |
| --- | --- |
| 與既有 `materialize_plan.WaveAllocator` 比較，block 大小為 4、7、192 WL，共 99 streams、27,539 pages | 每個 physical `(plane,block,WL)` address 完全相同，沒有重複；page padding 與 reserved capacity 相等 |
| 對以上三份 baseline，各抽 50 組 selected subsets | 全部 useful/page/cold/service counts 與既有 `aif_model.Demand` 相等 |
| E=8、K=4，固定兩-plane bins，枚舉全部 70 組 routing | latency 分佈與獨立超幾何公式精確相等；平均 70.4 μs，不能假設動態完美平衡的 56 μs |
| E=8、K=4，固定單-plane bins，枚舉全部 70 組 routing | 同樣精確相等；平均 49.6 μs；只有 16/70 組能達 28 μs |
| 上述 140 次固定 bins 評估 | addresses 前後不變，bandwidth 不超標；保守 timeline 每波付 28 μs，前一 page 完成 drain 後才重用同 plane PB |
| Granite uniform bundle 的有限母體組合公式 | 重新產生本文件 B=1/2/4 的 overfetch 反例 |
| Static bundles 的獨立 E=8、K=4 組合 oracle，B=1/2/4 各枚舉 70 routes | flash service、inactive weight bytes、physical pages、所有 result bytes 對應的 output latency、保守 ordinary-read service 均精確相等 |

`audit_trace_provenance.py` 另產生 `audit_trace_provenance.json`：五個模型、train/test 共十份 caches 的 corpus IDs、順序、model/corpus hashes、原始 mapping permutations 全部相符；200 個 training prompts 與 100 個 test prompts 互斥。每份 cache 的首末 raw JSONL，共 20 prompts、74,752 個 routing events，重建後 128 tokens 的 expert IDs 與 NPZ 逐 bit 相同；所抽 training raw trace SHA-256 也與原 frozen plan receipts 相符。這是抽查 raw cache 內容，加上全部 ID/hash 檢查，並非重新執行所有模型推論。

`static_wave_bundle_study.py` 的 read-only 審查確認：labels 僅在同一固定 chip group 內交換；同一 label 的完整矩陣實際讀取，`out_chips` 使用全部 bundle rows，`selected_useful_bytes` 維持 `3*H*I*K`。固定 labels 在開啟 test cache 前寫入 frozen plan，最後再次確認 hash。其 `capacity()` 明示只計 stream waves，未計 block-end 與 placement-boundary padding；完整容量應看後續 `physical_capacity.json`。host row-demux 時間仍未量測，model 預設為零。

完成後的 artifact 審核另記錄於：

| 審核產物 | 實際覆核內容與結果 |
| --- | --- |
| `audit_final_results.py/.json` | 五模型的 source/frozen/cache hashes 全符；training/test 互斥；train-only winner 可重現；所有 bins 維持原 group 內容量，所有 bundles 恰好包含每個 expert 一次且不跨 group。以舊模型獨立重算 500 個 prompt-model pairs、64,000 個 held-out tokens，saved baseline 的五個分項及總時間逐 bit 相等。 |
| `audit_physical_plans.py/.json` | 15 個替代 bundle address plans，合計覆核 17,602,560 個 physical pages 的 wave 區間不重疊、全 expert gate_up/down rows 恰好一次、同 group ownership、page/wave/block-run counts、block boundary/final tail 容量。另明列 128,640 個 addresses 確認無重複；source/frozen/compressed-file hashes 均相符。 |
| `audit_g16.py/.json` | 不呼叫新 runner 的 cost 函式，直接從固定 bin occupancy、原論文延遲與既有 vector/command 常數重建 g16 模型；五模型、三 widths、三 timing 的 45 組 arrays，576,000 個 token predictions 與 diagnostics 逐 bit 相等。 |

`materialize_static_bundles.py` 與 `static-bundles/physical-plans/*.json.gz` 已補齊共同 WL 的固定地址 metadata；這證明所述配置與容量可一致表示，並未寫入模型權重、NAND 或 NVMe namespace。15 個 plans 是替代配置，不能把其總頁數誤認為同時儲存的複本。

fixed g16 diagnostics 與舊「跨 g 選最佳配置」回答不同問題：前者量化保留原 chip group 時縮窄 plane stripe 的收益，後者檢查它是否超過既有最佳方案。兩個基準都應同時呈現。相同 input/output bytes 或更高 wave fill，不能單獨證明延遲必定下降；新增 contexts、block starts、長 stream 與 bandwidth 封頂也會影響結果。

本次已修正的實際建模缺口是 ECC drain 間隔下誤用 cr-read。仍然開放的硬體問題是目標 NAND 的 plane address/read 能力、跨 stall 保留 recycling state 的電路與時序、multi-context input/accumulator/FIFO 設計，以及 host row-demux 的真實成本。上述審核沒有把數值相等誤寫成這些功能已經實體驗證。
