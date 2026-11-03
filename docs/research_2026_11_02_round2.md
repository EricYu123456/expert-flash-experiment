# Expert 實體配置研究：第二輪證據與投稿缺口

更新日期：2026-11-02。這是持續研究紀錄，尚未建立足以宣稱投稿突破的新穎性。

目前收斂的候選系統主張、原型、算術界線與強基準見 [普通讀取配置候選](ordinary_read_placement_candidate.md)。它不以 first DP／first joint mapping 為貢獻；仍需建立合法、可重用的 lowering 在強控制後的跨形狀效益。

使用者已明確限定「只研究現有硬體能力下的實體配置」。主線使用既有 dense GEMV、完整 output rows、實體地址配置、控制器派送與 host row scatter/concat，保留每個完整 dot product。實體副本與 column 分段也可研究，但前者須計入容量、命令與傳輸，後者須核實既有 output/scale 契約及 host partial-sum 成本。使用者沒有另要求 bitwise exactness，也沒有預先禁止 column 分段；不能把代理自行採用的保守假設說成使用者限制。新 MAC、額外輸入 context、壓縮解碼器、量化或稀疏化不作配置方法貢獻，未公開的 raw-I32 介面不得當作已驗證能力。

## 目前可以支持的結論

1. 不依賴 CR-read 的配置有可重現收益；新公開題目上的 routed FFN 服務模型改善約 2.45–5.52%。這尚不足以支持一般性論文突破。
2. 新增 Granite 4.0 H Tiny 的真實 routing 後，12-chip 重疊分片仍有收益。但與更強的簡單基準比較後，public60 的改善應報 **4.64%**，保守 ECC drain 模型為 **2.29%**；不能只引用較弱基準的 5.01%。
3. 重疊 striping、query-aware declustering、選擇性複製、Hall matching 都有直接先例。有效的實驗結果不等同已成立的新穎性。
4. 已用強基準排除多個看似有希望的方向，避免將配置器漏洞、較多搜尋、缺少成本或額外硬體能力當成貢獻。
5. Qwen3.8 的 48 層含 shared FFN、凍結後單一外部 task，完整列幾何對強 overlapping 控制改善 **9.2349%**。另有 G4 全 40 層／全部 prefill 的固定 W8A8 CPU 參考：1.97 億個 rows 輸出在實體 row 搬移前後完全一致。前者仍為第三方 NVFP4 routes＋名義 I8 成本；後者也不是原 GGUF 品質或原生 AiF 證據。

第一輪方法、原五模型的下界與資料協定見 [第一輪紀錄](research_2026_11_02.md)。本輪完整結果在 `results/research-2026-11-02/round2/`。

## 實驗與硬體界線

數值推論由真實 llama.cpp 模型執行並擷取 router；flash 延遲由相同 routing 的成本重播估計。這不是實體 flash PE 的測量，也不是完整模型推論加速。早期 routed-only 服務模型未計 shared；後續 G4 shared、DeepSeek boundary 及 Qwen3.8 full-FFN 實驗已明確計入它。attention、router、SSM 與 activation/combine 的完整實測時間仍未納入。

目前主模型為 16 chips、每 chip 4 個同地址 planes、16 KiB pages、192 WL/block、普通讀取 28 μs。AiF 將 IFP weights 放在 be-enc(1,3,3) 的 single-sensing LSB，並非把整個 block 改成 pSLC；CSB/MSB 仍可放一般資料。6.4 GB/s 是 ECC/內部資料通道參數，不能改稱已測得的 MAC throughput。250 ns/chip-block command 與 2 μs/host phase 尚未校準。

**原 checkpoint 的數值格式尚未匹配**：六份 GGUF headers 的獨立稽核確認，Granite／OLMoE／G4 的 routed weights 為 Q4_K、Q6_K，DeepSeek／Qwen3／Mixtral 為 Q8_0。Q8_0 也不是 raw INT8：每 32 權重附 FP16 block scale，儲存多 6.25%。目前成本模型沒有執行這些 scale。所以下表是這些 CPU 模型的真實 routing，在名義 raw-I8 形狀下的成本重播；不是相同數值 GGUF 在 AiF 上的推論時間。scale bytes 不能直接補乘 1.0625，因為 pages/waves 的取整可能改變。新 G4 W8A8 參考補上了兩個 workload 的同格式配置／routing 檢查，但明確改變了 GGUF 的 FFN 數值基準，沒有解決原 checkpoint equivalence。詳見 `round2/precision-scope/{audit.json,scope.md}` 及 `round2/full-ffn-integer-reference/notes.md`。

AiF 公開論文只明示 INT8 multipliers、adder tree、FIFO 與 output address；本地 API 是 timing metadata。它們尚未證明 output dtype、累加位寬、scale、飽和或 rounding。plane 各放不同 x、PB replay 與 host partial sums 因此各有待查的契約；不能只因 host API 沒有表達能力，就推定一定需要新硬體。

GU 使用共同 layer input；down 使用每個 expert 不同的 input。每層仍遵守 GU → host 重組及 activation → down，不跨過真正的資料依賴。任意分片需要控制器的地址映射與派送軟體；本研究的 virtual addresses 不是已完成的驅動程式或已寫入 NAND 的權重。

實體容量須區分 useful weight bytes、實體 pages、佔用的 wave slots 與 block reservation。部分快篩使用 20 GiB/chip 的容量情境；這不是已驗證的晶片規格，也不是將 model payload/16 當作每 chip 配額。

## 新公開資料驗證：原先三個非零收益模型

來源為 [OpenAI GSM8K](https://github.com/openai/grade-school-math) test 題目及 [Google MBPP](https://github.com/google-research/google-research/tree/master/mbpp)。每來源固定抽 30 題，只提供題目、不含答案或測試；每題 64 decode steps。三個既有配置不再訓練，180 個 traced/untraced 對照的 token IDs 與輸出 bytes 完全相同。

| 模型 | 既有最佳配置 ms/token | 凍結的新配置 ms/token | 額外改善 | prompt paired 95% CI |
|---|---:|---:|---:|---:|
| Granite 3.1 | 10.147339 | 9.898670 | 2.4506% | 2.3821–2.5211% |
| DeepSeek V2 Lite | 41.316288 | 40.139085 | 2.8492% | 2.8310–2.8667% |
| Qwen3 | 58.105344 | 54.897082 | 5.5215% | 5.4539–5.5850% |

這些數字是 routed FFN 服務時間。固定步數包含 EOS 後的步數，不能視為實際對話完成時間或任務品質評分。模型及候選方向由既有實驗選出，不能宣稱是對全部模型無偏的盲測。

重現：`python scripts/research_20261102_public_validation.py --help`；結果見 `round2/public-validation/summary.json`。不要在重現時重新訓練已凍結配置。

## Granite 4.0 H Tiny：幾何自由度確實有效

[IBM 官方 GGUF](https://huggingface.co/ibm-granite/granite-4.0-h-tiny-GGUF) 的 routed FFN 為 40 層、64 experts、top-6、hidden 1536、intermediate 512。另有 shared FFN 與 SSM；本表不包含它們。

新增資料包含原 train 的 32 題×64 steps、原 test 的 16 題×64 steps，以及 public60×64 steps。配置在後兩份 routing 擷取前凍結，SHA256 為 `a55f17010b7b48c6c396dee92bb1d127a541e28a93b94aee2e42aac9f3dc2538`。新 test/external 的 traced/untraced 執行皆有獨立完整 prefill，token IDs 與 bytes 相同。

關鍵在 down 的 `1536×512` 矩陣。16-chip 等分每 chip 3 pages，但每個 expert 都要一個 4-plane wave。改用 12 chips、各 128 完整 rows，就能每 expert 48 pages 配成 12 個完整 waves。將 16 chips 分成四組，每個 expert 省掉一組，得到合法重疊配置；對 top-6 route，最大 chip 工作量為 5 或 6 waves。

GU 最終仍使用學習後的 g8。down 的 12-chip 省組標籤由 train routing 決定。全部 5,120 個矩陣維持 one-copy row coverage，總 useful bytes 與 368,640 個實體 pages 相同。

| 資料 | 每 phase 學 g 的基準 | 固定 g16 down 的較強基準¹ | 新配置 | 對較強基準改善 |
|---|---:|---:|---:|---:|
| test16 | 20.862661 | 20.822552 | 19.741950 | 5.1896% |
| public60 | 21.196644 | 21.114168 | 20.134125 | **4.6416%** |

¹ GU 相同，down 所有層固定 g16。每 phase 學 g 的方案在五層選到 g4，在新資料上反而較慢，因此不能自動把它稱為最強基準。

加入保守的 NAND + ECC serial drain 後，對固定 g16 down 的改善為 test16 **2.8264%**、public60 **2.2862%**。參數敏感度不能替代實體校準。

相同 g12 的簡單 ID ordering、train marginal ordering 在 public60 分別為 20.366806、20.369130 ms；學習後為 20.134125 ms。因此總體收益的一部分來自幾何，一部分來自 routing 排列，兩者必須分開。

對任意固定六個 experts，隨機 balanced 四色分配的 5-wave 機率可精確計為 `977920/2342949 ≈ 0.417389`。學習後 test16 與 public60 的 5-wave 比例分別約 72.90%、63.98%，低於 train 約 79.57%；不能用 train 頻率當作泛化結果。

重現：`python scripts/research_20261102_geometry_validation.py`；檔案見 `round2/geometry-study/heldout_summary.json`、兩份 physical plans、`audit_uniform_routing.json`、`audit_g12_physical_service.json`。

### 為何尚不能稱為新方法

[Staggered Striping（1994）](https://ftp.cs.ucla.edu/tech-report/1993-reports/930042.pdf) 的 D=16、M=12、stride=4 已產生相同的省組 mask family。[Hypergraph Declustering（2001）](https://ir.lib.nycu.edu.tw/bitstream/11536/30045/1/000172367000003.pdf) 已涵蓋 query frequencies、不同大小資料、階梯成本、容量與最大裝置工作量。以十二個 128-row atoms 表示 expert，即可表達本子問題。

所以「12-chip 重疊分片＋學習省組標籤」目前是有效候選與重要比較對象，尚未證明有獨立的新穎性。FlexLLM、DIAMoND 的全文取得限制仍在，也不能宣稱已排除其重疊。詳見 `round2/geometry-study/novelty_audit.json`。

## 強基準與失敗方向

以下百分比各有自己的分母及實驗範圍，不能直接合併排名。

| 方向 | 實驗或證據 | 決定 |
|---|---|---|
| 單請求 GU/down pipeline、chip permutation | 審核 28,032 份資源時間線；四模型無總 NAND headroom，DeepSeek 不到 1% | 不作主要貢獻 |
| 多輸入 read-once replay | B=1/4/16、不同 q、MAC 敏感度；強同 q 基準後最大 placement extra 很小；native replay 能力尚未證明 | 不將新 context 當配置 |
| 放寬每組 expert 數量相等 | Granite 全 24 層，balanced refit→relaxed 僅 0.1386%；G4/Qwen 小 screen 更小 | 強化基準，未開 heldout |
| 每 expert 不同 shard width | 所有 width 1…16，89 mixed starts、229 fixed-family controls，全部收斂；最終均是單 width/phase | 相對同 family 控制的 mixed attributable gain 為 0 |
| selected-WL union／共用輸入緊密 packing | 即使免算 inactive output 等成本，對強基準的 routed FFN 收益上界 G4 0.1284%、DeepSeek 0.3881%；G4 另有 row alignment 問題 | 不展開 |
| read-disturb 熱冷配置 | 強 WL reclaim 基準後，參考情境剩餘總 NAND 工作量上界不到 0.03%；這不是 p99 延遲上界 | 不作新可靠度論文主張 |
| g8 partial copies | 固定等粒度 tile 的 NAND makespan 被同容量 whole-expert copies 逐 route 支配；input/coalescing 是明列例外 | 不把切細本身當收益 |
| lossless weight compression | 真實 Q8 shards 的壓縮被 page/wave rounding 吃掉；且使用者現有硬體配置範圍不含新 decoder | 封存、排除主線 |
| parity／coded compute 避 erase | raw-I32 假設未證明；強 erase-suspend 與同容量 uncoded 控制使大部分收益消失 | 封存、排除主線 |

各分支對應資料夾為 `single-request/`、`replay-oracle/`、`unbalanced-group/`、`mixed-profiles/`、`selected-wl-union/`、`disturb-oracle/`、`clean-replica/`、`next-directions/`、`integer-parity/`。其中小 screen 的低收益只支持停止展開該候選，不證明全域無更佳方案。1% 的篩選門檻是研究代理的資源分配規則，不是使用者要求停止研究。

### Allocator 修正必須當作基準

Mixtral 原 WaveAllocator 為保留最少 block commands，留出大量 block-end holes。whole-stream 重排可讓每 chip LSB reservation 從 60.1875 GiB 降到 48 GiB，命令數相同；以完整 rows 做合法 block splits 可再到 42 GiB，增加命令與 input reload，useful bytes、pages、waves 均相同。

這是本專案配置器的修正，不是相對文獻最強 bin packing 的新貢獻。AiF 已說明跨 block 的 GEMV 分割。不能報成 30% 的新模型壓縮或延遲改善。詳見 `round2/existing-hardware/notes.json` 及獨立地址稽核。

## 有限副本的最壞 routing 證書

新增一個不用讀任何 routing 的構造，專門處理 G4 down。將每個 expert 固定分成十二個 `128×512` 完整 row atoms，每個恰為 64 KiB／一個 wave。

對每個 expert，先在四個 4-chip 組中使用三組，再將各原組的 atoms 複製到原先省掉的組。這樣可用共享的 atom 副本，提供四種「省一組」的執行配置。每次 route 僅選一個配置，每個原始 row 恰計算一次，不需加法重建或新 MAC。

在固定 canonical atoms、每個 expert 獨立執行其 GEMV 的範圍，要讓任意六個 experts 都不超過 5 waves/chip，至少需要 688 個額外 atom copies：每 chip 不得有六個未複製 atoms，否則選包含它們 owner 的至多六個 experts 就超載；全裝置至多保留 80 個未複製 atoms，768−80=688。

構造以 20 個不同的 special experts，各禁止一個非 default 省組顏色，每色五個，其餘 expert 四色皆可。Hall matching 可對所有六 expert routes 配成 2/2/1/1 個省組，並使所有 ONFI channels 工作量相同。分類枚舉 206 個可行 availability compositions，乘上其組合數，覆蓋全部 **74,974,368** 個 top-6 sets；另有 236 個整數 row reconstruction 檢查。

額外 copies 為 down payload 的 89.5833%，完整 routed FFN payload 的 29.8611%；GU 沒有複製。這是 physical storage 代價，不能省略。

但強比較已否定它目前的 mean/p99 投稿主張：在前三層 train32×64，57 個 whole-expert 備份已匹配相同 p99，三層 mean 只差 **0.875 ns/token**。使用完全相同 688-copy locations 的一般 tile-flow，對全部 4,221 個不同真實 queries 都能達到完全相同成本。可證明的 worst-case 性質不能改稱為對一般 replicated declustering 的平均速度優勢。

此外，[optimal batch codes with multiple queries（2012）](https://pefmath.etf.bg.ac.rs/accepted/rad1246.pdf) 已有 Hall 條件與 `2n−mt` 型 singleton storage bound；[selective replicated declustering（2009）](https://repository.bilkent.edu.tr/bitstreams/938fed97-cc40-48c5-bbd8-01da42e95ae2/download) 已聯合選擇副本及位置。因此目前只保留構造與 scope certificate，不宣稱新理論突破。

還有一個重要界線：單一 dense input vector 可以串接多個 expert 的 input，配合真正存入的 block-diagonal zeros，計算不同 expert 的小 fringes。兩個各 16 KiB 的 fringe 可膨脹成 64 KiB 的普通 block-diagonal GEMV，在一個 wave 中執行。因此「每 wave 只能算一個 expert」不是從原生硬體自動成立的定理。上述 688 證書不能直接擴張成對任意 zero-padded row layout 的最優性主張；zeros、讀取與輸出成本也必須全部計入。

已進一步篩掉特定 fringe 幾何：每 expert 十一個 full-wave tiles、四個 quarter-wave tiles及一個空 chip。即使放鬆實際 chip 可配置性、整數配對與額外傳輸，degree-4 fractional matching 的 dual certificate 仍顯示，40 層候選 NAND 下界比凍結 g12 慢 **6.07%**，每層皆輸。root 獨立重算 train pair frequencies 並逐邊驗證 dual feasible。這只否決該幾何，不外推所有 block-diagonal 配置。詳見 `round2/geometry-study/blockdiag_bound{,.root_audit}.json`。

檔案：`scripts/research_20261102_compound_replica_theorem.py`、`round2/compound-replica/{theorem,audit}.json`、`round2/compound-baseline-screen/{completion,audit}.json`。

## 正在篩選：非對稱 2D 分片

現有 GEMV 可以組合 column partials，再由 host 相加；這是使用者範圍內應檢查的候選。是否能保持原生數值，仍取決於前述 output/scale 契約，不能把它預先算作已驗證硬體。

G4 GU 可使用八個 128-row blocks：一半 chips 各存 1024 columns（兩 waves），另一半各存 512 columns（一 wave），expert 顏色決定兩者放置方向。若 top-6 兩色中最大數量為 m，舊 g8 的 chip peak 為 `3m`，此候選為 `6+m`。代價為每 expert 8→16 commands、1→2 份 row outputs及 host reduction。Qwen down 可用 256-row、512/256-column 的相同結構，top-8 peak 為 `8+m`。

只看這個式子會高估收益；需扣命令、額外輸出、保守 input reload 及實測 host sum。全 train32×64 routing、採用既有顏色而不重新學習的 shape 成本估算，G4 全 40 層在 host partial sum 之前為 19.463174→19.037302 ms，改善 2.19%；若每個新增 reduction 各付 1/5/10 μs，則剩 1.98/1.16/0.13%。Qwen 全 48 層為 54.408742→53.840055 ms，改善 1.05%；每層 reduction 付 1/5/10 μs 後為 0.96/0.60/0.16%。這不是新的 physical plan 或格式匹配後的數值推論。詳見 `round2/rectangular-study/inherited_color_costs.json`。

coarse/fine 的 matched control 已完成：每 expert 可選 0/1/2/3 份 column-thirds 到第一半 chips，其餘到第二半；同一組強 binary seeds、相同控制及多次收斂後，五個篩選 phase 的 mixed 額外收益皆為 **0**。因此不把更多 profile 或更多搜尋當成新機制。見 `round2/adversarial-design/mixed-columns/{screen,audit,completion}.json`。

EPYC 9354 上的 host reduction microbenchmark 已量測：6144 元素的 FP32 add 相對 copy，warm/stream 額外約 0.224/0.499 μs；16384 元素約 0.463/1.804 μs。若需 INT32 partials→INT64→一次 rescale，6144 元素約 1.745/2.231 μs，16384 元素約 2.671/8.962 μs。這是固定 CPU 上的陣列迴圈，沒有 DMA、gather、完整 activation；不能當作 edge host 成本或原生 ABI 證明。見 `round2/rectangular-study/host_reduce.json`。

這也不是「2D 本身」的新穎性：[Cambricon-LLM](https://arxiv.org/html/2409.15654v1) 已描述 page-sized 2D atoms與 input/result 成本，[HDA-MoE 作者實作](https://github.com/angerybob/HDA-MoE) 已有 fractional expert shares 與 coactivation-aware communication。後續若只把相同二色目標由 `3m` 換為 `K+m`，不能稱作新的學習演算法。必須與具相同 native rectangle coalescing 的 2D atom 基準比較。

### 分段數值與 raw-W8A8 控制

新增 runner 沒有改動原 llama.cpp source/library；先修正 CPU-repacked weight 指標不能當成原 GGUF bytes 的問題，再以原 tensor row views 做分段。G4 前三層 GU、Qwen 前兩層 down，兩題 train prompts×16 decode 的 CPU 分段 dot-product L2 誤差很小，但 G4 輸出 token 與兩模型的 ordered routing 仍可能改變。即使輸出 token 相同，Qwen 的部分 expert sets 也不同。不能以單一 dot-product 相對誤差小，宣稱固定 routing 不變或任務品質已通過。G4 full-overwrite control 本身與原 repacked kernel 也有微小差異，已分開記錄。見 `round2/split-numeric/summary.json`。

另建一個**條件性 raw-W8A8 CPU 參考**：每權重 row 固定 scale、每完整 activation vector 一次 scale，切 column 時只切既有 I8 bytes，整數 partials 先以 I64 相加，再做一次 rescale。在 8 個目標 phase、3,276,800 個 scalar observations 上，full/split 整數與 FP32 outputs 完全相同，token IDs、bytes、ordered routing 也相同。獨立 Python 對 7.864M 權重重算量化及 scale digests 通過，原 library hashes 未變。

這只證明該 CPU 數值契約下可分段；沒有證明 AiF PE 原生採用同一 output/scale 契約。raw-W8A8 與原 GGUF 的 routing 本身不同；目前只覆寫少數 decode phase，並非全 routed FFN/prefill 已轉換，也沒有品質評分。它是修復實驗一致性的控制，不是提出新量化法。所有 row scales、host rescale、格式容量與傳輸仍須付費。見 `round2/raw-w8a8-numeric/{notes.md,summary.json,independent_quantization_check.json}`。

## 新檢查：完整 expert 相依下的 GU/down 配置

舊 `single-request/` 的五模型不包含新增的 G4，且其 GU input ownership 較寬鬆。因此新增獨立的嚴格排程器：每個 native job 都 reload input，自 command 開始至 output channel drain 為止獨占 chip；down 必須等自己 expert 全部 GU outputs 到 host、完成 activation、再提交後才能開始。僅一個 native context，不假設 page-buffer weights 重用。所有舊／新配置使用同一 resource model 與 priorities。

全部 train32×64、前三層，以 gate-first、expert-ready、activation/combine=0 的條件性服務結果如下。P99 是三層服務和的經驗分位，非單層 P99 相加、queueing SLO 或完整模型 P99。

| 配置 | mean ms | P99 ms | 備註 |
|---|---:|---:|---|
| Frozen g8 GU／g12 down | 1.342119 | 1.554434 | 同排程的原配置 |
| 同預算 phase-only refit | 1.341220 | 1.553434 | 強化配置訓練控制 |
| 固定 GU、joint fit down 省組 | 1.319144 | 1.509914 | joined chip-work 為訓練目標 |
| GU wave rotations | 1.340531 | 1.420502 | GU 24→25 waves、96→100 pages/expert |
| down g8、與 GU 對向 | 1.337669 | 1.412364 | down 12→16 waves/expert |

joint 對最強簡單 mean 基準的改善為 **1.385%／18.525 μs**，但 P99 比它慢約 **6.91%**。activation 每 expert 1 μs、combine 每 expert 1 μs 後，mean 改善仍只有 1.375%。不宣稱同時改善 mean 與 tail。

僅讓原兩種配置改成 expert-ready，相對各自 barrier 的改善分別為 phase-group 1.206%、g12 2.087%；新增 pipeline 互動相對 phase-group barrier 只有 0.777%。不能把原配置本來的收益全部歸給流水。32-route smoke 曾顯示 opposite/wave 約 1.4% mean gain，但完整 2048-route 後只剩 0.33%／0.12%；這也說明不可拿小 screen 直接作投稿結果。

joined work 只是必要下界，沒有包含依賴與 input/output contention。所有 evaluated schedules 有內建資源與 ownership 稽核；五族另有 78 份代表 trace 的獨立逐節點檢查。sealed sources、原 frozen 配置未變，未讀新的 heldout。資料在 `round2/granite4-pipeline/`、`round2/pipeline-joint/`、`round2/pipeline-joint-replay/`。

GU work/depth 的共同配置亦已完成：增加一個 aggregate wave，可把 G4 GU 的最長單 job 由 3 降至 2 waves，搭配 down 的省組共同分配。它保持完整 rows，strict replay 的 mean／P99 為 **1.315149／1.366366 ms**；對前表最好的 mean 只再改善 **0.3028%**，對同一幾何的 phase-only fit 則為 1.1428%。24 份獨立 trace 稽核通過。tail 改善目前只在前三層的訓練服務重播看見，尚未外推其餘層或測試資料。資料在 `round2/pipeline-joint-depth{,-replay}/`。

流水原理有直接先例：[PowerInfer-2 §4.3–4.4](https://arxiv.org/html/2406.06282v3) 已用 neuron-cluster pipeline 打破 matrix barrier，並配對 Gate/Up rows 與 Down columns；[SLIM](https://arxiv.org/html/2507.09201v1) 也有 PE 端的 neuron GU→down。更細的 GU chunk 若使用完整 down input 的 dynamic amax，仍須等完整 expert 的 GU，不可暗中改成另一套 chunk quantization。

## 共享 FFN 與少量副本：強控制後的結果

前述 routed-only 表格省略了 G4 的 always-active shared FFN。已新增相同嚴格排程器下的完整 shared／routed FFN 重播；仍只使用 train32×64、前三層，非完整模型。shared GU 為 `2048×1536`；shared down 為 `1536×1024`，強 one-copy 幾何使用八個 128-row／2-wave shards 及八個 64-row／1-wave shards，共 24 waves，不能以較弱的 32-wave 均分作唯一基準。

候選在每對 `(p,p+8)` chips 上各存相同兩個相鄰 shared GU row tiles。執行時可分配為 0/2、1/1、2/0，每個完整 row 只算一次；只有現有 GEMV，不需新 context 或 partial-dot 相加。額外 useful weights 是 **3 MiB/layer**，40 層則為 120 MiB；不能因副本放入 allocator holes 而記為零容量。

| 配置／控制 | 三層 mean ms | 三層 P99 ms | 容量 |
|---|---:|---:|---|
| 最好的 one-copy 幾何 `depth_joined` | 1.717762 | 1.785146 | 無副本 |
| 相同 bytes 的 routed／shared hybrid copy 控制 | 1.666159 | 1.808352 | +3 MiB/layer |
| 最好的 paired-shared mean 配置 `source` | 1.642222 | 1.751352 | +3 MiB/layer |

對 one-copy 的 4.3976% mean gain，在補上相同容量的較強副本控制後只剩 **1.4366%**。最佳 tail 的 paired-shared 配置另為 `joint`，P99 1.723738 ms；不能把不同配置的最佳 mean 與最佳 P99 合併成一個設計。這個 tail 對最強 one-copy tail 的改善約 3.44%，仍只屬三層 train 服務和。

同容量的 copy locations 由 GU NAND-only oracle 選出，尚非完整 native FFN placement 的全域最優；因此仍可能低估更強的副本基準。已獨立檢查全部 **110,592** 個 scalar decisions／bounds、54 份完整資源時間線（39,974 nodes），以及所有合法 copied-owner／shared-half 選擇。完整頁面、columns、command、input reload、output 與 PCIe 均付費。見 `round2/shared-copy-controls/{screen,independent_audit}.json`、`round2/shared-pipeline-more/`。

### 實體映像與數值驗證的實際範圍

已將 G4 layer 0 的全部 64 routed experts 加 shared FFN 物化為 16 個 virtual-chip 二進位檔：one-copy useful payload 148.5 MiB，含 shared GU copies 後 151.5 MiB。各配置同樣保留 16 個 12-MiB blocks；這是一層 allocator reservation，不能直接外推全 40 層的 reservation。

每份映像的全部 1,312 allocations 均重讀、核對 SHA256、完整 row coverage、physical non-overlap 與 block/WL/plane 地址。另以不匯入 materializer、不持有 canonical weights 的 fresh process，逐頁 pread、執行 I64 dot，再重建 GU→SwiGLU→完整 activation A8→down→combine。35 個配置案例、**860,160 個 FP32 scalar outputs** 均與未分片的 raw-W8A8 CPU 參考逐位相同。

輸入是固定 PCG64 fixture；六個 routed experts 的 α 為固定 `[1…6]/21`，不是 trace 中的 router scores。這證明該 CPU 契約及虛擬物理映像的 row 搬移正確，沒有證明原 GGUF 數值、完整模型品質、AiF output/scale ABI 或實體 NAND latency。原生數值介面的可用性仍待查核。見 `round2/shared-materialize/{notes.md,summary.json,independent_replay.json}`。

### DeepSeek 邊界副本：小容量，收益仍有限

DeepSeek V2 Lite 的 shared GU 是 `5632×2048`，shared down 是 `2048×2816`；已對 78 個 GGUF／原 BF16 shared tensors 的 metadata 核對。在每對 chips 的固定相鄰範圍，僅雙向複製邊界的 r 個 32-row tiles，每次選一個合法連續 prefix／suffix 執行。r=1 額外 **1 MiB/layer**，26 層為 26 MiB；不變更運算或模型數值。

先前 26 層必要 chip-work bound 的改善不可當成速度。新增前三層全部 train32×64 的完整 native dependency replay 後，採兩個 priorities 中各自較好的結果：

| r | 額外容量／layer | 三層 mean ms | 對強 one-copy mean 改善 |
|---|---:|---:|---:|
| 0，強 one-copy | 0 | 5.821431 | — |
| 1 | 1 MiB | 5.764614 | 0.9760% |
| 2 | 2 MiB | 5.740128 | 1.3966% |
| 3 | 3 MiB | 5.735928 | 1.4688% |
| 11，完整 shared GU copies | 11 MiB | 5.723048 | 1.6900% |

全部 **73,728** 份排程內建資源／ownership checks 通過，另有 36 份代表 trace、40,956 nodes 的獨立檢查。one-copy 使用 shared work/depth frontier 與有記錄 optimality gap 的有限時 MILP/refit，並非故意固定均分。對較差的 down-first one-copy 排程會顯示很大的虛假相對收益，因此本表使用最佳 priority 的控制。

副本與 selector 本身也未建立新穎性：[overlapping range replication（2016）](https://www.jstage.jst.go.jp/article/imt/11/0/11_1/_pdf) 已有兩個 owners 的重疊區域及動態 boundary split；在這個固定 copy graph，generic replicated-tile allocation＋最佳連續 coalescing 可對全部 **2,129,920 pair-query cases** 精確匹配 native occupancy selector，差異為 0 ns。系統貢獻若要成立，必須來自更好的 budget／owners／phase-DAG 配置與原生執行實證，而非將 split minmax 另命名。見 `round2/deep-shared-boundary/`、`round2/deep-shared-replay/full3/{screen,independent_audit}.json`。

共享專家的複製、依 batch load 選副本與 shared/routed overlap 亦有直接先例：[DeepSeek-V3](https://arxiv.org/abs/2412.19437)、[官方 EPLB](https://github.com/deepseek-ai/EPLB)、[官方 LPLB](https://github.com/deepseek-ai/LPLB)。僅說一般方法「可表達」本問題，並不能否定一切系統貢獻；但這些可建構的同容量、同 native-cost 控制必須實際比較。

## 跨架構界線

AiF 的「每 wave 讀取有用 I8 係數」下界只適用相應的數位 sensing/PE 模型。[NASiC](https://arxiv.org/abs/2605.23294) 使用類比 SL input、BL current/ADC 與 CAM expert selection；應另計 input expansion、WL/CAM passes 與 ADC cycles，不能直接套同一 64 KiB floor。它本身已有 expert bitline interleave，不能把相同機制改名為本研究貢獻。

[NVLLM](https://arxiv.org/abs/2604.25699) 有 plane cluster、FIFO 與 OoO ECC，其可執行配置及成本也須獨立處理。新找到的 [3D-FPIM 作者 artifact](https://github.com/SNU-HPCS/3D-FPIM) 已成功建置 compiler、產生原生 MVM trace；完整電路模擬仍缺可校準的 parasitic configuration，不能自行填入數字當實測。

3D-FPIM 的真 G4 前三層 train movement screen 已完成；與強 marginal affinity 比，query mapping 只改善 movement units 1.11–1.73%，相同 local aggregation 控制下為 1.41–3.50%，沒有 latency 或 bytes 主張。12 個獨立核對證明目前的目標精確等價既有 connectivity hypergraph metric；22 次 native trace checks 中 20 次通過，另兩次是原版 random allocator assertion。已指出並繞開原 tile-graph partitioner 的 endpoint bug，不以它構成弱基準。

Depth co-packing 也沒有形成新結果：Granite3 可純跨 DNN 層密排至 **1,152 cores**，恰達 73,728/64 的容量下界，且任意同層同 phase 至多一個 active tile/core，不需 MoE anti-coactivation。G4 完整 40 層的 relaxed expert geometry 下界為 5,760 cores，仍大於原 5,376。非零 depth start 的 accumulator reset、weight programmer 及 ADC scales 則仍未知，不以自訂 interpreter 假裝確認。既有 [3D-aCortex §4.3](https://arxiv.org/pdf/1908.02472) 及 [IBM 2024 公開申請](https://patents.google.com/patent/US20240086682A1/en) 已包含跨 kernels／expert tiers 的 packing 概念。見 `round2/3dfpim-placement/{notes,depth_capacity_audit,depth_sources}.json`。

另已取得能執行真正 numeric operations 的 [IBM 官方 3D-CiM simulator](https://github.com/IBM/3D-CiM-LLM-Inference-Simulator) 與衍生 [MoEwithPIM](https://github.com/superstarghy/MoEwithPIM)。兩份 upstream 均未改動；固定 `(tile,tier)` weights、兩 tiers 的獨立輸出、600×700 矩陣的 partial-sum／concat、explicit-route FFN 均通過 dense reference。外層 programming 修正了一個 fringe index bug，未修改 native F.linear／scheduler。

這個 backend 的小型 phase-packing 候選亦由強控制否決：同四 tiles×六 tiers、兩層完整 weights，普通 phase-cyclic 配置在全部 36 個 query pairs 及 24 個 down-bank 置換均達原生 DAG critical-path 下界；IBM 為 712 ns，衍生版 828 ns。勝過預設 mapper 的部分是普通 cyclic baseline 的改善。其 native 是 ideal FP32 AIMC 模型，未獨立計入 SwiGLU／top-k／α multiplication，不能充當 AiF 或已製 NAND 精度的證據。見 `round2/moewithpim-audit/{notes,sources,moe_native_probe,moe_phase_cyclic}.json`。

### 新形狀資料與下一輪範圍

已找到 [Qwen3.8-Flash-Next 公開 routing traces](https://huggingface.co/datasets/aswinkumar99/qwen3.8-flash-next-expert-traces)。固定 dataset revision `fee2af4ea0da212d853d82943dad3897653066a2`，先只取同一 cpp task 的 smoke-batched stream 作 exploratory train，再於全部 240 份配置凍結後取得一個外部 document-QA task。checkpoint 為 RadixArk NVFP4、gate BF16；不能將這份 trace 假稱為原始 BF16 或 raw-I8 模型推論。train／外部各 20 組公開 hidden×routerW 在 BF16 rounding 後皆與紀錄 top-10／top-32 精確吻合；原始 recorder engine commits 仍無法取得，算術核對不等於重現該 recorder。

[官方固定 config](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/21a7c3ca68484a9987d62fd51b78130d9b9a11f8/config.json) 給出 D=2560、I=640、E=512、K=10、L=48。route-independent whole-row capacity DP 顯示 GU 的均分 16-chip 工作量 64 waves 可用較均勻的 integer-wave row geometry 降為 51；down 為 32→26。這只是每 expert 工作量，**不是 latency 改善百分比**。完整 rows 下的有用係數下界為 50／25，達到它但深度較高的 g10／g5 等配置亦列入控制。見 `round2/qwen38-placement/geometry_certificates.json`。

### Qwen3.8：48 層與凍結後的外部驗證已完成

以下全為 **48 層 FFN（含 always-active shared FFN）服務時間的軟體重播**。每個 chip job 完整持有 command→input→ordinary NAND→output；共同計入 host submit、PCIe、八組 ONFI channel、完整 GU→host activation→down 的依賴。沒有 CR-read、線上搬移或權重副本。activation／combine 為 0 的樂觀控制，rectangular partial reconstruction 每 expert／phase 付 1 μs；參數尚未由硬體校準。router、attention、PLE 等其餘模型時間未納入。

所有配置只用 train 的 16 個 requests、steps 0,4,…,796 共 3,200 tokens/layer 擬合，再重播全部 12,784 decode tokens。這 16 requests 是重複的同一 cpp task，不能當成 16 個獨立任務。phase-fit manifest 中的「…798」是文字錯誤，實際 indexing 為 796；sealed source 未改寫。

| 同一套資源／依賴模型 | Train mean / P99 (ms) | 外部 mean / P99 (ms) |
|---|---:|---:|
| 固定 g16 完整列分配 | 89.009520 / 89.009520 | 89.009520 / 89.009520 |
| 依 train phase bound 選 disjoint group | 88.320079 / 91.573045 | 91.225482 / 94.531732 |
| 全寬度搜尋的 cyclic overlapping stripes | 82.536814 / 84.264359 | 84.645319 / 85.752648 |
| 51＋26 waves 的完整列幾何 | 75.959594 / 76.639660 | 76.828409 / 77.407865 |
| 50＋25 waves 的 rectangular 控制（有 ABI 條件） | 73.801574 / 74.420114 | 74.624175 / 75.207012 |

外部完整列幾何對較強的 overlapping stripes 改善 **9.2349% mean**；對 g16 改善 **13.6852% mean／13.0342% P99**。不能只挑外部較差的 train-selected group baseline 報較大的百分比。rectangular 再比完整列改善 **2.8690% mean／2.8432% P99**，但 AiF 原生 output/scaling/partial-sum ABI 仍未確認，這一列只屬有條件控制。兩種幾何與 declustering 各有直接 prior，以上改善尚非新方法或投稿突破。

外部只有一個 document-QA task／一個 client session，41 個有 decode 的 work requests 共 **5,519 tokens**；額外 **11,866 warm-up decode tokens 已排除**，prefill 另存。外部 equal-request mean 為 overlap 84.692687、whole 76.858740、rect 74.680798 ms；完整列及 rectangular 的上述比較均無負收益 token／request。這些 requests 相關，不能用它們宣稱跨 41 個獨立任務的顯著性。

凍結紀錄：`native-full48/plans.frozen.json` SHA256 `9db047b5335c4d95e45755dc62ee1b545ba97b1119ec39294d09a86299b5e505`；外部 routes SHA256 `2111de845e4e8938e2158d0fc778f7910c48879184aeb1900759cd764d5f4736`。凍結發生於外部 payload 取得前，240 plans 事後 hash 全不變，沒有 refit。第三方 NVFP4 routes 只驅動 nominal raw-I8/A8/4B-output cost；**不是原 NVFP4 checkpoint 在 AiF 的執行或品質證據**。

外部 extractor 曾與 replay runner 發生命名碰撞；已將實際執行的 extractor 以相同 SHA 恢復為 `research_20261102_qwen38_external_extract.py`，另存原始 snapshot，四份 NPZ hashes 均不變。重現時須讀 `qwen38-trace-audit/external-doc-grounded-stride1-short/provenance_source_erratum.json`，不能把現同名 replay runner 當作舊 extractor。

加速後的 compiled replay 與原 Python Graph 的 230,112 筆前三層 costs 逐筆精確一致。另一個獨立 set-scheduler 對 train／external 各 399,600 個代表 nodes 的 start/end 全部一致，並分別重算全部 3,068,160／1,324,560 cost/bound scalars 與 48-layer paired mean／P99。這證明軟體模型及統計一致，不能替代實體時間校準或數值 ABI 驗證。結果見 `qwen38-placement/{native-full48,external-doc-grounded,independent-audit}/`。

### Joint placement 與自由 masks 的下一個強控制

初步 GU/down joint fit 在加上普通 global complementary placement 後，前三層的額外 mean 收益只剩 whole **0.2045%**、rectangular **0.3105%**；不以 phase-only 的弱比較主張突破。已有 [PIMCOMP](https://arxiv.org/abs/2411.09159)、[Fast-OverlaPIM](https://arxiv.org/abs/2407.00604)、[Stratum](https://arxiv.org/abs/2510.05245) 等 joint mapping／readiness prior；後兩者的 native capability 不可直接當成 AiF 能力。

另外測試了將 16 種 cyclic profile 放寬為每 expert 的自由 chip 子集合，嚴格保持每 chip 的 row-type inventory、所有完整列與 one-copy payload。這本身是強基準：既有 [Dependent Rounding in Bipartite Graphs](https://crab.rutgers.edu/users/rajivg/publications/depRound.pdf) 已能維持整數雙邊度數與 query-subset negative correlation，不能把「自由 masks＋容量平衡」命名為新發明。後續若再用已看過的外部 task，將明列為 development validation；新的最終測試需另保留未使用資料。

本輪已完成，沒有形成新突破。全為相同前三層、16 requests×32 decode steps 的 512 個 query，無外部資料擬合／選擇：

| 相同完整列幾何與固定 shared 位置 | Mean / P99 (ms) |
|---|---:|
| matched cyclic，2×24,000 valid swaps | 4.643684 / 4.693489 |
| coupled GU/down 自由四邊交換 | 4.647636 / 4.695956 |
| 各 phase 自由四邊交換 | 4.655580 / 4.723887 |
| route-blind diverse masks，4 固定 seeds，僅 train bound 選擇 | 4.708095 / 4.724888 |

每個 chip 的 75／76／101 GU rows 與 96／198／199 down rows 清冊均精確保持。Diverse 控制用全部 560 個 GU triples 刪除三個 full cyclic orbits，留下 512 個互異 masks，再以標準整數 flow 完成 GU rows 與互補 Down10 masks；12,288 個矩陣 coverage 檢查通過，平均仍慢 1.39%。這些負結果不構成對任意自由 masks／全域最佳解的否定。資料在 `residue-mask-pilot/`、`diverse-mask-control/`。

另以 5×5 個 GU/down 工作量／深度組合，在每層搜尋全部 16 個相對位移，使用相同 2×24,000 valid profile swaps 與完整 native replay。最佳 balanced-GU＋g13-down 為 4.642270 ms，同搜尋的 balanced/balanced 為 4.642975 ms，差僅 **0.0152%**。雖 g10／g5 每 expert 合計只讀 75 waves、達有用係數工作量下界，卻因較長 jobs 和 GU→down 因果順序，為 4.697766 ms，慢約 1.18%。不能把較少總 waves 當作較低 latency。見 `workdepth-control/`。

另一個限制只關乎 **routed top-10 的 joined pooled-chip-work**：互補 GU3／Down10 masks 可保證每 expert／chip 至多 5 waves，故任意 top-10 至多 50；由 high incidences 的平均計數，又必有一個 chip 可挑出十個各至少有一個 high 的 experts，故最優 worst case 至少 50。這只證明這個既有 geometry 的 robust pooled 指標已達界；不包含 shared expert、controller、channel 或 causal latency，不是新 batch-code／rounding theorem。完整證明及直接 declustering prior 見 `free-mask-prior/notes.json`。

### 實體地址降階已完成，未冒稱寫入真實 NAND

已將全部 240 份凍結 plans 降成 chip／four-plane block／LSB wordline 地址。每個 job 在單一 192-WL block 內，四個 planes 共用 block/WL 地址；係數 page 依 plane 0…3 排入，保留原本 command 數，不另假設跨 block 合併。另用獨立 bitmap 及逐 row interval 演算法檢查全部配置的地址不重疊、每個矩陣的係數完整覆蓋。

| 48 層 routed＋shared 配置 | Useful weights (GiB) | Programmed pages (GiB) | Reserved waves (GiB) | Allocated LSB blocks (GiB) |
|---|---:|---:|---:|---:|
| g16 | 112.719727 | 120.225586 | 144.225586 | 144.339844 |
| overlapping stripes | 112.719727 | 117.225586 | 117.225586 | 117.375000 |
| whole rows | 112.719727 | 115.725586 | 115.725586 | 115.875000 |
| conditional rectangles | 112.719727 | 112.725586 | 112.725586 | 112.875000 |

這是靜態虛擬地址證書，不是下載完整 checkpoint、實際 FTL programming、PE 數值執行或 flash 量測。CSB／MSB 仍未當成 LSB 權重容量。標準 best-fit allocator 只是將可執行配置具體化，不列為新壓縮／新 mapping 原理。見 `physical-addresses/{summary,address_check}.json`。比較圖在 `figures/frozen48_controls.{png,pdf,svg}`。

### 完整 FFN 的同格式數值參考與 wave compiler

新 CPU 參考不再只改前三層 gate/up 或 decode。G4 全部 40 層、240 種 routed/shared gate/up/down weight tensor 名稱，兩個訓練 prompts 的 75／153 prefill tokens 及各 16 decode 均執行相同固定 W8A8 算術。placed kernel 實際讀 chip-packed 完整 row bytes，使用相同 scale，回填原 logical row。197,017,600 個 row outputs 的 exact I64 dot 與跨執行 F32 檢查皆零差異，3,411,968 個 logits bitwise 一致，routes／生成 tokens 相同。兩種檢查用同一批 rows，不能加倍計數。

這不是原始模型的量化品質結論：原 GGUF control 的 token agreement 為 programming 0/16、summarization 4/16，free-running routes 也改變。Attention/router/SSM 維持相同原 GGUF host 操作。Harness 仍先執行原操作、再以參考結果覆寫，71.48 秒僅為驗證程式 wall time，不可用來比較 flash 效能。初次末層 prefill coverage 失敗已保留，修正後重新凍結，全部 15 個輸入及 7 個 reused source/library hashes 均吻合。

將上述實際 W8A8 decode routes 套用舊凍結 plans，全部 40 層加相同一份 shared geometry 的 mean service 為 phase-group **24.808711 ms**、whole rows **23.742340 ms**，改善 **4.2984%**。只屬兩個早先訓練 prompts／32 tokens 的同格式探索，不列為新的 held-out 或模型品質驗證。見 `full-ffn-integer-reference/{notes.md,main/,source-gguf-control/,frozen-placement-replay/}`。

另已建立可指定 rows／columns／chips／planes／pagebytes 的 `research_20261102_whole_row_compiler.py`，產生完整 row bands 與 work/depth certificates。基本整數 covering DP 及 divisibility lemma 不宣稱新穎。獨立直接列舉 3,456 個小形狀、39,060 個 depth-bound 比較及 1,664 個 exact-fill cases 全吻合；七模型十四 projection shapes 的 coverage／ceil 成本／單 block 條件通過。對任意超出單 block、跨 rows 的量化 metadata 或額外 jobs，仍需另行 native lowering。見 `whole-row-compiler/{certificates.json,independent/}`。

### 額外 padding 輸出成本

AiF 的 native 描述明示 input dimension／block／offset／page count，未明示 useful-row stop。完整 rows 可保持數值，但 padding 若也形成 outputs 就不能省去 drain。新 sensitivity 針對所有凍結配置、全部外部 tokens，分別傳回 requested pages／完整 reserved waves 中所有完整 rows；channel 與 PCIe 均支付 4B/row。

| 成本契約 | Whole rows mean (ms) | Overlap mean (ms) | 配置改善 |
|---|---:|---:|---:|
| Useful rows only | 76.828409 | 84.645319 | 9.2349% |
| Pages 中所有完整 rows | 76.834621 | 84.656573 | 9.2396% |
| Reserved waves 中所有完整 rows | 76.834621 | 84.656573 | 9.2396% |

Whole 的增加為 0.0081%，overlap 為 0.0133%；兩者的 pages 已填滿 wave slots，故後兩項相同。這不是已驗證的硬體上界：末尾 incomplete-row flush、unused planes、result dtype 與真正原生語義仍未知。`output-fringe/` 保留原模型、source hashes、各配置與 Python diagnostic traces；沒有改寫原始 useful-row 結果。

獨立 audit 重算 2,649,120 costs、799,200 saved nodes 與 1,440 個 first/last-query schedules 全吻合。Rectangle 的 291 個 per-token aggregate 成本在增加 drain 後稍降，最極端三筆亦獨立重現；它源自 greedy resource order 改變，並沒有減少工作。因此這個 scheduler 的 makespan 不可直接當作單調硬體上界。

### Controller 成本會使幾何選擇翻轉

以同一組 240 frozen plans、外部 256 個固定位置、全部 48 層與 requested-page fringe output，另執行 11 個參數情境、675,840 costs。預設參數的 61,440 筆與上一個完整實驗逐筆一致，2,640 個 Python diagnostic makespans 也逐一吻合。

| Per-chip matrix command 假設 | Whole mean (ms) | Overlap mean (ms) | Whole 改善 |
|---|---:|---:|---:|
| 0 ns | 76.535195 | 84.369459 | 9.2857% |
| 250 ns | 76.845573 | 84.664488 | 9.2352% |
| 1 μs | 77.931664 | 85.695291 | 9.0596% |
| 2 μs | 79.516787 | 87.141520 | 8.7498% |
| 4 μs | 87.021495 | 90.464510 | 3.8059% |
| 5 μs | 96.493326 | 92.579877 | −4.2271% |
| 10 μs | 170.427600 | 141.627600 | −20.3350% |

這是必要的負面界線：whole 發出 16,896 commands/token，overlap 為 14,016；controller 昂貴時，原本的 wave 節省不足以抵銷多出的 commands。250 ns 尚未校準，不應宣稱跨所有現有控制器都改善。各 family 未針對新參數 refit，這張表也不是每個情境的全域最佳幾何。

Controller floor 只對本次單一 serialized command-dispatch resource 成立。多路 controller 應另建 resource graph，不能套用同一翻轉點或視為跨硬體下界。

獨立 sanity audit 核對 240 plan hashes、96 份 whole/overlap plans 的 command counts、12 條代表 traces 與 12 次獨立 C++ one-event 重播。10 μs 時所有 48×256 個已存 layer costs 符合 controller work＋固定尾部的閉式值；全 48 層尾部為 1.4676 ms，兩配置多出的 2,880 commands 恰解釋 28.8 ms 差異。見 `parameter-sensitivity/independent/cost_sanity.json`。

普通 read 改為 37／56 μs 且 command 維持 250 ns，whole/overlap 的改善為 9.2895／9.3474%；host phase 0／10 μs 為 9.2458／9.1935%。資料在 `parameter-sensitivity/`，圖為 `figures/controller_sensitivity.{png,pdf,svg}`。下一個可驗證主張應是依原生 read／command／輸出契約選擇合法配置的 compiler，而非固定推銷某一個幾何。

另查核了 expert sensitivity／實體可靠區域的候選。AiF 是 ECC 後才進 PE，不能用 pre-ECC raw BER 注入推論誤差。已有 [Robust Heterogeneous AIMC for MoE（2026）](https://arxiv.org/abs/2603.02633) 及 [Dyn-A-Map（2026）](https://doi.org/10.1038/s44335-026-00072-6) 等近 prior；公開 3D-NAND／SONOS simulators 的 noise 不提供可直接使用的實體位置 covariance。未捏造空間噪聲來製造 placement 收益。見 `round2/physical-error-prior/review.json`。

以上新增來源與執行證據還在整理；這些論文所提架構與模擬器不等於已取得實體製造晶片。

### 新保留資料：同格式完整 FFN 與固定 teacher 品質

在固定 acquisition protocol 後，取得先前未用的 8 個 WikiText2 raw test chunks 與 8 個 ARC-Challenge validation 完整題目。使用 pinned revisions、seed 20261103 的 hash ranking；舊 G4 plan、模型與數值協定 hash 都在 target payload 下載前核對凍結。答案與 prompts 分離，沒有據此調整量化或配置。來源 metadata 頁曾顯示非 target train preview，已如實保留；WikiText card 的 license YAML／正文版本差異也未擅自統一。見 `fresh-format-heldout/{acquisition_protocol.frozen.json,manifest.json,SOURCE_README.md,selection_audit.json}`。

固定 teacher 的 8×128 tokens／1,016 個 next-token losses，來源 Q4 GGUF 的 mean NLL 為 2.7981087404，完整 W8A8 FFN 為 2.7990450129；PPL **16.41357507 → 16.42894984**，ratio **1.000936711**。每 chunk ΔNLL 介於 −0.05312718 與 +0.04502881，五份增加、三份下降；這是小樣本的相對 Q4 敏感度，不能推論品質等價、non-inferiority、原始 BF16 品質或 ARC task accuracy。所有 logits 有限，獨立 context 重跑第一份 W8 輸入的 12,845,056 logits bitwise 一致；first-two-logit 的 F64 NLL 重算差異不超過 4.44×10⁻¹⁴。

同一固定輸入下，40,960 router records 的 ordered exact 為 53.7109%、set exact 為 81.2354%、平均 intersection/top-6 為 96.7814%。這再次說明應以實際 W8 routes 驗證配置，不能以 free-running token agreement 代替量化品質測量。

16 個 prompts 的 unchanged-binary unsharded／placed runs 均使用完整原 prompt、各生成 16 tokens。每 mode 共 1,986 prefill＋256 decode tokens；40 層、全部 6 種 FFN projections 的 130,560 callbacks 均覆蓋。**1,698,897,920** 個 placed row outputs／跨執行 F32 scalars、**27,295,744** vocabulary logits、所有 prefill/decode routes 與 generated IDs 全部一致。這裡的兩種 row 檢查仍是同一批樣本，不能加倍計數。

將這批實際 W8 decode routes 重播至兩個舊凍結配置，40 層與同一份 shared FFN 都計入：

| 新保留資料 | Phase-group mean (ms) | Whole-row mean (ms) | 降低 |
|---|---:|---:|---:|
| 全 16 prompts／256 decode positions | 25.57371831 | 24.56473048 | 3.9454% |
| WikiText2 test 的 8 prompts | 25.54926989 | 24.57154402 | 3.8268% |
| ARC-Challenge validation 的 8 prompts | 25.59816673 | 24.55791694 | 4.0638% |

全體 P99 為 **26.573161 → 25.5723261 ms**。16 個 prompt mean 全部降低，範圍 3.5903%–4.2831%；256 個配對 token 也全部較低。這些是此 16 prompts 的描述性結果，不能把相關的 tokens 當成 256 個獨立任務。獨立 set-queue／linear-event engine 重排全部 **20,480** 個 layer-query/layout cases、檢查 **3,266,560 native jobs**，成本、resource locks、因果與所有統計均一致。仍是未校準的 FFN service model，不是實體 flash、完整模型時間或 native AiF numerical ABI。見 `fresh-format-heldout/{validation_summary.json,quality/,generation/,frozen-placement-replay/}`。

### Host row-scale metadata 與實測 scatter 成本

七個模型的 routed/shared 完整 row scales、address handles、輸出 bytes 已另行計數，不把 FP32 scales 插入 I8 MAC 係數頁。Qwen3.8 全 48 層 routed＋shared 的 row scales 是 **378,224,640 B（360.703125 MiB）**，每 token 選中輸出的 useful bytes 是 **8,110,080 B**。32B/band address descriptor 是本次可檢查的 host 結構假設，不是原生硬體要求、不可壓縮下界或每 row pointer。

使用現有 EPYC 9354 的 pinned single-thread C++，由 16 個 aligned chip staging buffers 將 opaque 4B rows 回填 logical outputs。七個主 copy cases 的 warm／大於單 CCD L3 的 rotating stream，以及兩個僅作條件參考的 I32-rescale cases，共 18 種配置×256 次採樣，所有 scalars bitwise／checksum 驗證通過。Qwen whole-row 的 copy median／P99 為 warm **0.242／0.282 ms**、stream **1.166／1.243 ms**；overlap stream 為 **1.230／2.125 ms**。G4 chosen 的 stream 為 **0.428／0.470 ms**。細小差值與 outliers 均保留，未把背景噪音當配置加速。

這是 host fallback 的量級檢查，未量到 edge CPU、DMA、driver callbacks、input quantization 或實體 flash。即使 whole 額外的 2,880 bands/token 每 band 再有未建模的 **2.715 μs** 軟體成本，才足以抵銷先前約 7.819 ms 的 modeled gap；不能把已計 controller dispatch 再算一次。若既有 DMA 支持直接 logical row offsets，這份 copy 可能不需要，但未假設該介面存在。native I32／完整 scale ABI 仍未證實；conditional rescale benchmark 不能證明它。見 `host-row-gather/{protocol.frozen.json,ledger.json,measurements.json,notes.json}`。

### 通用後端與依成本選幾何的 compiler pilot

新增 `research_20261103_ordinary_read_backend.py`／`ordinary_read_native.cpp`，以 chips、channels、planes、page/block 大小、輸入輸出 bytes、read／command／host／傳輸成本建立同一個 native-job resource graph。移除舊 Qwen 專用的 common/private upload 常數，shared down 使用自身完整 intermediate dimension；每次命令只處理完整 output rows、重新載入完整輸入，並持有 chip 至 output channel drain。跨 block 的 row band 會先切成獨立完整列命令，沒有假設原生 partial accumulation 或多 input contexts。

第一輪 regression 對 **7,744** 筆舊 G4／Qwen costs 完全一致；不同 chips／planes／channels、GF／DF、fringe、zero-duration 與多命令/chip 的 **144** synthetic costs 也與獨立 ready-queue scan 一致。**1,278** 個直接 row partition 小例核對最低命令數內的最少 waves。這只證明明列的 lexicographic lowerer；更多命令可能換得較少 waves，不能稱全域最低 service cost。獨立 audit 發現 imported malformed input metadata 可少計 upload，已加入 wrapper／CPP guards，原 regression 重新全過；另獨立核對全部 **20,480** fresh costs 及 **66,280** trace nodes。正常 `lower_rows` 產生的配置本來沒有此不一致。

`research_20261103_ordinary_read_compile.py` 已串起 training-only geometry／label fitting、GU/down 相對位置、合法 lowerbounds 排序的 event replay、單 block 地址與 per-chip capacity ledger。提供全部 disjoint groups、全部 cyclic widths，以及 whole-row work/depth 候選；所有 family 都有 16 個 GU/down 相對位移。LB 只剪除此固定 finite pool、既定 greedy scheduler 與同一 selection queries 中不能更好的候選；達 replay budget 者明列未證最佳。

首個 layer 的 development pilot（512 fit queries、32 selection queries、相同每 geometry 擬合預算）中：Qwen3.8 在 250 ns command 下，較強 cyclic **1.716468 → 1.593944 ms**；5 μs 下 compiler 改選 balanced GU＋cyclic down，**1.869296 → 1.857857 ms**，但後者 budget 用盡、未證 finite-pool 最佳。G4 在 250 ns 下只比強 cyclic 改善約 **0.28%**，5 μs 則直接保留 disjoint 控制；DeepSeek 首個 MoE layer 在 250 ns 下 **2.013708 → 1.940354 ms**。Granite3／OLMoE／Qwen3／Mixtral 在這個 pilot 未優於相同普通條帶控制。

這個結果支持成本會改變合法幾何選擇，也暴露零收益與小收益模型；它尚未建立新穎性、held-out 泛化或投稿突破。DeepSeek 首層 raw model ID=1，第一版 driver 以 0 讀不到資料而停止，四份已完成 G4／Qwen 輸出保留；修正後 CLI 明確使用 MoE-layer indices，並記錄 physical layer IDs。原執行 source snapshot 與 termination 紀錄均在 `ordinary-read-compiler/pilot-v1/`；其餘五模型在 `pilot-v2/`。

全層 v3 development 已完成 342 個 layer/cost cases（三模型 48／40／26 層×250 ns／5 μs／10 μs），並共同打包為 27 份多層常駐虛擬地址。**此版收益不能直接歸因新幾何**：attribution audit 發現部分 whole-row profiles 與既有 cyclic variants 完全等價，只因名稱不同而多做一次 seeded label search。特別是 Qwen3.8 在 10 μs 下 37 個有正收益的 layers，全部只選到既有 cyclic 幾何；此收益屬額外搜尋機會。G4 在 250 ns 的 38 個正收益 layers 則有 37 個至少含不同幾何，不能因一個反例反向抹掉全部結果。見 `full-campaign-v3/geometry_alias_audit.json`。

v4 已合併等價 search spaces，保留 v3 的 canonical first fits 並逐項重算目標／histogram，不重抽新 seeds；重新選 geometry、重播及全模型地址打包。獨立 tiny BFS 另指出「variants multiset＋初始化 histogram」是充分但非必要的 key，E8 g2 有漏合併反例；當前 E64／E512 的 684 個實際 phase inventories 不受影響。v5 之後使用**初始化已分派的實體 row-vector histogram**：在目前 unrestricted pair swaps 下，這才精確決定可達的 expert assignments；unused ordinal variants 不應改變 key。若改用 single-label moves／expert-specific constraints，須重證此等價。

v4 共 342 records、27 maps、14,160 canonical phase fits。去重後，Qwen3.8 的 10 μs 額外改善為 **0%**；不能保留舊版多 seed 的假收益。250 ns 下的全 MoE＋shared TRAIN selection：Qwen3.8 cyclic **82.203331 → 76.566537 ms（6.8571%）**、Deep **52.356580 → 50.453269 ms（3.6353%）**、G4 **23.549926 → 23.274385 ms（1.1700%）**。5 μs 分別為 1.2596%、2.1020%、0%；10 μs 為 0%、0.3746%、0%。部分高成本 cases 用盡 replay budget，不能稱 finite-pool 最佳。這批仍使用來源 checkpoint 的 routes＋名義 I8 成本，是 development evidence。見 `full-campaign-v4-deduplicated/`、`independent/geometry-dedup-v4/`。

### 實際 W8 TRAIN、輸入 fanout 與第二次凍結

兩模型已以同一組 **32 個 TRAIN prompts × 16 decode** 產生固定完整 FFN W8A8 routes，32 題跨六類任務，保留全部 prompt 文字；prefill 分離、沒有把 test 或 teacher quality 資料拿來 fit。G4 archive 是 `[32,16,40,6]`，Deep 是 `[32,16,26,6]`；全部 completed masks 為 true。獨立重建 **1,332,582** 個 routing records，並檢查跨 batch 同一 weight／expert 的 W8 bytes 與 F32 row scales 相同。這是 canonical-only TRAIN capture，不是 paired placement speed／quality 結果。`w8-format-training/compiler_inputs.frozen.json` 的 SHA256 為 `49563291e385e5c58ee23a21c8c8714aa0a6f007066206030916d1b7fe42c4c1`。

v5／v6 每份有 132 layer/cost cases、12 個 global maps；geometry aliases 只 fit 一次，使用相同 512 fit／64 selection TRAIN 位置。v5 每 job 的 ONFI input 傳一次，v6 保守付四份完整 input 的 ONFI 成本；**PCIe common/private input 都仍只 upload 一次**。四份不是已證 AiF 實際需要，而是 per-plane input broadcast 尚未知的敏感度。每個 job 仍只使用一個完整 input context。

| W8 TRAIN model | ONFI copies | Command | Cyclic MoE＋shared ms | Whole ms | 額外改善 |
|---|---:|---:|---:|---:|---:|
| G4 | 1 | 250 ns | 23.412580 | 23.217778 | 0.8320% |
| G4 | 4 | 250 ns | 24.053338 | 24.014731 | 0.1605% |
| G4 | 1／4 | 5 μs | 27.533587／28.092034 | 相同 | 0% |
| Deep | 1 | 250 ns | 52.356016 | 50.391603 | 3.7520% |
| Deep | 4 | 250 ns | 53.451503 | 51.448331 | 3.7476% |
| Deep | 1 | 5 μs | 55.165789 | 54.009974 | 2.0952% |
| Deep | 4 | 5 μs | 56.181866 | 55.002679 | 2.0989% |

這些是訓練選配置的數字。G4 低命令成本的 whole 原始 allocated LSB blocks 為 **6.0 GiB**，cyclic 為 **5.90625 GiB**，須另補同容量控制；Deep whole 為 **14.0625 GiB**，小於 cyclic 的 14.25／14.8125 GiB。不得把較弱 phase-group baseline 的約 4% 當作最新 strong-cyclic 幅度。

獨立審核兩 campaign 共 **24 maps、1,143,424 native jobs** 的 logical／array／global 地址、單 block、row coverage、nonoverlap、四種 byte ledger、aligned selection costs 與 TRAIN provenance 全通過。Deep 的 26 MoE＋shared maps **沒有 raw0 dense prefix**；另以已核 dimensions 的固定 equal16 完整 rows 計算相同 prefix，inputcopy1／command250 ns 時為 **1.852224 ms**。完整 Deep FFN service 比較必須對每 family 每 token 相同加一次；attention／router 仍不包含。

`ordinary-read-compiler/W8_selected_plans.frozen.json`（SHA256 `00507ff56f9197fb18511ad58303919214c03cba6f0c1440f6733ffc42944d00`）已先封存兩份 summary／protocol／authoritative source snapshot／TRAIN manifest，再取得第二批 heldout。v5 首次 source snapshot 曾以 basename 使兩個 `decode.routes.npz` 名稱碰撞；原 TRAIN archives／protocol hashes 未變，已保留 erratum 並改為 SHA-prefix snapshot names，anchor 只指向修正的 authoritative manifest。

第二批為固定 revision 的 OpenBookQA main/test 8 題及 BoolQ validation 8 題，seed20261104、依 ID hash 固定選樣，保留完整問題／選項或 passage／question；答案另存而不送入 prompt。Manifest SHA256 `ccfabf3b3e1a4293f2d0354cdea289b7d94197fc6733253601ca97448dabcd8f`，獨立 acquisition check 已驗 anchor → protocol → payload 時序。兩模型、兩種低成本 selected whole-row maps 的 CPU 完整 FFN paired campaign 共 **64 paired prompts** 全通過。G4 每 view **1,610,997,760** FFN F32 scalars、27,295,744 logits；Deep 每 view **2,198,611,200** scalars、27,852,800 logits。完整 canonical／placed streams 與跨 profile streams 逐 byte 相同，routes／tokens 一致；不能將三條 comparison 重算為三倍獨立輸出。G4／Deep 分別保留 1,870／1,982 prefill tokens 與各 256 decode，沒有挑除案例。見 `selected-plan-numeric/second-heldout/validation_artifact_index.json`。

成本分析事先凍結三 families × 兩 command costs × 兩 input-copy scenarios，仍只用 TRAIN-selected plans。全部 **202,752 costs／LB、792 traces（675,372 nodes）** 由不同已封存 engine 完整重算，另有 28 份 Python scheduler traces；24 組模型統計、256 每題統計、32 dataset 統計吻合，沒有 shared／dense prefix 漏計或重計。

| 新 heldout model | ONFI copies | Command | Cyclic MoE＋shared ms | Whole ms | 加固定 dense 後完整 FFN 改善 |
|---|---:|---:|---:|---:|---:|
| Deep | 1 | 250 ns | 52.357222 | 50.603891 | 3.2344% |
| Deep | 1 | 5 μs | 55.196110 | 54.122153 | 1.8779% |
| Deep | 4 | 250 ns | 53.458011 | 51.655093 | 3.2575% |
| Deep | 4 | 5 μs | 56.214576 | 55.127055 | 1.8677% |
| G4 | 1 | 250 ns | 24.510604 | 23.863417 | 2.6404% |
| G4 | 4 | 250 ns | 25.136775 | 24.991046 | 0.5797% |
| G4 | 1／4 | 5 μs | 28.796653／29.364461 | 相同 | 0% |

Deep 四情境的固定 dense prefix 各為 **1.852224／1.994236／1.888128／2.013724 ms**，最後一欄對每 family 都相同加一次後才算百分比。G4 input1 有 3／256 個 token 變慢；input4 有 **66／256** 個變慢，但兩者的全部 16 個 prompt means 都較低。Deep 每個情境的 256 tokens／16 prompt means 都較低。這些是 16 題描述性數據，不是 256 個獨立任務、原生 PE 性能或完整模型時間。見 `ordinary-read-compiler/second-heldout-replay-v2/`、`independent/second-heldout-v2/notes.json`；第一份未執行 v1 analysis freeze 的 API-key preflight 失敗紀錄亦保留，未讀新 routes 就改為 v2，沒有改分析選樣。

容量強控制進一步發現：逐 layer 共同把 GU／Down／shared 的 chip c 與 c+8 對調，保持 job array 順序與 ONFI channel 不變，是整個 resource graph 的同構，與 query 無關。固定每半 layers swap 的 route-blind pack，G4 cyclic 由 504 降為 **496 block bundles（5.8125 GiB）**，whole 仍 **512 bundles（6.0 GiB）**；速度收益不能稱同容量 Pareto 改善。Deep maps 的 BFD reservation 不變。24 maps／1,143,424 jobs 已重核，121 traces 同構、242 成本／LB 比較一致；23／24 maps 達此 swap8 族的有效 volume 下界，另有一份 Deep disjoint packing gap 保留。不是所有任意 chip permutations 的全域 packing 最佳證明。見 `capacity-symmetry-control/`。

另外以不匯入 compiler／DP 的 round-robin row fill 重建 simple balanced-wave control，對 v4／v5／v6 的 **606／606 selected winners**（354 positive cases）都能用既有 cyclic／disjoint profiles＋simple balanced-wave profile 精確覆蓋；canonical labels、相對 Down rotation 與全部 routed rowcounts 逐項相同。這不證明簡單 menu 在任意形狀永遠足夠，但已否定目前性能需要 DP frontier／branch-and-bound 才能得到的說法。見 `simple-wave-control/summary.json`。

Backend 的 input dimension guards 之外，獨立反例另發現 imported logical `waves` 若與 native array 不同，可能讓地址 lowerer 少算 reservation。已要求 logical／array 各 job 全欄位一致、row ranges 完整且 column_start=0，保留反例與修正前後紀錄。正常 compiler 產生的 plans 原先即一致。新版 3,456 筆 input1／4 costs、18 份完整 trace 與獨立引擎吻合；13 個 malformed guards 均拒絕。所有這些仍是軟體契約檢查，不是 PE／NAND 測量。

### 任意排程仍需支付的工作量證據

另外對先前三組 frozen Qwen3.8 controls，在全部 5,519 外部 tokens、48 層逐一計算必要 chip 工作量。即使放寬成每 layer/query 可以免費挑 g16、train-group、overlap 三者中最小的下界，strict native-job lower bound 合計仍為 **83.619525 ms**；whole-row 的已建模排程為 **76.834621 ms**，低 **8.1140%**。5,519 個 token 都低於這個對照 relaxation。

為避免把 post-compute drain 的保守 executor 當成原生唯一時序，另報允許 job 內 read/output streaming 的 **83.332165 ms** 下界，以及完全只保留同 jobs 必讀 NAND chip 工作量的 **82.254721 ms** 下界；whole modeled schedule 仍分別低 **7.7972%／6.5894%**，每個 token 均成立。若只對 whole 額外加前述獨立 EPYC copy P99 **1.243 ms**，仍低 NAND-only relaxation 約 **5.0783%**；這是明列情境，不是跨平台 copy worst-case。

這個證據的正確主張是：**只重排這三組凍結 native jobs 的執行順序，不能完全抹去本次配置收益**。它不約束其他 cyclic widths、其他 expert labels、retile/coalescing、額外副本或不同原生指令，也不把 whole 的時間升格成硬體量測。256 個獨立 native diagnostics 的 bounds／舊成本精確吻合。見 `qwen38-placement/schedule-independent-bounds/`。

### 第二個完整 FFN 數值參考與下一輪資料

DeepSeek V2 Lite 已完成完整固定 W8A8 CPU 參考：raw0 dense 的三個 projections 與 raw1–26 的 routed/shared 六種 projections，共 **159 weights**；shared I=2816 的 down 只計一次。兩個 TRAIN prompts 的 82／176 prefill、各 4 decode，全部 layer/token coverage 通過；**261,318,400** 個 placed row outputs 的 I64／跨 run F32 檢查，以及 **1,024,000** logits、routes／生成 tokens 完全一致。8 threads 執行 147.46 秒不是 flash 性能值。

5,067 個 cache entries 的原 GGUF offsets／types／dimensions 已獨立核對，711 個預定 Q8_0 row samples 的 row scales 逐位相同。沒有新模型下載或 quantizer tuning，仍未證 native AiF arithmetic／品質。見 `deepseek-full-ffn-integer-reference/`。接下來另以兩模型共用的 32 個 balanced TRAIN prompts 收集實際 W8 routes，保持 test 分離；這是 training data，不冒稱新 held-out。

Deep 的品質補充現已完成：沿用第一批凍結的 8 個 WikiText raw-text windows，每個取 Deep tokenizer 的首 128 tokens（兩 control 相同，非跨模型相同 tokenization），共 **1,016** next-token losses。原 Q8 GGUF mean NLL **2.7979040626 → 2.8171404759**，Δ **+0.01923641334 nats**；PPL **16.4102159155 → 16.7289453893（+1.9423%）**，8／8 windows 都較差。Top1 agreement 943／1016，26,624 router sites ordered／set agreement 為 53.6771%／80.4500%。這是固定 W8 representation 自身的品質差異，不是 row placement 所致；未調量化／配置來修飾結果，也不是新 blind test、任務 accuracy 或 BF16 品質。全部 159 weights／128 tokens 覆蓋、輸出 finite，首 W8 window 的獨立 context 重跑 13,107,200 logits 零 bit difference。見 `deepseek-teacher-quality/`。

### 新原生機制的有界篩選與補充 prior

GU 跨 expert 合併利用相同 full-H input，但對當前 whole-row Qwen 幾何，每 query/chip 全部 selected GU tails 合計小於一個 wave，已有 **153,600** query-chip cases 的零 NAND-saving 證據。Strict WL／row 共同起點為 128 rows；三種固定 ordering＋exact interval coalescing 在前三層／9,600 queries 都未勝過 singleton。32-row 分支還放寬了 native start/window accounting，只是額外樂觀 relaxation；即使弱 g16 有改善也仍輸 whole-row，不能拿來宣稱現有硬體收益。見 `gu-coalescing/`。

Shared row-job 小副本的樂觀 oracle 亦未構成突破：Qwen 全 48 層將 shared jobs 複製至 paired owners，額外 useful weights 僅全 FFN 的 **0.1949%**，但未付完整 DAG／selector 費用前，最多約 **1.504%** mean headroom。前三層四 owners 版本額外 **0.5848%**，對 incumbent 的樂觀 mean 空間不超過 **2.835%**。Generic replicated-job allocation 與 phase-independent whole-view 兩個 work-oracle objectives 相差約 **0.775 個百分點**；兩者 DAG／ready 損失可能不同，這不是實際 policy speed 差的嚴格上界。同 inventory selector 與 generic RDA 精確匹配，沒有新演算法效果。見 `native-mechanism-next/`。

Intra-expert hidden-chunk pipeline 還有目前數值契約的全域依賴：down 的 A8 scale 由完整 activation vector 的 maxabs 決定；剩餘 GU 未就緒時不能免費產生相同 full-vector A8 bytes。改 block quantization／static calibration 會改模型數值，不能混入純實體配置收益。

新查核的 [TileLens（2026）](https://arxiv.org/html/2607.04031v1) 已直接研究大粒度 HBF／LGMS 的 tile-major layout、對齊、shape 與 stragglers；其 DSL／TMA／prefetch 修改不能當成 AiF 免費能力，但已排除「首次用 page-granularity 2D layout」的說法。[MegaFlux（2026）](https://arxiv.org/html/2610.00671v1) 已有 runtime replica placement、tile-aligned token-block assignment 與 FC1／FC2 readiness；它的 token blocks 不能冒充 NAND weight-row aliases，亦不能假設無寫入成本的 flash 動態移動。

後續在固定 V4 plans 加入命令內 per-wave read → drain → next-read 的敏感度，保留全 expert GU 完成／完整 A8 的 barrier；6,912 筆成本、108 份 real traces 與 96 個 tiny independent checks 通過。Qwen／Deep 相對收益近似不變，G4 若同時付四份 plane inputs，舊固定 plans 的收益會翻負。這需要可暫停／backpressure 與一個 wave 的 output FIFO，容量／native 合約未公開，**不是確認的新硬體能力**；v6 只使用較保守整 job drain。

純 ONFI 拓樸也已停止擴展：共同重排 GU／Down／shared chips 只改 channel 配對，不改 chip-context 工作量。250 ns 下，全層 TRAIN 的必要 chip-work 給 whole rows 的平均收益上界為 G4 **0.383–0.514%**、Deep **0.109–0.215%**。5 μs 時 32 個共同 permutations 的前三層最佳收益均低於 0.006%；這排除本次純 channel 配對假說，沒有排除任意獨立 phase mapping／row geometry。見 `topology-placement/report.md`。

最新直接 prior 審核特別確認 [PIMCOMP 官方 repo](https://github.com/sunxt99/PIMCOMP-NN) 已有逐 instruction／ONNX Runtime verifier 與成本模擬連接，不能把「compiler＋verification＋cost model」本身作首次。可守的具體問題仍是 ordinary NAND 完整 row／full input 的整數 work-depth、單 block lowering 與地址／數值契約；尚未建立足夠的性能與新穎性。能力矩陣與可推翻檢查在 `candidate-novelty-audit/audit.md`。

### 任意 rows 的界、coupled 幾何與 routed 邊界 views

以實際 W8 TRAIN 的 G4／Deep layer0 作有限時 integer rows／waves／active MILP（各 90 秒），移除同一 template／固定 inventory 限制，仍固定 shared、完整 one-copy rows 與單 block。兩者停在 root relaxation，沒有找到新 geometry incumbent，不能報成 no-go／optimality。有效 mean lower bounds G4 **510.7185 μs**、Deep **1860.126 μs**；同 geometry 的 joint/free-label control 已使 Deep native **1940.600 → 1915.947 μs（1.270%）**，新 rows 仍未超過它。見 `arbitrary-row-headroom/`。

G4 的已知 coupled construction 可把 GU25waves 與 Down12waves 配成每 expert joined **5 chips×3＋11×2 waves**，同時補償固定 shared 的不均勻工作。在 W8 TRAIN512／L0，native mean 由強 joint/free-label **581.531 → 562.102 μs**；舊 aligned-omit 的同 budget 控制已達 **562.180 μs**，所以放寬任意 omit 的額外 mean 僅 **0.014%**。兩個 seeds 的 TRAIN P99 family envelope 為 **604.200 → 580.350 μs（3.947%）**，並非新 heldout／full-model tail 或全域最優。這說明 known coupled geometry 值得作更強控制；不能因匹配本研究先前的 aligned candidate 就宣稱整個方向已被已發表 prior 完全否定。見 `g4-coupled-mask-screen/`。

新分支用少量完整 row copies 提供同 expert 的兩個 prefixes。對 A／B rowcount vectors，在可自行構造 B logical row ownership 時，新增 rows 的下界 `½Σ|A−B|` 可由每 chip common core＋較長 view 專用 suffix 達成；若 B 的 logical row sets 已預先固定，則需另外算集合交集，該數字可能不可達。Q38 GU 只交換 101 與 75／76 rows 的一對 chips，複製 **26／25 rows（66,560／64,000B）**，max-view union 多四 pages／一 wave，但每次 view 仍 16 commands／51waves。不是把整個向量 rotate8；後者是不同多-edge control。

G4 實體數值構造已用既存 chip binaries，21 個完整 projection W8／scales 均與 W8 TRAIN fingerprints 相同，沒有重新量化。固定六個 experts＋shared、兩 fixtures 的全部 **2⁶ views**，共 128 cases／1,792 matrix phases；**3,145,728 F32 scalars、2,424,832 I64 rows** 零差異。258 copied rows 為 396,288B；兩 view 各 1,176 pages／294 waves，union 為 1,200 pages／300 waves。這是 v4 L0 geometry 的 construction proof，**非 v5／v6 的 L0 selected winner**。短40-row prefix的頁面會再讀到 packet 的兩個完整 rows 與 1,024B partial row，輸出須 drain／drop，未知 native reset／row-separability／output ABI 沒有被此 CPU 證據解決。見 `routed-boundary-view-numeric/`、`route-boundary-view-audit/`。

效能歸因必須含同 A／B choices 的零副本 static repack、相同 useful／page／wave budgets 的 whole-GU backups、frequency／coactivation placement 與 exact 小型 view selector。初步 Q38 layer0 TRAIN64 中，放開 39 個 edge 候選後 dynamic mean **1569.207 μs**，同 A/B static 為 **1575.785 μs**，副本本身只額外 **0.417%**；同 32MiB page budget 的十份完整 GU backup 強控制達 **1567.323 μs**。即使讓 packet 達 exact mandatory lower bound **1566.895 μs**，相對該 control 也只餘約 **0.027%** mean 頭空間，selector／host costs 尚未加。因此停止擴此 mean-speed 分支，而不把對原凍結配置的 1.67% 都算成副本新效果。完整結果封存後以 `routed-boundary-views/` 為準。

### 三層共同配置的強控制與工作量反例

`shared-counterbalanced/` 已完成兩模型前三層、inputcopy1／4、實際 W8 TRAIN512、command250ns 的共同配置。G4 最小 overlap 為 5，joined waves 是 11×2＋5×3；Deep 同為 5，joined 是 11×8＋5×9。第一版對 equal-row cyclic 的 mean 增幅 G4 3.0595／1.8309%、Deep 4.2758／4.1375%，但它漏了 joint balanced-wave menu，且 generic 只用單 phase swaps、coupled 則用雙 phase swaps，因此不得作最終歸因。

v2 共同載入所有 cyclic／disjoint／whole-row phase templates、16 relative rotations、相同 shared geometry pool，加入 single＋paired phase labels／local／global moves。Unrestricted 控制也保留 v1 coupled incumbent，並把 inherited 與 incremental 搜尋成本分開。按 TRAIN mean 選各 layer plans，再將相同 512 個 query indices 的三層 service 相加：

| 模型／ONFI copies | Joint simple-menu mean ms | Unrestricted paired mean ms | Mean 改善 | P99 改善 |
|---|---:|---:|---:|---:|
| G4／1 | 1.714969 | 1.690822 | 1.4080% | 1.8085% |
| G4／4 | 1.783105 | 1.761182 | 1.2295% | 2.4412% |
| Deep／1 | 5.771215 | 5.752082 | 0.3315% | 0.4658% |
| Deep／4 | 5.895378 | 5.877861 | 0.2971% | 0.4983% |

Deep 保留在 `v2/`；G4 首次因 literal `whole_rows` family assertion 在 fit 前停止，原因是原 compiler 去重將等價 orbit 存在另一 family，並非缺幾何。原失敗 source／protocol 留存，另 `v2-fix1/` 只修為完整 menu 及 incumbent orbit 檢查，搜尋／預算不變。獨立核對 432 unique artifacts、184,320 query-work、828,672 native jobs 及 72 Python／C++ costs 全通過，completion SHA `dfd8255b22f619cdc4b3e91f1f7a676e931371e46b982b3553aefa369c0a4b19`。未讀新 heldout，未跨過原定 2% mean／5% tail 擴展門檻，因此停止此版本的全層展開；這是研究資源分配門檻，不是使用者要求放棄。見 `shared-counterbalanced/independent-v2/report.md`。

正文／官方 code 的 closest-prior 查核確認，HDA-MoE 的 aggregate fraction 不分 GU／Down rowcounts，Stratum 的 joint ownership 依靠 Down input-column partial reduction；沒有找到相同 ordinary-NAND 三 category 的直接證據，但不能主張 first joint mapping 或優先權證明。查核範圍與能力差異在 `coupled-closest-prior/notes.json`。

另一項獨立審核構造了五張同一 TRAIN route／shared 的合法 G4 maps，所有 maps 每 expert、每 chip 的 joined wave 向量相同。兩張 joined work 都為 **573,058 ns**，native **575,356／575,122 ns**；另一對 work **573,058／573,734 ns**，native **575,946／575,872 ns**，顯示較小 work 仍可更慢。完整 rows／addresses／node traces 與 ReferenceGraph 精確一致。最小 overlap 僅平衡逐 expert wave histogram；真正的 service 排序還受 GU readiness、commands、channel／PCIe 與完整 activation barrier 影響。見 `coupled-work-contract-audit/`，此處基本 packing 不宣稱新 theorem。

Root exact-native refinement 的 cache／family 契約已由獨立 checker 通過 100 proposals、106 次 full recompute、12 份 ReferenceGraph；ordered route tuples 必須保留，不能排序 tuple 後當 native cache key。接下來只作兩個 TRAIN-only pilot cases、相同 proposal 上限的強化控制，不把不同 valid native-query／CPU 費用稱作同計算預算。另一有界候選是每 expert 不同 GU／Down 整數 geometry 的 joint native 搜尋；舊 phase-only scalar mixed screen 尚不能證明它無效，elastic／fractional expert sharding 本身則已有 prior。

這兩個 pilot 現已封存。`native-joint-refinement/pilot-v1/` 八 fits／兩 cases／兩 families／兩 seeds／2,048 proposals，七份 mean 未變，Deep unrestricted 第二 seed 僅省 **9.62890625 ns**；獨立重算 8,192 costs＋8,192 lower bounds、完整地址及四筆額外 ReferenceGraph，與 root 十六筆相符。Completion SHA `dbdfd17e117df695fc767d9228958d790b01593be1c3641c5e3a519b0074b11f`，停止擴展；有限 search 的 no-op 不等同全域最佳。

`native-mixed-geometry/pilot-v1/` 對同 G4 L0/input1、Deep L0/input4，完整 phase menu 按 row multiset 去重（G4 GU17／Down16、Deep20／18），增加單／雙 phase geometry 換法、其餘保持相同 native objective／2,048 proposals／兩 seeds／temp500。四 fits 雖暫時接受了 8／6／1／3 次 geometry moves，最後 mean-selected best 都回到原 geometry；相對最強 fixed-native mean 只省 **G4 0.000127%、Deep 0.003182%**，P99 0／0.014302%。獨立 2,048 costs、12 Python traces、7,808 final jobs／one-copy／one-block／完整地址通過；completion SHA `51e95fe363ce3cf5c37a4bd126aa43d0eca47be56340fd200810386a63f04888`。未展開 uniform／phasewise 補充控制或新資料，僅支持這個有限 menu／預算未見有效 mixed headroom。

`compact-coupled-solver/` 另以標準 CP-SAT 將固定 G4 geometry 改寫為每 expert／chip 的兩個 bool：X=GUhigh∩Downhigh、O=GUhigh∩Down0，逐 expert 的 X/O 數為5／4、互斥，完整 occupancy 精確 **57,864＋28,086X−676O ns**。總 solver 148.58 秒；19-wave60秒 UNKNOWN，20-wave為 SAT（feasibility回OPTIMAL不是native最佳），mean-work最後 FEASIBLE、保留有效鬆弛 bound，沒有全域近優證書。188 distinct masks／189 records 的96,768 costs、567完整ReferenceGraph及584,577 nodes owner／因果檢查皆通過；最佳native **562,670.570 → 561,099.020 ns（0.279302%）**，TRAIN單層P99改善4.00291%，容量皆155,713,536B useful／9,760pages／2,440waves／1,824jobs／16bundles192MiB。未過擴展門檻，停止此支；無新 solver／native optimum 主張。

### Ordinary GU 的共享 tile／重疊 read views

下一候選利用所有 routed／shared GU 都接收相同完整 H input。係數單 copy 密排，logical expert 的 read view 前後擴到完整 native WL，輸出中的其他 expert rows 實際讀出並丟棄；相鄰 coactive intervals 可 union 成一次 ordinary GEMV，之後依 row IDs scatter，Down 仍有完整 GU／完整 A8 barrier 及不同 private input，不融合。

Deep C2048 每page8rows、每wave32rows，所以 uniform176 的半wave尾部與舊 Qwen C2560 slack 證書不同。正確 parity 的一對176rows共11waves，兩singleton共12；奇數起點的pair仍12waves，不能套理想ceil忽略prefix。441個row-ID cases及484公式已核每selected row只consume一次、fringe全付、stored rows唯一及跨6144row／192WL block必拆command；readviews重疊不等於儲存copies。176singleton原本96waves/expert，必須比較192/160的88wave stronggeometry；最大權重 matching＋pair/block ordering是標準必要控制，greedy path不足。

`ordinary-gu-tile-sharing/work-screen-v1/` 已freeze並做同512TRAIN的160cases＋1,920獨立DP比對。250ns下，強native到最好的coalesced必要工作量 bound只餘input1 **0.32937%**、input4 **0.87119%**，best皆88wave free，GU NANDwaves未降，只少commands；uniform176即使matching追回部分padding仍輸該強控制。低成本停止擴展。5μs的樂觀bound剩約5.31／5.57%，但原native與原bound本來就有約96μs差距，相同layout singleton→DP的chipwork只降約0.76／1.12%，因此只展開有界fullDAG檢查。

`native-dag-v1/` 的完整 ordinary-read DAG 已封存，protocol SHA `8b2cd13c8f30e63380c1fb157af34353b607744ed0153d7100e144a3dd223e40`。同一16候選、兩priorities及原配置／tight singleton／DP三種執行均使用TRAIN512。最強 original mean 到最佳 DP：input1 **2.077521125→2.056312023ms（1.020885%）**，input4 **2.116102992→2.087236031ms（1.364157%）**；各自 mean-selected map 的P99反而變差 **0.498487%／0.760449%**。同一 layout 的 tight singleton→DP mean改善1.190245%／1.523584%，不足以支持超過最強不同layout控制的主張。16,384 costs與舊 engine精確橋接，192 traces／211,652 nodes核對；獨立union-DP另外查655,360 query-chip、1,310,720 service cases，全過。最大 fused view為1,472rows／7 owners；整個 fused command drain後才讓 owners GUready，沒有假定早完成。未達原定擴展門檻，不展開全層或新heldout。見 `ordinary-gu-tile-sharing/{native-dag-v1/screen.json,independent/report.md}`。

完整native rowstream仍要求controller的union/scatter mapping；現有aif.h只是metadata timing模型，未證FIFO／maxR／maxpagecount／output scaling。沒有假設新buffer或早ready，fused command必須全部drain後才能供每個參與expert的GUready。BigMoMo、RecFlash、PIMCOMP及TileLens已有co-loading、同頁合併、共input及overfetch概念；目前查核沒找到完全相同NAND半wave多expert GEMV，不代表首次。契約與read-depth在 `ordinary-gu-tile-sharing/{contract-audit,closest-prior}/`。

後續限定為條件式 FIFO 敏感度，另凍結 source／protocol。AiF§4.4描述controller在運算期間poll FIFO取出依序產生的結果，§4.1描述拼接各chip subvectors；若在controller收齊一個logical expert完整GU向量，就做一次logical full-vector DMA及該原CID的一次final CQE，可免除半向量host通知。但prefix最終ECC／成功狀態、row order、same-input union、FIFO容量／flow及暫存仍需核對。Deep的6個routed＋1 shared GU valid outputs在F32假設下為90,112B＝88KiB，只是buffer需求上限，不是AiF已公開的controller空餘RAM。所有families必須同給streaming能力與相同明列overhead，零額外成本只作樂觀界；未校準CID／DMA constants僅作敏感度。見 `ordinary-gu-tile-sharing/streaming-capability-audit/`。

這項敏感度現已封存為no-go：`conditional-stream-v1/` protocol SHA `3ec23e82f837d07c7d0eafc1cd8fbce0f8834735fc8d1f9fa871d01be342cdea`。同fullyaligned候選、兩priorities、原始／singleton／DP及四種transport，共240records／122,880 costs；30,720 legacy bridge與480Python traces／1,096,936 nodes精確核對。優先的完整expert DMA狀態，input1／4 best mean **2.053079953／2.083197488ms**，相對先前已知最佳legacy DP只再省 **0.157178%／0.193488%**，P99改善0.114665%／1.404241%；相對同layout、同streaming singleton的mean改善1.235345%／1.461201%，仍未過gate。零額外成本已不足，因此未啟動非零CQE／DMA敏感度，沒有全層、heldout或existing-device主張。

FIFO計數、無producer pause、直到最後channel drain才釋放input／chip、完整GU後才activation與合法streaming context LB均由trace核對。模型FIFO需求128B/chip，並非已確認硬體容量／poll時序。所選兩map的controller payload occupancy為28,160／38,784B，**不是實際需配置的完整buffer容量**；七CID完整buffer靜態reservation上界仍88KiB，還不含input／metadata。已公開output interface不足的限制未因模擬通過而消失。見 `conditional-stream-v1/independent/report.md`，completion SHA `d30c01ff06b9cf11f54cf3a9b87a2bfdb1d5a5ce6f112381325bcee4f2110880`；full-vector契約補充在相鄰 `streaming-full-vector-addendum/`。

### G4 Down 完整 tile copies 的停止證據

`g4-down-tile-copy-screen/` 使用前述CP控制最佳G4 L0 map（native mean561,099.019531ns），每個64 routed experts複製一個128-row×512-column完整Down tile，共4MiB／64waves。每expert可選A原12個128-rowshards，或B將一個source tile移到已有0／128rows的destination；B是private完整512維input、common core＋copy suffix，沒有partial dots。32個route-blind layouts與兩輪有限edge coordinate search後，所選layout對每TRAIN query完整枚舉2^6＝64個views，best chip-work只降 **1.391372%**。先前host＋work鬆弛曾給2.0084%樂觀空間，已由固定executor較強界取代：16個首GU commands在第一個28μs read完成前，以 `2000+c×250ns` 依序開始；所以finish至少是 `2000+max_c(Wc+c×250)`。對全部512×64 choices求最小後，mean有效界552,652.207031ns，native mean改善至多 **1.505405%**。

該界只適用目前dispatch順序及所選copy layout，不是任意scheduler／副本配置的不可能性。三份完整ReferenceGraph另核首command時序，32,768 masks獨立重算；一copy完整row IDs、單block、不重疊、GU/shared地址不動及useful/page/reserved bytes均過。51個copies落在已有128row destinations，13個落在0row處；即使4MiB副本填入allocator holes，仍完整計入新增stored bytes。未過2%資源分配門檻，未做重型同容量controls、全層或heldout，封存decision SHA與21-file completion ledger供重現。

### 無page-buffer重用的多請求配置篩選

`multi-request-onecopy-screen/` 只將同decode step、不同TRAIN prompts合成B2／4／8 ready cohorts，並不聲稱真實arrivals或queueing SLO。每request有獨立input、完整weight sensing與privateDown；不從舊replay-oracle借用多vector SRAM／page-buffer重用。兩模型各512單請求regression、完整rows與B2／B8獨立resource時間線通過。B8普通multiplexing本身對G4／Deep改善3.290%／1.225%，不能算新配置效益；同強map的可行排程已距aggregate必要LB僅0.084%／0.050%。所有g≤4及既有all-width／不等row控制都未勝出，因此停止新serving simulator。對useful-NAND的7.074%／4.529%鬆界仍混合其他geometry選項，不代表可達。PlaceMoE的主要背景是訓練，HDA已有hybrid placement／runtime dispatch，不據此主張本screen的新穎性。

### Deep GU 全部邊界的成對副本

`deep-gu-paired-replica-screen/` 另檢查單edge實驗沒有覆蓋的八edge／expert家族。Deep GU原本八chips192rows、八chips160rows；把每個high最後32完整rows複製到配對low的160row core後，兩端各有160／192row prefix。每edge只執行一處extra，所有GU仍88waves／16jobs，整個H input每job重新載入，Down／shared及完整GU→A8 barrier保持。64experts共512份64KiB copies＝32MiB。K6 selected experts的perfect matchings union為6regular，標準Euler orientation可讓每chip extra indegree3，routedGU33waves/chip；這是基本graph／batch-code原理，非新定理。

六個routeblind matchings在TRAIN512的exact minimax joined-work flow，best work只降 **0.807331%**；目前fixedexecutor首dispatch修正後，native mean收益至多 **1.074095%**。獨立SciPy Edmonds-Karp核6,144個optima／12,288次feasibility，3,072個獨立Euler配置及98,304個prefix masks的row-ID／scale identity全過。再去掉所有packet endpoint限制的樂觀native上限也只有 **1.074513%**，best matching離其mean界僅8.19ns；放寬每chip至多6extras再多13.32ns，仍不足2%。因此停止native、重型同容量控制與heldout，不主張所有replication不可能。

容量也不能只報32MiB useful：baseline coeff544.5MiB／35,108pages／8,777waves，resident576.5MiB／37,156pages／9,289waves。直接BFD要63blocks，另對chip11給有效3block修復後共 **62blocks／744MiB**，達固定perchip inventory的ceil下界，相對baseline48blocks／576MiB增加 **168MiB allocated LSB**。這是static repack，未假裝舊地址不動；3072 stored row slots／expert是2816原rows＋256副本，不能當獨立logical rows。若gate有過，同32MiB必須比較5份完整GU backups＋剩餘4.5MiB／72waves有用copies及相同copy graph的一般RDA flow；六份完整GU為33MiB overbudget控制。現因必要界不足，沒有擬合這些重型控制。Root completion SHA `543477d98619cb3b939864f02b4c8fac96d498642796ab7afa35a9ca3111350e`，契約／容量地址／row-ID與獨立演算法在 `independent-contract/`。

### G4 雙span與逐source prefix的配置

`g4-elastic-prefix-screen/` 改查同一expert以共core＋附加rows支持g8各128rows（24waves／8jobs）或既有g16（25waves／16jobs）。前者少work／commands，後者單job較淺。runtime完整expert選A/B，所有GU到齊後才形成原本的完整Down A8 input；沒有新增PE、input contexts、MAC、partial dots或改routing。

`work-v1/` 只讓A的八sources取自B九個82/83row高chips，每expert九個omit選項只供離線物化，runtime只有一A＋原B。17routeblind layouts及兩輪有限coordinate search後，host＋必要work給mean／P99改善上限 **0.829203%／0.970757%**；更緊的合法sorted-first-start界是0.724746%／0.884654%。未跑native或heldout。同固定prefix下的兩個zero-copy static controls都保留B；32,768 masks、row IDs、地址與容量已獨立核對。所選每層copy為35,638,272B＝33.9873MiB useful，新增512waves／2048pages＝32MiB physical reservation；它填入原padding，所以兩種增量不必相等。L0含固定Down/shared的allocated blocks由16增至20（192→240MiB），這是每層數字，不能假裝完整40層只需34MiB。

`free-span-bound/` 再移除A的位置／副本容量限制，對每query枚舉64個A/B expert subsets；n個A各需八個不同chips上的85,274ns jobs。Complete-bipartite容量條件精確給necessary work；32,768個值另由64,512次Edmonds-Karp＋distinct-chip row sums核對。Mean／P99 native下界544,199.246／544,898ns，對B只給 **3.01191%／6.16453%** 樂觀收益上限，未宣稱可部署。此界允許每query改位置，僅用來判斷還值不值得搜尋固定A。

`work-v2-free-A/` 因此放寬A可取任意八chips；32個routeblind seeds＋四個coactivation matching初始配置，best兩個各兩輪source swaps，共16,384 proposals／807,936 oraclequeries。Matching平均1.33789個co-selected pairs/query，但這是TRAIN初始化，不是新測試結果。最佳mean work下降0.88029%，host＋work的native mean／P99收益上限 **1.502387%／0.873587%**（更緊界1.409331%／0.855141%），仍未過gate；兩static controls保留B。副本增為每層41.0376MiB useful／625waves＝39.0625MiB reservation、22blocks，不能與v1稱同budget。獨立1,245,184個query-mask值／row aliases／地址皆過；未跑native／heldout。有限兩starts不能證明所有固定A位置無效。

`work-v3-independent-prefix/` 則在v1的固定33wave resident image內，每source可獨立讀短core或128row長prefix。七個sources的append各含一個40row low-destination packet與5/6個high-destination rows；special source持有high-destination的45/46row suffix。Low job在對應source長讀時取消；high job依剩餘logical rows讀兩waves、一wave或取消。這不是免費增加九種A位置：每expert仍只有一個固定image，只增加已有頁面prefix的合法讀法。每個query-expert可算出多於1024個meaningful rows，host只consume每個original row一次；全部duplicate／fringe輸出仍計費。65,536個expert-view cases已逐row驗證，meaningful computed最多1062，page-emitted最多1066，unique consumed恆1024。

四個固定pairings的連續LP仍留下4.67–4.99% mean樂觀空間，但它允許fractional wave choices，不能當收益。Root另用分母10^9的rational dual與任意精度整數核對全部2,048個box-dual certificates，與原保守floor相差至多1ns；`integer-dual-audit/`不匯入producer／solver。兩起點、兩pass的48bit greedy找到可行work545,895.816ns，對B work改善1.73948%；完整native後，**561,099.020→551,417.348ns（1.725484%）**，P99 **580,695.14→577,208ns（0.600511%）**。491queries較快、4相同、17較慢，最差慢2,450ns；不能只報正收益樣本。All-A endpoint反而650,493.359ns，顯示淺job／最少work取捨沒有被免除。

這個native結果尚未計selector或新軟體開銷，仍未過2% mean／5% tail的資源分配門檻，沒有開始大型同容量controls或heldout。原B512 costs／LB逐值bridge、q0／255／511×三endpoint共九份Python逐node時間線及3,072個query-expert ownership maps通過。Alias-aware lowerer將computed rowcounts與unique logical1024分開，原backend guard／C++未改。32個預先固定TRAIN indices的有界整數CP診斷已全部OPTIMAL，greedy到整數work最優平均只差784.0625ns、最大2,326ns，沒有case超過3μs；沒有把這32筆換入512筆native結果，也不代表native或全配置家族最優。見`cp32-integer-audit/`，protocol SHA `a5c0b00f64fd7859d96a366cb84a888a2ecdab3c7f4bae9215343513ef0e4244`。

Tutel已有共享參數layout的runtime degree選擇，但官方程式仍做weight all-gather；AnchorTP保留共同intervals後Transfer／Reload；重疊fragmentation與replicated declustering也已有直接prior。相同A/B幾何的完整兩份backup應與共prefix有相同service，**prefix共享的直接效益是省programmed容量，latency來自adaptive geometry**。公平性須分報兩種控制：逐項限制useful／page／wave／block的vector budget，及允許填滿相同allocated blocks的控制，不能互相替代。前者512waves容21份24wave完整GU backups＋8waves殘餘；後者可用候選block內其餘空間存更多完整備份，不能因超過前者programmed預算就排除。見 `closest-prior/notes.json`、`independent-contract/`及各版本`independent/`。

### G4 Down成對合併與GU／Down聯合副本選擇

`g4-down-paired-merge-screen/` 將每expert原12個128-row Down shards配成六對，destination保留128row core並附加source的128row副本；source原件仍存。每對可選兩個128row jobs，或略過source、一次讀destination256rows。六pairs獨立，全部仍12waves、1536個unique outputs及完整512維private input，沒有partial dot或新input context。64experts共384waves／24MiB新增copies。32張凍結的routeblind／heuristic graphs先作LP及unit-flow鬆弛；flow丟棄source額外506ns只作有效下界，重建可行配置時完整加回。

只有graph14進native，TRAIN512平均 **561,099.020→550,165.613ns（1.948570%）**，P99 **580,695.14→576,531.66ns（0.716982%）**；未計selector。B512 costs／LB、六份獨立Python完整traces、3,072 query-expert ownership及4,096 physical row views通過。沒有換測其他graphs或進入heldout。見`native-graph14/`；completion SHA `42ce746beff38d861d731cab8efbad22e8443e8379017d205a33264eca88211d`。

`g4-joint-prefix-replica-screen/` 再固定前述GUv3 layout3與Down graph14，聯合選48個GU bits／6個AND及36個Down bits；四starts、兩種既定順序、各兩passes的八paths為一個固定selector。凍結後的512筆native平均 **549,021.773ns（對B改善2.152427%）**，P99 **578,797.5ns（0.326788%）**。對較強Down-only只再省1,143.840ns、約0.207908%；P99反而比兩個單支都差，不能將兩支收益相加。476筆較快、36筆退步，最差慢29,118ns。這仍是TRAIN單層、零selector軟體成本的結果，不能當投稿突破。預選CP32僅6筆OPTIMAL／26筆FEASIBLE，平均work再省803ns、最大1,692ns，未替換native choices。

聯合影像新增60,804,096B＝57.9873MiB useful copies、896waves＝56MiB reservation；resident useful206.4873MiB／programmed208.5MiB，因perchip194–226waves而需 **32blocks＝384MiB**，原B為16blocks＝192MiB。獨立契約核16,384GU／4,096Down views、241,600prefix／address checks；native audit逐值重建512costs及LB、7,864,320個row-owner entries，q0／255／511的B＋joint共5,496nodes全符。Contract completion SHA `28ed893c92c3dd87272d6166f519136086ffb3982f8812266f0ba791a1711eb3`，native completion SHA `71ed46b962b87a48f950154b01560a030c26e5e198d706f7f3e52957f24792d8`。

容量強控制現已有具體地址及獨立證書：base＋Down14 maximal prefix images在每chip≤189waves，可全部放block0；另存完整A1 GU image即可逐job重現joint的GU長／短選擇，再存互補A2仍只需block1的192waves。因此同32blocks可容更多完整GU備份。完整A1＋Down14是272.5MiB programmed，A1＋A2＋Down14是368.5MiB；它們超過候選programmed預算，卻未超過其allocated容量。Down24MiB在此依賴core重用／靜態重排；若原B地址全保留且另存完整merged Down影像，額外Down應為48MiB。

`same-allocated-witness/`及`same-allocated-independent/`逐一核所有stored images、intervals／blocks、完整columns與row multiplicity，包含shared GU2048×1536及Down1536×1024各一次。全部512queries／75,594個ordered jobs及9,742,575個computed prefix row IDs均重現joint，只換long-source讀取地址；pages／waves／output數與resources圖完全相同，不需重新模擬即有服務等價。短source的fringe資料可能由copy變padding，但全部被drop，有用rows與成本不變。這排除本次單層allocated容量的優勢主張，不排除更嚴格programmed budgets的節省；未聲稱A2新選擇已量得更快。Witness audit SHA `c6a7e6fcefe427e41f4e8e302f45f11b872e038f46c8eec3e9b332bb78cb8efb`，獨立completion SHA `19620a62a9cd5b109e3cbe648fb826ba0f36ca3ead6e7d69d818825c272a50b6`。同一層block數不能乘40當完整模型最佳packing。

`scheduler-priority-control/`另公平給B／allA／GUgreedy／Down14／joint五個凍結plans和bits相同的GU-first／Down-first選擇，共5,120 native costs；全部df0 costs／LB與舊結果逐值相符，十份Python逐node traces共9,120nodes通過。五者皆GU-first較快；Down-first mean反而慢2.84–4.79%，allA為650.493→668.945μs，joint為549.022→575.219μs，P99也都更差。這排除只換該兩種priority即可救回本輪方案的解釋，不證明任意scheduler最佳。Completion SHA `dc685ed16340fa5cd67ce7776a85f46339ec5641844555aed18d5500384e6d08`。因此停止本joint版本的productionselector／newfits／nativeCP與heldout，研究目標仍未完成。

### 尚未取得全文的直接prior補查

`fulltext-gap-followup/` 仍未合法取得DIAMoND與Cambricon-FlexLLM全文。FlexLLM官方Crossref完整abstract已明寫co-activation neuron-inspired weight reordering及sparsity-aware dynamic partitioning，不能忽略其直接相關性；row geometry、GU／Down ownership、fusion prefix、copies及runtime mapping仍未核實。官方OA PDF單次回403，作者頁／公開repo未找到全文；timeout／DNS失敗不當作不存在證據。DIAMoND官方摘要的mask-based mapping也保留未知範圍，不因拿不到全文而排除重疊。

先前第二批 heldout 的圖已輸出至 `ordinary-read-compiler/second-heldout-replay-v2/figures/same_format_heldout.{png,pdf,svg}`，含 caption／provenance hashes；它不包含本節 TRAIN-only 的後續控制。

### 新一輪硬體規模與既有配置能力的查核

後續篩選維持使用者指定的現有硬體／實體配置範圍，不新增MAC、decoder、independent-plane inputs、input contexts或免費page-buffer重用。`chip-count-regime-prior/`以[SM2264官方brief](https://www.siliconmotion.com/download/3bn/a/SM2264_PB_EN.pdf)及[Cosmos官方code](https://github.com/Cosmos-OpenSSD/Cosmos-plus-OpenSSD)核對8channels×8CE／ways的storage拓樸。但CE／target／LUN／die不等同獨立GEMV資源；8／16／32／64個AiF-like owners只作各自重複相同PE／ECC／input資源的規模敏感度，並非商用SSD已有該運算能力。每個相同N內才比較配置，所有shared channels／serialized commands／input reload照付；增加chips的速度不能算placement收益。

[CRAFT v2（MLSys2026）](https://arxiv.org/html/2603.28768v2)已有逐層replay收益、有限副本預算的MCKP／DP及跨device容量配置，故不能把全模型按層分copy budget視為首次。[官方artifact](https://github.com/Accelsnow/CRAFT_core) inspected commit `968ecd9`中的CLI使用離散budget／benefit thresholds，不能直接叫exact paper DP復現；`input_dist // logcnt`也不能用於單token的離散work守恆native oracle。這是adapter限制，不據此否定其GPU批次結果。

[BALANCE官方slides](https://www.aspdac.com/aspdac2026/archive/pdf/4B-2.pdf)第12頁已明列跨die／plane配置重要SEUs及可靠lower pages；[KVNAND正文](https://arxiv.org/html/2512.03608v1)已有die grouping、KV page／head映射及相應DSE。前者不只ECC，後者的新FMAC／buffers不能當成AiF既有能力。AiF的CSB／MSB一般資料頁也不能在沒有原文證據下視作同GEMV的慢權重tier，因single-sensing LSB釋出的page latches正用於input／GEMV。Primary notes SHA `4950c4f91eb05cbe5ac9993a6bf4cd83ee7e26f62b0e087b6736c3d0bd835e94`，詳細讀取範圍及未取得正文／標準之處留於source ledger。

`chip-count-regime-screen/stage-a/`已完成兩模型×8／16／32／64owners、inputcopy1／4的routeblind profile篩選，含all widths、disjoint、精確work/depth及simplebalancedwave。94次geometry DP、996個去重row profiles由root的獨立inverse-capacity DP全部核對：對j個jobs、w個waves直接最大化`Σfloor(k_i*65536/C)`，再找容量≥R的最小w，不匯入producer／solver／routes。Completion SHA `a13420736a9489e8b5896ab6618082dc0fb6a11b4e58758dca26287e5d2f1c09`。

後續`stage-b/`在已凍結27個(profile,inputcopy)組合上公平跑GF／DF，54records／27,648 costs及54份Python逐node traces通過；16chip G4與Deep最強舊anchor逐值相符。此表只列inputcopy1有限候選的TRAIN單層模擬值，**不同N不可互算placement gain**：

| 模型 | Owners | 有限候選最佳mean μs | P99 μs | 所選profile |
|---|---:|---:|---:|---|
| G4 | 16 | 561.099 | 580.695 | 既有CP強anchor |
| G4 | 32 | 326.998 | 379.589 | finite proxy |
| G4 | 64 | 208.816 | 245.602 | minimum-depth |
| Deep | 16 | 1918.057 | 1958.913 | 舊強anchor的input1 addendum |
| Deep | 32 | 1034.693 | 1114.131 | finite proxy |
| Deep | 64 | 565.910 | 623.928 | finite proxy |

32／64owners的aggregate必要界比這些有限候選低，尚未證明可達或新配置收益；Deep8因初篩較小未跑native，不能稱所有排程無頭空間。原16chip anchors需跨另一inputcopy重播的不足，另用獨立frozen addendum補2,048costs而不改stageB seal：G4同CPB rows/input4為584.389μs，Deep同seed06 rows/input1為表內1918.057μs，兩者皆勝較弱routeblind maps，DF亦公平重播。接下來只作G4 M32一個standardCP／hypergraph強控制，保持GU25個1wave、Down12個1wave及shared完整rows，只改chip labels，不加copies；這是檢查routeblind基準的不足，不把使用CP-SAT當新方法。

### M32的學習配置、強排列控制與剩餘尾端差距

`g4-M32-coupled-cp/`已完成。每expert有25個GU一wave jobs（24×41＋1×40rows）、12個Down128row一wave jobs；minoverlap5用X5／O7互斥表示。Input1的精確chip-owned work為`29102＋28762X−340O ns`，input4為`31406＋29530X−1876O`，包含command、完整input reload、NAND及完整page輸出drain，**不是native makespan**。實際frozen shared的wave histogram為12×3＋17×2＋3×1；9×3＋23×2僅是另可構造的minoverlap shared，未當成此次凍結布局。

30s peak feasibility＋90s mean-work CP，標準OR-Tools 9.15／8workers／seed20261114。Peak10找到可行解（feasibility的OPTIMAL不代表native最優），mean為FEASIBLE；七個保存incumbents全部512queries皆10wave peak。9,216次native costs後，所選map/input1/GF平均 **299,342.484ns**、P99 **300,795ns**；input4同map、不重fit為320,484.436ns／324,183ns。對未學labels的有限候選為8.457%／20.758%與7.828%／18.909%，**不能將這些值歸因新方法**。

`independent-m32-final/`核8maps×512queries×32chips、兩fanout逐chip work、全rows／shared／單副本與單block地址；2424 resident jobs／9764pages／2441waves、useful148.5MiB相同，32blocks384MiB。54個PythonGraph，其中18份q0完整traces共25,578nodes逐欄一致，另36筆q255／511成本相同。任意此coupled profile每chip最多131WL＜192，不藉額外allocated容量。Completion SHA `daae1ec1a5f2f8e70a7f59b694c82ebaf4f6cf6bd031d0b2e98c67983102f0f4`。

隨後`g4-M32-cyclic-rotation-cp/`給相同30＋90s／workers／seed，學64experts的32種完整GU＋Down共同rotations，含40row位置的exact orbit；原stageB source可作有效hint。8,192次native costs後最強learned rotation為 **301,215.363ns／P99 328,789ns**。自由masks相對此基準只餘 **0.621774% mean／8.514275% P99**，233筆較快、8筆相同、271筆較慢，最差慢6,389ns。因此原先約9%mean大部屬既有排列學習，不能保留為新貢獻。

此rotation CP的peak10為UNKNOWN、mean為FEASIBLE；不能由有限求解推論cyclic orbit達不到10waves。獨立核8maps完整orbit／rows／shared／addresses及48個PythonGraph，completion SHA `2d88dd04acf74578b24f01719d60b57fb8e84ea9700aba150f18257fef043087`。原資源gate是2% mean **或**5% P99，殘餘tail仍過gate；目前決策是先補GU／Down可各自rotation的更強控制，不開始全層／heldout，也不宣稱整支已被量化gate否決。

`g4-M32-independent-rotation-cp/`已補完整phase-independent rotations：64×2×32 onehot，GU／Down各自保留exact row orbit及40row位置，不強制每expert overlap5，因此直接用G／D服務係數與wavecounts；shared固定。相同30＋90s／8workers／seed、8,192costs，最佳mean **301,168.140625ns**、P99 **328,789ns**，比共同rotation只省47.223ns。Free相對這個更強控制仍為 **0.606192% mean／8.514275% P99**；227筆較快、6相同、279較慢，最大退步7,002ns。保留menu中best-P99也為328,789ns，非只拿mean-selected壞tail作對照。Peak10 UNKNOWN、mean FEASIBLE，未證自由masks有全域最優或不可替代優勢。8maps／512逐chip work與one-copy地址、48個PythonGraph通過，completion SHA `1edde6dad4341466208ec802a76f70f490efdc25bfb34b2a79d3bf9e723dc412`。

`free-position-bound/`另移除逐expert固定chip位置，求本295wave geometry的必要界。對每chip枚舉6個selected experts的X／O數、全chip總X30／O42，用獨立set-DP證明owned-work最小max精確為input1 **287,752ns**、input4 **298,504ns**；T可行／T−1不可行。以6regular bipartite multigraph的六個perfect matchings重建六個expert的X5／O7與完整rows，這是標準edge-coloring應用，不主張新定理或可部署的每query免費搬移。

若NAND峰值≤10，295waves使至少7chips有10waves；七個heavy chips的首次acquire都經同一250ns command resource，至少一個初次開始不早於host2000＋6×250。若峰值≥11，單NAND已超過該界。因此排程無關native下界為 **291,252／302,004ns**；對目前free-map mean只餘至多2.703%／5.766%頭空間，不代表可達。獨立g／d=0..6的3,136狀態／1,074heavy枚舉確認此界也適用同25GU／12Down profile的phase-independent masks，不侷限X5／O7。主completion SHA `376b04696fcc3df66e6c8cf927d79c15cd79857ed96d743ba71d50dae96bb1b5`，addendum SHA `b61ef5b87d60be0d2cf7c1cdd5e13537de498a55cc15147fa7ae6d320fb61ba9`。

Tail仍過OR探索門檻，故下一個有界步驟為prospective新資料pilot。`m32-prospective-tail-validation/`先凍三個TRAIN mean-selected圖（freeinc4／commoninc0／phaseindinc2）與GF，primary固定free對phaseindependent、input1/cmd250，input4僅同圖敏感度。計畫32新prompts×16decode、兩sources各16，使用原完整40layer G4 W8 canonical collector取得真routing，再只比較L0服務；不選勝者或再fit。以32prompt clusters配對bootstrap2000draw／seed20261115，所有queries、負例與source各別結果都保留。Protocol SHA `911ecbdb77dcd18ccc880870f70a6a231c0a67730bfac7ac9144eea061e9b516`，取得新payload前的frozen manifest SHA `36eb40d994a2163a9d066bd73777510f1b6bff1ede05bf894f686f551c7ef895`。獨立preflight已核三圖與collector契約；尚未把canonical capture說成新三圖paired full-model數值證據。

`numeric-materialization/`已把原16banks完整W8重新配置成三圖各32banks。1312原allocation、130logical matrices與195個functional weight/scale hashes全核；每圖155,713,536 useful bytes、9764pages／2441waves、32blocks，一份完整權重而無copies。四組既定`x0/x1 × TRAINq0/q511`fixtures重算GU I64／F32、SwiGLU、完整A8、private Down與ordered alpha mix，共12placed cases／168phases／571,476scalars逐bit相同。Producer只讀原banks／scales，未重新量化或讀GGUF／新test；completion SHA `ad23a4242cfb174ee99e7413c3482a4a8d545d83d61318e9ec3fb25970919de1`。

獨立checker另由原`execution_views`重建所有rows／scales，不匯入producer helper，以I32乘積＋I64 reduce重算全部四fixtures；7272jobs的完整pages、零padding／partial tails與row地址、96banks fullSHA均核過。每圖169,115個完整emitted rows中167,424 valid、1691 fringe rows計算後丟棄，沒有造新scale或early partial flush。全部中間值／輸出亦吻合producer arrays；completion SHA `11d24374ecb0f1f1d556ae073b69c7eeeb2ab2afce40019a232961155d2cc752`。兩者都是離線CPU／實體byte證據，並非新32題placed full-model、原生output/reset ABI或裝置量測。

Root另補`m32-tail-directed-rotation-control/`，避免只用mean objective搜尋的cyclic基準造成tail歸因偏差。兩個已封mean winners為starts，peak violations／top-six work為兩個既定objective，固定四passes單phase32rotations＋兩passes joint1024rotations，共四fits；shared／geometry不變、零copies，總work不得超過原independent anchor的102%。保留兩anchors及四finalmaps，公平重播GF／DF與input1／4，共12,288costs及72份完整Python逐node traces，兩anchors input1逐值橋接原結果。

該補充TRAIN tail-selected GF為`independent_top6_work`：mean **300,756.683594ns**、P99 **328,368.44ns**；原free-map相對它只餘 **0.470214% mean／8.397104% P99**。這是有限local work-proxy control，不能證明cyclic不可能追上。Protocol SHA `e83272f01659ed70a570472bd200c03e3e8eff50f4a6013e2db5956f43d405e2`、completion SHA `1c5ee8f816eae0f0da86df03ea76a497f85f77bcc4e05325df9591fd66ae4d42`。此root補充在新prompt已由獨立capture agent取得、但root尚未讀其routes／costs時封存，原prospective primary三圖與分析完全不改；不能寫成全團隊皆未接觸新payload或原預先primary。

### 其他既定能力的低成本停止證據

`existing-capability-mechanism-screen/slim-train-union-bound/`明確採[SLIM §5.4](https://arxiv.org/html/2507.09201v1)已公開的batched read-once／GEMM能力，沒有偷借AiF page-buffer重用。實際W8 TRAIN同step B4／B8 cohorts含shared一次讀／B次MAC：G4 union平均19.609／31.375，Deep20.039／31.234；理想batching讀量約省31–48%，但16dies的強g16已對384cohorts同時達page與MAC均分floor，**這兩種資源的placement-only work改善上限為零**。SRAM64KiB是PE總量，input／output分配未公開；不能說buffer／整體native時間下界全達。8,320個地址descriptors與520個完整operator coverage核過，沒有啟動新serving simulator。Completion SHA `07d5b7c20188e2e4eed039d677f82600eeab36a515d3a876f4e8b8867207a9da`。

`physical-fault-placement-design/`則保留一個條件式fault-tail witness：拆開同時會用的expert domains可降低多expert同時故障，卻可能提高any-failure率，不能稱可靠度全面支配。QCFlash實測只涵蓋3個QLC WL，nativeLSB observed raw errors6／10／15及缺Vth40／96／106cells，不是AiF be-enc／ECCLite後的silent故障資料；原artifact以已知state補缺測也不能當實測無誤。固定16chip CPB所有expert都碰所有chip／block，coarse-domain declustering無空間。未過資料與native契約gate，不啟動LLM bitflip campaign。Completion SHA `b24f9c893cd83db6599530f578dd06e735853a715fab1e27cf26125e1bfa06e8`。

## 投稿前仍須解決的核心缺口

需要一個在現有原生運算下可執行、相對足夠強的 declustering／replication 基準仍有實質收益的機制，而非只超過原本的 power-of-two 分組。若走最壞情況證書，必須清楚區別既有 batch-code theorem 的直接應用與真正新增的問題或結果。

還需有完整模型時間、真實控制及重組成本、實體 NAND/PE 或足以校準的量測。現有 routing、成本模型、凍結紀錄、地址證書與獨立稽核可支持下一輪實驗；尚不能當成已完成的硬體或論文成品。
