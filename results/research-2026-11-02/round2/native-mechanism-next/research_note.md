目前沒有找到 2–3 個可誠實稱為新機制的方向。剩下一個值得作為強控制的 Qwen3.8 小副本實驗：複製 always-active shared expert 的既有完整 native jobs，允許每個 job 在固定 owners 間選擇。它完全沿用原本完整列、input、output 與單一 chip context；概念及 selector 屬於 replicated declustering／EPLB，尚不是新演算法。

GU coalescing 已另封存為負結果，見 `../gu-coalescing/research_note.md`。Strict 128-row/WL-aligned case 與 32-row 理想放寬分開；後者不能當 ordinary four-plane hardware 結果。本輪沒有新 placement fit、沒有開外部 workload、沒有下載模型或改動任何既有 source／plan。

所有數值都是 nominal I8/A8/4B 的合成 native-service 模型；routing 來自原 NVFP4 checkpoint。它們不是 AiF 實測速度，也不證明原 checkpoint 品質。

**小副本的樂觀上界。** 讀取目前 ordinary-read compiler 的 whole-row plans，完整 48 層、每層相同 64 個 training selection ordinals。所有候選固定 routed rows，保留原 job shape、pages、wave 數及 page-fringe output mode。每個完整 shared job 複製到 `chip xor 8`；不搬移執行時權重、不分割 dot、不新增 contexts。下表是忽略 ready 時間與其他資源衝突的必要 chip-context-work 下界，因此只限制對 incumbent 的最大可能收益。

| 全 48 層條件 | 額外 useful/full FFN payload | 平均 ms | P99 ms | 對 incumbent 樂觀平均空間 |
|---|---:|---:|---:|---:|
| 原 whole-row schedule | 0 | 76.566537 | 76.988951 | — |
| 原 one-copy work bound | 0 | 76.478388 | 76.899050 | 0.1151% |
| 只複製 shared GU，two owners | 0.1300% | 75.829195 | 76.209950 | 0.9630% |
| 只複製 shared down，two owners | 0.0650% | 75.461098 | 75.767107 | 1.4438% |
| shared 兩 phase 都複製，two owners | 0.1949% | 75.414839 | 75.729314 | 1.5042% |
| shared 任意 owner、fractional、免 copy 限制 | 未計容量的 relaxation | 72.614603 | 72.641231 | 5.1614% |
| 所有 jobs 任意 owner、fractional | 未計容量的 relaxation | 72.613296 | 72.613296 | 5.1631% |

Pair copy useful bytes 為 235,929,600；programmed pages／wave reservations 為 242,221,056 bytes；各 chip 另開新 block 的保守容量上界為 327,155,712 bytes。這三者不可混稱同容量。所有 48 層的 selected token 索引相同，故 paired sum 有定義；它是 training selection population，不是獨立測試。Free shared 與 free-all 下界接近，只說明固定 job inventory 的大部分 relaxed 平均空間可由 shared 吸收，**不證明有限小 budget 時 shared copies 永遠支配 routed copies**。

另在前三層相同 64 queries 枚舉 four-owner inventory：每個 shared job 放在 `{j,j+4,j+8,j+12} mod 16`，額外 3 個 shared-expert-equivalent copies，即全 FFN payload 的 0.5848%。每個四-chip group 有 8 個完整 jobs，精確枚舉 `4^8` assignments；四個 group 獨立求 minmax。

| 前三層條件 | 平均 ms | P99 ms | 對 incumbent 樂觀平均空間 |
|---|---:|---:|---:|
| 原 schedule | 4.784581 | 4.858651 | — |
| 每 query 選一個完整 shared expert view | 4.728161 | 4.796624 | 1.1792% |
| GU/down 可分別選完整 view | 4.686018 | 4.740273 | 2.0600% |
| 每個完整 native job 獨立選 owner：generic RDA | 4.648950 | 4.694419 | 2.8347% |

兩個 work-oracle objectives 的差，不是兩個實際 schedule 差的嚴格上界；它們的 readiness 損失可能不同。Quad 對 incumbent 的總樂觀空間約 45.2 μs/layer，pair 全 48 層平均約 24.0 μs/layer；真正 selector、native DAG 或更強 static shared refit 都可能吃掉它。

在完全相同 copies、完整 native jobs 與 owner choices 下，候選 selector 與 generic weighted indivisible-item RDA 是同一個問題。這不是因「形式上可表示」就否定所有 systems 貢獻，而是這份固定 inventory 的兩個 selectors **逐項相同**。同 backend 的強控制不能故意用未加 native costs 的 token-count solver。[Selective Replicated Declustering (2009)](https://repository.bilkent.edu.tr/bitstreams/938fed97-cc40-48c5-bbd8-01da42e95ae2/download) 已選擇副本與 owners；[TOS 2013](https://faculty.cse.louisville.edu/nihat/papers/TOS2013.pdf) 已處理 weighted retrieval、device offsets 與 delays。真正尚未直接解決的是 native future release／A8 barrier／chip-context 鎖定的完整執行成本，不能由 retrieval bound 取代。

下一個最小可反駁實驗只需前三層相同 training queries：對原 one-copy、同 budget whole-expert view、phase-independent views、generic native-job RDA，跑相同完整 native scheduler、相同 routing information／priority，實際付 selector、command、input reload、channel、PCIe、activation/combine 成本；保留完整 shared GU 才能產生 A8 down 輸入的依賴。補同 training budget 的 static shared row/owner refit，以及 globally allocated copy budget。若 generic RDA 與候選相同，不能聲稱新 selector；若完整執行吃光上界，就封存此模型個案。這是 portability／強基線測試的理由，不是已成立投稿點。

**Row aliases 尚缺原生構造。** Metadata alias 本身不會把 weights 移到別的 chip。Qwen GU `C=2560` 的 page-aligned row step 是 32，four-plane-WL-aligned step 是 128；down `C=640` 分別是 128、512。把 GU high/low 的 26 rows 或 down 的 102 rows 當成可搬移的單 wave 都不合法：兩者均已超過 64 KiB。即使 page-aligned，任意初始 plane 的 prefix skip／完整結果停止條件仍未公開。先構造一份有兩個可執行 views 的係數地址圖、真實副本 bytes、native commands 及完整 dot 數值證書，再值得做 route 搜尋。重疊 range 的動態切分本身已有[2016 direct prior](https://www.jstage.jst.go.jp/article/imt/11/0/11_1/_pdf)。本輪只算 granularity，沒有把缺失 contract 補成免費能力。

**Chunk pipeline 在現有數值 reference 下停止。** Down A8 scale 是整個原 activation vector 的 `maxabs/127`。相同已完成前綴 `[1]`，若未完成尾端是 `[1]`，該前綴量化為 127；尾端是 `[127]` 則量化為 1。因此 early chunk 在剩餘 GU 完成前無法產生相同 down A8 bytes。Block/static scale 是格式變更。即使 native output ABI 能合法做矩形加總，也不會消除此 input-scale barrier。另 fixed native-work schedule 與 work bound 只有 0.1151% 空間；這不涵蓋同時改 geometry／NAND work 的候選。[PowerInfer-2 §4.4](https://arxiv.org/html/2406.06282v1#S4.SS4) 已有 cluster-level matrix-barrier pipeline；[MegaFlux §2–3/A](https://arxiv.org/html/2610.00671v1) 已有 FC1/FC2 readiness 與 runtime replicas，但 GPU block scaling 不等於 AiF contract。

**其餘方向沒有新的可用硬體證據。** [AiF §4.3–5.2](https://doi.org/10.1145/3695053.3731073) 只建立 LSB weight compute；PB 空間存在不能推出 retained-input command 或多 contexts。舊 noise／disturb／interference no-go 仍適用，沒有取得同 be-enc LSB 的 PPA-dependent calibration，不造 spatial variation。新查 [FLINT §4](https://arxiv.org/html/2608.25062v1) 的 burst controller／額外 phantom plane 是硬體設計，不能免費加到 AiF。Route-oblivious Hall／batch-code guarantees 也已有 compound-study 直接控制；若轉作 native phase-DAG guarantee，需明確的新定量命題及可否證 counterexample。

Compiler 新穎性還需補 [TileLens (2026) §3–5/6.7](https://arxiv.org/html/2607.04031v1)：它已聯合考慮 tile-major page layout、shape 與 straggler 成本；SW 改 CuTe/TMA descriptors，HW 改 TMA／prefetch。搜尋所得同名 Triton visualization repositories 不是這份論文 artifact。本輪未驗證 paper-specific 官方 GitHub。Cambricon-FlexLLM 全文仍未取得，不能自稱已排除其 reordering／shape 方法。

封存資料見 `mechanism_review.json`、兩份 oracle JSON/NPZ、`alias_and_quantization_gates.json`、`independent_bound_audit.json`、`completion.json`。獨立 audit 改讀 native NPZ jobs 而非 plan JSON，核對 787,968 jobs、9,216 query-policy cases；quad 首 query 的每個 group 另由 scalar 65,536-assignment 枚舉核對。
