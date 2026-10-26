# MoE expert 放在 flash 哪裡，為什麼會影響速度？

這份報告整理 AiF 論文和目前專案的實作，希望先把問題講清楚，再看數字。整理日期為 **2026-10-23**；使用的是 **2026-10-19 保存的實驗結果**。這次有重新執行程式測試，沒有重跑模型推論或啟動 VM。

目前最值得記住的是：**在本專案採用的串流排程下，同樣一批 experts，分到最多 chip 不一定最快。** 切得太細，每顆 chip 只讀一小段，就得重新開始下一次讀取；集中在較少 chips，可以把連續讀取拉長，也讓其他 experts 在別組 chips 工作。但如果同時被選中的 experts 都擠在同一組，就得排隊。

這個專案已經把這個取捨做成可重現的時序模型，接上真實 MoE routing，並完成 NVMeVirt 重播。它目前研究的「位置差異」，主要是**放在哪些 chips、如何分頁、是否連續、會不會互搶資源**。某個實體 wordline 天生比較慢、某個 die 的電路特性不同，還不在現有模型裡。

---

## 1. 先認識一個 expert 到底做了什麼

MoE 可以想成每層準備了一群功能不同的工作者。輸入進來後，router 決定這次請哪些 experts 處理。例如 Granite 每層有 32 個 routed experts，每個 token 只選 8 個。

選擇的是誰，會隨輸入、層數和 token 改變。如果 expert 的權重固定放在 flash 裡，那麼 router 的選擇也同時決定了「這次哪些 flash 資源會忙」。因此，容量平均分配不代表每次執行都平均。

本專案研究的每個 expert 有三個矩陣：

| 名稱 | 接收什麼 | 做完交給誰 |
| --- | --- | --- |
| gate | 這一層的輸入向量 x | 經 activation，與 up 的結果逐元素相乘 |
| up | 同一份輸入向量 x | 與 gate 的結果一起形成中間向量 |
| down | 該 expert 自己的中間向量 | 還原到模型的 hidden dimension，之後加權合併 |

順序大致是：

```text
這一層的輸入 x
    │
    ├─ router 選出 experts
    │
    └─ 對每個被選中的 expert：
          gate(x) ─ activation ─┐
                              × ─ down ─┐
          up(x) ───────────────┘         │
                                       └─ 依 router 權重合併 → 下一層
```

**gate 和 up 共用輸入；down 必須等自己的中間向量出來。** 所以不能把所有矩陣當成同時可以開跑，也不能把下一層、下一個 token 的工作任意重疊。

若 hidden dimension 是 H、expert 的 intermediate dimension 是 I，三個矩陣分別是 `I×H`、`I×H`、`H×I`。以本專案的 INT8 假設，每個權重 1 byte，因此一個 projection 是 `H×I bytes`，整個 expert 是三倍。接下來提到的 expert 大小，都是用這個方式計算，並非 GGUF 檔案的實際大小。

依據：[矩陣維度與排程實作](/home/ericyu/expert_flash_experiment/scripts/aif_model.py:72)、[routing 擷取方式](/home/ericyu/expert_flash_experiment/scripts/llama_route_trace.cpp:25)。

## 2. AiF 論文想解決的，是權重搬不夠快

模型放不進裝置 DRAM 時，可以把部分權重放在 SSD。但在逐 token 的 decode 階段，每次都要讀大量權重，SSD 對外的頻寬容易成為瓶頸。

AiF 的做法是把矩陣乘向量，也就是 GEMV，移到 flash chip 裡。主機送入較小的輸入向量，chip 在內部讀權重、計算，再回傳結果。大量權重便不必每次都搬回主機。

![圖 1：一般 SSD offloading 與 AiF 的資料路徑](/home/ericyu/expert_flash_experiment/doc/assets/01_aif_datapath.png)

這裡的「in-flash」有一個容易誤解的地方：**AiF 是在 flash chip 內加上數位運算邏輯。** 資料仍要從 NAND array 讀進 page buffer，再經過錯誤修正，最後送入 INT8 乘法器和加法樹；不是直接讓儲存 cell 以類比方式完成乘法。

單純把運算搬進去還不夠，NAND 本身也要讀得夠快。論文另外做了兩個改造：

| 改造 | 白話說法 | 和位置的關係 |
| --- | --- | --- |
| **be-enc** | 調整 TLC 的編碼，讓 LSB page 只需一次 sensing；模型權重放在這類較快、較可靠的 pages | 要區分資料放在 LSB、CSB 還是 MSB；本專案只對 LSB 權重建模 |
| **cr-read** | 連續讀同一 block 時，沿用電壓狀態，省掉反覆完整充放電的時間 | 需要同一 block 內不中斷的連續讀取；跨 block 或重新開始一條 stream，收益就會縮短 |

TLC 每個 cell 可存三個 bits，對應 LSB／CSB／MSB。AiF 把權重放在 LSB，並不表示其餘兩類 pages 一定報廢；論文允許放其他一般資料。

論文 Table 2 的 LSB 普通讀取，在 be-enc 下為 **28 μs**；後續 cr-read 為 **9.7 μs**。AiF 每 chip 的內部頻寬上限是 **6.4 GB/s**，16 chips 合計 **102.4 GB/s**。這是 chip 內部處理權重的頻寬，並非 PCIe 的速度。對外 PCIe 仍是 8 GB/s，每條 channel 為 2 GB/s。

原論文也有測 MoE，使用的是 **Mixtral-8x7B**，不是只有 dense models。它在 Figure 16 報告 Mixtral 的 AiF 效能為 6.2 tokens/s，但那是論文整套系統、記憶體配置與排程下的模擬結果。原論文沒有對小 expert 的 chip group 大小、共同啟用碰撞或不同固定排列做本專案這樣的比較。

論文結合真實 NAND 量測、電路模擬和 RTL 綜合估計來支持設計，再用修改過的 NVMeVirt 評估系統時間；尚未製成完整 AiF SSD 來跑這組 LLM。本專案則依論文公開資訊建立可檢查的部分時序模型，沒有完整重現原作者所有硬體和 host 排程。

因此，這個專案接著問的是：**當 expert 小很多，而且每次只挑其中幾個時，AiF 原本把矩陣分到所有 chips 的方式，還適不適合？**

論文依據：§4.1／Figure 6（PDF 第 5 頁）、§4.2（第 6 頁）、§4.3–4.4（第 7–9 頁）、Table 2（第 10 頁）、Table 3／Figure 16（第 11 頁）。[本機完整 PDF](</home/ericyu/expert_flash_experiment/AiF - Accelerating On-Device LLM Inference Using In-Flash Processing.pdf>)；書目為 [Lee 等人，ISCA 2025，AiF](https://doi.org/10.1145/3695053.3731073)，亦可在[作者所屬大學的研究成果頁](https://research.knu.ac.kr/en/publications/aif-accelerating-on-device-llm-inference-using-in-flash-processin/)核對。

## 3. 這個專案怎麼表示 flash 的位置

![圖 2：flash 層級、chip 分組，以及 chip 內的讀取波次](/home/ericyu/expert_flash_experiment/doc/assets/02_flash_geometry.png)

可以從大到小看：SSD 有 channels，channel 下接 chips，每顆 chip 有四個 planes，plane 內再分 blocks 和 pages。這裡一個 LSB page 是 16 KiB。程式把四個 planes 同時各讀一頁稱為一個 read wave，最多處理 64 KiB。

目前比較 `g = 1、2、4、8、16`，g 代表**一個 expert 的矩陣分到幾顆 chips**，整台 SSD 的 chip 數始終是 16。

| g | 固定 groups 數 | 一個 expert 的權重放法 | 主要取捨 |
| ---: | ---: | --- | --- |
| 1 | 16 | 整個 expert 放一顆 chip | 連續讀取長，但一個 expert 只用到一顆 chip |
| 2 | 8 | 分到固定兩顆 chips | 更多 groups 可處理不同 experts |
| 4 | 4 | 分到固定四顆 chips | 在本輪 Granite 結果中較有利 |
| 8 | 2 | 分到固定八顆 chips | 單 expert 更快，但更容易和別的 expert 共用 group |
| 16 | 1 | 每個 expert 都分到所有 chips | 接近論文的全 chip row sharding；所有 experts 共用這批資源 |

分割時保留完整的輸出 rows：一顆 chip 負責一段輸出，每段仍讀取完整輸入向量。沒有把同一份權重複製到多個 groups，也沒有在 router 選完後臨時搬 expert。

原始排列採用 `group = expert ID mod (16/g)`。例如 g=4，expert 0、4、8 都在 group 0；expert 1、5、9 在 group 1。所有 expert 的 gate、up、down 都使用它所屬的 group。

chip 內則把某一條矩陣讀取 stream 的 pages 輪流放到四個 planes。**目前每條 stream 都獨占自己的整波位置**：最後一波就算只用了兩個 planes，也不把另外兩個位置分給另一個 expert。

這和「expert A 固定放 planes 0、1，expert B 固定放 planes 2、3，兩者一起執行」是不同設計；後者尚未實作。

另外，一般 NAND 還可能有 die 等層級，但本專案沒有獨立的 die 時序參數。channel 對應則是 `chip ID mod 8`，例如 chip 0 和 chip 8 共用 channel；向量傳輸會計入這項競爭。

依據：[chip group 與 row 分割](/home/ericyu/expert_flash_experiment/scripts/aif_model.py:48)、[四-plane wave 配置器](/home/ericyu/expert_flash_experiment/scripts/materialize_plan.py:27)、[channel 傳輸模型](/home/ericyu/expert_flash_experiment/scripts/overhead_model.py:16)。

## 4. 為什麼所有 chips 都忙，還可能比較慢

### 4.1 小 expert 被切得太碎，連續讀取的好處用不上

以 Granite 為例，一個 projection 是 `1024×512 = 512 KiB`。

![圖 3：Granite 的短讀取、plane 空位與 gate/up 串接](/home/ericyu/expert_flash_experiment/doc/assets/03_short_reads.png)

| 分法 | 每 chip 拿到的權重 | 16 KiB pages | 四-plane 讀取波次 | 單條 stream 的權重服務時間 |
| --- | ---: | ---: | ---: | ---: |
| g=16 | 32 KiB | 2 | 1 | 28 μs |
| g=8 | 64 KiB | 4 | 1 | 28 μs |
| g=4 | 128 KiB | 8 | 2 | 37.7 μs |

g=16 和 g=8 的單條 stream 都是 28 μs，但 g=16 的一波只有兩頁資料，四個 plane slots 有一半空著。而 g=8 可以填滿同一波。

g=4 的單條 stream 需要兩波，時間反而更久。不過它留下四個 groups，讓不同 experts 有機會同時工作。整層要等的是最忙的那組，不是單一 expert 的速度。這就是為什麼單 expert 變慢，整層卻可能更快。

專案依論文的讀取相位，採用以下模型：

```text
同一 plane、同一 block，連續讀 P 頁：
T_run = 28 + (P − 1) × 9.7 μs       （P > 0）
```

第一個 28 μs 已包含整段開始與結束所需的成本，不能在尾端再加一次。四個 planes 並行時取最慢 plane 的讀取時間，並用 6.4 GB/s/chip 限制串流速度；兩者取較大值，不把同一批權重重複算兩次傳輸。

這是專案從論文推導的計時方式，不是原論文明列的一條效能公式。

### 4.2 gate/up 串接，少付一次重新開始的成本

gate 和 up 使用同一份 x，可以把兩個矩陣上下接成一個 `2I×H` 連續矩陣。讀完後，把結果拆回 gate 和 up，再完成 activation 與逐元素相乘，之後才執行 down。

Granite 在 g=16 時，原本是兩條 32 KiB/chip 的 streams，各花一次 28 μs；串接後變成一條 64 KiB/chip stream，只需一次 28 μs，也剛好填滿四個 plane slots。沒有少算權重，只是改了實體連續版面。

程式有兩個容易混淆的選項：`fuse_gate_up` 只是把兩條 streams 放進同一個階段，仍各自收 cold-start 成本；**`stack_gate_up` 才會真的先串接矩陣，再計算 pages 和讀取波次**。目前主要結果用後者。

### 4.3 位置平均，不代表 router 每次都選得平均

![圖 4：同樣被選中的 experts，在不同固定排列下的 group 碰撞](/home/ericyu/expert_flash_experiment/doc/assets/04_expert_collisions.png)

圖中仍是同樣八個 experts。原本最擁擠的 group 要依序處理四個；換個固定位置後，每組只需處理兩個。每組總容量可以維持相同，只交換哪些 experts 佔用那些位置。

專案的 learned placement 就是利用訓練 routing，找出比較少碰撞的固定排列。每層、每個 g 最多嘗試 100 次 expert-slot 交換；若降低最忙 group 的累計負擔，或在同分時讓各組分布更平均，就保留交換。這是啟發式搜尋，不能稱為全域最佳解。

這裡的「學習」只調整擺放位置，**不訓練模型權重，也不改 router 選了哪些 experts**。

依據：[cold-start 計時](/home/ericyu/expert_flash_experiment/scripts/aif_model.py:24)、[gate/up 串接](/home/ericyu/expert_flash_experiment/scripts/aif_model.py:123)、[固定排列搜尋](/home/ericyu/expert_flash_experiment/scripts/optimize_placement.py:33)。

## 5. 從真實模型到結果，中間經過哪些步驟

```text
真實 llama.cpp 推論
  │ 擷取每個 decode token、每個 MoE layer 的 expert IDs
  ▼
固定 routing trace ＋ 從 GGUF 取得的矩陣維度
  │ 換算 INT8 權重，掃描 g、串接、固定排列
  ▼
Python flash 時序模型
  │ 加入命令、向量傳輸、host phase 成本
  ├─ 用兩組 prompts 學配置，第三組保留驗證
  ├─ 產生虛擬 chip／plane／block／wordline 地址計畫
  ▼
把每 chip 的需求數量編成 descriptor
  ▼
QEMU/KVM guest → 真正載入的 NVMeVirt module
  │ kernel 套用模型時間，延後命令完成
  ▼
核對 Python 預期、kernel counters 和實際 ioctl wall time
```

### 用了哪些模型

| 實際 trace 模型 | H／I | 每層 experts／選中數 | 納入的 routed 層數 | 每個 projection 的 INT8 大小 | routing records |
| --- | --- | --- | ---: | ---: | ---: |
| Granite-3.1-1B-A400M-Instruct | 1024／512 | 32／8 | 24 | 0.5 MiB | 2,304 |
| OLMoE-1B-7B-0924 | 2048／1024 | 64／8 | 16 | 2 MiB | 1,536 |
| DeepSeek-V2-Lite-Chat | 2048／1408 | 64／6 | 26 | 2.75 MiB | 2,496 |

每個模型使用儲存說明、程式設計、數學三組英文 prompts，各做 32 次 decode，共 96 次。表中的一筆 routing record 是「一個 token 在一個 routed layer 的 expert 選擇」，不是一個 expert，也不是一個 token。

DeepSeek 有 27 層 transformer，但第一層是 dense FFN，另外還有 always-active shared experts。表內只包含其餘 26 層的 routed experts，未納入 shared experts 和首層 dense FFN。

Granite、OLMoE 的數值推論使用 Q4_K_M GGUF，DeepSeek 使用 Q8_0。**真實 routing 來自這些量化模型，flash 成本則統一以 INT8 每參數 1 byte 計算。** 專案沒有把 Q4 檔案直接當成 AiF 的 INT8 權重影像；另一種量化下的 routing 也可能不同。

trace 來自真正的 router top-k，沒有用隨機 IDs 替代。開啟與關閉 tracing 的 greedy 生成文字已逐 byte 比對一致。這支持 callback 沒改變此次輸出，但不等於驗證了 AiF 硬體的數值正確性。

### 為什麼要保留第三組 prompt

每折只用兩組 prompts 學排列、挑 g 和版面，然後凍結配置，再看第三組。三組輪流當保留組，共三折。

如果先看完所有測試資料再挑最快配置，會高估實際收益。現有結果也看得到這個落差：v3 模型下，Granite 的學習配置相對原排列，在訓練資料改善約 17.94%，保留資料約 11.06%；OLMoE 是 11.28% 對 4.22%。三組 prompts 仍然很少，這只能算先導實驗。

此外，擷取程式使用 plain-text prompts，沒有套 chat template，並固定做 32 次 decode。這些結果不代表完整的聊天或長篇生成流量。

依據：[trace 與量化說明](/home/ericyu/expert_flash_experiment/docs/llama_trace.md)、[三折選擇與驗證](/home/ericyu/expert_flash_experiment/scripts/evaluate_learned_overhead.py:28)、[主要分析入口](/home/ericyu/expert_flash_experiment/scripts/run_analysis.py:19)。訓練改善由 NFS 各模型 `learned_overhead_fanout/heldout_all_configurations.csv`，依每折實際選定配置計算。

## 6. 目前結果告訴我們什麼

以下全部採用最新主要結果的 **v3 controller-fanout 模型**。每份邏輯輸入向量先跨 PCIe 傳到 controller 一次，再由 controller 分送 chips；複製到各 chips 的 channel 流量仍要計費。gate/up 可共用一份 x；down 則要各自傳入每個 expert 的中間向量。

額外成本假設是：每 chip/block GEMV command 0.25 μs、每 host phase 2 μs、輸入每元素 1 byte、輸出每元素 4 bytes。這些是用來展示取捨的參數，尚未經實體 AiF 校準。controller 的 chip commands 以序列發送計費，各階段之間有同步屏障。

### 6.1 三個模型的主要比較

![圖 5：主要配置比較](/home/ericyu/expert_flash_experiment/doc/assets/05_main_results.png)

單位：**每個 decoded token 的 routed FFN 服務時間，ms；越低越好。** 包含所有納入的 routed layers，但只包含 projections 及本節列出的控制、傳輸成本。

| 模型 | 16 chips，分開 gate/up | 16 chips，串接 gate/up | 訓練選 g＋串接，原排列 | 訓練選 g＋串接＋學習排列 | 原／學習配置的 g | 相對 16-chip 已串接降低 |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| Granite | 18.920 | 12.703 | 8.378 | **7.452** | 4／4 | **41.34%** |
| OLMoE | 16.568 | 13.648 | 13.390 | **12.824** | 8／8 | **6.04%** |
| DeepSeek V2 Lite | 24.917 | 21.333 | 21.333 | **21.015** | 16／8 | **1.49%** |

兩個訓練選配置欄位的時間是三折保留 prompt 平均。DeepSeek 的原排列選 g=16，學習排列選 g=8，因此它們是兩個各自經訓練選定的配置，不能把差值完全歸給同一 g 下的排列交換。

Granite 的改善可以拆開看：

- **先串接 gate/up：** 18.920 → 12.703 ms，減少短 streams 和重新開始的次數。
- **再選 g=4：** 12.703 → 8.378 ms，利用較長讀取與不同 groups 並行。
- **再學固定排列：** 8.378 → 7.452 ms，在保留 prompts 上進一步少了約 11.06%。

41.34% 是相對已經串接的強基準計算，不是把串接本身的收益也混進去。這讓「分組與放置還能增加多少收益」比較容易判斷。

資料來源：[原始主要結果 CSV](/home/ericyu/expert_flash_experiment/results/summary.csv)；更完整的每折資料位於 [NFS runs](/home/nfs_client/ericyu/expert_flash_experiment/runs) 各模型的 `learned_overhead_fanout/`。

### 6.2 g 沒有通用答案

![圖 6：不同模型在固定原排列、gate/up 串接下的 g 掃描](/home/ericyu/expert_flash_experiment/doc/assets/06_group_size.png)

這張圖先不使用 learned placement，直接看同一版面、原排列的 g 掃描；是三組 prompts 合併平均，和前表的訓練選配置程序不同。

Granite 的 expert 很小，g 太大時，切片過短、命令變多的損失明顯。OLMoE 和 DeepSeek 的 expert 較大，分給更多 chips 還有足夠工作可做，所以曲線的較低點往 g=8 或 16 移動。

這也解釋了為什麼不能拿模型總參數量直接猜配置。影響短讀取的，是**單個 projection 的尺寸、選中 expert 數，以及共同啟用的分布**。

資料來源：NFS 各模型 `overhead_fanout/overhead_placement.csv`，僅取 `stack_gate_up=True`。

專案也做過八個固定亂數排列的檢查。在**純 flash、gate/up 分開**的那組分析中，Granite 的最佳 g 都維持 4，OLMoE 都維持 8，表示這個趨勢不只出現在原始 expert 編號下。這是八個樣本的結果，不能當成所有排列的保證。另有 expert 尺寸與 block 大小掃描，其中假想尺寸沿用真實 routing IDs；它們是控制變因的敏感度分析，不是多測了那些尺寸的真實模型。詳見[排列敏感度紀錄](/home/ericyu/expert_flash_experiment/docs/placement_permutation_findings.md)。

### 6.3 DeepSeek 的例子：flash 變慢，總時間仍可能縮短

| DeepSeek 配置 | flash | chip command 控制 | host phase | 輸入傳輸 | 輸出傳輸 | 合計 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 原排列 g=16＋串接 | 19.3284 | 1.2480 | 0.1040 | 0.2729 | 0.3794 | 21.3327 |
| 學習排列 g=8＋串接 | 19.7449 | 0.6240 | 0.1040 | 0.1628 | 0.3794 | 21.0151 |

單位同為 ms/token。學習配置的純 flash 服務其實多了約 0.4165 ms，但 chip commands 少一半，控制成本省了 0.624 ms，輸入分送也比較少，最後才小幅勝出。

這個例子說明：**權重服務、控制和通訊要一起算，才知道哪個配置較快。**

資料來源：NFS `deepseek-aif/overhead_fanout/g16_stacked_overhead.expected.json`，以及 `learned_overhead_fanout/` 中三份 `*_learned_selected.expected.json` 的分項平均。

### 6.4 命令成本一變，較好的 g 也會變

![圖 7：chip command 成本的敏感度分析](/home/ericyu/expert_flash_experiment/doc/assets/07_control_sensitivity.png)

圖上每個成本點都重新用訓練 prompts 選配置，再評估保留 prompt。host phase 成本固定為 2 μs。

以 Granite 為例，每 chip command 成本為 0.25 μs 時選 g=4；提高到 1 μs 時，就改選 g=1。當發送命令很貴，把工作拆到更多 chips 的代價也更大。

DeepSeek 在成本接近零時，學習配置在保留資料上甚至略輸 16-chip 已串接基準。這是重要的限制：以訓練資料選出的策略，不保證在保留資料或另一種硬體成本下都贏。

完整實驗另有 host 成本 0、2、10 μs 的設定。圖中是參數掃描，不是從實體控制器量到的延遲曲線。

資料來源：NFS 各模型 `overhead_fanout/control_sensitivity.csv` 與 `learned_overhead_fanout/heldout_control_sensitivity.csv`。

## 7. 有具體地址計畫，但還沒有真的把權重寫進 AiF

如果只說「每個小 stream 一次 cold read」，可能隱藏一個問題：為了不跨 block，是不是每個 shard 都浪費一整個 block？

目前的 `materialize_plan.py` 已經處理這件事。它會產生每個 layer、expert、projection 的 chip、完整 row 範圍、plane、block 和 wordline offset，並安排短 streams 共用 block：

- 放得進 block 剩餘空間，就接著放，可以從非零 wordline 開始。
- 放不下就跳過尾端，從下一個 block 開始。
- 長 stream 才切成跨 block 的多段，各自收 cold-start 和 chip command 成本。

不同 expert 即使共用同一 block，每次請求仍各自重新開始，不假定讀取狀態能跨任意時間保留。

最新學習配置的容量如下，包含整個模型所有 **routed experts**，不是只存某次被選中的 experts：

| 模型／學習配置 | 有用的 INT8 權重 | 保留的 LSB 空間，含空位與 padding | 對應的 raw TLC cell 容量 |
| --- | ---: | ---: | ---: |
| Granite，g=4 | 1.125 GiB | 1.125 GiB | 3.375 GiB |
| OLMoE，g=8 | 6 GiB | 6 GiB | 18 GiB |
| DeepSeek，g=8 | 13.40625 GiB | 14.25 GiB | 42.75 GiB |

DeepSeek 的原排列 g=16 則保留 15 GiB LSB 空間，主要因為它的四-plane 波次有更多空位；兩種配置也各有 block 尾端 padding，權重本身沒有變多。

raw TLC 欄位以 LSB 空間的三倍計算，是對這批 cells 的保守容量保留方式。因為 CSB／MSB 仍可能存一般資料，不能把三倍全部稱為永遠浪費；也尚未計入 ECC、備用 blocks、其他模型權重或 GC 空間。

地址計畫已核對不重疊、rows 完整覆蓋、bytes 守恆，以及 pages／cold runs 和時序模型吻合。但它仍是**虛擬配置 metadata**，並沒有將模型權重寫進 replay 使用的約 2 GB 測試 namespace。

更細一點說，kernel descriptor 接收的是每 chip 的需求數量，沒有傳入這些實體 block／wordline 地址。地址計畫和重播統計雖然互相核對，VM 仍不會逐地址讀出權重，也沒有實作真實 FTL 搬移。

依據：[地址與容量配置程式](/home/ericyu/expert_flash_experiment/scripts/materialize_plan.py)、[descriptor 定義](/home/ericyu/expert_flash_experiment/kernel/aif.h)。最新版容量來源是 NFS 各模型 `learned_overhead_fanout/physical_plans/summary.json`。

## 8. 驗證做到哪裡，還有哪些問題沒回答

### 已有的證據

| 證據 | 目前狀態 | 能支持什麼 |
| --- | --- | --- |
| 真實推論與 tracing 對照 | 三模型、三 prompts，生成文字逐 byte 一致 | routing 擷取沒有改變此次 greedy 輸出 |
| Python 單元測試 | 本次重跑 **13 項全過** | 權重守恆、尾頁、block 邊界、串接、fanout 與成本計算符合測試案例 |
| 最新 v3 NVMeVirt 重播 | 保存了 **48 組 workloads、183,744 個命令** | 真正經過 VM 的 NVMe queue，kernel counters 和預期吻合 |
| 所有版本重播 | **100 組、813,504 個命令**，counters 全部吻合 | 包含較早純 flash 與相容版本的整體一致性紀錄 |
| 虛擬實體配置 | 三模型、各折選定配置均有地址與容量檢查 | 模型假設能對應到不重疊的靜態配置 |

但 Python 和 kernel 使用同一套時間方程。兩邊吻合，證明的是編碼、分派、完成時間與統計實作一致，**還不是獨立量測證明 AiF 真實硬體剛好這麼快**。

保存的 ioctl wall time 也另行記錄。它包含 VM、driver 和呼叫開銷，不能和模型服務時間混作同一個量。

驗證依據：[逐組重播彙整](/home/ericyu/expert_flash_experiment/results/nvmevirt_validation.json)、[實驗 manifest](/home/ericyu/expert_flash_experiment/results/experiment_manifest.json)、[重播檢查程式](/home/ericyu/expert_flash_experiment/scripts/validate_replay.py)。本次沒有重新啟動 VM；重播數字來自既有證據。

### 「實體位置差異」目前回答了哪些層次

| 想問的問題 | 目前覆蓋程度 |
| --- | --- |
| expert 放在哪組 chips，會不會和常一起啟用的 expert 衝突？ | **已比較**：modulo、固定亂數排列、逐層 learned placement |
| 每個 expert 分到多少 chips 比較好？ | **已掃描** g=1／2／4／8／16，並加入成本敏感度與保留 prompt 驗證 |
| stream 的 pages 能否連續、是否跨 block、容量是否放得下？ | **已建模並產生地址計畫**，尚未實體寫入 |
| 把不同 experts 固定分到不同 plane 子集合，能否同時算？ | **尚未實作**，只有條件式討論 |
| 同一顆 NAND 裡不同 wordline、block、die 本來的讀取延遲是否不同？ | **尚未研究**，目前位置使用相同基本時序參數 |
| FTL 搬移、GC、磨耗、其他資料流量會不會破壞配置？ | **尚未納入** |

這張表是理解專案定位最重要的界線。現在的結果已足以說明「位置安排會改變資源競爭與連續讀取」，但還不能說量到了「不同實體位置本身的電路延遲差」。

### plane 空位，不等於量到了 PE 使用率

Granite g=16、gate/up 分開時，有三個可以同時成立的數字：

- **100% chip 參與率**：所有 chips 都有工作。
- **50% plane-slot fill**：每條 stream 只用兩個 pages，卻保留一波四個位置。
- **約 18.3% 有效權重頻寬比例**：純 flash 下，真正有用的 bytes／時間，只到 102.4 GB/s 峰值的一小部分。

兩個有資料的 pages 本身可以完全填滿，所以 page fill 也可以是 100%。這些指標各自回答不同問題，都不是實測乘法器的忙碌比例；論文沒有公開 MAC lane 數。

原論文也沒有把「不同 expert 必須在同一 chip 完全串行」證明為不可突破的限制。專案現在採用這項排程假設；如果將來能讓不同 experts 使用互補 planes，同時準備輸入、維持獨立 accumulator，且 ECC／PE 吞吐量足夠，全 chip 配置也可能進一步改善。固定 plane 位置仍可能碰撞，不能先假設每次都完美互補。

[plane concurrency 討論文件](/home/ericyu/expert_flash_experiment/docs/plane_concurrency.md)中的 8.064 ms/token 是附加假設下的解析例子，不能當作已完成實驗。若使用「Plane Blocking」這個詞，應明確說是本專案目前 stream 排程造成的阻塞現象，不是 AiF 論文已命名並驗證的硬體定律。

### 尚未量測完整 AiF 推論速度

本專案時間沒有包含 attention、router、activation／combine 的計算、shared experts、dense FFN、DRAM cache、GC，以及 host 和 SSD 的張量分工與重疊。不能把 `1000 ÷ FFN ms/token` 直接當成完整模型 tokens/s，也不能把表中的 41.34% 寫成整體推論快了 41.34%。

原始 CPU 推論確實有完整 decode 的速度紀錄，但那是用來收集 trace 的量化 CPU 執行環境，也不是 AiF 的端到端測量。

## 9. 如果研究要往下走，最值得補什麼

以下是根據現有缺口提出的後續實驗建議，不是已完成的結果。

**第一個值得補的是 plane-aware placement。** 固定 chip 分組已經比較完整，下一步可以在相同 routing、相同權重與相同容量下，比較現有四-plane 整波 stream，和固定 plane 子集合的配置。需要明確建模每個 plane 的可用時間、矩陣 context、輸入選擇、accumulator、ECC／PE 共用上限，以及 NAND multi-plane 位址限制。這能回答現有的 g 收益有多少來自「chip 分組」，又有多少可以由更好的 chip 內排程取代。

**第二個是更多、較長且分布不同的 routing。** 目前只有三個短 prompts；learned placement 在訓練資料上的收益明顯較大。應先凍結某一批資料學到的配置，再測其他任務與長度，觀察碰撞和尾端延遲是否仍然改善。新的流量若不一樣，不應期待相同 g 永遠最好。

**第三個是校準控制與計算成本。** DeepSeek 目前只有約 1.49% 的收益，且勝負來自 flash 與命令成本的取捨。需要更具體的 controller／ECC／PE 模型或硬體證據，才能把展示用參數換成可信的系統預測。

**如果研究目標也包含「NAND 位置本身快慢不同」**，就要另外加入 block／wordline／die 的延遲分布及來源量測。現有模型中，只換 block 編號、沒有改變連續性或資源競爭，不會自動改變時間；這項研究不能只靠重新排列 expert IDs 完成。

## 附錄：想回頭看程式，從哪裡開始

### 專案地圖

| 路徑 | 在整個實驗中的角色 |
| --- | --- |
| [README](/home/ericyu/expert_flash_experiment/README.md) | 實驗入口、路徑與重現指令 |
| [scripts/llama_route_trace.cpp](/home/ericyu/expert_flash_experiment/scripts/llama_route_trace.cpp)、[run_llama_traces.py](/home/ericyu/expert_flash_experiment/scripts/run_llama_traces.py)、[inspect_gguf.py](/home/ericyu/expert_flash_experiment/scripts/inspect_gguf.py) | 取得真實 expert 選擇、檢查完整性、讀取矩陣維度 |
| [scripts/aif_model.py](/home/ericyu/expert_flash_experiment/scripts/aif_model.py) | 純 flash 計時、chip 分組、page／wave rounding、cold runs、gate/up 串接 |
| [scripts/overhead_model.py](/home/ericyu/expert_flash_experiment/scripts/overhead_model.py) | chip commands、host phase、PCIe／channel 向量傳輸與 fanout |
| [scripts/placement_sensitivity.py](/home/ericyu/expert_flash_experiment/scripts/placement_sensitivity.py)、[optimize_placement.py](/home/ericyu/expert_flash_experiment/scripts/optimize_placement.py)、[evaluate_learned_overhead.py](/home/ericyu/expert_flash_experiment/scripts/evaluate_learned_overhead.py) | 固定亂數排列、共同啟用學習、訓練 prompts 選配置、保留 prompt 評估 |
| [scripts/materialize_plan.py](/home/ericyu/expert_flash_experiment/scripts/materialize_plan.py) | 產生可檢查的虛擬 NAND 地址與容量計畫 |
| [kernel/aif.c](/home/ericyu/expert_flash_experiment/kernel/aif.c)、[aif.h](/home/ericyu/expert_flash_experiment/kernel/aif.h)、[patches/nvmevirt-aif.patch](/home/ericyu/expert_flash_experiment/patches/nvmevirt-aif.patch) | NVMeVirt 的 AiF metadata-only 時序擴充與完整 patch |
| [scripts/aif_replay.c](/home/ericyu/expert_flash_experiment/scripts/aif_replay.c)、[validate_replay.py](/home/ericyu/expert_flash_experiment/scripts/validate_replay.py)、[VM 說明](/home/ericyu/expert_flash_experiment/docs/vm-nvmevirt.md) | 送入真實 queue，核對 kernel counters；stock SSD baseline 是另一條測試路徑 |
| [scripts/run_analysis.py](/home/ericyu/expert_flash_experiment/scripts/run_analysis.py)、[results/report.md](/home/ericyu/expert_flash_experiment/results/report.md) | 串起分析流程、保存主要摘要 |
| [NFS 資料根目錄](/home/nfs_client/ericyu/expert_flash_experiment) | 大型模型、來源、編譯產物、VM、trace、逐組結果和 physical plans |

### 想核對計時，這幾行就夠

```text
一條 chip shard：
pages  = 向上取整(shard bytes / 16 KiB)
waves  = 向上取整(pages / 4)
cold   = 該 stream 跨越的 block runs 數

NAND 時間 = waves × 9.7 μs + cold × (28 − 9.7) μs
chip 服務時間 = max(NAND 時間, 實際 page bytes / 6.4 GB/s)

一個 phase：
服務時間 = 最忙 chip 的服務時間
         + controller 序列發送 chip/block commands 的成本
         + 一次 host phase 成本
         + 輸入傳輸 + 輸出傳輸

每個傳輸方向：
傳輸時間 = max(跨 PCIe bytes / 8 GB/s,
               最忙 channel 的 bytes / 2 GB/s)
```

目前每個 phase 的 expert 矩陣形狀相同，才能用這種彙整計算方式保留同 chip 串行的時間。若以後加入不同大小 experts、跨 expert plane packing 或多個非同步 requests，就需要更細的時間線，不能直接沿用現有彙整式。

每個 plane 內的 block 預設有 **192 條 wordlines**，因此每 block 有 192 個 LSB pages。192 由論文量測的 `3,686,400 WL ÷ 160 chips ÷ 120 blocks` 推得，不是 Table 2 直接公布的模擬器常數。

### 讀既有 docs 時，注意版本

`docs/implementation_review.md` 是較早的 review，提到 400-byte descriptor、7 項測試，以及尚未建立地址計畫。現況已有 v2／v3 的 800-byte descriptor、13 項測試和具體地址計畫。該文件提醒的實體硬體驗證限制仍成立，但不要把當時尚未完成的功能當成現在仍缺。

同樣地，`docs/physical_placement_plan.md` 主要整理前兩個模型與舊 `learned_overhead/` 目錄；最新三模型結果應看 `learned_overhead_fanout/`。`docs/plane_concurrency.md` 當時建議的 gate/up stacking 現在已有實作，跨 expert plane packing 則仍未完成。

### 本報告的圖與重現方式

七張圖均由本機 Matplotlib 產生，概念示意有標明範圍，數據圖使用保存的 v3 原始結果。PNG 供閱讀，SVG 供放大與後續簡報使用。

- [繪圖程式](/home/ericyu/expert_flash_experiment/doc/assets/build_figures.py)
- [圖表數據與來源快照](/home/ericyu/expert_flash_experiment/doc/assets/figure_data.json)
- [原始主要結果 JSON](/home/ericyu/expert_flash_experiment/results/summary.json)

從專案根目錄執行 `python doc/assets/build_figures.py` 即可重建圖，不會重跑推論、啟動 VM 或覆寫既有實驗結果。程式需要 Matplotlib、NumPy；有安裝 Noto CJK 字型時可直接顯示中文。
