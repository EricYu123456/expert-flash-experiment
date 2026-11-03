# 普通讀取下的 MoE 實體配置：候選系統研究

目前有值得深化的系統論文方向，尚未確認已達投稿突破：**依原生讀取粒度與完整列語義，編譯 expert 的實體配置，聯合處理工作量、job 深度、容量及 FFN 因果排程。** 核心不能寫成「首次使用 DP／joint mapping／overlapping stripes」，這些都有明確先例。可爭取的貢獻是問題的硬體界定、可重用的合法 lowering，以及在足夠強的既有幾何控制後仍可重現的效益。

最新一輪已將 training routes、配置選擇、完整 FFN CPU 數值檢查、原生粒度成本與多層虛擬地址串接。先前固定格式的新資料上，Deep 完整 FFN service 對當時 cyclic 控制改善 **1.87–3.26%**，G4 為 **0–2.64%**。後續更強的 joint simple-menu 控制，已將三層 TRAIN 的額外共同配置收益縮至 G4 **1.23–1.41%**、Deep **0.30–0.33%**；後者尚未作新 blind test。現有證據不能寫成已可投稿或通用加速。

## 問題與可證偽主張

原本的「分 group＋學 expert 排列」主要改變固定 shard 的碰撞與 CR-read 連續性。普通讀取下，地址排序不會讓相同數目的讀取自動變快；真正仍可改變的是每個 native job 的頁數、整數 read waves、chip 工作分布，以及 GU→activation→down 的就緒時間。

候選主張是：**以完整 output rows 為數值不變量，允許不等長 row bands，能避開均分與只選 group width 的整數取整損失；編譯器應直接搜尋可執行幾何，而非只學 expert ID 順序。** 這個主張若被同幾何的強 dense packing／declustering／native scheduling 控制消除，就不成立，不能改以較弱 baseline 宣稱成功。

## 一個有用的算術界線

令一個 wave 可讀 (B) bytes、一個完整係數 row 占 (C) bytes、矩陣有 (R) rows，配置族限制為至多 (M) 個 native jobs，通常每 chip 一個。每個 job 分別取整，不與另一 job 共用尾部：

\[
w_i=\left\lceil\frac{r_iC}{B}\right\rceil,\qquad \sum_i r_i=R.
\]

若 (RC=WB) 為 exact fill，達到有用係數工作量下界 (W) 時，每個 job 的非負 padding 都必須為零。設 (q=C/\gcd(C,B))，則各 (w_i) 都是 (q) 的倍數，因而：

\[
D_{\mathrm{exact}}=q\left\lceil\frac{W/q}{M}\right\rceil,\qquad
D_{\min}=\left\lceil\frac{\lceil R/M\rceil C}{B}\right\rceil.
\]

這是基本整數 packing 推論，並不主張新 theorem。它解釋一個容易被 service-byte 模型隱藏的現象：**最少總讀取工作與最短單 job 可以互相衝突。** Qwen3.8 GU 的最少工作是 50 waves，但需至少 5-wave job；若把最大 job 限為 4 waves，精確 DP 下界是 51。Down 的最少工作 25 waves 也需 5-wave job；2-wave 深度可用 26 waves。Native replay 已確認 g10／g5 的 75-wave FFN 反而比 77-wave balanced 幾何慢，不能只優化總 bytes。

這個界線不是任意多 jobs／跨 block 配置的全域 makespan 定理。跨 block 的完整列重啟可能再取整；每個 job 的 wave 起點、實體容量及命令數必須另外 lower。Metadata 若與係數一起讀，也必須納入 footprint。

## 已有可重用原型

[whole-row compiler](/home/ericyu/expert_flash_experiment/scripts/research_20261102_whole_row_compiler.py) 接受 rows、columns、chip 數、plane 數與 page bytes，產生三種明確分開的控制：等分 rows 的 wave profile、可行的均衡 wave profile、在指定最大 job 深度下最少 aggregate waves 的精確 integer-covering DP witness。它輸出完整 row ranges、pages、waves 與 block-command 限制；不把 scalar Pareto dominance 當成真實 query latency dominance。

例如：

```bash
python scripts/research_20261102_whole_row_compiler.py \
  --rows 1280 --columns 2560 --chips 16 --planes 4 --page-bytes 16384 \
  --output /tmp/qwen_gu_geometry.json
```

獨立直接列舉非遞減 row partitions，已核對 3,456 個小形狀、39,060 個 depth-bound 比較及 1,664 個 exact-fill cases。七模型的十四個實際 projection shapes 亦通過完整係數 coverage、獨立 ceil 成本與單 block 檢查。[獨立證據](/home/ericyu/expert_flash_experiment/results/research-2026-11-02/round2/whole-row-compiler/independent/notes.json)保留了曾發現並修正的 balanced-profile 邊界反例。

| 形狀 | GU 等分／均衡 waves | Down 等分／均衡 waves | 已有解釋 |
|---|---:|---:|---|
| Granite3 | 16 / 16 | 16 / 8 | Down 容量不足一個全 chip wave，可減少 active chips；需支付 query 碰撞 |
| OLMoE | 64 / 64 | 32 / 32 | 等分已達係數工作量下界 |
| DeepSeek V2 Lite | 96 / 88 | 48 / 45 | 不等長 bands 降低整數取整；Down exact-work 深度需 11 |
| Qwen3 | 48 / 48 | 32 / 25 | Down 的 row/wave 週期為 3 |
| Mixtral | 1792 / 1792 | 896 / 896 | 原幾何已達工作量下界 |
| Granite4 Tiny | 32 / 25 | 16 / 12 | GU work/depth 取捨；Down 省組配置已有 staggered-striping prior |
| Qwen3.8 | 64 / 51 | 32 / 26 | GU／Down 的週期皆為 5 |

這張表是幾何證書，不能當作 speedup 表。它也不支持「所有 in-flash computing 都一定受益」；不同 backend 的原生粒度與可組合指令可能產生零收益或不同最適解。

## 數值與實體證據

完整 rows 的轉換可對 opaque native row function 證明：保留原 row、完整 input、dot length 與 row／expert scale metadata，原生輸出若逐 row 可分離，就只需 scatter 回原邏輯位置。它不需要 column splitting 的 raw-I32 partial-return 契約。會跨 rows 重新校準的 output quantizer 則需另外處理。

完整 G4 FFN CPU 參考已覆蓋全部 40 層、240 個 FFN weight tensor 名稱與全部 prefill／decode。兩個訓練 prompts、各 16 decode 下，197,017,600 個相同 row outputs 的 exact I64 dot 與跨執行 F32 輸出檢查皆零差異，3,411,968 個 logits bitwise 相同，routes／tokens 相同。每個 placed row 確實讀取 chip-packed W8 bytes，再 scatter 回邏輯 rows。

這是固定 W8A8 CPU 語義，不是新量化方法、AiF ABI 或原 GGUF 品質證明。原 GGUF control 的兩組 token agreement 分別為 0/16、4/16，已明確揭露數值基準改變。所有 placement 比較必須在同一個數值參考內進行。[完整說明與來源](/home/ericyu/expert_flash_experiment/results/research-2026-11-02/round2/full-ffn-integer-reference/notes.md)。

第二模型 DeepSeek V2 Lite 的同契約 reference 已覆蓋 raw0 dense 及 raw1–26 routed／shared FFN，共 159 weights。其 shared I=2816 是一個完整分支，down 計一次。兩模型另以同 32 個 TRAIN prompts × 16 decode 取得實際 W8 routes，再各自 fit／選 geometry，沒有將新測試 routes 用於配置。

最新第二批 heldout 是 OpenBookQA test 8 題＋BoolQ validation 8 題，固定 revision、ID hash 選樣與完整文字，答案不附 prompt。兩模型 × input1／input4 selected maps 共 64 個 paired prompts（16 decode／題）全部完成。G4 每個 view 為 **1,610,997,760 FFN F32 scalars、27,295,744 logits**；Deep 為 **2,198,611,200 scalars、27,852,800 logits**。canonical／placed 以及兩種配置間的完整輸出逐 byte 相同，全部 routes／生成 tokens 相同。三條 comparisons 證明四個 views 一致，不能把同一批輸出重複計成三倍樣本。[完整驗證索引](/home/ericyu/expert_flash_experiment/results/research-2026-11-02/round2/selected-plan-numeric/second-heldout/validation_artifact_index.json)。

固定格式本身仍有品質代價：G4 的 8-window WikiText teacher PPL 相對來源 GGUF 增加 0.094%；Deep 相對原 Q8 GGUF **16.410216 → 16.728945（+1.9423%）**，8 個 windows 都退化。這是 W8 reference 的差異，placement 在同 reference 內保持完全相同輸出，沒有新量化或原 checkpoint 品質保證。[Deep 固定格式品質補充](/home/ericyu/expert_flash_experiment/results/research-2026-11-02/round2/deepseek-teacher-quality/summary.json)。

Qwen3.8 的全部 240 個 frozen views 已降成虛擬 chip／four-plane block／LSB wordline 地址，另以 bitmap 與逐 row interval checker 檢查。所有 job 在單一 192-WL block 內，無地址碰撞，保留 command 數；whole-row 配置 useful weights 為 112.719727 GiB，allocated LSB blocks 為 115.875 GiB。這仍是地址證書，未下載完整模型權重、寫入實體 NAND 或執行 native PE。[地址證據](/home/ericyu/expert_flash_experiment/results/research-2026-11-02/round2/qwen38-placement/physical-addresses/summary.json)。

## 最新共同配置強控制

這一輪只用兩模型的實際 W8 TRAIN512 routes、各前三個 MoE layers，沒有再讀 heldout。先讓 GU／Down 的高 wave chips 互補；G4 的 25＋12 waves 可配成每 expert `11×2＋5×3` joined waves，Deep 的 88＋45 可配成 `11×8＋5×9`。這是最小交集的基本 packing 計數，不能主張新數論結果或原生延遲保證。

第一版對 equal-row cyclic 的三層 mean 改善約 1.83–4.28%，但不足以歸因。修正後基準包含全部 cyclic／disjoint／balanced-wave／depth templates、共同 shared candidates、雙 phase label／chip swaps，以及從 coupled incumbent 繼續搜尋的 unrestricted row permutations。以 TRAIN mean 選出的 plans，將三層逐 token 對齊相加：

| 模型 | ONFI copies | Joint simple-menu ms | Unrestricted paired ms | Mean 改善 | P99 改善 |
|---|---:|---:|---:|---:|---:|
| G4 | 1 | 1.714969 | 1.690822 | 1.4080% | 1.8085% |
| G4 | 4 | 1.783105 | 1.761182 | 1.2295% | 2.4412% |
| Deep | 1 | 5.771215 | 5.752082 | 0.3315% | 0.4658% |
| Deep | 4 | 5.895378 | 5.877861 | 0.2971% | 0.4983% |

這些是開發／訓練資料的描述性結果，不是完整模型、新測試或硬體速度。Unrestricted 控制保留了先前 fitted incumbent，其上游搜尋成本不能記為免費，也不能宣稱有限搜尋的 constrained family 優於 unrestricted 最優。G4 的第一次 literal family assertion 在 fit 前停止，保留原 source／protocol 與 erratum 後，改用完整 geometry orbit 檢查；Deep 原成功輸出未重跑。獨立核對 432 unique artifacts、184,320 query-work、828,672 jobs 及 72 Python／C++ 成本皆通過。[完整報告](/home/ericyu/expert_flash_experiment/results/research-2026-11-02/round2/shared-counterbalanced/independent-v2/report.md)。

此版本沒有跨過預先設定的擴展門檻，已停止全層展開。相近 prior 的具體差異仍有研究空間：[HDA-MoE](https://arxiv.org/html/2609.08682v1) 使用每 expert aggregate fraction；[Stratum](https://arxiv.org/html/2510.05245v1) 共同處理 GU／Down ownership，但 Down 採 input-column 切分及 partial reduction。這支持問題差異，不能支持 first joint mapping 或完整新穎性。[正文／官方 code 查核範圍](/home/ericyu/expert_flash_experiment/results/research-2026-11-02/round2/coupled-closest-prior/notes.json)。

必要工作量也不足以替代完整排程。獨立合法 G4 witness 在相同每 expert／chip joined waves 下，joined work 都為 573,058 ns，native service 卻為 575,356／575,122 ns；另一對 work 573,058 ns 比 573,734 ns 小，service 575,946 ns 卻比 575,872 ns 慢。五張 maps 的完整地址與 ReferenceGraph 皆已核對。後續以有限的 exact-native objective 搜尋作控制，並有界篩選 expert-specific GU／Down geometry；這是既有搜尋與 elastic sharding 方法的具體移植，尚未產生新的投稿主張。[反例與適用條件](/home/ericyu/expert_flash_experiment/results/research-2026-11-02/round2/coupled-work-contract-audit/notes.txt)。

這兩項有界控制現已完成。G4 L0/input1、Deep L0/input4 的 exact-native refinement 共八 fits（各 family 兩 seeds、2,048 proposals），七份 mean 完全未改善，唯一增幅只省 9.63 ns。額外允許每 expert 換 GU／Down geometry 的四 fits，最佳輸出都沒有保留新 geometry；相對最強 fixed-profile mean，G4／Deep 只改善 **0.000127%／0.003182%**。因此停止擴展，不把 geometry 方法本身判成全域無效。[直接排程成本控制](/home/ericyu/expert_flash_experiment/results/research-2026-11-02/round2/native-joint-refinement/pilot-v1/report.md)、[mixed-geometry 報告](/home/ericyu/expert_flash_experiment/results/research-2026-11-02/round2/native-mixed-geometry/pilot-v1/independent/report.md)。

另一個標準 CP-SAT 控制把 G4 row geometry 壓成每 expert／chip 的兩個 bool，joined occupancy 精確為 `57,864＋28,086X−676O ns`。總 solver 148.58 秒，19-wave feasibility 超時 UNKNOWN；20-wave 找到可行配置，mean-work 仍未證最佳。189 份記錄的 native best mean **562.671 → 561.099 μs（0.2793%）**，TRAIN 單層 P99 改善 4.0029%，仍未達展開門檻；96,768 costs、567 traces 與完整同容量地址帳通過。這是更強的有限搜尋，不能宣稱 exact native optimum、新 solver 或全域近優。[CP 控制範圍及結果](/home/ericyu/expert_flash_experiment/results/research-2026-11-02/round2/compact-coupled-solver/notes.json)。

GU 的共同 input 候選亦已完成第一輪完整 DAG：緊密放置不同 experts 的完整 rows，使 logical read views 可重疊同一 native wave，coactive experts 再合成一次 ordinary GEMV。Deep C2048 每 wave 恰32完整 rows，兩個176-row bands在正確 parity 下可由兩個6-wave views降成11waves；singleton必付完整前後 fringe，shared／Down仍遵守原本依賴。它在先前 Qwen C2560 的小 tail 證書之外，但 uniform176 本身是96waves/expert，仍輸88-wave balanced geometry。5μs command 的 TRAIN512／L0 完整 DAG，input1／4 最佳 mean 只降低 **1.0209%／1.3642%**，同一 mean-selected map 的 P99 反而變差 **0.4985%／0.7604%**。16,384 筆舊 engine bridge costs、192 traces 與獨立 interval-DP 查核通過；未達擴展門檻，沒有全層或新 heldout 主張。[結果](/home/ericyu/expert_flash_experiment/results/research-2026-11-02/round2/ordinary-gu-tile-sharing/native-dag-v1/screen.json)、[獨立查核](/home/ericyu/expert_flash_experiment/results/research-2026-11-02/round2/ordinary-gu-tile-sharing/independent/report.md)、[直接 prior 與 matching 控制](/home/ericyu/expert_flash_experiment/results/research-2026-11-02/round2/ordinary-gu-tile-sharing/closest-prior/notes.json)。

另有 G4 Down 小型 copies 篩選：每 routed expert 複製一個128-row完整 tile，共64waves／4MiB。所選 layout 枚舉每 query 的64種 A/B views，chip-work 最佳只降 **1.3914%**；加入目前 executor 的首次 chip dispatch 時序後，對 native mean 的收益上限為 **1.5054%**。所有 stored-copy bytes、addresses、row IDs 及32,768個 view choices 已核對，未進入重型副本控制或 heldout。這只否定該有限 layout／固定 executor 的展開價值。[封存決策](/home/ericyu/expert_flash_experiment/results/research-2026-11-02/round2/g4-down-tile-copy-screen/decision.json)。

條件式 FIFO 敏感度亦已停止：完整expert DMA、零額外軟體成本下，相對先前最強legacy DP，input1／4 mean只再改善 **0.1572%／0.1935%**；240records／122,880 costs、30,720 bridges及480traces通過核對。未啟動非零cost或全層。Prefix最終成功／ECC、row order、有限FIFO與controller buffer仍未驗證；trace的28–38KiB payload occupancy不能當實際buffer allocation，七個完整GU buffers的靜態需求仍88KiB。[敏感度與條件](/home/ericyu/expert_flash_experiment/results/research-2026-11-02/round2/ordinary-gu-tile-sharing/conditional-stream-v1/independent/report.md)。

Deep八個GU邊界全部成對複製，也只給出 **1.0741%** 的native mean樂觀上限；去掉endpoint限制仍約1.0745%。32MiB useful copies讓固定inventory最低LSB配置由576增至744MiB，已封存no-go。所有6,144 flow optima及98,304 prefix masks獨立核對；沒有啟動heldout或把Euler orientation當新定理。[契約、下界與完整容量帳](/home/ericyu/expert_flash_experiment/results/research-2026-11-02/round2/deep-gu-paired-replica-screen/independent-contract/contract_notes.txt)。

G4共prefix的g8／g16整expert雙span亦未過篩選；放寬A位置、加入互補matching後，有限搜尋的native mean／P99樂觀上限仍只有1.5024%／0.8736%。改成逐source長／短prefix後，完整native TRAIN512則是 **561.099→551.417μs（1.7255%）**，P99改善0.6005%，但尚未計selector，仍未進大型同容量副本或heldout控制。17個queries反而較慢；沒有把LP fractional空間當突破。[版本及完整原生結果](/home/ericyu/expert_flash_experiment/results/research-2026-11-02/round2/g4-elastic-prefix-screen/work-v3-independent-prefix/native-v1/report.json)。

Down六對128row shards的獨立split／merge副本，平均改善1.9486%；與GU逐source選擇聯合後為 **549.022μs（2.1524%）**，P99只改善0.3268%，相對較強Down-only再省0.2079%。聯合結果有36／512筆退步，最差慢29.118μs，且尚未計selector。新增57.9873MiB useful copies／56MiB programmed，單層allocated LSB由192變384MiB。[聯合結果及限制](/home/ericyu/expert_flash_experiment/results/research-2026-11-02/round2/g4-joint-prefix-replica-screen/native-v1/report.json)。

同allocated容量的完整GU副本控制已給出具體地址及逐job等價證書：原B＋Down14 maximal prefix images各chip≤189waves可放block0；完整A1 GU備份可逐job重現原joint讀法，再存互補A2仍只用block1的192waves。全部512queries／75,594 jobs的chip、full columns、rows、pages、waves與counters相同，shared兩projection各計一次。因此prefix確實少用programmed頁數，卻沒有本次單層allocated容量／服務優勢；不能排除填滿相同blocks的強控制。此結論不等同全模型global packing，也未測A2更多選擇的性能。[地址構造](/home/ericyu/expert_flash_experiment/results/research-2026-11-02/round2/g4-joint-prefix-replica-screen/same-allocated-witness/audit.json)、[獨立覆核封存](/home/ericyu/expert_flash_experiment/results/research-2026-11-02/round2/g4-joint-prefix-replica-screen/same-allocated-independent/completion.json)。

另一輪改查8／16／32／64個相同AiF-like compute owners的規模敏感度；新增owners須複製原本PE／ECC／input資源，並非商用SSD已具該運算能力，也不能跨硬體計placement gain。94個row work／depth前沿由獨立inverse-capacity DP核對，29,696筆有限控制成本包含跨fanout的原16chip強anchor。[範圍與結果](/home/ericyu/expert_flash_experiment/results/research-2026-11-02/round2/chip-count-regime-screen/stage-ab-seal/report.md)。

G4 M32的零副本free-mask CP保持GU25個一wave、Down12個一wave與原shared，單層TRAIN512 mean／P99為299.342／300.795μs。相對未學位置的有限候選約8.46%／20.76%，但補上learned共同cyclic rotations後，大部mean收益消失；再讓GU／Down各自學rotation，基準為301.168／328.789μs，free只餘 **0.606% mean／8.514% P99**，227筆較快、279筆較慢。這不能保留約9%為新方法歸因。Rows、單副本、單block地址與timeline獨立核對通過；32owner仍是重複相同PE資源的規模敏感度，不是現有AiF實物證據。[phase-independent強控制](/home/ericyu/expert_flash_experiment/results/research-2026-11-02/round2/chip-count-regime-screen/independent-m32-phase-rotation-final/audit.json)。

另加TRAIN-only、直接針對peak／top-six work的cyclic coordinate控制，保留兩個anchors及四個固定fits、公平重播GF／DF與input1／4，共12,288costs。其TRAIN tail-selected GF為300.757／328.368μs，free仍約 **0.470% mean／8.397% P99**；有限搜尋不是cyclic無法追上的證明。此補充控制在新題已取得、root尚未讀取新routes或成本時封存，不能冒稱原prospective primary或全團隊皆未接觸新payload。[補充控制](/home/ericyu/expert_flash_experiment/results/research-2026-11-02/round2/chip-count-regime-screen/m32-tail-directed-rotation-control/native_report.json)。

原三個TRAIN-selected圖與primary分析先於新資料取得封存，32個新PIQA／WinoGrande prompts的完整40層canonical W8 capture進行中；只用其L0路由驗證，所有16decode步保留，32prompt-cluster bootstrap2000次。三圖另已實際寫成96個bank files，獨立由原16banks重建原W8與scales，四組固定TRAIN fixtures的571,476 scalars、GU／完整A8／Down／orderedmix逐bit吻合。這是離線CPU數值／實體證明，不是新32題paired full-model或native ABI證據。[預封分析](/home/ericyu/expert_flash_experiment/results/research-2026-11-02/round2/chip-count-regime-screen/m32-prospective-tail-validation/protocol.frozen.json)、[獨立數值複驗](/home/ericyu/expert_flash_experiment/results/research-2026-11-02/round2/chip-count-regime-screen/m32-prospective-tail-validation/independent/numeric-materialization-audit/audit.json)。

## 先前固定配置的新資料性能證據

較新的主要比較使用**兩模型自己的實際 W8 TRAIN routes**，搜尋 disjoint、全部 cyclic widths 與 balanced／integer-depth complete-row profiles。等價 geometry 只 fit 一次，避免因不同名稱多抽 seeds。132 layer/cost cases／campaign 的配置、來源與 TRAIN manifest 先凍結成 anchor SHA `00507ff56f9197fb18511ad58303919214c03cba6f0c1440f6733ffc42944d00`，才取得 OpenBookQA／BoolQ payload；之後不再 fit。

以下是 16 題 × 16 decode 的 heldout 描述性結果，三種 family 都使用各自的 TRAIN-selected plan。ONFI inputcopy4 是 per-plane reload 成本情境，PCIe 仍只 upload 一份 common／private input；實際 broadcast ABI 尚未知。

| 模型 | ONFI copies | Command ns | Cyclic MoE＋shared ms | Whole ms | 完整 FFN service 改善 |
|---|---:|---:|---:|---:|---:|
| Deep | 1 | 250 | 52.357222 | 50.603891 | 3.2344% |
| Deep | 1 | 5000 | 55.196110 | 54.122153 | 1.8779% |
| Deep | 4 | 250 | 53.458011 | 51.655093 | 3.2575% |
| Deep | 4 | 5000 | 56.214576 | 55.127055 | 1.8677% |
| G4 | 1 | 250 | 24.510604 | 23.863417 | 2.6404% |
| G4 | 4 | 250 | 25.136775 | 24.991046 | 0.5797% |
| G4 | 1／4 | 5000 | 28.796653／29.364461 | 相同 | 0% |

Deep 的表內毫秒欄只列 26 MoE＋shared；最後一欄已對兩 family **各加一次**固定 raw0 dense prefix（1.852224–2.013724 ms，隨兩成本軸），所以是完整 FFN service 的分母。attention／router／SSM、host row scales／scatter、activation／combine 時間仍未包含，不能稱完整模型速度。G4 低成本兩情境的 16 個 prompt means 都改善，但 input1 有 3／256、input4 有 **66／256** 個 token 變慢；不報成逐 token 支配。[全部配置、成本、逐題與 dataset 結果](/home/ericyu/expert_flash_experiment/results/research-2026-11-02/round2/ordinary-read-compiler/second-heldout-replay-v2/comparison.md)。

相同資料的逐題、mean／P99 與容量取捨已整理成 [獨立 PDF 圖](/home/ericyu/expert_flash_experiment/results/research-2026-11-02/round2/ordinary-read-compiler/second-heldout-replay-v2/figures/same_format_heldout.pdf)，並保留繪圖 input／output hashes 與 caption。這張圖使用先前已凍結的控制，不包含後續 TRAIN-only joint simple-menu 方案。

原始 G4 low-command whole 為 6.0 GiB allocated LSB blocks，cyclic 為 5.90625 GiB。新的 route-blind 跨層共同 chip-swap 控制保持完整 resource graph 同構，cyclic 可再降至 **5.8125 GiB**，whole 仍為 6.0 GiB；因此目前 G4 速度不是同容量的 Pareto 改善。Deep whole 的 14.0625 GiB 小於 cyclic 的 14.25／14.8125 GiB。這裡的容量只含係數 LSB reservation，不是實機可用容量或 full-model footprint；capacity symmetry 是所有 family 的強控制，不是新方法。

目前所有正收益的 TRAIN winner 都只用既有 cyclic profiles 或簡單 balanced-wave profile，沒有 evidence 證明 DP／branch-and-bound 搜尋本身帶來額外效益。下一步必須比較簡單 wave-aware tiler 與同成本 joint mapper，或找出不同的有效機制，不能把整合工程自動升格為演算法突破。

下列 Qwen 與早期 G4 是開發階段的補充證據，保留其各自格式／控制限制：

Qwen3.8 的 240 plans 在外部 payload 取得前凍結；外部只有一個 document-QA session，排除 warm-up 後為 5,519 decode tokens。相同 native resource／因果模型中，48 層含 shared FFN 的 mean service：

| 控制 | Mean (ms) |
|---|---:|
| 固定 g16 | 89.009520 |
| 依 train 選出的 disjoint groups | 91.225482 |
| 搜尋全部寬度的 overlapping stripes | 84.645319 |
| 完整列不等長配置 | 76.828409 |
| 有 partial ABI 條件的 2D rectangle 控制 | 74.624175 |

完整列對較強 overlapping 控制為 9.2349% mean 改善。這是 nominal I8/A8/4B-output 的 FFN 成本重播，routes 來自 NVFP4 checkpoint，不是該 checkpoint 在 AiF 的實測或端到端模型時間。Source engine 無法取得的 provenance 限制亦仍存在。[凍結與獨立核對](/home/ericyu/expert_flash_experiment/results/research-2026-11-02/round2/qwen38-placement/independent-audit/audit-external.json)。

另外，直接使用完整 W8A8 CPU 參考產生的真實 routes，將舊凍結 G4 plans 原封不動套用至全部 40 層、同一 shared 配置，32 個 decode 的 FFN mean 為 phase-group 24.808711 ms、whole rows 23.742340 ms，改善 4.2984%。這只補上同格式 routing 的探索證據，兩題都是早先訓練 prompts，不能當新的 held-out 性能結果。[同格式服務重播](/home/ericyu/expert_flash_experiment/results/research-2026-11-02/round2/full-ffn-integer-reference/frozen-placement-replay/summary.json)。

AiF page-count 指令是否能在 useful rows 截止仍待確認。若 padding 也輸出完整 rows，就須 drain 額外結果。已用相同 240 plans、全部外部 tokens 重播 requested-page／full-wave 兩種輸出契約，channel 與 PCIe 均支付額外結果：whole mean 為 76.834621 ms、overlap 84.656573 ms，較 useful-only 假設僅增加 0.0081%／0.0133%，配置收益仍存在。這是有明確假設的敏感度，沒有涵蓋未公開的 partial-row flush／unused-plane 行為，不能稱硬體 worst-case 保證。[結果](/home/ericyu/expert_flash_experiment/results/research-2026-11-02/round2/qwen38-placement/output-fringe/summary.json)。

**Controller 成本會改變勝負，250 ns 不能當成硬體定論。** 另外用相同凍結配置、外部 256 個固定 tokens、全部 48 層測試 11 組成本參數，額外輸出也付費：

| 每 native matrix command 的假設成本 | Whole mean (ms) | Overlap mean (ms) | Whole 改善 |
|---|---:|---:|---:|
| 0.25 μs | 76.845573 | 84.664488 | 9.2352% |
| 1 μs | 77.931664 | 85.695291 | 9.0596% |
| 2 μs | 79.516787 | 87.141520 | 8.7498% |
| 4 μs | 87.021495 | 90.464510 | 3.8059% |
| 5 μs | 96.493326 | 92.579877 | −4.2271% |
| 10 μs | 170.427600 | 141.627600 | −20.3350% |

原因可直接核對：whole 的 48 層共有 16,896 commands，overlap 為 14,016；在 10 μs 時 controller 本身的 serialized work 就是 168.96／140.16 ms。配置因而必須依平台成本選擇，不能只用最少 waves 的硬體無關常數。普通 read 改為 37／56 μs、維持 command 250 ns 時，whole 的改善仍約 9.29／9.35%；host phase 改為 0／10 μs 也未消除本輪收益。這只是同一組 frozen schemes 的參數敏感度，沒有替每個新平台重訓或主張最佳配置。[結果與假設](/home/ericyu/expert_flash_experiment/results/research-2026-11-02/round2/qwen38-placement/parameter-sensitivity/summary.json)。

這裡採用單一 serialized command-dispatch resource，是本次可執行排程的模型假設。若實際 controller 可多路派送，就需修改 resource graph，不能將上述 controller floor 或翻轉點直接套到該硬體。

## 已否決的增量方向與直接 prior

相同搜尋預算下，自由 masks 沒有勝過 cyclic profiles；四個 route-blind diverse-mask seeds 平均慢約 1.39%。GU/down joint fit 對簡單 complement 的額外改善僅約 0.2–0.3%；25 組 work/depth 幾何控制再改善僅 0.0152%。這些結果保留為強控制，不能重新包裝為突破。

直接比較應包括 [weighted hypergraph declustering](https://ir.lib.nycu.edu.tw/bitstream/11536/30045/1/000172367000003.pdf)、[dependent rounding](https://crab.rutgers.edu/users/rajivg/publications/depRound.pdf)、[Cambricon-LLM §V](https://arxiv.org/html/2409.15654v1)、[PIMCOMP](https://arxiv.org/abs/2411.09159)、[Fast-OverlaPIM](https://arxiv.org/abs/2407.00604) 與 [Stratum](https://arxiv.org/abs/2510.05245)。較新的 [BigMoMo §4.3](https://arxiv.org/html/2609.14643v2#S4.SS3) 已用實際 Flash co-loading hypergraph 做動態 expert 交換，但需真的搬寫 whole-expert slots；不能只改 metadata 假裝重配置了實體權重。

有限的一手文獻查核尚未找到完全相同的 ordinary-NAND whole-row work/depth compiler＋數值／實體證書系統。這不是優先權證明，也不能因通用 declustering 能表達某個子問題，就反向斷定整個系統沒有研究價值。

## 尚未達到投稿的部分

主要 compiler、凍結配置、numeric reference、native schedule 及地址 lowerer 已接成可重現流程，兩份 W8 TRAIN campaign 的 24 maps／1,143,424 jobs 由獨立 checker 核對。這些功能不是首次：PIMCOMP 已有逐 instruction verifier／成本回饋，TileLens 已有粒度／地址編譯。係數之外的 scale／address metadata、原生 valid-row／fringe output、activation／combine、controller 校準成本仍須如實計入。

同格式的兩模型／兩來源 heldout 已完成，但幅度有限，尚缺完整模型時間、可校準的 native ABI 與足夠強的同容量／同預算 mapper。Qwen 第三方 NVFP4 routing＋nominal I8 成本只作探索。若簡單幾何與同成本既有 mapper 精確匹配，就應放棄此候選的演算法新穎性，不能僅靠地址證書或較弱 baseline 完成論文主張。具體 prior 能力矩陣見 [貢獻審核](/home/ericyu/expert_flash_experiment/results/research-2026-11-02/round2/candidate-novelty-audit/audit.md)。
