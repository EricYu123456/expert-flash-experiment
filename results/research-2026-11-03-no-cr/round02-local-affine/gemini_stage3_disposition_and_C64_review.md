# Stage-3 處置審視與 `earliest64`（`C-first64`）前瞻理論審稿

- **回合與身分**：同一 `Round 2`（`layer-shared-input-partition-local-affine-expert-compiler`），`gemini-4-argon-high` 理論與批判審稿。本文件**不啟動任何新方法、不主張任何新 30 分鐘思考計時（clock credit）、不擴充超參數網格、不要求聯絡外部論文作者，亦不授權 `first64` 實作、擬合或新驗證集拆封**。
- **唯讀審視依據**：
  1. [`earliest64_Cfirst_prospective_stage.sol.md`](file:///home/ericyu/expert_flash_experiment/results/research-2026-11-03-no-cr/round02-local-affine/earliest64_Cfirst_prospective_stage.sol.md#L1-L77)
  2. [`aif_r128_16lane_construction.sol.md`](file:///home/ericyu/expert_flash_experiment/results/research-2026-11-03-no-cr/round02-local-affine/aif_r128_16lane_construction.sol.md#L1-L76)
  3. [`Root_DEV_QUALITY_independent_statistics_audit.json`](file:///home/ericyu/expert_flash_experiment/results/research-2026-11-03-no-cr/round02-local-affine/stage3-control-cost/Root_DEV_QUALITY_independent_statistics_audit.json#L1-L81)

---

## 1. 新 `NL8` 與 `BENCH` 結果對 `C` 價值的實質衝擊，以及為何不能據此自動宣布 Round 2 失敗

### 1.1 `NL8` 確實大幅收窄了 `C` 的機制主張：不能再宣稱「局部條件輸出斜率」對維持品質唯一必要
根據已獨立複核閉合的 [`Root_DEV_QUALITY_independent_statistics_audit.json`](file:///home/ericyu/expert_flash_experiment/results/research-2026-11-03-no-cr/round02-local-affine/stage3-control-cost/Root_DEV_QUALITY_independent_statistics_audit.json#L17-L75)，固定無網格配方的 `NL8`（共用同一層 `P128/tree32`、每個 head 僅 **1 組全域 pooled `FP32` `Gate/Up` 斜率** + `C32` 葉節點 `GU` 偏置 + 完整 `I` 維原生 `SwiGLU` + 原始 canonical `W8 Down` + `C32` 實際預測輸出偏置，`0` 次 canonical fallback）在既有開發集 `PG-19 validation16` 上取得：
- **`DeepSeek`**：`NL - B` 平均 $\Delta\text{NLL} = +0.00299089$，simultaneous 95% 信賴區間 $[-0.00678947,\ +0.01277125] < \ln(1.02) = 0.01980263$；描述性 `mean(NL - C) = +7.35e-5` nats（與 `C8` 的 `+0.0029174` 實質重疊）。
- **`OLMoE`**：`NL - B` 平均 $\Delta\text{NLL} = +0.00235975$，simultaneous 95% 信賴區間 $[-0.00464790,\ +0.00936740] < \ln(1.02)$；描述性 `mean(NL - C) = -0.00232812` nats（在已曝光 `PG16` 上平均 NLL 甚至略低於 `C8` 的 `+0.00468787`）。

**直接科學含義**：
1. 在 8 個算子的局部替換尺度下，一旦給定了層共享低秩投影 `P128` 與 `C32` 分區偏置（局部工作點平移），**保留原生 `SwiGLU ∘ Down` 非線性結構並僅使用單一全域 `Gate/Up` 斜率，即可達到與 32 套葉專屬線性輸出斜率 `A_{e,c}` 幾乎不可區分的語言模型品質**。
2. 因此，`C` **不能**再主張「因為 MoE 專家非線性極強，任何全域低秩投影都會崩潰，唯獨分區局部線性斜率能保住品質」。`C` 剩餘的可辯護價值，已從「唯一可行的函數近似空間」收斂為**「將 `Gate + Up + SwiGLU + Down` 摺疊成單一固定 `256 KiB`（`16 × 16 KiB` 頁）整數矩陣、讓同層 $K$ 個選中專家共享單一 `128 B` `A8` 輸入並消除中間 `I` 維非線性屏障的執行與快取粒度優勢」**。

### 1.2 為何絕對不能據此自動宣布 Round 2 失敗（四項嚴格邊界）
1. **極低曝光稀釋與開發集重用（Exposure & Development Boundary）**：`C8` 與 `NL8` 僅替換每模型 2 層各 2 個專家（`DeepSeek` `3,599` 列、`OLMoE` `4,770` 列，僅佔全模型路由槽位約 `0.28%` 與 `0.46%`），且 `NL8` 是在 `C8` 已使用過的同一批 `PG16` 上進行 `development_only` 評估（[`Root_DEV_QUALITY_independent_statistics_audit.json`](file:///home/ericyu/expert_flash_experiment/results/research-2026-11-03-no-cr/round02-local-affine/stage3-control-cost/Root_DEV_QUALITY_independent_statistics_audit.json#L76-L80)）。兩者在 `>99.5%` 算子維持精確時皆低於 `ln(1.02)`，既不能線性外推至全模型，也不能證明兩者在全層 `64` 專家（含長尾專家與跨專家誤差疊加）下仍然等價。
2. **全模型 `8 GB`（`8e9 B` 與 `8 GiB`）容量與 I/O 契約上的結構性不對稱**：
   - `C32/r128` 並非靜態總容量壓縮器（每專家 `32` 葉共 `8.39 MB` `W8` 權重 + `0.52 MB` `FP32` 縮放與偏置 = `8.91 MB`，單層 `64` 專家約 `0.57 GB`；全模型路由銀行在 `OLMoE` 16 層為 `9.13 GB`、`DeepSeek` 26 層為 `14.83 GB`），而是**主動工作集（active working-set）與單次讀取頁數編譯器**：每次命中或未命中僅需存取 **單一葉 `256 KiB` `W8`（`16` 個 `16 KiB` 頁）+ `16 KiB` 縮放／偏置**。
   - 反觀 `NL` 在 `DeepSeek` 全模型（`1,664` 個路由專家）下，僅常駐 `FP32` 全域 `GU` 斜率（`2.40 GB`）+ `C32` 雙偏置（`1.04 GB`）+ `P128`（`0.03 GB`）就達 **`3.46 GB`**，加上 `4.81 GB` 原始 `W8 Down` 與至少 `~1.53 GB` 非路由層／shared／dense／KV 開銷，總常駐需求 **`>= 9.80 GB`**，**無法全部放入 `8e9 B`（`8.00 GB`）或 `8 GiB`（`8.59 GB`）主機記憶體**；一旦將 `NL` 的 `Down` 溢位至 Flash，每次未命中每專家需讀取 `2.88 MB`（**`176` 個 `16 KiB` 頁，為 `C` `16` 頁的 `11×`**），且帶有 **6 個私有 `1,408 B` `A8` 輸入向量**、**`Host GU+SiLU → Flash Down` 序列屏障**，主機可用於快取熱專家的空間還比 `C` 少約 `3.4 GB`。因此 `NL` 並未在目標資源上限下對 `C` 建立品質與成本雙重支配（Pareto dominance）。
3. **Host `BENCH` 形狀代理的實測方向相反**：在 `EPYC 9354` 單層 `full_K` warm shape proxy 中，`C`（`153–211 us`）快於 `D`（`306–443 us`）與 `NL`（`323–336 us`）約 **`1.6–2.1×`**。雖須嚴格註記 `NL` 與 `B/C/D` 分屬不同測試 phase（warm-cache 歷史不同）、僅以 2 個專家模板輪替填入 $K$ 個獨立記憶體槽位，且 `EPYC 9354`（`512 MiB` L3）並非目標 `i9-14900KS`（`36 MiB` L3），完全不能宣稱全模型 `>= 10%` 加速；但這足以說明 **在主機計算形狀上，`NL` 與 `D` 目前並未淘汰 `C`**。
4. **科學失敗判定準則**：強控制組在低曝光開發集上同樣達標，代表「機制不可替代性尚未獲證」，屬於**未決必要性問題**，而非預註冊候選假說遭到實證擊穿。

---

## 2. `C-first64` 是否為下一個真正必要、可證偽、不浪費的實驗？還是存在更便宜的前置 source/analytic 關卡？

### 2.1 在「模型擬合與品質檢定」層級，`C-first64` 是最小且誠實的可證偽實驗
若團隊決定繼續推進 Round 2 的實證品質驗證，[`earliest64_Cfirst_prospective_stage.sol.md`](file:///home/ericyu/expert_flash_experiment/results/research-2026-11-03-no-cr/round02-local-affine/earliest64_Cfirst_prospective_stage.sol.md#L7-L15) 提出的 **`C-first64`（僅編譯最早 MoE 層 `Deep L1` 與 `OLM L0` 全部 64 個路由專家，先測 `C64` 必要品質，通過後才支付 `NL64/D64`）** 確實是最小不浪費設計：
1. **消除層內稀釋與檢驗長尾專家**：8-op 在單層僅覆蓋 `2/64 = 3.125%` 專家；唯有同層 64 個專家全數編譯，每個 token 的全部 $K=6$ 或 $K=8$ 個路由槽位才會 **100% 由近似算子同時輸出並疊加**，正面檢驗：(i) 單一層共享 `(tree32, P128)` 能否支撐同層全部 64 個異質與長尾專家；(ii) 固定單一 `eta=0, lambda=0.01` 配方能否零調參遷移至 `124` 個未見專家；(iii) 同層跨專家近似誤差在加總後是否相長放大。
2. **零上游狀態混雜**：最早 MoE 層保證在相同完整 GT 歷史與執行設定下，`source`、`B64`、`C64` 的層輸入 `X`、路由 `IDs` 與 `final_alpha` 完全一致，且輸出誤差會真實穿過後續全部 `25` 層（`DeepSeek`）或 `15` 層（`OLMoE`）網路。
3. **精確止損（不預付控制組成本）**：先只支付 `124` 個新專家的 `30.08 GB` `Y` 與 `27.43T` 整數 MAC（[`earliest64_Cfirst_prospective_stage.sol.md`](file:///home/ericyu/expert_flash_experiment/results/research-2026-11-03-no-cr/round02-local-affine/earliest64_Cfirst_prospective_stage.sol.md#L55-L61)），暫緩 `NL64` 額外的 `18.29T` `GU` 查詢 MAC、`35.72 GB` 原始 `GU` 儲存與 `9.15T` `Down` 校準 MAC。一旦 `C64` 在新鮮驗證書上出現有界品質失敗（simultaneous 95% 下界 `> ln 1.02`），即可乾淨止損，省下近半算力與過半磁碟寫入。

### 2.2 但在投入 `27.43T` MAC 之前，存在兩個「零擬合、零新數據」的前置 source/analytic 關卡應先閉合
在核准任何 `first64` 工作負載前，建議先用純解析與既有程式碼審計閉合以下兩個零算力問題：

1. **前置解析關卡 A：`C_inflash`（依 [`aif_r128_16lane_construction.sol.md`](file:///home/ericyu/expert_flash_experiment/results/research-2026-11-03-no-cr/round02-local-affine/aif_r128_16lane_construction.sol.md#L50-L71)）對決同銀行主機基準 `C_Cache_DMA_CPU` 的延遲窗口交叉審計**
   - 依 [`aif_r128_16lane_construction.sol.md`](file:///home/ericyu/expert_flash_experiment/results/research-2026-11-03-no-cr/round02-local-affine/aif_r128_16lane_construction.sol.md#L62-L68) 的明確 `SenseReady-before-PE`、無跨波重疊構造，即便完全忽略 `POST_Q`、`ECC`（名目 `10.24 us` 尚需未證實的 `6.4 GB/s` 有效 PB/ECC/staging 服務率）、64 次 pipeline fill 與每波 4 個 plane 的序列 `512 B` FIFO drain，單波理想下界已為 `28 + 20.48 = 48.48 us`，**雙波忙碌晶片的理想下界已達 `>= 96.96 us`**；若計入名目 `10.24 us` ECC（`58.72 us`/波），僅兩波 `sense + ECC + PE` 即達 **`117.44 us`**，尚不含主機端 `tree + P128`（註：深度 5 的樹只需 5 次 `H` 維內積即可確定葉索引 `c` 並提前發出 `PREP_READ`，早於 `P128` 的 128 次內積，但 `POST_Q` 仍須等 `P128` 完成）、序列 `Q_i` 寫入、雙波共 8 次跨晶片序列 plane FIFO drain 與最終 rescale/alpha 累加。
   - 對照剛取得的 `EPYC` warm `full_K` `C` 純主機代理（已包含 `tree + P128 + A8 + K 個專家矩陣乘 + rescale/bias/alpha` 全部工作）：**`153–211 us`**；且根據 [`aif_r128_16lane_construction.sol.md`](file:///home/ericyu/expert_flash_experiment/results/research-2026-11-03-no-cr/round02-local-affine/aif_r128_16lane_construction.sol.md#L70) 的公平條款，最強同銀行主機基準 `C_Cache_DMA_CPU` 享有**完全相同的早期 `PREP_READ` 地址、相同的 `8 GB` 主機 `(expert, leaf)` `256 KiB` 熱快取預算，以及普通讀取邊 DMA 邊跑 CPU 整數 GEMV 的能力**。
   - **關卡意義**：這項零成本交叉檢查**不會否定 `C` 作為「主機細粒度快取 + 普通 NVMe DMA 局部仿射編譯器」的潛力**（因 `C_host` 在主機形狀上仍比 `D` 與 `NL` 快約 `1.6–2.1×`），但它清楚揭示：**自訂 `16-lane` In-Flash PE 相對 `C_Cache_DMA_CPU` 的淨勝出窗口極窄甚至可能為負**。Root 與 Critic 應在花費 `27.43T` MAC 前先做策略確認：若最終發現 `C` 的真實價值主要來自「純軟體／主機 `256 KiB` 葉快取 + 普通頁讀取」而非自訂 NAND PE，團隊是否仍認同此研究方向？

2. **前置原始碼關卡 B：最早層 `final_alpha` 提取語義與 `all-x` 葉支援度無退化驗證**
   - 僅透過靜態原始碼審計（零模型執行），確認 `DeepSeek`（`norm_top_k` + `routed_scaling_factor`）與 `OLMoE` 在最早 MoE 層 hook 處擷取與回寫的 `final actual multiply-alpha` 完全對齊，並鎖死下方第 3 節列出的 5 項協定細節缺口。

---

## 3. 固定配方、數據邊界、Earliest-Input 定理、Held-Out 切分與資源上限的現存缺口

審視 [`earliest64_Cfirst_prospective_stage.sol.md`](file:///home/ericyu/expert_flash_experiment/results/research-2026-11-03-no-cr/round02-local-affine/earliest64_Cfirst_prospective_stage.sol.md#L17-L76)，整體骨架嚴謹，但在凍結成正式執行協定前仍須補齊以下 **5 個具體缺口**（全程**不擴充任何網格**）：

1. **缺口 1（固定配方與 Guard 邊界）：嚴禁將長尾專家的「自然路由命中數為 0」誤當作不支援葉而觸發 Canonical Fallback**
   - 在 [`earliest64_Cfirst_prospective_stage.sol.md`](file:///home/ericyu/expert_flash_experiment/results/research-2026-11-03-no-cr/round02-local-affine/earliest64_Cfirst_prospective_stage.sol.md#L21) 與 [#L31](file:///home/ericyu/expert_flash_experiment/results/research-2026-11-03-no-cr/round02-local-affine/earliest64_Cfirst_prospective_stage.sol.md#L31) 中提到沿用層級或專家級支援／guard metadata，並警告 guard-heavy fallback 會造成假性品質通過。
   - 由於 `C64` 採用**層共享 `(tree32, P128)` + 無權重 `all-x` 擬合**，全部 64 個專家在每個葉 $c$ 的擬合設計樣本數 $N_c$ 完全相同（在既有 `29,629` / `29,593` 列下 32 葉皆非空滿秩）；但長尾專家在某些葉上的**自然路由命中數** $N_{e,c}^{\text{routed}}$ 可能為 0 或極低。協定必須明確釘死：在 `guard=100` 下，**唯有當 `all-x` 擬合樣本數 $N_c = 0$ 或擬合權重出現非有限值時才允許觸發 fallback**；$N_{e,c}^{\text{routed}}$ 僅供審計記錄，嚴禁觸發推論時退回 `B64`。同時應在 manifest 中載明新專家固定選 `eta=0, lambda=0.01`（同時保留舊 `OLM L0.E17` `lambda=0.001` 原封不動）的先驗決定規則。

2. **缺口 2（延遲控制組配方）：`D64` 的精確正規化映射尚未無歧義閉合，且缺一個固定無網格的「層共享輸出基底」參照點**
   - [`earliest64_Cfirst_prospective_stage.sol.md`](file:///home/ericyu/expert_flash_experiment/results/research-2026-11-03-no-cr/round02-local-affine/earliest64_Cfirst_prospective_stage.sol.md#L47-L49) 提議新專家預設採用 `PE-RRR q512/W8/host-floating-latent/identity-gauge` 並使用新專家固定 `C` 的 `eta/lambda` 與 `Y` 動差。必須在拆封新 16 本書之前逐行鎖死：(i) Root 究竟選定 `PE-RRR` 還是統一全域輸入 `G-RRR` 作為唯一預設；(ii) `PE-RRR` 從 32 個局部斜率提取 `q=512` 專家子空間 $U_e$ 與重擬合時，其確切權重／懲罰公式是否與早期通過的 4 個 `D` 算子（`Deep L1.{E14,E40}`, `OLM L0.{E6,E17}`）逐位元同構。
   - 此外，`PE-RRR` 每專家仍各自持有獨立的 `2048 × 512` 輸出基底 $U_e$（$K$ 個專家需搬移並乘上 $K$ 份 $U_e$，正是 `D` 在主機慢至 `306–443 us` 的瓶頸），未能代表 `MoLAE`（同層共享 **1 個**輸出提升基底 $U_{\text{layer}}$）這類強先例。若未來需要在延遲控制階段釐清「層共享輸出基底能否追平 `C`」，應在拆封新 16 本書前就預先釘死**單一固定無網格的層共享輸出基底控制組配方**（並強制實測 `A8/FP32` 下的實際品質，不假設代數合併恆等），避免事後補選控制組。

3. **缺口 3（Earliest-Input 定理邊界）：需將逐位置三元組位元組相等列為硬性有效性門檻，並補足自回歸生成盲區**
   - 最早層輸入不變性定理僅在**相同完整 token 歷史、相同 `4 × 128` 批次切分、每次獨立清空 KV cache、相同後端與執行緒設定**下成立（[`earliest64_Cfirst_prospective_stage.sol.md`](file:///home/ericyu/expert_flash_experiment/results/research-2026-11-03-no-cr/round02-local-affine/earliest64_Cfirst_prospective_stage.sol.md#L27-L30)），既不跨不同批次形狀成立，也不在自由生成（一旦某步預測 token 分岔，下一步進入 `L1/L0` 的輸入即不同）下成立。
   - 因此應明確規定：(i) `source`、`B64`、`C64` 在新 16 本書全部 `512 × 16 = 8,192` 個 GT 位置上的 `(ffn_norm F32 bytes, ordered I32 IDs, final actual multiply-alpha F32 bytes)` **完全一致（0 mismatch）** 為硬性數值有效性門檻；(ii) 除主檢定 `GT512` NLL 外，建議附加固定短視窗（例如每書自固定切點起 `<= 32` 個 EOG-stopped token）自由生成退化診斷（作為預先宣告的次要健全性紀錄，不納入主檢定 bootstrap 家族、亦不用於調參），以捕捉 teacher-forcing NLL 看不見的自回歸重複或崩潰。

4. **缺口 4（Held-Out 驗證池預算）：須核實 `PG-19 validation` 剩餘未用書本存量，防止單層階段耗盡後續全層開發／驗證池**
   - 官方 `PG-19 validation` 總數僅 50 本；先前已曝光 `16` 本，若 `first64-v1` 再拆封 `16` 本（[`earliest64_Cfirst_prospective_stage.sol.md`](file:///home/ericyu/expert_flash_experiment/results/research-2026-11-03-no-cr/round02-local-affine/earliest64_Cfirst_prospective_stage.sol.md#L37-L41)），剩餘未曝光驗證書最多僅剩 `18` 本（尚須扣除長度不足或不可用書本），而之後若進入全 42 層階段仍需乾淨驗證集，且密封的 `PG test32` 與 `TEST100` 絕不可提前動用。
   - 在拆封這 16 本書之前，必須先以純 metadata 核實剩餘合規驗證書總數與未來全層階段的驗證切分預留計畫；同時完全支持第 41 行的保守統計標記——一旦新 16 本書為 `C64` 拆封，後續在其上重跑的 `D64/NL64` 必須一律標記為 **`secondary / development`**，不得聲稱為獨立新鮮驗證證據。

5. **缺口 5（磁碟峰值與全系統記憶體上限）：本地磁碟下「串流偏置校準、禁止落地全量預測快取」必須由建議升級為強制約束**
   - 本地可用空間為 `162 GB`（[`earliest64_Cfirst_prospective_stage.sol.md`](file:///home/ericyu/expert_flash_experiment/results/research-2026-11-03-no-cr/round02-local-affine/earliest64_Cfirst_prospective_stage.sol.md#L63-L68)）。第一階段 `C64` 增量上限 `64 GB`（`30.08 GB` 新 `Y` + 模型與收據）可安全保留約 `98 GB > 32 GB` 餘裕；但若進入延遲控制階段，僅保留原始 `Y`（`30.08 GB`）與原始 `GU`（`35.72 GB`）即已佔用 `65.80 GB`，若再落地 `65.80 GB` 預測快取將達 `131.60 GB`，直接跌破 `32 GB` 安全保留線。
   - 因此，若在本地磁碟執行延遲控制組，「不落地 `GU/Down` 全量預測快取、改以單專家批次串流執行與既有確定性順序完全同構的 `FP64` 殘差均值累積，且以分塊／單專家粒度寫入避免 `30+ GB` 雙倍暫存檔峰值」**必須列為硬性強制條件（mandatory）**，而非僅是 `prefer`；同時單層 `0.57 GB` 銀行評估絕不可外推宣稱滿足全模型 `8e9 B` / `8 GiB` 常駐記憶體上限。

---

## 4. 真正具備投稿價值的論文需要什麼具體機制與最強基準證據？

一篇站得住腳的頂會系統／機器學習論文，**不能**建立在「`256 KiB < 8.65 MB` 活躍位元組壓縮比」、單層 `1/26` 或 `1/16` NLL、2 模板輪替的 `EPYC` 形狀代理，或未經實測的原生 AiF PE 假設之上，而需要同時具備以下三項實證支柱：

1. **全路由層（`DeepSeek` 26 層 1,664 專家、`OLMoE` 16 層 1,024 專家）端到端密封集品質閉合**
   - 在密封的 `PG-19 test32`（連同預先凍結的下游零樣本評估與真實自由生成審計）上，全路由層同時替換的 `C_full` 維持在預設相對困惑度容忍帶內，完整揭露 `C - B`、`B - source` 與 `C - source`，且不依賴高比例 canonical fallback 掩護。

2. **在嚴格閉合的 `8e9 B` 與 `8 GiB` 全系統記憶體上限下，對最強零改硬體與共享基底基準展現真實帕累托勝出**
   - 全系統記憶體帳本必須計入 non-FFN 權重、`DeepSeek` shared/dense 層、KV cache、執行期緩衝與作業系統開銷，並在**目標級硬體與真實冷熱路由軌跡**上正面擊敗：
     - **原始／量化最強混合基準**：`OLMoE` 上的全主機常駐基準 `B_AllHost`（測試 `C` 的細粒度活躍讀取能否在 CPU 記憶體頻寬與計算上實質勝出），以及 `DeepSeek` 上的 `B_FlashGU_HostDown` 與主機熱專家快取 + 普通讀取 DMA 基準 `B_HybridCache`。
     - **同座標系低秩／非線性強基準**：全層 `D`（`PE-RRR` / `G-RRR`）、全層 `MoLAE`/`EBC`（層共享或分群共享輸出提升基底）與全層 `NL`（檢驗全層誤差累積下 `C` 是否優於共享基底控制組，或憑 `256 KiB` 單階段無屏障執行與更低常駐 `FP32` 開銷在 `8 GB` 上限下實測擊敗 `NL`）。

3. **軟體編譯貢獻與自訂硬體貢獻的誠實切割**
   - **若定位為純軟體／系統 MoE 編譯器（`C_Cache_DMA_CPU`）**：直接以「主機 `8 GB` 細粒度 `(expert, leaf)` `256 KiB` 快取 + 未修改 NVMe 普通頁讀取 DMA + 主機整數 GEMV」在真實單節點上證明端到端 B1 token 延遲顯著勝出，完全不必捆綁任何未驗證的 NAND 內部電路假設。
   - **若額外主張自訂 In-Flash 架構（如 [`aif_r128_16lane_construction.sol.md`](file:///home/ericyu/expert_flash_experiment/results/research-2026-11-03-no-cr/round02-local-affine/aif_r128_16lane_construction.sol.md#L5-L75)）**：除了必須在給定 `C_Cache_DMA_CPU` 完全相同早期 `PREP_READ` 地址、快取預算與普通讀取重疊能力下仍證明具備可觀端到端淨加速外，還必須提供公開製程（open PDK）下實作該新增區塊（`16` 乘法樹 + `I32` 累加器、`128 B` RF、`1 KiB` staging、`512 B` FIFO、標籤控制與匯流排 mux）的 **RTL + STA（`200 MHz` 時序收斂）+ 完整面積／功耗報告**，或誠實呈現條件式損益平衡包絡，絕不宣稱未實測的「微小 PPA 代價」或原生 AiF 相容性。

---

## 5. 直言先例距離評估 vs. 科學實驗失敗計數，與有限下一步建議

### 5.1 直言不諱的先例距離審視（Blunt Prior-Art Assessment）
必須坦白指出：**`C` 目前與既有公開文獻之間的純演算法距離相當薄（thin novelty margin）**：
- **`Flash-Jacobian`（anonymous ACL OpenReview）** 已經提出對上下文隱藏狀態做區域分群、局部中心化 $(x - \bar{x}_c)$，並以局部脊回歸擬合低秩仿射映射 $\bar{y}_c + A_c B_c (x - \bar{x}_c)$ 來近似並替換 Transformer MLP/FFN 區塊。
- **`MoLAE`（`2503.23100v2`）與 `EBC`** 已經提出在同層或同群 MoE 專家之間共享輸入投影基底與輸出提升基底。
- `C` 的實際創新點並非局部仿射近似本身，而是**在 MoE 層級強制全部 64 個專家共用同一棵遞迴 PCA 樹 `tree32` 與同一組輸入投影 `P128`，使每一個 `(expert, leaf)` 非線性 SwiGLU 專家摺疊成固定 `256 KiB`（`16 × 16 KiB` 頁）的單階段整數映射，讓被選中的 $K$ 個專家共用同一個葉索引 $c$ 與同一個 `128 B` `A8` 輸入向量**。
- 這是一項乾淨且具工程巧思的**「面向頁對齊與細粒度快取的系統綜合（systems-aligned synthesis）」**，但並非全新的數學近似機制。再加上 `NL8` 顯示在低曝光下「葉專屬斜率」對比「全域 `GU` 斜率 + 葉偏置 + 原生 `SwiGLU`」並無品質獨佔性，且 [`aif_r128_16lane_construction.sol.md`](file:///home/ericyu/expert_flash_experiment/results/research-2026-11-03-no-cr/round02-local-affine/aif_r128_16lane_construction.sol.md#L68-L70) 的 `>= 96.96 us` 雙波晶片下界對照主機 `153–211 us` 壓縮了自訂硬體的邊際收益，這個方向的**投稿新穎度風險確實偏高**。

### 5.2 嚴格區分「先例新穎度／策略停損判斷」與「可計數的科學實驗失敗」
這兩個概念在研究治理上**必須嚴格分開**：
1. **先例距離與策略取捨（Strategic / Novelty Disposition）**：先例重疊度高或自訂硬體窗口狹窄，屬於**研究組合與論文定位的策略評估**。Root 若認定「`Flash-Jacobian` + 層共享 MoE 分區 + 主機細粒度快取」即使實驗成功也難以達到目標投稿的新穎度高度，完全有充分理由做**策略性轉向或停損**——但這**不計入**競賽的「科學實驗失敗次數」。
2. **可計數的科學實驗失敗（Countable Scientific Failure，目前仍維持先前 Round 1 的 `1` 次失敗、`9` 次剩餘機會）**：唯有當一個**預先凍結、數值與統計完整性閉合的實證實驗**執行完畢，且候選方法在預註冊判準下遭到實證結果正面證偽——例如若未來獲准執行 `first64-v1`，在新鮮 16 本驗證書上 `C64 - B64` 的有效 simultaneous 95% 信賴區間**下界 `> ln(1.02)`**（有界品質失敗），或在嚴格閉合的同資源條件下被無須局部斜率的強基準在品質與延遲上全面支配，並經 Root 與 Critic 正式裁定——才會構成一次可計數的科學實驗失敗。目前 Stage 3 的 `NL8` 與 `BENCH` 結果**不構成任何科學實驗失敗**（[`Root_DEV_QUALITY_independent_statistics_audit.json`](file:///home/ericyu/expert_flash_experiment/results/research-2026-11-03-no-cr/round02-local-affine/stage3-control-cost/Root_DEV_QUALITY_independent_statistics_audit.json#L79-L80)）。

### 5.3 有限、可測試的下一步建議（不擴網格、不聯絡作者、不授權 `first64`）
1. **第一優先（零算力策略與解析裁決，建議先於任何 `27.43T` MAC 查詢）**：由 Root 與 Critic 先對照 `C_Cache_DMA_CPU` 與 [`aif_r128_16lane_construction.sol.md`](file:///home/ericyu/expert_flash_experiment/results/research-2026-11-03-no-cr/round02-local-affine/aif_r128_16lane_construction.sol.md#L66-L74) 的雙波排程包絡，明確拍板：**若不依賴自訂 In-Flash PE、而是以「主機 `8 GB` 細粒度 `(expert, leaf)` 快取 + 普通頁 DMA + 單階段局部仿射編譯」作為核心系統貢獻，在扣除 `Flash-Jacobian` / `MoLAE` 先例後是否仍值得投入 `first64` 成本？**
2. **第二優先（僅當第 1 步拍板繼續推進 Round 2 時——在凍結 `first64-v1` 前補齊 4 項有限條款）**：
   - **(a)** 鎖死 `guard=100` 僅依層共享 `all-x` 樣本數 $N_c$ 判定支援度，禁止長尾專家因自然路由計數 0 而退回 `B64`，並載明固定 `eta=0, lambda=0.01` 的預設依據（零網格擴張）；
   - **(b)** 在拆封新 16 本書前鎖死延遲控制組 `D64`（`PE-RRR` 或統一 `G-RRR`）的唯一無分支公式（可選擇一併預先釘死一個固定無網格的層共享輸出基底控制組），將後續在該 16 本書上的控制組重播明確標記為 **`secondary / development`**，並以純 metadata 確認剩餘未曝光 `PG-19 validation` 書數足以留給未來全層階段；
   - **(c)** 將 `source / B64 / C64` 在 `GT512` 全部位置上的 `(ffn_norm X, IDs, final_alpha)` 0-mismatch 列為硬性有效性門檻，並附加固定短視窗 EOG-stopped 自由生成次要健全性診斷；
   - **(d)** 若使用本地磁碟執行後續控制組，將「禁止落地 `GU/Down` 全量預測快取、強制單專家批次串流 `FP64` 殘差均值累積」列為硬性約束，嚴守 `>= 32 GB` 實際可用空間保留線。
3. **權限邊界**：本審稿文件僅提供理論與協定審查意見，**不授權啟動 `first64` 實作、教師查詢、控制組擬合或拆封任何驗證／測試資料**。
