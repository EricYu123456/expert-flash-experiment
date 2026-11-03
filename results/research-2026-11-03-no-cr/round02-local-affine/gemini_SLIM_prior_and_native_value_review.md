# SLIM 先驗、既有文獻邊界與 Native In-Flash 價值審核

- **回合與身分**：同一 `Round 2`（`layer-shared-input-partition-local-affine-expert-compiler`），`gemini-3.8-flash-high` 理論與來源審查。`newmethodclock = 0`。
- **邊界宣告**：`C64 FIT` 正在背景運行；**不讀取、不修改**任何模型權重、陣列、快取、標籤、原始書籍、品質評估或運行中輸出；**不執行**候選程式碼、測試、建置、硬體模擬（sim/RTL）或外部通訊。不新增超參數網格或新方法。
- **審核依據**：[`docs/slim_aif_baseline_source_audit_terra.md`](file:///home/ericyu/expert_flash_experiment/docs/slim_aif_baseline_source_audit_terra.md)、[`docs/novelty_followup_terra.md`](file:///home/ericyu/expert_flash_experiment/docs/novelty_followup_terra.md)、[`critic_gemini_stage3_C64_review.20261103T081116Z.json`](file:///home/ericyu/expert_flash_experiment/results/research-2026-11-03-no-cr/round02-local-affine/critic_gemini_stage3_C64_review.20261103T081116Z.json) 與 [`actual64_ordinary_cpu_deployment_path.sol.md`](file:///home/ericyu/expert_flash_experiment/results/research-2026-11-03-no-cr/round02-local-affine/actual64_ordinary_cpu_deployment_path.sol.md)。

---

## 1. 既有先驗（Flash-Jacobian / EBC / MoLAE / SLIM）已涵蓋的貢獻與剩餘的窄系統問題

### 1.1 既有文獻已明確涵蓋的技術點
1. **輸入區域分群與局部低秩仿射替代**：[Flash-Jacobian（Zenodo 19069517/19150764 / OpenReview au5FGRgwcZ）](https://openreview.net/attachment?id=au5FGRgwcZ&name=pdf) 已明確提出依隱藏狀態區域分群、中心化 $(x - \bar{x}_c)$，並以局部脊回歸擬合低秩仿射算子 $\bar{y}_c + A_c B_c (x - \bar{x}_c)$ 取代 Transformer MLP/FFN。
2. **MoE 專家全域線性化與群共享轉換**：[EBC（commit aa323a40）](https://github.com/danishberg/Expert-Bank-Compression/tree/aa323a40cd0ecd280102fa29377ec816177f4a6c) 已實作將非線性 MoE 專家擬合為全輸入線性矩陣 $W_e$，並在專家群（expert clusters）間共享正交投影與低秩分解基底 $(U_c, V_c)$。
3. **MoE 層共享輸入與輸出基底**：[MoLAE（arXiv:2503.23100v2）](https://arxiv.org/abs/2503.23100) 已確立在層級共享單一輸入降維投影與輸出提升基底。
4. **NAND 存放 MoE 權重、低秩預測取用與近儲存 GEMM**：[SLIM（arXiv:2507.09201v1）](https://arxiv.org/html/2507.09201v1) §3.4、§5.2 與 §5.4 已明確提出將 FFN/MoE 權重置於 NAND、以低秩預測器（$|X L R| > \text{threshold}$）動態決定取用神經元、在晶片附近（die-level PE）進行數位 GEMM，並實作頁對齊向量打包（page-aligned packing）與 LPA-to-buffer 映射以避免冗餘讀取。

### 1.2 尚未被證明、亦未被先驗完全涵蓋的窄系統問題
上述先驗**未曾提出亦未曾證明**：是否能透過**全層 64 個異質 MoE 專家共用單一遞迴 PCA 樹 `tree32` 與單一 `P128` 投影坐標系**，將每個 `(expert, leaf)` SwiGLU 專家摺疊成固定 $256\text{ KiB}$（$16 \times 16\text{ KiB}$ 頁）的單階段整數矩陣，使得被選中的 $K$ 個專家共用同一個葉索引 $c$ 與單一 $128\text{ B}$ $A8$ 輸入向量；且此種設計在真實驗證集上能否維持語言模型品質，並在真實頁對齊儲存與快取層級中展現相對於同功能主機基準與共享輸出基底控制組的淨延遲／頻寬優勢。

---

## 2. 若 C64 最早層必要品質通過，證明 Native NoCR In-Flash 價值所需的最少有界證據

若 `first64-v1`（`Deep L1` 與 `OLM L0` 各 64 專家）在新鮮 16 本驗證書上通過必要品質檢定（雙模型 `C64 - B64` 95% 信賴區間上界均低於 $\ln(1.02)$），要證明自訂 In-Flash PE 的原生價值，**最少且必須閉合**以下兩組同等條件下的有界實證：

### 2.1 對決同函數主機快取與普通讀取基準（`C_Cache_DMA_CPU`）
依據 [`actual64_ordinary_cpu_deployment_path.sol.md`](file:///home/ericyu/expert_flash_experiment/results/research-2026-11-03-no-cr/round02-local-affine/actual64_ordinary_cpu_deployment_path.sol.md) 與 Critic 訂定的公平計帳規則，兩者必須在**相同未命中數 $m \in [0, K]$、相同主機快取預算 $M_{\text{cache}} = M_{\text{total}} - M_{\text{aux}} - M_{\text{peak\_workspace}}$、相同副本 placement 與相同的早期預取資訊（樹深度 5 僅需 5 次內積即可在 $P128$ 算完前確認葉地址並發送讀取）**下比較：
- **當 $m = 0$（全部命中）**：兩者皆執行相同的 CPU-hit 策略，In-Flash PE 無加速空間。
- **當 $m \ge 1$（存在未命中）**：PE 將每個未命中專家的搬移量從 $262,144\text{ B}$（W8 權重）降至 $8,192\text{ B}$（I32 結果）（**$32\times$ 數據減量**），但須支付輸入向量廣播（$128\text{ B} \times \text{chips}$）、序列波次、ECC 延遲與 FIFO 匯流排排空。
- **最少證據要求**：提供一個可行的非同步普通讀取上限排程 $R_{\text{load}}(m)$（重疊 $P128$、CPU-hit GEMV 與普通 DMA，接續 CPU-miss GEMV），並證明在實際運作未命中率下，經過嚴格時序驗證的 PE 服務包絡（含真實 sense、ECC、PE 執行與 FIFO 傳輸）之總延遲確實顯著優於該普通讀取排程，且其硬體開銷不違反熱與面積預算。

### 2.2 對決實際四捨五入共享輸出控制組（`SO64_pre`，秩 $q=512$）
依據 Critic 審查意見，凍結的 `SO64_pre` 控制組（固定 $q=512$、W8、共享 $P128$、32 葉偏置、層共享輸出提升矩陣 $U_{\text{layer}} \in \mathbb{R}^{H \times q}$）：
- 領先乘加運算量在 $K=6$ 時為 $1.44\text{M}$（$C$ 為 $1.57\text{M}$），在 $K=8$ 時為 $1.57\text{M}$（$C$ 為 $2.10\text{M}$）；全層 64 專家權重僅約 $156\text{ MB}$（$C$ 約 $570\text{ MB}$，小了 $3.65\times$）。
- **最少證據要求**：在考慮實際 `final_alpha` FP32 潛在變數預先加權合併（嚴格按照加權 slot hook 處理捨入，不除以 alpha）的數值條件下，實測證明 $C64$ 的全域生成品質顯著優於 `SO64_pre`，或者實測證明兩階段低秩映射的中間 $q$ 維存取開銷使得其端到端延遲劣於 $C$ 的單階段整數執行。

---

## 3. 現有證據是否已能嚴格否決（Rigorously Veto）此有界品質階段？

**結論：現有證據絕不足以、亦不能在邏輯上嚴格否決當前進行中的 `C64` 必要品質階段。**

理由如下：
1. **未解決的硬體可行性 $\neq$ 已證實的科學失敗**：正如 Terra 在 [`docs/slim_aif_baseline_source_audit_terra.md`](file:///home/ericyu/expert_flash_experiment/docs/slim_aif_baseline_source_audit_terra.md) 所指出，公開文獻（SLIM 與 AiF-like 復現）未提供原生 AiF PE 的實體測試數據、短 row reset 或確定性 ECC 服務率；但硬體微架構證據的缺失代表**可行性尚未閉合（unresolved feasibility）**，絕非數學矛盾或負面實證，不能藉此捏造為演算法品質失敗。
2. **混淆了單一構造下界與可行基準上限**：先前推導的 $96.96\,\mu\text{s}$ 僅是特定「SenseReady-before-PE、無跨波重疊」串行構造在雙波忙碌時的**理想下界（lower bound）**，而非所有 In-Flash 架構的上限；將其與 EPYC 純記憶體常駐測試（$153–211\,\mu\text{s}$）對比，既未包含未命中時普通讀取的 DMA 延遲，亦未構成嚴格證明。要嚴格否決 PE，必須提供一個在所有合法未命中數 $m$ 下、皆嚴格優於所有可行 PE 排程的**可行普通基準上限（feasible ordinary upper bound）**；此類證明目前並不存在。
3. **品質檢定的獨立性與必要性**：正在進行的 `C64 FIT` 與即將到來的新鮮驗證集評估，檢驗的是「單一分區能否支撐 64 個異質專家同時替代」這一基本數值假說。若品質檢定出現有界失敗（95% 下界 $> \ln(1.02)$），該候選方案在演算法層面即被證偽，省下後續硬體驗證成本；若品質通過，才需要進一步衡量硬體與控制組。現有先驗與分析完全無法預先預測該統計檢定結果，因此絕無理由提前否決。

---

## 4. 修正過度主張與研究邊界確認

嚴格落實 Critic 審查意見（[`critic_gemini_stage3_C64_review.20261103T081116Z.json`](file:///home/ericyu/expert_flash_experiment/results/research-2026-11-03-no-cr/round02-local-affine/critic_gemini_stage3_C64_review.20261103T081116Z.json)），收回並修正前一版筆記中的不嚴謹表述：
1. **收回品質等價性斷言**：`C8` 與 `NL8` 分別通過各自的局部開發集評估，僅代表兩者在極低曝光（$<0.5\%$ 槽位）下未達損害門檻，**不代表兩者統計等價、品質無差異或已完成完整路由覆蓋**。
2. **收回強制性容量下限結論**：先前稱 `NL >= 9.80 GB` 因而無法放入主機記憶體，依賴了未閉合的非路由層常駐假設；完整的容量計算必須在扣除 $P$、樹、縮放、偏置、KV cache 與執行期峰值工作空間後，精確計算可用快取空間。
3. **收回 42 層全替換作為論文發表充分必要條件（iff）的主張**：全模型 42 層替換僅是未來的潛在評估範疇，並非本階段必要品質檢定的先決條件，亦非論文發表的唯一硬性標準。
4. **現狀確認**：本輪研究**尚未建立任何全模型 $\ge 10\%$ 的加速或發表級成功**。當前工作嚴格限定於等待凍結的 `C64` 必要品質結果，不增設非必要前置關卡，亦不展開任何未授權的實驗。
