# Round 3 候選思考 B Block 03：數值契約、有限善意訓練架構與初級品質橋樑形式化備忘錄

**記錄標識**：`round03-thinking/gemini_formal_b_block03_contract_training_bridge.md`（純文字直接交付，無任何工具呼叫、無檔案讀寫、無計時或門檻通過宣稱）  
**當前累積實質工作**：Block 01（112.73s）+ Block 02（151.12s）= 263.85s。未達 1800 秒門檻，本區段持續進行深入推導。

---

## 0. B02 重大斷言修正與撤銷清單

1. **模型結構與幾何核算撤銷**：
   - DeepSeek-V2-Lite 實體結構為：**第 1 層為純 Dense 層**，後續 **26 個 Routed MoE 層**（每一 Routed 層各自包含獨立的 Shared-Expert 分支），先前「1 個 shared / 26 個 routed」表述不精確。
   - 跨 26 個 Routed 層之 GU 全池總量在 W8 下為 $26 \times 22\text{ MiB} = \mathbf{572\text{ MiB}}$（非 572.5）；在 Q4_K 理想比例（$\times 0.5625$）下為 **$321.75\text{ MiB}$**。同時撤銷「$I=1408$ 可隨意適配 native 256 Q4_K」之假設，其非 256 整除性必須由填充或特定區塊對齊處理。
2. **PCoMoE 等價與支援集重用撤銷**：
   - 專家間即使支援集不等於全池（$s < M$），只要選中專家的支援集子集完全一致，依然能實現全域 $U$ 重用。**全面撤銷所有關於 PCoMoE 等價之武斷「當且僅當（iff）」規則**，不再設定未經證明的結構等價界線。
3. **主機常駐與延遲判定撤銷**：
   - 權重字節數小於 8 GB **絕不等於** 系統總記憶體可完全容納（必須計入 KV Cache、Activation 暫存、系統 Runtime 及框架開銷）；更不能由此推導出 CPU/NPU 總延遲必勝。
   - **全面撤銷「$\ge 4\text{ GB}$ Host 即 Thought-Veto」之非必要條件**，亦不透過退守至「DRAM-less 控制器」來強行維持候選生命。Host-Cached-GU + Flash-Down 形式作為對等競賽架構，其勝負完全由實測/嚴格推導之服務時間（Service Time）決定。
4. **未授權之訓練步數與死亡線撤銷**：
   - 全面撤銷「5000 步」、「RelMSE 2.0 倍 / 0.05」、「$\bar{U} \ge 30$」及「6 waves」等未經批准的凍結門檻。
   - 訓練 MSE 與 Down 條件數僅為過程診斷指標，**絕不可替代真實 Ground-Truth（GT）原始源模型 NLL 品質評估**。特定有限優化器之不收斂或數值發散僅代表該 Recipe 失敗，不構成表徵容量的理論下界。
5. **稀有專家與損失函數斷言修正**：
   - 自然路由稀有**不等於**無法訓練；溫度平滑採樣（$\alpha^{1/\tau}$）無法憑空創造在資料流形上不存在的專家狀態；任何 Off-route 評估必須明示其計算成本。
   - $\lambda_{\text{anchor}} > 0$ 能抑制同 Token 抵消，但**不能數學保證未見分佈之泛化**。撤銷「3 倍 rare MSE / 激活為零即科學死亡線」之先驗標準。
6. **成本節省範疇澄清**：
   - 3-Group Venn 構造節省 25% 僅針對「GU 等成本群組數量」，**絕不能偷換為整個 FFN 或端到端延遲的 25% 節省**。

---

## 1. 數值與 Phase 契約比較：Whole-Expert A8 vs. T128 Block-Quantized

在 AiF 原生硬體（INT8 MAC、逐行 Scale、全域 A8、I32 點積累加、Host F32 縮放及 SiLU）基礎上，深入比較候選架構若採用 Canonical W8 時的兩種活頁量化協議。源模型（Q4_K/Q6_K/Q8_0）並非同等數值，此處專注於 Student 內部的完整依賴 DAG 與記憶體代價。

```
方案 1：Whole-Expert I A8 流程
[輸入 x (F32/A8)] ──> [計算活躍聯集 U 之 GU 原子 (F32)] ──> [暫存 U*4 Bytes 於 SRAM]
                                                                  │
      ┌───────────────────────────────────────────────────────────┘
      ▼ (對每個活躍專家 e 分別執行)
[提取專家私有 I 個原子特徵] ──> [計算專家局部 amax^(e)] ──> [量化為專屬 int8 codes q_8^(e)]
                                                                  │
                                                                  ▼
                                                      [INT8 MAC: Down_e = V^(e) * q_8^(e)]
                                                                  │
                                                      [單一乘法: s_W * s_A^(e) * F32_Sum]
```

### 1.1 方案 1：Canonical W8 GU/Down + 每專家 Whole-$I$ A8
- **依賴衝突推導**：
  雖然活躍聯集中的 GU 原子 $\phi_j(x)$ 算術值相同，但對於任意兩個同時選中的專家 $e_1$ 與 $e_2$：
  $$s_A^{(e_1)} = \frac{\max_{j \in \mathcal{S}_{e_1}} |\phi_j(x)|}{127}, \quad s_A^{(e_2)} = \frac{\max_{j \in \mathcal{S}_{e_2}} |\phi_j(x)|}{127}$$
  因支援集 $\mathcal{S}_{e_1} \neq \mathcal{S}_{e_2}$，其集合最大值一般不相等（$s_A^{(e_1)} \neq s_A^{(e_2)}$）。
  因此，即使是完全相同的原子 $j \in \mathcal{S}_{e_1} \cap \mathcal{S}_{e_2}$，其量化整數碼：
  $$q_{8, j}^{(e_1)} = \text{round}\left( \frac{\phi_j(x)}{s_A^{(e_1)}} \right) \neq \text{round}\left( \frac{\phi_j(x)}{s_A^{(e_2)}} \right) = q_{8, j}^{(e_2)}$$
- **剛性狀態儲存要求**：
  **共享原子無法以 int8 形式跨專家直接複用**。控制器必須以 **F32（或 FP16）完整精度暫存活躍聯集特徵**。
  - *活存記憶體需求*：$U_{\text{atoms}} \times 4\text{ Bytes}$。
    對 OLM（$U \le 4096$），活存需求為 $4096 \times 4\text{ B} = \mathbf{16\text{ KiB}}$（完全可容納於控制器本地 SRAM）。
  - *計算代價*：每個選中專家必須對其 $I$ 個輸入重新掃描 $\max$ 並執行一次除法量化，隨後直接驅動私有 Down 的標準單一 Scale INT8 MAC。

---

### 1.2 方案 2：T128 Block-Quantized Activation（群組局部量化）
- **機制**：
  直接對每個 128-neuron Group $g$ 計算獨立的 Scale：$s_{A, g} = \max_{j \in \text{Group}_g} |\phi_j(x)| / 127$。
  量化碼 $q_{8, j}$ 在群組內部完全確定，所有包含該 Group 的專家**100% 共享相同的 int8 碼**。
- **剛性狀態與算術代價（Paid Costs）**：
  - *狀態記憶體節省*：活存僅需 $U_{\text{atoms}} \times 1\text{ Byte} = \mathbf{4\text{ KiB}}$。
  - *Down 投影碎裂化*：Down 矩陣乘法無法作為長度為 $I$ 的單一向量內積完成。它被迫切分為 $s/T = 8$ 個獨立的 128 維部分和（Partial Sums）：
    $$\text{Down}_e = \sum_{g \in \mathcal{S}_e} \left( s_{W, g}^{(e)} \cdot s_{A, g} \right) \sum_{k=1}^{128} V_{e, g, k} \cdot q_{8, g, k}$$
  - *代價*：每個輸出 Row 必須維護 8 個獨立的浮點乘累加（FMA）與中間 Carry，PE 端需要額外的定點轉浮點及多 Scale 乘法邏輯，且區塊過小引入了高頻 Scale 儲存與量化階雜訊。

---

### 1.3 [本段推導] W8 Student vs. Native Q4_K / Q6_K 之 Byte Break-Even 嚴格推導

#### A. 對照組：Uniform Native Q4_K
- 基準 Native Q4_K 權重字節率：$144\text{ B} / 256\text{ weights} = \mathbf{0.5625\text{ B/weight}}$。
- 原始 MoE 每 Token 讀取權重數：$2KIH\text{ (GU)} + KIH\text{ (Down)} = 3KIH$。
- **Native Q4_K 總字節數**：$3KIH \times 0.5625 = \mathbf{1.6875 K H I\text{ Bytes}}$。
- **W8 Student（GU 讀取 $U$ 原子，Down 讀取 $KI$）總字節數**：
  $$W_{\text{W8}} = (2U \cdot H + K I \cdot H) \times 1.0\text{ B} = H(2U + KI)$$
- **打平條件（Break-Even Inequality）**：
  $$H(2U + KI) < 1.6875 K H I \implies 2U < 0.6875 KI \implies \mathbf{U < 0.34375 KI}$$
  - 當 $K=8$ 時：$U < 0.34375 \times 8 I = \mathbf{2.75 I}$。
  - 當 $K=6$ 時：$U < 0.34375 \times 6 I = \mathbf{2.0625 I}$。
- *結論*：若 $K=8$ 下活躍聯集 $\mathbb{E}[U] \approx 3.6I$，則 **$3.6I > 2.75I$，W8 Student 讀取的總字節數嚴格大於 Native Q4_K！** 唯有強正則化迫使 $U < 2.75I$ 時，W8 候選方具備字節優勢。

#### B. 對照組：Mixed Q4_K (GU) + Q6_K (Down)
- Q6_K 權重字節率：$210\text{ B} / 256\text{ weights} = \mathbf{0.8203125\text{ B/weight}}$。
- **Native 混合精度總字節數**：
  $$\text{Bytes}_{\text{native}} = 2KIH \times 0.5625 + KIH \times 0.8203125 = (1.125 + 0.8203125) KIH = \mathbf{1.9453125 K H I\text{ Bytes}}$$
- **W8 Student 打平條件**：
  $$H(2U + KI) < 1.9453125 K H I \implies 2U < 0.9453125 KI \implies \mathbf{U < 0.472656 KI}$$
  - 當 $K=8$ 時：$U < 0.472656 \times 8 I = \mathbf{3.781 I}$。
  - 當 $K=6$ 時：$U < 0.472656 \times 6 I = \mathbf{2.836 I}$。
- *結論*：若源模型 Down 採用更重的 Q6_K，W8 的門檻放寬至 $3.781I$。此時隨機路由下的 $\mathbb{E}[U] \approx 3.6I < 3.781I$，W8 Student 始能獲得微弱的字節節省空間。但這絕不等於部署性能必勝，因其放棄了低位元特化硬體的算術密度。

---

## 2. 有限善意訓練設計（Finite Benevolent Training Recipe）

### 2.1 參數體積與硬體記憶體現實
以 OLM 單層為例：
- 共享 GU 池（$M=4096, H=2048$）：$2 \times 4096 \times 2048 = \mathbf{16.78\text{M}}$ 參數。
- 64 個私有 Down 矩陣（若初始未遮罩，全池 $M=4096$）：$64 \times (4096 \times 2048) = \mathbf{536.87\text{M}}$ 參數。
- **訓練峰值記憶體（FP32 權重 + 梯度 + Adam 兩階動量）**：
  $$(16.78\text{M} + 536.87\text{M}) \times 4\text{ 份} \times 4\text{ Bytes} \approx \mathbf{8.86\text{ GB}}$$
  僅單層全 Down 訓練的參數狀態即突破 8 GB 顯存限制。若直接進行端到端未遮罩全訓練，在有限資源下必然 OOM。必須設計結構化縮減與分階段協議。

---

### 2.2 [本段推導] 硬遮罩死鎖（Hard Mask Dead-Lock）之導數證明

設專家 $e$ 對群組 $g$ 的遮罩為 $m_{e, g} \in \{0, 1\}$，由連續 Logit $s_{e, g}$ 經硬閾值生成。
輸出為：$y_e = \sum_{g=1}^G m_{e, g} V_{e, g} \phi_g(x)$。
損失函數為 $\mathcal{L}$。對未被選中群組（$m_{e, g} = 0$）的 Down 權重 $V_{e, g}$ 與 Logit $s_{e, g}$ 求導：
1. **Down 權重梯度**：
   $$\frac{\partial \mathcal{L}}{\partial V_{e, g}} = \frac{\partial \mathcal{L}}{\partial y_e} \cdot \frac{\partial y_e}{\partial V_{e, g}} = \frac{\partial \mathcal{L}}{\partial y_e} \cdot \left( m_{e, g} \phi_g(x)^\top \right) = \mathbf{0} \quad (\text{因 } m_{e, g} = 0)$$
2. **Logit 梯度（若未選中且 Down 初始化為 0）**：
   $$\frac{\partial \mathcal{L}}{\partial s_{e, g}} = \sum \frac{\partial \mathcal{L}}{\partial y_e} \cdot \left( V_{e, g} \phi_g(x) \right) \cdot \frac{\partial m_{e, g}}{\partial s_{e, g}}$$
   即使採用 Straight-Through Estimator（$\frac{\partial m}{\partial s} = 1$），若未選中群組的權重處於零初始化狀態（$V_{e, g} = 0$），則：
   $$\frac{\partial \mathcal{L}}{\partial s_{e, g}} = \mathbf{0}$$
- **死鎖結論**：**「硬遮罩 + 零初始化」會徹底阻斷未選中群組的梯度流**，使得初始隨機指派的支援集永遠無法藉由反向傳播探索更好的群組組合。

---

### 2.3 可審查的三階段有限訓練協議（Three-Phase Recipe）

```
[Phase 1: Seed & Ridge 初始化]
  選取 4 個代表專家拼裝池種子 ──> 特徵餘弦初始化 32 群組 Logit ──> 閉式 Ridge 求解初始 Down (134M 參數)
                                                                             │
                                                                             ▼
[Phase 2: 溫度退火聯合微調]
  Gumbel-Softmax 平滑遮罩 ──> Joint Mixture + Anchor 損失 ──> 溫度退火 (tau: 1.0 -> 0.1) 更新 GU 與 Down
                                                                             │
                                                                             ▼
[Phase 3: 硬支援集鎖定與收尾]
  固定 Top-8 硬遮罩 (拋棄無效 Down) ──> 凍結拓撲 ──> 僅微調 16.8M GU + 134M 私有 Down
```

1. **Phase 1（特徵字典聚類與閉式 Ridge 初始化）**：
   - 依 TRAIN 子集的激活共現頻率，選取 4 個最具正交代表性的 Teacher 專家，將其 $4 \times 1024$ 個原子作為初始池 $\Theta_{\text{GU}}^{(0)}$。
   - 計算各專家 Teacher 輸出與 32 個群組特徵的投影像，依餘弦相關度初始化支援集 Logits $S \in \mathbb{R}^{64 \times 32}$。
   - 僅對各專家當前選中的 8 個群組，利用封閉式嶺回歸（Ridge Regression）解出初始 Down 權重（此時參數量嚴格限制在部署級別的 $134.2\text{M}$，記憶體峰值 $<2.5\text{ GB}$）。
2. **Phase 2（軟化溫度退火聯合微調，Soft Warmup）**：
   - 群組遮罩採用平滑連續鬆弛：$m_{e, g} = \text{Sigmoid}(s_{e, g} / \tau)$，溫度 $\tau$ 隨訓練進程自 $1.0$ 退火至 $0.1$。
   - **損失函數雙重錨定**：
     $$\mathcal{L} = \left\| \sum_{e \in R} \alpha_e (y_e^{\text{student}} - y_e^{\text{teacher}}) \right\|_2^2 + \lambda_{\text{anchor}} \sum_{e \in R} \alpha_e \| y_e^{\text{student}} - y_e^{\text{teacher}} \|_2^2 + \beta \mathcal{R}_{\text{cohort}}(S, R)$$
     $\lambda_{\text{anchor}} > 0$ 嚴格壓制跨專家相互掩蓋；$\beta$ 梯度拉動群組活躍聯集向硬體目標靠攏。
   - **稀有專家補償底線**：在 TRAIN 採樣中，若某專家路由計數低於門檻 $N_{\min}$，自歷史 Buffer 中重採樣該專家被激發的 Token（Re-sampling），保證全 64 專家皆有基礎梯度暴露。
3. **Phase 3（硬支援集鎖定與最終擬合）**：
   - 將 $S$ 離散化為嚴格的 8-hot 二元遮罩，永久固定各專家的支援集 $\mathcal{S}_e$。
   - 丟棄所有未被選中的虛擬 Down 參數，在固定拓撲下僅對 $16.78\text{M}$ GU 與 $134.2\text{M}$ Down 進行低學習率收斂微調。

---

### 2.4 文獻邊界與有限預算歸因約束
- **文獻先驗事實（Provided Source Facts）**：
  - *PCoMoE*（25K samples）：凍結專家本體，僅訓練 Rank-16 的 Gate LoRA 與路徑相容矩陣。
  - *Mixture-of-Masks (MoM)*：需要高達 20B Tokens 進行密集到 MoE 的神經元剪枝。
  - *DERN*：128 維神經元三元組加權聚類，本質為免訓練重組。
- **科學誠實歸因界線**：
  既有 TRAIN 16PG（約 30k positions）對於 64 專家、平均每專家僅約 3750 次路由暴露。若在此有限資料與計算預算下訓練發散，**僅能報告「在 30k tokens 與三階段 Recipe 下該初始化無法收斂」，絕不可擴大宣稱為「數學上所有共享非線性池族皆無法表徵 MoE」**。嚴禁調用未見的 VAL 集反覆試錯救援。

---

## 3. Primary 品質橋樑（Primary Quality Bridge Protocol）

### 3.1 單一預定中間層評估架構（Layer 8 of 16 in OLM）

為杜絕挑選有利層數的後見偏差，嚴格預定以 **Layer 8** 作為唯一評估對象，其餘 15 層嚴格凍結為原始 Native 權重。

```
[原始 Native 基準 S] ──(量化轉換)──> [未共享 Canonical W8 基準 B] ──(池化共享)──> [共享學生架構 C]
         │                                      │                                  │
         └─────────────── 基準差 (B - S) ───────┴─────── 結構差 (C - B) ───────────┘
                                                │
                                                ▼
                                    總體主要評估 (C - S)
```

1. **三方橋樑實體定義**：
   - **$S$（源模型 Native 基準）**：原始完整未動之 Native Q4_K/Q6_K 權重。
   - **$B$（量化橋樑，Unshared Canonical W8）**：保持與 $S$ 完全相同的 64 專家獨立拓撲，但將 GU 與 Down 權重轉為標準 Canonical W8 格式。
   - **$C$（候選 Student）**：$M=4096$ 共享 GU 池 + 私有 Down，使用與 $B$ 完全相同的 Canonical W8 量化器。
2. **主要與次要指標劃分**：
   - **Primary Metric**：$C - S$ 在預定校準資料集上的 **Ground-Truth NLL 差異（$\Delta \text{NLL}_{C-S}$）**。
   - **Diagnostic Metric 1（$B - S$）**：純粹量化誤差診斷。隔離「轉換為 W8」本身造成的品質下降。
   - **Diagnostic Metric 2（$C - B$）**：純粹結構壓縮誤差診斷。隔離「神經元池化共享」造成的品質下降。
   - *鐵律*：**絕對禁止以訓練 MSE 替代 GT NLL 作為通過標準**。

---

### 3.2 [本段推導] Layer 8 入口 $x$ 與 Router 逐位可比的嚴格條件與反例

#### A. 嚴格成立條件（Bit-Exact Condition）
在評估 Layer 8 時，$S, B, C$ 三者在 Layer 8 入口處的輸入向量 $x_8$、Router Logits 與選中專家集合 $\mathcal{E}(x_8)$ 達到**逐位完全一致（Bit-Exact）的充分必要條件是**：
> **「Layer 0 至 Layer 7 的全部權重、算子、KV Cache 提取邏輯與殘差連接，在三者中皆 100% 採用原始 Native $S$ 實作，且僅在推論單步（Single Forward Pass）中單向注入 Layer 8。」**

#### B. 多層替換與自回歸步反例（Violation Counterexample）
1. *多層替換破壞*：若同時將 Layer 4 與 Layer 8 替換為 Student 結構。Layer 4 輸出的微小數值擾動（哪怕只有 1 ULP）在經過 Layer 5~7 的 Attention 與 LayerNorm 放大後，進入 Layer 8 的輸入 $x_8'$ 將與原始 $x_8$ 產生夾角，導致 Router 計算出完全不同的 Top-$K$ 專家清單。此時三者入口完全不可比。
2. *自回歸自循環破壞*：在多 Token 生成中，即便只替換 Layer 8，第 1 個 Token 在 Layer 8 產生的誤差會寫入該層及後續層的 KV Cache。在生成第 2 個 Token 時，前序 KV Cache 的分歧將反向污染 Layer 0~7 的輸出，使 Layer 8 的入口 $x_8$ 在 Step 2 立即喪失逐位可比性。

---

### 3.3 Sanity Check（接線與恆等對照測試）
在正式評估前，必須執行 **Identity Copy Wire-Test**：
- 將 Teacher 專家的前 4 個專家原樣複製填入 Student $C$ 的池中，令前 4 個專家的支援集指向自身對應區塊，其餘專家置零。
- 對這 4 個專家輸入測試向量，$C$ 的輸出必須與 $B$ 在 F32 機器精度內嚴格相等（$\|Y_C - Y_B\| / \|Y_B\| < 10^{-6}$）。
- 若未通過，直接判定軟體框架存在索引、轉置或接線錯誤（Wiring Bug），嚴禁進行任何蒸餾訓練。

### 3.4 橋樑失效處置規則
- 若 **$B - S$ 已顯著劣化**（如 $\Delta \text{NLL}_{B-S} > +0.10$）：
  表明 Canonical W8 量化本身已破壞該層的表徵平衡。此時無論 $C - S$ 表現如何，**必須如實報告 $B - S$ 的失敗**，不可隱瞞量化器的缺陷，亦不可將此罪責歸咎於池化結構。
- 若 $B - S$ 正常但 **$C - B$ 劇烈劣化**：
  確定為 $M=4I$ 池化共享之表徵容量不足或訓練發散。

---

## 4. 條件式可行前沿 vs. 低位元毀滅性反例

### 4.1 仍有正面可能的情境案例：超大模型與低頻寬儲存

在不依賴 DRAM-less 控制器假定的前提下，本候選在以下符號邊界內仍具備正向優勢：

#### 符號條件定義：
- 隱藏維度 $H$，單專家寬度 $I$，專家總數 $E \ge 128$，選中數 $K \ge 8$。
- Flash 介面頻寬為 $\text{BW}_{\text{flash}}$，內部高頻運算 SRAM 頻寬為 $\text{BW}_{\text{internal}} \gg \text{BW}_{\text{flash}}$。
- 物理感測延遲為 $t_R = 28\,\mu\text{s}$。
- 當模型尺度擴展至 $H=8192, I=4096$ 時，單層 GU 全池體積：
  $$\text{Vol}_{\text{GU}} = 2 \times 4I \times H \times 1\text{B} = 8 \times 4096 \times 8192 = \mathbf{256\text{ MiB / 層}}$$
  全模型 32 層 GU 總量高達 **8.0 GB**，已達到 Host DRAM 的整體容量上限，**全池無法在 Host 中常駐**。
- **正向服務時間不等式**：
  若透過 Cohort 訓練將活躍聯集壓縮至 $U \le 1.5 I$，使得 Flash 端讀取的實體 Wave 數大幅下降：
  $$T_{\text{candidate}} = \left( \left\lceil \frac{U}{\text{Plane}} \right\rceil \cdot t_R + \frac{2UH + K s H}{\text{BW}_{\text{flash}}} \right) < \left( K \cdot t_R + \frac{3KIH \cdot B_{\text{native}}}{\text{BW}_{\text{flash}}} \right) = T_{\text{native}}$$
  在此超大模型、Flash 感測波次（$t_R$）佔據主要延遲、且 Host 無法常駐的邊界下，In-Flash 的群組共享具備真實的端到端時間縮短價值。

---

### 4.2 致命反例：極端低位元源模型之字節倒掛（Low-Bit Inversion）

#### 反例構造：
- 設源模型採用極限低位元量化（如 2.5-bit 或 3-bit 混合壓縮 Q3_K_S，平均字節率 $B_{\text{native}} = \mathbf{0.375\text{ B/weight}}$）。
- 設 Student 即使在神話級別的最優假設下運作：活躍聯集達到絕對極限 $U = s = I$（100% 完全重用，無任何冗餘原子）。
- 由於 Student 維持 Canonical W8 數值契約（$B_{\text{W8}} = 1.0\text{ B/weight}$）：
  - **Student 每 Token 讀取字節數**：
    $$\text{Bytes}_{\text{W8}} = H(2U + Ks) \times 1.0\text{ B} = H(2I + KI) = (K + 2) H I\text{ Bytes}$$
  - **Native 2.5-bit 原始每 Token 讀取字節數**：
    $$\text{Bytes}_{\text{native}} = 3KIH \times 0.375\text{ B} = \mathbf{1.125 K H I\text{ Bytes}}$$
- **代數對比（以 $K=8$ 代入）**：
  - $\text{Bytes}_{\text{W8}} = (8 + 2) H I = \mathbf{10.0 H I\text{ Bytes}}$
  - $\text{Bytes}_{\text{native}} = 1.125 \times 8 H I = \mathbf{9.0 H I\text{ Bytes}}$
- **結論**：
  即使 Student 達到了理論上不可能超越的完美共享（$U=I$），**其搬運的總字節數依然比 Native 低位元模型高出 11.1%**！
  若 $U$ 為真實隨機的 $3.6I$，則 Student 需讀取 $15.2 HI$ 字節，**比 Native 模型膨脹 68.9%**。
  在此情境下，無論快閃記憶體架構如何調度、控制器多麼先進，W8 Student 在傳輸延遲、感測能耗與儲存容量上全面潰敗。此反例嚴格證明：**非線性池化共享的紅利若無法跨越「量化位元差」的鴻溝，其架構優勢將被徹底抹除**。

---

## 5. 總結與 Block 04 唯一聚焦問題

### 5.1 核心成果摘要
1. **數值合約釐清**：證明了 Whole-Expert A8 必須以 F32 暫存 $16\text{ KiB}$ 活躍原子，各專家需獨立動態量化；而 T128 Block-Quantized 雖可共享 int8，但會將 Down 投影碎裂為 8 次浮點累加。
2. **Break-Even 解析解**：推導出 W8 Student 唯有在 $U < 2.75I$（對照 Q4_K）或 $U < 3.78I$（對照 Q4/Q6 混精）時方能具備字節節省。
3. **死鎖機制破除**：給出硬遮罩下未選中群組梯度為零的嚴格導數，確立了基於特徵餘弦初始化、溫度退火與雙重錨定的三階段善意 Recipe。

### 5.2 Block 04 唯一核心問題
> **「在固定中間層（Layer 8）上，當採用三階段 Recipe 與有限 30k tokens 時，$\lambda_{\text{anchor}}$ 的取值下界應如何透過 Fisher 資訊矩陣或梯度範數解析推導，以確保在阻斷跨專家抵消（Cancellation）的同時，不至於過度拘束共享池 $\Theta_{\text{GU}}$ 探索出具有實質重疊度（$U \le 2.5I$）的新特徵子空間？」**
