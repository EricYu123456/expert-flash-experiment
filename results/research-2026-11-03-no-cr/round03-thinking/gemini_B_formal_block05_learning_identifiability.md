# Round 3 候選思考 B Block 05：梯度飽和、共變更死鎖與實體 Wave 學習邊界備忘錄

**記錄標識**：`round03-thinking/gemini_formal_b_block05_wave_learning_and_gauges.md`（純文字交付，無工具呼叫、無資料/權重讀取、無偽造計時）  
**當前累積實質工作**：Block 01~04 合計 502.30s。未達 1800 秒門檻，本區段聚焦於「如何學習到真正能減少實體 Wave 的硬遮罩，而非僅在平滑代理中最佳化虛構成本」。

---

## 1. 梯度飽和、單專家死鎖與協同變更（Coordinated Move）

### 1.1 [本段推導] 單專家群組替換對實體 Slot 的消減條件

設物理配對 $p = \{g_a, g_b\}$ 為固定實體 Slot。
對 Token 路由選中的專家集合 $R$（$|R| = K$），定義請求該 Slot 的專家計數：
$$n_p(R) = \sum_{e \in R} \mathbb{I}\{ m_{e, g_a} = 1 \lor m_{e, g_b} = 1 \}$$
實體感測波次成本為：$\text{Cost}(R) = \sum_{p=1}^{16} \mathbb{I}\{ n_p(R) > 0 \}$。

#### 單專家替換（Single-Expert Replacement）：
考慮專家 $e_1 \in R$ 將其支援集中的群組 $g_{\text{old}} \in p_{\text{old}}$ 替換為 $g_{\text{new}} \in p_{\text{new}}$，保持其基數 $s/T = 8$ 不變。
**此操作能使總 Slot 數減少 1 的充分必要條件為**：
1. **釋放條件（Release Condition）**：在替換前，$p_{\text{old}}$ 的請求計數必須恰好為 1（$n_{p_{\text{old}}}(R) = 1$），即專家 $e_1$ 是整個 Cohort 中**唯一需要該 Slot 的專家**。
2. **非自鎖條件（No Self-Lock）**：專家 $e_1$ 在 $p_{\text{old}}$ 中僅使用了 $g_{\text{old}}$ 單一細群組，未同時使用該 Slot 的另一半（即 $m_{e_1, g_a} + m_{e_1, g_b} = 1$）。
3. **吸收條件（Absorption Condition）**：新群組 $g_{\text{new}}$ 所屬的實體對 $p_{\text{new}}$ 在替換前**已經處於活躍狀態**（$n_{p_{\text{new}}}(R) \ge 1$），從而不引發新的 Slot 開銷。

---

### 1.2 [本段推導] Toy 幾何下的孤立請求機率與梯度飽和分析

在 $G=32, q=8, K=8$ 的均勻獨立選擇假設下進行嚴格組合核算：
1. **單一專家避開特定實體對 $p$ 的精確機率**：
   $$P_{\text{avoid}} = \frac{\binom{30}{8}}{\binom{32}{8}} = \frac{552}{992} = \frac{69}{124} \approx 0.55645$$
2. **單一專家需要特定實體對 $p$ 的機率**：
   $$P_{\text{need}} = 1 - P_{\text{avoid}} = \frac{55}{124} \approx 0.44355$$
3. **特定 Slot 恰好只有 1 個專家需要的機率（$n_p = 1$）**：
   $$P(n_p = 1) = \binom{8}{1} \cdot P_{\text{need}} \cdot (P_{\text{avoid}})^7 = 8 \times 0.44355 \times (0.55645)^7 \approx 8 \times 0.44355 \times 0.01650 \approx \mathbf{0.0585} \quad (5.85\%)$$
4. **特定細群組 $g$ 恰好只有 1 個專家需要的機率**：
   單一專家選中 $g$ 的機率為 $q/G = 8/32 = 0.25$；避開機率為 $0.75$。
   $$P(\text{only 1 expert needs group } g) = \binom{8}{1} \cdot (0.25)^1 \cdot (0.75)^7 = 8 \times 0.25 \times 0.13348 \approx \mathbf{0.2670} \quad (26.7\%)$$

#### 梯度飽和的數學本質：
若採用平滑代理目標函數：
$$C_{\text{soft}}(p) = 1 - \prod_{e \in R} (1 - m_{e, g_a})(1 - m_{e, g_b})$$
對專家 $e_1$ 的 Logit 求導：
$$\frac{\partial C_{\text{soft}}(p)}{\partial m_{e_1, g_a}} = (1 - m_{e_1, g_b}) \cdot \prod_{e \in R \setminus \{e_1\}} (1 - m_{e, g_a})(1 - m_{e, g_b})$$
- 當 $n_p(R) \ge 2$ 時（在此幾何下機率高達 $1 - 0.0092 - 0.0585 = \mathbf{93.23\%}$），乘積項中至少包含一個接近 0 的因子（來自其他已選中該 Slot 的專家）。
- **結果**：在超過 93% 的情況下，單一專家的梯度被連乘項直接壓制至近乎零！平滑代理損失在未剪枝初期呈現嚴重的**高原飽和（Plateau Saturation）**。

---

### 1.3 [本段推導] 協同變更（Coordinated Move）之最小死鎖反例

為證明單專家貪婪無法突破實體 Wave 屏障，給出以下具體驗算範例：
- **環境**：$G = 4$ 個細群組，組成 2 個實體對 $P_1 = \{1, 2\}, P_2 = \{3, 4\}$。
- **配置**：$K = 2$ 個專家（$e_1, e_2$），每專家硬基數 $s/T = 2$。
- **初始支援集**：
  $$\mathcal{S}_1 = \{1, 3\}, \quad \mathcal{S}_2 = \{2, 4\}$$
- **初始實體狀態**：
  - Slot $P_1$：$1 \in \mathcal{S}_1, 2 \in \mathcal{S}_2 \implies n_{P_1} = 2 > 0$（活躍）。
  - Slot $P_2$：$3 \in \mathcal{S}_1, 4 \in \mathcal{S}_2 \implies n_{P_2} = 2 > 0$（活躍）。
  - **總感測波次**：$\mathbf{2\text{ Slots}}$。

#### 單專家單步嘗試（保持基數為 2）：
- 若專家 $e_1$ 試圖將群組 3 換為群組 2：$\mathcal{S}_1' = \{1, 2\}$。
  - 此時 $e_2$ 依然為 $\{2, 4\}$。
  - Slot $P_1$ 仍有 $\{1, 2\}$ 請求；Slot $P_2$ 仍有 $\{4\}$ 請求。
  - 總 Slot 數依舊為 **2**（改善量 $\Delta = 0$）。
- 若專家 $e_1$ 試圖換為 $\{3, 4\}$：
  - $e_2$ 依然為 $\{2, 4\} \implies P_1$ 仍被 $e_2$ 佔用，$P_2$ 被兩者佔用。
  - 總 Slot 數依舊為 **2**（改善量 $\Delta = 0$）。
- 由對稱性可知，**任何單一專家的單步變更，均無法使 Slot 數從 2 降至 1**。單專家局部優化在此處陷入完全死鎖。

#### 協同變更（Coordinated Move）：
- 令專家 $e_1$ 變更為 $\{1, 2\}$，**同時** 令專家 $e_2$ 亦變更為 $\{1, 2\}$。
- 變更後：
  - Slot $P_1$ 包含 $\{1, 2\}$（$n_{P_1} = 2 > 0$）。
  - Slot $P_2$ 完全無專家請求（$n_{P_2} = 0$）。
- **總感測波次從 2 降至 1（節省 50% Wave）**！
- **結論**：實體 Wave 成本具有強烈的階躍非線性與多專家相依性，必須依賴能夠傳遞跨專家梯度的聯合學習機制。

---

## 2. $mD$ Gauge 自由度與固定基數策略

### 2.1 [本段推導] 尺度轉換（Gauge Transformation）與規範破壞

在連續鬆弛下，專家輸出為 $y_e = \sum_g m_{e, g} D_{e, g} \phi_g(x)$。
對於任意純量 $c > 0$，變換 $m_{e, g} \to c \cdot m_{e, g}$ 且 $D_{e, g} \to \frac{1}{c} D_{e, g}$ 保持實數數學輸出完全不變。

| 正則化 / 約束機制 | 規範破壞方式（How Gauge is Broken） | 對函數空間族與梯度的實質改變 |
| :--- | :--- | :--- |
| **$L_2$ 權重衰減（$\lambda \|D\|_F^2$）** | 代價為 $\frac{1}{c^2} \|D\|_F^2$，驅使 $c \to \infty$ | 迫使 $m \to 1$（飽和），嚴重弱化稀疏性，無法破除尺度欺騙 |
| **W8 權重動態量化** | 量化階距 $\Delta = \max \|D\| / 127$ 具有絕對尺度 | $c$ 過大時小數值截斷歸零；$c$ 過小時觸發飽和截斷；破除連續規範 |
| **Down 矩陣行範數截斷（Clipping）** | 強制約束 $\|D_{e, g, :}\|_2 \le C_{\max}$ | 給定輸出需求下，強制約束 $m_{e, g} \ge \frac{\|y_e\|}{C_{\max} \|\phi_g\|}$，直接阻斷 $m \to 0$ 逃逸 |

---

### 2.2 [本段推導] 精確超幾何避開公式 vs. 獨立 Bernoulli 近似

在無放回挑選 $q$ 個群組的真實幾何下：
- **獨立 Bernoulli 近似**（假設兩群組獨立被避開）：
  $$P_{\text{avoid}}^{\text{Bernoulli}} = (1 - p)^2 = \left( 1 - \frac{q}{G} \right)^2 = \left( 1 - \frac{8}{32} \right)^2 = \left( \frac{3}{4} \right)^2 = \mathbf{0.56250}$$
- **精確超幾何公式（Hypergeometric Without Replacement）**：
  $$P_{\text{avoid}}^{\text{Exact}} = \frac{\binom{G-2}{q}}{\binom{G}{q}} = \frac{(G-q)(G-q-1)}{G(G-1)} = \frac{24 \times 23}{32 \times 31} = \frac{552}{992} = \mathbf{0.55645}$$
- **偏差辨析**：
  Bernoulli 近似高估了避開機率（$+0.00605$）。在 8 個專家連乘後，$(0.5625)^8 \approx 0.0100$ 對比 $(0.5565)^8 \approx 0.0092$。Bernoulli 近似會輕微低估 Slot 衝突率。

---

### 2.3 單一有限誠實實作選擇（Single Finite Recipe）

為規避大規模超參數網格搜索，定義以下**理論誠實且可實作的規範**：
1. **前向硬二值化（Hard Top-$q$ Forward）**：
   在 Forward pass 中，專家 $e$ 的遮罩嚴格取連續評分 $s_{e, g}$ 的 Top-$q$（$q=8$）：
   $$m_{e, g} = \mathbb{I}\{\text{Rank}(s_{e, g}) \le 8\}$$
   **前向完全無尺度浮動，絕對不出現 $m \to 0, D \to \infty$ 的虛擬成本洩漏**。
2. **反向 STE 梯度（Straight-Through Estimator）**：
   Backward pass 僅對處於邊界競爭區的群組傳遞梯度：
   $$\frac{\partial m_{e, g}}{\partial s_{e, g}} = \text{HardTanh}'(s_{e, g} - s_{e, \text{threshold}})$$
3. **成本記錄隔離（True Cost Accounting）**：
   訓練損失中的 Slot 正則化項僅作為梯度引導，**最終報告的硬體 Wave 數嚴格在離線硬遮罩凍結後、在合法 MWPM 佈局下由離散模擬器統計**。

---

## 3. 代表性初始化：4 Source 專家 vs. 響應特徵（Signature）原子選取

### 3.1 [本段推導] 64 點響應特徵提取與計算量精確核算

設自 TRAIN-only 子集中抽取 $N_{\text{sig}} = 64$ 個具備多樣性的固定輸入向量 $\{x_k\}_{k=1}^{64}$。
對原始模型全體專家神經元計算響應特徵向量：
$$\mathbf{r}_j = \left[ \phi_j(x_1), \phi_j(x_2), \dots, \phi_j(x_{64}) \right]^\top \in \mathbb{R}^{64}$$
- **尺度吸收約束**：
  Up 向量與特徵響應的純量縮放及符號翻轉（$\pm \gamma$）可由私有 Down 吸收；
  **但 Gate 向量由於 SiLU 非齊次性，嚴禁在初始化時進行任意方向縮放**！

#### 運算代價階梯核算：
1. **OLM 規格（$H=2048, I=1024, E=64$）**：
   - 原始候選原子總數：$64 \times 1024 = \mathbf{65,536\text{ 個原子}}$。
   - 計算 64 個特徵點之 GEMV 開銷：$65,536 \times 2048 \times 64 \times 2 \approx \mathbf{17.18\text{ G MACs}}$。
   - 自 65,536 個原子中以最遠點採樣（Farthest Point Sampling, FPS）選取 $M = 4096$ 個原子：
     每次迭代更新 65,536 個點到已選集合的最小距離（維度 64）：
     $$4096 \times 65,536 \times 64 \approx \mathbf{17.18\text{ G Operations}}$$
2. **DeepSeek-V2-Lite 規格（$H=2048, I=1408, E=64$）**：
   - 原始候選原子總數：$64 \times 1408 = \mathbf{90,112\text{ 個原子}}$。
   - 抽取 $M = 5632$ 個原子之 FPS 開銷：
     $$5632 \times 90,112 \times 64 \approx \mathbf{32.48\text{ G Operations}}$$

---

### 3.2 群組形成規則（Group Formation Rule）與配額分配

1. **配額保證（Quota Guarantee）**：
   每個原始專家保證分配 $4096 / 64 = \mathbf{64\text{ 個專屬配額原子}}$（依其內部方差最高者選取）；剩餘原子由全域 FPS 補齊，確保稀有專家的基礎表徵不被全域高頻特徵完全吞噬。
2. **固定平衡 128 群組劃分規則**：
   將選出的 4096 個原子依據其 64 維特徵向量執行**平衡 $K$-means（Balanced $K$-means）**，約束劃分為恰好 32 個群組，每組嚴格包含 128 個原子。

---

### 3.3 [本段推導] 兩類特徵混疊反例（Signature Aliasing Counterexamples）

#### 反例 A：高頻/小眾域特徵混疊（Out-of-Sample Aliasing）
- **現象**：設原子 $\phi_A$ 與 $\phi_B$ 在常規文字輸入下的響應完全一致（在 64 個校準點上 $\mathbf{r}_A \approx \mathbf{r}_B$）。
- **衝突**：$\phi_A$ 為通用特徵，而 $\phi_B$ 具有微小的 Gate 偏移，專門用於在特定代碼縮排或極限數學符號下觸發。
- **代價**：僅基於 64 點採樣會將兩者判定為冗餘並予以合併。在未覆蓋的長尾領域推論時，下游 Down 矩陣喪失了關鍵的非線性正交維度，引發長尾任務崩潰。

#### 反例 B：Up 符號歸一化引發 Down 量化截斷（Dynamic Range Blow-up）
- **現象**：為消除特徵符號歧義，初始化強行令所有原子滿足 $\sum_{k=1}^{64} r_{j, k} > 0$；若某原子為負，將 Up 乘以 $-1$ 並將對應 Down 乘以 $-1$。若將極小響應原子強制放大歸一化（$\mathbf{r} \to \mathbf{r} / \|\mathbf{r}\|$）：
- **代價**：Down 矩陣吸收了過大的逆向縮放係數（如 $\times 100$）。當該 Down 矩陣後續轉換為 Canonical W8 時，該 Row 的 FP16 Scale 被極值拉扯，導致同 Row 其他權重在整數量化階中被嚴重截斷為 0，破壞數值精度。

---

## 4. 有限訓練架構與科學可否證邊界

### 4.1 訓練資料與損失函數結構規範

1. **真實 Cohort 捕捉**：
   在既有 30k positions 上，完全依據原始凍結 Router 在各位置即時計算產出的 $\alpha_e(x)$ 進行加權，嚴禁使用靜態常數或平滑平均。
2. **錨定損失正規化（Anchor Normalization）**：
   $$\mathcal{L}_{\text{anchor}} = \sum_{e \in R} \frac{\alpha_e}{E_e} \| y_e^{\text{student}} - y_e^{\text{teacher}} \|_2^2$$
   其中 $E_e = \mathbb{E}_{\text{TRAIN}}[\|y_e^{\text{teacher}}\|_2^2] + \epsilon$ 為專家在訓練集上的**全域固有能量常數**。
   **嚴禁使用單樣本動態除法 $\frac{1}{\|y_e(x)\|_2}$**，防止因分母接近零而產生數值爆炸。

---

### 4.2 三階段有限預算分配協議（Parameterized Budget Protocol）

設總訓練步數預算為 $B$ 步，Batch Size 為 $N$：

```
[Phase 1: Full-Down Warmup] (預算: 0.3 B)
  動態全 Down (537M 參數實際配置) ──> 軟遮罩連續鬆弛 ──> 專注於讓 GU 池適應全局特徵
                                                                │
                                                                ▼
[Phase 2: Mask Hardening] (預算: 0.4 B)
  引入 Top-q STE 硬二值化 ──> 加入 Cohort Slot 懲罰項 ──> 溫度退火誘導群組協同聚集
                                                                │
                                                                ▼
[Phase 3: Fixed-Support QAT] (預算: 0.3 B)
  硬支援集永久凍結 (釋放無效 Down, 縮容至 134M) ──> 啟用 Canonical W8 模擬 ──> 最終權重收斂
```

- **記憶體真實性澄清**：在 Phase 1 與 2，系統必須**真實配置 537M 參數**進行全連接微調；Phase 3 凍結後方可物理縮容至部署級別的 134M 參數。

---

### 4.3 驗證分離與數值後端契約

1. **進度檢查 vs. 最終驗證**：
   - *進度檢查（Progress Check）*：僅允許在 TRAIN 集保留的 Holdout 分割上評估損失下降。
   - *科學品質驗證（Fresh GT Quality）*：僅在訓練完全終止、硬遮罩與權重離線鎖定後，於未曝光的驗證集上執行單次評估。
2. **計算精度契約**：
   - 模擬驗證時，若在 Host 端使用 F64 累積 INT8 點積，整數累加至 $2^{53}$ 皆為 Bit-exact；
   - 若在 F32 下進行累積，數值超過 $2^{24}$（$16,777,216$）將觸發 ULP 捨入分歧。最終生產驗證必須以目標 Canonical 後端的真實整數 MAC 為準，不將 PyTorch 浮點模擬誤差誤認為模型架構缺陷。

---

### 4.4 技術故障與假說否證的嚴格界線

- **技術故障（Technical Failure / Hold）**：
  若在 Phase 1 出現梯度 NaN、數值溢位、或因學習率未對齊導致的發散，屬於 Recipe 實作層面問題，**觸發 Technical Hold，不消耗 Round 3 寶貴的實驗失敗次數**。
- **假說否證（Scientific Falsification）**：
  若訓練完整執行完畢三個階段，在固定的 MWPM 實體佈局下：
  1. 實體感測波次未顯著低於 Control 1（未能突破 MWPM 後排程極限）；或
  2. 為了達到 Wave 節省，最終模型在校準集上的 GT NLL 相較於 Native 基準劣化超過預定容許值。
  **此時判定候選 B 的「Wave-aware Joint Learning」假說遭到科學否證**。

---

## 5. 獨立審查判斷與 Block 06 核心問題

### 5.1 獨立現狀判斷
本候選在經過 Block 04 與 05 的深度推導後，**結構幾何、MWPM 等價性、梯度飽和機制與三階段訓練協議已足夠明確，具備在 Block 06 進行理論收斂的條件**。
目前未發現不可逾越的先驗數學自相矛盾，但其最大的科學威脅依然存在：
> **「在真實 Transformer 的稀疏路由分佈下，MWPM 後排程是否已經拿走了絕大部分實體 Wave 節省紅利，使得反向傳播中的 Joint Slot Objective 成為邊際效用極低的過度設計？」**

### 5.2 Block 06 唯一核心理論問題
> **「在給定模型品質容許擾動 $\Delta \text{NLL} \le \delta$ 下，透過理論泛函分析或 Pareto 極限推導，Joint Learning 相對於『純品質池化 + MWPM 最佳佈局』所能額外減少的實體感測波次 $\Delta N_{\text{slot}}$ 的理論上界是多少？若該理論上界 $\Delta N_{\text{slot}} < 1.0\text{ Wave}$，我們是否應立即對 Joint Objective 發起 Thought-Veto？」**
