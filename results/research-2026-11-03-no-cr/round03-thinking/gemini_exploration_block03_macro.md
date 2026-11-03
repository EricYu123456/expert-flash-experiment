## 0. 前序備忘錄勘誤與邊界澄清（800 字內）

1. **$g4$ 與 $g6$ 查表次數修正**：先前「1 fetch / 4 weights」表述錯誤。在位元平面（Bit-plane）查表模型下，$g4$ Q4_K 的 64 組 4-input 表需按 4 個 bit-plane 分別檢索，每 256 weights 實際為 $64 \times 4 = 256$ 次 fetch；Q6_K 的 $g2$（128 組 2-input）對 6 個 bit-plane 需 $128 \times 6 = 768$ 次 fetch，查表頻寬開銷未被低估。
2. **Batch=1 查表攤提修正**：在單 Token 生成中，輸入 activation 雖只是一組向量，但在同一層內被**該 Token 啟動的所有 Output Rows 及所有選中 Experts 共享**。因此 $T256$ 的建表開銷可跨 $R$ 及 Top-$K$ 專家完整攤提，非單次計算即棄。
3. **F32 精度計量修正**：在區間 $[2^{24}, 2^{25}]$ 內，F32 的 ULP 實際為 $2.0$。先前數值差為 $2.0$，實為 **1 ULP**（非 2 ULP）；其非結合性核心本質在於中間捨入導致的尾數截斷。
4. **實體頁與 ECC 邊界**：實體 16 KiB 頁整除 block，**不保證** 1 KiB/2 KiB 的 ECC Codeword 剛好整除 block。Codeword 邊界與量化區塊交錯時，管線仍需跨 Codeword 拼裝暫存。
5. **In-Flash 匯流排路徑修正**：若運算直接位於 In-Flash / Near-Flash 控制器端，其權重直接自 Page Buffer 或內部 ECC 引擎取用，**不得無條件強加外部 NV-DDR 晶片間傳輸延遲（40 $\mu$s）為必付代價**。
6. **工作集數值與 PPA 依賴修正**：先前暫存器加總存有代數算術錯誤，不可作剛性下界；無具體晶圓廠 Library 支撐的「250–350 GE / <2 ns」屬無效常數，不可據此斷言 Direct MAC 必勝。
7. **Root Veto 核心本質**：在相同儲存格式、相同頁幾何與相同前進邊界下，Direct MAC 與 LUT 共享相同資料流，目前**缺乏 NAND 獨有的微架構不對稱優勢**，本質退化為標準單元面積常數競爭。

---

## 1. 「共享非線性神經元 Atom Pool」架構深入數學推導

### 1.1 參數化模型定義
設模型結構如下：
- $H$：隱藏層維度（Hidden dimension，如 2048）
- $I$：單一專家之中間維度（Intermediate dimension，如 1024）
- $E$：Routed 專家總數（如 16 或 64）
- $K$：Top-$K$ 啟用專家數（如 2 或 4）
- $M$：跨專家共享的非線性神經元 Atom Pool 總數（$I \le M \le E \times I$）。每個 Atom 包含一對配對的 Gate 與 Up 權重行：$\mathbf{w}_{\text{gate}}^{(m)}, \mathbf{w}_{\text{up}}^{(m)} \in \mathbb{R}^{H}$。
- $d$：Down 投影矩陣的稀疏係數密度（Sparsity density，$d \in (0, 1]$）。若為稠密重組則 $d=1$。
- $\mathcal{S}_k \subset \{1, \dots, M\}$：專家 $k$ 所依賴的 Atom 子集，大小 $|\mathcal{S}_k| = I$。

### 1.2 構造性例子：可學（Learnable）與不可學（Unlearnable）

#### A. 可學例子（Constructive Learnable Example）
- **場景**：語義特徵解耦與任務路由重組。
- **構造**：在多語言或多領域對齊模型中，不同領域專家的高階特徵偵測器（如「程式碼括號匹配」、「實體名詞邊界」）具備高度共性。設 $M = 2I$（僅為獨立專家的 $2/E$ 大小）。不同專家 $k_1$ 與 $k_2$ 的 Gate/Up 神經元在 feature space 中存在大量共線單元。
- **機制**：共享相同的非線性響應：
  $$z_m = \text{SiLU}(\mathbf{w}_{\text{gate}}^{(m)\top} \mathbf{x}) \cdot (\mathbf{w}_{\text{up}}^{(m)\top} \mathbf{x}), \quad m \in \bigcup_{k \in \text{Top-}K} \mathcal{S}_k$$
  各專家的專異性完全由 Down 投影係數 $\mathbf{V}^{(k)} \in \mathbb{R}^{H \times M}$ 實現：$\mathbf{y}_k = \mathbf{V}^{(k)} \mathbf{z}$。在此結構下，神經元的表徵基底共享，梯度能透過多個專家的 Down 投影反向聯合更新，模型困惑度（PPL）損失小於可壓縮閾值。

#### B. 不可學例子（Constructive Unlearnable Example）
- **場景**：正交互斥的數值啟動域（Disjoint Activation Domain）。
- **構造**：設專家 $A$ 專精於長文本符號歸納（Induction Heads，需要高度銳利、接近階梯函數的 SiLU 飽和區），專家 $B$ 專精於平滑機率預測（需要線性區附近的微小擾動響應）。
- **衝突**：若強行將兩者的 Gate/Up 配對神經元約束於共享池中（$M \ll E \times I$），由於 SiLU 具有嚴格的點態非線性，Down 投影的線性組合**無法改變中間特徵的零點與曲率分佈**。聯合蒸餾時，專家 $A$ 與 $B$ 的梯度產生劇烈正交干擾（Gradient Interference），導致中間特徵坍縮為平均低頻分量，PPL 出現不可逆的發散（類似 R2 實驗中超越 +15% 的破壞）。

### 1.3 讀取、計算與品質的剛性 Trade-off 矩陣

#### A. 標準 MoE（無 Atom Pool）每 Token 負擔：
- **Gate/Up 讀取量**：$2 \times K \times H \times I \times B_{\text{weight}}$
- **Down 讀取量**：$K \times I \times H \times B_{\text{weight}}$
- **總讀取量**：$3 K H I B_{\text{weight}}$
- **非線性計算量（FLOPs）**：$K \times I$ 次 SiLU 及元素乘法。

#### B. Atom Pool 結構每 Token 負擔：
定義活躍 Atom 聯集大小 $M_{\text{active}} = |\bigcup_{k \in \text{Top-}K} \mathcal{S}_k|$，滿足 $I \le M_{\text{active}} \le \min(M, K \times I)$。
- **Gate/Up 讀取量**：$2 \times M_{\text{active}} \times H \times B_{\text{weight}}$
- **Down 讀取量**：
  若 Down 矩陣稀疏化（每專家僅自 $M_{\text{active}}$ 中選取 $I$ 個 Atom 進行重組，非零項密度 $d = I / M_{\text{active}}$）：
  $$\text{Down Read} = K \times (d \cdot M_{\text{active}}) \times H \times B_{\text{weight}} = K \times I \times H \times B_{\text{weight}}$$
  若 Down 矩陣維持稠密連接至整個聯集（$d = 1$）：
  $$\text{Down Read} = K \times M_{\text{active}} \times H \times B_{\text{weight}}$$
- **非線性計算量**：$M_{\text{active}}$ 次 SiLU，**嚴格小於或等於** 原版 $K \times I$。

#### C. Trade-off 判別式：
令讀取節省比例為 $\Delta_{\text{read}}$：
$$\Delta_{\text{read}} = \frac{3 K H I - (2 M_{\text{active}} H + K \cdot d \cdot M_{\text{active}} H)}{3 K H I} = 1 - \frac{M_{\text{active}}}{K I} \left( \frac{2 + K \cdot d}{3} \right)$$
1. **讀取節省條件**：必須滿足 $\frac{M_{\text{active}}}{K I} < \frac{3}{2 + K \cdot d}$。
   - 若 Down 為稠密（$d=1$），當 $K=2$ 時，需 $\frac{M_{\text{active}}}{2I} < \frac{3}{4} \implies M_{\text{active}} < 1.5 I$。即兩個選中專家重疊的 Atom 必須超過 50%，否則**總讀取量反而上升**（因為 Down 矩陣膨脹）。
   - 若 Down 為稀疏（$d \cdot M_{\text{active}} = I$），則 $\Delta_{\text{read}} = \frac{2}{3} \left(1 - \frac{M_{\text{active}}}{K I}\right)$，讀取節省完全由 Gate/Up 聯集的重疊度決定。
2. **與普通載入器（Host CPU）的利益對稱性審查**：
   若模型被重構成 Atom Pool 格式，**普通 Host 記憶體載入器同樣能只讀取一次 $M_{\text{active}}$ 的 Gate/Up 權重**！
   - *誠實承認*：此演算法數學節省**並非 NAND In-Flash 所獨占**。若要在硬體上成立，必須證明 NAND 的實體組織（如 Plane 分佈）能使 $M_{\text{active}}$ 具有局部連續讀取特性，而 Host 受限於隨機 Gather 頻寬。

---

## 2. 六個互斥的 MoE In-Flash 巨觀架構候選探索

---

### 方向 1：Atom-Pool 聯集特徵串流與硬體分散式 Top-$K$ Down 規約

#### 1. 核心機制
將全模型所有專家的 Gate/Up 神經元去重收斂為實體儲存於 16 顆 Flash Chip 中的 Atom Pool。控制器根據 Router 產生的 Top-$K$ 專家清單，由硬體計算出 $M_{\text{active}}$ 的聯集索引，在一次 Flash 讀取週期內完成 Gate/Up 特徵計算並廣播；各 Chip 本地僅存儲對應專家的稀疏 Down 投影行，直接在本地將特徵加權累加，最後僅匯總最終 $H$ 維輸出。
- **省去的必要工作**：省去重複專家的 Gate/Up 權重感測（$t_R$）與快閃記憶體匯流排傳輸，省去多個專家重複執行 SiLU 的計算與暫存。
- **有無 Shared Expert 適用性**：皆適用。有 Shared 時，Shared 本質為 Atom Pool 中永遠包含的常駐子集。

#### 2. 為何普通 Loader 無法完全獲取相同收益？
普通 CPU/GPU 透過 DRAM 存取時，稀疏 Down 矩陣的索引乘法會引發嚴重的非連續記憶體 Gather 開銷與 Cache Miss；而在 In-Flash 架構下，Down 矩陣權重依 Chip/Plane 靜態排列，特徵向量以硬體廣播方式流經乘累加單元，屬於串流 Inner-Product，不產生隨機記憶體定址代價。

#### 3. 最致命直覺反例
**「專家專異性坍縮」**：若為追求高讀取節省而強行壓低 $M$（例如 $M < 2I$），導致所有專家輸出的中間表徵高度同質化，Top-$K$ 專家的線性組合實質退化為單一 Dense FFN，模型丧失 MoE 容量優勢，PPL 劇烈劣化。

#### 4. 最強 Prior（無捏造）
- 模型層面：MoE-Dropping, Shared-Neuron MoE (如 DeepSeek 系列中的 Shared/Routed 混合架構)。
- 硬體層面：Sparse GEMM Accelerator with Broadcast Activation (如 Cambricon-X/EIE 類似廣播資料流)。

#### 5. 30 分鐘可證偽之 Earliest Kill-Test
**「靜態重疊率與 PPL 敏感度邊界推導」**：給定固定 $E=16, I=1024$，在數學上推導隨機或基於餘弦相似度聚類的 Atom 共享下，當聯集重疊率大於 40% 時，跨任務專家梯度的正交投影餘弦值是否必定低於理論容量下限，若是則直接否決。

---

### 方向 2：MLC/TLC 物理感測階差投機解碼（Speculative Asymmetric $t_R$ Sensing）

#### 1. 核心機制
NAND Flash 的 Multi-Level Cell 在物理感測時，不同位元頁的感測時間有巨大差異（例如 LSB 頁感測僅需 $15\sim 20\,\mu\text{s}$，而 MSB 頁需 $45\sim 60\,\mu\text{s}$）。
將權重或 Router 路由資訊做位元平面非對稱排列：
- 僅利用快速讀出的 **LSB 粗糙權重（1-bit 或 2-bit 估計）** 立即執行先行推論，同時計算下一層的 Router 權重預測；
- 若投機確認命中，則後續 MSB 頁感測被整流或取消；若不命中，則流水線無縫等待 MSB 補齊精度。
- **省去的必要工作**：直接消除部分高延遲 MSB 頁的物理感測時間（平均 $t_R$ 從 $28\,\mu\text{s}$ 降至更低）。
- **有無 Shared Expert 適用性**：皆適用。

#### 2. 為何普通 Loader 無法完全獲取相同收益？
普通 Loader 與 Host CPU 接收的是經由標準 Host 控制器讀出並封裝好的完整位元資料，**無法介入或利用 NAND 內部的 LSB/MSB 物理微秒級階差時序**；此機制高度依賴 NAND Controller 對 Die 內部感測指令的微碼控制。

#### 3. 最致命直覺反例
**「投機驗證失敗引發管線氣泡」**：低階位元重構的啟動誤差過大，導致 Router 預測的 Top-$K$ 分支頻繁猜錯，引發管線清空與重複讀取，平均延遲反而比同步確定性讀取更長。

#### 4. 最強 Prior
- 近年快閃記憶體非對稱感測優化研究（如 FAST/HPCA 上的 Asymmetric MLC/TLC Read Scheduling）。
- Bit-slice Speculative Inference（如 BitNet/Adaptive Precision DNNs）。

#### 5. 30 分鐘可證偽之 Earliest Kill-Test
**「LSB 角度投機成功率臨界值計算」**：推導在維持 PPL 擾動 $<1\%$ 下，至少需要多少量化位元才能確保 Router Top-$K$ 命中率 $>95\%$；若該位元數必須涵蓋全部位元平面，則 $t_R$ 節省確定為 0。

---

### 方向 3：儲存端全閉環零回傳層融合（Zero-Roundtrip Inter-Layer Fusion with In-Storage Residual Pipeline）

#### 1. 核心機制
傳統系統每次完成一個 FFN 層的矩陣乘法，必須將中間輸出經由匯流排傳回 Host CPU/GPU 執行 Residual Add、LayerNorm 與下一層的 Attention/Router 計算。
本機制在 Flash 控制器本地配置極小型的專用向量算子管線（包含 SiLU、低階 LayerNorm 與殘差加法器）：
- 當前層 Down 投影的局部輸出直接進入控制器 SRAM；
- 本地常駐當前 Token 的殘差向量（$2048 \times 2\text{B} = 4\text{ KiB}$），在 Flash 端直接完成 Residual 相加與下一層 Router Logits 的計算；
- 下一層選中專家的 Flash Page 讀取指令由 Flash 本地狀態機自主觸發，**全過程無中斷、無 Host 介入、無資料跨匯流排來回搬移**。
- **省去的必要工作**：完全消除每一層 FFN 運算在 Host 與 Flash 之間的兩次往返通訊延遲及匯流排序列化/反序列化開銷。
- **有無 Shared Expert 適用性**：完全適用。

#### 2. 為何普通 Loader 無法完全獲取相同收益？
普通 Loader 架構下，Flash 僅為單純 Block 裝置，計算必須在 Host 端完成。無論 Host CPU 如何優化，每一層都必須承擔 PCIe/NVMe 協定棧與匯流排往返的固定延遲（通常每層數微秒至數十微秒）。

#### 3. 最致命直覺反例
**「Attention 算子無法在儲存端閉環」**：Transformer 由 Attention 與 FFN 交替構成。即使 FFN 在 Flash 端閉環，Attention 仍需龐大的 KV Cache 與跨 Token Softmax，極難置於 Flash 控制器內。如果每層 Attention 仍必須在 Host 執行，則 FFN 閉環的往返節省將被 Attention 的交替調度完全打散。

#### 4. 最強 Prior
- Computational Storage Architecture（如 NVMe Computational Storage Task Force 標準）。
- In-Storage Transformer Layer Pipelining（如 FlashEmbed / DeepStore）。

#### 5. 30 分鐘可證偽之 Earliest Kill-Test
**「Amdahl 定律匯流排邊界測試」**：在固定 Attention 必須由 Host 執行的前提下，計算 FFN 單向卸載所節省的傳輸時間，是否被 Flash 控制器處理非線性與 Norm 的慢速時脈（200 MHz vs CPU 3 GHz）所反向抵消。

---

### 方向 4：基於專家協同啟動圖的物理 Plane 超平面分佈佈局（Graph-Optimized Multi-Plane Hyperplane Mapping）

#### 1. 核心機制
*(本方向完全不同於低秩分解、快取、預取或精度改動)*
在預訓練 MoE 模型中，專家之間的協同啟動（Co-activation）呈現高度不均勻的圖拓撲結構（某些專家對經常同時被 Top-2 選中，某些則永遠互斥）。
- 傳統配置將專家按 Row/Column 循序填入 Flash 頁面，導致 Top-$K$ 讀取時，多個專家的資料落在同一個 Die/Plane 內，引發嚴重的 **Physical Plane 衝突（Bank Conflict）**，不得不串行化讀取。
- 本機制將 MoE 路由歷史建模為超圖（Hypergraph），利用多路圖分割演算法，將**最常同時啟動的專家正交打散映射到不同的 16 個 Chips 與 4 個 Planes**；
- 使得任意高機率出現的 Top-$K$ 專家組合，其權重讀取皆能以單一 4-Plane 同步 Multi-plane Command 在單一 $t_R$ 週期內平行完成。
- **省去的必要工作**：消除因實體 Plane 碰撞所導致的多次序列化 $t_R$ 感測等待（將最差情況的 $K \times t_R$ 壓縮為理想的 $1 \times t_R$）。
- **有無 Shared Expert 適用性**：完全適用。Shared 專家常駐於特定無衝突的 Dedicated Plane。

#### 2. 為何普通 Loader 無法完全獲取相同收益？
DRAM 是具有極高隨機存取自由度的位址空間（Row Buffer 衝突代價為數十奈秒級）；而 NAND Flash 的 Plane 衝突代價是**數十微秒級（$28\,\mu\text{s}$）**，且受限於單一 Die 內部嚴格的共用 Wordline 控制硬體。普通 Loader 即使重排記憶體，也無法解決底層 Flash 的物理感測排程衝突。

#### 3. 最致命直覺反例
**「Token 分佈漂移引發圖分割失效」**：若推論時的文字領域發生漂移（Domain Shift），專家協同啟動拓撲被重塑，原先正交的專家組合產生新的 Plane 碰撞，硬體退化為最差串行讀取，且 Flash 唯讀特性使得物理佈局無法動態重組。

#### 4. 最強 Prior
- 檔案系統中的 Disk Striping 與 De-clustering 演算法。
- Multi-channel SSD Parallelism Optimization for Sparse Tensor Workloads。

#### 5. 30 分鐘可證偽之 Earliest Kill-Test
**「4-Plane 碰撞率理論下界分析」**：針對 $E=16, K=2$，在 16 Chips $\times$ 4 Planes 的維度下，計算任意專家啟動分佈在最優圖著色下，是否存在至少一組合理的啟動情境其 Plane Collision 機率必大於 50%。

---

### 方向 5：容錯型跳躍式 ECC 感測與神經網路雜訊遮罩（Error-Tolerant Direct In-Flash Compute with Selective ECC-Bypass）

#### 1. 核心機制
Flash 讀取的最大延遲除 $t_R$ 外，ECC 引擎的解碼延遲與吞吐量（6.4 GB/s）亦構成嚴重瓶頸。
本架構利用神經網路對低階權重局部隨機位元翻轉的內生容忍度：
- 對權重的高階位元（MSB 及 Scale/Min metadata）進行強保護並強制執行 ECC 校正；
- 對佔據大宗體積的低階 Mantissa/Nibbles，**完全繞過（Bypass）或僅執行極輕量檢錯（Parity-check）**，在 Flash 輸出暫存器中直接串流送入運算單元；
- 同時在累加器端配置異常值鉗位（Outlier Clamping）硬體，防止單一重大 Bit-flip 導致數值溢位。
- **省去的必要工作**：消除大容量權重串流對 6.4 GB/s ECC 引擎的排隊等待時間，降低管線 Staging 記憶體需求。
- **有無 Shared Expert 適用性**：皆適用。

#### 2. 為何普通 Loader 無法完全獲取相同收益？
標準作業系統與 Host NVMe 驅動程式具有嚴格的資料完整性契約（Data Integrity Field, DIF/DIX），任何未經硬體 ECC 驗證的資料區塊將直接觸發匯流排 CRC 錯誤或系統中斷，作業系統無法接收「部分帶有隨機位元翻轉」的裸資料。唯有 In-Storage 深度整合的專用架構方可合法實作。

#### 3. 最致命直覺反例
**「隨機錯誤累積破壞量化分佈」**：在低位元量化（如 Q4_K）中，每個 nibble 的最高位若發生翻轉，其數值跳變相對於量化範圍極為巨大。少量此類錯誤在深層網路（如 32 層）中隨殘差流逐層放大，將迅速造成 Attention Softmax 或極值激增，導致輸出亂碼（PPL 爆炸）。

#### 4. 最強 Prior
- Approximate Storage for Deep Learning (如 DAC/ISCA 關於 MLC Soft-decision bypass 的系列論文)。
- Error-resilient Neural Network Acceleration with Raw Flash Data。

#### 5. 30 分鐘可證偽之 Earliest Kill-Test
**「BER-to-PPL 數值靈敏度解析計算」**：根據標準 NAND 壽命末期 Raw Bit Error Rate（RBER，例如 $10^{-3}$），計算在無 ECC 保護下，單一 256-block 出現 $>4 \sigma$ 擾動的統計機率；若該機率導致每生成 10 個 Token 必發生一次激增，則該方案直接判定死亡。

---

### 方向 6：物理跨層權重-激發對偶循環預充電（Dual-Buffer Interleaved Bank-Precharge Pipeline）

#### 1. 核心機制
利用標準 16-Chip NAND 架構中每個 Chip 內部的 Bank/Plane 獨立定址能力：
- 當 Layer $L$ 正在執行 FFN 的 Down 投影運算時，硬體調度器利用上一層 Attention 的已知趨勢，**提前向對應的 Flash Plane 發出 Layer $L+1$ Gate/Up 專家的預充電與感測指令（Pre-sense Command）**；
- 由於感測是陣列內部電容操作（$28\,\mu\text{s}$），不佔用對外匯流排，此操作可與當前層的資料傳輸及運算 100% 重疊；
- 將 $t_R$ 的暴露延遲近乎完全隱蔽於前序計算中，使系統端到端延遲趨近於純傳輸與計算極限。
- **省去的必要工作**：隱蔽掉每一層 MoE 專家的核心物理感測等待時間 $t_R$。
- **有無 Shared Expert 適用性**：皆適用。

#### 2. 為何普通 Loader 無法完全獲取相同收益？
普通作業系統驅動程式在進行 Host 讀取時，I/O Request 經過複雜的 Block Layer、佇列排程與中斷通知，其調度粒度與 Jitter 通常在數十微秒級以上，無法做到微秒級精確的 Pre-charge 陣列與計算管線剛性鎖步。

#### 3. 最致命直覺反例
**「Router 相依性屏障（Dependency Hazard）」**：Layer $L+1$ 的 Router 運算必須以 Layer $L$ 的最終輸出作為輸入。在 Layer $L$ 的 Down 投影及 Residual Add 完全計算完成前，**根本無法得知 Layer $L+1$ 究竟會啟動哪一個專家**！強行預充電只能依賴猜測，而對於 $E=16$ 選 $K=2$，猜中機率極低（$<15\%$），高額的預充電無效功耗與陣列衝突反使延遲翻倍。

#### 4. 最強 Prior
- Speculative Prefetching in Storage Systems。
- Branch-prediction-guided I/O Scheduling。

#### 5. 30 分鐘可證偽之 Earliest Kill-Test
**「預測覆蓋率 vs. 衝突罰時不平等式」**：推導在 Top-2 專家選擇下，若下一層預測準確率低於多少閾值時，預充電失誤所導致的 Plane 重新感測延遲將嚴格大於完全不預取的標準排程。

---

## 3. 綜合評估與後續深入方向篩選

在嚴格防範無效 PPA 假設的前提下，對上述 6 個方向進行第一階段審查：

| 方向名稱 | 核心代價節省 | 普通 Loader 隔絕性 | 致命弱點 | 存活狀態 |
| :--- | :--- | :--- | :--- | :--- |
| **1. Atom-Pool 聯集特徵串流** | Flash 讀取與計算雙降 | 中等（需依賴儲存端廣播） | 語義坍縮、Down 膨脹 | **保留候選 1** |
| **2. MLC/TLC 物理感測階差投機** | 降低平均感測時間 $t_R$ | 高（純物理層感測） | 投機錯誤引發管線氣泡 | 暫緩（需依賴特殊 Flash 韌體） |
| **3. 儲存端全閉環零回傳層融合** | 消除每層 Host 往返 | 極高（專用硬體管線） | Attention 算子撕裂閉環 | **保留候選 2** |
| **4. 協同啟動物理 Plane 超平面分佈** | 消除實體 Plane 讀取衝突 | 高（底層物理映射） | 領域漂移引發圖分割失效 | 備選（純排程，收益受限於 $K$） |
| **5. 容錯型跳躍式 ECC 感測** | 繞過 ECC 吞吐瓶頸 | 高（打破作業系統合約） | 隨機錯誤引發 PPL 數值崩潰 | 否決（數值風險過高） |
| **6. 跨層對偶循環預充電** | 隱蔽 $t_R$ 延遲 | 中等 | 嚴格的 Router 資料相依性阻斷 | 否決（因果律限制，命中率過低） |

### 篩選結論：選出 2 個值得在下一區段推進深入分析的候選方向
1. **方向 1（Atom-Pool 聯集特徵串流）**：
   - *未驗證假設*：假設在不損害模型基準困惑度（PPL 劣化 $<2\%$）的前提下，跨專家 Gate/Up 神經元可壓縮至重疊度 $\ge 40\%$ 的共享池，且 Down 投影的稀疏索引開銷能被硬體高效隱蔽。
2. **方向 3（儲存端全閉環零回傳層融合）**：
   - *未驗證假設*：假設在典型端側 Transformer 結構中，即使 Attention 仍在 Host 計算，僅對 MoE FFN 部分進行「本地殘差注入 + 本地 Router 觸發」的半閉環架構，其節省的端到端 PCIe/DMA 延遲仍能超越在 200 MHz 控制器上執行輕量級算子的時間代價。
