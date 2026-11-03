# Round 3 候選思考 B Block 02：高階 Union Objective 必要性、Host-Cached 翻轉與有限蒸餾架構形式化備忘錄

**記錄標識**：`round03-thinking/gemini_formal_b_block02_union_host_distill.md`（純文字交付，無任何工具呼叫、無檔案讀寫、無計時宣稱）  
**當前狀態**：Round 3, Started: 2, Failures: 2, Remaining: 8。本區段接續 B formal block 01，專注於研究問題 B 之核心可否證性、高階目標函數必要性、主機記憶體階層翻轉威脅及善意蒸餾協議。

---

## 0. B01 核心錯誤全面修正清單

1. **有限群組構造邊界修正**：
   - B01 中提出的「$C=4$ 加上 8 個專家各 4 個私有群組」需要 $4 + 8 \times 4 = 36$ 個群組，**嚴格超過了 $G=32$ 的總池預算**；若推廣至全 64 個專家更需 260 個群組，此構造在 $G=32$ 空間中無效。
   - 在 $M=4I, s=I, K=8$ 條件下，單一 Token 總共需要 $8 \times I = 8I$ 的神經元支援。因全池僅有 $4I$ 個神經元，由鴿籠原理可知，被選中的 8 個專家**至少被迫重複計算 $8I - 4I = 4I$ 的神經元**（即不可能完全 disjoint）。但必須嚴格澄清：**「被迫重用」是壓縮幾何下的必然結果，絕非「模型品質必然可保」的證據**。
   - 小維度範例修正：若 $I=2$，則 $M=4I$ 對應的原子數應為 **$M=8$**（B01 誤寫為 $M=4$）。
2. **PCoMoE 等價條件修正**：
   - 「PCoMoE 等價當且僅當固定 Cohort」陳述錯誤。當所有專家共享完全相同的中間支援集（$s=M$）時，任何動態路由組合均可達成全域 $U$ 重用。部分共享結構亦可將來源群組分配給多個目標專家，但必須對其 Path 路由改動、Down 量化動態範圍、訓練梯度干擾與 Metadata 查表成本公平計費。
3. **數值縮放與非齊次性語意修正**：
   - Up/Down 的逆向尺度縮放（$\gamma$ 與 $1/\gamma$）**僅在實數域（$\mathbb{R}$）嚴格恆等**；在 IEEE 754 F32 中，縮放會改變尾數對齊，引入 ULP 捨入分歧，不可聲稱 F32 bit-exact。
   - SiLU 非齊次性（$\text{SiLU}(cz) \neq c\text{SiLU}(z)$）是全域函數恆等式的否定，在特定孤立點（如 $z=0$ 或 $c=0$）可偶然相等。非齊次性僅證明「不能僅靠單一 Gate 向量透過純線性縮放表徵兩個任意不同 Gate 的 Teacher」，**但不能證明 $M=4I$ 的池空間無法表徵兩個 Teacher**（配置兩個獨立原子即可）。
4. **單位與容量精確核算修正**：
   - $18,874,368\text{ Bytes} = \mathbf{18\text{ MiB}} \approx 18.87\text{ MB}$（十進制），B01 誤將其標為 $18.87\text{ MiB}$。
   - OLM（$H=2048, I=1024, K=8$）全 Q4 理想形狀帳：未共享時 GU 為 $18\text{ MiB}$、Down 為 $9\text{ MiB}$、總計 $27\text{ MiB}$；全池 GU 體積為 $9\text{ MiB}$。
   - 活躍群組聯集與總讀取量：
     - $U_{\text{groups}} = 8 \implies \text{GU } 2.25\text{ MiB} + \text{Down } 9\text{ MiB} = \mathbf{11.25\text{ MiB}}$
     - $U_{\text{groups}} = 16 \implies \text{GU } 4.50\text{ MiB} + \text{Down } 9\text{ MiB} = \mathbf{13.50\text{ MiB}}$
     - $U_{\text{groups}} = 32 \implies \text{GU } 9.00\text{ MiB} + \text{Down } 9\text{ MiB} = \mathbf{18.00\text{ MiB}}$
   - 理論最高字節節省率為：
     $$1 - \frac{2s + Ks}{3Ks} = 1 - \frac{2 + K}{3K} = 1 - \frac{10}{24} = \frac{14}{24} = \mathbf{58.333\%} \quad (\text{非 } 58.7\%)$$
     *(註：以上皆為對齊形狀之理想推算，非真實 mixed Q4_K_M 來源張量之實測數據)*。
5. **實體頁封裝跨頁代價修正**：
   - 單個 Group（128 神經元）之 Q4_K raw byte 為 $294,912\text{ B}$。若 144B block 不跨頁，每個 16 KiB 頁僅容納 113 個完整 block，2048 個 block 需 $\lceil 2048 / 113 \rceil = \mathbf{19\text{ 頁}}$（非 18 頁）；若採緊湊跨頁排列雖可壓縮至 18 頁，但必須在硬體端配置跨頁拼接 FIFO、Plane 定址與 ECC 邊界對齊緩衝區。
   - Down 剛性地板僅適用於「未經額外跨專家有損壓縮之獨立 Down 矩陣族」；若引入動態 Cache 或新型編碼，Down 感測亦有降低空間，不可稱為宇宙普世不可壓縮量。
6. **主機層級與架構邊界修正**：
   - 單層 9 MiB 共享池**不能外推**為「全模型可在 Host LLC 中無痛常駐且延遲僅數十微秒」；亦不可強行假設池必須留置於 Flash。評估必須涵蓋 Host-All、Host-Cached-GU + Flash-Down 以及 In-Flash 完整管線三種對照。
   - 撤銷無實證支撐的「$M=I$ 必導致 +15% PPL」、「Clustering 是唯一希望」及「28 Groups / 5% PPL 死亡線」等先驗斷言。模型層數依規格確立：OLM 為 16 層，DeepSeek-V2-Lite 為 27 層（含 1 個 shared / 26 個 routed 層），拒絕後見之明挑選層數。

---

## 1. 高階 Union Objective 是否真有不同最佳解

### 1.1 形式化證明：固定基數下加性成本退化為常數
設群組選取為二元矩陣 $m_{e, g} \in \{0, 1\}$，其中 $e \in \{1, \dots, E\}$，$g \in \{1, \dots, G\}$。
在硬約束 $\sum_{g=1}^G m_{e, g} = s/T$ 下，對任意被 Router 選中的活躍專家集合 $R \subset \{1, \dots, E\}$（$|R| = K$）：
- **純加性稀疏成本（Additive Cost）**：
  $$\text{Cost}_{\text{add}}(R) = \sum_{e \in R} \sum_{g=1}^G m_{e, g} = \sum_{e \in R} \frac{s}{T} = K \cdot \frac{s}{T} = \text{常數}$$
- **結論**：純加性目標函數在固定專家基數下，對專家間如何重疊群組的梯度貢獻恆等於零，無法為「跨專家共享拓撲」提供任何優化牽引力。

### 1.2 [本段推導] 有限可行構造：Pairwise Surrogate 與 Cohort Union 之嚴格分歧

為證明高階 Union Objective 具備 Pairwise（成對協同）無法替代的獨立自由度，給出以下具體可行構造：
- **環境設定**：總群組數 $G = 4$，專家數 $E = 3$，路由選中集合 $R = \{1, 2, 3\}$（$K=3$），每專家支援度 $s/T = 2$。

#### 構造 Case A（1 個全域共享 + 3 個私有）
- 專家 1 支援集：$\mathcal{S}_1 = \{1, 2\}$
- 專家 2 支援集：$\mathcal{S}_2 = \{1, 3\}$
- 專家 3 支援集：$\mathcal{S}_3 = \{1, 4\}$
- **單專家基數**：$|\mathcal{S}_1| = |\mathcal{S}_2| = |\mathcal{S}_3| = 2$。
- **成對交集大小（Pairwise Intersections）**：
  $$|\mathcal{S}_1 \cap \mathcal{S}_2| = |\{1\}| = 1, \quad |\mathcal{S}_1 \cap \mathcal{S}_3| = |\{1\}| = 1, \quad |\mathcal{S}_2 \cap \mathcal{S}_3| = |\{1\}| = 1$$
  成對重疊總和：$\sum_{i < j} |\mathcal{S}_i \cap \mathcal{S}_j| = 1 + 1 + 1 = \mathbf{3}$。
- **三重交集**：$|\mathcal{S}_1 \cap \mathcal{S}_2 \cap \mathcal{S}_3| = |\{1\}| = 1$。
- **活躍聯集大小（True Cohort Union）**：
  $$\mathcal{U}_A = \mathcal{S}_1 \cup \mathcal{S}_2 \cup \mathcal{S}_3 = \{1, 2, 3, 4\} \implies |\mathcal{U}_A| = \mathbf{4}$$

#### 構造 Case B（3 個成對共享 + 0 個三重共享）
- 專家 1 支援集：$\mathcal{S}_1 = \{1, 2\}$
- 專家 2 支援集：$\mathcal{S}_2 = \{1, 3\}$
- 專家 3 支援集：$\mathcal{S}_3 = \{2, 3\}$
- **單專家基數**：$|\mathcal{S}_1| = |\mathcal{S}_2| = |\mathcal{S}_3| = 2$。
- **成對交集大小**：
  $$|\mathcal{S}_1 \cap \mathcal{S}_2| = |\{1\}| = 1, \quad |\mathcal{S}_1 \cap \mathcal{S}_3| = |\{2\}| = 1, \quad |\mathcal{S}_2 \cap \mathcal{S}_3| = |\{3\}| = 1$$
  成對重疊總和：$\sum_{i < j} |\mathcal{S}_i \cap \mathcal{S}_j| = 1 + 1 + 1 = \mathbf{3}$。
- **三重交集**：$|\mathcal{S}_1 \cap \mathcal{S}_2 \cap \mathcal{S}_3| = \emptyset \implies \mathbf{0}$。
- **活躍聯集大小**：
  $$\mathcal{U}_B = \mathcal{S}_1 \cup \mathcal{S}_2 \cup \mathcal{S}_3 = \{1, 2, 3\} \implies |\mathcal{U}_B| = \mathbf{3}$$

#### 數學分析與本質區分：
1. **Pairwise 代理損失之盲區**：若優化目標採用成對相關性代理（如 $\max \sum_{i < j} P(i, j \in R) |\mathcal{S}_i \cap \mathcal{S}_j|$），由於 Case A 與 Case B 的成對交集向量皆為 $[1, 1, 1]$，**成對目標函數完全無法區分 Case A 與 Case B**。
2. **True Cohort Union 之敏感性**：真實硬體讀取直接取決於活躍聯集 $|\mathcal{U}|$。Case B 的聯集大小為 3，Case A 為 4。**Case B 相比 Case A 減少了 25% 的實體讀取與計算**。
3. **品質與硬體成本的不對稱性警告**：
   - *品質層面*：Case A 動用了 4 個不同群組，其張成的特徵子空間維度可能高於僅動用 3 個群組的 Case B。不能因 Case B 節省讀取即假定其品質必定與 Case A 等價。
   - *NAND 實體映射層面*：若 Case B 的三個群組恰好分佈在不同 Plane，則可 1 次 Multi-Plane 感測完成；若 Case A 的四個群組均勻分佈在 4 個 Plane，其感測延遲亦為 1 次感測。此時「群組數減少 1」**並不一定能轉化為物理感測波次（Wave）的減少**。

#### 嚴格對照組實驗設計建議：
為驗證高階 Union Objective 的真實必要性，必須設定以下同預算對照組：
- **Control 1（獨立最佳化 + 後排程映射）**：各專家獨立擬合支援集，固定支援度後，利用離線圖分割演算法對 32 個群組進行 Plane 物理重排。
- **Control 2（Pairwise 協同正則化）**：訓練時僅加入二階成對共現獎勵 $\sum_{i, j} P(i, j) |\mathcal{S}_i \cap \mathcal{S}_j|$。
- **Treatment（真實 Cohort Page Objective）**：直接以 $\mathbb{E}_{R \sim \pi(x)} [ \text{Pages}(\bigcup_{e \in R} \mathcal{S}_e) ]$ 作為訓練懲罰項。
只有當 Treatment 在維持相同 PPL 條件下，物理 Page/Plane 衝突波次顯著低於 Control 1 與 Control 2 時，高階 Union Objective 的研究價值方獲成立。

---

## 2. 全 Pool 運算與 Hybrid Host-Cached-GU 強控制分析

### 2.1 [本段推導] 均勻隨機選擇下的期望聯集閉式解
設全池包含 $M = 4I$ 個神經元，每個專家均勻隨機選取 $s = I$ 個神經元（選取率 $p = s/M = 0.25$）。
在連續獨立近似下，對於池中任意給定的神經元或群組 $j$，單一專家不選取它的機率為 $1 - p = 0.75$。
當 Router 選出 $K$ 個專家時，該單元未被任何專家選中的機率為 $(1 - p)^K = 0.75^K$。
因此，被選中神經元的期望活躍聯集為：
$$\mathbb{E}[U] = M \left( 1 - \left( 1 - \frac{s}{M} \right)^K \right) = 4I \left( 1 - 0.75^K \right)$$

- **對 OLM（$K=8$）**：
  $$0.75^8 \approx 0.1001 \implies \mathbb{E}[U] = 4I (1 - 0.1001) = 4I \times 0.8999 \approx \mathbf{3.60 I}$$
- **對 DeepSeek-V2-Lite（$K=6$）**：
  $$0.75^6 \approx 0.1780 \implies \mathbb{E}[U] = 4I (1 - 0.1780) = 4I \times 0.8220 \approx \mathbf{3.29 I}$$

#### 殘酷的幾何事實：
當 $K=8$ 時，隨機路由下的活躍聯集已佔全池的 **90.0%**（$3.60I / 4I$）。
動態精細追蹤活躍群組（計算 $3.60I$），與「直接將全池 $4I$ 完整計算」（Dense Full-Pool Compute），**兩者在計算量上僅僅相差 10%**！
為這 10% 的動態節省，硬體必須支付：動態 Group Gather 邏輯、不規則稀疏索引緩衝區、非同步 Page 讀取排程與潛在的管線氣泡。

---

### 2.2 [本段推導] 全模型儲存足跡與 Host 記憶體階層覆蓋

在標準 Canonical W8 基準下進行全層核算（未計入非 FFN、KV Cache 與系統 Runtime）：
- **單層結構對比**：
  - 原始獨立 MoE（64 專家）：$3 \times E \times H \times I = 3 \times 64 \times H I = 192 H I$。
  - 共享池 MoE（$M=4I$）：GU 全池 $2 \times 4I \times H = 8 H I$；Down 私有 $64 \times I \times H = 64 H I$；合計 $(8 + 64) H I = 72 H I$。
  - 參數體積壓縮比：$72 / 192 = \mathbf{37.5\%}$。
- **全模型 Routed Bank 總容量**：
  - **OLM（16 層，$H=2048, I=1024$）**：
    - 原始 W8 容量：$16 \times 192 \times 2048 \times 1024 \times 1\text{B} = 6.442\text{ GB}$。
    - 共享池 W8 容量：$37.5\% \times 6.442\text{ GB} = \mathbf{2.416\text{ GB}}$。
    - **全模型 16 層 GU 全池總量**：
      $$16 \times 8 \times 2048 \times 1024 \times 1\text{B} = 268,435,456\text{ B} = \mathbf{256\text{ MiB}} \quad (\text{Q4\_K 僅 } 144\text{ MiB})$$
  - **DeepSeek-V2-Lite（26 個 Routed 層，$H=2048, I=1408$）**：
    - 共享池 W8 容量：$26 \times 72 \times 2048 \times 1408 \times 1\text{B} = \mathbf{5.398\text{ GB}}$。
    - **全模型 26 層 GU 全池總量**：
      $$26 \times 8 \times 2048 \times 1408 \times 1\text{B} \approx 600.3\text{ MB} = \mathbf{572.5\text{ MiB}} \quad (\text{Q4\_K 僅 } 322\text{ MiB})$$

---

### 2.3 [本段推導] Host-Cached-GU 對 In-Flash Page 假說的毀滅性翻轉

在目標為 **8 GB Inclusive Memory** 的系統中：
1. **全模型全主機常駐（Host-All）之可行性**：
   OLM 共享後全權重僅 2.42 GB（Q4_K 下更僅約 1.36 GB），加上 KV Cache 與 Runtime 亦能**完全常駐於 8 GB Host DRAM 中**。在此情境下，Flash 甚至不需要被頻繁讀取，In-Flash 計算命題失去主要動機。
2. **混合架構（Hybrid Host-Cached-GU + Flash-Down）之極限威脅**：
   即使對於 DeepSeek-V2-Lite（Routed Bank 5.4 GB），其全模型 GU 全池在 Q4_K 下僅佔 **322 MiB**（在 8 GB 中佔比 $<4.1\%$）：
   - Host 記憶體以極小代價常駐全部 26 層的 GU 池；
   - 每個 Token 抵達時，Host CPU/NPU 以連續稠密矩陣乘法極速完成該層 $4I$ 個神經元的前向計算；
   - Flash 僅需承擔 Top-$K$ 專家的 Down 矩陣讀取（每次讀取固定為 $K \times I \times H$）。
- **硬體核心審問**：
  若 GU 完全在 Host 計算，則**快閃記憶體中根本不存在 GU 實體頁的讀取操作**！
  如此一來，耗費龐大算法精力進行的「Route-Cohort Page/Plane 共同存取優化」，其在硬體上的優化對象直接歸零。
- **In-Flash GU 唯一存活的嚴苛邊界**：
  In-Flash 運算要保住其存在價值，必須處於以下邊界條件：
  - *邊界 A（超低成本 DRAM-less 環境）*：Host 端可用 RAM 極度匱乏（如 $<512\text{ MiB}$ 的微控制器或超低功耗端側設備），無法劃分出 144~322 MiB 空間常駐 GU 池。
  - *邊界 B（超大模型維度）*：當模型擴展至 $H=8192, I=4096, M=4I=16384$ 時，GU 池全模型達到數十 GB，Host RAM 無法承載。
  - *邊界 C（Host-Flash 匯流排極度窄頻）*：Host 與 Flash 間的介面頻寬極低，若在 Host 算完 GU，將 $4I$ 或 $K \times I$ 維度的 intermediate activations 傳輸給 Flash Controller 執行 Down 投影所花費的時間，超越了 Flash 直接在內部感測並計算 GU 的時間。

---

## 3. 可執行且善意的有限訓練假說與早期 Kill-Test

### 3.1 參數量級與最佳化本質
以 OLM 單層為例：
- 共享 GU 池參數：$2 \times 4096 \times 2048 = \mathbf{16.78\text{M}}$ 參數。
- 64 個私有 Down 矩陣：$64 \times (1024 \times 2048) = \mathbf{134.22\text{M}}$ 參數。
- 單層待訓練參數量高達 **151.0M**。這是一個標準的中型神經網路聯合蒸餾任務，絕非微不足道的閉式 Ridge 回歸。

---

### 3.2 兩種有限蒸餾 Recipe 對比

#### Recipe (a)：凍結 Teacher 神經元字典 + 結構化支援集選擇 + Down 閉式回歸
- **步驟 1（字典建構）**：抽取原始 64 專家共 65,536 個配對神經元，依激活方差與特徵餘弦進行加權聚類，固定產出 $M=4096$ 個凍結原子，並指派為 32 個 Group。
- **步驟 2（支援集選擇）**：在給定凍結原子下，利用貪婪前向選擇（Forward Greedy）或 Group Lasso 為各專家確定 8 個 Group。
- **步驟 3（Down 權重求解）**：凍結 GU 池與支援集，對每個專家在校準集上的中間表徵 $\Phi_{\mathcal{S}_e}(X)$ 求解嶺回歸（Ridge Regression）：
  $$V^{(e)} = Y_{\text{teacher}}^{(e)} \Phi_{\mathcal{S}_e}^\top \left( \Phi_{\mathcal{S}_e} \Phi_{\mathcal{S}_e}^\top + \lambda I \right)^{-1}$$
- **優缺點與否定歸因**：計算確定且極快；但若此 Recipe 失敗，**僅能否定「凍結 Teacher 字典假設」**，不能否定「可聯合微調的動態池空間族」。

#### Recipe (b)：Teacher 激活響應初始化 + 有限聯合梯度最佳化
- **步驟 1（初始化）**：以聚類原子作為 GU 初始值，以嶺回歸解作為 Down 初始值。
- **步驟 2（連續鬆弛訓練）**：引入連續群組選擇參數矩阵 $S \in \mathbb{R}^{E \times G}$。採用 Gumbel-Softmax 或 Straight-Through Estimator 生成 8-hot 遮罩。
- **步驟 3（反向傳播）**：在固定步數（如 5000 步）內，以 AdamW 聯合更新 GU 權重、Down 權重與群組參數 $S$。
- **否定歸因**：若 Recipe (b) 在合理學習率網格搜索下依然發散，則可正式質疑「$M=4I$ 非線性特徵池在該模型層上的表徵容量下界」。

---

### 3.3 損失函數設計：混合損失與單專家錨定

為防止「跨專家抵消效應（Cross-Expert Cancellation）」，蒸餾目標必須結合群組輸出與單專家錨點：
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{cohort-mix}} + \lambda_{\text{anchor}} \mathcal{L}_{\text{expert-anchor}} + \beta \mathcal{R}_{\text{cohort}}$$

1. **Cohort-Mix 損失**：
   $$\mathcal{L}_{\text{cohort-mix}} = \left\| \sum_{e \in R} \alpha_e y_e^{\text{student}} - \sum_{e \in R} \alpha_e y_e^{\text{teacher}} \right\|_2^2$$
2. **單專家錨定項（Anchor Regularization）**：
   $$\mathcal{L}_{\text{expert-anchor}} = \sum_{e \in R} \alpha_e \left\| y_e^{\text{student}} - y_e^{\text{teacher}} \right\|_2^2$$
   - 藉由強制約束 $\lambda_{\text{anchor}} > 0$，確保即使專家 $A$ 與 $B$ 在特定樣本上同時啟動，兩者亦各自逼近真實 Teacher 輸出，消除梯度的代數相互掩蓋。
3. **稀有專家採樣覆蓋策略**：
   - 嚴禁使用純粹的自然 Routed 採樣（長尾專家因出現率 $<0.1\%$ 將完全欠擬合）。
   - 採用**溫度平滑路由採樣（Temperature-Smoothed Sampling）**：在離線建構蒸餾 Buffer 時，依據軟化機率 $P_{\tau}(e \in R) \propto (\alpha_e)^{1/\tau}$（取 $\tau = 2.0$）進行重要性採樣，人為提升罕見專家的曝光率。
   - **品質損害判定界線**：在 TRAIN 評估中，若稀有專家的相對重構誤差超過熱門專家 3 倍以上，但未對總體加權 Token Loss 造成顯著推升，屬統計可接受範圍；若稀有專家出現激活完全歸零（Dead Expert），則判定為科學品質受損。

---

### 3.4 單一最早必要科學檢驗（Earliest Scientific Kill-Test）

#### 檢驗規範（純提案，嚴禁本次執行）：
- **測試目標**：選取 OLM 中間具備典型特徵的單一層（**Layer 8**，16 層之正中位置），凍結其餘 15 層。
- **訓練數據**：嚴格限制於既有 TRAIN 16PG（約 30k positions）的子集，**絕對不讀取任何 VAL 或 TEST 資料**。
- **參照橋樑（Reference Bridges）**：
  1. *Bridge 0（原始基準）*：原生 Native Q4_K Teacher 輸出。
  2. *Bridge 1（Canonical B）*：維持獨立 Q4_K Down，但 GU 採用未共享的標準寬度。
  3. *Bridge 2（Candidate B）*：$M=4096, G=32, s=8$ 的共享池 Student。
- **核心觀測指標**：
  1. 該層輸出張量相對於 Bridge 0 的相對均方誤差（Relative Frobenius MSE）：
     $$\text{RelMSE} = \frac{\|Y_{\text{student}} - Y_{\text{teacher}}\|_F^2}{\|Y_{\text{teacher}}\|_F^2}$$
  2. 私有 Down 矩陣的平均條件數（Condition Number）：$\kappa(V^{(e)})$，監控 Down 權重是否發生數值病態。
  3. 訓練收斂時的平均活躍群組數：$\bar{U} = \mathbb{E}[|\bigcup_{e \in R} \mathcal{S}_e|]$。

#### 否定性判定標準（Kill Criteria）：
1. 若在固定 5000 步訓練後，Candidate B 的 RelMSE **大於 Bridge 1 的 2.0 倍**，且伴隨 Down 條件數激增超過一個數量級（表明私有 Down 竭盡全力仍無法擬合畸變的 GU 特徵），則**候選假說遭到強烈否定**。
2. 若 RelMSE 達標（$<0.05$），但學習出的群組遮罩呈現極端發散，使得平均活躍群組數 $\bar{U} \ge 30$（在 32 群組中重疊率幾乎為零，與全池無異），則**「動態群組可落實共享」假說遭到否定**。

#### 正向結果後的接續測試：
若單層檢驗通過（RelMSE 受控且 $\bar{U} \le 18$），下一步**不是直接進行全模型 16 層訓練**，而是：
> **「將學得的 18 個活躍群組（共 $18 \times 18 = 324$ Pages）代入 16-Chip 4-Plane 物理定址模擬器中，檢驗其 Multi-Plane 同步讀取波次是否真能壓低至 $\le 6$ Waves。」**
只有物理頁排程被證實有效，全模型擴展才具備工程與科學意義。

---

## 4. 本區段總結：三組結論與 Thought-Veto 邊界

### 4.1 已確認結論 (Confirmed)
1. **加性成本退化定理**：在固定每專家基數約束下，純加性稀疏正則化對重疊拓撲完全鈍化；而高階 Union Objective 具備成對（Pairwise）代理無法區分的結構敏感性（如 Case A 與 Case B 之本質差異）。
2. **隨機路由聯集幾何**：在 $M=4I, s=I, K=8$ 下，均勻隨機選擇導致單 Token 活躍聯集高達全池的 **90.0%**。動態群組選取相較於全池稠密運算，理論計算量差距僅 10%。
3. **儲存足跡真實尺度**：OLM 全模型 16 層的 GU 總量僅為 **144 MiB（Q4_K）/ 256 MiB（W8）**，DeepSeek-V2-Lite 僅為 **322 MiB（Q4_K）**。

### 4.2 被否定/已淘汰主張 (Rejected / Refuted)
1. **否定「In-Flash 頁排程優化在所有系統層級皆為必備」**：在具備 8 GB DRAM 的端側系統中，Host 常駐全部 GU 池（僅佔 $<4.1\%$ RAM）並以 Host 執行前向計算，將使 Flash 端完全無需讀取 GU 頁面，使 In-Flash GU 排程研究失去物質載體。
2. **否定「稀有專家可用自然 Routed 採樣完成蒸餾」**：自然長尾分佈必將導致稀有專家因樣本貧瘠而嚴重失真，必須引入溫度平滑採樣與單專家錨定損失。

### 4.3 嚴格 Thought-Veto 評估與開放問題
- **Thought-Veto 評估**：
  若本研究被限定在「一般具備 $\ge 4\text{ GB}$ Host DRAM 之邊緣加速卡/主機」場景，則基於 **Host-Cached-GU 壓倒性優勢**，本候選應被 Thought-Veto；
  但若研究範疇包含「**極限 DRAM-less（無外部記憶體）智慧儲存控制器**」（即控制器內僅有數十 MiB SRAM，完全依賴 NAND 串流運算），則 In-Flash 頁面分佈依然是決定性瓶頸。
- **接續 Block 03 核心開放問題**：
  > **「在 DRAM-less 控制器假定下，若將 32 個 GU Group 物理條帶化（Striped）分散至 16 Chips $\times$ 4 Planes，是否存在一種靜態 Placement 策略，能使自然語言中常見的 Top-8 專家協同聯集，其實體感測波次（Sensing Waves）的變異數最小化？」**
