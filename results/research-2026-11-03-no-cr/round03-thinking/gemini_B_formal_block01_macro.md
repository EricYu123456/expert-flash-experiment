# Round 3 候選思考 B Block 01：固定 Router 與 Route-Cohort Page-Cost 訓練之共享非線性 GU 特徵池形式化推導

**記錄標識**：`round03-thinking/gemini_formal_b_block01_shared_gu_pool.md`（純文字直接交付，無任何工具呼叫、無檔案讀寫、無背景計時偽造）  
**狀態與時間依據**：Root formal start 2026-11-03T11:05:43.454450 UTC；前序 A 探索與歷史區段全部歸零不轉移。本區段為 **研究問題 B 的正式思考 Block 01**。未滿 1800 秒實質工作前嚴禁申請實驗採納；允許經嚴格推導後提出 Thought-Veto。

---

## 1. 新研究命題：Union Objective 與 Cohort 共享結構的必要條件

### 1.1 形式化問題定義
設輸入向量為 $x \in \mathbb{R}^H$。原始凍結之 Router 輸出 Top-$K$ 專家集合及其原始權重：
$$\mathcal{E}(x) = \{e_1, e_2, \dots, e_K\} \subset \{1, \dots, E\}, \quad \alpha_e(x) \in \mathbb{R} \quad (\text{不強制 } \sum \alpha_e = 1)$$
定義單一非線性配對原子（Paired G/U Atom）$\phi_j(x)$：
$$\phi_j(x) = \text{SiLU}(g_j^\top x) \cdot (u_j^\top x), \quad g_j, u_j \in \mathbb{R}^H, \quad j \in \{1, \dots, M\}$$
- **原子池尺度**：$M = 4I$。
- **群組量子（Group granularity）**：$T = 128$ 個神經元為一個 Group。全池共 $G = M / T = 4I / 128$ 個 Groups。
  - 對 OLM（$H=2048, I=1024, E=64, K=8$）：$M = 4096$，總群組數 $G = 32$；每專家硬支援度 $s = I = 1024 \implies$ 每個專家恰好選擇 $s/T = 8$ 個 Groups。
  - 對 DeepV2Lite（$H=2048, I=1408, E=64, K=6$）：$M = 5632$，總群組數 $G = 44$；每專家選擇 $s/T = 11$ 個 Groups。
- **Down 投影規則**：每個專家保留自己私有的 $H$ 維 Down 列向量（$V^{(e)} \in \mathbb{R}^{H \times s}$），嚴格禁止共享輸出原子（即 Down 權重不允許跨專家融合）。

### 1.2 [本段推導] Constructive 共享 $C$ 與私有 $P_e$ 構造 vs. 完全 Disjoint 反例

#### A. 構造性共享模型（Shared $C$ + Private $P_e$）
令全域 Group 集合為 $\mathcal{G} = \{1, \dots, G\}$。將每個專家 $e$ 的 8 個 Group 支援集 $\mathcal{S}_e \subset \mathcal{G}$ 分解為：
$$\mathcal{S}_e = \mathcal{C} \cup \mathcal{P}_e$$
其中 $\mathcal{C}$ 為高頻協同核心群組（Core Groups，$|\mathcal{C}| = c$），$\mathcal{P}_e$ 為專家私有群組（Private Groups，$|\mathcal{P}_e| = 8 - c$），且對任意兩專家 $e_a \neq e_b$，$\mathcal{P}_{e_a} \cap \mathcal{P}_{e_b} = \emptyset$。
- **Token 活躍聯集（Active Union）**：
  $$\mathcal{U}(\mathcal{E}(x)) = \bigcup_{e \in \mathcal{E}(x)} \mathcal{S}_e = \mathcal{C} \cup \left( \bigcup_{e \in \mathcal{E}(x)} \mathcal{P}_e \right)$$
- **活躍群組數**：
  $$U_{\text{groups}} = |\mathcal{U}(\mathcal{E}(x))| = c + K \cdot (8 - c) = 8K - c(K - 1)$$
  - 當 $c = 4$（半數共享）、$K=8$ 時：$U_{\text{groups}} = 64 - 4 \times 7 = 36$ 個 Groups。
  - 相對於無共享的 $8 \times 8 = 64$ 個 Groups，GU 特徵計算與讀取量減少了 $43.75\%$。

#### B. 構造性完全 Disjoint 反例（Zero-Reuse Counterexample）
設訓練未對齊 Cohort 成本，專家間群組完全隨機或相互排斥：
$$\forall e_a, e_b \in \mathcal{E}(x) \ (e_a \neq e_b), \quad \mathcal{S}_{e_a} \cap \mathcal{S}_{e_b} = \emptyset$$
此時 $U_{\text{groups}} = K \times 8 = 64$。
然而，總池大小 $M = 4I$ 僅有 32 個 Groups！在鴿籠原理（Pigeonhole Principle）下，當 $K=8$ 且每個專家選 8 個 Groups 時，**在數學上不可能做到完全 Disjoint**。
- **推導結論**：在 $M=4I, K=8$ 下，專家間的群組重疊是**結構性強制發生的**。關鍵在於：**這些重疊究竟是「受控地發生在同一個 Token 同時啟動的專家之間」，還是「隨機跨 Token 錯位重疊導致單一 Token 內部零重用」**。若選中的 8 個專家各自選取不同 Group 直到佔滿 32 個 Groups，則 $U_{\text{groups}} = 32$，此時 GU 雖然達到了池的上限，但相對於理論最佳值（$U_{\text{groups}} = 8$）產生了高達 $4\times$ 的膨脹。

### 1.3 [本段推導] PCoMoE 全域模擬條件與 Union Objective 必要條件
- **PCoMoE 邊界對照（Provided Source Fact）**：PCoMoE（arXiv:2609.01024）將一整個中間投影 $U_i$ 綁定並在多個專家間重用，但限制每個 $U_i$ 最多被重用給特定 $D_j$，且必須重新微調 Router。
- **本候選與 PCoMoE 的區別**：
  本研究中 Router **完全凍結**，Router 的選擇是任意的組合。
  1. *何時本架構等價於 PCoMoE？* 當且僅當 Router 的 Top-$K$ 路由在語義上退化為固定的專家聚類（例如 Expert 1~8 永遠同時被啟動，其餘永遠不被選中）。此時 Group 支援集 $\mathcal{S}_e$ 對該聚類完全重合，退化為塊狀靜態共享。
  2. *何時必須支付額外 Down 表徵與 Path 預算？* 當 Router 展現高熵隨機性（任意兩個專家皆有非零機率被共同選中）。此時，固定每個專家的 $s=I$，若要強迫共享，專家的 Down 投影矩陣必須學習在「非最優特徵子集」上重構輸出，這必然導致 Down 權重數值動態範圍劇增，需更高精度的 Down 表徵或引入額外的補償路徑。
- **Union Objective 產生非平凡解的必要條件**：
  傳統加性正規化（Additive Regularizer）目標為：
  $$\mathcal{L}_{\text{add}} = \sum_{e=1}^E \mathcal{L}_e(\mathcal{S}_e) + \lambda \sum_{e=1}^E |\mathcal{S}_e|$$
  由於 $|\mathcal{S}_e| = s = I$ 受到硬約束，加性正規化對「跨專家交集」完全無感，其最優解僅是各專家獨立貪婪擬合自身誤差。
  **Cohort Union Objective 的必要條件**：
  $$\mathcal{L}_{\text{union}} = \mathbb{E}_{x \sim \mathcal{D}} \left[ \sum_{e \in \mathcal{E}(x)} \alpha_e \mathcal{L}_e(\mathcal{S}_e) + \beta \cdot \text{Cost}_{\text{flash}} \left( \bigcup_{e \in \mathcal{E}(x)} \mathcal{S}_e \right) \right]$$
  式中 $\text{Cost}_{\text{flash}}$ 是以實體 16 KiB Page 與 4-Plane Wave 為梯度的階梯函數（階躍成本）。
  只有當「專家 $e_a$ 屈就使用專家 $e_b$ 的特徵所增加的重構損失 $\Delta \mathcal{L}_{e_a}$」，**嚴格小於**「減少載入一個實體 18-page GU Group 所節省的硬體 Flash 讀取懲罰 $\beta \cdot \Delta \text{Cost}$」時，Union Objective 才會誘導出與獨立加性擬合完全不同的非平凡拓撲解。

---

## 2. 數學可檢驗之可學性與不可學性 Constructive Proofs

### 2.1 [本段推導] 小維度 $H=2, I=2, M=4$ 嚴格數學構造

令隱藏層維度 $H=2$，單一專家寬度 $I=2$，共享原子池總數 $M=4$。
輸入向量 $x = [x_1, x_2]^\top$。
原子計算：$\phi_j(x) = \text{SiLU}(g_j^\top x) \cdot (u_j^\top x)$，其中 $j \in \{1, 2, 3, 4\}$。
專家輸出：$y_e = V^{(e)} \Phi_{\mathcal{S}_e}(x) \in \mathbb{R}^2$，其中 $V^{(e)} \in \mathbb{R}^{2 \times 2}$ 為 Down 投影。

### 2.2 [本段推導與數學證明] 尺度對稱性與不可移轉性分析

#### A. Up 向量與 Down 列的完全吸收性（Linear Scaling Invariance）
考慮任意純量 $\gamma \neq 0$：
$$\phi_j'(x) = \text{SiLU}(g_j^\top x) \cdot ((\gamma u_j)^\top x) = \gamma \cdot \left[ \text{SiLU}(g_j^\top x) \cdot (u_j^\top x) \right] = \gamma \cdot \phi_j(x)$$
在專家輸出端，其對應的 Down 權重列向量 $v_{e, j}$ 可同步縮放為 $v_{e, j}' = \frac{1}{\gamma} v_{e, j}$：
$$v_{e, j}' \cdot \phi_j'(x) = \left( \frac{1}{\gamma} v_{e, j} \right) \cdot (\gamma \cdot \phi_j(x)) = v_{e, j} \cdot \phi_j(x)$$
- **結論 1**：Up 向量的**任何正負尺度縮放與符號翻轉（$\gamma < 0$）**，皆可被專家的私有 Down 矩陣**無失真（Lossless）精確吸收**。因此在學習共享池時，所有原子的 Up 向量可施加約束 $\|u_j\|_2 = 1$ 而不損失任何函數表達空間。

#### B. Gate 向量不可移轉性定理（SiLU Non-Homogeneity）
對於 SiLU 函數 $\text{SiLU}(z) = \frac{z}{1 + e^{-z}}$，其具備非齊次性：
$$\forall c \neq 1, \quad \text{SiLU}(c \cdot z) \neq c \cdot \text{SiLU}(z)$$
**證明**：
設存在常數 $c > 0$ 與 $k(c)$ 使得對所有 $z \in \mathbb{R}$ 皆有 $\text{SiLU}(c z) = k(c) \text{SiLU}(z)$。
即：
$$\frac{c z}{1 + e^{-c z}} = k(c) \frac{z}{1 + e^{-z}} \implies c(1 + e^{-z}) = k(c)(1 + e^{-c z}) \quad (\forall z \neq 0)$$
對 $z \to +\infty$，極限為 $c = k(c)$。代入原式得：
$$1 + e^{-z} = 1 + e^{-c z} \implies e^{-z} = e^{-c z} \implies c = 1$$
矛盾。因此不存在此類線性等價。

### 2.3 [本段推導] 可學與不可學之 Constructive 實例

#### 1. 可學構造（Learnable Case）：方向一致僅 Up 響應相異
- **Teacher 專家 1**：$g^{(1)} = [1, 0]^\top, \ u^{(1)} = [0, 1]^\top, \ V^{(1)} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$
- **Teacher 專家 2**：$g^{(2)} = [1, 0]^\top, \ u^{(2)} = [0, -3]^\top, \ V^{(2)} = \begin{bmatrix} 2 & 0 \\ 0 & -1 \end{bmatrix}$
- **Student 共享池解**：
  配置共享原子 $1$：$g_{\text{pool}, 1} = [1, 0]^\top, \ u_{\text{pool}, 1} = [0, 1]^\top$。
  - 專家 1 的 Down 列 $v_{1, 1} = V^{(1)}_{:, 1} = [1, 0]^\top$。
  - 專家 2 的 Down 列吸收係數 $\gamma = -3$：$v_{2, 1} = (-3) \cdot V^{(2)}_{:, 1} = [-6, 0]^\top$。
- **數學結果**：專家 1 與 2 完全共用同一個原子 $\phi_1(x)$，在實數域與 F32 下其輸出重構誤差 $\epsilon \equiv 0$。

#### 2. 不可學構造（Unlearnable Case）：Gate 激活曲線跨域衝突
- **Teacher 專家 1（弱信號探測器）**：$g^{(1)} = [0.1, 0]^\top, \ u^{(1)} = [1, 0]^\top$
- **Teacher 專家 2（飽和硬閾值探測器）**：$g^{(2)} = [10.0, 0]^\top, \ u^{(2)} = [1, 0]^\top$
- **求證**：當共享池大小被壓縮，強迫兩專家共用單一 Gate 向量 $g_{\text{pool}} = [\theta, 0]^\top$。
- **評估點**：取測試輸入 $x_A = [1, 0]^\top$ 與 $x_B = [5, 0]^\top$：
  - Teacher 1 在兩點之比值：
    $$\frac{\phi^{(1)}(x_B)}{\phi^{(1)}(x_A)} = \frac{\text{SiLU}(0.5) \cdot 5}{\text{SiLU}(0.1) \cdot 1} = \frac{0.3112 \times 5}{0.0525 \times 1} \approx 29.64$$
  - Teacher 2 在兩點之比值：
    $$\frac{\phi^{(2)}(x_B)}{\phi^{(2)}(x_A)} = \frac{\text{SiLU}(50.0) \cdot 5}{\text{SiLU}(10.0) \cdot 1} = \frac{50.0 \times 5}{10.0 \times 1} = 25.00$$
  - 任何單一 Student 原子 $\phi_{\text{pool}}(x)$ 在這兩點的比值為：
    $$\frac{\phi_{\text{pool}}(x_B)}{\phi_{\text{pool}}(x_A)} = 5 \cdot \frac{\text{SiLU}(5\theta)}{\text{SiLU}(\theta)}$$
    此單調比值函數在實數線上嚴格受限於其自身曲率，無法在不同專家間同時滿足 $29.64$ 與 $25.00$ 的映射要求。
- **結論**：Down 矩陣的任何線性權重組合均無法彌合由 SiLU 非齊次性導致的曲率失真。若強迫兩者共用，必在有限樣本空間引發不可消除的下界誤差：
  $$\min_{g, u, V} \mathbb{E}_x [\|y_{\text{teacher}} - y_{\text{student}}\|^2] > \delta > 0$$

### 2.4 數值層級劃分：Exact Real vs. F32 vs. Quantized (Q4_K/Q8_K)
- **實數域（Exact Real）**：Up 縮放與 Down 係數完全對稱。
- **F32 域**：當 Down 吸收極大縮放（如 $\gamma = 100$）或極小縮放時，F32 尾數精度在累加時產生 1 ULP 捨入累積，且可能觸發次正則數（Subnormal numbers）。
- **量化域（Q4_K Down / Q8_K Activation）**：Down 吸收過大的 $\gamma$ 將大幅拉寬 Down 矩陣該 32-weight subgroup 的動態範圍，導致其 FP16 $d$ 與 $d_{\min}$ 劇增，進而造成該列其餘權重的量化階截斷誤差劇烈惡化。**可學性的邊界不僅取決於函數空間，更取決於量化動態範圍的抗失真能力**。

---

## 3. $M=4I, s=I, T=128$ 的 Active Work、Down 地板與 Goldilocks 邊界

### 3.1 [本段推導] 核心幾何與快閃記憶體實體頁對照帳

依給定事實與硬體參數：
- **OLM 規格**：$H=2048, I=1024, E=64, K=8$。
- **總池大小**：$M = 4I = 4096$ 原子。
- **Group 劃分**：$T = 128 \implies$ 全池 $G = 32$ 個 Groups。每個專家硬支援度 $s = I = 1024 \implies 8$ 個 Groups / 專家。
- **單一 Group 的實體儲存體積**：
  一個 Group 包含 128 個 paired G/U 神經元，每個神經元有 $H=2048$ 的 Gate 與 Up：
  - 權重總數：$2 \times 128 \times 2048 = 524,288$ weights。
  - **Q4_K 格式（每 256 weights 佔 144B）**：
    $$\text{Group 體積} = 524,288 \times \frac{144}{256} = 294,912\text{ Bytes} = 288\text{ KiB}$$
  - **實體 16 KiB 頁換算**：
    $$294,912\text{ B} / 16,384\text{ B/page} = \mathbf{18\text{ Pages / Group}}$$
  *(嚴格澄清：128 個神經元對應 18 個實體頁，絕非單一 16 KiB 頁！)*

### 3.2 [本段推導] GU 最大收益與 Down 剛性地板

#### A. 活躍 Group 數的極值分佈
在單一 Token 下，Router 選出 $K=8$ 個專家，每個專家需要 8 個 Groups。
- **最優重疊（Best-Case Union）**：8 個專家選取完全相同的 8 個 Groups。
  - $U_{\text{groups}} = 8$ Groups（共 $8 \times 18 = 144$ Pages，體積 $2.25\text{ MiB}$）。
  - GU 特徵僅需計算 1024 個神經元。
- **最差重疊（Worst-Case Union）**：8 個專家的群組均勻覆蓋全池。
  - $U_{\text{groups}} = \min(32, 8 \times 8) = 32$ Groups（全池飽和，共 $32 \times 18 = 576$ Pages，體積 $9.0\text{ MiB}$）。
  - GU 特徵需計算 4096 個神經元。

#### B. 權重讀取量精確計算（Weights Read Accounting）
設每個 Token 需處理的權重總位元組數為 $W_{\text{token}}$：
- **原始基準（Original Native Q4_K Baseline）**：
  - GU 權重讀取：$2 \times K \times I \times H = 2 \times 8 \times 1024 \times 2048 = 33,554,432$ weights $\rightarrow \mathbf{18.87\text{ MiB}}$（1152 Pages）。
  - Down 權重讀取：$K \times I \times H = 8 \times 1024 \times 2048 = 16,777,216$ weights $\rightarrow \mathbf{9.44\text{ MiB}}$（576 Pages）。
  - **原始總讀取量**：$18.87 + 9.44 = \mathbf{28.31\text{ MiB}}$（1728 Pages）。
- **共享 Atom Pool 候選架構**：
  - **Down 剛性地板（Unavoidable Down Floor）**：
    因每個專家必須維持私有 Down 矩陣以重構專異性輸出，Down 讀取量完全不變，**剛性鎖定在 9.44 MiB（576 Pages）**。
  - **總讀取量依 $U_{\text{groups}}$ 之動態區間**：
    $$\text{Total Read}(U) = (U_{\text{groups}} \times 18\text{ Pages} \times 16\text{ KiB}) + 9.44\text{ MiB}$$
    - 當 $U_{\text{groups}} = 8$（極限最佳）：$2.25\text{ MiB (GU)} + 9.44\text{ MiB (Down)} = \mathbf{11.69\text{ MiB}}$。**總頻寬節省上限為 58.7%**。
    - 當 $U_{\text{groups}} = 16$（中度共享）：$4.50\text{ MiB (GU)} + 9.44\text{ MiB (Down)} = \mathbf{13.94\text{ MiB}}$。**總頻寬節省為 50.8%**。
    - 當 $U_{\text{groups}} = 32$（全池打滿）：$9.00\text{ MiB (GU)} + 9.44\text{ MiB (Down)} = \mathbf{18.44\text{ MiB}}$。**總頻寬節省為 34.9%**。

### 3.3 [本段推導] 致命威脅：全池 Host SRAM/DRAM 常駐翻轉（Host Residency Inversion）

請檢視上述極其殘酷的實體數字：
- **整個 64 專家共用的完整 GU Atom Pool（$M=4096$）之總體積**：
  $$M \times 2 \times H \times \frac{144}{256} = 4096 \times 2 \times 2048 \times 0.5625\text{ B} = \mathbf{9.437\text{ MiB}}$$
- **架構命題之自我瓦解**：
  在現代端側推論裝置（如 Apple M 系列、高通 Snapdragon 或一般 x86 邊緣主機）中，**9.44 MiB 甚至小於處理器的最後一級快取（L3/LLC Cache，通常為 16～32 MiB），且完全可無痛常駐於 8GB Host DRAM 中**！
  - 若 Host 能夠常駐這 9.44 MiB 的 GU 池，Host 可以在讀取 NAND 前，直接在本地計算全部 4096 個神經元的特徵（耗時僅數十微秒），**根本不需要向 Flash 發出任何 GU 讀取請求**！
  - Flash 僅需單純負責串流讀取 9.44 MiB 的 Down 矩陣。
- **結論**：
  若該共享池無法證明其在 In-Flash PE 上具有「無法被 Host 常駐替代」的計算/頻寬屏障，則「In-Flash GU Group 讀取優化」將淪為偽需求。

### 3.4 [本段推導] Goldilocks 兩側極端反例

1. **容量過小反例（Under-capacity Collapse, $M \to I$）**：
   若設 $M = I = 1024$（僅 8 個 Groups）。此時所有 64 個專家被迫使用完全相同的 8 個 Groups。雖然 Flash 讀取達到了理論極限（每次固定讀 2.25 MiB GU），但 64 個專家的非線性特徵完全無差異，模型退化為單一 Dense FFN 加上 64 個線性變換頭。在多工/複雜 reasoning benchmark 上，表徵容量崩潰，PPL 必將大幅惡化（突破 +15% 失敗線）。
2. **容量過大與稀疏稀釋反例（Over-capacity Dilution, $M \ge 16I$）**：
   若設 $M = 16384$（128 個 Groups）。專家群組選擇高度發散。在隨機路由下，每次選中的 8 個專家其群組聯集 $U_{\text{groups}} \to 64$。GU 讀取量接近未共享狀態，但每個專家卻承擔了從 128 個候選中選擇 8 個的龐大定址/遮罩 metadata 與動態控制開銷，架構毫無淨收益。

---

## 4. 有限蒸餾架構、訓練公平性與 Kill-Test 設計

### 4.1 [本段推導] 蒸餾樣本覆蓋與稀有專家困境

- **原始 Router 凍結**：$\pi(x)$ 權重不變，輸入分佈 $x \sim \mathcal{D}_{\text{train}}$ 決定了專家啟動頻率。
- **全樣本 Teacher Query vs. Routed-Only**：
  - *全樣本 Query 方案*：不論 Router 是否選中，對所有 $x$ 強制計算全 64 個 Teacher 專家的輸出 $y_e(x)$ 作為蒸餾目標。
    - *缺陷*：背離了 MoE 的自然流形。某些專家專門處理小眾語義（如特定程式碼語法），其在自然分佈下啟動率極低。若強迫其在全體數據上擬合，共享池將被不相關的無效特徵佔據。
  - *Routed-Only 方案*：僅當 $e \in \text{Top-}K(x)$ 時才對專家 $e$ 計算蒸餾損失。
    - *缺陷（Rare Expert Underspecification）*：對於低頻啟動的稀有專家（長尾專家），其支援集 $\mathcal{S}_e$ 與私有 Down 矩陣獲得的梯度更新次數遠少於熱門專家。在共享池由熱門專家主導更新的壓力下，稀有專家的專屬表徵會被快速邊緣化，導致特定長尾任務品質暴跌。

### 4.2 [本段推導] 初始化（Initialization）vs. 從零學習（From-scratch）
1. **從零隨機初始化（Random Init）**：
   在 SiLU 雙重乘積的非凸地形下，從零聯合訓練 32 個 Group 的 Gate/Up 向量極易陷入局部鞍點或非線性死區，且無法確保初期與凍結 Router 產生的表徵空間對齊。
2. **基於 Teacher 聚類初始化（Teacher Clustering Initialization）**：
   - 提取原始 64 專家共 $64 \times 1024 = 65,536$ 個原始神經元。
   - 依據神經元激活相關性（Cosine affinity of Gate/Up parameters）執行受約束的 $K$-means 或譜聚類（Spectral Clustering），聚合成 $M=4096$ 個原子中心。
   - 將聚類中心指派給 32 個 Group，以原始各專家對原子的投影權重作為 Down 的初始值。
   - **此為唯一具有數值收斂希望的冷啟動路徑**。

### 4.3 [本段推導] 損失函數結構風險：Joint Mixture Loss vs. Per-Expert Loss

#### A. 聯合混合損失（Joint Mixture Loss）
$$\mathcal{L}_{\text{joint}} = \mathbb{E}_x \left\| \sum_{e \in \mathcal{E}(x)} \alpha_e y_e^{\text{student}}(x) - \sum_{e \in \mathcal{E}(x)} \alpha_e y_e^{\text{teacher}}(x) \right\|^2$$
- **致命風險（Cross-Expert Cancellation）**：
  梯度僅約束最終總和。專家 $A$ 可以嚴重高估某特徵，而專家 $B$ 低估該特徵，兩者在當前 Token 下相互抵消使得 $\mathcal{L}_{\text{joint}} \approx 0$。然而，當後續推論中專家 $A$ 與未參與抵消的專家 $C$ 組合時，該偏差將全面爆發，造成嚴重的泛化坍縮。

#### B. 單專家保真損失（Per-Expert Distillation Loss）
$$\mathcal{L}_{\text{per-expert}} = \mathbb{E}_x \left[ \sum_{e \in \mathcal{E}(x)} \alpha_e \left\| y_e^{\text{student}}(x) - y_e^{\text{teacher}}(x) \right\|^2 \right]$$
- **優勢**：強制約束每個專家在自身啟動區間內必須具備獨立的函數逼近能力，徹底消除跨專家掩蓋效應。

### 4.4 [本段推導] Group 遮罩的公平訓練（Fair Mask Learning）
為使每個專家硬性挑選 8 個 Groups，不能使用不可微的硬採樣，亦不可採用鬆弛的無界 $L_1$：
- 引入連續 Group 選擇參數 $P \in \mathbb{R}^{E \times G}$。
- 採用 **Top-8 Gumbel-Softmax 或平滑二值化（Straight-Through Estimator, STE）** 確保 Forward pass 嚴格為 8-hot 遮罩，Backward pass 提供連續梯度。
- 引入 **Cohort 非加性惩罰項**：
  $$\mathcal{R}_{\text{cohort}} = \mathbb{E}_{x} \left[ \text{ReLU} \left( \sum_{g=1}^G \max_{e \in \mathcal{E}(x)} (\text{mask}_{e, g}) - U_{\text{budget}} \right) \right]$$
  直接對活躍聯集超出目標群組數（如 $U_{\text{budget}} = 16$）施加二次懲罰，迫使不同專家在梯度反向傳播中主動向同一個 Group 靠攏。

### 4.5 最早必要 Kill-Test 規範（Proposal Only）

#### 測試標的與條件：
- **測試位置**：選定模型中間層（如 32 層模型中的 Layer 16，避開敏感的首尾兩層），凍結其餘 31 層。
- **測試配置**：$H=2048, I=1024, E=64, K=8$。將該層替換為 $M=4096, G=32, s=8$ 的共享 Atom Pool。
- **評估指標**：
  1. 該層局部輸出重建相對誤差（Relative MSE）：$\frac{\|Y_{\text{student}} - Y_{\text{teacher}}\|_F^2}{\|Y_{\text{teacher}}\|_F^2}$。
  2. 凍結其餘層條件下，在未見驗證集（Unseen Validation）上的端到端 PPL 擾動 $\Delta \text{PPL}$。
  3. 驗證集平均活躍群組數 $\bar{U}_{\text{groups}} = \mathbb{E}[|\bigcup_{e \in \mathcal{E}} \mathcal{S}_e|]$。

#### 核心判定線（Kill Criteria）：
1. 若要達到 $\Delta \text{PPL} < 2\%$，訓練收斂後的 $\bar{U}_{\text{groups}}$ 必須大於 **28 Groups**（即聯集重疊率低於 12.5%，幾乎全滿），則**判定架構死亡**（說明為了保證品質，群組根本無法實質共享）。
2. 若在強制作為 $\bar{U}_{\text{groups}} \le 16$（至少節省 50% GU）的約束下，單層替換引發的 $\Delta \text{PPL} \ge 5.0\%$，則**判定架構死亡**（說明非線性特徵池在該壓縮率下容量嚴重不足）。

#### 局限性澄清：
單層通過 Kill-Test 僅為必要條件，**絕不能外推斷言全模型 32 層同步替換後能維持品質**，亦不能直接冒充端到端硬體加速成功。

---

## 5. 總結：核心推導結論、Thought-Veto 邊界與下一 Block 關鍵問題

### 5.1 [已確認結論]
1. **數值對稱性分離**：Up 向量的尺度可被私有 Down 列完全無失真吸收；但 SiLU 的非齊次性使得 Gate 向量的尺度與方向具有絕對剛性，曲率衝突的特徵在數學上不可透過共享 Gate 消除。
2. **Down 剛性地板**：在維持專家私有 Down 的前提下，無論 GU 特徵如何極致共享，Down 權重的 9.44 MiB 讀取是不可壓縮的物理下界，全模型單 Token 頻寬節省的理論天花板嚴格受限於 58.7%。
3. **實體頁跨度修正**：一個包含 128 個神經元的 Group 在 Q4_K 下佔用 288 KiB，實體映射為 18 個 16 KiB 頁，並非單一頁面操作。

### 5.2 [Thought-Veto 觸發條件]
> **若在進一步系統評估中確認：因全池 $M=4096$ 的總體積僅有 9.44 MiB，Host 端以極低代價（8GB 記憶體內佔用 <0.12%）常駐全池並以 CPU/NPU 零 NAND 延遲先行計算 GU，其能效與延遲全面碾壓「In-Flash PE 執行 18-page 讀取與本地運算」，則本研究命題中「In-Flash GU Group 加速」之硬體必要性不復存在，應立即給予 Thought-Veto 結案。**

### 5.3 [Block 02 唯一核心未解問題]
> **「若假設 9.44 MiB 必須保留在 Flash 端（例如超低成本無 DRAM 控制器環境），這 32 個 Group（每組 18 Pages）應如何在 16 Chips $\times$ 4 Planes（共 64 個實體 Plane）之間進行靜態交錯映射（Interleaved Allocation），才能使任意隨機選出的 16 個活躍 Groups（共 288 Pages）在 Multi-Plane 同步讀取時，其實體 Plane 衝突率低於理論閾值？」**
