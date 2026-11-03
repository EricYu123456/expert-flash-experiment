# Round 3 候選思考 B Block 04：NAND 實體波次成本、最大權重匹配與聯合學習必要性形式化備忘錄

**記錄標識**：`round03-thinking/gemini_formal_b_block04_nand_waves_matching.md`（純文字直接交付，無工具呼叫、無檔案讀寫、無新候選計時轉移）  
**當前累積實質工作**：Block 01（112.73s）+ Block 02（151.12s）+ Block 03（133.27s）= 397.12s。距離 1800 秒門檻仍遠，本區段專注於實質數學與架構推導，不作門檻通過宣稱。

---

## 1. 實體組態核算、期望 Slot 閉式解與極限外推邊界

### 1.1 實體組態字節與 Page 幾何精確驗算
- **基礎參數**：Canonical W8 格式，隱藏維度 $H = 2048$，群組粒度 $T = 128$ 個配對神經元。
- **單一 Group 實體維度**：
  每個 Group 包含 128 個 Gate 與 128 個 Up 向量，共 $2 \times 128 = 256$ 條 Rows，每條 Row 為 $H = 2048\text{ Bytes}$。
  $$\text{Group 大小} = 256 \times 2048\text{ B} = 524,288\text{ Bytes} = \mathbf{512\text{ KiB}} = \mathbf{32\text{ 個 16 KiB 實體頁}}$$
- **16-Chip 均勻條帶化（Striping）**：
  將 32 個頁面均勻橫跨 16 顆 Flash Chip：
  $$\text{每 Chip 分配頁數} = \frac{32\text{ 頁}}{16\text{ Chips}} = \mathbf{2\text{ 頁/Chip}} \quad (32,768\text{ Bytes/Chip})$$
- **雙群組實體對（Physical Pair）封裝**：
  將 Group $A$ 配置於 Chip 的 Plane 0 與 Plane 1（各 1 頁），Group $B$ 配置於同 Block/WL 位址的 Plane 2 與 Plane 3（各 1 頁）。
  - 兩群組合計佔用：$2 \text{ Groups} \times 512\text{ KiB} = 1024\text{ KiB} = \mathbf{1\text{ MiB}}$。
  - 每個 Chip 恰好佔用 4 個 Plane 各 1 頁（$4 \times 16\text{ KiB} = 64\text{ KiB}$）。
  - **全域感測波次（Full Wave）**：發出一次 16-Chip $\times$ 4-Plane 同步 Multi-plane 讀取指令，可在單一感測週期 $t_R = 28\,\mu\text{s}$ 內讀出整對（1 MiB）資料。
  - **剛性約束**：即便當前 Token 僅需 Group $A$ 而不需要 Group $B$，因其處於相同 Block/WL 的物理對中，硬體仍必須發出該 Slot 的讀取，**支付完整的 $28\,\mu\text{s}$ 感測延遲**。動態跨 WL 重新打包（Bin-packing）在 Flash 物理層被嚴格禁止。

---

### 1.2 [本段推導] 隨機均勻選擇下的期望 Slot 閉式解
設 OLM 全池 $G = 32$ 個細群組，固定結合成 16 個實體對（Physical Slots，$p = 1, \dots, 16$）。
每個專家硬性挑選 $s/T = 8$ 個細群組。
考慮無真實路由結構的均勻獨立選擇作為零假設基準（Null Hypothesis Baseline）：
1. **單一專家避開特定實體對 $p$ 的機率**：
   特定實體對 $p$ 包含 2 個特定群組 $\{g_a, g_b\}$。專家要完全避開 $p$，必須自其餘 30 個群組中挑選全部 8 個群組：
   $$P(\text{expert } e \text{ avoids pair } p) = \frac{\binom{30}{8}}{\binom{32}{8}} = \frac{\frac{30!}{8! \cdot 22!}}{\frac{32!}{8! \cdot 24!}} = \frac{24 \times 23}{32 \times 31} = \frac{552}{992} = \frac{69}{124} \approx \mathbf{0.5564516}$$
2. **$K=8$ 個獨立專家全部避開該實體對的機率**：
   $$P(\text{slot } p \text{ is idle}) = \left( \frac{69}{124} \right)^8 \approx (0.5564516)^8 \approx \mathbf{0.009184}$$
3. **期望活躍 Slot 數**：
   $$\mathbb{E}[N_{\text{slot}}] = 16 \times \left( 1 - P(\text{slot } p \text{ is idle}) \right) = 16 \times (1 - 0.009184) \approx 16 \times 0.990816 = \mathbf{15.853\text{ Slots}}$$

#### 外推邊界批判：
- **可以外推之處**：證明了「若僅依賴未約束的獨立專家剪枝或隨機挑選群組，即使細群組聯集大小有縮減（如均勻時 $\mathbb{E}[U_{\text{groups}}] \approx 28.8 < 32$），**實體物理感測波次幾乎 100% 飽和（15.85 / 16）**，完全無法轉化為 Flash 感測延遲的縮短」。
- **不可外推之處**：此閉式解建立在專家路由相互獨立且均勻的虛構假設上。真實 Transformer 的 Router 具有高度語義聚集性（Co-activation Clustering）。若選中的專家高度協同，真實期望 Slot 會大幅低於 15.85。因此不能將 15.85 作為真實模型的性能預言。

---

### 1.3 [本段推導] 最小可行範例：同 Fine-Union 異 Slot vs. 異 Fine-Union 同 Slot

#### 範例環境：$G = 8$ 個細群組，固定組成 4 個實體對：
$$\text{Pair } P_1 = \{1, 2\}, \quad P_2 = \{3, 4\}, \quad P_3 = \{5, 6\}, \quad P_4 = \{7, 8\}$$

#### A. 相同 Fine-Union，不同 Slot 數之最小反例
- **情境 1（高度對齊）**：活躍專家聯集恰好需要群組 $\mathcal{U}_1 = \{1, 2, 3, 4\}$（大小為 4）。
  - 觸發的實體對：$P_1$（含 1, 2）與 $P_2$（含 3, 4）。
  - **活躍 Slot 數**：$\mathbf{2\text{ Slots}}$（耗時 $2 \times 28\,\mu\text{s} = 56\,\mu\text{s}$）。
- **情境 2（正交打散）**：活躍專家聯集需要群組 $\mathcal{U}_2 = \{1, 3, 5, 7\}$（大小同樣為 4）。
  - 觸發的實體對：$1 \in P_1, 3 \in P_2, 5 \in P_3, 7 \in P_4$。
  - **活躍 Slot 數**：$\mathbf{4\text{ Slots}}$（耗時 $4 \times 28\,\mu\text{s} = 112\,\mu\text{s}$）。
- *代數結論*：**相同數量的細群組計算（皆為 4 個 Groups），Flash 暴露延遲相差整整 2 倍！**

#### B. 相同 Slot 數，不同 Fine-Union 之最小反例
- **情境 3**：活躍專家聯集需要 $\mathcal{U}_3 = \{1, 3\}$（大小為 2）。
  - 觸發實體對：$1 \in P_1, 3 \in P_2 \implies \mathbf{2\text{ Slots}}$。
- 對照情境 1：$\mathcal{U}_1 = \{1, 2, 3, 4\}$ 同樣觸發 $\mathbf{2\text{ Slots}}$。
- *代數結論*：情境 1 與情境 3 的 Flash 感測波次完全一致（皆為 2 Waves），但情境 1 的 ALU 運算量與 F32 暫存狀態為情境 3 的整整 2 倍。
- **範疇嚴格分離**：Slot 成本嚴格決定 **Flash 感測延遲（$t_R$ 波次）**；Fine-Union 大小決定 **PE 本地 ALU 運算時間與 SRAM 活頁狀態**；兩者絕非線性等價，亦不可直接代換為整體 FFN 延遲。

---

## 2. 最優靜態配對與最大權重完全匹配（MWPM）的等價性

### 2.1 [本段推導與數學證明] 最佳實體配對等價於 MWPM

#### 命題：
給定一組**已經訓練完成且固定的專家支援集** $\{\mathcal{S}_e\}_{e=1}^E$。
定義群組需求二元指示變數：$A_g(R) = \mathbb{I}\{\exists e \in R \text{ s.t. } g \in \mathcal{S}_e\}$。
單一實體對 $p = (a, b)$ 被選中啟動的機率為：
$$P(\text{Pair } (a, b) \text{ is active}) = \mathbb{E}_R [A_a \cup A_b] = P(A_a) + P(A_b) - P(A_a \cap A_b) = p_a + p_b - p_{ab}$$
其中 $p_a = P(A_a)$ 為群組 $a$ 的單獨啟動率，$p_{ab} = P(A_a \cap A_b)$ 為群組 $a$ 與 $b$ 同時在同一個 Token Cohort 被需要的**協同共現機率**。

全池 32 個群組的一組完全配對 $\mathcal{M}$ 包含 16 個無交集的配對邊 $(a_k, b_k)$，滿足 $\bigcup_{k=1}^{16} \{a_k, b_k\} = \{1, \dots, 32\}$。
整體的期望活躍 Slot 成本為：
$$\mathbb{E}[N_{\text{slot}}(\mathcal{M})] = \sum_{(a, b) \in \mathcal{M}} (p_a + p_b - p_{ab}) = \sum_{g=1}^{32} p_g - \sum_{(a, b) \in \mathcal{M}} p_{ab}$$

#### 關鍵結構觀察：
在專家支援集固定時，每個群組自身的邊際啟動率 $p_g$ 完全確定，因此項 $\sum_{g=1}^{32} p_g$ 是一個**與排程佈局無關的剛性常數**！
因此：
$$\min_{\mathcal{M}} \mathbb{E}[N_{\text{slot}}(\mathcal{M})] \iff \max_{\mathcal{M}} \sum_{(a, b) \in \mathcal{M}} p_{ab}$$

#### 結論：
在已知支援集下尋找最佳靜態實體佈局，**在數學上嚴格等價於在頂點數為 32、邊權重為協同共現機率 $w(a, b) = p_{ab}$ 的完全圖上，求解最大權重完全匹配（Maximum Weight Perfect Matching, MWPM）**！

---

### 2.2 對候選「新意（Novelty）」之誠實衝擊評估

此等價性帶來極具破壞性的科學審查事實：
1. **後排程基線（Post-Layout Baseline）並非弱對照**：
   在 32 個頂點的圖上，MWPM 具備多項式時間精確演算法（Edmonds' Blossom 演算法，$O(V^3) \approx 32^3 = 32,768$ 次操作，在常規 CPU 上僅需數毫秒）。
   這意味著：**任何「先做純品質訓練，訓練完後離線執行 MWPM 匹配」的管線，已經能夠取得該支援集下 100% 數學全域最優的物理配對！**
2. **兩種 Pairwise 概念的嚴格區分**：
   - **概念 1（群組實體對 MWPM）**：在給定群組啟動事件下，尋找 $a$ 與 $b$ 的最優實體打包。這是已解的圖論最優問題。
   - **概念 2（專家支援集學習中的 Pairwise 代理損失）**：在訓練階段，用 $\sum P(e_i, e_j) |\mathcal{S}_i \cap \mathcal{S}_j|$ 試圖誘導群組重疊。
3. **對 Joint Training 必要性的嚴格挑戰**：
   若「品質優先訓練 + 精確 MWPM 佈局」已經可以在不損害任何模型 PPL 的前提下達到幾何最優，那麼宣稱「必須在反向傳播中把 NAND Slot 成本加入 Loss 進行 Joint 訓練」，**其真實增量空間被大幅壓縮**。若無法證明 Joint 訓練能找到常規 MWPM 根本觸及不到的拓撲，則 Joint Objective 的必要性將被徹底質疑。

---

## 3. Joint Objective 何時與「品質訓練 + Best Matching」具備不同 Pareto 前沿

### 3.1 存在不同 Pareto 前沿的嚴格數學條件

設模型品質損失為 $\mathcal{L}_{\text{qual}}(\mathcal{S}, \Theta)$，硬體 Slot 成本為 $C_{\text{slot}}(\mathcal{S}, \mathcal{M}) = \mathbb{E}[N_{\text{slot}}]$。
- **情境 A（單一最優孤島，No Pareto Gain）**：
  若 $\mathcal{L}_{\text{qual}}$ 存在強烈的唯一局部極小值（Unique Sharp Basin），使得滿足 $\mathcal{L}_{\text{qual}} \le \epsilon$ 的合法支援集 $\mathcal{S}$ 本質上只有一種拓撲（其餘候選皆產生顯著的品質惡化）。
  此時，任何偏離該支援集以試圖湊合更高 $p_{ab}$ 的嘗試，皆會直接突破 PPL 預算。
  在此條件下，**Joint Objective 絕對無法勝過「獨立訓練 + MWPM」**。
- **情境 B（簡併流形與退化等價面，True Pareto Gain）**：
  在過參數化神經網路中，特徵表徵通常存在龐大的簡併空間（Degenerate Quality Equivalence Class）。
  即存在多個截然不同的支援集矩陣 $\mathcal{S}^{(1)}, \mathcal{S}^{(2)}$，滿足：
  $$\mathcal{L}_{\text{qual}}(\mathcal{S}^{(1)}) \approx \mathcal{L}_{\text{qual}}(\mathcal{S}^{(2)}) \le \epsilon$$
  然而：
  - $\mathcal{S}^{(1)}$ 的專家交集碎片化，其群組共現邊權重分佈均勻平坦，導致其最優匹配 $\max_{\mathcal{M}} \sum_{(a, b) \in \mathcal{M}} p_{ab}^{(1)}$ 數值很低（活躍 Slots 高達 14）。
  - $\mathcal{S}^{(2)}$ 在訓練時被 Joint Slot 梯度誘導，將高頻共現專家的群組集中分配在少數群組對上，使得邊權重呈現強烈的「團簇化（Clustering）」，其最優匹配 $\max_{\mathcal{M}} \sum_{(a, b) \in \mathcal{M}} p_{ab}^{(2)}$ 數值極高（活躍 Slots 降至 8）。
- **結論**：
  **Joint Training 的唯一真實價值，在於利用梯度將解空間導向「品質等價但共現圖譜更易被完美匹配」的特徵子空間。**

### 3.2 群組置換對稱性（Permutation Symmetry）辨析
- **置換操作本質**：
  對 32 個群組進行索引重排 $\pi(g)$，同時重排 Down 矩陣對應的 Column 區塊，**完全不改變神經網路的前向傳遞函數**（Functional Identity）。
- MWPM 本質上就是在這個龐大的置換對稱群（大小為 $32! / (16! \cdot 2^{16}) \approx 6.19 \times 10^{26}$ 種劃分）中進行組合尋優。
- 因此，Joint 訓練如果只是試圖尋找一個好的排列，那是無效的（MWPM 毫秒級即可完成）；**Joint 訓練唯有改變了專家對群組的依賴模式（即改變了高階超圖超邊結構），才算發揮了演算法層面的實質作用**。

---

## 4. 物理命令、PE、ECC 與暫存之完整 Paid 契約

在此條件構造下，硬體各階段必須支付明確的資源與延遲帳，拒絕無代價假設：

| 階段 / 單元 | 實體行爲與狀態存留 | 資源代價與儲存屬性 | 延遲與同步屏障（Barrier） |
| :--- | :--- | :--- | :--- |
| **Input $x$ (A8)** | 2048 Bytes 廣播輸入在控制器 SREG 中保持常駐 | 佔用控制器 2 KiB 專屬暫存，全 Wave 週期不可覆寫 | 零感測等待；若為新 Token 則覆寫 |
| **Raw Page Buffer** | 4-Plane 感測將資料鎖存至 NAND 晶粒內部 Page Buffer | 16 Chips $\times$ 64 KiB = 1 MiB 內部陣列暫存 | 剛性 $t_R = 28\,\mu\text{s}$，期間外部無法讀取 |
| **ECC 傳輸與校正** | 自 Page Buffer 串流經 ECC 引擎解碼 | 需本地雙緩衝 FIFO（$2 \times 2\text{ KiB}$ Codeword） | 6.4 GB/s 介面傳輸，排隊流入 PE |
| **GU GEMV 運算** | 讀取校正後權重，與輸入 $x$ 執行 8-bit MAC | PE 陣列乘加，維護 Gate 與 Up 的 I32 局部累加器 | 運算時間隱蔽於下一次 Flash 感測管線中 |
| **中間特徵暫存** | 產出活躍群組之 $\phi_j(x) = \text{SiLU}(g_j x) \cdot (u_j x)$ | 依 B03 證明，必須以 **F32 格式** 暫存活躍聯集特徵 | 佔用本地 SRAM：$U_{\text{atoms}} \times 4\text{ B}$（最大 16 KiB） |
| **Down 執行屏障** | Down 投影必須等待該專家全部 8 個群組特徵齊全 | 需全域計算專家級 $a_{\max}^{(e)}$ 並量化為 A8 | **剛性屏障**：下游 Down 在該專家最後一組 GU 就位前不可發射 |

#### 關鍵 ABI 與硬體修改量（Hardware Delta）：
1. **Down 列重排契約**：
   若 MWPM 實體配對重排了 32 個群組的物理順序，為使 Down 投影能以連續串流執行，**私有 Down 矩陣的對應 Column 區塊必須在離線部署時預先完成排列置換**。此時模型權重為新 ABI，不可宣稱與原始來源權重在記憶體佈局到位元一致。
2. **最小硬體改動量**：
   - *韌體（Firmware）層*：擴充 Command Descriptor，支援「靜態 Slot 遮罩查表」發射指令。
   - *硬體（Hardware）層*：無須改變 NAND 陣列或 ECC 結構；但控制器端必須新增 **16 KiB F32 特徵活頁 SRAM** 以及維護群組就位狀態的記分板（Scoreboard）。

---

## 5. 基準 Loader 對等比較與端到端服務時間方程

### 5.1 對等基準設定（Fair Matched Controls）
- **對照組 1（Ordinary Loader + Best Layout）**：
  若將完全相同的 MWPM 配對權重佈局交給常規主機讀取，Flash 陣列感測的實體 Wave 數與 In-Flash 架構**完全一致**。
  - In-Flash（PIM）唯一省下的是：**權重自 Flash 控制器跨 PCIe/CXL 匯流排搬運至 Host 的傳輸時間，以及 Host CPU 執行 GEMV 的運算能耗**。
- **對照組 2（Host-Cached GU）**：
  全模型 GU 池（OLM 僅 144 MiB Q4 / 256 MiB W8）常駐於 Host 8 GB RAM 中：
  - Flash 感測波次直接降為 **0 Waves**（$N_{\text{slot}} \times t_R = 0$）。
  - 此時本研究的「NAND Slot 優化」在 GU 側獲得的節省為零。

---

### 5.2 [本段推導] 符號化端到端服務時間方程（Service Time Equation）

定義單一 Token 推論中，FFN 層的端到端服務時間 $T_{\text{service}}$：

#### A. In-Flash PE 架構：
$$T_{\text{in-flash}} = \underbrace{\sum_{s=1}^{N_{\text{slot}}} \max\left( t_R, \frac{\text{Bytes}_{\text{pair}}}{\text{BW}_{\text{internal}}} \right)}_{\text{GU 串流感測與計算}} + \underbrace{T_{\text{barrier}}}_{\text{特徵收集與量化}} + \underbrace{\sum_{k=1}^{K} \left( t_R^{\text{Down}} + \frac{s \cdot H}{\text{BW}_{\text{flash}}} \right)}_{\text{Down 私有讀取與計算}}$$
- 由於 1 MiB 實體對內部傳輸極快，感測延遲 $t_R = 28\,\mu\text{s}$ 主導第一項：
  $$T_{\text{in-flash}} \approx N_{\text{slot}} \cdot t_R + T_{\text{barrier}} + T_{\text{Down}}$$

#### B. Host-All 架構（權重自 Flash 串流至 Host 運算）：
$$T_{\text{host-all}} = \sum_{s=1}^{N_{\text{slot}}} \left( t_R + \frac{\text{Bytes}_{\text{pair}}}{\text{BW}_{\text{PCIe}}} \right) + T_{\text{CPU-GEMV}} + T_{\text{Down-transfer}}$$
- **In-Flash 相對於 Host-All 的淨優勢**：
  $$\Delta T = N_{\text{slot}} \cdot \frac{\text{Bytes}_{\text{pair}}}{\text{BW}_{\text{PCIe}}} + T_{\text{CPU-GEMV}}$$
  優勢純粹來自於省去了匯流排搬運瓶頸，而非省去了 Flash 感測時間。

#### C. Host-Cached-GU + Flash-Down 混合架構：
$$T_{\text{hybrid}} = T_{\text{Host-GU-Compute}} + T_{\text{barrier}} + T_{\text{Down-from-Flash}}$$
- **勝負判別條件**：
  $$T_{\text{in-flash}} < T_{\text{hybrid}} \iff N_{\text{slot}} \cdot t_R < T_{\text{Host-GU-Compute}}$$
  - 若 Host CPU/NPU 運算能力極強，能在 $100\,\mu\text{s}$ 內算完全池 GU，而 In-Flash 若需 $N_{\text{slot}} = 8$ 波次（$8 \times 28 = 224\,\mu\text{s}$），**則 In-Flash 架構在延遲上直接落敗**。
  - In-Flash 要想勝出，必須滿足 $N_{\text{slot}} \cdot t_R$ 極小化，或 Host 端運算受限且匯流排極度擁塞。

---

## 6. 歸因對照組設計與訓練尺度鬼影（Ghost Scale）消除

### 6.1 [本段推導] 軟遮罩尺度鬼影（Ghost Scale）之數學本質與破除

#### A. 尺度鬼影問題：
在連續鬆弛訓練中，專家輸出為 $y_e = \sum_g m_{e, g} V_{e, g} \phi_g(x)$。
若對遮罩施加 $L_1$ 或期望 Slot 懲罰 $\lambda \sum m_{e, g}$：
- 優化器會迅速發現一個欺騙性漏洞：
  令 $m_{e, g} \to \epsilon \cdot m_{e, g}$（$\epsilon \to 0$），同時令 Down 權重 $V_{e, g} \to \frac{1}{\epsilon} V_{e, g}$。
- 此時乘積 $m_{e, g} V_{e, g}$ 保持不變，前向品質損失完全不變，但**代價項 $\lambda \sum m_{e, g}$ 被人為縮小了 $\epsilon$ 倍**！
- 這導致遮罩數值看似接近零，但一旦在部署時強行二值化（Hard Thresholding），Down 矩陣數值爆炸，模型直接發散。

#### B. 單一有限封閉 Recipe 破除方案（無須巨額 Grid Search）：
1. **嚴格前向二值化（Straight-Through Estimator, STE）**：
   Forward pass 嚴格執行硬二值化：$m_{e, g} = \mathbb{I}\{\text{Rank}(s_{e, g}) \le 8\}$；
   Backward pass 僅在接近閾值的窄區間傳遞連續梯度：$\frac{\partial m}{\partial s} = \text{HardTanh}'(s)$。
2. **Down 權重球面約束（Spherical Norm Constraint）**：
   在每個 Optimizer Step 後，對私有 Down 矩陣的每一列強制執行範數歸一化投影：
   $$\|V_{e, g, :}\|_2 \le C_0$$
   徹底切斷 $V \to \infty$ 的病態代償路徑。

---

### 6.2 最少兩個關鍵歸因對照組（Value Attribution Controls）

為在後續科學評估中完全隔離「非線性表徵共享原有紅利」與「NAND Wave 目標函數新紅利」，必須嚴格鎖定以下兩個控制組：

```
[原始 Native 基準] ─── (性能與品質絕對參照)
        │
        ├───────────────────────────────────────────────────────┐
        ▼ (Control 1: 隔離表徵本身紅利)                         ▼ (Treatment: 驗證聯合目標增量)
[純品質驅動池化訓練]                                    [品質 + NAND Slot 聯合目標訓練]
        │                                                       │
        ▼ (離線精確 MWPM 配對)                                  ▼ (離線精確 MWPM 配對)
[Control 1 最優實體波次 N_slot^(1)]                     [Treatment 最優實體波次 N_slot^(2)]
```

1. **Control 1（純品質池化 + 離線精確 MWPM 佈局）**：
   - 相同隨機種子、相同 30k 訓練 Token、相同 Canonical W8 精度。
   - 訓練時**完全不引入任何 NAND Slot 懲罰項**（僅以品質與獨立支援集擬合）。
   - 訓練結束後，統計群組共現機率矩陣 $P(A_a \cap A_b)$，呼叫標準 Blossom 演算法求解 MWPM 最優實體對佈局，產出波次數 $N_{\text{slot}}^{(1)}$ 與品質 $\text{NLL}^{(1)}$。
2. **Treatment（NAND Wave Joint 目標訓練 + 離線精確 MWPM 佈局）**：
   - 相同超參數下，訓練時加入 Cohort Slot 懲罰項。
   - 訓練結束後，以相同 MWPM 演算法求解佈局，產出波次數 $N_{\text{slot}}^{(2)}$ 與品質 $\text{NLL}^{(2)}$。

#### 科學判決準則：
- **若 $N_{\text{slot}}^{(2)} \approx N_{\text{slot}}^{(1)}$ 且品質無顯著差異**：
  **正式證偽 Joint NAND Wave Objective 的必要性**！表明純品質訓練後的表徵分佈透過 MWPM 即可自然獲得等價的物理波次優化，額外的聯合硬體損失純屬多餘複雜度。
- **若 $N_{\text{slot}}^{(2)} \ll N_{\text{slot}}^{(1)}$ 且 $\text{NLL}^{(2)} \le \text{NLL}^{(1)} + \delta$**：
  **Joint Objective 之科學存在價值獲得確立**，證明其成功利用了簡併流形引導出具有更強共現團簇特性的實體物理佈局。

---

## 7. 總結與 Block 05 唯一聚焦問題

### 7.1 本區段已確認與澄清結論
1. **實體封裝剛性**：1 Group = 32 頁（512 KiB），每 Chip 2 頁；雙 Group 固定組成 1 MiB 實體對，單一感測固定為 $28\,\mu\text{s}$。均勻隨機選擇下實體對啟用率高達 $15.85 / 16$（$99.1\%$），證實粗放剪枝無法降低感測波次。
2. **MWPM 等價定理**：證明了固定支援集下的最優配對在數學上等價於最大權重完全匹配，後排程基線具備多項式全域最優解，強烈挑戰了 Joint 訓練的必要性。
3. **尺度鬼影可解性**：推導出 STE 離散前向搭配 Down 矩陣球面範數約束，可在單一封閉 Recipe 下徹底解決 $m \to 0, D \to \infty$ 的退化問題。

### 7.2 Block 05 唯一核心未解問題
> **「在真實 TRAIN 30k tokens 的路由激活日誌下，利用 Control 1（純品質特徵池）計算出的真實共現權重圖 $p_{ab}$，其經由 MWPM 匹配後的理論波次下界 $N_{\text{slot}}$ 到底是多少？它與理論下界 $\lceil \mathbb{E}[U] / 2 \rceil$ 之間是否存在顯著落差，足以為 Joint Objective 留下不可替代的改進空間？」**
