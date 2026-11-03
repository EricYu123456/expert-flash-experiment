# Round 3 候選思考 B Block 06：離線雙群組全副本對照、不完整副本圖論下界與候選價值重構備忘錄

**記錄標識**：`round03-thinking/gemini_formal_b_block06_replica_graph_falsification.md`（純文字交付，無工具呼叫、無程式碼/資料讀取、無偽造計時）  
**當前累積實質工作**：Block 01~05 合計 502.30s + 本區段實質推導。未達 1800 秒門檻，本區段直面黑臉提出的「付費離線全配對副本（Paid Offline Pairwise Replication）」強對照，全面重審候選 B 的真實必要性與科學邊界。

---

## 1. 完整配對副本構造（All-Pair Replica Bank）容量與波次分析

### 1.1 [本段推導] 幾何容量閉式公式與兩模型精確驗算

設 Canonical W8 下，隱藏維度為 $H$，群組粒度為 $T = 128$。
- **單一 Group 實體體積**：
  $$B_{\text{group}} = 2 \times T \times H \times 1\text{ B} = 2 \times 128 \times 2048 = 524,288\text{ Bytes} = \mathbf{512\text{ KiB}}$$
- **全池 Group 總數**：$G = 4I / T$。
  - 對 OLM（$I=1024$）：$G = 4096 / 128 = \mathbf{32}$。
  - 對 DeepSeek-V2-Lite（$I=1408$）：$G = 5632 / 128 = \mathbf{44}$。
- **所有無序群組對（Unordered Pairs）組合數**：
  $$N_{\text{pairs}} = \binom{G}{2} = \frac{G(G - 1)}{2}$$
- **單層全配對副本庫總體積（Pairwise Bank Volume）**：
  每個實體對包含 2 個完整的 Group，固定預先燒錄於同一 Block/WL 位址之 4-Plane $\times$ 16-Chip（1 MiB 全域 Wave）：
  $$\text{Vol}_{\text{GU-bank}} = \binom{G}{2} \times (2 \cdot B_{\text{group}}) = \frac{G(G - 1)}{2} \times 2 B_{\text{group}} = G(G - 1) B_{\text{group}} = (G - 1) \times \text{Canonical GU}$$
  - *代數本質*：全域 $G$ 個群組中，**每個群組恰好在副本庫中重複出現 $G - 1$ 次**。全池副本放大係數（Replication Factor）精確為 **$G - 1$**。

#### A. OLM 全層核算（16 層，$G=32$）
- 單層 Canonical GU 基準：$32 \times 512\text{ KiB} = \mathbf{16\text{ MiB}}$。
- 單層全配對副本 GU 體積：$\binom{32}{2} \times 1\text{ MiB} = 496 \times 1\text{ MiB} = \mathbf{496\text{ MiB}}$（放大 $31\times$）。
- 單層私有 Down 矩陣（$E=64, I=1024, H=2048$）：$64 \times 1024 \times 2048 \times 1\text{ B} = \mathbf{128\text{ MiB}}$。
- 單層權重總量：$496\text{ MiB} + 128\text{ MiB} = \mathbf{624\text{ MiB}}$。
- **全模型 16 層總儲存需求**：
  $$16 \times 624\text{ MiB} = 9984\text{ MiB} = \mathbf{9.75\text{ GiB}} \quad (\text{十進制 } \approx 10.47\text{ GB})$$

#### B. DeepSeek-V2-Lite 全層核算（26 個 Routed 層，$G=44$）
- 單層 Canonical GU 基準：$44 \times 512\text{ KiB} = \mathbf{22\text{ MiB}}$。
- 單層全配對副本 GU 體積：$\binom{44}{2} \times 1\text{ MiB} = 946 \times 1\text{ MiB} = \mathbf{946\text{ MiB}}$（放大 $43\times$）。
- 單層私有 Down 矩陣（$E=64, I=1408, H=2048$）：$64 \times 1408 \times 2048 \times 1\text{ B} = \mathbf{176\text{ MiB}}$。
- 單層權重總量：$946\text{ MiB} + 176\text{ MiB} = \mathbf{1122\text{ MiB}}$。
- **全模型 26 層總儲存需求**：
  $$26 \times 1122\text{ MiB} = 29,172\text{ MiB} = \mathbf{28.49\text{ GiB}} \quad (\text{十進制 } \approx 30.59\text{ GB})$$

#### C. NAND 容量對比與硬體/韌體契約邊界
- 在 1 TB TLC Flash 中，若僅使用高可靠度/低延遲的 LSB Page，物理可用空間約為 $333.3\text{ GB} = \mathbf{310.4\text{ GiB}}$。
  OLM 的 9.75 GiB（佔比 3.1%）與 DeepSeek-V2-Lite 的 28.49 GiB（佔比 9.2%）**完全處於 Flash 物理容量承受範圍之內**。
- **波次最優可達性（Wave Minimality）**：
  由於任意群組對 $(a, b)$ 皆已在離線預先佈局完成，當活躍群組聯集為 $U$ 時，控制器可在已存在的副本中直接挑選 $\lfloor |U| / 2 \rfloor$ 個獨立邊；若 $|U|$ 為奇數，最後一個群組隨意挑選包含該群組的任意一個實體對（另一半讀出後直接丟棄）。
  **全域感測波次確定達到物理極限**：
  $$\text{Waves}_{\text{min}} = \left\lceil \frac{|U|}{2} \right\rceil$$
- **命令與控制分工契約**：
  - *Host 韌體*：僅負責依據 Token 路由產出的活躍集合 $U$，進行簡單的查表匹對，產出 $\lceil |U|/2 \rceil$ 個離散 Block/WL 實體位址 Descriptor。
  - *In-Flash 控制器*：無需任何動態 Cross-WL 拼裝邏輯，僅需依序執行既有標準的 Multi-Plane Read 指令流。
- **反例（Shape 溢出反例）**：
  若模型擴展至超大維度（如 $H=7168, I=2048, T=128$），單一 Group 為 $2 \times 128 \times 7168 = \mathbf{1.75\text{ MiB}}$。一個實體對體積高達 $3.5\text{ MiB}$，遠超 16-Chip 4-Plane 單次感測的 1 MiB 幾何波次！此時一個實體對本身就需要跨多次 Wave 感測，上述 $1\text{ Pair} = 1\text{ Wave}$ 的基本假設直接瓦解。

---

## 2. 容量受限時的不完整副本圖（Incomplete Replica Graph）

### 2.1 [本段推導與數學證明] 任意圖結構下的最小感測波次定理

#### 形式化模型：
- 定義圖 $G = (V, E)$，頂點集 $V = \{1, \dots, G\}$ 代表細群組；邊集 $E \subset \binom{V}{2}$ 代表預先燒錄的雙群組實體對副本（每個邊對應 1 MiB 實體 Wave）。
- 設系統保留單份獨立的 Canonical Fallback 副本（每個群組單獨存放，讀取 1 個孤立群組消耗 1 次 Wave）。
- 對於任意輸入 Token，活躍群組需求為頂點子集 $U \subseteq V$。
- 令 $G[U] = (U, E[U])$ 為由 $U$ 誘導的子圖，其中 $E[U] = \{(u, v) \in E \mid u, v \in U\}$。
- 令 $\nu(G[U])$ 為子圖 $G[U]$ 的**最大匹配數（Maximum Matching Number）**。

#### 定理：
在包含 Fallback 單副本的任意副本圖 $G$ 下，服務活躍集合 $U$ 的最小實體感測波次為：
$$\text{Cost}(U) = |U| - \nu(G[U])$$

#### 證明：
設 $M^* \subseteq E[U]$ 為子圖 $G[U]$ 的一組最大匹配，其包含的邊數為 $|M^*| = \nu(G[U])$。
1. 由於 $M^*$ 是匹配，這 $\nu(G[U])$ 條邊互不相交，恰好覆蓋了 $U$ 中 $2 \nu(G[U])$ 個頂點。
2. 每一條匹配邊代表一個實體對，發射 1 次 Wave 感測即可同時獲取其連接的 2 個群組。因此處理這 $2 \nu(G[U])$ 個頂點共消耗 $\nu(G[U])$ 次 Waves。
3. 集合 $U$ 中剩餘的頂點數量為 $|U| - 2 \nu(G[U])$。根據最大匹配的定義，這些頂點之間在 $E[U]$ 中不存在任何未被覆蓋的邊，否則可加入 $M^*$ 擴充匹配，與 $M^*$ 的最大性矛盾。
4. 因此，剩餘的 $|U| - 2 \nu(G[U])$ 個頂點必須全部透過 Fallback 單副本獨立讀取，每個頂點消耗 1 次 Wave。
5. 總感測波次為：
   $$\text{Total Waves} = \nu(G[U]) + \left( |U| - 2 \nu(G[U]) \right) = |U| - \nu(G[U])$$
由於不可能在 $U$ 中找出覆蓋超過 $2 \nu(G[U])$ 個頂點的邊不相交集合，該波次數為嚴格下界。 $\blacksquare$

#### 極值邊界驗算：
- *完全圖（Complete Graph，$E = \binom{V}{2}$）*：
  $\nu(G[U]) = \lfloor |U|/2 \rfloor \implies \text{Cost}(U) = |U| - \lfloor |U|/2 \rfloor = \lceil |U|/2 \rceil$（與 Section 1 完全吻合）。
- *空圖（Empty Graph，$E = \emptyset$）*：
  $\nu(G[U]) = 0 \implies \text{Cost}(U) = |U|$（每個群組完全獨立感測）。

---

### 2.2 [本段推導] 成本函數次模性（Submodularity）破缺反例

成本函數 $f(U) = |U| - \nu(G[U])$ 是否為次模函數（Submodular）？
若 $f$ 具備次模性，則邊際成本需滿足遞減性質：對 $A \subseteq B$ 及 $x \notin B$，$f(A \cup \{x\}) - f(A) \ge f(B \cup \{x\}) - f(B)$。

#### 嚴格反例構造：
設頂點集 $V = \{1, 2, 3, 4\}$。邊集僅包含一條路徑：$E = \{(1, 2), (2, 3), (3, 4)\}$。
取子集：
$$A = \{1, 3\}, \quad B = \{1, 2, 3\} \quad (A \subset B)$$
考慮新加入元素 $x = 4$：
1. **在子集 $A$ 上**：
   - $A = \{1, 3\}$，內部無邊 $\implies \nu(G[A]) = 0$。
     $$f(A) = 2 - 0 = \mathbf{2}$$
   - $A \cup \{4\} = \{1, 3, 4\}$，內部邊僅有 $(3, 4) \implies \nu(G[A \cup \{4\}]) = 1$。
     $$f(A \cup \{4\}) = 3 - 1 = \mathbf{2}$$
   - **邊際增量**：$\Delta_f(A, 4) = f(A \cup \{4\}) - f(A) = 2 - 2 = \mathbf{0}$。
2. **在子集 $B$ 上**：
   - $B = \{1, 2, 3\}$，內部邊有 $(1, 2), (2, 3) \implies$ 最大匹配僅能選 1 條邊，$\nu(G[B]) = 1$。
     $$f(B) = 3 - 1 = \mathbf{2}$$
   - $B \cup \{4\} = \{1, 2, 3, 4\}$，內部包含邊 $(1, 2)$ 與 $(3, 4)$，兩者不相交 $\implies \nu(G[B \cup \{4\}]) = 2$。
     $$f(B \cup \{4\}) = 4 - 2 = \mathbf{2}$$
   - **邊際增量**：$\Delta_f(B, 4) = f(B \cup \{4\}) - f(B) = 2 - 2 = \mathbf{0}$。
*(在此點相等。)*

#### 次模性破缺的確鑿反例：
設 $E = \{(2, 3), (3, 4)\}$（中心節點為 3）。
取 $A = \{2\}$，$B = \{1, 2\}$。加入節點 $x = 3$：
- $f(A) = 1 - 0 = 1$。$A \cup \{3\} = \{2, 3\} \implies$ 有邊 $(2, 3)$，$\nu = 1 \implies f(A \cup \{3\}) = 2 - 1 = 1$。
  邊際增量：$\Delta_f(A, 3) = 1 - 1 = \mathbf{0}$。
- 設另一個配置，令某匹配加入使匹配數不增反使節點純增：
  取邊集 $E = \{(1, 2)\}$。
  取 $A = \{3\}$，$B = \{1, 3\}$。加入 $x = 2$：
  - $A \cup \{2\} = \{2, 3\} \implies$ 無邊，$\nu = 0 \implies f(A \cup \{2\}) - f(A) = (2 - 0) - (1 - 0) = \mathbf{1}$。
  - $B \cup \{2\} = \{1, 2, 3\} \implies$ 有邊 $(1, 2)$，$\nu = 1 \implies f(B \cup \{2\}) - f(B) = (3 - 1) - (2 - 0) = 2 - 2 = \mathbf{0}$。
- **結論**：匹配數 $\nu(G[U])$ 在圖論中本質上是**整數值不可微的非凸階躍函數**，成本函數 $f(U)$ 既非嚴格次模亦非超模。這意味著**任何試圖套用次模貪婪最優化（Submodular Greedy）保證近似比的理論在不完整副本圖上皆無法直接成立**。

---

## 3. 候選 B 哪些核心 Claim 仍然存活？

直面付費全配對副本強對照，對候選 B 的原創邊界進行冷酷解剖：

```
[原始候選 B 主張]
  1. 共享非線性 GU 特徵池 (M=4I) 消除跨專家重複計算 ──────> [存活] (相對於 3EHI 依然具備算術與讀取壓縮)
  2. 依賴特定 NAND 實體 Slot 聯合訓練來降低感測 Wave ────> [大幅削弱 / 瀕臨證偽]
                                                                     │
                                  ┌──────────────────────────────────┴──────────────────────────────────┐
                                  ▼                                                                     ▼
             [若採用付費全配對副本 (9.75 GB)]                      [若採用不完整圖或純品質訓練 + 離線 MWPM]
             物理 Wave 嚴格退化為 ceil(|U|/2)                       離線 Blossom 演算法已達全域最優，
             訓練只需壓低 |U|，無需感知實體 Pair                   聯合 Slot Objective 增量價值極其微弱
```

1. **存活的主張**：
   - **非線性神經元去重（Deduplication）**：將 $E=64$ 專家的 $64I$ 神經元壓縮至 $4I$ 共享池，每個 Token 的活躍特徵計算與讀取從 $K \times I = 8I$ 降至 $U \le 4I$，這項數學層面的算術與頻寬節省完全存活。
   - **非加性（Non-additive）聯集目標**：在訓練中以最小化活躍聯集 $\mathbb{E}[|U|]$ 為目標，相對於傳統獨立 $L_0$ 剪枝，依然能誘導出更密集的專家協同特徵，避免各專家各自獨立發散。
2. **被實質摧毀的主張（Fatal Degradation）**：
   - **「NAND-Specific Pair Placement Co-Training」的必要性被徹底動搖**：
     一旦付費離線副本在容量上被證明完全可行（9.75 GiB 僅佔 1 TB Flash 的 3%），硬體感測波次就直接等於 $\lceil |U|/2 \rceil$。**硬體層面根本不存在所謂「特定的兩個 Group 必須靠演算法撮合到同一個 WL」的剛性限制**！演算法退化為通用的「縮小活躍支援集聯集」，與 NAND 底層物理幾何徹底脫鉤。
   - **新意界線（Prior Art Distance）**：
     若目標純粹為縮小跨 Token 或同 Token 的 Expert/Neuron 聯集，Apple 的跨 Token 路由一致性（arXiv:2609.22471）、Mixture-of-Masks、ModularMoE 及 SpecFit 已經在通用算法層面探索過類似機制。候選 B 若無法證明其與 Flash 物理特性有不可分割的綁定，將退化為「現有模型剪枝技術在特定形狀上的工程重組」。

#### 是否還值得進行單次有界 GT 品質測試？
- **支持理由（PRO）**：
  儘管 NAND-specific 必要性被削弱，但「$M=4I$ 非線性配對池能否在固定 Router 下以 $\le 2.75I$ 的活躍度維持品質」本身依然是一個未解的**模型壓縮核心問題**。單層測試耗時有限，能為該表徵家族蓋棺定論。
- **反對理由（CON）**：
  若該測試即使成功，最終架構也僅是一個「模型壓縮 + 離線副本查找」，完全失去了原先宣稱的「In-Flash PIM 專屬物理優勢」。若目標是追求頂級體系結構論文的 публикации 價值，該命題已喪失核心硬體立足點。

---

## 4. Student 精度與 Host 記憶體階層的雙重絞殺

### 4.1 W8 Student vs. Uniform Native Q4_K 的字節絕境

在 B03 中已推導出 W8 Student 打平 Native Q4_K 的剛性字節條件：
$$U < 0.34375 K I \implies U < 2.75 I \quad (\text{對 OLM } K=8, I=1024)$$
換算為細群組粒度（$T=128 \implies I = 8\text{ Groups}$）：
$$U_{\text{groups}} < 2.75 \times 8 = \mathbf{22\text{ Groups}}$$

#### 偶數實體對讀取過衝（Read Overshoot）：
在全配對副本下，硬體以 1 MiB 實體對（2 個 Groups）為單位讀取：
- 若訓練學出的精確聯集為 $U_{\text{groups}} = 21$ 個群組：
  硬體必須發出 $\lceil 21/2 \rceil = 11$ 次 Wave，**實際讀出 $11 \times 2 = \mathbf{22\text{ 個 Groups}}$（11.25 MiB）**！
- 此時 W8 Student 讀取的 GU 字節數剛好被偶數取整推升至臨界點 $22$ Groups。加上 Descriptor 與 Metadata 的開銷，**W8 Student 的搬運總字節數已實質超越 Native Q4_K**！

---

### 4.2 Host 全池快取（Host-Cached GU）的現實封殺

- **OLM 全模型 16 層 GU 總量**：Canonical W8 下為 **256 MiB**，Q4_K 下僅為 **144 MiB**。
- **DeepSeek-V2-Lite 26 層 GU 總量**：Canonical W8 下為 **572 MiB**，Q4_K 下僅為 **322 MiB**。
- **現實絞殺**：
  在任何標配 8 GB RAM 的邊緣主機（如 Apple M 系列、邊緣 AI 盒子）中，**常駐 144~322 MiB 的 GU 全池完全不構成記憶體壓力（佔比 $<4\%$）**。
  - 當 Host 快取了 GU，Flash 端感測 GU 的次數直接歸零。
  - Flash 僅需串流讀取私有 Down 矩陣（每次固定 $K \times I \times H$）。
- **In-Flash 唯一殘存的極窄邊界**：
  唯有在「模型擴展至 $H \ge 8192$ 導致全池超數十 GB」或「極限無 DRAM 邊緣微系統」中，In-Flash 的群組調度才具備物理存在意義。

---

## 5. 訓練演算法的對等性重構

面對全配對副本強對照，訓練演算法若繼續懲罰單一固定 16-pair 的覆蓋，本質上是在**最佳化一個虛構的錯誤目標**。

### 5.1 修正後的真物理 Wave 損失函數
在全配對副本假設下，訓練目標應回歸其真實硬體成本：
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{qual}} + \beta \cdot \mathbb{E}_{R \sim \pi(x)} \left[ \left\lceil \frac{1}{2} \left| \bigcup_{e \in R} \mathcal{S}_e \right| \right\rceil \right]$$
- 由於 $\lceil |U|/2 \rceil$ 具有半整數階躍，在連續優化中採用軟鬆弛平滑項：
  $$\mathcal{R}_{\text{union}} = \sum_{g=1}^G \left( 1 - \prod_{e \in R} (1 - m_{e, g}) \right)$$
  優化器只需專注於**最小化活躍細群組的聯集總數 $|U|$**，將實體配對工作完全交給部署時的離線圖查找。

---

### 5.2 隔離驗證協議（Progress Tracking without Quality Cheating）

為確保訓練過程的真實性，建立以下嚴格封閉規範：
1. **TRAIN-Only 進度檢查**：
   在訓練過程中，僅允許在 TRAIN 保留的 Holdout 子集上計算：
   - 平均活躍群組數：$\bar{U} = \mathbb{E}[|U|]$。
   - 硬遮罩離散化後的相對均方誤差（Discrete RelMSE）。
2. **硬部署狀態完整性判斷（Non-empty Functional Check）**：
   在進度檢查中，若發現任何專家的有效支援集基數 $\sum_g m_{e, g} < 8$，或私有 Down 矩陣出現數值全零，判定為**死鎖退化（Trivial Degeneracy）**，直接終止訓練，不進入昂貴的最終評估。

---

## 6. 科學可否證對照與最窄未解命題

### 6.1 最少兩個先驗否證對照組（Two Value-Attribution Controls）

若單層 GT 品質檢驗通過，必須以同一輪次內的以下兩個關鍵對照組決定本候選的發表價值：

```
                              [單層檢驗品質通過 (NLL 劣化 < 容許值)]
                                                 │
                   ┌─────────────────────────────┴─────────────────────────────┐
                   ▼                                                           ▼
       [Control A: 純重建池化 + 離線全配對]                         [Control B: Host 常駐 GU + Flash 讀取 Down]
       - 相同 M=4I 池，無任何硬體損失引導                           - 全池 16 MiB 放入 Host RAM 運算
       - 部署時直接使用 9.75 GB 全配對副本                          - 測量端到端延遲與能耗
                   │                                                           │
                   ▼                                                           ▼
       [檢驗: 聯合訓練是否真有顯著的 |U| 壓縮?]                   [檢驗: In-Flash 是否在真實延遲上能擊敗 Host?]
```

- **判決準則 1（演算法價值否證）**：
  若 Control A（純重建）在離線使用全配對副本時，其平均感測波次 $\lceil |U|/2 \rceil$ 與經過複雜 Joint 訓練的 Candidate 差距小於 **1 個 Wave**，則判定 **Joint Objective 缺乏實質演算法貢獻**。
- **判決準則 2（系統架構價值否證）**：
  若 Control B 在標準 PCIe/CXL 匯流排環境下的單 Token 總延遲顯著優於 In-Flash 方案（因為 In-Flash 仍需等待 Down 的多波次感測），則判定 **In-Flash GU 候選缺乏系統級部署價值**。

---

### 6.2 最終命名：最窄尚未解科學命題

經過六個 Block 的極限剖析，候選 B 的命題已被剝離掉所有虛構的硬體防禦，收斂至唯一具備可否證性的核心科學問題：

> **「在固定原始 Router 與保持 Ground-Truth NLL 損失不大於 Native Q4_K 的前提下，是否存在一種非線性特徵池聯合訓練方法，能將 64 個專家的 Top-8 活躍群組聯集 $\mathbb{E}[U]$ 壓縮至小於 $22\text{ Groups}$（即 $\le 2.75I$），從而在搬運總字節數與理論最優物理感測波次上，同時超越 Native Q4_K 與純重建離線全配對基線？」**
