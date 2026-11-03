# Round 3 候選思考 B Block 07：2×8-Chip 子群副本幾何、圖匹配非次模性與真實介面邊界備忘錄

**記錄標識**：`round03-thinking/gemini_formal_b_block07_subgroup_replica_and_interfaces.md`（純文字交付，無工具呼叫、無程式碼/資料讀取、無計時轉移）  
**當前累積實質工作**：Block 01~06 合計 594.72s + 本區段實質推導。未達 1800 秒門檻，本區段針對黑臉提出的「更低成本之 $2\times$ 8-Chip 子群副本（2×8-Chip Subgroups）」公平能力進行完整數學、圖論與系統級展開。

---

## 0. 前序備忘錄數值核算與邊界精確自查

1. **字節與形狀核算修正**：
   - 在 OLM 幾何下（$H=2048, T=128$），單個 Group 大小為 $256 \times 2048\text{ B} = 512\text{ KiB}$。
   - 若活躍群組聯集為 21 個，因實體配對需補齊偶數，讀出 22 個群組：
     $$22 \times 512\text{ KiB} = 11,264\text{ KiB} = \mathbf{11.0\text{ MiB}} \quad (\text{先前誤植為 } 11.25\text{ MiB})$$
   - 加上 W8 私有 Down（$K=8, I=1024, H=2048$）：
     $$8 \times 1024 \times 2048 \times 1\text{ B} = 16,384\text{ KiB} = \mathbf{16.0\text{ MiB}}$$
   - **Student 讀取總量**：$11.0\text{ MiB} + 16.0\text{ MiB} = \mathbf{27.0\text{ MiB}}$。
   - 這一數值**完全等於** Uniform Native Q4_K 原始模型的未壓縮形狀總量（GU $18\text{ MiB}$ + Down $9\text{ MiB}$ = $27\text{ MiB}$）。在此邊界上，Student 相對於 Native Q4_K 在權重傳輸字節上**淨收益恰好為零**。嚴禁將 W8 GU（11 MiB）與低位元 Down（9 MiB）非法拼裝以宣稱 20.25 MiB 的不對稱優勢。
2. **審查規則與方法學邊界收斂**：
   - 撤銷「若 Down 為零必為技術 Bug」的斷言；撤銷「單層測試可對整個表徵家族蓋棺定論」的過度外推；撤銷自訂「$\Delta < 1\text{ Wave}$ 即判定論文失敗」的武斷門檻。
   - 主機快取常駐僅證明「無 Flash GU 讀取延遲」，**不自動等於全系統端到端延遲必勝**；全面廢除「絞殺」、「唯一殘存」等情緒化修辭，以嚴格服務時間方程推導。
   - 原始來源模型的真實 Mixed-Q4_K_M 張量未經實體解析，不可隨意代換為均勻 Q4 數值契約。

---

## 1. 2×8-Chip 子群副本構造與實體執行契約

### 1.1 [本段推導] 子群物理條帶化與儲存容量核算

- **硬體架構假定**：
  系統具備 16 顆 Flash Chip，硬體板級具備獨立的 Channel/Chip-Select 控制能力，可劃分為兩個對稱的 8-Chip 子群：
  - **Subgroup A**：Chip 0 至 Chip 7。
  - **Subgroup B**：Chip 8 至 Chip 15。
- **單群組跨 8-Chip 條帶化（Striping）**：
  單一 Group 體積為 $512\text{ KiB}$，跨 8 顆 Chip 均勻分配：
  $$\text{每 Chip 分配體積} = \frac{512\text{ KiB}}{8\text{ Chips}} = \mathbf{64\text{ KiB/Chip}}$$
  - 每個 Chip 內部剛好佔用 4 個 Plane 各 1 個 16 KiB 頁面（$4 \times 16\text{ KiB} = 64\text{ KiB}$）。
  - 單一 Group 即可**完整填滿單一 8-Chip 子群的單次 4-Plane Wave**！
- **雙份副本池配置（2× Full Canonical GU Pool）**：
  - Subgroup A 存放一份完整的 Canonical GU 池（$G$ 個群組）。
  - Subgroup B 存放第二份完全相同的 Canonical GU 池（$G$ 個群組）。
  - **全池副本放大係數僅為 $2\times$**（而非 B06 全配對方案的 $31\times$ 或 $43\times$）。
  - *OLM（16 層，$G=32$）*：單層 GU 僅佔 $2 \times 16\text{ MiB} = \mathbf{32\text{ MiB}}$；16 層總量僅為 **512 MiB**。
  - *DeepSeek-V2-Lite（26 層，$G=44$）*：單層 GU 僅佔 $2 \times 22\text{ MiB} = \mathbf{44\text{ MiB}}$；26 層總量僅為 **1144 MiB**。

---

### 1.2 [本段推導] 執行管線、排程競爭與時間軸方程

當 Router 產出活躍群組集合 $U$（大小為 $|U|$）時，Host 控制器執行輪派排程（Alternating Dispatch）：
- Subgroup A 負責讀取前 $\lceil |U| / 2 \rceil$ 個群組；
- Subgroup B 負責讀取剩餘 $\lfloor |U| / 2 \rfloor$ 個群組。
- 理論最大並行感測波次為：
  $$W_{\text{ideal}} = \left\lceil \frac{|U|}{2} \right\rceil \text{ Waves}$$

#### 物理時間軸（Feasible Timeline）精確拆解：
若 Host Command 匯流排為串行發射，每個群組命令發送時間為 $\tau_{\text{cmd}}$；感測延遲為 $t_R = 28\,\mu\text{s}$；Chip 內部 GEMV 運算時間為 $t_{\text{gemv}}$：

```
時間推進 Timeline (設第 k 個 Wave 發射):
Host Cmd Bus:  [Cmd to Subgroup A: tau_cmd] ──> [Cmd to Subgroup B: tau_cmd]
Subgroup A:                   [Array Sensing t_R = 28 us] ──────> [ECC & GEMV: t_gemv]
Subgroup B:                                [Array Sensing t_R = 28 us] ──────> [ECC & GEMV: t_gemv]
                                                                        │
                                                                        ▼
                                                         [Wave k 完成屏障: Barrier]
```

- **單一波次真實服務時間方程**：
  $$T_{\text{wave}}(k) = 2 \tau_{\text{cmd}} + \max\left( t_R^{(A)} + t_{\text{gemv}}^{(A)}, \ t_R^{(B)} + t_{\text{gemv}}^{(B)} \right)$$
- **限制與脆弱點**：
  1. *命令序列化瓶頸*：若 $\tau_{\text{cmd}}$ 較大，發射延遲將直接暴露於關鍵路徑。
  2. *子群負載不均（Straggler Effect）*：若 Subgroup A 與 Subgroup B 因壞塊重映射或 ECC 迭代次數不同導致感測時間分歧，$t_R^{(A)} \neq t_R^{(B)}$，整體管線受限於較慢的子群。
  3. *Down 投影屏障*：每個專家對應的 8 個群組可能分散於 Subgroup A 與 B，**Down 投影依然必須等待該專家全部群組就位後，經由全域 F32 $a_{\max}^{(e)}$ 量化方可啟動**。

---

## 2. 有限成本圖匹配的數學邊界：非次模性與非超模性證明

### 2.1 [本段推導與數學證明] 成本函數 $f(U) = |U| - \nu(G[U])$ 之結構性質

設預置實體對邊集為 $E$，單副本 Fallback 讀取已付費。
服務集合 $U$ 的波次成本為：$f(U) = |U| - \nu(G[U])$。
定義邊際成本增加量為：
$$\Delta_f(S, x) = f(S \cup \{x\}) - f(S) = 1 - \left( \nu(G[S \cup \{x\}]) - \nu(G[S]) \right) = 1 - \Delta_\nu(S, x)$$
由於向圖中加入一個頂點 $x$ 最多使最大匹配數增加 1，故 $\Delta_\nu(S, x) \in \{0, 1\}$，進而 $\Delta_f(S, x) \in \{0, 1\}$：
- 若 $x$ 能與 $S$ 中的未匹配頂點形成新匹配邊，$\Delta_\nu = 1 \implies \Delta_f = 0$（邊際波次為 0）；
- 若 $x$ 無法擴展匹配，$\Delta_\nu = 0 \implies \Delta_f = 1$（邊際波次為 1）。

---

### 2.2 [本段推導] 嚴格紙上反例（Paper Math Only）

#### A. 非次模性證明（Not Submodular）
次模性要求：對任意 $A \subset B$ 及 $x \notin B$，必有 $\Delta_f(A, x) \ge \Delta_f(B, x)$。

- **構造反例**：
  考慮 3 頂點路徑圖：$a - b - x$。邊集 $E = \{(a, b), (b, x)\}$。
  取子集：
  $$A = \{b\}, \quad B = \{a, b\} \quad (A \subset B)$$
  考慮新加入頂點 $x$：
  1. **在子集 $A$ 上**：
     - $G[A] = \{b\}$，邊數為 $0 \implies \nu(G[A]) = 0$。
     - $A \cup \{x\} = \{b, x\}$，包含邊 $(b, x) \implies \nu(G[A \cup \{x\}]) = 1$。
     - 邊際匹配增量：$\Delta_\nu(A, x) = 1 - 0 = 1$。
     - **邊際成本**：$\Delta_f(A, x) = 1 - 1 = \mathbf{0}$。
  2. **在子集 $B$ 上**：
     - $G[B] = \{a, b\}$，包含邊 $(a, b) \implies \nu(G[B]) = 1$。
     - $B \cup \{x\} = \{a, b, x\}$，包含邊 $(a, b)$ 與 $(b, x)$。由於兩邊共享頂點 $b$，最大匹配數仍為 1：$\nu(G[B \cup \{x\}]) = 1$。
     - 邊際匹配增量：$\Delta_\nu(B, x) = 1 - 1 = 0$。
     - **邊際成本**：$\Delta_f(B, x) = 1 - 0 = \mathbf{1}$。
- **結論**：
  $$\Delta_f(A, x) = 0 < 1 = \Delta_f(B, x)$$
  較大集合 $B$ 的邊際成本反而嚴格大於較小集合 $A$。**$f(U)$ 嚴格不是次模函數（Not Submodular）**。

#### B. 非超模性證明（Not Supermodular）
超模性要求：對任意 $A \subset B$ 及 $x \notin B$，必有 $\Delta_f(A, x) \le \Delta_f(B, x)$。

- **構造反例**：
  考慮獨立邊 $(a, b)$ 與孤立頂點 $c$。邊集 $E = \{(a, b)\}$。
  取子集：
  $$A = \{c\}, \quad B = \{a, c\} \quad (A \subset B)$$
  考慮新加入頂點 $b$：
  1. **在子集 $A$ 上**：
     - $A = \{c\} \implies \nu = 0$；$A \cup \{b\} = \{b, c\}$，無邊 $\implies \nu = 0$。
     - $\Delta_\nu(A, b) = 0 \implies \Delta_f(A, b) = 1 - 0 = \mathbf{1}$。
  2. **在子集 $B$ 上**：
     - $B = \{a, c\} \implies \nu = 0$；$B \cup \{b\} = \{a, b, c\}$，包含邊 $(a, b) \implies \nu = 1$。
     - $\Delta_\nu(B, b) = 1 - 0 = 1 \implies \Delta_f(B, b) = 1 - 1 = \mathbf{0}$。
- **結論**：
  $$\Delta_f(A, b) = 1 > 0 = \Delta_f(B, b)$$
  較小集合 $A$ 的邊際成本反而嚴格大於較大集合 $B$。**$f(U)$ 亦嚴格不是超模函數（Not Supermodular）**。

#### 演算法理論推論：
由於 $f(U)$ 既非次模亦非超模，**經典的次模貪婪覆蓋定理（Nemhauser $(1 - 1/e)$ 近似界限）在不完整配對圖上徹底失效**。
對此類硬體成本的優化不存在通用的多項式近似保證，任何聲稱具備貪婪保證的說法在數學上皆不成立。

---

## 3. 大模型容量 Goldilocks 邊界（非 DRAM-less 假定）

### 3.1 [本段推導] 符號化記憶體預算方程

定義系統全域參數：
- 模型層數 $L$，隱藏維度 $H$，單專家中間維度 $I$，總專家數 $E$，活躍專家數 $K$。
- 單權重字節數 $b$（W8 下 $b=1$）。
- 共享原子池大小 $M = 4I$。
- **全模型 Canonical GU 池總量**：
  $$\text{Vol}_{\text{GU}} = L \times (2 \cdot M \cdot H \cdot b) = 2 L (4I) H b = \mathbf{8 L H I b}$$
- **全模型私有 Down 總量**（每專家支援度 $s = I$）：
  $$\text{Vol}_{\text{Down}} = L \times (E \cdot s \cdot H \cdot b) = \mathbf{L E H I b}$$

設主機 DRAM 實體預算為 $B_m$。定義扣除 KV Cache、Shared Expert、Attention/Non-FFN 權重與系統 Runtime 後的剩餘可用記憶體：
$$B_{\text{avail}} = B_m - \left( B_{\text{KV}} + B_{\text{shared}} + B_{\text{nonFFN}} + B_{\text{runtime}} \right)$$

---

### 3.2 [本段推導] 三大系統區域劃分

```
[區域 1: Host-Dominant 區間]
  條件: 8 L H I b <= B_avail
  現象: 全模型 GU 全池可完整常駐於 Host RAM 中。
  後果: Flash 端 GU 感測次數為 0，In-Flash 的任何 GU 波次優化毫無存在價值。
         │
         ▼
[區域 2: In-Flash Goldilocks 區間]
  條件: 8 L H I b > B_avail  且  B_avail >= B_KV + B_runtime
  現象: 模型維度巨大或 Context 極長使得 B_KV 膨脹，Host 無法容納 GU 池。
  後果: GU 與 Down 必須串流自 Flash；In-Flash 的算術去重與子群並行產生真實頻寬節省。
         │
         ▼
[區域 3: Capacity Breakdown 區間]
  條件: Flash 物理儲存無法容納模型或實體 Wave 嚴重碎裂。
```

- **真實數值範例（非極端微控制器）**：
  設 $H = 4096, I = 2048, L = 32, E = 64$（中型 MoE）：
  $$\text{Vol}_{\text{GU}} = 8 \times 32 \times 4096 \times 2048 \times 1\text{ B} = \mathbf{2.147\text{ GB}}$$
  - 若為短文本（$\text{seq} = 2\text{k}$），$B_{\text{KV}} \approx 0.5\text{ GB}$，在 8 GB 主機中 $B_{\text{avail}} \approx 4\text{ GB} > 2.15\text{ GB}$ $\implies$ **處於區域 1，Host 快取勝出**。
  - 若為長文本推論（$\text{seq} = 64\text{k}$），$B_{\text{KV}}$ 膨脹至 $6.0\text{ GB}$，導致 $B_{\text{avail}} < 1.0\text{ GB} < 2.15\text{ GB}$ $\implies$ **進入區域 2，Host 無法快取 GU，In-Flash 具備客觀生存動機**。

---

## 4. 泛用性與來源模型介面（Source Interface）邊界審查

### 4.1 [本段推導] Pre-FFN Scaling 對特徵共享的數學破壞

在部分現代 MoE 架構中（如 LLaMA-4 所見之 `weight_before_ffn = true`）：
- 進入第 $e$ 個專家的輸入並非原始 $x$，而是先經過 Router 權重縮放：$x_e = \alpha_e \cdot x$。
- 專家的中間特徵為：
  $$\phi_j^{(e)} = \text{SiLU}(g_j^\top x_e) \cdot (u_j^\top x_e) = \text{SiLU}(\alpha_e \cdot g_j^\top x) \cdot (\alpha_e \cdot u_j^\top x)$$
- **SiLU 非齊次性破壞**：
  因 $\text{SiLU}(\alpha z) \neq \alpha \text{SiLU}(z)$，純量 $\alpha_e$ **無法自 SiLU 內部提取**！
  $$\phi_j^{(e)} \neq \alpha_e^2 \cdot \left[ \text{SiLU}(g_j^\top x) \cdot (u_j^\top x) \right]$$
- **結論**：
  若來源模型包含 Pre-FFN Scaling，不同專家對同一個輸入 $x$ 產生的非線性響應完全不同。**候選 B 的「先算一次 $\phi_j(x)$ 再跨專家共享」的底層數學基石在此類模型上直接失效**！若要強行使用，必須作為有損近似結構（重新訓練），不可宣稱為無損替代。

### 4.2 動態路由與形狀邊界清單
1. **SoftMoE / Slot-MoE**：不同專家處理不同的混合 Token Slot（$x_{\text{slot}} \neq x$），完全不滿足同 $x$ 假設。
2. **$I$ 非 $T=128$ 整除**：如 $I=1408$（DeepSeek-V2-Lite），$1408 / 128 = 11$ 個整數群組，可整除；若遇非整除維度，必須支付 Padding 或 Tail Group 額外存取代價。

---

## 5. 最小單層品質測試的可解讀性規範（Layer 8 of 16 in OLM）

### 5.1 [本段推導] 逐位可比性與統計相關性陷阱

1. **入口逐位一致性（Bit-Exact Entrance）**：
   在固定測試序列且使用 Ground-Truth Token IDs 條件下：
   - 由於 Layer 8 是**全模型中唯一被替換的層**，Layer 0 至 Layer 7 完全由原始 Native 權重生成。
   - 高層的任何變動在當前 Token 前向傳遞中**無法穿越時間或反向傳播污染低層**。
   - 因此，三者（$S, B, C$）在 Layer 8 入口處的 $x_8$、Router Logits 與 $\alpha_e$ **嚴格具備逐位一致性（Bit-Exact）**。
2. **長度與採樣覆蓋界線**：
   若特徵捕獲僅取前 32 個 Token，而測試評估包含長達 512 的序列，**不得宣稱完成了全位置覆蓋**。
3. **統計檢定之成對群集 Bootstrap（Clustered Paired Bootstrap）**：
   - 自然語言 Token 之間存在強烈的自回歸局部相依性，**採用 Token 級別的 I.I.D. Bootstrap 會嚴重低估方差，導致偽陽性結論**。
   - 正確協議：必須在 **Book / Document 級別** 進行成對差值抽樣：
     $$\Delta \text{NLL}_k = \text{NLL}_C(\text{Book}_k) - \text{NLL}_S(\text{Book}_k)$$

---

## 6. 有限訓練故障與活動規則表（Campaign Case Table）

為消除「發散即否定架構」或「反覆微調逃避失敗」的模糊性，確立以下封閉判決表：

| 測試現象 / 終止狀態 | 判定類別 | 是否消耗 8 次失敗額度 | 後續強制動作 |
| :--- | :--- | :--- | :--- |
| **梯度出現 NaN / 數值下溢 / 流程中斷** | **Technical Hold** | **否（不計入）** | 修復學習率、數值截斷或程式碼實作，重新發起該 Run |
| **張量維度不符 / 索引接線錯誤 / Schema 錯誤** | **Wiring Error** | **否（不計入）** | 判定為測試治具缺陷，修復測試碼 |
| **三階段完整執行，$\Delta \text{NLL}_{C-S}$ 信賴區間跨過容許門檻** | **Inconclusive** | **否（不計入）** | 擴大校準樣本量以收斂信賴區間 |
| **三階段完整執行，確認 $\Delta \text{NLL}_{C-S} > \delta_{\text{budget}}$** | **Bounded Failure** | **是（消耗 1 次）** | **承認該有限 Recipe 失敗**，進入新方向或新候選 |
| **三階段完整執行，$\Delta \text{NLL}_{C-S} \le \delta_{\text{budget}}$ 通過** | **Candidate Pass** | **否（成功）** | **立即啟動後驗必要性對照（Controls A & B）** |

---

## 7. 總結：最窄未解命題與唯一最強 Cheap Control

### 7.1 唯一最強 Cheap Control：$2\times$ 8-Chip 子群副本 + 純品質訓練

> **對照組構造**：
> 採用本區段提出的 $2\times$ 8-Chip 子群副本（OLM 僅需 32 MiB/層），**訓練時完全不加入任何 NAND 實體 Slot 損失（純品質重建）**；推論時由 Host 依據選中的活躍群組，均勻輪派至兩個子群平行感測。

#### 摧毀力分析：
此對照組在儲存端僅支付微不足道的 $2\times$ 副本（32 MiB/層），即可在推論時**無條件保證實體感測波次精確達到 $\lceil |U|/2 \rceil$**。
它徹底拔除了「演算法必須感知底層 Physical Pair」的藉口。**若候選 B 的複雜 Joint Training 所學出的 $|U|$ 無法顯著小於純品質訓練所自然產出的 $|U|$，則候選 B 的硬體聯合學習假說宣告實質死亡**。

---

### 7.2 目前唯一具備辨識力的最窄科學命題

> **「在輸入共享的 SwiGLU MoE 中，當採用僅具備 $2\times$ 副本開銷的雙 8-Chip 子群架構時，以最小化活躍聯集 $\mathbb{E}[|U|]$ 為目標的聯合遮罩訓練，能否在不突破單層 GT NLL 預算的前提下，將活躍群組聯集壓縮至 $|U| \le 20$（即感測波次 $\le 10$ Waves），從而在端到端 Flash 讀取延遲上，實質戰勝『純品質訓練 + 輪派排程』對照組？」**
