# Round 02 Thinker Note (Gemini 4 Argon High): Overcomplete Pair/Subcoalition Student Bank

- **角色**：Gemini 4 Argon (High) 思考者（僅讀原始文件／公開文獻與撰寫本筆記；未執行任何 terminal 指令、未探測任何候選模型 weights/activations、未撰寫實驗碼）
- **正式候選時鐘與階段紀錄（已依 Root 最新指示修正）**：
  - 保守正式思考時鐘：**`2026-11-02 20:38:00 UTC` 起算，最早 `2026-11-02 21:08:00 UTC`**（修正原先誤記之 `20:37 / 21:07`；且時鐘屆滿**絕非自動實驗准入**，一切以 Root 凍結協定與審查門檻為唯一准入依據）。
  - 若中途因邏輯支配或先例覆蓋被否決，則執行 `thought-reject` 並切換候選，不消耗 Round 2 實驗次數（目前剩餘 9 次有效科學實驗機會）。
  - 本文件包含 **Block 1 初探紀錄** 與 **Block 2 交叉詰問、明確勘誤撤回、Goldilocks 機理重構、Factorial 基準與 Stage-0 准入評估**。

---

## Block 2：交叉審閱勘誤、先例全文更新、Goldilocks 深層機理與方向修正

在讀取 `results/research-2026-11-03-no-cr/round02-pairbank/thinking.md` 並交叉 Sol 與黑臉（Critic）之審查意見後，以下先對 Block 1 的不精確推導作**明確勘誤與撤回**，再展開更嚴密的機理與反駁分析。

### 2A. 明確勘誤與撤回清單（Errata & Retractions on Block 1）

1. **撤回「SwiGLU 二次正齊次性」與「部署態嚴格對 $t$ 仿射」之過度宣稱（修正 Block 1 改進 I1）**：
   - **數學勘誤**：$\text{SiLU}(c g) = \frac{c g}{1 + e^{-c g}} \neq c \, \text{SiLU}(g)$（$\sigma(cg)$ 隨尺度改變飽和區與泰勒高階項），故 SwiGLU **並非**一般正齊次函數，亦不能宣稱拼接 `[x; t]`「必然因破齊次而失敗」。Host 端通道調製 $\tilde{h}(x, t) = (\mathbf{a}_{ij} + t \mathbf{b}_{ij}) \odot h(x)$ 僅是結構乾淨的**架構候選**，而非代數必然定理。
   - **量化部署契約勘誤（極關鍵）**：上述對 $t$ 的仿射性**僅存在於未量化實數數學**！一旦進入真實 `dynamic A8` 部署，Host 在送入 `Down` 前需計算 $s(x, t) = \max_r |\tilde{h}_r(x, t)| / 127$ 並做 `round(h̃ / s(x, t))`——由於 $s(x, t)$ 對 $t$ 為分段凸函數且含捨入階梯，**實際部署輸出已不再對 $t$ 嚴格仿射**！
   - **`J = 2I` Tagged-Neuron 亦不等價於雙原專家**：即便取 $J = 2I$ 將兩原專家的 `GU` 與 `Down` 原封拼接並分別乘上 $t, 1-t$，在單一學生分支下也會因 ① **共用 `dynamic A8` 的單一 `amax` scale**（大激活專家壓垮小激活專家的量化動態範圍）與 ② **共用 `Down` 每列單一權重 scale $s_{\text{row}, r}^{(ij)}$**，而與各自獨立執行 $\alpha_i F_i^{\text{W8A8}}(x) + \alpha_j F_j^{\text{W8A8}}(x)$ 產生數值差異（除非額外付出雙 partial accumulator 或 `2H` 輸出列的硬體／傳輸成本）。
   - **修正後契約**：僅未匹配的**單一原專家 fallback 分支**保留既有精確數值；**所有學生分支一律明定為近似契約（matched-quality approximation contract），絕不宣稱數值等價**。
2. **撤回「Submodular 貪婪選邊保證」（修正 Block 1 改進 I3 與 U1）**：
   - **反例確認**：對 `top-K` 路徑 $1-2-3-4$ 上的三邊 $a=(1,2), b=(2,3), c=(3,4)$，令目標函數 $\nu(B)$ 為最大 disjoint matching 大小。取 $S=\{b\} \subset T=\{b, c\}$：
     $$\nu(S)=1, \; \nu(S \cup \{a\}) = \nu(\{(1,2),(2,3)\}) = 1 \implies \Delta(a \mid S) = 0$$
     $$\nu(T)=1, \; \nu(T \cup \{a\}) = \nu(\{(1,2),(2,3),(3,4)\}) = 2 \implies \Delta(a \mid T) = 1 > \Delta(a \mid S)$$
     由於增廣路徑（augmenting path）上的互補效應，最大匹配邊際收益**違反邊際遞減，不是 submodular 函數**！正式撤回任何 `submodular` 定理保證字眼；選邊僅能作為無近似比保證之組合啟發式（如考量 2-swap／增廣對前瞻的 heuristic 或訓練集離線整數規劃）。
3. **撤回未實測之 `CPU < 1 μs` 斷言、修正 Host 成本與容量單位**：
   - $K=6$ 完全圖至多 $1 + 15 + 45 + 15 = 76$ 種 matchings 為純組合事實，但**不得在無實測下斷言 CPU 耗時 `< 1 μs`**；Host 端匹配表、`a_ij / b_ij` 或 scale metadata 的 DRAM 佔用、cache miss 與命令打包皆為**非零顯式成本**。
   - **容量單位嚴格區分**：商用 `1 TB` SSD 為 $10^{12}$ bytes（`1000 decimal GB` $\approx 931.3\text{ GiB}$，扣 OP/FTL 前單頁 LSB 三分之一約為 **`333.3 decimal GB` $\approx 310.4\text{ GiB}$**），不可混寫為 `333 GiB`。
4. **撤回「輸出列空間正交即必然損失一半能量」之粗糙下界（修正 Block 1 反例 C3）**：
   - 即使 $\text{im}(W_d^{(i)}) \perp \text{im}(W_d^{(j)})$，若在交集域 $\mathcal{X}_{ij}$ 上兩專家共享相近的 $I$ 維隱層激活特徵 $h(x)$ 且 $t \approx t_0$ 集中，單一寬度 $J=I$ 的矩陣 $W_d^{(ij)} = t_0 W_d^{(i)} + (1-t_0) W_d^{(j)}$ 仍可**零誤差**同時投射至兩個正交子空間的加權和方向！真正決定寬度 $J$ 線性輸出下界的是**交集條件輸出分佈 $T_{ij}(x, t)\big|_{(x,t)\sim\mathcal{D}_{ij}}$ 的二階矩特徵值尾和** $\sum_{r > J} \lambda_r(\mathbb{E}[T_{ij} T_{ij}^T])$（同時取決於激活稀疏度與條件變異數 $\text{Var}(t \mid x) \|F_i(x) - F_j(x)\|^2$），而非單純權重列空間夾角。

---

### 2B. 強先例全文核實更新（壓縮後僅剩的極窄新穎性走廊）

結合 Root／Sol／黑臉之全文核查與公開文獻檢索，更新先例邊界如下：

| 先例 | 核實層級 | 關鍵機制與修正認知 | 對 Pair Bank 的直接約束 |
| :--- | :--- | :--- | :--- |
| **NimbleMoE** (`2026.emnlp-main.2036` §3.1/3.3/App.E) | **Root/Sol/黑臉已核全文** | **絕非單純靜態合併**！在 $G=2$ 設定下已採 **max-weight perfect matching** 配對專家、保留**原 router 與即時 $\alpha$**、使用 shared input/GU basis + `Down` latent $\alpha$-sum + 共用 output basis + Monarch 殘差，並報告 TPOT 加速。 | 徹底封殺「共啟用圖配對＋共享基底＋即時 $\alpha$ 潛空間加權」新意！本方向唯一結構差異僅剩 **Overlapping 圖拓撲＋原專家無損 Solo Fallback（以額外 Flash 容量換取不傷 solo 域）**，且必須正面擊敗同 bank 拓撲下的 `Pairwise NimbleMoE Views`！ |
| **PCoMoE** (`2026.emnlp-main.2048` §4) | **Root/Sol/黑臉已核全文＋公開摘要** | 保留對角線原路徑 $(U_i, D_i)$ 作穩定錨點／fallback，學習相容的非對角跨專家路徑 $(U_i, D_j)$ 並做 source-grouped `GU` 計算重用。 | 「保留原對角路徑 fallback＋跨專家複合路徑」已有明確先例。 |
| **IntBMoE** (`arXiv:2609.21346v1` §4.2/4.7) | **Root/Sol/黑臉已核全文＋公開摘要** | 明確解耦 participation / execution / materialization：以有限 codebook 的複合區塊做 **offline materialization/caching**，執行期由 token 稀疏選取以節省 active work。 | 雖為 from-scratch 訓練而非 pretrained MoE 事後 pair KD，但**「以有限離線實體化空間換取更少執行期主動工作量」之一般系統概念已有直接先例**！ |
| **KDEM** (NeurIPS 2025) | **Root/Sol/黑臉已核全文** | 其 $\alpha$ 為 **per-graph / dataset 層級**，故可**每個資料集／圖離線合併一次權重**，並非如 per-token `MEO` 般必逐 token 重讀原權重。 | 修正 Block 1 將 `KDEM` 與逐 token 讀取混談之誤差；兩者差異在於 per-token 動態路由交集 vs. per-graph 靜態 $\alpha$。 |
| **MoNE / Matryoshka MoE / Lower-$K$ Distill** | **已核公開摘要** | 巢狀專家寬度（Nested width 依重要性彈性執行）、專家內神經元稀疏（Mixture of Neuron Experts）、以及彈性／蒸餾降低活躍專家數 $K$。 | 若同離線蒸餾預算的 `Lower-K Distilled MoE` 或單專家巢狀半寬副本已達同等品質—讀取曲線，跨專家成對 Bank 即屬多此一舉。 |

---

### 2C. 圍繞「Goldilocks 區間是否可能非空」的高層想像與反駁

在扣除上述強先例後，Overcomplete Pair Bank 要成立，必須同時滿足**「Solo 域不可被全域合併（否則靜態分割 `NimbleMoE` 足矣）」**且**「交集域 $\mathcal{X}_{ij}$ 卻能壓進 $J < 2I$（甚至學生比單獨跑 $F_i, F_j$ 更簡單）」**。這在物理與數學上有沒有可能非空？以下提出**三個可檢驗的非空生成機理**，並立即給予**反方向致命詰問**：

#### 想像（支持 Goldilocks 非空的三大可能機理）
1. **機理 M1：路由錐交集上的「語境神經元稀疏預緊湊（Contextual Neuron Pre-Compaction）」**：
   - 在多語義（polysemantic）預訓練專家 $F_i$ 中，整個路由 Voronoi 胞 $\mathcal{X}_i$ 涵蓋多個子主題，因此全域來看 $I$ 個神經元皆不可刪（一旦做全域靜態合併或 `NimbleMoE` 互斥配對，solo 或與第三專家 $k$ 共現時就會損毀非交集子主題）。
   - 然而，當輸入落入**雙超平面同時高分的狹窄交集楔形錐 $\mathcal{X}_{ij} = \{x : i, j \in \text{top-}K(x)\}$**（例如「同時觸發語法結構專家 $i$ 與特定領域專家 $j$」），每個專家內部可能僅有約 $35\%\text{–}45\%$ 的特定神經元子集 $A_{i \mid j}, A_{j \mid i}$ 具顯著輸出貢獻！
   - **與 AiF Flash 頁物理的連結**：Round 1 已證明在原專家矩陣內「線上跳過零星神經元」無法湊滿整張 `16-KiB` page；但若活躍神經元子集高度取決於**零 Flash 成本即可預知的離散共路由事件 $(i, j)$**，則預存學生 $S_{ij}$ 等價於將 $A_{i \mid j} \cup A_{j \mid i}$ **離線緊湊重排成連續滿頁的寬度 $J$ 矩陣**！更關鍵的是：此機理下**根本不必從頭 SGD 訓練數千個 MLP**——直接固定原教師 `GU` 篩選出的活躍特徵列、僅對 `Down` 解閉式解嶺迴歸（**Fixed Teacher-GU Feature Pruning + Closed-Form Down Ridge**），即可在極低離線成本下驗證交集壓縮性！
2. **機理 M2：共選專家的「反向共模對沖（Common-Mode Hedging Cancellation）」使合輸出比單專家更簡單**：
   - 預訓練 MoE 的各專家常在殘差流的少數高範數 outlier 方向帶有大幅度正負對沖分量（單看 $F_i(x)$ 與 $F_j(x)$ 動態範圍與有效秩都大，直接各自壓成 `W4` 會因大 `amax` 放大兩份獨立量化誤差）；但加總後的淨輸出 $t F_i(x) + (1-t) F_j(x)$ 抵消了對沖共模，其條件輸出流形的奇異值衰減可能快於單一 $F_i$ 或 $F_j$，使針對合輸出擬合／蒸餾的 $S_{ij}$ 在相同主動 byte 下勝過 `2 × W4` 原專家獨立相加。
3. **機理 M3：交界處局部混合比集中（Narrow Conditional $t$-Spread）**：
   - 若真實路由在特定高頻對 $(i, j)$ 上的 $t = \frac{\alpha_i}{\alpha_i + \alpha_j}$ 分佈並非均勻鋪滿 $[0, 1]$，而是高度集中在特徵重心 $t_0$ 附近（$\text{Var}(t)$ 小），則前述 $t$-展開秩膨脹項 $\sigma_t^2 (F_i - F_j)(F_i - F_j)^T$ 極小，使單一 $t_0$ 學生外加極低秩 $(t - t_0) U V^T x$ 修正（甚至不需 $t$ 修正）即可高精度覆蓋該對——此時我們**主動捨棄 $t$-conditioning 新意主張**，專注驗證 Overcomplete Bank 拓撲本身的必要性。

#### 反駁（為何上述三個機理仍可能全部落敗或被更簡單設計支配？）
1. **反駁 R1 —— 「單專家尾位半寬視圖（Rank-Conditional Solo Views）」方向性挑戰（潛在支配者！）**：
   - 仔細追問機理 M1：交集神經元可壓縮，究竟是因為**「專屬於特定夥伴 $j$ 的雙邊交互作用 $(i, j)$」**，還是單純因為**「當專家 $i$ 處於中後段次要權重（如 rank 3..6）時，高誤差容忍度允許只保留 $i$ 最強的前 $I/2$ 個主成分神經元」**？！
   - **構造更聰明的簡單對照／替代方向**：不建 $M=128$ 個跨專家對（需 $+2.0\times$ 容量且受困於 $\binom{E}{2}$ 覆蓋率與非 submodular 匹配損耗），改為對每個原專家 $i \in \{1..E\}$ 預存一份寬度 $I/2$ 的「次要職責精簡視圖 $\tilde{F}_i^{(I/2)}$」（僅需 **`+0.5×` 容量**、僅 $E$ 個閉式 Ridge 子模型、**100% 覆蓋所有 token 的中後段專家**、零圖匹配衝突）！
   - 若「單專家精簡視圖」或「`Conditional Representative`（兩者共選時直接只跑較強的 $F_i$ 並乘上 $m = \alpha_i + \alpha_j$，**零額外儲存**！）」在相同主動讀取量下已追平 $S_{ij}$，則跨專家成對 Bank 的 $\Theta(E^2)$ 複雜度即被證明為**假性交互（spurious pairwise complexity）**！
2. **反駁 R2 —— 雙向擠壓（Pincer Squeeze）**：
   - 對高頻對（通常含通才樞紐專家），星狀衝突限制每 token 匹配對數上限；對中低頻對，交集樣本數急遽萎縮（如均勻估計 $E=64, K=6$ 每 1024 tokens 僅約 `7.6` 個樣本；即便偏斜分佈使前段邊樣本較多，中後段邊仍面臨嚴重統計過擬合，若改用 off-route 教師查詢擴增則離線算力成本暴增）。
3. **反駁 R3 —— 量化耦合逆風**：
   - 當原專家使用各自獨立的 W8A8 量化尺度，而學生 $S_{ij}$ 被迫共用單一 `Down` 矩陣尺度與動態激活 `amax` 時，若未做跨通道尺度均衡（channel equilibration），量化雜訊可能直接吃掉機理 M2 的對沖紅利。

---

### 2D. 必需的 Factorial 對照矩陣與更聰明的高層修正建議

若進入實證檢驗，不能只拿單一 $S_{ij}(x, t)$ 比原始 `W8` baseline，必須以 **拓撲（Topology）$\times$ 表示法（Representation）** 正交拆解貢獻來源，並納入零儲存與同離線預算強基準：

1. **軸一：Bank 拓撲（驗證 Overcomplete + Solo Fallback 是否真有必要）**
   - `T0`：**靜態互斥配對分割**（如 `NimbleMoE` $G=2$ max-weight perfect matching，無原專家 solo 獨立全寬副本）；
   - `T1`：**單專家雙寬度／精度庫**（原專家全寬 + 單專家 $I/2$ Ridge 精簡視圖或 `W4/W2` 低位元副本，100% 覆蓋、無跨專家邊）；
   - `T2`：**Overcomplete Pair Bank + 無損 Solo 原專家 Fallback + 執行期 Disjoint Matching**（全部共享相同實體 Flash 容量上限、channel 與硬體能力）。
2. **軸二：成對表示法（在相同 `T2` 拓撲與主動讀取預算下，驗證何種函數形式真正有效）**
   - `R0`：**Conditional Representative**（零額外儲存！$(i, j)$ 共選時只讀較佳原專家 $F_{i^*}$ 並乘 $m = \alpha_i + \alpha_j$，solo 跑原樣）；
   - `R1`：**靜態參數／輸出平均合併** 與 **Fixed Teacher-GU Feature Pruning + Closed-Form Down Ridge**（低訓練成本試金石）；
   - `R2`：**$t_0$ 重心無條件學生 $S_{ij}(x)$** vs. **$t_0$ 學生 + 低秩 $(t - t_0)$ 修正** vs. **Host 通道調製／Tagged 學生**（若無 $t$ 已足夠，立即移除 $t$-conditioning 新意宣稱）；
   - `R3`：**Pairwise NimbleMoE Views**（在相同過完備 bank 中放入 `NimbleMoE` 共享基底＋Monarch 視圖，檢驗是「拓撲」贏還是「MLP 蒸餾表示」贏）。
3. **軸三：外部同預算強基準**
   - 同離線蒸餾算力預算之 **Lower-$K$ Distilled Student MoE**（直接把 top-$K$ 蒸餾成 top-$K'$）；
   - 同品質預算之 **`W8/W4/W2` 單／雙精度原專家庫**（含雙方皆在 `W4` 基底下的比較）。

**大方向性修正建議（降低風險與強化差異化）**：
1. **以「輸出端外掛 $(t - t_0)$ 低秩修正」取代量化前通道調製，徹底解開 `dynamic A8` 對 $t$ 的非仿射耦合**：
   - 如 `round02-pairbank/thinking.md` 所示，雙 `Down` 頭雖對 $t$ 仿射但耗費 `4HJ` 係數（當 $J \ge 1.5I$ 即無係數節省）；而 Block 1 的 `GU`-`Down` 間通道調製雖僅 `3HJ` 係數，卻因 $t$ 進入 `dynamic A8` 的 `amax/round` 而失去部署態仿射性。
   - **更優解**：主體採用在經驗重心 $t_0$ 訓練／擬合的無 $t$ 學生 $S_{ij}^{(0)}(x)$（`3HJ` 係數），若且唯若該對的 $\text{Var}(t \mid \mathcal{X}_{ij})$ 大，才外掛極小秩 $r \ll I$ 的差值頭 $\Delta S_{ij}^{(r)}(x) \approx F_i(x) - F_j(x)$（僅額外 `2Hr` 或共用 $h(x)$ 的 `Hr` 係數），並在 **Host 最終浮點輸出加總時**才乘上純量 $(t - t_0)$：
     $$\hat{y}_{ij}(x, t) = m \cdot \Big(S_{ij}^{(0),\text{W8A8}}(x) + (t - t_0)\,\Delta S_{ij}^{(r),\text{W8A8}}(x)\Big)$$
     如此一來，$S_{ij}^{(0)}$ 與 $\Delta S_{ij}^{(r)}$ 各自的 `dynamic A8` 量化器完全不含 $t$，**在真實部署態下依然保持對 $t$ 的嚴格仿射線性**，且不需雙 `Down` 全寬矩陣（`4HJ`）！
2. **閉式 Teacher-GU 篩選 + 對角尺度均衡（Diagonal Equilibration）+ Down Ridge 作為低成本主力試金石**：
   - 為避免兩原專家特徵拼接時大激活通道吃掉小激活通道的 `dynamic A8` 精度，在閉式解 `Down` Ridge 後利用校準分位數做無損對角尺度遷移 $(W_d D)(D^{-1} h)$ 平衡各通道動態範圍（仍屬近似契約，不宣稱數值等價）。
3. **以「2-Swap 增廣前瞻 / 訓練集小規模整數規劃」取代單邊貪婪選邊**：
   - 既然單邊邊際增益因路徑 $a\text{-}b\text{-}c$ 增廣效應而非 submodular（若先選橋接邊 $b=(2,3)$，單加 $a=(1,2)$ 或單加 $c=(3,4)$ 邊際皆為 0，必須做 `-b + {a, c}` 交換或成對前瞻才能跳脫局部陷阱；同理高互資訊三角團 $\{1,2,3\}$ 塞入三邊每步僅配 1 對，遠不如選互斥三邊 $\{(1,2),(3,4),(5,6)\}$ 每步配 2–3 對），選邊器應直接在訓練集壓縮路由型態（unique top-$K$ patterns）上以啟發式 2-swap 或受限 ILP 解最佳波次削減邊集，不援引錯誤的 submodular 保證。

---

### 2E. Block 2 當時之暫時性評估（已被下方 Block 3 新先例審查取代，保留供稽核）

- *(註：Block 2 當時在尚未納入 `CRISP` MobiCom'26、`SPICE` §4.2、`Parallax`、`PuzzleMoE`、`MoEQuant`、`BitsMoE` 前，曾有條件建議極小 Bounded Stage-0；經下方 **Block 3** 完整納入新先例與支配性符號推導後，**最終獨立裁決已更新為建議立即 `THOUGHT-REJECT`**，詳見 §3D。)*

---

## Block 3：新近緊密先例審查、支配性反例推導、最小 Stage-0 規範與最終 Thought-Reject 獨立裁決

- **時鐘與權限重申**：正式候選起始時間維持 **`2026-11-02 20:38:00 UTC`**（最早 `21:08 UTC`，但時間屆滿絕不構成實驗授權；本階段無任何 terminal、無模型／激活讀取、無實驗碼，僅更新本筆記）。
- **證據標記規範**：以下嚴格區分 **`[已引先例 Cited Prior]`**、**`[符號推導 Symbolic Derivation]`** 與 **`[未實測假說 Unmeasured Speculation]`**。

### 3A. 六項新近緊密先例審查與四大支配性反例（Task A）

#### 1. 新近緊密先例核實（`[已引先例 Cited Prior]`）
- **`CRISP` (`arXiv:2610.37442v1`, ACM MobiCom 2026)**：依原路由器之專家成對共現頻率與對內主從不對稱性（`dominant` vs. `auxiliary`，依移除後 PPL 增幅定義），離線將每個原專家的神經元按重要性**重排一次為連續前綴（prefix）**，僅用 **`1.0×` 原始專家儲存（無任何 student bank！）**即動態支援 `Hot`（`full/full`）、`Warm`（`full/half`）與 `Cold`（`full/skip`）寬度排程（於 Mixtral-8x7B、DeepSeek-MoE 評估）。*(嚴格聲明：絕不借用 CRISP 報告之 13% 玩具平均記憶體或 58% I/O 數字作為 AiF 數據，因兩者快取與系統範疇完全不同。)* **直接衝擊**：正面覆蓋本方向之「成對頻率＋主從角色寬度排程」與「神經元排序前綴裁剪」。
- **`SPICE` (`arXiv:2608.21240v2` §4.2) 及其無 Shared 公平延伸**：`SPICE` 保留精確原專家 fallback，並在 prefetch miss 時以駐留 shared expert 搭配低秩殘差代理（LoRE）近似目標專家。
- **`Parallax` (`parallax-draft-tech-report.pdf`)**：不依賴原生 shared expert，保留原專家＋低秩 surrogates 與原全域路由器（用於分散式訓練分解，無 Flash 推論與 wall-clock 證明，但確立「原專家＋低秩代理／fallback」屬既有架構）。
- **`PuzzleMoE` (`arXiv:2511.04805v2`, ICML)**：透過雙遮罩與正負號共享進行細粒度成對專家參數合併並封裝進浮點指數未用位元。
- **`MoEQuant` (`arXiv:2505.03804v1`) 與 `BitsMoE` (`arXiv:2606.00079v1`)**：`MoEQuant`（EBSS + AGQ）提供高品質真實 `W4` MoE 強基準；`BitsMoE`（共享基底譜分解＋ILP 位元配置）實證顯示 `DeepSeek-V2-Lite` 壓至 `2-bit` 時品質顯著退化——確立 **`W4` 為強而真實的低位元控制組，但不可假設 `W2` 免付品質代價**。

#### 2. 四面夾擊的支配性反例（`[符號推導 Symbolic Derivation]`）

- **支配反例 A1 —— 已選原專家基底＋低秩成對差值殘差（`SPICE` 公平延伸，非原論文宣稱實作）**：
  - `[符號推導]`：對任一共選對 $(i, j)$（令 $i$ 為較強或主導者，$m = \alpha_i + \alpha_j$），恆等式
    $$\alpha_i F_i(x) + \alpha_j F_j(x) \equiv m \, F_i(x) + \alpha_j \big(F_j(x) - F_i(x)\big)$$
    允許直接重用已儲存之原專家 $F_i$（`3HI` 係數），僅額外儲存並讀取秩 $r$ 線性殘差 $D_{ij}(x) = U_{ij} V_{ij}^T x \approx F_j(x) - F_i(x)$（原始係數僅 `2Hr`）。
  - **容量與主動讀取比**：同位元寬度下，單邊靜態儲存比為 $\frac{2Hr}{3HJ} = \frac{2r}{3J}$。以符號維度 $I = J = 1024, r = 64$ 代入，$\frac{2Hr}{3HI} = \frac{128}{3072} = \mathbf{\frac{1}{24} \approx 4.17\%}$（若 $I = J = 1408, r = 64$ 則為 $\mathbf{\frac{1}{33} \approx 3.03\%}$）。即在相同額外 Flash 容量下，低秩殘差庫可覆蓋 **$24\times\text{–}33\times$ 邊數**（幾乎消除組合覆蓋率瓶頸）；主動係數為 $3HI + 2Hr \approx 1.042 \times (3HI)$（即便計入 **$V_{ij}^T x$ 與 $U_{ij} z$ 兩階段串列命令、`16-KiB` 頁對齊補零 padding、獨立量化 scales**，只要學生需 $J \ge 1.25 I$ 才能保質，$D_{ij}$ 的主動讀取與靜態空間雙雙嚴格勝出）。
  - **端點精確性支配**：當 $\alpha_j \to 0$（$t \to 1$）時，$m F_i(x) + \alpha_j D_{ij}(x)$ **平滑收斂回無損原專家 $F_i(x)$**（零 `dynamic A8` 跨專家 `amax` 壓制、零主導專家失真），而全寬學生 $m S_{ij}(x, t)$ 即使在 $t=1$ 仍以近似網路取代 $F_i$。更一般地，當單步選中 $K$ 個專家並保留前 $K_0$ 個原專家精確執行時，**全體已讀之 $\{F_{e_1}, \dots, F_{e_{K_0}}\}$ 在零額外權重讀取下即構成高維非線性基底**，其餘 $K - K_0$ 個專家僅需極小低秩修正即可逼近，甚至不必按 $\Theta(E^2)$ 成對建表！
- **支配反例 A2 —— 零額外容量之原專家重要性前綴排程（`CRISP` 對照）**：
  - `[符號推導 + 已引先例]`：將每個專家的 `GU` 列與 `Down` 行（採頁對齊之 column-tile 佈局以避免行稀疏無法省頁）按離線重要性重排一次，**額外儲存為 `0 B`（`1.0×` 容量）、覆蓋率 `100%`、零長尾對樣本飢餓**，即可依成對主從關係執行 `full/half`（`4.5HI`）或 `full/skip` / `half/half`（`3HI`），直接挑戰任何需 `3.0×` 容量的 Tagged-Neuron 學生庫。
- **支配反例 A3 —— 真實 `W4` 原專家（`MoEQuant`）與 `W2` 品質地板（`BitsMoE`）之夾擊**：
  - `[符號推導 + 已引先例]`：讀取兩份 `W4` 原專家耗費 $2 \times 0.5 \times 3HI = 3HI$ bytes，與讀取單一 `W8, J=I` 學生完全同主動 bytes，但僅需 `0.5×`（純 `W4`）或 `1.5×`（`W8+W4` 雙庫）總空間且 `100%` 覆蓋所有共選專家。若將 Pair Bank 也壓至 `W4`（單對讀 `1.5HI` bytes），雖公平基準不能無損退至 `W2`（因 `BitsMoE` 證實 `2-bit` 在 DeepSeek-V2-Lite 品質下滑），但 `W4` Pair Bank 仍同時被 **`W4` 版 `CRISP` 前綴裁剪（`0.5×` 總容量）** 與 **`W4` 原專家＋低秩 $D_{ij}$（`~0.54×` 總容量）** 夾擊。

---

### 3B. Pair Bank 唯一可能勝出的狹窄非空情境（Task B：`[符號條件 + 未實測假說]`）

若不依賴不切實際的寬學生（如 $J \ge 1.5I$）且不依賴稀缺長尾對資料（僅對自然高頻前 $M_{\text{hot}}$ 對建庫），`3HJ`（$J \approx I$）成對學生庫要在數學上同時擊敗 A1（`m F_i + \alpha_j D_{ij}`）、A2（`CRISP` 單一前綴）與互斥 `NimbleMoE`，`[未實測假說]` 真實模型的高頻共選交集域 $\mathcal{X}_{ij}$ 必須**同時滿足以下三項嚴苛結構條件**：
1. **差值具高維非線性秩**：在 $\mathcal{X}_{ij}$ 上，$F_j(x) - F_i(x)$ 的有效秩遠大於 $r$（使線性低秩殘差 $D_{ij}$ 失效，必須動用數百個非線性 SwiGLU 門控）；
2. **交集局部雙邊稀疏**：在 $\mathcal{X}_{ij}$ 上，$F_i$ 與 $F_j$ 各自僅激發約 $\le 0.5 I$ 個活躍神經元（使總和寬度壓入 $J \approx I$）；
3. **夥伴旋轉之非前綴專精（Partner-Rotating Non-Prefix Specialization）**：同一樞紐專家 $i$ 與夥伴 $j$ 共選時激發的神經元子集 $A_{i \mid j}$，必須與其跟夥伴 $k$ 共選時的子集 $A_{i \mid k}$（及全域前綴）**顯著錯位**（$A_{i \mid j} \neq A_{i \mid k}$）——否則單一靜態重排前綴（`CRISP`）在零額外儲存下即可同時滿足 $(i,j)$ 與 $(i,k)$！

**懷疑論審視**：要在避免長尾樣本飢餓的小 $M_{\text{hot}}$ 下，既維持足夠的全模型 disjoint-matching 波次覆蓋率，又讓上述三條件在 16–48 層同時強成立並跨過量化耦合與 16-chip 波次取整損耗，生存窗口極度狹窄。

---

### 3C. 最小具科學意義之 Stage-0 規範（Task C：僅作條件式完整定義，非啟動建議）

倘若 Root 與黑臉最終仍決定測試上述狹窄窗口，最小且科學嚴密之 Stage-0 必須同時具備以下控制、成本入帳與失效邊界：
1. **模型與切分**：依 Root 凍結協定，涵蓋有／無 shared expert 之兩個獨立模型類別；所有邊挑選、重要性排序、對角尺度均衡、Ridge／低秩擬合**嚴格僅用 `TRAIN` 前綴／校準集**，於未見保留集評估。
2. **精確強制對照組（同實體容量上限、同硬體能力、同離線校準資料預算）**：
   - `C0`：Canonical `W8A8` 原模型與 **`MoEQuant` 風格校準之 `W4A8` / `W8+W4` 原專家庫**；
   - `C1（零額外儲存成對寬度控制）`：**`CRISP` 式重要性前綴重排**（`GU` 列與頁對齊之 `Down` 行塊前綴，`full/full`、`full/half`、`full/skip` 及 `half/half`）＋ **`Conditional Representative`**（共選時僅跑主導 $F_i$ 乘 $m$）；
   - `C2（低秩殘差強控制）`：**`SPICE` 延伸之成對殘差 $m F_i(x) + \alpha_j D_{ij}(x)$** 與**多已選原專家基底＋低秩修正**（**必須如實計入 $V_{ij}^T x$ 與 $U_{ij} z$ 兩次串列階段延遲、`16-KiB` 頁補零對齊波次、以及獨立 scale metadata**）；
   - `C3（同拓撲表示法控制）`：相同過完備邊集與 fallback 下的 **`Pairwise NimbleMoE Views`**、參數平均合併、以及閉式 `Teacher-GU Pruning + Diagonal Equilibration + Down Ridge`（對照含／不含輸出端 $(t - t_0)$ 低秩項）。
3. **顯式成本契約**：精確入帳實體儲存（統一標示 `decimal GB` 與 `GiB`）、`16-KiB` 頁內切齊與尾頁補零、16-chip `28 μs` 普通讀取波次上限、Host DRAM metadata 容量與匹配／量化耗時；學生分支明確列為非等價近似，必須量測真實自回歸生成下之對照組 KL／PPL／任務品質漂移（絕不僅以舊 routing 重播較少 bytes 充當推論證據）。
4. **預先宣告之失效範圍（Declared Failure Scope）**：若在凍結校準預算與容量上限下，Overcomplete Pair Bank 在品質—全模型波次（含 Shared 與 Attention 分母）Pareto 前沿上未能實質超越 `C0`、`C1`、`C2`、`C3` 之聯集包絡，則判定該機制在此架構與校準範疇下無效（科學負結果，消耗 1 輪）。

---

### 3D. 獨立最終裁決（Own Reasoned Verdict）：建議立即 `THOUGHT-REJECT`（消耗 0 輪），並轉向真正不同的物理機制

#### 1. 為何建議現在直接 `THOUGHT-REJECT`（不花費寶貴的實驗輪次）？
1. **獨立投稿新穎性已被四方先例鎖死**：
   - 「離線實體化有限複合區塊以減少執行期主動工作」已有 `IntBMoE`（`2609.21346v1` §4.2/4.7）；
   - 「共啟用最大權重配對＋原路由器即時 $\alpha$ 潛空間加權＋共享基底」已有 `NimbleMoE`（`2026.emnlp-main.2036`）；
   - 「成對頻率與主從角色之異質寬度排程」已有 `CRISP`（`2610.37442v1`, MobiCom'26，且零額外儲存）；
   - 「保留原專家精確 fallback＋低秩殘差／跨專家路徑」已有 `SPICE`（`2608.21240v2` §4.2）、`Parallax` 與 `PCoMoE`（`2026.emnlp-main.2048`）。
   - **沒有找到一模一樣的四合一論文並不等於具備頂會新穎性**。在缺乏原生 In-Flash 硬體物理創新的情況下，僅在 Host 端加一個 LBA 選擇控制器去挑選重疊的壓縮專家視圖，極易被審稿人合理質疑為「拼湊既有壓縮視圖的多餘複雜化（unnecessary controller complication）」。
2. **風險報酬極度不對稱**：如 3A–3B 推導，`3HJ` 學生庫同時被 `1/24` 儲存成本且保留端點精確性的 `m F_i + \alpha_j D_{ij}`、`1.0×` 儲存的 `CRISP` 前綴裁剪、以及 `100%` 覆蓋的 `MoEQuant` `W4` 三面夾殺。「並非邏輯上絕對不可能（Task B 未被純數學證偽）」絕非消耗 9 次實驗機會之一的充分理由；**在思考階段果斷 `thought-reject` 成本為 0，是科學上最嚴謹的決策。**

#### 2. 建議切換之全新物理方向（需重開全新 30 分鐘正式時鐘，目前絕不啟動任何實驗）

- **新物理候選方向**：**單字線預充電攤提之雙閂鎖漸進式普通讀取（Intra-Wordline Amortized-Precharge Dual-Latch Progressive Read, `WL-DualSense`）**
- **物理機制核心（`[已引先例數據]` + `[符號推導]`）**：
  1. `[已引先例 — paper/aif.txt §2.2, §4.2, Table 2]`：在**完全關閉跨字線 `CR-read`**（每次讀取命令前後字線與位元線皆由 `0V` 預充至 `V_PASS = 6V` 並完整放電回 `0V` 待命，**零跨命令電位保持、零連續同 block 順序限制**）的標準普通讀取下，`paper/aif.txt` 實測單次感測（`(1,3,3)` LSB，`V_REF4`）耗時 **`28 μs`**（`16 KiB`），而同一條已預充字線上的雙感測與三感測耗時分別為 **`37 μs`** 與 **`46 μs`**——即整塊 ~100 條字線的 RC 預充與放電固定稅約佔 **`~19 μs`**，而**在同一條已充電目標字線上每多踩一次 `V_REF` 感測（`tEVAL`）僅需 `~9 μs`**，且每個 TLC plane 的 Page Buffer 原生即具備 **4 個 `16-KiB` page latches**！
  2. `[符號推導與硬體協同設計]`：不在 IFP 區塊混放一般檔案資料，改將每個 TLC cell 僅編程於 **4 個雙倍寬裕度（wide-margin）閾值態**（如 `{ER, P2, P5, P7}`，每 cell 存 `2 bits` = **每條 WL 存 `32 KiB`**），並以階層式 Gray 映射將同組權重的**高 2 位元平面 `W2_hi`（`16 KiB` = `65,536` 個權重之骨幹）**綁定於單次中心閾值 `V_REF4`（Latch 0），**低 2 位元精化平面 `W2_lo`（`16 KiB` = 補齊為完整 `W4`）**綁定於同字線兩側閾值 `V_REF2, V_REF6`（Latch 1）：
     - **模式一（`1-Sense Early-Discharge 28 μs` 普通讀取）**：對容錯度高之層／投影／次要專家，感測 `V_REF4` 入 Latch 0 後直接放電回 `0V`，耗時嚴格為 **`28 μs`**，單波次每 plane 餵入 **`65,536` 個 2-bit 骨幹權重**（較標準 LSB 上之 `W4` 快 **`2.0×`**、較 `W8` 快 **`4.0×`**）；
     - **模式二（`3-Sense Single-WL Joint 46 μs` 普通讀取 + 管線重疊）**：對精度敏感之層／Attention／Shared Expert／主導專家（解決 `BitsMoE` 指出之純 `2-bit` 品質退化！），在**單一預充—放電週期（`46 μs`）**內於同條 WL 連續感測並同時讀出 Latch 0 (`W2_hi`) 與 Latch 1 (`W2_lo`) 共 **`32 KiB`（完整 `65,536` 個 `W4` 權重）**——且在感測 Latch 1 的 `18 μs` 期間，空閒的 `ECC_LITE`（`6.4 GB/s`，處理 4 planes `64 KiB` 僅需 `10.2 μs`）已同步完成 Latch 0 解碼與高位元乘累加！即便全數跑滿完整 `W4` 精度，**`46 μs / 32 KiB` 仍較兩次獨立 `28 μs` LSB 讀取（`56 μs / 32 KiB`）直接減少 `17.9%` 感測時間，且靜態 TLC 區塊消耗減半**！
- **為何此方向具備明確物理差異且非既有泛用先例？**
  1. **非 `CR-read`**：完全不跨字線保留 `V_PASS`，任意隨機跳躍讀取單一 WL 皆獨立成立。
  2. **涵蓋全模型（Attention + 有／無 Shared Expert + Routed MoE）**：不依賴專家配對與路由器分數，消滅 Amdahl 分母稀釋。
  3. **區別於 `DPA-MoE` / `BitsMoE`**：在傳統單頁 LSB Flash 上做漸進位元（`W2_hi` + `W2_lo` 分存兩條 WL），一旦需要完整 `W4` 就得付 `28 + 28 = 56 μs`（無加速且耗雙倍塊）；`WL-DualSense` 利用**單字線多閂鎖共享單次 `19 μs` 預充放電物理**，使 `W2`（`28 μs`）與完整 `W4`（`46 μs` vs `56 μs`）同時加速。
  4. **區別於 `BALANCE` (ASP-DAC'26) 與 `E-Flash`**：`BALANCE` 針對位元／層調降 **ECC 電路面積與功耗**（未改變 NAND 感測波次時間）；`E-Flash` 針對位元圖樣映射降低**讀取電流能耗**（未做單 WL 多閂鎖雙頁聯合感測與漸進位元解碼）。
- **`[未實測假說／待新 30 分鐘時鐘嚴格拷問的關鍵風險]`**：① 4 態 `{ER, P2, P5, P7}` 在缺乏原始 160 顆晶片未公開 $V_{TH}$ 分佈參數下的 `ECC_LITE` 容錯邊界如何嚴謹界定（不能捏造未公開 SPICE 數據）；② 若公平基準（如純靜態 `MoEQuant` `W4`）亦被允許使用完全相同的 4 態 `46 μs / 32 KiB` 雙頁單字線讀取硬體，則本方法必須靠「部分波次切回 `28 μs` 單感測 `W2_hi`（或線上認證提前於 `Sense 1` 後放電）」相對「全時 `46 μs` 靜態 `W4`」拉開額外差距，否則僅剩純裝置狀態編碼貢獻。此點須待 Root 若採納切換後，重新計時 30 分鐘與 Sol／黑臉共同嚴審。


---

## 附錄：Block 1 保留項目與搜尋階段排除紀錄（經 Block 2 修正後留存）

- **Block 1 仍成立之有效設計點**（已剔除上述 2A 四項勘誤）：
  - **主導專家保護匹配**：在 `top-K` 子圖匹配時，避免將高佔比主導專家（$t \to 1$ 或 $t \to 0$）與微弱尾巴強行配對（因近似誤差乘上大 $m = \alpha_i + \alpha_j$），優先配對中後段專家並結合 16-chip 波次上限感知，不減瓶頸 chip 波次之邊直接退回原專家無損 fallback。
  - **等主動讀取量基準（C1）**：`W8, J=I` 學生主動讀取 `3IH` bytes 等同於兩份原專家 `W4` 副本（`2 × 0.5 × 3IH = 3IH` bytes），必須同時檢驗雙方在 `W8` 與 `W4` 基底下的品質—讀取量 Pareto 前沿。
- **搜尋階段 Thought-Reject 備忘（不消耗實驗輪次）**：
  - *Generic Speculative Frontier + Extra PB Contexts*：前層與當前層爭奪同一組 16 chips，且單 plane 空閒 PB latches 放不下完整專家；已就緒向量順便做 MAC 等價於既有 `SLIM` grouped GEMM，分階段驗證與預算控制重疊 `FlowSpec` / `FASER` / `BigMoMo` / `DraftExpert` / `MoE-Spec`。
  - *獨立主張 Monarch / 低秩共享或位元差異化 ECC / 狀態能耗*：分別由 `NimbleMoE`（EMNLP'26）、`BALANCE`（ASP-DAC'26）、`EFlash` 直接覆蓋。


---

## Post-Block-3 勘誤與正式 Thought-Reject 結案紀錄（`2026-11-02 21:08:29 UTC`）

- **正式處置紀錄**：Root 於 `2026-11-02 21:08:29 UTC`（滿 30 分鐘以上審查後）正式 **thought-reject** 本輪之 tagged-prune 成對視圖構造；**未啟動任何實驗（仍維持 1 次有效科學失敗、剩餘 9 次實驗機會）**。需明確區分：**一般非線性成對知識蒸餾（general nonlinear pair KD）並未被數學證偽**；NativeSol 與黑臉獨立給出更強之構造性反例指出，tagged student $D'[t\phi_i; (1-t)\phi_j] = \frac{\alpha_i}{m} D'_i \phi_i + \frac{\alpha_j}{m} D'_j \phi_j$ 在代數上直接拆解為夥伴專屬之緊湊單專家視圖（partner-specific compact solo views），且相同硬體基準完全可共享同一 row scale、實際 `A8` 量化位元組／`amax`、緊湊封裝頁與 GEMV 累加器；此外單一原專家上的任意兩個二元遮罩 $A, B$ 恆可重排為單一靜態順序 $(A\setminus B,\, A\cap B,\, B\setminus A,\, \text{neither})$ 使兩者皆為連續區段——故先前廉價試點缺乏的是**模型可辨識性（lack of identification）**，而非對所有條件視圖之普遍不可能定理。
- **對上方 Block 3 自身過度推論之五項修正**：
  1. **係數比不等於品質匹配或波次絕對支配**：原始係數比（`2Hr` vs `3HJ`）**不能**證明低秩差值控制 $D_{ij}$ 在 $J \ge 1.25 I$ 時必然達到匹配品質（matched quality），亦不能單憑原始係數比略過兩階段串列執行與 `16-KiB` 頁對齊補零而直接宣稱實際波次支配。
  2. **無證據支持其餘專家僅需小殘差**：撤回「單步前 $K_0$ 個已選原專家足以使剩餘 $K - K_0$ 個專家僅需極小低秩修正」之未驗證推測（目前無任何實證支持）。
  3. **殘差端點僅單側精確**：分解式 $m F_i(x) + \alpha_j D_{ij}(x)$ **僅在單一端點 $\alpha_j \to 0$（$t \to 1$）** 精確退化回原專家 $F_i(x)$；在另一端點 $\alpha_i \to 0$（$t \to 0$）給出的是 $\alpha_j(F_i(x) + D_{ij}(x))$，並非精確 $\alpha_j F_j(x)$。
  4. **公平硬體基準同享能力**：所有對照基準均允許使用完全相同之硬體能力（含共用 row scale、`A8` 量化、`16-KiB` 緊湊頁封裝與累加契約）。
  5. **`WL-DualSense` 位元與物理邊界勘誤（並已於未採納物理池階段淘汰）**：一個 `16-KiB` 邏輯頁 = `131,072` cells = `131,072` bits；單次中心比較器感測每 cell 僅能產出 **`1 bit/cell`**（絕非單次感測每 cell 出 2 bits），故 `65,536` 個 `W2` 高位權重係跨**相鄰兩個 cells** 存放，同組 cells 之兩個邏輯頁（`32 KiB`）始構成 `65,536` 個 `W4` 權重。此外，單字線多邏輯頁一次讀入多閂鎖已有直接先例（`US9851899B2` Fig. 15A、`US9230677B2` claim 9、`IBM US20160118110`、`US9613704B2`），[aif.txt](file:///home/ericyu/expert_flash_experiment/paper/aif.txt) Table 2 之 `28/37/46 μs` 僅量測**單一邏輯頁**感測而非合法雙頁聯合 `32 KiB / 46 μs` 命令，且原生 4 個 PB latches 中輸入向量已佔 2 個，加計感測工作閂鎖、快取傳輸與共用 `ECC_LITE` 排隊將超支閂鎖預算，截斷高位 `W2` 亦不等同獨立校準之 `W2` 品質——故該方向不進入正式候選時鐘。
