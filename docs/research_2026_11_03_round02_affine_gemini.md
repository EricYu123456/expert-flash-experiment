# Round 02 新候選正式思考筆記（Gemini 4 Argon High）：Host 共用投影座標下之局部仿射專家編譯庫（Local Affine Expert Bank）

- **角色**：Gemini 4 Argon (High) 思考者（已閱讀 Root [thinking.md](file:///home/ericyu/expert_flash_experiment/results/research-2026-11-03-no-cr/round02-local-affine/thinking.md) 與 NativeSol [research_2026_11_03_new_mechanism_sol.md](file:///home/ericyu/expert_flash_experiment/docs/research_2026_11_03_new_mechanism_sol.md)）
- **正式候選時鐘與狀態**：保守正式起算時間 **`2026-11-02 21:26:00 UTC`**，最早可准入實驗時間 **`2026-11-02 21:56:00 UTC`**（須滿 $\ge 30$ 分鐘實質研究與交叉審查，時間屆滿絕不自動授權實驗；`21:26 UTC` 前之廣搜不計入時鐘）。目前累計 **1 次有效科學失敗（Round 1）、剩餘 9 次實驗機會**；本輪尚未啟動任何實驗。
- **嚴格工具邊界**：僅使用檔案閱讀／筆記寫入與公開網路文獻查詢；**零 terminal、零模型權重／激活讀取或探測、零實驗程式碼**。
- **證據標記規範**：全文嚴格區分 **`[已引先例 Cited Prior]`**、**`[符號推導 Symbolic Derivation]`** 與 **`[未實測假說 Unmeasured Speculation]`**。

---

## Block 1：第一性原理分離構造、SwiGLU 奇偶與範數收縮陷阱、六大強反例、實體資源審計與可辨識 Stage-0 設計

### 1A. 緊密先例定位與核心機制缺口（`[已引先例]` + `[符號推導]`）

1. **`TARDIS` (`arXiv:2501.10054v1` §5.1, §9) — 為何全局常數折疊在 SwiGLU 上參數暴增 `254×`？**
   - `[已引先例]`：`TARDIS` 對非門控 FFN 在高頻熱區做全局單矩陣線性折疊並配線上離群預測器回退原 FFN（故**不可宣稱「線性折疊＋離群 fallback」為首創**）；其 §5.1 指出若對每個中間神經元獨立切 $m$ 段線性區間，笛卡兒積狀態數將呈 $m^I$ 指數爆炸；§9 更明確指出門控 SwiGLU 因元素乘積具二次項，使 LLaMA-2 折疊參數暴增 **`254×`** 而未解。
   - `[符號推導 — 精確重現 254× 之代數根源與本機制之逃逸路徑]`：在非門控 $D\sigma(Gx)$ 中，$\sigma(g_i) \approx \alpha_i g_i + \beta_i$ 折疊為單一 $H \times H$ 矩陣 $D\operatorname{diag}(\alpha)G$。但在無偏置 SwiGLU $F(x) = D(\operatorname{SiLU}(Gx) \odot Ux)$ 中，即便 $\operatorname{SiLU}(g_i) \approx \alpha_i g_i + \beta_i$ 完全線性，乘上 $u_i = u_i^T x$ 仍產生**關於 $x$ 的齊次二次型** $\alpha_i \langle g_i \otimes u_i,\, x \otimes x \rangle$！要在整個 $\mathbb{R}^H$ 上全局線性折疊二次型，必須展開至對稱張量積空間 $\operatorname{Sym}^2(\mathbb{R}^H)$，算子維度為 $H \times \binom{H+1}{2} \approx \frac{1}{2}H^3$。代入 LLaMA-2-7B（$H=4096, I=11008$）：
     $$\frac{\frac{1}{2}H^3}{3HI} = \frac{H^2}{6I} = \frac{4096^2}{6 \times 11008} = \mathbf{254.02\times}!$$
     **本機制的結構差異**：不在 $I$ 個神經元上做笛卡兒分段（$m^I$），也不在全空間展開 $x \otimes x$（$\frac{1}{2}H^3$），而是利用**輸入流形上的有限 $C$ 個聯合錨點（joint anchors $c \in \{1..C\}$）＋每層單一共用投影 $z = P_l x \in \mathbb{R}^r$**，在錨點鄰域將雙線性乘積 $(g_c + \delta_g)(u_c + \delta_u)$ 線性化為一階項 $u_c \delta_g + g_c \delta_u + \mathcal{O}(\|\delta\|^2)$，把每個激活專家的兩階段非線性子圖（`GU -> Host SiLU/A8 -> Down`，讀 `3HI` bytes）編譯為 **Flash 內單階段普通讀取之 `H × r` GEMV**（讀 `Hr` bytes，無第二個 NAND 階段、無跨 Host 非線性同步屏障）。
2. **`How Linear Is a Transformer FFN?` (`arXiv:2606.19379v1`)**：
   - `[已引先例]`：以閉式精確最小平方法（OLS）定義每層線性可恢復度 $R^2_{\text{lin}}$，證實：(i) 閉式全局 OLS 是極強基準，訓練不足的 SGD 線性替代會嚴重低估線性基準；(ii) 線性可恢復度隨深度高度非單調異質（相鄰層可由 $>0.99$ 驟降至 $<0.3$）；(iii) 單層局部 $R^2$ 與完整模型 Perplexity 並非單調綁定（可能解離）。**直接協定約束**：所有線性基準與局部映射必須使用閉式解（SVD／Cholesky Ridge），且必須在深度異質選擇下直接評估端到端自回歸 PPL／任務品質。
3. **`Locally Linear Mappings` (`arXiv:2505.24293v1`)**：
   - `[已引先例]`：其輸入處重構係基於**截斷梯度之 Detached Jacobian**（滿足局部齊次關係而非真實 Fréchet 導數 $J(c)$），且僅在**同一操作點 $x$** 近乎精確；§2.2 與 §4 明載鄰近點或不同輸入即產生曲率偏離而需重算新 Jacobian。故**絕不可引用該文之奇異值低秩性作為跨 token 區域重用的證據**。
4. **`PREP` (`OpenReview:nsblPHuc4L`)、`MoLE` (ICML'25)、`SPICE` (`2608.21240`)、`NimbleMoE`**：
   - `[已引先例]`：`PREP` 用層均值錨點 Taylor 近似與 CPU Faiss 檢索做**專家重要性剪枝**而非輸出替代算子；`MoLE` 在訓練期改架構為按静态 **token ID 查表**（非上下文隱藏態專家近似，但提示必須納入分段常數輸出查表 $F_e(c)$ 作為 $r=0$ 基準）；`SPICE` 保留原專家 fallback 並加低秩殘差（但在 AiF Flash 上兩因子低秩需**兩次串列 NAND 階段**，而本機制將 $z = P_l x$ 置於 Host 一次共用計算，Flash 僅需**單階段** `H × r` GEMV）。

---

### 1B. SwiGLU 精確奇偶分解與「超越單一全局映射（$C=1$）」之構造性分離定理（`[符號推導]`）

先推導任意**無偏置 SwiGLU 專家** $F(x) = D\big(\operatorname{SiLU}(Gx) \odot Ux\big)$ 的精確全域奇偶分解。由恆等式 $\operatorname{SiLU}(g) \equiv \frac{g}{2} + \frac{g}{2}\tanh\!\left(\frac{g}{2}\right)$，對任意 $x \in \mathbb{R}^H$ 恆有：
$$F(x) \equiv \underbrace{\frac{1}{2} D\big((Gx) \odot (Ux)\big)}_{F_{\text{even}}(x)\,:\; F_{\text{even}}(-x) = +F_{\text{even}}(x)\text{（精確純二次偶函數）}} \;+\; \underbrace{\frac{1}{2} D\!\left((Gx) \odot \tanh\!\left(\frac{Gx}{2}\right) \odot (Ux)\right)}_{F_{\text{odd}}(x)\,:\; F_{\text{odd}}(-x) = -F_{\text{odd}}(x)\text{（首項三次奇函數）}}.$$

由此立得兩個正反雙向的關鍵結構結論：

1. **構造性逃逸定理（為何有限 $C$ 個局部秩 $r$ 映射可嚴格支配任意單一全局滿秩映射 $C=1$）**：
   - 設輸入位於 RMSNorm 球面且在某子空間 $\operatorname{row}(P_l)$（$\dim = r$）內由 $C_0 = 2M \le C$ 個對稱語義簇 $\pm \mu_1, \dots, \pm \mu_M$（$\mu_m \perp \mu_{m'}$，簇內擾動 $x = \pm \mu_m + \delta$，$\|\delta\|_2 \le \epsilon \ll 1$）組成；考慮專家之偶二次分量 $F_{\text{even}}(x) = \sum_{m=1}^M v_m (\mu_m^T x)(b_m^T x)$（其中 $b_m \in \operatorname{row}(P_l)$）。
   - **全局單一映射（$C=1$，即便給滿秩 $r=H$ 與精確閉式 OLS）**：對任何正負對稱分佈 $p(x)=p(-x)$，偶函數與奇函數（線性項 $Wx$）在 $L^2$ 中嚴格正交，$\mathbb{E}[F_{\text{even}}(x) x^T] = \mathbf{0}$，故最優全局線性斜率 $W_{\text{OLS}}^{\text{global}} \equiv \mathbf{0}$，**對整個二次偶分量 $F_{\text{even}}$ 的解釋變異數 $R^2_{\text{lin}} = 0$**！
   - **局部錨點庫（$C = 2M$，錨點 $c = \pm \mu_m$）**：在每個簇錨點 $\pm \mu_m$ 處破除對稱性，局部 Jacobian $J(\pm \mu_m) = \pm v_m \big[(b_m^T \mu_m)\mu_m^T + b_m^T\big] + \sum_{m'\neq m} v_{m'}(b_{m'}^T \mu_m)\mu_{m'}^T$ 秩 $\le M \le r$，單一局部映射 $F(\pm\mu_m) + A_{e,\pm m} P_l \delta$ 將誤差壓至 **$\mathcal{O}(\epsilon^2)$**（當 $\epsilon \to 0$ 時 $R^2 \to 1$）。
2. **隱藏其內的自我反駁（Router 球冠效應使 $C=1$ 成為極強必測基準！）**：
   - 在真實 MoE 中，專家 $e$ 並非接收全空間對稱分佈，而是僅接收滿足 `top-K` 路由條件 $w_{\text{router}, e}^T x > \tau(x)$ 的**單側球冠（polar cap）** $\mathcal{X}_e$！因此路由本身已沿一維方向 $w_{\text{router}, e}$ 破除正負對稱（$\mu_e = \mathbb{E}[x \mid e] \neq 0$），使**每專家單一局部 OLS（$C=1$）** 即可捕捉沿 $\mu_e$ 展開的線性斜率！
   - 然而，在與 $w_{\text{router}, e}$ 正交的 **$H - 1 = 2047$ 維切空間** $x_\perp$ 上，若條件分佈仍近似正負平衡，則所有讀取 $x_\perp$ 的神經元之二次偶分量 $F_{\text{even}}(x_\perp)$ 在 $C=1$ 下斜率仍為 0，必須靠 $C > 1$ 個簇內錨點才能破除對稱。**這精確界定了本方法的核心實證問題：真實預訓練 MoE 專家在路由球冠內部的正交切空間曲率，究竟需要 $C > 1$ 還是 $C = 1$ 已足？**

---

### 1C. 六大強反例推導與硬體—數值設計契約（`[符號推導]`）

1. **反例一：原點零導數（$J(0)=0$）與高維歐氏質心「範數收縮坍塌（Norm-Shrinkage Collapse）」**
   - 由上節奇偶分解，$F_{\text{even}}(x) = \mathcal{O}(\|x\|^2)$、$F_{\text{odd}}(x) = \mathcal{O}(\|x\|^3)$，故在 $c = 0$ 處 **$F(0)=0$ 且真實 Jacobian $J(0) \equiv \mathbf{0}$**。
   - **更隱蔽的陷阱**：即便不取 $c=0$ 而取簇內歐氏均值 $c = \mathbb{E}[x \mid \mathcal{R}_{e,c}]$，在 $H=2048$ 維 RMSNorm 球面上（真實 token 範數恆為 $\|x\|_2 \approx \sqrt{H}\bar{\gamma}$），寬簇歐氏平均會發生顯著範數收縮 $\|c\|_2 = \rho \|x\|_2$（常有 $\rho \in [0.15, 0.4]$）。此時將 $c$ 代入解析 Jacobian $J(c)$，偶部斜率按 $\rho$ 萎縮、奇部斜率按 $\rho^2$ 萎縮，使 $Gc$ 落入接近 0 的死區而嚴重低估真實球面上的門控導數！
   - **設計對策**：錨點必須採用**球面投影質心或實際樣本中心點（spherical medoid，$\|c\|_2 \approx \text{median}\|x\|_2$）**，或直接以球面樣本之閉式局部 Ridge 擬合斜率。

2. **反例二：高維覆蓋數（Covering Number）與二階 Hessian 殘差下界**
   - 對局部二階展開，SwiGLU 沿 $\frac{\partial^2}{\partial u^2} \equiv 0$ 雖無曲率，但交叉項與門控二階項給出精確二階餘項：
     $$R_{2,i}(c; \delta) = \operatorname{SiLU}'(g_{c,i})\,\delta_{g,i}\delta_{u,i} + \frac{1}{2} u_{c,i}\operatorname{SiLU}''(g_{c,i})\,\delta_{g,i}^2.$$
   - 若專家輸入子流形之輸出敏感有效維度為 $d_{\text{eff}}$，則 $C$ 個球覆蓋之最近錨點距離僅按 $C^{-1/d_{\text{eff}}}$ 收縮：當 $C=32$ 時，若 $d_{\text{eff}}=5$ 距離減半（餘項降至 $\sim 0.25\times$）；但若 $d_{\text{eff}} \ge 20$，$32^{-1/20} = 0.841$（最近距離僅縮 `15.9%`，二次餘項仍高達 `71%`）。此外，若直接對未加權 $x$ 做 k-means，$C=32$ 個錨點極易被少數高變異但與專家無關的 outlier／位置方向耗盡。

3. **反例三：全層共用 $P_l \in \mathbb{R}^{r \times H}$ 丟棄專家專屬曲率方向（不可約零空間下界）**
   - 任何以 $z = P_l x$ 為輸入的映射（即便 $C \to \infty$ 且允許任意非線性查表）皆受限於零空間 $\ker(P_l)$（維度 $H - r$）上的條件變異下界：
     $$\mathbb{E}\big[\|F_e(x) - \hat{F}_e(P_l x)\|_2^2 \mid e\big] \ge \mathbb{E}\Big[\operatorname{Tr}\!\big(J_e(c)(I - P_l^T P_l)\Sigma_{e,c}(I - P_l^T P_l)J_e(c)^T\big)\Big].$$
   - 在 $E=64$ 的細粒度 MoE 中，若 64 個專家的敏感輸入子空間彼此正交（每個秩 $64$，聯集張滿 $H=2048$），且層輸入協方差 $\Sigma_{e,c}$ 未集中於單一低維流形，則單一共用 $P_l$（如 $r=128$ 或 $256$）平均分給每個專家的專屬維度僅 $2\text{–}4$ 維！此零空間地板是決定 $r \ll H$ 能否成立的頭號數學瓶頸。

4. **反例四：非平滑 `W8A8` 量化、中心化輸入廣播取捨與錨點邊界跳變**
   - 原生 canonical `W8A8` 含兩次動態 `amax` 量化與取整，非平滑可微；解析 $J(c)$ 僅能作為平滑初始化或收縮先驗，絕非位元等價展開。
   - **中心化 vs 廣播之精確取捨**：若不中心化而算 $A_{e,c} Q_{\text{A8}}(z) + b_{e,c}$，在 $z = z_c$ 處會發生大數抵銷且殘留 $\propto \|z\|_\infty/127$ 的量化誤差；反之採 NativeSol 建議之**中心化動態量化** $\hat{F}_e(x) = F_e^{\text{canon}}(c) + A_{e,c} Q_{\text{A8}}(z - z_{e,c})$，在 $\delta z = 0$ 處嚴格無損還原 $F_e^{\text{canon}}(c)$ 且量化誤差隨 $\|\delta z\|_\infty$ 線性縮小。**關鍵硬體核驗**：中心化使 $K$ 個選中專家擁有**各自不同的輸入向量 $\delta z_{e_k, c_k} \in \mathbb{R}^r$**（不可共用單一廣播向量），但對 $K \le 8, r \le 512$，全部 $K$ 個向量總長僅 $K \cdot r \le \mathbf{4\text{ KiB}}$（即便 $r=2048$ 亦僅 $16\text{ KiB}$），**完全塞得進 AiF 每 plane 既有的 2 個輸入 Page Buffer latches（`32 KiB`，見 [aif.txt](file:///home/ericyu/expert_flash_experiment/paper/aif.txt#L727-L743)）**，Host PCIe 額外傳輸僅 `1–4 KiB`（`< 0.5 μs`）！
   - **邊界不連續與自回歸漂移**：Voronoi 邊界切換存在 $\mathcal{O}(\|\delta z\|^2)$ 跳變；且前層近似誤差改變後續層之輸入 $x$ 與路由決策，後層 fallback **無法抹除前層已寫入 residual stream 之累積漂移**。
   - **讀前零空間守衛（修正內積盲點）**：注意 $\|x - c\|_2^2 - \|P_l(x - c)\|_2^2 = \|x_\perp - c_\perp\|_2^2$ 含有未知交叉項 $-2\langle x_\perp, c_\perp\rangle$，**無法**僅由 $z = P_l x$ 與 $\|x\|_2^2$ 精確算出！若不存完整 $H$ 維錨點 $c$，讀前守衛應採用**嚴格可算的 Cauchy–Schwarz 上界** $\sqrt{\|x\|_2^2 - \|z\|_2^2} + \|c_\perp\|_2 \le \tau_{\perp}$（每層僅算一次 $\|x\|_2^2 - \|z\|_2^2$，每錨點僅存 1 個純量 $\|c_\perp\|_2$，零額外 $H$ 維內積！）。

5. **反例五：長尾罕見專家校準樣本飢餓（$N_{e,c} \ll r$）**
   - 以 $16,384$ 個校準 tokens、$E=64, K=6, C=32$ 計，均勻平均每 `(expert, anchor)` 僅 **48 個樣本**，長尾專家更可能 **$< 5$ 個樣本**！若直接跑無正則局部 OLS（自由度 $r \in [128, 2048]$），在 $N_{e,c} < r$ 時必嚴重過擬合崩潰。
   - **識別性解法（Empirical-Bayes `J-Shrunk Ridge`）**：以無需迴歸樣本即可由原權重精確算出的球面中心解析切面 $A_{e,c}^{(0)} = J_e(c_{\text{medoid}}) P_l^T$ 作為收縮先驗：
     $$A_{e,c}^* = A_{e,c}^{(0)} + (Y_{\text{res}}\Delta Z^T)(\Delta Z \Delta Z^T + \lambda I_r)^{-1}.$$
     當 $N_{e,c} \to 0$ 時自動退化回解析切面 $A_{e,c}^{(0)}$（零變異爆炸）；當 $N_{e,c} \gg r$ 時收斂至最優局部線性 MMSE。實驗中必須將 `J-only` ($\lambda=\infty$)、`0-shrunk Ridge` ($A^{(0)}=0$) 與 `J-shrunk Ridge` 三者並列比較，如實歸因增益來源。

6. **反例六：Prefill／Speculative 多 token 批次重用碎裂化**
   - 當單批次有 $B_e$ 個 tokens 路由至同一專家 $e$ 時，原專家僅讀一次 `3HI` bytes（由 PB 多向量批次重用）；若這 $B_e$ 個 tokens 分散命中 $m$ 個不同錨點，局部庫需讀 $m \cdot Hr$ bytes。當 $m > \frac{3I}{r}$ 時（例如 `DeepSeek` $I=1408$：若 $r=2048$，僅 **$m \ge 3$** 即超過原專家 `8.65 MB`！反之若 $r=128$，$\frac{3I}{r} = 33 > C=32$，**對任意批次大小皆嚴格少於原專家**！），局部庫反而更慢。
   - **讀前批次合併規則**：Host 在讀前已知所有選中錨點，凡 $m_{\text{unique}}(e)\cdot \text{waves}(H\times r) \ge \text{waves}_{\text{orig}}(3HI)$ 時，直接整批切回原專家，嚴格鎖死批次退化上限。

---

### 1D. 實體波次、Flash 容量與 Host 8GB 公平帳本審計（`[已引先例]` + `[符號推導]`）

以 AiF 普通讀取基準（`16 chips × 4 planes = 64 planes`、`16 KiB` page、`tR = 28 μs`、關閉 `CR-read`）精確計算 `H = 2048` 下之實體頁數與容量（統一區分 `decimal GB` 與 `GiB`）：

| 表示形式（單專家） | 原始係數 Bytes | `16-KiB` 頁數 | Flash 階段數 | `DeepSeek` ($K=6$) 理想 64-plane 波次 | `OLMoE` ($K=8$) 理想 64-plane 波次 | `DeepSeek` 全庫靜態容量（$1664 \times C$，未含原模型 `15.7 GB`） |
| :--- | ---: | ---: | :---: | ---: | ---: | ---: |
| 原專家 `W8A8` (`DeepSeek` $I=1408$) | `8,650,752 B` | `352 + 176 = 528` | **2 串列** | `33 + 17 = 50 waves` | — | 原模型 `15.7 decimal GB` (`14.6 GiB`) |
| 原專家 `W8A8` (`OLMoE` $I=1024$) | `6,291,456 B` | `256 + 128 = 384` | **2 串列** | — | `32 + 16 = 48 waves` | 原模型 `6.9 decimal GB` (`6.4 GiB`) |
| **強基準：`MoEQuant` `W4A8` 原專家** | `4.33 MB` / `3.15 MB` | `264` / `192` | **2 串列** | **`17 + 9 = 26 waves`** | **`16 + 8 = 24 waves`** | 原模型減半（`~7.9 GB` / `~3.5 GB`） |
| 局部仿射 `W8` **$r = H = 2048$** | `4,194,304 B` | `256` | **1** | `24 waves` | **`32 waves`（劣於 `W4`！）** | $C=32$: **`223.3 decimal GB`** (`208.0 GiB`) |
| 局部仿射 `W8` **$r = 512$** | `1,048,576 B` | `64` | **1** | **`6 waves`** | **`8 waves`** | $C=32$: **`55.8 decimal GB`** (`52.0 GiB`) |
| 局部仿射 `W8` **$r = 256$** | `524,288 B` | `32` | **1** | **`3 waves`** | **`4 waves`** | $C=32$: **`27.9 decimal GB`** (`26.0 GiB`) |
| 局部仿射 `W8` **$r = 128$** | `262,144 B` | `16` | **1** | **`2 waves`** | **`2 waves`** | $C=32$: **`14.0 decimal GB`** (`13.0 GiB`) |

**兩項關鍵資源洞察（`[符號推導]`；註：關於 `W8 r=H`、群組排布 $g$、`SPICE` Host 駐留與 `FP32` $F_e(c)$ 帳本之重要修正請務必以見下方 **Block 2 §2A** 為準）**：
1. **`W8, r = H = 2048` 相對 `MoEQuant W4A8` 之頁數劣勢（惟品質與單階段效益仍待實測，見 §2A.5 修正）**：在 `DeepSeek` 上 $r=H$ 讀 `256` 頁，相對 `W4A8` 的 `264` 頁僅少 `3%` 頁數卻多耗 **`223.3 decimal GB`** 容量；在 `OLMoE` 上 `256` 頁甚至比 `W4A8`（`192` 頁）**多讀 `33.3%`**。因此降秩區間 $r \in \{64, 128, 256, 512\}$ 或同位元 `W4` 局部映射更具頁數空間，但不得在未驗證匹配品質與階段／命令開銷前先驗斷言 `W8 r=H` 已被鎖死。
2. **部分回退（Partial Fallback）之雙階段屏障留存效應**：只要單一 token 的 $K$ 個選中專家中有 **至少 1 個專家** 未通過讀前守衛而回退原專家（$K_{\text{orig}} \ge 1$），該層仍必須支付一次 `GU -> Host SiLU -> Down` 的串列同步延遲（儘管總感測頁數仍按 $K_{\text{aff}}/K$ 比例大幅下降）。成本模型必須如實計入此混合階段排程，不得假設全層屏障自動消失。
3. **Host `8 GB` 公平預算控制（已在 §2A.6 撤回雙列 `INT8` 無損編碼假設）**：`FP32` $F_e(c)$（`DeepSeek` $C=16$ 約 `218 MB`、$C=32$ 約 `436 MB`）、$P_l$、投影鍵與守衛皆須誠實計入 Host `8 GB` 預算，並賦予所有對照基準完全相等的 Host DRAM 與快取預算。

---

### 1E–1F. Block 1 當時之初步藍圖與階段問題（已被下方 Block 2 完整先例與勘誤更新，保留供稽核）

- *(Block 1 原建議聚焦低秩閉式驗證並提出五項待審問題；經下方 **Block 2** 納入 Terra 新查得之 `Flash-Jacobian`、`HalfTheNonlinearity`、`MoLAE`、`ITC-MoE`、`LAWS` 及 Root 取得之 `EBC-LLM` 作者公開源碼，並完成六項強制數學／硬體修正與黑臉新增辨識控制後，更新之完整協定與去留裁決詳見 **§2A–§2D**。)*

---

## Block 2：交叉審查強制勘誤、六大新近強先例對齊與「條件輸出子空間 vs 全局非線性抬升」核心辨識決策

- **時鐘紀錄**：同一正式候選時鐘（`2026-11-02 21:26:00 UTC` 起，最早 `21:56:00 UTC` 方可考慮實驗准入；本區塊維持純文獻與第一性原理推導，無任何實驗碼或模型／激活讀取）。

### 2A. 對 Block 1 之六項強制數學與硬體勘誤（Mandatory Corrections）

1. **條件風險精確下界 vs 局部一階跡近似，以及正交子空間維度修正**：
   - **修正條件風險地板**：Block 1 §1C.3 將 $\operatorname{Tr}(J_e(c)(I - P_l^T P_l)\Sigma_{e,c}(I - P_l^T P_l)J_e(c)^T)$ 稱為一般不可約 MMSE 下界並宣稱「$C \to \infty$ 亦無法消除」**並不成立**。對任何僅依賴觀測量 $\mathcal{I}(x) = (z = P_l x,\, s(x),\, g(x))$ 之確定性預測器，在通過守衛事件上之**精確條件風險地板**為 $\mathbb{E}[\operatorname{Tr}\operatorname{Cov}(F_e(x) \mid z, s(x), g(x), e)]$。若選擇器 $s(x)$ 或守衛 $g(x)$ 讀取完整輸入 $x \in \mathbb{R}^H$（如完整錨點距離或殘差範數），該離散索引即傳遞了關於 $\ker(P_l)$ 的資訊——當 $C \to \infty$ 以全空間錨點細分緊集時，分段近似可突破任何純 $z$ 函數之誤差限制！前述 Jacobian 協方差跡公式僅在平滑可微與高階項可略假設下的**局部一階 Taylor 近似**。
   - **維度相容性修正**：在 $H = 2048$ 空間內，`64` 個專家不可能同時擁有彼此正交之秩 `64` 子空間（因 $64 \times 64 = 4096 > 2048$）；正交最壞情況舉例應修正為「`64` 個專家各自秩 $\le 32$ 彼此正交」或「至多 `32` 個專家各自秩 `64` 彼此正交」。

2. **偶函數構造性分離定理之截距與範數修正、路由幾何聲明、撤回未實測 $\rho$ 與首項餘項標註**：
   - **補齊截距並修正構造性分離定理（`[符號推導]`）**：Block 1 §1B 未顯式檢查全局仿射截距 $b_0$——若偶函數在所有錨點 $\pm\mu_m$ 取相同常數值，則單一全局映射 $C=1$ 雖斜率 $W^*=0$，但光靠常數截距 $b_0$ 即可精確命中錨點！**嚴格修正構造**：在 $\operatorname{row}(P_l)$ 內取兩組彼此正交之單位正交向量 $\{\mu_1,\dots,\mu_M\} \perp \{u_1,\dots,u_M\}$（$\|\mu_m\|_2 = \|u_m\|_2 = 1$，$2M \le r$）及線性獨立輸出向量 $\{v_1,\dots,v_M\} \subset \mathbb{R}^H$，定義雙線性偶函數 $F(x) = \sum_{m=1}^M v_m (\mu_m^T x)(u_m^T x)$；設輸入均勻取自 $2M$ 個對稱簇 $x = \sigma \mu_m + \delta$（$\sigma \in \{\pm 1\}$，$\delta$ 在 $\operatorname{row}(P_l)$ 上零均值各向同性、標準差 $\epsilon/\sqrt{r}$）。此時：
     - 在每個錨點 $c_{m,\sigma} = \sigma \mu_m$ 處，因 $u_{m'}^T \mu_m = 0$，恆有 **$F(c_{m,\sigma}) \equiv \mathbf{0}$** 且 $\mathbb{E}[F(x)] = \mathbf{0}$！
     - 因此：(i) `r = 0` 錨點查表 $\hat{F}(x) = F(c) = 0$ 解釋變異數 **$R^2 = 0$**；(ii) 含截距之全局閉式仿射 $Wx + b_0$（即便 $r=H$）因 $b_0^* = \mathbb{E}[F(x)] = 0$ 且偶奇正交 $W^* = 0$，得 **$R^2 = 0$**；(iii) 黑臉新增控制「錨點截距＋單一全局斜率」$F(c) + A_e(z - z_c)$ 因 $+\mu_m$ 與 $-\mu_m$ 兩簇真實局部斜率符號相反（$+v_m u_m^T$ vs $-v_m u_m^T$），任何單一 $A_e$ 最佳解仍為 $A_e^* = 0$（**$R^2 = 0$**）；
     - 反之局部錨點庫在 $c_{m,\sigma} = \sigma\mu_m$ 之精確 Jacobian 為 $J(c_{m,\sigma}) = \sum_{m'} v_{m'}[(\mu_{m'}^T c_{m,\sigma}) u_{m'}^T + (u_{m'}^T c_{m,\sigma})\mu_{m'}^T] = \sigma \|\mu_m\|_2^2 v_m u_m^T = \sigma v_m u_m^T$（秩 $1 \le r$），預測 $\hat{F}(x) = \sigma v_m(u_m^T \delta)$ 對真實變異 $\Theta(\epsilon^2)$ 達成相對誤差 $\mathcal{O}(\epsilon^2) \to 0$（**$R^2 \to 1$**）。
   - **四項嚴格標註修正**：(a) MoE `top-K` 路由域係多個比較不等式之交集（且可含分組路由、sigmoid 門控或路由偏置），**並無實證顯示真實專家路由域為簡單極球冠（polar cap）**，球冠僅作玩具幾何示意；(b) Block 1 提及之 $\rho \in [0.15, 0.4]$ 屬**未實測假設（unmeasured illustration）**，撤回任何經驗典型值暗示；(c) 二階式 $R_{2,i}(c;\delta)$ 僅為 Taylor **首項（leading term）**，非有限 $\delta$ 之精確餘項（SwiGLU 三階以上高階項仍存在）；(d) 球面 medoid $c_{\text{medoid}}$ 僅為設計選項而非必要定理——區域閉式 OLS／Ridge 之自由截距 $\bar{y}_{e,c} - A_{e,c}\bar{z}_{e,c}$ **可自動吸收 Jensen 均值偏差** $\mathbb{E}[F_e(x)] - F_e(\mathbb{E}[x])$。

3. **AiF 輸入向量傳遞契約與命令延遲 vs 線路頻寬**：
   - $K \cdot r$ bytes 總量小於 `32 KiB` **不能證明**原生 AiF 免費支援 $K$ 個同時駐留之 PB 輸入上下文、命令級任意位元組偏移或 `< 0.5 μs` 發射延遲；位元組除以 PCIe 頻寬僅為傳輸時間下界而非命令啟動延遲。原生 GEMV 雖具輸入維度運算元（[aif.txt](file:///home/ericyu/expert_flash_experiment/paper/aif.txt#L747-L749)），但最小寬度與對齊格式未公開；若逐專家傳送不同中心化向量 $\delta z_{e,c}$，必須明確計入逐命令投遞延遲（或對照在 Host 端預合併偏置 $b_{e,c} = F_e(c) - A_{e,c}^{\text{deq}} z_{e,c}$ 以廣播單一共用 $Q_{\text{A8}}(z)$ 之量化精度／命令數取捨）。

4. **Host `8 GB` 常駐能力打破「唯獨共用 $P_l$ 可單階段 Flash」之宣稱，並確立全 Host／CPU 執行強基準**：
   - **低秩基準同樣可單階段讀 Flash**：`SPICE` 等全局低秩基準完全可將每專家輸入因子 $B_e \in \mathbb{R}^{r \times H}$ 常駐於 Host DRAM（以 `DeepSeek` $26 \times 64 = 1,664$ 個專家計，$r=64$ `W8` 僅 **`218.1 decimal MB`**，$r=128$ `W8` 僅 **`436.2 MB`**！）；即便 `Flash-Jacobian` 的區域專屬輸入因子 $B_{e,c}$ 在 $C=16, r=64$ `W8` 下亦約 **`3.49 decimal GB`**（`3.25 GiB`），在扣除非 FFN 權重與 KV cache 後仍可能放入 `8 GB` Host DRAM（須與基準按相同精度與容量帳本公平核計），從而同樣只需 **1 次 NAND 輸出矩陣讀取**！故**絕不可強迫低秩基準走兩次 Flash 階段**。
   - **小規模局部庫自身即可全駐 Host DRAM 由 CPU 算完**：在 $C=16, r=64$ `W8` 下，全模型局部輸出矩陣庫 $A_{e,c}$ 自身總量僅 **`3.49 decimal GB`（`DeepSeek`）／`2.15 decimal GB`（`OLMoE`）**，可能整庫直接放入 Host `8 GB` DRAM 跑 CPU 小型 GEMV，無需觸碰 `28 μs` NAND 讀取地板（但 CPU 算力耗時非零，必須實測，且所有對照基準均須獲配同等 Host DRAM 快取與熱圖 CPU 執行能力）！真正必須依賴 Flash 容量的區間為較大庫（如 **$C=32, r=128$：`DeepSeek` `13.96 decimal GB`、`OLMoE` `8.59 decimal GB`**；或 $C=32, r=256$：`27.92 GB`／`17.18 GB`）——而在該區間，不僅 `Flash-Jacobian` 的 $B_{e,c}$（同為 `13.96–27.92 GB`）確定爆出 `8 GB` Host DRAM 而被迫退回雙階段 Flash，每區域校準樣本稀釋問題亦同步加劇。

5. **撤回「`W8 r=H` 先驗鎖死」與簡化除數波次表；強制採用合法群組排布 $g$ 與等感測足跡之更高全局秩**：
   - 在未實測確認 `W4` 達到匹配品質（matched quality）前，`W8 r=H` 並未被先驗鎖死（單階段去屏障與減半命令數在真實排程中仍可能重要）；同時局部學生映射自身亦允許採用同位元數之 **`W4` 量化**。
   - Block 1 表格僅為理想頁數下界，非可達排程或端到端波次。在 $H=2048, \text{W8}, 16\text{-KiB page}$ 完整輸出列配置下：$r=64$（`8` 滿頁）可配 **`g=2` chips**（每 chip `4` 頁＝1 波，16 chips 具 8 組並行），$r=128$（`16` 頁）配 **`g=4`**，$r=256$（`32` 頁）配 **`g=8`**，$r=512$（`64` 頁）配 **`g=16`**。若將 $r=64$ 弱配於 `g=16`（每 chip `8 KiB` 半頁補零成 `16` 頁），則**全局秩 $r=128$（每 chip `16 KiB` 滿頁）具完全相同的單專家感測頁數**卻多一倍投影維度且省去 $(C-1)$ 倍儲存（見 Terra [research_2026_11_03_new_mechanism_sources.md](file:///home/ericyu/expert_flash_experiment/docs/research_2026_11_03_new_mechanism_sources.md#L32-L34)）！所有基準與候選皆須在合法群組、副本、衝突與命令成本下獲得最強排布與等足跡更高全局秩。

6. **誠實入帳 `FP32` $F_e(c)$ 記憶體，並明辨閉式解之統計極限**：
   - `FP32` 錨點輸出 $F_e(c)$ **無法**在共用 $A_{e,c}$ 單一 `INT8` row scale 下無損塞入額外兩列（會擠壓斜率動態範圍或需額外獨立偏置編碼、PE 縮放硬體與非整頁補零成本）；在驗證合法壓縮格式前，一律誠實按 Host `FP32` 預算入帳（`DeepSeek` $C=16$ 約 `218 MB`、$C=32$ 約 `436 MB`）。
   - 閉式求解僅消除了 **SGD 訓練欠收斂**（`2606.19379`），**並不能消除**樣本數不足之欠定（$N_{e,c} < r$）、共線性病態條件數或分佈偏移誤差。

---

### 2B. 新近強先例精確核實與技術邊界（`[已引先例]`）

1. **`Flash-Jacobian`（匿名 ACL 投稿，OpenReview `au5FGRgwcZ` §§3.1–3.2, App. C；依 Terra 已披露筆記 [research_2026_11_03_new_mechanism_sources.md](file:///home/ericyu/expert_flash_experiment/docs/research_2026_11_03_new_mechanism_sources.md#L48-L49) 引用，不繞過瀏覽器驗證）**：
   - 已對上下文 LLM 隱藏態做聚類，並在每區域擬合條件低秩 Ridge 算子 $J_k = A_k B_k$ 預測 $\hat{y} = \bar{y}_k + A_k B_k(x - \bar{x}_k)$（名稱中 `Flash` 指 GPU 快速擬合演算法，**非 NAND In-Flash 運算**）。其每區域存獨立輸入因子 $B_k$，未建立跨活躍 MoE 專家之全層共用投影 $P_l$。
   - **關鍵指標區分**：其 App. C Table 34 在 Qwen-3.5-4B L12 MLP 上報告 64 簇低秩 Ridge 的 `cos 0.096 / rel-L2 0.997` 係評估**微分方向導數（exact differential JVP $J(x)v$）保真度**，**並非**同一輸入下之條件 FFN 輸出近似誤差 $\|F(x)-\hat{y}(x)\|$，亦非端到端生成品質——故絕不可誤引為局部輸出近似失敗之直接證據；但其 App. C.35 指出簇數增加導致每簇訓練樣本枯竭，則是極真實的校準風險。
2. **`EBC-LLM`（IEEE IVMSP 2026；直接核對本地下載之作者公開源碼 [source_cells.txt](file:///home/ericyu/expert_flash_experiment/results/research-2026-11-03-no-cr/prior-sources/ebc-author-code/source_cells.txt#L34010-L35358)，完整論文正文仍不可得，絕不佯稱已讀全文）**：
   - 源碼第 `34010` 行 [`build_Ws`](file:///home/ericyu/expert_flash_experiment/results/research-2026-11-03-no-cr/prior-sources/ebc-author-code/source_cells.txt#L34010-L34131) 對每個專家擬合**單一全局 $H\times H$ Ridge 矩陣**；第 `35344–35358` 行對 **$E$ 個專家的線性化矩陣特徵**做聚類（靜態專家分群，非運行期上下文分區）；第 `34581` 行 [`apply_expert`](file:///home/ericyu/expert_flash_experiment/results/research-2026-11-03-no-cr/prior-sources/ebc-author-code/source_cells.txt#L34581-L34602) 取固定專家簇索引 `c = cluster_of_pos[pos]`，以專家簇共用正交旋轉基底 $U_c, V_c$ 搭配稀疏核心塊與低秩殘差執行推論。這直接確立了**「每專家全局閉式 Ridge＋跨專家共用基底旋轉」**之強控制基準。
3. **`Half the Nonlinearity Is Wasted` (`arXiv:2603.03459` §§3.1–3.2, 4.3.4, 5.8)**：
   - 以全局仿射矩陣替代完整 MLP，並以**下游 token cross-entropy 增量**訓練輕量級上下文門控決定是否回退原 MLP（極強之讀前守衛基準！）；其 `PCA-50, K=20` K-means 結果（AUC `0.519–0.611`）僅說明輸入聚類難以預測**單一全局線性映射**是否安全，並非對每簇獨立擬合局部映射庫之否定。
4. **`MoLAE` (`arXiv:2503.23100v2`)、`ITC-MoE` (`arXiv:2610.01296`) 與 `LAWS` (`arXiv:2605.04069`)**：
   - `MoLAE` 與 `ITC-MoE` 已建立跨專家共用輸入投影＋**非線性潛空間專家 $\phi_e(z)$**＋**群組共用輸出抬升（group-shared output lift）**——更關鍵的是，在 AiF 上 Host 可先將同群組選中專家的潛向量按路由權重聚合成 $\sum_{e \in \mathcal{G}} \alpha_e \phi_e(z)$，再對**每個活躍群組僅發起一次 Flash 輸出矩陣讀取**（少於每專家讀一次）！`LAWS` 則已提出快取錨點輸出＋Jacobian 一階修正＋有效半徑回退之概念框架。
   - **新穎性誠實定界**：局部 Jacobian／Ridge 庫、全層共用輸入投影 $P_l$、快取錨點＋切線＋回退皆**各自已有明確先例**。本候選不能宣稱上述單一積木為新發明。

---

### 2C. 核心科學問題收斂：在 `Host 8 GB + 實體群組波次` 聯合約束下，「運行期條件切換輸出子空間」能否擊敗「固定全局非線性輸出抬升」？

先承認一項不可迴避的**代數等價性（`[符號推導]`）**：
- 一個具有區塊稀疏 one-hot 潛狀態 $\tilde{\phi}_{e}(z, c) = [0, \dots, z - z_c, \dots, 0]^T \in \mathbb{R}^{Cr}$ 並在寬輸出矩陣 $[A_{e,1}, \dots, A_{e,C}] \in \mathbb{R}^{H \times Cr}$ 中按塊選列讀取的網路，與局部仿射映射庫 $A_{e,c}(z - z_c)$ **在數學上完全同構**——審查與實驗絕不可僅因命名不同而排除此等價基準。

因此，剝除所有包裝後，真正決定此方向生死的核心科學與系統問題只有一個：
- 在**真正需要 Flash 駐留的容量區間**（如 $C \in \{16, 32\}, r \in \{128, 256\}$，全庫 `7–28 decimal GB` 超出 `8 GB` Host DRAM；注意在此區間 `Flash-Jacobian` 的每區域輸入因子 $B_{e,c}$ 同樣高達 `7–28 GB` 而無法駐留 Host DRAM，唯有共用／專家級輸入投影能維持單階段 Flash 讀取！），當**固定每 token 主動實體群組波次（cohort physical waves）與相同 `8 GB` Host DRAM 預算**時：
  - **候選機制（運行期條件切換輸出子空間）**：以 Flash 容量換取每個 `(expert, region)` 各自獨立的秩 $r$ **輸出子空間** $\operatorname{col}(A_{e,c}) \subset \operatorname{col}(D_e)$（模擬 SwiGLU 門控在不同語義區域點亮 $D_e \in \mathbb{R}^{H \times I}$ 中不同子集之輸出列），並透過閉式 `J-shrunk Ridge`（融合無需樣本之平滑權重導數 $J_e(c)P_l^T$ 與局部條件 Ridge 截距／斜率）克服 $N_{e,c} \lesssim r$ 的區域樣本稀釋；
  - **最強同資源非銀行對手（固定／群組輸出子空間＋Host 非線性或更高秩）**：不買 $C$ 份 Flash 輸出矩陣，改用：(i) 黑臉提出的**同錨點截距＋單一全局斜率** $F_e(c) + A_e(z - z_c)$；(ii) **同錨點截距＋Host 小型非線性潛映射＋單一 Flash 輸出矩陣** $F_e(c) + A_e \phi_e(z - z_c)$（例如每專家 $r \to r$ 小型 SwiGLU 駐留 Host DRAM，在 $r=128$ `W8` 下全模型 `1,664` 個專家僅佔 **`81.8 decimal MB`**！）；(iii) **`MoLAE` 式群組聚合後單次輸出抬升**，或把省下的 Flash 容量與空閒 plane 用於**更高全局秩 $r_{\text{glob}} > r$**（含 `EBC-LLM`／`SPICE` 式 Host 駐留 $B_e$），以及 (iv) 原生 **`W4A8` 原專家＋同容量 Host 熱快取**。
- **精確可反證分離條件**：唯有當真實預訓練 SwiGLU 專家的局部條件輸出協方差 $\operatorname{Cov}(F_e(x) - F_e(c) \mid \mathcal{R}_{e,c})$ 具快速譜衰減（低局部秩 $r \ll I$），而跨區域邊際輸出殘差 $\operatorname{Cov}(F_e(x) - F_e(c(x)) \mid e)$ 具慢速譜衰減（輸出子空間隨上下文實質旋轉，單一秩 $r$ 或等波次秩 $r_{\text{glob}}$ 輸出基底 $A_e$ 無法覆蓋），**且** `J-shrunk` 閉式編譯能在真實離線校準預算下穩定估出該旋轉並傳遞為端到端自回歸生成品質優勢時，本候選方能成立！

---

### 2D. Block 2 當時之去留建議與四項鐵律（其中「逐專家輸入投遞／48 樣本佔有率」與「鐵律三要求 Stage-0 必跑全模型 PPL」已於下方 **Block 3** 修正更新，保留原記錄供稽核）

1. **坦誠評估：我們當前面臨的雙重風險**：
   - **新穎性收窄風險**：在 `Flash-Jacobian`、`LAWS`、`MoLAE`、`HalfTheNonlinearity`、`TARDIS` 與 `EBC-LLM` 齊備後，本方向已**不具備純演算法原語級新穎性**；若最終僅在小 $(C, r)$（如 $C=16, r=64$）靠全部塞進 Host `8 GB` CPU 記憶體獲勝，或僅贏過弱 `W8` 卻輸給 `F(c) + A_e \phi_e(z)`／更高全局秩／`W4A8`，按 [campaign.md](file:///home/ericyu/expert_flash_experiment/docs/research_2026_11_03_campaign.md#L24) 第 24 行只能列為「組合既有元件之工程實驗，未具投稿價值」。
   - **科學未知性**：與上一輪 tagged-pair（被純代數證明直接退化為同算子類之靜態單專家緊湊視圖）不同，目前沒有任何文獻或代數反例能先驗回答「真實預訓練門控 SwiGLU MoE 專家在固定主動實體波次下，究竟單靠固定輸出基底即可涵蓋，還是確有跨上下文之局部斜率／輸出子空間旋轉收益」。

2. **Block 2 原列條件與 Block 3 修訂對照**：
   - *(更新註記：① 先前關於「逐專家不同錨點需 $K$ 次獨立輸入投遞」與「每葉僅 `~48–88` 樣本」之前提，在下方 **Block 3 §3A** 改採「全層共用平衡遞迴 PCA／中位數樹＋離線全 `FIT` 狀態查詢」後已不再適用，每葉樣本數提升至約 `480` 且全層 $K$ 個選中專家共用**單一** `A8` 輸入向量；② Block 2 原「鐵律三」要求在 Stage-0 必要條件篩選階段即強制跑完所有全模型 PPL 與全部非線性／`W4` 先例，經 **Block 3 §3D** 嚴格區分「必要條件否證 vs 充分條件成功」後予以撤回修正：若候選在 8 個凍結算子上已被最小控制集 `{R0, R1, R2}` 完全支配，即足以構成該定義方法族之有界科學負結果；唯有出現非支配正向訊號時，才必須於同一輪擴展至全模型 free-running 品質與完整強基準集。)*

---

## Block 3：全層共用平衡分割精煉審查、離線非路由擬合偏差、量化設計矩陣、最小 Stage-0 辨識邏輯與真實投稿價值裁決

- **時鐘與邊界**：同一正式候選時鐘（`2026-11-02 21:26:00 UTC` 起，最早 `21:56:00 UTC`，須經 Root 與黑臉審查凍結協定後方可決定是否實驗；本階段零實驗碼、零模型／激活探測、零 terminal）。

### 3A. 對舊前提之明確更新與共用分割機制核驗（`[已引先例]` + `[符號推導]`）

1. **更新一：由「逐專家錨點 $K$ 次輸入投遞」更新為「全層單一共用中心化 `A8` 輸入廣播」**：
   - 在 Block 1 §1C.4 與 Block 2 §2A.3 中，因假設每個專家各自擁有獨立錨點 $c_{e,m}$，導致 $\delta z_{e, c(e)}$ 隨專家而異、需支付 $K$ 次獨立輸入向量投遞成本。
   - 改採 Root 精煉方案——**每層單一 `FIT`-only 平衡遞迴 PCA／中位數樹**（葉數 $C \in \{1, 4, 16, 32\}$，全層共用葉均值 $\bar{x}_{\text{leaf}}$ 與共用嵌套 PCA 投影 $P_l \in \mathbb{R}^{r \times H}$）後：對任意輸入 $x$，Host 先經深度 $\log_2 C \le 5$ 次分裂超平面內積定位唯一葉節點 `leaf`（取代原本 $K \times C = 192$ 次最近鄰距離計算，且全模型共用葉均值與樹節點在 `DeepSeek` 26 層 `FP32` 下分別佔約 `6.82 decimal MB`（32 葉均值）+ `6.60 decimal MB`（31 內部節點法向量），合計約 `13.42 decimal MB` 另加切分閾值，較逐專家中心省 `64×` Host 鍵值空間；惟逐專家葉截距 $\bar{y}_{e,\text{leaf}}$ 在 $C=32$ `FP32` 下仍佔約 `436.2 decimal MB`，須連同 row-scales、`P_l`、非 FFN 權重與 KV 一併誠實計入 `8,000,000,000 B` 與 `8 GiB` 雙重 Host 帳本 *［詳見 Block 4 §4A.6 單位修正：`OLMoE` `C32/r128` 裸 `W8` 圖庫為 `8.590 decimal GB = 8.000 GiB` 整，加上附屬結構後方逾 `8 GiB`］*），僅計算**一次**中心化投影向量 $\delta = P_l(x - \bar{x}_{\text{leaf}}) \in \mathbb{R}^r$ 並動態量化為**單一 `A8` 向量** $q_\delta = Q_{\text{A8}}(\delta)$，即可**由同層全部 $K$ 個選中專家完全共用**（恢復與原生 AiF `GU` 階段一致之單一輸入廣播語義；惟低維向量 $r < H$ 之最小硬體寬度／補零線路契約仍須按 §2A.3 明示入帳，且 Stage-0 維持 `DeepSeek` 原生 Shared Expert 不變，其 `GU->Down` 屏障留待 Stage-1 DAG 評估）。
2. **更新二：由「自然路由稀釋（`~48` 樣本／葉）」更新為「離線全狀態查詢（條件式名義 `~480` prefill 樣本／葉，且跨葉合併改善 `R2` 樣本數）」**：
   - 先前 `~48–88` 樣本／葉之估計係假設僅在自然路由命中專家 $e$ 的子集上擬合。若在離線編譯期對 8 個試點專家**查詢全部 `FIT` 層輸入狀態**（`200` 題 `TRAIN` 確定性切分為 `120 FIT / 40 VALID / 40 SCORE`，另保留 `100 TEST` 供同輪正向擴展驗證；`120` 題 `FIT` × 前 `min(128, N)` prefill tokens 加 `8` 步 decode 名義上限為 `15,360 + 960 = 16,320` 個狀態，**實作前必須審計短題與完全相同輸入向量 $x$ 無法按 sample ID 拆葉後之真實葉佔有數，且須注意名義 $N_{\text{leaf}} \ge 2(r+1)$ 不自動保證低共線性或良態條件數** *［詳見 Block 4 §4A.2–4A.3 修正］*），則在無大量重複 $x$ 時 $C=32$ 下每個專家在每一葉約有 `~480–510` 個擬合狀態，對主區間 $r = 128$（全庫 `DeepSeek` `13.96 decimal GB / 13.00 GiB`、`OLMoE` `8.59 decimal GB / 8.00 GiB` 裸 `W8`，另設次要區間 $r=64$）改善了樣本維度比；同時跨 32 葉合併後總樣本達 `~15,360–16,320`，亦改善更高全局秩控制 `R2`（$r_{\text{glob}} \in \{512, H=2048\}$）之樣本維度比。
   - **統計誠實揭露契約**：此狀態集來自 `120` 個獨立 prompts（同題內 tokens 具序列相關性，有效獨立樣本數遠小於總列數，不可冒充為 i.i.d. 樣本，必須採 **prompt 層級配對 bootstrap** 與確定性 `前 2/3 (80) vs 全部 (120) FIT prompts` 學習曲線檢驗，並揭露無監督樹與 $P_l$ 共用全 `FIT`），且狀態擷取自既有 GGUF 推論軌跡（off-policy 狀態擷取須明確揭露），所有控制組 `R0/R1/R2` 均獲完全相同的全狀態離線查詢與查詢成本記錄。主方法採**向同葉去均值池化共用斜率收縮之閉式局部 Ridge（`shrink-to-pooled-slope ridge`，$\lambda \in \{0, \dots, \infty\}$ 由自然路由 `VALID` 選取，且 $\lambda = \infty$ 降階圖必須實體去重為單一圖）**。
3. **平衡遞迴 PCA／中位數樹之先例定性（絕不宣稱分群新穎性）**：
   - `[已引先例]`：遞迴主方向二元分割搭配葉內局部線性迴歸即經典 **Principal Direction Divisive Partitioning（`PDDP`, Boley 1998）**／**PCA-tree（Sproull 1991; Verma, Kpotufe & Dasgupta 2009）** 與 **線性模型樹（`M5`, Quinlan 1992）**。此元件在協定與論文中僅能作為「支援 $O(\log C)$ 單次查找之標準確定性空間分割工具」，**零聚類演算法新穎性主張**。

---

### 3B. 離線全狀態擬合之「非路由分佈偏移（Off-Route Fit Bias）」與重加權控制推導（`[符號推導]`）

離線對全部狀態查詢專家 $e$ 雖提升了樣本數，但在評估期（`SCORE`）專家 $e$ **僅在自然路由選中子集 $\mathcal{S}_{e,\ell} = \{x \in \ell : e \in \text{top-}K(x)\}$ 上計分**！這引入兩項潛在機制：

1. **非線性死區斜率稀釋（Gate Regime Mismatch）**：
   - 若某葉節點 $\ell$ 中僅有少數（如 `5%–10%`）tokens 自然路由至專家 $e$，其餘為非路由 tokens；只要門控權重 $G_e$ 與路由向量 $w_{\text{router},e}$ 存在正相關，非路由 tokens 可能較常落在 SwiGLU 門控抑制區，使等權重擬合之 $A_{e,\ell}^{\text{all}}$ 偏離真實路由子集 $\mathcal{S}_{e,\ell}$ 上之活躍局部斜率。
2. **投影零空間 $\ker(P_l)$ 之路由條件均值偏移可能性（Nullspace Selection Shift）**：
   - *［Block 4 §4A.2 強制修正：撤回原稿「必發生非零偏移」之過度宣稱！一般競爭式 top-$K$ 不等式或對稱分佈下，即便 $w_{e,\perp} \neq 0$，條件均值仍可能嚴格為零；僅在簡化橢球／單一半空間假設下可能產生偏移。同時須嚴格區分三種葉均值：未加權原始 `FIT` 葉中心化輸入均值恆為 `0`，但路由加權葉輸入均值與實際動態 `A8` 反量化設計均值 $\bar{v}_\ell$ 一般皆不為 `0`。］*
   - 令 $x - \bar{x}_\ell = P_l^T \delta + x_\perp$（$x_\perp \in \ker(P_l)$）。在未加權全狀態葉分佈下 $\frac{1}{N_\ell}\sum_{n \in \ell} x_{\perp,n} = \mathbf{0}$；但在簡化橢球分佈與單一線性門檻假設下，條件化於 $e \in \text{top-}K(x)$ **可能**使 $\mu_{\perp, e, \ell} \triangleq \mathbb{E}[x_\perp \mid x \in \ell, e \in \text{top-}K(x)] \neq \mathbf{0}$，進而在截距中留下 $J_{e,\perp}(\bar{x}_\ell)\,\mu_{\perp, e, \ell}$ 一階偏移。
3. **零額外推論成本之有界路由重加權控制（`Bounded-Floor Route Weighting`）**：
   - *［Block 4 §4A.4 強制修正：撤回原稿 $\gamma \in \{0.05, 0.0\}$ 近零／純路由權重網格！將非路由權重設為 `0` 或 `0.05*(K/E)` 會使 Kish 有效樣本數 $N_{\text{eff}} = (\sum w)^2 / \sum w^2$ 崩潰回 `~45` 樣本／葉之飢餓區而引發欠定假陰性。改採 `draft-003` 之預註冊雙權重配置：主配置等權重 $w=1$，次配置有界底線權重 `w=1 (routed) / w=0.25 (off-route)`，詳見 §4A.4。］*
   - 在離線閉式擬合時，對所有方法族（`R0, R1, R2` 與候選）平等提供**全等權重主配置（$w=1$）**與**有界路由重加權次配置（自然路由列 $w=1$，非路由列 $w=1/4$）**，並強制報告每葉 $\sum w$、Kish 有效樣本數 $N_{\text{eff}}$、有效自由度、數值秩／條件數與獨立 prompt 數。

---

### 3C. 量化設計矩陣迴歸（Quantized-Design Ridge）vs 浮點擬合，以及共用分割幾何風險（`[符號推導]`）

1. **精確量化協變數迴歸與實際部署態 `F32` 截距校準**：
   - 部署時進入 Flash PE 的並非浮點 $\delta = P_l(x - \bar{x}_\ell)$，而是動態 `A8` 反量化值 $v(x) = s_\delta(x)\,q_\delta(x)$（其中 $s_\delta(x) = \|\delta(x)\|_\infty / 127$）。若在浮點 $\delta$ 上做 OLS 再直接餵入 $v(x)$，會產生自變數含噪偏差且葉內 $\bar{v}_\ell \neq 0$。
   - **標準閉式流程** *［Block 4 §4A.5 修正對齊 `draft-003`］*：(i) 直接以部署態 `A8` 協變數 $v(x_n)$ 與 canonical `W8A8` 原專家目標輸出 $y_{n,e}^{\text{canon}}$ 建立葉內加權中心化設計矩陣；(ii) 先跨全葉合併葉內去均值充分統計量，獨立由 `VALID` 在 $\eta$ 網格上選取最優池化共用斜率 $\hat{A}_e^{\text{pooled}}$（**絕不可先擬合過原點全局斜率再事後補葉均值**）；(iii) 再以正規化局部懲罰 $\lambda \frac{\operatorname{tr}(\Sigma_\ell)}{r} \|A_{e,\ell} - \hat{A}_e^{\text{pooled}}\|_F^2$ 解閉式局部斜率（$\lambda = \infty$ 退化為池化共用斜率並實體去重為單一圖，且同時保留最佳有限 $\lambda < \infty$ 點以防空洞負結果）；(iv) 將擬合矩陣量化為部署態 `W8` 後，**直接由實際 `I64/F32` 縮放整數內積預測值**按凍結歸約順序計算加權均值殘差 `bias = weighted_mean_FIT(target - deployed_pred)` 作為最終 `F32` 葉截距（精確吸收權重量化與逐樣本 `F32` 乘法捨入均值偏移），並平等賦予所有控制組。
2. **Host 座標方差均衡／白化與各向同性 Ridge 正則化之非不變性（純量化衛生控制，非新穎點）**：
   - 在原始 PCA 座標下，前幾個主成分方差遠大於尾端主成分，會主導動態 `amax` 使尾端維度在 `A8` 中被壓縮。若在 Host 將正交均衡旋轉 $Q_r$（如確定性帶號 Walsh-Hadamard）預先乘入 $P_l' = Q_r P_l$（$r \le 512$ 推論零額外算力，且因 $\|A Q_r^T\|_F = \|A\|_F$ **嚴格保持各向同性 Ridge 懲罰不變**），可同時均衡 `A8` 輸入與 `W8` 權重列動態範圍；反之對角縮放（如 `draft-003` 預註冊之 `σ-floor 1/16`：除以 $\max(\sigma_j, \sigma_1/16)$）會改變各向同性 Ridge 先驗並將高達 $16\times$ 的尺度差轉移至權重列，故必須如 `draft-003` 第 60 行明確認定其為**不同的預註冊正則化配置**（各自重新量化 `A8` 設計並重擬合池化先驗，嚴禁事後跨格式拼湊），詳見 **Block 4 §4B 問題 A**。
3. **全層共用樹之節點局部幾何特性與盲區（Unmeasured Structural Risk）**：
   - *［Block 4 §4A.3 強制修正：撤回原稿「僅沿層輸入全局方差最大的 5 個方向切分」之錯誤幾何敘述！$C=32$ 二元樹共有 $31$ 個內部節點，各自在節點條件子集上計算局部第一主方向，跨分支最多可張成 $31$ 個不同全局方向，而非僅前 5 個全局主成分！］*
   - 每條根至葉路徑由 **5 個節點局部主方向半空間不等式**界定（全樹共 `31` 個局部主方向法向量）。由於切分方向完全由無監督層輸入方差決定而非針對個別專家門控矩陣 $G_e$ 監督式切分，若某專家 $e$ 的門控切換超平面與該節點輸入主方差方向正交，葉分割對該專家曲率之解析度仍可能受限（使其局部斜率趨近 $\hat{A}_e^{\text{pooled}}$）。這是共用單一樹換取 $O(\log C)$ 查找與單一廣播輸入之真實結構取捨。

---

### 3D. 最小 Stage-0 辨識邏輯審定（撤回 Block 2 過度要求，並嚴格限定 8 算子負結果之推論邊界）

針對 Block 2「鐵律三」與黑臉之分歧，經嚴格邏輯檢驗，**非對稱判準是正確的，惟必須嚴格刪除任何由局部算子 MSE 直接外推全模型 PPL 之過度宣稱**：

1. **最小控制子集 `{R0, R1, R2}` 在 8 個算子上之支配究竟證明什麼、不證明什麼？**
   - *［Block 4 §4A.1 強制修正：撤回原稿「絕無可能在全模型推論中憑空翻盤」之錯誤外推！集合單調性僅在「同一損失函數、同一組算子、同一輸入分佈」下成立，絕不跨指標推導全模型 PPL／生成品質！］*
   - 固定相同葉分割、葉均值 $\bar{x}_\ell$ 與自由葉截距後，Stage-0 測試的是必要條件命題：**「在所抽樣的 8 個凍結算子與自然路由分佈上，隨葉切換的局部斜率 $A_{e,\ell}$（$\lambda < \infty$）能否在量化算子平方誤差—主動工作量前緣上超越葉常數 `R0`、同葉去均值最優池化斜率 `R1`、以及同主動成本之更高全局秩 `R2`（$r_{\text{glob}} \in \{512, H\}$）？」**
   - 在**同一 8 算子、同一自然路由輸入分佈與同一平方誤差指標**下，由集合單調性 $\mathcal{B}_{\text{full}} \supset \{R0, R1, R2\}$ 可知：若所有由 `VALID` 選出的有限 $\lambda$ 局部候選點均被不具更高相關成本之 `{R0, R1, R2}` 在統計信賴支撐下支配（或精確打平），則在**該凍結候選族與該算子損失定義下**構成乾淨的**有界科學負結果（Bounded Scientific Negative）**，具備充分科學理由終止此具體試點並保留剩餘實驗配額——但這**不是**全模型 PPL 之不可能性定理（因跨專家帶號誤差 $\sum_k \alpha_k \varepsilon_{e_k}$ 可相消或沿下游敏感方向對齊，且未抽樣深層與自回歸閉環軌跡不同）。同時，**信賴區間跨零（無顯著性）、僅排除任意 `2%–5%` 改善幅度、decode 命中稀疏或學習曲線未收斂，一律屬於「不確定（Inconclusive）」，必須留在同一輪（Round 2）續查，絕不計為科學失敗**！
   - 反之，**Stage-0 局部非支配絕不等於方法成功（Pilot Positive NEVER Success）**：任何可信非支配點均**必須在同一輪（Round 2）擴展至 Stage-1**（全層替換、真實 free-running 品質、保留之 `100 TEST`、直接共同輸出子空間控制 `U_q B_c`／`MoLAE`、Host 投影 `GU`+`Down`、`SPICE`、`W4`、熱圖快取與完整實體排程 DAG）。
2. **Stage-0 內建之兩項邊界保護（已納入 `draft-003`）**：
   - **(a) 讀前距離守衛分位曲線（`50%, 70%, 85%, 100%`）**：以讀前可算之 `||x - meanX_leaf||_2` 在自然路由 `VALID` 上凍結分位門檻，對所有控制組一視同仁報告覆蓋率、覆蓋內誤差、含原專家精確 fallback 之混合總誤差與混合雙階段實體工作量（詳見 **Block 4 §4B 問題 B**）。
   - **(b) 深度與軌跡範圍明示**：誠實標註試點涵蓋 `DeepSeek` L1, L13（26 層中淺／中層）與 `OLMoE` L0, L8（16 層中淺／中層）及原始 GGUF 輸入軌跡，不作跨深度或全模型品質外推。

---

### 3E. 真實投稿價值坦誠裁決（Candid Value Verdict）

- **坦誠判斷：演算法層面無新原語；唯一潛在價值在於「受限 Host DRAM 與關閉 CR-read 下之全層共用單一廣播輸入 In-Flash 編譯部署契約」，且須通過嚴格的多階段必要與充分條件檢驗**。
  - **為何不能走演算法創新敘事**：局部條件低秩 Ridge（`Flash-Jacobian`）、共用投影潛空間（`MoLAE`）、平衡 PCA 模型樹（`PDDP`/`M5`）、Host-Flash 切分（`AiF`）均為既有技術。
  - **唯一站得住腳的可投稿系統／架構命題** *［Block 4 §4B 問題 C 修正：刪除「`Flash-Jacobian` 必然被迫落入兩階段 Flash」與「`100%` 取決於單一未知」之過度宣稱］*：在關閉 `CR-read` 的邊緣 In-Flash MoE 中，本設計測試能否利用 SSD 靜態普通讀取容量（裸 `W8` 圖庫：`DeepSeek` `13.96 decimal GB / 13.00 GiB`、`OLMoE` `8.59 decimal GB / 8.00 GiB`）配合全層共用樹鍵值達成 **$O(\log C)$ 路由前查找與全層單一中心化 `A8` 輸入廣播**。
  - **修正後之條件式價值判斷**：在 Stage-0 顯著擊敗 `{R0, R1, R2}` 僅是**第一道必要門檻（Necessary Gate）**，而非充分條件；即便通過 Stage-0，仍必須在同一輪 Stage-1 證明其在相同的「Host DRAM（`8e9 B` 與 `8 GiB` 雙帳本）+ Flash 容量 + 實測 CPU／Flash 排程」聯合預算下，能實質勝過直接共同輸出基底分解 `A_{e,c} ≈ U_{e,q} B_{e,c}`（`MoLAE`-like Host 條件潛變換）、`Flash-Jacobian` 最佳合法 Host 快取／流式配置、`W4A8` 原專家與全模型 free-running 品質基準。

---

## Block 4：收官交叉審查、Block 3 六項強制修正、`draft-003` 四大設計問題裁決與最終協定條件簽核

- **時鐘與閘門狀態**：同一正式候選時鐘（`2026-11-02 21:26:00 UTC` 起，最早實驗資格時間 `21:56:00 UTC`；惟時間到達絕不自動授權執行，實作與合成算術檢查須待 Root＋黑臉完成思考與協定凍結，真實模型／語料擷取另須通過不可變原始碼／建置凍結與 Root 原始碼審查收據）。本階段零實驗碼、零模型／激活探測、零 terminal。
- **審查對象**：[stage0_protocol.draft.json](file:///home/ericyu/expert_flash_experiment/results/research-2026-11-03-no-cr/round02-local-affine/stage0_protocol.draft.json)（`revision: draft-003-before-data`）與 [research_2026_11_03_new_mechanism_sol.md](file:///home/ericyu/expert_flash_experiment/docs/research_2026_11_03_new_mechanism_sol.md)。以下嚴格區分 **`[已證數學／IEEE-754 事實 Proven]`**、**`[已引先例 Cited Prior]`** 與 **`[未實測假設 Unmeasured]`**。

---

### 4A. Block 3 六項強制修正逐條核實（Mandatory Corrections Audit）

1. **修正一（撤回局部算子 MSE 對全模型推論之絕對外推；嚴格界定有界負結果 vs 不確定續查）**：
   - **`[已證數學事實 Proven]`**：基準集合單調性 $\min_{b \in \mathcal{B}_{\text{full}}} \mathcal{L}(b) \le \min_{b \in \mathcal{B}_{\text{sub}}} \mathcal{L}(b)$ **僅在完全相同的損失泛函 $\mathcal{L}$、相同的 8 個算子與相同的輸入分佈下成立**。在 MoE 層輸出聚合 $y_{\text{layer}}(x) = y_{\text{shared}}(x) + \sum_{k=1}^K \alpha_{e_k}(x)\,\hat{F}_{e_k}(x)$ 中，各專家帶號誤差 $\varepsilon_{e_k}(x)$ 之間存在內積項 $2\sum_{j<k}\alpha_{e_j}\alpha_{e_k}\langle \varepsilon_{e_j}, \varepsilon_{e_k}\rangle$，且下游殘差流與 Logit 梯度方向具有高度各向異性（`[已引先例 Cited Prior]`：`2606.19379` 實證局部 $R^2$ 與全局 PPL 可嚴重解離），外加未抽樣深層（如 `DeepSeek` L14–L25、`OLMoE` L9–L15）與自回歸閉環路由軌跡偏移，故 **8 算子局部 MSE 被支配絕不構成全模型 PPL 不可能改善之定理**。
   - **正確判準（對齊 `draft-003` 第 93–94 行）**：若所有由 `VALID` 選出的有限 $\lambda$ 局部候選點均在同時信賴帶支撐下被不具更高相關成本之 `{R0, R1, R2}` 支配（或精確位元打平），其結論嚴格限於**「該凍結候選方法族於所抽樣 8 算子與自然路由分佈上之有界科學負結果」**（足以合理終止本具體試點，不作全局外推）。反之，信賴區間跨零（無顯著性）、僅排除任意 `2%–5%` 誤差差距、decode 命中不足或學習曲線未穩定，**一律歸為 `Inconclusive` 並於同一輪（Round 2）續查**；任何可信非支配點則必須進入同輪 Stage-1 擴展，絕不直接宣稱成功。

2. **修正二（撤回 $w_{\text{router},\perp} \neq 0 \Rightarrow \mu_{\perp,e,\ell} \neq 0$ 之必然蘊含；區分三種葉均值與條件數風險）**：
   - **`[已證數學反例 Proven]`**：設葉內 $x \sim \mathcal{N}(0, I_2)$、$P_l = [1, 0]$（故 $\ker(P_l)$ 座標為 $x_2$），考慮 3 個專家路由向量 $w_1 = (0, 1),\, w_2 = (0, 2),\, w_3 = (0, -2)$ 與 top-$K = 2$：專家 $e=1$ 雖滿足 $w_{1,\perp} = 1 \neq 0$，但對一切 $x_2 > 0$ 有 $2x_2 > x_2 > -2x_2$（$e=1$ 入選 top-2），對一切 $x_2 < 0$ 有 $-2x_2 > x_2 > 2x_2$（$e=1$ 仍入選 top-2！），故對所有 $x_2 \neq 0$ 恆有 $e=1 \in \text{top-2}(x)$，從而條件均值 $\mathbb{E}[x_2 \mid e=1 \in \text{top-2}(x)] = 0$！因此 $w_{e,\perp} \neq 0$ **一般並不蘊含**路由條件零空間均值非零；僅能表述為「在簡化橢球／單一半空間假設下可能產生偏移（`[未實測假設 Unmeasured]`）」。
   - **三種均值與條件數區分（`[已證數學事實 Proven]`）**：(i) 未加權原始 `FIT` 葉中心化輸入均值 $\frac{1}{N_\ell}\sum_{n\in\ell}(x_n - \bar{x}_\ell) = \mathbf{0}$；(ii) 路由加權葉輸入均值一般 $\neq \mathbf{0}$；(iii) 實際動態 `A8` 反量化設計向量均值 $\bar{v}_\ell = \text{mean}_\ell(s_\delta(x)q_\delta(x))$ 因動態 `amax` 與四捨五入對偏態座標之非線性作用，即使未加權一般亦 $\neq \mathbf{0}$。此外，離線全狀態查詢滿足名義 $N_{\text{leaf}} \ge 2(r+1)$ 僅為樣本數啟發式，**不自動保證良態條件數**（若葉內狀態落於低維流形或重複前綴，協方差仍可病態，故必須依賴 `draft-003` 第 65、77 行之實際數值秩／條件數審計與 `1e-10` 相對特徵值截斷最小範數解）。

3. **修正三（更正 $C=32$ 節點局部 PCA 樹之 `31` 法向量幾何與純 $x$ 函數重複值限制）**：
   - **`[已證數學事實 Proven]`**：$C=32$ 葉之二元樹雖在任一單一推理路徑上深度為 $\log_2 32 = 5$，但全樹擁有 $C - 1 = \mathbf{31}$ **個內部節點**，每個內部節點皆在其**節點條件樣本子集**上計算第一主方向；因不同分支子集之局部協方差主軸彼此旋轉，這 31 個節點法向量在 $\mathbb{R}^H$ 中最多可張成 **31 個線性獨立方向**（絕非 Block 3 原稿誤稱之「全局前 5 主成分」）。其真正的未實測幾何風險（`[未實測假設 Unmeasured]`）在於：每個葉胞僅由 5 個由**無監督層輸入方差**決定的半空間交集界定，未必對齊特定專家 $G_e$ 之門控切換方向。
   - **`[已證確定性約束 Proven]`**：執行期選葉器 $c(x)$ 僅能依賴輸入向量 $x$（推理時不存在訓練樣本 ID！）。若多個樣本在某層（特別是淺層 `L0`/`L1` 之 `BOS` 或共同前綴 tokens）具有**完全相同的輸入向量 $x$**，任何合法執行期規則必將其分入同一葉節點；外加短於 `128` tokens 之題目會減少總列數，故名義 `~480`（含 8 步 decode 約 `~510`）樣本／葉屬**條件式而非保證值**，必須如 `draft-003` 第 57 行在 `FIT` 上重跑實際執行期選葉器並報告真實各葉樣本數。

4. **修正四（以 `1` vs `1/4` 有界路由權重取代近零權重網格，確保 Kish 有效樣本數不崩潰）**：
   - **`[已證數學事實 Proven]`**：在每葉自然路由率為 $p \in [0, 1]$ 下，設路由列權重為 $1$、非路由列權重為 $w_0 = 1/4$（`draft-003` 第 65 行次配置），則單葉 Kish 有效樣本比例為：
     $$\frac{N_{\text{eff}}(p)}{N_\ell} = \frac{\bigl(p + \tfrac{1}{4}(1-p)\bigr)^2}{p + \tfrac{1}{16}(1-p)} = \frac{(0.25 + 0.75\,p)^2}{0.0625 + 0.9375\,p}.$$
     在典型路由率 $p = 6/64 = 0.09375$（`DeepSeek`）與 $p = 8/64 = 0.125$（`OLMoE`）下，$N_{\text{eff}}/N_\ell$ 分別高達 **`0.682`（約 `327/480`）** 與 **`0.656`（約 `315/480`）**，即使在極端 $p = 0$ 之零自然命中葉亦恆保 $N_{\text{eff}} = N_\ell$（徹底避免原稿 $\gamma \in \{0.05, 0.0\}$ 使 $N_{\text{eff}}$ 暴跌至 $\le 45 < r$ 之欠定崩潰！），同時將路由樣本在加權損失中的總權重佔比從 `9.4% / 12.5%` 提升至 **`29.3% / 36.4%`**。配合報告 $\sum w$、Kish $N_{\text{eff}}$、有效自由度、條件數、每葉獨立 prompt 數，以及確定性 `前 2/3 (80) vs 全部 (120) FIT prompts` 係數學習曲線（明確揭露無監督樹與 $P_l$ 使用全 `FIT`），統計衛生完全閉環。

5. **修正五（由實際部署態縮放整數預測重校準 `F32` 葉截距，並強制去重 $\lambda=\infty$ 等價圖）**：
   - **`[已證算術事實 Proven]`**：部署態單列輸出為 `pred_i(x) = (float)dot_i * (sw_i * sx)`，其逐樣本 `F32` 捨入發生在加截距之前；直接按凍結歸約順序以 `bias_i = weighted_mean_FIT(target_i - deployed_pred_i)` 設定 `F32` 截距（`draft-003` 第 78 行），較實數公式 $\bar{y} - \hat{A}^{\text{deq}}\bar{v}$ 更精確地同時消除 `W8` 量化偏移與逐樣本 `F32` 縮放捨入均值誤差。同時，當 $\lambda = \infty$（或局部修正為零）時，$C$ 個葉降低後之 `W8` 權重矩陣位元級完全相同，**必須按 `draft-003` 第 102 行實體去重為單一儲存矩陣（絕不虛報 $C$ 份儲存成本）**，且 `R0` 與全零矩陣發送 **`0` 次 NAND 讀取**。

6. **修正六（嚴格區分 `8,000,000,000 B` 與 `8 GiB`、補齊樹節點與葉均值雙重元數據，並明示 Shared Expert 維持不變）**：
   - **`[已證位元組核算 Proven]`**：
     - `OLMoE`（$16 \times 64 = 1024$ 專家，$H=2048$）於 `C=32, r=128` 之裸 `W8` 矩陣庫為 $1024 \times 32 \times 2048 \times 128 = 8,589,934,592\text{ B} =$ **`8.590 decimal GB` = 精確 `8.000 GiB`**（按十進位 `8,000,000,000 B` 解釋已超標 `589.9 MB`；按二進位 `8 GiB` 解釋則裸圖庫恰等於上限，加計每圖 `F32` 截距 `268.4 MB`、每列 `F32` scale `268.4 MB`、`P_l` `17.0 MB`、樹元數據 `8.3 MB`、非 FFN 權重與 KV 後方超過 `8 GiB`）。故 `draft-003` 第 101 行同時審計 **`8,000,000,000 B` 與 `8 GiB` 雙重解釋**完全必要。
     - `DeepSeek` 26 層 `C=32, H=2048` 之全層共用樹元數據包含兩部分：`31` 個內部節點 `F32` 法向量佔 $26 \times 31 \times 2048 \times 4 = \mathbf{6.603\text{ decimal MB}}$，`32` 個葉 `F32` `meanX` 佔 $26 \times 32 \times 2048 \times 4 = \mathbf{6.816\text{ decimal MB}}$，合計 **`13.42 decimal MB`**（另加切分閾值；單一 token 每層僅走訪 `5` 個節點法向量與 `1` 個葉均值），外加 `r=128` 之 `P_l` 佔 **`27.26 decimal MB`**。
     - `DeepSeek` 原生 Shared Expert（$I_s = 2816$）於 Stage-0 維持雙階段原樣不變（`draft-003` 第 105 行），故即使路由專家折疊為單階段，該層仍保留 Shared `GU->Down` 相依路徑（除非於 Stage-1 納入 Host 快取或重疊排程 DAG）；`OLMoE` 無 Shared Expert，僅在當層全部 $K=8$ 個選中專家皆未觸發 fallback 時完全消除雙階段屏障。

---

### 4B. 對 `draft-003` 四大開放設計問題（Questions A–D）之具體分析與裁決建議

#### 問題 A：座標均衡配置（`Normal PCA` vs `σ-floor 1/16` vs 正交帶號 Hadamard 旋轉）
1. **三種座標之精確代數性質對比（`[已證數學事實 Proven]` + `[已引先例 Cited Prior]`）**：
   - **`Normal PCA`（$P_l$ 列正交）**：保持標準歐氏幾何與各向同性 Ridge 先驗，但因前數個奇異值 $\sigma_1 \gg \sigma_r$，少數大方差主成分主導動態 `A8` `amax`，壓縮尾端座標有效位元。
   - **`draft-003` 第 60 行之 `Variance-Balanced PCA`（除以 $d_j = \max(\sigma_j, \sigma_1/16)$）**：緩解了輸入端 `A8` 尾端壓縮，但因精確乘積關係 $A_{\text{bal}} \delta_{\text{bal}} = A_{\text{raw}} \delta_{\text{raw}} \iff A_{\text{bal}, :, j} = d_j A_{\text{raw}, :, j}$，會將最高 $16\times$ 的尺度比反向乘入權重矩陣前段列（可能放大每列 `W8` `row-amax` 量化步長），且各向同性懲罰 $\|A_{\text{bal}} - A_{\text{pooled,bal}}\|_F^2 = \sum_{j=1}^r d_j^2 \|\Delta a_{\text{raw}, :, j}\|_2^2$ 實質變為**對高方差主成分重罰最高 $256\times$ 之非各向同性先驗**（`draft-003` 第 60 行已誠實揭露此為不同正則化器、各自獨立重擬合 `A8` 設計與池化先驗，邏輯自洽無欺瞞）。
   - **正交確定性帶號 Walsh-Hadamard 旋轉（$P_l^{\text{had}} = \frac{1}{\sqrt{r}} H_r D_{\text{sign}} P_l$，`QuIP`/`QuaRot` `[已引先例]`）**：因本實驗所有維度 $r \in \{64, 128, 512, 2048\}$ **皆為 2 的整數次冪**，$Q_r = \frac{1}{\sqrt{r}} H_r D_{\text{sign}}$ 嚴格正交（$Q_r^T Q_r = I_r$）：(i) **100% 保持原各向同性 Ridge 先驗不變**（$\|\Delta A Q_r^T\|_F^2 = \|\Delta A\|_F^2$）；(ii) 同時使輸入座標二階矩 $\mathbb{E}[(Q_r \delta)_j^2] = \frac{1}{r}\operatorname{tr}(\Sigma_\delta)$ 與權重列能量同步均勻化（**不會把 $16\times$ 動態範圍轉嫁給 `W8` 權重列**）；(iii) $r \le 512$ 時離線預乘入 $P_l^{\text{had}}$ **推論零額外成本**，$r = H = 2048$ 時可用快速 `FWHT` 以 $H \log_2 H = 22,528$ 次加減完成（無需稠密 $H^2$ 矩陣乘）。
2. **具體建議（避免膨脹為三格式怪獸網格）**：
   - **絕不新增第 3 種格式造成多重比較爆炸**。我們支持兩種乾淨鎖定方案二選一（於開數據前敲定）：
     - **方案 A1（零修訂直接沿用 `draft-003` 第 60 行）**：維持 `normal PCA` + `σ-floor 1/16` 雙配置完全不變（因 `draft-003` 已正確聲明其為不同正則化器並各自重擬合池化先驗），僅需在 source-freeze 中明確規定：**由自然路由 `VALID` 為每個 `(算子, C, 權重)` 預先選出單一格式進入 `SCORE` 主比較（同時保留雙格式完整診斷表）**，將正式同時比較數減半以維持 `max-|t|` 檢定力。
     - **方案 A2（若 Root 偏好嚴格保持各向同性 Ridge 不變，做 1 對 1 預註冊替換）**：在凍結前將 `σ-floor 1/16` 1 對 1 替換為固定種子 `Frozen Signed Hadamard PCA`（$r \in \{64, 128, 512\}$ 預乘入 $P_l$，$H=2048$ 採 `FWHT`）。兩種方案在科學上皆完全乾淨，個人微幅偏好 **A2**（同時保住 `W8` 列尺度與同構 Ridge 先驗），但 **A1（現行 `draft-003`）亦完全合格可簽核**。

#### 問題 B：Stage-0 是否納入讀前守衛（Pre-Read Guard）與混合 Fallback 實體成本核算
1. **為何 `draft-003` 第 96 行將「`100%` 無守衛主評估＋凍結 `VALID` 距離分位 `{50%, 70%, 85%}` 守衛曲線」並列是最佳設計**：
   - 僅測 `100%` 無守衛僅能否證「無守衛全覆蓋局部仿射庫」；若少數極端大範數／離群 token 在線性外推下爆出巨額平方誤差，可能掩蓋常態 `70%–85%` 區域內局部斜率相對於共用斜率的真實差異。
   - `draft-003` 第 96 行選用 `distance = ||x - meanX_leaf||_2` 具極佳不變性（`[已證數學事實 Proven]`）：該距離由全層共用 `FP32` `meanX_leaf` 在讀圖前以 $O(H)$ 算出，**與投影秩 $r$ 及座標縮放完全無關**，確保在每個凍結 `VALID` 分位門檻（`50%, 70%, 85%, 100%`）下，**候選與 `{R0, R1, R2}` 面對完全相同的受保護 token 子集與完全相同的精確原專家 fallback 子集**，且零 `SCORE` 洩漏！
2. **必須誠實披露之兩階段群組屏障放大效應（Cohort Fallback Barrier Amplification）**：
   - **`[已證機率不等式 Proven]`**：單一算子在覆蓋率 $q \in \{0.50, 0.70, 0.85\}$ 下雖僅支付 $(1 - q)$ 比例之原專家 `3HI` 雙階段頁面與額外 `Down` 輸入投遞工作量，但在整層 $K \in \{6, 8\}$ 個選中專家並發時：
     - 若採用**逐專家獨立門檻 $\tau_e$**，只要 $K$ 個選中專家中**有任一專家**觸發 fallback，該層即被迫恢復 `GU -> SiLU -> Down` 雙階段串列屏障（全命中免屏障機率下界為 $\max(0, 1 - K(1-q))$，獨立近似為 $q^K$：當 $q=0.85, K=8$ 時 $0.85^8 \approx \mathbf{27.2\%}$！）；
     - 反之，若採用**全層共用單一葉距離分位門檻 $\tau_{\text{leaf}}$**（因 `||x - meanX_leaf||_2` 本就全層共用！），則當層 $K$ 個專家**同進同退**——要麼 $K$ 個專家 `100%` 同步走單階段局部圖（機率恰為 $q = 85\%$！），要麼整層同步回退至原生雙階段路徑（機率 $1 - q = 15\%$，完全消除混合路徑長尾拖慢）！建議在報告守衛曲線時，一併記錄此「全層同步守衛 vs 逐算子守衛」之實體屏障差異，且嚴格維持「僅報告算子混合 SSE 與混合實體工作量，不外推全模型品質」。

#### 問題 C：條件式系統價值與聯合預算帳本（不預設 `Flash-Jacobian` `C16/r128` 必落兩階段 Flash）
1. **精確聯合儲存帳本（`[已證位元組核算 Proven]` + `[未實測假設 Unmeasured]`）**：
   - 以 `DeepSeek`（1664 路由專家，$H=2048$）為例：`Flash-Jacobian` 在 `C=16, r=128` 下之區域輸入投影矩陣 $B_{e,c}$ 若存為裸 `W8`，大小為 $1664 \times 16 \times 128 \times 2048 = 6,979,321,856\text{ B} =$ **`6.979 decimal GB`（`6.500 GiB`）**（另加 row scales 約 `54.5 MB`）。由於系統非 FFN 常駐權重、Shared Experts、KV cache 與緩衝區在 `8,000,000,000 B`／`8 GiB` 下之精確剩餘額度尚未在目標板上鎖定，且 `Flash-Jacobian` 亦可採「熱區 $B_{e,c}$ 常駐 Host DRAM + 冷區串流／fallback」混合策略，**絕對不可在無實測下宣稱 `Flash-Jacobian` `C16/r128` 必然被迫支付兩階段 Flash 讀取**！
   - 同理，`draft-003` 第 95 行與 NativeSol 提出之**直接共同輸出基底控制（`A_{e,c} ≈ U_{e,q} B_{e,c}`）**極具鑑別力：在 `DeepSeek` `C=32, r=128, q=128` 下，全部 $1664 \times 32$ 個 $128 \times 128$ 條件潛矩陣 $B_{e,c}$ 存為 `W8` 僅佔 **`0.872 decimal GB`（`0.8125 GiB`）**，Flash 僅需為每專家存單一 $H \times q$ 矩陣 $U_{e,q}$（全庫 `0.436 decimal GB`），代價是每層須在 Host 執行 $K$ 次小型 `128x128` 變換、$K$ 次獨立潛向量 `A8` 量化與 $K$ 次獨立 Flash 輸入投遞（失去全層單一廣播輸入）。
   - **結論**：本候選之潛在價值嚴格定性為**「以靜態普通讀取 Flash 容量換取『單次 $O(\log C)$ 路由前查找＋全層單一廣播 `A8` 輸入＋專家條件輸出子空間隨葉旋轉』之有限硬體契約」**。Stage-0 勝過 `{R0, R1, R2}` 是入場必要條件；Stage-1 再於同一聯合預算下與 `U_{e,q} B_{e,c}`、`Flash-Jacobian` 合法快取配置及 `W4` 正面對決。

#### 問題 D：`EBC-LLM` 公開程式碼溯源紀律與 `FP64 DGEMM` 整數內積精確性證明
1. **`EBC-LLM` 來源定性與 Fail-Closed 鐵律**：
   - 我們僅檢視了作者公開開發 Notebook 原始碼（[source_cells.txt](file:///home/ericyu/expert_flash_experiment/results/research-2026-11-03-no-cr/prior-sources/ebc-author-code/source_cells.txt)，IEEE Xplore 論文全文受付費／JS 挑戰保護，嚴守規範未予繞過），**不推論已發表論文之數據曾觸發該 Notebook 中之 `NaN -> 隨機高斯張量` 替換分支**。`draft-003` 第 54 行規定任何非有限值、維度／模型身分不符、擷取缺漏或算術不一致一律 **Fail-Closed（列為實作／不確定狀態並留於同輪修復，絕不以合成隨機數據頂替，亦不誤判為科學失敗）**，完全正確。
2. **純 `FP64 DGEMM` 計算帶號 `I8` 內積之位元級精確性證明（`[已證 IEEE-754 事實 Proven]`）**：
   - IEEE-754 `binary64` 具 53 位有效尾數，故區間 $[-2^{53}, 2^{53}]$（$2^{53} = 9,007,199,254,740,992$）內之所有整數均可**零捨入誤差精確表示**。
   - 對任意帶號 `I8` 權重 $q_{w,j} \in [-127, 127]$ 與帶號 `I8` 激活 $q_{x,j} \in [-127, 127]$，單項乘積 $|q_{w,j} q_{x,j}| \le 127^2 = 16,129 \ll 2^{53}$ 為精確整數；對任意內積長度 $n$ 及**任意求和樹／BLAS 分塊歸約順序**（在 `alpha = 1.0, beta = 0.0`、純 `FP64` 運算且禁用任何混合精度降階累加下），任意中間部分和均滿足：
     $$\left|\sum_{j \in S} q_{w,j} q_{x,j}\right| \le \sum_{j=1}^n |q_{w,j} q_{x,j}| \le 16,129\,n.$$
   - 在本協定所有維度 $n \in \{64, 128, 512, 1024, 1408, 2048, 2816\}$ 下，最壞情況絕對總和上限為 $16,129 \times 2816 = \mathbf{45,419,264} < 2^{26} \ll 2^{53}$（安全餘裕逾 **$1.98 \times 10^8$ 倍**！），故純 `FP64 DGEMM` 所得浮點結果與精確 `I64` 整數內積在數學上 `100%` 恆等！配合實作前對抗性極值（全 `±127` 交錯／同號）與隨機帶號 `I64` 驗證，並嚴格保留內積後之 canonical 單精度順序 `float s = sw * sx; float f = (float)dot; return f * s;` 與全維度 `I` 之 `ggml` `SiLU`（且明確聲明離線批次 CPU 吞吐量絕非 `B=1` 部署延遲），完全合法安全。

---

### 4C. 對 `stage0_protocol.draft.json`（`draft-003-before-data`）之最終條件簽核意見（Final Conditional Signoff）

經逐條核對 [stage0_protocol.draft.json](file:///home/ericyu/expert_flash_experiment/results/research-2026-11-03-no-cr/round02-local-affine/stage0_protocol.draft.json)（`draft-003-before-data`），該草案已完整吸收我們與 NativeSol／黑臉在四輪交叉審查中發現的所有統計、幾何、量化與資源帳本漏洞。**本人（Gemini 4 Argon High Thinker）對該有界必要條件試點協定給予「條件式簽核同意（Conditional Signoff for Implementation Gate after `>= 21:56 UTC`）」**，僅提請 Root 與執行者 `executor_sol` 在進入不可變原始碼凍結（Source Freeze，開啟真實模型數據前）時鎖定以下 **3 項實作細則**：

1. **確認 `R2`（`r512` 與 `H2048`，含 `P = I` 全秩單位座標）享有完全相同的池化 Ridge $\eta$ 網格調優（第 80、86 行對齊）**：
   - 在 $H = 2048$ 且樣本來自 `120` 個具序列相關性之 prompts（尤其在 `1 vs 1/4` 路由重加權下），尾端主方向可能存在共線性；若 `R2(H2048)` 僅跑無正則化 $\eta = 0$，尾端方差膨脹可能人為削弱 `R2` 基準。請在程式碼中確保 `R2(512)` 與 `R2(2048)` 同樣在葉內去均值統計量上由自然路由 `VALID` 從同一 `eta_grid = [0, 1e-4, 1e-3, 1e-2, 0.1, 1, 10, infinity]` 選取最優 $\eta$。
2. **在 Source Freeze 中明確枚舉 `max-|t|` 同時信賴帶之正式主比較清單（第 60、77、92 行對齊）**：
   - 明確鎖定在進入 `SCORE` 前，每個 `(算子, C, 權重)` 是否先由自然路由 `VALID` 選出單一獲勝格式（`normal` vs `balanced`）及單一 $\lambda$（外加最佳有限 $\lambda$ 點），並與相同能力下由 `VALID` 選出的 `{R0, R1, R2(512), R2(2048)}` 建立固定配對差向量，以及當某 bootstrap 重抽樣出現分母方差為零／無自然命中時之精確統計處理規則（按第 92 行直接計數並標記該比較為 `inconclusive`，絕不靜默丟棄重抽樣）。
3. **守衛評估同時輸出「逐算子獨立 `VALID` 分位門檻」與「全層共用葉距離 `VALID` 分位門檻」診斷（第 96、105 行對齊）**：
   - 由於 `distance = ||x - meanX_leaf||_2` 天然為全層共用純量，順帶記錄全層同步門檻下之真實「整層零 fallback 比例」成本，無需額外模型查詢即可防止逐算子獨立覆蓋率高估整層單階段比例。



