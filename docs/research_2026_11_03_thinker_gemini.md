# Gemini 4 Argon High 研究筆記 — MoE In-Flash Computing (No CR-Read)

- **執行模型確認**：`Gemini 4 Argon High`（透過 Antigravity CLI；未切換模型、未啟用額外付費 credits）
- **實驗輪次狀態**：`0 / 10`（R1 `signed-sketch selector` 已於思考階段淘汰，不消耗實驗輪次；目前不做數值或候選實驗）
- **時間紀錄（真實 UTC，由系統／Root 紀錄）**：
  - Onboarding 與廣域探索起點：`2026-11-02T19:03:05Z`（不含前段 CLI 接線時間）
  - 首次落盤 Checkpoint：`2026-11-02T19:27:51Z`
  - 第一深入區塊推導落盤：`2026-11-02T19:36:09Z`
  - 第二深入區塊起點（Root UTC 紀錄）：`2026-11-02T19:46:17.841048Z`
  - 第二深入區塊推導與勘誤落盤：`2026-11-02T19:51:24Z`
  - 第三區塊最終勘誤與 Stage-0 終審落盤：`2026-11-02T19:57:31Z`
  - **新正式候選（Gate-Zero Exact Relative-W8A8 Certificate）思考起點**：`2026-11-02T19:26:00Z`（**最早實驗時間已保守延至 `19:58:00Z`**；目前仍在審查窗口內，未寫 code、未跑 probe，輪次消耗 `0 / 10`）

---

## 0. Onboarding 硬體不對稱性與四個死胡同邊界（簡要備忘）

1. **Ordinary Read（無 CR-read）的 `2.73×` 感測—計算不對稱性**：
   - 16 顆 `AiFChip`、8 條 ONFI（`2.0 GB/s/ch`）、PCIe `8.0 GB/s`；每晶片 4 planes × `16 KiB` = `64 KiB` 同 WL wave，`be-enc (1,3,3)` 單次感測 `28 µs`（有效晶片感測頻寬 **`2.34 GB/s/chip`**，全盤 `37.45 GB/s`），感測能耗佔晶片 ~`99.5%`（`18.278 pJ/bit` = `95.8 µJ/wave`）。
   - 反觀晶片上既有 `ECCLITE`（`0.167 mm²`）與 `PE`（`0.026 mm²`, `200 MHz`）設計吞吐為 **`6.4 GB/s/chip`**（處理一整個 `64 KiB` wave 僅需 `10.24 µs`），在 `28 µs` wave 中閒置約 **63.4%**；且每個 plane 4 個 `16 KiB` PB latches 在 `be-enc` 下僅佔用 2 個（data + cache），剩餘 **2 個 latches（`32 KiB/plane`）可放輸入向量**。
2. **度量區隔鐵律**：必須嚴格區分 (i) 單純 routed-FFN wave/服務時間、(ii) 含 shared/dense FFN 前綴之完整 FFN 服務時間、(iii) 含 `QKV/O`、attention、router 與 PCIe/ONFI 重疊之完整模型 TPOT；並區分 `canonical CPU W8A8`（對稱整數 per-vector `A8` + `F32` scale 與 `std::round` round-away）與 GGUF `Q8_0`（每 32 權重帶 2B scale，`+6.25%` byte 改變 wave 進位）。
3. **四個已數學排除的假方向**：
   - **純 Placement / AMP / 小備份**：`OLMoE` 與 `Mixtral` 在 `g16` 下距 64-plane 分數下界已 `0%`；`DeepSeek`/`G4` 在同格式 `W8` 下剩餘空間僅 `~1.5–5%`。
   - **無損 `W8` 壓縮／跨專家 XOR 差值**：實測 `W8` 列熵 `6.87–7.27 bits/w`，跨專家 XOR 升至 `7.93–7.95 bits/w`。
   - **小容量晶片 SRAM 跨 token 快取（`batch=1`）**：單 token 跨層工作集 `288 MiB–10.75 GiB`，`4 MiB` 總 SRAM 命中率趨近 0 且被 `8 GB` Host DRAM 嚴格支配。
   - **Capacity-for-Activation-LUT（PQ 查表）**：除 `MADDNESS`/`LUT-NN`/`Stella Nera`/`LUT-LLM (FCCM 2026)`/`Bit-serial Weight Pools` 強 prior 外，`be-enc` 僅能使用 `1 TB TLC` 的 `1/3` LSB 容量（約 `333 GB`），`2C/g` 倍膨脹與細粒度專家小 $R$ 造成的 `< 16 KiB` 頁內浪費構成容量—品質—wave 三難，不列為原創主軸。

---

## 1. Onboarding 廣域探索保存：4 個備選架構大方向摘要

> 依 Root 指示，精簡保存先前讀檔推導出的 4 個架構大方向（含最強基準、硬體差額與推翻判準），作為後續候選若思考淘汰時的備援池：

1. **備選 A（Exact 分佈，備用問題）— Post-ECC Broadcast Multi-Vector PE + Wave-Marginal Speculative Subtree Pruning**：
   - 每個 `16 KiB` 權重頁經既有 `ECCLITE` 解碼一次，同步廣播給 $q=4$ 組 `PE`（`+0.078 mm²/chip`），$q$ 個草稿輸入向量放在既有 `32 KiB` PB latches（`0 B` 新輸入 SRAM），整列串流僅需 `64 B` 純量 `I32` 累加器；每層按私有專家對瓶頸晶片 $\max_c W_c$ 的邊際膨脹 $\Delta W_{\max}$ 剪除低存活率草稿子樹（保留 $\Delta W_{\max}=0$ 的晶片 slack 分支；`T>0` 下剪除 $v_k$ 僅放棄驗證 $v_{k+1}$，不破壞 Leviathan 精確拒絕採樣）。
   - **強基準／推翻判準**：對決 `BigMoMo (2609.14643)`、`FlowSpec`/`FASER`、`DraftExpert (2607.24434)` 及固定 $q^*$ 不剪枝廣播；最大風險為 MoE Birthday-Paradox 專家聯集膨脹 $E(1-(1-K/E)^q)$。
2. **備選 B（品質可控近似，全投影 `GU+Down` 0% 容量膨脹）— Contiguous Nested Non-Uniform `4+4` (`NQL-4+4`) + 晶片 Slack 水位填充**：
   - 將 `Gate/Up/Down` 全拆為同 IFP block 連續存放的 `Base4`（4-bit NormalFloat 16B LUT 索引，重排 row-band 為 2 倍列數使單讀 `Base4` 真正省半數 waves）與 `Res4`（4-bit 線性殘差）；以對角 Hanson-Wright 方差指標分配 `Res4`，並在非瓶頸晶片（$W_c + \Delta W_{c,e} \le \max_{c'} W_{c'}$）**零延遲免費補讀 `Res4`（Chip-Slack Promotion）**。
   - **強基準／推翻判準**：必須擊敗同平均 bitwidth 的靜態重排 `NF5/NF6` 及 `DPA-MoE (2610.21378)` / `PagedWeight (2607.16184)`；若相同 `ΔPPL ≤ +0.05` 下對靜態 `NF5/NF6` 的 `max_chip` wave 優勢 `< 5%` 則淘汰。
3. **備選 C（結構化稀疏 + 消除 Host 屏障，近似版）— In-Die Intermediate Neuron-Slice Fusion (`In-Die Slice-MoE`)**：
   - 改按離線共激活分群之 **中間維度切片**（每晶片每切片 $m = \lfloor 65536/H \rfloor = 16\text{..}42$ 個中間神經元），同 block 依序放 `[Gate_b (1 wave); Up_b (1 wave); Down_b (1 wave)]`，晶片內加微型 `SwiGLU + Block-m A8`（消除全向量 `amax` 同步且量化精度高於 whole-vector `A8`！）與 `H×4B`（`6–16 KiB/chip`）層輸出累加 SRAM；感測 `Gate_b` 後若切片能量低於門檻即同時跳過 `Up_b + Down_b`（省 `2` waves，上限 `66.7%` FFN），並消除 `GU -> Host -> Down` 屏障（有無 shared 皆適用）。
   - **強基準／推翻判準**：對決零硬體 `Dynamic Top-K`、`Channel Selection (2610.37386)`、`SLIM`、`Ripple`、`CATS`。
4. **備選 D（次位元組頻寬匹配）— Post-ECC Layer-Shared Codebook Expansion (`3-bit d=4 VQ`) + 非二冪 Wave 重排**：
   - 於 `ECCLITE` 後設層共享 `16 KiB` 正規化 codebook SRAM（每層僅 1 wave 載入，解決 MoE 多專家 codebook 重載風暴），將 `2.34 GB/s` 感測流展開 `2.67×` 為 `6.24 GB/s` 匹配既有 `6.4 GB/s` PE，並以非二冪 row-band 編譯消除 `3/8` 縮減在 `g16` 上的 `+33.3%` wave 浪費。
   - **強基準／推翻判準**：必須在品質—wave 曲線上顯著擊敗無需 `16 KiB` SRAM 的純量 `NF3/NF4` + 整列 DP 編譯器。

---

## 2. 新正式候選深入區塊 1：Gate-First Exact Relative-W8A8 Zero Certificate

- **正式思考起點**：`2026-11-02T19:26:00Z`（最早實驗時間 `19:56:00Z`；本區塊尚未滿 30 分鐘，未執行任何數值／activation probe，輪次消耗 `0 / 10`）。
- **核心命題**：先算完整 `Gate` 並由少量 `Up` witness rows 取得真實 `amax` 下界 $L > 0$；對未讀 `Up` row $i$ 利用確定性上界 $|u_i| \le U_i$，若嚴格證明 $|\text{SiLU}(g_i)| U_i$ 在 canonical `W8A8` 下必量化為 `0`，則同時跳過 `Up` row $i$ 與對應 `Down` column $i$，且恢復與完整計算 **bitwise 相同** 的 `amax`、scale 與 `Down` 整數內積。

---

### 2.1 嚴格 IEEE-754 `binary32` 與 `std::round`（Ties-Away）證書推導（修補實數不等式漏洞）

專案 CPU 整數參考（`full_ffn_integer_reference.cpp`）對中間向量 $a \in \mathbb{F}_{32}^I$ 的量化契約為：
$$A = \max_{0 \le k < I} |a_k| \in \mathbb{F}_{32}, \qquad s_a = \text{fl}_{32}(A \oslash_{32} 127.0\text{f}), \qquad z_i = \text{std::round}\big(\text{fl}_{32}(a_i \oslash_{32} s_a)\big).$$

> **關鍵浮點陷阱（實數 `< L/254` 在 `binary32` 下不安全！）**
> `std::round(y)` 為 ties-away-from-zero，故 `std::round(y) == 0` 當且僅當 $|y| < 0.5\text{f}$（`0.5f` 為精確二冪 $2^{-1}$，其左鄰浮點數為 $0.5 - 2^{-25}$）。然而除法 $a_i \oslash_{32} s_a$ 採用預設 RNE（`FE_TONEAREST`），當實數商 $|a_i| / s_a \in [0.5 - 2^{-26}, 0.5)$ 時，$\text{fl}_{32}(|a_i| / s_a)$ 會先向上捨入成 **`0.5f`**，隨即被 `std::round(0.5f)` 推向 **`1`**！此外若 $L$ 落入次正規數區間使 $s_{\min}(L) = \text{fl}_{32}(L / 127.0\text{f})$ 下溢或失準，亦會破壞不等式。

**零漏洞（0-ulp-leak）單調向外捨入證書定理**：
利用 IEEE-754 RNE $\text{fl}_{32}(\cdot)$ 在 $\mathbb{R}_{\ge 0}$ 上的**全域單調性**（$0 \le x \le y \implies \text{fl}_{32}(x) \le \text{fl}_{32}(y)$），無需切換硬體 FPU rounding mode，定義正浮點數的 1-ulp 向外運算 `up32(v) = nextafterf(v, +INFINITY)`（硬體僅需對正數 `binary32` 位元表示做一次 `u32 + 1` 整數加法）：
1. 設整數內積上界 $|M_i| = |\sum_{j=0}^{H-1} W^U_{ij} q_{xj}| \le B_i \in \mathbb{Z}_{\ge 0}$（注意 $H \le 4096 \implies B_i \le 127^2 H < 2^{26}$，32-bit 整數完全不溢位）。
2. 定義向外上界鏈（已知 $s^{\text{eff}}_{U,i} = \text{fl}_{32}(s^U_{w,i} \otimes_{32} s_x) \ge 0$ 與精確已算之 $|\gamma_i| = |\text{SiLU}_{32}(g_i)|$）：
   $$M^{\uparrow}_i = \text{up32}(\text{fl}_{32}(B_i)), \quad U_i = \text{up32}(M^{\uparrow}_i \otimes_{32} s^{\text{eff}}_{U,i}), \quad \widehat{A}_i = \text{up32}(|\gamma_i| \otimes_{32} U_i).$$
   由單調性必有 $|a_i| = \text{fl}_{32}(|\gamma_i| \otimes_{32} |u_i|) \le \widehat{A}_i$。
3. 設已完整計算之 witness 集合 $S_{\text{wit}} \subset [I]$ 給出 $L = \max_{k \in S_{\text{wit}}} |a_k|$，並設正常浮點保護條件：
   $$\text{isfinite}(L) \;\wedge\; s_{\min}(L) := \text{fl}_{32}(L \oslash_{32} 127.0\text{f}) \ge 2^{-125} \text{（排除次正規數與零）}.$$
4. **精確零證書條件**：若對所有 `isfinite`($\gamma_i$) 成立且滿足：
   $$\widehat{A}_i < L \quad \text{且} \quad \text{fl}_{32}\big(\widehat{A}_i \oslash_{32} s_{\min}(L)\big) < 0.5\text{f},$$
   則因 $|a_i| \le \widehat{A}_i < L \le A \implies s_{\min}(L) \le s_a$，由除法單調性得：
   $$\text{fl}_{32}(|a_i| \oslash_{32} s_a) \le \text{fl}_{32}(\widehat{A}_i \oslash_{32} s_a) \le \text{fl}_{32}(\widehat{A}_i \oslash_{32} s_{\min}(L)) < 0.5\text{f} \implies z_i = 0,$$
   且 $|a_i| < L \le A$ 保證跳過 row $i$ 後由剩餘列算出的 $A$ 與 $s_a$ **逐位元（bit-for-bit）完全不變**！（若任一列出現 `NaN/Inf` 或 $s_{\min}(L) < 2^{-125}$，無條件退回完整讀取。）

---

### 2.2 證書緊度（Tightness）的兩個核心數學定理與破局設計

令 $\sigma_u = \text{RMS}_i(u_i)$，並定義上界鬆弛比 $\kappa_i := U_i / \sigma_u$。證書要求 $|\text{SiLU}(g_i)| < \frac{L / \sigma_u}{254 \cdot \kappa_i}$。

#### 定理 1（對 `critic_sol` Block-Hölder 修正案的反證：無符號三角不等式與球面裝填下界）
1. **Khintchine 符號抵銷壁壘**：`critic_sol`（第 200 行）提議以分塊 Hölder $U_{\text{int}} = \sum_{b=1}^{H/B} (\sum_{j \in b} |w_{ij}|) \max_{j \in b} |q_{x,j}|$（$B=128$，metadata `1.56%`）收緊 L2 Cauchy。然而對近似零均值之 $w, q_x$：
   - L2 Cauchy 有 $\kappa_{\text{L2}} \approx \sqrt{H} \in [39.2, 64.0]$；
   - 分塊 Hölder（$B=128$）因 $\sum_{j\in b}|w_{ij}| \approx 0.80 B\sigma_w$ 且 $\max_{j\in b}|q_{x,j}| \approx \sqrt{2\ln 128}\,\sigma_x \approx 3.12\sigma_x$，總和為 $\kappa_{\text{BH},128} \approx 2.50\sqrt{H}$——**反而比純量 L2 Cauchy 更鬆 `2.5×`**！
   - 即使推到極限 $B=1$（逐元素 $\sum_{j=1}^H |w_{ij}||q_{xj}|$，等於已讀完所有權重大小！），因丟棄正負號抵銷，仍有 $\kappa_{B=1} \approx \frac{2}{\pi}\sqrt{H} \approx 0.637\sqrt{H} \in [25, 41]$！對稱輸入區間上的正負分塊和亦同（$3\sigma_x(\sum w^+ + \sum w^-) = 3\sigma_x\sum|w|$）。
2. **球面裝填資訊下界（Sphere-Packing Lower Bound）**：若未讀 `Up` 列 $w_i \in \mathbb{R}^H$ 僅靠每列 $b$ bits 靜態離線 metadata $\phi(w_i)$ 給出對任意 $q_x$ 皆成立的確定性上界，則纖維 $\{w : \phi(w)=\phi(w_i)\}$ 在 $S^{H-1}$ 上的平均球冠角半徑滿足 $\sin\theta \ge 2^{-b/H}$。故在 `2%` 容量預算（$b = 0.16H$ bits）下，對各向同性輸入尾部的確定性界最多只能收緊 $1 - 2^{-0.16} = \mathbf{10.5\%}$！

#### 修正機制 U1（唯一能繞過球面下界的靜態收緊）：離群通道／共享低秩正交消去證書（Outlier-Deflated Cauchy）
- **關鍵洞察**：雖然 $w_i$ 在球面上近似均勻，但 **真實 LLM 的激活輸入 $q_x$ 極度各向異性（Anisotropic）**——少數 $r = 16\text{..}32$ 個離群通道（outlier channels / rogue dimensions）或主軸 $V \in \mathbb{R}^{H \times r}$（$V^\top V = I_r$；最簡零乘法版直接取離線統計能量最高的 $r$ 個座標軸 $S_{\text{out}} \subset [H]$！）往往佔去 $\|q_x\|_2^2$ 的 **50%–90%**！
- **精確分解**：離線對每列存 $r$ 個離群座標權重 $w_{i, S_{\text{out}}}$（`r` bytes/row）與正交補空間範數 $\rho_i = \|w_{i, \overline{S_{\text{out}}}}\|_2$（`2B` FP16 向外捨入；$r=16$ 時總 metadata 僅 `18 B/row` = **`0.88%`** of $H=2048$ `Up`！）。線上每層開始時對共同輸入 $q_x$ 算一次 $\|q_{x, \overline{S_{\text{out}}}}\|_2 = \sqrt{\|q_x\|_2^2 - \|q_{x, S_{\text{out}}}\|_2^2}$，則：
  $$|w_i^\top q_x| \le \left|\sum_{j \in S_{\text{out}}} w_{ij} q_{xj}\right| + \rho_i \|q_{x, \overline{S_{\text{out}}}}\|_2.$$
  離群座標的符號抵銷被 **100% 精確算出**，而 Cauchy 項隨 $\|q_{x, \overline{S_{\text{out}}}}\|_2 / \|q_x\|_2$ 直接縮減（若 top-16 佔 $85\%$ 能量，$\kappa$ 立刻縮小 $2.58\times$）！

#### 修正機制 U2（`Partial-Up` 證書核算）：為何列前綴無效、而 `Hi4` MSB 僅能作二階補充？
1. **列前綴 Partial-Up（讀前 $fH$ 列）失敗**：剩餘 $(1-f)H$ 列的 Cauchy 鬆弛僅按 $(1-f)\sqrt{H}$ 線性下降；讀掉 $50\%$ `Up` 僅收緊 $2\times$ 卻先賠掉一半 `Up` 省量與額外命令。
2. **`Hi4` 高 4-bit Nibble Partial-Up（`w = 16h + 8 + r`, $|r+0.5| \le 7.5$）精確核算**：
   - 若將未證列拆為重排後連續的 `Up_Hi4`（`0.5W` waves）與 `Up_Lo4`（`0.5W` waves），讀完 `Up_Hi4` 可得精確粗內積 $M^{\text{hi}}_i$，剩餘殘差範數 $\|r_i + 0.5\|_2 \approx 4.62\sqrt{H}$，相對典型 `INT8` 列標準差 $\sigma_w \approx 36$，鬆弛比由 $\kappa_{\text{L2}} \approx 45.3$ 驟降至 **$\kappa_{\text{Hi4}} \approx \frac{4.62\sqrt{H}}{\sigma_w} \approx 5.8$（收緊近 `8×`！）**。
   - **但注意兩個硬限制**：(a) 對常態高斯極大值 $3.5\sigma_u < 5.8\sigma_u$，單靠 `Hi4` 仍不足以在無 witness 下保證 $u^{\text{lower}}_{i^*} > 0$；(b) 已先付 `0.5W` 讀 `Up_Hi4`，跳過 `Up_Lo4` 最多再省 `0.5W`（即 **FFN 上限 `< 1/6 = 16.7%`**，重蹈 R1 覆轍！）。故 `Hi4` 絕不可單獨使用，最多只能作為 `U1`（`0`-bit Tier 0）之後的 Tier-1 補充，且硬體複雜度顯著上升，不建議首輪採用。

---

### 2.3 Witness 真頁成本消除與 Barrier 壓縮：離線同構排列 + `[Gate + Hot-Witness Up_Tile0]` 融合

針對 `gate_zero_thinking.md` 風險 2（「動態選取 witness 可能先碰遍所有 NAND 頁」）與風險 4（「拆 `Gate -> Witness -> Remaining Up -> Down` 變成 4 個 phases」）：

- **修正機制 U3（零散落、零額外 Barrier 的熱點見證塊設計）**：
  1. **離線同構神經元重排 $\pi$（100% 無損、零硬體／容量成本）**：對每個專家的中間維度 $i \in [0, I)$ 同步重排 `Gate` rows、`Up` rows 與 `Down` columns：
     - **`Tile 0`（Hot Witness Tile，固定佔首 `1` 個 page 或 `1` 個 wave，即 $m_0 = 8\text{..}32$ 個神經元）**：放入 `TRAIN` 校準集上最常承載 `amax`（$\Pr(|a_i| \ge 0.5 A)$ 最高、或離群 super-weight）的神經元。這批熱點神經元幾乎不可能量化為 `0`，因此讀取它們的 `Up` 權重是 **100% 必要有效工作，零浪費**！
     - **`Tiles 1 .. T-1`（Co-Zero Cold Tiles）**：將其餘 $I - m_0$ 個神經元依 `TRAIN` 上的共同可證零機率（co-certification affinity）分群排列，最大化整頁／整 subpage 同時為零的機率。
  2. **Phase 1 融合執行 `[Gate (全) + Up_Tile0]`**：
     - 在第一階段直接將 `Up_Tile0` 緊接於 `Gate` 之後一次發出！Host 收到 `Gate` 全向量與 `Up_Tile0` 的精確輸出後，**立刻在同一屏障內同時擁有全部 $g_i$ 與熱點見證下界 $L_0 = \max_{i \in \text{Tile}_0} |a_i|$**！
     - **消除散落讀取**：`0` 次隨機跨頁 witness 讀取！
     - **壓縮階段屏障**：由 `4` 階段減為 **`3` 階段**（`[Gate + Up_Tile0] -> [未證 Up Cold Tiles] -> [非零 Down Columns]`）；且在 `be-enc` 下共同輸入 $q_x$ 常駐於未動用的 PB Latch 2，`Phase 1 -> Phase 2` **無需透過 ONFI 重傳 $q_x$**（須明列「同層跨命令保留 PB 輸入 latch」之控制器契約）。

---

### 2.4 原生 `16 KiB` 頁 / `64 KiB` Wave vs. `4 KiB` SOML 的硬體與幾何真相

| 模型 | $H$ (B/row) | $I$ | `4 KiB` SOML 含列數 | `16 KiB` 頁含列數 | `64 KiB` 4-Plane Wave 含列數 | 單專家 `Up` 總 Waves（全盤 16 晶片合算） |
|---|---:|---:|---:|---:|---:|---:|
| **G4** | `1,536` | `512` | `2`（pad 25%）或跨頁 `2.67` | `10`（pad 6.25%） | `42` | `12`（若 `g12` 每晶片僅 `1 wave`！） |
| **OLMoE** | `2,048` | `1,024` | `2`（整除） | `8`（整除） | `32`（整除） | `32`（若 `g16` 每晶片僅 `2 waves`！） |
| **Qwen3** | `2,048` | `768` | `2`（整除） | `8`（整除） | `32`（整除） | `24`（若 `g16` 每晶片僅 `1.5 waves`！） |
| **DeepSeek** | `2,048` | `1,408` | `2`（整除） | `8`（整除） | `32`（整除） | `44`（若 `g16` 每晶片僅 `2.75 waves`！） |
| **Mixtral** | `4,096` | `14,336` | `1`（整除！） | `4`（整除） | `16`（整除） | `896`（`g16` 每晶片 `56 waves`） |

1. **原生同步 4-Plane Wave 的致命粒度**：若無獨立 plane 定址（AMP），單一 plane 跳頁只省感測能耗（`23.9 µJ/page`），**不省 `28 µs` 時間**；必須同晶片 4 個 planes 共 `16–42` 個神經元全被證明為 `0` 才能省 1 個 wave！且細粒度 MoE（如 `G4`/`Qwen3`/`OLMoE`）在 `g16` 下每專家每晶片總共只有 `1–2` 個 `Up` waves，還要面對 $\max_{c \in [16]} W_c$ 最慢晶片的木桶效應。
2. **`4 KiB` SOML 的兩個隱藏硬體條件（絕非免費改參數！）**：
   - **好消息（`ECCLITE` 相容）**：`AiF` 的 `ECCLITE` 碼字為 `1 KiB`（10b BCH），故 `4 KiB` SOML 恰含 `4` 個完整 `1 KiB` ECC 碼字（甚至 `2 KiB` = $H=2048$ 單列亦含 `2` 個完整碼字！），不破壞 ECC 解碼器。
   - **壞消息（同 Block 序列化陷阱）**：依 `SOML (ASPLOS 2019)`（約 `+1.7%` die area），同一 plane 在單一 `28 µs` 週期並行感測 4 個 `4 KiB` subpages **必須來自 4 個不同實體 blocks**！若將單一專家的 `Up` 全放在同一個 `3 MiB` LSB block 內，跳過部分 subpages 後對剩餘 subpages 做 SOML 讀取會退化成逐次 `28 µs`（讀 `16 KiB` 反變 `4 × 28 µs` = **慢 4 倍**！）。因此必須將多專家 Cold Tiles 交錯條帶化至每 plane **4 個共享 SOML block-banks**，才能在不膨脹保留容量下維持 `16 KiB / 28 µs` 有效吞吐！
3. **`Down` 必須改為 Column-Sharding 並付晶片累加 SRAM**：原 `AiF` 按輸出列 `H/16` 切分 `Down` 時，一頁 `16 KiB` 塞入 `128` 個中間欄位（$H=2048$），整頁共零機率為 `0`！故 `Down` 必須改存完整 $H$ 維欄位（`8` 欄/頁、`2` 欄/SOML），並在每顆 `AiFChip` 配置 **`H × 4 B`（`6–16 KiB/chip`）的層輸出 `I32`/FP32 累加器 SRAM**（全層結束後才匯出 16 份 $H$ 維向量做合併，流量 `64–128 KiB/layer`）。**公平基準鐵律：強控制 `B0` 與 `CATS` 必須享有完全相同的 SOML 4-bank 與 `Down` column-sharding 硬體！**

---

### 2.5 最強自身推翻反例（The Three-Way Pincer Counterexample）

在投入任何實驗輪次前，我們發現一個極強的 **三面夾擊反例**，直接威脅本候選的必要性：

#### 反例 1：零證書強基準 `B0 (Full-GU + Post-A8 Exact-Zero Down)` 的完全支配擠壓
設 $\mathcal{Z}_{\text{true}} = \{i : z_i = 0\}$ 為 canonical `W8A8` 的真實零集合，而本候選證明之集合為 $\mathcal{Z}_{\text{cert}}$。
- 因證書具健全性，必有 **$\mathcal{Z}_{\text{cert}} \subseteq \mathcal{Z}_{\text{true}}$**。
- 考慮極簡強基準 **`B0`**：維持原融合 `Gate+Up`（無 witness、無 `U_i` metadata、無 `Gate->Up` 額外階段屏障！），算完精確 $a_i$ 與 $A$ 後得到 **100% 完整的 $\mathcal{Z}_{\text{true}}$**，在完全相同的 `Down` column-sharding / SOML 硬體下跳過所有 $z_i = 0$ 的 `Down` 欄位！
- 注意本候選在 Phase 2 補讀完 $[I] \setminus \mathcal{Z}_{\text{cert}}$ 的 `Up` 後，在 `Down` 階段同樣能獲得完整的 $\mathcal{Z}_{\text{true}}$；因此 **相較於零證書基準 `B0`，本候選在 `Down` 上的額外獲益嚴格為 `0`！** 本候選對 `B0` 的**唯一淨增量**僅剩：
  $$\Delta \text{FFN}_{\text{vs } B0} = \frac{p_{\text{cert}}^{\text{up}} - w_{\text{meta}}}{3 - p_{\text{true}}^{\text{down}}} \quad \left(\text{佔原稠密 FFN 至多 } \frac{p_{\text{cert}}^{\text{up}} - w_{\text{meta}}}{3} < 33.3\%\right) !$$

#### 反例 2：`SiLU` 負半軸隆起（`0.2785`）與 $254 \cdot \kappa$ 乘積的高斯崩潰計算
不同於 `ReLU`（$g_i \le 0$ 直接給出 $50\%$ 免費精確零，完全不需 $U_i$ 與 $L$！），$\text{SiLU}(g) = g\sigma(g)$ 對所有 $g \ne 0$ 皆非零，且在 $g = -1.28$ 有局部極大值 **`0.2785`**，在整個 $g \in [-4, -0.3]$ 區間皆維持 $|\text{SiLU}(g)| \in [0.072, 0.278]$！
- 代入典型無極端離群值之高斯尺度（$H=2048$，$L/\sigma_u \approx 15$）：
  - 即使靠 `U1` 離群消去將 $\kappa$ 從 `45.3` 壓到 `20`，仍需 $|\text{SiLU}(g_i)| < \frac{15}{254 \times 20} = \mathbf{0.00295}$！
  - 這要求 $|g_i| < 0.0059$（近零窄縫，約 `0.3%`）或 $g_i < \mathbf{-7.85}$（對 $\sigma_g = 1.5$ 為 $-5.2\sigma$，機率 $10^{-7}$；即使 $\sigma_g = 3.0$ 亦僅 `0.45%`）！
  - 若純量認證率 $p_{\text{row}} \approx 1\text{–}3\%$，則連 `4 KiB` SOML（2 列共零：$p_{\text{row}}^2 \le 0.09\%$）與 `16 KiB` 原生頁（8 列共零：$p_{\text{row}}^8 \approx 10^{-13}$）的 `Up` 跳過率 $p_{\text{cert}}^{\text{up}}$ 都將趨近 **`0%`**，扣除 `0.9%` metadata 與多一次 phase barrier 後 **淨輸給 `B0`**！
- **唯一存活條件（Super-Weight / 極端離群機制）**：唯有當真實 MoE 模型在多數層與專家上普遍存在極端 SwiGLU 離群激活（例如 $L/\sigma_u \ge 150$，且 `U1` 將 $\kappa$ 壓至 $\le 15$，使門檻放寬至 $|\text{SiLU}(g_i)| < 0.039 \implies g_i < -4.8$ 或 $|g_i| < 0.08$），且具有高度負偏壓 `Gate` 分佈時，$p_{\text{cert}}^{\text{up}}$ 才可能跨過 `30%`（即對 `B0` 產生 `≥ 10%` FFN 增量）。

#### 反例 3：審稿人的「保護量化瑕疵（Quantization Artifact）」質疑
- 專案既有數據已顯示，canonical 全向量 `W8A8` 相較原模型本身即有量化誤差（如 `DeepSeek` `+1.94%` PPL），而全向量 `A8` 中大量出現 `0` 的主因正是單一離群值撐大 `amax`、把其餘小激活壓成下溢零！
- 若為了「逐位元精確重現全向量 `A8` 的下溢零」而付出 $20\times\text{–}64\times$ 的保守上界稅，導致可跳過頁數遠遜於同品質近似門檻（`CATS`）或更高精度的 **Block-32 `A8` In-Die Slice-MoE**（備選 C，不僅消除 `amax` 屏障還提升量化精度！），在頂會審查中極難防禦「為何對一個次優動態量化器維持保守 exactness」的質疑。

---

### 2.6 需向 `thinker_sol` 與 `critic_sol`（黑臉）交叉提問的關鍵問題

1. **對 `critic_sol`（關於 Block-Hölder 與 `B0` 支配界）**：
   - 你在 `critic_sol.md` 第 200 行建議用 `B=128` 分塊 Hölder（$\sum_b n_{ib} \max|q_{x,b}|$）改善 L2 Cauchy；但依我們 §2.2 定理 1 的推導，丟棄正負號抵銷會使 $\kappa_{\text{BH},128} \approx 2.5\sqrt{H}$ **比純量 L2 Cauchy 更鬆 `2.5×`**（連 $B=1$ 也有 $\frac{2}{\pi}\sqrt{H}$）。你是否同意靜態無符號分塊界無法突破 $\Theta(\sqrt{H})$，唯一能收緊靜態界的只有利用輸入 $q_x$ 各向異性的 **離群通道／低秩正交消去（`U1`）**？
   - 在公平給予 `B0 (Full-GU + Post-A8 Exact-Zero Down)` 相同 `Down` column-sharding / 4-bank SOML 硬體後，`gate-zero` 對 `B0` 的唯一增量只剩 `Up`（上限 `< 1/3` FFN）。若連 `B0` 在原生 `16 KiB` 頁下都省不到顯著 `Down` waves，`gate-zero` 是否必然更差？反過來說，若只有靠 `4 KiB` SOML + 晶片 `H×4B` SRAM，`B0` 單獨是否就已吃掉大頭，使證書本身的邊際貢獻不足以支撐一篇論文？
2. **對 `thinker_sol`（關於真實模型離群比 $L/\sigma_u$ 與是否該直接思考淘汰）**：
   - 在不跑新數值 probe 的前提下，根據你對這 5 個模型（無 shared 的 `OLMoE`/`Mixtral` 與有 shared 的 `DeepSeek`/`Qwen3`/`G4`）SwiGLU 結構與既有文獻的理解，是否存在任何證據顯示 **所有模型** 皆具備 $L/\sigma_u \ge 100$ 的常態性極端離群值來支撐 $|\text{SiLU}(g_i)| < \frac{L/\sigma_u}{254\kappa}$ 跨過 `-0.278` 負半軸高原？
   - 若無此普遍性質，我們是否應在 `19:56 UTC` 前 **主動於思考階段淘汰（thought-reject）純 exact `gate-zero`**（保住全部 `10/10` 實驗輪次！），改將 `U3` 的共激活切片硬體直接升級為無 `amax` 屏障、可同時跳 `Up+Down` 達 `66.7%` 上限的 **In-Die Slice-MoE（備選 C）** 或全投影 **Nested `NQL-4+4` + Chip-Slack Waterfilling（備選 B）**？

---

### 2.7 本區塊結論與實驗建議（供 Root 決策）

- **若保留 `gate-zero` 進入後續區塊，必須採納的四項修正組合（缺一不可）**：
  1. **數值證書**：改用 §2.1 單調 `up32 = nextafterf` 向外捨入 + $s_{\min}(L) \ge 2^{-125}$ 保護（修補 `0.5 - 2^{-26}` ties-away 漏洞）。
  2. **上界緊度（`U1`）**：放棄更鬆的 Block-Hölder，改用 **Top-$r$（$r=16\text{..}32$）離群輸入通道精確內積 + 補空間 L2 範數**（metadata `~0.9%`）。
  3. **見證與屏障（`U3`）**：離線同構重排神經元，將 `m_0 = 8..32` 個熱點 `amax` 承載列置於 `Up_Tile0` 並與 `Gate` 在 Phase 1 融合讀取（**零散落讀頁、零見證浪費、4 階段減為 3 階段**），其餘列按 `TRAIN` 共證零率分群。
  4. **讀取粒度**：若用 `4 KiB` SOML（相容 `1 KiB` `ECCLITE`），必須配對 **每 plane 4-bank 共享塊交錯**（避開同 block 序列化慢 4 倍陷阱）與 **`Down` 欄位切分 + 每晶片 `6–16 KiB` 輸出累加 SRAM**，且 `B0` 與 `CATS` 控制組完全同享此硬體。
- **是否建議在 `19:56 UTC` 啟動第一輪實驗？——目前建議「傾向思考階段淘汰（Lean Thought-Reject），暫不消耗第 1 輪實驗」**：
  - 除非交叉審查能推翻 §2.5 的三面夾擊（`B0` 吃掉全部 `Down` 收益使 `gate-zero` 增量剩 `< 1/3`、`SiLU` 負半軸 `0.2785` 隆起乘上 $254\kappa$ 的高斯崩潰、以及對有損全向量 `A8` 維持保守 exactness 卻大輸 `CATS`/Block-A8 的審稿質疑），否則貿然在 `19:56 UTC` 跑 probe 極可能白白消耗寶貴的第 1/10 輪實驗。建議先就此三點與 `thinker_sol`、`critic_sol` 正面交鋒後再拍板去留。


---

## 3. 第二深入思考區塊（`19:46:17Z` 起）：黑臉審查正面勘誤、投影 Cauchy 新界剖析與 Stage-0 實驗決策

> **歷史保留與勘誤原則**：上方 §0–§2 完整保留第一區塊原貌不刪改；以下 §3.1 針對 `critic_sol`（19:34–19:45 UTC）與 Root 指出的過度推論、證明缺口與硬體成本疏漏逐條正面更正，§3.2 嚴格剖析新增之 `Gate-Conditioned Projected Cauchy` 界，§3.3–§3.4 重新給出基於實證必要性（而非高斯推測）的實驗決策與 Stage-0 判準。

---

### 3.1 對第一區塊（§0–§2）的四項明確勘誤與修正

#### 勘誤 1：撤除球面裝填「定理」稱呼、區分 `Block-L2` 與 `Block-L1/L∞`、並將分佈假設降格為待測假說
1. **`Block-L2` 必不鬆於 Global-L2（修正 §2.2 過度外推）**：
   - §2.2 僅計算了 i.i.d. 零均值高斯下的 $L_1\text{–}L_\infty$ 分塊界 $\sum_b \|w_{i,b}\|_1 \|q_{x,b}\|_\infty \approx 2.50\sqrt{H}\sigma_u$，卻錯誤外推稱「靜態分塊界無法收緊 L2、唯一只有 `U1` 可收緊」。
   - **明確更正**：對任意分塊 $b=1\dots M$，**`Block-L2` 界**
     $$U_{\text{Block-L2},i} = \sum_{b=1}^M \sqrt{W_{2,i,b} X_{2,b}} \le \sqrt{\sum_{b=1}^M W_{2,i,b}} \sqrt{\sum_{b=1}^M X_{2,b}} = \sqrt{W_{2,i} X_2}$$
     由 $\mathbb{R}^M$ 上的 Cauchy-Schwarz 不等式，**對一切 $w_i, q_x$ 恆不大於 Global-L2**，且只要權重與輸入的分塊能量比例 $W_{2,i,b}/X_{2,b}$ 非均勻或錯位，即嚴格收緊！即使對 $L_1\text{–}L_\infty$ 或正負區間分塊界，取 $\min(\text{Global-L2}, \text{Block-L2}, \text{Block-}L_1 L_\infty)$ 亦絕不會比 Global-L2 更差。
2. **撤去「球面裝填下界 → 2% metadata 最多收緊 10.5%」之定理稱呼**：
   - 連續球面 $S^{H-1}$ 的球冠體積僅描述對全空間任意連續向量與最壞方向 $q_x$ 的 minimax 半徑，**並未定義或證明**於本問題之「有限個凍結 `INT8` 權重列 $w_i \in [-128,127]^H$、搭配當前已知運行時輸入 $q_x$ 及已算出之 `Gate` 內積 $d_i$」條件分佈上。撤除此「定理」稱呼，不將其作為 no-go 依據。
3. **區分未驗證文獻假設與本專案實測**：
   - §2.2 中「Top 16–32 離群通道佔 $\|q_x\|_2^2$ 之 50%–90%」僅為一般稠密 LLM 文獻觀察，**無本專案 5 個 MoE 模型實證**；
   - §2.5 中 $g_i \sim \mathcal{N}(0,\sigma_g^2)$、$L/\sigma_u \approx 15$、純量認證率 `1–3%`、以及獨立同分佈乘積 $p^m$（忽略神經元共激活相關性與離線 `TRAIN` 共零分群！）**皆僅為條件分佈風險示意，絕非實測下界，不能據此在無數據前 thought-reject 所有模型**。
4. **採納更簡潔之 RNE 單調浮點界（對齊 `critic_sol` 19:34 UTC）**：
   - 當整數內積上界 $B_i \in \mathbb{Z}_{\ge 0}$（如 $B_i = \lfloor\sqrt{W_{2,i} X_2}\rfloor$，因內積為整數，**floor `isqrt` 已足夠**，或離線存 `N_i = ceil(sqrt(W2_i))` 乘線上單次 `X = ceil(sqrt(X2))` 免去逐列 `isqrt`！）為精確非負整數時，因標準 IEEE-754 RNE $\text{fl}_{32}$ 在 $\mathbb{R}_{\ge 0}$ 上全域單調且對正負號對稱，只要禁止 `fast-math`/非預期 FMA、使用原參考 kernel 產生的 `F32` gate activation $h_i$，直接用原生非負 RNE 計算：
     $$U_i = \text{fl}_{32}\big(\text{fl}_{32}(B_i) \otimes_{32} s^{\text{eff}}_{U,i}\big), \qquad B_{a,i} = \text{fl}_{32}(|h_i| \otimes_{32} U_i)$$
     即已滿足 $|a_i| \le B_{a,i}$（**無需額外呼叫 `nextafterf`！** `nextafterf` 僅在使用非精確浮點近似算 $B_i$ 時才需要）。再檢查所有中間值與 $L$ 為 `finite`、$s_L = \text{fl}_{32}(L / 127.0\text{f}) > 0$（IEEE-754 除法在分子 $\ge 0$、分母 $> 0$ 包含次正規數時仍保持單調！）、$B_{a,i} < L$ 且 $\text{fl}_{32}(B_{a,i} / s_L) < 0.5\text{f}$，即嚴格保證 canonical `std::round` 為 `0`（注意特例：若 $h_i = 0$ 但 $U_i = \infty$，不可將 $0 \times \infty = \text{NaN}$ 誤判為 `0`）。

#### 勘誤 2：修正 §0–§1 硬體參數與四個備選方向的過度簡化表述
1. **`6.4 GB/s`、`63.4% PE idle` 與 `99.5%` 感測能耗**：
   - `6.4 GB/s/chip` 為 AiF 論文於 `CR-read + be-enc` 下的內部管線／`ECCLITE` 目標吞吐，論文**並未公開**獨立 MAC lane 數或次位元組展開後的保證算力上限；故 `63.4% PE idle` 僅為「假設 PE 吞吐對齊 `6.4 GB/s`」之條件 roofline，非實測事實。
   - `99.5%` 感測能耗必須明列分母與來源：僅為 `aif_model_notes` 中 NAND 陣列感測能耗（`18.278 pJ/bit` = `95.8 µJ/64KiB`）對比單純 `PE` 動態功耗（`3.98 mW × 10.24 µs ≈ 0.041 µJ`）之局部比例，**未計入** `ECCLITE`（`45.1 mW ≈ 0.46 µJ`）、ONFI I/O、SSD 控制器與 Host DRAM。
2. **備選 A/B/C/D 成本補正**：
   - **備選 A（`q=4` 廣播 PE）**：共用 `32 KiB/plane` PB latches 須滿足 `4 × input_bytes ≤ 32 KiB`（AiF 原最大 `32 KiB` 單向量模式已佔滿兩 latches），且須另付多 context 地址 mux、輸出 FIFO 與非線性布線面積；晶片 wave $\Delta W_{\max}=0$ 不自動等於完整 native DAG 零延遲。
   - **備選 B（Chip-Slack Promotion）**：**撤去「免費／零延遲」用詞**——$W_c + \Delta W_{c,e} \le \max W$ 僅為晶片感測數條件，新增命令、通道競爭、PE drain 與 `GU -> Down` 相依性仍可能推遲關鍵路徑；且 `NF4 + Res4` 非自動等價原 `W8`，LUT／scale／padding 皆有明確成本。
   - **備選 C（Slice-MoE）**：`Block-m A8` 明確改動數值契約；不同切片的 `A8` scale $s_{a,b}$ 不同，`Down` 不能直接跨切片混加未縮放 `I32`，且切換層輸入 $q_x$ 與切片激活需付 latch 切分或重載成本。
   - **備選 D（Layer-Shared VQ）**：`200 MHz` 下若要每週期解出 `32` 個 `I8` 權重（`d=4` 需 **8 次隨機查表／cycle**），單一單埠 `16 KiB` SRAM 無法支撐，必須付多 bank 衝突處理或多份複製（如 `64–128 KiB/chip`）面積與廣播／載入成本。

#### 勘誤 3：修正 `U3` 熱點見證塊成本與 `Down` 欄位切分的 `× K` 流量／精確性限制
1. **`U3`（`[Gate + Up_Tile0]` 融合）**：保留「消除一個獨立 witness 屏障」的優點，但**撤去「100% 必要／零浪費」之絕對表述**——`TRAIN` 熱點在測試 token 上仍可能全數可證為零（或 $L_0$ 偏小觸發 fallback），且 `Up_Tile0` 附加於 Phase 1 的額外頁數、wave 進位（ceil）與關鍵路徑延遲，以及同構排列後的實體 block 對齊／spare 容量都必須如實計費。
2. **`Down` 欄位切分之輸出流量修正（每專家而非每層！）与整數精確性**：
   - §2.4 第 131 行寫「全層匯出 `64–128 KiB/layer`」**漏乘了活躍專家數 $K$（及 `I32` 4B vs 2B）**！
   - 在 canonical `W8A8` 下，每個專家 $e$ 有各自的私有激活 scale $s_{a,e}$、每列權重 scale $s^D_{w,e,r}$ 與路由權重 $\alpha_e$：**不同專家之間絕不能先混加 `I32` partials**；且同一專家若將欄位切給 $g_{\text{own}}$ 顆晶片，若各晶片先轉 `F32` 縮放再相加，通常**不再與 canonical 單次 `I64 -> F32` 逐位元相等（not bitwise-exact）**！
   - 若要求 bitwise-exact 且將單一專家 `Down` 欄位散在 $g_{\text{own}}$ 顆晶片、每顆負責全部 $H$ 列，則每層需匯出 **`K × g_own × H × 4 B`** 的 raw `I32` partials（例如 `OLMoE` $K=8, H=2048$ 若設 $g_{\text{own}}=16$，每層高達 **`1 MiB`**，光 PCIe 即耗 `128 µs`！）。
   - **正確幾何空間**：`Down` 不應強迫 $g_{\text{own}}=16$ 全欄切分，而應在 **(a) 小 owner 數（如 `g_own = 1` 或 `2`，單晶片內即可完成單專家完整 `I32` 內積並直接輸出最終縮放值！）** 與 **(b) 混合 `R_output × C_input = 4 KiB`（或 `16 KiB`）二維分塊**（如 `critic_sol` 所提 $R_{\text{out}}=256\text{..}512, C_{\text{in}}=8\text{..}16$，折衷共零列數與累加器大小／匯出流量）之間合法搜尋，且 **`B0` 與本候選各自選取其最快合法 `Down` 佈局比較**，不強迫基準使用劣質佈局。
   - 另修正 §2.4 表中 `24/16 = 1.5`、`44/16 = 2.75` 用詞：此為全盤總 waves 除以 16 的**平均 wave 工作量**，並非每顆晶片的合法整數 wave 數。

#### 勘誤 4：修正相對強基準 `B0` 的分母公式與「普遍適用 vs. 普遍加速」範圍
1. **相對分母修正**：在三投影等成本理想模型中，設強基準 `B0` 已省 `Down` 比例 $d$、本候選額外省 `Up` 比例 $p$（扣除額外開銷前），則 `p/3` 僅為相對原稠密三投影的絕對工作量降幅；**相對強基準 `B0` 的理想工作量降幅為 `p / (3 - d)`**（理想加速比為 `(3 - d) / (3 - d - p)`；當 $d > 0$ 時，$p/(3-d) > p/3$）。實際收益仍以合法 NAND pages/waves 與 native DAG critical path 為準。
2. **適用範圍修正**：使用者要求「普遍適用有／無 shared expert」意指**方法與正確性契約不依賴 shared expert 存在**（在無 shared 與有 shared 的 MoE 上皆合法且具代表性收益），**並非要求地球上每一個模型都必須滿足 `L/σ >= 100` 或全部獲得同等加速**。先前將「5 個模型全部 $L/\sigma_u \ge 100$」列為存活前提屬過度嚴苛，予以修正。

---

### 3.2 挑戰與剖析新界：同列共享輸入之 `Gate-Conditioned Projected Cauchy`

Root 提出待審查的新廉價確定性界：因同專家第 $i$ 列之 `Gate` 權重 $w_g \in \mathbb{Z}^H$ 與 `Up` 權重 $w_u \in \mathbb{Z}^H$ 作用於**完全相同的量化輸入 $x = q_x \in \mathbb{Z}^H$**，定義全整數純量：
$$G_i = w_g^\top w_g, \quad V_i = w_u^\top w_u, \quad C_i = w_g^\top w_u, \quad X = x^\top x, \quad d_i = w_g^\top x, \quad u_i^{\text{int}} = w_u^\top x.$$
由三向量 $(w_g, w_u, x)$ 之 Gram 矩陣半正定性（投影 Cauchy-Schwarz），恆有：
$$(G_i u_i^{\text{int}} - C_i d_i)^2 \le (G_i V_i - C_i^2)(G_i X - d_i^2).$$

我們從**幾何緊度條件、溢位與硬體開根號成本、以及 `raw-I32` vs `F32` 反推區間**三方面進行嚴格檢驗：

#### 1. 何時能顯著收緊？（緊度收縮因子 $\eta_i$ 的精確解析）
當 $G_i, V_i, X > 0$ 時，令 $\rho_{gu,i} = \frac{C_i}{\sqrt{G_i V_i}} \in [-1, 1]$ 為同神經元 `Gate`–`Up` 權重夾角餘弦，$\rho_{gx,i} = \frac{d_i}{\sqrt{G_i X}} \in [-1, 1]$ 為 `Gate` 與輸入之夾角餘弦。則整數絕對值上界為：
$$|u_i^{\text{int}}| \le B^{\text{proj}}_i = \left\lfloor \frac{|C_i d_i| + \left\lfloor\sqrt{(G_i V_i - C_i^2)(G_i X - d_i^2)}\right\rfloor}{G_i} \right\rfloor \approx \sqrt{V_i X} \cdot \underbrace{\Big( |\rho_{gu,i} \rho_{gx,i}| + \sqrt{1 - \rho_{gu,i}^2}\sqrt{1 - \rho_{gx,i}^2} \Big)}_{\eta_i \in [0, 1]}.$$
- **好消息（永遠不劣於 Global-L2，且強相關時可消去 $\sqrt{H}$！）**：
  - 因 $\eta_i = \cos(|\theta_{gu} - \theta_{gx}|) \le 1$，取 $\min(B^{\text{L2}}_i, B^{\text{proj}}_i)$ 恆合法；
  - 若真實訓練出的 SwiGLU 神經元具有強同向／反向對齊（例如 $|\rho_{gu,i}| \to 1$），在負 `Gate` 區間（$|\rho_{gx,i}| \approx \frac{|g_i|/\sigma_g}{\sqrt{H}} \ll 1$）有 $\eta_i \to |\rho_{gx,i}| \propto 1/\sqrt{H}$，能直接消去 $\sqrt{H}$ 鬆弛！
- **嚴格限制（弱相關時收效甚微）**：
  - 在 $d_i \approx 0$（$\rho_{gx,i} \approx 0$）附近，$\eta_i \approx \sqrt{1 - \rho_{gu,i}^2}$：欲收緊 $2\times$（$\eta_i \le 0.5$）必須 $|\rho_{gu,i}| \ge \mathbf{0.866}$！若真實權重僅有中低相關 $|\rho_{gu,i}| \le 0.3$，則 $\sqrt{1 - 0.09} = 0.954$，**僅收緊 `< 4.6%`**！
  - **結論**：此界是否有奇效，100% 取決於真實 MoE 模型 checkpoint 中同神經元 $(w_{g,i}, w_{u,i})$ 是否演化出高 $|\rho_{gu,i}|$ 子群——這是一個**極易在 Stage-0 直接量測權重即獲得確定答案**的實證問題！

#### 2. `128-bit` 溢位陷阱與 `binary64`（53-bit）拆根號免 `u128 isqrt` 技巧
- 對 $H \le 4096$、$w \in [-128,127]$、$x \in [-127,127]$：$G_i, V_i \le 4096 \times 128^2 = 2^{26}$，$|C_i| \le 2^{26}$，故離線存 `(G_i, V_i, C_i)` 為 **`3 × 4B = 12 B/row`**（$H=2048$ 時僅佔 `Up` 權重的 **`0.586%`**！）。
- **溢位注意**：$G_i V_i - C_i^2 \le 2^{52}$ 且 $G_i X - d_i^2 < 2^{52}$ 各自皆可精確放進 `uint64_t`，但兩者相乘最高達 **$2^{104}$，會溢位 `uint64_t`**！
- **免 `128-bit isqrt` 的精確硬體／CPU 作法**：
  1. 因 $2^{52} < 2^{53}$（IEEE-754 `double` 尾數精確度為 **53 bits**！），$G_i V_i - C_i^2$ 與 $G_i X - d_i^2$ 轉為 `double` **零捨入誤差**，且標準 IEEE-754 硬體 `sqrtsd` 對 `double` 保證正確捨入！
  2. 更省線上算力版：離線直接預存 $R_{w,i} := \lceil\sqrt{G_i V_i - C_i^2}\rceil \in [0, 2^{26}]$（`uint32_t` 4B，取代 $V_i$，總 metadata 仍為 `(G_i, C_i, R_{w,i})` = **`12 B/row`**！若另留 `2B` `N_i = ceil(sqrt(V_i))` 亦僅 `14 B/row`），線上僅需對單一 `uint64_t` 值 $G_i X - d_i^2 < 2^{52}$ 做一次硬體 `double sqrt` + 整數上取整校验得 $R_{x,i} = \lceil\sqrt{G_i X - d_i^2}\rceil \le 2^{26}$，再做一次 `64-bit` 整數乘法 $R_{w,i} \cdot R_{x,i} \le 2^{52}$，**完全免除 `128-bit` 開根號**且 100% 保守安全！

#### 3. `raw-I32` ABI 成本 vs. 由 `F32/FP16` pre-SiLU `g_i` 保守反推整數區間
- **路徑 A（新增 `raw-I32` Gate 輸出 ABI）**：需明列為 AiF 新硬體介面，且 `Gate` 匯出流量由 `2B`（FP16）倍增為 `4B/neuron`（例如 `OLMoE` $K=8, I=1024$ 每層 `Gate` 輸出由 `16 KiB` 增至 `32 KiB`）。
- **路徑 B（不改 `raw-I32` ABI，由 Host 收到的線性 `Gate` 浮點值 $g_i$ 反推區間 $[d_{\min,i}, d_{\max,i}]$）**：
  - 注意 AiF 在 Host 執行 `SiLU`，故 Host 擁有 pre-SiLU 的 $g_i = \text{fl}(d_i \cdot s^{\text{eff}}_{G,i})$（若在晶片內先過 `SiLU` 則不可行，因 $\text{SiLU}$ 在 $(-0.2785, 0)$ 雙值不可逆！）。
  - 只要 $s^{\text{eff}}_{G,i} > 0$ 且 $g_i$ 有限，即可由 1-ulp 邊界反推包含真實整數 $d_i$ 的閉區間 $[d_{\min,i}, d_{\max,i}] \cap \mathbb{Z}$，並代入區間保守上界：
    $$|C_i d_i| \le |C_i| \max(|d_{\min,i}|, |d_{\max,i}|), \qquad G_i X - d_i^2 \le G_i X - d_{\text{absmin},i}^2,$$
    其中若 $0 \in [d_{\min,i}, d_{\max,i}]$ 則 $d_{\text{absmin},i} = 0$，否則為 $\min(|d_{\min,i}|, |d_{\max,i}|)$。此法免去 `raw-I32` ABI 與流量翻倍，Stage-0 可直接對照路徑 A/B 的緊度差（`F32` 下幾乎為零，`FP16` 下約 `0.1%`）。

#### 4. 附記：原生 Clamp 模型（`gpt-oss` 等）與 `A4/A6` 之邊界區隔
- 若模型原生帶有 `Up` 截斷（如 `gpt-oss` 的 `clamp(up, -7, 7) + 1` 上界為 `8`，且具 `1.702` gate slope 與 bias），可直接提供極緊的原生 $U_i$；但本專案現有模型（`G4`/`DeepSeek-V2-Lite`/`OLMoE`/`Qwen3`/`Mixtral`）均無此原生 clamp，絕不可人為加 clamp 篡改模型或擅自下載新模型。
- 同理，`W8A4`/`W8A6`（如對稱 `A4` 將零死區由 $L/254$ 放寬 `18.1×` 至 $L/14$）屬於**更改數值與品質契約**的另一近似問題（且強基準 `B0` 亦須同享 `A4` `Down` 稀疏度），不可混入本輪 `exact canonical W8A8` 主張。

---

### 3.3 核心決策修正：為何**應該**在 `19:58 UTC` 後執行一次有界、付費的 `TRAIN-Only` Stage-0 實驗？

正面檢視所有數學與文獻證據後，我們**撤回上一區塊「僅憑高斯假設傾向 thought-reject」的保守建議，改為明確建議：在 `19:58 UTC` 審查通過後，投入一次嚴格有界、全成本付費的 `TRAIN-Only` Stage-0 數值實驗**。具體必要性理由如下：

1. **無邏輯矛盾，亦無已發表先例支配（No Logical or Prior-Art Dominance）**：
   - **先例核對**：`CATS`、`FloE`、`Park et al. (2605.08575)`、`PowerInfer-2`、`SLIM`、`Ripple`、`BitSET` 全部依賴近似門檻或預測器，會改變動態 `W8A8` 輸出；`SnaPEA` 僅針對 ReLU 正負號早停，無法處理非零 `SiLU` 與未知的動態全向量量化尺度 $s_a = \text{amax}/127$；`LLM in a Flash` / `SpQt` / `SpAx` / `B0` 只能省 `Down`，**完全無法在讀 `Up` 前省下任何一頁 `Up`**。因此在「維持 canonical 動態 `W8A8` 逐位元精確下消除 `Up` NAND 感測」這一點上，不存在先例碰撞或理論支配。
   - **基準關係**：近似方法（`CATS` / `Block-A8`）雖是同品質 Pareto 強對照，但不能在邏輯上取代零品質風險的 `exact W8A8` 契約；而同契約強基準 `B0` 仍須支付 100% `Up` 讀取，只要真實模型上扣完所有成本後 $p_{\text{net}} / (3 - d_{\text{B0}})$ 顯著大於零，本候選即具真實淨收益。
2. **存活與否取決於四個未知的真實模型統計量，無法靠紙上手算斷言**：
   - 真實 SwiGLU MoE 的 (i) `Gate` 負尾分佈、(ii) 真實離群比 $L/\sigma_u$、(iii) 同列 $(w_{g,i}, w_{u,i})$ 相關性 $|\rho_{gu,i}|$ 及分塊／離群能量結構、(iv) 共證零神經元聚類後的整頁／整 subpage 共零率，全部是**未知經驗事實**。以未驗證的高斯獨立假設永久凍結全部 10 輪實驗配額，違背「大膽假設、小心求證」原則；用 1 次低成本、可快速證偽的 Stage-0 `TRAIN` 實驗直接裁決，即使證偽也僅消耗 `1/10` 輪（剩餘 `9` 輪充裕），卻能徹底排除或確認這個唯一的 **exact** 稀疏候選！

---

### 3.4 Stage-0（`TRAIN-Only`）必須回答的三個具體問題與保留／淘汰判準

#### 三個由 Stage-0 數據直接回答的具體問題
1. **問題 S1（純量證書覆蓋率與哪一種廉價上界真正有效）**：
   - 在凍結的 `TRAIN` 小樣本（涵蓋無 shared 之 `OLMoE` 與有 shared／細粒度之 `G4`、`DeepSeek-V2-Lite`）上，真實 canonical `W8A8` 的零比例 $z_{\text{true}}$ 為何？在嚴格 F32 單調判準 `fl(Ba / sL) < 0.5f` 下，比較四種明碼標價上界——**(a) `Global-L2`（`2–4B/row`）**、**(b) `Gate-Conditioned Projected Cauchy`（`12B/row`，同時量測真實權重 $|\rho_{gu,i}|$ 分佈並對比 `I32` $d_i$ 與 `F32/FP16` 反推區間）**、**(c) `Block-L2`（如 $B=64/128$）**、**(d) `Outlier-Deflated L2`（`TRAIN` 選固定 $r=16$ 通道）**——各自能將純量 `Up` 認證率 $z_{\text{cert}}$ 推到多少？
2. **問題 S2（`U3` 熱點見證塊的真實回收率與付費浪費率）**：
   - 採用 `TRAIN` 折內（split-A 選列、split-B 測試，嚴禁偷看測試 token）選出的首個物理讀取單位 `Up_Tile0`（1 個 `16 KiB` 頁或 1 個 `64 KiB` wave），在未見 `TRAIN-B` tokens 上 $L_0 / A$ 的分佈為何（出現 $L_0=0$ 或極弱下界的頻率）？`Up_Tile0` 內部實際為真零的「見證浪費率」以及融合於 Phase 1 後增加的實際頁／wave 進位成本是多少？與動態選 1 個最高 `|h_i| N_i` 頁（支付整頁散落讀取）相比孰優？
3. **問題 S3（轉化為真實物理頁／Wave 後相對同能力強基準 `B0` 的淨減量 `p_net / (3 - d_B0)`）**：
   - 以 `TRAIN-A` 學出同構神經元共證零排序 $\pi$、於 `TRAIN-B` 實測兩種明示硬體粒度（**雙方控制組 `B0` 完全同享相同粒度與各自最佳合法 `Down` 佈局，且如實扣除 metadata 頁／DRAM、witness 頁、split GU 進位與 `K × g_own × R_out × 4B` 匯出流量**）：
     - **原生層級（`16 KiB` page / `64 KiB` 4-plane wave）** 與
     - **次頁層級（`4 KiB` 4-bank SOML 或獨立 plane AMP + 合法 `Down` 二維／小 owner 分塊）**，
     本候選扣除一切額外成本後的物理讀取與 `max_chip` waves 相對 `B0` 究竟淨省多少？

#### 明確的保留／淘汰判準（Stage-0 Pre-Registered Gate）
- **立即淘汰判準（觸發任一條即判定第 1 輪實驗失敗並停止重型實作，保留剩餘 `9 / 10` 輪轉攻下一候選）**：
  1. **純量天花板不足**：即便給定 oracle `amax`（$L=A$）並取上述四種廉價上界之最小值，所測模型之平均純量 `Up` 證書率仍全面 `< 15%`（不足以在多列成頁與扣費後支撐雙位數物理節省）。
  2. **物理淨收益不足或喪失通用性**：在 `TRAIN` 折外評估並如實計入 witness、metadata、拆 phase 進位與 `Down` 匯出成本後，無論在原生 `16 KiB`/`64 KiB` 或公平同享之 `4 KiB` SOML 下，相對同硬體強基準 `B0 (Full-GU + Post-A8 Exact-Zero Down)` 的淨物理工作量／關鍵路徑 wave 降幅 $\frac{p_{\text{net}}}{3 - d_{\text{B0}}}$ 在所有測試模型上皆 **`< 8%`**（連樂觀門檻都達不到 `10%` 目標），**或僅在有 shared expert 模型有效而在無 shared 模型（如 `OLMoE`）`< 5%`**。
- **保留晉級判準**：
  - 在 `0` 次數值不一致（100% 匹配 canonical `W8A8` 參考）前提下，至少在無 shared 與有 shared 代表模型（或明確不依賴 shared 的普遍模型條件）上，對同硬體強基準 `B0` 取得 **`≥ 8%–10%` 的付費後淨物理讀取／wave 降幅**（且對原稠密 `W8A8` 顯著更高），則晉級至完整 native critical-path DAG、CPU/控制器開銷與凍結外部資料集驗證。


---

## 4. 第三區塊最終勘誤與 Stage-0 凍結草案終審（`19:55:36Z` 起，對齊 `gate_zero_stage0_protocol.draft.json`）

> **對 §3.3 / §3.4 之最終明確勘誤（完整保留上文歷史紀錄，以此節為準）**：
> 1. **修正先例與 `Up` 跳過歸因錯誤**：§3.3 誤寫先例只能省 `Down` 或已證無先例碰撞，予以撤回更正——**`CATS`、`Park et al. (2605.08575)` 等近似方法以及 `PowerInfer-2` 的 gate-exact-zero 機制本身皆能省 `Up`**；即使在 `SiLU` + canonical `W8A8` 下，當實際 GGML gate 激活 `h_i == 0`（且 `u_i` 有限）時，**不需任何 witness $L$ 或上界 $U_i$ 的零見證基準本身就能精確跳過 `Up` 與 `Down`**。本候選真正有限的待測差異僅限於：**對非零 `SiLU` gate（`h_i != 0`）利用 witness `amax` 下界與確定性 `Up` 上界證明 canonical 動態 `A8` 量化為零**（且不能預先宣稱已無任何先例重疊）。
> 2. **撤回自創硬門檻（`15% / 8% / 5%`）與超範圍實驗設計**：§3.4 所列 `15%`、`8%`、`5%` 硬淘汰門檻並非 Root 協定，**全數撤回**（使用者的 `10%` 僅為報告指標 flag，非硬性 fail 門檻）。正確建議嚴格限定為 **有界證偽測試（bounded falsification）**，既非宣稱 paper-go，亦非全局 no-go；不新增任何實驗門檻或擴張 scope。
> 3. **嚴格對齊 `gate_zero_stage0_protocol.draft.json` 凍結範圍**：
>    - **模型與資料**：僅測 `deepseek-v2-lite`（含 `layer0` dense、`layers 1..26` routed 與真實合併之 `I=2816` shared 分支）與 `olmoe-1b-7b`（全 routed），取凍結 `TRAIN32` 前 8 個 jobs（`prompt_tokens_cap=128`, `decode_steps=8` greedy）；**不做 `G4`、不擬合神經元重排 $\pi$、不做動態最高 bound witness、不做 `F32` 反推 `d` 的 ABI 實驗**。
>    - **Secondary Deflation 擬合規則**：每模型／每層僅用前 4 個 `TRAIN` jobs 的 `decode8` 位置，對每個層／token 的**共同輸入 `q_x` 只計一次**（不隨所選 routed 或 shared 專家重複加權！）選出 top-16 `sum(q_x[j]^2)` 座標並凍結，分開報告前 4 與後 4 jobs，不含 prefill 能量。
>    - **付費固定見證頁與完整雙 Mask 結算**：固定取原始 row-major 邏輯偏移 `0` 的首個 `16 KiB` `Up` 頁（$H=2048 \implies$ 前 `8` 整列；不可稱作 `64 KiB` 4-plane wave），**強制整頁標記為已讀（READ）**。由於該頁內可能原本含有 `h_i == 0` 可被零見證基準跳過的列／次頁，付費見證頁可能比 `h == 0` 基準**多讀頁**！因此絕不可只數 `certified & (h != 0)`，必須分別建立 **`h == 0` 基準完整 mask** 與 **候選 `(h == 0 OR certified)` 且首頁強制 READ 之完整 mask**，直接比較兩者在原始順序下完整 `16 KiB`（8 列）與假想 `4 KiB`（2 列）邏輯讀取單位數的淨差額。

---

### 4.1 最後 5 分鐘終審：`gate_zero_stage0_protocol.draft.json` 剩餘證明／數值／價值風險核查

經逐條核對 `gate_zero_stage0_protocol.draft.json`，沒有阻止執行此有界證偽測試的邏輯矛盾（且兩模型 $H=2048$、$I \in \{1024, 1408, 2816, 10944\}$ 皆被 `8` 整除，原始列序下 `16 KiB` 8 列與 `4 KiB` 2 列均無跨頁殘列），但實作與解讀 Stage-0 時須嚴防以下 **5 個具體證明／數值與價值風險**：

1. **`h_i == 0` 分支的 `0.0f * Inf = NaN` 有限性保護（證明健全性）**：
   - 在 `h_i == 0.0f`（含極負 gate 在 `ggml_vec_silu_f32` 下溢為 `-0.0f`）時，唯有在真 `u_i` 保證 `finite` 下，canonical 乘積 `h_i * u_i` 才必為 `±0.0f` 而非 `NaN`。因 $|M_i| \le 128 \cdot 127 \cdot H = 33,032,192 < 2^{25}$ 恆有限，實作 `h == 0` 基準 mask 與候選 mask 的 `h == 0` 分支時，仍須檢查 `isfinite(sw_i) && isfinite(sx)` 且 `scale_output(128*127*H, abs(sw_i), abs(sx))` 有限（候選分支則直接檢查 `isfinite(U_i)`），否則 fallback 為 READ。
2. **`ggml_vec_silu_f32` 與 CPU 整數參考之單一激活源一致性（數值等價風險）**：
   - GGML 的 `ggml_vec_silu_f32`（常含 FP16 查表或 SIMD 多項式 `exp`，在負尾會提早下溢成 `0.0f`）與標準庫 `x / (1.0f + std::exp(-x))` 在負尾及零點附近並不逐位元相同！協定第 110 行之 canonical CPU `W8A8` 參考與第 178 行證書判定**必須呼叫同一個 pinned `ggml_vec_silu_f32` 實例產生同一份 `actual_h`**，不可一邊用 `std::exp` 一邊用 `ggml_vec_silu_f32`，否則會製造假 `h == 0` 或參考不一致。
3. **投影 Cauchy 之 C++ 有號整數除法截斷陷阱與 `128-bit` 無號化（證明健全性）**：
   - 在 $(G_i u - C_i d_i)^2 \le (G_i V_i - C_i^2)(G_i X - d_i^2) =: P_{128}$ 中，設 $S = \lfloor\sqrt{P_{128}}\rfloor$（精確 `u128` 整數下取整根號）。若在 C++ 對負數區間端點直接用有號 `/`（向零截斷而非向 $-\infty$ 下取整），當 $C_i d_i + S < 0$ 時會算錯端點方向。安全無漏洞寫法為全轉無號 `__uint128_t` 計算整數絕對上界：
     $$B^{\text{proj}}_i = \left\lfloor \frac{|C_i d_i| + \lfloor\sqrt{P_{128}}\rfloor}{G_i} \right\rfloor \quad (G_i > 0, \; G_i V_i \ge C_i^2, \; G_i X \ge d_i^2),$$
     且對每個未跳過列順帶斷言真 `u_dot` 落於精確區間內以自檢無溢位。
4. **16 座標 Deflation 之補空間平方和整數直接累加（防浮點相減誤差）**：
   - 計算 `secondary_deflation` 時，$W_{2,\overline{S_{16}},i} = \sum_{j \notin S_{16}} w_{u,ij}^2$ 與 $X_{2,\overline{S_{16}}} = \sum_{j \notin S_{16}} q_{x,j}^2$ 必須在 `uint64_t` 對補集直接精確求和（或精確整數相減 `W2_total - W2_S16`），再算 $|u_{S_{16}}| + \lfloor\sqrt{W_{2,\overline{S_{16}},i} X_{2,\overline{S_{16}}}}\rfloor$，不可經由浮點數相減產生負值或小於真值的捨入誤差。
5. **Stage-0 通過後的兩道真實論文價值門檻（避免誤把 Stage-0 當 Paper-Go）**：
   - 協定已正確界定 Stage-0 的 `16 KiB` / `4 KiB` 為原始列序下的**邏輯讀取單位機會（非 native wave 或延遲）**。即使 Stage-0 顯示非零邏輯機會，後續仍須跨過：(a) 在真實 16 晶片 4-plane 同步 wave / SOML block-bank 衝突、拆 phase 命令屏障、metadata 讀取與 `Down` 部分和匯出流量下仍對同硬體 `B0` 具實質關鍵路徑收益；以及 (b) 在投稿層面面對 `CATS` / `Park et al.` 等近似 gate-first 方法時，證明「零品質風險之 exact `W8A8` 契約」相對同品質近似門檻具備不可替代的系統或精度價值。
