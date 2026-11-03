# Round 3 候選思考 Block 01：原生 Mixed-Q4_K/Q6_K 256-Input 窗格之數值合約、存活狀態與 NAND 物理映射設計備忘錄

**記錄標識**：`round03-thinking/gemini_formal_block01_fresh_numeric_resource.md`（由文字回覆直接交付，不呼叫任何外部工具與檔案讀寫）  
**時間與計時宣告**：Root formal start 10:35:14 UTC；前序 exploration block01/02（384s）、舊 formal 中斷與 CLI 權限失敗均記為 0 credit。本次僅以本 CLI 區段之實質有效分析開始計時；未滿 1800s 實質推導與 Critic Valuegate 前，嚴禁宣稱通過時間門檻或申請實驗採納。  
**分析立場**：嚴格待否證候選研究，非實驗採納預設立場。本段落拒絕任何「必勝預言」，不使用無庫估計之面積常數，不忽視量化/形狀 ceiling 與跨晶片屏障。

---

## 模組 A：Generic 8-Lane 數值合約、浮點非結合性紙上反例與分組邊界

### 1. [提供的 Source 事實] 基準事實標記
- **Q4_K 結構**：每 256 個權重佔 144 Bytes，包含 128B 的 4-bit nibbles、12B 的 6-bit subgroup scales/mins，以及 2 個 FP16 基準值（$d, d_{\min}$）。權重分為 8 個 32-weight subgroups，反量化公式為：
  $$w = d \cdot s \cdot q - d_{\min} \cdot m$$
  其中 $s, m \in [0, 63]$。
- **Q6_K 結構**：每 256 個權重佔 210 Bytes，包含 128B low 4-bit、64B high 2-bit、16B signed 8-bit scale，以及 1 個 FP16 基準值 $d$。每 16 個權重為一組，量化碼 $q \in [-32, 31]$。
- **Q8_K 輸入結構**：每 256 個 activation 佔用 1 個 signed-max scale $d_{in}$（**注意：GGML 規範中 $d_{in}$ 可帶負號**）、256 個 int8 量化碼 $q_8 \in [-128, 127]$，以及每 16 個元素一組的預計算整數和 `bsums`。
- **Generic Dot ABI 契約**：GGML generic `vec_dot_q4_K_q8_K` 與 `vec_dot_q6_K_q8_K` 在每個 output row 的點積累積過程中，**嚴格維持 8 條獨立跨 256-block 的 F32 累加器通道（$sumf[0..7]$）**。Q4_K 另有額外的 $d_{\min}$ 修正項累加。最後在區塊結尾才執行 8 條通道的規約加法。

---

### 2. [本段推導與數學證明] 小型紙上 F32 非結合性反例

#### 命題：
令整數內積預累加結果在數學上嚴格等價：
$$\sum_{j} (w_j \cdot a_j)_{\mathbb{Z}} = \sum_{l=0}^{7} \sum_{k} (w_{l,k} \cdot a_{l,k})_{\mathbb{Z}}$$
若硬體為了提升 LUT 效率或減少累加器，將原本屬於不同 float 通道 $l$ 的整數積提前合併為單一整數後再轉為 F32，則**不能保證結果與 generic 8-lane 參考實作位元完全一致（bit-exact）**。

#### IEEE 754 Float32 精簡紙上反例：
標準 IEEE 754 單精度浮點數（F32）具備 24 位元有效位數（23 位元 mantissa + 1 位隱含位元）。當浮點數值大小達到 $2^{24} = 16,777,216$ 時，其最小精度單位（ULP, Unit in the Last Place）為 $1.0$。當數值介於 $2^{24}$ 至 $2^{25}$ 之間時，最小奇偶捨入（Round-to-Nearest, Ties-to-Even）使得奇數整數無法精確表示。

設某 256-block 運算過程中，經 scale 放大後的局部中間浮點數累加狀態為：
- 通道累加器當前累積值 $A = 2^{24} = 16,777,216.0$
- 當前待累加的兩個整數點積項經 scale 縮放後分別為 $B = 1.0$ 與 $C = 1.0$

**情境 1：硬體提前合併整數（Pre-merge to single F32 lane）**
整數項先在整數域合併：$B_{\mathbb{Z}} + C_{\mathbb{Z}} \rightarrow (B+C) = 2.0$。
將合併結果加入累加器 $A$：
$$\text{Sum}_1 = A + (B + C) = 16,777,216.0 + 2.0 = 16,777,218.0$$
在 F32 中，$16,777,218.0 = 2^{24} + 2$，可精確由有效位數表示。

**情境 2：Generic 8-lane 順序累積（Sequential Accumulation on separate steps）**
依 generic 迴圈順序，逐項加至累加器：
$$\text{Step 1: } A + B = 16,777,216.0 + 1.0 = 16,777,217.0$$
在 IEEE 754 預設 Ties-to-Even 規則下：$16,777,217.0$ 位於 $16,777,216.0$ 與 $16,777,218.0$ 的正中間（halfway），因最低有效位取偶數，**捨入至 $16,777,216.0$**！
$$\text{Step 2: } (A + B)_{\text{rounded}} + C = 16,777,216.0 + 1.0 \rightarrow \text{捨入至 } 16,777,216.0$$
因此：
$$\text{Sum}_2 = 16,777,216.0 \neq 16,777,218.0 = \text{Sum}_1$$
**數值差異達 2 ULP**。

#### 結論：
只要累加順序改變或提前 cross-lane 聚合，即使中間純整數算術完全無溢位，在乘上 scale 轉入 F32 累加器時即刻產生差異。因此，**「整數算術等價」絕非「F32 bit-exact」的充分條件**。任何聲稱維持 GGML generic numeric contract 的 LUT 引擎，必須嚴格維持其 8 條獨立 F32 通道及其特定累加拓撲。

---

### 3. [本段推導與架構缺口] 不改數值合約的 LUT 分組邊界與狀態分流

為確保符合 generic 8-lane 契約，我們分析 activation 分組查表的合法拓撲邊界：

```
Generic 256-block 拓撲 (Q4_K):
256 weights -> 8 個 32-weight subgroups (各有獨立 6-bit scale s 與 6-bit min m)
Generic 迴圈步進：8 條 float 通道交錯 (stride-8: lane l 負責 k = l, l+8, l+16, l+24, ...)
```

#### A. Q4_K 的 $g4$ 分組限制
- 若採用 4 個 activation 合併查表（$g4$），為避免 cross-scale 與 cross-lane 污染：
  1. 這 4 個 weight 必須處於**同一個 32-weight subgroup**（共享相同的 $s$ 與 $m$）。
  2. 這 4 個 weight 必須對應到**同一個 F32 累加器通道 $l$**。
- 因 generic 實作中 8 條通道為 stride-8 交錯，若將記憶體中連續 4 個 activation（$a_k, a_{k+1}, a_{k+2}, a_{k+3}$）合併為一張表，其對應的權重分別屬於 lane 0, 1, 2, 3。**直接查表將強制把 4 條獨立 lane 在整數域融合，徹底破壞 generic 8-lane 契約**！
- **唯一合法的 Generic $g4$ 構造**：
  必須採用 **stride-8 分組**：即群組 $g4$ 由 $\{a_{l}, a_{l+8}, a_{l+16}, a_{l+24}\}$ 組成，剛好涵蓋該 subgroup 內屬於 lane $l$ 的全部 4 個元素！
  - 每個 32-weight subgroup 恰好有 8 條 lane，每條 lane 恰好分配 4 個權重。
  - 每個 256-block 共有 $8 \times 8 = 64$ 個獨立的 $g4$ 子表。
  - 每個 $g4$ 表具有 $2^4 = 16$ 項，每項為 16-bit 符號整數，大小為 32 Bytes。
  - 64 個表總容量：$64 \times 32\text{ B} = 2048\text{ Bytes} = 2\text{ KiB}$。

#### B. Q6_K 的分組限制（退化為 $g2$）
- Q6_K 每 256 權重被劃分為 16 個 16-weight subgroups，每個 subgroup 擁有獨立的 signed 8-bit scale。
- 在 stride-8 交錯下，每個 16-weight subgroup 落在特定 lane $l$ 的元素數量僅有：
  $$\frac{16\text{ weights}}{8\text{ lanes}} = 2\text{ weights/lane}$$
- **推導結論**：在維持 generic 8-lane 與 subgroup 邊界的前提下，Q6_K **根本無法組成 $g4$ 分組**！其最大合法分組量子僅為 **$g2$（2-input group）**：
  - 由 $\{a_{l}, a_{l+8}\}$ 組成。
  - 每個 $g2$ 表僅有 $2^2 = 4$ 項（若對 low 4-bit 查表則為 16 項，但組合項僅 2 個權重）。
  - 這意味著 Q6_K 採用 LUT 查表時，查表共享度（amortization）大幅腰斬，每 2 個權重就需一次 lookup，查表效率優勢在此被強烈壓縮。

#### C. 未驗證假設與架構缺口
- **[未驗證假設 1]**：假設硬體調度器在建表時，能以零開銷（Zero-overhead）完成 activation 從連續向量佈局（contiguous layout）到 stride-8 分組佈局的轉置（gather/shuffle）。若此 shuffle 需額外暫存器或週期，建表延遲將侵蝕計算時間。
- **[未驗證假設 2]**：假設 Q8_K 的負 scale（signed $d_{in}$）可在進入 F32 轉換時直接以符號位異或（sign-bit flip）修正，不破壞前段整數累加。

---

## 模組 B：$T256$ 窗格 $\times R$ Active Rows 同時存活狀態與字節帳

### 1. 存活狀態構成與儲存屬性公式

設目前處理一個 $T256$ activation window（256 個 int8 輸入），同時平行或流水線服務 $R$ 個作用中輸出 row（$R \in \{128, 256, 448\}$）。

| 狀態項目 | 符號 | 尺度公式 | 共享/私有屬性 | 存取衝突與 Port 特性 |
| :--- | :--- | :--- | :--- | :--- |
| **Activation LUT** | $S_{\text{LUT}}$ | 64 表 $\times$ 16 項 $\times$ 2B = **2,048 B** | 全部 $R$ rows 共享廣播 | $R$ 個 row 同時存取不同 index；需多副本或多 Bank 避免 port 衝突 |
| **原始 Q8 輸入與 Scales** | $S_{\text{act}}$ | 256B ($q_8$) + 32B (bsums) + 4B ($d_{in}$) = **292 B** | 全域共享 | 僅建表時讀取，建表完成後不佔 ALU 關鍵路徑 port |
| **8-Lane FP32 累加器** | $S_{\text{carry}}$ | $R \times (8 \text{ lanes} \times 4\text{ B}) = \mathbf{32 R\text{ B}}$ | 每個 Row 私有 | 每個 Row 的 8 條通道在 subgroup 完成時累加，需專屬寫入埠 |
| **Q4_K Min 修正累加器** | $S_{\min}$ | $R \times (1 \text{ accumulator} \times 4\text{ B}) = \mathbf{4 R\text{ B}}$ | 每個 Row 私有 | 每個 32-subgroup 更新一次，需與 scale FMA 同步 |
| **權重與 Metadata Staging** | $S_{\text{w\_buf}}$ | 依排程：若串流則為 $1 \sim 2$ 條 codeword；若並行則 $R \times 144\text{ B}$ | 管道管線暫存 | 來自 ECC 引擎輸出，需循序填入，不可假設隨機多埠讀寫 |
| **Row 控制與 ID 追蹤** | $S_{\text{ctl}}$ | $R \times 4\text{ B} = \mathbf{4 R\text{ B}}$ | 每個 Row 私有 | 記錄當前 row index、subgroup 計數器與狀態機標誌 |
| **Output FIFO / Staging** | $S_{\text{out}}$ | $R \times 4\text{ B} = \mathbf{4 R\text{ B}}$ | 每個 Row 私有 | 當 256-block 完成最終規約時暫存輸出 F32 結果 |

---

### 2. [本段推導] 紙上字節帳（Paper Byte-Budget Table）

考慮**權重串流模型（Weights Streaming Model）**：$T256$ 活頁窗格（2 KiB LUT）常駐於運算單元本地暫存器/SRAM 中，權重自 NAND Page Buffer 經 ECC 解碼後以 $K_{\text{batch}}$ 寬度串流流經運算單元，同時累積 $R$ 個 Row 的 carry 狀態。

> **嚴格限制**：NAND 64 KiB Page Buffer 為靜態傳輸緩衝區，**絕非隨機多埠 Multi-port SRAM**。運算單元本地必須具備專屬的 SRAM/RF 容納活存狀態。

#### 表 1：不同 $R$ 下的本地 Working Set 總字節帳

| 資源項目 | $R = 128$ | $R = 256$ | $R = 448$ | 備註與計算依據 |
| :--- | :--- | :--- | :--- | :--- |
| **Activation LUT ($g4$)** | 2,048 B | 2,048 B | 2,048 B | 靜態 2 KiB，完全共享 |
| **Raw Activation & bsums** | 292 B | 292 B | 292 B | 靜態 292 B |
| **8-Lane FP32 Carries** | 4,096 B | 8,192 B | 14,336 B | $R \times 32$ B (8 條 IEEE F32) |
| **Min Corrections ($d_{\min}$)** | 512 B | 1,024 B | 1,792 B | $R \times 4$ B (單精度浮點) |
| **Row State & Metadata FIFO** | 1,024 B | 2,048 B | 3,584 B | $R \times 8$ B (RowID + Control + FIFO) |
| **ECC Codeword Staging (2CW)**| 2,048 B | 2,048 B | 2,048 B | 雙緩衝 1 KiB ECC codeword |
| **Q4_K 核心活動狀態總和** | **9,980 B** ($\approx 9.75$ KiB) | **15,604 B** ($\approx 15.24$ KiB) | **24,092 B** ($\approx 23.53$ KiB) | 必須實體存在於 PE 控制器中 |
| **Q6_K 核心活動狀態總和** | **8,956 B** ($\approx 8.75$ KiB) | **14,580 B** ($\approx 14.24$ KiB) | **23,068 B** ($\approx 22.53$ KiB) | Q6_K LUT 僅 1 KiB ($g2$)，無 min 補正 |

#### Port 衝突與 Multi-lane 廣播機制分析：
- 若 PE 配置 $N_{\text{lane}} = 16$ 條平行運算通道：
  - 每個週期有 16 個 weight nibbles 同時作為 address 存取 LUT。
  - **若將 2 KiB LUT 做成單一 1-Port SRAM**：16 條 lane 存取將造成嚴重 serialization，需 16 個 cycle 才能完成一次並行查表，直接使 ALU 空轉率高達 93.75%。
  - **真實硬體合法解法**：
    16-entry 16-bit table 每組僅 **32 Bytes**。與其耗費巨額代價建構 16-port SRAM，最佳結構為**分散式暫存器檔案（Distributed RF Replicas）**或 **16-to-1 MUX 樹**。
    每個 Lane 私有其專屬的 32B MUX 陣列，直接自當前 active subgroup 的 16 個 16-bit 暫存器中選取數值。
    - 32B $\times$ 16 lanes = 512 Bytes 的高頻活頁暫存器。
    - 其餘 56 組表的數值暫存於常規單埠 SRAM 中，依 subgroup 推進順序預先置換（pre-fetch buffer）。
- **結論**：16 lane 不需要 16-port 2KiB SRAM，但代價是需要流水線切換機制的 512B 暫存器檔案與多工選通邏輯。

---

## 模組 C：16 KiB Page / 4-Planes 實體佈局與流水線生命週期

### 1. [本段推導] 實體 Page 封裝候選構造

在標準 NAND Flash 幾何架構下：
- 每個 Chip 包含 4 個獨立 Plane。
- 每個 Plane 的實體 Page 容量為 16 KiB（16,384 Bytes 加上額外 Spare 區間放置 ECC parity）。
- 4-plane 同步讀取 Wave 容量：$4 \times 16\text{ KiB} = 64\text{ KiB}$。

#### 權重區塊對齊構造：
- **Q4_K Block (144 Bytes)**：
  - 若採**嚴格跨頁對齊（Strict Alignment）**：
    $$\lfloor 16,384 / 144 \rfloor = 113\text{ blocks/page}$$
    $113 \times 144 = 16,272\text{ Bytes}$。
    頁尾剩餘 $16,384 - 16,272 = 112\text{ Bytes}$ 作為頁內 metadata 或填充保護。
    一個 16 KiB Page 完整容納 $113 \times 256 = 28,928$ 個權重。
- **Q6_K Block (210 Bytes)**：
  - 若採嚴格跨頁對齊：
    $$\lfloor 16,384 / 210 \rfloor = 78\text{ blocks/page}$$
    $78 \times 210 = 16,380\text{ Bytes}$。
    頁尾剩餘 4 Bytes 填充。
    一個 16 KiB Page 完整容納 $78 \times 256 = 19,968$ 個權重。
- **此對齊構造之巨大價值**：
  直接消除跨頁（Page-crossing）權重拼接邏輯，保證 ECC 解碼單元完成一個 1 KiB 或 2 KiB Codeword 時，其內部所包含的全部量化區塊皆為完備區塊，不需 cross-page reassembly buffer。

---

### 2. [本段推導] 核心管線生命週期時序（Direct MAC vs. LUT Window）

兩種架構共享完全相同的 NAND 感測（$t_R$）、傳輸與 ECC 解碼前端：

```
時間軸 (Timeline) 推進：
[NAND Array Sensing: t_R = 28 us] (4 Planes 同步感測 64 KiB 至 Page Buffers)
       │
       ▼
[NAND Bus Transfer @ NV-DDR3 1.6 GB/s: ~40 us] (串流送入控制器)
       │
       ▼
[ECC Decoding Pipeline @ 6.4 GB/s: 10 us] (產出 corrected weights 串流)
       │
       ├────────────────────────────────────────┬────────────────────────────────────────┐
       ▼ (路徑 1: Direct MAC Engine)             ▼ (路徑 2: LUT Window Engine)
[Nibble Unpacker & Sign Extender]        [4-bit / 2-bit Index Decoder]
       │                                        │
[8-bit x 4-bit / 8-bit MAC Array]        [16-to-1 MUX Table Lookup]
       │                                        │
[Subgroup Int Accumulator]               [Subgroup Int Adder Tree]
       │                                        │
       └────────────────────────────────────────┴────────────────────────────────────────┘
                               │
                               ▼
        [FP16 Scale Multiplication (d * d_in * s)]
                               │
        [8-Lane FP32 Carry Accumulators Update (32B / row)]
                               │
        [Block Finalization: Reduction to F32 Output]
```

#### 生命週期與邊界條件比較：
1. **ECC 必須先於查表**：
   在 Direct MAC 中，若未校正位元發生在權重 mantissa，僅產生局部算術誤差；但在 LUT 引擎中，權重位元即為 table index，未經 ECC 校正的 bit-flip 將導致**查表位址嚴重走偏（Address misdirection）**，讀出完全無關的數值。因此兩者都**必須在 ECC 引擎輸出校正完畢後方可進入計算**。
2. **8-Lane Carry 生命期**：
   - 當處理 Hidden dimension $H=2048$ 時，共需 $2048 / 256 = 8$ 個 activation windows。
   - 每個 Row 的 8 條 FP32 carry 必須跨越這 8 個連續 window 保持活存。
   - 總運算完成後，執行 8 條 float 的加法樹：
     $$\text{Result} = \sum_{l=0}^7 sumf[l] - d_{\min} \cdot \text{Correction}$$
     隨後寫入輸出緩衝區，釋放該 Row 的狀態空間。

---

### 3. [本段推導] 巨觀神經元所有權（Neuron Ownership）與跨晶片成本

在 SwiGLU 結構中：
$$\text{Intermediate } h = \text{SiLU}(W_{\text{gate}} x) \odot (W_{\text{up}} x)$$
$$\text{Output } y = W_{\text{down}} h$$

- **提供的 Source 事實**：
  - Gate 與 Up 投影共享相同的輸入 token $x$（長度 $H=2048$）。
  - Down 投影則以 intermediate activation $h_e$ 為輸入，每個專家（expert）的 $h_e$ 互不相同。
- **神經元所有權權衡（Neuron Ownership Trade-off）**：
  - 若要使 Down 投影的 $T256$ activation quantizer 在本地直接「Ready」（即在本地產生 Q8_K 的 $d_{in}$ 與 256 個 int8 碼）：
    - **構造方案 A（Neuron Co-location）**：將計算 $h_e$ 的對應 256 個 Gate/Up 神經元完全配置在同一個 NAND Chip 節點。
      - *代價*：雖然避開了 activation gather，但該 Chip 產生的 Down 投影輸出僅為 $y$ 的 partial sum（部分和），必須在晶片間執行長度為 $H$ 的跨晶片 All-Reduce / Ring-Accumulate，並支付跨晶片匯流排頻寬與浮點加法結合律順序成本。
    - **構造方案 B（Activation Scatter/Gather）**：Gate/Up 分散計算，但在進入 Down 投影前，必須跨晶片收集完整的 256 個 $h_e$ 元素以計算 $\max |h_e|$ 獲得 scale $d_{in}$。
      - *代價*：引入全域同步屏障（Global Synchronization Barrier），在 256-input 量化未完成前，Down 投影的 LUT 建表完全受阻。
- **結論**：
  絕不存在「單一晶片僅靠 2 KiB 本地暫存即完成整個專家運算」的無代價硬體神經網路。無論如何切分，**非線性啟動函數（SiLU）的浮點精度契約、Q8_K 量化縮放因子的全域依賴，以及跨晶片累加傳輸**，皆是實體架構無法規避的剛性開銷。

---

## 模組 D：可被否證的單一 Trade-off 命題、致命反例與未解關鍵問題

### 1. [本段推導] 可被否證的單一 Trade-off 命題

#### 命題公式化表述：
令每個 NAND PE 在受限晶粒面積 $A_{\text{budget}} \le A_0$ 下，可配置的運算資源上限為 $M$ 個平行 MAC 單元，或等面積替換為容量為 $K \times 32\text{B}$ 的 LUT 查表多工單元與加法樹。
定義 NAND 實體頁讀取與傳輸週期為 $T_{\text{page}} = t_R + t_{\text{trans}} \approx 28\,\mu\text{s} + 40\,\mu\text{s} = 68\,\mu\text{s}$（對應 64 KiB 4-plane wave）。
定義在該 Page 週期內需處理的有效權重總數為 $W_{\text{page}}$（對於 Q4_K，$W_{\text{page}} = 116,480$ weights）。

**命題**：
> 存在一個臨界運算頻寬門檻 $M_{\text{crit}} = \frac{W_{\text{page}}}{T_{\text{page}} \times f_{\text{clk}}}$：
> 1. 當晶片面積與功耗預算嚴格限制 PE 運算能力，使得等效硬體並行度 **$M < M_{\text{crit}}$** 時，運算單元成為全系統唯一瓶頸（Compute-bound），此時 $g4$ LUT 藉由每 4 個權重僅需 1 次檢索與多工選通、免除硬體乘法器陣列的特性，能在相同面積下提供高於 Direct MAC 的等效吞吐量，縮短暴露延遲（Exposed Latency）。
> 2. 當 PE 面積允許配置足夠單元使得 **$M \ge M_{\text{crit}}$** 時，系統完全進入 **NAND 感測與介面瓶頸區間（Sense/Transfer-bound）**。此時計算管線完全隱蔽於 $T_{\text{page}}$ 之後，LUT 查表無法帶來任何端到端延遲縮短；反之，因其額外的 2 KiB 建表開銷、Q6_K 退化為 $g2$ 的低效查表，以及 8-lane F32 嚴格排程約束，**LUT 架構在綜合指標（PPA、數值合約風險、控制複雜度）上嚴格劣於 Direct MAC 搭配標準載入器**。

---

### 2. [本段推導] 致命反例（Fatal Counterexample）

#### 反例情境：Batch Size $B = 1$ 單 Token 推論下的 Q6_K 密集計算
- **條件**：
  在 FFN 的 Down-projection 中，部分重要權重採用 Q6_K 封裝以維持困惑度（PPL）。此時輸入為單一向量（$B=1$），$T256$ activation 僅被使用一次即丟棄。
- **衝突與崩潰**：
  1. **建表開銷無法攤提**：
     對於 Q6_K，為維持 generic 8-lane 契約，分組退化為 $g2$（每組僅 2 個 activation）。對 256 個輸入，需動態建立 128 組 $g2$ 小表。每個表包含 4 項，總共需進行 $128 \times 4 = 512$ 次算術預算。而每個 256-block 的權重僅被掃描一次，LUT 的複用次數（Reuse factor）極低。
  2. **面積與乘法器對照**：
     在標準 CMOS 製程下，一個 8-bit $\times$ 4-bit 符號乘法器僅需約 250～350 個等效邏輯閘（Gate Equivalents），其延遲在 200 MHz 下遠小於 2 ns。
     相反地，維護 128 組動態表的暫存器、多工定址解碼邏輯，其邏輯閘數與動態功耗遠超 16 條直接 MAC 通道。
  3. **結果**：
     在此情境下，**LUT Window 引擎的建表延遲與控制開銷直接暴露於關鍵路徑上，其總延遲與面積皆顯著劣於 Direct MAC 引擎**。此反例證偽了「LUT Window 在所有量化格式與所有配置下皆能勝過 Direct MAC」的通用假設。

---

### 3. [未驗證假設與架構缺口] 下一正式 Block 最該攻的單一關鍵問題

#### 單一優先問題：
> **「在 16-Chip 4-Plane 混合分佈式架構下，當 Row-batching $R \in [128, 256]$ 時，滿足 $M < M_{\text{crit}}$ 的實體 PE 面積與標準邏輯閘預算區間到底是否存在？即：在台積電或同等製程單元庫中，『2 KiB 分散式 MUX/RF LUT 引擎 + 8-Lane 累加器』與『16 通道 8-bit Direct MAC 陣列 + 展開器』的真實標準單元面積差額究竟是正還是負？」**

若後續無製程單元庫數據支撐，或實測顯示 16 條 8-bit MAC 的實體面積本就小於 2 KiB 暫存器加多工定址邏輯，則本候選架構在硬體層面的核心立足點將被徹底 Thought-veto。

---

## 總結：三組核心結論

### 1. 已確認結論 (Confirmed)
- **F32 順序敏感性**：GGML generic dot 的 8-lane float 累加契約具有數學剛性。整數預累加等價絕不等於 F32 等價；跨 lane 或跨 subgroup 的提早合併會產生高達數個 ULP 的位元分歧。
- **分組量子界限**：在 generic 8-lane 限制下，Q4_K 的最大合法查表分組為 stride-8 的 $g4$（剛好吻合 32-subgroup）；而 Q6_K 受限於 16-subgroup，合法分組被嚴格限制為 $g2$。
- **存活狀態下限**：$R=256$ 時，維持 8 條 F32 carry、min 補正與基本控制管線所需的本地高速存活狀態至少為 **15.6 KiB**，絕不可將 64 KiB 的 NAND 頁緩衝區視為隨機多埠記憶體。

### 2. 被否定/已淘汰主張 (Rejected / Refuted)
- **否定「16-lane 必須依賴 16-port 2KiB SRAM」**：已證明可透過每組 32B 的分散式 RF/MUX 與預取機制實作，不需昂貴的巨型多埠 SRAM。
- **否定「LUT Window 在所有混合精度格式下均具備優勢」**：在 Q6_K 下退化為 $g2$，建表開銷在低複用情境下直接導致吞吐量與面積劣於直接 MAC。
- **否定「單晶片 2 KiB 暫存即可閉環處理整專家」**：SwiGLU 的非線性啟動函數與 Down 投影所需的 Q8_K 全域 max-scale 依賴，證明跨晶片同步屏障與資料傳輸無可避免。

### 3. 未解關鍵問題 (Unresolved / Open)
- **實體邏輯閘預算未知**：在無具體 Technology Library 條件下，尚無法判定「16 條 Direct MAC + Nibble 展開器」與「2 KiB LUT 查表結構 + 多工選通網」何者在矽晶面積上真正佔優。
- **跨晶片 All-Reduce 成本量化**：Down 投影在選擇神經元本地聚合 vs. 部分和晶片間規約時，其傳輸開銷是否會直接抹平在儲存端運算所獲得的頻寬紅利。

