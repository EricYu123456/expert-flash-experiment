# Round 3 探索 Block 02：錯誤更正、資源權衡與最窄前瞻命題

- **回合與身分**：`Round 3` 探索階段 Block 02（`round03-thinking`），Gemini 藍軍思考者（`gemini-3.8-flash-high`）。
- **計時與狀態邊界**：`newmethodclock = 0`。本探索**未採納**任何候選，**不計入**未來候選的 1800 秒實質思考時鐘，禁止宣稱已滿半小時。
- **資料與執行邊界**：嚴格遵守唯讀與禁制規則；**未讀取**任何模型權重、張量陣列、中間活化值、目標資料、FIT 暫存、驗證書籍或測試集；**未執行**任何候選程式碼、模擬器、RTL 或實證驗證。本探索失敗或否決**不消耗**剩餘 8 次實驗機會。
- **唯讀審核輸入**：
  1. [`round03-thinking/gemini_exploration_block01.md`](file:///home/ericyu/expert_flash_experiment/results/research-2026-11-03-no-cr/round03-thinking/gemini_exploration_block01.md)
  2. [`round03-thinking/round03_research_charter.json`](file:///home/ericyu/expert_flash_experiment/results/research-2026-11-03-no-cr/round03-thinking/round03_research_charter.json)
  3. [`round03-thinking/research_terra_landscape.md`](file:///home/ericyu/expert_flash_experiment/results/research-2026-11-03-no-cr/round03-thinking/research_terra_landscape.md)

---

## 1. 正面逐項更正 Block 01 初稿的五大重大錯誤

初稿中提出的方向 A（LoSparse SVD+R）與方向 B（Base 錨定 + Delta-MoE）已被 Root 與 Critic 明確執行 **thought-veto**，不進入候選時鐘，亦不消耗實驗機會。現保留原稿供歷史審計，並在此對初稿的五大核心錯誤進行嚴肅更正：

### 更正 1：任意矩陣的 $W = UV + R$ 只有保留完整稠密 $R$ 時才代數恆等；截斷稀疏 $R$ 屬全新有損近似
- **初稿錯誤**：聲稱「任意量化矩陣 $W$ 皆可做 $UV + R$ 分解且將 $R$ 截斷保留 $5\%$ 離群值，此為 100% 數值精確恆等式，保證 $\Delta\text{PPL} = 0.00\%$ 與 bitwise 等價」。
- **實質糾正**：
  1. 奇異值分解（SVD）的尾項殘差 $R = W - UV$ 在數學上**通常是稠密的（Dense）**，其能量分布於所有矩陣元素中。若強行將其截斷為僅保留 $5\%$ 的最大係數，本質上是引入了一個未經誤差界定、不可預測的全新**有損啟發式近似**，絕非代數恆等。
  2. 在推論時計算 $(UV)x + R_{\text{sparse}}x$，改變了浮點運算的結合順序（Floating-point reassociation），在 IEEE 754 下根本無法保證與原始 GEMM 逐位元一致（Bitwise exact），更不可能無條件保證 $\text{PPL}$ 劣化為 0。
  3. 真正的 LoSparse（NeurIPS 2023）是端到端**聯合訓練（Joint Training）**，透過反向傳播讓低秩與稀疏結構共同補償損失；初稿將其偷換為 post-hoc 無訓練硬分解，在數學與機器學習理論上完全站不住腳。

### 更正 2：非線性函數差 $F_e - F_{\text{base}}$ 絕不等於權重線性相減 $\Delta W x$；Base 無全模型品質下界保證
- **初稿錯誤**：聲稱「$W_e = W_{\text{base}} + \Delta W_e$，主機算 Base 即可提供 $90\%$ 品質下界保護，Flash 僅需 1 頁計算線性增量 $\Delta W_e x$」。
- **實質糾正**：
  1. SwiGLU 是高度非線性算子：$\text{SwiGLU}(W_e x) - \text{SwiGLU}(W_{\text{base}} x) \neq \text{SwiGLU}(\Delta W_e x)$，更不可能簡化為線性矩陣相乘 $\Delta W_e x$。在包含門控乘法的情況下，權重之差與函數之差不可交換。
  2. 即使存在某種形式的 Base 輸出，其前向係數應為實際選中專家的路由權重和 $\sum_{e \in \text{live}} \alpha_e$。在 DeepSeek-V2/V3 與 OLMoE 中，被選中的 $\alpha_e$ 總和並不恆等於 1（如 DeepSeek-V3 約為 2.5，V2/OLMoE 亦不強制歸一化）。
  3. 聲稱「Base 提供了全模型 $90\%$ 的品質下界保護」與「增量可壓縮進 1 頁（$16\text{ KiB}$）」是**完全未經證實的凭空臆測**。若殘差非線性極強，Base 輸出的偏移可能直接破壞下游層的輸入分佈，導致嚴重困惑度爆炸。

### 更正 3：OLMoE 維度計算嚴重失實；單一葉片不可冒充全專家儲存；位元平面未降低資訊工作量
- **初稿錯誤**：聲稱「$W_{\text{down}}$ 佔 FFN 一半以上參數」、「單專家可壓縮至 $16\text{ KiB}$」、「位元平面映射至 4 個平面降低了計算延遲與工作量」。
- **實質糾正**：
  1. 在 OLMoE（$H=2048, I=1024$）中，$W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$ 三個矩陣各為 $2048 \times 1024$，W8 量化下各為 $2\text{ MiB}$，合計約 $6\text{ MiB}$（未計 scale/bias 等 metadata）。**$W_{\text{down}}$ 僅佔總參數的 $1/3$，絕非「超過一半」**。
  2. R2 實驗中，單個 leaf 是 $256\text{ KiB}$，但 C32 每個專家是 $8\text{ MiB}$，整層 64 專家高達 $512\text{ MiB}$。**絕不能用單一 leaf 的大小偷換概念冒充全專家或全模型儲存銀行**。
  3. 將權重的 4 個 bit-plane 分布於 4 個實體平面，本質上只是改變了資料的空間布局。每個平面感測 $16\text{ KiB}$，總共感測的位元數依然是 $N \times b$；整數乘加的資訊熵與乘積工作量並未減少，不能將平面並行感測等同於「資訊工作量降低」。

### 更正 4：SLIM 已在 Die/Channel PE 實作 FFN 融合；R2 的失敗不能泛化為全域不可能定理
- **初稿錯誤**：聲稱「SLIM 必須依賴 DRAM-PIM，我們將 SwiGLU 移至 NAND die 屬於首創」、「R2 證明了非線性絕對不可丟、所有低秩與線性近似均不可能」。
- **實質糾正**：
  1. [SLIM (arXiv:2507.09201v1 §5.4)](https://arxiv.org/html/2507.09201v1) 早已在 die-level / channel PE 上實作了神經元粒度的 fused gate/up/down 部分和累積（partial sums）；純粹將計算移入 NAND die 既非首創，亦非實質新穎性。
  2. R2 的實證失敗（OLMoE L0 64 專家 $\text{PPL} +15.8\%$）**僅嚴格否定了那一個特定的凍結配置**（$C32/r128$ 無條件局部仿射在固定超參下的擬合），並未在因果機制上嚴格隔離是「無條件分區崩潰」、「秩 128 過小」還是「非線性門控缺失」。科學上不能據此斷言「所有低秩均不可能」或「任何近似均無法保持品質」。

### 更正 5：收回所有未經實證的斷言與虛假宣稱
- 正面收回初稿中所有包含「從來未做」、「首創」、「PPL 增加 $<0.05\%$」、「延遲小於 $200\,\mu\text{s}$」等無任何實驗或嚴密文獻依據的浮誇表述。在後續探索中，所有推論嚴格標註為「假設」或「待證條件」。

---

## 2. 審視原生量化區塊運算（`Q4_K / Q6_K`）在普通 NAND 上的物理資源權衡

Sol 提出應在紙上推導：**直接沿 GGUF 原生量化區塊邊界（如 Q4_K 的 32 權重子塊），在普通 NAND die 上直接執行 $GU \to \text{SiLU} \to \text{Down}$ 的原生區塊運算，而非搞替代函數**。

然而，單純將量化 GEMM 搬入 Flash 絕非論文級貢獻：
- **T-MAC (arXiv:2407.00088)** 已經在 CPU 上使用查表法（LUT）消除了低位元整數乘法；
- **CodeGEMM (arXiv:2512.17970)** 已經實作了啟動值聚類與 partial-sum book 的 index gather；
- **SLIM (arXiv:2507.09201v1)** 已經實作了近儲存 FFN 融合。

若要在普通 NAND 上成立，必須考察是否存在**NAND 特有的、不可迴避的底層物理資源 Tradeoff**。

### 2.1 紙上檢驗結構 (a)：小 LUT、多輸入 Epoch、細粒度 Page Packing
- **構造設定**：為了壓低 NAND die 上極度昂貴的 SRAM 面積，僅構建 2-bit（4 項）或 3-bit（8 項）微型活化值查表（Activation LUT），每個 LUT 僅佔 $16\sim 32\text{ B}$。
- **物理反噬（The Packaging & Input Tax）**：
  1. 由於每個 LUT 僅能處理 2~3 個位元，計算一個 4-bit 權重的點積必須拆解為多個次字元（Sub-nibble）輪次（Input epochs）。
  2. 為了配合微型查表，NAND 實體頁面（$16\text{ KiB}$）內的權重必須採用高度交錯的細粒度打包。然而，NAND 的實體讀取與 ECC 校驗是以 Codeword（通常為 $1\text{ KiB}$ 或 $4\text{ KiB}$）為最小邊界的；細粒度交錯將導致 Codeword 邊界對齊失效，產生大量的**填充 Padding 浪費**。
  3. 每次更換輸入向量的分塊，都需要向晶片內部暫存器重新廣播（Broadcast）輸入查表項；多輪次的輸入廣播延遲與指令開銷（Input tax），將徹底抵銷低位元儲存所節省的位元組收益。

### 2.2 紙上檢驗結構 (b)：大 LUT 常駐、大 SRAM 與多 Port 需求
- **構造設定**：為了避免多輪次廣播，在每個 Plane 附近部署完整的 4-bit / 6-bit 常駐查表（例如 16~64 項，加上多通道並行，需 $2\text{ KiB}\sim 8\text{ KiB}$ SRAM）。
- **物理反噬（SRAM / Port Area Blowup vs. 乘法器陣列）**：
  1. 一個 16-lane 的 PE 要在 1 個週期內完成 16 個權重的並行查表，需要該 SRAM 具備 **16 個獨立讀取端口（16 Read Ports）**，或者將該 SRAM 複製 16 份獨立 Bank。
  2. 在 NAND 週邊 CMOS 製程（通常為 45nm 或 28nm）下，一個 16 端口的 $4\text{ KiB}$ SRAM（或 16 副本 SRAM）的矽晶面積與布線開銷極為龐大，其面積與功耗將遠超 $0.07\text{ mm}^2$ 的限制。
  3. **殘酷的矽智財事實**：在專用 ASIC / NAND PE 上，一個 4-bit 整數乘法器本質上只是由幾十個 AND 門和半加器組成的極簡組合邏輯；**在晶片內部做多端口 SRAM 查表，其面積與靜態功耗遠比直接放 16 個實體硬體乘法器昂貴數倍！** T-MAC 在 CPU 上用 LUT 是因為 CPU 缺乏專用的 INT4 張量單元，依賴向量暫存器 shuffle 廉價；將此邏輯生搬硬套至 NAND 專用硬體上是典型的架構錯配。

### 2.3 跨專家攤還的根本不對稱：Gate/Up 可以，Down 絕不可以
- 在同層被選中的 $K$ 個專家（$K=6$ 或 $K=8$）中，$W_{\text{gate}}$ 與 $W_{\text{up}}$ 接收的是同一個 token 輸入向量 $x$；因此輸入查表或輸入廣播確實在 $K$ 個專家間存在**一次廣播、多專家共用**的攤還機會。
- 但在 SwiGLU 之後，$W_{\text{down}}$ 接收的輸入是各專家私有的 $h_e = \text{SiLU}(W_{g,e} x) \odot (W_{u,e} x)$。**$K$ 個專家的 $h_e$ 彼此完全不同**！
- 這意味著在 Down 階段，若採用查表或專屬預處理，必須為 $K$ 個專家分別構建 $K$ 套完全獨立的輸入狀態；任何試圖在 Down 階段假設「共享輸入查表或共享緩衝」的設計，在資料流依賴上均屬偽命題。

### 2.4 GGUF 數值契約與強基準的真實邊界
1. **數值精確度**：GGUF 的 Q4_K / Q6_K 包含超區塊（Super-block）浮點縮放、子區塊（Sub-block）6-bit 縮放與偏置，並非純粹的整數代數。晶片內 PE 在做整數累加後，必須將定點數轉換為浮點數並乘以 scales 進行減偏置處理。晶片內定點累加順序與主機 AVX-512 FMA 浮點累加順序的不同，必然產生 ULP 級別的捨入漂移，**不能隨口宣稱「Bitwise Exact」**。
2. **強基準的現實**：對於 OLMoE（全模型 Q4_K 權重僅 $\sim 4.2\text{ GB}$），在標稱 $8\text{ GB}$ 主機記憶體下，**完全可以直接將全模型 Q4_K 常駐於主機 DRAM 中（All-Host Resident）**！主機 CPU 依託 DRAM 高頻寬與向量指令集，其延遲在數百微秒量級。In-Flash 就算感知 0 等待，PE 的微弱算力也無法超越 DRAM 常駐 CPU。In-Flash 的真正必要性**僅存在於權重總量顯著超過主機可用 DRAM（如 DeepSeek 14GB+）導致強行觸發 Flash 頁面換入換出的場景**。

---

## 3. 最窄可正式思考命題與其直接否決反例（Veto Counter-Example）

為避免漫無邊際的發散，以下提出一個邊界最窄、最具體、值得雙 Thinker 進行 $\ge 1800\text{s}$ 嚴格審核的系統命題，並同時給出一個能直接將其否決的簡潔反例。

### 最窄候選命題：
**「基於實體頁對齊的原生 Q4_K Fused SwiGLU 串流引擎（NAND-Aligned Native Q4_K Fused SwiGLU Engine）」**
- **明確主張**：
  1. 拒絕任何基底近似、低秩分解或線性化替代；完全保留 GGUF 原始 Q4_K 權重區塊與標準 SwiGLU 非線性拓撲。
  2. 拒絕多端口 SRAM 查表架構；PE 端採用面積最小的 16-lane INT4 原生乘法邏輯樹。
  3. 將專家的 $W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$ 權重按照 16-chip × 4-plane 的物理 $16\text{ KiB}$ 頁面嚴格對齊交錯條帶化存儲。
  4. 利用單晶片內 $2\text{ KiB}$ 內部暫存器，在 2 個連續感測波次（$56\,\mu\text{s}$）內串流完成 $W_g, W_u$ 矩陣乘，經數位 SiLU 激活後，中間向量 $h$ 留存於晶片內，緊接著在第 3 個波次（$28\,\mu\text{s}$）完成 $W_d$ 運算，直接向主機輸出 $H$ 維結果向量。
  5. 核心價值在於：單個專家的權重永遠不經過 ONFI 或 PCIe 傳輸，消除了數 MB 權重的傳輸瓶頸；且模型品質完全保證與原生 GGUF 浮點結果保持在同等數值穩定度內。

---

### 能直接否決該命題的簡潔反例（Direct Veto Counter-Example）：
**「多專家實體晶片碰撞引發的波次序列化下界（The Multi-Expert Chip Conflict Serialization Lower Bound）」**

- **反例構造與嚴格算術**：
  1. 系統硬體為固定的 **16 個實體晶片（16 Chips）**。
  2. 當一個 token 經過 Router 時，需要同時選中 $K=8$ 個不同的路由專家（以 OLMoE 或 DeepSeek 為例）。
  3. 每個專家的 $W_g, W_u, W_d$ 即使最優化條帶化，也必須分散佔用這 16 個晶片。當 $K=8$ 個專家在同一 token 被同時激活時，根據抽屜原理與隨機路由分布，**多個選中的專家不可避免地映射到同一批實體晶片上**（晶片衝突概率 $> 99.9\%$）。
  4. 由於每個實體晶片在同一時間只能執行一個專家的感測與 PE 運算（單 Context、單命令隊列，無 CR 雙緩衝）：
     - 晶片**無法同時感測專家 A 的 $W_g$ 與專家 B 的 $W_g$**。
     - 這意味著原本單個專家所需的 3 個感知波次（$84\,\mu\text{s}$），在 $K=8$ 個專家碰撞時，必須在衝突晶片上**嚴格序列化（Serialized）**！
  5. 最保守的序列化波次需求為 $K \times 3 = 24$ 個波次（若 16 晶片滿載度為 2，亦至少需要 $2 \times 3 \times 2 = 12\sim 16$ 個全域同步波次）。
  6. 單純感知時間下界即為：
     $$16 \times 28\,\mu\text{s} = \mathbf{448\,\mu\text{s}} \quad (\text{若完全序列化則高達 } 24 \times 28\,\mu\text{s} = \mathbf{672\,\mu\text{s}})$$
  7. **基準擊穿**：
     反觀同資源下的強主機基準——一個帶有智慧型非同步預取的普通快取 Loader，只需利用 PCIe 傳輸真正未命中的專家；對於命中快取的專家，CPU 依靠多核心並行僅需幾十微秒。
     在單 token 延遲上，**In-Flash PE 因晶片衝突而導致的數百微秒序列化感知時間，直接被主機普通讀取 + CPU 運算徹底支配！**

---

## 4. 若該命題被否決：兩個全新的宏觀系統問題（Macro Questions）

如果上述基於原生 Q4_K 的管線依然被「晶片碰撞序列化」擊穿，則說明任何試圖在單一 Token（$B=1$）下靠 Flash 晶片內計算解決全量專家的思路，在物理排程上均存在天花板。此時必須退回宏觀架構層面，審視以下兩個全新的系統問題：

1. **宏觀問題一（批次吞吐 vs. 單 Token 延遲的定位重構）**：
   In-Flash MoE 是否在物理上根本**不適合**作為 $B=1$ 的單 token 延遲加速器，而本質上是一個**高併發批次吞吐（Batched Throughput, $B \ge 16$）加速器**？
   - *邏輯*：在單 token 下，NAND 的 $28\,\mu\text{s}$ 是無法隱藏的硬延遲；但當 Batch 增大時，多個 token 命中的專家大幅重疊，此時 1 次 $28\,\mu\text{s}$ 感知讀取的權重，可在晶片內部連續對 16 個 token 的輸入進行點積，將感測延遲分攤至 $< 2\,\mu\text{s}/\text{token}$，同時徹底消除 PCIe 被權重複製塞爆的問題。R1 與 R2 的困境，是否源於我們用單 token 延遲指標去硬套一個本質上屬於批次頻寬型的儲存媒介？

2. **宏觀問題二（極度非對稱的「主機稠密門控＋Flash 稀疏查表」解耦）**：
   是否存在一種非對稱拓撲：**將門控決策、前向特徵提升（Projection Lift）與全域殘差 100% 留在主機，而 Flash 僅被視為一個純粹由硬體觸發的「鍵值記憶庫（Key-Value Memory Bank）」**？
   - *邏輯*：如同 Geva 等人的研究（Transformer FFN 是 Key-Value Memories），將專家計算完全解構為向量檢索。Flash PE 不做任何通用浮點矩陣乘，僅執行單波次的局部漢明距離（Hamming distance）或粗粒度點積匹配，直接輸出檢索向量給主機。這是否能徹底打破 GEMM 對感測波次與計算單元的沉重依賴？

---

## 5. 總結與交接

本 Block 02 完成了對 Block 01 嚴重概念與數值錯誤的徹底更正，紙上剖析了 T-MAC / CodeGEMM 式的 LUT 在 NAND PE 上的面積反噬，提出了一個邊界極窄的原生 Q4_K 命題，並揭示了「多專家晶片碰撞序列化」這一潛在的致命物理反例。本記錄不含任何已批准的候選代碼，不消耗 30 分鐘時計與 8 次失敗額度，留待 Critic 與團隊進行下一步攻防。
