這輪建立的是功能性的 raw-W8A8 軟體參考與同數值 split 對照，不提出量化方法，也不宣稱原 GGUF 模型數值維持不變。仍只跑前輪相同的兩個原始 train prompts、每題 16 decode steps：G4 前三層 gate/up，以及 Qwen 前兩層 down。Prefill、其他 routed matrices、attention、router 等仍由原 GGUF llama.cpp 執行，因此目前是限定 decode operators 的混合模型原型，不是完整 W8A8 inference 或完整全 routed-FFN revalidation。

新獨立 runner 是 `scripts/research_20261102_raw_w8a8_numeric.cpp`。原 GGUF 唯讀映射、原 llama source/libraries、原 runner/binaries 與 frozen placement 均未修改，沒有下載新模型、讀取 test prompts 或搜尋 placement。

固定數值契約如下。每個權重 row 用原 GGUF `to_float` 解碼為 F32，取 `amax=max(abs(row))`，以 F32 計算 `rowScale=amax/127`；全零 row 使用 scale=1。每一個原 GEMV input vector 同樣在完整 C 上計算一次 F32 `inputScale`。兩者皆先以 F32 division 得到 normalized values，再採 nearest、halfway away from zero rounding，clip 到 `[-127,127]` 的 signed I8；零點為 0。非 finite input 或 underflow 到 0 的 scale 會拒絕，而非默默替換。權重於 expert 首次出現時量化，存入同一 RAM cache，full 與 split 共用；沒有依 placement 或 partial 重量化。

Full reference 對整條 I8 row/input 累加到 I64。Split 對相同 byte slices 各用 I32 累加，再在 host 以 I64 相加。兩者最後執行完全相同的運算：`float(integerSum) * float(rowScale * inputScale)`。這裡兩個乘法均明定為 F32，只有最終完整整數和轉換一次；編譯停用 fast-math 與 FP contraction。G4 的 functional split 是 512+512+512，另外同步檢查 1024+512；Qwen 是 512+256。

| 實際檢查 | G4 | Qwen |
|---|---:|---:|
| 每模式 scalar dots | 589,824 | 1,048,576 |
| Full/split 整數不一致 | 0 | 0 |
| 縮放後 F32 bit 不一致 | 0 | 0 |
| 最大完整整數絕對值 | 286,080 | 194,543 |
| 最大二段 partial 絕對值 | 209,763 | 144,101 |
| 最大三段 partial 絕對值 | 104,966 | 不適用 |
| Full/split token、輸出 bytes、ordered routing | 全部一致 | 全部一致 |
| 每模式 routing events | 1,280 | 1,536 |

每個 phase 的量化 input digest 與替換 output digest 也完全相同。Split 沒有新增任何 expert cache entry，全部 cache 都由 full reference 建立。兩模式真正覆寫原 GEMV output 並將值傳入後續 layers，不是只在旁路比較。

理論上單項乘積的絕對值不超過 `127²=16,129`。C=512 partial 不超過 8,258,048；C=1024 不超過 16,516,096，皆在 FP32 可精確表示所有整數的 `2^24=16,777,216` 範圍內。C=256 上界為 4,129,024。G4 完整 C=1536 上界 24,774,144，仍在 I32 範圍，但已不能保證每個可能整數都能由 F32 精確表示；reference 會先保留 I64 完整和，再執行共同的最終 F32 conversion。這只提供「若 native 回傳未縮放、未提前 rounding/requantization 的精確 I32 或符合條件的 FP32 partial，host 可恢復整數和」的條件證明。CPU 模型採 I32/I64 並不能證明 AiF PE/FIFO 就具有該契約，原生 output/accumulator/scales 仍待證。

權重 row scales 全部存於各模型 `scales/*.f32`，格式為 little-endian F32。G4 實際 cache 292 個 expert matrices、229,638,144 I8 bytes、598,016 scale bytes；Qwen 為 190 matrices、298,844,160 I8 bytes、1,556,480 scale bytes。I8 weights 僅放 RAM，numeric JSON 保存每個 expert 的原 GGUF offset、I8 digest 與 scale digest。全部 482 個 scale 檔均核對 digest、row 數與 finite positive values。另一個獨立 Python GGUF dequantizer＋NumPy 實作，針對每個目標 matrix 各一個真 expert，共 8 matrices／7,168 rows／7,864,320 weights，核對 row scale bits 與完整 I8 expert digest，全部通過。這項獨立檢查覆蓋八種實際 matrix 路徑，不宣稱重新驗算所有 cached weights。

量化影響與 placement 一致性必須分開解讀。同一個被攔截 GEMV 的當前 input 下，raw reference 與原 GGUF CPU output 的 phase relative L2：G4 為 1.8245–3.3901%，Qwen 為 3.1610–3.1846%。它不是前輪只有 floating reduction 的 1e-7 級誤差，不能當作零數值損失。這四個小 case 的 16 tokens 恰好均與既有原 GGUF native control 相同，但 expert sets 已改：G4 兩題分別 76/640 與 67/640，Qwen 分別 134/768 與 60/768。因此後續必須使用新 reference 自己產生的 routes；不能沿用原 Q4/Q8 routes 代表 raw-W8A8 模型，也不能由短 token prefix 相同宣稱品質維持。

全 routed-FFN reference 的資料量在本機可行，但本輪沒有擴跑：

| 全 routed-FFN inventory | G4 | Qwen |
|---|---:|---:|
| Raw I8 weights | 5.625 GiB | 27 GiB |
| Host-resident F32 row scales | 25 MiB | 84 MiB |
| 每 decode token 的 row outputs | 614,400 | 1,376,256 |
| 每 token 若各 scale 讀一次的 host bytes | 2,457,600 | 5,505,024 |
| 每 token integer MACs | 566,231,040 | 1,811,939,328 |

完整擴充需要支援所有 routed projection shapes，產生該 reference 的真 routes，並決定一致的 prefill/decode 計算契約。若要求同一量化設定涵蓋 prefill，現有原型尚須把 prefill routed FFN 也替換；本輪沒有付出大量 prefill 或取新資料。Raw quantization 是離線 baseline 建構，不是 placement 的收益。Host row-scale 容量、activation quantization、partials recovery、rescaling 的延遲與傳輸都要在成本模型中列入；不能因 scales 放 host 就把其代價當零。

目前 callback 仍先執行原 GGUF GEMV，並同時計算 full 與多種 split 控制，full 首輪還包含 lazy weight quantization，split 則使用 warm cache。其 wall times 只能作工程紀錄，不能拿兩模式時間比當 placement speedup。這輪證明限定軟體參考內的 full/split 數值一致，尚未完成完整模型品質、硬體格式或硬體效能驗證。
