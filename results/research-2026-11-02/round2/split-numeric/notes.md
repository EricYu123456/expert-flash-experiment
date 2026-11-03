完成兩個既有原始 train prompts × 16 decode steps 的真實 GGUF 數值 smoke。G4 使用官方 Q4_K_M checkpoint，檢查前 3 層 gate/up；Qwen 使用既有 Q8_0 checkpoint，檢查前 2 層 down。新 runner 連結既有 pinned llama.cpp 靜態函式庫；未更動 llama source、原 runner、原 binary、模型或任何 frozen placement。沒有下載權重、讀取 test prompts 或 fit 方法。

Callback 在 decode 的 `GGML_OP_MUL_MAT_ID` 完成後讀取 F32 input、expert IDs 與原 CPU output。權重取自原 GGUF 的唯讀 byte views，保留原量化 blocks 與 scales。完整 input 先用原 `vec_dot_type.from_float` 量化一次，再切原量化 blocks，使用同一 `vec_dot` 核重算 full 與 partial dots。G4 比較 1024+512 及只讀的 512+512+512；Qwen 比較 512+256。Partial outputs 在 host 按 FP32 相加，沒有把 Q4/Q8 當 raw I8。

| 只讀 capture 比較 | Scalar dots | 最大絕對差 | 最大 phase relative L2 | 最大未裁切 relative error | 最大 ULP |
|---|---:|---:|---:|---:|---:|
| G4 full vs 原 CPU dst | 589,824 | 3.8147e-6 | 2.1640e-7 | 5.0000% | 524,288 |
| G4 split2 vs full | 589,824 | 3.8147e-6 | 1.9848e-7 | 5.4054% | 655,360 |
| G4 split3 vs full | 589,824 | 2.3842e-6 | 2.2000e-7 | 4.0541% | 393,216 |
| Qwen full vs 原 CPU dst | 1,048,576 | 0 | 0 | 0 | 0 |
| Qwen split2 vs full | 1,048,576 | 1.7881e-7 | 9.1339e-8 | 9.3750% | 1,572,864 |

G4 原 CPU 使用 runtime-repacked Q4 kernel，其 reduction 與原 row `vec_dot` 不逐 bit 相同。上表呈現這項差異；沒有將它當成 exact equality。Qwen full 重算則逐 bit 吻合所有原 CPU outputs。全部有效數值檢查沒有 NaN/Inf。Relative error 採 `abs(error)/max(abs(reference),1e-30)`，沒有用 magnitude threshold 過濾；近零輸出會放大 relative error 與 ULP，故仍完整保留在 summary，而非以小 relative L2 取代。

每個模型先獨立執行 native routing control，再做不寫入模型的 capture。四個 prompt/model cases 的全部 token IDs、ordered expert IDs 與 route records 均完全相同，確認旁路數值檢查本身未改變 inference。之後才進行真正的 output overwrite，讓替換值流入後續 layers。另做 full-dot overwrite，分離「原 row dot 取代 optimized CPU reduction」與「切 columns」的影響。

| Overwrite 對照 | Prompt | Token IDs | Ordered routes 改變 | Expert sets 改變 |
|---|---|---|---:|---:|
| G4 full vs native | summarization | 16/16 相同 | 194/640 | 65/640 |
| G4 full vs native | programming | 16/16 相同 | 148/640 | 52/640 |
| G4 split2 vs full overwrite | summarization | index 4 首次分歧 | 526/640 | 490/640 |
| G4 split2 vs full overwrite | programming | 16/16 相同 | 156/640 | 58/640 |
| Qwen full vs native | 兩題 | 全部相同 | 0/1,536 | 0/1,536 |
| Qwen split2 vs full overwrite | summarization | 16/16 相同 | 225/768 | 76/768 |
| Qwen split2 vs full overwrite | programming | 16/16 相同 | 96/768 | 22/768 |

G4 summarization native/full 的開頭是 “The project involves setting up a weather station”，split2 是 “The project involves a weather station”。這只是可檢查的輸出例子，未經品質評分，不能用語意看似相近宣稱 quality preserved。Token 分歧後的 routing 比較也混入了 token-stream 分歧；`summary.json` 額外分列共同 token prefix 內的 routing changes。Qwen token 相同時仍發生 expert-set 改變，說明 routing exact 並不能從 token 一致推出。

這個 smoke 支持原 GGUF scales 下的 CPU software composition，以及被量化的局部誤差；它不支持 bitwise、routing exact、普遍 token 不變或模型品質維持。使用者並未要求 bitwise identity，因此變化本身也不等於否決所有 column-split 方法，但若有足夠成本收益，後續必須做有固定 scoring protocol 的品質驗證。G4 3×512 僅做只讀 local comparison，沒有做其完整傳播的 overwrite run。

CPU callback 的 F32 output 不代表 AiF 原生 FIFO 是 F32，也不證明 native scales/accumulator/rounding 契約。這輪是同一 GGUF checkpoint 的軟體組合檢查，不是 native AiF format matching 或硬體 latency 實測。所有 runner wall times 都只供工程追蹤，不列為 placement latency。

最初工程 run 曾錯把 runtime-repacked Q4 buffer 當原 GGUF row bytes；full-vs-dst 控制立即失敗。其輸出保存在 `initial-repacked-buffer-invalid/`，明確標 invalid，沒有進入有效 summary。有效控制與 split overwrite 使用 `build/initial-valid-build.json` 指定的 source/binary snapshot；後續僅新增 full-overwrite 控制旗標，使用 final source/binary。這些版本與完整 artifact hashes 由 `completion.json` 記錄。
