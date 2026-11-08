# GPU 短訓與 readout spectrum

後續使用者授權 GPU 後，新增獨立程式，沒有改原 CPU learned-GU 程式或既有 frozen sources。
GPU0 為 NVIDIA L40S；Torch2.5.1+cu121，TF32／AMP／compile 關閉。
只有 TRAIN STE／Adam 用 GPU；VALID 固定 CPU64–65、2 threads 的精確 integer W8A8 與 pinned SwiGLU。

## 固定 C／affine 的 GU25 短訓

完整12 heads、四 checkpoints0／16／32／64、三個 learning rates，各64 steps／batch128／seed20261109。
step0 全12 heads 的 canonical SSE／energy 與現有模型一致，7,372個 unique VALID rows 的
10,379,776個 native phi floats 逐 bit 相符，原 W8 codes/scales 沒有 dequantize-requantize。
三組皆完成，peak reserved顯存966MiB；GPU0及CPU64–65已釋放。

| Learning rate | 最佳 step | Pooled VALID NMSE | 相對原 GU 的 SSE 改善 |
|---|---:|---:|---:|
| 0.0001 | 64 | 0.544903 | 0.526% |
| 0.0003 | 0 | 0.547786 | 0% |
| 0.001 | 0 | 0.547786 | 0% |

最佳 checkpoint 的 seed3 SSE 改善3.87%，非 BOS pooled改善0.528%。
這個固定 C／affine、64-step grid 的增益很小，因此沒有新增 runtime。
updated GU 是獨立 bank，不能與 native25 exact GU 去重；若兩份都保留，GU W8/scales共11,556,864 bytes。
每個 checkpoint 的 W8 GU、source／command／hardware與 hashes 都保存在 NFS `learned-gu-gpu`。

## 相同壓縮能力的雙方 SVD 診斷

對 prototype25 與同寬 rawsubset1408 的各12個實際 deployed W8 C，
先 F32 dequant，再 CUDA FP64 SVD；24矩陣的 singular energy 與輸入 Frobenius norm 相對差均小於1e-10。

種子 experts4／51／60 的 rank256 能量保留：

| Expert | Prototype25 | Rawsubset1408 |
|---|---:|---:|
| 4 | 96.95% | 98.92% |
| 51 | 96.26% | 98.46% |
| 60 | 95.53% | 98.33% |

rank128 時 prototype 為85.19–88.61%，rawsubset為91.47–94.05%。
同寬 raw 控制至少也有壓縮潛力，種子三 heads 的 weight energy 更集中。
rank256 的兩個 W8 因子加 row scales／bias 每 head為902,144 bytes，12 heads為10,825,728 bytes；
rank128 每 head為459,264 bytes，GU及共享 transform 另計。

這些是 weight-space 最佳 FP 近似的能量診斷，未測量新的 factor-W8／latent-A8 路徑。
weight Frobenius 能量不能直接保證 expert output NMSE、LLM品質或延遲；本階段沒有實作壓縮 runtime。
全程未開新 TEST；fresh article-held-out ≥1000 prompts 的最終方法驗證仍由 root 在凍結方法後執行。
