# 11/09 GPU shared-GU 獨立短訓協定

此 revision 記錄使用者後續授權 GPU 可用。它延伸既有 CPU-only 實驗，不修改原 CPU 程式或任何已 freeze source／selection。
GPU0 已確認為 NVIDIA L40S、driver535.288.01。相容環境為
`/home/ericyu/miniconda3/envs/petseg/bin/python`，Torch2.5.1+cu121。
CPU exact VALID 固定 affinity64–65、2 threads，避免干擾 root 的 quality workers。

## 固定資料與訓練範圍

沿用 prototype25-cohort12 的完整12 heads、TRAIN600 prompts／37,845 rows、VALID200／12,597 rows，context上限64 tokens。
selection adapter 保留各 head 已部署 C 的 VALID weighted SSE／target energy 與 TRAIN／VALID identities。
新舊 prototype25 transform 已逐 byte 核對一致，adapter 只將 prefix 統一；不改任何 C、affine 或原始 payload。
cohort 與原始 C 已使用 calibration VALID 選擇，這輪仍是 calibration 搜索，不是新 holdout。
完全不接受 TEST 參數，也不開 TEST inputs／targets。

只有 native GU25 初始化的 gate／up 矩陣可更新。GPU 執行 FP32 Torch STE surrogate 與 Adam；
C、bias、TRAIN center/inverse-scale 都固定。TF32、mixed precision、torch.compile 關閉。
TRAIN uniform sampled selected expert-row pairs，loss 按 native alpha² 加權，以固定 TRAIN target energy 正規化；
regularization=0.001 為相對初始化 displacement，gradient norm clip10。

先執行2-step／batch16 numerical smoke。通過後正式三個 learning rates為0.0001／0.0003／0.001，
每組64 steps、batch128、seed20261109，相同 minibatch sequence，checkpoints0／16／32／64。
GPU allocator上限8GiB，記錄型號、UUID、CUDA／Torch版本、driver與peak allocated／reserved memory。

## CPU canonical VALID 與 step0 正控制

每 checkpoint 將 GU F32 weights 送回 CPU，重新 row-W8；VALID 使用 frozen Numeric 的 exact integer dot、
F32 rescale、pinned native SwiGLU，再執行原 TRAIN affine、whole-vector A8、固定 row-W8 readout。
GPU surrogate loss 不作 canonical VALID 指標。
step0直接使用原 native GU 的 W8 codes/scales，禁止 dequantize-requantize。
對所有 VALID selected unique rows 的 raw prototype phi 與 frozen capture 逐 bit 核對，
並逐 head 核對已存在 baseline SSE／energy，pooled NMSE亦須一致。
另保留 BOS／非 BOS 與逐 head 結果，避免 pooled改善只來自單一位置或 expert。

每個 checkpoint 保存新的 W8 GU codes/scales與 hashes；三組結果完成後凍結。
updated GU 是新的 bank，不能當 native owner25 exact GU 使用；若同時保留 native hot-GU25 cache，
兩份 W8/scales共11,556,864 bytes，且各自 GU 計算不能去重。
每個 updated bank unique input row 額外 GU工作為5,767,168 MAC，加上 SwiGLU 與原 affine/A8。
VALID沒有實質改善時不新增 runtime；本階段只交付局部訓練／VALID診斷。

## Host readout 壓縮潛力的後續診斷

短訓grid結束後，對 prototype25 與 same-width rawsubset1408 的12個實際 dequantized-W8 C，
用同一 GPU 量 singular energy，固定報 rank128／256／512 的 cumulative Frobenius energy。
同時列兩個線性因子的係數／row-scale成本。這只是離線 spectrum 診斷，
不把 FP SVD 的能量保留當作 W8因子或新增 latent-A8 的品質證據，不直接寫 runtime。
