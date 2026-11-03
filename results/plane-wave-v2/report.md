# Group＋plane 聯合搜尋與無 cr-read 實驗

日期：2026-11-01。**這次已真正聯合移動 expert 的 group 與 plane 位置；最佳結果仍全部選擇 `w=4`，與只學 group、相同預算的兩階段搜尋相同。關掉 cr-read 後，Granite 仍降低 21.05% 延遲，其餘四個模型選回 baseline。**

## 比較方式

- **Baseline**：16 chips，`g=16`（不分 group）、gate/up 串接、expert 原始排列、`w=4`（每個 stream 使用全部 planes）。兩組實驗分別使用自己的 CR 開／關 baseline。
- `g` 是每個 expert 使用的 chip 數，chip group 數為 `16/g`；`w` 是每個 expert stream 使用的 plane 數。`w=2` 分成兩組 plane pairs，`w=1` 分成四個單 plane bins，可讓不同 experts 同時讀取。
- **聯合搜尋**：每一 layer 的 expert 同時具有 `(chip group, plane bin)` 位置，交換可跨 group、跨 bin，或同時跨兩者。位置離線學好後固定，不隨 token 搬動，也不複製權重。
- **強制 plane 分工**：只從 `w=1/2` 選擇，避免最佳結果退回 `w=4` 後看不到 plane 方案的表現。

以下數值為 **ms / decode token，越低越好**，加總所有 routed FFN layers 的 flash、命令、host、輸入與輸出服務時間；不是完整模型的端到端 token latency。改善百分比是 `1 − 候選延遲 / 同讀取模式 baseline 延遲`。

![兩種讀取模式的延遲比較](/home/ericyu/expert_flash_experiment/results/plane-wave-v2/figures/joint_no_cr_overview.png)

## 1. 保留 cr-read：聯合搜尋 group＋plane

| 模型 | Baseline | 最佳聯合搜尋（配置） | 延遲降低，95% CI | 強制 plane 分工（配置） |
|---|---:|---:|---:|---:|
| Granite | 12.703 | 6.778 (`g4/w4`) | 46.65% [46.41, 46.90] | 7.204 (`g8/w2`) |
| OLMoE | 13.648 | 12.161 (`g8/w4`) | 10.90% [10.78, 11.03] | 12.748 (`g16/w2`) |
| DeepSeek | 21.333 | 20.188 (`g8/w4`) | 5.37% [5.11, 5.63] | 20.962 (`g16/w2`) |
| Qwen3 | 37.024 | 30.361 (`g8/w4`) | 18.00% [17.86, 18.13] | 32.047 (`g16/w2`) |
| Mixtral | 112.838 | 112.838 (`g16/w4`) | 0.00% [0.00, 0.00] | 143.999 (`g16/w2`) |

**五模型的最佳聯合搜尋，都與新版「只學 group」及「先 group、再 plane」逐 token 相同；最佳配置沒有額外 plane 分工收益。** 與舊實驗相比的小幅改善，來自這次更充分、以完整成本為目標的 group 搜尋，不能歸因於 plane。

聯合搜尋確實有執行：Granite 的強制 plane 最佳配置 `g8/w2`，被選起點的搜尋紀錄累計接受 262 次跨 group、263 次跨 plane，其中 134 次同時跨兩者。它比相同預算兩階段搜尋的 **7.231 ms** 降至 **7.204 ms**，改善 **0.374%**，成對 95% CI **[0.242%, 0.527%]**；仍慢於只學 group 的 **6.778 ms**。其餘四模型的強制 plane 最佳配置皆為 `g16`，只有一個 group，兩方法退化成相同的 plane-only 搜尋，所以結果相同。

只做 plane、固定 `g=16` 也沒有超過最佳 group 方法：Granite 為 **8.010 ms（w1）**；其餘四模型等於表中的強制 plane 結果。

## 2. 關掉 cr-read：候選與 baseline 都使用 ordinary read

只把 NAND 讀取時間由 `9.7 μs × waves + 18.3 μs × block runs` 改為 **`28 μs × waves`**。保留 `max(NAND, shared ECC)` overlap、6.4 GB/s ECC/PE、原有命令與向量傳輸成本，沒有混入先前的串行 drain 假設。

| 模型 | Baseline | 最佳聯合搜尋（配置） | 延遲降低，95% CI | 強制 plane 分工（配置） |
|---|---:|---:|---:|---:|
| Granite | 12.703 | 10.030 (`g8/w4`) | 21.05% [20.93, 21.16] | 10.843 (`g16/w2`) |
| OLMoE | 23.018 | 23.018 (`g16/w4`) | 0.00% [0.00, 0.00] | 26.186 (`g16/w2`) |
| DeepSeek | 41.316 | 41.316 (`g16/w4`) | 0.00% [0.00, 0.00] | 46.788 (`g16/w2`) |
| Qwen3 | 58.105 | 58.105 (`g16/w4`) | 0.00% [0.00, 0.00] | 60.687 (`g16/w2`) |
| Mixtral | 303.793 | 303.793 (`g16/w4`) | 0.00% [0.00, 0.00] | 395.266 (`g16/w2`) |

這組最佳聯合搜尋，也與只學 group、相同預算兩階段搜尋逐 token 相同。具體結果：

- **Granite 仍能提升**：只分 group、不學排列為 `g8/w4`、**11.251 ms（降低 11.43%）**；學 expert 排列後為 **10.030 ms（降低 21.05%）**。只做 plane 為 `g16/w2`、**10.843 ms（降低 14.65%，95% CI [14.51%, 14.78%]）**，仍慢於學 group。
- **其餘四模型沒有找到優於 baseline 的配置**。強制 plane 後，OLMoE／DeepSeek／Qwen3／Mixtral 分別增加 **13.76%／13.24%／4.44%／30.11%** 延遲。

Granite 能保留收益，是因為 baseline 的 gate/up 每 chip 為 64 KiB、down 為 32 KiB，down 的每波只用到 2/4 planes。`g8` 讓 down shard 變成 64 KiB，配合學習後的共選負載分散，能抵銷較大 shard 的成本。Baseline 中每個 expert 的 gate/up stream 與 down stream 各只有一波，所以關閉 cr-read 後 Granite 的 baseline 數字不變。

無 CR 的 Granite，group 與 plane 方案的 NAND 時間分別為 8.896／8.891 ms，差異極小；group 方案把 block 命令成本由 1.536 降至 0.768 ms，因此總延遲更低。

其他模型的多波 shard 失去 CR 後，每一波都付 28 μs。縮小 group 或 plane width 所增加的波數與負載不均，會吃掉並行收益。本次無 CR 的全部候選都由 NAND 時間限制；四 planes 的理想 NAND 吞吐上限約為 `64 KiB / 28 μs = 2.34 GB/s`，低於共享 ECC/PE 的 6.4 GB/s。

另以舊訓練資料學好的 gate/up bundles（B=1/2/4）重算無 CR 成本，重新用 training 選 `g/B`：五模型都選回 **B=1**；強制 B>1 的最佳方案均退化。此補充保留舊 group 與 bundle 成員排列，沒有新增 bundle 聯合搜尋，因此 B=1 與新版 group 搜尋可能略有差異，詳見 [補充結果](/home/ericyu/expert_flash_experiment/results/plane-wave-v2/no-cr-bundles/summary.json)。

## 實驗與驗證範圍

- 每個模型沿用 **200 training prompts＋100 held-out prompts，各 128 decode steps**；另檢查前 32 steps，結論一致。配置只由 training 選擇；兩個主要實驗皆先凍結全部五模型，再開啟該組 test，bundle 補充則逐模型凍結後測試。95% CI 以整個 prompt 成對 bootstrap 2,000 次。
- 搜尋 `g∈{1,2,4,8,16}`、`w∈{4,2,1}`；每個 configuration/layer 使用原始排列與舊 training-only 排列兩個起點，每起點 1,024 個交換提案。兩階段對照在兩個維度各用 512；只有一個可用維度時，全部預算用於該維度。目標包含完整 FFN 服務成本，保留原始起點避免退步。
- 使用 INT8、完整 output-row shards。`g/w` 在每個模型共用，expert 位置按 layer 分別學習。每個 expert 在該 layer 的所有 chips／兩個 projections 共用同一個 plane bin；只搜尋互不重疊的固定 plane bins，維持每個 group/bin 的 expert 容量。這是有限預算的局部搜尋，不能證明所有可能布局都無收益。
- **60 個 unit tests 通過；1,170 組獨立 scalar 事件核對；另外 330 組儲存 token 成本重建通過。** 凍結與來源 hashes、train/test 分離、training 最小值、所有儲存成本加總，以及 30 個全模型布局的 `(chip, plane, block, WL)` 不重疊與權重不重複均通過。[獨立算術驗證](/home/ericyu/expert_flash_experiment/results/plane-wave-v2/arithmetic_audit.json)、[布局與結果稽核](/home/ericyu/expert_flash_experiment/results/plane-wave-v2/artifact_audit.json)、[兩組結果交叉核對](/home/ericyu/expert_flash_experiment/results/plane-wave-v2/cross_study_checks.json)。

這些是架構成本模型結果，沒有實際 NAND／PE 或新的 NVMe kernel 測量。`w<4` 仍需要獨立 plane 定址、部分 plane 命令與多 expert input/accumulator contexts。主表未加未知的額外 context setup／每波命令成本；無 CR 的 [敏感度資料](/home/ericyu/expert_flash_experiment/results/plane-wave-v2/no-cr/sensitivity.csv) 已對 baseline 與候選一致加價，Granite 的 plane 方案比 group 方案更敏感。

## 資料與重跑

[精確比較 CSV](/home/ericyu/expert_flash_experiment/results/plane-wave-v2/comparison.csv)；[CR 詳細結果與 CI](/home/ericyu/expert_flash_experiment/results/plane-wave-v2/joint-cr/summary.json)；[無 CR 詳細結果與 CI](/home/ericyu/expert_flash_experiment/results/plane-wave-v2/no-cr/summary.json)；[預先固定的比較規則](/home/ericyu/expert_flash_experiment/results/plane-wave-v2/protocol.json)。各目錄保留 frozen plans、逐 token NPZ、逐 prompt 結果與全部 training／held-out candidates；held-out grid 僅供描述，未用於重新選配置。

在 `/home/ericyu/expert_flash_experiment` 執行，需保留原 corpus route caches：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 scripts/run_joint_plane_study.py --workers 8 --swap-trials 1024 --stage fit
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 scripts/run_joint_plane_study.py --workers 8 --swap-trials 1024 --stage test
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 scripts/run_no_cr_study.py --workers 8 --swap-trials 1024
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 scripts/evaluate_no_cr_bundles.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 scripts/audit_joint_no_cr.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 scripts/audit_joint_study_artifacts.py
python3 scripts/summarize_joint_no_cr.py
```
