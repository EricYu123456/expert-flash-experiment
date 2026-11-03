# 無 cr-read：固定配置實驗

兩側全部使用普通讀取：每波 28 μs。baseline 為 g=16、四 planes、原始 expert ID、無學習，gate/up 已串接。其餘 NAND/ECC 重疊、向量與命令成本相同。

**在本次 group/plane 搜尋中，只有 Granite 找到提升：分 group＋學排列降低 21.05%；只做兩-plane 分工降低 14.65%。聯合搜尋沒有再超過分 group＋學排列。其他四個模型均選回 baseline。**

200 個 training prompts（各 128 tokens）學配置與選 g/plane 寬度；全部模型凍結後，評估獨立 100 個 test prompts。下表為 test 的 ms/input；括號為相對各模型 baseline 的降低百分比。

| 模型 | baseline | 只分組 modulo | 分組＋學排列 | 只 plane g16 | 先 group 再 plane | 聯合 group＋plane |
|---|---:|---:|---:|---:|---:|---:|
| granite | 12.7035 | 11.2512 (+11.43%, g=8,w=4) | 10.0300 (+21.05%, g=8,w=4) | 10.8429 (+14.65%, g=16,w=2) | 10.0300 (+21.05%, g=8,w=4) | 10.0300 (+21.05%, g=8,w=4) |
| olmoe | 23.0180 | 23.0180 (+0.00%, g=16,w=4) | 23.0180 (+0.00%, g=16,w=4) | 23.0180 (+0.00%, g=16,w=4) | 23.0180 (+0.00%, g=16,w=4) | 23.0180 (+0.00%, g=16,w=4) |
| deepseek | 41.3163 | 41.3163 (+0.00%, g=16,w=4) | 41.3163 (+0.00%, g=16,w=4) | 41.3163 (+0.00%, g=16,w=4) | 41.3163 (+0.00%, g=16,w=4) | 41.3163 (+0.00%, g=16,w=4) |
| qwen3 | 58.1053 | 58.1053 (+0.00%, g=16,w=4) | 58.1053 (+0.00%, g=16,w=4) | 58.1053 (+0.00%, g=16,w=4) | 58.1053 (+0.00%, g=16,w=4) | 58.1053 (+0.00%, g=16,w=4) |
| mixtral | 303.7932 | 303.7932 (+0.00%, g=16,w=4) | 303.7932 (+0.00%, g=16,w=4) | 303.7932 (+0.00%, g=16,w=4) | 303.7932 (+0.00%, g=16,w=4) | 303.7932 (+0.00%, g=16,w=4) |

上述策略均允許退回 w=4（無 plane 分工）。以下強制使用 w<4，避免把 fallback 當成 plane 的收益。

| 模型 | 聯合搜尋強制 plane | 相對 baseline 降低 | paired 95% CI |
|---|---:|---:|---:|
| granite | 10.8429 (g=16,w=2) | +14.65% | [+14.51%, +14.78%] |
| olmoe | 26.1858 (g=16,w=2) | -13.76% | [-13.93%, -13.59%] |
| deepseek | 46.7881 (g=16,w=2) | -13.24% | [-13.56%, -12.93%] |
| qwen3 | 60.6875 (g=16,w=2) | -4.44% | [-4.63%, -4.27%] |
| mixtral | 395.2663 (g=16,w=2) | -30.11% | [-30.64%, -29.60%] |

Granite 的 g=8 配置把原本半滿的 down waves 填滿，已發出 waves 的 plane-slot 填滿率由 75% 提至 100%。學習排列減少共選 experts 擠在同一 group；NAND 臨界路徑由 10.7520 降至 8.8957 ms，既有 block 命令成本由 1.5360 降至 0.7680 ms。收益來自更少 waves 和較低命令／向量成本，沒有使用 cr-read。

每層、每個 g/w 採相同兩個起點（原始 ID、舊 training-only group/plane 配置），每起點 1,024 次 swap proposals。聯合搜尋可跨 group/bin 交換；staged 在 g<16,w<4 時分 512 次 group、512 次固定 group 的 plane 搜尋；g=16 或 w=4 則把全部預算用於唯一可學的維度。完整延遲作為 training objective。這是有限預算搜尋，無法證明全域最佳。

模型仍假設獨立 plane 定址、多 contexts 及 max(NAND,ECC) 管線。額外 context/每 lane read 命令成本的敏感度保存在 `sensitivity.csv`；prefix32、完整 128、paired prompt bootstrap 2,000 次 CI、每 prompt 和 per-token 成本分別見 `summary.json`、`per_prompt.csv`、`*.token_costs.npz`。250 ns/matrix-block 的既有命令成本一直計入；跨 block 不再改變 sense 延遲。

所有 g/w cells 見 `training_candidates.json` 與 `heldout_candidates.json`；held-out grid 僅用於描述，沒有用來重選配置。搜尋／資料 hashes 與凍結紀錄見 `*.frozen_plan.json`、`*.validation.json`。
