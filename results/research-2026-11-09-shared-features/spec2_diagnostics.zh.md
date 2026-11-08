原六 experts、16-token TRAIN/VALID 的 shared residual 診斷。TRAIN 為 600 prompts／321 articles／9600 rows；VALID 為 200 prompts／3200 rows。所有 feature selection 與擬合僅使用 TRAIN。

下表列出完整 B ridge {.03, .3, 3} 網格的增益範圍；正數代表較匹配 raw-x 的 VALID selected-alpha² NMSE 降低，沒有據此凍結新的部署 winner。raw-x 與 z nuisance ridge 固定為 1，conditional fit beta=0。

| Layer／Expert | β=.99 row N_eff | FP conditional 128 | FP conditional 256 | W8A8 folded 128 | W8A8 folded 256 |
|---|---:|---:|---:|---:|---:|
| 1／14 | 222.4 | -1.47% 至 +0.04% | -3.28% 至 -0.02% | -1.47% 至 +0.04% | -3.28% 至 -0.03% |
| 1／40 | 480.8 | -1.46% 至 +0.05% | -2.97% 至 +0.10% | -1.47% 至 +0.04% | -2.99% 至 +0.09% |
| 13／10 | 500.1 | -0.06% 至 +0.53% | -0.68% 至 +0.64% | -0.08% 至 +0.52% | -0.70% 至 +0.63% |
| 13／56 | 262.7 | -1.22% 至 +0.16% | -2.11% 至 +0.26% | -1.24% 至 +0.15% | -2.14% 至 +0.26% |
| 26／0 | 241.8 | -3.54% 至 +3.73% | -6.29% 至 +3.70% | -4.46% 至 +3.74% | -7.50% 至 +3.69% |
| 26／1 | 545.1 | +4.92% 至 +14.56% | +7.49% 至 +18.79% | +4.83% 至 +12.88% | +7.47% 至 +17.84% |

只有 layer26／expert1 在此網格達到 10% 級別的局部改善；前兩層尚未達新版規格的非最後層機制門檻。所有六個 TRAIN experts 在 BOS 都未被選中，因此 BOS selected-alpha² NMSE 為 null，另保留 all-row NMSE。

這個結果只說明相對固定 ridge 模型的條件增益。z residualization 使用 ridge，仍可能保留因 shrinkage 而未移除的線性 x 訊號；不能直接宣稱額外訊號全來自非線性。B 與 indices 看過整份 TRAIN OOF nuisance residual，TRAIN 分數也不是 B 的無偏 OOF 泛化估計。獨立 VALID 才用來觀察未參與擬合的增益。

FP folding、直接 W8A8 folding、QDQ joint refit 分開報告。實際六個 canonical 模型（2048／2176／2304 三種寬度、各兩個 expert）的 86,016 個輸出 floats 與 frozen C++ 逐 bit 一致；A8 codes／scales 亦相同。此檢查覆蓋數值規則，沒有測量壓縮部署速度。

完整各 λ、位置切片、TRAIN／VALID gap、β 支持度與文章／prompt 聚合 N_eff 保存在同目錄 spec2_conditional_layer*.json 與 spec2_support_layer*.json。未使用新的 TEST。
