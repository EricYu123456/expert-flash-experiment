# Native shared features：CPU 實驗結果

已完成 600 TRAIN、200 VALID、1000 TEST prompts；TEST 來自 61 個 WikiText-2 articles。
每題使用含 BOS 的前 16 個 tokens，評 15 個 next-token targets，另生成最多 8 tokens。

主要品質判定：**passes_this_pilot_margin**。這只適用固定六 experts 的短上下文配置。

| 真正執行的模型 | Mean NLL | Subset PPL | 相對 native PPL | Article-cluster 95% CI |
|---|---:|---:|---:|---:|
| native | 4.546572 | 94.3086 | reference | — |
| shared_z_all6 | 4.551967 | 94.8187 | +0.541% | [-0.072%, +1.194%] |
| raw_x_all6 | 4.548216 | 94.4637 | +0.164% | [-0.400%, +0.727%] |

2% relative subset-PPL 容忍值在測試前固定；僅 primary shared-vs-native interval 用於該品質判定。
Native capture 與 quality runner 的全部 15000 個 NLL／top1／token records 完全一致。

## 本輪能回答什麼

在這個 checkpoint／校準資料／完整 native shared basis 的配置中，shared pre-Down readout 的 W8A8 NMSE 六點全輸更小的 raw x readout。
因此，即使少量替換的整體品質落在容忍值內，也不能據此證明保存這份 shared nonlinear basis 比便宜的線性 input control 更有價值。
本輪未訓練新 feature pool、未做 router-weighted readout、未做 quantization-aware fitting，也沒有否定這些其他配置。

## 局部 W8A8 readout NMSE（1000 個 TEST prompts）

| Layer | Expert | raw x | shared output | x+shared output | shared pre-Down | random SwiGLU |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 14 | 0.7862 | 0.8440 | 0.7941 | 0.8187 | 0.8395 |
| 1 | 40 | 0.7425 | 0.8148 | 0.7487 | 0.7880 | 0.8230 |
| 13 | 10 | 0.7192 | 0.7940 | 0.7260 | 0.7746 | 0.8202 |
| 13 | 56 | 0.7466 | 0.7831 | 0.7473 | 0.7671 | 0.8275 |
| 26 | 0 | 0.4294 | 0.4570 | 0.4305 | 0.6332 | 0.4606 |
| 26 | 1 | 0.2085 | 0.3077 | 0.2152 | 0.5096 | 0.3140 |

x+shared output 為 4096 維，其他新 nonlinear bases 為 2816 維；不能將較大容量當同成本優勢。
完整 FP／W8A8、selected-only NMSE、cosine、各層 aggregate、ridge 與容量數值見 quality_summary.json 及 NFS fits-refined。

以下只評 native router 真正選中該 expert 的 token rows：

| Layer | Expert | Selected rows | raw x NMSE | shared pre-Down NMSE |
|---:|---:|---:|---:|---:|
| 1 | 14 | 1146 | 0.8564 | 0.9081 |
| 1 | 40 | 1796 | 0.7133 | 0.7580 |
| 13 | 10 | 3521 | 0.6609 | 0.7021 |
| 13 | 56 | 577 | 0.8689 | 0.9111 |
| 26 | 0 | 705 | 0.6097 | 0.6988 |
| 26 | 1 | 1881 | 0.2244 | 0.5710 |

TRAIN 擬合對所有捕捉 x 使用相同權重，未依 router eligibility／alpha 加權。更大量的 routed-only 校準是尚未驗證的其他配置。

## 真正生成與路由

| 替換版本 | 與 native 完整 continuation 相同率（最多8 tokens，EOG停止） | positional token match | NLL causal replacements |
|---|---:|---:|---:|
| shared_z_all6 | 51.60% | 73.79% | 9017 |
| raw_x_all6 | 51.10% | 73.72% | 9011 |

後續 hidden states、routing、KV 與 greedy token 均重新執行；生成一致率不是任務 accuracy。
NLL exposure 只算 input positions0..14，最後 prefill 及未用的最後 decode prediction 分開列在 JSON。

## 為何晚層的 W8A8 特別差

VALID-only 歸因顯示：layer26 的 shared z 經 whole-vector A8 後，每向量中位 90.13% features 變零；amax/RMS 中位為 41.95。
下表固定同一份 W8A8-selected C/bias，僅改診斷量化邊界，未訓練新候選：

| Expert | 同 C 全精度 | 僅 C row-W8 | 僅 z vector-A8 | 正式 W8A8 |
|---:|---:|---:|---:|---:|
| 0 | 0.42288 | 0.42482 | 0.59703 | 0.62570 |
| 1 | 0.23868 | 0.24216 | 0.46694 | 0.50530 |

## 容量與 flash 工作量界線

六個原生 experts 含 W8 scales 共 49.6113 MiB；改為六個 full-width C/bias/scales 共 33.0938 MiB。
刪掉對應原生 FFN 才淨省 16.5176 MiB；保留全部原生 fallback 則額外增加 33.0938 MiB。
三層 resident shared GU 為 33.0645 MiB；再加六個 C 共 66.1582 MiB，另需 shared Down／非 FFN／KV／workspace。
這些 C 可全放 host 的預算下，必須比較 host-resident readout，不能刻意送 flash；同樣六個 raw-x C 只需約24.09 MiB。
本 pilot 的靜態 routed bank coefficient saving 只有 0.120192%。
固定 native TEST routing、無 cache、只計 routed 係數的反事實工作量節省為 0.128552%；這不是 live candidate 的實測 flash 讀量，也不是 TPOT speedup。
全64 experts／26層的寬度與host容量外推另列cost_audit.json，未驗證其品質。

## 解讀與限制

- 16-token English WikiText paragraph starts; 61 correlated article clusters, not long-context/multitask accuracy or standard benchmark PPL.
- Only six fixed experts in raw layers1,13,26 are replaced; other experts/layers and the native shared branch remain active.
- Native reference is canonical FFN row-W8/vector-A8 over a Q8_0 GGUF checkpoint, not BF16 or bit-identical original GGUF inference.
- FP readouts use a separate declared diagnostic accumulation contract; only W8A8 readouts enter live quality arms.
- Functional harness still computes original FFN and duplicates shared-GU verification; wall times are not compressed deployment performance.
- Eight greedy tokens measure early divergence only; sequence agreement is not answer correctness.
- NLL exposure counts prefill positions0..14 only; final prefill and final unused decode are listed separately.
- Secondary intervals are descriptive, individually95%; primary quality endpoint is shared_vs_native only.
- No actual flash reads, NVMeVirt latency, complete-model speedup or novelty claim is established by these experiments.

## 重現與產物

- 協定：`docs/research_2026_11_08_shared_features_protocol.md`。
- corpus → capture_run → fit --defer-test → evaluate → quality_run → report；各實際指令與 SHA 保存在 NFS。
- 大型產物：`/home/nfs_client/ericyu/expert_flash_experiment/research-2026-11-08-shared-features`。
- `cost_audit.json`／`cost_static.json` 區分 fixed-six 真正成本與全層／全 expert 假想外推。
- 前次較粗 ridge grid 只用於 VALID 診斷，保留於 `fits`；正式 `fits-refined` 使用同樣十個 λ 比較所有 bases。
- 第一個 quality launch 在任何 TEST candidate 啟動／TEST 結果檢視前中止，保留於 `quality-first-grid-cancelled`；正式結果只使用 refined selections。
- 資料：[Salesforce WikiText pinned source](https://huggingface.co/datasets/Salesforce/wikitext/tree/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-2-raw-v1)。
