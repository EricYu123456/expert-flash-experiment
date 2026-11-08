# Native shared-feature readout：擬合、局部 TEST 與數值審核

本文件只報告固定 native hidden states 上的局部 readout 診斷；真正 autoregressive 品質由主實驗另行報告。
實驗依據為 `1108_spec.txt`，採 CPU，未改歷史程式或模型權重。

**本次六個固定 experts 的結果，不支持 native shared pre-Down features 優於較便宜的 raw-input readout。**
在 1000 個 held-out TEST prompts 上，canonical W8/A8 的 shared-feature readout 六點都輸給 raw x。
它在前兩個 layers 優於 shared output 與同寬 random SwiGLU，但最後一層的 whole-vector A8 丟失許多 readout 所需的小特徵。
這是本 checkpoint、三層六 experts、16-token 上下文與本次線性校準方法的結果，不能推成所有 shared nonlinear bases 都無效。

## 輸入與防洩漏

- 原始層：1、13、26；experts 分別為 14/40、10/56、0/1，皆事前固定。
- 每個固定 expert 在所有 token 計算 native target，避免僅取 router-selected rows 導致高維 readout 欠定。
- TRAIN 600 prompts / 9600 rows；VALID 200 / 3200；TEST 1000 / 16000，每 expert 各有全部 rows。
- 直接讀四個 worker shards，使用 `rows.jsonl` 的全域 prompt ID，禁止把 shard-local integer indices 當作全域 ID。
- 三個集合的 prompt IDs 無交集。來源、各 raw array、metadata、模型與 deploy exports 均有 SHA256。
- `--defer-test` 的 FIT 完全不開 TEST capture。各層先寫 `selection.json`；三層全部 freeze 後才啟動 TEST。
- 最終 grid 為 `1e-6, 1e-4, .01, .03, .1, .3, 1, 3, 10, 100`，所有五個 bases 共用。
  初始較粗的 VALID grid 產物保留在 `fits`；在未檢視任何 TEST candidate/target 前，以 VALID 補密，正式結果只用 `fits-refined`。
- TEST 不挑 expert、basis、ridge、threshold 或部署模型。FP 與 W8A8 各自使用 VALID 最低 NMSE 選 ridge。

## Readout 與控制

對每個 basis，TRAIN 的 feature mean/std 與 target mean 決定 centered ridge：

`min_B ||X_std B - Y_centered||² / N + lambda ||B||²`。

使用 FP64 Gram eigendecomposition，所有負 eigenvalues 必須位於 roundoff 容忍範圍內；
同 basis 的分解供各 expert 重用。最後把 feature normalization 折回原始 feature 座標的 `C,b`，
存 F32，之後才做 row-W8。`C` 是從 target 重新擬合，沒有使用原 expert Down 硬接 shared z。

TRAIN rows 大於所有 feature widths：raw x / shared output 為 2048、x+shared output 為 4096、shared z / random SwiGLU 為 2816。
三層所有 bases 的 Gram numerical rank 均滿秩（相對 cutoff `1e-10`）。shared z 的 rows/features 為 3.409；
實際 W8A8 選中 readouts 的 effective degrees of freedom 為 406–1022。
這排除了代數上因 rows 不足而欠定；同 prompt 的 tokens 仍相關，不能宣稱沒有過擬合。
固定 readout 的 TRAIN/VALID NMSE 已另存，前兩層 shared z 可見約 0.13–0.24 的泛化差距。

同寬 input-only control 為固定 seed 的 random SwiGLU。
TRAIN input normalization 折成 GU weights/bias，GU 使用 row-W8、原始 x whole-vector A8，再計 F32 SwiGLU。
它與 native shared generator 有相同 nominal GU/readout 係數數目和 2816 寬度，並非訓練過的最強 nonlinear predictor。
保留原 native shared branch 時，random control 另外增加自己的 GU，因此總 host 容量較大；此額外成本明列，沒有視為免費。
x+shared output 的 4096 維則是較高 readout 容量參考。

## 1000-prompt TEST 結果

下表為全部 16000 個 native-state rows 的 expert-output NMSE，越低越好；每個模型都只由 VALID 選 ridge。

| 原始層 / expert | raw x (2048) | shared output (2048) | x + shared output (4096) | shared z (2816) | random SwiGLU (2816) |
|---|---:|---:|---:|---:|---:|
| 1 / 14 | **0.78624** | 0.84401 | 0.79408 | 0.81873 | 0.83952 |
| 1 / 40 | **0.74253** | 0.81475 | 0.74867 | 0.78796 | 0.82297 |
| 13 / 10 | **0.71921** | 0.79395 | 0.72596 | 0.77460 | 0.82023 |
| 13 / 56 | **0.74661** | 0.78311 | 0.74730 | 0.76714 | 0.82750 |
| 26 / 0 | **0.42939** | 0.45702 | 0.43050 | 0.63325 | 0.46060 |
| 26 / 1 | **0.20847** | 0.30768 | 0.21515 | 0.50962 | 0.31401 |

只看原生 router 確實選中的 rows，raw x 也六點都勝過 shared z。
依上述順序，shared z 的 selected-only NMSE 為 `.90810, .75799, .70212, .91108, .69876, .57100`；
raw x 為 `.85643, .71333, .66091, .86886, .60966, .22436`。

原生 slot coverage：layer 1 為 3.0646%、layer 13 為 4.2688%、layer 26 為 2.6938%。
保留其他 native slots 與 shared output、只替換指定兩 experts 後，整層聚合 NMSE 如下：

| 原始層 | raw x | shared z |
|---|---:|---:|
| 1 | 0.003533 | 0.003729 |
| 13 | 0.036573 | 0.038488 |
| 26 | 0.000121 | 0.000377 |

聚合按原生 slot 順序，以 F32 alpha multiplication / addition 重新組裝，未使用抵消誤差的高精度捷徑。
低 coverage 與大量 unchanged output 會稀釋整體誤差；小 aggregate NMSE 不代表 expert 重建良好。
它也不是後續 hidden state、router、KV 已改變時的 LLM 品質結論。

## 數值契約與 layer 26 量化歸因

canonical readout 契約：row-W8 與 whole-vector A8，F32 `amax/127`，zero scale=1；
先做 F32 quotient，再採 `std::round` ties away from zero 到 `[-127,127]`。
整數 dot 精確累加，接著 `F32(dot) * F32(sw*sx)`，最後獨立 F32 bias addition。
FP 診斷則為 stored-F32 C / FP64 dot / cast-F32 / F32 bias，不冒充既有 C++ FMA reduction 位元契約。

獨立檢查涵蓋 full width 2816 的 signed integer dot、零向量與 adversarial near-half rounding。
Python quantizer/rescale 與歷史 immutable C++ canonical primitives 逐 bit 相同。
合成可識別 shared teacher 的 TEST NMSE 為 `2.13e-15`；獨立 basis 負控制為 `0.928`。
四個專用 tests 通過，包括真正 1000-prompt synthetic end-to-end、TEST read guard、deferred TEST、shard IDs、欠定與 overlap fail-closed。

量化歸因只讀 VALID，固定同一個 W8A8-selected C/bias，不產新候選：

| layer 26 expert | 同 C，FP | 只量化 C | 只量化 z | canonical W8A8 |
|---|---:|---:|---:|---:|
| 0 | 0.42288 | 0.42482 | 0.59703 | 0.62570 |
| 1 | 0.23868 | 0.24216 | 0.46694 | 0.50530 |

layer 26 shared z 的 A8 編碼每 row 中位數有 **90.13% 為零**，amax/RMS 中位數為 **41.95**。
z 整體重建 NMSE 僅 0.00230，但其誤差經 readout 放大；只量 z 對 FP predictor 造成相對 NMSE 0.296/0.277，
只量 C 則為 0.00276/0.00404。主因為 whole-vector A8 與 outliers 使需要的小特徵消失。
目前 C/bias 從 FP feature ridge 折回後量化，沒有 quantization-aware retraining 或量化後 bias 重新校準；
後續可以另設 TRAIN-only protocol 研究，不能把本輪結果宣稱為所有合法量化校準的最優下界。

## 容量控制與成本界線

原 routed expert 為 `3HI` 係數，full shared readout 為 `Hr`；`H=2048,I=1408,r=2816` 的 raw 下降為 33.3%。
它不等於系統 speedup，仍有 shared feature readiness、host bandwidth、傳輸、NAND 與 fallback 成本。

| 實際六個 experts，W8 + row scales/bias | 容量 |
|---|---:|
| 原 native GU/Down | 49.6113 MiB |
| full shared-z readouts | 33.0938 MiB |
| raw-x readouts | 24.0938 MiB |

33.0938 MiB host 預算可以保存全部六個 shared-z readouts，或四個完整 native experts。
因此這個小範圍若通過品質，**把全部六 C 放 host** 是必要強控制，不應刻意送 flash。
若原始 native fallback tensors 仍保留，容量要加上它們，不能只列新 C。

全部 64×26 experts 的容量僅做算術外推：full r2816 readouts 為 8.9629 GiB；r2112 為 6.7285 GiB；r1408 為 4.4941 GiB。
後兩個 feature widths 沒有量品質，不能稱為已驗證模型。全模型 shared GU 的 row-W8 含 scales 約 286.56 MiB，
還需 native shared Down、non-FFN weights、KV、workspace、metadata，不能把係數預算當完整 host footprint。

## 可重跑產物

- 程式：`scripts/research_20261108_shared_feature_fit.py`；專用 tests：`tests/test_research_20261108_shared_feature_fit.py`。
- 所有大產物根目錄：`/home/nfs_client/ericyu/expert_flash_experiment/research-2026-11-08-shared-features`。
- 正式 fitting/evaluation：`fits-refined/layer{1,13,26}`；各自有 `selection.json`、`registry.tsv`、models、`test_report.json` 與 per-prompt stats。
- 彙整：`fits-refined/local_TEST_summary.json`、`local_TEST_per_expert.csv`、`local_TEST_nmse.png` / `.svg`。
- 診斷：`fits-refined/layer26.quantization_VALID_diagnosis.json`、`TRAIN_frozen_model_fit_diagnosis.json`。
- 成本：`capacity-comparison.json`、`coefficient-costs.json`。
- CPU affinity、完整 argv、source SHA、exit code 與 logs：`fits-refined/layer*.launch.json`、`layer*.evaluate.launch.json` 等。
  FIT 三層各約 115–120 秒；TEST evaluation 各約 55–60 秒，僅為本機 offline 工具 wall time。

```bash
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 python3 -m unittest discover -s /home/ericyu/expert_flash_experiment/tests -p test_research_20261108_shared_feature_fit.py -v
```
