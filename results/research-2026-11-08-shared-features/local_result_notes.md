# 局部 readout 結果

**1000 個 held-out prompts 上，native shared pre-Down readout 在六個固定 experts 都弱於較便宜的 raw-x readout。**
每 expert 有 16000 個 native-state rows；兩方法皆為重新擬合 C/bias、canonical W8/A8，完全不以 TEST 選參。

| 原始層 / expert | raw x NMSE | shared z NMSE |
|---|---:|---:|
| 1 / 14 | 0.78624 | 0.81873 |
| 1 / 40 | 0.74253 | 0.78796 |
| 13 / 10 | 0.71921 | 0.77460 |
| 13 / 56 | 0.74661 | 0.76714 |
| 26 / 0 | 0.42939 | 0.63325 |
| 26 / 1 | 0.20847 | 0.50962 |

shared z 在前四點優於 shared output 與同寬 random SwiGLU，最後兩點則較差。
只看原生 router 選中的 rows，raw x 同樣六點勝出；所有五 bases、FP/W8A8、cosine 與聚合值見 JSON/CSV。

TRAIN/VALID 為 600/200 prompts、9600/3200 rows，prompt IDs 與 TEST 無交集。
原始 `fits` freeze 保留不改；在尚未檢視 TEST candidate/target 前，純 VALID 補密 ridge 到
`[1e-6,1e-4,.01,.03,.1,.3,1,3,10,100]`。正式只使用 `fits-refined`，三層全部保存 selection 後才開 TEST。
full width 2816 的 TRAIN Gram 滿秩，N/r=3.409，但同 prompt tokens 有相關性，TRAIN/VALID 仍有泛化差距，不能宣稱沒有過擬合。

Layer 26 的主要量化問題是 shared z 的 **whole-vector A8**。
VALID 中每 row 中位數 90.13% features 變零，amax/RMS 中位數 41.95。
固定同一 C/bias 時，expert 0 的 NMSE：FP .42288、僅量 C .42482、僅量 z .59703、W8A8 .62570；
expert 1 為 .23868/.24216/.46694/.50530。僅 C row-W8 的影響小很多。
這是歸因診斷，不是新候選；沒有依此重新訓練或調 TEST 模型。

實際六個 readouts 的 W8/scales/bias：shared z 33.0938 MiB，raw x 24.0938 MiB；原 native 六 experts 49.6113 MiB。
33.0938 MiB host 預算足以把六個新 C 全放 host，或放四個 native experts；host-resident readout 必須是強控制。
raw coefficient reduction 不代表系統 speedup。

限制：16-token 短上下文、固定三層六 experts、all-row 而非 router-weighted TRAIN，random control 未經最佳化。
native selected-slot coverage 僅 2.69–4.27%，小 aggregate 誤差會受低 exposure 稀釋。
本文件只判局部機制，**不據此判定完整 LLM 品質**；真 autoregressive 結果見主報告。
量化後 bias 校準、quantization-aware readout 或新 basis 都是後續 protocol，並未在本輪證明無效。

`fit_validation_test_receipt.json` 與 `fit_numeric_self_test.json` 保存四個專用 tests、1000-prompt synthetic guard、
full-width exact dot、near-half rounding、原 C++ quantizer/rescale 逐 bit 一致的通過記錄。
`local_artifacts_manifest.json` 綁定本目錄小產物的 SHA256；大模型與 activation tensors 都保留在 NFS。

```bash
python3 /home/ericyu/expert_flash_experiment/scripts/research_20261108_shared_feature_local_artifacts.py --kind plot --fit-dir /home/nfs_client/ericyu/expert_flash_experiment/research-2026-11-08-shared-features/fits-refined --output-dir /tmp/shared-feature-local-plot
taskset -c 56-63 env CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 python3 /home/ericyu/expert_flash_experiment/scripts/research_20261108_shared_feature_local_artifacts.py --kind diagnosis --fit-dir /home/nfs_client/ericyu/expert_flash_experiment/research-2026-11-08-shared-features/fits-refined --output-dir /tmp/shared-feature-VALID-diagnosis
```

詳細 fit 審核：`docs/research_2026_11_08_shared_feature_fit_audit.md`。
