# 兩個 row-W8 factors 的獨立實驗

此 revision 接在 GPU 短訓與 spectrum 診斷後，沿用固定 native prototype25 與 same-width rawsubset1408。
原 GU／C／affine／所有 full-C fallback 不修改。cohort及 parent readouts 已使用 calibration VALID 選擇；本輪仍是 calibration 搜索。
TRAIN600 prompts／37,845 rows、VALID200／12,597 rows，context上限64 tokens；不接受或開啟 TEST。

## TRAIN-only activation-aware 分解

對每個既有 dequantized row-W8 C，使用同一 parent beta 所定義的 TRAIN weights：
`w=(1-beta)+beta*a2/mean_TRAIN(a2)`。
在已存在 input affine/A8 的 QDQ feature 上，計算 weighted covariance G。
GPU FP64 求 `C G C^T` 的 top-k output eigenvectors P，固定 `U=P, V=P^T C`。
這是 TRAIN output-error 目標的低秩投影，不重新以 VALID 擬合 factors。

兩個 families 都測 k=128／256／512。先 rank256、seed experts4／51／60 兩方 smoke，
通過6個 canonical模型後一次完成完整12 heads／兩方／三 ranks；本輪不再追加 factor grid。
GPU0為L40S，CPU64–65／2threads；TF32關閉。GPU只計 TRAIN covariance／projection，所有 VALID 皆CPU integer-exact。

## 實際數值路徑

1. 沿用原 input source gather、TRAIN affine 與 whole-vector A8。
2. V row-W8，精確 integer dot後原 frozen F32 rescale，沒有 V bias，產生 raw latent F32。
3. 從完整 TRAIN 的實際 canonical V 輸出，用同 parent weights 算 latent mean/std；variance floor規則同原 mean/std。
4. `F32(F32(latent-center)*inv_scale)` 保留在runtime，然後新的 whole-vector latent A8。
5. U先換成這個 latent座標系 `P/latent_gain`、row-W8；最後 F32 bias以完整 TRAIN 實際兩段 canonical residual均值校準。

VALID只評 naturally selected rows，保留 per-head／跨12 pooled alpha² SSE、BOS／非 BOS，
另按原 slot順序的 F32 alpha乘加重建整層 aggregate，未註冊 experts保持native。
每個模型另與 frozen C++ Numeric 的 V／latent affine／U 逐 bit核對；外部capture agent會獨立驗證7-row完整C++ kernel fixtures。

## Payload／registry 契約

head prefix對應：

- `.V.i8[rank,1408]`、`.V.scales.f32[rank]`
- `.latent.center.f32[rank]`、`.latent.inv_scale.f32[rank]`
- `.U.i8[2048,rank]`、`.U.scales.f32[2048]`、`.bias.f32[2048]`

`.factor.json` version1明示 input_width／width=1408、H=2048、rank、input_transform_prefix與payload_sha256；
最初smoke descriptor僅width alias，formal descriptor同時輸出兩欄。六欄 registry最後一欄為factor headprefix。
input A8可按既有basis共用；latent affine/A8是每個expert自己的新階段，不能算成免費共用 input context。

serialized factor payload bytes為 `rank*(2048+1408)+12*rank+8*2048`，
包括 V/U兩份 row scales、final bias、latent center/gain。
rank128／256／512每head分別460,288／904,192／1,792,000 bytes，原full-C為2,899,968。
原input transform／prototype GU resident bytes／GU工作／新增量化phase仍另計，不能直接將payload減少當作speedup。

只有若局部canonical VALID有可接受Pareto點，才交capture agent實作獨立factor adapter；
既有frozen proto adapter保持不動。root以quality-VALID決定3／6／9／12 eligibility與budget，全部方法與policy凍結後才使用新≥1000-prompt TEST。
