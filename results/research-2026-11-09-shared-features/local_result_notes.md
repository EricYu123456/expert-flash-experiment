# 11/09 fitting 與數值旁支交付

依 `docs/1108_2_spec.txt`，主實驗優先原六 experts 的 optimized raw-x／小型 native-shared residual。
這裡保存的是已完成的 prototype 旁支、強線性對照與低曝光 controls；所有數字只屬 calibration-VALID 或明示的 TRAIN 診斷。
沒有開新的 TEST targets，也沒有從 local NMSE 推論 LLM 品質或系統 speedup。

## 先完成的全池診斷

三層64 experts、原生 hot8 prototype pool 的完整擬合已完成，每 expert 只對 TRAIN 排名的兩個 prototypes 完整擬合。
TRAIN 600 prompts／37,845 rows、VALID 200／12,597 rows，context 上限64 tokens；原 grid 是 beta=0/.99、lambda=1/3/10。
排除 native prototype owners 後，prototype/raw-x 的 pooled selected-alpha² SSE 比例為 L1 1.068、L13 1.099、L26 1.143。
384 個 raw/prototype winning payloads 的 C++ bit audit，共核對3,145,728個 output floats，全部通過。

少量 L1 experts 有反例。按 BOS／非 BOS 分開後，prototype25 對 experts 4／51／60 的非 BOS SSE 改善約11.45%／14.34%／11.35%。
四 beta=0/.5/.9/.99 的公平重擬合仍保留這三個改善；同一 prototype 確實供三個不同 experts 使用。
experts 27、63 亦有局部改善，但後續 bank 固定單一 L1 prototype25。
expert57 雖有非 BOS 改善，其 alpha² mass 約78.3%在 BOS，未納入這個種子 cohort。
article bootstrap 區間是 VALID 選參後的描述性診斷，不能當作獨立 holdout。

## 同寬度與低曝光強控制

raw-x1408 以 fullrawx 已部署 W8 C 的 pooled column squared norm 選1408個座標。
因 parent C 曾用 VALID 選 lambda/beta，這個選法明確是 calibration-selected。
座標 gather、TRAIN affine 之後重新 whole-vector A8，再 fit QDQ 輸入，沒有新增 GU。
原六加五個候選共11 heads，以及後續 prototype25 新增九 heads，皆使用同四 beta／三 lambda。

種子三 heads 的 selected-alpha² NMSE：

| Expert | Raw-x2048 | Prototype25-1408 | Raw-x1408 |
|---|---:|---:|---:|
| 4 | 0.11795 | 0.10444 | 0.12439 |
| 51 | 0.24193 | 0.20723 | 0.25269 |
| 60 | 0.14988 | 0.13288 | 0.15682 |

zero、TRAIN naturally-selected mean、TRAIN selected-alpha² mean 都匯出標準 width1 readout。
其 source0／index0／center0／gain1、C 全零、row scales 全一，F32 bias 為零或對應 TRAIN 均值。
均值用 FP64 累加後一次轉 F32；alpha² mean 是相同局部 least-squares 目標的最佳常數，未看 VALID labels。
原16-token六 heads 以及64-token新 cohorts 分開保存，不能混為同一 calibration 資料量。

## 固定單一 prototype25 的 coverage 檢查

immutable cohort 順序為 `4,51,60,58,5,17,62,22,1,28,38,50`，保留3／6／9／12個 prefixes。
rawfull、prototype25、same-width rawsubset1408、zero、兩種 TRAIN mean 都有相同 eligibility registries。
原種子三 heads payloads 完全重用；新舊 transform 在統一 prefix 前逐 byte 核對相同。
72個部署模型的 transform／A8／W8／bias C++ bit audit 全部通過。
只有種子三 heads 保留至少10%的非 BOS 局部增益，新增九 heads 是否值得替換由 quality-VALID curve 決定。

runtime 的非線性 owner union 只有 `[25]`；舊 target manifest 的八個 prototype IDs 是可用離線 pool，不能當成本次部署 bank。
prototype25 GU 本身另需5,778,432 bytes 的 W8/scales，且每個實際 prototype/input row 都要付 GU MAC／SwiGLU 成本。
只看 readout 變小會高估收益；例如三個 prototype readouts 的 C 節省仍小於一份新增 GU 的係數成本。
必須另比較相同 hot-GU cache、同 host 預算下的 rawfull／rawsubset，以及可全留 host 的 C。

完整 fitting sources、commands、logs、payload hashes 和 registries 位於 NFS 的
`research-2026-11-09-shared-features`，repo 只保存小型 JSON／CSV receipts。
3／6／9／12 coverage 的50／200題 quality-VALID 與最終新 article-held-out ≥1000 prompts 由 root 執行。
joint aggregate correction 程式及數值 smoke 已備妥；依新實驗優先序未追加正式 joint 搜索。
