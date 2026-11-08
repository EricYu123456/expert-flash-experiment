# 最終 local／fitting 交付：原生 shared、單 prototype 與低秩 readouts

本文件只結算 fitting 與 calibration-VALID。**模型級 quality-VALID、相同品質下的可消除工作量，以及新≥1000-prompt TEST 尚由 root 執行／結算，這裡不填入結果。**
所有既有 full-C fallback、selection、payload 與 registries 保持 immutable。本輪 factor128／256／512 完成後不再開新訓練 grid；root已在 TEST 前選定 rank256 為唯一 compressed live 主候選。

## 原 native-shared 六 experts：沒有足夠的增量

`spec2-screen16` 對原六 experts，同時給 raw／shared／hybrid runtime equalization、QDQ-input fitting、beta=0/.5/.9/.99、lambda=1/3/10。
資料為原 TRAIN600×16=9600 rows、VALID200×16=3200 rows。下表是 selected-alpha² NMSE，最後一欄為 hybrid128／256 中較佳者相對 raw 的誤差改善。

| Layer／Expert | Optimized raw-x | Optimized full shared-z | 小 hybrid 最佳改善 |
|---|---:|---:|---:|
| 1／14 | 0.81799 | 0.86845 | −0.13% |
| 1／40 | 0.67547 | 0.70439 | 0.18% |
| 13／10 | 0.81207 | 0.84618 | 0.46% |
| 13／56 | 0.83948 | 0.86386 | 0.05% |
| 26／0 | 0.71282 | 0.75987 | 2.89% |
| 26／1 | 0.13274 | 0.16307 | 1.33% |

full native shared-z 六個都輸給同步優化 raw。小 hybrid 的早層增益不到0.5%，沒有達到多 expert／非最後層約10%增量的投入門檻。
量化修正本身已正確保留在 A8 前，但不足以成立 native-shared 重用的主張。這是局部機制結果，不能拿來代填 LLM 品質。
同 eligibility 的 zero／TRAIN mean／TRAIN alpha² mean 控制已匯出；低曝光下是否也能通過品質界線，須看 root 的 quality 評估。

## 單一 prototype25：種子三 heads 有正例，全12 pool 仍 raw 略優

此旁支使用64-token captures：TRAIN600 prompts／37,845 rows，VALID200／12,597 rows，與上面的16-token主表分開。
完整 hot8 pool 搜尋後，固定 L1 native prototype25，seed為 experts4／51／60；三個的非 BOS SSE 對 raw2048 分別改善11.45%／14.34%／11.35%。
兩方補齊四 beta／三 lambda 後仍保留；同一 GU 確實供三個不同 experts 使用。
article bootstrap是 calibration 選參後的描述性區間，不是新的 holdout。

將 cohort按已凍結 VALID 排序擴成 `4,51,60,58,5,17,62,22,1,28,38,50`，保存3／6／9／12 prefixes。
rawfull、prototype25、同寬 rawsubset1408、zero、兩種 TRAIN mean 全有相同 eligibility。
rawsubset1408 座標來自 parent fullrawx W8 C 的 pooled column norm；parent C曾用 VALID選參，因此明標 calibration-selected。
它不需要額外 GU。全12 dense pool 的 NMSE為 prototype25 0.547786、rawsubset約0.54147，不能把種子三的正例說成全12都勝出。

## 共享 GU 短訓：這個固定 C／affine grid 增益太小

後續使用者授權 GPU。L40S僅做 TRAIN FP32 STE／Adam，TF32／AMP／compile關閉；VALID保持CPU64–65、2threads、integer-exact W8A8與pinned SwiGLU。
固定完整12 heads的 C／affine，只更新 GU；learning rates0.0001／0.0003／0.001、各64 steps、checkpoints0／16／32／64。
step0直接用原W8 codes/scales，逐head baseline SSE／energy一致，10,379,776個native phi floats逐bit相符。

最佳lr0.0001／step64：pooled NMSE0.547786→0.544903，SSE只改善0.526%；seed3改善3.87%、nonBOS pooled改善0.528%。
另外兩個lr最佳都停在step0。這輪沒有新增 learned-GU runtime；若保留新舊兩個GU banks，還要多付一整份5,778,432 bytes與不可去重的GU工作。

## 兩個 row-W8 factors：保留 latent affine／A8 的真數值驗證

對兩個 families的固定 parent C，在 TRAIN QDQ features上用 parent beta 的 weighted covariance G，
GPU FP64求 `C G Cᵀ` top-k output eigenvectors P，再設 `U=P,V=PᵀC`。
實際候選為 input原affine/A8→row-W8 V（無Vbias）→TRAIN weighted latent center/gain→新latent A8→row-W8 U＋F32 bias。
latent affine留在runtime；final bias用完整TRAIN的實際兩段canonical residual校準。
所有 VALID 數值都在CPU integer-exact產生；原full-C fallback未改。

全12 pool 結果如下。SSE增幅相對**各自**full-C；BOS與非BOS亦完整保留。

| Family | Rank | Pooled NMSE | SSE增幅 | NonBOS NMSE | BOS NMSE | SSE增幅≤5%的heads |
|---|---:|---:|---:|---:|---:|---:|
| Prototype25 | 128 | 0.560283 | 2.281% | 0.584884 | 0.00002084 | 8／12 |
| Prototype25 | 256 | 0.551911 | 0.753% | 0.576144 | 0.00002092 | 12／12 |
| Prototype25 | 512 | 0.548797 | 0.185% | 0.572893 | 0.00002039 | 12／12 |
| Rawsubset1408 | 128 | 0.556952 | 2.859% | 0.581407 | 0.00001913 | 11／12 |
| Rawsubset1408 | 256 | 0.546903 | 1.003% | 0.570916 | 0.00001875 | 12／12 |
| Rawsubset1408 | 512 | 0.542808 | 0.247% | 0.566641 | 0.00001876 | 12／12 |

prototype seed3在rank128的SSE增幅為6.66%／6.36%／7.99%，seed pool增6.919%，超過5%局部門檻。
rank256為1.89%／1.75%／2.35%，seed pool增1.956%；rawsubset同rank seed pool增0.779%。
rank256後seed pool NMSE仍為prototype0.159416、rawsubset0.189527，保留種子上的局部優勢。
rank512的seed pool增幅prototype0.657%、rawsubset0.511%，但容量較大。

BOS相對誤差不能隱藏：rank256 prototype的BOS SSE約是dense的1.80倍，但絕對NMSE只有2.09e−5；raw約1.78倍、1.87e−5。
主結論同時看非BOS：全12仍raw略優，種子三有非BOS正例。local指標包括capture的全部 retained positions，不等同63-position NLL分母。

72個模型皆通過 fitter內的 frozen C++ V／latent affine／U逐bit probes。
另已交capture agent全部72模型×7row fixtures，供獨立整kernel驗證；實驗handoff保存兩方12heads、三rank、3／6／9／12 cohorts。
root根據rank256在種子與全12兩方都通過局部≤5% gate，凍結256為唯一live rank；不能開TEST後再從128／512選較好者。

## Payload、MAC與strong raw剩餘cache控制

完整factor binary bytes包含V／U、兩份row scales、final bias、latent center/gain：

| Rank | 每head binary bytes | 相對full-C下降 | 每head V＋U coefficient MAC |
|---|---:|---:|---:|
| 128 | 460,288 | 84.13% | 442,368 |
| 256 | 904,192 | 68.82%（約69%） | 884,736 |
| 512 | 1,792,000 | 38.21% | 1,769,472 |

原full-C為2,899,968 bytes。input transform、prototype25 GU、GU計算與新增private latent affine/A8 phase仍另計。
rank256 binary payload減少不代表flash或延遲改善；目前是可供quality–cost測試的容量候選。

review `research_20261109_cost_audit.py` 後，同意7個binary payload＋每head4B rank作runtime coefficient inventory。
`.factor.json`的絕對路徑／SHA是frozen artifact provenance，另列bytes，不算resident coefficient B；否則只因路徑字數就會使raw少792B而不能放一整GU。
實際檢查k3／6／9／12，兩方同rank readout bytes相等，method B−raw B都恰為GU5,778,432 bytes。
k12 rank256 method B為16,652,720，raw為10,874,288；strong raw可合法將剩餘budget配置到其他native full／GU／D units。

只看fixed whole-matrix byte-work目標，frozen VALID頻率的role-count枚舉合法；排除已resident readout experts避免double credit。
V＋U MAC=`rank*(1408+2048)`正確，input A8按basis去重，private latent A8每替換expert/event一次；額外GU只計required但未naturally selected的owners。
此B仍是**incremental coefficient residency budget**，method-specific activation workspace只列shape；不是完整host peak memory。
MAC只計coefficient乘加，quant／affine／rescale另有context/value計數，不能當總CPU工作或TPOT。
本次review沒有修改root檔；具體核對與source SHA見 `factorized_cost_review.json`。

## 尚未結算的判斷

- rank256新adapter的完整model-level quality-VALID與EOG／context驗證：由root／capture結算。
- 相同品質容忍值、相同完整部署預算下是否勝過strong raw＋剩餘native cache：待root成本／品質曲線。
- 凍結方法與eligibility後的新article-held-out≥1000 prompts TEST：未由本fitting流程開啟或填入結果。

可重現sources／commands／logs／payloads都在NFS `research-2026-11-09-shared-features/factorized-readouts`。
repo只保存小型JSON／CSV／receipts；逐head72列見 `factorized_local_per_expert.csv`，local aggregate與BOS細目見 `factorized_local_summary.json`。
