# 共享非線性基底：第二輪 CPU 優化協定

日期：台北 2026-11-09。延續 `docs/1108_spec.txt` 與 11/08 真模型結果。
本輪的目標是改善品質與可消除的係數工作量，並超越同樣校準過的 raw-input 線性控制。
不將只保留六個低 exposure experts 的小 PPL 變化解讀成通用方案成功。

## 使用者補充 1108_2_spec 後的優先序修訂

在任何新 TEST 推論之前，使用者提供 `docs/1108_2_spec.txt`。主問題改以原六 experts 的
raw-x＋128／256 shared residual 為優先；已完成的64-token全expert targets／prototype fitting
保存為獨立探索，不能代替小量 shared features 的增量證據。原 native-shared 分支未達機制門檻；
另有單一 routed prototype25 在早層多 expert 的局部改善超過10%，因此僅對此固定小群補充
GPU GU-only 訓練與 readout 低秩化，正式品質仍以 CPU canonical W8A8 重算。

- 先用原16-token、600 TRAIN／200 calibration-VALID 做固定六expert因子矩陣：
  raw-x／full shared-z／hybrid128／256／512，β0/.5/.9/.99，λ1/3/10。
- 以3-fold article-grouped TRAIN OOF residual選J，獨立VALID量測條件增益；
  nuisance z~x 的 ridge1/.01/.001 分開檢查，不能將ridge造成的剩餘線性訊號稱純非線性。
- 保留已輸出的最佳raw-x，另擬小B、重新fold／量化／校準，另外保留B=0 concat編碼控制。
- 固定六expert的drop／自然selected TRAIN mean／selected α² TRAIN mean皆實際跑quality-VALID。
- 報row／prompt／article的weight N_eff與BOS／nonBOS／position分層；同BOS重複值不作獨立新訊號。
- 約10%局部誤差改善、涵蓋多expert且至少一個非末層，作後續投入的機制門檻，
  不作模型品質保證。未達標時不靠增加prototype數或predictor拯救native-shared分支。
- 已完成的prototype探索只對少數有非BOS增益的cohort補齊gentle β與同寬rawx控制，再決定
  是否值得任何live品質／成本投入；不將全64專家擬合完成等同部署成功。

品質執行各plan明記16或64-token輸入：spec2固定六的主比較維持16-token校準契約，
64-token探索與其對照另列。新1000篇article-heldout TEST在最後configuration、prefix長度與
所有payload/source hashes凍結後一次開啟；同一比較兩臂必須使用相同prefix與target positions。

## 已知問題與本輪候選

11/08 native shared-z 的六個 readouts 全部弱於 raw x；晚層還有 whole-vector A8 outliers。
本輪僅用 TRAIN／VALID 搜尋下列方向，原 TEST 已看過，不再作新盲測：

1. TRAIN affine equalization 留在 runtime、在 transformed features 上 A8，直接使用實際 QDQ features 擬合。
2. 依原 router selected alpha² 加權 ridge，混入少量 all-row 權重，並在 TRAIN 校準量化後 bias。
3. raw x + 512/768 shared features、縮窄 native shared features，總 readout 寬度受 `r < 3I = 4224` 限制。
4. 改變 resident basis：取 TRAIN routing 熱度最高八個 routed experts 的 native GU 為 prototypes，
   各 expert 經 TRAIN 篩選、VALID 選擇 prototype/readout。其餘新 GU weights、host 運算及多 input context 均須計入。
5. 必要時以 TRAIN 真 routed aggregate residual 做共用 readout correction，再折回 C/bias，
   VALID 選擇 correction step；部署不增加一個免費 residual head。
6. 對 frozen-prototype readouts 的 cross-expert group，額外嘗試 CPU GU-only STE distillation：
   固定 C／affine，僅 TRAIN selected-route targets 更新共用 G/U；VALID checkpoints0/16/32/64
   必須用 integer-exact W8A8／pinned native SwiGLU 重算。Torch 的 FP32 surrogate loss 不作正式品質結果。
   更新後的 GU 視為全新基底，不能再拿 native-owner 命中抵掉它的運算或宣稱 exact self reuse；
   若還保留原 native hot-GU cache，兩份權重均要列入 host 容量。

random／prototype／native shared 等是不同候選，不能把單一 native shared basis 的失敗推成所有 nonlinear pools 無效。
所有線性 controls 也獲得相同 equalization、QDQ-aware fit、router weighting、bias 與合法 aggregate 校準能力。
不以不公平的未校準 raw x 作 baseline。
另外加入與 prototype 同寬的 raw-x 1408-coordinate readout：用校準所得 full-raw C 的
pooled column energy 選擇1408個原始座標，重新擬合；其 gather 不增加 GU 權重或 dense projection。
原 full-raw C 的 λ 已由 VALID 選擇，因此此座標選擇明列為 calibration TRAIN／VALID 搜尋，不能稱獨立 TRAIN-only。

## 校準與 targets

- 延續 DeepSeek-V2-Lite-Chat、pinned llama.cpp 927dada6c、canonical FFN W8A8。
- 校準仍用原600 TRAIN、200 calibration-VALID paragraphs；新捕捉保留最多64 native tokens，
  全文不足64者保留真實長度，禁止 padding／複製 token 來增加樣本數。
- raw1/13/26 三層先擴至全部64 routed experts，改善11/08六expert範圍過小的問題。
- 從新 capture 的 x 離線計算各 expert native target，不重跑 attention/router。
  FP64 BLAS 僅乘可精確表示的 A8/W8 整數，partial sum 有界，再使用 frozen C++ F32 rescale／native SwiGLU。
  新 target producer 必須完整逐位重現舊及新 capture 中的既知 expert targets。
- targets expert-major 各自連續存檔；prototype hidden 只保存 TRAIN-only 熱度選出的八個，避免存全部64份。
- β、ridge、feature subset、prototype assignment、模型 eligibility 都只能由 TRAIN／VALID 決定。

## 新品質資料

WikiText103 raw pinned revision `b08601e04326c79dfdd32d625aee71d232d685c3`。
從 publisher TRAIN 取出本實驗未使用過的文章；**排除全部舊 WikiText2 三個來源 split 的文章**。
每篇文章只選一段 >=96-word paragraph，以 input-only SHA256 決定順序：

- 200 篇作 quality-VALID（用於實際模型選擇、coverage 與 eligibility）。
- 1000 篇作全新 TEST，與上述文章及舊校準文章均不重複。

這是本實驗自訂 article-held-out split，不假稱 WikiText103 官方 TEST，也不是標準 concatenated WikiText PPL。
初始64-token探索每題前64 native tokens（含 BOS）評63個 adjacent next-token targets；
spec2固定六另用16-token／15 targets保持原校準條件。最終各凍結比較另做最多16 token greedy generation。
TEST 只在所有候選、超參數、eligibility 與 source/payload hashes 凍結後執行。
不可在看過新 TEST 結果後改配置再宣稱同一批是盲測。
原生 quality-VALID 可以預先執行；主程式與 fitting agents 在選模型前不開 TEST activations／推論結果。

## 品質與覆蓋率選擇

先在 calibration-VALID 評 selected-alpha² NMSE、selected-only NMSE、真 slot-order routed aggregation，
在 quality-VALID 真正執行候選：fresh KV、live hidden states／routing、原生 fallback。
使用 VALID 固定的候選 coverage／eligibility，優先選可消除更多 routed 係數、品質仍在容忍值內的配置。
初始 coverage 為跨三層合計4/8/16/32/64/128/168個 nonowner experts，
以 calibration-VALID selected alpha² SSE /（selected events × avoided coefficient MAC）排序，
此排序只產生候選，不是品質保證。先用50題 quality-VALID 篩選，再以全部200題確認；
同 eligibility 的 enhanced raw-x 必須一併比較。後續 joint／learned-GU 仍只用 TRAIN／VALID 決定。
若低 exposure 配置才通過，必須如實報告，不能只報「2% 內」而省略替換比例。
TEST 至少完整比較 native、最終 nonlinear candidate、同樣優化過的 raw-x／hot-GU 強控制各1000題。
逐題 input IDs、scored positions、unique ID 集合、EOG／生成長度、實際成功替換與 source hashes 必須核對。

primary 為 nonlinear-vs-native mean-NLL difference，品質容忍值為 `log(1.02)`；
以文章／prompt為單位做10000次 paired bootstrap、報95% interval 與 `exp(delta)-1`。
上界<=門檻才支持本配置通過；下界>門檻才判該配置失敗；跨門檻為未定。
新TEST每篇僅一prompt；secondary candidate-vs-control 區間只作描述，不能套 native 的2%門檻。
生成一致率不是 task accuracy；64-token prefix不是長上下文能力證明。

## Prototype 的特別控制

prototype 自己的 GU + 原生 D 可以構成 exact positive control，必須區分：

- prototype self 命中：相當於把該 expert 的 GU 放 host，不是跨 expert 共用的新證據。
- cross-expert reuse：用同一 prototype GU 的特徵重建其他 expert，才是本方法要新增的能力。

正式比較需納入相同 hot-GU residency、self 原生 D、其他 eligible experts 使用 enhanced raw x 的控制。
prototype GU 依真正使用的 rows／prototypes 去重執行，不允許把全八份每token運算當免費。
每個 readout 只算真正 selected 的 token rows，再依原 slot scatter；不做64個heads×全部tokens 的冗餘預測。
若 mixed feature vector 或 equalization 改了 A8 encoding，須明示其數值與 input/context 成本。

## 成本、資源與產物

起初 CPU only；使用者後續明確允許 GPU。GPU L40S 用於 GU-only STE 與 readout 分解探索；
canonical VALID、正式品質推論仍設 `CUDA_VISIBLE_DEVICES=''`、`n_gpu_layers=0`。
兩個 subagents 分別負責 fit／controls 與 numeric／adapter。
原生 FFN `3HI`、readout `Hr`，連同 transform metadata、bias/scales、resident prototype GU、
native shared GU/Down、fallback、non-FFN／KV／workspace 列帳。
保留完整 native fallback 時不得宣稱已刪除原模型容量；functional harness 還計算原生 FFN 時，其 wall time 不是壓縮部署速度。
分開列 static bank saving、固定 native routing 的反事實係數工作量、candidate live routing 的建模工作量、實測CPU kernels。
不得以它們直接宣稱實測 flash bytes、NVMeVirt 或端到端 TPOT。
同 host 預算必須給 cache-hot-GU／完整 expert／host-resident C 等合法強控制。
若壓縮後矩陣可全留 host，應列此事實，不能刻意送 flash。

第二輪收斂時以 native layer1 GU25 為唯一 resident nonlinear bank，對固定12個 nonowner experts
比較 dense 與 activation-aware rank128／256／512 的 readouts。兩階量化必須明列：
輸入 TRAIN-affine A8 → V W8 → private TRAIN-affine A8 → U W8＋TRAIN-calibrated bias。
raw-x 同樣接受低秩化、相同 rank 與 canonical VALID；private latent A8 不能當跨 expert 共用。
同預算 cache policy 由新 quality-VALID200 的全部26層 routing 固定，分別保存含／不含 BOS 版本；
以 full expert／GU pair／D 三種整矩陣單位枚舉，最優性僅限這些單位與係數工作量。
方法 payload、特有 phi／transform／latent workspace 分開列；共同 backbone／KV／workspace
不能從總部署容量中消失。若僅比較 coefficient-residency 預算，須明確稱該預算而非全 host 容量。

正式 TEST 開啟前的最終成本規則：為排除重複 BOS 對 cache ranking／共享覆蓋的影響，
**成本主比較固定為 native reference routing、全部26層、位置1..62（nonBOS）與
quality-VALID nonBOS v2 policy**。同一 v2 policy 也評 TF63／BOS／generation-causal，
包含 BOS 的 v1 policy 僅作預先指定敏感度；不能看 TEST 後挑較有利的 ranking。
品質 primary 仍維持 TF63 的 rank256 prototype25-vs-native；dense／raw／常數與壓縮前後皆是 secondary。
每一個既定 coefficient budget 的 mixed／raw-plus-cache placements 也在 receipt 中先凍結。
二階 head 7個 binary payload 連同每head4-byte rank metadata計入 runtime B；
包含絕對路徑與 hashes 的 factor.json provenance另列 artifact bytes，但所有 JSON／binary 都凍結 hash。

大型資料、模型、weights、targets、prototype hidden、編譯及執行logs放：
`/home/nfs_client/ericyu/expert_flash_experiment/research-2026-11-09-shared-features`。
repo只保留新日期程式、協定與小型結果；11/08凍結產物不覆寫。
