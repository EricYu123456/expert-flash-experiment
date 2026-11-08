# 共享非線性基底：第二輪 CPU 優化協定

日期：台北 2026-11-09。延續 `docs/1108_spec.txt` 與 11/08 真模型結果。
本輪的目標是改善品質與可消除的係數工作量，並超越同樣校準過的 raw-input 線性控制。
不將只保留六個低 exposure experts 的小 PPL 變化解讀成通用方案成功。

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

random／prototype／native shared 等是不同候選，不能把單一 native shared basis 的失敗推成所有 nonlinear pools 無效。
所有線性 controls 也獲得相同 equalization、QDQ-aware fit、router weighting、bias 與合法 aggregate 校準能力。
不以不公平的未校準 raw x 作 baseline。

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
每題前64 native tokens（含 BOS）評63個 adjacent next-token targets；最終另做最多16 token greedy generation。
TEST 只在所有候選、超參數、eligibility 與 source/payload hashes 凍結後執行。
不可在看過新 TEST 結果後改配置再宣稱同一批是盲測。
原生 quality-VALID 可以預先執行；主程式與 fitting agents 在選模型前不開 TEST activations／推論結果。

## 品質與覆蓋率選擇

先在 calibration-VALID 評 selected-alpha² NMSE、selected-only NMSE、真 slot-order routed aggregation，
在 quality-VALID 真正執行候選：fresh KV、live hidden states／routing、原生 fallback。
使用 VALID 固定的候選 coverage／eligibility，優先選可消除更多 routed 係數、品質仍在容忍值內的配置。
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

CPU only，`CUDA_VISIBLE_DEVICES=''`、`n_gpu_layers=0`。兩個 subagents 分別負責 fit／controls 與 numeric／adapter。
原生 FFN `3HI`、readout `Hr`，連同 transform metadata、bias/scales、resident prototype GU、
native shared GU/Down、fallback、non-FFN／KV／workspace 列帳。
保留完整 native fallback 時不得宣稱已刪除原模型容量；functional harness 還計算原生 FFN 時，其 wall time 不是壓縮部署速度。
分開列 static bank saving、固定 native routing 的反事實係數工作量、candidate live routing 的建模工作量、實測CPU kernels。
不得以它們直接宣稱實測 flash bytes、NVMeVirt 或端到端 TPOT。
同 host 預算必須給 cache-hot-GU／完整 expert／host-resident C 等合法強控制。
若壓縮後矩陣可全留 host，應列此事實，不能刻意送 flash。

大型資料、模型、weights、targets、prototype hidden、編譯及執行logs放：
`/home/nfs_client/ericyu/expert_flash_experiment/research-2026-11-09-shared-features`。
repo只保留新日期程式、協定與小型結果；11/08凍結產物不覆寫。
