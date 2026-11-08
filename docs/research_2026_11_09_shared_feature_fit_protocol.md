# 11/09 nonlinear basis：TRAIN/VALID 擬合協定

本文件描述子實驗的擬合與數值契約。新盲測與真 LLM 品質由主實驗管理；本工具沒有 TEST 參數。
11/08 TEST 已被看過，不當作新的 held-out 證據。11/08 frozen 程式與產物不修改。

## 第一階段：舊 16-token 快篩

比較相同強化的 raw x、native shared z、TRAIN 選 1408 個 shared features，以及 x+shared512/768。
所有組先做 TRAIN-only affine equalization，部署保留此變換，之後才 whole-vector A8；直接以 QDQ features fit。
每 expert 採 `w=(1-beta)+beta*selected_alpha²/mean_TRAIN(selected_alpha²)`；beta=0/.99，ridge=.03/.3/3。
row-W8 後重新用 TRAIN weighted residual 校 bias，VALID 以 selected-alpha² NMSE 選參。

六 expert 結果：純 shared 仍全輸同樣強化的 raw x。hybrid 只有三點微小改善（相對 .32–.74%），
另外三點較差，卻多 25–37.5% readout coefficients，未構成有意義的成本優勢。
結果在 `results/research-2026-11-09-shared-features/VALID_screen_summary.json` / `.csv`。
這只否定本次 frozen native-shared 配置，不能否定其他 resident nonlinear pools。

## 第二階段：完整 64-token TRAIN/VALID 與 hot routed prototype pools

使用原 600/200 TRAIN/VALID prompts 的較長擷取，實際 rows 由 captures/targets manifest 驗證；
不可假設每段必然滿 64 tokens。每層對全部 64 experts 產生完整 canonical W8A8 targets。
每 expert 一個連續 F32[N,2048] 檔，並保存每層 TRAIN routing 頻率最高 8 experts 的 pre-Down hidden phi[N,1408]。

1. 依固定 seed 將 TRAIN prompts 分 80% rank-fit / 20% rank-score，再各取最多 4096/1024 rows。
   同 prompt 不得跨兩邊。這是 TRAIN 內部候選排序，不使用外部 VALID 或 TEST labels。
2. 對完整 targets 做固定 128 維 Rademacher output sketch，只供離線排序，不進部署。
   每 prototype 共用一次 unweighted Gram、固定 ridge=3，批次擬合 64×128 sketch outputs。
   在 rank-score rows 上按原 router alpha² 排序，每 expert 保留最佳兩個 cross prototypes。
3. owner==target expert 的自身 prototype 排除 cross ranking。其原始 native D 搭配原始 phi 的 A8
   另外做逐 bit 正控制，不能算 cross-expert 共用收益。
4. 完整 2048 輸出只擬合 enhanced raw x 與這兩個 prototypes，ridge=1/3/10、beta=0/.99。
   VALID 的 selected-alpha² NMSE 選 ridge/beta/assignment，不開任何盲測 target。
5. 正式 runtime 的 resident prototype owners 一律保留 native，使用 `excluding_owners` registries。
   三層最多替換 168 個 nonowner experts，對照也使用相同 eligibility 與同 hot-GU cache。

## 精確部署契約

每個 feature 順序固定：gather raw x / raw native phi → `F32(F32(raw-center)*inv_scale)` → whole-vector A8。
center/inv_scale 從全部 TRAIN inputs 計算、存 F32，不折回原始 raw-coordinate C。
C 直接 fit QDQ(A8(transformed feature))；row-W8 與 dynamic A8 採 F32 scale、F32 quotient、ties away from zero。
dot 是精確 integer sum，接 `F32(dot)*F32(weight_scale*input_scale)`，再獨立 F32 bias。
native shared Down 及 native prototype owner 的原生路徑，仍使用原始 hidden 與各自原生 A8。

正式 bias 使用 TRAIN weighted QDQ-feature / dequantized-W8 moments 校準，省去每候選完整 TRAIN predictor pass。
canonical rounding 不完全線性；每個選中模型保留 TRAIN probes 的 roundoff 差異，部署與 VALID 評估仍用真正 canonical predictor。
beta weighted Gram/RHS 由 all-row 與 selected-row moments 組合，專用測試與直接 weighted ridge 一致。
候選只預測 VALID 真選中 rows，因其他 rows 的 alpha²=0；贏家再補 all-row 與整層原生 slot-order aggregation 診斷。

Registry 為六個 TSV 欄位：

`layer, expert, basis_id, width, transform_prefix, readout_prefix`。

transform_prefix 對應 `.sources.u8`、`.indices.i32`、`.center.f32`、`.inv_scale.f32`、`.prototypes.i32`；
source 0=x、1=native shared z、2=routed prototype phi，非 prototype 欄位的 owner=-1。
source2 只在外層 remap 到同一 frozen C++ gather kernel 的 FP32 pool，不引入新的量化邊界。
readout_prefix 對應 `.i8[2048,width]`、`.scales.f32[2048]`、`.bias.f32[2048]`。

## True aggregate 與 joint correction

所有 VALID aggregate 按原 slot 順序重組 `F32(alpha*expert_output)` 的 F32 additions，未替換者保持 native。
同時報 routed-only 與 shared+routed NMSE、slot/alpha coverage、去重後所需 prototype 數量，
以及超過原生 hot-GU 命中的額外 prototype 計算；不把低 exposure 當精度成功。

可再對同一 feature group 擬合 TRAIN 共用 correction：

`delta_y ~= g*(R*QDQ(v)+b_R), g=sum_eligible alpha`。

直接使用 g·v 和 intercept g 的 weighted moments，避免除以零或極小 g。
固定 ridge=1、VALID 只選 eta=0/.25/.5/1。eta=0 保留原部署 payload bytes；
eta>0 從實際 dequantized C 加 eta·R、bias 加 eta·b_R，重新 row-W8。
多 prototype 分 group 順序校準；實際 VALID aggregate 必須重跑 canonical predictors。
R 只存作離線診斷，部署沒有新增 R head、hidden width、private GU 或 readout 係數。

## 成本與執行

prototype width1408 的 readout 係數較 raw x2048 少 31.25%；原 expert 的 3HI 對 Hr 減少 66.7%。
但 resident hot8 GU 的權重、scales、GU MAC、SiLU，以及 transform/gather 都另計。
每 token 只計實際需要的 prototype/input pairs，與原生 owner GU 命中去重。
必須比較「同 hot-GU cache + native owners + enhanced raw x」，以及 readouts 全放 host 的可行性。
係數減少與 offline CPU 時間均不當系統 speedup。

所有大產物根目錄為 `/home/nfs_client/ericyu/expert_flash_experiment/research-2026-11-09-shared-features`。
正式 sources 在 `formal/source-v1`，SHA、argv、CPU affinity、logs、exit 與逐層 completion 在 `formal/execution`。
watcher 每層獨立等 TRAIN/VALID targets status=complete 後跑 rank→fit→C++ numerical audit；不等三層一起 ready。
初始 CPU 24–31、56–63，全部 VALID target workers 完成後才啟用 32–39。
未完成的完整 64-expert targets 或 fitting 不宣稱完成。

專用六個 tests 已通過。另有真資料 partial numerical smoke：TRAIN16 prompts/979 rows、VALID8/502、
三 experts 與一 prototype，rank/fit/source2 C++ bit audit/joint 皆 exit0；此 smoke 不作 full64 或 LLM 品質證據。

## 同 width 的 raw-x control

另以 `research_20261109_shared_feature_fit_raw_subset.py` 做 readout width1408 的純線性對照。
從已選 fullrawx W8 C 反量化後，跨所有 expert/output channels 累加各 input column 的平方 norm，
取前1408個座標（同分按原始座標），部署 gather 順序固定為原座標順序。
因 fullrawx 的 C 已用 VALID 選 lambda/beta，這是明確的 calibration-selected feature set，
不宣稱 TRAIN-only 選 features。它沒有看任何 TEST。

保留原 TRAIN mean/std 對應座標，但 gather 後重新執行 whole-vector A8；不能切片原 fullwidth QDQ。
重新用同 lambda=1/3/10、beta=0/.99 擬合全部64個 readouts，並報原生 slot-order aggregate、
排除 resident owners 後同 eligibility 的結果及 C++ bit audit。runtime 不需要額外 GU 或非線性。
此程式與相依檔獨立 freeze 於 `raw-subset/source-v1`，不修改正式 prototype 的 frozen source。
partial 真資料三 expert/1408 維 smoke 及 gather-before-A8、W8 column-scale 排序測試已通過。

## 1108_2 規格的低曝光 controls

`research_20261109_shared_feature_fit_constant.py` 只讀原600個 TRAIN prompts、每題16 tokens 的 captures，
對原六 expert 匯出 drop zero、自然 selected rows 的常數均值、selected-alpha² 加權均值。
後者是相同 selected-alpha² least-squares 目標下的最佳常數控制，避免因均值定義而低估常數能力。
均值只用 TRAIN native expert targets，FP64 累加後一次轉 F32；VALID/TEST targets 都不開。

三種控制皆匯出非空 width1 的標準六欄 registry：source0、index0、center0、gain1；
`C.i8[2048,1]` 全零、row-W8 scales 全一、F32 bias 為零或對應均值。
因此可直接通過既有 live adapter，沒有 drop 特例。新 runtime 仍有一維 gather/A8 和零 C，不能把它算成免控制開銷。
`spec2-controls/source-v1` 保存獨立 frozen sources，`spec2-controls/train16` 保存 payloads、TRAIN 支持度/energy/SSE 與 hashes。
18 個 control readouts 的 gather/affine/A8/全部2048 output channels 已與 frozen C++ 逐 bit 相符。
此處未執行 control 的 LLM quality；是否因曝光稀疏而也能通過品質界線，交由 root 的 quality-VALID 判斷。

## 固定 prototype25 的 calibration cohort revision

初次全池 fit 完成後，另按 calibration-VALID 非 BOS 結果選定 L1 prototype25 的三個 seed experts：4／51／60。
raw 與 prototype 同時補 beta=.5/.9，完整四 beta=0/.5/.9/.99、三 lambda=1/3/10 後仍保留大於10%的局部改善。
再固定同一個 GU owner25，按既定 calibration 排序加入58／5／17／62／22／1／28／38／50，形成3／6／9／12個 cohorts。
這些 eligibility、assignment 與排序明確使用 calibration VALID，不宣稱 TRAIN-only selection。
新增九 heads 只 fit rawfull 與 prototype25，沒有新增其他 GU owner；同寬 rawsubset1408、TRAIN means/drop 也補齊同12 heads。

`proto25-cohort12/source-v1` 保存沿用的 frozen fitting sources，`source-merge` 保存 standalone merger。
`registry_handoff.json` 記錄 old/new readouts 來源、hashes、transform 逐 byte 等同證明與統一 cache prefix。
所有種子 payloads、既有 selection／registries 保持 immutable。
`registries/registry.{family}.k{3,6,9,12}.tsv` 對六個 families 使用相同 eligibility；
72個最終模型均通過 frozen C++ transform／A8／W8／bias bit audit。
原 target pool metadata 有八個可用 prototypes，實際部署 registry 的 owner union 只有25；host 成本必須由實際 registry 計算。
quality-VALID 50／200題曲線與未來盲測由 root 執行，本 fitting 階段沒有接觸新 TEST。
