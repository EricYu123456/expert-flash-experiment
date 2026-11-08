# Native shared nonlinear features：CPU 驗證協定

依據：`docs/1108_spec.txt`。執行日期：台北 2026-11-08。

## 問題與界線

本次檢查既有 checkpoint 的 shared expert 在 Down 前的特徵
`z_s = SiLU(G_s x) * U_s x`，是否足以用 expert-specific、重新擬合的
`C_e z_s + b_e` 取代 routed expert 的整個 FFN。shared branch 與 router 程式保留。
這是近似模型；前層替換之後的後續 hidden state、router、KV 與生成必須重新執行。
固定 native trace 只能供局部機制分析。

本次先做固定小範圍機制測試，不宣稱全部 64 experts／26 層都可替換。
既有專案提供 CPU W8A8 FFN 參考、GGUF 模型與 llama.cpp；本次新增日期檔案，保留歷史結果。
NAND、AiF timing 與 CPU harness wall time 分別呈現；未使用 GPU 或實體 AiF。

## 事前固定的實驗條件

- 模型：DeepSeek-V2-Lite-Chat，既有 Q8_0 GGUF；原始來源唯讀。
- llama.cpp：專案既有 pinned 927dada6c 來源與 CPU libraries。
- raw layer 0 為 dense；測試 raw MoE layers 1、13、26。
- experts：layer 1 使用 14、40；layer 13 使用 10、56；layer 26 使用 0、1。
  前兩層沿用歷史 pilot IDs，最後一層使用 0、1。未依本輪 TEST 選擇。
- 每個固定 expert 在每個捕捉 token 都計算 target，不只用 router 選中時的稀疏樣本。
- shared feature width 2816，routed intermediate width 1408，hidden width 2048。
- 不訓練 shared GU，不改 router，不新增 confidence predictor。
- 主要比較完整 native shared basis；private features 與 feature selection 留待另輪。

## 資料

使用 Salesforce/wikitext 的 pinned revision
`b08601e04326c79dfdd32d625aee71d232d685c3`、`wikitext-2-raw-v1`。
依官方 train／validation／test 分別固定抽 600／200／1000 個段落。
選取由 input-only 規則與 seed 決定，排除標題、過短段落與重複輸入；保存原始來源 SHA256、
paragraph row、article ID、全文與選取排名。輸入前 15 個非特殊 tokenizer IDs 也做跨集合去重。
這些是英文自然文字 continuation prompts，不是多任務 instruction accuracy benchmark。
同 article 的 prompts 有相關性，品質區間應按 article cluster 重抽樣。
WikiText TEST 曾被專案歷史實驗使用，這次 readout 不以其訓練或選參。

首輪保留前 16 個 native tokens（含 BOS）；每個 expert 可獲 9600 個 TRAIN rows，
3200 個 VALID rows 與 16000 個 TEST rows。實際 native token IDs 與完整筆數由 runner 驗證。
16-token 上下文只用作機制診斷，不能當成長上下文品質或標準 WikiText PPL。
真生成測試使用 fresh context/KV，greedy autoregressive decode，保存 token IDs、替換次數及 routing。

## 對照、選參與數值契約

局部 readout 比較：raw x、shared output、x+shared output、shared pre-Down feature，
以及固定 seed、相同 2816 寬度的 random SwiGLU input-only nonlinear control。
x+shared output 的 4096 維 readout 必須標示額外容量，不能當同容量比較。
random basis 需計入其 resident GU，不當免費特徵。
每 expert 的 ridge 與所有 preprocessing 僅用 TRAIN 擬合；VALID 選正則化，
在開 TEST targets 前保存 selection。TEST 不用來挑 expert、ridge、basis、threshold 或模型。
局部 NMSE/cosine 與 native routed aggregation 誤差是診斷，不是 LLM 品質通過的證明。

主要數值參考沿用完整 FFN canonical row W8 + full-vector dynamic A8：
F32 max-abs scale、ties away from zero、[-127,127]、exact integer dot、F32 rescale。
每個新 readout 必須使用自己的 activation scale；不可假設各 expert 可共用原生 A8 編碼。
GGUF 原生 FFN 與 canonical W8A8 是不同參考，結果不得混稱。
FP readout 是診斷組，需另標明累加與 rounding；部署比較以明示 W8A8 契約為準。

## 真模型品質判定

事前品質容忍值沿用專案 2% relative subset-PPL，即 paired mean-NLL 差異門檻 `log(1.02)`。
報告 native canonical W8A8 與替換版本的 paired NLL、exp(delta)-1、article-cluster bootstrap 95% CI，
同時保留 greedy continuation 的 token 一致率、first divergence 與實際替換 exposure。
只有 CI 上界不超過門檻時，才支持本測試範圍內的品質通過；CI 下界超過門檻是該配置失敗，
跨越門檻仍未定。生成 token 一致率是診斷，不是 task accuracy。
局部低 MSE 或 router 程式相同均不能替代這一步。
固定兩個 expert 的低 exposure 不足以支持全 expert 部署，即使其整體 NLL 看似良好。

## 成本

原 expert 為 `3 H I` 個係數，新 readout 為 `H r` 加 bias/scales。
`r=2816, I=1408` 的 raw routed coefficient saving 為 33.3%，不是系統 speedup。
若保留原始 fallback tensors，容量須同時列出；固定 eligibility 才能考慮刪除對應 GU/Down。
必須計 native shared GU/Down residency、readout、非 FFN、KV、workspace 與 metadata。
同 host 容量控制包含壓縮 readout 全放 host 的可行性；不能刻意將放得下的矩陣送 flash。
硬體模型需給 baseline 同 block free-skip、全部 chips row striping，並列 shared feature dependency、
host bandwidth、傳輸、native fallback 與 shared GU 不免費等條件。
品質未通過之前，不將 raw bytes 或本機 CPU harness 時間解讀成 AiF end-to-end speedup。

## 儲存與重現

程式及精簡報告：`/home/ericyu/expert_flash_experiment`。
模型、下載資料、activation arrays、readouts、binary、logs：
`/home/nfs_client/ericyu/expert_flash_experiment/research-2026-11-08-shared-features`。
corpus 建構器：`scripts/research_20261108_shared_feature_corpus.py`。
capture 與 fit 工具由兩個使用者指定的 `gpt-6.1-sol / xhigh` subagents 分工。
各執行器設定 `CUDA_VISIBLE_DEVICES=''`、`n_gpu_layers=0`，記錄指令、threads、hashes、completion。
未完成 1000 個 TEST prompts 時不得以預定題數冒稱完成。
