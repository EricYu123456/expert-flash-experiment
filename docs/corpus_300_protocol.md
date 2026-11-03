# 300-prompt 固定配置實驗

這一輪沿用原專案的真實 llama.cpp routing、INT8 AiF 時序與 NVMeVirt v3 queue 驗證。
研究對象仍是 routed expert 的 gate/up/down projection 服務時間，不是完整模型 tokens/s。
模型是 Granite、OLMoE、DeepSeek-V2-Lite、Qwen3-30B-A3B、Mixtral-8x7B，使用原有 GGUF 與 metadata。

## 題庫與切分

`scripts/build_prompt_corpus.py` 在讀取任何 routing 前產生固定種子 `20261027` 的 300 題。
六類任務各 50 題：問答、程式、數學、摘要、翻譯、長文理解。每類任務都包含三種輸入長度。
問答、程式各用 34/16 題作 train/test，其他任務各用 33/17 題，合計 200/100。
short/medium/long 共 104/100/96 題；具背景文件的題目分別使用 1/4/16 段，另外加入有優先順序的最終決議。
實際 token 數由各模型 tokenizer 計算，寫入每題 summary 與最終長度分布圖。

這是本地編寫的合成 routing 工作負載。程式題包含不同語言、資料結構與演算法，數學題涵蓋算術、代數、機率與離散計數。
摘要、翻譯與文件問答使用自包含的虛構紀錄，包含原始數值、提案及正式修訂，避免依賴外部時事。
測試情境 ID、名稱與數值獨立；部分模板與題型仍共用。結果只支持這組新輸入上的放置泛化，沒有品質基準或未知領域泛化的宣稱。
任務／長度分組為描述統計；不同長度未固定相同子題型組成，不能單獨推論輸入長度的因果影響。

`prompts.jsonl` 包含 ID、split、task、length_bucket、完整文字與 SHA-256。
題庫生成器拒絕覆蓋文字不同的既有 corpus；要換題庫必須使用新目錄。
五模型共用相同文字與切分。

## 真實推論與驗證

新增的 `scripts/llama_corpus_trace.cpp` 連結相同 pinned、未修改的 llama.cpp。
它用不同 sequence IDs 隔離每題 KV state，逐步 greedy 產生 token，再將每個生成 token 輸入一次 `llama_decode`。
每題固定執行 128 步，包含 EOS 後的 token；prefill 不擷取 routing。
callback 讀取實際 `ffn_moe_topk`，同時處理 expert 與 batch 維度的 GGML strides。
若 callback 看到非預期的 microbatch shape，會中止並拒絕該批結果。

每題也在沒有 callback 的 context 中執行全部 128 步，要求完整 token IDs 與輸出 bytes 一致。
最初四個模型使用 8 sequences、8 threads，各自重新 prefill。
Mixtral 使用 32 sequences、16 threads，untraced context 從相同 prefill 的逐 sequence KV snapshot 啟動，然後獨立執行全部 decode。
剩餘 Mixtral batches 由兩個推論 process 並行收集；每題原始 sequence batch、threads 與 binary 不變，已通過的批次直接續用。
Qwen3 也在原有 8-sequence batches 完成 checkpoint 後，將剩餘批次交給兩個 process，以相同初始 binary 續跑。
整個 prefill snapshot 不含任何已生成的 decode token。小型 smoke run 先核對重新 prefill 的對照，並檢查 snapshot 對照的輸出一致性。

不能使用 pinned 版本的 whole-cache snapshot 來還原非 unified 多 stream KV：其 reader 每讀一個 stream 都清除全 cache。
本地 harness 使用 `llama_state_seq_get_data` / `llama_state_seq_set_data` 逐序列儲存與還原，不修改 upstream。

`scripts/run_corpus_traces.py` 核對每筆事件的 token index、position、token ID、routed layer、expert 數量、expert 唯一性及範圍。
每完成一個原始 batch 才寫入 checkpoint；續跑保留同一組 sequence batch，避免重組改變浮點運算次序。
batch size、threads、binary hash、題庫 hash、metadata hash 均是 checkpoint identity。

保存的 `prefix32/ID.routing.jsonl` 與 `.prefix32.txt` 直接取自相同的 128 步執行。
這一輪是有效的量化模型路由；批次推論的浮點運算次序可能不同於舊的單序列推論，不宣稱跨協定 routing 逐位元相同。

## 學習與固定配置測試

`scripts/corpus_analysis.py` 的 fit 階段只讀取 split=train 的 200 題 routing。
對每個 layer 與 g ∈ {1,2,4,8,16}，以固定種子做 512 次容量守恆 expert-slot swaps。
目標先減少頻率加權的最大 group 同時啟用數，再減少 group occupancy 平方和。
它壓縮相同 expert sets 並保留實際出現次數；不把不同 prompt 的相同 route 當成一次。

用完整 128 步 train 平均 routed FFN 服務時間，聯合選擇 g 與 gate/up separate 或 vertical stacked layout。
成本沿用 flash、controller fanout、per-chip control、host phase、向量傳輸與 channel contention。
原始 expert-ID 排列與學習排列各自只用 train 選定一套固定配置。
所有候選分數、各 g 的固定排列與 winner 寫入 `frozen_plan.json`，另記 SHA-256 與 freeze time。

test 階段在 freeze 後才讀取 100 題測試 routing，計算預先定義的五個比較：

- 16 chips、gate/up 分開。
- 16 chips、gate/up 串接。
- train 選定的 g/layout、原始 expert-ID 排列。
- train 選定的 g/layout、學習排列。
- 學習配置相同 g/layout 下的原始 expert-ID 排列，用於隔離排列本身的增益。

前 32 步與完整 128 步都用相同 frozen winner，沒有重新用測試題選 g、layout 或 swaps。
報告每題平均、token P50/P95、各任務、各輸入長度及改善比例，保留零改善或負改善。
95% CI 使用配對 prompt bootstrap、2,000 次，整題重抽樣；不把同題的 tokens/layers 當獨立樣本。

向量化模型依每個固定 group 的選中 expert 數量還原各 chip 的服務與 channel 流量，保留整數 rounding。
每模型每個 g/layout/policy/layer 都取六任務、三長度與第 1/32/128 步，與原 scalar model 核對五項成本。
這些 held-out invariant checks 不能影響 frozen plan，亦不匯出 held-out 候選組態排名。

## NVMeVirt 與輸出

`scripts/export_corpus_replays.py` 對全部 100 題測試，匯出 16-chip stacked 基線與 frozen learned 配置。
每配置拆成 prefix32 與 tail96，合計覆蓋全部 128 步而不重複：每模型 4 個 workloads，五模型共 20 個。
等價 group occupancy 的 descriptors 可快取，但每個真實 input/layer 仍按原順序寫入、逐一 ioctl，沒有抽樣或壓縮重播次數。
逐 byte 對照 scalar exporter 的測試檢查此快取。匯出 counters 另與向量化分析逐項核對。

同時產生容量守恆的 virtual INT8 NAND 地址計畫，檢查完整 row coverage、非重疊與 cold/page/read wave 數。
這些地址描述不寫入模型權重；現有小型 NVMe namespace 仍只重播 metadata timing。

`scripts/run_corpus_replays.sh` 在既有 guest serial 執行，核對所有 chip 與 v3 control/vector counters，保留實際 ioctl wall time。
可重新核對並略過已通過的 workloads。NVMeVirt counters 為共享狀態，replay 使用 lock。
若學習配置與基線的完整 descriptor stream 逐 byte 相同，以同一次真實重播核對兩者 expected counters，另存 `alias.json`。
最終報告分開列出配置／步段核對數、不同 workloads 數與實際 ioctl 數，不重複計算等價重播。

`scripts/finish_corpus_experiment.py` 等待各模型 300 題驗證完成，再排入 fit/evaluation/export 與 serial replay。
全部完成才以 `scripts/plot_corpus_results.py` 產生 PNG/PDF/SVG、summary、strata、manifest 及中文報告。
圖表包括新輸入延遲、32/128 改善與 CI、任務/長度分組、train g sweep、實際輸入長度及舊三題 32 步比較。
舊三題與新一輪的工作負載、搜尋預算與訓練協定不同，舊結果比較圖不能當作 decode 長度的因果實驗。

原始資料目錄：`/home/nfs_client/ericyu/expert_flash_experiment/runs/corpus-300-v1`。
小型最終輸出目錄：`results/corpus-300-v1`。舊結果保留在原位置。

## 重現

```bash
cd /home/ericyu/expert_flash_experiment
python3 -m unittest discover -s tests -v
bash scripts/build_llama_trace.sh
python3 scripts/build_prompt_corpus.py \
  --output-dir /home/nfs_client/ericyu/expert_flash_experiment/runs/corpus-300-v1/corpus

# 四個模型與 Mixtral 可分別在背景執行；CPU 使用量可用 workers/threads 調整。
python3 scripts/run_corpus_traces.py --models granite olmoe deepseek qwen3 --workers 4
python3 scripts/run_corpus_traces.py --models mixtral --workers 1 \
  --threads 16 --batch-size 32 --reuse-prefill --inference-workers 2

bash scripts/vm_start.sh
bash scripts/vm_ssh.sh 'mount -a; bash /mnt/workspace/scripts/vm_guest_prepare.sh'
python3 scripts/finish_corpus_experiment.py --stop-vm
```

既有實驗第一次收集的前四模型使用 `artifacts/llama-corpus-trace-initial` 保存的 binary。
若續跑該批，需加 `--binary` 指向該版本；binary hash 不符時 runner 會拒絕混入新版本。
要用最新 source 從頭重現，使用新的 run directory；同時將 `--run-dir` 傳給收集與分析程式。

可獨立重做後處理：

```bash
python3 scripts/corpus_analysis.py --models granite
python3 scripts/export_corpus_replays.py --models granite
bash scripts/vm_ssh.sh \
  'bash /mnt/workspace/scripts/run_corpus_replays.sh /mnt/experiment/runs/corpus-300-v1/granite/replays'
python3 scripts/plot_corpus_results.py
```

控制成本與向量精度沿用原有未校準假設；shared experts、dense FFN、attention、router、activation/combine、cache、GC 與跨層 overlap 仍不計入。
