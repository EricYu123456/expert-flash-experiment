# MoE FFN 的 AiF chip 分組與放置實驗

**2026-11-09 第二輪優化：**依 `docs/1108_2_spec.txt` 完成強 raw-x／OOF residual／單一 GU25／GPU 短訓及低秩 readout 實驗，正式 CPU TEST 為五組各 1000 題（64 input tokens＋最多16 token生成）。rank256 共享方案將係數常駐量從38.72降至15.88 MiB，PPL +1.420%（95% CI +1.035%～+1.820%），通過2%門檻；但同預算 raw-x＋cache 品質較好、避免的係數工作也較多（0.7115%對0.6197%）。實際只替換0.5666%的 routed events，常數控制區間亦低於2%，尚未支持非線性共享的額外部署價值。這些成本是模型，沒有實測 flash／TPOT；詳見 [第二輪報告](results/research-2026-11-09-shared-features/report.zh.md)與 [方法詳解](results/research-2026-11-09-shared-features/method_details.zh.md)。

**2026-11-08 新實驗：**依 `docs/1108_spec.txt` 完成 DeepSeek native shared pre-Down features 的 CPU 驗證，使用 600 TRAIN／200 VALID／1000 TEST prompts，並對同一 1000 題實際執行原生、shared-feature 與 raw-x 三組生成。三層六 experts 的 shared 替換通過短上下文 2% subset-PPL 門檻（+0.541%，article-cluster 95% CI −0.072%～+1.194%），但局部六點均弱於較小的 raw-x readout，尚未支持 shared nonlinear basis 的部署優勢。完整結果、範圍與重現紀錄見 [本輪報告](results/research-2026-11-08-shared-features/report.zh.md)及 [協定](docs/research_2026_11_08_shared_features_protocol.md)。大型產物均留在 NFS。

目前研究主線是 **不依賴 CR-read、適用有／無 shared expert 的 MoE in-flash 方法**。進度與判定見 [研究紀錄](docs/research_2026_11_03_campaign.md)、[專案理解](docs/research_2026_11_03_project_understanding.md)及 [即時狀態](results/research-2026-11-03-no-cr/state.json)。截至 2026-11-03，第二輪最早層全64-expert固定配置已完成品質驗證：OLMoE相對subset-PPL退化95%區間為10.62%–21.25%，超過2%預定門檻，判此配置有界失敗；DeepSeek保留未定。已依使用者於 2026-11-03 的要求停止研究並保存最後快照。累計兩次實驗失敗、剩八次機會；第三輪候選尚未進入實驗，十輪尚未完成。尚未證實完整模型延遲收益或投稿價值。下方保留較早的配置與 CR-read 實驗背景。

GitHub 保存程式、協定、審查紀錄及適合版控的結果。大型原始張量、模型、資料快取和重複產物保留在研究機器；範圍與檔案清單見 [版本保存說明](docs/github_snapshot.md)。

本實驗找的是 routed FFN 延遲較低的固定配置，包含 chip 數量、短讀取造成的 cr-read 損失、命令與向量傳輸成本，以及 expert 同時啟用時的 chip 衝突。最新的 300 題結果與圖表見 [results/corpus-300-v1/report.md](results/corpus-300-v1/report.md)；原三題先導結果見 [results/report.md](results/report.md)。

300-prompt 擴充實驗使用每模型相同的 200 題配置學習、100 題保留測試，涵蓋六任務與三種輸入長度，每題 128 次 decode 並保留前 32 步。方法、資料切分、固定配置驗證與重現指令見 [docs/corpus_300_protocol.md](docs/corpus_300_protocol.md)；新一輪輸出獨立放在 `results/corpus-300-v1/`，原有三題先導結果保留。

使用真實 llama.cpp 推論擷取 routing，再將相同 expert 選擇送入 QEMU/KVM guest 裡的 NVMeVirt AiF 時序擴充。NVMeVirt 是實際載入的 Linux module；AiF GEMV、控制與向量傳輸是明確建模的延遲，沒有實體 flash PE 或 MAC lane 量測。模型數值運算保留在擷取 routing 的 llama.cpp 中，重播不以 dummy activations 決定後續 routing。

## 儲存位置

- 論文、實驗程式、patch、少量圖表：`/home/ericyu/expert_flash_experiment`。
- 模型、來源樹、編譯產物、VM 映像與原始結果：`/home/nfs_client/ericyu/expert_flash_experiment`。
- 既有模型來源：`/home/nfs_client/ymtuan2/models`，唯讀使用。

實驗沒有修改主機 kernel、GRUB 或使用 host sudo。既有 sibling `llama.cpp-aif`、`nvmevirt` 專案沒有修改。

## 原三題先導方法

1. 以有效 MoE 推論擷取每個 decode input、每層選中的 expert IDs；與不開 tracing 的 greedy 輸出逐位元比較。
2. 以論文的 INT8 權重、16 chips、每 chip 4 planes、16 KiB page、28 us 普通 LSB read、9.7 us cr-read、6.4 GB/s/chip 建立模型。GGUF 的 Q4/Q8 格式只用來取得真實 routing。
3. 比較每 expert 固定使用 1、2、4、8、16 chips。允許不同 chip groups 的獨立 experts 並行，同一 chip 逐 stream 處理。
4. 將共用輸入的 gate/up 垂直堆疊成連續的 `2I × H` 矩陣，增加連續讀取長度，之後才執行相依的 down projection。
5. 在相同容量限制下，用共同啟用 trace 學習 expert 到 group 的固定排列，減少同時選中的 experts 落到相同 group。每一折只用兩組 prompts 學習與選擇 chip 數，第三組保留驗證。
6. 掃描 per-chip control 成本 0–5 us、per-host-command 成本 0–10 us，並計入 PCIe 8 GB/s、channel 2 GB/s 下的向量流量。展示用參數 0.25 us/chip command、2 us/host phase 是假設，並非實體 AiF 校準值。
7. 將需求送入真實 NVMeVirt queue；逐項核對 Python 與 kernel 的 pages、cold reads、bytes、逐 chip service、控制與向量傳輸 counters。

每個 block run 的時間是 `28 + (P-1) × 9.7 us`；28 us 已包含第一次 precharge 與最後 discharge。四 planes 並行時，以最慢 plane 與 6.4 GB/s 管線上限的最大值計算，避免重複收取同一次讀取成本。

## 重現原三題分析

主機需有 Python、NumPy、Matplotlib；模型轉換需要本機已有的 Torch、Transformers、safetensors。使用來源樹的 `gguf-py`。所有分析使用 NFS 上既有 artifacts：

```bash
cd /home/ericyu/expert_flash_experiment
python3 -m unittest discover -s tests -v

python3 scripts/run_analysis.py --name granite \
  --metadata /home/nfs_client/ericyu/expert_flash_experiment/models/granite-metadata.json \
  --traces-dir /home/nfs_client/ericyu/expert_flash_experiment/runs/granite-q4

python3 scripts/run_analysis.py --name olmoe \
  --metadata /home/nfs_client/ericyu/expert_flash_experiment/models/olmoe-metadata.json \
  --traces-dir /home/nfs_client/ericyu/expert_flash_experiment/runs/olmoe-q4

python3 scripts/run_analysis.py --name deepseek \
  --metadata /home/nfs_client/ericyu/expert_flash_experiment/models/deepseek-metadata.json \
  --traces-dir /home/nfs_client/ericyu/expert_flash_experiment/runs/deepseek-q8

python3 scripts/run_analysis.py --name qwen3 \
  --metadata /home/nfs_client/ericyu/expert_flash_experiment/models/qwen3-metadata.json \
  --traces-dir /home/nfs_client/ericyu/expert_flash_experiment/runs/qwen3-q8

python3 scripts/run_analysis.py --name mixtral \
  --metadata /home/nfs_client/ericyu/expert_flash_experiment/models/mixtral-metadata.json \
  --traces-dir /home/nfs_client/ericyu/expert_flash_experiment/runs/mixtral-q8

python3 scripts/plot_results.py
python3 scripts/plot_optimized.py
```

`run_analysis.py` 會重建純 flash、固定 expert-ID 排列敏感度、共同啟用放置、含 overhead 模型與保留提示驗證結果。檔案寫入各模型的 `runs/NAME-aif/`，每個階段有完整 log。

主要結果使用 `overhead_fanout/` 與 `learned_overhead_fanout/`：每個邏輯輸入只經 PCIe 傳送一次，再由 controller 向 chips 複製。較保守的 v2 unicast 參考結果保留在原有 `overhead/` 與 `learned_overhead/`。`learned_overhead_fanout/physical_plans/` 提供每個 expert 的 chip、完整 row range、block、WL offset 與頁數；短 streams 共用 block，不必各占完整 block。

## llama.cpp 與既有模型

```bash
bash scripts/build_llama_trace.sh
bash scripts/convert_existing_moe.sh deepseek
bash scripts/convert_existing_moe.sh qwen3
bash scripts/convert_existing_moe.sh mixtral
```

轉換由現有 safetensors 直接產生 Q8_0 GGUF，避免大型 F16 中間檔。2026-10-27 補測 Qwen3-30B-A3B-Instruct-2507-FP8 與 Mixtral-8x7B；前者先依 block scales 解量化，再轉為 Q8_0。模型與 tokenizer 來源均維持唯讀，AiF 計時仍統一使用 INT8。

Mixtral 的來源設定 `add_prefix_space: null` 被 pinned converter 寫成 false，與 tokenizer.json 的 Metaspace 規則不符。`fix_mixtral_tokenizer.py` 修正 GGUF 中的單一 metadata byte，保留權重與 tensor layout，再重新計算完整 hash。修正後三組 prompt token IDs 與 Hugging Face 完全一致。DeepSeek 的 BOS 相容處理、來源清單、數值抽查、hashes 與 routing 驗證見 [docs/llama_trace.md](docs/llama_trace.md)。Qwen3.6、Gemma 4 尚未列為已測模型。

補測結果與圖表為 `results/additional_models.json`、`results/additional_models.png`；五模型摘要由 `plot_optimized.py` 匯出。原三模型結果封存在 `results/archive/2026-10-19/`。`finalize_additional_models.py` 會先確認數值、tokenizer、routing、地址計畫與全部 v3 重播證據，再更新彙整；不會自行產生實驗結果。

## 無 host sudo 的 NVMeVirt

```bash
bash scripts/prepare_nvmevirt_source.sh
bash scripts/vm_setup.sh       # 首次建立映像；已有 VM 時略過
bash scripts/vm_start.sh
bash scripts/vm_ssh.sh 'cloud-init status --wait'
# 首次 cloud-init 完成後 reboot guest，讓記憶體保留生效；再執行以下指令。
bash scripts/vm_ssh.sh 'mount -a; bash /mnt/workspace/scripts/vm_guest_prepare.sh'

bash scripts/vm_ssh.sh \
  'bash /mnt/workspace/scripts/run_overhead_replays.sh /mnt/experiment/runs/granite-aif/overhead_fanout'
bash scripts/vm_ssh.sh \
  'bash /mnt/workspace/scripts/run_overhead_replays.sh /mnt/experiment/runs/granite-aif/learned_overhead_fanout'

bash scripts/vm_stop.sh
```

Guest 固定 4 vCPUs、8 GiB RAM，保留 guest physical `4G..6G` 給 NVMeVirt。主機只需既有 `/dev/kvm` 權限。SSH 僅轉發到 localhost。使用 `EXPERT_FLASH_DATA` 可改 VM/NVMeVirt 資料根目錄，llama build 的對應變數是 `EXPERIMENT_STORAGE`。

完整 kernel patch：`patches/nvmevirt-aif.patch`，來源版本與 checksum：`patches/nvmevirt-source.json`。`kernel/aif.c`、`kernel/aif.h` 是易於閱讀的相同擴充來源。stock SSD 的 NAND baseline 與 AiF metadata timing 有各自的 counters；兩者不能混作相同 flash geometry。VM、早期 9p 問題與成功證據見 [docs/vm-nvmevirt.md](docs/vm-nvmevirt.md)，ABI 與非法需求測試見 [docs/vm-aif-abi.md](docs/vm-aif-abi.md)。

## 解讀範圍

本輪比較的是 routed FFN projection 的服務時間，不是完整模型 tokens/s。Attention、router、activation、combine、shared experts、dense layers、DRAM caching、GC 與 host/SSD tensor overlap 未納入。控制成本與傳輸精度仍須實體設計校準；不同成本會改變最佳 group。更進一步的跨 expert plane packing 需要多個矩陣 context，可能改善全 chip 配置，目前沒有假設硬體已支援。

本輪測試完成後已正常關閉 VM，釋放 guest CPU/RAM；磁碟與所有重播證據保留在 NFS。完整模型 hashes、程式 hashes、routing hashes 與檢查結果見 [results/experiment_manifest.json](results/experiment_manifest.json)。

論文沒有公開乘法器 lane 數；active-chip fraction、page/plane-slot fill、有效頻寬與 chip service occupancy 都不能稱為實測 MAC lane 使用率。三組 prompts 的結果是可重現的先導實驗，不能保證其他 routing 分布的最佳配置相同。論文依據與推導見 [docs/aif_model_notes.md](docs/aif_model_notes.md)。
