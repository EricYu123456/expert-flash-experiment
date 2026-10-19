# MoE routed FFN 的較快 AiF 配置

本輪找到的改進方法是：依成本選每個 expert 使用的 chip 數、將 gate/up 實體串接以延長 cr-read，再以共同啟用紀錄學習固定容量的 expert 放置。Granite 改善明顯，OLMoE 與 DeepSeek V2 Lite 收益較小；沒有適用所有模型的最佳 chip 數。

下表為 v3 controller-fanout 模型的 **routed FFN 服務時間，ms／decoded token**。每模型使用三組 prompts、各 32 次 decode；分組與放置只用兩組訓練，第三組驗證，數字為三折 held-out 平均。16-chip 已串接版本是較強的比較基準。

| 模型 | 16 chips，gate/up 分開 | 16 chips，gate/up 串接 | 訓練選分組＋串接，原排列 | 分組＋串接＋學習排列 | 選中 g：原／學習 | 較 16-chip 已串接降低 |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| Granite 3.1 1B A400M | 18.920 | 12.703 | 8.378 | **7.452** | 4／4 | 41.34% |
| OLMoE 1B 7B | 16.568 | 13.648 | 13.390 | **12.824** | 8／8 | 6.04% |
| DeepSeek V2 Lite | 24.917 | 21.333 | 21.333 | **21.015** | 16／8 | 1.49% |

![三種模型的配置比較](/home/ericyu/expert_flash_experiment/results/optimized_layouts.png)

三個模型的 expert 維度 H×I 分別為 1024×512、2048×1024、2048×1408，top-k 為 8、8、6。真實 llama.cpp routing 共 2,304／1,536／2,496 筆；開關 tracing 的 greedy 輸出逐位元組一致。DeepSeek 只比較 26 個 MoE layers 的 routed experts，未加入 shared experts 或首層 dense FFN。

## 為什麼全部 chip 有工作仍可能較慢

每個連續 block run 的時間為 `28 + (P−1)×9.7 μs`，第一次 precharge 與最後 discharge 已含在 28 μs。Granite 單個 0.5 MiB projection 切成 16 份後，每 chip 只有 32 KiB，填不滿四個 16 KiB planes 的一個 read wave，卻仍須支付 28 μs。其純 flash 有效頻寬只有理論峰值的 18.3%；加入控制與向量成本後為 15.6%。縮小 group 可延長每 chip 的連續讀取，並讓不同 experts 在不同 groups 並行。

Gate/up 共用輸入，故串接成連續 `2I×H` 矩陣，輸出再拆回 gate/up，完成 activation 後才執行 down。學習排列只交換 expert 的實體 placement slots，保留原選中 IDs 與每組容量，減少同時啟用的 experts 相撞。每層最多抽樣 100 次交換；這是啟發式配置，非全域最優解。

## 控制與傳輸成本

採用論文的 INT8、16 chips、四 planes/chip、16 KiB/page、6.4 GB/s/chip。v3 將每份 logical input 經 PCIe 傳一次，再由 controller 經八條 channels 複製到各 chip；PCIe／channel 頻寬為 8／2 GB/s。輸入 1 byte、輸出 4 bytes/element，以及 **0.25 μs/chip command、2 μs/host phase 都是展示用假設**，並非實體 AiF 校準值。已掃描控制成本 0–5 μs、host 成本 0–10 μs，且在每個成本設定下重新以訓練資料選配置。

![控制成本與較佳配置](/home/ericyu/expert_flash_experiment/results/control_tradeoff.png)

虛擬實體配置讓短 streams 共享 block，但各自保留一次 cold run；不夠放時跳過尾端，長 stream 才跨 block。所選學習配置的 LSB 保留容量為 **1.125／6／14.25 GiB**，含 padding；對應 TLC cell 容量 **3.375／18／42.75 GiB**，約占 1 TB 的 0.36／1.93／4.59%。已核對 rows、pages、cold runs 及位址無重疊；沒有把這些權重寫進 2 GB metadata-replay namespace。

## 驗證與解讀範圍

v3 主要結果已完成 **48 組 QEMU/KVM＋NVMeVirt queue 重播、183,744 個 NVMe 命令**；逐 chip bytes、pages、read rounds、cold reads、flash service，以及 control、host、PCIe/ONFI 向量成本全部與 Python 預期值吻合，實際 ioctl wall time 另存以區別 VM/driver 開銷。含較早純 flash 與相容性對照，共核對 100 組、813,504 個命令；[逐項驗證彙整](/home/ericyu/expert_flash_experiment/results/nvmevirt_validation.json)。

13 項需求守恆、頁面尾端、block 邊界、向量 fanout 與成本測試通過；完整 kernel patch 也在乾淨的 pinned source 套用並核對 hashes。QEMU guest 使用 4 vCPUs、8 GiB RAM，其中 2 GiB 保留給 NVMeVirt；全程沒有 host sudo。

實驗完成後已正常關閉 VM。模型與完整 artifacts 都在使用者指定 NFS 目錄；本機工作資料夾僅約 3.5 MiB。模型、程式與 routing hashes，以及最終驗證狀態見 [experiment manifest](/home/ericyu/expert_flash_experiment/results/experiment_manifest.json)。

結果不含 attention、router、activation/combine 計算、shared/dense FFN、GC 或 DRAM cache，不能換算完整模型 tokens/s。論文未公開 MAC lane 數；有效頻寬與 active-chip fraction 也不能當作實測 PE 使用率。三組 prompts 的訓練改善通常大於 held-out 改善；分布改變時須重新驗證，per-layer 自適應 g 亦未穩定優於固定 g。

原始 CSV、mappings、hashes 與 physical plans 均在 [NFS runs](/home/nfs_client/ericyu/expert_flash_experiment/runs) 的 `NAME-aif/overhead_fanout/`、`NAME-aif/learned_overhead_fanout/`；較早的 unicast 分析保留在舊目錄，不與本表混用。重現步驟見 [README](/home/ericyu/expert_flash_experiment/README.md)，版圖證明見 [physical placement](/home/ericyu/expert_flash_experiment/docs/physical_placement_plan.md)。
