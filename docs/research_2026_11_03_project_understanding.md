# 本次研究的專案理解

日期：台北 2026-11-03。此文件整理既有程式與證據，不是本次新方法的實驗結果。

## 專案實際在做什麼

專案以 llama.cpp 真實推論擷取每個 token／layer 的 MoE routing，再將權重形狀、固定地址與相依關係送入 AiF 成本模型。原 NVMeVirt 擴充在 VM 中使用真實 Linux queue；NAND sensing、ECC／PE 與傳輸延遲仍是建模，並無實體 AiF 晶片。

本地 `/home/ericyu/expert_flash_experiment` 包含程式、PDF、文件與結果；模型、原始來源、編譯產物及 VM 多位於 `/home/nfs_client/ericyu/expert_flash_experiment`。目前工作區有大量既存修改與未追蹤研究檔；本次另開日期路徑，保留原始證據。

AiF 是 chip 內的數位 INT8 GEMV、ECC 與控制邏輯。主模型為 16 chips、每 chip 4 planes、16 KiB page；be-enc LSB 普通讀取 28 μs，CR-read 9.7 μs。CR-read 依賴同 block 無間隔的連續讀取與電位回收；本次不得依賴它。6.4 GB/s/chip 是管線能力參數，關掉 CR 後不能視為可達 NAND 讀取率。

AiF 利用 TLC page-buffer 的空閒 latches 存 input。新增多向量重用、任意 plane context、可回存 partial sum、nibble unpack 或 raw-I32 accumulation，均須明示契約及硬體差額。論文沒有公開全部 PE lane、output dtype、scale／rounding 與 buffer 容量細節。

## 已經走過的路

1. 起始方法是 gate/up 合併、每 expert 固定 g 個 chips、以共選 routing 學習 group 放置。舊 g8 已在部分形狀提供大於 10% 的收益；不能算成本次貢獻。
2. 後續 ordinary-read 研究拆開 GU 與 Down 的幾何，研究完整 rows、整數 read waves、不等分片及 cyclic／overlap 排列。有效收益依模型而異，較強控制後多為幾個百分點。
3. 既有工作已進一步檢查共同 phase placement、小副本、prefix views、GU 合併、native DAG、controller flow、32-chip 情境與強搜尋控制。許多看似大收益被同能力、同容量或簡單幾何基準抵消。
4. 新的 W8A8 CPU 參考、實體 byte 映像、地址與獨立 checker，補上同格式數值與 layout 證據；仍不是原生 flash PE 量測或完整模型硬體加速。
5. Shared 的後期成本已有部分納入，早期 routed-only 主表則沒有；各分母不能混用。Q4_K／Q6_K／Q8_0 的真 routing 配 nominal raw-I8 成本，也不能叫同格式端到端推論。

## 為何需要新機制

OLMoE、Mixtral 在既有 ordinary-read、單請求、每個完整必要權重讀一次的模型下，已達 NAND 工作量下界。更換排列不能刪掉必要 sensing；幾何或排程仍有零星空間，但目前證據不足以支持通用的大幅加速。

因此，本次優先找能減少每 token 必讀資訊、讓一次 sensing 服務更多已 ready 計算、或改變獨立普通讀取成本的新機制。允許小幅硬體修改擴大了搜尋空間，也使同硬體能力與品質基準更重要。若新方法採近似，必須在真實推論中評品質，而非只對舊 routing 重播較少 bytes。

## 投稿缺口與本次交付方式

真正需要補的是：一個相對強 prior 仍有必要性的機制、合法且可估成本的最小硬體改動、有／無 shared 的實際模型證據，以及品質與完整延遲的公平比較。局部 FFN 服務時間、架構模擬、CPU 參考與實體量測會分開呈現。

本次進度與輪次見 [研究協定](research_2026_11_03_campaign.md) 及 `results/research-2026-11-03-no-cr/state.json`。截至正式候選開始時，尚未跑任何本次新方法實驗；既有結果不占十輪機會。

主要依據：`README.md`、`docs/aif_model_notes.md`、`docs/ordinary_read_placement_candidate.md`、`docs/research_2026_11_02.md`、`docs/research_2026_11_02_round2.md`、`paper/aif.txt`，以及對應結果與程式。角色個別筆記保留具體讀取、推導與來源紀錄。
