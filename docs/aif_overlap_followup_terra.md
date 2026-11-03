# AiF 普通讀取、ECC、PE 與跨 cohort overlap：來源與未揭露契約

查核時間：2026-11-03 06:04 UTC（本機 UTC clock）。範圍：論文資料路徑與鄰近 NAND cache-read 的原始文獻；沒有執行模型、RTL、模擬或取得作者資料。本文的 *ordinary read* 指每一被讀 WL 均有完整 precharge、evaluation、discharge；禁止 AiF 的 charge-recycling **cr-read**。cache-read 是另一種資料暫存與指令概念，名字相近但不代表 charge recycling。

主要依據：AiF，Lee et al., ISCA 2025，§2.2、§4.2–4.4、§5.1–5.2、§6.1，原始論文 DOI <https://doi.org/10.1145/3695053.3731073>；此次核讀的是本地論文文字抽取 `paper/aif.txt`，以下行號均指該檔。ACM 網頁此次未能直接打開，故以本地論文原文與 DOI 定位；文字抽取存在雙欄交錯。與一般 NAND 鄰近的第一手專利：<https://patents.google.com/patent/US20190220226A1/en>，圖 2 與背景敘述 [0040] 前後（Google Patents 行 243–253）。專利描述技術，不是 AiF 實作量測。

## 結論與逐項原文

1. **一個 chip 的四個 16-KiB plane page 共 64 KiB，PE 是否必須整體等普通 28 µs 完成？未知，不能一概斷言。** §2.2（190–210）把普通讀取分成 precharge/evaluation/discharge；§4.2（535–543）明定 `tR=tPRE+tEVAL+tDISCH`；表 2（921–924）列 (1,3,3) LSB 的 28 µs。§4.4（761–782）說矩陣片段在讀取時與向量從 PB 取出，每個矩陣片段先經 ECC 才進 INT8 PE。資料依賴保證的是 **相應 raw bits 有效 → 相應 ECC 修正完成 → 對應 PE 計算**，不是公開保證 PE 必須等待四 plane 整體的完整 tR。論文沒有提供 page latch/data 或 ECC codeword 的 release time，也未說明 discharge 與 latch/ECC/PE 的資源隔離；在 discharge 中開始 ECC/PE 是有待實作驗證的可能，不是已實現的能力。反過來，28 µs 不是包含 ECC、完整 PE 與結果 drain 的量測上界。`28 µs + 全 ECC + 全 PE` 是可採用的保守序列化模型，並非論文證明的唯一時序。

2. **K=6、四個 chip groups、兩輪 cohort：後輪讀取是否能和前輪 PE overlap？未知。** §4.4 說 planes 連續讀矩陣，並描述「as matrix segments are read」進行運算（761–782），但沒有定義下一頁或下一條 GEMV 命令可否在前頁/前 cohort PE 服務仍在執行時啟動 sensing、ECC 及其衝突處理；同一命令的連續頁流也不能直接推出跨命令 context 重用。§5.2 的 `aif_gemv` 帶 input dimension/address、matrix LBA 和 output address（870–877），沒有揭露並行命令、input 殘留、FIFO 所有權及 context lifetime。四 group 可以各自完成，不應憑空假設兩輪間有全域同步 barrier；重點是**同一 chip/plane 的後續 cohort**能否在前次尚使用其 PB/ECC/PE/輸出資源時進行。

3. **已有 latch 和可流式結果，但不是雙緩衝與無背壓的證明。** §4.4（724–743）說每 plane PB 至少四個 page latches：一個供資料 transfer 的 cache latch，三個為 data latches。LSB 單次 sensing 只需一個 data latch，其餘兩個可供 input vector；不能把兩 latch 自動解讀為兩個彼此獨立、可切換的 input contexts。該 cache latch 是否空閒、能否容納舊或新 weight、是否可同時由 ECC/PE 讀取與 plane 寫入、input latches 是否被重用/覆寫及需要哪些端口皆未公開。故不能斷言必須新增 weight latch，也不能認定已有可用的雙 buffer。文中 PE 只描述 INT8 multipliers 與 adder tree；四個 plane 的 PE/ECC 共用、port 數、lane 數及仲裁亦未明確定量。

   論文（816–823）確實說產生的 output elements 放入 FIFO，controller 輪詢各 chip 取得 FIFO 結果，SREG 追蹤 FIFO status；可支持**同一 computation 中 output drain 與 output generation 並進**。FIFO depth、接口頻寬、滿時 ready/backpressure、in-flight 數、能否跨命令保留/標記兩份結果均未披露，亦不能推出 drain 與下一次 sensing/下一 GEMV 一定重疊。§5.1 論述 host 與 AiF 的 head/tensor-level 重疊（811–835），是另一層的並行，不證明 chip 內部 read/PE overlap。§6.1（870–879）虛擬 SSD 根據 timing model 回傳 dummy vector，而非數值運算量測，故不能從結果推斷 FIFO/PE 實際波形。

4. **ECC 6.4 GB/s 不是 PE throughput。** §4.4（816–817）給的是 ECCLITE 的 decoder throughput，PE 速率、PE lane 數和 PE 吞吐均沒有同等明示。若並且只有在這個 6.4 GB/s 是四 plane **共用的 chip 累計資料服務率**、適用所需 workload 且持續達峰值時，64 KiB / 6.4 GB/s 約 10.24 µs 是假設下的 ECC 理想服務時間；它可能與其他 stage 重疊，不能加成必然串列費用，也不能用來代替 PE 速率。論文另在 §6.1（899–904）也使用 6.4 GB/s 作為特定 AiFChip 有效 flash bandwidth；名稱和數值相同不等於已證明 ECC 及 PE 共享相同速率或資源。

## 最小增量是契約，不是既成的面積估計

要驗證 `next cohort sensing` 與 `previous cohort PE` 的實際重疊，至少要有下列資源及生命週期保證；某些也許已由 AiF 現有電路提供，文件未定，故下列**是需求而非已知新增項**：

- 確保每次普通 read 的 WL/BL 完成 discharge，且下一 sensing 只有在該 array/plane 允許且不破壞尚需資料時開始；明定 sense→raw-data valid、ECC codeword valid、每頁可釋放時間。透過 RTL/波形量測並驗證各 plane 和共用資源的衝突，不把 cache 指令當成電位證據。
- 明確定義舊 weight、下一個 raw weight、input 的 buffer ownership、可同時使用的 port/transfer/mux 與 ECC/PE 仲裁。若已存在的 cache latch 可同時安全保存上一頁並提供 PE/ECC 獨立讀取、且下一頁能寫入另一 latch，額外 weight 容量可能為零；這些能力必須驗證。若現有 latch 和 ports 不夠，一種**條件式整頁 lookahead 設計例**是每參與 plane 多一個可供雙緩衝的 16-KiB weight buffer，即四 plane 共 64 KiB/chip、16 chips 共 1 MiB；這不是下界或 AiF 額外面積測量。較小 chunk buffer 也可能可行，但大小須由 ECC 真正的交付單位、資料路徑/port、消費率與需要覆蓋的時間決定；論文「每 1 KiB 可修正 10 bit」不單獨證明可按 1 KiB 任意提前釋放。額外 buffer 的 mux、路由、仲裁、控制及 port 實現仍需計入。
- 命令排隊和 context ID、input/scale lifetime 與 output 排序必須明確；若 cohort 使用相同 input，可嘗試保留一份，但不能假設 input latch 跨命令保留；若不同 input 在前次消費結束前即開始後次，需另一份 context 或等待，包括 vector bytes、比例因子、tag 與所需 port。
- 對 FIFO 加入或確認無丟失的 producer/consumer 握手、足以覆蓋 in-flight 輸出的 credit/backpressure，並標記次序或 context。必要深度由 `max_t(produced(t) − drained(t))` 決定，包含不能立即停下的 pipeline；既有 FIFO 也許已足夠，額外容量可以是零。**僅為明確例子**：若條件為每頁 128-byte 完整 row、I32 輸出，16-KiB page 有 128 rows、四 plane 一個 chip 共 512 個 I32＝2 KiB；若整個 cohort 結束前完全不 drain 且無其他儲存，就須能存 2 KiB/chip。兩個完整 cohort 都累積則需 4 KiB/chip。可及時 drain 或可安全 stall 時需要較少；這不是 native FIFO depth 或保證 128-byte rowpack 的論文結論。

不同時序假設應分開：保守無 overlap、頁/片段 overlap、跨 cohort overlap；分別按已知或掃描的 ECC/PE/輸出/控制速率、buffer 與 command overhead 計算，不能僅把每 wave 固定設成 28 µs。上述資源上限、面積與功耗尚無量測，也沒有據此宣稱修改幅度小。

## 重要 page-type 邊界

表 2（921–924）的 (1,3,3) CSB/MSB 46 µs 是普通 page read 數字，本身不是 charge recycling；但 §4.3（715–720）與 §4.4（733–743）說 CSB 需三次 sensing、三個 data latches，(1,3,3) 下 MSB 也需三次 sensing，而 AiF 的 input PB 重用依賴 LSB 僅佔一個 data latch。故不可把 46 µs 直接代入並宣稱原本兩個 input latch 同時可用；可能需要額外 input buffer、另一種已驗證 latch lifetime/串流、或讀後再載入 input，及其時間與資料搬移成本。甚至「讀後再載入」也須先保住所需的完整 page result，不能假設覆寫沒關係。ECCLITE 的修正能力依 LSB 錯誤特性設計（722–732、816–818），CSB/MSB 的可靠度與 ECC 能力亦須獨立驗證。LSB-only、be-enc (1,3,3) 容量足夠時才保持本文主要 28 µs 配置，並另計該容量需求；不能把 LSB 的 latch 與 ECC 條件免費挪到其他 page 類型。

## 一般 NAND cache-read 的可用證據與界限

上述專利圖 2 及背景敘述描述分離 cache register / page register：先前 page 從 cache register 輸出時，下一頁可從 NAND array 轉移至 data register；並描述 sequential cache read `31h`、random cache read `00h-31h` 與最後 `3Fh`，以及 ready/busy 狀態。這是 **外部資料輸出與下一頁讀取的通用 NAND 管線範例**，不是 AiF 的 in-die ECC/PE pipeline、四 plane buffer mapping、cross-GEMV scheduler、input retention 或 FIFO 深度的證據。專利相關段落沒有給出可驗證完整 precharge/discharge 的物理波形，故 cache-read 和 charge-recycling 在概念上不同，亦不能由其名稱或文件單獨保證某裝置該 mode 符合本研究 ordinary-read 電位契約。
