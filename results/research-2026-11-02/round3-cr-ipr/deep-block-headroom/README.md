# Deep GU：雙 block 幾何篩選

普通的 flat round-robin 已將舊三 block 配平配置的平均波數從 53.68 降至 45.28；一般容量修補與頁面搬移可達約 44.40，但 P99 增至 62 波。這輪沒有支持新的 layout 演算法收益，依預先規則停止，不擴大完整 FFN。

這是既有 W8/A8 TRAIN 的 26 層 × 512 個輸入，共 13,312 個 layer/query。全部擬合與評估使用同一批已見 TRAIN，沒有打開 heldout。來源、配置和搜尋規則先凍結於 [protocol.frozen.json](protocol.frozen.json)，再執行 fitting。源碼為 [deep_block_headroom.py](deep_block_headroom.py)。

| 配置 | 平均波數 | P99 波數 | 平均 cold blocks | NAND 幾何成本 μs | NAND/ECC 必要下界均值 μs |
|---|---:|---:|---:|---:|---:|
| Static flat RR | 45.2825 | 47 | 1.9929 | 475.7113 | 480.4218 |
| TRAIN frequency whole-expert grouping | 45.2486 | 47 | 1.8435 | 472.6484 | 480.2585 |
| Static 11/11 striped, uniform phases | 46.4795 | 48 | 2.0000 | 487.4511 | 488.0479 |
| TRAIN rare-owner capacity repair | 44.3948 | 62 | 2.0000 | 467.2299 | 481.8398 |
| TRAIN generic one-page moves | 44.4049 | 62 | 1.9938 | 467.2134 | 481.5952 |
| 原三 block compensation，額外容量參考 | 53.6811 | 58 | 2.9979 | 575.5683 | 575.5683 |

最後一欄是必要下界，沒有量測或證明可達的完成時間。每個 query 的下界取 `max(NAND geometry cost, 28 μs + 176×16384 / 6.4 GB/s)`，第二項為 478.56 μs。容量修補降低平均 NAND 波數，卻因 rare experts 的長尾使這個下界均值高於普通 RR；不能把波數改善直接寫成速度改善。四輪一般 page moves 以 NAND 成本為主目標，因此可能略增波數以減少 touched blocks；沒有宣稱全域最優或 tail 最優。

每個匹配 control 在每顆 chip 都配置相同的兩個 block，四 planes、每 plane/block 192 個 WL，共 1536 個 page slots。所有 control 都只寫入 1452 個 pages，query 精確讀取 176 個 pages，沒有 replica、overfetch 或額外 coefficients。共享 GU 的 44 頁已計入容量，routed expert 各 22 頁。原三 block compensation 仍寫入相同頁數，但預留容量較大，故明列為額外容量參考。

實際位址存在 [actual_page_address_ledgers.json.gz](actual_page_address_ledgers.json.gz)。排程依 block 0、block 1 順序進行；各 block 中，每 plane 的 requested pages 按 WL 排序，wave k 取每個尚有第 k 頁的 plane，使用固定 `0123` source 順序。active plane 集合只會縮小，閒置 plane 不會在同一 block 內恢復成 CR hot。每個 active block 首波付 28 μs，之後每波付 9.7 μs，計入明示的任意 WL skip-CR 前提。C=2048 的每頁包含八個完整 rows，所有 dot 都讀完整 columns，保留原 expert/row 身分再 scatter；沒有 partial-dot context。原始 receipt 的文字錯字與 RR whole-row ordering 說明保存在 [metadata_erratum.json](metadata_erratum.json)。

176 頁的全域 work 下界是 44 波，但不是可執行的 block 排程。對某個實際 layout，`sum_b max_p selected[b,p]` 才是上述排程的波數；`sum_b ceil(selected_pages[b]/4)` 是較弱的 block 整數下界。允許各 plane 獨立進入不同 block 的工作量只保留為 optimistic 下界，沒有賦予主要 controls 這項能力。全域 universal-44 的必要容量條件為 `44+(64-6)*ceil(22/4)=392` 頁/plane，大於雙 block 的 384 頁；這不是 finite-PB、row scheduling 或 native timing 的充分條件。獨立的小例全枚舉證書在 [capacity-balance-bound/report.md](../capacity-balance-bound/report.md)，條件限定 owner-homogeneous 整頁、任意 top-K、零 overfetch 與固定 shard geometry。

Static RR 已關閉舊三 block 配置相對 44 波下界約 86.75% 的平均幾何差距；一般 TRAIN compiler 關閉約 95.82%。剩餘差距小、存在尾部取捨，而且 finite PB、ECC 競爭、命令、輸入、輸出與完整 FFN 都未執行。頁面 striping、頻率排序、容量修補、座標搜尋和單頁搬移均視為標準控制，不宣稱新演算法。這些結果不能推廣為任何硬體的實測效能。

數值、histograms、每層 resident counts 與 lower bounds 在 [report.json](report.json) 和 [TRAIN_maps_and_metrics.frozen.npz](TRAIN_maps_and_metrics.frozen.npz)。獨立 [independent_audit.py](independent_audit.py) 不 import fitter，直接由物理 ledger 重建所有 routes 的頁面排程；結果在 `independent_audit.json`。`representative_NAND_only_traces.json` 提供兩個完整輸入的逐波小 trace。

重現時，把 fitter 複製至本 repository 內另一個空白 sibling 目錄，再依序執行 `python3 deep_block_headroom.py freeze` 和 `python3 deep_block_headroom.py run`。程式拒絕覆寫既有封存結果。獨立 audit 可直接對本目錄執行：

```bash
python3 results/research-2026-11-02/round3-cr-ipr/deep-block-headroom/independent_audit.py
```
