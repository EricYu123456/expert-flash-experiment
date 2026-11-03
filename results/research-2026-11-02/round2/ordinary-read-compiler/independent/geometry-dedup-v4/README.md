v4 的 342 records 通過本輪 correctness／attribution audit。14,160 個 canonical phase fits 的 labels、base_rows、mean objective 與 fitstats，全部精確等於 v3 同 search space 的首次 fit；全部 objectives 另由原 1,024 training routes 與獨立 ceil/page/output 算式重算，逐值相同。共去除 2,658 個重複 phase fits，沒有新增 fit 或 seed。

去重鍵在現有假設下**充分但不必要**。每個 expert 都可使用相同 variants 表；所有 expert-pair transpositions 都允許；沒有 ordinal-specific cost 或 expert-specific owner 約束。Pair swaps 可生成初始 ordinal-label multiset 的全部排列。投影成 physical row vectors 後，可達集合恰好是初始 assigned row-vector histogram 的全部排列：對任何目標 physical assignment，可在每個相同 row-vector 類別內分配回原 ordinal-label counts。因此相同 assigned histogram 已充分；額外要求 variant multiset 相同是保守限制。

已用 BFS 窮舉 31,062 個 `M=1..4,E=1..4` 小類比，包含所有三值 variant tables、duplicate ordinal variants 與全部初始 labels，未發現錯誤合併。另只抽取實際 v4 snapshot 的 `initial_labels`／`deduplicate` 函式測 `E=1,2,8,16,64,512`，沒有 import live backend。E8 g1 的 disjoint／cyclic inventory 不同，保留正確；g16 全 uniform vectors，合併正確。E1 g1 也可合併，因兩者唯一 expert 都在 chip0；不能泛稱所有 `E<M` 都必須分開。

找到一個 **E8 漏合併反例**：`M=16,R=4096,g=2`，disjoint 初始 labels `0..7` 對應 starts `0,2,...14`；cyclic 初始 labels `0,2,...14` 也對應同 starts。兩者初始 physical assignments 逐 expert 相同，全部 8! = 40,320 個可達 assignments 逐項匹配。但 disjoint 的 variants 是 8 種各重複兩次，cyclic 是 16 種各一次，故 v4 較保守的 key 仍保留兩個 fits。這不是誤合併，但若將相同 policy 用於 E8，可留下額外 seed 機會。

目前 342 records 僅涵蓋 E64／E512。對全部 684 phases 再以 **assigned histogram alone** 檢查，沒有任何殘留重複 search spaces，因此上述 E8 反例不影響本輪結果。Root 已知此 scope；本 audit 沒有修改 sealed v4。未來採 histogram-only key 時也必須保留 unrestricted-swap 等假設。

Family eligibility 現在用 canonical `family`，而不是 `families` union；在本 snapshot 的 disjoint→cyclic→whole_rows 生成順序與巢狀 allowed sets 下，canonical 一定是 aliases 的最小 rank，所以兩種判定完全相同。全部 records 均核對此 invariant、finite-pool pair 數、候選索引、16 relative rotations、prefix incumbent 單調性與 replay budget／certification flags。若將來改 candidate 順序，這個推論需要重新成立。

已從 1,026 個 NPZ **只讀 cost 與 selection_routes** 重算全部 27 個 multilayer totals；未讀 jobs、未掃 PPA、未做事件 replay 或 native numeric inference。Global maps 的地址合法性由 architecture audit 負責。Finite-pool flags 的一致性已核對，但沒有重算全部候選對的 LB；這依賴先前 backend/LB audit。`certified` 只表示固定 fitted finite pool、相同 greedy scheduler 的最佳 mean，並不表示最佳 P99、全域 mapping optimum 或真硬體速度。

| Model | Command ns | cyclic mean ms | whole-row mean ms | 額外 mean gain | cyclic／whole certified layers |
|---|---:|---:|---:|---:|---:|
| Granite4Tiny | 250 | 23.549926 | 23.274385 | 1.170030% | 40／40 |
| Granite4Tiny | 5,000 | 27.731588 | 27.731588 | 0% | 40／40 |
| Granite4Tiny | 10,000 | 35.213906 | 35.213906 | 0% | 31／40 |
| DeepSeek | 250 | 52.356580 | 50.453269 | 3.635286% | 26／26 |
| DeepSeek | 5,000 | 55.175141 | 54.015361 | 2.101997% | 26／26 |
| DeepSeek | 10,000 | 59.710988 | 59.487334 | 0.374561% | 16／0 |
| Qwen3.8 | 250 | 82.203331 | 76.566537 | 6.857136% | 48／48 |
| Qwen3.8 | 5,000 | 89.653081 | 88.523787 | 1.259626% | 48／0 |
| Qwen3.8 | 10,000 | 99.239695 | 99.239695 | 0% | 43／48 |

全部 206 個有正 gain 的 layer-cost cases 都選到真正新增的 row geometry，沒有僅靠 cyclic／disjoint 別名的新 seed 勝出。250 ns 下的具體差異是：

- G4：37／40 層由 GU g8 的 24 total waves、depth3、8 jobs，換成不等 rows 的 25 waves、depth2、16 jobs；down 仍 cyclic g12。更多 work 但較淺，不可寫成純 NAND work 減量。
- DeepSeek：26／26 層採 GU 88-wave、down45-wave 的不等 rows，較 g16／cyclic g15 改變 work distribution；shared rows 全 family 固定。
- Qwen3.8：48／48 層採 GU51／down26 waves 的 16-chip 不等 rows。主要 cyclic control 是 GU52／down26 waves、13 chips；其中一層 GU cyclic g10。不是把相同 row geometry 換名稱。

Qwen3.8 10 μs 的 v3 額外 gain 為 0.504974%，修正後為零。v3 有 37 層的 positive winner 全屬 cyclic 已有 search spaces，例如 down `depth_5` 與 `cyclic_5`；v4 保留 canonical 首次 fit，不再從別名額外 seed 取得優勢。5 μs 還有 1.259626% candidate gain，但 whole-row 全 48 層都用盡 replay budget，不可稱 finite-pool optimum。DeepSeek 10 μs 也只有 candidate gain。

所有結果都是已使用過的 train development population、合成 ordinary-read service model，排除 attention／SSM／首層 dense 等全模型成本；不構成新穎性、heldout generalization、原 checkpoint 品質或 device-performance 證據。

可重現檔案：`audit.py`、`searchspace_guard.py`、`audit.json`、`equivalence.json`、`attribution.json`、`searchspace_guard.json`。Source provenance 以 v4 `source-snapshot/manifest.json` 的五份實際 snapshot 為準；live backend 後來加入 physical-import guard，不把它錯認為本 campaign 執行 source。
