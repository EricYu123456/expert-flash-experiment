# 同一 chip 內的 plane 配置與 read-wave 利用率研究

日期：2026-10-30，Asia/Taipei。程式、資料與結果皆位於本次新增的 `plane-wave-v1` 實驗；原有實驗結果未覆寫。

**固定使用 16-chip group 時，固定的 plane 分工確實能改善部分模型：Granite 的兩-plane／單-plane方案，分別降低 29.21%／35.46% 的模型延遲。但允許重新選擇 chip group 後，五個模型仍全部選回原本的四-plane方案。** 因此，這次研究找到了可改善「固定全-chip配置」的方法，尚未找到優於專案既有最佳配置的方法。

以上百分比採用與舊實驗一致的理想管線時序；它們是**附帶硬體前提的架構模型結果**。新方案尚未在實體 NAND、AiF PE 或 NVMeVirt 新排程 ABI 上實作。本文另外提供不保留 cr-read 狀態的普通讀取排程，避免把理想界當成已證明可達的延遲。

## 1. 問題：page padding 與空 plane slots 要分開

單一 expert 的某個 projection，在一顆 chip 上分到 `S` bytes 時：

```text
L = 16 KiB
P = ceil(S / L)                   實際 pages
W = ceil(P / 4)                   舊配置的四-plane waves
page padding = P × L − S          最後一個 page 裡沒用的 bytes
idle slots = 4 × W − P            wave 裡沒放 page 的 plane 位置
useful slot fill = S / (4 × W × L)
```

| 每 chip 的 stream 大小 | Pages | 舊配置 waves | Page fill | Useful slot fill |
|---|---:|---:|---:|---:|
| 32 KiB | 2 | 1 | 100% | 50% |
| 40 KiB | 3 | 1 | 83.33% | 62.50% |
| 64 KiB | 4 | 1 | 100% | 100% |
| 80 KiB | 5 | 2 | 100% | 62.50% |
| 96 KiB | 6 | 2 | 100% | 75% |

40 KiB 的例子同時存在兩種浪費：page 裡的 8 KiB padding，以及一個空的 16 KiB plane slot。把不同 streams 放到互補 planes，可以處理空 slots；若不允許跨 stream 混放同一 page，page padding 仍然存在。

本文的 useful slot fill 以各 chip 實際參與的 waves 為分母，包含 wave 內的空 planes；它不是整個裝置時間軸的利用率，也不是實測 MAC lane 使用率。

## 2. 實作與比較的方法

所有方法都先保留完整 output rows 的 chip 分工。進入單一 chip 後才以 page 分配到 planes；**page 不必等於一列，完整的一列可以跨 pages／planes**。數值驗證包含 1,025-byte rows 跨越 16 KiB page 邊界的情況。

### A. 固定兩個或一個 planes 給一條 stream

| Policy | 每條 stream 使用的 planes | 每 chip 最多同時執行的 streams |
|---|---|---:|
| baseline，width 4 | `{0,1,2,3}` | 1 |
| pair，width 2 | 固定 `{0,1}` 或 `{2,3}` | 2 |
| single，width 1 | 固定一個 plane | 4 |

例如兩條各 32 KiB 的 streams：

```text
舊配置                         固定互補 plane pairs
          P0  P1  P2  P3                   P0  P1  P2  P3
wave 0    A0  A1   .   .         wave 0    A0  A1  B0  B1
wave 1    B0  B1   .   .
```

`A/B` 的 plane 位置在部署前固定。若 A、B 恰好分到同一 pair，仍須串行，不能在看到此 token 的 routing 後把 B 搬到空 planes。

固定 bin 的學習只使用 training routing：在同一 chip group 內交換 expert 的 plane bin，維持每個 bin 的 expert 容量。每 layer／g／width 最多 256 次 swaps，先降低加權 flash 時間，再以平方 occupancy 解同分。該 layer 的 gate/up 與 down 共用同一份固定 bin map；沒有針對測試 token 動態變更位置。

這個方法需要獨立 plane 定址、子集合讀取命令、多個矩陣 contexts，以及輸入／累加器／輸出歸屬管理。gate/up 可共用 layer input；down 的各 expert 使用不同 activation vectors，必須保留獨立 input contexts。四個 planes **共同受限於一顆 chip 的 6.4 GB/s ECC/PE 上限**。

### B. 單一 context 的固定 gate/up supermatrix

將同一 chip group 內固定 `B=2` 或 `4` 個 experts 的 gate/up shards，垂直串成一個普通矩陣。因為它們共用同一個 layer input `x`，可作為一次單一矩陣 GEMV。實際儲存採 chip-major、再 expert-major 的 row 順序；輸出 metadata 可還原 `(expert, gate/up, 原始 row)`。

為了維持單一連續 stream 與共同 WL 的普通四-plane命令，**只要選到 bundle 的任一成員，就讀取、計算、回傳整個 bundle**。未選 experts 的權重、乘加與輸出流量全部計入；host 最後才丟棄無用輸出。down 維持原本各 expert 的獨立 stream。

這個方法不需要獨立 expert plane contexts，也沒有複製權重；代價是 overfetch。學習採 training coactivation 的 greedy 分組，再執行最多 64 次同 group、容量守恆的 swaps。以完整 training 延遲選擇 `g` 與 `B`，並保留 `B=1` 作為不合併的退路。

### C. 動態任意 tail packing

把本次選中的所有尾頁臨時湊成滿 wave，可以作為機會上界；若未處理固定位置、block/WL、權重搬移與 context，就不能當成可部署方案。本次主要結果全部使用固定配置，沒有把這種 oracle 算成實驗改善。

## 3. 硬體與時序邊界

原 [AiF 論文 §4.2、§4.4 與 Table 2](https://doi.org/10.1145/3695053.3731073) 提供了四 planes、16 KiB page、28 μs 普通 read、9.7 μs cr-read、每 plane PB，以及一條矩陣命令的 block／offset／page-count 等資訊。它沒有直接證明不同 experts 可以在不同 planes 各自執行。cr-read 的條件是同 block 的連續讀取，不能任意在中間加上停頓後仍沿用原成本。

因此這次明確保留三種時序：

| 時序名稱 | 計算方式 | 解讀 |
|---|---|---|
| `ideal_overlap` | `max(最忙 plane bin 的 NAND 時間, 所有 pages 的共享 ECC/PE 時間)` | 精確相容舊模型的理想資源下界；主要公平比較使用此欄 |
| `retained_cr_schedule` | 每同步 wave sensing 後依共享頻寬 drain，仍沿用連續 cr-read 成本 | 額外要求 drain 間隔仍能保留 cr 狀態；尚未由原論文證明 |
| `conservative_drain` | 每一個 wave 都收 28 μs 普通 read，完成後再 drain | 不依賴跨間隔保留 cr；仍需要候選方法的 plane 定址與 contexts |

若一條 stream 的 striping width 為 `w`：

```text
P = ceil(S / 16 KiB)
R = ceil(P / w)
C = ceil(R / 192)
T_NAND = R × 9.7 μs + C × 18.3 μs
T_ECC  = 所有已讀 pages × 2.56 μs
```

不同 expert 或 block 之間重新收 cold start，不能因實體相鄰而延續 cr-read。`192 LSB WLs/block` 沿用既有推導值，不是新量測。

NAND multi-plane 的位址限制也取決於版本與產品。[ONFI 3.0 §3.1.1](https://onfi.org/files/onfi_3_0_gold.pdf) 與 [ONFI 5.1 §3.1.1](https://onfi.org/files/onfi_5_1_final_1%2C-d-%2C0%20%281%29.pdf) 的 page-address 規定不同；本次官方 PDF 直連受阻，細節由官方搜尋索引正文核對，不能據此宣稱特定 AiF NAND 已支援獨立位址。另一方面，[Micron 官方的六-plane NAND 公告](https://investors.micron.com/news/press-release/2022/Micron-Ships-Worlds-First-232-Layer-NAND-Extends-Technology-Leadership-07-26-2022/default.aspx) 確認每 plane independent read 確實存在於特定產品；這也不能直接移植為本專案四-plane chip 的能力。

完整依據與獨立審核見 [架構研究文件](/home/ericyu/expert_flash_experiment/docs/plane_wave_architecture_review.md)。

## 4. 微型與合成測試：何時真的有用

以下只有單 chip 的 flash/ECC 時間，不含 host、向量傳輸與控制成本。每個案例的權重 bytes 完全相同。

| 情境 | Baseline 理想界 μs | Pair 理想界 μs | Baseline 普通-read排程 μs | Pair 普通-read排程 μs |
|---|---:|---:|---:|---:|
| 兩條各 2 pages、互補 pairs | 56.0 | 28.0 | 66.24 | 38.24 |
| 兩條各 2 pages、同一 pair | 56.0 | 56.0 | 66.24 | 66.24 |
| 1 page 加 3 pages | 56.0 | 37.7 | 66.24 | 66.24 |
| 單條 5 pages | 37.7 | 47.4 | 68.80 | 96.80 |

兩個 32 KiB streams 互補時，useful slot fill 從 50% 到 100%，理想 flash 時間減半。但 `1+3 pages` 在固定 pairs 中仍要兩個 waves，無法任意視為一次滿 wave；單一 80 KiB stream 改用 pair 反而變慢。

另對 **32 個固定儲存 experts**、均勻選擇 `K={1,2,4,8,16}`，以多變量超幾何分布精確計算固定 bins 的碰撞，不使用抽樣或動態配置。涵蓋 11 種 shard 大小、三種 width，共 165 個格點。

以 32 KiB shard 為例：

| 每次選中 experts 數 K | Baseline 理想 μs | Pair 理想 μs | Single 理想 μs |
|---|---:|---:|---:|
| 1 | 28.000 | 28.000 | 37.700 |
| 2 | 56.000 | 41.548 | 46.213 |
| 8 | 224.000 | 138.453 | 125.049 |

只有一條 stream 時，沒有其他有用工作可填空位；較多 streams 同時 ready，才有跨 expert 利用空 planes 的機會。不能預設實際 routing 會平均落在各 bin。

![不同 shard 大小與共同啟用數的精確合成測試](/home/ericyu/expert_flash_experiment/results/plane-wave-v1/figures/synthetic_size_sweep.png)

原始輸出：[24 組微型測試](/home/ericyu/expert_flash_experiment/results/plane-wave-v1/microbench/results.csv)、[165 格合成測試](/home/ericyu/expert_flash_experiment/results/plane-wave-v1/synthetic/results.csv)。

## 5. 真實 routing 的資料與公平比較

沿用既有五模型、同一組 300 題：每模型 200 題 training、100 題 held-out，每題 128 次 decode；前 32 步另列，使用相同固定配置。總計 held-out 有 64,000 個 prompt-model decode inputs、1,868,800 筆 layer routing events。

掃描 `g={1,2,4,8,16}` 與 `width={4,2,1}`。每個 g 沿用舊 training-only expert→chip mapping，只額外學 plane bins；不是重新訓練模型，也不改 router 選擇。配置與 SHA-256 先落盤，之後才載入 held-out cache。共同啟用頻率、g、width、bundle 大小都不依 held-out 結果選擇。

所有表格包含原先的 250 ns／matrix-block command、2 μs／host phase、INT8 input、FP32 output、8 GB/s PCIe 與 2 GB/s/channel controller fanout。信賴區間採 **整個 prompt 為單位、配對 bootstrap 2,000 次**；不把同題 128 步當成獨立樣本。

這是既有本地合成題庫的真實 llama.cpp routing，結果不代表所有輸入分布。新時序未重播到舊 NVMeVirt kernel；舊 kernel 不具備本次多-plane contexts 排程。

## 6. 固定 g=16：plane 分工可以改善，但不是所有模型都受益

以下皆為完整 128 步、100 題 held-out 的 `ideal_overlap` routed-FFN **ms／decode input**。改善為 `1 − 候選延遲／同 g=16 的四-plane延遲`；負值代表變慢。

| 模型 | Width 4 基線 | Width 2 | 改善與 95% CI | Width 1 | 改善與 95% CI |
|---|---:|---:|---|---:|---|
| Granite | 12.7035 | 8.9923 | **29.21% [29.12, 29.31]** | 8.1993 | **35.46% [35.26, 35.66]** |
| OLMoE | 13.6484 | 12.8890 | 5.56% [5.44, 5.70] | 14.4544 | −5.91% [−6.11, −5.70] |
| DeepSeek | 21.3327 | 21.1473 | 0.87% [0.66, 1.07] | 24.7843 | −16.18% [−16.61, −15.76] |
| Qwen3 | 37.0237 | 32.4334 | 12.40% [12.26, 12.53] | 35.8227 | 3.24% [3.00, 3.48] |
| Mixtral | 112.8376 | 144.2677 | −27.85% [−28.34, −27.37] | 229.0169 | −102.96% [−103.56, −102.40] |

Granite 的 g=16 gate/up shard 是四 pages，down shard 是兩 pages。合計 useful slot fill 從 width 4 的 **75.00%**，變成 width 2 的 **89.58%**；width 1 則為 **72.83%**，延遲卻更低。原因是 slot fill 以波次計算，並沒有包含 cold-start 重疊與每條 stream 連續讀取時間。**不能只用「一波有多滿」決定最快配置。**

OLMoE 的 g=16 基線原本就有 100% slot fill，pair 仍改善 5.56%；它的收益來自多條 streams 的 cold-start／服務重疊。Mixtral shard 很大、每次只選兩個 experts，縮窄 plane striping 會延長單條 stream，碰到同一 bin 時代價尤其大。

![固定 g16 的延遲及有效 slot fill](/home/ericyu/expert_flash_experiment/results/plane-wave-v1/figures/fixed_g16.png)

Granite g=16 的成本分解顯示，本次沒有偷偷減少向量或控制流量：

| 分量，ms/input | Width 4 | Width 2 | Width 1 |
|---|---:|---:|---:|
| Flash/ECC 理想界 | 10.7520 | 7.0408 | 6.2478 |
| Matrix-block 控制 | 1.5360 | 1.5360 | 1.5360 |
| Host phases | 0.0960 | 0.0960 | 0.0960 |
| 輸入傳輸 | 0.1229 | 0.1229 | 0.1229 |
| 輸出傳輸 | 0.1966 | 0.1966 | 0.1966 |

完整 prefix32／full128 結果與 CI 見 [固定 g16 專項結果](/home/ericyu/expert_flash_experiment/results/plane-wave-v1/fixed_g16_summary.json)。這是預先定義 grid 的逐格描述，沒有依測試表重新選部署配置。

## 7. 和原本已選好 g 的最佳配置比較：這次沒有更快

不能只和較弱的 g=16 基線比較。對每一種 width，在 training 上重新選 g，再看 held-out：

| 模型 | 舊最佳 g／width | 舊最佳 ms/input | 最佳 width 2：g／ms | 相對舊最佳改善 | 最佳 width 1：g／ms | 相對舊最佳改善 |
|---|---|---:|---|---:|---|---:|
| Granite | 4／4 | 6.8712 | 8／7.3407 | −6.83% | 8／7.9068 | −15.07% |
| OLMoE | 8／4 | 12.2430 | 16／12.8890 | −5.28% | 16／14.4544 | −18.06% |
| DeepSeek | 8／4 | 20.3131 | 16／21.1473 | −4.11% | 16／24.7843 | −22.01% |
| Qwen3 | 8／4 | 30.5421 | 16／32.4334 | −6.19% | 16／35.8227 | −17.29% |
| Mixtral | 16／4 | 112.8376 | 16／144.2677 | −27.85% | 16／229.0169 | −102.96% |

三種 width 與所有 g 一起選時，五模型均選回舊的四-plane配置，held-out 改善為 0%，配對 CI 為 [0,0]。

舊最佳配置的 useful slot fill 為 Granite／OLMoE／Qwen3／Mixtral **100%**、DeepSeek **97.06%**。這解釋了尾波填補剩下的空間有限；但不能僅據此證明 plane 並行永遠無效，因為 cold-start 重疊仍可能有收益。本次負結果是窄 stripe 的單 stream 時間、固定 bins 的共同啟用不均與控制成本共同造成。

在 page 對齊且 shard 能均分時，部分 `較大 g＋較窄 width` 方案，會產生近似 `較小 g＋四-plane` 的每-plane 讀取長度。前者通常需要更多 chip descriptors，並共用 chip 內 ECC/PE。這是本次現象的機制解釋，不是所有維度或硬體條件下的支配定理。

![相對既有最佳配置的延遲增加](/home/ericyu/expert_flash_experiment/results/plane-wave-v1/figures/best_group_comparison.png)

完整 CI、兩個 decode windows、gate/up 與 down 分解見 [summary.json](/home/ericyu/expert_flash_experiment/results/plane-wave-v1/summary.json)；所有 g／width／時序的 450 格 held-out 結果見 [heldout_candidates.json](/home/ericyu/expert_flash_experiment/results/plane-wave-v1/heldout_candidates.json)。

## 8. 單一 context 的 supermatrix：少啟動，卻讀了更多無用資料

在各模型舊最佳 g 上使用 training 學得的固定 bundles，仍沿用理想管線模型：

| 模型 | 不合併 ms/input | B=2 ms/input | B=2 權重讀取倍數 | B=4 ms/input | B=4 權重讀取倍數 |
|---|---:|---:|---:|---:|---:|
| Granite | 6.8712 | 8.1627 | 1.358× | 10.2874 | 1.990× |
| OLMoE | 12.2430 | 15.8489 | 1.421× | 21.7170 | 2.093× |
| DeepSeek | 20.3131 | 27.8243 | 1.486× | 40.8244 | 2.305× |
| Qwen3 | 30.5421 | 39.1863 | 1.453× | 54.3569 | 2.218× |
| Mixtral | 112.8376 | 166.5202 | 1.480× | 250.1035 | 2.229× |

讀取倍數的分母是原本選中 experts 的 gate/up/down useful bytes，分子包含 inactive bundle 成員的 gate/up 權重。輸出傳輸也完整計入；額外 host row-demux CPU 成本預設為零，因此表格甚至仍偏向 bundle。

training 在全部 g 與 `B={1,2,4}` 中選擇後，五模型均回到 `B=1`。例如 Granite B=2 的 pages 可以全部占滿 slots，但只有 **73.63%** 的 wave 容量是原本 routing 真正需要的權重；不能把 inactive 權重讀取當成有效利用率。

這個方法的數學與同-WL配置可成立，這組 routing 下的 overfetch 成本則大於減少 cold-start／命令的收益。沒有證據支持把所有 experts 一律綁成大 bundle。

## 9. 時序、額外成本與容量敏感度

### 9.1 cr-read 是否能維持，會改變結論

固定 Granite g=16 的完整 routed-FFN 延遲：

| 時序 | Width 4 | Width 2 | Width 1 |
|---|---:|---:|---:|
| 理想 overlap，ms | 12.7035 | 8.9923 | 8.1993 |
| Drain 間隔保留 CR，ms | 15.6526 | 11.9414 | 11.1484 |
| 每 wave 普通 read＋drain，ms | 15.6526 | 13.9026 | 15.9727 |

pair 在最後一列對同列 baseline 改善 **11.18%**，但它的 13.9026 ms 仍比舊理想基線 12.7035 ms 慢 **9.44%**。single 在普通-read排程中反而變慢。不同時序的 baseline 不能混用來宣稱已超越舊系統。

在普通-read模型中重新選 g／width，五模型仍選 width 4；其 g 選擇會改變。所有時序的數值都保留，沒有只展示有改善的那一欄。

### 9.2 新增 contexts 不一定便宜

在原先 250 ns／matrix-block 與 2 μs／phase 之外，額外掃描：每個額外同時使用的 context setup 為 `0/250/1000/5000/20000 ns`；每個 NAND lane-wave command 為 `0/50/250/1000 ns`。同一費率也對 baseline 收費。

固定 Granite g=16，新增 lane-command 成本先設零時：

| 每額外 context setup | Pair 延遲／改善 | Single 延遲／改善 |
|---|---|---|
| 0 μs | 8.9923 ms／29.21% | 8.1993 ms／35.46% |
| 0.25 μs | 9.1843 ms／27.70% | 8.7656 ms／31.00% |
| 1 μs | 9.7603 ms／23.17% | 10.4645 ms／17.62% |
| 5 μs | 12.8322 ms／−1.01% | 19.5254 ms／−53.70% |

這些 setup 值是公開的敏感度參數，不是硬體量測。對於已經比舊最佳配置慢的窄-plane候選，增加這類成本不會使其勝出。

### 9.3 固定配置確實能落到 block／WL，並核算 padding

plane 模型保存固定 page→plane／block／WL metadata，包含每 stream tail、block boundary padding 與最後 block 尾端空間。supermatrix 另產生 15 份壓縮的虛擬 NAND 地址計畫，使用現有四-plane `WaveAllocator`；同一 wave 的 pages 位於同一 block stripe／WL。

以下是 supermatrix plans 的實際 LSB 保留空間，含 boundary 與 final-block padding；單位 GiB：

| 模型與舊最佳 g | 唯一 INT8 權重 | B=1 保留 | B=2 保留 | B=4 保留 |
|---|---:|---:|---:|---:|
| Granite，4 | 1.1250 | 1.1250 | 1.1250 | 1.1250 |
| OLMoE，8 | 6.0000 | 6.0000 | 6.0000 | 6.0000 |
| DeepSeek，8 | 13.4063 | 14.0625 | 14.6250 | 14.6250 |
| Qwen3，8 | 27.0000 | 27.0000 | 27.0000 | 27.0000 |
| Mixtral，16 | 42.0000 | 60.1875 | 60.0000 | 48.0000 |

bundle 沒有增加唯一權重數量，但 stream 長度改變後，block packing 可能增加或減少保留空間。這也說明 runtime overfetch、wave 空位、page padding 與儲存容量是不同指標。

這些 plans 是**地址 metadata**，沒有實際燒錄 NAND。容量只涵蓋 routed FFN INT8 權重與指定配置的空間，不含 attention、router、shared experts、ECC parity、SSD spare／GC 保留；LSB 對應的保守 TLC cell raw reservation 為三倍。

## 10. 驗證、重現與交付內容

本次已完成：

- **46 項 unit tests 全部通過**，包括原模型回歸、固定位置、互補／衝突、partial pages、跨 block、context 不足、共享頻寬、實際 GEMV row 重建、bundle overfetch 與完整輸出流量。
- **675 組 scalar scheduler 交叉核對**；25 組全陣列 baseline 檢查，涵蓋五模型所有 g 與六個時間分量。
- 獨立重新計算五模型合計 **64,000 個 held-out inputs**，與持久化 baseline NPZ 的六個分量逐 bit 相等。
- 固定 g16 的 45 組 arrays、576,000 個 token predictions，以不呼叫新 cost 函式的獨立 occupancy 公式逐 bit 核對；專項 CLI 重跑也保持全部數值相同。
- 99 條 streams、27,539 pages 與既有 materializer 的 physical address 相等；150 個 selected subsets 與舊 `Demand` 時間相等。
- 固定 bins 的獨立超幾何 oracle；五模型 train/test IDs、hash、容量守恆與 training-only winner 驗證。
- 原始 JSONL 另外抽查 20 題、74,752 筆 routing events，與 128 步 NPZ cache 逐 bit 相等；並沿用原實驗全題 traced／untraced 輸出驗證。
- 全 15 份 bundle plans、17,602,560 pages 的區間不重疊與 row coverage 證明；其中再明列 128,640 個 physical addresses 驗證不重複。

主要重現指令：

```bash
cd /home/ericyu/expert_flash_experiment
python3 -m unittest discover -s tests -v

python3 scripts/plane_wave_model.py --output-dir results/plane-wave-v1/microbench
python3 scripts/run_plane_wave_study.py --swap-trials 256 --seed 20261030 --bootstrap 2000
python3 scripts/evaluate_fixed_g16_planes.py --results results/plane-wave-v1
python3 scripts/plane_wave_synthetic_sweep.py

python3 scripts/static_wave_bundle_study.py --trials 64
python3 scripts/materialize_static_bundles.py
python3 scripts/plot_plane_wave_study.py

python3 results/plane-wave-v1/audit_reference.py
python3 results/plane-wave-v1/audit_trace_provenance.py
python3 results/plane-wave-v1/audit_final_results.py
python3 results/plane-wave-v1/audit_physical_plans.py
python3 results/plane-wave-v1/audit_g16.py
```

需要本機既有 Python／NumPy／Matplotlib，以及原始資料根目錄 `/home/nfs_client/ericyu/expert_flash_experiment`。主要全語料 plane sweep 執行約 392 秒；這是分析程式的主機執行時間，和表中的 AiF 服務時間無關。若要保留本次 artifacts，重跑時使用各腳本的輸出目錄參數。

| 內容 | 檔案 |
|---|---|
| 核心固定 plane 配置與逐 plane timeline | [plane_wave_model.py](/home/ericyu/expert_flash_experiment/scripts/plane_wave_model.py) |
| 真實 routing 的學習、選擇與評估 | [run_plane_wave_study.py](/home/ericyu/expert_flash_experiment/scripts/run_plane_wave_study.py) |
| 固定 g16 專項 CI 與敏感度 | [evaluate_fixed_g16_planes.py](/home/ericyu/expert_flash_experiment/scripts/evaluate_fixed_g16_planes.py) |
| 固定單一 context bundles | [static_wave_bundle_study.py](/home/ericyu/expert_flash_experiment/scripts/static_wave_bundle_study.py) |
| Bundle NAND 地址與容量 | [physical_capacity.json](/home/ericyu/expert_flash_experiment/results/plane-wave-v1/static-bundles/physical_capacity.json) |
| 主實驗設定、code hash、corpus hash | [manifest.json](/home/ericyu/expert_flash_experiment/results/plane-wave-v1/manifest.json) |
| Bundle 設定與來源 hashes | [bundle manifest](/home/ericyu/expert_flash_experiment/results/plane-wave-v1/static-bundles/manifest.json) |
| 原始 cache／trace provenance 審核 | [audit_trace_provenance.json](/home/ericyu/expert_flash_experiment/results/plane-wave-v1/audit_trace_provenance.json) |
| 最終結果獨立審核 | [audit_final_results.json](/home/ericyu/expert_flash_experiment/results/plane-wave-v1/audit_final_results.json) |
| 物理地址計畫獨立審核 | [audit_physical_plans.json](/home/ericyu/expert_flash_experiment/results/plane-wave-v1/audit_physical_plans.json) |
| 固定 g16 的獨立公式與重跑證據 | [audit_g16.json](/home/ericyu/expert_flash_experiment/results/plane-wave-v1/audit_g16.json)、[reproduction check](/home/ericyu/expert_flash_experiment/results/plane-wave-v1/fixed_g16_reproduction_check.json) |
| 最終報告、程式與輸出完整清單 | [study_manifest.json](/home/ericyu/expert_flash_experiment/results/plane-wave-v1/study_manifest.json) |

## 11. 建議採用的方向與尚未回答的問題

若 chip group 必須固定為 g=16，短 shard 值得進一步實作固定兩-plane方案；在本次五模型中，它比 single 更穩定。Granite single 的理想結果雖更快，卻對額外 context 成本和 CR 連續性更敏感。部署前要先證明目標 NAND 的 plane 定址、context／累加器與連續讀取資料路徑。

若仍可自由選擇 chip group，目前應保留既有 training 選出的四-plane配置。這次的同-chip plane 改動沒有提供超越它的證據。把單一 context 的多-expert gate/up 一律綁成 bundle，也不適合這組 routing，因為 overfetch 已在最有利的零額外 demux 成本下造成明顯退化。

本次沒有搜尋每個 projection 各自的 width、混合 `{1,3}` plane masks、單一 expert 內 bulk 四-plane加固定尾頁區、跨 token batching 或帶副本的熱尾頁配置；也沒有把 host activation/combine、GC、功率、面積與額外 ECC/PE 微架構做成量測。這些方向需在固定位置、數值正確性、容量與相同 baseline 下另行比較，不能由本次結果推成「plane-aware 方法已無改善空間」。
