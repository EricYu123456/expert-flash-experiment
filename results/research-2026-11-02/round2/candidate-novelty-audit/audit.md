# Ordinary-read whole-row compiler：投稿貢獻審核

仍有具體的系統問題可研究，但目前未證明已達投稿突破。最窄而可守的方向是：在完整 output row／完整 input／原始 scales 的契約下，將矩陣配置 lower 成合法 ordinary NAND commands，選擇 work／depth／command 的整數取捨，並輸出可獨立核對的數值、地址與成本證據。DP、不等長 sharding、joint mapping、流水與驗證本身均不能當首次貢獻。

## 直接能力矩陣

「未見」僅表示本輪讀到的方法未提供，不是不存在的證明。六軸細節、版本與實際閱讀範圍在 capability_matrix.json、sources.json。

| 方法 | 完整 output row／完整 dot | NAND 單 block／整數 wave | Work 與 depth | 強 cyclic 控制 | 數值／地址證據 | 成本適配 |
|---|---|---|---|---|---|---|
| PIMCOMP | AG 可分割 dot、累加 partials | Crossbar，不適用 | 已處理計算、載入、memory 取捨 | 未見 NAND widths | 有逐 instruction／ONNX verifier；未見 NAND 地址證書 | 有 profiler 回饋 |
| Fast-OverlaPIM | Loop／data-space 分割 | 主測 HBM bit-serial | 已聯合 readiness、overlap 與 mapping | 有 overlap／同預算控制，非 NAND cyclic | 有 temporal／spatial 模型；未見本題契約 | 可配置 timing／links |
| Stratum | GU 分割與 Down input 分割配對 | DRAM tiers，不適用 | 已處理因果、ownership 與 reduction | 非 NAND cyclic | 未獨立執行其數值模型 | In-house 系統模型 |
| Cambricon-LLM | Page 大小的 2D fragments | 有 NAND page atoms；非本題 whole-row jobs | 已有 aspect-ratio／傳輸／異構負載取捨 | 未見全部 cyclic widths | 未驗證本題 row/address 證書 | 明確計 read／compute／broadcast |
| TileLens | GPU K-fragments | 4KiB memory tiles，非 IFP commands | 已有 read amplification／straggler 分析 | 有 row／column／tile-shape 控制 | 明確地址公式與 descriptors；非 chip/block/WL | 有成本敏感度；完整系統含新 HW |
| AiF | 等分 dense GEMV；output ABI 未公開 | Block／offset／pages 與跨 block 拆 command 已有 | 未見本題 frontier | 等分，非全部 widths | 原評估用 dummy vectors | 有 published timing／emulation |
| RecFlash | Embedding／SLS，非 dense GEMV | Page cofetch／plane distribution 已有 | Utilization、command、dataout | Frequency／cache 控制 | 未驗證 dense-row 契約 | NAND 三階段成本 |
| 目前候選 | CPU W8 完整 rows 已證；native 契約待證 | 有 explicit lowering／地址 checker | 有 exact work-at-depth witness | 有 widths、去重及同預算；非全域 optimum | CPU 數值與虛擬地址已核；非實機 | 有 command／fanout／fringe 參數；未校準 |

PIMCOMP 的官方 README 已提供逐 instruction 與 ONNX Runtime 比較的 verifier，以及 pimsim-nn 介面。因此「compiler＋功能驗證＋成本模擬」的廣義組合已被直接涵蓋。TileLens 也有既有軟體路徑，不能只因其完整系統增添硬體就忽略其 layout／address compiler prior。

## 目前應使用的實驗幅度

以下是已封存且獨立審核的 W8 TRAIN selection，全部 MoE layers＋shared 的 FFN service；不是 heldout 或完整模型速度。

| 模型 | Input copies | Command | Whole 相對 cyclic 改善 | Allocated bytes 差異 |
|---|---:|---:|---:|---:|
| Deep | 1 | 250ns | 3.7520% | −5.0633% |
| Deep | 4 | 250ns | 3.7476% | −1.3158% |
| Deep | 1 | 5µs | 2.0952% | −1.3158% |
| Deep | 4 | 5µs | 2.0989% | −1.3158% |
| G4 | 1 | 250ns | 0.8320% | +1.5873% |
| G4 | 4 | 250ns | 0.1605% | +1.5873% |
| G4 | 1／4 | 5µs | 0% | 0% |

早期 Qwen external＋nominal I8 的約 9.2%，以及 G4 相對 phase-group 的約 4.3%，不能替代此處同格式、較強控制的結果。Deep 不含 raw layer0 dense prefix。G4 也需要同容量比較，才能主張 Pareto 改善。

## 尚可證偽的系統問題

1. **原生合法性是否真能成立。** CPU reference／地址 bitmap 無法驗證未知 PE output dtype、row scales、job reset、fringe 行為或 input fanout。若取得 published functional backend／driver contract，先用跨 page 的 tiny 已知 rows 核對完整 input、輸出與 scatter；若需要新 context、未公開 partial returns 或新量化／硬體，該 backend 必須退出 existing-HW 主張。
2. **是否超過簡單 wave-aware tiler。** 先在既有 W8 前三層資料比較 greedy max-rows-at-depth／wave water-filling 與現有 profile，對完整 variant arrays 去重，aliases 共用同一次 label fit。若最佳 profile 全可由簡單 tiler 得到、同預算 prior-style mapping 成本相同，就放棄演算法／搜尋新穎性，只保留具體契約、lowering 與系統證據。標準 ILP 可以當獨立 frontier oracle，不是新方法。
3. **效益是否存在於可證明的成本與容量範圍。** 先用現成 global ledgers／cost vectors 做同容量控制及 baseline fallback，不捏造 device budget，也不把 BFD 超額當 infeasibility 證明。若 documented controller／input 能力下效益消失，該 backend 尚無性能貢獻。

本輪沒有推薦新的實體配置假說。Plane 私有 inputs、PB replay、命令內 wave pause、FIFO 容量與 safe per-WL tR 都缺足夠公開契約；不能以新 buffer／context 或合成異質性補出收益。也不能因 generic declustering 可表示子問題，就否定整個具體系統的研究可能性。

有限閱讀中未找到完全相同的 ordinary-NAND full-row work/depth compiler＋此套數值／實體證書。但這不是 priority proof；未取得全文的 FlexLLM／DIAMoND 仍是限制。

## 最直接一手來源

- [PIMCOMP，TCAD 2025／v1](https://arxiv.org/html/2411.09159v1)：§IV–VII；[官方程式](https://github.com/sunxt99/PIMCOMP-NN)，本輪只讀 README，未執行。
- [Fast-OverlaPIM，v1](https://arxiv.org/html/2407.00604v1)：§IV-B/C/I/J/K、V-A。
- [Stratum，MICRO 2025](https://arxiv.org/html/2510.05245v1)：§4.1、Figs8–9、§5.2。
- [Cambricon-LLM，MICRO 2024](https://arxiv.org/html/2409.15654v1)：§IV、V-A/B；不是未取得全文的 FlexLLM。
- [TileLens，2026 v1 preprint](https://arxiv.org/html/2607.04031v1)：§3、4.2–4.4、§5；沒有驗證與論文對應的公開 artifact。
- [AiF，ISCA 2025](https://doi.org/10.1145/3695053.3731073)：§4.4、§5.2、footnotes8–9。
- [RecFlash，2026 v1／APCCAS 2025 extended](https://arxiv.org/html/2604.25338v1)：§III。v1 正式題名是 Frequency-Based Data Mapping。

完整 source read-depth、能力界線、目前證據與檔案 hashes 見同目錄 JSON。
