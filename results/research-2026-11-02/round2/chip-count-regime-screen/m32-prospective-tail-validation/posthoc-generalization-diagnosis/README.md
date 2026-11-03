這是 post-hoc development diagnosis，原 primary 負結果與三張圖均未更動，沒有 refit 或新 native 重播。已看過的這批資料不再稱新 heldout。完整且有效的診斷位於 v2/；v1 在最後 JSON 輸出時遇到 NumPy integer serialization 錯誤，原檔全保留，v2 只修 serialization 與輸出路徑，31 個已算好的 arrays 逐元素相同。

沒有「TRAIN 未見 expert」：64 個 expert 的 TRAIN 次數為 2–99。低頻（1–8 次）只有 E11/E26/E50/E52；324 個退步 queries 只有26個含低頻 expert。因此未見 expert／冷 expert 任意 mask 不是主要證據支持的解釋。

可直接重算的機制是未見過的共同啟用子集：free 在512個 TRAIN queries 全部為 peak10；dev 的 peak10/11/12 分別為239/263/10。所有444個 query×chip 的 >10-wave 事件都包含一組 TRAIN 共同出現次數為0的最小 X 子集（110 pairs、271 triples、63 quadruples），410組的每一個 expert 都曾在 TRAIN 出現至少9次。這是有限 trace 的確切 witness，不是人口分布或因果證明。

最差 q458 的 route=[3,55,62,17,37,14]；chip31 有 shared2 waves、四個 X experts [14,17,37,62]，所以6+2+4=12 waves，baseline 為10。子集[14,17,37]各自 TRAIN 次數26/54/56，但共同0次；native 退步62.231us。q13 在 shared1-wave chip18 有五個 X，造成12 waves，表示不能只歸咎於 shared3-wave chips。每個事件與全部512 queries 都已保存。

反例也保留：324個 native 退步中，36個 owned peak相同、48個反而較低；177個改善中53個 owned peak反而較高。q363 的 NAND與owned peaks均不增，native仍慢6.139us；q263含低頻E26但改善22.426us。必要work不能替代完整DAG成本。

固定 shared 各tier的平均X degree幾乎未變（TRAIN→dev）：tier1 1.9030→1.8789、tier2 1.0986→1.1009、tier3 0.46794→0.47070。聚合tier marginals不足以解釋尾端，新的共同啟用集中是更直接的witness。

相關結構以非定值chip的query-owned-work correlation matrix計算：effective dimension=(sum eigenvalues)^2/sum(eigenvalues^2)。free為18.22→13.76；common5.99→5.52、independent6.10→5.53、tail6.10→5.72。這不是獨立PE或統計獨立負載的數量。correlation>=0.90的正相關connected components為free31群、其餘9群，也僅是描述性分群。

experts.json/degrees.json提供全部頻率與G/D/X/O inventory；correlations.json含矩陣與定義；joint_activation_violations.json與witnesses.json提供小反例；all_query_records.jsonl.gz保留全部負例。tail fresh L0僅讀Boundary已封 secondary，原三圖的L0 cost/bound也逐值橋接，沒有搶寫或選勝者。
