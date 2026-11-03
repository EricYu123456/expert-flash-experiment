# Shared Down 可移動配置：貢獻界線

本補充的具體系統問題是：共同輸入 GU 完成後，六個私有 Down 與 shared Down 競爭有限的 chip、PB、ECC 與 channel。固定 shared 專區會把可供 routed jobs 使用的 pair groups 減到六個；少量 expert 副本一旦缺 Hall matching，就要讓一個 pair 做兩個完整冷啟動 GEMV。新增配置把 shared Down 切成兩個完整列區段，兩者仍使用同一個完整 shared input，可移到 routed matching 留下的 pair。

這個觀察與可執行實體配置有研究價值，但其組合核心是既有 combinatorial batch codes（CBC）與標準增廣路徑 matching。不能把它們改名當作新演算法或新定理。以下一手文獻於本輪重新查讀：

- [Bhattacharya、Ruj、Roy，Combinatorial Batch Codes: A Lower Bound and Optimal Constructions](https://arxiv.org/abs/1102.4951)。研究不變資料副本、每 server 有限讀取及總儲存量；本文的 expert incidence／Hall 解碼屬於該既有框架。這個引用不表示其論文已處理 NAND PB 或完整 GEMV。
- [Silberstein、Gál，Optimal Combinatorial Batch Codes based on Block Designs](https://www.cs.utexas.edu/~panni/batch.pdf)。介紹 CBC 與 load balancing、支持集合的 Hall 條件，以及既有 `c=k−2` uniform constructions。八個 pair、top6、degree4 的建構必須列為強基準，不能宣稱發明了新的 code。
- [PlaceMoE，2026-10 版本](https://arxiv.org/html/2610.14798v1)。已聯合處理 expert 實體配置、副本與 token-to-copy 派送，其主要問題是分散訓練的階層通信及計算負載。不能把「聯合 placement 與 dispatch」本身當作新貢獻；該來源沒有為本實驗的 NAND／PB 時序提供證據。
- [Fast MoE Inference via Predictive Prefetching and Expert Replication，2026-05 版本](https://arxiv.org/html/2605.11537v1)。以預測熱 expert 和 GPU 副本處理多 token 壅塞；表明 expert replication 已是常用系統策略。其自述的效能數字未用於本專案比較或硬體參數。

先前的一手 AiF、Stratum、SLIM、HDA-MoE 稽核仍在 [asymmetric-down-prior](asymmetric-down-prior/notes.md)。它們已覆蓋 GU／Down 不同切分、共同輸入、packing 與 grouping；一般 IR 能表達一個 supplied layout，並不等於既有獨立 mapper 已找到相同可執行配置。

可投稿方向應收斂到可重用的實體配置編譯：從矩陣 shape、輸入相等關係、native GEMV 與 PB contract，產出完整 rows、plane/block 地址、副本支持、派送程式及可獨立驗證的證書。比較時須給 CBC、group replication、一般 row mapper 相同硬體與容量，並分別呈現有用 payload、實際 block 數、普通讀取和 CR 的效益。手工 G4 witness、既有 CBC 與一個條件式服務模型，尚不足以單獨宣稱投稿突破。

本建構是在第一批新自編診斷路由暴露 fallback 後提出；在同批路由上的後續數字只能稱診斷重用，不能重標成未見測試。組合上的任意 top6 保證可以由 Hall 證書支持，完整 FFN 時間仍要接受輸入／command／output 與實體硬體條件的限制。
