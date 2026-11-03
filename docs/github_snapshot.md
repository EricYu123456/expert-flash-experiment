# GitHub 研究版本保存範圍

本次保存截至使用者於 2026-11-03 要求停止時的專案狀態，包含第二輪 Stage4 完整品質驗證、失敗判定，以及第三輪來源審查與未完成的思考。累計已開始兩輪、確認科學失敗兩輪，剩八次機會；第三輪 B 未進入實驗，十輪尚未完成。所有研究已停止，僅執行這次最後保存與推送；完整模型延遲收益與投稿價值尚未證實。

程式、研究筆記、協定、審查紀錄、適合版控的結果與圖表納入 Git。新增的 no-CR 結果單檔上限為 2 MiB，其他新增結果為 256 KiB，圖表與 Markdown 為 2 MiB；既有受 Git 追蹤的檔案全部保留。下載的第三方來源與資料、原始 traces、張量、編譯產物、重複 placement plans 和超過上述大小的中間結果保留在本機，不刪除、不改寫。

這是可檢閱的研究版本，**不是完整的原始產物備份，也不是從 Git clone 即可重播所有歷史實驗的自足套件**。現有收據可能引用研究機器的絕對路徑，完整數值重播仍需那些本機／NFS 產物、模型及指定工具鏈。

- [納入清單與 SHA-256](github_snapshot_manifest.json)：列出版本中的檔案與內容身分；清單自身不計 hash，避免循環。
- [保留本機清單](github_snapshot_omitted.jsonl.gz)：gzip 壓縮的逐檔 JSONL，含相對路徑、bytes 與原因。只讀檔案屬性，不開啟被排除內容；原本已被 `.gitignore` 忽略的模型等檔案不在此清單。
- [保存工具](../scripts/prepare_github_snapshot.py)：重新產生清單與 `.git/publication_snapshot_paths.zlist`；不執行模型或研究實驗。先檢閱清單，再使用 Git 的 `--pathspec-from-file` 明確加入檔案。

Stage3 的 [執行收據](../results/research-2026-11-03-no-cr/round02-local-affine/stage3-control-cost/BENCH_BCD.executor_closure.json)、[Root 輸出查核](../results/research-2026-11-03-no-cr/round02-local-affine/stage3-control-cost/Root_BENCH_BCD_closed_output_audit.json)與 [96 組描述性計時](../results/research-2026-11-03-no-cr/round02-local-affine/stage3-control-cost/BENCH_BCD.executor_timing_summary.csv)均保存。計時來自 EPYC 研究機器；完整 K 使用模板複本，天然 partial 多數零 pilot hit，均不能當成完整 FFN、i9 或 AiF 實測延遲。


最後狀態見 [停止收據](../results/research-2026-11-03-no-cr/USER_REQUESTED_STOP.final.json) 與 [即時狀態](../results/research-2026-11-03-no-cr/state.json)。大型模型、原始張量與資料快取依既有政策留在研究機器／NFS；沒有為了 GitHub 保存重跑研究或開封剩餘品質／TEST 資料。
