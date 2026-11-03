# AiF 普通讀取 MoE：2026-11-03 新研究

本次依使用者新授權，允許對 AiF 作小幅硬體改動。舊文件「只研究現有硬體配置」的限制屬舊任務，不能沿用來排除本次方向。目標是不依賴 CR-read、涵蓋有／無 shared expert 的 MoE，形成可投稿的方法與證據。舊研究結果均保留，不算本次輪數。

## 執行規則

- 最多十個新實驗輪次。思考階段淘汰方向不消耗輪數；只有已進入實驗的方法失敗，才進下一輪。成功候選留在原輪補齊證據，不為湊十輪故意判失敗。
- CLI、編譯、資料接線及實作錯誤修復後重試同一輪；不算科學實驗失敗。只有完成可判讀實驗並經黑臉審閱的科學負結果才推進輪次。
- 每個進入實驗的方向，先有至少 1,800 秒的實際研究、交叉質疑與修正紀錄；不能以平行代理時間相加、純等待、啟動時間或舊研究替代。
- 專案理解與工具接線單獨記錄。候選進入正式思考時記錄 UTC 開始，實驗前記錄結束、黑臉意見與未解假設。
- 思考者：Antigravity `gemini-4-argon-high`、`gpt-6.1-sol`／`xhigh`。只有 Gemini 額度耗盡時改 `gemini-3.8-flash-high`，保留原始錯誤；不啟用額外付費 credits。
- 資料研究：`gpt-6.1-terra`／`xhigh`；查論文、官方硬體資料、公開實作及專利，分清已讀全文與摘要。
- 黑臉：內建 `gpt-6.1-sol`／`max`。內建介面只有單一 reasoning 參數，沒有字面上的 `6.1sol-max-xhigh`；此差異已告知使用者。
- 實驗執行者：`gpt-6.1-sol`／`xhigh`，待思考與審查 gate 完成才啟動。

## 評價契約

方法必須列出原生 AiF 能力、所需小改動、buffer／運算／頻寬／容量／能耗成本及失效條件。shared expert 可以作效能情境，不能作機制成立的必要條件。CR-read 關閉是主實驗，不能默認其維持電位或連續讀取條件。

完整模型延遲、FFN 服務時間、必要讀取工作量、CPU 數值驗證與實體硬體證據分開報告。不得把 trace replay 當成實體 AiF 推論，或把 GGUF routing＋名義 INT8 成本當同格式端到端結果。若新量化／近似改變數值，須有相同品質／格式控制與真正推論品質評估。

每輪在讀取新保留資料前凍結方法、強基準、成本假設、資料切分、指標及失敗判準。公平基準必須獲得相同硬體能力、容量、等待窗與品質預算。有限搜尋失敗不等於全域不可能。

若整體延遲改善達 10% 但新穎性或必要性不足，另列「有工程效益、未具投稿價值」，不可當作已達論文目標。只有局部服務改善達 10% 時，也須明確標成局部結果。

## 專案理解的初步結論

既有 ordinary-read placement 已包含 projection-specific geometry、cyclic／overlapping striping、共同 GU／Down 配置、有限副本、合法 row 降階、同格式 W8 實體 byte 與數值驗證。OLMoE、Mixtral 在既有單請求、每個必要權重完整讀一次的模型中已達 NAND 工作量下界；純排列不能使必要 sensing 消失。

新研究必須改變必要讀取量、一次 sensing 可完成的有效工作，或能被完整模型量測支持的其他瓶頸。既有新穎性檢查與強基準不可丟棄；舊配置收益也不可重新歸因。

## 目前狀態

第一輪 gate-first certified A8-zero 經完整數值驗證後，新增可省原生頁數為零，判有界科學失敗；科學失敗數一，剩九次（含進行中的第二輪）。第二輪 pairbank 在至少半小時思考後被等價強基準淘汰，未消耗實驗機會。

目前第二輪研究 layer-shared input partition + local affine expert compiler。正式思考21:26 UTC開始，22:02凍結協定；Gemini4 Argon High四段實際CLI工作共1827.88秒，Sol與黑臉均超過半小時。Stage0及Stage1強對照均完成，兩次全比較族推論仍未定。Stage2唯一自然文本TRAIN16刷新與固定PG validation16檢驗已完成；八個算子的局部品質通過2%subset-PPL預算，完整性與統計經Root及黑臉覆核。Stage3固定非線性強對照及直接C++成本協定已凍結；rev003合成測試與BIND全部通過。BENCH前正窄修runner的跨session清理，數值來源與已閉合BIND保持固定；之後先量既有B/C/D，才決定GU_FIT。尚未放行Stage3 BENCH、GU_FIT或DEV。TEST保持封存。

這是必要條件的局部替換試驗，未量得全路由層替換後的品質、完整模型延遲或投稿價值。任何可信正結果仍須同輪比較更強的共同輸出子空間、同能力兩階段正則化、全host與W4控制，並驗證全覆蓋品質。所有正式時戳、輪次與模型調用收錄於 `results/research-2026-11-03-no-cr/`；狀態檔是即時進度依據。

第二輪 Stage0 已完成 FIT／VALID（2h12m52s）與 SCORE（23m17s，OS0）。唯一稀疏 decode 算子使全族 bootstrap 未定，黑臉與 Root 判同輪繼續，科學失敗數仍一。八個主算子的兩階段點估計均較 R1 低，但 decode 絕對誤差較高、學習曲線未收斂，不能視為完整品質或投稿成功。固定的 G-RRR／PE-RRR 強對照已於查看評分數值前凍結，Stage1 FIT／VALID（10m38.69s，OS0）產生160端點／560比較；Root獨立查核2,365個相依檔案雜湊全部吻合。舊SCORE重用屬開發比較，不能改稱新的確認性結果；TEST100保持封存。

Stage2的7個新來源檔與95個已列相依檔完成獨立雜湊審核；兩個模型共12個fresh contexts通過source/nohook／observer／copy／repeat以及canonical／copycanonical完整logits與路由trace位元控制。45.14秒是接線測試成本，不能當壓縮推論速度。

為避免合成提示的域外失敗誤算方法失敗，另凍結一次PG-19 train16校準：原FIT120與B的768-token prefix、B／舊C／舊D各最多32個自然EOG停止的decode states合併，只重建一次共享tree/P，局部方法與固定PE6/G2對照共用相同狀態／canonical targets，不加搜尋。之後才在固定validation16書籍窗口比較刷新後C與B的NLL，預設2%相對subset-PPL容忍值；此具體有限配置的有界失敗才可經黑臉與Root判下一輪。沒有宣稱所有guarded／重訓變體不可能。原TEST100與PG-19 test32均保持封存。

資料封存更正：Stage0的1,961個雜湊涵蓋所列輸出／相依檔案，未包含原始capture inputs／rows，不能宣稱當時已固定其內容身分。正式重用old FIT前，新增唯讀補綁審計：固定目前capture身分、逐op全量重算canonical輸出與歷史凍結target逐bit核對，前後確認內容穩定。這建立目前X與target的語義對應，不補造歷史X身分，也不增加訓練樣本／調參輪次。

公開硬體來源尚不能確認128-byte A8輸入、短列reset／FIFO與PE吞吐量；後續只可將它們列為待驗證硬體合約及敏感度條件，不可用ECC6.4GB/s代替運算速度。尚未量得任何完整模型≥10%延遲收益。

04:35:51 UTC補綁審計完成：8個算子共126,296列／258,654,208個F32元素全部bit相符，OS0、58.99秒；Root獨立核327個檔案（1.67GB）雜湊吻合。這筆943,740,420,096 integer MAC為共同完整性核對成本，沒有新增訓練樣本或重訓迭代。04:38:32 UTC依固定協定啟動PG-19 TRAIN16 acquisition與B／舊C／舊D狀態收集；FIT、validation與TEST仍分階段保留。

04:54:31 UTC TRAIN16收集完成（OS0，15m58.41s）：兩模型各48個fresh contexts，四層各13,824新列，所有自然decode皆未遇EOG而達32上限。Root核72files947.4MB全部吻合。唯一FIT於04:59:06完成（OS0，2m51.54s），新教師查詢110,592列／905,969,664 bytes的F32 target／826,244,333,568 integer MAC，完全共用於局部方法與固定PE6/G2。Root核470個相依檔案3.70GB吻合，八op均finite、0unsupported、B1/batch/archive/IPC通過；新tree/P與模型均封存。品質六來源檔之build/generated tests通過（152相依檔獨立hash吻合），等待正式來源審查，PG validation與TEST尚未開啟。

05:18:01 UTC固定PG-19 validation16品質檢驗完成（OS0，11m26.02s）：32,768個逐token NLL全有限，完整logits／route／live-alpha位元控制通過；Root核55個輸出雜湊、470個FIT依賴身分，並從閉合分數獨立重算bootstrap。C−B的同時95% NLL區間為Deep[-0.006839,0.012674]、OLM[-0.002222,0.011598]，兩個上界均低於ln1.02=0.019803；黑臉判為固定八op的局部品質通過，仍在第二輪，科學失敗數一。這不翻轉Stage0／1的未定判定，也不支持全層品質或延遲推論。完整模型routed slot實際替換僅約0.2825%／0.4560%；OLM單書較大退化保留，沒有剔除樣本。原TEST100與PGtest32保持封存。

後續先凍結相同TRAIN／P／tree的projected-GU＋完整SiLU＋原Down非線性強基準，以及直接C++、公平SIMD的host成本量測；已觀測validation16僅作開發比較。暫不啟動昂貴的全64教師查詢。若成本窗口仍有價值，下一個覆蓋品質候選是每模型最早MoE層全64：固定相同token history／位置／batch plan及fresh KV時，該層FFN不回饋自己的attention KV，能減少可避免的上游狀態偏移；仍須實際位元控制，且不適用自由生成歷史或跨rechunk。單層原始權重也可能全host，不能用這個局部品質階段證明flash必要性。尚未量得完整模型≥10%延遲收益，亦未判達投稿程度。

Stage3 rev003三個build、兩組generated數值測試全部OS0；Root獨立核490個source／build／generated相依檔206.7MB吻合，黑臉審數值順序與計時公平性通過。06:22:00→06:22:16 UTC完成BIND（15.80s，OS0）：698個新輸出167.4MB逐hash吻合，原470個FIT依賴SHA/stat前後一致；四層固定PGTRAIN32候選列、first4跨八op的B/C/D全輸出及private A8／自然與fullK結構cohort加權均bit一致。這是完整性成本，沒有GU production query、FIT、BENCH trial或新TRAIN states。BENCH源審發現nested new sessions可能逃過外層600s killpg，已先hold並授權只改run.py／synthetic.py生命週期監督與dummy清理測試；其餘五數值／模型來源保持逐byte不動。BIND成功結果保留，工具修復不計科學失敗。

rev006只修復程序監督，原五個數值來源與三個 binaries 逐 byte 不變，BIND 依身分橋接重用。06:40:19→06:41:08 UTC完成 BENCH_BCD（49.06秒，OS0）：四層共2,976個 raw trials／96 cases 全部保留，無刪除 outlier 或追加重測。Root 獨立核30個輸出3,695,916 bytes雜湊，並從全部raw重算分位數、31次輸入與mod3順序，均吻合。執行者確認1,168個FIT／BIND相依身分前後相同、資源限制零違規、程序零遺留。

這只量到研究機器上的CPU呼叫。自然partial的31個輸入大多沒有pilot命中；fullK使用兩個既有expert模板的實體獨立複本，只是資源形狀測量，不能視為全64模型或完整FFN品質／延遲。局部C在該CPU量測中較D快，但AiF sensing／PE／ECC／FIFO、普通load強基準與目標i9尚未閉合，不能推算完整模型≥10%收益。成本審查待續，GU_FIT／NL timing／DEV／TEST均保持關閉，同輪與失敗數不變。
