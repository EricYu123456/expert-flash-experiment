# AiF 普通讀取 MoE：2026-11-03 新研究

> 已依使用者「直接停止所有工作，最後push到github」停止研究；僅保存與推送最後快照。兩次實驗失敗、剩八次，第三輪 B 未實驗，十輪尚未完成。停止收據：`results/research-2026-11-03-no-cr/USER_REQUESTED_STOP.final.json`。


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

第一輪gate-first certified A8-zero新增可省原生頁數為零，判有界科學失敗。第二輪pairbank在思考時被等價強基準淘汰，不消耗機會；後續local-affine first64-v1固定配置在完整早層必要品質檢驗中，OLMoE明確超過預定2%容忍值，經Root／黑臉判有界科學失敗，DeepSeek保留未定。累計兩次實驗失敗，剩八次（含第三輪），實際已進入實驗的方法數仍為二。現在第三輪僅宏觀思考與來源研究，候選尚未選定，沒有新實驗授權。

第二輪研究紀錄： layer-shared input partition + local affine expert compiler。正式思考21:26 UTC開始，22:02凍結協定；Gemini4 Argon High四段實際CLI工作共1827.88秒，Sol與黑臉均超過半小時。Stage0／1全比較族推論未定。Stage2刷新後固定八op局部C與Stage3固定NL8強對照皆通過各自2%subset-PPL品質門檻；兩組CPU成本量測及數值／統計覆核已完成。這不表示C與NL品質相等。earliest64協定已凍結：每模型最早MoE層全64個experts，先驗固定新124個C配方、舊四個early C不重擬合，另用未讀的validation16必要品質檢驗。Stage4 rev002來源／合成測試、BIND與有限preflight皆通過Root／黑臉審查，09:36:39 UTC固定124個新experts FIT已OS0完成，Root／執行者／黑臉artifact closure均通過；固定16本新驗證資料取得與native分詞已OS0完成並通過Root／黑臉核對，已另放行QUALITY_C64-only，強控制仍保持關閉。原TEST100與PGtest32保持封存。

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

07:08:32→07:11:00 UTC依另行凍結來源審查完成唯一GU_FIT（148.47秒，OS0）：八op共236,888 expert rows，新增1,179,989,835,776 GU integer MAC／2,304,667,648 raw target bytes；一次cached GU預測73,749,364,736 FMA，原Down校準589,994,917,888 integer MAC。Root與執行者分別核934個新輸出6,583,381,331 bytes全部SHA/stat吻合，四層262,144個輸出元素與private activation/A8／slot-alpha數值檢查通過。額外archive/B1、direct CXX及I64審計逐項另列，未混入單次production預算。原C/D/P/tree/targets保持不動。

07:21:03→07:21:35 UTC完成BENCH_NL（31.91秒，OS0），保留992 raw trials／32 cases；Root獨立核30輸出1,282,412 bytes及全部分位數。FullK模板NL的warm中位數約323–336微秒、cold約1,019–1,123微秒。NL-only與先前BCD三arm輪替的快取歷史、執行時間不同，跨phase數字只描述各自定義條件，不能當matched-cache競速或推論整機加速。未新增重測或剔除尾端。

07:27:18→07:31:42 UTC完成固定NL8開發品質（264.41秒，OS0）：兩模型各24個實際fresh contexts／16本書，8,192個新NL token NLL全有限。第一本所有actual copy位元控制與歷史checksum／trace／NLL一致性通過；歷史檢查不等於歷史full-vector memcmp。Root與黑臉各自核20輸出12,028,821 bytes、逐token分數與scatter ledger，獨立重算固定bootstrap，得到NL−B simultaneous95 NLL區間Deep[-0.006789,0.012771]、OLM[-0.004648,0.009367]，相對subset-PPL上界約+1.285%／+0.941%，均低於2%容忍值。個別書超標仍保留。NL實際causal介入僅3,595／4,761 expert calls，0fallback，不能外推全覆蓋品質。參考harness峰值觀測RSS18.77GB屬載入原GGUF／覆寫原始work的研究成本，不是8GB部署或速度證據。

黑臉正式判固定NL8 development screen通過，同Round2續行、失敗數仍一；原Stage0／1未定與舊C8局部品質判定不變。兩個局部pass不能證明C與NL品質相等，也削弱「只有條件輸出斜率才能維持品質」的主張。尚無完整模型≥10%延遲正結果或投稿成功。AiF原生PE lane數仍未知；新16-lane/r128/RF/FIFO構造僅為有界source/theory假設，沒有RTL、綜合、PPA或模擬執行。

下一個first64-v1協定目前只有草案：Deep L1與OLM L0各64experts，舊四個早層C映射保持逐byte相同、後層舊映射不掛載，新124experts採固定eta0/lambda.01/C32/r128/all-x。先做必要品質，通過後再另行釋出同覆蓋D／NL／共享輸出基底強控制。預計新增30.079GB canonical Y／27.435T integer MAC，獨立64GB磁碟上限與32GB保留空間，不套用Stage3預算。已從剩餘34本validation metadata依原hash排序提出16本，正文未讀；另18本、PGtest32與TEST100保持封存。新source、target、fit與驗證均未因此自動釋出。

08:13 UTC first64-v1協定凍結（SHA32213b7b76cba5c18254f8bb13c7cfb8091c1db6c800468a541c4020e66c6b33），黑臉核draft002至frozen僅status/auth改變。只授權執行者新增七個隔離來源、build與合成控制；BIND／preflight／FIT／acquisition／quality仍各自關閉。Root從50筆原始validation metadata及已凍結TRAIN／舊validation排除集獨立重算，34候選的hash排序前16筆與協定的ID、generation、size、MD5、CRC逐項相同，餘18本保留；沒有開啟正文。

Stage4 rev002兩個CXX build與三組generated測試皆OS0，Root與黑臉獨立核285files107,855,063B與13parents全吻合，7current sources=snapshot、舊37source不變，512筆preflight選樣逐原metadata record/digest/order皆吻合。Peak observed aggregateRSS1.73GB，四種跨session清理fixtures通過。rev001未命中fallback測試分支屬工具fixture錯誤，完整保留；R2只改diagnostic預設與生成fixtures。08:46:25 UTC依BIND-only exact-source gate啟動兩早層全128canonicalexports／原TRAINunion／舊四C功能身分綁定，0教師查詢與0refit；尚未放行preflight或其後階段。

Stage4 BIND於08:47:09 UTC OS0完成（43.94s）：1550輸出1,481,782,713B全部SHA/stat吻合；128experts／384heads共956,301,312個I8係數與2,293,760B row scales，原union／old4映射／geometry不變。有限PREFLIGHT於08:53:30 UTC OS0完成（19.10s）：兩model各256的batch/archive/B1全H位元一致、固定4vector privatequant/GU/Down與獨立I64通過；4輸出4,700,193B與1550BIND前後身分吻合。額外integerDGEMM7,710,179,328與I64 controls68,119,296分帳，不能用含B1/IO的wall推productionrate或部署速度。09:00:29 UTC按R2 freeze e02ea754...與獨立Root／黑臉exactsource gate啟動FIT_C64，固定124新expert／3,671,764 rows／30,079,090,688 rawY bytes／27,434,763,681,792 productionintegerMAC。7200s／8e9 RSS／64GB增量與32GB保留為硬上限；新16本正文／quality／D-NL-SO／TEST仍關閉，失敗數不變。

09:04 UTC Gemini4 Argon High 的有限來源覆核回覆 RESOURCE_EXHAUSTED429／Individual quota reached（OS3，25.90秒）；原始 stream 與 exit receipt 保留。依使用者既定備援改用 Gemini3.8 Flash High，09:10:16→09:11:52 UTC OS0完成95.95秒的 source-only 意見，未啟用額外credits、未執行候選、未讀新資料，不計新方法思考clock或科學失敗。黑臉有限覆核保留 common-P／noCR 的系統問題，修正全域新穎性、最強ordinary基準、SO host-resident U、所有硬體不可能性與整個方法族否證等過強主張；不改進行中的固定FIT。詳見 round02-local-affine 的 gemini_SLIM_prior_and_native_value_review.md 與 critic_gemini_SLIM_native_value_review.20261103T091317Z.json。

來源泛化審核見 [共享投影適用範圍](moe_shared_projection_generality_terra.md)：已核對的標準 routed FFN 可共用同 token layer input，但實際 alpha／shared gate／求和順序須按各模型保留，不能假設權重總和為1。Soft/slot MoE 與具 K/V 狀態的 attention experts 另有機制邊界。固定C32/r128 dense W8／每expert每leaf獨立儲存時，V3 maps約406GiB，超過1TB TLC理想LSB-only約310.4GiB；這是具體格式容量限制，非方法族不可能性或品質／延遲實測。H2048兩模型的結果不得外推所有MoE。

OLM原mixed-Q4_K_M全host基準的source/closed-resource帳見 `round02-local-affine/olmoe_original_w4_host_feasibility.sol.md`。4,213,511,776B GGUF加context1024的F16 KV134,217,728B與128logits25,755,648B共4.373485GB；若全檔來源頁與logged3006MiB repack同resident，則為7.525505GB，尚欠workspace/runtime等。mmap跨度不等RSS，file-backed頁與pagecache不能雙計。既有OLM quality harness觀測8.676352GB含multiarm/IPC/worker，不能據此宣判source-only原W4不fit；其isolated完整峰值與速度仍未量，仍是必要強基準。沒有新增候選執行或削弱baseline設定。

09:00:29→09:36:39 UTC Stage4固定FIT完成（wrapper2169.55s、phase2169.11s，OS0）。124新expert各一次完整productionY與一次FIT，共3,671,764 rows、30,079,090,688 rawF32 bytes、27,434,763,681,792 productionintegerMAC；0reuse／skip／unsupported，全部32leaf與pooled rank128。額外C實際bias校準962,530,902,016 integerMAC；archive/private/B1/IPC的19,438,501,888 integerMAC與獨立I64等另列；F64responsecross962,530,902,016與functional+diagnostic solves823,828,414,464是source-derived工作量，非實測硬體吞吐。全部研究程序已退出、10,678 samples零資源違規、觀測aggregateRSS峰1,296,179,200B（sampled下界，非部署峰值證明）。

622新輸出31,187,455,563B已封存；registrySHA8b8515274c31e4d2a6aa16a6c8c5e5c80e42972d20e4c803a7eb723e192a871b。Root獨立核merged2482檔32,857,536,487B SHA/stat、128local logicalarrayhash/finite scale/bias及124Y全部7,519,772,672F32元素finite通過，old4與共用geometry不變；未重算教師／預測或候選。執行者formalclosure b86395b5df97084d711f9e589a0cef0a50ebb7e29862abc1d02bd71ccbf51921 另列其含old4歷史label的scope，兩者檔案集合不同，不能混當同一數量。此時仍未讀新validation正文或開始QUALITY；FIT完整性不是品質／速度或投稿結果。

黑臉正式FIT closure收據699e8c371a1736d66c6bf6e9e8cc3e99a2c2709c69b05f5458db3dfb52f7eeed通過，無blocker；獨立closed-hash範圍與Root不同，分別保存各自檔案集合。Root據此另發ACQUIRE_FRESH16_TOKENIZE-only gate e72032ab95a71dbfe9f955f71361048f52185ef3caa74725d6749387c5246473：exact frozen16 GCS objects/raw共5,881,269B，先核generation/size/MD5和strictUTF8/window，再model-specific vocab-only native joint512 IDs。不做model forward／quality／新target／fit，不補書或改window。其餘18validation bodies與PGtest32／原TEST100仍封存；需acquisition/token closure後再獨立QUALITY gate。

09:44:03→09:46:19 UTC ACQUIRE_FRESH16_TOKENIZE完成（wrapper135.45s、phase135.02s，OS0），16raw共5,881,269B／48outputs9,115,225B，兩model各16 nativevocab-only jobs皆512IDs，0forward／target／fit。Root獨立重算全部rawMD5/SHA／pinnedgeneration/header/URL、strictUTF8 Unicode窗口／suffixbytes及兩組token metadata/order，0mismatch；資源samplepeak216,731,648B、0violations／ownedgone。Root重用既有FIT fullSHA＋currentstat bridge，actualphase完整前後hash保持相同；黑臉另做其currentSHA集合及窗口核對，不能補造prelaunch歷史stat。新16正文現已暴露，餘18與PGtest32／原TEST100仍封存。

ACQUIRE closureSHAc2fe870d2cce767ab9b3178843a23e240d9aa90060e39c568b75d6a99256abef、Root auditSHAc192427a17a5f47d5a834a987b2e59baa1c22b1ec4edea1f9387ebc300ab7b59、critic reviewSHA1d99522fcd2f7c0c3cf2a8649282099647e9bdb8e20e064c2a28becc8682311c。據既有source預審與這次closure，Root另發QUALITY-only gate6ff41dfba2503546887ed854ae22bb99e436905d054370a0732de1e25a278ed1，沿用全部凍結方法／GT512／52contexts／2endpoint統計，不改配方或選資料。32e9 RSS是reference-harness研究預算，不是8GB部署證據；3600s是整phase checkpoint，工具／完整性或區間未定不算方法失敗。後續D／NL／SO、自由生成、nativehardware／sim、TEST和自動push仍關閉。

09:51:53→10:06:51 UTC QUALITY_C64完成（wrapper897.82s、phase897.41s，OS0）。兩model各16books／52ownKVcontexts，24,576個保存source/B/C NLL皆finite；全512earliestX/orderedIDs/finalactualalpha位元控制、firstbook fullvocab/copy與typedall64scatter皆通過。C實際Down替換Deep49,152rows、OLM65,536rows，0fallback；其中pos511未用於任何scoredNLL，不能混算因果介入。兩model64experts皆有實際命中，未聲稱每expert-leaf pair或全層覆蓋。20outputs24,763,862B SHA/stat與1574saved依賴前後一致。RSS觀測19.956GB屬reference研究成本，非部署或速度。

Root與黑臉獨立重算twoendpoint/16pairedbooks/10000shared draws/seed2026110304/centeredmaxabsT95。Deep C-B NLL mean.0419045994，sim95[-.0026040959,.0864132947]跨2%界線，保留未定。OLM mean.1468220056，sim95[.1009434763,.1927005348]；相對subset-PPL+15.8148%，區間[+10.6214%,+21.2520%]，全部16books均超門檻。OLM B-source描述性mean+.0010122173，個別範圍[-.007975,.011776]，不能解釋C的大幅退化，未追加source equivalence測試。沒有丟outlier、改rank/guard/data或重抽。

Root final adjudication459c4a96ef6731d41317288ab733757714c2a5a52b32fb701ac20763af2efb21與critic08d4367d1ea6d027a744da31563e4b60113239f02fa088f8eddf797afa0f9eb0正式記第二輪fixedfirst64-v1／有限all-x校準／2%GT窗口預算失敗。這不否定所有local-affine／commonP家族，不翻轉Stage0／1未定或舊C8／NL8局部development passes。必要配置失敗後停止延後D64／NL64／SO64，不把未跑的控制宣告失敗。第三輪charter已凍結為source/thinking-only，採納候選後另計至少1800s真實思考與黑臉gate，尚無candidate/experiment release。PGtest32／原TEST100／餘18validation正文仍封存；無完整模型≥10%延遲工程正結果或投稿成功。


### 第三輪思考：原生混合格式 LUT 視窗提前淘汰

原生 Q4_K/Q6_K、有限輸入查表視窗方向經 Sol 與黑臉獨立比對後，在思考階段因投稿價值不足淘汰：同格式、頁面、ECC、256-input frontier 與數值契約給直接低位元 MAC 後，尚無可指認的新 NAND 必要工作節省；剩餘核心屬已知 LUT 硬體資源比較。這不表示所有 LUT 不可行，也不以未知 PDK 或 Gemini 的錯誤數字當反證。Sol 實質1139秒、Gemini完整回覆142.288907秒，均未宣稱通過1800秒門檻；沒有實驗、不消耗機會，維持已失敗2次、剩8次。記錄：`round03-thinking/Root_native_mixed_format_thought_veto.json`。後續回到宏觀探索，新候選另起時鐘。


### 使用者要求停止

研究、原生子代理與進行中的 Gemini CLI 鏈已停止。B 的 Sol 實質思考1817秒；Gemini七段完整回覆合計803.732976932秒，第八段被中止、完整工作計時0秒，後續兩段只有預寫提示而未啟動。B 未通過雙方1800秒門檻，沒有實作或實驗釋出。保留所有已完成與中斷紀錄，不把停止、工具中斷或思考淘汰當作實驗失敗。此後僅整理 GitHub 最後快照，不繼續研究。
