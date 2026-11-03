# 候選：以 amax 見證與上界證明 A8 零值，跳過 Up／Down

正式思考起點：2026-11-02 19:26:00 UTC（台北 03:26）。最早實驗時間：19:58:00 UTC（保守延長，涵蓋 Gemini 的專題加入時間）。時間只是必要條件；實際互審、合法數值契約與凍結計畫也須完成。這仍是第一個實驗輪次的候選，尚未開始實驗。

## 方法與必要性

先完整計算 gate，並計算少量 up witness rows。令 `a_i=SiLU(g_i)*u_i`，witness 給出真實 activation 最大絕對值的下界 `L>0`。對其他 up row，有確定上界 `|u_i|<=U_i`。若

`|SiLU(g_i)|*U_i < L/(2*127)`，

則不論其餘 activation 為何，canonical A8 的最終 `amax>=L`，該 row 一定 round 成零。使用嚴格小於以排除 tie。已證 row 的絕對 activation 小於 L，故也不會影響 amax；算完其餘 rows 可恢復與完整計算相同的 amax、scale 與 A8。

在此相同 W8A8 數值契約下，可以跳過已證 row 的 up dot product 及相應 down column。新機制若成立，刪除的是實際必要權重讀取，而非換 expert 排列。shared 可以照同一規則處理，但不是成立前提。

最初 `U_i` 採 row norm 與 input norm 的保守不等式；進階可用 partial-up 與剩餘係數界。但不能將取得 partial-up 或 bounds 的權重讀取當免費。既有 CPU 參考使用 F32 scale 與 `std::round`，不是自行假定的 RNE。

## 主要風險

- 高維 Cauchy bound 可能比實際 up 大約 `sqrt(H)`，僅能證少量極小 gate。
- witness 的動態 row 選取可能先碰遍所有 NAND 頁；必須按實際頁／wave與地址付費。
- scalar 上證零不等於能省 page。要整個原生讀取單位都不需要，或明示 SOML 等較小讀取粒度的硬體改動及衝突。
- 拆 gate/up 增加 phase、命令與 input reload；norm metadata、浮點保守 rounding、accumulator 與 down 的 column reduction 須算成本。
- exact 僅相對明示的 canonical W8A8；既有 W8 參考不等於原 GGUF 或 AiF 未公開的 arithmetic。

## 19:28–19:32 UTC：浮點契約與可收緊方向

原參考的 up dot 是 I64 累加，輸出為 `fl(float(dot)*fl(sw*sx))`。若預存每 row 的整數 `W2=sum(w8^2)`，本 token 有 `X2=sum(x8^2)`，則 `B=isqrt(W2*X2)` 是整數 dot 的確定絕對上界。乘積需防溢位，scale 必須 finite／非負。使用原函式對正 B 計算 U，藉單調與正負對稱可得到相同數值契約的 up magnitude bound。Block norms 可給更緊的和式上界，但 metadata 與 CPU 讀取隨 block 數增長。

證明時應使用原 GGML 產生的實際 gate activation h，不能另外呼叫不同 libm 重算 SiLU 後冒稱相同。先形成 `Ba=fl(abs(h)*U)`，對真正 F32 witness L 用 `sL=fl(L/127)`；要求 sL>0、相關值 finite，且 `fl(Ba/sL)<0.5f`，才保證原 `std::round` 給零。L=0、underflow／NaN／無法保守界定者 fallback，不判零。此處是推導，尚未執行樣本驗證。

若 checkpoint 本來有 up clamp，可以用原生 limit 收緊 U，不必人工改模型增加稀疏。官方 gpt-oss 還有線性項 +1、bias 與特殊 gate slope，其 up factor bound 是 8，而非直接套無 bias SwiGLU 的 7；DeepSeek-V4 亦須按實際 checkpoint 讀取 clamp。現有本專案模型主要是無此 clamp 的 G4／DeepSeek-V2／OLMoE；未因這個構想下載或測試新模型。

## 新收益的歸因

強基準已可在完整 Up 算完後跳過真正 A8-zero 的 Down；本候選不能把這項 Down sparsity 重算成新增貢獻。若額外略過比例 p 的 Up，省下 p/3 的**原 dense 三投影係數工作量**；這不是相對較強 sparse-Down 基準的相對延遲上界。

在三投影相同密集成本的理想模型中，若基準 Down 已省比例 d，基準成本為 `3-d`，候選成本為 `3-d-p`，相對降低是 `p/(3-d)`，speedup 是 `(3-d)/(3-d-p)`。若先讀一半 Up 做界，新增省量須按實際未讀半部計算，不能只按 certified row 數。Witnesses、norm metadata、頁面浪費、CPU 與多 phase 另計。真實 NAND 需要按 per-chip waves／critical path 分析，scalar 比例不是完整模型時間或實測。

## 直接先例與強控制

CATS 已有 gate-first 與跳過 up/down，但使用近似閾值。FloE 已在 MoE 做 activation sparsity 的近似選取。SLIM／Ripple 等也覆蓋 sparse dataflow 與重排。待證差異是以 witness amax 下界與確定 up 上界，消除未知 quantization scale 的循環，並在真實 NAND 粒度下保留收益；不能主張首次 gate-first。

強控制須含原完整 W8A8、所有 up 算完後才 skip exact A8-zero down、同讀取粒度的普通 sparse GEMV、相同品質設定的 CATS／FloE 類方法，以及相同 metadata／seed page 預算的簡單判準。若新增 subpage／多 context 能力，所有基準同享。

## 尚未執行的實驗構想

先以凍結的 TRAIN-only 小樣本量測 certificate coverage、seed 成本與真正可省的頁／waves。若連樂觀但付費的必要工作界也很小，停止重型實作並將第一個實驗判為失敗；若有空間，才擴成完整數值等價與新資料服務／品質比較。不能在 30 分鐘前執行 activation／norm probe，不能把既有結果當新實驗。

## 19:54–19:58 UTC：最終反證範圍與實作邊界

Root／Sol／Terra／黑臉逐項凍結：以decode pooled Up weight-byte比例衡量，不平均不同shared/dense/routed大小之ratio；logical每expert頁對齊僅是機會上界。支付固定首16KiB witness時，即使seed的h==0也讀，必須與gate0控制各自完整重算mask差。以免費真amax及最強事前界仍≤1% Up bytes且無完整16KiB新增免讀（兩模型與last4 deflation都如此）才形成此數值／bound／樣本下的決定性性能負結果；其他結果續同輪。1%相對等成本三投影僅是≤0.334%原dense coefficient-work，非native延遲界。

條件界使用signed128的正確ceil/floor整除與checked exact-isqrt，G=0及異常Gram fallback。Coordinate-deflation取abs(exact selected-coordinate sum)+residual integerL2，再與global界取min；不可把被忽略座標的讀取當免費部署。F32上界必須沿canonical scale_output與actual full-I GGML h計算；證零後保持q8/amax/scalebits，並不宣稱未量化F32 activation值不變。這些均尚為設計／source核對，沒有候選activation或weight統計。

Gemini3.3/3.4的新增15%/8%/5%門檻、G4/重排/動態witness擴張與「所有prior不能省Up」過廣敘述不採納，已要求其在獨立筆記勘誤；Root凍結協定為執行依據。
