# 第一輪候選：合併 Down residual 誤差的 sketch 與頁面補讀

正式思考起點：2026-11-02 19:07:01 UTC（台北 2026-11-03 03:07:01）。最早實驗時間為 19:37:01 UTC；實際研究紀錄與黑臉審查也須完成，時間到不代表自動通過。

尚未進入實驗。若思考階段否決，本候選不消耗輪次。

**處置：2026-11-02 19:16:42 UTC 在思考階段淘汰原版，沒有開始實驗，不消耗第一輪。** 未達 30 分鐘，因此沒有資格進入實驗；淘汰方向不需要硬等或硬跑 probe。後續候選進實驗前另有完整思考時段。以下保留原始假說與逐步反駁，不刪去曾採用但後來修正的想法。

## 機制假說

完整計算 gate/up，形成各 expert 的完整 A8 activation 後，將原 W8 Down 權重精確拆成兩份 packed signed INT4：

`w = 16*h + 8 + r`，其中 `h=floor(w/16)`、`r=(w mod 16)-8`，兩者範圍均為 `[-8,7]`。

先讀高位部分，以 `16*dot(h,z)+8*sum(z)` 形成 base，再選擇讀取低位 residual `dot(r,z)`。全補讀時，在相同整數累加與 row-scale 契約下應還原原 W8 Down；未補讀時為明列的近似，不能稱 bit-exact。

在實體權重已知時預先儲存小矩陣 `S*diag(row_scale)*R_e`。每個真實 token 用完整 activation `z_e` 與 router 權重 `alpha_e` 計算 residual output 的 sketch。選補讀集合時評估所有未補讀 experts 的**合併**誤差，而非只按單一 expert 的 router score、權重 norm 或各自誤差相加。shared 可加入同一誤差帳，但不是成立前提。

## 必須先解決的問題

1. 同一 sketch 選擇補讀集合又自行驗證可能過度樂觀。獨立 audit sketch 的成本與保證範圍必須分開；跨 token／layer 的 adaptive inputs 也不可默認獨立。
2. 只改 Down，忽略其他成本時最多節省約六分之一的 FFN 權重讀取。完整模型改善 10% 並非自動成立，頁／四-plane wave 取整會再縮小空間。
3. sketch metadata 的實際讀取／常駐容量、host/controller MAC、selector、額外命令、row-scale 與 partial-I32 continuation 均須付費。
4. INT4 unpack、partial accumulation 與 scale ABI 是新增設計，不能當成 AiF 論文已證實的能力；最小硬體差額須具體。
5. 必須比較相同 packed hardware 的固定 4/8-bit、alpha／norm／variance 選精度，以及相同 error budget 的一般 mixed precision。精確 combined-error oracle 只作不可部署界。
6. 最新 channel selector／ternary sketch＋low-rank recovery（arXiv:2610.37386）須納入直接先例與品質／讀取成本比較，不能只對未壓縮 W8 宣稱新方法勝出。

## 實驗前判準草案

先判斷是否存在相對強控制的 error/read Pareto 空間，再規劃同格式兩類 MoE 的小型完整推論與品質試驗。不得在思考期偷跑 activation／residual 數值 probe。採樣資料、固定 seeds、比較基準、品質門檻及實體成本規則在正式實驗前凍結。若僅是既有量化帶來超過 10% 效益，獨立標記工程價值，不包裝成投稿突破。

## 19:09–19:12 UTC：先例與反例修正

Terra 找到 DPA-MoE（https://arxiv.org/html/2610.21378v1）與 DP-LLM（https://openreview.net/pdf/523b6f8097aa032b4da47977d854556f0113ad67.pdf）。前者有 live router score 與 residual profile 的逐 expert 精度分配；後者已用預算的 residual projection 在 runtime 估 matvec 量化誤差。兩者須作直接基準。本候選不能主張首次以 sketch 決定精度；剩餘待證價值只在**共同 projection 對合併 residual 的補讀選擇**，或其 NAND 執行上的必要新機制。

Sol 與 root 推導：若 H 維 residual 接近各向同性，真 cross cosine 的尺度約 `1/sqrt(H)`；m 維 random sketch 的 cross 估計噪聲約 `1/sqrt(m)`。m 遠小於 H 時，subset 搜尋可能在挑虛假抵銷，不能把 sketch 預測改善當真誤差改善。這是明示分布假設下的分析，不是對真模型的實驗結論。

若改為資料感知的共同低秩子空間 U，另有更強控制：相同 `U^T R_e` metadata 與 host MAC 已足以直接重建並加回該子空間的 residual。必須比較「直接低秩補償＋餘項的 diagonal refinement」，不能只比較不使用 metadata 的粗糙基準。尚未證明此控制逐項全面支配所有 nonorthogonal sketches，也不把它誇大成全域不可能。

廉價控制另加入 residual 的 row/column 均值補償，以及 `alpha_e^2 * residual_variance_e * ||z_e||^2`。若改善只是更準確地估本 token 單 expert 誤差，須與真正 cross-term 價值分開。

## 19:11 UTC：共同子空間控制的精確範圍

令 `q_e=alpha_e*diag(s_e)*R_e*z_e`，P=`U U^T` 為 orthogonal projector，F 是實際補讀集合。原候選未補讀誤差為 `d_F=sum_{e not in F} q_e`。如果相同 metadata 已提供 `c_e=U^T q_e`，則另付 `H*m` 個 MAC 即可加回 `U*sum_{e not in F} c_e`；新誤差為 `(I-P)*d_F`，其 L2 norm 對同一 F 不會大於原 norm。

這證明的是相同 subset、精確投影、L2 誤差下的控制優勢，**不是**零新增成本、全模型品質必定更好，或所有 random/nonorthogonal sketches 都被同式支配。新增 `H*m` MAC、metadata 量化與輸出精度必須列入。若全 residual 都補讀，未補讀和為零，仍可保留原 W8 fallback。

候選若改成「估計並控制 orthogonal tail」，便須重新證明 tail 有可捕捉的 input-dependent 結構，且勝過同 metadata 的補償與 variance/activation-norm 基準；不能把修正後的低秩補償直接稱原本 signed-sketch selector 已成立。
