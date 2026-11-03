# 黑臉審查紀錄：ordinary-read MoE AiF

研究開始：2026-11-02 18:58:18 UTC（Asia/Taipei 2026-11-03 02:58:18）。

任務：在第一輪至少 30 分鐘的研究期，先攻擊候選的必要性、新穎性、硬體可行性、強基準與端到端效益，不啟動候選實驗。研究時間以實際 UTC 記錄，閱讀／推導與實驗分開。本文只修改此新檔，不改舊證據。

## 18:58–18:59 UTC：舊研究的審查基線

已讀 README、2026-11-02 第一輪及第二輪前段，開始核對 AiF 原論文。尚未完整讀完所有 report，以下是目前已被舊證據支持的陷阱。

1. **效益分母不能偷換。** README 主表是 routed FFN projection 服務時間。後期 full-FFN 補 shared 與 activation/combine 邊界，但 attention/router/SSM、host/SSD overlap 與實際 decode wall time 仍未閉合。不能稱其為端到端 tokens/s。
2. **routing 來源和計算數值格式不同。** 原模型 Q4_K/Q6_K/Q8_0 的真實 routing 重播到名義 raw-I8 成本；Q8_0 每 32 權重還有 FP16 scale。單加 6.25% bytes 不足以修補 wave rounding。CPU W8A8 reference 是另一個明示數值契約，不自動證明 AiF 原生能力。
3. **必要工作量下界排除大量純放置方向。** OLMoE、Mixtral 在 ordinary read、one-copy、batch-one、完整權重讀取下已達 sensing 工作下界；純重新排列沒有通用 10% 的 NAND headroom。
4. **舊最佳配置是必比基準。** Granite 舊 g8 已有約 21% 對 g16 的收益。不能將已有收益當新方法。每 phase fitting 也可能輸固定 g16，不能自動稱學習方案最強。
5. **方法名稱不構成新穎性。** G4 的 12/16 overlapping striping 是 staggered striping 的特例；共選加權最大裝置工作量是 hypergraph declustering；副本、Hall matching、singleton storage bounds 有 replicated declustering/batch-code 先例。
6. **更細不一定更強。** g8 tile copies 被同容量 whole-expert control 逐 route 支配；688-copy 構造的真實 queries 被相同 locations 的一般 tile flow完全匹配。需在相同能力與預算下比較。
7. **原生契約未明示處要付設計費。** 不同 planes 輸入 x、page-buffer replay、raw-I32 partial outputs、scale/rounding、跨 expert context 都不是既有 timing API 足以證明的能力。使用者允許小幅硬體改動；這容許設計，但不能把改動當免費。
8. **控制和重新組合可吃掉小收益。** 命令成本 250 ns/chip-block、2 us/host phase未經實體校準；Granite 新投影配置約 361 ns/chip command 即 break-even。NVMeVirt 的實際 queue 不等於 flash PE校準。

## 投稿門檻（初稿）

- 對明確的 NAND 執行限制提出機制；先比較 AiF、Cambricon、NVLLM、NASiC及可得的 MoE/flash placement prior，不能以換 solver 或術語自稱首創。
- 無 shared 的 MoE也須有機制與量測收益；shared 只能作額外場景，不能是唯一收益來源。
- 普通讀取 28 us設定下有效；若靠 CR-read、連續WL保留電壓、free PB replay或不明示的multi-context，需明列為硬體設計與成本。
- 所有容量（useful/pages/wave slots/reserved blocks）、SRAM/context、ECC/PE吞吐、input/output/descriptor流量、host activation/combine與critical path要閉合。
- 先列最強簡單 baseline與same-budget control，再用凍結外部工作負載確認；trace replay只叫服務模型。數值格式契約與完整token路徑需可檢查。
- 將「>=10%局部服務收益但不具投稿價值」與「有獨立新穎性且有端到端價值」分開判定。門檻不是只看百分比。

已將以上初步結論在約18:59 UTC傳給 root與thinker_sol。下一步：讀原論文 GEMV介面、buffer與pipeline，及round2後段／round3的強基準反例。

## 19:03 UTC：本次 scope 與必要性判斷

已讀 `docs/research_2026_11_03_campaign.md`。**上述時段是 onboarding，不冒算正式候選的 1,800 秒。** 本次允許小幅硬體改動，舊研究的 existing-hardware-only 限制不適用；新 decoder、precision 或 accumulator 可以設計，但面積、吞吐、能耗與公平能力對照不能免費。

已讀 round2 後段的重要反例：606/606 selected winners 可由既有 cyclic/disjoint 加 simple balanced-wave menu重建；geometry alias 的額外 seeded search曾造成假歸因；相同 joined work 不保證較低 native latency；G4容量也可被route-blind chip swap改善。因此新方向應改必要 work或實際 critical path，不能只放大搜尋。

同時讀了 round3 contribution review，採納其關鍵界線：**通用 IR可表達或generic+新pass精確打平，不代表已發表prior已提出此pass。** 前者只移除相同execution之間的額外速度主張。若 unchanged earlier mapper獨立找出相同合法方案，才縮小 mapping/search貢獻；有直接語義與backend限制的prior，才構成具體新意碰撞。不能過度否決一整個系統問題。

### 已向 thinker_sol 質疑的初步方向（均未定候選／未跑實驗）

**1. Exact bit-plane refine。** 若最終所有 INT8 bits都要讀取，ordinary NAND sensing work不減，分pass還需accumulator與更多命令。GU低位未完時不能越過SiLU×up讓正確Down就緒。若early stop改品質，需與progressive precision方法比較。這是機制必要性的質疑，不是禁止新量化研究。

**2. Expert base/residual。** GU同input可共享base product；Down的expert-private input一般不能直接重用同一product。精確residual若仍dense／full-rank，可能只加一份base而無讀取減少；SVD等近似必須計品質。這條被直接MoE compression prior重疊，不以「base+delta」單獨立新意。

**3. Post-A8 exact-zero Down columns。** full-vector amax完成後，零是canonical W8A8契約內的exact zero，可在該契約下精確跳weights，無需賭預測；這確實可能刪除必要sensing，值得先篩。但頁／wave粒度與accum決定可行性。填滿16KiB的 R_b×C_b tile且R_b≤H時，C_b至少ceil(16384/H)。G4 H=1536至少11 columns/page；填滿64KiB四plane wave至少43。若8 columns並把rows pad至2048，頁面25%為padding，儲存相對useful增加33.3%。Mixtral H=4096則4/page、16/wave。只保留H/16 rows/chip會令column tile更寬；改成column sharding則要每chip完整H的I32 accumulator與reduction。

在獨立零機率z的粗略控制，整tile可跳機率是z^C；整wave需要更多columns同零。這不是實測假設，只用來要求co-zero correlation必須超過簡單隨機／hotness permutation。Down一般約1/3 FFN權重，若只省Down sensing，要10% FFN工作量降低需Down先省逾30%，還沒扣reload/accum/output。若改獨立plane addresses及動態壓實，需明列hardware contract並給generic sparse tiled GEMV相同能力。

### 新查 primary prior（以實際已讀深度為準）

| 來源 | 已讀深度與限制 | 對候選的要求 |
|---|---|---|
| [Any-Precision LLM](https://arxiv.org/abs/2402.10517) | 摘要與HTML入口；bit-plane overlay、多位寬engine。尚未逐式讀全文。 | progressive bit representation不是空白。 |
| [BitWave](https://arxiv.org/html/2507.12444v1) | 摘要、背景、method headings；bit-column-serial、structured bit sparsity、compression。 | 窄位serial／skip-zero-bit需同類控制。 |
| [D²-MoE](https://arxiv.org/abs/2502.17298) | 摘要與原論文搜尋段；shared base、SVD deltas。 | base/delta不是新點；exact與approx分開。 |
| [DeRS](https://openaccess.thecvf.com/content/CVPR2025/papers/Huang_DeRS_Towards_Extremely_Efficient_Upcycled_Mixture-of-Experts_Models_CVPR_2025_paper.pdf) | 原PDF摘要／method段；upcycled MoE shared base加delta。 | 不能只挑具高度相似expert宣稱一般MoE方法。 |
| [LorExperts/BTExperts](https://arxiv.org/html/2608.07814v1) | HTML摘要、related work及motivation；local anchors、alignment、coactivation communities、amortized shared compute。 | 群組base與共選重用也有直接prior。 |
| [LLM in a Flash](https://arxiv.org/abs/2312.11514) | 原文摘要及既有專案review；zero-related selective column fetch與bundling。 | exact-zero只改適用數值契約，不自動給新意。 |
| [PowerInfer-2](https://arxiv.org/html/2406.06282v1) | HTML入口與摘要搜尋；neuron clusters。 | 查明其sparse checkpoint及offline hardware-aware clustering。 |
| [VLM in a Flash](https://openreview.net/pdf?id=3xwsD68F2K) | 原PDF搜尋段§3/4.3；activation-stat reorder、contiguity-aware chunks。 | cozero placement需勝此類簡單統計排序；approx選擇與exact分開。 |
| [SpAx](https://arxiv.org/html/2610.02598v1) | HTML§2/3/5.3與原文摘要；activation column sparsity、flash coarse reads、co-access contiguity，另含近似representation。 | 普通column skip與聚合reads有direct prior；AiF的可讀粒度／有限accum才可能是未解機制。 |

research_terra另傳 Ripple（arXiv2410.19274）§4.1直接GU/down綁定與共激活重排、SpQt（2511.04477）quantized layout+sparsity；尚待本人讀原文。其Micron專利US10877696B2指出獨立plane state machines是用來解除傳統同步限制的明確設計，不能把空plane免費放另一expert。

本階段未執行任何候選實驗。與root／thinker_sol／research_terra已交換上述質疑；正式候選起點等待root同步。

## 19:07:01 UTC：正式候選 R1 開始（最早 19:37:01 實驗）

root已同步正式研究起點。本輪候選是 **selected experts合併Down residual error sketch，讀少量低4bit頁refine**。先保留完整W8 GU與full-A8 barrier。Down以signed4 nibble精確分解 `w=16h+8+r`，base `16 dot(h,z)+8 sum(z)`，residual r∈[-8,7]。離線存 `C_e=S diag(s_weight_row)R_e`，線上以實際expert-private A8向量及router alpha算各expert的sketched error；選擇refine哪些低4bitactions，使未讀residual的合併誤差小。Shared可作額外expert，機制不依賴它。

這改成明示的近似候選；只有讀回全部residual並於同一整數累加／同一次scale前加回，才是exact W8 fallback。不是把所有模式都叫exact。

### 19:08–19:10 UTC：首輪黑臉

**真必要性：** 相較純placement，此設計有機會真正刪普通sensing；不同expert的Down誤差會在alpha combine相加，單expert重要性未必捕捉互相抵銷。尚未找到與「runtime cross-expert residual cancellation＋page refinement」完全同義的prior。這只保留研究價值，不代表已證新意或效益。

**品質／安全：** 若S rank r<H，有非零u位於其nullspace，Su=0但真誤差可大。不能用固定sketch下的small norm當決定性品質證書。單一固定token的2^K subset sums可做finite JL union bound，dimension應付O((K+log(1/δ))/distortion²)，不是只付一條向量。Free-running後續z又依之前同S決策產生，固定input獨立性失效。[Hardt–Woodruff STOC2013](https://arxiv.org/abs/1211.1056)正是linear sketches對adaptive inputs的不穩健性證據；它不證真LLM會遭攻擊，但阻止無條件安全主張。可改成經驗selector，需真正長序列與多seed品質驗證；另設validation sketch也不能自動消除跨token依賴。

**資料量：** FP32 C_e為r×I，對rawDown H×I bytes的額外比為4r/H。r32的G4為8.33%Down；40層×64experts×512×32×4=160MiB metadata。Mixtral 32層×8×14336×32×4=448MiB。若從flash讀，同樣扣感知省量；若放DRAM，必須與相同160/448MiB用於ordinary cache的控制比較，並付selected sketches每token的CPU memory traffic與K r I MAC。這些是尺寸推導，未跑候選資料。

**天花板：** 只將Down減半，理想完整FFN必要權重讀取最多少1/6=16.67%。若G4 r32 sketches也走flash，先扣8.33%Down後，天花板降至約13.89%FFN，尚未付任何refinement、page padding、decoder、control或額外barrier。完整模型還受Amdahl限制；不能把16.67%寫成全模型收益。

**feedback與數值：** 相對base-combined norm的判準需先得到完整coarse Down outputs，然後才決定residual reads；新增critical-path roundtrip。若base與residual先分別scale至F32再相加，未必與canonical W8 bitwise相同。需保留每expert的I32 partials至決策，或明示另一數值契約及品質控制。

**預先要求的強控制：** 最佳static4/5/6/8bit品質／讀取Pareto；普通alpha、activation norm及variance ranking；同一sketch只用各expert norm而忽略cross terms（隔離cancellation價值）；相同residual頁actions與同能力progressive方法；同DRAM預算cache；true-residual oracle僅作可達性診斷，不當runtime方法。

新增近prior [Channel Selection for MoE Decoding Needs No Calibration Data](https://arxiv.org/html/2610.37386v1)已讀§5及appendix相關段。它以低bit GU screen選exact channels並對其餘作lowrank恢復，另含error-variance scoring；與本候選的Down residual subset不相同，但其較激進讀取／品質Pareto是必要對照，不能只比uniform W8。PGWeight exact URL已向root索取，網搜精確名稱未命中，未猜內容。

本節所有內容仍是研究／推導，沒有讀新保留routes、執行候選數值或服務性能實驗。

## 19:14:46 UTC 落盤：直接 prior 後的剩餘貢獻與強反例

root更正名稱為 [PagedWeight](https://arxiv.org/html/2607.16184v1)。已讀§3：AP bitplanes/LUT分頁、offline sensitivity、routing statistics及prompt residual品質planner。這不是NAND physical page證據，不能把其weight page直接當16KiB sense單位。

已讀 [DPA-MoE](https://arxiv.org/html/2610.21378v1) §4/7/Appendix B：progressive chunks、offline per-expert residual與live alpha²作postrouting allocation已有先例。它明示忽略cross terms，但Appendix B量的是同precision下跨dispatch的correlation；不等於同token的不同refine subsets排名。不能靠該correlation直接否決本候選的剩餘假說。

已親讀 [DP-LLM](https://arxiv.org/html/2508.06041v2) §5.1：先存投影後的quantization error matrix，runtime小GEMV估error來選precision；也有input-norm regression低成本分支。這是generic error-sketch estimator的直接先例，不能再將它作新點。其校準後的矩陣亦不應被當純隨機JL保證。

[BASE](https://arxiv.org/html/2609.36222v1) §3.3已有對同token expert interactions與cross-aware greedy control的真輸出分析。它處理wholeexpert removal，不是目前的Down quantization residual；其負結果只構成強反例與控制，不能擴成全域no-go。

### DPA correlation不能證subset排名：解析反例

令三個weighted residuals為 `r1=(1,0), r2=(-1,0.01), r3=(0,0.02)`。各dispatch同乘任意正數s，all-expert exact與diagonal norm²均正比s²，跨dispatch Spearman=1。但固定refine一個expert，diagonal會選r2，留下true norm²=1.0004；signed最優選r3，留下0.0001，差10004倍。這只證該統計無法支持所需排名，不代表真模型有此強反相關。

一般還有非單調性：更多exact refinements可破壞本來的error cancellation，令combined error上升。按每expert residual單調修正的knapsack不能自動延伸到signed objective。

### 原版pure selector的具體同能力反例

對任何固定未refine subset，令其combined error為u。若sketch rowspace正交化成Qᵀ，既有metadata與計算已給 `c=Qᵀu`。直接在host補 `Qc` 後，error變 `(I−QQᵀ)u`，因此：

`||error||² = ||u||² − ||c||² ≤ ||u||²`。

對一般full-row-rank S也可用 `Sᵀ(SSᵀ)^−1Su` 作同rowspace projection。這是在候選自身layer-L2 surrogate下、同一residual-read subset的逐case改善；多付H×r MAC與basis讀量，不能冒稱下游品質／latency全面支配。若改data-aware U捕捉較強signal，這個直接lowrank compensation基準更必要。改良後的selector若只剩未觀測tail，其cross資訊又是原sketch沒有的部分，需解釋為何勝cheap tail-variance allocation。

### SNR與路由寬度

對各向同性H維residual，真cross cosine標準差約1/√H；r維random sketch的cross估計雜訊約1/√r。r≪H時，2^K搜尋容易挑出假的cancellation。提高r會吃掉metadata/CPU/BW，不能只引用單vector norm估計準確率。相反若真cross由少數方向主導，data-aware低rank補償可能直接取得同資訊的價值。

無shared top1沒有cross；top2且固定refine一個時，未refine只剩一個，signed和exact per-expert norm排序完全相同。嚴格quality tolerance若常留下0或1個unrefined expert，同樣沒有cross收益。適用有／無shared仍可成立，但不能宣稱所有K都有額外機制。

### 原生普通讀取的天花板再收緊

Packed4bit base的N MAC沒有減少。若PE確實可承受6.4G weights/s，一個64KiB packed wave展開成131072個I8乘法需20.48us，普通sense28us在理想pipeline可能掩蓋；但PE lane與吞吐仍須核實。保守serial sense+drain時，base成本比例是 `(28+20.48)/(2×(28+10.24)) ≈0.634`，只省36.6% Down，理想FFN天花板約12.2%，尚未扣sketch、refinement與額外feedback。

G4舊Down每128×512 I8 shard恰64KiB一wave；壓成4bit後32KiB仍一wave，沿用原g12 layout沒有sense節省。要改固定geometry，例如256×512 packed4bit一wave、較少activechips；需重新計collision、capacity與fallback。不能把舊native Down成本直接除2。nibble decoder可把h解為I8的16h+8送原PE，不必免費假設MAC數減半。

截至本段仍未跑任何候選probe。原版傾向thought-reject；剩餘研究針對是否有夠強且可廉價觀察的真cross結構，以及新硬體是否提供可投稿的獨立必要性。

## 19:16:42 UTC：root 思考淘汰 signed-sketch 原版

root已決定不讓原版進實驗。**本次新實驗仍0/10，並非一次實驗失敗；正式候選思考19:07:01–19:16:42未滿30分鐘，故沒有執行任何probe。** 不為補輪數或湊時間跑已判不值得的方向。

原版被淘汰的具體鏈條：DP-LLM已提供runtime quantization-error projection，DPA已提供selected-expert progressive allocation；新餘地只剩signed cross terms。小random sketch對真低量級cross有SNR／selection-bias問題；若改data-aware basis取得signal，直接lowrank補償可在同一read subset下逐case改善其自身L2目標；cheap per-column variance與DC correction還沒比較。Native普通read的page rounding與MAC-expanded drain會再縮小理想16.7%FFN讀量頭空間。這是新穎性／必要性不足的thought rejection，不是已證任何signed selector全球無效。

也保留兩個不應被用來錯誤封殺的點：DPA acrossdispatch correlation不能證同token subset排名；BASE的wholeexpert greedy結果不能證Down residual全域不可能。原相對base-norm門檻的feedback barrier也可用固定budget或layer-input norm解除，不是所有sketch機制必然。

### 最強 cheap controls 的可實作公式（尚未執行）

對每expert殘差R與row scales s_w，預先存每column `v_i=Σ_j(s_w,j R_ji)²`。線上以 `alpha² s_x² Σ_i v_i z_i²` 作diagonal variance score，只有O(I)運算及4I bytes metadata，對r×I sketch便宜r倍。更粗的scalar beta乘||z||²也應列控制。

另存每row residual mean μ_j，以 `alpha s_x(sum_i z_i) diag(s_w)μ` 補回DC誤差，處理midrise4+4的共同偏差。若所謂cancellation改善主要來自該DC，普通bias correction已取得它。若去DC後tail近isotropic，random sketch SNR疑慮更強。

低rank補償也有直接MoE先例 [MiLo MLSys2025](https://arxiv.org/html/2504.02658v1)，已讀摘要及method入口。另查 [NatScale](https://arxiv.org/html/2610.20353v1) primary，joint是同expert GU/Up/Down投影聯合量化，不是runtime cross-expert subsets；未誤當同義prior。

## 19:19 UTC：capacity-for-LUT 探索（尚未正式新候選計時）

root新探索：activation product quantization，將D維input分D/g個groups，每group C個centroids，預存每個weight block乘centroid的R維output。GU共享input，Down為private input。這是新探索，不自動延續R1 timer。

FP16 table outputs下，table容量對raw-W8比例為 `2C/g`；activation索引bitrate為 `log2(C)/g` bits/dimension。C16/g16僅0.25bit/dim但已2×容量；C256/g16為0.5bit/dim且32×容量。若要canonical A8對任意g-dimensional input exact，需C=256^g，不可能把有限小codebook稱exact。

**小R普通wave反例：** Deep GU R2816的FP16向量5632B小於16KiB page／64KiB wave。D2048/g16=128 groups，若每lookup獨立四plane wave，128waves/expert反而多於W8 GU的88waves。把byte ratio直接當wave ratio會製造假收益；原chip row-sharding使每lookup更小。獨立plane地址／動態壓實可改善，但屬需明列的新硬體與公平能力基準。

**不能過度封殺的大R例：** Mixtral GU R28672、FP16向量57344B接近一個64KiB wave；D4096/g16=256 lookupwaves/expert，比原W8 GU1792waves有大幅讀取空間，不要求獨立plane。問題轉為極粗activation PQ能否維持品質、容量、完整output accum、centroid search、LUT sum，以及與同activation和matched-quality directW4/W8的公平比較。這是尺寸解析，不是實驗結果。

已找到 direct prior：[MADDNESS](https://arxiv.org/abs/2106.10860)、[LUT-NN MobiCom2023](https://arxiv.org/abs/2302.03213)已做centroid products與lookup/accum；[Stella Nera](https://arxiv.org/abs/2311.10207)已有PQ-based multiplier-free硬體。LUT本身不能立新意，但NAND page／plane原生執行、capacity–quality–wave tradeoff仍需具體核對才可決定。

## 19:24 UTC：LUT 的三方限制與更強 prior

令entry為s bytes、table容量對W8比例F=sC/g、理想selected bytes比例ρ=s/g、activation code rate q=log2(C)/g。三者必滿足 `q=(ρ/s)log2(F/ρ)`。FP16 s2、容量不超2×且想selected bytes不超W8一半時，q最多0.5bit/dim；容量4×也僅0.75bit/dim。這是表示成本恆等式，不是對真activation品質的普遍下界；若activation有低維結構仍可能有效，但三種好處不能同時免費拿。

Exact A8若整group列出全部輸入要C=256^g。若改binary bit-slice八次lookup，selected bytes比為16/g，只有g>16才比W8少bytes，table卻至少2×2^g/g（逾萬倍）。因此合理容量下的exact LUT不能直接沿用approx PQ的低讀取宣稱；其他對稱／分解可另研，不能在未計額外lookups前稱解決。

已讀 [LUT-LLM FCCM2026 v2](https://arxiv.org/html/2511.06174v2) §III、§IV-C與§V-A。它已分析activation-only table的bandwidth／storage限制，採activation–weight co-quantization、2D LUT與parallel centroid search；Qwen3轉換需訓練。其SRAM／FPGA索引存取不是NAND page sensing，不能直接拿來宣稱flash執行，但已移除「LLM改查表」本身的新意。[LUT-GEMM ICLR2024](https://arxiv.org/abs/2206.09557)則是weight-only低bit表示的lookup計算，與permanent activation-output table分開比較。

### GU 共input可救小向量，但需合法固定 packing

同group的GU centroid index對所有selected experts相同，故可把多expert LUT outputs放一起，再以固定selected expert集合讀views。Down各expert的centroid index不同，無此一般共用。若table vector5632B、每wave約容11experts，但E64/K6散於約6waves，所選experts常會觸多個waves；必須計真正resident wave union，而不是將六份useful outputs相加後ceil一次。Coactivation grouping有既有declustering prior；它可作強基準但不能免費看test。

### 可能更合理的另一方向，尚未定候選

永久存所有W·centroid outputs受容量與可讀粒度夾擊；可考慮只讀compressed weight indices＋小codebook，再於chip解碼或算activation LUT。例weight vector g2、Cw16、G512 vectors/codebook：indices為0.25W8，I8 codebook再占0.03125W8（FP16則0.0625），表示上約0.28125／0.3125W8。這是真減必要讀取量，而非省MAC後仍讀完整weights。

但它已屬GPTVQ/VPTQ、LUT-GEMM與LUT-LLM家族，不因移到flash就自動投稿成立。需核實I8 codebook／row scale、page整數packing、decoder吞吐、ECC feed、有限SRAM及與同硬體nibble／codebook direct GEMV控制的必要性。ordinary28us提供compute slack，重度壓縮後展開到原MAC數則仍受PE roof；現有6.4GB/s參數不能代替PE lane證據。

截至本段：未正式啟動第二候選timer，未做LUT品質或性能probe；新實驗仍0/10。

## 19:26 UTC：新正式優先假說 Gate-first exact-A8-zero

root同步此候選正式研究start19:26 UTC，最早19:56 UTC才可進實驗。另一frontier假說19:23–19:53由其他角色研究；我優先審查gate-first，不把平行時間相加。

方法：先完整gate值g_i；利用部分真正up witness取得canonical SwiGLU全向量maxabs的下界L>0；預存up weight row bounds並用實際共同input x計 |up_i|≤U_i。若 `|SiLU(g_i)|U_i < L/254`，則全向量動態A8的該座標必為0，可以省未讀up row與Down column。Shared與routed都可用，沒有shared依賴。

### 邏輯成立的範圍

若bound對實際F32 up/SwiGLU值真確，L是真正已算activation的max，且L>0，則被certify的row也小於L，不可能影響最終amax。其餘rows全算後，max與canonical一致；被省row的A8恰0，故Down整數dot可保相同。這比預測sparsity或近似CATS多一個可檢查的exact契約，暫保留研究價值。L=0、無witness、Nonfinite、特殊bias／activation／額外quantization契約需要明列fallback。

### 已發現需要修正的數值細節

本repo `scripts/research_20261102_full_ffn_integer_reference.cpp:52`使用 `s=F32(amax/127)`，再 `std::round(F32(v/s))`，**ties-away而非RNE**。條件必嚴格、處理s與division的F32rounding，不能用理想實數threshold冒稱exact。Canonical up先exact I64 dot，再F32轉換，與F32(rowScale×inputScale)相乘；bounds也要覆蓋該轉換。

### 目前最重要的反例與強控制

- 普通L2/L1上界對典型高維dot常有約√H的alignment gap，H1536–4096約39–64倍。若gate幅度一般，可能只有極負gate或極近0才能cert；真正A8zeros若多來自small-up，不能在未讀up時取得。
- split Gate→witness Up→remaining Up→Down新增commands、vector residency及barriers；完整GU串接的最佳row geometry不一定能直接拆半。Witness讀過的up rows不能再計「省up」，讀到同page的其他rows也要付。
- 真zero rows要形成NAND可省的整page／4plane wave。4KiB SOML是硬體設計，不是ordinary16KiB read改參數；必計bitline/block-decoder衝突、面積與ECC/PE安排。所有strong baselines均得同能力。
- 對原dense三投影，若up/down實際可省fraction為p/q，絕對weight-work減幅=(p+q)/3；這不是相對strong baseline的延遲上限。strong baseline也省Down真zero，正確相對分母見19:40修正；row sparsity百分比不是可省physical bytes／waves比例。
- 至少比FullGU+postA8 exact-zero Down-only、gate-first無證書、同SOML／同SRAM generic sparse executor、CATS/SLIM的適用模式；exact對照不許以近似品質省算混分母。

### 避免只攻loose L2稻草人

可用實際qx∈I8與exact row metadata建立更強保守bound：每row每B列存 `n_ib=Σ|w_ij|`，線上 `U_int=Σ_b n_ib max|qx_b|`，之後以outward-rounded output conversion變成F32界。B128時每block sum≤16256可用2B，metadata為2/B=1.5625%up weight bytes。Signed positive/negative block sums配qx min/max可再產interval；global三種Holder bounds取min亦合法。這些是設計選項，尚未跑任何統計。

更緊的partial-high-bit Up＋lowbit envelope會增加前置讀取與continuation；若改走此法，需比較existing exact early termination。已查 [SnaPEA ISCA2018](https://cseweb.ucsd.edu/~vakhlagh/ISCA18-SnaPEA.pdf)有exact與predictive兩種mode，前者以partialsum與weight ordering證ReLU零。它不是本次dynamic A8 max-witness證書的同義prior，但「exact early zero」大類不能宣稱首次。

本候選研究中，尚未進probe或新數值統計；本次新實驗仍0/10。

## 19:34 UTC：Gate-first 數值證明收斂與 Gemini 交叉審查

root將最早實驗時間保守延到 **19:58:00 UTC**，確保Gemini正式加入也滿30分鐘。本文不以平行時間相加縮短。

### 對 canonical F32 契約的直接安全判準

root提出exact row metadata `W2_i=Σ q_w²`、當前共同input `X2=Σ q_x²`，令 `B_i=isqrt(W2_i X2)`。因dot為整數且Cauchy給 |dot|≤sqrt(product)，**floor integer sqrt已足夠**，不必ceil。現有模型維度下product可用U64/I64明列檢查，不能在未檢查範圍時讓平方乘法overflow。

Canonical up為 `s=fl(sw*sx); f=fl(I64dot); up=fl(f*s)`。若s有限非負、round-mode一致、禁止fast-math／非預期FMA，I64→F32與正數乘法單調且正負對稱，`U=fl(fl(B_i)*s)`界住真|up|。若s或U非有限則不cert。用真正原kernel產生的F32 gate activation h_i，令 `Ba=fl(abs(h_i)*U)`；Ba也要有限。

L取真正F32 seed activations的absmax。若 `sL=fl(L/127)>0`，用 `fl(Ba/sL)<0.5f`作判斷，比理想L/254更直接匹配canonical的F32 division與ties-away。必要時另要求Ba<L以明列amax-owner安全性。Actual max M≥L，故actual quantization scale≥sL，真quotient的F32 magnitude受此界，round-away必為0。

NaN不能被忽略；尤其h=0但U=Inf時，不能將0×Inf當0去跳過canonical可能的NaN。L=0、sL underflow為0、空witness集均fallback；Ba=0且已證up/h finite時可直接確定raw activation為zero。若全rows都已如此證明，canonical全zero scale為1也可單獨處理。

權重norm metadata每row4B、線上X2每layer共input只算一次，是合理輕量方案，但每token全部selected Up rows的isqrt／scale／mask與metadata讀取還要量測。用hardware sqrt近似再整數修正可保持exact；不能把整數isqrt運算與edge CPU成本當零。

### Gemini稿件的具體修正意見（已傳root轉交）

已讀 `docs/research_2026_11_03_thinker_gemini.md`。其真實參與時間保留，但下列不是已驗證事實：ECC6.4GB/s不等於已公開PE吞吐，63.4% PE idle只是條件roofline；99.5%感測能耗需來源與分母；q4 inputs免費放PB要先滿足4×input bytes≤32KiB/plane等明確配置，max32KiB input已用滿；PE面積線性複製不含mux／SRAM／布線合成。

四個備案審查：

- A read-once q4 speculative：需計input residency、FIFO/output channels與正確acceptance；max-chip marginal work零不代表native critical path零。
- B slack promotion：`Wc+Δ≤maxW`只是chip-work條件，新增commands、PE、GU ready、Down dependency可能延遲。NF4＋linearRes4不自動exact重建W8，LUT／scales／padding也非零metadata。可保留latency-aware precision的真gap，需勝DPA同能力allocator。
- C slice fusion：block-A8明確換數值；Down各block input scales不同，單一I32 accumulator不能免費代替scale-aware partial reduction；切換原GU input與private Down context需付buffer／reload。
- D layer-shared VQ：3bit/weight、d4對應4096entry×4B=16KiB codebook；若要32 I8 weights/cycle需8次random lookup/cycle，單16KiB SRAM沒有自動8 read ports，bank衝突／複製／面積都要計；一次wave載入也須分perchip與global broadcast。

這些意見針對原生能力、成本和同能力基準，不以generic可表達性反向否定所有系統貢獻。所有備案仍未進本次實驗。

## 19:40 UTC：新增收益歸因、分母修正與 A4 備案

最強公平控制先完整GU，再用**所有真正A8-zero**跳過Down columns；本候選也應在完整剩餘Up之後利用所有真zeros，而非只用事前certificate rows。因此新可歸因的必要讀取差別主要是不用讀已證zero的Up。

此前討論的`p/3`只表示相對原dense三投影的絕對weight-work fraction，**不能稱為相對已稀疏Down基準的延遲上限**。若理想化gate/up/down各cost1、strong baseline省Down比例d、候選省Up比例p，則相對work reduction為`p/(3-d)`，speedup為`(3-d)/(3-d-p)`。若看10%相對work reduction，條件是`p>0.1(3-d)`；若看speedup≥1.1，則是`p≥(3-d)/11`。真實NAND必須改用合法pages/waves與native DAG critical path；上述等cost scalar式不證明實際延遲上限或下限。

若先讀Up高4bits才證明略過低4，則新省的p是lowbit物理讀取占完整Up的實際fraction，而非certificate row比例本身；它仍應除以strong baseline實際成本。Witness pages、rowgroup共同cert、padding、commands、metadata與host bound運算會進入此成本。

Thinker提出A4/A6作後續備案：Down-input對稱A4令Qmax7，理想zero門檻由L/254變L/14，寬18.1倍。但這是**新的W8A4 reference與品質問題**，不能偷換本候選A8範圍；baseline也得同A4與同zero-Down能力。FP8的zero deadzone可能比A8更小，不能泛稱任何低精度都更有利。概念上的『剩餘值域落同quantizer bin便early terminate』已有exact早停脈絡；尚未查到同一dynamic expert-amax witness、SwiGLU gate-first且避免NAND Up讀取的完全直接先例，不能只憑通用框架能表達就判沒有研究價值。

整個候選仍在30分鐘正式思考窗口；未執行probe、未讀新保留樣本，新實驗仍0/10。

## 19:45 UTC：可行的 cheap bound、Down 資源取捨與 Gemini 過度推論

### 不把逐 row integer sqrt 成本當必然

可offline存`N_i=ceil(sqrt(W2_i))`，token共同計一次`X=ceil(sqrt(X2))`，則`B_i=N_i X≥|dot_i|`是安全整數界，再經canonical `scale_output`。常見H下N可用uint16（仍須依實際q範圍／shape檢查），省每row integer sqrt與一半norm metadata，代價是略鬆界。精確floor-sqrt-product、block-L2與其他界可作strong comparison；在未量測前，不能因一種慢實作先否定整個證書。

### Down 的可省比例 d 需要真幾何

4KiB tile面積`R_output×C_input=4096`。H4096全output×1column可一zero column跳一tile，但每owner要16KiB I32 output accum；若分column給多owners，host需多份H維partials再exact整數reduction。若維持16chip rowshard、R_output256，4KiB含16input columns，須共同zero。H2048全rows×2columns需8KiB累加。可在中間R_output取捨，不能說只有full-column唯一合法方案。

同一expert不同column shards須先I32/I64合併再做一次canonical F32 scale；先各自F32 scale再相加通常不是bitexact。不同experts各有private input scale、output row scales、router α，不能先跨experts把I32直接混合。若16owners各輸出H×4B，這是**每expert**流量；每layer還須乘實際K，除非增加明列的跨expert F32處理與新數值契約。雙方應各取dense／sparse Down最快合法執行，避免稀疏強基準反而被強迫慢layout。

### 對 Gemini 新深入稿的實質反駁

其固定hot Up witnesses與Gate同phase發出，可去除一個獨立witness barrier，值得保留。但『train上幾乎從不zero』不是100%必要讀取證明；phase1的額外payload、wave ceil與critical path仍付。相同Gate/Up/Down permutation對本integer-dot reference可無損，實體block packing／spare capacity並非因此自動零成本。

Gemini的Gaussian block-Hölder約2.5√H與B1約0.637√H，是同分布且符號無結構情境的合理風險分析；我的原提議已明列可能比L2鬆、取min，沒有宣稱必然變緊。不能由這個情境證『只有outlier-deflation能收緊』。Block間能量非均勻、structured weights、input support等均可改緊度；真top16能量50–90%須來源／量測。

其sphere-packing `sinθ≥2^(−b/H)`進而『2%metadata至多改善10.5%』欠缺明確weight/input分布、fiber形狀、平均／最壞case定義與證明，不能當本模型no-go定理。Gaussianσ_g、L/σ_u=15、certificate率1–3%均不是實測。`p^m`僅在獨立row events假設下成立，不能當co-certification clustering之後的普遍上界；Down共同zero機率也不能無條件稱0。

24/16=1.5與44/16=2.75是平均wave work，不能稱每chip合法waves。G4 raw bytes/64KiB=12也不保證full-row＋4plane packing合法；partial-row continuation、padding或geometry須分別計。原生同步fixed waves可能要整wavezero才省時間，但允許何種合法重排／SOML打包後必重算，不可混用凍結舊wave與理想新packing。

CATS／Park等近似gate-first是同品質Pareto強控制，尚不是對exact W8A8契約的數學支配；『原quantizer也有誤差，所以exact加速無價值』不是充分否決。若matched-quality block quant或approx sparse方法更好，才構成具體必要性問題。

新增primary核讀：[Park等，Uncovering Intra-expert Activation Sparsity](https://arxiv.org/html/2605.08575v1) §4／Appendix B，已有真GPU gate-first＋fused sparse Up/Down；附錄以raw `g>=cutoff`，不是exact dynamic-A8證書。已讀[BitSET作者CASES2023 slides](https://pyjhzwh.github.io/files/CASES23_BitSET_slides.pdf)，其early termination明屬prediction／accuracy tradeoff；不能把benchmark無loss當bitexact定理。Terra補[PowerInfer-2](https://arxiv.org/html/2406.06282v2) gate後才繼續讀flash Up/Down，待直接核全文後細列差異。

仍無候選probe，新實驗0/10；正式Gate研究已持續，最早19:58前不開實驗。

## 19:50 UTC：已付 Gate dot 的條件界與 pinned SiLU kernel

### Conditional Cauchy 是 stronger control，尚未證效益

Thinker與我獨立推得同一標準Gram投影界。對同row的canonical gate/up權重，令`A=||qG||², B=||qU||², C=qG·qU, T=qG·qx, X=||qx||²`。若A>0，則

`A*tUp ∈ C*T ± sqrt[(A*B−C²)(A*X−T²)]`。

它由將Up和x各投影到Gate正交補而得；A/B/C可offline存，T已由完整Gate計算支付。可對integer差量取floor sqrt與signed ceil/floor division形成安全區間。但AiF若只輸出scaledF32 gate，不能當成raw integer T；須新raw-I32 flag或從F32輸出保守求T preimage interval。128-bit乘積／sqrt、metadata與host時間也要付。Dyadic β 的`Up−βGate` residual norm是可能便宜版本。典型Gate/Up低correlation時改善或很小，這不是既有模型有效的證據；它只直接說明純靜態metadata球面論證漏掉已付Gate-dot的動態資訊。沒有把standard Cauchy稱新定理。

### 對真正 F32 SiLU 的源碼確認

已讀pinned upstream `/home/nfs_client/ericyu/expert_flash_experiment/sources/llama.cpp-927dada6c/ggml/src/ggml-cpu/vec.cpp:396`與`:433`：`ggml_vec_silu_f32`及`ggml_vec_swiglu_f32`各架構SIMD branch調同一`ggml_v_silu`，後者再乘Up；scalar尾端也同`ggml_silu_f32`（vec.h:1064的expf式）。因此可直接對**原完整I長度**Gate向量調原vec_silu，取得真正h。若只對壓縮subset算h，SIMD／scalar尾端分配可能改變，不能不查就稱相同。

對exact integer B路徑，沿用canonical cast與正scale乘法，RNE單調與正負對稱已足夠界住實際F32 up與activation。即使B轉F32向下round，仍大於等於實際dot轉F32，故不需要每步nextafter。Gemini的outward鏈可選但更鬆，不是此路徑『缺一不可』。若norm本身用浮點近似求得，才須另外證明outward界。現有有限值、roundmode、denorm處理與nonfastmath契約仍必要。

### 更直接 prior 的核讀結果

已直接核[PowerInfer-2](https://arxiv.org/html/2406.06282v2) §4.3–4.4：Q4 neuron bundle拆4KiB讀取，先取Gate、非zero才續讀Up/Down；整體另有predictor。它不是本次dynamic-A8 witness證書，但明確排除『首個gate決定flash Up/Down讀取』宣稱。Terra再找到[PADE](https://arxiv.org/html/2512.14322v1) §IV-A，已核其partial highbits、未知bits uncertainty、lowerbound更新threshold再停止後續讀取；attention sparsification的品質語義不同，但generic witness-bound＋precision early stop也已有。

目前黑臉判斷：有嚴重效益與新意門檻，卻未有能覆蓋實際weights／所有更緊界的合法no-go。若root凍結stage0 protocol，可在19:58研究時間滿後做有限headroom falsification；這只是實驗准入，不是論文價值批准。免費true-amax oracle與付費fixed-hot-witness應分列，CPU數值harness walltime不得當AiF延遲。還沒有啟動候選實驗。

## 19:55 UTC：Stage0 凍結草案的限定審查通過

已先後唯讀審`results/research-2026-11-03-no-cr/gate_zero_stage0_protocol.draft.json`原稿與root按互審修訂稿，以及Terra的`gate_zero_source_review.draft.md`。本階段沒有候選codegen、權重／activation probe或數值實驗。

修訂稿明列兩模型DeepV2-Lite（shared）與OLMoE（無shared）、既有TRAIN32前8jobs、raw prompt tokenizer含special後取首128tokens、固定greedy8步（即使EOS）、all-layers完整canonical W8A8 prefill/decode、decode為主且prefill另表。Deep layer0 dense與I2816實際merged shared branch皆覆蓋，不拆成假metadata experts各自amax。

Deflation只用前4jobs的decode8、unique共同input的qx²選top16，凍結後才score後4。Bounds包含global integerCauchy、block128、rawGate conditional oracle及固定coordinate residual；是否有用仍未知。TrueM免費oracle與付費首16KiB Up頁（H2048首8rows）分列。兩個4K/16K機會表均用此同一paid seed，4K清楚標hypothetical，沒有nativewave／latency宣稱。

最關鍵的歸因修補已加入：baseline用h==0 Up-skip mask，candidate用`h==0 OR certified`且seed整頁強制read；**各算整unit再相減**。混合hzero/cert頁不能拆集合後漏算，seed內原hzero頁被candidate讀回的損失也不能漏。Strong fullGU＋postA8zeroDown另列；Down d若僅逐coefficient理想值就標ideal，不能當實體省頁比例。

浮點契約已要求full-I pinnedGGML h、與實際graphproduct核對、F32嚴格0.5判準、finite／underflow／FTZ fallback、Gram G0 fallback與128bit溢位檢查。A6/A4僅primaryA8trajectory上的shadow，對自己的quantizer驗zero，未宣稱其free-running品質。所有來源、collector與模型hash須在實際candidate資料讀取前凍結；tooling修復留同輪，不將bug當科研失敗。

**黑臉決定：限定批准 bounded TRAIN-only Stage0 falsification，但最早生效仍是19:58:00 UTC，且source/protocol freeze必須完成；不是目前已可launch，更不是paper approval。** 這一方向仍未有已驗證效益，強prior已涵蓋gate-first、flash selective loading與generic witness bounds。若結果負面，只能淘汰這組bound/reference/model/prefix regime；若正面，只授權同一輪補真NANDlayout／cost／correctness，不授權投稿成功宣稱。User10%僅標誌，未採Gemini額外15/8/5%硬門檻。

直接核pinned `llama-graph.cpp`：`weight_before_ffn`僅LLAMA4；本兩模型為普通SILU split、router α在Down後，沒有另加clamp／bias，故本pilot的Gate/Up/Down契約適用。這是source審查結果，沒有讀新候選數據。

## 19:58:17 UTC：限定實驗准入的最後 receipt

Clock實讀`2026-11-02 19:58:17 UTC`；Gate正式19:26:00起至此32分17秒。19:28後Gemini專題參與的保守截止也已滿。這期間有文獻、數學、硬體幾何與凍結協定的持續交叉審查；最後短等待僅為時間gate，不充作額外研究成果。

最後核准draft SHA256：`d429210585c6c2ec3342ece41b00e8002cf0c467eca8709bc243f9e1e13cff88`。

新增事前negative criterion已核：BOTH models、free trueM與all-primary-min，在decode **pooled Up weight-byte** denominator下，額外超過hzero控制的scalar skip≤1%、額外完整16KiB機會為0，且前4fit後4score deflation亦符合，才可在黑臉再審後退除此有限regime；較大、混合或僅page-negative都inconclusive，留同輪follow-up。≤0.334%僅equal-projection dense coefficient-work之理想絕對比例，非native latency bound。每expert頁對齊／row-major／padding診斷不是全域最佳layout上界。

**時間與黑臉research gate現已滿足。限定Stage0 go；executor仍必須在讀候選資料前凍結最終protocol、source及hash，之後才可執行。** 未批准paper成功，也未批准擅自切換新方向或新保留資料。截至本receipt，我自己沒有生成candidate實作或跑probe；實驗輪數仍須由root記錄真正launch才更新。

## 20:05:40–20:09:28 UTC：執行前獨立 source review 通過

Root再次委派唯讀審查新collector與runner；我未生成candidate碼、未執行synthetic或模型probe，也未讀模型內容。已讀完整C++、Python runner與rev001→rev002 runner差異；frozen protocol相對先前核准draft只有狀態、freeze UTC與reviewed hash欄位增加，無偷偷改數值契約。

最後核實四個hash：

- Collector C++：`7ac28acadba5c7535e49647cfa43e37ca7838e6013a942a9edbd9d23cf6b7942`
- Runner：`4fb08952f10738861b805d1dc075e036cbadab0a4f5714f1c417c71dca778cc5`
- Frozen protocol：`39f9fed28de41de740bbd659a00360c7294bd61ee81c9832028b76598ce5e65d`
- Rev002 binary：`a96e4ebef1760a052ce290923dcf7d7a1c823bd3a1244b9db77beab23b77641e`

### 核對通過的關鍵路徑

1. `target/audit`依真weight名與shape辨識Deep dense0、26層routed/shared及OLMoE16層routed，三投影均換成固定row-W8／whole-vector A8 exact-I64 dot／原F32 scale順序。Finish要求每job／phase／branch／projection完整coverage；prefill all logits避免末層只算末token。非FFN保留原graph。
2. Gate/Up以`(layer,class,local-token,expert-ID)`配對，驗same qx/sx及shape；Down實際input與full-I pinnedGGML SiLU乘真正canonical Up逐element比bits，任何不符令run invalid。沒有用scalar expf、不同expert/globalamax或觀察Up值做假便宜證書。
3. Global Cauchy、block128各floor-isqrt再求和取global min，整數界成立。Conditional Gram用checked128bit、exact division-corrected sqrt、signed ceil/floor，G0／invalidGram退global。Deflate以固定16coords的exact signed sum之abs加剩餘norm界，取global／primary min；沒有漏掉已讀座標或把浮點rawGate冒充integerdot。
4. 每row逐一assert integer B≥真Up dot、F32 U≥真Up magnitude。Certificate用sameF32順序、finite guards、Ba<L與實際F32 Ba/sL<0.5。Mask移除後逐一驗own q8/q6/q4、amaxbits與scalebits，不能靜默接受falsezero。
5. 首16KiB seed在兩個logical grain表都強制read。Baseline h0與candidate h0 OR cert各算完整頁，added/lost分列；scalarextra與netbyte也分開。Pooled分母I×H且兩model全部H2048，runner乘2048換byte正確。Strong理想係數分母扣Down ownzero與baseline h0，標示非native latency。
6. 每batch common input依layer/token去重，前4jobs只decode8能量，逐layer32unique assert；stable top16 ties按原coordinate ID。座標flush/frozen後才讀後4jobs，fitrecords之後回算並分表，沒有把fit當新heldout。
7. Rev002在讀model內容前核source/build hashes與modelstat；保存CPU ISA、compiler、protocol、binary、library及sourcehash。讀model時hash前後stat一致。Synthetic receipt已存在且returncode0，我僅核其收據與binaryhash，未替它重新跑測試。

Prefill只保留完整canonical／finite／full-I product與ownquantzero診斷，不跑certificate score以縮小開銷；主decode、fit及negative判準不變，runner已明說這個邊界。Metadata報的是decode-touched Up caches，不是整模型部署；deflation儲存是logical16-I8＋U64residual估算，現CPU evaluator從fullW8 cache讀coords，未冒稱實體小SRAM實作。這些限制在rev002已補齊，不屬阻斷缺陷。

**結論：20:09:28 UTC limited Stage0 launch GO，未發現需阻擋此有限pilot的source bug。** 已傳root與executor。此決定只覆上述hash版本與原凍結範圍，沒有批准任何效能／投稿結論；任何實際assert失敗都須保留並在同輪修復，不當科研negative。

## 20:16–20:21 UTC：Round1 科學結果獨立審計，符合事前負面判準

Root委派在兩模型完成後才作科學判讀。實際rev003 run receipt已complete、兩process returncode0。我先唯讀核rev002→rev003差異：C++僅加同語意`require(bool,const char*)`避免每次驗證建立std::string；runner僅於freeze stat未變時重用原model SHA256。原attempt001與當時已讀model／部分forward資料完整保留，不是假稱本campaign尚未讀資料，也不把host overhead修補當另一個科研失敗。

### 獨立重算而非引用 summary verdict

新增審計artifact：`results/research-2026-11-03-no-cr/round01-stage0/critic_independent_audit.json`。審計只讀已產生的coverage／dispatch／common-qx／summary artifacts，不跑新模型、權重probe或新候選。對Deep的421,632與OLMoE的294,912行dispatch CSV，逐欄重聚合全部21metrics，與aggregates CSV完全一致；再核primary／secondary的summary數值。每個expert dispatch均有完整6bounds×2witnesses×3quantizers，無重複／缺失組合。

| 模型 | Primary all8 新增 Up bytes／全部 Up bytes | 比例 | 新增完整16KiB單位 | Last4＋deflate比例 | Last4新增16KiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| DeepV2-Lite（有shared） | 40,300,544／39,820,722,176 | 0.1012049551% | 0 | 0.1145666013% | 0 |
| OLMoE（無shared） | 15,454,208／17,179,869,184 | 0.0899553299% | 0 | 0.0953435898% | 0 |

Primary是免費trueM witness＋global／block128／conditional的per-row min；secondary是前4fit固定16coords後，只在後4score的all-rowmin。兩套分母都是pooled Up weight bytes，已扣除h==0 cheap-control重複歸因。不是平均expert百分比，也不是native latency。Primary all8與secondary last4不在同一資料集合，二者差值不能直接稱deflation收益；同last4比較，Deep primary 0.1033444643%→allrowmin0.1145666013%，OLMoE0.0910282135%→0.0953435898%，仍均無16KiB機會。

付費首16KiB seed的primary新增scalar rows僅Deep2235、OLMoE1550；新增4KiB logical units分別0、1，兩者16KiB仍0。本資料seed_lost_h0、lost4、lost16確為0；這是重算結果，並非省略seed-loss項。免費oracle hypothetical4KiB也僅Deep35／9,721,856 units、OLMoE3／4,194,304 units；不能把它稱原生AiF或SOML裝置加速。

### Coverage、數值與 fit 的獨立檢查

每token Deep full-I product rows=`10944 + 26*6*1408 + 26*2816 = 303808`。按各job**實際retained prefill＋8decode**總1042tokens，應316,567,936，與report完全相符；其中dense11,403,648、routed228,873,216、shared76,291,072。OLMoE每token`16*8*1024=131072`，總1041tokens，應136,445,952，亦相符。這裡沒有把每個prompt都假定128tokens，兩tokenizer對短prompt長度不同。

Deep2544、OLMoE768個job/phase/branch/projection coverage keys全符合預期；各prefill1callback覆所有retained tokens、各decode8callback皆1token。Finite output總數從coverage重新加總亦匹配report。Decode row-evaluation數對應Deep19,443,712、OLMoE8,388,608；6倍bound checks、36倍own-q/amax/scale invariant rows與block／deflate各16倍operation counts皆核實。Full-I product mismatch為0。

另外直接讀**已錄下**的common decode qx bytes，按first4／decode8／unique layer-token重新累積qx²，逐layer恰32inputs，stable top16與凍結coords及selected_energy完全吻合；後4未參與fit。此項是對同一實驗artifacts的審計，不是新增forward或校準資料。

### 科學判決與邊界

**符合凍結decisive bounded negative：BOTH models primary≤1%、added16=0，secondary last4亦≤1%且added16=0。建議Round1記一次科研失敗，退除此組bound/reference/短prefix performance regime。** 工具／host overhead修補留同一輪，thought-rejected residual sketch不計實驗。

這不是全域證書不可能定理，沒有排除其他bound、clamped checkpoint、重新訓練或不同格式；也不是所有cozero layouts的native時間上界。A6/A4是primary A8 forward上的shadow：A4 Deep可見1805個16KiB logical opportunities、OLMoE仍0，但它們不證各自free-running品質／routing／速度，不能用來救本次A8判決。此輪沒有量到≥10%的完整模型或局部裝置效益，不屬『工程效益≥10%但欠投稿』分類。

結果與審計已同步root及executor；未自行啟動下一候選或新增模型probe。

# Round2 廣層價值搜尋（不是新候選時計）

## 20:26–20:35 UTC：frontier 的必要性反例與候選缺口

Root在Round1正式失敗後重新委派廣層source／primary-web審查；此段不沿用19:23 sidebar或前輪30分鐘。尚未選正式Round2方法，沒有codegen、weights／activation probe或實驗。

已核read-once／speculation直接prior：[SLIM §5.4](https://arxiv.org/html/2507.09201v1)已批次input SRAM＋output SRAM GEMM；[FlowSpec §3.1–3.3](https://arxiv.org/html/2507.02620v2)以topological draft segments逐pipeline stage推進，fulltarget完成後prune各stage；[FASER §4.3](https://arxiv.org/html/2604.20503v1)明列frontier chunks、即時commit、真reject重設，另有近似early-exit模組。不能將後者近似reject預測混同fulltarget真reject。

新增更直接primary：[BigMoMo v2](https://arxiv.org/html/2609.14643v2)已用SD multi-token window統籌flash expert移動、branch／expert admission、co-loading組織與ready-expert batching；[DraftExpert](https://arxiv.org/html/2607.24434v1)已有expert-expansion-aware truncation與draft-router prefetch；[Cascade](https://arxiv.org/html/2506.20675v1)已有MoE utility驅動horizon調整。因此read-once＋route-aware＋早停這幾個標籤本身沒有剩餘新意。

### 為何 generic frontier 目前應 thought-reject

leader為了早出真logits跨到layer l+1，其NAND讀取與suffix在layer l的future-only讀取爭同16組array。若suffix inputs早已ready，同頁附帶MAC就是同能力SLIM／groupedGEMM；若未ready，leader後續sense覆寫PB，有限小page cache不能保住整個expert矩陣給suffix。若將不同layers隔離在獨立pipeline stages，已落入FlowSpec主要資源模型。這些是對必要work／resource ownership的反例，不是僅因generic框架能表示就封殺。

若仍要保留，必須先給一個合法finite-buffer causal DAG，證明其page-live／ready-token規則能勝所有固定chunk＋earlyabort與同硬體adaptive groupedGEMM，且不暗增full-expert SRAM、array bandwidth或權重副本。保留普通read後PB資料做q個MAC並不自動等於CR-read；但若依賴保持WL／跨sense省precharge，便有CR別名風險。multi-context與q×MAC必須明付，不能從ECC6.4GB/s推PE throughput。

### 其餘池的價值反駁

Coactivation＋Monarch/shared-lowrank已被[NimbleMoE](https://aclanthology.org/2026.emnlp-main.2036/)直接覆蓋；只有放到NAND不夠。Lossless BF16＋NVMe有ZipMoE，低bit LUT有CodeGEMM/EVA，rawW8既有sample壓縮少但不是普遍熵定理。Temporal exact delta須保留integer accum／oldqx／scale，近似高cosine不等於qx座標不變；同時跨層PB不能維持整expert，generic temporal output reuse也須查直接prior。這幾項尚無可投稿必要性證據。

### 暫保留的可反證未知：overcomplete pair/subcoalition student bank

不以完整topK coalition作索引，而預存有限pair(i,j)的`S_ij(x,t)`，`t=α_i/(α_i+α_j)`；選中的topK中作disjoint matching，用`(α_i+α_j)S_ij`代替兩expert，其餘fallback原expert。這把必要讀取從兩份函數參數換成一份，沒有shared或CR必要性，代價是多存functional views及蒸餾近似。

純組合例（非routing資料）：E64、K6、M128條pair邊，均勻topK的期望共選邊為`M K(K−1)/(E(E−1))≈0.95`，避免完整coalition頻率爆炸；但實際matching/hit、容量與品質完全未知。若每student約一expert大小，64original＋128students約3×容量，仍是同16chips的容量交換而非新增讀頻寬。

真正致命的兩端：pair很相似時，簡單merge／Nimble或fixed-average蒸餾已足夠，liveα conditioning沒有必要；pair不相似時，student可能要接近2I寬度才能保品質，於是必要讀取不省。KDEM已蒸餾liveα teacher ensemble，但runtime還合併原weights，不能直接宣稱它已省本候選的NAND讀取；也不能由此反稱本候選首次ensemble蒸餾。強控制必有同容量／訓練預算的較少active-expert student MoE、固定pair student、pair merge、低bit及Nimble／LorExperts。單純重命名更多stored experts而降低topK不具投稿新意。

另外搜尋直接降低ordinary sense時間的可靠度／ECC共同設計。已找到[reduced-sense-time readout專利 US8493783B2](https://patents.google.com/patent/US8493783B2/en)、[read-retry latency優化](https://arxiv.org/abs/2104.09611)與DNN storage/ECC prior MaxNVM。故『提早sense＋ECC retry』不是新機制；沒有實體器件timing/BER曲線時也不能把28µs任意改小。暫不推薦當正式Round2候選。

目前結論：generic frontier可思考淘汰；pair/subcoalition functional materialization僅是較明確的可反證未知，尚未推薦實驗或宣稱新穎性。正式候選及其全新>=1800秒時計仍由root指定。

# Round2 正式 pair-bank 審查

## 20:38:00–20:42:34 UTC：Goldilocks 條件與最強簡單控制

Root已指定正式clock為20:38:00 UTC，最早21:08:00才可能實驗；此時計沒有挪用20:26廣搜或舊frontier sidebar。讀本輪`round02-pairbank/thinking.md`，機制固定為保留E originals＋有限M overlapping pair students，topK內disjoint matching，未涵蓋或未接受的pair回原expert。t conditioning可消融，不能把它偷換為bank機制的必要定義。截至此checkpoint，我僅做primary-web、source與代數審查，無candidate code、模型內容／activation probe或matching oracle。

### 真正可能成立的分離，及為何目前仍不夠

Joint-selection domain上的函數可壓縮，而solo domain需各自專門性，確實可能使global merge不合適、conditional pair view有價值；因此不能僅以merge prior封殺。但`F_i≈F_j`在joint域仍不足：**conditional representative**控制可在兩者同時入選時只跑一個原expert，乘`α_i+α_j`，solo與未配對照原樣；TRAIN選固定較優代表，幾乎不加儲存。必須勝過這個控制，不能只比較無條件pruning。

有用的Goldilocks例僅是存在性直覺，不是實際模型證據：joint域上`F_i=f+d,F_j=f-d`，相近權重使合併功能接近較簡單f，solo域仍各有需求。但若t分布很窄，固定barycenter student已可能足夠；若d可低rank近似，`S_0(x)+(t-t_0)D(x)`差分修正已是強控制；若d高複雜且t寬，student可能需要接近兩expert寬度，省讀取消失。沒有數值依據可先假定中間區存在。

其他必要控制為same-bank／fallback／capacity的parameter merge與unconditioned KD、同precision Pareto下原experts、相同訓練預算的lowerTopK／少active studentMoE，以及NimbleMoE／LorExperts／PCoMoE的同品質可行點。**一份W8、J=I學生與兩份W4 originals的bytes相同**；若W4原模型已保品質，不能把W8→單W8的表面二合一稱感測節省。低bit代價也須給學生；結論應比較整條matched-quality curve。

### 純代數容量／覆蓋帳（不是測量）

均勻topK時，M條bank邊的co-selection邊數期望是`M K(K-1)/(E(E-1))`，匹配數不會超過它。E64/K6/M128的值0.95238；samebit、J=I的plain3HI student，其routed coefficient saving上限因此15.87%。兩Down-head affine-in-t學生需4HI，其saving上限10.58%，額外bank儲存使總容量為`1+(4/3)(128/64)=3.667×`original bank，而plain學生為3×。這些不是路由實測，也非延遲bound；DeepV2還有約兩routed-width的shared投影，attention、commands、padding繼續稀釋。

兩Down-head若J≥1.5I則沒有raw coefficient saving；它還傳回兩個H維輸出供合併。把t作額外input座標使H2048→2050可破壞原row/page packing；hostFiLM／bias可保持H，但condition metadata、CPU work與數值誤差須付帳。runtime加權產生原weights必須先讀兩份original，沒有這個bank的NAND省讀取。

Batch是另一個必要性反例：單token少讀expert不代表batch union少讀。若不同tokens用許多不同pair students，原本共享的expert i可被拆成多個bank views，破壞ordinary grouped GEMM的read-once機會。可限定B1 decode論點，但不能把它直接外推prefill／多request throughput；若設batch-aware選擇，也要同能力原expert grouping control。

### 新核 primary prior

- [MEO, EMNLP2023](https://aclanthology.org/2023.emnlp-main.907.pdf)先以live gate權重合併selected parameters再做single expert。它仍讀原weights，故非本bank直接支配，但已吃掉generic『先合併再算』新意。
- [PCoMoE, EMNLP2026 §4](https://aclanthology.org/2026.emnlp-main.2048.pdf)已把GU與Down跨expert組合、保留所有diagonal fallback、learned compatibility／pruning，再source-grouped共用GU。它不預存pair-sum students；但twoDown-head方案若只靠共用GU省讀，必須勝過其可利用既有GU的控制，不能稱保留原路＋有限新routes首次提出。
- [Matryoshka MoE, EMNLP2026](https://aclanthology.org/2026.emnlp-main.1449/)以變動active K訓練支持elastic inference，是相同trainbudget lowerK控制的近prior；目前只核官方abstract，不冒稱已逐節看PDF。
- [Bandwidth-Efficient Adaptive MoE via Low-Rank Compensation §3](https://arxiv.org/html/2512.17073v1)已有低bit originals＋offline低rank residual、runtime按router Top-n補精度。不是pair-bank直接prior，卻使『只拿原W8』的precision control明顯不充分。

截至20:42:34尚未找到完全等同overlapping prestored pair-sum bank的primary，這只表示有限搜尋結果，不能主張absence proof或投稿新意。Root要求約20:48–20:50存完整receipt後暫釋slot給Terra做更專注prior審查，約21:00再喚黑臉；Root／Sol／Gem繼續正式研究，等待不當思考。

## 20:42:34–20:47:01 UTC：tagged feature 修正、兩個更近 prior 與交接門檻

### Sol 提出單 Down tagged-feature 變體：保留，不可被錯誤成本下界封殺

從兩original experts共2I個hidden neurons取J個，保存各neuron origin標籤。GU仍是J×H；host在SiLU×Up後依origin乘`t`或`1−t`，再送同一個Down矩陣。實數算術下結果對t affine，參數只3HJ＋J-bit tags，另有J次host乘法。因此前述4HJ只是**特定共享GU、雙Down-head**建構，絕非對所有affine-in-t學生的lower bound。已明確同步root／Sol，不能用10.58%那個特定建構上限否定全部pairbank。

這個變體本身接近conditional-domain neuron pruning＋Down ridge，不可把origin tag／hidden重加權稱主要新意。FullJ=2I也不自動等同canonical W8A8 teacher：先乘t、再做一份global A8會改掉原兩expert各自amax，量化亦破壞嚴格affineness。可把它明定為近似學生並實測品質，或保留per-origin quantizer／partial Down各自scale，後者須付額外commands與partials。不能將實數等式冒充原bitwise reference。

Pair fit樣本數與joint域偏移仍是主要反證未知。均勻E64/K6時每1024tokens僅期望7.6個co-selected samples；幾百／千維ridge可在少量train rows上插值，train低MSE沒有說服力。這不是n<J便不可能的定理：原Down初始化／小調整／低秩先驗可減sample需求，但須分別記錄真正teacher queries、每edge樣本分布、prompt獨立heldout、regularization／選寬／edge selection所用資料。模型free-forward的x、t、routes改變後，teacher固定trace的hit率不能當成功證明。

### NimbleMoE 現已核全文，比 abstract 更強

[官方PDF §3.1、3.3、Appendix E](https://aclanthology.org/2026.emnlp-main.2036.pdf)現在可取得，已直接讀forward與算法：保留所有experts／original router；G=2用max-weight perfect matching選固定groups；group共享input basis，expert-specific GU／Down latent仍各自算，live routing權重在共用Down output basis前聚合。它已針對co-selection降低active parameters，不能只當「global merge會丟solo」的弱反例。

尚未見它維持overlapping pair views＋原solo完整weights，但更強控制可直接是**同pair bank邊／同fallback的pairwise Nimble或低rank compression views**，只在co-selected時用壓縮view，solo仍用原expert。這檢查收益來自conditional materialization本身，還是本輪student/pruning方法。能被這個控制表達並不構成自動否決；若實測被其品質／reads Pareto支配，則沒有必要上新方法。

### 新近 prior IntBMoE：generic finite composition bank 已非空白

[IntBMoE §4.2、4.7](https://arxiv.org/html/2609.21346v1)明確用固定codebook建立有限、多expert參數合成blocks，可離線預算並cache，online只執行少量blocks；主要實驗為新架構訓練，並有causal language modeling。它不提供本候選的drop-in、pair conditional distillation與original solo fallback，所以尚非直接支配；但「prestore有限composed functions，分開participation／execution／materialization」本身已有。本文含shared支路也不代表有限bank概念依賴shared，不能以這一點迴避prior。

剩餘可研究缺口應收斂為：**低成本post-training轉換、重疊pair views、保留solo功能、在matched-quality與同physicalcapacity下減少ordinary reads**。若只把新的低TopK大MoE部署到NAND，便缺少這個分離。

### 暫時判決與下階段准入邊界

目前沒有足夠證據thought-kill全部pairbank，也沒有足夠證據批准實驗／投稿。最可反證未知仍是joint域可壓縮性×matching coverage×precision，而不是matching演算法本身。Root／Sol／Gem須持續到正式20:38起滿30min；此交接不使clock提前達標，也不擅自新增Round2 experimental failure。

若後續只做zero-error perfect-student matching／route上界，必須凍結bank容量、width／bitwidth、TRAIN與score分離、page padding與byte denominator；negative只能排除此有限設定，positive只代表有機會。若做local student fit，除固定學生／conditional representative／低rank差分外，必須清楚顯示train與heldout，不能把局部MSE或感測work代理當full model品質與端到端速度。

分類亦先明定：只有模型／native simulator在matched-quality、同資源下真正≥10%的工程效益，且勝不過直接prior或無新機制時，才列『≥10%工程但無投稿價值』；理想bytes上界≥10%不是該分類的量測。現階段未有任何本輪數據可放入成功或失敗欄。

## 20:57:40–21:05:17 UTC：恢復最終 gate，識別缺口與 constructive equivalence

Root於約20:58召回，同一候選clock仍是20:38→最早21:08。我已讀最新root dossier、Sol20:56後Stage0提案、Gemini Round02專檔Block2與Terra最後四節。另親自開primary核[CRISP §3.2–3.3](https://arxiv.org/html/2610.37442v1)與[SPICE §4.2](https://arxiv.org/html/2608.21240v2)：前者已按組合角色調原expert prefix寬度，後者已保original並用shared-base＋低rank residual surrogate／fallback。Terra補的Parallax與PuzzleMoE亦須尊重其training／fixed-partition範圍；沒有把它們錯稱完全相同pair student。

### 先指出 B-pilot 的實質不可識別性

原建議每預定layer選兩條distinct-node pair，只能測pair壓縮表示；兩條不共享node，不能測唯一殘存的overlapping topology／partner-domain價值。A中M128 perfect-student matching有headroom，與B中兩個disjoint pair可fit，合起來仍不能證明overlap bank必要。若保留pilot，最少需一個3-expert wedge `(i,j),(i,k)`、在fit路由統計上凍結，分開兩joint域、triple-selected overlap、i solo／其他partner域，並給rank／alpha條件single-expert compact view及CRISP同資料／讀取能力。

但這個修正仍不足救具體tagged-prune方法，因Sol隨後提出更強、我獨立核實成立的代數反例。

### Tagged 單 Down 與 partner-specific compact views 具有相同必要讀取

設tagged學生的GU特徵分為`φ_i(x)∈R^Ji`、`φ_j(x)∈R^Jj`，`J=Ji+Jj`，Down矩陣按origin分欄`D'=[D'_i D'_j]`。實數輸出為

`D'[t φ_i ; (1-t) φ_j] = t D'_i φ_i + (1-t)D'_j φ_j`。

因此可以把同一學生**逐位元拆成兩份依夥伴選擇的精簡原expert視圖**：各自存本來同一份Gate／Up rows與D'對應columns。總係數仍`3H(Ji+Jj)`，tags／selectors不增加必要權重read。這不是只說generic框架可表達：兩者可以使用**完全相同的packed arrays、page addresses、GU／Down command與NAND sensing序列**。

量化亦不能憑空製造新意。給control與candidate同樣的pair-specific Down row scales、同樣host tag modulation、同一concat後dynamic A8 amax／rounding，並在同GEMV內對兩個column blocks一起累加，則輸入bytes、INT8 weights、integer accumulation與最後scale運算都可一致。若強迫control各自獨立quantizer／獨立command，才會造出差異；那是人為限縮公平基準。若原生controller需新coalescing／continuation能力，兩方都得到它；將相同packed bytes叫single student或兩compact views不改物理工作。

這**不**構成所有conditional materialized views無價值的定理；partner-specific masks或共同Down擬合確可能比CRISP一套global prefix更好。但那個收益應歸條件壓縮／物化本身，不能再把tagged pair MLP、matching或原fallback當必要的新感測機制。單純不把這個直接control加入，就會讓ridge pilot假識別。

### 資料與訓練的可行範圍，不能用 underfit 裁全家族

200個本地TRAIN prompt依120fit／40valid／40score切分，是有限synthetic routing corpus，非正式task benchmark。若只collect8decode：fit960個獨立x、score320個，均勻E64/K6 pair分別僅期望7.1／2.4rows，無法支撐128→256 conditional fit或8prompt score要求。若用128prefill＋8decode上限，fit最多16320x，pair頻率需約1.57%才期望256rows；score最多5440x，頻率約1.18%才期望64rows。短prompt更少，prompt內相關也使effective sample size更低。這些是採樣算術，不是實際route統計。

頻繁wedge可能足量，尚未證明；synthetic t混合不增加獨立x。若資料不足／learning curve仍下降，最多記inconclusive，不能把fixedGU ridge的失敗當一般nonlinear pair KD不可能。原GGUF收x再局部重算canonical W8亦只能稱offpolicy operator-level screen，不是完整W8 free-forward品質。

### 21:05 暫定建議：具體 tagged-prune 方向 thought-reject，保留9次實驗

在本輪已研究的低成本學生族中，taggedGU＋Downridge相對partner-specific compactviews沒有可識別的新necessary sensing work。CRISP、Nimble、SPICE、IntBMoE又已覆蓋pair roles、共享／補償、原surrogate fallback、有限materialization等主要機制。當前A＋B pilot即使正面，仍主要顯示一個既有壓縮技巧在條件資料上可用；即使負面，也不能否定尚未研究／訓練的general nonlinear KD。因此我傾向**不啟Stage0，於21:08研究gate後正式thought-reject具體publication claim**。

這是價值／識別性決定，不是宣稱普遍不可能；沒有用「標題未找到完全相同先例」作放行理由。沒有本輪candidate數據、沒有≥10%效益可分類，也不新增科研experiment failure。一般nonlinear pair KD留為未證未知，若未來要重啟，需先有新的可識別機制與足量訓練設計，不能拿ridge陰性替代。

### 非採納的新物理軸備案（未開新clock、未probe）

向root僅提供兩個需先作thought screen的具體問題，而非聲稱已有新候選：

1. **Neuron-owned完整FFN融合**：把同neuron的GU、SiLU／local activation storage與Down column partial放同die，先算通信／amax barrier可省的wholemodel ceiling，再比SLIM同硬體fusion；若同array sensing work已主導且Host往返很小，立即淘汰，不能只為加LUT宣稱新意。
2. **Speculative input matvec＋late integer-delta repair**：在ordinary page仍在PB時對預測input先算，真input來時僅修少量changed quantized coordinates；真正未知是dynamic-scale下qx差是否dense、PB被後層sense覆寫後repair是否必須重讀全weights。若只是readyinputs批GEMM、保整expert SRAM或跨sense保WL，分別落回SLIM、隱藏容量或CR別名，立即淘汰。這與本輪pairbank無關，廣搜不抵下一候選1800秒。

## 21:08:08 UTC：Round2 正式最終 receipt — concrete pair construct thought-reject

Clock實讀`2026-11-02 21:08:08 UTC`，距正式20:38起點30分08秒。Root／Sol／Gem持續針對同候選作研究；我前段20:38–20:48、恢復後20:57:40–21:08有primary、代數、量化、識別性與資料可行性實質互審，中段釋slot由Terra作專注source查核，未拿等待冒充研究。**黑臉最終建議：THOUGHT-REJECT目前具體overcomplete tagged-GU/prune＋Down-ridge pair construct作為投稿方法，不啟A＋B Stage0。** Root已同意將依此正式記錄，experimental failure仍僅Round1一次，九次機會不變。

拒絕的可檢核理由是上述constructive stack equivalence、強先例與小pilot不可識別性，非推測所有學生都學不會。兩個partner-specific compact views可以用相同bytes／pages／scale／GEMV復現tagged學生。只有把control強制拆commands或獨立量化，才出現假必要性。General nonlinear pair KD沒有被邏輯否定，但沒有本輪可識別、可負擔的小實驗去建立其新機制；更不能用ridge fail否它。

已讀Gemini新增Block3並要求撤回過強部分：SPICE啟發的低rank control是強比較，不代表r64已有品質或實體wave支配保證；`α_j→0`只保證返回Fi端點，不是兩端exact；其餘expert能否由極小殘差近似仍未知。Root亦明確不同意這些無證推論。因此本判決不建立在這些過強語句之上。

下一方向只是**WL dual-page普通讀取的prior screen**，尚未採納、未開正式clock，前30min不轉移。Gemini從AiF單logicalpage28/37/46µs直接推joint32KiB46µs與免費ECC/MAC overlap，尚無command／latch證據；2個input latches既已占用，hi-cache／low-result／sense不能全當空閒。初搜已找到[US9851899B2 Fig.15A](https://patents.google.com/patent/US9851899B2/en)既有one-shot多logicalbit讀入main/cache/aux再輸出，以及[US9230677B2 claim9](https://patents.google.com/patent/US9230677B2/en)多bit read與cache數關係；這些不證Gemini的指定時序，卻阻止把joint latch讀本身當新意。已交root，將由Terra接續primary與PB資源核查。

本輪我沒有讀新model weights／activation、寫candidate或oracle碼、跑候選數值實驗。沒有≥10%本輪實得效益，也沒有將thought rejection假記科研失敗。

# Fresh Round2 local-affine 正式審查

## 21:29:35 UTC 起：新時計、primary核對與公平物理邊界

Root採納新的具體localAffine候選，正式21:26→最早21:56 UTC。這不沿用pairbank、WL-dualpage pool或21:18的廣搜。我於21:29:35實讀clock後恢復專題審查；已讀root新dossier、Sol新mechanism note、Terra新source ledger及Gemini affine Block1，僅source／primary-web／代數，無實驗程式或候選資料讀取。

機制固定為一層shared host投影`z=P_l x`，讀前按expert／anchor選map，單次ordinary in-flash GEMV得`F_e(c)+A_e,c(z-P_l c)`；原expert fallback亦須在任何昂貴map／original讀取前決定。這確實取消GU／SiLU／Down的原雙階段dependency，與上一輪tagged features不同，沒有發現把它拆回原teacherfeatures而保持相同必要work的反例。但Taylor／局部線性／共享latent／fallback均有直接prior，不能由原始DAG變小直接推出研究新意或品質。

### Primary核實的範圍

- 獨立讀[MoLAE §4–5](https://arxiv.org/html/2503.23100v2)與[Half the Nonlinearity §3.1–3.2](https://arxiv.org/html/2603.03459v1)：共享projection＋latent非線性experts、全局ridge MLP surrogate＋contextualgate均已存在。後者的global surrogate用centered regression/intercept，gate以downstream cross-entropy變化作標籤，是比單純距離／output MSE更強的選擇基準；不能將其GPT/Pythia結果冒稱本MoE品質保證。
- Flash-Jacobian兩正常公開OpenReview URL於我此時回browser-verification challenge，我沒有繞過；Terra先前已直接讀primary的筆記保留，但我不冒稱自己已核全文。沒有本地PDF，不為此停止其他審查。
- EBC publisher fullpaper仍不可取得。Root提供作者公開notebook之唯讀snapshot，我已核`source_cells.txt`34010–34130、34560–34625、35325–35390：每expert用`forward_mlp(X)`作global ridge，對線性矩陣的random-projected features作expert clustering，`apply_expert`的cluster由expert ID固定，不依當下x選anchor。因此其author-code是global linearized expert＋expert-clustered rotation的強控制，非context-local map bank。這不是聲稱已讀全文或作者歷史notebook必與paper最終配置一致。
- EBC這段ridge沒有常數input/intercept，故本研究最強global affine基準還應有center/intercept；不能靠補bias取得假local-bank收益。Source含NaN→synthetic-calibration fallback分支，但我沒執行、不知paper是否走過，不據此推翻其結果；本campaign若calibration NaN則應記invalid，不能靜默轉synthetic。

### 新增最強簡單控制：同 region bias，但 global slope

候選把C個`F_e(c)`放host並零flash取用，global基準也應得到相同selector、keys、C個anchor outputs與centering。至少比較：

`F(c)`；`F(c)+A_e δz`；`F(c)+A_e φ_e(δz,c)`；`F(c)+A_e,c δz`。

如此才分出「局部bias／查表」與「局部slope／output subspace」的貢獻。只給global model一個bias、卻給local C個host輸出向量，是資源與表達不對稱。Nonlinear host latent network＋global decoder的rank要依同physicalwaves選，不能固定為local的rank。若r=H而`P=I`，global/fullrank baseline可免無必要的H² host旋轉。

### Native rank／context契約比名義rank更重要

Root／Terra已指出g16 rowshard的r64與r128每expert皆16pages，padding使globalr128是最低sense-equal強控制。再強一層：若採保守合法契約——每chip一次僅一expert/input context、K experts順序——H2048的r512 map每chip64KiB，恰四planes一wave；r64/128也一wave。因此globalr512可與localr64同K次sensingwaves。它有較高ECC／MAC／host投影cost，故不是總延遲免費，仍應作明示對照。

若要讓小r局部map透過空planes在cohort內少sense，必須證明同chip能合法選不同expert／region地址、各plane持不同δz與獨立accum/context，並滿足multi-plane位址與命令限制。Input總bytes能塞入32KiB latches，不能證原生有K contexts。不能一邊給local任意plane packing，另一邊讓global g16串行。編碼、短向量padding及PE吞吐契約仍需source／硬體說明，不從ECC6.4GB/s推MAC。

### 小 bank 的 all-host 必要性問題

純W8係數算術：global r128 A全Deep約0.436GB；local r64/C16約3.49GB，OLMoE約2.15GB。這些配置有可能連nonFFN／keys／KV全部留在8GB主機，故須比較all-host與hot-map cache＋CPU decoder；不准強制每次一定讀NAND。CPU是否更快需測，沒有先斷言。更可能確需flash的區域如C32/r128，Deep mapbank約13.96GB、OLMoE約8.59GB，但calibration occupancy與host selector成本更嚴格。小bank低localerror不是已解flash必要性。

此checkpoint判斷：尚無與pairbank一樣的直接同work等價淘汰，但Stage0的最小可識別問題應是**conditional slope/output subspace是否在same native waves、同host memory/time下勝region-bias/global nonlinear decoder**，而非僅證某個局部Taylor能fit。研究繼續至21:56，不預先批准probe。

## 21:39:24–21:43 UTC：修正g16弱配置、free-intercept family與最小Stage0建議

先明確修正上段native敘述：r64的完整H2048×64 W8 map共8頁，可分在**g2兩chips**、每chip四planes各一滿頁；r128可g4、r256可g8、r512可g16。不同chipgroup各持一個input context即可並行不同expert，不需要我上段所說的同chip多expert／plane context。因此g16下r512與r64同sensewave只是弱配置的對比，**不能用作所有合法配置的no-go**。Root／Sol已指出並採用這個修正，我同意。合法固定副本仍需同block/pageoffset、command／padding契約；候選與全部controls都得到同容量與placement能力，不宣稱舊replica solver為新。

Root指出Deep C32/r64全map約6.98GB，完整複製到8個g2 groups約55.8GB，加原weights15.7GB仍低於名義333GB LSB；K6可各派不同group達1wavecohort。這是有條件的符號排程與容量帳，不是實測native latency。Globalr128只有4個g4 groups，即使同樣充分複製也需2waves；globalr512可能全host約1.74GB，因此physicalwaves不能獨立決定勝負。尚無samework等價能像上一輪直接淘汰這種真正改expert operator的家族。

### Primary應是local ridge自由截距，而非固定F(c)的Taylor

最小可部署family採每region的`zbar,ybar`，`yhat=ybar+A_c(z-zbar)`。Host輸出向量與原`F(c)`同量級成本，卻可吸收Jensen均值偏差與off-manifold anchor問題。固定canonicalF(c)的J-only／J-shrunk作次要歸因，不能拿其失敗替代所有localOLS失敗。CanonicalW8A8有round／amax非平滑，解析J永遠只是平滑initializer。

最必要control是同一C、selector、host ybar、centering，但只用**一個跨regions共享slope**。它須在每region去均值後把所有rows合併，重新最優fit：不是先對未分區資料globalridge、再事後補regionalbias。另一control是r0純regionmean查表。Globalr128／r512也得到相同key與regionalbias，但可有自己的更高維zbar；依同可達cohortwaves及host成本比較。如此才識別conditional slope，避免把C個免費bias與一個globalbias比較造成假優勢。

### 給Root之21:44前bounded pilot建議（不是launch授權）

- 每模型4個事前depth layers×2個只按FIT頻率選的experts（Deep有shared、OLM無shared）；不按MSE／改善率選expert，不聲稱這8ops覆蓋全模型。
- 主family shared PCA r64，C1／4／16，lambda只VAL選；r128／512 global controls、regionalmean與sharedslope不可豁免。r128/C16可列容量壓力secondary，但欠定時不得當decisive negative。
- 全部真正W8 row-scale maps＋每region deltaA8後重測；floatridge與量化版本分報，避免浮點fit好卻無部署品質。Original canonical W8A8 teacher為真localoperator，但x來自原GGUF free-forward，明稱offpolicy operator pilot。
- Sol提出的host projectedGU＋regionoffset，保留SiLU×Up、flash讀原Down，可作免SGD的非線性曲率控制；完整MoLAE／CEguard／W4匹配品質／all-host實算與wholemodelquality可在positive後**同一輪**延伸，不要求必要條件pilot一次完成整篇paper。但localpositive絕不構成quality／wholemodeltime成功。
- FIT N→2N curve、每region row／distinctprompt數、條件數、有效自由度必報。固定SCORE只用最後一次，不用SCORE選rank／C／lambda／guard。不足則inconclusive，不用underfit裁一般family。

資料量是現有限制的真風險：120FIT prompts、每prompt至多128prefill+8decode，最多16320個layerstates。假設某expert頻率p=.1（僅算術假設，未probe），C16平均102rows／region，對r64並不寬裕；C4約408。若以4r rows作有限樣本heuristic，r64/C16需N約40960，r128/C16需81920；這不是統計定理或自動hardgate。40SCORE prompts只8decode則總320states、該expert約32rows，對C16平均2rows／region，不能據此宣稱regionguard品質／coverage。Prefill可供fit，prefill與decode的score分開，不把prefill可擬合冒稱decode品質。

讀Gemini Block2後，主要數學／硬體修正已正確納入。我不採其『Stage0必完成全模型PPL／free-running實算延遲與全部prior』的鐵律，因這會把bounded necessary-condition pilot誤要求完整paper；也不接受『J-shrunk克服n<r』的品質保證，prior只能regularize，不能製造缺失資訊。迄今沒有新candidate資料、沒有實驗程式、沒有模型權重probe。

## 21:44:12–21:48 UTC：Root新版shared-balanced/all-x family與判負邊界

Root吸收occupancy問題，提出新的**同一正式候選內具體family**：每層只在FIT x建一棵recursive-PCA median tree，C1／4／16／32 leaves由所有experts共用；共同leaf meanX，故同token所有K experts用同一`delta=P(x−meanX_leaf)`及同A8 input。每個預選expert對**全部FIT x**離線query canonical F_e(x)，而不只route-hit rows。預選縮成Deep layers1／13與OLM layers0／8，各取FIT頻率最高與中位expert，共8ops。Original-source x、FIT-only tree、共同input、freeintercept localridge及讀前fallback維持明確；尚未授權candidate code或probe。

我已於21:44前回Root：這個改版**值得bounded Stage0**，不因Flash-Jacobian／MoLAE各積木已有便直接thought-reject。它不是新Taylor／新局部回歸原理；可識別的是shared-input local operator在finite host/flash下的實體化與activework tradeoff。主要r128/C32 bank約13.96GB Deep／8.59GB OLM，不能假設整庫全放8GB host；較小r64/C16仍需all-host對照。這是科學scope支持，最早21:56 gate前不構成launch許可。

All-x query把每leaf名義training rows提高到約480–510，解除了r128最明顯的parameter-count欠定；代價是每expert都query非自然路由inputs，必完整計離線成本。**480個all-x rows不是480個自然routed-domain rows**；SCORE按自然命中評估時存在covariate reweighting。權重w=1或routehit4／others1可由VAL選作明確family，須報effective N與distinct prompts，不宣稱這已解所有expert-conditioned localOLS的資料問題。若陰性，限定否這個shared-balanced/offroute-query/frozen-ridge family；不可擴大為expert-specific regions、充分routeconditional訓練或任意nonlinearstudents不可能。

Balanced tree須按真runtime rule重走FIT／VAL／SCORE核每leaf counts。若以sampleID拆median ties，會使相同x落入不同leaf，無法由部署input重現；不准把TRAIN人工balance當可部署balance。Exact repeated x需同路徑，空／過小leaf應明報或按frozen fallback處理。這點Sol亦獨立提出。

### Quantized free intercept 的公平性

主fit應使用真正decoded-A8 input `v=sx*q`。雖未量化delta以leafmean居中，quantized v的FIT mean不必為零。故不能僅固定ybar；真正freeintercept在最終W8 map後重算，概念式`b_c=ybar_c−A_q vbar_c`。若用actual F32 scaled-integer GEMV predictions的FIT mean作recenter，可避免非結合rounding的小差異；全部family給同freebias與量化順序。Sharedslope仍需pooled within-leaf demeaned optimalfit，而非先globalridge的弱版本。Normal／variance-balanced P可僅在VAL事前選，SCORE不得回選。

### 可接受的Stage0科學陰性，不使用任意2–5%門檻

Root詢問same-r local未達2–5% normalizedMSE改善是否可判fail。我明確建議**不把這個未與NLL連結的數字當科學否定**：1%局部改善也可能在真正qualitycliff有用，而大MSE改善亦可能無PPL價值。

支持『任何可信、非受支配operator點→同輪Stage1』。可以接受的限定Stage0 fail是：對凍結的8ops／SCORE分布／family，所有VAL-selected local configs在真正量化後均被同tree/freebias controls的operator error–activework／host／retainedbytes envelope支配或數值完全tie，且prompt-paired不確定性足以排除改善。`lambda=∞`退化pooled而finite lambda顯著較差，是清楚的conditional-slopes未買到收益案例；此時不要為相同slope重複存C份map造出假成本。

若只排除『>2%』而仍允許微小正改善，應記實務screening negative或inconclusive，不冒稱operator Pareto fail。WideCI、learningcurve60→120仍下降、mixedops或decode coverage太少，維持同輪補資料／縮scope，不能另算新fail。即使取得bounded operator fail，也不表示已否完整模型NLL或所有local-affine族；收據須保留這個邏輯限度。

另重讀AiF原文§6.1：其In-Memory control用i9-14900KS、DDR5上限86.4GB/s，AiF host8GB；小mapbank的CPU/hotcache不是虛構比較，原文亦允許tensor-level host/flash並行。這是已知primary設定，不是本候選CPU實測。不能用NAND理想waves代替同host處理能力的真正成本比較。

## 21:50:18–21:55 UTC：完整protocol draft審查與統計closure

已完整讀`round02-local-affine/stage0_protocol.draft.json` draft001及draft003。Primary8ops／r128C32、same-tree r0／pooled sharedslope／higher-rank controls、actualW8/A8再以FIT residual mean重置bias、offroute weights、2/3→full coefficient curve、source-GGUF x/offpolicy、ordinary28us noCR、host/capacity與partial shared barrier限縮均明確。Draft後續還須凍source／build／synthetic／統計精確算法，**思想gate不是模型執行gate**。

本輪要求並已納入的closure包括：

- Shared/global pooled prior自己的`eta` zero-target ridge與local shrink-to-pooled的`lambda`分開，前者eta∞=zero，後者lambda∞=own-basis pooled；不可形成循環定義。
- Normal／variance-balanced兩個完整預列配置皆報，actualA8 design各自重fitprior；full-H identity是另一固定control。不能把不同basis矩陣直接轉當相同量化design。Whitening各用isotropic ridge合法，但明稱不同regularizers，不冒稱純量化或座標等價。
- VALID若選lambda∞，仍保留best-finite-lambda-by-VALID候選於SCORE，避免empty finite set的vacuous scientific failure。Literal同loweredmap/scale/bias可直接記function tie；C重複global maps要去重計成本。
- trace0／singular leaf／missinghits／zeroenergy／undefined bootstrap均有明定zero/prior/fallback或inconclusive，不因實作異常記科學fail。
- Tree median ties按runtime input-only rules重走；任何depth5路徑的PC是node-specific，不等於全局top5PC。Balanced／N>=2r是條件性診斷，不保證條件數或generalization。
- Deep shared2816不改時，**整層仍保留shared GU→Down barrier**。本方法只刪被替換routed expert的dependency；無shared OLM亦只有全selected均map且無fallback時才可能全層onephase。
- Guard使用讀前full-x distance與VALID quantile cutoffs，不能按SCOREerror挑outliers；要同報coveredloss、exactoriginalfallback後total loss與mixed DAG成本。Guard的lambda選擇規則與cutoff `<`／`<=`需sourcefreeze精確指定。

Root問512左右比較、10k bootstrap若Bonferroni尾端不足一次draw怎處理。我支持預列整族比較的**paired max-|t| simultaneous bootstrap10k**，同40prompt IDs跨所有候選／controls／models共同重抽，重算ratio差；以bootstrap SD標準化中心化差，取max統計95%分位c構造同時帶。所有會用來判支配的預列比較均入族，不能看SCORE縮族。這是40prompt下近似empirical uncertainty，**非distribution-free／有限樣本FWER定理**。非algebraic tie之zero variance、undefined分母或稀疏route則inconclusive，不默默丟bootstrap重複。Sign不清楚保持同輪，不拿bootstrap方法本身當必能fail保證。

已提醒Root不要接受Gemini仍殘留的『8ops local-MSE被支配絕不可能全模型翻盤』。Subset在相同operator criterion下的envelope單調性成立，**不**傳遞到其他layers、signedcross-expert誤差、自由軌跡或NLL。Root draft已正確保留這個界線。

新增的samefamily強control也合理：把fit好的C個local A串接作output SVD，保留共同U_q與host條件B_c，runtime只讀U_q。它直接測conditional output-space rotation是否真需要更多active rank；q=r128時B W8約.872GB Deep（FP32約3.49GB），可能hostfit，U約.436GB。q增大會同步花host B／inputposts與decoderwork；q到H時B可超host，故不是無條件samework等價淘汰。所有factor須真W8/A8重新量化，不從實數SVD冒稱bit-exact。這個control被列在positive後同輪擴展，不能因Stage0未跑便先claim勝MoLAE／非線性控制。

截至本段仍零candidatecode、零新model／activation／routing probe；只source／primaryweb／數學審查。最終gate等待實讀21:56與最新protocol hash，時間到不自動通過。

## 21:56:07 UTC 正式 gate receipt — LIMITED GO implementation，尚非run

實讀clock為`2026-11-02 21:56:07 UTC`，距本候選正式21:26起點**30分07秒**。Root／Sol／Gemini均持續專題研究，我21:29:35起至本gate完成primary／author-source、代數、host/flash公平控制、minimalfamily、offroute資料、quantized regression、bootstrap及完整draft互審；沒有沿用pairbank／WL pool時計，也未在思考期間寫candidate或讀新model數據。

本次已核草稿`results/research-2026-11-03-no-cr/round02-local-affine/stage0_protocol.draft.json`：

- revision：`draft-003-before-data`
- SHA256：`7fd4a15d98da1c207dc5ad8945a058209ae5142fb1c707105bdc2022bbb78d16`
- status仍是`not_frozen_not_launch_authorization`。

**黑臉決定：LIMITED GO，支持Root凍結本bounded Stage0後，由指定executor開始implementation／build／synthetic checks。** 現無blocking科學設計缺陷；不因localridge／sharedprojection積木既有而空泛拒絕。共享tree/input與conditional outputmaps有真operatorwork／DAG變化；最小控制集可回答其局部斜率是否在所測operator分布買到收益。這不是投稿通過，不是已證quality／latency，也不是單靠時間到自動准入。

**本receipt不授權實際model／prompt capture或fit。** 那些仍需immutable source／binary／library／protocol freeze、synthetic arithmetic與runtime-tree驗證及獨立source-review receipt。Gemini closing feedback若產生數值或方法實質改動，Root須送最後diff/hash再核；本gate不回溯批准未看的變更。

判讀凍結：任意2–5% MSE門檻不構成科學fail；所有凍finite-local candidates被更省work/host/storage控制在同operator criterion下可信支配或exacttie，才可記這個bounded family負結果。Mixed／sparsehits／undefinedCI／未穩learningcurve保持同輪inconclusive；可信non-dominated點須同輪展開真正fullmodel品質與完整強基準，非成功。局部MSE陰性不傳遞成全模型NLL、signedcross-error或所有localmapping不可能。

Campaign截至gate仍**1次有效experimental failure、9次機會**；本候選尚無實驗。上述科學gate已同步Root，筆記完成後釋slot待source-review followup。

## 22:09:49 UTC 起：localAffine early adversarial source review（未final）

Root回報Gemini四個focused blocks自身CLI研究累計1827.9s、final closing於22:01:08結束；Root正式研究已36min以上，凍結`stage0_protocol.frozen.json` frozen001 SHA256 `9254e7fdf71f965b8a8716a35838522fb378ca2421c6779d9650e385aad62dc8`。與我21:56核draft003只差revision／status／frozen_at元資料。Executor獲准implementation，**尚未獲model／prompt讀取、capture或fit許可**。

本次只讀C++ capture／numeric、Python core及pinned upstream graph／vec／model sources；未讀模型權重、activations、routing新資料或prompt內容，未執行candidate程式。Parent負責pipeline資料split／VALIDselection／SCORE入口，我負責core numeric／stat／cost及C++semantics，不重複分工。

### 已核合理的語意

- Capture callback只讀tensor，不見任何graph write。UP `GGML_OP_MUL_MAT_ID`的src1在兩模型pinned graph是共同未加router權重的x（`weight_before_ffn`僅LLAMA4），shapeH2048×1×T；IDs來同op src2，可配相同選擇順序。
- Alpha callback會隨`weights`／`norm`／`scaled`後繼tensor覆寫成最終權重。Pinned OLM model明確`norm_w=false`，因此不能對OLM selected-top8權重強求和=1；Deep由native hparams norm/scale決定。需final shape/event closure，但未發現錯誤把alpha先乘進x。
- Numeric用完整I調pinned `ggml_vec_silu_f32`後F32乘Up，逐bits對照同長`ggml_vec_swiglu_f32`，符合canonical activation contract。Sw*Sx先F32、dot轉F32後乘scale順序正確；FP64 BLAS只算integer-valued dots，exactI64抽查與2^53界有hook。
- Python core的within-leaf weighteddemean、pooled statistics、own-basis prior、minimum-norm correction、shrink-to-pooled、最終W8實際predict後weightedFIT residual mean校bias基本符合frozen規則。True runtime tree用storedF32 normal及sequentialFP64 dot判定，FIT重走實際selector，無sampleID拆同x。
- maxT core採共同40prompt重抽、重算ratio差、centered／bootstrapSD標準化、max95%quantile；undefined會全族inconclusive，沒有默默丟draw。這很保守，但不會由稀疏hit造科學fail。

### Early需修／final需閉合事項（已送Root及executor）

1. **完整weight adapter closure**：目前export僅GU input C2048／Down output R2048／E64，缺`Gate_R=Up_R=Down_C=Deep1408或OLM1024`及ne3=1的完整assert。Pinnedmodel/API還須確認沒有忽略bias／scale2／mergedGU；若Python runner補manifestassert也可。不能把staticH/E檢查稱完整GUDadapter。
2. **threadlocal float環境**：`la_environment`只檢caller；OMP workers的FE_TONEAREST／MXCSR FTZ/DAZ未驗。需要實際8threads synthetic環境receipt及parallel-entry failclosed，避免BLAS／GGML或先前worker狀態污染canonical scaling。Root亦已獨立提醒。
3. **C++ no-mutation vs trajectory數值證據**：static `graph_tensors_written=0`只是宣告；source的確無write，但callback分割graph可能改fusion／計算順序。Root同意僅固定每model第一FIT-ID一題samecp/no-callback control，比8greedyIDs及lastprefill＋8decode全vocab bithashes。不是雙跑200，不改science；微差先當tool/numeric定位，不能算localmethod科研fail。
4. **成本漏洞A，需修**：`cost()`以`rank==H`一律設hostprojectionMAC0，會錯低估full-H normal／balancedPCA對手，僅fmt=identity可省H²投影。
5. **成本漏洞B，需修**：`mapbytes=H*rank if any unique else0`把zero-map leaves也收完整active讀/MAC。若拿candidate這個上界宣稱被control支配，可能假陰性；須按自然eval leaf/guard使用率算真正nonzero mapwork。Stored全bank與actualactivebytes分開。
6. **Literal ties的範圍**：core接收`literal_ties`，pipeline必同P/format、tree/center、guardcutoff、qmaps／scales／bias、F32運算順序才稱部署函式相同；只modelarray hash相同不足，因不同basis可能餵不同qinput。Parent負責pipeline終審已獲提醒。
7. `cost`目前retained_map_scale_bias欄位名誠實但只是partialoperatorledger，不含hostP／tree／originalfallback／commonprojection amortization，不能拿作完整8GB／nativecohortwaves證明。Same-r R1控制可直接比較許多共成本；higher-rank/host controls仍需完整resource ledger。

22:13:28讀到的**非final source snapshot** hashes：capture `076fed5425eb56354f5658c45046facca3dc581abb60f0172b32545dea7f9253`；numeric `686e738ab3e9d4b83871ec67f847ca4ba8db2af5ab3c0e58811705320ece688e`；core `ea451314025045898a41b43a5c10825a2fe046af6f299d4d221867cdfceaaeef`。Executor仍在寫與修，這是early review而非immutable source receipt；編譯/include/type錯誤不作科學fail。

## 22:28:56 UTC 起：rev002 immutable source／synthetic正式審查，先HOLD等rev003

Root要求核`stage0/latest_ready.json`所指rev002，只讀source／receipts、不執行模型／prompt／candidate。已在22:29逐一SHA256核freeze_receipt列出的**29檔當前內容全部match**，並另核**6份immutable source snapshots**均與對應receipt/current source一致；包含pinned原靜態libraries、上游graph/model/vec source、numeric bridge／worker／capture binary、buildargv、protocol與algorithmmanifest。未讀model tensor內容或promptbody。

Rev002來源／合成收據宣告在任何Round2 candidate data前建立，protocol仍`9254e7...aad62dc8`。已完整讀immutable numeric、core、capture、synthetic、algorithmmanifest、buildargv及與cost/stat有關的pipeline；Root平行核runner／data gate／split／VALIDselection／SCORE入口。

### Early findings在rev002的closure

- **兩cost漏洞已修**：hostprojection只對identity／none為0；activework按自然eval、guard、leaf與nonzero lowered map比例計，不再收zero-leaf完整Hr。Retained P/tree/centers/threshold/guardmetadata亦列，並明標projection是layer-shared不可按expert加總、沒有physical latency claim。
- C++已assert完整Gate_R=Up_R=Down_C，I只容1408／1024，ne3=1，拒同prefix額外bias／scale tensors；Python load_weights再按model-specific I與weight/scalefile sizes核。UP commonx與naturalID／α語意仍與pinned graph一致，capture不改tensor。
- No-callback控制已固定每model第一FIT一題，以直接memcmp比last-prefill及8decode全vocab F32 bits、8greedyIDs，再記FNVhash；control mismatchfailclosed是tool/numeric state，不是科學fail。
- la_workers_environment實際8threads核FE_TONEAREST／FTZ/DAZ，parallel rescale與activation入口重查；compile flags`-fno-fast-math -ffp-contract=off -march=native`，pinned vec ISA保持。
- 原靜態GGML library非PIC，rev001 build故障保留；rev002把purePIC arithmetic bridge與samepinned static-linked activation worker分離。Worker IPC為int32 N/I＋F32 gate/up，status0＋F32 product，schema與partial-read loop明確；binary stdout不混diagnostic，full-I逐bit對fused GGML，關stdin並requireexit0。這是transport修補，不改算法／teacher。
- Synthetic收據passed，包含10765 near-half cases、DGEMM signed/random widths64／128／512／1024／1408／2048／10944、syntheticGUD I1024／1408、runtime ties／unsplittabletree、pooled/minnorm/trace0/shrink端點、actualA8/W8 bias／dedup、兩costbug、linear guards、40prompt maxT與undefined／SD0／literal ties、fallbackloss與emptydecode。審查者只讀收據與測試source，未重跑。

### rev002仍須hold的確定wiring bug

Root與我獨立發現`load_model()`沒重建`map_hashes`，但`score_stage`新增SCORE成本會呼叫`cost(model,...)`取該key，造成KeyError。**不能核rev002run**；這是tooling fault，未讀candidate data，不是科學fail。Root已要求rev003補loadhash與archive→load→predict→cost／小SCORE synthetic smoke。

我另找到behavior archive identity edge並已同步Root／executor：`archive_model`的key與model_hash只含qmaps／scales／bias，不含`unsupported`。若不同80／120 fit恰有相同loweredarrays但unsupported葉不同，後存者會重用先存npz的unsupported，改變fallback行為。應把unsupported納入archive／point／model behavior identity（或驗existing exactsame再分key）；literal functionties也要matchunsupported。需一個同arrays、不同unsupported的synthetic不共檔檢查。這不是新算法或擴資料，而是不可變artifact對runtimebehavior的閉合。

### 最終科學判讀保留的成本限度

比較control在VALID按cost eligible選定，SCORE自然leaf頻率可使zero-map activework順序反轉；所以最後判dominance必重核**同一個凍control**的SCORE成本仍不高，不按SCORE另挑control。Pipeline已輸出SCORE_costs，可供此核。SameC／sameguard的fallback fraction通常相同，但不得由假設跳過zero-map分布檢查。

Stat core共享40prompt draw、重算ratio、centered maxT與全族undefined→inconclusive正確；仍只是近似empirical bands。Literal ties目前sameC／guardcutoff＋sameformat/rank/hash或constant-outputhash，有同operator樹/P的前提；unsupported也須加入上述behavioridentity closure。沒有auto科學fail/success。除此目前未見新的blocking numeric/stat/primary-fit缺陷，等待rev003窄diff與最新hash，**尚未批准FITVAL資料執行；SCORE仍sealed**。

### 22:38前closing conceptual edge：C1也可因two-stage regularizer改善，非bank證據

Root提出正確細節：C1的local shrink並不必等pooled R1，因prior本身可用eta>0 regularization。設同一demeaned covariance S、cross C，`A_p=(S+eta_bar I)^−1 C`，則

`A_lambda=A_p+(S+lambda_bar I)^−1(C−S A_p)`。

沿eigenvalue d的filter為`g(d)=(d+eta_bar+lambda_bar)/[(d+eta_bar)(d+lambda_bar)]`，一般不等單一ridge`1/(d+rho)`。因此C1-only positive只能記global-surrogate calibration signal，不能證conditionalbank必要。C>1亦可能部分受這項regularizer能力影響，Stage1應給同tree／freebias的global pooled two-stage correction（與local相同eta→lambda能力），再比較conditional slopes／common-output SVD。**現在不擴Stage0凍grid合理**：positive只准同輪續研，從來不算novelty／quality成功；較小control集的可信operator-negative仍是該限定criterion的負結果，不是wholemodel theorem。

Root另告知rev003將把hostP由FP64 BLAS改為明定C++ fixed8-lane F64 reduction，處理batch形狀改變可能造成的association／F32-rounding差異。支持此data前numerical-order閉合，但algorithmmanifest／sourcehash須更新並保留rev002；synthetic應驗normal與balanced projector batch-vs-one bit equality。這不改local ridge方法族，不重啟候選時計；仍等最新immutable READY後才sourcegate。

## 22:47:52 UTC：rev003 正式 FITVAL-only source approval

已核最新immutable rev003並再驗33個receipt-listed current files與6份source snapshots**全部match**。Freeze receipt SHA256為`e2f6818ed3669ca09d6b898ad5d32e4f69fa4fa058137a5d79a5594963950977`，scientific protocol仍`9254e7...aad62dc8`。舊rev002 manifest實體歷史副本hash為`60d2acce...f783d7`，正合其舊receipt；rev003 manifest `3aee5141...1083efa`亦match，差異僅明定hostP運算順序。

**APPROVED rev003 FIT／VALID source，無blocking findings；SCORE仍sealed。** Loadhash、unsupported入behavioridentity／literal tie、archive roundtrip、H2048 SCORE synthetic smoke、fixed8lane hostP及passedreceipt均已閉合。Root另行獨立核runner／split／VALID選擇／data gate並決定launch；本審查沒有執行candidate或讀model／activation／routing／promptbody。

完整獨立收據：`stage0/critic_source_review.rev003.json`，SHA256 `a868ca3b5c95f560674e5b1fc8dbdd159af1ff87d6bd408e8416a13a8c38ef33`。其中保留40prompt bootstrap近似性、C1-only屬globalcalibration非bank證據、SCORE重核同凍control成本、offpolicy8ops不推全模型NLL／latency等限度。若current sources偏離latest approved receipt需hold再核；預定接續FITVAL與SCORE artifact審查，不重做候選thinking，也不將前述source修補計科學fail。

## 2026-11-03 02:06:56 UTC：FITVAL closure 通過，准許同一凍結 SCORE

獨立核1961份FITVAL檔案、5,716,605,717 bytes全部hash一致；rev003的33項source receipt亦一致。FIT120／VALID40的hash排序、兩模型每job兩層完整retained prefill＋8 decode、FIT-only八expert選擇及兩個first-FIT no-callback bit-exact controls均閉合，未讀SCORE／TEST或執行candidate。

僅用保存JSON獨立重建6136單operator＋4576合成比較，合計10712，VALID-only cost-eligible control及所有IDs完全吻合；16個inconclusive來自C1 selected lambda-infinity的複合缺項，best-finite保留。**APPROVED只開既有凍結SCORE；沒有blocking findings，不是方法成功／失敗。** 整個comparison family不得因bootstrap undefined而事後過濾；SCORE須重核同一凍control的實際成本，不可用SCORE重選。

C32真葉484–501 rows且primary VALID無unsupported；但Deep L13最低rank68、最高condition約6.98e8，樣本數不保良態。所有primary 80→120 FIT學習曲線仍改善約0.0138–0.0688 normalized MSE，故之後陰性不能自動算family failure。局部MSE不推全模型品質／延遲，C1-only改善不證conditional bank必要。

正式獨立收據：`stage0/critic_FITVAL_review.before_SCORE.20261103T020656Z.json`。FITVAL complete SHA256 `836b4a1a...516a2a6`；frozen comparisons `dfc69821...17c38`；protocol與source revision未變。Root另建bound release並決定實際執行，留待同輪SCORE結果審查。

## 2026-11-03 02:25:01 UTC：同輪 Stage1 強控制 theory-only gate

讀完Stage1 draft、Sol／Gemini控制推導及Root修正，**支持小範圍 G-RRR＋PE-RRR implementation design**，不是source／data-run批准，不開新輪、不改Stage0凍比較。主點維持八個unguarded primary、q128 FP32／q512 W8；primary eta0只需pooled OLS二階數值closure，不重跑eta×lambda大grid。G測expert-specific full-input資訊，PE測共同output basis；兩者是必要識別控制，不是新穎性本身。

Mtarget推導正確，但最適範圍僅固定float ridge predictions的output projection，非unrestricted joint RRR／quantized最優。必須用actual-dequant A8的**within-leaf weighted demeaned** Z、Y；最終bias則由actual runtime原始v校準。G的global slope仍含contextual中心／bias／quantizer，不能叫字面F32x的global affine。W8後oracle floor如要約束部署，須對actual dequantU的column span用QR或U U-dagger，不能拿已改span的理想U floor當界。

新增learning陷阱：U依teacher targets學得；N80曲線須只用N80 targets重建U／M／A，不能借N120 U只refit B。既有fullFIT unsupervised tree/P convention可分別明示，兩者不可混同。Gauge、eigen／rank規則、host F32 vs native A8路徑與比較族須在執行前凍結；all-host latent不用被強迫額外A8。

把「表示上C full-output maps必要」與「實際系統優勢」拆開：同error＋小storage可反例前者，若CPU較慢仍是Pareto取捨。Inclusive8e9／8GiB、KV／workspace、全bank cache、P／bias／scale、shared branch放置和CPU/NAND合法重疊都付帳；小8-op EPYC warm timing不能當全模型AiF host。任一固定rank失敗不殺所有factor模型，operator SSE不推整模型品質；改善中曲線不能自動記Round2 fail。若存可信non-dominated bank點可同輪前進，不必嚴格贏每個control每個metric。

獨立收據：`round02-local-affine/critic_stage1_control_review.20261103T022501Z.json`；本次只讀四份theory文件，未讀SCORE outcome／TEST，未執行implementation／probe，未改候選碼。

## 2026-11-03 02:36:11 UTC：原凍 Stage0 SCORE 科學判讀

Root在SCORE正常完成後明確開放結果審查，TEST仍sealed。獨立確認6664 point metadata與10712 comparison原欄位／順序完全等於pre-SCORE freeze，未改族。整族undefined的唯一root是Deep L13e10 decode：僅9hits／8prompts；獨立用seed20261103重建10000個40-prompt draws，正好0-based2748、5167兩抽全部漏掉8個positive-energy prompts。384單op＋764含該op composite＝1148直接undefined；另8126比較被整族規則連帶inconclusive，1438 literal ties，沒有nonliteral zero-SD。其中174真functiontie亦依既定undefined優先規則不救；不刪draw、不事後拆prefill／模型子族。

八primary兩phase所有描述性local−frozenR1差均負，但**沒有有效simultaneous bands**，不能因此宣告正結果。獨立核same VALID-frozen control於全部16個SCORE比較仍cost-eligible：active與hostP均262144 MAC；local retained10,478,072B、R1 2,097,656B，primary無unsupported／fallback。這是logical單op帳，非native波次／全模型延遲；共享P不可每expert重複相加。

decode絕對normalized SSE達0.0307–0.7454，prefill僅0.0091–0.1387；不能用先前mixed／prefill VALID range代表decode品質。N80→120 SCORE prefill八點都改善，decode七點改善，Deep13e56由0.45996升至0.47950；仍不是plateau。FIT約94%prefill rows，這是bounded校準目標風險，非local方法必死證明；若之後改decode校準須明示同輪development和新heldout，不偷改已凍Stage1。

**結論：inconclusive，繼續同一Round2，新增有效失敗0。** 原凍結果保留，已於02:31:18凍好的Stage1 controls不依本結果加格或換control；source／data gate另核。局部error、表示容量與全系統quality／latency分層，均無publication success推論。正式獨立收據：`stage0/critic_SCORE_review.20261103T023611Z.json`；未讀TEST、未執行candidate或改source。

## 2026-11-03 02:54:53 UTC：Stage1 rev002 core／FMA source gate 暫hold一項

先完整讀immutable rev001 core/FMA/synthetic，再核final rev002窄diff：core／FMA完全相同；synthetic只加incomplete composite檢查。三份current／snapshot／receipt hash均吻合，synthetic passed receipt已讀，未由我執行。Centered decoded-A8 statistics、same-N supervised Mtarget、固定ridge projection、gauge／實際bias、兩階段host／native順序及archive reload／unsupported identity未見其他blocking數學缺陷；Root另核runner／成本／資料gate。

唯一具體closure：FMA `environment()`只檢`fegetround()==FE_TONEAREST`與MXCSR FTZ/DAZ；本機libm `fegetround`反組譯明確是`fnstcw`再mask0xc00，僅看x87，沒有看AVX/FMA實際用的MXCSR rounding bits0x6000。故不能支撐manifest「每call檢RN」。已請Root只補新FMA guard的MXCSR RN mask，保留Stage0所有immutable碼；窄rev003＋既有synthetic即可，不增科學grid。**沒有證據既有資料或synthetic真的用錯rounding**，這是執行前contract檢查缺口，不計科學失敗。

Oracle另限縮：actual U已正確dequant；SVD relative cut1e-10會產生truncated span，若丟真非零方向，residual不一定是完整actual span的lower bound；FMA rounding亦可離開理想real span。只作限定diagnostic。Baseline rebuild hash只診斷，不可未解釋material差異就叫same frozen estimator。獨立receipt：`stage1-controls/critic_source_review.rev002.20261103T025453Z.json`；暫hold FITVAL批准，未讀新資料／TEST、未執行candidate或修改source。

## 2026-11-03 02:57:34 UTC：Stage1 rev003 FITVAL-only source approval

窄diff已核：FMA guard補MXCSR0x6000；focused synthetic在x87仍RN時逐一設三種MXCSR non-RN，environment與actual FMA entrypoint均拒絕，恢復原狀後全既有synthetic通過。Core僅oracle comment／輸出scope更正relative-SVD截斷與FMA rounding限制，科學公式／grid完全未變。三份current／immutable snapshot／receipt hashes一致；receipt SHA84299da2...fad550e，synthetic SHAd5b7d121...0f59ee。

**APPROVED rev003 FITVAL-only source，無blocking findings。** 舊hold已閉合；Root另核runner／58deps／資料release，無SCORE_DEVELOPMENT或TEST批准。保留baseline重建hash只是diagnostic、Mtarget非unrestricted RRR、operator／representation／whole-system結論分層。正式收據：`stage1-controls/critic_source_review.rev003.20261103T025734Z.json`。未由我執行synthetic／candidate，未讀Stage1資料或TEST，未改候選碼。

## 2026-11-03 03:14:47 UTC：Stage1 FITVAL閉合，SCORE_DEVELOPMENT release批准

FITVAL正常完成，獨立讀小份closure artifacts而不重hash全upstream大檔。288個raw VALID lowerings全finite、160個selected IDs唯一，逐組重建VALID最小值＋identity tie完全吻合，固定rank／format／path／N均保留，無unsupported。16份baseline metadata與原freeze完全相同；16個rebuilt local hashes全exact，pooled correction量化前後全bit-identical（float relative2.34e−15–4.76e−15），之前same-estimator疑點已實際閉合。32份eigen報告皆保留512正方向，無material negative。

比較族320單op＋240合成＝560 unique IDs，point／baseline／component links全完整；原40個SCORE IDs相同，incomplete composites為0。32個主要endpoint/op N80→120 VALID仍全改善，未plateau。q128 FP32控制目前mixed VALID弱於local，q512 W8 host較接近但仍取捨；q512 actual-span diagnostic低於其linear-latent error，不能排除更強latent／其他U。不得把mixed VALID當decode quality或把本控制失敗當所有低秩不可能。

**APPROVED只有凍結SCORE_DEVELOPMENT evaluation，無blocker。** Root另核2365 dependency／final source並發實際release。560全族與兩phase保留，不能因同樣稀疏routes／bootstrap undefined再事後拆族。全model residency／cache／KV／CPU／native FIFO與shared DAG仍未知，表示容量不等全系統延遲。原Stage0 inconclusive不變，本輪不自動增fail；TEST未讀，新evaluation outcomes未讀，未跑candidate或改碼。

正式獨立收據：`stage1-controls/critic_FITVAL_review.before_SCORE_DEVELOPMENT.20261103T031447Z.json`。FITVAL SHA`c901907f...acefff8`；comparison SHA`1740a4ce...3d04553`，仍bind scientific protocol`42483d...3269a`與rev003 source`84299da2...fad550e`。

## 2026-11-03 03:25:21 UTC：Stage1 development結果closure與partial quality gate意見

Stage1 EVALUATION OS0正常完成。獨立核160 point metadata、560比較原欄位與順序完全同freeze，metrics含16 baseline共176entries。500個比較被整族undefined傳播，60直接undefined全同Deep13e10 decode稀疏root：9hits／8prompts、seed20261103 draw2748／5167無energy；其中20單op＋40含它composite。無nonliteral SD0，不刪draw、不拆子集救推論。

N120描述性方向顯示G512 host／native在decode有5/8 operator error比local低，prefill僅1/8；PE512 decode1/8、prefill0/8；q128 FP32 PE皆弱於local，G只1/8 decode較低。這是實質phase取捨，不是bank必勝或factor全支配。某些G decode隨N80→120反而退化，不能拿mixed VALID改善當收斂證書。全560保留、兩次舊SCORE使用都是development；局部error／小retainedbank不等完整系統latency／NLL品質。**仍同Round2 inconclusive，有效fail增量0。**

支持另立**前瞻性的具體8-op adopted配置 deployment-failure**，不用先證所有guarded family死，但不能改判舊凍inconclusive。B只把同8op換canonical，其他sourceGGUF；C_local/G/PE同8op替換且各自KV／live route，並報sourceGGUF作hybrid loss背景。替換disabled的graph等價、canonical replacement API、alpha／merge／局部路由與未改branches先閉合，capture-only no-callback舊證據不能自動涵蓋新intervention。

看fresh非TEST資料前，需凍配置／primary failure unit、quality margin、fresh IDs／rollout／計分positions、mutation exposure、有限多重比較及prompt-paired CI。例子可用lowerCI(deltaNLL)>預定delta，delta=log1.02對應2%PPL工程容忍值；此只是明示budget例子，非通用投稿門檻且不能依新結果挑。教師強迫NLL用共同groundtruth位置但各自affected狀態；free-run各自tokens／routes，token不同本身不等quality fail。明確超限可否決這個adopted配置，不能否定allguards／全routed canonical／全模型；pass也只准同輪前進。

獨立收據：`stage1-controls/critic_EVALUATION_review.20261103T032521Z.json`；EVALUATION SHA`15f379be...33687a`，全族保留。未讀TEST、未執行candidate或改source；新的partial gate本次只有設計意見，沒有資料執行授權。

## 2026-11-03 03:33:20 UTC：Stage2 harness與fresh PG19概念批准

Harness draft001 SHA222986...abd54已讀；八個q512W8 host factor selector逐一重建mixed VALID minimum，完全相同，PE6／G2。要求的epoch single-use／zero-mask、replay與free-run獨立prefix/KV、相同tokenizer/BOS/truncation與所有實際numerical worker RN／MXCSR closure已由Root加入；已核frozen harness SHA0e59b5cc...5e57c6。**概念批准code/build/synthetic，不批准真資料。** B只pilotcanonical，C與D只同scope replacement；原source livealpha／slot與其他branches保留。D是單一VALID-selected混合arm，不是所有G/PE部署的全面勝負。

Fresh PG19 draft002 SHA2b80990a...7c9e0a4概念亦批准：16本metadata先選、generation/size/MD5固定、strict UTF8＋CRLF normalization、Unicode midpoint後首LF suffix、joint tokenizer512 IDs、context256／target256，用logits255..510打分，跨arms同GTtokens但各自state。平均16本paired deltaNLL與exp(delta)−1的customsubset PPL定義一致；不是official全PG19 word-PPL／跨tokenizer比較。兩模型primary共同book resampling，16本小樣本bootstrap明示approx；missing／nonfinite／nonliteral zeroSD使全primary inconclusive，literal樣本相同不能變population零寬CI。Secondary D／source不選模型。

預先選log1.02作2%relative PPL budget是可明示的工程screen門檻，不是社群標準。若工具／copy／canonical gate有效，任一模型C−B simultaneous lowerband超額，可前瞻性否決此具體已採用8-op joint配置，Root＋critic再判Round2 bounded結果；不必證所有guards不可能，也不改Stage0／1 inconclusive。不代表另外一個模型也fail、相對sourceGGUF一定差或模型普遍不可用；必報B−source／C−source。

Coverage須按arm／operator報prefix與scored predecessor的真替換slots；無任何intervention卻C/B不同屬tooling，未exposed operator不能拿到個別quality結論，已exposed部分真回歸仍可否決configured set。原始bytes／token／window hashes及16本逐book values保留，短書不換。正式收據`critic_stage2_concept_review.20261103T033320Z.json`；未下載／讀PG19或TEST payload、未跑candidate、未改source。本輪沒有現在就判fail，也沒有payload或eval執行授權。

## 2026-11-03 03:44:33 UTC：收緊輪次價值，先一次代表性TRAIN refresh

Root提出正確方法價值疑慮：舊FIT是template／128prefix＋8decode、曲線還改善，直接以PG自然書籍否決它雖可作很窄config結論，卻不足以當本campaign有意義的方法失敗輪。**建議不以舊配置PG domain-shift失敗consume；在任何PGvalidation payload/outcome前固定一次代表性PGtrain refresh，primary改刷新配置。** 原數學qualitygate概念有效，但其對象須前瞻更新，舊模型保留tool/dev，不偷改Stage0／1判定。

接受Root／Sol唯一有限recipe：oldFIT120＋16 pinned PGtrain books共同unweighted rows；每本B768-prefix＋B/C_old/D_old各至多32真decode states（各自KV/routes／固定舊種子／EOG不補樣），student prefix只建KV不重複納FIT。新row上限13824/layer，加舊約15.8k，平均C32 leaf約900而非16×(256＋32)單獨僅144；這仍不是獨立樣本／收斂證明。所有families同union、same all-x canonical target budget，舊targets可cache reuse，新rows同teacher只query一次。TRAIN-only重建shared tree/P/centers一次；固定8IDs/C/r/ηλ transfer／格式／q512與PE6G2角色，重fit slope/U/bias，不開grid、不讀PGval。

Gauge規則須一行凍清：目前8個q512host既有selected gauge皆identity，沿用policy最簡；若用RMS則新TRAIN重算，不能沿用舊scale或用PGval挑。重要wiring：Stage1舊`prompts<nfit120`不可排掉PG rows，須explicit union index／typed provenance；新registry需bind新tree/P/center hashes，worker不可默用舊Stage0 preprocessor。

此為唯一finite rescue，不自動追求plateau；工具／資料閉合後，刷新配置若仍越過事前2%PG16 paired門檻，可否決其有限校準budget下的adopted deployment attempt並由Root＋critic判輪次。不是allguards／retraining／更大預算不可能。Optional TRAIN-distance85若保留只secondary，不救或觸發unguarded primary，coverage本身不是latency fail。若區間仍寬則誠實inconclusive，不自動再加校準。正式addendum `critic_stage2_refresh_advice.20261103T034433Z.json`，本次未讀payload、未跑code/probe，無資料執行授權。

## 2026-11-03 03:54:03 UTC：refresh draft002＋PG quality draft004正式概念approve

已逐項讀最新兩檔，hash分別b7d35ada...f288ed6與0522e046...d1ddd6c，**APPROVED scientific design freeze，無blocking概念缺口**。Common unit-weight typed multiset、old cached target identity、每layer最多13824新rows／8op110592 expert queries／905969664 F32 bytes、Bprefix＋三arm自然EOG state因果順序與共用label budget均一致。新tree/P/center只TRAIN一次、newregistry綁preprocessor、explicit rowmask、固定identity gauge、重新trace-scaled beta／quantizer／actualbias已閉合。

PGvalidation16不供refit／選模型，primary明確是refreshedC−unchangedB；舊synthetic-only domain-shift不consume。兩model simultaneous pairedbook family、joint512 IDs之256targets/logits255..510、2%預定budget、missing/short/zeroSD全族inconclusive與secondary不作模型選擇皆合理。Finite calibration-budget breach可作bounded adopted-config failure，不需假裝sampleconverged，亦不等contextbank所有變體死；Root＋critic結果後判定。Optionalguard／coverage均不作primary或latencyfailure。

保留真groups／rank／conditioning／fallback曝光與B/C-source數據；生成arm不是獨立book，C−B超限也不必C−source差。只准metadata/path/hash凍結更新，不在validation後補rescue。正式收據 `critic_stage2_final_concept_review.20261103T035403Z.json`。無新grid要求，無source／download／COLLECT／FIT／eval執行授權，未讀payload或跑candidate。

## 2026-11-03 04:05:06 UTC：Stage2 rev004 TOOLING-only source批准

讀worker／harness／wire、activation與generated數值／IPC測試，正式rev004七份current sources／immutable snapshots／receipt hash全match。Freeze SHA00a08bfc...c811e1，registry7ed8766e...a8bfc9，protocol0e59b5cc...5e57c6，95deps由Root另獨立核後實際release。

Owned Up x/IDs、ask不讀寫、exact epoch／IDs／shape、zero-mask roundtrip＋single consume、全部reply先驗後按true strides只寫pilot完整rows、原live-alpha／slot／sum保留、每subrun freshKV均閉合。Preprocessing由明確registry path與payloadmanifest hash綁定，新registry不得默讀舊tree。Worker BLAS exact DGEMM單caller1另限與readback，沒有用OMP池冒稱查BLAS池；FMA/GGML/activation各自真8workers與RN/MXCSR守衛一致。

Synthetic receipt pass已讀：canonical H2048／I1408／I1024 reference、B1/batch／framing、local／PE／G介面、strided bytes／ownedcopy／epoch／zero mask與5種worker fault無部分寫入、256GTindex。B/copycanonical本身是repeat/copy control，獨立數值證據來自generated pinned-reference測試，不偷當第二套獨立teacher。

**APPROVED僅TOOLING**：每model第一固定FIT128prefix＋≤32EOGstop greedy bitcontrols；不是DEVELOPMENT／PG／refresh／quality／TEST release。Observe仍pass的extension，不是collector；aggregate hit/trace不是完整每op曝光或原生壓縮成本。未由我讀真payload或執行candidate／tests，未改source。正式收據`stage2-quality/critic_source_review.rev004.TOOLING.20261103T040506Z.json`。


## 2026-11-03 04:17:41 UTC：old capture 歷史缺口與 current target binding 補充

Root／executor 發現原 Stage0 FITVAL.complete.files 的1961個 hashed outputs 不含 capture inputs.f32／rows.jsonl；歷史 query receipt 只有 target SHA，沒有 input SHA。更正證據範圍：先前 output closure 不能宣稱原始 capture 同樣不可變綁定。現在 hash 只能固定 current，不追溯歷史；canonical 非 injective，重新 output bitmatch 也不能證歷史 X 相同。

**概念批准 BIND_OLD_CAPTURE**，不改科學配置。必須以歷史 freeze 已綁 frozenY SHA 為權威，完整綁 current inputs／ordered rows／FIT120 selection／8op target mapping，審計前後 content hash＋stat 保證穩定；固定 weights／numeric 實作與環境，全部 prescribed 元素 bitmatch 才原子 PASS。FIT_REFRESH release 綁 audit completion 與 current capture／rows／targets／weights／numeric／audit source SHA；所有 family 共用 verified labels，額外 verification queries 成本另計，無新 states／rows／grid／第二次 refresh。

Mismatch 先技術 hold，保留逐點證據，不得改 labels／放寬容差／剔除 rows／繞過失配重fit，不消耗科學輪。PASS 只建立 current X→canonical frozenY 的可用 supervised binding，不能修補歷史 provenance；舊結果仍須明示該限制。本次未讀／hash activation、rows、targets、weights 或新 corpus payload，未執行候選。正式補充收據 `stage2-quality/critic_old_capture_binding_concept.20261103T041741Z.json`；source gate 與 Root 實際審計／FIT_RELEASE 仍另行。

## 2026-11-03 04:26:31 UTC：refresh 四檔 source-only 窄預審

讀 mutable refresh_core／fit／worker／harness source，及固定 inherited Stage0／1 數學實作；未讀 payload／執行 tests。尚無 formal immutable freeze，**不授權資料執行**。fit:49–63 真正拼接 oldFIT＋new multiset，include_allones；core:88–94 不用 legacy prompt<nfit。Within-leaf actual-A8 centering、新 union covariance-trace η／λ、pooled prior／local correction、teacher-SSE Mtarget、q512 identity W8 hostlatent、部署後 mean(Y−pred) bias均沿 frozen 算法，PE6／G2固定，所有 family 共用一次 target query。

core:52–86 以完整 typed epoch join owned inputs／actual final alpha，B-only768prefix＋三arm各自然0..32decode，按book／arm／absoluteposition排序且逐一核 expected states，保留 multiset unitweights與共同bookgroup。新tree/P一次由完整union重建，registry明訂新preprocessing；old seed仍只經rev004載入，新collection未自動改模型。Old-label audit:28–42 綁ordered current mapping與歷史frozenY權威、全部F32bits、前後identity與原子complete，符合補充概念收據。

目前未見數學／資料 blocker。唯一窄報告補齊：fixed_fit目前有 leaf_solve，但G_RRR的leaf_solve=None且未落 pooled solve rank／df／condition，未完整滿足 frozen fit.support。已通知 executor；其確認用same pcov/pfactor/fixedeta補 pooled_solve diagnostics，無新grid／選擇。Generated rowjoin／mismatch／newpreprocessor integration tests由executor準備，待final READY/hash另審。原rev004四檔不得改，無新候選／新輪次。

## 2026-11-03 04:33:18 UTC：refresh rev002 BIND-only source正式批准

正式source freeze SHA8f89fc18...3e9206b，protocol78045b99...b5b5d167，oldrev004／registry仍00a08bfc...c811e1／7ed8766e...a8bfc9。七current refresh sources與immutable snapshots／receipt全match；另外核build commands／兩buildlogs／binaries／synthetic receipts／manifest與歷史FITVAL metadata836b4a1a...516a2a6。全127deps Root獨立核，未冒稱由我重核。

rev001→002唯一source差為old row mapping補tokenID／absolute position；對照原Stage0capture:44，prefill位置token_in_batch與decode generated_ids[step]在prefix_len＋step的feed語義正確。G pooled_solve rank／df／condition已補。Generated Python／C++ receipts全PASS：hook alpha/request join、Bprefix／Cprefix排除／Ddecode、duplicate／short／missingalpha／wrongID拒絕、typed multiset保留、unionmask negativecontrol、newpreprocessor roundtrip、PE/G fixedtrace／archive B1batch／actualbias、single-element oldY mismatch檢出，均只generated資料。

**APPROVED source gate僅 BIND_OLD_CAPTURE**，正式receipt `stage2-refresh/critic_source_review.rev002.BIND_OLD_CAPTURE.20261103T043318Z.json`。All8舊FIT currentX逐元素F32bits與歷史權威Y／weights比較，完整ordered mapping、current input／rows／tokenmetadata前後hash＋stat與atomicPASS閉合；只建立current supervised binding，不追溯歷史capture。Mismatch technicalhold，不改labels／容差／刪rows，不算科學失敗。Root仍另發實際audit release；COLLECT／FIT／acquisition／PGquality／TEST均不由本receipt放行。未讀實際payload或跑候選／tests，未改candidate source。

## 2026-11-03 04:35:53 UTC：同rev002 COLLECT_REFRESH source-only批准

再核七current sources＝rev002 immutable；TRAIN metadata SHA70facca3...f2c5e與protocol16 unique objects完全一致，未讀任何book或BIND結果。Exact GCS generation query＋response generation、size／base64MD5、strictUTF8／Unicode midpoint LF suffix與完整suffix native joint tokenize(add_special=true,parse_special=false)後768IDs皆符合；changed／missing／short只技術hold，不換書。

B/C_old/D_old固定每bookfreshKV、相同768prefix各自狀態；smallestID greedy自然EOG不feed，≤32不補樣。只Bprefix入池＋三arm全部真decode，不按hit/error抽樣或dedup。OwnedX／IDs與實際livealpha雙ledger以typed完整state identity join，各kind/layout另外核，所有expected states、request／position／IDs／slots與alpha總數閉合；依book、Bprefix/Bdecode/Cdecode/Ddecode、position排序。Unit multiset保留重複，一book三arm共同group，不冒稱獨立樣本。

Old seed registry／preprocessor／canonical weights／models均hash binding；GGUF重用既有stat identity並前後不變，並非重新算content hash。8distinct physical CPU affinity父子承襲、CPUonly、GGML／FMA／activation8與BLAScaller1保持，livecanonical／activation counters只offlinecollection成本。**APPROVED source-only COLLECT gate**，receipt `stage2-refresh/critic_source_review.rev002.COLLECT_REFRESH.20261103T043553Z.json`；Root須先核BIND.complete再把其SHA綁單獨actual COLLECT release，現收據不能直接開跑。FIT／PGvalidation／quality／TEST未放行，未執行candidate或讀payload。

## 2026-11-03 04:39:44 UTC：同rev002 FIT_REFRESH source-only批准

七current sources仍＝immutable rev002，無新數學差異。核fit:47–82與core:89–116：完整union alltrue mask、old ordered target mapping＋new typedbookgroup multiset，shared newtree/P/leafmean一次重建；η／λ／Gη維持原dimensionless值、newtrace／within-leaf decodedA8 covariance與prior／Mtarget重算。固定PE6/G2、q512W8hostidentity，actual loweredFIT residual bias重新校準；各newrow/operator target僅query一次共用，舊Y只用audit已權威綁定者。

新geometry／model paths與manifest寫入fresh registry，loaderfinite／scalepositive／shape／metadata checks、hash回讀、每leaf與old/new B1batch及naturalID IPC檢查均在source；incompletegeometry直接hold，unsupported保持明確canonicalfallback。Audit／collection全部檔案前後hash、oneiteration輸出與futurecomparisonsSHA閉合；literal PG quality protocol3f95838e...73ac01與實際metadata檔hash一致。Generated evidence沿先前PASS，不重跑。

**APPROVED source-only FIT_REFRESH gate，無blocker**。正式receipt `stage2-refresh/critic_source_review.rev002.FIT_REFRESH.20261103T043944Z.json`。Root必先核COLLECT完整closure，再單獨綁BIND／COLLECT complete SHA＋oldcapture mapping發actual FIT；本receipt沒有這些實際資料binding，不可直接開跑。有限refresh不是收斂或品質成功、operator／offline成本不等完整部署結果。本次未讀任何candidate／book／收集outcomes payload，未執行候選或測試，PGvalidation／quality／TEST未授權。

## 2026-11-03 04:47:34 UTC：PG quality core 初版 source-only 預審

只讀新quality_core與frozenprotocol metadata；paired統計Root已獨立核，未重跑。索引logits255..510對targets256..511共256、共同16book配對、兩primary全族保留、missing／zeroSD保守inconclusive與unused511不作先前target因果解釋均未見科學blocker。Safe exp/expm1 overflow metadata由Root已要求補齊，NLL有限與PPL轉換overflow須分別對待。

精確兩個join guard已通知Root／executor：summarize_outputs原:40 zip前assert兩paths／worker_exposure都長2；原:56–59 cpp／returned／fallback須非負整數且0≤fallback≤returned＝cppcount，防默默漏來源或負approx coverage。核心尚依賴未完成C++／launcher對context跨book／arm唯一、reportedarm與runtime mode/model/book一致，以及固定firstbook nohook／copy數值control作硬閉合；final整套source再核，不能把目前job.get(controls)展示誤當已實施control gate。無新grid／科學設計修改要求，未讀payload或跑candidate，尚無QUALITY source／actual release。

## 2026-11-03 04:53:35 UTC：PG quality 三檔 wiring 窄預審

讀新core／worker／harness source；先前zip／counts guards與safeexp已閉合。context_plan固定67唯一contexts（firstbook7＋其他15×4），summary核主64arm/model/uniquecontext/commonrun。source在contextmap標mode0而實跑OBSERVER1是合理設計：nohook／observer皆不走worker，summary明核internal_hook_mode1；接IPC的B3／copyB4／C5／D6都對映context mode/run/subrun5／position/phase。Workerobserve先由base serve呼叫，再evaluate按currentepoch累計；base h.model與serial等guard保留。

Registry直接依FIT.complete SHA與其registry path載新模型，manifest驗新tree/P/means/maps與canonicalweights，沒有默用oldgeometry。四批GT128、position255跨batch與最後511的zones在兩語言一致，source counts是potentialhit、其餘cpp counts在真正Down/scatter後加；workerreturned/fallback/approx與summary按context/layer/expert/zone配對。Firstbook actualmemcmp與zero-causal0..510 sanity由Root已核，source顯式throw而非僅展示flag。

**目前未見必要wiring blocker**。源仍mutable，launcher／synthetic尚在寫，不能據缺檔假定完成；formal immutablefreeze與actualregistry資料closure另審，無QUALITY release。未讀payload／FIT或quality結果、未執行candidate／tests。此時三檔sourceSHA分別core e1da599c...d3de69、worker bd79fdc9...a9fdc7、harness 0e8f78c1...c6f881，僅觀察當下來源，不是freeze。

## 2026-11-03 05:03:51 UTC：PG quality rev001正式source-only批准

Sourcefreeze00ea6f2a...bb10c09／protocol3f95838e...73ac01，六current sources＝immutable snapshots＝receipt；READY d0177cfd...24bd56，雙build／generatedCPP＋Python receipts9f6d73d6...44b46f／cad97e7a...79e95全PASS。獨立核source／build／test／manifest與FITcomplete969a6711...9a3585、registry16069fb3...bca5cc的metadata hash；實際152deps／470FITfiles由Root獨立驗證，未冒稱我讀其payload。

Launcher acquire前full FIT files contentSHA＋stat，summary後同review／q.authorize與全部identity重核；protocol／source／FIT／registry gates與新preprocessor path閉合。Exactgeneration validation16 acquisition與rawMD5/UTF8、jointnative512、GT四批128、255..510→256..511、freshKV／67contexts／sourceobserver vs truefirstbooknohook/copy與Bcopy memcmp、typedscatter/fallback/zone join均符合。固定control只book0不隨短書移動，missing全族inconclusive；unused511不作因果解釋。兩endpoint全族bookbootstrap與finiteNLL/derivedPPLoverflow分離保持。

Generated suites含independent scalar10kbootstrap reference、missing/nonfinite/zeroSD、overflowJSON、whole2model summary、malformedindex/context/count拒絕、actualobserver/scatter zones與memcmp因果反例、FITdependency mutation拒絕；未由我重跑。**APPROVED formal PG19_QUALITY source-only gate，無blocker**，receipt `stage2-pg19-quality/critic_source_review.rev001.PG19_QUALITY_SOURCE_ONLY.20261103T050351Z.json`。特意不在top-level提供actualFIT／registry release欄位，Root須另發完整actualapproval；不授權PGpayload／context執行，TEST仍封。無科學自動分類、無部署速度／8GB claim，未讀候選payload／quality結果或執行candidate。
## 2026-11-03 05:22:17 UTC：完整PG quality partialscreen裁定，同Round2續行

QUALITY OS0，complete SHA3706e544...12faab／summary bdeb8c0a...067136。獨立核32完整book-model jobs／32768finite NLL／每arm256target的255..510→256..511與GTID，重算perbookmean；1152非source曝光joins全match，兩model固定firstbook actualmemcmp controls與causal sanity均PASS。核closed numeric outputs／prepost receipt hashes；470 FIT before＝after，所有metadata SHA亦match FIT.complete。Root另核全部55closed files並獨立重算bootstrap；我未跑候選或讀raw書籍／fitted payload／TEST。

**通過原訂8-op平均品質partialscreen，不是失敗；Round2續行，fail增量0（累計仍1）。** Deep C−B Δ.0029174022，simult95[-.0068389393,.0126737437]，relative upper1.27544%；OLM Δ.0046878670，[-.0022219033,.0115976373]，upper1.16652%，皆<log1.02之2%budget。C−source Deep+.0032664092／OLM+.0026778075NLL，B轉換差與D固定secondary完整報；不選winner，不把Stage0／1全族未定逆改。

C真approx allrows3607／4781，fallback0；causal0..510為3603／4772。對16×511×K×routedlayers的call數分母，約0.2825%／0.4560%，只作介入範圍診斷，非權重／延遲比例，更不可線性外推NLL。全8op有曝光，但OLM L8E3只有16prefix＋5scored rows，無單op強qualityclaim。42306 OLM Δ.05050533完整保留，Deep亦有單book>2%；budget是16book mean，不保每book。16book近似bootstrap不是population/outlier保證，無刪書／改threshold／追加PG16調參。

下一步先以低成本凍same-capability必要性與全系統optimistic cost窗：local bank vs global/RRR與shared nonlinear-host-latent＋oneflashmap，給同host／flash容量、allhost/cache、合法ordinary頁waves／commands／ECC／MAC／finitePB／CPU/metadata/KV與native shared/dense barriers。若尚有可行優勢，再在同Round2事前凍有限all-routed／full-model quality expansion及新heldout／校準scope；不能拿目前0.3–0.5%call介入推全模型品質，也不把這16本變成新refit訊號。投稿前仍須匹配非線性control Pareto與真正E2E成本；<10%僅flag，不能事後改成舊screen自動fail。無新資料或執行授權。

正式收據 `stage2-pg19-quality/critic_scientific_adjudication.20261103T052217Z.json`。
## 2026-11-03 05:33:35 UTC：earliest64 GT不回饋source/theory審查，非實驗批准

Root提DeepL1／OLML0全64、其他全部source的新version。Pinned source支持固定GT證明：OLM qkv74/82–88→attention119→ffn127/131/136；Deep attention kv264/313/317→attn327或364→ffn373/376/388。graph KV copies2217–18／MLA2309只存attention K/V；llama-model:2002 reuse初始null，只有Gemma3n/4設callback，Deep/OLM沒有laterlayer cache回饋。Router1383–1499只吃同preFFNcur，weighted1668在專家後。以相同tokenhistory、positions、freshstate、batch/ubatch與numerics作token induction，earliest UpX／IDs／finalalpha相同；後層state/KV/logits可不同。

必要限縮：existing TRAIN union對「每段已記錄tokenhistory按原切分重播」才是exact counterfactual inputs，不是新first64 freegeneration的on-policy分布，更不是未見GT512的相同states。舊canonical B1/batch接口測試只證固定X的f(X)，不證native attention跨rechunk Xbits相同。新GT四批128可事前核各newarms X／IDs／alpha bitmatch，保持freshKV／无alias／先前层unchanged；freegeneration另驗，不宣稱所有hidden相同。

Newversion＋freshheldout不必違反舊8map不得回頭tune：舊artifact／pass／maps不動，earliest現有4maps及commongeometry可按宣告重用，late2layer intervention取消是新scope。新增124targets／penalty及control角色／commontargetbudget／停止規則須freshoutcomes前凍，不用PG16挑新參數。這是較便宜乾淨的whole-layer品質stage，不是allrouted或paper成功。

最強系統反駁：singlelayer C32r128 W8 maps512MiB、bias16MiB、scales16MiB、P1MiB；原canonical整層Deep528MiB／OLM384MiB。兩者係數都單獨<1GiB，必給同總host/KV預算的allhost/cache控制，不能由此證NAND bank必要。若只部署一層，equalprojection routed係數工作份額1/26／1/16，不是嚴格E2Etime上界，亦非≥10%失敗判準。All-depth品質與matched nonlinear/global／合法硬體costwindow仍欠。

正式source/theory收據 `critic_earliest64_source_theory.20261103T053335Z.json`。只讀/hashed source，無payload／實作／probe／candidate／實驗批准。
## 2026-11-03 05:48:37 UTC：Stage3 NL8/control-cost protocol draft002概念批准

讀Root planning／Sol nonlinear與break-even notes、Terra i9 primary-source ledger、draft001與Root draft002完整semantic diff。正式review `critic_stage3_protocol_review.rev001.json` 綁draft002 SHA5f21869a...9b58be（draft001保留6887638a...4c9616）。**APPROVED for source implementation/build/generatedsynthetic only，無剩餘protocol blocker；不是payload或phase執行release。** 同Round2／同method，無新clock／fail。

閉合Root/critic必要修訂：C/D/P/tree immutable、NL另命名；fullunion eta0 pooledGUOLS、raw2.304667648GB新GU＋1.179989835776T MAC與Downcalibration/FMA另計；D first exactint／secondFMA、G centered-fullH A8不可偷重用B qx；PGTRAIN-only32bench／first4numeric避免oldalpha未證；B/G/C/PE/NL按真正function付preprocess，zerohit不虛做P。Warm≠LLC，cold8worker各128MiB cache-pollution、不是全cache分布。研究8e9／quality32e9 RSS與部署8e9/8GiB inclusive分開；EPYC AVX2不是i9besthost，86.4／89.6均非sustained。

DEV兩NL−B endpoint固定seed2026110303／10k pairedbook／max|t|／ln1.02；兩upper below支持此fixedNL開發可行、任lower above只否決NLrecipe，其餘/undefined未定，不consumeRound2；NL−C僅描述，PG16重用不fresh。共同qualitybudget不等strictPareto無更高loss。UnknownPE/FIFO／cache／合法wholeDAG保持sensitivity，不從兩lowerbound或microtime聲稱硬體勝負。

源審必要checkpoints：partial-cohort返回expertoutputs或pilotcontribution須明標，不冒稱fullnativeMoE；不為此新增非pilot targets。OldB/C/D64bit checksum只是consistency，非歷史vector實memcmp。Fullmodel allhost/cache分析必要，但額外全模型export/run不在8op scope。Optional earliestperturbpreflight建議可先omit，無需拖NL/costclosure。實際BIND／GU_FIT／BENCH／DEVQUALITY各待source/numeric/manifest獨立release。未讀候選payload／targets／書籍／model、未寫候選碼或執行。

## 2026-11-03 05:56:12 UTC：Gemini Stage3 cost challenge，採先BCD而非採硬否決

讀Gemini source/theory note（SHAf27dacb0...c47ab4）；該文基於舊draft001，現行draft002／frozen SHA5f21869a...9b58be已修研究RSS、D integer/FMA、G centered A8、PGTRAIN bench、逐worker cache與target ISA。**支持事前凍最小 B/C/D-before-GU phase-order amendment**，不採其「NL8在Deep永不能hardware negative」或D先必須quality輸給C的Boolean gate。Root新D secondary posthoc development兩upper<2%只加強先量BCD優先度，不是fresh／strictPareto／allrouted證據。

獨立純算術核GU2,399,141,888B＋GU biases599,785,472B＋Y biases436,207,616B、Down4,798,283,776B＋scales13,631,488B主項正確。Gem subtotal8,281,135,904B>8e9但<8GiB，尚餘308,798,688B；compulsory nonFFN≥1.53GB未閉合。其common34.085MB亦漏完整selector成本：P＋leafmeans＋31full-Hnormals已40,681,472B，另threshold6,448B。這些是指定全pin獨立表示／encoding的帳，不是所有hybrid／cache／partial8或壓縮表示的no-go。

明列強控制B_FlashGU_HostDown與samebank ordinaryload＋CPU，但stockAiF bitexact未由software整數代數證明；需ABI／scale／cast／activation與alpha順序閉合。只有無overflow的整數partials可associative，F32不可。RawtR＋PCIe payload不是stockNVMe feasible time；hostcache預算、DMA／controller／metadata／queue／hostwork須算。Shared＋routed各自8core測值不能直接相加成全局下界；用jointresourcework／可行wholeDAG。86.4是paper配置、89.6是ARKadvertised，皆非sustained。

Minimal amendment只改order／單獨release，不增grid／fit／dataset，BENCH總1200秒不因拆段翻倍；optional earliestpreflight可省。GU前若無完整feasiblebaseline對bestlegalcandidatebound的 scoped domination，只得conditionalcostwindow或budgetdefer，不算methodfail。正式收據 `critic_gemini_stage3_challenge_review.json`。未讀candidate／book／model／target／activationpayload，未執行候選／tests，未改candidate source；Root supplied D summary只作明示來源的開發背景。

## 2026-11-03 06:12:21 UTC：ordinary AiF overlap／latch／ECC source邊界

核Terra `docs/aif_overlap_followup_terra.md` SHA8860c6c4...83e0cbd與本地primary `paper/aif.txt` SHA7511dfad...e39a3b7，讀§4.3–4.4／5／6相關原文。**ordinary CSB/MSB46µs是合法非CR讀取數字，並不自帶native AiF computable contract**。AiF只在be-enc單次sensing LSB釋出兩個data latches作input；CSB／MSB三次sensing需三個data latches，且ECCLITE10bit/1KiB依LSB較低錯誤率設計。主native cost只用LSB並閉合LSB容量；CSB須另列buffer/lifetime與可靠度/ECC方案及其成本，不能把46直接替換28卻保留免費input/ECC假設。這是契約未證，非禁止小幅硬體改動或證CSB永不可計算。

原文確有outputFIFO及計算時輪詢、邊產出邊drain，不能說native無FIFO。Depth／backpressure／in-flight命令tag／input跨命令生命週期未披露，故不證跨cohort sense／PE重疊；同一command連續pages也不證下一GEMV可同時進場。反面亦不能把28＋全ECC＋全PE說成唯一時序，或把額外每plane16KiB、全16chip共1MiB當必要新增SRAM下界。Paper 6.4GB/s decoder不是PE throughput，6.4有效flashbandwidth另混合cr-read/be-enc，無法直接移植本ordinary配置。保守serial／片段pipeline／cross-cohort三種conditional DAG各自保持buffer／ports／ECC／PE／drain資源條件。

Root指定先不審尚mutable／過時rev002，等最終rev003 READY後才核 exactsource／freeze／phasegates。本次僅讀上述文字primary與note，未讀candidate payload、未跑任何candidate／build／test，無phase release。

## 2026-11-03 06:19:21 UTC：Stage3 rev003 BIND source-only正式批准

Exactfreeze9dcd169b...db4ef8f／READY a84ac2b4...d92614e，base5f21869a...9b58be＋phaseordercd8fb022...a64a835b同綁。核七snapshot sourcehash、manifest／buildcommands／三binary與三build＋兩generated receipts，全部match／OS0／無resourceviolations；Root獨立490files206,733,784B完整核驗，未冒稱我重讀實際payload。正式receipt `stage3-control-cost/critic_source_review.rev003.BIND_SOURCE_ONLY.20261103T061921Z.json`。

專審numeric.hpp/cpp、synthetic、worker/harness：signedI8→I16→nonsaturatingI32 exactdot全寬2048 bound33,554,432；D exactintfirst／FMAsecond、G full-H centeredA8、P固定FP64lanes與tree sequentialFP64均正確。Team實際8worker RN/MXCSR FTZDAZguard；首段跨slot一次dispatch、輸出rowowner逐slot F32alpha乘加；scratch計時前配置計resident。Timed只單一pinnedfullI SiLU×Up／finite，fusedbitcomparison只audit；各privateactivation獨立wholeImax。零hit不付P/tree，fullK複製物理arrays且明標resourceonly，partial是pilotcontribution非完整nativeMoE。

Generated receipts核signedextremes/tails、FMA/P/A8bits、workerfault、I1024/1408activation、8worker128MiBpollution、rankdeficientfit/archive、allrawBCD/PE/G/NL、fallback、signedalpha cancellation natural/fullK/zero、mode7/8 IPC／badframe、bootstrap獨立ref／missing／zeroSD與metadata mutation拒絕。Worker7/8新arm不重解釋D，仍綁refreshedpreprocessor／manifest；品質freshKV／GT512／actualfirstbookcopy／zerocausalB重播及historical checksum≠memcmp明確，未據source觀察預放DEVQUALITY。

**APPROVED BIND source-only，無blocker**。Root另發actualreview；BENCH須BINDclosure，GU須BCD後checkpoint，其他階段仍封。31timedcalls用rows0..30，selected32pool第32row不timed，只須如實說明rawindices，不增calls/grid。Researchmicrotime／pollution／RSS不當8GB或E2E證據；ordinaryLSB與conditionalCSB／FIFO契約限制維持。未跑任何candidate/build/tests、未讀真weights／activation／target／book，未改candidate source。

## 2026-11-03 06:24:07 UTC：BENCH_BCD source續審，nested-session checkpoint需閉合

沿用同rev003數學／C++審核，只補launcher benchmark／hardware／prepost source。三armmod3、5warmups31timedcalls／32pool、四cases、每worker128MiB coldpollution、persistentcohort/preallocbytes均閉合。Hardware實際CPU0..3／4..7分兩個32MiBL3，非整socket512MiBcache假設。FIT＋BIND exports完整before/afterSHA/stat包含typedx/IDs/livealpha與layer manifests；actual BENCH仍需BINDcomplete/allbits/Rootindependenthashclosure。No numeric／performancepayload讀取或candidate執行。

發現一個真launcher blocker：run.py:143 outer monitored為_execute建新session，bench:102內層monitored:58又讓host-numeric建另一session；outer600s／RSS／affinity違規時:64只killpg(_execute.pid)，漏掉另group的numeric子孫，可能phase已報stop但計算仍在跑。已即報Root，**BENCH source gate暫hold**，不影響BIND數學或既有BIND source批准。需窄修完整descendant cleanup或外層真正能收整tree的watchdog，然後immutable diff/hash再核；不是科學失敗，不擴grid。Formal pre-review `stage3-control-cost/critic_source_pre_review.rev003.BENCH_BCD_HOLD.20261103T062407Z.json`。沒有發actualBENCH／GU permission。

## 2026-11-03 06:36:52 UTC：rev006 cleanup閉合，BENCH_BCD source-only批准／BIND橋接

R4 Anaconda無os.pidfd_open在spawn前failclosed，R5 fallback通過，R6明確Linux+x86_64 syscall434／c_long／errno guard後finalfreeze87832ea2...e15c04bf，READY13315255...2fcc6fc8。只讀R3→R6 run／synthetic diff，獨立核其餘五source與三binary逐byte全等R3；manifest／geometry／numerics／trial不變。Root全524files207,019,402B與7current=snapshot已獨立核，另重核原BIND698files未變；我未讀actualpayload。

新subreaper以starttime＋pidfd綁owned identities，持續discover不同session與adoptedorphans，排除preexistingchildren；TERM→KILL後reap，需兩次stableemptyscan才成功，rootOS0留child也fail。四個generated timeout／RSS／exception／normal-root-orphan全部用三層各自session且忽略TERM；各case3owned全消失、2adoptedreaped、KILL escalation有receipt，sentinel與既有activationworker存活。Lifecycle receipt88728d01...01c7abd，三build兩generatedOS0／resourcezero／allownedgone均核metadatahash，不重跑。600秒是啟動checkpoint teardown，boundedcleanup開銷不混入invocationtime／科學失敗。

**APPROVED BENCH_BCD source-only，唯一cleanup hold解除**。Formal receipt `stage3-control-cost/critic_source_review.rev006.BENCH_BCD_SOURCE_ONLY.20261103T063652Z.json`。明准pure-runner橋接既有BIND5f666211...3e66ea8／closureba8c621b...83b06b5：數學producer／binaries未變，歷史收據仍是R3execution，不能重寫成R6，也不為ordinal重跑。Actual Root release仍須綁R6＋原BINDsha＋bridge，再做既有typeddata/FITprepost。GU／NLbench／DEV／TEST仍封，無performance/GUgo/scientificoutcome。本次未讀numericpayload／執行candidate或tests／改candidate源。
