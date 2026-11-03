# 2026-11-03 第一輪候選思考：thinker_sol

## 計時與範圍

- 實際開始：2026-11-02 18:58:04 UTC（台北 2026-11-03 02:58:04）。
- 最短深入思考截止：2026-11-02 19:28:04 UTC。
- 本記錄僅含專案理解、唯讀證據查核、數學分析及候選設計；沒有開始新候選數值實驗，不將既有結果或分析腳本計為新一輪。
- 指派角色：gpt-6.1-sol / xhigh；不再派子代理。

## 18:58:04–18:58:46 UTC：第一批讀取與淘汰清單（後續持續研讀）

已讀 README、aif_model_notes、ordinary_read_placement_candidate、research_2026_11_02、research_2026_11_02_round2 的主要機制及後段失敗分支，並讀 round3-cr-ipr 的總覽與 full-ffn README。其餘細節繼續按候選需要查核。

專案不是 AiF 硬體量測：llama.cpp 提供真 routing；NVMeVirt/軟體模型負責 timing，raw W8A8 CPU 參考與 virtual-chip 映像提供數值／地址證明。普通讀取主線是16 chips×4 planes×16KiB、28μs；CR前緣另有9.7μs及PB/copy/scan的條件契約。不能把 CR 分支的較大數字挪入不用 CR 的論點。

目前 ordinary-read 有效前緣：whole-row + phase-specific geometry + learned cyclic/overlap/depth maps，及同格式W8 TRAIN/heldout。Deep 最新完整FFN service改善約1.87–3.26%；G4約0–2.64%。Q38外部單task nominal-I8約9.23%，不是同格式品質／全模型結果。更強joint/coupled/native/free-mask等只剩小幅收益；simple balanced-wave menu精確覆蓋目前606個selected winners，DP/compiler本身沒有額外性能歸因。

不能重複作為新候選：共選hypergraph重排、projection分開g、wave tiling、不等rows、shared counterbalance、更多native search、少量prefix/whole copies、GU common-input coalescing、普通streaming readiness、純channel配對、batch read-once的一般形式、read-disturb熱冷位置、bitwise parity避erase。round3新低容量CBC／g2 availability依賴CR/特定PB契約；不能只把它關掉CR再稱全新。

新使用者允許小幅硬體修改，因此舊文件排除decoder/新context的規則是舊任務範圍，不能當新禁令。但引入解碼、額外context或數值改動必須明列面積／SRAM／能耗／品質與強基準。

第一階段判斷：要跨過已接近NAND工作下界的模型，只改位置不夠，須減少每token必讀係數、讓同次讀取產生更多有效工作、或改變flash感知/weight表示。下一步優先思考三類：(1) exact lossless expert間共享／base-residual表示；(2) canonical I8 的bit-plane選讀及可驗證refinement；(3)有限SRAM、跨token/請求的resident頁重用，並嚴格排除等容量通用cache已取得的部分。

## 19:06:11 UTC：第一輪構想淘汰與新候選草案

Root更新：共同正式候選思考起點稍後指定；18:58起的專案理解不能自動充作某新方法1800秒。以下均為實際文獻／數學工作，不將思考淘汰計為實驗失敗。

補讀 `round2/next-directions/diagnostic_notes.json` 顯示，generic lossless compression、exact shared base以及zero-column/bitplane已被先前深入考慮。舊raw-Q8 entropy約7.3–7.7bits，Deep/Qwen壓縮在既有shards上不減waves；exact XOR-delta反而更差。A8 exact-zero column packing也已有 z^m page-skip 界，不能重新命名為新候選。Terra／critic補Ripple、SpQt、SLIM等直接prior；共同GU/Down neuron permutation與die-side sparse partial sums已有直接工作。

### 草案：以合併誤差的微型 sketch 決定 Down correction pages

目的：用真實本token的Down activation與router係數，決定哪些低bits需讀，改變必要sensing；有／無shared expert皆成立。不是僅依expert熱門度或router α選固定精度。

Canonical W8 integer可精確拆為 B=16q+8、R=(low4bits−8)，其中q為signed4bit高nibble，R∈[−8,7]。B與R皆各存4bits/weight，B+R=W。Base解碼值仍在I8範圍[−120,120]。若全部residual讀取、I32 partial相加後僅一次row-scale，可回復原W8；只略讀某些residual時是明列數值近似，須真正品質測試。

對Down實際誤差 e=Σ_e α_e diag(s_e)R_e z_e，預存 C_e=S diag(s_e)R_e（S為r×H隨機投影、C_e是r×I）。真正 z_e ready後，v_e=α_e C_e z_e可用host小矩陣讀取算出。對K6/8/10枚舉2^K個expert correction subsets，以Σ未讀v_e估計合併誤差，選NAND成本最小且誤差預算可接受者。Shared若存在只是額外固定α分支；不作必要前提。

必要小改：nibble unpack／midrise bias、signed residual unpack、I32 continuation／一次rescale、可識別base/residual的命令與地址。不能當已公開AiF能力。沒有新多input PB、沒有CR，也不跨過完整GU／dynamic-A8 amax barrier。

最樂觀Down-only FFN sensing改善上限為1/6=16.7%；若FFN占完整decode時間f且完全免residual，E2E speedup≤1/(1−f/6)。超過10%延遲降幅至少需f>0.6且其餘成本極小（若用speedup≥1.1則f≥6/11≈54.5%，需明分指標）。Correction讀取會縮小此空間。不能把本候選當all-model2×。

未決：sketch決策的adaptive selection偏差；JL norm confidence與任務品質不同；兩pass release／barrier；sketch精度誤差、host DRAM容量／帶寬及CPU時間；residual pages粒度與最強same-quality mixed precision。新搜尋發現2026-10-27 arxiv2610.37386，以ternary sketch選channels＋lowrank recovery大幅省reads，是必須補讀的直接強prior。尚未開始任何候選數值實驗。

## 正式候選計時

Root於UTC 2026-11-02 **19:07:01**指定合併Down residual sketch為第一個正式候選思考。最早可實驗時間為 **19:37:01 UTC**；18:58以來的專案理解不抵扣。以下實質工作包含文獻差異、數學反例、硬體數值契約與強控制；尚未執行數值probe。

## 19:10:34 UTC：候選最大的必要性挑戰

新增直接primary prior：DP-LLM（NeurIPS2025）§5已有預存G=AΔW、runtime Gx估quantization error再選precision；DPA-MoE（arxiv2610.21378）已有progressive chunks、offline reconstruction-error LUT與live α²分配。因此本候選不能宣稱sketch／token-dependent precision新；餘下gap只可能是同一token多expert signed residual interactions與ordinaryNAND的真實成本。

DPA Appendix B的Spearman≥.999是在uniform k、跨dispatch的exact／diagonal排序，不是固定token、不同refinement subsets的排序，不能直接當反證。但它足以要求使用actual-token diagonal oracle（完全已知個別norm、無cross）作必要強控制，而非只贏offline LUT。

若K個weighted residual近似獨立isotropic，各長度相近，cross energy相對diagonal總energy之標準差約 sqrt(2(K−1)/(KH))。H2048/K6約2.85%；這是條件性手算，不是本模型量測。低r隨機sketch的pair cosine noise反而約1/sqrt(r)，r16約25%，會淹沒真小cross。枚舉2^K subsets可能只是偏挑假抵銷。首／末層或有共同rank1 bias時可以例外；必須找到這個結構，不能只看平均error下降。

Midrise residual仍可能有DC均值；強基準應可直接補per-row或per-column mean，再以α²σ²||z||²估剩餘tail，不將可用極少metadata消掉的bias算作新cross收益。

Data-aware共同subspace U是可能修正：C_e=Uᵀdiag(s)R_e，runtime c=ΣαC_ez_e能精確捕捉此subspace的combined error。但有明顯支配控制：既已付同C與MAC，可直接加U c到basecombined，成為普通lowrank residual compensation；只作selector浪費已知可修分量。應比較『同subspace直接compensate＋cheap tail diagonal refine』；若後者匹配，sketch新意消失。一般lowrank correction已大量存在，不因把它當error predictor而變新。

統計界線：固定r<H的S有nullspace，大error u可Su=0。固定token且query vectors獨立於S時，可對2^K subsets作JL union bound；跨token的z會受先前同S precision decisions影響，不能再直接套fixed-vector JL。Hardt–Woodruff STOC2013 arxiv1211.1056給出adaptive linear-sketch風險。獨立audit S只修單token的selection reuse，不自動修復長序列adaptivity；不主張distribution-free certificate。

## 19:15 UTC 前後：NAND 幾何、數值資料流與強控制細化

這個時段仍是持續手算與文獻互審，沒有數值probe。精確分解不代表把native Down成本直接除二：G4舊g12的128rows×512 I8恰64KiB/wave；4bit後32KiB仍需一wave，原geometry sensing完全不減。需新固定geometry，例如256rows×512×.5=64KiB，六個owners/expert，再正確計top6碰撞。Base／residual各一wave同owner，全fallback總work等同舊12waves但job深度／commands／ready time不一定等同。Deep的half-byte row長704B，exact22wave work會要求深job；2-wave-depth balanced geometry至少需23waves。這只是按完整rows容量的手算，不是配置結果。

小改硬體的可行兩種路徑：

1. 在同owner的base／residual命令之間保留每row I32 partial及同一A8 input context，最後一次row rescale；selected lowbits未讀時可完成coarse output。對256-row owner，僅partials需1KiB；輸入、ECC、FIFO與多expert ownership另計。
2. 提供明列的新raw-I32 output flag，由host以I64做16·dot(h,z)+8·sum(z)+dot(r,z)，再一次row rescale。原公開AiF尚未證raw-I32；不得假裝已有。Base與R若分不同owner，必須支付更多輸出與host gather，不得免費。

每expert每row scale不同，不能先把不同experts的I32 partials跨expert加總再只乘一個scale。加回全部R時應保存未scale integer，不能先各自F32 rescale再聲稱bitwise exact。

Nibble decoder不減MAC次數；普通28μs每4planes最多64KiB packed，解出128KiB weights，若假設6.4Gweights/s PE則需20.48μs，可在理想read/compute重疊下被28μs遮蔽，但保守sense後drain時不能除二。AiF未公開lane數，6.4GB/s ECC參數不能逕當實際MAC吞吐；需另外掃decoded-MAC service並實作/合成decoder後才主張小面積。

Selector不必必然插入base→decision→residual的完整feedback barrier：若使用固定layer error budget，sketch可在z完整ready後與base普通read重疊。若以本次basecombined norm作相對門檻，則須等base完成；若再加full-output norm sketch則metadata和MAC要加倍。不能把不同版本的最佳成本混用。

強控制已擴充為：uniform原8／coarse4／6bit（各自可retile）、靜態same-quality混精度；DPA α²·offline residual LUT；α²σ²||z||²與mean compensation；DP-LLM同metadata逐expert norm sketch；actual-token diagonal norm oracle；signed combined residual oracle（兩個oracles只作界）；同subspace直接lowrank compensation＋tail diagonal；最新ternary channel reader。所有控制均須使用相同nibble硬體、DRAM預算、原生geometry/commands及品質資料。

Cross新意對top1完全不存在；K2且固定refine一個時，未讀只剩單expert residual，signed與exact individual norm排名相同。這限制其模型適用範圍，不能把Mixtral top2當天然正向案例。

## 19:16:39 UTC：原版候選的思考期決定

Root允許已充分受反例挑戰的方向在30分鐘前思考淘汰，只有進入實驗的方法必須先完成至少1800秒。**原signed-residual sketch selector版本建議thought-reject，尚無實驗、不消耗任何輪數。** 本記錄不宣稱已完成此版本30分鐘；19:07:01至此是實際候選思考，結束時間以接續clock receipt為準。

理由是新意僅餘跨expert signed interactions，尚缺優於同metadata DP-LLM norm估計＋DPA分配的必要性；isotropic cross小、低維noise大、data-aware補救必須先勝直接lowrank compensation。小K有嚴格零新意子例。DPA既有acrossdispatch correlation不能證subset排序，故仍不把此淘汰擴張成全域無效定理。

Cheap stronger control可只存每input column的residual能量 v_i=Σ_j(s_j R_ji)^2，runtime score=α² s_x² Σ_i v_i z_i8²；O(I)算力及4I metadata，比r×I sketch便宜r倍。若DC bias顯著，可用row均值加sum(z)直接補償，不能將此部分報成新cancellation。

下一個尚未正式選定的構想是容量換activation lookup/precompute。將另列新計時，不延用19:07時計、不啟數值probe。

## 19:18:35 UTC 起：容量換 activation lookup 的構想核算

此項仍未選作正式實驗候選，沒有沿用前一候選計時，也未跑數值probe。參考一手來源：MADDNESS（PMLR139/ICML2021）、LUT-NN（MobiCom2023）、PIM-DL（ASPLOS2024）。將input依g維切chunks，C個centroids，預存T_{j,c}=W[:,j]c，每output partial占b bytes。理想query讀量為原I8係數的b/g，常駐表容量是Cb/g倍；這是既有LUT思路，尚非新意。

NAND不能只用byte ratio：G4 GU每expert1024outputs，FP16 entry僅2KiB，16KiB page包8experts。若top6約均勻散落64experts，8頁的expected occupied pages約4.4（with-replacement粗算，真無放回應另計），每input chunk約讀70KiB；g8有192chunks，約13MiB，已高於原selected6 GU的9MiB，尚未加4plane同WL限制。Down小output同樣受page amplification。訓練共選packing只能改此條件，不可把generic clustering改稱新法。

大output是不同regime：Mixtral GU R28672，FP16 table向量57344B≈一wave。g4/C16有4096/4=1024lookups，對raw GU1792waves有42.9%必要work空間，Down保持原I8則FFN最多約28.6% sensing減量；不是E2E實測。常駐GU表約8倍raw GU，即224GiB，加原Down約14GiB。AiF 1TB TLC不代表1TB LSB係數容量；須精查LSB約1/3 reservation與spare/ECC。C16/g4只有1bit/input dimension，品質非常不明；C64提高至1.5bit/dim卻使GU表約896GiB，難在同物理預算成立。

Exact activation bit-pattern LUT沒有免費出路：g8的binary-pattern table有256entries，但8bit input需8次lookup；若partial用2B，理想query量為8×2/8=2倍原I8weights，尚未page amplification。較大g可降query量但2^g容量迅速爆炸。用PQ是必要近似，而非無損重寫。

小改硬體至少包含table-mode accumulation、index gather、partials scale與output SRAM；完整GU後的SwiGLU仍需重新品質驗證。現階段只有窄regime headroom，演算法與已有LUT-NN/PIM-DL相同，不能只因放到AiF就視為投稿突破。

## 第二個正式候選：Gate-first exact A8-zero

- Root指定起點：2026-11-02 **19:26:00 UTC**。
- 原最早實驗時間19:56:00；為讓Gemini正式加入後也滿30分鐘，Root延至 **19:58:00 UTC**。時間到不自動通過審查。
- Frontier/speculative verification於19:23:00被列另一active hypothesis；本agent現在優先深入Gate-first，不把兩者時間相加。

## 19:30:51 UTC：證書、真正新歸因與模型界

令實際canonical gate activation為f_i（沿用GGML實際產生值），Up未讀時有確定幅度上界U_i≥|u_i|。先實算少數Up witness，令L=max_witness |fl32(f_i u_i)|>0。若某row的canonical activation上界經實際F32量化比值仍嚴格小於0.5，就能保證該row的A8值為0。實數充分式是 |f_i|U_i<L/(2·127)，但bit-exact實作不能只套實數式。

本repo `quantize` 是F32 `s=amax/127`、`std::round(v/s)` ties-away。`scale_output` 是先F32 s=sw*sx，再將integer dot轉F32並乘s。可預存exact integer W2_i=Σ w8²，token算X2=Σ x8²，取integer floor_sqrt(W2_i X2)作dot幅度上界，再按相同正scale與F32順序變成Up界。或用整數block-L1與input block-max；可列min界、signed interval，但不可預設tight。SnaPEA exact-mode、range-bound CNN skipping已是早停直接prior，不能主張首次exact early zero。

證書無global-amax循環：被證row幅度<L/254<L，不可能是true amax owner。其餘rows全算後amax與canonical相同，證rows設為0後，Down input A8 bytes與scale仍相同。因此不需保留Up intermediate F32相同；真正應驗的是Down quantized input／scale、Down outputs、combine、logits/routes/tokens相同。將每個Up row bitwise相同當必要條件會錯誤否決合法方法。

強baseline必須先完整Up，再對**所有實際A8 zeros**省Down columns。因此新方法可歸因的額外saving只有省Up weights；不能把既有Down zero sparsity重算一遍。若full-Up row certificate比例為z，新增可免係數占原dense三projection工作量的z/3；若先讀高4bit才證明省low4，則為z/6；高7bits後省最後bit只有z/24。這些不是相對strong baseline的delay上界。兩種方法最後都應利用全部真正A8-zero Down columns，不只利用提前證出者。

純L2界的危險：對近isotropic rows/input，U/典型真Up約sqrt(H)=32–64，可能只有極負gate能證明。高4bits部分Up＋lowbit確定界能收緊但先付半Up，且同page全部rows可省才減sensing；不是看到scalar z就直接報wave gain。

一個有實體數值語義的收緊來源：checkpoint原本若clamp Up，直接取U≤limit，不新增loss。Terra核官方DeepSeek-V4-Flash的up ±10（shared/routed皆有）及gpt-oss的clamp(up,−7,7)+1，所以後者乘數界為8且沒有shared。這僅作模型界例，不下載大型新模型。它們原格式FP4/BF16，若本研究改成W8A8是新數值參考，須品質評估，不能稱原checkpoint原生bitexact。

現有非shared OLMoE已有來源與routing，可先用同一W8A8契約思考cheap-bound/headroom；它沒有上述native clamp，不能把clamp例的潛在收益套用。H4096時一條完整Up I8 row恰4KiB，是SOML粒度自然匹配；H2048時兩rows/subpage；G4 H1536與gpt-oss H2880則有row/subpage不整除與padding成本。

已讀Gemini19:27:51筆記。其B slack precision、C近似slice fusion、D VQ codebook均保留作備選，不能挪入本exact證書的歸因；A的q4 PB與PE吞吐需要獨立資源帳，不能默認免費。

## 19:40:33 UTC：分母、SOML 真約束與證書價值審查

Root/Terra修正分母：等projection係數成本下，FullGU＋actual-zero Down baseline為3−d，candidate為3−d−p；relative work reduction=p/(3−d)，speedup=(3−d)/(3−d−p)。p/3只是對原dense的絕對work比例。以d=.5、p=.25，理想relative work reduction已是10%。真正native timing還有waves／chip critical path，不能從scalar p直接給任意排程的delay上限。

SOML原文§4/5已核：quarter-page的1.7% die estimate不代表僅控制器改動；其需要額外select transistor layers、PD/BD改動。Gather操作有bitline群及block-decoder群不得重疊的限制，同block不同pages不能同時獨立read。任何未來native模型必須保留此constraint，不能用『全active subpages數除16』當可執行wave數。Gate/Up同layer x，跨experts gather可共用operand，這部分可免private-input contexts；Down則不能免費跨expert共用x。

Clamped-Up的free-witness樂觀篩式：若U_i=C，G=max_i|f_i|，則true amax≤CG，所以任何可cert row必滿足|f_i|<G/254。這個只看gate的必要條件可界定最樂觀cert pattern，並不代表真正seed達得到。一般U_i不等時同理可用B_i=|f_i|U_i、Bmax作非常鬆的amax上界；實際oracle trueamax會更強。

Witness row若產生L，就不可能被該證書skip。故seed有用weights可併入最終必讀Up，不必雙讀；真正新增成本是早期dispatch、fullinput reload、可能頁內overfetch及保存seed outputs。若未來lowering重讀seed頁，要照實支付，不能算『免費oracle』。

新增exact bound選項：用已完成gate dot作control variate，Up row分解為λ·GateRow＋residual，界為|λ D_gate|＋||residual||·||x||；λ=0退回原界。若成對gate/up weights不近共線，幾乎不收緊。需raw integer gate dot或保守反推F32 interval的新數值契約；不把它當現有AiF免費能力，也不在首個stage0無界擴family。

品質價值不能靠exact性自行豁免比較：使用者未要求bitwise exact，CATS／TEAL／FloE及arxiv2605.08575的dense-gate→sparse-up/down皆需同品質／同硬體比較。若近似方法在相同實測品質預算更快，而此方法只有保守證書，應如實報其窄保證而非自動宣稱投稿成功。

A6/A4 shadow是同proof的備案：Qmax31/7使zero deadzone變大；但只能作指定A8 forward輸入上的offpolicy conditional分析，不是其自身free-running routing／quality／performance。FP8的tiny underflow deadzone反可能比A8更窄，不能泛稱所有lowprecision格式都受益。

## 19:42:56 UTC：建議的 pre-experiment protocol（僅設計，未實作）

Root指定首個stage0：既有TRAIN32 frozen selection的前8個prompt IDs，每題前8 decode，DeepSeek-V2-Lite（shared）＋OLMoE（無shared），兩模型各自全部layers；G4若列第三shape須在capture前明定。所有原始prompt、weight/format/source hashes及ordered routes都保留。不得用既有Q4/Q8 GGUF路由冒稱新W8模型的路由。

首階段先給不部署的免費真amax witness oracle，即L=true canonical amax。對global exact-L2、預列block-L2、gate-conditioned bounds計cert上界；若連這個明顯寬鬆的oracle也幾乎不能cert整個物理read unit，才有理由停止此boundfamily。這不是seed policy性能，更不證明任意certificate機制不可能。Stage0不包含任意新mapper搜尋。

預列輸出：每prompt/layer/expert的真A8 zero比例d、oracle certificate比例p、p在真zero中的recall、保守bound相對真Up的鬆弛分布、任何certificate false-positive（必為0），以及4/16/64KiB候選read-unit全cert比例。保持scalar／stored bytes／可省array read／wave／native makespan分開；不以scalar p直接宣稱delay。Prompt是cluster，64 decode positions不是64獨立任務。

Strong comparison首先是完整GU＋全部真A8-zero Down sparse。理想係數work基準3−d、候選3−d−p；seed、metadata、額外phase與block/row/BD/BL restrictions只會讓實際部署成本更大。是否值得下一stage，不用使用者未設定的10%硬門檻；應看oracle gain是否有足夠空間支付不可避免的setup、是否在有／無shared都出現、以及相對同品質近似prior是否有值得主張的價值。

可預列A6/A4 **shadow** diagnostics（只換Down input quantizerQmax31/7，GU仍A8）；其A8-forward inputs/routes是offpolicy，不是W8A4模型自身品質／routing／TPOT。若shadow有潛力，須另做真正該格式free-running及品質matched comparison，不能直接把shadow重新標成full-model結果。

較緊的gate-conditioned oracle可完全整數表達。記 G=wg·wg、V=wu·wu、C=wg·wu、X=x·x、d=wg·x，u=wu·x，則

`(G*u − C*d)^2 <= (G*V−C*C) * (G*X−d*d)`。

取右邊exact integer square root B（範圍需uint128），可得u的整數interval `[ceil((C*d−B)/G), floor((C*d+B)/G)]`，並與globalL2交集；G=0另fallback。這是普通條件投影Cauchy，不主張新theorem。G/V/C metadata約12B/row，計算及rawgate dot取得都須計費。Stage0 rawgate只作oracle；部署可研究按canonical monotone F32 scale反推d的保守interval，不能假定原AiF已有raw-I32。

便宜部署比較可存每row ceil_sqrt(W2)並token只算一次ceil_sqrt(X2)，二者乘積保守但較鬆；避免每row isqrt CPU成本不必要壓垮global bound。Block-L2在isotropic情境通常只微幅收緊而metadata增加，實際是否有異質性需資料，不先下結論。

## 19:46:42 UTC：Gemini 互審、F32 合約與 effective-work 分母

已讀Gemini新2.2–2.7。其coordinate-deflation有可研究價值：TRAIN凍結top-r輸入座標S，逐row存w[S]及其餘exact norm，線上用已知dot_S加補空間Cauchy界。r=16約16B weights＋4B norm／row，相對H2048約0.98% Up容量（不假定2B norm可無代價精確向外表示）。必須由TRAIN離線挑S；若每token動態選top-r輸入通道，不能還假裝只存固定r係數。此family可與global/block/paired-gate界取min，未知真實input能量是否集中，不先套50–90%常數。

不接受其sphere-packing段落作本專案no-go theorem：未給精確纖維／分布條件與模型有限weights保證，而且已完成gate dot提供動態觀測，超出只有靜態metadata的問題設定。『U1唯一能收緊』『四項缺一不可』均過強。近isotropic下BH通常比globalL2鬆，我們本就取min而非強迫替換；高斯尾率可提出失敗假說，不能充作實際模型結果。

目前沒有所有五模型都常態L/σ_up≥100的證據；這不等於已證有／無shared的兩類都沒有可用層或模型。使用者也未要求所有五個checkpoint等幅改善。正確決策是明確的有限family optimistic oracle篩選，若尚有真unknown則可讓一次正式實驗判斷，而非依缺少普遍性證據提前宣布全域失敗。

U3固定hot-witness tile與Gate同phase讀，確實能把額外seed barrier併入原Gate phase；同輸入q_x也讓同層Gate/Up保留operand有合理硬體路徑。但PB跨command保留須明列新契約，不能占用AiF已作weight/input的latches再當免費staging。『hot witness weights百分之百必要、零浪費』不成立為全域宣稱：seed在自身L下不能被證零，但後續trueamax更大仍可能使其canonicalA8為零。只能說它是本policy強制Up工作，可避免重讀，並不比full-Up baseline增加有用係數bytes；其放棄的oracle skip、padding和commands照算。

數值證書可不使用模糊的實數margin：令B_up為保守整數dot幅度界，以原`scale_output`的單調正scale產生F32 Up幅度界，再用實際已完成GGML gate activation的絕對值相乘，對上界做向外round。令s_low=fl32(L/127)，當s_low為合法正有限值且實際F32除法結果`fl32(A_upper/s_low) < 0.5f`，std::round ties-away一定給0。Trueamax≥L使true scale≥s_low，monotonicity提供安全性；非有限、underflow/FTZ不明或不符合ABI時fallback。實際gate/activation binary op順序、編譯fastmath/FTZ設定必須同reference，不能以另一libm代算gate。

符號統一以免scalar偷渡native：`z_down`是真A8 scalar零率，`z_cert`是scalar證出率；`d`是既定layout實際可免的Down工作比例，`p`是額外可免的Up工作比例。只有在等projection、同工作單位的理想模型，才寫3−d與3−d−p。H4096 Down全H×1column是4KiB，可一zero skip但每owner需16KiB I32 accum並付columnshard reduction；原rowslice256×16cols則需16zeros共組。強baseline須同SRAM選最強dense/sparse實現，不能把z_down直接帶入d。

## 19:51:35 UTC：Stage0 draft 審查與額外強控制

已唯讀審Root的`results/research-2026-11-03-no-cr/gate_zero_stage0_protocol.draft.json`。支持兩模型、all-layers、完整W8A8 prefill/decode、first8 frozenTRAIN×8decode、前4fit／後4score，以及trueM oracle＋paid fixed-page witness兩個層次。此時仍未實作／執行；19:58及黑臉通過之前不launch。

凍結前建議明定：prompt token cap128對chat-template/BOS後序列取哪128tokens；fixedfirstUp page為logical row-major16KiB的哪些完整rows與跨頁/padding策略；deflation top16只由first4 TRAIN的decode8共同q_x能量選，不能因K routed copies/shared callback重複加權（若決定含prefill也須明定）。完整prefill使用W8A8，是forward正確性要求，不代表所有prefill樣本都要拿來fit。現有G4 integer reference硬編I512與layer<40，兩新model需shape-general hook並驗target operator coverage，不能只換模型路徑。

直接gate-zero強控制：實際GGML post-SiLU f_i==0且Up可證finite時，乘積已精確0，不需U／witness即可跳Up。raw gate dot恰0、underflow或既定FTZ皆可能造成此情形；數量未知。其收益不能歸到新的nonzero-gate A8證書，baseline須含此控制並與fullGU＋actualA8zeroDown取最強。GGML要對完整I向量呼叫原silu，subset壓縮可能改SIMDtail/scalar exp路徑。

Terra補witness-free標準branch-and-bound：若Up區間[lo_i,hi_i]不跨0，可用最小幅度乘gate建立activation lowerbound，所有rows的最大lowerbound可當L。對稱L2 intervals皆跨0時L=0退回真seed；coordinate/gate-conditioned或partial-bit區間可能不跨0，未有資料證明。此處不是新theorem。

獨立核讀PADE（arxiv2512.14322v1）§IV-A/C與accuracy tradeoff：已用已讀high-bit dot與unknown-bit保守interval，並用目前最大lowerbound決定是否繼續讀lower-bit，tiling重用partial。它的threshold服務approximate sparse attention，不是本A8 exact-zero／NAND；但若採Hi4-Up版，不能把branch-and-bound/max-lower核心思路主張首次。主版本仍是zero-bit cheap-bound，不默默切換為已付半Up的版本。

Gatefirst phase penalty不能固定當一個28us，也不能宣稱U3必免費消除。純理想16chip均分例：OLMoE每chip GU256KiB=4waves，Gate128KiB2＋Up128KiB2同4；Deep GU352KiB→6waves，Gate176KiB3＋Up176KiB3同6；Qwen GU192KiB3wave，Gate96KiB2＋Up96KiB2成4。Witness塞Gate padding可救部分case，但需真whole-row/page/block/BD/BL地址，少數8–32seed不一定填平。跨selected experts共用x聚合會改ceil，又受固定地址約束；這些算術例僅提醒成本shape依賴，不是native模擬結果。

## 19:55:31 UTC：Root 要求釋出實驗執行者 slot；最終 handoff

Root指示此刻保存handoff並結束本agent回合，19:58之後才建立Executor／實作，仍沒有任何新候選實驗讀取。**本agent Gate-first 實際研究為19:26:00–19:55:31，29分31秒；不宣稱自己已滿30分鐘。** 候選整體仍由Root／critic／Gemini保持思考至共同deadline；Root必須確認真正研究時間及審查通過再launch，不能把本agent提早結束或其他agent分鐘相加冒算。

最終建議：支持凍結後的bounded TRAIN-only stage0，用一次正式實驗判斷兩類真模型是否存在effective Up read-unit headroom。這不是positive預測，更不是論文成功；不應只因高斯假設便thought-reject，也不接受Gemini自行新增15%／8%／5%硬門檻或G4／重排mapper／dynamicwitness擴張覆寫Root協定。真正尚未回答的是：true zeros是否足夠、廉價保守bound能抓到多少、page共證是否足夠、paidwitness和metadata／splitphase是否吃掉收益。若最強預列freeM oracle也無有效整readunits，精確停止此boundfamily／reference regime，保留negative，不用A4 shadow救A8主張。

數學與降低契約目前自洽：非zero-SiLU gate配合causal amax lower witness可在Up未讀之前證canonical dynamic-A8為0；非循環owner論證與F32單調判準成立於明示finite／rounding／ABI前提。已有大量直接prior涵蓋gatefirst、partial-bit intervals、max-lowerbranchbound、flash selective load與coactivation排布，因此新意只能由這個特定cross-projection quantizer證書與可實現ordinaryNAND收益支撐，不能從『exact』兩字推論可投稿。

額外Executor提醒：existingDeep metadata確認27blocks、leading_dense_block_count1、denseI10944、routedI1408/K6/E64，以及shared_count2合併tensorI2816。Amax／A8 scale按實際tensor向量，不可把shared拆成兩個I1408任意量化；dense首layer和router `gate_inp`的數值scope須逐tensor明定。OLMoE16layers/E64/K8/I1024無shared，source Q4_K_M重canonical為row-W8是新reference，並非FP16checkpoint直接W8。要保存source/canonical/scales／hook hashes及完整coverage。

Gate與Up必須從同canonical q_x及其scale產生；collector對實際GGML full-I activation路徑做一致性核對，不能以subset SIMDtail或另一exp實作代算。Down quantizer輸入bytes與scale、全FFN輸出／logits／routes的exact檢查分層報告；stage0僅完整reference capture和證書counterfactual，不把CPU hook速度當AiF native latency。

本agent只改本新筆記；沒有collector、codec、候選實作或新數值probe。Signed-sketch原版為thought-reject，LUT/frontier只是旁支思考，本次新實驗數截至handoff仍 **0/10**。後續候選正式30分鐘計時須等真正本輪實驗失敗後重新開始，不重用本輪／旁支時間。

## 19:57:25 UTC：Root 要求延續至個人滿30分鐘後；amended protocol 最後審查

Root於前次handoff後要求本agent繼續到19:58以嚴格滿足個人實質研究30分鐘；因此前段29分31秒只是當時receipt，不是最終研究結束。19:56:14已重新核讀amended draft；這段仍只有協定／數學／成本審查，無collector或數值probe。

修訂已補齊首128 raw-token/BOS規則、greedy固定8step、完整FFN包含Deep dense首層與mergedsharedI2816、deflation只fit前4題decode8且共同input不重複加權、fixed首16KiB Up page的8完整rows、seed整頁強制read並扣其原gatezero機會，以及perexpert獨立alignment的logical機會模型。H2048兩模型的4KiB/16KiB單位分別恰為2/8完整rows，無需靠跨row/padding猜測來過關；仍不等同SOML合法物理wave。

事前negative以兩模型freeM oracle、最強預列bound的額外scalar skip≤1%且無額外完整16KiB機會，secondarydeflation也滿足，才限定淘汰這個bound/reference performance regime。已提醒Root把mean明定為按實際selected Up rows／bytes加權的pooled比例並另列prompt均值；不能先平均dense/shared/routed ratio再推總work比例。0.334%僅理想equalprojection dense係數工作量，非native delaybound。僅16KiB陰性、4KiB或scalar混合的case依協定仍屬同輪inconclusive，不能擅自換輪。

未付的cheapbound成本只會削弱oraclepositive：每row isqrt／conditional uint128、fixed16coeff deflation MAC、metadata DRAM和same-budgetcache、全gate後CPUmask、額外command／phase，都不是免費。作量級提醒的純尺寸算術：Deep每MoElayer selectedI=6×1408+2816=11264，含26MoE與首denseI10944，三projection總約1.87GB W8係數／token；理想ordinary sensing aggregate16×64KiB/28us下約50ms。只省1% Up對此floor約0.17ms，約每MoElayer6us額外criticalhost成本便可吃掉。這不是實測、不給PE吞吐保證，也不是最終native latency模型。

安全性沒有因此放寬：deflation partial只供bound，不當作原PE免費少算或自動重建其F32；gate-conditioned仍明列rawdot oracle能力與合法整數range；任何boundmetadata壓縮、float換算、FTZ差異或invalidGram都需向外保守／fallback。未發現amended protocol裡必然不安全或被強baseline完全支配的邏輯矛盾，故維持支持boundedstage0，等待最後clock receipt而不自動launch。

## 最終 receipt：2026-11-02 19:58:04 UTC

本agent Gate-first正式實質研究起訖 **19:26:00–19:58:04 UTC，32分04秒**；中間前次handoff後按Root指示繼續核amended protocol、原生幾何與未付成本，未用sleep湊時間。正式30分鐘已超過；思考與時間gate只准Root連同critic決定是否launch，不代替protocol freeze或實驗授權。截止此receipt，本agent沒有新候選實作或probe，本次新實驗仍0/10。最終建議維持：支持凍結amended TRAIN-only stage0，風險與未知照前記錄；無效或mixed結果按事前bounded判準及同輪修復／follow-up處理，不假裝全域不可能。

# Round2：第一輪有效陰性後的新搜尋

## 2026-11-02 20:22:09 UTC 新搜尋 receipt（尚非正式候選時計）

已讀`round01_scientific_decision.json`、campaign state與stage0 result。Round1於20:21:28正式判定scientific failure，已用1輪／剩9。Deep／OLMoE primary extra Up各0.10120%／0.08996%，預列deflation亦0.11457%／0.09534%，兩者0整16KiB新增機會，數值與coverage通過；失敗原因是性能價值，不是工具或數值錯誤。不再沿便宜norm-gate小修；A4 shadow也不當新成功。

新搜尋20:22:09開始，前輪／sidebar時間不沿用。採納正式候選後仍由Root共同凍結新的>=1800秒實質思考時計；以下只是候選pool、未有實作或新weights/activation probes。

首批五種實質不同機制：

1. **chip-side lossless／entropy-constrained W8 stream decode**：每sense存的係數更多，直接減sensed bytes。必須勝同品質低bit量化＋同能力entropydecoder；既有樣本entropy7.3–7.7bit限制rawW8無損headroom，普通codec移植不自動有新意。Decoder／ECCcodeword／變長offset／plane空洞都要算。
2. **α-conditioned route-coalition student**：對常見完整共選集合預存可吸收liveα的compound FFN，容量換K→一個student的selected必要參數下降；罕見route fallback原experts。這不是weight平均或fixedα表，但要付額外student容量／訓練／distribution-shift品質；必勝直接merge／prune／sharedlowrank。未知精確route組合hit率與student相對topK-reduction的quality優勢。
3. **temporal activation delta／expert output state reuse**：同expert跨tokens只對變動quantizedinput columns讀weights、更新cachedoutput，或在品質門檻內近似復用；NANDcolumn粗粒度與長期誤差是瓶頸。原始routerlocality不是新：官方TemporalMoE已用訓練rolling residency限制每token最多換一expert；不能把prefetch改名新法。
4. **exact speculative causal frontier＋page-stationary multi-context**：一sense服務數個targetverification tokens，同時盡早完成leaderlogits，firstreject取消futurework；不是固定microbatching。必勝同draft／horizon／SRAM／PE的fullbatch expertGEMM、所有固定chunks與earlyabort。未知真正causalDAG是否允許material超越chunkedbaseline，以及q×MAC、PBinputs／outputcontexts能否付得起。舊19:23側支不抵新時計。
5. **structured expert operators（Kronecker／Monarch／BTT）＋in-flash vector-state reuse**：將densematrix權重必要量換為可重用小因子、改運算演算法，不只是lowrank／layout；付intermediate vector SRAM／factor通信與真品質。直接prior非常強，需先確認是否還剩架構新必要性。

目前偏向第4的有限PB與firstreject causalfrontier問題作投稿主方向，但尚未採納。第5不能因『非lowrank』就忽視直接prior：已獨立讀NimbleMoE EMNLP2026官方頁，明確routecoactivation clustering＋sharedlowrank＋Monarch residual補償，且有TPOT；BTT-MoE NeurIPS2024也已structured input-dependent experts。Generic結構化expert方案基本被覆蓋，需要獨立NAND-specific問題才保留。

Terra提供KDEM NeurIPS2025直接近鄰：teacher=Σα_i M_i(x)、student=M(x;Σα_i θ_i)，MSE蒸餾動態αensemble；但runtime仍合併原權重，需要全讀，和預存α-conditionedcoalitionstudent的flash必要量不同。故不可宣稱ensemble蒸餾概念首創；若保留，剩餘假說僅是額外容量換可讀一個compoundstudent、同品質且route組合hit足夠，尚無證據。

## 20:29:20 UTC：frontier 必要性反例、whole-model 搜尋

獨立核讀FASER2604.20503§4.2/4.3：已有intermediate-logit earlyexit與token-prefix frontier chunks，FlowSpec2507.02620也有step-wiseverification／causalprune與pipeline。候選若僅把它們和普通page-stationaryGEMM拼起來，沒有新必要性。

進一步做純因果／resource toy推演：Layer1 leader／future routes disjoint、Layer2共C時，早leader且C只讀一次，需要future-L1能在C page存活時由不同stagechips完成。這正是既有pipeline能做；在AiF全matrix rowshard16chips、ordinary sensing主導時，future-only前層reads與leader上層reads爭同chips，PB多input不會憑空增加array state machine parallelism。若futureinput本已ready，附帶MAC收益就是普通groupedGEMM已有。小pagecache也不能保整個Cmatrix。因此genericfrontier目前傾向thought-reject，Root同意不為它硬湊30分鐘；尚未正式候選、未消耗新輪。

擴大到wholemodel：LMhead的dense全vocab讀取可用MIPS／clusterretrieval省掉，但已搜尋到FlashHead的作者文章，two-stage equal-size token clusters＋multiprobe＋selectivequantization，故generic輸出候選檢索不是新；尚需primary paper核讀與真wholemodel分母。QKVprojections、attention和CPU／PCIe barriers也可研究，但AiF本來就有head-level QKV/MHA overlap與hosttensorparallel，controller融合SiLU不是自動新意。

另保留但不硬當新方法的靈感：精度speculation／wholemodelerror feedback、certification-aware quantizer margins、NANDshortread＋ECCretry、VQindices→productlookup計算。每一項都有很近的AnyPrecision／PADE／AR2／CodeGEMM／EVA等prior；需找到它們不能解的物理矛盾或quality/latency Pareto，不能只加controllersolver。

## 20:34:59–20:37:41 UTC：集中有限 overlapping pair bank（正式時計待 Root）

Root暫採黑臉提出的有限pairbank主軸：保留原E experts，加M個pair學生`S_ij(x,t)`，其中`t=α_i/(α_i+α_j)`；只在原topK同時有i/j時，可用`(α_i+α_j) S_ij`替代兩者。TopK內選disjoint matching，未配對及不合格pair全部fallback原experts。這是capacity換selected參數必要量下降，不是改原router、globalpartition或full-K組合table。Root會另通知共同>=1800秒起點，目前不把20:22廣搜或舊coalition/sidebar計時抵扣。

成本先以同weightbytes b計：原expert約3HIb；pair學生hiddenJ、普通concat一個t時約3HJb＋2Jb，原pair為6HIb。一條pair省`1−J/(2I)`；每token matching q條時routed讀取理想減幅`q(2−J/I)/K`。J必須<2I才可能省；M128/E64/J=I額外bank為2倍原routed容量、總3倍，不可叫免費。兩現有模型在1TB AiF約333GB LSB係數預算內可能放得下，但須含ECC／spare／baseline同物理容量的replica／variants用途。

普通concat-t學生不保對t affine或端點一致；一個affine變體`S=A(x)+tB(x)`用共享GU、雙Downhead，J=I時4HI vs原6HI，只省1/3（不是1/2），若J需≥1.5I便無讀取收益。故要比fixedmean-t student與barycenter＋cheap lowrank(t−t0) correction，不能將αconditioning本身當新。

樣本效率是核心hardrisk：uniformE64/K6任意pair共選機率`K(K−1)/(E(E−1))=30/4032≈.00744`，每1024tokens僅約7.6pair teacher examples（手算非實際觀察）。每學生百萬參數若只用少量TRAIN就fit，完全不能證跨prompt或free-running品質。先證matching hit／readheadroom再付訓練，需考慮足夠conditional teacherdata、parameter initialization與跨prompt/域驗證；不能teacher token同批fit同批score。

已獨立核MergeMoE2510.14436§3.2/4：它為globalpartition、sum原routerweights與fixed-B輸出合併；exactconcat hidden會成員數倍大而不省參數，再以T1/T2/T3降維近似。這提供Goldilocks強控制但不是overlapbank直接同義。AAAI26 B-Distill為全局MoE-to-MoE均衡蒸餾與routermatch；NeurIPS25 KDEM已有動態αensemble student概念。剩餘待證差異是conditional co-selected domain上的可壓縮性，且原solo功能可保留fallback，是否勝samebank/simplemerge及globalcompression品質-Pareto。

此刻所有失敗仍為thought rejection，不新增實驗輪；正式Round2尚無新probe。

## 正式 Round2 候選起點：2026-11-02 20:38:00 UTC

Root共同凍結pairbank正式起點20:38:00，最早21:08:00才可能進實驗；Gemfocusedblock已join，仍需黑臉／協定gate，時間到不自動launch。20:22廣搜與20:34集中預研都不抵扣。

## 20:40:22 UTC 起：必要性、條件域與組合 headroom

Pairbank的可防禦機制不是發明distillation。可構造conditional-domain例：E3/K2 routes只出現{1,2}和{2,3}；每個jointdomain的pair函數可由窄student擬合，solo域仍需要各原expert的獨特功能。Overlapbank S12/S23加originalfallback可以兩種route都減activeweights，單一globalpartition不能同時保兩條重疊pair。這只證機制在某些函數／route分布下不矛盾，不證真MoE有此Goldilocks條件；samebank fixed-t學生若也做到，應將收益歸bank而非αconditioning。

強反例包括：pair函數彼此無關而J需2I（不省）；jointdomain本身與solo域相同（globalmerge夠）；t幾乎常數（fixedstudent夠）；t幅度大且teacherdifference高rank（conditioningstudent需大寬度）；pairhit集中同hub（edgefreq高但matching小）。若銀行confidence判斷需先跑student甚至teacher才能做，fallback doublework會吃gain；部署應在原router後、讀student前以預先凍結的cheapmetadata／t域決定，不宣稱新證書。

Uniform route的純算術（非模型觀察）：E64/M128的4-regular pair graph，K6時expected inducededges=.9524，expected adjacentedgepairs=.1843。因任意圖maximum matchingν≥edgecount−adjacentedgepairs，有Eν介於.768與.952；J=I時routed係數省12.8–15.9%。Deep shared總I相當額外2個routedexpert units，則總FFN理想係數省9.6–11.9%，未乘wholemodelFFN比例。K8時Eν介於1.262與1.778，省15.8–22.2%。Graph若變高hub，此界與真收益可急降；不能只報edgehits。

K2/E8是另一regime：存全28pair，每個top2都有一條bankedge；J=I則可省50%routed係數，但額外bank容量3.5倍原routed，總4.5倍。Mixtral原W8 routed約45GB、此bank約158GB，再加原weights約203GB，可能仍在AiF333GB LSB預算內；這僅尺寸例，不擴stage0模型、不啟新weightprobe，學生訓練和fullmodelquality成本很高。

KDEM NeurIPS2025§4.1/AppendixC.2獨立核讀：其per-graphα合併原θ，推論merge僅onceperdataset，攤大量nodes；不是per-tokenflash預存pairbank，但αensemble蒸餾概念須承認。PCoMoE EMNLP2026§4亦獨立讀：source-grouped sharedGU＋multiDown composition、保diagonalfallback與learnedcompatibility，對雙Downhead變體是很強控制，不能宣稱firstGUreuse。

Concat t若讓Gate/Up成H+1欄會破H2048自然rowpack。最小hostlowering可將t欄拆成2J metadata向量dG/dU：NAND主matvec仍J×H對齊，Host於既有GU回傳後做g+=t dG、u+=t dU，再SiLU；付metadata／2J multadd及其量化scales，不新增免費能力。FiLM更複雜則另付參數與hostops。

## 20:42:47 UTC 起：Goldilocks 強控制與 tagged-feature 學生

黑臉補零容量強控制：pair同時selected時只跑Fi並乘α_i+α_j，solo仍原路。若jointFi≈Fj，此控制已吃收益；pairbank需勝conditionalrepresentative，而非只比globalmerge。還須同activebytes lowerK＋相同offlinebudgetdistillation／Matryoshka控制，避免投入學生訓練卻不讓baseline訓練。

新學生變體（尚純推導）：從原pair的2I hiddenneurons選／蒸餾J個taggedfeatures，每個固定標origin i或j。NAND做普通J×H Gate/Up，Host把i-tag的SiLU×Up feature乘t、j-tag乘1−t，再一個H×J Down。未量化實數function為`S=Σ D'_r t φ_r`或`Σ D'_r (1−t)φ_r`，所以對t affine，rawweights仍3HJ＋J-bit tags＋J次hostmultiply，而非必須4HJ雙head。4HJ是另一架構選項，不是affine必然下界。

J=2I且用原實數weights可表示pairteacher的weightedconcat；J<2I則是conditional-domain neuronpruning／distillation，並無品質保證。可考慮先固定GU、ridgefitDown，降低罕見pair的全參數訓練成本。它與MergeMoE concatcompress、conditionalprune很近；若保留，真正差異仍是overlapbank與solo fallback，不是假稱新nonlinear蒸餾。

數值不可過稱：若先乘t再用單一dynamicA8，scale／round依t變，部署函數不再嚴格affine；可以另定approxcontract並量品質。若要固定quantizedfeatures下保affine，需origin兩個input scales、Down兩個I32partial bins，各自scale後t加權，付額外2H輸出／context。不同原expert的Down row scales也不能合併後自動保持既有W8bitexact；requant是一個新studentformat。上述只是思考變體，未寫collector／未跑weights或activation probe。

## 20:46:06 UTC 起：表示與銀行分離、精度／波次公平

獨立核IntBMoE2609.21346§4.2/4.7：有限codebook blocks可offlinecompose/cache、token只執行少量blocks，故『預存recipes／空間換compute』是既有思想。它改原router、fromscratch、sharedfullpool組成，與drop-in co-selected overlappingpair＋solooriginalfallback仍不同；不能因未同義就宣稱整個思想首次。Nimble完整正文的G2是static perfectmatching，alloriginalexpert IDs/router保留而共享低秩basis，liveαlatent相加再Downbasis，亦為必要控制。

應做factorial而非混淆兩種貢獻：bank拓撲（staticpartition vs overlap+solo fallback）與representation（mean／conditionalrep／ridge／tagS／Nimbleviews）。給Nimble同pairbank可測studentrepresentation增量；如果Nimbleviews最佳，銀行主張不因此自動消失，但必須重寫為conditionalmaterialization架構＋既有compressor，不可宣稱新distillation。Fallback只是保留originaloperator；前層student誤差已改下層x/routes，並不保證整個free-running模型退回teacher。

同精度只是第一層公平，還須matchedquality precisionPareto。純理想g16/W8排程算術：DeepI1408/H2048每expert GU/chip352KiB→6waves，Down176KiB→3waves，總9；OLMoEI1024的GU256KiB→4、Down128KiB→2，總6。J=I的W8student替pair才有一expert unit的原始work節省。

若比較W8student和兩個W4 originals，DeepW4每expert GU176KiB→3、Down88KiB→2，兩者10waves而S8為9，僅padding-level一wave差；OLMoE兩個W4是2×(2+1)=6，與S8完全相同。這些忽略group scales／解碼／sameWL等，只是說明不能把『一個student』當必然半讀。StudentW4 vs兩個W4才對應結構半讀，但若兩個W2品質已夠也須比較。真正native模型需格式bytes（group scales／codebooks）和PE吞吐，不是名義bits。

整個正式候選仍思考中，尚無新候選data/probe；最早21:08 UTC且root/critic另行clear。

## 20:48:54 UTC 起：容量、訓練與 batch 反例

M128不是1TB物理容量的必然上限。OLMoE routed W8 `3HI×64×16≈6.44GB`；全2016pair、J=I/W8的bank約203GB，加原weights仍可能在~333GB LSB預算。此時top8總可perfectmatch4，cooccur coverage不再瓶頸，但需約32k學生的龐大離線訓練。Deep全pair W8約453GB超額；W4學生約227GB加orig可放，但必須matchedprecision品質比較。因此cost是capacity／active reads／offlinefit三軸，不可用小M失敗概括所有bank。

Batch為真反例：單token省expert數不代表多token union少讀。若每token平均q條pair、學生寬度比γ，粗略獨立模型的原expert union約`E[1−(1−K/E)^B]`；bank路徑約`E[1−(1−(K−2q)/E)^B] + γM[1−(1−q/M)^B]`。B大時大量不同pairviews會把同一originalexpert拆成多份，bank可更慢。這些只是分布假設下的尺寸式；可限定B1decode，prefill仍teacher，但不能外推多請求／spec驗證吞吐。若batch-awarepolicy包含all-original選項，應按真readunion成本fallback，而非固定每tokenmatching後才group。

資料可攤薄但不免費：一次teacher token已算K experts outputs，可對所有入bank的co-selected pairs產生targets，不需為每pair重算teacher；平均樣本量仍由pair概率限制。Synthetic t對同一x只增混合比覆蓋，不是新的獨立input，也不保證routingconditional-domain泛化。若用top2K近路由樣本，新增teacherqueries與distribution mismatch須記錄。從零訓練所有bank學生的optimizer容量也很大，需layerwise／pairwisefit或強initialization，不能靠『只局部』便稱成本小。

已讀Gemini第一block並傳回具體修正：C3兩個rankI Downimages正交要求2I≤H，只適合如OLMoE的構造；DeepI1408/H2048不滿足、MixtralI14336>H4096更不能由outputrank推出J=I必失半能量。可保留有限r_eff≤H/2的counterexample，不泛用。其Beta-t／near-routeaugmentation不自動解樣本飢餓；未量化affine／SwiGLU homogeneity／CPU<1us／metadatafree／submodular等問題由Root共同修正。

## 20:56:31 UTC receipt 後：Bounded Stage0 提案與有效判負範圍

獨立再讀corpus protocol，既有本地合成routing corpus有200 TRAIN／100 TEST，不是只有TRAIN32。題庫不是品質benchmark，prompt templates與任務部分共用；可供局部pilot的prompt分離資料，不可將其score當正式模型品質。建議事先hash將200 TRAIN分120fit／40valid／40score，100原TEST留後續凍結測試。資料取得仍需Root准入後新collector；目前沒有讀取新候選activation或運行任何候選分析。

向Root提出 bounded Stage0：兩模型Deep(shared)／OLMoE(no shared)、B1decode、原shared保持、M128主設定、J/I={1,1.5}與W8學生；W4學生可作事前列明的格式控制，不用posthoc亂換precision救結果。A是TRAIN-only route／原生geometry headroom，保留maxmatching而非edgecount、16KiB waves、命令與非FFN分母。完美學生的hit只是樂觀上界；若M128差，只能否決這個M／width／workload，因OLMoE全2016pair學生仍可能在LSB容量內，不能概括所有bank。

B是4個事先定layer（Deep首MoE與中間，OLMoE0／8），每layer按TRAIN頻率與distinct-node規則選2個pair，不看score品質挑好看的pair。至少比較teacher-GU tagged-prune＋Down residual-ridge、固定t0學生、conditionalrepresentative、samebankNimble／lowrankview、SPICE啟發的pairbase `m Fi+alpha_j B_ij A_ij x`、CRISP／symmetricwidth＋Downridge，以及兩W4 originals。各法都得到相同pairselector／fallback／容量與phase能力；metadata、額外串行matvec、student量化與nativepadding付實際成本。

SPICE2608.21240v2 §4.2已獨立讀取：原文為sharedbase＋perexpert低rankresidualbank＋原expertfallback，並非pairbank；Terra據此提出無shared的pair差分控制。rankr線性差分含2Hr係數，r64／I1408僅相當約3.03%原單expert係數（純尺寸，未含command／padding／品質）。不能把它當已證支配者，但不比較便不足以說整個MLP學生必要。

資料足量不是任意小樣本門檻可證。初步建議每pair做N→2N learningcurve（例如128→256 conditionalfit tokens、至少8個fit prompts；valid／score各至少8prompts），若樣本不夠、score分布不足或曲線仍改善／optimization未收斂，記inconclusive並擴資料或同輪修復，不能以undertrainedfewsamplefail判整方法死亡。Synthetic多個t只有混合比增廣，沒有新的獨立x。若以原GGUF forward捕x/routes再對選定pair做canonicalW8教師與學生，此為operator-level/offpolicy；不能冒充完整W8free-running品質。若從canonical全模型捕則數值契約另凍結，採集CPU成本要計。

有效陰性只能是：足量資料與預列兩個width／regularization、合理初始化及收斂檢查後，此frozen學生family在預定層／模型／資料上全部被matchedphysicalread strongcontrols的score誤差Pareto支配；不宣稱pairbank普遍不可能。少數pair的局部非支配結果也只是支持下一stage，仍需全層訓練、真正free-running品質、route／matching變化及wholemodelDAG；完美matchingheadroom或低MSE都不等於成功。

資源：一個Deep J=I學生約8.65M參數，單pair layerwise處理可控制RAM；CPU8threads是預設，GPU現有job不停止。沒有實測前不保證訓練耗時或Host匹配<1us。正式candidate仍20:38起、最早21:08且另需Root／黑臉freeze；以上僅研究提案，尚無新候選code／probe。

## 21:01:04 UTC receipt 後：tagged-prune 的 separability 與 candid thought-reject 建議

Root與黑臉正確指出：原Stage0每layer兩條disjointpair只測已有representationcompression，沒有識別overlap／partner-domain必要性。若仍做辨識，至少須TRAIN-only deterministic wedge `(i,j),(i,k)` 且兩joint域與i-solo／otherpartner域都觀測，與rank／alpha條件single-i view、CRISP固定prefix、pair-specificviews比較。然而這個修正仍不足以救具體tagged-prune方法的新物理主張。

明確分離證明：學生從原i/j各選J_i／J_j個GU features，J=J_i+J_j；寫其Down為`B=[B_i B_j]`。實數輸出

`(alpha_i+alpha_j) B [t phi_i ; (1-t) phi_j] = alpha_i B_i phi_i + alpha_j B_j phi_j`。

因此可將同一學生重新命名成兩個partner-specific compact original views，係數仍為`3H(J_i+J_j)`、同等bank容量。若jointDownridge改了B_i/B_j，baseline也存完全相同改後的兩個columnblocks，不能被限定必須原Down；否則是不公平禁止student得到的離線訓練。

部署量化亦不能憑名字製造必然差：W8 blocks可共享原學生每row scale，Host按照相同concat／tag／amax生成完全相同的A8 bytes，Down兩個columnblocks以相同順序進同一I32accumulator。將blocks以相同packed physical layout／page order存放，可使權重bits、pages、MAC與輸出bits都同學生。強迫E-only views各自量化／另發命令／分開padding，再稱單student物理勝出，是人為設弱baseline。若量化／accum ABI不允許此fused baseline，則同樣要問學生何以可用；名字本身沒有新增硬體能力。

此證明範圍僅是 **由原features做origin-tagging＋Downridge的具體學生**，不是所有條件化表示／非線性學生／銀行的普遍不可能。它說明這個表示沒有在conditionalcompactviews之外消除必要sensework；比較單一CRISP globalprefix可以顯示partner-specificmasks需求，但同容量多views仍有完全同義控制。實際command／I32contexts／重排的有限差可研究，卻不足以將publication主張建立在『一個學生取代兩expert故少一次sense』。

已向Root坦率建議：**thought-reject concrete tagged-prune＋Downridge pairbank作獨立投稿主軸，不消耗9次實驗機會**。一般新feature非線性pairKD仍未死；但如果唯一理由是『全參數KD也許會贏』，沒有更清楚可識別機制，既不值得拿ridgefail替它判死，也不值得硬跑一輪湊fail。Root同意拆分並要求繼續至21:08以滿真正30min，無候選probe或code獲授權。

另獨立核新primary `https://arxiv.org/html/2609.25809v1`：它針對12個finegraincheckpoints／9families，uniform保留約2/3chosenexperts平均保98.8%11benchmarkperformance，動態allocation在保守預算額外改善有限。不能外推我們兩模型必如此，但說明zero-training lowerK必是強控制，不能拿完整K作唯一品質參照。

## 21:05:02 UTC receipt 後：非線性逃逸條件、不同物理 pool 的反例

General nonlinear學生並不受tagged-separability限制：若其GU新features可在jointdomain直接表示合函數，而非原phi_i／phi_j的tag與線性重加權，J<2I可能有真representationgap。但topK共選不等式只定義full-dimensionalcone，本身不給低intrinsicdimension或低rank。必須另外證真x分布的conditional薄流形、穩定新feature重組與free-running品質；Nimble pairconditionalinput/outputbasis仍應為強控制。嘗試代數例SiLU(g)−SiLU(−g)=g，發現sharedGU／共同Downlatent本就能取得同等權重節省，沒有發現需要bank的額外sense機制。目前沒有超越『fullKD或許贏』的明確新feature機理，因此維持具體方法thought-reject建議，不以ridgepilot裁generalKD。

Root要求至少兩個不同物理pool，以下只廣思，沒有新候選時計：

1. **Input SRAM 解開 AiF PB 角色衝突、ordinary cache-read 與 PE 併行**。把input從reclaimedPB搬到小SRAM，釋PB讓下一普通sense與當頁drain／MAC重疊；read數不減，改的是串行依賴。若tR28us與draincompute10us只是示例，可由38us趨向max(28,10)而非憑空小於tR。實際需AiF PE吞吐／cachecommand／input SRAM bandwidth與area證據。獨立再讀FLINT2608.25062v1§4.1：已有明確page/cachebuffer overlap與burstcontroller，不能把搬PB後的標準doublebuffer當新發明；剩餘可能僅AiF-specific resourceconflict／最低硬體修補，尚無投稿新意證據。
2. **低rank control-variate 加 NAND 整頁 residual importance sampling**。不是找零散column，而是以已知z與離線bounds直接只讀部分完整pages，HT重權I32／FPpartialaccum。它確實減necessaryordinaryreads，但馬上有純理論強反例：若residual pages近獨立isotropic，取樣fraction f造成unbiasederrorvariance約`(1/f−1)||residualoutput||²`；base-only忽略residual誤差約`||residualoutput||²`，要MSE勝base-only已需f>1/2。若有coherentlowrank結構，又可能被SPICE／lowrankbase先吃。故此pool目前不值得凍clock；需新結構證據，不能把randomdrop本身當novel。

另兩個立即遇直接prior的物理靈感：autonomous全FFN／neuron-column fusion已SLIM2507.09201v1§5.4、§6.1明確涵蓋GU→Down與outputSRAMreduce；先做rawMAC再由ECC稀疏更正partialsum已找到2022 primary專利EP4016304A1 Concurrent Compute and ECC for In-Memory Matrix Vector Operations，不能泛稱首創。AI-awareprogramverify按MSB／LSB重要性縮窄Vt distributions也有US20200057561A1。以上只是source／firstprinciples排除，不消耗新實驗機會。

截止本節仍只有Round1一個有效scientificnegative，剩9；pairbank沒有候選probe或code。所有pool被prior或必要性反例淘汰都是thoughtreject，不把它們當輪數。

### 最後 packing 識別修正（21:06:21 receipt 後持續推導）

Wedge不僅不足以識別student新features，也未必識別Flash的conditionalmask packing必要性。對同一i的任意兩個masks A／B，排列為`A\B, A∩B, B\A, neither`，二者都可各成一段contiguousinterval；disjointA／B更可直接prefix／suffix選址，不需要兩份bank。不能將CRISP的固定prefix限制錯當ordinaryNAND只可讀prefix。對Down使用相同column-tile／partialaccum能力，baseline也可選對應columnrange，padding／對齊按真幾何付帳。

至少多個non-laminar、無consecutive-ones-order的maskfamily才有單排列無法全部contiguous的組合障礙；如三個masks{1,2}/{2,3}/{1,3}不能同時各是一段。但即使有多runs，NAND每頁本來獨立tR，若feature自身已填滿page，noncontiguity未必增加necessaryread數，只可能多commands或地址限制；只有多feature共page造成leak時才有readwaste。這又回到既有hypergraphdeclustering／replication／materializedviews優化，不能單憑非laminar就稱大新機制。上述皆純邏輯，沒有mask／activation數值probe。

## 最終 Round2 思考 receipt：2026-11-02 21:08 UTC 後

正式20:38:00起連續進行候選必要性、prior、原生成本、資料協定、separability／packing反例與Root要求的物理pool互審；已越過21:08:00的1800秒gate，未用sleep湊時。結論建議為 **concrete tagged-prune＋Downridge pairbank thought-reject**，不是實驗陰性；general nonlinear pairKD仍未定，不藉局部ridge失敗否決它。此刻候選實驗仍未開始，不能因滿時自動launch。只有Round1已validfail，剩9次機會；Root決定下一個方向與新的正式時計，本輪思考與sidebar不沿用。

本agent只追加此新研究note，未寫候選collector／訓練程式、未跑weights／activation／routing新probe。完整handoff含同bitpattern／同pages的拆分證明、wedge不足的consecutive-ones推導、boundedstage0不當判負範圍，以及不同物理pool和直接prior，已逐項送Root與黑臉。
