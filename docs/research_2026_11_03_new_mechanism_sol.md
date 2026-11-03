# New mechanism screen — local affine expert operators

## Scope／receipt

Source＋first-principles screen開始UTC2026-11-02 21:18:37；不是已採納formal30minclock。Root要求8–12分鐘找可識別機制，不因每個primitive有prior就直接殺；本檔僅notes／source／math，沒有候選code或weights／activation／routing probe。僅Round1一validfail，剩9。

## 候選主軸：shared host projection＋region-dependent affine tangent

對每layer存host projection P_l（r×H，先以orthonormal rows推導），先算z=P_lx。每expert保留originalfallback，另在flash存C個localmap A_e,c（H×r）與anchoroutput F_e(c)。Host以z、離線anchor與cheapguard在讀前決定region；執行

`Fhat_e(x)=F_e(c)+A_e,c [P_lx−P_lc]`。

P_lx已供key使用且所有Kexpert共用；flash只需讀一個H×r operator，不讀GU、不經SiLU／A8-down的跨hostbarrier，也沒有低rank兩factor的第二個NANDphase。這是必要operators與dependency的具體變更，不是把原teacherfeatures重tag。Sharedexpert非必要，任意routedexpert皆可用。

其可能性有明確toy，而非fullKD信仰：在多個輸入region內函數各自affine、每個局部Jacobianrank≤r，但region的slopes聯集span≫r，則一個globalrankr map不足，region-dependentmap一次讀可成立。這只證表示類可能較強，不證真模型有可控C／localcurvature。Co-selection／高cosine都不是低維／低曲率的證據。

## 直接先例與待辨識差異

- [PREP: Input-Aware Expert Pruning](https://openreview.net/pdf?id=nsblPHuc4L) 已用layermean作Tayloranchor，offlineexpertJacobian與Faiss CPUsearch估importance；目的為pruning，不是直接替代整expert輸出的localmap。不能稱Taylor／Jacobian／Faisskey首次。
- [Mixture of Lookup Experts, ICML2025](https://proceedings.mlr.press/v267/jie25b.html) 訓練時experts輸入為embedding、推論按tokenID查precomputedoutput；它改原架構，不能直接代表既有contextualexpert的activationcache可命中。但它是outputlookup強控制／架構基準。
- SPICE已有original＋surrogatefallback；global／regionlocal低rankbase、低K、W4與Nimble都給同P、同總capacity、同hostcache。差異只能由localmapaccuracy／可控C和一flashphase成本證明。
- 尚未找到明確TaylorMoE直接outputsurrogate primary；未找到不是novelty證明，繼續核。

## Canonical數值與部署中心化

Smooth SwiGLU expert `F=D[SiLU(Gx)⊙Ux]` 的Jacobian

`J(c)=D[diag(u_c SiLU'(g_c))G + diag(SiLU(g_c))U]`，`A=J(c)P^T`。

Canonical W8A8會因quantizer／rounding不可微而不同，不能用straight-throughJacobian宣稱exact。可用相同W8dequantweights的smoothJ初始化、anchoroutput取canonicalF(c)，再直接量近似branch對canonical輸出的有限差誤差；這是定義新近似operator，不是bitexactTaylor。

建議以δz=P_lx−P_lc作該map的動態A8輸入，再加FP32 anchoroutput，而非先把大A Pc併進bias再做兩個大數抵銷。δz=0可直接回storedF(c)；附近quantnoise絕對量跟δz縮小。但不同expert／region有不同δz，需要各自smallinputpost／quantization，不能假稱一次globalinputbroadcast。P_lx仍只算一次。所有globalridge／OLS控制也獲同centering契約。

## 高維 constructive objection／pre-readguard

若全部regions共用P，則projectionkerP方向的variation無法被任何A_c恢復；即使C無限，對同Px而不同F(x)仍有irreducibleconditionalvariance。僅用projectednearestanchor可能把遠距離tailinput當近鄰。可加fullnorm與投影taildistanceguard `||x−c||²−||P(x−c)||²`，或完整anchor距離，且都在read前算；不宣稱分布無關的qualitycertificate。

其核心未知應量為output-sensitive coveringnumber／Taylorremainder，而不是rawhiddencosine。Region切換邊界可不連續，fallback只能保留originaloperator，不能抹掉前層approx導致的x/routes漂移；最後必真正free-running品質。

## 21:23:28 UTC receipt 後：成本、初始化與 source 更新

### 一次 flash phase 的實際代價

W8 map係數數量Hr、row scales約4Hbytes、anchoroutput若FP32約4Hbytes。H2048時，r64每map128KiB，r128為256KiB，r256為512KiB，r512為1MiB，rH為4MiB（未含scales）。理想g16rowshard每chip分別8／16／32／64／256KiB；四plane64KiBwave模型下r64/128/256/512均可能1wave（planeoccupancy不同），rH為4waves。此與Deep原GU6+Down3、OLMoE4+2相較有真工作／phaseheadroom；不能當已驗證nativeplan，更不能忽略highg／並行不同expertgroups的最強baseline。

C64? Root建議純推導C∈{1,4,16,32}、r∈{64,128,256,512,H}。以Deep L26/E64/H2048為例，每anchorFP32output8KiB，C16的全模型anchoroutputs約218MB，C32約436MB；P_l若FP32存各layer，r64約13.6MB、rH約436MB。若keys只存P c，r64/C16約6.8MB；若需fullc做完整distanceguard，另約218MB。這些均占用host8GB並給baseline同容量cache／lowrankmetadata，不能稱免費。NAND mapbank容量為L E C H r bytes；例如r64/C16約3.49GB，rH/C32約223GB，後者加原Deepweights可能仍fit~333GBLSB，但全模型metadata／spare／capacity仍須真ledger。

Nearestanchor keycost為selectedK×C×r的distanceops，P_lx共Hr MAC／layer；全norm或fullc guard再加KCH，descriptor／PCIe每e/c的δz也需量。Keyguard必在mapread前決定，若先算map或讀original才決fallback則doublework計入。C與r越大quality／coverage可能改善，但read／metadata／offlinecost同步上升，不能只挑最佳quality不算cost。

### 導數初始化不用形成 dense Jacobian

固定P後，預算每expert `B_G=G P^T`、`B_U=U P^T`（I×r）。每anchor計g=Gc／u=Uc／F(c)，再

`A=D[diag(u SiLU'(g)) B_G+diag(SiLU(g)) B_U]`，約2HIr MAC／anchor。

這避免先生成H×H的J；但當C／r大仍是龐大離線工作，不能說零training便成本零。Runtime若直接算J(c)δx卻不物化A，仍要讀G/U/D原weights，完全失去方案所省read；只有offlineprecompute才是此機制。

誤差可清楚拆為`J(c)(I−P^TP)(x−c)`的projectiontail，加上真正Taylorremainder。故r掃至H不是為救低rank任意加參數，而是隔離『投影不足』與『局部curvature／anchorcoverage不足』兩原因；同樣不能把fullrankJac失敗當所有OLS／nonlinearstudent不可能。

### 新primary與contribution boundary

- [TARDIS2501.10054v1](https://arxiv.org/html/2501.10054v1) §5.1已處理多range的組合矩陣爆炸，採singlefoldedmatrix＋outlierpredictor／fallback；§9明說GatedFFN quadratic令LLaMA2 folding膨脹254×。本候選可能避開的正是perneuronCartesianproduct：只在實際jointinputregions保留C個H×r maps，近似整個SwiGLU，不做全neuronrange枚舉。這是可比較機制，不是聲稱首次folding／fallback。
- [How Linear Is a Transformer FF Block2606.19379v1](https://arxiv.org/html/2606.19379v1) 提供exactclosed-formOLS控制，指出under-convergedtrainedlinearbaseline會誤造新模型優勢，且heldoutR²與真正PPL可嚴重分離。其TinyLlama兩blockglobalOLSR²只有約.04–.07，是強負先例但非本MoE／localmaps的否定。Stage0須量finiteoutputerror且後續真正free-running品質，不能只高R²便宣稱成功。
- [LLMs are Locally Linear Mappings2505.24293v1](https://arxiv.org/html/2505.24293v1) 為detachedJacobian／sameinputequivalence，不是trueJacobian的跨inputanchorreuse保證；不可借其lowdim結論填補本候選的coveringnumber／curvature證據。

## 另一方向的 constructive screen（未採納）：packed VQ index＋LUT

[CodeGEMM2512.17970](https://arxiv.org/html/2512.17970v1) 已將input/centroiddot存Psumbook、indices直接lookupaccum；[EVA2605.24144](https://arxiv.org/html/2605.24144v1) 已專門硬體化outputcodebook與conflictfreebanking。不能把AiF PE換LUTadder本身當新方法。

AiF真正constraint是Psumbookresidentstate。d8／C256／m2時fullinputH2048的FP16 Psumbook需2×256×256×2=256KiB，超出兩reclaimedPB。Columnshardinput至H/16=128可縮至16KiB，卻產fulloutputpartialsum、需crosschipreduce；rowshard保持小output卻需頻繁換LUT／host傳輸。這是可畫出的dataflow矛盾，可能研究chip-localinput／indexlayout聯合設計，但目前最直接收益仍為既有VQ低bits與既有EVA計算，尚未證同容量EVA式baseline不能解。

Sharedcodebook acrossKexpert可讓GU的Psumbookbuild只做一次，但Down每expertinput不同，不能全模型一套table免費重用。新primary KBVQ-MoE2602.11184已有跨expertsharedlowrank＋expert-specificVQ；另Beyond4bits2610.28537聲稱expert-sharedcodebook（待primary核）。這些不能自動殺co-design，但表示sharedbook本身不夠；目前主力集中localmap的真operator消除。

# 正式候選：2026-11-02 21:26:00 UTC

Root通知Gemini精確ArgonHigh於21:24:34 join localAffinefocusedblock，保守正式起點21:26:00、最早21:56:00 UTC才能考慮實驗。21:18:37開始的screen不抵扣；時間到不自動准入，仍需Root／黑臉與protocolfreeze。沒有任何新候選code／probe授權。

### Offline A 計算勘誤（採 Root 指定 fused family）

前述2HIr／anchor是把兩個D乘法分開實作的示例，不應當必然成本。正確fusedfamily為：每expert一次計B_G／B_U共約2HIr；每anchor先合成`M=diag(uφ')B_G+diag(φ)B_U`，需O(Ir)，再一次`A=D M`，約HIr。另計GUc、F(c)、dequant／精度／量化與anchor選擇。後續成本一律以此明確family，不沿用2D-GEMM過度帳。

## Formal 21:26後第一輪深入：native cohort、曲率與等價類

### 小rank不能固定用g16弱baseline

H2048、W8、16KiBpage且完整rows時，r64全map只8pages，可g2每chip4pages／一wave；r128有16pages可g4；r256有32pages可g8；r512有64pages需g16。g16把r64每chip8KiB padding成16pages、每expert依次1wave，只是其中一配置，不是最強吞吐。對K6，g2有8組可同時處理不同experts，g4只有4組，g16只有1組；實際topKgroupcollisions、命令、scales、inputposts／channel都要排。候選與globalhigher-rank controls都應得到既有placement／replica／Hall能力，不把那個老solver當新貢獻。

### Curvature與anchor選擇的必要診斷

若r維states近isotropic，Ccentroids帶來的量化distortion改善粗量級只有2logC/r；C32在r64約10.8%、r512約1.35%，並不自動形成smallTaylorneighborhood。這只是isotropic反例，非模型觀測。Kmeanscentroid可能離RMSnormalizedinputmanifold，SwiGLU在c≈0會J≈0；purequadratic例F(x)=xᵀQx的一階剩餘恰δᵀQδ，升至rH仍不會消除。

故實驗應分開actualTRAINmedoid與centroid、Jacobian初始化與regionOLS/intercept。後者能吸收Jensenmeanbias，不能拿Tayloranchor選差／bias未校正造成的失敗當所有localgeometry死亡。這也不是靠fullKD，全部是有明確closedform／operator的可辨識控制。

### Same hostmemory＋CPU execution 是真控制

r64/C16 Deep全W8map約3.49GB，另FP32anchoroutputs／keys／P約.24GB。在host8GB是否容得下nonFFN／KV須真ledger；若容得下，hostresidentmap的CPU小GEMV可能比28usNANDwavefloor更快，不能為了AiF故事硬讓smallmaps走flash。Baseline同DRAM可放original/shared／W4cache，但cachedrawFFN還要CPU算，cachehit不等零delay。候選／control都需可比hotmapcache／CPUfallback能力，最終主張可能是operatorcompression＋hybridexecution，而非AiF獨有primitive。

### 同能力 global nonlinear control 的邊界

Local `A_c z+b_c` 的各regionoutputsubspace可不同，單一global `A φ(z)+b` 受一個rankr輸出子空間限制；globalhigher-rank可花同storage增加rank，並用實際cohortwaves比較。不過若其φ允許block-sparseonehotlatent `[0,…,z,…,0]` 且A的對應columnblock可選讀，這就與localmapbank完全等價。不能把這個equivalentlatentconditional-expertcontrol排除後宣稱新表示定理；承認方法族，研究的是對既有checkpoint變換後是否在quality／hostcapacity／activeflashwork上有真Pareto點。

### 既有數值參考／品質（唯讀舊資料）

已重讀舊docs的完整W8品質scope：Deep固定8×128 WikiTextteacherloss上原Q8GGUF PPL16.4102→canonicalW8A8 16.7289（+1.9423%），G4相同類小樣本近+.094%。這是既有結果，不是本輪probe，也不表示canonical等價原nativecheckpoint。新approx分支須先相對明確teacher，再對原GGUF／nativequalitycontrol披露representation自身差異，不把原W8數值誤差全歸本方法。

## 21:36:10 UTC checkpoint：控制組與可部署資訊界線

### Shared projection 的 informationfloor 有條件

先前『kerPvariation不能靠更多centroids恢復』只在相同Px且相同region／guardstate時成立。若key可看fullx，它可用log2C離散bits編碼部分omittedinput資訊，c不同則b_c／A_c可不同，不能無條件否定。故預列z-onlynearest與full-xnearest／小C×Hclassifier控制；C32/H2048 gate約64Kcoeff／layer，未必需排除。Grant同signal給nonlinear／globalcontrols，不能靠限制baselineinputs造優勢。

Full-xnearest每selectedexpert讀C×Hanchors；C32/K6/H2048/FP32/26層約40.9MB metadata／token，P_r64另13.6MB/token，已足以形成數ms的memoryfloor（實測前不給確定時間）。若layer-globalcentroids全E共享，key只算一次可省K倍，但conditionalcoverage可能較差，與expert-conditionedcentroids是公平成本／品質對照。資料稀疏也不同，不能混稱sameC。

### Conditional slopes 必須與 conditional bias 分離

最直接控制是完全相同C、key與anchoroutputs／regionintercepts，但只用一個globalA_e斜率；r0只查b_c亦必列。這能分辨收益來自查表meanoutput、global線性補償，還是實際region-dependent slopes。rH／P=I可免hostprojection；globalhigher-rank按真pages／cohortwaves比較，不只同r。

### 免 full-KD 的 nonlinear control

Host存B_G=G P^T、B_U=U P^T，每region只存offset `G(I−P^TP)c`、`U(I−P^TP)c`。Runtime由同z算ghat／uhat，做真SiLU×Up，flash只讀原Down（單phase）。它保留P方向全部非線性，C只買2I offsets，對localJac是很強的curve控制。Deep r512 hostGU W8約2.40GB，r64約.30GB；實際inputquantization／CPU MAC／DRAM讀取／P／bias需算，不能以host放得下便當零成本。這可視MoLAEinput-only加contextoffset，不主張首次。

MoLAE的outputlift還可group共享：Eq4 Wdown_e=Bdown_group A_down_e，若latentbody在hostbudget內，先把livealpha加權latentoutputs在group內相加，再只讀每activegroup一次Bdown。不能將baseline故意每expert重讀相同B。Group整體outputsubspace限制與localcontext切換形成可比的expressivity／readtrade，但quality、r²latentbody和hostcapacity仍未知。

### Whole-MoE-layer extension 僅pool，沒有沿用formalclock

可對含livealpha／shared的整layer FFN直接存localmap，局部同route時`J_total=Σalpha_e J_e + ΣF_e (grad alpha_e)^T`，每層只一map且N/C樣本不受K/E稀釋；但topK邊界跳變與routercurvature是新增hardpart。已向Root提出作strongsamefamilycontrol／未來pool，不能把當前21:26clock冒用另一正式方法，也不把『只需fullKD』當支持理由。

### 新來源的獨立核讀狀態

Root／Terra已核Flash-Jacobian匿名ACL稿（clusteredcontext／separateperregionA B／conditionalridgeALS）；本agent的webattachment目前被browserchallenge，尚無法獨立讀原PDF，不冒充已核。其直接局部map先例收窄novelty，但沒有logicalequivalence證明sharedP／NANDhostbudget全family已被品質／cost支配，繼續研究。

## 21:46 UTC checkpoint：可識別的 minimal Stage0 主 family

Root將主實作收斂為**layer-shared balanced input tree + free-intercept local ridge**。全E共享leaf/meanX，故`δz=P(x−meanX_c)`與它的A8量化在同一layer可全K共用；前述expert-specific keys需要K次inputposts的帳不再適用於這個primary。若同層混用original fallback與map，仍需map epoch/original epoch或明列input重載。

主點r128/C32，C1/4/16作少量ablation。FIT-only global nested PCA，tree每node沿該node第一PC作median split。Root的120FIT prompts ×(128prefill+8decode)上限16320inputs，使均勻leaf約510rows，而不是routed-only每expert的幾十rows。8pilot ops（Deep layers1/13、OLMoE0/8，各FIT預選most-frequent與median-frequency expert）對全部FITx求真canonicalW8A8 teacher outputs，評分仍主要用SCORE自然routing命中域。這消除nominal欠定，但不保證泛化／有效獨立樣本，亦不等同expert-conditioned training distribution。

### Tree ties 是部署契約，不只是實作細節

FIT依sampleID破median ties可以造出表面平衡、卻不能由未來input部署重現的leaf。應使用真正input-dependent tie-break（例如預定next-PC／full-x lexicographic secondary key），完全相同x不能硬拆。用最後部署tree重走FIT，量真Nleaf、distinct prompts與自然route hitcount；若重複x令leaf失衡，2r保證只是條件性而非數學已保證。VALID/SCORE不重平衡。

### Actual A8 design + free intercept

對共享leaf，先產生真實部署輸入`q_i=Q_A8(P(x_i−meanX_c))`與`v_i=s_i q_i`。Ridge用decoded-A8 design v，而不是FP32δz擬合後宣稱相同。令每leaf weighted vbar/ybar，部署為`b_e,c + map_W8(e,c, q_i,s_i)`。因A8 rounding／route weights，vbar未必為0；不能把free-intercept主方法誤寫成固定ybar+A v。每λ以FIT閉式解A，再row-W8 quantize成Aq，然後`b=ybar−Aq vbar`重新置中，VALID以真正integer→F32scale+b路徑選λ。另報FP32-map oracle區分表示不足與量化損失；它不當部署結果。

SameC/tree共享斜率控制必須最優pooled **WITHIN-leaf demeaned** fit：把各leaf的v/y去均值後合併解同一A_e，再各leaf自由b_e,c。Globalfit後硬塞intercepts不是這個強控制。另r0 conditional output mean；shared-slope ranks128/512/H（H時P=I免projection）給相同key/bias與最強物理group/cache配置。

Normal PCA與whitened PCA兩種主輸入格式由8ops pooled VALID一次決定，全candidate/control共享該選擇，避免per-op/per-leaf超額挑選。Whitening的對角縮放可併P、反縮放進A，不是新貢獻，目的是避免單A8amax抹掉小variance PCs。P非正交後，projection-tail guard須保留原orthonormal V計算，不可直接用whitened norm作Pythagoras。

### Weighting / convergence / negative scope

最簡主fit為all-FIT無權重，on-route VALID選λ；secondary可預列低方差route weights：onroute1、offroute1/4，candidate與controls相同。若Root改此為primary，negative就限定此floorfamily。逐leaf報`N_eff=(Σw)^2/Σw²`，不把510 correlated tokenrows視為510獨立樣本。Ridge penalty按Σw與trace(cov)/r正規化，固定λ grid `{0,1e-4,1e-3,1e-2,.1,1,10,∞}`，每op/C一個λ，不perleaf調。

Learning curve可用2/3 vs full FIT的coefficient data、共用凍結unsupervised P/tree，避免half≈255恰低於2(r+1)。須披露P/tree見過全部FIT的unsupervised資料。資料曲線仍改善、condition/effective-dof未穩、route SCORE命中太少，都是**inconclusive，留同輪繼續**，不能當第2次valid experiment fail。

120/40/40 prompt-disjoint FIT/VALID/SCORE；SCORE naturally-routed prefill/decode分開，按prompt paired-bootstrap。40SCORE×8decode只有320layer tokens，單expert約幾十hit，不能靠token-iid CI製造強陰性。SourceGGUF x上的逐expertcanonical teacher只是offpolicy operator pilot，不是整個canonical model free-running quality。8ops local-positive僅准同輪進全層真品質／host/native／W4/低K/MoLAE/SPICE等強控制，不能算方法成功。

Concrete bounded negative只能裁此shared-tree／all-x／部署量化／closed-form-fit family及其明定loss/cost評估，不能宣稱所有local maps、expert-conditioned partitions或KD被否定。即使local output MSE被控制支配，跨expert signed errors與autoregressive傳播仍需另證，不能直接宣稱whole-model quality支配定理。

## 21:50 UTC checkpoint：draft-001 closure 與 Gemini cross-review

已唯讀完整 `results/research-2026-11-03-no-cr/round02-local-affine/stage0_protocol.draft.json`。主family現在明確採**shrink-to-pooled**而非zero-ridge：先同leaf optimal pooled slope A_shared，再解每leaf correction ΔA，penalty λ||ΔA||²，λ∞恰退shared-slope；前文zero-ridge λ∞=r0的描述僅適用另一objective，正式primary以本節修訂。R0獨立保留。建議明確prior自身regularization/pinv規則；trace=0時沒有資料可識別localcorrection，取ΔA=0，λ0用預定relative eigenvalue cutoff的minimum-norm correction，不能結果後任意jitter／synthetic資料。

Bias calibration以實際W8/A8 integer→F32 FIT predictions之weighted mean(target−prediction)為最忠實規則；解析Aq*vbar只差rounding但不是同一路徑。所有family同法。Canonical舊helper zero-amax用scale1與全zero q，map/input都沿用；teacher-energy0的relativeerror未定義，改報absolute並明示，不能製造infinite error。

Fullrank `P=I`可免projection，卻在W8/A8下不等價full-PCA coordinates。故應預列為獨立R2配置，VALID決策或全報，不能看SCORE後選basis。Normal/whiten一次全8ops的選擇也需明定pooled VALID objective與tie-break。若SCORE有很多candidate/control點，不能事後挑bestdominator再用普通95%CI聲稱嚴格負判；可VALID預選每candidate的comparison，或對全預註冊paired comparisons作simultaneous/Bonferroni CI。全SCOREfrontier作描述可以，證據強度需處理postselection。

### Gemini Block3 修正建議（已送 Root／黑臉）

已讀 `docs/research_2026_11_03_round02_affine_gemini.md` Block3。支持sharedtree/decoded-A8/shrink-to-shared family，但指出三個過稱：

1. 八個operator局部MSE被controls支配，**不推出**全模型『絕無可能翻盤』；signed cross-expert errors、未測深度與autoregressive trajectory都可反轉。只可給frozen operatorloss上的bounded scientific negative。
2. Router在kerP有分量，**不必然**令conditionalmean非0；分布／其他topK inequalities可能抵銷，應改may。FITLEAF inputmean0不等於route-weighted均值0。
3. Depth5的node-wise PCA tree不是只沿global前5PC切割；每node方向可不同，總31internalnodes。每條path5方向的確仍有限，但不得誤寫全球共同5個方向。

Gamma近0的routeweights會重新造成effective-N低，不能用因此欠擬合當公平negative；draft採1/.25 boundedfloor可避免極端變化但仍須N_eff/curve診斷。

### EBC author code 已獨立唯讀核驗；全文仍未讀

讀本地author notebook extraction `prior-sources/ebc-author-code/source_cells.txt` 34010–34135、35325–35365。`build_Ws`對每expert實際teacher-output擬合單一globalH×H ridge，可按routerP加權；kmeans對象是Ws_norm之random-proj特徵，即expert matrixclusters，不是runtime inputregions。這比索引摘要更可靠，但來源仍是public development implementation，**不冒充論文正文已核**。

該code在NaNcalibration會自動退synthetic Gaussian／uniformrouter；本campaign不能繼承此fallback。真capture／teacher非finite必須invalid修復，不以synthetic填數形成科學負結果。

### 容量與投遞閉合

相同lowered maps必physicaldedup，包括λ∞C份sharedslope、任意bit-identical localmap；r0／zero-map直接hostbias查表，不能硬發NANDread。OLMoE主bank恰8GiB本體，aux/nonFFN/KV另占，8e9與8GiB兩種host解釋都審，但仍允許besthotmapcache而非全有全無。

Sharedtree的32leaf meanX在H2048/L26/FP32約6.82MB，另31internal PCA normals約6.60MB，合計13.4MB加thresholds。每token只走5normals，不讀全31。r128 P全26層FP32約27.3MB，bias/rowscales active流量另計；不能稱sharedkeys完全免費。

Root建議teacher批次integerdots以純FP64 DGEMM加速是可證安全的：若任意partial絕對總和Σ|qw*qx|<2^53，所有integerproducts與加法均exact；需alpha1/beta0，無mixedprecision，先對I64最壞range與randomsigned驗證，再保留canonicalF32cast/scale/wholeI SiLU。此離線batch吞吐完全不代表B1部署host latency。

## 21:54 UTC checkpoint：統計 closure 與一個可構造的正向強控制

Draft-002已納入兩種全layer預註冊format（normal與balanced），不靠SCORE挑basis；pooled eta與local lambda分離；finite-local VALID winner也保留以免lambda∞造成空洞negative；全family採真FIT部署prediction重置bias。Tree／zero-amax／zero-trace／mapdedup／guardVALIDquantile與negative scope均已修。我的判斷是此bounded Stage0有科學辨識性，可以在Root／黑臉凍結後執行，尚無數據支持成功或失敗。

尚提醒一個統計解析度問題：若完整formal comparisons達約1024，10000bootstrap + Bonferroni的單尾alpha/(2M)約2.4e-5，低於1/10000解析度，不能插值極端tail形成強negative。可縮小事前正式主比較集，其餘descriptive；或凍結prompt-bootstrap max-|t| simultaneous bands，用10000次重抽的全比較向量最大標準化偏差之95%分位。後者仍是40prompt bootstrap近似，不是finite-sample certificate。若維持Bonferroni則resamples不足就是inconclusive，不能悄悄降低familywise scope。

### Direct common-output factor control，僅正向擴展，不是新候選或實驗

Local slope勝sharedslope只證需要nonlinearconditional斜率，未必證需要flashbank中的不同outputsubspaces。若全部A_e,c其實近共享一個q維outputspace，可離線把horizontal concat `[A_e,1 ... A_e,C]`（或按FIT latentcovariance加權的blocks）做共同output SVD，得到`A_e,c≈U_e,q B_e,c`。Host存每leaf小q×r矩陣B，Flash每expert僅一H×q U。這在q=r時保留相同一次大mapread而大幅減retainedflashbytes，正是MoLAE-like／hostpiecewise latentcontrol，不必靠fullKD相信會贏。

Deep C32/r128/q128、全1664experts時B bank W8約0.87GB，FP32約3.49GB；U W8約0.44GB，另bias／scales／P／keys／nonFFN／KV。Host `Bv` 後再A8傳U會新增量化、CPU與metadata bandwidth，因此**實數factorization不是bitexact或全latency支配**。若q需512才保quality，U的activepages／cohortwaves同步增加。這個直接構造能辨識究竟需要flash容量換輸出basis切換，還是小hostlatentpiecewise nonlinearity已足夠。Stage0 positive後必真正測，不在沒有資料時預先判死。

### Offline query scale 與後續 on-policy 限制

8ops×16320×H2048 FP32 target約1.07GB；Deep全1664expert-ops全查相同FIT約222GB targets、約230T integerMAC量級。可逐expert／layer流式累積sufficientstatistics，無需常駐所有targets，但離線cost不是zero。所有baseline得同queries／資料預算。SourceGGUF pilot的x與candidatefree-running x可能shift；若正向需on-policy重收／再fit，仍同Round2，不以offpolicylocalfit冒充完成。

## Formal handoff：2026-11-02 21:56 UTC

正式起點21:26:00，持續實質source／math／prior／protocol互審至21:56之後，滿至少1800秒；沒有以舊screen時間、代理時間相加或sleep抵扣。最後獨立receipt以工具UTC回覆為準。本agent未寫／跑候選程式、未讀取新model／activation／route probe，round仍1個valid fail／9剩餘。Root後續還須Gemini Block4／黑臉／protocolfreeze，不因本handoff自動授權實驗。

**最終概念與methodprotocol意見：建議通過 bounded Stage0 的思考門檻。** 主family是既有conditional affine/ridge工具在shared inputtree／共同A8廣播／有限flash與host容量下的具體compiler，不宣稱新Taylor／cluster／fallback原語。它確實能取消被替代expert的GU→Down感測dependency；原shared／fallback branch仍可能保留整layer barrier。r128/C32容量與activework有真可構造區間，未出現像舊taggedpairbank的same-bit／same-pages無條件equivalence kill。

Draft-003已提供必要識別：同leaffreebias、optimalwithin-leafpooled slope、高rank成本、真A8design＋W8lowering、足夠但非保證的allFITqueries、routeweightfloor、學習曲線、prompt不確定性、VALID凍guard、mapdedup、hostcache與兩format。Normal與balanced之isotropic ridge確是不同regularizer；兩者預註冊並實評合法，不再為追coordinateinvariance擴scope。

局部正向只能同輪續全模型／非線性／W4／cache控制，不能稱outputsubspace旋轉或投稿成功。Common-output SVD + host piecewise latentcontrol已交Root列mandatory Stage1；它需付CPU／metadata／再A8量化，不先假稱支配。局部陰性也只裁凍結family／sampledoperatorloss，不能證所有非線性KD或所有guardedoperators不可能。資料／CI／數值／實作不充分保持同輪inconclusive。

交接後結束本turn釋出指定Sol-xhigh實驗執行者slot；若未來本輪真的fail再啟下一輪fresh正式思考時鐘，不能重用本次時間。
