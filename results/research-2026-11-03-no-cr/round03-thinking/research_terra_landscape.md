# Round 3：ordinary-read MoE 機制廣域來源篩查

2026-11-03 UTC；研究角色 Terra/xhigh。依 `round03_research_charter.json`，這是 source/theory landscape，未採納候選、未開始正式 30 分鐘時計、未執行候選、模擬或 RTL，未讀任何模型權重、activation、target、FIT、品質題文或 TEST。來源查核不限 paper；有界搜尋不是無 prior 或無 artifact 的證明。Round 1 的 gate-first zero 及 Round 2 的 conditional affine leaf 是已判定的**有界失敗**，不能被偷換成其他方法的全域否決。

評分標準：比較不靠 charge-recycling 的普通 NAND 下真正減少的**實體 sensing waves、不可重疊依賴或必要資料／計算**，且每個控制都可使用相同早期資訊、合法格式/頁布局、主機 cache 和硬體 budget。對 AI F 論文未揭露的 short-input、ECC→PE rate、latch/FIFO/多 context 和跨階段重疊，皆應標假設而非當已實作。以下來源連結是此次確實讀到的 paper 指定段、作者 code/document 或 patent，不是搜尋摘要充作全文。

## 1. Exact temporal integer-delta MVM：可能刪讀，但實體粒度最可能判死

**明確機制。** 對同一 expert 的同一固定 Q8 權重 `W`，保存該 expert 上次被選中時的量化輸入 `q_old` 和每一 row 精確 integer dot `a_old=W q_old`；本次以 `Δq=q_new−q_old` 和 column-tiled 權重更新 `a_new=a_old+WΔq`。若某個**完整實體可跳讀單位**所涵蓋的 Δq 均為零，其 weight page 不必讀。對 gate/up 可用同一 token input，Down 則依 expert 自己的 SwiGLU input 和自己的歷史另做；無論模型是否有 shared expert，routed path 的構造都適用。W8A8 動態 input scale 改變時，要以保存的 **raw integer accumulator** 和當前 input scale/原 weight row scale 重新產生輸出，不能重用已乘舊 scale 的 float。`Δq` 可達約 ±254，需 9-bit signed 或分解為多個 INT8 products；I32 中間項、加法溢位、路徑尺度、ties、SiLU 和既定累加順序均須逐一證明，並不能只憑整數代數宣稱跨現有 floating implementation bit-exact。

**先例與界線。** [DeltaCNN CVPR 2022 §3](https://openaccess.thecvf.com/content/CVPR2022/papers/Parger_DeltaCNN_End-to-End_CNN_Inference_of_Sparse_Frame_Differences_in_Videos_CVPR_2022_paper.pdf) 已傳遞、累積跨時間差，並利用 threshold 丟棄小更新；[Chipmunk 作者研究頁](https://sandyresearch.github.io/chipmunk-part-I/) 已在 diffusion transformer 對跨步 activation 用 sparse deltas／快取輸出。兩者不證明自迴歸 token 間 Q8 `Δq==0` 多，也不等於在 AiF NAND 少讀 pages。**差分 MVM 和 activation cache 不是新發明。**

**頁粒度反例（算術，非 trace）。** 若一個邏輯 tile 包含 2048 output rows × 8 input columns × 1 byte = 16 KiB，八個 Δq 只要**任一非零**就要讀整頁。1 column 獨佔 16 KiB，則本來僅 2 KiB 的權重實體存放需要 8× 容量／填充；不得把它稱為免費細粒。這個 2048-row 例子並非自動對應原生 chip 切分：若 g=16、每 chip 只有 128 rows，填滿一個 16 KiB page 可涵蓋 128 columns；四 plane 同步 wave 需要按實際地址與同步規則合併，不可把 element zero fraction 直接當成 wave 跳讀 fraction。變更 row/column tile、輸入廣播或 partial sum continuation 是**對兩方都有成本與權利**的硬體／布局變更。[SOML ASPLOS 2019](https://www.cse.psu.edu/hpcl/docs/2019_ASPLOS_chun.pdf) 是 subpage NAND 架構直接先例，涉及額外選擇／解碼結構；不能宣稱普通 16-KiB 原生頁改軟體就能感測任意 2-KiB column，亦不能憑其不同設計面積預估本案 PPA。

**強普通基準與可否證點。** 原 W8 的 ordinary-read CPU 取得**同樣 column-tiled W、早期 Δq、每 expert 相同歷史/cache、及可用的 subpage 能力**，也可以執行相同 exact delta 更新。應同時比較 Q4 原 expert 是否能在同品質時放入主機，並量化 **last-hit expert** 而不是 last-token 的距離、完整對齊 pages/waves、歷史更新與首次/eviction 的 cold fallback、input/accumulator/cache/transpose 稅、host SIMD 及控制／輸出成本。若 exact Δq 的 physical waves 幾乎不減、history 或多 chip broadcast 成本掩蓋節省，或同 budget host baseline 已足夠，則拒絕；不能用小 delta 近似或獨立性模型冒充零的實測證據。

## 2. 全域 A8 amax 的 exact 前瞻證書：改依賴、未必改 sensing

**明確機制。** Gate/up 對 neuron tiles 串流，某已完成的實數中間項 `z_i=SiLU(g_i)u_i` 確定後，設已觀察 `L≤M=max_j|z_j|`，以保守上界對全部尚未完成 rows 建立 `M≤U`。`q_i(M)=round(127 z_i/M)` 對固定有限 `z_i` 在合法正區間單調；若 `q_i(L)=q_i(U)` 且 tie/浮點邊界按保守規則處理，則其 Q8 code 不論最終 M 為何都已確定，可提早送相應 Down page，按原 integer dot 條件逐項 accum。L=0、U=∞、非有限值、underflow、F32 四捨五入和 accumulator 溢位須明確 fallback。上界可能來自原 weights 的保守 norms 與剩餘 input，**不是**假設能看到真正 M；額外 metadata 與 CPU 計算不得免費。

**近 prior。** [PowerInfer-2 §4.3–4.4](https://arxiv.org/html/2406.06282v2) 早已用 exact gate-zero / predictor 控制後續 neuron data 載入並作 cluster pipeline；[SLIM §5.4–6.1](https://arxiv.org/html/2507.09201v1) 已 fused neuron-wise gate/up/down 和 partial sums；[LeanStream §3.1–3.2](https://arxiv.org/html/2609.03079v1) 已在普通 SSD→GPU 上執行部分權重到達即計算、SwiGLU neuron permutation 和 partial residual。以上不證 whole-vector A8 q 的這個 interval release，也沒有證明新方法能戰勝它們。LeanStream 的 real-sum permutation 說法不是自動的 IEEE bit-exact 保證；本方向若需 exact，必須給固定整數/浮點次序參考。

**強控制／否證。** 完整 dense GU→全量 A8→Down，及普通 CPU 使用相同 interleaved neuron packets、同 dynamic bounds、相同 prefetch，以及不作證書的 earliest legal scheduling。Down 即使 q_i 提早確定，也只能在與剩餘 GU **不衝突的 plane/chip/port** 上贏得 critical-path slack；同一物理 NAND 若串行 sensing，常只重排相同工作。應明確算 four-plane 佔用、分組/multiplex、input broadcast、activation SRAM、完整 A8 scale、partial accum、跨指令輸出與 control；若 U 太鬆導致絕大多數 q 直到最後才定，或相同總 sensing critical path 無可隱藏重疊，thought-reject。這是**提早 Down readiness**，不是重跑 Round 1 的 gate-zero 跳 Up，亦不主張必有波數節省。

## 3. 改量化尺度以解除 amax barrier：受控近似而非 exact 原模型

**機制與界限。** 將 SwiGLU intermediate 改為 block-wise dynamic A8，使每 block 的 `M_b` 一完成即量化並啟動 Down 的相應 tile；要留意不同 block scales 使整個 Down 不再只有一個 shared I32 scale，需逐 block rescale/累積或多個 accum。若以 global A8 作原參考，對無 clipping、有限非零 M、理想 real round-to-nearest，兩種量化的逐元素誤差可用三角不等式保守界：`|Q_M(z_i)−Q_Mb(z_i)|≤(M+M_b)/254`；乘上原 Down matrix 可進一步由每 row `Σ_i |W_ji|·bound_i` 上界 output 誤差。該界通常可能非常鬆，不是 logit/top-token、全序列或精確 rounding 保證；要求另計 weight quantization、clamp、ties、F32 累積和 fallback。

**先例與控制。** [NVIDIA TensorRT 官方 dynamic quantization](https://docs.nvidia.com/deeplearning/tensorrt/11.4.1/inference-library/quantized-types-workflows.html) 已按 block 決定 scale；[FlashInfer 官方 fused gated-activation quantization](https://docs.flashinfer.ai/api/quantization.html) 和 [FlashInfer block-scale MoE API](https://docs.flashinfer.ai/generated/flashinfer.fused_moe.trtllm_fp8_block_scale_moe.html) 已實現 SwiGLU、block scales、MoE 的近鄰組合。這些 GPU 格式/硬體不等於 AiF W8A8。必須讓 ordinary-read CPU/GPU 亦使用同一 block quant、同誤差與相同 early-read/weight tiling，並與原 Q4、靜態 A8、existing group quant 作 matched-quality 比較。若離散 page wave 數不變、上下兩階段搶同一四 planes、FP accumulation overhead 抵銷收益，或 quality 評估失敗，則放棄。僅把已知 block quant 移入 flash 非足夠貢獻。

## 4. Lossless W8 稀疏/字典或 entropy stream + die 內解碼：改必要 bytes

**機制。** 把每 expert 的 I8 weights 精確編碼為可重啟／可索引較長 stream，普通 read 載入壓縮 physical pages、die 內解碼器流式供既有 INT8 PE，藉少於原始 `ceil(bytes/plane-wave)` 的**整 waves**降低 sensing。需要 output row 邊界、restart metadata、dictionary/decoder 速率、PE starvation、跨四 plane 壓實與尾段 padding、對齊、不同 expert 的空間重複與 ECC 計帳；格式不得要求每次解碼無關的整個巨型 stream。若 entropy 沒足夠跨 wave 門檻就沒有省時。這裡不依賴 shared expert、也無需改最終數值，但不保證任意 W8 可壓縮。

**最強先例。** [HardCompress ISQED 2021 §III](https://www.isqed.org/English/Archives/2021/ISQED_2021_Papers/pdf/4A-3-208.pdf) 已是 8-bit NN weights 的字典壓縮＋硬體解碼送 MAC；[ZipServ §4](https://arxiv.org/html/2603.17435v1) 已將 lossless decoding 和 GEMM 融合；[ZipMoE ICML 2026](https://arxiv.org/html/2601.21198v1) 是 MoE＋lossless compression＋CPU reconstruction／cache scheduling，其主要 BF16 exponent 分布**不代表 W8 的可壓縮性**。額外 primary example：[UC Irvine 學位論文章 4](https://escholarship.org/content/qt2tk163fh/qt2tk163fh_noSplash_12e813f349bedc441e1abef2be778eef.pdf) 的 ZipNN 把 group-varint／delta/RLE、FPGA near-storage compute 與真實或模擬 flash bandwidth 結合，但工作是近鄰搜尋的整數資料、不是 LLM weights；不能拿其 FPGA 壓縮比／speed 作 AiF 數據。通用 inline decoder 並非新意。

**最強基準／否證。** 相同精確壓縮格式、restart、logical mappings、host cache 給普通 flash→CPU 的高度最佳化解壓/GEMV；另以 matched-quality Q4/其他低 bit 作更強功能 control。比較壓縮 physical wave、芯片解析/PE 速率及 CPU、外部流量、佈局空間和未壓縮 fallback。若壓縮節省不足以跨 wave 門檻或 decoder stalls、host 解碼更便宜，則不開候選；舊少量樣本的零波收益是**有限條件診斷**，不是 W8 的普遍不可能性。

## 5. 共用非線性 feature 的 one-read bank：保留非線性，但重 prior/品質風險

**機制。** 以所有 routed experts 共用的 `φ(x)=SiLU(Px)⊙(Qx)` 或另一小型非線性 feature，預先離線擬合 expert-specific `B_e φ(x)+b_e`，執行時 layer/token 只算一次 shared feature、每個選中 expert 讀一次 B。與已失敗 Round 2 的 per-region affine slope 不同，這個表達有**全域非線性**且沒有要求新增 shared expert；但它仍是 feature compression，不能從更強表示能力推論品質。B 的寬度、W8 scales、shared P/Q（主機或 flash）、fitting、route-weight combine、熱 cache/fallback 和全模型品質均須付費。

**先例與強控制。** [NVIDIA LatentMoE 官方設計](https://research.nvidia.com/labs/nemotron/LatentMoE/) 已把 shared down/up projections 放在 routed experts 周圍並保留 latent expert 非線性，也明說過小 feature rank 有品質地板；這是不同的從訓練開始架構，不是此 post-hoc Bφ 實現證據，但反駁「共享 latent＋保留非線性」首創。另有本 repo 已核 [Flash-Jacobian](https://openreview.net/attachment?id=au5FGRgwcZ&name=pdf)、[SPICE §4.2](https://arxiv.org/html/2608.21240v2)、Nimble/MoLAE。必比同 host budget 的 shared-output lift、全 host quantized original、同品質 W4、同 feature budget 的 CPU student／普通 load、原生低 rank or same-capacity original。若 common output span 已能以更少 hot-host bytes 重建、或小非線性 feature 的全模型品質與 R2 相似失敗，thought-reject；單憑 global feature 有非線性或不完全相同於 R2 不足採納。

## 6. 精度平面／importance 的動態讀取與計算：進一步強 prior

**機制。** 僅讀 base W4 planes 先算，若根據 live router/剩餘輸出貢獻等可用資訊決定不需 refinement 就不感測 residual pages；高精度才讀剩餘 bits。這可以減少 physical necessary reads，亦適用無 shared expert，但只在**讀 residual 之前**完成決策才省 sensing。一次已把兩 pages sense 出來後再丟低位，不是減少感測。

**直接先例。** [DPA-MoE](https://arxiv.org/abs/2610.21378) 已對同一 MoE checkpoint 用 prefix-decodable weights、offline residual 與 live router scores 分配每 expert bits；[DP-LLM 原文](https://openreview.net/pdf/6a8698040c86e898dd1e71e75391931b269ca019.pdf) 已利用 projection 做輸入相關相對誤差估計、搭配 Any-Precision 動態選層級精度；[BitWave](https://arxiv.org/abs/2507.12444) 為 bit-column serial 計算近鄰。它們沒有因而證實 AiF 普通 LSB page sensing 或同一 token experts' signed cancellation 有效，也不應僅因缺少 NAND die 版就將已知方法換名為新方法。

**控制／否證。** 同 quality、同存儲容量、同 adaptive bitplane 的 ordinary-read CPU，fixed W4 and W8 quality/latency Pareto，與 same-metadata 用直接 low-rank correction 的控制；若 full refinement 不減 8-bit sensing、base+residual 不降低完整 wave、依預測的 quality 無法證明，或複雜的 cross-expert sketch 打不過直接用同一 metadata 修正，拒絕。此前 round2 前的 sketch selector 已因具體強控制接近 thought-reject；不以換字重啟同一構造。

## 7. 普通 NAND 時序／可靠度自適應：真硬體問題但目前不可移植

**機制。** 保持每次普通讀取的完整 precharge/evaluation/discharge、不跨 WL 殘留電位，依已校準 page condition／ECC margin 合法縮短各 phase 或選用不同 ECC 策略；可能降低每次 sense 物理時間，與前述其他方向真正不同。若允許 residual bit errors，則另變成模型語義近似，不等於 exact。該方向需 first-read 的 state-/wear-/temperature-/retention-/data-pattern-dependent error map、實際 ECC、retry 和最終 quality。

**物理和專利先例。** [AR²/PR² ASPLOS 2021 §4–5](https://arxiv.org/html/2104.09611) 在 160 顆實體 48-layer TLC 研究 reduced timing、ECC margin、retry 與 cache read；其 72-errors/KiB ECC 和**final retry** Vref 範圍不能作 AiF LSB 10-errors/KiB first-read 的安全界。NAND [US20200057561A1](https://patents.google.com/patent/US20200057561A1/en) 及 [US20200117539A1](https://patents.google.com/patent/US20200117539A1/en) 已討論 AI weight significance 與不同 ECC，[US20230280976A1](https://patents.google.com/patent/US20230280976A1/en) 已按 partial sum 控制 memory read energy/timing，但其列舉的 memory 實施例不構成 AiF NAND 驗證。

**控制／否證。** 首先和**同裝置、同錯誤率與 ECC 目標下最好的安全縮時普通 read**比，而非只和 28 µs conservative 模型比；同一最短可靠 read 也應給 host ordinary-load。不可用獨立均勻隨機 bitflips 冒充具實測依賴的 NAND error map，不可把 AR² 的 25% 當 AiF 首讀折扣。缺少 target 物理 trace、只靠 offline 應用 importance 或 paper latency 表點，則不能做可信效能/可靠性主張，應暫緩而不是消耗一輪模型實驗。

## 優先順序與未解問題

最值得做**先驗紙上可否證分析**的是 1（exact temporal delta 對實體 pages/trace/state 的必要條件）和 2（exact whole-A8 interval release 能否帶來真的跨 stage critical-path overlap）。兩者可以在 no-shared/shared routed FFN 討論，但目前都沒有命中率、數值正確實作、可用 PE contract 或相對同成本 ordinary-load 的優勢證據；一旦具體 layout/依賴帳判零則應直接 veto、不耗 experimental round。3–7 是用來界定更大設計空間及最強控制的 source map，不建議僅憑此文承諾投稿或執行。任何一條若被正式採納，必須重新開始該候選實質思考時鐘，並按 charter 進行雙 thinker／critic 與實驗授權；本次廣搜時間不得計入。
