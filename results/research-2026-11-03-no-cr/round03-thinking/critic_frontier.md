# Round 3 黑臉宏觀邊界與 Gem block01 修正

2026-11-03 UTC；critic_sol。依 round03 charter，只有來源／摘要／代數思考；未讀模型、權重、X/Y、FIT、quality body、書籍或 TEST，未執行候選、分解、擬合、量測或 simulation。正式 candidate 尚未採納，method clock=0；目前實驗失敗2次，剩8次。本筆記不放行任何實驗。

## 五個具體反例與必要對照

1. **縮小權重不等於需要 flash compute。** 單 leaf 的256KiB，不能當成 C32 每 expert 只有256KiB：其係數是8MiB/expert、512MiB/layer。反過來，小 bank 也可能全 resident；對照必給同函數的 all-host／hot cache／async ordinary-load。原 OLM mixed-Q4 是另一個 matched-quality 強對照：H2048、I1024、64 experts、16 layers 的三投影純 W4 係數約3GiB，尚未含 scales、非FFN、KV及 workspace，這不是 inclusive8GB 已通過的證書。不能為了製造 misses 而禁止 host residency；也不能由容量可能 fit 推出 PE 永遠沒有 compute 價值。原生低 bit CPU kernel 必須正面面對，而非只對比自寫 W8 loop。[T-MAC paper](https://arxiv.org/abs/2407.00088)、[作者 artifact](https://github.com/microsoft/T-MAC)。

2. **共 P、線性 adapter、共 output basis 需要同能力 reuse 對照。** 在實數算術下，若多 expert 共 U，`sum(alpha_e * U*h_e)=U*sum(alpha_e*h_e)`，所以不能強制控制組做 K 次 lift。實際 FP32 premerge 另定數值契約，不能偷偷叫 bit-exact。MoLAE/EBC/Flash-Jacobian 的來源邊界已在 [novelty follow-up](/home/ericyu/expert_flash_experiment/docs/novelty_followup_terra.md) 關閉。R2 的 rank／guard／geometry 微調不是新的宏觀機制；不過「通用框架能表達」本身也不是否決證書，要指出同資源的具體等價或支配路徑。

3. **一頁多 input、fusion、chunk frontier 本身已是強 prior。** [SLIM §5.4](https://arxiv.org/html/2507.09201v1) 已含 read-once input batch、output-stationary buffers，以及 gate/up neuron接Down row的 fused FFN；§6 有 page packing。Speculation 再給 [FASER](https://arxiv.org/html/2604.20503v1)／[FlowSpec](https://arxiv.org/html/2507.02620v2) 同因果-ready、chunk／early-abort能力。未來 token 的 X、同一 PB 的存活時間、跨 plane 的獨立狀態及多 input contexts 都不是免費。普通 loader 也能 early-read、read-once後在 CPU 做 grouped GEMM；batch throughput 不能冒充單流 decode latency。

4. **LUT／bitplanes／動態 precision 不能只重新命名。** [CodeGEMM](https://arxiv.org/abs/2512.17970) 已用 activation-centroid partial sums 加 index gather 避免逐權重重建；T-MAC 已有 bitwise LUT 計算。四 bit-plane 放四 NAND planes，不減少 `N*b/8` 必讀 payload：等資源的 packed-byte布局同樣使用四 planes。新收益必須來自具體 codec/PE/port/sensing 差異，不能來自把控制組設成單 plane。漸進讀取的控制至少含 [DP-LLM](https://arxiv.org/html/2508.06041v2)、[DPA-MoE](https://arxiv.org/html/2610.21378v1) 的誤差／live-alpha分配能力，以及 cheap norm基準。 bytes 減少尚不等於合法 page waves 減少。

5. **tiny RF／FIFO 是容量算術，不是 minor PPA 已證。** AiF 原文留下 activation 給 host，已有 FIFO/SREG；不是完全沒有 buffer，也沒有公布足以免費套用16 lanes、短 row reset、多 context或雙 port的服務證書。須列 input／corrected-codeword／accumulator／output的所有權與同時存活量，普通 read 與 GEMV 共用真 bank busy排程。ECC6.4GB/s不是MAC率，CSB46us不是免費可計算格式，28us不是任意可調的 knob。特定構造的96.96us下界與 resident EPYC服務時間，不能比較成架構支配；命名的 feasible ordinary實作也不等最優ordinary。缺 artifact 留作 unresolved，不等不可能。[來源稽核](/home/ericyu/expert_flash_experiment/docs/slim_aif_baseline_source_audit_terra.md)。

## 哪些問題值得下一個正式30分鐘，而非直接做 probe

目前沒有直接背書採納的候選。以下是可收束的問題，不是新意或收益承諾：

- **優先窄核 native mixed-format＋producer/consumer frontier。** Sol 新 pool 的 Q4_K/Q6_K×Q8_K block運算，可能相對 AiF plain-W8改變必要 sense bytes；原生256-neuron quantizer邊界也可能改變 GU→Down dependency。值得思考的是實際 scales/min/bsums、signed range、block rescale／round order、partial accumulator和 packed-page 生命週期。SLIM fusion 已舊，同 byte 下的 padding／同 array爭用可能吃掉整個 frontier收益；ordinary-load得到同早期 tile。要有一個可命名的新增 primitive 或資源差異，不是只換目的地。
- **read-effort pool 只剩 NAND-specific 不確定性。** 若 Down 讀誤差零均值且不相關，條件於實際 alpha,z，有 `E||delta_y||^2 = alpha^2 * sum_ij(row_scale_i^2 * z_j^2 * Var(error_ij(t)))`。因此最強便宜控制是 per-page block-energy×effort-variance曲線分配，另給逐 expert alpha²norm和靜態配置；不需要 sketch 才能做到。stochastic read error與固定量化殘差不同，這個差異本身尚不是貢獻。必須說清 ECC-success本已exact、是否允未校正 bytes、bias/correlation/burst tails、重試政策及真正nonCR的可控讀取曲線；不能把 precharge/charge reuse換名。沒有合法 control knob 就先 source veto，別用合成 BER 填實驗輪。
- **compressed-domain primitive 保留作備案。** 要保留真 SwiGLU並減少物理讀取／解碼必要工作，卻不能只移植 CodeGEMM/T-MAC。Root轉述 Terra 新找到 PALUTE 的 bank LUT／weight-index／accumulation／nonlinearity prior；該來源應正面比，不移植其 DRAM PPA到NAND。若只剩一般 lookup排程，先 thought-veto；若有明確 NAND ECC／page／PB port tradeoff，才值得獨立正式思考。

這些項目不要求證所有架構全域最優、所有42層pass，或事前已有完整RTL；需要的是一個可反駁的 work/resource假說和合理強控制。有限思考後若沒有必要差異，可以否決而不耗8次。

## Gem exploration block01：A/B 不進實驗

已實讀 `gemini_exploration_block01.md`。其 A/B 的採納理由含錯誤前提，應 thought-veto；不為驗證未證稀疏率開 weight probe，不為已破裂的 exact主張先補30分鐘時計。

- **A：`R=W-UV` exact不推出 sparse。** 完整SVD尾殘差一般是 dense、也不自動是相同尺度的整數。只留5%／10%殘差是近似；若保留完整R，flash bytes／waves通常回來。分開浮點乘加再相加也不保 canonical bit-exact，更不能宣布 PPL=0。CSR索引、row pointers、尺度與不同 GU/Down輸出維度都要計入；`H維8KiB永遠固定`不適用所有投影。現有稿沒有新必要機制，只有未證的 residual sparsity假設。
- **B：exact base不是品質下界。** `W_e=W_base+DeltaW_e`不能穿過 SwiGLU自動得到 `F_e(x)=F_base(x)+DeltaW_e*x`。若在函數層擬合殘差，仍是近似，90%共同表徵、32KiB增量、單 wave均未證。即使 `F_e=F_base+D_e`，合併式也是 `(sum live alpha_e)*F_base + sum alpha_e*D_e`，不能默認係數和為1。差增量可比base更差，沒有天然NLL／PPL防護。必給 base-only、相同小adapter全host、SPICE／common-basis類強控制；但此稿未過理論前提，不要求先做這些實驗。
- **量級與因果更正。** OLM三投影純W8是6MiB/expert、W4是3MiB；Down是三投影的1/3，不是超過一半。C32的單leaf／expert／layer大小不可混用。R2只否決那個凍結配置與校準，沒有隔離出唯一根因，更沒證所有非線性替代都不行。
- **其餘方向不能靠名稱放行。** 4-plane bit slicing沒減必讀bitwork；精確整數 dot也不自動等於帶 block scales/min與既定F32累加順序的 GGML算子。SLIM已有die/channel fused FFN，不能稱fusion由DRAM-PIM才成立。256-entry SiLU LUT需先定義輸入範圍、尺度與額外近似；整expert dynamic amax亦非免費。Full-rank circulant只說秩，不能證逼近任意已訓練W；一般block-circulant的參數量也不是單個長度H向量。共活整頁跳過需真zero或明定近似；99.9%能量不是PPL<0.05%證書，ordinary-load也給相同mask及controller能力。章程沒有一概禁止重訓，不能虛構此限制來否決／保護方向。

`首次`、`未曾`、PPL零／0.05%／0.1%、1-wave和小PPA的無來源數字一律撤回。保留能改寫成具體可檢驗問題的部分，不把草稿錯誤算成科學實驗失敗。method clock仍0，實驗次數仍2。
