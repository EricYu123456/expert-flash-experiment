本輪沒有確認新的 >5% 同容量配置收益。

SLIM §5.4 明示 batched GEMM 的 read-once 權重能力，可先做 TRAIN B4/B8 必要工作量上界篩選；Table2 的64KiB是 PE SRAM總量，不能全部當input。所有布局須用相同batching/reduce強控制，總batch收益不等於placement收益。

NVLLM §3.1明示MLC/TLC多讀取電壓，因此不是LSB-only；但沒有頁類可見延遲/地址資料，FIFO還可能隱藏差異。Cambricon有獨立input buffer與共享die core，卻未給混合頁類契約。SLIM比較兩種SSD、KVNAND採SLC，均不能當同一器件兩級GEMV。Lincoln全文仍未取得。

NASiC已在舊priorbase，本次重讀確認CAM+expert interleave直接相關；其analog/ADC能力不移植到普通NAND GEMV。另只保留Cambricon既有read/compute雙路角色的低優先理想bound，未證有新配置headroom。
