# Proposed short-input AiF contract — unvalidated

This is a proposed minimal interface/row-loop extension for the same Round2 study, not a description of proven native silicon. It authorizes no implementation. AiF's paper specifies an input-dimension field and sequential page-range commands, but does not establish the minimum operand dimension, the short-row packing loop, PE lane count/rate, FIFO capacity or an exact short-vector wire format. These facts must be verified or explicitly treated as changes before performance claims.

## Arithmetic and row layout

For an input width r, store complete output rows contiguously, with r signed-I8 coefficients per row. No output row crosses a16KiB page. At r128, a page has128 complete rows and produces128 signed dot results. An H2048 map has16 pages and2048 outputs. At r64 the page holds256 rows; at r512 it holds32 rows. Row scale metadata and final bias are separate paid data.

The proposed PE loop takes the declared r, resets the accumulator at each row boundary, and emits an I32 integer sum after r products. For the shapes here, `r*127^2 < 2^31`, so I32 is sufficient for the signed dot; its output then follows the frozen host F32 row/input scaling and bias order. This does not imply a multiplier count or a particular rate. If the original datapath cannot change its row boundary at these widths, add an explicitly costed row-length counter/accumulator control and lane enable logic; do not label it a firmware-only change without the hardware information.

Ordinary be-enc133 LSB page access uses28us from AiF Table2. CR-read is excluded. Pages from different blocks retain ordinary independent sensing and address commands. Sequential ranges, per-plane restrictions, ECC, input posts, command splits, collisions and output drain remain paid. ECC6.4GB/s is not a PE or universal ordinary-read rate.

## Input context and commands

There is at most one live input context per physical chip/group unless another buffer is added and charged. The candidate has one shared layer A8 vector; factored controls can have different expert latents. Neither receives free K resident PB contexts or free input retention. A batching extension may carry multiple legal matrix descriptors and input vectors in one host submission, but both candidate and controls receive it. Each actual context replacement, input DMA/broadcast, page/block command and completion still appears in the schedule.

If a context ID permits repeated maps to use an already posted vector, specify its storage, PB lifetime across sensing, invalidation, group ownership and reuse dependencies. It is not implied merely by the vector fitting in the32KiB input buffer. Tiny input payloads do not establish that command latency is negligible.

## Output traffic and lower bounds

Assume one signed I32 result per complete output row for this contract. Then a r128 page produces512B. An H2048 map produces8192B, irrespective of r. Six routed maps produce49152B before scales, bias, routing and protocol. At AiF's modeled8GB/s external PCIe bandwidth, this payload alone takes at least6.144us; eight maps need8.192us. No claim about real FIFO service latency follows.

With64planes, one full simultaneous r128-page wave produces32768B. Equal distribution gives4096B per2GB/s channel, a2.048us channel-payload lower bound, and4.096us at8GB/s external bandwidth. Whether sensing, PE work and draining overlap depends on the actual FIFO and command implementation. The FIFO streams results; the status register is not a whole-output buffer.

The minimum total page counts for K6/K8 r128 maps are96/128 pages. Dividing by64planes gives a sensing-work lower bound of two waves, not a realizable schedule or a whole-FFN latency. Placement collisions, complete-row geometry, group sizes and the unchanged shared/fallback branches can raise or alter the critical path. A shared U512 has64 pages for one full H2048 lift, but still pays its host latent work, input posting, row scales, output traffic and any fallback/shared dependencies.

## Validation needed before a system claim

Obtain or model the short-width PE loop, legal multi-plane commands, corrected-byte delivery, accumulator output/FIFO behavior and command/input timing. Validate actual W8/A8 predictions against the same numeric reference. Compare the original expert, local bank, factored U and W4 variants under the same modifications, host capacity and legal scheduler. Keep current-host CPU measurements, target-host sensitivity, modeled AiF timing and real-device evidence distinct.
