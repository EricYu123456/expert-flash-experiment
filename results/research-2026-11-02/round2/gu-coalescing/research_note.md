# Ordinary-read GU row coalescing

Classification: **generic strong compiler control; no positive new systems result established**. This screen preserves expert-to-chip ownership while testing static row order. It does not exclude all joint reshards or cluster layouts.

The old static-bundle and selected-WL-union studies were read first. Static bundles always read every member; optional prefix/gap coalescing is a different policy. The old G4/Deep bounds do not certify the newer Qwen38 workload. Earlier horizontal down bundles express a different weighted-sum operation; down remains untouched here.

## Native lowering

Routed and shared GU rows have the same full2560-dimensional input, so vertical concatenation is ordinary dense GEMV algebra. A static expert/row map separates returned results. This does not permit skipping inactive gaps or resetting the input index at a hidden expert boundary.

The sealed address image reserves whole waves for each matrix. Padding between adjacent reservations need not be divisible by C, so these reservations cannot automatically become a continuous matrix. Repack actual complete rows, or pad with complete C-wide dummy rows. Read prefixes, inactive gaps and dummy rows all cost sensing, computation and results; runs crossing192WL must split. [AiF §4.4/Figure14](https://doi.org/10.1145/3695053.3731073) does not establish an intra-command expert reset/skip operation.

C2560 has a page/row common boundary of32rows=5pages and a WL/row common boundary of128rows=5waves. The128row rule is the conservative modeled start. The32row branch additionally ignores the initial intra-WL offset in its wave count: it is an ideal-window relaxation, **not executable ordinary multipane**.

Output-count/scaling/fringe semantics remain conditional. Early expert-ready events inside a still-running merged call would need a separate streaming-output timing model; this ledger does not assume them.

## Training screen

Existing Qwen38 training data only: all16requests, steps0,4,...796, first3layers,3200queries/layer. Four frozen full-input families receive identical identity, frequency and generic coactivation-path controls, tight/padded layouts and exact interval-cover DP:144cases. No external/test was opened and placement labels were not refitted.

Costs include ordinary28us/wave,250ns/command,1280ns input reload and all emitted4B results. Inter-chip/channel/controller/PCIe queues and GU-to-down readiness are relaxed. These are **GU native-work bounds, not FFN latency predictions**.

| Family | Original peak-chip GU context mean, L0/L1/L2 (ms) | Best conservative reduction |
|---|---:|---:|
| group_g16 | 1.251074 / 1.251074 / 1.251074 | 0 / 0 / 0 |
| group_best | 1.186373 / 1.210321 / 1.209801 | 0 / 0 / 0 |
| overlap_best | 1.144716 / 1.153179 / 1.158580 | 0 / 0 / 0 |
| whole_row | 1.041490 / 1.044822 / 1.044638 | 0 / 0 / 0 |

Best strict controls retain singleton calls. Tight packing is11–59% slower across tested cases because prefix work outweighs merging. Even the favorable32row relaxation only improves weaker g16/g8 families and still loses to whole-row.

Capacity is explicit. L0 aligned128row-stride g16/whole_row retain1.56555GiB useful GU weights but program/reserve2.50488GiB including padding and allocate2.625GiB LSB blocks. These are alternative one-copy layouts, not vendor-capacity savings.

[screen.json](/home/ericyu/expert_flash_experiment/results/research-2026-11-02/round2/gu-coalescing/screen.json) retains all results/inventories. [independent_audit.json](/home/ericyu/expert_flash_experiment/results/research-2026-11-02/round2/gu-coalescing/independent_audit.json) checks disjoint intervals, one-copy row counts, aggregates and6912 independent Python/C++ DP cases.

## Fixed whole-row certificate

A nonempty GU job has75,76 or101rows. Its ceiling slack is at most0.0703125wave. At most11objects are selected (top10 plus shared), giving total slack below1:

`sum ceil(x_i) - ceil(sum x_i) = floor(sum ceiling-slacks) = 0.`

Even perfect query-adaptive adjacency without alignment/gaps cannot save ordinary NAND waves for this fixed geometry. All153600train query/chip cases reproduce this. Changed sharding, dimensions or selected-set size are outside the certificate.

Only command/input work remains: at most15.3us per native chip context,25.6us input demand per channel and40us aggregate controller work (176 to16commands) can disappear. These overlapping resource budgets are not additive or FFN speed bounds. GU already has one2us host submission and one common320ns PCIe upload; neither may be credited once per expert. Down private input/submission remains. Extra CPU row demultiplexing/filter time is uncalibrated and zero in the favorable screen.

## Prior and next evidence

[NVIDIA FuseGemms](https://github.com/NVIDIA/TensorRT-LLM/blob/c295dd9fca143a3fdd2769617699fdd4d4a93eb5/tensorrt_llm/_torch/auto_deploy/transform/library/fusion.py#L45) implements shared-input output-row concatenation and output slicing. [Official MoE-as-Dense-GEMM](https://nvidia.github.io/TensorRT-LLM/1.3.0rc26/blogs/tech_blog/blog24_MoE_as_Dense_GEMM.html) places expert FC1 along output N and charges inactive work. Same-input fusion itself is established.

[Neuralink §5.1](https://arxiv.org/html/2410.19274v3#S5.SS1) already joins nearby accesses by reading intervening inactive neurons. [SLIM §6.1](https://arxiv.org/html/2507.09201v1#S6.SS1) packs consecutive fused vectors into aligned pages. Their retrieval/cache or custom datapath capabilities cannot silently become AiF operations.

The generic compiler should give every baseline coalescing, lower to actual legal addresses, retain single-context ownership and whole-expert readiness, then replay shared/routed GU and down together. More elaborate aligned asymmetric clusters or changed geometry remain untested. A positive systems claim would need to beat these adapted controls and demonstrate the responsible native constraint. This screen supplies no reason to expand fitting or open fresh heldout data.

