# Asymmetric GU / private Down: prior and attribution audit

The proposed mechanism is a plausible **CR-aware retuning of an existing mapping capability**, not evidence of a new sharding concept. This repository already permits independent GU and Down row profiles/depths, and its initial AiF notes explicitly identify short-run startup loss and grouped sharding. Even a large new full-FFN gain should first update the strongest baseline.

## Primary-source scope

| Source | Direct prior observed | Exact proposed combination observed? |
|---|---|---|
| [Stratum §4.1](https://arxiv.org/html/2510.05245v1#S4.SS1) | Explicitly reuses the common token input across active experts and chooses different GU/Down partition directions. All PUs process one expert at a time; Down uses input partitions and partial-output reduction. | No compact private-Down group chosen to amortize NAND CR cold startup in inspected sections. Common-input availability and projection-specific mapping are already prior. |
| [HDA-MoE §§IV–V](https://arxiv.org/html/2609.08682v1#S5.SS2) | Fractional expert deployment trades compute balance against communication/capacity; coactivated-group node participation controls traffic. | Inspected formulation has aggregate expert fraction P_ic, without separate GU/Down row maps or NAND-run startup. This distinguishes the concrete cost model, not hybrid placement in general. |
| [Official HDA code](https://github.com/PKU-SEC-Lab/HDA-MoE-TCAD26/blob/56ea3725b2c621d4aea248f16345c34f5e06b8ed/node_allocation.py) | Placement tensor is layer × expert × node; compute proxy and group-node communication use that tensor. Current main was also fetched read-only. | No phase axis in the inspected optimizer. No author experiment was rerun. |
| [SLIM §§5.4,6.1](https://arxiv.org/html/2507.09201v1#S6.SS1) | Fuses neuron-associated GU/Down vectors, stores page-aligned runs and packs vectors because every page read pays tR. Requires PE partial sums/reduction. | Not the same full-output-row private-Down construction or CR-specific degree choice. Packing to amortize read cost is already explicit prior. |
| [AiF §§4.1,4.2,5.2](https://doi.org/10.1145/3695053.3731073) | Equal submatrices across chips, sequential same-block runs and CR phase reuse; host handles model-specific operations. | No inspected claim of different GU/Down group degrees with cross-expert GU aggregation. The disclosed native GEMV remains the hardware boundary. |

This is a bounded inspection, not priority clearance. No verified author Stratum/SLIM/AiF code artifact was established in this pass; unrelated same-name GitHub results were excluded. SLIM inspection is v1. Source hashes, exact read scopes and local primary extracts are in sources.json.

## Existing repository capability is the closest implementation baseline

- `docs/aif_model_notes.md:60–125` already states `28+(P−1)*9.7`, explains poor startup efficiency of small shards, sweeps grouped row-shard degree and requires real group collisions. It initially assigns all three projections the same group; it does not make cold amortization a new observation.
- `scripts/research_20261102_whole_row_compiler.py:164` independently calls the row/depth compiler for GU and Down. `docs/research_2026_11_02_round2.md:62` already discusses per-phase degree fitting, and earlier repository experiments have asymmetric GU/Down groups. Independent projection mapping cannot be claimed as a new primitive.
- Existing common-input GU coalescing and coupled GU/Down refinements already require matching coalescing, row geometry and shared-expert controls. Replacing their ordinary-read cost by CR startup/service is the immediate strongest retuning control.

## Mechanism arithmetic and limits

Use **W8_A8**, four planes, 16KiB pages. Down has 1536 output rows and 512bytes/row. A g12 one-wave profile uses twelve 128row jobs; a g4 profile uses four 384row jobs, each12pages/3full waves with fixed0123 drain and complete32row pages. Neither changes coefficients, dot length or input contexts.

Ordinary28us/wave gives both profiles12aggregate wave-services =336us. Under the stated cold/CR service convention, g12 costs12×28=336us aggregate array occupation, while g4 costs4×(28+2×9.7)=189.6us. The146.4us aggregate difference is eight avoided cold starts. A single g4 job nevertheless grows from28 to47.4us, and co-selected experts can collide on four-chip groups. These sums are neither full-FFN latency nor achievable speedup estimates.

## Required strongest same-capability controls

1. Reoptimize the existing generic whole-row compiler for the CR cost, allowing GU/Down independent degree, depth, unequal row profiles and train-selected masks/rotations. Include intermediate degrees/profiles, not only powers of two. Do not compare only with an ordinary-frozen high-degree Down map.
2. Give every GU comparator the identical private-Down and shared-Down menus, train budget and scheduler. Also give the generic baseline the same common-input GU aggregation, same-block placement and native fixed-source options. A two-by-two GU/Down ablation must isolate geometry retuning from any new GU layout.
3. Keep single-copy capacity, actual physical row coverage, block transitions, channels, command launch, full A8 private-input reload and output FIFO accounting matched. Compact Down decreases commands/input fanout but increases each job's output burst and critical path; those effects belong to the mechanism attribution.
4. Preserve complete-expert GU → host activation/A8 → private Down causality. Different experts' Down vectors cannot be merged as one common input. Stratum/SLIM partial reductions and local nonlinear fusion are prior capabilities, not free AiF controls.
5. Use real routed groups, include route-blind balanced grouping and train-learned grouping, retain collision regressions/P99, and select geometry before held-out evaluation. Per-query collision-free reassignment is a relaxed oracle unless weights are actually available there.
6. Report sensitivity to cold/CR service, legal read cadence, finite PB/ECC assumptions and channel/command costs. Keep effective6.4GB/s-only service separate. A physical timing claim needs native calibration; finite timing-model correctness alone does not supply it.

## Publication attribution red lines

Do not claim first projection-specific mapping, first asymmetric shard degree, first common-input reuse, first startup amortization, a new standard DP/CP placement algorithm, or unconditional native support. A gain against frozen ordinary-cost geometry is retuning benefit, even if numerically large. If the reoptimized generic compiler matches it, no residual mapping-algorithm claim remains.

A defensible narrow statement is: **under the declared AiF-like native contract, CR startup makes the tradeoff between inter-expert group collisions and private-projection depth materially different from an ordinary-read model; this is measured with matched compiler controls, fixed numerical semantics and a causal full-FFN replay.** This can be a useful finding or baseline correction without establishing a publication breakthrough.

The sealed pipeline metadata misread of `q=4` is separately corrected by `pipeline/metadata_erratum.json`: q denotes four planes; reference weights are W8, with1536bytes per single GU output row and512bytes per Down output row. The sealed event arithmetic is unchanged.
