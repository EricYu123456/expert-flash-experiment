# Recent Flash/MoE placement prior audit

Audit date: 2026-11-02. The five latest available versions and pinned public repositories were inspected. This is a literature/code audit; no model, trace payload, fitting or performance experiment was run. [review.json](/home/ericyu/expert_flash_experiment/results/research-2026-11-02/round2/recent-placement-prior/review.json) records exact metadata, reading depth, source sections and repository pins. [source_receipts.json](/home/ericyu/expert_flash_experiment/results/research-2026-11-02/round2/recent-placement-prior/source_receipts.json) records local source hashes.

| Primary work | Placement level | Direct novelty threat | Transfer boundary |
|---|---|---|---|
| [BigMoMo v2](https://arxiv.org/html/2609.14643v2#S4.SS3), Oct 3 | Runtime whole-expert file-slot swaps | Co-loading-conditioned online contiguity optimization | Actual writes; resident-DRAM relocation premise |
| [Paging the Experts v1](https://arxiv.org/html/2609.29032v1#S3), Sep 24 | Aligned expert files and DRAM cache | Packing/cache accounting and transparency controls | No exposed NAND physical ownership |
| [FlashAccel v2](https://arxiv.org/html/2607.10186v2#S5), Aug 22 | Physical megaplane chunks | Plane-aware static weight striping | New HBF/GPU integration |
| [HBFSim / SlugLab v3](https://arxiv.org/html/2609.09800v3#S4), Sep 30 | Registered ranges, modeled HBF | Live execution plus address/service ledgers | Injected timing, unavailable HBF |
| [HBFSim / FlashLabAE v1](https://arxiv.org/html/2610.23503v1#S3.SS3), Oct 16 | Host FTL to physical planes | Stripe/resource/copy accounting | Assumed HBF timing, synthetic MoE routing |

The two HBFSim titles refer to distinct projects. The FlashAccel author-associated repository contains explanation slides, not implementation code. BigMoMo's first public version is September 13, although its revision is in October. Repository evidence and search scope are retained in the receipts.

## Consequences for the current project

No confidently new drop-in placement mechanism was identified in this bounded search. Online co-loading grouping, plane striping, and address manifests should be treated as established ingredients. Existing AiF's complete-matrix command, input reload, result transfer, single-context lifetime and GU/down dependency constraints remain a legitimate systems question; their combination must be lowered and tested, not declared novel from vocabulary alone.

BigMoMo optimizes retrieval contiguity. A read-in-place engine optimizes legal native jobs and their critical resource path. Converting one objective to the other is a baseline adaptation, not a new algorithm by itself. A fair adaptation must keep router outputs, precision, capacity and hardware capabilities fixed. It must charge inactive overfetch, command count, all input/output traffic, activation/combine, and any explicit migration.

Similarly, FlashAccel's round-robin striping is a required physical control. Its new command grouping and SRAM support cannot silently become existing AiF functionality. The relevant comparison is the strongest legal native realization of its balancing principle.

## One bounded, falsifiable gap

The only concrete follow-up supported by this search is **costed phase-boundary reconfiguration of native-legal layouts**. This is a hypothesis, not a novelty claim: classic placement migration and BigMoMo are strong direct priors.

Use only past requests to choose among precompiled legal layouts at an epoch boundary. Keep weights resident in Flash during inference. If changing physical ownership requires moving weights, execute and charge ordinary reads, programs, erase-block effects, scratch space, metadata updates and quiescence. If selecting a preinstalled copy avoids moving weights, charge that copy's full storage and legal retrieval constraints. A virtual-ID permutation alone cannot create new physical adjacency.

First compute a generous switch oracle over frozen candidate plans. Its total includes migration on the same chip/channel/host resources as foreground native jobs. For a stable epoch of N requests, a necessary amortization condition is

`N * (T_old - T_new) > C_move`

where `C_move` is the incremental foreground cost under an explicit migration schedule, not an assumed free background duration. Without overlap, the nominal break-even count is `ceil(C_move / (T_old - T_new))`; this is only a screening bound. With overlap, replay the complete resource schedule.

Strong controls are: static robust/mixture-trained placement; the best all-width/full-row and native rectangle families permitted by the verified ABI; a BigMoMo-style recent-window swap policy lowered to the same backend and charged the same migration ledger; fixed copies plus generic replicated-declustering selection at identical capacity; and an epoch oracle that pays all switching costs. No oracle may refit on the existing external task.

Fail the direction if even the optimistic migration-paid oracle does not beat the best static control; if gains disappear under realistic epoch lengths; or if they require free source reads, per-query weight motion, unsupported simultaneous contexts, precision changes, or an unverified native output contract. A positive trace-cost result would still need ordinary program/erase calibration and executable address lowering before a hardware latency claim.

Neither HBFSim supplies those AiF measurements. Its instrumentation or calibrated frontend can inspire validation methodology, but cannot validate an unrelated native interface. Read-only weights also do not spontaneously generate heavy GC: only modeled real writes or documented retention maintenance justify that traffic.

A publishable claim would need to identify which native constraint causes the static/dynamic tradeoff, demonstrate it against these adapted controls on true routes and disjoint tasks, and expose the physical plan plus migration ledger. Without that evidence, retain the present work as a conditional characterization rather than announce another runtime coactivation placement method.

