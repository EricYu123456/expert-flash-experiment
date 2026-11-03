# TRAIN-only ONFI topology placement audit

The unsearched common-chip pairing freedom exists, but it has no promising latency headroom in the current ordinary-read service contract. At 250 ns per native command, a placement-independent work bound limits any common GU/Down/shared chip permutation, even with an improved scheduler, to at most **0.383–0.514% G4** and **0.109–0.215% Deep** average FFN-service improvement over the frozen whole-row plans. A bounded 5 μs-command replay finds only **0–0.006%** improvement from 32 fixed common permutations. Do not expand this branch into a channel-label fitter.

This is a fixed-plan topology result. It does not establish an optimum over different GU/Down chip workloads, row partitions, copies, or schedules. In particular, independently moving GU and Down can change joined chip load; such a gain must be compared against ordinary joint phase placement and cannot be attributed to ONFI pairing alone.

## Existing coverage and the missing freedom

The inspected compiler fits each phase using maximum per-chip native context work. That context includes command duration, full-input reload, NAND waves and output-channel drain. It does not directly fit shared-channel arbitration. However, the pair selector already considers all 16 relative Down rotations, sums traffic over the actual chip-to-channel topology, and runs the full channel-aware event replay. Its controls include disjoint groups, every cyclic width and whole-row work/depth geometry with identical cost contracts.

The backend has 16 chips and eight channels, with channel `chip % 8`. Therefore chips `(0,8),…,(7,15)` contend. An arbitrary common permutation of all routed GU, routed Down and shared jobs changes these pairs while preserving every original chip's complete workload and dependencies. Cyclic relative rotations do not enumerate every such perfect matching. A common permutation also preserves the multiset of physical chip inventories. All commands remain full-row, ordinary-read dense GEMVs with one context per chip; this audit preserves logical row/job order and changes physical chip IDs only.

## Frozen inputs

Both inputs are the W8A8-format TRAIN campaigns, not the older checkpoint-routing campaigns:

- `full-campaign-v5-W8-training`: one channel input copy, summary SHA256 `093801006fe2788a5fcd4439b1171fb850095fa376dc3cef56d9a1684a02238e`.
- `full-campaign-v6-W8-input4`: four channel input copies, summary SHA256 `c85111c0ece822615b8f38e079de23bbf0fd383849a9f1b42ed1215aaf7c0392`.
- Selected-plan anchor SHA256 `00507ff56f9197fb18511ad58303919214c03cba6f0c1440f6733ffc42944d00`.

Each layer uses the same 64 frozen TRAIN selection indices from the 32-prompt × 16-step W8 capture. Analysis covers all 40 G4 FFN layers and all 26 Deep MoE FFN layers, including each always-active shared FFN. No test prompts, external traces, answers, or numeric outputs were read or used. These are synthetic FFN-service times; attention/router/other model work and hardware calibration are absent.

The service contract uses 28 μs ordinary reads, four 16 KiB pages per wave, one chip context from command acquisition through channel-output drain, 2 GB/s per ONFI channel, 8 GB/s PCIe, 2 μs host submissions, page-fringe output drain, and complete GU → host activation → full-vector Down-input dependence. The two input-copy counts are explicit ABI sensitivities. This audit introduces no quantization change, retained input, extra context, column partial or cache reuse.

## Strict common-permutation bound

For each selected expert, phase and chip, independently sum:

`command + full-input ONFI transfer + ceil(pages/4) × 28 μs + emitted-output ONFI transfer`.

A chip owns its context throughout these operations. Any common chip permutation preserves the maximum total context work across chips. Every legal schedule is at least initial host submit plus that maximum. The certificate additionally takes the maximum with controller work, PCIe work, dependency-only expert paths, host work and pooled ONFI work. These terms remain invariant under every common permutation. Their maximum is a valid per-query lower bound, so `current latency − lower bound` upper-bounds possible improvement. Layer sums preserve this inequality; the reported P99 bound is the percentile of the paired sums, not a sum of layer percentiles.

At 250 ns command cost:

| Model / input copies | Strong cyclic mean service | Whole-row mean service | Cyclic maximum possible mean reduction | Whole-row maximum possible mean reduction | Whole-row maximum possible P99 reduction |
|---|---:|---:|---:|---:|---:|
| G4 / 1 | 23.412580 ms | 23.217778 ms | 0.3949% | **0.3830%** | 0.3846% |
| G4 / 4 | 24.053339 ms | 24.014731 ms | 0.5145% | **0.5145%** | 0.4805% |
| Deep / 1 | 52.356016 ms | 50.391603 ms | 0.1299% | **0.1090%** | 0.1084% |
| Deep / 4 | 53.451503 ms | 51.448331 ms | 0.3010% | **0.2150%** | 0.1954% |

These upper bounds generously include every remaining scheduler inefficiency, not only channel contention. Across all 792 plan/layer cases, the current channel-volume lower bound never exceeds the chip-context bound; the maximum channel/chip ratio is 11.512% at 250 ns and 6.865% at 5 μs. Low volume alone does not exclude burst collisions; the replay below checks that distinction.

At 5 μs command cost, the invariant bound is looser: about 7.0% for G4 and 3.4–3.5% for Deep whole-row plans. This is not evidence of channel headroom. Commands, chip acquisition and expert readiness can also leave this gap.

## Bounded replay and strong controls

Replay covers the first three layers, all 64 TRAIN selection queries, all three families, both command costs and both input-copy contracts: 72 original plans. The original costs reproduce exactly for all **4,608** queries. Independently computed invariant bounds also equal the existing native-node bounds on all 4,608. Twenty-four direct Python graph checks, totaling **21,864 nodes**, agree with the snapshot C++ engine.

At 5 μs, each plan receives the same 32 route-blind common permutations: identity, adjacent virtual-chip pairing, and 30 fixed random permutations from seed 20261103. The winner is selected only by TRAIN mean. Selection is per layer and preserves original command order. At 250 ns, the already small strict bound is sufficient; only identity and a private-channel diagnostic were run, with no permutation fitting.

Three-layer 5 μs results:

| Model / input copies | Strong cyclic: best common permutation | Whole-row: best common permutation | Whole-row: private-channel diagnostic |
|---|---:|---:|---:|
| G4 / 1 | 0% | 0% | 0% |
| G4 / 4 | 0% | 0% | 0% |
| Deep / 1 | 0.00585% | 0.00192% | 0.00020% |
| Deep / 4 | 0.00038% | 0.00533% | 0.01179% |

The private-channel diagnostic gives each chip its own 2 GB/s channel while retaining all job durations and the same greedy policy. It is an optimistic resource relaxation, **not an asserted lower bound on optimal scheduling**: greedy schedules can change when contention is removed. Indeed, a common permutation can occasionally beat this particular private-channel greedy schedule. Its observed gain at 250 ns is only 0.01284%/0.18809% for G4 and 0.01152%/0.13111% for Deep whole-row plans (one/four input copies). These values support the attribution, while the prior strict bound supports the normal-cost no-go conclusion.

## Primary prior and limits

[AiF, ISCA 2025](https://dl.acm.org/doi/10.1145/3695053.3731073), read in the repository full-text extraction, states the multi-drop channel constraint (§3.2), the eight-channel/two-chip configuration and ONFI bandwidth (Table 2), and input-vector/control overhead in the §6 scalability discussion. Its topology is real; it does not justify treating input broadcasts, controller dispatch or channel drain as free. Local extraction SHA256: `7511dfada743ed7cbc9884dc7664c704e2430f4197ea19afb1cbe2db5e39a3b7`.

[Cambricon-LLM, MICRO 2024 / arXiv v1](https://arxiv.org/html/2409.15654v1), primary full text read in §§IV-C and V, already optimizes channel-aware tile layout and input/output traffic and explains why read-compute output traffic alone underutilizes channels. Its Slice Control interleaves sliced conventional reads with read-compute operations and needs architectural support. That mechanism cannot be silently added to this AiF ordinary-read backend. The paper's channel-aware dense tiling is a direct control for any future claim about topology-aware placement; its setting does not directly test this MoE common-chip pairing experiment.

The result is a bounded negative finding, not a global novelty verdict. Reopening this direction requires native measurements or a documented existing contract showing material ONFI burst stalls after the same cyclic/whole-row controls, or a new workload with materially larger input/output traffic relative to NAND time. A new solver name or arbitrary chip-pair search is insufficient. Independent GU/Down permutations should first be tested as ordinary joined chip-load balancing with channel contention separately ablated.

Detailed records: `bounds.json`, `fixed_plan_bounds.npz`, `small_replay.json`, `small_replay.npz`. Frozen sources and output hashes are listed in `completion.json`.
