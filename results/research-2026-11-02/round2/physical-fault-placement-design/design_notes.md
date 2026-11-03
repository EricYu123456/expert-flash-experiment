# Fixed-ECC, fault-domain-aware expert placement: bounded design audit

The candidate is **not ready for a model fault-injection experiment**. It has an identifiable conditional mechanism, but the newly inspected public NAND data does not identify the post-correction fault domains needed to measure it. No latency, quality, or reliability gain is claimed. This refreshes the earlier `physical-error-prior/review.json` evidence rather than presenting reliability placement as a new idea.

The possible mechanism is input-conditioned failure impact across experts sharing a physical failure domain. A placement that uses the same bytes, ECC and full-row operations can alter the joint distribution of affected selected experts, even after matching each expert's marginal fault exposure and importance. This is distinct from read-disturb hotness/reclaim latency, but both reliability-aware mapping and sensitivity-aware allocation already have close prior.

## Primary evidence and the closest controls

[BALANCE's official ASP-DAC 2026 slides](https://www.aspdac.com/aspdac2026/archive/pdf/4B-2.pdf), slides 6–12, combine bit and layer importance, allocate ECC, distribute units across dies/planes, and map important units to reliable lower pages. The full paper was not obtained in this subtask. Its changed ECC and fixed BF16 bits are outside this experiment's contract. Its **importance ranking and physical mapping** remain mandatory controls; the current W8 row format must not be split into bit planes to imitate it.

[Groundhog's author preprint](https://arxiv.org/html/2608.25276v1), abstract and sections 1–3, supplies a nearby example of input/token-conditioned expert significance. Its adversarial router faults and expert deactivation are not natural NAND errors and cannot calibrate this experiment. No attack or expert deactivation was run.

[QCFlash's official artifact](https://zenodo.org/records/21517028) provides actual cell-indexed threshold voltages, programmed QLC states and shared-page bytes for three wordlines of a Micron 96-layer QLC device. Its code is MIT licensed and its data CC BY 4.0. It is a genuine raw sensing dataset, not an AiF post-ECC fault map. The [FAST 2016 field study](https://www.usenix.org/conference/fast16/technical-sessions/presentation/schroeder) independently cautions that RBER does not predict uncorrectable-error incidence. We use neither source to invent service times.

AiF's local primary text, `paper/aif.txt` lines 779–782 and 816–817, places ECCLite **before** the digital PE and states correction of up to 10 errors per 1 KiB. Its single-sensing weight pages use be-enc and are not the released QCFlash QLC state encoding. No ECC bypass, altered correction strength, new parity, bit format, decoder, latch allocation or independent context is assumed.

## Measured-data inspection

The frozen dependency-free script reads the original XLSX XML directly, validates every programmed state against all four shared-page byte columns, and reconstructs native reads at the **released, unshifted** reference voltages. It does not execute QCFlash's shifted-voltage bitwise operations. The original artifact archive has SHA256 `f4e434bb05d994b73bee78a77b95d66f68a70aa46f9d96077f7a2837aea06fbb`; its MD5 matches the official record.

| WL | Cells / bytes per logical page | Observed Vth | Missing Vth | Native LSB raw errors | RBER among observed cells |
|---|---:|---:|---:|---:|---:|
| 10 | 65,528 / 8,191 | 65,488 | 40 | 6 | 0.00916198% |
| 11 | 65,528 / 8,191 | 65,432 | 96 | 10 | 0.01528304% |
| 12 | 65,528 / 8,191 | 65,422 | 106 | 15 | 0.02292807% |

The artifact fills missing Vth entries using their known programmed state. That is label substitution, not a measured successful read. We preserve a validity mask and report missing observations separately. Missing entries occur beyond just the erased state; their censoring mechanism is not documented sufficiently to infer their read outcomes.

Each 8,191-byte page contains seven complete logical 1 KiB windows and an 8,184-bit final window. The maximum observed LSB errors in a complete window are respectively 2, 3 and 2. These are **not verified ECC codewords**: parity, physical interleaver, spare-area errors and decoder outcomes are absent. Even a window with more than 10 errors would only exceed the stated correction guarantee; it would not establish a measured silent error or actual uncorrectable event. Missing values also prevent claiming all reads would be corrected.

The data identifies same-cell raw errors across four QLC page labels. That does not establish four independently usable AiF compute-page tiers. It has no repeated read/time series, per-device/block replication, stable post-ECC residual distribution, or joint domain-failure events. A cell-index error mask from one programmed QLC pattern cannot simply be transplanted onto arbitrary W8 data: changed values change the cell state, and any controller randomizer/interleaver must be respected. Neither raw errors nor error correlations are used as PE output errors.

## A small executable witness, with opposite objectives

Consider equally frequent routes `{a,b}` and `{c,d}`, four equally important experts with equal selection frequency, and two equal-capacity domains. A hypothetical domain fails with independent probability `p`; all experts assigned to that domain are affected. Exhaustive enumeration covers all three balanced placements (up to renaming domains), seven rational values of `p`, and all four domain states.

| Placement | Expected affected selected experts | Probability any selected expert affected | Probability two selected experts affected |
|---|---:|---:|---:|
| `{a,b}` together; `{c,d}` together | `2p` | `p` | `p` |
| `{a,c}` together; `{b,d}` together | `2p` | `2p−p²` | `p²` |

At the purely illustrative `p=1/1000`, spreading coactive experts reduces the two-expert event from `0.001` to `0.000001`, but raises the any-expert event to `0.001999`. This is a **tradeoff**, not a measured improvement. If failures are detected and execution requires every selected expert, the concentrated layout has better request availability. Exact recovery/abort must be charged; dropping an expert or changing routing is forbidden. Only a measured residual silent-corruption process plus demonstrated nonlinear conditional quality impact could make the second column of risk relevant to quality tail.

The one-point tail reduction is `p−p²`, never larger than 1/4 in this example, and small in absolute terms for small `p`. For expert residual marginal probabilities `p_e,p_f`, their joint probability is bounded by `max(0,p_e+p_f−1) ≤ q_ef ≤ min(p_e,p_f)`. Marginals alone leave that entire interval unidentified. With additive expected damage and matched marginal exposures, all these placements have identical expected loss. Equal marginal scores also create ties: a strong marginal baseline is permitted to choose the route-optimal tied placement; an unfavorable tie is not a valid evidence of improvement.

The witness establishes what must be measured. It does not supply a NAND failure rate, LLM quality response, theorem novelty, or a legal mapping on the current all-chip expert geometry.

## The actual strong baseline rules out the coarse-domain version

Read-only checks of frozen CP0107 establish that every one of its 64 routed GU operators has positive full-row storage on all 16 chips, and every one of the 512 TRAIN routes uses all 16. Therefore every total chip loss removes required GU rows for **every** query, independent of a legal within-chip expert-row permutation. When a complete local layer occupies one block per chip, total loss of that block has the same null. Copying or moving data into more independent blocks changes the resource budget and must compete with ordinary replicated recovery.

This is a finite domain-touch result, not a claim that all fault-aware placement or replication is impossible. Finer WL/codeword fault domains could escape it, but the inspected dataset does not establish their residual law or their relation to the native logical byte mapping.

## Smallest physically measurable next contract, if new data arrives

1. Keep exact W8 coefficients, logical row scales, routing, host arithmetic, correction strength and ECC/interleaver fixed. Establish the actual data randomizer and physical row/page/codeword mapping first. A known-zero-fault read must reproduce the original numerical fixture bitwise.
2. Measure repeated reads and reprogramming on identifiable physical regions with the original decoder enabled. Record corrected counts, detected failures and any verified silent corruption separately, including codeword parity, programmed state and temperature/retention conditions. Keep whole-row dense jobs and their original full input. Do not substitute independent bit flips for these measurements.
3. Start with existing TRAIN routes and a tiny conditional-impact probe only after the physical law passes identification. Compare random legal mappings, frequency, data-aware bit×layer importance aggregated to whole rows, sensitivity/Hessian matching, and generic fault-domain declustering/replicated recovery. Give every control the same useful bytes, programmed pages, allocated blocks, copies, commands and recovery policy. Shared projections/scales/metadata are included, not silently assumed perfect for only one family.
4. Report detected request failures, retries/recovery, completed-request quality **and** quality with failed requests counted, worst affected-expert multiplicity, and capacity. A quality-tail surrogate alone cannot establish model accuracy. Do not silently omit failed requests from denominators.
5. Stop if post-ECC outcomes are zero at measurable resolution, exchangeable after controlling programmed state, or explained by marginal/data-aware importance; if domain correlation is unstable after reprogramming; or if generic domain-aware replication matches the route-conditioned map. A future validation would need independent physical devices/regions and prompts, but none are read or fitted here.

This subtask advances **one** conditional mechanism and recommends no new heavy experiment. It does not add generic power placement or synthetic timing tiers to fill a three-item quota. SLIM native batching and Cambricon hybrid read/compute capabilities are separately screened in `existing-capability-mechanism-screen`; their advertised mechanisms are prior, and total batching or hybrid gains are not placement gains.
