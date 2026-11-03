# Conditional AiF localbank break-even — Sol, same Round2

Started 2026-11-03 03:57:23 UTC. Source/theory only. Read Terra's hardware audit, the proposed short-input contract and Root's Stage2 qualifications. No experiment, candidate code, model/data/quality payload or new rank/grid. These are conditional performance regions, not measured speedups or proof of a minor hardware modification.

## What could be valuable, and what is already prior

AiF already separates host nonlinear processing from flash GEMV; host/flash splitting is not new. Local affine compilation could remove a replaced routed expert's GU→host-SwiGLU/A8→Down dependency and use one layer-shared input for the active expert cohort. Its possible contribution is the **joint frontier of contextual output capacity, active page geometry, host capacity/traffic and end-to-end quality**. Fixed low-rank host factors already remove flash barriers too. A local-MSE gap or the existence of a short input field does not establish publication value.

Deep's shared branch still has its own GU→Down dependency when run canonically in flash; replacing routed experts does not remove that whole-layer barrier. Exact source-GGUF Q8_0 shared (~.478GB supplied source footprint, distinct from raw canonical W8 coefficient arithmetic) is host-pinned unless a source-format execution contract is provided. A separately declared shared-W8A8 reference may compare host/AiF placement, after measuring its own quality loss. Both candidate and controls receive the same numerically legal placement choices. OLM has no shared branch and independently tests the mechanism.

## Symbols and one legal-shape resource accounting

Let R_PE be sustained useful INT8 products/s **per chip**, R_E corrected bytes/s actually delivered to the PE, B_CPU effective target-host DRAM bandwidth, P_CPU the relevant sustained host kernel rate, D_out the FIFO/channel/PCIe/polling service, and tau setup/input/context/block-command overhead. R_E may be bounded by the published6.4GB/s ECC decoder, but that is not R_PE and not a guaranteed ordinary-read effective rate. Use real cache-miss fractions h for each array; resident host bytes do not imply LLC hits. Current EPYC and target i9 are separate machines.

At H2048/r128 W8, each map has16 full16KiB pages and can use g4 chips, each processing64KiB/65,536 products and emitting512 I32 results (2KiB). With four disjoint groups and sufficient legal replicas, K6/K8 **all-approximate, flash-resident** cohorts have a two-round sensing-work lower bound. This is not the same as a DRAM cache hit: cached maps instead pay their CPU path. Partial coverage, canonical fallback, cache masks and group collisions are evaluated jointly per actual cohort, without independence assumptions.

For a complete uncached two-round cohort, the most-loaded chip executes131,072 products. Other resource lower bounds include total KHr/(16 R_PE), corrected bytes/(16 R_E), I32 output4KH/B_PCIe and actual per-channel traffic. K6 output is49,152B (6.144us payload at8GB/s); K8 is65,536B (8.192us). These resource services may overlap; they are not automatically additive elapsed times.

A useful **explicit hypothetical no-array/PE-overlap profile** is

`D_L = 2*[28us + max(65536/R_E, 65536/R_PE) + pipeline_fill] + tau_L + D_out,critical`,

where the max assumes corrected-byte→PE streaming, and D_out counts only service not already overlapped. A fully serial ECC/PE design replaces max by their sum. A validated cross-wave pipeline can differ. Neither is claimed as the published AiF schedule. The unconditional statement is only that a legal DAG must account for each service and dependency;56us is a sensing lower bound, not the answer.

If one hypothetically has L fully utilized product lanes at200MHz,64KiB MAC service is327.68/L us: L12→27.31us, L16→20.48us, L32→10.24us. The advertised decoder service for64KiB is10.24us. Thus a single-wave-after-sense datapath can be much slower than28us even when two sensing rounds suffice. AiF discloses neither L nor utilization; these are conditions to test, not inferred lane counts or proposed lane upgrades.

Narrow rows also increase result event rate. At r128 the per-chip producer rate is at most min(R_E,R_PE)/128 rows/s; a4byte row stream must be drained accordingly. A gap of delta seconds needs approximately4*producer_rate*delta FIFO bytes to avoid blocking, bounded by the finite outputs of the command. FIFO/status semantics, burst depth and polling latency are unknown. SREG is status, not a free output SRAM. If stalls occur they enter D_out and PE service, rather than disappearing behind aggregate bandwidth.

## Conditional comparison with the fixed q512 host factors

For PE, leading selected coefficient bytes per layer are `M_PE = K*q*(H+r)` at W8, plus common P/tree, U/B scales, biases and vectors. H2048/r128/q512 gives6.685MB at K6, plus common FP32 P1.049MB. G's global full-input factors instead read `M_G=2*K*H*q` (12.583MB at K6), with no common P; retained storage can be smaller while selected traffic is larger. Apply actual cache-miss fractions and actual kernel arithmetic, including W8→float conversion when used. A host floating latent needs no compulsory extra native A8 quantizer.

Write the isolated routed comparison as

`T_L = H_common + V_L + D_L`,
`T_PE = H_common + V_PE`.

Here V is **actual** host service under its cache/CPU/bandwidth model, including preprocessing unique to that method, scale/bias/alpha work and host-cached-map compute. When common P/tree truly match, localbank wins this serial isolated profile iff

`D_L < V_PE - V_L`.

For G, include the absence of P and its larger full-input factor work; do not cancel an operation it never performs. Host service has the roofline lower bound `V >= max(M_miss/B_CPU, MACs/P_CPU)`, but measured/validated feasible service, not this lower bound alone, decides the inequality. Comparing two lower bounds cannot establish dominance.

At the target's86.4GB/s **ceiling**, roughly7.8MB total PE layer traffic has a transfer-only floor near90us; local P alone is near12us. Whether D_L fits the remaining margin depends on PE/ECC/overlap, setup, outputs and cache. At a lower sustained bandwidth the host-factor cost rises; with cache hits it falls. These are crossover illustrations, not predictions. No claim that q512 is a necessary minimum or that a new rank will repair quality is made.

## Original host and native original are distinct strong controls

OLM's canonical W8 original routed weights≈6.44GB plus non-FFN≈.31GB and actual KV/workspace may fit the8GB target (known floor around6.8GB, not a fit proof). If it fits, it needs **zero routed NAND reads**. Its selected weight traffic is approximately`3KHI` and it pays CPU GU, full-I activation/A8, Down and combination. The localbank can win only if saved host work/traffic exceeds its real flash/input/output/setup service at matched quality. Capacity alone or zero NAND alone proves neither side faster. Exact source GGUF original is another numerical point and gets its own quality/cost.

For native original, W8 coefficient work is3KHI, with2KHI GU then KHI Down and private expert intermediate inputs. Model per-expert full-GU readiness, activation/amax and private Down upload; an optimized baseline need not wait for **every** expert's GU globally before starting any Down. Compare legal original and bank DAGs with the same chip/channel/context/command/metadata extensions. Less total work and one replaced-path phase can help the bank; they do not imply the same fraction of whole-model delay reduction.

## Shared placement, non-FFN and cache require a whole-DAG decision

Define `T_m = min_legal_placement makespan(DAG_m)` for each method under the same inclusive host budget. Add attention/KV/LM-head, host threads/bandwidth, shared work, P, scale/bias, transfer and joins. Shared-host can overlap bank flash but compete with all-host factors for CPU/memory; shared-flash can overlap host factors but compete with bank reads for chips. Either reversal is possible. Non-FFN dominance can erase an otherwise useful routed gain. Source precision restricts which placements are legal.

Localbank caching must be allowed too: a complete bank exceeding8GB does not imply every selected map misses. Use the actual mixed CPU-hit/flash-miss schedule, load/export/startup and eviction cost. Do not use independent hit probabilities or a fixed two-wave charge for a cohort served from DRAM. Target i9 cache and bandwidth must be distinguished from EPYC512MiB L3 measurements. Report8e9 and8GiB capacity interpretations, actual context-dependent KV/workspace, raw source/canonical formats and metadata separately.

## Small modification is a hypothesis, not an established fact

The proposed 128-row counter/reset/indexing loop might require only control if the operand addressing, row packing, plane streams, I32 accumulator and FIFO already suffice. Their availability is undisclosed. Required new lanes, muxing, accumulator/scale paths, buffering or FIFO backpressure can alter area/power and service. Until an authoritative ABI/RTL or a costed concrete design exists, report a **break-even region** in R_PE, tau, D_out, cache and B_CPU, and label short-row/packing requirements explicitly. Do not claim minor hardware changes have been proved merely because the command has a dimension field. CSB/MSB46us and different plane encodings are not silently used as extra capacity/bandwidth; no CR is allowed.

## Minimum useful negative evidence

For the bounded proposed value claim, a matched-quality baseline with a legal complete8GB implementation and a measured/validated feasible target service can refute the bank if it is no slower than even the bank's optimistic **whole-DAG** bound, allowing the bank its best legal cache/shared placement. A baseline lower bound alone cannot do so. Likewise, a validated short-row datapath whose PE/FIFO/setup costs lie outside every crossover region supported by real host controls would refute that hardware profile. If the required extra lanes/buffers exceed a demonstrated small-change budget, reject the small-modification claim; without PPA evidence, say unestablished rather than invent a budget failure.

If factors/original-host match quality but only their CPU time is unknown, or the bank wins some placements and loses others, the result is unresolved/Pareto, not hardware victory or failure. If only a routed microbenchmark wins, whole-model engineering value is unproved. Publication value additionally needs actual quality/resource evidence beyond prior local-affine/host-flash composition. No linear NLL extrapolation, invented speedup or universal impossibility follows from this note.

### Final fairness/contract closure

The r128 no-row-crossing layout is a convenient candidate shape, not a license to force wasteful page padding on the original Down width1408. The original baseline receives every supported packed/row-continuation capability with its own explicit command arithmetic; do not invent a zero-padding penalty solely because1408 does not divide16KiB. Likewise, a shared input's small byte count does not establish retention across commands or an arbitrary K-descriptor batch: every required group repost/context change remains in tau until the interface proves otherwise. Wire inputs may differ from128 packed bytes; the decoder/latch source audit does not settle that ABI.

This source-only review establishes the conditional inequalities and missing evidence, not an achieved hardware point. No additional rank, parameter grid, experiment or claimed hardware-clock trial is introduced. Final receipt follows the tool UTC timestamp; Root can use the region as protocol/resource scope while keeping implementation/quality releases separate.
