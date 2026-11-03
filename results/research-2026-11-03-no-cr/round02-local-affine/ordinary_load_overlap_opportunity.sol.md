# Ordinary map loading versus short-row in-flash compute

2026-11-03, same Round2; source/shape/algebra only. No new source implementation, run, simulator, benchmark or payload access. This does not change Stage3. The question is the incremental necessity of custom short-row PE execution for an already fitted local bank, independently of whether that approximation is useful.

## Compare the identical function first

Compare (O) ordinary-read map pages → DMA → exact CPU GEMV against (I) ordinary-read in-flash GEMV → result drain. Both use the same frozen map/scale/bias bytes, selector/P/A8, support/fallback, final live alpha and output-add order. CPU exact integer dots and proposed native I32 accumulation/host rescaling must be proven bit-compatible; an unspecified native output format does not establish this. If compatibility holds, both trajectories, quality and routing agree, so any difference isolates implementation/resource placement.

Give both the identical-map host-cache option and the same inclusive8e9/8GiB budgets. A fully cached cohort is already a CPU path in both methods; it creates no short-row PE advantage. For each real token use its **joint** selected expert/leaf/cache state and command placement. Do not multiply independent per-expert hit probabilities, assume all misses, or force O to discard reusable maps. For an implementation-level paired comparison, fix the same miss set and bill different scratch requirements. For best-system comparisons, allow each method its best legal cache/prefetch policy; any shared-branch bound must remain valid under that choice.

The H2048/r128 W8 map is262144B plus8192B row scales and8192B bias if those metadata are not already host-resident. The native raw-I32 result is8192B/map. A miss count n≤K therefore entails262144n weight bytes for O versus8192n output bytes for I, before framing, scale/bias, inputs and fallback. The weight/output payload ratio is32. Metadata residence and fetch traffic must be stated explicitly for both.

For the all-miss **shape**, not an observed routing/cache outcome:

| Cohort | O weights | I raw results | O PCIe payload floor | I PCIe payload floor |
|---|---:|---:|---:|---:|
| K6 |1572864B|49152B|196.608us|6.144us|
| K8 |2097152B|65536B|262.144us|8.192us|

These use the AiF **modeled** PCIe8GB/s, not an NVMe measurement. Eight modeled2GB/s channels provide a16GB/s aggregate ceiling only with balanced accessible placement and legal commands; PCIe8GB/s can still bottleneck. The16-chip/four-plane geometry gives a two-wave ordinary-LSB sensing lower bound of56us for the stated full-miss cohort layout. It is not valid to add this automatically to PCIe time or declare either `max(56us,payload)` an achievable elapsed schedule. Sense/cache/drain overlap, channel arbitration and page-register lifetime need a contract. No CR is used.

## Dependencies and contention determine what can hide

At x-ready, Deep's source shared function consumes full x. Its source-host GU → source SwiGLU/input conversion → Down dependency remains. It does not consume the local P128 input, so its input quantization cannot automatically be reused. The ordinary loader can request addressed maps after IDs/leaf are ready and may fetch while P/A8 is still running; CPU dot rows need both their weights and qv. Streaming row/page DMA into CPU compute is a legal strong O design if its buffers, synchronization, cache effects and exact arithmetic are paid. It need not wait for every complete map before computing any row.

I needs its input/context posts, weight access, PE service, FIFO and drain. Whether sense can start before qv is ready or the input can survive command changes is unknown. Neither a free K-input PB context nor an artificial global wait-for-all-maps is allowed. Hits, ordinary misses, native misses and canonical fallbacks can be mixed in a legal schedule.

Deep's approximately18MB of source-Q8_0 shared weights per layer is a known byte footprint, **not its service time**. If those bytes really miss LLC and are read from DRAM during this interval,86.4GB/s gives only a conditional transfer floor around208us. That bandwidth is an i9-target ceiling, not EPYC throughput. Cached weights, prefetch before x-ready, source kernel conversion/compute and nonlinear dependencies change the interval. A measured standalone shared wall time is not automatically a lower bound on every concurrent implementation.

O and shared work contend for CPU threads, caches and DRAM, and DMA writes also consume host-memory service. To claim complete hiding, exhibit a schedule with at most the fixed host cores, actual bandwidth/queues, resident memory and scratch. Account P/selector, hit GEMVs, metadata, alpha/merge, other unchanged experts and non-FFN work. A core partition or persistent task pool can make a schedule feasible, but eight-core standalone timings for shared and GEMV cannot both be granted their own eight cores at once. Cumulative work/bandwidth inequalities are necessary; dependency releases and task scheduling must also be feasible.

Thus `max(shared_standalone,read_standalone+GEMV_standalone)` is not an established layer time. Nor may the three CPU wall times simply be added. Use actual node completion under one resource schedule, with per-resource busy intervals and a real final join. In-flash work can win partly by freeing CPU/DRAM for shared; ordinary loading can win partly by hiding otherwise expensive map service behind that same shared work. Both possibilities are consistent with the shape.

## A usable upper bound on incremental PE benefit

Let `T_O*` and `T_I*` denote the best legal completion times of the **same output join**, each allowed identical-function caching and its legal schedule. Let `U_O` be an attainable upper time established by one complete O schedule. Let `L_common` be a valid lower bound for every I schedule from its unchanged common branch, including Deep source shared when applicable. Then

`max(0,T_O* - T_I*) <= max(0,U_O - L_common)`.

For `L_common>0`, a corresponding acceleration bound is `T_O*/T_I* <= U_O/L_common`. This compares a **feasible upper bound** with a lower bound, not two lower bounds. A short-row PE giving zero routed service cannot remove that unchanged common branch.

To conclude that PE can save at most epsilon, one must establish `U_O <= L_common + epsilon`. Showing that ordinary routed work completes by a measured shared endpoint only proves that particular schedule has no exposed routed tail. It does not prove the endpoint is optimal: I might let shared finish earlier by freeing CPU/memory. The residual bound includes the gap between attained common completion and its defensible lower bound. In particular, `shared_standalone_walltime` cannot replace `L_common` without a proof.

Any DRAM-miss-based shared lower bound must remain valid under I's best legal cache/prefetch policy. Ordinary DMA scratch could change cache allocation; treating O's observed shared misses as mandatory for I would invalidate the upper-bound argument. If no informative common lower bound or attainable joint O schedule exists, the overlap hypothesis is unresolved rather than demonstrated.

An equivalent crossover view is to compare the exposed output-join tails and any common-branch interference under the two **joint schedules**. Saving a large standalone map GEMV/read time does little when both tails are hidden and their common branch completes almost identically. Conversely, if O's DMA/GEMV delays that branch or its own ready time exceeds the join, native offload can matter even with a long shared function. The native PE/FIFO/setup/input costs and output drain must fit within the savings; none is known from the ECC6.4GB/s figure.

## OLM and the evidence that would change the judgment

OLM has no source shared branch, so this particular shared-service floor vanishes. Other unchanged critical work, cached-hit compute or implementation dependencies can still overlap, but token t+1 at the same request generally cannot supply its next-layer work before the current FFN is ready. Do not invent spare overlap or another request in a B1 comparison. O's miss-weight transfer/CPU service can become exposed directly, making native output reduction more valuable; PE/FIFO and host-cache uncertainty still prevent a numerical speedup claim. Original canonical OLM may itself fit near the known6.8GB host floor and remains a legal, separately costed control.

Evidence strengthening in-flash necessity would be: same-function CPU/native numeric closure; complete inclusive capacity/cache traces; an attainable O implementation whose unavoidable miss/CPU/DRAM interference leaves a material exposed tail; and a feasible small-change I datapath/schedule that removes it under the same function and resources. Ordinary misses must be real consequences of the budget/trace, not chosen to handicap O. Whole-layer/model improvement must survive shared and non-FFN joins.

Evidence weakening the incremental PE need would be: a feasible equal-budget ordinary-load/cache/CPU schedule nearly saturating a valid common lower bound, with correspondingly small `U_O-L_common`; or a valid full-resource ordinary implementation no slower than I's optimistic whole-DAG bound. This would weaken custom PE execution, not automatically the local approximation itself. A fitted bank may still be valuable using conventional reads and host GEMV.

Currently there is no measured attainable target joint schedule or disclosed short-row PE/FIFO service here. PCIe and shared byte floors suggest an overlap question worth testing, not its answer. EPYC host-kernel measurements must remain labeled EPYC; target i9 bandwidth/cache and native command parameters require separate calibration/modeling. No new test or source work is authorized by this note.
