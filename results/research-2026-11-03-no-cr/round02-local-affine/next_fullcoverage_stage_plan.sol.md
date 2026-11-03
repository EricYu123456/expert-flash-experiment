# Same Round2: next complete-coverage stage

Prepared 2026-11-03 05:20:51–05:25:49 UTC. Source/theory planning only. No candidate execution, new data/model/activation/target/evaluation payload access, or implementation edits. This note proposes a prospective release; it does not amend a running experiment.

## Recommendation: complete the four existing pilot layers first

Use a separately preregistered **4L-v1** version covering all64 routed experts at Deep layers1/13 and OLM layers0/8:256 operators total,248 newly fitted. Do not immediately expand to all42 MoE layers/2688 operators. Four-layer completion tests whether the successful8-operator screen survives natural selection of every expert at those layers, with roughly one tenth of the full teacher work and a label archive that fits the present disk. It also exposes routing/state feedback from complete layer replacement. It does **not** establish behavior at the other38 layers.

Freeze the passed8 maps, trees, P128, centers, quantization/gauges, guards and hashes byte-for-byte. For each of the248 new experts use the same existing TRAIN union for its layer, C32/r128/normal, unweighted fitting, pooled eta0, and local lambda=.01 under the existing normalized penalty definition. Preserve the existing lambda=.001 exception only in its already frozen OLM map; do not propagate it by inspecting quality. There is no new fit/grid, format selection, guard tuning, projection selection or refresh from PG validation. Save this exact fixed transfer rule as a new-version choice, rather than relabeling the original8-map version.

Use the existing layer TRAIN union:29629 Deep rows and29593 OLM rows, including the bound original FIT120 and PG TRAIN16 capture provenance. This deliberately retains the finite representation budget and old B/C/D pilot trajectories. Those rows are not observations from a new all64 student trajectory. Any later on-policy refresh would require a separately declared version; it is not hidden in this stage.

At each model's two pilot layers define B4L by replacing **every** selected routed expert with the declared canonical W8A8 function. C4L uses the frozen/new local maps at exactly those operators. Everything else remains the same source-GGUF function. Deep source shared/dense functions remain host-pinned; canonical W8 on AiF cannot silently implement their source block-scale functions. Both B4L and C4L run their own prefill, decode, live alpha, routes and KV. Record source-versus-B4L change separately from C4L-minus-B4L. Neither is an all-routed/all-layer reference yet.

## Fixed sequence and nonlinear control

1. Freeze version, numeric contracts, input/row manifests, fit rules, retention budget and evaluation identities. Run the small engineering preflight below after explicit release.
2. Query canonical final Y for248 new operators once, fit/lower their local maps, and perform copy/canonical/numeric controls. During the same teacher pass also accumulate the projected-GU control's sufficient statistics. Existing8 final-Y archives do not contain GU: obtain their GU statistics in a separately counted pass. Never overwrite their passed local maps.
3. Use the already exposed PG validation16 only as **development/regression** evidence for C4L and B4L. If complete-layer replacement has a clear, reproducible quality failure, the cheaper necessary screen has failed; do not pay for full42-layer fitting first. A borderline result, canonical-reference degradation or an implementation failure is not a scientific failure of the method.
4. If that necessary screen remains viable, complete exactly the single nonlinear control in `nonlinear_baseline_design.sol.md` for all256 operators: global FP32 projected Gate/Up slopes at r128, the same32 leaf GU/output biases, eta0 pooled within-leaf fit, actual host arithmetic, whole-I SiLU/A8 and original canonical W8 Down. Use the shared TRAIN data and the saved GU statistics; no GU-rank/penalty/format search. Fit the final output bias from actual FIT predictions and the retained canonical Y. Call this N4L.
5. Freeze all C4L/N4L artifacts before opening a **new** validation cohort. Compare both with B4L on that cohort. Only a surviving result justifies an all42-layer preregistered expansion and a complete whole-resource comparison in the same Round2.

The old PE6/G2 roles at8 experts do not determine which family to choose at248 unseen experts. Preserve them as earlier diagnostics. Do not manufacture a full-model mixed control from those8 role assignments or select roles using PG validation. The fixed projected-GU control is the next necessary nonlinear comparison; it can be built without a new broad G/PE grid. It is a reproducible strong control, not an optimum over nonlinear latent models.

The shared P and decoded input can be reused across selected experts; the nonlinear control's GU slopes and activations cannot. Retain original Down pages once, with aliasing rather than duplication. C4L/N4L must receive equal host/cache capacity and all-host placement options. A quality result alone does not rank their runtime.

## Fresh evaluation boundary

The original PG validation16 result remains valid evidence for its frozen8-map version. It is now development evidence for this newly fitted256-operator version; its old confidence bounds cannot be carried forward.

Before reading new text, select16 unused, disjoint PG validation books by a fixed hash order over the remaining eligible metadata, excluding all TRAIN books and the exposed16. Pin the exact identities, tokenizer/window/continuation recipe and hashes. Keep PG test32 and the original TEST100 sealed. If16 eligible unused books do not exist under the current manifest, stop this selection step and preregister another source; do not recycle the exposed cohort as fresh.

Use the existing fixed NLL recipe and ln(1.02) noninferiority margin without tuning. Prespecify the simultaneous family over the four C/N-versus-B contrasts across two models, with books as bootstrap clusters and the existing10000-replicate procedure. Report prefill/decode separately, including failure/zero-energy bookkeeping where applicable. This new cohort is one frozen validation event, not a repeated policy-selection loop. Passage authorizes further same-round expansion, not deployment success; failure identifies only this fixed4L-v1 family and budget. Investigate source-to-B4L degradation separately. No layer-count, exposure or local-MSE linear extrapolation gives all42-layer NLL.

## Work and storage arithmetic

All figures below are coefficient arithmetic, decimal GB unless explicitly labeled; they exclude archive headers and do not predict measured throughput.

| Item | Four-layer total | Consequence |
|---|---:|---|
| FP32 Y, Deep128 experts |31.07GB|Retain for all fits/calibration/audit.|
| FP32 Y, OLM128 experts |31.03GB|Total Y62.10GB; newly queried248 add about60.16GB.|
| Raw FP32 Gate+Up intermediates |73.75GB|Do not persist this additional full archive.|
| Local W8 C32/r128 maps |2.147GB|256 maps; original8 remain byte-identical.|
| Local FP32 biases and row scales |.134GB|Plus P/tree/guard and artifact overhead.|
| Canonical original W8 GU/Down for these256 operators |1.912GB|Down reused by N4L; source/model storage remains separately accounted.|
| N4L FP32 GU slopes and GU/output biases |.466GB|Plus common P/tree/metadata.|
| N4L original W8 Down |.638GB|Alias existing retained pages.|

Keeping Y and raw GU together would consume about135.85GB before maps, snapshots, workspaces or other jobs, so it is a poor use of the reported161GB free disk. Retain four-layer Y, but stream GU. Before allocation recheck actual free space and require a declared reserve, e.g.32GB, plus a maximum incremental persistent budget (approximately90GB for this stage). A space shortfall is an engineering blocker, not a method failure. Do not assume the filesystem is reserved exclusively for this task.

For each expert's two **global** projected-GU heads, save FP64 sums of targets per leaf and global cross-products `sum(v * [g,u]^T)`. Common input/leaf means and covariance already suffice to subtract `sum_c n_c vbar_c targetbar_c^T` and obtain the pooled within-leaf solve. This is below1GB for all256 experts, rather than73.75GB raw GU. Final GU intercepts can use saved target means and the actual host-prediction means recomputed from fixed X; no claim of real-arithmetic/FMA equivalence is required. Final Y bias uses retained Y and actual nonlinear/Down predictions. Freeze accumulation order and sufficient-statistic layout/hashes.

Fit local maps one expert at a time. A raw Y operator is about243MB; FP64 leaf input/output cross-products are about67MB and its temporary FP32 maps about34MB. Do not retain a second8.6GB full float-map bank merely for convenience. Reuse the layer's design, tree, eigen/covariance work across experts and across output heads. This avoids multiplying the old7972s Stage0 grid by the number of experts.

Canonical teacher arithmetic for256 operators is approximately56.6 trillion integer MACs:32.8T Deep and23.8T OLM, about54.8T newly needed beyond the existing8 final-Y operators. This is one fixed teacher pass, not256 copies of every previous hyperparameter sweep. Collecting GU during new final-Y queries adds no duplicate GU matrix multiplication; recomputing GU for the old8 is extra and must be logged. N4L's actual FIT Down calibration adds about18.9T integer MACs, plus approximately2.36T projected-GU host MACs, SiLU, quantization and IO. Those are offline fitting costs, not deployment work.

Budget elapsed time from a future measured preflight rate, `T ≈ teacher_MAC / effective_batch_rate + fit + IO + calibration`, not a scalar-loop estimate or the quality harness's wall time. For scale only,56.6T at10/50/100GMAC/s would take94/19/9 minutes for the teacher matrix operations alone. These are hypothetical rates, not measurements of the current8-core EPYC or AiF's target i9. Abort/continue estimates must include conversions and statistics, not DGEMM peak alone.

At42 layers and2688 operators, analogous raw Y is roughly650GB at these row counts, beyond the current disk budget. Moreover, the other38 layers have no bound TRAIN capture in this stage: all-layer expansion needs its own fixed capture/teacher release. It cannot invent their states from these four layers.

## Small preflight: correctness and budget only

Choose one previously unfitted expert per pilot layer by a fixed hash of `(model,layer,expert)` independent of targets, frequency rank and quality. Use a bounded, identity-pinned subset of existing TRAIN rows spanning old FIT and available B/C/D capture origins, at most256 rows per layer. Do not interpret a C32/r128 fit on this tiny subset as a quality test; some leaves will necessarily be unsupported.

The preflight checks batch versus declared B1 arithmetic, a small independent I64 integer-dot audit, full-I activation/quantization, row identities, canonical Y compatibility with a small existing-map audit, GU-statistic accumulation, write/hash/read integrity, worker memory and batch/IO cost. It produces no new hyperparameter or expert-selection decision. Store preflight row identities so its target chunks can be reused in the full query; separately count deliberate audit recomputation. A numeric/IPC/disk failure blocks execution until fixed and consumes no scientific failure.

## Auditable streaming when all-layer expansion is justified

Four-layer Y fits, so keep it. For a later separately released42-layer version, use one expert/layer at a time: query chunked Y and GU, fit **all already fixed families** and actual-prediction calibrations from that chunk's complete operator data, then delete ephemeral raw labels only after an audit receipt. The peak raw Y for one expert is about243MB, not650GB.

Retain immutable input/capture row provenance, canonical weight/scale source hashes, pinned source/build and floating-point environment, seeds, fixed chunk boundaries/dtypes/shapes, per-chunk hashes and a whole-operator hash, final sufficient statistics/model artifacts, and predetermined independent audit rows. Record a deletion receipt only after successful fitting/audit; preserve failed chunks for investigation. A hash alone is insufficient: the preserved input/weight/engine identities must permit deterministic regeneration and comparison. Never describe discarded targets as available for arbitrary later baseline tuning; any new family needing lost intermediates pays an explicit regeneration budget.

## What would decide continuation

A clear C4L-minus-B4L quality failure is a bounded negative for this frozen complete-pilot-layer version, subject to numeric integrity and the declared finite fit budget. C4L viability plus N4L matching quality is **not** automatically a local-bank failure: the control may have greater native Down traffic but lower host storage; complete equal-budget resource comparison decides the mechanism's necessity. Conversely, C4L viability by itself cannot establish that local32 output banks are needed.

For hardware value keep mandatory original-canonical all-host OLM (known approximate6.8GB floor), legal all-host/latent controls and local hot-cache options. Close inclusive8GB with source non-FFN/shared, KV, allocator/workspace and metadata. Source shared host work is not removed by routed local maps. AiF short-row ABI, PE rate and FIFO remain unknown; sensing waves are lower bounds. No teacher-query, IPC quality-harness or current-server elapsed time is a compressed AiF speedup. The potential contribution remains the joint layer-shared-input/conditional-map/active-geometry resource frontier, not host/flash splitting by itself.

## Addendum: source proof favors first-MoE64 before four-layer64

Source review started2026-11-03 05:27:11 UTC in response to Root's cheaper counterproposal. This addendum **supersedes the four-layer-first ordering above**. Recommend the new prospective **first64-v1**: all64 experts at Deep L1, with its original dense L0 unchanged, and all64 at OLM L0. Every later layer remains source-GGUF. Preserve the two previously passed maps at each retained early layer (four maps total), tree/P and numeric rules exactly; leave all original8-map artifact files unchanged. Not installing their later-layer maps in a newly declared version is a change of experimental coverage, not retuning those artifacts.

### The invariant and its assumptions

Let `j` be the earliest replaced MoE layer. For a common externally supplied GT token sequence, compare source, Bfirst64, Cfirst64 and the prospective controls with the same token positions, sequence IDs, masks, context shifts, cache precision, graph/batching, adapters and deterministic kernels. All layers below j are unchanged. At layer j, its attention uses only its own previous attention KV and current input from layer j−1; those KV values were computed **before** its FFN. No later-layer KV is an input to this layer. Induction over token positions and unchanged lower layers therefore gives identical attention inputs/KV through j, identical `ffn_norm` x at j, and identical router outputs/selected IDs/final alpha. The replacement first changes the layer-j FFN output, which flows only into higher layers. Those higher-layer hidden/KV values can differ freely without invalidating this invariant.

This is a DAG/function argument, not proof that different batch shapes/backends/fusion choices reproduce floating-point bits. Hold the execution choices fixed and preregister an actual bitwise gate. The claim also requires fixed GT tokens and no model-output-based early stopping. It does not extend past the first divergent sampled token in free generation.

Pinned source is `/home/nfs_client/ericyu/expert_flash_experiment/sources/llama.cpp-927dada6c`, commit927dada6c9143ba4c940b72004d2698fa5e4e930. Concrete inspected evidence:

* `src/models/olmoe.cpp:69–155`: layer input supplies attention norm/Q/K/V; `build_attn` precedes attention residual, `ffn_norm` and MoE. The MoE result is added to the residual and then assigned only as the next layer's `inpL`.
* `src/models/deepseek2.cpp:264–365`: MLA's compressed KV and rotary key come from `wkv_a_mqa * attn_norm(inpL)`. Absorbed MLA concatenates normalized compressed KV with rotary key, using the compressed KV as V; the non-absorbed branch expands the same attention-side value. Both feed `build_attn` before `ffn_inp/ffn_norm` at373–377. Dense-leading logic at379 and routed/shared branches at389–417 do not write KV. The next-layer assignment is423. No recurrent FFN-state or later-layer input feeds either attention path.
* `src/llama-graph.cpp:2217–2218`: ordinary attention copies current attention K/V into the cache at layer `il` before reading that layer's cache. MLA's `build_attn(llm_graph_input_attn_k*)` at2283 copies only its combined K at2309 and views V from that cache; it does not cache FFN output.
* `src/llama-kv-cache.cpp:215–248,1145–1232`: caches have per-layer tensors and an explicit `map_layer_ids`; read/copy operations use the same supplied layer index. `src/llama-model.cpp:2002–2048` introduces optional cross-layer reuse only for Gemma3n/Gemma4, not these Deep/OLM graphs. The standard non-SWA constructor also passes no reuse callback. Model metadata/graph identity still belongs in the run receipt; do not extrapolate to an uninspected architecture or modified cache policy.
* `src/llama-graph.cpp:1382–1503`: router logits, probabilities, top-k, optional normalization and final scale depend on the current x and fixed parameters. These Deep/OLM calls pass no external `probs_in`. There is no online load-balancing state. Importantly, debug tensor `ffn_moe_weights` at1472 can precede normalization/scaling. Compare the **actual final weights operand** used by the expert-output multiply, not merely that earlier debug node.

Deep L0 itself also remains invariant at every GT position, including its attention KV: its inputs come from fixed embeddings and its own unchanged cache, not Deep L1 output. MLA compression is not a cross-layer feedback exception. OLM starts directly at L0. The inspected graphs have no exception to the induction.

### Required runtime gate and what existing TRAIN rows mean

For each model, token position and batch, compare owned F32 `ffn_norm` bytes, ordered I32 top-k IDs and final live F32 alpha across independent source/Bfirst64/Cfirst64 contexts. Include prefill and sequential GT continuation, exact byte lengths/strides and source/copy controls. Also hash the earliest attention cache's written values where the source hook can expose them safely. Never share whole KV between arms. Downstream KV is expected to diverge; do not demand it match or patch it from source. A failed early-layer invariant is a source/numeric/hook investigation, not permission to ignore differences or start a new training refresh automatically.

This eliminates **intervention-induced early-layer state shift conditional on identical token prefixes**. Existing TRAIN rows that really use the same GT prefixes are consequently exact early-layer source/Bfirst64/Cfirst64 inputs, so this new version need not pay another on-policy refresh merely because64 experts are installed there. However, the union also contains B/C/D-generated continuations from the older refresh: their distinct token histories remain distinct samples. Their early-layer x is invariant when those same historical tokens are replayed, but this does not turn them into natural GT text or prove a matched free-generation distribution. Preserve origin/phase provenance and do not rename the entire union "GT-only". Domain coverage and finite-fit limitations remain.

### Smaller fixed work and evaluation scope

Fit124 new operators, retaining four existing early-layer maps, with the fixed transfer rule above: eta0/lambda.01/C32r128normal and no new grid. Raw Y across the128 operators is about31.05GB, newly needed124 about30.08GB. Canonical teacher work is approximately28.3T total/27.4T new integer MACs. Local W8 maps occupy1.074GB across both models, plus approximately.067GB biases/scales. GU-statistic accumulation and subsequent nonlinear calibration have half the four-layer costs above. Do not fit later-layer maps yet.

Use Bfirst64 as the matched canonical reference and name this a **complete earliest-layer routed coverage** screen, not complete-model routed coverage. The exposed PG validation16 remains development for this version, and the fresh-version/fresh-validation rule above remains necessary. Existing passed8-map validation is not relabeled as first64 validation. An earliest-layer success cannot imply all42-layer quality; failure concerns this frozen version, not arbitrary retraining or later-layer-only deployment.

### Freeze the resource/necessity window before spending this work

First64 has a crucial capacity limitation: one model's complete early-layer local bank is only about.537GB W8 maps plus.034GB biases/scales and common P/tree. It plausibly fits DRAM with the rest of a legal host/flash placement. Thus this stage is a necessary **quality** test and a way to obtain a complete active expert cohort; it does not establish that the bank must execute on NAND. Give the exact same bank an all-host optimized-kernel path, alongside original-host and fixed global/PE/nonlinear controls.

Before the expensive teacher release, freeze the following source-level feasible-window calculation, with no measured speed claims:

* For a selected cohort, a contiguous row-major host implementation of local W8 maps can reuse the common latent input, use the actual frozen integer-dot/scaling path, and combine outputs in the declared order. No Python/tree/IPC time belongs in this deployment estimate. Its coefficient stream is `KHr`, its products `KHr`, plus common P/selection, scale/bias/alpha and output work. Allow ordinary prepacking/SIMD kernels within the fixed inclusive resident budget.
* Fixed q512 host PE requires approximately `Kq(H+r)` W8 coefficient bytes/products; full-input G requires `2KHq`. A float host latent has no compulsory second A8 quantizer. Its actual W8-to-float conversion/FMA or exact integer-dot paths must match the declared baseline function. Predequantizing coefficients can be legal only after billing the larger resident bytes; this is especially plausible for a single early layer, even when it is not for the complete model.
* At H2048/r128, K6/K8 local selected coefficients are1.573/2.097MB; PE q512 is6.685/8.913MB and G q512 is12.583/16.777MB, before scales/biases. Common P reads about1.049MB when not cached. Divide cache-miss bytes by an actual sustained target `B_cpu`, not automatically86.4GB/s. Divide actual operations by a sustained suitable `P_cpu`, including conversion and nonlinear overhead. Treat `max(bytes/B_cpu,ops/P_cpu)` as a roofline **lower bound** only; a source-constructible blocked algorithm establishes implementability, not its measured upper time.
* For the local flash path, use a legal resource DAG `D_map(R_PE,R_E,FIFO,setup,drain,cache)`, not56us alone. Ordinary all-hit r128 sensing has the two-wave56us lower bound under the stated cohort geometry; PE/ECC/FIFO/input/output can add service. Shared source-host execution, attention/KV, CPU competition and joins must appear in both whole-DAGs. In the isolated matched-P comparison, the crossover is `D_map < V_factor − V_local_host`, with legal placements and cache fractions included. If its right side is uncertain, record an open interval rather than reject the candidate.
* An actual feasible, matched-quality control **upper** service bound below the candidate's optimistic whole-DAG lower bound could refute that resource profile. Two lower bounds cannot. Conversely, a candidate lower bound smaller than a control lower bound does not prove a speedup. PE/FIFO uncertainty permits a conditional break-even region and clearly costed ABI requirements; it neither proves "minor hardware" nor justifies an automatic no-go.

Freeze one fixed global-control and the single projected-GU nonlinear-control recipe and their numerical/capacity accounting now. Source-only review may determine their legal optimized-kernel operation counts and layouts. Their quality must subsequently be evaluated at adequate coverage; q512 necessity, CPU throughput and host/cache domination cannot be inferred from bits or MSE. If no plausible equal-budget resource crossover remains after honest, attainable control calibration, stop before all42-layer expansion. If the window remains unresolved, first64 quality is still informative without pretending that the hardware hypothesis has been verified.

Addendum source-review receipt:2026-11-03 05:27:11–05:30:50 UTC. Read pinned model/graph/cache source and existing theory notes only; no experiment, candidate execution, payload inspection or implementation changes.
