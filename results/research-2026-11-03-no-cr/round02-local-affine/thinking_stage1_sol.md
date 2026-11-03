# Same-Round2 Stage1 necessity review — Sol xhigh

Resumed 2026-11-03 02:03:07 UTC. Theory/source-only; no implementation, model, prompt, activation, target, SCORE or TEST bodies accessed. This is the same candidate after its completed 30-minute thought gate, not another round or another claimed clock. Root supplied the FIT/VALID summary: local C32/r128/normal/unweighted/120/guard100 has output SSE/energy .0147–.1667 across eight operators, versus pooled .1408–.5407 and full-H global .0191–.1749; 80→120 curves still improve and FP32-to-quantized gap is around1e-4. These are not new measurements by this agent.

Read `positive_control_derivation.md` and Terra's latest source dossier. Independently reopened [MoLAE §4.2, eq4](https://arxiv.org/html/2503.23100v2), which explicitly has group-shared output lifts and expert-specific latent transformations; it excludes native shared experts from its analysis. Also checked [Swift-SVD's official ICML2026 record](https://proceedings.mlr.press/v306/qi26k.html), which describes output-activation-covariance/eigendecomposition fitting. Neither source supplies AiF latency or our checkpoint quality. Root/Terra's source-only EBC notebook distinction remains; no claim to have read its inaccessible paper.

## Precise hypothesis to test

The observed VALID gap identifies a useful conditional operator at this calibration budget. It does **not** yet identify whether C flash-resident output matrices are necessary. Three alternatives must be separated: an equally capable global two-stage estimator; common output factors with conditional small host transforms; and group-shared factors that aggregate live routed latents before one large lift.

Full-H global has potentially comparable operator error but sixteen times as many map coefficients/MACs as r128. That factor alone does not establish latency: identity input coordinates remove P, host cache/residency may change the path, page groups/cohort occupancy matter, and unchanged native shared/fallback branches can retain the FFN barrier. It is a mandatory real cost point, not a baseline to dismiss by FLOPs.

## Exact reduced-rank control that preserves shrink-to-pooled

Use column samples within expert/leaf j, weighted and centered so each leaf has a free output bias. Normalize the declared weights consistently. Let S_j=Z_j W_j Z_j^T, C_j=Y_j W_j Z_j^T, pooled prior A0_j, and beta_j be the frozen local shrink penalty. Restrict each slope to A_j=U B_j with U^T U=I_q, and minimize

`sum_j p_j [ ||Y_j-U B_j Z_j||^2_W + beta_j ||U B_j-A0_j||^2_F ]`.

For fixed U, define G_j=S_j+beta_j I and N_j=C_j+beta_j A0_j. Then B_j=U^T N_j G_j^dagger. Profiling B gives an objective constant independent of U minus

`tr(U^T M U), M=sum_j p_j N_j G_j^dagger N_j^T`.

Thus the top-q eigenvectors of M are globally optimal **for this real-valued, separable, centered, fixed-penalty objective**. This is stronger than coefficient SVD and directly uses differing latent input covariances. Singular/zero-trace rules are the same frozen pseudoinverse/prior policy. Beta infinity needs an explicit limit: for per-expert q>=r, the common prior's column space can be retained; do not numerically form infinity.

Crucially, the penalty is `||U B-A0||²`, matching restriction of the original full-slope prior. Penalizing `||B-U^T A0||²` instead has a U-dependent missing-subspace constant and does not permit blindly reusing the same profiled M. Additional W8/A8, nonorthogonal lowered U, host finite arithmetic, grouped summed-output objectives and VALID selection invalidate any claim that this real optimum is already the deployed optimum.

## Compact experiment recommendation (before new held-out evaluation)

Freeze the existing eight operators and primary C32/r128/normal/unweighted/120/guard100 point. Keep C1/4/16 and alternate formats as existing context; do not rerun a giant grid. Reuse FIT means, shared input designs, covariances/eigendecompositions and existing teacher outputs.

1. Give the pooled same-leaf slope the same second correction stage toward its frozen pooled eta prior, using the existing lambda grid and the identical deployed-bias calibration. This tests the estimator-capacity confound without a new input representation or new teacher queries. If joint eta/lambda search is added, grant matching capability to the local fit and predeclare the finite set; do not expand adaptively after SCORE.
2. Per-expert common U at q128 and q512, using the exact shrink-preserving M above and the current frozen local beta first. These are two predeclared rank points, not a fresh rank search. Refit conditional B/bias on the same FIT data. A beta0 diagnostic is permissible only if prespecified equally; it is not necessary to turn the first screen into an endless grid.
3. Evaluate both actual native-flash lowering and a legal all-host high-precision path. A flash U requires the real per-expert latent A8, different input posts/group residency and ordered row-scale arithmetic. An all-host FP32 U/B path needs no compulsory native A8 latent quantizer: measure its CPU/bytes/capacity and preserve FP32 latent until the host output. Do not impose a quantizer solely to weaken the CPU control. At q=H with U=I, eliminate the identity lift and extra quantizer entirely.
4. Only expand to group/layer U with honest data scope. A common U built from two queried experts/layer establishes a two-expert pilot, not all64 expert fidelity or all-route quality. Actual grouped quality needs enough expert targets to evaluate the complete naturally selected mixture; a group speedup cannot be inferred by pretending unqueried experts fit. Per-expert U can already decide much of the bank-necessity question with existing data.

No SCORE/TEST is needed to build/select these FIT/VALID controls. Before opening fresh SCORE, freeze winning finite control choices, arithmetic, storage placements and paired comparisons. Existing VALID is development selection and cannot itself supply unbiased final sign claims.

## 02:10 UTC checkpoint — precision, groups and whole-budget classification

### Two naturally matched byte-volume endpoints

For per-expert U, fixed C32/r128, the pair q128 with U/B FP32 and q512 with U/B W8 has the same leading coefficient byte sizes: U_e is1,048,576B and B_e,c is65,536B in both. Deep's complete coefficients sum to about5.234GB; add .436GB biases, P/tree, any scales, non-FFN/router, native shared weights, KV, workspace and allocator overhead. Parent estimates around6.6GB before several of these moving quantities; actual inclusive8e9/8GiB feasibility must be measured, not inferred from coefficient equality. OLM is smaller but not automatically free.

These endpoints differ in rank, arithmetic, scale metadata and quantization, not storage headline. Deep selected U/B logical coefficient traffic is K6×26×(1,048,576+65,536)≈173.8MB/token, plus P27.3MB, biases and other arrays, absent CPU-cache reuse. All-host latency is therefore real. q512 has four times the scalar map/latent MAC count of q128; VNNI/float throughput and actual locality can change execution time. An all-host q128-FP32 path should not inherit a second native A8 quantizer. A flash-U path must pay per-expert latentquant/inputposts; grouped aggregation can reduce these only when one U really is shared.

At q=H and U=I, remove the identity lift and extra latent quantization. A baseline should never be made worse by fictitious identity work. Full-H global direct maps still have their actual dense coefficients/CPU work/flash pages unless a further valid factorization changes them.

### Honest grouped pilot with only two experts per layer

A possible compact arithmetic pilot shares U between the two already queried experts at each layer. Use actual natural route coefficients, with alpha=0 for an unselected member; compute pair residual `alpha_i(y_i-b_i,c)+alpha_j(y_j-b_j,c)` and compare one grouped latent sum followed by one lowered U against the same selected local maps. Other experts remain identical originals in both compared constructions, so absolute output-error differences from this pair are meaningful, but a pair-energy denominator is not a full-layer energy denominator. Single-selected, co-selected and neither-selected cases must be separated. Sparse co-selection is inconclusive, not a reason to invent alpha observations.

A two-expert shared-U fit does not establish quality for a64-expert layer or arbitrary grouping, and estimated all-model bytes/waves are only extrapolated resource examples until all members are fitted/evaluated. Per-expert U avoids this extrapolation and should be the initial decisive control. Full group/layer U requires honest extra target scope or an independently specified stronger fitting source; fitting to two experts then deploying on64 is not valid.

With a genuine common U_g, the real arithmetic identity is `sum_e alpha_e U_g h_e = U_g(sum_e alpha_e h_e)`. Native dynamic A8/F32 changes the paths: quantize after aggregate once per group if that is the declared implementation, retain livealpha/bias order, and evaluate actual outputs. Do not aggregate independently scaled integer latents as if their scales agreed. Native shared experts are unchanged unless explicitly included and independently validated.

### Compact stopping interpretation

- If matched pooled two-stage estimators erase the gap, it is an estimator-capacity/global-surrogate signal; the conditional output-bank explanation fails for this frozen calibration family. This does not automatically prove whole-model quality or latency equivalence.
- If per-expert common-U controls retain the local operator quality and are no worse under complete host/flash/time budgets, the claimed necessity of C full output maps is boundedly falsified for the frozen family. A useful simpler engineering variant may remain; no publication claim follows.
- If only an infeasible FP32 factor bank matches, or factors read less flash but cost materially more CPU/bytes/inputposts, there is no dominance: it is a Pareto question needing actual timing/quality. A rank128 failure alone does not kill factor controls; the prespecified512 point supplies one compact higher-capacity check.
- VALID selects the finite controls. Fresh SCORE comparisons, prompt-clustered uncertainty and frozen source/build are required before a scientific sign decision. Improving80→120 learning curves, low decode coverage, unconverged fits or numeric faults remain explicit inconclusive limitations. A fixed-budget conclusion must not be relabeled an asymptotic impossibility.
- Any credible non-dominated bank point continues this SAME round to complete-mixture, all-layer free-running quality and the full nonlinear/W4/lower-K/cache controls. Local SSE/energy .01–.17 alone cannot establish acceptable NLL or generation. No next-round clock starts here.

## Offline expansion need not repeat Stage0's giant grid

Root reports7972s for the eight-operator original grid; naively multiplying that by208 is the wrong engineering plan. Shared unweighted FIT design/tree/P/leaf covariances/eigenvectors are layer artifacts reusable across experts; targets and Cxy are expert-specific and can be streamed once. Reuse analytic covariance factors across the prespecified ranks/penalties rather than redo decompositions per expert/configuration. First finish the compact strong-control pilot; if it survives, transfer a finite set of VALID-selected rank/C/format/hyperparameters to a frozen all-layer experiment, with a declared limited calibration retune rule and appropriate held-out check. No surrogate claim justifies fitting hundreds of hyperparameter grids per expert. Canonical teacher querying remains aboutN*3HI per expert; batching is offline acceleration, never B1 latency evidence.

## 02:14 UTC checkpoint — primary OLS closure and real machine limits

Root supplies a useful additional selection fact: all eight primary pooled eta values are0; seven local lambdas are.01 and one OLM L0e17 lambda is.001. In real arithmetic, pooled OLS satisfies C−A0 S=0, including a true singular covariance: a null-S direction is orthogonal to the design and therefore also annihilates C. Thus the two-stage global correction is zero on this primary family. A single matched-correction/arithmetic audit suffices; there is no reason to rerun a giant eta×lambda grid solely to chase that confound. Float/pseudoinverse cutoff and post-fit lowering remain audited. Alternate C/formats with eta>0 retain their separate estimator issue.

The RRR work can reuse existing float local solutions A*_c=N_c G_c^dagger: M=sum p_c A*_c G_c A*_c^T, and B_c=U^T A*_c. Compute top512 once, prefix128 for the nested rank point. Shared unweighted S/G eigendecompositions are reused by layer/leaf/beta. Refit bias and evaluate true deployed precision, but do not repeat every rank/penalty's sufficient-statistic construction.

### Target versus actual CPU

Independently read paper/aif.txt875–905: AiF compares against Intel Core i9-14900KS, DDR5 up to86.4GB/s, and8GB host capacity for AiF/Memory+SSD. Root reports our available execution server is EPYC9354 with512MiB aggregate L3, unlike the paper's CPU/cache. Actual server8-thread walltimes, ISA, NUMA placement, cold/warm state and working-set cache hits must remain labeled actual-machine results. They cannot be ported as measured target-AiF host performance. Use the paper's host bandwidth ceiling as an explicit modeled lower-bound/sensitivity input, with realistic lower sustained rates and explicit cache-capacity assumptions; ECC throughput is irrelevant to CPU DRAM rate.

The roughly174MB selected U/B working set per Deep token could fit far more comfortably in server aggregate L3 than in the paper's host cache. Warm repeated trajectories must not give the all-host path an unearned target-cache hit rate. Conversely, forcing cache misses on every load would also weaken it. Both methods get equal cache/resource models and actual cold/warm measurements, separately reported.

AiF's original statement that dummy vector contents do not affect its performance evaluation cannot be transplanted to this MoE study: real outputs affect routing, group conflicts, fallback and cache accesses. Original and approximate methods require their own real trajectories and associated DAGs.

### Shared-branch resource overlap can reverse standalone routed timing

For either matched-volume per-expert endpoint, U/B plus P/bias is approximately7.8MB per Deep layer. At86.4GB/s, its transfer-only ideal floor is around90us. The local bank's P alone is around12us at that ceiling, plus a hypothetical legal two-wave routed schedule56us, around68us before all other costs. This does **not** establish localbank latency superiority.

Deep's unchanged W8 native shared branch has about17.3MB coefficients and a separate GU→Down dependency, roughly17 aggregate four-plane waves in a raw coefficient lower-bound model (~476us), with finite chip/command resources. Host routed compute may run while shared executes on AiF, whereas additional routed flash waves compete for those same16chips. It is therefore possible for a nominally slower all-host routed control to make the entire FFN faster. The exact result requires a legal resource DAG, layouts, scale/output payload, actual CPU throughput and overlap constraints. OLM without a shared branch can reverse that outcome. This cross-resource scheduling issue must be tested; a standalone routed microbenchmark or sixteen-times-MAC slogan cannot decide it.

## Final protocol closure after parent review

Native shared functions stay numerically unchanged, but their **placement is free for both methods**. Deep's shared W8 coefficients across26layers are approximately.450GB plus scales; both bank and factor controls may keep them in host8GB or run them on AiF. Shared-host coefficient traffic alone is about17MB/layer (~197us at86.4GB/s as a transfer-only lower bound), which can overlap candidate flash routed work while all-host routed work competes for host bandwidth. This can reverse the previous shared-flash example. Take the best legal resource DAG for each family, accounting resident bytes, CPU cores/bandwidth, NAND chips/channels and actual overlap; do not fix shared placement to favor a narrative.

All-host W8 factors also need not quantize the intermediate latent to A8: a CPU path can convert U's packed/int8 coefficients and multiply a floating latent, preserving W8 coefficient storage/traffic while paying conversion/F32 compute. Only a native flash-U path requires the specified native input quantizer (unless a justified alternative hardware contract exists). Prespecify these as execution paths of the same fitted factors, not a new fit/rank search. Their actual arithmetic and FIT-bias calibration must be measured separately.

Keep the two identity cases distinct: full-input-rank global r=H can use P=I but still has its nontrivial H×H slope; full-output-rank factor q=H can use U=I and eliminate the output lift, but then conditional B_c is just the full H×r local bank and its retained storage remains. Neither identity transformation magically deletes the actual nontrivial matrix.

The eight-op W8 localbank is only about64MB (FP32 about256MB), easily cache-resident on the available server. It cannot be timed warm and presented as a full14GB bank on the target CPU. Timing calibration needs explicitly controlled shape-only whole-footprint pressure, cold/warm conditions or a transparent modeled-cache sensitivity, without claiming synthetic coefficient timing establishes model quality. Whole-model inclusive bytes remain the residency criterion; tiny pilot data residency does not imply all-model host fit.

### Recommended freeze in one paragraph

Keep the same eight VALID-selected primary points; record the eta0 global-OLS correction closure instead of a new hyperparameter grid. Build one exact shrink-preserving per-expert RRR basis to max q512, with nested q128; evaluate q128 FP32 all-host and q512 W8 all-host as matched-leading-byte main controls. Include actual flash-U/A8 and W8q128 paths only as predeclared arithmetic/placement diagnostics necessary for the bank comparison, not another search. Bias calibration and input information remain matched. Optionally validate grouped livealpha arithmetic on the already queried two-expert pairs, explicitly limited to those pairs; full64 grouped conclusions await real target scope. Audit actual inclusive8GB, target versus server cache/CPU, legal DAG with shared placement free, startup/load/export and mixed CPU/flash paths. Choose finite controls on VALID, freeze comparisons/source/build, then use unopened SCORE once for bounded signs; TEST remains reserved for full-quality expansion.

**Opinion:** continue this SAME Round2 with the compact strong-control pilot. The primary eta0 fact makes the conditional-slope signal more credible, but the all-host/shared-U alternatives are cheap enough and strong enough that full-bank necessity is still unresolved. Reject the bank explanation only with bounded fresh evidence of quality-and-cost dominance; mark tradeoffs, limited samples and improving curves inconclusive; any credible bank Pareto point requires all-layer true quality and complete strong controls. No result here establishes publication success, a valid second experimental failure, or permission to start another round.

## 02:20 UTC addendum — essential G-RRR identification control from Gemini

Read `gemini_stage1_control_review.md` and Root's corrections. It identifies a missing hypothesis: a single **expert-specific global full-input map** may achieve the localbank's quality even though the layer-shared input PCA128 pooled slope fails. The local leaves can approximate omitted global input directions on the routed data manifold; this requires neither rotating output bases nor genuinely necessary conditional slopes.

Therefore add exactly two G-RRR endpoints, q128 FP32 and q512 W8, to the same compact pilot. Use the existing full-H R2 FIT design/statistics and its frozen VALID eta; factor a single global slope into U_e V_e while retaining the same32 free biases/selector. Reuse the actual full-H decoded-A8 design to isolate expert-specific full-input information from new format choices; do not pretend it is identical to the low-rank shared-P design. Both U and V are fixed across leaves. For q128-FP32 or q512-W8, G's leading coefficients total `2 H q * bytes` (~3.49GB across Deep), cheaper retained storage than PE's `H q + C q r` (~5.23GB for the matched PE endpoints). Selected-token host traffic can nevertheless be **higher**, because V is a full-H input map rather than a selected small q×r B; residency and latency are distinct.

Do not silently reconstruct U V into a dense full-H matrix for timing. G's benefit is factor execution, and its CPU matrix/vector conversions, latent precision, bias and optional flash-U posts must be the measured path. A good G result would refute the outputbank explanation more strongly than PE alone; a failed fixed rank/precision point does not mathematically eliminate every expert-specific global factorization.

### Distinct exact basis objectives

Gemini derives a useful alternative to my shrink-penalized M. Let fixed ridge predictions be Yhat_j=Z_j A_j^T in row-sample notation. For orthonormal U, predicting Yhat_j U U^T gives unpenalized FIT teacher SSE equal to a constant minus trace(U^T M_target U), where

`M_target=sum_j [Y_j^T W_j Yhat_j + Yhat_j^T W_j Y_j - Yhat_j^T W_j Yhat_j]`.

Top-q eigenvectors optimize this objective **among projections of the fixed ridge estimator**, with B_j=U^T A_j; it is not the same objective as optimizing the explicit full-slope shrink penalty profiled earlier. M_target is a sensible primary because the reported quality criterion is teacher SSE. The shrink-penalty M remains a valid derivation/diagnostic; no need to cross every estimator/rank/precision into an endless grid. Freeze one as primary (or exactly both if Root deliberately budgets two), rather than switch on SCORE. If a general fixed estimator yields negative eigenvalues, a rank-at-most-q model need not include negative-improvement directions.

M_plug=sum Yhat^T W Yhat equals M_target under OLS normal-equation orthogonality, but the converse claim 'iff eta=lambda=0' is false (e.g. zero targets/perfect-prior special cases). Isotropy/orthonormality likewise does not guarantee latent RMS equality. Optional FIT RMS gauge balancing must use max_j rho_j in its floor, not assume rho1 is maximal; count its row-amax/quantization tradeoff. Given the original observed quantization gap~1e-4, no elaborate rotation/format search is justified before the two direct precision endpoints.

### Revised minimal freeze

Primary existing localbank; pooled eta0 closure; G-RRR and PE-RRR each at q128 all-host FP32 and q512 W8 with explicitly declared host-floating-latent/native-A8 execution paths. Reuse inputs/statistics/one top512 basis per fitted objective; no new full-H lambda/rank grid. Two-expert grouped arithmetic remains scope-limited optional-to-claim. Build/select on existing FIT/VALID, freeze source/build/comparisons, keep SCORE/TEST sealed until that gate. Any eventual claim of real bank necessity must defeat **both** G input-subspace and PE output-subspace explanations, then pass actual whole-quality/system controls.

## Final receipt / objective freeze

Root accepts M_target as the compact primary fixed-ridge-projection estimator; M_penalty stays theory/diagnostic, not a crossed experiment grid. There is an exact PSD refinement for the present matching-weight FIT linear estimator: in whitened sample coordinates, pooled hat R and block-local hat L are symmetric PSD contractions, H=R+L(I−R), and I−H=(I−L)(I−R). Thus H+H^T−H^T H=I−(I−H)^T(I−H) is PSD, and M_target=Y^T[H+H^T−H^T H]Y is PSD. Conditions matter: same global weighted metric with consistent leaf masses, centered FIT design/targets, pooled prior from that FIT data, and fixed penalties. This is not a guarantee for an externally trained prior, different route metrics, or the W8-rounded nonlinear fitting map. Clamp/diagnose only tiny negative roundoff under a declared rule; rank-at-most-q remains safe without depending on unverified assumptions.

Sol opinion remains **go for compact same-round strong controls, no success claim**. The required identification set is now G-RRR **and** PE-RRR at exactly the two rank/precision endpoints, a pooled eta0 closure audit, and honest CPU/shared-placement/cache/DAG evaluation. A few all-host matrices fitting a capacity equation do not establish latency dominance; a local-MSE gain does not establish whole-model quality or rotating output subspaces. Fresh frozen SCORE evidence can yield a bounded finite-family negative, a real tradeoff/insufficient-data inconclusive, or authorization for full-mixture/all-layer quality expansion in the same round. No new experiment, data access or later-round clock was initiated by this review.
