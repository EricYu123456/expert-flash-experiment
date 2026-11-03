# Same-round Stage1 control review — Root

This is a conceptual review after frozen Stage0 FIT/VALID completed, while the unchanged Stage0 SCORE process runs. It authorizes no implementation or new experiment. TEST remains unopened. The original Round2 thought gate is unchanged; these are stronger controls for the same candidate.

## Scope and corrections to the Gemini Stage1 note

Gemini4 Argon High completed the control-design block from 2026-11-03 02:01:34.172089 to 02:12:52.489964 UTC, exit0. Its statement that SCORE is sealed reflects the prompt's start-time context. Root separately authorized the existing frozen Stage0 SCORE at02:08:10; executor started02:08:30. Gemini received only FIT/VALID aggregate ranges and was forbidden candidate-data access. Its note is preserved as authored, with these qualifications:

- The common-output restricted-ridge objective and the unpenalized error of projected fixed ridge predictions are **different objectives**. Sol's matrix `sum p N G^dagger N^T` is optimal for the stated shrink-to-pooled penalized fit. Gemini's symmetric `sum(Y^T W Yhat + Yhat^T W Y - Yhat^T W Yhat)` selects the best projection for the unpenalized FIT error of those fixed fitted predictions. Neither theorem implies optimal deployed W8/A8 error or full-model NLL.
- Gemini's converse “plug-in covariance equals target-exact matrix iff eta=lambda=0” is false. Zero targets are a simple counterexample; a perfectly fitted prior can also make the residual orthogonal despite a nonzero penalty. Unregularized OLS is a sufficient condition under the stated fit, not a necessary one.
- Orthonormal U does not force an arbitrarily ill-balanced latent distribution. Latent RMS balancing is a possible FIT-only lowering choice, not a guarantee of better error. A floor should use the maximum measured coordinate RMS, not assume coordinate1 has the largest RMS. Both the selected matrix objective and actual lowered arithmetic must be stated.
- The suggested host footprints are illustrative coefficient/metadata counts. They do not prove an inclusive8GB deployment fits before KV length, runtime workspace, caches and retained shared/fallback data are included. Count both8e9 and8GiB. Zero routed NAND reads does not mean zero host latency or energy, and cannot alone establish cost dominance.
- An oracle residual outside a fixed U with fixed biases lower-bounds only that frozen representation and its latent functions. It does not rule out another U, another bias policy, or a context-dependent output basis.
- Local/operator or pilot-mixture SSE dominance does not imply whole-model NLL/task/latency dominance. A necessary-condition failure must retain its bounded scope and unresolved learning-curve caveats. Publication value does not require strict dominance of every conceivable control on every metric; the question is a meaningful, well-supported quality/resource/latency frontier beyond the relevant strong controls.

## Additional strong control: expert-specific global input directions

The full-H pooled control has nearly the same VALID error as localC32/r128 in the supplied eight-operator summary. Local coefficients might reconstruct input directions discarded by the layer-wide PCA, rather than require a rotating output subspace. A global per-expert low-rank full-input map `U_e V_e (x-meanX_c)+b_e,c` is therefore a necessary control alongside conditional `U_e B_e,c P(x-meanX_c)+b_e,c`.

Use the same contextual free biases and FIT-only fitting/VALID selection. A compact first screen can compare predeclared q128 FP32 and q512 W8 coefficient representations, reuse the selected full-H pooled penalty, and avoid repeating the complete Stage0 grid. Neither rank is presumed to fit the quality target. All-host execution may retain a F32 latent; an A8 latent is required only by the chosen native flash INT8 path. Quantization, row scales, calibration and CPU conversion work are real costs.

## Resource fairness

The native Deep shared branch is numerically unchanged, but placement is not fixed. Both candidate and controls may host-pin it or put it in AiF under the same memory budget. Its roughly0.45decimalGB W8 coefficient footprint can change the best resource schedule. Host-routed maps may overlap flash-shared execution; flash-routed maps may overlap host-shared execution. Evaluate legal DAGs rather than choosing one placement to favor a method.

The actual machine is EPYC9354, with a very different LLC/NUMA configuration from AiF's i9-14900KS/DDR5 setup. AiF describes up to86.4GB/s host memory bandwidth; ECC6.4GB/s is not a host or PE throughput. Actual current-host walltime, target-host bandwidth/cache sensitivity, modeled native AiF schedules and silicon measurements remain distinct. The eight-operator quality pilot fits in the current server LLC and cannot stand in for an all-expert full-model cache workload.

The primary normal-C32/r128/unweighted120/unguarded VALID points all selected pooled eta0. Their large conditional gap is not explained by an extra iterated-ridge stage in exact OLS arithmetic. Retain one numerical/lowering closure check; alternate configurations with eta>0 require their own interpretation.

## 03:18 UTC closure and reference qualifications

Stage1 FIT/VALID completed with OS exit0 at03:10:50.989055 UTC. Root independently verified all2,365 dependency hashes (9,261,111,014 bytes),160 unique selected endpoints and560 comparisons; critic separately reconstructed the288 raw VALID lowerings and all gauge selections. All16 rebuilt local W8 maps/scales/biases match the original hashes; the pooled correction is numerically negligible and lowers identically. The separate SCORE_DEVELOPMENT release binds these files. Reuse of old SCORE remains exploratory; no TEST or new teacher query is authorized.

The earlier shared-placement paragraph needs a numerical qualification: source GGUF Q8_0 shared/dense functions and canonical row-W8/whole-A8 functions are distinct. A source-preserving shared branch remains host-pinned unless an explicit source-format AiF execution contract exists. A B_fullFFNcanonical reference can compare canonical shared host/AiF placements, but must first measure its representation loss. Its capacity and timing cannot be paired with source-preserving quality by assumption. The roughly0.45GB canonical shared coefficient count is therefore not the complete source-format shared/dense residency bill (approximately0.478GB source shared plus0.071GB source dense-layer weights, before auxiliaries).

OLMoE's original all-routed W8 FFN coefficients occupy6,442,450,944 bytes. Adding the current361,364,603-byte known non-FFN/scales/KV floor gives6,803,815,547 bytes, below both target budgets before unresolved workspace and longer KV. An all-host original W8 baseline is mandatory; model size alone does not establish a need for flash on this model. This is a capacity floor, not a complete deployment-fit or performance claim. The full local C32/r128 bank exceeds the host budget, while compact controls can fit a known floor; neither observation supplies measured latency.

Stage1 fitting initially exceeded the intended eight-core CPU budget because separate BLAS/OpenMP pools spun simultaneously. The executor restricted all process threads to eight distinct physical cores at03:05:54.931067 UTC, preserved the adjustment receipt, and completed without changing source or arithmetic. Development evaluation starts with eight-core affinity already applied. Offline fitting/evaluation walltime is never used as deployed inference timing.
