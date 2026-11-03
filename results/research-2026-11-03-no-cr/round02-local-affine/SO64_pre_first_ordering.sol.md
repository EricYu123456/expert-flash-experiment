# SO64_pre before NL64_GU: bounded prospective ordering

2026-11-03; same Round2, source/theory only. No new clock/failure, execution, payload inspection, quality opening or source change. In particular, the ongoing QUALITY_C64 output and new book/FIT/target bodies were not read. This note uses only the already frozen Stage4 recipe, its referenced SO8 note, pinned estimator/numeric source and Root-provided resource metadata. It releases no source/fit/quality/hardware phase.

**Recommendation, conditional on C64's necessary-quality review:** SO64_pre can precede NL64_GU. It addresses common-output sharing using existing final-Y archives, without first paying124 new GU-target passes. This reorders a deferred strong control; it neither changes its recipe nor removes the later nonlinear comparison if the local-bank claim survives.

## Frozen function and estimator

Use Stage4 protocol `deferred_controls_frozen_recipes_not_execution.SO64_pre` exactly: one layer-common q512 output basis per model's earliest MoE layer, mean over all64 experts with unit per-expert weight; same shared tree32/P128/normal/decoded-A8 design, identity gauge, W8 B/U and host FP32 latent. Preserve each C estimator's pooled eta/local lambda: eta0/lambda.01 for the124 new experts and the inherited old4 settings, including **OLM L0E17 lambda.001**. All old C artifacts remain unchanged. SO64 has newly common factors for all64; an old two-expert SO8 U or private old4 U cannot substitute for the64-expert basis.

In column notation, with fixed leaf covariance S_c, cross moment T_e,c and fixed unquantized slope A_e,c (H×r), reconstruct

`M_e = sum_c p_c (T_e,c A_e,c^T + A_e,c T_e,c^T - A_e,c S_c A_e,c^T)`;

`M_group = arithmetic_mean_e=0..63(M_e)`.

Use the referenced FP64 symmetric-EVD, positive relative1e-10 cutoff, deterministic sign/order, q512 zero padding and existing finite/diagnostic rules, once per model. Then `B_e,c = U^T A_e,c` before the existing F32 cast and canonical row-W8 lowering. Do not substitute a penalty-objective matrix, route/alpha/energy weighting, per-leaf U, truncated private U512, dequantized C maps or a new rank/gauge. Freeze expert/leaf accumulation order and library/source identities before fitting; an algebraically reordered mean or reassociated matrix product need not preserve floating-point bits.

### Recovering floating slopes is real work

The frozen `stage1-controls/freezes/rev003/...controls_core.py::fitted_slopes` (60–72) constructs FP64 cross moments from decoded-A8 FP64 v and F32 Y, and calls FP64 `old.solve`; **As are FP64**. `old.lower` casts As to F32 only immediately before row quantization. The SO matrix and B construction use the preceding FP64 As. A FP32 slope cache would change this recipe. W8 maps cannot invert the lost floating slopes.

Preferred prospective storage plan: reconstruct each expert **once** with the pinned estimator, atomically persist its FP64 As in a new SO-only scratch namespace with shape/dtype/order/source/Y/design/penalty hashes, accumulate its M_e into the common mean and release its other response temporaries. Persist no bank of per-expert H×H M_e. After one common EVD/model, stream those same hashed As back expert by expert to form/lower B and calibrate SO. One expert's32 slopes require67,108,864B (64MiB); all128 operators require **8,589,934,592B** on disk, not RAM. Raw FP64 shapes/bytes and actual hashes must be audited; no silent F32 conversion. Do not overwrite any C archive or canonical Y.

Alternative with identical mathematics, only if separately selected before implementation: discard As after mean-M construction and regenerate them after U is fixed. That saves the8GiB scratch but repeats the response/solve work and Y IO. Root reports about32.9GB current Stage4 use and129GB free; those metadata alone give no disk reason to require regeneration. They are not a phase-cap release or a guarantee about later actual free space. Keep the preferred one-reconstruction option as the prospective implementation target, not a quality-arm selection.

There were no archived original FP64-As bytes here against which to claim historical bit equality. Estimator/source/data identity plus reproducible reconstruction is the defined control. Optionally re-lowering/recalibrating reconstructed C and comparing existing C hashes provides an additional provenance check; it is **not** silently added as a new necessary-quality gate. Coefficient/scale comparisons alone do not verify the original floating values or biases.

## Explicit offline cost, without new canonical queries

Across64 experts/model at N_Deep29629 and N_OLM29593, total expert rows are3,790,208. Existing final-Y data for all128 operators are31,049,383,936B; the new124 portion was already produced under Stage4. SO produces **zero additional canonical Y or GU targets**, but rereads existing Y and performs the following work. Let H2048,r128,C32,q512. These are structural dense-product counts, not CPU time or AiF throughput:

| Work for the one-reconstruction plan, both models | Leading products/MACs |
| --- | ---: |
| Rebuild centered response moments, sum_e N_e rH | 993,580,285,952 FP64 |
| Actual pooled+32 local solves, up to128×33×3r²H | 425,201,762,304 FP64 |
| Existing literal target-matrix products,128×32×(3H²r+r²H) | 6,734,508,720,128 FP64 |
| B=U^T A,128×32×Hrq | 549,755,813,888 FP64 |
| Actual quantized B calibration predictions, sum_e N_e qr | 248,395,071,488 signed-I8 MAC |
| Actual U calibration lift, sum_e N_e Hq | 3,974,321,143,808 eight-lane FP32 FMA products |

Zero-trace/skipped solves reduce the stated solve count; scalar centering, means, row quantization, rescale/bias, common design/covariance construction if not retained, two H2048 EVDs and BLAS/library workspace/IO remain separately paid. The old `fixed_fit` wrapper also repeats solves to generate diagnostics; do not accidentally hide that extra cost if reused unchanged. Reusing its actual solve outputs for new reports is an implementation optimization to review, not permission to alter the estimator. The regeneration alternative repeats at least the response and actual-solve rows of this table (about1.419T leading FP64 products), plus Y reads. Optional complete-C bias replay adds about993,580,285,952 signed-I8 MAC and its actual rescale/FP64 residual-mean work; it is not a canonical-target query.

SO lowering is also mandatory: one common U row-quantization/model and64×32 B row-quantizations/model, with their scales and zero/support metadata. Calibrate each expert/leaf's output bias from **canonical Y minus the actual W8-B → FP32-latent → W8-U FMA8 → F32 Us-scaled prediction**, with alpha1, sequential FP64 residual accumulation and one final F32 cast, as in the frozen note. Do not project the old C bias or calibrate the ideal UB product. Keep row order within each leaf; do not average chunk biases. A leaf-at-a-time or proven stateful sequential accumulator can bound memory without persisting full output predictions. Changed batching must pass the frozen B1/batch arithmetic checks.

The resulting W8 factors/scales/biases total156,246,016B per full64 layer,312,492,032B across the two models, before P/tree/guards/container/alignment. Common H×H mean-M is32MiB/model, not64 expert copies; U is fitted once/model. Stream model/expert/leaf, with one expert's Y/As/B and bounded calibration buffers resident. The8GiB scratch plus final factors is roughly9GB incremental logical storage before temporary files/logs/manifests. Future release must check cumulative cap/reserve and cleanup policy explicitly; old Stage3's16GB allowance is not authorization. Retain Y and final SO artifacts; scratch deletion, if allowed, follows hashes/audit closure. Research-fit8e9 and reference-quality32e9 caps remain distinct from eventual inclusive8e9/8GiB deployment capacity.

## Primary runtime function and necessary numeric closure

For supported selected experts in original slot order, compute their actual B latent h_e, then

`t = ordered_F32_sum(F32(alpha_e*h_e))`,

`b = ordered_F32_sum(F32(alpha_e*b_e,c))`,

`z = F32(F32(Us*FMA8(Uq,t)) + b)`.

Use actual final graph alpha values as finite FP32 operands; do not require positivity or sum1, renormalize them, or divide by any alpha. The bias is weighted separately and is not replaced with an unweighted leaf lookup. No extra latent A8 quantizer is introduced. Scalar multiplication, each ordered sum, lift reduction, output scaling and bias addition retain their frozen order; alpha cannot be moved through FMA/scales by real-arithmetic distributivity.

The sole arm remains the weighted-slot premerge hook: write z to the first participating supported slot and +0 to the remaining participating slots, then preserve the graph's remaining slot-add order. Fallback experts keep their canonical function and weighted original slot. Source shared/dense/later layers remain their declared functions; each arm owns its KV/routes/alpha. This is a new rounded SO function, not bit-equivalent to independently lifted experts or to C. Do not implement it as Down/alpha division or apply alpha again after the hook.

Separately reviewed generated/source tests must bind U/B/scales/bias/support manifests, cover signed/nonunit/zero alpha and cancellation, confirm single-expert alpha1 calibration, exercise the actual premerge weighted-slot path, and compare frozen eight-lane dot/rescale/bias/order against an independent reference. Check B1/batch equality, typed frames/epoch/unsupported behavior, no-pilot/no-supported cases, nonfinite technical holds, archive round trips and source/copy controls. Only numerical equivalences within the declared SO function are bit gates; real-arithmetic SO versus C equality is not expected.

## Finite conditional phases and interpretation

1. **After C64 adjudication only:** freeze a separate source/generated-numeric and SO-fit amendment identifying the preferred FP64 scratch plan, phase resources/checkpoints and exact input identities. Existing Stage4 sources/C bytes stay immutable. Source/generated approval precedes any real SO data read. SO_FIT then uses the fixed TRAIN union and existing Y only, freezes factors/bias/support, closes costs/hashes and stops for review. Do not collect GU or query new states as part of SO_FIT.
2. **If the fit/numeric closure passes:** separately release only SO64_pre on the already exposed C64 validation16, same fixed512GT/256targets/ownKV/batching. The cohort is now **development for this secondary comparison**, not a fresh SO confirmation. Firstbook source/B and copy controls must bind to the historical frozen function before reusing B64/C64 per-book reference scores; if reference identity cannot be established, rerun the needed fixed references honestly. Do not open any additional cohort: the original16 remains prior development, while remaining18 validation and TEST remain sealed. No generation, new selection, rank/guard tuning, alternative merge arm or C refit is added.
3. Report fixed per-book SO−B and C−SO paired differences with the existing resampling/endpoint conventions frozen in the amendment; label SO−B a secondary application of the existing tolerance, and C−SO descriptive. Preserve all16 books, zero/insufficient/nonfinite handling and technical holds. **No new requirement that C significantly beat SO is introduced.** Acceptable secondary SO quality motivates the next actual host/cache/cost comparison; unacceptable or uncertain SO quality only limits this fixed global-U512 family. It does not by itself reject all shared-output/nonlinear families or consume a failure. Wide uncertainty, numerical/resource failure or missing coverage remains inconclusive.

SO-first is useful because for m active supported experts its leading runtime work is mqr+Hq, rather than m full output lifts, and its whole-layer bank may reside in host. This neither proves matched quality nor an attainable wall-time advantage over identical-map host/cache/ordinary-load/PE. Both factors can be host-resident and two algebraic factors do not require two NAND phases. Conversely local maps can also be cached, and CPU/DDR contention, bias/alpha/P work, canonical fallback, Deep source shared and full-model residency remain charged. Leading MACs, capacities or two optimistic lower bounds cannot establish domination. A later sufficient claim still needs appropriate D/NL/shared-output coverage and attainable whole-resource evidence; a passing early-layer control is not a publication or whole-model-speed result.
