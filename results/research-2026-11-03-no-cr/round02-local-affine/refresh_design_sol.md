# One representative FIT refresh — fixed Sol recipe (same Round2)

Planning started 2026-11-03 03:40:45 UTC. Source/theory only; no payload, model/activation/target/SCORE/TEST read, implementation or execution. Natural TRAIN16 selection is supplied by Root/Terra: `train-proposed-selection.json` SHA256 `70facca3342311acc19d956b85388b95c6d16e7dd740d4355084c21e075f2c5e`. PG validation16, test32 and original synthetic TEST100 remain sealed. This is exactly one recompilation, not a new round or a new candidate clock.

## Decision: rebuild shared geometry once on the complete TRAIN union

**Rebuild C32 tree, leaf means and normal PCA128 once**, using all old FIT120 inputs plus the frozen natural refresh union, unweighted, separately for each model/layer. Keeping the synthetic-only tree/P would confound a representative-data trial with a potentially obsolete input geometry and can manufacture an unfair domain-aliasing failure. Do not run a competing frozen-geometry refresh and select between them on PG validation. The old compiled artifacts are the before-reference only.

This single recompile does not isolate whether an improvement came from geometry or supervised coefficients; state that explicitly. The scientific question is whether this fixed representative compilation makes the concrete localbank/controls useful, not a one-factor causal ablation. All methods receive the same rebuilt selector/leaf means and the same target/query budget. G controls retain full-H input coordinates; they receive the new shared leaf centering/bias class, not an artificially restricted PCA128 input.

## Exactly which states enter

Before collection, freeze hashes for the **old** B_pilotcanonical, local C and selected D (PE6/G2, same existing eight operators) artifacts and the book/tokenization/position policy. These frozen models collect the entire refresh; no model is updated during collection.

For each TRAIN book/model, feed the same first768 ground-truth tokens to B, C and D using separate fresh contexts/KV and the same128-token chunking. Retain only B's768 prefill states. Continue each model on its own live logits/routes for up to32 tokens; retain B/C/D decode states. C/D must process their own full768-token prefix even though those prefix states are not added to FIT: copying B's prefix KV would erase the relevant student shift.

Thus the nominal maximum is16×(768+32+32+32)=13,824 new rows per model/layer. **Recommended stopping rule:** natural EOG stop, at most32 decode inputs, no post-EOG filling/resampling; actual count can be smaller. Root must bind this rule or explicitly bind fixed32 stress before capture. Fixed32 after EOG must not be mislabeled natural decoding. Books shorter than the frozen768-token requirement invoke the predeclared acquisition/coverage failure policy, not a content/output-dependent replacement. No high-error/rare-route selection.

A row retains `(model,layer,book/prompt ID,origin B/C/D,phase,absolute position,decode step,input hash,liveIDs/alpha,artifact hashes)`. New B/C/D decode rows are all retained even when byte-identical; multiplicities define this predeclared training mixture. Exact duplicates may share a memoized teacher result, but their statistical weight must not disappear. No cross-model or cross-layer input pooling.

Append to old FIT120, **not** old VALID, SCORE or PG validation. If old FIT has16,320 rows, maximum combined N is30,144/layer, about942 rows/leaf under realizable balance. Token correlation remains: report distinct old prompts/new books and mode/phase contributions, not30k independent examples. Natural prefixes contribute12,288 rows; the proposed refresh is still prefix-heavy and is not a universal decode distribution.

## Supervised targets and fit — no new search

For each row, evaluate each of that model/layer's two frozen pilot experts' canonical W8A8 function, whether routed there or not. Across the four model/layer pairs these are the eight operators; do not evaluate a layer's weights on another layer/model's row. All local/G/PE methods get identical union inputs and canonical targets. Reuse old FIT targets only after exact input/model/weight/numeric-contract identity checks; acquire new-row targets once and share them. Live routing labels come from the row's origin trajectory; primary fitting remains unweighted all-row.

Freeze the current dimensionless eta/lambda **per operator** and control-role registry. Recompute covariance-based absolute ridge strengths `eta*trace(S)/r` and `lambda*trace(S_leaf)/r` on the new FIT; do not keep old dimensional beta after changing P/distribution. No eta/lambda/rank/format/gauge-algorithm/role re-selection on PG validation. Keep C32/r128/normal local and q512-W8-host selected PE6/G2 exactly.

1. Refit shared runtime tree/leaf means and normal nested PCA on the full union, using the frozen sign/tie/median and operation-order rules. Replay the actual selector on FIT. Do not split identical x by row ID, reduce C silently, or keep an invalid tree. Unsupported geometry is inconclusive, not a manufactured method failure.
2. Recompute actual deployed decoded-A8 designs under new means/P, not transformed old latent samples. Cache the common r128 designs and leaf sufficient statistics once per layer. G operators also use the refreshed full-H centered/dequantized-A8 design, with identity input projection as before.
3. Refit optimally pooled within-leaf-demeaned prior and local slopes on union targets with fixed eta/lambda. Quantize the local maps using the frozen row-W8 rule and recalculate F32 biases from actual FIT deployment residual means. Do not leave old biases after changing centers.
4. Refit each selected q512 control using its own fixed family: PE uses the new local r128 slopes/stats; G uses the new full-H pooled slopes/stats. Build the fixed teacher-SSE M_target, take the same positive-eigen/cutoff top512 rule, and recompute latent factors. No alternate role or basis-estimator competition.
5. Preserve each control's chosen gauge **algorithm**, but recompute numerical FIT RMS/gauge parameters on this new union; old RMS values belong to old input statistics. Lower U/B with the same W8/host-floating-latent arithmetic and recalibrate biases from the actual refreshed path. Local and controls receive the same targets, same new geometry and complete recalibration.

PCA/tree and row statistics use TRAIN inputs only. Teacher fitting/calibration uses TRAIN targets only. Original VALID-selected role/hyperparameter decisions are transferred without another selector. A covariance eigendecomposition or output-basis refit is a deterministic fitting step, not a new hyperparameter search.

## Support and guards before PG validation

Primary guard remains100: it needs no new threshold. Record every leaf's N, numerical rank/effective degrees of freedom, distinct books/prompts and origin-phase counts, plus per-expert natural-hit counts. Balanced N>=2r is a diagnostic, not a generalization theorem. Require valid support and finite compiled outputs before opening PG validation; failed numeric/tree coverage is tooling/inconclusive.

If already frozen guard50/70/85 shadows must be carried, compute their distance thresholds **only from this union TRAIN's live-routing hits**, using the same shared full-x distance and frozen quantile convention. No PG validation losses/distances determine them. Missing-hit regions use documented canonical fallback and are reported; do not tune percentiles, add a new risk predictor or pick a guard after PG results. TRAIN-calibrated coverage is an empirical policy with distribution-shift limitations, not a certificate.

## Cost reuse and finite stopping rule

Per layer, compute tree/P/design, means, covariances and their eigensystems once and reuse across the two target experts. Cache only expert-specific cross-statistics/targets as necessary. Reuse local float slopes for PE M_target/factors; G has its full-H shared input covariance. Stream each expert target once, do not rerun canonical inference per penalty/rank. The new targets are roughly8×13,824 expert evaluations; there is no justification for another all-format/all-C/all-eta grid. The quality-only current-state oracle may remain batched and cached in its worker; it is not a compressed-runtime benchmark.

Freeze refreshed artifact hashes and source/build before PG validation16. Compare source, B_pilot, refreshed C and refreshed D with their own contexts/KV/routes. PG validation is a development quality screen, not a selector for geometry/roles/hyperparameters. No second refresh or alternate frozen-geometry rescue after viewing it. If it exposes a new unresolved data/implementation issue, report it; do not call a sparse/undertrained run a universal method failure. Actual bounded quality-and-resource failure remains Root/critic adjudication, and partial8operator quality never proves full routed deployment success. Natural test32 and synthetic TEST100 remain sealed for a separately frozen final evaluation.

Final Sol recommendation: adopt the single rebuilt-geometry union recipe above, retain the current role/hyperparameter registry and primary guard100, and bind natural-EOG-stop (nominal13,824-row maximum) unless Root explicitly amends to a separately labeled exact32 stress acquisition. Approximate new teacher work from the two model shapes is0.83T integer MACs before reuse/batching; collection still requires all three independent prefix runs, so do not estimate it as one prefix plus three cheap continuations. Full covariance/eigen fits are shared by layer/family and expert-specific cross-statistics stream once. No alternate geometry, refresh iteration or PG-selected rescue is part of this proposal.
