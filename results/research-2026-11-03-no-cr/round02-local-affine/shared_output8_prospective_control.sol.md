# Prospective shared-output8 control: one host lift after latent mixing

2026-11-03, same Round2; source/theory only. This is not an addition to the released Stage3 work. No candidate implementation, experiment, teacher query, payload inspection or new method clock. Consider this single future control only if the current fixed B/C/D/NL quality-and-cost review leaves the conditional output-bank necessity unresolved.

## Decision and scope

The existing canonical final-Y archives are sufficient to fit a common output basis. No GU or new canonical-target queries are needed. Use the existing eight operators, grouped **only by their four existing pilot layers**, two experts per group. Call the future arm **SO8-pre**. Fix q512/W8 factors, normal shared P128, C32 and the current local-fit penalties. There is no new expert clustering, rank/gauge/grid search or teacher/state refresh.

Choose **host floating-latent alpha premerge followed by one common output lift** as the sole prospective quality arm. A per-expert, non-premerged implementation of the same factors is needed for actual single-expert bias calibration and numeric reference, but is not a second quality arm to select between. The premerge arm tests whether sharing the expensive output operation can remove a host bottleneck. It defines a new rounded function and must receive its own quality evaluation; do not inherit D's passed development band.

This is an eight-operator control, not evidence that a basis trained for two experts serves all64. Each group contains at most two naturally selected approximated experts, regardless of the original K6/K8. Common output lifts and latent experts are established architecture ideas: the source dossier identifies MoLAE §4.2 and its group-shared output projection. EBC-LLM is a close shared-rotation lead, but its inaccessible full mathematics remain unverified. Neither label establishes this control's quality or AiF latency. No new architectural novelty is claimed for SO8-pre.

## One fixed FIT recipe

Reuse the frozen TRAIN union, row identities, tree/P/centers, decoded-A8 design v and leaf assignments. At each of the two experts e, recover the **unquantized fixed-estimator slopes** A_e,c using its existing pooled eta/local lambda and singular/cutoff rules. Reuse complete sufficient statistics if retained; otherwise read the existing verified Y and recompute those statistics. Do not derive the basis solely from a truncated per-expert U512: omitted directions can matter to the joint basis. No residual-based row selection or normalization by an expert's output energy is introduced.

In column-vector notation, A_e,c is H×r. Let vbar_c and ybar_e,c be FIT leaf means,

`S_c = mean_c[(v-vbar_c)(v-vbar_c)^T]`,

`T_e,c = mean_c[(y_e-ybar_e,c)(v-vbar_c)^T]`,

`p_c = N_c/N`.

The existing estimator predicts centered output `A_e,c (v-vbar_c)`. Define

`M_e = sum_c p_c [T_e,c A_e,c^T + A_e,c T_e,c^T - A_e,c S_c A_e,c^T]`,

`M_group = (M_e1 + M_e2)/2`.

This gives **unit per-expert weight**, with each expert's row mean normalized by its own fixed N. It is not router-, alpha-, energy- or frequency-weighted. With orthonormal U, projecting the fixed estimators onto U changes the unpenalized centered FIT teacher SSE by `-trace(U^T M_group U)` plus a constant. Thus the top positive q512 eigenvectors provide the best rank-at-most512 **projection of those fixed estimators** for this objective. It is not the optimal quantized predictor or an optimal nonlinear latent network.

Use the existing symmetric FP64 EVD, relative1e-10 positive-eigenvalue cutoff, deterministic source/library/sign convention and zero padding to512. Record zero-trace/rank/negative-roundoff diagnostics. Fix the identity gauge; no FIT-RMS/rotation alternative or validation selection. Set `B_e,c = U^T A_e,c`. Because the original shrink penalty is scalar across output coordinates, projection commutes with the fixed real-valued ridge solve; a new eta/lambda search is unnecessary. The post-quantization statement is only that this is a reproducible control, not a quantized optimum.

Lower **one** shared U per layer to canonical row-W8 with common row scales Us. Lower each q×128 B_e,c to the same row-W8 scheme and its own scales. Keep host latent FP32. Use the actual frozen qv/sv and exact integer-dot/scale arithmetic for B. The host output lift uses the existing I8-weight/F32-input eight-lane FMA and exact reduction order, with Us applied in the frozen order. No extra latent A8 quantizer is imposed on this all-host control.

For each expert/leaf compute its output bias from the sequential FP64 mean of canonical Y minus the **actual rounded single-expert prediction** `F32(Us * FMA8(Uq,h_e))`, then round the bias once to FP32. This is the actual SO8-pre function restricted to one member with alpha1; it includes W8 B/U, row scales and the deployed FMA path. Do not use the ideal U B product's bias and pretend rounding is unchanged. Retain exactly32 free H-dimensional biases per expert and the corresponding immutable C8 support/fallback policy.

Zero new teacher queries does not mean zero fit cost: four common H2048 eigensystems, moment reconstruction/IO, factor quantization and actual-prediction bias calibration must be logged. There is no GU target collection or new corpus. The original C8/D bytes remain unchanged.

## Actual primary arithmetic and graph placement

For the supported pilot members selected at a token, in their original selected-slot order:

1. Compute their h_e from B_e,c and the one common P-derived qv/sv.
2. Form `t = ordered_F32_sum(F32(alpha_e * h_e))` in q512 coordinates.
3. Form `b_mix = ordered_F32_sum(F32(alpha_e * b_e,c))` in H coordinates.
4. Compute `z = F32(F32(Us * FMA8(Uq,t)) + b_mix)` once.

When no pilot member is selected, skip this group. Canonical-fallback members retain their own canonical function and original weighted slot; only supported approximations participate in premerge. Record both kinds of exposure. All nonpilot expert functions and source shared/dense functions remain the declared partial reference; no pre-SiLU shared-expert substitution is made.

In exact real arithmetic a common U gives `sum alpha_e(Uh_e+b_e)=U(sum alpha_e h_e)+sum alpha_e b_e`. This **does not give F32 bit equivalence**: alpha moves across the FMA and output scale, bias/addition order changes, and the selected-slot sum is regrouped. Cancellation can amplify relative rounding differences. If a future flash-U path A8-quantizes t, its quantizer adds another changed function; that path is not part of this sole host primary.

The source graph has an important harness requirement. `llama-graph.cpp:1669–1707` constructs `ffn_moe_weighted` after actual final alpha multiplication and then adds its H×K×T slots in order. A Down-only override cannot implement premerge by dividing z by one member's alpha and relying on the graph to multiply it back: that changes arithmetic and can be undefined or ill-conditioned.

Instead, a separately source-reviewed future hook must patch **weighted slots**: put z in the earliest participating pilot slot and zero the other participating pilot slots, then keep the graph's existing remaining slot-add order. Read actual final alpha from the weight operand, not the possibly pre-normalized debug weights tensor. This explicitly defines the regrouped new function, including interleaved nonpilot slots. Keep each arm's own KV/routes/alpha; no original trajectory is pinned for quality. Existing Stage3 does not acquire this hook or arm by implication.

The non-premerged reference computes `F32(alpha_e * F32(p_e+b_e,c))` for each member and lets the original global sum proceed. It can preserve per-expert factor arithmetic and original slot organization. It uses m lifts for m active members; a blocked host kernel may reuse U's cached coefficients across those inputs while preserving every row's lane order. It is not legitimate to bill m independent DRAM reads when U is actually reused. Conversely, its mHq MACs do not disappear simply because U is common. Use it for derivation/calibration/unit tests only; do not run both arms on PG16 and choose the better result.

## The resource claim that this eight-operator control can test

For one group and m=1 or2 supported active members, SO8-pre requires approximately

`m*q*r + H*q` products and the same leading W8 coefficient bytes,

plus common tree/P, latent/alpha mixing, scales/biases and outputs. Non-premerge needs `m*q*r + m*H*q` products; its ideal shared coefficient stream can still be close to `m*q*r + H*q`. The latter is a cache/layout opportunity, not free compute.

At H2048/r128/q512, for m2 the premerged coefficient/product count is1,179,648; non-premerge products are2,228,224. Two local maps use524,288 products/bytes before metadata. Thus two-expert shared-U does **not** beat local's coefficient/MAC count: its potential advantage is an attainable all-host execution with reusable lift, changed capacity and no flash service, rather than fewer arithmetic operations.

One complete group stores4,194,304B of B matrices,1,048,576B of U,131,072B B scales,8,192B U scales and524,288B output biases:5,906,432B before common P/tree/guards. Four groups total23,625,728B before their common inputs/metadata. Compare actual inclusive host residency and traffic with C8/D; do not extrapolate a micro working set to complete-model cache behavior. Source shared, non-FFN, KV and CPU competition remain billed. Array residency is not LLC residency or zero latency.

At native16KiB granularity one U512 is64 pages, whereas each local r128 map is16 pages. For these actual groups m≤2, one shared U still has more coefficient pages than16m local pages. The often tempting64-versus96/128-page comparison requires a **single basis for all K6/K8 members**; that is not this trained/tested scope. Multiple host inputs can reuse U without new hardware, while page-stationary multi-input native execution needs a paid input/context/buffering contract; no free PB staging is assumed.

For scale arithmetic only, a hypothetical full64 group would store roughly156MB/layer with these q/r/C formats and metadata before common P/tree and original fallback. This could fit a complete model's latent representation in host memory, but **its U would have to be fitted from all64 targets and retain quality**. A U trained from two experts provides no evidence for that condition. No full64 target queries, quality or runtime conclusions are authorized here.

The only current cost comparison is conditional: can measured attainable shared-lift host service, with real cache/CPU/alpha/bias costs, challenge local's legal host or flash service at matched quality? Allow the identical local bank on host and its hot cache; allow fixed D/other legal placements. A host-control upper service below a candidate optimistic whole-DAG lower bound could refute a bounded hardware profile. Comparing two bandwidth/sense lower bounds cannot. The original OLM all-host baseline and nonlinear Stage3 control remain necessary, not superseded by this algebra.

## Interpretation and later gate

Root reports that fixed D's reused-development D−B bands also lie below the existing margin, while emphasizing that this is neither fresh confirmation nor all64 quality. This supports finishing the authorized direct host-cost evidence first. It does not justify retuning D or automatically launching SO8-pre.

If B/C/D/NL plus the actual host-cost review leave a material shared-lift crossover, freeze this one SO8-pre recipe and new hook/arm separately before any payload/quality release. Existing Y permits the fit. Evaluate its own actual rounded mixtures on the already exposed development cohort only under that explicit release, without adapting C8, basis/rank, grouping or gauge. Any later fresh-validation event needs its own frozen-version boundary.

A good SO8-pre result would weaken the necessity of conditional **full-H** output banks for these eight operators; it would not prove equality, complete-model quality, full64 basis sufficiency or all-native speed. A bad result would reject only unit-per-expert/common-U512/linear-latent/premerge under the fixed budget. It would not rule out route-weighted bases, nonlinear MoLAE latents, other ranks or other groups. There is no theorem equating arbitrary32 conditional full-output maps with one rank512 lift: their output-span union can exceed512. Therefore shared-U is a necessary strong empirical control, not an algebraic thought-rejection of the entire localbank family.

One frontier qualification, not another proposed arm: a future full64 layer/context-specific lift U_l,c is a stronger conditional-sharing family that global U does not encompass. In the current two-expert/one-leaf setting, however, the concatenated H×256 local slopes have rank at most256. A nominal per-leaf q512 basis is therefore rank-deficient for representing those two maps; it cannot be assumed to contain512 positive/useful directions or to establish full-K value. Per-leaf factor capacity, quantization and input-dependent basis reads would still be paid. This is an untested general-family possibility only, with no new fit, grid, source scope or clock authorized.

Source/theory receipt:2026-11-03 05:52:22–05:58:10 UTC; only existing source, protocol/theory notes and Root-provided summaries were consulted. No target/model/archive/book/evaluation payload was read.
