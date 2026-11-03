# Prospective nonlinear control: host projected-GU + original canonical Down

2026-11-03, same Round2. Source/theory only; no implementation, model/data/target/activation or evaluation payload accessed. This is a **future necessary strong control if the current local-quality screen survives**, not an amendment to the active refresh/BIND/collection protocol. Current teacher archives preserve final Y, not Gate/Up side outputs. Implementing this control requires a later explicit source/data-budget release; do not describe its targets as already saved or its fitting as zero-query.

## One fixed operator, not a grid

Use the same refreshed C32 tree/centers, normal layer-shared P128, actual decoded-A8 input design v, and exact same FIT input union as the local bank. For each expert e, store two **global** host FP32 slopes B_G,e and B_U,e, each I×128, plus leaf-specific FP32 Gate/Up intercepts. Runtime:

`qv,sx = canonical_A8(P128(x-meanX_c))`
`g_hat = F32(sx * FMA32(B_G,e, float(qv))) + b_G,e,c`
`u_hat = F32(sx * FMA32(B_U,e, float(qv))) + b_U,e,c`
`a = pinned_fullI_SiLU(g_hat) * u_hat`  (declared F32 product)
`qa,sa = canonical_wholeI_A8(a)`
`y_hat = canonical_W8_Down_e(qa,sa) + b_Y,e,c`.

Every multiply/add and final bias uses the existing frozen numerical-worker order. Full-I SiLU is the pinned GGML routine, not a new libm/GELU or latent nonlinearity. The final C32 output residual bias b_Y is included, because localbank has the same free-output-bias capability. No extra output quantizer is introduced. Original **Down** here means the exact original row-W8 Down matrix/scales of the declared canonical reference, not a source-GGUF block-scaled Q4/Q8 tensor silently substituted under that name.

Use the real natural selected IDs and final live alpha. Evaluate each expert, then preserve the original selected-slot alpha multiplication/add order. No weighted Gate/Up merge before SiLU: generally `SiLU(sum alpha*g)` is not `sum alpha*SiLU(g)`, and the subsequent Up product is nonlinear too. Unreplaced operators/shared/dense follow the same reference as the local comparison.

## Fixed strong fitting rule and explicit extra target cost

For each expert, obtain its canonical preactivation targets g_e(x),u_e(x) on the same FIT inputs (complete-H A8 integer GU and original F32 scale order). Fit B_G/B_U by **eta0 optimally pooled within-leaf-demeaned least squares** on actual decoded-A8 v, with the existing1e-10 pseudoinverse/zero-trace rules and no eta/rank/format search. Recompute each GU intercept from the actual host prediction residual mean. Then run the exact nonlinear+original-Down path on FIT and calibrate the final b_Y from the actual canonical-Y residual mean. Parameters depend on FIT only; PG validation never selects a projection or baseline version.

This is a reproducible input-projected nonlinear control. The direct weight projection `G_dequant P^T, U_dequant P^T` with anchor offsets is a reasonable training-free construction, but can be weaker because the canonical full-H quantizer and latent design differ. The pooled preactivation regression is chosen here as the **single primary**; do not run both and select on held-out text. Eta0 is a strong fixed choice, not proof of an optimal nonlinear projection or all nonlinear model families.

Current final-Y archives cannot recover g/u uniquely. A later approved pass must recompute the canonical GU intermediates, or explicitly amend a future teacher pass to return them. Additional target work is approximately2HI per FIT row/expert, plus evaluating the proposed Down path for calibration. Existing canonical Y may be reused only after exact identity checks. Record those extra offline operations/bytes. Numerical fitting can share the same input covariance/eigensystem across all experts/G/U heads and concatenate the2I targets in one solve; no new SGD pipeline or giant grid is needed.

## Exactly what is shared

P128, selector, leaf means and the qv/sx input are shared across selected routed experts at a layer. B_G,e and B_U,e are expert-specific because original G_e/U_e differ; sharing P does not make their projected weights equal. Host execution can concatenate selected heads into a batched GEMV and reuse the input, while still paying every selected head's coefficients/MACs. No free cross-expert activation reuse or pre-nonlinearity live-alpha aggregation is assumed.

For Deep's native source shared expert, **do not apply this approximation** or replace its block-scale/source activation function. In the current B_pilot/routed reference it stays exact source-GGUF on the host. It may consume the same full x but does not automatically reuse routed B_G/B_U. A separately declared canonical-shared reference can compare legal placements, with its own representation-quality audit. The function/placement distinction is identical for candidate and control.

## Full-model capacity and retained-weight honesty

At H2048/r128/C32, FP32 projected GU plus its GU/output biases require, before P/tree/Down scales:

| Shape | Host projected GU | Host GU biases | Host output biases | Original W8 Down |
|---|---:|---:|---:|---:|
| Deep I1408, E64, L26 |2.399GB|.600GB|.436GB|4.798GB|
| OLM I1024, E64, L16 |1.074GB|.268GB|.268GB|2.147GB|

These are decimal coefficient/parameter arithmetic, not a proof of inclusive8GB fit. Add P/tree, row scales, source non-FFN/shared, KV at the actual length, workspace/cache/allocator and load costs. FP32 host GU is deliberately allowed because it is plausible within the target budget and avoids a strawman extra GU weight quantizer. Candidate and all controls get equal remaining host-cache bytes and policy ability.

Without retaining unused original GU, approximate operator bytes including Down are about8.23GB Deep /3.76GB OLM before small metadata, below canonical original routed W8 totals14.40GB/6.44GB. **With canonical fallback retained**, original G/U/D remain stored and these projected GU/biases are additional: the retained representation is then larger than plain canonical original, not compression of it. Down must alias the already retained original pages rather than be counted/stored twice. Localbank's original-fallback copy is billed by the same rule.

OLM may fit the whole projected-GU+Down function on the host, and its original canonical W8 model may itself fit near the known6.8GB floor. Both are mandatory legal execution alternatives at closed inclusive budgets. Forcing an otherwise host-resident Down onto flash would weaken the baseline. The operator above is fixed; host/flash/cache placement of the same Down function is a scheduling choice, not another fitted grid.

## Native Down path, traffic and dependencies

If Down is on AiF, the host first completes projected GU, full-I SwiGLU/amax and qa. Every expert has a different I-length qa/sa, so it pays its own input post/context replacement to each owner group. No K resident PB inputs or shared bank-style A8 broadcast is inherited. There is only **one** original-Down sensing phase after host nonlinear work, with an actual host→Down dependency; the two original flash GU/Down phases are gone, but the localbank's host-input and output work are not thereby free.

W8 Down coefficient pages per expert are176 for Deep (H2048×I1408) and128 for OLM (H2048×I1024), versus16 for a local r128 output map. The aggregate no-cache K6/K8 Down-only sensing lower bounds are17/16 full64-plane waves, respectively, before legal layout/command/PE/drain details; they are not executable elapsed schedules. Packed original rows may continue across pages under their explicit command contract: do not force1408-byte rows into padded16KiB pages. No coefficient read is skipped solely because two experts share their input P.

Output is H rows/expert: under the proposed I32 ABI,8KiB per expert plus host scale/bias/alpha work, as for the bank. Original Down's I-length input width is distinct from the r128 proposed short-row extension, and its PE products are KHI rather than KHr. Charge ECC/PE rates, row completion, channel/FIFO backpressure, input fanout1/4 as justified, setup/fragmentation, output transport and cache misses. One input context per chip/group is the default. The source's6.4GB/s ECC rate is not a MAC rate or proven short-vector service.

Host projected GU per selected Deep expert reads roughly1.44MB FP32 coefficients and executes2Ir=360,448 host MACs, plus GU intercepts/nonlinearity/quantization and common P. Original-Down service can dominate. A control losing flash read time might still preserve quality where the bank fails; only a matched-quality whole-resource comparison decides usefulness. Exact source shared on host competes for the same CPU/bandwidth, while other legal functions/resources may overlap. Count these in the same target i9 model and separately labeled EPYC measurement, not the quality IPC harness walltime.

## W4 comparison boundary and interpretation

Do not silently replace the fixed original W8 Down with W4. A future matched-quality low-bit comparison is allowed only with a separately frozen format, numeric audit and equally available bank/original quantization. Two nibbles per byte do not imply two PE MACs/cycle. Per-row W4 scales and groupwise/block scales have different accumulation/rescale and output-partial requirements; source Q4_K_M is not automatically the proposed simple native W4 datapath. ECC-before-unpack, sign/code convention, index ordering, page packing, decoder/mux/control, row/group scales and actual PE throughput must be charged. No W4/rank sweep is part of this prospective baseline.

If this one fixed nonlinear control matches quality and wins complete costs, C32 full output-bank necessity is unsupported for that bounded configuration. If it loses quality, that does not disprove every nonlinear latent/control family. If only forced-flash placement loses while a legal host/cache path fits, report the stronger path. Nothing here delays or modifies the current refresh and necessary quality screen; implementation is deferred until those gates justify this additional baseline and its target/source budget is explicitly released.
