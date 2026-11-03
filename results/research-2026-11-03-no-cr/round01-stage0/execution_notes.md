# Stage0 execution and cost boundary

The valid run used frozen revision003. Candidate arithmetic, bounds, witnesses, data order and thresholds are unchanged from revision002; revision003 only avoids constructing an allocation-backed error message on successful literal checks. Attempt001 is retained under attempts/attempt001 and is a tooling/performance retry, with no scientific failure assigned by the executor.

All eight frozen TRAIN prompts were tokenized without chat wrapping, retained as first min(128,N) IDs, and followed by eight greedy fed tokens with no EOS early stop. DeepSeek retains 82 tokens for the programming prompt; OLMoE retains 81; other prompts retain 128. Both full prefill and decode replace every active Gate/Up/Down projection with canonical row-W8/dynamic-A8 I64-dot outputs.

DeepSeek: 27 layers, dense layer0 I10944, routed layers1–26 I1408 K6/E64, and actual merged shared I2816. OLMoE: 16 all-routed layers I1024 K8/E64. All Up inputs have H2048. Projection and class coverage is complete.

Prefill has finite/output/product-bit and own-quantizer sparsity diagnostics; its certificates were intentionally not collected. All certificate scoring is on decode. Fit16 uses unique common layer/token qx once, first4 jobs ×8 positions only, with ties by coordinate ID. The frozen coordinates precede last4 inference/scoring; first4 decode records are then scored under those coordinates.

| Model | Primary extra rows / Up rows | Primary extra Up bytes / selected Up bytes | Last4 all-bound extra rows / Up rows | Full-I product checks | Ownq/amax/scale invariant rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| deepseek-v2-lite | 19678 / 19443712 | 40300544 / 39820722176 | 11138 / 9721856 | 316567936 | 699973632 |
| olmoe-1b-7b | 7546 / 8388608 | 15454208 / 17179869184 | 3999 / 4194304 | 136445952 | 301989888 |

No product-bit, finite-value, integer-bound, ownq, amax-bit or scale-bit check failed. The independent audit rederived fit coordinates from saved common qx, checked integer conditional/global and sample F32 arithmetic, and reaggregated every decode dispatch into the collector aggregates. See independent_audit.json.

The primary control computes full dense W8A8 Gate/Up/Down. Logical matched strong controls give exact post-A zero Down coefficients for free, and gate-exact-zero Up skips. The certificate comparison adds only Up rows beyond that gate-zero control; it cannot reattribute ordinary Down zeros. Paid witness forces its original first16KiB Up page read, including exact gate zeros, for both logical 16KiB and hypothetical4KiB analyses. All actual canonical Down dots remain dense; eliminating q==0 terms is exact I64 algebra, and the work-ratio controls are logical opportunity calculations, not measured sparse-kernel or device timings.

The source uses the pinned, unchanged libggml-cpu full-I SiLU and checks its product against the actual graph Down input on every active row. Every claimed zero is checked against its own A8/A6/A4 quantizer and removal leaves amax/scale/q bits unchanged. Conditional raw-gate dots are an optimistic added ABI capability. Conditional invalid/G0 cases fall back to the global bound; OLMoE recorded51 fallback row evaluations, all checked against actual reference dots and own quantizers.

Global, block128, Gram, seed and deflation metadata/host work are exposed in reports. report.json capacities count the union of decode-touched Up matrices. analytic_metadata_capacity.json separately reports full-model deployment capacity directly from frozen shapes. Deflation bytes use a logical16 selected I8 weights plus U64 residual representation; the current host evaluator reads selected coordinates from full cached weights and recomputes the exact integer residual. Caching all Gate/Up/Down row norms and blocks is collector overhead, not a proposed deployment requirement.

A6/A4 are post-hoc own-quantizer shadows of the A8 reference activation stream. In particular, DeepSeek free-oracle A4 has positive logical16KiB opportunities; this remains a shadow diagnostic and supplies no free-running W8A4 quality/speed claim. The A8 frozen decision is unaffected.

The valid run started 2026-11-02T20:12:46.020321+00:00 and completed 2026-11-02T20:16:26.024326+00:00. Timing includes CPU GGUF inference, callback replacement, cold canonical cache construction, and host certificate diagnostics, so it is not an AiF service-time or native speed measurement. No physical layout, energy, or end-to-end quality improvement is claimed.

The frozen numerical conditions are met for both models, including last4 deflation. Root and critic retain responsibility for the scientific round classification.
