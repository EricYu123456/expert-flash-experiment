The full-FFN same-format CPU placement reference passes. This closes the earlier three-layer gate/up and decode-only coverage gap for the executed Granite4Tiny workload. It validates placement equivalence under a fixed numeric contract; it does not establish original-checkpoint quality, a native AiF return format, or flash execution performance.

The frozen main workload uses only two previously selected training prompts: `train_programming_short_015` (75 prefill tokens) and `train_summarization_short_027` (153 prefill tokens), with 16 greedy decode tokens each and eight CPU threads. No test prompts were read. Independent contexts execute prefill separately in each mode; no prefill state is reused.

| Check | Main result |
|---|---:|
| FFN logical weight names covered | 240 = 40 layers × six projections |
| Callback records | 16,320 |
| Per-mode/job/phase/weight coverage keys | 1,920 |
| Placed I64 row dots directly checked against canonical rows | 197,017,600 |
| Cross-run intermediate FFN F32 values compared exactly | 197,017,600 |
| Same-input integer mismatches | 0 |
| Same-input scaled F32 bit mismatches | 0 |
| Cross-run intermediate FFN F32 bit mismatches | 0 |
| Vocabulary logits compared after final prefill token and each decode | 3,411,968 |
| Cross-run vocabulary-logit bit mismatches | 0 |
| Prefill/decode route records per mode, programming prompt | 3,640 |
| Prefill/decode route records per mode, summarization prompt | 6,760 |
| Route files and generated token IDs identical | yes |
| Cached expert/projection entries reached by the workload | 7,359 |
| Actual packed W8 bytes held in the 16 chip buffers | 5,881,724,928 |

All 40 layers execute replacements for routed `ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps` (`MUL_MAT_ID`) and shared `ffn_gate_shexp`, `ffn_up_shexp`, `ffn_down_shexp` (`MUL_MAT`). The independent full-token audit checks the exact input/weight/output tensor shapes and confirms every prompt token is replaced by all six projections in every layer, as well as every decode token. Ordinary shared matrix multiplication uses `[columns,tokens,1,1]`; routed multiplication uses `[columns,input_vectors,tokens,1]`. It verifies every saved reference FFN output and logit is finite.

Both placements use original GGUF `to_float` weights followed by fixed per-row symmetric signed W8 in [-127,127], and whole-vector dynamic signed A8 in [-127,127]. Scale is F32 max-absolute-value/127, with all-zero scale 1. Quantization uses F32 division followed by nearest rounding with ties away from zero and clipping. Every full row uses exact I64 accumulation, one I64-to-F32 conversion, and one F32 multiplication by the F32 product of row and input scales. There are no column partials. The compiler disables fast-math and floating-point contraction for the new runner. This is a fixed functional reference, not a proposed quantization method.

The routed chip assignment comes from the existing frozen 80-phase physical plan. For the gate/up phase, chip prefix counts refer to concatenated gate512 followed by up512 rows. Down uses 1536 full rows with 512 columns. Each selected row's actual W8 bytes and scale are copied into its chip's packed buffer. Within each chip and expert/projection, physical slots reverse logical row order. The placed kernel reads these packed bytes, computes whole-row dots, and scatters outputs to their original logical indices. The unsharded routed kernel reads canonical logical-row buffers. Shared rows use a fixed one-copy 16-chip equal-row partition and the identical packed buffers in BOTH modes. On-demand cache materialization is used; the workload does not claim to enumerate all 64 experts in every matrix. There is no untouched-source fallback for any selected FFN matrix.

Attention, router and SSM operators remain upstream source-GGUF host operators in both modes. The eval callback requests each selected FFN node, lets the existing host operation complete, and overwrites its F32 result before downstream consumers. Timings therefore include the discarded original operation, W8A8 reference work, and validation; they are not a flash latency or throughput measurement. Initial unsharded timings also include cold dequantization/cache construction. No performance comparison between these CPU times is claimed.

The frozen protocol and build manifest precede all revision-2 executions. Seven reused upstream source/static-library hashes exactly match the previous numeric experiment; no upstream file or static library was rebuilt or edited. The final input-integrity audit rechecks every frozen input, including the model, source, driver, binary, placement, and static libraries. Full intermediate reference outputs and all compared vocabulary-logit vectors are retained as F32 snapshots; callback and cache ledgers retain hashes, geometry, source offsets and chip row counts.

A preceding 75-token + eight-decode pilot passed the same checks: 62,894,080 exact integer/output comparisons and 903,168 vocabulary logits, with zero mismatches. The main run took 71.48 seconds of wall time, including loading and validation.

An initial harness-v1 pilot failed after 234 callbacks because upstream compacts the last prefill layer to requested output tokens. Its code, binary, frozen protocol, partial snapshots, logs and failure completion record are retained under `harness-v1/`. Revision 2 requests logits for all prefill tokens, which makes the unchanged upstream graph execute all last-layer FFN tokens. Only the final prefill logit vector is used for greedy continuation. This was a coverage-instrumentation correction, not a placement-method result.

The small generated continuations are retained for inspection. Token/route agreement between placements proves equivalence within this CPU contract; it is not evidence of checkpoint quality or task accuracy. A separate unchanged source-GGUF trace control is retained under `source-gguf-control/` solely to disclose the changed numerical reference. It uses the same two prompt-token sequences, greedy sampling, one sequence per context, eight threads, and 16 decode steps. Programming continuation token IDs match at 0/16 positions; GGUF starts with a Rust code block and `use std::collections::HashMap`, while W8 starts “Implement the function or class in Rust…”. Summarization token IDs match at 4/16 positions, with a four-token identical prefix; GGUF says “involves setting up a weather station…”, and W8 says “involves a weather station…”. Ordered decode routing matches at 0/640 and 66/640 records, respectively, under free-running continuations whose inputs diverge. These counts are not quality scores or isolated per-layer quantization-error rates. The original control executable was neither edited nor rebuilt and its before/after SHA256 agrees. Its default prefill requests only the last logit; the W8 coverage runner requests all prefill logits. No formal task-quality evaluation was attempted.

The 197,017,600 integer checks and 197,017,600 cross-run F32 checks concern the same output rows; they must not be added into a doubled sample count. Per placed run, programming prefill contributes 56,832,000 row outputs (240 callbacks), programming decode 12,124,160 (3,840 callbacks), summarization prefill 115,937,280 (240 callbacks), and summarization decode 12,124,160 (3,840 callbacks). Both modes together produce the 16,320 callback records. `main/count_definitions.json` preserves this breakdown.

The source-control executable completed successfully; the first postprocessor assumed different output filename suffixes. The unchanged frozen control launcher is retained, and the separate `source_gguf_control_analysis.py` analyzes the executable's actual `.traced` text/summary and unsuffixed route filenames without rerunning or changing sealed inputs.
