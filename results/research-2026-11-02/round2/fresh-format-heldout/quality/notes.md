The fixed-teacher full-FFN quality comparison completed. On eight acquisition-frozen WikiText2 raw test chunks, the same 128 GGUF-token sequence is used independently by the configured Q4 GGUF reference and by the full-FFN W8A8 reference. Each chunk scores its 127 next tokens. GGUF special-token/BOS behavior is preserved through `llama_tokenize(add_special=true, parse_special=true)`; no chat template is applied.

| Metric, 1,016 scored next tokens | Q4 GGUF | Full-FFN W8A8 |
|---|---:|---:|
| Mean next-token NLL, F64 | 2.798108740412623 | 2.7990450129151756 |
| Perplexity, exp(total NLL / 1016) | 16.413575066821085 | 16.428949842191308 |

W8 minus GGUF mean NLL is +0.0009362725025527929. The perplexity ratio is 1.0009367109424747, a +0.093671094% change in this bounded sample. Five chunks have higher NLL and three lower; chunk-level changes range from -0.053127178 to +0.045028805. The small aggregate change is not an equivalence or non-inferiority result. This is sensitivity relative to the configured Q4 GGUF reference, not original BF16 quality, official WikiText benchmark perplexity, certified task accuracy, or a population-level generalization claim.

Teacher-forced top-1 predictions agree at 935/1016 positions (92.0275591%). This is a numerical prediction-agreement diagnostic; the next-token likelihood comparison is the quality-sensitivity measurement. The fixed teacher inputs remove the free-running token-divergence confound from the earlier generation control.

Across all 128 teacher input positions and 40 layers per chunk, the 40,960 paired router records have ordered top-six identity at 22,000 records (53.7109375%), unordered set identity at 33,274 records (81.2353515625%), mean intersection/6 of 96.7814127604%, and mean Jaccard of 94.5265109240%. The router metrics include the final input position whose next-token prediction is not scored. Per-layer and per-chunk details are preserved in `summary.json`.

All six routed/shared gate/up/down FFN projections are intercepted in all 40 layers for every one of the 128 teacher tokens. Logits are requested for every position so that the unchanged upstream last-layer graph does not compact prefill to one output token. Source-GGUF FFN outputs are scanned for finiteness without replacement; W8 mode uses the identical previous fixed row-W8, whole-vector dynamic-A8, exact-I64-dot, one-F32-rescale contract. Attention, router and SSM remain original-GGUF host operators in both modes. Seven numeric/cache/operator function bodies were checked byte-for-byte against the earlier full-FFN runner before protocols were frozen; no quantizer or placement fitting was performed.

All 205,520,896 main source/W8 vocabulary-logit values were finite. The first two full logit vectors from every chunk/mode are retained, and an independent NumPy F64 logsumexp recomputation differs by at most 4.44e-14. The first W8 chunk was repeated in a fresh context; every one of its 12,845,056 vocabulary logits matched bitwise, as did its route file, per-position NLL records and per-operator FFN hashes. The 17 total passes produce 4,080 fully covered FFN callback records.

Quality protocol SHA256 is `d614c5f1cfcc8922b77902dad8f69be634880f7441497184de681fcc8227281d`. It was frozen before inference and points to the previously frozen acquisition protocol and prompt manifest. The execution completed successfully in 96.50 seconds while a separate eight-thread generation validation ran concurrently. These are CPU harness timings, not hardware service measurements.

The actual GGUF metadata is `tokenizer.ggml.add_bos_token=false` (BOS token ID100257 exists but is not automatically inserted). Both references follow that same metadata, so no artificial BOS token was added.
