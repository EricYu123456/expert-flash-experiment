# Round1 Stage0 completed numerical results

TRAIN8 cap128 decode8 canonical CPU W8A8 logical opportunity only; no native AiF speed or heldout quality claim

| Model | Primary extra Up bytes | Primary added16KiB | Last4 + deflation extra Up bytes | Last4 added16KiB |
| --- | ---: | ---: | ---: | ---: |
| deepseek-v2-lite | 0.10120496% | 0 | 0.11456660% | 0 |
| olmoe-1b-7b | 0.08995533% | 0 | 0.09534359% | 0 |

The table uses free true-amax witnesses and pooled physical Up weight bytes. Primary is per-row minimum of global/block128/conditional bounds; last4 also includes fit16 deflation. A6/A4, paid seed, class and prompt-mean tables remain in summary.json; prefill diagnostics are in prefill_diagnostics.csv and summary.json. Prefill certificates were not collected; all prefill FFNs were canonicalized and checked for finite outputs and exact full-I product bits. Root and critic make the scientific round decision.

All certificates are checked against each quantizer’s own q bits and unchanged amax/scale bits. Coverage and finite checks must complete. 16KiB/4KiB pages are independently aligned original-order logical opportunity diagnostics; 4KiB is hypothetical. Equal-projection coefficient work is not device latency.

Metadata in report.json covers the union of decode-touched Up matrices, not full-model deployment capacity. Deflation storage is a logical selected16-I8 plus U64 residual estimate; host code reads the selected coordinates from full cached Up weights. Weight-cache storage includes Gate/Up/Down touched during prefill. See metadata_scope in summary.json for the cost boundary.
