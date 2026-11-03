# Equal-capacity fixed shared control

All 12,288 TRAIN events passed the finite-PB model and 128-byte output FIFO checks. Both policies keep the same 303 MiB, 544 resident jobs, layout, and physical addresses. The control fixes S0 on pair 6 and S1 on pair 7; routed experts use pairs 0–5 with deterministic cap1, then cap2 matching. Joint dispatch can place both shared halves on the remaining pairs.

| GU | Array | Input copies | Fixed mean (µs) | Joint mean (µs) | Joint mean reduction | Event P99 reduction |
|---|---|---:|---:|---:|---:|---:|
| candidate | cr | 1 | 263.971 | 254.712 | 3.507% | 25.297% |
| candidate | cr | 4 | 268.082 | 258.856 | 3.442% | 24.927% |
| RR | cr | 1 | 265.974 | 256.793 | 3.452% | 20.517% |
| RR | cr | 4 | 270.771 | 261.739 | 3.336% | 19.979% |
| g8_fixed | cr | 1 | 290.618 | 282.957 | 2.636% | 14.472% |
| g8_fixed | cr | 4 | 295.189 | 287.608 | 2.568% | 14.330% |
| g8_fixed | ordinary | 1 | 595.318 | 579.682 | 2.626% | 11.468% |
| g8_fixed | ordinary | 4 | 599.888 | 584.333 | 2.593% | 11.414% |

Each axis has 166/1536 cap2 events (10.807%). Candidate fixed shared is slower in exactly these 166 events and equal in all 1370 cap1 events. RR and native g8 retain 20 events where fixed shared is faster. All differences, failures, aligned three-layer P99, and resource counts remain in the raw reports.

The 64-index screen was only a PB/FIFO correctness gate. All eight axes were frozen before screening and all eight ran on 512 TRAIN indices, regardless of speed. This is a posthoc structural ablation on previously observed TRAIN routes; it is not independent validation or a new dispatch algorithm. The native g8 CR and ordinary axes keep their original capabilities. Candidate/RR retain the original conditional source-order/drop assumptions.

The saved-trace checker independently verifies 96 JSON traces (48 screen and 48 full), including every dependency, resource interval, chip owner, continuous ONFI drain, complete output-row coverage, and selected resident address. It checks the saved finite-PB certificates; it does not independently resimulate the latch model. Original sealed dependencies remain unchanged. No prospective files were read.

Source: `control.py`; frozen axes and inputs: `protocol.frozen.json`; full results and arrays: `TRAIN512/`; descriptive cap1/cap2 stratification: `interpretation.json`; saved trace verification: `saved_trace_checks.json`.
