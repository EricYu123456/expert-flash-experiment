# CR-preserving overfetch/drop shaping

A13wave schedule is feasible at500ns copy under the stated two-weight-latch, cache-only, evaluation-start overwrite model. It uses one28us cold start and twelve uninterrupted9.7us CR waves. Missing planes read real unselected resident expert pages and conditionally drop them before ECC.

**Priority result:** split original wave10 into1+3useful pages. Useful/ECC pages remain48; physical reads become52 (+8.333%). The finite-latch event schedule completes at155.140us, versus170.140us for reactive cold reset. Original12waves fail the6400>6040ns necessary deadline bound. Thus13waves are minimal within this contract.

No new stored weights, weight FIFO, read/cache mux, idle CR bias hold, or extra partial-row contexts are introduced. Useful page order is identical. Drop-before-ECC and programmable source order are explicit conditional capabilities.

| Copy ns | Best13waves us | Feasible splits/36 | Best14waves us | Feasible splits/594 |
|---:|---:|---:|---:|---:|
| 0 | 150.880 | 36 | not swept | — |
| 100 | 150.980 | 35 | not swept | — |
| 320 | 151.200 | 35 | not swept | — |
| 321 | 154.961 | 31 | not swept | — |
| 500 | 155.140 | 31 | 157.160 | 585 |
| 1000 | 156.720 | 20 | 157.660 | 510 |
| 2000 | FAIL | 0 | 166.880 | 56 |
| 3000 | FAIL | 0 | FAIL | 0 |
| 4000 | FAIL | 0 | FAIL | 0 |
| 5000 | FAIL | 0 | FAIL | 0 |

Controls use the same drop capability. Adjacent ordinary packed12 can receive the same shaping transformation and match this envelope; no stencil-exclusive benefit is claimed. Spaced packed15 keep-hot/drop is also reported. Twelve varied route-address instantiations validate that dummies always belong to unselected experts.

The public API is shaping.py: build_shaped_candidate(expert_ids, splits), select_shaped_candidate(cache_copy_ns, expert_ids, max_added_waves), evaluate. Canonical structural selection is cached; each actual route is rebuilt and address-certified. Results are an addendum; sealed v1 and stream-burst outputs are unchanged.

This is conditional timing feasibility ending at ECC completion. Native support for drop/source control, source-switch overhead, MAC throughput, power, read disturb and output buffering require separate evidence.
