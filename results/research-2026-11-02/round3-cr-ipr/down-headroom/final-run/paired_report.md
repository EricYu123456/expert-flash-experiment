# Fixed shared-Down partition: TRAIN512 causal replay

All 24 configurations completed 1,536 layer/query events with finite PB and a 128-byte output FIFO. The 12 incumbent profiles reproduce the sealed asymmetric-Down cost and FIFO arrays exactly. Selection and priorities remain the frozen TRAIN64 choices.

| GU | input copies | baseline mean µs | partition mean µs | mean reduction | event P99 change | 3-layer P99 change |
|---|---:|---:|---:|---:|---:|---:|
| candidate | 1 | 338.596 | 318.720 | 5.870% | -3.665% | 2.189% |
| candidate | 4 | 344.971 | 323.714 | 6.162% | -3.402% | 2.571% |
| RR | 1 | 339.961 | 320.085 | 5.847% | -1.465% | 2.688% |
| RR | 4 | 346.336 | 325.079 | 6.138% | -1.343% | 3.071% |
| g8_fixed | 1 | 348.256 | 324.954 | 6.691% | 7.913% | 8.746% |
| g8_fixed | 4 | 353.963 | 329.813 | 6.823% | 7.993% | 8.937% |

Positive percentages mean lower latency. The primary pair uses screen-selected DF for the partition and GF for the incumbent; same-priority controls are in paired_report.json. Per-query regressions and tail changes must accompany mean gains.

The map reserves chips 12–15 for shared Down. Routed Down uses three disjoint groups of four chips with 22/21/21 experts; every routed job has 384 complete output rows, 12 pages, one cold start and three CR waves. Shared jobs have 384 rows, 24 pages and six waves. GU kernels are unchanged. A whole private job may occupy block 1 under the common two-block budget; no job crosses blocks or incurs an extra cold start.

This is a stronger standard resource-partition baseline supported by the existing generic whole-row compiler. It does not establish algorithmic novelty. All fitting and replay use TRAIN, conditional PB/CR assumptions are inherited, and no heldout or measured hardware performance is claimed.

TRAIN64 static grouping and 63+1 one-block variants remain separate screen controls; they were not selected again using TRAIN512. The inherited gate-name metadata error is documented in gate_metadata_erratum.json.
