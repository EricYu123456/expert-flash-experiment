# Conditional read-phase cadence control

**A slower legal read phase can beat dummy shaping in this model.** With copy500ns, candidate12 at10.24us/read completes151.380us, versus155.140us for13wave shaping at fixed9.7us. The nominal9.7us candidate fails finite-PB timing.

This is a source timing parameter sensitivity. No idle interval is inserted: every wave starts exactly when the previous array operation completes. Native legality or programmability of longer evaluation/recycling is unresolved; neither existing support nor a hardware requirement is established.

| Phase assignment | CR us | Candidate copy100 ns | copy500 ns | copy1000 ns | copy2000 ns |
|---|---:|---:|---:|---:|---:|
| fixed_recycle | 9.7 | 150.980 | FAIL | FAIL | FAIL |
| fixed_recycle | 10.0 | 150.980 | 151.380 | 151.880 | FAIL |
| fixed_recycle | 10.24 | 150.980 | 151.380 | 151.880 | 152.880 |
| fixed_recycle | 10.5 | 153.840 | 154.240 | 154.740 | 155.740 |
| fixed_recycle | 11.0 | 159.340 | 159.740 | 160.240 | 161.240 |
| fixed_evaluation | 9.7 | 150.980 | FAIL | FAIL | FAIL |
| fixed_evaluation | 10.0 | 150.980 | 151.380 | 151.880 | FAIL |
| fixed_evaluation | 10.24 | 150.980 | 151.380 | 151.880 | 152.880 |
| fixed_evaluation | 10.5 | 153.840 | 154.240 | 154.740 | 155.740 |
| fixed_evaluation | 11.0 | 159.340 | 159.740 | 160.240 | 161.240 |

Entries are completion times in us. Fixed-recycle keeps6.04us and assigns the added duration to evaluation. Fixed-evaluation keeps3.66us and assigns it to recycling. Both modes preserve one cold start and continuous issue.

The JSON also includes packed adjacent12 and spaced15 keep-hot/drop controls, for120cases total. Physical reads, useful/ECC bytes, latch deadlines, startup and final ECC drain remain explicit.

A10.24us four-page arrival period matches shared ECC service and removes the accumulating540ns/wave backlog. Strict9.7us failure therefore cannot be a standalone novelty claim. Dummy shaping is useful only under the narrower fixed9.7us contract when legal phase extension is unavailable or otherwise worse.

Published raw minimum latency and effective6.4GB/s may already incorporate cadence or overhead. This experiment does not infer a native timer interface, and it does not combine an effective-bandwidth approximation with a second raw array charge.
