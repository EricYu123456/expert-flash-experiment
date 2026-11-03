# Down work/headroom oracle

This is TRAIN-only chip-occupation analysis, not a new fullFFN replay or a latency claim. All rows retain their full private input: routed C512, shared C1024. All native jobs use fixed0123 drain and strict500ns finite PB. Existing variable-row compiler capability is reused.

| Layer | Fixed map | Mean critical us | P99 us | Reduction vs g4/shared16 |
|---:|---|---:|---:|---:|
| 0 | baseline_g4_shared16 | 177.780 | 224.024 | 0.000% |
| 0 | shared_exact | 177.705 | 239.384 | 0.042% |
| 0 | free4_equal_rows | 177.705 | 239.384 | 0.042% |
| 0 | variable_rows_support | 177.705 | 239.384 | 0.042% |
| 1 | baseline_g4_shared16 | 176.963 | 224.024 | 0.000% |
| 1 | shared_exact | 176.963 | 224.024 | 0.000% |
| 1 | free4_equal_rows | 176.963 | 224.024 | 0.000% |
| 1 | variable_rows_support | 176.963 | 224.024 | 0.000% |
| 2 | baseline_g4_shared16 | 177.663 | 224.024 | 0.000% |
| 2 | shared_exact | 177.125 | 239.384 | 0.303% |
| 2 | free4_equal_rows | 177.125 | 239.384 | 0.303% |
| 2 | variable_rows_support | 177.125 | 239.384 | 0.303% |

The route-dependent collision-free Down relaxation has exact minmax 90.734us (input1); it assigns two chips per routed expert and four to shared, all24pages/chip. It is only a lower bound because fixed one-copy weights cannot move per query.

Shared compaction enumerates all2925 integer four-group compositions, including unequal group row counts. Free4 support then moves equal384row shards, exposing ordinary declustering headroom. The variable stage permits complete-page row transfers and changing support, while checking the same one-block capacity. All maps are fixed across TRAIN and use exactly the same coefficient bytes.

The g8 shared-first/shared-last side calculation uses optimistic GU service floors and per-chip release scheduling. Its difference is exposure in a relaxation, not demonstrated FFN speedup. Root may evaluate shared-GU-first scheduling separately.

Every result must be compared fairly across GU layouts by giving each the same sharedDown/variable-row/support compiler menu and fitting budget. No sharding, local-search or mapping novelty is claimed. Native replay is required before treating any work reduction as a fullFFN improvement.
