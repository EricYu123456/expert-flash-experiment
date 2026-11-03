# Whole-row macro lowering with a fixed PB source

Every physical wave reads four real resident pages; dummy weights are conditionally dropped before ECC. All useful three-page macros remain contiguous after drop and preserve per-owner order. No coefficient is moved or copied.

Candidate fixed0123 needs at least18waves: each of six B=[0,1,3] macros requires a plane2dummy, on top of12usefulplane2pages. The exact DP below checks attainability.

| Layout | Source | Exact minimum waves | Minimum witness strict500 | Best tested feasible waves | Best latency µs |
|---|---|---:|---|---:|---:|
| candidate | fixed | 18 | pass | 18 | 195.96 |
| candidate | permutation | 12 | fail | 13 | 155.14 |
| RR.native.even0 | fixed | 15 | pass | 15 | 175.62 |
| RR.native.even0 | permutation | 15 | pass | 15 | 171.04 |
| RR.native.even1 | fixed | 14 | pass | 14 | 165.92 |
| RR.native.even1 | permutation | 14 | pass | 14 | 163.9 |
| RR.native.even2 | fixed | 13 | pass | 13 | 156.22 |
| RR.native.even2 | permutation | 13 | pass | 13 | 154.34 |
| RR.native.even3 | fixed | 12 | fail | 13 | 155.14 |
| RR.native.even3 | permutation | 12 | fail | 13 | 155.14 |
| RR.native.even4 | fixed | 13 | pass | 13 | 155.68 |
| RR.native.even4 | permutation | 13 | pass | 13 | 156.22 |
| RR.native.even5 | fixed | 14 | pass | 14 | 165.38 |
| RR.native.even5 | permutation | 14 | pass | 14 | 165.92 |
| RR.native.even6 | fixed | 15 | pass | 15 | 175.62 |
| RR.native.even6 | permutation | 15 | pass | 15 | 175.62 |
| RR.hybridShared.even0 | fixed | 19 | pass | 19 | 205.66 |
| RR.hybridShared.even0 | permutation | 18 | pass | 18 | 198.52 |
| RR.hybridShared.even1 | fixed | 17 | pass | 17 | 194.48 |
| RR.hybridShared.even1 | permutation | 17 | pass | 17 | 188.82 |
| RR.hybridShared.even2 | fixed | 16 | pass | 16 | 184.78 |
| RR.hybridShared.even2 | permutation | 16 | pass | 16 | 179.12 |
| RR.hybridShared.even3 | fixed | 15 | pass | 15 | 175.08 |
| RR.hybridShared.even3 | permutation | 15 | pass | 15 | 169.42 |
| RR.hybridShared.even4 | fixed | 14 | pass | 14 | 165.38 |
| RR.hybridShared.even4 | permutation | 14 | pass | 14 | 159.72 |
| RR.hybridShared.even5 | fixed | 15 | pass | 15 | 175.08 |
| RR.hybridShared.even5 | permutation | 13 | fail | 14 | 159.72 |
| RR.hybridShared.even6 | fixed | 16 | pass | 16 | 177.64 |
| RR.hybridShared.even6 | permutation | 12 | fail | 13 | 155.14 |

Independent labelled-order brute force passed 54 small cases. A separate finite2slot recurrence matched every tested minimum and bounded one-extra-wave trace, including all latch-deadline violations.

The DP proves minimum cyclic slots (fixed0123) or distinct-plane waves (permutation) for the declared whole-row macro contract. The timing compiler additionally tests one full cyclic rotation before each useful fixed slot, or every one-wave split for permutation; best tested latency is not a general timing optimum. A12wave full-useful chain atcopy500 is universally infeasible in this cache-only model, so a feasible13wave witness is wave-optimal under that timing contract.

Native global RR uses E even=[012][301], odd=[230][123], and shared=[012][301][230][123]. The hybrid shared-geometry rows are a separately labelled stronger control with candidate-style shared patterns, not the native RR layout. All seven phase compositions are covered.

Pre-ECC drop and programmable source order remain conditional capabilities. Fixed-source results remove source permutation only; they still require same-block independent plane addresses and whole-row owner switching already assumed by the IPR model. These are service-envelope/compiler witnesses, not measured hardware or fullFFN gains.
