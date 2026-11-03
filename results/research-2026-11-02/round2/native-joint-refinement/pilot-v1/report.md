# Exact native-objective refinement: bounded no-go

Two TRAIN-only pilot cases; two standard families, two seeds and 2,048 proposal attempts per fit. All incumbents retained. Same attempt cap does not mean equal CPU or native-query work.

| Case | Native simple-menu mean (us) | Native unrestricted mean (us) | Relative mean gap | Relative P99 gap |
|---|---:|---:|---:|---:|
| granite4tiny.L0.input1 | 572.464039 | 562.670570 | 1.710757% | 0.466912% |
| deepseek.L0.input4 | 1966.475625 | 1959.950031 | 0.331842% | 0.134762% |

Seven of eight fits retain the original mean exactly. Only Deep unrestricted seed20261106 improves, by 9.62890625 ns (0.000491% of its initial mean). No first3L/full-model expansion and no heldout access. This bounds the usefulness of this finite refinement, not the global placement optimum.

Independent audit recomputed 8,192 initial/final costs and 8,192 bounds, all exact; all saved rows, offsets, jobs, page counts, row intervals, physical addresses and family invariants passed. The auditor adds four ReferenceGraph checks to the root sixteen checks.
