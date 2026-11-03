| Chunk | Q4 GGUF PPL | W8A8 PPL | ΔPPL | ΔPPL % | ΔNLL |
|---|---:|---:|---:|---:|---:|
| 00 | 12.9257 | 13.0165 | +0.0908 | +0.702% | +0.006998 |
| 01 | 15.3074 | 15.8106 | +0.5032 | +3.287% | +0.032345 |
| 02 | 10.6334 | 10.6937 | +0.0603 | +0.567% | +0.005658 |
| 03 | 17.7627 | 17.2307 | -0.5320 | -2.995% | -0.030409 |
| 04 | 10.8197 | 10.2599 | -0.5598 | -5.174% | -0.053127 |
| 05 | 54.8916 | 57.4198 | +2.5282 | +4.606% | +0.045029 |
| 06 | 23.6864 | 23.8396 | +0.1531 | +0.647% | +0.006444 |
| 07 | 10.0200 | 9.9655 | -0.0544 | -0.543% | -0.005448 |
| Pooled | 16.4136 | 16.4289 | +0.0154 | +0.094% | +0.000936 |

Each chunk scores127 fixed next tokens. Positive deltas mean higher W8 NLL/PPL. PooledPPL is exp(totalNLL/1016), not an arithmetic mean of chunk PPL. These eight chunks provide descriptive sensitivity relative to Q4 GGUF; no inferential statistics, original-BF16 equivalence or formal task-accuracy claim.
