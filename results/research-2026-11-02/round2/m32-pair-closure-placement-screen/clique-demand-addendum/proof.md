For C=[1, 4, 10, 20, 28, 29, 31, 35, 39, 41, 42, 48], all66pairs occur in original TRAIN; hence every one of its924six-subsets belongs to the frozen pair closure.

Assume every protected route has NAND peak<=10. On chip c with shared waves S_c, every six-subset has at most4-S_c X experts. Since4-S_c is1,2or3, any larger number of marked C experts could be extended to a violating six-subset. Thus sum_{e in C}X[e,c]<=4-S_c.

Summing all32chips gives at most12*1+17*2+3*3=55 X incidences. But12experts each require5X placements, so the same sum is60. Contradiction.

This applies only to the stated pair closure, coupledX5/O7 profile and fixed shared layout; no dev data or universal-router robustness claim. The earlier uniform469>=320 necessary screen remains inconclusive on its own.
