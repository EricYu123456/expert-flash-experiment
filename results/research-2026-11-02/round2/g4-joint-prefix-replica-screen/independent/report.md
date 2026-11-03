The fixed GU-layout3 + Down-graph14 joint model passes independent correctness checks and the mean stage gate, with strong controls still pending.

| Case | Mean (ms) | Empirical P99 (ms) |
|---|---:|---:|
| Frozen CP0107 B | 0.561099020 | 0.580695140 |
| GU-only greedy | 0.551417348 | 0.577208000 |
| Down-only graph14 | 0.550165613 | 0.576531660 |
| Joint fixed image/policy | 0.549021773 | 0.578797500 |

The joint sensitivity improves mean 2.152427% and P99 0.326788% versus B. Its mean increment versus Down-only is only 0.207908%; its P99 is worse (-0.393012% change). Branch gains cannot be added. There are 36 regressing queries, with maximum regression 29,118 ns.

Independent checks rederived all per-bit costs, verified 512 exact rational LP box-dual certificates with Python integer arithmetic, recomputed all 4,096 policy row-job costs and 512 chosen chip vectors, checked every routed expert GU1024/Down1536 consumed-row map, and matched all six representative native event traces node by node. Actual GU computed totals may exceed1024; the unchanged compiler guard uses computed totals while a separate row-ID map proves logical consumption. Original B's512 costs/bounds bridge exactly. CP32 reports six OPTIMAL and26 FEASIBLE cases, mean feasible improvement803.4375 ns, max1692 ns; no CP bits replaced nativegreedy512 and nonzero gaps are retained.

The images add35,638,272 GU useful bytes plus25,165,824 Down bytes =60,804,096 bytes (57.987305 MiB), but programmed growth is896waves=56MiB due reused GU padding. The static joint BFD allocation is32LSBblocks/384MiB versus B16/192MiB; counts cover full routed/shared FFN coefficients in this layer, exclude attention/dense/FTL/spares/hostmetadata, and do not demonstrate device programming. The separate independent-contract audit checks row coverage, all view aliases and this physical inventory.

This is synthetic ordinary-read TRAIN L0 timing with zero selector/scatter/software overhead. It passes only the mean resource-allocation gate for deciding whether to run production selector and strong zero-copy/same-budget wholeGU+Down/partial-RDA controls. It does not establish a new algorithm, numerical quality, actual ABI execution, full-model speedup or heldout behavior. No such expansion was run in this stage.
