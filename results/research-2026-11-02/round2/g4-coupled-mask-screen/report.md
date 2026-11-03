The G4 L0 W8 TRAIN coupled-coloring control improves mean native FFN service by3.34% over the preceding strongest joint/free-label incumbent. Almost all of this improvement is already available in the older aligned-omit construction. Allowing arbitrary omitted chips adds only0.01375% mean improvement in this bounded screen. Do not attribute the3.34% to a new mask mechanism.

| Control,512 TRAIN queries | Mean native FFN service | P99 native FFN service |
|---|---:|---:|
| Frozen strong cyclic/whole-row | 583.041559 us | 712.073400 us |
| Same-geometry joint/free-label refit | 581.530629 us | 659.374680 us |
| Aligned omit,seed20261103 | 562.241027 us | 604.199860 us |
| Aligned omit,seed20261104 | 562.179563 us | 606.045040 us |
| Arbitrary omit,seed20261103 | 562.102242 us | 580.350100 us |
| Arbitrary omit,seed20261104 | 562.601023 us | 580.788940 us |

Best mean arbitrary omit versus the preceding strongest control is3.34091%; versus best mean aligned omit it is0.01375%. Comparing the best observed tail in each two-seed family gives3.94733% lower P99 for arbitrary omit. The latter is a TRAIN-only family envelope; the mean objective selected the displayed best-mean candidates, and no P99 fit or external validation was performed. These are single-layer synthetic FFN-service times, not whole-model or device speed.

The older joint-depth source already constructed joined NAND demand `5 chips×3 waves +11×2` per expert, but constrained Down's four omitted chips to one of four aligned groups. It therefore allowed2,832 of the4,368 possible five-high masks;1,536 masks hit all four groups and cannot avoid any old omitted group. Arbitrary omission allows all4,368 masks, each with330 choices of four omitted chips among the remaining11. This combinatorial difference is independently enumerated. It is a standard coupled-coloring/declustering freedom, not a novel algorithm.

The two families start from the same route-blind category assignment for each of the same two seeds. Each receives20,000 proposal attempts plus at most four deterministic greedy sweeps. The aligned control can exchange high/low bits and change the aligned omitted group. The free control can exchange any different pair of categories. The free control has fewer valid annealing proposals (13,735/13,886 versus20,000); valid counts, accepted moves and greedy truncation are recorded, so equal attempt budgets are not presented as equal convergence. The objective is exact maximum joined mandatory chip context work, including fixed shared GU/Down, full inputs, commands and page-fringe outputs, rather than the old phase-additive scalar.

GU uses six83-row high jobs, three82-row high jobs and seven40-row low jobs:1,024 complete rows,25waves/100pages. Down uses twelve128-row jobs and four omitted chips:1,536 complete rows,12waves/48pages. Omitted chips are a subset of GU's nine high chips, leaving exactly five chips with joined three-wave demand. Every weight row occurs once; no columns, quantization, context, caching or instruction semantics are changed. Full-vector A8 and whole-expert GU readiness remain barriers in native replay.

The best free candidate has mean necessary joined chip work556.348605us and actual native service562.102242us; their difference is not counted as placement speed. Shared rows stay frozen. Native replay uses the same backend and v5 input-one/250ns contract as every original control. All four candidates pass three direct Python graph comparisons each. Independent row, page, wave, objective and physical inventory checks are recorded in independent_checks.json.

Coefficient payload remains155,713,536bytes. Programmed/reserved page storage rises from155,713,536 to159,907,840bytes:4MiB additional page padding in this layer. GU read work rises from24 to25waves/expert. Each chip's total layer inventory still fits one192-wave block, so both old and new reserve201,326,592LSB block bytes in this model. This is no physical-capacity improvement, and per-chip useful inventory is allowed to vary. The best free candidate's useful inventory ranges from9,321,472 to10,085,376bytes/chip; full vectors are recorded rather than imposing an artificial average cap.

Total screening runtime was2.90seconds. No model expansion, heldout access or frozen-plan mutation occurred. The arbitrary-row MIP's valid510.718500us bound remains below this improved incumbent, leaving an unresolved9.14135% upper bound on further arbitrary-geometry improvement. The MIP did not certify a placement attaining that bound. The main practical result is a stronger known coupled-geometry control for future compiler comparisons; the incremental arbitrary-omit mean gain does not justify a larger search in this screen.
