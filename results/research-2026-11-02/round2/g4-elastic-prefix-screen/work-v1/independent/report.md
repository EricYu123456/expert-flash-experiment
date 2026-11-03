The bounded G4 L0 TRAIN512 elastic-prefix screen is sealed as a negative result. Runtime has exactly A/B for each expert; the nine omitted-high ranks are offline candidates. Independent integer calculations reproduced all 32,768 masks, row-ID coverage, short/long prefix views, physical interval nonoverlap, and the -36,214 ns aggregate per-expert A-minus-B regression. No native replay, heldout, or extra replica campaign was run.

| Quantity | Mean (ms) | P99 (ms) |
|---|---:|---:|
| Existing B native | 0.561099020 | 0.580695140 |
| Selected min maxchip work | 0.554446367 | 0.573058000 |
| Host + minimum work LB | 0.556446367 | 0.575058000 |
| Optimal first-start assignment LB | 0.557032477 | 0.575558000 |

Even assuming zero selector cost and no further queue/dependency gaps, the plain LB permits at most 0.829203% mean / 0.970757% P99 improvement relative to B. The stronger valid first-start LB reduces those ceilings to 0.724746% / 0.884654%. Both miss the 2% mean or 5% tail expansion gate. Static same-layout A/B and stronger B-plus-all-nine-A coordinate controls retain B. These finite-search observations are not a universal no-go theorem for all copy assignments.

The selected image copies 23202 complete rows = 35638272 useful coefficient bytes (33.987305 MiB), but uses only 512 extra resident waves / 2,048 pages = 32 MiB by consuming old padding. This is one layer, not all forty. Full-layer per-chip physical allocation grows from sixteen to twenty LSB blocks (192 waves/block), with the exact per-chip inventory in audit.json. Same-resident-budget whole-GU alternatives can hold 21 minimum-24-wave copies plus eight waves, or 22 copies if only useful coefficient bytes are compared; those are distinct budgets and were not used as weakened controls. Whole A+B duplication is service-equivalent to prefix sharing under the declared address-invariant native model; prefix sharing saves capacity, not a separate latency mechanism.

B prefix reads still drain all 85/42 page-fringe rows per high/low job, including appended values in discarded slots. A's eight jobs each drain 128 rows. Stored original row IDs and host scatter preserve all complete GU rows; Down, shared, full-input A8 barrier, and single-context behavior are unchanged. This is a synthetic necessary-work screen, not measured hardware or model quality evidence.
