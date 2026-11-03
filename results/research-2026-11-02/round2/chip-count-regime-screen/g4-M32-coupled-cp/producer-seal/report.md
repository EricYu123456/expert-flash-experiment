M32 standard coupled-coloring control: producer seal

G4 actual W8 TRAIN, L0, 32 chips / 8 channels (hardware scale sensitivity). No replicas; frozen shared rows and routed row multisets unchanged.

| Input copies | Strong finite control mean/P99 (us) | Coupled mean/P99 (us) | Mean / P99 reduction |
|---|---:|---:|---:|
| 1 | 326.997506 / 379.589140 | 299.342484 / 300.795000 | 8.457258% / 20.757743% |
| 4 | 347.699428 / 399.771700 | 320.484436 / 324.183000 | 7.827160% / 18.907967% |

Peak10 feasibility succeeded; all seven retained mean maps independently have exactly ten maximum readwaves in every TRAIN query. Mean solver status is FEASIBLE, not OPTIMAL. Native input1 selected incumbent4/GF; input4 reuses that exact map.

The primitive is ordinary zero-copy coloring, not a new algorithm. Search budgets differ from finite deterministic controls. The result motivates a fair matched-search control; it does not establish a publication claim, full-model benefit, or an implemented 32-PE AiF device.

The original native report compares with the weaker minimum_depth source. This report corrects attribution to the strongest StageB finite candidate. No original artifact was changed.

Producer verification: row-count coverage, shared identity, all512 chip-work coefficients, page/fringe billing, 32 nonoverlapping single-block inventories; saved q0 C++/Python trace equality from native runner. Independent review is separately owned by the architecture agent.
