# First-feasible PB-contract control

Samepolicy for every staticlayout:fill-only iflegal,thenfixedsplit,thenfirstlegalcut,thenexplicitcoldreset.No fullFFNcost lookup.

| Down | Baseline | Joint queries | Mean gain | P99 gain | Slower | Equal |
|---|---|---:|---:|---:|---:|---:|
| g16_equal | RR | 24 | 0.27048% | 3.42315% | 5 | 8 |
| g16_equal | matching | 24 | 1.95962% | 4.57580% | 0 | 0 |
| g12_cyclic | RR | 24 | 0.26594% | 3.05227% | 5 | 8 |
| g12_cyclic | matching | 24 | 1.92724% | 3.90969% | 0 | 0 |

Genericcompilerexactlytiescandidate. Alltimingsremainconditional. Fixed9.7us timerassumption is fragile:slightlyextendedreadphase/cadence mayeliminatePBfailureorbeatshaping; hardwaretimingunverified.
