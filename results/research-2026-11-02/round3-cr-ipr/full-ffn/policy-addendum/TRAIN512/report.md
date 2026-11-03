# First-feasible PB-contract control

Samepolicy for every staticlayout:fill-only iflegal,thenfixedsplit,thenfirstlegalcut,thenexplicitcoldreset.No fullFFNcost lookup.

| Down | Baseline | Joint queries | Mean gain | P99 gain | Slower | Equal |
|---|---|---:|---:|---:|---:|---:|
| g16_equal | RR | 1536 | 0.31863% | 3.70661% | 378 | 533 |
| g16_equal | matching | 1536 | 2.03563% | 4.69079% | 0 | 71 |
| g12_cyclic | RR | 1536 | 0.31113% | 3.17638% | 378 | 533 |
| g12_cyclic | matching | 1536 | 1.98854% | 4.48381% | 0 | 71 |

Genericcompilerexactlytiescandidate. Alltimingsremainconditional. Fixed9.7us timerassumption is fragile:slightlyextendedreadphase/cadence mayeliminatePBfailureorbeatshaping; hardwaretimingunverified.
