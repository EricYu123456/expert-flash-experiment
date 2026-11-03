# Conditional finite-PB full-wave burst bound

Independent max-plus recurrence, closed form and event simulator agree in all 768 cases. This is a declared microarchitecture sensitivity, not a device measurement.

For N>=3 full waves: `2*c + (N-2)*540 <= 6040 ns`; fixed 0,1,2,3 drain attains this condition in the tested contract.

| Copy ns | Maximum strict full-wave burst |
|---:|---:|
| 0 | 13 |
| 100 | 12 |
| 320 | 12 |
| 321 | 11 |
| 500 | 11 |
| 1000 | 9 |
| 1940 | 6 |
| 1941 | 5 |
| 2000 | 5 |
| 3020 | 2 |
| 6040 | 2 |
| 6041 | 1 |

A twelve-wave burst admits c<=320ns; a six-wave burst admits c<=1940ns. Longer chains need a documented different buffering/streaming path or explicit cold restarts under this contract. Plane load balance alone does not prove continuous CR feasibility.

Both read-latch deadline and consumer buffering assumptions are explicit in execution.frozen.json. No change to any hardware is asserted.
