# Conditional whole-row Down stage certificate

Under the declared cold common-release array/cache/ECC model, the pure Down stage optimum is **89.94µs** at copy500ns. This is a service-envelope certificate, not fullFFN or measured-hardware optimality.

Six routed1536×512 W8 matrices plus one shared1536×1024 matrix require6,291,456bytes =6MiB. Each chip first supplies useful ECC data no earlier than28.5µs. By timeT,16chips can supply at most16×6.4GB/s×(T−28.5µs) useful bytes. ThereforeT≥28.5µs+61.44µs=89.94µs. This aggregate proof does not require summing startup over multiple jobs.

The original job-count proof also checks out. ForT<89.94µs, a job has fewer than393,216 useful bytes. Complete rows imply at most767routed rows or383shared rows/job, hence at least18+5=23jobs. If whole-job intervals on a chip are disjoint, total work is at least23×28.5+983.04=1638.54µs, exceeding16×89.94=1439.04µs. The margin is199.5µs. The aggregate proof avoids this extra serialization premise.

| Copy ns | First ECC startup µs | Lower bound µs | Sealed16-job witness µs |
|---:|---:|---:|---:|
| 0 | 28.000 | 89.440 | 89.440 |
| 100 | 28.100 | 89.540 | 89.540 |
| 500 | 28.500 | 89.940 | 89.940 |

Attainment uses12routedjobs on six2-chipgroups plus4sharedjobs on fourchips. Everyjob reads393,216bytes in24pages/sixfullsame-WLwaves, with onecoldstart and fiveCRwaves. Fixed0123 and finite2weightlatches pass at copy0/100/500; no drop, IPR, extra inputcontext, FIFO orhold is used. Every expert receives all1536 complete output rows. A resident perfectmatching is required; it is not guaranteed for every route.

Two reduced exhaustive whole-row allocation oracles enumerate arbitrary unequal row counts and multiple owners/chip for3matrices on4chips, at startup0/1/2/7units. The balanced H4 geometry attains its aggregate bound; the H3 geometry demonstrates that row granularity and additional jobs can make the bound unattainable. These are abstract arithmetic sanity checks, not native timing simulations.

The stage starts with all private inputs available and no prior Down service. Command/input/output transfer, GU calendars, extra MAC latency and other shared resources are excluded. If a fullFFN prefetches Down weights or begins some experts before the last input is ready, this common-release stage certificate cannot be applied to its remaining Down interval or added to a GU lower bound. First-ready28us/copy500ns remain conditional model parameters.
