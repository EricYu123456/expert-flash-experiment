# Equal-payload fixed-shared control (full)

The same303MiB/544residentjobs and exactaddresses are retained. Onlyexecution isrestricted: S0 ongroup6,S1 ongroup7; routedexperts cap1then2 ongroups0..5. Positive mobilityreduction means thejointdispatch isfaster.

| GU | array | inputcopies | fixedmean µs | jointmean µs | jointmean reduction vsfixed | eventP99 reduction | cap2events |
|---|---|---:|---:|---:|---:|---:|---:|
| candidate | cr | 1 | 263.971 | 254.712 | 3.507% | 25.297% | 166 |
| candidate | cr | 4 | 268.082 | 258.856 | 3.442% | 24.927% | 166 |
| RR | cr | 1 | 265.974 | 256.793 | 3.452% | 20.517% | 166 |
| RR | cr | 4 | 270.771 | 261.739 | 3.336% | 19.979% | 166 |
| g8_fixed | cr | 1 | 290.618 | 282.957 | 2.636% | 14.472% | 166 |
| g8_fixed | cr | 4 | 295.189 | 287.608 | 2.568% | 14.330% | 166 |
| g8_fixed | ordinary | 1 | 595.318 | 579.682 | 2.626% | 11.468% | 166 |
| g8_fixed | ordinary | 4 | 599.888 | 584.333 | 2.593% | 11.414% | 166 |

Completed 12288events; allfinitePB/FIFO128valid=True. Everyregression,aligned3layerP99,resourcecount,andfailureremainsinthereport. Q0andfirstcap2tracesinclude exactselectedresidentjobs.

Screenindices0..63 are solelyaPB/FIFOcorrectnessgate. All8GFaxes arefrozen; no gain-basedselection orfitting. The fullcomparison reusesonlymatchedjoint8TRAINarrays. This is a posthoc standardresourcepartition/sharedmobilityablation, notnewsolver/CBC/hardwareorindependentvalidation. ConditionalGUfeaturesandnativeg8CR/ordinarycontractsareinheritedunchanged. No prospective data/outcomes are read.
