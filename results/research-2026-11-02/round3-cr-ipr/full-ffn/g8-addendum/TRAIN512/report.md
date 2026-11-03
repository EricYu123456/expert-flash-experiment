# Fixed-source g8 TRAIN control

G8 uses two static groups of32experts. Eachselectedexpert128GUrows/chip on8chips,12pages in3full0123waves; shared128rows/chip onall16chips. This requires no programmable PB source order forGU. Fitting is standard balanced swaps, not an optimizer novelty.

| Profile | Down | input copies | priority | map | complete / requested | meanus(validonly) | P99us |
|---|---|---:|---|---|---:|---:|---:|
| strict100 | g16_equal | 1 | GF | identity | 533/1536 | 420.1320 | 420.1320 |
| strict100 | g16_equal | 1 | GF | learned | 932/1536 | 420.1320 | 420.1320 |
| strict100 | g16_equal | 1 | DF | identity | 533/1536 | 420.1320 | 420.1320 |
| strict100 | g16_equal | 1 | DF | learned | 932/1536 | 420.1320 | 420.1320 |
| strict100 | g16_equal | 4 | GF | identity | 533/1536 | 429.6520 | 429.6520 |
| strict100 | g16_equal | 4 | GF | learned | 932/1536 | 429.6520 | 429.6520 |
| strict100 | g16_equal | 4 | DF | identity | 533/1536 | 429.6520 | 429.6520 |
| strict100 | g16_equal | 4 | DF | learned | 932/1536 | 429.6520 | 429.6520 |
| strict100 | g12_cyclic | 1 | GF | identity | 533/1536 | 430.2774 | 435.4920 |
| strict100 | g12_cyclic | 1 | GF | learned | 932/1536 | 430.7596 | 435.4920 |
| strict100 | g12_cyclic | 1 | DF | identity | 533/1536 | 430.2774 | 435.4920 |
| strict100 | g12_cyclic | 1 | DF | learned | 932/1536 | 430.7596 | 435.4920 |
| strict100 | g12_cyclic | 4 | GF | identity | 533/1536 | 439.4279 | 445.0120 |
| strict100 | g12_cyclic | 4 | GF | learned | 932/1536 | 439.9278 | 445.0120 |
| strict100 | g12_cyclic | 4 | DF | identity | 533/1536 | 439.4279 | 445.0120 |
| strict100 | g12_cyclic | 4 | DF | learned | 932/1536 | 439.9278 | 445.0120 |
| cold_reset500 | g16_equal | 1 | GF | identity | 1536/1536 | 467.4672 | 532.2520 |
| cold_reset500 | g16_equal | 1 | GF | learned | 1536/1536 | 454.6212 | 501.5320 |
| cold_reset500 | g16_equal | 1 | DF | identity | 1536/1536 | 467.4672 | 532.2520 |
| cold_reset500 | g16_equal | 1 | DF | learned | 1536/1536 | 454.6212 | 501.5320 |
| cold_reset500 | g16_equal | 4 | GF | identity | 1536/1536 | 476.5567 | 540.7000 |
| cold_reset500 | g16_equal | 4 | GF | learned | 1536/1536 | 463.9135 | 509.9800 |
| cold_reset500 | g16_equal | 4 | DF | identity | 1536/1536 | 476.5567 | 540.7000 |
| cold_reset500 | g16_equal | 4 | DF | learned | 1536/1536 | 463.9135 | 509.9800 |
| cold_reset500 | g12_cyclic | 1 | GF | identity | 1536/1536 | 469.5255 | 517.6420 |
| cold_reset500 | g12_cyclic | 1 | GF | learned | 1536/1536 | 461.4888 | 516.5545 |
| cold_reset500 | g12_cyclic | 1 | DF | identity | 1536/1536 | 469.5255 | 517.6420 |
| cold_reset500 | g12_cyclic | 1 | DF | learned | 1536/1536 | 461.4888 | 516.5545 |
| cold_reset500 | g12_cyclic | 4 | GF | identity | 1536/1536 | 478.1851 | 527.1620 |
| cold_reset500 | g12_cyclic | 4 | GF | learned | 1536/1536 | 470.3942 | 525.0025 |
| cold_reset500 | g12_cyclic | 4 | DF | identity | 1536/1536 | 478.1851 | 527.1620 |
| cold_reset500 | g12_cyclic | 4 | DF | learned | 1536/1536 | 470.3942 | 525.0025 |

Strict100 failures are retained as-1 inarrays and included in requested counts. Its valid-only cost is not an all-query result; comparisons must retain jointlyvalid counts and the remaining infeasibility. Cold-reset500 completes all1536queries in eachcase. All completedqueries have outputFIFOpeak128B:64B is invalid;128/256/2048B pass this posthoc audit.

No heldout or hardware measurement. Fullpaired comparison against the immutable primary candidate/matching arrays will be recorded separately after primary completion.
