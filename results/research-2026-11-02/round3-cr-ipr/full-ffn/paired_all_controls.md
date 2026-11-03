# Paired conditional FFN controls

TRAIN first3layers ×512inputs;32prompts. Genericcompiler exactlytiescandidate. Statistics with fewer than1536jointlyvalid events are subset-only, with infeasible queries retained.

| Candidate | Baseline | Joint events | Event mean gain | Three-layer P99 gain | Slower events |
|---|---|---:|---:|---:|---:|
| strict100.g16_equal.input1.GF.a0.candidate | strict100.g16_equal.input1.GF.a0.g8_identity | 533/1536 | 0.0000% | 0.0000% | 0 |
| strict100.g16_equal.input1.GF.a0.candidate | strict100.g16_equal.input1.GF.a0.g8_learned | 932/1536 | 0.0000% | 0.0000% | 0 |
| strict100.g16_equal.input1.DF.a0.candidate | strict100.g16_equal.input1.DF.a0.g8_identity | 533/1536 | 0.0000% | 0.0000% | 0 |
| strict100.g16_equal.input1.DF.a0.candidate | strict100.g16_equal.input1.DF.a0.g8_learned | 932/1536 | 0.0000% | 0.0000% | 0 |
| strict100.g16_equal.input4.GF.a0.candidate | strict100.g16_equal.input4.GF.a0.g8_identity | 533/1536 | 0.0000% | 0.0000% | 0 |
| strict100.g16_equal.input4.GF.a0.candidate | strict100.g16_equal.input4.GF.a0.g8_learned | 932/1536 | 0.0000% | 0.0000% | 0 |
| strict100.g16_equal.input4.DF.a0.candidate | strict100.g16_equal.input4.DF.a0.g8_identity | 533/1536 | 0.0000% | 0.0000% | 0 |
| strict100.g16_equal.input4.DF.a0.candidate | strict100.g16_equal.input4.DF.a0.g8_learned | 932/1536 | 0.0000% | 0.0000% | 0 |
| strict100.g12_cyclic.input1.GF.a0.candidate | strict100.g12_cyclic.input1.GF.a0.g8_identity | 533/1536 | 0.0062% | -0.0192% | 100 |
| strict100.g12_cyclic.input1.GF.a0.candidate | strict100.g12_cyclic.input1.GF.a0.g8_learned | 932/1536 | 0.0128% | 0.0084% | 96 |
| strict100.g12_cyclic.input1.DF.a0.candidate | strict100.g12_cyclic.input1.DF.a0.g8_identity | 533/1536 | 0.0062% | -0.0192% | 100 |
| strict100.g12_cyclic.input1.DF.a0.candidate | strict100.g12_cyclic.input1.DF.a0.g8_learned | 932/1536 | 0.0128% | 0.0084% | 96 |
| strict100.g12_cyclic.input4.GF.a0.candidate | strict100.g12_cyclic.input4.GF.a0.g8_identity | 533/1536 | 0.0056% | -0.0188% | 100 |
| strict100.g12_cyclic.input4.GF.a0.candidate | strict100.g12_cyclic.input4.GF.a0.g8_learned | 932/1536 | 0.0126% | 0.0082% | 96 |
| strict100.g12_cyclic.input4.DF.a0.candidate | strict100.g12_cyclic.input4.DF.a0.g8_identity | 533/1536 | 0.0056% | -0.0188% | 100 |
| strict100.g12_cyclic.input4.DF.a0.candidate | strict100.g12_cyclic.input4.DF.a0.g8_learned | 932/1536 | 0.0126% | 0.0082% | 96 |
| cold_reset500.g16_equal.input1.GF.a0.candidate | cold_reset500.g16_equal.input1.GF.a0.g8_identity | 1536/1536 | 5.4282% | 10.2446% | 0 |
| cold_reset500.g16_equal.input1.GF.a0.candidate | cold_reset500.g16_equal.input1.GF.a0.g8_learned | 1536/1536 | 2.7560% | 8.2261% | 0 |
| cold_reset500.g16_equal.input1.DF.a0.candidate | cold_reset500.g16_equal.input1.DF.a0.g8_identity | 1536/1536 | 5.4282% | 10.2446% | 0 |
| cold_reset500.g16_equal.input1.DF.a0.candidate | cold_reset500.g16_equal.input1.DF.a0.g8_learned | 1536/1536 | 2.7560% | 8.2261% | 0 |
| cold_reset500.g16_equal.input4.GF.a0.candidate | cold_reset500.g16_equal.input4.GF.a0.g8_identity | 1536/1536 | 5.2344% | 9.9810% | 0 |
| cold_reset500.g16_equal.input4.GF.a0.candidate | cold_reset500.g16_equal.input4.GF.a0.g8_learned | 1536/1536 | 2.6517% | 7.9647% | 0 |
| cold_reset500.g16_equal.input4.DF.a0.candidate | cold_reset500.g16_equal.input4.DF.a0.g8_identity | 1536/1536 | 5.2344% | 9.9810% | 0 |
| cold_reset500.g16_equal.input4.DF.a0.candidate | cold_reset500.g16_equal.input4.DF.a0.g8_learned | 1536/1536 | 2.6517% | 7.9647% | 0 |
| cold_reset500.g12_cyclic.input1.GF.a0.candidate | cold_reset500.g12_cyclic.input1.GF.a0.g8_identity | 1536/1536 | 3.6437% | 9.1789% | 318 |
| cold_reset500.g12_cyclic.input1.GF.a0.candidate | cold_reset500.g12_cyclic.input1.GF.a0.g8_learned | 1536/1536 | 1.9657% | 6.0238% | 230 |
| cold_reset500.g12_cyclic.input1.DF.a0.candidate | cold_reset500.g12_cyclic.input1.DF.a0.g8_identity | 1536/1536 | 3.6437% | 9.1789% | 318 |
| cold_reset500.g12_cyclic.input1.DF.a0.candidate | cold_reset500.g12_cyclic.input1.DF.a0.g8_learned | 1536/1536 | 1.9657% | 6.0238% | 230 |
| cold_reset500.g12_cyclic.input4.GF.a0.candidate | cold_reset500.g12_cyclic.input4.GF.a0.g8_identity | 1536/1536 | 3.4763% | 8.7724% | 318 |
| cold_reset500.g12_cyclic.input4.GF.a0.candidate | cold_reset500.g12_cyclic.input4.GF.a0.g8_learned | 1536/1536 | 1.8777% | 5.7860% | 230 |
| cold_reset500.g12_cyclic.input4.DF.a0.candidate | cold_reset500.g12_cyclic.input4.DF.a0.g8_identity | 1536/1536 | 3.4763% | 8.7724% | 318 |
| cold_reset500.g12_cyclic.input4.DF.a0.candidate | cold_reset500.g12_cyclic.input4.DF.a0.g8_learned | 1536/1536 | 1.8777% | 5.7860% | 230 |
| strict500.g16_equal.input1.GF.a0.candidate_IPR_uniform_split10 | strict500.g16_equal.input1.GF.a0.matching_layout_IPR_uniform_split10 | 1536/1536 | 3.8687% | 5.9722% | 0 |
| strict500.g16_equal.input1.GF.a0.candidate_IPR_uniform_split10 | strict500.g16_equal.input1.GF.a0.matching_layout_IPR_fill_only | 1465/1536 | 2.1322% | 3.9767% | 0 |
| strict500.g16_equal.input1.GF.a0.candidate_IPR_uniform_split10 | strict500.g16_equal.input1.GF.a0.RR_IPR_uniform_split10 | 1536/1536 | 1.6348% | 3.7671% | 0 |
| strict500.g16_equal.input1.GF.a0.candidate_IPR_uniform_split10 | cold_reset500.g16_equal.input1.GF.a0.candidate | 1536/1536 | 3.3930% | 3.3930% | 0 |
| strict500.g16_equal.input1.GF.a0.candidate_IPR_uniform_split10 | cold_reset500.g16_equal.input1.GF.a0.sameWL_matching_keep_hot | 1536/1536 | 2.1912% | 4.0327% | 0 |
| strict500.g16_equal.input1.DF.a0.candidate_IPR_uniform_split10 | strict500.g16_equal.input1.DF.a0.matching_layout_IPR_uniform_split10 | 1536/1536 | 3.8687% | 5.9722% | 0 |
| strict500.g16_equal.input1.DF.a0.candidate_IPR_uniform_split10 | strict500.g16_equal.input1.DF.a0.matching_layout_IPR_fill_only | 1465/1536 | 2.1322% | 3.9767% | 0 |
| strict500.g16_equal.input1.DF.a0.candidate_IPR_uniform_split10 | strict500.g16_equal.input1.DF.a0.RR_IPR_uniform_split10 | 1536/1536 | 1.6348% | 3.7671% | 0 |
| strict500.g16_equal.input1.DF.a0.candidate_IPR_uniform_split10 | cold_reset500.g16_equal.input1.DF.a0.candidate | 1536/1536 | 3.3930% | 3.3930% | 0 |
| strict500.g16_equal.input1.DF.a0.candidate_IPR_uniform_split10 | cold_reset500.g16_equal.input1.DF.a0.sameWL_matching_keep_hot | 1536/1536 | 2.1912% | 4.0327% | 0 |
| strict500.g16_equal.input4.GF.a0.candidate_IPR_uniform_split10 | strict500.g16_equal.input4.GF.a0.matching_layout_IPR_uniform_split10 | 1536/1536 | 3.7875% | 5.8496% | 0 |
| strict500.g16_equal.input4.GF.a0.candidate_IPR_uniform_split10 | strict500.g16_equal.input4.GF.a0.matching_layout_IPR_fill_only | 1465/1536 | 2.0867% | 3.8934% | 0 |
| strict500.g16_equal.input4.GF.a0.candidate_IPR_uniform_split10 | strict500.g16_equal.input4.GF.a0.RR_IPR_uniform_split10 | 1536/1536 | 1.5997% | 3.6880% | 0 |
| strict500.g16_equal.input4.GF.a0.candidate_IPR_uniform_split10 | cold_reset500.g16_equal.input4.GF.a0.candidate | 1536/1536 | 3.3214% | 3.3214% | 0 |
| strict500.g16_equal.input4.GF.a0.candidate_IPR_uniform_split10 | cold_reset500.g16_equal.input4.GF.a0.sameWL_matching_keep_hot | 1536/1536 | 2.1444% | 3.9482% | 0 |
| strict500.g16_equal.input4.DF.a0.candidate_IPR_uniform_split10 | strict500.g16_equal.input4.DF.a0.matching_layout_IPR_uniform_split10 | 1536/1536 | 3.7875% | 5.8496% | 0 |
| strict500.g16_equal.input4.DF.a0.candidate_IPR_uniform_split10 | strict500.g16_equal.input4.DF.a0.matching_layout_IPR_fill_only | 1465/1536 | 2.0867% | 3.8934% | 0 |
| strict500.g16_equal.input4.DF.a0.candidate_IPR_uniform_split10 | strict500.g16_equal.input4.DF.a0.RR_IPR_uniform_split10 | 1536/1536 | 1.5997% | 3.6880% | 0 |
| strict500.g16_equal.input4.DF.a0.candidate_IPR_uniform_split10 | cold_reset500.g16_equal.input4.DF.a0.candidate | 1536/1536 | 3.3214% | 3.3214% | 0 |
| strict500.g16_equal.input4.DF.a0.candidate_IPR_uniform_split10 | cold_reset500.g16_equal.input4.DF.a0.sameWL_matching_keep_hot | 1536/1536 | 2.1444% | 3.9482% | 0 |
| strict500.g12_cyclic.input1.GF.a0.candidate_IPR_uniform_split10 | strict500.g12_cyclic.input1.GF.a0.matching_layout_IPR_uniform_split10 | 1536/1536 | 3.7808% | 5.5484% | 0 |
| strict500.g12_cyclic.input1.GF.a0.candidate_IPR_uniform_split10 | strict500.g12_cyclic.input1.GF.a0.matching_layout_IPR_fill_only | 1465/1536 | 2.0826% | 3.5620% | 0 |
| strict500.g12_cyclic.input1.GF.a0.candidate_IPR_uniform_split10 | strict500.g12_cyclic.input1.GF.a0.RR_IPR_uniform_split10 | 1536/1536 | 1.5968% | 3.0902% | 0 |
| strict500.g12_cyclic.input1.GF.a0.candidate_IPR_uniform_split10 | cold_reset500.g12_cyclic.input1.GF.a0.candidate | 1536/1536 | 3.3155% | 3.2808% | 0 |
| strict500.g12_cyclic.input1.GF.a0.candidate_IPR_uniform_split10 | cold_reset500.g12_cyclic.input1.GF.a0.sameWL_matching_keep_hot | 1536/1536 | 2.1406% | 3.5950% | 0 |
| strict500.g12_cyclic.input1.DF.a0.candidate_IPR_uniform_split10 | strict500.g12_cyclic.input1.DF.a0.matching_layout_IPR_uniform_split10 | 1536/1536 | 3.7808% | 5.5484% | 0 |
| strict500.g12_cyclic.input1.DF.a0.candidate_IPR_uniform_split10 | strict500.g12_cyclic.input1.DF.a0.matching_layout_IPR_fill_only | 1465/1536 | 2.0826% | 3.5620% | 0 |
| strict500.g12_cyclic.input1.DF.a0.candidate_IPR_uniform_split10 | strict500.g12_cyclic.input1.DF.a0.RR_IPR_uniform_split10 | 1536/1536 | 1.5968% | 3.0902% | 0 |
| strict500.g12_cyclic.input1.DF.a0.candidate_IPR_uniform_split10 | cold_reset500.g12_cyclic.input1.DF.a0.candidate | 1536/1536 | 3.3155% | 3.2808% | 0 |
| strict500.g12_cyclic.input1.DF.a0.candidate_IPR_uniform_split10 | cold_reset500.g12_cyclic.input1.DF.a0.sameWL_matching_keep_hot | 1536/1536 | 2.1406% | 3.5950% | 0 |
| strict500.g12_cyclic.input4.GF.a0.candidate_IPR_uniform_split10 | strict500.g12_cyclic.input4.GF.a0.matching_layout_IPR_uniform_split10 | 1536/1536 | 3.7062% | 5.3670% | 0 |
| strict500.g12_cyclic.input4.GF.a0.candidate_IPR_uniform_split10 | strict500.g12_cyclic.input4.GF.a0.matching_layout_IPR_fill_only | 1465/1536 | 2.0409% | 3.4184% | 0 |
| strict500.g12_cyclic.input4.GF.a0.candidate_IPR_uniform_split10 | strict500.g12_cyclic.input4.GF.a0.RR_IPR_uniform_split10 | 1536/1536 | 1.5646% | 3.0054% | 0 |
| strict500.g12_cyclic.input4.GF.a0.candidate_IPR_uniform_split10 | cold_reset500.g12_cyclic.input4.GF.a0.candidate | 1536/1536 | 3.2498% | 3.2139% | 0 |
| strict500.g12_cyclic.input4.GF.a0.candidate_IPR_uniform_split10 | cold_reset500.g12_cyclic.input4.GF.a0.sameWL_matching_keep_hot | 1536/1536 | 2.0977% | 3.4532% | 0 |
| strict500.g12_cyclic.input4.DF.a0.candidate_IPR_uniform_split10 | strict500.g12_cyclic.input4.DF.a0.matching_layout_IPR_uniform_split10 | 1536/1536 | 3.7062% | 5.3670% | 0 |
| strict500.g12_cyclic.input4.DF.a0.candidate_IPR_uniform_split10 | strict500.g12_cyclic.input4.DF.a0.matching_layout_IPR_fill_only | 1465/1536 | 2.0409% | 3.4184% | 0 |
| strict500.g12_cyclic.input4.DF.a0.candidate_IPR_uniform_split10 | strict500.g12_cyclic.input4.DF.a0.RR_IPR_uniform_split10 | 1536/1536 | 1.5646% | 3.0054% | 0 |
| strict500.g12_cyclic.input4.DF.a0.candidate_IPR_uniform_split10 | cold_reset500.g12_cyclic.input4.DF.a0.candidate | 1536/1536 | 3.2498% | 3.2139% | 0 |
| strict500.g12_cyclic.input4.DF.a0.candidate_IPR_uniform_split10 | cold_reset500.g12_cyclic.input4.DF.a0.sameWL_matching_keep_hot | 1536/1536 | 2.0977% | 3.4532% | 0 |

Everytrace fails64B under the declared page-completion burst model. Common-GU chain controls peak at128B; some input4/DF separate-GU andlegacywhole-row cases require256B. Validity must follow each recorded FIFOcount. Realistic timing/source/FIFO support remains conditional. Native measured latency, generalization and publication readiness are not established.

Readshaping is bounded by the fixed9.7us cadence assumption; conditional extendedphase/cadence controls can be faster. Uniformone-split kernels are controls, not globalnativeoptima.
