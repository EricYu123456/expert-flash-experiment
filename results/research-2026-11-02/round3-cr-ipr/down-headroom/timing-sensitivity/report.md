# Fixed-map Down work timing sensitivity

This is a chip-owner work relaxation, not fullFFN replay. All maps remain frozen and no timing case is used to reselect them.

| Timing model | GU | input copies | critical-work reduction | P99-work reduction |
|---|---|---:|---:|---:|
| nominal_CR9700_copy500 | candidate | 1 | 10.773% | -6.756% |
| nominal_CR9700_copy500 | candidate | 4 | 11.248% | -6.306% |
| nominal_CR9700_copy500 | RR | 1 | 10.773% | -6.756% |
| nominal_CR9700_copy500 | RR | 4 | 11.248% | -6.306% |
| nominal_CR9700_copy500 | g8_fixed | 1 | 12.865% | 24.360% |
| nominal_CR9700_copy500 | g8_fixed | 4 | 13.037% | 24.308% |
| ordinary_28us_every_wave_copy500 | candidate | 1 | 7.868% | -9.492% |
| ordinary_28us_every_wave_copy500 | candidate | 4 | 8.208% | -9.173% |
| ordinary_28us_every_wave_copy500 | RR | 1 | 7.868% | -9.492% |
| ordinary_28us_every_wave_copy500 | RR | 4 | 8.208% | -9.173% |
| ordinary_28us_every_wave_copy500 | g8_fixed | 1 | 24.595% | 36.674% |
| ordinary_28us_every_wave_copy500 | g8_fixed | 4 | 24.595% | 36.569% |
| conditional_CR10240_copy500_recycle6040 | candidate | 1 | 10.773% | -6.756% |
| conditional_CR10240_copy500_recycle6040 | candidate | 4 | 11.248% | -6.306% |
| conditional_CR10240_copy500_recycle6040 | RR | 1 | 10.773% | -6.756% |
| conditional_CR10240_copy500_recycle6040 | RR | 4 | 11.248% | -6.306% |
| conditional_CR10240_copy500_recycle6040 | g8_fixed | 1 | 12.865% | 24.360% |
| conditional_CR10240_copy500_recycle6040 | g8_fixed | 4 | 13.037% | 24.308% |
| nominal_CR9700_copy0 | candidate | 1 | 10.709% | -6.817% |
| nominal_CR9700_copy0 | candidate | 4 | 11.189% | -6.361% |
| nominal_CR9700_copy0 | RR | 1 | 10.709% | -6.817% |
| nominal_CR9700_copy0 | RR | 4 | 11.189% | -6.361% |
| nominal_CR9700_copy0 | g8_fixed | 1 | 12.992% | 24.516% |
| nominal_CR9700_copy0 | g8_fixed | 4 | 13.163% | 24.461% |

The ordinary model pays 28us for every wave. CR10.24us is conditional cadence sensitivity with recycle6040ns, not evidence that hardware permits changing the phase. Copy0 removes modeled copy cost. The comparisons show whether the same static resource partition reduces mandatory Down work under each model; they do not establish corresponding native fullFFN speedups.
