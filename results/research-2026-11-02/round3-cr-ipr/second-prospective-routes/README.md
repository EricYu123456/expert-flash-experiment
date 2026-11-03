# Second authored diagnostic

All16newprompts and16comparisonaxes were frozen before this model execution. No prompts, maps, fits, profiles or winners were selected or replaced after results.

This is a second authored diagnostic, not a representative benchmark, task-quality result or extrapolation guarantee. Universal top6matching comes from the existing Hall certificate. CanonicalW8/A8 routes and conditional timing retain the earlier numerical/hardware limits.

Capture:16prompts×16steps×40layers; comparison:first3layers×256inputs×16profiles=12288events. All 0 failures are retained. Saved-trace independent replay covers52representatives, not allquerygraphs.

| Configuration | Mean µs | Event P99 µs | 3layer P99 µs | Fallbacks | FIFO128valid |
|---|---:|---:|---:|---:|---:|
| candidate.cr.g4triples.input1.GF | 280.526000 | 280.526000 | 841.578000 | 0 | 768/768 |
| candidate.cr.g2_fitted.input1.GF | 254.824310 | 254.712000 | 764.136000 | 1 | 768/768 |
| candidate.cr.joint8.input1.GF | 254.712000 | 254.712000 | 764.136000 | 0 | 768/768 |
| candidate.cr.g4triples.input4.GF | 285.438000 | 285.438000 | 856.314000 | 0 | 768/768 |
| candidate.cr.g2_fitted.input4.GF | 258.967914 | 258.856000 | 776.568000 | 1 | 768/768 |
| candidate.cr.joint8.input4.GF | 258.856000 | 258.856000 | 776.568000 | 0 | 768/768 |
| RR.cr.g4triples.input1.GF | 281.701990 | 301.006000 | 871.210000 | 0 | 768/768 |
| RR.cr.g2_fitted.input1.GF | 256.715367 | 275.908000 | 798.030000 | 1 | 768/768 |
| RR.cr.joint8.input1.GF | 256.604490 | 275.908000 | 795.922600 | 0 | 768/768 |
| RR.cr.g4triples.input4.GF | 286.613990 | 305.918000 | 885.946000 | 0 | 768/768 |
| RR.cr.g2_fitted.input4.GF | 261.625971 | 280.820000 | 812.766000 | 1 | 768/768 |
| RR.cr.joint8.input4.GF | 261.580490 | 280.820000 | 811.503400 | 0 | 768/768 |
| g8_fixed.cr.g4triples.input1.GF | 313.453135 | 355.966000 | 1008.358000 | 0 | 768/768 |
| g8_fixed.cr.joint8.input1.GF | 287.976729 | 329.880000 | 931.064000 | 0 | 768/768 |
| g8_fixed.ordinary.g4triples.input1.GF | 601.190089 | 720.666000 | 1992.898000 | 0 | 768/768 |
| g8_fixed.ordinary.joint8.input1.GF | 593.973682 | 712.840000 | 1970.384000 | 0 | 768/768 |

| New / reference | Mean reduction % | Event P99 reduction % | 3layer P99 reduction % | Slower/equal/faster | MiB new/reference |
|---|---:|---:|---:|---|---|
| candidate.cr.g2_fitted.input1.GF / candidate.cr.g4triples.input1.GF | 9.161964 | 9.201999 | 9.201999 | 1/0/767 | 289.5/244.5 |
| candidate.cr.joint8.input1.GF / candidate.cr.g4triples.input1.GF | 9.201999 | 9.201999 | 9.201999 | 0/0/768 | 303/244.5 |
| candidate.cr.joint8.input1.GF / candidate.cr.g2_fitted.input1.GF | 0.044073 | 0.000000 | 0.000000 | 0/767/1 | 303/289.5 |
| candidate.cr.g2_fitted.input4.GF / candidate.cr.g4triples.input4.GF | 9.273498 | 9.312705 | 9.312705 | 1/0/767 | 289.5/244.5 |
| candidate.cr.joint8.input4.GF / candidate.cr.g4triples.input4.GF | 9.312705 | 9.312705 | 9.312705 | 0/0/768 | 303/244.5 |
| candidate.cr.joint8.input4.GF / candidate.cr.g2_fitted.input4.GF | 0.043215 | 0.000000 | 0.000000 | 0/767/1 | 303/289.5 |
| RR.cr.g2_fitted.input1.GF / RR.cr.g4triples.input1.GF | 8.869878 | 8.338040 | 8.399812 | 1/0/767 | 289.5/244.5 |
| RR.cr.joint8.input1.GF / RR.cr.g4triples.input1.GF | 8.909238 | 8.338040 | 8.641705 | 0/0/768 | 303/244.5 |
| RR.cr.joint8.input1.GF / RR.cr.g2_fitted.input1.GF | 0.043191 | 0.000000 | 0.264075 | 32/735/1 | 303/289.5 |
| RR.cr.g2_fitted.input4.GF / RR.cr.g4triples.input4.GF | 8.718353 | 8.204159 | 8.260097 | 1/0/767 | 289.5/244.5 |
| RR.cr.joint8.input4.GF / RR.cr.g4triples.input4.GF | 8.734221 | 8.204159 | 8.402611 | 0/0/768 | 303/244.5 |
| RR.cr.joint8.input4.GF / RR.cr.g2_fitted.input4.GF | 0.017384 | 0.000000 | 0.155346 | 32/735/1 | 303/289.5 |
| g8_fixed.cr.joint8.input1.GF / g8_fixed.cr.g4triples.input1.GF | 8.127660 | 7.328228 | 7.665333 | 0/0/768 | 303/244.5 |
| g8_fixed.ordinary.joint8.input1.GF / g8_fixed.ordinary.g4triples.input1.GF | 1.200353 | 1.085940 | 1.129712 | 0/0/768 | 303/244.5 |

Sources are separate from all earlier sealed campaigns. comparison.protocol.frozen.json is the authoritative evaluation plan, superseding only the old evaluation paragraph inherited by the exact capture clone. The clone differs only in OUT/PROMPTS; its capture/extraction/run logic is identical.

Commands:

```sh
python scripts/research_20261102_cr_second_prospective_replay.py freeze
python scripts/research_20261102_cr_second_prospective_replay.py capture
python scripts/research_20261102_cr_second_prospective_replay.py run
python scripts/research_20261102_cr_second_prospective_audit.py
```

Final manifest hashes the two protocols, prompttexts, rawcapture reports/logs/routes/ledgers, frozen dependencies, all comparison arrays and savedtraces, and independent verification. No existing sealed source or campaign was changed.
