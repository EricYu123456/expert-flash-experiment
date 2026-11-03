Chip-count StageA/B finite controls (anchor addendum incorporated)

| Model | Chips | Input copies | Best finite native mean / P99 (us) | Necessary LB (us) |
|---|---:|---:|---:|---:|
| granite4tiny | 8 | 1 | 1052.050711 / 1112.280000 | 1010.000 |
| granite4tiny | 8 | 4 | 1074.354039 / 1134.546000 | 1010.000 |
| granite4tiny | 16 | 1 | 561.099020 / 580.695140 | 506.000 |
| granite4tiny | 16 | 4 | 584.388680 / 605.445940 | 506.000 |
| granite4tiny | 32 | 1 | 326.997506 / 379.589140 | 254.000 |
| granite4tiny | 32 | 4 | 347.699428 / 399.771700 | 254.000 |
| granite4tiny | 64 | 1 | 208.815906 / 245.602000 | 142.000 |
| granite4tiny | 64 | 4 | 241.821725 / 318.101640 | 142.000 |
| deepseek | 8 | 1 | not run | 3698.000 |
| deepseek | 8 | 4 | not run | 3698.000 |
| deepseek | 16 | 1 | 1918.057055 / 1958.913000 | 1850.000 |
| deepseek | 16 | 4 | 1959.950031 / 2000.963820 | 1850.000 |
| deepseek | 32 | 1 | 1034.693246 / 1114.131000 | 926.000 |
| deepseek | 32 | 4 | 1080.061359 / 1152.997360 | 926.000 |
| deepseek | 64 | 1 | 565.910246 / 623.927660 | 478.000 |
| deepseek | 64 | 4 | 624.824760 / 687.814320 | 478.000 |

94 minimum-work/depth frontier calls, independently checked together with996 menu profiles. Native total29696 query-costs, including2048 strong-anchor fanout mirrors. GF/DF choices are offered equally to each map.

The maps are finite deterministic controls. Their gap to a necessary bound is not an algorithm benefit or proof that the bound is attainable. Subsequent learned CP controls are separate artifacts and should replace these controls when assessing novelty.

Hardware count is held fixed inside each comparison;32/64PE AiF is a scale sensitivity, not an implemented hardware claim. FullGU/A8, privateDown, singlecontext, complete inputs, commands and page-fringe outputs are charged.
