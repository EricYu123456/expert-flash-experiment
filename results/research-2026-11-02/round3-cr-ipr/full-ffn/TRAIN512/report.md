# Conditional full-FFN TRAIN service replay

First3MoE layers × 512decode inputs/layer. Fixed source/input hashes precede execution. These are explicit conditional service models, not measured hardware latency.

| Profile / Down / copies / priority / activation | Mean gain vs exact matching keep-hot | P99 gain | Slower queries | Joint valid FIFO64/128/256/2048 |
|---|---:|---:|---:|---|
| strict100.g16_equal.input1.GF.a0 | 2.8805% | 5.5696% | 0/1536 | 0/1536/1536/1536 |
| strict100.g16_equal.input1.DF.a0 | 2.8805% | 5.5696% | 0/1536 | 0/1536/1536/1536 |
| strict100.g16_equal.input4.GF.a0 | 2.8185% | 5.4530% | 0/1536 | 0/1536/1536/1536 |
| strict100.g16_equal.input4.DF.a0 | 2.8185% | 5.4530% | 0/1536 | 0/1536/1536/1536 |
| strict100.g12_cyclic.input1.GF.a0 | 2.8131% | 5.3324% | 0/1536 | 0/1536/1536/1536 |
| strict100.g12_cyclic.input1.DF.a0 | 2.8131% | 5.3324% | 0/1536 | 0/1536/1536/1536 |
| strict100.g12_cyclic.input4.GF.a0 | 2.7562% | 5.2242% | 0/1536 | 0/1536/1536/1536 |
| strict100.g12_cyclic.input4.DF.a0 | 2.7562% | 5.2242% | 0/1536 | 0/1536/1536/1536 |
| cold_reset500.g16_equal.input1.GF.a0 | -1.2440% | 1.3434% | 1259/1536 | 0/1536/1536/1536 |
| cold_reset500.g16_equal.input1.DF.a0 | -1.2440% | 1.3434% | 1259/1536 | 0/1536/1536/1536 |
| cold_reset500.g16_equal.input4.GF.a0 | -1.2174% | 1.3155% | 1259/1536 | 0/1536/1536/1536 |
| cold_reset500.g16_equal.input4.DF.a0 | -1.2174% | 1.3155% | 1259/1536 | 0/1536/1536/1536 |
| cold_reset500.g12_cyclic.input1.GF.a0 | -1.2152% | 1.2456% | 1259/1536 | 0/1536/1536/1536 |
| cold_reset500.g12_cyclic.input1.DF.a0 | -1.2152% | 1.2456% | 1259/1536 | 0/1536/1536/1536 |
| cold_reset500.g12_cyclic.input4.GF.a0 | -1.1909% | 1.2205% | 1259/1536 | 0/1536/1536/1536 |
| cold_reset500.g12_cyclic.input4.DF.a0 | -1.1909% | 1.2205% | 1259/1536 | 0/1536/1536/1536 |
| strict100.old_strong.input1.GF.a0 | 2.8277% | 5.2174% | 0/1536 | 0/1536/1536/1536 |
| strict100.g12_cyclic.input1.GF.a2000 | 2.8131% | 5.3324% | 0/1536 | 0/1536/1536/1536 |
| strict100.old_strong.input1.DF.a0 | 2.8277% | 5.2174% | 0/1536 | 0/1536/1536/1536 |
| strict100.g12_cyclic.input1.DF.a2000 | 2.8131% | 5.3324% | 0/1536 | 0/1536/1536/1536 |
| strict100.old_strong.input4.GF.a0 | 2.7723% | 5.1126% | 0/1536 | 0/1536/1536/1536 |
| strict100.g12_cyclic.input4.GF.a2000 | 2.7562% | 5.2242% | 0/1536 | 0/1536/1536/1536 |
| strict100.old_strong.input4.DF.a0 | 2.7723% | 5.1126% | 0/1536 | 0/1536/1536/1536 |
| strict100.g12_cyclic.input4.DF.a2000 | 2.7562% | 5.2242% | 0/1536 | 0/1536/1536/1536 |

The generic sparse compiler uses the exact candidate layout and graph:0%gain. Strict100ns copy is selected after prior pipeline sensitivity; it is a declared unmeasured conditional timing value. Cold-reset500ns explicitly pays28us cold reads. Strict500ns and fixed-source failures are preserved.

All private Down jobs wait for full expert GU at host, whole-vector activation/A8 and a full private input upload. Chip owner remains held until compute end and final ONFI output. FIFO validity is posthoc by64/128/256/2048B; costs with an insufficient FIFO are not feasible bounded-FIFO schedules.

Event counts and attributed work are in report.json; all paired arrays in service_arrays.npz. No new heldout data was opened.
