Independent audit PASS. A separately implemented geometry compiler and scheduler reproduced all 2,560 layer-query makespans exactly: 40 layers × 32 actual W8A8 decode routes × two frozen placements. It imports none of the replay, `Graph`, or `build_jobs` code. The independent scheduler uses per-resource sets and a linear event list, while preserving the declared gate-first `(phase, expert, node)` priority and resource-dispatch order.

| Whole-40-layer FFN service statistic | Frozen phase group | Frozen whole rows |
|---|---:|---:|
| Mean | 24.808711375 ms | 23.7423404375 ms |
| P99 over the 32 whole-model sums | 25.7279529 ms | 24.55891076 ms |
| Programming prompt mean | 24.985874875 ms | 23.93848375 ms |
| Summarization prompt mean | 24.631547875 ms | 23.546197125 ms |

Mean reduction is exactly 4.2983729440078555%. The audit recomputes every per-layer NPZ cost, native-chip lower bound, 40-layer per-position sum array, reported layer mean/P99, whole-model mean/P99 and per-prompt mean. P99 is computed after summing all 40 layers for each decode position, rather than by summing layer P99 values.

All 80 retained first-query traces match the independent schedule at every node's start and end timestamps. The trace checks cover 66,280 nodes and 12,760 native matrix jobs. They validate node duration and dependency metadata, resource exclusivity, input/compute/output ordering, and exclusive chip ownership from command start until output-channel drain. Whole-expert GU output must reach the host before activation, a down host submission and the down input upload. Host activation and final combine have the declared zero-duration assumptions in both placements; each expert's down submission costs 2 microseconds. Shared work has one corresponding seventh expert item per query, not a duplicated extra gate or up item.

The reconstructed native geometry uses C=1536 for routed and shared GU, C=512 for routed down, and C=1024 for shared down. R is the frozen chip row count. Every positive-R native job reloads its full C-byte input and returns 4R bytes, uses ceil(R*C/65536) ordinary 28-microsecond waves, and stays within the modeled 192-wave native-command limit. The 16 shared chips each receive 128 GU rows and 96 down rows. The reference's 120 shared cache entries confirm gate64 + up64 rows per chip and down96 rows per chip, with the expected columns. Shared gate/up rows use the same input vector; a whole-row concatenation and logical scatter represents the modeled 128-row GU job. This audit checks counts and native job shapes, not programmed flash weight addresses.

The route archive has shape [2,16,40,6]. Every stored expert ID matches the actual CPU W8A8 main-run decode trace, with six unique selected experts per layer. Both placed/unsharded source route files are byte-exact. The replay preserves the original frozen plan SHA256 `a55f17010b7b48c6c396dee92bb1d127a541e28a93b94aee2e42aac9f3dc2538`; `group_rows` and `rows` are read directly without fitting. Every protocol input/source hash is checked before and after the audit.

These results validate an uncalibrated software FFN service replay. They are not measured hardware latency, full-model wall time, task quality, heldout performance, a native AiF output ABI, or a new method claim. Nominal read timing and four-byte output costs remain model assumptions. The workload is the same two earlier training prompts and 32 decode positions, using actual W8A8 routes.

`audit.json` records the exact totals and checks; `layer_case_records.json` records all 80 layer/layout groups. The two `*.independently_rescheduled.npy` files preserve the independently reproduced [40,32] cost arrays. No pre-existing source or replay artifact was edited.
