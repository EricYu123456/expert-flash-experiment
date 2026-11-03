# Independent output-fringe audit

**PASS**, with a conditional native-ABI interpretation. [audit-fringe.json](/home/ericyu/expert_flash_experiment/results/research-2026-11-02/round2/qwen38-placement/output-fringe/independent/audit-fringe.json) checks 240 unchanged frozen plans, 480 fringe records, 2,649,120 saved cost samples and all paired 48-layer aggregates. The separate set-based scheduler reproduces 799,200 saved nodes, checks first/last queries for each fringe plan, and checks original useful-mode first queries: 1,440 invocations (1,200 distinct plan/mode/query cases).

All values below are nominal FFN service milliseconds, including the always-active shared FFN.

| Frozen family | Useful rows: mean / P99 | 16KiB pages: mean / P99 | 64KiB waves: mean / P99 |
|---|---:|---:|---:|
| group_g16 | 89.009520 / 89.009520 | 89.031456 / 89.031456 | 89.073696 / 89.073696 |
| group_best | 91.225482 / 94.531732 | 91.244517 / 94.550728 | 91.281178 / 94.588109 |
| overlap_best | 84.645319 / 85.752648 | 84.656573 / 85.764050 | 84.656573 / 85.764050 |
| whole_row | 76.828409 / 77.407865 | 76.834621 / 77.414336 | 76.834621 / 77.414336 |
| rect_min_io4 | 74.624175 / 75.207012 | 74.624955 / 75.207848 | 74.624955 / 75.207848 |

With complete-wave outputs, whole_row improves mean/P99 over overlap_best by **9.239628% / 9.735681%**, and over group_g16 by **13.740392% / 13.089566%**. The conditional rectangle control adds **2.875873% / 2.850232%** over whole_row. All 41 request means retain these improvement signs. The train-selected group_best remains **2.478265% slower** in mean than group_g16; it is not the strongest external grouping baseline.

Whole-row geometry emits 14 extra GU rows and 92 extra down rows per expert in either fringe mode. Routed rectangles add zero fringe rows, but their frozen shared expert uses whole-row geometry, so rectangle-family cost is not exactly unchanged. Its full48 mean rises by 779.87ns, or 0.001045%.

In `inventory_examples`, `useful_output_rows_per_expert` means the pre-fringe sum of native job rows. For rectangles that includes distinct column-stripe partial outputs (GU3840/down5120), rather than the reconstructed model dimensions (GU1280/down2560).

There are 291 rectangle-family tokens whose summed greedy makespan falls slightly after outputs increase. [audit-negative-cases.json](/home/ericyu/expert_flash_experiment/results/research-2026-11-02/round2/qwen38-placement/output-fringe/independent/audit-negative-cases.json) independently replays all 48 layers for the three largest decreases (288 more scheduler invocations). Every node duration is nondecreasing; resource-order changes reproduce the decreases. This is a greedy-scheduling effect, not a reduction in work or a claim about optimal scheduling.

The [hardware evidence ledger](/home/ericyu/expert_flash_experiment/results/research-2026-11-02/round2/qwen38-placement/output-fringe/independent/hardware-evidence.json) distinguishes disclosure from assumption. AiF §4.4 specifies input dimension and page count plus FIFO/SREG result collection, but does not establish valid-R termination, output width/scales, incomplete-row flushing or unused-plane output behavior. [Primary paper](https://doi.org/10.1145/3695053.3731073). The two floor-based fringe cases are therefore **not proven hardware worst-case bounds**. This audit verifies their implementation and conditional cost implications.

The stream remains 5,519 decode tokens, 41 work requests in one external QA task/client session; warmup and prefill stay excluded. No fitting or additional data access occurred. P99 is per-token summed48FFN service, not full-model wall time or a request SLO. Native numerical/output ABI and actual NAND/PE timing remain unverified; nominal I8/A8/four-byte results do not reproduce the source NVFP4 checkpoint's quality.

