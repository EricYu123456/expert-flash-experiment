# Bounded native-DAG mixed whole-row geometry pilot

The pilot does not justify expansion. Against the strongest matched fixed-profile
native control, the best Granite4Tiny result improves TRAIN mean by 0.000127%,
and DeepSeek by 0.003182%. Both best states retain every expert's original
GU/Down row-count multiset: **zero new geometries survive incumbent retention**.
The tiny differences can reflect permutation/search-path changes and do not
demonstrate a heterogeneous-geometry benefit. This is a bounded negative search
result, not a proof that every heterogeneous geometry is unhelpful.

## Frozen experiment

Only Granite4Tiny L0/input-copy=1 and DeepSeek L0/input-copy=4 are used. Each
case uses 512 actual-W8 TRAIN selections, six routed experts plus the always
active shared expert. Shared GU/Down rows and placement are fixed to the v2
`independent_paired` winner throughout. There is no heldout read or fit, model
download, quantization change, new operator/context, cache or replica.

Each of seeds 20261105/20261106 receives 2,048 proposals, temperature 500 ns
decaying to 0.05 ns, 4% global moves and 50% paired-phase moves. Of non-global
proposals, 25% change geometry (about 24% overall); the rest use ordinary
single/paired chip swaps. A geometry proposal draws another saved row profile
and aligns its sorted row counts with the old stable chip-row ranks. A paired
proposal changes GU and Down together. This is standard elastic/fractional
sharding and finite local search, not a new algorithm claim; AnchorTP/HDA are
direct unequal-sharding priors.

The menu includes all corresponding saved phase templates and deduplicates
only by row-count multiset because arbitrary chip permutations are allowed.
It has GU/Down sizes 17/16 for Granite and 20/18 for DeepSeek, including equal
widths, balanced-wave and depth profiles. Every expert keeps its complete row
sum; every native shard fits one ordinary block. Different experts may select
different profiles, widths and active-chip counts. No fixed per-expert profile
histogram or model-average per-chip capacity constraint is imposed.

Every geometry change rebuilds ragged jobs/offsets. Only ordered unique routes
containing the changed expert are replayed; global moves replay all unique
routes. Full-cache checks occur every 128 attempts and for each first new
active-count transition. The finalizer restores best rows, lowers the full plan
again, resets its cache and replays all 512 original ordered queries. It does
not reuse the fixed-profile runner's terminal state or row-multiset assertion.

Protocol SHA256:
`ca5eeb62933b2eae8cc5478d678d5a0eddd64c4002707f7092715260d566dfae`.
Fitter source SHA256:
`32aa3fdda1fb9bab54ce2bd8fd24299aa8a3dd632ed300d477c81281ed6c5942`.
The protocol freezes 42 input/source hashes and source snapshots before formal
fit, including both v2 incumbents, original TRAIN sources, menus, backend and
the root's fixed-profile pilot outputs. The separate correctness smoke and
forced accepted-geometry → permutation → restore tests passed before the
formal run; those are not additional candidate-selection seeds.

## Exact TRAIN results

The fixed baseline is the minimum TRAIN mean among the root's two families
(`joint_simple_menu`, `independent_paired`) and two matched 2,048-attempt seeds.
It is also the best same-start `independent_paired` control in both cases.
The mixed result retains the same initial incumbent and selects the lower
TRAIN mean of its two seeds. P99 is descriptive, with no separate tail fitting.

| Case | Fixed mean (ns) | Mixed mean (ns) | Mean gain | Fixed P99 (ns) | Mixed P99 (ns) | P99 gain |
|---|---:|---:|---:|---:|---:|---:|
| G4 L0 input1 | 562670.5703125 | 562669.85546875 | 0.000127% | 604909.10 | 604909.10 | 0% |
| Deep L0 input4 | 1959950.03125 | 1959887.65625 | 0.003182% | 2000963.82 | 2000677.64 | 0.014302% |

All four mixed best states have zero changed GU/Down profiles. Geometry moves
were attempted 462/502 times for Granite and 498/511 times for DeepSeek, with
8/6/1/3 accepted transient moves respectively. They were discarded by
best-state retention. Per-expert initial/final profile indices, initial sorted
row multisets, complete rows and all original/final query costs are saved in
each fit's `arrays.npz`; complete move/checkpoint statistics are in `result.json`.

For the mean-selected mixed and fixed states, Granite improves 10 queries,
ties 487 and worsens 15; Deep improves 91, ties 316 and worsens 105. Gains are
paired on the same ordered queries, rather than computed from unrelated
quantiles. These are one-layer routed-plus-shared FFN synthetic service times,
not full-model or measured hardware speedups.

## Work, storage and compute accounting

Both final Granite seeds have the same aggregate inventory: useful weights
155,713,536 B, programmed pages/reserved waves 159,907,840 B, and 201,326,592 B
of allocated LSB block capacity (one block per chip). Both Deep seeds use
570,949,632 B useful, 575,209,472 B programmed/reserved and 603,979,776 B
allocated (three blocks per chip). Exact nonuniform per-chip values are in
each result's `per_chip_ledger` and the independent summary. Capacity is
reported directly; no average-weight cap is silently imposed.

Per query, Granite actually reads 1,176 pages/294 waves and issues 200 native
commands; Deep reads 4,252 pages/1,063 waves and issues 224 commands. Granite
uploads 225,280 input bytes and emits 76,616 output bytes; Deep uploads
409,600 input bytes and emits 148,428 output bytes. These ledgers include the
shared expert and every page-fringe output. The input-copy=4 contract charges
four channel copies of the listed logical input bytes. All dispatch, transfer,
host activation and combine dependencies are handled by the unchanged native
graph, rather than inferred from the work totals.

| Case/seed | Proposal native calls | Proposal queries | Validation calls | Validation queries | Fit CPU seconds |
|---|---:|---:|---:|---:|---:|
| G4 / 20261105 | 1547 | 104671 | 212 | 94829 | 46.4924 |
| G4 / 20261106 | 1520 | 98535 | 213 | 95276 | 45.3994 |
| Deep / 20261105 | 1391 | 93257 | 193 | 88640 | 45.0894 |
| Deep / 20261106 | 1375 | 89605 | 204 | 93689 | 44.8076 |

Additionally each fit makes one initial native call over 447 Granite or 459
Deep ordered unique routes. The three direct Python graph checks are separate.
Proposal count is matched, not CPU or valid-query count: the matching root
fixed controls report 92,239/94,991 Granite and 82,534/83,714 Deep total native
query evaluations and about 19–20 seconds wall time, without these extensive
mixed-cache validation calls. Earlier v2 search inherited by both is upstream
work, not charged as a new pilot budget. Thus the result is not an equal-CPU
claim; actual costs are exposed.

## Independent validation and stopping decision

The independent auditor does not import the mixed fitter. It verifies all
frozen hashes, reconstructs all final row/profile/offset/job/physical ledgers,
checks all 7,808 final jobs for complete one-copy rows, one-block legality and
non-overlapping physical addresses, freshly lowers/replays all 2,048 query
costs, and matches 12 selected Python graph costs. All pass. Full incremental
cache checks inside the formal fits and the separate forced geometry-to-swap
probe also pass.

Neither case approaches the frozen expansion gate of 2% mean or 5% P99 versus
the strongest exact-native fixed-profile control. No all-layer, uniform-menu,
phasewise-mixed or heldout extension is performed. The evidence rejects this
specific bounded candidate and budget; it does not certify global optimality
over arbitrary integer row geometries or all elastic-sharding optimizers.
