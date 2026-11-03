# Shared-aware counterbalanced whole-row screen: sealed v1 and stronger v2

This branch stops after the first three layers. Its strongest matched controls
leave 1.23–1.41% mean improvement for Granite4Tiny and 0.30–0.33% for DeepSeek,
below the predeclared expansion criteria (2% mean or 5% summed-three-layer P99).
These are TRAIN-only, synthetic ordinary-read native-graph timings. They are
neither heldout evidence nor measured hardware/full-model speedups.

## Inputs and legal construction

The screen uses the frozen actual-W8 TRAIN routes: 32 prompts × 16 decode
positions = 512 paired queries, six routed experts per query, layers 0–2.
It uses the frozen v5 input-copy=1 and v6 input-copy=4 native cost profiles,
250 ns per command, one active context per chip, ordinary NAND reads, and
page-fringe outputs. All routed and shared matrices are full-row, one-copy.
There is no new quantizer, partial-column computation, cache, extra context,
or heldout fit. Per-chip inventory need not equal the model average; exact
useful/page/wave/block inventory is in `winner_ledgers.json` and all static rows
and addresses remain in the referenced NPZ artifacts.

Granite GU has 9 two-wave chips and 7 one-wave chips; Down has 12 one-wave
chips and 4 omitted chips. Counterbalancing places the four omitted Down
chips within the GU high set, giving five joined-three-wave and eleven
joined-two-wave chips per expert. Deep GU has 8 six-wave and 8 five-wave
chips; Down has 13 three-wave and 3 two-wave chips. The analogous construction
gives five joined-nine-wave and eleven joined-eight-wave chips. This is a
standard coupled-coloring/search guide, not a new optimizer claim.

All families receive the same shared candidate pool: Granite's source shared
geometry; Deep's source 89-wave shared Down and a legal balanced 89-wave
alternative (9 six-wave chips + 7 five-wave chips). Each has the same two
fixed seed/shared-placement contexts. The same shared options are available
to every routed family; chosen shared geometry is recorded per layer.

## Attribution correction

V1 is immutable. Its equal-row cyclic/disjoint control omitted balanced/depth
phase templates; its generic independent control allowed only single-phase
moves, while the coupled guide allowed paired moves. Consequently its apparent
3.06/1.83% Granite and 4.28/4.14% Deep mean gains over `joint_cyclic` must not
be presented as gains over the strongest generic/simple-menu control.

V2 adds two stronger families:

* `joint_simple_menu` reads every saved phase template, including balanced-wave
  and depth profiles. It ranks all 16 relative Down rotations, keeps three
  distinct geometry pairs plus the actual frozen whole-row/simple incumbent,
  and allows single and paired phase-label/global-permutation moves. This is
  a finite four-pair heuristic pool, not an exhaustive cyclic/permutation optimum.
* `independent_paired` allows arbitrary per-expert row permutations with both
  single and paired phase/global swaps. Seed zero starts from the same
  route-blind coupled initialization; seed one starts from the retained v1
  generic/coupled winner. Every context retains its best eligible v1 incumbent.

Each family/shared geometry/seed receives 20,000 new stochastic proposals and
at most 4,096 bounded-greedy attempts. The four simple-menu pairs split that
total. Both initial and final states receive exact native replay over all 512
queries; the best initial/final/inherited native mean is retained. Earlier
v1 search effort inherited by v2 is **additional**, not charged to its new
budget. This is explicit baseline strengthening, not a fresh equal-total-CPU
contest. An unrestricted family retaining a constrained winner cannot be
claimed to lose to that winner on mean.

## Paired three-layer results

Layer winners are selected by TRAIN native mean, then their costs are summed
for the same 512 queries before computing mean and P99. Times below are ms.

| Model | Input copies | Simple-menu mean | Generic paired mean | Mean gain | Simple-menu P99 | Generic P99 | P99 gain |
|---|---:|---:|---:|---:|---:|---:|---:|
| Granite4Tiny | 1 | 1.714969305 | 1.690821957 | 1.4080% | 1.788076260 | 1.755738540 | 1.8085% |
| Granite4Tiny | 4 | 1.783104559 | 1.761181969 | 1.2295% | 1.856384240 | 1.811065540 | 2.4412% |
| DeepSeek | 1 | 5.771214891 | 5.752081775 | 0.3315% | 5.843619770 | 5.816397930 | 0.4658% |
| DeepSeek | 4 | 5.895378383 | 5.877861314 | 0.2971% | 5.967643120 | 5.937905140 | 0.4983% |

Relative to retaining v1's coupled winners, v2 generic search adds only
0.00149%, 0.00426%, 0.00101%, and 0.00110% mean improvement in those four
cases. Thus the guide does not demonstrate an additional performance gain
over an incumbent-retaining unrestricted search. The remaining comparison
with a finite simple menu concerns mask/search freedom and the whole-row
geometry family; it does not establish a new algorithm or a published-prior
novelty claim. Root's separately bounded exact-native refinement is outside
this sealed campaign.

## Independent checks and provenance

The independent auditor imports the frozen native reference scheduler but
does not import the fitter. It recomputes all 184,320 initial/final per-query
mandatory-work values, validates 432 unique initial/final/inherited artifacts,
checks 828,672 native jobs for row coverage, page/wave cost and address
non-overlap, and matches 72 selected query costs between Python and C++.
It verifies unchanged per-expert row-type multisets, cyclic-orbit membership
after global permutations, exact shared candidate equality, proposal limits,
incumbent retention and mean selection. See `summary.json` and
`paired_totals.v2.npz` for precise paired values and checks.

V2 initially failed **before fitting** in Granite: its Down balanced-wave
profile had already deduplicated to `cyclic_12`, so a literal `whole_rows`
family-name assertion was invalid. Original v2 source/protocol/logs and all
six successful Deep cases remain immutable. The explicit `v2-fix1` adapter
changes only that assertion to complete-template-count plus frozen-incumbent
orbit-membership checks, with identical menu, seeds, fit rules and budgets.
Only the six Granite cases use this separately frozen adapter.

V1 source SHA256: `b684522a894ba31acd6d076caff4454ee37bbd56ed7e3579f349b2bc7372a0fd`.
V1 completion SHA256: `4e997fe24f97ec0224894f88951281be5993fc8321e2fe1d904bec28179cb4a5`.
V2 source SHA256: `8eda7630bc3c1fea52fd10380035b0e93f71c1544f7fd2d68e6f47f0d48d7ebd`.
V2 protocol SHA256: `ed7b2faae2bd0045262632a2195f79348b1908c9470126a3bac323422311a375`.
Fix1 source SHA256: `457848676bfedc06c0306530fccbe80d3c97eeb004aab763ef449db773896f79`.
Fix1 protocol SHA256: `47a06cd9544b5b497126b89db02f445584c8e2b31f775311a3b4bafee3ee5925`.

No layer, model, heldout task, or dataset is expanded by this report.
