# Ordinary-read GU tile sharing: bounded DeepSeek screen

This branch stops without expansion. The 250 ns command profile leaves only
0.33–0.87% mean service-to-necessary-work headroom in the screened layouts.
At the more favorable 5,000 ns command sensitivity, complete native-DAG replay
improves mean by 1.02–1.36% versus the strongest native baseline, while P99
worsens by 0.50–0.76%. It does not meet the frozen 2% mean / 5% tail expansion
gate. Those thresholds allocate research effort, not establish a general
publication threshold or prove that all layouts are optimal.

## What differs from the older Qwen screen

The prior `gu-coalescing` study used C=2560 and a conservative 128-row/5-wave
joint boundary. DeepSeek GU uses C=2048: each 16 KiB page holds eight complete
rows, and each 64 KiB ordinary four-plane wave holds 32 complete rows. A
176-row expert band can therefore share its last half-wave with the next
expert. Tight stored intervals contain each coefficient once; **read views**
may overlap and reread coefficients. This is not a storage replica or CR-read.

Each native view starts/ends on a wordline, emits all 32 rows per wave and
splits at 192-WL/6,144-row block boundaries. Its full-C input is the same H
for all routed GU and the shared GU. Inactive/fringe/padding rows are genuinely
read, computed and drained, then dropped by a logical ownership map. Singleton
views can reread selected fringe rows, but only their assigned logical fragment
is scattered; every needed output row is accepted once. No earliest-arrival
reuse is credited. Down remains private-input, and waits for the complete
expert GU vector, including the unchanged full-vector A8 max-abs barrier.

For uniform176, a run of q experts beginning at half-wave parity p has
`ceil((11*q+p)/2)` waves. Thus only even-start pairs save a wave: an aligned
pair costs 11 rather than two singleton six-wave calls. The code does not
credit a wave for every adjacent pair. A standard exact maximum-weight perfect
matching places 32 routed pairs at wave-aligned slots. The first block holds
17 pairs/187 waves; the second holds 15 pairs plus the 11-wave shared band,
176 waves. No expert crosses a block in this control.

## Frozen work screen and controls

Only DeepSeek L0 actual-W8 TRAIN is used: 32 prompts × 16 decode positions,
512 ordered top6 selections, plus the always-active shared expert. No heldout
or new payload is opened, and no expert/chip labels are refitted. Sixteen
deduplicated full row maps cover v2 simple/free, the original v5/v6 250 ns and
5 µs disjoint/cyclic/whole-row winners, and uniform176 GU on each v2 free Down
and shared plan. Every map receives identical controls:

* identity, frequency and standard coactivation-path dense packing;
* maximum-weight matching dense packing;
* the same matching with whole, wave-aligned, non-crossing pair slots.

The coactivation/matching controls use TRAIN only, with no additional random
seeds or annealing. Every case includes tight singleton and exact interval-DP
coalesced views. The DP charges `waves*28,000 + commands*(cmd+C*copies/2) +
emitted_rows*2` ns per chip, and adds unchanged routed/shared Down work. This
is necessary native context work, not a complete dependency/resource schedule.
All programmed padding, page/wave reservations, GU-only blocks and safe Down
tail-fill block counts are recorded. Down never occupies a possible GU read
range. Useful coefficients remain single-copy; physical capacity is reported
directly rather than constrained to model-average per-chip payload.

The TRAIN matching has 658 co-selected pair occurrences: 1.28515625 pairs per
query, versus 0.238095 under uniform random top6 routing. Its uniform176 GU
work drops from 752 to 731.4375 aggregate waves/query (including shared).
Despite real affinity and saved sensing, it remains worse than the 88-wave
192/160 geometry. The strongest 88-wave geometry has 704 aggregate GU waves
per query, unchanged by fusion. Its benefit comes from fewer commands/input
reloads, not recovered half-wave tails.

| Input copies | Command (ns) | Strong native mean (ms) | Best DP joint-work bound (ms) | Possible gap |
|---:|---:|---:|---:|---:|
| 1 | 250 | 1.918061953 | 1.911744445 | 0.3294% |
| 4 | 250 | 1.959959660 | 1.942884750 | 0.8712% |
| 1 | 5000 | 2.077521125 | 1.967148742 | 5.3127% |
| 4 | 5000 | 2.116102992 | 1.998196273 | 5.5719% |

The 5 µs numbers above are deliberately not called speedups. About 96 µs of
their apparent gap already exists between the original native schedule and
its old node/work bound. In the same tight layout, singleton→DP chip-work
reductions are only 0.758% / 1.123%. Controller queues and dependency timing
therefore require complete DAG replay before assigning performance credit.

## Bounded complete-DAG sensitivity

The separately frozen 5 µs campaign keeps at most eight static map/order
candidates per input profile. It includes the v2 simple/free, original 5 µs
three-family DP-order winners, uniform176 matching control, and free identity
and matching-block controls. Every candidate receives original-plan,
tight-singleton and DP modes and both gate-first/down-first priorities.
TRAIN mean selects uniformly over the same scheduler choices for all families.

The new translation unit includes the sealed C++ `Graph` unchanged. Each fused
job gets the minimum selected-expert ordinal, and retains one chip context
through its complete channel drain. Its entire PCIe-output completion is a
dependency of **every** participating expert's GU-ready node. No expert is
released at an early row inside the stream. The common H PCIe upload occurs
once, and each native command pays its one/four full-H ONFI input copies.
Private Down, CPU activation/submission, full-GU scale barrier and combine
remain unchanged. New row demultiplexing CPU overhead is favorably zero.

| Input copies | Strong original mean (ms) | Best fused mean (ms) | Mean gain | Original P99 (ms) | Fused P99 (ms) | P99 gain |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.077521125 | 2.056312023 | 1.0209% | 2.110435780 | 2.120956020 | −0.4985% |
| 4 | 2.116102992 | 2.087236031 | 1.3642% | 2.149073720 | 2.165416340 | −0.7604% |

Both fused winners are the original 5 µs **88-wave** whole-row map with the
coactivation path and gate-first priority. Their matched same-layout singleton
means are 2.081081996 / 2.119528842 ms; fusion gains 1.1902% / 1.5236% mean
and 0.4978% / 0.2332% P99 against those controls. The strongest original/free
baseline remains faster in the tail. No uniform176 tail-sharing case wins.

Compared query-by-query with the strongest original, input1 improves 435 and
worsens 77 selections; input4 improves 450 and worsens 62. Against the matched
singleton, the corresponding counts are 475 improved/24 tied/13 worse and
485 improved/24 tied/3 worse. Thus additive-work savings do not imply a
pointwise complete-DAG improvement. `paired_deltas.npz` preserves all deltas.

The singleton controls emit all whole-wave fringe outputs and pay actual
cross-block extra calls. Their differences from original page-fringe jobs are
explicit in `audit.json`; there is no useful-row-only output shortcut.
All winners' safe tail-filled layouts use three LSB blocks per chip,
603,979,776 B total, while separate GU-only and Down zones can require an extra
block on some chips. Full per-chip ledgers and unique stored intervals are
saved in the work-layout artifacts.

## Validation, ABI limits and attribution

Independent code uses connected interval unions rather than the fitter's DP.
The result is exact here: a positive one-wave gap costs at least 28,064 ns,
larger than the maximum 9,096 ns command/input saving, while touching or
overlapping views always benefit from merging. It verifies all 80 stored
layouts, 655,360 query/chip cases, 1,310,720 work controls, unique coefficient
intervals, one-owner scatter coverage, block splits and safe Down-tail fills.
The uniform176 matching wave count exactly equals `752−16*658/512`.

For the complete DAG, all 16,384 original bridge query costs equal the sealed
native backend. A second Python builder uses independent connected unions and
the slow scheduler: 192 representative C++/Python traces and 211,652 nodes
match exactly, including starts/ends, owners, resources and dependency order.
The independent audit also recomputes means, P99 and paired differences.

This is a virtual-address and timing-model result, not hardware execution.
The architecture audit separately checks combinatorial row legality; native
scalar/output scaling, FIFO/backpressure, maximum R/pagecount encodings and
actual dispatch support remain unverified. Fullwave reads avoid a valid-R or
partial-row stop assumption, but do not establish those other ABI details.
Neither the conditional 5 µs cost nor zero host scatter overhead is presented
as a calibrated device measurement.

Same-input row concatenation is established in official TensorRT-LLM
[FuseGemms](https://github.com/NVIDIA/TensorRT-LLM/blob/c295dd9fca143a3fdd2769617699fdd4d4a93eb5/tensorrt_llm/_torch/auto_deploy/transform/library/fusion.py#L45).
[Neuralink §5.1](https://arxiv.org/html/2410.19274v3#S5.SS1) already combines
nearby reads across inactive gaps, and [SLIM §6.1](https://arxiv.org/html/2507.09201v1#S6.SS1)
packs fused vectors into aligned pages. The previous local primary-source
audit is retained in `gu-coalescing/prior_and_legality_audit.json`; those
retrieval/custom-datapath capabilities are not added to AiF. This branch
establishes a stricter ordinary-read control and a bounded negative result,
not a novel fusion/matching/interval-DP algorithm or an impossibility theorem.

Work protocol SHA256:
`edac9aa435e66be282d9a5eeee69cb6397bb62bb749ac43dff2b0e359a299a80`.
DAG protocol SHA256:
`8b2cd13c8f30e63380c1fb157af34353b607744ed0153d7100e144a3dd223e40`.
All source/input snapshots and individual costs remain in `work-screen-v1/`
and `native-dag-v1/`. No all-layer or heldout extension is performed.
