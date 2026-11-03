# Conditional GU FIFO / full-expert DMA sensitivity

This is **not a validated existing-hardware placement method**. It is an
optimistic controller/host ABI sensitivity, and the bounded pilot stops.
Even with zero added CQE/DMA setup/poll costs, the preferred full-expert DMA
contract improves mean only 0.157% / 0.193% beyond the retained strongest
legacy implementation. Its matched same-layout singleton gains are 1.235% /
1.461% mean and 0.461% / 0.535% P99. Neither input profile meets the frozen
2% mean / 5% P99 expansion gate. The nonzero-cost stage is not run; no layers,
heldout tasks, models or placement fits are added.

## Evidence and unresolved conditions

The supplied AiF text, lines 819–823, says output elements enter FIFO
sequentially and the controller retrieves them during computation as generated.
Lines 826–832 discuss beginning host attention work as independent heads become
available. This supports investigating output streaming; it does **not** prove
the complete MoE prefix-DMA/CID/error contract.

The sealed `streaming-capability-audit/conditional_contract.json` is an input
to this experiment. Its unresolved requirements remain mandatory:

* Prefix rows must have committed ECC/success status. A late, command-wide
  failure cannot invalidate a logical CID already reported successful. The
  published error-commit granularity is unknown.
* Controller software must group same-input logical requests, retain their
  original CIDs and buffer/input lifetimes, demultiplex known row ranges and
  issue exactly one final CQE after the entire expert GU is visible to host.
  This is not demonstrated existing AiF firmware.
* Finite chip FIFO and controller RAM must support the required occupancy and
  lossless ordered flow. Their capacities and polling/flow-control details
  are not established. No pause, producer backpressure or new context is added.
* The sensitivity assumes 32 four-byte results become available after each
  28 µs read wave, with no extra PE/poll delay. This is a declared optimistic
  timing model, not a measured PE rate or an inference from ECC throughput.

Without those conditions, retain whole-native-view logical completion. The
legacy fallback and known nonaligned whole-view timings remain strong controls.

## Bounded and uniformly applied experiment

Only DeepSeek L0 actual-W8 TRAIN512 is used, with 5 µs commands and one/four
ONFI full-input copies. The preceding frozen pool supplies five candidates
per input profile whose GU rows and physical starts/ends are all multiples
of 32 (160/192/352 rows). Nonaligned 176/187/188-row candidates receive no
invented partial-row streaming contract; their earlier known costs are retained.
No labels, ordering, geometry or sparsity/quantization are refitted.

Every candidate receives original-plan, tight-singleton and interval-DP
coalesced views, both gate-first/down-first priorities, and all four states:

| State | Transport | Logical expert ready |
|---:|---|---|
| 0 | Legacy whole-view output | After complete view PCIe drain |
| 1 | Per-wave FIFO/channel/host DMA | After every whole participating view drains |
| 2 | Per-wave FIFO/channel/host DMA | After all of that expert's GU rows drain |
| 3 | Per-wave FIFO/channel, controller collects complete expert | One full-vector host DMA, then one final CID CQE |

State 3 is the preferred, closer-to-current-API **conditional** contract.
State 2 is a more optimistic segmented-host-DMA submode. State 3 needs no
half-vector host readiness or multiple completions for one CID. The existing
full-expert SwiGLU/A8 global-scale barrier remains before private Down.

Every fused native job keeps its chip/input-latch owner until its final
channel output drains, including outputs belonging to experts already ready.
Next-wave sensing depends only on the previous sensing wave, never its output
drain; producer stalls/backpressure are not silently introduced. Commands use
the same minimum-selected-ordinal policy as all controls. The common H PCIe
upload occurs once, and each native job pays its one/four ONFI copies. All
emitted 128 B/wave channel traffic is charged. In this fully aligned pool every
read wave has exactly one selected logical owner; there are no uncharged
inactive/fringe outputs. Down remains unstreamed with its original input.

The original batched 2 µs GU submission cost is unchanged; an additional
per-CID submission fee is not added on top. Stage 0 sets **extra** CQE and DMA
setup costs to zero only as an optimistic sensitivity. A frozen optional
250 ns CQE / 50 ns DMA setup tier was explicitly uncalibrated, not a main
hardware setting, and is skipped because Stage 0 fails the gate. Byte-transfer
time, channels, PCIe, native commands, host tasks and context ownership are
still charged in Stage 0.

## Preferred full-expert DMA results

Times are synthetic one-layer routed-plus-shared FFN service, not full-model
latency or hardware measurements. All choices use TRAIN mean; tail numbers
are descriptive and are computed from the same paired queries.

| Input copies | Retained best legacy mean (ms) | Preferred mean (ms) | Mean gain | Legacy P99 (ms) | Preferred P99 (ms) | P99 gain |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.056312023 | 2.053079953 | 0.1572% | 2.120956020 | 2.118524020 | 0.1147% |
| 4 | 2.087236031 | 2.083197488 | 0.1935% | 2.165416340 | 2.135008680 | 1.4042% |

Relative to the strongest original non-fused plan with the **same** full-expert
DMA capability, mean gains are 1.069% / 1.442%; P99 changes are −0.493% /
+0.534%. Relative to the matched same-layout singleton with that capability,
mean gains are 1.235% / 1.461% and P99 gains 0.461% / 0.535%. These comparisons
also fail the gate without relying on the extra retained legacy control.

Input1's preferred winner retains the earlier source-5µs whole-row map;
input4's preferred winner is the v2 free map. Both use the coactivation path
and gate-first priority. Compared with retained legacy winners, input1's 512
queries all improve slightly; input4 improves 274 and worsens 238, with
individual changes from −61,140 to +73,630 ns. Paired differences are preserved
in `paired_deltas.npz` rather than inferred from separate P99 values.

Per-wave DMA and full-expert DMA give identical complete service costs for
all 15,360 gate-first plan/query combinations, despite very different DMA
graphs and buffering. Under down-first, 11,422 combinations differ, by both
signs. Thus the implementation is not ignoring the new DMA contract: the
gate-first command/context schedule hides its completion-time differences in
this workload. Streaming transport alone removes about 2.5 µs from the old
winner; early logical completion adds only a small remaining mean benefit.

## FIFO and controller-buffer accounting

All explicit-stream traces require a peak of 128 B per chip under this model's
instantaneous 128 B production bursts and completed-wave drains. This is a
**modeled capacity requirement**, not evidence that existing FIFO capacity is
128 B or sufficient. Polling gaps, real PE production timing and flow control
could change it. Drains are processed before arrivals at equal timestamps.

The preferred winners' peak resident GU payload in controller memory is
28,160 / 38,784 B. Across the entire preferred pool the peaks are 39,040 /
40,576 B. Those fields count bytes already arrived but not DMA-completed;
they do **not** prove that fixed contiguous buffers require only those sizes.
With six routed GU vectors of 11,264 B and one shared vector of 22,528 B,
preallocating every full vector needs 90,112 B = **88 KiB**. Sampled first-row
to-DMA-end allocation intervals peak at 33,792 B, but they do not bound all
queries or establish a fragmentation-free allocator. Static 88 KiB is the
explicit conservative reservation condition; controller RAM remains unknown.

The more optimistic per-wave host-DMA winners hold only 384 / 512 B of GU
payload pending DMA, but require segmented host DMA/notification semantics
that are not established. Neither buffering option is presented as free or
as verified device capacity. See `buffer_accounting` in the independent audit
for the frozen field-name clarification and sampled allocation intervals.

## Scheduler correction and verification

The old scheduler's summed chip-work bound assumes that read and output nodes
inside a native job serialize. It is invalid once the next read overlaps
earlier output drain. The new source retains exactly the same ready queues,
priorities and real job/context owner IDs, and independently uses the local
job dependency-chain length through final channel release as its legal
per-chip occupancy bound, alongside resource and global critical-path bounds.
It does not hide output nodes behind job-less tags or release contexts early.
Old sources are unchanged.

All 30,720 zero-extra-overhead legacy bridge query costs **and bounds** match
the prior sealed source. Independent Python scheduling matches 480 complete
C++ traces / 1,096,936 nodes. The additional auditor verifies context intervals,
contiguous unblocked read-wave progression, final-channel latch release,
one complete GU vector before DMA/CQE, FIFO/payload events, local-chain bounds,
all 122,880 stored query costs/means and paired statistics. All pass.

Protocol SHA256:
`3ec23e82f837d07c7d0eafc1cd8fbce0f8834735fc8d1f9fa871d01be342cdea`.
Python source SHA256:
`80e89472f89c82db98ccba0a07326b7c961c44e777b5f1baa012b2d8e8f1a368`.
Sources, explicit conditions, input snapshots and full cost/FIFO arrays are
retained. The outcome is a bounded conditional negative result; it does not
validate the missing firmware/ECC/DMA/FIFO contract or prove global placement
optimality. No heldout, all-layer experiment or nonzero-cost extension follows.
