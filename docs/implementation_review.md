# Independent implementation review

Reviewed `scripts/aif_model.py`, `tests/test_aif_model.py`, `kernel/aif.c`,
`kernel/aif.h`, `scripts/aif_replay.c`, and the current NVMeVirt integration
on 2026-10-19. This review concerns the explicit metadata-only timing model;
it does not validate an AiF hardware datapath or independently reproduce the
authors' unpublished simulator.

## Valid within the stated scope

- The binary descriptor is 400 bytes: a 16-byte header plus 16 records of
  24 bytes. Python encodes little endian; the kernel decodes the corresponding
  fields. The C replay forwards these bytes unchanged. The C static size assertion
  passes on the experiment's x86-64 platform.
- The replay allocates a 4096-byte aligned buffer, places the descriptor in its
  first 400 bytes, and submits opcode 0xc1 synchronously. The kernel's single-page
  PRP1 access matches that client contract. The custom I/O handling bypasses
  normal weight reads; these commands are explicitly metadata-only.
- The bandwidth arithmetic is correct: at 6.4 decimal GB/s, a byte takes 5/32 ns.
  Thus `ceil(pages * 16384 * 5 / 32)` is the correct physical-page transfer bound.
  Kernel bounds prevent overflow of these calculations for accepted descriptors.
- Whole output-row chip shards conserve useful matrix bytes, including a row
  remainder. Per-chip sequential pages occupy four planes; rounding pages before
  computing read rounds distinguishes physical page fill from idle plane slots.
- For the supported homogeneous expert dimensions, every stream on a given chip
  in a phase has the same shape and timing. Therefore aggregating NAND times and
  physical bytes before taking their maximum preserves the serialized-stream
  duration. Gate and up are verified to have identical dimensions.
- The integration initializes `nvmev_result.status` to success and its target
  time to the request start before calling `aif_process`; accepted commands then
  set target time to start plus modeled duration. No uninitialized status issue
  exists on this call path.
- Single-threaded synchronous replay prevents multiple command descriptors from
  simultaneously reserving the same chip. The present kernel does not need a
  cross-request chip queue for this client contract.

## Interpretation and extension constraints

1. The physically motivated default is one ordinary-read-equivalent term per
   block run: `28 + (P - 1) * 9.7` microseconds. The initial precharge and final
   discharge are both covered by that one 28-microsecond term. Two ordinary
   reads per run is a separate conservative sensitivity assumption, not the
   paper's phase equation. The model default was corrected to one during review.

2. `fuse_gate_up=True` currently batches two independent, identical-shaped
   streams in one descriptor. It retains both page rounding and both cold starts.
   Consequently it changes command counts but cannot change modeled NAND service
   time in this homogeneous setup. This is command batching, not packed-matrix
   layout or an implemented computation fusion. Actual gate/up packing would
   concatenate the per-chip rows before page/run rounding and model whether the
   flash command can continue between the two projection streams.

3. Aggregate `max(sum(NAND), sum(bandwidth))` must not be generalized to mixed
   stream shapes without retaining stream durations or a proper pipeline model.
   For example, serializing a 768-page stream and a one-page stream gives
   1,994,080 ns, while aggregating them first gives 1,968,640 ns. The difference
   occurs because one stream is bandwidth-limited and the other startup-limited.
   The current equal-shaped per-phase streams avoid this case.

4. Each stream's block-boundary accounting begins at offset zero. This models
   ideal contiguous placement without tracking actual physical starting offsets.
   If many matrix shards share blocks, some cross additional boundaries; if each
   shard is instead allocated a separate block, capacity waste must be assessed.
   Neither capacity allocation nor real FTL block placement is established by the
   aggregate descriptor.

5. The ready policy permits experts in disjoint chip groups to overlap inside a
   projection phase and keeps barriers between gate/up/down. It is a valid explicit
   policy, not a general optimal DAG scheduler. Equal stream shapes and fixed
   disjoint groups make those barriers benign for the current workload's service
   bound; heterogeneous experts would need a stronger scheduler.

6. Kernel counters and Python expected totals share the same timing equation.
   Agreement validates descriptor encoding, dispatch, delay scheduling, and
   accounting. It is not independent evidence that physical AiF latency or MAC
   utilization matches the equation. Report replay wall time separately from
   modeled service time, since QEMU/NVMe/software overhead is excluded by the model.

7. The kernel presently assumes the supported synchronous client. Future
   concurrent submission requires chip availability reservations across commands;
   otherwise two commands can overlap on the same chips with no contention.

## Checks run

All seven current Python unit tests passed, and `scripts/aif_replay.c` passed
`cc -std=c11 -Wall -Wextra -Wpedantic -Werror -fsyntax-only`.

An additional direct calculation using gate/up dimensions 512 x 1024, down
dimensions 1024 x 512, and selected IDs `[0,1,2,7,8,12,19,30]` confirmed identical
modeled service time with and without gate/up batching for g=1/2/4/8/16, while
the command count changed from three to two. The mixed-stream example above was
also reproduced directly with the current `Demand` implementation.

The updated tests include the default three-read run (47,400 ns). Additional useful
regression coverage would check one additional block cold term at 193 read rounds
with a 192-page block, ABI field decoding, and the no-service-time-benefit invariant
for current gate/up batching.
