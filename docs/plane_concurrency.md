# Plane concurrency and the scope of the placement result

The current model compares chip groups under an explicit rule: each chip
serializes independent expert/projection streams, and each stream is rounded
separately to a four-plane read round. Therefore a smaller chip group can win by
reducing plane rounding and amortizing stream cold starts. This does not
establish a universal optimum chip group or a fundamental MoE limit on AiF.

## What the paper establishes

The supplied AiF paper partitions each matrix into equally sized submatrices
across chips (§4.1, printed p. 533; §5.2, p. 537). It describes a page buffer per
plane, with two otherwise unused page latches available for input vectors
(§4.4, p. 536). Its on-chip GEMV request specifies one matrix block ID, an
offset, page count and input dimension; the controller supplies the input
vector. Figure 14 shows four PBs feeding an ECCLITE/PE path, but does not specify
independent per-plane GEMV contexts, accumulator ownership, or concurrent
requests for different expert matrices. Table 2 specifies four planes per chip,
16-KiB pages, 28-us ordinary LSB reads and 9.7-us cr-reads (p. 538).

Consequently, per-plane buffering makes more concurrency plausible, while
independent expert execution is an architectural extension that the paper does
not demonstrate. The block diagram alone is insufficient to prove either that
this concurrency exists or that it is impossible.

## Granite example and conditional alternatives

For Granite at INT8, one expert projection is 512 KiB. Splitting its output rows
over all 16 chips leaves 32 KiB, or two full pages, per chip. The baseline gives
each such stream its own four-plane round and ordinary first read. With eight
selected experts, three projections, and 24 layers, this yields
`8 × 3 × 24 × 28 us = 16.128 ms/token`. The 100% active-chip metric follows from
spreading every expert over all chips; 50% plane-slot fill means two page slots
per four-plane round are occupied under this schedule. Neither is measured
hardware multiplier utilization.

Two alternatives can improve the all-chip layout without shrinking the group:

* **Physically stack gate and up.** All selected experts' gate/up projections
  use the same layer input vector. For a given expert, storing gate and up as
  one stacked matrix gives four pages per chip and one cold start rather than
  two. Even keeping down serialized, the idealized weight-service time becomes
  `8 × 2 × 24 × 28 us = 10.752 ms/token`. The existing model's
  `fuse_gate_up` flag only puts separate streams in a common ready phase; it
  does not implement this physical stacking or remove their separate starts.
* **Pack independent expert streams into complementary planes.** If two
  selected two-page shards can occupy distinct plane pairs and start together,
  eight expert streams require four cold read rounds per projection rather
  than eight. The resulting conditional bound is
  `4 × 3 × 24 × 28 us = 8.064 ms/token`, before input, command, result, and
  computation overheads. This can beat the observed approximately 9.12-ms
  four-chip-group result under its different scheduling assumptions.

The second number assumes conflict-free plane assignments for the selected
experts, compatible NAND multi-plane addressing, concurrent starts, and enough
ECC/PE capacity. Actual routing can create fixed-placement plane conflicts.
Gate/up share an input vector, simplifying operand selection. Down uses a
different activated intermediate vector for each expert; supporting concurrent
down streams requires vector/context selection and separate output
accumulators. Available PB capacity alone does not implement these functions.
Switching blocks or losing sequential placement can also destroy the assumed
cr-read locality.

For reference, the weight-byte-only lower bound for this Granite workload is
`24 × 8 × 3 × 512 KiB / (16 × 6.4 GB/s) = 2.94912 ms/token`.
It ignores NAND first reads, transfers, placement conflicts and dependencies;
it is not an achievable measured FFN latency.

The defensible conclusion is that **placement and stream scheduling materially
affect flash utilization**, and that smaller groups improve this experiment's
chip-serialized schedule. A plane-aware scheduler and layout form an important
alternative experiment. The current result should not say that distributing a
small expert across all chips inherently wastes half the hardware PEs.

Paper references refer to the supplied local PDF,
*AiF: Accelerating On-Device LLM Inference Using In-Flash Processing*, ISCA 2025,
§4.1, §4.4/Figure 14, §5.2, and §6.1/Table 2. These are PDF pages 5, 8, 9 and 10,
respectively. The numerical alternatives above are analytical inferences under
the stated additional assumptions, not results reported by the paper.
