# AiF evidence and MoE experiment model

This document separates facts from the supplied AiF paper, assumptions needed
for an executable timing model, and quantities that the proposed experiment can
actually establish. Local references use the PDF's printed page number and the
line numbers of `paper/aif.txt`; printed page 533 is PDF page 5.

## What the supplied paper establishes

| Item | Evidence | Consequence for this experiment |
| --- | --- | --- |
| Storage geometry | Table 2, p. 538; text lines 915–928: 8 channels, 2 chips/channel, 4 planes/chip, 16 KiB/page. | A 1 TB system has 16 independent AiFChips and 64 planes. Channels are shared by two chips. |
| Weight representation | Section 6.1, p. 539; lines 1032–1035: evaluated models are INT8. Section 4.4, p. 536; lines 775–779: PEs contain INT8 multipliers and an adder tree. | A paper-compatible weight model uses one byte/weight. A Q4 GGUF does not automatically implement the paper's INT8 datapath. |
| Maximum internal bandwidth | Section 6.1, p. 538; lines 899–914: 6.4 GB/s/chip, 102.4 GB/s/SSD for AiF; 1.6 GB/s/chip, 25.6 GB/s/SSD for AiF--. | Use the per-chip cap when many flash pages can be streamed; do not charge an extra full bandwidth-transfer time after charging the same read service. |
| Read latency | Table 2, p. 538; lines 921–928: conventional TLC LSB/CSB/MSB = 37/46/37 microseconds; be-enc = 28/46/46 microseconds; cr-read = 9.7 microseconds. | The first read of a new consecutive LSB run has a different cost from subsequent cr-reads. |
| Weight layout | Section 4.1/Figure 6, p. 533; lines 442–455: equal submatrices, sequentially stored across chips, all chips operate concurrently. Figure 6 depicts whole output-row bands spanning the input dimension. | Whole-expert placement on one chip is a useful alternate baseline, but it is not the paper's advertised matrix layout. |
| Locality requirement | Section 4.2, p. 534; lines 517–545: cr-read requires consecutive accesses to the same block without an interval. | Random expert selection does not inherently prevent cr-read *inside* each selected expert's contiguous matrix. Small matrix shards shorten that run. |
| Block crossing | Section 5.2, footnote 9, p. 538; lines 925–928: a submatrix crossing two IFP blocks is split into separate GEMV operations. | Account for new-run latency and command overhead at block boundaries; do not assume every page uses cr-read. |
| Block geometry disclosure | Section 4.3, footnote 6, p. 535; lines 664–667: 160 measured chips, 120 blocks/chip, 3,686,400 wordlines total. | This implies **192 wordlines/block for the characterized NAND**. Table 2 does not state the simulator's wordlines/block. A 192-LSB-page/block model is an evidence-based inference, not a disclosed simulator constant. |
| PE hardware details | Section 4.4/Table 1, pp. 536–537; lines 705–716 and 823–833: 200 MHz, 0.026 mm² PE area, 3.98 mW PE power per chip. | The paper does **not disclose the multiplier/MAC lane count** or per-cycle lane activity. Active-chip fractions cannot establish actual MAC lane utilization. |
| ECC | Section 4.4, pp. 536–537; lines 778–783 and 810–817: BCH ECCLITE corrects up to 10 bits/1 KiB, at 6.4 GB/s. | ECC and compute can be modeled as streaming stages; do not silently assume unlimited ECC throughput. |
| Command operands | Section 4.4, p. 536; lines 746–751: vector dimension, block ID, block offset, number of pages. Section 5.2, pp. 537–538; lines 847–877: aif_post and aif_gemv include matrix dimensions/ranges and host vector addresses. | Requests must refer to the selected expert's actual projection shape and shard pages, not merely a layer-wide byte count. |
| Activation storage | Section 4.4, p. 536; lines 724–742: typical input vector 8–32 KiB; two page-buffer latches can hold input vectors. | Input/output transfer precision and supported accumulation precision are not explicitly fixed by this text. Byte widths must be exposed as modeling assumptions. |
| Channel and PCIe limits | Table 2, p. 538; lines 926–928: 2.0 GB/s/channel and 8.0 GB/s PCIe. | Matrix weights stay within chips, but input replication and result transfers still use these interfaces. |
| Host scheduling | Section 5.1/Figure 15, p. 537; lines 830–838: head-level overlap and host/SSD FFN tensor partitioning. | Pure SSD FFN results isolate chip use; full reproduction of the paper's tokens/s additionally needs the host split and overlap policy. |
| Evaluation implementation | Section 6.1, p. 538; lines 871–882 and 910–920: llama.cpp plus modified NVMeVirt/NVMe driver; timing is simulated and dummy GEMV vectors are returned. | Upstream NVMeVirt alone is not the AiF accelerator, and this is not a physical AiF hardware measurement. |

The supplied paper **does include MoE**. Table 3 and Section 6.1, p. 539
(lines 1004 and 1038–1050), evaluate Mixtral-8x7B with 46.2 GiB total weights and
14.7 GiB weights accessed/token in INT8. It does not publish a fine-grained-expert
sweep, per-chip balance traces, or MAC lane utilization. Its positive MoE result
therefore does not answer the small-expert placement question by itself.

## Workload geometry

For an expert with hidden dimension H and intermediate dimension I, a gated FFN
has three projection matrices: gate I x H, up I x H, down H x I. At one INT8
byte/weight, one projection has H*I bytes and an expert has 3*H*I bytes. These
figures exclude scale metadata, routing, biases, shared experts, and attention.

The model configurations below were checked against their publishers' own
configuration files on 2026-10-19. Sizes are derived, not measured GGUF sizes.

| Model | H | I per routed expert | Routed E / selected K | Layers | INT8 bytes/projection | INT8 bytes/expert |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [IBM Granite 3.1 1B A400M](https://huggingface.co/ibm-granite/granite-3.1-1b-a400m-base/raw/main/config.json) | 1024 | 512 | 32 / 8 | 24 | 0.5 MiB | 1.5 MiB |
| [Qwen3 30B A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B/raw/main/config.json) | 2048 | 768 | 128 / 8 | 48 | 1.5 MiB | 4.5 MiB |
| [OLMoE 1B 7B 0924](https://huggingface.co/allenai/OLMoE-1B-7B-0924/raw/main/config.json) | 2048 | 1024 | 64 / 8 | 16 | 2 MiB | 6 MiB |
| [DeepSeek V2 Lite](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/raw/main/config.json) | 2048 | 1408 | 64 / 6 | 27 | 2.75 MiB | 8.25 MiB |
| [Qwen1.5 MoE A2.7B](https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B/raw/main/config.json) | 2048 | 1408 | 60 / 4 | 24 | 2.75 MiB | 8.25 MiB |

DeepSeek V2 Lite has an initial dense layer and two always-active shared experts;
Qwen1.5 MoE has an always-active shared FFN of intermediate size 5632. Account
for these separately instead of treating them as routed experts. Small fine-grained
experts in this table are much smaller than Mixtral's H=4096, I=14336 projections
(56 MiB/projection at INT8). The model dimensions, rather than a model's total
parameter count, drive short-run and page-tail effects.

## A conservative flash-service model

For P consecutive LSB pages within one plane/block, the two disclosed latencies
give the following run model in microseconds:

```
T_run(P) = 0                         if P = 0
T_run(P) = 28 + (P - 1) * 9.7        if P > 0
```

This is consistent with the paper's phase decomposition: one initial precharge,
P evaluations, P-1 recycling transitions, one final discharge. The 28-microsecond
term includes the first evaluation and the final discharge; do not add another
28-microsecond last-page charge. Starting a different block creates a new run.
Starting a new GEMV is conservatively a new run unless continuation across commands
is explicitly implemented. Cr-read cannot be assumed after arbitrary idle gaps.

For a chip with four planes, parallel-plane read service is bounded by the
slowest plane. A simple pipeline bound is:

```
T_chip >= max(max_plane(sum_block_runs(T_run(P))),
             bytes_read_on_chip / 6.4e9)
```

The plane/read calculation and bandwidth cap represent overlapping pipeline
constraints, not sequential transfers of the same bytes. Four 16 KiB pages every
9.7 microseconds would nominally provide 6.756 GB/s, so the disclosed 6.4 GB/s
cap matters. For a stronger simulator, retain a per-plane availability timeline,
per-chip ECC/compute timeline, per-channel vector-transfer timeline, and command
release/completion dependencies.

Only chip partitioning needs whole-output-row ownership. Within each chip,
sequential page packing can distribute pages round-robin over planes. For example,
96 KiB/chip is six pages and may occupy planes with page counts 2,2,1,1. Rounding
four independent 24 KiB plane allocations instead would read eight pages. Both
take two read waves, but the latter has larger read amplification. The paper does
not specify this fine plane layout, so document the choice and distinguish idle
planes in a final wave from bytes padded into pages.

Ignoring channel/command/host costs, full 16-chip row sharding yields useful
bandwidth efficiencies of approximately 18.3% for a 0.5 MiB projection (one
28-microsecond read wave), 40.7% for a 1.5 MiB projection (37.7 microseconds), and
54.3% for a 2 MiB projection (37.7 microseconds), relative to 102.4 GB/s. These are
**illustrative model calculations**, not simulation or hardware results. All 16
chips can participate in each case. Full participation does not imply efficient
useful work.

## Layouts and schedules that test the user's concern

1. **Whole expert on one fixed chip**: chip = expert_id mod C. This measures sparse
   active expert count, routing collisions, and chip imbalance. It is an alternate
   placement baseline rather than AiF's equal-submatrix layout.
2. **Paper-style full row shard**: split every expert projection's whole output
   rows across all C chips. This tests page tails, read-run startup, input
   replication, and NVMe command cost even when chip participation is 100%.
3. **Grouped row shard**: use g=1,2,4,8,16 chips/expert, with fixed disjoint chip
   groups and group = expert_id mod (C/g). All three projection matrices of an
   expert use its assigned group. For g=1 this reduces to whole-expert placement.
   Routing can select multiple experts in the same group; preserve these collisions.
4. **Optional ideal placement bound**: assign ready experts to groups dynamically
   without collisions. Label it as an ideal upper bound requiring replication or
   remapping; it is not achievable with a single fixed weight copy in general.

Run two explicit command schedules if practical: the observed llama.cpp issue
order, and a ready-work schedule that permits independent selected experts to
overlap. If each expert GEMV is forced to finish before the next expert begins,
g=16 is favored by construction and a grouped layout's chief benefit is excluded.
There must still be per-chip and per-channel contention when independent requests
overlap.

The FFN dependency graph is:

```
router -> selected expert IDs
       -> gate_e(x) and up_e(x)       [independent for every selected e]
       -> activation(gate_e) * up_e   [requires that expert's gate and up]
       -> down_e                     [requires that expert's activation]
       -> weighted combine           [requires all selected expert outputs]
```

Do not globally count every expert matrix as simultaneously runnable. Do not
overlap dependent layers or successive autoregressive tokens in a batch-one decode
test. Gate/up fusion is a separate implementation/scheduling choice and must be
held consistent when comparing layouts. Report shared-expert work and host
activation/router/combine costs separately.

Use actual decoded-token routing IDs when available. Dummy GEMV outputs can alter
the next layer's router choices, so the paper's statement that dummy outputs do not
affect inference performance is insufficient for a placement-sensitive MoE replay.
Capture valid llama.cpp inference once, then replay the same routed work under all
layouts. Clearly label synthetic uniform, concentrated, and adversarial traces;
they are useful controlled experiments but do not establish application routing.

## Metrics and checks

| Metric | Definition / interpretation |
| --- | --- |
| FFN latency | Per-layer and per-token critical-path time, with p50/p95 over decoded tokens. |
| Useful bandwidth efficiency | `useful_selected_weight_bytes / (C * 6.4e9 * FFN_service_seconds)`; includes startup, tails, imbalance, and waiting. Name the time interval and whether host/command cost is included. |
| Chip participation | Fraction of chips that execute any selected-expert work. A spatial metric only. |
| Chip service occupancy | Sum of chip service-busy intervals divided by `C * FFN_service_time`. Busy is flash/ECC service, not measured MAC lane activity. |
| Balance efficiency | `sum(per_chip_work) / (C * max(per_chip_work))`, for both useful bytes and service time. |
| Page efficiency | Useful selected weight bytes divided by physical page bytes read. Distinct from plane tail-wave occupancy. |
| Run/startup distribution | Pages/run, new block runs, one-page runs, and startup time share; isolates loss of cr-read amortization. |
| Command/communication costs | Commands/layer, input broadcasts, returned vector bytes, channel busy time, command/interrupt delay. |
| Scaling | Sweep chip counts 8/16/32/64, g, expert I, selected K, and block geometry. State what additional capacity/bandwidth each configuration assumes. |
| Actual PE lane use | Requires lane count and lane-cycle activity from a concrete hardware/RTL model. Not inferable from byte striping, active chips, or NVMeVirt alone. |

For conservation, every selected expert must contribute exactly H*I useful bytes
per gate/up/down projection at INT8; whole-row shards must cover all rows exactly
once. No unselected expert's useful MACs count toward utilization. Page bytes may
exceed useful bytes, and metadata/quantization bytes must be recorded separately.
Changing the layout must preserve the routed workload and mathematical FFN.
Tests should cover expert IDs colliding on a group, empty shards when rows<chips,
one-page and multi-block runs, channel contention, and down-after-gate/up ordering.

The strongest conclusion available from llama.cpp routing plus an AiF-style
NVMeVirt timing extension is whether a stated layout and schedule efficiently
use **modeled chip/flash service capacity** for this workload. A claim that actual
physical PE lanes are saturated would require additional hardware detail and
instrumentation beyond what this paper discloses.
