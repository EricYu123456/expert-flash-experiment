# Concrete virtual placement and capacity proof

`scripts/materialize_plan.py` turns every held-out fold's trained modulo/learned
configuration in `learned_overhead/plans.json` into a virtual INT8 NAND address
plan. Six plans per model are stored on NFS in:

- [Granite physical plans](/home/nfs_client/ericyu/expert_flash_experiment/runs/granite-aif/learned_overhead/physical_plans).
- [OLMoE physical plans](/home/nfs_client/ericyu/expert_flash_experiment/runs/olmoe-aif/learned_overhead/physical_plans).

These are address metadata, not copied weights. No model storage is claimed in
the 2 GB metadata-replay NVMe namespace. Each Granite plan is about 456 KB; each
OLMoE plan is about 1.25 MB. Files are named `{heldout_prompt}_{policy}.json`, with
an aggregate `summary.json` in each directory.

The allocator uses 16 chips, four planes/chip, 16 KiB LSB pages, and the inferred
192 LSB wordlines/block. A block stripe consists of one block in each plane of a
chip. A read wave reserves one wordline position in all four planes; a partially
filled last wave leaves some plane slots unused. Each chip has an independent,
monotonically increasing wave address, which ensures no overlap.

A short stream whose rounded page count fits in the remaining wordlines is
packed inside the current block stripe, even at a nonzero wordline offset. If
it would cross the boundary, the unused suffix is reserved as padding and the
stream starts in the next block. A stream longer than a block starts at a block
boundary and is split into full block runs and a final partial run. Thus a short
stream always has one cold run, and a long stream has exactly
`ceil(read_rounds / 192)` cold runs, matching `Demand.add_stream` without dedicating
a whole block to each shard.

For example, Granite's selected stacked g=4 layout puts a 256 KiB gate/up shard
in four read waves and its 128 KiB down shard in two waves. One chip can place
them at WL offsets 0 and 4, then put the next expert's shards at offsets 6 and 10.
The six-wave pattern fits exactly into a 192-WL block. The initial precharge and
final discharge occur separately for every requested stream; sharing a physical
block does not assume that arbitrary expert requests retain cr-read state.

Each stream record identifies the logical layer/expert, projection, placement
slot, chip, whole output-row interval, input dimension, useful bytes, physical
pages, read rounds, and cold-run count. Its segments give per-plane block ID,
WL offset, page offset within the stream, page count, and counts for each plane.
Column names are included in the JSON to keep the raw plans compact.

Whole output rows are divided once across the expert's assigned chip group.
Within one chip, pages are distributed over planes in round-robin order. For a
stacked matrix, rows `[0,I)` are gate weights and `[I,2I)` are up weights, so the
returned vector is split in that order before activation and multiplication.
Only then is the dependent down projection eligible. Some stacked shards may
contain gate rows only or up rows only; all rows remain uniquely owned.

For all six selected plans of each model, the capacity and page geometry are
identical between modulo and learned placement:

| Routed FFN allocation | Granite, stacked g=4 | OLMoE, stacked g=8 |
| --- | ---: | ---: |
| Useful INT8 weights and reserved LSB capacity | 1.125 GiB | 6 GiB |
| Streams/chip shards in a complete stored model | 6,144 | 16,384 |
| Four-plane block stripes, summed across chips | 96 | 512 |
| Streams starting at nonzero WL offsets | 6,048 | 15,872 |
| Boundary, final-block, page, or idle-plane padding | 0 | 0 |
| Raw TLC cell capacity corresponding to the LSB allocation | 3.375 GiB | 18 GiB |
| Raw TLC fraction of decimal 1 TB | 0.3624% | 1.9327% |

The three-times TLC figure accounts for storing one bit/cell as LSB weights when
the cells have three-bit capacity. The paper permits unrelated data on CSB/MSB
pages, so this figure is a conservative reservation of the cells' raw TLC data
capacity. ECC parity, spare blocks, and non-FFN model weights are excluded; this
is not a complete SSD capacity or reliability implementation.

Every allocated stream is checked against `Demand.add_stream` for useful bytes,
page count, read rounds, and cold runs. The serialized plans were independently
reparsed to verify nonoverlapping per-chip wave intervals and to compare all
layer/projection counters with `aif_model.phase_demand`. Whole-row coverage,
fixed mapping capacity, and total useful-byte conservation also passed. Built-in
allocator checks cover sharing a partly used block, skipping a non-fitting short
stream, and splitting a long stream across blocks.

This proves that the current timing model's single cold run for a short shard is
compatible with a concrete **block-contained**, space-efficient static layout.
The short shard need not start at WL zero. It does not prove physical AiF hardware
or implement `aif_post`, weight programming, FTL relocation, or garbage collection.
Long-stream block splits would also need their separate GEMV command overhead
in an implementation; the currently selected model plans contain only short,
single-block streams.
