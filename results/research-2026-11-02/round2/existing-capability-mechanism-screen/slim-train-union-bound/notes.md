# SLIM native batching: TRAIN union and minimum-work screen

**No placement-only opportunity is established by the weight-page or MAC work counts on 16 dies.** A fixed equal g16 one-copy full-row map attains both lower bounds for every inspected cohort. Batched weight reuse itself can save substantial reads, but every same-batching control receives it. No native, quality, latency, arrival or SLO experiment was run.

The primary capability is [SLIM section 5.4](https://arxiv.org/html/2507.09201v1): its PE input SRAM caches a batch and its output-stationary GEMM fetches each weight once. Table 2 lists 64 KiB **total** die-PE SRAM and 16 MACs at 1 GHz. The input/output split is not published. The primary paper's standalone full-projection output interface and compatibility with the fixed full-vector dynamic A8 fixture are not established. Its fused neuron computation must not replace that arithmetic. This work therefore describes necessary resource counts and a physical row/page construction, not an executable exact-numeric SLIM port.

The original table has 16 channels × 4 chips, while the prose has a channel-count discrepancy. Our 16-die construction is an explicitly bounded subconfiguration; it is not the paper's entire 64-die native system. Its page accounting uses the paper's 4 KiB SLC and 16 KiB TLC configurations separately, without mixing them or importing AiF four-plane/64 KiB waves.

## Actual existing TRAIN cohorts

We reuse the frozen `multi-request-onecopy-screen` cohort rule: adjacent distinct TRAIN prompt IDs, same decode step, all ready simultaneously. There are 32 prompts × 16 steps. B4 yields 128 cohorts and B8 yields 64, separately for existing G4 and Deep layer-0 route caches. These are trace-derived synthetic ready sets, not measurements of serving arrivals. Ordered routes remain unchanged; every request retains six experts, a complete GU/activation/A8 barrier and private Down input.

| Model | B | Mean selected-expert union | Separate routed evaluations | Ideal read saving, including shared weights |
|---|---:|---:|---:|---:|
| G4 | 4 | 19.609375 | 24 | 32.470703125% |
| G4 | 8 | 31.375 | 48 | 47.8515625% |
| Deep | 4 | 20.0390625 | 24 | 31.1279296875% |
| Deep | 8 | 31.234375 | 48 | 48.0712890625% |

For hidden size H, routed intermediate I and shared intermediate S, a routed GU/Down pair has `3HI` W8 bytes and the shared pair has `3HS`. A cohort with union U ideally reads `3HI·U + 3HS` bytes once, whereas separate requests read `B(6·3HI + 3HS)`. In both cases MAC work remains exactly `B(6·3HI + 3HS)`. The displayed percentages are this **batching read-volume** difference. They are neither placement gains nor achievable native latency savings.

`arrays.npz` preserves every cohort's routes, expert multiplicities, union size and per-phase g16 MAC totals. `report.json` includes multiplicity histograms, min/mean/P99/max union and byte counts, exact MAC totals and physical row/page counts.

## Strong minimum-work control and its proof

For each routed and shared operator with shape `R × C`, assign die c its ordinary contiguous logical-row interval `[cR/16,(c+1)R/16)`. It stores all C input columns. All R divide 16. No rows are replicated, aliased or omitted; row scales follow those original logical IDs. Every interval occupies an exact integer number of both 4 KiB and 16 KiB pages, with no complete-row or partial-row fringe.

For TLC 16 KiB pages:

| Model | Routed GU per die | Routed Down per die | Shared GU per die | Shared Down per die |
|---|---:|---:|---:|---:|
| G4 | 64×1536 = 6 pages | 96×512 = 3 pages | 128×1536 = 12 pages | 96×1024 = 6 pages |
| Deep | 176×2048 = 22 pages | 128×1408 = 11 pages | 352×2048 = 44 pages | 128×2816 = 22 pages |

The SLC 4 KiB page counts are four times those entries. For each cohort, `max_die pagework ≥ total_useful_pages/16` and `max_die MACwork ≥ total_MACs/16`. All expert contributions and the shared contribution are individually equal across the 16 dies under this fixed map. Thus g16 attains both inequalities exactly for **every union and every multiplicity vector**, not only for the TRAIN means. It also balances GU and Down separately. Page/MAC placement-only work headroom against this same-batching control is consequently zero in this scope.

This result does not assert that g16 has the best native latency: g-small controls reduce input/command occupancy, and GU→privateDown dependencies, output transfer, bus service and read/compute overlap remain. Any investigation of those effects must give all widths and simple unequal-row/phase-coupled maps the same native batch engine. They are familiar placement/scheduling controls rather than evidence of a new mechanism.

The coefficient-only resident image includes all 64 routed experts and **both shared projections once**. All local jobs have fewer than 512 pages and can be repacked into whole 512-page blocks. BFD matches the simple total-page block lower bound in these cases:

| Model | Coefficient bytes | SLC coefficient-only allocated bytes (16 dies) | TLC coefficient-only allocated bytes (16 dies) |
|---|---:|---:|---:|
| G4 | 148.5 MiB | 160 MiB (5 blocks/die) | 256 MiB (2 blocks/die) |
| Deep | 544.5 MiB | 576 MiB (18 blocks/die) | 640 MiB (5 blocks/die) |

The image is a new static repack, not a claim that previous addresses remain valid. Physical parity, reserved pages, scales and controller metadata are not specified by this coefficient ledger and are not granted free capacity. Both the allocation and numerical-interface caveats remain explicit. Block packing does not assume independently usable plane contexts.

## SRAM and 32/64-die sensitivity

For a single full batched projection on g16, an A8 input needs `B·C` bytes and its 32-bit per-die output needs `4B·R/16` bytes. At B8 the largest sum across all routed/shared projections is 16 KiB for G4 and 27 KiB for Deep. These counts do **not** require treating all 64 KiB as input, but they exclude scale/scratch/metadata buffers and do not prove the unpublished partition or full-vector output ABI fits. The peak-MAC lower bound likewise excludes every other resource.

The optional 32/64-die entries in `report.json` are legal page-exact **row multisets**, formed with row quantum `page_bytes/gcd(C,page_bytes)`. Their counts preserve every logical row once and have zero fringe or copies. They are provided as page-DSE sensitivity only. They do not establish a fixed cohort-balanced mapping, a capacity-equivalent full image, a native schedule or a 64-die gain. No layout was fitted and no larger serving simulator is recommended from these data.
