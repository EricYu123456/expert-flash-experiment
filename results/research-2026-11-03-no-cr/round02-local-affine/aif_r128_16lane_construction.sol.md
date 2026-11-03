# Conditional r128/16-lane AiF construction

Source/theory review started2026-11-03 07:14:25 UTC, same Round2. No RTL, synthesis, simulation, benchmark, candidate source modification or target/DEV access. This is a falsifiable hypothetical construction, not native-AiF capability, a proved minor PPA change, publication readiness or a full-model speed claim. Stage3 is unchanged.

## Explicit new block and numerical contract

Assume one **chip-wide** PE with16 signed-I8 multipliers accepting16 useful products/cycle at200MHz. Four planes share this PE sequentially. AiF's paper does not disclose native lanes; its6.4GB/s number is an ECC decoder rate. The independently sourced BlockDialect/NVDLA comparisons in `docs/aif_pe_lanes_ppa_followup_terra.md` do not establish AiF's lane count, total PPA or operand bandwidth.

The proposed added/changed resources per chip are:

* A128-byte input RF, logically eight128-bit words, one128-bit read/cycle; input writes occur before execution. It supplies16 qv bytes/cycle without a PB input-read port. Its1024 storage bits, routing, load mux and enables are paid; it is not a claim of free PB input reuse.
* A1KiB **corrected-codeword staging buffer**, with a single256-bit port used in separate fill/read phases, plus one32-byte read holding register. Fill accepts up to32B/cycle only if the ECC interface provides it. During consume, one256-bit read every two cycles supplies two16B PE beats. There is no simultaneous ECC fill/PE read and no dual-port assumption. ECC's own raw/parity scratch and PB-to-ECC service are separate disclosed costs; they are not hidden inside this1KiB.
* A128-entry I32 result FIFO:512B payload, one write or read phase at a time, plus pointers/count/credits. It holds exactly one plane's128 row results. Do not reuse SREG status as storage. A context-tagged512B bulk drain and completion handshake are required.
* One context/tag descriptor, row/page/beat counters, pipeline valid/last-row metadata, error/abort state, FIFO reservation and ready/valid backpressure. No second token/expert context or input bank is assumed. The example counts at least1696B of explicit payload/RF/holding storage per chip, before descriptors, pipeline registers, ECC internal scratch, parity and control. Across16 chips that is27136B, not a PPA estimate.
* A16-product signed reduction tree and I32 running sum. A registered multiplier/tree can have latency p cycles while accepting a group each cycle; p and200MHz timing require actual STA. Preserve row/last metadata through every pipeline stage. On the eighth group's completion, enqueue the final sum and clear before the next row's first sum. Full-pipeline clock enables/backpressure must preserve order and prohibit lost/repeated beats.

For full signed[-128,127], each product fits signed16bits;16 positive maxima sum to262144, requiring20 signed bits at the tree output. A128-product row can reach2097152, requiring23 signed bits. I32 is safe. Under canonical[-127,127], the row maximum2064512 fits22 signed bits. All arithmetic is exact, with no saturation, wraparound or intermediate narrowing. Host performs the frozen F32 `s=sw*sx; f=(float)dot; out=f*s`, then bias, actual final alpha and original slot-add order. RF stores only qv; scales/bias are separately resident/fetched and billed. Native I32 wire format is a new explicit contract, not an inference from the paper.

At16 products/cycle, r128 takes8 issue cycles/row. A64KiB chip payload has512 rows and4096 issue cycles:20.48us **ideal issue demand**, before p, codeword gaps, ECC, input, output or stalls. A row is wholly inside a16KiB page, so no cross-page accumulator is required.

## Read-first command extension; no hidden PB double buffer

The paper's described sequence loads input before sequential reads. This design changes it:

1. `PREP_READ(tag,addresses,rowbase,shape)` reserves the one context, senses one ordinary LSB page in each of four planes, and holds the resulting data latches valid.
2. After SenseReady and qv readiness, `POST_Q(tag,128bytes)` loads RF; `EXEC(tag)` starts ECC/PE. Conservatively, input posting waits for array completion: no RF command/data acceptance while NAND is busy is assumed. Supporting a truly concurrent POST during sense would require an additional command/port/busy-state contract and is not this baseline schedule.
3. `READ_RESULT(tag,plane,rowbase,128)` bulk-drains that plane's FIFO, followed by an acknowledgement. The next plane starts only after it is empty. After all four planes and final status are acknowledged, release the context; only then may that chip start the next expert/read wave.

This needs command decoding, explicit prepared/input-valid/sense-valid state, PB hold-valid/abort rules, tag checking, RF load routing and ECC/PE/FIFO handshakes. Address-known-before-P can therefore remove a **P→sense** edge, but adds no proof stock AiF accepts these commands. Digital PB retention is not CR: no cross-wordline charge is retained, and all array reads use ordinary LSB behavior. Specify a finite hold/timeout envelope; expiry or uncorrectable ECC aborts the context. Partial outputs are not committed until whole-context success.

Use only the LSB data latch needed for the held page at each plane. The remaining original latches are **not** credited as spare next-page buffers. No next read, cache-latch transfer or array/ECC overlap is assumed. The input RF avoids PB operand-port contention, but it does not prove any PB→ECC width/rate. Effective codeword service must include that path, parity and ECC scratch. Adding a wider PB mux/port if needed is another paid hardware change.

## Resource ownership for one wave

Each expert owns four chips; each chip holds four16KiB pages,512 rows total. Process plane0,1,2,3 in a fixed order. The proposed global wave schedule uses a barrier after each plane: all active chips finish its codewords, then controller drains their FIFOs in a fixed order. It sacrifices overlap deliberately to make finite storage and ownership explicit.

| Phase | Array / PB | ECC / staging | RF / PE | FIFO / host |
|---|---|---|---|---|
| Prepare/sense | Four planes sense current pages; their data latches become held weights | Idle | Context reserved; RF not executing | Empty; no old context outstanding |
| Wait for qv / POST | Array idle; all four pages held | Idle | RF written once after readiness; no PE reads during write | Empty |
| One1KiB correction | Selected plane PB read supplies raw/parity; other pages held | One decode; staging filled, no PE access | RF retained; PE idle | Existing results retained, no drain |
| Same1KiB consume | Array idle; PB pages held | Decoder idle; staging read-only through32B holding register |16 products/cycle, eight rows; then wait pipeline completion | Reserve before row launch, enqueue exact I32; no host read |
| Plane complete | All pages still held | Staging idle/empty | PE/pipeline idle; RF/context retained |128 entries=512B; drain active chips in fixed order and acknowledge |
| Next plane / finish | No latch overwritten before all use ends | Reuse staging after acknowledgement | Same RF/tag until all four planes complete | Empty before next plane; release tag only after final drain/status |

There are16 codewords per page and64 per chip/wave. FIFO credits include in-flight rows, not just already written entries. The page has exactly128 rows, so128 entries suffice without concurrent host drain; stopping after each codeword's pipeline completion makes reservations straightforward. If an implementation cannot stall the decoder output, the complete corrected codeword still fits staging. It must not launch another decode until staging is available. No simultaneous PB program/read, staging write/read or FIFO write/read port is required by this schedule.

## Model-conditional completion upper schedule

Define **assumed service envelopes**, not silicon measurements: prepared-command service bound A_i; successful ordinary sense bound S_i; post-input bound Q_i; one-codeword PB/parity/ECC-to-staging bound E_i; an issue/pipeline bound P_i=(64+p_i+b_i)/f_i for1KiB, with f_i at least200MHz after timing closure; and per-plane tagged FIFO-drain bound D_i,j under a fair controller/ONFI/PCIe/host reservation. Here p_i extends through final accumulator/FIFO enqueue, and b_i includes the initial staging-SRAM/holding-register read and any bounded control/reset bubbles. Include setup/polling and host-contention delays in these envelopes. A published peak rate is not a guaranteed minimum-service envelope.

For active chip set G, address-ready times a_i and actual common qv-ready time q, one deliberately conservative schedule waits for all addresses, posts PREP commands sequentially, waits for the last possible sense completion, and then posts all input vectors sequentially. Its feasible execution-start bound is

`B = max(q, max_i(a_i) + sum_i A_i + max_i S_i) + sum_i Q_i`.

The deliberately serial page schedule then has a conditional bound

`C_wave <= B + sum_j=0..3 [ max_i(16*(E_i+P_i)) + sum_i D_i,j ] + finish_bound`.

The sums of posts and drains are conservative but constructible: grant chips one at a time; do not assume unbounded parallel control/output ports. E includes all raw/parity transport and decode latency. P includes p and b at each codeword because the chosen schedule waits for pipeline empty. Controller/execution-control gaps belong in E/P/finish rather than vanishing. PB hold timeout must cover this schedule and a bounded q-ready deadline. This is an upper bound **inside those declared envelopes and successful-read conditions**, not a hard silicon upper bound. ECC failure/retry needs a finite separate policy and failure receipt; an uncorrectable context is not a valid completed GEMV.

The nominal ledger28us sense +10.24us ECC +20.48us PE=58.72us omits guaranteed link delivery,64 pipeline fills, RF post, metadata, output and software contention. It is therefore **not** the bound above. Even10.24us ECC requires effective6.4GB/s PB/decoder/staging service, which is unproved. A1KiB decoder may have substantial startup/control service beyond the byte-rate arithmetic.

K6/K8 all-miss shapes use four disjoint four-chip groups. Wave1 can occupy four groups; wave2 occupies respectively two/four, with no speculative same-chip next-wave read. Each group's second command waits its first context/output completion and its own qv post; a conservative global barrier waits all wave1 chips before starting wave2. Count actual cache hits/fallback groups rather than charging two full waves universally. q is shared only where the frozen layer P/tree genuinely are shared. Host computation of q, post/poll, metadata/rescale/parse/final alpha and source shared competition remain in the enclosing DAG.

## Optimistic lower bounds and fairness

Generic independent-resource sensing/compute/bus maxima are lower bounds only. For **this specific SenseReady-before-PE, no-next-read-overlap construction**, a stronger ideal chip bound is that a busy wave cannot finish its PE earlier than `max(earliest_SenseReady,q_ready)+20.48us`, ignoring input posting/ECC/fill/output. With addresses and q already ready and nominal28us sense, that is48.48us; a chip busy in both waves has at least96.96us of these serial demands. These are not universal lower bounds for every conceivable AiF datapath or a claim of achieved timing. Releasing data before SenseReady or a different multi-page pipeline requires another explicit ownership/port design.

The strongest same-bank ordinary-load CPU baseline receives the **same** address-known early read, ordinary array/ECC capabilities, legal cache budget and layout. It can DMA weights while host computes P, compute rows as they arrive and prefetch a next ordinary read once its PB ownership actually permits. It need not mimic this PE schedule's waiting after its own PB data have been exported. Conversely, neither side gets free double buffering, command overlap, input contexts or host cache. CPU exact integer GEMV and all-host identical-map cache remain legal. Compare attainable schedules, not the PE's peak demand against a weak serial loader.

One can falsify the proposed point by showing that the declared PB/ECC service or RF/PE timing cannot be supplied without extra ports/lanes/buffers, that finite drain/hold envelopes fail, or that its matched-function best-resource service is dominated by an attainable ordinary-load/cache/CPU implementation. Failure of an unproved service assumption is not proof all in-flash designs fail. Conversely, satisfying the arithmetic model alone does not establish PPA or useful whole-model acceleration.

PPA must include the integrated multipliers/tree/accumulator, RF/staging/FIFO, tags/control, added PB/ECC/ONFI muxes and wires, clocks, enables, library/corner/timing and activity. Multiplying an unrelated scalar-MAC area by16 does not give this block's total or prove it fits AiF's0.026mm² PE figure. The incremental area relative to stock is unknown because stock internals are unknown. Claim only a concrete parameterized design and the additional evidence needed; later RTL/STA/PPA and protocol review require a separate release.

Source/theory receipt:2026-11-03 07:14:25–07:22:48 UTC. Only AiF text, Terra's source audit and existing theory were consulted; no Stage3 source or protocol was changed.
