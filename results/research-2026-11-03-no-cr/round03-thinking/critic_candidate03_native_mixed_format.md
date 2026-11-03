# Candidate03 native mixed-format: adversarial source/math review

Scope: source and mathematical review only. Formal Root hypothesis starts 2026-11-03 10:35:14.871021 UTC; this critic's independently recorded source-review interval starts **10:36:05 UTC**, not backdated. No candidate code, build, numeric trial, synthesis, simulation, model/weight/activation/target/book/TEST payload or network dataset acquisition is authorized or performed. This is not experiment approval, and 2 failed methods / 8 opportunities remain unchanged.

Root's hypothesis SHA: `93ef0eeb7265a4cfdbd19eb67271efc793bc50ce132552fdd0d1e266e7ef822f`. Main question: whether a finite native-block LUT window creates a useful NAND resource frontier against the same-format expander+MAC and ordinary-loader paths. Generic LUT generation, raw Q4 storage, FFN fusion and input tiling are already prior; the question cannot be answered by comparing native Q4 against the previous W8 reconstruction.

## Checkpoint 10:40 UTC: numerical state is larger than an I32 output

Read source only in `/home/ericyu/llama.cpp-aif/ggml/src/`:

- `ggml-common.h` SHA `33667042e675da25d8ea0c6d5d0999506f1edfce2b7d97c2d78830ac5fb3fa46`: Q4_K = 144 B/256 values (128 codes + 12 scale/min bytes + 4 FP16 bytes); Q6_K = 210 B/256 (192 code bytes + 16 signed scales + 2 FP16 bytes). Q8_K = 292 B/256 (256 I8 + 16 I16 sums + F32 d).
- `ggml-quants.c` SHA `62a1a886245c8a59af5f5bc8fe50ffdb71167fcd5a8321bc0aa67d6d55ad4b11`, Q8_K ref at2692: signed maximum sets `iscale=-127/max`, so `d=1/iscale` can be negative. Do not reuse the former positive-scale canonical API. Per256 amax is real for this chosen source, not a theorem about every backend. Specify zero/nonfinite/tie behavior rather than inheriting unspecified scratch state.
- `ggml-cpu/quants.c` SHA `ac93fdaa58385f782c89c8191758b90cf03712516e9724adb31a345716237ad3`, generic Q4 at645, Q6 at800: eight F32 sums persist across native256 blocks; Q4 additionally accumulates its minimum correction separately. The x86 source at1900/2288 instead uses eight FMA lanes, plus four minimum-correction FMA lanes for Q4. Generic and AVX2 are different numerical references.

For R simultaneously live output rows, generic Q4 needs at least36R bytes for those F32 chains; AVX2 Q4 has48R bytes before integer scratch, row state, metadata, ECC staging and output buffering. At R256 these are9/12KiB. A 2KiB LUT can therefore exchange table capacity for unfinished-row capacity rather than reduce the whole live state.

The contiguous g4 subset-LUT identity merges four terms belonging to distinct generic F32 lanes. It proves an integer dot identity, not original F32 output bits. A strided g4 arrangement can preserve one Q4 generic lane within a32-weight same-scale group; Q6 has16-weight scale groups and needs different grouping or explicit partials. One may instead declare a different reduction ABI and verify quality, but must not call it original-source bit exactness. The same direct-MAC and host controls get the selected ABI.

Source Q4_K/Q6_K dot functions require inner dimension divisible by256. A shape such as the previous Deep intermediate1408 cannot silently use Q6_K Down; native fallback or a paid, explicitly changed padding representation is necessary. This is an arithmetic shape fact, not a read of its actual GGUF tensor types.

## Finite correction of Gem exploration block02

The original block02 note remains intact. Its correction of block01 is useful, but several replacement claims cannot serve as evidence:

1. Read-out parallelism does not require a monolithic16-port SRAM. Registers plus independent MUXes, banking, controlled replication and serialized fetches are alternatives. Their RF/MUX/port area is paid, but no technology-library result establishes `>0.07 mm²` or a universal multiple of MAC area.
2. A4.2GB GGUF file suggests OLM residency may be feasible; it does not close inclusive8GB or prove CPU dominates every in-flash implementation. Conversely, residency cannot be prohibited to manufacture misses.
3. Three waves for an entire Q4_K expert omit real total bytes, four-plane/chip occupancy, scales/min, page ceilings, GU→Down dependencies and partial-state service. They are not an achieved or valid general bound.
4. No distribution or map supports the stated `>99.9%` chip-collision probability. Total shared-chip work creates a deterministic resource constraint; selected experts can also be serialized, row-striped or assigned replicas under paid capacity. Ordinary-load receives the same legal replicas and schedule. Neither collision nor a PE lower bound establishes ordinary-loader dominance without a feasible matched comparator.
5. Down tables cannot be presumed shared across experts with different h. They can be served sequentially by one bounded table RF; K independent inputs do not logically require K simultaneously resident full tables. Either serialization or simultaneous contexts must be charged.

These are finite corrections, not new gates and not a veto of every LUT implementation.

## Primary architecture comparison still required

[Geens et al.](https://arxiv.org/html/2604.25183v1), §§II-C/V-C/VI, already model LUT-generation registers, configurable read-out MUXes and tile geometry. Their integer-activation result warns that read-out cost can erase adder savings. This is ternary/16nm, not a Q4_K NAND area certificate. [PALUTE](https://arxiv.org/html/2606.08891v1), §§3.2–3.4, already has bank-attached input-table generation and scheduling; its DRAM substrate cannot supply NAND ports/PPA. These priors consume a generic generator/window narrative, while leaving this numerical-format/physical-page combination unverified.

Provisional status at this checkpoint: **no fatal impossibility proof, no value endorsement**. The resource difference must be made concrete; a format port alone remains too weak.

## Checkpoint 10:46 UTC: lookup counts, ECC packing and the movable F32 boundary

For g4 subset tables, one256-value Q4 block needs64 groups×4 bit-plane queries=256 LUT fetches. Q6 needs384. With16 fetches/cycle, ideal dot service is16/24cycles, before generation, code/mask extraction, subscales, shifts, row-state and F32 work. A16-lane native4×8 or6×8 direct MAC does256 products in16cycles. This is a counterexample at equal lane count, **not an equal-area proof**: different MUX and multiplier costs, clock rates and banking can alter the result. A valid same-area advantage must establish more useful fetch throughput or another paid resource saving, not assume one lookup replaces four full low-bit products.

The table RF is not obliged to be16-port SRAM. Sixty-four small tables can be banked so distinct input groups use distinct read-out MUXes; extra fetchers per table are separately charged. A direct-MAC comparator has the input codes and decoder and receives the same compressed pages/frontier. It may choose its own efficient legal layout, in addition to a same-layout attribution control. Geens' configurable LUT/FAC/output-buffer tradeoff already covers much of the generic resource question; its ternary results do not settle the present Q4/6 cell costs.

ECC does not require every144/210-byte code block to be padded to1KiB. Dense streaming can carry a block across corrected-codeword boundaries, with a bounded carry buffer, retained metadata and an explicit decoder/PE pause contract. As alternative layouts, sevenQ4 blocks fit1008B of1KiB (1.5625% padding); fourQ6 blocks fit840B (17.96875% padding). At full16KiB page packing,113Q4 blocks use16272B and78Q6 blocks use16380B; imposing a multiple-of8 row tile can lower the first count to112. Neither dense packing nor these counts proves legal multi-plane alignment or free random access to PB. The physical mapping must say where a split block/row resumes.

If on-die F32 is too costly, a legitimate bounded alternative is to return integer lane partials and retain the exact selected F32/FMA order on host. Generic Q4's8I32 plus minimum integer correction is36B per144B block; Q6's8I32 is32B per210B, before identifiers and necessary d/dmin metadata. These are approximately25%/15.24% of compressed weight payload, not the original AiF one-result-per-whole-row traffic. Host FMA state/work, FIFO and transfer service then count. This option prevents the critic's F32-state warning from becoming a false requirement to place all F32 units in die; it also prevents the candidate from hiding those costs.

The generator has a reuse threshold. A g4 binary subset table has11 nontrivial additions after the four singleton inputs and zero are supplied;64 such tables require704 integer additions per256-input window, plus materialization/ports. Serially rebuilding this for every output block is unfavorable; input-major layout amortizes it across R rows but keeps their output states live. Full-H table residency avoids those rebuilds at the cost of8N table bytes (I16 entries). Smaller windows are thus a real space/time tradeoff, already a known tiled-compute pattern; a NAND-specific contribution still needs a distinct resource consequence.

## Checkpoint 10:46:20 UTC: frontier attribution has a concrete null example

Source Q8_K per256 input quantization permits a256-neuron downstream frontier, but the **same-format direct-MAC and ordinary-load controls have the same frontier**. Eliminating the former artificial whole-expert A8 barrier is not a LUT contribution.

For the shape-only example H2048/I1024, all three projections in Q4_K use GU=2.25MiB and Down=1.125MiB. A fully useful16-chip×4-plane×16KiB wave carries1MiB; separate full phases have byte ceilings3+2=5waves. One256-neuron GU tile uses0.5625MiB, its Down input-block slice0.28125MiB. A simple first-GU + three mixed GU/Down + final-Down schedule also takes5waves under otherwise favorable placement. This is not a general schedule lower bound, but it is a decisive counterexample to claiming that this frontier automatically removes sensing waves. Smaller fragmented tiles can add waves. The host nonlinear overlap and partial-accumulator opportunity are common to SLIM-style/direct-MAC controls.

At this checkpoint the novel necessary difference remains unestablished: native compressed reads change bytes relative to W8 for **both** candidate and matched controls; LUT alone changes a compute implementation. A credible next gate would need an explicit equal-resource compute/energy frontier or a concrete NAND-only resource coupling. Unsupported library-area numbers and mere absence of prior exact-format support cannot supply it.

## Checkpoint 10:48:17 UTC: one arithmetic objection is repairable

The earlier contiguous-g4 warning is specifically about the **generic** eight-lane reduction. In the inspected AVX2 Q4 kernel, `maddubs` forms adjacent pairs and `madd` combines them into four-term integer groups before subscale accumulation. Thus contiguous g4 can preserve this chosen grouping. It must still preserve all eight main FMA lanes, four minimum-correction lanes and final horizontal-reduction order. Q6 must preserve the actual AVX2 distribution of its `q8sclsub` correction among lanes, rather than subtract a mathematically equivalent scalar correction after an arbitrary repartition. This is a feasible numeric-design direction, not a proof that a candidate implementation exists.

Offline transposition of four quant codes into bit-plane mask nibbles can retain the same number of raw code bits and exact affine quantization parameters. It is a custom storage ABI rather than byte-identical GGUF layout; descriptors, codeword packing and preprocessing are charged. A direct-MAC/ordinary-host control may use the same packing, or its own better legal packing. This repair does not manufacture a native LUT throughput/area benefit.

The finite unresolved choice is now explicit: (a) full source-lane F32 state/FMA on die, or (b) integer lane partials returned to host. Neither may borrow the other one's low traffic or low area. Allhost source-format execution remains a live control, including inclusive capacity and same nonlinear function.

## Checkpoint 10:52:36 UTC: formal Gem block01 requires finite corrections

Read `gemini_formal_block01_textonly.md` as an unapproved source/math draft, not new measurements. Its source-format and finite-row warnings are useful; its supposed fatal examples are not evidence for rejection:

- `2^24` has an upward F32 spacing of2, not1. Its `(A+1)+1` versus `A+(1+1)` example differs by2 absolute units, one upward ULP. It illustrates nonassociativity, but does not itself instantiate a complete valid Q4_K block/scales example. Generic reduces its eight F32 chains after the **whole row**, not after each256 block.
- For four-bit weights a g4 subset table needs four bit-plane queries per four weights, not one query per four weights. With Q6 generic g2, six queries serve two weights:768 queries per256 block. B1 does not destroy LUT reuse across R output rows or the two GU projections; claiming the T256 input is used only once contradicts the proposed layout.
- Aligning complete blocks within a16KiB page does not align144/210-byte blocks to1KiB ECC codewords. A block may still cross a codeword; a corrected streaming/carry contract is needed. An in-die AiF compute path is not obliged to transfer all weights across the external1.6GB/s NV-DDR interface first. The extra40us and the resulting68us roofline are inapplicable as a general in-flash timeline.
- Its listed Q4 working-set formula is4388+44R bytes, yielding10020/15652/24100B atR128/256/448. Its Q6 g2 formula is3364+40R, yielding8484/13604/21284B. Even corrected, this is one assumed two-CW-buffer/FIFO/ID design, not a universal minimum; implicit row IDs, backpressure, host lane state and smaller staging may change it. The active16-table plus remaining56-table inventory also needs correction because64−16=48, unless extra replicated copies are explicitly counted.
- No cell library supports its250–350GE multiplier or<2ns claim. No feasible comparator supports a global delay/area dominance conclusion. A resource roofline being below sense bandwidth does not hide compute if the actual PB ownership forces serial sense/ECC/compute. Conversely, a cell co-locating h does not force every Down row to remain there: paying a256-activation broadcast can avoid the asserted mandatory all-reduce. Neither communication path is free, but the alleged architecture-wide impossibility does not follow.
- `g4 impossible for Q6` is too broad: that statement only concerns the specific generic stride8 / same-subscale grouping without decomposing partials. An explicitly pinned AVX2 ABI can use contiguous four-term unsigned products plus its original lane-specific offset correction. Changing ABI must be declared; no arbitrary zero-cost sign-bit shortcut may replace the actual FMA semantics.

[LUT Tensor Core](https://arxiv.org/html/2408.06003v1), §§III-B/C, already gives bit-serial precision flexibility, elongated reuse and typed LMMA metadata/compiler support. Its §IV-B2 includes INT8 activation configurations and reports favorable LUT area/power in most investigated cases. Thus neither Geens' **ternary** low-INT-return observation nor simple query counts proves a general low-bit MAC dominance theorem. [FIGLUT](https://arxiv.org/html/2503.06862v1), §§III-C/D, supplies conflict-free FF/MUX read-out and half-table techniques; the full2KiB table is not the only known LUT architecture. All these are non-NAND and do not establish the proposed source-exact PE's PPA.

## Provisional adversarial disposition sent to Root

**Recommend thought-veto of this concrete candidate's publication/value case, with no experimental opportunity consumed.** This is not a theorem that LUT cannot outperform MAC, and does not use absence of a PDK as an impossibility proof.

The reason is concrete attribution closure: grant the native-format direct-MAC engine the same compressed format, legal block packing, early256 quantizer frontier, original nonlinear operations, input reuse, and choice of host partials or on-die source FMA. Then the NAND read bytes, commands, ECC work and external dependency DAG are identical. A bounded LUT window trades table state against output state/rebuild traffic; it does not remove another sensing operation. The remaining implementation difference is useful lookup throughput/energy versus native4/6×8 MAC under area, a design question already central to LUT Tensor Core / FIGLUT / Geens. Native block scale/min and old typed precision/tile constants do not currently create a new NAND resource separation.

There could be a favorable PPA point, and a NAND-constrained accelerator could in principle be publishable. But no such **new mechanism** has been identified here; starting an experiment merely to discover whether known LUT compute wins a library constant would not validate the claimed new joint quantum. A later distinct physical primitive or a concrete nonlinear resource separation may merit a fresh hypothesis. The present finite critique asks for neither global optimality nor all-layer success, and uses no weak-baseline or two-lower-bound comparison.

This recommendation is an **early thought rejection**, not a claim that either blue thinker has completed1800 substantive seconds. Their actual intervals and failed/quota calls are Root's independent records. No candidate experiment, model probe, synthesis or implementation was performed by this critic.

### Final numerical qualification and closure

The host-partial36B Q4 example above is **generic ABI**, not the naturally contiguous-g4 AVX2 construction. AVX2 preserves eight main plus four minimum I32/FMA chains, hence48B integer partials per144B block (one third of compressed payload), before d/dmin or IDs. The two traffic counts cannot be interchanged. Q6 AVX2 still needs its exact eight lane-specific corrected partials. This changes costs, not the matched-DAG attribution argument.

Root accepted the finite early thought-veto at the final source-review boundary. The critic review started10:36:05UTC and ended**10:55:34UTC**, an interval of1169seconds. **It ended before1800 seconds** and does not claim the candidate met the research-duration gate. No candidate experiment was begun. Final campaign state remains2 failed experimental methods and8 opportunities. A later macro pool has zero formal method-clock credit from this rejected candidate.
