# Frozen W8/A8 GU and private Down experiments

All results here are conditional service models driven by existing W8/A8 TRAIN routes. They are not measured hardware latency, full inference throughput, or new heldout evidence. The experiment uses INT8 coefficients and activations; four denotes the plane count, not four-bit weights. The pipeline metadata erratum is in `../pipeline/metadata_erratum.json`.

The strongest current result is wide common-input GU with compact private-input Down. Every GU family received the same thirteen Down geometry/group menus on the same 64 fixed TRAIN indices per layer. Candidate and round-robin retained old TRAIN learned g4 groups; fixed-source g8 GU retained old learned g2 groups. These choices were frozen before replaying all 512 existing TRAIN inputs across three layers.

| GU and selected Down | Input copies | Mean per layer-event | P99 per layer-event |
|---|---:|---:|---:|
| Candidate / g4 old TRAIN groups | 1 | 338.596219 us | 385.962000 us |
| Round-robin / same g4 groups | 1 | 339.961401 us | 395.392000 us |
| Fixed-source g8 / g2 old TRAIN groups | 1 | 348.256142 us | 465.672500 us |
| Candidate / g4 old TRAIN groups | 4 | 344.970646 us | 393.178000 us |
| Round-robin / same g4 groups | 4 | 346.335828 us | 402.232800 us |
| Fixed-source g8 / g2 old TRAIN groups | 4 | 353.963040 us | 471.922900 us |

GF and DF give the same final costs in these selected cases. All twelve configurations complete 1,536 layer-events. The same 512 inputs contribute to all three layers; they are not 1,536 independent prompts.

Compact Down improves the candidate's mean by **20.7205%** against the same GU with g16 Down, and similarly improves round-robin by **20.6545%** at input1/GF. This is a private Down sharding and cold-start effect. It is not new shared-layout gain. After both GU families receive the same Down reoptimization, the candidate's remaining mean advantage is only **0.4016% / 0.3942%** for input1/input4; it is slower on 378 layer-events and equal on 533. A generic compiler emitting the exact candidate layout and stream ties it completely.

For the candidate with unchanged GU, private Down compaction changes total commands/cold starts from 128 to 56 and input ONFI bytes from 90,112 to 53,248. Actual array pages remain 1,216; useful ECC bytes remain 18,874,368; output ONFI and PCIe bytes remain 75,776. Resource work and query latency overlap, so these work differences must not be added as latency savings.

Every private Down job receives its complete original input vector: C=512 for a routed expert and C=1024 for shared. Its full GU reaches the host before whole-vector activation/A8 and the private upload. Down contexts are never treated as a common operand. Fixed-source g2/g4 private jobs use six/three full four-plane waves and pass this strict copy500 model; g1's twelve-wave burst is infeasible and remains a negative result.

All selected traces peak at 128 B in the declared per-page output burst model. They are invalid with 64 B; 128/256/2048 B pass the posthoc backlog check. This does not establish the physical FIFO size. The selected one-copy weight layouts also have explicit single-block certificates: candidate GU+Down ends at WL178, round-robin and g8 at WL149, within 192. Selection comparisons used the same conservative two-block budget for all screened menus.

The 9.7 us producer period, 500 ns copy time, latch overwrite phase, source order, skip-WL controller continuation, and inactive discard before ECC are conditional contracts. Slightly extended read cadence can eliminate a PB hazard or outperform dummy-read shaping; see `../pipeline/cadence/`. Do not treat a fixed-period sensitivity as the unique hardware model. Same-GEMV cold reset assumes input/dot-context continuation; if another ABI requires a new GEMV, it must pay a new command and full input.

The earlier experiments remain sealed:

- `TRAIN512/`: 88 complete cases, including strict100 and cold-reset500; some input4/DF separate-GU and legacy cases need 256 B, unlike the selected asymmetric cases.
- `g8-addendum/`: fixed-source group control; strict100 infeasible queries remain recorded.
- `shaping-addendum/`: uniform split10/prefix1 controls applied equally to candidate, round-robin, and matching layouts.
- `policy-addendum/`: first-feasible PB policy. Reoptimized round-robin reduces candidate GU-only mean gain to about 0.31–0.32%, confirming that shared placement is not the main result.
- `asymmetric-down/`: thirteen-menu screen, frozen winners, complete twelve-case replay, paired attribution, and capacity certificates.

The main v1 recipe cache omitted block from its key. Every actual v1 recipe stays in local block0, so these runs do not exercise that defect. It must not be reused as a generic cross-block cache. The newer policy/asymmetric recipes include complete page addresses, masks, read pages, and settings. Private-job physical relocation is an entire fresh-cold job, not cross-block CR reuse.

## Reusable API

The public graph/timing API is `scripts/research_20261102_cr_ipr_full_ffn.py`: `Graph`, `Graph.matrix`, `Executor.run`, `audit`, and `dump_trace`. The C++ executor is the same prefix with `.cpp`. It schedules one host resource, command dispatch, PCIe, eight two-chip channels, sixteen chip compute resources, owner lifetimes, and resource-free causal page milestones.

For the current experiment, import artifact-local `asymmetric_down.py`:

- `checked_inputs()` verifies frozen hashes and returns protocol, TRAIN routes, frozen Down menu row maps, and frozen g8 GU group labels.
- `KernelRecipes(pipeline, shaper)` provides complete-cache-key finite-PB recipes. `private(rows, columns, owner)` returns `None` for a strict infeasible private job.
- `graph_for(route, layer, GU, down_rows, service, recipes, g8_groups)` constructs a complete FFN graph. GU is `candidate`, `RR`, or `g8_fixed`; `down_rows` has shape `[65,16]`, with expert64 shared.
- `Executor(OUT).run(graph)` returns the trace and modeled cost; `audit(graph, result, full=True)` checks dependencies, resources, owners, and continuous produced-minus-drained output backlog.

Reproduction stages were `asymmetric_down.py freeze`, `screen`, then `run`. Frozen outputs must never be overwritten. Use a separate copied workspace/output destination for a fresh campaign. `asymmetric_summary.py` reconstructs paired attribution; `asymmetric_capacity_audit.py` reconstructs the selected unique addresses. Source snapshots and hashes are in `archive-sources/` and `final_archive.manifest.json`; snapshots preserve bytes and do not automatically relocate workspace-dependent imports.

No heldout was opened, no forty-layer extension was run, and no publication readiness or new optimizer novelty is claimed. Further work should follow independently frozen headroom evidence rather than expanding this branch automatically.
