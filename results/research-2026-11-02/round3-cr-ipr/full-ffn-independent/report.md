Independent full-FFN replay: all88 saved smoke configurations match every C++ node start/end and makespan. 353,640 nodes, 567,536 edges and 11,552 chip jobs were checked. FIFO128 is valid for 84/88 traces under exact continuous2B/ns ONFI drain.

A separate scalar pending-vertex/time-heap executor checks causal timers without busy resources and holds each chip owner from command acquisition until compute plus all ONFI results finish. PCIe does not retain chip ownership. Full GU shards must arrive at the host before whole-vector activation/A8 and full private Down upload; all seven Down outputs precede combine. Produce/ONFI/PCIe bytes and padded native emitted rows are conserved.

Scoped API erratum: the recipe key omits block transitions. Two full waves in one block cost48580ns at strict100; changing the second to block1 returns stale48580ns from the populated cache, while a fresh cache costs66440ns. Current main G4 templates all use block0, so this counterexample does not change the audited primary campaign. No sealed source was edited.

Three manual oracles confirm concurrent resource-free timers, chip release before delayed PCIe completes, and a continuous-drain FIFO peak96B where an incorrect instant-start-clear model would report64B. Five semantic corruptions (missing GU shard, shortened private Down vector, early ownership release, missing shared Down at combine, and a timer occupying compute) are all detected.

The trace audit verifies the saved conditional DAG and scheduler; it does not establish measured native hardware timing or arithmetic/task quality.
6.4GB/s is an explicit ECC service cap and an implicit assumption that the PE can sustain the matching W8 MAC stream; no multiplier-lane or MAC-throughput hardware proof is supplied.
28us cold first availability is service accounting for published tPRE+tEVAL+tDISCH; the numerical first-phase/final-discharge split is unknown.
Skip-WL CR and IPR are user premises. Programmable source order, per-plane cache copies and before-ECC dropping of keep-hot overfetch remain conditional.
Host row scales, positional scatter/fringe removal, full-vector A8 activation at declared0/2us, and zero-cost weighted combine are model assumptions.
Down and private GU recipes reuse block0/WL templates for timing. They do not form a unique-address resident physical-layout certificate across jobs/experts.
Graph caching preserves deterministic node/edge service identity, not arbitrary physical-address identity. A block-omission recipe-cache counterexample is retained; current single-block main inputs are unaffected.
Sorted active IDs set this service model scheduling order. They can differ from original router rank/fit order; this replay performs no arithmetic and does not prove numerical routing-order invariance.
FIFO validity is posthoc. Above a stated budget the unconstrained trace is invalid; no hidden backpressure or larger FIFO is granted.
Only 88 saved primary smoke traces are replayed here, rather than every query of the running TRAIN512 campaign. No new heldout/quality data is opened.

Reproduce: `python /home/ericyu/expert_flash_experiment/scripts/research_20261102_cr_ipr_full_ffn_audit.py`
