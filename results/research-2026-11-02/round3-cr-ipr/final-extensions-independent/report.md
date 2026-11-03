All180 saved qindex0 representatives pass exact scalar replay:72 final partition and108 replica traces, 662,220 nodes, 1,083,960 edges and 8,640 chip jobs. Every saved start/end and makespan matches. All180 fit128B outputFIFO under continuous2B/ns drain.

This new driver reuses the earlier independent scalar_execute and semantic_audit. It does not claim a new independent scheduler or replay of everyTRAINquery. Additional checks bind each representative to its saved TRAIN cost/FIFO, exact8192GU+10752Down complete rows, frozen partition chip-row menus, replica expert/copy/row scatter, full private inputs, fixed-source native Down jobs and resource busy work excluding timers. FullGU-at-host→activation/A8→fullprivateDown upload and allsevenDown→combine, channel exclusion and chip ownership through compute+ONFI are checked; PCIe is outside ownership. Seven corruption controls are detected.

All27 frozen replica resident layouts are independently expanded to unique page coordinates with no GU/Down collisions, complete1536row expert-copy coverage and at most2blocks/chip. Onecopy anchors program155713536B/layer and use32chipblocks in this sealed intact-block1 placement; triples program256376832B, +96MiB, and use28. This differs from the earlier valid compact-firstfit oracle placement, not from its common32block budget.

The candidate restricted-calendar lower bound is280510ns for input1 and285422ns for input4. Replayed representatives cost280526/285438ns, a16ns gap; the saved1536cost entries per profile agree. The bound fixes GU releases, sharedg4 and native384row jobs and is not global over arbitrary geometry.

Only qindex0 representative traces are replayed:72 partition and108 replica. EveryTRAINquery is not independently replayed; fullarrays are read only for representative binding and the explicitly named candidate constant-cost check.
Saved nodes/edges are the replay input. This audits conditional scheduling, complete-row counts, dependencies, FIFO and resident metadata; it performs no GEMV arithmetic or task-quality test.
The scheduler and primary semantic checker are reused independent code, not a new second independent scheduling implementation. Audited sources and their own audits are not imported.
FIFO uses continuous2B/ns drain, not instantaneous clearing at ONFI start. Resource-free row timers contribute no resource busy work. PCIe does not retain chip ownership.
PB feasibility is the sealed job-recipe declaration, whose timing contract was independently audited previously; this pass does not reconstruct every array/ECC/cachecopy page deadline.
Skip-WL CR/IPR, source permutation and preECC dummy dropping remain conditional; native PE sustain6.4GB/s and cold28us firstavailability are modeled accounting, not measured firstphase/discharge evidence.
Resident coordinates, row extents and fixed operand dimensions are checked; physical coefficient programming and original-model numeric/ABI equivalence are not established.
Anchor Down jobs are placed intact in block1, thus32nonempty chipblocks in this sealed layout; triples use28. The earlier independent firstfit witness used fewer anchor blocks. Both fit the same32block budget.
The280510/285422ns bound preserves current GU release calendars, sharedg4 and384row private-job geometry. Its16ns gap is not a proof of optimality over other GU/Down geometries or hardware.

Reproduce: `python /home/ericyu/expert_flash_experiment/scripts/research_20261102_cr_ipr_final_extensions_audit.py`
