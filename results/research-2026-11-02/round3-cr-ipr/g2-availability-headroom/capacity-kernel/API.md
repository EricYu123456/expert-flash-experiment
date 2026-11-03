# g2 complete-row Down builder

`kernel_builder.canonical_supports()` returns64 expert support lists. The first60 experts are four experts per lexicographically ordered four-subset of groups0..5. Experts60..63 use012,345,034,125. Each group contains42 complete expert copies.

`build_layout(supports=None, gu_reserved_wl=128)` accepts the canonical map or a frozen expert-label permutation with the same60degree4/4degree3 profile and42copies/group. It returns actual per-group resident expert IDs and whole-job block/WL bands. It rejects the186WL GU replica placement because its local capacity cannot hold252DownWL within two blocks.

`build_private_job(expert, group, local_chip=0, layout=None)` returns `waves` and `metadata`. For a fitted support map, pass the returned fitted `layout`; omitting it selects the canonical map. `group=chip//2`, `local_chip=chip%2`, on privatechips0..11. Each job has24fullpages,6fixed0123 same-WL waves,768rows and a512-byte private input. Its source rows are `[local_chip*768,(local_chip+1)*768)`, so the two chips cover1536 complete rows exactly. Each copy uses one command and one cold start; no job crosses a block.

`build_shared_job(local_chip=0, gu_reserved_wl=128)` covers sharedchips12..15 with `local_chip=chip-12`. It returns24pages/6same-WL waves,384rows and a1024-byte private input, with source rows `[local_chip*384,(local_chip+1)*384)`.

Call the unchanged pipeline with `source_policy='fixed'`, `cache_copy_ns=500`, and `row_bytes=512` or1024. Both kernels are feasible at89940ns. No pre-ECC drop, dummy read, IPR, PB source permutation, input replication context, FIFO or idle hold is used. Native skip-WL CR is the user premise, and the existing conservative first-ready/PB timing envelope is inherited.

`assign_route(route, supports=None, capacity=1)` is an ordinary augmenting-path feasibility helper. Try capacity1 first; capacity2 guarantees feasibility for this minimumdegree3 support graph. Capacity2 alone does not minimize group load: it may return two jobs/group even for a route having a six-way perfect matching. It is not a training optimizer or novel solver.

No TRAIN, heldout or fullFFN is read here. The physical address ledger checks the conservative GU128WL reservation and all whole Down jobs. The source-byte offsets preserve complete dots; synthetic full-shape W8_A8 packed fixtures confirm exact row/scatter output, without making a real-model numerical validation claim.
