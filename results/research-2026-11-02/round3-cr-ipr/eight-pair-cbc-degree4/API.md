This is the frozen standard CBC baseline with eight two-chip pairs. All64 experts have distinct degree4 support sets and every pair stores32 experts. Shared Down has two output-row halves, S0 and S1, with a complete copy of each half resident in every pair.

The builder is `scripts/research_20261102_eight_pair_cbc.py`. Its public functions are:

- `build_layout(supports=None, gu_reserved_wl=128)`
- `build_private_job(expert, group, local_chip=0, layout=None)`
- `build_shared_job(half, group, local_chip=0, layout=None)`
- `assign_route(route, layout=None)`; `dispatch` is the same function.
- `execution_view(bundle)` and `resident_jobs(layout=None)` return existing FFN execution-view records.

`assign_route` matches six distinct expert IDs to six distinct pairs, then puts S0/S1 in the two remaining pairs in increasing pair order. It returns `assignment`, `shared_half_assignment`, `group_owners`, and `selected_jobs`. The last contains16 records, one per physical chip, with `expert`, `copy=0`, `resident_copy`, `shared_half`, `resident_job_id`, `chip`, `rows`, `columns`, `block`, `wl_start`, `pages`, `row_start`, and `reserved_WL`.

Both shared halves use expert64 and execution copy0. IDs start with capital `S0` or `S1`. Their four selected row_start positions are0,384,768,1152, while every job retains the full1024-byte input. Private jobs use two768-row bands and full512-byte input. Half denotes output rows only. Each job keeps original canonical coefficient rows and scale indices `[row_start:row_start+rows]`; no host partial sum or extra dot context is required.

Each pair stores32 private jobs plus S0/S1. Ten six-WL jobs occupy block0 WL128..187 and24 occupy block1 WL0..143. Each24-page job stays within one block and uses six same-WL fixed0123 waves. The default layout fits GU reserved through WL127. Replicated GU requiring186WL is rejected by this local two-block allocator.

The independent verifier is `scripts/research_20261102_eight_pair_cbc_verify.py`; its sealed result is `verification_report.json`, status `PASS`. It does not write `report.json`: any root adapter gate receipt with that name is only a compatibility pointer to this verification and its hashes, not another independent checker.

The mathematical certificate checks all256 server subsets, which establishes Hall for every request of six distinct experts plus the two fully flexible shared halves. The2,892 synthetic runtime routes are additional implementation checks, not an enumeration of74,974,368 requests. The physical ledger covers19,392 pages and303MiB of useful GU+Down coefficients; canonical byte ranges and row positions are certified, but weights were not programmed or numerically retested in this task.

This layout was proposed after prospective failures were observed. Future replay of previously observed prospective data is diagnostic reuse, not a new heldout evaluation. CBC, replication and matching are standard ideas. No new theorem, solver, hardware, measured latency or globally optimal storage claim is made.
