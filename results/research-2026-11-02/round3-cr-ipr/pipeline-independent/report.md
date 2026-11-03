Independent pipeline audit passes 37 named trace cases, 200 generated mask/configuration cases, 8 saved trace files and 5 corruption controls. No target-code discrepancy was found.

The full candidate requires one read and one cache weight slot per plane, one serial ECC, ordered common-input rows and no hidden weight FIFO. For strict CR, evaluation-start overwrite and copy/recycle overlap, the exact copy boundary is 320ns: 320ns passes and 321ns fails. The limiting wave10/plane2 read-latch slack is 640−2×copy_ns. Copy0/100/500ns gives hypothetical makespans150880/150980/151380ns; 500ns is infeasible.

The independent reference builds its own physical pages and max-plus read/cache/ECC dependency graph. It imports the target only to generate candidate traces, then separately checks overwrite deadlines, cache retention, per-plane two-slot inventory, exclusive resources, per-plane CR continuity, startup/drain, row ownership and byte/page conservation. Four manual time oracles cover single-page, four-page, two-wave and newly active cold-plane cases.

Ordinary balanced partial masks require7 cold+8 CR waves. Keep-hot/drop reads60 pages and decodes48; keep-hot/decode reads and decodes60 and fails its finite-buffer schedule at500ns copy. Positive discharge splits retain total array service while moving first availability and discharge; they remain phase sensitivities.

Five deliberately corrupted traces were detected: ECC overlap, copy before availability, changed byte count, false feasibility flag and wrong row owner. Target source/report/trace hashes are checked before and after.

Reproduce with `python /home/ericyu/expert_flash_experiment/scripts/research_20261102_cr_ipr_pipeline_audit.py`. Frozen source SHA256: `ad35d08e17b4458cb5a0f54732aafce338d0526f5888fb02fcfdcc507ad99905`.

Timing is a conditional capability/accounting study, not a measured hardware schedule.
28us cold = tPRE+tEVAL+tDISCH in the paper; treating 28us as first availability is an accounting assumption.
First availability and final discharge phase split is not publicly fixed by these aggregate timings.
With a nonzero cold_discharge_ns, issue_stall_ns includes mandatory inter-run discharge gaps; adding it directly to array_busy_ns would count those gaps twice.
Skip-WL CR and IPR are user premises. PB source permutation and read-to-cache path remain conditional.
Host output metadata/scales and full-model quality are outside this timing audit.
