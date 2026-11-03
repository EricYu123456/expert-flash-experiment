# Finite PB / CR / IPR pipeline audit

This is a timing envelope under explicitly named capabilities, not a device feasibility certificate.

All 36 meaningful checks passed. The G4 resident construction occupies [128, 128, 70, 70] WLs per plane in one192-WL block.

The explicit model has a28us cold first wave,9.7us subsequent synchronized CR waves, a6.04us recycling phase, two weight latches per plane, and one6.4GB/s ECC. All four page latches are accounted for; two remain input latches. Four-page ECC service is10.24us. Pages are copied read→cache and held through full-page ECC completion; no weight FIFO or cache/read mux is supplied.

A plane must have been read in the immediately preceding wave to remain CR-eligible. If any selected plane was idle, the synchronous wave is cold28us. The default ordinary balanced15 case consequently has7cold/8CR waves. Keep-hot controls instead read all4planes, then either conditionally drop unselected pages before ECC or explicitly decode them before discarding.

For the sampled copy sweep, candidate maximum feasible cache-copy latency is Nonens at sense-start overwrite and 320ns at evaluation-start overwrite (copy/recycle overlap enabled). A null cap means no sampled point was feasible.

| Layout / default conditional capabilities | Waves | Functional | Finite timing | Makespan us |
|---|---:|---|---|---:|
| candidate12 | 12 | True | True | 150.88 |
| ordinary_sameWL_balanced15 | 15 | True | True | 284.92 |
| ordinary_IPR_balanced15 | 15 | True | True | 284.92 |
| ordinary_packed_adjacent12 | 12 | True | True | 150.88 |
| ordinary_packed_spaced15 | 15 | True | True | 285.46 |
| IPR_compensator12_functional_negative | 12 | False | True | FAIL |

The defaults assume programmable plane-source order, evaluation-start overwrite, zero copy latency and copy/recycle overlap. These are sensitivity settings, not established AiF controls. Readout with fixed0123 order fails the candidate complete-row stream check. A one-weight-slot control fails finite timing. Shared compensator packing is also functionally invalid because its two-page shared runs interrupt a row.

A physical ordinary packed layout can use12waves for adjacent co-selected experts and15waves for spaced ones. The15wave balanced matrix-aligned control is therefore not the only ordinary layout. No route-independent latency advantage over every ordinary packing is claimed.

Strict no-interval failures remain failures. The cold-reset control waits for latch release and pays another28us read. A paced9.7us continuation is separately labeled as conditional CR bias retention. Copies that cannot overlap recycling are evaluated separately.

At copy500ns, keep-hot balanced15 with before-ECC drop is feasible at175.620us, physically reads60pages, and decodes48pages. Decoding all60pages instead fails the finite-PB schedule. These capabilities are explicit controls, not granted to the candidate alone.

The exact default candidate cache-copy boundary is320ns: the minimum latch-deadline slack is640−2*copy ns. Both320ns pass and321ns fail are tested. Ordinary packed adjacent12 has the same boundary and same150.880us zero-copy cost.

Cold first availability28us and final discharge0 is a conservative service accounting convention. The paper gives28us=tPRE+tEVAL+tDISCH and does not disclose the numeric phase split. Optional cold_discharge_ns sensitivities move d from initial availability to final run discharge while retaining the same total array service. No measured first-page phase is claimed.

The independent byte/service and release-time bounds are necessary bounds only. The event schedule verifies latch deadlines, serial ECC, per-plane copy serialization, exact page coverage, startup and drain. Hypothetical times for failed schedules are diagnostic and must not be used as feasible costs.

The effective-bandwidth-only6.4GB/s cost is a separate model with unspecified startup. It is never added to a second9.7us/wave charge and does not certify this pipeline.

The public Python API returns page_completions, full-row completion ranges, per-owner completion timestamps, read_latch_releases, array_waves, finite-storage lifecycle intervals and event traces. Caller-supplied waves are accepted. See assumption_ledger.json and small_two_wave_trace.csv.
