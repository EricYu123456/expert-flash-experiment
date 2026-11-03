Independent phase-replica construction passes all 74,974,368 top6 combinations by16 count classes and a capacitated Hall proof;924 labeled reduced-universe routes are explicitly checked. At least58extra fullcopies are necessary within the fixed two-phase/3experts-per-phase dense6page model. This is a known two-server capacity3 CBC special case.

Real-checkpoint W8 bytes and existing per-row scales were packed into unique phase-copy addresses and read back byte-exact. 79 cases, both12wave/no-dummy and13wave/whole-dummy streams, all16chips: 1,294,336 exact I64 and F32-bit comparisons. Every122available expert/phase copy is numerically exercised. Independent NumPy agrees. No changed quantization; inputs remain the earlier explicit synthetic fixtures, so this is not task-quality evidence.

GU uses186WL/plane and186MiB useful coefficients (+87MiB). SharedDown g4 fits block0 WL186..191; routedDown triples fill block1 on12chips. Total331.5MiB useful/28nonempty chipblocks fits common32blocks. Full input is1536B; output scatter maps back to original route slots and canonical row scales. Only one live row accumulator is used; dummy reads do not advance it or enter ECC.

Independent finite-buffer recurrence:12waves copy100 150980ns feasible=True;12waves copy500 feasible=False; whole-dummy after10usefulwaves copy500 155680ns feasible=True. This full-wave spacer is distinct from the earlier3+1dummy split policy.

Actual samebytes standard control: two8chipgroups, each61wholeGU experts×12pages plus shared12pages,186WL/plane and the same+87MiB. Localchip128row shards reconstruct each full1024row expert; scales and output scatter are exact. Its 1,294,336 I64/F32-bit comparisons pass and complete output files are bit-identical to the phase variant. All12readwaves are sameWL and fixed0123, so this control needs noIPR. Both have150980ns copy100/no-drop or155680ns copy500/whole-dummy modeled timing. This removes phase-specific gain/novelty for the tested witness; ordinaryCBC/group replication is standard.

Frozen builder bridge passes744phase and1488ordinary-group resident page coordinates and all11saved traces. Nine nonempty traces reuse the earlier independent page checker; two new whole-dummy g8 traces match this file's independent recurrence at every copy/ECC timestamp, read/cache lifetime and two-slot count. Plain500 is correctly infeasible. The ordinaryg8 saved split500 trace requiresIPR; it is distinct from the primary sameWL whole-dummy control.

Known two-server cap3 CBC/count special case, not new theorem or solver.
58copy minimum is only under two full dense6page phase choices with everytop6 split3/3; not arbitrary placement optimum.
Weights/scales are existing reconstructed real-checkpoint W8 per-row tensors; fixture inputs are explicit synthetic vectors, not captured real activations. No requantization, task quality, nativeABI or GGUF equivalence claim.
All16chips execute one complete common1536B A8 input and one scalar row accumulator through page boundaries. Dummy drop preserves the live partial row/input index and produces no ECC or output.
Software usesI64 as an exact reference; the conservative1536*128*128=25165824 sum bound fitsI32, so this audit assumes no wider native accumulator or additional dot context.
Physical replicas add87MiB GU coefficients perlayer. Total331.5MiB includes standard tripleDown plus sharedg4, within common2blocks/chip; scale/scatter host metadata assumed resident.
IPR sameblock independent page addressing remains a premise. Whole-dummy preECC dropping remains conditional; copy100 uses12waves and no dummy.
Timing is a separate modeled recurrence, not measured hardware. Cold28us firstavailability and implicit6.4GB/s PE/MAC sustain remain assumptions; firstphase/discharge split unknown.
The same87MiB ordinaryg8 wholeGU group-replica control has the same CBC availability guarantee and identical scattered outputs/timing, while using sameWL reads with noIPR. Thus this witness supports no phase-specific performance or novelty advantage over that standard control.
This script builds its own layouts and streams. The saved builder bridge reuses the prior independent page checker; no target simulator or builder is imported.

Reproduce: `python /home/ericyu/expert_flash_experiment/scripts/research_20261102_gu_phase_replica_audit.py`
