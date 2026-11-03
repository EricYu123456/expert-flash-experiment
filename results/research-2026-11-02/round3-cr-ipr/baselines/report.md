# Common-input GU page-wave baselines

TRAIN only; all 40 G4 layers and 512 aligned decode inputs. These are sense-wave/page counters, not native completion latency.

| Control | Mean waves/layer | P99 | Range |
|---|---:|---:|---:|
| per_expert_runs | 15.000000 | 15.00 | 15–15 |
| sameWL_packed_skipCR_global_roundrobin | 14.764404 | 15.00 | 13–15 |
| IPR_global_roundrobin | 12.915967 | 15.00 | 12–15 |
| IPR_static_uniform_four_phase | 13.273633 | 15.00 | 12–15 |
| IPR_per_expert_phase_aligned_balanced_shared | 15.000000 | 15.00 | 15–15 |
| IPR_fixed_two_plane_masks | 14.747900 | 21.00 | 12–21 |
| IPR_fixed_one_plane_masks | 19.458691 | 33.00 | 15–39 |
| sameWL_joint_compensation_compact_fixed_layout | 22.909619 | 24.00 | 16–24 |
| IPR_joint_onecopy_shared_compensation | 12.000000 | 12.00 | 12–12 |
| IPR_generic_sparse_row_compiler_same_layout | 12.000000 | 12.00 | 12–12 |
| sameWL_packed_TRAIN_exact_mean_matching | 13.746680 | 15.00 | 12–15 |
| IPR_same_TRAIN_exact_mean_matching_layout | 12.674756 | 14.00 | 12–15 |
| IPR_TRAIN_learned_phase_coordinate_descent | 12.504150 | 14.00 | 12–15 |

All controls: one copy;396 resident pages/chip;48 selected pages/chip/query; no overfetch; one reserved four-plane192-WL block/chip.

The shared-compensated profile gives6×(2,2,1,1)+(0,0,6,6)=(12,12,12,12), meeting48/4=12 for every possible top6. Its resident counts(128,128,70,70) fit192WL. A generic sparse compiler granted identical primitives emits exactly this layout and ties it; this is elementary water filling, not a new layout algorithm.

The same-layout comparisons isolate same-WL mask from IPR; fixed one/two-plane masks are additional controls. The learned phase baseline is a small TRAIN-only coordinate descent, frozen before opening new heldout.

The strong packed same-WL control exactly minimizes TRAIN mean within contiguous6page global packing: waves=15−numberofselected-both adjacentpairs, so maximum-weight perfect matching on expert coactivations solves it. This is a standard matching baseline, not a new algorithm or native/tail optimum.

Sparse identity masks have10.098730mean extra plane cold restarts if an idle read epoch breaks CR. Conditional keep-hot reads everyplane, with11.057617mean inactive pages/chip/query; these may be dropped beforeECC only under an explicit conditional-discard capability. Full gap/overfetch counters are in`sparse_mask_and_keep_hot_ledger.json`.

Deep requires resident(392,392,334,334), hence3/3/2/2 plane blocks and3 reserved four-plane blocks.44 is only a global page-work bound; the explicit common-block grouping averages53.681115waves. It cannot be reported as44consecutive same-block CR reads.

Without a shared branch there is no shared compensation. When P is divisible by4 and each expert is balanced, per-expert runs already match aggregate IPR page-wave work. This G4 case is not a universal gain.

No native ROW/PB/CR timing, numerical lowering, down execution, hardware speedup, power or end-to-end FFN claim is made here.

Reproduce: `python3 scripts/research_20261102_cr_ipr_baselines.py --output-dir /tmp/cr-ipr-baselines-reproduction`.

Provenance and old strong plan API: `provenance_and_old_plan_api.json`. Fixed physical page ledgers: `physical_page_ledgers.json`. Frozen phase maps: `frozen_TRAIN_phase_maps.npz`. Frozen rules: `protocol.frozen.json`.
