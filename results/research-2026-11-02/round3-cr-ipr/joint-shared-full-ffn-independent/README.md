# Independent saved actual trace audit

PASS: 128 saved traces; every node start/end and makespan exactly replays the previously independent scalar executor. All128B continuous-drain FIFO, resource/chip leases, fullGU→A8→privateDown→combine dependencies and exact18,944outputrows pass.

Trace inventory: {'native': 12, 'prospective': 60, 'joint_TRAIN': 32, 'joint_diagnostic': 24}. Prospective includes48 first plus12 fallback traces; all cap2 jobs finish with full rows. Native v2 saved12traces because cache merged q0 graphs; no missing representatives were regenerated. Summary arrays are arithmetic/route-binding checks only, not full event replay.

All544joint resident jobs independently reconstruct exact jobID/group/chip/block/WL/row/C. Selected sharedS0/S1 provide1536rows total with completeC1024; each joint chip runs exactlyoneDown. Native g8 GU binds the learned physical8chipgroups, fixed addresses and128row/chip scatter; all selected waves are whole0123/sameWL, with no dummy/drop/shape. Ordinary is entirely cold.

Nine negative controls reject sharedhalf row/C, actualchip, nativeblock including changed recipe digest with unchanged metadata, ordinaryarray_mode, drop, extra physicalwaves, and wrongGUgroup.

This is a new binding/mutation driver reusing the existing independent scalar/semantic/FIFO/criticalpath checkers. No new independent executor and no whole campaign rerun. Kernel digests bind already verified recipes; page-level deadlines are not re-proved. Conditional service model and existing source/drop/phase/MAC assumptions remain. Standard CBC/group replication; no newtheorem/hardware/quality/globaloptimality claim. Joint diagnostic uses already observed prompts after design, not fresh validation.

Reproduce: `python scripts/research_20261102_joint_shared_full_ffn_audit.py` in a new/unsealed output directory. A sealed manifest intentionally prevents rerun in place. See report.json for pertrace detail and critical_paths.json for exact resource-augmented chains.
