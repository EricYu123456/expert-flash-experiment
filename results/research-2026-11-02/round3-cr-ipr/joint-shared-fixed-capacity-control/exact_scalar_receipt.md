# Exact independent scalar replay

PASS: all 48 full TRAIN traces, 173,520 node start/end pairs and all 48 makespans exactly match the saved C++ execution. The unchanged independent `scalar_execute` reconstructs the resource calendars; its `semantic_audit` also passes for every trace, including whole-vector barriers, chip ownership, byte conservation, and continuous output FIFO occupancy.

Protocol SHA-256 `091c1d90a30c46a7a40be3a4994d1805ac2f6c8f74a0d05211507a89f537e119` was saved before replay. The source, control, reports, and all traces retain their frozen hashes afterward. Oracle source SHA-256: `6b3d2cd4b167a8e38b4e527d3d1f9561bb334601628c2edd6d9560a3f510691f`. Physical selection/address checks are in `saved_trace_checks.json`; this receipt adds exact independent calendar replay and does not claim a new finite-latch simulation. No new routes or prospective inputs/outcomes were read.
