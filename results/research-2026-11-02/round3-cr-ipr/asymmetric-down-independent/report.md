All 12 saved asymmetric winner traces pass the independent scalar/DAG audit. Every start/end and makespan matches exactly across 44,340 nodes, 72,504 edges and 624 jobs. All traces fit 128 B in the declared continuous-drain output model; peak is 128 B.

Checks cover full GU arrival before private whole-vector activation and upload, chip ownership, resource exclusion, page-output causal timing, all seven Down outputs before combine, and byte conservation.

Representative saved traces only, not every TRAIN query.
Conditional service-model verification, not physical device measurements or model quality.
Uses the previously validated independent scalar implementation and semantics checker; it is not a new independent implementation.
Capacity certificate is recorded as an input; this script does not reconstruct weight bytes or physical addresses.
Programmable source order, pre-ECC inactive discard, PB overwrite/copy timing and actual FIFO size remain unverified capabilities.
