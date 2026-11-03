# Deterministic6×6 matching: host CPU microbenchmark

Fresh synthetic top6sets: 4096. Eachsidecapacity1 or2 uses the same standard deterministic augmenting-path order as the frozen Python helper. All12,291 semantic outputs match, including exact assignments/failed partial results. The separate deliberate Hall failure route is40,41,42,43,44,45.

CPU: AMD EPYC 9354 32-Core Processor; pinned logical CPU127. Compiler: g++ (Ubuntu 11.4.0-1ubuntu1~22.04.3) 11.4.0, -O3 C++20. Threeprocessrepetitions,31samples/profile/mode/process,16,384calls/sample. No CPUfrequency orpriority changes.

| Synthetic profile | Mode | median ns/call | P95 ns/call | P99 ns/call |
|---|---|---:|---:|---:|
| cap1_failure | cap1 | 87.72 | 88.12 | 88.41 |
| cap1_failure | cap1_then_cap2 | 153.75 | 154.91 | 155.65 |
| cap1_failure | cap2 | 62.39 | 62.93 | 63.20 |
| cap1_failure | harness_control | 6.59 | 6.62 | 6.66 |
| cap1_success | cap1 | 96.53 | 96.99 | 97.76 |
| cap1_success | cap1_then_cap2 | 99.03 | 99.56 | 100.42 |
| cap1_success | cap2 | 67.06 | 67.64 | 67.79 |
| cap1_success | harness_control | 6.61 | 6.84 | 7.01 |
| deliberate_Hall_failure | cap1 | 98.59 | 99.06 | 99.29 |
| deliberate_Hall_failure | cap1_then_cap2 | 171.32 | 172.27 | 172.70 |
| deliberate_Hall_failure | cap2 | 73.30 | 74.65 | 75.16 |
| deliberate_Hall_failure | harness_control | 6.58 | 6.83 | 6.97 |
| mixed_fresh_synthetic | cap1 | 96.99 | 97.72 | 98.80 |
| mixed_fresh_synthetic | cap1_then_cap2 | 99.67 | 100.43 | 101.95 |
| mixed_fresh_synthetic | cap2 | 67.34 | 67.78 | 68.49 |
| mixed_fresh_synthetic | harness_control | 6.59 | 6.80 | 6.86 |

These percentiles describe repeated batch-average ns/call, not individual-call P99. Timed intervals include the direct no-inline call, route cursor, compiler fences and8-byte output stores; inputmask preparation, classification, file I/O and checksum hashing are outside timing. The harness_control floor is reported separately and is not subtracted.

All24,379,392 timedcalloutputs feed observable checksums; globalchecksumeb2f5932772089c7 and eachprofile/mode/samplechecksum are identical acrossallthreeprocesses. Disassembly confirms matching/fallback/control calls remain; no-inline functions and compiler fences prevent loop elimination/hoisting.

Cap2 is ordinary matching to12capacityslots acrosssixgroups, not extra PE inputcontexts. cap1_then_cap2 first attempts exactcapacity1 and falls back fromscratch onlywhen needed. Profileclassifications are arithmetic on syntheticcanonical supports, not estimates of TRAINorprospective route behavior.

No TRAIN, prospective prompts/routes/outcomes, weights or frozen main drivers are read or changed. This is a host CPU software microbenchmark, not firmware, controller, flash or end-to-end scheduling measurement.
