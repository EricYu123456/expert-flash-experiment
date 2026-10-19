# Fixed placement permutation robustness

The original expert-ID modulo mapping was compared with eight deterministic
permutations (seeds 0–7) of expert IDs onto fixed placement slots. Every run uses
the same three real llama.cpp decode traces, logical selected expert sets, original
tensor dimensions, ready scheduling inside each projection phase, and one cold
ordinary-read-equivalent per block run. The permutation is fixed across layers;
it changes where expert weights reside, not which experts the model chooses.

All raw outputs are on NFS:

- Granite: [CSV](/home/nfs_client/ericyu/expert_flash_experiment/runs/granite-aif/placement_permutations.csv), [manifest](/home/nfs_client/ericyu/expert_flash_experiment/runs/granite-aif/placement_permutations.manifest.json).
- OLMoE: [CSV](/home/nfs_client/ericyu/expert_flash_experiment/runs/olmoe-aif/placement_permutations.csv), [manifest](/home/nfs_client/ericyu/expert_flash_experiment/runs/olmoe-aif/placement_permutations.manifest.json).

The following ranges are over the eight placements. Latencies are modeled FFN
service milliseconds per decoded token, including all MoE layers; bandwidth
efficiency is useful selected INT8 bytes divided by 102.4 GB/s times service time.

| Model | Chips per expert | Minimum ms/token | Mean ms/token | Maximum ms/token | Useful bandwidth efficiency range |
| --- | ---: | ---: | ---: | ---: | ---: |
| Granite | 1 | 11.307 | 11.631 | 11.778 | 25.04–26.08% |
| Granite | 2 | 9.354 | 9.499 | 9.734 | 30.30–31.53% |
| Granite | 4 | 8.848 | 8.997 | 9.171 | 32.16–33.33% |
| Granite | 8 | 9.786 | 9.965 | 10.161 | 29.03–30.14% |
| Granite | 16 | 16.128 | 16.128 | 16.128 | 18.29% |
| OLMoE | 1 | 29.223 | 29.714 | 30.230 | 26.01–26.91% |
| OLMoE | 2 | 19.958 | 20.595 | 21.031 | 37.39–39.40% |
| OLMoE | 4 | 15.440 | 15.924 | 16.282 | 48.30–50.94% |
| OLMoE | 8 | 13.499 | 13.797 | 14.073 | 55.88–58.26% |
| OLMoE | 16 | 14.477 | 14.477 | 14.477 | 54.32% |

The best group size stayed **4 chips/expert for Granite and 8 chips/expert for
OLMoE in all eight seeds**, matching the original-ID mapping. Relative to full
16-chip sharding, the best group gives 1.76–1.82x lower service time for Granite
and 1.029–1.072x for OLMoE. Thus the observed preferred group size is not explained
by one arbitrary modulo numbering of the experts in these samples. The OLMoE
benefit remains modest and is sensitive in magnitude to placement.

Full 16-chip sharding is invariant under expert placement, so its result was
computed once and reused. Each CSV contains 40 rows (8 seeds x 5 group sizes).
Manifests store exact permutations, source hashes, script/model hashes, parameters,
and the separate original-ID baseline path. Conservation checks verified equal
selected useful bytes and route counts across placements. The original input
trace files were not edited.

These are eight sampled placements, not confidence bounds over all permutations.
The script does not optimize placement, generate synthetic routes, or move actual
hardware weights. All timing and hardware-utilization limitations of the base AiF
model continue to apply.

Reproduce with:

```bash
python3 scripts/placement_sensitivity.py --run-manifest /home/nfs_client/ericyu/expert_flash_experiment/runs/granite-aif/manifest.json
python3 scripts/placement_sensitivity.py --run-manifest /home/nfs_client/ericyu/expert_flash_experiment/runs/olmoe-aif/manifest.json
```
