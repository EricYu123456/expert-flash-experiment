# Coactivation-aware static placement with held-out prompts

`scripts/optimize_placement.py` fits a fixed expert-to-placement-slot permutation
for every layer and chips-per-expert value g=1/2/4/8/16. The assignment keeps the
same capacity as modulo placement: every physical slot is occupied exactly once,
and every logical expert retains its original weights and identity. Frequently
co-selected experts can consequently be separated across chip groups.

The search starts from the original modulo layout and considers 100 deterministic
random expert-slot swaps per layer/g. A swap is accepted if it improves the training
sum of maximum selected experts per group; ties prefer a lower sum of squared
group counts. For homogeneous experts this is the exact flash-service objective
up to a per-layer coefficient obtained from `aif_model.phase_demand`. The secondary
objective only resolves equal critical-path cost. The search is a local heuristic,
not a global optimum.

Each of three folds trains on two entire prompt traces and holds out the third
prompt. Swaps and selection of g use only the training prompts. Held-out IDs are
used only for evaluation. Both the fixed global g and optional per-layer g are
selected again inside each training fold. This avoids choosing the group size on
the held-out workload. All reported times below are **flash-only**, with one cold
ordinary-read-equivalent per block run. Command/vector/host costs are excluded and
must be evaluated separately before selecting an overhead-aware deployment policy.

The means below aggregate the three held-out prompts. Each policy selects its
global g using training data only.

| Model | Weight layout | Modulo g | Modulo ms/token | Learned g | Learned ms/token | Held-out latency reduction |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Granite | Separate gate/up/down | 4 | 9.122 | 4 | 8.019 | 12.09% |
| Granite | Physically stacked gate+up, then down | 4 | 7.646 | 4 | 6.722 | 12.09% |
| OLMoE | Separate gate/up/down | 8 | 13.943 | 8 | 13.310 | 4.54% |
| OLMoE | Physically stacked gate+up, then down | 16 | 12.134 | 8 | 11.888 | 2.03% |

Each fold made the same global g choice within each row. At the same g=8, OLMoE's
stacked layout improves from 12.454 to 11.888 ms/token (4.54%); the stronger modulo
control chooses g=16, leaving the smaller 2.03% improvement shown above. Report
the stronger control when describing the total benefit.

Training gains exceed held-out gains: Granite improves 19.62% in training versus
12.09% held out; OLMoE separate improves 12.13% versus 4.54%. This gap cautions
against fitting and evaluating placement on the same routing records.

Per-layer adaptive g was also tested with the same training-only selection.
Granite's separate learned adaptive policy reaches 8.296 ms/token, worse than
the learned fixed-g policy's 8.019. Its stacked learned adaptive policy is 6.726,
essentially the fixed-g result. OLMoE separate is 13.287 versus 13.310, while its
stacked policy is unchanged. Adaptive g therefore does not consistently improve
these held-out workloads and should not be presented as the preferred method.

All outputs are on NFS:

- [Granite results directory](/home/nfs_client/ericyu/expert_flash_experiment/runs/granite-aif/learned_placement): `cross_validation.csv`, `optimization.csv`, `mappings.json`, `summary.json`, and `manifest.json`.
- [OLMoE results directory](/home/nfs_client/ericyu/expert_flash_experiment/runs/olmoe-aif/learned_placement): the same outputs.

Each result CSV includes every fixed g for modulo and learned placement, both
training and held-out costs, both weight layouts, global/per-layer training-selected
g controls, p50/p95 token latency, and useful bandwidth efficiency. Manifests
record input hashes, exact seeds, optimization budget, algorithm, and limitations.
The complete sweep took under one second per model on this host.

For the separate overhead model, read `mappings.json` as follows:

```
fold = folds[...]
slots = fold["placements"]["g4"][str(layer)]
physical_group = slots[logical_expert_id] % (16 // 4)
```

`layout_choices["separate" or "stacked"]` contains the training-selected global
and per-layer g for both controls. `flash_coefficient_ns[layout][layer][g]` gives
the source flash coefficient. A model including command or vector overhead should
reselect g on the training prompts under that model, then freeze choices for the
held-out prompt. The mappings were learned using flash cost only; do not call
them optimal for another cost model.

Checks confirmed that every mapping is a permutation, training/held-out prompt
sets are disjoint, useful bytes are conserved across policies, and g=16 is unchanged
by placement learning. All twelve final held-out fixed-g policies (two models,
three folds, two layouts) match full `aif_model.evaluate` exactly, including useful
bytes and total service nanoseconds. Original routing files were not edited.

These three prompts provide limited cross-validation evidence. Folds share training
data, and a deployment distribution can change expert coactivation. The result
shows a plausible faster static placement on the measured workloads; it does not
establish a universal best partition or measured physical PE utilization.
