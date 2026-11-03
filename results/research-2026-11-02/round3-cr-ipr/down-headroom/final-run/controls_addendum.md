# Partition controls addendum

GF and DF cost/FIFO arrays are exactly identical for all 24 fullTRAIN profiles. Therefore the reported partition gain also holds at the same scheduling priority.

| GU | TRAIN64 learned all64 mean µs | 63+1 one-block mean µs | best static all64 mean µs | static vs incumbent |
|---|---:|---:|---:|---:|
| candidate | 320.394 | 320.416 | 351.908 | -3.410% |
| RR | 321.800 | 321.821 | 353.313 | -3.396% |
| g8_fixed | 327.824 | 327.839 | 356.300 | -1.718% |

Static grouping is ID-sorted; the same rare expert was selected using TRAIN. The table describes only the original 64-query screen, not an independent validation or a later fullTRAIN selection. Learned label fitting matters in this workload; the shared/private separation alone does not guarantee an improvement.

The common budget is two blocks per chip (32 blocks/layer). The selected all64 layout occupies 20 nonempty blocks/layer; the original and 63+1 learned maps occupy 16. Four entire private jobs use block1, one on each chip of the 22-expert routed group. Each still has a single command and cold start. The all64 control slightly improves the 63+1 timing while removing its rare-expert-on-shared-group restriction.

The frozen-map timing sensitivity is in ../timing-sensitivity/report.md. It compares mandatory Down work only, and does not establish native fullFFN gains under those alternative timing models.
