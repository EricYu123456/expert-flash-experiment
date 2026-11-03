# Full-wave chain and explicit cold-burst follow-up

The sealed v1 simulator and its reports are unchanged. This follow-up independently reconstructs finite-latch copy/ECC timing and checks1..64full waves at seven copy latencies and three source-order sequences (1344cases).

For N>=3, a necessary condition is **2c+(N−2)×540ns<=6040ns**, plus initial c<=6040ns. It follows from the last prior-wave ECC plane, irrespective of source permutation. Fixed0123 attains the bound here; other orders can add bubbles. This is a model-specific whole-page cache-only limit.

| Copy ns | Maximum fixed-order full-wave chain | Alternating reverse | Random order |
|---:|---:|---:|---:|
| 0 | 13 | 13 | 13 |
| 100 | 12 | 11 | 12 |
| 320 | 12 | 8 | 11 |
| 321 | 11 | 8 | 11 |
| 500 | 11 | 7 | 10 |
| 1000 | 9 | 5 | 7 |
| 2000 | 5 | 3 | 4 |

N12 allows at most320ns copy; N6 allows1940ns. N44 necessarily fails this finite-bank model even at zero copy. This does not show the published hardware is wrong: it shows why raw9.7us array timing plus a separate6.4GB/s whole-page ECC cannot be assumed to pipeline arbitrarily long with two latches.

An explicit cold-voltage restart is a separate conditional control. It waits for the read latches and pays28us, never extending a9.7us CR through an idle interval. Every possible cold-cut subset for G4’s12waves was enumerated; full-row-safe cut indices are3,6,9.

| Copy ns | Reactive cold reset us | Best arbitrary cuts us | Best full-row cuts us | Fixed6+6 us |
|---:|---:|---:|---:|---:|---:|
| 0 | 150.880 | 150.880 | 150.880 | 168.640 |
| 100 | 150.980 | 150.980 | 150.980 | 168.940 |
| 320 | 151.200 | 151.200 | 151.200 | 169.600 |
| 321 | 169.603 | 169.282 | 169.603 | 169.603 |
| 500 | 170.140 | 169.640 | 170.140 | 170.140 |
| 1000 | 171.640 | 171.640 | 171.640 | 171.640 |
| 2000 | 196.400 | 196.400 | 218.160 | FAIL |

At500ns, reactive reset is170.140us. A full-row cut at3/6/9ties this timing; fixed6+6 is feasible. A one-wave early cut saves500ns (169.640us) if the existing partial-row GEMV state survives the cold restart; that extra capability is explicit. At2000ns, full-row-only cuts need four3wave bursts and are slower than reactive cuts.

These are conditional service envelopes ending at weight ECC completion. Source-switch overhead, MAC throughput, output buffering and the existence of an in-command cold restart trigger are not certified.
