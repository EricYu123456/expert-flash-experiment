The broader fixed-A8 placement work screen is sealed as a bounded negative result. Thirty-two routeblind layouts (h=2/4/6/8 high sources, eight seeds each) and four TRAIN-derived maximum-coactivation perfect-matching layouts were evaluated. Matching covers 685 co-selected pairs across TRAIN512; each paired initial A geometry is complementary and minimizes copied core rows for that complementary choice. The two best starts were matching layouts, followed by two coordinate passes and 8,192 source swaps each.

| Quantity | Mean (ms) | Empirical P99 (ms) |
|---|---:|---:|
| Frozen B native incumbent | 0.561099020 | 0.580695140 |
| Selected exact min maxchip work | 0.550669141 | 0.573622260 |
| Host + work necessary LB | 0.552669141 | 0.575622260 |
| Optimal first-start assignment LB | 0.553191277 | 0.575729380 |

The selected layout permits at most 1.502387% mean / 0.873587% P99 native improvement, even with zero selection overhead and no additional scheduling stalls. The stronger first-start bound reduces those ceilings to 1.409331% / 0.855141%. The other trained endpoint permits 1.386813% / 0.970757%. Both fail the 2% mean or 5% tail expansion gate. Same-layout static A/B and broader zero-copy B/free-A local-search controls retain the original all-B incumbent; additional static search is explicitly reported in work_report.json. No native/selector/full-copy/partial-copy/heldout campaign was run.

The independent checker reproduced 1,245,184 query-mask values across all 36 initial layouts and both trained endpoints, both final output mask tie choices, static final costs, original and copied row identities, prefix coverage and physical interval nonoverlap. Each expert retains exactly one A plus B at runtime. Nine/sixteen/all possible A geometries are never supplied free to the query.

This selected layout copies 28,015 complete rows = 43,031,040 useful coefficient bytes (41.037598 MiB) per layer. It needs 625 additional resident waves / 2500 pages = 39.062500 MiB programmed storage. Whole-layer L0 address allocation, including unchanged Down/shared, grows from 16 to 22 LSB blocks. Per-chip inventory is explicit in audit.json; the fit imposed no fixed-average chip-capacity budget. This variable budget cannot be called equal to v1's 512-wave budget. It could buy at most 27 minimum-work full-GU images under only a useful-byte bound or 26 under the page/wave bound, leaving one wave; no feasible whole-copy placement/control is claimed from those volume counts.

The unrestricted per-query free-span relaxation retains its 3.011906% mean / 6.164532% P99 possible ceilings. It removes fixed address and replica-capacity constraints and is not an implemented layout. The finite heuristic result here does not prove all offline A source placements fail, nor does prefix sharing itself add speed relative to storing identical A/B views separately. Complete-row reads, page-fringe drains/drop, full H input, single native context, unchanged shared/Down and full-GU A8 dependency remain the declared contract.
