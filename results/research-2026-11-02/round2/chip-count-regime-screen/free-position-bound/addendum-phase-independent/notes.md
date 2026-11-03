# Scope extension to independent GU/Down positions

The original sealed source and report are unchanged. This independently frozen addendum removes the per-expert X5/O7 minimum-overlap restriction from the **necessary bound**. It retains exactly 25 positive GU bands (24×41 rows and 1×40, C1536) and 12 positive Down bands (128 rows, C512) per routed expert, K6, the same 32 chips, original shared layout and Service.

For each chip independently enumerate `g,d∈{0,…,6}`, the selected experts' active GU and Down job counts. Its owned work is `G·g + D·d + S_c`, and it is heavy when `g+d+sharedwaves_c≥10`. No condition is imposed on GU/Down mask overlap, or on their union covering all chips.

Both input profiles enumerate **1,568 states** (32×49), including **537 heavy states**. The minimum heavy owned work is again **287,752 ns / 298,504 ns**, attained on the 12 three-shared-wave chips with `g=1,d=6`.

There is also a direct reason the old heavy-state minimum remains valid. Shared contributes at most three waves, so a heavy chip has `g+d≥7`. Set `x=g+d−6` and `o=6−g`. Then `x,o≥0`, `x+o=d≤6`, and

`G·g + D·d + S_c = 6G + D·x + (D−G)·o + S_c`.

Every independent heavy state therefore appears in the previous X/O enumeration. States with `g+d<6` need not be represented there, but none can be heavy. The fresh enumeration confirms the identity for every heavy state rather than assuming the prior subfamily is complete.

Every query still reads `6×25 + 6×12 + 73 = 295` waves. If all chips have at most ten, at least seven have ten; their first acquisitions share the global 250 ns command resource. At least one starts at or after initial host completion plus 1,500 ns, and its serialized chip work is at least the heavy minimum. If any chip has eleven or more waves, read service plus initial host is at least 310,000 ns. The resulting unconditional schedule-independent necessary bounds remain **291,252 ns / 302,004 ns**.

The coupled six-expert construction is a feasible subfamily of independent masks and attains owned work 287,752 / 298,504 ns. Therefore the broader **free-position, per-query owned-work optimum** is the same. This does not supply one fixed static map attaining it for all queries, nor a native schedule meeting the strengthened command-rank bound.

The bound applies to the upcoming phase-independent cyclic controls when they retain those row multisets, all command/input/output costs, original shared and one chip owner. It does not extend to other geometry, shared remapping, different row page/fringe costs, additional contexts, preempted commands, free page-buffer reuse or another hardware/ABI model. No native execution, placement fit, weights, inference or heldout was used.
