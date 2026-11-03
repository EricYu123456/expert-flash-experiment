# M32 fixed geometry: exact free-position work bound

This audit finds exact minimum **owned work**, 287,752 ns for input1 and 298,504 ns for input4, when each query may freely choose row positions. A schedule-independent first-command argument strengthens the corresponding native necessary bounds to **291,252 ns / 302,004 ns**. Neither bound establishes a native schedule or a fixed static placement attaining it. No native scheduler, new fit, model inference, weights or heldout was run.

## Frozen geometry and one correction

Every routed expert has 25 GU jobs (24×41 rows plus 1×40 rows, C=1536) and 12 Down jobs (128 rows, C=512), all one read wave. The coupled subfamily has X5 both-phase positions, O7 Down-only positions, and 20 GU-only positions. Six selected experts therefore contribute 222 read waves; frozen shared contributes 73, giving **295** per query. All input reload, command and emitted complete-row output costs remain charged.

The actual frozen stageB `granite4tiny.M32.minimum_depth.plan.npz` shared geometry has **12 three-wave chips, 17 two-wave chips and 3 one-wave chips**. It is not the initially proposed 9/23 distribution. For input1, shared owned work is 86,078 ns on 12 chips, 57,992 ns on 12, 57,188 ns on 5, and 29,102 ns on 3. Input4 gives 89,918 / 61,832 / 59,492 / 31,406 ns for those same groups.

Routed job costs are:

| Input copies | GU40/41 | Down128 | O replacement delta |
|---|---:|---:|---:|
| 1 | 29,102 ns | 28,762 ns | −340 ns |
| 4 | 31,406 ns | 29,530 ns | −1,876 ns |

Thus a chip with aggregate X count x and O count o has work `6G + S_c + D·x + (D−G)·o`, with `0≤x,o,x+o≤6`, `sum x=30`, `sum o=42`. It has `6+x+sharedwaves_c` read waves. Service was recomputed independently using integer transfer ceilings and compared with the frozen backend occupancy table and every charged job node in the original ReferenceGraph builder. The graph was inspected, not scheduled.

## Exact state DP and a constructive witness

For a trial T, each chip's allowed `(x,o)` states are enumerated. A DP indexed by accumulated X stores a bitset of reachable O totals. The recurrence only shifts those bitsets by the current O and advances X; totals are capped at 30 and 42. Binary search over the finite set of exact state costs returns T. A separate Python set-of-pairs DP agrees on **all** final reachable states at T and T−1. Both input profiles certify T feasible and T−1 infeasible.

The corrected hand witness is also feasible for both profiles:

- Seven three-wave chips use `(x,o)=(1,5)`; the remaining five use `(0,0)`.
- All 17 two-wave chips use x=1; the three one-wave chips use x=2.
- Add O=5 and O=2 on two two-wave chips, with all remaining O zero.

This gives X30/O42, every chip degree at most six, exactly seven 10-wave chips and 25 nine-wave chips. The heavy work equals the DP optimum. Detailed columns and the independent DP predecessor witness are saved in `report.json` and `arrays.npz`.

Every admissible column vector can be realized by six coupled expert masks. Split the 30 X incidences over five left vertices of degree six, and the 42 O incidences over seven. Allocate each chip's remaining `6−x−o` incidences to 20 dummy left vertices. The resulting 32×32 bipartite multigraph is 6-regular. Hall's condition follows from counting edges: `6|A|≤6|N(A)|`. Remove a perfect matching, retain regularity, and repeat six times. Each matching is one expert with exactly five X and seven O positions and no overlapping positions. This is the standard bipartite multigraph decomposition underlying [Kőnig's edge-colouring theorem](https://www.math.tau.ac.il/~krivelev/teaching/co22/LectureNotes.pdf), not a new theorem or selector algorithm.

## Schedule-independent command-rank bound

If every chip has at most 10 waves, 295 waves across 32 owners force at least `295−32×9=7` owners with 10 waves. Enumerating all actual shared states and allowed `(x,o)` gives minimum heavy owned work 287,752 / 298,504 ns. Their seven distinct first-acquire commands all use the same 250 ns global command resource. At least one starts no earlier than initial host completion plus `6×250=1,500 ns`.

The chip owns its context from command acquire through output release. Therefore its finish time is at least that first start plus all its owned work, including its own commands. Waiting and GU/privateDown dependencies only increase this. The bound for this case is initial host 2,000 + 1,500 + heavy work.

If any chip instead has at least 11 waves, its read service alone requires at least 308,000 ns after the initial host, giving at least 310,000 ns. Taking the smaller bound over these exhaustive cases gives **291,252 / 302,004 ns** for every scheduler of this fixed resource/owner model. The argument uses neither a fixed chip dispatch order nor a `chipID×250` term; it adds no selector time and assumes no extra plane contexts or PB reuse.

The pointwise floor also bounds mean and quantiles. Against the currently frozen CP input1 mean 299,342.484375 ns, remaining mean improvement is at most 2.702752% in this fixed geometry/model. This is an optimistic upper bound on further gain, not achieved improvement or a global statement about other geometries or shared placements.

## Full logical rows and one-route physical image

The six masks are assigned to the ordered q0 TRAIN expert IDs. The other 58 experts use the frozen coupled hint; shared remains unchanged. Each of the 65 operators' two projections is partitioned by increasing chip into original complete logical rows. Gate/up rows are scattered to their original positions before SwiGLU and complete-vector A8; Down uses each expert's original full private 512-vector (1024 for shared). Row scales retain the same logical IDs. There are no partial dots, altered input columns, copied weights, aliases or changes in quantization.

`arrays.npz` contains both the selected `(6,2,32)` row counts and complete `(65,2,32)` images. `physical_row_IDs.json.gz` explicitly lists every logical row, scale identity, full input columns, physical block/wordline range and charged service components for both input profiles. Static repacking is allowed; prior source addresses are not assumed unchanged.

Each full image includes 167,424 logical rows, 148.5 MiB useful coefficients, 9,764 programmed pages, 2,441 reserved waves and 2,424 jobs. It uses one block per chip (32 blocks, 384 MiB allocated); the actual maximum resident extent is 80 waves, below 192. A general safe bound for any map in this geometry is `64×2 + sharedmax3 = 131` waves per chip.

Routed GU short reads emit 42 complete rows, requiring 40 or 41; those extra complete rows and 1,024 trailing bytes are drained/dropped. Shared two-wave GU emits 85 complete rows, requiring 84, plus 512 trailing bytes; shared one-wave GU has its corresponding 41/42-row fringe. The full image's charged emitted rows are 169,115, including 1,691 discarded complete rows and 1,662,464 partial-row bytes. Down has exact whole-row page boundaries. These are symbolic row/metadata and service checks; they add no valid-R, stop-MAC, reset or native output-scale ABI proof beyond the existing ordinary-row candidate assumptions.

The image is feasible for **one chosen route**. Giving every query its own repack is a relaxation; a deployed static map must satisfy all queries with the same addresses. No runtime relocation method, deployable layout, native optimality, quality result, hardware measurement or new theorem is claimed. The phase-independent scope extension is separately frozen in `addendum-phase-independent/`.
