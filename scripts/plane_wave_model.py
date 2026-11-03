#!/usr/bin/env python3
"""Fixed NAND plane placement and conditional cross-expert scheduling.

This module places *whole output-row shards* supplied by its caller. Pages are
only storage/transfer units: a page need not contain a complete matrix row.
``baseline`` uses all four planes and serializes streams; ``pair`` uses fixed
disjoint plane pairs; ``single`` uses four fixed one-plane lanes. Placement is
built without a selected route and cannot change during evaluation.

The baseline-compatible ``service_ns`` is an ideal-overlap lower bound, not a
cycle-accurate hardware execution claim. A second, explicitly constructible
``conservative_service_ns`` serializes ordinary (cold) NAND reads and shared
ECC/PE draining at every synchronized epoch. Each epoch deliberately pays the
ordinary-read cost because the intervening drain breaks the paper's no-gap
cr-read condition. The optional timeline describes that latter schedule.
Pair/single policies require independent addressing, context ownership, and
the ability to issue reads on a subset of planes. The AiF paper does not
establish these capabilities. Same-WL static bundles are a different layout.
"""
from collections import defaultdict
from dataclasses import dataclass, field
from math import ceil
from typing import Mapping

PAGE = 16_384
PLANES = 4
BLOCK_PAGES = 192
COLD_NS = 28_000
CR_NS = 9_700
CHIP_BW = 6_400_000_000
POLICY_WIDTH = {"baseline": 4, "pair": 2, "single": 1}


def ceildiv(a, b):
    return (a + b - 1) // b


def transfer_ns(nbytes, bandwidth=CHIP_BW):
    if bandwidth <= 0:
        raise ValueError("ECC/PE bandwidth must be positive")
    if bandwidth == CHIP_BW:
        # Exact integer arithmetic, matching aif_model.Demand.service_ns.
        return ceildiv(nbytes * 5, 32)
    if isinstance(bandwidth, int):
        return ceildiv(nbytes * 1_000_000_000, bandwidth)
    return ceil(nbytes * 1_000_000_000 / bandwidth)


def stream_cost(nbytes, width=4, block_pages=BLOCK_PAGES):
    """Closed form used by vectorized trace evaluators for a single stream."""
    if nbytes < 0 or width not in (1, 2, 4) or block_pages <= 0:
        raise ValueError("Invalid stream geometry")
    pages = ceildiv(nbytes, PAGE)
    waves = ceildiv(pages, width)
    cold = ceildiv(waves, block_pages)
    return dict(useful_bytes=nbytes, physical_pages=pages, stream_waves=waves,
                cold_reads=cold, nand_ns=waves * CR_NS + cold * (COLD_NS - CR_NS),
                tail_slots=waves * width - pages)


@dataclass(frozen=True)
class Segment:
    block: int
    wordline: int
    waves: int
    page_offset: int
    pages: int


@dataclass(frozen=True)
class Shard:
    expert: int
    nbytes: int
    mask: tuple
    pages: int
    waves: int
    segments: tuple

    @property
    def nand_ns(self):
        return self.waves * CR_NS + len(self.segments) * (COLD_NS - CR_NS)

    def page_address(self, page_index):
        """(plane, block, WL), preserving the shard's row-major byte order."""
        if not 0 <= page_index < self.pages:
            raise IndexError(page_index)
        for segment in self.segments:
            local = page_index - segment.page_offset
            if 0 <= local < segment.pages:
                return (self.mask[local % len(self.mask)], segment.block,
                        segment.wordline + local // len(self.mask))
        raise AssertionError("Page missing from segments")


@dataclass
class Layout:
    policy: str
    block_pages: int
    shards: dict
    lane_cursors: dict
    boundary_padding: dict
    _summary: dict = field(default_factory=dict, repr=False)

    def summary(self):
        if self._summary:
            return dict(self._summary)
        useful = sum(s.nbytes for s in self.shards.values())
        pages = sum(s.pages for s in self.shards.values())
        slots = sum(s.waves * len(s.mask) for s in self.shards.values())
        reserved = sum(ceildiv(cursor, self.block_pages) * self.block_pages * len(mask)
                       for mask, cursor in self.lane_cursors.items())
        boundary = sum(self.boundary_padding.get(m, 0) * len(m) for m in self.lane_cursors)
        final = sum((ceildiv(cursor, self.block_pages) * self.block_pages - cursor) * len(mask)
                    for mask, cursor in self.lane_cursors.items())
        assert reserved == pages + (slots - pages) + boundary + final
        self._summary = dict(policy=self.policy, streams=len(self.shards), useful_bytes=useful,
                             physical_pages=pages, physical_page_bytes=pages * PAGE,
                             intra_page_padding_bytes=pages * PAGE - useful,
                             allocated_plane_slots=slots, stream_tail_slots=slots - pages,
                             boundary_padding_slots=boundary, final_block_tail_slots=final,
                             reserved_plane_slots=reserved, reserved_lsb_bytes=reserved * PAGE,
                             reserved_tlc_raw_bytes=reserved * PAGE * 3,
                             lsb_byte_utilization=useful / (reserved * PAGE) if reserved else 0,
                             page_coverage_once=True, fixed_without_test_routes=True,
                             addressing="same WL within each stream mask; flexible address across lanes")
        return dict(self._summary)

    def validate(self):
        """Enumerate addresses for audit fixtures, without assuming pages are rows."""
        addresses = set()
        for expert, shard in self.shards.items():
            assert expert == shard.expert
            assert sum(s.pages for s in shard.segments) == shard.pages
            assert sum(s.waves for s in shard.segments) == shard.waves
            assert all(s.wordline + s.waves <= self.block_pages for s in shard.segments)
            assert len(shard.segments) == ceildiv(shard.waves, self.block_pages)
            for page in range(shard.pages):
                address = shard.page_address(page)
                assert address not in addresses, (expert, page, address)
                addresses.add(address)
        assert len(addresses) == self.summary()["physical_pages"]
        return True


def build_layout(expert_bytes: Mapping[int, int], *, policy="baseline", masks=None,
                 block_pages=BLOCK_PAGES):
    """Place every expert shard once, independently of selected expert IDs.

    Default lane selection is expert ID modulo number of lanes. Callers may
    supply fixed masks learned using *training routes only*. Masks must have
    the policy's width and be identical or disjoint; overlapping masks are not
    silently treated as independent resources. Short streams are block
    contained. Long streams begin at a block boundary, retaining the baseline
    cold-count convention. Different lanes can have different block/WLs.
    """
    if policy not in POLICY_WIDTH or block_pages <= 0:
        raise ValueError("Expected baseline/pair/single and positive block_pages")
    width = POLICY_WIDTH[policy]
    shards, cursors, padding = {}, defaultdict(int), defaultdict(int)
    lane_masks = set()
    for expert, nbytes in sorted(expert_bytes.items()):
        if not isinstance(expert, int) or expert < 0 or not isinstance(nbytes, int) or nbytes < 0:
            raise ValueError("Expert IDs and byte counts must be nonnegative integers")
        if nbytes == 0:
            continue
        lane = expert % (PLANES // width)
        mask = tuple(sorted(masks[expert])) if masks is not None else tuple(range(lane * width, (lane + 1) * width))
        if len(mask) != width or len(set(mask)) != width or any(p not in range(PLANES) for p in mask):
            raise ValueError("Invalid plane mask for policy width")
        if any(set(mask) & set(other) and mask != other for other in lane_masks):
            raise ValueError("Independent lane masks must be identical or disjoint")
        lane_masks.add(mask)
        cost = stream_cost(nbytes, width, block_pages)
        cursor = cursors[mask]
        offset = cursor % block_pages
        if offset and (cost["stream_waves"] > block_pages or cost["stream_waves"] > block_pages - offset):
            pad = block_pages - offset
            cursor += pad
            padding[mask] += pad
        segments, left, page_offset = [], cost["stream_waves"], 0
        while left:
            block, wl = divmod(cursor, block_pages)
            waves = min(left, block_pages - wl)
            pages = min(cost["physical_pages"] - page_offset, waves * width)
            segments.append(Segment(block, wl, waves, page_offset, pages))
            cursor += waves
            page_offset += pages
            left -= waves
        cursors[mask] = cursor
        shards[expert] = Shard(expert, nbytes, mask, cost["physical_pages"], cost["stream_waves"], tuple(segments))
    return Layout(policy, block_pages, shards, dict(cursors), dict(padding))


def _stream_waves(shard):
    for segment in shard.segments:
        for wave in range(segment.waves):
            pages = min(len(shard.mask), segment.pages - wave * len(shard.mask))
            yield dict(expert=shard.expert, block=segment.block, wordline=segment.wordline + wave,
                       page_offset=segment.page_offset + wave * len(shard.mask),
                       mask=shard.mask[:pages], cold=wave == 0,
                       duration=COLD_NS if wave == 0 else CR_NS)


def evaluate_chip(layout, selected_experts, *, common_input=True, ecc_bw=CHIP_BW,
                  contexts=None, include_timeline=False):
    """Evaluate a fixed layout; all requested experts are ready in this phase.

    ``service_ns`` takes max(NAND lane loads, shared ECC/PE load) in each context
    batch. It is a resource lower bound under the existing ideal pipeline
    convention. ``conservative_service_ns`` is a schedule with all lanes
    synchronized: one ordinary NAND wave, sequential ECC/PE drain, then the
    next wave. The drain introduces an interval, so every NAND read pays 28 us.
    ``retained_cr_schedule_ns`` uses 9.7-us sequential reads only under an
    additional, unverified ability to retain recycling state across that gap.
    All exclude host dispatch, vector transfer, activation, or routing.

    A context remains assigned to one expert until that expert's last wave.
    No cr-read state survives an expert/block change, even with shared input.
    Timeline entries cover NAND intervals and shared-drain intervals; a plane
    cannot start its next NAND read before its previous drain has completed.
    """
    selected = list(selected_experts)
    if len(set(selected)) != len(selected):
        raise ValueError("Duplicate selected expert")
    unknown = set(selected) - set(layout.shards)
    if unknown:
        raise ValueError(f"Selected expert has no nonempty shard: {sorted(unknown)}")
    # Stable order prevents request order from changing this deterministic policy.
    selected = sorted(selected)
    by_mask = defaultdict(list)
    for expert in selected:
        by_mask[layout.shards[expert].mask].append(layout.shards[expert])
    maximum_contexts = PLANES // POLICY_WIDTH[layout.policy]
    contexts = maximum_contexts if contexts is None else contexts
    if not isinstance(contexts, int) or contexts <= 0:
        raise ValueError("contexts must be a positive integer")
    masks = sorted(by_mask)
    batches = [masks[i:i + contexts] for i in range(0, len(masks), contexts)]
    useful = sum(layout.shards[e].nbytes for e in selected)
    pages = sum(layout.shards[e].pages for e in selected)
    cold = sum(len(layout.shards[e].segments) for e in selected)
    total_stream_waves = sum(layout.shards[e].waves for e in selected)
    service = nand = ecc = retained_sync_nand = conservative = waves = peak = plane_cold = 0
    wave_records = []
    plane_timeline = [[] for _ in range(PLANES)]
    for batch in batches:
        batch_pages = sum(s.pages for mask in batch for s in by_mask[mask])
        lane_loads = [sum(s.nand_ns for s in by_mask[mask]) for mask in batch]
        nand += max(lane_loads)
        batch_ecc = transfer_ns(batch_pages * PAGE, ecc_bw)
        ecc += batch_ecc
        service += max(max(lane_loads), batch_ecc)
        lane_sequences = [iter(w for s in by_mask[mask] for w in _stream_waves(s)) for mask in batch]
        # Keep fractional shared-drain time until a whole schedule/batch ends.
        # Per-page integer ceiling would spuriously reduce the stated bandwidth.
        drain_numerator = 0
        while True:
            current = [next(seq, None) for seq in lane_sequences]
            current = [w for w in current if w is not None]
            if not current:
                break
            peak = max(peak, len(current))
            retained_duration = max(w["duration"] for w in current)
            duration = COLD_NS
            start = conservative
            ready = start + duration
            retained_sync_nand += retained_duration
            occupied = sum(len(w["mask"]) for w in current)
            plane_cold += sum(len(w["mask"]) for w in current if w["cold"])
            previous_drain = transfer_ns(drain_numerator, ecc_bw)
            for w in current:
                for j, plane in enumerate(w["mask"]):
                    drain_start = ready + transfer_ns(drain_numerator, ecc_bw) - previous_drain
                    drain_numerator += PAGE
                    drain_end = ready + transfer_ns(drain_numerator, ecc_bw) - previous_drain
                    if include_timeline:
                        entry = dict(wave=waves, expert=w["expert"], plane=plane,
                                     block=w["block"], wordline=w["wordline"],
                                     page_index=w["page_offset"] + j, cold=True,
                                     stream_cold=w["cold"], pipeline_nand_duration_ns=w["duration"],
                                     nand_start_ns=start, nand_end_ns=start + duration,
                                     drain_start_ns=drain_start, drain_end_ns=drain_end)
                        plane_timeline[plane].append(entry)
            # Only the drain increment in this epoch follows its NAND read.
            increment = transfer_ns(drain_numerator, ecc_bw) - previous_drain
            conservative = ready + increment
            if include_timeline:
                addresses = {(w["block"], w["wordline"]) for w in current}
                wave_records.append(dict(wave=waves, start_ns=start, nand_end_ns=ready,
                                         end_ns=conservative, active_planes=occupied,
                                         experts=[w["expert"] for w in current],
                                         same_block_wordline=len(addresses) == 1))
            waves += 1
    assert conservative >= service
    assert pages <= PLANES * waves
    result = dict(service_ns=service, ideal_overlap_bound_ns=service, nand_ns=nand,
                  ecc_ns=ecc, sync_nand_ns=waves * COLD_NS,
                  retained_cr_sync_nand_ns=retained_sync_nand,
                  retained_cr_schedule_ns=retained_sync_nand + ecc,
                  conservative_service_ns=conservative,
                  useful_bytes=useful, physical_pages=pages, read_waves=waves,
                  cold_reads=cold, plane_cold_reads=plane_cold,
                  conservative_cold_waves=waves, conservative_plane_cold_reads=pages,
                  command_count=cold, matrix_commands=cold,
                  nand_read_commands=total_stream_waves,
                  page_read_commands=pages, peak_contexts=peak,
                  input_contexts_required=(1 if common_input and peak else peak),
                  accumulator_contexts_required=peak,
                  page_slot_fill=pages / (PLANES * waves) if waves else 0,
                  plane_slot_fill=useful / (PLANES * waves * PAGE) if waves else 0,
                  physical_page_fill=useful / (pages * PAGE) if pages else 0,
                  idle_wave_slots=PLANES * waves - pages,
                  shared_ecc_pe_bandwidth_bytes_per_second=ecc_bw,
                  timing_kind="ideal cr-pipeline bound; all-cold serialized-drain schedule; conditional retained-cr schedule",
                  flexible_address_required=layout.policy != "baseline",
                  common_input=common_input)
    if include_timeline:
        result.update(plane_timeline=plane_timeline, wave_timeline=wave_records)
    return result


def evaluate_phase(selected_experts, chip_layouts, **kwargs):
    """Independent chips operate concurrently; empty/unassigned chips stay idle."""
    chips = [evaluate_chip(layout, [e for e in selected_experts if e in layout.shards], **kwargs)
             for layout in chip_layouts]
    additive = ("useful_bytes", "physical_pages", "read_waves", "cold_reads", "plane_cold_reads",
                "command_count", "matrix_commands", "nand_read_commands", "page_read_commands", "idle_wave_slots")
    result = {key: sum(c[key] for c in chips) for key in additive}
    result.update(service_ns=max((c["service_ns"] for c in chips), default=0),
                  conservative_service_ns=max((c["conservative_service_ns"] for c in chips), default=0),
                  chips=chips, peak_contexts=max((c["peak_contexts"] for c in chips), default=0))
    return result


def microbench_results():
    """Small reproducible fixtures; no route-dependent relocation or byte inflation."""
    fixtures = {
        "two_plus_two_complementary": {0: 2 * PAGE, 1: 2 * PAGE},
        "two_plus_two_pair_conflict": {0: 2 * PAGE, 2: 2 * PAGE},
        "one_plus_three": {0: PAGE, 1: 3 * PAGE},
        "five_pages_one_expert": {0: 5 * PAGE},
        "four_one_page_experts": {e: PAGE for e in range(4)},
        "four_one_byte_experts": {e: 1 for e in range(4)},
        "two_pairs_crossing_block": {0: 193 * 2 * PAGE, 1: 193 * 2 * PAGE},
        "long_full_stream_bandwidth": {0: 192 * 4 * PAGE},
    }
    records = []
    for fixture, expert_bytes in fixtures.items():
        for policy in POLICY_WIDTH:
            layout = build_layout(expert_bytes, policy=policy)
            assert layout.validate()
            small = sum(s.pages for s in layout.shards.values()) <= 16
            result = evaluate_chip(layout, list(expert_bytes), include_timeline=small)
            result.update(fixture=fixture, policy=policy, expert_bytes=expert_bytes,
                          capacity=layout.summary())
            records.append(result)
    return records


def main():
    import argparse
    import csv
    import json
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = microbench_results()
    (args.output_dir / "results.json").write_text(json.dumps(records, indent=2) + "\n")
    scalars = [{k: v for k, v in r.items() if not isinstance(v, (list, dict))} for r in records]
    with (args.output_dir / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(scalars[0]))
        writer.writeheader()
        writer.writerows(scalars)
    for r in records:
        print(f'{r["fixture"]:34} {r["policy"]:8} bound={r["service_ns"]/1000:9.3f} us '
              f'schedule={r["conservative_service_ns"]/1000:9.3f} us '
              f'slots={r["page_slot_fill"]:.3f} useful={r["plane_slot_fill"]:.6f}')


if __name__ == "__main__":
    main()
