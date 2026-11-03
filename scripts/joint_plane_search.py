#!/usr/bin/env python3
"""Capacity-preserving, train-only joint chip-group/plane placement search.

An expert owns one immutable slot per layer.  The slot determines its chip
group and its disjoint plane subset; swapping slots moves entire output-row
shards, never selected pages or runtime requests.  ``use_cr=False`` changes
only NAND read latency, preserving the existing ideal ECC overlap convention.
"""
import random

import numpy as np

import aif_model as flash
import run_plane_wave_study as previous

GROUPS = previous.GROUPS
WIDTHS = previous.WIDTHS
COMPONENTS = previous.COMPONENTS
COUNTERS = previous.COUNTERS
BLOCK_PAGES = previous.BLOCK_PAGES


def phase_profiles(dimensions, g, width, use_cr=True):
    result = previous.phase_profiles(dimensions, g, width)
    for p in result:
        p['use_cr'] = bool(use_cr)
        if not use_cr:
            p['nand'] = p['waves'] * flash.COLD_NS
    return result


def decode_slots(slots, g, width):
    slots = np.asarray(slots, dtype=np.int16)
    ngroups = 16 // g
    return slots % ngroups, (slots // ngroups) % (4 // width)


def counts_for(routes, slots, g, width):
    groups, bins = decode_slots(slots, g, width)
    codes = groups[routes] * (4 // width) + bins[routes]
    counts = np.stack([(codes == code).sum(axis=-1)
                       for code in range((16 // g) * (4 // width))], axis=-1)
    return counts.reshape((*routes.shape[:-1], 16 // g, 4 // width))


def slots_from_group_bins(placement, bins, g, width):
    """Recover a slot permutation from v1 group and trained-bin assignments."""
    placement = np.asarray(placement, dtype=np.int16)
    ngroups, nbin = 16 // g, 4 // width
    target_codes = (placement % ngroups) * nbin + np.asarray(bins)
    original_groups, original_bins = decode_slots(np.arange(len(placement)), g, width)
    original_codes = original_groups * nbin + original_bins
    result = np.empty(len(placement), dtype=np.int16)
    for code in range(ngroups * nbin):
        experts = np.flatnonzero(target_codes == code)
        available = np.flatnonzero(original_codes == code)
        if len(experts) != len(available):
            raise ValueError('The v1 seed does not preserve group/bin capacity')
        result[experts] = available
    assert sorted(result.tolist()) == list(range(len(placement)))
    return result.tolist()


def costs_from_counts(counts, profiles, g, width, active):
    """Complete latency/counters; NAND/ECC pipeline is identical in both modes."""
    result = previous.costs_from_counts(counts, profiles, g, width, active,
                                        timing='ideal_overlap')
    if profiles and not profiles[0]['use_cr']:
        flat = np.asarray(counts).reshape((-1, 16 // g, 4 // width))
        peak_bins = flat.max(axis=-1)
        epochs = sum(peak_bins.sum(axis=-1) * int(p['waves'].sum()) for p in profiles)
        result['ordinary_read_epochs'] = epochs.reshape(counts.shape[:-2])
    return result


def evaluate_routes(routes, layers, dimensions, g, width, mappings, use_cr=True):
    shape = routes.shape[:2]
    result = {k: np.zeros(shape, dtype=np.int64)
              for k in (*COMPONENTS, *COUNTERS, 'model_ns')}
    result['phases'] = {p: {k: np.zeros(shape, dtype=np.int64) for k in COMPONENTS}
                        for p in ('gate_up', 'down')}
    for li, layer in enumerate(layers):
        counts = counts_for(routes[:, :, li], mappings[str(layer)], g, width)
        values = costs_from_counts(counts, phase_profiles(dimensions[layer], g, width, use_cr),
                                   g, width, routes.shape[-1])
        for k in result:
            if k != 'phases':
                result[k] += values[k]
        for phase in result['phases']:
            for k in COMPONENTS:
                result['phases'][phase][k] += values['phases'][phase][k]
    return result


def model_ns_from_counts(counts, profiles, g, width, active):
    """Exact full objective without allocating the diagnostic counter arrays.

    Channel pairs remain physical chips c and c+8.  Common gate/up input is
    sent once per occupied chip, whereas down input is expert-specific.
    """
    counts = np.asarray(counts).reshape((-1, 16 // g, 4 // width))
    group_counts = counts.sum(axis=-1)
    bin_peak = counts.max(axis=(1, 2)).astype(np.int64)
    group_peak = group_counts.max(axis=1)
    ngroups = 16 // g
    if ngroups > 1:
        half = ngroups // 2
        paired = group_counts[:, :half] + group_counts[:, half:]
        paired_peak = paired.max(axis=1)
        occupied_pair_peak = ((group_counts[:, :half] > 0).astype(np.int64) +
                              (group_counts[:, half:] > 0)).max(axis=1)
    else:
        paired_peak = 2 * group_counts[:, 0]
        occupied_pair_peak = 2 * (group_counts[:, 0] > 0)
    result = np.zeros(len(counts), dtype=np.int64)
    for p in profiles:
        result += np.maximum(bin_peak * int(p['nand'].max(initial=0)),
                             group_peak * int(p['ecc'].max(initial=0)))
        result += 250 * active * int(p['cold'].sum()) + 2000
        channel_in = (occupied_pair_peak if p['common'] else paired_peak) * p['columns']
        host_in = p['columns'] * (1 if p['common'] else active)
        result += np.maximum((host_in + 7) // 8, (channel_in + 1) // 2)
        host_out = active * int(p['rows'].sum()) * 4
        if np.all(p['rows'] == p['rows'][0]):
            channel_out = paired_peak * int(p['rows'][0]) * 4
        else:
            outputs = np.repeat(group_counts, g, axis=1) * np.tile(p['rows'] * 4, ngroups)
            channel_out = (outputs[:, :8] + outputs[:, 8:]).max(axis=1)
        result += np.maximum((host_out + 7) // 8, (channel_out + 1) // 2)
    return result


def score_slots(unique, weights, slots, g, width, profiles):
    counts = counts_for(unique, slots, g, width)
    latency = model_ns_from_counts(counts, profiles, g, width, unique.shape[-1])
    squared = np.square(counts.astype(np.int64)).sum(axis=(1, 2))
    return int(latency @ weights), int(squared @ weights)


def optimize_slots(unique, weights, experts, initial_slots, g, width, profiles,
                   trials, seed, allow_group_moves=True):
    """Fixed proposal budget with exact incremental full-cost acceptance.

    Each accepted proposal swaps two complete expert slots.  Only routes
    selecting exactly one of those experts can change, so rejected proposals
    need not reevaluate unrelated routes.  Lexicographic occupancy is only a
    tie breaker after the complete latency objective.
    """
    slots = np.asarray(initial_slots, dtype=np.int16).copy()
    if sorted(slots.tolist()) != list(range(experts)):
        raise ValueError('initial_slots must be an expert-to-slot permutation')
    unique = np.asarray(unique)
    weights = np.asarray(weights, dtype=np.int64)
    ngroups, nbin = 16 // g, 4 // width
    groups, bins = decode_slots(slots, g, width)
    codes = groups * nbin + bins
    incidence = np.zeros((len(unique), experts), dtype=np.int8)
    incidence[np.arange(len(unique))[:, None], unique] = 1
    counts = counts_for(unique, slots, g, width).reshape(len(unique), -1).astype(np.int16)
    latency = model_ns_from_counts(counts, profiles, g, width, unique.shape[-1])
    squares = np.square(counts.astype(np.int64)).sum(axis=1)
    initial = current = (int(latency @ weights), int(squares @ weights))
    capacity_before = np.bincount(codes, minlength=ngroups * nbin)
    accepted = attempted = cross_group = cross_plane = both = 0
    rng = random.Random(seed)
    # Fixed-group sampling always chooses an eligible group, avoiding the
    # otherwise overwhelming fraction of invalid pairs for narrow groups.
    eligible = [np.flatnonzero(groups == group).tolist() for group in range(ngroups)
                if len(set(codes[groups == group].tolist())) > 1]
    has_moves = len(set(codes.tolist())) > 1 if allow_group_moves else bool(eligible)
    for _ in range(trials if has_moves else 0):
        attempted += 1
        if allow_group_moves:
            a, b = rng.sample(range(experts), 2)
        else:
            a, b = rng.sample(rng.choice(eligible), 2)
        ca, cb = int(codes[a]), int(codes[b])
        if ca == cb:
            continue
        difference = incidence[:, b] - incidence[:, a]
        affected = np.flatnonzero(difference)
        delta = difference[affected]
        old_ca, old_cb = counts[affected, ca].copy(), counts[affected, cb].copy()
        counts[affected, ca] += delta
        counts[affected, cb] -= delta
        changed = counts[affected]
        new_latency = model_ns_from_counts(changed, profiles, g, width, unique.shape[-1])
        new_squares = squares[affected] + 2 * delta * (old_ca - old_cb) + 2
        candidate = (current[0] + int((new_latency - latency[affected]) @ weights[affected]),
                     current[1] + int((new_squares - squares[affected]) @ weights[affected]))
        if candidate < current:
            moved_group, moved_plane = groups[a] != groups[b], bins[a] != bins[b]
            slots[a], slots[b] = slots[b], slots[a]
            groups[a], groups[b] = groups[b], groups[a]
            bins[a], bins[b] = bins[b], bins[a]
            codes[a], codes[b] = codes[b], codes[a]
            latency[affected] = new_latency
            squares[affected] = new_squares
            current = candidate
            accepted += 1
            cross_group += int(moved_group)
            cross_plane += int(moved_plane)
            both += int(moved_group and moved_plane)
        else:
            counts[affected, ca] = old_ca
            counts[affected, cb] = old_cb
    assert sorted(slots.tolist()) == list(range(experts))
    np.testing.assert_array_equal(capacity_before,
                                  np.bincount(codes, minlength=ngroups * nbin))
    assert current <= initial
    return slots.tolist(), dict(proposals_budget=trials, attempted=attempted, accepted=accepted,
                               accepted_cross_group=cross_group, accepted_cross_plane=cross_plane,
                               accepted_both=both, unique_routes=len(unique),
                               initial_model_ns_weighted=initial[0], final_model_ns_weighted=current[0],
                               initial_squared_occupancy=initial[1], final_squared_occupancy=current[1],
                               capacity_before=capacity_before.tolist(),
                               capacity_after=np.bincount(codes, minlength=ngroups * nbin).tolist(),
                               allow_group_moves=bool(allow_group_moves), seed=seed)
