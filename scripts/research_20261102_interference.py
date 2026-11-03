#!/usr/bin/env python3
"""Train-only screen of fixed NAND row replicas under synthetic chip stalls.

This is a sensitivity model, not an FTL simulation or hardware measurement.
Every base expert remains evenly striped over 16 chips.  A canonical tile is
whole output rows, starts on a four-plane wave boundary, and has one optional
fixed alternate chip.  Alternate shifts are replicated cyclically for all 16
source chips, so every chip has exactly the same additional reserved capacity.
No weight moves during inference, no routing changes, and no CR reads occur.

The exact online selector is deliberately shared by all policies, including
the strong frequency/striping baselines.  Its objective is the completion of
all selected experts in the phase, not independent request hit rate.
"""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np

import aif_model as flash
from materialize_plan import WaveAllocator
from research_20261102_projection_study import equal_rows, evaluate, model_phases
from run_corpus_traces import DATA, ROOT, geometry, sha256

DEST = ROOT / 'results/research-2026-11-02/round2/interference'
PAGE = 16384
WAVE = 4 * PAGE
READ = 28000
STALLS = (0, 28000, 56000, 112000, 3000000, 5000000)
BUDGETS = (0.0, 0.1, 0.25, 0.5, 1.0)
SEED = 2026110207


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def canonical_tiles(total_rows, columns):
    assert total_rows % 16 == 0
    shard_rows = total_rows // 16
    aligned_rows = WAVE // math.gcd(columns, WAVE)
    tile_rows = min(aligned_rows, shard_rows)
    tiles = []
    for start in range(0, shard_rows, tile_rows):
        count = min(tile_rows, shard_rows - start)
        pages = (count * columns + PAGE - 1) // PAGE
        tiles.append(dict(row_start=start, rows=count, bytes=count * columns,
                          pages=pages, waves=(pages + 3) // 4))
    q = ((shard_rows * columns + PAGE - 1) // PAGE + 3) // 4
    assert sum(t['waves'] for t in tiles) == q
    assert sum(t['rows'] for t in tiles) == shard_rows
    # The five current models have homogeneous tile read-work, even if the
    # final tile occupies fewer useful bytes/pages.  Reject other geometries.
    grains = {t['waves'] for t in tiles}
    assert len(grains) == 1, (columns, tiles)
    return tiles, q, next(iter(grains))


def copies_to_counts(colors, selected):
    """Counts by alternate shift 1..15, relative to the blocked source chip."""
    counts = np.zeros((colors.shape[0], 15), dtype=np.int32)
    for shift in range(1, 16):
        counts[:, shift - 1] = (colors == shift).sum(axis=1)
    return counts[selected].sum(axis=1)


def completion_extra(counts, base_tiles, grain, stall_ns, relaxed=False):
    """Exact minimum extra NAND time for one blocked, otherwise balanced chip.

    Baseline work per chip is B=base_tiles*grain waves.  If each destination
    accepts at most j tiles, m(j)=sum_d min(available_d,j) tiles can leave the
    blocked chip.  The best completion for this cap is
      B*READ + max(j*grain*READ, stall_ns-m(j)*grain*READ),
    except that the blocked chip is not waited for after all B work is moved.
    The optimum lies at one of the two integer caps around the crossing.
    This is an exact bounded-assignment result for this explicitly stated
    single-source/equal-tile model, not an oracle placement claim.
    """
    n = len(counts)
    if stall_ns == 0:
        return np.zeros(n, dtype=np.int64)
    unit = grain * READ
    limit = (stall_ns + unit - 1) // unit
    lo = np.zeros(n, dtype=np.int64)
    hi = np.full(n, limit, dtype=np.int64)

    def available(cap):
        if relaxed:
            return np.minimum(base_tiles, 15 * cap)
        return np.minimum(counts, cap[:, None]).sum(axis=1)

    for _ in range(int(limit).bit_length() + 1):
        mid = (lo + hi) // 2
        moved = available(mid)
        crossed = (mid * unit >= stall_ns - moved * unit) | (moved == base_tiles)
        hi = np.where(crossed, mid, hi)
        lo = np.where(crossed, lo, mid + 1)
        if np.all(lo >= hi):
            break

    def cost(cap):
        moved = available(cap)
        donor = np.where(moved == base_tiles, -base_tiles * unit,
                         stall_ns - moved * unit)
        return np.maximum(cap * unit, donor)

    value = np.minimum(cost(hi), cost(np.maximum(0, hi - 1)))
    return np.minimum(value, stall_ns).astype(np.int64)


def brute_force(counts, base_tiles, grain, stall):
    jobs = [d for d, number in enumerate(counts) for _ in range(int(number))]
    base = base_tiles * grain * READ
    best = base + stall
    for mask in itertools.product((0, 1), repeat=len(jobs)):
        loads = [0] * 15
        moved = 0
        for choose, destination in zip(mask, jobs):
            if choose:
                loads[destination] += grain * READ
                moved += 1
        donor = 0 if moved == base_tiles else stall + (base_tiles - moved) * grain * READ
        best = min(best, max(donor, base + max(loads)))
    return best - base


def audits():
    rng = np.random.default_rng(SEED)
    number = 0
    for grain in (1, 2, 3, 7):
        for base_tiles in range(1, 7):
            for _ in range(12):
                count = np.zeros(15, dtype=np.int64)
                for destination in rng.integers(0, 5, size=rng.integers(0, base_tiles + 1)):
                    count[destination] += 1
                for stall in (0, 1, 28000, 56000, 112000, 3000000, 5000000):
                    fast = completion_extra(count[None], base_tiles, grain, stall)[0]
                    slow = brute_force(count, base_tiles, grain, stall)
                    assert fast == slow, (count, base_tiles, grain, stall, fast, slow)
                    number += 1
    # Every all-to-all symmetric zero-control has exactly its original work.
    assert np.all(completion_extra(np.ones((20, 15), dtype=int), 15, 1, 0) == 0)
    return dict(independent_exhaustive_assignments_checked=number,
                homogeneous_zero_control=True)


def marginal_null(routes, experts, seed):
    """Curveball trades preserve every row's top-k and exact expert counts."""
    result = np.sort(routes.copy(), axis=1)
    before = np.bincount(result.ravel(), minlength=experts)
    rng = np.random.default_rng(seed)
    for _ in range(4 * len(result)):
        a, b = rng.choice(len(result), size=2, replace=False)
        common = np.intersect1d(result[a], result[b])
        left = np.setdiff1d(result[a], common)
        right = np.setdiff1d(result[b], common)
        if not len(left):
            continue
        pool = np.concatenate([left, right])
        rng.shuffle(pool)
        result[a] = np.sort(np.concatenate([common, pool[:len(left)]]))
        result[b] = np.sort(np.concatenate([common, pool[len(left):]]))
    after = np.bincount(result.ravel(), minlength=experts)
    assert np.array_equal(before, after)
    assert np.all(np.diff(result, axis=1) > 0)
    incidence = np.eye(experts, dtype=np.int32)
    a = incidence[routes].sum(axis=1)
    b = incidence[result].sum(axis=1)
    return result, dict(exact_marginal_counts_preserved=True,
                        top_k_preserved=True,
                        changed_row_fraction=float(np.any(result != np.sort(routes, axis=1), axis=1).mean()),
                        coactivation_l1_change=int(np.abs(a.T @ a - b.T @ b).sum()))


def balanced_colors(mask, frequencies, seed=0):
    """Frequency-only physical striping, with per-expert destination diversity."""
    rng = np.random.default_rng(seed)
    colors = np.zeros_like(mask, dtype=np.int16)
    load = np.zeros(15, dtype=float)
    for expert in np.argsort(-frequencies, kind='stable'):
        within = np.zeros(15, dtype=np.int32)
        for tile in np.flatnonzero(mask[expert]):
            # First spread each expert over distinct chips, then balance global
            # expected redirected work.  No coactivation enters this rule.
            scores = load + 2 * frequencies[expert] * within
            choices = np.flatnonzero(np.isclose(scores, scores.min(), rtol=0, atol=1e-12))
            color = int(rng.choice(choices))
            colors[expert, tile] = color + 1
            load[color] += frequencies[expert]
            within[color] += 1
    return colors


def marginal_masks(experts, tile_count, quota, frequencies):
    number = min(experts * tile_count, quota)
    whole = np.zeros((experts, tile_count), dtype=bool)
    whole[np.argsort(-frequencies, kind='stable')[:number // tile_count]] = True
    result = {'whole_frequency': whole}
    hot = np.zeros_like(whole)
    flat_order = [(int(e), t) for e in np.argsort(-frequencies, kind='stable')
                  for t in range(tile_count)]
    for e, t in flat_order[:number]:
        hot[e, t] = True
    result['tile_frequency'] = hot
    for exponent in (0, .5, 1):
        count = np.zeros(experts, dtype=int)
        weight = np.maximum(frequencies, 1e-9) ** exponent
        for _ in range(number):
            # Weighted fair apportionment is a stronger baseline than filling
            # complete hot experts and already exploits partial-row replicas.
            score = np.where(count < tile_count, weight / (count + 1), -1)
            count[int(np.argmax(score))] += 1
        result['tile_marginal_power_' + str(exponent)] = np.arange(tile_count)[None, :] < count[:, None]
    return result


def score(colors, selected, base_tiles, grain, stall):
    return float(completion_extra(copies_to_counts(colors, selected), base_tiles, grain, stall).mean())


def optimize(start, selected, frequencies, base_tiles, grain, stall, proposals, seed):
    rng = np.random.default_rng(seed)
    current = start.copy()
    current_score = score(current, selected, base_tiles, grain, stall)
    initial_score = current_score
    experts, tile_count = current.shape
    accepted = 0
    for step in range(proposals):
        candidate = current.copy()
        occupied = np.argwhere(candidate > 0)
        empty = np.argwhere(candidate == 0)
        if not len(occupied):
            break
        mutation = step % 4
        if mutation == 0 and len(empty):
            a = occupied[int(rng.integers(len(occupied)))]
            b = empty[int(rng.integers(len(empty)))]
            candidate[tuple(b)] = candidate[tuple(a)]
            candidate[tuple(a)] = 0
        elif mutation == 1:
            full = np.flatnonzero((candidate > 0).all(axis=1))
            vacant = np.flatnonzero((candidate == 0).all(axis=1))
            if len(full) and len(vacant):
                a, b = int(rng.choice(full)), int(rng.choice(vacant))
                candidate[[a, b]] = candidate[[b, a]]
            else:
                continue
        else:
            e, t = occupied[int(rng.integers(len(occupied)))]
            candidate[e, t] = int(rng.integers(1, 16))
        value = score(candidate, selected, base_tiles, grain, stall)
        if value < current_score:
            current, current_score = candidate, value
            accepted += 1
    assert np.count_nonzero(current) == np.count_nonzero(start)
    return current, dict(proposals=proposals, accepted=accepted,
                         initial_extra_ns=initial_score, final_extra_ns=current_score)


def fit_policies(selected, experts, tiles, q, grain, budget, proposals, seed):
    tile_count = len(tiles)
    quota = min(experts * tile_count, int(budget * experts * q) // grain)
    frequencies = np.bincount(selected.ravel(), minlength=experts).astype(float) / len(selected)
    masks = marginal_masks(experts, tile_count, quota, frequencies)
    base_tiles = selected.shape[1] * tile_count
    primary = 3000000
    candidates = {}
    for name, mask in masks.items():
        for cseed in (0, 1):
            colors = balanced_colors(mask, frequencies, cseed)
            candidates[name + '_color' + str(cseed)] = colors
    best_name = min(candidates, key=lambda n: (score(candidates[n], selected, base_tiles, grain, primary), n))
    whole_name = min((n for n in candidates if n.startswith('whole_')),
                     key=lambda n: score(candidates[n], selected, base_tiles, grain, primary))
    chosen = candidates[best_name]
    starts = [chosen]
    # Co-loading seeds can cross the all-selected-copies coverage barrier that
    # single-item greedy selection misses; the remaining slots use frequency.
    if quota >= selected.shape[1] * tile_count and quota < experts * tile_count:
        for row in selected[np.linspace(0, len(selected) - 1, 6, dtype=int)]:
            mask = np.zeros((experts, tile_count), dtype=bool)
            mask[row] = True
            remaining = quota - np.count_nonzero(mask)
            for e in np.argsort(-frequencies, kind='stable'):
                for t in range(tile_count):
                    if remaining and not mask[e, t]:
                        mask[e, t] = True
                        remaining -= 1
            starts.append(balanced_colors(mask, frequencies, 0))
    start = min(starts, key=lambda c: score(c, selected, base_tiles, grain, primary))
    fitted, stats = optimize(start, selected, frequencies, base_tiles, grain,
                             primary, proposals, seed)
    return dict(no_replica=np.zeros_like(chosen), whole_frequency=candidates[whole_name],
                tile_frequency=candidates['tile_frequency_color0'],
                tile_uniform=candidates['tile_marginal_power_0_color0'],
                strongest_marginal=chosen, coactivation=fitted), dict(
                    capacity_budget_fraction=budget, tile_quota_per_chip=quota,
                    best_marginal_rule=best_name, optimizer=stats,
                    marginal_candidate_scores_ns={n: score(c, selected, base_tiles, grain, primary)
                                                 for n, c in candidates.items()})


def materialize_case(path, colors, tiles, columns):
    """Append immutable alternate copies after the equally striped base weights."""
    allocators = [WaveAllocator() for _ in range(16)]
    records = []
    base_rows = sum(t['rows'] for t in tiles)
    for expert in range(len(colors)):
        for chip in range(16):
            pages, waves, runs, segments = allocators[chip].allocate(base_rows * columns)
            records.append([0, expert, -1, chip, chip, chip * base_rows, base_rows, pages, waves, segments])
    for expert, ti in np.argwhere(colors > 0):
        tile = tiles[int(ti)]
        shift = int(colors[expert, ti])
        for source in range(16):
            destination = (source + shift) % 16
            assert source != destination
            pages, waves, runs, segments = allocators[destination].allocate(tile['bytes'])
            assert waves == tile['waves'] and pages == tile['pages']
            records.append([1, int(expert), int(ti), source, destination,
                            source * base_rows + tile['row_start'], tile['rows'], pages, waves, segments])
    summaries = [a.summary(c) for c, a in enumerate(allocators)]
    assert len({s['useful_int8_bytes'] for s in summaries}) == 1
    last = [0] * 16
    for kind, expert, tile, source, chip, row, count, pages, waves, segments in records:
        for block, wl, nwaves, page_offset, npages, plane_counts in segments:
            assert block * 192 + wl >= last[chip]
            assert wl + nwaves <= 192 and sum(plane_counts) == npages
            last[chip] = block * 192 + wl + nwaves
    value = dict(artifact='virtual fixed addresses; weights not programmed',
                 columns=['replica', 'expert', 'tile', 'base_chip', 'owner_chip', 'output_row_start',
                          'output_row_count', 'pages', 'waves', 'block_wl_segments'],
                 records=records, chips=summaries,
                 all_copies_whole_rows=True, nonoverlapping_addresses=True,
                 equal_extra_capacity_per_chip=True)
    with gzip.GzipFile(filename=str(path), mode='wb', mtime=0) as stream:
        stream.write(json.dumps(value, separators=(',', ':')).encode())
    return dict(path=str(path), sha256=sha256(path), streams=len(records),
                nonoverlapping_addresses=True, equal_capacity_per_chip=True)


def oracle_token_sensitivity(name, metadata, routes):
    layers, experts, active = geometry(metadata)
    dimensions = flash.tensor_dimensions(metadata)
    base = 0
    phases = []
    selected = np.arange(active, dtype=int)[None]
    for layer in layers:
        for pname, rows, columns, common in model_phases(dimensions, layer):
            tiles, q, grain = canonical_tiles(rows, columns)
            value = evaluate(selected, equal_rows(rows, experts), columns, common)
            base += int(value['model_ns'][0])
            phases.append((q, grain, len(tiles)))
    outputs = []
    # Synthetic nuisance samples are independent of routing and identical for
    # the base and oracle.  The occurrence probability is per phase, not chip.
    for probability in (0, .0001, .001, .01):
        rng = np.random.default_rng(SEED + len(layers))
        blocked = rng.random((*routes.shape[:2], len(phases))) < probability
        for stall in STALLS:
            base_times = base + blocked.sum(axis=2) * stall
            increments = []
            for q, grain, tile_count in phases:
                increments.append(int(completion_extra(np.zeros((1, 15), dtype=int),
                                  active * tile_count, grain, stall, relaxed=True)[0]))
            oracle = base + blocked @ np.asarray(increments)
            assert np.all(oracle <= base_times)
            outputs.append(dict(model=name, nuisance_probability_per_phase=probability,
                                stall_ns=stall, original_ffn_ms=base / 1e6,
                                sampled_tokens=int(base_times.size),
                                baseline_mean_ms=float(base_times.mean() / 1e6),
                                oracle_mean_ms=float(oracle.mean() / 1e6),
                                baseline_p99_ms=float(np.quantile(base_times, .99) / 1e6),
                                oracle_p99_ms=float(np.quantile(oracle, .99) / 1e6),
                                oracle_mean_gain_pct=float(100 * (1 - oracle.mean() / base_times.mean())),
                                oracle_p99_gain_pct=float(100 * (1 - np.quantile(oracle, .99) /
                                                                 np.quantile(base_times, .99))),
                                bound='all base tiles have freely selectable alternate owners; '
                                      'ignores replica capacity and fixed destinations',
                                nuisance_scope='one chip per blocked phase, iid; not measured GC rate'))
    return outputs


def token_baselines(args, name, metadata, routes, ids):
    """All-layer strong uncoded controls, with fits frozen before scoring.

    This supplements the three-anchor-layer coactivation screen.  The
    coactivation candidate is deliberately not extrapolated to unfit layers.
    """
    layers, experts, active = geometry(metadata)
    dimensions = flash.tensor_dimensions(metadata)
    names = ('no_replica', 'whole_frequency', 'tile_frequency', 'tile_uniform', 'strongest_marginal')
    plans = []
    base = 0
    for li, layer in enumerate(layers):
        train = np.sort(routes[:100, :, li].reshape(-1, active), axis=1)
        rng = np.random.default_rng(SEED + li)
        selected = train[np.sort(rng.choice(len(train), min(args.fit_events, len(train)), replace=False))]
        for pname, rows, columns, common in model_phases(dimensions, layer):
            tiles, q, grain = canonical_tiles(rows, columns)
            base += int(evaluate(np.arange(active)[None], equal_rows(rows, experts), columns, common)['model_ns'][0])
            by_budget = {}
            for budget in BUDGETS:
                policies, receipt = fit_policies(selected, experts, tiles, q, grain, budget, 0, SEED + li)
                by_budget[str(budget)] = dict(fit=receipt,
                    policies={n: policies[n].tolist() for n in names})
            plans.append(dict(layer=layer, layer_index=li, phase=pname, tiles=tiles, q=q, grain=grain,
                              base_tiles=active * len(tiles), budgets=by_budget))
    frozen_path = DEST / 'plans' / (name + '.all_layer_marginal_baselines.json.gz')
    with gzip.GzipFile(filename=str(frozen_path), mode='wb', mtime=0) as stream:
        stream.write(json.dumps(dict(model=name, fit_prompt_ids=ids[:100],
                    validation_prompt_ids=ids[100:], plans=plans,
                    fitted_on_first_100_training_prompts_only=True), separators=(',', ':')).encode())
    uniform = np.random.default_rng(SEED + len(layers)).random((100, 128, len(plans)))
    # All policies see precisely the same event stream.  No time-dependent FTL
    # state is fabricated from this iid sensitivity distribution.
    sums = {(budget, policy, stall, probability): np.full((100, 128), base, dtype=np.int64)
            for budget in BUDGETS for policy in names for stall in (3000000, 5000000)
            for probability in (0, .0001, .001, .01)}
    for pi, plan in enumerate(plans):
        selected = np.sort(routes[100:, :, plan['layer_index']].reshape(-1, active), axis=1)
        for budget in BUDGETS:
            for policy in names:
                colors = np.asarray(plan['budgets'][str(budget)]['policies'][policy], dtype=np.int16)
                counts = copies_to_counts(colors, selected)
                for stall in (3000000, 5000000):
                    extra = completion_extra(counts, plan['base_tiles'], plan['grain'], stall).reshape(100, 128)
                    for probability in (0, .0001, .001, .01):
                        sums[(budget, policy, stall, probability)] += (uniform[:, :, pi] < probability) * extra
    rows = []
    for (budget, policy, stall, probability), value in sums.items():
        baseline = sums[(budget, 'no_replica', stall, probability)]
        rows.append(dict(model=name, extra_capacity_budget=budget, policy=policy,
                         stall_ns=stall, nuisance_probability_per_phase=probability,
                         original_ffn_ms=base / 1e6, mean_ms=float(value.mean() / 1e6),
                         p99_ms=float(np.quantile(value, .99) / 1e6),
                         mean_gain_vs_no_replica_pct=float(100 * (1 - value.mean() / baseline.mean())),
                         p99_gain_vs_no_replica_pct=float(100 * (1 - np.quantile(value, .99) /
                                                                  np.quantile(baseline, .99))),
                         scope='all-layer routed FFN synthetic service, extra redirection control '
                               'costs omitted; no attention/full-model/hardware claims'))
    arrays = {f'b{budget:g}_{policy}_d{stall}_p{probability:g}': value
              for (budget, policy, stall, probability), value in sums.items()}
    arrays['nuisance_uniforms'] = uniform
    arrays['validation_prompt_ids'] = np.asarray(ids[100:])
    np.savez_compressed(DEST / (name + '.all_layer_baseline_tokens.npz'), **arrays)
    write_json(DEST / (name + '.token_baselines.json'), dict(model=name,
        frozen_plan_sha256=sha256(frozen_path), records=rows,
        original_test_routes_opened=False,
        validation_is_disjoint_half_of_existing_training_prompts=True,
        source_sha256=sha256(Path(__file__))))
    print(name, 'all-layer uncoded token baselines complete', flush=True)
    return rows


def run(args):
    DEST.mkdir(parents=True, exist_ok=True)
    audit = audits()
    protocol = dict(
        created_utc=datetime.now(timezone.utc).isoformat(), source_sha256=sha256(Path(__file__)),
        use_cr=False, chips=16, planes=4, page_bytes=PAGE, ordinary_read_ns=READ,
        scope='synthetic architecture sensitivity, not hardware or full-model end-to-end measurement',
        nuisance='one uniformly chosen chip unavailable at phase start for specified residual stall; '
                 'other chips start idle; iid occurrence per phase only in token oracle experiment',
        probabilities=[0, .0001, .001, .01], stall_ns=list(STALLS),
        primary_fit_stall_ns=3000000, budget_fractions=list(BUDGETS),
        negative_control='homogeneous read-only p=0, and every stall_ns=0 case',
        read_only_gc='read-only model weights do not generate GC; nonzero nuisance presumes '
                     'shared-device writes/GC/maintenance and is not calibrated',
        physical_family='g16 equal whole-row base; canonical wave-aligned row tiles; one '
                        'alternate per tile; cyclic source->destination shifts keep capacity equal',
        online_selector='exact minmax redirection; all baselines have identical selector',
        fit='first 100 existing training prompts only; deterministic sample of decode events',
        validation='last 100 existing training prompts, 128 steps; evaluated after fit per case; '
                   'the same loaded training cache contains both halves, but no validation '
                   'routes are passed to fitting; original corpus test_routes never opened',
        anchor_layers='first/middle/last before reading routing' if not args.all_layers else 'all',
        excluded='FTL state, write/GC simulation, interference correlation, multiple blocked chips, '
                 'host activation/combine, full-model timing, physical energy, extra tile command costs',
        overhead_note='additional addressing/setup must be charged in deployment; NAND-only '
                      'policy gains are optimistic. Existing per-phase fixed costs enter only '
                      'the full-token oracle bound, not conditional phase policy tables.',
        literature=[
            dict(title='S34MS16G2 NAND datasheet, Rev F, 2018',
                 url='https://community.infineon.com/gfawx74859/attachments/gfawx74859/USBHostsHubsTransceivers/3668/1/cypress%20semiconductor_cypr-s-a0004514289-1.pdf',
                 evidence='manufacturer p1: erase 3.5ms typical, program300us; old embedded SLC '
                          'supports order of magnitude only, not AiF calibration'),
            dict(title='HotR, DATE 2019', url='https://past.date-conference.com/proceedings-archive/2019/pdf/0111.pdf',
                 evidence='read/write interference and hot read page replication; baseline family threat'),
            dict(title='RAIL, ACM TOS 2022', url='https://doi.org/10.1145/3465406',
                 evidence='redundancy for avoiding NAND reads blocked behind writes; mechanism threat'),
            dict(title='ReadGuard, ACM TOS 2024', url='https://doi.org/10.1145/3676884',
                 evidence='block quality, read-retry and priority-aware allocation; distinct from '
                          'synthetic write/erase stalls modeled here')])
    write_json(DEST / 'protocol.json', protocol)
    summaries, oracle, receipts, token_controls = [], [], [], []
    for name in args.models:
        started = time.time()
        metadata_path = DATA / 'models' / (name + '-metadata.json')
        metadata = json.loads(metadata_path.read_text())
        train_path = DATA / 'runs/corpus-300-v1' / name / 'analysis/train_routes.npz'
        with np.load(train_path) as cache:
            routes = cache['routes']
            ids = cache['prompt_ids'].tolist()
        assert routes.shape[:2] == (200, 128) and len(ids) == 200
        layers, experts, active = geometry(metadata)
        dims = flash.tensor_dimensions(metadata)
        oracle.extend(oracle_token_sensitivity(name, metadata, routes))
        indexes = range(len(layers)) if args.all_layers else sorted({0, len(layers) // 2, len(layers) - 1})
        for li in indexes:
            train = np.sort(routes[:100, :, li].reshape(-1, active), axis=1)
            rng = np.random.default_rng(SEED + li)
            chosen_indices = np.sort(rng.choice(len(train), min(args.fit_events, len(train)), replace=False))
            fit = train[chosen_indices]
            null, null_receipt = marginal_null(fit, experts, SEED + li)
            for pname, rows, columns, common in model_phases(dims, layers[li]):
                tiles, q, grain = canonical_tiles(rows, columns)
                base_tiles = active * len(tiles)
                base_ns = active * q * READ
                for budget in BUDGETS:
                    policies, fit_receipt = fit_policies(fit, experts, tiles, q, grain, budget,
                                                       args.proposals, SEED + li)
                    if budget > 0:
                        null_policies, null_fit = fit_policies(null, experts, tiles, q, grain, budget,
                                                             args.proposals, SEED + li)
                        policies['coactivation_null_trained'] = null_policies['coactivation']
                    else:
                        null_fit = None
                        policies['coactivation_null_trained'] = policies['coactivation'].copy()
                    case_name = f'{name}.layer{layers[li]}.{pname}.budget{budget:g}'
                    frozen = dict(case=case_name, layer=layers[li], phase=pname,
                                  rows=rows, columns=columns, tile_specs=tiles, q=q, grain=grain,
                                  train_cache_sha256=sha256(train_path),
                                  metadata_sha256=sha256(metadata_path),
                                  fit_prompt_ids=ids[:100], validation_prompt_ids=ids[100:],
                                  fit_event_indices=chosen_indices.tolist(),
                                  exact_marginal_null=null_receipt, fit=fit_receipt, null_fit=null_fit,
                                  policies={n: c.tolist() for n, c in policies.items()},
                                  frozen_before_validation=True)
                    frozen_path = DEST / 'plans' / (case_name + '.json')
                    write_json(frozen_path, frozen)
                    # Validation uses a disjoint half of the original training
                    # prompts, not the previously inspected corpus test set.
                    validate = np.sort(routes[100:, :, li].reshape(-1, active), axis=1)
                    counts_by_policy = {n: copies_to_counts(c, validate) for n, c in policies.items()}
                    selected_baseline = policies['strongest_marginal']
                    for stall in STALLS:
                        comparison = {}
                        for policy, counts in counts_by_policy.items():
                            extra = completion_extra(counts, base_tiles, grain, stall)
                            total = base_ns + extra
                            assert np.all(extra <= stall) and np.all(extra >= 0)
                            if stall <= READ:
                                assert np.all(extra == stall), (name, pname, policy)
                            comparison[policy] = dict(mean_ns=float(total.mean()),
                                p99_ns=float(np.quantile(total, .99)),
                                mean_extra_ns=float(extra.mean()),
                                max_extra_ns=int(extra.max()),
                                complete_copy_coverage_fraction=float((counts.sum(axis=1) == base_tiles).mean()),
                                used_reserved_capacity_fraction=float(np.count_nonzero(policies[policy]) * grain /
                                                                        (experts * q)))
                        candidate = comparison['coactivation']
                        baseline = comparison['strongest_marginal']
                        relaxed = completion_extra(np.zeros((1, 15), dtype=int), base_tiles, grain,
                                                   stall, relaxed=True)[0]
                        summaries.append(dict(case=case_name, model=name, layer=layers[li], phase=pname,
                            budget_fraction=budget, stall_ns=stall, base_phase_nand_ns=base_ns,
                            tile_count_per_expert_chip=len(tiles), tile_wave_grain=grain,
                            validation_prompt_count=100, validation_events=len(validate),
                            policies=comparison, unlimited_copy_lower_bound_ns=int(base_ns + relaxed),
                            coactivation_gain_vs_strong_marginal_mean_pct=float(100 * (1 - candidate['mean_ns'] /
                                                                                   baseline['mean_ns'])),
                            coactivation_gain_vs_strong_marginal_p99_pct=float(100 * (1 - candidate['p99_ns'] /
                                                                                  baseline['p99_ns']))))
                    # Same-marginal, different-coactivation diagnostic on the
                    # matched calibration multigraph, with no fake accuracy claim.
                    matrix = {n: {d: score(c, r, base_tiles, grain, 3000000)
                                   for d, r in (('actual', fit), ('same_marginal_null', null))}
                              for n, c in policies.items()}
                    address = None
                    # Save full (chip, block, WL) address witnesses at 1x copies
                    # for both learned and strongest conventional policies.
                    if budget == 1 and li == indexes[0] and args.addresses:
                        address = {}
                        for policy in ('strongest_marginal', 'coactivation'):
                            path = DEST / 'plans' / (case_name + '.' + policy + '.physical.json.gz')
                            address[policy] = materialize_case(path, policies[policy], tiles, columns)
                    receipts.append(dict(case=case_name, frozen_plan_sha256=sha256(frozen_path),
                                         same_marginal_cross_matrix_extra_ns=matrix,
                                         physical_addresses=address))
                print(f'{name} layer={layers[li]} {pname}: q={q}, tile={grain} waves; '
                      f'elapsed={time.time()-started:.1f}s', flush=True)
        print(name, 'complete', time.time() - started, flush=True)
        if args.token_baselines:
            token_controls.extend(token_baselines(args, name, metadata, routes, ids))
    write_json(DEST / 'summary.json', dict(protocol_sha256=sha256(DEST / 'protocol.json'),
               audit=audit, cases=summaries, oracle_token_sensitivity=oracle,
               all_layer_token_controls=token_controls,
               receipts=receipts, test_routes_opened=False,
               script_sha256=sha256(Path(__file__))))
    write_json(DEST / 'oracle_token_sensitivity.json', oracle)
    main_cases = [c for c in summaries if c['stall_ns'] == 3000000 and c['budget_fraction'] > 0]
    compact = {name: dict(
        max_conditional_mean_gain_vs_strong_marginal_pct=max(c['coactivation_gain_vs_strong_marginal_mean_pct']
                                                           for c in main_cases if c['model'] == name),
        max_conditional_p99_gain_vs_strong_marginal_pct=max(c['coactivation_gain_vs_strong_marginal_p99_pct']
                                                          for c in main_cases if c['model'] == name))
               for name in args.models}
    write_json(DEST / 'screen.json', compact)
    print(json.dumps(compact, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=['granite', 'olmoe', 'deepseek', 'qwen3', 'mixtral'])
    parser.add_argument('--fit-events', type=int, default=512)
    parser.add_argument('--proposals', type=int, default=128)
    parser.add_argument('--all-layers', action='store_true')
    parser.add_argument('--addresses', action='store_true')
    parser.add_argument('--token-baselines', action='store_true')
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    if args.audit_only:
        print(json.dumps(audits(), indent=2))
    else:
        run(args)
