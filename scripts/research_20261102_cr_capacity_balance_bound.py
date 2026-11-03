#!/usr/bin/env python3
"""A page-count capacity certificate, with small exhaustive cross-checks.

Elementary swap-one-expert argument; no claim of a new general theorem or
achievable flash schedule. Owner-homogeneous pages, one copy, all top-K sets.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import itertools as it
import json
from math import ceil
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/research-2026-11-02/round3-cr-ipr/capacity-balance-bound"


def compositions(total, width):
    if width == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in compositions(total - first, width - 1):
            yield (first, *rest)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def formula(planes, experts, selected, pages_per_expert, shared_pages):
    total = selected * pages_per_expert + shared_pages
    if total % planes:
        return dict(feasible_exact_full_waves=False, reason="selected pages not divisible by planes")
    waves = total // planes
    ceiling = ceil(pages_per_expert / planes)
    if selected * ceiling > waves:
        return dict(feasible_exact_full_waves=False, reason="insufficient shared pages to compensate integer expert profiles", waves=waves)
    profile = [pages_per_expert // planes + (p < pages_per_expert % planes) for p in range(planes)]
    shared = [waves - selected * x for x in profile]
    resident = [experts * a + b for a, b in zip(profile, shared)]
    bound = waves + (experts - selected) * ceiling
    assert sum(profile) == pages_per_expert and sum(shared) == shared_pages and min(shared) >= 0
    assert max(resident) == bound
    return dict(feasible_exact_full_waves=True, waves=waves,
                minimum_max_resident_pages=bound, expert_profile=profile,
                shared_profile=shared, resident_pages_per_plane=resident)


def exhaustive(planes, experts, selected, pages, shared):
    predicted = formula(planes, experts, selected, pages, shared)
    if (selected * pages + shared) % planes:
        return dict(parameters=[planes, experts, selected, pages, shared], skipped_nonintegral_total=True)
    waves = (selected * pages + shared) // planes
    profiles = np.asarray(list(compositions(pages, planes)), dtype=np.int64)
    public = np.asarray(list(compositions(shared, planes)), dtype=np.int64)
    routes = np.asarray(list(it.combinations(range(experts), selected)), dtype=np.int64)
    valid = 0
    minimum = None
    by_capacity = {}
    evaluated = 0
    # Enumerate every assignment, including different profiles per expert.
    # Broadcast over every shared layout and every route, without invoking
    # the swap-one-expert lemma to prune the search.
    for assignment in it.product(range(len(profiles)), repeat=experts):
        matrix = profiles[list(assignment)]
        routed = matrix[routes].sum(axis=1)
        accepted = np.all(routed[None, :, :] + public[:, None, :] == waves, axis=(1, 2))
        evaluated += len(public)
        for index in np.flatnonzero(accepted):
            valid += 1
            assert np.all(matrix == matrix[0]), "swap-one-expert necessity failed"
            footprint = int(np.max(matrix.sum(axis=0) + public[index]))
            minimum = footprint if minimum is None else min(minimum, footprint)
            for capacity in range(0, 2 + experts * pages + shared):
                if footprint <= capacity:
                    by_capacity[capacity] = by_capacity.get(capacity, 0) + 1
    assert bool(valid) == predicted["feasible_exact_full_waves"], (planes, experts, selected, pages, shared)
    if valid:
        assert minimum == predicted["minimum_max_resident_pages"]
    for capacity in range(0, 2 + experts * pages + shared):
        expected = predicted["feasible_exact_full_waves"] and predicted["minimum_max_resident_pages"] <= capacity
        assert bool(by_capacity.get(capacity, 0)) == expected
    return dict(parameters=[planes, experts, selected, pages, shared], layout_pairs_checked=evaluated,
                all_routes_per_layout=len(routes), accepted_universal_layouts=valid,
                exhaustive_minimum_max_resident=minimum, predicted=predicted,
                all_capacity_thresholds_match=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / "report.json").exists(), "preserve sealed artifacts"
    started = time.monotonic()
    grid = [
        (p, e, k, l, s)
        for p, e, ks, ls, ss in [
            (2, 4, [1, 2, 3], range(4), range(7)),
            (3, 4, [2], range(3), range(5)),
            (4, 3, [1, 2], range(3), range(4)),
        ]
        for k in ks for l in ls for s in ss
    ]
    dump(OUT / "execution.frozen.json", {
        "utc": dt.datetime.now(dt.timezone.utc).isoformat(), "source_sha256": sha(__file__),
        "grid": grid, "method": "exhaust all expert-profile assignments, all shared profiles, all selected subsets; then compare every capacity threshold",
        "data": "No routing data used; mathematical geometry certificate only.",
    })
    records = [exhaustive(*parameters) for parameters in grid]
    examples = {}
    for model, values in {
        "granite4tiny_GU_W8_g16": (4, 64, 6, 6, 12),
        "deepseek_GU_W8_g16": (4, 64, 6, 22, 44),
    }.items():
        result = formula(*values)
        result["parameters_P_E_K_L_S"] = values
        result["minimum_reserved_blocks_per_plane_LB"] = ceil(result["minimum_max_resident_pages"] / 192)
        result["mean_occupancy_block_LB"] = ceil((values[1] * values[3] + values[4]) / (values[0] * 192))
        result["native_schedule_certified"] = False
        examples[model] = result
    report = dict(
        status="pass", source_sha256=sha(__file__), parameter_cases=len(records),
        layout_pairs_checked=sum(r.get("layout_pairs_checked", 0) for r in records),
        all_universal_and_capacity_threshold_checks_match=True, records=records, examples=examples,
        statement={
            "premises": "P planes, E>K>=1 equal-size experts with L owner-homogeneous full pages each, S shared full pages, one copy, every top-K subset, zero overfetch, exactly W=(K*L+S)/P full waves.",
            "necessity": "Swapping one expert in a top-K set forces identical per-expert plane counts v_p. Shared count is W-K*v_p. Resident count is W+(E-K)*v_p.",
            "criterion": "If W is integral, page-count feasibility requires K*ceil(L/P)<=W. When it holds, minimum maximum resident pages is W+(E-K)*ceil(L/P).",
            "proof": "For any pair of experts, choose a common K-1 subset excluding the pair. Equal per-plane demand in the two routes gives equal expert profiles. Some v_p>=ceil(L/P); nearly balanced integer v attains the minimum and gives nonnegative shared counts exactly when the criterion holds.",
        },
        limitations=[
            "Elementary counting lemma, not a new general theorem or a publication breakthrough.",
            "Sufficiency is only for global page counts: it proves no row-stream, same-block epoch, finite-PB, ECC or native timing feasibility.",
            "No guarantee for nonuniform expert sizes, mixed-owner pages, copies, inactive overfetch, variable shard widths, or changing the selected set.",
            "A TRAIN-only layout need not serve all top-K sets equally and can trade a small route-specific imbalance for lower resident capacity.",
            "A lower bound on resident blocks is not a bound on actually touched blocks, cold starts, or latency.",
        ], elapsed_seconds=time.monotonic() - started,
    )
    dump(OUT / "report.json", report)
    (OUT / "report.md").write_text(
        "For owner-homogeneous full pages and every top-K set, universal full-wave utilization forces all experts to have the same plane-count profile. "
        "If W=(K L+S)/P is integral and K ceil(L/P)<=W, the minimum possible maximum resident footprint is W+(E-K)ceil(L/P). "
        "This is an elementary global page-count certificate, not native scheduling sufficiency.\n\n"
        f"Exhaustive checks: {report['parameter_cases']} parameter cases and {report['layout_pairs_checked']:,} layout pairs, including nonidentical expert profiles. Every route and capacity threshold agrees.\n\n"
        "G4 (L=6,S=12,E=64,K=6,P=4): W=12, minimum maximum footprint128pages; one192-WL block suffices for counts. "
        "DeepSeek (L=22,S=44): W=44, minimum392pages on a plane, exceeding two192-WL blocks. "
        "Thus no such one-copy universal44-full-wave layout fits two blocks per plane, although the mean resident footprint363pages does.\n\n"
        + "\n".join(report["limitations"]) + "\n"
    )
    print(json.dumps({k: report[k] for k in ["status", "parameter_cases", "layout_pairs_checked", "elapsed_seconds"]}))


if __name__ == "__main__":
    main()
