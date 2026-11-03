#!/usr/bin/env python3
"""TRAIN-only, one-copy GU page-wave controls; no native timing claim.

The counterfactual capabilities (skip to arbitrary WL in one block and IPR)
are assumptions supplied to this study. A page count is neither a legal ROW
command nor a PB/CR/ECC execution timeline. Every read below is selected GU
coefficient data; no inactive-page overfetch or replica is hidden.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ROUND2 = ROOT / "results/research-2026-11-02/round2"
DEFAULT_OUT = ROOT / "results/research-2026-11-02/round3-cr-ipr/baselines"
TRAIN = ROUND2 / "w8-format-training"
PAGE, PLANES, BLOCK, CHIPS, EXPERTS, K = 16384, 4, 192, 16, 64, 6


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def digest(array):
    a = np.ascontiguousarray(array)
    return hashlib.sha256(a.tobytes()).hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def read(path):
    return json.loads(Path(path).read_text())


def stats(x):
    a = np.asarray(x)
    return dict(mean=float(a.mean()), p99=float(np.percentile(a, 99)),
                minimum=int(a.min()), maximum=int(a.max()),
                histogram={str(int(k)): int(v) for k, v in zip(*np.unique(a, return_counts=True))})


def page_profile(pages, phase):
    return np.bincount((np.arange(pages) + phase) % PLANES,
                       minlength=PLANES).astype(np.int32)


def compact_addresses(profiles):
    """Fixed one-copy pages[expert,local_page] -> (plane,block,WL).

    Compact each plane independently. Logical page identity/order is retained
    separately; this function does not implement row accumulation or a native
    instruction. The final expert index is the always-active shared GU.
    """
    cur = np.zeros(PLANES, dtype=np.int32)
    addresses = []
    for row in profiles:
        expert = []
        for plane, n in enumerate(row):
            for wl in range(int(cur[plane]), int(cur[plane] + n)):
                expert.append((plane, wl // BLOCK, wl % BLOCK))
            cur[plane] += n
        addresses.append(np.asarray(expert, dtype=np.int32).reshape(-1, 3))
    assert len({tuple(a) for e in addresses for a in e}) == int(np.sum(profiles))
    return addresses, cur


def rr_addresses(routed_pages=6, shared_pages=12):
    addresses, start = [], 0
    for n in [routed_pages] * EXPERTS + [shared_pages]:
        ids = np.arange(start, start + n)
        addresses.append(np.column_stack((ids % PLANES,
                                          ids // (PLANES * BLOCK),
                                          ids // PLANES % BLOCK)).astype(np.int32))
        start += n
    cur = np.bincount(np.concatenate(addresses)[:, 0], minlength=PLANES)
    return addresses, cur


def packed_addresses(order):
    physical, cur = rr_addresses()
    logical = [None] * 65
    for position, expert in enumerate(order):
        logical[int(expert)] = physical[position]
    logical[64] = physical[64]
    return logical, cur


def packed_matching(routes):
    """Exact TRAIN mean optimum for P6 contiguous global page packing.

    Positions(2j,2j+1) share one WL, and different pairs never share a WL.
    Each selected expert otherwise touches2 WL; shared touches3 disjoint WL.
    The total is15 minus the selected-both matching edges. This objective
    reduces exactly to a maximum-weight perfect matching on coactivations.
    """
    import networkx as nx
    incidence = np.zeros((len(routes), 64), dtype=np.int64)
    incidence[np.arange(len(routes))[:, None], routes] = 1
    coactivation = incidence.T @ incidence
    graph = nx.Graph()
    graph.add_nodes_from(range(64))
    for e in range(64):
        for f in range(e + 1, 64):
            graph.add_edge(e, f, weight=int(coactivation[e, f]))
    matching = nx.max_weight_matching(graph, maxcardinality=True, weight="weight")
    pairs = sorted(tuple(sorted(edge)) for edge in matching)
    assert len(pairs) == 32 and sorted(e for pair in pairs for e in pair) == list(range(64))
    order = np.asarray(pairs, dtype=np.int32).reshape(64)
    both = sum(incidence[:, e] * incidence[:, f] for e, f in pairs)
    weight = sum(int(coactivation[e, f]) for e, f in pairs)
    addresses, _ = packed_addresses(order)
    waves = wl_union_waves(addresses, routes)
    assert np.array_equal(waves, 15 - both) and int(both.sum()) == weight
    return order, waves, dict(
        algorithm="NetworkX maximum-weight perfect matching", library_version=nx.__version__,
        objective="maximize sum TRAIN coactivations of paired experts = minimize sum same-WL packed waves",
        exact_within_fixed_global_P6_packing_mean_family=True,
        exact_for_tail_or_native_latency=False, selected_pair_weight=weight,
        TRAIN_wave_sum=int(waves.sum()), pairs=[list(p) for p in pairs])


def sparse_samewl_gap_ledger(addresses, routes):
    """Count idle-plane breaks under a selected-page-only same-WL mask.

    Issue touched WLs in ascending order in this one-block fixture. An omitted
    plane is not presumed CR-hot when it becomes active again. Keep-hot reads
    every actual page in the four-plane wave and conditionally skips inactive
    data before ECC/PE; this is a strong conditional control, not proven ABI.
    """
    present = np.zeros((65, BLOCK, 4), dtype=np.bool_)
    for e, pages in enumerate(addresses):
        for p, b, w in pages:
            assert b == 0
            present[e, w, p] = True
    active = present[routes].any(axis=1) | present[64]
    touched = active.any(axis=2)
    # Compress out WLs for which no plane reads. Address jumps by themselves
    # are allowed by the user's arbitrary-WL same-block skip-CR assumption.
    waves, runs, gaps = [], [], []
    for a, t in zip(active, touched):
        mask = a[t]
        starts = mask & ~np.concatenate((np.zeros((1, 4), dtype=np.bool_), mask[:-1]))
        nrun = starts.sum(axis=0)
        waves.append(len(mask))
        runs.append(nrun)
        gaps.append(nrun - mask.any(axis=0))
        assert int(mask.sum()) == 48
    waves, runs, gaps = map(np.asarray, (waves, runs, gaps))
    physical = waves * 4
    overfetch = physical - 48
    assert np.all(overfetch >= 0)
    return dict(
        ordering="Ascending issued WL; all addresses in one block. Skipped addresses alone assumed CR-compatible, omitted plane epochs not assumed hot.",
        selected_only=dict(physical_pages=48, overfetch_pages=0,
                           plane_cold_runs=[stats(runs[:, p]) for p in range(4)],
                           total_plane_cold_runs=stats(runs.sum(axis=1)),
                           extra_plane_restarts_after_idle=stats(gaps.sum(axis=1)),
                           any_idle_restart_queries=int(np.count_nonzero(gaps.sum(axis=1))),
                           strict_no_interval_requires_cold_restart=True),
        conditional_keep_hot=dict(
            physical_read_pages=stats(physical), inactive_overfetch_pages=stats(overfetch),
            physical_read_bytes_per_chip=stats(physical * PAGE),
            array_read_amplification_mean=float(physical.mean() / 48),
            plane_cold_runs=[1, 1, 1, 1], all_planes_read_every_issued_wave=True,
            selected_ECC_PE_pages=48, inactive_pages_dropped_before_ECC_PE=True,
            capability_unproven="Requires per-plane validity metadata and conditional discard before shared ECC/PE; full array reading retains one page/PB perplane but PB ownership still needs native audit.",
            full_ECC_drain_alternative_pages=stats(physical),
            performance_claim=False)), dict(waves=waves.astype(np.int32),
                                            plane_cold_runs=runs.astype(np.int32),
                                            plane_idle_restarts=gaps.astype(np.int32),
                                            keep_hot_overfetch_pages=overfetch.astype(np.int32))


def wl_union_waves(addresses, routes):
    # This keeps the SAME physical addresses when switching capability.
    slots = 1 + max(int(e[:, 1].max()) for e in addresses if len(e))
    present = np.zeros((len(addresses), slots * BLOCK), dtype=np.bool_)
    for e, pages in enumerate(addresses):
        present[e, pages[:, 1] * BLOCK + pages[:, 2]] = True
    active = present[routes].any(axis=1) | present[-1]
    return active.sum(axis=1).astype(np.int32)


def blockwise_ipr_waves(addresses, routes):
    """Feasible page-count grouping by common block, NOT a CR timeline."""
    blocks = 1 + max(int(e[:, 1].max()) for e in addresses if len(e))
    counts = np.zeros((len(addresses), blocks, PLANES), dtype=np.int32)
    for e, pages in enumerate(addresses):
        for plane, block, _ in pages:
            counts[e, block, plane] += 1
    active = counts[routes].sum(axis=1) + counts[-1]
    return active.max(axis=2).sum(axis=1), (active > 0).sum(axis=1), active


def address_ledger(addresses):
    return [dict(expert=e, shared=(e == EXPERTS),
                 pages=[dict(local_page=i, plane=int(p), block=int(b), wordline=int(w))
                        for i, (p, b, w) in enumerate(a)])
            for e, a in enumerate(addresses)]


def evaluate(profiles, routes, addresses=None, mode="ipr"):
    profiles = np.asarray(profiles, dtype=np.int32)
    if addresses is None:
        addresses, resident = compact_addresses(profiles)
    else:
        resident = profiles.sum(axis=0)
    assert profiles.shape == (65, 4)
    assert np.all(profiles[:64].sum(axis=1) == 6) and profiles[64].sum() == 12
    assert np.all(resident <= BLOCK), "G4 controls must fit one four-plane block"
    selected = profiles[routes].sum(axis=1) + profiles[-1]
    assert np.all(selected.sum(axis=1) == 48)
    if mode == "ipr":
        waves = selected.max(axis=1)
    elif mode == "same_wl":
        waves = wl_union_waves(addresses, routes)
    elif mode == "per_expert":
        waves = profiles[routes].max(axis=2).sum(axis=1) + profiles[-1].max()
    else:
        raise ValueError(mode)
    assert np.all(waves >= 12)
    return waves.astype(np.int32), dict(
        mode=mode, selected_pages_per_chip=48, selected_bytes_per_chip=48 * PAGE,
        selected_bytes_all_chips=48 * PAGE * CHIPS,
        programmed_pages_per_chip=int(resident.sum()), resident_pages_per_plane=resident.tolist(),
        allocated_four_plane_blocks_per_chip=1,
        allocated_lsb_bytes_per_chip=BLOCK * PLANES * PAGE,
        replicas=1, overfetch_pages=0, page_wave_lower_bound=12,
        stats=stats(waves), addresses_sha256=digest(np.concatenate(addresses)),
        legal_native_ROW_PB_CR_timeline_proven=False)


def phase_fit(routes, start, sweeps=3):
    """Small deterministic coordinate descent; ordinary load balancing control.

    No algorithmic novelty is claimed. TRAIN score is (sum maximum-plane load,
    sum squared load, lexicographic phase); only strict score improvements are
    accepted. Four static initial maps are compared on TRAIN before this fit.
    """
    phase = np.asarray(start, dtype=np.int32).copy()
    table = np.asarray([page_profile(6, p) for p in range(4)])
    load = table[phase][routes].sum(axis=1) + 3
    members = [np.flatnonzero((routes == e).any(axis=1)) for e in range(EXPERTS)]
    accepted = attempted = 0
    for _ in range(sweeps):
        changed = 0
        for e, ii in enumerate(members):
            if not len(ii):
                continue
            old = phase[e]
            block = load[ii]
            before = (int(block.max(axis=1).sum()), int((block ** 2).sum()))
            best, bestload, bestp = before, block, int(old)
            for new in range(4):
                if new == old:
                    continue
                trial = block + table[new] - table[old]
                score = (int(trial.max(axis=1).sum()), int((trial ** 2).sum()))
                attempted += 1
                if score < best:
                    best, bestload, bestp = score, trial, new
            if bestp != old:
                phase[e] = bestp
                load[ii] = bestload
                changed += 1
                accepted += 1
        if not changed:
            break
    return phase, dict(attempted=attempted, accepted=accepted,
                       objective="TRAIN lexicographic(sum max plane pages, sum squared plane load)",
                       maximum_sweeps=sweeps, not_a_new_algorithm=True)


def source_provenance(out):
    compiler = read(TRAIN / "compiler_inputs.frozen.json")
    protocol = read(TRAIN / "protocol.frozen.json")
    result = dict(compiler_inputs=dict(path=str(TRAIN / "compiler_inputs.frozen.json"),
                                      sha256=sha(TRAIN / "compiler_inputs.frozen.json")),
                  capture_protocol=dict(path=str(TRAIN / "protocol.frozen.json"),
                                        sha256=sha(TRAIN / "protocol.frozen.json")), models={})
    arrays = {}
    for name, shape in [("granite4tiny", (1536, 512, 64, 6, 40, 1024)),
                        ("deepseek", (2048, 1408, 64, 6, 26, 2816))]:
        frozen = compiler["models"][name]
        p = Path(frozen["archive"])
        assert sha(p) == frozen["sha256"]
        with np.load(p, allow_pickle=False) as z:
            a = {k: z[k].copy() for k in z.files}
        assert a["split"].item() == "train" and a["completed"].all()
        assert list(a["routes"].shape) == frozen["shape"]
        assert np.all((0 <= a["routes"]) & (a["routes"] < 64))
        assert np.all(np.diff(np.sort(a["routes"], axis=-1), axis=-1) > 0)
        meta_path = Path("/home/nfs_client/ericyu/expert_flash_experiment/models") / (name + "-metadata.json")
        meta = read(meta_path)
        # Receipt verification uses the pre-existing capture hashes; do not
        # repeatedly stream several GB of immutable model data to rediscover it.
        assert protocol["inputs_sha256"][str(meta_path)] == sha(meta_path)
        assert protocol["inputs_sha256"][meta["model_path"]] == meta["sha256"]
        model = dict(H=shape[0], I=shape[1], E=shape[2], K=shape[3],
                     routed_layers=shape[4], shared_I=shape[5], chips=16,
                     pages_per_routed_GU_chip=2 * shape[0] * shape[1] // (16 * PAGE),
                     pages_per_shared_GU_chip=2 * shape[0] * shape[5] // (16 * PAGE))
        result["models"][name] = dict(
            config=model, archive=str(p), archive_sha256=sha(p),
            array_shape=list(a["routes"].shape), ordered_routes_sha256=digest(a["routes"]),
            prompt_ids=a["prompt_ids"].tolist(), prompt_ids_sha256=digest(a["prompt_ids"]),
            layers=a["layers"].tolist(), completed_prompts=int(a["completed"].sum()),
            representation=a["representation"].item(), split="TRAIN", previously_observed=True,
            source_model=meta["model_path"], source_model_sha256=meta["sha256"],
            model_hash_verification="Capture receipt checked against metadata; model not rehashed in this page study.",
            metadata_path=str(meta_path), metadata_sha256=sha(meta_path),
            numeric_scope="canonical fixed full-FFN W8A8 routes; original GGUF attention/router; not original GGUF routing")
        arrays[name] = a
    result["data_policy"] = "Only existing W8 TRAIN opened. No new heldout, existing prospective, or heldout routes opened."
    result["old_strong_plan_api"] = old_plan_index()
    dump(out / "provenance_and_old_plan_api.json", result)
    return result, arrays


def old_plan_index():
    summary_path = ROUND2 / "shared-counterbalanced/independent-v2/summary.json"
    summary = read(summary_path)
    strong = []
    for c in summary["cases"]:
        assert sha(c["path"]) == c["sha256"]
        case = read(c["path"])
        for family, rec in c["families"].items():
            a = Path(rec["arrays"])
            with np.load(a, allow_pickle=False) as z:
                assert z["rows"].shape == (65, 2, 16)
                assert z["routes"].shape == (512, 6)
            strong.append(dict(model=case["model"], moe_layer=case["layer"],
                               input_channel_copies=case["copies"], family=family,
                               arrays=str(a), sha256=sha(a), result=c["path"],
                               selected_TRAIN_native_mean_ns=rec["mean_ns"]))
    cp_path = ROUND2 / "compact-coupled-solver/summary.json"
    cp = read(cp_path)
    best = cp["candidates"][cp["best_native_index"]]
    full_path = ROUND2 / "ordinary-read-compiler/full-campaign-v5-W8-training/summary.json"
    full = read(full_path)
    all_layers = []
    for c in full["records"]:
        for a in c["artifacts"]:
            all_layers.append(dict(model=c["model"], moe_layer=c["layer"],
                                   raw_layer=c["training"]["physical_model_layer"],
                                   command_ns=c["command_ns"], family=a["family"],
                                   arrays=str(ROOT / a["arrays"]), sha256=a["arrays_sha256"],
                                   plan=str(ROOT / a["plan"]), plan_sha256=a["plan_sha256"]))
    return dict(
        API="np.load(arrays): rows[expert=0..64, phase=0 GU/1 Down, chip=0..15]; expert64 shared; routes[512,6]. input_columns[e,p], total_rows[e,p], jobs, logical and addresses are emitted views. Existing backend.lower_rows accepts rows[:,0],rows[:,1], GU columns H, Down columns vector([I]*64+[shared_I]), Service.",
        caution="These are ordinary-read plans/costs. Do not reuse old cost/lower_bound as CR or IPR timing. Latest strong independent joint plans cover first3 MoE layers, not40 independent fits.",
        latest_three_layer_summary=dict(path=str(summary_path), sha256=sha(summary_path)),
        latest_three_layer_arrays=strong,
        G4_L0_strong_CP_incumbent=dict(path=best["arrays"], sha256=best["arrays_sha256"],
                                      native_mean_ns=best["native_mean_ns"],
                                      constraints="one-copy complete rows, frozen shared, ordinary read cmd250/input1; bounded CP control, no global optimality"),
        all_layer_campaign=dict(path=str(full_path), sha256=sha(full_path),
                                source_TRAIN_queries=512, selection_queries=64,
                                limits="Complete all-MoE-layer ordinary-read frozen plans; older and weaker than later joint controls on first3 layers.",
                                arrays=all_layers))


def self_checks():
    # Enumerate all routes for a small fixture and compare incidence-union to
    # a direct address set; this exercises same-WL vs independent-plane counts.
    import itertools
    profiles = np.asarray([page_profile(6, e % 4) for e in range(8)] + [[3, 3, 3, 3]])
    aa, resident = compact_addresses(profiles)
    rr = np.asarray(list(itertools.combinations(range(8), 3)), dtype=np.int32)
    got = wl_union_waves(aa, rr)
    for route, n in zip(rr, got):
        direct = {tuple(a[1:]) for e in [*route, 8] for a in aa[e]}
        assert len(direct) == n
        counts = Counter(int(a[0]) for e in [*route, 8] for a in aa[e])
        assert max(counts.values()) == (profiles[route].sum(axis=0) + profiles[-1]).max()
    return dict(small_direct_address_union_cases=len(rr), pass_all=True)


def run(out):
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / "report.json").exists(), "Preserve results; choose a new output directory."
    provenance, archives = source_provenance(out)
    raw = archives["granite4tiny"]["routes"]
    routes = raw.transpose(2, 0, 1, 3).reshape(40, 512, 6)
    flat = routes.reshape(-1, 6)
    rr_addr, _ = rr_addresses()
    rr_prof = np.asarray([np.bincount(a[:, 0], minlength=4) for a in rr_addr])
    aligned = np.asarray([[2, 2, 1, 1]] * 64 + [[3, 3, 3, 3]], dtype=np.int32)
    candidate = np.asarray([[2, 2, 1, 1]] * 64 + [[0, 0, 6, 6]], dtype=np.int32)
    uniform = np.asarray([page_profile(6, e % 4) for e in range(64)] + [[3, 3, 3, 3]])
    fixed_pair = np.asarray([[3, 3, 0, 0] if e % 2 == 0 else [0, 0, 3, 3]
                             for e in range(64)] + [[3, 3, 3, 3]])
    fixed_single = np.asarray([[6 if p == e % 4 else 0 for p in range(4)]
                               for e in range(64)] + [[3, 3, 3, 3]])
    variants = [
        ("per_expert_runs", rr_prof, rr_addr, "per_expert"),
        ("sameWL_packed_skipCR_global_roundrobin", rr_prof, rr_addr, "same_wl"),
        ("IPR_global_roundrobin", rr_prof, rr_addr, "ipr"),
        ("IPR_static_uniform_four_phase", uniform, None, "ipr"),
        ("IPR_per_expert_phase_aligned_balanced_shared", aligned, None, "ipr"),
        ("IPR_fixed_two_plane_masks", fixed_pair, None, "ipr"),
        ("IPR_fixed_one_plane_masks", fixed_single, None, "ipr"),
        ("sameWL_joint_compensation_compact_fixed_layout", candidate, None, "same_wl"),
        ("IPR_joint_onecopy_shared_compensation", candidate, None, "ipr"),
        ("IPR_generic_sparse_row_compiler_same_layout", candidate, None, "ipr"),
    ]
    report = dict(status="complete_TRAIN_page_wave_headroom_only",
                  source_sha256=sha(__file__), created_utc=datetime.now(timezone.utc).isoformat(),
                  scope="GU only. Common layer input; skip arbitrary WL within same block and IPR are explicit capability assumptions. No native delay/FFN speedup/hardware perf claim.",
                  config=provenance["models"]["granite4tiny"]["config"],
                  TRAIN_shape=list(raw.shape), evaluations=512 * 40,
                  controls={}, limits=[])
    values = {}
    ledgers = {}
    for name, profiles, addresses, mode in variants:
        waves, rec = evaluate(profiles, flat, addresses, mode)
        report["controls"][name] = rec
        rec["per_layer_mean_waves"] = waves.reshape(40, 512).mean(axis=1).tolist()
        rec["per_token_sum_40_layers"] = stats(waves.reshape(40, 512).sum(axis=0))
        values[name] = waves.reshape(40, 512)
        if addresses is None:
            addresses, _ = compact_addresses(profiles)
        ledgers[name] = dict(profiles=profiles.tolist(), addresses=address_ledger(addresses))

    # Strong packed same-WL control: exact TRAIN mean matching,40independent
    # layer maps. Pair order/orientation leave same-WL mean objective unchanged.
    packed_orders, packed_values, packed_ipr_values, matching_records = [], [], [], []
    packed_ledgers, packed_gap_records, packed_gap_arrays = [], [], []
    for layer, layer_routes in enumerate(routes):
        order, waves, fitting = packed_matching(layer_routes)
        addresses, _ = packed_addresses(order)
        profiles = np.asarray([np.bincount(a[:, 0], minlength=4) for a in addresses])
        ipr, _ = evaluate(profiles, layer_routes, addresses, mode="ipr")
        gap, gap_arrays = sparse_samewl_gap_ledger(addresses, layer_routes)
        packed_orders.append(order)
        packed_values.append(waves)
        packed_ipr_values.append(ipr)
        matching_records.append(dict(layer=layer, **fitting))
        packed_gap_records.append(dict(layer=layer, **gap))
        packed_gap_arrays.append(gap_arrays)
        packed_ledgers.append(dict(layer=layer, profiles=profiles.tolist(), addresses=address_ledger(addresses)))
    packed_orders = np.asarray(packed_orders, dtype=np.int32)
    for name, data, mode in [
        ("sameWL_packed_TRAIN_exact_mean_matching", np.asarray(packed_values), "same_wl"),
        ("IPR_same_TRAIN_exact_mean_matching_layout", np.asarray(packed_ipr_values), "ipr")]:
        values[name] = data
        report["controls"][name] = dict(
            mode=mode, stats=stats(data), per_layer_mean_waves=data.mean(axis=1).tolist(),
            per_token_sum_40_layers=stats(data.sum(axis=0)), selected_pages_per_chip=48,
            selected_bytes_per_chip=48 * PAGE, programmed_pages_per_chip=396,
            resident_pages_per_plane=[99] * 4, allocated_four_plane_blocks_per_chip=1,
            replicas=1, overfetch_pages=0, page_wave_lower_bound=12,
            selected_on_TRAIN=True, exact_mean_optimum_scope="fixed6pages/expert contiguous global packing; no native/tail optimality",
            legal_native_ROW_PB_CR_timeline_proven=False)
    report["controls"]["sameWL_packed_TRAIN_exact_mean_matching"]["fitting"] = matching_records
    ledgers["TRAIN_exact_matching_per_layer"] = packed_ledgers
    np.savez_compressed(out / "frozen_TRAIN_packed_expert_order.npz",
                        slot_to_expert=packed_orders, layers=np.arange(40))
    gap_identity, identity_gap_arrays = sparse_samewl_gap_ledger(rr_addr, flat)
    dump(out / "sparse_mask_and_keep_hot_ledger.json", dict(
        identity_global_roundrobin=gap_identity, TRAIN_matching_per_layer=packed_gap_records,
        IPR_joint_candidate=dict(issued_waves=12, selected_pages=48, physical_array_read_pages=48,
                                overfetch_pages=0, plane_cold_runs=[1] * 4,
                                idle_plane_restarts=0,
                                condition="Each of12abstract IPR epochs reads1selectedpage on everyplane; native row/PB/ECC/CR validity separate.")))
    gap_saved = {"identity_" + k: v.reshape((40, 512) + v.shape[1:]) for k, v in identity_gap_arrays.items()}
    gap_saved.update({"TRAIN_matching_" + k: np.asarray([x[k] for x in packed_gap_arrays])
                      for k in packed_gap_arrays[0]})
    np.savez_compressed(out / "sparse_mask_and_keep_hot_arrays.npz", **gap_saved)

    # The exact same profiles+addresses compare same-WL against IPR; fixed
    # masks are separate control families, not a substitute for IPR baseline.
    report["capability_isolation"] = [
        dict(same_layout="standard global roundrobin", sameWL="sameWL_packed_skipCR_global_roundrobin",
             IPR="IPR_global_roundrobin", physical_addresses_identical=True),
        dict(same_layout="joint shared compensation", sameWL="sameWL_joint_compensation_compact_fixed_layout",
             IPR="IPR_joint_onecopy_shared_compensation", physical_addresses_identical=True),
        dict(same_layout="TRAIN exact mean maximum matching", sameWL="sameWL_packed_TRAIN_exact_mean_matching",
             IPR="IPR_same_TRAIN_exact_mean_matching_layout", physical_addresses_identical=True),
    ]
    static_phases = [np.zeros(64, dtype=np.int32), np.arange(64, dtype=np.int32) % 4,
                     np.arange(64, dtype=np.int32) * 6 % 4,
                     (np.arange(64, dtype=np.int32) // 16) % 4]
    learned, fit_records, learned_values = [], [], []
    for layer, layer_routes in enumerate(routes):
        scores = []
        for phase in static_phases:
            pp = np.asarray([page_profile(6, int(p)) for p in phase])
            load = pp[layer_routes].sum(axis=1) + 3
            scores.append((int(load.max(axis=1).sum()), int((load ** 2).sum())))
        chosen = min(range(len(scores)), key=lambda i: scores[i])
        phase, fitting = phase_fit(layer_routes, static_phases[chosen])
        profiles = np.asarray([page_profile(6, int(p)) for p in phase] + [[3, 3, 3, 3]])
        waves, _ = evaluate(profiles, layer_routes)
        assert int(waves.sum()) <= scores[chosen][0]
        learned.append(phase)
        learned_values.append(waves)
        fit_records.append(dict(layer=layer, initial_static_index=chosen,
                                initial_static_score=list(scores[chosen]), **fitting))
    learned = np.asarray(learned, dtype=np.int32)
    learned_values = np.asarray(learned_values, dtype=np.int32)
    np.savez_compressed(out / "frozen_TRAIN_phase_maps.npz", learned=learned,
                        static_phase_library=np.asarray(static_phases), layers=np.arange(40),
                        TRAIN_archive_sha256=np.asarray(provenance["models"]["granite4tiny"]["archive_sha256"]),
                        split=np.asarray("train"))
    values["IPR_TRAIN_learned_phase_coordinate_descent"] = learned_values
    report["controls"]["IPR_TRAIN_learned_phase_coordinate_descent"] = dict(
        mode="ipr", replicas=1, overfetch_pages=0, selected_pages_per_chip=48,
        programmed_pages_per_chip=396, allocated_four_plane_blocks_per_chip=1,
        stats=stats(learned_values), per_layer_mean_waves=learned_values.mean(axis=1).tolist(),
        per_token_sum_40_layers=stats(learned_values.sum(axis=0)),
        selected_on_TRAIN=True, heuristic="three deterministic coordinate-descent sweeps, strongest of four static starts",
        phase_maps=str(out / "frozen_TRAIN_phase_maps.npz"), fitting=fit_records,
        legal_native_ROW_PB_CR_timeline_proven=False)
    # Record every learned map's independent capacity evidence.
    report["controls"]["IPR_TRAIN_learned_phase_coordinate_descent"]["resident_pages_per_layer"] = [
        (np.asarray([page_profile(6, int(p)) for p in phase]).sum(axis=0) + 3).tolist()
        for phase in learned]
    dump(out / "physical_page_ledgers.json", ledgers)
    np.savez_compressed(out / "TRAIN_page_wave_arrays.npz", **values)
    frozen = dict(frozen_utc=datetime.now(timezone.utc).isoformat(), source_sha256=sha(__file__),
                  TRAIN_archive=provenance["models"]["granite4tiny"]["archive"],
                  TRAIN_archive_sha256=provenance["models"]["granite4tiny"]["archive_sha256"],
                  TRAIN_shape=list(raw.shape), learned_phase_sha256=digest(learned),
                  frozen_phase_file_sha256=sha(out / "frozen_TRAIN_phase_maps.npz"),
                  static_layouts_file_sha256=sha(out / "physical_page_ledgers.json"),
                  frozen_packed_matching_sha256=sha(out / "frozen_TRAIN_packed_expert_order.npz"),
                  sparse_mask_keep_hot_ledger_sha256=sha(out / "sparse_mask_and_keep_hot_ledger.json"),
                  heldout_opened=False, heldout_evaluation="Wait for root frozen protocol and designated new archive; no fit after this freeze.",
                  capability_assumptions="Common-input GU; arbitrary same-block WL skipping; independent plane page reads. Native row/PB/ECC/CR correctness delegated separately.")
    dump(out / "protocol.frozen.json", frozen)

    # Deep compensation is page-balanced globally but not one-block resident.
    dprof = np.asarray([[6, 6, 5, 5]] * 64 + [[8, 8, 14, 14]], dtype=np.int32)
    da, dr = compact_addresses(dprof)
    deep_routes = archives["deepseek"]["routes"].reshape(-1, 6)
    dw, touched, active = blockwise_ipr_waves(da, deep_routes)
    global_load = dprof[deep_routes].sum(axis=1) + dprof[-1]
    assert np.all(global_load == 44) and dr.tolist() == [392, 392, 334, 334]
    report["Deep_crossblock_audit"] = dict(
        routed_GU_pages_per_chip=22, shared_GU_pages_per_chip=44,
        routed_profile=[6, 6, 5, 5], shared_profile=[8, 8, 14, 14],
        selected_pages_per_chip=176, selected_page_bytes_per_chip=176 * PAGE,
        global_plane_page_lower_bound=44, global_selected_plane_load=[44] * 4,
        resident_pages_per_plane=dr.tolist(), fits_one_block=False,
        minimum_required_blocks_per_plane=((dr + BLOCK - 1) // BLOCK).tolist(),
        allocated_four_plane_blocks_per_chip=3,
        allocated_lsb_bytes_per_chip=3 * PLANES * BLOCK * PAGE,
        global_roundrobin_comparison=dict(resident_pages_per_plane=[363] * 4,
                                         allocated_blocks_per_chip=2,
                                         note="Equal programmed bytes but compensation uses an extra reserved four-plane block; common capacity comparison must allow/pay three blocks for all controls."),
        common_block_grouping_page_waves=stats(dw),
        touched_blocks_per_plane_stats=[stats(touched[:, p]) for p in range(4)],
        selected_block_plane_count_shape=list(active.shape),
        claims="44 is aggregate page-work lower bound only. Static compact placement crosses blocks. Do not call this 44 consecutive same-block CR reads or native latency.")
    report["no_headroom_cases"] = [
        dict(condition="No shared branch", conclusion="The shared-compensation mechanism is unavailable. P=6 still leaves phase/IPR opportunities, but all-top6 9-wave guarantee is impossible for static integer page profiles over64experts: some plane is heavy in at least16experts, so choose6 heavy experts.",
             aligned_P6_worst_case_waves=12, total_page_lower_bound=9),
        dict(condition="P divisible by4 and every expert balanced over4planes", conclusion="Per-expert and aggregate IPR sense-wave counts both equal K*P/4. With balanced shared pages this remains exact: zero page-wave headroom for this mechanism. Cold starts/commands can still differ and require native model.",
             example=dict(P=8, K=6, shared_pages=0, per_expert_waves=12, IPR_waves=12)),
    ]
    report["generic_control"] = dict(
        same_layout_exact_match=True, IPR_waves=12, gain_vs_generic=0,
        explanation="A generic sparse page/row compiler granted exactly the candidate's page-order/partial-row/selected-page execution capability emits the identical static one-copy ledger. Joint shared compensation is elementary water filling; no algorithm/layout novelty follows. If native complete-row constraints forbid it, they forbid both equally.",
        count_proof="K*(2,2,1,1)+(0,0,6,6)=(12,12,12,12);48selectedpages/4=12. Resident64*(2,2,1,1)+(0,0,6,6)=(128,128,70,70)<=192.")
    report["sparse_mask_CR_gap_audit"] = dict(
        ledger=str(out / "sparse_mask_and_keep_hot_ledger.json"),
        arrays=str(out / "sparse_mask_and_keep_hot_arrays.npz"),
        identity=gap_identity,
        interpretation="sameWL page unions are not uninterrupted per-plane CR runs. Every inactive-to-active transition after the first is a potential cold restart under strict no-interval CR. Conditional full-wave inactive reads can keep everyplane reading, at explicit physical overfetch; drop-before-ECC is an additional execution capability, not established.")
    report["verification"] = self_checks()
    report["limits"] = [
        "TRAIN descriptive counters only; no heldout significance or generalization claim.",
        "G4 any-top6 exact page-wave equality is an algebraic certificate, not native completion latency.",
        "All G4 controls have396programmedpages/chip, one copy,48selectedpages/query/chip, one reserved four-plane block/chip. No additional replicas/inactive overfetch.",
        "Page ledgers preserve unique logical pages but do not establish complete-row instruction legality, PB ownership, row-fringe outputs, accumulator switches, ECC stalls, sustained CR, or vector/control/output traffic.",
        "Down uses different expert inputs and is outside the common-input GU page-wave model.",
        "Fixed one/two-plane bins are separate static controls. They do not establish the general IPR capability or allow route-dependent page remapping.",
        "Generic matching control eliminates layout novelty under identical capabilities; research attribution must come from validated native lowering/execution if a useful advantage survives.",
    ]
    dump(out / "report.json", report)
    lines = ["# Common-input GU page-wave baselines", "",
             "TRAIN only; all 40 G4 layers and 512 aligned decode inputs. These are sense-wave/page counters, not native completion latency.", "",
             "| Control | Mean waves/layer | P99 | Range |", "|---|---:|---:|---:|"]
    for name, rec in report["controls"].items():
        s = rec["stats"]
        lines.append(f"| {name} | {s['mean']:.6f} | {s['p99']:.2f} | {s['minimum']}–{s['maximum']} |")
    lines += ["", "All controls: one copy;396 resident pages/chip;48 selected pages/chip/query; no overfetch; one reserved four-plane192-WL block/chip.", "",
              "The shared-compensated profile gives6×(2,2,1,1)+(0,0,6,6)=(12,12,12,12), meeting48/4=12 for every possible top6. Its resident counts(128,128,70,70) fit192WL. A generic sparse compiler granted identical primitives emits exactly this layout and ties it; this is elementary water filling, not a new layout algorithm.", "",
              "The same-layout comparisons isolate same-WL mask from IPR; fixed one/two-plane masks are additional controls. The learned phase baseline is a small TRAIN-only coordinate descent, frozen before opening new heldout.", "",
              "The strong packed same-WL control exactly minimizes TRAIN mean within contiguous6page global packing: waves=15−numberofselected-both adjacentpairs, so maximum-weight perfect matching on expert coactivations solves it. This is a standard matching baseline, not a new algorithm or native/tail optimum.", "",
              f"Sparse identity masks have{gap_identity['selected_only']['extra_plane_restarts_after_idle']['mean']:.6f}mean extra plane cold restarts if an idle read epoch breaks CR. Conditional keep-hot reads everyplane, with{gap_identity['conditional_keep_hot']['inactive_overfetch_pages']['mean']:.6f}mean inactive pages/chip/query; these may be dropped beforeECC only under an explicit conditional-discard capability. Full gap/overfetch counters are in`sparse_mask_and_keep_hot_ledger.json`.", "",
              f"Deep requires resident(392,392,334,334), hence3/3/2/2 plane blocks and3 reserved four-plane blocks.44 is only a global page-work bound; the explicit common-block grouping averages{stats(dw)['mean']:.6f}waves. It cannot be reported as44consecutive same-block CR reads.", "",
              "Without a shared branch there is no shared compensation. When P is divisible by4 and each expert is balanced, per-expert runs already match aggregate IPR page-wave work. This G4 case is not a universal gain.", "",
              "No native ROW/PB/CR timing, numerical lowering, down execution, hardware speedup, power or end-to-end FFN claim is made here.", "",
              "Reproduce: `python3 scripts/research_20261102_cr_ipr_baselines.py --output-dir /tmp/cr-ipr-baselines-reproduction`.", "",
              "Provenance and old strong plan API: `provenance_and_old_plan_api.json`. Fixed physical page ledgers: `physical_page_ledgers.json`. Frozen phase maps: `frozen_TRAIN_phase_maps.npz`. Frozen rules: `protocol.frozen.json`."]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({k: v["stats"] for k, v in report["controls"].items()}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    run(args.output_dir.resolve())
